"""Vocabulary trimming at the run level (design-02 §3).

mmBERT-small keeps ~98M of its ~140M parameters in a 256k × 384 embedding matrix. A country model
needs a small part of that vocabulary: we keep every token produced by tokenising the run's
training texts, the non-test real strings and the context prefixes, plus the BPE merge closure of
those tokens (so their tokenisation is unchanged), every added/special token, the 256 byte-fallback
tokens and every single-character token (so unseen text still tokenises without <unk>). Test
strings are never used, so evaluation sees the trimming's real effect.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer

EMBEDDING_SUFFIXES = ("tok_embeddings.weight", "embed_tokens.weight", "word_embeddings.weight")
CONFIG_IDS = ("pad_token_id", "bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id")


def _pair(m) -> tuple[str, str]:
    return (m[0], m[1]) if isinstance(m, list) else tuple(m.split(" ", 1))  # type: ignore[return-value]


def _is_byte(t: str) -> bool:
    return len(t) == 6 and t.startswith("<0x") and t.endswith(">")


def kept_ids(
    tok_json: dict, texts: Iterable[str], batch: int = 8192, single_chars: bool = True
) -> set[int]:
    tok = Tokenizer.from_str(json.dumps(tok_json))
    vocab: dict[str, int] = tok_json["model"]["vocab"]
    keep: set[int] = set()
    buf: list[str] = []
    for t in texts:
        buf.append(t)
        if len(buf) >= batch:
            keep.update(i for e in tok.encode_batch(buf, add_special_tokens=False) for i in e.ids)
            buf = []
    if buf:
        keep.update(i for e in tok.encode_batch(buf, add_special_tokens=False) for i in e.ids)
    keep.update(a["id"] for a in tok_json.get("added_tokens", []))
    keep.update(i for t, i in vocab.items() if _is_byte(t) or t == "▁")
    if single_chars:
        keep.update(i for t, i in vocab.items() if len(t.replace("▁", "")) <= 1)
    # Closure over *every* merge that builds a kept token: the tokenizer's real path to a token
    # may use any of them, and dropping one of its intermediates lets other kept merges take
    # the characters first (the trimmed tokenizer then stops short of the token).
    producers: dict[str, list[tuple[str, str]]] = {}
    for m in tok_json["model"]["merges"]:
        a, b = _pair(m)
        producers.setdefault(a + b, []).append((a, b))
    inv = {i: t for t, i in vocab.items()}
    stack = [inv[i] for i in keep if i in inv]
    while stack:
        for pair in producers.get(stack.pop(), ()):
            for part in pair:
                j = vocab.get(part)
                if j is not None and j not in keep:
                    keep.add(j)
                    stack.append(part)
    return keep


def _remap_special_ids(node, old_to_new: dict[int, int]) -> None:
    """Remap `"ids"` lists inside post-processor special-token entries, in place."""
    if isinstance(node, dict):
        if isinstance(node.get("ids"), list) and "tokens" in node:
            node["ids"] = [old_to_new[i] for i in node["ids"]]
        for v in node.values():
            _remap_special_ids(v, old_to_new)
    elif isinstance(node, list):
        for v in node:
            _remap_special_ids(v, old_to_new)


def trim_tokenizer(tok_json: dict, keep: set[int]) -> tuple[dict, list[int]]:
    """New tokenizer json with ids 0..len(keep)-1 (old order) and the new → old id list."""
    new_to_old = sorted(keep)
    old_to_new = {o: n for n, o in enumerate(new_to_old)}
    tj = copy.deepcopy(tok_json)
    inv = {i: t for t, i in tok_json["model"]["vocab"].items()}
    vocab = {inv[o]: n for n, o in enumerate(new_to_old) if o in inv}
    tj["model"]["vocab"] = vocab
    tj["model"]["merges"] = [
        m
        for m in tok_json["model"]["merges"]
        for a, b in [_pair(m)]
        if a in vocab and b in vocab and a + b in vocab
    ]
    for a in tj.get("added_tokens", []):
        a["id"] = old_to_new[a["id"]]
    _remap_special_ids(tj.get("post_processor"), old_to_new)
    if isinstance(tj.get("padding"), dict) and "pad_id" in tj["padding"]:
        tj["padding"]["pad_id"] = old_to_new[tj["padding"]["pad_id"]]
    return tj, new_to_old


def trim_state_dict(sd: dict, new_to_old: Sequence[int]) -> dict:
    """Copy of the state dict with the token-embedding rows reduced to `new_to_old`."""
    out = dict(sd)
    keys = [k for k in sd if k.endswith(EMBEDDING_SUFFIXES)]
    if len(keys) != 1:
        raise ValueError(f"expected one token-embedding matrix, found {keys}")
    out[keys[0]] = sd[keys[0]][list(new_to_old)]
    return out


def trim_run(run: Path, out: Path, texts: Iterable[str], single_chars: bool = True) -> dict:
    """Write a trimmed copy of a training run (weights, tokenizer, run.json with config
    overrides for the new vocabulary size and special-token ids)."""
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig

    tj = json.loads((run / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    keep = kept_ids(tj, texts, single_chars=single_chars)
    new, new_to_old = trim_tokenizer(tj, keep)
    old_to_new = {o: n for n, o in enumerate(new_to_old)}
    sd = trim_state_dict(load_file(str(run / "model.safetensors")), new_to_old)
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, out / "model.safetensors")
    (out / "tokenizer" / "tokenizer.json").write_text(json.dumps(new, ensure_ascii=False))
    tc = json.loads((run / "tokenizer" / "tokenizer_config.json").read_text(encoding="utf-8"))
    if isinstance(tc.get("added_tokens_decoder"), dict):
        tc["added_tokens_decoder"] = {
            str(old_to_new[int(k)]): v for k, v in tc["added_tokens_decoder"].items()
        }
    (out / "tokenizer" / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    cfg = AutoConfig.from_pretrained(meta["config"]["base"])
    overrides: dict[str, int] = {"vocab_size": len(new_to_old)}
    for k in CONFIG_IDS:
        if (v := getattr(cfg, k, None)) is not None:
            overrides[k] = old_to_new[int(v)]
    meta["config_overrides"] = overrides
    meta["trimmed_from"] = str(run)
    (out / "run.json").write_text(json.dumps(meta, indent=1, default=float))
    return {"vocab_old": len(tj["model"]["vocab"]), "vocab_new": len(new_to_old)}
