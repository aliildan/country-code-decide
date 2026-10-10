"""Non-LLM baselines (plan-02 Task 6, design-01 §7). Every baseline maps one string to a
`Prediction`; only `lexicon_context` reads the field context. Copyleft libraries (pycountry: LGPL,
country_converter: GPL) are imported lazily from the `eval` dependency group and never ship.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from countrycode import labels
from countrycode.evaluation.harness import Prediction
from countrycode.lexicon.build import Lexicon
from countrycode.real.gold import GoldItem, fuzzy_candidates
from countrycode.text import compact_key, match_key

ABSTAIN = Prediction(None, (), 0.0)


def _from_set(targets: frozenset[str] | set[str], conf: float) -> Prediction:
    if not targets:
        return ABSTAIN
    ts = tuple(sorted(targets))
    return Prediction(ts[0], ts, conf) if len(ts) == 1 else Prediction(None, ts, 0.0)


def _code(code: str | None) -> Prediction:
    if not code:
        return ABSTAIN
    code = labels.RESERVED_TEXT_CODES.get(code.upper(), code.upper())
    return Prediction(code, (code,), 1.0) if code in labels.countries() else ABSTAIN


class LexiconExact:
    name = "lexicon_exact"

    def __init__(self, lexicon: Lexicon) -> None:
        self.lexicon = lexicon

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        return [_from_set(self.lexicon.lookup(g.text), 1.0) for g in items]


class LexiconContext:
    """Exact lexicon lookup plus the context rules that label the training strings (v1.2.1, external
    review): a Canadian province code in a US jurisdiction field is Canada; a set that mixes
    countries and US states is the state in a US jurisdiction field and the country in a country or
    nationality field; a set that contains the register's own place is that place. Anything still
    ambiguous abstains with the set."""

    name = "lexicon_context"

    def __init__(self, lexicon: Lexicon) -> None:
        self.lexicon = lexicon

    def _one(self, g: GoldItem) -> Prediction:
        from countrycode.train.dataset import _collision, _context_override

        field, reg = g.field_kind, (g.register_country or "unknown").upper()
        if override := _context_override(g.text, field, reg):
            return _from_set(set(override), 1.0)
        targets = self.lexicon.lookup(g.text)
        if len(targets) > 1 and (col := _collision(sorted(targets))):
            countries, states = col
            us = reg == "US" or reg.startswith("US-")
            if field == "jurisdiction" and us:
                return _from_set(set(states), 1.0)
            if field in ("country", "nationality"):
                return _from_set(set(countries), 1.0)
        if len(targets) > 1 and reg in targets:
            return _from_set({reg}, 1.0)
        return _from_set(targets, 1.0)

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        return [self._one(g) for g in items]


class LexiconFuzzy:
    """Exact lexicon hit, else the best fuzzy key (Levenshtein ratio on compact keys, partial
    ratio for long keys) when its score is >= threshold; confidence = score / 100."""

    name = "lexicon_fuzzy"

    def __init__(self, lexicon: Lexicon, threshold: float = 80.0) -> None:
        self.lexicon = lexicon
        self.threshold = threshold

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        out = []
        for g in items:
            exact = self.lexicon.lookup(g.text)
            if exact:
                out.append(_from_set(exact, 1.0))
                continue
            cands = fuzzy_candidates(g.text, self.lexicon, limit=1)
            if cands and cands[0][1] >= self.threshold:
                key, score = cands[0]
                out.append(_from_set(self.lexicon.lookup(key), score / 100.0))
            else:
                out.append(ABSTAIN)
        return out


def _collapse(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", compact_key(s))


class OnsStyle:
    """ONS Census 2021 style: collapse doubled letters, exact index match, else Levenshtein
    similarity >= 0.8 (here over the multilingual lexicon instead of an English index)."""

    name = "ons_style"

    def __init__(self, lexicon: Lexicon, threshold: float = 0.8) -> None:
        self.lexicon = lexicon
        self.threshold = threshold
        index: dict[str, set[str]] = {}
        for key in lexicon.index_keys():
            index.setdefault(_collapse(key), set()).update(lexicon.lookup(key))
        self.index = index
        self.keys = list(index)

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        out = []
        for g in items:
            q = _collapse(g.text)
            if not q:
                out.append(ABSTAIN)
            elif q in self.index:
                out.append(_from_set(self.index[q], 1.0))
            else:
                best = process.extractOne(q, self.keys, scorer=fuzz.ratio)
                if best and best[1] / 100 >= self.threshold:
                    out.append(_from_set(self.index[best[0]], best[1] / 100))
                else:
                    out.append(ABSTAIN)
        return out


class CountryNames:
    """OpenSanctions `countrynames` (MIT) with fuzzy matching — the strongest open tool."""

    name = "countrynames"

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        import countrynames

        out = []
        for g in items:
            if not match_key(g.text):
                out.append(ABSTAIN)
                continue
            exact = countrynames.to_code(g.text)
            if exact:
                out.append(_code(exact))
                continue
            p = _code(countrynames.to_code(g.text, fuzzy=True))
            out.append(Prediction(p.target, p.candidates, 0.5) if p.target else ABSTAIN)
        return out


class PyCountry:
    """pycountry `lookup`, then `search_fuzzy` (LGPL; evaluation only)."""

    name = "pycountry"

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        import pycountry

        out = []
        for g in items:
            text = g.text.strip()
            if not match_key(text):
                out.append(ABSTAIN)
                continue
            try:
                out.append(_code(pycountry.countries.lookup(text).alpha_2))
                continue
            except LookupError:
                pass
            try:
                hit = pycountry.countries.search_fuzzy(text)[0]
                p = _code(getattr(hit, "alpha_2", None))
                out.append(Prediction(p.target, p.candidates, 0.5) if p.target else ABSTAIN)
            except LookupError:
                out.append(ABSTAIN)
        return out


class Coco:
    """country_converter (GPL-3; evaluation only)."""

    name = "coco"

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        import country_converter as coco

        logging.getLogger("country_converter").setLevel(logging.CRITICAL)
        cc = coco.CountryConverter()
        out = []
        for g in items:
            if not match_key(g.text):
                out.append(ABSTAIN)
                continue
            res = cc.convert(g.text, to="ISO2", not_found="")
            res = res if isinstance(res, str) and len(res) == 2 else None
            out.append(_code(res))
        return out


EmbedFn = Callable[[list[str]], np.ndarray]


def hf_embedder(model_name: str = "intfloat/multilingual-e5-small", batch: int = 512) -> EmbedFn:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    @torch.inference_mode()
    def embed(texts: list[str]) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), batch):
            enc = tok(
                texts[i : i + batch],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1)
            outs.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 1), dtype=np.float32)

    return embed


class E5Knn:
    """Untrained multilingual embedding nearest neighbour over lexicon aliases (no abstention
    by design: confidence = cosine similarity of the nearest alias)."""

    name = "e5_knn"

    def __init__(
        self,
        lexicon: Lexicon,
        embed_fn: EmbedFn | None = None,
        cache: Path | None = Path("data/processed/baselines/e5_index.npz"),
    ) -> None:
        self.lexicon = lexicon
        self.embed = embed_fn or hf_embedder()
        reps: dict[str, str] = {}
        for r in lexicon.rows:
            reps.setdefault(compact_key(r.alias), r.alias)
        self.keys = [k for k in reps if k]
        self.aliases = [reps[k] for k in self.keys]
        sig = hashlib.sha256("\x1f".join(self.keys).encode()).hexdigest()[:16]
        if cache is not None and cache.exists() and np.load(cache)["sig"].item() == sig:
            self.matrix = np.load(cache)["m"]
        else:
            self.matrix = self.embed([f"query: {a}" for a in self.aliases])
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, m=self.matrix, sig=np.array(sig))

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        texts = [g.text for g in items]
        out: list[Prediction] = [ABSTAIN] * len(items)
        live = [i for i, t in enumerate(texts) if match_key(t)]
        if not live:
            return out
        q = self.embed([f"query: {texts[i]}" for i in live])
        sims = q @ self.matrix.T
        best = sims.argmax(1)
        for j, i in enumerate(live):
            k = self.keys[int(best[j])]
            p = _from_set(self.lexicon.lookup(k), float(sims[j, best[j]]))
            if p.target is None and p.candidates:  # nearest key is ambiguous: answer anyway
                p = Prediction(p.candidates[0], p.candidates, float(sims[j, best[j]]) / 2)
            out[i] = p
        return out
