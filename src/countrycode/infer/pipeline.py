"""CountryCoder: the shippable inference pipeline (plan-05 Task 1).

A bundle directory holds `model.int8.onnx` (and `model.onnx`), `tokenizer.json`,
`calibration.json` and `meta.json`. Serving needs only onnxruntime, tokenizers and numpy.

    coder = CountryCoder.load("dist/countrycode-small-v1")
    coder.code("Østerrike")                      # Decision(target='AT', ...) at α = 5 %
    coder.code("CA", field="jurisdiction", register_country="US", alpha=None)  # always answer
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from countrycode import labels
from countrycode.calib.calibration import (
    Calibration,
    CalibrationSet,
    candidates,
    decide,
    domain_of,
    softmax_t,
)
from countrycode.infer.heads import combine
from countrycode.train.dataset import COUNTRY_CLASSES, STATE_CLASSES, context_text

TARGETS = list(labels.targets())
FIELDS = ("country", "jurisdiction", "nationality", "unknown")


@dataclass(frozen=True)
class Decision:
    text: str
    target: str | None  # None = abstain
    country: str | None
    us_state: str | None
    confidence: float
    candidates: tuple[str, ...]
    abstain: bool
    alpha: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _name(t: str) -> str:
    if t == labels.NONE:
        return "not a country or US state"
    return labels.us_state_name(t) if t.startswith("US-") else labels.country_name(t)


def make_bundle(
    model,
    tokenizer_json: Path,
    cal: Calibration | CalibrationSet,
    out: Path,
    meta: dict | None = None,
) -> Path:
    from countrycode.infer.onnx_export import export_onnx, quantize_weights_only

    out.mkdir(parents=True, exist_ok=True)
    fp32 = export_onnx(model, out / "model.onnx")
    quantize_weights_only(fp32, out / "model.int8.onnx")
    shutil.copy(tokenizer_json, out / "tokenizer.json")
    cal.save(out / "calibration.json")
    full = {
        "max_len": 48,
        **(meta or {}),
        "targets": TARGETS,
        "names": {t: _name(t) for t in TARGETS},
        "country_classes": list(COUNTRY_CLASSES),
        "state_classes": list(STATE_CLASSES),
        "context_format": context_text("<field>", "<register>"),
    }
    (out / "meta.json").write_text(json.dumps(full, indent=1, ensure_ascii=False))
    return out


class CountryCoder:
    def __init__(self, session, tokenizer, cals: CalibrationSet, meta: dict) -> None:
        self.session, self.tok, self.cals, self.meta = session, tokenizer, cals, meta
        self.max_len = int(meta.get("max_len", 48))

    @classmethod
    def load(cls, bundle: Path | str, int8: bool = True, threads: int = 4) -> CountryCoder:
        from tokenizers import Tokenizer

        from countrycode.infer.runtime import ort_session

        b = Path(bundle)
        session = ort_session(b / ("model.int8.onnx" if int8 else "model.onnx"), threads=threads)
        tok = Tokenizer.from_file(str(b / "tokenizer.json"))
        tok.enable_truncation(max_length=int(json.loads((b / "meta.json").read_text())["max_len"]))
        tok.enable_padding(pad_id=tok.token_to_id("[PAD]") or 0)
        cals = CalibrationSet.load(b / "calibration.json")
        return cls(session, tok, cals, json.loads((b / "meta.json").read_text()))

    @property
    def cal(self) -> Calibration:
        """Policy for an unknown source (strictest merge of all domains)."""
        return self.cals.for_domain(None)

    def policy(self, source: str | None, domain: str | None = None) -> Calibration:
        """`domain` ("register", "curated") wins over `source` (a known source id)."""
        if domain is not None and domain not in self.cals.policies:
            known = ", ".join(sorted(self.cals.policies))
            raise ValueError(f"unknown calibration domain {domain!r} (known: {known})")
        return self.cals.for_domain(domain or domain_of(source))

    def logits(
        self, texts: Sequence[str], fields: Sequence[str], registers: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        pairs = [
            (context_text(f if f in FIELDS else "unknown", (r or "unknown").upper()
                          if r and r != "unknown" else "unknown"),
             t if t.strip() else "∅")
            for t, f, r in zip(texts, fields, registers, strict=True)
        ]  # fmt: skip
        enc = self.tok.encode_batch(pairs)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        c, s = self.session.run(None, {"input_ids": ids, "attention_mask": mask})
        return np.asarray(c, dtype=np.float64), np.asarray(s, dtype=np.float64)

    def probs(
        self,
        texts: Sequence[str],
        fields: Sequence[str],
        registers: Sequence[str],
        sources: Sequence[str | None] | None = None,
        domains: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calibrated probabilities; temperatures from each item's domain (given, or the
        source's)."""
        c, s = self.logits(texts, fields, registers)
        srcs = sources or [None] * len(texts)
        doms = domains or [None] * len(texts)
        rows = []
        for i, (src, dom) in enumerate(zip(srcs, doms, strict=True)):
            pol = self.policy(src, dom)
            rows.append(
                combine(
                    softmax_t(c[i : i + 1], pol.temps[0]), softmax_t(s[i : i + 1], pol.temps[1])
                )[0]
            )
        return np.array(rows) if rows else np.zeros((0, len(TARGETS)))

    def code_batch(
        self,
        texts: Sequence[str],
        fields: Sequence[str] | None = None,
        registers: Sequence[str] | None = None,
        alpha: float | None = 0.05,
        sources: Sequence[str | None] | None = None,
        domains: Sequence[str | None] | None = None,
    ) -> list[Decision]:
        n = len(texts)
        srcs = list(sources) if sources is not None else [None] * n
        doms = list(domains) if domains is not None else [None] * n
        rows = self.probs(
            texts, fields or ["unknown"] * n, registers or ["unknown"] * n, srcs, doms
        )
        out = []
        for text, row, src, dom in zip(texts, rows, srcs, doms, strict=True):
            if alpha is None:  # always answer; candidates by the policy's rule, as with a target
                j = int(np.argmax(row))
                p_target: str | None = TARGETS[j]
                cands = tuple(TARGETS[i] for i in candidates(row, self.policy(src, dom)))
                conf = float(row[j])
            else:
                pred = decide(row, self.policy(src, dom), alpha, TARGETS)
                p_target, cands, conf = pred.target, pred.candidates, pred.confidence
            country = labels.country_of(p_target) if p_target else None
            out.append(
                Decision(
                    text=text,
                    target=p_target,
                    country=None if country == labels.NONE else country,
                    us_state=p_target if p_target and p_target.startswith("US-") else None,
                    confidence=conf,
                    candidates=cands,
                    abstain=p_target is None,
                    alpha=alpha,
                )
            )
        return out

    def code(
        self, text: str, field: str = "unknown", register_country: str = "unknown",
        alpha: float | None = 0.05, source: str | None = None, domain: str | None = None,
    ) -> Decision:  # fmt: skip
        return self.code_batch([text], [field], [register_country], alpha, [source], [domain])[0]
