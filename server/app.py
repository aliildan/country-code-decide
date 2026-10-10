"""CountryCode-Decide inference server (plan-05 Task 2).

    uv run python server/app.py --bundle dist/countrycode-small-v1 [--port 8320]

- POST /v1/countrycode — batch: {"items": [{"text", "field"?, "register_country"?, "domain"?}],
  "alpha"?}; "domain" ("register" or "curated") picks the certified calibration policy
- POST /v1/systemone — typed questions (Jev-compatible shape): country (choice), us_state
  (choice), is_country (noul), each with probabilities and `abstain`.
Only `text`, `field`, `register_country`, `source` and `domain` are read from a request; anything
else is ignored and never logged or echoed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from countrycode import labels
from countrycode.infer.pipeline import FIELDS, TARGETS, CountryCoder

ALPHAS = (0.01, 0.02, 0.05)


class Item(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(max_length=500)
    field: str = "unknown"
    register_country: str = "unknown"
    source: str | None = None  # e.g. "us-co"; selects the calibration domain (configs/domains.yaml)
    domain: str | None = None  # "register" / "curated": the calibration policy, wins over source

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty")  # a blank cell is not a question about a country
        return v


class BatchRequest(BaseModel):
    items: list[Item] = Field(max_length=10_000)
    alpha: float | None = 0.05


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    state: Item
    questions: dict[str, dict] = {}
    policy: dict = {}


def _norm(item: Item) -> tuple[str, str, str, str | None, str | None]:
    field = item.field if item.field in FIELDS else "unknown"
    reg = item.register_country.upper()
    ok = labels.is_target(reg) and (len(reg) == 2 or reg.startswith("US-"))  # "US-CO" register
    return item.text, field, reg if ok else "unknown", item.source, item.domain


def _alpha(a: float | None) -> float | None:
    return a if a is None or a in ALPHAS else 0.05


def create_app(coder: CountryCoder) -> FastAPI:
    def check_domain(domain: str | None) -> None:
        if domain is not None and domain not in coder.cals.policies:
            known = sorted(coder.cals.policies)
            raise HTTPException(422, f"unknown calibration domain {domain!r}; known: {known}")

    app = FastAPI(title="CountryCode-Decide", version=str(coder.meta.get("version", "dev")))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": coder.meta.get("version"), "targets": len(TARGETS)}

    @app.post("/v1/countrycode")
    def batch(req: BatchRequest) -> dict:
        norm = [_norm(i) for i in req.items]
        for n in norm:
            check_domain(n[4])
        decisions = coder.code_batch(
            [n[0] for n in norm],
            [n[1] for n in norm],
            [n[2] for n in norm],
            _alpha(req.alpha),
            [n[3] for n in norm],
            [n[4] for n in norm],
        )
        return {"results": [d.to_dict() for d in decisions]}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest) -> dict:
        text, field, reg, source, domain = _norm(req.state)
        check_domain(domain)
        alpha = _alpha(req.policy.get("alpha", 0.05))
        row = coder.probs([text], [field], [reg], [source], [domain])[0]
        d = coder.code_batch([text], [field], [reg], alpha, [source], [domain])[0]
        top = np.argsort(-row)[:5]
        p_country: dict[str, float] = {}
        for t, p in zip(TARGETS, row, strict=True):
            c = labels.country_of(t)
            p_country[c] = p_country.get(c, 0.0) + float(p)
        p_us = p_country.get("US", 0.0)
        states = {t: float(p) / p_us for t, p in zip(TARGETS, row, strict=True)
                  if t.startswith("US-") and p_us > 0}  # fmt: skip
        answers: dict[str, dict] = {}
        want = req.questions or {"country": {}, "us_state": {}, "is_country": {}}
        if "country" in want:
            best = sorted(p_country.items(), key=lambda kv: -kv[1])[:5]
            answers["country"] = {
                "type": "choice", "probs": dict(best), "choice": d.country,
                "abstain": d.abstain, "candidates": list(d.candidates),
            }  # fmt: skip
        if "us_state" in want:
            best_s = sorted(states.items(), key=lambda kv: -kv[1])[:5]
            answers["us_state"] = {
                "type": "choice", "probs": dict(best_s), "choice": d.us_state,
                "abstain": d.abstain or d.country != "US",
            }  # fmt: skip
        if "is_country" in want:
            p_yes = 1.0 - p_country.get(labels.NONE, 0.0)
            lam = coder.policy(source, domain).thresholds.get(str(alpha)) if alpha else None
            answers["is_country"] = {
                "type": "noul", "p_yes": p_yes,
                # λ is certified for the top code, not for this yes/no question: never answer
                # "is a country" when the country question itself abstains
                "abstain": d.abstain or lam is None or max(p_yes, 1 - p_yes) < lam,
            }  # fmt: skip
        return {
            "answers": answers,
            "top_targets": {TARGETS[int(j)]: float(row[j]) for j in top},
            "alpha": alpha,
            "model": coder.meta.get("version"),
        }

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8320)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    uvicorn.run(create_app(CountryCoder.load(args.bundle)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
