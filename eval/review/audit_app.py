"""Owner audit of synthetic data (G1, plan-03 Task 9). Local only: http://127.0.0.1:8311

    uv run python eval/review/audit_app.py

Shows one sampled synthetic item at a time: the generated text, the label it would train with
(codes and names) and the alias it came from. Enter = label correct, x = wrong (optionally type
the right codes first), s = skip, b = back. Verdicts are appended to
data/gold/synth-v1/audit.jsonl; the gate report counts them (countrycode.synth.audit).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from countrycode import labels
from countrycode.synth.audit import (
    AUDIT_PATH,
    audit_status,
    file_sha256,
    read_verdicts,
    sample_for_audit,
)
from countrycode.synth.framework import SynthItem

PAGE = (Path(__file__).parent / "audit.html").read_text(encoding="utf-8")


class Verdict(BaseModel):
    tag: str
    text: str
    verdict: str
    targets: list[str] = []


def _name(t: str) -> str:
    if t == labels.NONE:
        return "not a country / US state"
    return labels.us_state_name(t) if t.startswith("US-") else labels.country_name(t)


def create_app(
    sample: Sequence[SynthItem], audit_path: Path = AUDIT_PATH, synth_dir: Path | None = None
) -> FastAPI:
    from countrycode.synth.framework import SYNTH_DIR

    synth = Path(synth_dir) if synth_dir is not None else SYNTH_DIR
    app = FastAPI(title="Synthetic audit")
    keys = {(f"{s.generator}@{s.version}", s.text): s for s in sample}
    order = list(keys)
    skipped: set[tuple[str, str]] = set()

    def payload(k: tuple[str, str] | None) -> dict:
        prog = audit_status(sample, audit_path, synth_dir=synth)
        if k is None:
            return {"item": None, "progress": prog}
        s = keys[k]
        return {
            "item": {
                "tag": k[0],
                "text": s.text,
                "targets": list(s.targets),
                "names": [_name(t) for t in s.targets],
                "source_alias": s.source_alias,
            },
            "progress": prog,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/next")
    def nxt() -> dict:
        done = set(read_verdicts(audit_path))
        pending = [k for k in order if k not in done]
        fresh = [k for k in pending if k not in skipped]
        return payload((fresh or pending or [None])[0])

    @app.get("/api/item")
    def item(tag: str, text: str) -> dict:
        if (tag, text) not in keys:
            raise HTTPException(404, "unknown item")
        return payload((tag, text))

    @app.post("/api/verdict")
    def verdict(v: Verdict) -> dict:
        k = (v.tag, v.text)
        if k not in keys:
            raise HTTPException(404, "unknown item")
        if v.verdict == "skip":
            skipped.add(k)
            return {"ok": True}
        if v.verdict not in ("ok", "wrong"):
            raise HTTPException(422, "verdict must be ok, wrong or skip")
        bad = [t for t in v.targets if not labels.is_target(t.upper())]
        if bad:
            raise HTTPException(422, f"invalid targets: {bad}")
        rec = {
            "tag": v.tag,
            "text": v.text,
            "targets": list(keys[k].targets),
            "verdict": v.verdict,
            "owner_targets": [t.upper() for t in v.targets],
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "file_sha256": file_sha256(synth / f"{v.tag}.jsonl"),  # binds the verdict to the file
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        skipped.discard(k)
        return {"ok": True}

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8311)
    ap.add_argument(
        "--host", default="127.0.0.1", help="0.0.0.0 = reachable from the LAN (no auth)"
    )
    args = ap.parse_args()
    sample = sample_for_audit()
    print(f"audit: http://127.0.0.1:{args.port}  ({len(sample)} sampled items)")
    uvicorn.run(create_app(sample), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
