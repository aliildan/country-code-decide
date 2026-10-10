"""Generated result tables (CLAUDE.md §6: tables are generated, never typed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = (
    ("accuracy", "acc"),
    ("accuracy_weighted", "acc (freq-weighted)"),
    ("country_accuracy", "country acc"),
    ("coverage@1%", "coverage @1 % risk"),
    ("coverage@5%", "coverage @5 % risk"),
    ("none_false_accept", "non-place false accept"),
    ("ece", "ECE"),
    ("latency", "ms / item"),
)


def _row(rep: dict) -> dict[str, float | None]:
    o = rep["overall"]
    cov = o.get("coverage_at_risk", {})
    return {
        "accuracy": o.get("accuracy"),
        "accuracy_weighted": o.get("accuracy_weighted"),
        "country_accuracy": o.get("country_accuracy"),
        "coverage@1%": cov.get("0.01"),
        "coverage@5%": cov.get("0.05"),
        "none_false_accept": o.get("none_false_accept"),
        "ece": o.get("ece"),
        "latency": rep.get("latency_ms_per_item"),
    }


def _fmt(key: str, v: float | None) -> str:
    if v is None or v != v:  # None or NaN
        return "–"
    return f"{v:.1f}" if key == "latency" else f"{v:.3f}"


def markdown_table(part_dir: Path) -> str:
    reps = [json.loads(p.read_text()) for p in sorted(Path(part_dir).glob("*.json"))]
    reps = [r for r in reps if "overall" in r]
    reps.sort(key=lambda r: -(r["overall"].get("accuracy") or 0))
    head = "| predictor | n | " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 2)
    lines = [head, sep]
    for r in reps:
        row = _row(r)
        cells = " | ".join(_fmt(k, row[k]) for k, _ in COLUMNS)
        if r["predictor"].startswith("llm_") and (row["latency"] or 0) < 1.0:
            cells = cells.rsplit(" | ", 1)[0] + " | cached"  # answers read from the LLM cache
        lines.append(f"| {r['predictor']} | {r['overall']['n']} | {cells} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "dev"
    out = Path("results") / part / "TABLE.md"
    out.write_text(f"# Results on real {part} strings (generated)\n\n" + markdown_table(out.parent))
    print(out.read_text())
