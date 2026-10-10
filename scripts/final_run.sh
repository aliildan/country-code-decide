#!/bin/sh
# Final release evaluation for one chosen training run (plan-04 T5–T7, plan-05 T4/T6).
# Usage: scripts/final_run.sh <run-name> <version> <h4-dev-gain>
# Test is read once: needs docs/preregistration.md with "Status: frozen".
set -eu
RUN="$1"; VERSION="$2"; GAIN="$3"
B="dist/countrycode-$VERSION"
uv run python -m countrycode.infer bundle --run "runs/$RUN" --out "$B" --version "$VERSION"
uv run python -m countrycode.calib.fit --bundle "$B"
uv run python -m countrycode.infer bench --bundle "$B" --threads 4 --n 1000 > /tmp/countrycode-bench.json
P50=$(python3 -c "import json; print(json.load(open('results/latency.json'))['p50_ms'])")
uv run python -m countrycode.evaluation.final --bundle "$B" --part dev --h4-dev-gain "$GAIN" --cpu-p50-ms "$P50"
uv run python -m countrycode.evaluation.final --bundle "$B" --part test --allow-test --h4-dev-gain "$GAIN" --cpu-p50-ms "$P50"
uv run python -m countrycode.evaluation.report dev > /dev/null
uv run python -m countrycode.evaluation.report test > /dev/null
uv run python scripts/linkedin_assets.py --part test
uv run python scripts/make_docs.py
echo "final run done: $B"
