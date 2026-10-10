#!/bin/sh
# v1.2 release evaluation (plan-07 Task 7) for a bundle already built from a (trimmed) run.
# Usage: scripts/final_run_v12.sh <version> <training run>   e.g. small-v1.2 small-v12f
# test_v12 is read once: needs docs/preregistration-v1.2.md with "Status: frozen".
set -eu
VERSION="$1"
RUN="$2"
B="dist/countrycode-$VERSION"
V1="dist/countrycode-small-v1"
LAT="results/latency-$VERSION.json"
TRIM="results/trim-$VERSION.json"
uv run python -m countrycode.infer stamp-g1 --bundle "$B" --train-summary "results/train/$RUN.json"
uv run python -m countrycode.calib.fit --bundle "$B" --v12
uv run python -m countrycode.infer bench --bundle "$B" --threads 4 --n 1000 --out "$LAT" > /dev/null
P50=$(python3 -c "import json; print(json.load(open('$LAT'))['p50_ms'])")
F="uv run python -m countrycode.evaluation.final --v12 --bundle $B --cpu-p50-ms $P50 --v1-bundle $V1 --trim $TRIM"
$F --part select
$F --part ood
$F --part test_v11 --allow-test
$F --part test_v12 --allow-test
for P in select ood test_v11 test_v12; do
  uv run python -m countrycode.evaluation.report "$P" > /dev/null
done
echo "v1.2 final run done: $B"
