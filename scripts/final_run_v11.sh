#!/bin/sh
# v1.1 release evaluation for one chosen training run (plan-06 Task 6).
# Usage: scripts/final_run_v11.sh <run-name> <version>
# test_v11 is read once: needs docs/preregistration-v1.1.md with "Status: frozen".
# Writes results/{select,test_v11,ood,test_seen_v1}/ and results/latency-<version>.json; the
# README/model card are regenerated only when v1.1 is released (after talking with the owner).
set -eu
RUN="$1"; VERSION="$2"
B="dist/countrycode-$VERSION"
V1="dist/countrycode-small-v1"
LAT="results/latency-$VERSION.json"
uv run python -m countrycode.infer bundle --run "runs/$RUN" --out "$B" --version "$VERSION"
uv run python -m countrycode.calib.fit --bundle "$B" --v11
uv run python -m countrycode.infer bench --bundle "$B" --threads 4 --n 1000 --out "$LAT" > /dev/null
P50=$(python3 -c "import json; print(json.load(open('$LAT'))['p50_ms'])")
F="uv run python -m countrycode.evaluation.final --v11 --bundle $B --cpu-p50-ms $P50 --v1-bundle $V1"
$F --part select
$F --part test_v11 --allow-test
$F --part ood
$F --part test_seen_v1 --allow-test
for P in select test_v11 ood test_seen_v1; do
  uv run python -m countrycode.evaluation.report "$P" > /dev/null
done
echo "v1.1 final run done: $B"
