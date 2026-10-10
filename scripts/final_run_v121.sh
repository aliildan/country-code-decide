#!/bin/sh
# v1.2.1 (plan-11): the v1.2 weights with a certificate over (source, normalised key) units.
# Post-hoc: the pre-registered read of test_v12 (results/test_v12, v1.2 calibration) stays as it
# was; this run writes results/posthoc-v1.2.1/ and documents v1.2.1 in results/release.json.
set -eu
B=dist/countrycode-small-v1.2.1
OUT=results/posthoc-v1.2.1
rm -rf "$B" && cp -r dist/countrycode-small-v1.2 "$B"
python3 -c "import json,pathlib; p=pathlib.Path('$B/meta.json'); m=json.loads(p.read_text()); \
m['version']='v1.2.1'; p.write_text(json.dumps(m, indent=1, ensure_ascii=False))"
uv run python -m countrycode.calib.fit --bundle "$B" --v12 --units
uv run python -m countrycode.infer bench --bundle "$B" --threads 4 --n 1000 \
  --out results/latency-small-v1.2.1.json > /dev/null
P50=$(python3 -c "import json; print(json.load(open('results/latency-small-v1.2.1.json'))['p50_ms'])")
for P in test_v12 test_v11 ood; do
  uv run python -m countrycode.evaluation.final --v12 --bundle "$B" --cpu-p50-ms "$P50" \
    --v1-bundle dist/countrycode-small-v1 --trim results/trim-small-v1.2.json --part "$P" \
    --allow-test --results-dir "$OUT"
  uv run python -m countrycode.evaluation.report "posthoc-v1.2.1/$P" > /dev/null
done
uv run python scripts/risk_coverage.py --part test_v12 --results-dir "$OUT"
python3 -c "import json, pathlib; pathlib.Path('results/release.json').write_text(json.dumps({ \
'version': 'small-v1.2.1', 'part': 'posthoc-v1.2.1/test_v12', 'preregistered': 'test_v12', \
'hypotheses': 'v12', 'train_run': 'small-v12f', 'model_version': 'small-v1.2'}, indent=1) + '\\n')"
uv run python scripts/make_tables.py --bundle "$B"
uv run python scripts/run_examples.py --bundle "$B" --alpha 0.05
uv run python scripts/make_docs.py
echo "v1.2.1 done: $B"
