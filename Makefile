# CountryCode-Decide — common commands. Data commands print aggregates only.
UV := uv run

.PHONY: env hooks test lint fmt publish-check tickets lexicon real review review-json review-import audit audit-json audit-import blind-json blind-import synth serve

env:            ## sync dependencies
	uv sync

hooks:          ## install git hooks (pre-commit publish check, pre-push CI rehearsal + main guard)
	git config core.hooksPath scripts/hooks

test:
	$(UV) pytest -q

lint:
	$(UV) ruff check . && $(UV) ruff format --check . && $(UV) pyright

fmt:
	$(UV) ruff format . && $(UV) ruff check --fix .

publish-check:  ## scan the public allow-list for data and secrets
	python3 scripts/publish_check.py --publish

tickets:        ## mirror docs/plans/*.md to GitHub milestones and issues (private dev repo)
	$(UV) python scripts/tickets.py sync

lexicon:        ## build data/processed/lexicon/lexicon-v1.json + results/lexicon/stats.json
	$(UV) python -m countrycode.lexicon build

real:           ## ingest real evaluation strings (aggregates) and split them
	$(UV) python -m countrycode.real ingest && $(UV) python -m countrycode.real split

review:         ## owner review tool for real gold labels: http://127.0.0.1:8310
	$(UV) python eval/review/app.py

review-json:    ## the review queue as an editable JSON file: data/gold/real-v1/review.json
	$(UV) python eval/review/json_queue.py export

review-import:  ## owner's JSON answers → data/gold/real-v1/owner.jsonl
	$(UV) python eval/review/json_queue.py import

audit:          ## owner G1 audit of the synthetic release: http://127.0.0.1:8311
	$(UV) python eval/review/audit_app.py

audit-json:     ## the G1 audit sample as an editable JSON file: data/gold/synth-v1/audit.json
	$(UV) python eval/review/audit_json.py export

audit-import:   ## owner's JSON verdicts → data/gold/synth-v1/audit.jsonl
	$(UV) python eval/review/audit_json.py import

blind-json:     ## blind re-labelling sample (no proposals): data/gold/real-v1/blind.json
	$(UV) python eval/review/blind_json.py export

blind-import:   ## owner's blind answers → blind.jsonl, gold-quality report, adjudication file
	$(UV) python eval/review/blind_json.py import && $(UV) python eval/review/blind_json.py report && $(UV) python eval/review/blind_json.py adjudicate

synth:          ## synthetic release (needs OCR + LLM caches) + G2 realism report
	$(UV) python -m countrycode.synth release --gens junk,regjunk,keyboard,format,translit,geo,pairs,affix,region,caprov

serve:          ## inference server on a release bundle: make serve BUNDLE=dist/<bundle>
	$(UV) python server/app.py --bundle $(BUNDLE)
