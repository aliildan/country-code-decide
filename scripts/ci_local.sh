#!/bin/sh
# Rehearse CI on the committed tree only (no docs/, data/, CLAUDE.md or other git-ignored files),
# exactly what a clean checkout sees. Catches code and tests that silently depend on private
# local files. Used by the pre-push hook; also runnable by hand.
set -eu
repo=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ref="${1:-HEAD}"
git -C "$repo" archive "$ref" | tar -x -C "$tmp"
cd "$tmp"
"$repo/.venv/bin/ruff" check . && "$repo/.venv/bin/ruff" format --check .
# The committed src/ must win over the editable install of the working tree.
PYTHONPATH="$tmp/src" "$repo/.venv/bin/python" -m pytest -q -p no:cacheprovider -x
(cd "$repo" && python3 scripts/publish_check.py --publish)  # needs git ls-files
(cd "$repo" && python3 scripts/make_docs.py --check)  # README/model card generated from results/
(cd "$repo" && python3 scripts/make_tables.py --check)  # paper tables generated from results/
