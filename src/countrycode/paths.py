"""Repository locations. Configs are read from the repository, never from the working directory, so
the library behaves the same wherever it is called from (pipeline outputs under data/ and results/
stay relative to the working directory, the repository root by convention)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
