"""Polite cached HTTP client (CLAUDE.md §2.7): rate limit, backoff, descriptive User-Agent,
every successful response cached with fetch time and SHA-256, never re-downloaded."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

DEFAULT_UA = "CountryCode-Decide/0.1 (research; +https://github.com/aliildan)"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class CachedClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        user_agent: str = DEFAULT_UA,
        max_rps: float = 5.0,
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max_rps
        self.max_retries = max_retries
        self.sleep = sleep
        self._last = 0.0
        self._client = httpx.Client(
            transport=transport, timeout=120, headers={"User-Agent": user_agent}
        )

    def _path(self, url: str, params: Mapping[str, str] | None) -> Path:
        key = url + "?" + urlencode(sorted((params or {}).items()))
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    def get(
        self,
        url: str,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        path = self._path(url, params)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["body"].encode("utf-8")
        for attempt in range(self.max_retries + 1):
            self._throttle()
            resp = self._client.get(url, params=dict(params or {}), headers=dict(headers or {}))
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                ra = resp.headers.get("retry-after", "")
                self.sleep(float(ra) if ra.isdigit() else min(60.0, 2.0**attempt))
                continue
            resp.raise_for_status()
            body = resp.content
            envelope = {
                "url": url,
                "params": dict(params or {}),
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(body).hexdigest(),
                "body": body.decode("utf-8"),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
            return body
        raise RuntimeError("unreachable")

    def get_json(
        self,
        url: str,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        return json.loads(self.get(url, params, headers))
