import json

import httpx
import pytest

from countrycode.http import DEFAULT_UA, CachedClient


def make(tmp_path, responses):
    calls = []

    def handler(req):
        calls.append(req)
        status, body = responses[min(len(calls), len(responses)) - 1]
        return httpx.Response(status, text=body)

    waits = []
    c = CachedClient(tmp_path, transport=httpx.MockTransport(handler), sleep=waits.append)
    return c, calls, waits


def test_cache_hit_no_second_request(tmp_path):
    c, calls, _ = make(tmp_path, [(200, '{"a": 1}')])
    assert c.get_json("https://x.test/q", {"b": "2"}) == {"a": 1}
    assert c.get_json("https://x.test/q", {"b": "2"}) == {"a": 1}
    assert len(calls) == 1
    env = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert env["url"] == "https://x.test/q" and len(env["sha256"]) == 64


def test_user_agent_sent(tmp_path):
    c, calls, _ = make(tmp_path, [(200, "{}")])
    c.get("https://x.test/ua")
    assert calls[0].headers["user-agent"] == DEFAULT_UA


def test_retry_on_429_then_ok(tmp_path):
    c, calls, waits = make(tmp_path, [(429, ""), (200, "ok")])
    assert c.get("https://x.test/r") == b"ok" and len(calls) == 2 and waits


def test_errors_not_cached(tmp_path):
    c, _, _ = make(tmp_path, [(500, "")] * 10)
    with pytest.raises(httpx.HTTPStatusError):
        c.get("https://x.test/e")
    assert not list(tmp_path.glob("*.json"))
