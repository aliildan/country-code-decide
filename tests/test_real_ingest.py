import json
from urllib.parse import parse_qs, urlparse

import httpx

from countrycode.http import CachedClient
from countrycode.real.socrata import fetch_socrata_values


def paged_client(tmp_path, pages):
    seen = []

    def handler(req):
        q = parse_qs(urlparse(str(req.url)).query)
        seen.append(q)
        offset = int(q["$offset"][0])
        body = pages.get(offset, [])
        return httpx.Response(200, text=json.dumps(body))

    return CachedClient(
        tmp_path, transport=httpx.MockTransport(handler), sleep=lambda s: None
    ), seen


def test_pages_until_short_page(tmp_path):
    pages = {
        0: [{"f": "Delaware", "n": "10"}, {"f": "Deleware", "n": "2"}, {"n": "5"}],
        3: [{"f": "  ", "n": "1"}],
    }
    client, seen = paged_client(tmp_path, pages)
    values = fetch_socrata_values(client, "data.x.gov", "abcd-1234", "f", page_size=3)
    assert values == [("Delaware", 10), ("Deleware", 2), ("  ", 1)]  # null skipped, blank kept
    assert [int(q["$offset"][0]) for q in seen] == [0, 3]
    assert "count(*)" in seen[0]["$select"][0] and seen[0]["$group"] == ["f"]
