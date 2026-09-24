#!/usr/bin/env python3
"""The shared CMR pager of family 1 (`ml/family1/adapters/_common.py`) — no
network.

    python3 -m pytest -q tests/test_family1_cmr_walk.py

`cm.cmr_entries` pages one CMR granule query with the `CMR-Search-After`
cursor and checks the walk against CMR's own `CMR-Hits` count. It exists
because of the 2026-09-23 irtb (cloud-top temperature) build: during a slow
hour of CMR the service answered HTTP 200 with a `CMR-Timed-Out: true`
header and a partial page, and the pager of the day took a page shorter than
`page_size` as the end of the listing — 247 of 2,160 hours of 2013 Q1, the
other 1,913 booked `absent_upstream`, sixteen years short by 15,388 frames
from listings that looked complete. These tests pin every refusal and the
retry, and that the four adapters (irtb, sst_acspo02, xco2, swot) refuse a
cut listing rather than build from it.
"""
import io
import json
import os
import sys
import urllib.error
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

from family1.adapters import _common as cm                      # noqa: E402


class _Resp:
    def __init__(self, entries, headers):
        self._raw = json.dumps({"feed": {"entry": entries}}).encode()
        self.headers = {k: str(v) for k, v in headers.items()}
        self.status = 200

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(monkeypatch, pages):
    """`pages` is a list of (entries, headers) or an Exception, served in
    order; the requests are recorded as (url, CMR-Search-After)."""
    asked = []

    def fake(req, timeout=None):
        asked.append((req.full_url, req.get_header("Cmr-search-after")))
        p = pages[len(asked) - 1]
        if isinstance(p, Exception):
            raise p
        return _Resp(*p)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)
    return asked


def _e(i):
    return {"id": f"G{i}", "title": f"g{i}", "time_start": f"t{i}"}


def test_a_full_walk_follows_the_cursor_and_matches_cmr_hits(monkeypatch):
    asked = _serve(monkeypatch, [
        ([_e(1), _e(2), _e(3)], {"CMR-Hits": 5, "CMR-Search-After": "A"}),
        ([_e(4), _e(5)], {"CMR-Hits": 5, "CMR-Search-After": "B"}),
    ])
    seen = []
    got = cm.cmr_entries("https://cmr/x", {"collection_concept_id": "C"},
                         page_size=3, count=seen.append)
    assert [g["id"] for g in got] == ["G1", "G2", "G3", "G4", "G5"]
    assert asked[0][1] is None and asked[1][1] == "A"
    assert "page_size=3" in asked[0][0] and len(seen) == 2


def test_a_partial_page_is_retried_not_parsed(monkeypatch):
    """`CMR-Timed-Out: true` is CMR giving up mid-query: the rows it did
    return are NOT the listing, and the page is asked for again."""
    asked = _serve(monkeypatch, [
        ([_e(1)], {"CMR-Hits": 3, "CMR-Timed-Out": "true"}),
        ([_e(1), _e(2), _e(3)], {"CMR-Hits": 3}),
    ])
    got = cm.cmr_entries("https://cmr/x", {}, page_size=2000)
    assert [g["id"] for g in got] == ["G1", "G2", "G3"] and len(asked) == 2


def test_a_short_walk_is_retried_whole_and_then_refused(monkeypatch):
    """The 2013 Q1 case: 247 rows against CMR-Hits 2160, no cursor. Three
    walks, then a refusal naming the shortfall — never a listing."""
    page = ([_e(i) for i in range(247)], {"CMR-Hits": 2160})
    asked = _serve(monkeypatch, [page, page, page])
    with pytest.raises(cm.CMRTruncated, match="short by 1913"):
        cm.cmr_entries("https://cmr/x", {}, what="irtb 2013 Q1")
    assert len(asked) == 3
    # …and a walk that recovers on its second try is returned
    asked = _serve(monkeypatch, [
        page, ([_e(i) for i in range(2160)], {"CMR-Hits": 2160})])
    assert len(cm.cmr_entries("https://cmr/x", {})) == 2160


def test_the_slack_tolerates_the_header_overcount(monkeypatch):
    """CMR-Hits 243 for 242 granules (the measured VNP02IMG case) is the
    header's bookkeeping, inside `CMR_HITS_SLACK`."""
    _serve(monkeypatch, [([_e(i) for i in range(242)], {"CMR-Hits": 243})])
    assert len(cm.cmr_entries("https://cmr/x", {})) == 242


def test_more_rows_than_hits_no_hits_and_a_cut_page_refuse(monkeypatch):
    _serve(monkeypatch, [([_e(1), _e(2)], {"CMR-Hits": 1})])
    with pytest.raises(cm.CMRTruncated, match="does not count"):
        cm.cmr_entries("https://cmr/x", {}, walk_attempts=1)
    _serve(monkeypatch, [([_e(1)], {})])
    with pytest.raises(cm.CMRTruncated, match="never sent a CMR-Hits"):
        cm.cmr_entries("https://cmr/x", {}, walk_attempts=1)
    # a short page ends the walk; CMR-Hits says whether that was the end
    asked = _serve(monkeypatch, [
        ([_e(1)], {"CMR-Hits": 30, "CMR-Search-After": "A"}),
        ([_e(2), _e(3)], {"CMR-Hits": 30}),
    ])
    with pytest.raises(cm.CMRTruncated, match="short by 29"):
        cm.cmr_entries("https://cmr/x", {}, page_size=2, walk_attempts=1)
    assert len(asked) == 1
    _serve(monkeypatch, [
        ([_e(1)], {"CMR-Hits": 3, "CMR-Search-After": "A"}),
        ([_e(2)], {"CMR-Hits": 4}),
    ])
    with pytest.raises(cm.CMRTruncated, match="CMR-Hits changed"):
        cm.cmr_entries("https://cmr/x", {}, page_size=1, walk_attempts=1)


def test_transport_failures_retry_and_a_bad_request_does_not(monkeypatch):
    asked = _serve(monkeypatch, [
        urllib.error.URLError("reset"),
        ([_e(1)], {"CMR-Hits": 1}),
    ])
    assert len(cm.cmr_entries("https://cmr/x", {})) == 1 and len(asked) == 2
    bad = urllib.error.HTTPError("https://cmr/x", 400, "bad", {}, io.BytesIO())
    asked = _serve(monkeypatch, [bad, bad, bad, bad])
    with pytest.raises(IOError, match="HTTP Error 400"):
        cm.cmr_entries("https://cmr/x", {}, walk_attempts=1)
    assert len(asked) == 1


def test_every_adapter_refuses_a_cut_listing(monkeypatch):
    """irtb, sst_acspo02, xco2 and swot all list through the shared walker,
    and a `CMRTruncated` reaches their call sites as a REFUSAL (sys.exit),
    never as an empty or short listing to build from."""
    from family1.adapters import irtb, sst_acspo02, swot, xco2

    def cut(*a, **k):
        raise cm.CMRTruncated("short by 1913")
    monkeypatch.setattr(cm, "cmr_entries", cut)
    with pytest.raises(cm.CMRTruncated):
        irtb.cmr_hours("2013-01-01T00:00:00Z", "2013-03-31T23:59:59Z")
    with pytest.raises(cm.CMRTruncated):
        sst_acspo02.cmr_days("C1", "2013-01-01T00:00:00Z",
                             "2013-03-31T23:59:59Z")
    with pytest.raises(cm.CMRTruncated):
        xco2.cmr_days("C1")
    with pytest.raises(cm.CMRTruncated):
        swot.cmr_granules(temporal="2024-01-01T00:00:00Z,2024-01-31T23:59:59Z")
    for mod in (irtb, sst_acspo02, xco2, swot):
        src = open(mod.__file__).read()
        assert "except (FormatError, cm.CMRTruncated) as e:" in src, mod
        assert "cm.get_bytes(f\"{CMR}" not in src, mod


def test_sst_acspo02_keeps_one_of_an_identical_granule_listed_twice(monkeypatch):
    """The whole-record assembly #735 (2026-09-24) refused on 2005-09-30
    listed twice with the SAME name — the old pager's page seam. Two
    identical rows are one granule; two DIFFERENT names for a day stay the
    day/night-split refusal."""
    from family1.adapters import sst_acspo02 as s
    n = "20050930120000-STAR-L3S_GHRSST-SSTsubskin-LEO_Daily-ACSPO_V2.81-v02.0-fv01.0"
    link = [{"href": "https://x/protected/" + n + ".nc"}]
    rows = [{"title": n, "links": link, "granule_size": "400"},
            {"title": n, "links": link, "granule_size": "400"}]
    monkeypatch.setattr(s.cm, "cmr_entries", lambda *a, **k: rows)
    got = s.cmr_days("C", "2005-09-30T00:00:00Z", "2005-09-30T23:59:59Z")
    assert list(got) == [__import__("datetime").date(2005, 9, 30)]
    other = n.replace("LEO_Daily", "LEO_PM_N")
    rows.append({"title": other, "links": link, "granule_size": "400"})
    with pytest.raises(s.FormatError, match="two granules"):
        s.cmr_days("C", "2005-09-30T00:00:00Z", "2005-09-30T23:59:59Z")
