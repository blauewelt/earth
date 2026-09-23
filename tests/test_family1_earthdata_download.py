#!/usr/bin/env python3
"""`earthdata_download` (ml/family1/adapters/_common.py) — no network.

    python3 -m pytest -q tests/test_family1_earthdata_download.py

The helper every Earthdata adapter (swot, irtb, sst_acspo02, xco2) downloads
through. Three outcomes are pinned against a fake `requests` session: a
runner that cannot CONNECT exits the lane (`SystemExit`, which the adapters'
per-granule `except IOError` does not catch) instead of noting every
remaining granule absent — family1-build #498/#516, 2026-09-23; an HTTP 500
after the retries is still an `IOError` about one file; and a 404 is a
counted absence, returned at once without a retry.
"""
import os
import sys

import pytest
import requests.exceptions as rqe

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

from family1.adapters import _common as cm                     # noqa: E402

URL = ("https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-"
       "protected/SWOT_L2_LR_SSH_2.0/granule.nc")


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.history = []
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, n):
        return iter(())


class _Session:
    """`get()` either raises `exc` or answers `status`, and counts calls."""

    def __init__(self, exc=None, status=None):
        self.exc, self.status, self.calls = exc, status, 0

    def get(self, url, **kw):
        self.calls += 1
        self.kw = kw
        if self.exc is not None:
            raise self.exc
        return _Resp(self.status)


def test_a_runner_that_cannot_connect_exits_the_lane(tmp_path):
    s = _Session(exc=rqe.ConnectTimeout(
        "HTTPSConnectionPool(host='urs.earthdata.nasa.gov', port=443): "
        "Connection to urs.earthdata.nasa.gov timed out. "
        "(connect timeout=30)"))
    with pytest.raises(SystemExit) as ei:
        cm.earthdata_download(s, URL, str(tmp_path / "g.nc"),
                              attempts=2, sleep=0.0)
    msg = str(ei.value.code)
    assert URL in msg
    assert "cannot reach" in msg
    assert "ConnectTimeout" in msg
    assert s.calls == 2
    # SystemExit is not an IOError / Exception: the adapters' per-granule
    # handlers cannot swallow it
    assert not isinstance(ei.value, Exception)
    assert not (tmp_path / "g.nc").exists()


def test_a_read_timeout_also_exits_the_lane(tmp_path):
    s = _Session(exc=rqe.ReadTimeout("read timed out"))
    with pytest.raises(SystemExit) as ei:
        cm.earthdata_download(s, URL, str(tmp_path / "g.nc"),
                              attempts=2, sleep=0.0)
    assert "cannot reach" in str(ei.value.code)


def test_the_default_timeout_connects_in_30_s_and_reads_in_300(tmp_path):
    s = _Session(status=404)
    cm.earthdata_download(s, URL, str(tmp_path / "g.nc"))
    assert s.kw["timeout"] == (30, 300)


def test_an_http_500_is_still_an_ioerror_about_one_file(tmp_path):
    s = _Session(status=500)
    with pytest.raises(IOError) as ei:
        cm.earthdata_download(s, URL, str(tmp_path / "g.nc"),
                              attempts=2, sleep=0.0)
    assert "HTTP 500" in str(ei.value)
    assert s.calls == 2


def test_a_404_is_a_counted_absence_without_a_retry(tmp_path):
    s = _Session(status=404)
    got = cm.earthdata_download(s, URL, str(tmp_path / "g.nc"),
                                attempts=6, sleep=0.0)
    assert got == (None, "notfound")
    assert s.calls == 1
