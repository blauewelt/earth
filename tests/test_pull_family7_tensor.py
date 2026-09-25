#!/usr/bin/env python3
"""E-083 §3.1 · `ml/pull_family7_tensor.py`: parallel, resumable, verified.

No network: every request goes to a FAKE session that serves a byte blob by
`Range:` the way the Hub does (206 + Content-Range), and can be told to fail.
What is pinned:

  * a pull reassembles the exact bytes, every request carries a Range and the
    env token, and nothing is left behind but the file;
  * an interrupted pull RESUMES — the second run asks only for the chunks the
    first did not record, and a chunk is recorded only after its bytes landed;
  * 429/5xx and a short body are retried; a 200 (range ignored) is refused
    and never written at an offset;
  * a sha256 mismatch deletes the partial file and refuses;
  * the index and the Hub manifest must agree on every group's sha256;
  * the token is never an argument.

    python3 -m pytest -q tests/test_pull_family7_tensor.py
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

import pull_family7_tensor as PT                                   # noqa: E402

TOKEN = "hf_fake_token_for_tests"


class FakeResp:
    def __init__(self, status, headers=None, body=b"", cut=None, js=None):
        self.status_code = status
        self.headers = headers or {}
        self.body = body
        self.cut = cut
        self._js = js

    def iter_content(self, n):
        if self.cut is not None:            # some bytes, then a reset
            yield self.body[:self.cut]
            raise ConnectionError("connection reset mid-body")
        for pos in range(0, len(self.body), n):
            yield self.body[pos:pos + n]

    def json(self):
        return self._js

    def close(self):
        pass


class FakeSession:
    """Serves `files` {url: bytes} by Range. `script` maps a (url, start) to a
    list of behaviours consumed in order: 'crash' (fatal), 503, 'short'."""

    def __init__(self, files, script=None, ignore_range=False, json_urls=None):
        self.files = files
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.ignore_range = ignore_range
        self.json_urls = json_urls or {}
        self.calls = []

    def get(self, url, headers=None, stream=False, timeout=None,
            allow_redirects=True):
        headers = dict(headers or {})
        self.calls.append((url, headers))
        if url in self.json_urls:
            return FakeResp(200, js=self.json_urls[url])
        blob = self.files.get(url)
        if blob is None:
            return FakeResp(404)
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", headers.get("Range", ""))
        if self.ignore_range or not m:
            return FakeResp(200, {}, blob)
        a, b = int(m.group(1)), int(m.group(2))
        todo = self.script.get((url, a))
        if todo:
            what = todo.pop(0)
            if what == "crash":
                raise PT.PullError("simulated crash")
            if what == "short":
                return FakeResp(206, {"Content-Range":
                                      f"bytes {a}-{b}/{len(blob)}"},
                                blob[a:b + 1], cut=1)
            return FakeResp(int(what))
        return FakeResp(206, {"Content-Range": f"bytes {a}-{b}/{len(blob)}"},
                        blob[a:b + 1])


def blob(n=10_007, seed=0):
    return np.random.default_rng(seed).integers(0, 256, n, np.uint8).tobytes()


def sha(b):
    return hashlib.sha256(b).hexdigest()


URL = "https://huggingface.co/datasets/x/resolve/main/tensors/s/f.npy"


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)


def test_pull_reassembles_the_bytes_with_range_and_token(tmp_path):
    data = blob()
    s = FakeSession({URL: data})
    dest = str(tmp_path / "f.npy")
    n, _ = PT.pull_file(s, URL, dest, sha(data), workers=4, chunk=1000,
                        sleep=lambda t: None)
    assert open(dest, "rb").read() == data and n == len(data)
    assert sorted(os.listdir(tmp_path)) == ["f.npy"], "part files left behind"
    ranges = [h["Range"] for _, h in s.calls]
    assert all(h.get("Authorization") == f"Bearer {TOKEN}" for _, h in s.calls)
    assert ranges[0] == "bytes=0-0"                      # the size probe
    assert sorted(ranges[1:]) == sorted(
        f"bytes={a}-{min(a + 1000, len(data)) - 1}"
        for a in range(0, len(data), 1000))


def test_an_interrupted_pull_resumes_where_it_stopped(tmp_path):
    data = blob()
    dest = str(tmp_path / "f.npy")
    crash = FakeSession({URL: data}, script={(URL, 6000): ["crash"]})
    with pytest.raises(PT.PullError, match="simulated crash"):
        PT.pull_file(crash, URL, dest, sha(data), workers=1, chunk=1000,
                     sleep=lambda t: None)
    done = PT.read_done(dest + ".part.done")
    assert done == set(range(6)), done           # 0..5 landed, 6 did not
    assert not os.path.exists(dest)
    s = FakeSession({URL: data})
    n, _ = PT.pull_file(s, URL, dest, sha(data), workers=3, chunk=1000,
                        sleep=lambda t: None)
    asked = sorted(int(h["Range"][6:].split("-")[0]) for _, h in s.calls[1:])
    assert asked == list(range(6000, len(data), 1000)), asked
    assert n == len(data) - 6000
    assert open(dest, "rb").read() == data


def test_a_changed_chunk_size_starts_over_rather_than_misreading(tmp_path):
    data = blob()
    dest = str(tmp_path / "f.npy")
    crash = FakeSession({URL: data}, script={(URL, 4000): ["crash"]})
    with pytest.raises(PT.PullError):
        PT.pull_file(crash, URL, dest, sha(data), workers=1, chunk=1000,
                     sleep=lambda t: None)
    s = FakeSession({URL: data})
    n, _ = PT.pull_file(s, URL, dest, sha(data), workers=2, chunk=3000,
                        sleep=lambda t: None)
    assert n == len(data) and open(dest, "rb").read() == data


def test_transient_errors_are_retried(tmp_path):
    data = blob()
    s = FakeSession({URL: data}, script={(URL, 2000): [503, 429, "short"],
                                         (URL, 9000): [502]})
    naps = []
    dest = str(tmp_path / "f.npy")
    PT.pull_file(s, URL, dest, sha(data), workers=2, chunk=1000,
                 sleep=naps.append)
    assert open(dest, "rb").read() == data
    assert len(naps) == 4


def test_a_host_that_ignores_range_is_refused(tmp_path):
    data = blob()
    s = FakeSession({URL: data}, ignore_range=True)
    dest = str(tmp_path / "f.npy")
    with pytest.raises(PT.PullError, match="206"):
        PT.pull_file(s, URL, dest, sha(data), workers=2, chunk=1000,
                     sleep=lambda t: None)
    assert not os.path.exists(dest)
    # and a 200 on a chunk (after a good size probe) is refused, not written
    s2 = FakeSession({URL: data})
    s2.ignore_range = False
    fd = os.open(str(tmp_path / "x"), os.O_RDWR | os.O_CREAT)
    try:
        s2.ignore_range = True
        with pytest.raises(PT.PullError, match="want 206"):
            PT.fetch_chunk(s2, URL, fd, 0, 999, sleep=lambda t: None)
        assert os.fstat(fd).st_size == 0
    finally:
        os.close(fd)


def test_a_wrong_sha256_deletes_the_part_and_refuses(tmp_path):
    data = blob()
    s = FakeSession({URL: data})
    dest = str(tmp_path / "f.npy")
    with pytest.raises(PT.PullError, match="sha256"):
        PT.pull_file(s, URL, dest, "0" * 64, workers=2, chunk=1000,
                     sleep=lambda t: None)
    assert os.listdir(tmp_path) == []


def test_a_finished_file_is_left_alone(tmp_path):
    data = blob()
    dest = str(tmp_path / "f.npy")
    open(dest, "wb").write(data)
    s = FakeSession({URL: data})
    assert PT.pull_file(s, URL, dest, sha(data))[0] == 0
    assert s.calls == []
    with pytest.raises(PT.PullError, match="does not match"):
        PT.pull_file(s, URL, dest, "1" * 64)


def _index(files):
    return dict(stem="st", groups={
        g: dict(file=f"st_X_{g}.npy", sha256=sha(b)) for g, b in files.items()})


def test_plan_takes_the_npz_sha_from_the_manifest_and_cross_checks(tmp_path):
    files = {"g100": b"a" * 50, "rg100": b"b" * 70}
    idx = _index(files)
    man = {"files": [{"name": "st.npz", "sha256": "n" * 64},
                     {"name": "st_X_g100.npy", "sha256": sha(files["g100"])}]}
    got = PT.plan(idx, [], man)
    assert got[0] == ("st.npz", "n" * 64)
    assert [n for n, _ in got[1:]] == ["st_X_g100.npy", "st_X_rg100.npy"]
    man["files"][1]["sha256"] = "f" * 64
    with pytest.raises(PT.PullError, match="different bytes"):
        PT.plan(idx, [], man)
    with pytest.raises(PT.PullError, match="not in the index"):
        PT.plan(idx, ["oc025"], None)


def test_main_end_to_end_on_a_fake_hub(tmp_path):
    files = {"g100": blob(3001, 1), "rg100": blob(2500, 2)}
    npz = blob(900, 3)
    idx = _index(files)
    ip = str(tmp_path / "idx.json")
    json.dump(idx, open(ip, "w"))
    base = PT.resolve_base("st")
    served = {base + f"st_X_{g}.npy": b for g, b in files.items()}
    served[base + "st.npz"] = npz
    man = {"files": [{"name": "st.npz", "sha256": sha(npz)}]}
    s = FakeSession(served, json_urls={base + "manifest.json": man})
    dest = str(tmp_path / "d")
    assert PT.main(["--dest", dest, "--index", ip, "--workers", "3",
                    "--chunk-mb", "1"], session=s) == 0
    for g, b in files.items():
        assert open(os.path.join(dest, f"st_X_{g}.npy"), "rb").read() == b
    assert open(os.path.join(dest, "st.npz"), "rb").read() == npz
    assert all(h.get("Authorization") == f"Bearer {TOKEN}" for _, h in s.calls)


def test_the_token_is_never_an_argument():
    with pytest.raises(SystemExit):
        PT.main(["--dest", "/tmp/x", "--token", "abc"])
    src = open(os.path.join(ML, "pull_family7_tensor.py")).read()
    assert "--token" not in src and 'os.environ.get("HF_TOKEN"' in src


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
