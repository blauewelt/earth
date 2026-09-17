#!/usr/bin/env python3
"""The Earthdata Login check (ml/family1/earthdata_check.py) — no network.

    python3 -m pytest -q tests/test_family1_earthdata_check.py

`classify` is driven with the response shapes Earthdata Login and the
archives produce (an approval page, a EULA refusal, bad credentials, a data
file that mentions a licence, a signed-S3 403). Then the whole request path —
the session that attaches Basic auth on the hop INTO Earthdata Login and
nowhere else — runs against two local servers standing in for an archive and
for Earthdata Login (`localhost` plays urs.earthdata.nasa.gov, `127.0.0.1`
the archive), for an approved account, an unapproved one and a wrong
password. The CLI refuses when the credentials are unset.
"""
import base64
import http.server
import json
import os
import sys
import threading
import urllib.parse

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family1_stores as b1                               # noqa: E402
from family1 import earthdata_check as edc                      # noqa: E402

URS = "https://urs.earthdata.nasa.gov"
AUTH = (f"{URS}/oauth/authorize?client_id=FtSFfbOeuxDcdf4px-elGw&response_"
        f"type=code&redirect_uri=https://data.lpdaac.earthdatacloud.nasa.gov"
        f"/login&state=%2Flp&app_type=401")


def test_classify_reads_the_answers_earthdata_gives():
    data = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/x"
    ok = edc.classify(206, data, b"\x89HDF...", [data, AUTH])
    assert ok["verdict"] == "ok" and not ok["definite"]
    # an application nobody approved: URS answers with its approval page
    ap = edc.classify(200, AUTH, b"<!DOCTYPE html><html>Approve application "
                      b"LP DAAC Data Pool</html>", [data], "text/html")
    assert ap["verdict"] == "needs_approval" and ap["definite"]
    assert ap["approval_url"] == (f"{URS}/approve_app?client_id="
                                  f"FtSFfbOeuxDcdf4px-elGw")
    # ... or a 401 saying so
    ap2 = edc.classify(401, AUTH, b"The application has not yet been "
                       b"authorized by the user", [data])
    assert ap2["verdict"] == "needs_approval"
    # a EULA refusal from Earthdata Cloud, with its resolution link
    eula = edc.classify(403, data, json.dumps({
        "error_description": "This data requires accepting the EULA",
        "resolution_url": f"{URS}/approve_app?client_id=Q&eula=true"})
        .encode(), [AUTH])
    assert eula["verdict"] == "needs_eula" and eula["definite"]
    assert eula["approval_url"].endswith("eula=true")
    bad = edc.classify(401, AUTH, b"Invalid username or password", [data])
    assert bad["verdict"] == "bad_credentials" and bad["definite"]
    # a DATA file that mentions a licence is data
    xml = edc.classify(206, data, b"<?xml version='1.0'?><Granule>Users must "
                       b"accept the license terms of NASA</Granule>", [AUTH],
                       "application/xml")
    assert xml["verdict"] == "ok"
    # not verdicts about the account
    for st in (500, 502, 404):
        r = edc.classify(st, data, b"oops", [])
        assert r["verdict"] == "error" and not r["definite"]
    s3 = edc.classify(403, "https://bucket.s3.us-west-2.amazonaws.com/x",
                      b"<Error>SignatureDoesNotMatch</Error>", [])
    assert s3["verdict"] == "error" and not s3["definite"]


# ======================================================== a fake login ====
USER, PASSWORD = "alice", "s3cret"


def make_servers(approved=True):
    state = {"auth_to_data": 0}

    class URSHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            want = "Basic " + base64.b64encode(
                f"{USER}:{PASSWORD}".encode()).decode()
            if self.headers.get("Authorization") != want:
                body = b"Invalid username or password"
                self.send_response(401)
            elif not approved:
                body = (b"<!DOCTYPE html><html>Approve application "
                        b"<a href='/approve_app?client_id=CID'>yes</a></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
            else:
                self.send_response(302)
                self.send_header("Location",
                                 q["redirect_uri"][0] + "?code=abc")
                body = b""
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class DataHandler(http.server.BaseHTTPRequestHandler):
        urs_port = None

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.headers.get("Authorization"):
                state["auth_to_data"] += 1
            if self.path.startswith("/login"):
                self.send_response(302)
                self.send_header("Set-Cookie", "session=ok; Path=/")
                self.send_header("Location", "/file.nc")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if "session=ok" not in (self.headers.get("Cookie") or ""):
                self.send_response(302)
                me = f"http://127.0.0.1:{self.server.server_address[1]}"
                self.send_header("Location", (
                    f"http://localhost:{self.urs_port}/oauth/authorize?"
                    f"client_id=CID&response_type=code&redirect_uri="
                    f"{me}/login"))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = b"\x89HDF" + bytes(1020)
            self.send_response(206)
            self.send_header("Content-Range", "bytes 0-1023/99999")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    urs = http.server.ThreadingHTTPServer(("127.0.0.1", 0), URSHandler)
    DataHandler.urs_port = urs.server_address[1]
    data = http.server.ThreadingHTTPServer(("127.0.0.1", 0), DataHandler)
    for s in (urs, data):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    return urs, data, state


@pytest.mark.parametrize("approved,password,verdict,code", [
    (True, PASSWORD, "ok", 0),
    (False, PASSWORD, "needs_approval", 1),
    (True, "wrong", "bad_credentials", 1),
])
def test_the_whole_request_path(monkeypatch, capsys, approved, password,
                                verdict, code):
    pytest.importorskip("requests")
    monkeypatch.setattr(edc, "URS_HOST", "localhost")
    urs, data, state = make_servers(approved)
    try:
        url = f"http://127.0.0.1:{data.server_address[1]}/file.nc"
        monkeypatch.setattr(edc, "TARGETS", (
            ("fake", lambda s: edc._request(s, "GET", url)),
            ("broken", lambda s: (_ for _ in ()).throw(
                ConnectionError("no route")))))
        rc = edc.main(env={"EARTHDATA_USERNAME": USER,
                           "EARTHDATA_PASSWORD": password})
        out = capsys.readouterr().out
        assert rc == code
        rep, _ = json.JSONDecoder().raw_decode(out[out.index("{"):])
        r = rep["archives"]["fake"]
        assert r["verdict"] == verdict
        assert rep["archives"]["broken"]["verdict"] == "error"
        assert r["via_urs"] is True
        if verdict == "ok":
            assert r["status"] == 206 and r["bytes"] == 1024
            assert r["final_host"] == "127.0.0.1"
            # Basic auth went to the login host, never to the archive
            assert state["auth_to_data"] == 0
        if verdict == "needs_approval":
            assert "approve_app?client_id=CID" in out
            assert "::error::fake: open" in out
        assert PASSWORD not in out and "wrong" not in out
        assert state["auth_to_data"] == 0
        assert "code=abc" not in out
    finally:
        urs.shutdown()
        data.shutdown()


def test_unset_credentials_refuse_and_the_cli_needs_no_store(monkeypatch,
                                                             capsys):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.setenv("EARTHDATA_PASSWORD", "pw-DISTINCT-7")
    assert b1.main(["--check-credentials"]) == 1
    out = capsys.readouterr().out
    assert "EARTHDATA_USERNAME" in out and "pw-DISTINCT-7" not in out
    assert "EARTHDATA_PASSWORD" not in out
    with pytest.raises(SystemExit, match="--store is required"):
        b1.main([])


def test_the_workflow_runs_the_check_on_hosted_runners_only():
    import yaml
    d = yaml.safe_load(open(os.path.join(
        ROOT, ".github", "workflows", "family1-build.yml")))
    trig = d.get("on", d.get(True))
    inputs = trig["workflow_dispatch"]["inputs"]
    assert inputs["check_credentials"]["type"] == "boolean"
    assert inputs["check_credentials"]["default"] is False
    assert len(inputs) < 25
    steps = d["jobs"]["build"]["steps"]
    names = [s.get("name") for s in steps]
    chk = steps[names.index("Earthdata Login check (hosted runner only)")]
    assert chk["if"] == "${{ github.event.inputs.check_credentials == 'true' }}"
    assert "--check-credentials" in chk["run"]
    for k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        assert "github.event.inputs.runner == 'ubuntu-latest'" in \
            chk["env"][k]
    for n in ("Preflight", "Build or probe"):
        st = steps[names.index(n)]
        assert "check_credentials != 'true'" in st["if"]
    inst = steps[names.index("Install")]
    for pkg in ("zstandard", "netCDF4", "requests"):
        assert pkg in inst["run"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
