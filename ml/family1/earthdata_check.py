"""The Earthdata Login check — does THIS account reach each NASA archive?

PLAIN ENGLISH. Most of family 1's credentialed stores come from three NASA
archives, and a NASA account is not enough on its own: each archive is an
"application" the account must have approved once in Earthdata Login, and
some collections also need a licence (EULA) accepted. A build that finds this
out an hour in has wasted the hour. This check makes ONE small authenticated
request to each archive and reports exactly what came back:

  lp_daac   LP DAAC's cloud archive — a MOD11C1 granule's `.cmr.xml`
            (found by a CMR search, a few kB), a ranged GET.
  ges_disc  GES DISC — the smallest-looking MERGIR `.nc4` of a day listed in
            the archive's own directory page, a ranged GET of its first KB.
  podaac    PO.DAAC / Earthdata Cloud — a CMR granule search for MUR SST,
            then one authenticated HEAD on the granule URL (and, when the
            cloud's signed S3 hop refuses a HEAD, a ranged GET of 1 KB).

For each: the HTTP status, the final host, whether the answer was an
Earthdata Login approval page or a EULA page, the bytes received, and — when
it needs a human — THE EXACT URL to open (the app-approval link carries the
archive's client_id, read from the redirect it sent).

Verified 2026-09-17 WITHOUT credentials (listings and redirects only): LP
DAAC's old `e4ftl01.cr.usgs.gov/MOLT/` tree answers 404 (moved to
`data.lpdaac.earthdatacloud.nasa.gov`); a protected LP `.cmr.xml` answers
302 to `urs.earthdata.nasa.gov/oauth/authorize?client_id=FtSFfbOeuxDcdf4px-
elGw...`; a GES DISC `.nc4` answers 302 to URS with client_id
`e2WVk8Pw6weeLUKZYOxvTQ`, and GES DISC's directory page links
`https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ`;
CMR returns `archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/`
links for MUR.

EXIT CODE (ml/CLAUDE.md §5.17 — only a definite answer is fatal):
  0  every archive answered with data, OR something failed that is not a
     verdict about the account (a timeout, a 5xx, an archive that moved)
  1  the credentials are not set, or an archive DEFINITELY refused: bad
     username/password, an application not approved, a EULA not accepted, a
     401/403 from Earthdata Login.

Credentials come from EARTHDATA_USERNAME / EARTHDATA_PASSWORD in the
environment and nowhere else; they are never printed. Run on a GitHub-hosted
runner only (`.github/workflows/family1-build.yml`, check_credentials=true).
"""
import json
import os
import re
import socket
import sys
import time
import urllib.parse

URS_HOST = "urs.earthdata.nasa.gov"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
LP_SHORT, LP_VERSION = "MOD11C1", "061"
GES_BASE = "https://disc2.gesdisc.eosdis.nasa.gov/data/MERGED_IR/GPM_MERGIR.1/"
GES_DAY = "2020/001/"
# The host the irtb / xco2 adapters READ (run #259/#260, 2026-09-19: HTTP 401
# after 2 redirects while disc2.gesdisc answered 206 in run #90) — a second
# GES DISC target so the check answers for the archive that refused.
GES_DATA_BASE = "https://data.gesdisc.earthdata.nasa.gov/data/MERGED_IR/GPM_MERGIR.1/"
GES_DATA_CLIENT = "e2WVk8Pw6weeLUKZYOxvTQ"   # "NASA GESDISC DATA ARCHIVE" in URS
PODAAC_SHORT = "MUR-JPL-L4-GLOB-v4.1"
TIMEOUT = 60
RANGE = "bytes=0-1023"
UA = "earth-science-pipeline/1.0 (research; github blauewelt/earth)"

APPROVE_RE = re.compile(
    r"approve[_ ]?app|authorize (this )?application|has not (yet )?been "
    r"(authorized|approved)|unauthorized application|application.{0,40}"
    r"(not|n't) (been )?(authorized|approved)|requires? (your )?"
    r"(approval|authorization)", re.I)
EULA_RE = re.compile(r"\beula\b|end[- ]user licen[cs]e agreement|accept.{0,30}"
                     r"licen[cs]e", re.I)
BADCRED_RE = re.compile(r"invalid (username|user name|credentials|password)|"
                        r"incorrect (username|password)|bad credentials|"
                        r"login failed", re.I)
CLIENT_RE = re.compile(r"client_id=([A-Za-z0-9_\-]+)")
RESOLUTION_RE = re.compile(r'"resolution_url"\s*:\s*"([^"]+)"')


# ============================================================ classify ====
def classify(status, final_url, body, history=(), content_type=""):
    """What a response means for the account. Pure — the tests drive it.

    Returns {"verdict", "definite", "approval_url", "why"}. `history` is the
    list of URLs the request passed through (redirects), so the client_id of
    the archive's Earthdata application can be read even when the final page
    does not repeat it.
    """
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) \
        else str(body or "")
    host = urllib.parse.urlparse(final_url or "").hostname or ""
    trail = " ".join(list(history) + [final_url or ""])
    client = None
    for u in list(history) + [final_url or ""]:
        m = CLIENT_RE.search(u or "")
        if m and URS_HOST in (u or ""):
            client = m.group(1)
    m = CLIENT_RE.search(text) if "approve_app" in text else None
    if m and not client:
        client = m.group(1)
    approve_url = (f"https://{URS_HOST}/approve_app?client_id={client}"
                   if client else None)
    html = "html" in (content_type or "").lower() or \
        text.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
    at_urs = host == URS_HOST
    # a licence PAGE, not a data file that happens to mention a licence
    not_data = html or status >= 400 or at_urs
    if (not_data and EULA_RE.search(text)) or \
            "eula" in (final_url or "").lower():
        m = RESOLUTION_RE.search(text)
        return {"verdict": "needs_eula", "definite": True,
                "approval_url": (m.group(1) if m else final_url),
                "why": "the archive answered with a licence (EULA) that "
                       "this account has not accepted"}
    if status in (401, 403) and BADCRED_RE.search(text):
        return {"verdict": "bad_credentials", "definite": True,
                "approval_url": None,
                "why": "Earthdata Login rejected the username or password"}
    if (at_urs or status in (401, 403)) and (
            APPROVE_RE.search(text) or "approve_app" in (final_url or "")):
        return {"verdict": "needs_approval", "definite": True,
                "approval_url": approve_url or final_url,
                "why": "the archive's Earthdata application is not approved "
                       "for this account"}
    if at_urs and status == 200 and html:
        # a login or approval FORM, not data — Earthdata Login stopped us
        return {"verdict": "needs_approval", "definite": True,
                "approval_url": approve_url or final_url,
                "why": "the request ended on an Earthdata Login page "
                       "instead of the file"}
    if at_urs and status in (401, 403):
        return {"verdict": "refused", "definite": True,
                "approval_url": approve_url,
                "why": f"Earthdata Login answered HTTP {status}"}
    if status in (200, 206) and not at_urs:
        return {"verdict": "ok", "definite": False, "approval_url": None,
                "why": f"HTTP {status} from {host}"}
    if status in (401, 403) and URS_HOST in trail:
        return {"verdict": "refused", "definite": True,
                "approval_url": approve_url,
                "why": f"HTTP {status} from {host} after an Earthdata Login "
                       f"hop — the account is not authorised for this data"}
    return {"verdict": "error", "definite": False, "approval_url": approve_url,
            "why": f"HTTP {status} from {host or '?'} — not a verdict about "
                   f"the account"}


# ============================================================== IPv4 =====
# GITHUB'S UBUNTU RUNNERS RESOLVE AAAA RECORDS AND HAVE NO IPv6 ROUTE.
# Measured 2026-09-17 (family1-build run #3): LP DAAC and GES DISC both came
# back `[Errno 101] Network is unreachable` with no HTTP status at all, before
# any login — which reads exactly like an archive refusing the account and is
# nothing of the kind. urs.earthdata.nasa.gov and disc2.gesdisc.eosdis.nasa.gov
# publish AAAA records; the runner picks one and the connection never leaves
# the machine. So every lookup this check makes is restricted to A records.
_REAL_GETADDRINFO = socket.getaddrinfo


def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """`socket.getaddrinfo` with the IPv6 answers dropped."""
    res = _REAL_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)
    return [r for r in res if r[0] == socket.AF_INET]


def force_ipv4(on=True):
    """Make every socket in this process resolve to IPv4 only (idempotent)."""
    socket.getaddrinfo = ipv4_getaddrinfo if on else _REAL_GETADDRINFO
    return socket.getaddrinfo


# ============================================================= session ====
def session(user, password):
    """A requests session that sends Basic auth to EARTHDATA LOGIN ONLY.

    NASA's documented recipe keeps the Authorization header across every hop
    that touches urs.earthdata.nasa.gov, which also hands the password to the
    archive on the way back. Every archive checked here redirects an
    unauthenticated request to Earthdata Login (verified 2026-09-17), so the
    header is attached on the hop INTO the login host and removed on every
    other hop — the password never reaches an archive or a signed S3 URL.
    """
    import requests

    creds = (user, password)

    class _S(requests.Session):
        def rebuild_auth(self, prepared, response):
            prepared.headers.pop("Authorization", None)
            host = urllib.parse.urlparse(prepared.url).hostname
            if host == URS_HOST:
                prepared.prepare_auth(creds)

    s = _S()
    s.trust_env = False            # never let a stray .netrc or proxy decide
    s.headers["User-Agent"] = UA
    return s


def _request(s, method, url, range_bytes=True):
    headers = {"Range": RANGE} if range_bytes and method == "GET" else {}
    t0 = time.time()
    r = s.request(method, url, headers=headers, allow_redirects=True,
                  timeout=TIMEOUT, stream=True)
    body = b""
    if method == "GET":
        for chunk in r.iter_content(4096):
            body += chunk
            if len(body) >= 8192:
                break
    r.close()
    hist = [h.url for h in r.history] + \
        [h.headers.get("Location", "") for h in r.history]
    out = {"method": method, "url": url, "status": r.status_code,
           "final_url": _strip_query(r.url),
           "final_host": urllib.parse.urlparse(r.url).hostname,
           "redirects": len(r.history),
           "via_urs": any(URS_HOST in (u or "") for u in hist),
           "bytes": len(body),
           "content_type": r.headers.get("Content-Type", ""),
           "seconds": round(time.time() - t0, 2)}
    out.update(classify(r.status_code, r.url, body, hist,
                        r.headers.get("Content-Type", "")))
    if out["approval_url"]:
        out["approval_url"] = _strip_query(out["approval_url"],
                                           keep=("client_id",))
    return out


def _strip_query(u, keep=()):
    """A URL fit to print: signed S3 query strings and OAuth codes removed."""
    p = urllib.parse.urlparse(u or "")
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query) if k in keep]
    return urllib.parse.urlunparse(p._replace(
        query=urllib.parse.urlencode(q)))


def _cmr_first(s, **params):
    import requests
    r = requests.get(CMR, params={**params, "page_size": 1,
                                  "sort_key": "-start_date"},
                     timeout=TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    entries = r.json().get("feed", {}).get("entry", [])
    if not entries:
        raise LookupError(f"CMR has no granule for {params}")
    return entries[0]


def _link(entry, rel_suffix, pred):
    for ln in entry.get("links", []):
        if str(ln.get("rel", "")).endswith(rel_suffix) and \
                str(ln.get("href", "")).startswith("https") and \
                pred(ln["href"]):
            return ln["href"]
    return None


# ============================================================== targets ===
def check_lp_daac(s):
    e = _cmr_first(s, short_name=LP_SHORT, version=LP_VERSION)
    url = _link(e, "metadata#", lambda h: "protected" in h and
                h.endswith(".cmr.xml")) or \
        _link(e, "data#", lambda h: "protected" in h)
    if not url:
        raise LookupError(f"no protected https link in the {LP_SHORT} granule "
                          f"{e.get('title')}")
    return {"granule": e.get("title"), **_request(s, "GET", url)}


def check_ges_disc(s):
    import requests
    page = requests.get(GES_BASE + GES_DAY, timeout=TIMEOUT,
                        headers={"User-Agent": UA})
    page.raise_for_status()
    names = sorted(set(re.findall(r'href="(merg_[^"]+\.nc4)"', page.text)))
    if not names:
        raise LookupError(f"no .nc4 in the listing {GES_BASE + GES_DAY}")
    m = CLIENT_RE.search(page.text) if "approve_app" in page.text else None
    out = {"listing": GES_BASE + GES_DAY, "files_listed": len(names),
           **_request(s, "GET", GES_BASE + GES_DAY + names[0])}
    if m and out["verdict"] != "ok" and not out.get("approval_url"):
        out["approval_url"] = (f"https://{URS_HOST}/approve_app?client_id="
                               f"{m.group(1)}")
    return out


def check_ges_disc_data(s):
    """Same probe against data.gesdisc.earthdata.nasa.gov (what irtb reads)."""
    import requests
    # Even the DIRECTORY LISTING sits behind Earthdata Login on this host, and
    # an unapproved application answers HTTP 401 straight from URS
    # (`app_type=401` in the redirect) — measured on run #273, 2026-09-19.
    # That is a definite verdict about the ACCOUNT, so it carries the
    # approval URL rather than surfacing as an "error" nobody can act on.
    page = s.get(GES_DATA_BASE + GES_DAY, timeout=TIMEOUT,
                 headers={"User-Agent": UA})
    if page.status_code == 401 and URS_HOST in page.url:
        client = (CLIENT_RE.search(page.url) or CLIENT_RE.search(page.text))
        cid = client.group(1) if client else GES_DATA_CLIENT
        return {"listing": GES_DATA_BASE + GES_DAY, "status": 401,
                "final_host": URS_HOST, "via_urs": True, "definite": True,
                "verdict": "needs_approval",
                "approval_url": f"https://{URS_HOST}/approve_app?client_id={cid}",
                "why": ("Earthdata Login answered 401 for the GES DISC "
                        "application — this account has not approved it")}
    page.raise_for_status()
    names = sorted(set(re.findall(r'href="(merg_[^"]+\.nc4)"', page.text)))
    if not names:
        raise LookupError(f"no .nc4 in the listing {GES_DATA_BASE + GES_DAY}")
    m = CLIENT_RE.search(page.text) if "approve_app" in page.text else None
    out = {"listing": GES_DATA_BASE + GES_DAY, "files_listed": len(names),
           **_request(s, "GET", GES_DATA_BASE + GES_DAY + names[0])}
    if m and out["verdict"] != "ok" and not out.get("approval_url"):
        out["approval_url"] = (f"https://{URS_HOST}/approve_app?client_id="
                               f"{m.group(1)}")
    return out


def check_podaac(s):
    """One KB of a PROTECTED PO.DAAC granule, BY GET AND THROUGH URS.

    A HEAD on `archive.podaac.earthdata.nasa.gov/...-protected/...` answered
    HTTP 200 from CloudFront with `via_urs: false` (run #3, 2026-09-17) — the
    bytes were never asked for and Earthdata Login was never in the path, so
    the account was not tested at all. The check therefore GETs the first
    kilobyte (Earthdata Cloud's last hop is an S3 URL signed FOR GET), and
    `ok` additionally requires that the request passed through
    urs.earthdata.nasa.gov: an answer that skipped the login says nothing
    about these credentials.
    """
    e = _cmr_first(s, short_name=PODAAC_SHORT)
    url = _link(e, "data#", lambda h: "podaac" in h and "protected" in h)
    if not url:
        raise LookupError(f"no protected PO.DAAC link in {e.get('title')}")
    out = {"granule": e.get("title"),
           "cmr_granule_size_mb": e.get("granule_size"),
           **_request(s, "GET", url)}
    return require_urs(out, "podaac")


def require_urs(out, name):
    """An `ok` that never touched Earthdata Login is NOT a verdict.

    It may mean the archive serves this object anonymously, or that a cached
    hop answered; either way the credentials were not exercised. Reported as
    `ok_without_login` — not definite, so it never fails the job, and never
    counted as a PASS either.
    """
    if out.get("verdict") == "ok" and not out.get("via_urs"):
        out["verdict"] = "ok_without_login"
        out["definite"] = False
        out["why"] = (f"HTTP {out.get('status')} from "
                      f"{out.get('final_host')} WITHOUT passing through "
                      f"{URS_HOST} — the bytes arrived but this account was "
                      f"never checked, so {name} is untested")
    return out


TARGETS = (("lp_daac", check_lp_daac), ("ges_disc", check_ges_disc),
           ("ges_disc_data", check_ges_disc_data), ("podaac", check_podaac))


def run(user, password, targets=None, ipv4=True):
    if ipv4:
        force_ipv4()
    s = session(user, password)
    targets = TARGETS if targets is None else targets
    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "archives": {}}
    for name, fn in targets:
        try:
            report["archives"][name] = fn(s)
        except Exception as e:                              # noqa: BLE001
            report["archives"][name] = {
                "verdict": "error", "definite": False,
                "why": f"{type(e).__name__}: {str(e)[:300]}"}
    bad = {k: v for k, v in report["archives"].items() if v.get("definite")
           and v.get("verdict") != "ok"}
    report["refused"] = sorted(bad)
    report["ok"] = sorted(k for k, v in report["archives"].items()
                          if v.get("verdict") == "ok")
    report["untested"] = sorted(k for k, v in report["archives"].items()
                                if v.get("verdict") == "ok_without_login")
    report["ipv4_only"] = bool(ipv4)
    return report


def print_report(report):
    print("EARTHDATA LOGIN CHECK")
    for name, r in report["archives"].items():
        print(f"  {name:9s} {r.get('verdict', '?'):16s} HTTP "
              f"{r.get('status', '-')} from {r.get('final_host', '-')}, "
              f"{r.get('bytes', 0)} byte(s), {r.get('redirects', 0)} "
              f"redirect(s){' via URS' if r.get('via_urs') else ''} — "
              f"{r.get('why', '')}")
        if r.get("url"):
            print(f"            requested {r['url']}")
        if r.get("verdict") not in ("ok", None) and r.get("approval_url"):
            print(f"::error::{name}: open {r['approval_url']} while logged "
                  f"in to Earthdata and approve / accept, then re-run")
    print(json.dumps(report, indent=1, default=str))


def main(out=None, env=None):
    env = os.environ if env is None else env
    user = env.get("EARTHDATA_USERNAME", "")
    password = env.get("EARTHDATA_PASSWORD", "")
    if not user or not password:
        missing = [k for k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
                   if not env.get(k)]
        print(f"::error::--check-credentials: {', '.join(missing)} not set in "
              f"the environment — nothing was checked. They reach GitHub-"
              f"HOSTED runners only (ml/CLAUDE.md §6).")
        return 1
    report = run(user, password)
    print_report(report)
    if out:
        with open(out, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    if report["refused"]:
        print(f"::error::definite refusal from {report['refused']} — see the "
              f"approval URL(s) above")
        return 1
    if report.get("untested"):
        print(f"::warning::{report['untested']} answered without passing "
              f"through Earthdata Login — the bytes arrived, the ACCOUNT was "
              f"not tested")
    errs = [k for k, v in report["archives"].items()
            if v.get("verdict") == "error"]
    if errs:
        print(f"::warning::{errs} could not be checked (not a verdict about "
              f"the account) — re-run later")
    return 0


if __name__ == "__main__":
    sys.exit(main())
