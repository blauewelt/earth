#!/usr/bin/env python3
"""Create / list / inspect / delete the Cloud TPU v5e VM that runs the JAX
port's first measurement, and stage multi-GB tensors into the GCS bucket that
makes deleting it cheap — driven from a session over the Cloud TPU REST API,
the same way `scripts/gpu_box.mjs` drives the Vast fleet.

WHY THIS EXISTS, and why it is deliberately small. `ml/plans/TPU_ACCESS.md`
§6-7 asks for exactly three things: **one v5e-8, one smoke run, one
measurement.** The question is whether the per-batch host gather
(`LazyPixels`/`gather_px`, `ml/plans/JAX_PORT.md` §1b) can feed a TPU, because
if it cannot the chips idle at full price and the whole TPU bet is off. That
is a measurement, not a migration, and this script is sized for it. It does
not reserve a large slice, it does not manage a fleet, and it has no `start` —
see the cost model below for why there is nothing to start.

THE COST MODEL IS THE PART TO READ TWICE, because it is where the Vast
intuition is actively wrong. On Vast, `stop` keeps the disk and drops the box
to storage-only, so stop/start is the cheap everyday loop and create/destroy
the rare one (`gpu_box.mjs`). **A Cloud TPU node has no such state in the v2
API.** There is no "stopped" node that costs less: the resource either exists
and bills, or it does not exist. v5e list price is $1.20 per chip-hour in
us-central1/us-east5/us-west1/us-west4, and a v5e-8 is 8 chips, so **an
idle-but-existing node burns ~$9.60/hour** — 10-30x the rented 4090s
(TPU_ACCESS.md §6). Spot is cheaper and dynamic; the only honest spot number
is the one the console shows on the day.

So: **DELETING THE NODE IS WHAT STOPS THE BILL.** `delete` is not the rare
destructive command here, it is the normal end of every session, and the two
things that make it cheap to reach for are (a) the staging bucket — the
tensors live in GCS, not on the node's disk, so a fresh node costs a download
rather than a rebuild — and (b) the repo's own GitHub releases, which already
seed `data-cache-v1` tensors and `model-checkpoints-v1` checkpoints onto a
stranger's machine with nothing but `curl -fsSL`. The node holds no state
worth keeping. Treat it as disposable and the arithmetic works; treat it as a
box you park overnight and it costs ~$230 to learn otherwise.

CREDENTIALS: read from a FILE, never argv — the same rule as `gpu_box.mjs`,
for the same two reasons (the permission classifier correctly blocks secrets
in command lines, and a key in a shell history is a key on disk anyway). Here
it is sharper: the service-account JSON is the credential TPU_ACCESS.md §5
says is *"never written into this repository, never committed, never left on a
rented box"*, so this script **REFUSES a key file that resolves inside the
repo working tree** rather than trusting a future session to remember. The
path comes from `--key-file` or `$GCP_SA_KEY`; the key CONTENT is never
accepted on the command line at all.

  --key-file /path/outside/the/repo/sa.json     (or GCP_SA_KEY=<that path>)
  --project  <project id>                       (or GCP_PROJECT)
  --zone     <zone with the v5e quota>          (or GCP_ZONE)

Auth is the JWT-bearer grant, done in the standard library: build the header
and claims, sign RS256 by piping the signing input through `openssl dgst
-sha256 -sign`, and exchange the assertion at oauth2.googleapis.com for an
access token. No pip dependency, because this has to run on macOS system
python and on a Linux box with nothing installed.

ASSERT THE EFFECT, NOT THE INVOCATION (`ml/CLAUDE.md` §0.2). An API that
answers 200 has told you about a request, not about the world:
  · `create` polls the operation and then GETs the node. If the final read is
    not state READY, this exits non-zero naming the state it actually read.
  · `delete` polls the operation and then GETs the node. If that GET does not
    404, this exits non-zero — a delete that "succeeded" while the node still
    answers is a node still billing.
Nothing is best-effort and nothing is `|| true`: every HTTP failure prints the
status code and the first 300 characters of the response body, and a quota
refusal gets its own pointer at TPU_ACCESS.md §3 (the metric names are not
what the marketing calls them, which is the single commonest way this fails).

Usage:
  python3 scripts/tpu_box.py token
  python3 scripts/tpu_box.py create earth-tpu-1 --spot \\
      --startup-file ml/jaxport/tpu_smoke.sh
  python3 scripts/tpu_box.py list
  python3 scripts/tpu_box.py status earth-tpu-1
  python3 scripts/tpu_box.py delete earth-tpu-1        # this stops the bill
  python3 scripts/tpu_box.py stage ml/cache/family3_na025.npy \\
      gs://<bucket>/tensors/family3_na025.npy
Every mutating subcommand takes --dry-run: it prints the method, URL and JSON
body and makes no network call.
"""
import argparse
import base64
import http.client
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

TPU_BASE = "https://tpu.googleapis.com/v2"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
JWT_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
GCS_UPLOAD = "https://storage.googleapis.com/upload/storage/v1"

# TPU_ACCESS.md §3: ask for v5e, not v6e — on-demand v5e auto-approves to 64
# cores, both v6e metrics auto-approve at 0 and go to a human.
#
# The default is what the project's quota actually GRANTS — 4 cores per zone,
# both kinds — because a default larger than the grant produces the nastiest
# 429 in the API: "Quota limit ... exceeded. Limit: 4" reads as "the counter
# is full", not "your request is too big", and on 2026-08-26 that cost ~2 h of
# create-retry loops, a zone sweep, and a wrong lore entry hunting a phantom
# stuck counter — with zero nodes billing the whole time — when every create
# had simply been asking for a v5litepod-8 against a 4-core limit. If the
# grant is ever raised, raise this WITH it, and remember the 429 ambiguity:
# on "Limit: N", first compare N against the cores you just asked for.
DEFAULT_ACCEL = "v5litepod-4"
DEFAULT_RUNTIME = "v2-alpha-tpuv5-lite"

# Hosts this project has MEASURED as lemons, which must never be used again
# (Chris, 2026-08-26: "hardcode somewhere that the buggy node will never get
# used again"). Keyed by EXTERNAL IP — the only identifier for the underlying
# machine the v2 API exposes to us; the same IP came back on every one of the
# four sightings, so it is the right key in practice. Stated caveat so a
# future reader can weigh a false positive: ephemeral IPs can in principle be
# recycled onto an innocent machine later, and then this list costs one
# re-create — against the four dead launches (two E-051, two e052-verify)
# that trusting this host cost. `--allow-unhealthy` is the deliberate escape
# hatch (investigation only), and `lemon_reason` is the single decision point
# create/list/status all share.
LEMON_HOSTS = {
    "35.252.240.169": "us-west1-c SPOT host, four in-maintenance sightings "
                      "2026-08-23..26 (E-051 nodes 2 and 3's pool, then "
                      "e052-verify twice); never executed a startup script",
}


def lemon_reason(node):
    """Why this node must not be used, or None if it may be.

    Two independent conditions, either is disqualifying:
      * its external IP is on the measured-lemon list above;
      * it was born UNHEALTHY (health starts with 'UNHEALTHY' right after
        create) — a node that begins life in maintenance is the exact
        signature all four lemon sightings shared, whatever its IP.
    """
    _, external = node_ips(node)
    if external in LEMON_HOSTS:
        return f"external IP {external} is a KNOWN LEMON: {LEMON_HOSTS[external]}"
    health = str(node.get("health") or "")
    if health.startswith("UNHEALTHY"):
        return (f"health reads {health!r} at creation "
                f"({node.get('healthDescription') or 'no description'}) — "
                f"born-in-maintenance is the lemon signature of 2026-08-23..26")
    return None

# 15 minutes. A v5e-8 create is typically well under 5; the ceiling exists so a
# wedged operation surfaces as a refusal rather than as a session that walks
# away from a node it believes did not come up. Note the node may EXIST and
# BILL past this timeout — the message says so.
OP_TIMEOUT_S = 900
OP_POLL_S = 10

CHUNK = 64 * 1024 * 1024          # 64 MiB resumable-upload chunk
CHUNK_TRIES = 3
BODY_SNIP = 300                   # chars of an error body we print


class HttpError(Exception):
    """An HTTP status we did not want, carrying enough to diagnose it."""

    def __init__(self, method, url, status, body):
        self.method, self.url, self.status = method, url, status
        self.body = body or ""
        super().__init__(f"{method} {url} -> {status}: {self.body[:BODY_SNIP]}")


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def repo_root():
    """The working tree this script lives in (scripts/ -> repo root)."""
    return os.path.realpath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), ".."))


def _inside(path, root):
    """True if `path` resolves inside `root`. os.path.commonpath, not a
    prefix test: a plain `startswith` says /Users/x/earthling is inside
    /Users/x/earth, which would refuse a perfectly legal key file."""
    path, root = os.path.realpath(path), os.path.realpath(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:                       # different drives (Windows)
        return False


def resolve_key_file(arg=None, env=None, root=None):
    """The service-account key file's path, or a refusal.

    TPU_ACCESS.md §5 puts this credential on the same footing as the CMEMS
    ones: read for the life of one command, **never written into this
    repository**. That rule is enforced here rather than remembered, because
    the failure mode is a key committed to a public repo and the cost of the
    check is a string comparison.

    The key CONTENT is never accepted — only a path. A caller who pastes the
    JSON itself gets told to write it to a file outside the tree instead.
    """
    val = arg if arg else (env if env is not None else os.environ.get("GCP_SA_KEY"))
    if not val:
        raise SystemExit(
            "no service-account key: pass --key-file <path> or set "
            "GCP_SA_KEY=<path>. It is a PATH, never the key itself, and the "
            "file must live outside this repository (ml/plans/TPU_ACCESS.md "
            "§5).")
    if val.lstrip().startswith("{") or '"private_key"' in val:
        raise SystemExit(
            "--key-file/GCP_SA_KEY looks like the key's CONTENTS, not a path. "
            "Key material never travels in argv or the environment's value "
            "position: write the JSON to a file outside the repo (mode 0600) "
            "and pass that path.")
    path = os.path.realpath(os.path.expanduser(val))
    if _inside(path, root or repo_root()):
        raise SystemExit(
            f"REFUSED: the service-account key resolves inside the repo "
            f"working tree ({path}).\nml/plans/TPU_ACCESS.md §5 and "
            f"ml/CLAUDE.md §6: this credential is never written into this "
            f"repository and never committed. Move it somewhere outside "
            f"{root or repo_root()} (e.g. ~/.gcp_sa.json, mode 0600) and "
            f"point --key-file/GCP_SA_KEY at that.")
    if not os.path.exists(path):
        raise SystemExit(f"service-account key file not found: {path}")
    return path


def load_sa(path):
    with open(path) as fh:
        sa = json.load(fh)
    for k in ("client_email", "private_key"):
        if not sa.get(k):
            raise SystemExit(f"{path}: service-account JSON has no {k!r} — "
                             f"is this the key file from IAM → Service "
                             f"accounts → Keys → JSON?")
    return sa


# --------------------------------------------------------------------------
# the JWT-bearer grant, in the standard library
# --------------------------------------------------------------------------
def b64url(raw):
    """base64url with the padding stripped, which is what JWS wants."""
    if isinstance(raw, str):
        raw = raw.encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def jwt_parts(sa, now=None, scope=SCOPE, audience=TOKEN_URL, lifetime=3600):
    """`(header_b64, claims_b64, signing_input_bytes)` for the assertion.

    Split out from the signing so a test can assemble the claims without an
    RSA key, and so the SIGNING INPUT is one object rather than something two
    call sites re-derive. Google's token endpoint wants `iss` = the service
    account's own address (no `sub`: the account acts as itself, which is what
    TPU_ACCESS.md §5's "Service Account User" role is for), `aud` = the token
    endpoint itself, and an `exp` no more than an hour out.
    """
    now = int(time.time() if now is None else now)
    header = {"alg": "RS256", "typ": "JWT"}
    if sa.get("private_key_id"):
        header["kid"] = sa["private_key_id"]
    claims = {"iss": sa["client_email"], "scope": scope, "aud": audience,
              "iat": now, "exp": now + int(lifetime)}
    h = b64url(json.dumps(header, separators=(",", ":"), sort_keys=True))
    c = b64url(json.dumps(claims, separators=(",", ":"), sort_keys=True))
    return h, c, f"{h}.{c}".encode()


def sign_rs256(signing_input, private_key_pem):
    """RS256 over `signing_input`, via `openssl dgst -sha256 -sign`.

    The PEM goes to a tempfile at mode 0600 and is deleted in a `finally`,
    because openssl reads a key from a FILE and the alternative — a key on the
    command line or in an environment variable a child process can read — is
    exactly what the credentials rule forbids. `mkstemp` already creates at
    0600; the explicit chmod is there so the guarantee is stated rather than
    inherited from a library default that could change.
    """
    fd, path = tempfile.mkstemp(prefix="tpu_box_", suffix=".pem")
    try:
        os.write(fd, private_key_pem.encode() if isinstance(private_key_pem, str)
                 else private_key_pem)
        os.close(fd)
        fd = None
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        p = subprocess.run(["openssl", "dgst", "-sha256", "-sign", path],
                           input=signing_input, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise SystemExit(
                "openssl could not sign the JWT assertion "
                f"(exit {p.returncode}): {p.stderr.decode(errors='replace').strip()}")
        if not p.stdout:
            raise SystemExit("openssl returned an EMPTY signature — refusing "
                             "to send an assertion that cannot verify.")
        return p.stdout
    finally:
        if fd is not None:
            os.close(fd)
        if os.path.exists(path):
            os.remove(path)


def make_assertion(sa, now=None, scope=SCOPE, audience=TOKEN_URL, lifetime=3600):
    """The signed JWT: `<header>.<claims>.<signature>`, all base64url."""
    h, c, signing_input = jwt_parts(sa, now, scope, audience, lifetime)
    sig = sign_rs256(signing_input, sa["private_key"])
    return f"{h}.{c}.{b64url(sig)}"


def mint_token(key_path, now=None):
    """`(access_token, expires_at_epoch)` from the service-account key."""
    sa = load_sa(key_path)
    assertion = make_assertion(sa, now=now)
    data = urllib.parse.urlencode({"grant_type": JWT_GRANT,
                                   "assertion": assertion}).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=data, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise HttpError("POST", TOKEN_URL, e.code, body) from None
    except urllib.error.URLError as e:
        raise SystemExit(f"POST {TOKEN_URL} failed to connect: {e.reason}")
    tok = payload.get("access_token")
    if not tok:
        raise SystemExit(f"token endpoint returned no access_token: "
                         f"{json.dumps(payload)[:BODY_SNIP]}")
    return tok, int(time.time()) + int(payload.get("expires_in", 3600))


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def api(method, url, token, body=None, timeout=120):
    """One JSON call. Returns the decoded body ({} when empty)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        raise HttpError(method, url, e.code,
                        e.read().decode(errors="replace")) from None
    except urllib.error.URLError as e:
        raise SystemExit(f"{method} {url} failed to connect: {e.reason}")
    return json.loads(raw) if raw.strip() else {}


def quota_hint(err):
    """A dedicated pointer for the one refusal that is not a bug in the call.

    Quota is the failure this setup is most likely to hit first and the one
    whose message is least actionable on its own — the metric names in the
    console are not what the marketing calls them, and the quota does not even
    appear until the API that owns it is enabled. So say where the click-path
    is instead of leaving a bare RESOURCE_EXHAUSTED on screen.
    """
    body = (err.body or "").lower()
    if err.status in (429, 403) or "resource_exhausted" in body or "quota" in body:
        return ("\nThis reads as a QUOTA refusal. FIRST compare 'Limit: N' "
                "against the CORES YOU JUST ASKED FOR — a v5litepod-8 against "
                "a 4-core grant 429s identically to a full counter, in every "
                "zone, with nothing billing (measured 2026-08-26; ~2 h lost "
                "to a phantom-quota hunt). If the request fits the limit, "
                "ml/plans/TPU_ACCESS.md §3 "
                "has the click-path and the exact metric names:\n"
                "  spot v5e      -> 'Preemptible TPU v5 lite pod cores per "
                "project per zone'\n"
                "  on-demand v5e -> 'TPU v5 lite pod cores per project per "
                "zone'\n"
                "Ask for 8 cores of v5e (auto-approves; both v6e metrics "
                "auto-approve at 0 and wait on a human), and enable the Cloud "
                "TPU API first (§2) or the metric is not in the list at all.")
    return ""


def die_http(err):
    print(f"{err.method} {err.url} -> HTTP {err.status}", file=sys.stderr)
    print(f"  body: {err.body[:BODY_SNIP]}", file=sys.stderr)
    hint = quota_hint(err)
    if hint:
        print(hint, file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------
# request builders — pure, so the tests can assert their shape offline
# --------------------------------------------------------------------------
def nodes_url(project, zone):
    return (f"{TPU_BASE}/projects/{urllib.parse.quote(project)}"
            f"/locations/{urllib.parse.quote(zone)}/nodes")


def create_request(project, zone, name, accelerator_type=DEFAULT_ACCEL,
                   runtime_version=DEFAULT_RUNTIME, spot=False,
                   startup_script=None):
    """POST .../nodes?nodeId=<name> and the node body.

    `schedulingConfig.spot` is the v2 spelling (the older `preemptible` field
    is a different, deprecated thing — do not send both). The startup script
    rides in `metadata`, which is where the TPU VM's own guest agent reads it,
    the same role Vast's `onstart` plays.
    """
    url = f"{nodes_url(project, zone)}?nodeId={urllib.parse.quote(name)}"
    # enableExternalIps: the raw v2 API defaults to NO external IP — unlike
    # `gcloud`, which requests one. Without it the VM has no internet egress,
    # so a startup script that fetches from GitHub hangs forever while the
    # node bills as READY with health TIMEOUT. Measured 2026-08-22: 8 hours
    # of exactly that on the first smoke attempt.
    body = {"acceleratorType": accelerator_type,
            "runtimeVersion": runtime_version,
            "networkConfig": {"enableExternalIps": True}}
    if spot:
        body["schedulingConfig"] = {"spot": True}
    if startup_script is not None:
        body["metadata"] = {"startup-script": startup_script}
    return "POST", url, body


def list_request(project, zone):
    return "GET", nodes_url(project, zone), None


def get_request(project, zone, name):
    return "GET", f"{nodes_url(project, zone)}/{urllib.parse.quote(name)}", None


def delete_request(project, zone, name):
    return ("DELETE",
            f"{nodes_url(project, zone)}/{urllib.parse.quote(name)}", None)


def operation_url(op_name):
    """`projects/P/locations/Z/operations/OP` -> the polling URL."""
    return f"{TPU_BASE}/{op_name.lstrip('/')}"


def stage_init_request(bucket, obj):
    """POST that opens a resumable upload session and returns its URI."""
    url = (f"{GCS_UPLOAD}/b/{urllib.parse.quote(bucket)}/o"
           f"?uploadType=resumable&name={urllib.parse.quote(obj, safe='')}")
    return "POST", url, {"name": obj}


def parse_gs(uri):
    if not uri.startswith("gs://"):
        raise SystemExit(f"destination must be gs://<bucket>/<object>, got {uri!r}")
    rest = uri[5:]
    if "/" not in rest:
        raise SystemExit(f"destination names a bucket but no object: {uri!r}")
    bucket, obj = rest.split("/", 1)
    if not bucket or not obj:
        raise SystemExit(f"destination must be gs://<bucket>/<object>, got {uri!r}")
    return bucket, obj


def show_dry_run(method, url, body):
    """Print exactly what would be sent, and nothing else.

    The startup script is ELIDED rather than dumped: it is a whole shell file,
    and burying the two fields a reader is actually checking (accelerator type
    and spot) under 200 lines of bash is how a dry run stops being read. The
    byte count and the first line are what say "yes, that is the file I meant".
    """
    print(f"DRY RUN — no network call made\n  {method} {url}")
    if body is None:
        print("  body: (none)")
        return
    shown = dict(body)
    meta = shown.get("metadata") or {}
    script = meta.get("startup-script")
    if script is not None:
        head = script.splitlines()[0] if script.splitlines() else ""
        shown["metadata"] = dict(meta, **{
            "startup-script": f"<{len(script)} bytes, elided — first line: "
                              f"{head[:80]!r}>"})
    print("  body: " + json.dumps(shown, indent=2).replace("\n", "\n  "))


# --------------------------------------------------------------------------
# operations and node read-out
# --------------------------------------------------------------------------
def poll_operation(token, op_name, timeout=OP_TIMEOUT_S, poll=OP_POLL_S,
                   sleep=time.sleep):
    """Poll until `done`, then return the operation. Raises on its error."""
    url = operation_url(op_name)
    t0 = time.time()
    while True:
        op = api("GET", url, token)
        if op.get("done"):
            if op.get("error"):
                err = op["error"]
                raise SystemExit(
                    f"operation {op_name} FAILED: code {err.get('code')} "
                    f"{str(err.get('message'))[:BODY_SNIP]}")
            return op
        waited = time.time() - t0
        if waited > timeout:
            raise SystemExit(
                f"operation {op_name} still not done after {waited / 60:.1f} "
                f"min (timeout {timeout / 60:.0f} min). The node may EXIST "
                f"AND BE BILLING — run `status` and then `delete`, because "
                f"deleting is what stops the meter.")
        print(f"  … operation running, {waited / 60:.1f} min elapsed",
              flush=True)
        sleep(poll)


def node_ips(node):
    """(internal, external) from networkEndpoints, '-' where absent."""
    eps = node.get("networkEndpoints") or []
    internal = ", ".join(e.get("ipAddress", "") for e in eps if e.get("ipAddress"))
    external = ", ".join((e.get("accessConfig") or {}).get("externalIp", "")
                         for e in eps
                         if (e.get("accessConfig") or {}).get("externalIp"))
    return internal or "-", external or "-"


def is_spot(node):
    return bool((node.get("schedulingConfig") or {}).get("spot"))


def print_node(node):
    internal, external = node_ips(node)
    if external in LEMON_HOSTS:
        print(f"  LEMON:   {LEMON_HOSTS[external]}")
    print(f"  state:   {node.get('state', '?')}")
    print(f"  health:  {node.get('health', '-')}  "
          f"{node.get('healthDescription', '') or ''}".rstrip())
    print(f"  accel:   {node.get('acceleratorType', '?')}  "
          f"({'spot' if is_spot(node) else 'on-demand'})")
    print(f"  ip:      internal {internal}   external {external}")


# --------------------------------------------------------------------------
# staging: resumable upload to the GCS JSON API
# --------------------------------------------------------------------------
def _https(url):
    """(connection, path) for a URL, using http.client rather than urllib.

    urllib's redirect handler treats 308 as a redirect to FOLLOW. In a
    resumable upload 308 is not a redirect at all — it is "chunk accepted,
    here is how far I got", the single status the resume logic is built on —
    so the upload path talks http.client directly and reads the status itself.
    """
    p = urllib.parse.urlsplit(url)
    conn = http.client.HTTPSConnection(p.netloc, timeout=600)
    path = p.path + (("?" + p.query) if p.query else "")
    return conn, path


def start_resumable(token, bucket, obj, total, content_type="application/octet-stream"):
    """Open the session; returns its URI (the Location header)."""
    method, url, body = stage_init_request(bucket, obj)
    conn, path = _https(url)
    try:
        conn.request(method, path, body=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": content_type,
            "X-Upload-Content-Length": str(total)})
        resp = conn.getresponse()
        raw = resp.read().decode(errors="replace")
        if resp.status not in (200, 201):
            raise HttpError(method, url, resp.status, raw)
        session = resp.getheader("Location")
        if not session:
            raise SystemExit("GCS accepted the resumable init but returned no "
                             "Location header — there is no session to PUT to.")
        return session
    finally:
        conn.close()


def resumable_offset(session, total):
    """Ask the session how much it already has: the 308 `Range` header.

    Returns the next byte offset. A 200/201 here means the object is already
    complete, which is reported as `total` so the caller stops rather than
    re-uploading a file the service already has.
    """
    conn, path = _https(session)
    try:
        conn.request("PUT", path, body=b"", headers={
            "Content-Length": "0", "Content-Range": f"bytes */{total}"})
        resp = conn.getresponse()
        raw = resp.read().decode(errors="replace")
        if resp.status in (200, 201):
            return total
        if resp.status == 308:
            rng = resp.getheader("Range")
            if not rng:                      # nothing stored yet
                return 0
            return int(rng.split("-")[-1]) + 1
        raise HttpError("PUT", session, resp.status, raw)
    finally:
        conn.close()


def put_chunk(session, blob, start, total):
    """One chunk. Returns (done, next_offset)."""
    end = start + len(blob) - 1
    conn, path = _https(session)
    try:
        conn.request("PUT", path, body=blob, headers={
            "Content-Length": str(len(blob)),
            "Content-Range": f"bytes {start}-{end}/{total}"})
        resp = conn.getresponse()
        raw = resp.read().decode(errors="replace")
        if resp.status in (200, 201):
            return True, total
        if resp.status == 308:
            rng = resp.getheader("Range")
            return False, (int(rng.split("-")[-1]) + 1) if rng else start
        raise HttpError("PUT", session, resp.status, raw)
    finally:
        conn.close()


def stage(token, local, bucket, obj, chunk=CHUNK, sleep=time.sleep):
    """Resumable upload of a local file. Prints progress every chunk.

    This exists because the tensors are the reason the bucket exists
    (TPU_ACCESS.md §4): `family3_na025` is 2.98 GB compressed and 10.88 GB
    expanded, and the pentad and daily tensors are far larger. A single-shot
    upload of that over a session's network is a coin flip; a resumable one
    that re-asks the session where it got to on every failure is not.
    """
    total = os.path.getsize(local)
    print(f"staging {local} ({total / 1e9:.2f} GB) -> gs://{bucket}/{obj}")
    session = start_resumable(token, bucket, obj, total)
    print(f"  resumable session opened, {chunk / (1 << 20):.0f} MiB chunks")
    sent = 0
    t0 = time.time()
    with open(local, "rb") as fh:
        while sent < total:
            fh.seek(sent)
            blob = fh.read(min(chunk, total - sent))
            if not blob:
                raise SystemExit(f"read 0 bytes at offset {sent} of {total} — "
                                 f"the file shrank under us; refusing to "
                                 f"finish an upload that would be truncated.")
            for attempt in range(1, CHUNK_TRIES + 1):
                try:
                    done, nxt = put_chunk(session, blob, sent, total)
                    sent = nxt
                    break
                except HttpError as e:
                    # 4xx other than 408/429 is our request being wrong; a
                    # retry sends the identical bytes and gets the identical
                    # answer, so it is a refusal, not a retry.
                    if 400 <= e.status < 500 and e.status not in (408, 429):
                        die_http(e)
                    if attempt == CHUNK_TRIES:
                        print(f"  chunk at {sent} failed {CHUNK_TRIES}x", file=sys.stderr)
                        die_http(e)
                    print(f"  chunk at {sent} -> HTTP {e.status}: "
                          f"{e.body[:BODY_SNIP]}", file=sys.stderr)
                    sleep(2 ** attempt)
                    sent = resumable_offset(session, total)
                except (http.client.HTTPException, OSError) as e:
                    if attempt == CHUNK_TRIES:
                        raise SystemExit(
                            f"chunk at offset {sent} failed {CHUNK_TRIES} "
                            f"times: {e}")
                    print(f"  chunk at {sent} network error ({e}); "
                          f"re-asking the session where it got to", file=sys.stderr)
                    sleep(2 ** attempt)
                    sent = resumable_offset(session, total)
            el = max(time.time() - t0, 1e-9)
            print(f"  {sent / total * 100:5.1f}%  {sent / 1e9:.2f}/"
                  f"{total / 1e9:.2f} GB  {sent / el / 1e6:.1f} MB/s", flush=True)
    print(f"staged gs://{bucket}/{obj} — {total / 1e9:.2f} GB in "
          f"{(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def need(value, env_name, flag):
    v = value or os.environ.get(env_name)
    if not v:
        raise SystemExit(f"missing {flag} (or ${env_name})")
    return v


def cmd_token(a):
    path = resolve_key_file(a.key_file)
    tok, exp = mint_token(path)
    # NEVER the whole token. It is a bearer credential for the whole
    # cloud-platform scope, and a session's output is not a private place.
    print(f"access token minted: {tok[:8]}… (first 8 chars only)")
    print(f"expires: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(exp))} "
          f"(in {(exp - time.time()) / 60:.0f} min)")


def cmd_create(a):
    project = need(a.project, "GCP_PROJECT", "--project")
    zone = need(a.zone, "GCP_ZONE", "--zone")
    startup = None
    if a.startup_file:
        with open(a.startup_file) as fh:
            startup = fh.read()
    method, url, body = create_request(project, zone, a.name, a.accelerator_type,
                                       a.runtime_version, a.spot, startup)
    if a.dry_run:
        show_dry_run(method, url, body)
        return
    token, _ = mint_token(resolve_key_file(a.key_file))
    print(f"creating {a.name}: {a.accelerator_type} "
          f"({'spot' if a.spot else 'on-demand'}) in {zone}")
    if not a.spot:
        print("  on-demand v5e-8 is ~$9.60/h (TPU_ACCESS.md §6). There is no "
              "cheap 'stopped' state — `delete` is what stops the bill.")
    op = api(method, url, token, body)
    op_name = op.get("name")
    if not op_name:
        raise SystemExit(f"create returned no operation name: "
                         f"{json.dumps(op)[:BODY_SNIP]}")
    print(f"  operation: {op_name}")
    poll_operation(token, op_name)

    # ASSERT THE EFFECT. The operation completing says the control plane is
    # finished; only the node itself says whether there is a TPU to use.
    _, get_url, _ = get_request(project, zone, a.name)
    node = api("GET", get_url, token)
    print_node(node)
    state = node.get("state")
    if state != "READY":
        raise SystemExit(
            f"create finished but {a.name} reads state {state!r}, not READY "
            f"({node.get('healthDescription') or 'no healthDescription'}). "
            f"The node exists and is BILLING — `delete {a.name}` if this is "
            f"not recoverable.")
    reason = lemon_reason(node)
    if reason and not a.allow_unhealthy:
        # NEVER hand a measured lemon back to the caller (Chris, 2026-08-26).
        # The node exists and is billing, so the refusal DELETES it first —
        # a guard that exits leaving the meter running would convert a lemon
        # into a bill. Only then does it exit non-zero, so an automation loop
        # can re-create for a fresh host draw without ever touching this one.
        print(f"  REFUSING this node: {reason}")
        print(f"  deleting it before exiting — the meter must not outlive "
              f"the refusal (--allow-unhealthy overrides, for deliberate "
              f"investigation only)")
        d_method, d_url, _ = delete_request(project, zone, a.name)
        d_op = api(d_method, d_url, token)
        poll_operation(token, d_op.get("name"))
        try:
            api("GET", get_url, token)
            raise SystemExit(
                f"refused {a.name} as a lemon but the delete did not stick — "
                f"the node still answers and is BILLING. Delete it by hand.")
        except HttpError as e:
            if e.status != 404:
                die_http(e)
        raise SystemExit(
            f"refused and deleted {a.name}: {reason}. Re-run create for a "
            f"fresh host draw.")
    if reason:
        print(f"  WARNING (--allow-unhealthy): using a node the guard would "
              f"refuse — {reason}")
    print(f"{a.name} is READY. Delete it when the measurement is done — that "
          f"is what stops the meter.")


def cmd_list(a):
    project = need(a.project, "GCP_PROJECT", "--project")
    zone = need(a.zone, "GCP_ZONE", "--zone")
    token, _ = mint_token(resolve_key_file(a.key_file))
    _, url, _ = list_request(project, zone)
    nodes = api("GET", url, token).get("nodes") or []
    if not nodes:
        print(f"no TPU nodes in {zone} — nothing is billing")
        return
    for n in nodes:
        short = (n.get("name") or "").rsplit("/", 1)[-1]
        print(f"{short:<24} {str(n.get('state', '?')):<12} "
              f"{str(n.get('acceleratorType', '?')):<14} "
              f"{'spot' if is_spot(n) else 'on-demand':<10} "
              f"{n.get('createTime', '-')}")


def cmd_status(a):
    project = need(a.project, "GCP_PROJECT", "--project")
    zone = need(a.zone, "GCP_ZONE", "--zone")
    token, _ = mint_token(resolve_key_file(a.key_file))
    _, url, _ = get_request(project, zone, a.name)
    print(a.name)
    print_node(api("GET", url, token))


def cmd_delete(a):
    project = need(a.project, "GCP_PROJECT", "--project")
    zone = need(a.zone, "GCP_ZONE", "--zone")
    method, url, body = delete_request(project, zone, a.name)
    if a.dry_run:
        show_dry_run(method, url, body)
        return
    token, _ = mint_token(resolve_key_file(a.key_file))
    print(f"deleting {a.name} — deleting is what stops the bill; a TPU node "
          f"has no cheap stopped state (TPU_ACCESS.md §6)")
    op = api(method, url, token)
    op_name = op.get("name")
    if not op_name:
        raise SystemExit(f"delete returned no operation name: "
                         f"{json.dumps(op)[:BODY_SNIP]}")
    print(f"  operation: {op_name}")
    poll_operation(token, op_name)

    # ASSERT THE EFFECT: a delete that "succeeded" while the node still
    # answers is a node still charging ~$9.60/h.
    _, get_url, _ = get_request(project, zone, a.name)
    try:
        node = api("GET", get_url, token)
    except HttpError as e:
        if e.status == 404:
            print(f"{a.name} is gone (GET 404). The meter has stopped.")
            return
        die_http(e)
    raise SystemExit(
        f"delete reported success but GET {a.name} still answers with state "
        f"{node.get('state')!r} — the node exists and is still billing. "
        f"Re-run delete and check the console.")


def cmd_stage(a):
    bucket, obj = parse_gs(a.dest)
    if not os.path.exists(a.local):
        raise SystemExit(f"local file not found: {a.local}")
    method, url, body = stage_init_request(bucket, obj)
    if a.dry_run:
        show_dry_run(method, url, body)
        print(f"  then PUT {os.path.getsize(a.local) / 1e9:.2f} GB to the "
              f"session URI in {CHUNK / (1 << 20):.0f} MiB chunks")
        return
    token, _ = mint_token(resolve_key_file(a.key_file))
    stage(token, a.local, bucket, obj)


def build_parser():
    ap = argparse.ArgumentParser(
        description="Drive one Cloud TPU v5e node and the staging bucket "
                    "(ml/plans/TPU_ACCESS.md §6-7).")
    ap.add_argument("--key-file", help="service-account JSON key PATH "
                                       "(or $GCP_SA_KEY). Never the key itself, "
                                       "and never inside this repo.")
    ap.add_argument("--project", help="GCP project id (or $GCP_PROJECT)")
    ap.add_argument("--zone", help="zone with the v5e quota (or $GCP_ZONE)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("token", help="mint a token; print only its expiry and "
                                 "first 8 chars").set_defaults(fn=cmd_token)

    c = sub.add_parser("create", help="create a TPU node and wait for READY")
    c.add_argument("name")
    c.add_argument("--accelerator-type", default=DEFAULT_ACCEL)
    c.add_argument("--runtime-version", default=DEFAULT_RUNTIME)
    c.add_argument("--spot", action="store_true",
                   help="schedulingConfig.spot — cheaper, preemptible")
    c.add_argument("--startup-file", help="file whose contents become the "
                                          "metadata startup-script")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--allow-unhealthy", action="store_true",
                   help="accept a node lemon_reason would refuse (a known-"
                        "lemon IP or born-UNHEALTHY health). Investigation "
                        "only — the refusal exists because four launches "
                        "died on exactly such a host (2026-08-23..26)")
    c.set_defaults(fn=cmd_create)

    l = sub.add_parser("list", help="nodes in the zone")
    l.set_defaults(fn=cmd_list)

    s = sub.add_parser("status", help="one node's state, health and IPs")
    s.add_argument("name")
    s.set_defaults(fn=cmd_status)

    d = sub.add_parser("delete", help="delete a node — this stops the bill")
    d.add_argument("name")
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(fn=cmd_delete)

    g = sub.add_parser("stage", help="resumable upload of a local file to GCS")
    g.add_argument("local")
    g.add_argument("dest", help="gs://<bucket>/<object>")
    g.add_argument("--dry-run", action="store_true")
    g.set_defaults(fn=cmd_stage)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except HttpError as e:
        die_http(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
