#!/usr/bin/env python3
"""Offline guards on `scripts/tpu_box.py`.

The operator's GCP project, bucket and service account do not exist yet
(`ml/plans/TPU_ACCESS.md` §7), so this script cannot be exercised against the
real API before it is needed — and the first time it IS needed, a node will be
billing ~$9.60/h while somebody debugs it. Everything here therefore runs with
NO NETWORK, against a throwaway RSA key this test generates itself.

Case 1: the JWT assembly is verifiable. Generate an RSA key with `openssl
        genrsa`, build a fake service-account JSON around it, call
        `make_assertion`, then reconstruct the signing input from the
        assertion's own first two segments and verify the signature with
        `openssl dgst -sha256 -verify`. This asserts the EFFECT (a signature
        that verifies) rather than the invocation (that openssl was called) —
        ml/CLAUDE.md §0.2. The claims are decoded and checked for iss/scope/
        aud/iat/exp, because a perfectly-signed assertion with the wrong `aud`
        is rejected by Google with a message that reads like a key problem.
Case 2: request shapes. The URL, method and body of create/delete/stage-init,
        including the spot flag and the startup-script inclusion — the two
        fields whose absence is silent (a node that quietly costs full price,
        or one that comes up with nothing to run).
Case 3: a key file inside the repo working tree is REFUSED. TPU_ACCESS.md §5
        and ml/CLAUDE.md §6: this credential is never written into the
        repository.

    python3 tests/test_tpu_box.py
"""
import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(ROOT, "scripts", "tpu_box.py")


def load_script():
    """Import scripts/tpu_box.py by path — it is an executable, not a package
    member, so there is nothing importable to name."""
    spec = importlib.util.spec_from_file_location("tpu_box", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def b64url_decode(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def fake_sa(tmp):
    """A service-account JSON around a real, throwaway RSA key."""
    key = os.path.join(tmp, "throwaway.pem")
    p = subprocess.run(["openssl", "genrsa", "-out", key, "2048"],
                       capture_output=True)
    if p.returncode != 0:
        raise SystemExit("could not generate a throwaway RSA key: "
                         + p.stderr.decode(errors="replace"))
    with open(key) as fh:
        pem = fh.read()
    return {"type": "service_account",
            "project_id": "earth-tpu-test",
            "private_key_id": "deadbeef",
            "private_key": pem,
            "client_email": "tpu-driver@earth-tpu-test.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token"}, key


def case1_jwt(m, tmp):
    sa, keyfile = fake_sa(tmp)
    now = 1_800_000_000
    assertion = make = m.make_assertion(sa, now=now)
    parts = assertion.split(".")
    if len(parts) != 3:
        raise SystemExit(f"case 1 FAILED: assertion has {len(parts)} segments, "
                         f"expected 3 (header.claims.signature)")
    h_b64, c_b64, sig_b64 = parts

    # --- the signature verifies against the RECONSTRUCTED signing input ---
    signing_input = f"{h_b64}.{c_b64}".encode()
    sig = os.path.join(tmp, "sig.bin")
    inp = os.path.join(tmp, "signing_input.bin")
    pub = os.path.join(tmp, "pub.pem")
    with open(sig, "wb") as fh:
        fh.write(b64url_decode(sig_b64))
    with open(inp, "wb") as fh:
        fh.write(signing_input)
    p = subprocess.run(["openssl", "rsa", "-in", keyfile, "-pubout",
                        "-out", pub], capture_output=True)
    if p.returncode != 0:
        raise SystemExit("case 1 FAILED: could not extract the public key: "
                         + p.stderr.decode(errors="replace"))
    p = subprocess.run(["openssl", "dgst", "-sha256", "-verify", pub,
                        "-signature", sig, inp], capture_output=True)
    if p.returncode != 0 or b"Verified OK" not in p.stdout:
        raise SystemExit(
            f"case 1 FAILED: the RS256 signature does not verify against the "
            f"signing input. openssl said: "
            f"{(p.stdout + p.stderr).decode(errors='replace').strip()}")

    # --- a signature that verifies against the WRONG input must not ------
    bad = os.path.join(tmp, "tampered.bin")
    with open(bad, "wb") as fh:
        fh.write(signing_input + b"x")
    p = subprocess.run(["openssl", "dgst", "-sha256", "-verify", pub,
                        "-signature", sig, bad], capture_output=True)
    if p.returncode == 0 and b"Verified OK" in p.stdout:
        raise SystemExit("case 1 FAILED: the signature verified against a "
                         "TAMPERED input — the check proves nothing")

    # --- the claims are the ones Google's token endpoint wants -----------
    header = json.loads(b64url_decode(h_b64))
    claims = json.loads(b64url_decode(c_b64))
    checks = {
        "alg": (header.get("alg"), "RS256"),
        "typ": (header.get("typ"), "JWT"),
        "kid": (header.get("kid"), sa["private_key_id"]),
        "iss": (claims.get("iss"), sa["client_email"]),
        "scope": (claims.get("scope"),
                  "https://www.googleapis.com/auth/cloud-platform"),
        "aud": (claims.get("aud"), "https://oauth2.googleapis.com/token"),
        "iat": (claims.get("iat"), now),
        "exp": (claims.get("exp"), now + 3600),
    }
    for field, (got, want) in checks.items():
        if got != want:
            raise SystemExit(f"case 1 FAILED: {field} is {got!r}, expected "
                             f"{want!r}")
    if claims.get("sub") is not None:
        raise SystemExit("case 1 FAILED: the assertion carries a `sub` — this "
                         "account acts as ITSELF; a sub asks for domain-wide "
                         "delegation and is refused")
    if make is not assertion:                 # keeps the name used
        raise SystemExit("case 1 FAILED: impossible")
    print("case 1 ok — RS256 assertion verifies; iss/scope/aud/iat/exp correct")


def case2_shapes(m):
    P, Z, N = "earth-tpu-1234", "us-central1-a", "smoke-1"

    # --- create, plain ----------------------------------------------------
    method, url, body = m.create_request(P, Z, N)
    want = (f"https://tpu.googleapis.com/v2/projects/{P}/locations/{Z}"
            f"/nodes?nodeId={N}")
    if method != "POST":
        raise SystemExit(f"case 2 FAILED: create method is {method!r}, not POST")
    if url != want:
        raise SystemExit(f"case 2 FAILED: create URL\n  got  {url}\n  want {want}")
    if body.get("acceleratorType") != "v5litepod-8":
        raise SystemExit(f"case 2 FAILED: default acceleratorType is "
                         f"{body.get('acceleratorType')!r}, not 'v5litepod-8' "
                         f"(TPU_ACCESS.md §3 asks for v5e, not v6e)")
    if body.get("runtimeVersion") != "v2-alpha-tpuv5-lite":
        raise SystemExit(f"case 2 FAILED: default runtimeVersion is "
                         f"{body.get('runtimeVersion')!r}")
    if "schedulingConfig" in body:
        raise SystemExit("case 2 FAILED: schedulingConfig present without "
                         "--spot — that silently changes what the node costs")
    if "metadata" in body:
        raise SystemExit("case 2 FAILED: metadata present with no "
                         "--startup-file")

    # --- create, spot + startup script -----------------------------------
    script = "#!/bin/bash\nset -euo pipefail\necho smoke\n"
    _, url2, body2 = m.create_request(P, Z, N, spot=True, startup_script=script,
                                      accelerator_type="v5litepod-8")
    if body2.get("schedulingConfig") != {"spot": True}:
        raise SystemExit(f"case 2 FAILED: --spot must set schedulingConfig "
                         f"{{'spot': True}}, got "
                         f"{body2.get('schedulingConfig')!r}")
    if (body2.get("metadata") or {}).get("startup-script") != script:
        raise SystemExit("case 2 FAILED: --startup-file must land verbatim in "
                         "metadata['startup-script'] — a node that comes up "
                         "with nothing to run still bills")
    if url2 != want:
        raise SystemExit("case 2 FAILED: --spot must not change the URL")

    # --- list / get / delete ---------------------------------------------
    lm, lu, lb = m.list_request(P, Z)
    if (lm, lb) != ("GET", None) or not lu.endswith(f"/locations/{Z}/nodes"):
        raise SystemExit(f"case 2 FAILED: list request is {lm} {lu}")
    dm, du, db = m.delete_request(P, Z, N)
    if dm != "DELETE" or db is not None:
        raise SystemExit(f"case 2 FAILED: delete is {dm} with body {db!r}")
    if du != (f"https://tpu.googleapis.com/v2/projects/{P}/locations/{Z}"
              f"/nodes/{N}"):
        raise SystemExit(f"case 2 FAILED: delete URL is {du}")
    gm, gu, _ = m.get_request(P, Z, N)
    if gm != "GET" or gu != du:
        raise SystemExit(f"case 2 FAILED: get is {gm} {gu}; it must address "
                         f"the same node resource delete does, or the "
                         f"post-delete 404 check proves nothing")

    # --- the operation polling URL ---------------------------------------
    op = f"projects/{P}/locations/{Z}/operations/operation-123"
    if m.operation_url(op) != f"https://tpu.googleapis.com/v2/{op}":
        raise SystemExit(f"case 2 FAILED: operation URL is {m.operation_url(op)}")

    # --- stage init -------------------------------------------------------
    sm, su, sb = m.stage_init_request("earth-tpu-staging",
                                      "tensors/family3_na025.npy")
    if sm != "POST":
        raise SystemExit(f"case 2 FAILED: stage init method is {sm!r}")
    if su != ("https://storage.googleapis.com/upload/storage/v1/b/"
              "earth-tpu-staging/o?uploadType=resumable"
              "&name=tensors%2Ffamily3_na025.npy"):
        raise SystemExit(f"case 2 FAILED: stage init URL\n  got {su}")
    if sb != {"name": "tensors/family3_na025.npy"}:
        raise SystemExit(f"case 2 FAILED: stage init body is {sb!r}")
    if m.parse_gs("gs://b/o/p.npy") != ("b", "o/p.npy"):
        raise SystemExit("case 2 FAILED: parse_gs mis-splits bucket/object")
    if m.CHUNK != 64 * 1024 * 1024:
        raise SystemExit(f"case 2 FAILED: chunk size is {m.CHUNK}, not 64 MiB")
    print("case 2 ok — create/list/get/delete/operation/stage-init shapes, "
          "spot flag and startup-script inclusion")


def case3_refusal(m, tmp):
    inside = os.path.join(ROOT, ".tpu_box_test_key.json")
    with open(inside, "w") as fh:
        json.dump({"client_email": "x@y.iam.gserviceaccount.com",
                   "private_key": "-----BEGIN PRIVATE KEY-----"}, fh)
    try:
        try:
            m.resolve_key_file(inside)
        except SystemExit as e:
            if "REFUSED" not in str(e):
                raise SystemExit(f"case 3 FAILED: refused, but not with the "
                                 f"rule: {e}")
        else:
            raise SystemExit(
                "case 3 FAILED: a service-account key INSIDE the repo working "
                "tree was accepted. TPU_ACCESS.md §5 says this credential is "
                "never written into this repository.")
        # a nested path must be refused too, not just the root
        nested = os.path.join(ROOT, "ml", "cache")
        os.makedirs(nested, exist_ok=True)
        deep = os.path.join(nested, ".tpu_box_test_key.json")
        with open(deep, "w") as fh:
            fh.write("{}")
        try:
            m.resolve_key_file(deep)
        except SystemExit as e:
            if "REFUSED" not in str(e):
                raise SystemExit(f"case 3 FAILED (nested): {e}")
        else:
            raise SystemExit("case 3 FAILED: a key under ml/cache/ was accepted")
        os.remove(deep)
    finally:
        if os.path.exists(inside):
            os.remove(inside)

    # a key OUTSIDE the tree is accepted, or the guard is just "refuse always"
    outside = os.path.join(tmp, "sa.json")
    with open(outside, "w") as fh:
        fh.write("{}")
    got = m.resolve_key_file(outside)
    if got != os.path.realpath(outside):
        raise SystemExit(f"case 3 FAILED: a key outside the tree resolved to "
                         f"{got!r}, expected {outside!r}")

    # a sibling directory whose name merely STARTS with the repo path is not
    # inside it — a prefix test would get this wrong.
    sibling = ROOT + "ling"
    os.makedirs(sibling, exist_ok=True)
    try:
        sib = os.path.join(sibling, "sa.json")
        with open(sib, "w") as fh:
            fh.write("{}")
        m.resolve_key_file(sib)
        os.remove(sib)
    except SystemExit as e:
        raise SystemExit(f"case 3 FAILED: {ROOT}ling is NOT inside {ROOT}, "
                         f"but the guard refused it: {e}")
    finally:
        if os.path.isdir(sibling) and not os.listdir(sibling):
            os.rmdir(sibling)

    # the key CONTENTS must never be accepted in the path position
    try:
        m.resolve_key_file('{"private_key": "-----BEGIN PRIVATE KEY-----"}')
    except SystemExit as e:
        if "CONTENTS" not in str(e) and "contents" not in str(e):
            raise SystemExit(f"case 3 FAILED: pasted key material refused, but "
                             f"not for the right reason: {e}")
    else:
        raise SystemExit("case 3 FAILED: the key's CONTENTS were accepted in "
                         "the --key-file position — key material must never "
                         "travel in argv")
    print("case 3 ok — in-repo key refused (root and nested), outside key "
          "accepted, pasted key material refused")


def main():
    m = load_script()
    with tempfile.TemporaryDirectory() as tmp:
        case1_jwt(m, tmp)
        case2_shapes(m)
        case3_refusal(m, tmp)
    print("\nall 3/3 tpu_box guards hold")


if __name__ == "__main__":
    main()
