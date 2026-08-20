#!/usr/bin/env python3
"""`scripts/sroll_run.sh` must derive its inputs from the RECIPE, not from
family-3 literals — and must refuse rather than guess.

Until 2026-08-19 this script named the monthly family-3 tensor four times in
values no dispatch input could override: the codec asset
`f3_anchor41M__pixelmae.pt`, the Z glob `Z_*_6c52f0687b_adcbe700fb.npy`, a
four-chunk fallback download, and `--x ml/cache/family3_X.npy --npz-small
ml/cache/f3_dec_small.npz`. A `recipe:` naming the pentad tensor would have
rolled family 3 with a family-3 codec and reported the result as a pentad
number — the failure that costs a day and looks like a result.

This test runs the REAL script against a toy `ml/cache` (no network: `curl`
is a stub that 404s everything, so every path the script takes must already
be satisfiable from disk), and separately runs the script's own final
verification block — lifted out of the file rather than copied — against
three artefacts.

    python3 tests/test_sroll_wiring.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = os.path.join("scripts", "sroll_run.sh")
sys.path.insert(0, os.path.join(ROOT, "ml"))
from temporal import codec_weight_hash, data_fingerprint      # noqa: E402


def sandbox():
    """A repo-shaped temp dir: `scripts/` and `ml/` symlinked file by file so
    the sandbox owns its own `ml/cache`, plus a `curl` that 404s everything —
    every path the script takes must be satisfiable from disk alone."""
    tmp = tempfile.mkdtemp()
    os.symlink(os.path.join(ROOT, "scripts"), os.path.join(tmp, "scripts"))
    os.makedirs(os.path.join(tmp, "ml"))
    for name in os.listdir(os.path.join(ROOT, "ml")):
        if name in ("cache", "__pycache__"):
            continue
        os.symlink(os.path.join(ROOT, "ml", name),
                   os.path.join(tmp, "ml", name))
    cache = os.path.join(tmp, "ml", "cache")
    os.makedirs(cache)
    bin_ = os.path.join(tmp, "bin")
    os.makedirs(bin_)
    with open(os.path.join(bin_, "curl"), "w") as f:
        f.write("#!/bin/sh\necho \"curl-stub $*\" >&2\nexit 22\n")
    os.chmod(os.path.join(bin_, "curl"), 0o755)
    return tmp, cache, bin_


def make_tensor(path, T=8, H=4, W=5, C=3, cadence=None):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, H, W, C)).astype(np.float32)
    months = np.array([f"1990-{i % 12 + 1:02d}" for i in range(T)])
    kw = {}
    if cadence:
        kw = dict(bin_index=np.arange(1000, 1000 + T),
                  pentad_days=np.array(cadence), cadence=np.array("pentad"))
    np.savez(path, X=X, months=months, lats=np.arange(H, dtype=np.float32),
             lons=np.arange(W, dtype=np.float32),
             chan=np.array([f"c{i}" for i in range(C)]),
             norm=np.zeros((C, 2), np.float32),
             rapid=np.stack([np.arange(T, dtype=float),
                             rng.standard_normal(T)], 1), **kw)


def make_ckpt(path):
    torch.save({"model": {"a": torch.arange(4.0), "b": torch.arange(3.0),
                          "c": torch.ones(2), "d": torch.zeros(2)},
                "chan": ["c0"], "d_z": 4,
                "args": {"holdout_years": "1992", "holdout_lon": "0,0"}},
               path)


def run(tmp, bin_, window, env=None):
    e = dict(os.environ)
    e["PATH"] = bin_ + os.pathsep + e["PATH"]
    e["GITHUB_REPOSITORY"] = "blauewelt/earth"
    e.update(env or {})
    return subprocess.run(["bash", SCRIPT, window], cwd=tmp, env=e,
                          capture_output=True, text=True, timeout=900)


def verify_block():
    """The script's OWN final check, lifted from the file it ships in."""
    src = open(os.path.join(ROOT, SCRIPT)).read()
    m = re.search(r'python - "\$OUT" <<\'PYEOF\'\n(.*?)\nPYEOF\n', src, re.S)
    assert m, "the final verification heredoc is no longer where this test looks"
    return m.group(1)


def check_artefact(payload, tmp):
    p = os.path.join(tmp, "art.json")
    json.dump(payload, open(p, "w"))
    src = os.path.join(tmp, "verify.py")
    open(src, "w").write(verify_block())
    return subprocess.run([sys.executable, src, p], capture_output=True,
                          text=True, cwd=ROOT)


def main():
    tmp, cache, bin_ = sandbox()
    tmp2, cache2, bin2 = sandbox()
    try:
        # ---- 1. tokens: heads, an optional ckpt:, and nothing else -------
        r = run(tmp, bin_, "sroll:a,b,junk:1")
        assert r.returncode != 0 and "unknown sroll token 'junk:1'" in r.stdout, \
            (r.returncode, r.stdout[-500:])
        r = run(tmp, bin_, "sroll:")
        assert r.returncode != 0 and "no head tags" in r.stdout, r.stdout[-500:]
        print("1. the window parses into head tags plus an optional `ckpt:`; "
              "an unknown `x:y` token is refused, an empty tag list is refused")

        # ---- 2. a non-family-3 tensor with no codec REFUSES, up front ----
        make_tensor(os.path.join(cache2, "family4_na025_pentad_r2.npz"),
                    cadence=5)
        r = run(tmp2, bin2, "sroll:h1",
                {"TENSOR": "ml/cache/family4_na025_pentad_r2.npz"})
        assert r.returncode != 0, r.stdout[-800:]
        assert "not the monthly family-3 tensor" in r.stdout, r.stdout[-800:]
        assert "#10/#11" in r.stdout, r.stdout[-800:]
        assert "X memmap" not in r.stdout and "embed cache key" not in r.stdout, \
            "the refusal fired AFTER doing work — it must cost only the inputs"
        print("2. a pentad tensor with no `ckpt:` is refused BEFORE any "
              "extraction or hashing, naming the #10/#11 failure it prevents")

        # ---- 3. the derived paths, on a real toy tensor ------------------
        tpath = os.path.join(cache, "family3_na025.npz")
        make_tensor(tpath)
        ck = os.path.join(cache, "toy_codec.pt")
        make_ckpt(ck)
        wh = codec_weight_hash(torch.load(ck, map_location="cpu",
                                          weights_only=False))
        dh = data_fingerprint(tpath)
        zname = os.path.join(cache, f"Z_actions_{wh}_{dh}.npy")
        np.save(zname, np.zeros((8, 4, 4), np.float16))
        r = run(tmp, bin_, f"sroll:h1,ckpt:{os.path.relpath(ck, tmp)}",
                {"TENSOR": "ml/cache/family3_na025.npz"})
        out = r.stdout + r.stderr
        assert f"embed cache key: codec {wh} · tensor {dh}" in out, out[-1500:]
        assert os.path.basename(zname) in out, out[-1500:]
        assert "6c52f0687b" not in out and "adcbe700fb" not in out, \
            "a family-3 literal is still in the Z path"
        assert os.path.exists(os.path.join(cache, "family3_X.npy")), \
            "X was not extracted to the historical family-3 name"
        assert os.path.exists(os.path.join(cache, "f3_dec_small.npz")), \
            "the small npz was not written to the historical family-3 name"
        # the heads are the only thing left, and the stubbed curl has none
        assert "no heads fetched" in out and r.returncode != 0
        print("3. on a toy tensor + toy codec the script derived the embed "
              "cache key (codec %s · tensor %s) and found the Z named for it, "
              "extracted X to the historical family-3 names (so no warm box "
              "re-extracts 10.9 GiB), and carried no family-3 hash literal"
              % (wh, dh))

        # ---- 4. the Z it will not accept --------------------------------
        os.remove(zname)
        bad = os.path.join(cache, f"Z_actions_{wh}_{dh}.npy")
        with open(bad, "wb") as fh:
            fh.write(open(os.path.join(cache, "family3_X.npy"), "rb").read(200))
        r = run(tmp, bin_, f"sroll:h1,ckpt:{os.path.relpath(ck, tmp)}",
                {"TENSOR": "ml/cache/family3_na025.npz"})
        out = r.stdout + r.stderr
        assert r.returncode != 0 and "embed cache rejected" in out, out[-1200:]
        print("4. a Z whose length disagrees with its own .npy header is "
              "REJECTED before the roll — it would map cleanly and return "
              "real numbers belonging to the wrong rows")

        # ---- 5. the final check: pass, uncertified, and the bad case -----
        head = {"corridor": {"chan_skill": [{"h": 1}], "horizon_auc": 0.5},
                "window": {"horizon_auc": 0.4}}
        ok_m = {"gate": {"pass": True}, "heads": {"a": head}}
        r1 = check_artefact(ok_m, tmp)
        assert r1.returncode == 0 and "gate: PASSED" in r1.stdout, r1

        bad_m = {"gate": {"pass": None, "skipped": True}, "heads": {"a": head}}
        r2 = check_artefact(bad_m, tmp)
        assert r2.returncode != 0, \
            "a MONTHLY roll whose gate was skipped was accepted — the " \
            "certificate is what makes the number believable"

        pentad = {"gate": {"pass": None, "skipped": True, "certified": False,
                           "cadence": "pentad", "reason": "no reference at "
                           "pentad cadence; the monthly one cannot certify it"},
                  "cadence": {"name": "pentad", "step_days": 5,
                              "horizon_steps": 12, "horizon_span_days": 60.0},
                  "heads": {"a": head}}
        r3 = check_artefact(pentad, tmp)
        assert r3.returncode == 0, r3.stderr[-800:]
        assert "NOT CERTIFIED at pentad cadence" in r3.stdout, r3.stdout
        assert "60" in r3.stdout, r3.stdout

        mute = dict(pentad, gate={"pass": None, "skipped": True,
                                  "certified": False, "cadence": "pentad"})
        r4 = check_artefact(mute, tmp)
        assert r4.returncode != 0, \
            "a pentad roll that skipped the gate WITHOUT a recorded reason " \
            "was accepted — a silent skip is what this whole change removes"
        print("5. the shipped verification block: monthly gate must PASS "
              "(a skipped monthly gate fails, rc %d), a pentad artefact is "
              "accepted and printed as NOT CERTIFIED with its reason and its "
              "step length, and a pentad skip with NO recorded reason fails "
              "(rc %d)" % (r2.returncode, r4.returncode))

        print("\nsroll_run.sh wiring: all 5 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


if __name__ == "__main__":
    main()
