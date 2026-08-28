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
import datetime as dt
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


EPOCH = dt.date(1982, 1, 1)


def make_tensor(path, T=8, H=4, W=5, C=3, cadence=None):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, H, W, C)).astype(np.float32)
    months = np.array([f"1990-{i % 12 + 1:02d}" for i in range(T)])
    kw = {}
    if cadence:
        # the LABELS must be the bins' own calendar months. TimeAxis checks
        # that for every row and refuses otherwise (ml/CLAUDE.md §4.9), which
        # is the point — so a fixture that wants to reach the horizon
        # derivation has to be a consistent axis, not a shaped one.
        bins = np.arange(1000, 1000 + T)
        d0 = [EPOCH + dt.timedelta(days=int(b) * cadence) for b in bins]
        months = np.array([f"{d.year:04d}-{d.month:02d}" for d in d0])
        kw = dict(bin_index=bins, pentad_days=np.array(cadence),
                  cadence=np.array("pentad"),
                  epoch=np.array(str(EPOCH)))
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
    m = re.search(r'python - "\$OUT" "\$HORIZON" <<\'PYEOF\'\n(.*?)\nPYEOF\n',
                  src, re.S)
    assert m, "the final verification heredoc is no longer where this test looks"
    return m.group(1)


def check_artefact(payload, tmp, horizon=None):
    p = os.path.join(tmp, "art.json")
    json.dump(payload, open(p, "w"))
    src = os.path.join(tmp, "verify.py")
    open(src, "w").write(verify_block())
    h = str(horizon if horizon is not None else payload.get("horizon", 12))
    return subprocess.run([sys.executable, src, p, h], capture_output=True,
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
        assert r.returncode != 0, out[-1200:]
        assert "named head(s) not fetched from the release" in out, out[-1200:]
        assert " h1" in out.split("named head(s) not fetched from the "
                                  "release:")[1][:40], \
            "the refusal must NAME the head it could not fetch"
        print("3. on a toy tensor + toy codec the script derived the embed "
              "cache key (codec %s · tensor %s) and found the Z named for it, "
              "extracted X to the historical family-3 names (so no warm box "
              "re-extracts 10.9 GiB), and carried no family-3 hash literal"
              % (wh, dh))

        # ---- 3b. ONE named head missing is a REFUSAL, not a skip --------
        # The #421 regression (2026-08-20). The dispatch named the gate and
        # `head-weights-e043b-xl144-nolonhold-s0`; the head's fetch 404'd
        # (Fastly had cached a `BlobNotFound` from a GET issued while the
        # asset's `state` was still `starter`); the loop warned and skipped
        # it; the run rolled the GATE ALONE and went green. Nothing in the
        # artefact could catch it — every head that IS present is complete.
        # So: a curl that succeeds for one tag and fails for the other must
        # take the whole run down, and must say WHICH tag.
        bin3 = os.path.join(tmp, "bin3")
        os.makedirs(bin3)
        with open(os.path.join(bin3, "curl"), "w") as f:
            # succeed for the gate, 404 everything else — exactly #421's shape
            f.write('#!/bin/sh\nfor a in "$@"; do case "$a" in\n'
                    '  *gate_ok*) : ;; esac; done\n'
                    'out=""; url=""\n'
                    'while [ $# -gt 0 ]; do\n'
                    '  case "$1" in -o) out="$2"; shift 2 ;;\n'
                    '    http*) url="$1"; shift ;; *) shift ;; esac\n'
                    'done\n'
                    'case "$url" in *gate_ok*) echo stub > "$out"; exit 0 ;;\n'
                    '  *) exit 22 ;; esac\n')
        os.chmod(os.path.join(bin3, "curl"), 0o755)
        np.save(zname, np.zeros((8, 4, 4), np.float16))
        r = run(tmp, bin3, f"sroll:gate_ok,missing_head,ckpt:"
                           f"{os.path.relpath(ck, tmp)}",
                {"TENSOR": "ml/cache/family3_na025.npz"})
        out = r.stdout + r.stderr
        assert r.returncode != 0, \
            "one named head 404'd and the run continued — this is #421, the " \
            "void roll that went green:\n" + out[-1500:]
        assert "named head(s) not fetched from the release" in out, out[-1500:]
        assert "missing_head" in out.split("not fetched from the release:")[1][:60], \
            "the refusal did not name the head that was missing"
        assert "gate_ok: fetched" in out, \
            "the head that DID arrive should still be reported"
        assert "rollout_spatial" not in out or "Traceback" not in out, out[-800:]
        print("3b. a run that names two heads and can fetch only one REFUSES "
              "and names the missing tag — #421 rolled the gate alone, went "
              "green, and answered nothing")

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
        head = {"corridor": {"chan_skill": [{"h": 1}], "horizon_auc": 0.5,
                             "horizon_auc_daymatched": 0.5},
                "window": {"horizon_auc": 0.4}}
        ok_m = {"gate": {"pass": True}, "horizon": 12,
                "heads": {"a": head}}
        r1 = check_artefact(ok_m, tmp)
        assert r1.returncode == 0 and "gate: PASSED" in r1.stdout, r1

        bad_m = {"gate": {"pass": None, "skipped": True}, "horizon": 12,
                 "heads": {"a": head}}
        r2 = check_artefact(bad_m, tmp)
        assert r2.returncode != 0, \
            "a MONTHLY roll whose gate was skipped was accepted — the " \
            "certificate is what makes the number believable"

        pentad = {"gate": {"pass": None, "skipped": True, "certified": False,
                           "cadence": "pentad", "reason": "no reference at "
                           "pentad cadence; the monthly one cannot certify it"},
                  "horizon": 73,
                  "cadence": {"name": "pentad", "step_days": 5,
                              "horizon_steps": 73,
                              "horizon_span_days": 365.0,
                              "horizon_daymatched_steps": 73,
                              "horizon_is_daymatched": True,
                              "starts_per_year": 3,
                              "starts_available_per_holdout_year": {
                                  "2009": 73, "2017": 73, "2023": 73},
                              "daymatched_leads": [6, 12, 18, 24, 30, 37, 43,
                                                   49, 55, 61, 67, 73]},
                  "heads": {"a": head}}
        r3 = check_artefact(pentad, tmp)
        assert r3.returncode == 0, r3.stderr[-800:]
        assert "NOT CERTIFIED at pentad cadence" in r3.stdout, r3.stdout
        assert "365" in r3.stdout, r3.stdout

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

        # ---- 6. THE HORIZON IS DERIVED AND PASSED (E-044 item 1) --------
        # Until 2026-08-20 the invocation carried no --horizon at all, so a
        # pentad roll took argparse's literal 12 = 60 DAYS and would have been
        # reported beside the monthly archive's 365. What is checked here is
        # the COMMAND LINE the script builds, captured by a `python` shim that
        # records argv and exits 1 — the script's own flags, not a rehearsal
        # of them. (rc 1 stops it at the roll, which is all this asks about.)
        tmp3, cache3, bin3 = sandbox()
        tpath3 = os.path.join(cache3, "family3_na025.npz")
        make_tensor(tpath3)
        ck3 = os.path.join(cache3, "toy_codec.pt")
        make_ckpt(ck3)
        wh3 = codec_weight_hash(torch.load(ck3, map_location="cpu",
                                           weights_only=False))
        dh3 = data_fingerprint(tpath3)
        np.save(os.path.join(cache3, f"Z_actions_{wh3}_{dh3}.npy"),
                np.zeros((8, 4, 4), np.float16))
        # a pentad tensor beside it, same ocean, so the SAME script call can
        # be asked what it does at each cadence
        pent = os.path.join(cache3, "family4_na025_pentad_r2.npz")
        make_tensor(pent, cadence=5)
        dhp = data_fingerprint(pent)
        np.save(os.path.join(cache3, f"Z_actions_{wh3}_{dhp}.npy"),
                np.zeros((8, 4, 4), np.float16))
        os.makedirs(os.path.join(tmp3, "ml", "runs", "heads"), exist_ok=True)
        open(os.path.join(tmp3, "ml", "runs", "heads", "h1.pt"), "w").write("x")
        argv_log = os.path.join(tmp3, "argv.log")
        real_py = sys.executable
        with open(os.path.join(bin3, "python"), "w") as fh:
            fh.write(
                "#!/bin/sh\n"
                'if [ "$2" = "ml/rollout_spatial.py" ] || '
                '[ "$1" = "ml/rollout_spatial.py" ]; then\n'
                f'  echo "$@" >> {argv_log}\n'
                '  echo "::error::stub roll" >&2\n  exit 1\nfi\n'
                f'exec {real_py} "$@"\n')
        os.chmod(os.path.join(bin3, "python"), 0o755)
        # a `curl` that "downloads" the head, so HPATHS is non-empty
        with open(os.path.join(bin3, "curl"), "w") as fh:
            fh.write('#!/bin/sh\nfor a in "$@"; do case "$a" in '
                     '*/h1__temporal.pt) shift; ;; esac; done\n'
                     'o=""; p=""\nwhile [ $# -gt 0 ]; do case "$1" in '
                     '-o) o="$2"; shift 2;; -*) shift;; *) p="$1"; shift;; '
                     'esac; done\n'
                     'case "$p" in *h1__temporal.pt) echo x > "$o"; exit 0;; '
                     'esac\nexit 22\n')
        os.chmod(os.path.join(bin3, "curl"), 0o755)

        def roll_argv(window, tensor):
            open(argv_log, "w").close()
            r = run(tmp3, bin3, window, {"TENSOR": f"ml/cache/{tensor}"})
            line = open(argv_log).read().strip().splitlines()
            return (line[-1].split() if line else []), r

        ck_rel = os.path.relpath(ck3, tmp3)
        av_m, r_m = roll_argv(f"sroll:h1,ckpt:{ck_rel}", "family3_na025.npz")
        av_p, r_p = roll_argv(f"sroll:h1,ckpt:{ck_rel}",
                              "family4_na025_pentad_r2.npz")
        assert av_m and av_p, (r_m.stdout[-1500:], r_p.stdout[-1500:])
        assert "--horizon" in av_m and "--horizon" in av_p, (av_m, av_p)
        h_m = av_m[av_m.index("--horizon") + 1]
        h_p = av_p[av_p.index("--horizon") + 1]
        assert h_m == "12", ("monthly must still be the archive's 12", av_m)
        assert h_p == "73", ("pentad must be day-matched to 12 months, "
                             "i.e. 73 steps = 365.0 d", av_p)
        assert "--starts-per-year" not in av_m + av_p, \
            "the starts knob must NOT be defaulted — it is a cost decision"
        # item 4: long/future are NOT passed without an override, because the
        # roll's own defaults resolve to 20 years of ITS axis at either cadence
        assert "--long-months" not in av_m + av_p, av_p
        assert "--future-months" not in av_m + av_p, av_p
        assert "365" in r_p.stdout and "60" not in h_p, r_p.stdout[-800:]
        print("6. the script DERIVES the horizon from the tensor and passes "
              "it explicitly: --horizon %s at monthly (the archive's 12 "
              "months, argparse's own default, so the artefact is unchanged) "
              "and --horizon %s at pentad (365.0 d, not the silent 60 d the "
              "missing flag used to buy). --starts-per-year, --long-months "
              "and --future-months are absent, so the roll's cadence-correct "
              "defaults govern" % (h_m, h_p))

        # ---- 7. the tokens that override them (items 2 and 4) -----------
        av, _ = roll_argv(
            f"sroll:h1,ckpt:{ck_rel},horizon:36,starts:3,long:99,future:0",
            "family4_na025_pentad_r2.npz")
        for flag, val in (("--horizon", "36"), ("--starts-per-year", "3"),
                          ("--long-months", "99"), ("--future-months", "0")):
            assert flag in av and av[av.index(flag) + 1] == val, (flag, av)
        _, r_bad = roll_argv(f"sroll:h1,ckpt:{ck_rel},horizon:x",
                             "family4_na025_pentad_r2.npz")
        assert r_bad.returncode != 0 and "whole number" in r_bad.stdout, \
            r_bad.stdout[-600:]
        assert "embed cache key" not in r_bad.stdout, \
            "the bad-token refusal fired after doing work"
        _, r_unk = roll_argv(f"sroll:h1,ckpt:{ck_rel},starts_per_year:3",
                             "family4_na025_pentad_r2.npz")
        assert r_unk.returncode != 0 and "unknown sroll token" in r_unk.stdout
        print("7. `horizon:`/`starts:`/`long:`/`future:` reach the roll as "
              "--horizon 36 --starts-per-year 3 --long-months 99 "
              "--future-months 0; a non-numeric value is refused before any "
              "hashing or extraction (rc %d) and an unknown token is still "
              "refused (rc %d)" % (r_bad.returncode, r_unk.returncode))

        # ---- 8. the dispatch numbers live in the header (item 5) --------
        src = open(os.path.join(ROOT, SCRIPT)).read()
        head = src[:src.index("set -e")]
        for want in ("job_timeout", ">= 1000", ">= 1900", "starts:3",
                     "1,461", "20 YEARS", "UNVERIFIED"):
            assert want in head, (
                f"{want!r} is not in sroll_run.sh's header. A dispatcher "
                f"copying this pattern reads this file and nothing else, so "
                f"the timeout arithmetic and the reason long/future are left "
                f"alone have to be HERE (ml/CLAUDE.md §5.24).")
        print("8. the header carries the dispatch numbers a copier needs: "
              "job_timeout >= 1000 min for one pentad head and >= 1900 for a "
              "pair, why --long-months/--future-months are left to the roll's "
              "own 20-years-of-this-axis defaults (1,461 at pentad), the "
              "starts:3 recommendation, and the UNVERIFIED marker on every "
              "hour of it")

        # ---- 9. `dumproll` — the BARE token, and the artifact that carries
        # what it writes (2026-08-22, Chris: "Save the roll forward sequence
        # for the held out years somewhere ... as animation in the UI").
        # Two ways this token could fail silently, and both are checked: the
        # `*)` arm of the token loop reads anything without a colon as a HEAD
        # TAG, so an unmatched `dumproll` would be fetched from the release,
        # 404 and (since 32e4e06) refuse the whole dispatch; and the workflow's
        # "Upload probe results" step takes a LIST OF NAMES, not a directory,
        # so a dump written into the bundle dir but not named there would be
        # deleted with the box while every step still reported success.
        av_d, r_d = roll_argv(f"sroll:h1,ckpt:{ck_rel},dumproll,starts:3",
                              "family4_na025_pentad_r2.npz")
        assert "--dump-roll" in av_d, av_d
        ddir = av_d[av_d.index("--dump-roll") + 1]
        assert ddir == "ml/runs/actions/roll_dump", ddir
        tail = av_d[av_d.index("--heads") + 1:av_d.index("--out")]
        assert tail == ["ml/runs/heads/h1.pt"], (
            "`dumproll` leaked into the head list — the bare-token arm is "
            "below the catch-all", tail)
        assert "--dump-roll" not in av_p and "--dump-roll" not in av_m, \
            "the dump must be OFF unless the window asks for it"
        wf = open(os.path.join(ROOT, ".github", "workflows",
                               "ml-train.yml")).read()
        up = wf[wf.index("name: Upload probe results"):]
        up = up[:up.index("retention-days")]
        assert ddir + "/*" in up, (
            "the workflow's probe artifact does not name " + ddir +
            "/* — the trajectories would never leave the box")
        print("9. `dumproll` reaches the roll as --dump-roll %s and is absent "
              "from every window that does not name it; the bare token does "
              "not leak into --heads; and the workflow's probe artifact names "
              "%s/* explicitly, which is the only reason the files survive "
              "the box" % (ddir, ddir))

        # ---- 10. `longstart:` — `+`-joined, because `,` splits tokens ----
        # The phase discriminator (2026-08-22). Three ways this could fail
        # silently and all three are checked: the separator (a `,` inside the
        # value would be read as the NEXT token and, having no colon, as a
        # HEAD TAG); the ordering against `long:` (a future `long*:` arm would
        # swallow it); and the shape of the labels, refused HERE rather than
        # discovered by the roll, which skips an unresolvable label with a
        # reason and would leave a dispatch asking for six hindcasts quietly
        # producing five.
        av_l, _ = roll_argv(
            f"sroll:h1,ckpt:{ck_rel},longstart:2004-12+2014-12+2024-03,long:99",
            "family3_na025.npz")
        assert "--long-start" in av_l, av_l
        assert av_l[av_l.index("--long-start") + 1] == \
            "2004-12,2014-12,2024-03", av_l
        assert av_l[av_l.index("--long-months") + 1] == "99", av_l
        assert av_l[av_l.index("--heads") + 1:av_l.index("--out")] == \
            ["ml/runs/heads/h1.pt"], av_l
        assert "--long-start" not in av_m and "--long-start" not in av_p, \
            "an absent longstart: must leave the roll's own single default"
        _, r_lbad = roll_argv(f"sroll:h1,ckpt:{ck_rel},longstart:2004_12",
                              "family3_na025.npz")
        assert r_lbad.returncode != 0 and "YYYY-MM" in r_lbad.stdout, \
            r_lbad.stdout[-600:]
        assert "embed cache key" not in r_lbad.stdout, \
            "the bad-label refusal fired after doing work"
        src_h = open(os.path.join(ROOT, SCRIPT)).read()
        assert src_h.index("longstart:*)") < src_h.index("long:*)"), \
            "`longstart:*` must be matched before `long:*`"
        print("10. `longstart:2004-12+2014-12+2024-03` reaches the roll as "
              "--long-start 2004-12,2014-12,2024-03 (the `+` is because `,` "
              "already splits tokens), composes with long:99, does not leak "
              "into --heads, is absent when the window does not ask for it, "
              "and a malformed label is refused before any hashing (rc %d)"
              % r_lbad.returncode)

        # ---- 11. `longm:`/`futm:` — the same two knobs in MONTHS ---------
        # `--long-months` is a count of AXIS STEPS despite its name, so
        # `long:240` is 20 years at monthly and 3.3 years at pentad. The
        # month-denominated spelling is converted HERE, by the roll's own
        # TimeAxis.steps_for_months, and passed explicitly — the `horizon:`
        # pattern (§5.24: this script does not rescale a flag silently, it
        # computes the step count and says so). At monthly the conversion is
        # the identity, which is what keeps every archived hindcast
        # reproducible; at pentad 240 months is 1,461 steps.
        av_lm, _ = roll_argv(f"sroll:h1,ckpt:{ck_rel},longm:240,futm:12",
                             "family3_na025.npz")
        assert av_lm[av_lm.index("--long-months") + 1] == "240", av_lm
        assert av_lm[av_lm.index("--future-months") + 1] == "12", av_lm
        av_lp, r_lp = roll_argv(f"sroll:h1,ckpt:{ck_rel},longm:240,futm:12",
                                "family4_na025_pentad_r2.npz")
        assert av_lp[av_lp.index("--long-months") + 1] == "1461", av_lp
        assert av_lp[av_lp.index("--future-months") + 1] == "73", av_lp
        assert "long/future in MONTHS" in r_lp.stdout, r_lp.stdout[-800:]
        # absent tokens change nothing: check 6 already asserts neither flag
        # is passed without one, and this is the same window at monthly.
        assert "--long-months" not in av_m and "--future-months" not in av_m
        _, r_both = roll_argv(f"sroll:h1,ckpt:{ck_rel},long:99,longm:240",
                              "family3_na025.npz")
        assert r_both.returncode != 0 and "same flag in two units" in \
            r_both.stdout, r_both.stdout[-600:]
        assert "embed cache key" not in r_both.stdout, \
            "the two-spellings refusal fired after doing work"
        _, r_mbad = roll_argv(f"sroll:h1,ckpt:{ck_rel},futm:x",
                              "family3_na025.npz")
        assert r_mbad.returncode != 0 and "whole number" in r_mbad.stdout, \
            r_mbad.stdout[-600:]
        src_m = open(os.path.join(ROOT, SCRIPT)).read()
        assert src_m.index("longm:*)") < src_m.index("long:*)"), \
            "`longm:*` must be matched before `long:*`"
        print("11. `longm:240,futm:12` reaches the roll as --long-months 240 "
              "--future-months 12 at MONTHLY (steps_for_months is the "
              "identity there, so archived hindcasts are unchanged) and as "
              "--long-months 1461 --future-months 73 at PENTAD, with the "
              "conversion printed; long:+longm: together is refused before "
              "any hashing (rc %d) and a non-numeric value still is (rc %d)"
              % (r_both.returncode, r_mbad.returncode))

        print("\nsroll_run.sh wiring: all 11 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)
        shutil.rmtree(locals().get("tmp3", tmp), ignore_errors=True)


if __name__ == "__main__":
    main()
