#!/usr/bin/env python3
"""`window: publishembed` — publish an EXISTING Z, training nothing.

Why the mode exists. `ml/embed_cache_sync.py push` was reachable from exactly
two places, both inside `scripts/probes_run.sh`'s stage-2 block — so the only
way to publish an embed cache that was already sitting on a box's disk was to
dispatch a job that TRAINS: ~1.5-2 h, temporal.py's ~85 GB `nan_to_num` peak,
and a head nobody wanted. E-044's pentad Z is **16.24 GiB and 8.5 h of a 4090**
(#423), was never published, and while it lives on one rented disk the seed-1
arm and the pentad `sroll:` are pinned to that one box (E-044 §4). `publishtensor`
had solved exactly this problem for the tensor since 2026-08-11; this is its
other half.

WHAT IS PINNED, by running the SHIPPED script with a stub `python` that records
its argv (the tests/test_sroll_wiring.py convention — what is asserted is the
command line the boxes will actually run):

  1. `publishembed` calls `ml/embed_cache_sync.py push --run actions --data
     ml/cache/<tensor>.npz` — the tensor is part of the cache's IDENTITY
     (`cache_name` = codec_weight_hash + data_fingerprint), so a mode that
     published without it would publish under a name that means less than it
     claims;
  2. it TRAINS NOTHING and PROBES NOTHING — no temporal.py, no probe_kfold.py,
     no probe_sequence.py, and it exits 0 before the ladder;
  3. a FAILED push does not take the step down and does not go silent: the
     caller's `|| echo "::warning::"` is what makes best-effort a decision of
     the CALLER (ml/CLAUDE.md §7 — the omission of exactly this once cost #131
     its whole probe ladder), and the run still exits 0;
  4. the step SAYS WHAT TO VERIFY. It goes green on a failed publish by
     construction, so the log names the two lines that mean the release
     actually gained the cache (§0.2);
  5. the anomaly gate still governs it, like every other mode in this file:
     `anomaly != true` exits before the case statement, so a dispatch that
     forgets that input publishes nothing and says so.

    python3 tests/test_publishembed_mode.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = "scripts/probes_run.sh"
TENSOR = "family4_na025_pentad_r2"


def sandbox(push_rc=0):
    """A repo-shaped temp dir with a `python` that records argv instead of
    running anything, and answers `embed_cache_sync.py push` with `push_rc`.

    `tensor-t` is answered with a number, because it is answered with a number
    on a box: it is the same script reading ~128 bytes of the tensor's own
    .npy header, and what the mode does with it — pass it to push as
    --expect-t — is the thing under test."""
    tmp = tempfile.mkdtemp()
    os.symlink(os.path.join(ROOT, "scripts"), os.path.join(tmp, "scripts"))
    os.makedirs(os.path.join(tmp, "ml"))
    for name in os.listdir(os.path.join(ROOT, "ml")):
        if name in ("cache", "__pycache__"):
            continue
        os.symlink(os.path.join(ROOT, "ml", name),
                   os.path.join(tmp, "ml", name))
    os.makedirs(os.path.join(tmp, "ml", "cache"))
    open(os.path.join(tmp, "ml", "cache", f"{TENSOR}.npz"), "w").write("x")
    bin_ = os.path.join(tmp, "bin")
    os.makedirs(bin_)
    log = os.path.join(tmp, "argv.log")
    with open(os.path.join(bin_, "python"), "w") as f:
        f.write("#!/bin/sh\n"
                f'echo "$@" >> {log}\n'
                'for a in "$@"; do case "$a" in tensor-t)\n'
                '  echo 3142\n  exit 0;; esac; done\n'
                'for a in "$@"; do case "$a" in *embed_cache_sync.py)\n'
                '  echo "embed cache for codec deadbeef01 is now durable"\n'
                f'  exit {push_rc};; esac; done\nexit 0\n')
    os.chmod(os.path.join(bin_, "python"), 0o755)
    return tmp, bin_, log


def run(tmp, bin_, window, env=None):
    e = dict(os.environ)
    e["PATH"] = bin_ + os.pathsep + e["PATH"]
    e["GITHUB_REPOSITORY"] = "blauewelt/earth"
    e["GITHUB_RUN_NUMBER"] = "999"
    e["IN_TENSOR"] = TENSOR
    e["IN_ANOMALY"] = "true"
    e["WINDOW"] = window
    e.update(env or {})
    return subprocess.run(["bash", SCRIPT], cwd=tmp, env=e,
                          capture_output=True, text=True, timeout=300)


def main():
    tmps = []
    try:
        # ---- 1/2. the call, and nothing else ----------------------------
        tmp, bin_, log = sandbox()
        tmps.append(tmp)
        r = run(tmp, bin_, "publishembed")
        assert r.returncode == 0, (r.returncode, r.stdout[-2000:])
        calls = [ln.split() for ln in open(log).read().splitlines()]
        sync = [c for c in calls if "ml/embed_cache_sync.py" in c]
        assert len(sync) == 2, calls
        # ASK THE TENSOR ITS SHAPE, THEN PUBLISH AGAINST IT. Run #462
        # published a strided Z under the unstrided key from this very mode's
        # sibling loop; the mode that exists to publish a single-copy Z is the
        # last place that should be allowed to publish the wrong one.
        assert sync[0][sync[0].index("ml/embed_cache_sync.py") + 1] == "tensor-t"
        assert sync[0][sync[0].index("--data") + 1] == f"ml/cache/{TENSOR}.npz"
        c = sync[1]
        assert c[c.index("ml/embed_cache_sync.py") + 1] == "push", c
        assert c[c.index("--run") + 1] == "actions", c
        assert c[c.index("--data") + 1] == f"ml/cache/{TENSOR}.npz", c
        assert c[c.index("--expect-t") + 1] == "3142", c
        assert len(calls) == 2, ("publishembed ran something else too", calls)
        for forbidden in ("temporal.py", "probe_kfold.py", "probe_sequence.py",
                          "probe_head.py", "dip_check.py"):
            assert forbidden not in open(log).read(), forbidden
        print("1. publishembed reads the tensor's T (`tensor-t --data "
              "ml/cache/%s.npz` -> 3142) and then calls exactly `push --run "
              "actions --data ml/cache/%s.npz --expect-t 3142` — the tensor "
              "is part of the cache's identity AND of its shape, so the mode "
              "can publish neither under a name nor at a size that means "
              "less than it claims" % (TENSOR, TENSOR))
        print("2. and NOTHING else runs: no temporal.py, no probe ladder, "
              "two subprocesses in the whole step, both the sync script "
              "(rc %d)" % r.returncode)

        # ---- 3/4. a failed push is loud, and does not kill the step ------
        tmp2, bin2, log2 = sandbox(push_rc=1)
        tmps.append(tmp2)
        r2 = run(tmp2, bin2, "publishembed")
        assert r2.returncode == 0, (
            "a failed push took the step down — `bash -e` plus a callee that "
            "exits 1 is exactly what cost #131 its probe ladder",
            r2.returncode, r2.stdout[-1500:])
        assert "::warning::embed cache publish failed" in r2.stdout, \
            r2.stdout[-1500:]
        assert "remains single-copy" in r2.stdout
        assert "is now durable" in r.stdout, r.stdout[-1500:]
        for want in ("VERIFY THE EFFECT", "is now durable",
                     "already published and complete"):
            assert want in r.stdout, want
        print("3. a push that exits 1 leaves the step at rc 0 with "
              "`::warning::embed cache publish failed ... remains "
              "single-copy` — best-effort is the CALLER's decision, never a "
              "lie told by the callee")
        print("4. the step names what to verify before it runs anything: the "
              "log tells the reader to look for 'is now durable' or 'already "
              "published and complete', because this step goes GREEN on a "
              "failed publish by construction")

        # ---- 5. the anomaly gate still governs it ------------------------
        tmp3, bin3, log3 = sandbox()
        tmps.append(tmp3)
        r3 = run(tmp3, bin3, "publishembed", {"IN_ANOMALY": "false"})
        assert r3.returncode == 0 and not os.path.exists(log3), r3.stdout
        assert "anomaly-space only, skipping" in r3.stdout, r3.stdout[-800:]
        print("5. `anomaly: false` still exits before the mode switch, "
              "publishing nothing and saying so — the gate is above the case "
              "statement and this mode does not sneak under it")

        print("\npublishembed window mode: all 5 checks hold ✓")
    finally:
        for t in tmps:
            shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    main()
