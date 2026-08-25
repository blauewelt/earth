#!/usr/bin/env python3
"""The in-training embed-cache push must fire ONCE PER JOB — and in every job.

THE BUG, measured 2026-08-21. `scripts/probes_run.sh` publishes the embedding
cache Z from two places: inside the 30 s monitor loop, every 20 ticks (~10 min)
while ml/temporal.py trains, and unconditionally after `wait $S2_PID`. The
in-training push was guarded by `[ ! -f /tmp/embed-cache-pushed ]` and marked
its success with `touch /tmp/embed-cache-pushed` — a path with nothing job-
specific in it. /tmp persists in the long-lived self-hosted runner container
ACROSS JOBS, and nothing clears that file: scripts/disk_hygiene.sh tier 0
removes /opt/runner/_diag/*.log and /opt/runner/_work/_temp/* and touches
nothing under /tmp. So the FIRST job on a box published its cache and every
later job on the same box was permanently suppressed.

The evidence it was read off:

  #414 (E-035 xl233 seed-1 roll-forward) — 14 h of stage 2 on
  gpu-box-30257785 with the monitor loop demonstrably alive (live metrics
  every 5 min, head snapshot every 30 min), and the string "embed cache"
  appears exactly ONCE in the whole log, at 07:07:29Z, AFTER
  "saved .../temporal.pt" — the post-training push, never the loop.

  #396 (E-035 seed-0 roll-forward) — ~15.5 h of stage 2 on
  gpu-box-45731106, loop alive, pull succeeding, ZERO push lines.

The cost: #427's (E-044 pentad stage-2) Z is 17.43 GB and 8.9 h of a 4090, and
it sat on one rented disk for the length of the run. ml/CLAUDE.md §5.20:
publish a shared artefact when it EXISTS, not when the job ends.

WHAT IS PINNED. The three blocks are LIFTED OUT OF THE SHIPPED SCRIPT (the
tests/test_sroll_wiring.py convention) and run for real, against a stub
embed_cache_sync.py and a stub /opt/earth-cache, so this is the actual shell
the boxes run and not a paraphrase of it:

  1. within one job the push happens ONCE, however many ticks pass;
  2. across two jobs in the SAME container it happens in BOTH — including
     when the legacy /tmp/embed-cache-pushed is lying there from an old job;
  3. the post-training push still runs, unconditionally, after a successful
     in-training push (embed_cache_sync.py no-ops on it: ml/embed_cache_sync.py
     210-216);
  4. a FAILED push writes no marker and is retried at the next window;
  5. every branch is OBSERVABLE — waiting, starting, done, failed each print
     a line naming the call site (ml/CLAUDE.md §4.6/§4.7), and the standing
     conditions print once rather than once per tick;
  6. the marker key is per JOB and cannot silently degrade to a shared path:
     with GITHUB_RUN_ID unset the fallback is the shell's own pid, never the
     bare path that caused this.

    python3 tests/test_embed_cache_push_marker.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(ROOT, "scripts", "probes_run.sh")
SRC = open(SCRIPT).read()

CACHE = "/opt/earth-cache"
LEGACY = "/tmp/embed-cache-pushed"


def lift(pattern, what):
    """Pull a block out of the real script, or fail loudly if it moved."""
    m = re.search(pattern, SRC, re.S)
    if not m:
        raise SystemExit(
            f"FAILED to lift the {what} out of scripts/probes_run.sh — this "
            f"test greps the shipped file on purpose, so a moved block is a "
            f"real failure: re-point the regex at where it went, and check "
            f"the behaviour below still holds there.")
    return m.group(1)


SETUP = lift(r'\n(  EMBED_MARK=.*?\n)  TICK=0\n', "marker setup")
LOOP = lift(r'\n(    if \[ \$\(\(TICK % 20\)\) -eq 0 \]; then\n.*?\n    fi\n)',
            "in-training push block")
POST = lift(r"\n(  echo 'embed cache push: STARTING post-training.*?\n  fi\n)",
            "post-training push block")

STUB = """#!/usr/bin/env python3
# Stands in for ml/embed_cache_sync.py. Records one line per invocation and
# fails the first $FAIL_FIRST calls, so a retry can be observed.
import os, sys
log = os.environ["PUSH_LOG"]
n = sum(1 for _ in open(log)) if os.path.exists(log) else 0
fail = n < int(os.environ.get("FAIL_FIRST", "0"))
open(log, "a").write("call %d rc=%d argv=%s\\n" % (n + 1, 1 if fail else 0,
                                                   " ".join(sys.argv[1:])))
print("stub embed_cache_sync: %s" % ("FAILING" if fail else "publishing"))
sys.exit(1 if fail else 0)
"""


def sandbox():
    """A container: one /tmp that persists across the jobs we run in it."""
    box = tempfile.mkdtemp()
    for d in ("tmp", "cache", "job/ml", "bin"):
        os.makedirs(os.path.join(box, d))
    stub = os.path.join(box, "job", "ml", "embed_cache_sync.py")
    open(stub, "w").write(STUB)
    # `python`, not `python3` — that is what the script invokes, and the boxes
    # have it. The sandbox may not, so shim it.
    shim = os.path.join(box, "bin", "python")
    open(shim, "w").write('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
    os.chmod(shim, 0o755)
    return box


def job(box, ticks, run_id="1000", z_at=1, fail_first=0,
        partial_at=None, done_at=None):
    """Run ONE job's worth of the lifted shell in the container `box`.

    `z_at` is the tick at which /opt/earth-cache/Z_*.npy appears (the embedding
    finishing); the surrounding for-loop is harness, everything indented under
    it is the shipped script.

    `partial_at` instead puts the box in the state ml/temporal.py leaves it in
    WHILE the embedding runs — `Z_*.npy.partial` holding the bytes and
    `Z_*.npy.progress` saying how many rows of it are real — and `done_at` is
    the tick at which that pass finishes: the final name appears with its
    `.done` beside it and both progress files go. Pass `z_at=0` to keep the
    finished-cache-with-no-markers state out of a partial scenario.
    """
    cache, tmp = os.path.join(box, "cache"), os.path.join(box, "tmp")

    def here(block):
        """The shipped block, with the two absolute paths pointed at this
        container. Both literals must still be there — a cache or marker that
        moved is a change this test has to see."""
        assert CACHE in block or LEGACY in block, block
        return block.replace(CACHE, cache).replace(
            LEGACY, os.path.join(tmp, "embed-cache-pushed"))

    s, lp, po = here(SETUP), here(LOOP), POST.replace(CACHE, cache)
    z = f"{cache}/Z_run_deadbeef01.npy"
    prog = f"""set -e
{s}
for TICK in $(seq 1 {ticks}); do
  if [ "$TICK" = "{z_at}" ]; then : > "{z}"; fi
  if [ "$TICK" = "{partial_at}" ]; then : > "{z}.partial"; : > "{z}.progress"; fi
  if [ "$TICK" = "{done_at}" ]; then rm -f "{z}.partial" "{z}.progress"; : > "{z}"; : > "{z}.done"; fi
{lp}
done
{po}
echo "MARKER=$EMBED_MARK"
"""
    p = os.path.join(box, "job", "harness.sh")
    open(p, "w").write(prog)
    env = dict(os.environ)
    env["PATH"] = os.path.join(box, "bin") + os.pathsep + env["PATH"]
    env["PUSH_LOG"] = os.path.join(box, "pushes.log")
    env["TENSOR"] = "ml/cache/family4_na025_pentad.npz"
    env["FAIL_FIRST"] = str(fail_first)
    if run_id is None:
        env.pop("GITHUB_RUN_ID", None)
    else:
        env["GITHUB_RUN_ID"] = run_id
    r = subprocess.run(["bash", p], cwd=os.path.join(box, "job"), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    return r.stdout


def pushes(box):
    p = os.path.join(box, "pushes.log")
    return open(p).read().splitlines() if os.path.exists(p) else []


def main():
    ok = 0

    # ---- 0. the bare, container-wide marker is GONE from the guard --------
    for bad in (f"[ ! -f {LEGACY} ]", f"touch {LEGACY}"):
        if bad in SRC:
            raise SystemExit(
                f"0 FAILED: scripts/probes_run.sh still uses the bare marker "
                f"({bad!r}). /tmp survives across jobs in the runner container, "
                f"so a fixed path means the first job on a box is the only one "
                f"that ever publishes Z.")
    if "EMBED_MARK=" not in SRC or "GITHUB_RUN_ID" not in SETUP:
        raise SystemExit("0 FAILED: the marker is no longer keyed by the run id")
    if f"rm -f {LEGACY}" not in SRC:
        raise SystemExit("0 FAILED: the legacy marker is no longer cleaned up; "
                         "leave the rm so a stale file cannot mislead a reader")
    print("0. ok — the guard is keyed by $GITHUB_RUN_ID, the legacy path is "
          "only ever removed")
    ok += 1

    # ---- 1. ONE push per job, however long the job runs -------------------
    box = sandbox()
    out = job(box, ticks=60, run_id="41400", z_at=1)
    intraining = [l for l in out.splitlines() if "STARTING in-training" in l]
    assert len(intraining) == 1, out
    assert "at tick 20" in intraining[0], intraining
    assert out.count("DONE in-training") == 1, out
    # ticks 40 and 60 are silent: the marker suppresses them and the DONE
    # line above already told the reader why. One line per STATE, not per tick.
    assert out.count("embed cache push:") == 4, out
    assert len(pushes(box)) == 2, pushes(box)      # in-training + post
    print("1. ok — within one job the push fires once (tick 20); the marker "
          "suppresses ticks 40 and 60 and the log stays four lines long")
    ok += 1

    # ---- 1b. a RE-ENTRY under the same run id (a step retry on the same
    # box) finds its own marker, says so once, and does not re-upload -------
    out1b = job(box, ticks=60, run_id="41400", z_at=1)
    assert out1b.count("already published by this job") == 1, out1b
    assert "STARTING in-training" not in out1b, out1b
    assert len(pushes(box)) == 3, pushes(box)      # only the post-training one
    print("1b. ok — a retry under the same run id skips the in-training push, "
          "names the marker that suppressed it, and still runs the post one")
    ok += 1

    # ---- 2. and the post-training push STILL runs -------------------------
    lines = [l for l in out.splitlines() if "embed cache push:" in l]
    order = [l.split("embed cache push: ")[1][:24] for l in lines]
    assert "STARTING post-training" in out and "DONE post-training" in out, out
    assert order[-1].startswith("DONE post-training"), order
    assert pushes(box)[1].startswith("call 2 rc=0"), pushes(box)
    print("2. ok — the post-training push is unconditional and still runs "
          "after a successful in-training one (embed_cache_sync no-ops it)")
    ok += 1

    # ---- 3. THE BUG: a second job in the SAME container -------------------
    # exactly the state the old code left behind, plus the legacy marker an
    # old job would have dropped there.
    open(os.path.join(box, "tmp", "embed-cache-pushed"), "w").close()
    before = len(pushes(box))
    out2 = job(box, ticks=20, run_id="42700", z_at=1)  # a DIFFERENT run id
    assert "STARTING in-training" in out2, out2
    assert len(pushes(box)) == before + 2, pushes(box)
    assert not os.path.exists(os.path.join(box, "tmp", "embed-cache-pushed")), \
        "the legacy marker should have been removed"
    assert os.path.exists(os.path.join(box, "tmp", "embed-cache-pushed.42700"))
    assert os.path.exists(os.path.join(box, "tmp", "embed-cache-pushed.41400"))
    print("3. ok — a second job on the same box publishes too; job 1's marker "
          "and the legacy container-wide file do not suppress it")
    ok += 1

    # ---- 4. a failed push writes no marker and is retried -----------------
    box2 = sandbox()
    out3 = job(box2, ticks=60, run_id="99", z_at=1, fail_first=2)
    assert out3.count("FAILED in-training") == 2, out3
    assert "at tick 20 " in out3 and "at tick 40 " in out3, out3
    assert out3.count("DONE in-training") == 1, out3
    assert os.path.exists(os.path.join(box2, "tmp", "embed-cache-pushed.99"))
    assert len(pushes(box2)) == 4, pushes(box2)    # 20, 40, 60, post
    print("4. ok — a failure writes no marker, warns with its tick, and the "
          "next ~10-min window retries")
    ok += 1

    # ---- 5. nothing to publish yet is SAID, once --------------------------
    box3 = sandbox()
    out4 = job(box3, ticks=60, run_id="7", z_at=45)
    assert out4.count("nothing to publish") == 1, out4
    assert "yet at tick 20" in out4, out4
    assert "STARTING in-training at tick 60" in out4, out4
    assert len(pushes(box3)) == 2, pushes(box3)
    print("5. ok — 'no Z yet' prints once with its tick, and the push fires at "
          "the first window after the embedding lands")
    ok += 1

    # ---- 6. an unset run id degrades to per-PROCESS, never to shared ------
    box4 = sandbox()
    out5 = job(box4, ticks=20, run_id=None, z_at=1)
    mark = [l for l in out5.splitlines() if l.startswith("MARKER=")][0][7:]
    assert re.fullmatch(re.escape(os.path.join(box4, "tmp"))
                        + r"/embed-cache-pushed\.\d+", mark), mark
    assert not mark.endswith("embed-cache-pushed"), \
        ("an unset GITHUB_RUN_ID must NOT fall back to the container-wide "
         "path — that is the bug, restored silently", mark)
    print("6. ok — with GITHUB_RUN_ID unset the key falls back to the shell's "
          "pid, so the worst case is one extra idempotent push")
    ok += 1

    # ---- 7. THE PARTIAL CADENCE: publish DURING the embedding -------------
    # Chris, 2026-08-25: "Publishing should happen 'during' the embedding
    # computation ... A new job that needs the same embedding can choose to
    # continue the computation (if 32/100 are already complete it will start
    # with chunk 33)." So while `.progress` is the only marker on the box,
    # every ~10-min window ships the finished chunks and NOTHING writes the
    # job marker: the loop must keep going until the cache is complete and a
    # full push has verified it durable.
    box5 = sandbox()
    out6 = job(box5, ticks=60, run_id="500", z_at=0, partial_at=1)
    assert out6.count("STARTING PARTIAL in-training") == 3, out6
    assert out6.count("PARTIAL done") == 3, out6
    assert "STARTING in-training" not in out6.replace("STARTING PARTIAL "
                                                      "in-training", ""), out6
    calls = pushes(box5)
    assert len(calls) == 4, calls               # 20, 40, 60, and the post one
    assert all("--partial" in c for c in calls[:3]), calls
    assert not os.path.exists(os.path.join(box5, "tmp",
                                           "embed-cache-pushed.500")), (
        "a PARTIAL publish must never write the marker that stops the loop — "
        "the cache is not complete and the release must not be left claiming "
        "it is")
    print("7. ok — while only .progress is on the box the loop pushes "
          "--partial every window and writes no marker")
    ok += 1

    # ---- 8. …and it STOPS the moment the full push verifies durable -------
    box6 = sandbox()
    out7 = job(box6, ticks=100, run_id="501", z_at=0, partial_at=1, done_at=45)
    assert out7.count("STARTING PARTIAL in-training") == 2, out7   # 20, 40
    assert "the embedding is COMPLETE (.done)" in out7, out7
    assert out7.count("VERIFIED durable") == 1, out7
    # ticks 80 and 100 are SILENT, not repetitive: EMBED_STATE is already
    # `published` from the DONE line at tick 60, and one line per STATE is the
    # rule this loop has kept since 2026-08-21.
    assert out7.count("already published by this job") == 0, out7
    calls = pushes(box6)
    assert len(calls) == 4, calls               # 20, 40 partial; 60 full; post
    assert "--partial" not in calls[2], calls   # the full push at tick 60
    assert os.path.exists(os.path.join(box6, "tmp", "embed-cache-pushed.501"))
    print("8. ok — .done switches the loop to the full push, whose success "
          "writes the marker and silences the rest of the run")
    ok += 1

    for b in (box, box2, box3, box4, box5, box6):
        shutil.rmtree(b, ignore_errors=True)
    print(f"\nall {ok} embed-cache push guards hold")


if __name__ == "__main__":
    main()
