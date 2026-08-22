#!/usr/bin/env python3
"""`--dump-roll`: the rolled state leaves the box, and NOTHING ELSE MOVES.

Chris, 2026-08-22: *"Roll forward all of the earth's pixels (these are
required by the stencil size, not just the relevant area)"* and *"Save the
roll forward sequence for the held out years somewhere (so that we can use it
as animation in the UI)"*.

The first half was already true — `roll_step` advances all P window ocean
pixels every step and the gate/corridor/window scopes are masks applied to the
DECODED field afterwards — so the work is the second half: keep the state
instead of discarding it one step after it exists. This test runs the REAL
evaluator on the pentad toy TWICE, with the same flags but for `--dump-roll`,
and asks the two questions that matter:

  1. **Does the roll itself move?** The two roll JSONs are compared BYTE FOR
     BYTE (with only the wall-clock readings stripped, exactly as
     tests/test_roll_monthly_identity.py strips them). A dump that perturbs
     the numbers it is a record of would be worse than no dump: the animation
     and the archived skill would be of two different rolls, and nothing in
     either file would say so. This is the stronger form of "token absent →
     bit-identical", because it holds with the flag ON.
  2. **Is what lands on disk actually the rolled state?** Not "a file exists"
     (§0.2) — every trajectory's state 0 must equal the TRUE embedding of its
     start row bit for bit, its length must equal the horizon the year
     actually granted that start, its pixel map must be the Z's own pixel
     order, and its codec identity must be `temporal.codec_weight_hash` of the
     checkpoint the roll was given. Those are what make the file decodable by
     somebody who was not here.

Plus the exclusion: the GATE head is named to the roll and must NOT be
dumped — it is a certificate, re-rolled in every wave, and nobody animates it.

    python3 tests/test_roll_dump.py

No GPU: the same 79-pixel synthetic pentad ocean the cadence test rolls.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)
from test_rollout_spatial import build_fixture, K                # noqa: E402
from test_roll_monthly_identity import WALL                      # noqa: E402

HORIZON, STARTS, N_LONG, N_FUT = 3, 3, 12, 4
HOLD_Y = "1991"                       # the pentad fixture's holdout year


def run(f, out, cache, extra=()):
    os.makedirs(cache, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ML + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-u", os.path.join(ML, "rollout_spatial.py"),
           "--x", f["x"], "--npz-small", f["npz"], "--z", f["z"],
           "--ckpt", f["ckpt"], "--out", out,
           "--horizon", str(HORIZON), "--starts-per-year", str(STARTS),
           "--long-start", "1990-07", "--long-months", str(N_LONG),
           "--future-months", str(N_FUT), "--cache-dir", cache,
           *extra, "--heads", *f["heads"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                       env=env, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"rollout_spatial.py failed (rc {r.returncode})")
    return open(out).read(), r.stdout


def main():
    tmp = tempfile.mkdtemp()
    try:
        f = build_fixture(tmp, cadence_days=5)
        # The gate head, by NAME — that is how rollout_spatial identifies it
        # (`GATE_HEAD in os.path.basename`), and it is a pentad roll, so no
        # monthly reference is applied to it (GATE_REF_BY_CADENCE).
        gate_hp = os.path.join(tmp, "e017_u1_s0.pt")
        shutil.copyfile(f["heads"][0], gate_hp)
        f["heads"] = [gate_hp, f["heads"][1]]
        main_head = os.path.basename(f["heads"][1])

        import rollout_spatial as rs
        from temporal import codec_weight_hash
        import torch
        ax = rs.TimeAxis(np.load(f["npz"], allow_pickle=False))
        Zm = np.load(f["z"], mmap_mode="r")
        ck = torch.load(f["ckpt"], map_location="cpu", weights_only=False)
        whash = codec_weight_hash(ck)
        starts = ax.starts_for_year(HOLD_Y, STARTS)
        ocean = f["ocean"]
        ys, xs = np.where(ocean)
        P, T = f["P"], f["T"]

        # ---- 1. the roll JSON does not move ------------------------------
        no_dump, _ = run(f, os.path.join(tmp, "a.json"),
                         os.path.join(tmp, "ca"))
        ddir = os.path.join(tmp, "roll_dump")
        with_dump, log = run(f, os.path.join(tmp, "b.json"),
                             os.path.join(tmp, "cb"), ("--dump-roll", ddir))
        a_s, b_s = WALL.sub("", no_dump), WALL.sub("", with_dump)
        n_wall = len(WALL.findall(no_dump))
        assert n_wall and n_wall == len(WALL.findall(with_dump))
        assert a_s == b_s, "the roll JSON MOVED under --dump-roll"
        assert len(a_s) == len(b_s)
        print("1. the roll JSON is BYTE-IDENTICAL with and without "
              "--dump-roll (%d bytes, %d `wall_s` clock readings stripped "
              "from each) — the dump is a record of the roll, not a "
              "participant in it" % (len(a_s), n_wall))

        # ---- 2. one file per (main head, scored start), and only those ----
        man = json.load(open(os.path.join(ddir, "dump_manifest.json")))
        files = man["files"]
        npzs = sorted(p for p in os.listdir(ddir) if p.endswith(".npz"))
        assert len(npzs) == len(files) == len(starts), (npzs, starts)
        assert {e["head_file"] for e in files} == {main_head}, files
        assert not any("e017_u1_s0" in p for p in npzs), npzs
        assert "--dump-roll skips the gate head" in log
        assert {int(e["start_row"]) for e in files} == set(starts)
        assert all(e["year"] == HOLD_Y for e in files)
        print("2. %d trajectories, one per scored start of holdout %s (rows "
              "%s), all from the MAIN head %s — the gate head was named to "
              "the roll and skipped, and says so in the log"
              % (len(npzs), HOLD_Y, [e["start_row"] for e in files],
                 main_head))

        # ---- 3. shapes, dtype, and the length the YEAR actually granted ---
        for e in files:
            s = int(e["start_row"])
            d = np.load(os.path.join(ddir, e["file"]), allow_pickle=False)
            z = d["z"]
            want_n = rs.scored_horizon(ax, s, HORIZON, T, HOLD_Y) + 1
            assert z.dtype == np.float16, z.dtype
            assert z.shape == (want_n, P, ck["d_z"]), (z.shape, want_n, P)
            assert list(e["shape"]) == list(z.shape) and e["n_states"] == want_n
            assert e["bytes"] == os.path.getsize(os.path.join(ddir, e["file"]))
            # STATE 0 IS THE TRUTH, bit for bit: the roll's own input window
            # ends at row s, so an animation that starts anywhere else is
            # starting from a state the model was never given.
            assert np.array_equal(z[0], np.asarray(Zm[s])), e["file"]
            assert not np.array_equal(z[1], z[0]), "the roll did not move"
            assert np.isfinite(z.astype(np.float32)).all(), e["file"]
            # rows/labels/dates advance ONE AXIS ROW per state
            rows = d["rows"]
            assert list(rows) == [s + k for k in range(want_n)]
            assert list(d["labels"]) == [ax.label_of_row(r) for r in rows]
            assert list(d["dates"]) == [ax.date_of_row(r).isoformat()
                                        for r in rows]
            assert (np.diff([np.datetime64(x) for x in d["dates"]])
                    == np.timedelta64(5, "D")).all(), d["dates"]
        print("3. every trajectory is [n_states, %d px, d_z %d] float16 with "
              "n_states = the horizon that START's year actually granted it "
              "(+1), state 0 == the TRUE Z of the start row bit for bit, and "
              "its dates advance one 5-day bin per state"
              % (P, ck["d_z"]))

        # ---- 4. the file is decodable by somebody who was not here --------
        d0 = np.load(os.path.join(ddir, files[0]["file"]), allow_pickle=False)
        meta = json.loads(str(d0["meta_json"]))
        assert meta["codec"]["weight_hash"] == whash == \
            man["codec"]["weight_hash"], (meta["codec"], whash)
        assert meta["head_file"] == main_head and meta["K"] == K
        assert meta["cadence"] == "pentad" and meta["step_days"] == 5.0
        assert np.array_equal(d0["px_y"], ys.astype(np.int32))
        assert np.array_equal(d0["px_x"], xs.astype(np.int32))
        lats = np.load(f["npz"])["lats"]
        lons = np.load(f["npz"])["lons"]
        assert np.allclose(d0["px_lat"], lats[ys])
        assert np.allclose(d0["px_lon"], lons[xs])
        assert d0["z"].shape[1] == len(d0["px_y"]) == P
        assert list(d0["grid_shape"]) == list(ocean.shape)
        print("4. each file carries its own pixel map (px_y/px_x/px_lat/"
              "px_lon over %d px, the Z's own order) and the codec identity "
              "%s — the #10/#11 failure is a wrong-codec decode, and a "
              "trajectory that cannot name its codec is one" % (P, whash))

        # ---- 5. the manifest describes exactly what is on the disk --------
        assert man["n_px"] == P and man["d_z"] == ck["d_z"]
        assert man["dtype"] == "float16" and man["cadence"] == "pentad"
        assert man["config"]["horizon"] == HORIZON
        assert man["config"]["starts_per_year"] == STARTS
        assert man["config"]["hold_years"] == [HOLD_Y]
        assert man["total_bytes"] == sum(
            os.path.getsize(os.path.join(ddir, e["file"])) for e in files)
        assert set(os.listdir(ddir)) == set(npzs) | {"dump_manifest.json"}
        for e in files:
            assert e["dates"][0] == ax.date_of_row(e["rows"][0]).isoformat()
            assert e["dates"][1] == ax.date_of_row(e["rows"][1]).isoformat()
        print("5. dump_manifest.json lists %d files, %s bytes total — the "
              "sum of what is actually there — with shapes, rows and the "
              "first/last DATE of each, and the directory holds nothing else"
              % (len(files), f"{man['total_bytes']:,}"))

        # ---- 6. no dump directory at all when the flag is absent ----------
        stray = [os.path.join(r, p) for r, _, ps in os.walk(tmp) for p in ps
                 if (p.startswith("roll_") and p.endswith(".npz"))
                 or p == "dump_manifest.json"]
        assert sorted(stray) == sorted(
            os.path.join(ddir, p) for p in npzs + ["dump_manifest.json"]), \
            stray
        print("6. the ONLY trajectory files and the only manifest under the "
              "whole fixture tree are the %d in %s — the run without the flag "
              "wrote nothing anywhere (its cache dir holds ocean_mask.npy and "
              "std_stats.npz, which every run writes)"
              % (len(npzs), os.path.basename(ddir)))

        # ---- 7. the FILENAME survives an artifact upload ------------------
        # Measured on #433: the roll finished, ml-metrics archived, and
        # actions/upload-artifact@v4 refused the whole path list with "The
        # path for one of the files in artifact is not valid" — because a
        # stencil+ring head label is `s145rspiral:111-4444-0.71-0.5_s0` and
        # the COLON went into the filename. 2.4 GB stranded on a rented box
        # for one character. The label is not reachable from the toy head's
        # args without building a real ring geometry, so the writer is driven
        # DIRECTLY here with the label #433 actually produced.
        assert all(re.fullmatch(r"[A-Za-z0-9._-]+", p) for p in npzs), npzs
        real_label = "s145rspiral:111-4444-0.71-0.5_s0"
        for bad in ('"', ":", "<", ">", "|", "*", "?", "\r", "\n", " ", "/"):
            assert bad not in rs._fs_safe(f"a{bad}b"), bad
        assert rs._fs_safe(real_label) == "s145rspiral-111-4444-0.71-0.5_s0"
        assert rs._fs_safe("plain_s0.v2-x") == "plain_s0.v2-x", \
            "a label that is already safe must not be rewritten"
        d2 = os.path.join(tmp, "dump2")
        dumper = rs.RollDump(d2, ax, ys, xs, lats, lons, ocean.shape,
                             f["ckpt"], ck, {"probe": True})
        s0 = starts[0]
        dumper.write(real_label, "head-weights-e044b.pt", {"stencil": 145},
                     HOLD_Y, s0, np.zeros((2, P, ck["d_z"]), np.float16))
        got = [p for p in os.listdir(d2) if p.endswith(".npz")]
        assert len(got) == 1 and re.fullmatch(r"[A-Za-z0-9._-]+", got[0]), got
        assert got[0] == f"roll_s145rspiral-111-4444-0.71-0.5_s0_{HOLD_Y}_r{s0}.npz"
        man2 = json.load(open(os.path.join(d2, "dump_manifest.json")))
        assert man2["files"][0]["head"] == real_label, man2["files"][0]
        assert man2["files"][0]["file"] == got[0]
        assert "filename_rule" in man2 and "upload-artifact" in \
            man2["filename_rule"]
        meta2 = json.loads(str(np.load(os.path.join(d2, got[0]),
                                       allow_pickle=False)["meta_json"]))
        assert meta2["head"] == real_label, meta2["head"]
        print("7. a head label carrying a COLON (%s — #433's own) writes as "
              "%s: every basename matches ^[A-Za-z0-9._-]+$, which is what "
              "upload-artifact will accept, while the manifest's files[].head "
              "and the npz's own meta_json still carry the ORIGINAL label, so "
              "nothing about the file's identity is lost to its name"
              % (real_label, got[0]))

        print("\nroll-forward sequence dump: all 7 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
