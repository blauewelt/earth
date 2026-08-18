#!/usr/bin/env python3
"""Stage 2 and the sequence probe must not train on all-zero channels.

THE BUG THIS PINS. `ml/temporal.py` and `ml/probe_sequence.py` each carried a
hand-inlined copy of the anomaly transform, both frozen at the pre-2026-08-17
arithmetic:

    v = X[..., c][np.isfinite(X[..., c]) & ~t_hold[...] & ~x_hold[...]]
    X[..., c] = (X[..., c] - v.mean()) / (v.std() + 1e-6)

numpy upcasts the accumulator for `np.mean` on float16 but NOT for
`np.std`/`np.var` (`_methods._var` upcasts integer and bool only). The z-score
sums the squared residuals of the whole train pool; on a float16 tensor that
total passes 65504, `v.std()` returns `inf`, and

    (X - mu) / (inf + 1e-6)  ==  0.0     exactly, for every entry

so EVERY DYNAMIC CHANNEL BECOMES ZEROS. Nothing downstream says so: the loss
is finite, gpu_util is normal, the probe returns a number. Families 4 (pentad)
and 5 (daily) are float16; family 3 was float32 and never reached the limit,
which is the only reason no published number came through this path.

WHAT IT ACTUALLY EXERCISES. Not a copy of the transform -- the real one, in
place, by running each script's `main()` on a float16 fixture and intercepting
`codec_from_ckpt`, the call each makes IMMEDIATELY after the transform. The
interception reads `X` out of the caller's frame, so the test is indifferent
to whether the transform is inlined (old) or a call (new) and runs unchanged
against both. That is the point: it is a test that FAILS on the old code.

Measured on this fixture (2026-08-18):
    old code, float16: temporal std 0.000000, exactly-zero 100.0%
    new code, float16: temporal std 1.000424, exactly-zero  0.0%

Check 0 proves the fixture is IN THE OVERFLOW REGIME rather than assuming it
from the element count -- tests/test_e038_anomaly_dtype.py records a first
version whose pool summed to ~45k and passed against the broken code.

Check 4 pins the OTHER half of the same commit: both scripts used to open the
tensor with `np.load(...)["X"].copy()` and so could not open family 5's
sidecar layout at all (measured on the old code: `KeyError: 'X is not a file
in the archive'`, both scripts). They now go through `tensor_io.load_tensor`,
which returns a READ-ONLY memmap, so each has to take a scratch copy -- and
must not write through to the stored tensor, and must not leave a 166 GB
orphan behind. That runs in a subprocess (tests/_sidecar_driver.py), because
the scratch is removed by `atexit`.

    python3 tests/test_stage2_float16_anomaly.py
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.abspath(os.path.join(HERE, "..", "ml"))
sys.path.insert(0, ML)

# Big enough that the sum of squared residuals over the train pool clears
# float16's 65504 -- the same shape signature as family 4/5, small in H/W and
# long enough in T to build a real 12-month climatology.
T, H, W, C = 120, 90, 90, 4
STATIC = 3                      # channels 0..2 dynamic, 3 a baked constant
HOLD_YEARS = "1990"
HOLD_LON = "-45,-35"


def fixture(dtype, tmp):
    """(tensor path, moy, t_hold, x_hold) with the family-4 signature: a
    seasonal cycle plus unit-variance noise on the dynamic channels, a
    time-invariant field on the static one, and a land block that is NaN at
    every timestep."""
    rng = np.random.default_rng(11)
    months = ["%04d-%02d" % (1982 + i // 12, i % 12 + 1) for i in range(T)]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    lats = np.linspace(20.0, 35.0, H).astype(np.float32)
    lons = np.linspace(-80.0, -13.0, W).astype(np.float32)

    X = np.empty((T, H, W, C), dtype=dtype)
    field = rng.standard_normal((H, W))
    season = np.sin(2 * np.pi * moy / 12)[:, None, None]
    for c in range(C):
        if c == STATIC:
            X[..., c] = np.broadcast_to(field, (T, H, W)).astype(dtype)
        else:
            X[..., c] = (season + rng.standard_normal((T, H, W))).astype(dtype)
    X[:, :4, :4, :] = np.nan                       # land, missing at all t

    path = os.path.join(tmp, "fixture_%s.npz" % np.dtype(dtype).name)
    np.savez(path, X=X, months=np.array(months), lats=lats, lons=lons,
             chan=np.array(["c%d" % c for c in range(C)]),
             rapid=np.zeros((0, 2), np.float32))
    t_hold = np.array([m[:4] in set(HOLD_YEARS.split(",")) for m in months])
    lo, hi = (float(v) for v in HOLD_LON.split(","))
    x_hold = (lons >= lo) & (lons < hi)
    return path, moy, t_hold, x_hold


def checkpoint(tmp, run):
    """The minimum `pixelmae.pt` both scripts read before the transform."""
    d = os.path.join(tmp, "runs", run)
    os.makedirs(d, exist_ok=True)
    torch.save({"args": {"anomaly": True, "holdout_years": HOLD_YEARS,
                         "holdout_lon": HOLD_LON},
                "chan": ["c%d" % c for c in range(C)],
                "step": 1, "model": {}},
               os.path.join(d, "pixelmae.pt"))


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def sidecar_fixture(tmp, tag):
    """The same float16 fixture in family 5's layout: metadata in the npz, X
    beside it as a bare memmappable .npy. Returns (npz path, _X.npy path)."""
    from tensor_io import save_tensor, sidecar_path
    flat, _, _, _ = fixture(np.float16, tmp)
    d = np.load(flat)
    side = os.path.join(tmp, "sidecar_%s.npz" % tag)
    save_tensor(side, d["X"], **{k: d[k] for k in d.files if k != "X"})
    return side, sidecar_path(side)


class _Stop(Exception):
    """Raised from the patched codec_from_ckpt: everything after it is model
    work this test has no opinion about."""


def run_transform(module_name, data, tmp, extra_argv):
    """Run <module>.main() on the fixture and return X as it stands at the
    moment the module hands it to the codec -- i.e. fully transformed."""
    import importlib
    mod = importlib.import_module(module_name)
    grabbed = {}

    def spy(ck, nchan, *a, **k):
        grabbed["X"] = np.array(sys._getframe(1).f_locals["X"], dtype=np.float64)
        raise _Stop()

    old_here, old_ccf, old_argv = mod.HERE, mod.codec_from_ckpt, sys.argv
    mod.HERE, mod.codec_from_ckpt = tmp, spy
    sys.argv = [module_name + ".py", "--run", "fix", "--data", data] + extra_argv
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod.main()
    except _Stop:
        pass
    finally:
        mod.HERE, mod.codec_from_ckpt, sys.argv = old_here, old_ccf, old_argv
    assert "X" in grabbed, (
        f"{module_name}.main() never reached codec_from_ckpt -- this test "
        f"measured nothing")
    return grabbed["X"]


def report(tag, X):
    """(pool size, std over the train pool, exactly-zero fraction) on the
    dynamic channels."""
    dyn = [c for c in range(C) if c != STATIC]
    v = X[..., dyn]
    fin = np.isfinite(v)
    sd = float(v[fin].std(dtype=np.float64))
    zero = float((v[fin] == 0.0).mean())
    print(f"    {tag:<34} std={sd:.6f}  exactly-zero={zero:.1%}")
    return sd, zero


def main():
    tmp = tempfile.mkdtemp(prefix="stage2_f16_")
    try:
        # ---- 0: the fixture really is in the overflow regime -------------
        data16, moy, t_hold, x_hold = fixture(np.float16, tmp)
        checkpoint(tmp, "fix")
        X16 = np.load(data16)["X"]
        anom = np.empty_like(X16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clim = np.stack([np.nanmean(X16[(moy == m) & ~t_hold], axis=0)
                             for m in range(12)]).astype(np.float32)
            for c in range(C):
                anom[..., c] = X16[..., c] - clim[moy, :, :, c]
            pool = anom[..., 0][np.isfinite(anom[..., 0])
                                & ~t_hold[:, None, None]
                                & ~x_hold[None, None, :]]
            broken_sd = pool.std()                 # float16, as the copies did
            good_sd = pool.std(dtype=np.float64)
        print(f"  0. train pool {pool.size:,} entries; the OLD arithmetic "
              f"`v.std()` reads {broken_sd}, float64 reads {good_sd:.6f}")
        assert not np.isfinite(broken_sd), (
            f"the fixture does NOT overflow float16 (v.std() = {broken_sd}), "
            f"so this test cannot fail on the old code and proves nothing. "
            f"Raise T/H/W until the sum of squared residuals clears 65504.")
        assert 0.9 < good_sd < 1.1
        del X16, anom, clim, pool

        # ---- 1 & 2: the two scripts, on float16 --------------------------
        print("  float16 fixture, through each script's own transform path:")
        results = {}
        for i, (mod, argv) in enumerate((("temporal", []),
                                         ("probe_sequence", ["--anomaly"])),
                                        start=1):
            Xa = run_transform(mod, data16, tmp, argv)
            sd, zero = report(f"{mod}.py", Xa)
            results[mod] = (sd, zero)
            assert zero < 0.01, (
                f"{mod}.py: {zero:.1%} of the dynamic channels came out "
                f"EXACTLY 0.0 on a float16 tensor. That is the float16 "
                f"np.std overflow: v.std() returned inf and (X - mu)/inf is "
                f"0.0. Stage 2 would have trained on an all-zero field with "
                f"every loss and probe still reading healthy. The transform "
                f"must be trainprobe.anomaly_transform, which names "
                f"dtype=np.float64 on every reduction.")
            assert 0.9 < sd < 1.1, (
                f"{mod}.py: dynamic-channel sd {sd:.6f} on a float16 tensor; "
                f"the transform z-scores on the train pool, so it must be ~1")
            # A baked-climatology channel is context, not a target in
            # disguise: it must come back BIT-IDENTICAL. The canonical
            # transform gets that by giving the static channels a climatology
            # of exactly +0.0 and a (mu, den) of exactly (0.0, 1.0); the old
            # inlined copies got it by never touching them. Both are exact, so
            # this assertion holds on either and is not a way of asking "did
            # you use the new code".
            raw = np.load(data16)["X"][..., STATIC].astype(np.float64)
            got = Xa[..., STATIC]
            same = np.isclose(raw, got, rtol=0, atol=0, equal_nan=True)
            assert same.all(), (
                f"{mod}.py moved the STATIC channel at "
                f"{(~same).sum()} of {same.size} entries")
            print(f"  {i}. {mod}.py: dynamic channels survive float16")

        # ---- 3: float32 gives the same answer (the dtype is not the point)
        print("  float32 control, same fixture shape:")
        data32, _, _, _ = fixture(np.float32, tmp)
        for mod, argv in (("temporal", []), ("probe_sequence", ["--anomaly"])):
            Xa = run_transform(mod, data32, tmp, argv)
            sd32, zero32 = report(f"{mod}.py (float32)", Xa)
            sd16 = results[mod][0]
            assert zero32 < 0.01 and 0.9 < sd32 < 1.1
            assert abs(sd32 - sd16) < 5e-3, (
                f"{mod}.py: float16 sd {sd16:.6f} vs float32 {sd32:.6f} -- "
                f"the storage dtype must not change the answer beyond its own "
                f"rounding")
        print("  3. float16 and float32 agree to < 5e-3 on both scripts")

        # ---- 4: the sidecar layout (family 5) ----------------------------
        # Both scripts used to open the tensor with `np.load(...)["X"].copy()`
        # and therefore could not open family 5 AT ALL: 165.6 GB does not
        # decompress. They now go through tensor_io.load_tensor, which hands
        # back a READ-ONLY memmap -- and anomaly_transform writes into X, so
        # they must take a scratch copy, must not write through to the stored
        # tensor, and must not leave a 166 GB orphan behind. Run in a
        # SUBPROCESS, because the scratch is removed by atexit.
        print("  sidecar layout (family 5), one subprocess per script:")
        for mod, argv in (("temporal", []), ("probe_sequence", ["--anomaly"])):
            side, xpath = sidecar_fixture(tmp, mod)
            before = sha256(xpath)
            scratch = side[:-4] + ("_temporal_scratch.npy" if mod == "temporal"
                                   else "_seqprobe_scratch.npy")
            out = subprocess.run(
                [sys.executable, os.path.join(HERE, "_sidecar_driver.py"),
                 mod, side, tmp, scratch] + argv,
                capture_output=True, text=True, cwd=HERE)
            assert out.returncode == 0, (
                f"{mod}.py failed on a sidecar tensor:\n{out.stdout}\n"
                f"{out.stderr[-2000:]}")
            sd, zero = float(out.stdout.split("SD=")[1].split()[0]), \
                float(out.stdout.split("ZERO=")[1].split()[0])
            print(f"    {mod + '.py (sidecar)':<34} std={sd:.6f}  "
                  f"exactly-zero={zero:.1%}  scratch existed during the run: "
                  f"{'SCRATCH=1' in out.stdout}")
            assert "SCRATCH=1" in out.stdout, (
                f"{mod}.py did not take a scratch copy -- either X arrived "
                f"writable (the canonical tensor is exposed) or the branch "
                f"never fired")
            assert zero < 0.01 and 0.9 < sd < 1.1
            assert sha256(xpath) == before, (
                f"{mod}.py WROTE THROUGH to the canonical sidecar tensor. It "
                f"would now hold anomaly-space data where a state-space "
                f"tensor is documented, and the next run would z-score it "
                f"again with nothing to say so.")
            assert not os.path.exists(scratch), (
                f"{mod}.py left its scratch copy behind -- 166 GB per run at "
                f"family 5")
        print("  4. both scripts open a sidecar tensor, through a scratch "
              "copy, leaving the canonical X byte-identical and no orphan")

        print("\ntests/test_stage2_float16_anomaly.py: all 5 checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
