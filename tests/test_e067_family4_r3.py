#!/usr/bin/env python3
"""E-067 step 2: recipe r3 of build_family4.py — r2 plus `cur_u`, `cur_v`.

E-042 appended `sst` as channel 40 and made its safety argument in one
sentence: channels 0..38 keep the indices every published result was measured
at, so an r1 result and an r2 result stay comparable. r3 appends the two
GLORYS12 current COMPONENTS after it (E-067 §4, the cone codec needs the
DIRECTION of local advection, and a magnitude cannot supply one), and it owes
the same argument one rung up. So this file checks exactly three things, and
they are the three the plan can be wrong about:

  1. **The channel list is r2's, plus `cur_u` at index 40 and `cur_v` at 41.**
     Metadata only, and deliberately cheap — a reorder is the one mistake that
     would train cleanly, produce numbers, and mean something else. `chan_emb`
     embeds the channel INDEX (ml/model.py), so nothing downstream would
     notice.

  2. **hypot(cur_u, cur_v) == cur_speed wherever both are finite**, on a
     synthetic uo/vo cache, and the two components are missing on exactly the
     cells where the speed is. This is the claim that the components are the
     SAME binned means channel 0 is the hypotenuse of and not a second read of
     the aggregation — if a future edit gives them their own `fill_cur()` pass
     the identity is what breaks. The comparison runs in float32 through the
     tensor's own z-score round trip, so its floor is float16 storage (~2^-10
     relative), which is what the tolerance is derived from rather than
     guessed at.

  3. **r2 does not move.** Built from the same fixtures, the r3 tensor's
     channels 0..39 equal the r2 tensor's float16 for float16, NaN pattern
     included, and so do `norm[:40]`, the axis, the months and the labels.
     Appending cannot perturb an existing channel; that is the whole reason
     the E-067 arm can be read against an r2 control at all.

Plus two guards that cost nothing: an r2 cache is REFUSED for an r3 build (a
recipe string is a claim about the CODE, and a 40-channel tensor under r3's
name would otherwise be loaded), and the recipe file's `tensor` value is the
builder's own output stem (`ml-train.yml` derives $TENSOR from that input
verbatim, so a drift here is a run that cannot find the tensor it just built).

The fixtures are `tests/test_e034_family4.py`'s, imported rather than copied:
one synthetic pentad aggregation, one RG pair, one wind year, one SST
artifact. A second copy of them would be a second definition of what the
builder reads.

    python3 -m pytest -q tests/test_e067_family4_r3.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "ml")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_family3 as f3                      # noqa: E402
import build_family4 as f4                      # noqa: E402
import test_e034_family4 as fx                  # noqa: E402  (the fixtures)

PD = fx.PD


# --------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def built():
    """Build r2 and r3 from ONE set of synthetic sources, and hand back both.

    Module-scoped: the two builds are the expensive part of this file, and
    every check below is a question about the same pair of tensors. Building
    r2 first and r3 second is the order that matters — check 3 asks whether
    the r3 code path moved the r2 output, and it can only ask that if the r2
    output exists independently of it.
    """
    rng = np.random.default_rng(20260902)
    tmp = tempfile.mkdtemp(prefix="e067r3_")
    cache = os.path.join(tmp, "cache")
    os.makedirs(cache)
    pentad_dir = os.path.join(tmp, "pentad")

    src = fx.write_pentad_dir(pentad_dir, rng)
    fx.write_fake_base025(cache, fx.LATS, fx.LONS)
    fx.write_rg(cache, rng)
    fx.write_wind(cache, rng)
    fx.write_truth(cache)
    sst_dir = os.path.join(tmp, "sst_na025")
    sst = fx.write_sst(sst_dir, fx.BINS, PD)

    out_r2 = os.path.join(tmp, "family4_r2.npz")
    out_r3 = os.path.join(tmp, "family4_r3.npz")
    with contextlib.redirect_stdout(io.StringIO()):
        fx.run_build(cache, pentad_dir, out_r2,
                     extra=("--rev", "r2", "--sst-dir", sst_dir))
        fx.run_build(cache, pentad_dir, out_r3,
                     extra=("--rev", "r3", "--sst-dir", sst_dir))
    try:
        yield dict(tmp=tmp, cache=cache, pentad_dir=pentad_dir,
                   sst_dir=sst_dir, src=src, sst=sst,
                   d2=np.load(out_r2), d3=np.load(out_r3),
                   out_r2=out_r2, out_r3=out_r3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def denorm(d):
    """The tensor in its own units: float32 through the stored z-score."""
    return np.asarray(d["X"], np.float32) * d["norm"][:, 1] + d["norm"][:, 0]


# ------------------------------------------------------------------ 1 ------
def test_r3_channels_are_r2_plus_cur_u_cur_v():
    """The channel list, and the two indices the plan names."""
    r2, r3 = f4.CHANS_BY_REV["r2"], f4.CHANS_BY_REV["r3"]
    assert r2 == list(f3.CHANS) + ["sst"]
    assert r3 == r2 + ["cur_u", "cur_v"], \
        f"r3 is not r2 + [cur_u, cur_v]: {r3[len(r2):]}"
    assert len(r3) == 42 and r3[:40] == r2
    assert r3[40] == "cur_u" and r3[41] == "cur_v"
    # the module constants the builder writes through, so an index and its
    # name cannot drift apart
    assert f4.C_CUR_U == 40 and f4.C_CUR_V == 41
    assert f4.C_SST == 39 == f3.NC


def test_r3_channel_metadata_on_the_built_tensor(built):
    d3 = built["d3"]
    assert [str(c) for c in d3["chan"]] == f4.CHANS_BY_REV["r3"]
    assert d3["X"].shape[3] == 42
    assert str(d3["recipe"]) == "f4r3"
    assert str(d3["cadence"]) == "pentad" and int(d3["pentad_days"]) == PD


# ------------------------------------------------------------------ 2 ------
def test_hypot_of_the_components_is_cur_speed(built):
    """cur_speed == hypot(cur_u, cur_v), cell by cell, bin by bin.

    Tolerance, derived rather than picked: the tensor is stored float16, so a
    value carries ~2^-11 relative error and the z-score round trip through
    (x - mu) / sd doubles it at worst. 4 * 2^-10 is a comfortable ceiling on
    that and is still ~200x tighter than the difference between hypot(mean)
    and mean(hypot), which is the wrong implementation this catches.
    """
    d3 = built["d3"]
    raw = denorm(d3)
    speed = raw[..., 0]
    u, v = raw[..., f4.C_CUR_U], raw[..., f4.C_CUR_V]

    fin = np.isfinite(speed)
    assert fin.any(), "the fixture has no finite current at all"

    # (a) missingness is the SAME set of cells — the convention r2's sst
    #     channel established: NaN is the missing token, and a component that
    #     was invented where the speed is missing would show up here.
    assert np.array_equal(np.isfinite(u), fin), \
        "cur_u is finite on cells where cur_speed is not (or vice versa)"
    assert np.array_equal(np.isfinite(v), fin), \
        "cur_v is finite on cells where cur_speed is not (or vice versa)"

    # (b) the identity itself, against a bound computed from the tensor's own
    #     storage rather than a picked epsilon. Each of the three values is a
    #     float16 z-score, so it carries at most half an ULP — |dx| <=
    #     2^-11 * sd_c * |z| in the channel's units — and hypot is
    #     1-Lipschitz in each argument, so the three errors can at worst add.
    #     3x that bound leaves room for the float32 arithmetic here and is
    #     still ~100x tighter than mean(hypot) vs hypot(mean), the wrong
    #     implementation this catches.
    z = np.asarray(d3["X"], np.float32)
    sdv = d3["norm"][:, 1]
    ulp = 2.0 ** -11 * (sdv[f4.C_CUR_U] * np.abs(z[..., f4.C_CUR_U])
                        + sdv[f4.C_CUR_V] * np.abs(z[..., f4.C_CUR_V])
                        + sdv[0] * np.abs(z[..., 0]))
    got = np.hypot(u[fin], v[fin]).astype(np.float32)
    want = speed[fin].astype(np.float32)
    err = np.abs(got - want)
    tol = 3 * ulp[fin] + 1e-6
    assert (err <= tol).all(), \
        (f"hypot(cur_u, cur_v) != cur_speed: {int((err > tol).sum())} of "
         f"{int(fin.sum())} cells exceed their float16 storage bound, worst "
         f"err {err.max():.3e} against tol {tol[err.argmax()]:.3e}")

    # (c) and the components are the aggregation's own means, not a rotation
    #     or a re-derivation of them: compare against the source arrays.
    src, bins = built["src"], built["d3"]["bin_index"].tolist()
    ocean = np.isfinite(raw[0, :, :, 0])
    j = 3                                        # an interior bin
    for name, c, arr in (("cur_u", f4.C_CUR_U, src["uo"]),
                         ("cur_v", f4.C_CUR_V, src["vo"])):
        want_src = np.asarray(arr[j], np.float32)
        m = ocean & np.isfinite(want_src)
        assert np.allclose(raw[j, :, :, c][m], want_src[m],
                           rtol=3e-3, atol=3e-3), \
            f"{name} is not the binned mean the aggregation wrote"
    assert len(bins) == raw.shape[0]


def test_the_check_can_fail_at_all(built):
    """A tolerance no wrong answer could exceed is not a check.

    The one plausible wrong implementation is `mean(hypot)` in place of
    `hypot(mean)` — the mistake the base channel's own docstring warns about.
    Confirm this fixture separates the two by far more than the tolerance
    above, so test_hypot_of_the_components_is_cur_speed is load-bearing.
    """
    src = built["src"]
    j = 3
    u = np.asarray(src["uo"][j], np.float32)
    v = np.asarray(src["vo"][j], np.float32)
    m = np.isfinite(u) & np.isfinite(v)
    ref = np.hypot(u, v)[m]
    # a deliberately WRONG pair: components scaled apart, same order of
    # magnitude — the gentlest error the identity has to see.
    bad = np.hypot(u * 1.1, v * 0.9)[m]
    floor = 3 * 2.0 ** -11 * 3 * np.abs(ref)      # the check's own bound
    assert (np.abs(bad - ref) > floor).mean() > 0.5, \
        "the fixture cannot distinguish a perturbed component pair from the " \
        "true one — the identity check above could not fail"


# ------------------------------------------------------------------ 3 ------
def test_r2_is_bit_identical_inside_r3(built):
    """Appending cannot perturb an existing channel."""
    d2, d3 = built["d2"], built["d3"]
    x2, x3 = np.asarray(d2["X"]), np.asarray(d3["X"])
    assert x2.dtype == x3.dtype == np.float16
    assert x2.shape[3] == 40 and x3.shape[3] == 42
    assert np.array_equal(x2, x3[..., :40], equal_nan=True), \
        ("appending cur_u/cur_v CHANGED one of the 40 published channels — "
         "the whole safety argument for an r2-controlled r3 arm is that it "
         "cannot")
    assert np.array_equal(d2["norm"], d3["norm"][:40]), \
        "the per-channel z-score of the 40 moved when two more were appended"
    for k in ("months", "bin_index", "lats", "lons", "truth_rapid", "rapid",
              "n_rg_live", "n_wind", "n_sst"):
        assert np.array_equal(d2[k], d3[k]), f"{k} differs between r2 and r3"
    assert str(d2["recipe"]) == "f4r2" and str(d3["recipe"]) == "f4r3"
    # r2 keeps the exact key set it published; the new counter is r3-only
    assert "n_cur" not in d2.files and "n_cur" in d3.files
    assert int(d3["n_cur"]) == int(np.isfinite(
        denorm(d3)[..., f4.C_CUR_U]).any(axis=(1, 2)).sum())


def test_r2_output_name_and_recipe_string_unchanged():
    """r1 and r2 still write what every published result was built under."""
    assert f4.CADENCE[5]["revs"]["r1"] == dict(
        recipe="f4r1", out="family4_na025_pentad.npz")
    assert f4.CADENCE[5]["revs"]["r2"] == dict(
        recipe="f4r2", out="family4_na025_pentad_r2.npz")
    assert f4.CADENCE[5]["revs"]["r3"] == dict(
        recipe="f4r3", out="family4_na025_pentad_r3.npz")


# ------------------------------------------------------------- the guards --
def test_an_r2_cache_is_refused_for_an_r3_build(built):
    """A recipe string is a claim about the CODE that wrote the file.

    A 40-channel r2 tensor sitting under r3's name would train a 42-channel
    experiment on a tensor with no current components — or, worse, skip the
    build and load it. The same guard E-042 pinned for the r1/r2 pair.
    """
    guard = os.path.join(built["tmp"], "guard_r3.npz")
    shutil.copy(built["out_r2"], guard)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fx.run_build(built["cache"], built["pentad_dir"], guard,
                     extra=("--rev", "r3", "--sst-dir", built["sst_dir"]))
    said = buf.getvalue()
    assert "cached tensor is recipe 'f4r2', want 'f4r3'" in said, said
    assert "already built" not in said
    dg = np.load(guard)
    assert str(dg["recipe"]) == "f4r3" and dg["X"].shape[3] == 42

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fx.run_build(built["cache"], built["pentad_dir"], guard,
                     extra=("--rev", "r3", "--sst-dir", built["sst_dir"]))
    assert "already built by recipe f4r3" in buf.getvalue(), \
        "an r3 cache is not being reused — every run would rebuild 33 GB"


def test_the_recipe_file_names_the_builder_s_own_stem():
    """ml-train.yml derives $TENSOR from the `tensor` input verbatim.

    So the recipe's tensor value and the builder's output stem are one fact
    written twice, and this is the check that keeps them one fact.
    """
    path = os.path.join(ROOT, "ml", "recipes", "f4r3-cone-5M.json")
    d = json.load(open(path))
    stem = f4.CADENCE[5]["revs"]["r3"]["out"][:-4]
    assert d["tensor"] == stem, \
        f"recipe says {d['tensor']!r}, builder writes {stem}.npz"
    assert d["head_probe"] == "true"          # ml/CLAUDE.md §3
    assert d["_description"].startswith(
        "E-067 cone-native codec on family4 r3 (r2 + cur_u, cur_v)")
    assert d["_provenance"].startswith("E-067 plan, 2026-09-02, not yet measured")
    # every key is a real ml-train.yml input or a declared recipe-only key,
    # and is consumed as $RECIPE_<KEY> — the resolver is the authority, so
    # run it rather than restating its rules.
    import subprocess
    r = subprocess.run(["bash", "scripts/resolve_recipe.sh",
                        "recipe:f4r3-cone-5M"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "GITHUB_ENV": "/dev/null"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RECIPE_TENSOR=family4_na025_pentad_r3" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
