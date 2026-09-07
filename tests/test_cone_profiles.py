#!/usr/bin/env python3
"""E-076a · the family-8 sparse gather: the k nearest profiles as dot tokens.

The arm under test replaces one GROUP of the tensor by raw observations. The
`rg100` group — the gridded Argo depth column, live one pentad in six — is not
read at all; instead each anchor gets the k nearest real profiles as dot
tokens carrying their own displacement in kilometres, their age in days and
the level's depth. What has to be true, and is checked here:

  1. the shape is RECTANGULAR and the coordinates are the profiles' own,
     against a brute-force nearest search;
  2. no observation later than the anchor's pentad ever appears;
  3. with `train_bins` given, no observation from a held-out bin appears;
  4. `withhold_nearest` takes the nearest profile OUT of the input and hands
     it back as the target;
  5. the VALUE a profile carries is the same number the gridded path would
     carry for the same measurement — the anomaly chain of E-076 §2.5,
     measured against the real `anomaly_transform` rather than argued;
  6. an archived E-069 checkpoint still builds and loads (the dot projection
     widens only when a run asks for the extra fields);
  7. the common-target eval writes finite numbers with its three baselines.

Everything is SYNTHETIC and nothing touches the network: the store is written
by `ml/family8_store.py::write_synthetic_store` and the tensor by
`ml/train_cone.py::smoke_family8`, both of which are the same code the
`--smoke-family8` end-to-end run uses.

    python3 -m pytest -q tests/test_cone_profiles.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

import cone_sampler as cs                                          # noqa: E402
from cone_sampler import ConeSampler, Group, GroupSet              # noqa: E402
from family8_store import (ArgoStore, offsets_km, verify_store,    # noqa: E402
                           write_synthetic_store)

LEVELS = [10.0, 30.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0,
          900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0]
RG_CHAN = (["rg_t%d" % int(v) for v in LEVELS]
           + ["rg_s%d" % int(v) for v in LEVELS])
G025_CHAN = ["cur_speed", "ssh", "sst"]
G100_CHAN = ["t2m"]

# The toy axes. A 0.25-degree master grid at 20N/-60E and its 1-degree coarse
# partner, the same two-group geometry tests/test_cone_global.py::_toy_groups
# builds, plus the live-bins group family 8 replaces.
NY, NX = 40, 48
H1, W1 = NY // 4 + 1, NX // 4 + 1
B0, T = 2045, 40                       # bin 2045 opens 2010-01-02
LAT0, LON0, STEP = 20.0, -60.0, 0.25


def _axes():
    return (LAT0 + STEP * np.arange(NY), LON0 + STEP * np.arange(NX),
            LAT0 + np.arange(H1, dtype=np.float64),
            LON0 + np.arange(W1, dtype=np.float64))


def _groups(rg_rows=None, rg_fill=1.0, seed=0):
    """A three-group set in family 7's shape: g025, g100, rg100 (live bins)."""
    rng = np.random.default_rng(seed)
    lats, lons, lat1, lon1 = _axes()
    dense = rng.normal(size=(T, NY, NX, len(G025_CHAN))).astype(np.float32)
    coarse = rng.normal(size=(T, H1, W1, len(G100_CHAN))).astype(np.float32)
    rows = list(rg_rows if rg_rows is not None else range(0, T, 6))
    rg = np.full((len(rows), H1, W1, 32), np.float32(rg_fill), np.float32)
    g0 = Group("g025", dense, lats, lons, G025_CHAN)
    g1 = Group("g100", coarse, lat1, lon1, G100_CHAN)
    g2 = Group("rg100", rg, lat1, lon1, RG_CHAN, bin_index=np.array(rows))
    return GroupSet([g0, g1, g2]), np.asarray(rows)


def _flat_stats(n_chan=32, clim=0.0, mu=0.0, den=1.0, shape=(H1, W1)):
    """Anomaly constants with a known, uniform value, so a hand computation of
    the chain is possible. `test_the_value_chain_matches_the_grid` uses the
    REAL transform instead; this is for the geometry tests, where the value is
    not what is being asked about."""
    return {"clim": np.full((12,) + shape + (n_chan,), np.float32(clim)),
            "mu": np.full(n_chan, float(mu)),
            "den": np.full(n_chan, float(den)),
            "sd": np.full(n_chan, float(den) - 1e-6)}


def _norm(n_chan=32, mean=0.0, sd=1.0):
    return np.stack([np.full(n_chan, float(mean)),
                     np.full(n_chan, float(sd))], axis=1)


def _store(tmp, n=400, seed=1, b_lo=B0, b_hi=B0 + T - 1, name="store"):
    """A synthetic store scattered over the toy window and its bins."""
    rng = np.random.default_rng(seed)
    bins = rng.integers(b_lo, b_hi + 1, n)
    frac = rng.uniform(0.0, 1.0, n)
    lat = rng.uniform(LAT0, LAT0 + STEP * (NY - 1), n)
    lon = rng.uniform(LON0, LON0 + STEP * (NX - 1), n)
    temp = rng.normal(10.0, 3.0, (n, 16)).astype(np.float32)
    psal = rng.normal(35.0, 0.5, (n, 16)).astype(np.float32)
    temp[rng.random((n, 16)) < 0.15] = np.nan
    psal[rng.random((n, 16)) < 0.15] = np.nan
    path = write_synthetic_store(
        os.path.join(str(tmp), name), bins,
        5.0 * bins + 5.0 * frac, lat, lon, temp, psal, levels=LEVELS)
    return ArgoStore(path)


def _sampler(tmp, k=5, train_bins=None, store=None, gs=None, rows=None,
             stats=None, L_in=2):
    if gs is None:
        gs, rows = _groups()
    st = store if store is not None else _store(tmp)
    s = ConeSampler(gs, None, gs.lats, gs.lons, gs.chan, L_in=L_in,
                    profile_store=st, profile_k=k,
                    profile_norm=_norm(), profile_stats=stats or _flat_stats(),
                    profile_bin_index=B0 + np.arange(T),
                    train_bins=train_bins)
    return s, gs, st


# ============================================================ 1. the shape ==
def test_the_f8_sample_is_k_times_32_profile_dots_and_no_rg_patch(tmp_path):
    """Rectangular by construction, and the gridded column is gone.

    E-076 §2.2's third rule: a fixed k per anchor, miss tokens where the array
    is thin, so a batch is never ragged. And §5's arm definition: the `rg100`
    group is not read — its lag-0 patch and its sunflower dots are absent, and
    what stands in their place is exactly k * 32 profile dots.
    """
    k = 5
    s, gs, st = _sampler(tmp_path, k=k)
    twin = ConeSampler(gs, None, gs.lats, gs.lons, gs.chan, L_in=2)
    y = 20
    assert s.n_profile_dots == k * 32
    # the dense dot table lost exactly the rg100 channels' dots
    assert s.n_dots(y) < twin.n_dots(y)
    rg_dots = sum(1 for c in twin.row(y)["chan"] if c >= len(G025_CHAN)
                  + len(G100_CHAN))
    assert twin.n_dots(y) - s.n_dots(y) == rg_dots > 0
    assert all(int(c) < len(G025_CHAN) + len(G100_CHAN)
               for c in s.row(y)["chan"])

    out = s.sample(np.array([[30, y, 24], [31, 12, 30]], np.int64))
    n_dense = s.n_dots(y)
    assert out["vals"].shape[1] == n_dense + k * 32
    prof = slice(out["vals"].shape[1] - k * 32, out["vals"].shape[1])
    ci = out["chan"][:, prof]
    base = len(G025_CHAN) + len(G100_CHAN)
    assert np.array_equal(ci[0], np.tile(np.arange(base, base + 32),
                                         k).astype(np.int16))
    # THE PATCH AND THE FUTURE carry nothing for those channels: they are not
    # read, so they are unobserved, which is the truth about a group this arm
    # does not open.
    assert not out["patch_obs"][:, base:, :].any()
    assert not (out["patch_vals"][:, base:, :] != 0).any()
    assert not out["fut_obs"][:, base:, :].any()
    # ...and the OTHER channels still have theirs
    assert out["patch_obs"][:, :base, :].any()
    # every profile dot carries the profile footprint, the dense dots do not
    assert np.allclose(out["fp"][:, prof, 0], cs.PROFILE_FOOTPRINT[0])
    assert np.allclose(out["fp"][:, prof, 1], cs.PROFILE_FOOTPRINT[1])
    assert np.allclose(out["fp"][0, :n_dense, 0][
        out["chan"][0, :n_dense] < len(G025_CHAN)], 0.0)


def test_the_rg_array_is_never_read_on_the_f8_arm(tmp_path):
    """Not read and then discarded — NOT READ.

    The dense gather is handed the other channels only, so poisoning the
    `rg100` array with a value no code could confuse for missing changes
    nothing about the sample. This is the difference between an arm that
    ignores the gridded interior and one that reads it and zeroes it.
    """
    gs, rows = _groups(rg_fill=1.0)
    st = _store(tmp_path)
    s = ConeSampler(gs, None, gs.lats, gs.lons, gs.chan, L_in=2,
                    profile_store=st, profile_k=3, profile_norm=_norm(),
                    profile_stats=_flat_stats(),
                    profile_bin_index=B0 + np.arange(T))
    a = np.array([[30, 20, 24], [24, 8, 40]], np.int64)
    before = s.sample(a)
    gs.groups[2].X[...] = 1e6                       # poison
    after = s.sample(a)
    for key in ("vals", "obs", "valid", "patch_vals", "patch_obs",
                "fut_vals", "fut_obs", "nr"):
        assert np.array_equal(before[key], after[key]), key


def test_the_coordinates_are_the_profiles_own(tmp_path):
    """Against a BRUTE-FORCE nearest search over the store's raw columns.

    Not against `knearest` — that is the function under test one level down —
    but against the plan's own definition: one-sided in time, inside
    (R_max, T_max), ranked by d^2 = (dx^2 + dy^2)/L^2 + dt^2/T^2, ties in the
    store's own order.
    """
    k = 4
    s, gs, st = _sampler(tmp_path, k=k)
    t, y, x = 33, 22, 30
    out = s.sample(np.array([[t, y, x]], np.int64))
    n = out["vals"].shape[1] - k * 32
    lat, lon = float(gs.lats[y]), float(gs.lons[x])

    b = np.asarray(st["bin"], np.int64)
    td = np.asarray(st["time_days"], np.float64)
    t_anchor = 5.0 * (B0 + t + 1)
    dt = t_anchor - td
    dx, dy = offsets_km(lat, lon, np.asarray(st["lat"], np.float64),
                        np.asarray(st["lon"], np.float64))
    r2 = dx * dx + dy * dy
    ok = (b <= B0 + t) & (dt >= 0) & (dt <= 30.0) & (r2 <= 1000.0 ** 2)
    d2 = np.where(ok, r2 / 1000.0 ** 2 + (dt / 30.0) ** 2, np.inf)
    want = np.argsort(d2, kind="stable")[:k]
    assert np.isfinite(d2[want]).all(), "the fixture gave this anchor no hits"

    for i, row in enumerate(want):
        j = n + i * 32
        assert np.allclose(out["dy_km"][0, j:j + 32], dy[row], atol=1e-3)
        assert np.allclose(out["dx_km"][0, j:j + 32], dx[row], atol=1e-3)
        assert np.allclose(out["lag_days"][0, j:j + 32], dt[row], atol=1e-3)
        assert out["valid"][0, j:j + 32].all()
    # the depth column is the CHANNEL's, not the profile's
    assert np.allclose(out["depth"][0, n:n + 16], LEVELS)
    assert np.allclose(out["depth"][0, n + 16:n + 32], LEVELS)
    # n_R travels as log1p on every slot of the anchor, miss slots included
    assert np.allclose(out["nr"][0, n:], np.log1p(float(ok.sum())))


def test_a_thin_anchor_gets_miss_tokens_not_a_shorter_row(tmp_path):
    """Fewer than k in range is the common case in the Southern Ocean and
    before 2004. The slots are still there and they are invalid — the miss
    token family 7 already has, made rare (E-076 §2.2)."""
    st = _store(tmp_path, n=6, seed=5)
    s, gs, _ = _sampler(tmp_path, k=5, store=st)
    out = s.sample(np.array([[2, 20, 24]], np.int64))     # early bin, thin
    k32 = 5 * 32
    prof = out["valid"][0, -k32:]
    assert len(prof) == k32
    assert not prof.all(), "this anchor was not thin — the test is vacuous"
    assert (out["vals"][0, -k32:][~prof] == 0).all()


# ======================================================= 2. the time rules ==
def test_no_profile_later_than_the_anchor_bin_ever_appears(tmp_path):
    """E-076 §2.2's first rule, and `certify`'s new check.

    The store is FULL of later profiles here — the anchor sits early in the
    axis — so a search that were not one-sided would return them.
    """
    s, gs, st = _sampler(tmp_path, k=5)
    for t in (8, 15, 30):
        out = s.sample(np.array([[t, 20, 24]], np.int64))
        lag = out["lag_days"][0, -5 * 32:][out["valid"][0, -5 * 32:]]
        assert (lag >= 0).all()
        assert (lag <= 30.0).all()
    # and the certificate says so by brute force over drawn anchors
    train_bins = np.ones(T, bool)
    anchors = np.array([[t, 20, 24] for t in range(8, 20)], np.int64)
    assert s.certify(anchors, train_bins) == 0


def test_certify_catches_a_store_that_answers_from_the_future(tmp_path):
    """The certificate is only worth having if it can FAIL.

    A store whose `bin_offsets` claims every row belongs to bin 0 makes the
    CSR slice return rows dated later than the anchor — precisely the leak
    §2.2 forbids — and `certify` must count that anchor as a violation.
    """
    s, gs, st = _sampler(tmp_path, k=3)
    off = np.zeros(st.n_bins + 1, np.int64)
    off[1:] = st.N                          # every row lands in bin 0
    st.bin_offsets = off
    anchors = np.array([[1, 20, 24]], np.int64)
    assert s.certify(anchors, np.ones(T, bool)) == 1


def test_a_held_out_bin_never_enters_a_training_sample(tmp_path):
    """The loader's half of the leak (docs/FAMILY8_DATA_HANDOVER.md §7.3).

    The window rule already keeps a training anchor's search inside its own
    admitted window; this is the check that SAYS so rather than the argument
    that says so, and it drops a slot instead of shifting the next one up —
    a shifted slot would silently be a different, farther measurement.
    """
    train = np.ones(T, bool)
    train[20:26] = False                       # a held-out block
    s, gs, st = _sampler(tmp_path, k=5, train_bins=train)
    t = 26                                     # its search reaches back into it
    a = np.array([[t, 20, 24]], np.int64)
    k32 = 5 * 32
    free = s.sample(a, train_pool=False)
    held = s.sample(a, train_pool=True)
    b = np.asarray(st["bin"], np.int64)

    def bins_of(out):
        lag = out["lag_days"][0, -k32:][out["valid"][0, -k32:]]
        return sorted({int(B0 + t - np.ceil((v - 5.0) / 5.0)) for v in lag})
    assert free["valid"][0, -k32:].sum() > held["valid"][0, -k32:].sum(), (
        "no profile of this anchor came from the held-out block — the test "
        "would pass with the filter deleted")
    # every profile the TRAINING sample kept is from a training bin
    kept = held["valid"][0, -k32:].reshape(5, 32)[:, 0]
    res = s.profile.knearest(float(gs.lats[20]), float(gs.lons[24]), t, 5)
    rows = np.asarray(res["row"], np.int64)
    for i, keep in enumerate(kept):
        if rows[i] < 0:
            continue
        row_bin = int(b[rows[i]])
        assert bool(keep) == bool(train[row_bin - B0]), (i, row_bin)
    # DROPPED, NOT SHIFTED: the surviving slots sit where they were
    assert np.array_equal(
        held["dy_km"][0, -k32:][held["valid"][0, -k32:]],
        free["dy_km"][0, -k32:][held["valid"][0, -k32:]])


# ========================================================= 3. the withhold ==
def test_withhold_nearest_drops_the_nearest_and_returns_it(tmp_path):
    """E-076 §5.1: the target profile is never in its own input."""
    k = 4
    s, gs, st = _sampler(tmp_path, k=k)
    a = np.array([[33, 22, 30], [30, 10, 20]], np.int64)
    plain = s.sample(a)
    held = s.sample(a, withhold_nearest=True)
    k32 = k * 32
    assert "profile_target" not in plain
    pt = held["profile_target"]
    assert pt["valid"].all()
    # the input's first slot is the plain sample's SECOND — one shift, no more
    assert np.allclose(held["dy_km"][:, -k32:-k32 + 32],
                       plain["dy_km"][:, -k32 + 32:-k32 + 64])
    # the withheld one is the plain sample's first, and is now the target
    assert np.allclose(pt["dy_km"], plain["dy_km"][:, -k32])
    assert np.allclose(pt["dx_km"], plain["dx_km"][:, -k32])
    assert np.allclose(pt["dt_days"], plain["lag_days"][:, -k32])
    # ...and it is GONE from the input, not merely moved
    for i in range(len(a)):
        d = held["dy_km"][i, -k32:]
        assert not np.isclose(d, pt["dy_km"][i]).any() or \
            not np.isclose(held["lag_days"][i, -k32:], pt["dt_days"][i]).any()
    # n_R is decremented by the one taken out
    assert np.allclose(held["nr"][:, -k32:],
                       np.log1p(np.expm1(plain["nr"][:, -k32:]) - 1.0))
    # PERSISTENCE is the nearest REMAINING profile — the input's first slot
    assert pt["pers_valid"].all()
    # the query block is the 32 channels at the target's own coordinates
    assert pt["q_chan"].shape == (len(a), 32)
    assert np.array_equal(pt["q_chan"][0], s.profile.chan)
    assert np.allclose(pt["q_depth"][0], np.concatenate([LEVELS, LEVELS]))
    assert np.allclose(pt["q_lag_days"][:, 0], pt["dt_days"])


def test_withhold_is_per_anchor(tmp_path):
    """The training objective withholds on a FRACTION of anchors, so one batch
    carries both regimes and a step's gradient is not a function of which side
    of a coin the whole batch landed on."""
    k = 3
    s, gs, st = _sampler(tmp_path, k=k)
    a = np.array([[33, 22, 30], [33, 22, 30]], np.int64)
    out = s.sample(a, withhold_nearest=np.array([True, False]))
    pt = out["profile_target"]
    assert bool(pt["valid"][0]) and not bool(pt["valid"][1])
    assert pt["q_obs"][1].sum() == 0            # weight zero for that anchor
    k32 = k * 32
    # the row that withheld nothing kept the nearest profile in its input
    plain = s.sample(a[1:], withhold_nearest=False)
    assert np.allclose(out["dy_km"][1, -k32:], plain["dy_km"][0, -k32:])
    assert not np.allclose(out["dy_km"][0, -k32:], plain["dy_km"][0, -k32:])


# ======================================================== 4. the value chain ==
def test_the_value_chain_matches_the_grid(tmp_path):
    """E-076 §2.5, MEASURED: a profile's value in the model's space is the
    number the gridded path would carry for the same measurement.

    The experiment: put one raw value into the `rg100` array at a cell and a
    month, run the REAL `anomaly_transform` over it (capturing its stats),
    then ask the sampler for a profile carrying that same raw value at that
    same cell and month. The two numbers must agree — the grid's to within
    the float16-ness of nothing (this fixture is float32) and the profile's
    exactly, because both go through the same clim, mu and den.
    """
    from trainprobe import anomaly_transform
    rng = np.random.default_rng(3)
    lats, lons, lat1, lon1 = _axes()
    # FOUR YEARS of monthly rows, not four months: a climatology is a MEAN
    # over the training samples of one calendar month, and with one sample per
    # month it would equal the value, the anomaly would be identically zero
    # and the comparison below would be vacuous (it was, on the first draft —
    # both sides read 0.0 / 1e-6).
    T2 = 300
    rows = np.arange(0, T2, 6)
    raw = rng.normal(12.0, 4.0, (len(rows), H1, W1, 32)).astype(np.float32)
    norm = _norm(32, mean=11.0, sd=3.0)
    z = ((raw - norm[:, 0]) / norm[:, 1]).astype(np.float32)

    months = [str(d) for d in
              (cs.PENTAD_EPOCH
               + (5 * (B0 + np.arange(T2))).astype("timedelta64[D]"))]
    moy = np.array([int(m[5:7]) - 1 for m in months])[rows]
    t_hold = np.zeros(len(rows), bool)
    t_hold[-3:] = True
    stats = {}
    A = z.copy()
    A, dyn = anomaly_transform(A, moy, t_hold, np.zeros(W1, bool), stats=stats)
    assert len(dyn) == 32 and "clim" in stats

    gs = GroupSet([
        Group("g025", np.zeros((T2, NY, NX, len(G025_CHAN)), np.float32),
              lats, lons, G025_CHAN),
        Group("g100", np.zeros((T2, H1, W1, len(G100_CHAN)), np.float32),
              lat1, lon1, G100_CHAN),
        Group("rg100", A, lat1, lon1, RG_CHAN, bin_index=rows)])

    # ONE profile, at the centre of a known coarse cell and INSIDE the row's
    # own calendar month — the chain is evaluated at the profile's month, so a
    # profile whose bin straddles a month boundary would be a different
    # question (and is the one `month_of` exists to get right).
    r = next(rr for rr in range(4, len(rows))
             if 5 <= int(months[rows[rr]][8:10]) <= 20)
    j, i = 3, 4
    prof_bin = B0 + int(rows[r])
    path = write_synthetic_store(
        os.path.join(str(tmp_path), "one"),
        [prof_bin], [5.0 * prof_bin + 1.0],
        [float(lat1[j])], [float(lon1[i])],
        raw[r, j, i, :16][None, :], raw[r, j, i, 16:][None, :], levels=LEVELS)
    s = ConeSampler(gs, None, gs.lats, gs.lons, gs.chan, L_in=1,
                    profile_store=ArgoStore(path), profile_k=1,
                    profile_norm=norm, profile_stats=stats,
                    profile_bin_index=B0 + np.arange(T2))
    out = s.sample(np.array([[int(rows[r]), j * 4, i * 4]], np.int64))
    got = out["vals"][0, -32:]
    want = np.asarray(A[r, j, i], np.float32)
    assert out["obs"][0, -32:].all()
    assert np.abs(want).max() > 0.1, "the grid anomaly is ~0 — vacuous"
    # THE ONE DIFFERENCE, and it is the store's not the chain's: the store
    # holds float16 degrees Celsius (docs/FAMILY8_DATA_HANDOVER.md §2), so the
    # profile's raw value is quantised where the grid's float32 copy is not.
    # The tolerance is that quantisation carried through the chain — not a
    # round number picked until the test passed.
    raw32 = np.concatenate([raw[r, j, i, :16], raw[r, j, i, 16:]])
    q = np.abs(raw32.astype(np.float16).astype(np.float64) - raw32)
    tol = q / norm[:, 1] / np.asarray(stats["den"]) + 2e-5
    assert np.abs(got - want).max() <= tol.max(), (got[:3], want[:3])
    assert (np.abs(got - want) <= tol).all(), (got[:3], want[:3])
    assert tol.max() < 0.02, "the float16 tolerance is not tight enough to say "\
                             "anything"

    # AND the climatology's own hole: a (cell, month) the transform could not
    # define reads UNOBSERVED, never a zero the model would take for a value.
    stats2 = dict(stats)
    clim = np.array(stats["clim"], copy=True)
    clim[:, j, i, :] = np.nan
    stats2["clim"] = clim
    s2 = ConeSampler(gs, None, gs.lats, gs.lons, gs.chan, L_in=1,
                     profile_store=ArgoStore(path), profile_k=1,
                     profile_norm=norm, profile_stats=stats2,
                     profile_bin_index=B0 + np.arange(T2))
    out2 = s2.sample(np.array([[int(rows[r]), j * 4, i * 4]], np.int64))
    assert not out2["obs"][0, -32:].any()
    assert (out2["vals"][0, -32:] == 0).all()
    assert out2["valid"][0, -32:].all()          # the SLOT still exists


def test_an_unfilled_level_is_unobserved_not_zero(tmp_path):
    """A NaN level of a present profile is `obs = False` with a zero value —
    the miss token — and the OTHER levels of that profile are unaffected."""
    lats, lons, lat1, lon1 = _axes()
    gs, rows = _groups()
    temp = np.full((1, 16), 10.0, np.float32)
    psal = np.full((1, 16), 35.0, np.float32)
    temp[0, 3] = np.nan
    psal[0, 9] = np.nan
    b = B0 + int(rows[2])
    path = write_synthetic_store(os.path.join(str(tmp_path), "nan"), [b],
                                 [5.0 * b + 1.0], [float(lat1[3])],
                                 [float(lon1[4])], temp, psal, levels=LEVELS)
    s = ConeSampler(gs, None, gs.lats, gs.lons, gs.chan, L_in=1,
                    profile_store=ArgoStore(path), profile_k=1,
                    profile_norm=_norm(), profile_stats=_flat_stats(),
                    profile_bin_index=B0 + np.arange(T))
    out = s.sample(np.array([[int(rows[2]), 12, 16]], np.int64))
    o = out["obs"][0, -32:]
    assert not o[3] and not o[16 + 9]
    assert o.sum() == 30
    assert out["vals"][0, -32:][3] == 0.0


# ====================================================== 5. the model's side ==
def test_an_e069_checkpoint_still_builds_and_loads(tmp_path):
    """`n_extra` defaults to 0, so `dot_proj` keeps its Linear(2, d) shape and
    every archived cone checkpoint — whose `args` has no `dot_extras` at all —
    loads into a model built from its own record, bit for bit."""
    import torch
    from cone_codec import ConeMAE
    old = ConeMAE(6, d_model=32, n_heads=4, n_latents=8, n_layers=1, d_z=8,
                  d_dec=32, dec_layers=1, n_fourier=4)
    blob = {"args": {"d_model": 32, "n_heads": 4, "n_latents": 8,
                     "n_layers": 1, "d_z": 8, "d_dec": 32, "dec_layers": 1,
                     "n_fourier": 4},                       # no dot_extras
            "model": old.state_dict()}
    p = str(tmp_path / "e069.pt")
    torch.save(blob, p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    args = ck["args"]
    fresh = ConeMAE(6, d_model=args["d_model"], n_heads=args["n_heads"],
                    n_latents=args["n_latents"], n_layers=args["n_layers"],
                    d_z=args["d_z"], d_dec=args["d_dec"],
                    dec_layers=args["dec_layers"],
                    n_fourier=args["n_fourier"],
                    n_extra=int(args.get("n_extra", 0)))
    missing, unexpected = fresh.load_state_dict(ck["model"], strict=True)
    assert not missing and not unexpected
    assert fresh.dot_proj.weight.shape == (32, 2)
    assert fresh.param_count() == old.param_count()
    for a, b in zip(fresh.state_dict().values(), old.state_dict().values()):
        assert torch.equal(a, b)
    # and the widened one is a DIFFERENT architecture, by exactly 3 * d_model
    wide = ConeMAE(6, d_model=32, n_heads=4, n_latents=8, n_layers=1, d_z=8,
                   d_dec=32, dec_layers=1, n_fourier=4, n_extra=3)
    assert wide.param_count() - old.param_count() == 3 * 32
    with pytest.raises(RuntimeError):
        wide.load_state_dict(ck["model"], strict=True)


def test_the_extras_reach_the_model_and_zeroing_one_is_not_a_rebuild(tmp_path):
    """`--dot-extras` changes what the model is TOLD, not how big it is.

    Naming any extra widens the dot token to [value, observed, nr, fp0, fp1];
    naming a subset zeroes the rest. So the arms of E-076 §5's ladder — the
    footprint zeroed, `n_R` withheld — are the same architecture and a
    difference between them cannot be a difference in parameter count.
    """
    import torch
    from cone_codec import ConeMAE
    from train_cone import dot_extras_of, to_torch
    assert dot_extras_of("") == (0, ())
    assert dot_extras_of("nr") == (3, ("nr",))
    assert dot_extras_of("fp,nr") == (3, ("nr", "fp"))
    with pytest.raises(SystemExit):
        dot_extras_of("nr,depth")

    s, gs, st = _sampler(tmp_path, k=3)
    smp = s.sample(np.array([[33, 22, 30]], np.int64))
    depth = torch.as_tensor([0.0] * len(gs.chan), dtype=torch.float32)
    b_all = to_torch(smp, depth, "cpu", ("nr", "fp"))
    b_nr = to_torch(smp, depth, "cpu", ("nr",))
    b_none = to_torch(smp, depth, "cpu", ())
    assert b_all["dot_extra"].shape[-1] == 3
    assert b_nr["dot_extra"].shape[-1] == 3
    assert torch.equal(b_nr["dot_extra"][..., 0], b_all["dot_extra"][..., 0])
    assert not b_nr["dot_extra"][..., 1:].any()
    assert b_all["dot_extra"][..., 1:].any()
    assert "dot_extra" not in b_none

    torch.manual_seed(0)
    m = ConeMAE(len(gs.chan), d_model=16, n_heads=2, n_latents=4, n_layers=1,
                d_z=8, d_dec=16, dec_layers=1, n_fourier=4, n_extra=3).eval()
    with torch.no_grad():
        z_all, _ = m.encode(b_all)
        z_nr, _ = m.encode(b_nr)
    assert not torch.equal(z_all, z_nr), (
        "zeroing the footprint changed nothing about z — the extras are not "
        "reaching the token")


def test_the_withheld_profile_becomes_a_query_family(tmp_path):
    """Masked-profile reconstruction rides the EXISTING dot-query machinery:
    one more family on the query axis, weighted like every other query on
    those channels, and absent entirely from a batch that withheld nothing."""
    import torch
    from cone_codec import ConeMAE, default_plan, FAMILY_W
    from train_cone import to_torch
    s, gs, st = _sampler(tmp_path, k=3)
    a = np.array([[33, 22, 30], [30, 10, 20]], np.int64)
    depth = torch.as_tensor([0.0] * len(gs.chan), dtype=torch.float32)
    plan = default_plan(gs.chan, n_dot_queries=8)
    m = ConeMAE(len(gs.chan), d_model=16, n_heads=2, n_latents=4, n_layers=1,
                d_z=8, d_dec=16, dec_layers=1, n_fourier=4).eval()

    b0 = to_torch(s.sample(a), depth, "cpu")
    with torch.no_grad():
        out0 = m(b0, plan)
    assert set(out0["families"]) == {"anchor", "future", "dots"}

    b1 = to_torch(s.sample(a, withhold_nearest=True), depth, "cpu")
    with torch.no_grad():
        out1 = m(b1, plan)
    fam = out1["families"]
    assert set(fam) == {"anchor", "future", "dots", "profile"}
    assert fam["profile"]["n_targets"] > 0
    # the weight is the channel's own family weight — B, for the rg column
    n = float(b1["pq_obs"].sum())
    assert abs(fam["profile"]["wsum"] - n * FAMILY_W["B"]) < 1e-4
    # the split still reassembles the headline exactly
    den = sum(f["wsum"] for f in fam.values())
    got = sum(f["nll"] * f["wsum"] for f in fam.values()) / den
    assert abs(got - out1["terms"]["nll"]) < 1e-5 * max(1.0, abs(got))
    # and the spans tile the query axis with `profile` LAST
    spans = m.query_family_spans(b1, plan, 1000)
    assert list(spans)[-1] == "profile"
    assert spans["profile"][1] == 1000


# ================================================== 6. the twin and the eval ==
def test_profile_k_zero_keeps_the_grid_and_still_has_the_target(tmp_path):
    """The TWIN of E-076 §5.1. With `--profile-k 0` the tensor's `rg100` group
    is read exactly as it always was — same dots, same patch — and the store
    supplies only the nearest-profile target, so both arms can be scored by
    one implementation on one measurement."""
    gs, rows = _groups()
    st = _store(tmp_path)
    twin = ConeSampler(gs, None, gs.lats, gs.lons, gs.chan, L_in=2,
                       profile_store=st, profile_k=0, profile_norm=_norm(),
                       profile_stats=_flat_stats(),
                       profile_bin_index=B0 + np.arange(T))
    plain = ConeSampler(gs, None, gs.lats, gs.lons, gs.chan, L_in=2)
    a = np.array([[30, 20, 24]], np.int64)
    A, B = twin.sample(a), plain.sample(a)
    for key in ("vals", "obs", "valid", "chan", "dy_km", "dx_km", "lag_days",
                "depth", "patch_vals", "patch_obs", "fut_vals", "fut_obs"):
        assert np.array_equal(A[key], B[key]), key
    assert twin.n_profile_dots == 0
    held = twin.sample(a, withhold_nearest=True)
    pt = held["profile_target"]
    assert pt["valid"][0]
    assert np.isfinite(pt["temp"][0]).any()
    # ...and the PERSISTENCE baseline too, which is a scoring quantity and not
    # an input: k = 0 still fetches slot 1, or the twin would be scored
    # against two baselines where the family-8 arm has three.
    assert pt["pers_valid"][0]
    assert np.isfinite(pt["pers_temp"][0]).any()
    # withholding put NOTHING in the twin's input — same dots as the plain
    # sampler, still
    assert np.array_equal(held["vals"], B["vals"])
    assert np.array_equal(held["valid"], B["valid"])


def test_a_store_that_does_not_match_its_own_hashes_is_refused(tmp_path):
    st = _store(tmp_path, n=20, name="bad")
    assert verify_store(st.path) == 12
    with open(os.path.join(st.path, "lat.npy"), "r+b") as fh:
        fh.seek(-4, os.SEEK_END)
        fh.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="does not match its own store.json"):
        verify_store(st.path)


# ============================================================== 7. end to end ==
def test_train_cone_smoke_family8_end_to_end(tmp_path):
    """The whole arm on two CPU cores: a three-group tensor, a store, the
    sparse gather, a withheld profile in the loss and the common-target eval
    with its three baselines."""
    out = tmp_path / "f8"
    cmd = [sys.executable, os.path.join(ML, "train_cone.py"),
           "--smoke-family8", "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, f"--smoke-family8 failed:\n{p.stdout}\n{p.stderr}"
    assert "sha256-verified against store.json" in p.stdout
    assert "profile dot tokens per anchor" in p.stdout

    lines = [json.loads(ln) for ln in
             open(out / "metrics.jsonl").read().splitlines() if ln.strip()]
    cfg = lines[0]["config"]
    assert cfg["n_extra"] == 3 and cfg["profile_k"] == 5
    assert cfg["n_profile_dots"] == 5 * 32
    evals = [r for r in lines if "held_out_nll" in r]
    assert evals[-1]["held_out_nll"] < evals[0]["held_out_nll"]

    recs = [r["profile_target"] for r in lines if "profile_target" in r]
    assert len(recs) >= 2, "the common-target eval ran fewer than twice"
    rec = json.load(open(out / "profile_target.json"))
    assert "in_progress" not in rec, "the final write is not marked partial"
    assert rec["step"] == cfg["steps"] and rec["n"] > 0
    assert rec["levels"] == LEVELS
    for key in ("rmse_t", "rmse_s", "clim_rmse_t", "clim_rmse_s",
                "pers_rmse_t", "pers_rmse_s"):
        v = rec[key]
        assert len(v) == 16, key
        assert all(x is not None and np.isfinite(x) and x >= 0 for x in v), key
    assert np.isfinite(rec["skill_t"]) and np.isfinite(rec["skill_s"])
    # RMSE grows with the signal, which decays with depth: the deep levels of
    # this fixture are quieter than the shallow ones, for all three read-outs.
    for key in ("rmse_t", "clim_rmse_t", "pers_rmse_t"):
        assert rec[key][0] > rec[key][-1], key
    assert set(rec["heldout_split"]) == {"2021-2024", "interspersed"}
    assert rec["heldout_split"]["interspersed"]["n"] > 0
    # nothing in either results file is a bare NaN token (ml/CLAUDE.md §5.22)
    for f in ("profile_target.json", "metrics.jsonl"):
        txt = open(out / f).read()
        assert "NaN" not in txt and "Infinity" not in txt, f


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_family_weights_cover_every_family7_channel():
    """Every channel of the REAL family-7 tensor (data/family7_index.json)
    must resolve to a loss weight — #549/#550/#551 died on `KeyError: 'L'`
    after the transform, the first time land channels met the cone codec."""
    import json
    import os
    from cone_codec import family_weights
    idx = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                      "data", "family7_index.json")))
    names = [c for g in ("g025", "g100", "rg100")
             for c in idx["groups"][g]["chans"]]
    assert len(names) == 54
    w = family_weights(names)
    assert len(w) == 54 and all(x > 0 for x in w)
