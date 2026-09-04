#!/usr/bin/env python3
"""E-069 · `ml/export_cone_sample.py` reads what the model reads, and nothing else.

The exporter's whole claim is that it CALLS the production code — `ConeSampler`
for the codec's inner stencil, `cone.outer_spiral` for stage 2's outer one, and
`trainprobe.anomaly_transform`'s arithmetic for the value space — rather than
reimplementing any of it. A claim like that is worth exactly as much as the
test that would fail if it stopped being true, so:

  * the schema is pinned (shapes agree with the arrays, dims agree with the
    shapes, no NaN token anywhere);
  * the DOT COUNTS come out equal to `data/cone_geometry.json`'s own `counts`
    for the r3 channel list, which is `ml/cone.py::budget`'s answer;
  * `valid` and `obs` are compared BIT FOR BIT against a direct
    `ConeSampler.sample` call on the same tensor;
  * `streaming_anomaly` is compared BIT FOR BIT against the real
    `anomaly_transform`, lifted out of `ml/trainprobe.py` by `ast` the way
    `ml/cone.py` lifts the spiral — so the check runs on a box with no torch,
    which is where it has to run.

    python3 -m pytest -q tests/test_export_cone_sample.py
"""
import ast
import json
import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

import cone                                                        # noqa: E402
from cone_sampler import ConeSampler                               # noqa: E402
import export_cone_sample as X                                     # noqa: E402

GEO = json.load(open(os.path.join(ROOT, "data", "cone_geometry.json"),
                     encoding="utf-8"))


def _lift_anomaly_transform():
    """`trainprobe.anomaly_transform` WITHOUT importing `ml/trainprobe.py`,
    which pulls in torch at module scope.

    Exactly `ml/cone.py::_lift_spiral_from_temporal`'s trick and for exactly
    its reason: the function under comparison must be THE function, byte for
    byte, and on a CPU box with no torch an import is not available to get it.
    """
    src = open(os.path.join(ML, "trainprobe.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "anomaly_transform"]
    if not keep:
        raise ImportError(
            "ml/trainprobe.py no longer defines anomaly_transform at module "
            "scope — tests/test_export_cone_sample.py lifts it from source so "
            "the streaming copy cannot drift. Fix the lift, never copy it.")
    ns = {"np": np, "warnings": warnings}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "trainprobe.py",
                 "exec"), ns)
    return ns["anomaly_transform"]


# --------------------------------------------------------------- the fixture --
def toy(seed=0, T=200, H=40, W=48, chans=None):
    """The same recipe `tests/test_cone_smoke.py::tiny_sampler` uses — random
    normal with 5 % of cells NaN — long enough in time to carry a 143-pentad
    outer cone. `tests/test_cone_smoke.py` cannot be imported here (it imports
    torch at module scope), so the recipe is repeated and `test_toy_matches_
    the_smoke_recipe` pins that the two have not drifted."""
    return X.smoke_tensor(seed=seed, T=T, H=H, W=W, chans=chans)


def export_toy(tmp_path, **kw):
    out = str(tmp_path / "s")
    written = X.main(["--smoke", "--out", out] +
                     [a for k, v in kw.items()
                      for a in (f"--{k.replace('_', '-')}", str(v))])
    return out, written


def test_toy_matches_the_smoke_recipe():
    """The synthetic tensor here IS the one tests/test_cone_smoke.py builds:
    same generator, same seed, same 5 % NaN rule. Asserted on the leading
    block so the two files cannot drift into testing different data."""
    A, chans = toy(seed=0, T=30, H=12, W=14,
                   chans=["cur_speed", "log_mld", "ssh", "tau_x", "tau_y",
                          "sst", "cur_u", "cur_v"])
    rng = np.random.default_rng(0)
    B = rng.normal(size=(30, 12, 14, 8)).astype(np.float32)
    B[rng.random(B.shape) < 0.05] = np.nan
    assert np.array_equal(A, B, equal_nan=True)


# ------------------------------------------------------------------ 1. schema --
def test_schema_and_shapes(tmp_path):
    out, written = export_toy(tmp_path)
    assert len(written) == 2
    for anc, path, text in written:
        s = json.loads(text)
        assert set(s) == {"meta", "inner", "patch", "future", "outer"}
        m, inner, patch, fut, outer = (s["meta"], s["inner"], s["patch"],
                                       s["future"], s["outer"])

        # meta says what produced it, where it came from, and when
        assert "ConeSampler.sample" in m["produced_by"]
        assert "outer_spiral" in m["produced_by"]
        assert m["exporter"] == "ml/export_cone_sample.py"
        assert len(m["dates"]) == len(m["bins"]) == 24
        assert m["dates"] == sorted(m["dates"])
        assert m["holdout_years"] == [2009, 2017, 2023]
        assert m["outer"]["empty_below"] == 7          # the design, not a gap
        assert m["outer"]["first"] == 7
        assert m["geometry"]["constants"]["L_IN"] == 6
        assert m["anchor"]["row"] >= 0 and m["anchor"]["col"] >= 0
        assert len(m["admissible"]) == len(m["dates"])

        # every array is exactly as long as the `shape` it declares
        nT, nD = inner["shape"]
        assert nT == len(m["dates"]) and nD == inner["n_dots"]
        for k in ("lag", "dy", "dx", "chan", "row", "col", "lat", "lon",
                  "dy_km", "dx_km", "lag_days"):
            assert len(inner[k]) == nD, k
        assert len(inner["raw"]) == len(inner["anom"]) == nT
        assert all(len(r) == nD for r in inner["raw"])
        assert len(inner["obs"]) == len(inner["valid"]) == nT * nD
        assert set(inner["obs"]) <= {"0", "1"}

        assert patch["shape"] == [nT, len(m["channels"]), 9]
        assert len(patch["raw"]) == nT * len(m["channels"]) * 9
        assert len(patch["obs"]) == len(patch["raw"])
        assert fut["shape"][2] == len(fut["lags"]) == 2
        assert len(fut["raw"]) == nT * len(m["channels"]) * 2

        oT, oK, oD, oC = outer["shape"]
        assert oT == nT and oK == len(outer["lags"]) and oC == len(
            outer["channels"])
        assert outer["lags"][0] == 7 and outer["lags"][-1] <= 143
        for k in ("dy", "dx", "row", "col", "lat", "lon", "dy_km", "dx_km"):
            assert len(outer[k]) == oK * oD, k
        assert len(outer["valid"]) == oK * oD
        assert len(outer["raw"]) == len(outer["anom"]) == oT * oK * oD * oC
        assert len(outer["obs"]) == oT * oK * oD * oC

        # NaN is `null`, never a bare token no strict JSON reader parses
        assert "NaN" not in text and "Infinity" not in text
        # deterministic: sorted keys, and a re-dump of the parsed object is
        # byte-identical to the file
        assert X.dumps(s) == text


def test_lag_zero_is_the_patch_and_the_dots_start_at_one(tmp_path):
    """Lag 0 is NOT a dot: the codec keeps `PixelMAE`'s 3x3 patch there, one
    token per channel, so every archived comparison stays like-for-like."""
    _, written = export_toy(tmp_path)
    s = json.loads(written[0][2])
    assert min(s["inner"]["lag"]) == 1
    assert max(s["inner"]["lag"]) == s["meta"]["L_in"]
    assert s["patch"]["cell_dy"] == [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    assert s["patch"]["cell_dx"] == [-1, 0, 1, -1, 0, 1, -1, 0, 1]


# ------------------------------------------------------------- 2. dot counts --
def test_dot_counts_match_cone_geometry_counts():
    """The exporter's dot table IS `ml/cone.py::budget`'s arithmetic.

    `data/cone_geometry.json` carries the count for the 42-channel r3 list at
    40 N — 42 patch tokens + 706 dots = 748 — and the sampler the exporter
    calls must reproduce it exactly. If the two ever disagree, one of them is
    describing a cone the model does not read.
    """
    chans = list(GEO["channels_r3"])
    C = len(chans)
    win = GEO["window"]
    H, W, T = 12, 12, 8
    A = np.zeros((T, H, W, C), np.float32)
    lats = 40.0 + win["dlat"] * np.arange(H)
    lons = -40.0 + win["dlat"] * np.arange(W)
    s = ConeSampler(A, np.isfinite(A), lats, lons, chans,
                    L_in=GEO["constants"]["L_IN"], dlat_deg=win["dlat"])
    counts = GEO["counts"]
    assert counts["n_channels"] == C
    assert s.n_dots(0) == counts["dot_tokens"] == 706
    assert counts["patch_tokens"] == C == 42
    assert counts["total_tokens"] == 748
    # and the per-family split the tab's tiles quote
    row = s.row(0)
    per = {}
    for ci, name in enumerate(chans):
        fam = "rg" if cone.is_depth_channel(name) else cone.channel_family(name)
        per.setdefault(fam, set()).add(int((row["chan"] == ci).sum()))
    assert per["A"] == {counts["inner_dots_A"]}
    assert per["B"] == {counts["inner_dots_B"]}
    assert per["C"] == {counts["inner_dots_C"]}
    assert per["rg"] == {counts["inner_dots_rg"]}


def test_outer_dot_count_and_the_empty_annulus(tmp_path):
    """Every exported outer lag carries `OUTER_N_PTS` dots, and lags 0-6 are
    absent because `cone.outer_spiral` returns [] there by construction."""
    _, written = export_toy(tmp_path)
    s = json.loads(written[0][2])
    o = s["outer"]
    assert o["n_dots_per_lag"] == GEO["constants"]["OUTER_N_PTS"] == 24
    lat = s["meta"]["anchor"]["lat"]
    for k in (0, 3, 6):
        assert cone.outer_spiral(lat, k, dlat_deg=0.25, L_in=6) == []
    assert all(k >= 7 for k in o["lags"])
    # the exported offsets ARE outer_spiral's, in its order
    nD = o["n_dots_per_lag"]
    for kk, k in enumerate(o["lags"][:6]):
        want = cone.outer_spiral(lat, int(k), dlat_deg=0.25, L_in=6)
        got = list(zip(o["dy"][kk * nD:(kk + 1) * nD],
                       o["dx"][kk * nD:(kk + 1) * nD]))
        assert got == [(int(a), int(b)) for a, b in want], k


# ----------------------------------------------- 3. valid / obs, bit for bit --
def test_valid_and_obs_agree_with_ConeSampler_bit_for_bit(tmp_path):
    """The exporter must CALL the sampler, never reimplement it.

    Rebuild the same sampler over the same toy tensor, sample the same
    anchors, and compare the flags the file carries against the arrays the
    sampler returns — element for element, both directions.
    """
    out, written = export_toy(tmp_path)
    A, chans = toy()
    win = dict(GEO["window"])
    T, H, W, C = A.shape
    win.update(lat0=30.0, lon0=-40.0, ny=H, nx=W)
    lats = win["lat0"] + win["dlat"] * np.arange(H)
    lons = win["lon0"] + win["dlat"] * np.arange(W)
    s = ConeSampler(A, np.isfinite(A), lats, lons, chans, L_in=6,
                    dlat_deg=win["dlat"])
    for anc, path, text in written:
        j = json.loads(text)
        y, x = j["meta"]["anchor"]["row"], j["meta"]["anchor"]["col"]
        bins = [b for b in j["meta"]["bins"]]
        got = s.sample(np.array([[t, y, x] for t in bins], np.int64))
        n = s.n_dots(y)
        assert n == j["inner"]["n_dots"]
        assert X.bits(got["valid"][:, :n]) == j["inner"]["valid"]
        assert X.bits(got["obs"][:, :n]) == j["inner"]["obs"]
        assert X.bits(got["patch_obs"]) == j["patch"]["obs"]
        assert X.bits(got["fut_obs"]) == j["future"]["obs"]
        # the VALUES too, at the exporter's four significant figures
        assert [[X.sig(v) for v in r] for r in got["vals"][:, :n]] \
            == j["inner"]["raw"]
        assert [X.sig(v) for v in got["patch_vals"].ravel()] == j["patch"]["raw"]
        assert [X.sig(v) for v in got["fut_vals"].ravel()] == j["future"]["raw"]
        # a toy anchor near the east edge must actually exercise the invalid
        # branch, or the comparison above is vacuous there
        if anc["id"] == "smoke_edge":
            assert "0" in j["inner"]["valid"] or "0" in j["outer"]["valid"]


def test_outer_values_are_the_tensor_at_those_cells(tmp_path):
    """Spot-check the outer stencil against the tensor directly: the value at
    (t - k, row, col, channel) is what the file carries, or it is null where
    the dot is off the window."""
    _, written = export_toy(tmp_path)
    A, chans = toy()
    j = json.loads(written[0][2])
    o = j["outer"]
    nT, nK, nD, nC = o["shape"]
    ci = o["chan_index"]
    bins = j["meta"]["bins"]
    checked = 0
    for ti in (0, nT - 1):
        for kk in (0, nK // 2, nK - 1):
            for dd in (0, nD - 1):
                idx = kk * nD + dd
                flat = ((ti * nK + kk) * nD + dd) * nC
                if o["valid"][idx] == "0":
                    assert all(v is None for v in o["raw"][flat:flat + nC])
                    continue
                t = bins[ti] - o["lags"][kk]
                want = [X.sig(A[t, o["row"][idx], o["col"][idx], c])
                        for c in ci]
                assert o["raw"][flat:flat + nC] == want
                checked += 1
    assert checked > 0, "no valid outer dot was checked — the test is vacuous"


# ------------------------------------------------- 4. the anomaly, bit for bit --
def test_streaming_anomaly_matches_trainprobe():
    """`streaming_anomaly` == `trainprobe.anomaly_transform`, exactly.

    The production tensor is 35.7 GB decompressed and `anomaly_transform`
    rewrites it in place, which this sandbox has neither the disk nor the RAM
    for. The streaming form is the same three passes over a decompressed
    stream — and "the same" is a claim, so it is measured here on a toy where
    both can run: identical values (NaN in the same places), identical dynamic
    channel list, identical mean and sd.
    """
    anomaly_transform = _lift_anomaly_transform()
    rng = np.random.default_rng(3)
    T, H, W, C = 60, 7, 9, 5
    A = rng.normal(size=(T, H, W, C)).astype(np.float16)
    A[rng.random(A.shape) < 0.08] = np.nan
    A[:, 0, 0, 0] = np.nan                       # a permanently-dry cell
    A[..., 4] = 2.5                              # a STATIC channel: passes through
    moy = np.array([(t // 6) % 12 for t in range(T)])
    t_hold = np.zeros(T, bool)
    t_hold[10:16] = True
    x_hold = np.zeros(W, bool)

    ref = A.copy()
    ref, dyn_ref = anomaly_transform(ref, moy, t_hold, x_hold, chunk=7,
                                     verbose=False)

    def blocks(chunk):
        for t0 in range(0, T, chunk):
            yield t0, min(t0 + chunk, T), A[t0:min(t0 + chunk, T)]

    for chunk in (7, 13, T):
        raw, anom, dyn, mu, sd, ocean = X.streaming_anomaly(
            blocks, A.shape, A.dtype, moy, t_hold, x_hold, 0, T,
            chunk=chunk, verbose=False)
        assert dyn == dyn_ref, chunk
        assert np.array_equal(raw, A, equal_nan=True), chunk
        assert np.array_equal(anom, ref, equal_nan=True), (
            f"chunk {chunk}: the streaming anomaly is not the trainer's — "
            f"max |diff| "
            f"{np.nanmax(np.abs(anom.astype(np.float64) - ref.astype(np.float64)))}")
        assert ocean.shape == (H, W)
        assert not ocean[0, 0] and ocean.sum() == H * W - 1
        # the static channel is untouched, which is what `stat` is for
        assert np.array_equal(anom[..., 4], A[..., 4])


def test_streaming_anomaly_keeps_only_the_requested_bins():
    """The whole point of the streaming form: the slab is the bins asked for,
    and they are the same numbers a full transform would have written there."""
    anomaly_transform = _lift_anomaly_transform()
    rng = np.random.default_rng(5)
    T, H, W, C = 40, 5, 6, 3
    A = rng.normal(size=(T, H, W, C)).astype(np.float32)
    A[rng.random(A.shape) < 0.1] = np.nan
    moy = np.array([(t // 4) % 12 for t in range(T)])
    t_hold = np.zeros(T, bool)
    t_hold[:4] = True
    ref = A.copy()
    ref, _ = anomaly_transform(ref, moy, t_hold, np.zeros(W, bool), chunk=5,
                               verbose=False)

    def blocks(chunk):
        for t0 in range(0, T, chunk):
            yield t0, min(t0 + chunk, T), A[t0:min(t0 + chunk, T)]

    raw, anom, dyn, mu, sd, ocean = X.streaming_anomaly(
        blocks, A.shape, A.dtype, moy, t_hold, np.zeros(W, bool), 12, 20,
        chunk=5, verbose=False)
    assert anom.shape == (8, H, W, C)
    assert np.array_equal(anom, ref[12:20], equal_nan=True)
    assert np.array_equal(raw, A[12:20], equal_nan=True)


# ----------------------------------------------------------------- 5. fixture --
def test_trim_sample_keeps_the_schema_and_the_values(tmp_path):
    """The in-repo fixture must be the SAME FORMAT as the real files.

    A fixture with its own shape would let a browser test pass against a
    schema the deployed page never sees. So: same keys, same dims, declared
    shapes that match the array lengths, and every kept value equal to the
    value it was cut from.
    """
    _, written = export_toy(tmp_path)
    full = json.loads(written[0][2])
    fx = X.trim_sample(full, n_dates=3, want_lags=(7, 36, 143))

    assert set(fx) == set(full)
    for blk in ("inner", "patch", "future", "outer"):
        assert set(fx[blk]) == set(full[blk]), blk
    assert len(fx["meta"]["dates"]) == 3
    assert fx["meta"]["dates"] == full["meta"]["dates"][:3]
    assert len(fx["outer"]["lags"]) == 3
    assert fx["outer"]["lags"][0] == 7 and fx["outer"]["lags"][-1] == 143

    nD = fx["inner"]["n_dots"]
    assert fx["inner"]["shape"] == [3, nD]
    assert fx["inner"]["raw"] == full["inner"]["raw"][:3]
    assert len(fx["inner"]["obs"]) == 3 * nD
    nC = len(fx["meta"]["channels"])
    assert fx["patch"]["shape"] == [3, nC, 9]
    assert len(fx["patch"]["raw"]) == 3 * nC * 9 == len(fx["patch"]["obs"])
    assert len(fx["future"]["raw"]) == 3 * nC * 2

    oT, oK, oDd, oC = fx["outer"]["shape"]
    assert [oT, oK] == [3, 3]
    assert len(fx["outer"]["raw"]) == oT * oK * oDd * oC
    assert len(fx["outer"]["obs"]) == oT * oK * oDd * oC
    assert len(fx["outer"]["valid"]) == oK * oDd
    for k in ("dy", "dx", "row", "col", "lat", "lon"):
        assert len(fx["outer"][k]) == oK * oDd, k

    # every kept outer value IS the value it was cut from
    nK0 = len(full["outer"]["lags"])
    keep = [full["outer"]["lags"].index(k) for k in fx["outer"]["lags"]]
    for t in range(3):
        for i, ki in enumerate(keep):
            for d in range(oDd):
                a = ((t * nK0 + ki) * oDd + d) * oC
                b = ((t * oK + i) * oDd + d) * oC
                assert fx["outer"]["raw"][b:b + oC] == \
                    full["outer"]["raw"][a:a + oC]
                assert fx["outer"]["obs"][b:b + oC] == \
                    full["outer"]["obs"][a:a + oC]
    # and the trim did not touch the original
    assert len(full["meta"]["dates"]) == 24


# --------------------------------------------------------------- 6. the sig() --
def test_sig_rounds_and_nulls():
    assert X.sig(None) is None
    assert X.sig(float("nan")) is None
    assert X.sig(float("inf")) is None
    assert X.sig(0.0) == 0.0
    assert X.sig(-0.0) == 0.0
    assert X.sig(1.23456789) == 1.235
    assert X.sig(-1.23456789e-5) == -1.235e-5
    assert X.sig(123456.0) == 123500.0
    assert X.bits(np.array([True, False, True])) == "101"


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
