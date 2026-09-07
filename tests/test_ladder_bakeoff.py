#!/usr/bin/env python3
"""Pins for E-074a's ladder bake-off (ml/ladder_bakeoff.py).

Four things are pinned, each because getting it wrong produces a number that
looks decisive and is not:

  1. alpha = 1 reproduces the plain quantiles. The whole grid is a
     one-parameter family around that point; if its own special case drifts,
     every alpha is measuring something else.
  2. Every scheme's probabilities over the common fine partition sum to 1.
     A cross-entropy against an unnormalised object is not a cross-entropy.
  3. The empirical piecewise-uniform model beats a single Gaussian on its own
     fit sample. If it does not, the fitting is broken, not the theory.
  4. The --smoke run produces every key the results file is supposed to carry.

Run:  python3 -m pytest tests/test_ladder_bakeoff.py -q
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import ladder_bakeoff as lb  # noqa: E402


# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample():
    """A continuous, decidedly non-Gaussian sample: the kind of right-skewed
    thing cur_speed and log_prate actually are."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.lognormal(0.0, 0.6, 300_000),
                        3.0 + rng.gamma(2.0, 0.8, 60_000)])
    return np.sort(x.astype(np.float64))


@pytest.fixture(scope="module")
def model(sample):
    m = lb.fit_channel(sample)
    assert m is not None
    return m


# --- 1 · alpha = 1 IS the plain quantile ladder ----------------------------

def test_alpha_one_reproduces_plain_quantiles(sample, model):
    part = model["part"]
    # the probabilities the clip range spans, measured on the sample itself
    n = len(sample)
    p_lo = np.searchsorted(sample, part["clip_lo"], side="left") / n
    p_hi = np.searchsorted(sample, part["clip_hi"], side="left") / n
    span = part["clip_hi"] - part["clip_lo"]
    for K1 in (16, 64, 256):
        e = lb.warp_edges(part, 1.0, K1)
        targets = p_lo + (p_hi - p_lo) * (np.arange(1, K1) / K1)
        ref = np.quantile(sample, targets)
        # same edges to a small fraction of the range
        assert np.max(np.abs(e - ref)) < 2e-3 * span, K1
        # and, the meaningful form: the empirical CDF at each edge is where it
        # should be, to well inside one fine cell's worth of probability
        got = np.searchsorted(sample, e, side="left") / n
        assert np.max(np.abs(got - targets)) < 1e-3, K1


def test_alpha_zero_is_uniform_in_x(model):
    """The other end of the family: alpha = 0 must give equal-width bins."""
    part = model["part"]
    e = lb.warp_edges(part, 0.0, 64)
    B = np.concatenate([[part["clip_lo"]], e, [part["clip_hi"]]])
    w = np.diff(B)
    assert np.max(np.abs(w - w.mean())) < 1e-9 * (part["clip_hi"] - part["clip_lo"])


# --- 2 · every scheme is a probability distribution over the partition -----

def test_every_scheme_sums_to_one(model):
    assert len(model["fitted"]) >= len(lb.SCHEMES)  # + the uniform reference
    n_cells = model["part"]["n_cells"]
    for name, f in model["fitted"].items():
        p = np.exp(f["logp"])
        assert p.shape == (n_cells,), name
        assert np.all(p > 0), name
        assert abs(p.sum() - 1.0) < 1e-9, (name, p.sum())


def test_ladder_piece_masses_sum_to_one(model):
    for name, f in model["fitted"].items():
        if "pi" not in f:
            continue
        assert abs(float(f["pi"].sum()) - 1.0) < 1e-9, name
        assert len(f["pi"]) == f["spec"]["K1"] * (1 if f["spec"]["depth"] == 1
                                                  else lb.K2), name
        assert np.all(np.diff(f["B"]) >= 0), name


# --- 3 · the empirical model beats the Gaussian on its own fit sample ------

def _ce_on(model, name, x):
    part = model["part"]
    xc = np.clip(x, part["clip_lo"], part["clip_hi"])
    ci = np.minimum(np.searchsorted(part["v"], xc, side="right"),
                    len(part["v"]) - 1)
    h = np.bincount(ci, minlength=part["n_cells"])
    return float(-np.sum(h * model["fitted"][name]["logp"]) / h.sum())


def test_empirical_piecewise_beats_gaussian_on_its_own_sample(sample, model):
    g = _ce_on(model, "gaussian", sample)
    for K1 in (16, 64, 256):
        for an in lb.ALPHA_NAMES:
            ce = _ce_on(model, f"a{an}_K{K1}_d1", sample)
            assert ce <= g + 1e-9, (an, K1, ce, g)


def test_uniform_over_cells_is_the_strong_reference(sample, model):
    """The fine partition is (near-)equiprobable under the fit sample, so
    'uniform over cells' is the sample's own empirical description at that
    resolution — a lower bar to beat, not a straw man. Pin that it really is
    close to log(n_cells)."""
    ce = _ce_on(model, "uniform_cells", sample)
    assert abs(ce - math.log(model["part"]["n_cells"])) < 1e-9


def test_depth2_matches_depth1_at_matched_pieces(sample, model):
    """The whitening argument of the plan, section 1.2, on clean continuous
    data: 64 x 64 uniform sub-bins should describe the sample about as well as
    4,096 bins placed directly."""
    for an in ("0", "1"):
        d2 = _ce_on(model, f"a{an}_K64_d2", sample)
        d1 = _ce_on(model, f"a{an}_K{lb.K1_MATCHED}_d1", sample)
        assert abs(d2 - d1) / abs(d1) < 0.02, (an, d2, d1)


# --- 4 · the smoke run produces every key ---------------------------------

@pytest.fixture(scope="module")
def smoke_run():
    out = tempfile.mkdtemp(prefix="e074a_smoke_")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "ml", "ladder_bakeoff.py"),
                        "--smoke", "--out", out,
                        "--fixture", os.path.join(ROOT, "data", "family7", "fixture")],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-4000:]
    with open(os.path.join(out, "E074a_results.json")) as fh:
        js = json.load(fh)
    with open(os.path.join(out, "E074a_results.md")) as fh:
        md = fh.read()
    return js, md


def test_smoke_top_level_keys(smoke_run):
    js, _ = smoke_run
    for k in ("experiment", "generated_utc", "comparability_note",
              "regulariser_note", "units_note", "sampling", "masks", "grid",
              "alpha1_equals_plain_quantiles", "fit_value_counts",
              "skipped_channels", "channels", "timings_s", "falsifiers"):
        assert k in js, k
    assert js["experiment"] == "E-074a"
    for k in ("fit_pairs_pentad", "eval_pairs_pentad", "fit_pairs_rg100_rows",
              "eval_pairs_rg100_rows", "seed", "spatial_subsample_of_fit_bins"):
        assert k in js["sampling"], k
    for k in ("F1_alpha1_vs_alpha0_heldout_nats",
              "F2_conditional_entropy_flat_in_alpha",
              "F3_tail_rmse_cur_speed",
              "F4_depth2_vs_depth1_matched"):
        assert k in js["falsifiers"], k
    assert js["timings_s"]["total_s"] > 0


def test_smoke_channel_keys(smoke_run):
    js, _ = smoke_run
    ch = js["channels"]["g025"]["cur_speed"]
    for k in ("unit", "n_fine_cells", "clip_lo_z", "clip_hi_z", "fit_n",
              "heldout_n", "heldout_n_tail", "heldout_n_rapid",
              "clipped_fraction", "schemes", "persistence"):
        assert k in ch, k
    names = set(ch["schemes"])
    for s in lb.SCHEMES:
        assert s["name"] in names, s["name"]
    assert "uniform_cells" in names
    d1 = ch["schemes"]["a1_K64_d1"]
    for k in ("ce_nats", "alpha", "K1", "depth", "tail_exception", "n_pieces",
              "rmse_all", "rmse_tail", "rmse_rapid",
              "floor_all", "floor_tail", "floor_rapid",
              "calib_max_abs_dev", "calib_mean_abs_dev"):
        assert k in d1, k
    for an in lb.ALPHA_NAMES:
        for K1 in lb.K1S:
            e = ch["persistence"][f"a{an}_K{K1}"]
            for k in ("n", "persistence", "h_cond", "h_marg", "ratio"):
                assert k in e, (an, K1, k)


def test_smoke_masks_and_markdown(smoke_run):
    js, md = smoke_run
    m = js["masks"]
    assert m["rapid_cols_g025"][0] <= m["rapid_cols_g025"][1]
    for h in ("# E-074a", "How to read this", "Held-out cross-entropy",
              "conditional entropy", "quantization floor", "Calibration",
              "Verdict block"):
        assert h in md, h
    assert "F1" in md and "F2" in md and "F3" in md and "F4" in md


def test_rapid_section_indices_on_the_real_grid():
    """Row 466 is 26.5 N and columns 400-668 are 80 W to 13 W on the 0.25 deg
    grid — the RAPID array's section, and the one place in this experiment
    where a wrong index would still look like a perfectly plausible ocean."""
    row, c0, c1 = lb.rapid_indices(721, 1440, -90.0, -180.0, 0.25)
    assert (row, c0, c1) == (466, 400, 668)
    # and its coarse image through the tensor's own round-half-up lookup rule
    y1, x1 = lb.coarse_of(np.full(c1 - c0 + 1, row), np.arange(c0, c1 + 1),
                          4, 181, 360)
    assert set(y1.tolist()) == {117}
    assert (int(x1.min()), int(x1.max())) == (100, 167)


def test_sampling_respects_the_holdout_years():
    fit, ev = lb.pentad_pair_sets()
    assert len(fit) == lb.FIT_PAIRS
    assert len(ev) == lb.EVAL_PAIRS
    for b in fit:
        for x in (b, b + 1):
            y = lb.bin_year(x)
            assert y <= lb.TERMINAL_TRAIN_LAST_YEAR and y not in lb.HOLDOUT_YEARS
    for b in ev:
        for x in (b, b + 1):
            assert lb.bin_year(x) in lb.HOLDOUT_YEARS
    assert not (set(fit) | set(b + 1 for b in fit)) & \
        (set(ev) | set(b + 1 for b in ev))
    years = sorted({lb.bin_year(b) for b in ev})
    assert years == sorted(lb.HOLDOUT_YEARS)


def test_sampling_covers_every_calendar_month():
    """The seasonal trap: 'about 1.6 pairs per year, spread evenly' picked the
    first and last candidate of each year — 1 January and 22 December — and
    left the fit sample with no Antarctic winter in it at all, which showed up
    as 3.1 % of held-out 2 m air temperature values falling outside the fit
    sample's clip range. Every calendar month must appear on both sides."""
    fit, ev = lb.pentad_pair_sets()
    for label, s in (("fit", fit), ("eval", ev)):
        h = lb._month_hist(s)
        assert min(h.values()) >= 2, (label, h)
    # and the training years are covered about evenly, 1 or 2 pairs each
    c = lb._year_counts(fit)
    assert set(c.values()) <= {1, 2}
    assert len(c) == 37          # 1982-2020 minus 2009 and 2017
