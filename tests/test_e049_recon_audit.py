#!/usr/bin/env python3
"""E-049 pre-audit: the reconstruction-audit tooling, adapted to family 4.

`ml/plans/E049_roadB_token.md` §4 names four blockers between the monthly
family-3 audit and the pentad family-4-r2 one, and says of the first that it is
SILENT. This file pins all four, plus the d_z-6/FSQ compatibility the verdict
runs through, on synthetic CPU-only data.

  1. **THE FLOAT16 ACCUMULATOR.** `stream_stats` summed a float16 memmap with
     a float16 accumulator. At family-4 shape one spatial slice is
     H*W = 135,161 values, so a channel whose mean is ~20 sums to 2.7e6
     against float16's 65,504 ceiling: the sum is `inf`, the per-bin spatial
     mean `nan`, `np.nanstd(...) > 1e-6` False, and the channel is classified
     NOT DYNAMIC and handed to the codec UN-STANDARDIZED. Check 1 runs
     `ml/recon_eval.py` AT ITS PRE-FIX REVISION beside today's, on the same
     bytes, and shows the old one drops `sst`, `rg_t*` and `rg_s*` out of
     `dynamic` while the new one matches a float64 reference to 8 decimals.
     Check 1b is its other half: on a FLOAT32 tensor the two revisions agree
     BIT FOR BIT, so no archived family-3 number moves.
  2. **THE uint8 CLIMATOLOGY COUNTER.** ~244 of 255 used at pentad cadence.
     Check 2 builds a tensor with 300 train timesteps in one month-of-year:
     uint8 wraps 300 -> 44 and the climatology reads 6.8x its true value,
     with nothing printed. uint16 gets it right.
  3. **THE .npz INPUT PATH.** Check 5 exercises all four routes through
     `open_x` — bare .npy, sidecar, uncompressed member (memmapped in place),
     and the DEFLATE member family 4 actually ships, which is refused with the
     `--extract-x` command rather than decompressed into RAM.
  4. **THE ARGO-BIN SPLIT**, the falsifier's own axis. Checks 3 and 4: the
     per-(bin, pixel) mask on a tensor whose Argo block is live one bin in six,
     and the additive scorer's keys and arithmetic.

Also here:

  6. **THE ANOMALY TRANSFORM IS THE SAME ONE `ml/train.py` USES.** The audit's
     streaming replica is compared against `ml/trainprobe.py`'s
     `anomaly_transform` — the function `train.py:515` calls — on the same
     synthetic float16 tensor. The residual is float16 STORAGE rounding and is
     measured, not assumed.
  7. **THE REFUSAL.** With the accumulator forced back to float16, the audit
     now STOPS and names the channels, instead of scoring them. Silence was
     the bug.
  8. **d_z 6 / FSQ [8,8,8,5,5,5] / --fsq-bound ln, END TO END** through
     `ml/recon_eval.py`'s own `main()` as a subprocess: a real checkpoint with
     a FITTED `auto` ladder, encoded through `embed_everything` (so the
     quantizer is in the path, exactly as it is in production), decoded, split
     and written out — including the bottleneck line the audit now states.
  9. **`ml/recon_decoder.py` END TO END** on the same bottleneck at pentad
     cadence, against a real Z cache: the DECODER-CEILING rung is where the
     falsifier is read, so it is run rather than inherited by inspection.

Nine checks (plus 1b), CPU only, ~50 s on this sandbox.
Run: python3 tests/test_e049_recon_audit.py
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import zipfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

import recon_eval as RE                                        # noqa: E402
import build_family3 as f3                                     # noqa: E402
from model import PixelMAE, codec_from_ckpt                    # noqa: E402
from trainprobe import anomaly_transform                       # noqa: E402

# The last revision in which `stream_stats` summed float16 in float16 and
# counted the climatology in uint8. Pinned rather than "HEAD", for
# tests/test_e049_fsq_bound.py's reason: HEAD stops being the archive the
# moment this change is committed.
BASE_SHA = "e207acf"

# family-4-r2's channel set, from the builder rather than typed out. r2 is
# family 3's 39 channels plus an APPENDED `sst` (ml/build_family4.py: "ONE
# definition of the channel set, imported rather than restated").
CHAN_R2 = list(f3.CHANS) + ["sst"]
C = len(CHAN_R2)                       # 40

# Realistic magnitudes, because the bug is a function of MAGNITUDE. (mean, sd)
# per channel family; the Argo temperature/salinity blocks and `sst` are the
# ones that overflow a float16 spatial sum, and they are the ones that carry
# absolute physical units.
MAG = {"cur_speed": (0.20, 0.10), "log_mld": (4.0, 0.5), "ssh": (0.5, 0.2),
       "tau_x": (0.0, 0.05), "tau_y": (0.0, 0.05),
       "tau_x_std": (0.03, 0.01), "tau_y_std": (0.03, 0.01),
       "sst": (22.0, 3.0)}

T_BINS, H, W = 292, 64, 56          # 4 years of pentads; H*W = 3,584
ARGO_EVERY = 6                      # "one bin in six" — the mid-month stamp


def chan_mag(nm):
    if nm in MAG:
        return MAG[nm]
    if nm.startswith("rg_t"):            # 20 C at the surface down to 3 C
        p = int(nm[4:])
        return (20.0 - 17.0 * min(1.0, p / 1900.0), 1.5)
    if nm.startswith("rg_s"):            # salinity, ~35 psu everywhere
        return (35.0, 0.3)
    raise AssertionError(nm)


def toy_tensor(seed=0, t_bins=T_BINS, h=H, w=W, argo_every=ARGO_EVERY,
               lat0=20.0):
    """A family-4-r2-shaped float16 tensor: [T,H,W,40], pentad labels.

    Small in T/H/W and REAL in magnitude and in sparsity structure — H*W is
    3,584, which is already 18x what a mean-20 channel needs to pass float16's
    ceiling, so the overflow reproduces at 0.4% of the real tensor's size.
    """
    rng = np.random.default_rng(seed)
    lats = (lat0 + 0.25 * np.arange(h)).astype(np.float32)      # 26.5 interior
    lons = (-70.0 + 0.25 * np.arange(w)).astype(np.float32)    # inside RAPID
    # pentad labels: the bin's START MONTH, exactly ml/build_family4.py's
    # `months` (a label, not the axis) — so ~6 bins share each %Y-%m key.
    months = []
    for b in range(t_bins):
        y = 2010 + (b // 73)
        m = min(12, 1 + (b % 73) * 5 // 31)
        months.append(f"{y:04d}-{m:02d}")
    months = np.array(months)
    X = np.empty((t_bins, h, w, C), np.float32)
    tt = np.arange(t_bins)[:, None, None]
    for c, nm in enumerate(CHAN_R2):
        mu, sd = chan_mag(nm)
        X[..., c] = (mu + sd * np.sin(2 * np.pi * tt / 73.0)
                     + 0.3 * sd * rng.standard_normal((t_bins, h, w)))
    X[:, :, :3, :] = np.nan                     # land: three western columns
    argo = [c for c, nm in enumerate(CHAN_R2) if nm.startswith(("rg_t", "rg_s"))]
    live = np.zeros(t_bins, bool)
    live[argo_every // 2::argo_every] = True    # one bin in six carries Argo
    X[np.ix_(~live, np.arange(h), np.arange(w), argo)] = np.nan
    return (X.astype(np.float16), months, lats, lons, np.array(CHAN_R2),
            live, argo)


def ref_stats(X, moy, t_hold, x_hold):
    """A float64 reference for `stream_stats`, written the obvious way."""
    T, h, w, c = X.shape
    Xd = np.asarray(X, np.float64)
    fin = np.isfinite(Xd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = np.where(fin, Xd, 0.0).sum(axis=(1, 2)) / fin.sum(axis=(1, 2))
        clim = np.full((12, h, w, c), np.nan)
        for m in range(12):
            sel = (moy == m) & ~t_hold
            if sel.any():
                clim[m] = np.nanmean(Xd[sel], axis=0)
        dyn = [k for k in range(c) if np.nanstd(sp[:, k]) > 1e-6]
        mean_c = np.zeros(c)
        std_c = np.ones(c)
        for k in dyn:
            d = (Xd[..., k] - clim[moy][..., k])[:, :, ~x_hold]
            d = d[np.isfinite(d) & ~t_hold[:, None, None]]
            mean_c[k], std_c[k] = d.mean(), d.std()
    return clim, dyn, mean_c, std_c


def base_recon_eval(tmp):
    """`BASE_SHA`'s ml/recon_eval.py, importable beside today's."""
    r = subprocess.run(["git", "-C", ROOT, "show", f"{BASE_SHA}:ml/recon_eval.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"cannot read ml/recon_eval.py at {BASE_SHA}: {r.stderr.strip()}. "
            f"Check 1 compares against the PRE-FIX code; without it there is "
            f"no reference and this must FAIL rather than pass vacuously.")
    p = os.path.join(tmp, "recon_eval_base.py")
    open(p, "w").write(r.stdout)
    spec = importlib.util.spec_from_file_location("recon_eval_base", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recon_eval_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_ckpt(path, chan, hold_years="2012", hold_lon="-60,-58", seed=3,
              levels="8,8,8,5,5,5", d_z=6, bound="ln", ladder="auto"):
    """A real d_z-6 FSQ codec checkpoint with a FITTED `auto` ladder.

    `codec_from_ckpt` REFUSES an `auto` checkpoint that carries no
    `fsq_ladder_fit` (ml/model.py), so the fit is produced here the way
    `ml/train.py`'s `fsq_auto_fit` produces it: capture `encode_pre` on a real
    batch — the pre-quantization activation, which is what the ladder has to
    serve — and hand it to `q.fit_auto`.
    """
    torch.manual_seed(seed)
    m = PixelMAE(n_chan=len(chan), d_z=d_z, patch=1, d_model=16, n_heads=2,
                 n_layers=2, d_dec=16, dec_layers=2, k_time=1,
                 fsq_levels=levels, fsq_ladder=ladder, fsq_bound=bound)
    m.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(512, len(chan), generator=g)
    obs = torch.rand(512, len(chan), generator=g) > 0.2
    ctx = torch.randn(512, 4, generator=g)
    fit = ""
    if ladder == "auto":
        with torch.no_grad():
            pre = m.encode_pre(x, obs, torch.zeros_like(obs), ctx)
        m.fsq.fit_auto(pre.numpy())
        fit = m.fsq.fit
    args = dict(holdout_years=hold_years, holdout_lon=hold_lon, patch=1,
                d_model=16, n_heads=2, n_layers=2, d_dec=16, dec_layers=2,
                k_time=1, anomaly=True, fsq_levels=levels, fsq_ladder=ladder,
                fsq_ladder_fit=fit, fsq_bound=bound)
    torch.save({"model": m.state_dict(), "d_z": d_z, "args": args}, path)
    return fit


def main():
    tmp = tempfile.mkdtemp(prefix="e049recon_")
    try:
        X, months, lats, lons, chan, live, argo = toy_tensor()
        moy = np.array([int(m[5:7]) - 1 for m in months])
        t_hold = np.array([m[:4] == "2012" for m in months])
        x_hold = (lons >= -60.0) & (lons < -58.0)

        # ---- 1. the float16 accumulator ---------------------------------
        base = base_recon_eval(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # the old path's overflow
            b_clim, b_dyn, b_mu, b_sd = base.stream_stats(X, moy, t_hold,
                                                          x_hold)
        n_clim, n_dyn, n_mu, n_sd = RE.stream_stats(X, moy, t_hold, x_hold,
                                                    chan=list(chan))
        r_clim, r_dyn, r_mu, r_sd = ref_stats(X, moy, t_hold, x_hold)

        # THE OLD ACCUMULATOR'S ARITHMETIC, straight from the source, so the
        # mechanism is measured rather than asserted. It fails TWO ways at
        # family-4 magnitudes, and only the first was the one anybody named:
        #   (a) OVERFLOW — the sum passes 65,504 and becomes inf, so the
        #       spatial-mean series is nan and `nanstd(...) > 1e-6` is False;
        #   (b) SATURATION — the sum stays finite but lands where float16's
        #       spacing is 32 (at 32,768 it is), so every bin's sum rounds to
        #       the SAME representable value and the series is exactly
        #       constant. nanstd is then 0.0, which fails the same test with
        #       no inf and no warning anywhere.
        # And a third mode that does not drop a channel but corrupts its
        # reading: a series whose true spread is 1.06 measured as 0.16.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fin = np.isfinite(X)
            x0 = np.where(fin, X, 0.0)
            s16 = x0.sum(axis=(1, 2))                     # float16 accumulator
            cnt = fin.sum(axis=(1, 2))
            s64 = x0.astype(np.float64).sum(axis=(1, 2))
            den = np.where(cnt > 0, cnt, np.nan)
            sd16 = np.array([np.nanstd(s16[:, c] / den[:, c]) for c in range(C)])
            sd64 = np.array([np.nanstd(s64[:, c] / den[:, c]) for c in range(C)])
        ovf = sorted(c for c in range(C)
                     if not np.isfinite(s16[:, c][cnt[:, c] > 0]).all())
        flat = sorted(c for c in range(C)
                      if c not in ovf and not (sd16[c] > 1e-6) and sd64[c] > 1e-6)
        skewed = sorted(c for c in range(C) if c not in ovf and c not in flat
                        and sd64[c] > 1e-6 and sd16[c] < 0.5 * sd64[c])
        dropped = sorted(set(r_dyn) - set(b_dyn))
        assert dropped, "the pre-fix path must LOSE channels, or check 1 is void"
        assert ovf and flat, (ovf, flat)
        assert set(dropped) == set(ovf) | set(flat), (dropped, ovf, flat)
        assert 39 in ovf, "`sst` is a FAST channel and must be among them"
        assert set(n_dyn) == set(r_dyn), (sorted(set(n_dyn) ^ set(r_dyn)))
        assert np.allclose(n_mu, r_mu, atol=1e-5), np.abs(n_mu - r_mu).max()
        assert np.allclose(n_sd, r_sd, atol=1e-5), np.abs(n_sd - r_sd).max()
        assert np.allclose(np.nan_to_num(n_clim, nan=0.0),
                           np.nan_to_num(r_clim, nan=0.0), atol=1e-3)
        live0 = int(np.where(live)[0][0])
        print(f"1. float16 accumulator: at H*W={H * W:,} (0.4% of family 4's "
              f"135,161) one spatial slice of `sst` sums to "
              f"{s64[live0, 39]:,.0f} against float16's 65,504. {BASE_SHA}'s "
              f"stream_stats keeps {len(b_dyn)}/{C} channels dynamic where the "
              f"truth is {len(r_dyn)}/{C}: {len(ovf)} channels OVERFLOW to inf "
              f"({[str(chan[c]) for c in ovf[:3]]}...) and {len(flat)} more "
              f"SATURATE — {[str(chan[c]) for c in flat]} sum to a constant "
              f"3.277e4 because float16's spacing there is 32, so their "
              f"spatial-mean series has std exactly 0.0 against a true "
              f"{sd64[flat[0]]:.3f}, with no inf and no warning. "
              f"{len(skewed)} more SURVIVE with a corrupted reading "
              f"({[str(chan[c]) for c in skewed]}: std {sd16[skewed[0]]:.3f} "
              f"vs {sd64[skewed[0]]:.3f}). All 28 dropped channels would reach "
              f"the codec UN-STANDARDIZED. Today's stream_stats matches a "
              f"float64 reference: dynamic sets identical ({len(n_dyn)}/{C}), "
              f"max |Δmean| {np.abs(n_mu - r_mu).max():.2e}, max |Δstd| "
              f"{np.abs(n_sd - r_sd).max():.2e}, max |Δclim| "
              f"{np.nanmax(np.abs(n_clim - r_clim)):.2e}")

        # ---- 1b. FLOAT32 IS BIT-IDENTICAL --------------------------------
        # The other half of the fix's contract: `acc_dtype` returns None for
        # float32, so nothing about a family-2/3 tensor's arithmetic moves,
        # and the 183 archived monthly bundles stay reproducible. Asserted
        # BIT-FOR-BIT against BASE_SHA's own code, not to a tolerance.
        rng32 = np.random.default_rng(9)
        X32 = (20.0 + rng32.standard_normal((120, 24, 20, 6))).astype(np.float32)
        X32[:, 0, 0, 2] = np.nan
        X32[..., 5] = 7.0                              # a genuinely static one
        m32 = np.array([f"{2000 + i // 12:04d}-{1 + i % 12:02d}"
                        for i in range(120)])
        my32 = np.array([int(m[5:7]) - 1 for m in m32])
        th32 = np.array([m[:4] == "2007" for m in m32])
        xh32 = np.zeros(20, bool)
        xh32[3:6] = True
        b32 = base.stream_stats(X32, my32, th32, xh32)
        n32 = RE.stream_stats(X32, my32, th32, xh32, chan=list("abcdef"))
        assert np.array_equal(b32[0], n32[0], equal_nan=True), "clim moved"
        assert list(b32[1]) == list(n32[1]), (b32[1], n32[1])
        assert np.array_equal(b32[2], n32[2]) and np.array_equal(b32[3], n32[3])
        assert b32[0].dtype == n32[0].dtype == np.float32
        print(f"1b. float32 is BIT-IDENTICAL: on a [120,24,20,6] float32 toy "
              f"(mean 20, one static channel, one NaN cell, a lon block held "
              f"out) {BASE_SHA}'s stream_stats and today's return the same "
              f"clim bit for bit (NaNs in the same places), the same dynamic "
              f"list ({list(n32[1])}) and the same mean/std arrays. "
              f"`acc_dtype` returns None off float32, so every family-2/3 "
              f"number this script has produced is reproducible")

        # ---- 2. the uint8 -> uint16 climatology counter ------------------
        # 300 train timesteps in ONE month-of-year: uint8 wraps at 256.
        Tw = 300
        Xw = np.ones((Tw, 2, 2, 1), np.float16)
        mw = np.zeros(Tw, int)
        hw = np.zeros(Tw, bool)
        xw = np.zeros(2, bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bw = base.stream_stats(Xw, mw, hw, xw)[0][0, 0, 0, 0]
        nw = RE.stream_stats(Xw, mw, hw, xw, chan=["c"])[0][0, 0, 0, 0]
        assert abs(nw - 1.0) < 1e-6, nw
        assert abs(bw - 1.0) > 1.0, bw
        src = open(os.path.join(ML, "recon_eval.py")).read()
        assert "np.uint16)         # ~244 of 65,535 at pentad" in src
        print(f"2. climatology counter: {Tw} train timesteps in one "
              f"month-of-year, every cell observed. uint8 counts "
              f"{Tw % 256} and the climatology reads {bw:.3f} for data that is "
              f"identically 1.0 ({bw:.1f}x, no warning); uint16 reads "
              f"{nw:.6f}. Pentad cadence uses ~244 of the old 255")

        # ---- 3. the Argo-bin mask ---------------------------------------
        obs_full = np.isfinite(X)
        a_idx, fast_idx = RE.argo_channels(list(chan))
        assert a_idx == argo and len(a_idx) == 32, len(a_idx)
        assert [str(chan[c]) for c in fast_idx] == [
            "cur_speed", "log_mld", "ssh", "tau_x", "tau_y", "tau_x_std",
            "tau_y_std", "sst"], [str(chan[c]) for c in fast_idx]
        bins = RE.argo_bin_mask(obs_full[:, 30], a_idx)      # one row, [T,W]
        ocean = np.isfinite(X[:, 30, :, 0]).any(0)
        assert (bins[:, ocean].any(axis=1) == live).all()
        frac = float(bins[:, ocean].mean())
        assert abs(frac - float(live.mean())) < 1e-9, (frac, live.mean())
        assert RE.argo_bin_mask(obs_full[:, 30][:, ~ocean], a_idx).sum() == 0
        try:
            RE.argo_channels(["cur_speed", "sst"])
            raise AssertionError("argo_channels must refuse a set with no rg_*")
        except SystemExit as e:
            assert "Argo-bin split cannot be built" in str(e), e
        print(f"3. Argo-bin mask: {len(a_idx)} rg_t*/rg_s* channels found BY "
              f"NAME (fast = the other {len(fast_idx)}: "
              f"{[str(chan[c]) for c in fast_idx]}); the synthetic stamp is "
              f"one bin in {ARGO_EVERY} and the mask recovers exactly those "
              f"bins over ocean pixels (fraction {frac:.6f} = "
              f"{int(live.sum())}/{len(live)} bins), "
              f"zero over land. A channel set with no rg_* is REFUSED rather "
              f"than scored as 100% Argo-free")

        # ---- 4. the split scorer's keys and arithmetic -------------------
        rng = np.random.default_rng(1)
        Tt, Pp = 60, 20
        truth = rng.standard_normal((Tt, Pp, C))
        pred = truth * 0.7 + 0.3 * rng.standard_normal((Tt, Pp, C))
        obs = rng.random((Tt, Pp, C)) > 0.1
        obs[..., a_idx] = False
        obs[::3][..., a_idx] = True                # one bin in three carries
        st = {"train": (np.arange(40), np.arange(Pp)),
              "heldout_months": (np.arange(40, Tt), np.arange(Pp))}
        blk = RE.argo_split_block(truth, pred, obs, list(chan), st)
        assert set(blk) == {"doc", "argo_channels", "argo_channel_names",
                            "fast_channels", "fast_channel_names", "census",
                            "argo_bins", "argo_free_bins"}
        for name in st:
            cen = blk["census"][name]
            assert cen["argo"] + cen["argo_free"] == cen["bins"]
            for side in ("argo_bins", "argo_free_bins"):
                assert name in blk[side]
                for c, v in blk[side][name].items():
                    assert set(v) == {"r", "rmse", "fvu", "fvu_local", "n"}
                    assert abs(v["fvu"] - v["rmse"] ** 2) < 2e-4, (c, v)
        # fvu is mse in standardized units; fvu_local divides by the local var
        c0 = blk["fast_channels"][0]
        m = obs[np.ix_(np.arange(40), np.arange(Pp))][..., c0] & \
            ~RE.argo_bin_mask(obs, a_idx)[np.ix_(np.arange(40), np.arange(Pp))]
        aa = truth[np.ix_(np.arange(40), np.arange(Pp))][..., c0][m]
        bb = pred[np.ix_(np.arange(40), np.arange(Pp))][..., c0][m]
        mse = float(((aa - bb) ** 2).mean())
        got = blk["argo_free_bins"]["train"][c0]
        assert abs(got["fvu"] - mse) < 1e-5, (got, mse)
        assert abs(got["fvu_local"] - mse / aa.var()) < 1e-4, (got, mse)
        assert got["n"] == int(m.sum())
        # and score() itself is UNTOUCHED — the archived key set, exactly
        s_old = RE.score(truth, pred, obs, np.arange(40), np.arange(Pp), "x")
        assert set(s_old[c0]) == {"r", "rmse", "n"}, s_old[c0]
        print(f"4. split scorer: `argo_split_block` writes census + "
              f"argo_bins + argo_free_bins per split, each channel carrying "
              f"{{r, rmse, fvu, fvu_local, n}}; fvu == rmse^2 (E-049 §4a's "
              f"standardized-unit definition) and fvu_local == mse/var(truth) "
              f"over the selection ({got['fvu']:.5f} / {got['fvu_local']:.5f} "
              f"on n={got['n']:,}). Census closes: argo + argo_free == bins. "
              f"`score()`'s own output is unchanged at {{r, rmse, n}} — the "
              f"183 archived monthly bundles read that one")

        # ---- 5. the .npz input path -------------------------------------
        npy = os.path.join(tmp, "bare_X.npy")
        np.save(npy, X)
        Xa, ma = RE.open_x(npy)
        assert isinstance(Xa, np.memmap) and ma is None and Xa.shape == X.shape

        stored = os.path.join(tmp, "stored.npz")
        np.savez(stored, X=X, months=months, lats=lats, lons=lons, chan=chan)
        off = RE.npz_member_offset(stored)
        assert off is not None
        Xb, mb = RE.open_x(stored)
        assert isinstance(Xb, np.memmap) and np.array_equal(
            np.asarray(Xb, np.float32), np.asarray(X, np.float32),
            equal_nan=True)
        assert list(mb["chan"]) == list(chan)

        comp = os.path.join(tmp, "family4_toy_r2.npz")
        np.savez_compressed(comp, X=X, months=months, lats=lats, lons=lons,
                            chan=chan)
        assert RE.npz_member_offset(comp) is None
        with zipfile.ZipFile(comp) as z:
            assert z.getinfo("X.npy").compress_type == zipfile.ZIP_DEFLATED
        try:
            RE.open_x(comp)
            raise AssertionError("a DEFLATE member must be refused, not loaded")
        except SystemExit as e:
            assert "--extract-x" in str(e) and "memory-map" in str(e), e
        side = RE.extract_x(comp, RE.sidecar_path(comp))
        assert side.endswith("_X.npy") and os.path.exists(side)
        Xc, mc = RE.open_x(comp)                 # now finds the sidecar
        assert np.array_equal(np.asarray(Xc, np.float32),
                              np.asarray(X, np.float32), equal_nan=True)
        try:
            RE.extract_x(comp, os.path.join(tmp, "nope_X.npy"),
                         headroom=1 << 62)
            raise AssertionError("the disk guard must refuse")
        except SystemExit as e:
            assert "Refusing" in str(e), e
        print(f"5. npz input: bare .npy memmaps (unchanged); an UNCOMPRESSED "
              f"member is memmapped IN PLACE at byte {off:,} (no copy); the "
              f"DEFLATE member np.savez_compressed writes — which is what "
              f"ml/build_family4.py:921 uses, so it is what family-4-r2 ships "
              f"— is REFUSED with the --extract-x command; --extract-x writes "
              f"tensor_io's own sidecar name ({os.path.basename(side)}) and "
              f"open_x then finds it with no flag. The disk guard refuses "
              f"rather than truncating")

        # ---- 6. agreement with ml/train.py's own anomaly transform -------
        # train.py:496-515 is `moy = int(m[5:7]) - 1` then
        # `anomaly_transform(X, moy, t_hold, x_hold)`. Run that on a COPY and
        # compare against stream_stats + build_slab on the same bytes.
        Xw2 = X.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xa2, dyn_tp = anomaly_transform(Xw2, moy, t_hold, x_hold, chunk=16,
                                            verbose=False)
        assert list(dyn_tp) == list(n_dyn), (dyn_tp, n_dyn)
        rows = [25, 26, 27]
        slab, sobs = RE.build_slab(X, rows, moy, n_clim, n_dyn, n_mu, n_sd)
        ref = np.asarray(Xa2[:, rows], np.float64)
        got = np.asarray(slab, np.float64)
        fin = np.isfinite(ref)
        assert (np.isfinite(got) == fin).all(), "finite masks differ"
        dmax = float(np.abs(ref[fin] - got[fin]).max())
        rmsd = float(np.sqrt(((ref[fin] - got[fin]) ** 2).mean()))
        assert dmax < 5e-3, dmax
        print(f"6. anomaly transform: `stream_stats` + `build_slab` agree with "
              f"ml/trainprobe.anomaly_transform — the function ml/train.py:515 "
              f"calls for a per-bin pentad codec — on the identical float16 "
              f"bytes: same dynamic set ({len(dyn_tp)}/{C}), identical finite "
              f"masks, max |Δ| {dmax:.2e} and rms Δ {rmsd:.2e} in standardized "
              f"units. The residual is float16 STORAGE rounding: "
              f"anomaly_transform writes (X - clim) back at the tensor dtype "
              f"and reads it back for the z-score (deliberate — see its "
              f"docstring, 'Do not optimise that into a fused ...'), while the "
              f"audit's replica keeps float32 throughout. The transform is NOT "
              f"changed by this work")

        # ---- 7. the refusal ---------------------------------------------
        # Forcing `acc_dtype` back to numpy's default IS the pre-fix
        # arithmetic — check 1 showed the two agree channel for channel — so
        # this asks the exact question that matters: on the tensor the fix
        # exists for, does the audit now STOP?
        #
        # THE FIRST VERSION OF THIS GUARD DID NOT, and the note is worth
        # keeping. It watched `mean_c`/`std_c` only, and an overflowed channel
        # leaves `dyn` and is handed the pass-through values 0.0 / 1.0, which
        # are finite. The guard has to watch the per-BIN spatial mean, which
        # is where the inf actually lands.
        saved = RE.acc_dtype
        try:
            RE.acc_dtype = lambda dt: None       # the pre-fix accumulator
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RE.stream_stats(X, moy, t_hold, x_hold, chan=list(chan))
            raise AssertionError("stream_stats must REFUSE, not score")
        except SystemExit as e:
            msg = str(e)
            assert "NON-FINITE" in msg and "UN-STANDARDIZED" in msg, msg
            named = [str(chan[c]) for c in range(C) if f"({chan[c]})" in msg]
            assert set(named) == set(str(chan[c]) for c in ovf), (named, ovf)
            assert "sst" in named and "rg_t10" in named, named
        finally:
            RE.acc_dtype = saved
        obs_n = np.isfinite(X).sum(axis=(0, 1, 2))
        zero_bad = np.zeros(C, np.int64)
        try:
            bad_sd = n_sd.copy()
            bad_sd[3] = np.nan
            RE.check_stats(list(chan), n_dyn, n_mu, bad_sd, obs_n, zero_bad)
            raise AssertionError("check_stats must refuse a nan std")
        except SystemExit as e:
            assert "rg_t10" in str(e), e
        # a channel with NO observed value is not an error
        RE.check_stats(list(chan), n_dyn, n_mu, n_sd, np.zeros(C, np.int64),
                       zero_bad)
        # and today's stats pass it cleanly
        RE.check_stats(list(chan), n_dyn, n_mu, n_sd, obs_n, zero_bad)
        print(f"7. the refusal: with the accumulator forced back to float16, "
              f"stream_stats now STOPS and names exactly the {len(ovf)} "
              f"overflowed channels (sst, rg_t10, ...) instead of quietly "
              f"dropping them from `dynamic`. The guard watches the per-BIN "
              f"spatial mean, not mean_c/std_c: an overflowed channel leaves "
              f"`dyn` and is then handed the pass-through 0.0/1.0, which are "
              f"finite — an outputs-only guard would have passed this tensor. "
              f"`check_stats` is exported so recon_decoder's CACHED-stats path "
              f"runs the same check, and it passes a channel with no observed "
              f"values at all")

        # ---- 8. d_z 6 / FSQ / --fsq-bound ln, end to end -----------------
        npz = os.path.join(tmp, "audit.npz")
        np.savez(npz, X=X, months=months, lats=lats, lons=lons, chan=chan)
        ckpt = os.path.join(tmp, "toy_dz6__pixelmae.pt")
        fit = make_ckpt(ckpt, chan)
        assert fit, "the auto ladder must have been fitted"
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cd = codec_from_ckpt(ck, C)
        assert cd.fsq is not None and cd.fsq_bound == "ln" and cd.d_z == 6
        spec = RE.bottleneck_spec(ck)
        assert spec["d_z"] == 6 and spec["fsq_levels"] == "8,8,8,5,5,5"
        assert abs(spec["codebook_log2"]
                   - np.log2(8 * 8 * 8 * 5 * 5 * 5)) < 1e-3, spec
        assert spec["fsq_bound"] == "ln" and spec["fsq_ladder"] == "auto"
        assert spec["fsq_ladder_fit"] == fit
        out = os.path.join(tmp, "audit.json")
        r = subprocess.run(
            [sys.executable, os.path.join(ML, "recon_eval.py"), "--x", npz,
             "--ckpt", ckpt, "--out", out, "--batch", "256"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout[-4000:] + "\n" + r.stderr[-4000:]
        res = json.load(open(out))
        assert res["bottleneck"] == spec and res["x_dtype"] == "float16"
        assert set(res["splits"]) == {"train", "heldout_months", "heldout_lons"}
        blk = res["argo_split"]
        assert len(blk["argo_channels"]) == 32 and len(blk["fast_channels"]) == 8
        cen = blk["census"]["heldout_months"]
        assert cen["argo"] + cen["argo_free"] == cen["bins"]
        assert 0.0 < cen["argo_fraction"] < 0.5, cen
        # the falsifier's own cell must be populated for every fast channel
        free = blk["argo_free_bins"]["heldout_months"]
        miss = [str(chan[c]) for c in blk["fast_channels"] if str(c) not in free
                and c not in free]
        assert not miss, miss
        assert "bottleneck audited: d_z 6 · FSQ [8,8,8,5,5,5]" in r.stdout
        assert "bound ln" in r.stdout and "E-049 falsifier reading" in r.stdout
        assert "verify chan" in r.stdout, "the streaming verify must have run"
        vmax = max(float(l.split("=")[-1].split()[0]) for l in
                   r.stdout.splitlines() if "verify chan" in l)
        fvus = [free[k]["fvu_local"] for k in free
                if int(k) in blk["fast_channels"]]
        print(f"8. end to end: ml/recon_eval.py's own main(), d_z 6 · FSQ "
              f"[8,8,8,5,5,5] auto-fitted · --fsq-bound ln, on the "
              f"[T={T_BINS} H={H} W={W} C={C}] float16 toy. Encoded through "
              f"embed_everything (so the quantizer is in the path, as in "
              f"production), decoded at offset 0, {res['P']} section pixels. "
              f"The streaming verify passed at max |Δ| {vmax:.2e}; the JSON "
              f"carries `bottleneck` {spec['fsq_levels']} / bound "
              f"{spec['fsq_bound']}, the unchanged `splits`, and "
              f"`argo_split` with {cen['argo_free']:,} Argo-free of "
              f"{cen['bins']:,} held-out bin-pixels "
              f"({cen['argo_fraction']:.3f} Argo). Fast-channel fvu_local on "
              f"Argo-free bins spans {min(fvus):.3f}-{max(fvus):.3f} on this "
              f"UNTRAINED toy codec — the plumbing, not a result")

        # ---- 9. recon_decoder.py end to end, at pentad cadence -----------
        # THE DECODER-CEILING RUNG (plan §4c) is where the falsifier is read,
        # so it gets its own end-to-end run rather than inheriting by
        # inspection. A smaller grid, because this one has to embed EVERY
        # ocean pixel to build a Z cache (the section-only audit above embeds
        # 53); the time axis stays at 4 years of pentads, because that is what
        # is being checked — the training pool counted in BINS.
        d2 = os.path.join(tmp, "dec")
        os.makedirs(d2)
        X2, mo2, la2, lo2, ch2, live2, argo2 = toy_tensor(
            seed=5, h=12, w=16, lat0=26.0)        # 26.5 at row 2, interior
        moy2 = np.array([int(m[5:7]) - 1 for m in mo2])
        th2 = np.array([m[:4] == "2012" for m in mo2])
        xh2 = (lo2 >= -68.0) & (lo2 < -67.0)
        npz2 = os.path.join(d2, "toy_dec.npz")
        np.savez(npz2, X=X2, months=mo2, lats=la2, lons=lo2, chan=ch2)
        ck2 = os.path.join(d2, "toy2__pixelmae.pt")
        make_ckpt(ck2, ch2, hold_lon="-68,-67", seed=11)
        ckd = torch.load(ck2, map_location="cpu", weights_only=False)
        cod = codec_from_ckpt(ckd, C)
        cod.load_state_dict(ckd["model"])
        cod.eval()
        cl2, dy2, mu2, sd2 = RE.stream_stats(X2, moy2, th2, xh2, chan=list(ch2))
        rows2 = list(range(X2.shape[1]))
        full, obs2 = RE.build_slab(X2, rows2, moy2, cl2, dy2, mu2, sd2)
        ocean2 = obs2[..., 0].any(axis=0)
        ys2, xs2 = np.where(ocean2)
        ctx2 = np.stack([np.sin(2 * np.pi * moy2 / 12),
                         np.cos(2 * np.pi * moy2 / 12)], 1)
        from temporal import embed_everything                    # noqa: E402
        Zf, _ = embed_everything(cod, torch.from_numpy(np.nan_to_num(full, 0.0)),
                                 torch.from_numpy(obs2), ctx2, la2, lo2,
                                 ys2, xs2, ckd["d_z"], cache_path=None,
                                 batch=1024)
        zpath = os.path.join(d2, "Z.npy")
        np.save(zpath, np.asarray(Zf, np.float16))
        out2 = os.path.join(d2, "dec.json")
        r2 = subprocess.run(
            [sys.executable, os.path.join(ML, "recon_decoder.py"),
             "--x", npz2, "--z", zpath, "--ckpt", ck2, "--hidden", "64",
             "--layers", "2", "--steps", "60", "--batch", "512",
             "--pairs", "20000", "--cache-dir", d2, "--out", out2],
            capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode == 0, r2.stdout[-4000:] + "\n" + r2.stderr[-4000:]
        rd = json.load(open(out2))
        assert rd["bottleneck"]["d_z"] == 6
        assert rd["bottleneck"]["fsq_levels"] == "8,8,8,5,5,5"
        assert set(rd["splits"]) == {"train", "heldout_months", "heldout_lons"}
        b2 = rd["argo_split"]
        assert len(b2["fast_channels"]) == 8 and len(b2["argo_channels"]) == 32
        c2 = b2["census"]["train"]
        assert c2["argo"] + c2["argo_free"] == c2["bins"] and c2["argo"] > 0
        assert "bottleneck audited: d_z 6" in r2.stdout
        assert "Z cache verified" in r2.stdout
        assert "deep-channel mean r" in r2.stdout and "nan" not in r2.stdout
        bins_line = [l for l in r2.stdout.splitlines()
                     if l.startswith("pairs: ") and "train bins" in l]
        assert len(bins_line) == 1, r2.stdout
        # and the pool refuses rather than gathering nothing
        r3 = subprocess.run(
            [sys.executable, os.path.join(ML, "recon_decoder.py"),
             "--x", npz2, "--z", zpath, "--ckpt", ck2, "--hidden", "64",
             "--layers", "2", "--steps", "2", "--pairs", "10",
             "--cache-dir", d2, "--out", os.path.join(d2, "x.json")],
            capture_output=True, text=True, cwd=ROOT)
        assert r3.returncode != 0 and "floors to" in r3.stderr, r3.stderr[-2000:]
        print(f"9. recon_decoder end to end at pentad cadence: same d_z 6 / "
              f"FSQ / bound codec, a real Z cache over all {len(ys2):,} ocean "
              f"pixels x {len(mo2)} bins, {bins_line[0]}. It inherits "
              f"stream_stats/check_stats/open_x/argo_split_block from "
              f"recon_eval (ONE copy), states its bottleneck, verifies the Z "
              f"cache, writes `argo_split` ({c2['argo']:,} Argo of "
              f"{c2['bins']:,} train bin-pixels), and prints no `nan` in its "
              f"summary. A --pairs too small for the bin count REFUSES "
              f"instead of training on an empty gather")

        print("\nE-049 recon audit: all 9 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
