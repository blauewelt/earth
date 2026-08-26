#!/usr/bin/env python3
"""E-053.0 · Measure the advective cone from the published pentad Z. $0 GPU.

The space-time stencil (ml/plans/E053_spacetime_stencil.md) needs one number
it must not hand-pick (CLAUDE.md bans hand-picked thresholds): c, the speed
that converts a spatial separation r into the lag -r/c at which a neighbour
is most informative. This script MEASURES it, from the published clean pentad
embedding `Z_8b639abe36_37e146384b` (the run-415 continuous z both E-051 and
every pentad stage-2 arm read), plus the second curve the plan's log-time
ramp rests on: the centre pixel's own autocorrelation in lag, out to two
years, where the seasonal-analog peaks at ~73 and ~146 bins either exist or
do not.

Method, stated so the numbers can be audited:
- Pixel order: Z row i is the i-th TRUE cell of
  `np.isfinite(X[..., 0]).any(axis=0)` in row-major (y, x) order — the exact
  construction at ml/temporal.py (`ocean = OBS[..., 0].any(axis=0)`;
  `ys, xs = np.where(ocean)`) and ml/train.py:488. The mask is recomputed
  here by STREAMING the tensor npz member (never 34 GB in RAM) and its count
  is asserted against the archived inventory (86,698) before anything else
  runs — a wrong ordering would produce a beautiful, wrong cone.
- Sampling: N_CENTRES ocean pixels drawn with a fixed seed, partners at
  fixed cell separations along E/W/N/S where the partner is ocean.
  Separations in km are haversine, so a zonal step shrinks with latitude.
- Correlation: per pair and lag L in [-MAX_LAG, MAX_LAG] bins, the Pearson r
  of the two 32-dim series per dimension, averaged over dimensions (signed:
  the dims are shared features across pixels, so nearby same-dim correlation
  is meaningfully positive). Series are deseasonalized per pixel, dim and
  bin-of-year (b = t mod 73; the 0.05-bin/year epoch drift this ignores is
  noted in the output). The RIDGE is, per pair, the lag maximising |rbar|;
  binned by separation its median |lag| vs r is the cone, and its slope is
  1/c.
- The self curve: each centre's own autocorrelation out to SELF_MAX_LAG bins,
  raw and deseasonalized — deseasonalized is the one the log-ramp reads;
  raw minus deseasonalized at lag 73/146 is the calendar's share, which
  CONFLATES ocean seasonality with the Argo one-live-bin-per-month observing
  pattern (plan §6's caveat, restated in the output JSON).

Outputs: ml/runs/e053_cone.json (all numbers + config) and
ml/figures/fig_e053_cone.png (ridge + self-curve panels). Inputs are pulled
from the public releases into --work (default /tmp/e053), ~22 GB transient,
deleted afterwards unless --keep.

    python3 ml/spacetime_corr.py --work /tmp/e053
"""
import argparse
import json
import math
import os
import struct
import sys
import urllib.request
import zipfile
import zlib

import numpy as np

REL = "https://github.com/blauewelt/earth/releases/download"
TENSOR_CHUNKS = [f"family4_na025_pentad_r2_37e146384b.npz.a{c}" for c in "abc"]
Z_NAME = "Z_8b639abe36_37e146384b"
Z_CHUNKS = [f"{Z_NAME}.npy.a{c}" for c in "abcdefghijkl"]
T, H, W, C = 3142, 281, 481, 40
P_EXPECTED = 86698
D_Z = 32
BINS_PER_YEAR = 73          # 365.2425/5 = 73.05 — the drift is noted, not fixed
LAT0, LON0, STEP = 0.0, -100.0, 0.25

N_CENTRES = 400
SEPS_CELLS = (2, 5, 12, 30, 72, 160)      # 0.25-deg cells; km computed per pair
DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))  # E, W, N, S as (dy, dx)
MAX_LAG = 36                # +-180 d for the pair ridge
SELF_MAX_LAG = 160          # 800 d for the centre's own curve
SEED = 20260826


def fetch(url, dst):
    if os.path.exists(dst):
        return
    print(f"  pull {os.path.basename(dst)}", flush=True)
    tmp = dst + ".part"
    # curl, not urllib: a 1.5 GB transfer that dies at 98% must RESUME, not
    # restart — -C - continues the .part, --retry covers transient resets
    # (urllib's urlretrieve raised ContentTooShortError on the first run).
    import subprocess
    r = subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-delay", "2",
                        "-C", "-", "-o", tmp, url])
    if r.returncode != 0:
        sys.exit(f"fetch failed rc={r.returncode}: {url}")
    os.replace(tmp, dst)


def ocean_mask_streamed(npz_path):
    """isfinite(X[..., 0]).any(axis=0), without ever holding X.

    The npz member is DEFLATE-compressed; zipfile.open streams it. One
    t-slice is H*W*C*2 bytes of float16; only channel 0's finiteness is
    kept. NaN in float16 is exponent all-ones + nonzero mantissa — cheap to
    test on the raw uint16 view, which spares a float conversion per slice."""
    mask = np.zeros((H, W), bool)
    slice_bytes = H * W * C * 2
    with zipfile.ZipFile(npz_path) as zf:
        name = [n for n in zf.namelist() if n.startswith("X")][0]
        with zf.open(name) as f:
            hdr = f.read(10)
            assert hdr[:6] == b"\x93NUMPY", "not an npy member"
            hlen = struct.unpack("<H", hdr[8:10])[0]
            f.read(hlen)                      # header dict — shape known above
            for t in range(T):
                buf = f.read(slice_bytes)
                if len(buf) < slice_bytes:
                    sys.exit(f"tensor stream ended early at t={t}")
                u = np.frombuffer(buf, np.uint16).reshape(H, W, C)[..., 0]
                finite = (u & 0x7C00) != 0x7C00   # exp != all-ones: not nan/inf
                mask |= finite
                if t % 500 == 0:
                    print(f"  mask pass t={t}/{T}", flush=True)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/e053")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    # ---- 1 · the pixel ordering, recomputed and asserted ------------------
    npz = os.path.join(a.work, "tensor.npz")
    if not os.path.exists(npz):
        for c in TENSOR_CHUNKS:
            fetch(f"{REL}/data-cache-v1/{c}", os.path.join(a.work, c))
        with open(npz + ".part", "wb") as out:
            for c in TENSOR_CHUNKS:
                p = os.path.join(a.work, c)
                with open(p, "rb") as f:
                    while True:
                        b = f.read(1 << 24)
                        if not b:
                            break
                        out.write(b)
                os.remove(p)
        os.replace(npz + ".part", npz)
    mask = ocean_mask_streamed(npz)
    ys, xs = np.where(mask)
    assert len(ys) == P_EXPECTED, \
        (f"ocean-pixel count {len(ys)} != archived {P_EXPECTED} — the Z row "
         f"ordering would be wrong and every number after it meaningless.")
    lin = np.full((H, W), -1, np.int64)
    lin[ys, xs] = np.arange(len(ys))
    lats, lons = LAT0 + ys * STEP, LON0 + xs * STEP
    if not a.keep:
        os.remove(npz)

    # ---- 2 · the published Z, reassembled and verified --------------------
    zpath = os.path.join(a.work, Z_NAME + ".npy")
    if not os.path.exists(zpath):
        for c in Z_CHUNKS:
            fetch(f"{REL}/embed-cache-v1/{c}", os.path.join(a.work, c))
        with open(zpath + ".part", "wb") as out:
            for c in Z_CHUNKS:
                p = os.path.join(a.work, c)
                with open(p, "rb") as f:
                    while True:
                        b = f.read(1 << 24)
                        if not b:
                            break
                        out.write(b)
                os.remove(p)
        os.replace(zpath + ".part", zpath)
    Z = np.load(zpath, mmap_mode="r")
    assert Z.shape == (T, P_EXPECTED, D_Z) and Z.dtype == np.float16, \
        f"Z is {Z.shape} {Z.dtype}, expected {(T, P_EXPECTED, D_Z)} float16"

    # ---- 3 · sample centres and partners ----------------------------------
    rng = np.random.default_rng(SEED)
    centres = np.sort(rng.choice(P_EXPECTED, N_CENTRES, replace=False))
    pairs = []                                    # (ci, pi, km)
    for ci in centres:
        y0, x0 = ys[ci], xs[ci]
        for s in SEPS_CELLS:
            for dy, dx in DIRS:
                y1, x1 = y0 + dy * s, x0 + dx * s
                if not (0 <= y1 < H and 0 <= x1 < W):
                    continue
                pi = lin[y1, x1]
                if pi < 0:
                    continue
                la0, lo0 = math.radians(lats[ci]), math.radians(lons[ci])
                la1 = math.radians(LAT0 + y1 * STEP)
                lo1 = math.radians(LON0 + x1 * STEP)
                dh = (math.sin((la1 - la0) / 2) ** 2
                      + math.cos(la0) * math.cos(la1)
                      * math.sin((lo1 - lo0) / 2) ** 2)
                km = 2 * 6371.0 * math.asin(math.sqrt(dh))
                pairs.append((int(ci), int(pi), km))
    need = np.unique([p for c_, p, _ in pairs] + list(centres))
    col = {int(p): i for i, p in enumerate(need)}
    print(f"{len(pairs)} pairs over {len(need)} pixels", flush=True)

    # ---- 4 · extract, deseasonalize, standardize --------------------------
    # This sandbox has ~7 GB of RAM against ~10k sampled pixels, so the big
    # array is kept FLOAT16 (1.6-2 GB) and every arithmetic pass streams in
    # float32 blocks; per-pair conversions in step 5 are 0.8 MB each.
    Sd = np.empty((T, len(need), D_Z), np.float16)
    for t0 in range(0, T, 128):                   # blocked memmap gather
        t1 = min(T, t0 + 128)
        Sd[t0:t1] = Z[t0:t1][:, need, :]
        if t0 % 1024 == 0:
            print(f"  extract t={t0}/{T}", flush=True)
    boy = np.arange(T) % BINS_PER_YEAR
    cen_cols = np.array([col[int(c_)] for c_ in centres])
    # the RAW standardized series are kept for the CENTRES only (80 MB):
    # the raw-vs-deseasonalized contrast is a self-curve question.
    Sr = Sd[:, cen_cols, :].astype(np.float32)

    def standardize_inplace_f16(A16):
        """Deseasonalize by bin-of-year, then standardize, per (pixel, dim),
        streaming over pixel blocks in float32 and writing float16 back."""
        for p0 in range(0, A16.shape[1], 512):
            p1 = min(A16.shape[1], p0 + 512)
            blk = A16[:, p0:p1, :].astype(np.float32)
            for b in range(BINS_PER_YEAR):
                m = boy == b
                blk[m] -= blk[m].mean(axis=0, keepdims=True)
            blk -= blk.mean(axis=0, keepdims=True)
            sd = blk.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            blk /= sd
            A16[:, p0:p1, :] = blk.astype(np.float16)
        return A16

    Sn = standardize_inplace_f16(Sd)
    Sr -= Sr.mean(axis=0, keepdims=True)
    sd_r = Sr.std(axis=0, keepdims=True)
    sd_r[sd_r == 0] = 1.0
    Sr /= sd_r

    # ---- 5 · pair ridge ----------------------------------------------------
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    ridge = []                                    # (km, best_abs_lag, rbar)
    for ci, pi, km in pairs:
        A = Sn[:, col[ci], :].astype(np.float32)
        B = Sn[:, col[pi], :].astype(np.float32)
        rs = np.empty(len(lags), np.float32)
        for li, L in enumerate(lags):
            if L >= 0:
                x, y = A[: T - L], B[L:]
            else:
                x, y = A[-L:], B[: T + L]
            rs[li] = float((x * y).mean())        # standardized: mean prod = r
        bi = int(np.abs(rs).argmax())
        ridge.append((km, abs(int(lags[bi])), float(rs[bi]),
                      float(rs[len(lags) // 2])))
    ridge = np.array(ridge)

    # bin by separation; the cone is median |lag*| per bin, days vs km
    edges = np.array([0, 80, 200, 450, 1000, 2500, 6000], float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ridge[:, 0] >= lo) & (ridge[:, 0] < hi)
        if m.sum() < 8:
            continue
        rows.append(dict(km_lo=lo, km_hi=hi, n=int(m.sum()),
                         km_med=float(np.median(ridge[m, 0])),
                         lag_med_bins=float(np.median(ridge[m, 1])),
                         lag_p75_bins=float(np.percentile(ridge[m, 1], 75)),
                         r_at_ridge_med=float(np.median(np.abs(ridge[m, 2]))),
                         r_at_lag0_med=float(np.median(np.abs(ridge[m, 3])))))
    # c from a zero-intercept LS fit of median lag_days on km, over bins that
    # actually moved off lag 0 (a flat ridge at 0 = no measurable cone).
    fit = [(r["km_med"], r["lag_med_bins"] * 5.0) for r in rows
           if r["lag_med_bins"] > 0]
    if len(fit) >= 2:
        km_v = np.array([f[0] for f in fit])
        dy_v = np.array([f[1] for f in fit])
        inv_c = float((km_v * dy_v).sum() / (km_v * km_v).sum())  # days/km
        c_km_day = (1.0 / inv_c) if inv_c > 0 else None
    else:
        c_km_day = None

    # ---- 6 · the self curve ------------------------------------------------
    ac_lags = np.arange(0, SELF_MAX_LAG + 1)

    def self_curve(M):
        out = np.empty(len(ac_lags), np.float32)
        for li, L in enumerate(ac_lags):
            x = M[: T - L][:, :, :] if L else M
            y = M[L:]
            out[li] = float((x * y).mean())
        return out

    Cs = self_curve(Sn[:, cen_cols, :].astype(np.float32))
    Cr = self_curve(Sr)

    res = dict(
        script="ml/spacetime_corr.py", seed=SEED, z=Z_NAME,
        n_centres=N_CENTRES, n_pairs=len(pairs), max_lag_bins=MAX_LAG,
        note_boy_drift="bin-of-year uses t mod 73; the axis's 0.05 bin/year "
                       "epoch drift (~2 bins over 43 years) is ignored",
        note_analog_confound="raw-minus-deseasonalized at lag 73/146 "
                             "conflates ocean seasonality with the Argo "
                             "one-live-bin-per-month observing pattern "
                             "(plan §6)",
        ridge_bins=rows, c_km_per_day=c_km_day,
        self_deseas={int(L): float(v) for L, v in zip(ac_lags, Cs)},
        self_raw={int(L): float(v) for L, v in zip(ac_lags, Cr)},
        self_deseas_key_lags={str(L): float(Cs[L])
                              for L in (1, 2, 3, 6, 12, 24, 36, 73, 146)},
    )
    out_json = os.path.join(here, "runs", "e053_cone.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out_json + ".tmp", out_json)
    print(f"wrote {out_json}", flush=True)

    # ---- 7 · figure --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    km_m = [r["km_med"] for r in rows]
    ax1.plot(km_m, [r["lag_med_bins"] * 5 for r in rows], "o-",
             label="median |lag*| (days)")
    ax1.plot(km_m, [r["lag_p75_bins"] * 5 for r in rows], "s--", alpha=.6,
             label="p75")
    if c_km_day:
        xx = np.linspace(0, max(km_m), 50)
        ax1.plot(xx, xx / c_km_day, ":", label=f"fit c = {c_km_day:.0f} km/d")
    ax1.set_xlabel("separation (km)"); ax1.set_ylabel("ridge lag (days)")
    ax1.set_title("E-053.0 · the measured cone"); ax1.legend()
    ax2.plot(ac_lags * 5, Cs, label="deseasonalized")
    ax2.plot(ac_lags * 5, Cr, alpha=.6, label="raw")
    for L in (73, 146):
        ax2.axvline(L * 5, color="k", lw=.5, ls=":")
    ax2.set_xlabel("lag (days)"); ax2.set_ylabel("autocorr (dim-mean)")
    ax2.set_title("centre self-correlation; dotted = 1y, 2y"); ax2.legend()
    fig.tight_layout()
    out_png = os.path.join(here, "figures", "fig_e053_cone.png")
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}", flush=True)

    if not a.keep:
        os.remove(zpath)


if __name__ == "__main__":
    main()
