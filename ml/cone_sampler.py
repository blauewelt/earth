#!/usr/bin/env python3
"""E-069 · gather the inner cone's raw values for a batch of anchors.

`ml/cone.py` says WHICH (lag, dy, dx) each channel reads; this file reads them
out of the tensor. It is the loader half of the cone codec, and it is the part
the plan expects to bound the run ("the sampler, not the network" — section 7),
because one anchor is ~750 scattered reads from a memmapped [T, 281, 481, C]
array and a batch is 256 of them.

Three properties are load-bearing:

  * **Offsets are per pixel ROW.** A cell is 27.83 km north-south everywhere
    and 27.83*cos(phi) km east-west, so the cone's dot list depends on latitude
    and nothing else — it is built once per row and cached, never per anchor.
  * **Values are RAW.** No climatology removal, no standardisation: that is the
    trainer's job and doing it here would bake one normalisation into every
    cached batch (ml/CLAUDE.md section 4.2 — normalise by properties of the
    DATA, in one place).
  * **Off-grid is INVALID, not wrapped.** `ml/model.py::gather_px` wraps x
    modulo W because it was written for a global tensor; this window is a
    basin (lon -100..20), so a wrap would put the Iberian shelf one cell west
    of Florida. Dots that leave the rectangle are marked invalid and zeroed.

Pool discipline (`admissible` / `certify`) generalises c25f6ff's
`--holdout-scope window` rule from one pixel-bin to the whole dot set: an
anchor is a training anchor only if every bin its cone touches — L_in pentads
back and every future target forward — is a training bin.

Pure numpy; no torch. Plan: ml/plans/E069_cone_codec.md section 3.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cone import (channel_dots, channel_depth_dbar, channel_family,   # noqa: E402
                  ground_km, KM_PER_DEG)

PENTAD_EPOCH = np.datetime64("1982-01-01")
PENTAD_DAYS = 5

# The 3x3 patch, in `ml/model.py::gather_px`'s order: dy outer, dx inner, both
# -1..1, so index 4 is the centre cell. Every archived codec's val_proj was
# trained on that layout; changing it silently changes what channel 0 means.
PATCH_DY = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], np.int64)
PATCH_DX = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], np.int64)


def pentad_doy(t):
    """Day-of-year of pentad bin `t`. Pentads are 5-day bins from 1982-01-01
    (ml/build_family4.py), so bin t opens on day 5t and the season the codec's
    context token carries is that day's — a bin never straddles more than five
    days, which is finer than any seasonal term in the loss."""
    t = np.asarray(t, np.int64)
    d = PENTAD_EPOCH + (PENTAD_DAYS * t).astype("timedelta64[D]")
    return (d - d.astype("datetime64[Y]")).astype("timedelta64[D]").astype(
        np.int64) + 1


class ConeSampler:
    """Gather the inner cone for anchors (t, y, x) out of X[T, H, W, C].

    Parameters
    ----------
    X, OBS : array-like [T, H, W, C]
        The tensor and its observed mask. Either may be a memmap or an npz
        member; both are indexed lazily and NEVER materialised (`X[:]` on the
        pentad tensor is 33 GB — ml/model.py::LazyPixels was written for
        exactly this failure).
    lats, lons : 1-D arrays
        The grid axes, used for the per-row cos(phi) and for the context token.
    chan_names : sequence of str
        Channel names in tensor order; each is mapped to a cone family by
        `cone.channel_family` and an unknown name raises there rather than
        being given a silent default reach.
    L_in : int
        Inner-window depth in pentads (6 = 30 days; plan section 2 argues the
        number from displacement per lag, not from convenience).
    future_lags : tuple of int
        Which forward bins the decoder is asked for (1, 2 = t+1, t+2).
    """

    def __init__(self, X, OBS, lats, lons, chan_names, L_in=6, dlat_deg=0.25,
                 future_lags=(1, 2)):
        self.X, self.OBS = X, OBS
        self.lats = np.asarray(lats, np.float64)
        self.lons = np.asarray(lons, np.float64)
        self.chan_names = list(chan_names)
        self.L_in = int(L_in)
        self.dlat_deg = float(dlat_deg)
        self.future_lags = tuple(int(f) for f in future_lags)
        self.T, self.H, self.W, self.C = (int(v) for v in X.shape)
        if len(self.chan_names) != self.C:
            raise ValueError(
                f"ConeSampler: {len(self.chan_names)} channel names for a "
                f"tensor with {self.C} channels — the names ARE the family "
                f"map, so a mismatch means some channel is reading another "
                f"channel's cone.")
        if len(self.lats) != self.H or len(self.lons) != self.W:
            raise ValueError(
                f"ConeSampler: lats/lons ({len(self.lats)}/{len(self.lons)}) "
                f"do not match the tensor ({self.H}/{self.W}).")
        self.families = [channel_family(n) for n in self.chan_names]
        self.depths = np.array([channel_depth_dbar(n) for n in self.chan_names],
                               np.float32)
        self._rows = {}
        self._flatX = self._flat(X)
        self._flatO = self._flat(OBS)

    # ------------------------------------------------------------ internals --
    @staticmethod
    def _flat(A):
        """A 1-D VIEW of a C-contiguous array, or None. One fancy index into a
        flat memmap is a single scatter-gather; the four-index form walks four
        broadcast index arrays and measured ~2x slower on the pentad shape."""
        try:
            if A.flags["C_CONTIGUOUS"]:
                return A.reshape(-1)
        except Exception:
            pass
        return None

    def _gather(self, A, flat, t, y, x, c):
        """A[t, y, x, c] for broadcast index arrays, flat where possible."""
        if flat is not None:
            idx = ((t * self.H + y) * self.W + x) * self.C + c
            return flat[idx.ravel()].reshape(idx.shape)
        t, y, x, c = np.broadcast_arrays(t, y, x, c)
        return A[t, y, x, c]

    def row(self, y):
        """The cone's flattened dot table for grid row `y` (cached).

        Keys: `lag`, `dy`, `dx`, `chan` (index arrays over the dot set),
        `dy_km`, `dx_km`, `lag_days`, `depth` (the per-dot coordinates the
        codec's Fourier encoding reads), and `n`.
        """
        y = int(y)
        hit = self._rows.get(y)
        if hit is not None:
            return hit
        lat = float(self.lats[y])
        lag, dy, dx, chan = [], [], [], []
        for ci, name in enumerate(self.chan_names):
            for l, ddy, ddx in channel_dots(lat, name, L_in=self.L_in,
                                            dlat_deg=self.dlat_deg):
                lag.append(l)
                dy.append(ddy)
                dx.append(ddx)
                chan.append(ci)
        lag = np.array(lag, np.int64)
        dy = np.array(dy, np.int64)
        dx = np.array(dx, np.int64)
        chan = np.array(chan, np.int64)
        ykm, xkm = ground_km(dy.astype(np.float64), dx.astype(np.float64),
                             lat, self.dlat_deg)
        rec = dict(
            n=int(len(lag)), lat=lat, lag=lag, dy=dy, dx=dx, chan=chan,
            chan16=chan.astype(np.int16),
            dy_km=ykm.astype(np.float32), dx_km=xkm.astype(np.float32),
            lag_days=(PENTAD_DAYS * lag).astype(np.float32),
            depth=self.depths[chan],
        )
        self._rows[y] = rec
        return rec

    def n_dots(self, y):
        """Token count of the dot set at row `y` (the patch adds C more)."""
        return self.row(y)["n"]

    # --------------------------------------------------------------- sample --
    def sample(self, anchors):
        """Gather the inner cone, the lag-0 patch and the future targets.

        `anchors` is an int array [B, 3] of (t, y, x). Returns a dict of numpy
        arrays padded to a fixed N = max dot count over the batch:

          vals[B, N]      float32, raw tensor values (NaN -> 0)
          obs[B, N]       bool, observed in the DATA
          valid[B, N]     bool, the dot exists: on the grid and 0 <= t-l < T.
                          Padding is invalid, so `valid` is the attention mask.
          chan[B, N]      int16 channel index
          dy_km, dx_km    float32 signed ground offsets, km
          lag_days[B, N]  float32, 5 * lag
          depth[B, N]     float32 dbar (0 for surface channels)
          patch_vals[B, C, 9], patch_obs[B, C, 9]   the lag-0 3x3
          fut_vals[B, C, F], fut_obs[B, C, F]       the anchor at t+f
          ctx[B, 4]       sin/cos of the season, lat/90, lon/180
          anchors[B, 3]   echoed, so a downstream cache is self-describing
        """
        anchors = np.asarray(anchors, np.int64)
        if anchors.ndim != 2 or anchors.shape[1] != 3:
            raise ValueError(f"sample(): anchors must be [B, 3], got "
                             f"{anchors.shape}")
        B = anchors.shape[0]
        t, y, x = anchors[:, 0], anchors[:, 1], anchors[:, 2]

        bands = {}
        for i in range(B):
            bands.setdefault(int(y[i]), []).append(i)
        N = max((self.n_dots(r) for r in bands), default=0)

        vals = np.zeros((B, N), np.float32)
        obs = np.zeros((B, N), bool)
        valid = np.zeros((B, N), bool)
        chan = np.zeros((B, N), np.int16)
        dy_km = np.zeros((B, N), np.float32)
        dx_km = np.zeros((B, N), np.float32)
        lag_days = np.zeros((B, N), np.float32)
        depth = np.zeros((B, N), np.float32)

        for r, ii in bands.items():
            R = self.row(r)
            n = R["n"]
            idx = np.asarray(ii, np.int64)
            tt = t[idx][:, None] - R["lag"][None, :]
            yy = r + R["dy"][None, :]
            xx = x[idx][:, None] + R["dx"][None, :]
            cc = R["chan"][None, :]
            ok = ((tt >= 0) & (tt < self.T) & (yy >= 0) & (yy < self.H)
                  & (xx >= 0) & (xx < self.W))
            ok = np.broadcast_to(ok, (len(idx), n))
            raw = self._gather(self.X, self._flatX,
                               np.clip(tt, 0, self.T - 1),
                               np.clip(yy, 0, self.H - 1),
                               np.clip(xx, 0, self.W - 1), cc)
            o = self._gather(self.OBS, self._flatO,
                             np.clip(tt, 0, self.T - 1),
                             np.clip(yy, 0, self.H - 1),
                             np.clip(xx, 0, self.W - 1), cc)
            raw = np.asarray(raw, np.float32)
            raw = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
            sl = (idx[:, None], np.arange(n)[None, :])
            vals[sl] = raw
            obs[sl] = np.asarray(o, bool) & ok
            valid[sl] = ok
            chan[sl] = R["chan16"][None, :]
            dy_km[sl] = R["dy_km"][None, :]
            dx_km[sl] = R["dx_km"][None, :]
            lag_days[sl] = R["lag_days"][None, :]
            depth[sl] = R["depth"][None, :]
        # A dot the data never observed is still a real token (PixelMAE's
        # `miss_tok`); only a dot that does not EXIST is invalid.
        vals = np.where(valid, vals, 0.0).astype(np.float32)

        patch_vals, patch_obs = self._patch(t, y, x)
        fut_vals, fut_obs = self._future(t, y, x)
        doy = pentad_doy(t).astype(np.float64)
        ang = 2.0 * np.pi * doy / 365.0
        ctx = np.stack([np.sin(ang), np.cos(ang),
                        self.lats[y] / 90.0, self.lons[x] / 180.0],
                       axis=1).astype(np.float32)
        return dict(vals=vals, obs=obs, valid=valid, chan=chan,
                    dy_km=dy_km, dx_km=dx_km, lag_days=lag_days, depth=depth,
                    patch_vals=patch_vals, patch_obs=patch_obs,
                    fut_vals=fut_vals, fut_obs=fut_obs, ctx=ctx,
                    anchors=anchors)

    def _patch(self, t, y, x):
        """The lag-0 3x3 for every channel, [B, C, 9] — `gather_px`'s tokens,
        except that a cell off the western or eastern edge is UNOBSERVED rather
        than wrapped (see the module docstring)."""
        B = len(t)
        C = self.C
        yy = y[:, None] + PATCH_DY[None, :]                    # [B, 9]
        xx = x[:, None] + PATCH_DX[None, :]
        ok = ((yy >= 0) & (yy < self.H) & (xx >= 0) & (xx < self.W)
              & (t[:, None] >= 0) & (t[:, None] < self.T))
        yc = np.clip(yy, 0, self.H - 1)[:, None, :]            # [B, 1, 9]
        xc = np.clip(xx, 0, self.W - 1)[:, None, :]
        tc = np.clip(t, 0, self.T - 1)[:, None, None]
        cc = np.arange(C, dtype=np.int64)[None, :, None]
        raw = np.asarray(self._gather(self.X, self._flatX, tc, yc, xc, cc),
                         np.float32)
        o = np.asarray(self._gather(self.OBS, self._flatO, tc, yc, xc, cc),
                       bool)
        ok3 = np.broadcast_to(ok[:, None, :], (B, C, 9))
        raw = np.where(np.isfinite(raw) & ok3, raw, 0.0).astype(np.float32)
        return raw, (o & ok3)

    def _future(self, t, y, x):
        """The anchor column at t+f for each f in `future_lags`, [B, C, F].
        Past the end of the tensor the target does not exist, so it is
        unobserved — never a zero the decoder could be scored against."""
        B, C = len(t), self.C
        F = len(self.future_lags)
        f = np.array(self.future_lags, np.int64)
        tt = t[:, None] + f[None, :]                            # [B, F]
        ok = (tt >= 0) & (tt < self.T)
        tc = np.clip(tt, 0, self.T - 1)[:, None, :]             # [B, 1, F]
        yc = y[:, None, None]
        xc = x[:, None, None]
        cc = np.arange(C, dtype=np.int64)[None, :, None]
        raw = np.asarray(self._gather(self.X, self._flatX, tc, yc, xc, cc),
                         np.float32)
        o = np.asarray(self._gather(self.OBS, self._flatO, tc, yc, xc, cc),
                       bool)
        ok3 = np.broadcast_to(ok[:, None, :], (B, C, F))
        raw = np.where(np.isfinite(raw) & ok3, raw, 0.0).astype(np.float32)
        return raw, (o & ok3)

    # ----------------------------------------------------------- pool rules --
    def bin_span(self):
        """The bins one anchor touches, as offsets from t: -F .. +L_in, i.e.
        every t - l for l <= L_in and every t + f."""
        back = list(range(0, self.L_in + 1))
        fwd = [-f for f in self.future_lags]
        return np.array(sorted(set(back + fwd)), np.int64)

    def admissible(self, anchors, train_bins):
        """[B] bool: is every bin this anchor's cone touches a training bin?

        c25f6ff's `--holdout-scope window` rule said a training pixel-bin may
        not read a held-out bin. The cone reads L_in pentads of history and
        predicts two forward, so the same rule over the whole span is what
        keeps a held-out bin out of the training set by ANY path — including
        the one that leaks hardest, a future target inside the holdout.

        A bin outside [0, T) is not a training bin: an anchor whose cone runs
        off the archive is inadmissible, not silently short.
        """
        anchors = np.asarray(anchors, np.int64)
        train_bins = np.asarray(train_bins, bool)
        if train_bins.shape != (self.T,):
            raise ValueError(f"admissible(): train_bins must be [T={self.T}], "
                             f"got {train_bins.shape}")
        t = anchors[:, 0]
        bins = t[:, None] - self.bin_span()[None, :]
        inside = (bins >= 0) & (bins < self.T)
        good = np.where(inside, train_bins[np.clip(bins, 0, self.T - 1)], False)
        return good.all(axis=1)

    def certify(self, anchors, train_bins):
        """Brute-force count of pool violations over `anchors` — 0 for an
        admitted batch.

        Deliberately a plain loop over anchors and dots and NOT a rearrangement
        of `admissible`: a certificate written from the same expression it
        certifies proves only that the expression is self-consistent. E-059
        ran the same check before training; this is its cone-shaped form, and
        the trainer calls it once per run, not per batch.
        """
        anchors = np.asarray(anchors, np.int64)
        train_bins = np.asarray(train_bins, bool)
        T = self.T
        bad = 0
        for a in range(anchors.shape[0]):
            t = int(anchors[a, 0])
            hit = False
            for lag in range(0, self.L_in + 1):
                b = t - lag
                if b < 0 or b >= T or not bool(train_bins[b]):
                    hit = True
                    break
            if not hit:
                for f in self.future_lags:
                    b = t + f
                    if b < 0 or b >= T or not bool(train_bins[b]):
                        hit = True
                        break
            bad += int(hit)
        return bad
