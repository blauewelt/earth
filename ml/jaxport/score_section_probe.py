#!/usr/bin/env python3
"""`probe_kfold`'s rapid ridge, end to end, with a SWAPPABLE encoder backend.

This is the instrument that scores tier 2's embedding path. The port's value
is that it is a second, independent implementation of the same arithmetic
(`ml/plans/JAX_PORT.md` §1), and that is worth nothing unless it can be held
against a number nobody chose afterwards. So:

**The read-out is the SAME numpy code for both backends.** `anomaly_transform`,
`section_of` and `kfold_r` are IMPORTED from the operational modules, never
copied — the only thing `--backend jax` changes is which encoder produced Z.
If the two backends disagreed because the ridge had been transcribed slightly
differently, the comparison would measure the transcription.

**The pre-registered constant.** `ml/EXPERIMENTS.md` records the archived
`probe_kfold` control for this exact codec (`f3_anchor41M`, tag run-80) on this
exact tensor (`f3r1`/`na025`, sha `adcbe700fb…`): rapid k-fold **r = 0.627,
95% CI [0.503, 0.735], n = 240, RMSE 2.17 Sv** (`probes-140.json`). The torch
backend here must land on it; a JAX number scored while the torch
reproduction is off target would be scored against the wrong thing, so this
script prints both and the operator compares them. **A mismatch is a finding,
never a tolerance to widen** (`ml/CLAUDE.md` §3b).

    python3 ml/jaxport/score_section_probe.py \
        --data /tmp/f3_X.npy --meta /tmp/f3_meta.npz \
        --ckpt /tmp/f3_anchor.pt --backend torch --out /tmp/probe_torch.json

**THE TENSOR IS TRANSFORMED IN PLACE, AND THAT IS LOAD-BEARING.**
`anomaly_transform` writes X back to disk (it must: 10.88 GB against a 7 GB
box is 1.55× oversubscribed, so there is nowhere to put a second copy). Running
it twice would z-score already-anomalised data — a known failure mode in this
repo, and one with no symptom: every downstream number stays finite and
plausible. So this script writes a MARKER next to the tensor recording the
transform and the holdout configuration it used, refuses to transform a tensor
that already carries one, and refuses outright if the marker's holdout does not
match the checkpoint's. To get the raw tensor back, delete the marker and
re-run the decompressor that produced the file.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path.insert(0, ML)

from trainprobe import anomaly_transform                       # noqa: E402
from temporal import section_of                                # noqa: E402
from probe_kfold import kfold_r, TARGETS                       # noqa: E402
from model import LazyPixels                                   # noqa: E402

# The archived control this script exists to be scored against. Quoted, not
# recomputed — see the module docstring.
PREREG = {"r": 0.627, "ci95": [0.503, 0.735], "n": 240, "rmse_sv": 2.17,
          "source": "probes-140.json, ml/EXPERIMENTS.md",
          "codec": "f3_anchor41M (run-80)", "tensor_sha_prefix": "adcbe700fb"}


# --------------------------------------------------------------------------
# the raw-tensor passes, both of which must happen BEFORE the transform
# --------------------------------------------------------------------------
def ocean_mask(X, chunk=16, cache=None):
    """`np.isfinite(X[..., 0]).any(0)`, in time blocks.

    Identical values; what changes is the peak. The one-liner materialises a
    [T,H,W] bool — **6.6 GB on this tensor**, on a box with 7 GB of RAM and a
    10.88 GB memmap already competing for the page cache. The block form holds
    one [chunk,H,W] bool (2.2 MB at chunk=16) and one [H,W] accumulator.

    It reads the RAW tensor, so it must run before `anomaly_transform`; the
    result is cached beside the tensor because the transform then makes it
    underivable from the file on disk.
    """
    if cache and os.path.exists(cache):
        return np.load(cache)
    T = X.shape[0]
    ocean = np.zeros(X.shape[1:3], dtype=bool)
    for i0 in range(0, T, chunk):
        blk = X[i0:min(i0 + chunk, T), ..., 0]
        ocean |= np.isfinite(blk).any(axis=0)
    if cache:
        np.save(cache, ocean)
    return ocean


class FlushingMemmap:
    """A write-through view of the on-disk tensor that msyncs as it goes.

    WHY THIS EXISTS, measured rather than modelled. `anomaly_transform`
    rewrites all 10.88 GB of this tensor twice (pass 2 writes the anomaly,
    pass 3 the z-score). numpy's memmap only msyncs on `flush()`/close, and
    anomaly_transform — an operational file, not ours to change — never calls
    it mid-pass, because on the 64/128 GB fleet boxes it never needed to. This
    box has 8 GB against a 10.88 GB file (**1.55× oversubscribed**), so the
    dirty pages accumulate faster than the kernel writes them back: the first
    attempt at this run was killed in pass 3 with no traceback, no OOM line
    and a half-z-scored tensor on disk.

    So the caller hands the transform this proxy instead. The arithmetic and
    the indexing are identical — every read and write goes straight to the
    same memmap — and an `msync` every `flush_bytes` of writes turns the dirty
    pages clean, and therefore reclaimable, before they can pile up. It costs
    nothing but ordering: the same bytes are written, just sooner.
    """

    def __init__(self, mm, flush_bytes=1 << 30):
        self._mm = mm
        self._flush_bytes = flush_bytes
        self._pending = 0
        self.shape = mm.shape
        self.dtype = mm.dtype
        self.size = mm.size
        self.ndim = mm.ndim

    def __getitem__(self, idx):
        return self._mm[idx]

    def __setitem__(self, idx, val):
        self._mm[idx] = val
        self._pending += int(np.asarray(val).nbytes)
        if self._pending >= self._flush_bytes:
            self._mm.flush()
            self._pending = 0

    def flush(self):
        self._mm.flush()
        self._pending = 0


def _marker_path(data):
    return data + ".anomaly.json"


def ensure_anomaly(X, moy, t_hold, x_hold, data, chunk=16):
    """Run `anomaly_transform` exactly once over the on-disk tensor.

    The marker records WHICH holdout the transform used, because the transform
    is not just idempotency-sensitive, it is configuration-sensitive: the
    climatology is a mean over non-held-out years and the z-score pool excludes
    the held-out longitudes. A tensor transformed under one holdout and probed
    under another is wrong in a way nothing downstream can see.
    """
    key = {"t_hold_sha": hashlib.sha256(
               np.asarray(t_hold, bool).tobytes()).hexdigest()[:16],
           "x_hold_sha": hashlib.sha256(
               np.asarray(x_hold, bool).tobytes()).hexdigest()[:16],
           "shape": list(X.shape)}
    mp = _marker_path(data)
    if os.path.exists(mp):
        with open(mp) as fh:
            have = json.load(fh)
        for k in ("t_hold_sha", "x_hold_sha", "shape"):
            if have.get(k) != key[k]:
                sys.exit(
                    f"{os.path.basename(data)} was already anomaly-transformed "
                    f"under a DIFFERENT configuration ({k}: {have.get(k)} vs "
                    f"{key[k]}). Re-decompress the raw tensor and delete "
                    f"{mp} — never transform twice.")
        print(f"  anomaly transform: already applied "
              f"(marker {os.path.basename(mp)}), skipping", flush=True)
        return
    t0 = time.time()
    # chunk well below the 64 default: the transform's own docstring puts the
    # peak at ~13·chunk·H·W·C bytes, which at 64 is 4.4 GiB on a box with 8 GB
    # already holding a 10.88 GB memmap's pages. `chunk` changes memory only,
    # never the answer (tests/test_anomaly_chunked.py check 2).
    guarded = FlushingMemmap(X)
    anomaly_transform(guarded, moy, t_hold, x_hold, chunk=chunk, verbose=True)
    guarded.flush()
    key["seconds"] = round(time.time() - t0, 1)
    key["when"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # FLUSH, THEN MARK (ml/CLAUDE.md §5.21): a marker written before the pages
    # are on disk would let the next run skip a transform that never landed.
    with open(mp, "w") as fh:
        json.dump(key, fh, indent=2)
    print(f"  anomaly transform: {key['seconds'] / 60:.1f} min, marker written",
          flush=True)


# --------------------------------------------------------------------------
# the target series — probe_kfold.main's `target_series`, rapid branch
# --------------------------------------------------------------------------
def rapid_series(meta, moy):
    """`probe_kfold.main.target_series(TARGETS['rapid'])`, mirrored.

    That function is a closure inside `main()` and cannot be imported, so the
    four lines it runs for the `rapid` key are reproduced here verbatim —
    including the deseasonalisation with the OVERALL monthly climatology,
    which probe_kfold documents as a deliberate simplification (every month is
    a test month in some fold; per-fold climatologies differ by <0.1 Sv).
    The `rapid` key takes neither the family-4 `bin_index` branch nor the
    YYYYMM decode: its first column is already a month index into the tensor.

    The construction is asserted, not assumed — see `check_series`.
    """
    arr = meta["rapid"]
    tidx = arr[:, 0].astype(int)
    vals = arr[:, 1].copy()
    tmoy = moy[tidx]
    clim = np.array([vals[tmoy == m].mean() for m in range(12)])
    return tidx, vals - clim[tmoy]


def check_series(tidx, v_des, vals_raw, moy, T):
    """Properties that pin the mirror against probe_kfold's construction.

    Not a restatement of the code (that would pass on a copied bug) but of
    what the code MEANS: month indices into this tensor, in range and unique;
    every calendar month's deseasonalised mean exactly zero, which is true of
    subtracting the overall monthly climatology and of nothing else; and the
    n the archive recorded.
    """
    problems = []
    if tidx.min() < 0 or tidx.max() >= T:
        problems.append(f"tidx out of range [0,{T})")
    if len(set(tidx.tolist())) != len(tidx):
        problems.append("tidx has duplicates")
    tmoy = moy[tidx]
    for m in range(12):
        s = tmoy == m
        if s.any() and abs(float(v_des[s].mean())) > 1e-4:
            problems.append(f"month {m + 1} deseasonalised mean "
                            f"{float(v_des[s].mean()):.2e} != 0")
    if len(tidx) != PREREG["n"]:
        problems.append(f"n = {len(tidx)}, archive recorded {PREREG['n']}")
    if problems:
        sys.exit("rapid target series does not match probe_kfold's "
                 "construction: " + "; ".join(problems))


# --------------------------------------------------------------------------
def embed_torch(ck, Xt, OBS, ctx_all, lats, lons, ys_s, xs_s):
    import torch
    from model import codec_from_ckpt
    from temporal import embed_everything
    codec = codec_from_ckpt(ck, Xt.shape[-1])
    codec.load_state_dict(ck["model"])
    codec.eval()
    # CPU only on this box. embed_everything follows the model's device.
    codec.to(torch.device("cpu"))
    Z, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons, ys_s, xs_s,
                            codec.d_z, cache_path=None)
    return Z, codec.d_z


def embed_jax(ck, Xt, OBS, ctx_all, lats, lons, ys_s, xs_s):
    from jaxport.convert import codec_from_ckpt_jax
    from jaxport.embed import embed_everything_jax
    codec = codec_from_ckpt_jax(ck, Xt.shape[-1])
    Z, _ = embed_everything_jax(codec, Xt, OBS, ctx_all, lats, lons, ys_s,
                                xs_s, codec.d_z)
    return Z, codec.d_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="the tensor as a .npy (memmapped r+; transformed in "
                         "place — see the module docstring)")
    ap.add_argument("--meta", required=True,
                    help="npz of the tensor's small members: months, lats, "
                         "lons, chan, rapid, truth_*")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backend", choices=["jax", "torch"], required=True)
    ap.add_argument("--target", default="rapid", choices=sorted(TARGETS))
    ap.add_argument("--out", default=None, help="write the numbers as JSON")
    ap.add_argument("--z-out", default=None,
                    help="write the section Z (small) for a cross-backend diff")
    ap.add_argument("--compare-z", default=None,
                    help="an earlier --z-out; report max elementwise |Δ|")
    # chunk=8, not the 64 default: the transform's peak is ~13·chunk·H·W·C
    # bytes, which is 4.4 GiB at 64 on a box with 8 GB. See ensure_anomaly.
    ap.add_argument("--anomaly-chunk", type=int, default=8)
    a = ap.parse_args()

    meta = np.load(a.meta, allow_pickle=True)
    months = [str(m) for m in meta["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    lats, lons = meta["lats"], meta["lons"]
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)

    import torch
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)

    # NEVER np.load this: 10.88 GB against 7 GB of RAM. r+ because
    # anomaly_transform writes.
    X = np.load(a.data, mmap_mode="r+")
    if len(ck["chan"]) != X.shape[-1]:
        sys.exit(f"codec has {len(ck['chan'])} channels but the tensor has "
                 f"{X.shape[-1]}")

    t_hold = np.array([m[:4] in set(ck["args"]["holdout_years"].split(","))
                       for m in months])
    lo_, hi_ = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo_) & (lons < hi_)

    print(f"tensor {X.shape} {X.dtype} · codec {ck.get('tag')} "
          f"patch {ck['args'].get('patch', 1)} d_z {ck['d_z']} · "
          f"backend {a.backend}", flush=True)

    ocean = ocean_mask(X, chunk=a.anomaly_chunk,
                       cache=a.data + ".ocean.npy")
    ys, xs = np.where(ocean)
    print(f"  ocean pixels: {len(ys)}", flush=True)

    ensure_anomaly(X, moy, t_hold, x_hold, a.data, chunk=a.anomaly_chunk)

    spec = TARGETS[a.target]
    sec_y, sec_sel = section_of(lats, lons, ys, xs, spec["lat"], *spec["lon"])
    # argmin() clamps to the window edge, so a section outside the window
    # would silently probe the wrong latitude — probe_kfold's own guard.
    if abs(float(lats[sec_y]) - spec["lat"]) > 1.0 or len(sec_sel) < 5:
        sys.exit(f"{a.target}: section outside the tensor window")
    print(f"  section: row {sec_y} (lat {float(lats[sec_y]):.2f}), "
          f"{len(sec_sel)} pixels", flush=True)

    if a.target != "rapid":
        sys.exit("only --target rapid is implemented in this reference "
                 "driver; the other targets take probe_kfold's bin_index / "
                 "YYYYMM branches, which nothing here exercises yet")
    tidx, v_des = rapid_series(meta, moy)
    check_series(tidx, v_des, meta["rapid"][:, 1], moy, X.shape[0])

    # LazyPixels: nan_to_num / isfinite evaluated AFTER the per-batch index.
    # `np.isfinite(X)` at full size is 10.6 GB here and there is nowhere to
    # put it. Note X must NOT be pre-filled with nan_to_num — OBS would then
    # be all-True and every land cell would enter the encoder as an observed
    # 0.0 (the trap probe_kfold documents at its own LazyPixels call).
    Xt, OBS = LazyPixels(X), LazyPixels(X, obs=True)

    t0 = time.time()
    embed = embed_jax if a.backend == "jax" else embed_torch
    Z, d_z = embed(ck, Xt, OBS, ctx_all, lats, lons, ys[sec_sel], xs[sec_sel])
    embed_s = time.time() - t0

    F = Z.mean(1)[tidx]
    r, lo95, hi95, n, rmse, sigma, pred = kfold_r(F, v_des, yr[tidx])

    zdelta = None
    if a.z_out:
        np.save(a.z_out, np.asarray(Z))
    if a.compare_z:
        other = np.load(a.compare_z)
        if other.shape != Z.shape:
            sys.exit(f"--compare-z shape {other.shape} != {Z.shape}")
        zdelta = float(np.max(np.abs(np.asarray(Z, np.float64)
                                     - other.astype(np.float64))))

    # probe_kfold's own one-line style.
    print(f"{a.backend:<10} {a.target}: r={r:.3f} [{lo95:.3f},{hi95:.3f}] "
          f"n={n} rmse={rmse:.2f} Sv (sigma {sigma:.2f})")
    print(f"           pre-registered control: r={PREREG['r']:.3f} "
          f"[{PREREG['ci95'][0]:.3f},{PREREG['ci95'][1]:.3f}] "
          f"n={PREREG['n']} rmse={PREREG['rmse_sv']:.2f} Sv "
          f"— Δr {r - PREREG['r']:+.4f}")
    if zdelta is not None:
        print(f"           max elementwise |Δ| vs {a.compare_z}: {zdelta:.3e}")
    print(f"           embedding {embed_s / 60:.1f} min "
          f"({Z.shape[0]}×{Z.shape[1]} encoder forwards)")

    res = {"backend": a.backend, "target": a.target, "r": round(r, 6),
           "ci95": [round(lo95, 6), round(hi95, 6)], "n": int(n),
           "rmse_sv": round(rmse, 6), "sigma_sv": round(sigma, 6),
           "section_pixels": int(len(sec_sel)), "section_row": int(sec_y),
           "section_lat": float(lats[sec_y]), "d_z": int(d_z),
           "months": int(Z.shape[0]), "embed_minutes": round(embed_s / 60, 2),
           "codec_tag": str(ck.get("tag")), "prereg": PREREG,
           "delta_r_vs_prereg": round(r - PREREG["r"], 6),
           "z_max_abs_delta_vs": ({"path": a.compare_z, "max_abs": zdelta}
                                  if zdelta is not None else None)}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"           wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
