"""Copy-reconstruction audit (E-019a): how close is encode→decode to identity?

Chris, 2026-08-12: "We take some input. We encode it. We decode it, and then
we look how far away it is, and it should be almost identical." This script
measures exactly that, for the embedding the whole programme consumes: each
section pixel-month is encoded at FULL VISIBILITY (mask = none — the same
call temporal.py's embed_everything makes when it builds Z for stage 2 and
for the 0.631 transport probe), then the decoder is queried for all C
channels at offset (0,0,0) and compared to the true standardized values.

Why this is the right first measurement for the decoder programme:
  · The training loss reconstructs MASKED channels (weight 1.0) and only
    incidentally the visible ones (weight 0.1) — so nobody has ever measured
    how faithful the full-visibility round trip is, and loss_rec (~0.09 at
    the end of run-62) is a masked-dominated Huber mix, not this number.
  · If the round trip already reads r ≈ 0.99 everywhere, the 0.631 probe
    ceiling is NOT a bottleneck-lossiness story and decoder work will not
    move it (that is this audit's falsifier). If specific channels lose
    real variance, those channels name what the bottleneck discards, and
    d_z / decoder capacity / loss weighting become the levers.

Protocol notes:
  · The encode path IS temporal.embed_everything — not a reimplementation.
    A 3-row latitude slab around the section keeps gather_px's behaviour
    bit-identical for interior rows (dy = ±1 stays in bounds, lon wraps on
    the full W); the slab is standardized by a STREAMING replica of
    temporal.py's anomaly transform (the in-RAM original needs > 11 GB),
    and the replica is verified against the exact in-RAM recipe on two
    full channels before anything is scored (--skip-verify to bypass).
  · Scores are split three ways, matching the codec's own blocked holdout:
    train (train months, non-holdout lons) · heldout months (2009/2017/2023
    by default, from the checkpoint args) · heldout lon block (-45,-25).
    A large train-vs-holdout fidelity gap = the decoder memorises rather
    than compresses; similar numbers = the round trip generalises.
  · Standardized units make the numbers legible: RMSE² ≈ fraction of the
    channel's variance the round trip loses; r is the same story scale-free.
  · The pooled view (section-mean of recon vs section-mean of truth, per
    month) is reported separately: the transport probe reads pooled
    embeddings, and pooling can cancel per-pixel error.

E-049 PRE-AUDIT ADAPTATION (2026-08-25). This script and recon_decoder.py
were written against the MONTHLY family-3 section (float32, T = 516, 39
channels). Road B audits a PENTAD family-4-r2 tensor (float16, T ~ 3139, 40
channels) through a d_z 6 / FSQ [8,8,8,5,5,5] bottleneck, and four things had
to change before the numbers could be trusted. Each is documented at its site;
the summary, because the first of them is silent:

  1. `stream_stats` summed float16 with a FLOAT16 ACCUMULATOR (§stream_stats).
     At family-4 shape one spatial slice is H*W = 135,161 values; a channel
     whose mean is ~20 sums to 2.7e6, which is 41x float16's 65,504 ceiling.
     The sum returns inf, the per-(t,c) mean nan, `np.nanstd(...) > 1e-6`
     False — and the channel is classified NOT DYNAMIC, handed to the codec
     un-standardized, and produces plausible garbage with no error anywhere.
     Fixed by accumulating in float64 when the memmap is float16 (float32
     tensors keep `dtype=None`, i.e. every family-3 number is bit-identical),
     and by a REFUSAL: a channel with observed values whose statistics come
     out non-finite stops the audit and is named.
  2. The climatology counter was uint8 (§stream_stats). It counted <= 43
     monthly samples per month-of-year at family 3 and ~244 of 255 at pentad
     cadence — a longer tensor wraps silently and divides the sum by the
     wrong count. uint16 now, with an explicit refusal above its own ceiling.
  3. `--x` accepts the TENSOR NPZ as well as an extracted X.npy
     (§open_x/`extract_x`), following `ml/tensor_io.py`'s sidecar convention
     and `scripts/dectrain_extract.py`'s extraction.
  4. The AUDIT'S OWN AXIS is now the Argo-bin split (§argo_bin_mask,
     §score_bins) — per-(t, pixel) "is any rg_t/rg_s channel observed here",
     scored as ADDITIONAL JSON keys. `score()` and `score_pooled()` are
     untouched, because 183 archived monthly bundles read their output.

Usage (sandbox or box; CPU is fine for the 265-pixel section):
  python3 ml/recon_eval.py --x ml/cache/family3_X.npy \
      --npz ml/cache/family3_na025.npz \
      --ckpt ml/cache/f3_anchor41M__pixelmae.pt \
      --out ml/runs/recon_audit/recon_eval.json

  # pentad family-4-r2: one argument, metadata read from the same npz
  python3 ml/recon_eval.py --x ml/cache/family4_na025_pentad_r2.npz \
      --ckpt ml/cache/f4r2-40M-dz6-fsq65k__pixelmae.pt \
      --out ml/runs/recon_audit/e049b.json
"""
import argparse
import json
import os
import shutil
import sys
import time
import warnings
import zipfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import codec_from_ckpt                     # noqa: E402
from temporal import embed_everything, section_of     # noqa: E402
from tensor_io import sidecar_path                    # noqa: E402

RAPID_LAT, RAPID_LON = 26.5, (-80.0, -13.0)

# The Argo channel family, BY NAME. `build_family3.CHANS` spells the 32 Roemmich
# -Gilson channels `rg_t<dbar>` / `rg_s<dbar>`, and family 4 imports that list
# rather than restating it (ml/build_family4.py: "ONE definition of the channel
# set, imported rather than restated"), so r1 and r2 carry the identical names
# at the identical indices. Matching on the NAME is what keeps the split
# correct when a revision appends a channel — r2's `sst` is index 39 — and it
# is why nothing here hardcodes 3..34.
ARGO_PREFIXES = ("rg_t", "rg_s")


def acc_dtype(dt):
    """The accumulator numpy must be TOLD to use, or None for its default.

    float16 is the only storage dtype in this repo whose default reduction
    accumulator cannot hold a spatial sum: at H=281, W=481 a channel with mean
    ~20 sums to 2.7e6 against float16's 65,504 ceiling. `ml/trainprobe.py`'s
    `anomaly_transform` already carries the long version of this lesson
    ("EVERY reduction below names float64, and that is not cosmetic") — this
    is the same rule for the audit's own streaming replica.

    Returns None for float32/float64 so every reduction on an existing
    family-2/3 tensor stays numpy's default and every archived number is
    reproduced bit-for-bit.
    """
    return np.float64 if np.dtype(dt) == np.float16 else None


def argo_channels(chan):
    """(argo indices, fast indices) from the channel NAMES. Refuses if empty.

    An audit that silently found zero Argo channels would score every bin as
    Argo-free and answer the falsifier's question with the wrong population —
    exactly the class of silence this pre-audit pass exists to remove.
    """
    argo = [i for i, nm in enumerate(chan) if str(nm).startswith(ARGO_PREFIXES)]
    if not argo:
        raise SystemExit(
            f"no channel name starts with {ARGO_PREFIXES} among {list(chan)} — "
            f"the Argo-bin split cannot be built, and scoring every bin as "
            f"'Argo-free' would answer the E-049 falsifier on the wrong "
            f"population. Refusing.")
    fast = [i for i in range(len(chan)) if i not in set(argo)]
    return argo, fast


def bottleneck_spec(ck):
    """What this checkpoint's BOTTLENECK is, read from the file (§0.1).

    d_z alone stopped describing the bottleneck at E-046: two checkpoints with
    the same d_z and the same parameter count can differ by a quantizer that
    carries no weights at all, and `codec_from_ckpt` rebuilds it from these
    exact keys. So the audit STATES what it audited, in its stdout and in its
    JSON, rather than leaving a reader to infer d_z 6 + FSQ from the filename.
    """
    a = ck.get("args", {}) or {}
    lv = str(a.get("fsq_levels", "") or "")
    spec = {
        "d_z": int(ck["d_z"]),
        "fsq_levels": lv,
        "fsq_ladder": str(a.get("fsq_ladder", "") or "") if lv else "",
        "fsq_ladder_fit": str(a.get("fsq_ladder_fit", "") or "") if lv else "",
        "fsq_bound": str(a.get("fsq_bound", "") or ""),
        "patch": int(a.get("patch", 1) or 1),
    }
    if lv:
        try:
            n = [int(v) for v in lv.split(",") if v.strip()]
            spec["codebook_log2"] = round(float(np.log2(np.prod(
                np.array(n, dtype=np.float64)))), 3)
        except ValueError:
            pass
    return spec


def bottleneck_line(spec):
    """One line naming the bottleneck audited. Printed by both scripts."""
    if not spec["fsq_levels"]:
        return (f"bottleneck audited: d_z {spec['d_z']} CONTINUOUS "
                f"(no --fsq-levels) · patch {spec['patch']}")
    return (f"bottleneck audited: d_z {spec['d_z']} · FSQ "
            f"[{spec['fsq_levels']}] ladder {spec['fsq_ladder'] or 'uniform'}"
            f"{' (fitted)' if spec['fsq_ladder_fit'] else ''} · bound "
            f"{spec['fsq_bound'] or 'none'} · codebook 2^"
            f"{spec.get('codebook_log2', float('nan')):.1f} · patch "
            f"{spec['patch']}")


# ---------------------------------------------------------------------------
# Opening X: an extracted .npy, a sidecar, or the tensor npz itself.
# ---------------------------------------------------------------------------

def npz_member_offset(npz_path, member="X.npy"):
    """Absolute byte offset of an UNCOMPRESSED zip member, or None.

    `np.load(..., mmap_mode=)` works only on a bare `.npy`, which is why
    `ml/tensor_io.py` invented the sidecar layout for family 5. But a member
    stored with `np.savez` (no compression) IS contiguous inside the zip and
    can be memory-mapped in place at its data offset — no copy, no second
    34 GB on disk. A member written by `np.savez_compressed` cannot, and that
    is what family 4 used (`ml/build_family4.py:921`), so this returns None
    for it and the caller refuses with the extraction command.
    """
    with zipfile.ZipFile(npz_path) as z:
        try:
            info = z.getinfo(member)
        except KeyError:
            return None
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        with open(npz_path, "rb") as fh:
            fh.seek(info.header_offset)
            hdr = fh.read(30)
            if hdr[:4] != b"PK\x03\x04":
                return None
            nlen = int.from_bytes(hdr[26:28], "little")
            xlen = int.from_bytes(hdr[28:30], "little")
            return info.header_offset + 30 + nlen + xlen


def _map_npy_at(path, offset):
    """np.memmap over an .npy image embedded at `offset` in `path`.

    The header is read through numpy's PUBLIC `read_array_header_{1,2}_0`:
    the private `_read_array_header` that dispatches on the version tuple was
    removed in numpy 2.x, and a loader that used it would break on the version
    of numpy the boxes actually have (2.4.4 here).
    """
    with open(path, "rb") as fh:
        fh.seek(offset)
        version = np.lib.format.read_magic(fh)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise SystemExit(f"{path}: .npy format version {version} is not "
                             f"one this loader can map. Use --extract-x.")
        shape, fortran, dtype = reader(fh)
        data = fh.tell()          # already absolute: `fh` is the real file
    if fortran:
        raise SystemExit(f"{path}: X is Fortran-ordered; refusing to map it "
                         f"as C-order. Use --extract-x.")
    return np.memmap(path, dtype=dtype, mode="r", offset=data, shape=shape,
                     order="C")


def extract_x(npz_path, out_path, member="X.npy", headroom=2 << 30):
    """ONE-TIME extraction of a compressed X member to a memmappable .npy.

    Follows `scripts/dectrain_extract.py` exactly — the same streaming
    `copyfileobj`, the same idempotence test (size against the zip entry), and
    the same disk guard SIZED FROM THE ALLOCATION IT GUARDS (ml/CLAUDE.md
    §5.18: the guard that fired below 8 GB against a 10.4 GiB write could not
    fire in time by construction).

    THE COST, stated because it is not small. family4_na025_pentad_r2 is
    T=3139, H=281, W=481, C=40 float16 = **33.9 GB** uncompressed, written
    once beside the npz, and the box needs that plus 2 GiB free. It buys a
    memmap, which is the difference between an audit whose resident set is the
    working set and one that decompresses 33.9 GB into RAM before it starts.
    The output is written to `<stem>_X.npy` by default — `ml/tensor_io.py`'s
    OWN sidecar name — so `load_tensor` and every future reader find it
    without being told, and the extraction is paid once per box.
    """
    with zipfile.ZipFile(npz_path) as z:
        need = z.getinfo(member).file_size
        if os.path.exists(out_path) and os.path.getsize(out_path) == need:
            print(f"X already extracted: {out_path} ({need:,} bytes)",
                  flush=True)
            return out_path
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(out_path))).free
        if free < need + headroom:
            raise SystemExit(
                f"extracting {member} from {npz_path} needs "
                f"{need / 1e9:.1f} GB (+{headroom / 1e9:.0f} GB headroom) and "
                f"the filesystem holding {out_path} has {free / 1e9:.1f} GB "
                f"free. Refusing: an ENOSPC part-way through leaves a "
                f"truncated .npy that np.load maps happily at the wrong "
                f"shape.")
        print(f"extracting {member} -> {out_path} ({need / 1e9:.1f} GB, "
              f"one-time)", flush=True)
        t0 = time.time()
        tmp = out_path + ".part"
        with z.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
    # flush THEN mark (ml/CLAUDE.md §5.21): the final name appears only once
    # the bytes are all there, so a killed job can only under-claim.
    os.replace(tmp, out_path)
    print(f"extracted {out_path} ({os.path.getsize(out_path):,} bytes, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return out_path


def open_x(path):
    """(X memmap, metadata mapping or None) from an .npy OR a tensor .npz.

    Three ways in, in the order that costs least:

      1. `--x foo_X.npy` — the extracted memmap. Unchanged, and still what the
         family-3 invocations in this docstring pass.
      2. `--x foo.npz` with `foo_X.npy` beside it — `ml/tensor_io.py`'s sidecar
         layout (family 5's, and what `extract_x` writes). Memmapped, free.
      3. `--x foo.npz` with X stored UNCOMPRESSED inside — memmapped in place
         at its zip offset (`npz_member_offset`). No copy.

    A compressed member (family 4, `np.savez_compressed`) is REFUSED with the
    exact `--extract-x` command, rather than silently decompressing 33.9 GB
    into RAM on a box that may not have it. That refusal is the whole point:
    `np.load` on such a member SUCCEEDS on a big box and OOM-kills a small
    one, and neither outcome tells the reader that the audit's memory profile
    is a function of the tensor's format.
    """
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r"), None
    if not path.endswith(".npz"):
        raise SystemExit(f"--x {path}: expected a .npy memmap or a tensor .npz")
    meta = np.load(path, allow_pickle=False)
    side = sidecar_path(path)
    if os.path.exists(side):
        print(f"X from sidecar {os.path.basename(side)} (memmapped)",
              flush=True)
        return np.load(side, mmap_mode="r"), meta
    off = npz_member_offset(path)
    if off is not None:
        print(f"X stored uncompressed inside {os.path.basename(path)} — "
              f"memmapped in place at byte {off:,}", flush=True)
        return _map_npy_at(path, off), meta
    with zipfile.ZipFile(path) as z:
        try:
            n = z.getinfo("X.npy").file_size
        except KeyError:
            raise SystemExit(f"{path} has no `X` member: {z.namelist()}")
    raise SystemExit(
        f"{os.path.basename(path)} stores X DEFLATE-COMPRESSED inside the "
        f"npz ({n / 1e9:.1f} GB uncompressed), which np.load cannot memory-map "
        f"— it would decompress the whole tensor into RAM. Extract it once:\n"
        f"  python3 ml/recon_eval.py --extract-x {path}\n"
        f"which writes {os.path.basename(sidecar_path(path))} beside it "
        f"(needs {n / 1e9:.1f} GB + 2 GB free) and is then found automatically "
        f"by this flag and by ml/tensor_io.load_tensor.")


# ---------------------------------------------------------------------------
# Streaming replica of temporal.py's anomaly transform (lines ~745-762).
# The original loads X whole (10.9 GB for family3) and standardizes in RAM;
# this computes the same clim / mean / std with chunked passes over a
# memmap, then materialises only a [T, 3, W, C] latitude slab.
# ---------------------------------------------------------------------------

def check_stats(chan, dyn, mean_c, std_c, obs_n, bad_mean_n=None,
                where="stream_stats"):
    """REFUSE if a channel with observed values has non-finite statistics.

    The E-049 blocker was not that the numbers were wrong — it was that
    nothing said so (ml/CLAUDE.md §0.2, and §5.22 "never write NaN into a
    results file: stop instead").

    THE GUARD HAS TO WATCH THE PER-BIN SPATIAL MEAN, not only the final
    mean_c/std_c, and the first draft of it did not — which is why this
    docstring says so. Follow the overflow through: `sp_mean[t, c]` goes inf,
    `np.nanstd(sp_mean[:, c])` goes nan, `nan > 1e-6` is False, and the
    channel leaves `dyn` — at which point `mean_c[c]` is 0.0 and `std_c[c]`
    is 1.0, both perfectly FINITE, because those are the pass-through values
    a genuinely static channel gets. A guard that looked only at the outputs
    would have passed the exact tensor the fix exists for. `bad_mean_n[c]`
    counts the bins where the channel HAD observations and its spatial mean
    came out non-finite, which is the signature itself.

    What this can and cannot see. It catches the OVERFLOW mode (sum -> inf).
    It cannot catch the SATURATION mode — a float16 sum near 32,768, where
    the representable spacing is 32, rounds every bin to the same value and
    the series is exactly constant, which is indistinguishable from a
    genuinely static channel by any finiteness test. That mode is closed by
    `acc_dtype` rather than by a guard, and `tests/test_e049_recon_audit.py`
    check 1 measures both.

    A channel with NO observed value legitimately has no statistics (Argo
    before 2004 on a short slice), and a genuinely STATIC channel legitimately
    sits outside `dyn` with mean 0 / std 1 — both pass. Split out of
    `stream_stats` so `recon_decoder.py`'s cached-stats path is guarded by the
    same check.
    """
    bad = []
    for c in range(len(std_c)):
        if obs_n[c] <= 0:
            continue
        if bad_mean_n is not None and bad_mean_n[c] > 0:
            bad.append((c, f"{int(bad_mean_n[c])} observed bins whose SPATIAL "
                           f"MEAN is non-finite; in `dyn`={c in dyn}"))
        elif c in dyn:
            if not (np.isfinite(mean_c[c]) and np.isfinite(std_c[c])
                    and std_c[c] > 0):
                bad.append((c, f"dynamic, mean={mean_c[c]} std={std_c[c]}"))
        elif not (np.isfinite(mean_c[c]) and np.isfinite(std_c[c])):
            bad.append((c, f"static, mean={mean_c[c]} std={std_c[c]}"))
    if bad:
        names = "; ".join(f"{c} ({chan[c] if chan is not None else '?'}): {why}"
                          for c, why in bad)
        raise SystemExit(
            f"{where}: channel(s) with observed values produced NON-FINITE "
            f"statistics — {names}. A channel in this state is classified NOT "
            f"DYNAMIC and passed to the codec UN-STANDARDIZED, which produces "
            f"plausible garbage and no error. Refusing rather than scoring it. "
            f"(The known cause is a float16 spatial sum overflowing 65,504; "
            f"see `acc_dtype` above. If the tensor is float32 this is new.)")


def stream_stats(Xm, moy, t_hold, x_hold, chunk=8, chan=None):
    """(clim [12,H,W,C], dynamic [c...], mean_c, std_c) — chunked over T.

    TWO E-049 FIXES LIVE HERE, and they are the reason the pentad audit could
    not simply be pointed at family 4.

    **THE ACCUMULATOR.** `xb0.sum(axis=(1, 2))` on a float16 memmap reduces IN
    FLOAT16: numpy's `np.sum` keeps the input dtype (only `np.mean` upcasts,
    and `np.std`/`np.var` do not — the same asymmetry `ml/trainprobe.py`'s
    `anomaly_transform` documents at length). One spatial slice at family-4
    shape is H*W = 135,161 values; `cur_speed`-scale data is fine, but a
    channel whose mean is ~20 sums to ~2.7e6 against float16's 65,504 ceiling.
    Measured on numpy 2.4.4: `np.full((281,481), 20.0, np.float16).sum()` is
    `inf` (with a RuntimeWarning nobody reads inside a chunk loop), so
    `sp_mean` is nan, `np.nanstd(sp_mean[:, c])` is nan, `nan > 1e-6` is
    False, and the channel is silently declared static. `acc_dtype` names
    float64 for float16 storage and `None` — numpy's default, i.e. no change
    at all — for float32, so every family-3 number this script has produced is
    reproduced bit-for-bit.

    **THE COUNTER.** `n` was uint8. It counts, per month-of-year and per
    (H, W, C) cell, how many TRAIN timesteps observed that cell: <= 43 at
    family 3's monthly cadence (43 years), and ~244 of 255 at pentad cadence
    (43 years x ~6 bins whose start-date falls in one month). One cadence step
    finer, or a longer record, and it wraps — 260 observations counted as 4,
    a climatology 65x too large, no warning. uint16 costs 2 bytes per cell
    (0.13 GB more at family-4 shape, against the 1.3 GB the float32 sum
    already holds) and moves the ceiling to 65,535, which the refusal below
    checks against T rather than trusting.
    """
    T, H, W, C = Xm.shape
    if T > np.iinfo(np.uint16).max:
        raise SystemExit(
            f"stream_stats: T = {T:,} exceeds the climatology counter's "
            f"uint16 ceiling ({np.iinfo(np.uint16).max:,}). A per-(moy, cell) "
            f"count cannot exceed T, so up to that ceiling the counter cannot "
            f"wrap; beyond it, it would wrap SILENTLY and divide the sum by "
            f"the wrong n. Widen `n` to uint32 before auditing this tensor.")
    acc = acc_dtype(Xm.dtype)                      # float64 iff float16 store
    s = np.zeros((12, H, W, C), np.float32)
    n = np.zeros((12, H, W, C), np.uint16)         # ~244 of 65,535 at pentad
    sp_sum = np.zeros((T, C), np.float64)          # spatial nansum per bin
    sp_cnt = np.zeros((T, C), np.int64)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        xb = np.asarray(Xm[t0:t1])                 # [c,H,W,C]
        fin = np.isfinite(xb)
        xb0 = np.where(fin, xb, 0.0)
        sp_sum[t0:t1] = xb0.sum(axis=(1, 2), dtype=acc)
        sp_cnt[t0:t1] = fin.sum(axis=(1, 2))
        for i, t in enumerate(range(t0, t1)):
            if t_hold[t]:
                continue                           # clim from train years only
            m = moy[t]
            s[m] += xb0[i]
            n[m] += fin[i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clim = s / n                               # nan where never observed
        sp_mean = np.where(sp_cnt > 0, sp_sum / sp_cnt, np.nan)
    # dynamic test, identical to temporal.py: nanstd of the per-bin spatial
    # nanmean series (all bins) > 1e-6
    dyn = [c for c in range(C)
           if np.nanstd(sp_mean[:, c]) > 1e-6]
    # pass 2: per-channel mean/std of the anomaly over train months and
    # non-holdout longitudes (temporal.py's `v`)
    acc2 = np.zeros((C, 3), np.float64)            # sum, sumsq, count
    keep_x = ~x_hold                               # [W]
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        sel = [t for t in range(t0, t1) if not t_hold[t]]
        if not sel:
            continue
        xb = np.asarray(Xm[sel])                   # [s,H,W,C]
        for c in dyn:
            d = xb[..., c] - clim[moy[sel], :, :, c]
            d = d[:, :, keep_x]
            f = np.isfinite(d)
            acc2[c, 0] += d[f].sum(dtype=np.float64)
            acc2[c, 1] += (d[f].astype(np.float64) ** 2).sum()
            acc2[c, 2] += f.sum()
    mean_c = np.zeros(C, np.float32)
    std_c = np.ones(C, np.float32)
    for c in dyn:
        cnt = acc2[c, 2]
        mu = acc2[c, 0] / cnt
        var = acc2[c, 1] / cnt - mu * mu
        mean_c[c], std_c[c] = mu, np.sqrt(max(var, 0.0))
    # THE GUARD, at the point where the inputs are all it has cost (§0.3).
    # `sp_cnt.sum(0)` is "how many observed values this channel has anywhere",
    # which is exactly the population the check is about; `bad_mean` counts
    # the bins where a channel HAD observations and its spatial mean came out
    # non-finite — the overflow's own signature, and the only place it is
    # still visible by the time `dyn` has been decided.
    bad_mean = ((sp_cnt > 0) & ~np.isfinite(sp_mean)).sum(axis=0)
    check_stats(chan, dyn, mean_c, std_c, sp_cnt.sum(axis=0), bad_mean)
    return clim, dyn, mean_c, std_c


def build_slab(Xm, rows, moy, clim, dyn, mean_c, std_c):
    """Standardized anomaly slab [T, len(rows), W, C] + its finite mask."""
    T, H, W, C = Xm.shape
    slab = np.asarray(Xm[:, rows]).astype(np.float32)      # [T,3,W,C]
    obs = np.isfinite(slab)
    # index the slab rows FIRST: clim[moy] would materialise [T,H,W,C]
    # (10.9 GB) — the exact fancy-index trap temporal.py documents.
    clim_slab = clim[:, rows]                              # [12,3,W,C]
    for c in dyn:
        slab[..., c] = ((slab[..., c] - clim_slab[moy][..., c]
                         - mean_c[c]) / (std_c[c] + 1e-6))
    slab[~obs] = np.nan
    return slab, obs


def verify_streaming(Xm, moy, t_hold, x_hold, clim, dyn, mean_c, std_c,
                     rows, slab, channels):
    """Replay temporal.py's exact in-RAM recipe on full single channels and
    assert the slab matches. A preprocessing mismatch here would silently
    invalidate every downstream number — this is the audit auditing itself.

    E-049: the reference's own reductions name float64 for a float16 tensor,
    for the same reason `stream_stats` does. This path upcasts to float32
    first, so it cannot overflow the way the streaming sum did — but `v.mean()`
    and `v.std()` here reduce over the WHOLE finite pool (424M values at
    family-4 shape), and float32 is not the dtype to settle a reference in.
    `acc_dtype` returns None on a float32 tensor, so the family-3 reference is
    unchanged.
    """
    acc = acc_dtype(Xm.dtype)
    for c in channels:
        xc = np.asarray(Xm[..., c]).astype(np.float32)     # [T,H,W] ~279 MB
        cl = np.full((12,) + xc.shape[1:], np.nan, np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for m in range(12):
                cl[m] = np.nanmean(xc[(moy == m) & ~t_hold], axis=0, dtype=acc)
        if c in dyn:
            xc = xc - cl[moy]
            v = xc[np.isfinite(xc) & ~t_hold[:, None, None]
                   & ~x_hold[None, None, :]]
            xc = (xc - v.mean(dtype=acc)) / (v.std(dtype=acc) + 1e-6)
        ref = xc[:, rows]
        got = slab[..., c]
        f = np.isfinite(ref)
        assert (np.isfinite(got) == f).all(), f"chan {c}: finite mask differs"
        err = np.abs(ref[f] - got[f]).max() if f.any() else 0.0
        assert err < 5e-4, f"chan {c}: streaming vs in-RAM recipe |Δ|={err}"
        print(f"  verify chan {c}: max |Δ| = {err:.2e} ✓", flush=True)


# ---------------------------------------------------------------------------


def score(truth, pred, obs, sel_t, sel_x, label):
    """Per-channel r / RMSE over (t in sel_t) × (px in sel_x), obs only."""
    C = truth.shape[-1]
    out = {}
    tt = truth[np.ix_(sel_t, sel_x)]     # [t,p,C]
    pp = pred[np.ix_(sel_t, sel_x)]
    oo = obs[np.ix_(sel_t, sel_x)]
    for c in range(C):
        m = oo[..., c]
        if m.sum() < 30:
            continue
        a, b = tt[..., c][m], pp[..., c][m]
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 else float("nan")
        out[c] = {"r": round(r, 4),
                  "rmse": round(float(np.sqrt(((a - b) ** 2).mean())), 4),
                  "n": int(m.sum())}
    return out


def argo_bin_mask(obs, argo):
    """[t, px] bool: is ANY Argo channel observed at this (bin, pixel)?

    THE FALSIFIER'S OWN AXIS (E-049 §1). Road B's hypothesis is about the 8
    fast channels on the 92% of bins that carry NO Argo profile; the 8% that
    do are expected to reconstruct badly at d_z 6 and "this experiment
    measures the damage rather than hiding it". Today's scorer can only
    express month x longitude products, so a bin-level split had to exist
    before the audit could answer the question that was registered.

    Why ANY rather than ALL: Roemmich-Gilson enters the pentad tensor as ONE
    live bin per month (`ml/build_family4.py`: "n_rg_live 252/3142, the
    mid-month stamp"), and when it is live it stamps the whole 32-channel
    block at once. `any` and `all` therefore agree on the real tensor almost
    everywhere; `any` is the conservative choice, because a bin holding even
    one profile channel is a bin whose 16 bits had to carry Argo.
    """
    return obs[..., argo].any(axis=-1)


def score_bins(truth, pred, obs, sel_t, sel_x, bins, min_n=30):
    """Per-channel r / RMSE / FVU over (sel_t x sel_x), restricted to `bins`.

    ADDITIVE, NEVER A REPLACEMENT. `score()` above keeps its exact output
    because 183 archived monthly bundles and every reference row in the E-049
    plan's §2 table read it; this writes NEW keys beside it. Same rule
    ml/CLAUDE.md §3 sets for the pooled transport read-out: "an unpooled
    transport read-out must be an ADDITIONAL function writing NEW keys beside
    `amoc_bands`, never a change to that one."

    `bins` is [T, P] bool over the FULL axes (it is indexed with the same
    `np.ix_` the values are), so one mask serves every split.

    TWO FVUs, because "FVU" is used for two things in this programme:
      · `fvu` = mean squared error IN STANDARDIZED UNITS = `rmse ** 2`. This
        is E-049 §4a's own definition ("FVU = rmse^2 in standardized units")
        and it is what makes a number here comparable to run-415's 0.4-0.6%
        and to E-047 Tier-1's 9-19% band. Its denominator is the channel's
        GLOBAL train variance, which is 1 by construction of the transform.
      · `fvu_local` = mse / var(truth over THIS selection). On a sub-population
        whose variance is not 1 — Argo-free bins are a different population
        from all bins — this is the honest "would a constant do as well?"
        reading, and it is the one that answers "at or near 100%". Reported
        beside, never instead: the two disagree exactly when the split has
        changed the variance, which is information.
    """
    C = truth.shape[-1]
    out = {}
    tt = truth[np.ix_(sel_t, sel_x)]
    pp = pred[np.ix_(sel_t, sel_x)]
    oo = obs[np.ix_(sel_t, sel_x)]
    bb = bins[np.ix_(sel_t, sel_x)]
    for c in range(C):
        m = oo[..., c] & bb
        if m.sum() < min_n:
            continue
        a, b = tt[..., c][m].astype(np.float64), pp[..., c][m].astype(np.float64)
        mse = float(((a - b) ** 2).mean())
        va = float(a.var())
        out[c] = {
            "r": (round(float(np.corrcoef(a, b)[0, 1]), 4)
                  if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")),
            "rmse": round(float(np.sqrt(mse)), 4),
            "fvu": round(mse, 5),
            "fvu_local": round(mse / va, 5) if va > 1e-12 else float("nan"),
            "n": int(m.sum()),
        }
    return out


def argo_split_block(truth, pred, obs, chan, splits):
    """The whole Argo-bin section of the JSON, for either script.

    `splits` is {name: (sel_t, sel_x)}. Returns argo / argo-free scores for
    each, plus the channel partition and the bin census that says how big each
    population actually was — a per-channel FVU over 200 bins is a different
    object from one over 20,000, and the census is what lets a reader tell.
    """
    argo, fast = argo_channels(chan)
    bins = argo_bin_mask(obs, argo)
    block = {
        "doc": "per-(bin, pixel) split: a bin is Argo-carrying iff ANY "
               "rg_t*/rg_s* channel is observed there. fvu = mse in "
               "standardized units (= rmse^2, E-049 §4a); fvu_local = mse / "
               "var(truth) over this selection.",
        "argo_channels": argo,
        "argo_channel_names": [str(chan[c]) for c in argo],
        "fast_channels": fast,
        "fast_channel_names": [str(chan[c]) for c in fast],
        "census": {},
        "argo_bins": {},
        "argo_free_bins": {},
    }
    for name, (st, sx) in splits.items():
        sub = bins[np.ix_(st, sx)]
        tot = int(sub.size)
        block["census"][name] = {
            "bins": tot, "argo": int(sub.sum()),
            "argo_free": int(tot - sub.sum()),
            "argo_fraction": round(float(sub.mean()), 5) if tot else float("nan"),
        }
        block["argo_bins"][name] = score_bins(truth, pred, obs, st, sx, bins)
        block["argo_free_bins"][name] = score_bins(truth, pred, obs, st, sx,
                                                   ~bins)
    return block


def score_pooled(truth, pred, obs, sel_t, sel_x):
    """Section-mean recon vs section-mean truth, per channel, over months."""
    C = truth.shape[-1]
    out = {}
    for c in range(C):
        tt = truth[:, sel_x, c]
        pp = pred[:, sel_x, c]
        oo = obs[:, sel_x, c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tm = np.where(oo, tt, np.nan)
            pm = np.where(oo, pp, np.nan)
            a = np.nanmean(tm[sel_t], axis=1)
            b = np.nanmean(pm[sel_t], axis=1)
        f = np.isfinite(a) & np.isfinite(b)
        if f.sum() < 24 or a[f].std() < 1e-9:
            continue
        out[c] = {"r": round(float(np.corrcoef(a[f], b[f])[0, 1]), 4),
                  "rmse": round(float(np.sqrt(((a[f] - b[f]) ** 2).mean())), 4),
                  "n": int(f.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", help="extracted X.npy (memmapped) OR the tensor "
                                ".npz (sidecar / uncompressed member)")
    ap.add_argument("--npz", help="tensor npz for small members; defaults to "
                                  "--x when that is itself an npz")
    ap.add_argument("--ckpt", help="codec checkpoint")
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--extract-x", metavar="TENSOR.npz",
                    help="one-time: write TENSOR's compressed X member out as "
                         "a memmappable <stem>_X.npy beside it, then exit. "
                         "See `extract_x` for the disk cost.")
    ap.add_argument("--extract-out", help="where --extract-x writes "
                                          "(default: tensor_io's sidecar name)")
    a = ap.parse_args()

    if a.extract_x:
        extract_x(a.extract_x, a.extract_out or sidecar_path(a.extract_x))
        return
    for req in ("x", "ckpt", "out"):
        if not getattr(a, req):
            ap.error(f"--{req} is required (unless --extract-x)")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    Xm, meta = open_x(a.x)
    if a.npz:
        d = np.load(a.npz, allow_pickle=False)
    elif meta is not None:
        d = meta
    else:
        ap.error("--npz is required when --x is a bare .npy")
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    chan = [str(c) for c in d["chan"]]
    T, H, W, C = Xm.shape
    assert len(chan) == C

    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    bspec = bottleneck_spec(ck)
    print(f"tensor [T={T} H={H} W={W} C={C}] {Xm.dtype} · holdout years "
          f"{sorted(hold_years)} · holdout lons [{lo},{hi})", flush=True)
    print(bottleneck_line(bspec), flush=True)

    t0 = time.time()
    clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold, chan=chan)
    print(f"streaming stats: {len(dyn)}/{C} dynamic channels "
          f"({time.time()-t0:.0f}s)", flush=True)

    sec_y = int(np.argmin(np.abs(lats - RAPID_LAT)))
    rows = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs = build_slab(Xm, rows, moy, clim, dyn, mean_c, std_c)

    if not a.skip_verify:
        # one dynamic channel (0 = cur_speed) and one deep RG channel
        deep = next((c for c, nm in enumerate(chan) if nm.startswith("rg_t")
                     and c in dyn), dyn[-1])
        verify_streaming(Xm, moy, t_hold, x_hold, clim, dyn, mean_c, std_c,
                         rows, slab, [0, deep])

    # section pixels exactly as the probe ladder selects them: ocean = channel
    # 0 observed at ANY month (temporal.py's mask), then section_of over the
    # slab's 3-row coordinate frame (its center row is nearest 26.5N, so
    # section_of resolves sec_y == 1 by the same argmin it runs in production)
    ocean_row = obs[:, 1, :, 0].any(axis=0)        # slab row 1 = sec_y
    xs_all = np.where(ocean_row)[0]
    _, sel = section_of(lats[rows], lons, np.ones(len(xs_all), dtype=int),
                        xs_all, RAPID_LAT, *RAPID_LON)
    xs_sec = xs_all[sel]
    print(f"section: {len(xs_sec)} pixels at lat {lats[sec_y]:.2f}", flush=True)

    # ---- encode through the production path -------------------------------
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval()
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    Xt = torch.from_numpy(np.nan_to_num(slab, nan=0.0))
    OBS = torch.from_numpy(obs)
    t0 = time.time()
    Z, _ = embed_everything(codec, Xt, OBS, ctx_all, lats[rows], lons,
                            np.full(len(xs_sec), 1), xs_sec, ck["d_z"],
                            cache_path=None, batch=a.batch)
    print(f"encoded [T={T} P={len(xs_sec)}] in {time.time()-t0:.0f}s", flush=True)

    # ---- decode every channel at offset 0 ---------------------------------
    P = len(xs_sec)
    pred = np.zeros((T, P, C), np.float32)
    qc = torch.arange(C)[None, :]
    off0 = torch.zeros(1, C, 3, dtype=torch.long)
    with torch.no_grad():
        for t in range(T):
            zb = torch.from_numpy(Z[t]).float()
            pred[t] = codec.query(zb, qc.expand(P, -1),
                                  off0.expand(P, -1, -1)).numpy()

    truth = slab[:, 1][:, xs_sec]                  # [T,P,C] center row
    obs_c = obs[:, 1][:, xs_sec]
    truth0 = np.nan_to_num(truth, nan=0.0)

    px_hold = x_hold[xs_sec]
    sel_train_t = np.where(~t_hold)[0]
    sel_hold_t = np.where(t_hold)[0]
    sel_train_x = np.where(~px_hold)[0]
    sel_hold_x = np.where(px_hold)[0]

    res = {
        "doc": "copy-reconstruction audit: full-visibility encode -> decode "
               "all channels at offset 0, vs standardized truth; std units "
               "(rmse^2 ~ fraction of channel variance lost)",
        "ckpt": os.path.basename(a.ckpt),
        "d_z": int(ck["d_z"]), "C": C, "T": T, "P": P,
        "bottleneck": bspec, "x_dtype": str(Xm.dtype),
        "chan": chan, "dynamic": dyn,
        "splits": {
            "train": score(truth0, pred, obs_c, sel_train_t, sel_train_x, "train"),
            "heldout_months": score(truth0, pred, obs_c, sel_hold_t,
                                    sel_train_x, "hold-t"),
            "heldout_lons": score(truth0, pred, obs_c, np.arange(T),
                                  sel_hold_x, "hold-x"),
        },
        "pooled": {
            "train": score_pooled(truth0, pred, obs_c, sel_train_t,
                                  np.arange(P)),
            "heldout_months": score_pooled(truth0, pred, obs_c, sel_hold_t,
                                           np.arange(P)),
        },
    }
    # E-049's own axis, ADDITIVE: `splits`/`pooled` above are byte-for-byte
    # what this script has always written.
    res["argo_split"] = argo_split_block(
        truth0, pred, obs_c, chan,
        {"train": (sel_train_t, sel_train_x),
         "heldout_months": (sel_hold_t, sel_train_x),
         "heldout_lons": (np.arange(T), sel_hold_x)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    # readable summary
    tr = res["splits"]["train"]
    hm = res["splits"]["heldout_months"]
    hx = res["splits"]["heldout_lons"]
    rs = [v["r"] for v in tr.values() if np.isfinite(v["r"])]
    print(f"\n== copy reconstruction, {os.path.basename(a.ckpt)} ==")
    print(bottleneck_line(bspec))
    print(f"mean r over {len(rs)} scoreable channels (train): "
          f"{np.mean(rs) if rs else float('nan'):.3f}  · worst 5:")
    for c, v in sorted(tr.items(), key=lambda kv: kv[1]["r"])[:5]:
        print(f"  {chan[c]:<12} r={v['r']:.3f} rmse={v['rmse']:.3f}")
    print(f"{'chan':<12} {'train r':>8} {'holdT r':>8} {'holdX r':>8} "
          f"{'train rmse':>10}")
    for c in range(C):
        if c not in tr:
            continue
        print(f"{chan[c]:<12} {tr[c]['r']:>8.3f} "
              f"{hm.get(c, {}).get('r', float('nan')):>8.3f} "
              f"{hx.get(c, {}).get('r', float('nan')):>8.3f} "
              f"{tr[c]['rmse']:>10.3f}")
    print_argo_summary(res["argo_split"], chan)
    print(f"\nwrote {a.out}")


def print_argo_summary(blk, chan, split="heldout_months"):
    """The falsifier's own table: FAST channels, Argo-FREE bins, held-out.

    Printed at the split E-049 §1 registers, with the Argo-carrying column
    beside it — the plan says the Argo bins "will reconstruct badly at d_z 6"
    and that the experiment "measures the damage rather than hiding it", so
    the damage is printed rather than filed.
    """
    cen = blk["census"].get(split, {})
    free = blk["argo_free_bins"].get(split, {})
    carry = blk["argo_bins"].get(split, {})
    print(f"\n-- Argo-bin split ({split}) -- {cen.get('argo_free', 0):,} "
          f"Argo-free of {cen.get('bins', 0):,} bin-pixels "
          f"({1 - cen.get('argo_fraction', float('nan')):.1%} free)")
    print(f"{'fast chan':<12} {'free fvu':>9} {'free fvuL':>10} "
          f"{'free r':>7} {'argo fvu':>9} {'argo r':>7}")
    for c in blk["fast_channels"]:
        f, g = free.get(c), carry.get(c)
        if f is None and g is None:
            continue
        print(f"{str(chan[c]):<12} "
              f"{(f or {}).get('fvu', float('nan')):>9.4f} "
              f"{(f or {}).get('fvu_local', float('nan')):>10.4f} "
              f"{(f or {}).get('r', float('nan')):>7.3f} "
              f"{(g or {}).get('fvu', float('nan')):>9.4f} "
              f"{(g or {}).get('r', float('nan')):>7.3f}")
    fv = [free[c]["fvu_local"] for c in blk["fast_channels"]
          if c in free and np.isfinite(free[c]["fvu_local"])]
    if fv:
        print(f"E-049 falsifier reading: worst fast-channel fvu_local on "
              f"Argo-free bins = {max(fv):.4f} (at or near 1.0 = a constant "
              f"would do as well)")


if __name__ == "__main__":
    main()
