#!/usr/bin/env python3
"""Stage 2: a temporal transformer over frozen codec embeddings.

The K-sweep (probe_sequence.py, protocol v2) established the precondition:
ANOMALY-space embeddings gain probe skill as history is concatenated
(state-space embeddings lost it — they were seasonally redundant). A linear
read-out of stacked embeddings is the crudest possible sequence model; this
file is the honest next rung — a small causal transformer over each pixel's
embedding sequence z_{t-K+1..t}, with two jobs:

  1. DYNAMICS: predict z_{t+1} (the next month's anomaly embedding).
     Channel-space score: decode ẑ_{t+1} through the FROZEN codec decoder
     at offset 0 and compare against the true next-month channels, vs the
     persistence forecast x_{t+1} := x_t. Same blocked holdout as training.
  2. STATE: the transformer's last hidden state at the 26.5°N section,
     pooled along the section, replaces the concatenated-z features in the
     RAPID probe — same seasonality-proof protocol (deseasonalised target,
     train-years climatology, seasonal-only floor, lambda on a train tail).

Both stages stay in ANOMALY space; the codec is never fine-tuned (two-stage
by construction, so codec improvements and dynamics improvements stay
attributable). Splits are inherited from the codec checkpoint — the same
held-out years and the same mid-Atlantic longitude block, never random.

Usage:  python3 ml/temporal.py --run pilot4_anom --steps 4000
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE, codec_from_ckpt
from probe_sequence import ridge_r

HERE = os.path.dirname(os.path.abspath(__file__))
# Box-persistent mirror, same directory train.py uses for codecs.
# Box-persistent mirror, same directory train.py uses for codecs. The
# override exists so tests/test_resume_temporal.py and the toy end-to-end run
# can exercise the real save/resume path without a Vast box and without
# writing anywhere real.
CKPT_DIR = os.environ.get("CKPT_DIR_OVERRIDE", "/opt/earth-cache/ckpt")


class TemporalTransformer(nn.Module):
    """Causal transformer over one pixel's embedding sequence.

    Inputs are codec embeddings z_t (d_z) plus a per-pixel static context
    (lat, lon, and the codec embedding of the pixel's STATIC channels alone —
    the climatological identity of the place), added to every step. The
    month-of-year enters as sin/cos per STEP: dynamics may be phase-dependent
    (winter mixing vs summer stratification) even when the state is an
    anomaly. Output head predicts z_{t+1} from the hidden state at t.
    """

    def __init__(self, d_z=32, d_model=96, n_heads=4, n_layers=3, k_max=36):
        super().__init__()
        self.inp = nn.Linear(d_z + 2, d_model)     # z_t + (sin m, cos m)
        self.static = nn.Linear(d_z + 2, d_model)  # static-z + (lat, lon)
        self.pos = nn.Embedding(k_max, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, d_z)
        self.d_model = d_model

    def forward(self, z_seq, month_seq, static_ctx):
        """z_seq [B,K,d_z] · month_seq [B,K,2] · static_ctx [B,d_z+2]
        → pred [B,K,d_z] (ẑ at t+1 for every step), h [B,K,d_model]."""
        B, K, _ = z_seq.shape
        h = (self.inp(torch.cat([z_seq, month_seq], -1))
             + self.static(static_ctx).unsqueeze(1)
             + self.pos.weight[None, :K])
        causal = nn.Transformer.generate_square_subsequent_mask(K, device=z_seq.device)
        h = self.encoder(h, mask=causal, is_causal=True)
        return self.head(h), h


# Protocol v3 (2026-08-07, the global window): the RAPID probe section is
# the grid row nearest 26.5°N clipped to the array's Atlantic span (Abaco
# to the African shelf). On the NA pilot window protocol v2 used the whole
# row; the clip drops its Gulf-of-Mexico and NW-African cells and is
# REQUIRED on the global window, where the unclipped row would circle the
# planet through the Pacific and drown the section pool.
RAPID_LON = (-80.0, -13.0)


def section_of(lats, lons, ys, xs, lat, lon_lo, lon_hi):
    """(sec_y, indices into ys/xs) of a zonal probe section: the grid row
    nearest `lat`, clipped to [lon_lo, lon_hi]. Every transport array gets
    its own section this way (probe_kfold.TARGETS)."""
    sec_y = int(np.argmin(np.abs(lats - lat)))
    sel = np.where((ys == sec_y) & (lons[xs] >= lon_lo)
                   & (lons[xs] <= lon_hi))[0]
    return sec_y, sel


def rapid_section(lats, lons, ys, xs):
    """(sec_y, indices into ys/xs) of the protocol-v3 RAPID section."""
    return section_of(lats, lons, ys, xs, 26.5, RAPID_LON[0], RAPID_LON[1])


RESERVE_BYTES = 3 << 30      # runner logs, checkpoints, pip, room to breathe
RAM_HEADROOM_BYTES = 8 << 30  # the tensor and mask are already resident

# THE CACHE IS float16; THE ARITHMETIC IS NOT. At float32 the quarter-degree
# embedding is 10.4 GiB (516 x 84,405 x 64 x 4), which does not fit beside a
# ~15 GB torch image and ~11 GB of tensors on a 50 GB box — so the hygiene
# step deleted it, the next run spent 95 minutes rebuilding it, and the
# rebuild put the box back under the threshold. At float16 it is 5.2 GiB and
# fits with room to spare, which ends that treadmill on the hardware we
# actually rent. Vast will not resize a disk (see CLAUDE.md Part 2), so making
# the artefact smaller was the available lever.
#
# The precision cost is measured, not assumed: on unit-scale embeddings the
# round trip through float16 introduces an MSE of 4.3e-8, which is ~1e-7 of
# the z-MSE the experiments report (0.39-0.82). The figure we actually argue
# from is the model/persistence RATIO, where the error is common-mode across
# numerator and denominator and shifts the ratio by 1.8e-7. Seven orders of
# magnitude below the effect being measured.
#
# Everything downstream casts to float32 at the point of use, so gradients,
# optimiser state and the loss are unchanged. An existing float32 cache still
# loads and still works — `.float()` on it is a no-op — so this is not a
# flag day.
CACHE_DTYPE = np.float16


def make_sched(opt, a, last_epoch=-1):
    """The LR schedule, with a HORIZON-FREE option.

    Chris, 2026-08-10: *"let's not 'bake' num steps into the LR. Maybe we can
    use some LR decay (in the future) that does not depend on the number of
    total steps."*

    He is pointing at the root of two separate problems, not one.

    The BUG: `CosineAnnealingLR(T_max=steps)` makes the rate a function of the
    total, so a checkpoint's schedule is only meaningful alongside the budget
    it was trained under. Reload it while asking for a larger total and it
    believes it has finished — lr = 0.0, sixteen hours of updating nothing,
    every status reading success. That cost a run on 2026-08-10 and needed a
    rebuild-the-schedule special case, a refusal guard and a test to contain.

    The deeper COMPARABILITY problem: with a horizon-baked schedule, a
    6,000-step run and a 200,000-step run are at different learning rates at
    every shared step, so they are two different experiments that happen to
    share an architecture. E-007's three points each had to be described as
    "its own converged cosine", and the 200k point could not be a continuation
    of the 60k one — which is the whole reason E-008 became a warm restart.

    `invsqrt` (the Noam schedule) removes both at once: lr(s) depends only on
    s, so a run stopped at 60,000 and continued to 200,000 sees exactly the
    rate an uninterrupted 200,000-step run would have seen at those steps.
    Resume needs no special case, checkpoints are interchangeable, and two
    budgets become a prefix and its extension rather than two experiments.

    The price, stated honestly: cosine anneals to zero and therefore CONVERGES
    at a known point, which is what makes "the 60k result" a settled number.
    invsqrt never reaches zero, so a run has no natural end and results are
    "at step N" rather than "converged". For a programme asking "does more
    compute help?" that is the better trade — the question presumes an
    open-ended curve — but it is a trade, not a free win, and switching should
    be a deliberate experiment (one budget, both schedules) rather than a
    default flipped in passing.
    """
    if a.lr_schedule in ("invsqrt", "wsd", "expdecay"):
        warm = max(1, int(a.lr_warmup))

        def _warm_cos(s):
            """Cosine-shaped ramp to the peak — smooth at BOTH ends, unlike a
            linear ramp which arrives at the peak with a corner."""
            import math as _m
            return 0.5 * (1 - _m.cos(_m.pi * min(1.0, s / warm)))

        if a.lr_schedule == "wsd":
            # WARMUP - STABLE - DECAY. The literature's current answer, and a
            # better fit for this programme than either cosine or pure
            # inverse-sqrt: the stable phase is horizon-free, so a run can be
            # extended and its checkpoints are interchangeable, while the
            # cooldown recovers a genuine CONVERGED endpoint — which is what
            # invsqrt gives up and what makes "the 60k result" a settled
            # number rather than a reading at step 60,000.
            #
            # The consequence for this programme is concrete: E-007's four
            # budgets could be ONE run with four short cooldowns branched off
            # the stable phase, instead of four experiments that cannot be
            # compared as a trajectory.
            cool = max(1, int(round(a.steps * a.lr_cooldown_frac)))
            stable_end = max(warm, a.steps - cool)

            def factor(step):
                s = step + 1
                if s <= warm:
                    return s / warm
                if s <= stable_end:
                    return 1.0
                # Linear to zero: convex theory puts the optimal cooldown
                # shape at linear, and D2Z finds decaying fully to zero beats
                # stopping at a floor, increasingly so the longer you train.
                return max(0.0, (a.steps - s) / max(1, a.steps - stable_end))
        elif a.lr_schedule == "expdecay":
            # COSINE WARMUP, THEN EXPONENTIAL DECAY. Chris, looking at the WSD
            # trapezoid on the status page: "123's learning schedule doesn't
            # look great (too constant, then too steep). if nothing better use
            # cosine warmup and then exp decay."
            #
            # He is right on both counts and the second one is not only
            # aesthetic. WSD's cooldown is sized as a FRACTION of the total, so
            # the schedule is horizon-free right up until the part that is not
            # — extend the run and the cooldown moves, which is the same
            # coupling cosine has, merely postponed. Exponential decay with an
            # ABSOLUTE half-life has no such term: lr(s) = peak * 2^(-s/H)
            # depends on s and H alone. Stop anywhere, extend anywhere, and the
            # prefix is unchanged.
            #
            # It is also smooth everywhere — no plateau, no corner into the
            # decay — and it decays fastest early, when the model is furthest
            # from any optimum, rather than holding a constant rate for 90% of
            # the run.
            #
            # It does not reach zero, which is deliberate: decay-to-zero is a
            # borrowed prior we have not tested (docs/ML_BASICS.md §9), and a
            # schedule that never arrives is the honest default until the
            # floor-vs-zero control has actually run.
            half = max(1.0, float(a.lr_halflife))

            def factor(step):
                s = step + 1
                if s <= warm:
                    return _warm_cos(s)
                return 0.5 ** ((s - warm) / half)
        else:
            def factor(step):
                s = step + 1
                return min(s / warm, (warm / s) ** 0.5)

        for g in opt.param_groups:
            g.setdefault("initial_lr", g["lr"])
        return torch.optim.lr_scheduler.LambdaLR(opt, factor,
                                                 last_epoch=last_epoch)
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps,
                                                      last_epoch=last_epoch)


def codec_weight_hash(ck):
    """Identity of the codec that produced an embedding, in ten hex digits.

    The embed cache MUST be codec-aware. A bare Z_<run>.npy poisoned runs
    #10/#11 (2026-08-07): the Actions cache carried run #8's embeddings, the
    (T, P, d_z) shape check matched, and two stage-2 models trained on the
    WRONG codec's z — healthy z-space skill, catastrophic decoded skill,
    because the z they predicted was not the z their decoder speaks. The hash
    in the filename makes a stale cache a miss rather than a lie.

    It lives in a function because embed_cache_sync.py has to derive exactly
    the same name to pull or push the cache, and two copies of a hash rule are
    two chances to disagree — which would silently reintroduce the #10/#11
    failure through the release instead of through the local disk.
    """
    import hashlib
    return hashlib.md5(b"".join(
        v.numpy().tobytes()
        for v in list(ck["model"].values())[:4])).hexdigest()[:10]


def embed_cache_path(run, whash):
    return os.path.join(HERE, "cache", f"Z_{run}_{whash}.npy")



def _progress_path(tmp):
    return tmp + ".progress"


def _resume_partial(tmp, T, P, d_z):
    """(memmap, months_already_done) for a half-built cache, or (None, 0).

    The marker is written AFTER the data is flushed, never before, so it can
    only ever under-claim. An over-claiming marker would be the worst possible
    outcome here: the run would skip months that were never written and the
    embedding would carry zeros for them — real numbers, wrong months, no
    symptom. Losing a few minutes of recomputation is the cheap side of that
    trade and it is the side this takes.
    """
    prog = _progress_path(tmp)
    if not (os.path.exists(tmp) and os.path.exists(prog)):
        return None, 0
    try:
        with open(prog) as f:
            mark = json.load(f)
        if (tuple(mark.get("shape", ())) != (T, P, d_z)
                or mark.get("dtype") != str(np.dtype(CACHE_DTYPE))):
            print(f"  ignoring a partial cache for a different shape/dtype "
                  f"({mark.get('shape')}, {mark.get('dtype')})")
            return None, 0
        done = int(mark.get("months_done", 0))
        out = np.load(tmp, mmap_mode="r+")
        if out.shape != (T, P, d_z) or not (0 < done < T):
            return None, 0
        print(f"  RESUMING the embedding at month {done}/{T} "
              f"({done / T * 100:.1f}% already on disk) — "
              f"{(T - done) / T * 100:.0f}% left to compute", flush=True)
        return out, done
    except Exception as e:                                    # noqa: BLE001
        print(f"  partial cache unusable ({type(e).__name__}: {e}) — "
              f"starting the embedding from scratch")
        return None, 0


def _mark_progress(tmp, out, months_done, T, P, d_z):
    """Flush the DATA, then record how far it got. Order is the whole point."""
    try:
        out.flush()
        p = _progress_path(tmp)
        with open(p + ".part", "w") as f:
            json.dump({"months_done": int(months_done), "shape": [T, P, d_z],
                       "dtype": str(np.dtype(CACHE_DTYPE))}, f)
        os.replace(p + ".part", p)
    except OSError as e:
        print(f"  (progress marker failed: {e})", flush=True)


def _free_ram_bytes():
    """MemAvailable, i.e. what can be allocated without swapping — not MemFree,
    which excludes reclaimable page cache and reads absurdly low on a box that
    has just streamed a 10 GB tensor through it."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _cache_plan(cache_path, need_bytes):
    """Decide WHERE the embedding lives: the disk cache, or RAM.

    The memmap was introduced because Z is ~10.4 GiB next to a 10.1 GiB
    tensor and that combination OOM-killed a 7 GB box twice on 2026-08-07.
    The boxes we rent now carry 126 GB of RAM and a 50 GB disk, so the
    constraint has moved and the code had not: #117 spent an hour writing a
    10.4 GiB cache toward 6 GiB of free disk, on a machine using 15 of its
    126 GB of memory. Memmapping to the scarce resource because the abundant
    one used to be scarce is the whole bug.

    Order of preference, and the reasons:
      1. DISK, if it fits after pruning — the cache is worth real money. A
         repeat stage-2 run on the same box skips ~50 minutes of embedding.
      2. RAM, if the disk cannot hold it but memory can. The run proceeds and
         only the cache is lost, which costs the NEXT run, not this one.
      3. Refuse. Both exhausted means the box is the wrong size for the job,
         and that is worth saying before an hour rather than after.
    """
    import shutil
    gb = lambda b: b / (1 << 30)
    d = os.path.dirname(cache_path)
    os.makedirs(d, exist_ok=True)
    # The headroom SCALES WITH THE ALLOCATION, capped. A flat 3 GiB reserve
    # reads as prudence until a smoke test with a 5 MB cache is refused on a
    # sandbox with 0.9 GiB free — the constant, not the risk, was doing the
    # refusing. What actually matters is that a big write leaves the box
    # usable afterwards, so the demand is proportional to the write and
    # bounded above by what a runner needs to keep working.
    want = need_bytes + min(RESERVE_BYTES, max(need_bytes, 256 << 20))
    _prune_stale(cache_path, want)
    free = shutil.disk_usage(d).free
    print(f"  embed cache needs {gb(need_bytes):.2f} GiB "
          f"(+{gb(want - need_bytes):.2f} headroom); "
          f"{gb(free):.2f} GiB disk free")
    if free >= want:
        return True
    ram = _free_ram_bytes()
    if ram >= need_bytes + min(RAM_HEADROOM_BYTES,
                               max(need_bytes, 512 << 20)):
        print(f"  disk cannot hold it — building Z in RAM instead "
              f"({gb(ram):.0f} GiB available). The cache is skipped, so the "
              f"NEXT stage-2 run on this box re-embeds; this one proceeds.")
        return False
    raise SystemExit(
        f"nowhere to put the embedding: needs {gb(need_bytes):.1f} GiB, "
        f"disk has {gb(free):.1f} GiB free after pruning every stale Z_*.npy "
        f"and RAM has {gb(ram):.1f} GiB available. Refusing to start — "
        f"open_memmap allocates lazily, so starting anyway would fail "
        f"mid-write an hour from now with the disk full and the runner "
        f"offline. Free space on the box, rent a larger one, or "
        f"use --max-pixels.")


def _prune_stale(cache_path, want):
    """Free space for the embedding cache BEFORE opening the memmap.

    `open_memmap` creates a SPARSE file: the 10.4 GiB is claimed lazily, page
    by page, over the ~50 minutes the embedding takes. So a box with 7 GiB
    free starts happily, runs for forty minutes, and dies on a write with the
    cache 90% built — and on these 50 GB boxes a full disk does not just fail
    the job, it takes the runner offline (CLAUDE.md Part 2). The failure is
    maximally expensive and maximally late.

    ml-train.yml has a hygiene step that prunes below 8 GB free. That number
    is SMALLER THAN THE SINGLE ALLOCATION IT GUARDS: a guard sized under the
    thing it is guarding against will pass and then the write will fail. The
    check belongs here, where T, P and d_z are known and the requirement is a
    computed number rather than a guess.

    Stale Z_*.npy from other codecs are the reclaimable tier — they are pure
    cache, keyed by a weight hash, and re-derive from a checkpoint. Anything
    else on that disk (tensors, checkpoints) is not ours to delete.
    """
    import glob
    import shutil
    d = os.path.dirname(cache_path)
    free = shutil.disk_usage(d).free
    gb = lambda b: b / (1 << 30)
    if free >= want:
        return
    stale = sorted((p for p in glob.glob(os.path.join(d, "Z_*.npy")) +
                    glob.glob(os.path.join(d, "Z_*.npy.partial"))
                    if os.path.abspath(p) != os.path.abspath(cache_path)),
                   key=lambda p: os.path.getmtime(p))
    for p in stale:
        try:
            n = os.path.getsize(p)
            os.remove(p)
            free = shutil.disk_usage(d).free
            print(f"  pruned stale embed cache {os.path.basename(p)} "
                  f"({gb(n):.1f} GiB) — {gb(free):.1f} GiB free")
        except OSError as e:
            print(f"  could not prune {p}: {e}")
        if free >= want:
            return
    # Not an error here — _cache_plan decides what to do when the disk still
    # cannot hold it, and RAM is usually the answer on these boxes.


def embed_everything(model, X, OBS, ctx_all, lats, lons, ys, xs, d_z,
                     cache_path=None, batch=8192, mask_chan=None,
                     progress=None):
    """Frozen codec embeddings for every (t, pixel in ys/xs): [T, P, d_z].
    Cached on disk — the embedding pass is the expensive part of stage 2
    (T×P encoder forwards), and every probe variant reuses it.

    Runs on whatever device the MODEL is on: 401 months x ~45k ocean pixels
    is 18M encoder forwards, which is hours of CPU and minutes of GPU. The
    big tensors (X, OBS, and the output) stay in host memory — only the
    per-batch slice crosses — because Z alone is ~4.6 GB at global scale and
    the point is to spend VRAM on arithmetic, not storage."""
    dev = next(model.parameters()).device
    T, H, W, C = X.shape
    P = len(ys)
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    # Z is T*P*d_z*4 bytes — 4.6 GB on the global grid at d_z=64, next to a
    # 1.4 GB tensor and a 0.3 GB mask. Built in RAM it OOM-kills a 7 GB box
    # (twice on 2026-08-07), and it is written to disk immediately afterwards
    # anyway. So it is BUILT in the cache file through a memmap: pages are
    # written as they are filled and the kernel may evict them, which turns a
    # hard 4.6 GB allocation into page-cache pressure. Reads go the same way.
    # Without a cache path (the --max-pixels smoke) it stays an ordinary array.
    if cache_path and os.path.exists(cache_path):
        out = np.load(cache_path, mmap_mode="r+")
        if out.shape == (T, P, d_z):
            print(f"  (cached: {cache_path})")
            return out, coords
    start_t = 0
    if cache_path and _cache_plan(cache_path,
                              T * P * d_z * CACHE_DTYPE(0).itemsize):
        tmp = cache_path + ".partial"
        # RESUMABLE. The embedding is ~95 minutes; dying at 80% and starting
        # again from zero is the difference between losing twenty minutes and
        # losing eighty. The half-written memmap already holds every completed
        # month — what was missing was a record of how many, so a restart
        # could trust it. Chris asked for exactly this on 2026-08-10.
        out, start_t = _resume_partial(tmp, T, P, d_z)
        if out is None:
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=CACHE_DTYPE,
                                            shape=(T, P, d_z))
    else:
        # NOT resumable, and say so rather than let it be discovered at 80%:
        # an in-memory array dies with the process. Since the cache went to
        # float16 this branch should be rare — 5.2 GiB fits where 10.4 did
        # not — and it is now a fallback rather than the normal path.
        cache_path = None                       # RAM path: nothing to publish
        print("  building Z in RAM: NOT resumable — if this process dies the "
              "whole embedding is lost. (Free disk so the cache fits and it "
              "becomes restartable.)", flush=True)
        out = np.zeros((T, P, d_z), dtype=CACHE_DTYPE)
    # THE EMBEDDING REPORTS ITS OWN PROGRESS. It is the longest single phase of
    # a stage-2 run — ~95 minutes for 43.5M encoder forwards on the
    # quarter-degree tensor — and until 2026-08-10 it printed one line when it
    # started and one when it finished. Actions will not serve logs for a
    # running job, so during that hour the only way to tell "working" from
    # "wedged" was to watch the box's resident memory climb as the array paged
    # in: a thermometer taped to the outside of the oven. Chris asked how far
    # along it was and the honest answer was an inference, which is not an
    # answer. Every 5% now costs one line and answers it directly.
    t_emb = time.time()
    next_mark = 0.0
    with torch.no_grad():
        for t in range(start_t, T):
            frac = (t + 1) / T
            if frac >= next_mark:
                el = time.time() - t_emb
                eta = el / frac - el if frac > 0 else 0
                print(f"  embedding {frac * 100:5.1f}%  month {t + 1}/{T}  "
                      f"{el / 60:.0f} min elapsed, ~{eta / 60:.0f} min left",
                      flush=True)
                if progress:
                    progress({"pct": round(frac * 100, 1), "month": t + 1,
                              "months": T, "elapsed_s": round(el),
                              "eta_s": round(eta),
                              "where": "disk" if cache_path else "ram"})
                next_mark = frac + 0.05
            for i in range(0, P, batch):
                sl = slice(i, min(i + batch, P))
                n = sl.stop - sl.start
                ctx = np.concatenate([np.tile(ctx_all[t], (n, 1)), coords[sl]], 1)
                mk = torch.zeros(n, C, dtype=torch.bool)
                if mask_chan is not None:
                    mk[:, mask_chan] = True
                patch = getattr(model, "patch", 1)
                ctx_t = torch.as_tensor(ctx, dtype=torch.float32).to(dev)
                if patch > 1:
                    from model import gather_px
                    tt = torch.full((n,), t, dtype=torch.long)
                    v, o = gather_px(X, OBS, tt, torch.as_tensor(ys[sl]),
                                     torch.as_tensor(xs[sl]), patch)
                    z = model.encode((v * (~mk).unsqueeze(-1)).to(dev),
                                     o.to(dev), mk.to(dev), ctx_t)
                else:
                    v = X[t, ys[sl], xs[sl]] * (~mk)
                    z = model.encode(v.to(dev), OBS[t, ys[sl], xs[sl]].to(dev),
                                     mk.to(dev), ctx_t)
                out[t, sl] = z.cpu().numpy()
            # Every 8 months (~1.5 minutes of work) flush the pages and record
            # the count. Cheap enough to be unnoticeable, fine-grained enough
            # that a crash costs a couple of minutes rather than an hour.
            if cache_path and (t + 1) % 8 == 0:
                _mark_progress(tmp, out, t + 1, T, P, d_z)
    if cache_path:
        # Already on disk in .npy form — flush the pages, then publish
        # atomically so an interrupted run never leaves a half-filled cache
        # that the shape check would happily accept next time.
        out.flush()
        del out
        os.replace(tmp, cache_path)
        # The marker describes a .partial that no longer exists; leaving it
        # would make the next run try to resume a file it cannot find.
        try:
            os.remove(_progress_path(tmp))
        except OSError:
            pass
        out = np.load(cache_path, mmap_mode="r+")
    return out, coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot4_anom")
    ap.add_argument("--K", type=int, default=24, help="context length (months)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr-schedule", default="cosine",
                    choices=["cosine", "invsqrt", "wsd", "expdecay"],
                    help="cosine bakes the TOTAL step count into the rate, so "
                         "a checkpoint's schedule only means anything next to "
                         "the budget it was trained under (this is what makes "
                         "a resumed run read lr=0.0 and what forces every "
                         "budget to be its own experiment). invsqrt is "
                         "horizon-free: lr(s) depends only on s, so a run "
                         "stopped at 60k and continued to 200k sees exactly "
                         "what an uninterrupted 200k run would have. Default "
                         "stays cosine because every existing result used it.")
    ap.add_argument("--lr-halflife", type=float, default=40000,
                    help="expdecay only: steps for the rate to halve. ABSOLUTE, "
                         "not a fraction of --steps, which is what makes the "
                         "schedule horizon-free: extend the run and the curve "
                         "it already walked is unchanged.")
    ap.add_argument("--lr-cooldown-frac", type=float, default=0.1,
                    help="wsd only: fraction of --steps spent decaying "
                         "linearly to zero at the end. The stable phase before "
                         "it is horizon-free, so a run can be extended; the "
                         "cooldown is what makes the result CONVERGED rather "
                         "than a reading at step N.")
    ap.add_argument("--lr-warmup", type=int, default=2000,
                    help="invsqrt only: steps to reach the peak, after which "
                         "lr = peak * sqrt(warmup / step)")
    ap.add_argument("--init-temporal", default="",
                    help="WARM RESTART: take the WEIGHTS of a stage-2 head and "
                         "train --steps more with a fresh cosine at --lr. "
                         "Adam's moments, the schedule position and the RNG "
                         "stream are NOT inherited, because the published "
                         "heads do not carry them — every checkpoint written "
                         "before 2026-08-10 is {args, model} only. This is a "
                         "separate flag from --resume-temporal on purpose: "
                         "the two produce different trajectories, and the one "
                         "mistake worth engineering against is reporting a "
                         "warm restart as though it were a continuation. Here "
                         "--steps is the EXTRA, not the total.")
    ap.add_argument("--resume-temporal", default="",
                    help="continue a stage-2 head: a path, or a tag under "
                         "/opt/earth-cache/ckpt (e.g. run-112-temporal). The "
                         "checkpoint carries model, optimiser, scheduler, step "
                         "and RNG state, so the continuation is the SAME "
                         "trajectory rather than a fresh run that happens to "
                         "start from these weights.\n\n"
                         "NOTE the schedule semantics: --steps is the TOTAL, "
                         "not the extra. Resuming a 60,000-step head with "
                         "--steps 200000 fast-forwards a 200,000-step cosine "
                         "to step 60,000 and carries on. The original head "
                         "annealed to ~0 over its own 60,000, so its LR steps "
                         "back UP — a warm restart, which is a different "
                         "object from a single 200,000-step run and must be "
                         "labelled as such when the numbers are compared.")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch/numpy seed (sweeps need more than one)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files: temporal_<tag>.json/.pt")
    ap.add_argument("--unroll", type=int, default=1,
                    help="AUTOREGRESSIVE UNROLL DEPTH in the loss. 1 (default) "
                         "is the original teacher-forced t+1 objective. >1 "
                         "feeds the model's OWN prediction back in for this "
                         "many extra steps and backpropagates through the "
                         "chain, which is the standard fix for EXPOSURE BIAS: "
                         "a model trained only on true context never sees its "
                         "own errors, so they compound at rollout — and "
                         "rollout horizon is a headline claim of this "
                         "programme (rollout.py), measured on a model that "
                         "was never trained for it. Costs one extra forward "
                         "and backward per extra step.")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="subsample ocean pixels (code-path smoke only; "
                         "the 26.5N section is always kept)")
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"),
                    help="tensor npz (family-3 runs pass family3_na025.npz)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run_dir = os.path.join(HERE, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    if not ck["args"].get("anomaly"):
        sys.exit("stage 2 requires an anomaly-space codec (train.py --anomaly): "
                 "state-space embeddings failed the K-sweep precondition.")
    d = np.load(a.data)
    X = d["X"].copy()
    months = [str(m) for m in d["months"]]
    lats, lons, chan = d["lats"], d["lons"], [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    # identical anomaly transform to train.py --anomaly (train-years clim)
    dynamic = [c for c in range(C)
               if np.nanstd(np.nanmean(X[..., c], axis=(1, 2))) > 1e-6]
    clim = np.full((12, H, W, C), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in range(12):
            clim[m] = np.nanmean(X[(moy == m) & ~t_hold], axis=0)
    for c in dynamic:
        # clim[moy, :, :, c] is [T,H,W] (98 MB); the equivalent-looking
        # clim[moy][..., c] materialises the whole [T,H,W,C] fancy index
        # (1.4 GB) and throws all but one channel away — once per dynamic
        # channel. That transient is most of why this OOM-killed the
        # xlarge stage-2 on a 7 GB box (2026-08-07).
        X[..., c] = X[..., c] - clim[moy, :, :, c]
        v = X[..., c][np.isfinite(X[..., c]) & ~t_hold[:, None, None]
                      & ~x_hold[None, None, :]]
        X[..., c] = (X[..., c] - v.mean()) / (v.std() + 1e-6)
    del clim

    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval()

    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)], 1)
    Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(X))
    # X is dead from here (everything downstream reads Xt/OBS), and the
    # embedding array alone is T*P*d_z*4 ~ 4.6 GB at global scale — so the
    # 1.4 GB anomaly copy has to go before it is allocated. Likewise `ocean`
    # comes from OBS rather than d["X"][..., 0], which would load the whole
    # 1.4 GB npz member again just to slice one channel off it.
    ocean = OBS[..., 0].any(axis=0).numpy()
    del X
    import gc
    gc.collect()

    ys, xs = np.where(ocean)
    sec_y, sec_sel0 = rapid_section(lats, lons, ys, xs)
    if a.max_pixels and a.max_pixels < len(ys):
        rng = np.random.default_rng(0)
        keep = rng.choice(len(ys), a.max_pixels, replace=False)
        keep = np.union1d(keep, sec_sel0)                   # probe needs the section
        ys, xs = ys[keep], xs[keep]

    # The embedding pass is the only part worth a GPU here (18M encoder
    # forwards); stage-2 training is a small transformer for a few thousand
    # steps, and every eval below is numpy-bound. So the codec visits the
    # accelerator for the embedding and the static-identity pass, and comes
    # straight back to the CPU — leaving all downstream code untouched
    # rather than device-threaded, which is where the bugs would be.
    EDEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec.to(EDEV)
    print(f"embedding every (month, ocean pixel) through the frozen codec "
          f"on {EDEV.type} …")
    t0 = time.time()
    # The embed cache must be CODEC-AWARE: a bare Z_<run>.npy poisoned runs
    # #10/#11 (2026-08-07) — the Actions cache carried run #8's embeddings,
    # the (T, P, d_z) shape check matched, and both stage-2s trained on the
    # WRONG codec's z (healthy z-space skill, catastrophic decoded skill:
    # the z they predicted was not the z their decoder speaks). The weight
    # hash in the filename makes a stale cache a miss, never a lie.
    whash = codec_weight_hash(ck)
    cache = (embed_cache_path(a.run, whash) if not a.max_pixels else None)
    # Progress goes into the run's OWN metrics.jsonl, which the publisher loop
    # in ml-train.yml pushes to ml-live-<n> every five minutes. A print reaches
    # the log, and Actions will not serve the log of a running job — so during
    # the hour this takes, stdout is write-only. The side channel is the only
    # one anybody can read while it matters.
    def _emb_note(rec):
        try:
            os.makedirs(os.path.join(HERE, "runs", a.run), exist_ok=True)
            with open(os.path.join(HERE, "runs", a.run, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"embedding": rec}) + "\n")
        except OSError:
            pass                      # instrumentation never breaks the run

    Z, coords = embed_everything(codec, Xt, OBS, ctx_all, lats, lons, ys, xs,
                                 ck["d_z"], cache_path=cache,
                                 progress=_emb_note)
    P = len(ys)
    print(f"  Z [T={T} P={P} d_z={ck['d_z']}]  ({time.time() - t0:.0f}s)")

    # static identity of each pixel: codec embedding of static channels only
    with torch.no_grad():
        stat_obs = OBS[0].clone()
        for c in dynamic:
            stat_obs[..., c] = False
        zs = []
        for i in range(0, P, 8192):
            sl = slice(i, min(i + 8192, P))
            n = sl.stop - sl.start
            ctx = np.concatenate([np.zeros((n, 2), np.float32), coords[sl]], 1)
            if getattr(codec, "patch", 1) > 1:
                from model import gather_px
                # One gather: values from month 0 (statics are constant in t;
                # dynamics are zeroed inside encode because their obs is
                # False), obs from stat_obs with gather_px's own out-of-range
                # latitude masking — exactly what training-time encode saw.
                t0i = torch.zeros(n, dtype=torch.long)
                vv, oo = gather_px(Xt, stat_obs[None], t0i,
                                   torch.as_tensor(ys[sl]),
                                   torch.as_tensor(xs[sl]), codec.patch)
                zs.append(codec.encode(vv.to(EDEV), oo.to(EDEV),
                                       torch.zeros(n, C, dtype=torch.bool, device=EDEV),
                                       torch.as_tensor(ctx).to(EDEV)).cpu().numpy())
            else:
                zs.append(codec.encode(Xt[0, ys[sl], xs[sl]].to(EDEV),
                                       stat_obs[ys[sl], xs[sl]].to(EDEV),
                                       torch.zeros(n, C, dtype=torch.bool, device=EDEV),
                                       torch.as_tensor(ctx).to(EDEV)).cpu().numpy())
        Zstat = np.concatenate(zs, 0)
    codec.to("cpu")          # everything below is CPU/numpy, unchanged
    static_ctx = torch.as_tensor(np.concatenate([Zstat, coords], 1))

    # ---- train pool: windows [t-K+1 .. t] whose TARGET month t+1 is a train
    # month and whose pixel is outside the longitude holdout. Windows may LOOK
    # at held-out months (persistence can too); they may never be SCORED on
    # them in training.
    Zt = torch.from_numpy(Z)
    Mt = torch.as_tensor(ctx_all, dtype=torch.float32)
    K = a.K
    # With --unroll U the loss reaches U months past the window, so the pool
    # must guarantee those months EXIST and are TRAIN months. Without this the
    # unrolled steps would either index off the end of the array or be scored
    # on the holdout — the second is the one that would not have crashed.
    U = max(1, a.unroll)
    ok_t = np.array([t + U < T and t + 1 >= K
                     and not t_hold[t + 1:t + U + 1].any()
                     for t in range(T)])
    ok_p = ~x_hold[xs]
    pool_t, pool_p = np.where(ok_t[:, None] & ok_p[None, :])
    pool_t = torch.as_tensor(pool_t, dtype=torch.long)
    pool_p = torch.as_tensor(pool_p, dtype=torch.long)
    print(f"train windows: {len(pool_t):,}")

    model = TemporalTransformer(d_z=ck["d_z"], d_model=a.d_model,
                                n_layers=a.layers, k_max=K)
    # STAGE-2 TRAINING RUNS ON THE ACCELERATOR TOO.
    # The comment above this block used to say stage-2 training is "a small
    # transformer for a few thousand steps" and therefore not worth a GPU.
    # That was true when stage 2 was 4,000 steps. It is now 140,000 and
    # 200,000, and the premise expired without the code noticing — the same
    # shape of error as memmapping to disk because the box used to have 7 GB
    # of RAM.
    #
    # Measured before changing it, rather than assumed: at batch 256, K=24 and
    # a 1.824M head, the data gather off the memmap is 12.4 ms of a 725 ms
    # step. The model is 98% of the cost, so the accelerator is worth between
    # 5x and 20x on a run that otherwise takes a full day.
    #
    # The batch gather stays on the CPU — Z is a 5.2 GiB memmap and random
    # rows out of it belong where the pages are — and only the assembled
    # batch crosses, which is 256 x 25 x 64 fp32, about 1.6 MB.
    TDEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(TDEV)
    print(f"stage-2 head on {TDEV.type} "
          f"({sum(p_.numel() for p_ in model.parameters()) / 1e6:.3f}M params)",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = make_sched(opt, a)

    start_step = 0
    init_from = None
    if a.init_temporal:
        # WARM RESTART, named as one. Every stage-2 head published before
        # 2026-08-10 is {args, model}: no optimiser moments, no schedule
        # position, no RNG stream — measured on f3_s2_60k, f3_s2_24k and the
        # rescue mirrors, all three. So a "continuation" from any of them is
        # impossible, and --resume-temporal correctly refuses.
        #
        # What IS available is a cosine restart: take the converged weights,
        # train a fresh schedule at a lower peak. That is a real and standard
        # way to spend more compute on a trained model, and it answers "do
        # 140,000 more steps help?" — it simply is not the same trajectory a
        # straight-through 200,000-step run would have taken, so it must never
        # be plotted as a fourth point on a curve whose other points were each
        # their own converged cosine from scratch.
        #
        # --steps is the EXTRA here, not the total, precisely because there is
        # no step count in the checkpoint to be the total OF.
        ip = (a.init_temporal if os.path.sep in a.init_temporal
              else os.path.join(CKPT_DIR, a.init_temporal + ".pt"))
        if not os.path.exists(ip):
            raise SystemExit(
                f"--init-temporal: no checkpoint at {ip}. Refusing to start a "
                f"fresh head under a doc string that says the weights came "
                f"from somewhere.")
        tk = torch.load(ip, map_location="cpu", weights_only=False)
        model.load_state_dict(tk["model"])
        parent_steps = int(tk.get("args", {}).get("steps", 0))
        parent_lr = tk.get("args", {}).get("lr")
        init_from = {"from": os.path.basename(ip),
                     "parent_steps": parent_steps, "parent_lr": parent_lr,
                     "extra_steps": a.steps, "lr": a.lr,
                     "inherited": ["model"],
                     "reset": ["optimiser moments", "schedule position",
                               "rng stream"],
                     "kind": "warm restart (cosine restart), NOT a continuation"}
        carried = [k for k in ("opt", "sched", "step") if k in tk]
        if carried:
            print(f"  note: {ip} DOES carry {carried} — --resume-temporal "
                  f"would give a true continuation and is the better choice",
                  flush=True)
        print(f"WARM RESTART from {ip}: weights of a {parent_steps:,}-step head "
              f"(peak lr {parent_lr}), now {a.steps:,} MORE steps on a fresh "
              f"cosine at peak {a.lr:.2e}. Adam's moments and the schedule "
              f"start from nothing — this is not the same trajectory as a "
              f"{parent_steps + a.steps:,}-step run and must not be reported "
              f"as one.", flush=True)
    _parent = {}
    if a.resume_temporal:
        rp = (a.resume_temporal if os.path.sep in a.resume_temporal
              else os.path.join(CKPT_DIR, a.resume_temporal + ".pt"))
        if not os.path.exists(rp):
            raise SystemExit(
                f"--resume-temporal: no checkpoint at {rp}. Refusing to start "
                f"a fresh head under a doc string that says 'continue' — that "
                f"is the mistake --require-resume exists to prevent on the "
                f"codec side.")
        tk = torch.load(rp, map_location="cpu", weights_only=False)
        model.load_state_dict(tk["model"])
        missing = [k for k in ("opt", "sched", "step") if k not in tk]
        if missing:
            raise SystemExit(
                f"--resume-temporal: {rp} predates optimiser-state saving "
                f"(missing {missing}). Loading the weights alone would reset "
                f"Adam's moments and the LR schedule, which is a warm restart "
                f"wearing a continuation's name. Refusing.\n\n"
                f"Every head published before 2026-08-10 is {{args, model}} "
                f"only — measured on f3_s2_60k, f3_s2_24k and the rescue "
                f"mirrors — so no existing checkpoint can be CONTINUED. If a "
                f"warm restart is what you want, ask for it by name: "
                f"--init-temporal {a.resume_temporal} --steps <EXTRA> --lr "
                f"<peak>, which trains a fresh cosine from these weights and "
                f"records that the moments were reset. Heads written from now "
                f"on carry opt/sched/step and are continuable.")
        opt.load_state_dict(tk["opt"])
        start_step = int(tk["step"])
        # THE SCHEDULE NEEDS A DECISION, and getting it wrong is silent.
        # CosineAnnealingLR.load_state_dict restores T_max and base_lrs from
        # the OLD run, so loading it while asking for a LARGER --steps leaves
        # T_max at the old total with last_epoch already there: the learning
        # rate is exactly 0.0 and the continuation trains 140,000 steps at
        # nothing. Measured, not feared — and the toy end-to-end run printed
        # "lr now 0.000e+00" while I read past it.
        _parent = dict(tk.get("args", {}))
        _parent["run_number"] = tk.get("run_number")
        prev_total = int(tk.get("args", {}).get("steps", start_step))
        extending = (a.steps != prev_total) or (abs(a.lr - float(
            tk.get("args", {}).get("lr", a.lr))) > 1e-12)
        if a.lr_schedule in ("invsqrt", "wsd"):
            # NOTHING TO DECIDE. A horizon-free schedule is a pure function of
            # the step, so extending is not a case: rebuild it at the same
            # position and it produces exactly what an uninterrupted run of
            # any length would produce there. This branch existing at all is
            # the cost of baking the total into the rate.
            for g in opt.param_groups:
                g["lr"] = a.lr
                g["initial_lr"] = a.lr
            sched = make_sched(opt, a, last_epoch=start_step - 1)
            print(f"  invsqrt: horizon-free, so the continuation simply "
                  f"resumes at step {start_step:,} — no extension case",
                  flush=True)
        elif extending:
            for g in opt.param_groups:
                g["lr"] = a.lr
                g["initial_lr"] = a.lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, a.steps, last_epoch=start_step - 1)
            print(f"  EXTENDING: new cosine over {a.steps:,} steps at peak lr "
                  f"{a.lr:.2e} (was {prev_total:,} steps at "
                  f"{tk.get('args', {}).get('lr')}), positioned at step "
                  f"{start_step:,}", flush=True)
        else:
            sched.load_state_dict(tk["sched"])
            print("  exact continuation: same total and same lr, schedule "
                  "state restored verbatim", flush=True)
        if tk.get("torch_rng") is not None:
            torch.set_rng_state(torch.as_tensor(tk["torch_rng"], dtype=torch.uint8))
        if start_step >= a.steps:
            raise SystemExit(
                f"--resume-temporal: checkpoint is at step {start_step:,} and "
                f"--steps is {a.steps:,}. --steps is the TOTAL, not the extra.")
        lr_now = sched.get_last_lr()[0]
        print(f"resumed stage-2 head from {rp} at step {start_step:,} "
              f"-> training to {a.steps:,} (lr now {lr_now:.3e})", flush=True)
        # An invariant with an exact expectation, which is worth more than any
        # amount of careful reading: you cannot train at zero. Refuse rather
        # than spend sixteen hours updating nothing.
        if not (lr_now > 1e-12):
            raise SystemExit(
                f"--resume-temporal: the resumed learning rate is {lr_now:.3e}. "
                f"Training {a.steps - start_step:,} steps at that rate would "
                f"change nothing and report success. Check --steps (total, not "
                f"extra) and --lr.")

    def batch_windows(idx_t, idx_p, n):
        k = torch.randint(0, len(idx_t), (n,))
        t, p = idx_t[k], idx_p[k]
        base = t - K + 1
        zseq = torch.stack([Zt[base + j, p] for j in range(K)], 1).float()
        mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
        ztgt = torch.stack([Zt[base + j + 1, p] for j in range(K)], 1).float()
        # True embeddings BEYOND the window, for the autoregressive unroll:
        # zfut[:, u] = Z[t+1+u] is the truth the model must hit after u
        # SELF-FED steps — u, not u+1. Column 0 is therefore the ordinary
        # teacher-forced target and is deliberately never read: the base term
        # takes it from ztgt, which scores the whole window rather than only
        # its last step. The loop below starts at u=1 for that reason, which
        # is also why U=1 leaves the objective bit-identical to the
        # pre-unroll one. (The comment here previously said "u+1 self-fed
        # steps", which contradicted both the code and ml/EXPERIMENTS.md;
        # the code was right.)
        zfut = torch.stack([Zt[t + 1 + u, p] for u in range(U)], 1).float()
        mfut = torch.stack([Mt[t + 1 + u] for u in range(U)], 1)
        return (zseq.to(TDEV), mseq.to(TDEV), static_ctx[p].to(TDEV),
                ztgt.to(TDEV), zfut.to(TDEV), mfut.to(TDEV))

    # Stage 2 goes into the RUN'S OWN metrics.jsonl, not just temporal.json.
    # temporal.json is uploaded as a build artifact, and artifacts need an
    # authenticated API call — the status page is deliberately credential-free
    # and reads only public raw branch content, so until now stage 2 was
    # invisible there: the page charted the codec's loss and its little
    # in-training probe, and said nothing about the model the whole second
    # stage exists to train. metrics.jsonl is already published to the live
    # branch and archived to ml-metrics AFTER this step runs, so writing here
    # needs no new transport.
    m2_path = os.path.join(run_dir, "metrics.jsonl")
    n_par2 = sum(p_.numel() for p_ in model.parameters())

    def m2(rec):
        try:
            with open(m2_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass                      # instrumentation never breaks the run

    if start_step:
        m2({"stage2_resumed": {
            "from": os.path.basename(a.resume_temporal),
            "at_step": start_step, "to_step": a.steps,
            # Enough to redraw the parent's cosine EXACTLY without fetching
            # anything: annealing is analytic given peak and total.
            "parent_run": _parent.get("run_number"),
            "parent_steps": _parent.get("steps"),
            "parent_lr": _parent.get("lr"),
            "lr": a.lr}})
    if init_from:
        # A DIFFERENT record name from stage2_resumed, deliberately. The
        # status page and every later reader must be able to tell a warm
        # restart from a continuation without parsing prose, because the two
        # answer different questions and only one of them belongs on E-007's
        # curve.
        m2({"stage2_warm_restart": init_from})
    m2({"stage2_config": {"d_model": a.d_model, "layers": a.layers, "K": K,
                          "steps": a.steps, "params_M": round(n_par2 / 1e6, 3),
                          "d_z": int(ck["d_z"]), "seed": a.seed,
                          "unroll": a.unroll, "tag": a.tag or ""}})

    print(f"training the temporal stage … ({n_par2:,} parameters)")
    t0 = time.time()
    log_every = max(1, a.steps // 100)     # ~100 curve points, as stage 1 does
    for s in range(start_step + 1, a.steps + 1):
        zseq, mseq, sctx, ztgt, zfut, mfut = batch_windows(pool_t, pool_p, a.batch)
        pred, _ = model(zseq, mseq, sctx)
        loss = (pred - ztgt).pow(2).mean()
        # AUTOREGRESSIVE UNROLL — the fix for EXPOSURE BIAS. rollout.py scores
        # this model by feeding its own predictions back in, but the objective
        # above only ever shows it TRUE context, so it is never trained on the
        # error distribution it will actually face and errors compound at
        # rollout. Here the context slides forward on the model's own last
        # prediction and the next true month is the target. Each extra step is
        # down-weighted 1/(u+1) so a deep unroll cannot outvote the t+1 term
        # that anchors the whole objective.
        zin, min_ = zseq, mseq
        for u in range(1, U):
            zin = torch.cat([zin[:, 1:], pred[:, -1:]], 1)      # graph intact
            min_ = torch.cat([min_[:, 1:], mfut[:, u - 1:u]], 1)
            pred, _ = model(zin, min_, sctx)
            loss = loss + (pred[:, -1] - zfut[:, u]).pow(2).mean() / (u + 1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % log_every == 0 or s == a.steps:
            m2({"stage2_step": s, "stage2_zmse": round(float(loss.item()), 5),
                # The RATE, logged rather than inferred. A resumed run's
                # schedule is the thing most likely to be wrong (it was: a
                # reloaded cosine gave lr 0.0), and a chart that shows the
                # loss without the rate cannot distinguish "converged" from
                # "not learning because the LR is zero".
                "stage2_lr": float(sched.get_last_lr()[0]),
                "stage2_wall_s": round(time.time() - t0, 1)})
            # MIRROR THE HEAD AS IT TRAINS, exactly as train.py mirrors the
            # codec. Until now the head existed only in the run's workspace
            # and was uploaded by a step that runs AFTER the whole probe
            # ladder — so a job that hit its timeout lost every step of it.
            # A 60,000-step head is seven hours of GPU; losing it to a
            # bookkeeping deadline is not an acceptable failure mode.
            # Cheap: 7 MB, ~100 writes over a run.
            try:
                os.makedirs(CKPT_DIR, exist_ok=True)
                tag = os.environ.get("CKPT_TAG", "")
                tmp_path = os.path.join(
                    CKPT_DIR, (tag + "-" if tag else "") + "temporal.pt")
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "step": s,
                            # OPTIMISER AND SCHEDULE TOO. Weights alone are not
                            # a resumable state: reloading them and building a
                            # fresh AdamW resets the moments and restarts the
                            # cosine, which is a warm restart wearing a
                            # continuation's name. RNG state as well, so the
                            # window draw continues rather than repeating.
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
                            "torch_rng": torch.get_rng_state().numpy().tolist()},
                           tmp_path + ".part")
                os.replace(tmp_path + ".part", tmp_path)
            except Exception as e:                       # never fatal
                print(f"  (head mirror failed: {e})", flush=True)
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  z-mse {loss.item():.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    results = {"run": a.run, "K": K, "d_model": a.d_model, "layers": a.layers,
               "steps": a.steps}

    # ---- eval 1: z-space t+1 on held-out target months --------------------
    with torch.no_grad():
        ev_t = np.array([t + 1 < T and t_hold[t + 1] and t + 1 >= K
                         for t in range(T)])
        et, ep = np.where(ev_t[:, None] & np.ones(P, bool)[None, :])
        sel = np.random.default_rng(a.seed).choice(len(et), min(20000, len(et)), replace=False)
        et = torch.as_tensor(et[sel], dtype=torch.long)
        ep = torch.as_tensor(ep[sel], dtype=torch.long)
        base = et - K + 1
        zseq = torch.stack([Zt[base + j, ep] for j in range(K)], 1).float()
        mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
        pred, hid = model(zseq.to(TDEV), mseq.to(TDEV), static_ctx[ep].to(TDEV))
        pred, hid = pred.cpu(), hid.cpu()        # back to CPU: everything
        zhat = pred[:, -1]                       # below here is numpy-bound
        ztrue = Zt[et + 1, ep].float()
        zlast = Zt[et, ep].float()                        # persistence in z
        results["z_t+1"] = {
            "mse_model": float((zhat - ztrue).pow(2).mean()),
            "mse_persistence": float((zlast - ztrue).pow(2).mean()),
        }
        results["z_t+1"]["beats_persistence"] = (
            results["z_t+1"]["mse_model"] < results["z_t+1"]["mse_persistence"])

        # ---- eval 2: decode ẑ through the frozen codec → channel space ----
        qc = torch.arange(C)[None, :].expand(len(et), -1)
        off0 = torch.zeros(len(et), C, 3, dtype=torch.long)
        xhat = codec.query(zhat, qc, off0)
        ys_t = torch.as_tensor(ys, dtype=torch.long)
        xs_t = torch.as_tensor(xs, dtype=torch.long)
        v1 = Xt[et + 1, ys_t[ep], xs_t[ep]]
        o1 = OBS[et + 1, ys_t[ep], xs_t[ep]]
        v0 = Xt[et, ys_t[ep], xs_t[ep]]
        o0 = OBS[et, ys_t[ep], xs_t[ep]]
        both = o0 & o1
        dyn = torch.zeros(C, dtype=torch.bool); dyn[dynamic] = True
        both = both & dyn[None, :]
        mse_m = float(((xhat - v1).pow(2) * both).sum() / both.sum())
        mse_p = float(((v0 - v1).pow(2) * both).sum() / both.sum())
        results["chan_t+1"] = {"mse_model": mse_m, "mse_persistence": mse_p,
                               "beats_persistence": mse_m < mse_p,
                               "channels": [chan[c] for c in dynamic]}

    # ---- eval 3: RAPID probe from temporal hidden state -------------------
    # protocol v2: deseasonalised target (train-years clim), seasonal floor,
    # lambda on a train tail — identical scoring path to probe_sequence.py.
    rapid = d["rapid"]
    _, sec_after = rapid_section(lats, lons, ys, xs)   # ys/xs possibly subsampled
    sec_pix = torch.as_tensor(sec_after, dtype=torch.long)
    with torch.no_grad():
        F = np.zeros((T, a.d_model), dtype=np.float32)
        for t in range(K - 1, T):
            base = t - K + 1
            zseq = torch.stack([Zt[base + j, sec_pix] for j in range(K)], 1).float()
            mseq = torch.stack([Mt[base + j].expand(len(sec_pix), -1)
                                for j in range(K)], 1)
            _, hid = model(zseq.to(TDEV), mseq.to(TDEV),
                           static_ctx[sec_pix].to(TDEV))
            F[t] = hid[:, -1].mean(0).cpu().numpy()   # pool along the section
    ridx = rapid[:, 0].astype(int)
    rv_raw = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv_raw[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv_raw - rclim[rmoy]
    ok = ridx >= K - 1
    ri = ridx[ok]
    tr, te = ~t_hold[ri], t_hold[ri]
    r_raw, _ = ridge_r(F[ri], rv_raw[ok], tr, te)
    r_des, _ = ridge_r(F[ri], rv_des[ok], tr, te)
    results["rapid_probe"] = {"r_raw": r_raw, "r_deseasonalised": r_des,
                              "n_test": int(te.sum()), "features": "hidden(-1) mean over section"}

    print(json.dumps(results, indent=2))
    # The verdict, next to the curve, for the same reason.
    m2({"stage2_result": {
        "d_model": a.d_model, "layers": a.layers, "K": K, "steps": a.steps,
        "params_M": round(n_par2 / 1e6, 3), "seed": a.seed, "tag": a.tag or "",
        "z_mse_model": results.get("z_t+1", {}).get("mse_model"),
        "z_mse_persistence": results.get("z_t+1", {}).get("mse_persistence"),
        "chan_mse_model": results.get("chan_t+1", {}).get("mse_model"),
        "chan_mse_persistence": results.get("chan_t+1", {}).get("mse_persistence"),
        "rapid_r_deseas": results.get("rapid_probe", {}).get("r_deseasonalised"),
        "rapid_r_raw": results.get("rapid_probe", {}).get("r_raw"),
    }})
    suffix = f"_{a.tag}" if a.tag else ""
    results["seed"] = a.seed
    torch.save({"model": model.state_dict(), "args": vars(a),
                "step": a.steps, "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
                "torch_rng": torch.get_rng_state().numpy().tolist()},
               os.path.join(run_dir, f"temporal{suffix}.pt"))
    json.dump(results, open(os.path.join(run_dir, f"temporal{suffix}.json"), "w"), indent=2)
    print(f"saved {run_dir}/temporal{suffix}.pt")


if __name__ == "__main__":
    main()
