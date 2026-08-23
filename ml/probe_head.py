#!/usr/bin/env python3
"""The top rung of the probe ladder: a supervised attention head over the
UNPOOLED section.

Every other probe in this project — ridge, MLP, the stage-2 hidden-state
read-out — mean-pools the section's ~67 pixel embeddings into one vector
before looking at them. Geostrophic transport is the east-minus-west
density difference ACROSS the section; mean-pooling destroys precisely the
cross-pixel structure the physics lives in. This probe keeps every
(pixel, month) token and lets one cross-attention query learn what to pool.

Ladder semantics (each rung isolates one capability):
  ridge  — what is LINEARLY accessible in the pooled embedding
  mlp    — plus pointwise nonlinearity            (probe_kfold --probe mlp)
  head   — plus spatial structure across the section        (this file)
If head >> mlp, the embedding carries section-structure information the
pooled probes cannot reach. If head ~= mlp, pooling loses nothing and the
representation itself is the limit.

The codec stays FROZEN — this is still a probe, not fine-tuning; gradients
stop at the cached embeddings. Same year-blocked folds, same inner-tail
early stopping, 3 seeds averaged per fold; with n~240 and ~25k parameters
the head is regularized hard (weight decay 1e-2, dropout on tokens).

Since 2026-08-21 this is the VERDICT read-out, not the top rung of a ladder
whose lower rungs are quoted (ml/CLAUDE.md §3, Chris: "we should not do pooled
evals anywhere"). Two consequences live in this file:

  · `--target` — every series ml/probe_kfold.py scores (rapid, fc, move,
    osnap, samba) now has an unpooled read-out. It was RAPID-only, so for
    four of the five the pooled ridge was the only number in existence.
  · `--raw --wind-only` — the UNPOOLED WIND BAR. Both sides of a comparison
    switch together; scoring an unpooled head against probe_kfold's POOLED
    wind baseline would manufacture a margin out of the read-out.

Usage:
  python3 ml/probe_head.py --run global14 --data ml/cache/na_pixels_c14_global.npz
  python3 ml/probe_head.py --run pixel25_40k --data ml/cache/na_pixels_c25_global.npz --K 3
  python3 ml/probe_head.py --run actions --data ... --target move
  python3 ml/probe_head.py --run actions --data ... --raw --raw-patch   # 3x3 control
  python3 ml/probe_head.py --run actions --data ... --raw --wind-only   # the bar
  python3 ml/probe_head.py --run global14 --data ... --head-device cpu

The codec/embedding pass always uses the GPU when there is one; --head-device
only says where the read-out TRAINS, and `auto` decides that with a real
SectionHead training step before the embedding starts (see `_usable_device`).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE, LazyPixels, codec_from_ckpt
from trainprobe import anomaly_transform
from temporal import embed_everything, section_of
# ONE definition of what a target's section and label series ARE.
# probe_kfold owns them; a second copy here would be a second way of
# deciding which rows a target has, and the pooled and unpooled
# numbers would then be scored on different months while looking
# directly comparable — the precise failure this file exists to expose.
from probe_kfold import TARGETS, target_series

HERE = os.path.dirname(os.path.abspath(__file__))


class SectionHead(nn.Module):
    """One learned query cross-attends over (pixel x month) tokens.

    Deliberately the SMALLEST architecture that can express 'compare the
    two ends of the section': tokens get a linear lift plus a longitude
    encoding (so 'east' and 'west' are distinguishable after attention),
    one single-head cross-attention pools them, a two-layer MLP reads the
    pooled vector out. ~25k parameters at d=64."""

    def __init__(self, in_dim, d=64, K=1, n_blocks=0):
        super().__init__()
        self.lift = nn.Linear(in_dim, d)       # features + (lon_frac, month_idx/K)
        # Optional pre-pooling self-attention blocks over the tokens — the
        # capacity axis for the parameter-scaling test. n_blocks=0 is the
        # original ~23k head; each block at d=128 adds ~200k parameters.
        self.blocks = None
        if n_blocks:
            layer = nn.TransformerEncoderLayer(
                d, max(1, d // 64), dim_feedforward=4 * d,
                batch_first=True, norm_first=True, dropout=0.1)
            self.blocks = nn.TransformerEncoder(layer, n_blocks)
        self.q = nn.Parameter(torch.randn(1, 1, d) / d ** 0.5)
        self.att = nn.MultiheadAttention(d, num_heads=1, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 32),
                                 nn.GELU(), nn.Linear(32, 1))
        self.drop = nn.Dropout(0.1)

    def forward(self, tok):                     # tok [B, P*K, d_z+2]
        h = self.drop(self.lift(tok))
        if self.blocks is not None:
            h = self.blocks(h)
        pooled, _ = self.att(self.q.expand(len(tok), -1, -1), h, h)
        return self.out(pooled[:, 0]).squeeze(-1)


def _selftest_step(dev, in_dim=8, d=16):
    """ONE full training step of a tiny SectionHead on `dev`: forward,
    backward, optimiser step. Separated out so a test can force it to raise.

    The shapes are irrelevant and deliberately minimal — what this exercises
    is the DISPATCH, i.e. which kernels the cross-attention forward and
    backward select on this device, on this box, with this torch build. That
    is the only thing `_usable_device` can decide from the inputs alone."""
    net = SectionHead(in_dim, d=d).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    tok = torch.randn(2, 3, in_dim, device=dev)
    y = torch.zeros(2, device=dev)
    loss = (net(tok) - y).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()


def _usable_device(pref):
    """The device the READ-OUT can actually train on, decided in the first
    second of the job instead of after the 13-minute embedding pass.

    Run #397 (2026-08-18) proved out both probe_head invocations the
    expensive way: full anomaly transform plus the entire 3142-month
    embedding, ~13 minutes each, and then, on the FIRST `loss.backward()` in
    fold_fit,

        RuntimeError: Failed to find C compiler. Please specify via CC
        environment variable or set triton.knobs.build.impl

    — the cross-attention backward dispatches to a Triton-JIT kernel, Triton
    builds its CUDA-utils C extension on first use, and that Vast box had no
    C compiler and no CC. Both calls are wrapped `|| echo "::warning::..."`
    in scripts/probes_run.sh, so the run went green with no head number: the
    second consecutive failure to produce the one read-out that is primary at
    pentad cadence (ml/CLAUDE.md §3), after #392's OOM.

    `torch.cuda.is_available()` was TRUE on that box — it is the wrong
    question. This runs the same forward AND backward that failed, so a
    device that cannot finish a training step is discovered while the inputs
    are all it has cost us (ml/CLAUDE.md §0.3 / §5.16), and ANY exception
    means CPU rather than no number at all: a slower read-out is a result, a
    fallen-over one is not.

    The global RNG is saved and restored around the probe, so the fold
    numbers stay a function of the data and the seed and never of whether
    this self-test ran."""
    dev = pref if isinstance(pref, torch.device) else torch.device(pref)
    if dev.type == "cpu":
        return dev
    state = torch.get_rng_state()
    try:
        _selftest_step(dev)
    except Exception as e:              # noqa: BLE001 — ANY failure means CPU
        msg = " ".join(str(e).split())
        print(f"head read-out: {dev.type} FAILED its self-test, falling back "
              f"to CPU — one SectionHead train step raised "
              f"{type(e).__name__}: {msg[:200]}"
              f"{'...' if len(msg) > 200 else ''}")
        return torch.device("cpu")
    finally:
        torch.set_rng_state(state)
    return dev


def fold_fit(Xtr, ytr, Xte, in_dim, seed, steps=4000, d=64, n_blocks=0):
    # THE READ-OUT TRAINS WHERE ITS DATA IS, and main() now decides where that
    # is BEFORE embedding (`--head-device`, `_usable_device`). Following Xtr's
    # device keeps this function correct whichever way the caller supplies the
    # tokens; it does not, by itself, make any device work.
    #
    # The #116 context, corrected. #116's tail was 96% CPU / 0% GPU because
    # codec.to(_dev) moved the EMBEDDING pass to the GPU while this function
    # stayed on the CPU tokens it was handed. Moving the tokens to the GPU was
    # a real fix for that — but the GPU path it opened had never once executed
    # end to end, and the claim that it had is what this comment used to
    # assert. Two bugs sat in it: the cross-attention BACKWARD needs a Triton
    # JIT build, which failed on the box in #397 (see `_usable_device`), and
    # the return below reached `.numpy()` on a CUDA tensor, which raises
    # `TypeError: can't convert cuda:0 device type tensor to numpy` — never
    # observed only because the backward died first.
    dev = Xtr.device
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = SectionHead(in_dim, d=d, n_blocks=n_blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(Xtr)
    fit = slice(0, int(0.8 * n))
    val = slice(int(0.8 * n), n)
    Xf = Xtr[fit]; yf = torch.as_tensor(ytr[fit], dtype=torch.float32).to(dev)
    Xv = Xtr[val]; yv = torch.as_tensor(ytr[val], dtype=torch.float32).to(dev)
    best, best_state, patience = np.inf, None, 0
    for s in range(steps):
        k = torch.randint(0, len(Xf), (min(32, len(Xf)),), generator=g)
        loss = (net(Xf[k]) - yf[k]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0:
            net.eval()
            with torch.no_grad():
                v = (net(Xv) - yv).pow(2).mean().item()
            net.train()
            if v < best - 1e-6:
                best, patience = v, 0
                best_state = {k2: v2.clone() for k2, v2 in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        # .cpu() is a no-op on a CPU tensor and the only way off a CUDA one:
        # `.numpy()` on cuda:0 raises TypeError. This line is why the GPU path
        # could not have worked even had the backward built.
        return net(Xte).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    ap.add_argument("--K", type=int, default=1,
                    help="months of context per sample (1 = instantaneous)")
    ap.add_argument("--head-dim", type=int, default=64,
                    help="head width (64 = the ~23k original)")
    ap.add_argument("--head-blocks", type=int, default=0,
                    help="pre-pooling self-attention blocks (0 = original)")
    ap.add_argument("--head-device", choices=("auto", "cpu", "cuda"),
                    default="auto",
                    help="where fold_fit trains the read-out. auto = try cuda "
                         "when it is available and fall back to CPU if a real "
                         "SectionHead train step fails there (run #397: no C "
                         "compiler for Triton's JIT); cpu = force CPU, no "
                         "self-test; cuda = ask for cuda, still self-tested. "
                         "The codec/EMBEDDING pass is on the GPU either way — "
                         "this flag only moves the read-out training, so the "
                         "device is reversible from the dispatch.")
    ap.add_argument("--raw-patch", action="store_true",
                    help="with --raw: raw tokens carry the 3x3 neighbourhood "
                         "(matches the patch codec's receptive field)")
    ap.add_argument("--raw", action="store_true",
                    help="the END-TO-END BASELINE: feed the head the raw "
                         "anomaly channel values (+observed flags) of each "
                         "section pixel instead of the codec embedding. Same "
                         "head, same folds, same regularization — the ONLY "
                         "difference is whether self-supervised pretraining "
                         "sits between the data and the read-out. If the "
                         "embedding does not beat this, the codec added "
                         "nothing a supervised head could not learn from "
                         "240 months alone. NOTE: raw tokens see one pixel; "
                         "a patch=3 codec's embedding saw its 3x3 "
                         "neighbourhood — pair raw against a PIXEL codec for "
                         "the strictly matched comparison.")
    ap.add_argument("--target", default="rapid", choices=sorted(TARGETS),
                    help="which transport series to probe. This file was "
                         "hardcoded to RAPID (rapid_section() and d['rapid']) "
                         "until 2026-08-21, so fc/move/osnap/samba — every "
                         "other target ml/probe_kfold.py scores — had NO "
                         "unpooled path at all, and the only number available "
                         "for them was the section-mean ridge the programme "
                         "has stopped treating as a verdict. Sections and "
                         "label decoding come from probe_kfold.TARGETS and "
                         "probe_kfold.target_series, so the unpooled read-out "
                         "is scored on exactly the rows the pooled one was. "
                         "A section outside the tensor window is refused by "
                         "name, as probe_kfold skips it by name.")
    ap.add_argument("--wind-only", action="store_true",
                    help="with --raw: the UNPOOLED WIND BAR. Raw tokens carry "
                         "tau_x/tau_y only, so this is probe_kfold's "
                         "wind_only_baseline with the section mean removed "
                         "and nothing else changed. It exists because BOTH "
                         "SIDES OF A COMPARISON MUST SWITCH TOGETHER: an "
                         "unpooled codec head scored against a POOLED wind "
                         "bar manufactures a margin out of the read-out. NOT "
                         "numerically comparable to the pooled bar — these "
                         "tokens are anomaly-transformed, exactly as the "
                         "codec's inputs are, where probe_kfold's bar reads "
                         "the raw tau channels. It is the matched bar for the "
                         "head, not a re-measurement of the old one.")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="the three per-fold seeds are (base, base+1, base+2). "
                         "This exists because run #116 was dispatched as 'head "
                         "probe, seed B' and returned 0.662 / [0.557, 0.745] / "
                         "2.10 Sv — bit-identical to seed A, because there was "
                         "no seed knob at all and the seeds were hardwired to "
                         "(0,1,2). It reproduced the estimator instead of "
                         "resampling it. --seed-base 3 gives a genuinely "
                         "independent draw; the file name carries it so two "
                         "draws cannot overwrite each other.")
    a = ap.parse_args()
    if a.wind_only and not a.raw:
        # A precondition that depends only on the inputs, checked while the
        # inputs are all it has cost us (ml/CLAUDE.md §5.16). --wind-only
        # selects a subset of the RAW channels; without --raw the tokens are
        # codec embeddings and the flag would silently do nothing, which is
        # the one failure mode this repository refuses everywhere else.
        sys.exit("--wind-only requires --raw: it selects raw tau channels, "
                 "and on an embedding there is nothing for it to select. The "
                 "unpooled wind bar is `--raw --wind-only`.")
    if a.wind_only and a.raw_patch:
        sys.exit("--wind-only with --raw-patch is not the bar: probe_kfold's "
                 "wind baseline reads one pixel's tau, pooled. Drop "
                 "--raw-patch, or say in the dispatch what the 3x3 wind "
                 "control is a control FOR.")
    spec = TARGETS[a.target]

    # FIRST, before the checkpoint, the anomaly transform and the ~13-minute
    # embedding pass: decide where the read-out can train. This is a
    # precondition that depends only on the inputs, so it is checked while the
    # inputs are all it has cost us (ml/CLAUDE.md §0.3 / §5.16) — #397 spent
    # two full embedding passes to learn it, twice, and reported success.
    if a.head_device == "cpu":
        head_dev = torch.device("cpu")
        print("head read-out on cpu (--head-device cpu; no self-test)")
    else:
        want = ("cuda" if (a.head_device == "cuda" or torch.cuda.is_available())
                else "cpu")
        head_dev = _usable_device(torch.device(want))
        why = ("no cuda on this box" if want == "cpu"
               else "passed the SectionHead forward+backward self-test"
               if head_dev.type == "cuda" else "self-test failed, see above")
        print(f"head read-out on {head_dev.type} ({why})")

    ck = torch.load(os.path.join(HERE, "runs", a.run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    # load_tensor == np.load for a single-file npz; for family 5's sidecar it
    # memory-maps X (see ml/tensor_io.py — 165.6 GB does not decompress). With
    # the bare np.load this script could not open a family-5 tensor AT ALL:
    # `KeyError: 'X is not a file in the archive'`.
    from tensor_io import load_tensor
    d = load_tensor(a.data)
    # ONE read of d["X"]. On an npz every subscript DECOMPRESSES the whole
    # member afresh, so `d["X"].shape[-1]` in the guard below cost a full
    # 33.1 GB materialisation (measured: +0.523 GiB on a 0.523 GiB fixture)
    # purely to read an integer, and the next line paid for it a second time.
    X = d["X"]
    if len(ck["chan"]) != X.shape[-1]:
        sys.exit(f"{a.run}: codec has {len(ck['chan'])} channels but the tensor "
                 f"has {X.shape[-1]} — pass --data with the matching tensor.")
    if isinstance(X, np.memmap) and not X.flags.writeable:
        # Sidecar tensor (family 5): anomaly_transform WRITES into X and
        # refuses a read-only map by design. The canonical map must never take
        # those writes either — a later run would z-score anomaly-space data
        # with nothing to say so. A per-run scratch copy is disk, not RAM
        # (tensor_io docstring). AFTER the channel guard, so a mismatched
        # tensor costs a message rather than a 166 GB copy (ml/CLAUDE.md §5.16).
        from tensor_io import writable_copy
        scratch = a.data[:-4] + "_head_scratch.npy"
        X = writable_copy(X, scratch, verbose=False)
        import atexit
        atexit.register(lambda p=scratch:
                        os.path.exists(p) and os.remove(p))
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    lats, lons = d["lats"], d["lons"]
    t_hold = np.array([m[:4] in set(ck["args"]["holdout_years"].split(","))
                       for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)
    Xa, _ = anomaly_transform(X, moy, t_hold, x_hold)
    del X               # transforms in place: Xa IS that buffer, nothing frees

    codec = codec_from_ckpt(ck, Xa.shape[-1])
    codec.load_state_dict(ck["model"])
    codec.eval()
    # Run the frozen codec on the GPU when there is one. embed_everything
    # moves each batch to the MODEL's device, so this one line is the whole
    # fix — and it is worth hours: the quarter-degree tensor has 84,405
    # ocean pixels over 516 months, i.e. ~43M encoder forwards of a 40.7M
    # model. train.py's in-training probe was moved to the GPU on
    # 2026-08-08; these standalone scripts were never given the same
    # treatment and quietly stayed on CPU, which is why a job with NO codec
    # training sat for two hours in the probe step.
    _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    codec.to(_dev)
    print(f"codec on {_dev.type}")
    # Derived PER BATCH, not materialised. This is where BOTH probe_head
    # invocations of run #392 were OOM-killed (2026-08-18), each ~112 s after
    # the "codec on cuda" line above, on family 4's [3142, 281, 481, 39]
    # float16 — the same allocation that took all three probes down in #388
    # and train.py in #365. probe_kfold, probe_sequence and dip_check were
    # converted to LazyPixels then; this file was missed, so it was the last
    # script in the ladder still building the eager pair, and it is the ONE
    # read-out Chris trusts at pentad cadence (ml/CLAUDE.md §3).
    #
    # THE ARITHMETIC, measured on a 0.523 GiB fixture and scaled by the real
    # element count (16,562,358,618):
    #     X                              33.1 GB   resident
    #     np.isfinite(Xa)                16.6 GB   resident
    #     np.nan_to_num(Xa, copy=False)  82.8 GB   TRANSIENT
    #                                   -------
    #                                   132.5 GB   peak
    # The third line is the surprise and it is why `copy=False` did not save
    # us: numpy's nan_to_num never copies the VALUES but its masked-copyto
    # form allocates full-size bools —
    #     idx_nan = isnan(d); idx_posinf = isposinf(d); idx_neginf = isneginf(d)
    # — and isposinf/isneginf each build isinf(d) and signbit(d) underneath,
    # so five [T,H,W,C] bools are live at once. Measured exactly 5.00x one
    # full bool (1.3082 GiB against 0.2615 GiB), i.e. 82.8 GB at pentad and
    # 414 GB at daily. No box we can rent survives that.
    #
    # Both arrays are elementwise pure functions of Xa and every consumer only
    # ever indexes a BATCH out of them, so evaluating them after the index is
    # arithmetically identical (LazyPixels in ml/model.py; ml/CLAUDE.md §4.1 —
    # remove the failure mode rather than guard it).
    #
    # THE `np.nan_to_num(Xa, copy=False)` LINE IS DELETED ON PURPOSE, and must
    # not come back. LazyPixels(Xa) fills each indexed batch, so filling Xa in
    # place would be redundant for the values and FATAL for the mask:
    # LazyPixels(Xa, obs=True) evaluates isfinite(Xa) per batch, and a
    # pre-filled Xa is finite everywhere. OBS would silently become all-True —
    # every land cell and every missing channel entering the encoder as an
    # observed 0.0 instead of a missing token, with no error and no NaN to
    # notice. `ocean` below is derived from OBS and would go all-True with it.
    Xt = LazyPixels(Xa)
    OBS = LazyPixels(Xa, obs=True)

    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    ocean = OBS[..., 0].any(axis=0).numpy()
    ys, xs = np.where(ocean)
    sec_y, sec_sel = section_of(lats, lons, ys, xs, spec["lat"], *spec["lon"])
    # argmin() clamps to the window edge, so a SAMBA request on a North
    # Atlantic tensor would silently probe the northern boundary row and
    # report a number. probe_kfold prints "section outside window, skipped"
    # and moves on; here the target IS the whole invocation, so REFUSE —
    # a probe with nothing to say must not write a file that looks like a
    # reading (ml/CLAUDE.md §5.22).
    if abs(float(lats[sec_y]) - spec["lat"]) > 1.0 or len(sec_sel) < 5:
        sys.exit(f"--target {a.target}: its section ({spec['lat']}N, "
                 f"{spec['lon'][0]}..{spec['lon'][1]}) is outside this "
                 f"tensor's window — nearest row {float(lats[sec_y]):.2f}N, "
                 f"{len(sec_sel)} ocean cells. ml/probe_kfold.py skips this "
                 f"target on this tensor too; there is no unpooled number to "
                 f"be had here and inventing a section would be worse than "
                 f"the pooled one.")
    if a.raw:
        # raw features per (pixel, month): C anomaly values (0 where
        # unobserved) + C observed flags — exactly what the encoder itself
        # receives for that pixel, minus the pretraining. With --raw-patch,
        # each token instead carries its 3x3 neighbourhood (2*C*9 features):
        # the same receptive field the patch codec had. This is the control
        # that decides whether the patch codec's 0.690 is PRETRAINING value
        # or merely RECEPTIVE-FIELD value.
        sy, sx = ys[sec_sel], xs[sec_sel]
        if a.raw_patch:
            from model import gather_px
            T_all = Xt.shape[0]
            feats = []
            for t in range(T_all):
                tt = torch.full((len(sy),), t, dtype=torch.long)
                v, o = gather_px(Xt, OBS, tt, torch.as_tensor(sy),
                                 torch.as_tensor(sx), 3)
                feats.append(torch.cat([v.reshape(len(sy), -1),
                                        o.reshape(len(sy), -1).float()], -1))
            Z = torch.stack(feats).numpy()
        else:
            v_ = Xt[:, sy, sx].numpy()
            o_ = OBS[:, sy, sx].numpy().astype(np.float32)
            if a.wind_only:
                # The UNPOOLED BAR. Same head, same folds, same section,
                # same anomaly space as the codec's inputs — the only thing
                # that differs from probe_kfold's wind_only_baseline is that
                # the section is NOT averaged away first.
                chan_names = [str(c) for c in d["chan"]]
                wsel = [i for i, c in enumerate(chan_names)
                        if c in ("tau_x", "tau_y")]
                if not wsel:
                    sys.exit("--wind-only: this tensor has no tau_x/tau_y "
                             "channel, so there is no wind bar to draw. "
                             f"channels: {chan_names}")
                v_, o_ = v_[..., wsel], o_[..., wsel]
            Z = np.concatenate([v_, o_], -1)
        feat_dim = Z.shape[-1]
    else:
        # E-047: a BLOCK codec embeds GRIDS, and embed_everything refuses
        # without the map (#448: every probe in that run raised, was caught
        # and skipped, and the run finished with no head number at all). The
        # map is rebuilt here from the tensor's own axis members, exactly as
        # ml/temporal.py's embed path rebuilds it.
        _bkw = {}
        if int(ck.get("args", {}).get("k_time", 1) or 1) > 1:
            import datetime as _dt
            from timeblocks import BlockAxis
            _tb = str(ck["args"].get("time_block", "") or "")
            if not _tb or "bin_index" not in d:
                raise SystemExit(
                    "probe_head: this codec is a BLOCK codec but its blocks "
                    "cannot be rebuilt (time_block "
                    f"{_tb!r}, bin_index present: {'bin_index' in d}). "
                    "Refusing to embed it one bin at a time.")
            _ba = BlockAxis(_tb, months, d["bin_index"],
                            _dt.date.fromisoformat(str(d["epoch"]))
                            if "epoch" in d else None,
                            int(np.asarray(d["pentad_days"]).item())
                            if "pentad_days" in d else None)
            _bkw = {"blk_rows": _ba.rows, "blk_pad": _ba.pad}
            ctx_all = _ba.ctx_phase()
        Z, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                                ys[sec_sel], xs[sec_sel], codec.d_z, **_bkw)
        feat_dim = codec.d_z
    P = Z.shape[1]
    lon_frac = ((lons[xs[sec_sel]] - lons[xs[sec_sel]].min())
                / max(1e-6, np.ptp(lons[xs[sec_sel]]))).astype(np.float32)

    # ONE decoder for the labels, shared with ml/probe_kfold.py. The four
    # lines this replaces read d["rapid"] directly and knew only the
    # (time-index, value) shape, which is RAPID's; families 4/5 write the
    # truth_* series onto the tensor's own axis and families 2/3 write
    # (YYYYMM, value) — so a target other than rapid decoded to nothing here.
    # Deseasonalisation is byte-for-byte the same operation it always was;
    # it now lives in one place instead of two.
    month_of_ym = {int(m[:4]) * 100 + int(m[5:7]): i
                   for i, m in enumerate(months)}
    ser = target_series(d, spec, moy, month_of_ym)
    if ser is None:
        sys.exit(f"--target {a.target}: this tensor carries no usable "
                 f"'{spec['key']}' series (absent, or under the 48-sample "
                 f"floor ml/probe_kfold.py applies). Run ml/fetch_truth.py, "
                 f"or probe a target this tensor actually has.")
    ridx, v_des = ser
    ok = ridx >= a.K - 1
    ridx, v_des = ridx[ok], v_des[ok]

    # tokens [n, P*K, d_z+2]: embedding + (lon position, month offset)
    toks = np.zeros((len(ridx), P * a.K, feat_dim + 2), dtype=np.float32)
    for i, t in enumerate(ridx):
        for j in range(a.K):
            z = Z[t - j]                              # [P, d_z]
            block = slice(j * P, (j + 1) * P)
            toks[i, block, :feat_dim] = z
            toks[i, block, feat_dim] = lon_frac
            toks[i, block, feat_dim + 1] = j / max(1, a.K - 1) if a.K > 1 else 0.0
    # ...and the tokens go to whichever device the read-out was CLEARED for at
    # the top of main(), which is not necessarily the codec's: fold_fit follows
    # Xtr.device, so this line is what decides where 4,000 optimiser steps per
    # fold happen. `_dev` (the codec) stays on the GPU regardless.
    T_ = torch.as_tensor(toks).to(head_dev)

    pred = np.full(len(v_des), np.nan)
    years = yr[ridx]
    for yy_ in np.unique(years):
        te = years == yy_
        tr = ~te
        # standardize target on train only
        mu, sd = v_des[tr].mean(), v_des[tr].std() + 1e-9
        p = np.mean([fold_fit(T_[tr], (v_des[tr] - mu) / sd, T_[te],
                              feat_dim + 2, sd_, d=a.head_dim,
                              n_blocks=a.head_blocks)
                     for sd_ in (a.seed_base, a.seed_base + 1,
                                 a.seed_base + 2)], axis=0)
        pred[te] = p * sd + mu
    okp = np.isfinite(pred)
    r = float(np.corrcoef(pred[okp], v_des[okp])[0, 1])
    rmse = float(np.sqrt(np.mean((pred[okp] - v_des[okp]) ** 2)))

    # block bootstrap over years, same as the ridge
    rng = np.random.default_rng(0)
    uy = np.unique(years)
    rs = []
    for _ in range(2000):
        pick = rng.choice(uy, len(uy), replace=True)
        sel = np.concatenate([np.where(years == p_)[0] for p_ in pick])
        if np.std(v_des[sel]) > 0:
            rs.append(np.corrcoef(pred[sel], v_des[sel])[0, 1])
    lo95, hi95 = np.percentile(rs, [2.5, 97.5])

    out = {"run": a.run,
           "head_dim": a.head_dim, "head_blocks": a.head_blocks,
           "probe": ("attention-head-raw3x3" if (a.raw and a.raw_patch)
                     else "attention-head-wind" if (a.raw and a.wind_only)
                     else "attention-head-raw" if a.raw
                     else "attention-head"), "K": a.K,
           # THE TARGET TRAVELS WITH THE NUMBER. Until 2026-08-21 this file
           # probed RAPID and only RAPID, so the field was implicit; now that
           # `fc` and `move` have unpooled read-outs, a bundle carrying two of
           # these files needs to say which is which, and a reader of an older
           # file needs "rapid" to be the answer it already was.
           "target": a.target,
           "pooled": False,
           "section": {"lat": spec["lat"], "lon": list(spec["lon"]),
                       "pixels": int(P)},
           "r_kfold_deseas": round(r, 3),
           "ci95": [round(float(lo95), 3), round(float(hi95), 3)],
           "rmse_sv": round(rmse, 2), "n": int(okp.sum()),
           "seed_base": a.seed_base,
           # The out-of-fold PREDICTIONS, not just their summary. Two probes
           # scored on the same months and the same year-blocks differ by a
           # PAIRED quantity, and a paired difference has a far tighter
           # interval than the gap between two independent CIs suggests:
           # head [0.557, 0.745] and raw-3x3 [0.514, 0.729] overlap almost
           # entirely, yet they are the same 240 months and share nearly all
           # of their error. Without the per-month values that comparison
           # cannot be made at all, and +0.034 stays unquotable forever.
           # scripts/paired_probe.py consumes exactly these three arrays.
           "pred": [round(float(v), 4) for v in pred],
           "target_sv": [round(float(v), 4) for v in v_des],
           "years": [int(v) for v in years],
           "note": "unpooled section: one query attends over "
                   f"{P} pixels x {a.K} months"}
    print(f"{a.run} head-probe ({out['probe']}, target {a.target}, "
          f"K={a.K}): unpooled k-fold r {r:+.3f} "
          f"[{lo95:+.3f}, {hi95:+.3f}] · RMSE {rmse:.2f} Sv")
    size = ("" if (a.head_dim == 64 and a.head_blocks == 0)
            else f"_d{a.head_dim}b{a.head_blocks}")
    # A second seed draw must not overwrite the first — that is how #116 came
    # to look like a confirmation of a number it had merely recomputed.
    if a.seed_base:
        size += f"_s{a.seed_base}"
    # rapid keeps the historical file names EXACTLY — probes-<n>.json bundles,
    # scripts/sweep_table.mjs, ml/make_table.py and status.html all address
    # these by name, and 183 archived bundles already use them. A non-default
    # target is a new file beside them, never a rename of one.
    tsfx = "" if a.target == "rapid" else f"_{a.target}"
    kind = ("raw3x3" if (a.raw and a.raw_patch)
            else "wind" if (a.raw and a.wind_only)
            else "raw" if a.raw else "")
    fn = f"probe_head{'_' + kind if kind else ''}{tsfx}{size}.json"
    path = os.path.join(HERE, "runs", a.run, fn)
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
