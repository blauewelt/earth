#!/usr/bin/env python3
"""Train PixelMAE on the North-Atlantic pixel tensor.

Splits are BLOCKED, not random (proposal §7): whole held-out YEARS and a
held-out lon/lat block, so spatial/temporal autocorrelation cannot fake skill.

Eval after training:
  · masked-channel reconstruction error on held-out years   (vs channel-mean)
  · temporal neighbour prediction (t+1) error               (vs persistence)
  · RAPID probe: ridge regression from the mean embedding of the 26.5N
    section to the RAPID overturning transport, fit on train years, scored
    (Pearson r) on held-out years. The transport was NEVER a channel.

Smoke (CPU, ~2 min):   python3 ml/train.py --smoke
Real  (Colab TPU/GPU): colab run --gpu v6e1 ml/train.py   (see ml/README.md)
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import torch

from model import (PixelMAE, gather_px, LazyPixels, obs_any_chunked,
                   pool_idx)
import fsq_ladder as fql

HERE = os.path.dirname(os.path.abspath(__file__))


def lon_holdout_mask(spec, lons):
    """The ONE reading of a `--holdout-lon` spec, shared with temporal.py.

    Returns a [W] bool: True where the longitude is EXCLUDED from training.

    `spec` is "lo,hi" (degrees east, half-open [lo, hi)) or — new on
    2026-08-19 — "" / "none", which holds out NO longitude at all. The empty
    mask is not a special case anywhere downstream and deliberately so:
      · anomaly_transform's `keep_x = ~x_hold` becomes all-True, so the
        z-score statistics are taken over all longitudes of the TRAIN YEARS.
        That is correct, and it IS a change — a codec trained under it sees a
        different input normalisation from the -45,-25 codecs.
      · the train pool `obs_any & ~t_hold & ~x_hold` collapses to
        `obs_any & ~t_hold`, and the validation pool
        `obs_any & (t_hold | x_hold)` to `obs_any & t_hold`.
      · trainprobe.probe_now's `ok_p = ~x_hold[kxs]` becomes all-True.
    None of those need an `if`; they need a mask that is honestly all-False,
    which is what this returns. A second parser for this string is how the
    anomaly transform ended up with four copies of one bug (see
    trainprobe.anomaly_transform) — so temporal.py imports this one.
    """
    s = "" if spec is None else str(spec).strip()
    if s == "" or s.lower() == "none":
        return np.zeros(len(lons), dtype=bool)
    lo, hi = (float(v) for v in s.split(","))
    return (np.asarray(lons) >= lo) & (np.asarray(lons) < hi)


def fit_schedule(s, steady_elapsed, total_elapsed, max_minutes, steps0):
    """How many steps fit the wall-clock budget, from the STEADY rate.

    Pure, so tests/test_max_minutes_refit.py can replay run #366's exact
    numbers against it. The three commitments (see the call site for the
    incident): the rate is `steady_elapsed / (s - 1)` — measured AFTER step 1,
    because step 1 carries one-time cost that is not a rate; the fit never
    exceeds the dispatched `steps0`, and never falls below `s + 1`; the budget
    spends from `total_elapsed`, step-1 cost included, because that time is
    genuinely gone. 15% of the remaining budget is held back for the probes.

    Returns (fit, rate).
    """
    rate = steady_elapsed / (s - 1)
    budget = max_minutes * 60 - total_elapsed
    fit = s + int(0.85 * budget / rate)
    return max(s + 1, min(steps0, fit)), rate


def _optint(v):
    """int, but an EMPTY string means "not set" rather than a parse error.

    The workflow always passes --d-model and friends; whether the dispatch
    filled them in is expressed by the value being empty. Without this,
    `--d-model ""` dies in argparse with "invalid int value: ''" — a refusal,
    but one whose message says nothing about what the operator should do.
    """
    v = (v or "").strip()
    return None if v == "" else int(v)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    p.add_argument("--out", default=os.path.join(HERE, "runs", "pilot"))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    # ARCHITECTURE: every one of these defaults to None, and None is FATAL
    # unless --resume supplies it or --smoke asks for the pilot. They used to
    # default to the 0.92M pilot (32/1/128/4/4/256), which nothing has trained
    # since run-62 — so the default was never the right answer, and omitting a
    # flag did not fail, it silently ran a DIFFERENT EXPERIMENT. Run #395 died
    # after 90 s with sixty "size mismatch ... [576] vs [128]" lines for
    # exactly that reason, and #387's 202M codec trained with n_heads=4
    # (head_dim 256) because the head count was the one field nobody restated.
    # A default that is never correct is not a default. See ml/CLAUDE.md §1.
    p.add_argument("--d-z", type=_optint, default=None)
    # None is in `choices` because the workflow passes --patch "" when the
    # dispatch did not set it, and argparse validates choices against the
    # PARSED value — so without None here an unset patch dies in argparse
    # ("invalid choice: None") instead of reaching the architecture check that
    # can actually explain itself.
    p.add_argument("--patch", type=_optint, default=None, choices=[1, 3, None],
                   help="encoder receptive field per channel token (3 = 3x3)")
    p.add_argument("--d-model", type=_optint, default=None,
                   help="encoder width (128x4 = the 0.92M pilot; 320x8 = ~10M "
                        "-- the Chinchilla-anchored size for the C=24 global "
                        "tensor: ~270M observed values / 20 ~ 13M params)")
    p.add_argument("--n-layers", type=_optint, default=None)
    p.add_argument("--n-heads", type=_optint, default=None,
                   help="attention heads. Keep d_model/n_heads (head_dim) in "
                        "64-128: the f3 anchor is 576/8 = 72. #387 ran "
                        "1024/4 = 256 and collapsed.")
    p.add_argument("--d-dec", type=_optint, default=None,
                   help="decoder width (scale with d_model; 512 at 320x8)")
    p.add_argument("--mask-ratio", type=float, default=0.5)
    # ---- E-019b: copy-reconstruction knobs (audit: ml/recon_eval.py) ------
    p.add_argument("--rec-w-visible", type=float, default=0.1,
                   help="loss weight of VISIBLE channels in the offset-0 "
                        "reconstruction (masked channels weigh 1.0). The "
                        "full-visibility round trip is the artefact stage 2 "
                        "and every probe consume, yet historically it was "
                        "trained only as this 0.1 side effect; E-019a "
                        "measured it losing 6.9%% of deep-T variance.")   # %% : argparse
                        # runs every help string through %-interpolation, and a bare
                        # `% o` is a valid octal conversion, so this one character
                        # made `train.py --help` raise TypeError for every flag.
    p.add_argument("--upweight-chans", default="",
                   help="regex over channel NAMES (e.g. "
                        "'rg_[ts](900|1100|1300|1500|1700|1900)'); matched "
                        "channels get --upweight in both recon and "
                        "neighbour losses. Refuses to run if it matches "
                        "nothing — a typo'd regex silently doing nothing is "
                        "the classic reports-success-does-nothing failure.")
    p.add_argument("--upweight", type=float, default=1.0,
                   help="loss multiplier for --upweight-chans channels")
    p.add_argument("--dec-layers", type=int, default=2,
                   help="decoder HIDDEN layers (2 = every pre-E-019b codec)")
    p.add_argument("--holdout-years", default="2009,2017,2023")
    # THE DEFAULT IS DELIBERATELY UNCHANGED (2026-08-19). Chris decided the
    # spatial holdout should stop being a standing training exclusion — hold
    # out YEARS only, train on every longitude — but flipping the argparse
    # default would silently change the meaning of every in-flight dispatch
    # and every replay of an old INPUTS_JSON block, none of which passes this
    # flag at all. The new regime is therefore OPT-IN and is selected by name:
    # `--holdout-lon none` (or ""), reached from a dispatch through
    # ml/recipes/*.json -> $RECIPE_HOLDOUT_LON -> this flag.
    p.add_argument("--holdout-lon", default="-45,-25",   # a mid-Atlantic block
                   help="longitude block excluded from TRAINING in both "
                        "stages, as 'lo,hi' in degrees east. Empty ('') or "
                        "'none' = NO spatial holdout: every longitude trains "
                        "and only the held-out YEARS are withheld. The "
                        "per-pixel anomaly transform already removes "
                        "location-memorised climatology, the block costs 25%% "
                        "of the training pixels, and it confounds the "
                        "spatial-coupling claim (about half of the stencil "
                        "head's measured advantage is earned patching the "
                        "hole in its own coords=(lat/90, lon/180) feature). "
                        "It stays AVAILABLE as a deliberate diagnostic — it "
                        "produced the paper's Figure 7 — it is just no "
                        "longer what a run gets for saying nothing. PASS IT "
                        "AS --holdout-lon=lo,hi, ONE WORD: a bare '-45,-25' "
                        "in the next argv slot starts with '-' and does not "
                        "match argparse's negative-number pattern, so it is "
                        "read as an option string and the process exits 2 "
                        "with 'expected one argument'.")
    p.add_argument("--time-block", default="",
                   help="E-047: embed a BLOCK of consecutive bins as ONE z "
                        "instead of one bin as one z. '' (default) is the "
                        "per-bin codec every archived checkpoint is, and is "
                        "bit-identical. 'month' groups the axis by calendar "
                        "month (a pentad month is 6 or 7 bins, k_max 7, short "
                        "months padded) so the resulting embedding axis is "
                        "MONTHLY while every input is 5-day; an integer N "
                        "takes fixed non-overlapping N-bin blocks; 'W/S' "
                        "(E-048) takes a width-W window advancing by S bins, "
                        "so 'W/W' is the integer mode and '6/3' is 30 days of "
                        "input every 15 days, consecutive embeddings sharing "
                        "3 of their 6 bins. S > W is REFUSED (it would leave "
                        "bins in no embedding). AT S < W THE PERSISTENCE "
                        "BASELINE IS STRONGER BY CONSTRUCTION — the previous "
                        "window overlaps this one — so a ratio against "
                        "persistence is comparable only with another roll at "
                        "the same stride; the axis says so at startup. The "
                        "encoder sees a k_max x C GRID of cells whose token "
                        "is chan_emb[c] + time_emb[j]; a padded or missing "
                        "cell rides the existing unobserved-cell path, which "
                        "is the point (Argo present in one bin of six stops "
                        "being a special case). ctx carries the CONTINUOUS "
                        "fraction-of-year phase of the block's centre. See "
                        "ml/plans/E047_block_codec.md")
    p.add_argument("--fsq-levels", default="",
                   help="E-046: quantize the codec's own bottleneck with FSQ "
                        "(arxiv 2309.15505) instead of leaving it continuous. "
                        "'' (default) is today's continuous bottleneck and is "
                        "BIT-IDENTICAL — no branch runs. A single integer N "
                        "puts N levels on EVERY d_z dimension (codebook "
                        "N^d_z); a comma list gives per-dimension counts and "
                        "its length must then equal --d-z exactly, or the run "
                        "refuses. L=2 is refused at any position: half = 0.5 "
                        "and offset = 0.5 make the bound's shift atanh(1) = "
                        "inf and every value collapses to one. The bound is "
                        "tanh on the raw pre-quantization activation with "
                        "sigma 1 — the paper's own form — so the SCALE is "
                        "learned into to_z rather than measured, which is the "
                        "one real difference from the head-side "
                        "--input-quant knob (ml/temporal.py, E-044c), where "
                        "sigma is measured off an existing Z. Gradients pass "
                        "straight through the round; there is no codebook, "
                        "so no commitment loss, no EMA and no dead-code "
                        "machinery is added. THE DISPATCHED ARM IS '8' AT "
                        "d_z 32 — [8]^32 — not the paper's 5-dimensional "
                        "option A: see ml/model.py's PixelMAE for the two "
                        "measurements that decide it. See "
                        "ml/plans/E046_fsq_codec.md.")
    p.add_argument("--fsq-ladder", default="uniform",
                   choices=["uniform", "exp", "auto"],
                   help="E-048: WHERE the --fsq-levels levels sit. 'uniform' "
                        "(the default) is E-046's evenly-spaced lattice and "
                        "is BIT-IDENTICAL to it — no branch runs. 'exp' puts "
                        "the same L levels on a symmetric GEOMETRIC ladder, "
                        "sign(v)*a*c^j with a*c^(n-1) = the same +/-2*sigma "
                        "saturation radius, assigned by nearest level in LOG "
                        "space; it replaces the tanh bound rather than "
                        "composing with it, because the tanh can only move "
                        "the boundaries and never the points (see "
                        "ml/fsq_ladder.py). 'auto' MEASURES the choice per "
                        "z-dimension: at --fsq-auto-step the run captures "
                        "--fsq-auto-n pre-quantization vectors, computes both "
                        "ladders' quantization MSE per dimension, keeps "
                        "uniform on a tie, and writes the fitted ladder into "
                        "the checkpoint (`fsq_ladder_fit`) — which every "
                        "loader rebuilds from and NONE re-fits. Chris asked "
                        "for this 'for each channel'; z-dimensions are not "
                        "channels, and per-dimension is the honest mapping "
                        "(ml/fsq_ladder.py says why).")
    p.add_argument("--fsq-exp-base", type=float, default=2.0,
                   help="the base c of the exponential ladder (--fsq-ladder "
                        "exp). c <= 1 is refused; so is a c so large the "
                        "innermost level falls below 1e-6 of the saturation "
                        "radius. Under 'auto' this is only the default the "
                        "fit starts from — the fitted base per dimension is "
                        "chosen from ml/fsq_ladder.py:AUTO_BASES and recorded.")
    p.add_argument("--fsq-auto-n", type=int, default=4096,
                   help="pre-quantization vectors the --fsq-ladder auto fit "
                        "measures on (one per pixel of one deliberate "
                        "forward, drawn from the TRAIN pool).")
    p.add_argument("--fsq-auto-step", default="2000",
                   help="the step(s) at which --fsq-ladder auto fits — one "
                        "number, or a COMMA LIST ('50,200,1000') to re-measure "
                        "and re-install at each of them, the LAST fit being "
                        "what the checkpoint carries. A list because the fit "
                        "now also measures the per-dimension SATURATION "
                        "RADIUS, and where `to_z`'s output sits is exactly "
                        "what moves during training. NOT 0: an untrained "
                        "encoder's activations are not the distribution the "
                        "ladder has to serve, and the whole point of 'auto' is "
                        "that it is measured. The FIRST entry is clamped to "
                        "steps//2 on a run too short to reach it, so an auto "
                        "run always fits and 'auto but never fitted' is not a "
                        "state a checkpoint can be in; later entries are "
                        "clamped to --steps and not pulled inward, because a "
                        "schedule that silently moves is worse than none.")
    p.add_argument("--fsq-ladder-fit", default="",
                   help="the per-dimension ladder, 'u,e2,e2,...' — normally "
                        "WRITTEN BY the run rather than passed. Pass it to "
                        "reproduce an exact lattice without re-measuring, or "
                        "let --resume adopt it from the checkpoint.")
    p.add_argument("--anomaly", action="store_true",
                   help="train dynamic channels as departures from their own "
                        "per-pixel monthly climatology (train years only)")
    p.add_argument("--eval-every", type=int, default=0,
                   help="every N steps, measure PREDICTIVE skill of the "
                        "frozen current embeddings (trainprobe.py: linear "
                        "section probe + mini temporal transformer) on the "
                        "blocked holdout; appends to <out>/metrics.jsonl. "
                        "MEASURED on #419's daily tensor (T=15,706): 1,683 s "
                        "per probe, of which 1,641 s is the ~864-pixel "
                        "embedding — ~371 s at pentad, ~96 s at monthly. This "
                        "is the EXPENSIVE cadence and it does NOT carry the "
                        "collapse guard's latency (the light probe fires that "
                        "too), so cut it hard: the workflow default is 25,000, "
                        "which is 9 full probes over a 200k run. "
                        "Requires --anomaly.")
    p.add_argument("--resume", default="",
                   help="continue a run from a checkpoint written by an "
                        "earlier job: path to a pixelmae.pt, or a bare tag "
                        "resolved under /opt/earth-cache/ckpt/<tag>.pt. "
                        "Restores weights, optimizer, LR schedule and the "
                        "step reached, then trains on to --steps. A "
                        "weights-only checkpoint (anything saved before "
                        "2026-08-08) still WARM-STARTS: it loads the weights "
                        "and restarts the schedule, which is stated in the "
                        "log so the run's provenance is never ambiguous. "
                        "Missing file = start fresh, loudly.")
    p.add_argument("--require-resume", action="store_true",
                   help="EXIT immediately if --resume finds no checkpoint, "
                        "instead of starting fresh. Checkpoint mirrors are "
                        "box-local, so a resume dispatch lands on the right "
                        "runner only by luck; without this a mislanded job "
                        "silently retrains from scratch for hours and calls "
                        "itself a continuation. Use it whenever the point of "
                        "the job is the EXISTING weights — e.g. a stage-2 "
                        "run over a frozen codec.")
    p.add_argument("--light-probe-every", type=int, default=0,
                   help="every N steps, run the CHEAP half of the probe (the "
                        "linear 26.5N section probe only — no mini temporal "
                        "transformer), over the RAPID timesteps thinned to "
                        "one sample per decorrelation time. THE OLD HELP HERE "
                        "SAID 'full ~300 s, light ~30 s' off a family-3 10M "
                        "codec and was wrong for every tensor in use. "
                        "Measured on #419 (daily, T=15,706, 38.0M codec): "
                        "full 2,296.6 s, light 611.3 s; after the hoisted "
                        "ocean mask, the ridx-only embedding and the thinning, "
                        "~1,683 s and ~17 s. Scaled from the same "
                        "decomposition: pentad ~371 s / ~22 s, monthly ~96 s "
                        "/ ~7 s. THIS is the cadence the collapse guard's "
                        "latency tracks — the guard fires on both probes, so "
                        "its worst case is 2 x this — which is why it stays at "
                        "2,000 while --eval-every goes to 25,000. Emits the "
                        "same linear_r_deseas key as the full probe (which "
                        "also emits linear_r_deseas_light, the identical "
                        "estimator on the identical subsample, so the two "
                        "curves are comparable by measurement). Requires "
                        "--anomaly; 0 disables.")
    p.add_argument("--lr-floor", type=float, default=0.0,
                   help="decay-then-constant schedule: cosine-decay the LR "
                        "over --lr-decay-steps to floor*peak, then hold it "
                        "there for the rest of the run. 0 (default) keeps "
                        "the pure cosine-to-zero schedule. The constant tail "
                        "exists to answer one question cheaply: is there "
                        "headroom left in simply not stopping? — watch the "
                        "probe curves and abort when they flatten.")
    p.add_argument("--lr-decay-steps", type=int, default=0,
                   help="steps of initial decay when --lr-floor > 0 "
                        "(default 0 = the full run, i.e. plain cosine)")
    p.add_argument("--max-minutes", type=int, default=0,
                   help="wall-clock budget for the TRAINING LOOP (0 = off). "
                        "After a short calibration the cosine schedule is "
                        "re-fitted to the step count that fits, so the LR "
                        "still anneals to zero inside the budget instead of "
                        "the job dying mid-schedule. Exists because Actions "
                        "kills at timeout-minutes with NO checkpoint: run "
                        "#12 (25 channels) measured ~1.3 steps/s against a "
                        "40k-step dispatch — 6 runner-hours, nothing saved.")
    p.add_argument("--collapse-r", type=float, default=0.05,
                   help="ABORT the run when the probe's linear_r_deseas falls "
                        "to or below this on --collapse-strikes consecutive "
                        "probes (0 = off). A codec whose embedding has stopped "
                        "carrying linearly decodable signal is dead, and it "
                        "burns a GPU at exactly the same rate as a live one.")
    p.add_argument("--collapse-strikes", type=int, default=2,
                   help="consecutive sub-threshold probes before aborting; 2 "
                        "so one bad probe cannot kill a healthy run")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    # `--resume !tag` is the same thing as `--resume tag --require-resume`.
    # It exists because workflow_dispatch allows at most 25 inputs and this
    # workflow is at the ceiling: adding a 26th made the FILE invalid, which
    # takes the whole workflow down — every dispatch 422s, not just the new
    # one. Encoding the flag in a string we already pass costs no input slot.
    if a.resume.startswith("!"):
        a.require_resume, a.resume = True, a.resume[1:]
    if a.smoke:
        # --smoke is the ONE place the 0.92M pilot architecture is still
        # spelled out, now that it is no longer anybody's silent default.
        a.steps, a.batch = 1500, 256
        for k, v in (("d_z", 32), ("patch", 1), ("d_model", 128),
                     ("n_layers", 4), ("n_heads", 4), ("d_dec", 256)):
            if getattr(a, k) is None:
                setattr(a, k, v)
    os.makedirs(a.out, exist_ok=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu")
    # load_tensor == np.load for every single-file npz (families 2/3/4,
    # unchanged); for family 5's sidecar layout it memory-maps X instead of
    # decompressing 165.6 GB into RAM no box has. See ml/tensor_io.py.
    from tensor_io import load_tensor, writable_copy
    d = load_tensor(a.data, allow_pickle=False)
    X, months = d["X"], [str(m) for m in d["months"]]
    if a.anomaly and isinstance(X, np.memmap) and not X.flags.writeable:
        # anomaly_transform writes into X in place — deliberate, and cheap for
        # an in-RAM array. A read-only map refuses it; an r+ map on the
        # CANONICAL tensor would succeed and leave anomaly-space data where
        # state-space data is documented, so the next run would z-score it
        # again. The scratch copy is the only correct option, and it is disk,
        # not RAM (chunked; the box needs the bytes free — 166 GB at daily,
        # which is why the family-5 box wants >= 400 GB).
        scratch = a.data[:-4] + "_scratch.npy"
        print(f"X is a read-only map ({X.nbytes / 2**30:.1f} GiB) — writable "
              f"scratch copy at {scratch} for the anomaly transform",
              flush=True)
        X = writable_copy(X, scratch)
        import atexit
        atexit.register(lambda: os.path.exists(scratch) and os.remove(scratch))
    lats, lons, chan = d["lats"], d["lons"], [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    print(f"X [T={T} H={H} W={W} C={C}] on {dev} · channels {chan}")

    # ---- blocked splits ----------------------------------------------------
    hold_years = set(a.holdout_years.split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    x_hold = lon_holdout_mask(a.holdout_lon, lons)
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    # Print what was held out EITHER WAY. "0/481 cols" is true but reads like
    # a bug; a run that trains on every longitude should say so in the words
    # a reader is looking for.
    _lonmsg = (f"held-out lon block {int(x_hold.sum())}/{W} cols "
               f"[{a.holdout_lon}]" if x_hold.any() else
               f"NO lon holdout — all {W} cols train (--holdout-lon "
               f"{a.holdout_lon!r})")
    print(f"held-out months {int(t_hold.sum())}/{T} · {_lonmsg} · "
          f"ocean {int(ocean.sum())}")
    # THE SPEC IS SAVED VERBATIM INTO ck["args"]["holdout_lon"] (save_ckpt
    # writes vars(a)), and TWELVE eval scripts under ml/ still read that
    # field as `lo, hi = (float(v) for v in ...split(","))`:
    # ablate_channels, dip_check, probe_head, probe_kfold, probe_sequence,
    # project_amoc, recon_decoder, recon_eval, rollout, rollout_spatial,
    # train_joint, trainprobe. A codec whose args say "none" trains fine, and
    # then EVERY ONE of them dies with `ValueError: could not convert string
    # to float: 'none'` — and probes_run.sh guards each with `|| echo
    # ::warning::`, so the run goes GREEN with an empty probe archive: §7's
    # "green run with no temporal.json" signature, from a new direction.
    # Say so HERE, where the inputs are all it has cost (§0.3), rather than
    # six hours later. `--holdout-lon=0,0` is the same empty mask ([0,0) is
    # the empty half-open interval) in a form all twelve can parse, and is
    # what ml/recipes/*-nolonhold.json pass for exactly this reason.
    if not x_hold.any() and a.holdout_lon.strip().lower() in ("", "none"):
        print(f"::warning:: --holdout-lon {a.holdout_lon!r} is saved verbatim "
              f"into this checkpoint's args, and the twelve eval scripts that "
              f"re-read that field parse it with float() — they will all "
              f"raise on it, each behind a best-effort guard, leaving a green "
              f"run with no probe results. Use --holdout-lon=0,0 for the "
              f"identical (empty) mask in a form they can read.", flush=True)

    # ---- anomaly space (proposal §5) ---------------------------------------
    # A reconstruction loss on raw state is dominated by the seasonal cycle —
    # the easiest, least interesting signal, and the sequence-probe experiment
    # showed the resulting embeddings are nearly redundant month to month.
    # Subtract each pixel's own monthly climatology (TRAIN years only, so the
    # holdout stays clean) from every DYNAMIC channel; what remains is the
    # anomaly — the part that carries trends, events, and the AMOC story.
    # Channels with no temporal variance (baked climatologies) are context,
    # not targets in disguise: they pass through unchanged.
    if a.anomaly:
        moy = np.array([int(m[5:7]) - 1 for m in months])
        # This was a hand-inlined SECOND copy of trainprobe.anomaly_transform,
        # and two copies of one transform are two places for the same
        # numerical bug to live. On 2026-08-17 they were exactly that: both
        # z-scored with numpy's default accumulator, which for float16 sums
        # ~204M squared residuals past 65504, returns inf, and drives every
        # dynamic channel to 0.0. Family 4 is the first float16 tensor, so the
        # duplicate would have had to be found and fixed twice. It is gone;
        # this is the one implementation, and it is the same one probe_kfold
        # scores against — which is what makes E-038's frozen control
        # comparable to the trained arms at all.
        #
        # "The one implementation" was still not TRUE when that was written:
        # temporal.py and probe_sequence.py each carried a THIRD and FOURTH
        # copy, both at the broken pre-2026-08-17 arithmetic, found on
        # 2026-08-18. Both are calls now, and the claim is pinned rather than
        # asserted: tests/test_one_anomaly_transform.py fails if any file
        # under ml/ re-implements the transform.
        from trainprobe import anomaly_transform   # lazy: plain runs skip it
        X, dynamic = anomaly_transform(X, moy, t_hold, x_hold)
        print(f"anomaly space: {len(dynamic)}/{C} dynamic channels "
              f"({[chan[c] for c in dynamic]})")

    # ---- E-047 time blocks ------------------------------------------------
    # AFTER the anomaly transform, never before: the climatology and the
    # per-channel z-score are properties of the SOURCE axis, and computing
    # them per block would change the statistics a frozen encoder is later
    # asked to reproduce (the same argument --time-stride carries in
    # ml/temporal.py). The blocks are a view of the transformed tensor.
    BLK = None
    if a.time_block:
        from timeblocks import BlockAxis
        BLK = BlockAxis(a.time_block, months,
                        d["bin_index"] if "bin_index" in d else None,
                        (dt.date.fromisoformat(str(d["epoch"]))
                         if "epoch" in d else None),
                        (int(np.asarray(d["pentad_days"]).item())
                         if "pentad_days" in d else None))
        a.k_time = int(BLK.k_max)
        a.ctx_mode = "block_phase"
        print(BLK.describe(C, a.d_z), flush=True)
        blk_rows = torch.as_tensor(BLK.rows.astype(np.int64))
        blk_pad = torch.as_tensor(BLK.pad)
        blk_ctx = BLK.ctx_phase()                       # [n_blocks, 2]
        # A BLOCK is held out iff ANY of its real bins is: a block that mixes
        # a holdout bin with train bins would leak the holdout into training
        # through the same embedding, which is the one thing the year-blocked
        # split exists to prevent.
        blk_hold = np.array([t_hold[BLK.rows[b, :int(BLK.n_bins[b])]].any()
                             for b in range(BLK.n_blocks)])
        print(f"  blocks held out: {int(blk_hold.sum())}/{BLK.n_blocks} "
              f"(any bin of the block held out makes the block held out)",
              flush=True)
    else:
        a.k_time = 1
        a.ctx_mode = "month_sincos"

    # train pool: any (t, y, x) with ≥2 observed channels, outside holdouts
    #
    # Chunked and int32 — identical values and identical order, a fraction of
    # the memory. The one-liner this replaces was measured as the trainer's
    # PEAK (a [T,H,W,C] bool + a [T,H,W] int64, live together), and the pool
    # itself is the largest thing that stays resident after it. See
    # obs_any_chunked / pool_idx in ml/model.py and
    # tests/test_train_pool_memory.py, which pins both against the originals.
    obs_any = obs_any_chunked(X)
    tt, yy, xx = pool_idx(obs_any & ~t_hold[:, None, None] & ~x_hold[None, None, :])
    vt, vy, vx = pool_idx(obs_any & (t_hold[:, None, None] | x_hold[None, None, :]))
    print(f"train pixels {len(tt):,} · held-out pixels {len(vt):,}")

    # Derived PER BATCH, not materialised: eagerly these two cost 49.7 GB
    # on top of X's 33.1 GB and OOM-killed run #365 on a 64 GB box (exit 137).
    # See LazyPixels in ml/model.py; arithmetic is unchanged.
    Xt = LazyPixels(X)
    OBS = LazyPixels(X, obs=True)
    mvec = np.array([int(m[5:7]) - 1 for m in months])
    ctx_all = np.stack([np.sin(2 * np.pi * mvec / 12), np.cos(2 * np.pi * mvec / 12)], 1)
    if BLK is not None:
        # The pool is over (BLOCK, pixel). A block is poolable where any of
        # its real bins is — obs_any is already the per-bin ">= 2 observed
        # channels" test, so the block-level test is "some bin qualifies",
        # which is exactly what a grid with unobserved cells is for.
        b_any = np.stack([
            obs_any[BLK.rows[b, :int(BLK.n_bins[b])]].any(axis=0)
            for b in range(BLK.n_blocks)])
        tt, yy, xx = pool_idx(b_any & ~blk_hold[:, None, None]
                              & ~x_hold[None, None, :])
        vt, vy, vx = pool_idx(b_any & (blk_hold[:, None, None]
                                       | x_hold[None, None, :]))
        ctx_all = blk_ctx
        print(f"block pool: train {len(tt):,} · held-out {len(vt):,} "
              f"(over {BLK.n_blocks} blocks)", flush=True)

    def batch(idx_t, idx_y, idx_x, n):
        k = np.random.randint(0, len(idx_t), n)
        t, y, x = idx_t[k], idx_y[k], idx_x[k]
        ctx = np.concatenate([ctx_all[t], (lats[y] / 90)[:, None], (lons[x] / 180)[:, None]], 1)
        # dtype=torch.long EXPLICITLY: the pool arrays are int32 now (half the
        # bytes, see pool_idx), and torch's advanced indexing has historically
        # accepted only long/byte/bool. Naming it here keeps every tensor that
        # leaves this function bit-identical to what the int64 pool produced,
        # so nothing downstream — gather, gather_px, the neighbour offsets —
        # can behave differently. The cast is n elements, not T·H·W.
        return (torch.as_tensor(t, dtype=torch.long),
                torch.as_tensor(y, dtype=torch.long),
                torch.as_tensor(x, dtype=torch.long),
                torch.as_tensor(ctx, dtype=torch.float32))

    # ---- per-channel loss weights (E-019b) --------------------------------
    cw = np.ones(C, np.float32)
    if a.upweight_chans:
        import re as _re
        hit = [c for c in range(C) if _re.fullmatch(a.upweight_chans, chan[c])]
        if not hit:
            raise SystemExit(f"--upweight-chans {a.upweight_chans!r} matches "
                             f"no channel in {chan} — refusing (a silent "
                             f"no-op here is a fake experiment)")
        cw[hit] = a.upweight
        print(f"upweight ×{a.upweight}: {[chan[c] for c in hit]}")
    cwt = torch.as_tensor(cw, device=dev)

    # ---- ARCHITECTURE RESOLUTION (2026-08-18) ----------------------------
    # DERIVE, DON'T RESTATE. A checkpoint already knows its own architecture:
    # save_ckpt() writes vars(a) into ck["args"]. Until now the model was
    # built from the CLI flags and the checkpoint was loaded into it 200 lines
    # later, so a dispatch that named a checkpoint but not its width built the
    # WRONG model and died in load_state_dict — #395, sixty size-mismatch
    # lines, 90 seconds. Restating a fact the file already holds is redundant
    # data entry, and redundant data entry is where the drift lives.
    #
    # So: resolve --resume FIRST, adopt the checkpoint's architecture for any
    # field the dispatch left unset, and REFUSE when the dispatch states one
    # that contradicts the file. Then refuse again if anything is still unset.
    # Both refusals cost seconds; the failures they replace cost hours.
    CKPT_DIR = "/opt/earth-cache/ckpt"

    def _resume_candidates(spec):
        cands = [c.strip() for c in spec.split(",") if c.strip()]
        for c in cands:
            pth = c if os.path.sep in c else os.path.join(CKPT_DIR, c + ".pt")
            if os.path.exists(pth):
                return pth, cands
        first = cands[0] if cands else ""
        return (first if os.path.sep in first
                else os.path.join(CKPT_DIR, first + ".pt")), cands

    RESUME_PATH, RESUME_CANDS, RESUME_CK = None, [], None
    if a.resume:
        RESUME_PATH, RESUME_CANDS = _resume_candidates(a.resume)
        if os.path.exists(RESUME_PATH):
            # ONE load, to CPU, reused by the resume block below. Loading to
            # CPU rather than the device also halves peak memory on a 2.5 GB
            # checkpoint; load_state_dict copies across devices anyway.
            RESUME_CK = torch.load(RESUME_PATH, map_location="cpu",
                                   weights_only=False)

    # `--resume !tag` on a box that does not hold the checkpoint has its own,
    # much more useful message, and it lives in the resume block below. Raise
    # it HERE too, because the architecture check now runs first and would
    # otherwise answer "no architecture" — true, but it would send the reader
    # after the wrong problem: the dispatch is fine, the box is wrong.
    if a.require_resume and RESUME_CK is None:
        raise SystemExit(
            f"--require-resume: no checkpoint at {RESUME_PATH}. This box is "
            f"not the one that wrote it (checkpoint mirrors are box-local). "
            f"Exiting in seconds rather than retraining from scratch for "
            f"hours under a doc string that claims to be a continuation.")

    # `fsq_levels` is an ARCHITECTURE field, not a training knob: it decides
    # what a `z` IS, every downstream consumer reads it back out of the
    # checkpoint, and a resume that contradicted it would continue a
    # quantized codec as a continuous one (or the reverse) while every tensor
    # in the state_dict loaded cleanly. So it rides the same adopt-or-refuse
    # machinery as d_z and the widths — E-046.
    # E-048 adds the two ladder fields for the same reason E-046 added
    # fsq_levels: they decide what a `z` IS. `fsq_ladder_fit` is NOT here —
    # it is a MEASUREMENT the run made rather than a setting a dispatch
    # states, so it is adopted unconditionally below instead of being
    # contradicted.
    ARCH = ("d_z", "patch", "d_model", "n_layers", "n_heads", "d_dec",
            "dec_layers", "fsq_levels", "fsq_ladder", "fsq_exp_base")
    if RESUME_CK is not None:
        ca = RESUME_CK.get("args", {}) or {}
        ca = ca if isinstance(ca, dict) else vars(ca)
        adopted, clash = [], []
        for k in ARCH:
            # d_z is also a top-level key on every checkpoint ever written;
            # args is the newer home. Prefer args, fall back to the old key.
            want = ca.get(k)
            if want is None and k == "d_z":
                want = RESUME_CK.get("d_z")
            if want is None:
                continue
            have = getattr(a, k)
            # "Did the dispatch ASK for this?" is not the same question as "is
            # it non-None". --dec-layers still carries a real default of 2
            # (every pre-E-019b codec), so a resume from one of E-019b's
            # 3-hidden-layer decoders would read have=2, want=3 and refuse a
            # dispatch that never expressed an opinion — the false-refusal
            # twin of the bug this block exists to kill. Ask argv instead.
            # BOTH SPELLINGS. argparse accepts `--flag value` and
            # `--flag=value`, and the workflow writes the =form for any value
            # that could open with a minus (--holdout-lon) and now for
            # --fsq-levels too. Matching only the bare token would read an
            # explicit `--fsq-levels=8` as "the dispatch said nothing", adopt
            # the checkpoint's value over it, and run a different experiment
            # in silence — the precise failure this block exists to remove.
            _fl = "--" + k.replace("_", "-")
            explicit = any(t == _fl or t.startswith(_fl + "=")
                           for t in sys.argv)
            if have is None or not explicit:
                if have != want:
                    setattr(a, k, want)
                    adopted.append(f"{k}={want}")
            elif have != want:
                clash.append(f"    {k}: dispatch says {have}, "
                             f"checkpoint holds {want}")
        if clash:
            raise SystemExit(
                "REFUSING to resume: the dispatch contradicts the "
                f"checkpoint's own architecture.\n{chr(10).join(clash)}\n"
                f"  checkpoint: {RESUME_PATH}\n"
                "  Either drop the contradicting flags and let the checkpoint "
                "supply them, or resume from a different checkpoint. Loading "
                "anyway is how #395 spent 90 s printing size mismatches.")
        if adopted:
            print(f"  architecture ADOPTED from {os.path.basename(RESUME_PATH)}: "
                  + " ".join(adopted), flush=True)
        # THE FITTED LADDER IS ALWAYS ADOPTED (E-048). A resumed `auto` run
        # must continue with the lattice it already measured, not measure a
        # second one against a half-trained encoder and change what its own
        # earlier steps meant. The only way to re-fit is to start a fresh run.
        _fit = str(ca.get("fsq_ladder_fit", "") or "")
        if _fit and _fit != a.fsq_ladder_fit:
            a.fsq_ladder_fit = _fit
            print(f"  FSQ ladder fit ADOPTED from "
                  f"{os.path.basename(RESUME_PATH)} ({_fit.count('e')} of "
                  f"{_fit.count(',') + 1} dimensions exponential) — a resume "
                  f"never re-fits", flush=True)

    missing = [k for k in ARCH if getattr(a, k) is None]
    if missing:
        raise SystemExit(
            "REFUSING to train: no architecture. Unset: "
            + ", ".join("--" + m.replace("_", "-") for m in missing) + ".\n"
            "  These no longer default to the 0.92M pilot, because that "
            "default was never the right answer and omitting a flag used to "
            "run a different experiment in silence.\n"
            "  Name a recipe (window: recipe:<name>, see ml/recipes/), resume "
            "a checkpoint that carries the architecture, or pass every flag.")

    model = PixelMAE(n_chan=C, d_z=a.d_z, patch=a.patch, d_model=a.d_model,
                     k_time=a.k_time,
                     n_layers=a.n_layers, n_heads=a.n_heads, d_dec=a.d_dec,
                     dec_layers=a.dec_layers, fsq_levels=a.fsq_levels,
                     fsq_ladder=a.fsq_ladder, fsq_exp_base=a.fsq_exp_base,
                     fsq_ladder_fit=a.fsq_ladder_fit).to(dev)
    print(f"codec parameters: {sum(p_.numel() for p_ in model.parameters())/1e6:.2f}M")
    # E-046: the codebook, logged ONCE at startup, from the object that will
    # actually do the rounding rather than from the flag string. A bits/dim
    # line nobody can trace back to a live quantizer is a claim; this is a
    # reading. (Nothing prints at all with the flag off — the log of a
    # continuous run is unchanged too.)
    if model.fsq is not None:
        _q = model.fsq
        print(f"FSQ bottleneck: --fsq-levels {_q.spec} -> "
              f"{[int(v) for v in _q.levels.tolist()][:8]}"
              f"{'...' if a.d_z > 8 else ''} · {_q.bits_per_dim:.3f} bits/dim "
              f"x d_z {a.d_z} = {_q.codebook_log2:.1f} bits "
              f"(codebook 2^{_q.codebook_log2:.1f}) · bound tanh, sigma 1 "
              f"(the scale is learned into to_z) · straight-through gradient, "
              f"no codebook parameters and no commitment loss", flush=True)
        # E-048: WHICH LADDER, and — for `auto` — that it has not measured
        # anything yet. A run whose ladder is not the default says so in the
        # first ten lines of its log, because `params_M` cannot distinguish
        # one lattice from another any more than it can distinguish a
        # quantized codec from a continuous one.
        if _q.ladder != "uniform" or _q.fit:
            print(f"FSQ ladder: --fsq-ladder {_q.ladder}"
                  + (f" (base {a.fsq_exp_base:g})" if _q.ladder == "exp"
                     else "")
                  + (f" · fitted lattice IN HAND from the dispatch/resume: "
                     f"{_q.fit}" if _q.fit else "")
                  + (f" · NOT YET FITTED — quantizing uniformly until step "
                     f"{fql.fit_steps(a.fsq_auto_step, a.steps)[0]}, "
                     f"re-fitting at "
                     f"{fql.fit_steps(a.fsq_auto_step, a.steps)}"
                     if _q.needs_fit else ""), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    # Cosine annealing whose TOTAL can be re-fitted mid-run (--max-minutes):
    # a LambdaLR closure over a mutable total is the same curve as
    # CosineAnnealingLR (eta_min=0) but survives having its denominator
    # changed, which the built-in scheduler's recursive formula does not.
    import math
    sched_total = [a.lr_decay_steps if (a.lr_floor > 0 and a.lr_decay_steps)
                   else a.steps]
    FLOOR = a.lr_floor
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(
            math.pi * min(e, sched_total[0]) / sched_total[0])))
    huber = torch.nn.HuberLoss(reduction="none")

    # neighbour offsets (Δx, Δy, Δt): 4 spatial + 2 temporal
    NEI = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def gather_block(b, y, x):
        """[B, k_max, C] values and observed flags for whole blocks.

        The cells of one block are k_max SOURCE rows of the same pixel, so
        this is the per-bin gather done k_max times and stacked — the memory
        stays per-batch (LazyPixels derives it), and a PAD cell is forced
        unobserved here rather than anywhere downstream."""
        rows = blk_rows[b]                                   # [B, k_max]
        vs, os_ = [], []
        for j in range(rows.shape[1]):
            v_, o_ = Xt[rows[:, j], y, x], OBS[rows[:, j], y, x]
            vs.append(v_)
            os_.append(o_)
        v = torch.stack(vs, 1).to(dev, torch.float32)
        o = torch.stack(os_, 1).to(dev)
        o = o & (~blk_pad[b]).to(o.device).unsqueeze(-1)
        return v, o

    def gather(t, y, x):
        # float32 EXPLICITLY. The batch's dtype follows the tensor's, and
        # family 4 is float16 — which reaches the codec as Half against Float
        # weights (handled in PixelMAE.encode) and, less obviously, becomes
        # the huber TARGET here. A Half target makes the backward pass fail
        # outright: `RuntimeError: Found dtype Half but expected Float`. So
        # the widening belongs at the gather, where values enter the graph at
        # all, not only at the encoder. Exact for float16 and a no-op for the
        # float32 families, so nothing already measured moves.
        v = Xt[t, y, x].to(dev, torch.float32)
        o = OBS[t, y, x].to(dev)
        return v, o

    def step_loss_block(b, y, x, ctx):
        """E-047: the same masked-reconstruction objective over a GRID.

        Everything is the per-bin objective with a cell axis added: the mask
        is drawn per CELL at the same --mask-ratio, the encoder reads the grid
        at once, the decoder is asked for every cell (channel c AT cell j via
        q_time, offset 0), and the loss weights are the existing
        mask + rec_w_visible*(observed & unmasked) scheme. The neighbour term
        keeps its meaning by moving with the axis: dt = +/-1 is the next or
        previous BLOCK, because the block axis IS the time axis downstream.

        E-048, ON AN OVERLAPPING AXIS (--time-block W/S with S < W): dt = +/-1
        is the adjacent WINDOW, which SHARES W-S bins with this one, so the
        temporal neighbour term is partly a reconstruction term. That is the
        training-side twin of the persistence caveat the axis prints at
        startup, and it is deliberate rather than overlooked: the alternative
        — stepping to the first NON-overlapping window instead — would make dt
        mean a different duration at every stride and break the one rule the
        block axis has, that `off`'s dt counts BLOCKS.
        """
        B = len(b)
        v, o = gather_block(b, y, x)                        # [B, KT, C]
        KT = v.shape[1]
        mask = (torch.rand(B, KT, C, device=dev) < a.mask_ratio) & o
        z = model.encode(v * (~mask), o, mask, ctx.to(dev))
        qc = (torch.arange(C, device=dev)[None, None, :]
              .expand(B, KT, -1).reshape(B, KT * C))
        qt = (torch.arange(KT, device=dev)[None, :, None]
              .expand(B, -1, C).reshape(B, KT * C))
        off0 = torch.zeros(B, KT * C, 3, dtype=torch.long, device=dev)
        pred = model.query(z, qc, off0, qt).reshape(B, KT, C)
        l_rec = huber(pred, v)
        w = (mask.float() + a.rec_w_visible * (o & ~mask).float()) \
            * cwt[None, None, :]
        l_rec = (l_rec * w).sum() / w.sum().clamp(min=1)
        # neighbours, in BLOCK units for dt
        pick = np.random.randint(0, len(NEI), B)
        dxyz = torch.as_tensor(np.array([NEI[i] for i in pick]), device=dev)
        bn = (b.to(dev) + dxyz[:, 2]).clamp(0, len(ctx_all) - 1)
        yn = (y.to(dev) + dxyz[:, 1]).clamp(0, H - 1)
        xn = (x.to(dev) + dxyz[:, 0]).clamp(0, W - 1)
        vn, on = gather_block(bn.cpu(), yn.cpu(), xn.cpu())
        offn = dxyz[:, None, :].expand(-1, KT * C, -1).long()
        predn = model.query(z, qc, offn, qt).reshape(B, KT, C)
        wn = on.float() * cwt[None, None, :]
        l_nei = (huber(predn, vn) * wn).sum() / wn.sum().clamp(min=1)
        return l_rec, l_nei

    def step_loss(t, y, x, ctx):
        if BLK is not None:
            return step_loss_block(t, y, x, ctx)
        B = len(t)
        v, o = gather(t, y, x)
        mask = (torch.rand(B, C, device=dev) < a.mask_ratio) & o
        if a.patch > 1:
            vp, op = gather_px(Xt, OBS, t, y, x, a.patch)
            z = model.encode(vp.to(dev, torch.float32), op.to(dev), mask,
                             ctx.to(dev))
        else:
            z = model.encode(v * (~mask), o, mask, ctx.to(dev))

        # self-reconstruction: all channels queried at offset 0
        qc = torch.arange(C, device=dev)[None, :].expand(B, -1)
        off0 = torch.zeros(B, C, 3, dtype=torch.long, device=dev)
        pred = model.query(z, qc, off0)
        l_rec = huber(pred, v)
        # masked channels dominate; visible weight and per-channel upweights
        # are E-019b knobs (defaults reproduce the historical 0.1 exactly)
        w = (mask.float() + a.rec_w_visible * (o & ~mask).float()) * cwt[None, :]
        l_rec = (l_rec * w).sum() / w.sum().clamp(min=1)

        # neighbours: one random offset per sample
        pick = np.random.randint(0, len(NEI), B)
        dxyz = torch.as_tensor(np.array([NEI[i] for i in pick]), device=dev)
        tn = (t.to(dev) + dxyz[:, 2]).clamp(0, T - 1)
        yn = (y.to(dev) + dxyz[:, 1]).clamp(0, H - 1)
        xn = (x.to(dev) + dxyz[:, 0]).clamp(0, W - 1)
        vn, on = gather(tn.cpu(), yn.cpu(), xn.cpu())
        offn = dxyz[:, None, :].expand(-1, C, -1).long()
        predn = model.query(z, qc, offn)
        wn = on.float() * cwt[None, :]
        l_nei = (huber(predn, vn) * wn).sum() / wn.sum().clamp(min=1)
        return l_rec, l_nei

    if (a.eval_every or a.light_probe_every) and not a.anomaly:
        raise SystemExit("--eval-every/--light-probe-every require --anomaly "
                         "(trainprobe measures anomaly-space embeddings; state "
                         "space is disqualified)")
    if a.eval_every or a.light_probe_every:
        import trainprobe                      # lazy: plain runs don't need it
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    loss_every = max(1, a.steps // 200)        # the loss curve, cheap to keep

    # A CONFIG RECORD, first line of the metrics file. The GitHub API does
    # not return a workflow_dispatch run's inputs, so a reader of the live
    # curves had no way to know how long the run is MEANT to be — "step
    # 30,000" alone cannot be told from "step 30,000 of 60,000". Writing the
    # plan next to the measurements is the cheapest fix and it travels with
    # the data (live branch, archive, and the harvested artifact alike).
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"config": {
            "steps": a.steps, "batch": a.batch, "d_z": a.d_z, "patch": a.patch,
            "d_model": a.d_model, "n_layers": a.n_layers, "n_heads": a.n_heads,
            "d_dec": a.d_dec, "anomaly": bool(a.anomaly),
            # E-046. The bottleneck is the one architecture field that adds no
            # parameters, so `params_M` above cannot distinguish a quantized
            # codec from a continuous one — and this record is sometimes the
            # only surviving account of what ran (#387). None = continuous.
            "fsq_levels": (a.fsq_levels or None),
            # E-048: and WHICH LADDER those levels sit on. `auto` records the
            # fitted lattice in its own metrics line at the fit step; this is
            # the setting, and it is here for the same reason fsq_levels is —
            # this record is sometimes the only surviving account of what ran.
            "fsq_ladder": (a.fsq_ladder if a.fsq_levels else None),
            "eval_every": a.eval_every, "light_probe_every": a.light_probe_every,
            "lr_floor": a.lr_floor, "lr_decay_steps": a.lr_decay_steps,
            "params_M": round(sum(p_.numel() for p_ in model.parameters()) / 1e6, 3),
            "data": os.path.basename(a.data), "C": int(C), "T": int(T),
            "resume": a.resume or None,
            # WHICH RECIPE produced this curve. Exported by
            # scripts/resolve_recipe.sh; empty on a hand-assembled dispatch.
            # #387's post-mortem had to be reconstructed from this record
            # because force-cancel had destroyed the job log — the config line
            # in metrics.jsonl was the only surviving account of what ran.
            "recipe": os.environ.get("RECIPE_NAME") or None,
        }}) + "\n")

    # Where a checkpoint can outlive its job. /opt/earth-cache is the box's
    # persistent cache directory: it sits OUTSIDE the Actions workspace, so
    # actions/checkout's clean does not touch it, and a later job on the same
    # box can resume from it with no upload and no download.
    # CKPT_DIR is defined in the architecture-resolution block above, which
    # has to resolve --resume before the model is built.
    ckpt_tag = os.environ.get("CKPT_TAG", "")

    def save_ckpt(step=None):
        """Write the checkpoint, and everything needed to CONTINUE from it.

        Until 2026-08-08 this saved weights only, which made every cancelled
        run a total loss: the weights alone cannot resume a schedule, and a
        job that is stopped at step 15,000 of 60,000 threw away an hour of
        real training. It now carries the optimizer and scheduler state and
        the step reached, so --resume picks the run up exactly where it
        stopped. Optimizer state roughly triples the file; that is the price
        of not repeating work, and these files are transient.
        """
        blob = {"model": model.state_dict(), "chan": chan, "d_z": a.d_z,
                "norm": d["norm"], "args": vars(a),
                "step": int(step if step is not None else 0),
                # WHOSE checkpoint this is. Without it a resumed run knows the
                # file it loaded but not the RUN that wrote it, so the status
                # page cannot find the parent's curves to stitch on — and an
                # orphan-latest.pt rescued off a box carries no run number in
                # its name at all. One string closes that gap.
                "tag": ckpt_tag,
                "opt": opt.state_dict(), "sched": sched.state_dict()}
        torch.save(blob, os.path.join(a.out, "pixelmae.pt"))
        # Mirror to the box-persistent directory when there is one, so a
        # cancel does not destroy the progress.
        if ckpt_tag and os.path.isdir(os.path.dirname(CKPT_DIR) or "/"):
            try:
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save(blob, os.path.join(CKPT_DIR, ckpt_tag + ".pt"))
            except OSError as e:                      # full disk, read-only …
                print(f"  (checkpoint mirror skipped: {e})", flush=True)

    def probe_on_device(**kw):
        """Run the in-training probe WITHOUT surrendering the GPU.

        train.py used to hand `model.cpu()` to every probe. embed_everything
        runs on whatever device the model is on and its own docstring says
        the difference is "hours of CPU and minutes of GPU" — so that one
        `.cpu()` was donating the accelerator back. Measured on the 41M
        anchored runs (#48/#49, 0.25-degree tensor): a single full probe cost
        3,697 s and 4,802 s, i.e. 70-74% of all wall-clock time, projecting
        to ~12 h of probing against ~3.5 h of training.

        Falls back to CPU on a CUDA OOM rather than killing the run: the
        probe is instrumentation, and instrumentation must never be the
        thing that loses a training job. empty_cache() first because the
        optimiser state and activations are still resident.

        `ocean=ocean` HOISTS the ocean mask out of the probe. It is exactly
        the expression probe_now used to evaluate on every call —
        `np.isfinite(X[..., 0]).any(axis=0)`, computed at line 295 above from
        the same array, before the anomaly transform — and it is a pure
        function of the tensor, so recomputing it per probe was buying a
        constant at 150.2 s a time on #419's daily tensor (8.5% of ALL probe
        wall clock; 125.8-3,657.2 s on #415 depending on page cache). It is a
        strided traversal of the whole [T,H,W,C] array at 78-byte stride,
        i.e. the pathology that wedged #389 for seven hours, and it was being
        paid ~120 times per run.
        """
        try:
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
            return trainprobe.probe_now(model, Xt, OBS, _probe_d,
                                        _probe_moy, _probe_hold,
                                        x_hold, dynamic, ocean=ocean,
                                        **_blkkw, **kw)
        except torch.cuda.OutOfMemoryError:            # cuda-only path
            print("  probe OOM on GPU — falling back to CPU for this one",
                  flush=True)
            torch.cuda.empty_cache()
            m_ = trainprobe.probe_now(model.cpu(), Xt, OBS, _probe_d,
                                      _probe_moy, _probe_hold,
                                      x_hold, dynamic, ocean=ocean,
                                      **_blkkw, **kw)
            model.to(dev)
            return m_

    # E-047: the probe path embeds through the SAME contract as the main one.
    # #448 (2026-08-23) ran a month-block codec whose every probe raised
    # "BLOCK codec and no block map was passed" — caught, warned and skipped,
    # which is the RIGHT failure mode for a probe and the WRONG outcome for a
    # run: the collapse guard eats probe values, so a run with no probes has a
    # guard with nothing to guard on, and the end-of-run head probe cannot
    # work either. The map is in scope exactly when --time-block is on.
    _blkkw = ({} if BLK is None
              else {"blk_rows": BLK.rows, "blk_pad": BLK.pad})
    # ...AND THE AXIS THE PROBE READS MUST BE THE ONE IT EMBEDS. probe_now
    # keys the RAPID truth, the month-of-year and the holdout mask on the
    # rows of the Z it just built, and a block codec's Z has one row per
    # BLOCK — so `rapid` is remapped to blocks (the same BlockAxis.remap_rows
    # the embed path uses), `months`/`moy`/`t_hold` become the block axis's,
    # and only the members the probe actually reads are copied (never X,
    # which is 34 GB at pentad).
    _probe_d, _probe_moy, _probe_hold = d, mvec, t_hold
    if BLK is not None:
        _probe_d = {"lats": d["lats"], "lons": d["lons"],
                    "months": np.array(BLK.labels),
                    "rapid": BLK.remap_rows(d["rapid"])}
        _probe_moy = np.array([int(m[5:7]) - 1 for m in BLK.labels])
        _probe_hold = blk_hold

    # ---- COLLAPSE GUARD (2026-08-18, from #387) --------------------------
    # #387 (f4-200M) trained for 27,000 steps. Its embedding died somewhere
    # between step 10k and 15k: linear_r_deseas went 0.540 -> 0.316 -> 0.392
    # -> 0.000 -> 0.000 -> 0.000 while z-space MSE ran 0.20 -> 323 -> 2,614
    # -> 9,685. Nothing stopped it, and it burned ~9 more hours before a human
    # looked at the curve.
    #
    # loss_rec could not have caught this and never will: the decoder is a
    # free MLP over z, so it absorbs an arbitrary rescaling of its input and
    # reconstruction stays mediocre-but-finite while the latent runs away.
    # #387's loss_rec sat at 0.27-0.32 throughout. The fleet health checks
    # could not catch it either — a collapsed model holds the GPU at 100%.
    #
    # The probe correlation CAN catch it, because a correlation is scale
    # invariant. Two consecutive readings at or below --collapse-r is the
    # signal; on #387 that fires at step 20,000.
    #
    # Deliberately NOT a relative test against the step-0 value: a resumed run
    # has no step-0 probe, and "half of baseline" would also fire on a healthy
    # run that started lucky. An absolute floor near zero only ever means one
    # thing.
    strikes = [0]

    def _collapse_check(m, step):
        # Step 0 is EXEMPT: it measures an untrained codec on purpose, and a
        # random encoder is allowed to read near zero. Counting it would let
        # one weak baseline plus one slow start abort a healthy run.
        if not a.collapse_r or m is None or step == 0:
            return
        r = m.get("linear_r_deseas")
        if r is None:
            return
        r = float(r)
        # NaN is NOT collapse — it is NO READING. A degenerate probe (too few
        # held-out pixels, an all-NaN slice) returns NaN on a perfectly
        # healthy run, and killing training on instrumentation failure is the
        # #56-#59 lesson in a new costume. Neither strike nor reset: wait for
        # a real number. A model that has genuinely gone non-finite is caught
        # by the loss check in the training loop, which is unambiguous.
        if r != r:
            print(f"  COLLAPSE WATCH: probe returned NaN at step {step} — "
                  f"no reading, strike count held at {strikes[0]}", flush=True)
            return
        if abs(r) > a.collapse_r:
            strikes[0] = 0
            return
        strikes[0] += 1
        print(f"  COLLAPSE WATCH: linear r_des {r:+.3f} at step {step} "
              f"— strike {strikes[0]}/{a.collapse_strikes}", flush=True)
        if strikes[0] < a.collapse_strikes:
            return
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "collapsed": {
                "linear_r_deseas": r, "threshold": a.collapse_r,
                "strikes": strikes[0]}}) + "\n")
        raise SystemExit(
            f"ABORTING at step {step}: the probe's linear_r_deseas has been "
            f"<= {a.collapse_r} on {strikes[0]} consecutive probes (last "
            f"{r:+.3f}). The embedding carries no linearly decodable signal — "
            f"this codec is dead and further steps cannot revive it. Check "
            f"LR/warmup/head_dim before re-dispatching; see the #387 "
            f"post-mortem. Pass --collapse-r 0 to disable this guard.")

    def run_probe(step, light):
        """Write one probe record to metrics.jsonl and return it (or None).

        A probe is INSTRUMENTATION, and instrumentation must never be the
        thing that loses a training job. #56-#59 all died at their FIRST full
        probe (step 10k) on a codec.query device mismatch — hours of training
        and every checkpoint, gone, because a probe raised. The device bug is
        fixed in trainprobe.py; THIS is the guard that keeps the next probe
        bug from being fatal. Any exception is caught, logged to the metrics
        file as a {"probe_error"} record (so it shows on the status page
        rather than vanishing), and training carries on.
        """
        try:
            m = probe_on_device(light=light)
        except Exception as e:                       # never fatal
            import traceback
            traceback.print_exc()
            print(f"  probe @{step} FAILED ({type(e).__name__}: {e}) — "
                  f"training continues, no probe point this interval",
                  flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "wall_s": round(time.time() - t0, 1),
                    "probe_error": f"{type(e).__name__}: {e}"[:200]}) + "\n")
            return None
        m["step"] = step
        m["wall_s"] = round(time.time() - t0, 1)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        _collapse_check(m, step)
        return m

    def run_light_probe(step):
        """The cheap probe, written to metrics.jsonl."""
        return run_probe(step, light=True)

    print("training …")
    t0 = time.time()
    # STEP-0 PROBE — the untrained control (Chris, 2026-08-08: "would it be
    # useful to compute the green (r) metric before any training has
    # happened … it may be useful to identify whether training is going in
    # the right direction or whether it does nothing"). A randomly
    # initialised encoder still emits embeddings, and a ridge on random
    # features is a real baseline. Without this point, a flat probe curve is
    # ambiguous: it could mean training converged early, or that training
    # never moved the metric at all. With it, the answer is one subtraction.
    # It costs one light probe (~30 s) and it is the cheapest control in the
    # programme.
    # ...but NOT on a resumed run. Verified on #52: the step-0 probe fired
    # before the resume restored the weights, so it measured a throwaway
    # random init and wrote it to metrics.jsonl as "step 0" — where the
    # status page draws it as the untrained-codec reference line. On a
    # continuation that line would be a lie about a different model.
    # Run the SAME probe at step 0 that runs at every eval interval, so every
    # metric the run reports has an untrained baseline to be read against —
    # not just the light linear one (Chris, 2026-08-08: "schedule the same
    # metric at step 0?"). If eval_every is set the step-0 probe is the FULL
    # one (k-fold RAPID, temporal r, chan%/z% on a random-init codec); if only
    # the light cadence is on, it stays light. The status page reads
    # linear_r_deseas for its untrained-codec line, which the full probe emits
    # too, so that reference is unaffected.
    if (a.light_probe_every or a.eval_every) and not a.resume:
        m0 = run_probe(0, light=not a.eval_every)
        if m0:
            extra = ("" if m0.get("light") else
                     f" · temporal r_des {m0.get('temporal_r_deseas', float('nan')):+.3f}"
                     f" · chan t+1 {m0.get('chan_vs_persistence_pct', float('nan')):+.1f}%")
            print(f"  step-0 probe (UNTRAINED codec): linear r_des "
                  f"{m0['linear_r_deseas']:+.3f}{extra} — every later probe "
                  f"should be read as a change from this", flush=True)
    elif a.resume:
        print("  (no step-0 probe: this run resumes, so there is no untrained "
              "baseline to measure — the original run's step-0 point is the "
              "one that applies)", flush=True)
    CAL = 200                                  # steps before the rate is trusted
    t1 = None                                  # wall clock AFTER step 1 lands
    next_cal = None                            # next step to re-check the fit
    steps0 = a.steps                           # the DISPATCHED total — refits
    #                                            may never exceed it, and a
    #                                            recovery may grow back to it
    s = 0
    if a.resume:
        # --resume takes a COMMA-SEPARATED CANDIDATE LIST and uses the first
        # one present on this box. Checkpoint mirrors are box-local and the
        # scheduler picks the runner, so naming a single tag means a job that
        # wants "a finished 40M codec" succeeds only by luck — even when two
        # of three boxes hold one. Listing them turns 1-in-3 into 2-in-3, and
        # the log still says exactly which checkpoint was used, so provenance
        # is unchanged.
        # Resolved once, up in the architecture block — the same path and the
        # same already-loaded checkpoint, so the file is read exactly once.
        rpath, cands = RESUME_PATH, RESUME_CANDS
        if RESUME_CK is None and len(cands) > 1:
            print(f"  --resume: none of {cands} is on this box", flush=True)
        if not os.path.exists(rpath):
            if a.require_resume:
                raise SystemExit(
                    f"--require-resume: no checkpoint at {rpath}. This box is "
                    f"not the one that wrote it (checkpoint mirrors are "
                    f"box-local). Exiting in seconds rather than retraining "
                    f"from scratch for hours under a doc string that claims "
                    f"to be a continuation.")
            print(f"  --resume {a.resume}: NOT FOUND at {rpath} — starting "
                  f"from scratch (this is not an error, but the run is now a "
                  f"fresh one; say so in its doc string)", flush=True)
        else:
            ck = RESUME_CK
            # SAY WHAT WE LOADED. /opt/earth-cache/ckpt/orphan-latest.pt is
            # whatever the last job on THIS box left behind, which is not
            # necessarily this experiment. load_state_dict fails loudly on an
            # architecture mismatch, but a same-architecture checkpoint from a
            # different run would load silently — so print its identity and
            # let it land in the log and the provenance record.
            ca = ck.get("args", {})
            print(f"  resume source: {rpath}\n"
                  f"    C={len(ck.get('chan', []))} d_z={ck.get('d_z')} "
                  f"d_model={ca.get('d_model')} layers={ca.get('n_layers')} "
                  f"patch={ca.get('patch')} data={os.path.basename(str(ca.get('data','?')))}\n"
                  f"    it was trained toward {ca.get('steps')} steps",
                  flush=True)
            ck_data = os.path.basename(str(ca.get("data", ""))) if ca.get("data") else ""
            if ck_data and ck_data != os.path.basename(a.data):
                # The refusal guards TRAINING: continuing a codec on a tensor
                # it was not trained on writes a checkpoint whose provenance
                # is a lie. An EVAL-ONLY pass trains nothing — the loop below
                # is `while s < a.steps`, so a checkpoint already at/past
                # --steps changes no weight — and cross-tensor evaluation is
                # not an accident here, it is E-038's FROZEN CONTROL: score
                # the monthly anchor on the pentad tensor, the one number
                # that can falsify the out-of-domain premise. So the refusal
                # keys on whether anything will train, which is exactly what
                # the stated reason protects. A checkpoint with no recorded
                # step cannot prove it will not train (the warm-start branch
                # restarts s at 0), so it stays refused.
                if "step" in ck and int(ck["step"]) >= a.steps:
                    print(f"  CROSS-TENSOR EVAL: codec trained on {ck_data}, "
                          f"evaluated on {os.path.basename(a.data)}. No "
                          f"training will occur (checkpoint step "
                          f"{int(ck['step'])} >= --steps {a.steps}); the "
                          f"saved artefact is the loaded weights, re-scored.",
                          flush=True)
                else:
                    raise SystemExit(
                        f"REFUSING to resume: checkpoint was trained on "
                        f"{ck_data} but this run uses "
                        f"{os.path.basename(a.data)}. Cross-tensor TRAINING "
                        f"would produce a codec whose provenance is a lie. "
                        f"(Eval-only is allowed: pass --steps at or below "
                        f"the checkpoint's recorded step, so nothing trains.)")
            model.load_state_dict(ck["model"])
            if "opt" in ck and "step" in ck:
                opt.load_state_dict(ck["opt"])
                if "sched" in ck:
                    sched.load_state_dict(ck["sched"])
                s = int(ck["step"])
                print(f"  RESUMED from {rpath} at step {s} "
                      f"(optimizer + schedule restored); training on to "
                      f"{a.steps}", flush=True)
                # A CONTINUATION RECORD, so the curves can be made whole
                # again. A resumed run's metrics file starts at the step it
                # was handed, which draws a chart that begins in mid-air and
                # silently loses everything the parent measured — including
                # the step-0 untrained-codec line, which is the reference the
                # whole probe chart is read against. Naming the parent and
                # the join step lets the status page fetch the parent's
                # archived metrics and prepend them, and MARK the seam rather
                # than hide it: two jobs, one training trajectory.
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"resumed": {
                        "from": os.path.basename(rpath),
                        "parent_tag": ck.get("tag") or "",
                        "at_step": s,
                    }}) + "\n")
            else:
                print(f"  WARM-STARTED from {rpath}: weights only, no "
                      f"optimizer or step — the LR schedule restarts from 0. "
                      f"Report this run as a warm start, not a continuation.",
                      flush=True)
            if s >= a.steps:
                print(f"  checkpoint is already at/past --steps; nothing to do")
    # ---- E-048: the --fsq-ladder auto FIT --------------------------------
    # Driven from HERE rather than by a counter hidden inside the quantizer,
    # so what the fit measured is a function of the run's own seed and step
    # schedule and nothing else. It fires at every step in --fsq-auto-step, on
    # a deliberate no-grad forward over a fresh TRAIN-pool batch, and the
    # lattice it installs is written into every checkpoint from that step on.
    # MORE THAN ONE FIT because the fit now measures the saturation RADIUS as
    # well as the ladder, and `to_z`'s output scale is exactly what moves
    # while the encoder trains: a single early fit bounds an encoder that no
    # longer exists, and a single late one leaves the run quantizing into a
    # constant bound for most of its steps.
    FSQ_FIT_AT = (set(fql.fit_steps(a.fsq_auto_step, a.steps))
                  if (model.fsq is not None and model.fsq.needs_fit) else set())

    def fsq_auto_fit(step):
        q = model.fsq
        n = max(2, int(a.fsq_auto_n))
        # `encode_pre`: the real encoder, without the bottleneck. The fit has
        # to see the distribution the quantizer will actually be handed, and
        # nothing masked — a masked cell changes what the encoder is asked,
        # not what a ladder must serve.
        with torch.no_grad():
            bt, by, bx, bctx = batch(tt, yy, xx, n)
            if BLK is not None:
                v_, o_ = gather_block(bt, by, bx)
                pre = model.encode_pre(v_, o_, torch.zeros_like(o_),
                                       bctx.to(dev))
            elif a.patch > 1:
                vp, op = gather_px(Xt, OBS, bt, by, bx, a.patch)
                pre = model.encode_pre(vp.to(dev, torch.float32), op.to(dev),
                                       torch.zeros(n, C, dtype=torch.bool,
                                                   device=dev), bctx.to(dev))
            else:
                v_, o_ = gather(bt, by, bx)
                pre = model.encode_pre(v_, o_, torch.zeros_like(o_),
                                       bctx.to(dev))
        sample = pre.detach().to("cpu", torch.float32).numpy()
        line = q.fit_auto(sample)
        a.fsq_ladder_fit = q.fit
        print(f"  step {step}: {line}", flush=True)
        R = q._scale.detach().cpu().numpy().astype(float)
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": int(step), "fsq_ladder_fit": {
                "spec": q.fit, "n": int(len(sample)),
                "n_exp": int(q.is_exp_np.sum()),
                "scale_min": round(float(R.min()), 6),
                "scale_med": round(float(np.median(R)), 6),
                "scale_max": round(float(R.max()), 6),
                "prequant_absmean": round(float(np.abs(sample).mean()), 6),
                "d_z": int(a.d_z)}}) + "\n")

    while s < a.steps:
        s += 1
        if s in FSQ_FIT_AT:
            fsq_auto_fit(s)
        t, y, x, ctx = batch(tt, yy, xx, a.batch)
        l_rec, l_nei = step_loss(t, y, x, ctx)
        loss = l_rec + 0.5 * l_nei
        # NON-FINITE LOSS IS UNAMBIGUOUS — stop before the NaN is written into
        # every weight by the next opt.step(). The collapse guard above works
        # on the probe and deliberately treats NaN as "no reading"; this is the
        # other half, and it is the one case where NaN means the model, not the
        # instrument.
        if not torch.isfinite(loss):
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": s, "diverged": {
                    "loss_rec": float(l_rec.item()),
                    "loss_nei": float(l_nei.item())}}) + "\n")
            raise SystemExit(
                f"ABORTING at step {s}: loss is {loss.item()} (rec "
                f"{l_rec.item()}, nei {l_nei.item()}). The model has gone "
                f"non-finite; every further step writes NaN into the weights. "
                f"There is no warmup and no gradient clipping on this path — "
                f"suspect the learning rate first.")
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s == 1:
            t1 = time.time()
        # Wall-clock refit — MEASURED steady-state rate, re-checked as it runs.
        # The first version of this block calibrated ONCE, at `s >= 200 or
        # elapsed > 60s`, over elapsed/s FROM THE START. Run #366 (2026-08-17,
        # the first pentad codec) hit all three of its latent defects at once:
        # step 1 carried ~9 minutes of one-time cost (first CUDA kernels +
        # first touch of a 33 GB tensor), so the 60 s branch fired at s=1,
        # concluded 537.54 s/step against a true steady rate of ~0.19, re-fit
        # 200,000 steps to SIXTY-SIX, annealed the LR to zero and reported
        # success — a green run carrying a near-random codec into the probe
        # archive, with 691 of its 700 budgeted minutes unspent. Three rules,
        # each of which would have prevented it alone:
        #   1. the rate EXCLUDES step 1 (t1, not t0) — one-time cost is not a
        #      rate;
        #   2. no fit from fewer than 3 steady steps — a 1-step sample is a
        #      guess wearing a measurement's clothes;
        #   3. the fit is RE-CHECKED (cheaply, on a step schedule), and may
        #      grow back toward the dispatched total — calibrate-once turns
        #      one bad reading into the run's final answer.
        # The budget itself still counts from t0: step 1's cost was real spend.
        if (a.max_minutes and FLOOR == 0 and t1 is not None and s >= 4
                and (s >= CAL or time.time() - t1 > 60)
                and (next_cal is None or s >= next_cal)):
            now = time.time()
            fit, rate = fit_schedule(s, now - t1, now - t0,
                                     a.max_minutes, steps0)
            # re-check soon while young (a shrunk schedule must be able to
            # recover before it expires), sparsely once the estimate is stable
            next_cal = s + max(25, min(2000, (fit - s) // 4))
            if abs(fit - a.steps) > max(50, a.steps // 20):
                print(f"  time budget: {rate:.2f} s/step steady (n={s - 1}) → "
                      f"re-fitting the cosine schedule from {a.steps} to {fit} "
                      f"steps so the LR anneals to zero inside "
                      f"{a.max_minutes} min")
                a.steps = fit
                sched_total[0] = fit
        if a.max_minutes and (time.time() - t0) > a.max_minutes * 60:
            print(f"  wall-clock budget reached at step {s} — stopping to save")
            break
        if s % loss_every == 0 or s == a.steps:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": s, "loss_rec": round(l_rec.item(), 5),
                                    "loss_nei": round(l_nei.item(), 5)}) + "\n")
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  rec {l_rec.item():.4f}  nei {l_nei.item():.4f}"
                  f"  ({time.time() - t0:.0f}s)")
        # Cheap probe first, and skipped on steps where the full probe runs
        # (the full one supersedes it and writes the same key).
        full_here = a.eval_every and (s % a.eval_every == 0 or s == a.steps)
        if a.light_probe_every and not full_here and s % a.light_probe_every == 0:
            m = run_probe(s, light=True)
            if m:
                print(f"  light probe @{s}: linear r_des "
                      f"{m['linear_r_deseas']:+.3f} "
                      f"({m['probe_seconds']:.0f}s)", flush=True)
        if full_here:
            m = run_probe(s, light=False)
            if m:
                print(f"  probe @{s}: chan t+1 {m['chan_vs_persistence_pct']:+.1f}% "
                      f"vs persistence · linear r_des {m['linear_r_deseas']:+.3f} · "
                      f"temporal r_des {m['temporal_r_deseas']:+.3f} "
                      f"({m['probe_seconds']:.0f}s)", flush=True)
            # Crash insurance is INDEPENDENT of the probe: save even when the
            # probe failed, so a probe bug never also costs the checkpoint.
            save_ckpt(s)

    # ---- evaluation on the BLOCKED holdout --------------------------------
    model.eval()
    results = {}
    if BLK is not None:
        # E-047: this evaluation is per-BIN throughout — it gathers one row
        # per sample, queries C channels at offset 0, and compares against a
        # t+1 that means "the next bin". Every one of those is a different
        # question under blocking, and a block-shaped rewrite of it is not
        # what decides E-047 (stage 2 and the roll are). So it is SKIPPED,
        # LOUDLY and in the artefact — never silently, and never by producing
        # a number that answers a question nobody asked (ml/CLAUDE.md 4.6).
        results["eval_skipped"] = (
            f"--time-block {a.time_block!r}: the codec's own per-bin "
            f"evaluation does not apply to a block codec (its recon rows, its "
            f"t+1 persistence and its channel skills are all per-bin "
            f"quantities). The verdict for a block codec is the stage-2 head "
            f"and the roll — see ml/plans/E047_block_codec.md 4.")
        print(f"::warning::{results['eval_skipped']}", flush=True)
        json.dump(results, open(os.path.join(a.out, "eval.json"), "w"),
                  indent=1)
        save_ckpt(a.steps)
        print(f"saved {os.path.join(a.out, 'pixelmae.pt')}", flush=True)
        return
    with torch.no_grad():
        n_eval = min(20000, len(vt))
        t, y, x, ctx = batch(vt, vy, vx, n_eval)
        v, o = gather(t, y, x)
        mask = (torch.rand(n_eval, C, device=dev) < a.mask_ratio) & o
        # CHUNK the final evaluation. Encoding all 20,000 held-out pixels in
        # one forward makes the encoder's feed-forward intermediate
        # n_eval x C x 4*d_model x 4 B — at C=39, d_model=576 that is 7.04 GiB,
        # exactly the allocation that OOM-killed #62 and #63 AFTER they had
        # trained all 60,000 steps and passed every probe. The evaluation is
        # the last thing a run does, so the cost of that crash was the whole
        # run's verdict. Chunking changes no reported number: z, pred and p1
        # are concatenated in the same order and every metric below is
        # computed from the complete arrays.
        EV_CH = 2048
        zs = []
        for i in range(0, n_eval, EV_CH):
            sl = slice(i, min(i + EV_CH, n_eval))
            if a.patch > 1:
                vp, op = gather_px(Xt, OBS, t[sl], y[sl], x[sl], a.patch)
                zs.append(model.encode(vp.to(dev), op.to(dev), mask[sl],
                                       ctx[sl].to(dev)))
            else:
                zs.append(model.encode(v[sl] * (~mask[sl]), o[sl], mask[sl],
                                       ctx[sl].to(dev)))
        z = torch.cat(zs)
        qc = torch.arange(C, device=dev)[None, :].expand(n_eval, -1)
        preds = []
        for i in range(0, n_eval, EV_CH):
            sl = slice(i, min(i + EV_CH, n_eval))
            nb = sl.stop - sl.start
            preds.append(model.query(
                z[sl], qc[sl],
                torch.zeros(nb, C, 3, dtype=torch.long, device=dev)))
        pred = torch.cat(preds)
        for c, name in enumerate(chan):
            m = mask[:, c]
            if m.sum() < 50:
                continue
            err = (pred[m, c] - v[m, c]).pow(2).mean().item()
            base = v[m, c].pow(2).mean().item()               # channel mean = 0 after z-score
            results[f"recon/{name}"] = {"mse": err, "mse_channel_mean": base,
                                        "skill": 1 - err / max(base, 1e-9)}

        # temporal neighbour t+1 vs persistence
        t1 = np.clip(t.numpy() + 1, 0, T - 1)
        v1, o1 = gather(torch.as_tensor(t1), y, x)
        p1s = []
        for i in range(0, n_eval, EV_CH):                 # chunked, as above
            sl = slice(i, min(i + EV_CH, n_eval))
            nb = sl.stop - sl.start
            offb = torch.zeros(nb, C, 3, dtype=torch.long, device=dev)
            offb[:, :, 2] = 1
            p1s.append(model.query(z[sl], qc[sl], offb))
        p1 = torch.cat(p1s)
        both = (o & o1)
        mse_m = ((p1 - v1).pow(2) * both).sum().item() / both.sum().item()
        mse_p = ((v - v1).pow(2) * both).sum().item() / both.sum().item()
        results["t+1"] = {"mse_model": mse_m, "mse_persistence": mse_p,
                          "beats_persistence": bool(mse_m < mse_p)}

        # ---- RAPID probe ---------------------------------------------------
        rapid = d["rapid"]
        if len(rapid):
            from temporal import RAPID_LON
            sec_y = int(np.argmin(np.abs(lats - 26.5)))
            sec_x = np.where(np.isfinite(X[0, sec_y, :, 0])
                             & (lons >= RAPID_LON[0]) & (lons <= RAPID_LON[1]))[0]
            emb = np.zeros((T, a.d_z), dtype=np.float32)
            for tix in range(T):
                n = len(sec_x)
                ctx = np.concatenate([np.tile(ctx_all[tix], (n, 1)),
                                      (np.full(n, lats[sec_y]) / 90)[:, None],
                                      (lons[sec_x] / 180)[:, None]], 1)
                v, o = gather(torch.full((n,), tix, dtype=torch.long),
                              torch.full((n,), sec_y, dtype=torch.long),
                              torch.as_tensor(sec_x))
                if a.patch > 1:
                    v, o = gather_px(Xt, OBS, torch.full((n,), tix, dtype=torch.long),
                                     torch.full((n,), sec_y, dtype=torch.long),
                                     torch.as_tensor(sec_x), a.patch)
                    v, o = v.to(dev), o.to(dev)
                zz = model.encode(v, o, torch.zeros(n, C, dtype=torch.bool, device=dev),
                                  torch.as_tensor(ctx, dtype=torch.float32).to(dev))
                emb[tix] = zz.mean(0).cpu().numpy()
            ridx = rapid[:, 0].astype(int); rv = rapid[:, 1]
            tr = ~t_hold[ridx]; te = t_hold[ridx]
            if te.sum() >= 12:
                A = np.c_[emb[ridx], np.ones(len(ridx))]
                lam = 1e-2 * np.eye(A.shape[1]); lam[-1, -1] = 0
                wgt = np.linalg.solve(A[tr].T @ A[tr] + lam, A[tr].T @ rv[tr])
                pr = A @ wgt
                r_te = float(np.corrcoef(pr[te], rv[te])[0, 1])
                r_tr = float(np.corrcoef(pr[tr], rv[tr])[0, 1])
                results["rapid_probe"] = {"pearson_train": r_tr, "pearson_heldout_years": r_te,
                                          "n_train": int(tr.sum()), "n_test": int(te.sum())}

    print(json.dumps(results, indent=2))
    save_ckpt(a.steps)
    json.dump(results, open(os.path.join(a.out, "eval.json"), "w"), indent=2)
    print(f"saved {a.out}/pixelmae.pt")

    try:                                       # every run gets its curve
        import plot_run
        plot_run.render(os.path.basename(a.out.rstrip("/")))
    except Exception as e:                     # a missing matplotlib never
        print(f"(curve not rendered: {e})")    # kills a finished run
    except SystemExit as e:                    # ...and neither does a
        print(f"(curve not rendered: SystemExit {e})")   # SystemExit, which
        # is a BaseException and slipped straight through the clause above.



if __name__ == "__main__":
    main()
