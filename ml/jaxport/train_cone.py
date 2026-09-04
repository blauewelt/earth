#!/usr/bin/env python3
"""E-069 · the JAX/optax trainer for the cone-native codec (§8.6).

`ml/train_cone.py` trains `cone_codec.ConeMAE` with torch/AdamW; this trains
`jaxport.cone_models.ConeMAEJax` with optax, emitting the SAME record family
(`ml/plans/E069_HANDOVER.md` §2.4) so `status.html` needs no change and a TPU
run reads exactly like a GPU one — with `"backend": "jax"` as the one mark,
because a TPU-trained number is a NEW TIER under `ml/CLAUDE.md` §3b.

    python3 ml/jaxport/train_cone.py --smoke --out /tmp/cone_jax
    python3 ml/jaxport/train_cone.py --tensor family4_na025_pentad_r3.npz \\
        --steps 20000 --batch 256 --velocity-probe --snapshot-ablation \\
        --ckpt-every 1000 --out ml/runs/cone_jax
    python3 ml/jaxport/train_cone.py ... --resume        # after a preemption

WHAT IS IMPORTED RATHER THAN REWRITTEN, and why. `load_data`, `admissible_bins`,
`draw_anchors`, `fold_labels`, `kfold_r2`, `smoke_tensor` and `FiniteView` come
from `ml/train_cone.py`, and `ConeSampler` from `ml/cone_sampler.py`. Importing
`train_cone` pulls in torch, which is deliberate and is why the TPU node carries
a CPU torch wheel (`ml/plans/E069_HANDOVER.md` §8.2, G5c's finding): a second
copy of the anomaly-space loader, of the admission test or of the ridge would be
a second thing that can drift from the arm it is supposed to be comparable with.
The framework-specific parts — the batch's padding, the mask draw, the step, the
checkpoint — are the only things written here.

FOUR THINGS THIS FILE DECIDES, each of them measurable rather than stylistic:

  * **FIXED SHAPES.** `ConeSampler.sample` returns `N` = the max dot count over
    the ROWS IN THE BATCH, so N moves from step to step and every move retraces
    the jitted step (the `embed.py` lesson). Every batch is padded here to
    `N_max = max(sampler.n_dots(y) for y in range(H))` and the query budget to
    `k = min(n_dot_queries, N_max)`, both computed once, so XLA compiles the
    step exactly once. The trainer COUNTS its own compiles and prints step 1's
    wall time beside step 2's; a second compile after step 2 is a finding.
    Padding rows and columns are inert by construction: a padded dot has
    `valid=False` (excluded by the key-padding mask, which is the ONE place
    existence is enforced) and a padded ROW has `patch_obs`/`fut_obs`/`obs` all
    False, so every query it contributes carries weight exactly 0 and neither
    `wsum` nor `n_targets` nor any mean moves.

  * **THE MASKS ARE DRAWN ON THE HOST, IN NUMPY** (`draw_masks_np`), reproducing
    `ConeMAE._masks` and `ConeMAE.draw_dot_queries` DISTRIBUTION for
    distribution. Two RNGs cannot be made to agree, so the cross-framework gate
    passes the DRAW and not the seed (`tests/test_jaxport_cone.py` C2/C3); what
    the trainer owes is the same distribution, and that is what this function is
    and what C4 measures.

  * **TWO INDEPENDENT numpy Generators, BOTH seeded from `--seed`.** The ANCHOR
    stream mirrors `train_cone.train_one`'s `rng` exactly — the certificate draw
    and then one `draw_anchors` per step — so a JAX run and a torch run at the
    same seed see the SAME anchors in the same order, and C4's band is a
    statement about the two codecs rather than about two different samples of
    the ocean. The MASK stream is a second `default_rng(seed)` (§8.6), because
    the torch run spends its mask randomness in a torch.Generator and there is
    nothing here for that to mean. The two streams share their first outputs'
    entropy and desync immediately (they consume different counts per step);
    that coupling is stated rather than hidden.

  * **THE SCHEDULE IS THE SAME FUNCTION OF THE STEP NUMBER.** torch's
    `CosineAnnealingLR(T_max=steps)` is stepped AFTER `opt.step()`, so update
    `s` (1-indexed) is taken at the lr the scheduler held BEFORE that call,
    i.e. `lr * 0.5 * (1 + cos(pi * (s - 1) / steps))`. optax's
    `cosine_decay_schedule(lr, steps, alpha=0.0)` is evaluated at the
    optimiser's own `count`, which is the number of updates ALREADY applied —
    `s - 1` at update `s`. The two are the same sequence; gate C3 checks one
    step of it (and asserts `sched(0)` is the bare lr) and C4 checks the whole
    smoke run against the torch trainer's.

Plan: `ml/plans/E069_HANDOVER.md` §8.6; gates §8.7 (C3, C4, C8).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
# ML FIRST: this file is `ml/jaxport/train_cone.py` and the module it imports is
# `ml/train_cone.py`. Run as a script, sys.path[0] is this directory, so a bare
# `import train_cone` would re-import THIS file under a second name.
if ML not in sys.path:
    sys.path.insert(0, ML)

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import optax                                                      # noqa: E402
import torch                                                      # noqa: E402
from flax import nnx                                              # noqa: E402

from cone import channel_depth_dbar                               # noqa: E402
from cone_sampler import ConeSampler                              # noqa: E402
from cone_codec import ConeMAE, default_plan                      # noqa: E402
from train_cone import (FiniteView, admissible_bins, draw_anchors,  # noqa: E402
                        fold_labels, kfold_r2, load_data, smoke_tensor)
from jaxport.cone_convert import cone_from_torch, export_cone_pt  # noqa: E402
from jaxport.cone_models import plan_to_jax                       # noqa: E402


# --------------------------------------------------------------------- CLI --
def parse(argv=None):
    """`ml/train_cone.py:parse`'s argument list, flag for flag and default for
    default, plus the three this backend needs (`--ckpt-every`, `--resume`,
    `--stop-at`). A flag that cannot be honoured is REFUSED, never ignored."""
    p = argparse.ArgumentParser()
    p.add_argument("--tensor", default="",
                   help="npz with X, months, lats, lons, chan (family4 r3); "
                        "OBS is derived as isfinite(X) exactly as train.py's "
                        "LazyPixels does. Required unless --smoke.")
    p.add_argument("--holdout-scope", default="window",
                   help="window is the only scope implemented — see "
                        "ml/train_cone.py's module docstring; any other value "
                        "is refused.")
    p.add_argument("--holdout-years", default="2009,2017,2023",
                   help="train.py's flag, same meaning and same default.")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-latents", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-z", type=int, default=32)
    p.add_argument("--d-dec", type=int, default=256)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--n-fourier", type=int, default=8)
    p.add_argument("--L-in", type=int, default=6,
                   help="inner-window depth in pentads. 0 = the SNAPSHOT "
                        "ablation: no dots, the lag-0 patch only.")
    p.add_argument("--future-lags", default="1,2")
    p.add_argument("--n-dot-queries", type=int, default=256)
    # ---- E-069b's masking plan, mirroring ml/train_cone.py flag for flag ---
    # The docstring's contract is that a flag which cannot be honoured is
    # REFUSED, never ignored — so these are here BECAUSE the host draw and
    # the anchor-weight rule are both mirrored (draw_masks_np, and
    # ConeMAEJax.query_sets_given). Every default is today's behaviour.
    p.add_argument("--chan-drop-scope", default="all",
                   choices=("all", "lag0"),
                   help="how far a channel drop reaches: 'all' hides the "
                        "channel at lag 0 and at every dot, 'lag0' hides its "
                        "lag-0 patch only and leaves its dots visible.")
    p.add_argument("--lag-band-p", type=float, default=0.3)
    p.add_argument("--sector-p", type=float, default=0.3)
    p.add_argument("--anchor-hidden-only", action="store_true",
                   help="score the anchor family only on the channels that "
                        "were dropped for that batch element.")
    p.add_argument("--aux-latent-w", type=float, default=0.25,
                   help="weight of the auxiliary loss through the decoder's "
                        "FULL memory ([z-token] + latents).")
    p.add_argument("--eval-every", type=int, default=0, help="0 = steps//10")
    p.add_argument("--eval-anchors", type=int, default=1024)
    p.add_argument("--save-every", type=int, default=0, help="0 = steps//4")
    p.add_argument("--certify-n", type=int, default=4096)
    p.add_argument("--velocity-probe", action="store_true",
                   help="H1: ridge from z (cur_* dropped from the input) to "
                        "the anchor's cur_u/cur_v, year-blocked folds.")
    p.add_argument("--snapshot-ablation", action="store_true",
                   help="also train an L_in=0 twin in-process and probe it, "
                        "so the two arms share the probe anchors exactly.")
    p.add_argument("--probe-anchors", type=int, default=2048)
    p.add_argument("--out", default=os.path.join(ML, "runs", "cone_jax"))
    p.add_argument("--metrics", default="metrics.jsonl")
    p.add_argument("--smoke", action="store_true")
    # ---- the three this backend adds --------------------------------------
    p.add_argument("--ckpt-every", type=int, default=1000,
                   help="write <out>/ckpt_latest.npz (resumable: NNX params + "
                        "optax moments + step + the host RNG) every N steps "
                        "and at the end.")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/ckpt_latest.npz. Refuses if it "
                        "is not there rather than silently starting over.")
    p.add_argument("--stop-at", type=int, default=0,
                   help="stop after this step while keeping the SCHEDULE of "
                        "--steps (0 = off). The schedule is a function of "
                        "--steps, so a two-leg run can only be compared with a "
                        "one-leg run if both legs know the total; this is what "
                        "gate C8 uses, and what a preemptible node's operator "
                        "needs to reproduce a seam by hand.")
    a = p.parse_args(argv)
    if a.holdout_scope != "window":
        raise SystemExit(
            f"--holdout-scope {a.holdout_scope!r}: only 'window' is "
            f"implemented. The cone reads L_in pentads back and two forward, "
            f"so 'window' is the rule that keeps a held-out bin out of the "
            f"training set by any path (ml/cone_sampler.py::admissible). A "
            f"terminal or longitude scope would need its own admission test "
            f"and its own certificate; refusing rather than silently using "
            f"the window one under another name.")
    if a.smoke:
        # `ml/train_cone.py:parse`'s smoke block verbatim: --smoke FIXES THE
        # WHOLE CONFIGURATION, learning rate included, so the two backends'
        # smoke runs differ in the backend and in nothing else.
        a.steps, a.batch, a.lr = 200, 32, 2e-3
        a.d_model, a.n_heads, a.n_latents, a.n_layers = 64, 4, 16, 2
        a.d_dec, a.dec_layers, a.n_fourier = 64, 2, 6
        a.n_dot_queries = 48
        a.eval_every = a.eval_every or 50
        a.eval_anchors = min(a.eval_anchors, 256)
        a.probe_anchors = min(a.probe_anchors, 768)
        a.certify_n = min(a.certify_n, 4096)
        a.holdout_years = "1983"
        a.velocity_probe = True
        a.snapshot_ablation = True
    a.eval_every = a.eval_every or max(1, a.steps // 10)
    a.save_every = a.save_every or max(1, a.steps // 4)
    a.ckpt_every = a.ckpt_every or a.steps
    if a.stop_at and not (0 < a.stop_at <= a.steps):
        raise SystemExit(
            f"--stop-at {a.stop_at} with --steps {a.steps}: the stop must lie "
            f"inside the run. --steps is the TOTAL the schedule is written "
            f"against; --stop-at only says where this leg puts the seam.")
    return a


# ------------------------------------------------------- batches, on fixed --
_DOT_KEYS = ("vals", "obs", "valid", "chan", "dy_km", "dx_km", "lag_days",
             "depth")
_ROW_KEYS = ("patch_vals", "patch_obs", "fut_vals", "fut_obs", "ctx")


def max_dots(sampler):
    """`N_max` — the dot count of the WIDEST grid row, over the whole tensor.

    `ConeSampler.sample` pads to the widest row IN THE BATCH, which moves from
    step to step; XLA would recompile on every move. This is the one number
    that makes the step's shape a constant. It is an upper bound (rows with no
    ocean in them are included), which is the safe direction: a batch can never
    need more than this.
    """
    return max((sampler.n_dots(y) for y in range(sampler.H)), default=0)


def pad_batch_np(s, n_max, b_target=None):
    """`(batch padded to [b_target, n_max], n_real)`, still numpy.

    Padding is INERT, not merely ignored: a padded dot column has `valid=False`
    and `obs=False`, so it is outside the attention (the key-padding mask) and
    outside every query set; a padded ROW additionally has `patch_obs=False`
    and `fut_obs=False`, so every query it contributes has weight exactly 0 and
    `wsum`, `n_targets`, `nll`, `mse` and `logvar_mean` are all unchanged. That
    is why a short final eval batch can be padded to the training batch width
    instead of forcing a second XLA compile.
    """
    B, N = np.asarray(s["vals"]).shape
    if N > n_max:
        raise SystemExit(
            f"pad_batch_np: the sampler returned N={N} dots but the run was "
            f"sized for N_max={n_max}. N_max is the max over ALL grid rows, so "
            f"this cannot happen unless the sampler changed under the run.")
    b_target = int(b_target) if b_target else B
    if B > b_target:
        raise SystemExit(f"pad_batch_np: {B} rows into a {b_target}-row batch")
    out = {}
    for k in _DOT_KEYS:
        v = np.ascontiguousarray(s[k])
        z = np.zeros((b_target, n_max), v.dtype)
        z[:B, :N] = v
        out[k] = z
    for k in _ROW_KEYS:
        v = np.ascontiguousarray(s[k])
        z = np.zeros((b_target,) + v.shape[1:], v.dtype)
        z[:B] = v
        out[k] = z
    return out, B


def to_jax(bn, chan_depth):
    """The padded numpy batch as the jnp arrays `ConeMAEJax` reads."""
    out = {k: jnp.asarray(v) for k, v in bn.items()}
    out["chan_depth"] = jnp.asarray(chan_depth)
    return out


def plan_to_np(plan):
    """The scalars/arrays `draw_masks_np` needs out of a torch `default_plan`."""
    return {"chan_drop_p": np.asarray(plan["chan_drop_p"], np.float64),
            "lag_band_p": float(plan.get("lag_band_p", 0.0)),
            "sector_p": float(plan.get("sector_p", 0.0)),
            # E-069b. Mirrored here because the masks are drawn on the HOST:
            # a scope the numpy draw did not know about would silently keep
            # hiding every dot of a dropped channel while the torch twin
            # stopped, and the parity gate would be the only thing that saw
            # it. `anchor_hidden_only` is NOT here — it is read by
            # `ConeMAEJax.query_sets_given` out of the jax plan, not by the
            # draw.
            "chan_drop_scope": str(plan.get("chan_drop_scope", "all")),
            "n_dot_queries": int(plan.get("n_dot_queries", 0))}


def draw_masks_np(b_np, plan_np, rng):
    """`(chan_mask [B, C], dot_mask [B, N], (idx [B, k], sel [B, k]))`, numpy.

    `ConeMAE._masks` and `ConeMAE.draw_dot_queries`, DISTRIBUTION for
    distribution, on a `np.random.default_rng`:

      * channel drop — `rng.random((B, C)) < chan_drop_p`, then, under
        `chan_drop_scope == "all"`, `dot_mask = chan_mask[b, chan]` (torch's
        `gather(1, chan)`). Under `"lag0"` the channel drop contributes
        NOTHING to `dot_mask` — the dropped channel's dots stay visible —
        and, exactly as on the torch side, the same random numbers are drawn
        either way;
      * lag band — on with probability `lag_band_p`, `l0` uniform on {1, 2, 3},
        hiding every dot with `lag_days <= 5 * l0`;
      * sector — on with probability `sector_p`, `th0 ~ U(0, 2*pi)`, hiding
        `remainder(atan2(dx_km, dy_km) - th0, 2*pi) < pi/2`;
      * dot queries — among `dot_mask & obs & valid`, a UNIFORM RANDOM SUBSET
        of size `k = min(n_dot_queries, N)`. torch takes the top-k of scores
        that are `rand` where eligible and −1 where not, which is exactly a
        uniform subset plus, in a row with fewer than k eligible dots, some
        ineligible indices carrying `sel=False`. This does the same with an
        argsort, so a short row pads with `sel=False` on whatever index sorts
        next and the query's weight is 0.

    The stream is NOT shared with torch and the gate does not pretend it is
    (§8.6): what is shared is the distribution.
    """
    vals = np.asarray(b_np["vals"])
    B, N = vals.shape
    p = np.asarray(plan_np["chan_drop_p"], np.float64)
    C = int(p.shape[0])
    chan_mask = rng.random((B, C)) < p[None, :]
    chan = np.asarray(b_np["chan"]).astype(np.int64)
    scope = str(plan_np.get("chan_drop_scope", "all"))
    if scope not in ("all", "lag0"):
        raise ValueError(f"chan_drop_scope {scope!r}: expected 'all' or "
                         f"'lag0' (ml/cone_codec.py::ConeMAE._masks).")
    dot_mask = (np.take_along_axis(chan_mask, chan, axis=1)
                if (N and scope == "all") else np.zeros((B, N), bool))

    lag_band_p = float(plan_np.get("lag_band_p", 0.0))
    if lag_band_p > 0.0:
        on = rng.random(B) < lag_band_p
        l0 = rng.integers(1, 4, B)                       # uniform on {1, 2, 3}
        band = np.asarray(b_np["lag_days"]) <= (5.0 * l0)[:, None]
        dot_mask = dot_mask | (band & on[:, None])

    sector_p = float(plan_np.get("sector_p", 0.0))
    if sector_p > 0.0:
        on = rng.random(B) < sector_p
        th0 = rng.random(B) * (2.0 * math.pi)
        th = np.arctan2(np.asarray(b_np["dx_km"]), np.asarray(b_np["dy_km"]))
        d = np.remainder(th - th0[:, None], 2.0 * math.pi)
        dot_mask = dot_mask | ((d < (math.pi / 2.0)) & on[:, None])

    k = min(int(plan_np.get("n_dot_queries", 0)), int(N))
    if k <= 0 or N == 0:
        return (chan_mask, dot_mask,
                (np.zeros((B, 0), np.int32), np.zeros((B, 0), bool)))
    elig = (dot_mask & np.asarray(b_np["obs"], bool)
            & np.asarray(b_np["valid"], bool))
    score = np.where(elig, rng.random((B, N)), -1.0)
    idx = np.argsort(-score, axis=1, kind="stable")[:, :k]
    sel = np.take_along_axis(score, idx, axis=1) >= 0.0
    return chan_mask, dot_mask, (idx.astype(np.int32), sel)


def masks_to_jax(cm, dm, di):
    return (jnp.asarray(cm), jnp.asarray(dm),
            (jnp.asarray(di[0]), jnp.asarray(di[1])))


# ------------------------------------------------------------- checkpoints --
def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def save_state_npz(path, opt, step, args, rng_state, mrng_state, curve):
    """`ckpt_latest.npz` — NNX params + optax moments + step + the host RNGs.

    `nnx.state(optimizer)` carries the model's params, the optax state and the
    optimiser's own update count in ONE tree whose leaf order is a
    deterministic function of the architecture, so a load rebuilds the
    structure first and refuses on any count or shape it did not expect
    (`ml/jaxport/train_field.py:save_state_npz`'s discipline, unchanged).

    BOTH numpy Generators ride along: the anchor stream and the mask stream are
    what make a resume BIT-IDENTICAL rather than merely similar (gate C8).
    Written to a temp sibling and `os.replace`d — flush, THEN publish
    (`ml/CLAUDE.md` §5.21) — because the shipping launcher watches this name.
    """
    st = nnx.state(opt)
    ls = _leaves(st)
    blob = {"_step": np.asarray(int(step)),
            "_args": np.asarray(json.dumps(args)),
            "_rng": np.asarray(json.dumps(rng_state)),
            "_mrng": np.asarray(json.dumps(mrng_state)),
            "_curve": np.asarray(json.dumps(curve)),
            "_n": np.asarray(len(ls))}
    for i, v in enumerate(ls):
        blob[f"s{i}"] = np.asarray(v)
    # A FILE HANDLE, not a name: np.savez appends ".npz" to a path that does
    # not end in it, so savez(path + ".tmp") would write path.tmp.npz.
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(tmp, "wb") as fh:
        np.savez(fh, **blob)
    os.replace(tmp, path)
    return path


def load_state_npz(path, opt):
    """Read `save_state_npz` back INTO the structure already built.

    Returns `(step, args, rng_state, mrng_state, curve)` and updates `opt` (and
    with it the model it holds) in place.
    """
    z = np.load(path, allow_pickle=False)
    st = nnx.state(opt)
    ls = _leaves(st)
    if int(z["_n"]) != len(ls):
        raise SystemExit(
            f"REFUSING to resume {path}: it holds {int(z['_n'])} state leaves "
            f"and this codec has {len(ls)}. The architecture or the optimiser "
            f"differs from the one that wrote it.")
    new = []
    for i, v in enumerate(ls):
        arr = z[f"s{i}"]
        if tuple(arr.shape) != tuple(np.shape(v)):
            raise SystemExit(f"REFUSING to resume {path}: state leaf {i} is "
                             f"{arr.shape}, the codec wants {np.shape(v)}")
        new.append(jnp.asarray(arr, jnp.asarray(v).dtype))
    st2 = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(st), new)
    nnx.update(opt, st2)
    return (int(z["_step"]), json.loads(str(z["_args"])),
            json.loads(str(z["_rng"])), json.loads(str(z["_mrng"])),
            json.loads(str(z["_curve"])))


# ------------------------------------------------------------ results file --
class Results:
    """`<out>/results.json`, written at every eval and marked partial.

    `ml/CLAUDE.md` §5.25: a compute step longer than ~30 min writes its result
    file incrementally, atomically (temp sibling + `os.replace`, so a reader
    never catches a half-written file) and marked with a top-level
    `in_progress` key whose ABSENCE is the run's only completion certificate.
    """

    def __init__(self, path, args):
        self.path = path
        self.args = dict(vars(args)) if not isinstance(args, dict) else dict(args)
        self.arms = {}
        self.probe = None

    def arm(self, tag, **kw):
        self.arms.setdefault(tag, {}).update(kw)

    def flush(self, in_progress=None):
        payload = {}
        if in_progress is not None:
            payload["in_progress"] = in_progress
        payload["backend"] = "jax"
        payload["args"] = self.args
        payload["arms"] = self.arms
        payload["velocity_probe"] = self.probe
        d = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, self.path)
        return self.path


def fresh_run_refusal(ck_npz, out_dir):
    """The `--resume`-with-nothing-to-resume message.

    Checked in `main` BEFORE the tensor is loaded and the codec built: the
    precondition depends only on the inputs, and `ml/CLAUDE.md` §5.16 is
    explicit that a guard which depends only on the inputs belongs where the
    inputs are all it has cost you. `train_one` keeps the same refusal as a
    backstop, because it is also reachable per arm.
    """
    return SystemExit(
        f"--resume: there is no {ck_npz}. THIS IS A FRESH RUN — nothing has "
        f"been trained into {out_dir} yet, so there is no optimiser state, no "
        f"schedule position and no RNG stream to continue. Drop --resume to "
        f"start one (a fresh run is not a resume with an empty checkpoint: it "
        f"would restart the cosine schedule and re-draw every anchor, and the "
        f"metrics file would show a seam that never happened).")


def _finite_or_die(name, *vals):
    """§5.22: never write NaN into a results file — stop instead."""
    for v in vals:
        if v is None:
            continue
        if not math.isfinite(float(v)):
            raise SystemExit(
                f"REFUSING to continue: {name} went non-finite ({v}). A "
                f"results file full of NaN is loud enough to notice and quiet "
                f"enough to misattribute (ml/CLAUDE.md §5.22).")


# ---------------------------------------------------------------- training --
def build_steps(plan_j):
    """The jitted step, eval and encode functions, and a COMPILE COUNTER.

    `plan_j` is CLOSED OVER rather than passed: `loss_given` reads
    `float(plan["aux_latent_w"])`, which is a python float and cannot be a
    tracer, and the plan is constant for a run anyway.

    The counter is incremented in the traced body, which executes only while
    XLA is tracing — so `compiles["train"]` is literally the number of times
    the step was compiled, and a value above 1 after step 2 means the shapes
    are not fixed (the whole reason `N_max` exists).
    """
    compiles = {"train": 0, "eval": 0, "encode": 0}

    @nnx.jit
    def train_step(model, optimizer, batch, chan_mask, dot_mask, dot_idx):
        compiles["train"] += 1

        def loss_fn(m):
            loss, _z, terms = m.loss_given(batch, plan_j, chan_mask, dot_mask,
                                           dot_idx)
            return loss, terms

        (loss, terms), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        # torch's `clip_grad_norm_(1.0)` THEN `AdamW.step()` is exactly the
        # order of the optax chain this optimiser was built with.
        optimizer.update(grads)
        return loss, terms

    @nnx.jit
    def eval_step(model, batch, chan_mask, dot_mask, dot_idx):
        compiles["eval"] += 1
        _loss, _z, terms = model.loss_given(batch, plan_j, chan_mask, dot_mask,
                                            dot_idx)
        return terms

    @nnx.jit
    def encode_step(model, batch, chan_mask, dot_mask):
        compiles["encode"] += 1
        bb = dict(batch)
        bb["chan_mask"], bb["dot_mask"] = chan_mask, dot_mask
        return model.encode(bb)[0]

    return train_step, eval_step, encode_step, compiles


def eval_loss(model, eval_step, sampler, anchors, plan_np, chan_depth, batch,
              n_max, seed=12345):
    """Held-out loss on a FIXED anchor set with a FIXED mask draw.

    `ml/train_cone.py:eval_loss`'s semantics exactly: the generator is
    RE-SEEDED at every eval, so two evals differ only in the weights — the
    curve measures the model, not which channels the dice hid — and the three
    numbers are weighted by `wsum`, which is the denominator both means were
    divided by.
    """
    rng = np.random.default_rng(int(seed))
    nll = mse = w = tgt = 0.0
    for i in range(0, len(anchors), batch):
        s = sampler.sample(anchors[i:i + batch])
        bn, _n_real = pad_batch_np(s, n_max, batch)
        cm, dm, di = draw_masks_np(bn, plan_np, rng)
        terms = eval_step(model, to_jax(bn, chan_depth),
                          *masks_to_jax(cm, dm, di))
        n = float(terms["wsum"])
        nll += float(terms["nll"]) * n
        mse += float(terms["mse"]) * n
        w += n
        tgt += float(terms["n_targets"])
    w = max(w, 1e-6)
    return nll / w, mse / w, tgt


def train_one(a, D, L_in, out_dir, metrics_name, ckpt_name, npz_name, tag,
              results, eval_anchors=None):
    """Train one arm (the cone codec, or its L_in=0 snapshot twin).

    Mirrors `ml/train_cone.py:train_one` step for step and record for record;
    returns dict(model, sampler, curve, certificate, params, ...).
    """
    # THE INIT-FROM-TORCH RULE (§8.4). `nn.Embedding` is N(0, 1) and
    # `nnx.Embed` is std ~ 1/sqrt(d); `ml/jaxport/README.md:105-118` measured
    # what that difference costs downstream. So the JAX codec is NEVER
    # flax-initialised: it is built as a torch `ConeMAE` under
    # `torch.manual_seed(seed)` and converted. The order below is
    # `train_cone.train_one`'s order, and nothing between the seed and the
    # construction consumes torch randomness, so the two backends start from
    # bit-identical weights at the same seed.
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)          # anchors — torch's own stream
    mrng = np.random.default_rng(a.seed)         # masks — §8.6
    chan, C = D["chan"], D["C"]
    fut = tuple(int(v) for v in a.future_lags.split(",") if v.strip())
    sampler = ConeSampler(D["X"], D["OBS"], D["lats"], D["lons"], chan,
                          L_in=L_in, future_lags=fut)
    if isinstance(D["OBS"], FiniteView):
        print(f"[{tag}] OBS is a FiniteView — isfinite(X) is derived per "
              f"gather, never materialised", flush=True)
    train_bins = ~D["t_hold"]
    ts = admissible_bins(sampler, train_bins)
    ys, xs = np.nonzero(D["ocean"])
    if not len(ts):
        raise SystemExit(
            f"[{tag}] no bin is admissible under --holdout-scope window with "
            f"L_in={L_in} and future lags {fut}: every anchor's cone would "
            f"touch a held-out or out-of-archive bin. Widen the archive or "
            f"shrink L_in.")
    n_dots = sampler.n_dots(int(ys[0]))
    print(f"[{tag}] L_in={L_in} · {n_dots} dot tokens + {C} patch tokens per "
          f"anchor · admissible train bins {len(ts)}/{sampler.T} · "
          f"{len(ys):,} ocean cells", flush=True)

    # ---- self-certification (E-059's pattern), BEFORE anything is spent ----
    cert = draw_anchors(rng, ts, ys, xs, min(a.certify_n, 4096))
    bad = sampler.certify(cert, train_bins)
    print(f"[{tag}] pool certificate: {bad} violations in {len(cert)} drawn "
          f"anchors (window scope, bins t-{L_in}..t+{max(fut)})", flush=True)
    if bad:
        raise SystemExit(
            f"[{tag}] POOL VIOLATION: {bad} of {len(cert)} training anchors "
            f"read a bin outside the training set. Refusing to train — a "
            f"codec trained on a leaked holdout cannot be evaluated on it "
            f"(ml/plans/E069_cone_codec.md §3, 'pool discipline').")

    # ---- FIXED SHAPES ------------------------------------------------------
    n_max = max_dots(sampler)
    k_q = min(int(a.n_dot_queries), int(n_max))
    n_fut = len(fut)
    print(f"[{tag}] fixed shapes for XLA · N_max {n_max} dots (widest of "
          f"{sampler.H} rows) · k {k_q} dot queries · Q {C + C * n_fut + k_q} "
          f"· B {a.batch}", flush=True)

    tm = ConeMAE(C, d_model=a.d_model, n_heads=a.n_heads,
                 n_latents=a.n_latents, n_layers=a.n_layers, d_z=a.d_z,
                 d_dec=a.d_dec, dec_layers=a.dec_layers,
                 n_fourier=a.n_fourier)
    model = cone_from_torch(tm, rngs=nnx.Rngs(a.seed))
    params = model.param_count()
    if params != tm.param_count():
        raise SystemExit(
            f"[{tag}] the converted codec has {params:,} parameters and the "
            f"torch one has {tm.param_count():,} — a converter that loses a "
            f"tensor loads and trains and produces numbers.")
    # `cone_from_torch` hands the JAX model buffers that ALIAS the torch
    # module's storage on the CPU backend (tests/test_jaxport_cone.py,
    # FINDING 1 — `jnp.asarray` of a contiguous numpy view is zero-copy, and
    # 68 of the 99 tensors are contiguous at that point). It is SAFE HERE and
    # only here: `tm` is dropped immediately and never written again, the JAX
    # buffers keep the storage alive (measured: delete, gc, churn the
    # allocator — 0 tensors corrupted), and every optax update allocates
    # rather than writing through. Do not add a torch optimiser, a
    # `load_state_dict` or any in-place torch write after this line: it would
    # silently rewrite the weights this trainer is training.
    del tm
    print(f"[{tag}] ConeMAEJax {params:,} params ({params / 1e6:.3f}M)",
          flush=True)

    chan_depth = np.array([channel_depth_dbar(n) for n in chan], np.float32)
    plan = default_plan(chan, n_dot_queries=a.n_dot_queries,
                        aux_latent_w=a.aux_latent_w, future_lags=fut,
                        lag_band_p=a.lag_band_p, sector_p=a.sector_p,
                        chan_drop_scope=a.chan_drop_scope,
                        anchor_hidden_only=a.anchor_hidden_only,
                        device="cpu")
    plan_j = plan_to_jax(plan)
    plan_np = plan_to_np(plan)

    # ---- optimiser ---------------------------------------------------------
    # torch: AdamW(lr, wd=0.01) + CosineAnnealingLR(T_max=steps), the scheduler
    # stepped AFTER opt.step(), plus clip_grad_norm_(1.0) before it. optax's
    # `count` IS the number of updates already applied, so update s (1-indexed)
    # sees count s-1 and `cosine_decay_schedule(lr, steps, alpha=0)` returns
    # lr * 0.5 * (1 + cos(pi * (s-1) / steps)) — torch's sequence exactly.
    sched = optax.cosine_decay_schedule(init_value=a.lr, decay_steps=a.steps,
                                        alpha=0.0)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=sched, b1=0.9, b2=0.999, eps=1e-8,
                    weight_decay=0.01))
    opt = nnx.Optimizer(model, tx)
    train_step, eval_step, encode_step, compiles = build_steps(plan_j)

    # ---- the held-out anchor set ------------------------------------------
    # Dots from held-out bins are ALLOWED here — that is what makes it a
    # held-out measurement rather than a second training pool.
    if eval_anchors is None:
        ev_ts = np.flatnonzero(D["t_hold"])
        ev_ts = ev_ts[(ev_ts - L_in >= 0) & (ev_ts + max(fut) < sampler.T)]
        if not len(ev_ts):
            raise SystemExit(f"[{tag}] no held-out bin has a complete cone")
        eval_anchors = draw_anchors(np.random.default_rng(a.seed + 991),
                                    ev_ts, ys, xs, a.eval_anchors)

    metrics_path = (os.path.join(out_dir, metrics_name)
                    if metrics_name and not os.path.isabs(metrics_name)
                    else metrics_name)

    def m2(rec):
        if not metrics_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(metrics_path)) or ".",
                    exist_ok=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- resume ------------------------------------------------------------
    ck_npz = os.path.join(out_dir, npz_name)
    step0, curve = 0, []
    resumed = False
    if a.resume:
        if os.path.exists(ck_npz):
            # The GEOMETRY check first, off the checkpoint's own `args`: it
            # costs one small string out of the npz and its message names the
            # flag, where the leaf-shape refusal below can only name a shape.
            sav = json.loads(str(np.load(ck_npz, allow_pickle=False)["_args"]))
            for kk in ("d_model", "n_heads", "n_latents", "n_layers", "d_z",
                       "d_dec", "dec_layers", "n_fourier", "steps"):
                if kk in sav and int(sav[kk]) != int(getattr(a, kk)):
                    raise SystemExit(
                        f"REFUSING to resume {ck_npz}: it was written with "
                        f"--{kk.replace('_', '-')} {sav[kk]} and this run asks "
                        f"for {getattr(a, kk)}. A resume derives nothing it "
                        f"can be told wrongly (ml/CLAUDE.md §1) — and --steps "
                        f"in particular IS the cosine schedule, so continuing "
                        f"under a different total is a different run.")
            step0, _sav, rng_state, mrng_state, curve = load_state_npz(ck_npz,
                                                                       opt)
            rng.bit_generator.state = rng_state
            mrng.bit_generator.state = mrng_state
            if step0 >= a.steps:
                print(f"[{tag}] checkpoint is already at step {step0:,} of "
                      f"{a.steps:,} — nothing left to train on this arm",
                      flush=True)
            else:
                print(f"[{tag}] RESUMED from {ck_npz} at step {step0:,} "
                      f"(params, optimiser moments, schedule position and BOTH "
                      f"host RNG streams restored)", flush=True)
            resumed = True
            # BEFORE the config record: the field trainer's convention, and
            # what lets status.html keep the seam across the reset-on-config.
            m2({"resumed": {"at_step": int(step0),
                            "from": os.path.basename(ck_npz),
                            "to_step": int(a.steps), "arm": tag,
                            "backend": "jax"}})
        elif tag == "cone":
            raise fresh_run_refusal(ck_npz, out_dir)
        else:
            print(f"[{tag}] --resume: no {os.path.basename(ck_npz)} for this "
                  f"arm — training it fresh (the cone arm resumed; the twin "
                  f"had not started)", flush=True)

    # `ml/train.py`'s RECORD FAMILY, key for key where the keys mean the same
    # thing (ml/CLAUDE.md §0d). status.html's parseJsonl resets on `config`,
    # charts {step, loss_rec, loss_nei}, and renders any other {step, ...}
    # record as a probe line — so this trainer needs no page change either.
    m2({"config": {
        "steps": a.steps, "batch": a.batch, "d_z": a.d_z, "patch": 3,
        "d_model": a.d_model, "n_layers": a.n_layers,
        "n_heads": a.n_heads, "d_dec": a.d_dec, "anomaly": True,
        "eval_every": a.eval_every, "light_probe_every": 0,
        "params_M": round(params / 1e6, 3),
        "data": os.path.basename(a.tensor), "C": int(C),
        "T": int(sampler.T), "resume": (os.path.basename(ck_npz)
                                        if resumed else None),
        "recipe": os.environ.get("RECIPE_NAME") or None,
        "trainer": "cone", "arm": tag, "L_in": int(L_in),
        "n_latents": a.n_latents, "n_dot_tokens": int(n_dots),
        "future_lags": list(fut), "aux_latent_w": a.aux_latent_w,
        # E-069b's masking plan, the same keys the torch trainer records.
        "chan_drop_scope": a.chan_drop_scope,
        "lag_band_p": a.lag_band_p, "sector_p": a.sector_p,
        "anchor_hidden_only": bool(a.anchor_hidden_only),
        "holdout_scope": a.holdout_scope,
        "holdout_years": a.holdout_years,
        "lr": a.lr, "seed": a.seed,
        # the one mark a TPU-trained number carries (ml/CLAUDE.md §3b)
        "backend": "jax",
    }})

    ckpt_path = os.path.join(out_dir, ckpt_name)

    def save_pt(step):
        """The torch-loadable twin, the SAME 8-key blob `train_cone.save`
        writes, so the probe and the eval ladder read a TPU-trained codec
        exactly as they read a GPU-trained one."""
        export_cone_pt(model, vars(a), ckpt_path, chan_names=chan,
                       norm=D["norm"], step=int(step), arm=tag,
                       L_in=int(L_in), params=params)

    def save_ckpt(step):
        save_state_npz(ck_npz, opt, step, dict(vars(a)),
                       rng.bit_generator.state, mrng.bit_generator.state,
                       curve)
        save_pt(step)

    def emit(step, wall):
        results.arm(tag, curve=curve, params=int(params), L_in=int(L_in),
                    steps=int(a.steps), step=int(step),
                    certificate={"anchors": int(len(cert)),
                                 "violations": int(bad)},
                    n_dot_tokens=int(n_dots), n_max=int(n_max),
                    wall_s=round(wall, 1))
        results.flush(in_progress={"arm": tag, "step": int(step),
                                   "of": int(a.steps)})

    loss_every = max(1, a.steps // 200)
    stop = int(a.stop_at) if a.stop_at else a.steps
    t0 = time.time()

    nll0, mse0, n0 = eval_loss(model, eval_step, sampler, eval_anchors,
                               plan_np, chan_depth, a.batch, n_max)
    _finite_or_die(f"[{tag}] held-out nll at step {step0}", nll0, mse0)
    curve.append({"step": int(step0), "held_out_nll": nll0,
                  "held_out_mse": mse0, "train_nll": None})
    print(f"[{tag}] step {step0} · held-out nll {nll0:+.4f} mse {mse0:.4f} "
          f"({int(n0):,} targets)", flush=True)
    m2({"step": int(step0), "held_out_nll": round(nll0, 5),
        "held_out_mse": round(mse0, 5), "held_out_targets": int(n0),
        "wall_s": round(time.time() - t0, 1)})
    emit(step0, time.time() - t0)

    step_wall = []
    terms = {}
    for s in range(step0 + 1, stop + 1):
        anchors = draw_anchors(rng, ts, ys, xs, a.batch)
        bn, _nr = pad_batch_np(sampler.sample(anchors), n_max, a.batch)
        cm, dm, di = draw_masks_np(bn, plan_np, mrng)
        ts0 = time.time()
        loss, terms = train_step(model, opt, to_jax(bn, chan_depth),
                                 *masks_to_jax(cm, dm, di))
        lv = float(loss)                      # the sync torch's `float()` is
        if len(step_wall) < 3:
            step_wall.append(time.time() - ts0)
        if not np.isfinite(lv):
            raise SystemExit(
                f"[{tag}] non-finite loss at step {s} — stopping rather than "
                f"writing NaN (ml/CLAUDE.md §5.22)")
        if s == step0 + 3 and compiles["train"] != 1:
            raise SystemExit(
                f"[{tag}] the training step compiled {compiles['train']} times "
                f"in three steps. It must compile ONCE: every batch is padded "
                f"to N_max={n_max} and k={k_q} for exactly that reason, so a "
                f"second compile means a shape is still moving (§8.6).")
        if s % loss_every == 0 or s == a.steps:
            m2({"step": s, "loss_rec": round(float(terms["nll"]), 5),
                "loss_nei": round(float(terms["mse"]), 5)})
        if s % a.eval_every == 0 or s == a.steps or s == stop:
            nll, mse, n = eval_loss(model, eval_step, sampler, eval_anchors,
                                    plan_np, chan_depth, a.batch, n_max)
            _finite_or_die(f"[{tag}] held-out nll at step {s}", nll, mse)
            curve.append({"step": s, "held_out_nll": nll, "held_out_mse": mse,
                          "train_nll": float(terms["nll"])})
            print(f"[{tag}] step {s:>6}/{a.steps} · train nll "
                  f"{float(terms['nll']):+.4f} mse {float(terms['mse']):.4f} "
                  f"· held-out nll {nll:+.4f} mse {mse:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            m2({"step": s, "held_out_nll": round(nll, 5),
                "held_out_mse": round(mse, 5), "held_out_targets": int(n),
                "wall_s": round(time.time() - t0, 1)})
            emit(s, time.time() - t0)
        if s % a.save_every == 0:
            save_pt(s)
        if s % a.ckpt_every == 0:
            save_ckpt(s)
    save_ckpt(stop)

    if len(step_wall) >= 2:
        print(f"[{tag}] step 1 {step_wall[0]:.2f}s (trace + compile) vs step 2 "
              f"{step_wall[1]:.3f}s · train_step compiled "
              f"{compiles['train']}x, eval_step {compiles['eval']}x",
              flush=True)
    return dict(model=model, sampler=sampler, curve=curve, params=params,
                certificate={"anchors": int(len(cert)), "violations": int(bad)},
                eval_anchors=eval_anchors, chan_depth=chan_depth,
                plan_np=plan_np, encode_step=encode_step, n_max=n_max,
                ckpt=ckpt_path, npz=ck_npz, stopped_at=stop, opt=opt,
                compiles=compiles)


# ------------------------------------------------------------ velocity probe --
def velocity_probe(res, chan, anchors, months, batch=64):
    """H1: ridge from z to (cur_u, cur_v) with the cur_* channels DROPPED.

    `ml/train_cone.py:velocity_probe`'s semantics on the JAX encoder: "dropped"
    = hidden, i.e. `mask_tok` — the same token a channel drop uses during
    training, so the probe's input distribution is one the codec has seen. The
    target is the anchor's own value in anomaly space (the patch centre),
    scored only where it was observed, and `kfold_r2` is `train_cone`'s own
    (which is `probe_kfold.kfold_r` where it imports), so the number is
    comparable with every other probe in the programme.
    """
    model, sampler = res["model"], res["sampler"]
    encode_step, n_max = res["encode_step"], res["n_max"]
    chan_depth = res["chan_depth"]
    C = len(chan)
    cur = np.array([n.startswith("cur_") for n in chan], bool)
    zs, tg, ob = [], [], []
    for i in range(0, len(anchors), batch):
        s = sampler.sample(anchors[i:i + batch])
        bn, n_real = pad_batch_np(s, n_max, batch)
        cm = np.broadcast_to(cur[None], (batch, C)).copy()
        dm = (np.take_along_axis(cm, bn["chan"].astype(np.int64), axis=1)
              if n_max else np.zeros((batch, 0), bool))
        z = np.asarray(encode_step(model, to_jax(bn, chan_depth),
                                   jnp.asarray(cm), jnp.asarray(dm)))
        zs.append(z[:n_real])
        tg.append(bn["patch_vals"][:n_real, :, 4])
        ob.append(bn["patch_obs"][:n_real, :, 4])
    Z = np.concatenate(zs)
    TG = np.concatenate(tg)
    OB = np.concatenate(ob)
    groups, how = fold_labels(anchors, months)
    out = {"folds": how, "n_anchors": int(len(anchors)), "d_z": int(Z.shape[1])}
    for name in ("cur_u", "cur_v"):
        if name not in chan:
            continue
        c = chan.index(name)
        m = OB[:, c] & np.isfinite(TG[:, c])
        if m.sum() < 32:
            out[name] = {"r2": float("nan"), "n": int(m.sum()),
                         "note": "too few observed targets"}
            continue
        out[name] = kfold_r2(Z[m], TG[m, c], groups[m])
    return out


# -------------------------------------------------------------------- main --
def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    # §5.16: a precondition that depends only on the inputs is checked while
    # the inputs are all it has cost you — not after the 36 GB anomaly
    # transform and the codec build.
    if a.resume and not os.path.exists(os.path.join(a.out, "ckpt_latest.npz")):
        raise fresh_run_refusal(os.path.join(a.out, "ckpt_latest.npz"), a.out)
    if a.smoke and not a.tensor:
        a.tensor = smoke_tensor(os.path.join(a.out, "smoke_tensor.npz"),
                                seed=a.seed)
        print(f"--smoke: synthetic tensor at {a.tensor}", flush=True)
    if not a.tensor:
        raise SystemExit("--tensor is required (or --smoke)")
    print(f"[cone] backend jax · devices "
          f"{[str(d) for d in jax.devices()]}", flush=True)
    D = load_data(a)

    results = Results(os.path.join(a.out, "results.json"), a)
    res = train_one(a, D, a.L_in, a.out, a.metrics, "cone_codec.pt",
                    "ckpt_latest.npz", "cone", results)
    print(f"[cone] checkpoint {res['ckpt']} (+ {res['npz']})", flush=True)

    probe = None
    if a.velocity_probe:
        # The PROBE ANCHORS ARE SHARED between the arms, drawn once here: two
        # probes on two anchor sets differ by the anchors as well as by the
        # codec, and H1 is a statement about the codecs.
        rng = np.random.default_rng(a.seed + 7717)
        ys, xs = np.nonzero(D["ocean"])
        fut = tuple(int(v) for v in a.future_lags.split(",") if v.strip())
        ts = np.arange(D["T"])
        ts = ts[(ts - max(a.L_in, 1) >= 0) & (ts + max(fut) < D["T"])]
        pa = draw_anchors(rng, ts, ys, xs, a.probe_anchors)
        # SORTED BY BIN, because probe_kfold.kfold_r picks its ridge lambda on
        # the LAST 20% of the training rows — an inner TAIL, which is a time
        # tail only if the rows are in time order. Handed a time-shuffled
        # anchor set it validates on an interleaved sample, sees no
        # autocorrelation, picks a lambda far too small and the outer fold
        # blows up (measured on the torch arm: out-of-fold r = -0.38 before
        # this line).
        pa = pa[np.argsort(pa[:, 0], kind="stable")]
        probe = {"cone": velocity_probe(res, D["chan"], pa, D["months"])}
        print(f"[probe] cone   cur_u R2 {probe['cone']['cur_u']['r2']:+.4f} · "
              f"cur_v R2 {probe['cone']['cur_v']['r2']:+.4f}", flush=True)
        if a.snapshot_ablation:
            snap = train_one(a, D, 0, a.out, "metrics_snapshot.jsonl",
                             "snapshot_codec.pt", "ckpt_latest_snapshot.npz",
                             "snapshot", results)
            probe["snapshot"] = velocity_probe(snap, D["chan"], pa,
                                               D["months"])
            print(f"[probe] snapshot cur_u R2 "
                  f"{probe['snapshot']['cur_u']['r2']:+.4f} · cur_v R2 "
                  f"{probe['snapshot']['cur_v']['r2']:+.4f}", flush=True)
            probe["delta_cur_u"] = (probe["cone"]["cur_u"]["r2"]
                                    - probe["snapshot"]["cur_u"]["r2"])
            probe["delta_cur_v"] = (probe["cone"]["cur_v"]["r2"]
                                    - probe["snapshot"]["cur_v"]["r2"])
        probe["L_in"] = int(a.L_in)
        probe["steps"] = int(a.steps)
        probe["seed"] = int(a.seed)
        probe["backend"] = "jax"
        path = os.path.join(a.out, "velocity_probe.json")
        with open(path, "w") as f:
            json.dump(probe, f, indent=2)
        print(f"[probe] wrote {path}", flush=True)
        results.probe = probe

    # THE FINAL WRITE DROPS `in_progress` — its absence is the run's only
    # completion certificate (ml/CLAUDE.md §5.25). A `--stop-at` leg has NOT
    # reached the end of the run it belongs to, so it keeps the marker: the
    # certificate must certify the run, not the invocation.
    partial = (a.stop_at and a.stop_at < a.steps)
    results.flush(in_progress=({"stopped_at": int(a.stop_at),
                                "of": int(a.steps)} if partial else None))
    with open(results.path) as f:
        chk = json.load(f)
    if ("in_progress" in chk) != bool(partial):
        raise SystemExit("the result file's in_progress marker does not match "
                         "whether the run finished — that key's absence is the "
                         "run's only completion certificate")

    print("\ncurve (step · train nll · held-out nll):", flush=True)
    for c in res["curve"]:
        tn = "     -" if c["train_nll"] is None else f"{c['train_nll']:+.4f}"
        print(f"  {c['step']:>6}  {tn}  {c['held_out_nll']:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
