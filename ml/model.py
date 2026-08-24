"""PixelMAE — a masked autoencoder over one pixel's channels, with a
queryable decoder that predicts the SAME pixel's masked channels and its
NEIGHBOURS in space and time from the bottleneck alone.

Shape of the idea (proposal §4–5, scaled to a pilot):

  channels of pixel (lat, lon, t)      ┌─ mask some observed channels
        │  one token per channel  ◄────┘
        ▼
  transformer encoder  (missing channels enter as explicit "missing" tokens —
        │               absence is information, not padding)
        ▼
  z = bottleneck(CLS)   d_z-dimensional; THE embedding
        ▼
  decoder(z, channel-id, Δlat, Δlon, Δt) → value
        Δ = (0,0,0)  reconstruct this pixel (masked channels score)
        Δ = space    predict the 4-neighbours' channels, same month
        Δ = time     predict this pixel's channels next/previous month

Why neighbour heads: a plain autoencoder can ace reconstruction by memorising
the seasonal cycle (proposal §5). Forcing z to answer for pixels it never saw
makes it carry STATE — the currency the AMOC probe then reads.

The decoder is a neural-field-style MLP conditioned on (z, query): it can be
asked for any offset at inference, which is what "use the embedding to predict
nearby pixels" means operationally.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# `ml/` on the path so `import fsq_ladder` works however this module was
# reached — bare `import model` (every script under ml/) or `ml.model` (the
# tests that import through the package path). ml/temporal.py and
# ml/rollout_spatial.py already do exactly this, for the same reason.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fsq_ladder as fql                                        # noqa: E402


class InputQuant:
    """FSQ-style scalar quantization — THE one implementation in this repo.

    It has two callers and they quantize different things. `ml/temporal.py`
    (E-044c, `--input-quant`) quantizes what the stage-2 HEAD READS, with a
    sigma MEASURED off an existing Z. `ml/model.py:PixelMAE` (E-046,
    `--fsq-levels`) quantizes the CODEC'S OWN BOTTLENECK, where there is no Z
    to measure — see `fsq_from_levels` below. The map, the level parsing, the
    even-L offset, the L=2 refusal and the straight-through gradient are the
    same object in both, which is why it lives HERE, in the module both sides
    import, rather than being copied. (This file is the bottom of the import
    graph: `temporal.py` imports `model.py`, never the other way round, so a
    second copy was the only alternative — and the standing rule against a
    second copy of one transform was written after the anomaly transform had
    four.)

    Chris's capacity hypothesis (2026-08-22): the stage-2 head has too much
    input for its capacity, and the pentad z is 32 continuous float32
    dimensions per pixel per step — at K=24 and stencil 145 that is 111,360
    real numbers entering one forward. FSQ (arxiv 2309.15505, "Finite Scalar
    Quantization: VQ-VAE Made Simple") is the cheapest way to ask "how much of
    that does the head actually need?": bound each dimension, round it to one
    of L levels, and pass the gradient straight through. No codebook, no
    commitment loss, no EMA, no dead-code problem — the paper's own claim is
    that this matches VQ-VAE at large codebook sizes while removing all of
    that machinery.

    ON THE HEAD SIDE, INPUTS ONLY, AND THE TARGET STAYS CONTINUOUS. What is
    under test there is what the head can READ, not what it must WRITE:
    quantizing the target as well would change the objective into "predict a
    bin index" and make every z-space MSE incomparable with every archived
    one. So the contract is quantized-state -> TRUE-next-state, and
    `val_zmse`, `z_t+1` and the roll's own numbers stay in the same units they
    have always been in.

    The map, per dimension d (`sigma_d` = that dimension's std over train
    bins, measured once at startup on the head side; identically 1 on the
    codec side), following the paper's own bound including its even/odd
    handling:

        half = (L-1)/2 ;  offset = 0.5 if L is even else 0
        shift = atanh(offset/half)                 (0 when L is odd)
        g = half * tanh(z/(2*sigma_d) + shift) - offset
        q = g + (round(g) - g).detach()            FSQ's straight-through
        z_q = (q + offset) * 2*sigma_d / half      back to z-space units

    THE OFFSET IS WHY "L LEVELS" IS TRUE. Without it an even L gives L-1
    reachable levels, because tanh never attains +/-1 and the two outermost
    integers are never rounded to: L=8 without the offset is 7 levels, which
    would make every bits/dim number in the log a small lie. With it, g lands
    in (-L/2, L/2 - offset) and round() reaches exactly L integers.

    The de-scale is LINEAR, not atanh: inverting the tanh would send the
    outermost levels to infinity, and the point of the bound is that the tails
    ARE the outermost levels. A dimension therefore saturates at
    +/-2*sigma_d, 95% of a Gaussian, which is a deliberate part of the
    capacity restriction rather than an accident of it.

    THE LADDER (E-048) — WHERE THOSE L LEVELS SIT. Everything above describes
    the `uniform` ladder, which is the default, is what every archived
    checkpoint has, and is bit-identical here: with `ladder="uniform"` the
    `__call__` below runs the same four statements it ran at 7f8dabb, and the
    exp branch is a Python `if` that is False. `ladder="exp"` places the L
    levels GEOMETRICALLY instead — sign(v)*a*c^j, saturating at the same
    +/-2*sigma_d — and `ladder="auto"` chooses per DIMENSION from a measured
    sample. The ladder arithmetic, the fit and the recorded-fit string live in
    `ml/fsq_ladder.py` because the JAX port reads the identical definitions;
    what stays here is the torch application of them, with its native
    straight-through gradient. Read that module for why `exp` REPLACES the
    tanh bound rather than composing with it, and why "for each channel"
    honestly means per z-dimension.

    `zero_passthrough` — EXACT ZEROS ARE PASSED THROUGH UNCHANGED, ON THE HEAD
    SIDE ONLY, and it is the one behavioural difference between the two
    callers. A zero in a STENCIL SLOT is not a small value, it is the absence
    of a neighbour (`zj[miss] = 0.0` in ml/rollout_spatial.py, and the same
    convention in the trainer's `live` mask for --input-znoise); an even L has
    no zero level, so without this the structural zeros would all become one
    small nonzero constant. The codec's bottleneck has no such convention —
    `to_z(h[:, 0])` is a dense linear output and an exact 0.0 there is an
    ordinary value that must land on the lattice like any other — so E-046
    constructs this with `zero_passthrough=False`. Defaulting it to True keeps
    every E-044c number reproducible.
    """

    def __init__(self, spec, sigma, d_z, flag="--input-quant",
                 zero_passthrough=True, ladder="uniform",
                 exp_base=fql.DEFAULT_EXP_BASE, fit=""):
        lv = [int(v) for v in fql.parse_levels(spec, d_z, flag)]
        self.spec = str(spec)
        self.flag = str(flag)
        self.zero_passthrough = bool(zero_passthrough)
        self.levels = torch.as_tensor(lv, dtype=torch.float32)
        self.sigma = torch.as_tensor(np.asarray(sigma, np.float32))
        assert self.sigma.shape == (d_z,), (self.sigma.shape, d_z)
        self.d_z = int(d_z)
        self._half = (self.levels - 1.0) / 2.0
        self._offset = torch.where(self.levels % 2 == 0,
                                   torch.full_like(self.levels, 0.5),
                                   torch.zeros_like(self.levels))
        self._shift = torch.atanh(self._offset / self._half)
        self._scale = 2.0 * self.sigma
        self.bits_per_dim = float(np.mean(np.log2(np.asarray(lv, float))))
        self.codebook_log2 = float(np.sum(np.log2(np.asarray(lv, float))))
        self.ladder = str(ladder or "uniform")
        self.exp_base = float(exp_base)
        _is_exp, _base, _fitted = fql.resolve(self.ladder, lv, self.d_z,
                                              self.exp_base, fit, flag=flag)
        # True only for an `auto` run that has not fitted yet: it quantizes
        # UNIFORMLY until the trainer measures the distribution, and the
        # trainer asks this rather than re-deriving the condition.
        self.needs_fit = not _fitted
        self._set_ladder(_is_exp, _base, fit=fit)

    def _set_ladder(self, is_exp, base, fit=""):
        """Install a per-dimension ladder. One place, so the constructor and
        `fit_auto` below cannot build the lattice two different ways."""
        self.fit = str(fit or "")
        self.is_exp_np = np.asarray(is_exp, bool)
        self.base_np = np.asarray(base, np.float64)
        self._any_exp = bool(self.is_exp_np.any())
        self._is_exp = torch.as_tensor(self.is_exp_np)
        lv = self.levels.to(torch.int64).numpy()
        n, a_rel, logc, has_zero = fql.exp_params(lv, self.base_np,
                                                  flag=self.flag + " exp")
        self._exp_n = torch.as_tensor(np.asarray(n, np.float32))
        self._exp_arel = torch.as_tensor(np.asarray(a_rel, np.float32))
        self._exp_logc = torch.as_tensor(np.asarray(logc, np.float32))
        self._exp_jmin = torch.as_tensor(
            np.where(has_zero, -1.0, 0.0).astype(np.float32))
        self._exp_zero = torch.as_tensor(np.asarray(has_zero, bool))
        return self

    def fit_auto(self, sample, bases=fql.AUTO_BASES):
        """MEASURE the per-dimension ladder on real pre-quantization values.

        `sample` is [N, d_z] — whatever the caller captured. Returns the one
        line the log prints; the caller writes `self.fit` into the checkpoint,
        and every later loader rebuilds from that string rather than
        re-fitting (a loader that re-fitted would score a different model)."""
        s = np.asarray(sample, np.float64).reshape(-1, self.d_z)
        lv = self.levels.to(torch.int64).numpy()
        sc = self._scale.detach().cpu().numpy().astype(np.float64)
        is_exp, base, mse_u, mse_b = fql.fit_auto(s, lv, sc, bases)
        dev = self.levels.device
        self._set_ladder(is_exp, base, fit=fql.format_fit(is_exp, base))
        self.needs_fit = False
        self.to(dev)
        return (f"{self.flag} auto fitted on {len(s):,} pre-quantization "
                f"vectors: " + fql.describe_fit(is_exp, base, mse_u, mse_b))

    def to(self, dev):
        for k in ("levels", "sigma", "_half", "_offset", "_shift", "_scale",
                  "_is_exp", "_exp_n", "_exp_arel", "_exp_logc", "_exp_jmin",
                  "_exp_zero"):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    def _exp(self, v):
        """The GEOMETRIC ladder, straight-through. See ml/fsq_ladder.py for
        the definition and for why it carries its own bound instead of
        sitting behind the tanh."""
        a = (self._exp_arel * self._scale).to(v.dtype).to(v.device)
        logc = self._exp_logc.to(v.dtype).to(v.device)
        n1 = (self._exp_n - 1.0).to(v.dtype).to(v.device)
        jmin = self._exp_jmin.to(v.dtype).to(v.device)
        one = torch.ones((), dtype=v.dtype, device=v.device)
        s = torch.where(v < 0, -one, one)
        m = torch.clamp(v.abs(), min=1e-30)
        g = torch.log(m / a) / logc
        gq = torch.maximum(torch.minimum(torch.round(g), n1), jmin)
        # Straight-through: the round and the clamp carry no gradient, so
        # d out/d v = |out|/|v| — exactly 1 on a level, within [c^-1/2, c^1/2]
        # in the interior, decaying as R/|v| under saturation. Measured in
        # tests/test_e048_fsq_ladders.py rather than asserted here.
        q = g + (gq - g).detach()
        out = s * a * torch.exp(q * logc)
        # The zero level of an odd L, with the plain straight-through
        # estimator: value 0, gradient 1.
        return torch.where(gq < -0.5, v - v.detach(), out)

    def __call__(self, z):
        """z [..., n*d_z] (the stencil flattens slots into the last axis)."""
        if z is None:
            return z
        shp = z.shape
        assert shp[-1] % self.d_z == 0, (shp, self.d_z)
        v = z.reshape(-1, self.d_z)
        half = self._half.to(v.dtype).to(v.device)
        sc = self._scale.to(v.dtype).to(v.device)
        off = self._offset.to(v.dtype).to(v.device)
        sh = self._shift.to(v.dtype).to(v.device)
        g = half * torch.tanh(v / sc + sh) - off
        q = g + (torch.round(g) - g).detach()          # FSQ straight-through
        out = (q + off) * sc / half
        # A PYTHON `if` ON A PYTHON BOOL: with the uniform ladder (the
        # default, and every archived checkpoint) nothing above or below this
        # line changed at E-048, which is what makes the bit-identity claim
        # bit-identity and not near-identity.
        if self._any_exp:
            out = torch.where(self._is_exp.to(v.device), self._exp(v), out)
        if self.zero_passthrough:
            out = torch.where(v == 0, v, out)
        return out.reshape(shp)

    def describe(self, zsample):
        """One line of what this costs, measured on real z rather than
        claimed: bits per dimension, the codebook size the paper counts, and
        the quantization MSE against the raw z it replaces."""
        with torch.no_grad():
            zq = self(zsample)
            mse = float((zq - zsample).pow(2).mean())
            var = float(zsample.pow(2).mean())
        return (f"{self.flag} {self.spec}: {self.bits_per_dim:.3f} bits/dim "
                f"x d_z {self.d_z} = {self.codebook_log2:.1f} bits "
                f"(codebook 2^{self.codebook_log2:.1f}) · quantization MSE vs "
                f"raw z {mse:.5f} on a mean-square {var:.5f} z "
                f"({100.0 * mse / max(var, 1e-12):.2f}% of its energy) · "
                f"sigma per dim {float(self.sigma.min()):.3f}.."
                f"{float(self.sigma.max()):.3f}")


def fsq_from_levels(spec, d_z, ladder="uniform",
                    exp_base=fql.DEFAULT_EXP_BASE, fit=""):
    """E-046: the CODEC-side FSQ bottleneck (`--fsq-levels`), or None.

    THE ONE REAL DIFFERENCE FROM THE HEAD-SIDE KNOB IS THE BOUND'S SCALE, and
    it is forced rather than chosen (plan §2.1, §8). `InputQuant` divides by a
    `sigma_d` MEASURED over an existing Z — it quantizes a Z that already
    exists. Here the thing being quantized IS what we are producing, so there
    is no Z to measure and no measurement that would not be circular. The FSQ
    paper's own answer is to bound the raw pre-quantization activation with
    `tanh` and let the encoder learn the scale, so `sigma_d = 1` for every
    dimension and the scale is learned into `to_z` (a plain `nn.Linear`, whose
    output scale is exactly a free parameter). Everything else — the level
    map, the even-L half-step offset that makes "L levels" true, the L=2
    refusal, the linear de-scale, the straight-through gradient — is
    `InputQuant`'s, unchanged, because it is the same transform.

    Consequences worth stating rather than discovering: a dimension saturates
    at +/-2 (the tails ARE the outermost levels, by design), and z_q is
    therefore bounded, which every downstream consumer sees only as "the
    values happen to live on a lattice" — `ml/temporal.py` still receives a
    [T, P, d_z] float array and nothing about its shape or dtype moved.

    `zero_passthrough=False`: see InputQuant's docstring. There is no
    absent-neighbour convention in a codec bottleneck, so an exact 0.0 rounds
    onto the lattice like any other value.

    E-048 adds the LADDER (`--fsq-ladder`), which is orthogonal to everything
    above: `uniform` is this docstring unchanged and bit-identical, `exp`
    places the same L levels geometrically inside the same +/-2 bound, and
    `auto` fits the choice per z-dimension on a measured sample. The bound's
    scale stays sigma = 1 for the same reason it does today — there is no Z to
    measure — so on the codec side a level's position is a pure function of
    the ladder, and `to_z` learns the scale into it either way.
    """
    spec = str(spec or "").strip()
    if not spec:
        return None
    return InputQuant(spec, np.ones(int(d_z), np.float32), int(d_z),
                      flag="--fsq-levels", zero_passthrough=False,
                      ladder=ladder, exp_base=exp_base, fit=fit)


class PixelMAE(nn.Module):
    def __init__(self, n_chan, d_model=128, n_heads=4, n_layers=4,
                 d_z=32, d_dec=256, max_abs_offset=3, patch=1, dec_layers=2,
                 k_time=1, fsq_levels="", fsq_ladder="uniform",
                 fsq_exp_base=fql.DEFAULT_EXP_BASE, fsq_ladder_fit=""):
        super().__init__()
        self.n_chan = n_chan
        self.d_z = d_z
        # E-046 FSQ BOTTLENECK. "" (the default) is None, and None is the
        # continuous bottleneck every archived checkpoint has — `_bottleneck`
        # then returns its argument, the same object, so a run that does not
        # name the flag is bit-identical. The quantizer holds plain tensors,
        # not Parameters or buffers, so `state_dict()` is key-for-key what it
        # has always been and `.to(device)` needs no change (InputQuant moves
        # its constants onto the input's device per call).
        #
        # THE DISPATCHED ARM IS d_z 32 WITH L=8 ON EVERY DIMENSION — [8]^32,
        # NOT the paper's option A [8,8,8,6,5] at d_z 5 that ml/plans/
        # E046_fsq_codec.md §4(a) recommends. Two measurements overrule the
        # plan, both made after it was written:
        #   · The round-6 reconstruction audit found 40 channels ALREADY
        #     squeezed at d_z 32 — Argo-bin round-trip FVU 15-18% against
        #     0.4-0.5% on Argo-free bins, because 40 channels compete for 32
        #     dimensions. Dropping to 5 dimensions compounds a KNOWN failure
        #     rather than testing the hypothesis; the plan's own §8 flags
        #     40 -> 5 (an 8x channel-to-dimension ratio, where the monthly
        #     anchor runs 39 -> 64) as its strongest open question.
        #   · E-045-A9 (#446) MEASURED the head-side 8-levels-per-dimension
        #     alphabet on the SAME 32-dimensional z: 0.4916 against controls
        #     at 0.5056 and 0.50447. The winning alphabet is [8]^32, and
        #     E-046 exists to move exactly that intervention from the head
        #     into the codec — so the arm keeps d_z at 32 and changes ONE
        #     thing. Shrinking d_z at the same time would confound the
        #     bottleneck's ALPHABET with its WIDTH and no single arm could
        #     attribute the result.
        # The paper's small-codebook configurations remain runnable through
        # this same flag ("--fsq-levels 8,8,8,6,5" with --d-z 5) as a later
        # arm; nothing here forecloses them.
        # E-048: the LADDER is part of the same architecture field. It is
        # carried on the module (not only inside the quantizer) because
        # `codec_from_ckpt` reads it back out of `ck["args"]` for every eval,
        # and because a run whose ladder is `auto` writes its FITTED ladder
        # into the checkpoint — the fit is what the codec IS, in exactly the
        # sense the level counts are.
        self.fsq_levels = str(fsq_levels or "")
        self.fsq_ladder = str(fsq_ladder or "uniform")
        self.fsq_exp_base = float(fsq_exp_base)
        self.fsq_ladder_fit = str(fsq_ladder_fit or "")
        self.fsq = fsq_from_levels(self.fsq_levels, d_z,
                                   ladder=self.fsq_ladder,
                                   exp_base=self.fsq_exp_base,
                                   fit=self.fsq_ladder_fit)
        if self.fsq is None and (self.fsq_ladder != "uniform"
                                 or self.fsq_ladder_fit):
            # A ladder with no levels quantizes nothing, so it would be a
            # setting that appears to apply and does not — the failure the
            # recipe mechanism exists to remove, one layer down.
            raise SystemExit(
                f"--fsq-ladder {self.fsq_ladder!r} without --fsq-levels: "
                f"there is no bottleneck to put a ladder on. Name the levels "
                f"(e.g. --fsq-levels 8) or leave the ladder at 'uniform'.")
        self.max_off = max_abs_offset
        # E-047 TIME BLOCKS. k_time = 1 is the per-bin codec every archived
        # checkpoint is, and it adds NO parameter and NO branch that runs:
        # `time_emb` is created only when k_time > 1, so a k_time=1 model's
        # state_dict is key-for-key what it has always been. k_time > 1 makes
        # the encoder's input a k_time x C GRID of cells — one month of pentad
        # bins, say — whose cell token is the channel embedding plus a learned
        # WITHIN-BLOCK TIME-OFFSET embedding, so the encoder can tell the 3rd
        # pentad of the month from the 5th. Missing and PAD cells go through
        # the existing miss_tok path unchanged, which is the whole design:
        # Argo present in one bin of six is not a special case, it is six
        # cells of which five are unobserved.
        self.k_time = int(k_time)
        # patch=1: one value per channel token (the pilot design).
        # patch=3: each channel token carries its 3x3 neighbourhood — 9
        # values + 9 observed flags through one projection. Same token
        # count, but the encoder can SEE GRADIENTS (thermal wind is a
        # density gradient), and z becomes a true compression
        # (~160 observed values -> d_z at 25 channels).
        self.patch = patch
        p2 = patch * patch

        # --- encoder tokens -------------------------------------------------
        self.val_proj = (nn.Linear(1, d_model) if patch == 1
                         else nn.Linear(2 * p2, d_model))
        self.chan_emb = nn.Embedding(n_chan, d_model)
        if self.k_time > 1:
            self.time_emb = nn.Embedding(self.k_time, d_model)
        self.mask_tok = nn.Parameter(torch.zeros(d_model))     # channel masked by US
        self.miss_tok = nn.Parameter(torch.zeros(d_model))     # channel unobserved in the DATA
        self.cls_tok = nn.Parameter(torch.zeros(d_model))
        # coords/season enter as one non-maskable context token:
        # [sin m, cos m, lat/90, lon/180]
        self.ctx_proj = nn.Linear(4, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.to_z = nn.Linear(d_model, d_z)

        # --- queryable decoder ---------------------------------------------
        # query = channel id + integer offset (Δlon, Δlat, Δmonth), each
        # embedded; decoder never sees the input values, only z.
        self.q_chan = nn.Embedding(n_chan, 64)
        self.q_off = nn.Embedding(2 * max_abs_offset + 1, 16)  # shared per axis
        # E-047, decided 2026-08-22 (option b): WITHIN-BLOCK POSITION GETS ITS
        # OWN QUERY EMBEDDING. The alternative was to reuse the `dt` slot of
        # `off`, which fits (a 7-cell block centred is dt in -3..+3, exactly
        # this table's range) and costs nothing — and would make ONE index
        # mean two things, "one bin later inside this block" for a cell query
        # and "one block later" for the neighbour loss, with nothing in the
        # input telling the decoder which. One symbol, one meaning. `off`'s
        # dt keeps meaning BLOCKS, which is what the axis means downstream.
        if self.k_time > 1:
            self.q_time = nn.Embedding(self.k_time, 16)
        # dec_layers = HIDDEN layers. The historical decoder (2 hidden) is a
        # ~1.3M-param MLP against a 40M encoder; E-019a measured the round
        # trip losing 6.9% of deep-temperature variance, so the depth is now
        # a knob (E-019b). dec_layers=2 reproduces every old checkpoint
        # exactly — codec_from_ckpt defaults it for args written before the
        # knob existed.
        dec = [nn.Linear(d_z + 64 + 3 * 16 + (16 if self.k_time > 1 else 0),
                         d_dec), nn.GELU()]
        for _ in range(dec_layers - 1):
            dec += [nn.Linear(d_dec, d_dec), nn.GELU()]
        dec += [nn.Linear(d_dec, 1)]
        self.decoder = nn.Sequential(*dec)
        for p in (self.mask_tok, self.miss_tok, self.cls_tok):
            nn.init.normal_(p, std=0.02)

    def _bottleneck(self, z):
        """`to_z`'s output, quantized when E-046's flag is on.

        ONE LINE OF FORWARD, at the one place `z` is born (plan §2.1). Every
        `encode_pre` branch — patch>1, k_time>1 (E-047 month blocks), and the
        per-bin path — returns into here, so the block codec composes with FSQ
        for free and neither knob had to learn about the other: a block's z is
        still one `to_z(h[:, 0])` and quantizing it is the same operation on
        the same tensor.

        With the flag OFF this returns its ARGUMENT, not a copy — no cast, no
        clone, no branch that touches the graph — which is what makes the
        default-off claim bit-identity rather than near-identity."""
        return z if self.fsq is None else self.fsq(z)

    def encode(self, x, obs, mask, ctx):
        """patch=1: x, obs [B,C] · patch>1: x, obs [B,C,patch²] (obs = that
        cell observed; the channel counts as observed iff its CENTER is).
        mask [B,C] bool masked-by-training · ctx [B,4] → z [B,d_z].

        ONE LINE, because `encode_pre` is the whole encoder and
        `_bottleneck` is the whole quantizer. The split exists so the E-048
        `auto` fit can MEASURE the pre-quantization distribution through the
        real encoder — the thing it has to fit — without a capture hook
        reaching into the forward pass, and so the JAX mirror can do the same
        under jit, where a side-effecting hook does not work at all."""
        return self._bottleneck(self.encode_pre(x, obs, mask, ctx))

    def encode_pre(self, x, obs, mask, ctx):
        """`encode` WITHOUT the bottleneck: `to_z`'s raw output."""
        # WIDEN THE INPUT TO THE WEIGHTS' DTYPE. Family 4 is the project's
        # first float16 tensor, and every reader hands the codec a batch whose
        # dtype follows the tensor — `torch.from_numpy(np.nan_to_num(X))` did,
        # and LazyPixels preserves that faithfully. Against float32 weights
        # that is an immediate
        #     RuntimeError: mat1 and mat2 must have the same dtype,
        #                   but got Half and Float
        # on the first forward pass. Run #365 never reached it: the host OOM
        # killer took the process during the preamble, so the pentad arm would
        # have died here on the re-dispatch instead, one failure later, after
        # another tensor build. Caught on a 48x12x14 toy in seconds
        # (ml/CLAUDE.md §4.8) rather than on a rented GPU.
        #
        # Here rather than at the six call sites: probe_kfold, temporal,
        # rollout, probe_sequence, ablate_channels and train all build their
        # own value tensor, and a cast per call site is five chances to miss
        # one. float16 -> float32 is exact, and on family 2/3 (float32) it is
        # a no-op, so no existing run's arithmetic moves.
        wdt = self.val_proj.weight.dtype
        if x.dtype != wdt:
            x = x.to(wdt)
        if ctx.dtype != wdt:
            ctx = ctx.to(wdt)
        if self.patch > 1:
            B, C, P2 = x.shape
            ce = self.chan_emb.weight[None, :, :].expand(B, -1, -1)
            feat = torch.cat([x * obs, obs.to(x.dtype)], -1)   # [B,C,2·P2]
            vt = self.val_proj(feat) + ce
            obs = obs[..., P2 // 2]                            # center defines the channel
            vt = torch.where((obs & ~mask).unsqueeze(-1), vt, torch.zeros_like(vt))
            vt = vt + torch.where((obs & mask).unsqueeze(-1),
                                  self.mask_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
            vt = vt + torch.where((~obs).unsqueeze(-1),
                                  self.miss_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
            toks = torch.cat([self.cls_tok.expand(B, 1, -1),
                              self.ctx_proj(ctx).unsqueeze(1), vt], dim=1)
            h = self.encoder(toks)
            return self.to_z(h[:, 0])
        if self.k_time > 1:
            # x, obs, mask [B, k_time, C] -> [B, k_time*C] tokens, cell (j, c)
            # carrying chan_emb[c] + time_emb[j]. Flattened j-major so a
            # reader of the attention map sees each bin's channels together.
            B, KT, C = x.shape
            assert KT == self.k_time, (KT, self.k_time)
            x = x.reshape(B, KT * C)
            obs = obs.reshape(B, KT * C)
            mask = mask.reshape(B, KT * C)
            ce = (self.chan_emb.weight[None, None, :, :]
                  + self.time_emb.weight[None, :, None, :]).reshape(
                      1, KT * C, -1).expand(B, -1, -1)
            vt = self.val_proj(x.unsqueeze(-1)) + ce
            vt = torch.where((obs & ~mask).unsqueeze(-1), vt,
                             torch.zeros_like(vt))
            vt = vt + torch.where((obs & mask).unsqueeze(-1),
                                  self.mask_tok.expand(B, KT * C, -1) + ce,
                                  torch.zeros_like(vt))
            vt = vt + torch.where((~obs).unsqueeze(-1),
                                  self.miss_tok.expand(B, KT * C, -1) + ce,
                                  torch.zeros_like(vt))
            toks = torch.cat([self.cls_tok.expand(B, 1, -1),
                              self.ctx_proj(ctx).unsqueeze(1), vt], dim=1)
            return self.to_z(self.encoder(toks)[:, 0])
        B, C = x.shape
        ce = self.chan_emb.weight[None, :, :].expand(B, -1, -1)
        vt = self.val_proj(x.unsqueeze(-1)) + ce
        vt = torch.where((obs & ~mask).unsqueeze(-1), vt, torch.zeros_like(vt))
        vt = vt + torch.where((obs & mask).unsqueeze(-1),
                              self.mask_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
        vt = vt + torch.where((~obs).unsqueeze(-1),
                              self.miss_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
        toks = torch.cat([self.cls_tok.expand(B, 1, -1),
                          self.ctx_proj(ctx).unsqueeze(1), vt], dim=1)
        h = self.encoder(toks)
        return self.to_z(h[:, 0])

    def query(self, z, chan_idx, off, tpos=None):
        """z [B,d_z] · chan_idx [B,Q] · off [B,Q,3] ints in [-max,max] → [B,Q].

        `tpos` [B,Q] is the WITHIN-BLOCK cell position and is required exactly
        when k_time > 1 — a block codec that was asked for "channel c" without
        saying WHICH CELL would be answering a question with no answer, so it
        raises rather than picking one."""
        B, Q = chan_idx.shape
        qc = self.q_chan(chan_idx)
        qo = self.q_off(off + self.max_off).reshape(B, Q, -1)
        zq = z.unsqueeze(1).expand(-1, Q, -1)
        parts = [zq, qc, qo]
        if self.k_time > 1:
            if tpos is None:
                raise ValueError(
                    f"query(): this codec has k_time={self.k_time}, so every "
                    f"query names a cell (tpos [B,Q] in 0..{self.k_time - 1}) "
                    f"as well as a channel")
            parts.append(self.q_time(tpos))
        return self.decoder(torch.cat(parts, dim=-1)).squeeze(-1)


class LazyPixels:
    """A drop-in stand-in for `nan_to_num(X)` / `isfinite(X)` that derives
    them PER BATCH instead of materialising either at full size.

    WHY. `ml/train.py` used to build both eagerly:

        Xt  = torch.from_numpy(np.nan_to_num(X, nan=0.0))
        OBS = torch.from_numpy(np.isfinite(X))

    For family 3 ([516, 281, 481, 39]) that is 13.6 GB alongside X and nobody
    noticed. For family 4's pentad tensor ([3142, 281, 481, 39] float16) it is
    **33.1 GB for X + 33.1 GB for the copy + 16.6 GB for the mask = 82.8 GB**
    against a 64 GB box, and run #365 was killed by the host OOM killer (exit
    137) after six hours. Measured, not modelled — the element count is
    16,562,358,618.

    Both arrays are pure functions of X evaluated elementwise, and every
    consumer only ever indexes a BATCH of pixels out of them. So computing
    them after the index rather than before is arithmetically identical and
    costs a few hundred KB instead of 49.7 GB. This removes the failure mode
    rather than guarding it (ml/CLAUDE.md §4.1); the daily tensor is 5x larger
    again and would not have fitted any box under the old shape.

    Behaviour is preserved by construction: the SAME numpy functions are
    applied to the SAME elements, and dtype follows X exactly as
    `torch.from_numpy` did — so a float16 tensor still yields float16 and a
    float32 one still yields float32. `tests/test_train_lazy_pixels.py` pins
    that against the eager arrays elementwise.
    """

    def __init__(self, X, obs=False):
        self._X = X
        self._obs = obs
        self.shape = X.shape          # gather_px reads .shape[1], .shape[2]

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, idx):
        # Consumers index with torch CPU tensors; numpy needs arrays.
        if isinstance(idx, tuple):
            idx = tuple(np.asarray(i) if hasattr(i, "numpy") else i for i in idx)
        elif hasattr(idx, "numpy"):
            idx = np.asarray(idx)
        raw = self._X[idx]
        if self._obs:
            return torch.from_numpy(np.isfinite(raw))
        return torch.from_numpy(np.nan_to_num(raw, nan=0.0))


def gather_px(Xt, OBS, t, y, x, patch):
    """Gather encoder inputs for pixels (t, y, x) from full tensors.
    patch=1 → ([B,C], [B,C]) as before. patch>1 → ([B,C,patch²],
    [B,C,patch²]): each channel's neighbourhood, longitude WRAPPED (the
    globe is periodic in x), latitude clamped with the out-of-range rows
    marked unobserved. Center cell is index patch²//2."""
    if patch == 1:
        return Xt[t, y, x], OBS[t, y, x]
    H, W = Xt.shape[1], Xt.shape[2]
    r = patch // 2
    vs, os_ = [], []
    for dy in range(-r, r + 1):
        yy = (y + dy).clamp(0, H - 1)
        vy = ((y + dy) >= 0) & ((y + dy) <= H - 1)
        for dx in range(-r, r + 1):
            xx = (x + dx) % W
            vs.append(Xt[t, yy, xx])
            os_.append(OBS[t, yy, xx] & vy.unsqueeze(-1))
    return torch.stack(vs, -1), torch.stack(os_, -1)


def codec_from_ckpt(ck, n_chan):
    """Rebuild the EXACT architecture a checkpoint was trained with.

    Every loader used to hand-construct PixelMAE(n_chan, d_z, patch) — fine
    while all codecs shared one size, silently wrong the day they didn't.
    The checkpoint's args carry the full architecture (train.py saves
    vars(a)); this is the one place that reads them. Old checkpoints predate
    the size knobs, so every .get() default is the pilot architecture."""
    a = ck.get("args", {})
    # E-046. THE BOTTLENECK IS PART OF THE ARCHITECTURE, and it is the part
    # that carries no parameters — so a loader that ignored it would build a
    # model whose `load_state_dict` SUCCEEDS, whose parameter count matches to
    # the byte, and whose embeddings are a different function of the input.
    # Every eval on this programme reads z: trainprobe's light and full
    # probes, probe_kfold, probe_head, recon_eval, temporal.py's embed and
    # rollout_spatial's re-encode guard. An eval that silently dropped the
    # quantizer would be scoring a DIFFERENT MODEL from the one that trained,
    # and nothing in its output would say so. That failure is exactly
    # ml/CLAUDE.md §0.2 — "a step that reports success is not evidence it did
    # anything" — so this reads the levels back and hands them to the model.
    fsq = str(a.get("fsq_levels", "") or "")
    # E-048: the LADDER travels with the levels, and the FITTED ladder of an
    # `auto` run travels with both. A loader that read the levels and dropped
    # the fit would rebuild a uniform lattice under an `auto` checkpoint's
    # name — the same silent-drop failure the levels themselves are read back
    # for, one field along. NOTHING HERE RE-FITS: `fsq_ladder_fit` is the
    # measurement the run made, and re-measuring it at eval time would score a
    # different model from the one that trained.
    fsq_ladder = str(a.get("fsq_ladder", "uniform") or "uniform")
    fsq_fit = str(a.get("fsq_ladder_fit", "") or "")
    fsq_base = float(a.get("fsq_exp_base", fql.DEFAULT_EXP_BASE) or
                     fql.DEFAULT_EXP_BASE)
    if fsq and fsq_ladder == "auto" and not fsq_fit:
        raise SystemExit(
            "codec_from_ckpt: this checkpoint says --fsq-ladder auto but "
            "carries no `fsq_ladder_fit`, so the per-dimension ladder it "
            "trained with is not recorded. Refusing rather than guessing a "
            "lattice: re-run the fit is not an option at eval time (it would "
            "measure this tensor's activations, not the ones that trained), "
            "and quantizing uniformly would score a different model.")
    # And REFUSE what this code does not understand, rather than ignoring it.
    # A checkpoint written by a later revision that adds (say) `fsq_bound` or
    # `fsq_groups` would otherwise load here as a plain FSQ codec and be
    # scored as one. Unknown-key refusal is cheap, fires at load, and costs
    # the inputs alone (§0.3).
    known = {"fsq_levels", "fsq_ladder", "fsq_exp_base", "fsq_ladder_fit",
             "fsq_auto_n", "fsq_auto_step"}
    unknown = sorted(k for k in a if str(k).startswith("fsq_")
                     and k not in known)
    if unknown:
        raise SystemExit(
            f"codec_from_ckpt: this checkpoint carries FSQ argument(s) "
            f"{unknown} that this revision of ml/model.py does not implement. "
            f"Refusing rather than rebuilding a codec whose bottleneck is not "
            f"the one that was trained — the state_dict would load cleanly and "
            f"every z would be wrong. Update ml/model.py, or score this "
            f"checkpoint with the revision that wrote it.")
    return PixelMAE(n_chan=n_chan, d_z=ck["d_z"],
                    k_time=int(a.get("k_time", 1) or 1),
                    patch=a.get("patch", 1),
                    d_model=a.get("d_model", 128),
                    n_layers=a.get("n_layers", 4),
                    n_heads=a.get("n_heads", 4),
                    d_dec=a.get("d_dec", 256),
                    dec_layers=a.get("dec_layers", 2),
                    fsq_levels=fsq, fsq_ladder=fsq_ladder,
                    fsq_exp_base=fsq_base, fsq_ladder_fit=fsq_fit)


def obs_any_chunked(X, min_chan=2, chunk=64):
    """`np.isfinite(X).sum(-1) >= min_chan`, without the full-size temporaries.

    Identical values, elementwise. What changes is the peak: the one-liner
    materialises a [T,H,W,C] bool AND a [T,H,W] int64 at once — **15.4 GiB
    plus 3.4 GiB on the pentad tensor, 77 GiB plus 17 GiB at daily** — and
    both are live simultaneously.

    That spike is the first place `ml/train.py` can die, and it was invisible
    in the diagnosis of run #365: it is transient, so an RSS delta column
    shows nothing, and only VmHWM records it (`ml/measure_train_memory.py`
    measured 85.2 GiB resident against a 146.9 GiB peak). LazyPixels removed
    the two RESIDENT copies and left this one untouched, which would have
    OOM-killed the re-dispatch on the same 63 GB box for a different reason
    — the classic "fixed the term you can see".

    A chunk of 64 timesteps costs 337 MB regardless of T, so this is the term
    that stops scaling with the tensor. `np.count_nonzero(..., axis=-1)` is
    the same reduction `.sum(-1)` performs on a bool, spelled so it cannot
    accidentally accumulate in the input dtype.
    """
    out = np.empty(X.shape[:3], bool)
    for i in range(0, X.shape[0], chunk):
        sl = X[i:i + chunk]
        out[i:i + chunk] = np.count_nonzero(np.isfinite(sl), axis=-1) >= min_chan
    return out


def pool_idx(mask, chunk=256):
    """`np.where(mask)` as int32 triples, in the identical order.

    Two savings, both structural rather than clever:

      · **int32, not int64.** These arrays are indices into a [T,H,W] volume
        whose largest axis is 15,706 at daily cadence, so int64 spends exactly
        half its bytes on sign-extension. Family 4's train pool is ~272M
        pixels — 6.5 GiB as int64, 3.3 GiB as int32 — and it stays resident
        for the whole run.
      · **chunked over T**, so the int64 array numpy builds internally is
        1/12th of the pool at a time rather than all of it. Counted first and
        written into a preallocated output rather than concatenated: a
        concatenate holds the parts AND the result at once, which doubles
        exactly the term this is trying to halve.

    Order is preserved exactly because the chunks partition the FIRST axis in
    ascending order and `np.where` returns C-order within each chunk;
    concatenating them reproduces the global C-order listing. `tests/
    test_train_pool_memory.py` asserts equality against `np.where` rather
    than trusting that argument.
    """
    n = int(np.count_nonzero(mask))
    ts, ys, xs = (np.empty(n, np.int32) for _ in range(3))
    o = 0
    for i in range(0, mask.shape[0], chunk):
        t, y, x = np.where(mask[i:i + chunk])
        k = len(t)
        ts[o:o + k], ys[o:o + k], xs[o:o + k] = t + i, y, x
        o += k
    if o != n:
        raise AssertionError(f"pool_idx filled {o} of {n} — the chunk walk "
                             f"and the count disagree, which can only mean "
                             f"the mask changed underneath")
    return ts, ys, xs
