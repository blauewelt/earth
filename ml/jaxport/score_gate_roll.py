#!/usr/bin/env python3
"""`rollout_spatial.py`'s twelve-month roll, end to end, with a SWAPPABLE
ROLLING BACKEND — the instrument that scores gate G3.

G3 (`ml/plans/JAX_PORT.md` §5): the `e017_u1_s0` stage-2 head re-rolls at gate
AUC **0.643**, corridor **0.589**, window **0.622**, reproduced identically in
eighteen separate eval runs (#228 … #413) — the single most reproducible
number in the archive, and `ml/CLAUDE.md` §3b's own example of PROTOCOL
DETERMINISM. That is exactly what makes it the right gate for a port: it is
not a replicate, it is a certificate that a fixed checkpoint through a fixed
protocol returns a fixed number, so anything that moves it is the port.

**Everything except the roll is IMPORTED, not copied.** `corridor_pixels`,
`dilate8`, `StdMonths`, `ar1_train`, `month_feats`, `new_sums`, `accumulate`,
`skill_block`, `rapid_section`, `build_stencil`, `stream_stats`, `build_slab`
all come from the operational modules. The read-out is therefore literally the
same code on both sides, which is what makes the comparison a measurement of
the backend rather than of a transcription — the same discipline that made the
embedding slice's 5.7e-8 mean something (`score_section_probe.py`).

**Z is a COMMON INPUT, on purpose.** Both backends read the published
`embed-cache-v1` array (`Z_6c52f0687b_adcbe700fb.npy`, the canonical Z for
this codec weight hash on this tensor fingerprint) and the same cached static
identity `Zstat`. Those are EMBEDDINGS, and the embedding path already has its
own gate — G2′, where the two backends agreed to 5.7e-8 in r. Re-measuring it
here would fold two independent differences into one number and G3 would stop
isolating the rollout, which is the only thing it is for.

**Run the torch backend FIRST and require it to land on the archive.** A JAX
number scored while the torch reproduction is off target would be scored
against the wrong thing. A mismatch is a finding, never a tolerance to widen
(`ml/CLAUDE.md` §3b).

    python3 ml/jaxport/score_gate_roll.py --x /tmp/f3_X.npy \
        --npz-small /tmp/f3_meta.npz --z /tmp/Z.npy --ckpt /tmp/f3_anchor.pt \
        --head /tmp/e017_u1_s0__temporal.pt --backend torch \
        --out /tmp/roll_torch.json --zhat-out /tmp/zhat_torch.npy
    python3 ml/jaxport/score_gate_roll.py ... --backend jax \
        --out /tmp/roll_jax.json --zhat-out /tmp/zhat_jax.npy \
        --compare-zhat /tmp/zhat_torch.npy

**`--scope` is a COST decision and it is recorded in the artefact.** One roll
step is one forward over every rolled pixel, and the protocol rolls 234 of
them (three holdout years × twelve staggered starts, truncated at the year
end). MEASURED on two CPU cores, at the gate scope's 864 pixels: **7.1 s per
step (torch, 27.6 min) and 8.6 s per step (jax, 33.4 min)**. The cost is
linear in the rolled pixel count — the head forward is per-pixel — so
`--scope window`'s 84,405 pixels are ~45 h and ~54 h respectively. That is
not a run, it is a rental.

The reduction is EXACT for the gate scope and only for it, and the reason is
structural rather than statistical: at stencil 1 (which every E-017 head is)
`roll_step` has no cross-pixel term at all — the window, the static context,
the decode, the AR1 baseline and `accumulate`'s sums are all per-pixel — so
the gate scope's sums are bit-identical whether the other 83,540 pixels were
rolled beside them or not. **The corridor and window scopes are NOT
recoverable this way**: an AUC over the subset of the corridor that happens to
lie in the gate's 600-pixel draw is a different quantity, and this driver
refuses to print one under a scope name it does not have. It says which scopes
it scored, and why the others are absent.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path.insert(0, ML)

import torch                                                   # noqa: E402

# The read-out, imported wholesale. Nothing below re-implements any of it.
from rollout_spatial import (TimeAxis, StdMonths, ar1_train,   # noqa: E402
                             corridor_pixels, month_feats,
                             new_sums, accumulate, skill_block,
                             roll_step, decode_all, gate_for_cadence,
                             GATE_HEAD, GATE_REF, GATE_TOL)
from recon_eval import stream_stats, build_slab                # noqa: E402
from temporal import (TemporalTransformer as TorchTemporal,    # noqa: E402
                      build_stencil, rapid_section,
                      embed_everything, _ring_on)
from model import codec_from_ckpt, gather_px                   # noqa: E402

# The pre-registered constants this script exists to be scored against.
# Quoted from the archive, never recomputed — `rollout_spatial.GATE_REF` is
# the same 0.643 and is imported above rather than restated, and the other two
# scopes come from the same eighteen runs.
PREREG = {"gate": GATE_REF["auc"], "corridor": 0.589, "window": 0.622,
          "tol": GATE_TOL, "head": GATE_HEAD,
          "source": "18 eval runs #228 … #413 (ml/CLAUDE.md §3b, "
                    "'protocol determinism')"}


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
class TorchBackend:
    """The operational path, called through `rollout_spatial`'s own
    functions. Nothing here is a re-implementation: this class is a thin
    holder so the two backends present the same three methods."""

    name = "torch"

    def __init__(self, ck, head_ck, C, d_z, stencil, k_max, ring_km):
        ta = head_ck["args"]
        dir_ = tuple(int(x) for x in
                     str(ta.get("direct") or "").split(",") if x.strip())
        self.model = TorchTemporal(d_z=d_z, d_model=ta["d_model"], n_heads=4,
                                   n_layers=ta["layers"], k_max=k_max,
                                   direct=dir_, stencil=stencil)
        self.model.load_state_dict(head_ck["model"])
        self.model.eval()
        self.codec = codec_from_ckpt(ck, C)
        self.codec.load_state_dict(ck["model"])
        self.codec.eval()

    def window(self, Zwin_np):
        return torch.from_numpy(Zwin_np)

    def roll(self, Zwin, NBR, sctx, mfeat_np, chunk):
        nbr = None if NBR is None else torch.as_tensor(NBR)
        with torch.no_grad():
            return roll_step(self.model, Zwin, nbr,
                             torch.from_numpy(sctx),
                             torch.from_numpy(mfeat_np), chunk)

    def slide(self, Zwin, zhat):
        return torch.cat([Zwin[:, 1:], zhat[:, None]], 1)

    def decode(self, zhat, C, chunk):
        with torch.no_grad():
            return decode_all(self.codec, zhat, C, chunk)

    def to_numpy(self, zhat):
        return zhat.numpy()


class JaxBackend:
    """`ml/jaxport` — the second implementation. Same three methods, same
    inputs, same chunking."""

    name = "jax"

    def __init__(self, ck, head_ck, C, d_z, stencil, k_max, ring_km):
        import jax.numpy as jnp
        from jaxport import convert as jc, models as jm
        from jaxport import roll as jr
        self._jnp, self._jr = jnp, jr
        ta = head_ck["args"]
        dir_ = tuple(int(x) for x in
                     str(ta.get("direct") or "").split(",") if x.strip())
        self.model = jm.TemporalTransformer(d_z=d_z, d_model=ta["d_model"],
                                            n_heads=4, n_layers=ta["layers"],
                                            k_max=k_max, direct=dir_,
                                            stencil=stencil)
        jc.load_temporal(head_ck["model"], self.model)
        self.codec = jc.codec_from_ckpt_jax(ck, C)

    def window(self, Zwin_np):
        return self._jnp.asarray(Zwin_np)

    def roll(self, Zwin, NBR, sctx, mfeat_np, chunk):
        return self._jr.roll_step_jax(self.model, Zwin, NBR, sctx,
                                      mfeat_np, chunk)

    def slide(self, Zwin, zhat):
        return self._jnp.concatenate([Zwin[:, 1:], zhat[:, None]], 1)

    def decode(self, zhat, C, chunk):
        return self._jr.decode_all_jax(self.codec, zhat, C, chunk)

    def to_numpy(self, zhat):
        return np.asarray(zhat, dtype=np.float32)


# --------------------------------------------------------------------------
def static_identity(ck, C, Xm, moy, clim, dyn, mean_c, std_c, ys, xs, coords,
                    cache):
    """`Zstat` — the codec embedding of each pixel's STATIC channels alone.

    Computed with TORCH on both backends, and cached to disk so the two runs
    read the identical bytes. It is an EMBEDDING, exactly like Z: the
    embedding path is scored by G2′ and folding it back in here would stop G3
    isolating the rollout (see the module docstring). It is also 84,405
    encoder forwards through a 40.7M patch-3 codec, which is not something to
    pay twice.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if z.shape != (len(ys), ck["d_z"]):
            sys.exit(f"Zstat cache {cache} is {z.shape}, wanted "
                     f"{(len(ys), ck['d_z'])} — it was written for a "
                     f"different pixel set. Delete it or name another "
                     f"--cache-dir; a silently reused one would hand the two "
                     f"backends different static contexts.")
        print(f"Zstat: {cache} ({z.shape[0]:,} px, cached)", flush=True)
        return z
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval()
    x0 = np.asarray(Xm[0]).astype(np.float32)
    obs0 = np.isfinite(x0)
    for c in dyn:
        x0[..., c] = ((x0[..., c] - clim[moy[0], :, :, c]
                       - mean_c[c]) / (std_c[c] + 1e-6))
    Xt0 = torch.from_numpy(np.where(obs0, x0, 0.0)[None])
    stat_obs = torch.from_numpy(obs0).clone()
    for c in dyn:
        stat_obs[..., c] = False
    P = len(ys)
    zs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, P, 4096):
            sl = slice(i, min(i + 4096, P))
            n = sl.stop - sl.start
            ctx = np.concatenate([np.zeros((n, 2), np.float32),
                                  coords[sl]], 1)
            if getattr(codec, "patch", 1) > 1:
                vv, oo = gather_px(Xt0, stat_obs[None],
                                   torch.zeros(n, dtype=torch.long),
                                   torch.as_tensor(ys[sl]),
                                   torch.as_tensor(xs[sl]), codec.patch)
            else:
                vv = Xt0[0, ys[sl], xs[sl]]
                oo = stat_obs[ys[sl], xs[sl]]
            zs.append(codec.encode(vv, oo,
                                   torch.zeros(n, C, dtype=torch.bool),
                                   torch.as_tensor(ctx)).numpy())
    Zstat = np.concatenate(zs, 0)
    if cache:
        np.save(cache, Zstat)
    print(f"Zstat: encoded {P:,} px in {time.time() - t0:.0f}s"
          + (f" → {cache}" if cache else ""), flush=True)
    return Zstat


def verify_z(ck, C, d_z, Xm, Zm, moy, clim, dyn, mean_c, std_c, lats, lons,
             ys, xs, sec_y, sec_sel, T):
    """`rollout_spatial`'s own Z-cache guard, verbatim in intent: re-encode a
    handful of section pixels live and demand the published cache agrees.

    A silent ordering mismatch between the tensor on disk and the downloaded Z
    would roll beautiful nonsense, and the gate would then be scored against a
    different experiment. Cheap (eight pixels), so it runs before anything
    expensive (`ml/CLAUDE.md` §0.3).
    """
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval()
    Zsec = np.asarray(Zm[:, sec_sel]).astype(np.float32)
    rows3 = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs_sl = build_slab(Xm, rows3, moy, clim, dyn, mean_c, std_c)
    slab_t = torch.from_numpy(np.nan_to_num(slab, nan=0.0))
    obs_t = torch.from_numpy(obs_sl)
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    rngv = np.random.default_rng(1)
    kv = rngv.choice(len(sec_sel), min(8, len(sec_sel)), replace=False)
    sxs = xs[sec_sel]
    Zl, _ = embed_everything(codec, slab_t, obs_t, ctx_all, lats[rows3], lons,
                             np.ones(len(kv), dtype=int), sxs[kv], d_z,
                             cache_path=None, batch=64)
    worst = 0.0
    for tt in (0, T // 2, T - 1):
        dmax = float(np.abs(Zl[tt] - Zsec[tt][kv]).max())
        zscale = float(np.abs(Zl[tt]).max())
        worst = max(worst, dmax)
        assert dmax < max(0.02, 0.005 * zscale), \
            f"Z mismatch at t={tt}: {dmax} vs scale {zscale}"
    print(f"Z cache verified vs live re-encode ✓ (max|Δ| {worst:.4f} over "
          f"{len(kv)} section px at t=0, T/2, T-1)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="the RAW tensor memmap")
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--z", required=True, help="Z cache .npy (f16 [T,P,dz])")
    ap.add_argument("--ckpt", required=True, help="the codec checkpoint")
    ap.add_argument("--head", required=True, help="the stage-2 head")
    ap.add_argument("--backend", choices=("torch", "jax"), default="torch")
    ap.add_argument("--scope", choices=("gate", "window"), default="gate",
                    help="which pixels to ROLL. 'gate' rolls the gate scope's "
                         "own pixels only and scores the gate scope exactly "
                         "(stencil 1 has no cross-pixel term); 'window' rolls "
                         "all of them and scores all three scopes — see the "
                         "module docstring for what that costs")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--pixels-gate", type=int, default=600)
    ap.add_argument("--corridor-pctl", type=float, default=75.0)
    ap.add_argument("--corridor-dilate", type=int, default=2)
    ap.add_argument("--cache-dir", default="/tmp/jaxport_roll_cache")
    ap.add_argument("--out", help="write the result json here")
    ap.add_argument("--zhat-out",
                    help="stack every rolled ẑ and save it — the array the "
                         "two backends are compared on elementwise")
    ap.add_argument("--compare-zhat",
                    help="a --zhat-out from the other backend; prints the max "
                         "elementwise |Δ| between the two rolls")
    ap.add_argument("--no-verify-z", action="store_true",
                    help="skip the live re-encode guard (it needs the codec "
                         "and a slab read; ~1 min)")
    a = ap.parse_args()
    os.makedirs(a.cache_dir, exist_ok=True)
    t_run = time.time()

    d = np.load(a.npz_small, allow_pickle=False)
    ax = TimeAxis(d)
    months, lats, lons, T, moy = ax.labels, d["lats"], d["lons"], ax.T, ax.moy
    print(ax.describe(), flush=True)
    gate_ref, gate_skip = gate_for_cadence(ax.cadence)
    if gate_ref is None:
        sys.exit(f"G3 is a MONTHLY reference and this axis is {ax.cadence}: "
                 f"{gate_skip}")
    if GATE_HEAD not in os.path.basename(a.head):
        sys.exit(f"--head {os.path.basename(a.head)} is not {GATE_HEAD}: this "
                 f"driver scores G3, and G3 names one head")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    C, d_z = len(ck["chan"]), ck["d_z"]
    hold_years = sorted(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in set(hold_years) for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    Xm = np.load(a.x, mmap_mode="r")
    Hg, Wg = Xm.shape[1], Xm.shape[2]
    om_path = os.path.join(a.cache_dir, "ocean_mask.npy")
    if not os.path.exists(om_path) and os.path.exists(a.x + ".ocean.npy"):
        om_path = a.x + ".ocean.npy"               # the sidecar the box has
    if os.path.exists(om_path):
        ocean = np.load(om_path)
    else:
        ocean = np.zeros((Hg, Wg), bool)
        for t0 in range(0, T, 16):
            ocean |= np.isfinite(np.asarray(Xm[t0:t0 + 16, :, :, 0])).any(0)
        np.save(om_path, ocean)
    ys, xs = np.where(ocean)
    P = len(ys)
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)
    print(f"window: {P} ocean px · section {len(sec_sel)} px at row {sec_y}",
          flush=True)

    Zm = np.load(a.z, mmap_mode="r")
    assert Zm.shape == (T, P, d_z), \
        f"Z {Zm.shape} != {(T, P, d_z)} — ordering mismatch, refusing"

    st_path = os.path.join(a.cache_dir, "std_stats.npz")
    if os.path.exists(st_path):
        s_ = np.load(st_path)
        clim, dyn = s_["clim"], list(s_["dyn"])
        mean_c, std_c = s_["mean_c"], s_["std_c"]
    else:
        t0 = time.time()
        clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold)
        np.savez(st_path, clim=clim, dyn=np.array(dyn),
                 mean_c=mean_c, std_c=std_c)
        print(f"stream_stats: {time.time() - t0:.0f}s → {st_path}", flush=True)

    if not a.no_verify_z:
        verify_z(ck, C, d_z, Xm, Zm, moy, clim, dyn, mean_c, std_c, lats,
                 lons, ys, xs, sec_y, sec_sel, T)

    # ---- the scopes, over the FULL window, exactly as rollout_spatial ------
    rng = np.random.default_rng(0)                 # rollout.py's exact subset
    keep = np.union1d(rng.choice(P, min(a.pixels_gate, P), replace=False),
                      sec_sel)
    gate_mask = np.zeros(P, bool)
    gate_mask[keep] = True
    if a.scope == "window":
        corridor, cor_thr = corridor_pixels(Xm, ocean, ys, xs, t_hold,
                                            sec_sel, a.corridor_pctl,
                                            a.corridor_dilate)
        scopes = (("gate", gate_mask), ("corridor", corridor),
                  ("window", np.ones(P, bool)))
        roll_sel = np.arange(P)
        not_scored = {}
    else:
        cor_thr = None
        scopes = (("gate", gate_mask),)
        roll_sel = np.where(gate_mask)[0]
        not_scored = {
            "corridor": PREREG["corridor"], "window": PREREG["window"],
            "why": ("NOT SCORED: this run rolled only the gate scope's "
                    f"{len(roll_sel)} pixels. The corridor (29,627 px) and "
                    "window (84,405 px) scopes are aggregates over pixels "
                    "this run never rolled, and an AUC over the subset of "
                    "them that happens to fall in the gate's 600-px draw is "
                    "a different quantity — so it is absent rather than "
                    "printed under a name it does not have. Re-run with "
                    "--scope window on hardware that can afford it."),
        }
    n_roll = len(roll_sel)
    print(f"scopes: {' · '.join(f'{n} {int(m.sum())} px' for n, m in scopes)}"
          f" · ROLLING {n_roll:,} of {P:,} px (--scope {a.scope})", flush=True)

    # ---- per-pixel numpy plumbing, restricted to the rolled pixels --------
    # StdMonths/ar1_train are per-pixel by construction, so restricting ys/xs
    # gives values identical to the full-P arrays indexed by roll_sel. The
    # tensor reads are per-TIMESTEP either way; what shrinks is the memory.
    std_m = StdMonths(Xm, ys[roll_sel], xs[roll_sel], moy, clim, dyn,
                      mean_c, std_c)
    t0 = time.time()
    print("AR1 damped-persistence pass over the record ...", flush=True)
    r1 = ar1_train(std_m, T, t_hold, n_roll, C)
    print(f"  AR1 done in {time.time() - t0:.0f}s", flush=True)

    # ONLY THE ROLLED PIXELS. Zstat is one encoder forward per pixel through
    # the 40.7M patch-3 codec — 84,405 of them cost ~47 min on this box — and
    # the 83,541 that are never rolled contribute to no sum. The cache file is
    # named by the pixel COUNT so a gate-scope cache can never be silently
    # picked up by a window-scope run.
    Zstat = static_identity(ck, C, Xm, moy, clim, dyn, mean_c, std_c,
                            ys[roll_sel], xs[roll_sel], coords[roll_sel],
                            os.path.join(a.cache_dir,
                                         f"zstat_{n_roll}.npy"))

    # ---- the head, and its geometry --------------------------------------
    hk = torch.load(a.head, map_location="cpu", weights_only=False)
    ta = hk["args"]
    K = ta["K"]
    stencil = ta.get("stencil", 1)
    ring_km = ta.get("ring_km", 0) or 0
    k_max = hk["model"]["pos.weight"].shape[0]     # the file, not a convention
    if stencil == 1:
        NBR = None
        sctx = np.concatenate([Zstat, coords[roll_sel]], 1)
    else:
        if a.scope != "window":
            sys.exit(f"--scope {a.scope} with stencil {stencil}: a stencil "
                     f">1 head reads its NEIGHBOURS' z, so a rolled subset is "
                     f"NOT the full roll restricted — the missing neighbours "
                     f"would zero-fill and the arithmetic would differ. "
                     f"Refusing rather than scoring a different experiment.")
        NBR = build_stencil(Hg, Wg, ys, xs, stencil, ring_km=ring_km,
                            lats=lats)
        # roll_sel is every pixel here (the refusal above guarantees it), so
        # NBR's indices address the rolled array directly.
        sctx = np.concatenate([Zstat, coords,
                               (NBR >= 0).astype(np.float32)], 1)
    sctx = np.ascontiguousarray(sctx.astype(np.float32))
    label = (f"s{stencil}"
             + (f"r{str(ring_km).replace(',', '-')}" if _ring_on(ring_km)
                else "")
             + (f"u{ta.get('unroll', 1)}" if ta.get("unroll", 1) != 1 else "")
             + f"_s{ta.get('seed', 0)}")
    print(f"head {label}: {os.path.basename(a.head)} "
          f"(d_model={ta['d_model']}, layers={ta['layers']}, K={K}, "
          f"stencil={stencil}, k_max={k_max})", flush=True)

    Backend = {"torch": TorchBackend, "jax": JaxBackend}[a.backend]
    be = Backend(ck, hk, C, d_z, stencil, k_max, ring_km)
    print(f"backend: {be.name}", flush=True)

    def zwin_from_true(s_end):
        arr = np.asarray(Zm[s_end - K + 1: s_end + 1])[:, roll_sel]
        # float16 STORAGE → float32 ARITHMETIC, at this boundary and nowhere
        # else. Both backends widen here, from the same bytes.
        return np.ascontiguousarray(arr.transpose(1, 0, 2)).astype(np.float32)

    # ---- the roll ---------------------------------------------------------
    Hh = a.horizon
    sums = {name: new_sums(Hh) for name, _ in scopes}
    n_planned = 0
    for Y in hold_years:
        for s_ in ax.starts_for_year(Y, 0):
            if s_ - K + 1 < 0 or s_ + 1 >= T:
                continue
            for h in range(1, Hh + 1):
                if s_ + h >= T or ax.year[s_ + h] != int(Y):
                    break
                n_planned += 1
    print(f"  {label}: {n_planned} scored roll steps over {n_roll:,} pixels",
          flush=True)
    zhats = []
    done = 0
    t_roll = time.time()
    for Y in hold_years:
        for s in ax.starts_for_year(Y, 0):
            if s - K + 1 < 0 or s + 1 >= T:
                continue
            Zwin = be.window(zwin_from_true(s))
            cur = list(moy[s - K + 1: s + 1])
            v_pers, obs_s = std_m.get(s)
            for h in range(1, Hh + 1):
                t_tgt = s + h
                if t_tgt >= T or ax.year[t_tgt] != int(Y):
                    break
                # month features are the CURRENT window's months; the window
                # advances AFTER the forward — rollout_spatial's order, and
                # `month_feats` is the ORIGINAL function, so both backends
                # read their month features out of one implementation.
                mf = np.asarray(month_feats(cur, torch.device("cpu")))
                zhat = be.roll(Zwin, NBR, sctx, mf, a.chunk)
                Zwin = be.slide(Zwin, zhat)
                cur = cur[1:] + [ax.moy_of_row(t_tgt)]
                xhat = be.decode(zhat, C, a.chunk)
                v_true, obs_tt = std_m.get(t_tgt)
                op = obs_tt & obs_s
                v_damp = v_pers * r1 ** h
                for name, m_ in scopes:
                    mm = m_[roll_sel]
                    accumulate(sums[name], h, xhat[mm], v_true[mm],
                               v_pers[mm], v_damp[mm], op[mm])
                zhats.append(be.to_numpy(zhat))
                done += 1
                if done % 20 == 0 or done == n_planned:
                    el = time.time() - t_roll
                    print(f"  {done}/{n_planned} steps · {el / 60:.1f} min "
                          f"elapsed · ~{el / done * (n_planned - done) / 60:.1f}"
                          f" min left", flush=True)
    zhats = np.stack(zhats, 0)                     # [steps, n_roll, d_z]

    # ---- the read-out -----------------------------------------------------
    res = {"backend": be.name, "head": os.path.basename(a.head),
           "label": label, "data": os.path.basename(a.x),
           "z": os.path.basename(a.z), "horizon": Hh,
           "hold_years": hold_years, "K": K, "stencil": stencil,
           "scope_rolled": a.scope, "n_rolled_px": int(n_roll),
           "n_window_px": int(P), "roll_steps": int(n_planned),
           "prereg": PREREG, "seconds": round(time.time() - t_run, 1),
           "scopes": {}}
    if cor_thr is not None:
        res["corridor_def"] = {"pctl": a.corridor_pctl,
                               "threshold": round(cor_thr, 4),
                               "dilate_cells": a.corridor_dilate}
    if not_scored:
        res["not_scored"] = not_scored
    print()
    print(f"=== {be.name} · {label} · {a.scope} scope ===")
    for name, m_ in scopes:
        blk = skill_block(sums[name], Hh, n_px=int(m_.sum()))
        res["scopes"][name] = blk
        auc = blk.get("horizon_auc")
        ref = PREREG.get(name)
        mark = ""
        if auc is not None and ref is not None:
            dv = auc - ref
            mark = (f"   archive {ref:.3f}   Δ {dv:+.4f}   "
                    f"{'PASS' if abs(dv) <= PREREG['tol'] else 'FAIL'} "
                    f"(tol {PREREG['tol']})")
        print(f"  {name:<10} horizon_auc {auc}   "
              f"auc_damped {blk.get('auc_damped')}   "
              f"n_px {blk.get('n_px')}{mark}")
    for name, ref in (("corridor", None), ("window", None)):
        if name in not_scored:
            print(f"  {name:<10} NOT SCORED (archive {not_scored[name]:.3f}) "
                  f"— see `not_scored.why` in the json")
    if a.zhat_out:
        np.save(a.zhat_out, zhats)
        print(f"rolled states → {a.zhat_out} {zhats.shape} float32")
    if a.compare_zhat:
        other = np.load(a.compare_zhat)
        if other.shape != zhats.shape:
            sys.exit(f"--compare-zhat shape {other.shape} != {zhats.shape}")
        dmax = float(np.abs(other - zhats).max())
        dmean = float(np.abs(other - zhats).mean())
        scale = float(np.abs(zhats).max())
        res["compare_zhat"] = {"file": os.path.basename(a.compare_zhat),
                               "max_abs_delta": dmax,
                               "mean_abs_delta": dmean, "z_scale": scale,
                               "n": int(zhats.size)}
        print(f"rolled states vs {os.path.basename(a.compare_zhat)}: "
              f"max|Δ| {dmax:.3e} · mean|Δ| {dmean:.3e} over "
              f"{zhats.size:,} values (|z| max {scale:.3f})")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
