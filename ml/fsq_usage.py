#!/usr/bin/env python3
"""EFFECTIVE BITS PER TOKEN, MEASURED FROM THE DIGITS.

    python3 ml/fsq_usage.py --ckpt run-455/pixelmae.pt --z Z.npy [--limit N]
                            [--json out.json]

WHY THIS SCRIPT EXISTS, IN ONE NUMBER. Run-455 is a HEALTHY FSQ codec: it
trained, its loss fell, its probe read a normal number, and its own log line
said `3.000 bits/dim x d_z 32 = 96.0 bits (codebook 2^96.0)`. That line is
arithmetic on the `--fsq-levels` argument. It is not a measurement of anything
the codec did. What the codec actually did was sit at a pre-quantization
|v| ~ 3e4 against a saturation radius R = 2 — saturated by a factor of 15,000,
so every dimension landed on one of its TWO outermost levels and an eight-level
ladder was worn as a ONE-BIT SIGN CODE. Ninety-six nominal bits, about
thirty-two used. Nothing in the run's metrics said so, because nothing in the
run's metrics was looking at the digits.

So: this reads the digits. It is the instrument E-049 requires beside every
FVU number (`ml/plans/E049_roadB_token.md` §3, §5), and the standing rule it
enforces is that **"how many bits does this token carry" is a measurement,
never an inference from the levels argument**.

---

HOW THE DIGITS ARE RECOVERED, AND WHY THAT IS EXACT RATHER THAN APPROXIMATE.

A stored `z` from a quantized codec is not "near" a lattice point — it IS one.
`InputQuant.__call__` returns `(round(g) + offset) * R / half` (uniform) or
`sign(v) * a * c^round(j)` (exp): in both, the value is one of exactly L
reconstruction points per dimension, and `ml/fsq_ladder.py:levels_of` is the
function that says which L. So the digit is recovered by NEAREST-POINT
matching against the rebuilt lattice, and the match is not a heuristic — the
distance to the assigned point is zero up to float32 round-off.

That is a claim about the input, so it is CHECKED rather than assumed
(ml/CLAUDE.md §0.1: verify the artefact). The report prints the worst
`|z - nearest point| / R_d` it saw, and the script REFUSES above
`sqrt(eps_float32)` = 3.45e-4. The threshold is not a taste: it is the standard
"half the significant digits" tolerance, it sits ~2,900x above the float32
round-off these lattices are stored at, and ~700x BELOW the ~0.25 R that a
CONTINUOUS (unquantized) z would produce, since a continuous value falls
uniformly between its neighbouring points. There is no regime in between, so
the refusal separates "this Z came from this codec" from "this Z is not on
this lattice at all" with three orders of magnitude of clearance on each side.

THE LATTICE IS REBUILT FROM THE CHECKPOINT AND NEVER RE-FITTED. `fsq_levels`,
`fsq_ladder`, `fsq_ladder_fit` (the per-dimension ladder AND radius an `auto`
run measured) and `fsq_bound` are read out of `ck["args"]` exactly as
`ml/model.py:codec_from_ckpt` reads them, through the same
`ml/fsq_ladder.py:resolve`. A checkpoint with no `fsq_levels` is refused: a
continuous bottleneck has no digits, and answering with a histogram of
something else would be worse than answering with nothing.

---

WHAT IT REPORTS, AND HOW TO READ IT.

Per dimension d:
  * the LEVEL-OCCUPANCY HISTOGRAM — how often each of the L digits was used;
  * its ENTROPY in bits, `H_d = -sum_k p_k log2 p_k`, against the nominal
    `log2 L_d`. H_d is what dimension d actually carries, on this sample;
  * `outer` — the fraction of mass on the TWO OUTERMOST levels. This is the
    SIGN-CODE SIGNATURE and it is reported separately from the entropy because
    it names the disease rather than only its size: a saturated dimension puts
    ~1.0 here and reads H_d ~ 1 bit whatever L is.

And for the token:
  * EFFECTIVE bits = sum_d H_d, beside NOMINAL bits = sum_d log2 L_d.

**The sum of per-dimension entropies is an UPPER BOUND on the token's true
entropy**, because dimensions may be correlated and H(X,Y) <= H(X) + H(Y).
This is stated rather than hidden: the number is generous by construction, so
an effective-bits figure far below nominal is unambiguous (the codec really is
not using its alphabet), while one close to nominal bounds the answer from
above and does not prove the joint code is full. Measuring the joint entropy of
a 2^16 alphabet needs far more samples than any probe draw we take; the
marginal sum is the honest, computable statement.

Exact and deterministic: no RNG anywhere. `--limit` subsamples by an EVENLY
SPACED index walk over the whole array (`np.linspace`), not a random draw, so
two runs of the same command return the same digits and a limit never
concentrates on one region of a [T, P, d_z] embedding's time axis.

torch is imported ONLY to read the checkpoint (`torch.load`, map_location
cpu); every digit, histogram and entropy is numpy, and `ml/fsq_ladder.py` —
the module both backends already share — is the single source of the lattice.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fsq_ladder as fql                                        # noqa: E402

# The digit-recovery tolerance, as a fraction of the dimension's own
# saturation radius. sqrt(eps) is the classical "half the digits survive"
# tolerance; see the module docstring for the two margins that make it a
# separation rather than a pick.
LATTICE_TOL = float(np.sqrt(np.finfo(np.float32).eps))          # 3.4527e-04


def read_ckpt(path):
    """`(d_z, levels, is_exp, base, scale, meta)` — the lattice, rebuilt.

    Reads the same four architecture fields `ml/model.py:codec_from_ckpt`
    reads, with the same defaults and the same refusals, so a Z that this
    script says is on the lattice is on the lattice the loaders build.
    """
    import torch                                    # checkpoint I/O only
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    a = a if isinstance(a, dict) else vars(a)
    d_z = int(a.get("d_z") or ck.get("d_z") or 0)
    if not d_z:
        raise SystemExit(
            f"{path}: no d_z in the checkpoint's args or at its top level. "
            f"This is not a codec checkpoint this repo wrote.")
    spec = str(a.get("fsq_levels", "") or "")
    if not spec:
        raise SystemExit(
            f"{path}: `fsq_levels` is empty — this codec's bottleneck is "
            f"CONTINUOUS, so its z has no digits and no per-dimension "
            f"alphabet. There is nothing here to measure the occupancy of; "
            f"refusing rather than reporting a histogram of a lattice that "
            f"was never applied. (The continuous arm's number is its "
            f"reconstruction FVU — ml/recon_eval.py.)")
    levels = fql.parse_levels(spec, d_z)
    ladder = str(a.get("fsq_ladder", "uniform") or "uniform")
    fit = str(a.get("fsq_ladder_fit", "") or "")
    base0 = float(a.get("fsq_exp_base", fql.DEFAULT_EXP_BASE)
                  or fql.DEFAULT_EXP_BASE)
    if ladder == "auto" and not fit:
        raise SystemExit(
            f"{path}: --fsq-ladder auto with no recorded `fsq_ladder_fit`. "
            f"The per-dimension lattice this codec trained with is not in the "
            f"file, and re-fitting here would measure THIS sample rather than "
            f"the one that trained — refusing, exactly as "
            f"ml/model.py:codec_from_ckpt does.")
    # sigma = 1 on the codec side, so the constructor's radius is 2*sigma = 2
    # (ml/model.py:fsq_from_levels). A legacy fit string with no `:<R>` field
    # resolves to exactly that, which is what those checkpoints trained with.
    is_exp, base, scale, _fitted = fql.resolve(
        ladder, levels, d_z, base0, fit, flag="--fsq-ladder", scale=2.0)
    meta = {"ckpt": os.path.abspath(path), "d_z": d_z,
            "fsq_levels": spec, "fsq_ladder": ladder,
            "fsq_ladder_fit": fit or None,
            "fsq_exp_base": base0,
            "fsq_bound": str(a.get("fsq_bound", "") or "") or None,
            "step": int(ck.get("step", 0) or 0),
            "params_d_model": a.get("d_model"), "patch": a.get("patch"),
            "k_time": a.get("k_time")}
    return d_z, levels, is_exp, base, scale, meta


def load_z(path, d_z, limit=0):
    """`[N, d_z] float64`, from a .npy of [N, d_z] or [T, P, d_z].

    Memory-mapped, so a multi-GB embedding costs the rows this actually reads.
    `limit` walks an EVENLY SPACED index set over the whole array rather than
    taking a prefix or a random draw: a prefix of a [T, P, d_z] embedding is
    the first few months, and the question "which digits does this codec use"
    must not be answered from one season.
    """
    z = np.load(path, mmap_mode="r")
    if z.ndim < 2 or z.shape[-1] != d_z:
        raise SystemExit(
            f"{path}: shape {tuple(z.shape)} — the last axis must be the "
            f"checkpoint's d_z = {d_z}. Pass the codec's own embeddings "
            f"([N, d_z] or [T, P, d_z]).")
    n_tot = int(np.prod(z.shape[:-1]))
    z = z.reshape(n_tot, d_z)
    if limit and limit < n_tot:
        idx = np.unique(np.linspace(0, n_tot - 1, int(limit)).astype(np.int64))
        v = np.asarray(z[idx], np.float64)
    else:
        idx = None
        v = np.asarray(z, np.float64)
    keep = np.isfinite(v).all(1)
    n_drop = int((~keep).sum())
    if not keep.any():
        raise SystemExit(f"{path}: every one of {len(v):,} vectors holds a "
                         f"non-finite value — there is nothing to measure.")
    return v[keep], n_tot, n_drop


def digits_of(v, levels, is_exp, base, scale):
    """`(digits [N, d_z] int, points, worst residual / R per dim)`.

    Nearest-point matching against the rebuilt lattice, per dimension, by
    `searchsorted` on the midpoints — the same decision boundary the
    quantizer's own `round` implements, so a value that came off this lattice
    lands on the digit it was written as.
    """
    n, d_z = v.shape
    dig = np.empty((n, d_z), np.int64)
    pts, resid = [], np.empty(d_z, np.float64)
    for d in range(d_z):
        p = fql.levels_of(int(levels[d]), float(scale[d]),
                          "exp" if bool(is_exp[d]) else "uniform",
                          float(base[d]))
        mid = 0.5 * (p[1:] + p[:-1])
        k = np.searchsorted(mid, v[:, d])
        dig[:, d] = k
        resid[d] = float(np.abs(v[:, d] - p[k]).max()) / max(
            float(scale[d]), 1e-30)
        pts.append(p)
    return dig, pts, resid


def entropy_bits(counts):
    """`-sum p log2 p` over a count vector, in bits. Empty levels contribute
    exactly 0 (0 log 0 = 0), which is the whole point: a dimension that never
    reaches half its alphabet reads as the bits it used, not the bits it has."""
    c = np.asarray(counts, np.float64)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c[c > 0] / tot
    return float(-(p * np.log2(p)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Effective bits per FSQ token, measured from the digits.")
    ap.add_argument("--ckpt", required=True,
                    help="a codec checkpoint (.pt). Its args supply the "
                         "lattice: fsq_levels, fsq_ladder, fsq_ladder_fit, "
                         "fsq_bound, d_z. Nothing is re-fitted.")
    ap.add_argument("--z", required=True,
                    help="a .npy of embeddings, [N, d_z] or [T, P, d_z]; "
                         "memory-mapped, so a multi-GB file is fine.")
    ap.add_argument("--limit", type=int, default=0,
                    help="measure at most this many vectors, taken at even "
                         "spacing across the whole array (deterministic). "
                         "0 = all of them.")
    ap.add_argument("--json", default="",
                    help="also write the full result — per-dimension "
                         "histograms included — to this path.")
    a = ap.parse_args(argv)

    d_z, levels, is_exp, base, scale, meta = read_ckpt(a.ckpt)
    v, n_tot, n_drop = load_z(a.z, d_z, a.limit)
    dig, pts, resid = digits_of(v, levels, is_exp, base, scale)

    worst = float(resid.max())
    if worst > LATTICE_TOL:
        raise SystemExit(
            f"REFUSING: these vectors are not on this checkpoint's lattice. "
            f"The worst |z - nearest level| is {worst:.4g} of the dimension's "
            f"saturation radius (dimension {int(np.argmax(resid))}), against "
            f"a tolerance of {LATTICE_TOL:.4g} = sqrt(eps_float32).\n"
            f"  A quantized z IS a lattice point, so a genuine one reads ~1e-7 "
            f"here; a CONTINUOUS z reads ~0.25, because it falls uniformly "
            f"between neighbouring points. {worst:.4g} says one of three "
            f"things, and every one of them makes a digit histogram "
            f"meaningless:\n"
            f"    · this Z was embedded by a DIFFERENT codec than --ckpt;\n"
            f"    · it was embedded by a loader that DROPPED the quantizer "
            f"(the failure ml/model.py:codec_from_ckpt exists to prevent);\n"
            f"    · this checkpoint's recorded ladder/radius is not the one "
            f"that produced it.\n"
            f"  Re-embed with ml/model.py:codec_from_ckpt against this exact "
            f"checkpoint, then measure.")

    nominal = np.log2(levels.astype(np.float64))
    rows = []
    for d in range(d_z):
        L = int(levels[d])
        cnt = np.bincount(dig[:, d], minlength=L).astype(np.int64)
        H = entropy_bits(cnt)
        outer = float(cnt[0] + cnt[L - 1]) / float(cnt.sum())
        rows.append({"dim": d, "L": L, "ladder": "exp" if is_exp[d] else "u",
                     "base": (float(base[d]) if is_exp[d] else None),
                     "R": float(scale[d]),
                     "nominal_bits": float(nominal[d]), "bits": H,
                     "used_levels": int((cnt > 0).sum()),
                     "outer_mass": outer,
                     "hist": [int(c) for c in cnt]})

    eff = float(sum(r["bits"] for r in rows))
    nom = float(nominal.sum())
    out = {"meta": meta, "z": os.path.abspath(a.z), "n_vectors": int(len(v)),
           "n_available": int(n_tot), "n_nonfinite_dropped": int(n_drop),
           "effective_bits": eff, "nominal_bits": nom,
           "effective_fraction": eff / nom if nom > 0 else 0.0,
           "worst_lattice_residual_frac_R": worst,
           "lattice_tol": LATTICE_TOL,
           "outer_mass_mean": float(np.mean([r["outer_mass"] for r in rows])),
           "dims": rows}

    # ---- one stdout block ------------------------------------------------
    print(f"FSQ USAGE · {os.path.basename(a.ckpt)} (step {meta['step']:,}) "
          f"· z {os.path.basename(a.z)}")
    print(f"  lattice: --fsq-levels {meta['fsq_levels']} · --fsq-ladder "
          f"{meta['fsq_ladder']}"
          + (f" · fit {meta['fsq_ladder_fit']}" if meta['fsq_ladder_fit']
             else "")
          + (f" · --fsq-bound {meta['fsq_bound']}" if meta['fsq_bound']
             else " · no intrinsic bound"))
    print(f"  measured on {len(v):,} vectors of {n_tot:,}"
          + (f" (evenly spaced, --limit {a.limit})" if a.limit else "")
          + (f"; {n_drop:,} non-finite dropped" if n_drop else "")
          + f" · worst |z - level| {worst:.3g} R (tol {LATTICE_TOL:.3g})")
    print(f"  {'dim':>4} {'L':>3} {'lad':>5} {'R':>9} {'used':>5} "
          f"{'bits':>7} {'of':>6} {'outer':>7}  occupancy")
    for r in rows:
        h = r["hist"]
        tot = max(sum(h), 1)
        bar = " ".join(f"{100.0 * c / tot:.0f}" for c in h)
        lad = r["ladder"] + ("%g" % r["base"] if r["base"] else "")
        print(f"  {r['dim']:>4} {r['L']:>3} {lad:>5} "
              f"{r['R']:>9.4g} {r['used_levels']:>5} {r['bits']:>7.3f} "
              f"{r['nominal_bits']:>6.3f} {r['outer_mass']:>7.3f}  {bar}")
    print(f"  EFFECTIVE {eff:.3f} bits/token of a NOMINAL {nom:.3f} "
          f"({100.0 * eff / nom:.1f}%) · mean outer-level mass "
          f"{out['outer_mass_mean']:.3f}")
    print(f"  (the sum of per-dimension entropies is an UPPER BOUND on the "
          f"token's entropy — dimensions may be correlated. 'outer' is the "
          f"mass on the two outermost levels: ~1.0 is the run-455 sign-code "
          f"signature, where {nom:.0f} nominal bits were worn as ~{d_z} — "
          f"one per dimension.)")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  wrote {a.json}")
    return out


if __name__ == "__main__":
    main()
