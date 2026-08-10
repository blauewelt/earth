#!/usr/bin/env python3
"""Is the gap between two probes real? Compare them PAIRED, not by their CIs.

The question this exists for: the unpooled attention head scores r = 0.662
[0.557, 0.745] and the raw-3x3 control scores 0.628 [0.514, 0.729]. Those
intervals overlap over almost their whole length, and the reflex is to call
the +0.034 gap unsupported. That reflex is wrong, and so is the opposite
reflex of quoting +0.034 as if it were established.

Both are wrong for the same reason: the two intervals describe how much each
probe's score would move if the RECORD were resampled — and the record moving
would move both probes together, because they are scored on the same 240
months, the same year-blocked folds, and largely the same errors. The
sampling variation they display is mostly shared, and shared variation
cancels in a difference. The quantity that answers the question is the
bootstrap distribution of

    r(head) - r(control)

computed by resampling YEARS and rescoring BOTH probes on the same resampled
years every time. If that interval excludes zero, the head is better on this
record even though its CI overlaps the control's; if it straddles zero, the
gap is not distinguishable from fold noise no matter how far apart the point
estimates look.

Blocks are YEARS because AMOC transport is autocorrelated over months
(r_lowpass18 = 0.82) — an i.i.d. month bootstrap would treat ~240 months as
~240 independent observations and report an interval several times too tight.

Usage:
  python3 scripts/paired_probe.py ml/runs/<run>/probe_head.json \\
                                  ml/runs/<run>/probe_head_raw3x3.json

Both files must carry `pred`, `target_sv` and `years` — probe_head.py has
written those since 2026-08-10. Older result files do not have them and
cannot be compared this way without a rerun; the script says so rather than
guessing.
"""
import argparse
import json
import sys

import numpy as np


def load(path):
    j = json.load(open(path))
    missing = [k for k in ("pred", "target_sv", "years") if k not in j]
    if missing:
        sys.exit(f"{path}: no {'/'.join(missing)} — this file predates the "
                 f"per-month dump, so the paired comparison is impossible "
                 f"without rerunning probe_head.py")
    return (j,
            np.asarray(j["pred"], float),
            np.asarray(j["target_sv"], float),
            np.asarray(j["years"], int))


def r_of(pred, targ, sel):
    p, t = pred[sel], targ[sel]
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 3 or np.std(t[ok]) == 0 or np.std(p[ok]) == 0:
        return np.nan
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="the probe under test (e.g. probe_head.json)")
    ap.add_argument("b", help="the control (e.g. probe_head_raw3x3.json)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ja, pa, ta, ya = load(args.a)
    jb, pb, tb, yb = load(args.b)

    # The pairing is the whole argument — if the two probes were not scored on
    # the same months, differencing them is not a paired test and the interval
    # below would be meaningless.
    if len(pa) != len(pb) or not np.array_equal(ya, yb):
        sys.exit("the two probes were not scored on the same months — a "
                 "paired comparison needs identical folds and identical rows")
    if not np.allclose(ta, tb, equal_nan=True):
        sys.exit("the two probes were scored against different targets")

    ra, rb = r_of(pa, ta, slice(None)), r_of(pb, tb, slice(None))
    print(f"{args.a.split('/')[-1]:<32} r = {ra:+.3f}  ({ja.get('probe')})")
    print(f"{args.b.split('/')[-1]:<32} r = {rb:+.3f}  ({jb.get('probe')})")
    print(f"{'point difference':<32} Δ = {ra - rb:+.3f}")

    rng = np.random.default_rng(args.seed)
    uy = np.unique(ya)
    deltas = []
    for _ in range(args.n_boot):
        pick = rng.choice(uy, len(uy), replace=True)
        sel = np.concatenate([np.where(ya == y)[0] for y in pick])
        da = r_of(pa, ta, sel) - r_of(pb, tb, sel)
        if np.isfinite(da):
            deltas.append(da)
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # One-sided: how often does the control match or beat the probe under test?
    p_one = float((deltas <= 0).mean())

    print()
    print(f"paired block bootstrap over {len(uy)} years, {len(deltas)} resamples")
    print(f"  Δr 95% CI      [{lo:+.3f}, {hi:+.3f}]")
    print(f"  P(Δr <= 0)      {p_one:.3f}")
    print()
    if lo > 0:
        print(f"  → the {ja.get('probe')} beats the control on this record: "
              f"the paired interval excludes zero.")
    elif hi < 0:
        print(f"  → the control BEATS the probe under test: the paired "
              f"interval excludes zero on the other side.")
    else:
        print(f"  → not distinguishable. The point gap of {ra - rb:+.3f} is "
              f"inside the fold-to-fold noise, so it must not be quoted as "
              f"an improvement — with {len(uy)} year-blocks this record "
              f"cannot resolve it.")


if __name__ == "__main__":
    main()
