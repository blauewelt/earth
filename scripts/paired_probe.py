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

  python3 scripts/paired_probe.py ml/runs/probe_kfold.json#f3_anchor41M/rapid \\
                                  ml/runs/probe_kfold.json#f4_nolonhold/rapid

Both files must carry `pred`, `target_sv` and `years`. Two shapes carry them:

  probe_head.py  writes them FLAT at the top level (since 2026-08-10) — it
                 probes one target, so there is one set.
  probe_kfold.py writes them per RUN per TARGET (since 2026-08-19) — it
                 sweeps several codecs over rapid/fc/move at once, and those
                 targets do not share rows, so a flat array could not describe
                 them. Select one with a `#` fragment: `#<run>/<target>`, or
                 `#<run>/<target>/wind_only_baseline` for the wind bar, which
                 inherits `target_sv` and `years` from the target above it.
                 With one run in the file and `--target` at its default, the
                 fragment can be left off.

Older result files have none of it and cannot be compared this way without a
rerun; the script says so rather than guessing.
"""
import argparse
import json
import os
import sys

import numpy as np


WANT = ("pred", "target_sv", "years")


def split_spec(spec):
    """`ml/runs/probe_kfold.json#codecA/rapid` -> (path, ['codecA','rapid']).

    A path that exists as given wins over the fragment reading, so a file
    with a literal `#` in its name is still openable.
    """
    if os.path.exists(spec) or "#" not in spec:
        return spec, []
    path, _, frag = spec.partition("#")
    return path, [p for p in frag.split("/") if p]


def walk(path, j, parts, target):
    """Descend to the selected block and return (chain, label).

    The chain is every node from the root down, because the three arrays do
    not all have to live at the same depth: probe_kfold's wind-only baseline
    carries its own `pred` and shares the target block's `target_sv` and
    `years` — which is exactly right, since being scored on the same rows is
    what makes the comparison paired in the first place.
    """
    if not parts and "pred" not in j:
        runs = [k for k, v in j.items() if isinstance(v, dict)]
        if len(runs) > 1:
            sys.exit(f"{path}: holds {len(runs)} runs "
                     f"({', '.join(runs)}) — name one with "
                     f"'{os.path.basename(path)}#<run>/{target}'")
        # No sub-blocks at all is not an ambiguous file, it is an OLD file —
        # a probe_head.json from before the per-month dump. Fall through and
        # let the caller say that, rather than reporting "0 runs" at someone
        # who is holding a legacy result and needs to be told to rerun.
        if runs:
            parts = [runs[0], target]
    chain, node = [j], j
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            have = [k for k, v in (node or {}).items() if isinstance(v, dict)]
            sys.exit(f"{path}: nothing at '{'/'.join(parts[:i + 1])}' — "
                     f"this level offers {', '.join(have) or '(nothing)'}")
        node = node[part]
        chain.append(node)
    label = node.get("probe") if isinstance(node, dict) else None
    return chain, (label or "/".join(parts) or "probe")


def load(spec, target):
    path, parts = split_spec(spec)
    j = json.load(open(path))
    chain, label = walk(path, j, parts, target)
    vals = {}
    for k in WANT:
        for node in reversed(chain):
            if isinstance(node, dict) and node.get(k) is not None:
                vals[k] = node[k]
                break
    missing = [k for k in WANT if k not in vals]
    if missing:
        sys.exit(f"{spec}: no {'/'.join(missing)} — this file predates the "
                 f"per-month dump, so the paired comparison is impossible "
                 f"without rerunning probe_head.py / probe_kfold.py")
    return (label,
            np.asarray(vals["pred"], float),
            np.asarray(vals["target_sv"], float),
            np.asarray(vals["years"], int))


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
    ap.add_argument("--target", default="rapid",
                    help="which target block, for probe_kfold.py's multi-target "
                         "files; overridden per file by a '#<run>/<target>' "
                         "fragment. Ignored for probe_head.py's flat shape.")
    args = ap.parse_args()

    ja, pa, ta, ya = load(args.a, args.target)
    jb, pb, tb, yb = load(args.b, args.target)

    # The pairing is the whole argument — if the two probes were not scored on
    # the same months, differencing them is not a paired test and the interval
    # below would be meaningless.
    if len(pa) != len(pb) or not np.array_equal(ya, yb):
        sys.exit("the two probes were not scored on the same months — a "
                 "paired comparison needs identical folds and identical rows")
    if not np.allclose(ta, tb, equal_nan=True):
        sys.exit("the two probes were scored against different targets")

    ra, rb = r_of(pa, ta, slice(None)), r_of(pb, tb, slice(None))
    # the last TWO path segments, because a probe_kfold comparison is very
    # often `<runA>/probe_kfold.json` against `<runB>/probe_kfold.json` and
    # the basename alone prints the same string twice.
    na, nb = ("/".join(x.split("/")[-2:]) for x in (args.a, args.b))
    print(f"{na:<44} r = {ra:+.3f}  ({ja})")
    print(f"{nb:<44} r = {rb:+.3f}  ({jb})")
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
        print(f"  → the {ja} beats the control on this record: "
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
