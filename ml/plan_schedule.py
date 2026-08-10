#!/usr/bin/env python3
"""Compute a run's planned LR curve FROM THE TRAINER'S OWN SCHEDULER.

Chris, 2026-08-10: *"obviously, the queueing graph needs to be computed from
the actual implementation."*

He is right, and the version this replaces was a second implementation of the
schedule: the plan carried `{steps, lr}` and `status.html` re-derived the curve
with its own cosine formula. Two copies of one rule is two chances to
disagree — the same defect that made a run-keyed embedding cache poison
#10/#11, and it was about to bite here in a very specific way. Add `wsd` to
`temporal.py` and the preview would have gone on confidently drawing a cosine,
certifying a schedule the run does not use. A precertification that can be
wrong about the thing it certifies is worse than none.

So the curve is sampled from `make_sched` — the same function, with the same
arguments the dispatch passes — and the page plots the points rather than
recomputing them.

    python3 ml/plan_schedule.py --steps 200000 --lr 1e-3 --schedule wsd
    python3 ml/plan_schedule.py --steps 140000 --lr 1e-4 --warm \\
            --parent-steps 60000 --parent-lr 1e-3 --parent-schedule cosine

Prints the plan JSON on stdout, ready for scripts/dispatch_run.mjs --plan.
"""
import argparse
import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal import make_sched                             # noqa: E402

N_POINTS = 240          # enough that a 200k cosine is smooth on a phone


def curve(steps, lr, schedule, warmup, cooldown_frac, offset=0):
    """[[step, lr], ...] by RUNNING the real scheduler, not modelling it."""
    a = types.SimpleNamespace(steps=int(steps), lr=float(lr),
                              lr_schedule=schedule, lr_warmup=int(warmup),
                              lr_cooldown_frac=float(cooldown_frac))
    net = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr))
    sch = make_sched(opt, a)
    # Sample rather than store every step: 200,000 pairs is 4 MB of JSON for a
    # curve that is visually identical at 240. Always include the first and
    # last step, and every warmup step up to the peak — the warmup is short
    # and steep, and undersampling it would draw a schedule that appears to
    # start at its peak when it does not.
    want = sorted(set([0, int(steps) - 1]
                      + list(range(0, min(int(warmup) + 1, int(steps)),
                                   max(1, int(warmup) // 20 or 1)))
                      + [round(i * (int(steps) - 1) / (N_POINTS - 1))
                         for i in range(N_POINTS)]))
    pts, wi = [], 0
    for s in range(int(steps)):
        if wi < len(want) and s == want[wi]:
            pts.append([s + offset, round(sch.get_last_lr()[0], 12)])
            wi += 1
        opt.step()
        sch.step()
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, required=True,
                   help="for a warm restart this is the EXTRA steps, matching "
                        "what temporal_steps passes to the trainer")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--schedule", default="cosine",
                   choices=["cosine", "invsqrt", "wsd"])
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--cooldown-frac", type=float, default=0.1)
    p.add_argument("--warm", action="store_true",
                   help="a warm restart: this run starts at --parent-steps of "
                        "total compute, on a fresh schedule")
    p.add_argument("--parent-steps", type=int, default=0)
    p.add_argument("--parent-lr", type=float, default=0.0)
    p.add_argument("--parent-schedule", default="cosine",
                   choices=["cosine", "invsqrt", "wsd"])
    p.add_argument("--parent-run", type=int, default=0)
    p.add_argument("--note", default="")
    a = p.parse_args()

    if a.warm and not (a.parent_steps > 0 and a.parent_lr > 0):
        sys.exit("--warm needs --parent-steps and --parent-lr: without them "
                 "the chart has no parent segment and no seam, and the run "
                 "cannot be compared with the sibling it exists to be "
                 "compared against.")

    plan = {"steps": a.steps, "lr": a.lr, "schedule": a.schedule,
            "warmup": a.warmup,
            "points": curve(a.steps, a.lr, a.schedule, a.warmup,
                            a.cooldown_frac,
                            offset=a.parent_steps if a.warm else 0)}
    if a.schedule == "wsd":
        plan["cooldown_frac"] = a.cooldown_frac
    if a.warm:
        plan.update({"warm": True, "parent_steps": a.parent_steps,
                     "parent_lr": a.parent_lr,
                     "parent_points": curve(a.parent_steps, a.parent_lr,
                                            a.parent_schedule, a.warmup,
                                            a.cooldown_frac)})
    if a.parent_run:
        plan["parent_run"] = a.parent_run
    if a.note:
        plan["note"] = a.note
    json.dump(plan, sys.stdout)
    print()


if __name__ == "__main__":
    main()
