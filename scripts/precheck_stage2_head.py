#!/usr/bin/env python3
"""Does this stage-2 checkpoint support the mode we are about to ask for?

Answer it in the first ten seconds of the job, not after the embedding.

Run #119 spent **93 minutes** rebuilding the frozen-codec embedding and then
exited one second later because `f3_s2_60k__temporal.pt` carries no optimiser
state and `--resume-temporal` — correctly — refuses to call a warm restart a
continuation. The refusal was right. Its POSITION was wrong: the checkpoint's
keys are readable the moment the file is on disk, so the job burned an hour
and a half of a rented 4090 to discover something knowable before it started.

That is a general shape worth naming: a guard placed at the point of USE
rather than the point of DISPATCH is correct and expensive. Wherever a
precondition depends only on inputs, check it while the inputs are all it has
cost you.

Usage (from ml-train.yml, before the embedding):
  python3 scripts/precheck_stage2_head.py <tag-or-path> --mode resume|warm

Exit 0 = the mode is supported. Non-zero = stop now, with the reason and the
flag that would have worked.
"""
import argparse
import os
import sys

CKPT_DIR = os.environ.get("CKPT_DIR_OVERRIDE", "/opt/earth-cache/ckpt")
NEEDED_FOR_RESUME = ("opt", "sched", "step")


def describe(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    keys = sorted(ck.keys())
    missing = [k for k in NEEDED_FOR_RESUME if k not in ck]
    args = ck.get("args", {}) or {}
    return keys, missing, args, ck.get("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--mode", choices=("resume", "warm"), required=True)
    a = ap.parse_args()

    path = a.tag if os.path.sep in a.tag else os.path.join(CKPT_DIR, a.tag + ".pt")
    if not os.path.exists(path):
        sys.exit(f"stage-2 head {path} is absent, so temporal.py would refuse "
                 f"after the embedding. Stopping now instead. Check the tag, "
                 f"and that the model-checkpoints-v1 release still has it.")

    keys, missing, args, step = describe(path)
    print(f"stage-2 head {os.path.basename(path)}: keys {keys}")
    print(f"  parent: {args.get('steps')} steps at peak lr {args.get('lr')}"
          f"{f', saved at step {step}' if step is not None else ''}")
    print(f"  continuable: {'yes' if not missing else 'NO (missing ' + ', '.join(missing) + ')'}")

    if a.mode == "resume" and missing:
        sys.exit(
            f"\n--resume-temporal needs {', '.join(missing)} and this "
            f"checkpoint has none of it. Loading the weights alone resets "
            f"Adam's moments and the LR schedule, which is a warm restart "
            f"wearing a continuation's name — temporal.py refuses, and it is "
            f"right to.\n\n"
            f"Every head published before 2026-08-10 is {{args, model}}. For "
            f"an explicit warm restart, dispatch with\n"
            f"    window: warm2:{a.tag}@<peak-lr>\n"
            f"where temporal_steps is the EXTRA steps, not the total.")

    if a.mode == "warm" and not missing:
        # Not fatal — a warm restart from a continuable checkpoint is a
        # legitimate choice — but it is almost always a mistake to throw away
        # optimiser state you actually have, so say so loudly.
        print("::warning::this head IS continuable (it carries "
              + ", ".join(NEEDED_FOR_RESUME) + "), so resume2: would give a "
              "true continuation. warm2: deliberately discards Adam's moments "
              "and the schedule position — make sure that is what you want.")

    print(f"precheck OK: {a.mode} is supported by this checkpoint")


if __name__ == "__main__":
    main()
