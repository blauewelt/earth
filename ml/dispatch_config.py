#!/usr/bin/env python3
"""One decoder for the dispatch `window` field, and a JSON path beside it.

WHY THIS EXISTS. `workflow_dispatch` caps a workflow at 25 inputs (raised from
10; GitHub has never published a rationale). `ml-train.yml` sits at exactly
25, and a 26th does not fail gracefully — the file becomes unparseable and
EVERY dispatch 422s. So knobs have been encoded into the `window` string
instead, and it now carries **ten distinct meanings**, decoded by shell
string-munging at eight separate sites in the workflow:

    global | na            the family-2 spatial window (its original job)
    lse | max | sum        joint loss mode, optionally  mode@ref  or  mode@twin
    unroll:N               autoregressive unroll depth for stage 2
    resume2:<tag>[@lr]     continue an existing stage-2 head
    warm2:<tag>[@lr]       warm restart from weights alone
    recw:k=v;k=v           reconstruction weights (may appear ANYWHERE in it)
    disktriage             disk investigation mode
    disktriage:rm <paths>  targeted removal, allowlisted roots
    rolleval:<tag,tag>     rolled evaluation over published heads
    dectrain… project:… sroll:…   route to their own scripts

Chris, 2026-08-17: *"it sounds like this should be a new setting and you
hacked around it"* — about a different change, but the observation lands
here. This IS the hack, and it predates the pentad work.

WHAT THIS FILE IS FOR, and what it deliberately is NOT yet. It is a **pure
decoder**: one function that turns the legacy string into a structured dict,
and a second that does the same for a JSON config. Nothing in the workflow
calls it yet. That is the point — the refactor is split so the risky half is
separable:

  phase 1 (this file)  characterise today's behaviour in one testable place
  phase 2 (later)      have the workflow call it once and export the results,
                       replacing the eight shell sites

Phase 1 is additive and cannot change any running job. Phase 2 edits the file
that takes the whole training system down when it is wrong, while seven runs
are live, so it wants its own session and its own care.

HOW "NOT CHANGING BEHAVIOUR" IS MADE CHECKABLE RATHER THAN PROMISED.
`tests/test_dispatch_config.py` enumerates every legacy form found at those
eight sites and asserts, for each one, that

    decode_window(s)  ==  decode_config(to_config(s))

i.e. the JSON path and the string path produce the identical structure. A
refactor that changes behaviour cannot pass that, because the legacy decode is
the reference and it is pinned by the same test.

The shell semantics are reproduced EXACTLY, including their sharp edges:
  · `recw:` is matched anywhere in the string (`case $W in *recw:*`), and its
    value runs to the first SPACE, not to the end.
  · the loss mode is everything before the first `@`; the reference is
    everything after it, defaulting to `0`.
  · `disktriage:rm ` requires the trailing space before its path list.
These are not improvements. Reproducing a quirk faithfully is the job; fixing
one here would be exactly the silent behaviour change this is meant to prevent.
"""
import json

SCRIPT_MODES = ("dectrain", "project:", "sroll:")


def decode_window(w):
    """The legacy string -> structured dict. Faithful to the shell, quirks included."""
    w = (w or "").strip()
    out = {"raw": w, "spatial": None, "loss_mode": None, "loss_ref": None,
           "recw": None, "mode": None, "mode_arg": None}

    # recw: matched ANYWHERE, value runs to the first space (${SPEC%% *})
    if "recw:" in w:
        spec = w.split("recw:", 1)[1].split(" ", 1)[0]
        kv = {}
        for part in spec.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k] = v
        out["recw"] = kv

    # the routed modes come first: they exit the job outright
    for m in SCRIPT_MODES:
        if (w.startswith(m) if m.endswith(":") else w.startswith(m)):
            out["mode"] = m.rstrip(":")
            out["mode_arg"] = w
            return out

    if w.startswith("disktriage:rm "):
        out["mode"] = "disktriage_rm"
        out["mode_arg"] = w[len("disktriage:rm "):]
        return out
    if w.startswith("disktriage"):
        out["mode"] = "disktriage"
        return out
    if w.startswith("rolleval:"):
        out["mode"] = "rolleval"
        out["mode_arg"] = [t for t in w[len("rolleval:"):].split(",") if t]
        return out
    for pref in ("unroll:", "resume2:", "warm2:"):
        if w.startswith(pref):
            out["mode"] = pref.rstrip(":")
            out["mode_arg"] = w[len(pref):]
            return out

    # the joint loss mode: MODE="${W%%@*}", REF after the first @ else 0
    head = w.split("recw:", 1)[0].strip() if out["recw"] else w
    base = head.split("@", 1)[0]
    if base in ("lse", "max", "sum"):
        out["loss_mode"] = base
        out["loss_ref"] = head.split("@", 1)[1] if "@" in head else "0"
    elif base in ("global", "na"):
        out["spatial"] = base
    return out


def to_config(w):
    """Legacy string -> the config dict a JSON input would carry."""
    d = decode_window(w)
    cfg = {k: v for k, v in d.items() if k != "raw" and v is not None}
    return cfg


def decode_config(cfg):
    """JSON config (dict or string) -> the same structure decode_window returns."""
    if isinstance(cfg, str):
        cfg = json.loads(cfg) if cfg.strip() else {}
    out = {"raw": None, "spatial": None, "loss_mode": None, "loss_ref": None,
           "recw": None, "mode": None, "mode_arg": None}
    unknown = [k for k in cfg if k not in out]
    if unknown:
        # A typo must be an error, not a silently ignored key. This is the one
        # thing the JSON path gives that the string path never could.
        raise ValueError(f"unknown config key(s): {sorted(unknown)}; "
                         f"known: {sorted(k for k in out if k != 'raw')}")
    out.update(cfg)
    return out


def equivalent(w):
    """True when the JSON route reproduces the legacy route for `w`."""
    a = decode_window(w)
    b = decode_config(to_config(w))
    b["raw"] = a["raw"]          # `raw` is provenance, not behaviour
    return a == b, a, b


def _cli():
    """Shadow mode: print what this decoder makes of a dispatch, change nothing.

    Phase 2 step 1. The workflow calls this AFTER checkout and BEFORE anything
    decodes `window` in shell, so every real run prints both readings and any
    disagreement shows up in the log without affecting the job. A refactor that
    silently changes behaviour is the failure mode here; running both in
    parallel for a while is how it gets caught by a run rather than by a
    retracted result.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="")
    ap.add_argument("--config", default="")
    a = ap.parse_args()
    if a.config.strip():
        d = decode_config(a.config)
        src = "config(JSON)"
    else:
        d = decode_window(a.window)
        src = "window(legacy string)"
    print(f"dispatch config decoded from {src}:")
    for k in sorted(d):
        if d[k] is not None:
            print(f"  {k:10s} {d[k]!r}")
    if a.window.strip() and not a.config.strip():
        ok, _, _ = equivalent(a.window)
        print(f"  json-route equivalence: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print("::warning::the JSON route disagrees with the legacy string "
                  "for this input — do NOT migrate this form until it matches")


if __name__ == "__main__":
    _cli()
