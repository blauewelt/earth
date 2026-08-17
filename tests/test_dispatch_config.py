#!/usr/bin/env python3
"""Pin what the dispatch `window` field means TODAY, before anything moves.

Chris, 2026-08-17: *"test the refactor well making sure it's not changing
behavior at all."* That is the right bar, and the only way to meet it rather
than assert it is to make today's behaviour a test first — so a later change
that alters it fails here instead of on a rented GPU.

Every string below was taken from an actual decode site in
`.github/workflows/ml-train.yml`. To re-derive the list:

    grep -n 'inputs.window' .github/workflows/ml-train.yml

Two things are asserted for each form:

  1. **The legacy decode is what the shell does**, quirks included. The
     expectations are hand-written from the shell, not from the decoder, so
     the decoder cannot define its own correctness.

  2. **The JSON route agrees with the legacy route.** decode_config(to_config(s))
     must equal decode_window(s) for every form. This is what makes "behaviour
     preserved" checkable once the workflow switches over: the legacy decode
     stays in the test as the reference even after it stops being the path.

    python3 tests/test_dispatch_config.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))

from dispatch_config import (decode_window, decode_config, to_config,  # noqa: E402
                             equivalent)

# (input, the subset of fields the shell would set)
CASES = [
    # the field's original job
    ("global",                  {"spatial": "global"}),
    ("na",                      {"spatial": "na"}),
    # joint loss mode, with and without a reference
    ("lse",                     {"loss_mode": "lse", "loss_ref": "0"}),
    ("max",                     {"loss_mode": "max", "loss_ref": "0"}),
    ("sum",                     {"loss_mode": "sum", "loss_ref": "0"}),
    ("lse@twin",                {"loss_mode": "lse", "loss_ref": "twin"}),
    ("max@0.31",                {"loss_mode": "max", "loss_ref": "0.31"}),
    # stage-2 knobs
    ("unroll:4",                {"mode": "unroll", "mode_arg": "4"}),
    ("resume2:run-201",         {"mode": "resume2", "mode_arg": "run-201"}),
    ("resume2:run-201@3e-4",    {"mode": "resume2", "mode_arg": "run-201@3e-4"}),
    ("warm2:run-88",            {"mode": "warm2", "mode_arg": "run-88"}),
    # investigation / routed modes
    ("disktriage",              {"mode": "disktriage"}),
    ("disktriage:rm /opt/x /opt/y",
                                {"mode": "disktriage_rm", "mode_arg": "/opt/x /opt/y"}),
    ("rolleval:run-355,run-356",
                                {"mode": "rolleval", "mode_arg": ["run-355", "run-356"]}),
    ("dectrain",                {"mode": "dectrain", "mode_arg": "dectrain"}),
    ("project:2004",            {"mode": "project", "mode_arg": "project:2004"}),
    ("sroll:89",                {"mode": "sroll", "mode_arg": "sroll:89"}),
    # recw, which the shell matches ANYWHERE and terminates at the first SPACE
    ("recw:a=1;b=2",            {"recw": {"a": "1", "b": "2"}}),
    ("lse recw:a=1;b=2",        {"loss_mode": "lse", "loss_ref": "0",
                                 "recw": {"a": "1", "b": "2"}}),
    ("lse recw:a=1 trailing",   {"loss_mode": "lse", "loss_ref": "0",
                                 "recw": {"a": "1"}}),
    ("",                        {}),
]


def main():
    # ---- 1: the legacy decode matches the shell, field by field -----------
    for s, want in CASES:
        got = decode_window(s)
        for k, v in want.items():
            assert got[k] == v, f"{s!r}: {k} = {got[k]!r}, shell gives {v!r}"
        # everything NOT named must be None — a form must not set a field the
        # shell leaves untouched, which is how an overloaded parser goes wrong
        for k, v in got.items():
            if k == "raw" or k in want:
                continue
            assert v is None, f"{s!r}: {k} = {v!r}, but the shell sets nothing"
    print(f"  1. {len(CASES)} legacy forms decode exactly as the shell does")

    # ---- 2: the JSON route is equivalent for every one --------------------
    for s, _ in CASES:
        ok, a, b = equivalent(s)
        assert ok, f"{s!r}: legacy {a} != json {b}"
    print(f"  2. decode_config(to_config(s)) == decode_window(s) for all "
          f"{len(CASES)} — the JSON path changes no behaviour")

    # ---- 3: the quirks are pinned as quirks, not fixed --------------------
    # recw runs to the first SPACE, so a second token is NOT part of it
    assert decode_window("lse recw:a=1 b=2")["recw"] == {"a": "1"}
    # a routed mode wins over a loss mode even if the string looks like both
    assert decode_window("project:lse")["mode"] == "project"
    # the reference defaults to the STRING "0", not the integer
    assert decode_window("lse")["loss_ref"] == "0"
    print("  3. the sharp edges are pinned as-is (recw stops at a space; a "
          "routed mode wins; loss_ref defaults to the string '0')")

    # ---- 4: the one thing JSON buys — a typo is an error ------------------
    try:
        decode_config({"loss_mode": "lse", "unrol": 4})
        raise AssertionError("a misspelled key was accepted")
    except ValueError as e:
        assert "unrol" in str(e)
    print("  4. an unknown config key raises instead of being silently "
          "ignored — the one thing the string form could never do")

    print("\ntests/test_dispatch_config.py: all 4 checks passed")


if __name__ == "__main__":
    main()
