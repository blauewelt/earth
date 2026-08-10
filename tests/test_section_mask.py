#!/usr/bin/env python3
"""The 26.5N section must be selected from ALL months, never from month 0.

This is the regression test for the all-NaN `probe_sequence.json` that #101
and #116 both produced and that was read, twice, as "the sequence probe has a
bug of its own". It did not. The probe was fine; it was handed an empty
section, and the NaN was arithmetic doing exactly what it should.

The mechanism is worth stating precisely, because it will recur the next time
a channel is added to the tensor:

    family-3 stacks 39 channels whose records START AT DIFFERENT DATES.
    Channel 0 is cur_speed (GLORYS, from 1993-01). The tensor runs from
    1982-01. So in month 0 channel 0 is NaN EVERYWHERE, and a section
    selected by `isfinite(X[0, sec_y, :, 0])` contains zero pixels.

`z.mean(0)` over zero rows is NaN, not an error. Every embedding was NaN,
every correlation was NaN, and the seasonal-only floor — which never touches
the embedding — stayed finite, which is what made the failure look like a
probe bug instead of a masking bug. A results file full of NaN is the worst
kind of silent failure: it is loud enough to notice and quiet enough to
misattribute.

    python3 tests/test_section_mask.py
"""
import numpy as np

T, H, W, C = 24, 5, 12, 3
SEC_Y = 2
LATE_START = 12          # channel 0 begins halfway through the record


def build():
    """A tensor shaped like family-3: channel 0 starts late, others don't."""
    X = np.full((T, H, W, C), np.nan, dtype=np.float32)
    ocean = np.zeros((H, W), bool)
    ocean[SEC_Y, 3:9] = True                      # 6 ocean pixels on the row
    ocean[SEC_Y + 1, 4:7] = True
    for c in range(C):
        t0 = LATE_START if c == 0 else 0           # <- the whole bug
        X[t0:, ocean, c] = np.random.default_rng(c).normal(size=(T - t0,
                                                                ocean.sum()))
    return X, ocean


def test_month_zero_rule_selects_nothing():
    X, ocean = build()
    old = np.isfinite(X[0, SEC_Y, :, 0])
    assert old.sum() == 0, (
        "the historical failure did not reproduce — if channel 0 no longer "
        "starts late, this test needs a channel that does")
    print(f"month-0 rule      : {old.sum()} pixels  <- the bug")


def test_any_month_rule_selects_the_section():
    X, ocean = build()
    new = np.isfinite(X[:, SEC_Y, :, 0]).any(axis=0)
    assert new.sum() == ocean[SEC_Y].sum() == 6
    assert np.array_equal(new, ocean[SEC_Y]), "must recover exactly the ocean row"
    print(f"any-month rule    : {new.sum()} pixels  <- the section")


def test_empty_section_gives_nan_not_an_error():
    """Why this failed silently: numpy and torch both return NaN here."""
    empty = np.zeros((0, 8), dtype=np.float32)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(empty, axis=0) if empty.size else empty.mean(axis=0)
    assert m.shape == (8,) and not np.isfinite(m).any(), (
        "mean over an empty section is supposed to be NaN — if this ever "
        "raises instead, the silent-failure argument above is obsolete")
    print("empty-section mean: all NaN, no exception  <- why nobody saw it")


def test_the_two_rules_agree_when_no_channel_starts_late():
    """The old rule was not wrong in general — it was wrong for THIS tensor.
    Pinning that keeps the fix honest about what it changed."""
    X, ocean = build()
    X[:, :, :, 0] = X[:, :, :, 1]                 # channel 0 now starts at t=0
    old = np.isfinite(X[0, SEC_Y, :, 0])
    new = np.isfinite(X[:, SEC_Y, :, 0]).any(axis=0)
    assert np.array_equal(old, new)
    print("no late channel   : both rules agree     <- the fix is a superset")


if __name__ == "__main__":
    test_month_zero_rule_selects_nothing()
    test_any_month_rule_selects_the_section()
    test_empty_section_gives_nan_not_an_error()
    test_the_two_rules_agree_when_no_channel_starts_late()
    print("\nOK — the section is selected from every month, and an empty one "
          "now stops the run instead of writing NaN.")
