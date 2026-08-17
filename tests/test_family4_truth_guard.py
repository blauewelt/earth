#!/usr/bin/env python3
"""A cached tensor with no labels must not be mistaken for a finished one.

Run #365 built the 33 GB pentad tensor on a box where `truth_pentad.npz` did
not exist yet, so it wrote a physically correct state tensor carrying **no
transport labels at all**. The recipe guard compares one string:

    if prev == RECIPE_REV: skip

and the recipe was unchanged, because the CODE was unchanged. So run #367 —
the 200 M pentad codec — started on that same box, skipped the rebuild in
0.0 minutes, and was on course to train for twenty hours and then die in
`probe_kfold` on `KeyError: 'rapid'`, which is the single number E-038 exists
to produce. It was cancelled four minutes in, and the guard is the fix: a
recipe string is a claim about the code, not about what was on disk beside it.

`missing_truth_keys` answers "which labels does the truth file offer that this
cached tensor does not carry" from two npz DIRECTORIES — headers, not data, so
it costs milliseconds against a 33 GB file. What is pinned:

  1. a tensor built without labels, checked against a truth file that has
     them, reports every missing key INCLUDING the `rapid` alias the trainer
     actually reads (`d["rapid"]`, train.py:671) — the alias is a separate
     key in the npz, so a check that only looked for `truth_*` would pass a
     tensor the trainer cannot use;
  2. a tensor built WITH labels reports nothing, so a good cache is still
     skipped and no box is put into a rebuild loop;
  3. with no truth file present there is nothing to compare against and the
     answer is empty — a box that legitimately has no labels must not rebuild
     forever;
  4. a truth file carrying a target the tensor missed (a cable series added
     later) is caught, which is the same failure one iteration on.

    python3 tests/test_family4_truth_guard.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml"))

import build_family4 as f4          # noqa: E402


def write_truth(path, keys):
    np.savez(path, epoch=np.array("1982-01-01"), pentad_days=np.array(5),
             **{k: np.zeros((3, 2)) for k in keys})


def main():
    tmp = tempfile.mkdtemp(prefix="f4guard_")
    truth = os.path.join(tmp, "truth_pentad.npz")
    original = f4.TRUTH_PENTAD
    try:
        # ---- 1: the #365 tensor — recipe current, labels absent -----------
        write_truth(truth, ["truth_rapid", "truth_fc"])
        f4.TRUTH_PENTAD = truth
        no_labels = {"X", "months", "lats", "lons", "chan", "norm", "recipe"}
        lack = f4.missing_truth_keys(no_labels)
        assert lack == ["rapid", "truth_fc", "truth_rapid"], lack
        print("  1. a label-less cached tensor reports all three missing keys, "
              "the `rapid` alias train.py reads included")

        # ---- 2: a properly built tensor is still skipped ------------------
        good = no_labels | {"truth_rapid", "truth_fc", "rapid"}
        assert f4.missing_truth_keys(good) == [], f4.missing_truth_keys(good)
        print("  2. a tensor built WITH labels reports nothing — a good cache "
              "is still skipped, no rebuild loop")

        # ---- 3: no truth file at all — nothing to compare against ---------
        f4.TRUTH_PENTAD = os.path.join(tmp, "absent.npz")
        assert f4.missing_truth_keys(no_labels) == []
        print("  3. with no truth file present the answer is empty — a box "
              "that legitimately has no labels does not rebuild forever")

        # ---- 4: a target added to the truth file later --------------------
        f4.TRUTH_PENTAD = truth
        write_truth(truth, ["truth_rapid", "truth_fc", "truth_move"])
        assert f4.missing_truth_keys(good) == ["truth_move"], \
            f4.missing_truth_keys(good)
        print("  4. a target added to the truth file after the build is "
              "caught — the same failure one iteration on")
    finally:
        f4.TRUTH_PENTAD = original

    print("\ntests/test_family4_truth_guard.py: all 4 checks passed")


if __name__ == "__main__":
    main()
