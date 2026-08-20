"""The RING geometry survives the cadence rewrite, bit for bit.

WHY THIS FILE EXISTS, and it is a gap that cost a diagnosis rather than a
hypothetical. `tests/test_roll_monthly_identity.py` pins `ml/rollout_spatial.py`
against `BASE_SHA` on the shared toy from `tests/test_rollout_spatial.py`, and
that fixture carries exactly two heads: stencil 1 and stencil 9 with `ring_km`
UNSET. `_ring_on(0)` is false for both, so `build_stencil` takes the fixed-table
branch in every archived comparison and the `spiral:` branch — the geometry
EVERY published stage-2 head since E-032 actually uses (`stencil 145`,
`ring spiral:111,4444,0.71,0.5`) — has never once been compared across a change
to this file.

On 2026-08-20 that gap turned a surprising number into an unfalsifiable one.
#422 rolled the E-043b all-longitude xl144 head, a stencil-145 spiral head, on
the first code no ring head had rolled before (the `TimeAxis` rewrite, e9f3d8d +
ef62fbf) and returned a corridor lead-time profile flat to 0.004 over twelve
months. The leading hypothesis — that the skill loop's neighbour gather reads
OBSERVED Z at rolled timesteps for the ring path, so the centre pixel rolls
while 144 neighbours carry ground truth — could be answered by reading
(`roll_step` is byte-identical across the rewrite and gathers from the rolled
`Zwin` in both loops) but not by any test in the repo. It can now.

WHAT IT ASSERTS. The same toy, plus a THIRD head whose only difference from the
fixture's stencil-9 head is `ring_km` — so slot count, model shape, weights and
seed are held fixed and the ring geometry is the single moving part (the same
isolation `build_stencil`'s own docstring claims for `--stencil 9 --ring-km 222`
vs `--stencil 9`). Both versions of `rollout_spatial.py` roll all three heads
and the two artefacts must be IDENTICAL after the two exclusions
`test_roll_monthly_identity.py` documents and counts — `wall_s`, a clock
reading, and `horizon_auc_daymatched`, a key BASE_SHA never wrote, which is
asserted equal to its scope's `horizon_auc` before it is dropped.

IT IS NOT VACUOUS, and that was measured rather than argued (ml/CLAUDE.md §0.2).
Against a copy of `ml/` in which the skill loop was edited to do exactly what the
hypothesis describes — `Zwin = zwin_from_true(t_tgt, K); Zwin[:, -1] = zhat` for
`NBR_t is not None`, i.e. neighbours read observed Z at the rolled timestep while
the centre keeps its rolled value — check 3 FAILS and checks 1 and 2 still pass.
So a green run here is evidence about that specific wiring, not just about the
file parsing.

WHAT IT DOES NOT ASSERT, said here so nobody reads more into a pass. The toy's
heads are RANDOM-INIT: they have no skill, so their `msss_clim` is ~0.05 flat
and this fixture cannot exhibit a decay profile at all. A pass means "the ring
path computes the same numbers it did before the rewrite", which is the
question. It does NOT mean "a flat lead-time profile is impossible", and no
synthetic head at this size could mean that.
"""

import json
import os

import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, ML)

from test_rollout_spatial import build_fixture, DZ, K            # noqa: E402
from test_roll_monthly_identity import (BASE_SHA, base_copy,     # noqa: E402
                                        run, strip_wall,
                                        strip_daymatched)
from temporal import TemporalTransformer                         # noqa: E402

# Small radii, because the toy ocean is 8x10 cells spanning 20 deg of latitude
# (~2.2 deg per row, ~250 km): a production 111-4444 km spiral would put every
# slot outside the window and score a stencil of misses, which is a valid
# geometry and a useless test. These land on distinct cells AND leave some
# slots off-window, so the `-1` missing path is exercised too.
RING = "spiral:200,900,0.71,0.5"
STENCIL = 9


def add_ring_head(tmp):
    """The fixture's stencil-9 head, with a ring and nothing else changed."""
    torch.manual_seed(10 + STENCIL)          # the fixture's own seeding rule
    hm = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                             k_max=max(K, 36), stencil=STENCIL)
    hp = os.path.join(tmp, f"toy_s{STENCIL}ring_s0.pt")
    torch.save({"model": hm.state_dict(),
                "args": {"K": K, "d_model": 8, "layers": 1, "unroll": 1,
                         "seed": 0, "stencil": STENCIL, "ring_km": RING}}, hp)
    return hp


def main():
    tmp = tempfile.mkdtemp(prefix="ringid-")
    try:
        f = build_fixture(tmp)
        f["heads"] = list(f["heads"]) + [add_ring_head(tmp)]

        new_txt, _ = run(os.path.join(ML, "rollout_spatial.py"), f,
                         os.path.join(tmp, "new.json"), os.path.join(tmp, "cn"))
        old_txt, _ = run(base_copy(tmp), f,
                         os.path.join(tmp, "old.json"), os.path.join(tmp, "co"))

        obj_new, obj_old = json.loads(new_txt), json.loads(old_txt)

        # 1. the ring head was actually SCORED. A label carrying the spiral
        #    spec is the only proof the ring branch of build_stencil ran; a
        #    silently-dropped head would otherwise leave both sides equal and
        #    the test green over nothing (the #421 shape of failure).
        ring_labels = [lab for lab in obj_new["heads"]
                       if "spiral" in lab.replace(",", "-")]
        assert len(ring_labels) == 1, sorted(obj_new["heads"])
        assert len(obj_new["heads"]) == 3, sorted(obj_new["heads"])
        assert sorted(obj_new["heads"]) == sorted(obj_old["heads"])
        print(f"1. three heads scored on both versions, one of them the RING "
              f"head — {ring_labels[0]}")

        # 2. the ring head's geometry is not the fixed table's. Slot masks live
        #    in static_ctx and never reach the artefact, so the check is the
        #    one the artefact CAN make: a ring head and a table head with the
        #    same slot count must not produce the same record.
        a = json.dumps(obj_new["heads"][ring_labels[0]], sort_keys=True)
        b = json.dumps(obj_new["heads"]["s9_s0"], sort_keys=True)
        assert a != b, ("the ring head and the fixed-table stencil-9 head "
                        "produced identical records — the ring_km argument "
                        "did not reach build_stencil, and this test is "
                        "comparing one geometry with itself")
        print("2. the ring head's record differs from the same-slot-count "
              "fixed-table head's — `ring_km` reached the geometry")

        # 3. and the two versions agree, byte for byte, after exactly the two
        #    documented exclusions — each `horizon_auc_daymatched` pinned
        #    against the `horizon_auc` it duplicates at monthly BEFORE it is
        #    dropped, so the strip can never hide a moved number.
        new_raw, obj_new, w_new = strip_wall(new_txt)
        old_raw, obj_old, w_old = strip_wall(old_txt)
        assert w_new == w_old == len(obj_new["heads"]) == 3, (w_new, w_old)
        new_raw, got_new = strip_daymatched(new_raw, obj_new)
        old_raw, got_old = strip_daymatched(old_raw, obj_old)
        assert got_old == [], ("BASE_SHA wrote horizon_auc_daymatched — the "
                               "strip is hiding a difference, not a new key")
        assert got_new, "no horizon_auc_daymatched keys found in the new payload"
        for lab, scope, v, auc in got_new:
            assert v == auc, (lab, scope, v, auc)
        new_s, old_s = new_raw, old_raw
        n_new = len(got_new)
        assert new_s == old_s, (
            f"ml/rollout_spatial.py no longer reproduces {BASE_SHA} on the "
            f"monthly toy for a RING-STENCIL head. Every published stage-2 "
            f"head since E-032 uses this geometry, so a monthly number that "
            f"differs here makes the whole archive incomparable — the same "
            f"claim test_roll_monthly_identity.py makes for the fixed table, "
            f"which does not cover this branch.")
        print(f"3. the ring head's monthly roll is BIT-IDENTICAL to "
              f"{BASE_SHA}'s, after the same two documented exclusions "
              f"({n_new} `horizon_auc_daymatched` keys, each equal to its "
              f"scope's `horizon_auc`, plus the `wall_s` clock readings)")

        print("\nring-geometry roll identity: all 3 checks hold ✓")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
