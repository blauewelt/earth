#!/usr/bin/env python3
"""`data/cone_geometry.json` is a build artefact of ml/cone.py — prove it.

The globe app's Cones tab draws the E-069 cone. Two things could rot: the
committed JSON could fall behind `ml/cone.py` (a reach changes, the page keeps
drawing last month's cone), or the export could stop being deterministic (every
regeneration a diff, so nobody regenerates it). This test closes both by
re-running the exporter in memory and comparing the WHOLE file, byte for byte.

`tests/test_cone_geometry.py` pins the geometry itself; this pins the copy the
browser reads. The JS port is certified separately, against this file's
reference dot sets, in `tests/data.spec.js`.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import export_cone_geometry as exp                                 # noqa: E402


class TestConeGeometryExport(unittest.TestCase):
    def setUp(self):
        with open(exp.OUT, encoding="utf-8") as fh:
            self.text = fh.read()
        self.doc = json.loads(self.text)

    def test_committed_file_is_a_fresh_export(self):
        """Byte-identical, not merely equivalent — a re-export must be a no-op
        diff, or the next session will not dare run it."""
        fresh = exp.dumps(exp.build())
        self.assertEqual(
            self.text, fresh,
            "data/cone_geometry.json is stale or hand-edited. Regenerate: "
            "python3 ml/export_cone_geometry.py")

    def test_export_is_deterministic(self):
        self.assertEqual(exp.dumps(exp.build()), exp.dumps(exp.build()))

    def test_token_budget(self):
        c = self.doc["counts"]
        # 4 wind x 8 + 4 surface-B x 80 + 32 depth x 6 + 2 C x 81 = 706 dots,
        # plus one 3x3 patch token per channel.
        self.assertEqual(c["inner_dots_A"], 8)
        self.assertEqual(c["inner_dots_B"], 80)
        self.assertEqual(c["inner_dots_rg"], 6)
        self.assertEqual(c["inner_dots_C"], 81)
        self.assertEqual(c["dot_tokens"], 706)
        self.assertEqual(c["patch_tokens"], 42)
        self.assertEqual(c["total_tokens"], 748)
        self.assertEqual(4 * 8 + 4 * 80 + 32 * 6 + 2 * 81, c["dot_tokens"])

    def test_every_r3_channel_has_a_family(self):
        fam = self.doc["channel_family"]
        self.assertEqual(len(fam), 42)
        self.assertEqual(sorted(fam) , sorted(self.doc["channels_r3"]))
        from collections import Counter
        self.assertEqual(Counter(fam.values()), Counter({"B": 36, "A": 4, "C": 2}))
        self.assertEqual(len(self.doc["depth_channels"]), 32)

    def test_reach_tables(self):
        inner = self.doc["reach_km"]["inner"]
        # A: 500 km at lag <= 1, then NOTHING (tau = 10 d is exhausted).
        self.assertEqual(inner["A"], [500.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # B: 129.6 km per pentad, strictly growing.
        self.assertEqual([round(r, 4) for r in inner["B"]],
                         [129.6, 259.2, 388.8, 518.4, 648.0, 777.6, 907.2])
        # C is L-shaped and DROPS at lag 2 — the atmospheric stirring going
        # out of memory, not a bug.
        self.assertEqual(inner["C"][:3], [500.0, 500.0, 388.8])
        self.assertLess(inner["C"][2], inner["C"][1])
        outer = self.doc["reach_km"]["outer"]
        self.assertEqual(len(outer), 144)
        self.assertTrue(all(b >= a for a, b in zip(outer, outer[1:])))
        self.assertEqual(outer[33], 4406.4)                 # last below the cap
        self.assertEqual(outer[34], 4444.0)                 # the cap binds here
        self.assertEqual(outer[-1], 4444.0)

    def test_outer_spiral_is_empty_inside_the_inner_window(self):
        """The codec already read that whole disc — by construction, not by
        accident, so the reference sets must show it."""
        import cone
        for lat in (0.0, 40.0, 70.0):
            for k in range(7):
                self.assertEqual(cone.outer_spiral(lat, k), [])
            self.assertEqual(len(cone.outer_spiral(lat, 7)), 24)

    def test_reference_sets_are_reproducible_from_cone_py(self):
        import cone
        for lat_s, fams in self.doc["reference"]["inner"].items():
            lat = float(lat_s)
            for f in "ABC":
                self.assertEqual(
                    [tuple(t) for t in fams[f]], cone.inner_dots(lat, f),
                    f"inner reference drifted at lat {lat_s} family {f}")
            self.assertEqual(
                [tuple(t) for t in fams["rg"]],
                cone.channel_dots(lat, self.doc["depth_channels"][0]))
        for lat_s, ks in self.doc["reference"]["outer"].items():
            for k_s, pts in ks.items():
                self.assertEqual([tuple(t) for t in pts],
                                 cone.outer_spiral(float(lat_s), int(k_s)))

    def test_dot_counts_do_not_depend_on_latitude(self):
        """Latitude converts km to cells; it never changes how many dots the
        cone has. A count that moved with the row would be three experiments
        in one tensor."""
        counts = {f: set() for f in ("A", "B", "C", "rg")}
        for fams in self.doc["reference"]["inner"].values():
            for f in counts:
                counts[f].add(len(fams[f]))
        self.assertEqual(counts, {"A": {8}, "B": {80}, "C": {81}, "rg": {6}})

    def test_window_is_the_tensor_window(self):
        w = self.doc["window"]
        self.assertEqual((w["lat0"], w["lat1"]), (0.0, 70.0))
        self.assertEqual((w["lon0"], w["lon1"]), (-100.0, 20.0))
        self.assertEqual(w["dlat"], 0.25)
        self.assertEqual(w["ny"], int(round((w["lat1"] - w["lat0"]) / w["dlat"])) + 1)
        self.assertEqual(w["nx"], int(round((w["lon1"] - w["lon0"]) / w["dlat"])) + 1)


if __name__ == "__main__":
    unittest.main()
