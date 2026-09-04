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

    # ------------------------------------------------------------ global ----
    # The `global` block is a CONTRACT: the browser replays its own port of
    # the sunflower against `global.refs[i].cells` and deep-equals the arrays,
    # the same discipline `reference` already carries for the North Atlantic
    # window. So the schema is pinned here, cell for cell, against ml/cone.py.
    def test_global_grid_is_the_family7_grid(self):
        g = self.doc["global"]
        self.assertEqual((g["ny"], g["nx"]), (721, 1440))
        self.assertEqual((g["lat0"], g["lon0"], g["step"]),
                         (-90.0, -180.0, 0.25))
        # 1440 x 0.25 = 360: the axis CLOSES, which is the whole reason the
        # sampler may wrap on it. That equality is what `wrap` asserts.
        self.assertEqual(g["nx"] * g["step"], 360.0)
        self.assertTrue(g["wrap"])
        self.assertEqual(g["lat0"] + (g["ny"] - 1) * g["step"], 90.0)
        self.assertIn("cells is cone.inner_dots", g["schema"])

    def test_global_adds_no_key_and_removes_none(self):
        """Everything outside `global` is what it was before family 7: the
        r3 channel list has no family-L channel in it, so the existing tables
        cannot move, and a change that moved one would fail here as well as in
        the byte-identity test above."""
        self.assertNotIn("L", self.doc["families"])
        self.assertNotIn("L", self.doc["slots"])
        self.assertNotIn("L", self.doc["reach_km"]["inner"])
        self.assertNotIn("L", set(self.doc["channel_family"].values()))

    def test_global_refs_reproduce_cone_py_with_the_wrap(self):
        import cone
        g = self.doc["global"]
        nx, step = g["nx"], g["step"]
        seen = set()
        for ref in g["refs"]:
            self.assertEqual(set(ref) - {"anchor_id"},
                             {"family", "lag", "anchor", "cells"})
            a = ref["anchor"]
            self.assertEqual(set(a), {"row", "col", "lat", "lon"})
            self.assertEqual(a["lat"], g["lat0"] + a["row"] * step)
            self.assertEqual(a["lon"], g["lon0"] + a["col"] * step)
            want = [[a["row"] + dy, (a["col"] + dx) % nx]
                    for l, dy, dx in cone.inner_dots(a["lat"], ref["family"],
                                                     L_in=6, dlat_deg=step)
                    if l == ref["lag"]]
            self.assertEqual(ref["cells"], want,
                             f"{ref['family']} lag {ref['lag']} at "
                             f"{ref['anchor_id']}")
            # the column ALWAYS wraps into range; the row may leave it
            for r, c in ref["cells"]:
                self.assertTrue(0 <= c < nx)
            seen.add((ref["family"], ref["lag"], ref["anchor_id"]))
        self.assertEqual(len(seen), len(g["refs"]))
        self.assertEqual({f for f, _, _ in seen}, {"A", "B", "C", "L"})
        self.assertEqual({l for f, l, _ in seen if f == "A"}, {0, 1})
        for f in ("B", "C", "L"):
            self.assertEqual({l for ff, l, _ in seen if ff == f},
                             {1, 2, 3, 4, 5, 6})
        self.assertEqual({a for _, _, a in seen},
                         {"wrap_west", "wrap_east", "near_pole",
                          "antarctic_coast", "equatorial_pacific"})

    def test_global_lag0_is_empty_and_the_patch_is_exported_once(self):
        """Lag 0 has no DOTS — it is the codec's 3x3 patch — so family A's
        lag-0 reference is empty by construction and a port that produced
        nine cells there would be drawing the patch twice."""
        g = self.doc["global"]
        zero = [r for r in g["refs"] if r["lag"] == 0]
        self.assertTrue(zero)
        self.assertTrue(all(r["family"] == "A" for r in zero))
        self.assertTrue(all(r["cells"] == [] for r in zero))
        self.assertEqual(g["patch_cells"],
                         [[dy, dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])

    def test_global_wrap_anchors_really_wrap(self):
        """The point of the two dateline anchors: at column 1 a family-B dot
        at lag 6 reaches columns near 1439, and at column 1439 it reaches
        columns near 0. Without the wrap both would be off the tensor."""
        g = self.doc["global"]
        west = [r for r in g["refs"]
                if r["anchor_id"] == "wrap_west" and r["family"] == "B"
                and r["lag"] == 6][0]
        east = [r for r in g["refs"]
                if r["anchor_id"] == "wrap_east" and r["family"] == "B"
                and r["lag"] == 6][0]
        self.assertEqual(west["anchor"]["col"], 1)
        self.assertEqual(east["anchor"]["col"], 1439)
        self.assertTrue(max(c for _, c in west["cells"]) > 1400)
        self.assertTrue(min(c for _, c in east["cells"]) < 40)

    def test_global_outer_refs_are_outer_spiral(self):
        import cone
        g = self.doc["global"]
        nx, step = g["nx"], g["step"]
        self.assertEqual(sorted({r["lag"] for r in g["outer_refs"]}),
                         [7, 35, 143])
        self.assertEqual(sorted({r["anchor_id"] for r in g["outer_refs"]}),
                         ["wrap_east", "wrap_west"])
        for ref in g["outer_refs"]:
            self.assertEqual(ref["family"], "outer")
            a = ref["anchor"]
            want = [[a["row"] + dy, (a["col"] + dx) % nx]
                    for dy, dx in cone.outer_spiral(a["lat"], ref["lag"],
                                                    dlat_deg=step, L_in=6)]
            self.assertEqual(ref["cells"], want)
            self.assertEqual(len(ref["cells"]), 24)

    def test_global_family_L_is_flat_and_wide(self):
        """Family L (land surface state, E-071 §6.3): v = 0, so the reach is
        the correlation length at EVERY lag — a flat cone — and the memory is
        months, so the inner window never cuts it off the way family A's
        ten-day memory does."""
        g = self.doc["global"]["families"]["L"]
        self.assertEqual(g["v_ms"], 0.0)
        self.assertEqual(g["L_corr_km"], 400.0)
        self.assertEqual(g["reach_km"], [400.0] * 7)
        self.assertEqual(g["slots"], [6] * 7)     # the SLOT_MIN floor binds
        self.assertEqual(sorted(g["channels"]), ["log_swe", "soilw", "tsoil"])
        self.assertEqual(g["inner_lags"], [0, 1, 2, 3, 4, 5, 6])
        # and those three names left A, which is what Phase E decided
        self.assertNotIn("log_swe", self.doc["global"]["families"]["A"]
                         ["channels"])

    def test_global_near_pole_is_measured_not_fixed(self):
        """E-071 §1 wants dots placed as destination points on the sphere.
        They are not, yet — so the block RECORDS what the flat-earth placement
        does at 85 N instead of pretending it is right."""
        n = self.doc["global"]["near_pole"]
        self.assertEqual((n["lat"], n["lon"]), (85.0, 0.0))
        # the cos floor does NOT bind at 85 N (cos = 0.0872 > 0.05); it would
        # only from ~87.1 N. What bites is the flat-earth approximation.
        self.assertFalse(n["floor_binds"])
        self.assertGreater(n["coslat"], n["cos_floor"])
        self.assertGreater(n["n_over_one_cell"], 0)
        self.assertLessEqual(n["n_over_one_cell"], n["n_dots"])
        self.assertGreater(n["max_error_km"], n["cell_km"])

    def test_window_is_the_tensor_window(self):
        w = self.doc["window"]
        self.assertEqual((w["lat0"], w["lat1"]), (0.0, 70.0))
        self.assertEqual((w["lon0"], w["lon1"]), (-100.0, 20.0))
        self.assertEqual(w["dlat"], 0.25)
        self.assertEqual(w["ny"], int(round((w["lat1"] - w["lat0"]) / w["dlat"])) + 1)
        self.assertEqual(w["nx"], int(round((w["lon1"] - w["lon0"]) / w["dlat"])) + 1)


if __name__ == "__main__":
    unittest.main()
