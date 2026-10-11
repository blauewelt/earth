"""Floating objects seen from the air — one row per OBJECT (family 1.3, E-092).

PLAIN ENGLISH. In October 2016 The Ocean Cleanup flew a C-130 over the Great
Pacific Garbage Patch and photographed strips of sea; every object visible in
the photo mosaics was outlined, measured and classified (net, rope, buoy,
container, ...), and its weight estimated from its top-view area. This store is
that list: 1,595 objects, each with its size, its estimated mass (low / mid /
high) and its type. It is the aircraft's own record of the biggest plastic,
the size class nets cannot catch (`ml/plans/E092_family13_plastic.md` §4).

SOURCE. Lebreton et al. 2018 (figshare 5873142, CC BY 4.0), the
`MosaicDebrisInfo` sheet, parsed by the shared `_lebreton2018.py` (which
asserts the 1,595 rows and their 100 % join to StationInfo through the
float-noise event ids).

WHAT A ROW IS.
  time_s    the mosaic's flight: StationInfo's start day and start hour of the
            object's event. The 31 aircraft events carry NO start hour
            (measured), so every row is that day at 12:00 UTC with qc bit 16 —
            the `gbif`/`glodap` precedent for a date without a time.
  lat, lon  the midpoint of the object's start and end coordinates.
  platform  platform_hash("lebreton2018|AERIAL expedition 2016|<platform>|
            mosaic <event id>") — one mosaic (a flight line) per hash.
  values    C = 7: length_m, width_m, topview_area_m2, kg_lo, kg_mid, kg_hi
            (the producer's weight estimates), lt50 (1 when the producer's
            "< 50 cm?" is yes, 0 when no, NaN otherwise).
  qc        bits 0-3: object type (`TYPE_CODES`, the producer's own labels;
            a label not in the table REFUSES the build); bit 4 (16): time of
            day not given.

THE BAKE-TIME CHECK (plan §6), measured and recorded in counts
(`mosaic_check`): the count of objects with "< 50 cm?" = no, divided by the
mosaic's `Sampling area (km2)`, against (a) Lebreton 2018's published
megaplastic concentration per mosaic and (b) the same mosaics' CD7 in
Lebreton et al. 2024 (figshare 25674120). Measured 2026-10-11: it reproduces
2018's published value on only 2 of 31 mosaics, and 2024's (to its two
decimals) on all 31 — the object list is the one the producer's 2024
reanalysis counts. The plan's falsifier names 2018's numbers, so this is
recorded, not refused: the decision belongs to the plan, not to the adapter.
"""
import numpy as np

import build_family10_stores as f10b
from family1.adapters import _lebreton2018 as lb
from family1.adapters import mplastic as mp

TYPE_CODES = {1: "Bundled net", 2: "Loose net", 3: "Rope", 4: "Buoy",
              5: "Container", 6: "Life ring", 7: "Unknown"}
CODE_OF = {v: k for k, v in TYPE_CODES.items()}
QC_NO_TIME = mp.QC_NO_TIME

CHANNELS = (("length_m", "m", 0.0, 100.0),
            ("width_m", "m", 0.0, 50.0),
            ("topview_area_m2", "m2", 0.0, 1000.0),
            ("kg_lo", "kg", 0.0, 1e5),
            ("kg_mid", "kg", 0.0, 1e5),
            ("kg_hi", "kg", 0.0, 1e5),
            ("lt50", "1 = shorter than 50 cm (producer's flag)", 0.0, 1.0))


def object_rows(L, counts):
    """The objects -> (t, lat, lon, values (n, 7), platform texts, qc)."""
    ev = L["events"]
    t, la, lo, v, pl, qc = [], [], [], [], [], []
    types = {}
    for o in L["objects"]:
        e = ev[o["event"]]
        code = CODE_OF.get(o["type"])
        if code is None:
            mp.refuse(f"MosaicDebrisInfo object type {o['type']!r} is not in "
                      f"TYPE_CODES {sorted(CODE_OF)}")
        types[o["type"]] = types.get(o["type"], 0) + 1
        tt, q = mp.t_of(e["y"], e["mo"], e["d"], e["hour"])
        t.append(tt)
        la.append((o["lat0"] + o["lat1"]) / 2)
        lo.append((o["lon0"] + o["lon1"]) / 2)
        v.append([lb.num(o[k]) for k in ("length_m", "width_m", "area_m2",
                                          "kg_lo", "kg_mid", "kg_hi")] +
                 [o["lt50"]])
        pl.append(f"lebreton2018|AERIAL expedition 2016|{e['platform']}|"
                  f"mosaic {o['event']}")
        qc.append(code | q)
    counts["objects_by_type"] = types
    vals = np.array([[np.nan if x is None else float(x) for x in r]
                     for r in v], np.float64).reshape(len(v), len(CHANNELS))
    return (np.asarray(t, np.int64), np.asarray(la), np.asarray(lo), vals,
            pl, np.asarray(qc, np.uint8))


def mosaic_check(L, book24):
    """Objects >= 50 cm per mosaic area vs 2018's and 2024's published
    megaplastic concentration (plan §6)."""
    ev = L["events"]
    per = {}
    for o in L["objects"]:
        if o["lt50"] == 0:
            per[o["event"]] = per.get(o["event"], 0) + 1
    aer = [e for e in ev.values() if e["type"] == "aerial"]
    rows24 = []
    if book24 is not None:
        for r in book24.get("Aerial survey", []):
            if mp.l2024_expedition(r["Expedition"]) == "AERIAL Expedition":
                rows24.append(r)
    m18 = m24 = n24 = 0
    worst = []
    for e in aer:
        a = lb.num(e["area_km2"])
        c = per.get(e["id"], 0) / a if a else None
        pub = lb.num(e["c"]["mega"]["n"][3])
        if c is not None and pub is not None and \
                abs(c - pub) <= 1e-6 * max(1.0, pub):
            m18 += 1
        else:
            worst.append({"event": e["id"], "objects_per_km2": c,
                          "published_2018": pub})
        hit = [r for r in rows24
               if (int(r["Year"]), int(r["Month"]), int(r["Day"])) ==
               (e["y"], e["mo"], e["d"])
               and abs(float(r["Lat"]) - e["lat"]) <= mp.DEDUP_DEG
               and abs(float(r["Lon"]) - e["lon"]) <= mp.DEDUP_DEG]
        if len(hit) == 1 and c is not None:
            n24 += 1
            cd7 = lb.num(hit[0].get("CD7"))
            if cd7 is not None and abs(round(c, 2) - cd7) <= 0.0051:
                m24 += 1
    return {"mosaics": len(aer), "match_2018": m18,
            "matched_to_2024": n24, "match_2024": m24,
            "not_matching_2018": worst[:40]}


class MObjectsAdapter(f10b.SourceAdapter):
    store = "mobjects"
    title = ("Objects identified in The Ocean Cleanup's 2016 aircraft photo "
             "mosaics of the Great Pacific Garbage Patch (Lebreton et al. "
             "2018), one row per object")
    family = "13"
    distribution = "public"
    licence = {"name": "CC BY 4.0 (The Ocean Cleanup)",
               "redistribution": "attribution", "derived_works": "free",
               "attribution": ("The Ocean Cleanup: Lebreton et al. 2018, Sci. "
                               "Rep. 8:4666, data "
                               "doi:10.6084/m9.figshare.5873142")}
    time_dtype = "int32"
    platform_meta = False
    credentials = ()
    channels = CHANNELS
    # one object in one photograph: a point and an instant, the family's
    # point label for both (an Argo profile is (-4, -4), E-078 §2)
    log2_fp = -4.0
    log2_dt = -4.0
    per_year = False
    first_year = 2016
    qc_policy = ("qc bits 0-3 = object type (" +
                 "; ".join(f"{k} {v}" for k, v in TYPE_CODES.items()) +
                 "); bit 4 (16): time of day not given, placed at 12:00 UTC. "
                 "Out-of-bounds values -> NaN, counted.")
    sources = (lb.API_FILES, mp.FIG2024_ARTICLE)
    verified = ("2026-10-11 from the sandbox: figshare 5873142's listing and "
                "MosaicDebrisInfo (1,595 objects, seven types, 100 % joined); "
                "the per-mosaic check against both papers' concentrations")
    notes = ("The objects of the 2021-22 aircraft surveys are not published "
             "per object (Lebreton 2024 gives per-survey concentrations only); "
             "they are in mplastic.")
    smoke_window = ("2016-01-01", "2016-12-31")
    smoke_probe_month = "2016-10"
    fetch_month_scope = "archive"

    def __init__(self):
        self._mp = mp.MPlasticAdapter()

    def _rows(self, ctx, t_lo, t_hi):
        counts = {}
        L, _f = self._mp.lebreton2018(ctx)
        book24 = self._mp.lebreton2024(ctx)
        mc = mosaic_check(L, book24)
        # a counts dict holds numbers per key (`_merge_counts` sums them);
        # the per-mosaic list is a ledger of its own
        counts["mosaic_check_not_matching_2018"] = mc.pop("not_matching_2018")
        counts["mosaic_check"] = mc
        t, la, lo, v, pl, qc = object_rows(L, counts)
        counts["objects_read"] = int(t.size)
        keep = (t >= t_lo) & (t <= t_hi)
        counts["objects_outside_window"] = int((~keep).sum())
        oob = self.mask_bounds(v)
        if oob:
            counts["out_of_bounds"] = oob
        counts["rows_kept"] = int(keep.sum())
        idx = np.flatnonzero(keep)
        return t[idx], la[idx], lo[idx], v[idx], [pl[i] for i in idx], \
            qc[idx], counts

    def _yield(self, ctx, t_lo, t_hi, label=None):
        t, la, lo, v, pl, qc, counts = self._rows(ctx, t_lo, t_hi)
        years = np.array([mp.day_of(x)[0] for x in t], np.int64)
        for y in sorted(set(years.tolist())):
            k = np.flatnonzero(years == y)
            yield (label or y), self.pack(
                t[k], la[k], lo[k], v[k],
                np.array([f10b.platform_hash(pl[i]) for i in k], np.int64),
                qc[k]), None
        yield (None if label is None else label), None, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        files = lb.resolve(self._mp._json(ctx, lb.API_FILES))
        return {"dataset": "Lebreton et al. 2018, MosaicDebrisInfo",
                "lebreton2018_files": {n: f["id"] for n, f in files.items()}}

    def fetch_stream(self, ctx):
        yield from self._yield(ctx, ctx.t_lo, ctx.t_hi)

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        yield from self._yield(ctx, max(lo, ctx.t_lo), min(hi, ctx.t_hi),
                               label=f"{year}-{month:02d}")

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi):
        from family1.adapters import _mplastic_smoke as sm
        return sm.make_mobjects_smoke(root, d_lo, d_hi)


ADAPTER = MObjectsAdapter
