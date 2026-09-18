"""Sentinel-3 OLCI L2 WFR — 300 m ocean-colour granules, with a coastal flag.

PLAIN ENGLISH. OLCI is the ocean-colour camera on the two Sentinel-3
satellites: 21 visible and near-infrared bands at 300 m across a 1,270 km
swath, cut into three-minute granules. The "WFR" (water, full resolution)
Level-2 product is the water-leaving reflectance and chlorophyll form. This
store is the LIST of those granules — when, where, how large, which satellite,
which baseline — plus a SIXTH channel the other seven catalogues do not have:
how much of the granule sits in the coastal band. The pixels, about 55 TB a
year, stay in the Copernicus Data Space.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (anonymous):
  https://catalogue.dataspace.copernicus.eu/odata/v1/Products
      ?$filter=Collection/Name eq 'SENTINEL-3' and <productType eq
              'OL_2_WFR___'> and ContentDate/Start ge <from> ...
      &$expand=Attributes&$orderby=ContentDate/Start asc&$top=1000
The CDSE STAC endpoint serves no Sentinel collection (see `cat_s2`), so OData
is the catalogue.

**SINCE 2026 EVERY OVERPASS IS PUBLISHED TWICE — ONCE `NR`, ONCE `NT` — AND
THE STORE KEEPS ONE.** Measured by the `timeliness` attribute:

  day           total   NT (non-time-critical)   NR (near-real-time)
  2016-04-25      181                      181                     0
  2019-06-01      457                      457                     0
  2024-06-01      452                      452                     0
  2026-09-01      846                      423                   423

and per year, NR is 0 for 2016-2025 and 15,063 in 2026 against 116,274 NT. So
`productType eq 'OL_2_WFR___'` alone would have doubled the recent record.
The adapter lists BOTH and de-duplicates inside each window on the overpass
itself — the satellite plus the sensing start and stop, which the product name
carries — keeping the NT product where both exist and counting the choice
(`rows_nr_superseded_by_nt`, `rows_non_nt_kept`). Keeping the NR where NT does not
yet exist is what stops the newest weeks from silently vanishing: NT is
produced about a month later, so a build run today would otherwise end a month
short. `qc` records which of the two a row is, so a consumer can select.

MEASURED ARCHIVE SIZE, 2026-09-18 (`@odata.count` per year, NT):
  2016 81,069 · 2017 114,493 · 2018 133,144 · 2019 162,588 · 2020 164,114 ·
  2021 164,420 · 2022 164,079 · 2023 164,201 · 2024 164,901 · 2025 163,990 ·
  2026 116,274 (to 2026-09-18) -> **1,593,273 NT**, plus 15,063 NR-only
  candidates in 2026.

THE SIXTH CHANNEL: `coastal`. The note asks this store for the coastal band
only, and the ±100 km GSHHG band static that would define it **is not built
yet**. Cataloguing only what a not-yet-existing static would have selected is
not possible, and guessing the band is worse than measuring a coarse one, so:
every WFR granule is catalogued, and each row carries `coastal` — the
FRACTION of the granule's footprint bounding box that lies within about
100 km of a land/water boundary, computed from a 1-degree land/ocean mask
derived here from the repository's own family-7 static
`data/family7_sphere.json` (1,440 x 721 cells at 0.25 degrees, classes ocean /
land / ice sheet / inland water). A flag is then `coastal > 0`; the fraction
is kept instead of a bit because an OLCI granule is 1,270 km wide and "does it
touch a coast" is true for most of them, while "how much of it is coast" is
the number a consumer can threshold. THIS IS A COARSE STAND-IN and store.json
says so: when the GSHHG band static lands, this channel is recomputed from it
and the store is rebuilt. If the static is absent from the checkout the
channel is NaN for every row and the count `coastal_mask_absent` says so —
never a zero, which would read as "no coast here".

WHAT A ROW IS.
  time_s    `ContentDate/Start` — the start of the three-minute granule.
  lat, lon  the spherical mean of `GeoFootprint`'s ring.
  platform  platform_hash(the product Name, e.g.
            `S3B_OL_2_WFR____20240601T040257_..._MAR_O_NT_003.SEN3`).
  values    cloud   NaN — CDSE's OLCI product attributes carry no cloud
                    fraction (the per-pixel cloud flags are inside the
                    .SEN3), so the channel is a gap rather than a guess
            valid   NaN — none is published
            angle   NaN — no mean sun zenith is published
            log2_area
                    log2 of the footprint's area on the sphere in km² —
                    a 1,270 km swath granule is 1.5 x 10**6 km², which
                    OVERFLOWS float16 in km² and is why the channel is log2
            sensor  1 S3A/OLCI, 2 S3B/OLCI, 3 S3C/OLCI, 4 S3D/OLCI
            coastal the coastal fraction above, 0..1
  qc        `<baselineCollection>/<timeliness>`, e.g. `004/NT`.
"""
import datetime as dt
import json
import math
import os

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st

PRODUCTS = (("wfr", "OL_2_WFR___"),)
SATS = ("S3A", "S3B", "S3C", "S3D")
SENSOR_TABLE = {i: f"{s}/OLCI" for i, s in enumerate(SATS, start=1)}
BASELINES = ("001", "002", "003", "004", "005", "006")
TIMELINESS = ("NT", "NR", "ST")
QC_TABLE = {}
for _i, _b in enumerate(BASELINES):
    for _j, _t in enumerate(TIMELINESS):
        QC_TABLE[_i * 3 + _j] = f"{_b}/{_t}"
RECORD_FIRST = (2016, 4)
S3_PREFIX = "/eodata/Sentinel-3/OLCI/OL_2_WFR___/<YYYY>/<MM>/<DD>/<Name>"
SPHERE_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    "data", "family7_sphere.json")
COAST_KM = 100.0
CHANNELS = st.CAT_CHANNELS + (("coastal", "1", 0.0, 1.0),)


# ============================================================ coastal band ==
def coastal_mask_1deg(path=SPHERE_JSON, coast_km=COAST_KM):
    """A 360 x 180 boolean band: is this 1-degree cell near a coast?

    The input is the repository's family-7 `sphere` static — a 1,440 x 721
    grid of single digits, one per 0.25-degree cell, 0 ocean / 1 land /
    2 ice sheet / 3 inland water, with its own geometry in the same file. It
    is reduced to 1 degree by MAJORITY (land-or-ice against water), because a
    mean of class codes is not a class, and a cell is then `coastal` when any
    cell within `coast_km` of it — measured on the sphere, so the longitude
    reach widens towards the poles and wraps at the dateline — is of the other
    kind. The static's row 0 is the SOUTHERNMOST row (measured, see the
    comment in the body) and column 0 starts at -180; the mask this returns
    keeps that order, so `mask[j, i]` is the cell whose centre is at
    `(j - 89.5, i - 179.5)`.

    Returns `(mask, meta)`; `(None, meta)` when the static is not in the
    checkout, which makes the channel NaN rather than 0.
    """
    meta = {"source": path, "coast_km": coast_km}
    if not os.path.exists(path):
        meta["present"] = False
        return None, meta
    with open(path) as fh:
        d = json.load(fh)
    nx, ny = int(d["nx"]), int(d["ny"])
    packed = d["packed"]
    if len(packed) != nx * ny:
        raise st.Refusal(f"{path}: `packed` is {len(packed)} characters and "
                         f"nx*ny is {nx * ny}")
    a = np.frombuffer(packed.encode("ascii"), np.uint8) - ord("0")
    a = a.reshape(ny, nx)
    land = (a == 1) | (a == 2)
    # THE STATIC'S ROW ORDER IS SOUTH-FIRST, AND THAT WAS MEASURED RATHER
    # THAN ASSUMED (ml/CLAUDE.md §0.1). Reading it north-first put Portugal,
    # the Sahara, Siberia and Greenland in the ocean and produced a coastal
    # band that covered 15 % of the globe and none of the coasts — a mask
    # that looks entirely plausible and is wrong everywhere, which is the
    # failure the AMOC eval mask's own orientation assertion exists for. The
    # evidence: row 0 is all land/ice and row 720 is all ocean, which is
    # Antarctica and the Arctic Ocean and not the other way round, and eight
    # known points (Sahara, Amazon, Portugal, Greenland, Antarctica, Siberia,
    # mid-Pacific, mid-Atlantic) all resolve correctly south-first and four
    # of them wrongly north-first. `tests/test_family1_cat_stac.py` pins
    # three of those points.
    south, dlat = float(d["south"]), float(d["dlat"])
    west, dlon = float(d["west"]), float(d["dlon"])
    lat0 = np.array([south + (j + 0.5) * dlat for j in range(ny)])
    lon0 = np.array([west + (i + 0.5) * dlon for i in range(nx)])
    # -> 1 degree by majority. The 0.25-degree grid has 721 rows (a half cell
    # past each pole), so the reduction bins by the cell's own latitude rather
    # than by reshaping, which would be off by that half row.
    jy = np.clip(((lat0 + 90.0) // 1.0).astype(int), 0, 179)
    ix = np.clip(((lon0 + 180.0) // 1.0).astype(int), 0, 359)
    cnt = np.zeros((180, 360), np.int32)
    tot = np.zeros((180, 360), np.int32)
    for j in range(ny):
        np.add.at(cnt[jy[j]], ix, land[j].astype(np.int32))
        np.add.at(tot[jy[j]], ix, 1)
    land1 = (cnt * 2 > tot)
    # the band: any cell of the other kind within `coast_km`
    mask = np.zeros((180, 360), bool)
    dlat_km = 111.195
    for j in range(180):
        lat = j - 89.5
        rj = max(1, int(math.ceil(coast_km / dlat_km)))
        cosl = max(math.cos(math.radians(abs(lat))), 1e-3)
        ri = max(1, int(math.ceil(coast_km / (dlat_km * cosl))))
        ri = min(ri, 180)
        j0, j1 = max(0, j - rj), min(180, j + rj + 1)
        block = np.concatenate(
            [land1[j0:j1, :], land1[j0:j1, :], land1[j0:j1, :]], axis=1)
        for i in range(360):
            win = block[:, 360 + i - ri: 360 + i + ri + 1]
            if win.any() and not win.all():
                mask[j, i] = True
    meta.update(present=True, nx=nx, ny=ny, cells_land=int(land1.sum()),
                cells_coastal=int(mask.sum()),
                classes=[c.get("label") for c in (d.get("classes") or [])],
                static_source=d.get("_source"))
    return mask, meta


def coastal_fraction(mask, lats, lons):
    """The share of a footprint's bounding-box 1-degree cells in the band."""
    if mask is None:
        return float("nan")
    la = np.asarray(lats, np.float64)
    lo = np.asarray(lons, np.float64)
    j0 = int(np.floor(la.min() + 90.0))
    j1 = int(np.floor(la.max() + 90.0))
    j0, j1 = max(0, min(179, j0)), max(0, min(179, j1))
    # the box may wrap the dateline: take whichever span is the shorter
    i0 = int(np.floor(lo.min() + 180.0)) % 360
    i1 = int(np.floor(lo.max() + 180.0)) % 360
    if (lo.max() - lo.min()) > 180.0:
        cols = list(range(i1, 360)) + list(range(0, i0 + 1))
    else:
        cols = list(range(min(i0, i1), max(i0, i1) + 1))
    sub = mask[j0:j1 + 1][:, cols]
    return float(sub.mean()) if sub.size else float("nan")


def overpass_key(name):
    """The overpass a WFR product belongs to, from the product's own name.

    `S3B_OL_2_WFR____20240601T040257_20240601T040557_20240602T103132_0180_093_
    318_1800_MAR_O_NT_003.SEN3` -> `("S3B", "20240601T040257",
    "20240601T040557")`. The satellite plus the sensing start and stop is the
    overpass; everything after them (the creation time, the timeliness, the
    baseline) is the PUBLICATION, and the NR and NT products of one overpass
    differ only there.
    """
    p = str(name).split("_")
    if len(p) < 8:
        raise st.Refusal(f"an OLCI product name this adapter cannot read: "
                         f"{name!r}")
    return (p[0], p[7], p[8]) if len(p) > 8 else (p[0], p[7], "")


def timeliness_of(name, aa):
    t = str(aa.get("timeliness") or "").strip()
    return t or "?"


class OlciCatalogue(st.OdataCatalogue):
    store = "cat_olci"
    title = ("Sentinel-3 OLCI L2 WFR granule catalogue (300 m ocean colour, "
             "S3A + S3B, CDSE), one row per granule, with a coastal fraction")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The imagery they "
                 "point at is Copernicus Sentinel-3 OLCI L2 WFR — free and "
                 "open under the Copernicus data policy (Regulation (EU) "
                 "No 1159/2013), redistribution allowed with attribution"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("Contains modified Copernicus Sentinel-3 data; "
                        "product list from the Copernicus Data Space "
                        "Ecosystem, "
                        "https://catalogue.dataspace.copernicus.eu")}
    first_year = 2016
    footprint_km = 1234.0          # a 1,270 km swath x ~1,200 km of track
    scene_seconds = 180.0
    collection_name = "SENTINEL-3"
    PRODUCTS = PRODUCTS
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    channels = CHANNELS
    window_step = "day"           # 846 products a day: one page of 1000
    qc_policy = (
        "qc is `<baselineCollection>/<timeliness>`, e.g. 9 = `004/NT` and "
        "10 = `004/NR`, so a consumer can tell the consolidated product from "
        "the near-real-time one. A combination the table does not list is 255 "
        "and counted by name in `qc_unlisted`. Since 2026 both timeliness "
        "classes exist for the same overpass; the adapter keeps the NT "
        "product where both exist and the NR one where NT does not exist yet, "
        "counting each choice (`rows_nr_superseded_by_nt`, `rows_non_nt_kept`). "
        "`cloud`, `valid` and `angle` are NaN for every row: CDSE's OLCI "
        "product attributes publish none of them. `coastal` is a COARSE "
        "stand-in for the not-yet-built GSHHG +/-100 km band static (see the "
        "module docstring) and is NaN, never 0, when the family-7 sphere "
        "static is absent from the checkout. A value outside a channel's "
        "bounds is NaN and counted (`out_of_bounds`).")
    sources = (st.CDSE_ODATA,
               "https://catalogue.dataspace.copernicus.eu/stac/collections",
               st.CDSE_DOWNLOAD, "data/family7_sphere.json")
    verified = (
        "2026-09-18 from the sandbox, anonymously: productType OL_2_WFR___ "
        "holds 452 products on 2024-06-01, 457 on 2019-06-01, 181 on "
        "2016-04-25 and 846 on 2026-09-01 — and the 2026 day is 423 NT plus "
        "423 NR of the SAME overpasses, while every day before 2026 is NT "
        "only (NR is 0 for 2016-2025 by year and 15,063 in 2026); 1,593,273 "
        "NT products over 2016-2026; baselineCollection 004 and serial "
        "identifiers A and B on 2026-09-01; and the family-7 sphere static "
        "(1,440 x 721 cells, four classes) reduces to a 1-degree coastal band")
    notes = (
        "The pixels are never copied. EVERY WFR granule is catalogued, not "
        "only the coastal ones: the GSHHG +/-100 km band static the note's "
        "'coastal band' refers to is not built yet, so the store carries a "
        "sixth channel, `coastal`, computed from the repository's own "
        "family-7 1-degree land/ocean static, and a consumer selects on it. "
        "Recompute the channel and rebuild when the GSHHG static lands. "
        "NR and NT products of one overpass are ONE row. "
        f"The S3 path of a product is {S3_PREFIX}.")
    smoke_window = ("2026-09-01", "2026-09-02")
    smoke_probe_month = "2026-09"

    def __init__(self):
        super().__init__()
        self._coast = None
        self._coast_meta = None

    def record_first(self):
        return RECORD_FIRST

    def coast(self):
        if self._coast_meta is None:
            self._coast, self._coast_meta = coastal_mask_1deg()
        return self._coast

    def sensor_text(self, name, aa):
        s = str(aa.get("platformSerialIdentifier") or "").strip().upper()
        sat = ("S3" + s) if s in ("A", "B", "C", "D") else str(name)[:3].upper()
        return f"{sat}/OLCI"

    def qc_text(self, name, prod, aa):
        return (f"{str(aa.get('baselineCollection') or '').strip()}/"
                f"{timeliness_of(name, aa)}")

    def values(self, name, prod, aa, area, sensor, counts):
        # the coastal fraction wants the RING, which `product` has and this
        # hook does not, so it is filled in there; NaN here is the default
        return [float("nan"), float("nan"), float("nan"),
                st.area_channel(area, counts), float(sensor), float("nan")]

    # -- one product, then the overpass de-duplication ---------------------
    def product(self, p, prod, counts, what):
        sc = super().product(p, prod, counts, what)
        rings = st.ring_from_geojson(p.get("GeoFootprint"))
        mask = self.coast()
        if mask is None:
            counts["coastal_mask_absent"] = \
                counts.get("coastal_mask_absent", 0) + 1
            sc.values[5] = float("nan")
        else:
            lats = [x for r in rings for x in r[0]]
            lons = [x for r in rings for x in r[1]]
            sc.values[5] = coastal_fraction(mask, lats, lons)
        return sc

    def scenes(self, ctx, window):
        """The window's products, with one row per OVERPASS.

        Since 2026 CDSE publishes the near-real-time and the consolidated
        product of the same overpass, both with the same `ContentDate/Start`,
        so they always land in the same window and the choice is local and
        deterministic: keep NT, fall back to NR, count both outcomes.
        """
        out, counts = super().scenes(ctx, window)
        best = {}
        order = []
        for sc in out:
            k = overpass_key(sc.stac_id)
            tl = self._timeliness_of_scene(sc)
            if k not in best:
                best[k] = (tl, sc)
                order.append(k)
                continue
            prev_tl, prev = best[k]
            if prev_tl != "NT" and tl == "NT":
                best[k] = (tl, sc)
                counts["rows_nr_superseded_by_nt"] = \
                    counts.get("rows_nr_superseded_by_nt", 0) + 1
            else:
                counts["rows_duplicate_overpass_dropped"] = \
                    counts.get("rows_duplicate_overpass_dropped", 0) + 1
        kept = [best[k][1] for k in order]
        # NOT "NR only": an overpass whose only publication is NR, ST or an
        # unknown timeliness is kept as it is, and this counts every such
        # row. Naming it `rows_nr_only` made the smoke's ST product look like
        # a second NR one.
        counts["rows_non_nt_kept"] = sum(
            1 for k in order if best[k][0] != "NT")
        counts["overpasses"] = len(order)
        return kept, counts

    @staticmethod
    def _timeliness_of_scene(sc):
        """`NT` / `NR` out of the product name's own field."""
        p = str(sc.stac_id).split("_")
        for tok in p:
            if tok in ("NT", "NR", "ST"):
                return tok
        return "?"

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        y, m = ctx.d_lo.year, ctx.d_lo.month
        if (y, m) < RECORD_FIRST:
            y, m = RECORD_FIRST
        a = dt.datetime(y, m, 1)
        b = dt.datetime(y + (m == 12), m % 12 + 1, 1)
        pt = PRODUCTS[0][1]
        total = st.odata_count(fet, st.CDSE_ODATA, self.filt(a, b, pt))
        per_tl = {}
        for tl in TIMELINESS:
            f = (self.filt(a, b, pt) +
                 " and Attributes/OData.CSC.StringAttribute/any(t:t/Name eq "
                 f"'timeliness' and t/OData.CSC.StringAttribute/Value eq "
                 f"'{tl}')")
            per_tl[tl] = st.odata_count(fet, st.CDSE_ODATA, f)
        if not total:
            raise st.Refusal(
                f"{self.store}: CDSE counts ZERO OL_2_WFR___ products in "
                f"{y:04d}-{m:02d} — a broken filter, not a measurement "
                f"(the 2026-09-14 rule).")
        mask = self.coast()
        one = None
        d, _ = fet.get(st.odata_url(
            st.CDSE_ODATA, filter=self.filt(a, b, pt), top=1,
            expand=self.expand, orderby="ContentDate/Start asc"))
        for p in (d.get("value") or []):
            c2 = {}
            sc = self.product(p, PRODUCTS[0], c2, "index")
            one = {"name": sc.stac_id, "time_s": sc.t, "lat": sc.lat,
                   "lon": sc.lon,
                   "values": dict(zip(self.channel_names,
                                      [None if not np.isfinite(x) else x
                                       for x in sc.values])),
                   "qc": sc.qc, "base_url": sc.base_url,
                   "asset_template": sc.asset_set,
                   "overpass_key": list(overpass_key(sc.stac_id)),
                   "attributes": st.odata_attrs(p), "counts": c2}
            break
        return {
            "dataset": "Sentinel-3 OLCI L2 WFR (CDSE OData)",
            "url": st.CDSE_ODATA,
            "collection": self.collection_name,
            "product_types": [pt],
            "top_max": st.ODATA_TOP, "skip_max": st.ODATA_SKIP_MAX,
            "window_step": self.window_step,
            "years": [int(v) for v in ctx.years],
            "first_month": f"{y:04d}-{m:02d}",
            "products_first_month": total,
            "products_first_month_by_timeliness": per_tl,
            "timeliness_note": (
                "an overpass published both NR and NT is ONE row: the NT "
                "product is kept and the NR one counted "
                "(`rows_nr_superseded_by_nt`); an overpass with no NT "
                "product is kept as it is (`rows_non_nt_kept`) so the newest "
                "weeks, "
                "for which NT does not exist yet, are not lost"),
            "coastal": self._coast_meta,
            "coastal_note": (
                "`coastal` is the share of the granule's bounding-box "
                "1-degree cells that lie within about 100 km of a land/water "
                "boundary, from data/family7_sphere.json reduced to 1 degree "
                "by majority. It is a COARSE STAND-IN for the GSHHG "
                "+/-100 km band static, which is not built yet; recompute and "
                "rebuild when it lands. NaN (never 0) when the static is "
                "absent."),
            "s3_path_pattern": S3_PREFIX,
            "sensor_table": {str(k): v for k, v in SENSOR_TABLE.items()},
            "qc_table": {str(k): v for k, v in QC_TABLE.items()},
            "first_record": one,
        }

    def extra_meta(self, ctx, dest, N, values):
        meta = super().extra_meta(ctx, dest, N, values)
        self.coast()                 # so the mask's own record is in store.json
        meta["coastal"] = self._coast_meta
        meta["coastal_channel_note"] = (
            "channel 5 `coastal` is a coarse stand-in for the GSHHG "
            "+/-100 km coastal band static, which is not built yet")
        return meta

    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        from family1.adapters import _cat_smoke as cs
        return cs.odata_sources(self, root, d_lo, d_hi, cs.OLCI_PRODUCTS)


ADAPTER = OlciCatalogue
