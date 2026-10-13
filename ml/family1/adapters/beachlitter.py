"""Litter on beaches and shorelines — one row per SURVEY (family 1.3, E-092).

PLAIN ENGLISH. Plastic that floats eventually strands. Volunteers and
monitoring programmes walk a fixed stretch of beach and count every piece of
litter they find, by type; others pick plastic pellets ("nurdles") off the
tide line for ten minutes; scientists dig cores of beach sand and count the
microplastic pieces in them. This store is those surveys, from four producers,
one row per survey, so that the floating-plastic read-out (`mplastic`) has a
shoreline counterpart. Plan `ml/plans/E092_family13_plastic.md`, decision D13
(the `beachlitter` bullet) and §6 (the falsifiers this adapter enforces);
measurement `ml/family1/probes/beachlitter_measure.json`.

SOURCES, IN PRECEDENCE ORDER (plan D13), all keyless, measured 2026-10-13:

 1. NOAA NCEI Global Marine Microplastics (CC0 1.0, accession 0306475), the
    harmonised Hub CSV (`NCEI_CSV`, the same file `mplastic` reads), rows with
    Marine Setting "Nurdle Patrol": citizen-science counts of plastic pellets
    picked by hand (University of Texas Marine Science Institute; 10,615 rows,
    2018-11 -> 2026-04). The harmonised table carries only the producer's
    STANDARDISED rate; the raw count, minutes and volunteers come from NCEI
    accession 0259486's own `NurdlePatrolDatabase_*.xlsx` (found through the
    landing page -> archive -> newest version -> data/0-data/ listing). The two
    copies renumber their ids, so they are JOINED ON CONTENT (date, position,
    standardised value, volunteers, minutes); 31 xlsx rows carry a POSITIVE
    longitude where the harmonised table has the (correct, Gulf of Mexico)
    negative one, and 5 differ in the fourth decimal of latitude — matched with
    the sign/rounding tolerance in `nurdle_join` and counted, the harmonised
    table's position kept. The rate is RECOMPUTED as pellets per 10
    PERSON-minutes, count * 10 / (minutes * volunteers) — the accession's own
    data dictionary: "the number of nurdles collected by one person in a
    10-minute period". The published value agrees (half-up rounding) on 9,183
    rows; it equals floor or ceil of the exact rate on 10,486; the rest (e.g.
    47 pellets in 5 minutes published as 47) carry qc bit 6.
 2. The same NCEI table, Marine Setting "Beach": microplastic pieces per m3 of
    beach SAND (893 cores and scoops from 13 studies) — a different quantity
    from litter per 100 m, in its own channel.
 3. EMODnet Chemistry European beach litter, aggregated dataset v2025
    (`EMODNET_ZIP`), ONLY the surveys whose per-survey data policy is "CB"
    (CC BY 4.0). The "LS" surveys (SeaDataNet licence, whose clause 5 forbids
    passing data to third parties without the data centre's consent) and the
    "UN" surveys are counted as `excluded_licence` and NOT read further — they
    belong to a private track that is not built here. One line per survey x
    litter type with the count; material per type from the COMMITTED table
    `ml/family1/probes/beachlitter_emodnet_materials.json` (no machine-readable
    map exists; the table cites the reference lists and gives the rule for
    every code). A type the list does not decide is AMBIGUOUS: counted in its
    own channel, never as plastic.
 4. NOAA Marine Debris Monitoring and Assessment Project (MDMAP), all three
    protocols (MDMAP 2.0, standing stock, accumulation), from the portal's
    site list (`MDMAP_SITES`) and one export zip per site (`MDMAP_EXPORT`;
    there is no bulk export — measured: the survey endpoints answer 400/500).
    A NOAA programme on a NOAA portal; no licence file is posted, so it is
    taken as US-government public data and recorded as UNCONFIRMED
    (`licence["redistribution_confirmed"] = False`, plan D13).

Natural Earth 1:10m (public domain, release v5.1.2, pinned): the coastline, the
minor-islands coastline, the lakes and the river centre-lines, used ONLY to
flag freshwater shores and positions far from any mapped shore (`Shores`).

A later source's LITTER-SURVEY row is dropped when an earlier source holds one
on the same UTC day within 0.01 deg (only EMODnet vs MDMAP can collide;
measured 0); every drop is counted per pair in `counts["dedup"]`. Pellet
counts, sand cores and litter surveys measure different things and are never
copies of one another (162 Nurdle Patrol / MDMAP same-day pairs exist and are
both kept).

WHAT A ROW IS.
  time_s    int32 seconds since 1982-01-01 UTC: every source gives a DATE only
            (NCEI writes local midnight as 05:00/06:00 UTC; MDMAP's search
            times are local with no zone), placed at 12:00 UTC with qc bit 4.
  lat, lon  NCEI's position; EMODnet's beach location; MDMAP's water's-edge
            midpoint (left/right ends; one end if the other is missing; the
            back-barrier midpoint if both are).
  platform  platform_hash("beachlitter|<source>|<beach id>"): EMODnet's
            location_code, MDMAP's shoreline_site_id, the Nurdle Patrol's
            position rounded to 0.001 deg (it has no site id), NCEI sand
            study + beach name.
  values    C = 13, in this order (NaN = not measured by this protocol):
     log_items_100m    log10(1 + litter items per 100 m of shoreline, whole
                       beach width) — EMODnet count / length * 100; MDMAP 2.0 and
                       standing stock sum(counts) / sum(transect widths along
                       the shore) * 100; accumulation count / site length * 100
     log_plastic_100m  the same, plastic items only
     log_ambig_100m    the same, items whose material the reference list does
                       not decide (EMODnet; MDMAP 0 — its categories decide)
     plastic_frac      plastic / all items (0..1; NaN when nothing was found);
                       a LOWER bound where ambiguous items exist
     log_items_m2      log10(1 + items per m2), where the surveyed area is
                       known (EMODnet with a width; MDMAP with a valid beach
                       width)
     accum_days        days since the previous accumulation survey of the site
                       (MDMAP accumulation; that survey cleared it)
     survey_len_m      metres of shoreline surveyed
     log_nurdles_10pmin  log10(1 + pellets per 10 person-minutes), recomputed
     log_nurdle_count  log10(1 + pellets counted)
     collect_min       minutes searched
     volunteers        people searching
     effort_pmin       person-minutes searched (minutes * volunteers)
     new_tide_line     1 = searched the newest tide line, 0 = elsewhere
     log_n_m3_sand     log10(1 + microplastic pieces per m3 of beach sand)
            WHY LOG10(1 + x): the store's values are float16 (max 65,504); a
            beach reaches 7.5e4 items per 100 m and a sand core 3.5e5 pieces
            per m3, and one patrol counted 500,000 pellets. A measured zero
            stays exactly 0 (plan D10, the `mplastic` precedent).
  qc        bits 0-3: PROTOCOL (`PROTOCOLS`); bit 4 (16): time of day not
            given, placed at 12:00 UTC (every row); bit 5 (32): FRESHWATER
            shore — farther than FRESH_SEA_KM from the sea coast and nearer a
            mapped lake or river than to it (Great Lakes, rivers; a coast-
            distance test, not a list of states); bit 6 (64): the value rests
            on a caveat — litter protocols: survey length not given, the
            protocol's default assumed (`DEFAULT_LENGTH_M`); Nurdle Patrol:
            NCEI's published standardised rate is neither floor nor ceil of the
            recomputed one; bit 7 (128): POSITION DOUBT — farther than FAR_KM
            from every shore Natural Earth maps, or published to fewer than 3
            decimals (NCEI), or survey start > 5 km from the beach location
            or a location code with two coordinates (EMODnet), or the site's
            two water's-edge ends > 5 km apart (MDMAP).
A row with no density value at all is dropped (`rows_no_value`). Values
outside the channel bounds are NaN and counted (`out_of_bounds`).
"""
import csv
import datetime as dt
import io
import json
import math
import os
import re
import zipfile

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm
from family1.adapters import mplastic as mp

# ================================================================ sources ==
NCEI_CSV = mp.NCEI_CSV
NCEI_LANDING = mp.NCEI_LANDING
NCEI_ARCHIVE = mp.NCEI_ARCHIVE
NURDLE_ACC = "0259486"
NURDLE_FILE = re.compile(r"^NurdlePatrolDatabase.*\.xlsx$")
EMODNET_ZIP = ("https://nodc.ogs.it/repository/aggregated-datasets/litter/"
               "EMODnet_beachLitter_European_2025_unrestricted.zip")
EMODNET_META = "beach_erddap_surveys_metadata.csv"
EMODNET_LITTER = "beach_erddap_surveys_litter.csv"
MDMAP_SITES = "https://portal.diver.orr.noaa.gov/pentaho/mdmap/sites"
MDMAP_EXPORT = "https://portal.diver.orr.noaa.gov/pentaho/mdmap/sites/{id}/export"
NE_TAG = "v5.1.2"
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
          f"{NE_TAG}/geojson/{{name}}.geojson")
NE_LAYERS = {"sea": ("ne_10m_coastline", "ne_10m_minor_islands_coastline"),
             "lake": ("ne_10m_lakes",),
             "river": ("ne_10m_rivers_lake_centerlines",)}
HERE = os.path.dirname(os.path.abspath(__file__))
MATERIALS_JSON = os.path.join(os.path.dirname(HERE), "probes",
                              "beachlitter_emodnet_materials.json")

#: source tags, in PRECEDENCE order (plan D13)
SOURCES = ("nurdle", "ncei_sand", "emodnet", "mdmap")
#: the sources whose rows are litter surveys (and so can copy one another)
LITTER = {"emodnet", "mdmap"}
DEDUP_DEG = 0.01

# ================================================================ channels ==
def _lg(x):
    return float(np.log10(1.0 + x))


CHANNELS = (
    ("log_items_100m", "log10(1 + litter items per 100 m of shoreline)",
     0.0, _lg(1e6)),
    ("log_plastic_100m", "log10(1 + plastic items per 100 m of shoreline)",
     0.0, _lg(1e6)),
    ("log_ambig_100m", "log10(1 + items of undecided material per 100 m)",
     0.0, _lg(1e6)),
    ("plastic_frac", "plastic items / all items", 0.0, 1.0),
    ("log_items_m2", "log10(1 + litter items m-2)", 0.0, _lg(1e4)),
    ("accum_days", "days since the previous clearance", 1.0, 3650.0),
    ("survey_len_m", "m of shoreline surveyed", 1.0, 20000.0),
    ("log_nurdles_10pmin", "log10(1 + pellets per 10 person-minutes)",
     0.0, _lg(1e7)),
    ("log_nurdle_count", "log10(1 + pellets counted)", 0.0, _lg(1e7)),
    ("collect_min", "minutes searched", 1.0, 1440.0),
    ("volunteers", "people searching", 1.0, 1000.0),
    ("effort_pmin", "person-minutes searched", 1.0, 60000.0),
    ("new_tide_line", "1 = the newest tide line was searched", 0.0, 1.0),
    ("log_n_m3_sand", "log10(1 + microplastic pieces m-3 of beach sand)",
     0.0, _lg(1e9)),
)
NAMES = [c[0] for c in CHANNELS]
COL = {n: i for i, n in enumerate(NAMES)}
#: a row with none of these finite carries no measurement and is dropped
DENSITY = ("log_items_100m", "log_items_m2", "log_nurdles_10pmin",
           "log_n_m3_sand")

PROTOCOLS = {1: "EMODnet, OSPAR 100 m list", 2: "EMODnet, OSPAR 1 km list "
             "(items > 50 cm)", 3: "EMODnet, MSFD TSG-ML master list",
             4: "EMODnet, Joint List (J codes)", 5: "EMODnet, Joint List and "
             "TSG-ML in one survey", 6: "EMODnet, UNEP/IOC list (incl. "
             "MARLIN)", 7: "EMODnet, Italian national list",
             8: "MDMAP 2.0", 9: "MDMAP standing stock",
             10: "MDMAP accumulation", 11: "Nurdle Patrol (pellets picked by "
             "hand)", 12: "NCEI beach-sand microplastics sample"}
#: EMODnet reference list(s) of a survey -> protocol code
LIST_PROTOCOL = {("OSPAR",): 1, ("OSPAR_1000",): 2, ("TSG-ML",): 3,
                 ("JLIST",): 4, ("JLIST", "TSG-ML"): 5, ("UNEP",): 6,
                 ("UNEP_MARLIN",): 6, ("ITA",): 7}
#: the survey length each protocol prescribes, used (and flagged, qc bit 6)
#: where EMODnet gives none: OSPAR 2010 (a 100 m stretch; the 1 km survey),
#: MSFD TSG-ML 2013 and the Joint List (100 m), UNEP/IOC 2009 (100 m sampling
#: unit), the Italian MSFD programme (100 m)
DEFAULT_LENGTH_M = {1: 100.0, 2: 1000.0, 3: 100.0, 4: 100.0, 5: 100.0,
                    6: 100.0, 7: 100.0}
MDMAP_PROTOCOL = {"MDMAP 2.0 Protocol": 8,
                  "MDMAP Standing Stock Protocol": 9,
                  "MDMAP Accumulation Protocol": 10}
QC_NO_TIME = 16
QC_FRESH = 32
QC_CAVEAT = 64
QC_POSITION = 128

# ------------------------------------------------------------------- NCEI --
NCEI_COLUMNS = ("Unique ID", "Sample Date", "Latitude (degree)",
                "Longitude (degree)", "Marine Setting", "Beach Location",
                "State", "Country", "Volunteers Number",
                "Collecting Time (min)", "Standardized Nurdle  Amount",
                "Microplastics Measurement", "Unit", "Short Reference",
                "NCEI Accession Number")
NCEI_NURDLE = "Nurdle Patrol"
NCEI_SAND = "Beach"
NURDLE_UNIT = "pieces/10 mins"
SAND_UNIT = "pieces/m3"
NURDLE_HEADER = ("ID", "Patrol Date", "Submitted Date", "Latitude",
                 "Longitude", "Country", "State", "Location", "New Tide Line",
                 "Volunteers Number", "Collecting Time", "Nurdle Amount",
                 "Standardized Nurdle  Amount", "Notes")
#: the accession's own definition of the standardised value (asserted)
NURDLE_DEFINITION = re.compile(r"by one person in a 10-minute period")
#: plan §6: the recomputed rate must equal the published one on at least this
#: many rows (the count measured 2026-10-13 under the narrower per-10-minute
#: reading; under the documented per-person rule, half-up, it is 9,183)
NURDLE_AGREE_MIN = 7275
#: content-join tolerance where the two copies' positions differ
NURDLE_JOIN_DEG = 2e-4
#: NCEI positions published to fewer decimals than this are flagged (bit 7)
MIN_DECIMALS = 3

# ---------------------------------------------------------------- EMODnet --
EMODNET_META_COLUMNS = ("country_code", "location_code", "location_name",
                        "location_latitude", "location_longitude",
                        "survey_code", "date", "survey_start_latitude",
                        "survey_start_longitude", "survey_length",
                        "survey_width", "survey_type", "data policy")
EMODNET_LITTER_COLUMNS = ("survey_code", "data policy",
                          "litter_reference_list", "type_code", "type_name",
                          "abundance")
POLICIES = {"CB": "CC BY 4.0 — kept",
            "UN": "'freely available ... usage acknowledgement may be "
                  "required' — no licence named; private track",
            "LS": "SeaDataNet licence (no third-party redistribution without "
                  "the data centre's consent); private track"}
PUBLIC_POLICY = "CB"
SURVEY_TYPES = {"MSFD_monitoring", "Monitoring", "Research", "Cleaning"}
#: a clean-up is not a monitoring protocol: its counts are not complete by
#: design. None is CB (measured); any that appear are dropped and counted.
DROP_SURVEY_TYPES = {"Cleaning"}
START_DOUBT_KM = 5.0

# ------------------------------------------------------------------ MDMAP --
MDMAP_MATERIALS = ("total_plastic_items", "total_metal_items",
                   "total_glass_items", "total_rubber_items",
                   "total_processed-lumber_items", "total_cloth-fabric_items",
                   "total_other_items")
MDMAP_REQUIRED = ("survey_protocol", "shoreline_site_id", "survey_date",
                  "survey_id", "transect_id", "beach_width_at_transect",
                  "transect_width", "site_length", "total_debris_items",
                  "total_plastic_items", "site_waters_edge_left_lat",
                  "site_waters_edge_left_lon", "site_waters_edge_right_lat",
                  "site_waters_edge_right_lon", "site_back_barrier_left_lat",
                  "site_back_barrier_left_lon", "site_back_barrier_right_lat",
                  "site_back_barrier_right_lon")
#: a beach width written as this is a sentinel, not a width (measured: 8 rows)
MDMAP_WIDTH_SENTINEL = 9999.0
MDMAP_END_DOUBT_KM = 5.0
MDMAP_WORKERS = 8
MDMAP_ATTEMPTS = 6

# ------------------------------------------------------------- the shores --
#: freshwater: farther than this from the sea coast AND nearer a mapped lake
#: or river than to the sea. Measured 2026-10-13: EMODnet's (marine) beaches
#: lie <= 26 km from the coast, every one of them nearer the sea than a lake;
#: the Great Lakes shores are 300-1,000 km from it.
FRESH_SEA_KM = 10.0
#: position doubt: farther than this from every shore Natural Earth maps
FAR_KM = 25.0
DENSIFY_DEG = 0.004
EARTH_KM = 6371.0


class FormatError(ValueError):
    """A source that is not the layout verified above."""


def refuse(msg):
    raise FormatError(f"REFUSING beachlitter: {msg}")


def _bump(d, k, n=1):
    d[k] = d.get(k, 0) + n


# ================================================================== time ==
def t_noon(y, m, d):
    return int(cm.days_from_civil(y, m, d)) * 86400 + 43200


def day_of(t):
    d = dt.date(1982, 1, 1) + dt.timedelta(days=int(t // 86400))
    return d.year, d.month, d.day


def iso_date(s, what):
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(s).strip())
    if not m:
        refuse(f"{what}: date {s!r} is not YYYY-MM-DD")
    y, mo, d = (int(x) for x in m.groups())
    if not cm.valid_date(y, mo, d):
        refuse(f"{what}: {s!r} is not a calendar date")
    return y, mo, d


def ncei_date(s):
    """'M/D/YYYY h:mm:ss AM' -> (y, m, d); the time is a midnight artefact."""
    y, m, d, _tod = mp.ncei_date(s)
    return y, m, d


def decimals(s):
    s = str(s).strip()
    return len(s.split(".", 1)[1]) if "." in s else 0


def num(s):
    """A number, or None for an empty / NULL / null cell."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return None if (isinstance(s, float) and math.isnan(s)) else float(s)
    s = str(s).strip()
    if s in ("", "NULL", "null", "NaN", "nan"):
        return None
    return float(s)


def hav_km(la1, lo1, la2, lo2):
    a, b, c, d = map(math.radians, (la1, lo1, la2, lo2))
    h = (math.sin((c - a) / 2) ** 2 +
         math.cos(a) * math.cos(c) * math.sin((d - b) / 2) ** 2)
    return 2 * EARTH_KM * math.asin(math.sqrt(min(1.0, h)))


def _log(x):
    return float(np.log10(1.0 + x))


# ============================================================ candidates ==
def cand(src, key, t, qc, lat, lon, plat, proto, **v):
    return {"src": src, "key": key, "t": int(t), "qc": int(qc) | int(proto)
            | QC_NO_TIME, "lat": float(lat), "lon": float(lon), "plat": plat,
            "day": day_of(t), "v": v}


def litter_values(total, plastic, ambig, length_m, area_m2=None):
    """The litter channels from counts, a length and (optionally) an area."""
    v = {"plastic_frac": (plastic / total) if total > 0 else None}
    if length_m:
        k = 100.0 / length_m
        v.update(log_items_100m=_log(total * k),
                 log_plastic_100m=_log(plastic * k),
                 log_ambig_100m=None if ambig is None else _log(ambig * k))
    if area_m2:
        v["log_items_m2"] = _log(total / area_m2)
    return v


# ============================================== the shores (Natural Earth) ==
def _rings(gj):
    out = []
    for f in gj.get("features") or []:
        g = f.get("geometry")
        if not g:
            continue
        t, cs = g["type"], g["coordinates"]
        if t == "LineString":
            out.append(cs)
        elif t in ("MultiLineString", "Polygon"):
            out.extend(cs)
        elif t == "MultiPolygon":
            for p in cs:
                out.extend(p)
        else:
            refuse(f"Natural Earth geometry type {t!r}")
    return out


def _densify(lines, step=DENSIFY_DEG):
    pts = []
    for ln in lines:
        a = np.asarray(ln, float)[:, :2]
        if len(a) < 2:
            pts.append(a)
            continue
        dd = a[1:] - a[:-1]
        d = np.hypot(dd[:, 0], dd[:, 1])
        n = np.where(np.abs(dd[:, 0]) > 180, 1,
                     np.maximum(1, np.ceil(d / step))).astype(np.int64)
        seg = np.repeat(np.arange(len(a) - 1), n)
        start = np.repeat(np.cumsum(n) - n, n)
        frac = (np.arange(int(n.sum())) - start) / np.repeat(n, n)
        pts.append(a[seg] + dd[seg] * frac[:, None])
        pts.append(a[-1:])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def _xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                 np.sin(la)]


class Shores:
    """Distance (km) from a point to the sea coast, a lake shore, a river."""

    def __init__(self, layers):
        from scipy.spatial import cKDTree
        self.trees = {}
        self.vertices = {}
        for kind, gjs in layers.items():
            p = _densify([r for gj in gjs for r in _rings(gj)])
            if not len(p):
                refuse(f"Natural Earth {kind} layer holds no geometry")
            self.trees[kind] = cKDTree(_xyz(p[:, 0], p[:, 1]))
            self.vertices[kind] = int(len(p))

    def km(self, kind, lat, lon):
        d, _ = self.trees[kind].query(_xyz(np.asarray(lon, float),
                                           np.asarray(lat, float)))
        return 2 * EARTH_KM * np.arcsin(np.minimum(np.asarray(d) / 2, 1.0))

    def flags(self, lat, lon):
        """(freshwater bool array, far-from-every-shore bool array, dists)."""
        sea = self.km("sea", lat, lon)
        lake = self.km("lake", lat, lon)
        river = self.km("river", lat, lon)
        inland = np.minimum(lake, river)
        fresh = (sea > FRESH_SEA_KM) & (inland < sea)
        far = np.minimum(sea, inland) > FAR_KM
        return fresh, far, (sea, lake, river)


# ======================================================= 1: Nurdle Patrol ==
def ncei_read(text, counts):
    rd = csv.DictReader(io.StringIO(text))
    miss = [c for c in NCEI_COLUMNS if c not in (rd.fieldnames or [])]
    if miss:
        refuse(f"the NCEI CSV lacks the column(s) {miss}")
    allr = list(rd)
    if not allr:
        refuse("the NCEI CSV holds no row")
    settings = {}
    for x in allr:
        _bump(settings, x["Marine Setting"])
    counts["ncei_rows_total"] = len(allr)
    counts["ncei_marine_setting"] = settings
    nurdle = [x for x in allr if x["Marine Setting"] == NCEI_NURDLE]
    sand = [x for x in allr if x["Marine Setting"] == NCEI_SAND]
    if not nurdle or not sand:
        refuse(f"the NCEI CSV has {len(nurdle)} Nurdle Patrol and {len(sand)} "
               f"Beach rows — both are needed")
    for rows, unit, what in ((nurdle, NURDLE_UNIT, NCEI_NURDLE),
                             (sand, SAND_UNIT, NCEI_SAND)):
        u = sorted({x["Unit"] for x in rows})
        if u != [unit]:
            refuse(f"NCEI {what} rows carry units {u}, not only {unit!r}")
    return nurdle, sand


def nurdle_xlsx(path):
    """The accession's Data sheet -> [row dict], its definition asserted."""
    import openpyxl
    import warnings
    with open(path, "rb") as fh:
        data = io.BytesIO(fh.read())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(data, read_only=True, data_only=True)
    if "Data" not in wb.sheetnames or "Data_Dictionary" not in wb.sheetnames:
        refuse(f"the Nurdle Patrol workbook has sheets {wb.sheetnames}")
    defs = {str(r[0]).strip(): str(r[1] or "") for r in
            wb["Data_Dictionary"].iter_rows(values_only=True) if r and r[0]}
    std = [v for k, v in defs.items() if k.startswith("Standardized Nurdle")]
    if not std or not NURDLE_DEFINITION.search(std[0]):
        refuse(f"the Nurdle Patrol data dictionary no longer defines the "
               f"standardised value per person per 10 minutes: {std}")
    it = wb["Data"].iter_rows(values_only=True)
    hdr = tuple(" ".join(str(h).split()) if h is not None else ""
                for h in next(it))
    if hdr[:len(NURDLE_HEADER)] != tuple(" ".join(h.split())
                                         for h in NURDLE_HEADER):
        refuse(f"the Nurdle Patrol Data sheet header is {hdr}")
    out = []
    for r in it:
        if r[0] is None:
            continue
        x = dict(zip(NURDLE_HEADER, r))
        day = x["Patrol Date"]
        if not isinstance(day, (dt.datetime, dt.date)):
            refuse(f"Nurdle Patrol id {x['ID']}: Patrol Date {day!r}")
        for k in ("Latitude", "Longitude", "Volunteers Number",
                  "Collecting Time", "Nurdle Amount",
                  "Standardized Nurdle  Amount"):
            if not isinstance(x[k], (int, float)):
                refuse(f"Nurdle Patrol id {x['ID']}: {k} = {x[k]!r}")
        if x["New Tide Line"] not in ("Yes", "No"):
            refuse(f"Nurdle Patrol id {x['ID']}: New Tide Line = "
                   f"{x['New Tide Line']!r}")
        x["_day"] = (day.year, day.month, day.day)
        out.append(x)
    if not out:
        refuse("the Nurdle Patrol workbook holds no row")
    return out


def nurdle_join(hub, xl, counts):
    """Pair every harmonised row with ONE accession row, on content."""
    by = {}
    for i, x in enumerate(xl):
        k = (x["_day"], int(x["Standardized Nurdle  Amount"]),
             int(x["Volunteers Number"]), int(x["Collecting Time"]))
        by.setdefault(k, []).append(i)
    used = set()
    pairs = []
    how = {"exact": 0, "longitude_sign_differs": 0, "position_rounding": 0,
           "unmatched": 0}
    for h in hub:
        y, m, d = ncei_date(h["Sample Date"])
        la, lo = float(h["Latitude (degree)"]), float(h["Longitude (degree)"])
        k = ((y, m, d), int(float(h["Standardized Nurdle  Amount"])),
             int(float(h["Volunteers Number"])),
             int(float(h["Collecting Time (min)"])))
        cands = [i for i in by.get(k, ()) if i not in used]
        pick, kind = None, None
        for i in cands:
            x = xl[i]
            if round(x["Latitude"], 4) == round(la, 4) and \
                    round(x["Longitude"], 4) == round(lo, 4):
                pick, kind = i, "exact"
                break
        if pick is None:
            for i in cands:
                x = xl[i]
                if abs(x["Latitude"] - la) <= NURDLE_JOIN_DEG and \
                        abs(abs(x["Longitude"]) - abs(lo)) <= NURDLE_JOIN_DEG:
                    pick = i
                    kind = ("longitude_sign_differs"
                            if (x["Longitude"] < 0) != (lo < 0) else
                            "position_rounding")
                    break
        if pick is None:
            how["unmatched"] += 1
            continue
        used.add(pick)
        how[kind] += 1
        pairs.append((h, xl[pick]))
    counts["nurdle_join"] = how
    counts["nurdle_accession_rows"] = len(xl)
    counts["nurdle_accession_rows_unjoined"] = len(xl) - len(used)
    return pairs


def nurdle_rows(pairs, counts):
    rows = []
    agree = {"half_up": 0, "floor_or_ceil": 0, "neither": 0, "zero_minutes": 0}
    for h, x in pairs:
        n = float(x["Nurdle Amount"])
        mins = float(x["Collecting Time"])
        vol = float(x["Volunteers Number"])
        pub = float(h["Standardized Nurdle  Amount"])
        if mins <= 0 or vol <= 0:
            agree["zero_minutes"] += 1
            continue
        rate = n * 10.0 / (mins * vol)
        caveat = 0
        if pub == math.floor(rate + 0.5):
            agree["half_up"] += 1
        elif pub in (math.floor(rate), math.ceil(rate)):
            agree["floor_or_ceil"] += 1
        else:
            agree["neither"] += 1
            caveat = QC_CAVEAT
        y, m, d = ncei_date(h["Sample Date"])
        la, lo = float(h["Latitude (degree)"]), float(h["Longitude (degree)"])
        doubt = QC_POSITION if min(decimals(h["Latitude (degree)"]),
                                   decimals(h["Longitude (degree)"])) \
            < MIN_DECIMALS else 0
        rows.append(cand(
            "nurdle", h["Unique ID"], t_noon(y, m, d), caveat | doubt, la, lo,
            f"beachlitter|nurdle|{la:.3f},{lo:.3f}", 11,
            log_nurdles_10pmin=_log(rate), log_nurdle_count=_log(n),
            collect_min=mins, volunteers=vol, effort_pmin=mins * vol,
            new_tide_line=1.0 if x["New Tide Line"] == "Yes" else 0.0))
    counts["nurdle_standardisation"] = agree
    counts["read"]["nurdle"] = len(pairs) + counts["nurdle_join"]["unmatched"]
    # PLAN §6: the producer's own rule reproduces its published value
    if agree["half_up"] < NURDLE_AGREE_MIN:
        refuse(f"the recomputed pellet rate equals NCEI's published one on "
               f"{agree['half_up']} rows, fewer than {NURDLE_AGREE_MIN} — the "
               f"source changed its standardisation")
    return rows


# ======================================================= 2: NCEI sand ====
def sand_rows(sand, counts):
    rows = []
    for h in sand:
        v = num(h["Microplastics Measurement"])
        if v is None:
            _bump(counts, "ncei_sand_no_value")
            continue
        y, m, d = ncei_date(h["Sample Date"])
        la, lo = float(h["Latitude (degree)"]), float(h["Longitude (degree)"])
        doubt = QC_POSITION if min(decimals(h["Latitude (degree)"]),
                                   decimals(h["Longitude (degree)"])) \
            < MIN_DECIMALS else 0
        rows.append(cand(
            "ncei_sand", h["Unique ID"], t_noon(y, m, d), doubt, la, lo,
            f"beachlitter|ncei_sand|{h['Short Reference'].strip()}|"
            f"{h['Beach Location'].strip()}", 12,
            log_n_m3_sand=-1.0 if v < 0 else _log(v)))
    counts["read"]["ncei_sand"] = len(sand)
    return rows


# ============================================================ 3: EMODnet ==
def load_materials(path=MATERIALS_JSON):
    doc = json.load(open(path, encoding="utf-8"))
    table = {}
    for r in doc["codes"]:
        if r["material"] not in ("plastic", "non_plastic", "ambiguous"):
            refuse(f"materials table: {r}")
        k = (r["list"], r["code"])
        if k in table:
            refuse(f"materials table lists {k} twice")
        table[k] = (r["material"], " ".join(r["name"].split()))
    sizes = {}
    for (L, _c) in table:
        _bump(sizes, L)
    if sizes != doc["list_sizes"]:
        refuse(f"materials table list sizes {sizes} != its own "
               f"{doc['list_sizes']}")
    return table, sizes


def _csv_rows(fh, want, what):
    rd = csv.reader(fh)
    hdr = next(rd)
    miss = [c for c in want if c not in hdr]
    if miss:
        refuse(f"{what} lacks the column(s) {miss}")
    ix = {c: hdr.index(c) for c in want}
    for r in rd:
        yield {c: r[i] for c, i in ix.items()}


def emodnet_rows(zpath, table, sizes, counts):
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        for n in (EMODNET_META, EMODNET_LITTER):
            if n not in names:
                refuse(f"{EMODNET_ZIP} holds no {n} ({names})")
        with z.open(EMODNET_META) as fh:
            meta = list(_csv_rows(io.TextIOWrapper(fh, encoding="utf-8"),
                                  EMODNET_META_COLUMNS, EMODNET_META))
        M = {}
        pol = {}
        coords = {}
        for r in meta:
            s = r["survey_code"]
            if s in M:
                refuse(f"EMODnet survey_code {s} appears twice")
            if r["data policy"] not in POLICIES:
                refuse(f"EMODnet survey {s}: data policy "
                       f"{r['data policy']!r} is not one of {sorted(POLICIES)}")
            if r["survey_type"] not in SURVEY_TYPES:
                refuse(f"EMODnet survey {s}: survey_type {r['survey_type']!r}")
            M[s] = r
            _bump(pol, r["data policy"])
            coords.setdefault(r["location_code"], set()).add(
                (r["location_latitude"], r["location_longitude"]))
        counts["emodnet_surveys_total"] = len(M)
        counts["emodnet_data_policy"] = pol
        counts["excluded_licence"] = {
            p: n for p, n in sorted(pol.items()) if p != PUBLIC_POLICY}
        keep = {s for s, r in M.items() if r["data policy"] == PUBLIC_POLICY}
        tot, pla, amb, seen = {}, {}, {}, {}
        lines = 0
        with z.open(EMODNET_LITTER) as fh:
            for r in _csv_rows(io.TextIOWrapper(fh, encoding="utf-8"),
                               EMODNET_LITTER_COLUMNS, EMODNET_LITTER):
                s = r["survey_code"]
                if s not in keep:
                    if s not in M:
                        refuse(f"EMODnet litter line for unknown survey {s}")
                    continue
                if r["data policy"] != PUBLIC_POLICY:
                    refuse(f"EMODnet survey {s}: a litter line's data policy "
                           f"{r['data policy']!r} differs from the survey's")
                lines += 1
                L, c = r["litter_reference_list"], r["type_code"]
                t = table.get((L, c))
                if t is None:
                    refuse(f"EMODnet type ({L}, {c}) {r['type_name']!r} is not "
                           f"in {os.path.basename(MATERIALS_JSON)}")
                if t[1] != " ".join(r["type_name"].split()):
                    refuse(f"EMODnet type ({L}, {c}) is named "
                           f"{r['type_name']!r}, the table says {t[1]!r}")
                a = num(r["abundance"])
                if a is None or a < 0 or a != int(a):
                    refuse(f"EMODnet survey {s} type {c}: abundance "
                           f"{r['abundance']!r}")
                sl = seen.setdefault(s, {})
                cl = sl.setdefault(L, set())
                if c in cl:
                    refuse(f"EMODnet survey {s} lists type ({L}, {c}) twice")
                cl.add(c)
                tot[s] = tot.get(s, 0.0) + a
                if t[0] == "plastic":
                    pla[s] = pla.get(s, 0.0) + a
                elif t[0] == "ambiguous":
                    amb[s] = amb.get(s, 0.0) + a
    counts["emodnet_public_litter_lines"] = lines
    counts["read"]["emodnet"] = len(keep)
    rows = []
    drop = {}
    by_proto = {}
    assumed = 0
    multi = sum(1 for v in coords.values() if len(v) > 1)
    counts["emodnet_location_codes_with_two_coordinates_any_policy"] = multi
    for s in sorted(keep, key=lambda x: int(x) if x.isdigit() else x):
        r = M[s]
        if s not in seen:
            _bump(drop, "no_litter_lines")
            continue
        # PLAN §6: every survey has as many type lines as its reference list
        lists = tuple(sorted(seen[s]))
        for L in lists:
            if len(seen[s][L]) != sizes.get(L):
                refuse(f"EMODnet survey {s}: {len(seen[s][L])} type lines of "
                       f"list {L}, the list has {sizes.get(L)}")
        proto = LIST_PROTOCOL.get(lists)
        if proto is None:
            refuse(f"EMODnet survey {s}: reference lists {lists}")
        if r["survey_type"] in DROP_SURVEY_TYPES:
            _bump(drop, f"survey_type {r['survey_type']}")
            continue
        la, lo = num(r["location_latitude"]), num(r["location_longitude"])
        if la is None or lo is None:
            _bump(drop, "no_position")
            continue
        L_m = num(r["survey_length"])
        caveat = 0
        if L_m is None:
            L_m = DEFAULT_LENGTH_M[proto]
            caveat = QC_CAVEAT
            assumed += 1
        elif L_m <= 0:
            _bump(drop, "survey_length_zero")
            continue
        w = num(r["survey_width"])
        area = L_m * w if (w is not None and w > 0 and not caveat) else None
        doubt = 0
        sla, slo = (num(r["survey_start_latitude"]),
                    num(r["survey_start_longitude"]))
        if sla is not None and slo is not None and \
                hav_km(la, lo, sla, slo) > START_DOUBT_KM:
            doubt = QC_POSITION
        if len(coords[r["location_code"]]) > 1:
            doubt = QC_POSITION
        y, m, d = iso_date(r["date"], f"EMODnet survey {s}")
        v = litter_values(tot[s], pla.get(s, 0.0), amb.get(s, 0.0), L_m, area)
        v["survey_len_m"] = L_m
        rows.append(cand("emodnet", f"survey {s}", t_noon(y, m, d),
                         caveat | doubt, la, lo,
                         f"beachlitter|emodnet|{r['location_code']}", proto,
                         **v))
        _bump(by_proto, str(proto))
    counts["emodnet_dropped"] = drop
    counts["emodnet_length_assumed"] = assumed
    counts["emodnet_kept_by_protocol"] = by_proto
    return rows


# ============================================================== 4: MDMAP ==
def mdmap_check_transect(r, mats, where):
    """PLAN §6: material totals sum EXACTLY to the total; plastic <= total."""
    t = num(r["total_debris_items"])
    vs = [num(r[m]) for m in mats]
    if t is None or any(v is None for v in vs):
        refuse(f"MDMAP {where}: a blank total ({r['total_debris_items']!r}, "
               f"{[r[m] for m in mats]})")
    if sum(vs) != t:
        refuse(f"MDMAP {where}: material totals sum to {sum(vs)}, "
               f"total_debris_items is {t}")
    if num(r["total_plastic_items"]) > t:
        refuse(f"MDMAP {where}: plastic {r['total_plastic_items']} > total {t}")
    return t, num(r["total_plastic_items"])


def _pair(r, a, b):
    la, lo = num(r[a]), num(r[b])
    return None if la is None or lo is None else (la, lo)


def mdmap_position(r):
    """(lat, lon, doubt) from a transect row's site coordinates, or None."""
    left = _pair(r, "site_waters_edge_left_lat", "site_waters_edge_left_lon")
    right = _pair(r, "site_waters_edge_right_lat",
                  "site_waters_edge_right_lon")
    ends = [p for p in (left, right) if p]
    if not ends:
        bl = _pair(r, "site_back_barrier_left_lat",
                   "site_back_barrier_left_lon")
        br = _pair(r, "site_back_barrier_right_lat",
                   "site_back_barrier_right_lon")
        ends = [p for p in (bl, br) if p]
        if not ends:
            return None
        how = "back_barrier"
    else:
        how = "waters_edge" if len(ends) == 2 else "one_end"
    la = sum(p[0] for p in ends) / len(ends)
    lo = sum(p[1] for p in ends) / len(ends)
    doubt = len(ends) == 2 and hav_km(*ends[0], *ends[1]) > MDMAP_END_DOUBT_KM
    return la, lo, doubt, how


def mdmap_read_export(path, site_id, counts):
    """One site's export zip -> transect rows (main file only)."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        refuse(f"MDMAP site {site_id}: the export is not a zip ({path})")
    with z:
        names = z.namelist()
        if not any(n.endswith("README.txt") for n in names):
            refuse(f"MDMAP site {site_id}: the export has no README ({names})")
        main = [n for n in names if n.endswith(".csv")
                and not n.endswith("_backbarrier.csv")]
        bb = [n for n in names if n.endswith("_backbarrier.csv")]
        if len(main) > 1:
            refuse(f"MDMAP site {site_id}: {len(main)} survey files {main}")
        for n in bb:
            k = list(csv.DictReader(io.StringIO(
                z.read(n).decode("utf-8-sig"))))
            _bump(counts, "mdmap_backbarrier_files")
            _bump(counts, "mdmap_backbarrier_rows_excluded", len(k))
        if not main:
            return [], None
        rd = csv.DictReader(io.StringIO(z.read(main[0]).decode("utf-8-sig")))
        rows = list(rd)
        return rows, list(rd.fieldnames or [])


def mdmap_rows(sites, exports, counts):
    """`sites` = the portal's list; `exports` = {site id: (rows, header)}."""
    rows = []
    drop = {}
    width = {"sentinel_9999": 0, "zero": 0, "blank": 0}
    pos_how = {}
    files_missing = {}
    survey_mismatch = []
    accum_first = 0
    transects = 0
    by_proto = {}
    surveys_seen = 0
    for s in sites:
        sid = int(s["id"])
        trs, hdr = exports[sid]
        want = len(s.get("surveys") or []) + int(s.get("legacyCount") or 0)
        if not trs:
            if want:
                refuse(f"MDMAP site {sid}: the site list names {want} "
                       f"survey(s), its export holds none")
            continue
        miss = [c for c in MDMAP_REQUIRED if c not in hdr]
        if miss:
            refuse(f"MDMAP site {sid}: the export lacks {miss}")
        mats = [m for m in MDMAP_MATERIALS if m in hdr]
        for m in MDMAP_MATERIALS:
            if m not in hdr:
                _bump(files_missing, m)
        surv = {}
        for i, r in enumerate(trs):
            if str(r["shoreline_site_id"]).strip() != str(sid):
                refuse(f"MDMAP site {sid}: a row of site "
                       f"{r['shoreline_site_id']!r}")
            transects += 1
            mdmap_check_transect(r, mats, f"site {sid} row {i + 2}")
            surv.setdefault(r["survey_id"], []).append(r)
        if len(surv) != want:
            survey_mismatch.append({"site": sid, "site_list": want,
                                    "export": len(surv)})
        surveys_seen += len(surv)
        accum_dates = []
        built = []
        for svid, trl in surv.items():
            p = {r["survey_protocol"] for r in trl}
            dset = {r["survey_date"] for r in trl}
            if len(p) != 1 or len(dset) != 1:
                refuse(f"MDMAP survey {svid}: protocols {p}, dates {dset}")
            pname = p.pop()
            proto = MDMAP_PROTOCOL.get(pname)
            if proto is None:
                refuse(f"MDMAP survey {svid}: protocol {pname!r}")
            y, mo, d = iso_date(dset.pop(), f"MDMAP survey {svid}")
            # a transect written twice, identically, is one transect
            uniq, keys = [], set()
            for r in trl:
                k = tuple(sorted(r.items()))
                if k in keys:
                    _bump(drop, "duplicate_transect_row")
                    continue
                keys.add(k)
                uniq.append(r)
            tot = sum(num(r["total_debris_items"]) for r in uniq)
            pla = sum(num(r["total_plastic_items"]) for r in uniq)
            v = {}
            if proto in (8, 9):
                tw = [num(r["transect_width"]) for r in uniq]
                if any(x is None or x <= 0 for x in tw):
                    refuse(f"MDMAP survey {svid}: transect widths {tw}")
                L_m = sum(tw)
                area, a_tot = 0.0, 0.0
                for r, w in zip(uniq, tw):
                    bw = num(r["beach_width_at_transect"])
                    if bw is None:
                        width["blank"] += 1
                    elif bw >= MDMAP_WIDTH_SENTINEL:
                        width["sentinel_9999"] += 1
                    elif bw <= 0:
                        width["zero"] += 1
                    else:
                        area += bw * w
                        a_tot += num(r["total_debris_items"])
                v = litter_values(tot, pla, 0.0, L_m, None)
                if area > 0:
                    v["log_items_m2"] = _log(a_tot / area)
                v["survey_len_m"] = L_m
            else:
                if len(uniq) != 1:
                    refuse(f"MDMAP accumulation survey {svid} has "
                           f"{len(uniq)} transects")
                r = uniq[0]
                L_m = num(r["site_length"])
                bw = num(r["beach_width_at_transect"])
                if bw is None:
                    width["blank"] += 1
                elif bw >= MDMAP_WIDTH_SENTINEL:
                    width["sentinel_9999"] += 1
                    bw = None
                elif bw <= 0:
                    width["zero"] += 1
                    bw = None
                if L_m is None or L_m <= 0:
                    _bump(drop, "accumulation_site_length_zero")
                    L_m = None
                v = litter_values(tot, pla, 0.0, L_m,
                                  L_m * bw if (L_m and bw) else None)
                if L_m:
                    v["survey_len_m"] = L_m
            pos = mdmap_position(uniq[0])
            if pos is None:
                _bump(drop, "no_position")
                continue
            la, lo, doubt, how = pos
            _bump(pos_how, how)
            if proto == 10:
                accum_dates.append(((y, mo, d), len(built)))
            built.append(cand("mdmap", f"survey {svid}", t_noon(y, mo, d),
                              QC_POSITION if doubt else 0, la, lo,
                              f"beachlitter|mdmap|{sid}", proto, **v))
            _bump(by_proto, str(proto))
        # the accumulation interval: days since the previous accumulation
        # survey of the site (each one cleared it); the first has none
        prev = None
        for (ymd, j) in sorted(accum_dates):
            if prev is None:
                accum_first += 1
            else:
                built[j]["v"]["accum_days"] = float(
                    (dt.date(*ymd) - dt.date(*prev)).days)
            prev = ymd
        rows.extend(built)
    counts["read"]["mdmap"] = surveys_seen
    counts["mdmap_sites_listed"] = len(sites)
    counts["mdmap_sites_with_surveys"] = sum(1 for s in sites
                                             if exports[int(s["id"])][0])
    counts["mdmap_transects"] = transects
    counts["mdmap_material_check"] = {"transects_tested": transects,
                                      "failures": 0}
    counts["mdmap_files_missing_material_column"] = files_missing
    counts["mdmap_site_list_vs_export"] = survey_mismatch
    counts["mdmap_width_not_usable"] = width
    counts["mdmap_position_from"] = pos_how
    counts["mdmap_accumulation_first_survey"] = accum_first
    counts["mdmap_dropped"] = drop
    counts["mdmap_kept_by_protocol"] = by_proto
    return rows


# ================================================================ dedup ====
def dedup(rows, counts):
    rank = {s: i for i, s in enumerate(SOURCES)}
    rows = sorted(rows, key=lambda r: (rank[r["src"]], r["seq"]))
    idx = {}
    keep = []
    dd = {}
    for r in rows:
        if r["src"] in LITTER:
            k = r["day"]
            hit = None
            for s in idx.get(k, ()):
                if s["src"] != r["src"] and \
                        abs(s["lat"] - r["lat"]) <= DEDUP_DEG and \
                        abs(s["lon"] - r["lon"]) <= DEDUP_DEG:
                    hit = s
                    break
            if hit is not None:
                _bump(dd, f"{hit['src']} > {r['src']}")
                continue
            idx.setdefault(k, []).append(r)
        keep.append(r)
    counts["dedup"] = dd
    return keep


def to_matrix(rows):
    v = np.full((len(rows), len(CHANNELS)), np.nan)
    for i, r in enumerate(rows):
        for n, x in r["v"].items():
            if x is not None:
                v[i, COL[n]] = x
    return v


# ================================================================ adapter ==
class BeachLitterAdapter(f10b.SourceAdapter):
    store = "beachlitter"
    title = ("Litter on beaches and shorelines: one row per survey (NOAA NCEI "
             "Nurdle Patrol pellet counts and beach-sand microplastics, "
             "EMODnet beach litter under CC BY, NOAA MDMAP)")
    family = "13"
    distribution = "public"
    licence = {
        "name": "CC0 1.0 (NOAA NCEI), CC BY 4.0 (EMODnet Chemistry, the "
                "surveys whose data policy is CB), US-government public data "
                "(NOAA MDMAP, no licence posted — unconfirmed)",
        "redistribution": "attribution", "derived_works": "free",
        "redistribution_confirmed": False,
        "unconfirmed_part": (
            "NOAA MDMAP: no data licence is posted; the portal's contributor "
            "disclaimer grants NOAA the right to 'further disseminate' and "
            "says uploads are 'viewable (and therefore downloadable) by ... "
            "members of the public'. Ask md.monitoring@noaa.gov before a "
            "public publish (plan E-092 D13)."),
        "attribution": (
            "NOAA NCEI Global Marine Microplastics, accession 0306475, "
            "doi:10.25921/v8gj-k896 (CC0 1.0), and Nurdle Patrol, NCEI "
            "accession 0259486, doi:10.25921/ccz7-mq57 (Tunnell, Dunning, "
            "Scheef & Swanson); EMODnet Chemistry, European beach litter "
            "aggregated dataset v2025 (CC BY 4.0, citation mandatory: 'This "
            "resource was generated in the framework of EMODnet Chemistry, "
            "under the support of DG MARE Call for Tender EASME/EMFF/2020/"
            "3.1.11/European Marine Observation and Data Network (EMODnet) - "
            "Lot 5 - Chemistry'); NOAA Marine Debris Program, Marine Debris "
            "Monitoring and Assessment Project (MDMAP); Natural Earth (public "
            "domain) for the shore tests")}
    time_dtype = "int32"
    platform_meta = False
    credentials = ()
    channels = CHANNELS
    # FOOTPRINT (E-078 §2: log2 of the support against 27.83 km / 5 days). A
    # litter survey covers 100 m of beach (the median EMODnet and MDMAP
    # length), a pellet patrol a few tens of metres, a sand core a point:
    # taken as 100 m. A survey is one visit of about an hour.
    log2_fp = float(np.log2(0.1 / 27.83))                # -8.12
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))         # -6.91
    per_year = False
    first_year = 2001
    qc_policy = (
        "qc bits 0-3 = protocol (" + "; ".join(f"{k} {v}" for k, v in
                                              PROTOCOLS.items()) +
        "); bit 4 (16): time of day not given, placed at 12:00 UTC (every "
        "row); bit 5 (32): freshwater shore (farther than "
        f"{FRESH_SEA_KM:g} km from the sea coast and nearer a Natural Earth "
        "lake or river); bit 6 (64): litter protocols — survey length not "
        "given, the protocol's default assumed; Nurdle Patrol — NCEI's "
        "published standardised rate is neither floor nor ceil of the "
        "recomputed one; bit 7 (128): position doubt (farther than "
        f"{FAR_KM:g} km from every mapped shore, fewer than "
        f"{MIN_DECIMALS} published decimals, EMODnet start > 5 km from the "
        "location or a two-coordinate location code, MDMAP ends > 5 km "
        "apart). Values outside the bounds -> NaN, counted. A row with no "
        "density is dropped.")
    sources = (NCEI_CSV, NCEI_LANDING.format(acc7=NURDLE_ACC), EMODNET_ZIP,
               MDMAP_SITES, MDMAP_EXPORT,
               *[NE_URL.format(name=n) for ns in NE_LAYERS.values()
                 for n in ns])
    verified = (
        "2026-10-13 from the sandbox: the NCEI Hub CSV (29,076 rows; 10,615 "
        "Nurdle Patrol, 893 Beach), accession 0259486 v2.2's workbook and "
        "data dictionary, the EMODnet beach-litter v2025 European zip "
        "(16,861 surveys, 2,084,972 lines; data policy CB 4,444, LS 8,130, "
        "UN 4,287), the MDMAP portal's site list (1,271 sites) and every "
        "site's export, and Natural Earth v5.1.2's coastline, minor-island "
        "coastline, lakes and river centre-lines")
    notes = (
        "Densities are stored as log10(1 + value). One streamed pass "
        "(per_year = False); the probe builds the whole store and keeps its "
        "month. The EMODnet code -> material table is "
        "ml/family1/probes/beachlitter_emodnet_materials.json; the ledger "
        "of every drop is store.json's counts and "
        "ml/family1/probes/beachlitter_ledger.json.")
    smoke_window = ("2001-01-01", "2026-12-31")
    smoke_probe_month = "2019-06"
    fetch_month_scope = "archive"

    # ------------------------------------------------------------- reading --
    def _path(self, ctx, url):
        name = mp.mirror_name(url)
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "beachlitter", name)
            if not os.path.exists(p):
                refuse(f"{url} is not in the source mirror ({p})")
            seen = getattr(ctx, "_beachlitter_counted", None)
            if seen is None:
                seen = ctx._beachlitter_counted = set()
            if p not in seen:
                ctx.count_bytes(os.path.getsize(p))
                seen.add(p)
            return p
        p = os.path.join(ctx.scratch, "beachlitter", name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
        import time
        for i in range(mp.HUB_PENDING_TRIES):
            got = f10b.fetch_first([url], path=p,
                                   attempts=int(getattr(ctx.a, "attempts", 3)
                                                or 3))
            if got is None:
                refuse(f"{url} answered 404 — every source of this store is "
                       f"needed whole")
            if os.path.getsize(p) == 0:
                refuse(f"{url} answered an EMPTY body")
            if not mp.hub_pending(p):
                return p
            # the ArcGIS Hub's "file is being generated" answer (mplastic)
            os.remove(p)
            print(f"  beachlitter: {url} — the export is being generated; "
                  f"retry {i + 1}/{mp.HUB_PENDING_TRIES}", flush=True)
            time.sleep(mp.HUB_PENDING_SLEEP)
        refuse(f"{url} kept answering 'being generated'")

    def _bytes(self, ctx, url):
        with open(self._path(ctx, url), "rb") as fh:
            return fh.read()

    def _listing(self, ctx, url):
        html = self._bytes(ctx, url).decode("utf-8", "replace")
        return [h for h in re.findall(r'<a href="([^"?/][^"]*)">', html)
                if not h.startswith("http")]

    def nurdle_workbook(self, ctx):
        land = self._bytes(ctx, NCEI_LANDING.format(acc7=NURDLE_ACC)).decode(
            "utf-8", "replace")
        m = re.search(r"(arc0\d+)/" + NURDLE_ACC, land)
        if not m:
            refuse(f"accession {NURDLE_ACC}: the landing page names no "
                   f"archive directory")
        base = NCEI_ARCHIVE.format(arc=m.group(1), acc7=NURDLE_ACC)
        vers = sorted({h.rstrip("/") for h in self._listing(ctx, base)
                       if re.fullmatch(r"\d+\.\d+/", h)},
                      key=lambda v: tuple(int(x) for x in v.split(".")))
        if not vers:
            refuse(f"accession {NURDLE_ACC}: {base} lists no version")
        d0 = f"{base}{vers[-1]}/data/0-data/"
        files = [h for h in self._listing(ctx, d0) if NURDLE_FILE.match(h)]
        if len(files) != 1:
            refuse(f"accession {NURDLE_ACC} {vers[-1]}: {d0} lists "
                   f"{len(files)} Nurdle Patrol workbook(s) {files}")
        return nurdle_xlsx(self._path(ctx, d0 + files[0])), \
            {"version": vers[-1], "file": files[0]}

    def mdmap_sites(self, ctx):
        sites = json.loads(self._bytes(ctx, MDMAP_SITES))
        if not isinstance(sites, list) or not sites:
            refuse(f"{MDMAP_SITES} answered no site list")
        ids = [s.get("id") for s in sites]
        if any(not isinstance(i, int) for i in ids) or \
                len(set(ids)) != len(ids):
            refuse(f"{MDMAP_SITES}: site ids are not unique integers")
        return sorted(sites, key=lambda s: s["id"])

    def _mdmap_get(self, ctx, sid):
        """One export, fetched in a worker; returns (path, bytes fetched)."""
        import time
        import urllib.request
        url = MDMAP_EXPORT.format(id=sid)
        p = os.path.join(ctx.scratch, "beachlitter", mp.mirror_name(url))
        if os.path.exists(p) and zipfile.is_zipfile(p):
            return p, 0
        os.makedirs(os.path.dirname(p), exist_ok=True)
        err = None
        for i in range(MDMAP_ATTEMPTS):
            part = f"{p}.part{os.getpid()}"
            try:
                req = urllib.request.Request(url, headers=f10b.UA)
                with urllib.request.urlopen(
                        req, timeout=f10b.SOCKET_TIMEOUT) as r, \
                        open(part, "wb") as fh:
                    while True:
                        b = r.read(1 << 16)
                        if not b:
                            break
                        fh.write(b)
                # no Content-Length is sent (measured): a zip that opens
                # is the completeness test — its directory sits at the end
                if not zipfile.is_zipfile(part):
                    raise IOError(f"{url}: the body is not a whole zip")
                n = os.path.getsize(part)
                os.replace(part, p)
                return p, n
            except Exception as e:                              # noqa: BLE001
                err = e
                if os.path.exists(part):
                    os.remove(part)
                time.sleep(3.0 * (2 ** i))
        raise IOError(f"{url}: {MDMAP_ATTEMPTS} attempts failed — {err}")

    def mdmap_exports(self, ctx, sites, counts):
        import time
        out = {}
        t0 = time.time()
        if ctx.source_dir:
            for s in sites:
                out[s["id"]] = mdmap_read_export(
                    self._path(ctx, MDMAP_EXPORT.format(id=s["id"])),
                    s["id"], counts)
            return out
        from concurrent.futures import ThreadPoolExecutor
        ids = [s["id"] for s in sites]
        done = 0
        with ThreadPoolExecutor(MDMAP_WORKERS) as ex:
            for sid, (p, n) in zip(ids, ex.map(
                    lambda i: self._mdmap_get(ctx, i), ids)):
                f10b.count_bytes(n)
                out[sid] = mdmap_read_export(p, sid, counts)
                done += 1
                if done % 100 == 0:
                    ctx.prog.item(f"beachlitter MDMAP exports {done}/"
                                  f"{len(ids)}")
        counts["mdmap_fetch_seconds"] = round(time.time() - t0, 1)
        return out

    def shores(self, ctx):
        layers = {}
        for kind, names in NE_LAYERS.items():
            layers[kind] = [json.loads(self._bytes(ctx, NE_URL.format(name=n)))
                            for n in names]
        return Shores(layers)

    # ------------------------------------------------------------ building --
    def build(self, ctx):
        counts = {"read": {}}
        table, sizes = load_materials()
        nur, sand = ncei_read(self._bytes(ctx, NCEI_CSV).decode("utf-8-sig"),
                              counts)
        xl, wb = self.nurdle_workbook(ctx)
        counts["nurdle_workbook"] = f"{wb['version']}/{wb['file']}"
        r1 = nurdle_rows(nurdle_join(nur, xl, counts), counts)
        r2 = sand_rows(sand, counts)
        r3 = emodnet_rows(self._path(ctx, EMODNET_ZIP), table, sizes, counts)
        sites = self.mdmap_sites(ctx)
        r4 = mdmap_rows(sites, self.mdmap_exports(ctx, sites, counts), counts)
        allrows = r1 + r2 + r3 + r4
        for i, r in enumerate(allrows):
            r["seq"] = i
        kept = dedup(allrows, counts)
        # the shore tests, on every kept row
        sh = self.shores(ctx)
        counts["natural_earth_tag"] = NE_TAG
        counts["natural_earth_vertices_densified"] = sh.vertices
        if kept:
            lat = np.array([r["lat"] for r in kept])
            lon = np.array([r["lon"] for r in kept])
            fresh, far, _d = sh.flags(lat, lon)
            fr, fa = {}, {}
            for r, f, a in zip(kept, fresh, far):
                if f:
                    r["qc"] |= QC_FRESH
                    _bump(fr, r["src"])
                if a:
                    r["qc"] |= QC_POSITION
                    _bump(fa, r["src"])
            counts["freshwater_by_source"] = fr
            counts["far_from_any_mapped_shore_by_source"] = fa
        rank = {s: i for i, s in enumerate(SOURCES)}
        kept.sort(key=lambda r: (r["t"], rank[r["src"]], r["seq"]))
        return kept, counts

    def _rows(self, ctx, t_lo, t_hi):
        kept, counts = self.build(ctx)
        inside = [r for r in kept if t_lo <= r["t"] <= t_hi]
        counts["rows_outside_window"] = len(kept) - len(inside)
        v = to_matrix(inside)
        oob = self.mask_bounds(v)
        if oob:
            f10b._merge_counts(counts, {"out_of_bounds": oob})
        if len(inside):
            alive = np.isfinite(v[:, [COL[c] for c in DENSITY]]).any(axis=1)
        else:
            alive = np.zeros(0, bool)
        counts["rows_no_value"] = int((~alive).sum())
        keep = [r for r, a in zip(inside, alive) if a]
        v = v[alive] if len(inside) else v
        kc, pc, bits, fill = {}, {}, {}, {}
        for r in keep:
            _bump(kc, r["src"])
            _bump(pc, str(r["qc"] & 15))
            for b, name in ((QC_FRESH, "freshwater"), (QC_CAVEAT, "caveat"),
                            (QC_POSITION, "position_doubt")):
                if r["qc"] & b:
                    _bump(bits, name)
        for j, n in enumerate(NAMES):
            fill[n] = int(np.isfinite(v[:, j]).sum()) if len(keep) else 0
        counts["kept_by_source"] = kc
        counts["kept_by_protocol"] = pc
        counts["qc_flags"] = bits
        counts["channel_rows_filled"] = fill
        counts["rows_kept"] = len(keep)
        return keep, v, counts

    def _yield(self, keep, v):
        by_year = {}
        for i, r in enumerate(keep):
            by_year.setdefault(day_of(r["t"])[0], []).append(i)
        for y in sorted(by_year):
            idx = by_year[y]
            yield y, self.pack(
                np.array([keep[i]["t"] for i in idx], np.int64),
                np.array([keep[i]["lat"] for i in idx]),
                np.array([keep[i]["lon"] for i in idx]),
                v[idx],
                np.array([f10b.platform_hash(keep[i]["plat"]) for i in idx],
                         np.int64),
                np.array([keep[i]["qc"] for i in idx], np.uint8))

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        c = {"read": {}}
        nur, sand = ncei_read(self._bytes(ctx, NCEI_CSV).decode("utf-8-sig"),
                              c)
        _xl, wb = self.nurdle_workbook(ctx)
        with zipfile.ZipFile(self._path(ctx, EMODNET_ZIP)) as z:
            with z.open(EMODNET_META) as fh:
                meta = list(_csv_rows(io.TextIOWrapper(fh, encoding="utf-8"),
                                      EMODNET_META_COLUMNS, EMODNET_META))
        pol = {}
        for r in meta:
            _bump(pol, r["data policy"])
        sites = self.mdmap_sites(ctx)
        load_materials()
        return {"dataset": "beachlitter: NCEI Nurdle Patrol + beach sand, "
                           "EMODnet beach litter (CB), NOAA MDMAP",
                "ncei_nurdle_rows": len(nur), "ncei_sand_rows": len(sand),
                "nurdle_workbook": wb, "emodnet_surveys": len(meta),
                "emodnet_data_policy": pol, "mdmap_sites": len(sites),
                "mdmap_survey_refs": sum(len(s.get("surveys") or []) +
                                         int(s.get("legacyCount") or 0)
                                         for s in sites)}

    def fetch_stream(self, ctx):
        keep, v, counts = self._rows(ctx, ctx.t_lo, ctx.t_hi)
        for y, rows in self._yield(keep, v):
            yield y, rows, None
        yield None, None, counts

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        keep, v, counts = self._rows(ctx, lo, hi)
        label = f"{year}-{month:02d}"
        for _y, rows in self._yield(keep, v):
            yield label, rows, None
        yield label, None, counts

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi):
        from family1.adapters import _beachlitter_smoke as sm
        return sm.make_smoke_sources(root, d_lo, d_hi)


ADAPTER = BeachLitterAdapter
