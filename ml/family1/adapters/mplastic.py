"""Floating plastic at the sea surface — one row per SAMPLE (family 1.3, E-092).

PLAIN ENGLISH. Plastic in the open ocean is measured almost only by nets: a
manta or neuston net skims the top few tens of centimetres while a ship tows
it for an hour, and the pieces it catches are counted (and sometimes weighed)
and divided by the area the net swept. Pumps, buckets and bottles measure the
same thing per cubic metre instead. This store is those samples, from four
producers, each tow taken from ONE producer — the original one first — so
the same tow is never counted twice (the plan's decision D2). It is the
ground truth any satellite plastic store in this family must be tested
against (`ml/plans/E092_family13_plastic.md`).

SOURCES, IN PRECEDENCE ORDER (plan D2, D9), all keyless, measured 2026-10-11:

 1. The Ocean Cleanup, Lebreton et al. 2018 (figshare 5873142, CC BY 4.0;
    parsed by the shared `_lebreton2018.py`, which the globe's layer uses
    too): 683 events of the 2015 MEGA expedition (501 manta tows, 151 large
    "mega" net tows) and the 2016 AERIAL expedition (31 photo mosaics), with
    per-size-class pieces/km2 and g/km2 RAW and wind-mixing-CORRECTED; and
    the compiled `HistoricalDataset` (3,532 tows 1972-2015 from Goldstein,
    Wong 1974, Shaw & Mapes 1979, Day & Shaw 1987, Eriksen 2014, Law et al.
    2014 = SEA's Pacific tows (D9: they come from HERE), Van Sebille and
    "This study"), year and month only.
 2. The Ocean Cleanup, Lebreton et al. 2024 (figshare 25674120, CC BY 4.0,
    `Lebreton2024_concentration_data.xlsx`): the expeditions NOT in 2018 —
    North Pacific Missions 2-5, The Vortex Swim, the System 002 aircraft
    surveys and catches, 2018-2022. The item's description defines every
    column (CD1..CD7 pieces/km2 and WD1..WD7 g/km2 over 0.5-1.5, 1.5-5, 5-15,
    15-50, 50-100, 100-500, > 500 mm; WD567 > 50 mm; H_/N_/P_/F_ by piece
    type). MEASURED against the 2018 file on the 2015 rows the two share: the
    2024 manta values are the 2018 MIDPOINT (wind-mixing-corrected)
    estimates, not the raw catch (`lebreton2024_manta_is_midpoint`, 352 of
    352 matched tows); the large-net values equal the raw ones (18 of 18).
    So a 2024 manta row fills only `log_n_km2_wc`. The AIRCRAFT counts were
    REPROCESSED for 2024: on the 31 shared mosaics CD7 equals 2018's count on
    2 and equals (objects >= 50 cm in 2018's own object list) / mosaic area on
    all 31 (see `mobjects`), while WD7 equals 2018's MODELLED midpoint mass;
    a 2024 aircraft row fills `log_n_km2` and `log_n_km2_mega` from CD7 and
    no mass channel. "Cleanup system" catches are kept (gear 8, g/km2 > 5 cm
    only) because the item's own text calls them surface concentrations.
 3. NOAA NCEI Global Marine Microplastics (CC0, accession 0306475): the
    harmonised table (29,076 rows; ocean-WATER samples kept). Dropped BY NAME:
    "Law et al. 2014" (D9 — its positions are wrong). The Ocean Cleanup's own
    submissions ("Egger et al. 2020/2021/2022") are dropped BY MATCH: a row
    with a sources-1-2 row within +-1 UTC day and 0.06 deg (`EGGER_*`) goes,
    the rest are kept (measured: 673 dropped, 96 kept — 2018-12 to 2020-11
    tows no Ocean Cleanup file holds; NCEI's value is the 2024 wind-corrected
    0.5-5 mm count over 0.075 m, cv 5e-6 on 64 pairs, a quantity no channel
    holds, so they stay per m3). Every value in the table is pieces/m3; for a
    SURFACE NET that
    is pieces/km2 divided by a depth the compiler ASSUMED, so per study and
    sampling method (D3):
      (a) the study's own NCEI accession holds an areal column -> n_km2 from
          there, the rows matched 1:1 by day and position (asserted);
      (b) else, if >= 20 of its rows match an areal source already loaded
          (sources 1-2) with ONE ratio (coefficient of variation < 1 %), the
          value is multiplied back by that measured depth;
      (c) else pieces/m3 only.
    Pumps, intakes, bottles, buckets and plankton/bongo nets are VOLUMETRIC
    and keep pieces/m3. The table is `ml/family1/probes/
    mplastic_depth_table.json`.
 4. EMODnet Chemistry floating micro-litter v2026 (CC BY 4.0, citation
    mandatory): one ODV line per sample x litter type x size class, RAW
    counts per net with mouth width and track length, or filtered volume.
    Aggregated here to one row per sample, summing only lines that are
    PLASTIC by EMODnet's own vocabularies (H01 litter types, H05 polymer
    types — `emodnet_classify`); a line whose material the vocabulary does
    not decide is AMBIGUOUS, excluded and counted, and a sample with an
    ambiguous line that caught something gets no total at all (it would be a
    lower bound dressed as a measurement).

A later source's row is DROPPED when an earlier source holds a row on the same
UTC day (the same year-month when either side is month-only) within 0.01° in
both latitude and longitude; every drop is counted per pair of sources in
`counts["dedup"]`, never silent.

WHAT A ROW IS.
  time_s    int32 seconds since 1982-01-01 UTC: the tow's UTC start where the
            producer gives it; a date without a time of day at 12:00 UTC (the
            `gbif`/`glodap` precedent) with qc bit 16; a year-month only at the
            middle of the month (the `gbif` precedent) with qc bits 16 + 32.
            NCEI's "Sample Date" carries 05:00 or 06:00 AM on EVERY row
            (measured): a local-midnight-to-UTC artefact, so NCEI rows are
            dates.
  lat, lon  the tow midpoint where start and end are given (The Ocean
            Cleanup 2018), else the producer's one position.
  platform  platform_hash("<source>|<study or expedition>|<cruise/vessel>").
  values    C = 12, in this order:
     log_n_km2        log10(1 + pieces/km2), AS CAUGHT, all sizes retained
     log_g_km2        log10(1 + g/km2), as caught
     log_n_m3         log10(1 + pieces/m3): volumetric samples, or a surface
                      net with no areal value (NCEI method c)
     log_n_km2_wc     log10(1 + pieces/km2), the PRODUCER'S wind-mixing-
                      corrected value (The Ocean Cleanup only)
     log_n_km2_micro  as caught, 0.05-0.5 cm   ) filled only where the
     log_n_km2_meso   as caught, 0.5-5 cm      ) producer's own size classes
     log_n_km2_macro  as caught, 5-50 cm       ) TILE the channel's range
     log_n_km2_mega   as caught, > 50 cm       ) exactly
     mesh_um          net or filter mesh, micrometres
     net_depth_m      the net's vertical extent: the producer's own where given,
                      else the depth an NCEI compiler assumed (when measured)
     sample_depth_m   depth below the surface of the sample (0 for a surface
                      net; a pump intake or a bottle is deeper)
     wind_ms          wind speed at sampling, m/s (knots converted)
            WHY LOG10(1 + x): the store's values are float16, which tops out at
            65,504; a busy tow reaches 10^6-10^8 pieces/km2 and NCEI holds a
            pump sample at 137,800 pieces/m3. log10(1 + x) keeps a measured
            zero exactly 0 (an empty net is an observation, never NaN) and is
            the quantity the plan's falsifier scores. float16 resolves it to
            ~0.9 % of x near 10^6 and ~1.8 % near 10^8.
  qc        bits 0-3: GEAR (`GEAR_CODES`): 0 unknown, 1 manta net, 2 neuston
            net (incl. AVANI), 3 large "mega" net, 4 pump / intake, 5 grab /
            bottle / bucket, 6 aircraft survey line, 7 other net (plankton,
            bongo, ring, seston, ovoid), 8 cleanup-system catch; bit 4 (16):
            time of day not given (placed at 12:00 UTC); bit 5 (32): day not
            given (placed mid-month).
A row with no concentration value at all is dropped (`rows_no_value`).
Linear values outside the physical bounds (pieces/km2 0..1e9, g/km2 0..1e8,
pieces/m3 0..1e7) are NaN and counted by channel (`out_of_bounds`).

NOT SOURCES OF THIS STORE (plan D4, D5): Isobe 2021, Japan's AOMI and Eriksen
2023 (licences unconfirmed or non-commercial). SEA's later AOMI resubmission
of the Pacific tows (plan §2a, D9 ledger) is therefore not compared here.
"""
import csv
import datetime as dt
import io
import json
import os
import re
import zipfile

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm
from family1.adapters import _lebreton2018 as lb

# ================================================================ sources ==
FIG2018_FILES = lb.API_FILES
FIG2024 = "25674120"
FIG2024_ARTICLE = f"https://api.figshare.com/v2/articles/{FIG2024}"
FIG2024_FILES = f"{FIG2024_ARTICLE}/files"
FIG2024_NAME = "Lebreton2024_concentration_data.xlsx"
NCEI_ITEM = "1b56a456ad8c43889652e62aa25a317e"
NCEI_CSV = (f"https://hub.arcgis.com/api/download/v1/items/{NCEI_ITEM}/csv"
            f"?redirect=true&layers=0")
NCEI_LANDING = ("https://www.ncei.noaa.gov/access/metadata/landing-page/bin/"
                "iso?id=gov.noaa.nodc:{acc7}")
NCEI_ARCHIVE = "https://www.ncei.noaa.gov/data/oceans/archive/{arc}/{acc7}/"
EMODNET_ZIP = ("https://nodc.ogs.it/repository/aggregated-datasets/litter/"
               "EMODnet_floating_microlitter_2026_unrestricted.zip")
EMODNET_TXT = "EMODnet_floating_microlitter_2026_unrestricted.txt"

#: source tags, in PRECEDENCE order (plan D2): an earlier tag wins a duplicate
SOURCES = ("lebreton2018", "lebreton2018_hist", "lebreton2024", "ncei",
           "emodnet")

KN_TO_MS = 0.514444
DEDUP_DEG = 0.01

# ================================================================ channels ==
LIN_MAX = {"n_km2": 1e9, "g_km2": 1e8, "n_m3": 1e7}


def _lg(x):
    return float(np.log10(1.0 + x))


CHANNELS = (
    ("log_n_km2", "log10(1 + pieces km-2), as caught, all sizes retained",
     0.0, _lg(LIN_MAX["n_km2"])),
    ("log_g_km2", "log10(1 + g km-2), as caught", 0.0, _lg(LIN_MAX["g_km2"])),
    ("log_n_m3", "log10(1 + pieces m-3), volumetric samples", 0.0,
     _lg(LIN_MAX["n_m3"])),
    ("log_n_km2_wc", "log10(1 + pieces km-2), the producer's wind-mixing-"
                     "corrected value", 0.0, _lg(LIN_MAX["n_km2"])),
    ("log_n_km2_micro", "log10(1 + pieces km-2), as caught, 0.05-0.5 cm",
     0.0, _lg(LIN_MAX["n_km2"])),
    ("log_n_km2_meso", "log10(1 + pieces km-2), as caught, 0.5-5 cm",
     0.0, _lg(LIN_MAX["n_km2"])),
    ("log_n_km2_macro", "log10(1 + pieces km-2), as caught, 5-50 cm",
     0.0, _lg(LIN_MAX["n_km2"])),
    ("log_n_km2_mega", "log10(1 + pieces km-2), as caught, > 50 cm",
     0.0, _lg(LIN_MAX["n_km2"])),
    ("mesh_um", "micrometres", 0.1, 20000.0),
    ("net_depth_m", "m", 0.0, 10.0),
    ("sample_depth_m", "m", 0.0, 6000.0),
    ("wind_ms", "m s-1", 0.0, 75.0),
)
NAMES = [c[0] for c in CHANNELS]
COL = {n: i for i, n in enumerate(NAMES)}
#: the log channels and the linear quantity (and bound) each one stores
LOG_OF = {"log_n_km2": "n_km2", "log_g_km2": "g_km2", "log_n_m3": "n_m3",
          "log_n_km2_wc": "n_km2", "log_n_km2_micro": "n_km2",
          "log_n_km2_meso": "n_km2", "log_n_km2_macro": "n_km2",
          "log_n_km2_mega": "n_km2"}
CONC = [n for n in NAMES if n in LOG_OF]
#: size-class channels, bounds in mm (upper None = unbounded)
SIZE_CH = {"log_n_km2_micro": (0.5, 5.0), "log_n_km2_meso": (5.0, 50.0),
           "log_n_km2_macro": (50.0, 500.0), "log_n_km2_mega": (500.0, None)}

GEAR_CODES = {0: "unknown", 1: "manta net", 2: "neuston net (incl. AVANI)",
              3: "large ('mega') net", 4: "pump / seawater intake",
              5: "grab / bottle / bucket / sampler", 6: "aircraft survey line",
              7: "other net (plankton, bongo, ring, seston, ovoid)",
              8: "cleanup-system catch (System 002)"}
QC_NO_TIME = 16
QC_NO_DAY = 32
#: NCEI "Sampling Method" -> gear; a method not listed is gear 0 and counted
NCEI_GEAR = {"Manta net": 1, "Neuston net": 2, "AVANI net": 2,
             "Plankton net": 7, "Bongo net": 7,
             "Pump": 4, "Intake seawater pump": 4, "Surface water intake": 4,
             "Grab sample": 5, "Van Dorn sampler": 5, "Niskin bottle": 5,
             "Stainless steel bucket": 5, "Aluminum bucket": 5,
             "Steel bucket": 5, "CTD rosette sampler": 5,
             "Stainless steel sampler": 5, "Glass jar": 5}
#: surface nets: an areal quantity is what they measure (D3)
AREAL_GEAR = {1, 2, 3}
HIST_GEAR = {"manta": 1, "neuston": 2, "ovoid plankton": 7, "seston": 7,
             "ring net": 7, None: 0}

#: NCEI rows dropped BY NAME (plan D9 and the instruction that The Ocean
#: Cleanup's own tows come from The Ocean Cleanup): {Short Reference: why}
NCEI_DROP_BY_NAME = {
    "Law et al. 2014": "D9: NCEI accession 0211008's positions are wrong "
                       "(SEA's own AOMI submission and Eriksen 2023 agree "
                       "with Lebreton 2018's copy, which this store uses)",
}
#: The Ocean Cleanup's own NCEI submissions. NOT dropped by name (that lost
#: 122 tows no Ocean Cleanup file holds — e.g. the 36 "Egger et al. 2022"
#: rows dated 2020, a year Lebreton 2024 has no expedition for): a row is
#: dropped only when a row of sources 1-2 lies within EGGER_DAYS UTC days
#: and EGGER_DEG degrees (latitude and longitude) of it — of the event's
#: start, end or midpoint for a 2018 event, since NCEI's one position may be
#: either end of the tow. Kept rows take the ordinary depth handling.
EGGER_REFS = ("Egger et al. 2020", "Egger et al. 2021", "Egger et al. 2022")
EGGER_DAYS = 1
# 0.06, not 0.05: Lebreton 2024's positions are rounded to 0.01 deg, and the
# 0.2-deg audit found one same-day pair 0.053 deg apart (NCEI 0276263-0000010
# vs 2024 Manta trawl row 604) — the same tow seen from its other end
EGGER_DEG = 0.06
#: the looser tolerance a KEPT Egger row is audited at (reported, not used)
EGGER_AUDIT = (2, 0.2)
NCEI_WATER = "Ocean water"
NCEI_COLUMNS = ("Unique ID", "Sample Date", "Latitude (degree)",
                "Longitude (degree)", "Marine Setting",
                "Water Sample Depth (m)", "Sampling Method", "Mesh Size (mm)",
                "Microplastics Measurement", "Unit", "Short Reference",
                "Organization", "NCEI Accession Number")

# ----------------------------------------------- NCEI accessions, method (a) --
# A study whose OWN NCEI accession holds an AREAL value. Verified 2026-10-11 by
# reading the header of every data file of every accession the water rows cite
# (78 accessions); `AREAL_HINT` finds a column like these, and an accession
# with such a column that is NOT in this table REFUSES the build (new
# information, a new decision). Columns are matched after collapsing
# whitespace; `n` is a sum of (column, factor) in pieces/km2, or ("ratio",
# numerator, denominator in m2) for a count over the swept area. `date` and
# `pos` may list ALTERNATIVES (deployment and retrieval, start and end): the
# NCEI table carries one of them per study (measured: Caldwell, Carbery and
# Frias by the END of the tow, Russell partly by the retrieval), and a row
# matches by the nearest.
AREAL_ACCESSIONS = {
    "0211007": {"study": "Law et al. 2010", "file": r"\.xlsx$", "sheet": "SEA",
                "date": "Sample Date", "pos": [("Lat(deg)", "Long(deg)")],
                "n": [("Pieces/KM2", 1.0)]},
    "0259692": {"study": "Russell and Webster, 2021", "file": r"\.xlsx$",
                "sheet": "MP_data_S2N", "date": ["DATE_Deploy", "DATE_Retrieve"],
                "pos": [("Decimal Latitude at deployment (WGS84)",
                         "Decimal Longitude at deployment (WGS84)"),
                        ("decLat_Retrieve", "decLong_Retrieve")],
                "n": [("SUM_micro_km2", 1.0), ("SUM_Macro_km2", 1.0)]},
    "0277995": {"study": "Trani et al. 2023", "file": r"\.xlsx$",
                "sheet": "MP_data_S2N", "date": "Date",
                "pos": [("Start latitude", "Start longitude"),
                        ("End latitude", "End longitude")],
                "n": [("Microplastics measurement (Items/m2)", 1e6)]},
    "0279322": {"study": "Caldwell et al. 2019", "file": r"\.xlsx$",
                "sheet": "MP_data_S2N", "date": "Date",
                "pos": [("Start Latitude (decimal degree)",
                         "Start Longitude (decimal degree)"),
                        ("End Latitude (decimal degree)",
                         "End Longitude (decimal degree)")],
                "unit": ("Unit", "items/km2"),
                "n": [("Microplastics measurement", 1.0),
                      ("Mesoplastics measurement", 1.0)]},
    "0279323": {"study": "Caldwell et al. 2020", "file": r"\.xlsx$",
                "sheet": "MP_data_S2N", "date": "Date",
                "pos": [("Start Latitude (decimal degree)",
                         "Start Longitude (decimal degree)"),
                        ("End Latitude (decimal degree)",
                         "End Longitude (decimal degree)")],
                "unit": ("Unit", "items/km2"),
                "n": [("Microplastics measurement", 1.0),
                      ("Mesoplastics measurement", 1.0)]},
    "0279325": {"study": "Thushari et al. 2023", "file": r"\.xlsx$",
                "sheet": "MP_data_S2N", "date": "Date",
                "pos": [("Latitude (decimal degree)",
                         "Longitude(decimal degree)")],
                "unit": ("Unit", "counts/km2"),
                "n": [("Microplastics measurement", 1.0),
                      ("Mesoplastics measurement", 1.0),
                      ("Macroplastics measurement", 1.0)]},
    "0284765": {"study": "Rose and Webber, 2019", "file": r"\.xlsx$",
                "sheet": "Microplastics_Data", "date": "Date (m/d/yyyy)",
                "pos": [("Latitude (decimal degree)",
                         "Longitude(decimal degree)")],
                "n": [("Microplastics measurement (items/km2)", 1.0)]},
    "0294865": {"study": "Frias et al. 2020", "file": r"\.xlsx$",
                "sheet": "Microplastics_Summary_Data",
                "date": "Date (m/d/yyyy)",
                "pos": [("Start latitude (decimal degree)",
                         "Start longitude(decimal degree)"),
                        ("End latitude (decimal degree)",
                         "End longitude(decimal degree)")],
                "n": ("ratio", "No of MPs without contaminatiion", "Area (m2)")},
}
#: accessions whose areal-looking column was READ AND REJECTED, with why
AREAL_REJECTED = {
    "0279321": ("Carbery et al. 2022: its 'Microplastics concentration "
                "(MPs/Km2)' column implies a net depth of 0.48 mm against the "
                "same row's MPs/m3 (measured 2026-10-11 on all 16 tows; 98 "
                "pieces in a 61 cm net over a tow of kilometres is ~10^4-10^5 "
                "per km2, not 54.6) — the unit is not what the header says, so "
                "the study stays per m3 (method c)"),
}
#: a method-(a) study whose NCEI value implies a net depth outside this range
#: (metres) has an areal column whose unit cannot be right: refused
A_DEPTH_RANGE = (0.01, 2.0)
AREAL_HINT = re.compile(r"(?i)km\s*\^?\s*2|km-2|km²|(?:items|pieces|mps|counts)"
                        r"\s*/\s*m\s*\^?\s*2\b|\(m2\)")
ACC_DATA = re.compile(r"(?i)\.(xlsx|xls|csv|txt)$")
ACC_SKIP = re.compile(r"(?i)^weather_boat_")
METHOD_B_MIN = 20
METHOD_B_CV = 0.01

# ----------------------------------------------------------------- EMODnet --
#: H05 polymer / material types (vocab.nerc.ac.uk/collection/H05, read
#: 2026-10-11): "plastic" where the definition is a synthetic polymer or
#: "undefined plastic"; "non-plastic" where it says natural, glass, metal, tar
#: or "undefined non-plastic"; tyre wear is a mixture -> ambiguous.
H05_PLASTIC = {f"H05{n:05d}" for n in list(range(1, 16)) + [25, 27, 29]}
H05_NONPLASTIC = {"H0500020", "H0500021", "H0500022", "H0500023", "H0500024",
                  "H0500026"}
#: H01 litter types (vocab.nerc.ac.uk/collection/H01): only H0100001
#: "microplastic items" is DEFINED as plastic ("any kind of micro-litter item
#: made of any kind of plastic material"); H0100007/8 are defined non-plastic.
#: The shapes (fragments, pellets, filaments, films, foams, granules) and
#: "undefined" are defined as MICRO-LITTER particles of that shape — the
#: vocabulary leaves their material to the polymer type (H05), so a shape line
#: without a polymer type is AMBIGUOUS by EMODnet's own text.
H01_PLASTIC = {"H0100001"}
H01_NONPLASTIC = {"H0100007", "H0100008"}
H01_SHAPES = {"H0100002", "H0100003", "H0100004", "H0100005", "H0100006",
              "H0100009", "H0100010"}
#: the one switch that would read a shape line without a polymer as plastic
#: (the MSFD protocol's reading). OFF: the instruction is EMODnet's own text.
EMODNET_SHAPES_ARE_PLASTIC = False
#: H03 size classes -> (lo, hi) in mm, parsed from the label (never typed)
H03_RE = re.compile(r"\(([\d.]+)\s*(micrometres|millimetres)?\s*to\s*([\d.]+)"
                    r"\s*(micrometres|millimetres)\)")
EMODNET_COLUMNS = (
    "LOCAL_CDI_ID", "EDMO_code", "sampID", "yyyy-mm-ddThh:mm:ss.sss",
    "Longitude [degrees_east]", "Latitude [degrees_north]",
    "Instrument / gear type", "Data set name", "Cruise name",
    "CSR platform name", "MinDepBelowSeaSurf [m]",
    "MaxDepBelowSeaSurf [m]", "mouth_width [cm]", "Mesh_size [micrometres]",
    "SampTrkLen [km]", "VolWBodySamp [m3]", "Microlitter_count_water",
    "Microlitter_count_water_VOR", "*Microlitter_weight_water [g]",
    "Microlitter_type_water", "Microlitter_size_class_water",
    "Microlitter_polymer_type_water", "WindSpd [m/s]", "Access restriction")


class FormatError(ValueError):
    """A source that is not the layout verified above."""


def refuse(msg):
    raise FormatError(f"REFUSING mplastic: {msg}")


# ================================================================== time ==
def t_of(y, m, d=None, hour=None):
    """(t seconds since 1982, qc time bits)."""
    if d is None:
        start = int(cm.days_from_civil(y, m, 1)) * 86400
        mlen = int(cm.MLEN[m] + (m == 2 and bool(cm.is_leap(y))))
        return start + (mlen * 86400) // 2, QC_NO_TIME | QC_NO_DAY
    base = int(cm.days_from_civil(y, m, d)) * 86400
    if hour is None:
        return base + 43200, QC_NO_TIME
    mins = int(round(hour * 60))
    return base + min(mins, 1439) * 60, 0


def day_of(t):
    d = dt.date(1982, 1, 1) + dt.timedelta(days=int(t // 86400))
    return d.year, d.month, d.day


# ============================================================= candidates ==
def cand(src, key, t, qc, lat, lon, plat, gear, study, **lin):
    """One candidate row before de-duplication. `lin` = linear values."""
    y, m, d = day_of(t)
    return {"src": src, "key": key, "t": int(t), "qc": int(qc) | int(gear),
            "lat": float(lat), "lon": float(lon), "plat": plat,
            "study": study, "ym": (y, m),
            "day": None if (qc & QC_NO_DAY) else (y, m, d), "v": lin}


def _sum(vals):
    """Sum of numbers; None if any is missing (a partial sum is not a total)."""
    out = 0.0
    for v in vals:
        x = lb.num(v)
        if x is None:
            return None
        out += x
    return out


# ============================================ source 1: Lebreton et al. 2018 ==
def lebreton2018_rows(L, counts):
    rows = []
    ev = L["events"]
    for eid in L["order"]:
        e = ev[eid]
        c = e["c"]
        mid_lat = (e["lat0"] + e["lat1"]) / 2
        mid_lon = (e["lon0"] + e["lon1"]) / 2
        t, q = t_of(e["y"], e["mo"], e["d"], e["hour"])
        dep = lb.num(e["depth_km"])
        wind = lb.num(e["wind_kn"])
        common = dict(net_depth_m=None if dep is None else dep * 1000.0,
                      sample_depth_m=0.0,
                      wind_ms=None if wind is None else wind * KN_TO_MS)
        expd = ("AERIAL expedition 2016" if e["type"] == "aerial"
                else "MEGA expedition 2015")
        plat = f"lebreton2018|{expd}|{e['platform']}"
        R, M = 3, 0                       # index of raw and midpoint estimates
        if e["type"] == "manta":
            v = dict(n_km2=_sum([c["micro"]["n"][R], c["meso"]["n"][R]]),
                     g_km2=_sum([c["micro"]["g"][R], c["meso"]["g"][R]]),
                     n_km2_wc=_sum([c["micro"]["n"][M], c["meso"]["n"][M]]),
                     n_km2_micro=lb.num(c["micro"]["n"][R]),
                     n_km2_meso=lb.num(c["meso"]["n"][R]))
            gear = 1
        elif e["type"] == "mega":
            v = dict(n_km2=lb.num(c["macro"]["n"][R]),
                     g_km2=lb.num(c["macro"]["g"][R]),
                     n_km2_wc=lb.num(c["macro"]["n"][M]),
                     n_km2_macro=lb.num(c["macro"]["n"][R]))
            gear = 3
        else:
            v = dict(n_km2=lb.num(c["mega"]["n"][R]),
                     g_km2=lb.num(c["mega"]["g"][R]),
                     n_km2_wc=lb.num(c["mega"]["n"][M]),
                     n_km2_mega=lb.num(c["mega"]["n"][R]))
            gear = 6
            common["net_depth_m"] = None
        rows.append(cand("lebreton2018", f"event {eid}", t, q, mid_lat,
                         mid_lon, plat, gear, expd, **v, **common))
        rows[-1]["alt"] = [(e["lat0"], e["lon0"]), (e["lat1"], e["lon1"])]
        if e["type"] == "manta":
            # not a channel: the quantity NCEI's Egger rows turn out to be
            rows[-1]["micro_wc"] = lb.num(c["micro"]["n"][M])
    counts["read"]["lebreton2018"] = len(rows)
    return rows


def lebreton2018_hist_rows(L, counts):
    rows = []
    gears = {}
    for i, h in enumerate(L["historical"]):
        if not h["mo"]:
            # none in the published file (measured); a year-only tow has no
            # precedent in this family's time rules, so it is not guessed
            refuse(f"historical row {i + 1} has a year and no month")
        t, q = t_of(h["y"], h["mo"])
        gear = HIST_GEAR.get(h["gear"])
        if gear is None:
            refuse(f"historical gear {h['gear']!r} is not in HIST_GEAR")
        gears[h["gear"] or "-"] = gears.get(h["gear"] or "-", 0) + 1
        ref = lb.txt(h["ref"]) or ""
        mesh = lb.num(h["mesh_um"])
        dep = lb.num(h["depth_m"])
        rows.append(cand(
            "lebreton2018_hist", f"historical row {i + 1}", t, q, h["lat"],
            h["lon"], f"lebreton2018-historical|{ref}|{lb.txt(h['origin'])}",
            gear, ref, n_km2=lb.num(h["n"]), g_km2=lb.num(h["g"]),
            mesh_um=mesh, net_depth_m=dep,
            sample_depth_m=0.0 if gear in AREAL_GEAR else None))
    counts["read"]["lebreton2018_hist"] = len(rows)
    counts["lebreton2018_hist_gear"] = gears
    return rows


# ============================================ source 2: Lebreton et al. 2024 ==
#: the expeditions already in Lebreton 2018, excluded BY NAME
L2024_IN_2018 = {"MEGA Expedition", "AERIAL Expedition"}
#: CDk / WDk size classes in mm, from the figshare item's description
L2024_CLASS_MM = {1: (0.5, 1.5), 2: (1.5, 5.0), 3: (5.0, 15.0),
                  4: (15.0, 50.0), 5: (50.0, 100.0), 6: (100.0, 500.0),
                  7: (500.0, None)}
L2024_SHEETS = {"Manta trawl": 1, "Mega trawl": 3, "Aerial survey": 6,
                "Cleanup system": 8}
L2024_DESCRIBED = ("CD1", "CD7", "WD1", "WD7", "WD567")


def l2024_expedition(s):
    return " ".join(str(s).replace("'", " ").split())


def lebreton2024_check_description(article):
    """The producer's own text must define every column read here."""
    desc = re.sub(r"<[^>]+>", " ", article.get("description") or "")
    desc = " ".join(desc.split())
    for k, kind, unit in (("CD", "numerical", "#/km"),
                          ("WD", "mass", "g/km")):
        for n, (lo, hi) in L2024_CLASS_MM.items():
            rng = (rf"> {lo:g} mm" if hi is None else
                   rf"{lo:g} mm - {hi:g} mm")
            want = f"{k}{n} = {kind} concentration, {rng.replace(chr(92), '')}"
            pat = (rf"{k}{n}\s*:\s*{kind} concentration of [\w ]+\({rng}\)"
                   rf" in {re.escape(unit)}")
            if not re.search(pat, desc):
                refuse(f"figshare {FIG2024}'s description does not define "
                       f"{want} in {unit}2 — the column cannot be read")
    if not re.search(r"WD567\s*:\s*mass concentration of macroplastics and "
                     r"megaplastics \(> 50 mm\) in g/km", desc):
        refuse("figshare 25674120's description does not define WD567")
    if "cleanup system" not in desc.lower() or \
            "surface concentration" not in desc.lower():
        refuse("figshare 25674120's description no longer calls the cleanup-"
               "system values surface concentrations")
    lic = (article.get("license") or {}).get("name")
    if lic != "CC BY 4.0":
        refuse(f"figshare 25674120's licence is {lic!r}, not CC BY 4.0")
    return desc


def lebreton2024_read(path):
    """{sheet: [dict per row]} with the header as given."""
    import openpyxl
    # a file object: the mirrored ndownloader path has no .xlsx extension,
    # and openpyxl refuses a path by its extension
    with open(path, "rb") as fh:
        data = io.BytesIO(fh.read())
    wb = openpyxl.load_workbook(data, read_only=True, data_only=True)
    out = {}
    for ws in wb.worksheets:
        it = ws.iter_rows(values_only=True)
        hdr = [str(h).strip() if h is not None else "" for h in next(it)]
        out[ws.title] = [dict(zip(hdr, r)) for r in it
                         if any(c is not None for c in r)]
    return out


def _cls_sum(r, prefix, ks):
    return _sum([r.get(f"{prefix}{k}") for k in ks])


def lebreton2024_rows(book, L, counts):
    """Rows of the expeditions not in 2018, and the measurements that say what
    the 2024 manta values are."""
    miss = sorted(set(L2024_SHEETS) - set(book))
    if miss:
        refuse(f"{FIG2024_NAME} lacks the sheet(s) {miss}")
    rows = []
    excl = {}
    ev18 = [e for e in L["events"].values()]
    by_day = {}
    for e in ev18:
        by_day.setdefault((e["y"], e["mo"], e["d"], e["type"]), []).append(e)
    agree = {"manta_mid": 0, "manta_raw": 0, "manta_pairs": 0,
             "mega_pairs": 0, "mega_raw": 0, "aerial_pairs": 0,
             "aerial_raw": 0, "aerial_mass_mid": 0}
    ratios = []

    def close(a, b):
        # the 2024 file rounds some values to two decimals (aircraft: 2.7 for
        # 2018's 2.7031844): equal within that rounding. The manta midpoint
        # and raw estimates differ by tens of percent, far outside it.
        return a is not None and b is not None and \
            abs(a - b) <= 0.005 + 1e-6 * abs(b)
    read = 0
    for sheet, gear in L2024_SHEETS.items():
        for i, r in enumerate(book[sheet]):
            read += 1
            exp = l2024_expedition(r["Expedition"])
            y, mo, d = int(r["Year"]), int(r["Month"]), int(r["Day"])
            la, lo = float(r["Lat"]), float(r["Lon"])
            if exp in L2024_IN_2018:
                excl[f"{exp} ({sheet})"] = excl.get(f"{exp} ({sheet})", 0) + 1
                typ = {1: "manta", 3: "mega", 6: "aerial"}.get(gear)
                m = [e for e in by_day.get((y, mo, d, typ), [])
                     if abs(e["lat"] - la) <= DEDUP_DEG
                     and abs(e["lon"] - lo) <= DEDUP_DEG]
                if len(m) == 1:
                    c = m[0]["c"]
                    if typ == "manta":
                        agree["manta_pairs"] += 1
                        s12 = _cls_sum(r, "CD", (1, 2))
                        agree["manta_mid"] += close(s12, lb.num(
                            c["micro"]["n"][0]))
                        agree["manta_raw"] += close(s12, lb.num(
                            c["micro"]["n"][3]))
                    elif typ == "mega":
                        agree["mega_pairs"] += 1
                        agree["mega_raw"] += close(_cls_sum(r, "CD", (5, 6)),
                                                   lb.num(c["macro"]["n"][3]))
                    else:
                        agree["aerial_pairs"] += 1
                        a24 = lb.num(r.get("CD7"))
                        a18 = lb.num(c["mega"]["n"][3])
                        agree["aerial_raw"] += close(a24, a18)
                        agree["aerial_mass_mid"] += close(
                            lb.num(r.get("WD7")), lb.num(c["mega"]["g"][0]))
                        if a24 and a18:
                            ratios.append(a24 / a18)
                continue
            t, q = t_of(y, mo, d)
            plat = f"lebreton2024|{exp}|"
            if gear == 1:
                v = dict(n_km2_wc=_cls_sum(r, "CD", range(1, 7)),
                         sample_depth_m=0.0)
            elif gear == 3:
                v = dict(n_km2=_cls_sum(r, "CD", (5, 6, 7)),
                         g_km2=_cls_sum(r, "WD", (5, 6, 7)),
                         n_km2_wc=_cls_sum(r, "CD", (5, 6, 7)),
                         n_km2_macro=_cls_sum(r, "CD", (5, 6)),
                         n_km2_mega=lb.num(r.get("CD7")), sample_depth_m=0.0)
            elif gear == 6:
                # the aircraft's count of objects > 50 cm, as the 2024 paper
                # processed it; WD7 is a MODELLED mass (it equals 2018's
                # midpoint mass estimate, and 2018 has no raw aircraft mass),
                # so it fills no as-caught channel
                v = dict(n_km2=lb.num(r.get("CD7")),
                         n_km2_mega=lb.num(r.get("CD7")), sample_depth_m=0.0)
            else:
                v = dict(g_km2=lb.num(r.get("WD567")), sample_depth_m=0.0)
            rows.append(cand("lebreton2024", f"{sheet} row {i + 2}", t, q, la,
                             lo, plat, gear, exp, **v))
            if gear == 1:
                # not a channel (see the Egger depth in ncei_rows)
                rows[-1]["micro_wc"] = _cls_sum(r, "CD", (1, 2))
    counts["read"]["lebreton2024"] = read
    counts["lebreton2024_excluded_by_name"] = excl
    counts["lebreton2024_vs_2018"] = agree
    # THE MEASUREMENT THE MANTA ROWS REST ON: on the 2015 rows both files
    # hold, the 2024 manta value is the 2018 MIDPOINT, not the raw catch.
    p = agree["manta_pairs"]
    if p < 100 or agree["manta_mid"] < 0.99 * p or agree["manta_raw"] > 0.5 * p:
        refuse(f"the 2024 manta values no longer read as 2018's midpoint "
               f"estimates ({agree}) — the channel they fill is undecided")
    if agree["mega_pairs"] and \
            agree["mega_raw"] < 0.99 * agree["mega_pairs"]:
        refuse(f"the 2024 large-net values no longer equal 2018's raw catch "
               f"({agree})")
    # THE AIRCRAFT COUNTS WERE REPROCESSED (measured 2026-10-11): on the 31
    # mosaics both files hold, 2024's CD7 equals 2018's count on only a few
    # while WD7 equals 2018's midpoint mass on all of them. Recorded, not
    # refused: the 2024 rows kept are 2021-22 flights 2018 never had.
    if ratios:
        counts["lebreton2024_aerial_count_ratio_2024_over_2018"] = {
            "median": float(np.median(ratios)), "min": float(min(ratios)),
            "max": float(max(ratios)), "n": len(ratios)}
    counts["lebreton2024_manta_is_midpoint"] = True
    return rows


# ==================================================================== NCEI ==
def ncei_date(s):
    """'M/D/YYYY h:mm:ss AM' -> (y, m, d, 'h:mm:ss AM')."""
    day, _, tod = str(s).strip().partition(" ")
    m, d, y = (int(x) for x in day.split("/"))
    return y, m, d, tod.strip()


def ncei_read(text, counts):
    rd = csv.DictReader(io.StringIO(text))
    miss = [c for c in NCEI_COLUMNS if c not in (rd.fieldnames or [])]
    if miss:
        refuse(f"the NCEI CSV lacks the column(s) {miss}")
    allr = list(rd)
    if not allr:
        refuse("the NCEI CSV holds no row")
    settings = {}
    water = []
    for x in allr:
        s = x["Marine Setting"]
        settings[s] = settings.get(s, 0) + 1
        if s == NCEI_WATER:
            water.append(x)
    counts["ncei_rows_total"] = len(allr)
    counts["ncei_marine_setting"] = settings
    units = {x["Unit"] for x in water}
    if units != {"pieces/m3"}:
        refuse(f"NCEI water samples carry units {sorted(units)}, not only "
               f"pieces/m3")
    tods = {}
    for x in water:
        tods[ncei_date(x["Sample Date"])[3]] = \
            tods.get(ncei_date(x["Sample Date"])[3], 0) + 1
    counts["ncei_time_of_day"] = tods
    return water


def _hdr(s):
    return " ".join(str(s).split()) if s is not None else ""


def read_table(path):
    """A spreadsheet or text table -> {sheet: [row tuple, ...]}."""
    low = path.lower()
    if low.endswith(".xlsx"):
        import openpyxl
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            return {ws.title: [tuple(r) for r in ws.iter_rows(values_only=True)
                               if any(c is not None for c in r)]
                    for ws in wb.worksheets}
    if low.endswith(".xls"):
        import xlrd
        wb = xlrd.open_workbook(path)
        out = {}
        for sh in wb.sheets():
            rs = []
            for i in range(sh.nrows):
                r = []
                for j, c in enumerate(sh.row(i)):
                    if c.ctype == xlrd.XL_CELL_DATE:
                        r.append(xlrd.xldate.xldate_as_datetime(c.value,
                                                                wb.datemode))
                    elif c.ctype == xlrd.XL_CELL_EMPTY:
                        r.append(None)
                    else:
                        r.append(c.value)
                if any(x not in (None, "") for x in r):
                    rs.append(tuple(r))
            out[sh.name] = rs
        return out
    raw = open(path, "rb").read().decode("utf-8", "replace")
    first = raw.splitlines()[:3]
    return {"text": [tuple(ln.split("\t" if "\t" in ln else ",")) for ln in
                     first]}


def areal_hints(tables):
    """Header cells (first three rows of every sheet) that look areal."""
    hits = []
    for sheet, rs in tables.items():
        for r in rs[:3]:
            for c in r:
                if isinstance(c, str) and len(c) < 120 and AREAL_HINT.search(c):
                    hits.append(f"{sheet}: {_hdr(c)}")
    return hits


def _as_date(v):
    if isinstance(v, dt.datetime):
        return v.year, v.month, v.day
    if isinstance(v, dt.date):
        return v.year, v.month, v.day
    s = str(v).strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return tuple(int(x) for x in m.groups())
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        mo, d, y = (int(x) for x in m.groups())
        return y, mo, d
    return None


def _alts(x):
    return [x] if isinstance(x, str) else list(x)


def areal_table(acc7, tables, spec):
    """The accession's own rows: [([(y, m, d), ...], [(lat, lon), ...],
    n_km2 or None)] — every alternative date and position a row carries."""
    want = spec["sheet"]
    if want not in tables:
        refuse(f"accession {acc7}: no sheet {want!r} ({sorted(tables)})")
    rs = tables[want]
    hdr = [_hdr(c) for c in rs[0]]
    col = {}
    for i, h in enumerate(hdr):
        if h and h not in col:
            col[h] = i
    dates = _alts(spec["date"])
    pos = list(spec["pos"])
    cols = dates + [c for p in pos for c in p]
    n = spec["n"]
    if isinstance(n, tuple) and n[0] == "ratio":
        cols += [n[1], n[2]]
    else:
        cols += [c for c, _f in n]
    if "unit" in spec:
        cols.append(spec["unit"][0])
    miss = [c for c in cols if c not in col]
    if miss:
        refuse(f"accession {acc7} ({spec['study']}): the sheet {want!r} lacks "
               f"{miss}; it has {hdr}")
    out = []
    for r in rs[1:]:
        r = list(r) + [None] * (len(hdr) - len(r))
        days = [d for d in (_as_date(r[col[c]]) for c in dates
                            if r[col[c]] is not None) if d]
        ps = []
        for la_c, lo_c in pos:
            la, lo = lb.num(r[col[la_c]]), lb.num(r[col[lo_c]])
            if la is not None and lo is not None:
                ps.append((la, lo))
        if not days or not ps:
            continue
        if "unit" in spec:
            u = str(r[col[spec["unit"][0]]] or "").strip().lower()
            if u != spec["unit"][1]:
                refuse(f"accession {acc7}: unit {u!r}, expected "
                       f"{spec['unit'][1]!r}")
        comp = {}
        if isinstance(n, tuple):
            num_, den = lb.num(r[col[n[1]]]), lb.num(r[col[n[2]]])
            val = None if num_ is None or not den else num_ / den * 1e6
        else:
            parts = [lb.num(r[col[c]]) for c, _f in n]
            val = None if any(p is None for p in parts) else \
                sum(p * f for p, (_c, f) in zip(parts, n))
            if len(n) > 1:
                comp = {c: (None if p is None else p * f)
                        for p, (c, f) in zip(parts, n)}
        out.append((days, ps, val, comp))
    return out


def match_one_to_one(nrows, arows):
    """Greedy nearest match: an NCEI row (day, lat, lon) and an accession row
    match on any of the accession row's dates and within 0.01 deg of any of
    its positions -> {ncei index: areal index}; ties go to the earlier areal
    row. Unmatched NCEI rows are absent from the result."""
    by_day = {}
    for j, row in enumerate(arows):
        for day in set(tuple(d) for d in row[0]):
            by_day.setdefault(day, []).append(j)

    def run(scale):
        # candidates first, then the MOST CONSTRAINED row first (fewest
        # candidates), so a row with one possible partner is never robbed of
        # it by a row that had a choice
        cands = {}
        for i, nr in enumerate(nrows):
            day, la, lo = nr[:3]
            v = nr[3] if len(nr) > 3 else None
            c = []
            for j in by_day.get(tuple(day), []):
                best_d = None
                for a_la, a_lo in arows[j][1]:
                    dl, dn = abs(a_la - la), abs(a_lo - lo)
                    if dl <= DEDUP_DEG and dn <= DEDUP_DEG:
                        d = round(dl * dl + dn * dn, 12)
                        best_d = d if best_d is None else min(best_d, d)
                if best_d is None:
                    continue
                # TIES AT ONE POSITION (two tows at the same station on one
                # day, measured in Law et al. 2010) are broken by how far the
                # pair's value ratio is from the study's own, so two tows are
                # not paired crosswise
                a = arows[j][2]
                if scale and v is not None and a is not None:
                    dev = (0.0 if (a == 0 and v == 0) else
                           9.0 if (a == 0) != (v == 0) else
                           abs(np.log(a / (v * scale))))
                else:
                    dev = 0.0
                c.append((best_d, dev, j))
            if c:
                cands[i] = sorted(c)
        used = set()
        out = {}
        for i in sorted(cands, key=lambda i: (len(cands[i]), i)):
            for _d, _dev, j in cands[i]:
                if j not in used:
                    used.add(j)
                    out[i] = j
                    break
        return out
    first = run(None)
    r = [arows[j][2] / nrows[i][3] for i, j in first.items()
         if len(nrows[i]) > 3 and arows[j][2] and nrows[i][3]]
    return run(float(np.median(r))) if r else first


def _cv(xs):
    xs = np.asarray(xs, np.float64)
    if xs.size < 2:
        return 0.0
    return float(xs.std() / xs.mean()) if xs.mean() else float("inf")


# =================================================================== EMODnet ==
def h03_bounds(label):
    """'SDN:H03::H0300008 (1 to 5 millimetres)' -> (1.0, 5.0) in mm."""
    m = H03_RE.search(label or "")
    if not m:
        return None
    lo, u1, hi, u2 = m.groups()
    u1 = u1 or u2
    f1 = 1e-3 if u1 == "micrometres" else 1.0
    f2 = 1e-3 if u2 == "micrometres" else 1.0
    return float(lo) * f1, float(hi) * f2


def _code(label, prefix):
    m = re.search(rf"\b({prefix}\d{{5}})\b", label or "")
    return m.group(1) if m else ""


def emodnet_classify(type_label, polymer_label,
                     shapes_plastic=EMODNET_SHAPES_ARE_PLASTIC):
    """'plastic' | 'non-plastic' | 'ambiguous' for one ODV line."""
    t = _code(type_label, "H01")
    p = _code(polymer_label, "H05")
    if t in H01_NONPLASTIC or p in H05_NONPLASTIC:
        return "non-plastic"
    if p in H05_PLASTIC:
        return "plastic"
    if t in H01_PLASTIC and not p:
        return "plastic"
    if shapes_plastic and t in H01_SHAPES and t != "H0100010" and not p:
        return "plastic"
    return "ambiguous"


def tiles(classes, lo, hi):
    """Do the (lo, hi) mm classes tile [lo, hi) exactly? hi None = open."""
    if not classes:
        return False
    cs = sorted(set(classes))
    if cs[0][0] != lo:
        return False
    for (a0, a1), (b0, _b1) in zip(cs, cs[1:]):
        if a1 != b0:
            return False
    return cs[-1][1] == hi


def emodnet_read(text, counts):
    lines = [ln for ln in text.split("\n") if ln and not ln.startswith("//")]
    if not lines:
        refuse("the EMODnet spreadsheet holds no line")
    hdr = lines[0].rstrip("\r").split("\t")
    H = {}
    for i, n in enumerate(hdr):
        H.setdefault(n, i)
    miss = [c for c in EMODNET_COLUMNS if c not in H]
    if miss:
        refuse(f"the EMODnet spreadsheet lacks {miss}")
    rows = [ln.rstrip("\r").split("\t") for ln in lines[1:]]
    counts["emodnet_lines"] = len(rows)
    return H, rows


def emodnet_rows(H, lines, counts):
    def g(r, n):
        i = H[n]
        return r[i] if i < len(r) else ""

    lic = {g(r, "Access restriction") for r in lines}
    if lic != {"Creative Commons Attribution 4.0 International(CB)"}:
        refuse(f"EMODnet lines carry access restrictions {sorted(lic)}")
    samples = {}
    order = []
    for r in lines:
        k = (g(r, "LOCAL_CDI_ID"), g(r, "EDMO_code"), g(r, "sampID"))
        if k not in samples:
            samples[k] = []
            order.append(k)
        samples[k].append(r)
    mapping = {}
    cls_count = {"plastic": 0, "non-plastic": 0, "ambiguous": 0}
    drop = {}
    out = []
    for k in order:
        rs = samples[k]
        r0 = rs[0]
        for n in ("yyyy-mm-ddThh:mm:ss.sss", "Longitude [degrees_east]",
                  "Latitude [degrees_north]", "mouth_width [cm]",
                  "SampTrkLen [km]", "VolWBodySamp [m3]"):
            if len({g(r, n) for r in rs}) > 1:
                refuse(f"EMODnet sample {k}: {n} varies within the sample")
        plast = []
        amb_caught = False
        vor = False
        for r in rs:
            tl, pl = g(r, "Microlitter_type_water"), \
                g(r, "Microlitter_polymer_type_water")
            c = emodnet_classify(tl, pl)
            mk = f"{tl} | {pl or '(no polymer type)'}"
            mapping[mk] = c
            cls_count[c] += 1
            n_ = lb.num(g(r, "Microlitter_count_water"))
            if g(r, "Microlitter_count_water_VOR") != "":
                vor = True
            if c == "ambiguous" and (n_ is None or n_ != 0):
                amb_caught = True
            if c == "plastic":
                plast.append((n_, lb.num(g(r, "*Microlitter_weight_water [g]")),
                              h03_bounds(g(r, "Microlitter_size_class_water"))))

        def dropped(why):
            drop[why] = drop.get(why, 0) + 1
        if vor:
            # Microlitter_count_water_VOR is FRACTIONAL (measured: 1.9768,
            # 0.975 ...) under a P01 term that calls it a count; neither the
            # vocabulary nor the file says per what. Not read.
            dropped("vor_count_undefined")
            continue
        if not plast:
            dropped("no_plastic_line")
            continue
        if amb_caught:
            dropped("ambiguous_line_with_catch")
            continue
        if any(n_ is None for n_, _w, _b in plast):
            dropped("plastic_line_without_count")
            continue
        total = sum(n_ for n_, _w, _b in plast)
        wts = [w for _n, w, _b in plast]
        wsum = sum(wts) if all(w is not None for w in wts) else None
        width = lb.num(g(r0, "mouth_width [cm]"))
        trk = lb.num(g(r0, "SampTrkLen [km]"))
        vol = lb.num(g(r0, "VolWBodySamp [m3]"))
        area_km2 = (width / 1e5) * trk if width and trk else None
        v = {}
        if area_km2:
            v["n_km2"] = total / area_km2
            v["g_km2"] = None if wsum is None else wsum / area_km2
            for ch, (lo, hi) in SIZE_CH.items():
                inside = [(b, n_) for n_, _w, b in plast if b is not None and
                          b[0] >= lo and (hi is None or b[1] <= hi)]
                if tiles([b for b, _n in inside], lo, hi):
                    v[ch[4:]] = sum(n_ for _b, n_ in inside) / area_km2
        if vol:
            v["n_m3"] = total / vol
        if not v:
            dropped("no_area_and_no_volume")
            continue
        ts = g(r0, "yyyy-mm-ddThh:mm:ss.sss").strip()
        tq = 0
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2})"
                         r"(?::(\d{2})(?:\.\d+)?)?)?", ts)
        if not m or not bool(cm.valid_date(*(int(x) for x in m.groups()[:3]))):
            refuse(f"EMODnet sample {k}: time {ts!r}")
        y_, mo_, d_ = (int(x) for x in m.groups()[:3])
        if m.group(4) is None:
            # a DATE without a time of day: noon UTC, flagged (measured: the
            # spreadsheet holds a few date-only samples)
            t, tq = t_of(y_, mo_, d_)
            counts["emodnet_date_only"] = counts.get("emodnet_date_only", 0) + 1
        else:
            t = f10b.seconds_since_epoch(dt.datetime(
                y_, mo_, d_, int(m.group(4)), int(m.group(5)),
                int(m.group(6) or 0)))
        gearl = g(r0, "Instrument / gear type")
        gear = (4 if "continuous water samplers" in gearl else
                2 if "neuston nets" in gearl else
                7 if "plankton nets" in gearl else 0)
        mn = lb.num(g(r0, "MinDepBelowSeaSurf [m]"))
        mx = lb.num(g(r0, "MaxDepBelowSeaSurf [m]"))
        v["net_depth_m"] = (mx - mn) if (gear in (2, 7) and mn is not None
                                         and mx is not None and mx > mn) \
            else None
        v["sample_depth_m"] = mn
        v["mesh_um"] = lb.num(g(r0, "Mesh_size [micrometres]"))
        v["wind_ms"] = lb.num(g(r0, "WindSpd [m/s]"))
        study = g(r0, "Cruise name") or g(r0, "Data set name")
        plat = f"emodnet|{study}|{g(r0, 'CSR platform name')}"
        out.append(cand("emodnet", f"sample {'/'.join(k)}", t, tq,
                        lb.num(g(r0, "Latitude [degrees_north]")),
                        lb.num(g(r0, "Longitude [degrees_east]")), plat, gear,
                        study, **v))
    counts["read"]["emodnet"] = len(order)
    counts["emodnet_line_classes"] = cls_count
    counts["emodnet_samples_dropped"] = drop
    return out, mapping


# ================================================================== dedup ==
def dedup(rows, counts):
    """Precedence de-duplication (plan D2). `rows` arrive in SOURCES order.

    A row is a copy of an EARLIER-SOURCE row on the same UTC day (the same
    year-month when either is month-only) within 0.01 deg in latitude AND
    longitude. Rows of one source are never compared with each other: the
    rule is about the same tow compiled by two producers.
    """
    kept = []
    index = {}
    pairs = {}
    for r in rows:
        hit = None
        for k in index.get(r["ym"], ()):
            if k["src"] == r["src"]:
                continue
            if abs(k["lat"] - r["lat"]) > DEDUP_DEG or \
                    abs(k["lon"] - r["lon"]) > DEDUP_DEG:
                continue
            if k["day"] is not None and r["day"] is not None and \
                    k["day"] != r["day"]:
                continue
            hit = k
            break
        if hit is not None:
            key = f"{hit['src']} > {r['src']}"
            pairs[key] = pairs.get(key, 0) + 1
            r["dup_of"] = hit["key"]
            continue
        kept.append(r)
        index.setdefault(r["ym"], []).append(r)
    counts["dedup"] = pairs
    return kept


# ==================================================== NCEI: methods a, b, c ==
def _ncei_parse(x):
    y, m, d, _tod = ncei_date(x["Sample Date"])
    return {"day": (y, m, d), "lat": float(x["Latitude (degree)"]),
            "lon": float(x["Longitude (degree)"]),
            "v": float(x["Microplastics Measurement"]),
            "method": x["Sampling Method"].strip(),
            "study": x["Short Reference"].strip(),
            "org": x["Organization"].strip(),
            "acc7": x["NCEI Accession Number"].strip().zfill(7),
            "uid": x["Unique ID"].strip(),
            "mesh_mm": lb.num(x["Mesh Size (mm)"]),
            "depth": lb.num(x["Water Sample Depth (m)"])}


def areal_pairs(nrows, sources):
    """NCEI rows of one (study, method) vs the areal rows of sources 1-2:
    the UNIQUE pairs (each side matched once) -> [(n_m3, n_km2, src key)]."""
    by_ym = {}
    for j, s in enumerate(sources):
        if s["v"].get("n_km2") is None:
            continue
        by_ym.setdefault(s["ym"], []).append(j)
    cand_of = {}
    hits = {}
    for i, r in enumerate(nrows):
        y, m, d = r["day"]
        c = []
        for j in by_ym.get((y, m), ()):
            s = sources[j]
            if s["day"] is not None and s["day"] != r["day"]:
                continue
            if abs(s["lat"] - r["lat"]) <= DEDUP_DEG and \
                    abs(s["lon"] - r["lon"]) <= DEDUP_DEG:
                c.append(j)
        if c:
            cand_of[i] = c
            for j in c:
                hits[j] = hits.get(j, 0) + 1
    out = []
    for i, c in cand_of.items():
        if len(c) == 1 and hits[c[0]] == 1:
            s = sources[c[0]]
            out.append((nrows[i]["v"], s["v"]["n_km2"],
                        f"{s['src']}: {s['study']}"))
    return out


def depth_of(pairs):
    """Measured ratio n_km2 / n_m3 (as metres of net depth) over the pairs
    where both are non-zero -> (median depth m, cv, n, zero mismatches)."""
    r = [a / b / 1e6 for b, a, *_ in pairs if a and b]
    zm = sum(1 for b, a, *_ in pairs if (a == 0) != (b == 0))
    if not r:
        return None, None, 0, zm
    return float(np.median(r)), _cv(r), len(r), zm


def _ordinal(day):
    return dt.date(*day).toordinal()


def toc_near(r, sources, days, deg):
    """Sources-1-2 rows within `days` UTC days and `deg` degrees of an NCEI
    row (any of a 2018 event's start, end and midpoint), nearest first.
    A month-only row (the 2018 compilation) is near when the NCEI date +-
    `days` falls in its month."""
    o = _ordinal(r["day"])
    near = []
    for s in sources:
        if s["day"] is not None:
            dd = abs(_ordinal(s["day"]) - o)
            if dd > days:
                continue
        else:
            yms = {dt.date.fromordinal(o + k).timetuple()[:2]
                   for k in range(-days, days + 1)}
            if tuple(s["ym"]) not in yms:
                continue
            dd = 0
        best = None
        for la, lo in [(s["lat"], s["lon"])] + s.get("alt", []):
            dl, dn = abs(la - r["lat"]), abs(lo - r["lon"])
            if dl <= deg and dn <= deg:
                d2 = dl * dl + dn * dn
                best = d2 if best is None else min(best, d2)
        if best is not None:
            near.append((dd, best, s))
    near.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in near]


EGGER_DEFS = (("n_km2 (as caught)", "n_km2"),
              ("n_km2_wc (producer-corrected)", "n_km2_wc"),
              ("micro 0.05-0.5 cm, producer-corrected", "micro_wc"))


def egger_pairs(rs, sources):
    """(n_m3, areal value) pairs for Egger rows matched to ONE Ocean Cleanup
    tow, per (source, definition) -> {"<source>: <definition>": pairs}. The
    2018 events and the 2024 rows are measured SEPARATELY (the compilation's
    'This study' rows are the 2018 events again): NCEI's Egger value is one
    quantity over one depth, and pooling two producers' definitions would
    hide which. Used to MEASURE the depth NCEI's compiler assumed."""
    out = {}
    for src in ("lebreton2018", "lebreton2024"):
        pool = [s for s in sources if s["src"] == src]
        used = {}
        for r in rs:
            near = toc_near(r, pool, EGGER_DAYS, EGGER_DEG)
            if len(near) == 1:
                used.setdefault(id(near[0]), []).append((r, near[0]))
        for k, ch in EGGER_DEFS:
            pr = []
            for lst in used.values():
                if len(lst) != 1:
                    continue
                r, s = lst[0]
                a = s.get("micro_wc") if ch == "micro_wc" else s["v"].get(ch)
                if a is not None:
                    pr.append((r["v"], a, f"{s['src']}: {s['study']}"))
            out[f"{src}: {k}"] = pr
    return out


def ncei_rows(water, acc_tables, sources, counts):
    """The NCEI water rows -> candidates, and the per-study depth table.

    `acc_tables` = {acc7: {file name: tables}} for the accessions read;
    `sources` = the candidate rows of sources 1-2 (method b's areal values).
    """
    parsed = [_ncei_parse(x) for x in water]
    by_name = {}
    groups = {}
    for r in parsed:
        if r["study"] in NCEI_DROP_BY_NAME:
            by_name.setdefault(r["study"], []).append(r)
            continue
        groups.setdefault((r["study"], r["method"]), []).append(r)
    table = []
    out = []
    unknown_gear = {}
    # --- dropped by name: measured anyway, so the depth and the loss are on
    #     the record (how many had NO copy in sources 1-2)
    dn = {}
    for study, rs in sorted(by_name.items()):
        for meth in sorted({r["method"] for r in rs}):
            sub = [r for r in rs if r["method"] == meth]
            pairs = areal_pairs(sub, sources)
            dep, cv, n, zm = depth_of(pairs)
            with_copy = sum(1 for r in sub if _has_copy(r, sources))
            table.append({"study": study, "sampling_method": meth,
                          "rows": len(sub), "method": "dropped by name",
                          "why": NCEI_DROP_BY_NAME[study],
                          "depth_used_m": None,
                          "implied_depth_m": dep, "n_matched": n,
                          "ratio_cv": cv, "zero_mismatch": zm,
                          "areal_source": sorted({p[2] for p in pairs}),
                          "rows_with_a_copy_in_sources_1_2": with_copy})
        dn[study] = len(rs)
    counts["ncei_dropped_by_name"] = dn
    # --- every other (study, sampling method)
    egger_drop = {}
    egger_kept = {}
    egger_audit = []
    for (study, meth), rs in sorted(groups.items()):
        egger = None
        if study in EGGER_REFS:
            # THE MATCH DROP: measured on the whole group first (the depth
            # comes from the copies), then only the rows with no Ocean
            # Cleanup copy go on
            kept_rs, n_drop = [], 0
            for r in rs:
                near = toc_near(r, sources, EGGER_DAYS, EGGER_DEG)
                if near:
                    n_drop += 1
                    key = f"{near[0]['src']} > ncei (Egger match)"
                    egger_drop[key] = egger_drop.get(key, 0) + 1
                else:
                    kept_rs.append(r)
                    y = str(r["day"][0])
                    egger_kept.setdefault(study, {})
                    egger_kept[study][y] = egger_kept[study].get(y, 0) + 1
                    loose = toc_near(r, sources, *EGGER_AUDIT)
                    if loose:
                        s0 = loose[0]
                        egger_audit.append({
                            "uid": r["uid"], "study": study,
                            "day": "%04d-%02d-%02d" % r["day"],
                            "lat": r["lat"], "lon": r["lon"],
                            "near": f"{s0['src']} {s0['key']}",
                            "near_day": ("%04d-%02d-%02d" % s0["day"]
                                         if s0["day"] else "%04d-%02d" %
                                         tuple(s0["ym"])),
                            "near_lat": s0["lat"], "near_lon": s0["lon"]})
            ep = egger_pairs(rs, sources)
            best = None
            cands_ = []
            for k, pr in ep.items():
                dep, cv, n, zm = depth_of(pr)
                if n:
                    cands_.append({"definition": k, "implied_depth_m": dep,
                                   "ratio_cv": cv, "n_matched": n})
                # the best-measured: enough pairs first, then the lowest cv
                if n and (best is None or
                          (n < METHOD_B_MIN, cv, -n) <
                          (best[3] < METHOD_B_MIN, best[2], -best[3])):
                    best = (k, dep, cv, n, zm, pr)
            egger = {"rows": len(rs), "dropped_by_match": n_drop,
                     "kept": len(kept_rs), "best": best,
                     "candidates": cands_}
            rs = kept_rs
            if not rs:
                table.append({"study": study, "sampling_method": meth,
                              "rows": egger["rows"],
                              "method": "all dropped by match",
                              "dropped_by_match": n_drop, "kept": 0,
                              "depth_used_m": None,
                              "depth_candidates": cands_,
                              **_egger_depth_fields(best)})
                continue
        gear = NCEI_GEAR.get(meth)
        if gear is None:
            unknown_gear[meth] = unknown_gear.get(meth, 0) + len(rs)
            gear = 0
        entry = {"study": study, "sampling_method": meth, "rows": len(rs),
                 "accessions": sorted({r["acc7"] for r in rs})}
        n_km2 = [None] * len(rs)
        n_m3 = [None] * len(rs)
        depth_used = None
        if gear not in AREAL_GEAR:
            entry.update(method="volumetric", depth_used_m=None)
            n_m3 = [r["v"] for r in rs]
        else:
            specs = [(a, AREAL_ACCESSIONS[a]) for a in entry["accessions"]
                     if a in AREAL_ACCESSIONS
                     and AREAL_ACCESSIONS[a]["study"] == study]
            if specs:
                # METHOD (a): the study's own accession, matched 1:1. n_km2 is
                # the TOTAL over every size column the accession gives (as
                # caught, all sizes); the depth NCEI's compiler assumed is
                # measured on whichever column (or the total) its per-m3
                # value is a constant multiple of, and recorded.
                matched = 0
                pairs = []
                for acc7, spec in specs:
                    tabs = acc_tables.get(acc7)
                    if not tabs:
                        refuse(f"accession {acc7} ({study}) was not read")
                    rx = re.compile(spec["file"])
                    cands = [n for n in sorted(tabs) if rx.search(n)
                             and spec["sheet"] in tabs[n]]
                    if len(cands) != 1:
                        refuse(f"accession {acc7}: {len(cands)} data files "
                               f"match {spec['file']!r} with sheet "
                               f"{spec['sheet']!r} ({sorted(tabs)})")
                    arows = areal_table(acc7, tabs[cands[0]], spec)
                    idx = [i for i, r in enumerate(rs) if r["acc7"] == acc7]
                    m = match_one_to_one(
                        [(rs[i]["day"], rs[i]["lat"], rs[i]["lon"], rs[i]["v"])
                         for i in idx], arows)
                    if len(m) != len(idx):
                        refuse(f"accession {acc7} ({study}): only {len(m)} of "
                               f"{len(idx)} NCEI rows match a row of "
                               f"{cands[0]} by day and 0.01 deg — the 1:1 "
                               f"match method (a) rests on fails")
                    for k, i in enumerate(idx):
                        a = arows[m[k]]
                        n_km2[i] = a[2]
                        pairs.append((rs[i]["v"], a[2], a[3]))
                    matched += len(m)
                    entry.setdefault("files", []).append(f"{acc7}/{cands[0]}")
                best = None
                comps = sorted({c for _v, _a, cp in pairs for c in cp})
                for c in [None] + comps:
                    rr = []
                    for v, a, cp in pairs:
                        x = a if c is None else cp.get(c)
                        if x and v:
                            rr.append(x / v / 1e6)
                    if len(rr) >= 2:
                        cv = _cv(rr)
                        if best is None or cv < best[1]:
                            best = (float(np.median(rr)), cv, c or "total")
                zm = sum(1 for v, a, _c in pairs if a is not None
                         and (a == 0) != (v == 0))
                med, cv, which = best if best else (None, None, None)
                if med is not None and not (A_DEPTH_RANGE[0] <= med
                                            <= A_DEPTH_RANGE[1]):
                    refuse(f"{study}: the accession's areal value implies a "
                           f"net depth of {med:.4g} m against NCEI's per-m3 "
                           f"value — outside {A_DEPTH_RANGE} m, so its unit "
                           f"cannot be what the header says")
                depth_used = med if (cv is not None and cv < METHOD_B_CV) \
                    else None
                entry.update(method="a", n_matched=matched,
                             implied_depth_m=med, ratio_cv=cv,
                             ncei_value_is=which, zero_mismatch=zm,
                             depth_used_m=depth_used,
                             depth_note=(
                                 "net_depth_m carries the depth NCEI's "
                                 "compiler assumed (one ratio, cv < 1 %)"
                                 if depth_used is not None else
                                 "no single assumed depth (cv >= 1 %): "
                                 "net_depth_m is NaN"))
            else:
                if egger is not None:
                    # the kept rows have, by construction, no copy to pair
                    # with: the depth is MEASURED on the group's copies
                    b_ = egger["best"]
                    pairs = b_[5] if b_ else []
                    entry.update(rows=egger["rows"], kept=egger["kept"],
                                 dropped_by_match=egger["dropped_by_match"],
                                 areal_definition=b_[0] if b_ else None,
                                 depth_candidates=egger["candidates"])
                else:
                    pairs = areal_pairs(rs, sources)
                dep, cv, n, zm = depth_of(pairs)
                ok = n >= METHOD_B_MIN and cv is not None and cv < METHOD_B_CV
                if egger is not None and egger["best"]:
                    rr = [a / b / 1e6 for b, a, *_ in pairs if a and b]
                    entry["share_within_1pct_of_median"] = (
                        float(np.mean([abs(x / dep - 1) <= 0.01 for x in rr]))
                        if rr else None)
                    if not egger["best"][0].endswith("n_km2 (as caught)"):
                        # the depth multiplies back to a quantity no channel
                        # holds (measured: NCEI's Egger value is the 2024
                        # wind-corrected 0.5-5 mm count over 0.075 m), so the
                        # rows stay per m3 whatever the cv
                        ok = False
                        entry["why_per_m3"] = (
                            f"the depth multiplies back to "
                            f"{egger['best'][0]}, which is not a channel")
                entry.update(implied_depth_m=dep, n_matched=n, ratio_cv=cv,
                             zero_mismatch=zm,
                             areal_source=sorted({p[2] for p in pairs}))
                if ok:
                    depth_used = dep
                    entry.update(method="b", depth_used_m=dep)
                    n_km2 = [r["v"] * dep * 1e6 for r in rs]
                else:
                    entry.update(method="c", depth_used_m=None,
                                 why=(f"{n} matched row(s) (< {METHOD_B_MIN})"
                                      if n < METHOD_B_MIN else
                                      f"ratio cv {cv:.4f} >= {METHOD_B_CV}"
                                      if cv >= METHOD_B_CV else
                                      entry.get("why_per_m3", "")))
                    n_m3 = [r["v"] for r in rs]
        table.append(entry)
        for i, r in enumerate(rs):
            t, q = t_of(*r["day"])
            out.append(cand(
                "ncei", f"ncei {r['uid']}", t, q, r["lat"], r["lon"],
                f"ncei|{study}|{r['org']}", gear, study,
                n_km2=n_km2[i], n_m3=n_m3[i],
                mesh_um=None if r["mesh_mm"] is None else r["mesh_mm"] * 1000,
                net_depth_m=depth_used, sample_depth_m=r["depth"]))
    counts["read"]["ncei"] = len(water)
    counts["ncei_unknown_sampling_method"] = unknown_gear
    counts["ncei_egger_dropped_by_match"] = egger_drop
    counts["ncei_egger_kept_by_year"] = {
        f"{st} {y}": n for st, d in sorted(egger_kept.items())
        for y, n in sorted(d.items())}
    counts["ncei_egger_kept_near_toc_loose"] = egger_audit
    by_m = {}
    for e in table:
        by_m[e["method"]] = by_m.get(e["method"], 0) + e["rows"]
    counts["ncei_rows_by_method"] = by_m
    return out, table


def _egger_depth_fields(best):
    if not best:
        return {"implied_depth_m": None, "n_matched": 0, "ratio_cv": None}
    k, dep, cv, n, zm, pr = best
    return {"areal_definition": k, "implied_depth_m": dep, "n_matched": n,
            "ratio_cv": cv, "zero_mismatch": zm,
            "areal_source": sorted({p[2] for p in pr})}


def _has_copy(r, sources):
    y, m, d = r["day"]
    for s in sources:
        if s["ym"] != (y, m):
            continue
        if s["day"] is not None and s["day"] != r["day"]:
            continue
        if abs(s["lat"] - r["lat"]) <= DEDUP_DEG and \
                abs(s["lon"] - r["lon"]) <= DEDUP_DEG:
            return True
    return False


# ================================================================ packing ==
def to_matrix(rows, counts):
    """Candidates -> (values float64 (n, C) in STORED units, oob counts)."""
    v = np.full((len(rows), len(CHANNELS)), np.nan)
    neg = {}
    for i, r in enumerate(rows):
        for name, ch in (("n_km2", "log_n_km2"), ("g_km2", "log_g_km2"),
                         ("n_m3", "log_n_m3"), ("n_km2_wc", "log_n_km2_wc"),
                         ("n_km2_micro", "log_n_km2_micro"),
                         ("n_km2_meso", "log_n_km2_meso"),
                         ("n_km2_macro", "log_n_km2_macro"),
                         ("n_km2_mega", "log_n_km2_mega")):
            x = r["v"].get(name)
            if x is None:
                continue
            if x < 0:
                neg[ch] = neg.get(ch, 0) + 1
                v[i, COL[ch]] = -1.0          # out of bounds below: counted
            else:
                v[i, COL[ch]] = np.log10(1.0 + x)
        for ch in ("mesh_um", "net_depth_m", "sample_depth_m", "wind_ms"):
            x = r["v"].get(ch)
            if x is not None:
                v[i, COL[ch]] = x
    return v


HUB_PENDING_TRIES = 12
HUB_PENDING_SLEEP = 30.0


def hub_pending(path):
    """Is this file the ArcGIS Hub's 'being generated' message, not data?"""
    if os.path.getsize(path) > 4096:
        return False
    with open(path, "rb") as fh:
        head = fh.read(4096)
    return head.lstrip().startswith(b"{") and b"being generated" in head


# ================================================================= mirror ==
def mirror_name(url):
    """A URL -> its path in a `--source-dir` mirror (the smoke writes these).

    host + path; a directory URL ends in `index.html`; a last segment with no
    extension gets `__body` (so `/articles/1` and `/articles/1/files` can both
    be files); a query string is appended, sanitised.
    """
    import urllib.parse
    u = urllib.parse.urlsplit(url)
    path = u.path or "/"
    if path.endswith("/"):
        path += "index.html"
    elif "." not in path.rsplit("/", 1)[-1]:
        path += "__body"
    name = u.netloc + path
    if u.query:
        name += "__" + re.sub(r"[^A-Za-z0-9._-]", "_", u.query)
    return name


# ================================================================ adapter ==
class MPlasticAdapter(f10b.SourceAdapter):
    store = "mplastic"
    title = ("Floating plastic at the sea surface: one row per net tow, water "
             "sample or aircraft survey line (The Ocean Cleanup, NOAA NCEI, "
             "EMODnet), de-duplicated across producers")
    family = "13"
    distribution = "public"
    licence = {
        "name": "CC BY 4.0 (The Ocean Cleanup; EMODnet Chemistry) and CC0 1.0 "
                "(NOAA NCEI)",
        "redistribution": "attribution", "derived_works": "free",
        "attribution": (
            "The Ocean Cleanup: Lebreton et al. 2018, Sci. Rep. 8:4666, data "
            "doi:10.6084/m9.figshare.5873142, and Lebreton et al. 2024, data "
            "doi:10.6084/m9.figshare.25674120 (CC BY 4.0); NOAA NCEI Global "
            "Marine Microplastics, accession 0306475, doi:10.25921/v8gj-k896 "
            "(CC0 1.0), and the studies it compiles; EMODnet Chemistry, "
            "floating micro-litter aggregated dataset v2026, unrestricted "
            "(CC BY 4.0, citation mandatory)")}
    time_dtype = "int32"
    platform_meta = False
    credentials = ()
    channels = CHANNELS
    # FOOTPRINT (E-078 §2: log2 of the support against the family's 27.83 km /
    # 5-day unit). A row is a mean over the line a net was towed along, so the
    # spatial support is the TOW LENGTH, not the net's 0.9 m mouth (its swept
    # area, ~0.002-0.007 km2, would claim a point the tow never was).
    # Measured: EMODnet's track lengths have their mode at 1.85 km (one
    # nautical mile), The Ocean Cleanup's 2015 manta tows a median of 7.3 km
    # and 1.0 h; SEA's tows, the largest single source, are standard half-hour
    # tows. A typical row is taken as 2 km and 1 hour.
    log2_fp = float(np.log2(2.0 / 27.83))                # -3.80
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))         # -6.91
    per_year = False
    first_year = 1972
    qc_policy = (
        "qc bits 0-3 = gear (" + "; ".join(f"{k} {v}" for k, v in
                                          GEAR_CODES.items()) +
        "); bit 4 (16): time of day not given, placed at 12:00 UTC; bit 5 "
        "(32): day not given, placed mid-month. Values outside the physical "
        "bounds -> NaN, counted. A row with no concentration is dropped.")
    sources = (lb.API_FILES, FIG2024_ARTICLE, NCEI_CSV,
               "https://www.ncei.noaa.gov/archive/accession/0306475",
               EMODNET_ZIP)
    verified = (
        "2026-10-11 from the sandbox: both figshare items (API listings, the "
        "2024 description that defines every column, the 2018 traps), the "
        "NCEI Hub CSV (29,076 rows, 16,988 ocean water, every time of day "
        "05:00 or 06:00 AM), the data files of every NCEI accession the "
        "surface-net rows cite (headers read for areal columns), the EMODnet "
        "v2026 zip (32,067 lines, 3,385 samples) and the NVS H01/H05/P01 "
        "vocabulary definitions")
    notes = (
        "Concentrations are stored as log10(1 + value): float16 cannot hold "
        "a busy tow's pieces/km2. One streamed pass (per_year = False); the "
        "probe builds the whole store and keeps its month. The per-study "
        "depth table (NCEI methods a/b/c) and the EMODnet type mapping are in "
        "store.json's counts and in ml/family1/probes/.")
    smoke_window = ("1972-01-01", "2023-12-31")
    smoke_probe_month = "2015-08"
    fetch_month_scope = "archive"

    # ------------------------------------------------------------- reading --
    def _path(self, ctx, url):
        name = mirror_name(url)
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "mplastic", name)
            if not os.path.exists(p):
                refuse(f"{url} is not in the source mirror ({p})")
            seen = getattr(ctx, "_mplastic_counted", None)
            if seen is None:
                seen = ctx._mplastic_counted = set()
            if p not in seen:
                ctx.count_bytes(os.path.getsize(p))
                seen.add(p)
            return p
        p = os.path.join(ctx.scratch, "mplastic", name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
        for i in range(HUB_PENDING_TRIES):
            got = f10b.fetch_first([url], path=p,
                                   attempts=int(getattr(ctx.a, "attempts", 3)
                                                or 3))
            if got is None:
                refuse(f"{url} answered 404 — every source of this store is "
                       f"needed whole")
            if os.path.getsize(p) == 0:
                refuse(f"{url} answered an EMPTY body")
            if not hub_pending(p):
                return p
            # THE HUB'S EXPORT CAN BE "BEING GENERATED" (measured 2026-10-11:
            # a 183-byte JSON message with status 200 in place of the CSV).
            # It is a wait, not a file: retried, and refused if it persists.
            os.remove(p)
            print(f"  mplastic: {url} — the export is being generated; "
                  f"retry {i + 1}/{HUB_PENDING_TRIES} in "
                  f"{HUB_PENDING_SLEEP:.0f} s", flush=True)
            import time
            time.sleep(HUB_PENDING_SLEEP)
        refuse(f"{url} kept answering 'download file is being generated' "
               f"({HUB_PENDING_TRIES} tries)")

    def _bytes(self, ctx, url):
        with open(self._path(ctx, url), "rb") as fh:
            return fh.read()

    def _json(self, ctx, url):
        return json.loads(self._bytes(ctx, url))

    def lebreton2018(self, ctx):
        files = lb.resolve(self._json(ctx, lb.API_FILES))
        paths = {n: self._path(ctx, f["download_url"])
                 for n, f in files.items()}
        return lb.read_all(paths), files

    def lebreton2024(self, ctx):
        art = self._json(ctx, FIG2024_ARTICLE)
        lebreton2024_check_description(art)
        files = {f["name"]: f for f in self._json(ctx, FIG2024_FILES)}
        if FIG2024_NAME not in files:
            refuse(f"figshare {FIG2024} lists no {FIG2024_NAME} "
                   f"({sorted(files)})")
        return lebreton2024_read(self._path(ctx,
                                            files[FIG2024_NAME]["download_url"]))

    def _listing(self, ctx, url):
        html = self._bytes(ctx, url).decode("utf-8", "replace")
        return [h for h in re.findall(r'<a href="([^"?/][^"]*)">', html)
                if not h.startswith("http")]

    def accession(self, ctx, acc7):
        """Every data file of one NCEI accession -> {file name: tables}."""
        land = self._bytes(ctx, NCEI_LANDING.format(acc7=acc7)).decode(
            "utf-8", "replace")
        m = re.search(r"(arc0\d+)/" + acc7, land)
        if not m:
            refuse(f"accession {acc7}: the landing page names no archive "
                   f"directory")
        base = NCEI_ARCHIVE.format(arc=m.group(1), acc7=acc7)
        vers = sorted({h.rstrip("/") for h in self._listing(ctx, base)
                       if re.fullmatch(r"\d+\.\d+/", h)},
                      key=lambda v: tuple(int(x) for x in v.split(".")))
        if not vers:
            refuse(f"accession {acc7}: {base} lists no version directory")
        todo = [f"{base}{vers[-1]}/data/0-data/"]
        out = {}
        import urllib.parse
        while todo:
            d = todo.pop()
            for h in self._listing(ctx, d):
                if h.endswith("/"):
                    todo.append(d + h)
                    continue
                name = urllib.parse.unquote(h)
                if ACC_SKIP.search(name) or not ACC_DATA.search(name):
                    continue
                out[name] = read_table(self._path(ctx, d + h))
        if not out:
            refuse(f"accession {acc7}: no data file under {todo or base}")
        return out

    def emodnet(self, ctx):
        p = self._path(ctx, EMODNET_ZIP)
        with zipfile.ZipFile(p) as z:
            if EMODNET_TXT not in z.namelist():
                refuse(f"{EMODNET_ZIP} holds no {EMODNET_TXT}")
            return z.read(EMODNET_TXT).decode("utf-8")

    # ------------------------------------------------------------ building --
    def build(self, ctx):
        """Every source -> (kept candidates in yield order, counts)."""
        counts = {"read": {}}
        L, _f = self.lebreton2018(ctx)
        r1 = lebreton2018_rows(L, counts)
        r1b = lebreton2018_hist_rows(L, counts)
        r2 = lebreton2024_rows(self.lebreton2024(ctx), L, counts)
        water = ncei_read(self._bytes(ctx, NCEI_CSV).decode("utf-8-sig"),
                          counts)
        accs = sorted({x["NCEI Accession Number"].strip().zfill(7)
                       for x in water
                       if x["Short Reference"].strip() not in NCEI_DROP_BY_NAME
                       and NCEI_GEAR.get(x["Sampling Method"].strip())
                       in AREAL_GEAR})
        acc_tables = {}
        hints = {}
        for i, a in enumerate(accs, 1):
            acc_tables[a] = self.accession(ctx, a)
            if a not in AREAL_ACCESSIONS and a not in AREAL_REJECTED:
                h = [f"{n}: {x}" for n, t in acc_tables[a].items()
                     for x in areal_hints(t)]
                if h:
                    hints[a] = h
            if i % 10 == 0:
                ctx.prog.item(f"mplastic accessions {i}/{len(accs)}")
        if hints:
            refuse(f"NCEI accession(s) not in AREAL_ACCESSIONS hold what look "
                   f"like areal columns — a new method-(a) source to decide "
                   f"on: {hints}")
        counts["ncei_accessions_read"] = len(accs)
        r3, table = ncei_rows(water, acc_tables, r1 + r1b + r2, counts)
        H, lines = emodnet_read(self.emodnet(ctx), counts)
        r4, mapping = emodnet_rows(H, lines, counts)
        allrows = r1 + r1b + r2 + r3 + r4
        for i, r in enumerate(allrows):
            r["seq"] = i
        kept = dedup(allrows, counts)
        counts["ncei_depth_table"] = table
        counts["emodnet_type_mapping"] = [{"type | polymer": k, "class": v}
                                          for k, v in sorted(mapping.items())]
        rank = {s: i for i, s in enumerate(SOURCES)}
        kept.sort(key=lambda r: (r["t"], rank[r["src"]], r["seq"]))
        return kept, counts

    def _rows(self, ctx, t_lo, t_hi):
        kept, counts = self.build(ctx)
        inside = [r for r in kept if t_lo <= r["t"] <= t_hi]
        counts["rows_outside_window"] = len(kept) - len(inside)
        v = to_matrix(inside, counts)
        oob = self.mask_bounds(v)
        if oob:
            f10b._merge_counts(counts, {"out_of_bounds": oob})
        alive = np.isfinite(v[:, [COL[c] for c in CONC]]).any(axis=1) \
            if len(inside) else np.zeros(0, bool)
        counts["rows_no_value"] = int((~alive).sum())
        keep = [r for r, a in zip(inside, alive) if a]
        v = v[alive] if len(inside) else v
        kc, gc, fill = {}, {}, {}
        for r in keep:
            kc[r["src"]] = kc.get(r["src"], 0) + 1
            g = str(r["qc"] & 15)
            gc[g] = gc.get(g, 0) + 1
        for j, n in enumerate(NAMES):
            fill[n] = int(np.isfinite(v[:, j]).sum()) if len(keep) else 0
        counts["kept_by_source"] = kc
        counts["kept_by_gear"] = gc
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
        files = lb.resolve(self._json(ctx, lb.API_FILES))
        art = self._json(ctx, FIG2024_ARTICLE)
        lebreton2024_check_description(art)
        f24 = {f["name"]: f for f in self._json(ctx, FIG2024_FILES)}
        if FIG2024_NAME not in f24:
            refuse(f"figshare {FIG2024} lists no {FIG2024_NAME}")
        c = {"read": {}}
        water = ncei_read(self._bytes(ctx, NCEI_CSV).decode("utf-8-sig"), c)
        H, lines = emodnet_read(self.emodnet(ctx), c)
        return {"dataset": "mplastic: Lebreton 2018 + 2024, NCEI, EMODnet",
                "lebreton2018_files": {n: f["id"] for n, f in files.items()},
                "lebreton2024_file": {FIG2024_NAME: f24[FIG2024_NAME]["id"]},
                "ncei_rows_total": c["ncei_rows_total"],
                "ncei_water_rows": len(water),
                "ncei_marine_setting": c["ncei_marine_setting"],
                "emodnet_lines": len(lines)}

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
        from family1.adapters import _mplastic_smoke as sm
        return sm.make_smoke_sources(root, d_lo, d_hi)


ADAPTER = MPlasticAdapter
