"""Sentinel-2 floating-debris indices over coastal water, one 20 m frame per
ACQUISITION — family 1.3 phase C, the PILOT (E-092, D13).

PLAIN ENGLISH. Floating plastic, seaweed and foam make the sea brighter in the
near infrared than clean water is. Four published indices turn that into one
number per 20 m pixel: the floating debris index (FDI, Biermann et al. 2020),
the floating algae index (FAI, Hu 2009), the vegetation index (NDVI) and the
plastic index (PI, Themistocleous et al. 2020). This store holds those four
and the four surface reflectances they are made from (red B04, red-edge B06,
near infrared B08, shortwave infrared B11) for every pixel of WATER in a
Sentinel-2 picture — on the tile's own map grid, one frame per picture. It is
NOT the coast of the world: it is a pilot over the places where the answer is
known, so the store can be falsified before anything large is built
(plan §6):

  (a) the 63 scenes of MARIDA (the Marine Debris Archive, Kikaki et al. 2022)
      — 59 are on Earth Search's Level-1C catalogue, 4 exist only in the
      Copernicus Data Space, whose downloads need an account this build does
      not have (measured 2026-10-13: HTTP 401 "Token not found" on both the
      download and the zipper endpoints); those four are COUNTED as
      `absent_upstream`, never skipped;
  (b) every acquisition of tile 35SMD (the Gulf of Gera, Lesvos — where the
      University of the Aegean's Plastic Litter Project moors KNOWN plastic
      targets) inside the build window whose Level-1C cloud cover is below
      20 %, plus every Plastic Litter Project campaign date whatever its
      cloud (two of the 32 are above 20 %: 2021-07-06 and 2021-09-14).

SOURCES, MEASURED FROM THE SANDBOX 2026-10-13 (keyless, all of them):
  * catalogue: Earth Search STAC v1, collections `sentinel-2-l1c` and
    `sentinel-2-l2a` (`grid:code` = `MGRS-<tile>`), paged and checked against
    `numberMatched` by `_stac.stac_search`;
  * Level-1C bands: JPEG 2000 in the `sentinel-s2-l1c` bucket over plain
    HTTPS, rebuilt into a minimal .SAFE exactly as the co-location probe does
    (`ml/family1/colocation_probe.py::fetch_safe`). **The bucket is NOT
    requester-pays**, which settles plan §7 open item 5: anonymous
    `GET /?requestPayment` answers `<Payer>BucketOwner</Payer>`, and the same
    proxy that let it through refuses an anonymous read of a bucket that IS
    requester-pays (`usgs-landsat`: HTTP 403 "Anonymous users cannot invoke
    requests against Requester Pays buckets"), so the proxy injects no
    credential and a GitHub-hosted runner reads it the same way. The index
    stage re-asks `?requestPayment` on every build and refuses if the answer
    changes;
  * scene classification (SCL, 20 m) from the matching `sentinel-2-l2a` item's
    COG in `sentinel-cogs`, used ONLY as the water / cloud mask;
  * MARIDA's scene list: Zenodo record 5151941's `MARIDA.zip`, whose central
    directory is read by HTTP range requests (a few hundred kilobytes of a
    1.16 GB file) and must name exactly `MARIDA_SCENES`.

ATMOSPHERIC CORRECTION — ACOLITE, EXACTLY AS IN THE CO-LOCATION PROBE.
Sentinel-2's own Level-2A is not usable over water: on tile 35SMD
(2024-08-19) 15–17 % of water pixels sit at the lowest digital number in B04,
B06 and B08, 18 % of water pixels have an index band pinned
(`probes/s2fdi_coast_measure.json`). So the store starts from Level-1C and
runs ACOLITE's dark spectrum fitting (`ACOLITE_SETTINGS`: one fixed aerosol
optical thickness per scene, scene-mean geometry, default ozone and pressure,
no glint correction, 20 m target resolution) over the WHOLE tile (no `limit`
— the one difference from the probe, which corrected crops). ACOLITE is the
public repository at the pinned commit `ACOLITE_COMMIT`. `tests/test_family13_
s2fdi_coast.py` asserts the settings, the wavelengths and the FDI factors are
the probe's.

FULL-RESOLUTION READS (`ml/family1/acolite_full_res.py`, 2026-10-14). ACOLITE
is never started directly: the adapter runs that launcher, which makes
`gdal.Warp` default to `overviewLevel='NONE'` and then runs ACOLITE unchanged.
Without it GDAL answers ACOLITE's 10 m -> 20 m "average" warp from the JPEG
2000 file's own 20 m wavelet level, not from the 2x2 mean of the 10 m pixels,
so B04/B08 sit on a footprint shifted ~7 m from B06/B11 and FDI decorrelates
within a scene (falsifier (i): 16PCC 2018-06-11 Spearman 0.970 -> 0.999;
`probes/s2fdi_coast_diagnosis_i.json`). The launcher prints a marker the
adapter requires in acolite.log, and every acquisition record carries
`band_read_path`, so a frame made without it is identifiable.

THE INDICES (plan §4, with each satellite's own centre wavelengths `WAVE`):
  FDI  = R8 - [R6 + (R11 - R6) * 10 (l8 - l4)/(l11 - l4)]   factor 1.7722 S2A,
                                                                    1.7770 S2B
  FAI  = R8 - [R4 + (R11 - R4) (l8 - l4)/(l11 - l4)]
  NDVI = (R8 - R4)/(R8 + R4)
  PI   = R8/(R8 + R4)
computed in float32 from ACOLITE's surface reflectance (rho_s) and stored as
float16. Bands outside [RHO_LO, RHO_HI] become NaN and are counted BEFORE the
indices are formed, so an index is never made from a masked band; indices
outside their bounds (NDVI outside [-1, 1], PI outside [0, 1] — possible when
a dark-water reflectance is a few thousandths below zero) become NaN and are
counted too (contract rule 3).

WATER ONLY, AWAY FROM CLOUD. A pixel is stored when Sen2Cor's SCL calls it
water (class 6), all four bands are finite and in bounds, and it lies more
than `CLOUD_BUFFER_M` = 1000 m from any SCL cloud shadow, medium- or
high-probability cloud or thin cirrus (classes 3, 8, 9, 10). THE BUFFER IS
MEASURED (`probes/s2fdi_coast_probe.json`, "cloud_buffer"): on the six cached
2021 Pacific crops of the co-location probe (open ocean, 9.5 M SCL-water
pixels with real cloud fields), the fraction of water pixels whose FDI exceeds
the 99.9th percentile of water more than 2 km from cloud — 0.1 % by
construction — is 8.5 % within 20 m of a flagged cloud, 2.0 % at 200–300 m,
0.58 % at 500–700 m, 0.29 % at 700–1000 m, 0.21 % at 1.0–1.5 km and 0.13 % at
1.5–2 km. Cloud fringes Sen2Cor does not flag are the brightest "debris" in a
scene (the probe's ten brightest disc pixels were 45–204 m from cloud). The
rule chosen: the smallest distance beyond which the excess is within twice
the nominal 0.1 % — 1000 m. It is one constant; the curve is stored so a
reader can choose another and knows what it costs. What it costs on a
CLEAR scene, measured on 35SMD 2024-08-19 (catalogue cloud cover 0 %): SCL
still flags 1,009 pixels as cloud or shadow — bright coast, ships — and the
buffer removes 629,405 of 15.7 M water pixels (4.0 %). A bright floating
target that Sen2Cor calls cloud is therefore masked with a kilometre around
it; falsifier (iii) is where that would show.

THE LAYOUT — the family's sharded tier-G writer, unchanged
(`ml/family1/sharded.py`). One GROUP per MGRS tile (`T35SMD`, `T16PCC`, …),
each on the tile's OWN UTM grid: 5490 x 5490 pixels of 20 m, row 0 northern-
most, x0/y0 the grid's north-west EDGE, exactly the L2A SCL asset's
`proj:transform` (`TILE_GRIDS`, measured on two acquisitions per tile and
re-checked against every acquisition's SCL and ACOLITE output, which must
agree to the metre — measured: ACOLITE's 20 m output reports `xrange`/
`yrange` identical to the grid and its pixel-centre lat/lon agree with the
grid's to < 0.2 m). Frames are DAYS: F = 5 frames of 86,400 s per five-day
bin, frame f of bin b is UTC day b*5 + f since 1982-01-01, and a tile is never
imaged twice on one UTC day (measured: 415 selected 35SMD acquisitions
2019–2023, at most one a day; the store refuses otherwise). Tiles are
256 x 256 x 8 float16, one zstd frame each (level 15); a tile with no stored
water pixel is not written at all, so land, cloud and the edge of the swath
cost nothing. Stored size, measured on 35SMD 2024-08-19 (52 % water, almost
no cloud): 141.7 MB for 15.7 M water pixels, 9.03 bytes a water pixel.

THE ACQUISITION INDEX. Every stored frame carries one record in the ledger —
`counts["acquisitions"]`, which the framework concatenates across lanes into
store.json's `counts` — with the tile, bin, frame, UTC instant, satellite,
Level-1C and Level-2A item ids, product name, processing baselines, the
catalogue's cloud cover, SCL's water fraction, the pixels stored and the
pixels the cloud buffer removed, ACOLITE's aerosol optical thickness and the
per-step seconds. Acquisitions the selection excluded (cloud >= 20 %, a
non-MARIDA date on a MARIDA tile, a superseded processing copy) are listed in
`counts["acquisitions_excluded"]` with the reason; frames with no picture
answer `no_frame_in_bin` (the product's cadence is the satellites' revisit,
coarser than one day) and the MARIDA scenes that are not readable answer
`absent_upstream`. `acquisitions(store_json)` returns the index.

WHICH COPY. Earth Search lists reprocessed copies beside the originals (one
datatake, several processing baselines, sometimes several sequence numbers
at one baseline — and the copies' cloud cover can differ by tens of percent,
measured on 35SMD). The store takes, per datatake, the HIGHEST processing
baseline, then the highest sequence number, decides the cloud cut on THAT
copy, and lists the others as superseded. The Level-2A twin (for SCL) is the
same datatake's highest-baseline copy.

LANES (ADAPTER_CONTRACT "Lanes"). `S2FDI_TILES` (a comma list of tiles)
restricts the groups, and a restricted run is a group-subset lane
(`group_subset`). The measured cost is ~300 s of ACOLITE plus ~35 s of
download and ~10 s of masking and compression per acquisition (measured
end to end, 2026-10-13: 347 s and 301 s for the two 35SMD acquisitions of
bin 3114), and a hosted runner has six hours: `plan_lanes()` (`python3 -m
family1.adapters.s2fdi_coast lanes`) prints one dispatch per lane — 35SMD
by calendar quarter (12-35 acquisitions), MARIDA by runs of consecutive
years (7-23 scenes) with their tiles — and `fetch_preflight` refuses a lane
whose own count would not fit (`MAX_LANE_HOURS` at `SEC_PER_ACQ`).

THE FALSIFIERS (plan §6) are `check_marida` and `report_plp` below, run as
`python3 -m family1.adapters.s2fdi_coast check ...` against a built store.

ON A HOSTED RUNNER (ubuntu-latest; the family1-build Install step needs, in
addition to its own pip line):
    sudo apt-get install -y gdal-bin libgdal-dev
    pip install "GDAL==$(gdal-config --version)" pyproj python-dateutil \
        scikit-image matplotlib
and ACOLITE itself is fetched by the adapter (`acolite_dir`: git init +
fetch --depth 1 of ACOLITE_COMMIT) unless S2FDI_ACOLITE_DIR names a checkout.
ACOLITE downloads its LUTs (~760 MB) from raw.githubusercontent.com on first
use, about 40 s once per job. Measured per acquisition: 301-412 s wall
(ACOLITE 249-343 s), peak RSS of the process tree 4.05 GB, ~2.3 GB of
transient disk.

Knobs, read at construction: S2FDI_TILES; S2FDI_CACHE (a directory holding
already-downloaded .SAFEs, reused and never deleted); S2FDI_ACOLITE_DIR (an
ACOLITE checkout; default: a clone at ACOLITE_COMMIT in the scratch space);
S2FDI_SMOKE_PX (the synthetic grid size); S2FDI_MAX_LANE_HOURS.
"""
import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1 import acolite_full_res as acr
from family1.adapters import _common as cm
from family1.adapters import _stac as st

# ================================================================ sources ==
STAC = "https://earth-search.aws.element84.com/v1"
STAC_SEARCH = STAC + "/search"
# 50, not 200: Earth Search answers 502 "Internal server error" on EVERY
# attempt for a 200-item page of some sentinel-2-l2a tile-years (2019 16PCC,
# 16PEC, 36JUN; measured 2026-10-13) -- its response is too large for the
# service's gateway (100 items = 4.6 MB, under the ~6 MB limit only just).
# Lanes #1087/#1088 and their re-runs #1109/#1110 died on it, six attempts
# each. 50 items = 3.3 MB on the same query.
STAC_LIMIT = 50
L1C_HTTP = "https://sentinel-s2-l1c.s3.eu-central-1.amazonaws.com/"
L1C_PAYMENT = L1C_HTTP + "?requestPayment"
ACOLITE_GIT = "https://github.com/acolite/acolite.git"
# the launcher every ACOLITE run goes through, and what it records
ACOLITE_LAUNCHER = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "acolite_full_res.py")
BAND_READ_PATH = acr.READ_PATH
# the commit the co-location probe and the 2026-10-13 measurement ran
ACOLITE_COMMIT = "ca477dd91dfb5f8eb77333824b38cb0bcce5e083"
MARIDA_ZIP = "https://zenodo.org/records/5151941/files/MARIDA.zip?download=1"
CDSE_NOTE = ("only in the Copernicus Data Space (CDSE), whose product "
             "download answers HTTP 401 'Token not found' without an account "
             "(measured 2026-10-13); this build holds no CDSE credential")

# ===================================================== the measured grids ==
# Each tile's 20 m grid: EPSG, north-west EDGE (x0, y0), measured from the
# sentinel-2-l2a SCL asset's proj:transform on two acquisitions per tile
# (2026-10-13). Every acquisition is re-checked against it.
TILE_GRIDS = {
    "16PCC": (32616, 300000, 1800000), "16PDC": (32616, 399960, 1800000),
    "16PEC": (32616, 499980, 1800000), "16QED": (32616, 499980, 1900020),
    "18QWF": (32618, 499980, 2100000), "18QYF": (32618, 699960, 2100000),
    "18QYG": (32618, 699960, 2200020), "19QDA": (32619, 399960, 2100000),
    "30VWH": (32630, 499980, 6300000), "35SMD": (32635, 399960, 4400040),
    "36JUN": (32736, 300000, 6800020), "48MXU": (32748, 600000, 9400000),
    "48MYU": (32748, 699960, 9400000), "48PZC": (32648, 799980, 1800000),
    "50LLR": (32750, 300000, 9100000), "51PTS": (32651, 199980, 1700040),
    "51RVQ": (32651, 399960, 3500040), "52SDD": (32652, 399960, 3900000),
}
PX = 20.0
NPX = 5490
LESVOS = "35SMD"

# MARIDA's 63 scenes (tile, UTC date), from the folder names of MARIDA.zip
# (`patches/S2_<d-m-yy>_<tile>/`), measured 2026-10-13; `index` re-reads the
# zip's central directory and refuses if the set differs.
MARIDA_SCENES = (
    ("16PEC", "2015-11-29"), ("51PTS", "2016-07-17"),
    ("16PCC", "2016-09-04"), ("16PDC", "2016-11-03"),
    ("16PCC", "2017-01-12"), ("16PEC", "2017-01-12"),
    ("16PCC", "2017-02-21"), ("51RVQ", "2017-08-29"),
    ("16PCC", "2017-08-30"), ("16PEC", "2017-10-09"),
    ("48MYU", "2017-12-06"), ("48PZC", "2018-01-18"),
    ("16PEC", "2018-02-16"), ("16PCC", "2018-02-21"),
    ("16PCC", "2018-02-26"), ("50LLR", "2018-03-04"),
    ("16PEC", "2018-03-08"), ("16QED", "2018-03-08"),
    ("30VWH", "2018-04-20"), ("16PCC", "2018-06-11"),
    ("16PCC", "2018-08-30"), ("16PCC", "2018-09-14"),
    ("16PCC", "2018-09-19"), ("16PDC", "2018-09-19"),
    ("52SDD", "2018-10-07"), ("16PDC", "2018-10-24"),
    ("16PDC", "2018-11-03"), ("48PZC", "2018-11-14"),
    ("48MXU", "2018-12-06"), ("16PCC", "2018-12-13"),
    ("19QDA", "2019-01-11"), ("16PEC", "2019-01-12"),
    ("16PCC", "2019-01-27"), ("16QED", "2019-01-27"),
    ("36JUN", "2019-04-24"), ("51PTS", "2019-05-18"),
    ("48MXU", "2019-05-25"), ("16PCC", "2019-09-04"),
    ("48PZC", "2019-11-24"), ("48MYU", "2019-12-01"),
    ("18QYG", "2020-03-07"), ("18QYF", "2020-03-14"),
    ("18QYF", "2020-03-19"), ("18QWF", "2020-03-22"),
    ("18QYF", "2020-03-24"), ("16PCC", "2020-08-24"),
    ("18QYF", "2020-09-15"), ("16PCC", "2020-09-18"),
    ("16PDC", "2020-09-18"), ("16PCC", "2020-09-23"),
    ("16PCC", "2020-09-28"), ("16PDC", "2020-09-28"),
    ("18QYF", "2020-10-15"), ("18QYF", "2020-10-20"),
    ("16PCC", "2020-11-15"), ("18QYF", "2020-11-29"),
    ("18QYF", "2020-12-04"), ("16PCC", "2020-12-12"),
    ("18QYF", "2020-12-14"), ("18QYF", "2020-12-22"),
    ("18QYF", "2020-12-29"), ("18QYF", "2021-01-03"),
    ("18QYF", "2021-01-23"),
)
# measured 2026-10-13: no sentinel-2-l1c (and no -l2a) item on Earth Search
MARIDA_CDSE_ONLY = (("16PEC", "2015-11-29"), ("51PTS", "2016-07-17"),
                    ("16PCC", "2016-09-04"), ("16PDC", "2016-11-03"))

# The Plastic Litter Project's Sentinel-2 dates on 35SMD (plan §2b): the
# 2019 record's Vector_Points folders (Zenodo 3752719), the 2021 record's 22
# per-date zips (7085112) and the 2022-23 record's `_S2` folders (10046182),
# all read from the records 2026-10-13.
PLP_DATES = (
    "2019-04-18", "2019-05-03", "2019-05-18", "2019-05-28", "2019-06-07",
    "2021-06-11", "2021-06-21", "2021-06-26", "2021-07-01", "2021-07-06",
    "2021-07-11", "2021-07-16", "2021-07-21", "2021-07-26", "2021-07-31",
    "2021-08-05", "2021-08-10", "2021-08-15", "2021-08-20", "2021-08-25",
    "2021-08-30", "2021-09-04", "2021-09-09", "2021-09-14", "2021-09-19",
    "2021-09-24", "2021-10-04",
    "2022-07-21", "2022-07-26", "2022-09-04", "2022-09-09", "2023-06-21")
# the target box the producer itself corrected with ACOLITE in 2022-23
# (`limit=` of every `_S2/L2W/*settings_user.txt`, read from the zip by range)
PLP_BOX_2022 = (39.03, 26.51, 39.05, 26.54)          # S, W, N, E

CLOUD_MAX = 20.0
CLOUD_BUFFER_M = 1000.0
SCL_WATER = 6
SCL_CLOUDISH = (3, 8, 9, 10)

# central wavelengths (nm) and FDI factors — the co-location probe's
WAVE = {"sentinel-2a": {"B04": 664.6, "B06": 740.5, "B08": 832.8,
                        "B11": 1613.7},
        "sentinel-2b": {"B04": 665.0, "B06": 739.1, "B08": 833.0,
                        "B11": 1610.4}}
FDI_FACTOR = {"sentinel-2a": 1.7722, "sentinel-2b": 1.7770}
# ACOLITE's rho_s names differ by a nanometre between S2A and S2B
ACO_ALT = {"B04": ("665",), "B06": ("740", "739"), "B08": ("833",),
           "B11": ("1614", "1610")}
# the co-location probe's settings, without its `limit` (whole tile here)
ACOLITE_SETTINGS = (
    "s2_target_res=20",
    "atmospheric_correction_method=dark_spectrum",
    "dsf_aot_estimate=fixed", "resolved_geometry=False",
    "ancillary_data=False", "l2w_parameters=None",
    "rgb_rhot=False", "rgb_rhos=False", "map_l2w=False",
    "l1r_export_geotiff=False", "l2r_export_geotiff=False",
    "l2r_include_bands=" + ",".join(
        f"rhos_{w}" for w in ("492", "560", "559", "665", "740",
                              "739", "833", "1614", "1610")),
)

RHO_LO, RHO_HI = -0.02, 1.7
RHO_UNIT = ("surface reflectance rho_s (ACOLITE dark spectrum fitting on "
            "Level-1C; dimensionless)")
CHANNELS = (
    ("fdi", "floating debris index (Biermann 2020), reflectance units",
     -2.0, 2.0),
    ("fai", "floating algae index (Hu 2009), reflectance units", -2.0, 2.0),
    ("ndvi", "normalised difference vegetation index (B08, B04)", -1.0, 1.0),
    ("pi", "plastic index (Themistocleous 2020) B08/(B08+B04)", 0.0, 1.0),
    ("rho_b04", RHO_UNIT, RHO_LO, RHO_HI),
    ("rho_b06", RHO_UNIT, RHO_LO, RHO_HI),
    ("rho_b08", RHO_UNIT, RHO_LO, RHO_HI),
    ("rho_b11", RHO_UNIT, RHO_LO, RHO_HI),
)
NAMES = [c[0] for c in CHANNELS]
COL = {n: i for i, n in enumerate(NAMES)}
BANDS = ("B04", "B06", "B08", "B11")

# measured per acquisition on 35SMD 2024-08-19 (sandbox, 2 cores): ACOLITE
# 298.8 s, download 34.1 s, mask + compress ~35 s
SEC_PER_ACQ = 380.0
MAX_LANE_HOURS = 5.5


class SelectionError(ValueError):
    """The catalogue does not add up to one acquisition per tile and day."""


# ================================================================ helpers ==
def group_name(tile):
    return "T" + tile


def tile_of(group):
    if not re.fullmatch(r"T\d{2}[A-Z]{3}", group or ""):
        raise ValueError(f"{group!r} is not a T<MGRS tile> group name")
    return group[1:]


def grid_of(tile, npx=NPX):
    epsg, x0, y0 = TILE_GRIDS[tile]
    return {"H": int(npx), "W": int(npx), "x0": float(x0), "dx": PX,
            "y0": float(y0), "dy": -PX, "crs": f"EPSG:{epsg}",
            "row_order": "north first (row 0 = northernmost); x0/y0 are the "
                         "grid's north-west EDGE",
            "mgrs_tile": tile,
            "coordinates": "pixel (r, c) centre = (x0 + (c + 0.5) dx, "
                           "y0 + (r + 0.5) dy) in the tile's UTM zone",
            "source": "sentinel-2-l2a SCL asset proj:transform; checked "
                      "against every acquisition's SCL and ACOLITE output"}


def baseline_key(pr):
    b = str(pr.get("s2:processing_baseline") or "0.0")
    return tuple(int(x) for x in re.findall(r"\d+", b))


def seq_of(item_id):
    m = re.search(r"_(\d+)_L[12][AC]$", item_id)
    return int(m.group(1)) if m else 0


def datatake_key(pr):
    """One acquisition whatever its processing (the probe's rule)."""
    return re.sub(r"_N\d\d\.\d\d$", "", pr["s2:datatake_id"])


def utc_day(pr):
    return dt.date.fromisoformat(pr["datetime"][:10])


def best_copy(items):
    """Highest processing baseline, then highest sequence number, then id."""
    return max(items, key=lambda f: (baseline_key(f["properties"]),
                                     seq_of(f["id"]), f["id"]))


def fdi_factor(plat):
    w = WAVE[plat]
    lin = (w["B08"] - w["B04"]) / (w["B11"] - w["B04"])
    fac = FDI_FACTOR[plat]
    if abs(lin * 10 - fac) > 2e-4:
        raise ValueError(f"FDI factor {fac} != wavelengths ({lin * 10:.4f})")
    return lin, fac


def indices(R, plat):
    """float32 FDI, FAI, NDVI, PI from {B04, B06, B08, B11} (plan §4)."""
    lin, fac = fdi_factor(plat)
    lin, fac = np.float32(lin), np.float32(fac)
    with np.errstate(invalid="ignore", divide="ignore"):
        fdi = R["B08"] - (R["B06"] + (R["B11"] - R["B06"]) * fac)
        fai = R["B08"] - (R["B04"] + (R["B11"] - R["B04"]) * lin)
        s = R["B08"] + R["B04"]
        ndvi = (R["B08"] - R["B04"]) / s
        pi = R["B08"] / s
    out = {"fdi": fdi, "fai": fai, "ndvi": ndvi, "pi": pi}
    for k in out:
        out[k][~np.isfinite(out[k])] = np.nan
    return out


def cloud_distance(scl, px=PX):
    """Distance (m) from each pixel to the nearest SCL cloud/shadow/cirrus."""
    from scipy import ndimage
    cl = np.isin(scl, SCL_CLOUDISH)
    if not cl.any():
        return np.full(scl.shape, np.inf, np.float32)
    return (ndimage.distance_transform_edt(~cl) * px).astype(np.float32)


def water_mask(scl, buffer_m=CLOUD_BUFFER_M):
    """(stored-pixel mask, counts): SCL water farther than the buffer."""
    water = scl == SCL_WATER
    near = water & (cloud_distance(scl) <= buffer_m)
    keep = water & ~near
    valid = int((scl != 0).sum())
    return keep, {"scl_water_px": int(water.sum()),
                  "scl_valid_px": valid,
                  "scl_water_fraction": (float(water.sum()) / valid
                                         if valid else 0.0),
                  "cloud_px": int(np.isin(scl, SCL_CLOUDISH).sum()),
                  "water_removed_by_buffer_px": int(near.sum())}


def make_frame(adapter, R, scl, plat, buffer_m=CLOUD_BUFFER_M):
    """-> ([H, W, 8] float16 with NaN off-water, counts). R: float32 bands."""
    lo, hi = RHO_LO, RHO_HI
    oob = {}
    R = {b: np.array(R[b], np.float32, copy=True) for b in BANDS}
    water = scl == SCL_WATER
    for b in BANDS:
        v = R[b]
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(v) & ((v < lo) | (v > hi))
        # counted where a value could have been STORED (SCL water); masked
        # everywhere, so no index is ever formed from it
        n = int((bad & water).sum())
        if n:
            oob[f"rho_{b.lower()}"] = n
        v[bad] = np.nan
    keep, counts = water_mask(scl, buffer_m)
    finite = np.ones(scl.shape, bool)
    for b in BANDS:
        finite &= np.isfinite(R[b])
    counts["water_band_missing_px"] = int((keep & ~finite).sum())
    keep &= finite
    ix = indices(R, plat)
    H, W = scl.shape
    out = np.full((H, W, len(CHANNELS)), np.nan, np.float32)
    for n in ("fdi", "fai", "ndvi", "pi"):
        out[..., COL[n]][keep] = ix[n][keep]
    for b in BANDS:
        out[..., COL["rho_" + b.lower()]][keep] = R[b][keep]
    flat = out.reshape(-1, len(CHANNELS))
    more = adapter.mask_bounds(flat)
    for k, v in more.items():
        if k.startswith("rho_"):
            continue          # bands were masked above, so nothing new here
        oob[k] = oob.get(k, 0) + v
    counts["stored_px"] = int(np.isfinite(out[..., COL["rho_b08"]]).sum())
    counts["index_masked_px"] = {k: v for k, v in oob.items()
                                 if not k.startswith("rho_")}
    return out.astype(np.float16), counts, oob


def acquisitions(store_meta):
    """The store's acquisition index: one dict per stored frame."""
    if isinstance(store_meta, str):
        with open(store_meta) as fh:
            store_meta = json.load(fh)
    rows = list((store_meta.get("counts") or {}).get("acquisitions") or [])
    return sorted(rows, key=lambda r: (r["group"], r["bin"], r["frame"]))


# ============================================== zip central directory ====
class RangeFile(io.RawIOBase):
    """A seekable file over HTTP range reads (`_common.range_reader`), so
    `zipfile` reads only a zip's central directory and the members asked for."""

    def __init__(self, url, size):
        self.url, self.n, self.p = url, int(size), 0
        self._read = cm.range_reader(url)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.p

    def seek(self, off, whence=0):
        self.p = off if whence == 0 else (self.p + off if whence == 1
                                          else self.n + off)
        return self.p

    def readinto(self, b):
        k = min(len(b), self.n - self.p)
        if k <= 0:
            return 0
        d = self._read(self.p, k)
        b[:len(d)] = d
        self.p += len(d)
        return len(d)


def http_size(url):
    import urllib.request
    req = urllib.request.Request(url, method="HEAD", headers=f10b.UA)
    with urllib.request.urlopen(req, timeout=f10b.SOCKET_TIMEOUT) as r:
        return int(r.headers["Content-Length"]), r.geturl()


def open_zip(path_or_url):
    if re.match(r"https?://", str(path_or_url)):
        n, final = http_size(path_or_url)
        return zipfile.ZipFile(io.BufferedReader(RangeFile(final, n),
                                                 buffer_size=1 << 16))
    return zipfile.ZipFile(path_or_url)


MARIDA_DIR = re.compile(r"^patches/S2_(\d{1,2})-(\d{1,2})-(\d{2})_"
                        r"(\d{2}[A-Z]{3})/")


def marida_scene_set(names):
    out = set()
    for n in names:
        m = MARIDA_DIR.match(n)
        if m:
            d, mo, yy, tile = m.groups()
            out.add((tile, str(dt.date(2000 + int(yy), int(mo), int(d)))))
    return out


# =========================================================== selection ===
def select(tile, l1c_items, l2a_items, days=None, marida_days=(),
           plp_days=(), cdse_only=()):
    """Decide, for one tile, which acquisitions the store holds.

    Returns ({day: acquisition dict}, [excluded dicts], {day: absent reason}).
    `days` limits to the days a lane owns; MARIDA tiles hold their MARIDA
    dates only, 35SMD every copy below CLOUD_MAX plus the PLP dates.
    """
    want = None if days is None else set(days)
    by_dt = {}
    for f in l1c_items:
        by_dt.setdefault(datatake_key(f["properties"]), []).append(f)
    l2_by = {}
    for f in l2a_items:
        l2_by.setdefault(datatake_key(f["properties"]), []).append(f)
    sel, excl, absent, seen = {}, [], {}, {}
    mdays = {str(d) for d in marida_days}
    pdays = {str(d) for d in plp_days}
    for key in sorted(by_dt):
        copies = by_dt[key]
        best = best_copy(copies)
        pr = best["properties"]
        day = utc_day(pr)
        if want is not None and day not in want:
            continue
        for c in copies:
            if c is not best:
                excl.append({"tile": tile, "day": str(day), "l1c": c["id"],
                             "reason": f"superseded copy (kept {best['id']})",
                             "cloud_cover": c["properties"].get(
                                 "eo:cloud_cover")})
        cc = pr.get("eo:cloud_cover")
        if tile == LESVOS:
            why = "plp_campaign_date" if str(day) in pdays else (
                "cloud_lt_20" if cc is not None and cc < CLOUD_MAX else None)
            if why is None:
                excl.append({"tile": tile, "day": str(day), "l1c": best["id"],
                             "reason": f"cloud cover {cc} >= {CLOUD_MAX} %",
                             "cloud_cover": cc})
                continue
        else:
            if str(day) not in mdays:
                excl.append({"tile": tile, "day": str(day), "l1c": best["id"],
                             "reason": "not a MARIDA date on a MARIDA tile",
                             "cloud_cover": cc})
                continue
            why = "marida_scene"
        if day in seen:
            raise SelectionError(
                f"{tile}: two datatakes on {day} ({seen[day]} and "
                f"{best['id']}) — a day frame holds one acquisition")
        seen[day] = best["id"]
        twins = l2_by.get(key) or []
        if not twins:
            absent[day] = (f"{best['id']}: no sentinel-2-l2a twin of datatake "
                           f"{key} — no SCL water / cloud mask")
            continue
        l2 = best_copy(twins)
        sel[day] = {"tile": tile, "day": str(day), "l1c": best["id"],
                    "l2a": l2["id"], "datatake": key,
                    "product": pr.get("s2:product_uri"),
                    "satellite": pr["platform"],
                    "datetime": pr["datetime"],
                    "processing_baseline": pr.get("s2:processing_baseline"),
                    "l2a_baseline": l2["properties"].get(
                        "s2:processing_baseline"),
                    "cloud_cover": cc, "selection": why,
                    "_l1c": best, "_l2a": l2}
    if tile != LESVOS:
        for d in sorted(mdays):
            day = dt.date.fromisoformat(d)
            if want is not None and day not in want:
                continue
            if day in sel or day in absent:
                continue
            if (tile, d) in set(cdse_only):
                absent[day] = f"MARIDA scene {tile} {d}: {CDSE_NOTE}"
            else:
                absent[day] = (f"MARIDA scene {tile} {d}: no sentinel-2-l1c "
                               f"item on Earth Search")
    return sel, excl, absent


# ====================================================== ACOLITE's runtime ==
# THE CHECK ACOLITE NEEDS BEFORE ITS FIRST BYTE (measured 2026-10-13 on
# family1-build run 38391739842, the first 35SMD lane: all 32 acquisitions
# failed inside ACOLITE's l1_convert at `ds.ReadAsArray()`). Ubuntu's
# `python3-gdal` (pulled in by `gdal-bin`) is built against numpy 1.26
# (`python3-numpy-abi9`), the workflow's pip installs numpy 2.x ahead of it on
# sys.path, and `pip install GDAL==3.8.4` is then "already satisfied" by the
# apt copy — so `osgeo.gdal` imports fine and `osgeo.gdal_array` does not:
# `ImportError: numpy.core.multiarray failed to import` (`_ARRAY_API not
# found`) at osgeo/gdal.py:3791, the first ReadAsArray ACOLITE makes (the
# detector footprints, l1_convert.py:363 for GML / :365 for JP2). Reproduced
# here with the extracted Ubuntu 3.8.4+dfsg-3ubuntu3 package under Python
# 3.12 + numpy 2.5.4; the same bindings built from source against numpy 2
# (`pip install --no-build-isolation GDAL==3.8.4`) read the array.
ACOLITE_RUNTIME_CHECK = (
    "import numpy\n"
    "from osgeo import gdal, ogr, gdal_array\n"
    "gdal.UseExceptions()\n"
    "assert gdal.GetDriverByName('JP2OpenJPEG') is not None, 'no JP2OpenJPEG'\n"
    "assert ogr.GetDriverByName('GML') is not None, 'no OGR GML driver'\n"
    "gdal.GetDriverByName('MEM').Create('', 2, 2, 1).ReadAsArray()\n"
    "import netCDF4, pyproj, scipy.ndimage, dateutil.parser\n"
    "print(numpy.__version__, gdal.__version__, gdal.__file__)\n")
GDAL_FIX = ("Fix (hosted runner): install libgdal-dev WITHOUT gdal-bin "
            "(`sudo apt-get install -y --no-install-recommends libgdal-dev`) "
            "and build the bindings against the pip numpy: `pip install "
            "--no-build-isolation --no-cache-dir --force-reinstall --no-deps "
            "\"GDAL==$(gdal-config --version)\"`")


def acolite_python_check(python=None, env=None):
    """(ok, detail): can `python` read a GDAL raster into numpy, open GML and
    JP2, and import ACOLITE's other modules? Runs in a SUBPROCESS, because
    that is how ACOLITE runs and an import here can be satisfied by a
    different copy than the child gets."""
    r = subprocess.run([python or sys.executable, "-c", ACOLITE_RUNTIME_CHECK],
                       capture_output=True, text=True, env=env)
    if r.returncode == 0:
        return True, r.stdout.strip()
    return False, last_exception(r.stdout + "\n" + r.stderr)


def last_exception(text, frames=1):
    """The END of the last Python traceback in `text`: the exception line(s)
    and the `frames` innermost `File …` frames with their source lines. A
    traceback's head is the outermost call; what went wrong is at its tail."""
    lines = [l.rstrip() for l in (text or "").splitlines()]
    starts = [i for i, l in enumerate(lines) if l.startswith("Traceback ")]
    if not starts:
        tail = [l for l in lines if l.strip()][-3:]
        return " | ".join(tail) or "(no output)"
    block = lines[starts[-1] + 1:]
    k = None
    for i, l in enumerate(block):
        if l and not l.startswith(" "):      # the first unindented line ends
            k = i                            # the frames: the exception
            break
    if k is None:
        return " | ".join(l.strip() for l in block[-3:] if l.strip())
    exc = [l.strip() for l in block[k:] if l.strip()]
    fr = [i for i, l in enumerate(block[:k]) if l.lstrip().startswith("File ")]
    where = []
    for i in fr[-frames:]:
        where.append(block[i].strip())
        if i + 1 < k and not block[i + 1].lstrip().startswith(("File ", "^")):
            where.append(block[i + 1].strip())
    return " | ".join(exc[:4]) + " — at " + " / ".join(where)


def build_private_gdal(python, target):
    """Build the GDAL Python bindings for the SYSTEM libgdal (`gdal-config`)
    into `target`, against whatever numpy pip's build isolation brings (a
    numpy-2 build runs under any numpy 2.x). Returns (ok, detail). Needs
    libgdal-dev and a compiler, which a hosted runner has."""
    try:
        ver = subprocess.run(["gdal-config", "--version"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return False, f"no gdal-config ({e}) — install libgdal-dev"
    os.makedirs(target, exist_ok=True)
    # NO build isolation: GDAL's setup builds `_gdal_array` only when numpy
    # imports at BUILD time, and an isolated build has no numpy (measured:
    # an isolated build of 3.8.4 installs osgeo without _gdal_array). The
    # build tools go in their own --target so the interpreter is not touched.
    tools = target + "_buildtools"
    r = subprocess.run([python, "-m", "pip", "install", "--quiet",
                        "--no-cache-dir", "--target", tools, "setuptools",
                        "wheel"], capture_output=True, text=True)
    if r.returncode != 0:
        return False, "pip could not fetch setuptools/wheel: " + \
            last_exception(r.stdout + "\n" + r.stderr)
    env = dict(os.environ)
    env["PYTHONPATH"] = tools + (os.pathsep + env["PYTHONPATH"]
                                 if env.get("PYTHONPATH") else "")
    r = subprocess.run([python, "-m", "pip", "install", "--quiet",
                        "--no-cache-dir", "--no-deps", "--no-build-isolation",
                        "--target", target, f"GDAL=={ver}"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return False, f"pip build of GDAL=={ver} failed: " + \
            last_exception(r.stdout + "\n" + r.stderr)
    return True, f"GDAL=={ver} bindings built into {target}"


def acolite_failure(log_path):
    try:
        with open(log_path, errors="replace") as fh:
            return last_exception(fh.read(), frames=2)
    except OSError as e:
        return f"(no ACOLITE log: {e})"


# ============================================================== adapter ===
class S2FdiCoastAdapter(sh.GridAdapter):
    store = "s2fdi_coast"
    title = ("Sentinel-2 floating-debris indices (FDI, FAI, NDVI, PI) and "
             "the four ACOLITE surface reflectances they use, 20 m, water "
             "pixels only, one frame per acquisition — PILOT: MARIDA's 63 "
             "scenes and tile 35SMD (Gulf of Gera, Lesvos)")
    family = "13"
    distribution = "public"
    licence = {
        "name": "Copernicus Sentinel data terms (free, full and open)",
        "redistribution": "attribution", "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Contains modified Copernicus Sentinel data "
                        "[2015-2023], processed with ACOLITE (Vanhellemont & "
                        "Ruddick; RBINS)"),
        "terms": ("the Legal Notice on the use of Copernicus Sentinel Data "
                  "and Service Information (EU Regulation 1159/2013): free, "
                  "full and open; the notice above is required"),
    }
    time_dtype = "int32"
    credentials = ()
    channels = CHANNELS
    # 20 m: log2(0.020 km / 27.83 km) = -10.44; one picture is a ~5 s scan of
    # the tile, the `cat_s2` precedent of one second
    log2_fp = float(np.log2(0.020 / 27.83))
    log2_dt = float(st.log2_dt_for_seconds(1.0))
    per_year = True
    first_year = 2015
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = sh.TILE
    zstd_level = 15
    note_estimate = {"bytes": 44e9,
                     "what": "plan D13: ~310 acquisitions, ~44 GB stored "
                             "(the 2024 rate; 2019-2023 measured 415 on 35SMD "
                             "alone)"}
    qc_policy = (
        "a pixel is stored when Sen2Cor's SCL (the matching L2A) calls it "
        "water (class 6), it lies more than 1000 m from an SCL cloud shadow, "
        "cloud or cirrus pixel (classes 3, 8, 9, 10; buffer MEASURED, see "
        "notes), and all four ACOLITE bands are finite and inside "
        f"[{RHO_LO}, {RHO_HI}]; everything else is NaN. A band outside its "
        "bounds is NaN and counted before any index is formed; an index "
        "outside its bounds is NaN and counted")
    sources = (STAC_SEARCH + " (sentinel-2-l1c, sentinel-2-l2a)",
               L1C_HTTP + "tiles/<utm>/<band>/<sq>/<y>/<m>/<d>/<seq>/<B>.jp2",
               "s3://sentinel-cogs (L2A SCL COG)", MARIDA_ZIP,
               ACOLITE_GIT + "@" + ACOLITE_COMMIT)
    verified = (
        "2026-10-13 from the sandbox: Earth Search STAC for all 63 MARIDA "
        "scenes (59 with sentinel-2-l1c and -l2a items, 4 with neither) and "
        "for 35SMD 2019-2023 (909 L1C items, 724 datatakes, 415 selected, at "
        "most one a day); anonymous GET of " + L1C_PAYMENT + " -> "
        "<Payer>BucketOwner</Payer>, and the control usgs-landsat -> 403 "
        "requester-pays; CDSE product download -> 401 without a token; "
        "every tile grid from two SCL assets; ACOLITE's 20 m L2R xrange/"
        "yrange equal to the 35SMD grid and its lat/lon equal to the grid's "
        "pixel centres to < 0.2 m; the cloud-buffer curve on six cached "
        "2021 Pacific crops")
    notes = (
        "Water pixels only, 1000 m from Sen2Cor cloud: the FDI exceedance of "
        "the far-field p99.9 (0.1 % by construction) is 8.5 % within 20 m of "
        "a flagged cloud, 2.0 % at 200-300 m, 0.58 % at 500-700 m, 0.29 % at "
        "700-1000 m, 0.21 % at 1-1.5 km (six open-ocean crops, 9.5 M water "
        "pixels); 1000 m is the first distance within twice nominal. One "
        "frame per acquisition (F = 5 day frames per five-day bin; never two "
        "acquisitions of a tile on one UTC day), each group a tile on its own "
        "UTM grid. Per datatake the highest processing baseline, then the "
        "highest sequence number, is kept and the cloud cut is decided on it. "
        "The acquisition index is counts.acquisitions; exclusions are "
        "counts.acquisitions_excluded. MARIDA's four CDSE-only scenes are "
        "absent_upstream (no CDSE account).")
    smoke_window = ("2018-08-27", "2018-09-25")
    smoke_probe_month = "2018-09"
    spec_doc_keys = ()

    def __init__(self):
        want = [t.strip().upper().lstrip("T") for t in
                (os.environ.get("S2FDI_TILES") or "").split(",") if t.strip()]
        for t in want:
            if t not in TILE_GRIDS:
                raise ValueError(f"S2FDI_TILES names {t}, which is not one of "
                                 f"the pilot's tiles {sorted(TILE_GRIDS)}")
        self.tiles = want or sorted(TILE_GRIDS)
        self._subset = list(want) or None
        self.npx = int(os.environ.get("S2FDI_SMOKE_PX") or NPX)
        self.cache = os.environ.get("S2FDI_CACHE") or ""
        self.acolite = os.environ.get("S2FDI_ACOLITE_DIR") or ""
        self.acolite_python = (os.environ.get("S2FDI_ACOLITE_PYTHON")
                               or sys.executable)
        self.acolite_env = None          # set by ensure_acolite_runtime
        self.runtime = {}
        self.max_hours = float(os.environ.get("S2FDI_MAX_LANE_HOURS")
                               or MAX_LANE_HOURS)
        self._stac = {}
        self._plan = None
        if want:
            self.notes = (f"{self.notes}\nRESTRICTED BUILD: S2FDI_TILES "
                          f"limited this lane to {', '.join(want)}.")
        if self.npx != NPX:
            self.notes = (f"{self.notes}\nSMOKE GRID: S2FDI_SMOKE_PX set each "
                          f"tile to {self.npx} x {self.npx}. Synthetic.")

    # ------------------------------------------------------------- grid ---
    def group_subset(self):
        return sorted(self.group_grids()) if self._subset else None

    def group_grids(self):
        return {group_name(t): grid_of(t, self.npx) for t in self.tiles}

    def grid_latlon(self, group, x, y):
        from rasterio.warp import transform
        crs = self.group_grids()[group]["crs"]
        X = np.asarray(x, np.float64)
        Y = np.asarray(y, np.float64)
        lon, lat = transform(crs, "EPSG:4326", X.ravel().tolist(),
                             Y.ravel().tolist())
        return (np.array(lat).reshape(X.shape), np.array(lon).reshape(X.shape))

    def specs(self):
        out = super().specs()
        for g, spec in out.items():
            spec["frame_semantics"] = (
                "frame f of bin b is UTC day b * 5 + f since 1982-01-01; a "
                "present frame is the ONE Sentinel-2 acquisition of this tile "
                "on that day that the store selected (store.json "
                "counts.acquisitions names it)")
            spec["mask"] = {"scl_water_class": SCL_WATER,
                            "scl_cloud_classes": list(SCL_CLOUDISH),
                            "cloud_buffer_m": CLOUD_BUFFER_M}
        return out

    def record_frames(self, ctx, group):
        return None

    # -------------------------------------------------------- catalogue ---
    def _fetcher(self, ctx):
        return st.Fetcher(ctx, self.store, attempts=getattr(ctx.a, "attempts",
                                                            3))

    @staticmethod
    def stac_body(collection, tile, year):
        return {"collections": [collection],
                "datetime": f"{year:04d}-01-01T00:00:00Z/"
                            f"{year:04d}-12-31T23:59:59Z",
                "query": {"grid:code": {"eq": f"MGRS-{tile}"}},
                # datetime ALONE drops an item at a page boundary: copies of
                # one acquisition share their datetime, and the cursor skips
                # a tie (measured 2026-10-13, sentinel-2-l2a 35SMD 2019:
                # 264 walked of numberMatched 265; with `id` as the second
                # key, or no sort, all 265)
                "sortby": [{"field": "properties.datetime",
                            "direction": "asc"},
                           {"field": "id", "direction": "asc"}]}

    def stac_year(self, ctx, collection, tile, year, counts):
        k = (collection, tile, int(year))
        if k not in self._stac:
            fet = self._fetcher(ctx)
            c = {}
            items = list(st.stac_search(
                fet, STAC_SEARCH, self.stac_body(collection, tile, year),
                STAC_LIMIT, c, f"{collection} {tile} {year}"))
            for f in items:
                gc = f["properties"].get("grid:code")
                if gc != f"MGRS-{tile}":
                    raise st.Refusal(f"{f['id']}: grid:code {gc}, asked for "
                                     f"MGRS-{tile}")
            self._stac[k] = items
            f10b._merge_counts(counts, {"stac_pages": c.get("pages", 0),
                                        "stac_items": len(items)})
        return self._stac[k]

    def plan_tile(self, ctx, tile, days, counts):
        years = sorted({d.year for d in days})
        l1, l2 = [], []
        for y in years:
            l1 += self.stac_year(ctx, "sentinel-2-l1c", tile, y, counts)
            l2 += self.stac_year(ctx, "sentinel-2-l2a", tile, y, counts)
        mdays = [d for (t, d) in MARIDA_SCENES if t == tile]
        return select(tile, l1, l2, days=days, marida_days=mdays,
                      plp_days=PLP_DATES, cdse_only=MARIDA_CDSE_ONLY)

    def _days(self, wanted):
        out = {}
        for (g, b, f) in wanted:
            out.setdefault(g, set()).add(sh.frame_day(b, f,
                                                      self.frame_seconds))
        return out

    def plan(self, ctx, wanted, counts):
        out = {}
        for g, days in sorted(self._days(wanted).items()):
            out[g] = self.plan_tile(ctx, tile_of(g), sorted(days), counts)
        return out

    def marida_check(self, ctx):
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "MARIDA.zip")
            ctx.count_bytes(os.path.getsize(p))
            names = zipfile.ZipFile(p).namelist()
        else:
            names = open_zip(MARIDA_ZIP).namelist()
        got = marida_scene_set(names)
        if got != set(MARIDA_SCENES):
            sys.exit(f"REFUSING s2fdi_coast: MARIDA.zip names {len(got)} "
                     f"scenes and MARIDA_SCENES {len(MARIDA_SCENES)} "
                     f"(only in zip {sorted(got - set(MARIDA_SCENES))[:4]}, "
                     f"only listed {sorted(set(MARIDA_SCENES) - got)[:4]})")
        return len(got)

    def payer_check(self, ctx):
        if ctx.source_dir:
            return "BucketOwner (smoke: not asked)"
        raw, why = cm.get_bytes(L1C_PAYMENT, attempts=3)
        txt = (raw or b"").decode("utf-8", "replace")
        m = re.search(r"<Payer>(\w+)</Payer>", txt)
        if not m or m.group(1) != "BucketOwner":
            sys.exit(f"REFUSING s2fdi_coast: {L1C_PAYMENT} answered "
                     f"{why} {txt[:200]!r} — measured 2026-10-13 as "
                     f"BucketOwner; a requester-pays bucket cannot be read "
                     f"anonymously, so the L1C source must change (CDSE)")
        return m.group(1)

    def index(self, ctx):
        counts = {}
        n_marida = self.marida_check(ctx)
        payer = self.payer_check(ctx)
        wanted = [k for y in ctx.years for k in ctx.grid_wanted(y)]
        plan = self.plan(ctx, wanted, counts)
        self._plan = plan
        per = {}
        n_sel = 0
        for g, (sel, excl, absent) in plan.items():
            per[g] = {"selected": len(sel), "excluded": len(excl),
                      "absent": len(absent),
                      "by_reason": {r: sum(1 for a in sel.values()
                                           if a["selection"] == r)
                                    for r in sorted({a["selection"]
                                                     for a in sel.values()})},
                      "first": min((str(d) for d in sel), default=None),
                      "last": max((str(d) for d in sel), default=None)}
            n_sel += len(sel)
        first = None
        if not ctx.source_dir:
            for g, (sel, _, _) in plan.items():
                if sel:
                    a = sel[min(sel)]
                    tinfo = self._tileinfo(ctx, a)
                    first = {"l1c": a["l1c"], "tileInfo_product":
                             tinfo["productName"]}
                    break
        return {"dataset": "Sentinel-2 L1C (Earth Search) + ACOLITE DSF",
                "url": STAC_SEARCH, "version": f"ACOLITE {ACOLITE_COMMIT}",
                "marida_scenes": n_marida, "l1c_bucket_payer": payer,
                "groups": per, "acquisitions_selected": n_sel,
                "estimated_hours": round(n_sel * SEC_PER_ACQ / 3600, 2),
                "first_file": first, "stac": counts}

    def fetch_preflight(self, ctx):
        if ctx.source_dir:
            return None
        wanted = [k for y in ctx.years for k in ctx.grid_wanted(y)]
        plan = self._plan or self.plan(ctx, wanted, {})
        self._plan = plan
        n = sum(len(p[0]) for p in plan.values())
        hours = n * SEC_PER_ACQ / 3600
        if hours > self.max_hours:
            sys.exit(f"REFUSING s2fdi_coast lane: {n} acquisitions x "
                     f"{SEC_PER_ACQ:.0f} s (measured) = {hours:.1f} h, more "
                     f"than {self.max_hours} h of a hosted runner. Narrow the "
                     f"window or the tiles (S2FDI_TILES); `python3 -m "
                     f"family1.adapters.s2fdi_coast lanes` prints a plan.")
        free = shutil.disk_usage(ctx.scratch).free
        if n and free < 4e9:
            sys.exit(f"REFUSING s2fdi_coast: {free / 1e9:.1f} GB free under "
                     f"{ctx.scratch}; one acquisition needs ~2.3 GB "
                     f"(SAFE 0.6 + ACOLITE L1R 0.85 + L2R 0.8)")
        self.ensure_acolite_runtime(ctx)
        return None

    def ensure_acolite_runtime(self, ctx):
        """Check ACOLITE's interpreter; when its GDAL bindings cannot read an
        array (Ubuntu's python3-gdal under pip's numpy 2 — the failure of
        family1-build run 38391739842), build matching bindings privately and
        put them first on ACOLITE's PYTHONPATH. Refuses if that fails too."""
        ok, why = acolite_python_check(self.acolite_python)
        self.runtime = {"python": self.acolite_python, "first_check": why}
        if ok:
            return why
        target = os.path.join(ctx.scratch, "gdal_py")
        built, how = build_private_gdal(self.acolite_python, target)
        self.runtime["private_build"] = how
        if built:
            env = dict(os.environ)
            env["PYTHONPATH"] = target + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
            ok2, why2 = acolite_python_check(self.acolite_python, env)
            if ok2:
                self.acolite_env = env
                self.runtime["second_check"] = why2
                print(f"  ::warning::s2fdi_coast: ACOLITE's GDAL could not "
                      f"read an array ({why}); {how}, and ACOLITE runs with "
                      f"it first on PYTHONPATH ({why2})", flush=True)
                return why2
            how = f"{how}, but the check still fails: {why2}"
        sys.exit(f"REFUSING s2fdi_coast: the interpreter ACOLITE would run "
                 f"under ({self.acolite_python}) cannot do what ACOLITE does "
                 f"first — {why}; {how}. {GDAL_FIX}")

    # ------------------------------------------------------------ L1C -----
    def _l1c_tile_path(self, a):
        href = a["_l1c"]["assets"]["nir"]["href"]
        m = re.match(r"s3://sentinel-s2-(l1c|l2a)/(tiles/.+/)B08\.jp2$", href)
        if not m:
            raise st.Refusal(f"{a['l1c']}: unexpected L1C href {href}")
        return m.group(2), m.group(1) != "l1c"

    def _tileinfo(self, ctx, a):
        tile_path, _ = self._l1c_tile_path(a)
        raw, why = cm.get_bytes(L1C_HTTP + tile_path + "tileInfo.json",
                                attempts=getattr(ctx.a, "attempts", 3))
        if raw is None:
            raise IOError(f"{L1C_HTTP}{tile_path}tileInfo.json: {why}")
        ti = json.loads(raw.decode("utf-8"))
        want = (a["product"] or "").replace(".SAFE", "")
        if ti.get("productName") != want:
            raise st.Refusal(f"{a['l1c']}: bucket product "
                             f"{ti.get('productName')} != STAC {want}")
        return ti

    def fetch_safe(self, ctx, a, workdir):
        """A minimal L1C SAFE (scene + tile metadata, 13 bands, detector
        footprints) — the co-location probe's `fetch_safe`, extended to the
        pre-04.00 GML footprints the probe (2021, baseline 05.00) never met.
        Reuses a SAFE found under S2FDI_CACHE."""
        tile_path, quirk = self._l1c_tile_path(a)
        ti = self._tileinfo(ctx, a)
        name = ti["productName"]
        root = self.cache or workdir
        safe = os.path.join(root, name + ".SAFE")
        prod = ti["productPath"].rstrip("/") + "/"

        def get(url, dst):
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                return os.path.getsize(dst), 0
            if f10b.fetch_first([url], dst, attempts=getattr(
                    ctx.a, "attempts", 3)) is None:
                raise IOError(f"{url}: 404")
            return os.path.getsize(dst), os.path.getsize(dst)
        nb = new = 0
        mtd = os.path.join(safe, "MTD_MSIL1C.xml")
        s, n_ = get(L1C_HTTP + prod + "metadata.xml", mtd)
        nb, new = nb + s, new + n_
        pmd = open(mtd).read()
        files = re.findall(r"<IMAGE_FILE>([^<]+)</IMAGE_FILE>", pmd)
        if not files:
            raise st.Refusal(f"{name}: MTD_MSIL1C.xml lists no IMAGE_FILE")
        gpath = os.path.join(safe, "GRANULE", files[0].split("/")[1])
        s, n_ = get(L1C_HTTP + tile_path + "metadata.xml",
                    os.path.join(gpath, "MTD_TL.xml"))
        nb, new = nb + s, new + n_
        for f in files:
            b = f.split("_")[-1]
            if b == "TCI":
                continue
            s, n_ = get(L1C_HTTP + tile_path + b + ".jp2",
                        os.path.join(safe, f + ".jp2"))
            nb, new = nb + s, new + n_
            # DETECTOR FOOTPRINTS: a JPEG 2000 mask from processing baseline
            # 04.00 on, a GML polygon file before it (measured 2026-10-13: the
            # 02.07 tile 35SMD 2019-05-03 has qi/MSK_DETFOO_B01.gml and no
            # qi/DETFOO_B01.jp2). ACOLITE reads either from QI_DATA/.
            new_style = baseline_key({"s2:processing_baseline":
                                      a["processing_baseline"]}) >= (4, 0)
            src, dst = ((f"qi/DETFOO_{b}.jp2", f"MSK_DETFOO_{b}.jp2")
                        if new_style else
                        (f"qi/MSK_DETFOO_{b}.gml", f"MSK_DETFOO_{b}.gml"))
            s, n_ = get(L1C_HTTP + tile_path + src,
                        os.path.join(gpath, "QI_DATA", dst))
            nb, new = nb + s, new + n_
        return {"safe": safe, "name": name, "bytes": nb, "bytes_new": new,
                "reused": bool(self.cache and new == 0), "bucket_quirk": quirk}

    def read_scl(self, ctx, a, workdir):
        """The L2A SCL at 20 m, checked against the group's grid."""
        g = grid_of(a["tile"], self.npx)
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "scl",
                             a["l2a"] + ".npy")
            ctx.count_bytes(os.path.getsize(p))
            scl = np.load(p)
            if scl.shape != (g["H"], g["W"]):
                raise sh.ShardError(f"{a['l2a']}: SCL {scl.shape}")
            return scl
        import rasterio
        asset = a["_l2a"]["assets"]["scl"]
        t = asset.get("proj:transform")
        want = [PX, 0, g["x0"], 0, -PX, g["y0"]]
        if t is None or [float(v) for v in t[:6]] != want:
            raise st.Refusal(f"{a['l2a']}: SCL transform {t} != the {a['tile']}"
                             f" grid {want}")
        href = asset["href"]
        if href.startswith("s3://sentinel-cogs/"):
            href = ("https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
                    + href[len("s3://sentinel-cogs/"):])
        dst = os.path.join(workdir, a["l2a"] + "_SCL.tif")
        if f10b.fetch_first([href], dst,
                            attempts=getattr(ctx.a, "attempts", 3)) is None:
            raise IOError(f"{href}: 404")
        with rasterio.open(dst) as d:
            tr = list(d.transform)[:6]
            if [float(v) for v in tr] != want or d.shape != (g["H"], g["W"]):
                raise st.Refusal(f"{a['l2a']}: SCL file grid {tr} {d.shape} "
                                 f"!= {want}")
            scl = d.read(1)
        os.remove(dst)
        return scl

    # -------------------------------------------------------- ACOLITE -----
    def acolite_dir(self, ctx):
        d = self.acolite or os.path.join(ctx.scratch, "acolite_src")
        if not os.path.isfile(os.path.join(d, "launch_acolite.py")):
            if self.acolite:
                sys.exit(f"REFUSING s2fdi_coast: S2FDI_ACOLITE_DIR={d} holds "
                         f"no launch_acolite.py")
            os.makedirs(d, exist_ok=True)
            for cmd in (["git", "init", "-q", d],
                        ["git", "-C", d, "fetch", "-q", "--depth", "1",
                         ACOLITE_GIT, ACOLITE_COMMIT],
                        ["git", "-C", d, "checkout", "-q", "FETCH_HEAD"]):
                subprocess.run(cmd, check=True)
        sha = ""
        if os.path.isdir(os.path.join(d, ".git")):
            sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        return d, sha

    def run_acolite(self, ctx, safe, out):
        aco, sha = self.acolite_dir(ctx)
        os.makedirs(out, exist_ok=True)
        stp = os.path.join(out, "settings.txt")
        with open(stp, "w") as f:
            f.write("\n".join((f"inputfile={safe}", f"output={out}")
                              + ACOLITE_SETTINGS + ("",)))
        t0 = time.time()
        log = os.path.join(out, "acolite.log")
        with open(log, "w") as lf:
            # ACOLITE runs through `ml/family1/acolite_full_res.py`, which
            # makes gdal.Warp read full-resolution JPEG 2000 data
            # (overviewLevel='NONE'): without it the 10 m bands B04/B08 come
            # from the file's own wavelet 20 m level, not the 2x2 mean
            # (ml/family1/probes/s2fdi_coast_diagnosis_i.json).
            rc = subprocess.run([self.acolite_python, ACOLITE_LAUNCHER, aco,
                                 "--cli", f"--settings={stp}"], cwd=aco,
                                stdout=lf, stderr=subprocess.STDOUT,
                                env=self.acolite_env).returncode
        done = [f for f in os.listdir(out) if f.endswith("_L2R.nc")]
        if rc != 0 or len(done) != 1:
            raise IOError(f"ACOLITE rc {rc}, {len(done)} L2R file(s): "
                          + acolite_failure(log))
        with open(log, errors="replace") as fh:
            if acr.READ_PATH_MARKER not in fh.read(4096):
                raise IOError(f"ACOLITE ran without the full-resolution read "
                              f"path (no {acr.READ_PATH_MARKER!r} in {log}); "
                              f"refusing its output")
        for f in os.listdir(out):
            if f.endswith("_L1R.nc"):
                os.remove(os.path.join(out, f))
        return os.path.join(out, done[0]), sha, time.time() - t0

    def read_l2r(self, path, tile):
        """{B04..B11 float32}, attrs; refuses an output off the tile grid."""
        import netCDF4
        g = grid_of(tile, self.npx)
        d = netCDF4.Dataset(path)
        try:
            xr = [float(v) for v in d.getncattr("xrange")]
            yr = [float(v) for v in d.getncattr("yrange")]
            ps = [float(v) for v in d.getncattr("pixel_size")]
            want_x = [g["x0"], g["x0"] + g["W"] * PX]
            want_y = [g["y0"], g["y0"] - g["H"] * PX]
            if xr != want_x or yr != want_y or ps != [PX, -PX]:
                raise sh.ShardError(f"{path}: ACOLITE grid x {xr} y {yr} "
                                    f"px {ps} != {tile} {want_x} {want_y}")
            names = set(d.variables)
            R = {}
            for b, alts in ACO_ALT.items():
                v = next((f"rhos_{w}" for w in alts if f"rhos_{w}" in names),
                         None)
                if v is None:
                    raise sh.ShardError(f"{path}: no rhos for {b}")
                R[b] = d[v][:].filled(np.nan).astype(np.float32)
                if R[b].shape != (g["H"], g["W"]):
                    raise sh.ShardError(f"{path}: {v} {R[b].shape}")
            attrs = {}
            for k in ("ac_aot_550", "ac_model", "acolite_version"):
                if k in d.ncattrs():
                    v = d.getncattr(k)
                    attrs[k] = v.item() if hasattr(v, "item") else str(v)
        finally:
            d.close()
        return R, attrs

    def correct(self, ctx, a, workdir):
        """-> (R, attrs, timings, bytes) for one acquisition."""
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "l2r",
                             a["l1c"] + ".npz")
            ctx.count_bytes(os.path.getsize(p))
            z = np.load(p)
            g = grid_of(a["tile"], self.npx)
            if [float(v) for v in z["xrange"]] != [g["x0"], g["x0"] +
                                                   g["W"] * PX]:
                raise sh.ShardError(f"{p}: stub off the grid")
            R = {b: z[b].astype(np.float32) for b in BANDS}
            return R, {"ac_aot_550": float(z["aot"]), "ac_model": "stub",
                       "acolite_version": "smoke stub"}, \
                {"download_s": 0.0, "acolite_s": 0.0}, \
                {"bytes": int(os.path.getsize(p)), "reused": False,
                 "bucket_quirk": False, "acolite_commit": "stub",
                 "read_path": "smoke stub (no ACOLITE)"}
        if not self.runtime:              # the probe skips fetch_preflight
            self.ensure_acolite_runtime(ctx)
        t0 = time.time()
        sf = self.fetch_safe(ctx, a, workdir)
        t_dl = time.time() - t0
        try:
            l2r, sha, t_aco = self.run_acolite(
                ctx, sf["safe"], os.path.join(workdir, "aco_" + a["l1c"]))
            R, attrs = self.read_l2r(l2r, a["tile"])
        finally:
            shutil.rmtree(os.path.join(workdir, "aco_" + a["l1c"]),
                          ignore_errors=True)
            if not self.cache:
                shutil.rmtree(sf["safe"], ignore_errors=True)
        return R, attrs, {"download_s": round(t_dl, 1),
                          "acolite_s": round(t_aco, 1)}, \
            {"bytes": sf["bytes"], "reused": sf["reused"],
             "bucket_quirk": sf["bucket_quirk"], "acolite_commit": sha,
             "read_path": BAND_READ_PATH}

    # ------------------------------------------------------ the contract --
    def fetch_frames(self, ctx, wanted):
        counts0 = {}
        plan = self.plan(ctx, wanted, counts0)
        first = dict(counts0)
        excl_all, absent_all = [], []
        for g, (sel, excl, absent) in plan.items():
            excl_all += excl
            absent_all += [{"tile": tile_of(g), "day": str(d), "why": w}
                           for d, w in sorted(absent.items())]
        first["acquisitions_excluded"] = excl_all
        first["acquisitions_absent"] = absent_all
        real = []
        sent_first = False
        for (g, b, f) in wanted:
            sel, excl, absent = plan[g]
            day = sh.frame_day(b, f, self.frame_seconds)
            if day in sel:
                real.append((g, b, f, sel[day]))
                continue
            c = {"frame_missing": (sh.FRAME_ABSENT_UPSTREAM if day in absent
                                   else sh.FRAME_NO_FRAME_IN_BIN)}
            if not sent_first:
                c.update(first)
                sent_first = True
            yield g, b, f, None, c
        workdir = os.path.join(ctx.scratch, "s2fdi_work")
        os.makedirs(workdir, exist_ok=True)
        for (g, b, f, a) in real:
            t0 = time.time()
            try:
                R, attrs, tim, src = self.correct(ctx, a, workdir)
                scl = self.read_scl(ctx, a, workdir)
            except (IOError, OSError, st.Refusal) as e:
                ctx.note_absent(str(sh.bin_year(b)),
                                f"{a['l1c']}: {type(e).__name__}: "
                                f"{str(e)[:1500]}")
                continue
            t1 = time.time()
            frame, mc, oob = make_frame(self, R, scl, a["satellite"])
            del R, scl
            rec = {k: v for k, v in a.items() if not k.startswith("_")}
            rec.update(group=g, bin=int(b), frame=int(f),
                       grid=grid_of(a["tile"], self.npx)["crs"],
                       cloud_buffer_m=CLOUD_BUFFER_M, **mc,
                       aot550=attrs.get("ac_aot_550"),
                       acolite_model=attrs.get("ac_model"),
                       acolite_version=attrs.get("acolite_version"),
                       acolite_commit=src["acolite_commit"],
                       acolite_private_gdal=bool(self.acolite_env),
                       band_read_path=src["read_path"],
                       l1c_bytes=src["bytes"], safe_reused=src["reused"],
                       l1c_href_in_l2a_bucket=src["bucket_quirk"],
                       seconds=dict(tim, mask_s=round(time.time() - t1, 1),
                                    total_s=round(time.time() - t0, 1)))
            c = {"acquisitions": [rec], "frames_by_satellite":
                 {a["satellite"]: 1}, "frames_by_selection":
                 {a["selection"]: 1}}
            if oob:
                c["out_of_bounds"] = oob
            if not sent_first:
                c.update(first)
                sent_first = True
            yield g, b, f, frame, c
            del frame
        shutil.rmtree(workdir, ignore_errors=True)

    def year_summary(self, cs):
        n = sum(len(c.get("acquisitions") or []) for c in cs)
        if not n:
            return None
        sel = {}
        for c in cs:
            for k, v in (c.get("frames_by_selection") or {}).items():
                sel[k] = sel.get(k, 0) + v
        return {"acquisitions": n, "by_selection": sel}

    year_summary_note = "acquisitions stored per year, by selection rule"

    # ------------------------------------------------------------ smoke ---
    def smoke_sources(self, root, d_lo, d_hi):
        from family1.adapters import _s2fdi_coast_smoke as sm
        truth = sm.make_smoke_sources(root, d_lo, d_hi)
        self.__init__()
        return truth


ADAPTER = S2FdiCoastAdapter


# ================================================================ lanes ===
def plan_lanes(per_lane=25, lesvos_years=range(2019, 2024)):
    """One family1-build dispatch per lane — inputs only, nothing is
    dispatched. MARIDA: CONSECUTIVE YEARS merged while the lane holds at most
    `per_lane` downloadable scenes, with every tile those years touch (the
    four CDSE-only scenes ride along so the store counts them as
    absent_upstream). 35SMD: one lane per calendar quarter (12-35
    acquisitions; measured 417 over 2019-2023 with the campaign dates, at
    most 35 in a third quarter)."""
    out = []
    years = sorted({int(d[:4]) for _, d in MARIDA_SCENES})
    cur = []

    def flush(ys):
        if not ys:
            return
        sc = [(t, d) for t, d in MARIDA_SCENES if int(d[:4]) in ys]
        out.append({"store": "s2fdi_coast", "stage": "index,fetch",
                    "start": f"{ys[0]}-01-01", "end": f"{ys[-1]}-12-31",
                    "extra_args": "--push-parts",
                    "adapter_env": "S2FDI_TILES=" + ",".join(
                        sorted({t for t, _ in sc})),
                    "_acquisitions": sum(1 for x in sc
                                         if x not in MARIDA_CDSE_ONLY),
                    "_cdse_only_absent": sum(1 for x in sc
                                             if x in MARIDA_CDSE_ONLY)})
    n = 0
    for y in years:
        k = sum(1 for t, d in MARIDA_SCENES if int(d[:4]) == y
                and (t, d) not in MARIDA_CDSE_ONLY)
        if cur and (n + k > per_lane or y != cur[-1] + 1):
            flush(cur)
            cur, n = [], 0
        cur.append(y)
        n += k
    flush(cur)
    for y in lesvos_years:
        for a, b in (("01-01", "03-31"), ("04-01", "06-30"),
                     ("07-01", "09-30"), ("10-01", "12-31")):
            out.append({"store": "s2fdi_coast", "stage": "index,fetch",
                        "start": f"{y}-{a}", "end": f"{y}-{b}",
                        "extra_args": "--push-parts",
                        "adapter_env": f"S2FDI_TILES={LESVOS}",
                        "_acquisitions": "12-35 (measured 2019-2023: 417 "
                                         "in 20 quarters, max 35 in Q3)"})
    return out


# ============================================================ falsifiers ==
def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return None
    r = spearmanr(a, b).statistic
    return None if not np.isfinite(r) else float(r)


HUB_STORE = ("https://huggingface.co/datasets/chfrank/earth-tensors/resolve/"
             "main/tensors/family1_3/s2fdi_coast")


HUB_PARTS = ("https://huggingface.co/datasets/chfrank/earth-tensors/resolve/"
             "main/partials/family1_3/s2fdi_coast")
HUB_TREE = ("https://huggingface.co/api/datasets/chfrank/earth-tensors/tree/"
            "main/partials/family1_3/s2fdi_coast")


class _PartsSource:
    """The `sharded._Source` interface over the LANE PARTS on the Hub
    (`partials/family1_3/s2fdi_coast/<year>/<lane>/<group>__bin_NNNN.*`) —
    the bytes the assembler links into the store unchanged, so a falsifier
    read here reads what the store will hold. `rel` is the store's own
    `<yyyy>/bin_<NNNN>.(zst|idx.npy)`; the bin picks the lane."""

    def __init__(self, group, lane_of_bin):
        self.group, self.lane_of_bin = group, dict(lane_of_bin)
        self.url = True
        self.ranges = 0

    def _url(self, rel):
        m = re.fullmatch(r"\d{4}/bin_(\d+)\.(zst|idx\.npy)", rel)
        if not m:
            raise sh.ShardError(f"parts: no mapping for {rel}")
        base = self.lane_of_bin[int(m.group(1))]
        return f"{base}/{self.group}__bin_{m.group(1)}.{m.group(2)}"

    def get(self, rel):
        return f10b.http_bytes(self._url(rel))

    def read_at(self, rel, off, n):
        self.ranges += 1
        b = cm.range_reader(self._url(rel))(off, n)
        if len(b) != n:
            raise sh.ShardError(f"{self._url(rel)}: {len(b)} of {n} bytes")
        return b


def _parts_reader(tree=HUB_TREE, base=HUB_PARTS):
    """(pseudo store.json, {group: ShardedGroup over _PartsSource},
    acquisitions) from every lane's counts.json on the Hub."""
    def ls(url):
        return json.loads(f10b.http_bytes(url))
    acq, specs, lanes, bins = [], {}, [], {}
    for y in ls(tree):
        if y["type"] != "directory":
            continue
        yr = y["path"].rsplit("/", 1)[-1]
        for ln in ls(tree + "/" + yr):
            if ln["type"] != "directory":
                continue
            lane = ln["path"].rsplit("/", 1)[-1]
            lb = f"{base}/{yr}/{lane}"
            led = json.loads(f10b.http_bytes(lb + "/counts.json"))
            lanes.append(f"{yr}/{lane}")
            for g, sp in (led.get("grids") or {}).items():
                specs.setdefault(g, sp)
            for a in (led.get("counts") or {}).get("acquisitions") or []:
                acq.append(a)
                bins.setdefault(a["group"], {})[int(a["bin"])] = lb
    grp = {}
    for g, sp in specs.items():
        o = sh.ShardedGroup.__new__(sh.ShardedGroup)
        o.src = _PartsSource(g, bins.get(g, {}))
        o.spec = sp
        o._dctx = sh.zstd().ZstdDecompressor()
        o._index = None
        grp[g] = o
    meta = {"groups": {g: {} for g in specs}, "lanes": lanes,
            "counts": {"acquisitions": acq}, "source": "lane parts"}
    return meta, grp, acquisitions(meta)


def _store_reader(store):
    """(store.json, {group: ShardedGroup}, acquisition index) of a built
    store, on a local directory OR the Hub (`HUB_STORE`, or any URL prefix):
    on the Hub only store.json and each group's tile_grid.json are fetched
    here; tiles come later by range reads (`_FrameReader`)."""
    if store in ("parts", HUB_PARTS):
        return _parts_reader()
    if sh._is_url(store):
        meta = json.loads(f10b.http_bytes(store.rstrip("/") + "/store.json"))
        grp = {g: sh.ShardedGroup(store.rstrip("/") + "/" + g)
               for g in meta["groups"]}
    else:
        with open(os.path.join(store, "store.json")) as fh:
            meta = json.load(fh)
        grp = {g: sh.ShardedGroup(os.path.join(store, g))
               for g in meta["groups"]}
    return meta, grp, acquisitions(meta)


class _FrameReader:
    """One (group, bin, frame) read with as few requests as the question
    needs: the bin's .idx.npy whole (a few kB, ONE read), then either single
    tiles (one range read each) or the whole frame as ONE range read — a
    frame's stored tiles are contiguous in the shard (frames ascending,
    tiles row-major), so its span is [first offset, last offset + length).
    Counts requests and bytes, so a report can say what it cost."""

    def __init__(self, grp, b, f, stats=None):
        self.g, self.b, self.f = grp, int(b), int(f)
        self.sp = grp.spec
        self._dctx = sh.zstd().ZstdDecompressor()
        self.stats = stats if stats is not None else {"requests": 0,
                                                      "bytes": 0}
        raw = grp.src.get(sh.index_relpath(self.b))
        self._count(len(raw))
        idx = np.load(io.BytesIO(raw))
        if idx.shape[0] <= self.f:
            raise sh.ShardError(f"bin {b}: index has {idx.shape[0]} frames")
        self.idx = idx[self.f]                     # [nty, ntx, 2]
        self.present = not (self.idx[..., 0] == sh.FRAME_ABSENT).all()

    def _count(self, n):
        self.stats["requests"] += 1
        self.stats["bytes"] += int(n)

    def _decode(self, blob, ty, tx):
        # its OWN decompressor: a zstd context is not thread-safe, and the
        # site control reads frames of one group on several threads
        sp = self.sp
        dt_ = sh.DTYPES[sp["dtype"]]
        want = sp["tile"] * sp["tile"] * sp["C"] * dt_.itemsize
        raw = self._dctx.decompress(blob, max_output_size=want)
        if len(raw) != want:
            raise sh.ShardError(f"a tile decompressed to {len(raw)} bytes, "
                                f"the declared tile is {want}")
        t = np.frombuffer(raw, dt_).reshape(sp["tile"], sp["tile"], sp["C"])
        r0, r1 = self.sp["row_extents"][ty]
        c0, c1 = self.sp["col_extents"][tx]
        return sh.decode(t[:r1 - r0, :c1 - c0], self.sp["dtype"])

    def tile(self, ty, tx):
        """float32 [rows, cols, C] cropped, NaN-filled for an empty tile;
        None if the frame is absent."""
        if not self.present:
            return None
        o, n = (int(v) for v in self.idx[ty, tx])
        r0, r1 = self.sp["row_extents"][ty]
        c0, c1 = self.sp["col_extents"][tx]
        if n == 0:
            return np.full((r1 - r0, c1 - c0, self.sp["C"]), np.nan,
                           np.float32)
        blob = self.g.src.read_at(sh.shard_relpath(self.b), o, n)
        self._count(n)
        return self._decode(blob, ty, tx)

    def frame(self, ch):
        """[H, W] float32 of channel `ch`, the whole frame in ONE read."""
        sp = self.sp
        out = np.full((sp["H"], sp["W"]), np.nan, np.float32)
        if not self.present:
            return None
        st = self.idx[..., 1] > 0
        if not st.any():
            return out
        offs, lens = self.idx[..., 0][st], self.idx[..., 1][st]
        lo, hi = int(offs.min()), int((offs + lens).max())
        span = self.g.src.read_at(sh.shard_relpath(self.b), lo, hi - lo)
        self._count(hi - lo)
        for ty, tx in zip(*np.nonzero(st)):
            o, n = (int(v) for v in self.idx[ty, tx])
            t = self._decode(span[o - lo:o - lo + n], ty, tx)
            r0, r1 = sp["row_extents"][ty]
            c0, c1 = sp["col_extents"][tx]
            out[r0:r1, c0:c1] = t[..., ch]
        del span
        return out


def _cell_values(fr, rows, cols, ch, tile_cache):
    """Channel `ch` at 20 m cells (rows, cols) of one `_FrameReader`; NaN
    where not stored."""
    T = sh.TILE
    out = np.full(len(rows), np.nan, np.float32)
    ok = (rows >= 0) & (cols >= 0) & (rows < fr.sp["H"]) & \
        (cols < fr.sp["W"])
    ty, tx = rows // T, cols // T
    for key in sorted(set(zip(ty[ok].tolist(), tx[ok].tolist()))):
        if key not in tile_cache:
            t = fr.tile(*key)
            tile_cache[key] = None if t is None else t[..., ch].copy()
        t = tile_cache[key]
        m = ok & (ty == key[0]) & (tx == key[1])
        if t is not None:
            out[m] = t[rows[m] - key[0] * T, cols[m] - key[1] * T]
    return out


def check_marida(store, marida_zip, out=None):
    """Falsifiers (i) and (ii) of plan §6 on the store's MARIDA scenes.

    (i) FDI from the store reproduces FDI computed from MARIDA's own
        (ACOLITE Rayleigh-corrected) bands at its "Marine Water" pixels:
        Spearman >= 0.9, pooled over scenes (also reported per scene and per
        20 m cell);
    (ii) the threshold baseline — FDI above its 95th percentile over TRAIN
        Marine Water — scored debris-vs-not on MARIDA's TEST split, from the
        store's FDI and from MARIDA's: |F1 difference| <= 0.05. A pixel with
        no store value (not water, near cloud) is "not debris".
    """
    import rasterio
    from family1.adapters import s2labels as s2l
    meta, grp, acq = _store_reader(store)
    z = open_zip(marida_zip)
    names = z.namelist()
    split_of = {}
    for sp in ("train", "val", "test"):
        nm = next(n for n in names if n.endswith(f"splits/{sp}_X.txt"))
        for line in z.read(nm).decode().split():
            split_of[line.strip()] = sp
    by_scene = {(a["tile"], a["day"]): a for a in acq
                if a.get("selection") == "marida_scene"}
    cols_all = {k: [] for k in ("scene", "cls", "split", "fdi_m", "fdi_s",
                                "cell")}
    per_scene = {}
    tile_cache = {}
    stats = {"requests": 0, "bytes": 0}
    for (tile, day), a in sorted(by_scene.items()):
        d = dt.date.fromisoformat(day)
        tag = f"S2_{d.day}-{d.month}-{d.year % 100}_{tile}"
        sat = 1 if a["satellite"] == "sentinel-2a" else 2
        fr = _FrameReader(grp[a["group"]], a["bin"], a["frame"], stats)
        g = fr.sp["grid"]
        pat = re.compile(rf"^patches/{tag}/{tag}_(\d+)\.tif$")
        n0 = len(cols_all["scene"])
        for nm in names:
            m = pat.match(nm)
            if not m:
                continue
            # the split files name a patch WITHOUT the "S2_" prefix
            pid = f"{tag[3:]}_{m.group(1)}"
            with rasterio.MemoryFile(z.read(nm)) as mf, mf.open() as ds:
                bands = ds.read().astype(np.float32)
                T = ds.transform
            with rasterio.MemoryFile(z.read(nm[:-4] + "_cl.tif")) as mf, \
                    mf.open() as ds:
                cl = ds.read(1)
            rr, cc = np.nonzero(cl > 0)
            if not len(rr):
                continue
            x = T.c + T.a * (cc + 0.5)
            y = T.f + T.e * (rr + 0.5)
            r20 = np.floor((g["y0"] - y) / PX).astype(np.int64)
            c20 = np.floor((x - g["x0"]) / PX).astype(np.int64)
            v = bands[:, rr, cc].T
            fm = s2l.fdi(v, np.full(len(rr), sat))
            fs = _cell_values(fr, r20, c20, COL["fdi"], tile_cache)
            cols_all["scene"] += [tag] * len(rr)
            cols_all["cls"] += cl[rr, cc].astype(int).tolist()
            cols_all["split"] += [split_of.get(pid, "none")] * len(rr)
            cols_all["fdi_m"] += fm.tolist()
            cols_all["fdi_s"] += fs.tolist()
            cols_all["cell"] += (r20 * fr.sp["W"] + c20).tolist()
        tile_cache.clear()
        per_scene[tag] = {"labelled_px": len(cols_all["scene"]) - n0}
    A = {k: np.array(v) for k, v in cols_all.items()}
    res = {"store": store, "marida_zip": str(marida_zip),
           "scenes_in_store": len(by_scene),
           "scenes_scored": sorted(f"{t} {d}" for t, d in by_scene),
           "marida_scenes_not_in_store": sorted(
               f"{t} {d}" for t, d in MARIDA_SCENES
               if (t, d) not in by_scene),
           "labelled_px": int(len(A["cls"])),
           "hub_reads": stats}
    if not len(A["cls"]):
        res["note"] = "no MARIDA scene in this store"
        _dump(res, out)
        return res
    water = A["cls"] == 7
    both = water & np.isfinite(A["fdi_m"]) & np.isfinite(A["fdi_s"])
    res["i"] = {"marine_water_px": int(water.sum()),
                "with_store_value": int(both.sum()),
                "spearman_pooled": _spearman(A["fdi_m"][both],
                                             A["fdi_s"][both]),
                "pass_threshold": 0.9}
    # per 20 m cell: MARIDA's 10 m FDI averaged over the cell's water pixels
    key = np.array([f"{s}|{c}" for s, c in zip(A["scene"][both],
                                               A["cell"][both])])
    u, inv = np.unique(key, return_inverse=True)
    mm = np.bincount(inv, A["fdi_m"][both]) / np.bincount(inv)
    ss = np.bincount(inv, A["fdi_s"][both]) / np.bincount(inv)
    res["i"]["spearman_per_20m_cell"] = _spearman(mm, ss)
    res["i"]["cells"] = int(len(u))
    # WITHIN-scene vs ACROSS-scene: a per-scene additive offset (the store's
    # aerosol correction against MARIDA's Rayleigh-only one) leaves a
    # within-scene rank untouched and lowers only the pooled one
    cscene = np.array([k.split("|", 1)[0] for k in u])
    per20 = {}
    adj = ss.copy()
    for tg in np.unique(cscene):
        mk = cscene == tg
        per20[tg] = {"cells": int(mk.sum()), "spearman": _spearman(mm[mk],
                                                                   ss[mk])}
        if mk.any():
            adj[mk] = ss[mk] - np.median(ss[mk] - mm[mk])
    vals = [v["spearman"] for v in per20.values()
            if v["spearman"] is not None]
    res["i"]["per_scene_20m_cell"] = per20
    res["i"]["spearman_per_20m_cell_scene_median"] = (
        float(np.median(vals)) if vals else None)
    res["i"]["spearman_per_20m_cell_pooled_after_removing_each_scenes_"
             "median_offset"] = _spearman(mm, adj)
    for tag in per_scene:
        m = both & (A["scene"] == tag)
        per_scene[tag].update(marine_water_with_store=int(m.sum()),
                              spearman=_spearman(A["fdi_m"][m],
                                                 A["fdi_s"][m]))
    rs = [v["spearman"] for v in per_scene.values()
          if v.get("spearman") is not None]
    res["i"]["spearman_per_scene_median"] = (float(np.median(rs)) if rs
                                             else None)
    # THE VERDICT IS READ AT THE STORE'S OWN RESOLUTION: a 20 m cell against
    # the mean of MARIDA's 10 m water pixels inside it. Per 10 m pixel the
    # comparison also measures the 10 m -> 20 m resampling (B08 is native
    # 10 m in MARIDA, a 20 m average in the store), so it is reported beside
    # the verdict, never as it.
    sp = res["i"]["spearman_per_20m_cell"]
    res["i"]["verdict_on"] = "spearman_per_20m_cell"
    res["i"]["pass"] = None if sp is None else bool(sp >= 0.9)
    # (ii)
    tr, te = A["split"] == "train", A["split"] == "test"
    out_ii = {}
    for lab, f in (("store", A["fdi_s"]), ("marida", A["fdi_m"])):
        base = tr & water & np.isfinite(f)
        if base.sum() < 10 or te.sum() == 0:
            out_ii[lab] = {"not_computable": (
                f"{int(base.sum())} TRAIN Marine Water pixel(s) with a value "
                f"and {int(te.sum())} TEST pixel(s) in the store's MARIDA "
                f"scenes — the threshold needs both")}
            continue
        thr = float(np.percentile(f[base], 95.0))
        pred = np.isfinite(f) & (f > thr)
        truth = A["cls"] == 1
        tp = int((pred & truth & te).sum())
        fp = int((pred & ~truth & te).sum())
        fn = int((~pred & truth & te).sum())
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else float("nan")
        out_ii[lab] = {"threshold": thr, "train_water_px": int(base.sum()),
                       "test_px": int(te.sum()),
                       "test_debris_px": int((truth & te).sum()),
                       "tp": tp, "fp": fp, "fn": fn, "precision": p,
                       "recall": r, "f1": f1}
    res["ii"] = out_ii
    if all("f1" in (out_ii.get(k) or {}) for k in ("store", "marida")):
        d1 = abs(out_ii["store"]["f1"] - out_ii["marida"]["f1"])
        res["ii"]["f1_abs_diff"] = d1 if np.isfinite(d1) else None
        res["ii"]["pass"] = (bool(d1 <= 0.05) if np.isfinite(d1) else None)
    res["per_scene"] = per_scene
    debris = A["cls"] == 1
    res["coverage"] = {"debris_px": int(debris.sum()),
                       "debris_px_with_store_value":
                       int((debris & np.isfinite(A["fdi_s"])).sum())}
    _dump(res, out)
    return res


def _dump(res, out):
    if out:
        with open(out + ".part", "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        os.replace(out + ".part", out)


def _dbf_records(path):
    """Minimal dBASE III reader: [{field: value}] (strings and numbers)."""
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[4:8], "little")
    hl = int.from_bytes(raw[8:10], "little")
    rl = int.from_bytes(raw[10:12], "little")
    fields = []
    p = 32
    while raw[p] != 0x0D:
        nm = raw[p:p + 11].split(b"\0")[0].decode()
        fields.append((nm, chr(raw[p + 11]), raw[p + 16]))
        p += 32
    out = []
    for i in range(n):
        rec = raw[hl + i * rl: hl + (i + 1) * rl]
        q = 1
        r = {}
        for nm, kind, ln in fields:
            s = rec[q:q + ln].decode("latin-1").strip()
            q += ln
            r[nm.lower()] = (float(s) if kind in "NF" and s else s)
        out.append(r)
    return out


def plp2019_targets(vector_dir):
    """{date: [(x, y, plastic %)]} from the PLP2019 Vector_Points folders
    (10 m pixel centres, EPSG:32635; plastic % = bags + bottles)."""
    out = {}
    for d in sorted(os.listdir(vector_dir)):
        sub = os.path.join(vector_dir, d)
        dbf = [f for f in os.listdir(sub) if f.lower().endswith(".dbf")]
        prj = [f for f in os.listdir(sub) if f.lower().endswith(".prj")]
        if len(dbf) != 1 or not re.fullmatch(r"\d{8}", d):
            continue
        if prj and "UTM_Zone_35N" not in open(os.path.join(sub, prj[0])
                                                  ).read().replace(" ", "_"):
            raise ValueError(f"{sub}: not UTM 35N")
        day = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        pts = []
        for r in _dbf_records(os.path.join(sub, dbf[0])):
            pl = sum(float(r.get(k) or 0) for k in ("cp_bags", "cp_bottles"))
            pts.append((float(r["point_x"]), float(r["point_y"]), pl,
                        r.get("pixel_name")))
        out[day] = pts
    return out


def _box_rank(fdi, w, rc_box, rng, n_null=200):
    """The maximum FDI inside a row/col box [r0, r1) x [c0, c1), as a
    percentile of the scene's stored water `w`, and ranked against the same
    statistic over `n_null` equal boxes placed at random over the frame
    (a box is used when at least half of it is stored water). The max of
    many pixels is biased upward against a single-pixel percentile; the
    random boxes carry the same bias, which is why both are reported."""
    r0, r1, c0, c1 = rc_box
    H, W = fdi.shape
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, H), min(c1, W)
    sub = fdi[r0:r1, c0:c1]
    ok = np.isfinite(sub)
    out = {"rows": [int(r0), int(r1)], "cols": [int(c0), int(c1)],
           "box_px": int(sub.size), "box_water_px": int(ok.sum())}
    if not ok.any():
        out["box_max_fdi"] = None
        return out
    bx = float(np.nanmax(sub))
    hh, ww = r1 - r0, c1 - c0
    null, tries = [], 0
    while len(null) < n_null and tries < n_null * 50:
        tries += 1
        rr = int(rng.integers(0, H - hh + 1))
        cc = int(rng.integers(0, W - ww + 1))
        q = fdi[rr:rr + hh, cc:cc + ww]
        if np.isfinite(q).mean() >= 0.5:
            null.append(float(np.nanmax(q)))
    out.update(box_max_fdi=bx,
               box_max_percentile_in_scene_water=(
                   float((w < bx).mean() * 100) if w.size else None),
               null_boxes=len(null),
               box_max_rank_among_random_boxes=(
                   float((np.array(null) < bx).mean()) if null else None))
    return out


def report_plp(store, vector_dir=None, box=PLP_BOX_2022, out=None, seed=0,
               boxes_2021=None):
    """Falsifier (iii), as a REPORT: on each Plastic Litter Project date in
    the store, where do the moored targets rank in their scene's water FDI?
    The bar (plan §6): above the 99th percentile of the scene's water.

    2019 dates: the producer's per-pixel points (10 m centres with their
    plastic fraction) mapped to 20 m cells; each target cell's FDI as a
    percentile of the frame's stored water FDI (and of water within 2 km);
    pass if the best target cell is above the 99th. 2022-23 dates: only the
    producer's ACOLITE box is published, so the box's maximum FDI is ranked
    in the scene's water and against random equal boxes. 2021 dates: a box
    per date only if `boxes_2021` ({date: [S, W, N, E]}, e.g. the orthophoto
    footprints) is given; otherwise named as missing. Each frame is read from
    the store in ONE range request (`_FrameReader.frame`)."""
    from rasterio.warp import transform
    meta, grp, acq = _store_reader(store)
    rng = np.random.default_rng(seed)
    pts = plp2019_targets(vector_dir) if vector_dir else {}
    stats = {"requests": 0, "bytes": 0}
    res = {"store": store, "dates": {}, "hub_reads": stats,
           "rule": "pass = the best target pixel (or the box maximum) above "
                   "the 99th percentile of the frame's stored water FDI"}

    def lonlat_box(g, bb):
        s_, w_, n_, e_ = bb
        xs, ys = transform("EPSG:4326", g["crs"], [w_, e_, w_, e_],
                           [s_, s_, n_, n_])
        return (int((g["y0"] - max(ys)) // PX),
                int((g["y0"] - min(ys)) // PX) + 1,
                int((min(xs) - g["x0"]) // PX),
                int((max(xs) - g["x0"]) // PX) + 1)

    for a in acq:
        if a["tile"] != LESVOS or a["day"] not in PLP_DATES:
            continue
        fr = _FrameReader(grp[a["group"]], a["bin"], a["frame"], stats)
        g = fr.sp["grid"]
        fdi = fr.frame(COL["fdi"])
        w = fdi[np.isfinite(fdi)]
        rec = {"l1c": a["l1c"], "processing_baseline":
               a.get("processing_baseline"), "cloud_cover":
               a.get("cloud_cover"), "selection": a.get("selection"),
               "water_px": int(w.size),
               "p99": float(np.percentile(w, 99)) if w.size else None}
        y = a["day"][:4]
        if a["day"] in pts:
            cells = {}
            for (x, yy, pl, nm) in pts[a["day"]]:
                r20 = int((g["y0"] - yy) // PX)
                c20 = int((x - g["x0"]) // PX)
                cells.setdefault((r20, c20), []).append(pl)
            rows = []
            # the LOCAL background: water within 2 km of the targets'
            # centre, so coastal turbid water elsewhere in the tile does not
            # set the bar
            rc = np.array(list(cells))
            r_c, c_c = rc.mean(axis=0)
            k = int(2000 / PX)
            r0, c0 = int(r_c) - k, int(c_c) - k
            loc = fdi[max(r0, 0):r0 + 2 * k + 1, max(c0, 0):c0 + 2 * k + 1]
            yy, xx = np.mgrid[0:loc.shape[0], 0:loc.shape[1]]
            disc = np.hypot(yy + max(r0, 0) - r_c, xx + max(c0, 0) - c_c) <= k
            wl = loc[disc & np.isfinite(loc)]
            for (r, c), pls in sorted(cells.items()):
                v = float(fdi[r, c])
                ok = np.isfinite(v)
                pr = float((w < v).mean() * 100) if ok and w.size else None
                pl = float((wl < v).mean() * 100) if ok and wl.size else None
                rows.append({"row": r, "col": c, "plastic_pct_mean_of_10m":
                             float(np.mean(pls)), "points": len(pls),
                             "fdi": v if ok else None,
                             "percentile_in_scene_water": pr,
                             "percentile_in_water_within_2km": pl})
            rec["local_water_px"] = int(wl.size)
            ok = [r for r in rows if r["fdi"] is not None]
            if len(ok) >= 3:
                rec["spearman_plastic_pct_vs_fdi"] = _spearman(
                    [r["plastic_pct_mean_of_10m"] for r in ok],
                    [r["fdi"] for r in ok])
            best = [r["percentile_in_scene_water"] for r in rows
                    if r["percentile_in_scene_water"] is not None]
            bl = [r["percentile_in_water_within_2km"] for r in rows
                  if r["percentile_in_water_within_2km"] is not None]
            rr_ = rc[:, 0]
            cc_ = rc[:, 1]
            rec.update(kind="2019 per-pixel points", cells=rows,
                       target_cells=len(rows),
                       best_percentile=max(best) if best else None,
                       best_percentile_local=max(bl) if bl else None,
                       cells_with_value=len(best),
                       target_bbox=_box_rank(
                           fdi, w, (int(rr_.min()), int(rr_.max()) + 1,
                                    int(cc_.min()), int(cc_.max()) + 1), rng),
                       pass_p99=(bool(max(best) > 99) if best else None))
        elif y in ("2022", "2023") or (boxes_2021 and a["day"] in
                                       boxes_2021):
            bb = box if y in ("2022", "2023") else boxes_2021[a["day"]]
            br = _box_rank(fdi, w, lonlat_box(g, bb), rng)
            pc = br.get("box_max_percentile_in_scene_water")
            rec.update(kind=("2022-23 producer ACOLITE box"
                             if y in ("2022", "2023")
                             else "2021 orthophoto footprint box"),
                       box=list(bb), **br,
                       pass_p99=(None if pc is None else bool(pc > 99)))
        else:
            rec.update(kind="2021", pass_p99=None, missing=(
                "target positions are published only as orthophotos inside "
                "per-date zips of 5-10 GB (Zenodo 7085112); no footprint "
                "box was given for this date"))
        del fdi
        res["dates"][a["day"]] = rec
    res["plp_dates_not_in_store"] = sorted(set(PLP_DATES)
                                           - set(res["dates"]))
    sc = [d for d, r in res["dates"].items() if r.get("pass_p99") is not None]
    res["summary"] = {"dates_in_store": len(res["dates"]),
                      "dates_scored": len(sc),
                      "dates_pass_p99": sum(1 for d in sc
                                            if res["dates"][d]["pass_p99"]),
                      "dates_unscored": sorted(set(res["dates"]) - set(sc))}
    _dump(res, out)
    return res


def _rc_box(g, bb):
    from rasterio.warp import transform
    s_, w_, n_, e_ = bb
    xs, ys = transform("EPSG:4326", g["crs"], [w_, e_, w_, e_],
                       [s_, s_, n_, n_])
    return (int((g["y0"] - max(ys)) // PX), int((g["y0"] - min(ys)) // PX) + 1,
            int((min(xs) - g["x0"]) // PX), int((max(xs) - g["x0"]) // PX) + 1)


def site_summary(rows, sites):
    """Campaign dates against every other acquisition, per site box, and
    the campaign dates split by YEAR — the 2019, 2021 and 2022-23 targets
    were moored at different places, so a site box is only a target box in
    its own campaign year(s)."""
    from scipy.stats import mannwhitneyu
    summ = {}
    for name in sites:
        def vals(pred):
            return [r["sites"][name]["box_max_percentile_in_background"]
                    for r in rows if pred(r)
                    and "box_max_percentile_in_background" in r["sites"][name]]

        def desc(v):
            return ({"n": len(v), "median": float(np.median(v)),
                     "p90": float(np.percentile(v, 90)),
                     "frac_above_99": float(np.mean(np.array(v) > 99))}
                    if v else {"n": 0})
        c = vals(lambda r: r["campaign"])
        o = vals(lambda r: not r["campaign"])
        sm = {"campaign_n": len(c), "other_n": len(o),
              "campaign": desc(c), "other": desc(o),
              "other_june_october": desc(vals(
                  lambda r: not r["campaign"]
                  and r["day"][5:7] in ("06", "07", "08", "09", "10"))),
              "campaign_by_year": {}}
        for yrs in (("2019",), ("2021",), ("2022", "2023")):
            v = vals(lambda r, yrs=yrs: r["campaign"] and r["day"][:4] in yrs)
            d = desc(v)
            if len(v) >= 3 and len(o) >= 3:
                d["mann_whitney_p_vs_other"] = float(
                    mannwhitneyu(v, o, alternative="greater").pvalue)
            sm["campaign_by_year"]["-".join(yrs)] = d
        if len(c) >= 3 and len(o) >= 3:
            sm["mann_whitney_p_campaign_greater"] = float(
                mannwhitneyu(c, o, alternative="greater").pvalue)
        summ[name] = sm
    return summ


def site_control(store, sites, pad_m=3000.0, workers=4):
    """THE CONTROL (iii) needs: is a target box bright because of the
    TARGETS or because of the SITE? For every Lesvos acquisition in the
    store — campaign dates and the ~385 others alike — the maximum FDI in
    each site box, as a percentile of the stored water within `pad_m` of the
    box (the box itself excluded). Read tile by tile (the window is a few
    tiles), never the whole frame. `sites` = {name: [S, W, N, E]}."""
    from concurrent.futures import ThreadPoolExecutor
    meta, grp, acq = _store_reader(store)
    les = [a for a in acq if a["tile"] == LESVOS]
    stats = {"requests": 0, "bytes": 0}
    T = sh.TILE

    def one(a):
        cnt = {"requests": 0, "bytes": 0}
        fr = _FrameReader(grp[a["group"]], a["bin"], a["frame"], cnt)
        g = fr.sp["grid"]
        k = int(pad_m / PX)
        cache = {}
        out = {"day": a["day"], "campaign": a["day"] in PLP_DATES,
               "cloud_cover": a.get("cloud_cover"), "sites": {}}
        for name, bb in sites.items():
            r0, r1, c0, c1 = _rc_box(g, bb)
            R0, R1 = max(r0 - k, 0), min(r1 + k, fr.sp["H"])
            C0, C1 = max(c0 - k, 0), min(c1 + k, fr.sp["W"])
            win = np.full((R1 - R0, C1 - C0), np.nan, np.float32)
            for ty in range(R0 // T, (R1 - 1) // T + 1):
                for tx in range(C0 // T, (C1 - 1) // T + 1):
                    if (ty, tx) not in cache:
                        t = fr.tile(ty, tx)
                        cache[(ty, tx)] = None if t is None else \
                            t[..., COL["fdi"]]
                    t = cache[(ty, tx)]
                    if t is None:
                        continue
                    a0, a1 = max(R0, ty * T), min(R1, ty * T + t.shape[0])
                    b0, b1 = max(C0, tx * T), min(C1, tx * T + t.shape[1])
                    win[a0 - R0:a1 - R0, b0 - C0:b1 - C0] = \
                        t[a0 - ty * T:a1 - ty * T, b0 - tx * T:b1 - tx * T]
            inb = np.zeros(win.shape, bool)
            inb[r0 - R0:r1 - R0, c0 - C0:c1 - C0] = True
            box = win[inb]
            bg = win[~inb & np.isfinite(win)]
            ok = np.isfinite(box)
            rec = {"box_water_px": int(ok.sum()), "box_px": int(box.size),
                   "background_water_px": int(bg.size)}
            if ok.any() and bg.size >= 1000:
                mx = float(np.nanmax(box))
                rec.update(box_max_fdi=mx,
                           box_max_percentile_in_background=float(
                               (bg < mx).mean() * 100),
                           background_p99=float(np.percentile(bg, 99)))
            out["sites"][name] = rec
        return out, cnt
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for r, cnt in ex.map(one, les):
            rows.append(r)
            stats["requests"] += cnt["requests"]
            stats["bytes"] += cnt["bytes"]
    rows.sort(key=lambda r: r["day"])
    summ = site_summary(rows, sites)
    return {"what": ("max FDI in each site box as a percentile of stored "
                     f"water within {pad_m:.0f} m of it (box excluded), on "
                     "every 35SMD acquisition; campaign = a PLP date"),
            "sites": sites, "summary": summ, "acquisitions": rows,
            "hub_reads": stats}


def falsifiers(store, marida_zip, vector_dir=None, boxes_2021=None,
               out=None):
    """(i), (ii) and (iii) against a built store — local or the Hub — written
    as ONE file. Reports what was measured against each falsifier's bar;
    decides nothing beyond that comparison."""
    t0 = time.time()
    m = check_marida(store, marida_zip)
    p = report_plp(store, vector_dir, boxes_2021=boxes_2021)
    sites = {"plp2022_23_producer_box": list(PLP_BOX_2022)}
    if boxes_2021:
        bx = np.array(list(boxes_2021.values()), np.float64)
        sites["plp2021_median_ortho_footprint"] = [
            round(float(v), 6) for v in np.median(bx, axis=0)]
    if vector_dir:
        from rasterio.warp import transform as _tr
        allp = [q for v in plp2019_targets(vector_dir).values() for q in v]
        lo, la = _tr("EPSG:32635", "EPSG:4326", [q[0] for q in allp],
                     [q[1] for q in allp])
        sites["plp2019_target_points_bbox"] = [
            min(la) - 1e-4, min(lo) - 1e-4, max(la) + 1e-4, max(lo) + 1e-4]
    sc = site_control(store, sites)
    i_ = m.get("i") or {}
    ii = m.get("ii") or {}
    s3 = p["summary"]
    res = {
        "written": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "experiment": "E-092 s2fdi_coast falsifiers (plan §6)",
        "store": store, "marida_zip": str(marida_zip),
        "verdicts": {
            "i": {"bar": "Spearman >= 0.9 between the store's FDI and FDI "
                         "from MARIDA's own bands, MARIDA 'Marine Water' "
                         "pixels, per 20 m cell",
                  "measured": i_.get("spearman_per_20m_cell"),
                  "n_cells": i_.get("cells"),
                  "per_10m_pixel": i_.get("spearman_pooled"),
                  "pass": i_.get("pass")},
            "ii": {"bar": "|F1(store FDI) - F1(MARIDA FDI)| <= 0.05, "
                          "threshold = 95th percentile of TRAIN Marine Water "
                          "FDI, debris-vs-not on MARIDA's TEST split",
                   "f1_store": (ii.get("store") or {}).get("f1"),
                   "f1_marida": (ii.get("marida") or {}).get("f1"),
                   "f1_abs_diff": ii.get("f1_abs_diff"),
                   "pass": ii.get("pass")},
            "iii": {"bar": "moored targets above the 99th percentile of "
                           "their scene's water FDI, per campaign date",
                    **s3,
                    "site_control": sc["summary"],
                    "site_control_note": (
                        "the same boxes on every NON-campaign acquisition: "
                        "if they rank as high there, the brightness belongs "
                        "to the site (shore, shallows), not to the targets")}},
        "inputs": {
            "marida": "Zenodo 5151941 MARIDA.zip (patches, _cl masks, "
                      "splits); FDI from its bands with s2labels.fdi",
            "plp2019": "Zenodo 3752719 PLP2019_dataset.rar, Vector_Points "
                       "(10 m target pixels with plastic %)",
            "plp2021_boxes": ("the georeferenced footprint of each date's "
                              "UAS orthophoto in Zenodo 7085112, read from "
                              "the GeoTIFF header by a streaming inflate of "
                              "the zip member "
                              "(ml/family1/probes/s2fdi_coast_plp2021_boxes"
                              ".json); 17 of 22 dates have one"
                              if boxes_2021 else None),
            "plp2022_23_box": list(PLP_BOX_2022)},
        "i_ii": m, "iii": p, "iii_site_control": sc,
        "wall_s": round(time.time() - t0, 1)}
    _dump(res, out)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="s2fdi_coast: lanes and checks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lanes")
    c = sub.add_parser("check")
    c.add_argument("--store", default=HUB_STORE,
                   help="a local store directory, a URL prefix (default: "
                        "the published store on the Hub), or `parts` — the "
                        "lane parts on the Hub before assembly")
    c.add_argument("--marida-zip", default=MARIDA_ZIP)
    c.add_argument("--plp2019-vectors", default="")
    c.add_argument("--plp2021-boxes", default="",
                   help="JSON {date: [S, W, N, E]} target boxes for 2021")
    c.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "lanes":
        print(json.dumps(plan_lanes(), indent=1))
        return
    boxes = None
    if a.plp2021_boxes:
        with open(a.plp2021_boxes) as fh:
            boxes = json.load(fh)
    r = falsifiers(a.store, a.marida_zip, a.plp2019_vectors or None, boxes,
                   out=a.out or None)
    print(json.dumps(r["verdicts"], indent=1, default=float))


if __name__ == "__main__":
    main()
