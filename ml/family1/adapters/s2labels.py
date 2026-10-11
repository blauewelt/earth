"""Human-labelled Sentinel-2 pixels of marine debris and the sea around it —
one row per LABELLED PIXEL (family 1.3, E-092).

PLAIN ENGLISH. Two published label sets from the same producer (the National
Technical University of Athens group, Kikaki et al.) mark individual 10 m
Sentinel-2 pixels by hand as "marine debris", "Sargassum", "foam", "ship",
"clear marine water", "turbid water", "oil spill" and so on, on images that
were atmospherically corrected for water with ACOLITE. This store is those
pixels: where and when each was imaged, the 11 band reflectances the producer
ships for it, the class the annotator gave it and how confident they were, and
the train / validation / test split the producer published. It is what any
satellite debris index must be scored against (plan §6: an FDI threshold is the
baseline on MARIDA's own test split; `ml/plans/E092_family13_plastic.md`).

SOURCES (both CC BY 4.0, both measured from the sandbox 2026-10-11):

 1. MARIDA, the Marine Debris Archive (Kikaki et al. 2022, PLOS ONE
    17(1):e0262247), Zenodo record 5151941, `MARIDA.zip` (1,164,612,748 B,
    md5 checked against the record's own checksum). 1,381 patches of
    256 x 256 at 10 m in 63 scenes (one scene = one Sentinel-2 tile on one
    date, folder `patches/S2_<d-m-yy>_<tile>/`), each patch an 11-band
    GeoTIFF in the tile's own UTM CRS plus `_cl` (class) and `_conf`
    (confidence) masks; `splits/{train,val,test}_X.txt` name every patch once.
    837,377 labelled pixels, 3,399 of them "Marine Debris" (measured,
    asserted at build time).
 2. MADOS, Marine Debris and Oil Spill (Kikaki et al. 2024, ISPRS J. Photogramm.
    Remote Sens.), Zenodo record 10664073, `MADOS.zip` (4,038,418,740 B).
    174 scenes (`MADOS/Scene_<n>/`), 2,803 crops of 240 x 240 at 10 m, each
    band its own GeoTIFF at its NATIVE resolution (`10/`: 492, 560, 665,
    833 nm; `20/`: 704, 740, 783, 865, 1614, 2202 nm; `60/`: 443 nm — ACOLITE
    names the files by each satellite's own band centre, so a crop's band
    names say whether it is Sentinel-2A or 2B), with `_cl` / `_conf` / `_rep`
    masks at 10 m and 20 m and `splits/{train,val,test}_X.txt` naming every
    crop once. 1,481,155 labelled 10 m pixels (measured).

THE MEASURED FACT THAT SHAPES THE BUILD: MADOS IS NOT GEOREFERENCED. Every
MADOS GeoTIFF carries an identity geotransform and a placeholder EPSG:4326 —
no tile, no date, no position (measured on the files' TIFF tags; the
producer's own `utils/stack_patches.py` writes its stacks as
"non-georeferenced (just to be recognised from gis)"). A row needs a place and
a time, so a MADOS crop enters the store only where it can be LOCATED, and the
only reference that can locate it is MARIDA, which is georeferenced and comes
from the same producer:

  * Both sets are ACOLITE Rayleigh-corrected reflectance of the same Level-1C
    images, but MADOS was reprocessed: practically no 10 m value is
    bit-identical to MARIDA's (at most 0.013 % of a crop, on the crops
    sampled), while the two agree to Pearson r >= 0.985 in every 10 m and
    20 m band of every located crop (r > 0.99 in all but 6; `match_min_r_by_
    band`).
    A reprocessing of the same digital numbers is an increasing map per band, so
    the ORDER of neighbouring pixels is preserved exactly. `rank_codes` turns
    each pixel's 3 x 3 neighbourhood in B02, B03, B04 and B08 into a 32-bit
    pattern of "brighter than my neighbour" bits (two adjacent pixels = one
    64-bit code); every code of every MARIDA patch is indexed, every code of a
    MADOS crop is looked up, each code that occurs ONCE in MARIDA votes for
    (patch, row offset, column offset), and a crop is located when its top
    vote has at least `MATCH_MIN_VOTES` and at least `MATCH_MARGIN` times the
    best vote for any inconsistent position. Measured on the real sets: 1,488
    of 2,803 crops located, every one with >= 54 votes (median 6,196); no
    crop that is not located scores more than 3.
  * A located crop inherits the matched patch's CRS, its scene's date and its
    tile, and its pixel grid is the patch's grid shifted by the vote's offset
    (the crops sit on the 20 m grid: every offset is even, measured).
    CHECKS that refuse the build: the crop's own satellite (from its band
    names) must equal the Copernicus catalogue's satellite for the matched
    MARIDA scene, and every located crop of one MADOS scene must point at one
    date.
  * What cannot be located is LEFT OUT AND COUNTED, by class: 1,315 crops and
    772,971 labelled pixels — exactly the 111 MADOS scenes that are not
    MARIDA's (the 63 located scenes are MARIDA's 63), so MADOS's oil spills,
    oil platforms, jellyfish and sea snot are all in the part that cannot be
    placed.
  * MEASURED OVERLAP: MADOS re-labels 707,583 of MARIDA's 837,377 pixels, and
    every change is a renamed or merged class — Sparse Sargassum -> Sparse
    Floating Algae 2,357, Waves -> Waves & Wakes 5,827, Wakes -> Waves &
    Wakes 8,482, Mixed Water -> Marine Water 410; the 3,399 Marine Debris
    pixels and every other class are unchanged, and no confidence changes.
    Inside the located crops MADOS adds 587 pixels of its own and leaves
    54,501 of MARIDA's unlabelled (46,669 Clouds, 7,767 Cloud Shadows, 65
    Marine Water — classes MADOS does not have).

THE TWO SETS TOGETHER (plan §4 and the brief's precedence rule). A pixel is the
same pixel in both when it is the same (date, tile, UTM position).
  * labelled in MARIDA only -> a MARIDA row;
  * labelled in both -> ONE row, the producer's LATER version (MADOS) wins for
    the label, the confidence and the band values (labels and bands are kept
    from the same version), MARIDA's split is kept, qc bit 64 says so; every
    label is counted per (MARIDA class, MADOS class) in `overlap_labels`,
    changed or not;
  * labelled in a located MADOS crop only -> a MADOS row with MADOS's split;
    when such a pixel lies inside a MARIDA patch whose split differs, that is
    counted (`mados_only_inside_marida_patch_split`), because a model trained
    on MADOS's train split would then see pixels inside MARIDA's test patches;
  * labelled in MARIDA but left unlabelled by a MADOS crop that covers it ->
    stays a MARIDA row, counted by class;
  * the same ground pixel labelled in two MADOS crops of one scene (crops
    overlap) -> the first in the producer's crop order, the rest counted;
  * the same ground pixel labelled in two MARIDA tiles of one date (adjacent
    tiles overlap) -> kept twice, as the producer published them, counted
    (`marida_same_ground_two_tiles`; measured 50, two of them with different
    classes).

WHAT A ROW IS.
  time_s    the Level-1C datatake's sensing start, from the Copernicus Data
            Space catalogue (`ContentDate/Start` of the scene's MSIL1C
            product for that tile on that date — the same instant the
            product's name carries), floored to the second. The DATE is the
            producer's (MARIDA's folder name `S2_<d-m-yy>_<tile>`; a located
            MADOS crop takes its MARIDA scene's). The catalogue is asked per
            scene and must answer one (satellite, instant) — reprocessed
            copies of a product share both. A scene it cannot answer is
            placed at 12:00 UTC on its date with qc bit 32 (the `mplastic` /
            `gbif` precedent for a date without a time). Measured: all 63
            MARIDA scenes answer (60 one product, 3 two copies of one).
  lat, lon  the pixel CENTRE in WGS84: the patch's affine transform at
            (col + 0.5, row + 0.5), then `rasterio.warp.transform` from the
            patch's UTM CRS (no pyproj).
  platform  platform_hash("MARIDA|S2_<d-m-yy>_<tile>") or
            platform_hash("MADOS|Scene_<n>") — one hash per producer scene.
            A pixel both sets label keeps MARIDA's: its place and time come
            from MARIDA's scene.
  values    C = 14: rho_B01 ... rho_B12 (11 bands, below), confidence, split,
            s2_unit.
     rho_<band>  ACOLITE Rayleigh-corrected reflectance (`rhorc`), as the
                 producer ships it. MARIDA: the 11 bands of the patch in the
                 producer's order (`utils/assets.py` `s2_mapping`: nm440, 490,
                 560, 665, 705, 740, 783, 842, 865, 1600, 2200 = B01 B02 B03
                 B04 B05 B06 B07 B08 B8A B11 B12), already resampled by the
                 producer to 10 m. MADOS: each band at its native resolution,
                 upsampled to the 10 m crop by NEAREST NEIGHBOUR exactly as the
                 producer's own `utils/stack_patches.py` does (an integer
                 factor, so 20 m pixel = row // 2, 60 m pixel = row // 6).
                 Measured on located crops: the 20 m bands then agree with
                 MARIDA's own 10 m resampling to r >= 0.997; B01's 60 m crop
                 agrees less well (r down to 0.50: its phase against the 60 m
                 grid is the producer's cropping, unverifiable from the files),
                 and is kept as shipped.
                 Bounds [-0.02, 1.7] (the `refl05` precedent for reflectance:
                 a dark-water value a few thousandths below zero is noise around
                 zero, not a fault; measured minima -0.0083 in MARIDA and
                 -0.0076 in MADOS). Outside -> NaN, counted.
     confidence  the producer's annotator confidence: 1 high, 2 moderate,
                 3 low (both sets use the same three codes).
     split       0 train, 1 validation, 2 test: MARIDA's where the pixel is in
                 MARIDA, MADOS's otherwise (qc bit 128 without bit 64 says
                 which). The plan's falsifier reads MARIDA's test split only.
     s2_unit     1 Sentinel-2A, 2 Sentinel-2B, from the catalogue (and, for a
                 MADOS row, cross-checked against the crop's band names); NaN
                 where the catalogue did not answer. Needed because the FDI's
                 coefficient differs per satellite (plan §4).
  qc        bits 0-4: the JOINT class code (`JOINT_CLASSES`; MARIDA's own
            codes 1-15 unchanged, MADOS's mapped by NAME: the same name is the
            same code, a class only one set has gets its own code — so
            "Sparse Sargassum" (MARIDA) and "Sparse Floating Algae" (MADOS's
            renamed, broader class) are 3 and 16, "Waves" and "Wakes" (MARIDA)
            12 and 14 against MADOS's merged "Waves & Wakes" 18; Clouds, Cloud
            Shadows and Mixed Water are MARIDA's only, Oil Spill, Oil Platform,
            Jellyfish and Sea snot MADOS's only); bit 5 (32): time of day not
            from the catalogue (12:00 UTC); bit 6 (64): the pixel is labelled
            in BOTH sets (MADOS's label); bit 7 (128): the row's values are
            MADOS's.

FOOTPRINT. A 10 m pixel: log2_fp = log2(0.010 km / 27.83 km) = -11.44 (E-078
§2's unit; the true support is written below the -4 point label, as `fire`
does for a 1 km MODIS pixel). An image line is sensed in milliseconds:
log2_dt = log2((1 s / 86400 s) / 5 d) = -18.72, the precedent of the other
Sentinel-2-derived store, `cat_s2` (`_stac.log2_dt_for_seconds(1.0)`).

OUT OF THIS BUILD, AND WHY (Zenodo record texts read 2026-10-11).
  * The Plastic Litter Project (University of the Aegean): its 2021 record
    (Zenodo 7085112) is 23 files, 159.5 GB, of Sentinel-2 Level-1C products,
    ACOLITE NetCDFs, drone orthophotos and target photographs per acquisition
    date — it publishes NO per-pixel label file; where the moored targets are
    is only visible in the drone imagery. (The 2019 record, Zenodo 3752719,
    does ship point vectors of "percentage plastic presence for each
    Sentinel-2 pixel" — 69 MB, a candidate for a later pass, not in this one.)
  * FloatingObjects (Mifdal et al. 2021): 16.6 GB of LINE annotations on 26
    scenes (plan §2b), not labelled pixels.
"""
import ast
import datetime as dt
import hashlib
import html
import json
import os
import re
import urllib.parse
import zipfile

import numpy as np

import build_family10_stores as f10b

# ================================================================ sources ==
ZENODO_RECORD = "https://zenodo.org/api/records/{rec}"
RECORDS = {"MARIDA": ("5151941", "MARIDA.zip"),
           "MADOS": ("10664073", "MADOS.zip")}
DATASETS = ("MARIDA", "MADOS")          # precedence order: earlier first
ASSETS = {
    # the producers' own band order and class tables, in their code
    "MARIDA": ("https://raw.githubusercontent.com/marine-debris/"
               "marine-debris.github.io/main/utils/assets.py",
               "cat_mapping"),
    "MADOS": ("https://raw.githubusercontent.com/gkakogeorgiou/mados/master/"
              "utils/assets.py", "mados_cat_mapping")}
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# ================================================================== bands ==
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
         "B11", "B12")
# the producers' `s2_mapping` keys in band order (identical in both repos)
ASSET_NM = ("nm440", "nm490", "nm560", "nm665", "nm705", "nm740", "nm783",
            "nm842", "nm865", "nm1600", "nm2200")
# ACOLITE's file names: each satellite's own band centre, rounded, in BANDS
# order (measured: the 1,597 S2A crops and 1,206 S2B crops carry exactly these)
ACOLITE_NM = {"S2A": (443, 492, 560, 665, 704, 740, 783, 833, 865, 1614,
                      2202),
              "S2B": (442, 492, 559, 665, 704, 739, 780, 833, 864, 1610,
                      2186)}
RES_M = (60, 10, 10, 10, 20, 20, 20, 10, 20, 20, 20)
CODE_BANDS = (1, 2, 3, 7)               # B02 B03 B04 B08: native 10 m
RHO_UNIT = "reflectance (ACOLITE Rayleigh-corrected, rhorc; dimensionless)"
RHO_LO, RHO_HI = -0.02, 1.7

# ================================================================ classes ==
MARIDA_CLASSES = {
    1: "Marine Debris", 2: "Dense Sargassum", 3: "Sparse Sargassum",
    4: "Natural Organic Material", 5: "Ship", 6: "Clouds", 7: "Marine Water",
    8: "Sediment-Laden Water", 9: "Foam", 10: "Turbid Water",
    11: "Shallow Water", 12: "Waves", 13: "Cloud Shadows", 14: "Wakes",
    15: "Mixed Water"}
MADOS_CLASSES = {
    1: "Marine Debris", 2: "Dense Sargassum", 3: "Sparse Floating Algae",
    4: "Natural Organic Material", 5: "Ship", 6: "Oil Spill",
    7: "Marine Water", 8: "Sediment-Laden Water", 9: "Foam",
    10: "Turbid Water", 11: "Shallow Water", 12: "Waves & Wakes",
    13: "Oil Platform", 14: "Jellyfish", 15: "Sea snot"}
SOURCE_CLASSES = {"MARIDA": MARIDA_CLASSES, "MADOS": MADOS_CLASSES}
# the joint table: MARIDA's codes unchanged, then every MADOS-only name
JOINT_CLASSES = dict(MARIDA_CLASSES)
for _n in MADOS_CLASSES.values():
    if _n not in JOINT_CLASSES.values():
        JOINT_CLASSES[max(JOINT_CLASSES) + 1] = _n
JOINT_OF_NAME = {v: k for k, v in JOINT_CLASSES.items()}
TO_JOINT = {ds: {k: JOINT_OF_NAME[v] for k, v in tab.items()}
            for ds, tab in SOURCE_CLASSES.items()}
CLASS_BITS = 31
QC_NO_TIME = 32
QC_BOTH = 64
QC_MADOS = 128
CONF_CODES = {1: "High", 2: "Moderate", 3: "Low"}
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SAT_CODE = {"S2A": 1, "S2B": 2}
# FDI's (lambda8 - lambda4) / (lambda11 - lambda4) * 10 per satellite (plan §4)
FDI_COEF = {"S2A": 1.7722, "S2B": 1.7770}

# THE VERIFIED ARCHIVES (measured 2026-10-11). When the archive the record
# serves is byte-for-byte the one measured (its md5), its counts must be the
# measured ones; a different archive (a new Zenodo version, the smoke's) is
# read on its own terms and its counts are only recorded.
MEASURED = {
    "MARIDA": {"md5": "9bf32266f6e3711c9dfa3699b856c76f",
               "marida_patches": 1381, "marida_scenes": 63,
               "marida_labelled_pixels": 837377, "marida_debris": 3399},
    "MADOS": {"md5": "1076b25d2797be3095d82a61105c7380",
              "mados_crops": 2803, "mados_scenes": 174,
              "mados_labelled_pixels": 1481155}}


def check_measured(files, counts):
    got = dict(counts)
    got["marida_debris"] = counts.get("marida_by_class", {}).get(
        "Marine Debris", 0)
    for ds, m in MEASURED.items():
        if files[ds]["md5"] != m["md5"]:
            counts[f"{ds.lower()}_is_the_measured_archive"] = 0
            continue
        counts[f"{ds.lower()}_is_the_measured_archive"] = 1
        for k, want in m.items():
            if k != "md5" and got.get(k) != want:
                refuse(f"{ds}: {k} = {got.get(k)} in the archive measured as "
                       f"{want} (same md5) — the reader has changed")


# ============================================================== matching ==
MATCH_MIN_VOTES = 20
MATCH_MARGIN = 10.0
UNLOCATED_VOTE_BUCKETS = (0, 1, 2, 5, 10, 20, 50)

CHANNELS = tuple((f"rho_{b}", RHO_UNIT, RHO_LO, RHO_HI) for b in BANDS) + (
    ("confidence", "annotator confidence: 1 high, 2 moderate, 3 low", 1.0,
     3.0),
    ("split", "0 train, 1 validation, 2 test (MARIDA's split where the pixel "
     "is in MARIDA, else MADOS's)", 0.0, 2.0),
    ("s2_unit", "1 Sentinel-2A, 2 Sentinel-2B", 1.0, 2.0))
NAMES = [c[0] for c in CHANNELS]
COL = {n: i for i, n in enumerate(NAMES)}
C = len(CHANNELS)

MARIDA_SCENE = re.compile(r"^patches/S2_(\d{1,2})-(\d{1,2})-(\d{2})_"
                          r"([0-9]{2}[A-Z]{3})/$")
MARIDA_PATCH = re.compile(r"^patches/(S2_(\d{1,2}-\d{1,2}-\d{2})_"
                          r"([0-9]{2}[A-Z]{3}))/S2_\2_\3_(\d+)\.tif$")
MADOS_CL = re.compile(r"^MADOS/(Scene_(\d+))/10/\1_L2R_cl_(\d+)\.tif$")
MADOS_BAND = re.compile(r"^MADOS/(Scene_\d+)/(10|20|60)/\1_L2R_rhorc_(\d+)_"
                        r"(\d+)\.tif$")


class FormatError(ValueError):
    """A source that is not the layout verified above."""


def refuse(msg):
    raise FormatError(f"REFUSING s2labels: {msg}")


# ============================================================ the record ==
def description_text(rec):
    t = re.sub(r"<[^>]+>", " ", rec.get("metadata", {}).get("description", ""))
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def class_table_of(rec):
    """The record description's 'Digital Numbers and Classes' table."""
    t = description_text(rec)
    m = re.search(r"Digital Numbers and Classes[^:]*:(.*?)"
                  r"(?:The mapping between Digital Numbers and Confidence|$)",
                  t)
    if not m:
        refuse("the Zenodo record's description has no 'Digital Numbers and "
               "Classes' table")
    out = {}
    for code, name in re.findall(r"(\d+)\s*[:.]\s+(.+?)(?=\s+\d+\s*[:.]\s|$)",
                                 m.group(1).strip()):
        out[int(code)] = name.strip()
    out.pop(0, None)                      # MADOS's "0. Non-annotated"
    return out


def zip_file_of(rec, key):
    for f in rec.get("files", []):
        if f.get("key") == key:
            cs = str(f.get("checksum", ""))
            if not cs.startswith("md5:"):
                refuse(f"{key}: the record's checksum {cs!r} is not md5")
            return {"size": int(f["size"]), "md5": cs[4:],
                    "url": f["links"]["self"]}
    refuse(f"the Zenodo record lists no {key} "
           f"({[f.get('key') for f in rec.get('files', [])]})")


def check_record(ds, rec):
    lic = (rec.get("metadata", {}).get("license") or {}).get("id")
    if lic != "cc-by-4.0":
        refuse(f"{ds}: the record's licence is {lic!r}, not cc-by-4.0")
    tab = class_table_of(rec)
    if tab != SOURCE_CLASSES[ds]:
        refuse(f"{ds}: the record's class table {tab} is not the one this "
               f"adapter maps ({SOURCE_CLASSES[ds]})")
    return tab


def assets_dict(text, var):
    """A dict literal assigned to `var` in the producer's assets.py."""
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == var for t in node.targets):
            return ast.literal_eval(node.value)
    refuse(f"assets.py assigns no {var}")


def check_assets(ds, text):
    """The producer's own band order and class table, as their code reads
    them; anything else is a layout this adapter was not verified on."""
    s2 = assets_dict(text, "s2_mapping")
    order = [k for k, _v in sorted(s2.items(), key=lambda kv: kv[1])
             if k.startswith("nm")]
    if tuple(order) != ASSET_NM:
        refuse(f"{ds}: assets.py's band order {order} is not {ASSET_NM}")
    cats = {v: k for k, v in assets_dict(text, ASSETS[ds][1]).items()}
    if cats != SOURCE_CLASSES[ds]:
        refuse(f"{ds}: assets.py's classes {cats} differ from the record's")


# ============================================================ catalogue ==
def cdse_url(tile, day):
    nxt = day + dt.timedelta(days=1)
    filt = (f"Collection/Name eq 'SENTINEL-2' and contains(Name,'_T{tile}_') "
            f"and contains(Name,'MSIL1C') and ContentDate/Start ge "
            f"{day.isoformat()}T00:00:00.000Z and ContentDate/Start lt "
            f"{nxt.isoformat()}T00:00:00.000Z")
    return CDSE_ODATA + "?" + urllib.parse.urlencode({"$filter": filt,
                                                       "$top": "50"})


def cdse_answer(doc):
    """{'value': [...]} -> (satellite, seconds since 1982, n products) or
    (None, None, n) when the products do not agree on one instant."""
    prods = doc.get("value", [])
    got = set()
    for p in prods:
        name = p.get("Name", "")
        sat = name[:3]
        start = (p.get("ContentDate") or {}).get("Start")
        if sat not in SAT_CODE or not start:
            continue
        ts = dt.datetime.strptime(start[:19], "%Y-%m-%dT%H:%M:%S")
        got.add((sat, int((ts - dt.datetime(1982, 1, 1)).total_seconds())))
    if len(got) == 1:
        s, t = got.pop()
        return s, t, len(prods)
    return None, None, len(prods)


# ================================================================ codes ==
def rank_codes(stack):
    """(4, H, W) -> (H-2, W-3) uint64: per pixel the 'brighter than each of
    my 8 neighbours' bits in four bands, two horizontally adjacent pixels per
    code. Invariant under any increasing per-band map (a reprocessing)."""
    stack = np.asarray(stack, np.float32)
    _, H, W = stack.shape
    code = np.zeros((H - 2, W - 2), np.uint32)
    bit = 0
    for b in range(stack.shape[0]):
        x = stack[b]
        c0 = x[1:-1, 1:-1]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = x[1 + dr:H - 1 + dr, 1 + dc:W - 1 + dc]
                code |= (c0 > nb).astype(np.uint32) << np.uint32(bit)
                bit += 1
    return (code[:, :-1].astype(np.uint64) << np.uint64(32)) | \
        code[:, 1:].astype(np.uint64)


class CodeIndex:
    """Every MARIDA code, sorted; a code that occurs once has an owner."""

    def __init__(self, codes):
        # codes: (P, h, w) uint64
        self.shape = codes.shape
        flat = codes.ravel()
        self.order = np.argsort(flat, kind="stable")
        self.sorted = flat[self.order]

    def votes(self, q_codes):
        """(h', w') codes of one crop -> {(patch, dr, dc): votes}."""
        q = q_codes.ravel()
        lo = np.searchsorted(self.sorted, q, "left")
        hi = np.searchsorted(self.sorted, q, "right")
        hit = np.flatnonzero(hi - lo == 1)
        if hit.size == 0:
            return {}
        own = self.order[lo[hit]]
        p, rr, cc = np.unravel_index(own, self.shape)
        r, c = np.unravel_index(hit, q_codes.shape)
        key = np.stack([p, rr - r, cc - c], 1).astype(np.int64)
        u, n = np.unique(key, axis=0, return_counts=True)
        return {(int(a), int(b), int(d)): int(k) for (a, b, d), k in
                zip(u, n)}


# ============================================================ geometry ==
def pixel_centres(T, rows, cols):
    """Affine (rasterio) at (col + 0.5, row + 0.5) -> (x, y) float64."""
    c = np.asarray(cols, np.float64) + 0.5
    r = np.asarray(rows, np.float64) + 0.5
    return T.c + c * T.a + r * T.b, T.f + c * T.d + r * T.e


def to_wgs84(crs, x, y):
    from rasterio.warp import transform
    if len(x) == 0:
        return np.zeros(0), np.zeros(0)
    lon, lat = transform(crs, "EPSG:4326", list(map(float, x)),
                         list(map(float, y)))
    return np.asarray(lat, np.float64), np.asarray(lon, np.float64)


def ground_key(day, tile, x, y):
    """The same pixel in both sets: (date, tile, UTM centre in decimetres)."""
    return (day, tile, int(round(float(x) * 10)), int(round(float(y) * 10)))


def marida_day(s):
    d, m, y = (int(v) for v in s.split("-"))
    return dt.date(2000 + y, m, d)


# ================================================================ reading ==
def _read_member(zpath, name, bands=None):
    import rasterio
    with rasterio.open(f"/vsizip/{zpath}/{name}") as d:
        a = d.read(bands) if bands is not None else d.read()
        return a, d.transform, d.crs


def read_splits(z, prefix, counts, ds):
    out = {}
    for s in SPLIT_CODE:
        n = f"{prefix}splits/{s}_X.txt"
        if n not in z.namelist():
            refuse(f"{ds}: the archive has no {n}")
        for line in z.read(n).decode("utf-8").split():
            if line in out:
                refuse(f"{ds}: {line} is in two splits ({out[line]}, {s})")
            out[line] = s
    counts[f"{ds.lower()}_split_entries"] = len(out)
    return out


def marida_patches(z):
    names = set(z.namelist())
    scenes = sorted({m.group(0) for n in names
                     for m in [MARIDA_SCENE.match(n)] if m})
    pats = []
    for n in names:
        m = MARIDA_PATCH.match(n)
        if not m:
            continue
        base = n[:-4]
        for suf in ("_cl.tif", "_conf.tif"):
            if base + suf not in names:
                refuse(f"MARIDA: {n} has no {base + suf}")
        sc, date, tile, k = m.group(1), m.group(2), m.group(3), int(m.group(4))
        pats.append({"scene": sc, "date": date, "tile": tile, "k": k,
                     "name": n, "key": f"{date}_{tile}_{k}"})
    if not pats:
        refuse("MARIDA: the archive lists no patch")
    pats.sort(key=lambda p: (marida_day(p["date"]), p["tile"], p["k"]))
    return pats, scenes


def mados_crops(z):
    names = set(z.namelist())
    bands = {}
    for n in names:
        m = MADOS_BAND.match(n)
        if m:
            bands.setdefault((m.group(1), int(m.group(4))), {})[
                int(m.group(3))] = (m.group(2), n)
    out = []
    for n in names:
        m = MADOS_CL.match(n)
        if not m:
            continue
        sc, num, k = m.group(1), int(m.group(2)), int(m.group(3))
        conf = f"MADOS/{sc}/10/{sc}_L2R_conf_{k}.tif"
        if conf not in names:
            refuse(f"MADOS: {n} has no {conf}")
        have = bands.get((sc, k), {})
        sat = [s for s, nm in ACOLITE_NM.items() if set(nm) == set(have)]
        if len(sat) != 1:
            refuse(f"MADOS: {sc} crop {k} carries bands {sorted(have)}, "
                   f"neither satellite's ACOLITE set")
        sat = sat[0]
        files = []
        for nm, res in zip(ACOLITE_NM[sat], RES_M):
            r, path = have[nm]
            if int(r) != res:
                refuse(f"MADOS: {path} is under {r}/, expected {res}/")
            files.append(path)
        out.append({"scene": sc, "num": num, "k": k, "cl": n, "conf": conf,
                    "sat": sat, "bands": files, "key": f"{sc}_{k}"})
    if not out:
        refuse("MADOS: the archive lists no crop")
    out.sort(key=lambda c: (c["num"], c["k"]))
    return out


def read_mados_crop(zpath, c):
    """-> (bands (11, H, W) at 10 m by the producer's nearest stacking, cl,
    conf)."""
    cl, _, _ = _read_member(zpath, c["cl"], 1)
    conf, _, _ = _read_member(zpath, c["conf"], 1)
    H, W = cl.shape
    out = np.empty((len(BANDS), H, W), np.float32)
    rr = np.arange(H)
    cc = np.arange(W)
    for i, (path, res) in enumerate(zip(c["bands"], RES_M)):
        a, _, _ = _read_member(zpath, path, 1)
        f = res // 10
        if a.shape != (H // f, W // f) or H % f or W % f:
            refuse(f"MADOS: {path} is {a.shape}, not the 10 m crop "
                   f"{(H, W)} / {f}")
        out[i] = a[rr // f][:, cc // f]
    return out, cl, conf


# ======================================================== the assembly ==
def _ground_xy_keys(day, tile, x, y):
    return [ground_key(day, tile, a, b) for a, b in zip(x, y)]


def locate(idx, pmeta, crop_codes):
    """One crop's votes -> (patch, dr, dc, votes) or (None, best votes)."""
    votes = idx.votes(crop_codes)
    if not votes:
        return None, 0
    (pi, dr, dc), nv = max(votes.items(), key=lambda kv: kv[1])
    if nv < MATCH_MIN_VOTES:
        return None, nv
    m = pmeta[pi]
    ox = m["T"].c + dc * m["T"].a
    oy = m["T"].f + dr * m["T"].e
    rival = 0
    for (pj, er, ec), nn in votes.items():
        if nn < 2 or nn <= rival:
            continue
        mj = pmeta[pj]
        same = (mj["day"] == m["day"] and str(mj["crs"]) == str(m["crs"])
                and abs(mj["T"].c + ec * mj["T"].a - ox) < 0.5
                and abs(mj["T"].f + er * mj["T"].e - oy) < 0.5)
        if not same:
            rival = nn
    if nv < MATCH_MARGIN * max(rival, 1):
        return None, nv
    return (pi, dr, dc, nv), nv


def _bump(d, k, n=1):
    d[k] = d.get(k, 0) + n


def build_rows(mzip, dzip, catalogue, splits_m, splits_d, counts,
               progress=None):
    """Both archives -> rows (dict of lists, in yield order); fills counts."""
    zm = zipfile.ZipFile(mzip)
    zd = zipfile.ZipFile(dzip)
    pats, scenes = marida_patches(zm)
    counts["marida_patches"] = len(pats)
    counts["marida_scenes"] = len({p["scene"] for p in pats})
    if counts["marida_scenes"] != len(scenes):
        refuse(f"MARIDA: {len(scenes)} scene folders but patches in "
               f"{counts['marida_scenes']}")
    for p in pats:
        if p["key"] not in splits_m:
            refuse(f"MARIDA: patch {p['key']} is in no split")
    if len(splits_m) != len(pats):
        refuse(f"MARIDA: {len(splits_m)} split entries for {len(pats)} "
               f"patches")

    # ---- MARIDA: every labelled pixel, and every patch's codes ----------
    rows = {"t": [], "x": [], "y": [], "crs": [], "v": [], "plat": [],
            "qc": [], "key": []}
    cls_m = {}
    codes, pmeta = [], []
    shape0 = None
    where = {}                 # ground key -> MARIDA row index
    for i, p in enumerate(pats):
        a, T, crs = _read_member(mzip, p["name"])
        cl, T1, _ = _read_member(mzip, p["name"][:-4] + "_cl.tif", 1)
        cf, T2, _ = _read_member(mzip, p["name"][:-4] + "_conf.tif", 1)
        if a.shape[0] != len(BANDS):
            refuse(f"MARIDA: {p['name']} has {a.shape[0]} bands, not 11")
        if T1 != T or T2 != T or cl.shape != a.shape[1:]:
            refuse(f"MARIDA: {p['name']}'s masks are not on its grid")
        if shape0 is None:
            shape0 = a.shape[1:]
        elif a.shape[1:] != shape0:
            refuse(f"MARIDA: {p['name']} is {a.shape[1:]}, not {shape0}")
        if not np.array_equal(cl, np.round(cl)) or cl.min() < 0 or \
                cl.max() > max(MARIDA_CLASSES):
            refuse(f"MARIDA: {p['name']}_cl holds a value outside 0..15")
        day = marida_day(p["date"])
        sat, t, _n = catalogue[(p["date"], p["tile"])]
        codes.append(rank_codes(a[list(CODE_BANDS)]))
        tt, q0 = (t, 0) if t is not None else (
            (day - dt.date(1982, 1, 1)).days * 86400 + 43200, QC_NO_TIME)
        meta = {"T": T, "crs": crs, "day": day, "tile": p["tile"],
                "scene": p["scene"], "split": splits_m[p["key"]],
                "sat": sat, "t": tt, "q0": q0, "H": a.shape[1],
                "W": a.shape[2], "rows": np.zeros(0, np.int64)}
        pmeta.append(meta)
        rr, cc = np.nonzero(cl > 0)
        if rr.size == 0:
            continue
        lab = cl[rr, cc].astype(np.int64)
        conf = cf[rr, cc].astype(np.int64)
        if set(np.unique(conf).tolist()) - set(CONF_CODES):
            refuse(f"MARIDA: {p['name']}_conf holds "
                   f"{sorted(set(np.unique(conf).tolist()))} on labelled "
                   f"pixels")
        x, y = pixel_centres(T, rr, cc)
        v = np.full((rr.size, C), np.nan)
        v[:, :len(BANDS)] = a[:, rr, cc].T
        v[:, COL["confidence"]] = conf
        v[:, COL["split"]] = SPLIT_CODE[splits_m[p["key"]]]
        if sat is not None:
            v[:, COL["s2_unit"]] = SAT_CODE[sat]
        j0 = len(rows["t"])
        for j, k in enumerate(_ground_xy_keys(day, p["tile"], x, y)):
            if k in where:
                refuse(f"MARIDA: two labelled pixels at {k}")
            where[k] = j0 + j
            rows["key"].append(k)
        n = rr.size
        rows["t"] += [tt] * n
        rows["x"] += x.tolist()
        rows["y"] += y.tolist()
        rows["crs"] += [crs] * n
        rows["v"] += list(v)
        rows["plat"] += [f"MARIDA|{p['scene']}"] * n
        rows["qc"] += [TO_JOINT["MARIDA"][int(c_)] | q0 for c_ in lab]
        meta["rows"] = np.arange(j0, j0 + n)
        for c_ in lab.tolist():
            _bump(cls_m, MARIDA_CLASSES[c_])
        if progress and (i + 1) % 200 == 0:
            progress(f"s2labels MARIDA {i + 1}/{len(pats)}")
    n_marida = len(rows["t"])
    counts["marida_labelled_pixels"] = n_marida
    counts["marida_by_class"] = cls_m
    # the FALSIFIER'S BASELINE straight from MARIDA's own files, before any
    # MADOS label replaces one (float32 bands, MARIDA's labels)
    vm = np.asarray(rows["v"], np.float64).reshape(n_marida, C)
    qm = np.asarray(rows["qc"], np.uint8)
    spm = vm[:, COL["split"]]
    counts["fdi_baseline_marida_archive"] = fdi_baseline(
        vm, qm, spm == 0, spm == 2)
    del vm
    # the same ground pixel in two tiles of one date (adjacent tiles overlap)
    by_ground, same_g, same_g_diff = {}, 0, 0
    for j, k in enumerate(rows["key"]):
        g = (k[0], str(rows["crs"][j]), k[2], k[3])
        if g in by_ground:
            same_g += 1
            if (rows["qc"][by_ground[g]] ^ rows["qc"][j]) & CLASS_BITS:
                same_g_diff += 1
        else:
            by_ground[g] = j
    counts["marida_same_ground_two_tiles"] = same_g
    counts["marida_same_ground_two_tiles_class_differs"] = same_g_diff
    del by_ground

    # ---- MADOS: locate every crop, then merge ---------------------------
    crops = mados_crops(zd)
    counts["mados_crops"] = len(crops)
    counts["mados_scenes"] = len({c["scene"] for c in crops})
    for c in crops:
        if c["key"] not in splits_d:
            refuse(f"MADOS: crop {c['key']} is in no split")
    if len(splits_d) != len(crops):
        refuse(f"MADOS: {len(splits_d)} split entries for {len(crops)} crops")
    idx = CodeIndex(np.stack(codes))
    del codes
    patches_of = {}
    for pi, m in enumerate(pmeta):
        patches_of.setdefault((m["day"], m["tile"]), []).append(pi)
    cls_d, cls_unloc, overlap, leak = {}, {}, {}, {}
    unloc_votes = {str(b): 0 for b in UNLOCATED_VOTE_BUCKETS}
    conf_changed = mados_only = dup_d = dup_d_diff = 0
    located_n = sat_checked = 0
    min_vote, max_unloc = None, 0
    band_r, low_r, votes_located = {}, 0, []
    absdiff = [[] for _ in BANDS]
    seen_d = {}
    scene_day = {}
    footprints = []            # (day, tile, xmin, xmax, ymin, ymax)
    for i, c in enumerate(crops):
        bands, cl, cf = read_mados_crop(dzip, c)
        H, W = cl.shape
        lab_mask = cl > 0
        labc = cl[lab_mask].astype(np.int64)
        if labc.size and labc.max() > max(MADOS_CLASSES):
            refuse(f"MADOS: {c['cl']} holds a class code > 15")
        for c_ in labc.tolist():
            _bump(cls_d, MADOS_CLASSES[c_])
        got, nv = locate(idx, pmeta, rank_codes(bands[list(CODE_BANDS)]))
        if progress and (i + 1) % 200 == 0:
            progress(f"s2labels MADOS {i + 1}/{len(crops)}")
        if got is None:
            max_unloc = max(max_unloc, nv)
            _bump(unloc_votes,
                  str(max(b for b in UNLOCATED_VOTE_BUCKETS if nv >= b)))
            for c_ in labc.tolist():
                _bump(cls_unloc, MADOS_CLASSES[c_])
            continue
        pi, dr, dc, nv = got
        m = pmeta[pi]
        located_n += 1
        min_vote = nv if min_vote is None else min(min_vote, nv)
        if m["sat"] is not None:
            if m["sat"] != c["sat"]:
                refuse(f"MADOS {c['key']} is {c['sat']} by its band names "
                       f"but located on MARIDA {m['scene']}, which the "
                       f"catalogue says is {m['sat']} — a wrong match")
            sat_checked += 1
        if scene_day.setdefault(c["scene"], m["day"]) != m["day"]:
            refuse(f"MADOS {c['scene']}: located crops point at two dates "
                   f"({scene_day[c['scene']]} and {m['day']})")
        T = m["T"]
        ox, oy = T.c + dc * T.a, T.f + dr * T.e
        footprints.append((m["day"], m["tile"], ox, ox + W * T.a,
                           oy + H * T.e, oy))
        # agreement with the matched patch on the pixels both hold
        rr_a, cc_a = np.mgrid[0:H, 0:W]
        pr, pc = rr_a + dr, cc_a + dc
        inside = (pr >= 0) & (pr < m["H"]) & (pc >= 0) & (pc < m["W"])
        if inside.sum() >= 16:
            pa, _, _ = _read_member(mzip, pats[pi]["name"])
            # MARIDA writes 0.0 in every band where its patch has no data (a
            # tile edge: measured, 65 % of 18-9-20_16PDC_29), and a crop
            # placed there by the grid carries its own granule's values — so
            # the agreement is measured where MARIDA has data
            sel = inside.copy()
            sel[inside] = (pa[:, pr[inside], pc[inside]] != 0).any(axis=0)
            crop_r = 1.0
            for b in range(len(BANDS)):
                if sel.sum() < 16:
                    break
                u = bands[b][sel]
                w = pa[b][pr[sel], pc[sel]]
                if np.std(u) > 0 and np.std(w) > 0:
                    r = float(np.corrcoef(u, w)[0, 1])
                    key = f"rho_{BANDS[b]}"
                    band_r[key] = min(band_r.get(key, 1.0), r)
                    if RES_M[b] != 60:
                        crop_r = min(crop_r, r)
            if crop_r < 0.99:
                low_r += 1
        votes_located.append(nv)
        if not lab_mask.any():
            continue
        rr, cc = np.nonzero(lab_mask)
        lab = cl[rr, cc].astype(np.int64)
        conf = cf[rr, cc].astype(np.int64)
        if set(np.unique(conf).tolist()) - set(CONF_CODES):
            refuse(f"MADOS: {c['conf']} holds "
                   f"{sorted(set(np.unique(conf).tolist()))} on labelled "
                   f"pixels")
        x, y = pixel_centres(T, rr + dr, cc + dc)
        v = np.full((rr.size, C), np.nan)
        v[:, :len(BANDS)] = bands[:, rr, cc].T
        v[:, COL["confidence"]] = conf
        if m["sat"] is not None:
            v[:, COL["s2_unit"]] = SAT_CODE[m["sat"]]
        sp_d = SPLIT_CODE[splits_d[c["key"]]]
        v[:, COL["split"]] = sp_d
        only = []
        for j, k in enumerate(_ground_xy_keys(m["day"], m["tile"], x, y)):
            jc = TO_JOINT["MADOS"][int(lab[j])]
            if k in seen_d:
                dup_d += 1
                dup_d_diff += int(seen_d[k] != jc)
                continue
            seen_d[k] = jc
            row = where.get(k)
            if row is None:
                only.append((j, k, jc))
                continue
            old = rows["qc"][row] & CLASS_BITS
            _bump(overlap, f"{JOINT_CLASSES[old]} -> "
                           f"{MADOS_CLASSES[int(lab[j])]}")
            ov = rows["v"][row]
            conf_changed += int(int(ov[COL["confidence"]]) != int(conf[j]))
            for b in range(len(BANDS)):
                absdiff[b].append(abs(float(ov[b]) - float(v[j, b])))
            nv_ = v[j].copy()
            nv_[COL["split"]] = ov[COL["split"]]       # MARIDA's split kept
            rows["v"][row] = nv_
            rows["qc"][row] = jc | m["q0"] | QC_BOTH | QC_MADOS
        if only:
            js = np.array([o[0] for o in only], np.int64)
            hit = np.zeros(js.size, bool)
            for pj in patches_of[(m["day"], m["tile"])]:
                mj = pmeta[pj]
                pcol = (x[js] - mj["T"].c) / mj["T"].a
                prow = (y[js] - mj["T"].f) / mj["T"].e
                ins = (~hit) & (prow >= 0) & (prow < mj["H"]) & \
                    (pcol >= 0) & (pcol < mj["W"])
                if ins.any() and SPLIT_CODE[mj["split"]] != sp_d:
                    _bump(leak, f"inside a MARIDA {mj['split']} patch, "
                                f"MADOS {splits_d[c['key']]}",
                          int(ins.sum()))
                hit |= ins
            for j, k, jc in only:
                rows["t"].append(m["t"])
                rows["x"].append(float(x[j]))
                rows["y"].append(float(y[j]))
                rows["crs"].append(m["crs"])
                rows["v"].append(v[j])
                rows["plat"].append(f"MADOS|{c['scene']}")
                rows["qc"].append(jc | m["q0"] | QC_MADOS)
                rows["key"].append(k)
            mados_only += len(only)
    # MARIDA pixels inside a located crop that no crop labelled
    kept_inside = {}
    fp_by = {}
    for f in footprints:
        fp_by.setdefault((f[0], f[1]), []).append(f[2:])
    for (day, tile), fps in fp_by.items():
        for pj in patches_of.get((day, tile), []):
            js = pmeta[pj]["rows"]
            if not js.size:
                continue
            xs = np.asarray([rows["x"][j] for j in js])
            ys = np.asarray([rows["y"][j] for j in js])
            ins = np.zeros(js.size, bool)
            for x0, x1, y0, y1 in fps:
                ins |= (xs >= min(x0, x1)) & (xs < max(x0, x1)) & \
                    (ys >= min(y0, y1)) & (ys < max(y0, y1))
            for j in js[ins]:
                if not rows["qc"][j] & QC_BOTH:
                    _bump(kept_inside,
                          JOINT_CLASSES[rows["qc"][j] & CLASS_BITS])
    counts["mados_labelled_pixels"] = sum(cls_d.values())
    counts["mados_by_class"] = cls_d
    counts["mados_crops_located"] = located_n
    counts["mados_crops_unlocated"] = len(crops) - located_n
    counts["mados_scenes_located"] = len(scene_day)
    counts["mados_unlocated_labelled_pixels"] = sum(cls_unloc.values())
    counts["mados_unlocated_by_class"] = cls_unloc
    counts["mados_unlocated_best_vote_at_least"] = unloc_votes
    counts["max_unlocated_best_vote"] = max_unloc
    counts["match_min_votes_located"] = min_vote or 0
    counts["match_satellite_checked"] = sat_checked
    counts["match_min_r_by_band"] = {k: round(v, 6)
                                     for k, v in band_r.items()}
    counts["match_crops_r_10_20m_below_0_99"] = low_r
    counts["match_median_votes_located"] = (
        float(np.median(votes_located)) if votes_located else 0.0)
    counts["overlap_pixels"] = sum(overlap.values())
    counts["overlap_labels"] = overlap
    counts["overlap_label_changed"] = sum(
        n for k, n in overlap.items()
        if k.split(" -> ")[0] != k.split(" -> ")[1])
    counts["overlap_confidence_changed"] = conf_changed
    counts["overlap_median_abs_band_diff"] = {
        f"rho_{b}": round(float(np.median(absdiff[i])), 6)
        for i, b in enumerate(BANDS) if absdiff[i]}
    counts["mados_only_rows"] = mados_only
    counts["mados_duplicate_pixels"] = dup_d
    counts["mados_duplicate_pixels_class_differs"] = dup_d_diff
    counts["marida_unlabelled_by_covering_mados_crop"] = kept_inside
    counts["mados_only_inside_marida_patch_split"] = leak
    return rows


def to_arrays(rows):
    """rows -> (t int64, lat, lon, values (n, C), platform texts, qc uint8)."""
    n = len(rows["t"])
    lat = np.empty(n)
    lon = np.empty(n)
    by = {}
    for j, crs in enumerate(rows["crs"]):
        by.setdefault(str(crs), (crs, []))[1].append(j)
    xs = np.asarray(rows["x"], np.float64)
    ys = np.asarray(rows["y"], np.float64)
    for _k, (crs, js) in by.items():
        js = np.asarray(js, np.int64)
        la, lo = to_wgs84(crs, xs[js], ys[js])
        lat[js], lon[js] = la, lo
    v = np.asarray(rows["v"], np.float64).reshape(n, C)
    return (np.asarray(rows["t"], np.int64), lat, lon, v, rows["plat"],
            np.asarray(rows["qc"], np.uint8))


def tally(t, v, qc, counts):
    """The ledger's breakdowns of what is stored."""
    def day(x):
        return dt.date(1982, 1, 1) + dt.timedelta(seconds=int(x))
    ds, cl, sp, yr, sat = {}, {}, {}, {}, {}
    for j in range(t.size):
        q = int(qc[j])
        d = "both (MADOS label)" if q & QC_BOTH else (
            "MADOS only" if q & QC_MADOS else "MARIDA only")
        ds[d] = ds.get(d, 0) + 1
        n = JOINT_CLASSES[q & CLASS_BITS]
        cl[n] = cl.get(n, 0) + 1
        s = v[j, COL["split"]]
        k = "none" if not np.isfinite(s) else \
            {0: "train", 1: "val", 2: "test"}[int(s)]
        k = ("MARIDA " if (not q & QC_MADOS or q & QC_BOTH) else "MADOS ") + k
        sp[k] = sp.get(k, 0) + 1
        y = str(day(t[j]).year)
        yr[y] = yr.get(y, 0) + 1
        u = v[j, COL["s2_unit"]]
        uk = "unknown" if not np.isfinite(u) else ("S2A" if u == 1 else "S2B")
        sat[uk] = sat.get(uk, 0) + 1
    counts["rows_by_dataset"] = ds
    counts["rows_by_class"] = cl
    counts["rows_by_split"] = sp
    counts["rows_by_year"] = yr
    counts["rows_by_satellite"] = sat
    counts["channel_rows_filled"] = {n: int(np.isfinite(v[:, i]).sum())
                                     for i, n in enumerate(NAMES)}


# ================================================== the falsifier's base ==
def fdi(v, sat_code):
    """FDI = R8 - (R6 + (R11 - R6) * coef), coef per satellite (plan §4).
    `v` (n, >= 11) bands in BANDS order; NaN where the satellite is unknown."""
    coef = np.where(sat_code == SAT_CODE["S2A"], FDI_COEF["S2A"],
                    np.where(sat_code == SAT_CODE["S2B"], FDI_COEF["S2B"],
                             np.nan))
    r6, r8, r11 = v[:, 5], v[:, 7], v[:, 9]
    return r8 - (r6 + (r11 - r6) * coef)


def fdi_baseline(v, qc, train_mask, test_mask, debris=1, water=7, pct=95.0):
    """The plan's §6 baseline: threshold = the `pct` percentile of FDI over
    TRAIN pixels of clear marine water; 'debris' = FDI > threshold; scored
    debris-vs-not on TEST. A pixel whose FDI is NaN is 'not debris'."""
    cls = qc.astype(np.int64) & CLASS_BITS
    f = fdi(v, v[:, COL["s2_unit"]])
    tw = train_mask & (cls == water) & np.isfinite(f)
    thr = float(np.percentile(f[tw], pct))
    pred = np.isfinite(f) & (f > thr)
    truth = cls == debris
    te = test_mask
    tp = int((pred & truth & te).sum())
    fp = int((pred & ~truth & te).sum())
    fn = int((~pred & truth & te).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return {"threshold_fdi": thr, "train_water_pixels": int(tw.sum()),
            "test_pixels": int(te.sum()),
            "test_debris_pixels": int((truth & te).sum()),
            "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1, "fdi_nan_in_test": int((~np.isfinite(f) & te).sum())}


# ================================================================ adapter ==
def _mirror(url):
    u = urllib.parse.urlsplit(url)
    path = u.path.strip("/") or "index"
    name = u.netloc + "/" + path
    if u.query:
        name += "__" + hashlib.sha1(u.query.encode()).hexdigest()[:20]
    return name + ".body"


def zip_name(ds):
    """Where an archive lives under the scratch (and in a smoke mirror)."""
    rec, key = RECORDS[ds]
    return f"zenodo.org/records/{rec}/{key}"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class S2LabelsAdapter(f10b.SourceAdapter):
    store = "s2labels"
    title = ("Human-labelled Sentinel-2 pixels of marine debris, Sargassum, "
             "foam, ships, oil and water types (MARIDA, MADOS), one row per "
             "labelled 10 m pixel")
    family = "13"
    distribution = "public"
    licence = {
        "name": "CC BY 4.0 (MARIDA; MADOS)",
        "redistribution": "attribution", "derived_works": "free",
        "attribution": (
            "MARIDA: Kikaki K, Kakogeorgiou I, Mikeli P, Raitsos DE, "
            "Karantzalos K (2022) MARIDA: A benchmark for Marine Debris "
            "detection from Sentinel-2 remote sensing data. PLoS ONE 17(1): "
            "e0262247, data doi:10.5281/zenodo.5151941; MADOS: Kikaki K, "
            "Kakogeorgiou I, Hoteit I, Karantzalos K (2024) Detecting Marine "
            "Pollutants and Sea Surface Features with Deep Learning in "
            "Sentinel-2 Imagery, ISPRS J. Photogramm. Remote Sens., data "
            "doi:10.5281/zenodo.10664073 (both CC BY 4.0). Contains modified "
            "Copernicus Sentinel data [2015-2021]; acquisition times from the "
            "Copernicus Data Space Ecosystem catalogue")}
    time_dtype = "int32"
    platform_meta = False
    credentials = ()
    channels = CHANNELS
    log2_fp = float(np.log2(0.010 / 27.83))              # -11.44, 10 m pixel
    log2_dt = float(np.log2((1.0 / 86400.0) / 5.0))      # -18.72, cat_s2's
    per_year = False
    first_year = 2015
    qc_policy = (
        "qc bits 0-4 = joint class (" + "; ".join(
            f"{k} {v}" for k, v in JOINT_CLASSES.items()) +
        "); bit 5 (32): time of day not from the Copernicus catalogue, "
        "placed at 12:00 UTC; bit 6 (64): labelled in both MARIDA and MADOS "
        "(MADOS's label, MARIDA's split); bit 7 (128): the row's values are "
        "MADOS's. Reflectance outside [-0.02, 1.7] -> NaN, counted.")
    sources = (ZENODO_RECORD.format(rec=RECORDS["MARIDA"][0]),
               ZENODO_RECORD.format(rec=RECORDS["MADOS"][0]),
               ASSETS["MARIDA"][0], ASSETS["MADOS"][0], CDSE_ODATA)
    verified = (
        "2026-10-11 from the sandbox: both Zenodo records (licence, class "
        "tables, md5 of both zips), every member of both archives, the "
        "producers' assets.py (band order, class names) and stack_patches.py "
        "(nearest stacking), MADOS's missing georeferencing (TIFF tags), the "
        "MARIDA-MADOS location by neighbour-rank codes (1,488 crops, >= 54 "
        "votes each, satellite agreeing on every one) and the CDSE catalogue "
        "for all 63 MARIDA scenes")
    notes = (
        "MADOS crops carry no georeferencing and no date; only the 1,488 "
        "crops that coincide with MARIDA patches can be placed, so the other "
        "111 MADOS scenes are counted and left out (mados_unlocated_*). "
        "Plastic Litter Project and FloatingObjects are out of this build "
        "(the adapter's docstring says why).")
    smoke_window = ("2015-01-01", "2022-12-31")
    smoke_probe_month = "2018-03"
    fetch_month_scope = "archive"

    # ------------------------------------------------------------- reading --
    def _path(self, ctx, url, want=None, name=None):
        # an archive keeps its own file name: GDAL's /vsizip/ finds the
        # archive by its `.zip` suffix
        name = name or _mirror(url)
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "s2labels", name)
            if not os.path.exists(p):
                refuse(f"{url} is not in the source mirror ({p})")
            seen = getattr(ctx, "_s2labels_counted", None)
            if seen is None:
                seen = ctx._s2labels_counted = set()
            if p not in seen:
                ctx.count_bytes(os.path.getsize(p))
                seen.add(p)
        else:
            p = os.path.join(ctx.scratch, "s2labels", name)
            if not (os.path.exists(p) and os.path.getsize(p) > 0 and (
                    want is None or os.path.getsize(p) == want["size"])):
                got = f10b.fetch_first([url], path=p, attempts=int(
                    getattr(ctx.a, "attempts", 3) or 3))
                if got is None:
                    refuse(f"{url} answered 404 — every source of this store "
                           f"is needed whole")
        if os.path.getsize(p) == 0:
            refuse(f"{url} answered an EMPTY body")
        if want is not None:
            if os.path.getsize(p) != want["size"]:
                refuse(f"{url}: {os.path.getsize(p)} bytes, the record says "
                       f"{want['size']}")
            got = md5_of(p)
            if got != want["md5"]:
                refuse(f"{url}: md5 {got}, the record says {want['md5']}")
        return p

    def _json(self, ctx, url):
        with open(self._path(ctx, url), "rb") as fh:
            return json.loads(fh.read())

    def _text(self, ctx, url):
        with open(self._path(ctx, url), "rb") as fh:
            return fh.read().decode("utf-8")

    def records(self, ctx):
        out = {}
        for ds in DATASETS:
            rec = self._json(ctx, ZENODO_RECORD.format(rec=RECORDS[ds][0]))
            check_record(ds, rec)
            check_assets(ds, self._text(ctx, ASSETS[ds][0]))
            out[ds] = zip_file_of(rec, RECORDS[ds][1])
        return out

    def catalogue(self, ctx, z, counts):
        """{(date, tile): (satellite, t, n products)} for every MARIDA scene."""
        out = {}
        n_ok = 0
        for n in sorted(z.namelist()):
            m = MARIDA_SCENE.match(n)
            if not m:
                continue
            d, mo, y, tile = m.groups()
            day = dt.date(2000 + int(y), int(mo), int(d))
            doc = self._json(ctx, cdse_url(tile, day))
            sat, t, k = cdse_answer(doc)
            if t is not None:
                tday = dt.date(1982, 1, 1) + dt.timedelta(seconds=t)
                if tday != day:
                    refuse(f"the catalogue's {tile} {day} product senses on "
                           f"{tday}")
                n_ok += 1
            out[(f"{d}-{mo}-{y}", tile)] = (sat, t, k)
            counts.setdefault("catalogue_products_per_scene", {})
            kk = str(k)
            counts["catalogue_products_per_scene"][kk] = \
                counts["catalogue_products_per_scene"].get(kk, 0) + 1
        counts["catalogue_scenes_timed"] = n_ok
        counts["catalogue_scenes_untimed"] = len(out) - n_ok
        return out

    def build(self, ctx):
        counts = {}
        files = self.records(ctx)
        paths = {}
        for ds in DATASETS:
            paths[ds] = self._path(ctx, files[ds]["url"], want=files[ds],
                                   name=zip_name(ds))
            counts[f"{ds.lower()}_zip_bytes"] = files[ds]["size"]
        zm = zipfile.ZipFile(paths["MARIDA"])
        zd = zipfile.ZipFile(paths["MADOS"])
        cat = self.catalogue(ctx, zm, counts)
        sm = read_splits(zm, "", counts, "MARIDA")
        sd = read_splits(zd, "MADOS/", counts, "MADOS")
        prog = getattr(getattr(ctx, "prog", None), "item", None)
        rows = build_rows(paths["MARIDA"], paths["MADOS"], cat, sm, sd,
                          counts, progress=prog)
        check_measured(files, counts)
        t, lat, lon, v, pl, qc = to_arrays(rows)
        return t, lat, lon, v, pl, qc, counts

    def _rows(self, ctx, t_lo, t_hi):
        t, lat, lon, v, pl, qc, counts = self.build(ctx)
        keep = (t >= t_lo) & (t <= t_hi)
        counts["rows_outside_window"] = int((~keep).sum())
        idx = np.flatnonzero(keep)
        t, lat, lon, v, qc = t[idx], lat[idx], lon[idx], v[idx], qc[idx]
        pl = [pl[i] for i in idx]
        oob = self.mask_bounds(v)
        if oob:
            counts["out_of_bounds"] = oob
        tally(t, v, qc, counts)
        counts["rows_kept"] = int(t.size)
        return t, lat, lon, v, pl, qc, counts

    def _yield(self, ctx, t_lo, t_hi, label=None):
        t, la, lo, v, pl, qc, counts = self._rows(ctx, t_lo, t_hi)
        years = np.array([(dt.date(1982, 1, 1) +
                           dt.timedelta(seconds=int(x))).year for x in t],
                         np.int64)
        for y in sorted(set(years.tolist())):
            k = np.flatnonzero(years == y)
            yield (label or y), self.pack(
                t[k], la[k], lo[k], v[k],
                np.array([f10b.platform_hash(pl[i]) for i in k], np.int64),
                qc[k]), None
        yield (None if label is None else label), None, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        files = self.records(ctx)
        return {"dataset": "s2labels: MARIDA (Zenodo 5151941) + MADOS "
                           "(Zenodo 10664073)",
                "zips": {ds: {"size": f["size"], "md5": f["md5"]}
                         for ds, f in files.items()},
                "joint_classes": {str(k): v for k, v in
                                  JOINT_CLASSES.items()}}

    def fetch_stream(self, ctx):
        yield from self._yield(ctx, ctx.t_lo, ctx.t_hi)

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        yield from self._yield(ctx, max(lo, ctx.t_lo), min(hi, ctx.t_hi),
                               label=f"{year}-{month:02d}")

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi):
        from family1.adapters import _s2labels_smoke as sm
        return sm.make_smoke_sources(root, d_lo, d_hi)


ADAPTER = S2LabelsAdapter


# ================================================================ ledger ==
def ledger(store_dir, store_meta=None):
    """The ledger the plan asks for: store.json's counts (which carry the
    baseline computed from MARIDA's own files, float32 bands and MARIDA's
    labels) and the same baseline on the BUILT store — bands as stored
    (float16), labels as stored (MADOS's where both label a pixel)."""
    import family10_store as f10
    st = f10.Store(store_dir)
    v = np.asarray(st["values"], np.float32).astype(np.float64)
    qc = np.asarray(st["qc"], np.uint8)
    meta = store_meta or json.load(open(os.path.join(store_dir,
                                                     "store.json")))
    in_marida = ((qc & QC_MADOS) == 0) | ((qc & QC_BOTH) != 0)
    sp = v[:, COL["split"]]
    base = fdi_baseline(v, qc, in_marida & (sp == 0), in_marida & (sp == 2))
    out = {"rows": int(st.N), "counts": meta.get("counts", {}),
           "fdi_baseline_store": base}
    return out


if __name__ == "__main__":                                   # pragma: no cover
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="s2labels ledger + FDI baseline")
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(ledger(a.store), open(a.out, "w"), indent=1, sort_keys=True)
    sys.exit(0)
