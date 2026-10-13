"""The synthetic archive of family 1.3's `s2fdi_coast` (E-092 D13, the pilot).

Written in the source's REAL shapes, so the smoke exercises the adapter's own
request building, paging, selection, masking and the family's sharded writer:

  <src>/s2fdi_coast/<canned_key>.json   Earth Search STAC answers, one per
                                        (collection, tile, year) search the
                                        adapter makes (`_stac.canned_key`)
  <src>/s2fdi_coast/MARIDA.zip          a zip whose central directory names
                                        MARIDA's 63 scene folders
  <src>/s2fdi_coast/l2r/<l1c id>.npz    the ACOLITE STUB: rho_s for B04, B06,
                                        B08, B11 on the tile grid + its xrange
  <src>/s2fdi_coast/scl/<l2a id>.npy    the L2A scene classification

Three tiles on a 300 x 300 grid (`S2FDI_SMOKE_PX`), 2018-08-27 .. 09-25, each
case a named expectation in tests/test_family13_s2fdi_coast.py:

  T16PCC  MARIDA 2018-08-30 with FOUR copies (02.06 x2, 05.00 x2: the
          05.00 sequence-3 copy is kept, its cloud cover decides nothing on
          a MARIDA tile); MARIDA 2018-09-14 (two copies); MARIDA 2018-09-19
          whose datatake has NO L2A twin (-> absent_upstream); 2018-09-04 a
          clear non-MARIDA date (-> excluded, no frame)
  T16PDC  MARIDA 2018-09-19; 2018-09-09 a non-MARIDA date
  T35SMD  2018-08-28 cc 5 (S2A); 2018-09-02 cc 45 (excluded); 2018-09-07
          cc 10 (S2B) with a cloud, its shadow, out-of-bounds bands and a
          near-zero B04 + B08 (NDVI and PI out of bounds, FDI kept);
          2018-09-12 cc 3 whose SCL holds NO water (a present, empty frame);
          2018-09-17 cc 19.9 (just inside)

The TRUTH is computed here from the plan's formulas (§4) and an independent
distance transform, never by calling the adapter.
"""
import datetime as dt
import json
import os
import zipfile

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _stac as st
from family1.adapters import s2fdi_coast as s2f

PXN = 300
TILES = ("16PCC", "16PDC", "35SMD")
SAT = {"A": "sentinel-2a", "B": "sentinel-2b"}

# (tile, day, sat, [(seq, baseline, cc)], l2a [(seq, baseline)])
CASES = (
    ("16PCC", "2018-08-30", "B", [(0, "02.06", 30.0), (1, "02.06", 31.0),
                                  (2, "05.00", 12.0), (3, "05.00", 11.0)],
     [(0, "00.01"), (2, "05.00"), (3, "05.00")]),
    ("16PCC", "2018-09-04", "A", [(0, "02.06", 1.0)], [(0, "00.01")]),
    ("16PCC", "2018-09-14", "A", [(1, "05.00", 40.0), (0, "02.06", 41.0)],
     [(1, "05.00"), (0, "00.01")]),
    ("16PCC", "2018-09-19", "B", [(0, "02.06", 3.0)], []),
    ("16PDC", "2018-09-09", "A", [(0, "02.06", 2.0)], [(0, "00.01")]),
    ("16PDC", "2018-09-19", "B", [(0, "02.06", 7.0)], [(0, "00.01")]),
    ("35SMD", "2018-08-28", "A", [(0, "02.06", 5.0)], [(0, "02.09")]),
    ("35SMD", "2018-09-02", "A", [(0, "02.06", 45.0)], [(0, "02.09")]),
    ("35SMD", "2018-09-07", "B", [(0, "02.06", 10.0)], [(0, "02.09")]),
    ("35SMD", "2018-09-12", "B", [(0, "02.06", 3.0)], [(0, "02.09")]),
    ("35SMD", "2018-09-17", "A", [(0, "02.06", 19.9)], [(0, "02.09")]),
)
# what the store must hold: (tile, day) -> (kept L1C id, kept L2A id)
KEPT = {
    ("16PCC", "2018-08-30"): ("S2B_16PCC_20180830_3_L1C",
                              "S2B_16PCC_20180830_3_L2A"),
    ("16PCC", "2018-09-14"): ("S2A_16PCC_20180914_1_L1C",
                              "S2A_16PCC_20180914_1_L2A"),
    ("16PDC", "2018-09-19"): ("S2B_16PDC_20180919_0_L1C",
                              "S2B_16PDC_20180919_0_L2A"),
    ("35SMD", "2018-08-28"): ("S2A_35SMD_20180828_0_L1C",
                              "S2A_35SMD_20180828_0_L2A"),
    ("35SMD", "2018-09-07"): ("S2B_35SMD_20180907_0_L1C",
                              "S2B_35SMD_20180907_0_L2A"),
    ("35SMD", "2018-09-12"): ("S2B_35SMD_20180912_0_L1C",
                              "S2B_35SMD_20180912_0_L2A"),
    ("35SMD", "2018-09-17"): ("S2A_35SMD_20180917_0_L1C",
                              "S2A_35SMD_20180917_0_L2A"),
}
ABSENT = {("16PCC", "2018-09-19")}
EXCLUDED = 3 + 1 + 1 + 1 + 1      # superseded 3+1, non-MARIDA 2, cloud 1
NO_WATER = ("35SMD", "2018-09-12")
OOB_CASE = ("35SMD", "2018-09-07")


def item_id(sat, tile, day, seq, kind):
    return f"S2{sat}_{tile}_{day.replace('-', '')}_{seq}_{kind}"


def feature(sat, tile, day, seq, base, cc, kind):
    d = day.replace("-", "")
    dtk = f"GS2{sat}_{d}T100000_000000_N{base}"
    iid = item_id(sat, tile, day, seq, kind)
    utm = tile[:2].lstrip("0")
    path = f"tiles/{utm}/{tile[2]}/{tile[3:]}/{int(d[:4])}/{int(d[4:6])}/" \
           f"{int(d[6:])}/{seq}/"
    pr = {"datetime": f"{day}T10:00:00.000000Z", "platform": SAT[sat],
          "s2:datatake_id": dtk, "s2:processing_baseline": base,
          "grid:code": f"MGRS-{tile}",
          "s2:product_uri": f"S2{sat}_MSI{kind}_{d}T100000_N{base.replace('.', '')}"
                            f"_R001_T{tile}_{d}T120000.SAFE"}
    if cc is not None:
        pr["eo:cloud_cover"] = cc
    g = s2f.grid_of(tile, PXN)
    if kind == "L1C":
        assets = {"nir": {"href": f"s3://sentinel-s2-l1c/{path}B08.jp2"}}
    else:
        assets = {"scl": {"href": f"s3://sentinel-cogs/{iid}/SCL.tif",
                          "proj:transform": [20, 0, g["x0"], 0, -20,
                                             g["y0"]]}}
    return {"type": "Feature", "id": iid, "properties": pr, "assets": assets}


def scl_for(tile, day, rng):
    """Water on the west two thirds, land east; a cloud + shadow for 35SMD
    2018-09-07; all land for the NO_WATER case."""
    s = np.full((PXN, PXN), 5, np.uint8)
    s[:, :200] = 6
    s[:, 260:] = 4
    s[0:3, :] = 0                                    # no-data row band
    if (tile, day) == NO_WATER:
        s[:, :] = 4
    if (tile, day) == OOB_CASE:
        s[100:120, 40:60] = 9
        s[125:135, 45:60] = 3
        s[200:205, 150:153] = 10
    return s


def bands_for(tile, day, rng):
    H = W = PXN
    R = {"B04": rng.uniform(0.008, 0.03, (H, W)),
         "B06": rng.uniform(0.003, 0.012, (H, W)),
         "B08": rng.uniform(0.002, 0.010, (H, W)),
         "B11": rng.uniform(0.0005, 0.004, (H, W))}
    R = {k: v.astype(np.float32) for k, v in R.items()}
    # a bright "debris" patch in the water
    R["B08"][250:254, 20:24] += np.float32(0.05)
    if (tile, day) == OOB_CASE:
        R["B08"][10, 10] = np.float32(2.0)           # > RHO_HI
        R["B04"][11, 11] = np.float32(-0.05)         # < RHO_LO
        R["B04"][12, 12] = np.float32(-0.0100)       # B04 + B08 ~ 0
        R["B08"][12, 12] = np.float32(0.0101)
        R["B06"][13, 13] = np.float32(np.nan)        # ACOLITE NaN
    return R


def truth_frame(R, scl, plat):
    """The plan's indices and the mask, computed independently."""
    from scipy import ndimage
    lo, hi = s2f.RHO_LO, s2f.RHO_HI
    B = {}
    for k, v in R.items():
        v = v.astype(np.float32).copy()
        v[(v < lo) | (v > hi)] = np.nan
        B[k] = v
    cloud = np.isin(scl, (3, 8, 9, 10))
    if cloud.any():
        dist = ndimage.distance_transform_edt(~cloud) * 20.0
    else:
        dist = np.full(scl.shape, np.inf)
    keep = (scl == 6) & (dist > 1000.0)
    for v in B.values():
        keep &= np.isfinite(v)
    w = {"sentinel-2a": (664.6, 740.5, 832.8, 1613.7),
         "sentinel-2b": (665.0, 739.1, 833.0, 1610.4)}[plat]
    lin = np.float32((w[2] - w[0]) / (w[3] - w[0]))
    fac = np.float32({"sentinel-2a": 1.7722, "sentinel-2b": 1.7770}[plat])
    with np.errstate(all="ignore"):
        fdi = B["B08"] - (B["B06"] + (B["B11"] - B["B06"]) * fac)
        fai = B["B08"] - (B["B04"] + (B["B11"] - B["B04"]) * lin)
        ndvi = (B["B08"] - B["B04"]) / (B["B08"] + B["B04"])
        pi = B["B08"] / (B["B08"] + B["B04"])
    ch = [fdi, fai, ndvi, pi, B["B04"], B["B06"], B["B08"], B["B11"]]
    out = np.full((PXN, PXN, 8), np.nan, np.float32)
    for i, a in enumerate(ch):
        out[..., i][keep] = a[keep]
    for i, (_, _, lo_, hi_) in enumerate(s2f.CHANNELS):
        v = out[..., i]
        with np.errstate(invalid="ignore"):
            v[np.isfinite(v) & ((v < lo_) | (v > hi_))] = np.nan
    return out.astype(np.float16)


def make_smoke_sources(root, d_lo, d_hi, seed=20261013):
    os.environ["S2FDI_SMOKE_PX"] = str(PXN)
    os.environ["S2FDI_TILES"] = ",".join(TILES)
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "s2fdi_coast")
    for sub in ("l2r", "scl"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    # MARIDA.zip with the 63 scene folders
    with zipfile.ZipFile(os.path.join(base, "MARIDA.zip"), "w") as z:
        for tile, day in s2f.MARIDA_SCENES:
            d = dt.date.fromisoformat(day)
            z.writestr(f"patches/S2_{d.day}-{d.month}-{d.year % 100}_{tile}/",
                       b"")
        z.writestr("splits/test_X.txt", b"")
    # STAC: one canned answer per (collection, tile, year)
    feats = {}
    for tile, day, sat, l1, l2 in CASES:
        for seq, b, cc in l1:
            feats.setdefault(("sentinel-2-l1c", tile), []).append(
                feature(sat, tile, day, seq, b, cc, "L1C"))
        for seq, b in l2:
            feats.setdefault(("sentinel-2-l2a", tile), []).append(
                feature(sat, tile, day, seq, b, None, "L2A"))
    years = sorted({d_lo.year, d_hi.year})
    for coll in ("sentinel-2-l1c", "sentinel-2-l2a"):
        for tile in TILES:
            for y in years:
                fs = [f for f in feats.get((coll, tile), [])
                      if f["properties"]["datetime"].startswith(str(y))]
                body = dict(s2f.S2FdiCoastAdapter.stac_body(coll, tile, y))
                body["limit"] = s2f.STAC_LIMIT
                k = st.canned_key("POST", s2f.STAC_SEARCH, body)
                with open(os.path.join(base, k + ".json"), "w") as fh:
                    json.dump({"type": "FeatureCollection", "features": fs,
                               "numberMatched": len(fs), "links": []}, fh)
    # per kept acquisition: the ACOLITE stub and the SCL
    frames = {}
    for (tile, day), (l1, l2) in sorted(KEPT.items()):
        sat = SAT[l1[2]]
        R = bands_for(tile, day, rng)
        scl = scl_for(tile, day, rng)
        g = s2f.grid_of(tile, PXN)
        np.savez(os.path.join(base, "l2r", l1 + ".npz"),
                 xrange=np.array([g["x0"], g["x0"] + PXN * 20.0]),
                 aot=np.float64(0.05), **R)
        np.save(os.path.join(base, "scl", l2 + ".npy"), scl)
        frames[(tile, day)] = truth_frame(R, scl, sat)
    # the truth: every frame of every bin overlapping the window
    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    truth = {}
    for tile in TILES:
        g = s2f.group_name(tile)
        for b in sh.bins_overlapping(t_lo, t_hi):
            for f in range(5):
                day = str(sh.frame_day(b, f, 86400))
                if (tile, day) in frames:
                    truth[(g, b, f)] = (frames[(tile, day)], None)
                elif (tile, day) in ABSENT:
                    truth[(g, b, f)] = (None, sh.FRAME_ABSENT_UPSTREAM)
                else:
                    truth[(g, b, f)] = (None, sh.FRAME_NO_FRAME_IN_BIN)
    return truth
