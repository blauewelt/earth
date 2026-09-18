"""MOD10C1 v61 — MODIS/Terra daily snow cover on the 0.05-degree Climate
Modeling Grid (family 1.0.tf, E-082 wave 4).

PLAIN ENGLISH. Snow is the land surface's fastest-changing property and the
one that changes its reflectivity most: a field that is bare on Monday and
white on Tuesday reflects three times as much sunlight back to space. MODIS
on Terra reports, for every 5.6 km cell and every day since February 2000,
what PERCENTAGE of the cell's land was snow-covered and what percentage was
hidden by cloud — the second number being what says how much to believe the
first.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (CMR is keyless; the granules
are behind Earthdata Login and were read on a hosted runner):
  CMR collection C3044686669-NSIDC_CPRD, short_name MOD10C1, version 61,
  9,634 granules, 2000-02-24 -> 2026-09-16 (the whole listing came back in
  12.7 s), about 4 MB each, one HDF-EOS2 file per calendar day, served from
  `https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/
  MODIS/MOD10C1/61/<yyyy>/<mm>/<dd>/<granule>.hdf`. The granule id carries
  the production timestamp, so the URL comes from CMR and is never guessed
  (contract rule 4). `_modis_cmg.py` holds the listing, the login, the
  download and the HDF4 reader.

WHAT IS STORED: C = 2, dtype UINT8 (255 = not measured), one group.
  snow_cover  Day_CMG_Snow_Cover      percent of the cell's land under snow
  cloud       Day_CMG_Cloud_Obscured  percent of the cell's land under cloud

THE CODES ARE KEPT, NOT COLLAPSED. Both SDS are uint8 whose valid_range is
0..100 percent, and whose remaining values are CLASSES, listed in the
producer's own `Key` attribute (file specification
ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD10C1, read 2026-09-18):

  0-100  percent (of snow, or of cloud, over the cell's land)
  107    lake ice          111  night (no daylight to observe by)
  237    inland water      239  ocean
  250    cloud-obscured water
  252    Antarctica mask (the cloud SDS only; the snow SDS maps Antarctica as
         snow deliberately, so its snow_cover reads 100 there)
  253    data not mapped   255  fill

Every one of those is a statement about the world and none of them is a
percentage, so the store keeps the producer's number and the channel bounds
are 0..253 — wide enough to carry the classes, narrow enough that 254 (which
the key does not define) and 255 (fill) are not stored as data. 255 is the
layout's own missing value, so fill arrives as NaN through the reader.
Nothing is homogenised: a consumer that wants "percent snow or nothing" masks
everything above 100 itself, and one that wants to know WHY a cell has no
percentage can (family1tf.tex §5: "the source's own quality flag is kept and
nothing is homogenised").

WHY UINT8 AND NOT FLOAT16. The values are integers 0..253 and uint8 stores
them exactly in one byte instead of two, which is the whole reason the note
expects this store to be ~60 GB where lst05 is ~350 GB ("classes compress far
below the float formula"). `sharded.py`'s uint8 groups reserve 255 for
missing and store 0..254.

THE GRID: 7,200 x 3,600 at 0.05 degrees, EPSG:4326, row 0 the northernmost
(90 N) — the source SDS' own orientation. F = 5 daily frames per five-day
bin, so a frame IS a calendar day.

THE CONTINUATION GROUPS. `family1tf.tex` §4.3 names VNP10C1 (Suomi-NPP) and
VJ110C1 (NOAA-20) as the VIIRS continuations; `SNOW05_GROUPS` selects them
once their collections are checked, and the default is Terra alone.

SIZE. The note's estimate is about 60 GB for Terra. The probe measures the
real bytes per frame; `ml/family1/probes/snow05_2015-02.json` is the number
that replaces the estimate.

`SNOW05_SMOKE_GRID` ("W,H") shrinks the grid for `--smoke`.
"""
import datetime as dt
import os

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _modis_cmg as mc

SDS_SNOW = "Day_CMG_Snow_Cover"
SDS_CLOUD = "Day_CMG_Cloud_Obscured"
FILL = 255
# the producer's own key, kept for `tile_grid.json` and store.json
CODES = {"0-100": "percent of the cell's land (snow, or cloud)",
         "107": "lake ice", "111": "night", "237": "inland water",
         "239": "ocean", "250": "cloud-obscured water",
         "252": "Antarctica mask (cloud SDS)", "253": "data not mapped",
         "255": "fill (stored as missing)"}
NSIDC_PREFIX = ("https://data.nsidc.earthdatacloud.nasa.gov/"
                "nsidc-cumulus-prod-protected/MODIS/MOD10C1/61/")


class Snow05Adapter(mc.CmgAdapter):
    store = "snow05"
    title = ("Daily snow-cover and cloud percent, MODIS MOD10C1 v61, "
             "0.05 degrees")
    short_name = "MOD10C1"
    version = "61"
    groups_available = {"terra": ("MOD10C1", "61")}
    groups_default = ("terra",)
    groups_env = "SNOW05_GROUPS"
    smoke_grid_env = "SNOW05_SMOKE_GRID"

    licence = {
        "name": "NASA open data (no restrictions)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Hall, D. K., G. A. Riggs (2021). MODIS/Terra Snow "
                        "Cover Daily L3 Global 0.05Deg CMG, Version 61. "
                        "NASA National Snow and Ice Data Center Distributed "
                        "Active Archive Center. "
                        "doi:10.5067/MODIS/MOD10C1.061"),
        "terms": ("nsidc.org/about/data-use-and-copyright and "
                  "earthdata.nasa.gov's data-information policy, read "
                  "2026-09-18: NASA's Earth science data are open, full and "
                  "without restriction and free of charge; users are asked to "
                  "cite the data set"),
    }
    channels = (("snow_cover", "percent_or_code", 0.0, 253.0),
                ("cloud", "percent_or_code", 0.0, 253.0))
    dtype = "uint8"
    first_year = 2000
    sds = (SDS_SNOW, SDS_CLOUD)
    sds_want = {SDS_SNOW: {"_FillValue": FILL},
                SDS_CLOUD: {"_FillValue": FILL}}
    note_estimate = {
        "bytes": 60e9,
        "what": ("family1tf.tex §4.3 ledger: '~60 GB (Terra; classes compress "
                 "far below the float formula)' for MOD10C1 v61")}
    qc_policy = (
        "MOD10C1 publishes its quality INSIDE the two channels rather than "
        "beside them: 0..100 is a percentage and every higher value is one of "
        "the producer's own classes — 107 lake ice, 111 night, 237 inland "
        "water, 239 ocean, 250 cloud-obscured water, 252 the Antarctica mask, "
        "253 data not mapped, 255 fill. The store keeps those numbers "
        "unchanged (family1tf.tex §5: nothing is homogenised); the channel "
        "bounds are 0..253, so 255 arrives as the layout's missing value and "
        "254, which the producer's key does not define, would be counted "
        "`out_of_bounds` rather than stored. The cloud channel is what says "
        "how much to believe the snow channel; the product's own third SDS "
        "(Day_CMG_Clear_Index) says the same thing the other way round and is "
        "not carried, because the note's ledger row is C = 2. Antarctica is "
        "deliberately mapped as snow by the producer, and its cloud value is "
        "the 252 mask")
    sources = (NSIDC_PREFIX + "<yyyy>/<mm>/<dd>/<granule>.hdf",
               mc.CMR + "?short_name=MOD10C1&version=61")
    verified = (
        "2026-09-18 from the sandbox, keyless: the CMR collection search "
        "(C3044686669-NSIDC_CPRD, MOD10C1 v61, cloud_hosted, time_start "
        "2000-02-24), the granule search for 2015-02-01 (4.70 MB, one https "
        "'nsidc-cumulus-prod-protected' .hdf data link), and the WHOLE "
        "collection paged with CMR-Search-After: 9,634 granules, "
        "MOD10C1.A2000055 .. MOD10C1.A2026259, 39 MB of JSON in 12.7 s. The "
        "SDS names, the uint8 fill 255 and the class key come from the "
        "producer's file specification "
        "(ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD10C1). The "
        "granules are behind Earthdata Login and were opened on a "
        "GitHub-hosted runner by the probe, which re-checks the shape and the "
        "fill value in every file it reads")
    notes = (
        "MOD10C1 v61, one HDF-EOS2 granule a day, stored as uint8 so the "
        "producer's integers survive exactly in one byte. Both channels carry "
        "percentages 0..100 AND the producer's class codes above 100 (107 "
        "lake ice, 111 night, 237 inland water, 239 ocean, 250 cloud-obscured "
        "water, 252 Antarctica mask, 253 not mapped); 255 is fill and becomes "
        "the layout's missing value. Antarctica is deliberately mapped as "
        "snow by the producer. Row 0 is the northernmost row, 90 N, as in the "
        "source SDS.")
    smoke_window = ("2015-02-01", "2015-02-20")
    smoke_probe_month = "2015-02"
    WORKERS = 4

    def specs(self):
        out = super().specs()
        for g in out:
            out[g]["value_key"] = dict(CODES)
            out[g]["value_key_note"] = (
                "the producer's own key for both channels: 0..100 is a "
                "percentage of the cell's LAND, every higher value is a "
                "class. Nothing is collapsed — a consumer that wants only "
                "percentages masks above 100 itself")
        return out

    # ---------------------------------------------------------- the frame --
    def frame_from(self, arrs, attrs):
        h, w = self.h, self.w
        out = np.empty((h, w, 2), np.float32)
        counts = {}
        for i, (name, ch) in enumerate(((SDS_SNOW, "snow_cover"),
                                        (SDS_CLOUD, "cloud"))):
            raw = arrs[name]
            if raw.dtype != np.uint8:
                raise mc.FormatError(f"{name}: {raw.dtype}, expected uint8")
            v = raw.astype(np.float32)
            fill = raw == FILL
            v[fill] = np.nan
            out[:, :, i] = v
            counts[f"fill_pixels_{ch}"] = int(fill.sum())
            # what the codes actually are, per frame, so the store records
            # the producer's own mix rather than an assumption about it
            for code in (107, 111, 237, 239, 250, 252, 253):
                n = int((raw == code).sum())
                if n:
                    counts.setdefault(f"code_{ch}", {})[str(code)] = n
            counts.setdefault(f"percent_pixels_{ch}", 0)
            counts[f"percent_pixels_{ch}"] += int((raw <= 100).sum())
        return out, counts

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.smoke_reinit()
        return truth


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 600, 300
SMOKE_ABSENT = ()                          # see lst05: a listed-and-missing
SMOKE_SKIP = (dt.date(2015, 2, 11),)       # granule is a REFUSAL, not a gap
SMOKE_OOB = dt.date(2015, 2, 4)            # one pixel at 254, undefined


def smoke_fields(d, w, h):
    """A deterministic granule: a snowy north, ocean south, codes in place."""
    k = (d - dt.date(2015, 1, 1)).days % 7
    yy = np.arange(h, dtype=np.float64)[:, None] / h            # 0 = north
    xx = np.arange(w, dtype=np.float64)[None, :] / w
    snow = np.full((h, w), 239, np.uint8)                       # ocean
    cloud = np.full((h, w), 239, np.uint8)
    land = (yy > 0.12) & (yy < 0.45) & (xx > 0.2) & (xx < 0.8)
    pct = np.clip(50 + 50 * np.sin(8 * xx + k) * np.cos(6 * yy - k), 0, 100)
    snow[land] = np.rint(pct[land]).astype(np.uint8)
    cloud[land] = np.rint(100 - pct[land]).astype(np.uint8)
    rows = yy[:, 0]
    snow[rows < 0.06, :] = 111                                  # polar night
    cloud[rows < 0.06, :] = 111
    snow[rows > 0.94, :] = 100                                  # Antarctica
    cloud[rows > 0.94, :] = 252
    snow[:2, :2] = 255                                          # fill
    cloud[:2, :2] = 255
    if d == SMOKE_OOB:
        snow[h // 2, w // 2] = 254                              # undefined
    return {SDS_SNOW: snow, SDS_CLOUD: cloud}


def make_smoke_sources(root, d_lo, d_hi, seed=20260918, skip=None,
                       absent=None, group="terra"):
    os.environ["SNOW05_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    w, h = SMOKE_W, SMOKE_H
    skip = tuple(SMOKE_SKIP if skip is None else skip)
    absent = tuple(SMOKE_ABSENT if absent is None else absent)
    days = [d_lo + dt.timedelta(days=i)
            for i in range((d_hi - d_lo).days + 1)]
    arrays = {d: smoke_fields(d, w, h) for d in days}
    sn, ver = Snow05Adapter.groups_available[group]
    mc.write_smoke_archive(
        root, "snow05", group, days, arrays, w, h, NSIDC_PREFIX, sn, ver,
        attrs={SDS_SNOW: {"_FillValue": FILL},
               SDS_CLOUD: {"_FillValue": FILL}},
        absent=absent, skip=skip)

    ad = Snow05Adapter()
    ad.w, ad.h = w, h
    listed = [d for d in days if d not in skip]
    lo, hi = min(listed), max(listed)
    t_lo = f10b.seconds_since_epoch(lo)
    t_hi = f10b.seconds_since_epoch(hi) + 86399
    truth = {}
    for b in sh.bins_overlapping(t_lo, t_hi):
        for f in range(ad.frames_per_bin):
            day = sh.frame_day(b, f, ad.frame_seconds)
            if day < lo:
                truth[(group, b, f)] = (None, "before_record")
            elif day > hi:
                truth[(group, b, f)] = (None, "after_record")
            elif day in skip or day in absent:
                truth[(group, b, f)] = (None, "absent_upstream")
            else:
                a, _ = ad.frame_from(arrays[day], {})
                mc.mask_bounds_low_memory(ad, a)
                stored = np.where(np.isnan(a), FILL,
                                  np.rint(a)).astype(np.uint8)
                truth[(group, b, f)] = (stored, None)
    return truth


ADAPTER = Snow05Adapter
