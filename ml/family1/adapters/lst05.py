"""MOD11C1 v061 — MODIS/Terra daily land-surface temperature, day and night,
on the 0.05-degree Climate Modeling Grid (family 1.0.tf, E-082 wave 4).

PLAIN ENGLISH. The land's skin temperature is not the air temperature a
weather station measures: it is how hot the GROUND is, which on a summer
afternoon can be forty degrees above the air and at night can be colder than
it. MODIS on Terra measures it twice a day for every 5.6 km cell of the
planet, cloud permitting, and has done so since February 2000 — the land twin
of the sea-surface temperature record the ocean families are built on
(`family1tf.tex` §4.3: "forcing, the land twin of SST").

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (CMR is keyless; the granules
are behind Earthdata Login and were read on a hosted runner):
  CMR collection C2565788888-LPCLOUD, short_name MOD11C1, version 061,
  9,587 granules, 2000-02-27 -> (still producing), about 40 MB each, one
  HDF-EOS2 file per calendar day. The data link is
  `https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD11C1.061/
  <granule>/<granule>.hdf`, and the granule id carries the PRODUCTION
  timestamp, so the URL cannot be derived from the date and CMR is not a
  convenience but the only honest way to find the file (contract rule 4).
  `ml/family1/adapters/_modis_cmg.py` holds the listing, the login, the
  download and the HDF4 reader; this module holds what the numbers MEAN.

WHAT IS STORED: C = 4, dtype FLOAT16, one group per satellite.
  lst_day    LST_Day_CMG   * 0.02 K - 273.15  -> degrees Celsius
  lst_night  LST_Night_CMG * 0.02 K - 273.15  -> degrees Celsius
  qc_day     QC_Day        the source's own uint8 bit field, unchanged
  qc_night   QC_Night      the source's own uint8 bit field, unchanged

WHY CELSIUS AND NOT KELVIN, measured rather than preferred: float16 spaces
its values 0.25 K apart at 300 K and 0.03 K apart at 300 - 273.15 = 27, so
storing the temperature in degrees Celsius keeps the quantisation an order of
magnitude below MOD11C1's own 1 K accuracy, and storing it in kelvin would
not. The source's raw step is 0.02 K; the store's is 0.03 K near freezing and
0.0625 K at 100 degrees C.

WHY THE QUALITY BITS ARE A CHANNEL AND NOT A MASK (`family1tf.tex` §5): a
consumer, not the store, decides the threshold, so `QC_Day` and `QC_Night`
are carried as they are — whole numbers 0..255, which float16 represents
EXACTLY (integers are exact in float16 up to 2,048). The two QC channels are
valid over the whole grid, including the ocean, where the mandatory-QA bits
say "LST not produced"; that is information, so it is kept, and it means
every tile of every frame is stored. The consequence is measured by the
probe, not argued: the QC planes are near-constant over water and cost very
little compressed.

THE FILL RULES, TAKEN FROM THE PRODUCER'S OWN FILE SPECIFICATION
(ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD11C1, read 2026-09-18)
AND CHECKED IN EVERY GRANULE'S SDS ATTRIBUTES:
  LST_Day_CMG / LST_Night_CMG  uint16, scale_factor 0.02, _FillValue 0,
                               valid_range 7500..65535. Raw 0 is not
                               measured; a raw value OUTSIDE the producer's
                               own valid_range is not measured either and is
                               counted `lst_outside_valid_range` — it is the
                               source's quality statement, not ours.
  QC_Day / QC_Night            uint8, valid_range 0..255, and the file
                               specification says in as many words that
                               "there is no _FillValue for this SDS", so 0 is
                               a legitimate value (good quality, every other
                               field zero) and is NEVER masked.
A granule whose attributes differ from those is a REFUSAL (`_modis_cmg`'s
`read_sds` compares them), because a product whose scaling changed must be
re-read by a human before a store is built on it.

THE CHANNEL BOUNDS ARE TRIPWIRES, NOT FILTERS. -150 to +100 degrees C: the
coldest place on Earth's surface ever measured from orbit is about -98 and
the hottest about +81 (the Lut Desert), and the producer's own valid_range
floor of 7500 is -123.15, so a real measurement never reaches either bound
and a corrupt one does. A value outside them becomes NaN and is COUNTED
(contract rule 3), never clipped.

THE GRID: 7,200 x 3,600 at 0.05 degrees, EPSG:4326, row 0 the northernmost
(90 N) — the source SDS' own orientation, so nothing is flipped. F = 5 daily
frames per five-day bin, `frame_seconds` = 86,400, so a frame IS a calendar
day.

THE CONTINUATION GROUPS. `family1tf.tex` §4.3 lists MYD11C1 (the same product
from Aqua, 2002-07-04 ->, 8,815 granules measured) and VJ121C1 / VJ221C1
(VIIRS on NOAA-20 and NOAA-21) as continuation groups of this store. They are
separate GROUPS on the identical grid, so a reader asks for the instrument it
wants; `LST05_GROUPS=terra,aqua` builds both. The default is Terra alone,
because each group is a store's worth of bytes (the note: about 350 GB per
satellite) and phase A is Terra.

SIZE. The note's arithmetic is 25.9 M pixels x 9,700 days x v ~= 0.29 x 0.5,
i.e. an assumed valid fraction of 0.145 over all four channels: about 350 GB
per satellite. The probe measures the real bytes per frame and the real valid
fraction per channel; `ml/family1/probes/lst05_2015-07.json` is the number
that replaces the estimate.

`LST05_SMOKE_GRID` ("W,H") shrinks the grid for `--smoke` and is read at
construction time, so the fresh adapter `stage_probe_grid` builds for itself
sees it too.
"""
import datetime as dt
import os

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _modis_cmg as mc

SCALE = 0.02
K0 = 273.15
LST_FILL = 0
LST_VALID = [7500, 65535]
QC_VALID = [0, 255]

SDS_LST_DAY = "LST_Day_CMG"
SDS_LST_NIGHT = "LST_Night_CMG"
SDS_QC_DAY = "QC_Day"
SDS_QC_NIGHT = "QC_Night"

LP_PREFIX = ("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
             "MOD11C1.061/")


class Lst05Adapter(mc.CmgAdapter):
    store = "lst05"
    title = ("Daily land-surface temperature day and night with quality, "
             "MODIS MOD11C1 v061, 0.05 degrees")
    short_name = "MOD11C1"
    version = "061"
    groups_available = {"terra": ("MOD11C1", "061"),
                        "aqua": ("MYD11C1", "061")}
    groups_default = ("terra",)
    groups_env = "LST05_GROUPS"
    smoke_grid_env = "LST05_SMOKE_GRID"

    licence = {
        "name": "NASA open data (no restrictions)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Wan, Z., S. Hook, G. Hulley (2021). MODIS/Terra Land "
                        "Surface Temperature/Emissivity Daily L3 Global "
                        "0.05Deg CMG V061. NASA EOSDIS Land Processes "
                        "Distributed Active Archive Center. "
                        "doi:10.5067/MODIS/MOD11C1.061"),
        "terms": ("earthdata.nasa.gov/engage/open-data-services-and-software/"
                  "data-information-policy, read 2026-09-18: NASA's Earth "
                  "science data are open, full and without restriction, free "
                  "of charge, with no period of exclusive access; users are "
                  "asked to cite the data set"),
    }
    channels = (("lst_day", "degC", -150.0, 100.0),
                ("lst_night", "degC", -150.0, 100.0),
                ("qc_day", "bitfield", 0.0, 255.0),
                ("qc_night", "bitfield", 0.0, 255.0))
    dtype = "float16"
    first_year = 2000
    sds = (SDS_LST_DAY, SDS_LST_NIGHT, SDS_QC_DAY, SDS_QC_NIGHT)
    sds_want = {
        SDS_LST_DAY: {"scale_factor": SCALE, "_FillValue": LST_FILL,
                      "valid_range": LST_VALID},
        SDS_LST_NIGHT: {"scale_factor": SCALE, "_FillValue": LST_FILL,
                        "valid_range": LST_VALID},
        SDS_QC_DAY: {"valid_range": QC_VALID},
        SDS_QC_NIGHT: {"valid_range": QC_VALID},
    }
    note_estimate = {
        "bytes": 350e9,
        "what": ("family1tf.tex §4.3 ledger: '25.9 M px x 9,700 d, v ~ 0.29 x "
                 "0.5: ~350 GB per satellite' for MOD11C1 v061 (Terra)")}
    qc_policy = (
        "the producer's own QC_Day and QC_Night uint8 bit fields are CARRIED "
        "AS CHANNELS, unchanged, so the consumer picks the threshold "
        "(family1tf.tex §5). Their bits are: 0-1 mandatory QA (00 good, 01 "
        "other quality, 10 not produced because of cloud, 11 not produced for "
        "another reason), 2 L1B data quality, 3 Terra/Aqua combined use, 4-5 "
        "emissivity error class, 6-7 LST error class. The file specification "
        "states that this SDS has NO fill value, so 0 is a real reading and "
        "is never masked; the two QC channels are therefore valid over the "
        "whole grid, ocean included, where the mandatory bits say why no LST "
        "was produced. LST itself is not measured where the raw value is 0 "
        "(_FillValue) or outside the producer's valid_range 7500..65535, and "
        "the second case is counted `lst_outside_valid_range`. The channel "
        "bounds -150..100 degrees C are tripwires: the coldest surface ever "
        "measured from orbit is about -98 and the hottest about +81, so a "
        "real value never reaches them and a corrupt one does; a value "
        "outside becomes NaN and is counted (`out_of_bounds`), never clipped")
    sources = (LP_PREFIX + "<granule>/<granule>.hdf",
               mc.CMR + "?short_name=MOD11C1&version=061")
    verified = (
        "2026-09-18 from the sandbox, keyless: the CMR collection search "
        "(C2565788888-LPCLOUD, MOD11C1 v061, cloud_hosted, time_start "
        "2000-02-27) and the granule search for 2015-07-01..03 (three "
        "granules of 44.7, 44.5 and 44.6 MB, each with one https "
        "'lp-prod-protected' .hdf data link whose name carries the production "
        "timestamp); the whole-collection paged listing measured 9,587 "
        "granules; and the producer's file specification "
        "(ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD11C1) for the "
        "SDS names, scale factor 0.02, fill 0, valid_range 7500..65535 and "
        "the QC bit legend. The granules themselves are behind Earthdata "
        "Login and were opened on a GitHub-hosted runner by the probe, which "
        "re-checks every one of those attributes in every file it reads")
    notes = (
        "MOD11C1 v061, one HDF-EOS2 granule a day. LST is stored in degrees "
        "CELSIUS, not kelvin, because float16 spaces its values 0.25 K apart "
        "at 300 K and 0.03 K apart near zero Celsius — in kelvin the store's "
        "own quantisation would be a quarter of the product's 1 K accuracy. "
        "QC_Day and QC_Night are carried as channels with the producer's own "
        "bits and no fill (the file specification says this SDS has none), so "
        "both are valid over the whole grid and every tile of every frame is "
        "stored; LST is valid only where a cloud-free retrieval was made, "
        "which the probe measures. Row 0 is the northernmost row, 90 N, as in "
        "the source SDS. Aqua (MYD11C1) and the VIIRS continuations are "
        "separate groups on the identical grid; LST05_GROUPS selects them and "
        "the default is Terra alone.")
    smoke_window = ("2015-07-01", "2015-07-20")
    smoke_probe_month = "2015-07"
    WORKERS = 3

    # ---------------------------------------------------------- the frame --
    def frame_from(self, arrs, attrs):
        """{sds: raw} -> (float32 [h, w, 4] with NaN, counts)."""
        h, w = self.h, self.w
        out = np.empty((h, w, 4), np.float32)
        counts = {}
        for i, (name, ch) in enumerate(((SDS_LST_DAY, "lst_day"),
                                        (SDS_LST_NIGHT, "lst_night"))):
            raw = arrs[name]
            if raw.dtype != np.uint16:
                raise mc.FormatError(f"{name}: {raw.dtype}, expected uint16")
            bad = (raw < LST_VALID[0]) | (raw > LST_VALID[1])
            n_fill = int((raw == LST_FILL).sum())
            n_bad = int(bad.sum()) - n_fill
            v = raw.astype(np.float32) * np.float32(SCALE) - np.float32(K0)
            v[bad] = np.nan
            out[:, :, i] = v
            counts[f"fill_pixels_{ch}"] = n_fill
            if n_bad:
                counts.setdefault("lst_outside_valid_range", {})[ch] = n_bad
        for i, (name, ch) in enumerate(((SDS_QC_DAY, "qc_day"),
                                        (SDS_QC_NIGHT, "qc_night")), start=2):
            raw = arrs[name]
            if raw.dtype != np.uint8:
                raise mc.FormatError(f"{name}: {raw.dtype}, expected uint8")
            out[:, :, i] = raw.astype(np.float32)
        return out, counts

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.smoke_reinit()
        return truth


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 600, 300
# NO LISTED-AND-MISSING GRANULE in the main smoke, and that is a property of
# the contract rather than a convenience: a granule CMR lists and the archive
# will not serve is `ctx.note_absent`, which stops the year's marker and makes
# the fetch refuse (the 2026-09-14 rule). The day the archive simply does not
# have — not listed at all — is the ordinary `absent_upstream` frame, and
# SMOKE_SKIP is that day. The refusal path is exercised by a test that passes
# `absent=` explicitly.
SMOKE_ABSENT = ()
SMOKE_SKIP = (dt.date(2015, 7, 14),)       # not listed: absent upstream
SMOKE_OOB = dt.date(2015, 7, 3)            # one pixel at raw 65535


def smoke_fields(d, w, h):
    """A deterministic granule: a warm land band, cloud elsewhere."""
    k = (d - dt.date(2015, 1, 1)).days % 11
    yy = (np.arange(h, dtype=np.float64)[:, None] - h / 2) / h
    xx = (np.arange(w, dtype=np.float64)[None, :] - w / 2) / w
    land = (np.abs(yy) < 0.28) & (np.abs(xx) < 0.42)
    clear = land & (np.sin(11 * xx + k) + np.cos(9 * yy - k) > -0.4)
    day_c = 20.0 + 18.0 * np.sin(7 * xx + k) * np.cos(5 * yy + k)
    night_c = day_c - 12.0
    out = {}
    for name, c in ((SDS_LST_DAY, day_c), (SDS_LST_NIGHT, night_c)):
        raw = np.zeros((h, w), np.uint16)
        raw[clear] = np.rint((c[clear] + K0) / SCALE).astype(np.uint16)
        out[name] = raw
    # the mandatory-QA bits: 00 where a retrieval was made, 11 where not
    for name in (SDS_QC_DAY, SDS_QC_NIGHT):
        qc = np.full((h, w), 3, np.uint8)
        qc[clear] = 0
        qc[land & ~clear] = 2                     # not produced: cloud
        out[name] = qc
    if d == SMOKE_OOB:
        out[SDS_LST_DAY][h // 2, w // 2] = 65535   # inside valid_range, absurd
    return out


def make_smoke_sources(root, d_lo, d_hi, seed=20260918, skip=None,
                       absent=None, group="terra"):
    """The archive in its real layout, and the truth for every frame.

    Returns {(group, bin, frame): (stored-dtype array [h, w, 4] or None,
    reason or None)} for every frame of every bin overlapping the record.
    """
    os.environ["LST05_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    w, h = SMOKE_W, SMOKE_H
    skip = tuple(SMOKE_SKIP if skip is None else skip)
    absent = tuple(SMOKE_ABSENT if absent is None else absent)
    days = [d_lo + dt.timedelta(days=i)
            for i in range((d_hi - d_lo).days + 1)]
    arrays = {d: smoke_fields(d, w, h) for d in days}
    sn, ver = Lst05Adapter.groups_available[group]
    mc.write_smoke_archive(
        root, "lst05", group, days, arrays, w, h, LP_PREFIX, sn, ver,
        attrs={SDS_LST_DAY: dict(Lst05Adapter.sds_want[SDS_LST_DAY]),
               SDS_LST_NIGHT: dict(Lst05Adapter.sds_want[SDS_LST_NIGHT]),
               SDS_QC_DAY: dict(Lst05Adapter.sds_want[SDS_QC_DAY]),
               SDS_QC_NIGHT: dict(Lst05Adapter.sds_want[SDS_QC_NIGHT])},
        absent=absent, skip=skip)

    ad = Lst05Adapter()
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
                truth[(group, b, f)] = (a.astype(np.float16), None)
    return truth


ADAPTER = Lst05Adapter
