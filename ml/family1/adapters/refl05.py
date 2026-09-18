"""MOD09CMG v061 — MODIS/Terra daily surface reflectance, bands 1-7 and the
state quality word, on the 0.05-degree Climate Modeling Grid (family 1.0.tf,
E-082 wave 4).

PLAIN ENGLISH. This is the COLOUR of the ground, corrected for the
atmosphere: how much of the sunlight in each of seven wavelengths a 5.6 km
cell sent back to space, every day since February 2000. From it come the
vegetation indices, the snow indices, the burn indices and the albedo — which
is why the note's ledger row calls it "vegetation, snow, burn state; albedo".
It is also the largest of the three MODIS fields by an order of magnitude:
about 540 MB a day, 5.4 TB for the Terra record.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (CMR is keyless; the granules
are behind Earthdata Login and were read on a hosted runner):
  CMR collection C2565788876-LPCLOUD, short_name MOD09CMG, version 061,
  9,622 granules, 2000-02-24 -> (still producing), 505-560 MB each, one
  HDF-EOS2 file per calendar day, `day_night_flag` DAY. The data link is
  `https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/
  MOD09CMG.061/<granule>/<granule>.hdf`; the granule id carries the
  production timestamp so the URL comes from CMR (contract rule 4).
  `_modis_cmg.py` holds the listing, the login, the download and the reader.

WHAT IS STORED: C = 9, dtype FLOAT16, one group.
  b1 .. b7        the seven "Coarse Resolution Surface Reflectance Band n"
                  SDS, int16 * 0.0001 -> reflectance (dimensionless)
  state_qa_lo     bits 0-7 of "Coarse Resolution State QA"
  state_qa_hi     bits 8-15 of the same word

C = 9, NOT THE NOTE'S 8, AND HERE IS WHY. `family1tf.tex` §4.3 plans "bands
1-7 and state QA (C = 8)". The state QA is a UINT16 bit field, and float16 —
the dtype seven reflectance channels require — represents integers exactly
only to 2,048, so a single channel CANNOT carry a 16-bit quality word without
losing bits. The choice is therefore between dropping half the producer's
quality word and adding one channel, and §5 of the same note settles it: "the
source's own quality flag is kept in qc and nothing is homogenised". So the
word is split into its two BYTES, each of which float16 holds exactly, and
`state_qa_hi << 8 | state_qa_lo` reconstructs the producer's uint16 bit for
bit. The deviation from the ledger's C = 8 is recorded here, in store.json's
notes and in `ml/family1/BUILD_LOG.md`.

  bits 0-1   cloud state: clear (00), cloudy (01), mixed (10), not set,
             assumed clear (11)              -> state_qa_lo
  bit  2     cloud shadow                    -> state_qa_lo
  bits 3-5   land/water flag: shallow ocean (000), land (001), coastline
             (010), shallow inland water (011), ephemeral water (100), deep
             inland water (101), continental/moderate ocean (110), deep ocean
             (111)                           -> state_qa_lo
  bits 6-7   aerosol quantity: climatology (00), low, average, high
                                             -> state_qa_lo
  bits 8-9   cirrus detected: none, small, average, high  -> state_qa_hi
  bit  10    internal cloud algorithm flag   -> state_qa_hi
  bit  11    internal fire algorithm flag    -> state_qa_hi
  bit  12    MOD35 snow/ice flag             -> state_qa_hi
  bit  13    adjacent to cloud               -> state_qa_hi
  bit  14    BRDF correction performed       -> state_qa_hi
  bit  15    internal snow algorithm flag    -> state_qa_hi

WHEN A PIXEL IS NOT MEASURED, AND WHY THE FILL IS READ RATHER THAN DECLARED.
The producer's 2010 file specification (revision 6.0.3, for Collection 6)
gives the reflectance bands `_FillValue` 0. THE v061 GRANULES SAY -28,672,
which is what the index stage found on a runner and refused on — correctly,
because "a product whose scaling changed is refused, not rescaled". A fill
value is a property of the FILE, and every one of these SDS states its own,
so the adapter now READS each `_FillValue` and uses it, records the value it
saw (`fill_value_seen`), and refuses an SDS that declares none. The scale
factor and the valid range, which must NOT differ between granules of one
collection, stay declared and are still compared in every file.

MOD09CMG is a DAY product, so the night side of a composite, and everything
the retrieval declined, is fill. Its State QA declares `_FillValue` 0 and
`valid_range` [1, 65535] — so in v061 an all-zero quality word IS the
producer saying "nothing here", not "clear, shallow ocean, climatological
aerosol". Both QA bytes are therefore NaN where the word is the producer's
fill OR where all seven bands are fill; the two agree almost everywhere and
the counts (`pixels_qa_fill`, `pixels_no_band_retrieved`,
`pixels_qa_fill_with_a_band`, `pixels_band_without_qa`) say by how much,
which is the cross-check. `lst05` keeps its QC EVERYWHERE for the opposite
reason: MOD11C1's file specification says in as many words that its QC SDS
has no fill value, and its mandatory bits say "not produced because of
cloud" versus "for another reason" — information about the absence itself.

THE CHANNEL BOUNDS CARRY ONE float16 STEP OF HEADROOM. The producer's
valid_range maps to reflectance [-0.01, 1.6], and float16's step THERE is
0.00098 — ten times the source's own 0.0001 — so a stored value sits up to
half a step either side of the number the file carried, and a bound set
exactly at the producer's endpoint could make the store fail its own tile
check on a legitimate extreme. The bounds are therefore -0.02 .. 1.7: one
whole float16 step outside the producer's range, and far inside anything a
corrupt file would produce. The QA bytes are bounded 0..255, which float16
holds exactly.

PRECISION. float16 spaces its values 0.000488 apart at reflectance 0.5, where
the source's raw step is 0.0001. The store therefore quantises about five
times coarser than the source scaling and about ten times FINER than
MOD09's own stated absolute accuracy (~0.005), which is the trade the
family's float16 storage makes everywhere.

THE GRID: 7,200 x 3,600 at 0.05 degrees, EPSG:4326, row 0 the northernmost
(90 N) — the source SDS' own orientation. F = 5 daily frames per five-day
bin, so a frame IS a calendar day.

SIZE, AND THE ORDER THE RECORD IS BUILT IN. The note's arithmetic is
v ~= 0.15: about 700 GB for Terra, which is more than one rented box should
assemble in one pass and 5.4 TB of downloads. The store is therefore built
2015-2026 FIRST and the rest only after the probe's measured bytes per frame
have been compared with that estimate (`ml/family1/probes/refl05_2015-07.json`).

`REFL05_SMOKE_GRID` ("W,H") shrinks the grid for `--smoke`.
"""
import datetime as dt
import os

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _modis_cmg as mc

SCALE = 1e-4
# THE FILL VALUE IS READ FROM THE FILE, NOT DECLARED. The producer's 2010
# file specification (revision 6.0.3, written for Collection 6) says the
# reflectance bands' `_FillValue` is 0; the v061 granule the index opened on a
# runner declares **-28,672**, and its State QA declares fill 0 with
# `valid_range` [1, 65535]. A fill value is a property of the FILE and the
# file states it, so the adapter reads each SDS' own `_FillValue` and uses it,
# and REFUSES an SDS that declares none — the scaling and the valid range,
# which must not change, stay declared and are still compared.
REFL_FILL_DOC = -28672
REFL_VALID = [-100, 16000]
BANDS = tuple(f"Coarse Resolution Surface Reflectance Band {i}"
              for i in range(1, 8))
SDS_QA = "Coarse Resolution State QA"
LP_PREFIX = ("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
             "MOD09CMG.061/")
QA_BITS = {
    "0-1": "cloud state: 00 clear, 01 cloudy, 10 mixed, 11 not set (assumed "
           "clear)",
    "2": "cloud shadow",
    "3-5": "land/water: 000 shallow ocean, 001 land, 010 coastline, 011 "
           "shallow inland water, 100 ephemeral water, 101 deep inland "
           "water, 110 continental/moderate ocean, 111 deep ocean",
    "6-7": "aerosol quantity: 00 climatology, 01 low, 10 average, 11 high",
    "8-9": "cirrus detected: 00 none, 01 small, 10 average, 11 high",
    "10": "internal cloud algorithm flag",
    "11": "internal fire algorithm flag",
    "12": "MOD35 snow/ice flag",
    "13": "pixel is adjacent to cloud",
    "14": "BRDF correction performed",
    "15": "internal snow algorithm flag",
}


class Refl05Adapter(mc.CmgAdapter):
    store = "refl05"
    title = ("Daily surface reflectance bands 1-7 with the state quality "
             "word, MODIS MOD09CMG v061, 0.05 degrees")
    short_name = "MOD09CMG"
    version = "061"
    groups_available = {"terra": ("MOD09CMG", "061")}
    groups_default = ("terra",)
    groups_env = "REFL05_GROUPS"
    smoke_grid_env = "REFL05_SMOKE_GRID"

    licence = {
        "name": "NASA open data (no restrictions)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Vermote, E., R. Wolfe (2021). MODIS/Terra Surface "
                        "Reflectance Daily L3 Global 0.05Deg CMG V061. NASA "
                        "EOSDIS Land Processes Distributed Active Archive "
                        "Center. doi:10.5067/MODIS/MOD09CMG.061"),
        "terms": ("earthdata.nasa.gov/engage/open-data-services-and-software/"
                  "data-information-policy, read 2026-09-18: NASA's Earth "
                  "science data are open, full and without restriction and "
                  "free of charge; users are asked to cite the data set"),
    }
    channels = tuple(
        (f"b{i}", "reflectance", -0.02, 1.7) for i in range(1, 8)) + (
        ("state_qa_lo", "bitfield", 0.0, 255.0),
        ("state_qa_hi", "bitfield", 0.0, 255.0))
    dtype = "float16"
    first_year = 2000
    sds = BANDS + (SDS_QA,)
    sds_want = {b: {"scale_factor": SCALE, "valid_range": REFL_VALID}
                for b in BANDS}
    note_estimate = {
        "bytes": 700e9,
        "what": ("family1tf.tex §4.3 ledger: 'v ~ 0.15: ~700 GB (Terra)' for "
                 "MOD09CMG v061; the native record is ~600 MB/day, 5.5 TB")}
    qc_policy = (
        "the producer's uint16 'Coarse Resolution State QA' word is carried "
        "WHOLE, split into its two bytes because float16 represents integers "
        "exactly only to 2,048: state_qa_hi << 8 | state_qa_lo is the "
        "producer's word bit for bit (this is the C = 9 the note's ledger row "
        "plans as C = 8; §5's rule that the source's quality flag is kept and "
        "nothing is homogenised is what decides it). Its bits are 0-1 cloud "
        "state, 2 cloud shadow, 3-5 land/water class, 6-7 aerosol quantity, "
        "8-9 cirrus, 10 internal cloud, 11 internal fire, 12 MOD35 snow/ice, "
        "13 adjacent to cloud, 14 BRDF corrected, 15 internal snow. "
        "Reflectance is not measured where the raw value is the SDS' OWN "
        "_FillValue -- READ from the file, because the 2010 file "
        "specification says 0 and the v061 granules say -28672, and a fill "
        "value is a property of the file (the value seen is recorded in "
        "`fill_value_seen`) -- or outside the producer's valid_range "
        "-100..16000, and the second case is counted "
        "`refl_outside_valid_range`. Both QA bytes are NaN where the State QA "
        "word is its own fill (v061 declares 0, with valid_range [1, 65535]) "
        "or where all seven bands are fill; the two agree almost everywhere "
        "and `pixels_qa_fill_with_a_band` and `pixels_band_without_qa` "
        "measure the difference. The channel bounds -0.02..1.7 are the producer's range plus "
        "one float16 step of headroom (that step is 0.00098 at reflectance "
        "1.6, ten times the source's own 0.0001), so a legitimate extreme "
        "cannot fail the store's own bounds check whichever way float16 "
        "rounds it; a value outside them becomes NaN and is counted "
        "(`out_of_bounds`), never clipped")
    sources = (LP_PREFIX + "<granule>/<granule>.hdf",
               mc.CMR + "?short_name=MOD09CMG&version=061")
    verified = (
        "2026-09-18 from the sandbox, keyless: the CMR collection search "
        "(C2565788876-LPCLOUD, MOD09CMG v061, cloud_hosted, time_start "
        "2000-02-24; the two LANCE near-real-time collections were seen and "
        "are NOT used), the granule search for 2015-07-01 (537.2 MB, "
        "day_night_flag DAY, one https 'lp-prod-protected' .hdf data link), "
        "and the per-year counts 301 (2000), 366 (2015), 252 (2026 to date) "
        "of 9,622 in all. The SDS names, the int16 scaling 0.0001, fill 0, "
        "valid_range -100..16000 and the State QA bit legend come from the "
        "producer's file specification "
        "(ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD09CMG, "
        "revision 6.0.3). The granules are behind Earthdata Login and were "
        "opened on a GitHub-hosted runner by the probe, which re-checks the "
        "shape and every declared attribute in every file it reads")
    notes = (
        "MOD09CMG v061, one 505-560 MB HDF-EOS2 granule a day. C = 9, NOT the "
        "ledger's C = 8: the uint16 state QA word is split into two byte "
        "channels because float16 cannot hold a 16-bit integer exactly, and "
        "state_qa_hi << 8 | state_qa_lo reconstructs it. Both QA channels are "
        "NaN where all seven bands are fill (no observation) and kept "
        "otherwise. Reflectance is stored as float16, which quantises about "
        "five times coarser than the source's 0.0001 step and ten times finer "
        "than MOD09's own ~0.005 accuracy. Row 0 is the northernmost row, "
        "90 N, as in the source SDS. The record is built 2015-2026 first: the "
        "note projects ~700 GB and 5.4 TB of downloads for the whole Terra "
        "record.")
    smoke_window = ("2015-07-01", "2015-07-20")
    smoke_probe_month = "2015-07"
    WORKERS = 2          # 540 MB a granule: two in flight is already 1.1 GB

    def specs(self):
        out = super().specs()
        for g in out:
            out[g]["state_qa_bits"] = dict(QA_BITS)
            out[g]["state_qa_rule"] = (
                "state_qa_hi << 8 | state_qa_lo is the producer's uint16 "
                "'Coarse Resolution State QA' word, bit for bit; both bytes "
                "are NaN exactly where all seven reflectance bands are fill")
            out[g]["channels_note"] = (
                "C = 9 where family1tf.tex §4.3's ledger row plans C = 8: the "
                "16-bit quality word is two float16-exact bytes rather than "
                "one lossy channel")
        return out

    # ---------------------------------------------------------- the frame --
    def frame_from(self, arrs, attrs):
        h, w = self.h, self.w
        out = np.empty((h, w, 9), np.float32)
        counts = {}
        allfill = np.ones((h, w), bool)
        for i, name in enumerate(BANDS):
            raw = arrs[name]
            if raw.dtype != np.int16:
                raise mc.FormatError(f"{name}: {raw.dtype}, expected int16")
            f = self.fill_of(name, attrs)
            fill = raw == np.int16(f)
            bad = (raw < REFL_VALID[0]) | (raw > REFL_VALID[1])
            n_bad = int((bad & ~fill).sum())
            v = raw.astype(np.float32) * np.float32(SCALE)
            v[fill | bad] = np.nan
            out[:, :, i] = v
            allfill &= fill
            counts[f"fill_pixels_b{i + 1}"] = int(fill.sum())
            counts.setdefault("fill_value_seen", {})[f"b{i + 1}"] = int(f)
            if n_bad:
                counts.setdefault("refl_outside_valid_range",
                                  {})[f"b{i + 1}"] = n_bad
        qa = arrs[SDS_QA]
        if qa.dtype != np.uint16:
            raise mc.FormatError(f"{SDS_QA}: {qa.dtype}, expected uint16")
        qf = self.fill_of(SDS_QA, attrs)
        qa_fill = qa == np.uint16(qf)
        lo = (qa & np.uint16(0x00FF)).astype(np.float32)
        hi = (qa >> np.uint16(8)).astype(np.float32)
        # NOT MEASURED where the producer's OWN fill says so, and also where
        # no band was retrieved — the two agree almost everywhere and the
        # counts say by how much, which is the cross-check.
        gone = qa_fill | allfill
        lo[gone] = np.nan
        hi[gone] = np.nan
        out[:, :, 7] = lo
        out[:, :, 8] = hi
        counts["pixels_no_band_retrieved"] = int(allfill.sum())
        counts["pixels_qa_fill"] = int(qa_fill.sum())
        counts["pixels_qa_fill_with_a_band"] = int((qa_fill & ~allfill).sum())
        counts["pixels_band_without_qa"] = int((allfill & ~qa_fill).sum())
        counts.setdefault("fill_value_seen", {})["state_qa"] = int(qf)
        return out, counts

    @staticmethod
    def fill_of(name, attrs):
        """The SDS' OWN `_FillValue`. An SDS with none is a refusal."""
        a = (attrs or {}).get(name) or {}
        if "_FillValue" not in a:
            raise mc.FormatError(
                f"{name}: the SDS declares no _FillValue. Without it 'not "
                f"measured' cannot be told from a reading, and the value is "
                f"not the same in every collection (the 2010 file "
                f"specification says 0 for the reflectance bands and the v061 "
                f"granules say -28672), so it is never assumed")
        return int(a["_FillValue"])

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.smoke_reinit()
        return truth


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 600, 300
SMOKE_ABSENT = ()                          # see lst05: a listed-and-missing
SMOKE_SKIP = (dt.date(2015, 7, 13),)       # granule is a REFUSAL, not a gap
SMOKE_OOB = dt.date(2015, 7, 5)            # one band-1 pixel at raw 16000


def smoke_fields(d, w, h):
    """A deterministic granule: a lit half with retrievals, a dark half."""
    k = (d - dt.date(2015, 1, 1)).days % 9
    yy = np.arange(h, dtype=np.float64)[:, None] / h
    xx = np.arange(w, dtype=np.float64)[None, :] / w
    lit = np.broadcast_to((xx > 0.15) & (xx < 0.85), (h, w))
    clear = lit & (np.sin(10 * xx + k) + np.cos(8 * yy - k) > -0.5)
    out = {}
    for i, name in enumerate(BANDS, start=1):
        raw = np.full((h, w), np.int16(REFL_FILL_DOC), np.int16)
        r = 0.03 + 0.28 * i / 7.0 + 0.10 * np.sin(6 * xx + i + k) \
            * np.cos(4 * yy - i)
        raw[clear] = np.rint(np.broadcast_to(r, (h, w))[clear]
                             / SCALE).astype(np.int16)
        out[name] = raw
    qa = np.zeros((h, w), np.uint16)
    qa[clear] = np.uint16(0b1000_0001_0100_1001)     # snow, fire, land, ...
    qa[np.broadcast_to(lit, (h, w)) & ~clear] = np.uint16(0b0000_0100_0000_0001)
    out[SDS_QA] = qa
    if d == SMOKE_OOB:
        out[BANDS[0]][h // 2, w // 2] = 16000        # the valid_range maximum
    return out


def make_smoke_sources(root, d_lo, d_hi, seed=20260918, skip=None,
                       absent=None, group="terra"):
    os.environ["REFL05_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    w, h = SMOKE_W, SMOKE_H
    skip = tuple(SMOKE_SKIP if skip is None else skip)
    absent = tuple(SMOKE_ABSENT if absent is None else absent)
    days = [d_lo + dt.timedelta(days=i)
            for i in range((d_hi - d_lo).days + 1)]
    arrays = {d: smoke_fields(d, w, h) for d in days}
    sn, ver = Refl05Adapter.groups_available[group]
    mc.write_smoke_archive(
        root, "refl05", group, days, arrays, w, h, LP_PREFIX, sn, ver,
        attrs={**{b: {**Refl05Adapter.sds_want[b],
                      "_FillValue": REFL_FILL_DOC} for b in BANDS},
               SDS_QA: {"_FillValue": 0, "valid_range": [1, 65535]}},
        absent=absent, skip=skip)

    ad = Refl05Adapter()
    ad.w, ad.h = w, h
    attrs = {**{b: {"_FillValue": REFL_FILL_DOC} for b in BANDS},
             SDS_QA: {"_FillValue": 0}}
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
                a, _ = ad.frame_from(arrays[day], attrs)
                mc.mask_bounds_low_memory(ad, a)
                truth[(group, b, f)] = (a.astype(np.float16), None)
    return truth


ADAPTER = Refl05Adapter
