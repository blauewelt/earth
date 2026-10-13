"""PACE OCI daily 4 km SURFACE REFLECTANCE in the near and shortwave infrared
(family 1.3, E-092 phase C, plan D13).

PLAIN ENGLISH. Floating plastic, foam, seaweed and ship wakes all have one
thing in common: water is almost black in the infrared, and anything floating
on it is not. A floating-debris index (FDI) or a floating-algae index (FAI)
is the reflectance at a near-infrared band minus a straight line drawn
between a red and a shortwave-infrared band. This store keeps the eight
bands such an index needs — the two red / red-edge baselines (645 and 740 nm),
the near infrared (865 nm) and the five shortwave-infrared bands PACE's Ocean
Color Instrument carries (1038, 1249, 1618, 2131, 2258 nm) — as daily 4 km
global maps, tiled and compressed the way `pace4k` keeps PACE's ocean colour,
on the SAME grid, so a model reads both with one tile scheme. ALL surfaces
are kept, not ocean only: coasts and estuaries are where floating debris
shows, and a land mask would have to be somebody's choice of coastline.

THE PRODUCT, MEASURED 2026-10-13 FROM THE SANDBOX, ANONYMOUSLY
(`ml/family1/probes/pace_swir_measure.json`). PACE_OCI_L3M_SFREFL version 3.1
(C3620140468-OB_CLOUD, doi:10.5067/PACE/OCI/L3M/SFREFL/3.1, time_start
2024-03-05), daily 4 km mapped `rhos` ("Surface reflectance" — corrected for
the molecular atmosphere and gases, NOT for aerosol):

    PACE_OCI.<YYYYMMDD>.L3m.DAY.SFREFL.V3_1.rhos.4km.nc        (final)
    PACE_OCI.<YYYYMMDD>.L3m.DAY.SFREFL.V3_1.rhos.4km.NRT.nc    (near real time,
        collection PACE_OCI_L3M_SFREFL_NRT, C3620140454-OB_CLOUD)

`rhos` is float32 [lat 4320, lon 8640, wavelength 122] — WAVELENGTH LAST —
`_FillValue` -32767.0, valid_min -0.05, valid_max 1.5, no scale or offset;
the wavelength axis is 122 int32 values, 346..895 nm in ~5 nm steps (2.5 nm
in the red edge) plus 1038, 1249, 1618, 2131 and 2258 nm.

THE ACCESS PATH IS OB.DAAC's OPeNDAP (Hyrax), DAP4 SERVER-SIDE SUBSETTING, AND
IT IS ANONYMOUS. A whole day is 3.94 GB (CMR's `granule_size`); the store
wants 8 of the 122 bands. A plain byte-range read of the file is REFUSED (HTTP
416 on the OPeNDAP path and on `getfile`; the cumulus bucket answers 302 to
Earthdata Login), so this store cannot range-read the netCDF's chunks itself.
A DAP4 constraint expression `rhos[r0:1:r1][c0:1:c1][w0:s:w1]` IS honoured,
including a STRIDE on the wavelength axis (measured 2026-10-13:
`[56:28:84]` returned exactly 645 and 739 nm), and `/wavelength[w0:s:w1]` in
the same expression returns the wavelengths of the slice, so every response
says which bands it holds. Two separate projections of `rhos` in one
expression are refused (HTTP 400), so the eight bands cost one request per
CONTIGUOUS run of their indices (`band_requests`): [56], [84], [110] and
[117..121] — four requests, plus `/lat;/lon` on the first. NOT strided runs,
although the server honours them: measured 2026-10-13 on uncached days, a
two-band strided slab ([56:28:84], [110:7:117]) took 289 s and 292 s, where
one contiguous band took 85 s and four contiguous bands ~90 s — the server
walks a stride element by element. Four contiguous requests move the same
bytes (~375 MB a day) in roughly half the time of three strided ones.

THE BANDS, AND WHY THESE INDICES. Chosen from the file's OWN wavelength axis
by nearest value, never typed in (`choose_bands`; the measured table
`MEASURED_BANDS` is a test fixture and a cross-check, not the source):

    rhos_645    index  56, 645 nm   exact — the FAI red baseline (Hu 2009)
    rhos_740    index  84, 739 nm   the nearest band to 740: 739 nm is 1 nm
                                    away, 742 nm (index 85) is 2 nm. 740 nm is
                                    Sentinel-2's red-edge B06 (~740 nm), the
                                    band FDI's baseline starts from (Biermann
                                    et al. 2020), so the store keeps the PACE
                                    band closest to it and RECORDS 739 as the
                                    wavelength in every tile_grid.json
    rhos_865    index 110, 865 nm   exact — the near infrared FDI/FAI read
    rhos_1038   index 117, 1038 nm
    rhos_1249   index 118, 1249 nm
    rhos_1618   index 119, 1618 nm  (the shortwave-infrared baseline; PACE has
                                    no band on plastic's 1,732 nm absorption)
    rhos_2131   index 120, 2131 nm
    rhos_2258   index 121, 2258 nm

A band whose nearest wavelength is more than 2.5 nm from its target is a
REFUSAL, and every response's wavelength slice is compared with the chosen
values, so a reordered axis cannot silently shift a channel.

FINAL FILES ONLY, AND THE NRT BOUNDARY IS RECORDED. The final (refined)
collection runs from 2024-03-05 to a live edge that lags by weeks (measured
2026-10-13: final through 2026-08-31, NRT through 2026-10-11, NRT from
2025-07-01). The store holds FINAL files only: an NRT file is replaced when
the refined one is made, and a store that mixed them would change under its
readers. So a day after the final edge is `after_record` even where an NRT
file exists (counted `nrt_after_final`), a day inside the record with only an
NRT file is `absent_upstream` (counted `nrt_only_inside_record`), and the
index writes the boundary (`nrt_boundary`: final's last day, NRT's first and
last day, how many NRT days lie past the final edge).

WHAT IS STORED: C = 8, float16, unit 1 (dimensionless reflectance), bounds
-0.05 .. 1.5 — the producer's own valid_min/valid_max. The fill (-32767) and
any non-finite value become NaN; a value outside the bounds becomes NaN and is
counted (`out_of_bounds`), never clipped. float16 quantises a reflectance
near 0.01 to 8e-6 and near 1 to 5e-4 (measured max abs error 4.9e-4, p99
relative 4.3e-4 on 2025-07-01). The producer masked six level-2 flags at
binning (ATMFAIL, CLDICE, HIGLINT, HILT, HISATZEN, HISOLZEN — the files'
`l2_flag_names`, recorded per frame); no per-pixel flag survives into level
3, so there is nothing further to apply.

TWO FALSIFIERS RUN ON EVERY FRAME (E-092 §6):

  1. THE PIXEL RE-READ. One valid pixel per frame (finite in all eight
     channels; the middle one in row-major order) is asked for AGAIN by an
     independent single-pixel DAP4 request over the whole band range
     `[56:1:121]`, and its eight values must equal the frame's to float16
     precision (float16(native) == float16(frame)). The measurement found
     them bit-identical in float32 (2025-07-01); the store holds float16, so
     that is the precision compared. A mismatch is a REFUSAL: it means the
     subset, the band selection or the decode is wrong, and every frame after
     it would be too.
  2. DARK WATER IN THE SHORTWAVE INFRARED. Over CLEAR WATER — the pixels where
     PACE's own chlorophyll retrieval (`pace4k`'s BGC product, v3.2, same
     day) is valid, i.e. the ocean-colour processing's own cloud, glint and
     quality screen — rhos at 2258 nm should have a median below 0.01 and a
     99th percentile below 0.03 (measured 0.0052 / 0.0217 on 2025-07-01).
     THE RULE: a day with at least `DARK_MIN_PIXELS` (10,000) clear-water
     pixels is JUDGED; a judged day missing either soft threshold is KEPT,
     counted (`dark_swir_soft_fail`) and named in `dark_swir_soft_fail_days`
     — real aerosol, sun glint at the mask's edge or a dust outbreak can do
     that, and a store that refused weather would refuse the record. A judged
     day at or above THREE times the soft median (0.03) or above 0.10 at the
     99th percentile is a GROSS failure and a REFUSAL: no atmosphere makes the
     global clear ocean that bright at 2258 nm, so it means a wrong band, a
     wrong unit or a wrong mask. Every judged day's (n, median, p99) is kept
     in `dark_swir_days`, and the largest are `max_dark_swir_p50` /
     `max_dark_swir_p99`. A day whose BGC file is not published is not
     judged (`dark_swir_no_mask`), and one with too few clear pixels is
     counted `dark_swir_too_few`.

LICENCE AND ATTRIBUTION FROM THE FILES' OWN ATTRIBUTES (read from the DMR of
2025-07-01 and from every response): `license` =
https://science.nasa.gov/earth-science/earth-science-data/data-information-policy/
(NASA's open data policy: full, open, no restriction), `institution` = NASA
Goddard Space Flight Center, Ocean Ecology Laboratory, Ocean Biology
Processing Group, `identifier_product_doi` = 10.5067/PACE/OCI/L3M/SFREFL/3.1.
Each frame counts the `license` and `processing_version` it was read with.

PHASE B / NOT HERE: the 2 km and 0.1 degree maps of the same product, the
1.2 km level-2 swath (Earthdata login, ~100 GB a day), and an aerosol
correction — `rhos` carries the aerosol path radiance, which is why the open
ocean's rhos at 865 nm correlates (weakly, r = 0.43) with aot_865.
"""
import calendar
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.parse

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _common as cm
from family1.adapters import _modis_sin as ms          # cmr_page
from family1.adapters import pace4k as p4              # grid, OPeNDAP, BGC

# ---- reused from pace4k: the same grid and the same server -----------------
H, W = p4.H, p4.W
X0, Y0 = p4.X0, p4.Y0
OPENDAP = p4.OPENDAP
NC_LOCK = p4.NC_LOCK        # ONE process-wide lock: HDF5 is not thread-safe
FormatError = p4.FormatError

SHORT = "PACE_OCI_L3M_SFREFL"
SHORT_NRT = "PACE_OCI_L3M_SFREFL_NRT"
CONCEPT = "C3620140468-OB_CLOUD"
CONCEPT_NRT = "C3620140454-OB_CLOUD"
VERSION = "3.1"
VTAG = "V3_1"
DOI = "10.5067/PACE/OCI/L3M/SFREFL/3.1"
RECORD_START = dt.date(2024, 3, 5)     # CMR's time_start and the first granule
FILL = np.float32(-32767.0)
LICENCE_URL = ("https://science.nasa.gov/earth-science/earth-science-data/"
               "data-information-policy/")

# target wavelength (nm) -> channel. The INDEX is chosen from the file's axis.
TARGETS = (("rhos_645", 645), ("rhos_740", 740), ("rhos_865", 865),
           ("rhos_1038", 1038), ("rhos_1249", 1249), ("rhos_1618", 1618),
           ("rhos_2131", 2131), ("rhos_2258", 2258))
BAND_TOLERANCE_NM = 2.5
# MEASURED 2026-10-13 on 2025-07-01's axis: channel -> (index, wavelength nm)
MEASURED_BANDS = {"rhos_645": (56, 645), "rhos_740": (84, 739),
                  "rhos_865": (110, 865), "rhos_1038": (117, 1038),
                  "rhos_1249": (118, 1249), "rhos_1618": (119, 1618),
                  "rhos_2131": (120, 2131), "rhos_2258": (121, 2258)}
# the whole measured axis (122 int32), what the smoke's synthetic files carry
MEASURED_AXIS = (
    346, 351, 356, 361, 366, 371, 375, 380, 385, 390, 395, 400, 405, 410, 415,
    420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490,
    495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565,
    570, 575, 580, 586, 615, 620, 625, 630, 635, 640, 642, 645, 647, 650, 652,
    655, 657, 660, 662, 665, 667, 670, 672, 675, 677, 679, 682, 697, 699, 702,
    704, 707, 709, 712, 714, 719, 724, 729, 734, 739, 742, 744, 747, 749, 752,
    754, 772, 774, 779, 784, 789, 794, 799, 804, 809, 814, 819, 824, 829, 835,
    840, 845, 850, 855, 860, 865, 870, 875, 880, 885, 890, 895, 1038, 1249,
    1618, 2131, 2258)

UNIT = "1 (dimensionless surface reflectance, rhos)"
CHANNELS = tuple((n, UNIT, -0.05, 1.5) for n, _t in TARGETS)

# the dark-water falsifier (E-092 §6), see the module docstring
DARK_CHANNEL = "rhos_2258"
DARK_P50, DARK_P99 = 0.01, 0.03                 # soft: reported
GROSS_P50, GROSS_P99 = 0.03, 0.10               # gross: refused
DARK_MIN_PIXELS = 10000


# ================================================================ names ====
def object_name(d, nrt=False):
    return (f"PACE_OCI.{d:%Y%m%d}.L3m.DAY.SFREFL.{VTAG}.rhos.4km"
            + (".NRT" if nrt else "") + ".nc")


def granule_name(entry, nrt=False):
    """The file name of a CMR entry. CMR's `title` for SFREFL is sometimes
    `<SHORT>_<file>_<version>` (2024 and 2025, measured) and sometimes the
    bare file (2026); `producer_granule_id` is the bare file throughout."""
    t = str(entry.get("producer_granule_id") or "")
    if not t:
        t = str(entry.get("title") or "")
        for sn in (SHORT_NRT, SHORT):
            if t.startswith(sn + "_"):
                t = t[len(sn) + 1:]
                break
        if t.endswith("_" + VERSION):
            t = t[:-len(VERSION) - 1]
    return t


def parse_day(name, nrt=False):
    """`PACE_OCI.20250701.L3m.DAY.SFREFL.V3_1.rhos.4km.nc` -> the date.
    REFUSES anything that is not this product's daily 4 km rhos granule (and
    an NRT name where a final one is expected, and vice versa)."""
    t = str(name or "")
    want = f".L3m.DAY.SFREFL.{VTAG}.rhos.4km" + (".NRT" if nrt else "") \
        + ".nc"
    if not t.startswith("PACE_OCI.") or not t.endswith(want):
        raise FormatError(f"{name!r}: not a PACE OCI daily 4 km SFREFL rhos "
                          f"{'NRT ' if nrt else ''}granule (expected "
                          f"PACE_OCI.<YYYYMMDD>{want})")
    ymd = t.split(".")[1]
    if len(ymd) != 8 or not ymd.isdigit():
        raise FormatError(f"{name!r}: {ymd!r} is not a YYYYMMDD date")
    return dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))


def day_dir(d):
    return f"L3SMI/{d:%Y}/{d:%m%d}/"


def choose_bands(axis):
    """The file's wavelength axis -> [(channel, index, wavelength nm)].

    Nearest value to each target; REFUSES a target with nothing within
    BAND_TOLERANCE_NM and a tie (two bands equally near), because either
    would be a choice this code made rather than a fact of the file."""
    ax = np.asarray(axis, np.float64).ravel()
    out = []
    for name, target in TARGETS:
        dist = np.abs(ax - float(target))
        i = int(np.argmin(dist))
        if dist[i] > BAND_TOLERANCE_NM:
            raise FormatError(f"{name}: the nearest wavelength to {target} nm "
                              f"is {ax[i]:g} nm (index {i}), more than "
                              f"{BAND_TOLERANCE_NM} nm away")
        if int((dist == dist[i]).sum()) > 1:
            raise FormatError(f"{name}: two wavelengths are equally near "
                              f"{target} nm ({ax[dist == dist[i]].tolist()})")
        out.append((name, i, int(round(ax[i]))))
    return out


def band_requests(indices):
    """Sorted band indices -> [(start, 1, stop)], the maximal CONTIGUOUS runs.
    DAP4 refuses two projections of one variable in an expression, so each
    run is one request; a strided run would save requests and cost ~3x the
    server time (measured, module docstring). [56, 84, 110, 117..121] ->
    (56, 1, 56), (84, 1, 84), (110, 1, 110), (117, 1, 121)."""
    idx = sorted(int(i) for i in indices)
    out, k = [], 0
    while k < len(idx):
        j = k
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append((idx[k], 1, idx[j]))
        k = j + 1
    return out


def run_indices(run):
    a, s, b = run
    return list(range(a, b + 1, s))


def subset_url(d, run, h=H, w=W, latlon=False):
    a, s, b = run
    ce = (f"/rhos[0:1:{h - 1}][0:1:{w - 1}][{a}:{s}:{b}]"
          f";/wavelength[{a}:{s}:{b}]") + (";/lat;/lon" if latlon else "")
    return (f"{OPENDAP}{day_dir(d)}{object_name(d)}.dap.nc4?dap4.ce="
            + urllib.parse.quote(ce, safe=";/:"))


def pixel_url(d, r, c, lo, hi):
    ce = f"/rhos[{r}][{c}][{lo}:1:{hi}];/wavelength[{lo}:1:{hi}]"
    return (f"{OPENDAP}{day_dir(d)}{object_name(d)}.dap.nc4?dap4.ce="
            + urllib.parse.quote(ce, safe=";/:"))


def axis_url(d):
    return (f"{OPENDAP}{day_dir(d)}{object_name(d)}.dap.nc4?dap4.ce="
            + urllib.parse.quote("/wavelength", safe=";/:"))


def grid_of(h=H, w=W):
    """pace4k's grid (the same 1/24-degree global lat/lon, row 0 north), with
    the three descriptive fields that are about pace4k's VARIABLES replaced."""
    g = dict(p4.grid_of(h, w))
    g["extent_note"] = ("[west, south, east, north] in degrees; ALL surfaces "
                        "— land, ice, inland and coastal water and the open "
                        "ocean — no land mask")
    g["same_grid_as"] = ("pace4k and oc4k — the ocean-colour stores share "
                         "this grid exactly, so one tile scheme reads all "
                         "three")
    g["source_variables"] = (
        "rhos (float32 [lat, lon, wavelength], _FillValue -32767, "
        "dimensionless) of PACE_OCI_L3M_SFREFL v3.1, eight bands of its 122, "
        "on this grid with no resampling")
    return g


# ============================================================ the files ====
def _check_grid(lat, lon, h, w, where):
    dy, dx = 180.0 / int(h), 360.0 / int(w)
    want_lat = Y0 - (np.arange(int(h)) + 0.5) * dy
    want_lon = X0 + (np.arange(int(w)) + 0.5) * dx
    if (lat.size != int(h) or lon.size != int(w)
            or not (np.allclose(lat, want_lat, atol=1e-4)
                    and np.allclose(lon, want_lon, atol=1e-4))):
        raise FormatError(f"{where}: lat/lon differ from the declared grid "
                          f"(lat {lat[0]}..{lat[-1]} n={lat.size}, lon "
                          f"{lon[0]}..{lon[-1]} n={lon.size}); row 0 is "
                          f"declared NORTHERNMOST")


ROW_BLOCK = 512     # rows per decode block: ~90 MB of a 5-band response


def iter_subset(path, want_nm, h=H, w=W, need_grid=False, block=ROW_BLOCK):
    """One DAP4 response (or its local emulation), in ROW BLOCKS -> yields
    (r0, float32 [rows, w, k] with NaN, file attributes).

    Blocks, not the whole array: a 5-band response decodes to 0.75 GB, and
    the probe already holds the previous frame and its float64 copy while
    this one is read (measured: the whole-array reader peaked at 5.8 GB RSS
    and was killed in a 7 GB sandbox). The response's chunks are 8 rows tall
    (measured, [8, 1024, 5]), so a 512-row block decompresses nothing twice
    — reading band by band instead was measured 4.4x slower (17.8 s vs 4.1 s
    for one 5-band day). REFUSES a response whose wavelength slice is not
    `want_nm`, whose grid is not the declared one, or whose `rhos` is not
    [lat, lon, wavelength] of the declared size."""
    import netCDF4
    try:
        ds = netCDF4.Dataset(path)
    except OSError as ex:
        raise IOError(f"{os.path.basename(path)}: not a readable netCDF file "
                      f"({ex}) — a truncated or refused response") from None
    try:
        where = os.path.basename(path)
        attrs = {k: ds.getncattr(k) for k in
                 ("license", "processing_version", "identifier_product_doi",
                  "institution", "l2_flag_names") if k in ds.ncattrs()}
        for k in ("rhos", "wavelength"):
            if k not in ds.variables:
                raise FormatError(f"{where}: no variable {k!r} "
                                  f"({sorted(ds.variables)})")
        wl = np.asarray(ds.variables["wavelength"][:]).ravel()
        if [int(x) for x in wl] != [int(x) for x in want_nm]:
            raise FormatError(f"{where}: wavelengths {wl.tolist()} where "
                              f"{list(want_nm)} were asked for")
        if need_grid:
            if "lat" not in ds.variables or "lon" not in ds.variables:
                raise FormatError(f"{where}: no lat/lon in the response")
            _check_grid(np.asarray(ds.variables["lat"][:], np.float64),
                        np.asarray(ds.variables["lon"][:], np.float64),
                        h, w, where)
        v = ds.variables["rhos"]
        if tuple(v.dimensions) != ("lat", "lon", "wavelength"):
            raise FormatError(f"{where}: rhos dimensions {v.dimensions}, "
                              f"expected (lat, lon, wavelength)")
        if tuple(v.shape) != (int(h), int(w), len(want_nm)):
            raise FormatError(f"{where}: rhos is {tuple(v.shape)}, expected "
                              f"{(int(h), int(w), len(want_nm))}")
        v.set_auto_maskandscale(False)
        at = {k: v.getncattr(k) for k in v.ncattrs()}
        if "scale_factor" in at or "add_offset" in at:
            raise FormatError(f"{where}: rhos carries scale_factor/add_offset "
                              f"({at}); the measured product has none")
        fill = np.float32(at.get("_FillValue", FILL))
        for r0 in range(0, int(h), int(block)):
            a = np.asarray(v[r0:r0 + int(block)], np.float32)
            a[a == fill] = np.nan
            a[~np.isfinite(a)] = np.nan
            yield r0, a, attrs
    finally:
        ds.close()


def read_subset(path, want_nm, h=H, w=W, need_grid=False):
    """`iter_subset` whole -> (float32 [h, w, k] with NaN, attributes)."""
    parts, attrs = [], {}
    for _r0, a, attrs in iter_subset(path, want_nm, h, w, need_grid):
        parts.append(a)
    return np.concatenate(parts, axis=0), attrs


def local_subset(native, out, run, latlon=False):
    """`--source-dir`: the DAP4 server's answer, emulated from a local native
    file — the SAME constraint semantics (a strided band slice plus its
    wavelengths, lat/lon when asked) written as a netCDF-4 file, so the reader
    above is exercised exactly as on the network."""
    import netCDF4
    idx = run_indices(run)
    with netCDF4.Dataset(native) as src:
        v = src.variables["rhos"]
        v.set_auto_maskandscale(False)
        a = np.asarray(v[:, :, idx[0]:idx[-1] + 1:run[1]], np.float32)
        wl = np.asarray(src.variables["wavelength"][idx[0]:idx[-1] + 1:run[1]])
        gattrs = {k: src.getncattr(k) for k in src.ncattrs()}
        lat = np.asarray(src.variables["lat"][:]) if latlon else None
        lon = np.asarray(src.variables["lon"][:]) if latlon else None
    with netCDF4.Dataset(out, "w", format="NETCDF4") as ds:
        for k, val in gattrs.items():
            ds.setncattr(k, val)
        ds.createDimension("lat", a.shape[0])
        ds.createDimension("lon", a.shape[1])
        ds.createDimension("wavelength", a.shape[2])
        wv = ds.createVariable("wavelength", "i4", ("wavelength",))
        wv[:] = wl
        if latlon:
            ds.createVariable("lat", "f4", ("lat",))[:] = lat
            ds.createVariable("lon", "f4", ("lon",))[:] = lon
        r = ds.createVariable("rhos", "f4", ("lat", "lon", "wavelength"),
                              zlib=True, complevel=1, fill_value=FILL)
        r.set_auto_maskandscale(False)
        r[:] = a
    return out


def local_pixel(native, r, c, lo, hi):
    import netCDF4
    with netCDF4.Dataset(native) as src:
        v = src.variables["rhos"]
        v.set_auto_maskandscale(False)
        a = np.asarray(v[r, c, lo:hi + 1], np.float32).ravel()
        wl = np.asarray(src.variables["wavelength"][lo:hi + 1]).ravel()
    return a, wl


def local_axis(native):
    import netCDF4
    with netCDF4.Dataset(native) as src:
        return [int(x) for x in np.asarray(src.variables["wavelength"][:])]


def read_pixel_response(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        v = ds.variables["rhos"]
        v.set_auto_maskandscale(False)
        return (np.asarray(v[:], np.float32).ravel(),
                np.asarray(ds.variables["wavelength"][:]).ravel())


def pick_pixel(arr):
    """The middle pixel, in row-major order, that is finite in EVERY channel;
    None when there is none."""
    ok = np.isfinite(arr).all(axis=2)
    flat = np.flatnonzero(ok)
    del ok
    if not flat.size:
        return None
    k = int(flat[flat.size // 2])
    return k // arr.shape[1], k % arr.shape[1]


def dark_stats(v2258, clear):
    """(n, median, p99) of rhos(2258) over the clear-water mask."""
    g = v2258[clear & np.isfinite(v2258)]
    if not g.size:
        return 0, None, None
    p50, p99 = np.percentile(g.astype(np.float64), [50, 99])
    return int(g.size), float(p50), float(p99)


def judge_dark(n, p50, p99):
    """-> 'too_few' | 'pass' | 'soft' | 'gross' (the rule in the docstring)."""
    if n < DARK_MIN_PIXELS:
        return "too_few"
    if p50 >= GROSS_P50 or p99 >= GROSS_P99:
        return "gross"
    if p50 >= DARK_P50 or p99 >= DARK_P99:
        return "soft"
    return "pass"


# ============================================================== adapter ====
class PaceSwirAdapter(sh.GridAdapter):
    store = "pace_swir"
    title = ("PACE OCI daily 4 km surface reflectance in the red edge, near "
             "and shortwave infrared (SFREFL v3.1 rhos at 645, 739, 865, "
             "1038, 1249, 1618, 2131 and 2258 nm) — the bands a "
             "floating-debris or floating-algae index needs, all surfaces")
    family = "13"
    distribution = "public"
    licence = {
        "name": "NASA open data (no restrictions)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("NASA Ocean Biology Processing Group (2024-2026). "
                        "PACE OCI Level-3 Global Mapped Surface Reflectance "
                        "Data, version 3.1 (doi:10.5067/PACE/OCI/L3M/SFREFL/"
                        "3.1). NASA Goddard Space Flight Center, Ocean "
                        "Ecology Laboratory, Ocean Biology Distributed Active "
                        "Archive Center."),
        "terms": ("the files' own `license` attribute, read 2026-10-13 from "
                  "the DMR of 2025-07-01 and counted per frame: "
                  f"'{LICENCE_URL}' — NASA's Earth science data are open, "
                  "full and without restriction, free of charge; the "
                  "`institution` attribute is 'NASA Goddard Space Flight "
                  "Center, Ocean Ecology Laboratory, Ocean Biology Processing "
                  "Group'"),
    }
    time_dtype = "int32"
    # NO CREDENTIAL, measured: OB.DAAC's Hyrax answered every DAP4 subset of
    # SFREFL with HTTP 200 and no login (2026-10-13).
    credentials = ()
    channels = CHANNELS
    log2_fp = p4.Pace4kAdapter.log2_fp
    log2_dt = p4.Pace4kAdapter.log2_dt
    per_year = True
    first_year = RECORD_START.year
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = sh.TILE
    zstd_level = sh.DEFAULT_LEVEL
    grid = grid_of()
    note_estimate = {"bytes": 42.8e9 * (8.0 / 6.0),
                     "what": ("pace_swir_measure.json: one real frame of the "
                              "six SWIR/NIR bands through the ShardWriter was "
                              "117.4 MB (all surfaces) -> 42.8 GB a year; "
                              "scaled by 8/6 for the two added baselines")}
    qc_policy = (
        "FINAL (refined) files only; NRT files are never stored — a day past "
        "the final edge is `after_record` (counted `nrt_after_final` where an "
        "NRT file exists) and a day inside the record with only an NRT file "
        "is `absent_upstream` (counted `nrt_only_inside_record`). The "
        "producer masked ATMFAIL, CLDICE, HIGLINT, HILT, HISATZEN and "
        "HISOLZEN at binning and no per-pixel flag reaches level 3; the "
        "store applies nothing further. The fill (-32767) and any non-finite "
        "value become NaN; a value outside -0.05..1.5 (the producer's own "
        "valid range) becomes NaN and is counted (`out_of_bounds`), never "
        "clipped. ALL SURFACES are kept — no land mask. `rhos` is corrected "
        "for Rayleigh scattering and gases, NOT for aerosol. Every frame "
        "passes two checks (E-092 §6): an independent single-pixel re-read "
        "equal to the frame to float16 precision (refused otherwise), and "
        "the dark-water test on rhos_2258 over the BGC product's "
        "clear-water pixels (soft failures kept and named, a gross failure "
        "refused)")
    sources = (OPENDAP + "L3SMI/<YYYY>/<MMDD>/PACE_OCI.<YYYYMMDD>.L3m.DAY."
               f"SFREFL.{VTAG}.rhos.4km.nc", ms.CMR)
    verified = (
        "2026-10-13 from the sandbox, anonymously: CMR's collections "
        "PACE_OCI_L3M_SFREFL (C3620140468) and PACE_OCI_L3M_SFREFL_NRT "
        "(C3620140454), both version 3.1, time_start 2024-03-05; the daily "
        "4 km final granules listed by name pattern — 289 in 2024 "
        "(2024-03-05..12-31), 363 in 2025, 241 in 2026 to 2026-08-31 — and "
        "the NRT ones, 184 in 2025 (from 2025-07-01) and 282 in 2026 to "
        "2026-10-11; CMR's `title` prefixed `<SHORT>_..._3.1` in 2024-25 and "
        "bare in 2026, `producer_granule_id` bare throughout; the DMR of "
        "2025-07-01 (rhos float32 [4320, 8640, 122], fill -32767, valid "
        "-0.05..1.5, wavelength int32 x 122); a plain byte range refused "
        "(416); DAP4 subsets with a wavelength stride ([56:28:84] -> 645, "
        "739 nm) and a wavelength slice in the same expression, and two "
        "projections of rhos refused (400); the six SWIR/NIR bands of "
        "2025-07-01 fetched (258 MB, 138 s), a single-pixel re-read equal "
        "in float32, and clear-water rhos(2258) median 0.0052, p99 0.0217; "
        "then SIX REAL DAYS (2024-03-05, 2024-07-15, 2025-01-15, 2025-07-01, "
        "2026-04-15, 2026-08-31) through this adapter's own fetch_frames and "
        "the real ShardWriter: four contiguous requests a day, the pixel "
        "re-read equal on all six, clear-water rhos(2258) median "
        "0.0052-0.0063 and p99 0.0213-0.0223 (2025-07-01 reproducing the "
        "earlier 0.0052 / 0.0217), no value out of bounds, 2024-03-13 "
        "absent upstream as the listing says")
    notes = (
        "C = 8 on pace4k's own global 4 km grid (4320 x 8640, 1/24 degree, "
        "row 0 north): rhos at 645, 739 (stored as `rhos_740`, the nearest "
        "band to 740 nm), 865, 1038, 1249, 1618, 2131 and 2258 nm, "
        "dimensionless, float16, all surfaces. Read through OB.DAAC's "
        "OPeNDAP with DAP4 constraint expressions (no login; plain byte "
        "ranges are refused): four requests a day, one per contiguous run "
        "of the band indices, about 375 MB a day against the file's 3.9 GB, "
        "plus pace4k's BGC chlorophyll (12 MB) as the clear-water mask for "
        "the dark-water check and one single-pixel re-read. Final files "
        "only; the index records where the near-real-time files take over "
        "(`nrt_boundary`). MEASURED 2026-10-13 on six real days spread "
        "over the record (ml/family1/probes/pace_swir_probe.json): a frame "
        "is 157.8 MB stored (151.9-161.4 MB, 462-542 of 578 tiles, valid "
        "fraction 0.319-0.338), 367 MB fetched and 38-43 s to encode, and "
        "167 s a frame from the sandbox with three download workers; the "
        "record's 893 final days to 2026-08-31 are ~141 GB stored and "
        "~328 GB read. Built in MONTHLY lanes on hosted runners: a month is "
        "~4.9 GB of shards and ~1.4 h, where a quarter (~14.4 GB) would not "
        "fit a runner's ~14 GB of disk.")
    smoke_window = ("2024-03-05", "2024-03-19")
    smoke_probe_month = "2024-03"

    WORKERS = 3

    def __init__(self):
        g = (os.environ.get("PACE_SWIR_SMOKE_GRID") or "").split(",")
        self.w, self.h = (int(g[0]), int(g[1])) if len(g) == 2 else (W, H)
        self._listing = {}
        self._bands = None
        if (self.w, self.h) != (W, H):
            self.notes = (f"{self.notes}\nSMOKE GRID: PACE_SWIR_SMOKE_GRID "
                          f"set the grid to {self.w} x {self.h} instead of "
                          f"{W} x {H}. This is a synthetic store.")

    def group_grids(self):
        return {self.store: grid_of(self.h, self.w)}

    def specs(self):
        out = super().specs()
        for spec in out.values():
            bands = {n: (i, nm) for n, i, nm in (self._bands or [])} or \
                MEASURED_BANDS
            for c in spec["channels"]:
                i, nm = bands[c["name"]]
                c["wavelength_nm"] = nm
                c["wavelength_index"] = i
            spec["record_start"] = str(RECORD_START)
            spec["product"] = {"short_name": SHORT, "version": VERSION,
                               "doi": DOI, "variable": "rhos",
                               "aerosol_corrected": False}
        return out

    # ------------------------------------------------------------ paths ----
    def _src(self, ctx, *parts):
        return os.path.join(ctx.source_dir, self.store, *parts)

    def native_path(self, ctx, d):
        return self._src(ctx, "opendap", *day_dir(d).strip("/").split("/"),
                         object_name(d))

    def bgc_path(self, ctx, d):
        return self._src(ctx, "opendap", *day_dir(d).strip("/").split("/"),
                         p4.object_name("bgc", d))

    # ---------------------------------------------------------- listing ----
    def listing(self, ctx, year, nrt=False):
        """{date: {name, bytes_cmr}} for one year of the final (or NRT)
        collection."""
        key = (int(year), bool(nrt))
        if key in self._listing:
            return self._listing[key]
        sn = SHORT_NRT if nrt else SHORT
        pat = (f"PACE_OCI.{int(year):04d}*.L3m.DAY.SFREFL.{VTAG}.rhos.4km"
               + (".NRT" if nrt else "") + ".nc")
        counts = {"cmr_pages": 0}
        if ctx.source_dir:
            p = self._src(ctx, f"cmr_sfrefl{'_nrt' if nrt else ''}_"
                               f"{int(year):04d}.json")
            if not os.path.exists(p):
                self._listing[key] = ({}, {"year_not_listed": 1})
                return self._listing[key]
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            entries = (json.loads(raw).get("feed") or {}).get("entry") or []
        else:
            entries, after = [], None
            while True:
                got, after, hits, nb = ms.cmr_page(
                    {"short_name": sn, "version": VERSION, "page_size": 2000,
                     "sort_key": "start_date", "readable_granule_name": pat,
                     "options[readable_granule_name][pattern]": "true"},
                    after, attempts=ctx.a.attempts)
                entries += got
                counts["cmr_pages"] += 1
                counts["cmr_hits"] = hits
                if not after or not got:
                    break
                if counts["cmr_pages"] > 100:
                    sys.exit(f"REFUSING pace_swir: CMR paged past 100 pages "
                             f"for {sn} {year}")
        files = {}
        for e in entries:
            name = granule_name(e, nrt)
            d = parse_day(name, nrt)
            if d.year != int(year):
                raise FormatError(f"{name}: not a {year} granule")
            if d in files:
                raise FormatError(f"{name}: two granules for {d}")
            try:
                size = float(e.get("granule_size") or 0.0) * 1024 * 1024
            except (TypeError, ValueError):
                size = 0.0
            files[d] = {"name": name, "bytes_cmr": int(size)}
        counts["granules_listed"] = len(files)
        self._listing[key] = (files, counts)
        return self._listing[key]

    def years(self, ctx):
        """The calendar years the window's FRAMES can need — through
        `ctx.grid_frame_days[1]`, the year-boundary rule pace4k learnt (a
        December lane's last bin runs into January)."""
        ys = sorted(ctx.years)
        if not ys:
            return []
        last = max(ys)
        span = getattr(ctx, "grid_frame_days", None)
        if span:
            last = max(last, span[1].year)
        return list(range(max(min(ys), RECORD_START.year), last + 1))

    def all_years(self, ctx):
        """Every year from the record's start to the later of the window and
        today: the FINAL EDGE is a property of the whole record, not of one
        lane's years (a lane ending 12-30 must not call 12-31 after_record
        because next year's listing was not asked)."""
        top = max([dt.datetime.now(dt.timezone.utc).year] + self.years(ctx))
        return list(range(RECORD_START.year, top + 1))

    def final_days(self, ctx, years=None):
        out = set()
        for y in (years or self.all_years(ctx)):
            out |= set(self.listing(ctx, y)[0])
        return out

    def nrt_days(self, ctx, years=None):
        out = set()
        for y in (years or self.all_years(ctx)):
            out |= set(self.listing(ctx, y, nrt=True)[0])
        return out

    def record(self, ctx):
        """(first, last) day of the FINAL record. The first is RECORD_START
        and the listing must agree; the last is the final collection's live
        edge over the whole record."""
        days = self.final_days(ctx)
        if not days:
            return None, None
        lo = min(days)
        if lo != RECORD_START:
            raise FormatError(f"the final listing starts {lo}, the product "
                              f"declares {RECORD_START}")
        return lo, max(days)

    def nrt_boundary(self, ctx):
        fin = self.final_days(ctx)
        nrt = self.nrt_days(ctx)
        last = max(fin) if fin else None
        after = sorted(d for d in nrt if last is None or d > last)
        inside = sorted(d for d in nrt if last is not None
                        and RECORD_START <= d <= last and d not in fin)
        return {"final_last_day": str(last) if last else None,
                "nrt_first_day": str(min(nrt)) if nrt else None,
                "nrt_last_day": str(max(nrt)) if nrt else None,
                "nrt_days": len(nrt),
                "nrt_days_after_final": len(after),
                "nrt_after_final_first": str(after[0]) if after else None,
                "nrt_only_inside_record": [str(d) for d in inside],
                "policy": ("final files only: NRT days past the final edge "
                           "are after_record, NRT-only days inside it are "
                           "absent_upstream")}

    def record_frames(self, ctx, group):
        lo, hi = self.record(ctx)
        if lo is None:
            return None
        return sum(1 for d in self.final_days(ctx) if lo <= d <= hi)

    def empty_months(self, ctx, y):
        lo, hi = self.record(ctx)
        have = {f"{d.year:04d}-{d.month:02d}"
                for d in self.listing(ctx, y)[0]}
        out = []
        for m in range(1, 13):
            m0 = dt.date(y, m, 1)
            m1 = dt.date(y, m, calendar.monthrange(y, m)[1])
            if lo is None or m1 < lo or m0 > hi:
                continue
            if f"{y:04d}-{m:02d}" not in have:
                out.append(f"{y:04d}-{m:02d}")
        return out

    def declared_lanes(self, years):
        """MONTHLY lanes, OPT-IN: with `PACE_SWIR_LANES=months` (the
        assembly's `adapter_env`) a month is a declared lane of its year when
        any of its days lies inside [RECORD_START, min(final edge, --end)],
        so the assembler refuses a missing month. `--lanes months` cannot say
        this: it demands m01..m12 of 2024 (the record opens 03-05) and of the
        live year. Without the knob nothing is declared — the smoke, the
        probe and an unnamed-lane build keep the framework's default."""
        if (os.environ.get("PACE_SWIR_LANES") or "").strip() != "months":
            return {}
        lo = RECORD_START
        hi = getattr(self, "_edge", None)
        top = getattr(self, "_window_hi", None)
        if hi is not None and top is not None:
            hi = min(hi, top)
        out = {}
        for y in years:
            names = []
            for m in range(1, 13):
                m0 = dt.date(int(y), m, 1)
                m1 = dt.date(int(y), m, calendar.monthrange(int(y), m)[1])
                if m1 < lo or (hi is not None and m0 > hi):
                    continue
                names.append(f"m{m:02d}")
            if names:
                out[str(y)] = names
        return out

    # ------------------------------------------------------------ bands ----
    def bands(self, ctx, d=None):
        """[(channel, index, nm)], chosen from a real file's axis once."""
        if self._bands is not None:
            return self._bands
        d = d or self.record(ctx)[0]
        if d is None:
            sys.exit("REFUSING pace_swir: no final granule to read the "
                     "wavelength axis from")
        if ctx.source_dir:
            axis = local_axis(self.native_path(ctx, d))
        else:
            tmp = tempfile.mkdtemp(prefix="swir_axis_", dir=ctx.scratch)
            try:
                p = os.path.join(tmp, "axis.nc4")
                if f10b.fetch_first([axis_url(d)], p,
                                    attempts=ctx.a.attempts) is None:
                    sys.exit(f"REFUSING pace_swir: {axis_url(d)} is 404 — "
                             f"the listed first day has no file")
                import netCDF4
                with NC_LOCK, netCDF4.Dataset(p) as ds:
                    axis = [int(x) for x in
                            np.asarray(ds.variables["wavelength"][:])]
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        try:
            self._bands = choose_bands(axis)
        except FormatError as e:
            sys.exit(f"REFUSING pace_swir: {d}: {e}")
        self._axis_len = len(axis)
        return self._bands

    def requests(self, ctx):
        b = self.bands(ctx)
        return band_requests([i for _n, i, _nm in b])

    # ------------------------------------------------------------ index ----
    def index(self, ctx):
        lo, hi = self.record(ctx)
        if lo is None:
            sys.exit(f"REFUSING pace_swir: CMR lists no final SFREFL granule "
                     f"for any of the years {self.all_years(ctx)} — an empty "
                     f"listing is a refusal (ml/CLAUDE.md, the 2026-09-14 "
                     f"rule)")
        self._edge = hi
        self._window_hi = ctx.d_hi
        empty = {str(y): self.empty_months(ctx, y) for y in self.years(ctx)
                 if self.empty_months(ctx, y)}
        if empty:
            sys.exit(f"REFUSING pace_swir: a calendar month inside the record "
                     f"has no final granule: {empty}")
        # the ONE real read: the first final day this context's window holds
        # (a monthly lane reads a day it fetches anyway), else the record's
        fin_w = sorted(d for d in self.final_days(ctx, self.years(ctx))
                       if ctx.d_lo <= d <= ctx.d_hi)
        first = fin_w[0] if fin_w else lo
        bands = self.bands(ctx, first)
        per, nb = {}, 0
        for y in self.all_years(ctx):
            files, _c = self.listing(ctx, y)
            per[str(y)] = len(files)
            nb += sum(r["bytes_cmr"] for r in files.values())
        fin = self.final_days(ctx)
        absent, d = [], lo
        while d <= hi:
            if d not in fin:
                absent.append(str(d))
            d += dt.timedelta(days=1)
        out = {
            "dataset": f"PACE OCI L3M SFREFL v{VERSION} daily 4 km rhos",
            "url": OPENDAP, "version": VERSION, "doi": DOI,
            "collections": {"final": [SHORT, CONCEPT],
                            "nrt": [SHORT_NRT, CONCEPT_NRT]},
            "access": ("OB.DAAC OPeNDAP (Hyrax) DAP4 server-side subsetting, "
                       "anonymous; plain byte ranges refused (416)"),
            "days_per_year_final": per,
            "bytes_cmr_whole_files": int(nb),
            "record": [str(lo), str(hi)],
            "days_in_record": (hi - lo).days + 1,
            "days_absent_upstream": absent,
            "nrt_boundary": self.nrt_boundary(ctx),
            "bands": [{"channel": n, "index": i, "wavelength_nm": nm,
                       "target_nm": dict(TARGETS)[n]} for n, i, nm in bands],
            "bands_match_measurement": all(
                MEASURED_BANDS[n] == (i, nm) for n, i, nm in bands),
            "requests_per_day": [list(r) for r in self.requests(ctx)],
            "frames_per_bin": self.frames_per_bin,
            "frame_seconds": self.frame_seconds,
            "grid": grid_of(self.h, self.w),
            "dark_swir_rule": {"channel": DARK_CHANNEL,
                               "soft": [DARK_P50, DARK_P99],
                               "gross": [GROSS_P50, GROSS_P99],
                               "min_pixels": DARK_MIN_PIXELS,
                               "mask": ("pace4k's BGC v3.2 chlor_a valid, "
                                        "same day")},
        }
        out["first_file"] = {"day": str(first),
                             **self.read_first_day(ctx, first)}
        return out

    def read_first_day(self, ctx, d):
        try:
            arr, counts, oob = self.read_day(ctx, d)
        except FormatError as e:
            sys.exit(f"REFUSING pace_swir: the first day {d} does not match "
                     f"the declared product: {e}")
        except (IOError, OSError) as e:
            sys.exit(f"REFUSING pace_swir: the first day {d} could not be "
                     f"read: {e}")
        out = dict(counts)
        out["valid_fraction"] = {
            n: round(float(np.isfinite(arr[:, :, i]).mean()), 6)
            for i, n in enumerate(self.channel_names)}
        out["range"] = {}
        for i, n in enumerate(self.channel_names):
            g = arr[:, :, i]
            g = g[np.isfinite(g)]
            out["range"][n] = [None, None] if not g.size else \
                [round(float(g.min()), 6), round(float(g.max()), 6)]
        if oob:
            out["out_of_bounds"] = oob
        return out

    # ---------------------------------------------------------- one day ----
    def fetch_day(self, ctx, d):
        """The day's subsets (+ the BGC mask) on local disk -> (paths, bytes,
        tmpdir). DOWNLOAD ONLY; the decode is in the main thread."""
        tmpdir = tempfile.mkdtemp(prefix="swir_", dir=ctx.scratch)
        paths, nbytes = {"runs": []}, 0
        try:
            for k, run in enumerate(self.requests(ctx)):
                p = os.path.join(tmpdir, f"run{k}.nc4")
                if ctx.source_dir:
                    nat = self.native_path(ctx, d)
                    if not os.path.exists(nat):
                        raise IOError(f"{object_name(d)}: listed and absent")
                    with NC_LOCK:
                        local_subset(nat, p, run, latlon=(k == 0))
                    ctx.count_bytes(os.path.getsize(p))
                else:
                    url = subset_url(d, run, self.h, self.w, latlon=(k == 0))
                    if f10b.fetch_first([url], p,
                                        attempts=ctx.a.attempts) is None:
                        raise IOError(f"{url}: listed, and 404")
                paths["runs"].append((run, p))
                nbytes += os.path.getsize(p)
            # the clear-water mask: pace4k's BGC chlorophyll, OPTIONAL
            if ctx.source_dir:
                bp = self.bgc_path(ctx, d)
                if os.path.exists(bp):
                    ctx.count_bytes(os.path.getsize(bp))
                    paths["bgc"] = bp
            else:
                bp = os.path.join(tmpdir, "bgc.nc4")
                url = p4.opendap_url("bgc", d, ("chlor_a",))
                if f10b.fetch_first([url], bp,
                                    attempts=ctx.a.attempts) is not None:
                    paths["bgc"] = bp
                    nbytes += os.path.getsize(bp)
        except BaseException:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        return paths, nbytes, tmpdir

    def native_pixel(self, ctx, d, r, c, lo, hi):
        """The INDEPENDENT re-read: one pixel over the band range [lo, hi]."""
        if ctx.source_dir:
            with NC_LOCK:
                return local_pixel(self.native_path(ctx, d), r, c, lo, hi)
        tmp = tempfile.mkdtemp(prefix="swir_px_", dir=ctx.scratch)
        try:
            p = os.path.join(tmp, "px.nc4")
            url = pixel_url(d, r, c, lo, hi)
            if f10b.fetch_first([url], p, attempts=ctx.a.attempts) is None:
                raise IOError(f"{url}: 404 on the pixel re-read")
            with NC_LOCK:
                return read_pixel_response(p)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def read_paths(self, ctx, d, paths, nbytes):
        """-> (FLOAT16 [h, w, 8] with NaN, counts, oob), both checks run.

        The frame is built in the STORED dtype, block by block: the bounds
        are applied to the float32 values (contract rule 3, before any
        rounding) and each block is then cast once — the same single
        float32 -> float16 rounding the writer would make, so the shards are
        bit-identical to a float32 hand-over, at half the memory (0.6 GB, not
        1.2 GB; the probe additionally holds a float64 copy of the previous
        frame while this one is read)."""
        bands = self.bands(ctx)
        chan_of = {i: k for k, (_n, i, _nm) in enumerate(bands)}
        nm_of = {i: nm for _n, i, nm in bands}
        lo_b, hi_b = self.bounds()
        names = self.channel_names
        arr = np.full((self.h, self.w, len(CHANNELS)), np.nan, np.float16)
        counts = {"files_read": len(paths["runs"]), "bytes_files": int(nbytes),
                  "requests": len(paths["runs"])}
        oob = {}
        for k, (run, p) in enumerate(paths["runs"]):
            idx = run_indices(run)
            attrs = {}
            with NC_LOCK:
                for r0, a, attrs in iter_subset(
                        p, [nm_of[i] for i in idx], self.h, self.w,
                        need_grid=(k == 0)):
                    for j, i in enumerate(idx):
                        c = chan_of[i]
                        v = a[:, :, j]
                        with np.errstate(invalid="ignore"):
                            bad = np.isfinite(v) & ((v < lo_b[c])
                                                    | (v > hi_b[c]))
                        n = int(bad.sum())
                        if n:
                            v[bad] = np.nan
                            oob[names[c]] = oob.get(names[c], 0) + n
                        arr[r0:r0 + a.shape[0], :, c] = v
                    del a
            if k == 0:
                for key in ("license", "processing_version"):
                    counts.setdefault(key, {})[str(attrs.get(key))] = 1
                if attrs.get("l2_flag_names"):
                    counts["l2_flag_names"] = {str(attrs["l2_flag_names"]): 1}
                if str(attrs.get("license")) != LICENCE_URL:
                    counts["licence_attribute_differs"] = 1
        counts["valid_pixels"] = int(np.isfinite(arr[:, :, 2]).sum())
        self.check_pixel(ctx, d, arr, bands, counts)
        self.check_dark(ctx, d, arr, paths.get("bgc"), counts)
        return arr, counts, oob

    def check_pixel(self, ctx, d, arr, bands, counts):
        """Falsifier 1: an independent single-pixel re-read equals the frame
        to float16 precision; REFUSES otherwise."""
        px = pick_pixel(arr)
        if px is None:
            counts["pixel_reread_no_valid_pixel"] = 1
            return
        r, c = px
        idx = [i for _n, i, _nm in bands]
        lo, hi = min(idx), max(idx)
        vals, wl = self.native_pixel(ctx, d, r, c, lo, hi)
        want_wl = list(range(lo, hi + 1))
        if len(vals) != len(want_wl) or len(wl) != len(want_wl):
            sys.exit(f"REFUSING pace_swir: {d}: the pixel re-read returned "
                     f"{len(vals)} values for {len(want_wl)} bands")
        native = np.array([vals[i - lo] for i in idx], np.float32)
        native_wl = [int(wl[i - lo]) for i in idx]
        if native_wl != [nm for _n, _i, nm in bands]:
            sys.exit(f"REFUSING pace_swir: {d}: the pixel re-read's "
                     f"wavelengths {native_wl} are not the bands "
                     f"{[nm for _n, _i, nm in bands]}")
        stored = np.asarray(arr[r, c, :], np.float16)
        if not np.array_equal(native.astype(np.float16), stored):
            sys.exit(f"REFUSING pace_swir: {d}: pixel (row {r}, col {c}) "
                     f"re-read from the native file is {native.tolist()}, "
                     f"the frame holds {stored.tolist()} — not equal to "
                     f"float16 precision (E-092 §6)")
        counts["pixel_reread_f16_equal"] = 1

    def check_dark(self, ctx, d, arr, bgc, counts):
        """Falsifier 2: dark clear water at 2258 nm (rule in the docstring)."""
        if bgc is None:
            counts["dark_swir_no_mask"] = 1
            return
        clear = None
        with NC_LOCK:
            for _name, a, _r0, _c0, _c in p4.read_nc(bgc, ("chlor_a",),
                                                     self.h, self.w):
                clear = np.isfinite(a)
                del a
        k = self.channel_names.index(DARK_CHANNEL)
        n, p50, p99 = dark_stats(arr[:, :, k], clear)
        del clear
        verdict = judge_dark(n, p50, p99)
        if verdict == "too_few":
            counts["dark_swir_too_few"] = 1
            return
        counts["dark_swir_judged"] = 1
        counts["dark_swir_days"] = [[str(d), n, round(p50, 6), round(p99, 6)]]
        counts["max_dark_swir_p50"] = round(p50, 6)
        counts["max_dark_swir_p99"] = round(p99, 6)
        if verdict == "gross":
            sys.exit(f"REFUSING pace_swir: {d}: clear-water {DARK_CHANNEL} "
                     f"median {p50:.4f} / p99 {p99:.4f} over {n} pixels — at "
                     f"or above the GROSS limits {GROSS_P50} / {GROSS_P99}: a "
                     f"wrong band, unit or mask, not weather (E-092 §6)")
        if verdict == "soft":
            counts["dark_swir_soft_fail"] = 1
            counts["dark_swir_soft_fail_days"] = [str(d)]

    def read_day(self, ctx, d):
        paths, nbytes, tmpdir = self.fetch_day(ctx, d)
        try:
            return self.read_paths(ctx, d, paths, nbytes)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------ the contract ---
    def classify(self, ctx, d):
        """-> (True or None, reason or None, extra counts)."""
        lo, hi = self.record(ctx)
        if lo is None or d < lo:
            return None, "before_record", {}
        if d > hi:
            extra = {"nrt_after_final": 1} \
                if d in self.listing(ctx, d.year, nrt=True)[0] else {}
            return None, "after_record", extra
        if d not in self.listing(ctx, d.year)[0]:
            extra = {"nrt_only_inside_record": 1} \
                if d in self.listing(ctx, d.year, nrt=True)[0] else {}
            return None, "absent_upstream", extra
        return True, None, {}

    def fetch_frames(self, ctx, wanted):
        self._edge = self.record(ctx)[1]
        refused = set()
        jobs = []
        for (g, b, f) in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            have, why, extra = self.classify(ctx, d)
            if why == "absent_upstream":
                mk = f"{d.year:04d}-{d.month:02d}"
                if mk in self.empty_months(ctx, d.year):
                    if mk not in refused:
                        ctx.note_absent(mk, f"the final listing of {d.year} "
                                            f"has no day in {mk} — an empty "
                                            f"month is refused")
                        refused.add(mk)
                    continue
            jobs.append((g, b, f, d, have, why, extra))

        def work(job):
            g, b, f, d, have, why, extra = job
            if have is None:
                return job, None, None
            try:
                return job, self.fetch_day(ctx, d), None
            except (IOError, OSError) as e:
                return job, None, f"{type(e).__name__}: {e}"

        if jobs and any(j[4] for j in jobs):
            self.bands(ctx)          # once, before the pool, not per thread
        workers = 1 if ctx.source_dir else self.WORKERS
        t0 = time.time()
        for job, got, err in cm.ordered_map(work, jobs, workers,
                                            lookahead=workers + 1):
            g, b, f, d, have, why, extra = job
            if have is None:
                yield g, b, f, None, {"frame_missing": why, **extra}
                continue
            if err is not None:
                ctx.note_absent(str(d), err)
                continue
            paths, nbytes, tmpdir = got
            try:
                arr, counts, oob = self.read_paths(ctx, d, paths, nbytes)
            except FormatError as e:
                sys.exit(f"REFUSING pace_swir: {d}: {e}")
            except (IOError, OSError) as e:
                ctx.note_absent(str(d), f"{type(e).__name__}: {e}")
                continue
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
            if oob:
                counts["out_of_bounds"] = oob
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, arr, counts
            del arr       # the caller holds it; the next decode must not

    # ------------------------------------------------------------ smoke ----
    def smoke_sources(self, root, d_lo, d_hi, seed=20261013):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 480, 240
SMOKE_FINAL_LAST = dt.date(2024, 3, 17)     # the final edge inside the window
SMOKE_NRT_FROM = dt.date(2024, 3, 14)       # NRT overlaps, then runs past it
SMOKE_ABSENT = dt.date(2024, 3, 10)         # no final file; an NRT one exists
SMOKE_OOB = dt.date(2024, 3, 8)             # one rhos_1249 of 1.7 -> NaN
SMOKE_NO_MASK = dt.date(2024, 3, 11)        # no BGC file: dark test not run
SMOKE_SOFT = dt.date(2024, 3, 12)           # clear water at 0.015 at 2258 nm


def smoke_fields(d, h, w, gross=False):
    """A deterministic day: 'land' bright in the SWIR, 'ocean' dark, a cloud
    band of fill. -> (rhos [h, w, 122] float32 with fill, chlor_a [h, w])."""
    k = (d - dt.date(2024, 1, 1)).days
    yy = (np.arange(h, dtype=np.float64)[:, None] - h / 2) / h
    xx = (np.arange(w, dtype=np.float64)[None, :] - w / 2) / w
    land = np.hypot(yy + 0.1, xx - 0.25) < 0.18
    cloud = np.broadcast_to(np.abs(yy - 0.2 - 0.02 * np.sin(k)) < 0.04,
                            (h, w))
    rh = np.full((h, w, len(MEASURED_AXIS)), FILL, np.float32)
    wave = 0.5 + 0.5 * np.sin(7 * xx + 5 * yy + 0.3 * k)
    for name, (i, nm) in MEASURED_BANDS.items():
        ocean = 0.002 + 0.004 * wave * (1000.0 / nm)
        if name == "rhos_2258":
            ocean = 0.002 + 0.005 * wave
            if d == SMOKE_SOFT:
                ocean = ocean + 0.012
            if gross:
                ocean = ocean + 0.08
        lnd = 0.05 + 0.25 * wave * (nm / 2258.0)
        v = np.where(land, lnd, ocean).astype(np.float32)
        v[cloud] = FILL
        rh[:, :, i] = v
    # a few unused bands carry values too, so the slicing is not trivial
    for i in (55, 57, 85, 111):
        rh[:, :, i] = np.where(cloud, FILL, 0.5).astype(np.float32)
    if d == SMOKE_OOB:
        rh[h // 2, w // 3, MEASURED_BANDS["rhos_1249"][0]] = np.float32(1.7)
    chl = np.where(land | cloud, np.float32(-32767.0),
                   (0.1 + wave).astype(np.float32)).astype(np.float32)
    return rh, chl


def write_native(path, h, w, rh, day):
    import netCDF4
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("product_name", os.path.basename(path))
        ds.setncattr("title", "OCI Level-3 Standard Mapped Image (SMOKE)")
        ds.setncattr("processing_version", VERSION)
        ds.setncattr("license", LICENCE_URL)
        ds.setncattr("institution", "NASA Goddard Space Flight Center, Ocean "
                                    "Ecology Laboratory, Ocean Biology "
                                    "Processing Group")
        ds.setncattr("identifier_product_doi", DOI)
        ds.setncattr("l2_flag_names",
                     "ATMFAIL,CLDICE,HIGLINT,HILT,HISATZEN,HISOLZEN")
        ds.setncattr("time_coverage_start", f"{day}T00:00:00.000Z")
        ds.createDimension("wavelength", len(MEASURED_AXIS))
        ds.createDimension("lat", h)
        ds.createDimension("lon", w)
        wv = ds.createVariable("wavelength", "i4", ("wavelength",))
        wv[:] = np.array(MEASURED_AXIS, np.int32)
        dy, dx = 180.0 / h, 360.0 / w
        ds.createVariable("lat", "f4", ("lat",))[:] = \
            Y0 - (np.arange(h) + 0.5) * dy
        ds.createVariable("lon", "f4", ("lon",))[:] = \
            X0 + (np.arange(w) + 0.5) * dx
        v = ds.createVariable("rhos", "f4", ("lat", "lon", "wavelength"),
                              zlib=True, complevel=1, fill_value=FILL,
                              chunksizes=(16, 64, len(MEASURED_AXIS)))
        v.set_auto_maskandscale(False)
        v.valid_min = np.float32(-0.05)
        v.valid_max = np.float32(1.5)
        v[:] = rh


def cmr_json(days, nrt=False, prefixed=False):
    sn = SHORT_NRT if nrt else SHORT
    entries = []
    for i, d in enumerate(sorted(days)):
        name = object_name(d, nrt)
        e = {"title": (f"{sn}_{name}_{VERSION}" if prefixed else name),
             "id": f"G{5000 + i}-TEST", "granule_size": "3981.0",
             "time_start": f"{d}T00:00:00.000Z"}
        if prefixed:                      # 2024-25: both fields present
            e["producer_granule_id"] = name
        entries.append(e)
    return {"feed": {"id": ms.CMR, "title": "ECHO granule metadata",
                     "entry": entries}}


def make_smoke_sources(root, d_lo, d_hi, seed=20261013, gross_day=None):
    """The archive in its real layout under `root/pace_swir/` and the truth.

    `cmr_sfrefl_<Y>.json` / `cmr_sfrefl_nrt_<Y>.json` in CMR's shape (the
    final entries with the prefixed titles CMR really serves for 2024),
    native 122-band files under `opendap/L3SMI/<YYYY>/<MMDD>/`, and pace4k's
    BGC chlor_a beside them. The final edge is SMOKE_FINAL_LAST, the NRT
    collection runs SMOKE_NRT_FROM .. d_hi; SMOKE_ABSENT has only an NRT
    file; SMOKE_NO_MASK has no BGC file. `gross_day` brightens clear water at
    2258 nm past the gross limit (the refusal's test).
    Returns {(group, bin, frame): (float16 [h, w, 8] or None, reason)}.
    """
    os.environ["PACE_SWIR_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    h, w = SMOKE_H, SMOKE_W
    base = os.path.join(root, "pace_swir")
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]
    final = [d for d in days if d <= SMOKE_FINAL_LAST and d != SMOKE_ABSENT]
    nrt = [d for d in days if d >= SMOKE_NRT_FROM or d == SMOKE_ABSENT]
    fields = {}
    for d in final:
        rh, chl = smoke_fields(d, h, w, gross=(d == gross_day))
        fields[d] = rh
        dd = os.path.join(base, "opendap", "L3SMI", f"{d:%Y}", f"{d:%m%d}")
        write_native(os.path.join(dd, object_name(d)), h, w, rh, d)
        if d != SMOKE_NO_MASK:
            p4.write_nc(os.path.join(dd, p4.object_name("bgc", d)), h, w,
                        {"chlor_a": chl})
    for y in sorted({d.year for d in final}):
        with open(os.path.join(base, f"cmr_sfrefl_{y:04d}.json"), "w") as fh:
            json.dump(cmr_json([d for d in final if d.year == y],
                               prefixed=True), fh)
    for y in sorted({d.year for d in nrt}):
        with open(os.path.join(base, f"cmr_sfrefl_nrt_{y:04d}.json"),
                  "w") as fh:
            json.dump(cmr_json([d for d in nrt if d.year == y], nrt=True),
                      fh)

    ad = PaceSwirAdapter()
    lo, hi = min(final), max(final)
    t_lo = f10b.seconds_since_epoch(lo)
    t_hi = f10b.seconds_since_epoch(hi) + 86399
    lo_b, hi_b = ad.bounds()
    order = [MEASURED_BANDS[n][0] for n, _t in TARGETS]
    truth = {}
    for b in sh.bins_overlapping(t_lo, t_hi):
        for f in range(ad.frames_per_bin):
            d = sh.frame_day(b, f, ad.frame_seconds)
            if d < lo:
                truth[(ad.store, b, f)] = (None, "before_record")
            elif d > hi:
                truth[(ad.store, b, f)] = (None, "after_record")
            elif d not in fields:
                truth[(ad.store, b, f)] = (None, "absent_upstream")
            else:
                a = fields[d][:, :, order].astype(np.float32)
                a[a == FILL] = np.nan
                with np.errstate(invalid="ignore"):
                    a[(a < lo_b) | (a > hi_b)] = np.nan
                truth[(ad.store, b, f)] = (a.astype(np.float16), None)
    return truth


ADAPTER = PaceSwirAdapter
