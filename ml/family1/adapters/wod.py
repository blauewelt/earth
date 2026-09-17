"""World Ocean Database casts on the Argo pressures — one row per cast (family 1.gf).

PLAIN ENGLISH. The World Ocean Database is NOAA's archive of every ocean
profile it could collect since 1772: bottle casts from ships, CTDs,
mechanical and expendable bathythermographs, moorings, drifting buoys,
undulating towed recorders, instruments on seals, gliders. This adapter
turns it into a tier-P store — one row per cast, eight quantities at the 16
pressures the Argo stores use — so a model can see the pre-Argo interior
ocean (and the non-Argo one since). Argo floats (WOD's PFL) are family 8 and
are left out.

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (NOAA NCEI, no account):
  https://www.ncei.noaa.gov/data/oceans/ncei/wod/                 (listing)
  https://www.ncei.noaa.gov/data/oceans/ncei/wod/<YYYY>/wod_<inst>_<YYYY>.nc
  * `https://www.ncei.noaa.gov/data/oceans/woa/WOD/YEARLY/` (the path the
    plan named) is WOD's NATIVE ASCII format (`<INST>/OBS/<INST>O<YYYY>.gz`),
    not netCDF. The ragged netCDF lives under `data/oceans/ncei/wod/`.
  * The root listing (Apache, 128 directories) holds `1800/` and `1900/` ..
    `2026/`. `1800/` is a BUNDLE: one file, `wod_osd_1800.nc` (286 M), with
    every cast before 1900; each later directory holds one file per
    instrument for its year (2005: apb 684M, ctd 583M, drb 30M, gld 177M,
    mrb 121M, osd 146M, pfl 313M, uor 27M, xbt 440M — no MBT; 1965: mbt,
    osd, xbt). Sizes are HUMAN-READABLE ("146M"), so byte totals from the
    listing are approximate; the probe counts exact bytes.
  * Files are NETCDF4 (HDF5), CF "contiguous ragged array" profiles
    (featureType Profile): dimension `casts`; per cast `lat`, `lon`, `time`
    (days since 1770-01-01, fill -1e10), `wod_unique_cast`,
    `WOD_cruise_identifier` and `Platform` (char arrays), `z_row_size`; per
    variable V a sample dimension `V_obs` with `V` (float32, fill -1e10),
    `V_WODflag` (per value: 0 accepted, 1-9 range/inversion/gradient/
    anomaly failures), `V_row_size` and `V_WODprofileflag` (per cast, 0
    accepted). Depth `z` (m, positive down) with `z_WODflag` (0 accepted,
    1 duplicate/inversion, 2 density inversion). MEASURED on wod_uor_2005
    (1,331 casts) and wod_osd_2005 (15,020 casts, 23 variables): EVERY
    variable's row size equals `z_row_size` wherever it is non-zero, i.e.
    all variables of a cast share its depth levels — the parser asserts
    this per cast and counts a mismatch (`var_rowsize_mismatch`). Units
    measured: Temperature degree_C, Oxygen/Nitrate/Phosphate/Silicate
    umol/kg, Chlorophyll ugram/l, Salinity and pH carry no units attribute.
    940 of 15,020 OSD casts have a blank cruise identifier.

WHAT A ROW IS.
  time_s    the cast's `time`, rounded to the second, int64 seconds since
            1982-01-01 (schema 3: the record starts in 1772). A cast with
            fill time is dropped (`casts_no_time`); a cast whose time is not
            in the file's year is dropped (`casts_other_year`) — except in
            the 1800 bundle, which is split by year (read once per process
            and held per year, `bundle_years`).
  lat, lon  the cast's `lat`, `lon` (lon wrapped to [-180, 180)).
  platform  platform_hash("<INST>:<WOD_cruise_identifier>"), e.g.
            "OSD:AU003587" — one cruise of one instrument class. FALLBACKS,
            counted: blank cruise id -> "<INST>:platform:<Platform>"
            (`platform_from_name`); both blank -> "<INST>:cast:<wod_unique_
            cast>" (`platform_from_cast`, one platform per cast).
  values    C = 128 = T (degC), S (PSU), O2 (umol/kg), NO3, PO4, SiO4
            (umol/kg), pH, CHL (mg/m3 = ugram/l) at the 16 pressures of
            `build_family8_argo.LEVELS` (imported, never restated),
            VARIABLE-MAJOR, by `build_family8_argo.interp_levels` (the gliders
            adapter's helper). Depth is converted to pressure with Saunders
            (1981): p = [(1-c1) - sqrt((1-c1)^2 - 8.84e-6 z)] / 4.42e-6,
            c1 = (5.92 + 5.25 sin^2(lat)) e-3 — the cast's own latitude.
            NaN where the cast lacks the variable or the level.
  qc        a bitmask. Low bits — which WOD flags were SEEN (and removed):
            1 a value with V_WODflag != 0, 2 a variable with
            V_WODprofileflag != 0 (the whole variable of that cast dropped),
            4 a depth with z_WODflag != 0. High nibble — the instrument:
            1 OSD, 2 CTD, 3 MBT, 4 XBT, 5 MRB, 6 DRB, 7 UOR, 8 APB, 9 GLD.
            Only WOD flag 0 values are interpolated.
A cast with none of the 128 values is dropped (`casts_no_level`).
Instruments other than the nine above are not read: PFL is family 8
(`files_excluded_pfl`), anything else is listed and counted
(`files_unknown_instrument`).

THE PROBE, 2005-03, MEASURED 2026-09-17 (ml/family1/probes/
wod_2005-03.json): the eight non-PFL 2005 files (osd ctd xbt mrb drb uor apb
gld) downloaded and read whole — 2,316,297,840 bytes in 143 s (peak RSS
270 MB) — 483,864 casts in the year, 40,541 March casts kept from 561
platforms (APB 32,101, MRB 2,706, XBT 2,310, CTD 1,880, OSD 798, GLD 746);
3,353 March casts with no usable level, 3,827 variable-profiles removed by
their profile flag, 116,194 flagged values removed, 3,091 casts with neither
cruise id nor platform name (their own platform), 15 NO3 values out of
bounds, 0 casts of another year. NaN fractions at 10 / 100 / 500 / 1100
dbar: T 0.02 / 0.18 / 0.69 / 0.99, S 0.89 / 0.96 / 0.98 / 0.99, O2 0.98 /
0.99 / 1.0 / 1.0, nutrients, pH and CHL ≥ 0.99 (the month is dominated by
seal tags that carry T only).
WHOLE ARCHIVE, from all 128 directory listings (fetched 2026-09-17): 167.06 GB
(human-readable sizes), 120.76 GB without PFL (OSD 26.06, GLD 39.81, XBT
18.64, CTD 17.99, MBT 8.83, APB 4.09, MRB 2.77, DRB 1.68, UOR 0.90) — the
note's 167.5 / 121 GB. At the note's 18.63 M casts less 2.75 M PFL and the
probe's 92 % keep rate: ≈ 14.7 M rows, ≈ 4.2 GB stored at 287 B a row (the
note's 2 GB was for C = 48); ≈ 2.1 h of download and parse at the probe's
16 MB/s.

MEMORY. One instrument-year file at a time is downloaded to the work
directory (size-verified), read variable by variable in blocks of at most
BLOCK_OBS = 2,000,000 samples (≈ 10 MB a variable), interpolated, packed
(≈ 580 bytes a cast) and yielded per block, then deleted. The largest 2005
file is 684 MB on disk; the resident peak is the block, not the file. The
1800 bundle's rows are held per year in memory (packed, ≈ 580 B a cast).
"""
import datetime as dt
import os
import re
import sys
import time

import numpy as np

import build_family10_stores as f10b
import build_family8_argo as f8
from family1.adapters import _common as cm

ROOT = "https://www.ncei.noaa.gov/data/oceans/ncei/wod/"
BUNDLE_DIR = "1800"
BUNDLE_BEFORE = 1900
BLOCK_OBS = 2_000_000
BLOCK_CASTS = 20_000

INSTRUMENTS = ("OSD", "CTD", "MBT", "XBT", "MRB", "DRB", "UOR", "APB", "GLD")
INST_CODE = {k: i + 1 for i, k in enumerate(INSTRUMENTS)}
EXCLUDED = ("PFL",)

# (channel prefix, WOD variable, unit, lo, hi, accepted units attribute)
QUANTITIES = (
    ("T", "Temperature", "degC", -2.5, 40.0, {"degree_c", "degrees_c"}),
    ("S", "Salinity", "PSU", 0.0, 42.0, {"", "psu", "1e-3", "1"}),
    ("O2", "Oxygen", "umol/kg", 0.0, 600.0, {"umol/kg"}),
    ("NO3", "Nitrate", "umol/kg", 0.0, 60.0, {"umol/kg"}),
    ("PO4", "Phosphate", "umol/kg", 0.0, 5.0, {"umol/kg"}),
    ("SIO4", "Silicate", "umol/kg", 0.0, 300.0, {"umol/kg"}),
    ("PH", "pH", "1", 6.5, 9.0, {"", "1"}),
    ("CHL", "Chlorophyll", "mg/m3", 0.0, 100.0, {"ugram/l", "mg/m3"}),
)
LEVELS = f8.LEVELS_ARR
NLEV = len(LEVELS)
CHANNELS = tuple((f"{q}_{int(p)}", u, lo, hi)
                 for q, _, u, lo, hi, _ in QUANTITIES for p in LEVELS)

DIR_ROW = re.compile(r'<a href="(\d{4})/">')
FILE_ROW = re.compile(r'<a href="(wod_([a-z]{3})_(\d{4})\.nc)">[^<]*</a>'
                      r'</td><td[^>]*>[^<]*</td><td[^>]*>\s*([0-9.]+[KMG]?)'
                      r'\s*</td>')
FILE_NAME = re.compile(r'<a href="(wod_[^"]*\.nc)"')
TIME_UNITS = re.compile(r"^\s*days since (\d{4})-(\d{2})-(\d{2})")
FILL = -1e9                       # WOD's fill is -1e10: anything below is fill


class FormatError(ValueError):
    """A listing or a file that is not the layout verified above."""


def approx_bytes(s):
    s = s.strip()
    mult = {"K": 1e3, "M": 1e6, "G": 1e9}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(float(s))


def parse_root(html):
    dirs = sorted(set(DIR_ROW.findall(html)))
    if not dirs:
        raise FormatError("the WOD root lists no <YYYY>/ directory — an "
                          "empty listing is a broken listing")
    return dirs


def parse_year_listing(html, ydir):
    """[(name, inst, bytes_approx)] of one year directory."""
    names = FILE_NAME.findall(html)
    rows = FILE_ROW.findall(html)
    if not names:
        raise FormatError(f"WOD {ydir}/ lists no wod_*.nc file — an empty "
                          f"listing is a broken listing")
    if len(rows) != len(names):
        raise FormatError(f"WOD {ydir}/ names {len(names)} files but the "
                          f"size column parsed for {len(rows)}")
    out = []
    for n, inst, y, sz in rows:
        if y != ydir:
            raise FormatError(f"{n} is listed under {ydir}/")
        out.append((n, inst.upper(), approx_bytes(sz)))
    return out


def saunders_pressure(z, lat):
    """Depth (m, positive down) -> pressure (dbar), Saunders (1981)."""
    c1 = (5.92 + 5.25 * np.sin(np.deg2rad(lat)) ** 2) * 1e-3
    return ((1 - c1) - np.sqrt((1 - c1) ** 2 - 8.84e-6 * z)) / 4.42e-6


def _chars(a):
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "S" and a.ndim == 2:
        a = a.view(f"S{a.shape[1]}").reshape(-1)
    return [x.decode("latin-1").strip() for x in a]


def _unit(v):
    u = v.getncattr("units") if "units" in v.ncattrs() else ""
    return str(u).strip().lower()


def _add(c, k, n):
    if n:
        c[k] = c.get(k, 0) + int(n)


class WODFile:
    """One ragged WOD netCDF, read block by block."""

    def __init__(self, path, inst):
        import netCDF4
        self.ds = netCDF4.Dataset(path)
        self.ds.set_auto_mask(False)
        self.ds.set_auto_scale(False)
        self.inst = inst
        V = self.ds.variables
        for k in ("lat", "lon", "time", "z", "z_row_size", "z_WODflag",
                  "wod_unique_cast"):
            if k not in V:
                raise FormatError(f"{path}: no `{k}` variable")
        m = TIME_UNITS.match(V["time"].getncattr("units"))
        if not m:
            raise FormatError(f"{path}: time units "
                              f"{V['time'].getncattr('units')!r}")
        self.t0 = int(cm.days_from_civil(int(m.group(1)), int(m.group(2)),
                                         int(m.group(3))))
        self.n = len(self.ds.dimensions["casts"])
        self.zrs = np.asarray(V["z_row_size"][:], np.int64)
        self.zrs[self.zrs < 0] = 0
        self.zoff = np.concatenate([[0], np.cumsum(self.zrs)])

    def close(self):
        self.ds.close()

    def cast_meta(self, counts):
        V = self.ds.variables
        lat = np.asarray(V["lat"][:], np.float64)
        lon = np.asarray(V["lon"][:], np.float64)
        tim = np.asarray(V["time"][:], np.float64)
        uid = np.asarray(V["wod_unique_cast"][:], np.int64)
        cru = _chars(V["WOD_cruise_identifier"][:]) \
            if "WOD_cruise_identifier" in V else [""] * self.n
        plat = _chars(V["Platform"][:]) if "Platform" in V else [""] * self.n
        ph = np.zeros(self.n, np.int64)
        cache = {}
        for i in range(self.n):
            if cru[i]:
                key = f"{self.inst}:{cru[i]}"
            elif plat[i]:
                key = f"{self.inst}:platform:{plat[i]}"
                _add(counts, "platform_from_name", 1)
            else:
                key = f"{self.inst}:cast:{uid[i]}"
                _add(counts, "platform_from_cast", 1)
            h = cache.get(key)
            if h is None:
                h = cache[key] = f10b.platform_hash(key)
            ph[i] = h
        bad_t = ~np.isfinite(tim) | (tim < FILL)
        t = np.where(bad_t, 0.0, tim)
        t_s = np.rint((t + self.t0) * 86400.0).astype(np.int64)
        return lat, lon, t_s, bad_t, ph

    def variables(self, counts):
        """{prefix: (var, row offsets, row size)} for the variables present
        with an accepted unit."""
        V = self.ds.variables
        out = {}
        for q, name, _u, _lo, _hi, units in QUANTITIES:
            if name not in V:
                continue
            u = _unit(V[name])
            if u not in units:
                bad = counts.setdefault("units_unrecognized", {})
                bad[f"{name} {u!r}"] = bad.get(f"{name} {u!r}", 0) + 1
                continue
            rs = np.asarray(V[f"{name}_row_size"][:], np.int64)
            rs[rs < 0] = 0
            out[q] = (name, np.concatenate([[0], np.cumsum(rs)]), rs)
        return out

    def blocks(self):
        """Cast index ranges [a, b) with at most BLOCK_OBS depth samples."""
        a = 0
        while a < self.n:
            b = min(self.n, a + BLOCK_CASTS)
            while b > a + 1 and self.zoff[b] - self.zoff[a] > BLOCK_OBS:
                b = a + max(1, (b - a) // 2)
            yield a, b
            a = b


def _bitmask_of(q, inst_code):
    return (inst_code << 4) | q


# ================================================================ adapter ==
class WODAdapter(f10b.SourceAdapter):
    store = "wod"
    title = ("World Ocean Database (NOAA NCEI) casts, all instruments but "
             "Argo floats, at the 16 Argo pressures, one row per cast")
    family = "1gf"
    distribution = "public"
    licence = {"name": "World Ocean Database — US government work (NOAA "
                       "NCEI), no restriction (CC0 per the note)",
               "redistribution": "yes", "derived_works": "free",
               "attribution": "Boyer et al. (2024), World Ocean Database "
                              "2023, NOAA NCEI"}
    time_dtype = "int64"
    platform_meta = False
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))
    per_year = True
    first_year = 1772
    qc_policy = (
        "Only WOD flag 0 values are used: a sample needs V_WODflag 0 and "
        "z_WODflag 0, and a variable of a cast needs V_WODprofileflag 0. qc "
        "is a bitmask: 1 a value flag seen, 2 a profile flag seen, 4 a "
        "depth flag seen; the high nibble is the instrument (1 OSD .. 9 "
        "GLD). Depth -> pressure by Saunders (1981); interpolation onto the "
        "16 RG pressures by build_family8_argo.interp_levels. Out-of-bounds "
        "level values -> NaN, counted; a cast with no level value dropped.")
    sources = (ROOT + "<YYYY>/wod_<inst>_<YYYY>.nc", ROOT)
    verified = (
        "2026-09-17 from the sandbox: the root listing (128 directories: "
        "1800 bundle + 1900..2026), the 2005, 1965, 1800 and 2026 "
        "directories, wod_uor_2005.nc (28.8 MB, 1,331 casts) and "
        "wod_osd_2005.nc (146 MB, 15,020 casts, 23 variables): layout, "
        "units, row-size alignment, flags, time units")
    notes = (
        "woa/WOD/YEARLY is the ASCII product; the netCDF is under "
        "data/oceans/ncei/wod. PFL (Argo) excluded — family 8. Directory "
        "1800 is one OSD file for every year before 1900.")
    smoke_window = ("1899-12-30", "1900-01-02")
    smoke_probe_month = "1900-01"
    fetch_month_scope = "year"

    def __init__(self):
        self._root = None
        self._years = {}
        self._bundle = None           # {year: [rows]}, counts
        self._bundle_counted = False

    # ------------------------------------------------------------ listing --
    def _local(self, ctx, *p):
        return os.path.join(ctx.source_dir, "wod", *p)

    def _page(self, ctx, rel):
        if ctx.source_dir:
            p = self._local(ctx, rel, "index.html") if rel else \
                self._local(ctx, "index.html")
            raw = open(p, "rb").read()
            ctx.count_bytes(len(raw))
            return raw.decode("latin-1")
        url = ROOT + (rel + "/" if rel else "")
        raw, why = cm.get_bytes(url, attempts=ctx.a.attempts)
        if raw is None:
            sys.exit(f"{url} answered 404 ({why}) — the WOD archive moved")
        return raw.decode("latin-1")

    def root(self, ctx):
        if self._root is None:
            try:
                self._root = parse_root(self._page(ctx, ""))
            except FormatError as e:
                sys.exit(f"REFUSING wod: {e}")
        return self._root

    @staticmethod
    def dir_of(year):
        return BUNDLE_DIR if year < BUNDLE_BEFORE else str(year)

    def year_files(self, ctx, ydir):
        if ydir not in self._years:
            if ydir not in self.root(ctx):
                self._years[ydir] = None
            else:
                try:
                    self._years[ydir] = parse_year_listing(
                        self._page(ctx, ydir), ydir)
                except FormatError as e:
                    sys.exit(f"REFUSING wod: {e}")
        return self._years[ydir]

    def _split(self, files):
        keep, pfl, other = [], [], []
        for n, inst, sz in files:
            if inst in EXCLUDED:
                pfl.append(n)
            elif inst in INST_CODE:
                keep.append((n, inst, sz))
            else:
                other.append(n)
        keep.sort(key=lambda x: INST_CODE[x[1]])
        return keep, pfl, other

    # ------------------------------------------------------------ reading --
    def _fetch_file(self, ctx, ydir, name):
        if ctx.source_dir:
            p = self._local(ctx, ydir, name)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        dest = os.path.join(ctx.scratch, "wod", name)
        url = f"{ROOT}{ydir}/{name}"
        err = None
        for i in range(max(1, ctx.a.attempts)):
            try:
                f10b.http_to_file(url, dest)
                return dest, True
            except f10b._NotFound:
                raise
            except (IOError, *cm.RETRY_ERRORS) as e:
                err = e
                time.sleep(3.0 * (2 ** i))
        raise IOError(f"{url}: {err}")

    def _file_rows(self, ctx, path, inst, year_of, t_lo, t_hi, counts):
        """Yield packed rows per block; `year_of` None = keep any year,
        else drop casts not in that year."""
        f = WODFile(path, inst)
        try:
            lat, lon, t_s, bad_t, ph = f.cast_meta(counts)
            _add(counts, "casts", f.n)
            _add(counts, "casts_no_time", bad_t.sum())
            ok = ~bad_t & np.isfinite(lat) & np.isfinite(lon) & \
                (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
            _add(counts, "casts_bad_position", (~bad_t & ~ok).sum())
            if year_of is not None:
                y0 = int(cm.days_from_civil(year_of, 1, 1)) * 86400
                y1 = int(cm.days_from_civil(year_of + 1, 1, 1)) * 86400
                other = ok & ((t_s < y0) | (t_s >= y1))
                _add(counts, "casts_other_year", other.sum())
                ok &= ~other
            inside = ok & (t_s >= t_lo) & (t_s <= t_hi)
            _add(counts, "casts_outside_window", (ok & ~inside).sum())
            ok = inside
            vars_ = f.variables(counts)
            code = INST_CODE[inst]
            V = f.ds.variables
            for a, b in f.blocks():
                sel = np.flatnonzero(ok[a:b]) + a
                if not sel.size:
                    continue
                n = sel.size
                vals = np.full((n, len(QUANTITIES) * NLEV), np.nan)
                qc = np.zeros(n, np.int64)
                z0, z1 = int(f.zoff[a]), int(f.zoff[b])
                z = np.asarray(V["z"][z0:z1], np.float64)
                zfl = np.asarray(V["z_WODflag"][z0:z1], np.int64)
                zgood = np.isfinite(z) & (z > FILL) & (z >= 0) & (zfl == 0)
                zseen = (zfl != 0)
                pres = np.full(z.shape, np.nan)
                # pressure per sample with its cast's latitude
                lat_s = np.repeat(lat[a:b], f.zrs[a:b])
                pres[zgood] = saunders_pressure(z[zgood], lat_s[zgood])
                for qi, (q, name, *_rest) in enumerate(QUANTITIES):
                    if q not in vars_:
                        continue
                    _vn, voff, vrs = vars_[q]
                    v0, v1 = int(voff[a]), int(voff[b])
                    if v1 == v0:
                        continue
                    val = np.asarray(V[name][v0:v1], np.float64)
                    fl = np.asarray(V[f"{name}_WODflag"][v0:v1], np.int64)
                    pf = np.asarray(V[f"{name}_WODprofileflag"][a:b],
                                    np.int64) \
                        if f"{name}_WODprofileflag" in V else \
                        np.zeros(b - a, np.int64)
                    for j, ci in enumerate(sel):
                        k = int(vrs[ci])
                        if k == 0:
                            continue
                        if k != f.zrs[ci]:
                            _add(counts, "var_rowsize_mismatch", 1)
                            continue
                        if pf[ci - a] != 0:
                            qc[j] |= 2
                            _add(counts, "profiles_flag_removed", 1)
                            continue
                        s0 = int(voff[ci]) - v0
                        zs = int(f.zoff[ci]) - z0
                        vv = val[s0:s0 + k]
                        ff = fl[s0:s0 + k]
                        good = zgood[zs:zs + k] & (ff == 0) & \
                            np.isfinite(vv) & (vv > FILL)
                        nf = int(((ff != 0) & (vv > FILL)).sum())
                        if nf:
                            qc[j] |= 1
                            _add(counts, "values_flag_removed", nf)
                        if zseen[zs:zs + k].any():
                            qc[j] |= 4
                        if not good.any():
                            continue
                        ps, vs = f8._sorted_unique(pres[zs:zs + k][good],
                                                   vv[good])
                        vals[j, qi * NLEV:(qi + 1) * NLEV] = \
                            f8.interp_levels(ps, vs)
                    del val, fl
                oob = self.mask_bounds(vals)
                if oob:
                    f10b._merge_counts(counts, {"out_of_bounds": oob})
                alive = np.isfinite(vals).any(axis=1)
                _add(counts, "casts_no_level", (~alive).sum())
                idx = sel[alive]
                if not idx.size:
                    continue
                q8 = ((code << 4) | qc[alive]).astype(np.uint8)
                qd = counts.setdefault("qc", {})
                for u, k in zip(*np.unique(q8, return_counts=True)):
                    qd[str(int(u))] = qd.get(str(int(u)), 0) + int(k)
                _add(counts, "casts_kept", idx.size)
                inst_c = counts.setdefault("casts_kept_by_instrument", {})
                inst_c[inst] = inst_c.get(inst, 0) + int(idx.size)
                yield self.pack(t_s[idx], lat[idx], lon[idx], vals[alive],
                                ph[idx], q8)
        finally:
            f.close()

    def _read_file(self, ctx, ydir, name, inst, year_of, t_lo, t_hi,
                   unit, counts):
        try:
            path, tmp = self._fetch_file(ctx, ydir, name)
        except f10b._NotFound:
            ctx.note_absent(unit, f"{ydir}/{name}: listed, and 404")
            return
        except IOError as e:
            ctx.note_absent(unit, f"{ydir}/{name}: {e}")
            return
        try:
            yield from self._file_rows(ctx, path, inst, year_of, t_lo, t_hi,
                                       counts)
            _add(counts, "files", 1)
        except OSError as e:
            ctx.note_absent(unit, f"{ydir}/{name}: not a readable netCDF "
                                  f"({e}) — a truncated download?")
        finally:
            if tmp and os.path.exists(path):
                os.remove(path)

    def _bundle_rows(self, ctx):
        """The 1800 bundle, read once, split by year."""
        if self._bundle is None:
            files = self.year_files(ctx, BUNDLE_DIR) or []
            keep, pfl, other = self._split(files)
            counts = {"files_listed": len(files),
                      "files_excluded_pfl": len(pfl),
                      "files_unknown_instrument": other}
            per = {}
            for name, inst, _sz in keep:
                for rows in self._read_file(ctx, BUNDLE_DIR, name, inst,
                                            None, ctx.t_lo, ctx.t_hi,
                                            BUNDLE_DIR, counts):
                    t = np.asarray(rows["time_s"], np.int64)
                    yrs = year_of_seconds(t)
                    for y in np.unique(yrs):
                        m = yrs == y
                        per.setdefault(int(y), []).append(
                            {k: v[m] for k, v in rows.items()})
            counts["bundle_years"] = len(per)
            self._bundle = (per, counts)
        return self._bundle

    def _year(self, ctx, year, t_lo, t_hi, label):
        ydir = self.dir_of(year)
        if ydir == BUNDLE_DIR:
            per, bc = self._bundle_rows(ctx)
            counts = {"bundle": BUNDLE_DIR}
            if not self._bundle_counted:
                # the bundle's own counts go to the first year that reads it
                self._bundle_counted = True
                f10b._merge_counts(counts, dict(bc))
            for rows in per.get(year, []):
                t = np.asarray(rows["time_s"], np.int64)
                keep = (t >= t_lo) & (t <= t_hi)
                yield label, {k: v[keep] for k, v in rows.items()}, None
            yield label, None, counts
            return
        files = self.year_files(ctx, ydir)
        if files is None:
            yield label, None, {"year_not_listed": 1}
            return
        keep, pfl, other = self._split(files)
        counts = {"files_listed": len(files), "files_excluded_pfl": len(pfl),
                  "files_unknown_instrument": other,
                  "bytes_listed_approx": sum(s for _, _, s in keep)}
        for name, inst, _sz in keep:
            for rows in self._read_file(ctx, ydir, name, inst, year, t_lo,
                                        t_hi, str(year), counts):
                yield label, rows, None
        yield label, None, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        dirs = self.root(ctx)
        want = sorted({self.dir_of(y) for y in ctx.years} & set(dirs))
        per = {}
        first = None
        for d in want:
            files = self.year_files(ctx, d)
            keep, pfl, other = self._split(files)
            per[d] = {"files": [n for n, _, _ in keep],
                      "bytes_approx": sum(s for _, _, s in keep),
                      "excluded_pfl": pfl, "unknown_instrument": other}
        if want:
            d = want[0]
            keep, _, _ = self._split(self.year_files(ctx, d))
            if keep:
                name, inst, _ = min(keep, key=lambda x: x[2])
                path, tmp = self._fetch_file(ctx, d, name)
                try:
                    f = WODFile(path, inst)
                    c = {}
                    vs = f.variables(c)
                    first = {"file": f"{d}/{name}", "casts": f.n,
                             "variables": sorted(vs),
                             "units_unrecognized": c.get(
                                 "units_unrecognized", {})}
                    f.close()
                finally:
                    if tmp:
                        os.remove(path)
        return {"dataset": "World Ocean Database (ragged netCDF)",
                "url": ROOT,
                "listing": {"directories": len(dirs), "first": dirs[0],
                            "last": dirs[-1]},
                "directories": per,
                "years_not_listed": [y for y in ctx.years
                                     if self.dir_of(y) not in dirs],
                "first_record": first}

    def fetch_year(self, ctx, year):
        t0 = time.time()
        agg = {}
        for label, rows, c in self._year(ctx, year, ctx.t_lo, ctx.t_hi,
                                         str(year)):
            if c:
                f10b._merge_counts(agg, c)
            if rows is not None:
                yield label, rows, None
        agg["fetch_seconds"] = round(time.time() - t0, 1)
        yield str(year), None, agg

    def fetch_month(self, ctx, year, month):
        """Every instrument file of the year is read (a file is a year);
        only the month's casts are kept, and the counts describe the year's
        files within the month window."""
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        agg = {}
        label = f"{year}-{month:02d}"
        for _l, rows, c in self._year(ctx, year, lo, hi, label):
            if c:
                f10b._merge_counts(agg, c)
            if rows is not None:
                yield label, rows, None
        yield label, None, agg

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


def year_of_seconds(t):
    """int64 seconds since 1982 -> calendar year (vectorised)."""
    days = np.floor_divide(np.asarray(t, np.int64), 86400)
    # civil_from_days (Hinnant), shifted from the 1982 epoch
    z = days + cm.EPOCH_DAYS_1970 + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = np.where(mp < 10, mp + 3, mp - 9)
    return np.where(m <= 2, y + 1, y)


# ================================================================== smoke ==
SMOKE_CASTS = (
    # inst, cruise, platform name, lat, lon, variables present
    ("OSD", "US000001", "SHIP A", 30.0, -40.0,
     ("Temperature", "Salinity", "Oxygen", "Nitrate", "Phosphate",
      "Silicate", "pH", "Chlorophyll")),
    ("OSD", "", "SHIP B", -20.0, 170.0, ("Temperature", "Salinity")),
    ("CTD", "JP000002", "", 10.0, 200.0, ("Temperature", "Salinity",
                                          "Oxygen")),
    ("XBT", "", "", 45.0, -30.0, ("Temperature",)),
)
# every 10 m to 1,500 m: all levels to 1500 dbar filled, 1700/1900 empty
Z_LEVELS = np.arange(0.0, 1501.0, 10.0)


def _truth_levels(z, lat, vals, good):
    p = saunders_pressure(z, lat)
    ps, vs = f8._sorted_unique(p[good], vals[good])
    return f8.interp_levels(ps, vs)


def _profile(name, z, rng):
    base = {"Temperature": 20 - z / 120.0, "Salinity": 35 + z / 5000.0,
            "Oxygen": 250 - z / 20.0, "Nitrate": 5 + z / 100.0,
            "Phosphate": 0.3 + z / 1500.0, "Silicate": 2 + z / 20.0,
            "pH": 8.1 - z / 10000.0, "Chlorophyll": np.exp(-z / 50.0)}[name]
    noise = rng.normal(0, 0.01, z.size)
    if name == "Chlorophyll":
        noise = np.abs(noise)
    return (base + noise).astype(np.float32)


def write_wod_nc(path, casts):
    """casts: list of dicts {inst, cruise, platform, lat, lon, days, uid,
    z, zflag, vars: {name: (values, flags, profile_flag)}} -> a WOD-layout
    NETCDF4 file."""
    import netCDF4
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    n = len(casts)
    ds.createDimension("casts", n)
    ds.createDimension("strnlen", 170)
    ds.createDimension("strnlensmall", 40)
    zz = np.concatenate([c["z"] for c in casts])
    ds.createDimension("z_obs", zz.size)

    def chars(name, dim, vals, width):
        v = ds.createVariable(name, "S1", ("casts", dim))
        a = np.zeros((n, width), "S1")
        for i, s in enumerate(vals):
            b = s.encode()[:width]
            a[i, :len(b)] = np.frombuffer(b, "S1")
        v[:] = a

    chars("country", "strnlensmall", ["US"] * n, 40)
    chars("WOD_cruise_identifier", "strnlensmall",
          [c["cruise"] for c in casts], 40)
    v = ds.createVariable("wod_unique_cast", "i4", ("casts",))
    v[:] = [c["uid"] for c in casts]
    for k, key in (("lat", "lat"), ("lon", "lon")):
        v = ds.createVariable(k, "f4", ("casts",))
        v.units = "degrees_north" if k == "lat" else "degrees_east"
        v[:] = [c[key] for c in casts]
    v = ds.createVariable("time", "f8", ("casts",), fill_value=-1e10)
    v.units = "days since 1770-01-01 00:00:00 UTC"
    v[:] = [c["days"] for c in casts]
    chars("Platform", "strnlen", [c["platform"] for c in casts], 170)
    v = ds.createVariable("z", "f4", ("z_obs",))
    v.units = "m"
    v[:] = zz
    v = ds.createVariable("z_WODflag", "i1", ("z_obs",))
    v[:] = np.concatenate([c["zflag"] for c in casts])
    v = ds.createVariable("z_row_size", "i4", ("casts",), fill_value=0)
    v[:] = [c["z"].size for c in casts]
    for _q, name, *_r in QUANTITIES:
        have = [c for c in casts if name in c["vars"]]
        if not have:
            continue
        obs = np.concatenate([c["vars"][name][0] if name in c["vars"]
                              else np.zeros(0, np.float32) for c in casts])
        fl = np.concatenate([c["vars"][name][1] if name in c["vars"]
                             else np.zeros(0, np.int8) for c in casts])
        ds.createDimension(f"{name}_obs", obs.size)
        v = ds.createVariable(name, "f4", (f"{name}_obs",),
                              fill_value=np.float32(-1e10))
        unit = {"Temperature": "degree_C", "Oxygen": "umol/kg",
                "Nitrate": "umol/kg", "Phosphate": "umol/kg",
                "Silicate": "umol/kg", "Chlorophyll": "ugram/l"}.get(name)
        if unit:
            v.units = unit
        v[:] = obs
        v = ds.createVariable(f"{name}_WODflag", "i1", (f"{name}_obs",))
        v[:] = fl
        v = ds.createVariable(f"{name}_row_size", "i4", ("casts",),
                              fill_value=0)
        v[:] = [c["vars"][name][0].size if name in c["vars"] else 0
                for c in casts]
        v = ds.createVariable(f"{name}_WODprofileflag", "i1", ("casts",))
        v[:] = [c["vars"][name][2] if name in c["vars"] else 0
                for c in casts]
    ds.close()


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """WOD in its real layout: the root listing, the 1800 bundle (casts of
    1898 and 1899, one of them with fill time), the 1900 directory with OSD,
    CTD, XBT and a PFL file that must not be read.

    Hostile on purpose: a blank cruise id (platform-name fallback), both
    blank (cast-id fallback), a flagged value, a flagged depth, a variable
    with a profile flag, a value out of bounds, a cast with no usable value,
    a cast of another year in a year file, and casts outside the window.
    """
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "wod")
    t_epoch = int(cm.days_from_civil(1770, 1, 1))
    w_lo = f10b.seconds_since_epoch(d_lo)
    w_hi = f10b.seconds_since_epoch(d_hi) + 86399
    files = {}                      # (ydir, inst) -> [cast]
    truth = []
    uid = [1000]

    def cast(inst, cruise, pname, la, lo, when, names, special=None):
        uid[0] += 1
        z = Z_LEVELS.copy()
        zflag = np.zeros(z.size, np.int8)
        days = (f10b.seconds_since_epoch(when) / 86400.0) - t_epoch
        vs = {}
        truth_v = np.full((len(QUANTITIES), NLEV), np.nan)
        qc = 0
        good_z = np.ones(z.size, bool)
        if special == "zflag":
            zflag[3] = 1
            good_z[3] = False
            qc |= 4
        for qi, (_q, name, *_r) in enumerate(QUANTITIES):
            if name not in names:
                continue
            val = _profile(name, z, rng)
            fl = np.zeros(z.size, np.int8)
            pflag = 0
            good = good_z.copy()
            if special == "vflag" and name == "Temperature":
                fl[5] = 3
                val[5] = 99.0                 # a spike the flag removes
                good[5] = False
                qc |= 1
            if special == "pflag" and name == "Salinity":
                pflag = 5
                qc |= 2
            if special == "oob" and name == "Oxygen":
                val[:] = -50.0                # a dead sensor
            vs[name] = (val, fl, pflag)
            if pflag:
                continue
            lv = _truth_levels(z, la, val.astype(np.float64), good)
            lo_b, hi_b = QUANTITIES[qi][3], QUANTITIES[qi][4]
            lv[(lv < lo_b) | (lv > hi_b)] = np.nan
            truth_v[qi] = lv
        c = {"inst": inst, "cruise": cruise, "platform": pname, "lat": la,
             "lon": lo, "days": days, "uid": uid[0], "z": z.astype(
                 np.float32), "zflag": zflag, "vars": vs}
        if special == "notime":
            c["days"] = -1e10
        year = when.year
        ydir = BUNDLE_DIR if year < BUNDLE_BEFORE else str(year)
        if special == "otheryear":
            ydir = str(year + 1)
        files.setdefault((ydir, inst), []).append(c)
        t = f10b.seconds_since_epoch(when)
        keep = (special not in ("notime", "otheryear", "empty")
                and w_lo <= t <= w_hi and np.isfinite(truth_v).any())
        if keep:
            key = (f"{inst}:{cruise}" if cruise else
                   f"{inst}:platform:{pname}" if pname else
                   f"{inst}:cast:{uid[0]}")
            lonw = float(f10b.f10.wrap_lon(lo))
            truth.append({"t": t, "lat": la, "lon": lonw,
                          "platform": f10b.platform_hash(key),
                          "v": truth_v.reshape(-1).tolist(),
                          "qc": (INST_CODE[inst] << 4) | qc,
                          "_order": (year, INST_CODE[inst],
                                     len(files[(ydir, inst)]))})

    specials = {(0, 0): "vflag", (1, 1): "zflag", (2, 3): "pflag",
                (0, 3): "oob"}
    day = d_lo - dt.timedelta(days=1)
    k = 0
    while day <= d_hi:
        for h in (3, 15):
            when = dt.datetime.combine(day, dt.time(h, 7, 30))
            for ci, (inst, cru, pn, la, lo, names) in enumerate(SMOKE_CASTS):
                if day.year < BUNDLE_BEFORE and inst != "OSD":
                    continue          # the real 1800 bundle is OSD only
                sp = specials.get((ci, k % 4)) if h == 3 else None
                cast(inst, cru, pn, la + day.day * 0.1, lo, when, names, sp)
        k += 1
        day += dt.timedelta(days=1)
    # droppers: fill time (bundle), a 1899 cast in the 1900 OSD file, a cast
    # whose only variable is all fill
    cast("OSD", "US000001", "SHIP A", 5.0, 5.0,
         dt.datetime(1899, 12, 31, 12), ("Temperature",), "notime")
    cast("OSD", "US000001", "SHIP A", 5.0, 5.0,
         dt.datetime(1899, 12, 31, 12), ("Temperature",), "otheryear")
    empty_when = dt.datetime(1900, 1, 1, 12)
    cast("CTD", "JP000002", "", 5.0, 5.0, empty_when, ("Temperature",),
         "empty")
    files[("1900", "CTD")][-1]["vars"]["Temperature"] = (
        np.full(Z_LEVELS.size, -1e10, np.float32),
        np.zeros(Z_LEVELS.size, np.int8), 0)
    # the PFL file: garbage, never opened
    os.makedirs(os.path.join(base, "1900"), exist_ok=True)
    with open(os.path.join(base, "1900", "wod_pfl_1900.nc"), "wb") as fh:
        fh.write(b"this is family 8")
    listing = {}
    for (ydir, inst), cs in sorted(files.items()):
        d = os.path.join(base, ydir)
        os.makedirs(d, exist_ok=True)
        name = f"wod_{inst.lower()}_{ydir}.nc"
        write_wod_nc(os.path.join(d, name), cs)
        listing.setdefault(ydir, {})[name] = os.path.getsize(
            os.path.join(d, name))
    listing["1900"]["wod_pfl_1900.nc"] = 16

    def human(n):
        return f"{n / 1e3:.0f}K" if n < 1e6 else f"{n / 1e6:.0f}M"

    def page(title, rows):
        return ("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
                f"<html>\n <head>\n  <title>Index of {title}</title>\n </head>"
                f"\n <body>\n<h1>Index of {title}</h1>\n  <table>\n"
                "   <tr><th valign=\"top\"><img src=\"/icons/blank.gif\" "
                "alt=\"[ICO]\"></th><th><a href=\"?C=N;O=D\">Name</a></th>"
                "</tr>\n   <tr><th colspan=\"5\"><hr></th></tr>\n"
                "<tr><td valign=\"top\"><img src=\"/icons/back.gif\" alt="
                "\"[PARENTDIR]\"></td><td><a href=\"/data/oceans/ncei/\">"
                "Parent Directory</a></td><td>&nbsp;</td><td align=\"right\">"
                "  - </td><td>&nbsp;</td></tr>\n" + "".join(rows) +
                "   <tr><th colspan=\"5\"><hr></th></tr>\n</table>\n</body>"
                "</html>\n")

    for ydir, fs in listing.items():
        rows = [f'<tr><td valign="top"><img src="/icons/unknown.gif" alt="'
                f'[   ]"></td><td><a href="{n}">{n}</a></td><td align="right"'
                f'>2025-09-29 10:22  </td><td align="right">{human(s)}</td>'
                f'<td>&nbsp;</td></tr>\n' for n, s in sorted(fs.items())]
        with open(os.path.join(base, ydir, "index.html"), "w") as fh:
            fh.write(page(f"/data/oceans/ncei/wod/{ydir}", rows))
    rows = [f'<tr><td valign="top"><img src="/icons/folder.gif" alt="[DIR]">'
            f'</td><td><a href="{d}/">{d}/</a></td><td align="right">'
            f'2020-12-22 01:55  </td><td align="right">  - </td><td>&nbsp;'
            f'</td></tr>\n' for d in sorted(listing)]
    with open(os.path.join(base, "index.html"), "w") as fh:
        fh.write(page("/data/oceans/ncei/wod", rows))
    truth.sort(key=lambda r: r.pop("_order"))
    return truth


ADAPTER = WODAdapter
