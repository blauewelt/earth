"""Biogeochemical Argo synthetic profiles — one row per profile (family 1.gf).

PLAIN ENGLISH. BGC-Argo floats carry, beside the temperature and salinity
that family 8 already stores, sensors for oxygen, chlorophyll, particle
backscatter, nitrate, pH and light. The Argo data centres merge each float's
core and biogeochemical files into one "synthetic" profile file per float.
This adapter reads those, one float at a time, and puts every profile's six
BGC quantities onto the same 16 pressures the Argo T/S store uses, so a model
can read the ocean's biology and carbon state beside its physics.

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (Argo GDAC, Ifremer, no account):
  https://data-argo.ifremer.fr/argo_synthetic-profile_index.txt.gz  (8,081,783 B)
  https://data-argo.ifremer.fr/dac/<dac>/<wmo>/<wmo>_Sprof.nc
  * The index: 8 `#` comment lines ("Format version : 2.2", "Date of update"),
    a header `file,date,latitude,longitude,ocean,profiler_type,institution,
    parameters,parameter_data_mode,date_update`, then 408,053 rows (one per
    synthetic profile file `<dac>/<wmo>/profiles/S[DR]<wmo>_<cyc>[D].nc`),
    `date` YYYYMMDDHHMISS (207 rows have it empty), `parameters` a
    space-separated list. 2,963 floats in 10 DACs; profiles per year 11
    (2002) .. 40,065 (2025). 2022-05: 2,084 profiles from 478 floats (DOXY
    2,015, CHLA 1,114, BBP700 1,114, PH 840, NITRATE 724, PAR 406).
  * The float directory holds `<wmo>_Sprof.nc` (the merged synthetic file,
    NETCDF4_CLASSIC) beside `_prof`, `_meta`, `_tech`, `_Rtraj`. Measured on
    eight 2022-05 floats: 1.4 .. 48.6 MB, N_PROF 106 .. 419, N_LEVELS
    424 .. 1,877. There is NO per-profile DATA_MODE: `PARAMETER_DATA_MODE`
    (N_PROF x N_PARAM, R/A/D) is aligned with `STATION_PARAMETERS`
    (N_PROF x N_PARAM x STRING64). Every parameter X has X, X_QC,
    X_ADJUSTED, X_ADJUSTED_QC (fill 99999, QC " "); units DOXY and NITRATE
    micromole/kg, CHLA mg/m3, BBP700 m-1, PH_IN_SITU_TOTAL dimensionless;
    JULD days since 1950-01-01 (fill 999999).
  * QC flags MEASURED on those floats (value present, per mode): adjusted
    values carry 1, 2, 3, 4, 5 and 8; real-time BBP700 and PAR carry 0 (no
    QC); real-time DOXY/NITRATE/PH carry 3/4. Flag 8 ("estimated") is how the
    synthetic merge marks a value interpolated onto the shared pressure axis.

THE RULE — family 8's, per parameter. `build_family8_argo` reads the ADJUSTED
triple for DATA_MODE D or A and the raw one for R, and never falls back from
one to the other ("would silently mix two products under one flag"). Here the
mode is PARAMETER_DATA_MODE of that parameter in that profile: D or A ->
X_ADJUSTED with X_ADJUSTED_QC, R -> X with X_QC; the pressure axis likewise
by PRES's own mode. A sample counts when its value is not fill, finite, its QC
is in {1, 2, 5, 8} (good, probably good, changed, estimated) and its pressure
QC is in the same set. Samples are sorted by pressure (duplicates: first
kept) and interpolated onto the 16 levels by `build_family8_argo.
interp_levels`. So "adjusted when available" means: when the data centre
says it has adjusted the parameter; a real-time parameter is used raw, with
its own QC, and the qc byte says which.

WHAT A ROW IS.
  time_s    JULD rounded to the second, int32 seconds since 1982-01-01.
  lat, lon  LATITUDE, LONGITUDE (lon wrapped).
  platform  platform_hash(WMO number as text, e.g. "1902303").
  values    C = 96: DOXY (umol/kg), CHLA (mg/m3), BBP700 (1/m), NITRATE
            (umol/kg), PH_IN_SITU_TOTAL, DOWNWELLING_PAR (umol photons/m2/s)
            at the 16 pressures of `build_family8_argo.LEVELS`,
            VARIABLE-MAJOR. NaN where the float lacks the sensor or the
            level.
  qc        a bitmask over the six parameters in channel order: bit j set
            when parameter j contributed at least one level FROM ADJUSTED
            values (mode A or D). 0 = everything in the row is real-time raw.
Profiles are dropped and counted when JULD_QC or POSITION_QC is not 1/2
(`profiles_bad_time_qc`, `profiles_bad_position_qc`), JULD or the position is
fill (`profiles_no_time`, `profiles_no_position`), the profile is not in the
year / window (`profiles_other_year`, `profiles_outside_window`), (cycle,
direction) repeats within a file (`profiles_duplicate`), or none of the 96
values survives (`profiles_no_level`).

A FLOAT IS READ WHEN the index lists at least one profile of it in the year
(or the probe's month) whose `parameters` names one of the six; floats whose
profiles there carry none of them are counted and skipped
(`floats_no_bgc_param`). A listed float whose Sprof is missing is an absence.

THE PROBE, 2022-05, MEASURED 2026-09-17 (ml/family1/probes/
bgcargo_2022-05.json): 478 floats listed and read (0 without a BGC
parameter, 0 Sprof missing), 4,462,238,532 bytes in 194 s (peak RSS 323 MB)
— every file is the float's WHOLE history, so a month costs 2.6 MB a kept
profile; 1,717 May profiles kept from 409 floats (index: 2,084 profiles);
dropped in May: 147 position QC, 7 position fill, 213 with no level left.
Samples removed by QC (value present): PH 147,699 (real-time pH is flagged
3/4), CHLA 84,819, DOXY 64,381, BBP700 52,046, PAR 29,298, NITRATE 4,140.
Sources used (profiles): DOXY adjusted 1,578, CHLA adjusted 840, BBP700
adjusted 700 / raw 119, NITRATE adjusted 525, PH adjusted 321, PAR raw 201 /
adjusted 55. NaN fractions at 10 / 100 / 500 / 1100 / 1900 dbar: DOXY 0.08 /
0.12 / 0.28 / 0.41 / 0.50, CHLA 0.51 / 0.52 / 0.65 / 0.77 / 0.81, BBP700
0.52 / 0.54 / 0.63 / 0.75 / 0.79, NITRATE 0.69 / 0.70 / 0.85 / 0.85 / 0.89,
PH 0.81 / 0.81 / 0.89 / 0.90 / 0.93, PAR 0.85 / 0.85 / 0.99 / 0.99 / 0.99.
52 DOXY level values out of bounds.
WHOLE ARCHIVE: a HEAD of every float's Sprof (2,963 floats, 0 missing,
2026-09-17) totals 18.20 GB (the note's ≈ 15 GB); counting each file once per
calendar year its float was active, a full build downloads 74.7 GB (≈ 55 min
at the probe's 23 MB/s). At the probe's keep rate (0.82) the index's 408,053
profiles give ≈ 336 k rows, ≈ 74 MB stored at 219 B a row (the note's
408 k profiles and 80 MB).

MEMORY. One float's Sprof at a time is parsed (downloads run 4 at a time,
at most 8 files on disk, each deleted after parsing). The largest measured
file is 48.6 MB with N_PROF x N_LEVELS = 419 x 1,525; only the six
parameters' four arrays each plus PRES are read (≈ 26 arrays of that shape,
≈ 170 MB float64 at that size), and a float's packed rows are ≈ 250 B each.
"""
import gzip
import io
import os
import sys
import time
import zlib

import numpy as np

import build_family10_stores as f10b
import build_family8_argo as f8
from family1.adapters import _common as cm

GDAC = "https://data-argo.ifremer.fr/"
INDEX = GDAC + "argo_synthetic-profile_index.txt.gz"
PARAMS = ("DOXY", "CHLA", "BBP700", "NITRATE", "PH_IN_SITU_TOTAL",
          "DOWNWELLING_PAR")
QUANT = (("DOXY", "umol/kg", 0.0, 600.0),
         ("CHLA", "mg/m3", -0.5, 100.0),
         ("BBP700", "1/m", -0.001, 0.1),
         ("NITRATE", "umol/kg", -2.0, 60.0),
         ("PH_IN_SITU_TOTAL", "1", 6.5, 9.5),
         ("DOWNWELLING_PAR", "umol/m2/s", -1.0, 5000.0))
LEVELS = f8.LEVELS_ARR
NLEV = len(LEVELS)
CHANNELS = tuple((f"{q}_{int(p)}", u, lo, hi)
                 for q, u, lo, hi in QUANT for p in LEVELS)
GOOD_QC = (b"1", b"2", b"5", b"8")
FILL = 99999.0
FILL_JULD = 999999.0
INDEX_HEADER = ("file,date,latitude,longitude,ocean,profiler_type,"
                "institution,parameters,parameter_data_mode,date_update")
WORKERS = 4


class FormatError(ValueError):
    """An index or a file that is not the layout verified above."""


def _juld_epoch():
    return int(cm.days_from_civil(1950, 1, 1))


def parse_index(raw):
    """gzip bytes -> {year: {(dac, wmo): {"months": set, "profiles": n,
    "bgc": bool}}}, plus header facts."""
    try:
        text = gzip.decompress(raw).decode("latin-1")
    except (OSError, EOFError, zlib.error) as e:
        raise FormatError(f"the synthetic index is not a whole gzip ({e})")
    lines = text.splitlines()
    hdr = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    if not body or body[0].strip() != INDEX_HEADER:
        raise FormatError(f"unexpected index header "
                          f"{body[0][:100] if body else ''!r}")
    out = {}
    no_date = 0
    floats = set()
    for ln in body[1:]:
        if not ln.strip():
            continue
        p = ln.split(",")
        if len(p) != 10:
            raise FormatError(f"index row with {len(p)} fields: {ln[:80]!r}")
        parts = p[0].split("/")
        if len(parts) != 4 or parts[2] != "profiles":
            raise FormatError(f"index file path {p[0]!r}")
        key = (parts[0], parts[1])
        floats.add(key)
        d = p[1].strip()
        if len(d) != 14 or not d.isdigit():
            no_date += 1
            continue
        y, m = int(d[:4]), int(d[4:6])
        e = out.setdefault(y, {}).setdefault(
            key, {"months": {}, "profiles": 0, "bgc_months": set()})
        e["profiles"] += 1
        e["months"][m] = e["months"].get(m, 0) + 1
        if set(p[7].split()) & set(PARAMS):
            e["bgc_months"].add(m)
    if not out:
        raise FormatError("the synthetic index lists no dated profile — an "
                          "empty listing is a broken listing")
    meta = {"comment_lines": len(hdr), "rows": len(body) - 1,
            "rows_no_date": no_date, "floats": len(floats),
            "date_of_update": next((h.split(":", 1)[1].strip() for h in hdr
                                    if "Date of update" in h), None)}
    return out, meta



def read_sprof(path, t_lo, t_hi, year, counts):
    """One Sprof -> (t, lat, lon, vals (n, 96), qc (n,)). Counts every drop."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    ds.set_auto_scale(True)
    try:
        V = ds.variables
        n = len(ds.dimensions["N_PROF"])
        juld = np.asarray(V["JULD"][:], np.float64)
        jqc = np.asarray(V["JULD_QC"][:]).reshape(-1)
        lat = np.asarray(V["LATITUDE"][:], np.float64)
        lon = np.asarray(V["LONGITUDE"][:], np.float64)
        pqc = np.asarray(V["POSITION_QC"][:]).reshape(-1)
        cyc = np.asarray(V["CYCLE_NUMBER"][:], np.int64)
        dirn = np.asarray(V["DIRECTION"][:]).reshape(-1)
        sp = V["STATION_PARAMETERS"][:]
        pdm = np.asarray(V["PARAMETER_DATA_MODE"][:])
        npar = sp.shape[1]
        names = np.array([[b"".join(sp[i, j]).strip() for j in range(npar)]
                          for i in range(n)])
        counts["profiles_in_files"] = counts.get("profiles_in_files", 0) + n
        # (profiles of other years/months are counted, not described)
        keep = np.ones(n, bool)

        def drop(mask, key):
            nonlocal keep
            m = keep & mask
            if m.any():
                counts[key] = counts.get(key, 0) + int(m.sum())
            keep &= ~mask

        # the date first, so every later counter describes the window only
        drop(~np.isfinite(juld) | (juld >= FILL_JULD) | (np.abs(juld) > 1e30),
             "profiles_no_time")
        jd = np.where(keep, juld, 0.0)
        t = np.rint((jd + _juld_epoch()) * 86400.0).astype(np.int64)
        y0 = int(cm.days_from_civil(year, 1, 1)) * 86400
        y1 = int(cm.days_from_civil(year + 1, 1, 1)) * 86400
        drop((t < y0) | (t >= y1), "profiles_other_year")
        drop((t < t_lo) | (t > t_hi), "profiles_outside_window")
        drop(~np.isin(jqc, (b"1", b"2")), "profiles_bad_time_qc")
        drop(~np.isfinite(lat) | ~np.isfinite(lon) | (lat == FILL)
             | (np.abs(lat) > 90) | (np.abs(lon) > 360),
             "profiles_no_position")
        drop(~np.isin(pqc, (b"1", b"2")), "profiles_bad_position_qc")
        # (cycle, direction) repeats within one file: first kept
        seen = set()
        for i in np.flatnonzero(keep):
            k = (int(cyc[i]), bytes(dirn[i]))
            if k in seen:
                keep[i] = False
                counts["profiles_duplicate"] = \
                    counts.get("profiles_duplicate", 0) + 1
            seen.add(k)
        idx = np.flatnonzero(keep)
        if not idx.size:
            return None

        def mode_of(pname):
            m = np.full(n, b" ", "S1")
            hit = names == pname.encode()
            r, c = np.nonzero(hit)
            m[r] = pdm[r, c]
            return m

        def arr(k):
            return np.asarray(V[k][:], np.float64) if k in V else None

        def qarr(k):
            return np.asarray(V[k][:]) if k in V else None

        pm = mode_of("PRES")
        p_raw, p_adj = arr("PRES"), arr("PRES_ADJUSTED")
        pq_raw, pq_adj = qarr("PRES_QC"), qarr("PRES_ADJUSTED_QC")
        vals = np.full((idx.size, len(PARAMS) * NLEV), np.nan)
        qc = np.zeros(idx.size, np.int64)
        used = counts.setdefault("source_used", {})
        qdrop = counts.setdefault("samples_bad_qc", {})
        for pj, pname in enumerate(PARAMS):
            if pname not in V:
                continue
            pmode = mode_of(pname)
            raw, adj = arr(pname), arr(pname + "_ADJUSTED")
            rq, aq = qarr(pname + "_QC"), qarr(pname + "_ADJUSTED_QC")
            for j, i in enumerate(idx):
                m = pmode[i]
                if m == b" ":
                    continue                      # not a parameter here
                use_adj = m in (b"A", b"D")
                v, q = (adj, aq) if use_adj else (raw, rq)
                pmi = pm[i]
                pa = pmi in (b"A", b"D")
                pp, ppq = (p_adj, pq_adj) if pa else (p_raw, pq_raw)
                if v is None or q is None or pp is None or ppq is None:
                    counts["params_missing_variable"] = \
                        counts.get("params_missing_variable", 0) + 1
                    continue
                vi, qi, pi, pqi = v[i], q[i], pp[i], ppq[i]
                present = np.isfinite(vi) & (vi != FILL) & (np.abs(vi) < 1e30)
                pok = np.isfinite(pi) & (pi != FILL) & np.isin(pqi, GOOD_QC)
                good = present & pok & np.isin(qi, GOOD_QC)
                nb = int((present & ~good).sum())
                if nb:
                    qdrop[pname] = qdrop.get(pname, 0) + nb
                if not good.any():
                    continue
                ps, vs = f8._sorted_unique(pi[good], vi[good])
                lv = f8.interp_levels(ps, vs)
                if np.isfinite(lv).any():
                    vals[j, pj * NLEV:(pj + 1) * NLEV] = lv
                    key = f"{pname} {'adjusted' if use_adj else 'raw'}"
                    used[key] = used.get(key, 0) + 1
                    if use_adj:
                        qc[j] |= 1 << pj
        return (t[idx], lat[idx], lon[idx], vals, qc)
    finally:
        ds.close()


# ================================================================ adapter ==
class BGCArgoAdapter(f10b.SourceAdapter):
    store = "bgcargo"
    title = ("Biogeochemical Argo synthetic profiles (Argo GDAC Sprof): "
             "oxygen, chlorophyll, backscatter, nitrate, pH and PAR at the "
             "16 Argo pressures, one row per profile")
    family = "1gf"
    distribution = "public"
    licence = {"name": "CC BY 4.0 (Argo data policy)",
               "redistribution": "attribution", "derived_works": "free",
               "attribution": "Argo (2000). Argo float data and metadata "
                              "from Global Data Assembly Centre (Argo GDAC). "
                              "SEANOE. doi:10.17882/42182"}
    time_dtype = "int32"
    platform_meta = False
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))
    per_year = True
    first_year = 2002
    qc_policy = (
        "Per parameter and profile, PARAMETER_DATA_MODE D/A -> X_ADJUSTED "
        "with X_ADJUSTED_QC, R -> X with X_QC (family 8's rule, no "
        "fallback); pressure likewise by PRES's mode. Samples with QC in "
        "{1,2,5,8} and pressure QC in {1,2,5,8} are interpolated onto the "
        "16 RG levels (build_family8_argo.interp_levels). Profiles need "
        "JULD_QC and POSITION_QC in {1,2}. qc bit j = parameter j came from "
        "adjusted values. Out-of-bounds level values -> NaN, counted.")
    sources = (GDAC + "dac/<dac>/<wmo>/<wmo>_Sprof.nc", INDEX)
    verified = (
        "2026-09-17 from the sandbox: argo_synthetic-profile_index.txt.gz "
        "(8,081,783 B, 408,053 rows, 2,963 floats); eight 2022-05 Sprof files "
        "(1.4-48.6 MB): variables, PARAMETER_DATA_MODE alignment, units, QC "
        "distributions per parameter and mode")
    notes = (
        "Each float's whole Sprof is downloaded once per year it is active "
        "in (a float lives ~4-6 years), so a full build reads the archive "
        "several times over; the probe measures bytes per profile.")
    smoke_window = ("2021-12-30", "2022-01-02")
    smoke_probe_month = "2022-01"
    fetch_month_scope = "month"

    def __init__(self):
        self._index = None

    def _local(self, ctx, *p):
        return os.path.join(ctx.source_dir, "argo", *p)

    def index_data(self, ctx):
        if self._index is None:
            if ctx.source_dir:
                raw = open(self._local(
                    ctx, "argo_synthetic-profile_index.txt.gz"), "rb").read()
                ctx.count_bytes(len(raw))
            else:
                raw, why = cm.get_bytes(INDEX, attempts=ctx.a.attempts)
                if raw is None:
                    sys.exit(f"{INDEX} answered 404 ({why})")
            try:
                self._index = parse_index(raw)
            except FormatError as e:
                sys.exit(f"REFUSING bgcargo: {e}")
        return self._index

    def _fetch(self, ctx, dac, wmo):
        """-> (path, is_temp) or raises."""
        name = f"{wmo}_Sprof.nc"
        if ctx.source_dir:
            p = self._local(ctx, "dac", dac, wmo, name)
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        dest = os.path.join(ctx.scratch, "bgcargo", f"{dac}_{name}")
        url = f"{GDAC}dac/{dac}/{wmo}/{name}"
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

    def _floats(self, ctx, year, month=None):
        idx, _meta = self.index_data(ctx)
        fl = idx.get(year, {})
        todo, skipped, nprof = [], 0, 0
        for key in sorted(fl):
            e = fl[key]
            if month is not None and month not in e["months"]:
                continue
            bgc = e["bgc_months"] if month is None else \
                (e["bgc_months"] & {month})
            if not bgc:
                skipped += 1
                continue
            todo.append(key)
            nprof += (e["profiles"] if month is None else e["months"][month])
        return todo, {"floats_listed": len(todo) + skipped,
                      "floats_no_bgc_param": skipped,
                      "profiles_in_index": nprof}

    def _rows(self, ctx, year, t_lo, t_hi, label, month=None):
        todo, counts = self._floats(ctx, year, month)

        def get(key):
            try:
                return key, self._fetch(ctx, *key), None
            except f10b._NotFound:
                return key, None, "404"
            except IOError as e:
                return key, None, str(e)

        workers = 1 if ctx.source_dir else WORKERS
        for key, got, err in cm.ordered_map(get, todo, workers):
            if got is None:
                ctx.note_absent(label, f"{key[0]}/{key[1]}_Sprof.nc: "
                                       f"listed in the index, {err}")
                counts["floats_failed"] = counts.get("floats_failed", 0) + 1
                continue
            path, tmp = got
            c = {}
            try:
                r = read_sprof(path, t_lo, t_hi, year, c)
            except OSError as e:
                ctx.note_absent(label, f"{key[0]}/{key[1]}_Sprof.nc: not a "
                                       f"readable netCDF ({e})")
                continue
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            counts["floats_read"] = counts.get("floats_read", 0) + 1
            f10b._merge_counts(counts, c)
            if r is None:
                continue
            t, la, lo, v, qc = r
            oob = self.mask_bounds(v)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            alive = np.isfinite(v).any(axis=1)
            counts["profiles_no_level"] = \
                counts.get("profiles_no_level", 0) + int((~alive).sum())
            if not alive.any():
                continue
            n = int(alive.sum())
            q8 = qc[alive].astype(np.uint8)
            qd = counts.setdefault("qc", {})
            for u, k in zip(*np.unique(q8, return_counts=True)):
                qd[str(int(u))] = qd.get(str(int(u)), 0) + int(k)
            counts["profiles_kept"] = counts.get("profiles_kept", 0) + n
            yield label, self.pack(t[alive], la[alive], lo[alive], v[alive],
                                   np.full(n, f10b.platform_hash(key[1]),
                                           np.int64), q8), None
        yield label, None, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        idx, meta = self.index_data(ctx)
        per = {}
        for y in ctx.years:
            todo, c = self._floats(ctx, y)
            per[str(y)] = {"floats": len(todo), **c}
        first = None
        want = [y for y in ctx.years if per[str(y)]["floats"]]
        if want:
            key = self._floats(ctx, want[0])[0][0]
            path, tmp = self._fetch(ctx, *key)
            try:
                import netCDF4
                ds = netCDF4.Dataset(path)
                first = {"float": f"{key[0]}/{key[1]}",
                         "n_prof": len(ds.dimensions["N_PROF"]),
                         "n_levels": len(ds.dimensions["N_LEVELS"]),
                         "params": [p for p in PARAMS if p in ds.variables],
                         "format": ds.data_model}
                ds.close()
            finally:
                if tmp:
                    os.remove(path)
        return {"dataset": "BGC-Argo synthetic profiles (Sprof)",
                "url": INDEX, "index": meta,
                "years_listed": [min(idx), max(idx)],
                "per_year": per, "first_record": first}

    def fetch_year(self, ctx, year):
        t0 = time.time()
        agg = {}
        for label, rows, c in self._rows(ctx, year, ctx.t_lo, ctx.t_hi,
                                         str(year)):
            if c:
                f10b._merge_counts(agg, c)
            if rows is not None:
                yield label, rows, None
        agg["fetch_seconds"] = round(time.time() - t0, 1)
        yield str(year), None, agg

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        yield from self._rows(ctx, year, lo, hi, f"{year}-{month:02d}",
                              month=month)

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


# ================================================================== smoke ==
def write_sprof(path, wmo, profs, n_levels, params_all):
    """profs: [{juld, jqc, lat, lon, pqc, cyc, dir, params: {name: mode},
    pres, pres_qc, vals: {name: (raw, rawqc, adj, adjqc)}}] -> a Sprof-layout
    NETCDF4_CLASSIC file."""
    import netCDF4
    ds = netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC")
    n = len(profs)
    npar = 1 + len(params_all)
    for d, s in (("N_PROF", n), ("N_PARAM", npar), ("N_LEVELS", n_levels),
                 ("STRING64", 64), ("STRING8", 8), ("DATE_TIME", 14)):
        ds.createDimension(d, s)

    def s1(name, dims, rows, fill=b" "):
        v = ds.createVariable(name, "S1", dims, fill_value=fill)
        v[:] = rows
        return v

    pn = np.full((n, 8), b" ", "S1")
    for i in range(n):
        pn[i, :len(wmo)] = np.frombuffer(wmo.encode(), "S1")
    s1("PLATFORM_NUMBER", ("N_PROF", "STRING8"), pn)
    sp = np.full((n, npar, 64), b" ", "S1")
    pdm = np.full((n, npar), b" ", "S1")
    for i, pr in enumerate(profs):
        names = ["PRES"] + [p for p in params_all if p in pr["params"]]
        for j, nm in enumerate(names):
            sp[i, j, :len(nm)] = np.frombuffer(nm.encode(), "S1")
            pdm[i, j] = (pr["pres_mode"] if nm == "PRES"
                         else pr["params"][nm]).encode()
    s1("STATION_PARAMETERS", ("N_PROF", "N_PARAM", "STRING64"), sp)
    s1("PARAMETER_DATA_MODE", ("N_PROF", "N_PARAM"), pdm)
    v = ds.createVariable("CYCLE_NUMBER", "i4", ("N_PROF",),
                          fill_value=99999)
    v[:] = [p["cyc"] for p in profs]
    s1("DIRECTION", ("N_PROF",), np.array([p["dir"] for p in profs], "S1"))
    v = ds.createVariable("JULD", "f8", ("N_PROF",), fill_value=999999.0)
    v.units = "days since 1950-01-01 00:00:00 UTC"
    v[:] = [p["juld"] for p in profs]
    s1("JULD_QC", ("N_PROF",), np.array([p["jqc"] for p in profs], "S1"))
    for k in ("LATITUDE", "LONGITUDE"):
        v = ds.createVariable(k, "f8", ("N_PROF",), fill_value=99999.0)
        v[:] = [p["lat" if k == "LATITUDE" else "lon"] for p in profs]
    s1("POSITION_QC", ("N_PROF",), np.array([p["pqc"] for p in profs], "S1"))

    def block(base, getter, units):
        for suf in ("", "_ADJUSTED"):
            a = np.full((n, n_levels), FILL, np.float32)
            q = np.full((n, n_levels), b" ", "S1")
            for i, pr in enumerate(profs):
                g = getter(pr, suf)
                if g is None:
                    continue
                vv, qq = g
                a[i, :len(vv)] = vv
                q[i, :len(qq)] = qq
            v = ds.createVariable(base + suf, "f4", ("N_PROF", "N_LEVELS"),
                                  fill_value=np.float32(FILL))
            v.units = units
            v[:] = a
            s1(base + suf + "_QC", ("N_PROF", "N_LEVELS"), q)

    block("PRES", lambda pr, suf: (pr["pres"], pr["pres_qc"]), "decibar")
    units = {"DOXY": "micromole/kg", "CHLA": "mg/m3", "BBP700": "m-1",
             "NITRATE": "micromole/kg", "PH_IN_SITU_TOTAL": "dimensionless",
             "DOWNWELLING_PAR": "microMoleQuanta/m^2/sec"}
    for p in params_all:
        def get(pr, suf, p=p):
            if p not in pr["vals"]:
                return None
            raw, rq, adj, aq = pr["vals"][p]
            return (raw, rq) if suf == "" else (adj, aq)
        block(p, get, units[p])
    ds.close()


SMOKE_FLOATS = (
    # dac, wmo, params (name -> mode), lat, lon
    ("aoml", "1902303", {"DOXY": "D", "NITRATE": "A", "PH_IN_SITU_TOTAL":
                         "R"}, 30.0, -60.0),
    ("coriolis", "6903041", {"DOXY": "R", "CHLA": "A", "BBP700": "R",
                             "DOWNWELLING_PAR": "R"}, -40.0, 190.0),
    ("csiro", "5905000", {"CHLA": "D"}, -50.0, 140.0),
)


def _pv(p, pres):
    base = {"DOXY": 250 - pres / 10, "CHLA": 0.5 * np.exp(-pres / 60),
            "BBP700": 0.001 * np.exp(-pres / 200), "NITRATE": 5 + pres / 50,
            "PH_IN_SITU_TOTAL": 8.05 - pres / 5000,
            "DOWNWELLING_PAR": 1000 * np.exp(-pres / 30)}[p]
    return base.astype(np.float32)


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """The GDAC layout: the gzip synthetic index (real header and comment
    lines) and dac/<dac>/<wmo>/<wmo>_Sprof.nc.

    Hostile on purpose: an adjusted parameter whose raw values are garbage
    (must not be read), a real-time parameter whose adjusted array is
    garbage (must not be read), raw QC 3 (dropped) and 0 (dropped), QC 5
    and 8 (kept), a bad pressure QC, a JULD_QC 4 profile, a position-fill
    profile, a duplicated (cycle, direction), a descending profile, a value
    out of bounds, a profile outside the window, and a float whose index
    rows name only CDOM (skipped).
    """
    import datetime as dt
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "argo")
    os.makedirs(base, exist_ok=True)
    epoch = _juld_epoch()
    w_lo = f10b.seconds_since_epoch(d_lo)
    w_hi = f10b.seconds_since_epoch(d_hi) + 86399
    pres = np.arange(4.0, 1200.0, 7.0).astype(np.float32)
    nl = pres.size + 5
    index_rows = []
    truth = []
    all_params = PARAMS
    for fi, (dac, wmo, pmodes, la0, lo0) in enumerate(SMOKE_FLOATS):
        profs = []
        cyc = 0
        day = d_lo - dt.timedelta(days=1)
        while day <= d_hi:
            for h in (2, 14):
                cyc += 1
                when = dt.datetime.combine(day, dt.time(h, 11, 5))
                t = f10b.seconds_since_epoch(when)
                juld = t / 86400.0 - epoch
                la, lo = la0 + cyc * 0.05, lo0 + cyc * 0.05
                pr = {"juld": juld, "jqc": b"1", "lat": la, "lon": lo,
                      "pqc": b"1", "cyc": cyc, "dir": b"A",
                      "params": dict(pmodes), "pres_mode": "D",
                      "pres": pres, "pres_qc": np.full(pres.size, b"1"),
                      "vals": {}}
                tag = (fi, day.day, h)
                truth_v = np.full((len(PARAMS), NLEV), np.nan)
                qc = 0
                keep = w_lo <= t <= w_hi
                bad_pres = np.zeros(pres.size, bool)
                if tag == (0, 1, 2):
                    pr["pres_qc"] = pr["pres_qc"].copy()
                    pr["pres_qc"][10:20] = b"4"       # bad pressure: removed
                    bad_pres[10:20] = True
                for pj, p in enumerate(PARAMS):
                    if p not in pmodes:
                        continue
                    m = pmodes[p]
                    good_v = _pv(p, pres) * np.float32(
                        1 + 0.001 * rng.standard_normal())
                    gq = np.full(pres.size, b"1")
                    if m in "AD":
                        raw = np.full(pres.size, -7.0, np.float32)
                        rq = np.full(pres.size, b"3")
                        adj, aq = good_v, gq.copy()
                        if p == "CHLA":
                            aq[::3] = b"5"             # changed: kept
                            aq[1::3] = b"8"            # estimated: kept
                    else:
                        adj = np.full(pres.size, -7.0, np.float32)
                        aq = np.full(pres.size, b"1")
                        raw, rq = good_v, gq.copy()
                        if p == "BBP700":
                            rq[:] = b"0"               # no QC: dropped
                    ok = np.isin(aq if m in "AD" else rq, GOOD_QC) & \
                        ~bad_pres
                    if tag == (1, 2, 14) and p == "DOXY":
                        rq = rq.copy()
                        rq[:] = b"3"                   # raw QC 3: dropped
                        ok[:] = False
                    if tag == (0, 2, 14) and p == "NITRATE":
                        adj = adj.copy()
                        adj[:40] = 500.0               # out of bounds
                    pr["vals"][p] = (raw, rq, adj, aq)
                    vsrc = adj if m in "AD" else raw
                    if ok.any():
                        ps, vs = f8._sorted_unique(
                            pres[ok].astype(np.float64),
                            vsrc[ok].astype(np.float64))
                        lv = f8.interp_levels(ps, vs)
                        lo_b, hi_b = QUANT[pj][2], QUANT[pj][3]
                        fin = np.isfinite(lv)
                        if fin.any():
                            if m in "AD":
                                qc |= 1 << pj
                        lv[(lv < lo_b) | (lv > hi_b)] = np.nan
                        truth_v[pj] = lv
                if tag == (2, 1, 2):
                    pr["jqc"] = b"4"
                    keep = False
                if tag == (2, 1, 14):
                    pr["lat"] = pr["lon"] = 99999.0
                    keep = False
                profs.append(pr)
                if tag == (1, 1, 14):
                    # a descending twin 30 min later: a profile in its own
                    # right; then an exact duplicate of it (dropped)
                    twin = dict(pr)
                    twin["dir"] = b"D"
                    twin["juld"] = juld + 30 / 1440.0
                    profs.append(twin)
                    profs.append(dict(twin))
                    if keep:
                        truth.append({"t": t, "lat": la, "lon": float(
                            f10b.f10.wrap_lon(lo)), "platform":
                            f10b.platform_hash(wmo),
                            "v": truth_v.reshape(-1).tolist(), "qc": qc,
                            "_o": (fi, len(profs) - 3)})
                        t2 = t + 1800
                        truth.append({"t": t2, "lat": la, "lon": float(
                            f10b.f10.wrap_lon(lo)), "platform":
                            f10b.platform_hash(wmo),
                            "v": truth_v.reshape(-1).tolist(), "qc": qc,
                            "_o": (fi, len(profs) - 2)})
                    keep = False
                if keep and np.isfinite(truth_v).any():
                    truth.append({"t": t, "lat": la,
                                  "lon": float(f10b.f10.wrap_lon(lo)),
                                  "platform": f10b.platform_hash(wmo),
                                  "v": truth_v.reshape(-1).tolist(),
                                  "qc": qc, "_o": (fi, len(profs) - 1)})
                for pr_i in profs[len(profs) - (3 if tag == (1, 1, 14)
                                                else 1):]:
                    d = f"{int(round((pr_i['juld'] + epoch) * 86400)):d}"
                    tt = dt.datetime(1982, 1, 1) + dt.timedelta(
                        seconds=int(d))
                    suffix = "D" if pr_i["dir"] == b"D" else ""
                    index_rows.append(
                        f"{dac}/{wmo}/profiles/S{'D' if 'D' in pmodes.values() else 'R'}"
                        f"{wmo}_{pr_i['cyc']:03d}{suffix}.nc,"
                        f"{tt:%Y%m%d%H%M%S},{la:.3f},{lo:.3f},A,846,AO,"
                        f"PRES TEMP PSAL {' '.join(pmodes)},"
                        f"{'D' * (3 + len(pmodes))},20260101000000")
            day += dt.timedelta(days=1)
        d = os.path.join(base, "dac", dac, wmo)
        os.makedirs(d, exist_ok=True)
        write_sprof(os.path.join(d, f"{wmo}_Sprof.nc"), wmo, profs, nl,
                    all_params)
    # a float whose profiles carry none of the six: listed, never fetched
    for day in (d_lo, d_hi):
        index_rows.append(f"meds/4900000/profiles/SR4900000_001.nc,"
                          f"{day:%Y%m%d}120000,45.000,-50.000,A,846,ME,"
                          f"PRES TEMP PSAL CDOM,RRRR,20260101000000")
    index_rows.append("meds/4900000/profiles/SR4900000_000.nc,,,,A,846,ME,"
                      "PRES TEMP PSAL,RRR,20260101000000")
    head = ("# Title : Synthetic-Profile directory file of the Argo Global "
            "Data Assembly Center\n# Description : The directory file "
            "describes all individual synthetic-profile files of the argo "
            "GDAC ftp site.\n# Project : ARGO\n# Format version : 2.2\n"
            "# Date of update : 20260917112409\n# FTP root number 1 : "
            "ftp://ftp.ifremer.fr/ifremer/argo/dac\n# FTP root number 2 : "
            "ftp://usgodae.org/pub/outgoing/argo/dac\n# GDAC node : CORIOLIS"
            "\n" + INDEX_HEADER + "\n")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write((head + "\n".join(index_rows) + "\n").encode())
    with open(os.path.join(base, "argo_synthetic-profile_index.txt.gz"),
              "wb") as fh:
        fh.write(buf.getvalue())
    truth.sort(key=lambda r: r.pop("_o"))
    return truth


ADAPTER = BGCArgoAdapter
