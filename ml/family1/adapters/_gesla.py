"""GESLA-4 through the UHSLC ERDDAP — shared by `tide` and `tide_private`.

PLAIN ENGLISH. GESLA (Global Extreme Sea Level Analysis) is the compilation of
the world's tide-gauge records: sea level measured every hour or more often at
some 6,000 coastal gauges, a few since the 1800s. Its contributors attach
different terms to their data, so this project builds TWO stores from it: the
public `tide` (contributors who allow redistribution) and the private
`tide_private` (the research-only contributors). This module holds everything
the two share — the record list, the sort rule, the per-record reader — and the
two adapter modules differ only in which side of the sort they keep.

WHERE THE DATA IS, VERIFIED 2026-09-17 FROM THIS SANDBOX.
  * gesla.org — NOT reachable: the TLS handshake fails verification through
    this sandbox's proxy ("unable to get local issuer certificate", curl and
    python alike, www. and bare host). The GESLA-4 zip and its metadata CSV
    (datum, originator, record years) were therefore NOT read, and nothing
    here depends on them.
  * The UHSLC ERDDAP serves GESLA-4.1 as the tabledap dataset
    `global_hourly_gesla` ("GESLA Sea Level (sub-hourly to hourly)", created
    2026-07-20, time 1800-01-01T07:40Z .. 2026-07-15), cdm_data_type
    TimeSeries, variables sea_level (m, _FillValue -99.9999), time (s since
    1970), flag1 (0 no QC, 1 correct, 2 interpolated, 3 doubtful, 4 spike /
    wrong, 5 missing), flag2 (1 use, 0 do not use), latitude, longitude
    (0..360), station_name, agency_id ("GESLA contributor abbreviation"),
    station_code, station_country_code (ISO 3166-1 alpha-3), record_id (the
    GESLA file stem). Its licence text: "may be used and redistributed for
    free". The per-record DATUM and ORIGINATOR of GESLA's metadata are not
    exposed per record (the dataset's global attributes are one file's), so
    platforms.json carries `datum: null`.
  * The record list is `record_id,agency_id,station_name,station_code,
    station_country_code,latitude,longitude&distinct()` — 497,611 bytes, 6,152
    records (the note's 6,474 is the full GESLA-4 zip), 42 agency ids, in 1.7 s.

QUIRKS OF THAT SERVER, MEASURED — the reason the reader is per record.
  * A time-constrained query over ALL records timed out at the gateway
    (HTTP 504 after 60 s for one month in `.ncCF`); per agency it is close to
    the limit (CMEMS, 514 records, one month: 48 MB in 55 s) and the
    connection is reset under parallel load.
  * Every `orderByMinMax`, `orderByMax`, `orderByCount` and a
    `record_id + time`-only query answers "Your query produced no matching
    results" in ~4 s EVEN FOR A RECORD THAT HOLDS DATA (the_battery-8518750-
    usa-noaa, 895,834 rows 1920-2026): the server cannot report a record's
    time coverage. So a window is requested from EVERY record of the track,
    and "no matching results" is a legitimate empty record-window
    (`records_empty_in_window`), not an error.
  * One record per request in `.nc` (NetCDF-3, 20 bytes a row: time f8,
    sea_level f8, flag1 i2, flag2 i2): 17,920,840 bytes for The Battery's
    whole record in 3.6 s; a record-month in 1.7 s; an empty record-window in
    3.5 s. Parallel requests are RESET intermittently, so the reader retries
    with backoff and runs `WORKERS = 3`.
  * `actual_range` of sea_level is -1.84e11 .. 2.4e12: garbage exists in the
    source, and the channel bounds (below) turn it into NaN, counted.
  * A `distinct()` query constrained on a DATA variable (record_id where
    sea_level > 50 in 2019-03) answered HTTP 200 with 4 records, the same 4
    on every retry, where direct reads find 52: a filtered distinct is a
    PARTIAL answer with no error, so nothing here relies on one.
  * Co-located records from different contributors are distinct records
    (Brest: REFMAR, UHSLC_RQ and CMEMS); they are NOT de-duplicated — the
    platform hash keeps them apart and the consumer chooses.

THE SORT RULE (one function, `track_of`, reading the record's own fields):
  private  agency_id in {CMEMS, CV, UZ} — GESLA's research-only contributor
           codes — or station_country_code == ZAF: the South African gauges
           are SANHO's, and in this dataset every one of them (22 records,
           all UHSLC_FD / UHSLC_RQ) is redistributed by UHSLC, so the
           originator the terms attach to is visible only as the country.
  public   everything else.
Measured 2026-09-17: 538 private records (CMEMS 514, CV 1, UZ 1, ZAF 22),
5,614 public.

WHAT A ROW IS.
  time_s    the source's timestamp (UTC), int64 seconds since 1982-01-01
            (schema 3: records begin in 1800).
  lat, lon  the record's position (lon wrapped into [-180, 180)).
  platform  platform_hash(record_id); platforms.json maps it back.
  values    sea_level (m, relative to the record's own datum), C = 1.
  qc        flag1: 0 -> 0 (not assessed), 1 -> 1 (correct), 3 -> 3
            (doubtful), 4 -> 4 (wrong); rows worse than --qc-keep (default
            2) are dropped and counted (`qc_dropped`). flag1 2 (interpolated
            — not a measurement) and 5 (missing) are dropped and counted;
            flag2 0 ("do not use") is dropped and counted; a fill value is
            dropped and counted (`value_missing`); a value outside the bounds
            is dropped and counted (`out_of_bounds`); a repeated second keeps
            the last row (`rows_duplicate_time`).

THE PROBES, 2019-03, MEASURED 2026-09-17.
  tide_private  538 records requested; 351 had March rows, 187 answered "no
                matching results", 0 failed (a first run with a 4-attempt
                budget gave up on 3 records to connection resets — hence
                `ATTEMPTS = 6`, and a record that still fails is asked once
                more, alone, after the pass: the public probe lost one record
                inside a single reset wave even with six); 752,928 rows
                read, 714,313 kept (34,351 flag2 "do not use", 4,264
                interpolated; qc 1 on 713,604, qc 0 on 709; 0 out of
                bounds); 16,600,608 bytes (23.2 per kept
                row) in 1,897 s (3.5 s per record — this run shared the
                server with the public probe; alone, the first run took
                824 s, 1.5 s per record). ≈ 23 k rows a day: the CMEMS
                10-minute records dominate.
  tide          5,614 records requested (three workers, 16,393 s — 2.9 s a
                record through the sandbox relay's reset waves); 2,288 had
                March rows, 3,325 had none that were kept, 1 failed
                (galveston_pier_21-775a-usa-uhslc_rq, lost to a reset wave:
                that run predates the retry pass above). 2,674,817 rows read,
                2,438,577 kept ≈ 78,700 a day (193,408 flag2 "do not use",
                4,235 interpolated, 38,597 out of bounds; qc 0 on 1,314,576,
                qc 1 on 1,124,001); 63,932,756 bytes (26.2 per kept row).
                THE 38,597 ARE THE GREAT LAKES: every public record in
                40.5-50.5 N, 93.5-73.5 W (295) was re-read for March 2019 —
                52 NOAA Great Lakes / St Lawrence gauges hold exactly 38,597
                usable rows, all between 74.3 and 183.8 m (IGLD85), and no
                other record in the box has a value out of bounds. They are
                excluded by design (CHANNELS).
                That run also predates `records_all_out_of_bounds`, so its
                3,325 "empty" records include those lake gauges.
Single requests measured the same day: 1.0 s for a record-month with data,
2.8 s for an empty one; the rest of the per-record time in the probes is the
sandbox proxy's tunnel resets on this host and the backoff after them.

THE WHOLE ARCHIVE, SAMPLED 2026-09-17. The server cannot say how long a record
is, so the size is measured on a seeded random sample of WHOLE records (30
public, 12 private; one request each, sequential, 1800 .. now):
  public   3.26 MB of `.nc` and 162,579 rows a record (sd 4.47 MB; largest
           16.6 MB), 94.6 % kept, 14.3 years a record, 4.6 s a request ->
           x 5,614 records ≈ 18.3 GB (± 4.6 GB, one standard error) of
           requests, ≈ 9.1e8 rows read, ≈ 8.6e8 kept ≈ 28 GB stored at
           33 B/row (the note: 1.2-1.5e9 rows, 40-50 GB).
  private  6.36 MB and 317,887 rows a record (largest 27.4 MB, a CMEMS
           10-minute record), 98.4 % kept, 10.5 s a request -> x 538
           records ≈ 3.4 GB (± 1.2 GB) ≈ 1.7e8 kept ≈ 5.5 GB stored (the
           note: 10-20 GB).
A full build is ONE request per record over the whole window (6,152 in all),
so its time is the per-request time, not the month probe's: ≈ 7.2 h + 1.6 h
sequentially through this sandbox's proxy, ≈ 3 h at `WORKERS = 3` if the
server's resets allow it; from a hosted runner, less.

MEMORY. One record-window response at a time per worker (the largest
measured, The Battery 1920-2026, is 17.9 MB; a 6-minute record for 30 years
would be ≈ 53 MB) and its arrays (≈ 5x), three workers; then the framework's
PartWriter, which holds each year's rows (33 bytes each) until the year
reaches 1 M. A modern GESLA year holds ~10 M rows and flushes; the years
before ~1950 hold fewer than 1 M each and stay in memory until the pass ends —
≈ 33 bytes x the rows of those years.
"""
import csv
import datetime as dt
import io
import os
import sys
import time
import urllib.parse

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm
from family1.adapters import _nc3

ERDDAP = "https://uhslc.soest.hawaii.edu/erddap"
DATASET = "global_hourly_gesla"
TABLEDAP = f"{ERDDAP}/tabledap/{DATASET}"
INFO_URL = f"{ERDDAP}/info/{DATASET}/index.csv"
RECORD_FIELDS = ("record_id", "agency_id", "station_name", "station_code",
                 "station_country_code", "latitude", "longitude")
RECORDS_URL = (f"{TABLEDAP}.csv?" + "%2C".join(RECORD_FIELDS) + "&distinct()")
DATA_VARS = ("time", "sea_level", "flag1", "flag2")
RESEARCH_ONLY_AGENCIES = ("CMEMS", "CV", "UZ")
SANHO_COUNTRY = "ZAF"
FILL = -99.9999
# +-50 m IS A DELIBERATE CHOICE, AND IT EXCLUDES THE GREAT LAKES. sea_level is
# relative to each record's own datum, and GESLA's Great Lakes / St Lawrence
# gauges are on IGLD85: 74.6 m (Alexandria Bay) to 176.9 m (Alpena) in March
# 2019, Lake Superior near 184 m. Those rows are REAL, and they are kept out on
# purpose: they are lake levels, not sea level, and the store's float16 values
# would hold them to 0.0625 m (64-128 m) or 0.125 m (128-256 m) — coarser than
# much of a lake's signal. Every such row is counted in `out_of_bounds`, and a
# record whose rows ALL fall outside is named in
# `records_all_out_of_bounds_ids` (see the module docstring). Widening this
# bound is a one-line change IF a consumer wants lakes and accepts the
# quantisation.
CHANNELS = (("sea_level", "m", -50.0, 50.0),)
QC_OF_FLAG1 = {0: 0, 1: 1, 3: 3, 4: 4}


def track_of(agency_id, country_code):
    """THE sort rule: 'private' for the research-only sources, else 'public'."""
    if (agency_id or "").strip() in RESEARCH_ONLY_AGENCIES:
        return "private"
    if (country_code or "").strip().upper() == SANHO_COUNTRY:
        return "private"
    return "public"


class FormatError(ValueError):
    """A GESLA answer that is not the layout this reader was written for."""


def parse_records(text):
    """The distinct() CSV -> {record_id: {...}}. Refuses an empty list."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or tuple(rows[0]) != RECORD_FIELDS:
        raise FormatError(f"the record list header is {rows[:1]}, expected "
                          f"{RECORD_FIELDS}")
    out = {}
    bad = 0
    for r in rows[2:]:                   # rows[1] is ERDDAP's units line
        if len(r) != len(RECORD_FIELDS):
            bad += 1
            continue
        d = dict(zip(RECORD_FIELDS, r))
        try:
            lat, lon = float(d["latitude"]), float(d["longitude"])
        except ValueError:
            bad += 1
            continue
        if not d["record_id"] or not (-90 <= lat <= 90):
            bad += 1
            continue
        if d["record_id"] in out:
            raise FormatError(f"record {d['record_id']} listed twice")
        out[d["record_id"]] = {
            "id": d["record_id"], "contributor": d["agency_id"],
            "name": d["station_name"].replace("_", " "),
            "station_code": d["station_code"],
            "country": d["station_country_code"], "lat": lat,
            "lon": float(f10b.f10.wrap_lon(lon)), "datum": None,
            "track": track_of(d["agency_id"], d["station_country_code"]),
            "platform": f10b.platform_hash(d["record_id"])}
    if not out:
        raise FormatError("the GESLA record list is EMPTY — an empty listing "
                          "is a broken listing, not an empty archive")
    return out, bad


def iso(t82):
    """Seconds since 1982 -> the ERDDAP time literal."""
    d = dt.datetime(1982, 1, 1) + dt.timedelta(seconds=int(t82))
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def data_url(record_id, t_lo, t_hi):
    q = urllib.parse.quote(record_id, safe="")
    return (f"{TABLEDAP}.nc?" + "%2C".join(DATA_VARS)
            + f"&record_id=%22{q}%22"
            + "&time%3E=" + urllib.parse.quote(iso(t_lo), safe="")
            + "&time%3C=" + urllib.parse.quote(iso(t_hi), safe=""))


def local_name(record_id):
    return urllib.parse.quote(record_id, safe="") + ".nc"


def read_nc(raw):
    """ERDDAP's `.nc` answer -> (time s since 1970, sea_level, flag1, flag2)."""
    nc = _nc3.NC3(_nc3.bytes_reader(raw))
    miss = [v for v in DATA_VARS if v not in nc.vars]
    if miss:
        raise FormatError(f"the .nc answer lacks {miss} (has "
                          f"{sorted(nc.vars)})")
    units = nc.vars["time"]["attrs"].get("units", "")
    if not units.startswith("seconds since 1970-01-01"):
        raise FormatError(f"time units {units!r}")
    su = nc.vars["sea_level"]["attrs"].get("units", "")
    if su != "m":
        raise FormatError(f"sea_level units {su!r}, expected 'm'")
    return (nc.read("time").astype(np.float64),
            nc.read("sea_level").astype(np.float64),
            nc.read("flag1").astype(np.int64),
            nc.read("flag2").astype(np.int64))


def rows_from(time70, sl, f1, f2, t_lo, t_hi, qc_keep, counts):
    """The per-row rules of the module docstring -> (t, value, qc)."""
    n = time70.size
    counts["rows_read"] = counts.get("rows_read", 0) + int(n)
    ok = np.isfinite(time70)
    t = np.where(ok, np.rint(time70), 0).astype(np.int64) - cm.EPOCH_S_1970
    inside = ok & (t >= t_lo) & (t <= t_hi)
    counts["rows_outside_window"] = counts.get("rows_outside_window", 0) + \
        int((~inside).sum())
    fill = ~np.isfinite(sl) | (np.abs(sl - FILL) < 1e-6)
    keep = inside.copy()

    def drop(mask, name):
        nonlocal keep
        m = keep & mask
        k = int(m.sum())
        if k:
            counts[name] = counts.get(name, 0) + k
        keep &= ~mask

    drop(f2 == 0, "flag2_do_not_use")
    drop(f1 == 5, "flag1_missing")
    drop(f1 == 2, "flag1_interpolated")
    drop(~np.isin(f1, list(QC_OF_FLAG1)), "flag1_unknown")
    drop(fill, "value_missing")
    lo, hi = CHANNELS[0][2], CHANNELS[0][3]
    with np.errstate(invalid="ignore"):
        oob = (sl < lo) | (sl > hi)
    m = keep & oob
    if m.any():
        ob = counts.setdefault("out_of_bounds", {})
        ob["sea_level"] = ob.get("sea_level", 0) + int(m.sum())
    keep &= ~oob
    qc = np.zeros(n, np.int64)
    for k, v in QC_OF_FLAG1.items():
        qc[f1 == k] = v
    drop(qc > qc_keep, "qc_dropped")
    t, v, q = t[keep], sl[keep], qc[keep]
    if t.size:
        order = np.argsort(t, kind="stable")
        t, v, q = t[order], v[order], q[order]
        last = np.ones(t.size, bool)
        last[:-1] = t[1:] != t[:-1]
        counts["rows_duplicate_time"] = counts.get("rows_duplicate_time", 0) \
            + int((~last).sum())
        t, v, q = t[last], v[last], q[last]
    return t, v, q


class GESLAAdapter(f10b.SourceAdapter):
    """The two stores' common body; `track` picks the side of the sort."""

    track = "public"
    family = "1tf"
    time_dtype = "int64"
    platform_meta = True
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))    # hourly (or finer)
    per_year = False
    first_year = 1800
    qc_policy = (
        "qc is GESLA's flag1: 0 not assessed, 1 correct, 3 doubtful, 4 wrong; "
        "rows worse than --qc-keep (2) are dropped (`qc_dropped`). flag1 2 "
        "(interpolated) and 5 (missing), flag2 0 (do not use), the fill value "
        "(-99.9999) and values outside [-50, 50] m are dropped and counted — "
        "including, by design, the Great Lakes gauges (74-184 m on IGLD85: "
        "lake levels, which float16 would quantise to 6-12 cm), whose records "
        "are listed in `records_all_out_of_bounds_ids`; a "
        "repeated second keeps the last row.")
    sources = (TABLEDAP + ".nc (one record per request)", RECORDS_URL,
               INFO_URL)
    verified = (
        "2026-09-17 from the sandbox: gesla.org unreachable (TLS verify "
        "fails through the proxy); UHSLC ERDDAP global_hourly_gesla info, the "
        "distinct record list (6,152 records, 42 agencies), per-record .nc "
        "answers (aberdeen-9441187-usa-noaa, the_battery-8518750-usa-noaa, "
        "17,920,840 B, 895,834 rows), the broken orderBy / time-only queries "
        "and the 504 on all-record time queries")
    notes = (
        "Per-record requests over the whole track, each with the --start/"
        "--end window; the server cannot report record coverage, so an "
        "empty record-window is counted, not an error. `tide` and "
        "`tide_private` request disjoint record sets: together they read the "
        "archive once. Co-located records of different contributors are kept "
        "apart. Datum is not exposed per record (platforms.json: null).")
    smoke_window = ("1899-12-30", "1900-01-02")
    smoke_probe_month = "1900-01"
    fetch_month_scope = "month"
    WORKERS = 3
    # connection resets arrive every ~10 s under load (measured through the
    # sandbox proxy): 3 + 6 + 12 + 24 + 48 s of backoff before a record is
    # given up as an absence
    ATTEMPTS = 6

    def __init__(self):
        self._records = None

    # ------------------------------------------------------------ sources --
    def _local(self, ctx, *p):
        return os.path.join(ctx.source_dir, "gesla", *p)

    def _read_local(self, ctx, *p):
        with open(self._local(ctx, *p), "rb") as fh:
            raw = fh.read()
        ctx.count_bytes(len(raw))
        return raw

    def records(self, ctx):
        """{record_id: {...}} for ALL records (both tracks), and the counts."""
        if self._records is None:
            if ctx.source_dir:
                raw = self._read_local(ctx, "records.csv")
                info = self._read_local(ctx, "info.csv")
            else:
                raw, why = cm.get_bytes(RECORDS_URL, attempts=ctx.a.attempts)
                if raw is None:
                    sys.exit(f"REFUSING: the GESLA record list answered "
                             f"{why} — an empty listing is a broken listing")
                info, _ = cm.get_bytes(INFO_URL, attempts=ctx.a.attempts)
                if info is None:
                    sys.exit(f"{INFO_URL} answered 404 — the dataset moved")
            itext = info.decode("utf-8", "replace")
            for v in DATA_VARS + RECORD_FIELDS:
                if f"\nvariable,{v}," not in itext:
                    sys.exit(f"REFUSING: {DATASET} no longer has the variable "
                             f"{v!r} — re-verify the dataset")
            try:
                recs, bad = parse_records(raw.decode("utf-8", "replace"))
            except FormatError as e:
                sys.exit(f"REFUSING: {e}")
            self._records = (recs, bad)
        return self._records

    def mine(self, ctx):
        recs, _ = self.records(ctx)
        return sorted(k for k, r in recs.items() if r["track"] == self.track)

    def _fetch(self, ctx, rid, t_lo, t_hi):
        """One record-window -> (raw .nc bytes or None, why)."""
        if ctx.source_dir:
            p = self._local(ctx, "data", local_name(rid))
            if not os.path.exists(p):
                return None, "notfound"
            return self._read_local(ctx, "data", local_name(rid)), "ok"
        return cm.get_bytes(data_url(rid, t_lo, t_hi),
                            attempts=max(self.ATTEMPTS, ctx.a.attempts))

    def _record(self, ctx, rid, t_lo, t_hi):
        counts = {"records_requested": 1}
        try:
            raw, why = self._fetch(ctx, rid, t_lo, t_hi)
        except IOError as e:
            return rid, None, counts, f"{rid}: {e}"
        if raw is None:
            if why == "empty":
                counts["records_empty_in_window"] = 1
                return rid, {}, counts, None
            return rid, None, counts, (f"{rid}: HTTP 404 without ERDDAP's "
                                       f"'no matching results' — the record "
                                       f"or the dataset has moved")
        try:
            tm, sl, f1, f2 = read_nc(raw)
        except (FormatError, _nc3.NC3Error) as e:
            raise FormatError(f"{rid}: {e}") from None
        t, v, q = rows_from(tm, sl, f1, f2, t_lo, t_hi, ctx.qc_keep, counts)
        if not t.size:
            oob = counts.get("out_of_bounds", {}).get("sea_level", 0)
            if oob:
                # rows WERE there, and every usable one was outside the
                # bounds — a Great Lakes gauge, measured: not "empty"
                counts["records_all_out_of_bounds"] = 1
                counts["records_all_out_of_bounds_ids"] = [rid]
            else:
                counts["records_empty_in_window"] = 1
            return rid, {}, counts, None
        counts["records_with_rows"] = 1
        rec = self.records(ctx)[0][rid]
        ty = (np.datetime64("1982-01-01", "s")
              + t.astype("timedelta64[s]")).astype("datetime64[Y]") \
            .astype(np.int64) + 1970
        by_year = {}
        for y in np.unique(ty).tolist():
            m = ty == y
            n = int(m.sum())
            by_year[int(y)] = self.pack(
                t[m], np.full(n, rec["lat"]), np.full(n, rec["lon"]),
                v[m].reshape(n, 1), np.full(n, rec["platform"], np.int64),
                q[m])
            qc = counts.setdefault("qc", {})
            for k in np.unique(q[m]).tolist():
                qc[str(k)] = qc.get(str(k), 0) + int((q[m] == k).sum())
        counts["rows_kept"] = int(t.size)
        return rid, by_year, counts, None

    def _stream(self, ctx, t_lo, t_hi, unit):
        todo = self.mine(ctx)
        counts = {"records_track": len(todo)}
        workers = 1 if ctx.source_dir else self.WORKERS
        t0 = time.time()
        fn = (lambda rid: self._record(ctx, rid, t_lo, t_hi))
        retry = []
        for i, (rid, by_year, c, err) in enumerate(
                cm.ordered_map(fn, todo, workers), 1):
            if err:
                # NOT YET AN ABSENCE: resets come in waves (measured: a
                # record lost all six attempts inside one), so a failed record
                # is asked again, alone, after the pass
                retry.append(rid)
                continue
            f10b._merge_counts(counts, c)
            for y in sorted(by_year):
                yield y, by_year[y], None
            if i % 200 == 0:
                ctx.prog.item(f"{self.store} records {i}/{len(todo)}", None,
                              {"rows_kept": counts.get("rows_kept", 0),
                               "elapsed_s": round(time.time() - t0, 1)})
        for rid in retry:
            time.sleep(0 if ctx.source_dir else 30)
            rid, by_year, c, err = self._record(ctx, rid, t_lo, t_hi)
            counts["records_retried"] = counts.get("records_retried", 0) + 1
            f10b._merge_counts(counts, c)
            if err:
                ctx.note_absent(unit, err)
                counts["records_failed"] = counts.get("records_failed", 0) + 1
                counts.setdefault("records_failed_ids", []).append(rid)
                continue
            for y in sorted(by_year):
                yield y, by_year[y], None
        counts["stream_seconds"] = round(time.time() - t0, 1)
        yield None, None, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        recs, bad = self.records(ctx)
        mine = self.mine(ctx)
        by = {}
        for r in recs.values():
            d = by.setdefault(r["track"], {})
            d[r["contributor"]] = d.get(r["contributor"], 0) + 1
        first = None
        # ONE real record: the first of (at most) five that has rows in the
        # window — the server cannot say which records cover it
        for rid in mine[:5]:
            raw, why = self._fetch(ctx, rid, ctx.t_lo, ctx.t_hi)
            if raw is None:
                continue
            tm, sl, f1, f2 = read_nc(raw)                 # raises if malformed
            t, _, _ = rows_from(tm, sl, f1, f2, ctx.t_lo, ctx.t_hi, 4, {})
            if not t.size:
                continue
            first = {"record": rid, "rows_in_window": int(t.size),
                     "first_time": iso(int(t[0]))}
            break
        return {
            "dataset": f"{ERDDAP} {DATASET}",
            "track": self.track,
            "records": len(recs),
            "records_bad_lines": bad,
            "records_this_track": len(mine),
            "sort": {k: {"records": sum(v.values()), "by_contributor":
                         dict(sorted(v.items()))} for k, v in by.items()},
            "sort_rule": (f"private if agency_id in {RESEARCH_ONLY_AGENCIES} "
                          f"or station_country_code == {SANHO_COUNTRY!r}"),
            "requests_per_window": len(mine),
            "first_record": first,
        }

    def fetch_stream(self, ctx):
        yield from self._stream(ctx, ctx.t_lo, ctx.t_hi, str(ctx.d_lo.year))

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        parts, counts = [], {}
        for y, rows, c in self._stream(ctx, lo, hi, str(year)):
            if y is None:
                counts = c
            else:
                parts.append(rows)
        rows = cm.concat_rows(parts, self.C, self.time_dtype)
        yield from cm.batch_rows(self, rows, counts, f"{year}-{month:02d}")

    def platforms(self, ctx):
        recs, _ = self.records(ctx)
        return {int(r["platform"]): {k: r[k] for k in (
            "id", "lat", "lon", "name", "country", "contributor",
            "station_code", "datum", "track")}
            for r in recs.values() if r["track"] == self.track}

    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, self.track, seed)


# ================================================================== smoke ==
SMOKE_RECORDS = (
    # record_id, agency, name, code, country, lat, lon(0..360), kind
    ("alpena_mi-9075065-usa-noaa", "NOAA", "Alpena_MI", "9075065", "USA",
     45.0603, 276.5714, "lake"),
    ("bergen-5-nor-cmems", "CMEMS", "Bergen", "5", "NOR", 60.39, 5.32,
     "ten_minute"),
    ("capetown-012a-zaf-uhslc_rq", "UHSLC_RQ", "Cape_Town", "012A", "ZAF",
     -33.9, 18.43, "hourly"),
    ("empty-1-usa-noaa", "NOAA", "Empty", "1", "USA", 30.0, 270.0, "none"),
    ("halifax-490-can-meds", "MEDS", "Halifax", "490", "CAN", 44.67, 296.42,
     "flags"),
    ("st_john's-1-can-meds", "MEDS", "St_John's", "1", "CAN", 47.56, 307.29,
     "hourly"),
    ("the_battery-8518750-usa-noaa", "NOAA", "The_Battery", "8518750", "USA",
     40.70056, 285.9858, "hourly"),
    ("venezia-vene-ita-cv", "CV", "Venezia", "VENE", "ITA", 45.42, 12.43,
     "hourly"),
)


def make_smoke_sources(root, d_lo, d_hi, track, seed=20260917):
    """The ERDDAP answers of an eight-record GESLA, written to disk.

    `records.csv` and `info.csv` as ERDDAP returns them, and one
    `data/<record_id>.nc` per record holding the record's WHOLE series (the
    local reader applies the window, as the server would). Hostile: every
    flag1 and flag2 case, the fill value, a 1e11 garbage value, a repeated
    second, an apostrophe in a record id, a ten-minute record, a record with
    no data at all, a Great Lakes record at 176.8 m (every row out of bounds
    by design, the record named), and records on both sides of the sort.
    Returns the truth
    rows of `track`.
    """
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "gesla")
    os.makedirs(os.path.join(base, "data"), exist_ok=True)
    with open(os.path.join(base, "records.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(RECORD_FIELDS)
        w.writerow(["", "", "", "", "", "degrees_north", "degrees_east"])
        for rid, ag, nm, code, cc, la, lo, _ in SMOKE_RECORDS:
            w.writerow([rid, ag, nm, code, cc, la, lo])
    with open(os.path.join(base, "info.csv"), "w") as fh:
        fh.write("Row Type,Variable Name,Attribute Name,Data Type,Value\n")
        for v in DATA_VARS + RECORD_FIELDS:
            fh.write(f"variable,{v},,double,\n")
    t0 = f10b.seconds_since_epoch(d_lo) - 86400
    t1 = f10b.seconds_since_epoch(d_hi) + 86400        # end of the window
    w_lo = f10b.seconds_since_epoch(d_lo)
    w_hi = f10b.seconds_since_epoch(d_hi) + 86399
    truth = []
    for rid, ag, nm, code, cc, la, lo, kind in SMOKE_RECORDS:
        step = 600 if kind == "ten_minute" else 3600
        ts = np.arange(t0, t1, step, dtype=np.int64)
        if kind == "none":
            # a record with no data in the window: the server answers "no
            # matching results"; locally, the window filter finds nothing
            ts = ts - 50 * 365 * 86400
        n = ts.size
        sl = np.round(rng.uniform(-2.0, 8.0, n), 3)
        if kind == "lake":
            # a Great Lakes gauge on IGLD85: every value near 176.8 m, all
            # real, all outside the +-50 m bound by design (CHANNELS)
            sl = np.round(rng.uniform(176.6, 177.0, n), 3)
        f1 = np.ones(n, np.int16)
        f2 = np.ones(n, np.int16)
        keep = np.ones(n, bool)
        qc = np.ones(n, np.int64)
        if kind == "flags":
            # inside the window, on consecutive hours
            i0 = int(np.searchsorted(ts, w_lo)) + 5
            f1[i0] = 0
            qc[i0] = 0                              # kept, not assessed
            f1[i0 + 1] = 2
            keep[i0 + 1] = False                    # interpolated
            f1[i0 + 2] = 3
            keep[i0 + 2] = False                    # doubtful > qc_keep
            f1[i0 + 3] = 4
            keep[i0 + 3] = False
            f1[i0 + 4] = 5
            sl[i0 + 4] = FILL
            keep[i0 + 4] = False                    # missing
            f2[i0 + 5] = 0
            keep[i0 + 5] = False                    # do not use
            sl[i0 + 6] = FILL
            keep[i0 + 6] = False                    # fill with flag1 1
            sl[i0 + 7] = 1.84e11
            keep[i0 + 7] = False                    # garbage: out of bounds
            f1[i0 + 8] = 9
            keep[i0 + 8] = False                    # unknown flag
            # a repeated second: the earlier row is dropped
            ts = np.insert(ts, i0 + 10, ts[i0 + 10])
            sl = np.insert(sl, i0 + 10, 0.5)
            f1 = np.insert(f1, i0 + 10, 1)
            f2 = np.insert(f2, i0 + 10, 1)
            keep = np.insert(keep, i0 + 10, False)
            qc = np.insert(qc, i0 + 10, 1)
        time70 = (ts + cm.EPOCH_S_1970).astype(np.float64)
        _nc3.write(os.path.join(base, "data", local_name(rid)),
                   [("row", ts.size)],
                   [("time", ("row",), time70,
                     {"units": "seconds since 1970-01-01T00:00:00Z"}),
                    ("sea_level", ("row",), sl,
                     {"units": "m", "_FillValue": np.float64(FILL)}),
                    ("flag1", ("row",), f1, {}),
                    ("flag2", ("row",), f2, {})],
                   {"title": "GESLA Sea Level (sub-hourly to hourly)"})
        if track_of(ag, cc) != track or kind == "lake":
            continue
        inside = (ts >= w_lo) & (ts <= w_hi) & keep
        for t, v, q in zip(ts[inside], sl[inside], qc[inside]):
            truth.append({"t": int(t), "lat": la,
                          "lon": float(f10b.f10.wrap_lon(lo)),
                          "platform": f10b.platform_hash(rid),
                          "v": [float(v)], "qc": int(q)})
    return truth
