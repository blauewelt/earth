#!/usr/bin/env python3
"""E-034 / E-070: GLORYS12 DAILY base channels — North Atlantic, or the globe.

Chris, 2026-08-16: *"Go ahead with the GLORYS12 daily fetcher, and more
generally, with preparing all data both daily and 5-day."*

THE ONE DESIGN DECISION THIS FILE ENCODES, because everything else follows
from it: **fetch DAILY, derive pentad by aggregation — never two pipelines.**
A 5-day mean is a pure reduction of the daily series, so building the two
cadences independently would be two implementations of one rule, and the two
would eventually disagree about a leap day or a bin edge. That is the same
defect class as the status page re-deriving the LR schedule (which drew a
cosine for a run that used expdecay) and the corridor being traced by hand in
the frontend. So: this script writes DAILY files, and `--cadence pentad`
aggregates the same downloaded bytes with `ml/aggregate_cadence.py`. The
daily files are the single source of truth for both tensors.

Pentads are the fixed 5-day bins of `ml/build_truth_pentad.py` — counted from
1982-01-01, index = floor(days_since_epoch / 5) — so the state axis and the
label axis land in the same bins by construction rather than by coincidence.

WHAT IS FETCHED. `cmems_mod_glo_phy_my_0.083deg_P1D-m` (GLORYS12 reanalysis,
daily, 1/12 degree), verified reachable 2026-08-16. Variables `uo`, `vo`
(surface current -> speed), `mlotst` (mixed-layer depth), `zos` (sea surface
height), depth slice 0-1 m only. Subsetted server-side to the NA window
(100 W..20 E, 0..70 N) — 1/12 degree globally is 16x the pixels of the 1/4
degree ensemble the monthly bake used, and we bin down to 0.25 degrees
anyway, so subsetting is what makes this affordable at all.

RESUME. One file per (year, month) chunk, skipped when already present and
non-empty. A month of the subsetted window is ~30 slices; the loop can be
killed and restarted without losing work, which matters because the full
1993-2024 pull is hours of transfer.

DISK. The caller must have room: the daily NA tensor is ~165 GB in fp16 and
does NOT fit a 100 GB Vast box or this sandbox (measured 8.7 GB free on
2026-08-16). Run this where the disk is — see ml/plans/E034_pentad_tensor.md
section 5. `--dry-run` reports the plan and the byte estimate without
fetching, and `--months N` limits the pull for a smoke test.

---

E-070: THE SAME MACHINERY, THE WHOLE GLOBE, BINNED ON ARRIVAL (`--window
global --bin-deg 0.25`). Family 7 is a global 0.25 degree tensor, and the
global window is 7.29x the NA window's pixels — 2.19 GB a month on the wire,
840 GB over 1993-2024 (measured, not modelled). That does not belong on the
Hub, and nothing downstream would read it at 1/12 degree anyway: every tensor
this programme has built bins to 0.25 degrees first. Binned, the same 384
chunks are **99 GB** — 257 MB each, measured.

So a global chunk is binned BEFORE it is stored, and the two rules that
governed E-034 both still hold, which is the only reason this is one script
and not a second pipeline:

  * **Fetch daily, derive by aggregation — never two pipelines.** The binned
    DAILY files are the single source of truth for the global family; the
    global pentad cadence is `ml/aggregate_cadence.py` over exactly these
    bytes, the same reduction the NA family already uses.
  * **One binning definition.** `bin_plan` / `bin_slice` are IMPORTED from
    `ml/aggregate_cadence.py` rather than restated here, so the global
    tensor's North-Atlantic sub-block is computed by the code that made
    family 4. A second implementation of a bin rule is the defect class that
    drew a cosine on the status page for a run that used expdecay.

The grid is point-aligned (samples ON multiples of 0.25 degrees), so the NA
window's 281x481 axes are an exact sub-block of the global 681x1440 ones —
asserted in `tests/test_glorys_global.py`, because a half-cell offset is
invisible in every plot anyone would draw. Each binned chunk records
`bin_deg`, `grid_align` and the source dataset id as NetCDF attributes so a
downstream builder can REFUSE a mismatched grid where the inputs are all it
has cost (ml/CLAUDE.md section 0.3) rather than discovering it in a result.

Peak RAM is bounded by ONE DAY, not one month, and the bound is on the
REQUEST rather than on the read: a whole global month in one `cm.subset`
peaked at **5.95 GB and was OOM-killed** on a 7 GB box (2026-09-03), before
any binning had happened. `--subset-days 1` fetches one day (70.6 MB, ~13 s),
bins it, appends it and deletes it; **measured peak RSS for a whole month is
1.91 GB**, and a chunk takes 411 s of fetch + 37 s of binning + the upload.

Credentials: env vars COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD, from the
project doc claude/copernicus-marine-access.md. Never written to disk here.

Run:
  python3 ml/fetch_glorys_daily.py --dry-run
  python3 ml/fetch_glorys_daily.py --start 1993-01 --end 2024-12 --out /data/glorys_daily
  python3 ml/fetch_glorys_daily.py --window global --bin-deg 0.25 \\
      --start 1993-01 --end 2024-12 --out "$RUNNER_TEMP/glorys_global"
"""
import argparse
import calendar
import datetime as dt
import os
import shutil
import sys

DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
# The interim stream covers the most recent months under a different id; it is
# resolved from the catalogue at run time rather than hardcoded, because a
# guessed id is how the first credential check earned a DatasetNotFound.
INTERIM_HINT = "myint"
VARIABLES = ["uo", "vo", "mlotst", "zos"]

# The family-3 window, verbatim. Any drift here silently makes a tensor that
# cannot be compared with anything already measured.
WINDOW = dict(minimum_longitude=-100.0, maximum_longitude=20.0,
              minimum_latitude=0.0, maximum_latitude=70.0)

# GLORYS12's FULL extent, MEASURED from the catalogue on 2026-09-03 rather
# than assumed: `copernicusmarine.describe(dataset_id=DATASET)` reports
# latitude -80.0..90.0 and longitude -180.0..179.9166717529297, both at
# 1/12 degree. The maxima here are the half-open request (180.0, 90.0); the
# served axis stops one cell short in longitude, which is what a global grid
# with no duplicated seam looks like.
GLOBAL_WINDOW = dict(minimum_longitude=-180.0, maximum_longitude=180.0,
                     minimum_latitude=-80.0, maximum_latitude=90.0)

WINDOWS = {"na": WINDOW, "global": GLOBAL_WINDOW}

# Cells at 1/12 degree, per window — the number the byte estimates scale on.
# NA is written as the original expression, digit for digit, because its
# printed estimate is a published number that other documents quote.
CELLS = {"na": int(120 / (1 / 12)) * int(70 / (1 / 12)),          # 1,209,600
         "global": 4320 * 2041}                                    # 8,817,120

# MEASURED, not modelled. NA: one real 2015-01 chunk came back at 287 MB for
# 31 days of the subsetted window (smoke test, 2026-08-16). GLOBAL: one real
# 1993-01 chunk came back at 2,188 MB on the wire, in 31 day-sized requests,
# and 257 MB binned to 0.25 degrees with zlib (smoke test, 2026-09-03). The
# wire figure is within 5% of 287 MB x the 7.29x pixel ratio, which is the
# sanity check that the subset really was global and not a wrapped strip.
MEAS_GB = {"na": 0.287, "global": 2.188}
# Binned output, measured the same day. The NA entry is NOT measured — nothing
# runs that combination — so it is derived from the pixel ratio and says so,
# rather than borrowing the global number's authority.
MEAS_BINNED_GB = {"global": 0.257,
                  "na": 0.287 * (281 * 481) / CELLS["na"]}      # derived

EPOCH = dt.date(1982, 1, 1)
PENTAD_DAYS = 5


# What the service actually SERVES for each window, as opposed to what is
# requested. They differ at the dateline: the request is -180..180 and the
# grid stops at 180 - 1/12, because a global 1/12 degree grid has no
# duplicated seam. Used only for the size estimates and the tests; the binner
# reads the real axes off the chunk it just fetched.
SERVED = {"na": dict(lat=(0.0, 70.0), lon=(-100.0, 20.0)),
          "global": dict(lat=(-80.0, 90.0), lon=(-180.0, 180.0 - 1 / 12))}


def binned_shape(window, bin_deg, align="point"):
    """(nlat, nlon) the binner will produce, from `aggregate_cadence.axis_for`.

    Derived with the SAME function the binning uses, so an estimate cannot
    describe a grid the run will not build.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aggregate_cadence import axis_for
    s = SERVED[window]
    return (len(axis_for(s["lat"][0], s["lat"][1], bin_deg, align)),
            len(axis_for(s["lon"][0], s["lon"][1], bin_deg, align)))


def chunk_name(window, bin_deg, y, m):
    """The chunk's file name. E-034's NA contract is preserved EXACTLY.

    `glorys_YYYYMM.nc` means one thing — an NA-window chunk on the native
    1/12 degree grid — and 384 of them are already on the Hub under that
    name, read by `aggregate_cadence.py`'s MONTH_RE and by two workflows. So
    the default path keeps it, and everything else is named for what it is:
    grid, window, month.
    """
    if window == "na" and not bin_deg:
        return f"glorys_{y}{m:02d}.nc"
    grid = f"{round(bin_deg * 100):03d}" if bin_deg else "raw"
    return f"glorys{grid}_{window}_{y}{m:02d}.nc"


def hub_folder(window, bin_deg):
    """Where the chunk lives in the Hub dataset (a prefix, '' = the root).

    The NA chunks stay in the repo root where E-034 put them — moving 384
    files to tidy a namespace would break every reader for nothing. A new
    family gets a folder, so `list_repo_files` can tell the two apart by
    prefix and neither pull can resume off the other's names.
    """
    if window == "na" and not bin_deg:
        return ""
    grid = f"daily{round(bin_deg * 100):03d}" if bin_deg else "daily_raw"
    return f"{grid}_{window}/"


def months_between(start, end):
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def sha256(path, buf=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def hf_connect(repo_name):
    """Resolve the namespace and ensure the dataset repo exists."""
    import os as _os
    tok = _os.environ.get("HF_TOKEN") or (
        open("/home/claude/.hf_token").read().strip()
        if _os.path.exists("/home/claude/.hf_token") else "")
    if not tok:
        sys.exit("no HF_TOKEN (env or ~/.hf_token) and backup is enabled — "
                 "see claude/huggingface-access.md, or pass --no-backup and "
                 "accept that a dead box costs the whole pull")
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    repo = repo_name if "/" in repo_name else f"{api.whoami()['name']}/{repo_name}"
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    return api, repo, tok


def hf_preflight(api, repo, tok, scratch):
    """PROVE THE BACKUP WORKS BEFORE FETCHING ANYTHING.

    Chris, 2026-08-16: *"I would build the HF backup into that script (test
    it in the beginning). Otherwise we will have built the training data and
    then lost it."* Exactly right, and the repo already has the scar: the
    embed-cache push sat AFTER `wait $S2_PID`, so a cache that existed for
    sixteen hours was published at the very end — and when the upload failed
    for lack of room, the run reported success (ml/CLAUDE.md §5.20, §4.6).

    So this uploads a few bytes, downloads them BACK, compares, and deletes —
    the entire round trip, exercised while it has cost nothing. If the token,
    the namespace, the quota or the network is wrong, the job dies here
    instead of after 110 GB."""
    import os as _os
    from huggingface_hub import hf_hub_download
    probe = _os.path.join(scratch, ".preflight")
    payload = b"earth/E-034 backup preflight\n"
    with open(probe, "wb") as f:
        f.write(payload)
    want = sha256(probe)
    api.upload_file(path_or_fileobj=probe, path_in_repo=".preflight",
                    repo_id=repo, repo_type="dataset",
                    commit_message="backup preflight")
    back = hf_hub_download(repo, ".preflight", repo_type="dataset", token=tok,
                           cache_dir=_os.path.join(scratch, "pf"))
    got = sha256(back)
    _os.remove(probe)
    shutil.rmtree(_os.path.join(scratch, "pf"), ignore_errors=True)
    if got != want:
        sys.exit(f"BACKUP PREFLIGHT FAILED: uploaded {want}, got back {got}. "
                 f"Refusing to fetch — the point of the preflight is that we "
                 f"find this out now and not after 110 GB.")
    print("backup   preflight OK: uploaded, downloaded back, sha256 matched")


class BinnedChunk:
    """Writes ONE month of binned daily slices, appended a sub-request at a time.

    ONE BINNING DEFINITION. `bin_plan` / `bin_slice` come from
    `ml/aggregate_cadence.py` — the same functions that produced family 4 —
    so the global tensor's NA sub-block is not a second opinion about what a
    0.25 degree cell is. The masked-array handling is the two lines from that
    module's own read loop, verbatim, for the same reason.

    WHY THIS IS AN APPENDER AND NOT A FILE-TO-FILE FUNCTION. It was written
    as one, and a global month OOM-killed the sandbox at 5.95 GB RSS
    (measured 2026-09-03, cgroup kill during `cm.subset`) — the memory went
    on the DOWNLOAD, before any binning happened, because a global 1/12
    degree month is 4.4 GB of float32 and the toolbox materialises the
    request. Splitting the *request* is therefore the fix, not splitting the
    read: `--subset-days 1` fetches a day (70.6 MB, ~13 s), bins it, appends
    it and deletes it — 1.91 GB peak RSS over a whole real month. A guard
    that only bounded the binner would have bounded the cheap half.

    The output records `bin_deg`, `grid_align` and `source_dataset` as
    attributes so a builder can refuse a mismatched grid at dispatch instead
    of at use (ml/CLAUDE.md section 0.3); `time` is copied with its units and
    calendar untouched, because `aggregate_cadence.py` parses that string.
    Every appended part is checked against the first part's axes, so a
    silently different sub-request is a loud failure rather than a month
    whose second half is on another grid.
    """

    def __init__(self, dst, bin_deg, align, dataset_id, window):
        self.dst, self.bin_deg, self.align = dst, bin_deg, align
        self.dataset_id, self.window = dataset_id, window
        self.o = None
        self.plan = None
        self.n = 0

    def _open(self, d, lat, lon):
        import numpy as np
        import netCDF4 as nc
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from aggregate_cadence import bin_plan
        self.plan = bin_plan(lat, lon, self.bin_deg, self.align)
        self.lat, self.lon = lat, lon
        o = nc.Dataset(self.dst, "w", format="NETCDF4")
        o.createDimension("time", None)              # appended part by part
        o.createDimension("latitude", self.plan["nlat"])
        o.createDimension("longitude", self.plan["nlon"])
        self.ot = o.createVariable("time", "f8", ("time",))
        t = d.variables["time"]
        for attr in ("units", "calendar", "standard_name", "axis"):
            if hasattr(t, attr):
                setattr(self.ot, attr, getattr(t, attr))
        olat = o.createVariable("latitude", "f8", ("latitude",))
        olat.units, olat.standard_name = "degrees_north", "latitude"
        olat[:] = self.plan["lat"]
        olon = o.createVariable("longitude", "f8", ("longitude",))
        olon.units, olon.standard_name = "degrees_east", "longitude"
        olon[:] = self.plan["lon"]
        self.ov = {}
        for v in VARIABLES:
            self.ov[v] = o.createVariable(
                v, "f4", ("time", "latitude", "longitude"),
                zlib=True, complevel=4, fill_value=np.float32(np.nan))
            sv = d.variables[v]
            for attr in ("units", "long_name", "standard_name"):
                if hasattr(sv, attr):
                    setattr(self.ov[v], attr, getattr(sv, attr))
        o.bin_deg = float(self.bin_deg)
        o.grid_align = str(self.align)
        o.source_dataset = str(self.dataset_id)
        o.source_window = str(self.window)
        o.source_grid = f"{len(lat)}x{len(lon)} at 1/12 deg"
        o.source_lat_range = f"{float(lat.min())}..{float(lat.max())}"
        o.source_lon_range = f"{float(lon.min())}..{float(lon.max())}"
        o.binning = ("ml/aggregate_cadence.py bin_plan/bin_slice — mean of "
                     "the finite source cells nearest each target point, NaN "
                     "where none; no interpolation")
        o.written_by = "ml/fetch_glorys_daily.py (E-070)"
        self.o = o

    def append(self, src):
        """Bin every slice of `src` and append it to the month."""
        import numpy as np
        import netCDF4 as nc
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from aggregate_cadence import bin_slice
        d = nc.Dataset(src)
        lat = np.asarray(d.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(d.variables["longitude"][:], dtype=np.float64)
        if self.o is None:
            self._open(d, lat, lon)
        elif not (np.array_equal(lat, self.lat) and np.array_equal(lon, self.lon)):
            d.close()
            raise RuntimeError(
                f"{os.path.basename(src)} arrived on a different source grid "
                f"({len(lat)}x{len(lon)} vs {len(self.lat)}x{len(self.lon)}) "
                f"— refusing to write two grids into one chunk")
        t = d.variables["time"]
        for i in range(len(t)):
            self.ot[self.n] = float(t[i])
            for v in VARIABLES:
                var = d.variables[v]
                sl = var[i, 0] if var.ndim == 4 else var[i]
                arr = np.ma.filled(
                    np.ma.masked_invalid(sl.astype(np.float32)), np.nan)
                self.ov[v][self.n] = bin_slice(arr, self.plan)
            self.n += 1
        d.close()

    def close(self):
        if self.o is None:
            raise RuntimeError("nothing was appended — refusing to write an "
                               "empty chunk that would look complete")
        self.o.close()
        self.o = None
        return self.plan


def bin_chunk(src, dst, bin_deg, align, dataset_id, window):
    """Bin one already-downloaded chunk. The single-part case of `BinnedChunk`."""
    w = BinnedChunk(dst, bin_deg, align, dataset_id, window)
    w.append(src)
    return w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1993-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache", "glorys_daily"))
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--window", default="na", choices=sorted(WINDOWS),
                    help="'na' is the family-3 window (100 W..20 E, 0..70 N) "
                         "and the default, so E-034's pull is unchanged; "
                         "'global' is GLORYS12's full extent (E-070).")
    ap.add_argument("--bin-deg", type=float, default=0.0,
                    help="bin every daily slice onto a regular grid of this "
                         "size in degrees BEFORE storing it (E-070: 0.25 for "
                         "the global pull — 800 GB of 1/12 deg dailies do not "
                         "belong on the Hub and nothing reads them). "
                         "0 = store the native grid.")
    ap.add_argument("--subset-days", type=int, default=1,
                    help="days per cm.subset request when --bin-deg is set. "
                         "1 (the default) is what keeps peak RSS ~1.2 GB: a "
                         "whole global month in one request was OOM-killed at "
                         "5.95 GB on a 7 GB box. Ignored without --bin-deg, "
                         "where E-034's one-request-per-month path stands.")
    ap.add_argument("--grid-align", default="point", choices=("point", "edge"),
                    help="'point' puts samples ON multiples of --bin-deg, "
                         "which is the family-3 grid and makes the NA window "
                         "an exact sub-block of the global one.")
    ap.add_argument("--months", type=int, default=0,
                    help="stop after N month-chunks (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="refuse to start below this much free disk")
    ap.add_argument("--hf-repo", default="earth-tensors",
                    help="Hugging Face dataset repo for the running backup; "
                         "namespace resolved from the token. '' disables, "
                         "which you should not do — see --no-backup.")
    ap.add_argument("--no-backup", action="store_true",
                    help="fetch without mirroring. Prints a loud warning: a "
                         "110 GB pull with no backup is one dead box away "
                         "from being done twice.")
    ap.add_argument("--report-mem", action="store_true",
                    help="print peak RSS at the end — this is how the "
                         "'a global month never has to be resident' claim "
                         "stays a measurement (same flag as "
                         "ml/aggregate_cadence.py)")
    ap.add_argument("--keep-local", action="store_true",
                    help="keep chunks on local disk after they are backed up "
                         "(default: delete, so the pull needs ~1 chunk of "
                         "disk rather than 110 GB)")
    a = ap.parse_args()

    chunks = list(months_between(a.start, a.end))
    if a.months:
        chunks = chunks[:a.months]
    ndays = sum(calendar.monthrange(y, m)[1] for y, m in chunks)

    win = WINDOWS[a.window]
    folder = hub_folder(a.window, a.bin_deg)

    # Size the job against the disk BEFORE spending anything on it: a guard
    # that depends only on the inputs must fire while the inputs are all it
    # has cost (ml/CLAUDE.md section 0.3). NA is 1440x840 cells at 1/12
    # degree; the globe is 4320x2041, i.e. 7.29x. 4 variables, float32, one
    # depth level.
    cells = CELLS[a.window]
    per_day = cells * len(VARIABLES) * 4
    est_gb = per_day * ndays / 1e9
    # MEASURED, not modelled — see MEAS_GB. NetCDF compression buys ~2x over
    # the raw arithmetic above. Both numbers are printed because the raw one
    # bounds peak decompressed memory and the measured one bounds the disk.
    meas_gb = MEAS_GB[a.window] * len(chunks)
    print(f"dataset   {a.dataset}")
    print(f"window    {win['minimum_longitude']}..{win['maximum_longitude']} E, "
          f"{win['minimum_latitude']}..{win['maximum_latitude']} N "
          f"({cells:,} cells at 1/12 deg)")
    print(f"span      {a.start}..{a.end} = {len(chunks)} month-chunks, {ndays:,} days")
    print(f"variables {', '.join(VARIABLES)} (surface, depth 0-1 m)")
    print(f"estimate  ~{est_gb:.1f} GB uncompressed · "
          f"~{meas_gb:.0f} GB on disk "
          f"(measured {MEAS_GB[a.window]*1000:.0f} MB/month-chunk)")
    if a.bin_deg:
        # What is STORED is the binned grid, so the number that matters for
        # the Hub is this one, not the wire figure above. The target axes are
        # derived with `aggregate_cadence.axis_for` over the SERVED extent —
        # the same function the binner itself will call — rather than from the
        # requested bounds, which differ from the served ones by one cell at
        # the dateline and would print a grid nobody will ever see.
        b_lat, b_lon = binned_shape(a.window, a.bin_deg, a.grid_align)
        store_gb = MEAS_BINNED_GB[a.window] * len(chunks)
        raw_day = b_lat * b_lon * len(VARIABLES) * 4 / 1e9
        print(f"binned    {a.bin_deg} deg, {a.grid_align}-aligned "
              f"({b_lat}x{b_lon}) · ~{raw_day*ndays:.1f} GB uncompressed · "
              f"~{store_gb:.0f} GB stored "
              f"(measured {MEAS_BINNED_GB[a.window]*1000:.0f} MB/month-chunk)")

    os.makedirs(a.out, exist_ok=True)
    st = os.statvfs(a.out)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    print(f"disk      {free_gb:.1f} GB free at {a.out}")
    if a.dry_run:
        have = sum(1 for y, m in chunks
                   if os.path.exists(os.path.join(
                       a.out, chunk_name(a.window, a.bin_deg, y, m))))
        if folder:
            print(f"hub path  {folder}"
                  f"{chunk_name(a.window, a.bin_deg, *chunks[0])}")
        print(f"resume    {have}/{len(chunks)} chunks already present")
        print("\n--dry-run: nothing fetched.")
        if a.bin_deg:
            # The totals above are what the ARCHIVE would weigh; what the
            # runner needs is one wire chunk plus one binned chunk at a time,
            # because each is deleted the moment it is restore-verified. That
            # is the whole reason this fits a GitHub-hosted runner.
            peak = MEAS_GB[a.window] + MEAS_BINNED_GB[a.window]
            print(f"NOTE: chunks are deleted after upload, so the pull needs "
                  f"~{peak:.1f} GB at a time (one wire chunk + one binned "
                  f"chunk), not the {meas_gb:.0f} GB total.")
        elif free_gb < meas_gb:
            print(f"NOTE: {meas_gb:.0f} GB needed vs {free_gb:.1f} GB free — this "
                  f"must run where the disk is (E-034 section 5).")
        return
    if free_gb < a.min_free_gb:
        sys.exit(f"refusing to start: {free_gb:.1f} GB free < --min-free-gb "
                 f"{a.min_free_gb}. Fetching into a full disk is how a box "
                 f"goes metrics-blind (ml/CLAUDE.md section 7).")

    if not os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"):
        sys.exit("set COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD "
                 "(claude/copernicus-marine-access.md)")
    try:
        import copernicusmarine as cm
    except ImportError:
        sys.exit("pip install copernicusmarine")

    api = repo = tok = None
    hf_have = set()
    if a.no_backup:
        print("::warning:: --no-backup: a 110 GB pull with no mirror is one "
              "dead box away from being done twice")
    else:
        api, repo, tok = hf_connect(a.hf_repo)
        print(f"backup   https://huggingface.co/datasets/{repo}")
        hf_preflight(api, repo, tok, a.out)
        try:
            hf_have = {f for f in api.list_repo_files(repo, repo_type="dataset")}
        except Exception:                            # noqa: BLE001
            pass
        if hf_have:
            # Count only THIS family's chunks: the dataset holds E-034's NA
            # chunks in the root and E-070's binned global ones under
            # daily025_global/, and a tally that mixes them would say a global
            # pull was nearly done on the strength of the other family.
            mine = {f for f in hf_have
                    if f in {folder + chunk_name(a.window, a.bin_deg, y, m)
                             for y, m in chunks}}
            print(f"resume   {len(mine)}/{len(chunks)} of this run's chunks "
                  f"already backed up ({len(hf_have)} file(s) in the dataset) "
                  f"— those are skipped")

    done = fail = skip = 0
    for y, m in chunks:
        fname = chunk_name(a.window, a.bin_deg, y, m)
        rpath = folder + fname                       # '' folder = the root
        out = os.path.join(a.out, fname)
        # The raw subset is a scratch file when binning, deleted the moment
        # its slices are in the binned chunk. It must still END IN `.nc`:
        # `cm.subset` appends `.nc` to an output_filename it does not
        # recognise, which sends the file somewhere the cleanup does not look.
        raw = os.path.join(a.out, "_raw_" + fname) if a.bin_deg else out
        # HF is the resume source, not the local disk: a box that dies takes
        # its disk with it, and the whole point of backing up per chunk is
        # that the next box starts where this one stopped.
        if rpath in hf_have:
            skip += 1
            continue
        if os.path.exists(out) and os.path.getsize(out) > 0 and a.keep_local:
            skip += 1
            continue
        last = calendar.monthrange(y, m)[1]
        try:
            t_fetch = dt.datetime.now()
            wire_mb = 0.0
            secs_bin = 0.0
            if not a.bin_deg:
                # E-034's path, untouched: one request for the whole month.
                cm.subset(dataset_id=a.dataset, variables=VARIABLES,
                          start_datetime=f"{y}-{m:02d}-01",
                          end_datetime=f"{y}-{m:02d}-{last:02d}",
                          minimum_depth=0, maximum_depth=1,
                          output_filename=raw, **win)
                wire_mb = os.path.getsize(raw) / 1e6
                extra = ""
            else:
                # SPLIT THE REQUEST, not just the read. A whole global month
                # in one `cm.subset` peaked at 5.95 GB and was OOM-killed on
                # a 7 GB box (2026-09-03); a day is 70.6 MB and 1.16 GB peak.
                writer = BinnedChunk(out, a.bin_deg, a.grid_align,
                                     a.dataset, a.window)
                d0 = 1
                while d0 <= last:
                    d1 = min(d0 + a.subset_days - 1, last)
                    cm.subset(dataset_id=a.dataset, variables=VARIABLES,
                              start_datetime=f"{y}-{m:02d}-{d0:02d}T00:00:00",
                              end_datetime=f"{y}-{m:02d}-{d1:02d}T23:59:59",
                              minimum_depth=0, maximum_depth=1,
                              output_filename=raw, disable_progress_bar=True,
                              **win)
                    wire_mb += os.path.getsize(raw) / 1e6
                    t_bin = dt.datetime.now()
                    writer.append(raw)
                    secs_bin += (dt.datetime.now() - t_bin).total_seconds()
                    os.remove(raw)
                    d0 = d1 + 1
                plan = writer.close()
                if writer.n != last:
                    raise RuntimeError(
                        f"{fname}: {writer.n} daily slice(s) for a "
                        f"{last}-day month — the sub-requests did not cover "
                        f"it, and a short month would look complete forever")
                extra = (f" -> {writer.n}x{plan['nlat']}x{plan['nlon']} at "
                         f"{a.bin_deg} deg, binned in {secs_bin:.0f}s")
            secs_fetch = (dt.datetime.now() - t_fetch).total_seconds()
            sz = os.path.getsize(out) / 1e6
            if api is not None:
                # PUBLISH WHEN THE ARTEFACT EXISTS, not when the job ends
                # (ml/CLAUDE.md §5.20) — and verify the restore, because an
                # upload returning 200 is not evidence the bytes come back.
                src = sha256(out)
                api.upload_file(path_or_fileobj=out, path_in_repo=rpath,
                                repo_id=repo, repo_type="dataset",
                                commit_message=f"glorys daily {y}-{m:02d}")
                from huggingface_hub import hf_hub_download
                back = hf_hub_download(repo, rpath, repo_type="dataset",
                                       token=tok,
                                       cache_dir=os.path.join(a.out, "vf"))
                if sha256(back) != src:
                    shutil.rmtree(os.path.join(a.out, "vf"), ignore_errors=True)
                    raise RuntimeError(f"{rpath} restored with a DIFFERENT "
                                       f"sha256 — backup not trustworthy")
                shutil.rmtree(os.path.join(a.out, "vf"), ignore_errors=True)
                if not a.keep_local:
                    os.remove(out)
            done += 1
            print(f"  {y}-{m:02d}: {sz:,.0f} MB"
                  + (f" (wire {wire_mb:,.0f} MB, {secs_fetch:.0f}s total"
                     f"{extra})" if a.bin_deg else "")
                  + ("" if api is None else " · backed up + restore-verified")
                  + f" ({done} fetched, {skip} skipped)", flush=True)
        except Exception as e:                       # noqa: BLE001
            # Say WHY it gave up — best effort is a promise about delivery,
            # never about reporting (ml/CLAUDE.md section 4.6).
            for p in {out, raw}:
                if os.path.exists(p):
                    os.remove(p)
            fail += 1
            print(f"  ::warning:: {y}-{m:02d} FAILED, chunk removed: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            if fail >= 5 and done == 0:
                sys.exit("five consecutive failures with nothing fetched — "
                         "stopping rather than hammering the service")
    print(f"\n{done} fetched · {skip} already present · {fail} failed")
    if a.bin_deg:
        # The chunks are ALREADY on the target grid, so the pentad reduction
        # runs with no --bin-deg: binning twice would average cell means of
        # cell means and quietly change the weighting.
        print(f"next: python3 ml/aggregate_cadence.py --hf-repo {a.hf_repo} "
              f"--cadence pentad --out <dir>   (no --bin-deg: these chunks "
              f"are already {a.bin_deg} deg, {a.grid_align}-aligned)")
    else:
        print(f"next: python3 ml/aggregate_cadence.py --in {a.out} "
              f"--cadence pentad   (daily stays the source of truth)")
    if a.report_mem:
        import resource
        print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")


if __name__ == "__main__":
    main()
