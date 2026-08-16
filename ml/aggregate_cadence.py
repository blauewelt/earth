#!/usr/bin/env python3
"""E-034: reduce DAILY fields to a coarser cadence. Pentad is derived, never fetched.

This file is the other half of the decision recorded in
`ml/fetch_glorys_daily.py`: **fetch daily once, derive every coarser cadence
by aggregation.** A 5-day mean is a pure reduction of the daily series, so a
separate pentad fetcher would be a second implementation of one rule — and
two implementations of one rule is the defect that drew a cosine on the
status page for a run that used expdecay, and that put a hand-traced corridor
in the frontend beside the evaluator's real one. There is one downloaded
byte-stream and one aggregation rule, here.

BIN DEFINITION, shared with `ml/build_truth_pentad.py` by construction rather
than by coincidence: fixed 5-day bins counted from 1982-01-01, index =
floor(days_since_epoch / 5). The state axis and the label axis therefore land
in the same bins, which is what makes a pentad label the target of a pentad
state without any re-alignment step.

WHAT IS AGGREGATED, AND HOW — the choice is per quantity, not global:

  uo, vo    -> mean of the VECTOR components, then speed = hypot(mean_u,
               mean_v) downstream. Averaging the SPEED instead would measure
               something different (a mean of magnitudes is not the magnitude
               of the mean) and would inflate quiet bins where the current
               reverses. The tensor's `cur_speed` channel is built from the
               binned components for exactly this reason.
  mlotst    -> mean. Deep-mixing events are what matter and a 5-day mean of a
               daily MLD keeps them; a max would track single-day storms and
               a min would erase the engine room.
  zos       -> mean.
  tau_std   -> NOT aggregable from a daily mean. A within-pentad standard
               deviation must be computed from the dailies directly, which is
               why `--stats std` exists: it emits the sigma over each bin
               rather than the mean of anything.

Bins with fewer than `--min-days` contributing days are written as MISSING,
not as a thinner mean. A 5-day bin holding one day is not a pentad average,
and letting it pass would put a noisier sample beside 3,000 clean ones with
nothing marking it (ml/CLAUDE.md §5.22: never write a number you would have
to caveat later).

---

THREE THINGS THIS FILE DOES DIFFERENTLY SINCE 2026-08-16, ALL OF THEM
BECAUSE THE FIRST VERSION COULD NOT COMPLETE THE JOB IT WAS WRITTEN FOR.

Measured before changing anything (ml/CLAUDE.md §0.3 — check a precondition
where the inputs are all it has cost you): the first version kept one
accumulator per (bin, variable) alive for the WHOLE run and materialised a
single `(nbins, H, W)` stack per variable at the end. One real month —
`glorys_199301.nc`, 7 bins — peaked at **0.93 GB RSS**, which matches the
arithmetic exactly. The full 1993-01..2024-12 archive is **2,339 pentad
bins** at 841x1441, so the same code needs **227 GB of RAM** and writes a
**45.4 GB** npz. It cannot finish, and it would have failed *after* the
~8-hour GLORYS12 pull rather than before it.

1. **BINS ARE STREAMED AND FLUSHED, NOT HELD.** Chunks are read in
   chronological order and so are the days inside a chunk, so a bin closes
   the instant the date crosses out of it. Only *open* bins are resident.
   **Measured over three real months (1993-01..03), peak open bins = 1 and
   peak RSS = 0.16 GB at 0.25 degrees / 0.34 GB on the native grid** — and
   those numbers do not move with the length of the run, which is the whole
   fix for the 227 GB. It is a change of bookkeeping, not of arithmetic, and
   the equality check below pins that.

2. **OPTIONAL SPATIAL BINNING, PER E-034 §2** (`--bin-deg 0.25`). The plan's
   cadence audit specifies the GLORYS channels as "pentad mean of the
   dailies, subsetted to the NA window and **binned to 0.25 degrees**", and
   doing it here rather than downstream is what takes the output from 45.4 GB
   to 5.0 GB — i.e. from "does not fit any disk we have" to "fits".
   **Each DAILY slice is binned spatially first, and only then accumulated in
   time.** That ordering is deliberate: it keeps `cnt` a count of *days*, so
   `--min-days` keeps the exact meaning its docstring above claims. Binning
   after the temporal mean instead would make `cnt` a count of
   source-cell-days (up to 9 per day at 0.25 degrees) and quietly turn a
   guard about temporal coverage into one about spatial coverage.
   The binning itself is the project's standard nearest scatter-binning onto
   a regular grid (`_write_grid` in `scripts/refresh_data.py`), by index
   rather than by interpolation, so no value is invented.

3. **OUTPUT IS ONE MEMMAPPED .npy PER VARIABLE, PLUS AN INDEX.** The old
   single `.npz` had to exist in RAM in full before `savez_compressed` could
   see it. `np.lib.format.open_memmap` writes each flushed bin straight to
   disk at its own row, so peak memory no longer depends on the length of the
   run. Nothing consumed `pentad_mean.npz` yet — `build_family4.py` is still
   step 4 of E-034 §6 — so this changes no reader. Load with:

       idx = np.load(f"{out}/index.npz")
       uo  = np.load(f"{out}/pentad_mean_uo.npy", mmap_mode="r")

4. **THE DAILIES CAN COME FROM HUGGING FACE** (`--hf-repo earth-tensors`).
   `fetch_glorys_daily.py` deletes each chunk once it is backed up and
   restore-verified, precisely so the pull needs ~290 MB of disk instead of
   110 GB — which means that after the pull there are no local dailies to
   aggregate. This mode downloads one month, folds it in, and deletes it,
   holding the same one-chunk disk footprint. HF stays the single resume
   source for both halves of E-034.

**The equality invariant.** With no `--bin-deg`, this file must produce
bit-identical arrays to the pre-streaming implementation — same means, same
NaNs, same dropped bins. `tests/test_e034_aggregate.py` pins that against a
real month rather than a synthetic one, because the thing most likely to
break is bin straddling at a month boundary and a synthetic fixture is where
that gets accidentally made easy.

Run:
  python3 ml/aggregate_cadence.py --in ml/cache/glorys_daily --cadence pentad \\
      --out ml/cache/glorys_pentad --bin-deg 0.25
  python3 ml/aggregate_cadence.py --hf-repo earth-tensors --cadence pentad \\
      --out ml/cache/glorys_pentad --bin-deg 0.25       # dailies streamed from HF
  python3 ml/aggregate_cadence.py --in ... --cadence daily --out ...   # passthrough
"""
import argparse
import datetime as dt
import glob
import os
import re
import resource
import shutil
import sys

import numpy as np

EPOCH = dt.date(1982, 1, 1)
CADENCE_DAYS = {"daily": 1, "pentad": 5}
MEAN_VARS = ("uo", "vo", "mlotst", "zos")
MONTH_RE = re.compile(r"_(\d{4})(\d{2})\.nc$")


def bin_index(d, days):
    return (d - EPOCH).days // days


def bin_start(i, days):
    return EPOCH + dt.timedelta(days=int(i) * days)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# --------------------------------------------------------------------------
# spatial binning
# --------------------------------------------------------------------------

def bin_plan(lat, lon, deg):
    """Index map from the source grid onto a regular `deg` grid.

    Nearest scatter-binning by INDEX (the `_write_grid` convention), so a
    target cell is the mean of the source cells whose centres fall in it and
    nothing is interpolated into existence. Edges are snapped to whole
    multiples of `deg` so that two runs over different subsets of the same
    archive land on the same grid — a grid defined by its own data's min/max
    would drift with the subset and silently make two tensors incomparable.
    """
    lat0 = np.floor(float(lat.min()) / deg) * deg
    lon0 = np.floor(float(lon.min()) / deg) * deg
    nlat = int(np.ceil((float(lat.max()) - lat0) / deg))
    nlon = int(np.ceil((float(lon.max()) - lon0) / deg))
    ilat = np.clip(((lat - lat0) / deg).astype(np.int64), 0, nlat - 1)
    ilon = np.clip(((lon - lon0) / deg).astype(np.int64), 0, nlon - 1)
    flat = (ilat[:, None] * nlon + ilon[None, :]).ravel()
    centres_lat = lat0 + (np.arange(nlat) + 0.5) * deg
    centres_lon = lon0 + (np.arange(nlon) + 0.5) * deg
    return dict(flat=flat, nlat=nlat, nlon=nlon, ncell=nlat * nlon,
                lat=centres_lat, lon=centres_lon, deg=deg)


def bin_slice(arr, plan):
    """Mean of the finite source cells in each target cell; NaN where none."""
    a = arr.ravel()
    ok = np.isfinite(a)
    s = np.bincount(plan["flat"][ok], weights=a[ok].astype(np.float64),
                    minlength=plan["ncell"])
    n = np.bincount(plan["flat"][ok], minlength=plan["ncell"])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out.reshape(plan["nlat"], plan["nlon"]).astype(np.float32)


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

class LocalSource:
    """Chunks already on disk. `get` is a no-op; nothing is ever deleted."""

    def __init__(self, src):
        self.paths = {os.path.basename(p): p
                      for p in sorted(glob.glob(os.path.join(src, "*.nc")))}
        if not self.paths:
            sys.exit(f"no .nc files in {src}")
        self.names = sorted(self.paths)
        print(f"source   {src} ({len(self.names)} chunk(s))")

    def get(self, name):
        return self.paths[name]

    def release(self, name):
        pass


class HFSource:
    """Stream the dailies back from the Hub, ONE chunk of disk at a time.

    Mirrors `fetch_glorys_daily.py`'s pattern exactly: the Hub is the resume
    source and the local disk is a window onto it, never the archive. The
    download therefore happens inside the read loop and the cache is torn
    down after each chunk — materialising the list eagerly would restore the
    110 GB footprint the fetcher exists to avoid.
    """

    def __init__(self, repo_name, scratch):
        tok = os.environ.get("HF_TOKEN") or (
            open("/home/claude/.hf_token").read().strip()
            if os.path.exists("/home/claude/.hf_token") else "")
        if not tok:
            sys.exit("no HF_TOKEN (env or /home/claude/.hf_token) — see "
                     "claude/huggingface-access.md")
        from huggingface_hub import HfApi
        self.tok = tok
        api = HfApi(token=tok)
        self.repo = (repo_name if "/" in repo_name
                     else f"{api.whoami()['name']}/{repo_name}")
        self.names = sorted(
            f for f in api.list_repo_files(self.repo, repo_type="dataset")
            if f.endswith(".nc"))
        if not self.names:
            sys.exit(f"no .nc files in "
                     f"https://huggingface.co/datasets/{self.repo}")
        self.cache = os.path.join(scratch, "_hf")
        print(f"source   https://huggingface.co/datasets/{self.repo} "
              f"({len(self.names)} chunk(s), streamed one at a time)")

    def get(self, name):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, name, repo_type="dataset",
                               token=self.tok, cache_dir=self.cache)

    def release(self, name):
        shutil.rmtree(self.cache, ignore_errors=True)


def plan_range(names, days):
    """Bin range covered by these chunks, from their month stamps.

    Parsed from the filename because that is `fetch_glorys_daily.py`'s own
    contract (`glorys_YYYYMM.nc`) and it costs no read — but every bin met
    while reading is asserted to fall inside this range, so a name that lies
    is a loud failure and never a silent mis-index.

    No slack is added, and none is needed: every bin touched by any day in
    [first day, last day] is by definition between the bin containing the
    first day and the bin containing the last, straddles included. Adding
    slack would pad the output with all-NaN rows that mean "no such bin"
    while looking exactly like "bin observed, all cells missing".
    """
    months = []
    for n in names:
        m = MONTH_RE.search(n)
        if not m:
            return None
        months.append((int(m.group(1)), int(m.group(2))))
    if not months:
        return None
    import calendar
    y0, m0 = min(months)
    y1, m1 = max(months)
    first = dt.date(y0, m0, 1)
    last = dt.date(y1, m1, calendar.monthrange(y1, m1)[1])
    return bin_index(first, days), bin_index(last, days)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=None,
                    help="directory of daily .nc chunks")
    ap.add_argument("--hf-repo", default=None,
                    help="Hugging Face dataset repo to stream the dailies "
                         "from instead of --in (one chunk of disk at a time)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cadence", default="pentad", choices=sorted(CADENCE_DAYS))
    ap.add_argument("--stats", default="mean", choices=("mean", "std"))
    ap.add_argument("--min-days", type=int, default=0,
                    help="bins with fewer contributing days are dropped; "
                         "default 0 = cadence-appropriate (pentad: 3)")
    ap.add_argument("--vars", default=",".join(MEAN_VARS))
    ap.add_argument("--bin-deg", type=float, default=0.0,
                    help="bin each daily slice onto a regular grid of this "
                         "size in degrees before accumulating (E-034 §2: "
                         "0.25 for the GLORYS channels). 0 = native grid.")
    ap.add_argument("--report-mem", action="store_true",
                    help="print peak RSS at the end — this is how the "
                         "streaming claim stays a measurement")
    a = ap.parse_args()

    if bool(a.src) == bool(a.hf_repo):
        sys.exit("give exactly one of --in <dir> or --hf-repo <repo>")

    import netCDF4 as nc

    days = CADENCE_DAYS[a.cadence]
    min_days = a.min_days or (3 if days == 5 else 1)
    want = [v for v in a.vars.split(",") if v]
    os.makedirs(a.out, exist_ok=True)

    source = (LocalSource(a.src) if a.src else HFSource(a.hf_repo, a.out))
    names = source.names

    rng = plan_range(names, days)
    if rng is None:
        sys.exit("cannot derive the bin range from the chunk names "
                 "(expected the fetcher's glorys_YYYYMM.nc contract)")
    b_lo, b_hi = rng
    nbins = b_hi - b_lo + 1

    print(f"{len(names)} daily file(s) -> {a.cadence} bins "
          f"({days} d from {EPOCH}), stat={a.stats}, min_days={min_days}")
    print(f"bins     {nbins:,} planned, {bin_start(b_lo, days)} .. "
          f"{bin_start(b_hi, days)}")

    # ---- state -----------------------------------------------------------
    acc, acc2, cnt = {}, {}, {}        # (bin, var) -> array; ONLY open bins
    out_arr = {}                       # var -> memmap
    plan = None                        # spatial binning, None = native grid
    inited = False                     # output allocated (once, on chunk 0)
    written = set()
    kept = dropped = 0
    max_open = 0

    def ensure_out(shape):
        for v in want:
            p = os.path.join(a.out, f"{a.cadence}_{a.stats}_{v}.npy")
            m = np.lib.format.open_memmap(
                p, mode="w+", dtype=np.float32, shape=(nbins,) + shape)
            m[:] = np.nan
            out_arr[v] = m

    def flush(b):
        """Write bin `b` out and free it. Flush, THEN mark (ml/CLAUDE.md §5.21)."""
        nonlocal kept, dropped
        row = b - b_lo
        n_any = 0
        for v in want:
            key = (b, v)
            if key not in acc:
                continue
            n = cnt[key]
            n_any = max(n_any, int(n.max()))
            good = n >= min_days
            with np.errstate(invalid="ignore"):
                mu = acc[key] / np.maximum(n, 1)
                if a.stats == "mean":
                    val = np.where(good, mu, np.nan)
                else:
                    var_ = acc2[key] / np.maximum(n, 1) - mu ** 2
                    val = np.where(good, np.sqrt(np.maximum(var_, 0)), np.nan)
            out_arr[v][row] = val
            del acc[key], acc2[key], cnt[key]
        if n_any >= min_days:
            kept += 1
        else:
            dropped += 1
        written.add(b)

    # ---- stream ----------------------------------------------------------
    for fi, name in enumerate(names):
        path = source.get(name)
        d = nc.Dataset(path)
        t = d.variables["time"]
        units = t.units
        base = units.split("since", 1)[1].strip().split()[0]
        y, m, dd = (int(x) for x in base.replace("/", "-").split("-"))
        t0 = dt.date(y, m, dd)
        per = 24.0 if units.strip().lower().startswith("hours") else 1.0

        # `inited`, NOT `plan is None`: with no --bin-deg the plan stays None
        # for the whole run, so keying the one-time setup off it re-created
        # the output memmap (mode="w+", filled with NaN) once per chunk and
        # erased every bin written so far. Caught by
        # tests/test_e034_aggregate.py, which is a THREE-month fixture for
        # exactly this reason — a one-file test cannot see it.
        if not inited:
            inited = True
            if a.bin_deg:
                plan = bin_plan(d.variables["latitude"][:].astype(np.float64),
                                d.variables["longitude"][:].astype(np.float64),
                                a.bin_deg)
                print(f"grid     {plan['nlat']}x{plan['nlon']} at "
                      f"{a.bin_deg} deg (from "
                      f"{len(d.variables['latitude'])}x"
                      f"{len(d.variables['longitude'])} native)")
                ensure_out((plan["nlat"], plan["nlon"]))
            else:
                shp = (len(d.variables["latitude"]),
                       len(d.variables["longitude"]))
                print(f"grid     {shp[0]}x{shp[1]} native (no --bin-deg)")
                ensure_out(shp)

        last_date = None
        for i in range(len(t)):
            date = t0 + dt.timedelta(days=float(t[i]) / per)
            last_date = date
            b = bin_index(date, days)
            if not (b_lo <= b <= b_hi):
                sys.exit(f"{name}: {date} lands in bin {b}, outside the "
                         f"planned range {b_lo}..{b_hi} — the chunk name "
                         f"disagrees with its own time axis")
            if b in written:
                sys.exit(f"{name}: {date} belongs to bin {b}, already "
                         f"flushed — the chunks are not in time order and "
                         f"the streaming assumption does not hold")
            # Days inside a chunk are in time order too, so a bin closes the
            # moment the date crosses out of it — no need to wait for the end
            # of the file. Flushing here is what keeps the resident set at
            # one bin instead of one month's worth.
            for ob in sorted({x for x, _ in acc}):
                if ob < b:
                    flush(ob)
            for v in want:
                var = d.variables[v]
                sl = var[i, 0] if var.ndim == 4 else var[i]
                arr = np.ma.filled(np.ma.masked_invalid(sl.astype(np.float32)),
                                   np.nan)
                if plan is not None:
                    arr = bin_slice(arr, plan)
                key = (b, v)
                if key not in acc:
                    acc[key] = np.zeros(arr.shape, np.float64)
                    acc2[key] = np.zeros(arr.shape, np.float64)
                    cnt[key] = np.zeros(arr.shape, np.int32)
                ok = np.isfinite(arr)
                acc[key][ok] += arr[ok]
                acc2[key][ok] += arr[ok] ** 2
                cnt[key] += ok
        d.close()
        source.release(name)

        # Close every bin no remaining file can reach. Files are in time
        # order, so the next file's FIRST day is the cutoff; at the end of
        # the stream everything closes.
        if fi + 1 < len(names):
            nxt = MONTH_RE.search(names[fi + 1])
            cutoff_bin = (bin_index(dt.date(int(nxt.group(1)),
                                            int(nxt.group(2)), 1), days)
                          if nxt else None)
        else:
            cutoff_bin = None
        open_bins = sorted({b for b, _ in acc})
        max_open = max(max_open, len(open_bins))
        for b in open_bins:
            if cutoff_bin is None or b < cutoff_bin:
                flush(b)
        print(f"  read {name} ({last_date}) · {len(open_bins)} open bin(s) · "
              f"{kept + dropped} written", flush=True)

    for b in sorted({b for b, _ in acc}):
        flush(b)

    # bin_index is the FULL planned range, so row i is bin_index[i] with no
    # lookup — and `has_data` says which of those bins a chunk actually
    # covered. Storing only the written bins would make the two disagree the
    # first time a month is missing from the archive, which is precisely the
    # case where a silent off-by-one would go unnoticed.
    idx = np.arange(b_lo, b_hi + 1, dtype=np.int64)
    has = np.array([b in written for b in idx], bool)
    store = {"bin_index": idx, "has_data": has, "epoch": np.array(str(EPOCH)),
             "cadence_days": np.array(days), "stat": np.array(a.stats),
             "min_days": np.array(min_days),
             "bin_deg": np.array(a.bin_deg or 0.0),
             "vars": np.array(want)}
    if plan is not None:
        store["lat"] = plan["lat"]
        store["lon"] = plan["lon"]
    np.savez(os.path.join(a.out, "index.npz"), **store)
    total = 0
    for v in want:
        out_arr[v].flush()
        total += os.path.getsize(
            os.path.join(a.out, f"{a.cadence}_{a.stats}_{v}.npy"))

    print(f"\n{len(written)} bins · {kept} kept · {dropped} below min_days "
          f"({bin_start(idx[0], days)} .. {bin_start(idx[-1], days)})"
          + ("" if has.all() else
             f" · {int((~has).sum())} planned bin(s) had no chunk"))
    print(f"wrote {a.out}/{a.cadence}_{a.stats}_<var>.npy + index.npz "
          f"({total/1e6:,.0f} MB, {len(want)} variables)")
    print(f"peak open bins {max_open} (streaming: memory does not grow with "
          f"the length of the run)")
    if a.report_mem:
        print(f"peak RSS {peak_gb():.2f} GB")


if __name__ == "__main__":
    main()
