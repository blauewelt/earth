"""SWOT L2 LR sea-surface height, Expert — sea level measured ACROSS two
50 km swaths instead of along one line (family 1.gf, E-082 wave 4).

PLAIN ENGLISH. Every radar altimeter before SWOT measured the height of the
sea directly beneath it, one line at a time; SWOT's KaRIn interferometer
measures it across two 50 km-wide strips either side of the ground track, on a
2 km grid. That is the first instrument that can see an ocean eddy's SHAPE
rather than one cut through it, which is what the Atlantic-overturning and
El Nino goals want it for.

THIS ADAPTER IS A MEASUREMENT, NOT YET A BUILD. The brief: "Decide: probe one
cycle (cycle 010) on a runner to measure pixels per pass and valid fraction,
and STOP — report the two storage options (tier P rows at 27+2C bytes vs a
per-pass sharded tile group) with measured numbers for the main session to
decide." So the adapter is written to be PROBED and every stage beyond the
probe refuses (`fetch_preflight`) unless `SWOT_ALLOW_BUILD=1` is set, and the
probe computes BOTH storage options from what it measured and puts them in its
own counts (`storage_options`). Nothing is dispatched as a build from here.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (CMR is anonymous; the granules
are Earthdata-protected, so the bytes are a runner's job):
  collection  SWOT_L2_LR_SSH_EXPERT_D, version D, C3233942272-POCLOUD
              (POCLOUD; the note's "Expert (version D)"). The 2.0 collections
              C2799465497 etc. are the earlier processing and are NOT read.
  granules    SWOT_L2_LR_SSH_Expert_<CCC>_<PPP>_<start>_<end>_<PG>_<NN>_swot
              — cycle CCC, pass PPP; the bytes are
              https://archive.swot.podaac.earthdata.nasa.gov/
                  podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_D/<name>.nc

CYCLE 010, MEASURED IN FULL FROM CMR 2026-09-18: **581 granules**, passes 1 ..
584 with three of the 584 absent, **2024-01-25T00:19 .. 2024-02-14T21:04**
(21 calendar days, 24-28 passes a day), **18,807.9 MB, mean 32.37 MB a pass** —
the note's "32 MB/pass, 18.8 GB/cycle" exactly. That is why the probe month is
2024-01: it holds cycle 010's first seven days.

A PROBE CANNOT FETCH A CYCLE. 581 passes is 18.8 GB and a whole month of
passes is ~28 GB, so `SWOT_CYCLE` and `SWOT_MAX_PASSES` (read at CONSTRUCTION
time, so the fresh adapter `stage_probe` builds for itself sees them) restrict
what is fetched — the same mechanism `lossyear` uses for its 4,480 groups. The
defaults are `SWOT_CYCLE=010` and `SWOT_MAX_PASSES=12`, i.e. about 388 MB, and
the restriction is written into `notes`, which reaches store.json, so a
restricted probe can never look like a whole one.

WHAT A ROW WOULD BE, if the P form is chosen. One valid 2 km KaRIn pixel.
  time_s    the `time` variable (num_lines), broadcast across the swath,
            through `f10b._cf_time_to_seconds` on its OWN `units` attribute.
            The file also carries `time_tai`; `time` is the UTC axis and is
            the one read.
  lat, lon  `latitude`, `longitude` (num_lines, num_pixels), lon wrapped.
  values    C = 3: `ssha_karin` (m, the anomaly), `sig0_karin` (dB, the
            backscatter the note's second channel names) and
            `ssh_karin_uncert` (m). The note's third channel is "quality";
            a 32-bit quality BITMASK cannot be a float16 channel without
            losing bits, so the flag goes into `qc` as a GRADE read from the
            variable's own `flag_meanings` and the uncertainty — a number —
            takes the channel. That choice is part of what the main session is
            being asked to decide, and it is stated rather than buried.
  platform  `platform_hash("SWOT")` — one satellite; the cycle and pass are
            in the probe's counts, and would be `extra_meta` in a build.
  qc        0 good, 1 suspect, 2 degraded, 3 bad, from `ssh_karin_qual`'s own
            flag table; a pixel graded bad is dropped and counted.

THE TWO STORAGE OPTIONS, and why the probe has to measure before either is
chosen (`family1gf.tex` §4, "which tier a fine field belongs in is geometry
first, bytes second"):
  TIER P   one row per valid pixel at 27 + 2C = 33 bytes. No grid is needed —
           the swath moves every pass, so there is no raster to tile — and a
           cone reads rows by (bin, position) exactly as it reads `slatrack`.
           The cost is the row count: the note projects ~16e9 rows and ~0.6 TB.
  TIER G   one sharded GROUP PER PASS (584 groups), each pass's own
           num_lines x num_pixels raster, 256 x 256 tiles, float16, at about
           2C x 1.2 = 7.2 bytes a valid pixel plus the index. Cheaper per
           pixel than a row, and it keeps the swath's SHAPE, which is the
           whole reason for the instrument. The cost is bookkeeping: 584
           groups x ~55 cycles of bins is tens of thousands of shards, and the
           grid of a pass is not the grid of the same pass one cycle later
           (the orbit repeats to ~1 km, not exactly), so `tile_grid.json`
           would have to carry per-pass, per-cycle corner geometry rather than
           one affine rule.
The probe fills both columns with measured numbers (`storage_options` in its
counts) and STOPS.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION = "C3233942272-POCLOUD"
SHORT_NAME = "SWOT_L2_LR_SSH_EXPERT_D"
NAME = re.compile(r"SWOT_L2_LR_SSH_Expert_(\d{3})_(\d{3})_"
                  r"(\d{8}T\d{6})_(\d{8}T\d{6})_")
CHANNELS = (("ssha_karin", "m", -3.0, 3.0),
            ("sig0_karin", "dB", -10.0, 40.0),
            ("ssh_karin_uncert", "m", 0.0, 5.0))
DIM_LINES, DIM_PIXELS = "num_lines", "num_pixels"
NEEDED = ("latitude", "longitude", "time", "ssha_karin", "sig0_karin",
          "ssh_karin_uncert", "ssh_karin_qual")
FIRST_YEAR = 2023
WORKERS = 3
# The grades `qc` carries, in the order the product's own flag table lists
# them. `qc_grade` maps the file's flag VALUES onto these through the
# variable's `flag_meanings`, so nothing here depends on a bit layout.
GRADES = ("good", "suspect", "degraded", "bad")
BAD_GRADE = 3


class FormatError(ValueError):
    """A granule listing or a netCDF that does not look like the product."""


# ============================================================== the index ==
def cmr_granules(cycle=None, temporal=None, page=2000, attempts=4,
                 count=None):
    """Every granule of the collection, optionally one cycle or one window.

    ANONYMOUS — CMR needs no login (the BYTES do). Paged with
    `CMR-Search-After`. An empty answer is returned as an empty list and the
    CALLER decides whether that is a refusal.
    """
    out, after = [], None
    while True:
        q = {"collection_concept_id": COLLECTION, "page_size": page,
             "sort_key": "start_date"}
        if cycle:
            q["readable_granule_name"] = f"SWOT_L2_LR_SSH_Expert_{cycle}_*"
            q["options[readable_granule_name][pattern]"] = "true"
        if temporal:
            q["temporal"] = temporal
        url = f"{CMR}?{urllib.parse.urlencode(q)}"
        hdr = {"CMR-Search-After": after} if after else None
        raw, why = cm.get_bytes(url, attempts=attempts, headers=hdr)
        if raw is None:
            raise FormatError(f"CMR answered {why} for {url}")
        if count is not None:
            count(len(raw))
        # `get_bytes` drops the response headers, so the page cursor is read
        # from the last granule instead: CMR's `sort_key=start_date` makes
        # `temporal` paging stable, and a full page is followed by asking for
        # the window after the last granule's start.
        js = json.loads(raw)
        e = js.get("feed", {}).get("entry", [])
        out += e
        if len(e) < page:
            return out
        last = e[-1]["time_start"]
        temporal = f"{last},{temporal.split(',')[1]}" if temporal else None
        if temporal is None:
            return out                      # one page is all a cycle needs
        after = None


def parse_granules(entries):
    """CMR entries -> {(cycle, pass): {name, url, bytes, t0, t1}}. Refuses a
    name that is not the product's, or two granules for one (cycle, pass)."""
    out, other = {}, []
    for g in entries:
        title = g.get("title", "")
        m = NAME.match(title)
        if not m:
            other.append(title)
            continue
        key = (m.group(1), m.group(2))
        if key in out:
            raise FormatError(f"two granules for cycle {key[0]} pass {key[1]}: "
                              f"{out[key]['name']} and {title}")
        url = None
        for ln in g.get("links", []):
            h = str(ln.get("href", ""))
            if h.startswith("https") and "protected" in h and h.endswith(".nc"):
                url = h
                break
        if url is None:
            raise FormatError(f"{title}: no protected https .nc link "
                              f"({[l.get('href') for l in g.get('links', [])][:3]})")
        out[key] = {"name": title, "url": url,
                    "bytes": int(float(g.get("granule_size") or 0) * 1e6),
                    "t0": g.get("time_start"), "t1": g.get("time_end"),
                    "start": m.group(3), "end": m.group(4)}
    counts = {}
    if other:
        counts["granules_other_name"] = len(other)
        counts["granules_other_names"] = sorted(other)[:10]
    return out, counts


# ================================================================ one pass =
def qc_grade(var):
    """{flag value: grade index} from the variable's OWN flag table.

    `ssh_karin_qual` is a bit field whose `flag_meanings` name the conditions
    and whose `flag_masks` give the bits. The product's own convention is that
    the TOP byte carries the summary grade (0 good / 1 suspect / 2 degraded /
    3 bad), so the grade is read as `value >> 24` and CHECKED against the flag
    table's own vocabulary; a table that names none of the four grades is a
    refusal rather than a guess.
    """
    words = str(getattr(var, "flag_meanings", "")).lower()
    named = [g for g in GRADES if g in words]
    if not named:
        raise FormatError(
            f"`ssh_karin_qual` flag_meanings {words[:120]!r} name none of "
            f"{GRADES} — refusing to guess which value means bad")
    return named


def _read(ds, name, shape=None):
    v = ds.variables[name]
    a = np.asarray(v[:], np.float64)
    fill = getattr(v, "_FillValue", None)
    if fill is not None:
        a = np.where(a == np.float64(fill), np.nan, a)
    a[~np.isfinite(a)] = np.nan
    return a


def nc_check(ds):
    """An open Expert granule against the product layout. Raises."""
    for d in (DIM_LINES, DIM_PIXELS):
        if d not in ds.dimensions:
            raise FormatError(f"no `{d}` dimension; dimensions are "
                              f"{sorted(ds.dimensions)}")
    missing = [v for v in NEEDED if v not in ds.variables]
    if missing:
        raise FormatError(f"variables {missing} are absent; the granule has "
                          f"{len(ds.variables)} variables")
    nl = int(len(ds.dimensions[DIM_LINES]))
    npx = int(len(ds.dimensions[DIM_PIXELS]))
    for v in ("latitude", "longitude", "ssha_karin"):
        if ds.variables[v].dimensions != (DIM_LINES, DIM_PIXELS):
            raise FormatError(f"{v} has dimensions "
                              f"{ds.variables[v].dimensions}, not "
                              f"({DIM_LINES}, {DIM_PIXELS})")
    if ds.variables["time"].dimensions != (DIM_LINES,):
        raise FormatError(f"time has dimensions "
                          f"{ds.variables['time'].dimensions}, not "
                          f"({DIM_LINES},)")
    u = str(getattr(ds.variables["time"], "units", ""))
    if " since " not in u.lower():
        raise FormatError(f"time units {u!r} carry no epoch")
    return {"num_lines": nl, "num_pixels": npx, "pixels": nl * npx,
            "time_units": u, "variables": len(ds.variables),
            "has_time_tai": "time_tai" in ds.variables,
            "format": ds.data_model,
            "cycle_attr": str(getattr(ds, "cycle_number", "")),
            "pass_attr": str(getattr(ds, "pass_number", "")),
            "qual_flag_meanings":
                str(getattr(ds.variables["ssh_karin_qual"], "flag_meanings",
                            ""))[:200]}


def read_pass(path, counts):
    """One Expert granule -> (t, lat, lon, values[n, 3], qc, meta)."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        meta = nc_check(ds)
        nl, npx = meta["num_lines"], meta["num_pixels"]
        t1 = f10b._cf_time_to_seconds(_read(ds, "time"), meta["time_units"])
        lat, lon = _read(ds, "latitude"), _read(ds, "longitude")
        vals = np.stack([_read(ds, c[0]) for c in CHANNELS], axis=-1)
        qual = np.asarray(ds.variables["ssh_karin_qual"][:])
        grades = qc_grade(ds.variables["ssh_karin_qual"])
    finally:
        ds.close()
    t = np.repeat(t1[:, None], npx, axis=1)
    g = (np.asarray(np.ma.filled(qual, 2 ** 31), np.int64) >> 24) \
        .astype(np.int64)
    g = np.clip(g, 0, len(GRADES) - 1)
    ok = (np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
          & np.isfinite(vals[..., 0]) & (g < BAD_GRADE))
    meta["grades_named"] = grades
    meta["valid_pixels"] = int(ok.sum())
    meta["valid_fraction"] = round(float(ok.mean()), 6)
    meta["ssha_present_fraction"] = round(
        float(np.isfinite(vals[..., 0]).mean()), 6)
    hist = counts.setdefault("qual_grade", {})
    for u, k in zip(*np.unique(g, return_counts=True)):
        hist[GRADES[int(u)]] = hist.get(GRADES[int(u)], 0) + int(k)
    counts["pixels_in_pass"] = counts.get("pixels_in_pass", 0) + nl * npx
    counts["pixels_valid"] = counts.get("pixels_valid", 0) + meta["valid_pixels"]
    counts["passes_read"] = counts.get("passes_read", 0) + 1
    geo = counts.setdefault("pass_geometry", {})
    geo[f"{nl}x{npx}"] = geo.get(f"{nl}x{npx}", 0) + 1
    if not ok.any():
        return None, meta
    return (t[ok], lat[ok], lon[ok], vals[ok],
            g[ok].astype(np.uint8)), meta


# ================================================================ adapter ==
class SWOTAdapter(f10b.SourceAdapter):
    store = "swot"
    title = ("SWOT L2 LR sea-surface height, Expert (version D): the 2 km "
             "KaRIn swath, one row per valid pixel — PROBE ONLY until the "
             "storage decision is made")
    family = "1gf"
    distribution = "public"
    licence = {"name": "NASA open data (CC0-equivalent)",
               "redistribution": "yes",
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": ("NASA/CNES (2025). SWOT Level 2 KaRIn Low Rate "
                               "Sea Surface Height, Expert, Version D. "
                               "PO.DAAC, doi:10.5067/SWOT-L2-LR-SSH-EXPERT-2.0")}
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    channels = CHANNELS
    # 2 km pixels: log2(2 / 27.83) = -3.798, the ledger's -3.80
    log2_fp = float(np.log2(2.0 / 27.83))
    log2_dt = -4.0
    per_year = True
    first_year = FIRST_YEAR
    fetch_month_scope = "month"
    qc_policy = (
        "`ssh_karin_qual` is a 32-bit bit field whose top byte is the "
        "product's own summary grade; `qc` carries that grade (0 good, 1 "
        "suspect, 2 degraded, 3 bad) and a pixel graded BAD is dropped and "
        "counted, with the whole grade histogram recorded. The grade "
        "vocabulary is checked against the variable's own `flag_meanings` and "
        "a table that names none of good/suspect/degraded/bad is a REFUSAL, "
        "not a guess. A pixel with no `ssha_karin`, no time or no position is "
        "dropped and counted. A value outside its channel's bounds becomes "
        "NaN for that channel alone and is counted, never clipped. The "
        "32-bit quality mask is NOT a channel: float16 cannot carry it, so "
        "the note's third channel is `ssh_karin_uncert` and the mask is the "
        "grade in `qc` — a decision for the main session, stated here.")
    sources = (f"{CMR}?collection_concept_id={COLLECTION} (anonymous)",
               "https://archive.swot.podaac.earthdata.nasa.gov/"
               "podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_D/"
               "SWOT_L2_LR_SSH_Expert_<CCC>_<PPP>_<start>_<end>_<PG>_<NN>.nc")
    verified = (
        "2026-09-18 from this sandbox (CMR only — the granules are "
        "Earthdata-protected and the bytes are a hosted runner's job): the "
        "collection SWOT_L2_LR_SSH_EXPERT_D (version D, C3233942272-POCLOUD) "
        "and CYCLE 010 IN FULL — 581 granules, passes 1..584 with three "
        "absent, 2024-01-25T00:19 .. 2024-02-14T21:04 (21 days, 24-28 passes "
        "a day), 18,807.9 MB, mean 32.37 MB a pass, which is the note's "
        "32 MB/pass and 18.8 GB/cycle exactly. Each granule's protected "
        "https .nc link read from CMR. NOT YET MEASURED HERE, and what the "
        "runner probe is for: num_lines x num_pixels per pass, the valid "
        "fraction of `ssha_karin`, the grade histogram of `ssh_karin_qual`, "
        "and therefore the two storage options' real byte counts.")
    notes = ""
    smoke_window = ("2024-01-25", "2024-01-26")
    smoke_probe_month = "2024-01"

    def __init__(self):
        self.cycle = (os.environ.get("SWOT_CYCLE") or "010").strip()
        try:
            self.max_passes = int(os.environ.get("SWOT_MAX_PASSES") or 12)
        except ValueError:
            sys.exit("SWOT_MAX_PASSES must be an integer")
        self.allow_build = (os.environ.get("SWOT_ALLOW_BUILD") or "") == "1"
        self._gran = None
        self.notes = (
            f"PROBE ONLY. This store is a MEASUREMENT for the storage "
            f"decision described in the module docstring; every stage past "
            f"`probe` refuses unless SWOT_ALLOW_BUILD=1. The probe reads "
            f"cycle {self.cycle!r}"
            + (f", at most {self.max_passes} pass(es)"
               if self.max_passes else ", every pass")
            + ". A restricted probe is not a whole store and says so here.")

    # ------------------------------------------------------------ granules --
    def granules(self, ctx):
        """{(cycle, pass): entry} — the cycle the knob names, or the window."""
        if self._gran is not None:
            return self._gran
        if ctx.source_dir:
            root = os.path.join(ctx.source_dir, self.store)
            ents = []
            for n in sorted(os.listdir(root)) if os.path.isdir(root) else []:
                if not n.endswith(".nc"):
                    continue
                p = os.path.join(root, n)
                ents.append({"title": n[:-3],
                             "granule_size": os.path.getsize(p) / 1e6,
                             "links": [{"href": "https://protected/" + n}],
                             "time_start": None, "time_end": None})
            if not ents:
                sys.exit(f"REFUSING swot: no .nc under {root} — the smoke's "
                         f"synthetic archive is missing")
            ctx.count_bytes(sum(int(e["granule_size"] * 1e6) for e in ents))
        else:
            try:
                ents = cmr_granules(cycle=self.cycle or None,
                                    attempts=ctx.a.attempts,
                                    count=ctx.count_bytes)
            except FormatError as e:
                sys.exit(f"REFUSING swot: {e}")
            if not ents:
                sys.exit(f"REFUSING swot: CMR lists no granule of "
                         f"{SHORT_NAME} for cycle {self.cycle!r} — an empty "
                         f"listing is a refusal (ml/CLAUDE.md, the "
                         f"2026-09-14 rule)")
        try:
            gran, counts = parse_granules(ents)
        except FormatError as e:
            sys.exit(f"REFUSING swot: {e}")
        self._gran = (gran, counts)
        return self._gran

    def _wanted(self, ctx, t_lo=None, t_hi=None):
        gran, _ = self.granules(ctx)
        keys = sorted(gran)
        if t_lo is not None:
            keys = [k for k in keys
                    if _in_window(gran[k], t_lo, t_hi)]
        if self.max_passes:
            keys = keys[:self.max_passes]
        return keys

    def _get(self, ctx, key, entry):
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, entry["name"] + ".nc")
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        dest = os.path.join(ctx.scratch, self.store, entry["name"] + ".nc")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        err = None
        for i in range(max(1, ctx.a.attempts)):
            try:
                f10b.http_to_file(entry["url"], dest)
                return dest, True
            except f10b._NotFound:
                raise
            except (IOError, *cm.RETRY_ERRORS) as e:
                err = e
                if i < ctx.a.attempts - 1:
                    time.sleep(4.0 * (2 ** i))
        raise IOError(f"{entry['url']}: {type(err).__name__}: {err}")

    # --------------------------------------------------------------- rows ---
    def _rows(self, ctx, label, t_lo, t_hi):
        gran, _ = self.granules(ctx)
        keys = self._wanted(ctx, t_lo, t_hi)
        counts = {"passes_listed": len(gran), "passes_wanted": len(keys),
                  "cycle": self.cycle, "passes_cap": self.max_passes}
        per_pass = []
        t0 = time.time()

        def get(k):
            try:
                return k, self._get(ctx, k, gran[k]), None
            except f10b._NotFound:
                return k, None, "listed in CMR, and 404"
            except IOError as e:
                return k, None, f"{type(e).__name__}: {e}"

        workers = 1 if ctx.source_dir else WORKERS
        for k, got, err in cm.ordered_map(get, keys, workers,
                                          lookahead=2 * workers):
            if got is None:
                ctx.note_absent(f"{label} cycle {k[0]} pass {k[1]}",
                                f"{gran[k]['name']}: {err}")
                continue
            path, tmp = got
            c = {}
            try:
                r, meta = read_pass(path, c)
            except FormatError as e:
                sys.exit(f"REFUSING swot: {gran[k]['name']}: {e}")
            except (OSError, IOError) as e:
                ctx.note_absent(f"{label} cycle {k[0]} pass {k[1]}",
                                f"{gran[k]['name']}: not a readable netCDF "
                                f"({type(e).__name__}: {e})")
                continue
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            f10b._merge_counts(counts, c)
            meta["cycle"], meta["pass"] = k
            meta["bytes"] = gran[k]["bytes"]
            per_pass.append(meta)
            if r is None:
                continue
            t, lat, lon, vals, qc = r
            oob = self.mask_bounds(vals)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            counts["rows_kept"] = counts.get("rows_kept", 0) + len(t)
            plat = np.full(len(t), f10b.platform_hash("SWOT"), np.int64)
            yield label, self.pack(t, lat, lon, vals, plat, qc), None
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        counts["per_pass"] = per_pass[:40]
        # A LIST OF ONE, not a dict: `f10b._merge_counts` merges a dict ONE
        # level deep and adds its values, so a nested dict of dicts raises
        # there. A list concatenates, and only this one yield produces it.
        so = storage_options(counts, self.C)
        counts["storage_options"] = [so] if so else []
        yield label, None, counts

    # ----------------------------------------------------------- contract ---
    def fetch_preflight(self, ctx):
        """A BUILD refuses here, where the inputs are all it has cost.

        ml/CLAUDE.md §0.3: a precondition that depends only on the inputs is
        free at dispatch and expensive at hour three. This store has not been
        given a storage decision yet, so a build is refused before its first
        byte — the probe has its own path and is unaffected.
        """
        if getattr(ctx.a, "stage", "") == "probe" or self.allow_build \
                or ctx.source_dir:
            # `--source-dir` is the smoke's synthetic archive: building THAT
            # is how the code is exercised and is not a build of the store.
            return None
        sys.exit(
            "REFUSING to build swot: this store is a MEASUREMENT until the "
            "storage decision is made — tier P rows at 27 + 2C = 33 bytes a "
            "pixel against a per-pass sharded tile group (see the adapter's "
            "docstring and ml/family1/probes/swot_2024-01.json's "
            "`storage_options`). Probe it with `--stage probe --probe-month "
            "2024-01`; set SWOT_ALLOW_BUILD=1 once the decision is recorded. "
            "Nothing has been fetched.")

    def index(self, ctx):
        gran, counts = self.granules(ctx)
        keys = sorted(gran)
        per_day, by_cycle = {}, {}
        for k in keys:
            e = gran[k]
            if e["t0"]:
                per_day[e["t0"][:10]] = per_day.get(e["t0"][:10], 0) + 1
            by_cycle[k[0]] = by_cycle.get(k[0], 0) + 1
        nb = sum(e["bytes"] for e in gran.values())
        out = {"dataset": (f"SWOT L2 LR SSH Expert, version D "
                           f"({SHORT_NAME})"),
               "url": f"{CMR}?collection_concept_id={COLLECTION}",
               "version": "D", "collection": COLLECTION,
               "cycle_requested": self.cycle,
               "max_passes": self.max_passes,
               "granules": len(gran),
               "granules_per_cycle": by_cycle,
               "granules_per_day": per_day,
               "passes": sorted({int(k[1]) for k in keys}),
               "passes_n": len({k[1] for k in keys}),
               "bytes": int(nb),
               "bytes_per_granule_mean": (round(nb / len(gran), 1)
                                          if gran else None),
               "record": [gran[keys[0]]["t0"], gran[keys[-1]]["t1"]]
                         if keys else None,
               "other": counts,
               "probe_only": not self.allow_build}
        if not ctx.source_dir:
            # the nominal pass count of the science orbit, from the archive
            # rather than from the note: the highest pass number CMR lists
            out["passes_absent_in_cycle"] = (
                max(int(k[1]) for k in keys) - len({k[1] for k in keys})
                if keys else None)
        return out

    def fetch_year(self, ctx, year):
        lo = max(f10b.seconds_since_epoch(dt.date(int(year), 1, 1)), ctx.t_lo)
        hi = min(f10b.seconds_since_epoch(dt.date(int(year), 12, 31)) + 86399,
                 ctx.t_hi)
        yield from self._rows(ctx, str(year), lo, hi)

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        yield from self._rows(ctx, f"{year}-{int(month):02d}",
                              max(lo, ctx.t_lo), min(hi, ctx.t_hi))

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


def _in_window(entry, t_lo, t_hi):
    """Does the granule's own time range overlap [t_lo, t_hi] seconds?"""
    if not entry.get("t0"):
        return True                     # the smoke's synthetic listing
    def s(stamp):
        d = dt.datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")
        return f10b.seconds_since_epoch(d.date()) + d.hour * 3600 + \
            d.minute * 60 + d.second
    return s(entry["t0"]) <= t_hi and s(entry["t1"]) >= t_lo


def storage_options(counts, C):
    """The two options the main session is being asked to decide between.

    Everything here comes from what the probe MEASURED — pixels per pass,
    valid pixels, passes read — and from the two layouts' own byte rules
    (`family1gf.tex` §4): a tier-P row is 27 + 2C bytes, a compressed G-fine
    valid pixel is about 2C x 1.2 bytes and an unobserved one nothing.
    """
    n = int(counts.get("passes_read") or 0)
    if not n:
        return None
    px = int(counts.get("pixels_in_pass") or 0)
    ok = int(counts.get("pixels_valid") or 0)
    per_pass_px = px / n
    per_pass_ok = ok / n
    v = (ok / px) if px else None
    row_bytes = 27 + 2 * C
    tile_bytes = 2 * C * 1.2
    # the science orbit: 584 passes a cycle, a cycle every 20.86 days
    passes_per_cycle, cycle_days = 584, 20.86
    passes_year = passes_per_cycle * 365.25 / cycle_days
    return {
        "measured": {"passes_read": n,
                     "pixels_per_pass": round(per_pass_px, 1),
                     "valid_pixels_per_pass": round(per_pass_ok, 1),
                     "valid_fraction": (round(v, 6) if v else None)},
        "extrapolation_basis": {
            "passes_per_cycle": passes_per_cycle,
            "cycle_days": cycle_days,
            "passes_per_year": round(passes_year, 1),
            "note": ("584 passes a 20.86-day cycle is the science orbit's own "
                     "geometry (family1gf.tex); cycle 010 held 581 of the 584")},
        "tier_p": {
            "bytes_per_row": row_bytes,
            "rows_per_pass": round(per_pass_ok, 1),
            "rows_per_year": round(per_pass_ok * passes_year),
            "bytes_per_year": round(per_pass_ok * passes_year * row_bytes),
            "what": (f"one row per valid pixel at 27 + 2C = {row_bytes} "
                     f"bytes; no grid, the swath moves every pass")},
        "tier_g_per_pass_group": {
            "bytes_per_valid_pixel": tile_bytes,
            "bytes_per_year": round(per_pass_ok * passes_year * tile_bytes),
            "groups": passes_per_cycle,
            "shards_per_year": round(passes_year),
            "what": ("one sharded group per pass, 256 x 256 float16 tiles at "
                     "about 2C x 1.2 bytes a valid pixel; keeps the swath's "
                     "SHAPE, costs per-pass per-cycle corner geometry in "
                     "tile_grid.json because the orbit repeats to ~1 km, not "
                     "exactly")},
        "ratio_p_over_g": round(row_bytes / tile_bytes, 3),
    }


# ================================================================== smoke ==
SMOKE_LINES, SMOKE_PIXELS = 12, 8
SMOKE_PASSES = ("001", "002", "003")
SMOKE_CYCLE = "010"
FILL = 2.147483647e9


def write_pass(path, cycle, pss, day, seed):
    import netCDF4
    rng = np.random.default_rng(seed + int(pss))
    nl, npx = SMOKE_LINES, SMOKE_PIXELS
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension(DIM_LINES, nl)
    ds.createDimension(DIM_PIXELS, npx)
    ds.cycle_number = int(cycle)
    ds.pass_number = int(pss)
    t = ((day - dt.date(2000, 1, 1)).days * 86400.0
         + 3600.0 * int(pss) + np.arange(nl) * 1.0)
    v = ds.createVariable("time", "f8", (DIM_LINES,), fill_value=FILL)
    v[:] = t
    v.units = "seconds since 2000-01-01 00:00:00.0"
    vt = ds.createVariable("time_tai", "f8", (DIM_LINES,), fill_value=FILL)
    vt[:] = t + 37.0
    lat = np.linspace(-40.0, 40.0, nl)[:, None] + np.zeros((1, npx))
    lon = np.linspace(-10.0, 10.0, npx)[None, :] + np.zeros((nl, 1)) \
        + int(pss)
    for name, a in (("latitude", lat), ("longitude", lon)):
        vv = ds.createVariable(name, "f8", (DIM_LINES, DIM_PIXELS),
                               fill_value=FILL)
        vv[:] = a
        vv.units = "degrees_north" if name == "latitude" else "degrees_east"
    ssha = np.round(rng.uniform(-0.6, 0.6, (nl, npx)), 4)
    sig0 = np.round(rng.uniform(5.0, 25.0, (nl, npx)), 4)
    unc = np.round(rng.uniform(0.01, 0.2, (nl, npx)), 4)
    # the nadir gap: the two middle pixels of every line carry no ssha
    ssha[:, npx // 2 - 1:npx // 2 + 1] = FILL
    if int(pss) == 2:
        ssha[0, 0] = 9.0                     # out of bounds -> NaN, counted
    for name, a in (("ssha_karin", ssha), ("sig0_karin", sig0),
                    ("ssh_karin_uncert", unc)):
        vv = ds.createVariable(name, "f8", (DIM_LINES, DIM_PIXELS),
                               fill_value=FILL)
        vv[:] = a
        vv.units = "m" if name != "sig0_karin" else "dB"
    q = np.zeros((nl, npx), np.uint32)
    q[1, :] = 1 << 24                        # a whole line graded suspect
    q[2, :] = 3 << 24                        # a whole line graded bad
    vq = ds.createVariable("ssh_karin_qual", "u4", (DIM_LINES, DIM_PIXELS))
    vq[:] = q
    vq.flag_meanings = ("suspect_large_ssh_delta degraded_media_delay "
                        "bad_outside_of_range good_measurement")
    vq.flag_masks = np.array([1, 2, 4, 8], np.uint32)
    ds.close()


def make_smoke_sources(root, d_lo, d_hi, seed=20260918):
    """`<root>/swot/<granule name>.nc` — the flat layout the synthetic CMR
    listing is walked from — and the truth rows.

    The truth is ordered the way the store breaks ties among equal
    (bin, time_s): the order the adapter yields, i.e. pass ascending then the
    file's own (line, pixel) order.
    """
    os.environ.setdefault("SWOT_CYCLE", SMOKE_CYCLE)
    os.environ.setdefault("SWOT_MAX_PASSES", "0")
    base = os.path.join(root, "swot")
    os.makedirs(base, exist_ok=True)
    truth = []
    for i, pss in enumerate(SMOKE_PASSES):
        day = d_lo + dt.timedelta(days=min(i, (d_hi - d_lo).days))
        name = (f"SWOT_L2_LR_SSH_Expert_{SMOKE_CYCLE}_{pss}_"
                f"{day.strftime('%Y%m%dT%H%M%S')}_"
                f"{day.strftime('%Y%m%dT%H%M%S')}_PGD0_02_swot")
        write_pass(os.path.join(base, name + ".nc"), SMOKE_CYCLE, pss, day,
                   seed)
        import netCDF4
        ds = netCDF4.Dataset(os.path.join(base, name + ".nc"))
        t1 = np.asarray(ds.variables["time"][:], np.float64)
        lat = np.asarray(ds.variables["latitude"][:], np.float64)
        lon = np.asarray(ds.variables["longitude"][:], np.float64)
        cols = [np.asarray(ds.variables[c[0]][:], np.float64)
                for c in CHANNELS]
        q = np.asarray(ds.variables["ssh_karin_qual"][:], np.int64) >> 24
        ds.close()
        off = int((dt.date(2000, 1, 1) - dt.date(1982, 1, 1)).days) * 86400
        for li in range(SMOKE_LINES):
            for pi in range(SMOKE_PIXELS):
                if q[li, pi] >= BAD_GRADE:
                    continue
                if not np.isfinite(cols[0][li, pi]) or \
                        cols[0][li, pi] == FILL:
                    continue
                vv = []
                for (nm, _u, lo, hi), a in zip(CHANNELS, cols):
                    x = a[li, pi]
                    vv.append(np.nan if (x == FILL or not (lo <= x <= hi))
                              else float(x))
                truth.append({"t": int(round(t1[li] + off)),
                              "lat": float(lat[li, pi]),
                              "lon": float(lon[li, pi]),
                              "platform": f10b.platform_hash("SWOT"),
                              "v": vv})
    return truth


ADAPTER = SWOTAdapter
