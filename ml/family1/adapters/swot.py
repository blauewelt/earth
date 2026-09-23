"""SWOT L2 LR sea-surface height, Expert — sea level measured ACROSS two
50 km swaths instead of along one line (family 1.gf, E-082 wave 4).

PLAIN ENGLISH. Every radar altimeter before SWOT measured the height of the
sea directly beneath it, one line at a time; SWOT's KaRIn interferometer
measures it across two 50 km-wide strips either side of the ground track, on a
2 km grid. That is the first instrument that can see an ocean eddy's SHAPE
rather than one cut through it, which is what the Atlantic-overturning and
El Nino goals want it for.

THE STORAGE DECISION IS MADE: TIER P (main session, 2026-09-23). One row per
valid 2 km KaRIn pixel at 27 + 2C = 33 bytes, read the way `slatrack` is read,
2024 first, then 2023-07 -> 2026. The per-pass tier-G group was rejected: a
pass is 69 pixels wide, so the layout's 256-column tiles are 73 % pad, and it
would need a tile reshape in `family1/sharded.py` and a new reader
(`ml/family1/BUILD_LOG.md`, "swot: the storage decision, with measured
numbers"). A build is still opt-in — `SWOT_ALLOW_BUILD=1` names the intention
— and the probe is unchanged apart from the bounds and the reader below.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (CMR is anonymous; the granules
are Earthdata-protected, so the bytes are a runner's job):
  collection  SWOT_L2_LR_SSH_EXPERT_D, version D, C3233942272-POCLOUD
              (POCLOUD; the note's "Expert (version D)"). The 2.0 collections
              C2799465497 etc. are the earlier processing and are NOT read.
  granules    SWOT_L2_LR_SSH_Expert_<CCC>_<PPP>_<start>_<end>_<CRID>_<NN>_swot
              — cycle CCC, pass PPP, processing baseline CRID (PGD0, PID0, …)
              and product counter NN; the bytes are
              https://archive.swot.podaac.earthdata.nasa.gov/
                  podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_D/<name>.nc

CYCLE 010, MEASURED IN FULL FROM CMR 2026-09-18: **581 granules**, passes 1 ..
584 with three of the 584 absent, **2024-01-25T00:19 .. 2024-02-14T21:04**
(21 calendar days, 24-28 passes a day), **18,807.9 MB, mean 32.37 MB a pass** —
the note's "32 MB/pass, 18.8 GB/cycle" exactly. That is why the probe month is
2024-01: it holds cycle 010's first seven days.

WHAT THE PROBE READS AND WHAT A BUILD READS — TWO DIFFERENT LISTINGS.
  PROBE (`fetch_month`): cycle `SWOT_CYCLE` (default 010) as CMR lists it by
      granule name, at most `SWOT_MAX_PASSES` (default 12) passes that meet
      the month — about 388 MB. The restriction is written into `notes`, so a
      restricted probe can never look like a whole store.
  BUILD (`fetch_year`, i.e. every hosted lane): EVERY granule CMR lists for
      the lane's window, all cycles (a calendar month holds the tail of one
      cycle, a whole cycle and the head of a third — 2024-01 is cycles 008,
      009 and 010), and NO cap. The two knobs are the probe's: a build that
      names either of them REFUSES before its first byte unless
      `SWOT_ALLOW_CAPPED_BUILD=1` says a partial window is what is wanted —
      the laser stores' lesson (BUILD_LOG, "Two things the first lanes
      taught": twelve granules of 4,884 marked a year done in 38 s).

A ROW BELONGS TO THE LANE ITS OWN SECOND FALLS IN. A pass lasts ~52 minutes,
so the pass that starts 2024-01-31T23:31 ends 2024-02-01T00:23 and CMR lists
it for BOTH months (it did: 865 granules for 2024-01 and 809 for 2024-02 count
their shared straddlers twice). Both lanes read it; each keeps only the rows
whose own `time_s` is inside its window and counts the rest
(`rows_outside_window`). So twelve monthly lanes, four quarters or one year
tile the rows exactly once, which is what `_part_paths`' tie-break promise
("no two lanes can hold rows sharing a (bin, time_s)") assumes of every
tier-P adapter. The tier-G "bin belongs to its first day" rule (77dff61) is
not involved: a tier-P lane owns rows, not bins, and a bin split across two
lanes is re-joined by the assembler's (bin, time_s) sort. Pass-level counters
(`passes_read`, `pixels_in_pass`, `qual_grade`) count every pass a lane read,
so a straddling pass appears in both lanes' ledgers; `rows_kept` is exact.

WHAT A ROW IS. One valid 2 km KaRIn pixel.
  time_s    the `time` variable (num_lines), broadcast across the swath,
            through `f10b._cf_time_to_seconds` on its OWN `units` attribute.
            The file also carries `time_tai`; `time` is the UTC axis and is
            the one read.
  lat, lon  `latitude`, `longitude` (num_lines, num_pixels), lon wrapped.
  values    C = 3: `ssha_karin` (m, the anomaly), `sig0_karin` (the KaRIn
            backscatter, **linear, unit "1" — NOT dB**) and
            `ssh_karin_uncert` (m). The note's third channel is "quality";
            a 32-bit quality BITMASK cannot be a float16 channel without
            losing bits, so the flag goes into `qc` as a GRADE read from the
            variable's own `flag_meanings` and the uncertainty — a number —
            takes the channel.
  platform  `platform_hash("SWOT")` — one satellite.
  qc        0 good, 1 suspect, 2 degraded, 3 bad, from `ssh_karin_qual`'s own
            flag table; a pixel graded bad is dropped and counted.

HOW A VALUE IS READ. The product packs its channels: `ssha_karin` is int32 x
1e-4 m, `ssh_karin_uncert` uint16 x 1e-4 m, `sig0_karin` float32, each with a
`_FillValue` and a `valid_min`/`valid_max` (CMR UMM-Var V3572361…-POCLOUD,
read 2026-09-23). netCDF4's automatic masking hides a value outside that range
but hands back its RAW, UNSCALED integer through `np.asarray` — measured with
netCDF4 1.7.4: an uncertainty of raw 62,000 (6.2 m, over valid_max 60,000)
came back as 62000.0 "metres", which is what the probe's one out-of-bounds
`ssh_karin_uncert` was. So `_read` switches the automatic mask and scale OFF
and does it itself: fill -> NaN, outside the product's own valid range -> NaN
and COUNTED per variable (`outside_product_valid_range`), then scale and
offset. Each channel's `units` attribute must equal the declared unit, or the
granule is a refusal (ADAPTER_CONTRACT rule 4) — which is the check that
would have caught `sig0_karin` being declared "dB".

THE CHANNEL BOUNDS, AND THE RULE THEY FOLLOW. A value outside [lo, hi] becomes
NaN for that channel alone and is COUNTED under `out_of_bounds`, never
clipped. The rule: **a bound refuses only what cannot be a measurement of the
quantity — the product's own valid range where the producer states one that
is physical, the physics where the producer's range is only an integer
container, and never more than the store's float16 can hold.**
  ssha_karin        [-3, 3] m. The product's valid range (-1,500 .. 15,000 m)
                    is the int32 container `ssh_karin` shares, not a statement
                    about an ANOMALY. After the product's own tide, pole-tide,
                    internal-tide and dynamic-atmosphere (storm-surge)
                    corrections, the largest real open-ocean anomaly is a
                    mesoscale eddy or meander of about +/- 1.5-2 m, so
                    |ssha| > 3 m is an error (land or inland water leaking
                    into the swath, a mis-corrected pixel). Kept, with the
                    reason written down.
  sig0_karin        [-1000, 65504] (linear). -1000 is the product's own
                    valid_min (noise subtraction makes small negatives real).
                    65,504 (48.2 dB) is the largest finite float16: the
                    product's valid_max of 1e7 (70 dB) cannot be stored in a
                    float16 channel at all, and anything above 65,504 would
                    become +inf. Ocean KaRIn sigma0 at 0.6-3.9 deg incidence is
                    ~10-20 dB (10-100 linear); calm or specular water runs to
                    30-40 dB (1,000-10,000). The old [-10, 40] was a dB range
                    applied to a LINEAR field — it refused everything above
                    16 dB, i.e. ordinary low-wind ocean.
  ssh_karin_uncert  [0, 6] m — the product's own valid range (uint16 60,000 x
                    1e-4), which the reader above already enforces.
THE PROBE'S NUMBERS AND THEIR LIMITATION. The 2024-01 probe (#168) recorded
COUNTS, not a distribution: 47,390 `sig0_karin` values (1.126 % of 4,209,008
rows) outside the old [-10, 40] and 4,629 missing at source (NaN fraction
0.012359), 33,675 `ssha_karin` (0.800 %) outside +/- 3 m (its NaN fraction
0.008001 is exactly that, so every ssha NaN in the store was a bound, none a
fill), one `ssh_karin_uncert` over 5 m. A count cannot say how far out a value
was, so the bounds above come from the product's own attributes and the
physics, and the probe now RECORDS THE DISTRIBUTION as well: `hist_<channel>`,
a fixed-edge histogram of every kept row's value before the bounds, so the next
probe or any lane's ledger shows where the tail sits (for ssha: the mass in
3-5 m against beyond 10 m).

THE PROBE'S STORAGE ARITHMETIC (`storage_options` in its counts) is kept as
the record of the decision; a build does not compute it.
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
# processing baseline (CRID) and product counter at the end of the name
VERSION = re.compile(r"_([A-Z0-9]{4})_(\d{2})(?:_swot)?$")
# (name, unit, lo, hi) — the unit is CHECKED against the file's own `units`
# attribute; the bounds' reasons are in the module docstring.
F16_MAX = 65504.0
CHANNELS = (("ssha_karin", "m", -3.0, 3.0),
            ("sig0_karin", "1", -1000.0, F16_MAX),
            ("ssh_karin_uncert", "m", 0.0, 6.0))
# Fixed histogram edges per channel, in the channel's own unit: the
# distribution the bounds are judged against (`hist_<name>` in the counts).
HIST_EDGES = {
    "ssha_karin": (-10.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0,
                   3.0, 5.0, 10.0),
    # linear sigma0; in dB: 0, 10, 16, 20, 30, 40, 48.2
    "sig0_karin": (-1000.0, -10.0, 0.0, 1.0, 10.0, 40.0, 100.0, 1000.0,
                   10000.0, F16_MAX),
    "ssh_karin_uncert": (0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 6.0),
}
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
EPOCH = dt.datetime(1982, 1, 1)


class FormatError(ValueError):
    """A granule listing or a netCDF that does not look like the product."""


# ============================================================== the index ==
def cmr_granules(cycle=None, temporal=None, page=2000, attempts=4,
                 count=None):
    """Every granule of the collection, optionally one cycle or one window.

    ANONYMOUS — CMR needs no login (the BYTES do). An empty answer is
    returned as an empty list and the CALLER decides whether that is a
    refusal.

    PAGING. `get_bytes` drops the response headers, so the `CMR-Search-After`
    cursor is not available; a full page is followed by asking for the window
    from the last granule's START. CMR's temporal filter is inclusive, so that
    granule (and any sharing its start second) comes back on the next page —
    entries are therefore de-duplicated by their concept id. Without that, a
    window of more than one page (a whole year is ~9,700 granules) raised
    "two granules for cycle … pass …" on the page seam.
    """
    out, seen, after = [], set(), None
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
        js = json.loads(raw)
        e = js.get("feed", {}).get("entry", [])
        new = 0
        for g in e:
            k = g.get("id") or g.get("title")
            if k in seen:
                continue
            seen.add(k)
            out.append(g)
            new += 1
        if len(e) < page:
            return out
        if temporal is None:
            return out                      # one page is all a cycle needs
        if not new:
            raise FormatError(
                f"CMR returned a full page of {len(e)} granule(s) that were "
                f"all already listed — {page} granules share one start "
                f"second, and this pager cannot move past them ({url})")
        last = e[-1]["time_start"]
        temporal = f"{last},{temporal.split(',')[1]}"
        after = None


def parse_granules(entries):
    """CMR entries -> {(cycle, pass): {name, url, bytes, t0, t1}}.

    Refuses a name that is not the product's (counted, not refused: the
    collection carries a few misfiled `Basic`/`Unsmoothed` granules — eleven
    in 2024-05) and two granules for one (cycle, pass) — EXCEPT a
    re-processing of the same granule: the same processing baseline (CRID)
    with a higher product counter supersedes the lower one, the product's own
    versioning rule. CMR lists both for 76 passes of 2025-06 (PID0_01 ..
    PID0_04) and one each of 2026-08 and 2026-09; 2024 has none. Two
    DIFFERENT baselines for one pass stay a refusal: which processing a store
    holds is a decision, not a tie-break.
    """
    out, other, superseded = {}, [], []
    for g in entries:
        title = g.get("title", "")
        m = NAME.match(title)
        if not m:
            other.append(title)
            continue
        key = (m.group(1), m.group(2))
        v = VERSION.search(title)
        crid, nn = (v.group(1), int(v.group(2))) if v else (None, None)
        url = None
        for ln in g.get("links", []):
            h = str(ln.get("href", ""))
            if h.startswith("https") and "protected" in h and h.endswith(".nc"):
                url = h
                break
        if url is None:
            raise FormatError(f"{title}: no protected https .nc link "
                              f"({[l.get('href') for l in g.get('links', [])][:3]})")
        ent = {"name": title, "url": url,
               "bytes": int(float(g.get("granule_size") or 0) * 1e6),
               "t0": g.get("time_start"), "t1": g.get("time_end"),
               "start": m.group(3), "end": m.group(4),
               "crid": crid, "counter": nn}
        if key in out:
            prev = out[key]
            if crid and prev["crid"] == crid and nn != prev["counter"]:
                keep, drop = (ent, prev) if nn > prev["counter"] else (prev, ent)
                out[key] = keep
                superseded.append(drop["name"])
                continue
            raise FormatError(f"two granules for cycle {key[0]} pass {key[1]}: "
                              f"{prev['name']} and {title}")
        out[key] = ent
    counts = {}
    if other:
        counts["granules_other_name"] = len(other)
        counts["granules_other_names"] = sorted(other)[:10]
    if superseded:
        counts["granules_superseded"] = len(superseded)
        counts["granules_superseded_names"] = sorted(superseded)[:10]
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


def _attr(v, name):
    a = getattr(v, name, None)
    if a is None:
        return None
    a = np.asarray(a)
    return a.ravel()


def _read(ds, name, counts=None, ranges=None):
    """One variable -> float64 in PHYSICAL units, NaN for not-measured.

    The automatic mask and scale are OFF (the module docstring says why):
    `_FillValue` -> NaN; a raw value outside `valid_range` / `valid_min` /
    `valid_max` -> NaN and counted under
    `counts["outside_product_valid_range"][name]`; then `scale_factor` and
    `add_offset`. `ranges[name]` receives the valid range in physical units.
    """
    v = ds.variables[name]
    v.set_auto_maskandscale(False)
    raw = np.asarray(v[:])
    a = raw.astype(np.float64)
    miss = np.zeros(raw.shape, bool)
    fill = _attr(v, "_FillValue")
    if fill is not None:
        miss |= raw == np.asarray(fill[0]).astype(raw.dtype)
    lo = hi = None
    vr = _attr(v, "valid_range")
    if vr is not None and len(vr) == 2:
        lo, hi = vr[0], vr[1]
    else:
        vmin, vmax = _attr(v, "valid_min"), _attr(v, "valid_max")
        lo = vmin[0] if vmin is not None else None
        hi = vmax[0] if vmax is not None else None
    out = np.zeros(raw.shape, bool)
    if lo is not None:
        out |= raw < lo
    if hi is not None:
        out |= raw > hi
    out &= ~miss
    scale = _attr(v, "scale_factor")
    off = _attr(v, "add_offset")
    sc = float(scale[0]) if scale is not None else 1.0
    of = float(off[0]) if off is not None else 0.0
    a = a * sc + of
    a[miss | out] = np.nan
    a[~np.isfinite(a)] = np.nan
    n_out = int(out.sum())
    if counts is not None and n_out:
        d = counts.setdefault("outside_product_valid_range", {})
        d[name] = d.get(name, 0) + n_out
    if ranges is not None:
        ranges[name] = [None if lo is None else float(lo) * sc + of,
                        None if hi is None else float(hi) * sc + of]
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
    # THE UNIT IS THE FILE'S, NOT OURS (ADAPTER_CONTRACT rule 4). sig0_karin
    # was declared "dB" from the note while the product stores LINEAR sigma0
    # with units "1"; the [-10, 40] bound built on that misreading refused
    # every pixel above 16 dB.
    units = {}
    for name, unit, _lo, _hi in CHANNELS:
        got = str(getattr(ds.variables[name], "units", "")).strip()
        units[name] = got
        if got != unit:
            raise FormatError(
                f"`{name}` carries units {got!r} and the adapter declares "
                f"{unit!r} — refusing to store a channel under a unit the "
                f"file does not state")
    return {"num_lines": nl, "num_pixels": npx, "pixels": nl * npx,
            "time_units": u, "variables": len(ds.variables),
            "has_time_tai": "time_tai" in ds.variables,
            "format": ds.data_model,
            "units": units,
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
        t1 = f10b._cf_time_to_seconds(_read(ds, "time", counts),
                                      meta["time_units"])
        lat = _read(ds, "latitude", counts)
        lon = _read(ds, "longitude", counts)
        ranges = {}
        vals = np.stack([_read(ds, c[0], counts, ranges) for c in CHANNELS],
                        axis=-1)
        vq = ds.variables["ssh_karin_qual"]
        vq.set_auto_maskandscale(False)
        qual = np.asarray(vq[:]).astype(np.int64)
        grades = qc_grade(vq)
    finally:
        ds.close()
    meta["product_valid_range"] = ranges
    t = np.repeat(t1[:, None], npx, axis=1)
    # the fill (0xFFFFFFFF) has top byte 255 and grades BAD, as before
    g = np.clip(qual >> 24, 0, len(GRADES) - 1).astype(np.int64)
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


def _hist_labels(edges):
    fmt = lambda x: f"{x:g}"                                    # noqa: E731
    return ([f"<{fmt(edges[0])}"]
            + [f"[{fmt(a)},{fmt(b)})" for a, b in zip(edges[:-1], edges[1:])]
            + [f">={fmt(edges[-1])}"])


def value_histograms(vals):
    """{f"hist_{channel}": {bin label: n}} over the FINITE values of each
    channel, before the bounds. One level of dict of ints, so
    `f10b._merge_counts` sums it across passes, parts and lanes."""
    out = {}
    for i, (name, _u, _lo, _hi) in enumerate(CHANNELS):
        edges = np.asarray(HIST_EDGES[name], np.float64)
        x = vals[:, i]
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        idx = np.searchsorted(edges, x, side="right")
        n = np.bincount(idx, minlength=len(edges) + 1)
        labels = _hist_labels(HIST_EDGES[name])
        out[f"hist_{name}"] = {labels[j]: int(n[j]) for j in range(len(n))
                               if n[j]}
    return out


# ================================================================ adapter ==
class SWOTAdapter(f10b.SourceAdapter):
    store = "swot"
    title = ("SWOT L2 LR sea-surface height, Expert (version D): the 2 km "
             "KaRIn swath, one row per valid pixel (tier P)")
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
        "dropped and counted. Values are read with netCDF4's automatic mask "
        "and scale OFF: the fill is NaN, a raw value outside the product's "
        "own valid range is NaN and counted per variable "
        "(`outside_product_valid_range`), then scale and offset are applied. "
        "A value outside its channel's bounds becomes NaN for that channel "
        "alone and is counted (`out_of_bounds`), never clipped; the bounds "
        "are ssha_karin +/- 3 m (after the product's tide and "
        "dynamic-atmosphere corrections a larger open-ocean anomaly is an "
        "error), sig0_karin -1000 .. 65504 LINEAR (the product's valid_min; "
        "the largest finite float16, 48.2 dB), ssh_karin_uncert 0 .. 6 m (the "
        "product's valid range). `hist_<channel>` records each channel's "
        "distribution before the bounds. A row belongs to the lane whose "
        "window holds its own second; rows of a straddling pass outside the "
        "window are dropped and counted (`rows_outside_window`). The 32-bit "
        "quality mask is NOT a channel: float16 cannot carry it, so the "
        "note's third channel is `ssh_karin_uncert` and the mask is the grade "
        "in `qc`.")
    sources = (f"{CMR}?collection_concept_id={COLLECTION} (anonymous)",
               "https://archive.swot.podaac.earthdata.nasa.gov/"
               "podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_D/"
               "SWOT_L2_LR_SSH_Expert_<CCC>_<PPP>_<start>_<end>_<CRID>_<NN>.nc")
    verified = (
        "2026-09-18 from this sandbox (CMR only — the granules are "
        "Earthdata-protected and the bytes are a hosted runner's job): the "
        "collection SWOT_L2_LR_SSH_EXPERT_D (version D, C3233942272-POCLOUD) "
        "and CYCLE 010 IN FULL — 581 granules, passes 1..584 with three "
        "absent, 2024-01-25T00:19 .. 2024-02-14T21:04 (21 days, 24-28 passes "
        "a day), 18,807.9 MB, mean 32.37 MB a pass. 2026-09-18, runner probe "
        "#168: every pass 9,866 x 69 pixels, 51.52 % valid. 2026-09-23, CMR "
        "UMM-Var for the collection's 103 variables: sig0_karin units '1' "
        "(LINEAR, 'not decibels'), valid -1000 .. 1e7, fill 9.96921e36; "
        "ssha_karin int32 x 1e-4 m, valid -15,000,000 .. 150,000,000 raw; "
        "ssh_karin_uncert uint16 x 1e-4 m, valid 0 .. 60,000 raw. 2024 "
        "listed month by month by time window: 865, 809, 857, 837, 369, 835, "
        "859, 869, 834, 859, 835, 864 Expert granules (a straddling pass is "
        "listed in both months), 25.7-27.9 GB a month except 2024-05, no "
        "duplicate (cycle, pass); 2024-05 is short upstream (369) and carries "
        "eleven misfiled Basic/Unsmoothed granules; 2025-06 lists 76 passes "
        "twice (PID0 re-processings, counters 01..04).")
    notes = ""
    smoke_window = ("2024-01-30", "2024-01-31")
    smoke_probe_month = "2024-01"

    def __init__(self):
        # READ AT CONSTRUCTION TIME, so the fresh adapter `stage_probe` builds
        # for itself sees the same knobs; `family1-build.yml`'s `adapter_env`
        # is how a dispatch sets them.
        self._session = None
        raw_cycle = (os.environ.get("SWOT_CYCLE") or "").strip()
        raw_cap = (os.environ.get("SWOT_MAX_PASSES") or "").strip()
        self.cycle = raw_cycle or "010"
        try:
            self.max_passes = int(raw_cap) if raw_cap else 12
        except ValueError:
            sys.exit("SWOT_MAX_PASSES must be an integer")
        # THE PROBE'S KNOBS, NEVER THE BUILD'S (module docstring).
        self.knobs_set = [k for k, v in (("SWOT_CYCLE", raw_cycle),
                                         ("SWOT_MAX_PASSES", raw_cap)) if v]
        self.allow_capped_build = (
            os.environ.get("SWOT_ALLOW_CAPPED_BUILD") or "") == "1"
        self.allow_build = (os.environ.get("SWOT_ALLOW_BUILD") or "") == "1"
        self._gran = None
        self._listing = {}
        self.notes = (
            "TIER P (decided 2026-09-23): one row per valid 2 km KaRIn "
            "pixel. A BUILD reads every granule CMR lists for its window, "
            "all cycles, no cap, and keeps exactly the rows whose own second "
            "lies in the window, so the lanes of a year tile it once. A "
            f"PROBE reads cycle {self.cycle!r}"
            + (f", at most {self.max_passes} pass(es)"
               if self.max_passes else ", every pass")
            + "; a restricted probe is not a whole store and says so here."
            + (f" THIS BUILD WAS CAPPED by {', '.join(self.knobs_set)} "
               f"(SWOT_ALLOW_CAPPED_BUILD=1)"
               if self.knobs_set and self.allow_capped_build else ""))

    # ------------------------------------------------------------ granules --
    def _local_entries(self, ctx):
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
        return ents

    def _parse(self, ents):
        try:
            return parse_granules(ents)
        except FormatError as e:
            sys.exit(f"REFUSING swot: {e}")

    def granules(self, ctx):
        """THE PROBE'S LISTING: {(cycle, pass): entry} for the knob's cycle."""
        if self._gran is not None:
            return self._gran
        if ctx.source_dir:
            ents = self._local_entries(ctx)
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
        self._gran = self._parse(ents)
        return self._gran

    def listing(self, ctx, t_lo, t_hi):
        """THE BUILD'S LISTING: every granule whose time range meets
        [t_lo, t_hi] seconds, all cycles, straight from CMR's temporal
        search. An empty window is a refusal."""
        key = (int(t_lo), int(t_hi))
        if key in self._listing:
            return self._listing[key]
        if ctx.source_dir:
            ents = self._local_entries(ctx)
        else:
            window = f"{_iso(t_lo)},{_iso(t_hi)}"
            try:
                ents = cmr_granules(temporal=window, attempts=ctx.a.attempts,
                                    count=ctx.count_bytes)
            except FormatError as e:
                sys.exit(f"REFUSING swot: {e}")
            if not ents:
                sys.exit(f"REFUSING swot: CMR lists no granule of "
                         f"{SHORT_NAME} for {window} — an empty listing is a "
                         f"refusal (ml/CLAUDE.md, the 2026-09-14 rule)")
        self._listing[key] = self._parse(ents)
        return self._listing[key]

    def _get(self, ctx, key, entry):
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, entry["name"] + ".nc")
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        # EARTHDATA LOGIN, not a plain GET. `archive.swot.podaac.
        # earthdata.nasa.gov` answers an unauthenticated request with a 302
        # to urs.earthdata.nasa.gov and then `HTTP 401 — HTTP Basic: Access
        # denied.`, which is what family1-build run #152 measured on every
        # one of the twelve passes it asked for. `cm.earthdata_download`
        # carries the .netrc through the redirect chain (and only to the
        # login host) and forces IPv4.
        dest = os.path.join(ctx.scratch, self.store, entry["name"] + ".nc")
        if self._session is None:
            self._session = cm.earthdata_session()
        n, why = cm.earthdata_download(self._session, entry["url"], dest,
                                       attempts=max(1, ctx.a.attempts))
        if n is None:
            raise f10b._NotFound(f"{entry['url']}: {why}")
        return dest, True

    # --------------------------------------------------------------- rows ---
    def _rows(self, ctx, label, t_lo, t_hi, probe):
        if probe:
            gran, _ = self.granules(ctx)
            cap = self.max_passes
        else:
            gran, _ = self.listing(ctx, t_lo, t_hi)
            cap = self.max_passes if self.allow_capped_build else 0
        keys = sorted((k for k in gran if _in_window(gran[k], t_lo, t_hi)),
                      key=lambda k: (gran[k]["t0"] or "", k))
        if cap:
            keys = keys[:cap]
        counts = {"passes_listed": len(gran), "passes_wanted": len(keys),
                  "cycle": self.cycle if probe else "all",
                  "passes_cap": cap}
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
            # A ROW BELONGS TO THE LANE ITS OWN SECOND FALLS IN — on the
            # second `_pack` will store, so the test and the store agree.
            ts = np.rint(t)
            inside = (ts >= t_lo) & (ts <= t_hi)
            n_out = int((~inside).sum())
            if n_out:
                counts["rows_outside_window"] = \
                    counts.get("rows_outside_window", 0) + n_out
                if not inside.any():
                    continue
                t, lat, lon, vals, qc = (t[inside], lat[inside], lon[inside],
                                         vals[inside], qc[inside])
            f10b._merge_counts(counts, value_histograms(vals))
            oob = self.mask_bounds(vals)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            counts["rows_kept"] = counts.get("rows_kept", 0) + len(t)
            plat = np.full(len(t), f10b.platform_hash("SWOT"), np.int64)
            yield label, self.pack(t, lat, lon, vals, plat, qc), None
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        # a probe keeps forty passes' inventory; a build lane keeps two, as a
        # sample of the file's own geometry, units and valid ranges — twelve
        # lanes' ledgers concatenate this list in store.json
        counts["per_pass"] = per_pass[:40] if probe else per_pass[:2]
        if probe:
            # A LIST OF ONE, not a dict: `f10b._merge_counts` merges a dict
            # ONE level deep and adds its values, so a nested dict of dicts
            # raises there. A list concatenates, and only this yield makes it.
            so = storage_options(counts, self.C)
            counts["storage_options"] = [so] if so else []
        yield label, None, counts

    # ----------------------------------------------------------- contract ---
    def _refuse_capped_build(self):
        if self.knobs_set and not self.allow_capped_build:
            sys.exit(
                f"REFUSING to build swot with {', '.join(self.knobs_set)} set: "
                f"those are the PROBE's knobs. A build reads every granule "
                f"of its window, all cycles; a capped build would mark the "
                f"lane done with a fraction of it and report success (the "
                f"ICESat-2 lanes, BUILD_LOG 2026-09-20). Unset them, or say "
                f"SWOT_ALLOW_CAPPED_BUILD=1 if a partial window is what is "
                f"wanted (it is written into notes). Nothing has been fetched.")

    def fetch_preflight(self, ctx):
        """Every guard where the inputs are all it has cost (§0.3).

        A BOX ASSEMBLING FROM HUB PARTS reads no source: it passes, with no
        Earthdata credential (a box never has one, ml/CLAUDE.md §6) and no
        SWOT_ALLOW_BUILD — the parts it pulls exist only because a lane was
        allowed to build them.

        A PROBE passes the build guard. A BUILD needs SWOT_ALLOW_BUILD=1, and
        refuses the probe's knobs (SWOT_CYCLE, SWOT_MAX_PASSES) unless
        SWOT_ALLOW_CAPPED_BUILD=1. Either, reading the real archive, needs an
        Earthdata netrc: the granules are protected, so a fetch without one
        spends the whole listing to discover a page of 401s (run #152).
        """
        if getattr(ctx.a, "parts_from_hub", False):
            return None
        probe = getattr(ctx.a, "stage", "") == "probe"
        if not (probe or self.allow_build or ctx.source_dir):
            sys.exit(
                "REFUSING to build swot without SWOT_ALLOW_BUILD=1: the "
                "storage decision is made (tier P, 2026-09-23) and a build is "
                "opt-in so that no dispatch fetches ~28 GB a month by "
                "accident. Pass adapter_env=SWOT_ALLOW_BUILD=1 on a hosted "
                "lane (one calendar month, extra_args=--push-parts). Nothing "
                "has been fetched.")
        if not probe and not ctx.source_dir:
            self._refuse_capped_build()
        if not ctx.source_dir and not cm.earthdata_ready():
            sys.exit(
                "REFUSING swot: the granules are Earthdata-protected and "
                "this process can authenticate to Earthdata Login in "
                "NEITHER way: no EARTHDATA_USERNAME / "
                "EARTHDATA_PASSWORD in the environment and no netrc "
                "naming urs.earthdata.nasa.gov. family1-build.yml gives "
                "both on a HOSTED runner from the repository secrets "
                "(ml/CLAUDE.md §6); a box gets neither. Nothing has been "
                "fetched.")
        return None

    def index(self, ctx):
        probe = getattr(ctx.a, "stage", "") == "probe"
        if probe:
            gran, counts = self.granules(ctx)
        else:
            gran, counts = self.listing(ctx, ctx.t_lo, ctx.t_hi)
        keys = sorted(gran, key=lambda k: (gran[k]["t0"] or "", k))
        per_day, by_cycle, by_crid = {}, {}, {}
        for k in keys:
            e = gran[k]
            if e["t0"]:
                per_day[e["t0"][:10]] = per_day.get(e["t0"][:10], 0) + 1
            by_cycle[k[0]] = by_cycle.get(k[0], 0) + 1
            c = f"{e.get('crid')}_{e.get('counter'):02d}" \
                if e.get("crid") else "?"
            by_crid[c] = by_crid.get(c, 0) + 1
        nb = sum(e["bytes"] for e in gran.values())
        out = {"dataset": (f"SWOT L2 LR SSH Expert, version D "
                           f"({SHORT_NAME})"),
               "url": f"{CMR}?collection_concept_id={COLLECTION}",
               "version": "D", "collection": COLLECTION,
               "listing": ("probe: cycle by granule name" if probe else
                           "build: every granule meeting the window, all "
                           "cycles"),
               "cycle_requested": self.cycle if probe else None,
               "max_passes": self.max_passes if probe else 0,
               "granules": len(gran),
               "granules_per_cycle": by_cycle,
               "granules_per_processing": by_crid,
               "granules_per_day": per_day,
               "passes": sorted({int(k[1]) for k in keys}),
               "passes_n": len({k[1] for k in keys}),
               "bytes": int(nb),
               "bytes_per_granule_mean": (round(nb / len(gran), 1)
                                          if gran else None),
               "record": [gran[keys[0]]["t0"], gran[keys[-1]]["t1"]]
                         if keys else None,
               "other": counts,
               "build_allowed": bool(self.allow_build)}
        if probe and not ctx.source_dir:
            # the nominal pass count of the science orbit, from the archive
            # rather than from the note: the highest pass number CMR lists
            out["passes_absent_in_cycle"] = (
                max(int(k[1]) for k in keys) - len({k[1] for k in keys})
                if keys else None)
        return out

    def fetch_year(self, ctx, year):
        """A BUILD (every lane): the year clipped to the lane's window,
        every granule of it, no cap."""
        self._refuse_capped_build()
        lo = max(f10b.seconds_since_epoch(dt.date(int(year), 1, 1)), ctx.t_lo)
        hi = min(f10b.seconds_since_epoch(dt.date(int(year), 12, 31)) + 86399,
                 ctx.t_hi)
        yield from self._rows(ctx, str(year), lo, hi, probe=False)

    def fetch_month(self, ctx, year, month):
        """THE PROBE: the knob's cycle, capped, clipped to the month."""
        lo, hi = f10b.month_bounds_s(year, month)
        yield from self._rows(ctx, f"{year}-{int(month):02d}",
                              max(lo, ctx.t_lo), min(hi, ctx.t_hi),
                              probe=True)

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


def _iso(s):
    """Seconds since 1982-01-01 -> the ISO stamp CMR's `temporal` takes."""
    return (EPOCH + dt.timedelta(seconds=int(s))).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    """The two options the main session decided between (tier P, 2026-09-23).

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
SMOKE_PASSES = ("001", "002", "003", "004")
SMOKE_CYCLE = "010"
# pass 004 starts five seconds before the window's last midnight, so lines
# 0-4 fall inside the window (and inside the probe month) and lines 5-11 fall
# on the next day, which is also the next MONTH: a straddling pass, which the
# store and the probe must both clip rather than keep or drop whole
SMOKE_STRADDLE = "004"
FILL = 2.147483647e9
UNC_FILL = 65535
UNC_SCALE = 1e-4
UNC_VALID_MAX = 60000


def _smoke_times(pss, day, d_hi):
    nl = SMOKE_LINES
    if pss == SMOKE_STRADDLE:
        t0 = ((d_hi + dt.timedelta(days=1)) - dt.date(2000, 1, 1)).days \
            * 86400.0 - 5.0
    else:
        t0 = (day - dt.date(2000, 1, 1)).days * 86400.0 + 3600.0 * int(pss)
    return t0 + np.arange(nl) * 1.0


def write_pass(path, cycle, pss, day, d_hi, seed):
    """One synthetic Expert granule in the product's own packing.

    `ssh_karin_uncert` is written as the product writes it — uint16 x 1e-4 m
    with a fill and a valid_max of 60,000 — and one pixel of pass 003 carries
    raw 62,000 (6.2 m, over valid_max), the value the old reader handed back
    as 62,000 "metres". Returns the physical arrays the reader must produce.
    """
    import netCDF4
    rng = np.random.default_rng(seed + int(pss))
    nl, npx = SMOKE_LINES, SMOKE_PIXELS
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension(DIM_LINES, nl)
    ds.createDimension(DIM_PIXELS, npx)
    ds.cycle_number = int(cycle)
    ds.pass_number = int(pss)
    t = _smoke_times(pss, day, d_hi)
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
    # LINEAR sigma0, as the product stores it: 10-40 dB
    sig0 = np.round(10.0 ** (rng.uniform(1.0, 4.0, (nl, npx))), 2)
    unc_raw = rng.integers(100, 2000, (nl, npx)).astype(np.uint16)
    # the nadir gap: the two middle pixels of every line carry no ssha
    ssha[:, npx // 2 - 1:npx // 2 + 1] = FILL
    if int(pss) == 2:
        ssha[0, 0] = 9.0                     # out of bounds -> NaN, counted
    if int(pss) == 3:
        unc_raw[0, 0] = 62000                # over valid_max -> NaN, counted
        unc_raw[3, 0] = UNC_FILL             # the fill -> NaN, not counted
    for name, a in (("ssha_karin", ssha), ("sig0_karin", sig0)):
        vv = ds.createVariable(name, "f8", (DIM_LINES, DIM_PIXELS),
                               fill_value=FILL)
        vv[:] = a
        vv.units = "m" if name == "ssha_karin" else "1"
    vu = ds.createVariable("ssh_karin_uncert", "u2", (DIM_LINES, DIM_PIXELS),
                           fill_value=np.uint16(UNC_FILL))
    vu.scale_factor = UNC_SCALE
    vu.valid_min = np.uint16(0)
    vu.valid_max = np.uint16(UNC_VALID_MAX)
    vu.units = "m"
    vu.set_auto_maskandscale(False)
    vu[:] = unc_raw
    q = np.zeros((nl, npx), np.uint32)
    q[1, :] = 1 << 24                        # a whole line graded suspect
    q[2, :] = 3 << 24                        # a whole line graded bad
    vq = ds.createVariable("ssh_karin_qual", "u4", (DIM_LINES, DIM_PIXELS))
    vq[:] = q
    vq.flag_meanings = ("suspect_large_ssh_delta degraded_media_delay "
                        "bad_outside_of_range good_measurement")
    vq.flag_masks = np.array([1, 2, 4, 8], np.uint32)
    ds.close()
    unc = unc_raw.astype(np.float64) * UNC_SCALE
    unc[(unc_raw == UNC_FILL) | (unc_raw > UNC_VALID_MAX)] = np.nan
    ssha_p = np.where(ssha == FILL, np.nan, ssha)
    return t, lat, lon, [ssha_p, sig0, unc], q.astype(np.int64) >> 24


def make_smoke_sources(root, d_lo, d_hi, seed=20260918):
    """`<root>/swot/<granule name>.nc` — the flat layout the synthetic CMR
    listing is walked from — and the truth rows.

    The truth is ordered the way the store breaks ties among equal
    (bin, time_s): the order the adapter yields, i.e. pass ascending then the
    file's own (line, pixel) order. It holds only rows whose second lies in
    [d_lo, d_hi] — the straddling pass's lines on the next day belong to the
    next lane — and d_hi must be a month's last day, so the probe's month
    clips the same lines the store's window does.
    """
    base = os.path.join(root, "swot")
    os.makedirs(base, exist_ok=True)
    truth = []
    off = int((dt.date(2000, 1, 1) - dt.date(1982, 1, 1)).days) * 86400
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    t_lo = f10b.seconds_since_epoch(d_lo)
    for i, pss in enumerate(SMOKE_PASSES):
        day = d_lo + dt.timedelta(days=min(i, (d_hi - d_lo).days))
        name = (f"SWOT_L2_LR_SSH_Expert_{SMOKE_CYCLE}_{pss}_"
                f"{day.strftime('%Y%m%dT%H%M%S')}_"
                f"{day.strftime('%Y%m%dT%H%M%S')}_PGD0_02_swot")
        t1, lat, lon, cols, q = write_pass(
            os.path.join(base, name + ".nc"), SMOKE_CYCLE, pss, day, d_hi,
            seed)
        for li in range(SMOKE_LINES):
            ts = int(round(t1[li] + off))
            if not (t_lo <= ts <= t_hi):
                continue
            for pi in range(SMOKE_PIXELS):
                if q[li, pi] >= BAD_GRADE:
                    continue
                if not np.isfinite(cols[0][li, pi]):
                    continue
                vv = []
                for (nm, _u, lo, hi), a in zip(CHANNELS, cols):
                    x = a[li, pi]
                    vv.append(np.nan if not (np.isfinite(x) and lo <= x <= hi)
                              else float(x))
                truth.append({"t": ts,
                              "lat": float(lat[li, pi]),
                              "lon": float(lon[li, pi]),
                              "platform": f10b.platform_hash("SWOT"),
                              "v": vv})
    return truth


ADAPTER = SWOTAdapter
