#!/usr/bin/env python3
"""E-086 · Per-year monthly SUMS and COUNTS of family 7, so a climatology over
ANY set of years can be composed exactly in the browser.

WHAT THIS WRITES. The model's climatology (`ml/trainprobe.py::anomaly_transform`,
published in three fixed versions by `ml/export_family7_clim.py`, E-083) is,
per calendar month, cell and channel, the mean of the stored value over the
training bins whose date falls in that month. A mean over a set of bins is a
SUM divided by a COUNT, and both add across years. So instead of one
climatology per chosen set of years, this script publishes, per group of the
global tensor family 7.2 (stem `family7_global025_pentad_l2`), ONE sum and ONE
count per (year, calendar month, channel, cell):

    <out>/<group>/sum.npy     [12, C, Y, H, W] float32, C order, Z-UNITS (the
                              tensor's stored units, exactly like clim.npy) —
                              the float64 sum of that year's finite bins in
                              that month, rounded once to float32; 0.0 where
                              the count is 0
    <out>/<group>/count.npy   [12, C, Y, H, W] uint8 — how many of those bins
                              were finite (at most 7: a calendar month holds
                              at most seven five-day bins; asserted)
    <out>/<group>/stats.json  the year axis, bins per (year, month), channels,
                              norm (copied from the family-7 index), the
                              static channels (copied from the published
                              climatology — the function's own answer), the
                              tensor sha256 per group, git sha, UTC time

and then, for a set of years Ys, mean = Σ_{y∈Ys} sum / Σ_{y∈Ys} count is the
function's climatology over Ys (to the float32 rounding of each stored sum —
the bound is exact and `verify` checks it).

AXIS ORDER [month, channel, year, lat, lon] (plan §3). One (month, channel,
year) plane is one contiguous range, AND a run of years for one (month,
channel) — "every February from 1991 to 2020" — is ONE contiguous range. Byte
offset of plane (m, c, y) in either file:

    header_len + ((m * C + c) * Y + (y - year_first)) * H * W * itemsize

THE MONTH OF A BIN is the month its five-day window OPENS in — the npz's own
`months` key, parsed by `export_family7_clim.master_calendar` and charged to
each group's OWN rows by `cone_sampler.group_time` (the derivation
anomaly_transform's callers use): `rg100` on its monthly rows, `oc025` from its
own first bin. No second calendar.

NOTHING IS WRITTEN INTO THE TENSOR. The group arrays are read through the
read-only memmap `load_tensor` returns; one (year, month) run of at most seven
bins is in memory at a time (≤ 102 MB of float16 at 0.25°), so the export needs
no more than a few GB of RAM whatever the group size.

    python3 ml/export_family7_monthly.py export --tensor <dir>/<stem>.npz \\
        --index data/family7_index.json --clim-index data/family7_clim_index.json \\
        --out <dir>/monthly [--groups g025,g100,oc025,rg100] [--skip-sha]
    python3 ml/export_family7_monthly.py verify --out <dir>/monthly \\
        --clim-index data/family7_clim_index.json --clim-dir <dir>/clim [--fetch]
    python3 ml/export_family7_monthly.py fixture   # data/family7_monthly/fixture/
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import export_family7_clim as E                                  # noqa: E402

PLAN = "ml/plans/E086_monthly_by_year.md"
PLAN_URL = f"https://blauewelt.github.io/earth/docs.html?f={PLAN}"
AXES = ["month", "channel", "year", "lat", "lon"]
MAX_BINS_PER_MONTH = 7          # ceil(31 / 5): asserted, and why uint8 is enough
MONTH_RULE = (
    "a bin belongs to the calendar month and year its five-day window OPENS "
    "in (the npz's `months` key, through export_family7_clim.master_calendar "
    "and cone_sampler.group_time — the derivation anomaly_transform's callers "
    "use); rg100 on its own monthly rows, oc025 from its own first bin")
LAYOUT = (
    "sum.npy is [month, channel, year, lat, lon] float32 in z-units (physical "
    "mean = z * norm[c][1] + norm[c][0]); count.npy the same shape, uint8. "
    "Plane (m, c, y) starts at header_len + ((m * C + c) * Y + (y - "
    "year_first)) * H * W * itemsize; a run of years for one (m, c) is one "
    "contiguous range. Mean over a set of years = sum of sums / sum of counts; "
    "NaN where the summed count is 0 (sum is 0.0 wherever count is 0).")

FIXTURE_OUT = os.path.join(ROOT, "data", "family7_monthly", "fixture")
FIXTURE_INDEX_OUT = os.path.join(FIXTURE_OUT, "family7_monthly_index.json")
FIXTURE_CLIM_INDEX = os.path.join(ROOT, "data", "family7_clim", "fixture",
                                  "family7_clim_index.json")
FIXTURE_CLIM_DIR = os.path.join(ROOT, "data", "family7_clim", "fixture")
# The same two groups the climatology fixture ships (E-083): both grids, and
# one bin-aligned group next to one with OFFSET rows. All four groups would be
# 3.4 MB for a tensor holding five pentads of January 2010.
FIXTURE_GROUPS = ("g100", "oc025")


# ----------------------------------------------------------------- calendar --
def group_calendars(d, raw):
    """{group: (moy, year, valid)} on each group's OWN rows.

    The master calendar (month and year a pentad opens in) through
    `cone_sampler.group_time`, the same call `export_family7_clim.group_masks`
    makes: a live-bins group (rg100) is charged to its rows' master bins, and a
    row whose master bin is off the axis comes back held out — `valid` False —
    so it can never enter a sum, exactly as it never enters the climatology."""
    from cone_sampler import group_time
    moy_m, years_m = E.master_calendar(d)
    none = np.zeros(len(moy_m), bool)
    out = {}
    for g in raw.groups:
        moy, off = group_time(g, moy_m, none)
        yrs, _ = group_time(g, years_m, none)
        out[g.name] = (np.asarray(moy, np.int64), np.asarray(yrs, np.int64),
                       ~np.asarray(off, bool))
    return out


def month_runs(moy, years, valid):
    """[(lo, hi, year, month)] — each (year, month)'s rows as ONE contiguous
    slice, in time order. Refuses a time axis that revisits a (year, month)
    (unsorted rows would silently split one month's sum into two writes) and
    a month holding more than MAX_BINS_PER_MONTH rows (the uint8 count and the
    whole five-day calendar argument would both be wrong)."""
    moy = np.asarray(moy, np.int64)
    years = np.asarray(years, np.int64)
    valid = np.asarray(valid, bool)
    key = np.where(valid, years * 12 + moy, -1)
    if len(key) == 0:
        return []
    edges = np.flatnonzero(np.diff(key)) + 1
    lo = np.concatenate(([0], edges)).astype(int)
    hi = np.concatenate((edges, [len(key)])).astype(int)
    runs, seen = [], set()
    for a, b in zip(lo, hi):
        k = int(key[a])
        if k < 0:
            continue
        if k in seen:
            raise SystemExit(
                f"REFUSING: (year {k // 12}, month {k % 12 + 1}) appears in "
                f"two separate runs of rows — the group's time axis is not "
                f"sorted, and a month's sum would be written twice")
        seen.add(k)
        if b - a > MAX_BINS_PER_MONTH:
            raise SystemExit(
                f"REFUSING: (year {k // 12}, month {k % 12 + 1}) holds {b - a} "
                f"rows; a calendar month opens at most {MAX_BINS_PER_MONTH} "
                f"five-day bins, and count.npy is uint8 on that bound")
        runs.append((int(a), int(b), k // 12, k % 12))
    if len(runs) > 1:
        ks = [y * 12 + m for _, _, y, m in runs]
        if any(b <= a for a, b in zip(ks, ks[1:])):
            raise SystemExit("REFUSING: the group's (year, month) runs are not "
                             "in increasing time order")
    return runs


def static_channels(clim_index, group, chans):
    """The channels anomaly_transform found static for `group`, copied from
    the published climatology's index — the FUNCTION's answer, which does not
    depend on the year set (its dynamic test runs over every row, held out or
    not). Every version must agree, or the index is refused. None when no
    climatology index covers the group."""
    if clim_index is None:
        return None
    files = clim_index.get("files", {})
    seen = {tuple(files[v][group]["static_chans"]) for v in files
            if group in files[v]}
    if not seen:
        return None
    if len(seen) != 1:
        raise SystemExit(f"{group}: the climatology versions disagree about "
                         f"the static channels ({seen}) — they cannot, the "
                         f"dynamic test ignores the holdout")
    st = list(next(iter(seen)))
    unknown = [c for c in st if c not in chans]
    if unknown:
        raise SystemExit(f"{group}: static channel(s) {unknown} are not in "
                         f"the channel list")
    return st


def plane_offset(header_len, C, Y, H, W, itemsize, m, c, yi):
    """Byte offset of plane (month m, channel c, year index yi) — the formula
    the index states and the page uses."""
    return header_len + ((m * C + c) * Y + yi) * H * W * itemsize


# ------------------------------------------------------------------- export --
def export_group(g, cal, blk, out_dir, *, static, common, verbose=True):
    """Stream one group into <out_dir>/{sum,count}.npy + stats.json.

    The sum is `np.sum(rows, axis=0, where=isfinite(rows), dtype=float64)` —
    the expression anomaly_transform's pass 1 accumulates its climatology
    with. Every float16 value is a multiple of 2**-24 below 2**16, so a
    float64 sum of a few hundred of them is EXACT in any order: the function's
    monthly sum over all training years equals the float64 sum of these
    per-year sums before their float32 rounding, which is the only difference
    `verify` has to bound."""
    from numpy.lib.format import open_memmap
    moy, yrs, valid = cal
    runs = month_runs(moy, yrs, valid)
    if not runs:
        raise SystemExit(f"REFUSING {g.name}: no valid row at all")
    y0 = min(r[2] for r in runs)
    y1 = max(r[2] for r in runs)
    Y, C, H, W = y1 - y0 + 1, g.C, g.H, g.W
    os.makedirs(out_dir, exist_ok=True)
    sum_p = os.path.join(out_dir, "sum.npy")
    cnt_p = os.path.join(out_dir, "count.npy")
    st_p = os.path.join(out_dir, "stats.json")
    for p in (sum_p, cnt_p, st_p):
        if os.path.exists(p):
            os.remove(p)
    shape = (12, C, Y, H, W)
    # open_memmap('w+') creates the file zero-filled, so every (month, year)
    # the group never reaches — oc025 before 1997-09, a month past the record —
    # is sum 0.0 / count 0 without a write.
    S = open_memmap(sum_p + ".tmp", mode="w+", dtype="<f4", shape=shape)
    N = open_memmap(cnt_p + ".tmp", mode="w+", dtype="u1", shape=shape)
    bins = np.zeros((Y, 12), np.int64)
    t0, last = time.time(), time.time()
    max_cnt = 0
    for i, (a, b, y, m) in enumerate(runs):
        rows = g.X[a:b]                                   # read-only memmap
        fin = np.isfinite(rows)
        acc = np.sum(rows, axis=0, where=fin, dtype=np.float64)   # [H,W,C]
        cnt = fin.sum(axis=0, dtype=np.int32)
        del rows, fin
        mx = int(cnt.max()) if cnt.size else 0
        if mx > b - a or mx > MAX_BINS_PER_MONTH:
            raise SystemExit(f"{g.name} {y}-{m + 1:02d}: count {mx} exceeds "
                             f"the {b - a} rows of that month")
        max_cnt = max(max_cnt, mx)
        yi = y - y0
        S[m, :, yi] = np.moveaxis(acc, -1, 0).astype(np.float32)
        N[m, :, yi] = np.moveaxis(cnt, -1, 0).astype(np.uint8)
        bins[yi, m] = b - a
        del acc, cnt
        if verbose and (time.time() - last > 30 or i == len(runs) - 1):
            last = time.time()
            print(f"  [{g.name}] {y}-{m + 1:02d} · {i + 1}/{len(runs)} "
                  f"year-months · {time.time() - t0:.0f} s", flush=True)
    S.flush()
    N.flush()
    del S, N
    os.replace(sum_p + ".tmp", sum_p)
    os.replace(cnt_p + ".tmp", cnt_p)
    years = list(range(y0, y1 + 1))
    n_valid = int(np.asarray(valid, bool).sum())
    if int(bins.sum()) != n_valid:
        raise SystemExit(f"{g.name}: {int(bins.sum())} rows summed but "
                         f"{n_valid} valid rows on the axis")
    st = dict(
        _source="ml/export_family7_monthly.py — do not hand-edit",
        plan=PLAN_URL, group=g.name,
        chans=list(blk["chans"]),
        norm=[[float(a), float(b)] for a, b in blk["norm"]],
        static_chans=static,
        static_chans_source=("data/family7_clim_index.json — the channels "
                             "anomaly_transform subtracts nothing on (its "
                             "climatology is 0.0 there); independent of the "
                             "year set" if static is not None else
                             "no climatology index was given"),
        axes=AXES, shape=list(shape),
        sum_dtype="float32", count_dtype="uint8", units="z-units",
        year_first=y0, year_last=y1, n_years=Y, years=years,
        bins_per_month=bins.tolist(), n_rows=int(g.T), n_rows_summed=n_valid,
        max_count=max_cnt, max_bins_per_month=MAX_BINS_PER_MONTH,
        row_kind="monthly" if blk.get("live_only") else "pentad",
        month_rule=MONTH_RULE, layout=LAYOUT,
        lats=[float(g.lats[0]), float(g.lats[-1])],
        lons=[float(g.lons[0]), float(g.lons[-1])],
        **common)
    E_write_json(st_p, st)            # written LAST: the group's done-marker
    if verbose:
        print(f"  [{g.name}] wrote sum.npy {os.path.getsize(sum_p) / 1e9:.2f} "
              f"GB + count.npy {os.path.getsize(cnt_p) / 1e9:.2f} GB · years "
              f"{y0}–{y1} · max count {max_cnt} · {time.time() - t0:.0f} s",
              flush=True)
    return st


def E_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def export(tensor, index, out, *, clim_index=None, groups=None, skip_sha=False,
           resume=True):
    """Every requested group. Returns {group: stats}."""
    t_start = time.time()
    print(f"export_family7_monthly: {tensor} → {out}", flush=True)
    d, raw, idx, shas, verified, want = E.open_tensor(tensor, index, groups,
                                                      skip_sha)
    cidx = json.load(open(clim_index, encoding="utf-8")) if clim_index else None
    if cidx is not None and cidx["stem"] != idx["stem"]:
        raise SystemExit(f"REFUSING: {clim_index} describes {cidx['stem']!r}, "
                         f"the tensor is {idx['stem']!r}")
    cals = group_calendars(d, raw)
    by_name = {g.name: g for g in raw.groups}
    common = dict(
        tensor_stem=idx["stem"],
        tensor_sha256_by_group={k: idx["groups"][k]["sha256"]
                                for k in idx["groups"]},
        git_sha=E.git_sha(), generated_utc=E.now_utc())
    res = {}
    for gname in want:
        where = os.path.join(out, gname)
        st_p = os.path.join(where, "stats.json")
        if resume and os.path.exists(st_p) and all(
                os.path.exists(os.path.join(where, f))
                for f in ("sum.npy", "count.npy")):
            st = json.load(open(st_p, encoding="utf-8"))
            if (st.get("tensor_sha256") == shas[gname]
                    and st.get("tensor_sha256_verified") == verified[gname]):
                print(f"[{gname}] already exported (stats.json present) — "
                      f"skipped", flush=True)
                res[gname] = st
                continue
        g = by_name[gname]
        blk = idx["groups"][gname]
        st = None if cidx is None else static_channels(cidx, gname,
                                                       blk["chans"])
        print(f"[{gname}] {g.T} rows [{g.T}x{g.H}x{g.W}x{g.C}] {g.X.dtype}",
              flush=True)
        res[gname] = export_group(
            g, cals[gname], blk, where, static=st,
            common=dict(common, tensor_sha256=shas[gname],
                        tensor_sha256_verified=bool(verified[gname])))
    print(f"export_family7_monthly: done in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    return res


# ------------------------------------------------------------------- verify --
def year_set(version, years):
    """[Y] bool — True where year years[i] is a TRAINING year of `version`
    (export_family7_clim.held_out, the published versions' own rule)."""
    return ~E.held_out(version, np.asarray(years, np.int64))


def compose(S, N, keep):
    """(mean float64 [H, W], count int64 [H, W], bound float64 [H, W]) for one
    (month, channel) from its [Y, H, W] sums and counts over the years `keep`.

    `bound` is a rigorous bound on |mean − the function's float64 mean|: the
    float64 sums of float32 sums and of float16 values are both exact (all are
    multiples of 2**-24 below 2**25), so the only error is each stored sum's
    float32 rounding (≤ half its own spacing), plus float64 division."""
    s = S[keep].astype(np.float64)
    n = N[keep].astype(np.int64).sum(axis=0)
    half_ulp = 0.5 * np.spacing(np.abs(S[keep])).astype(np.float64).sum(axis=0)
    tot = s.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
        bound = half_ulp / np.maximum(n, 1)
    return mean, n, bound


def compare_plane(mean, n, bound, pub, *, static=False):
    """Check one composed plane against the published float32 climatology.

    Returns dict(max_abs, max_ulp, max_over_bound, n_finite, ok, why)."""
    pub = np.asarray(pub)
    if static:
        ok = bool((pub == 0).all())
        return dict(max_abs=0.0, max_ulp=0.0, max_over_bound=0.0,
                    n_finite=0, ok=ok,
                    why=None if ok else "static channel is not 0.0 in clim")
    nan_pub = np.isnan(pub)
    if not np.array_equal(nan_pub, n == 0):
        bad = int((nan_pub != (n == 0)).sum())
        return dict(max_abs=float("inf"), max_ulp=float("inf"),
                    max_over_bound=float("inf"), n_finite=0, ok=False,
                    why=f"{bad} cell(s) where 'no sample' differs")
    fin = ~nan_pub
    if not fin.any():
        return dict(max_abs=0.0, max_ulp=0.0, max_over_bound=0.0, n_finite=0,
                    ok=True, why=None)
    p64 = pub[fin].astype(np.float64)
    diff = np.abs(mean[fin] - p64)
    # half an ulp of the published float32 rounding + float64 division slack
    tol = bound[fin] + 0.5 * np.spacing(np.abs(pub[fin])).astype(np.float64) \
        + 4 * np.finfo(np.float64).eps * np.abs(p64) + 1e-300
    ulp = np.spacing(np.abs(pub[fin])).astype(np.float64)
    over = diff / tol
    ok = bool((diff <= tol).all())
    return dict(max_abs=float(diff.max()), max_ulp=float((diff / ulp).max()),
                max_over_bound=float(over.max()), n_finite=int(fin.sum()),
                ok=ok, why=None if ok else
                f"{int((diff > tol).sum())} cell(s) beyond the rigorous bound")


def verify(out, clim_index, clim_dir, *, groups=None, versions=None,
           fetch=False, report=None, verbose=True):
    """THE FALSIFIER (plan §5): for every published climatology version and
    group, Σsum/Σcount over that version's training years must reproduce the
    published clim.npy — NaN exactly where it is NaN, values within the
    rigorous float32 bound — and Σ bins_per_month must equal the version's
    n_train_bins. Writes `report` (verify.json) and raises SystemExit on any
    failure."""
    cidx = json.load(open(clim_index, encoding="utf-8"))
    vers = versions or [v["key"] for v in cidx["versions"]]
    gnames = groups or [g for g in cidx["groups"]
                        if os.path.exists(os.path.join(out, g, "stats.json"))]
    if not gnames:
        raise SystemExit(f"{out}: no exported group to verify")
    if fetch:
        fetch_clim(cidx, clim_dir, vers, gnames)
    result = dict(_source="ml/export_family7_monthly.py verify",
                  generated_utc=E.now_utc(), clim_index=os.path.relpath(
                      os.path.abspath(clim_index), ROOT),
                  stem=cidx["stem"], groups={}, ok=True,
                  tolerance=("|Σsum/Σcount − clim| ≤ Σ_y ½ulp32(sum_y)/Σcount "
                             "+ ½ulp32(clim) + 4·eps64·|clim| per cell — a "
                             "rigorous bound: the float64 sums are exact, so "
                             "only the float32 rounding of each stored "
                             "per-year sum and of the published mean remain"))
    failures = []
    for gname in gnames:
        st = json.load(open(os.path.join(out, gname, "stats.json"),
                            encoding="utf-8"))
        if st["tensor_stem"] != cidx["stem"]:
            raise SystemExit(f"{gname}: monthly export is of "
                             f"{st['tensor_stem']!r}, the climatology of "
                             f"{cidx['stem']!r}")
        S_all = np.load(os.path.join(out, gname, "sum.npy"), mmap_mode="r")
        N_all = np.load(os.path.join(out, gname, "count.npy"), mmap_mode="r")
        _, C, Y, H, W = S_all.shape
        years = st["years"]
        bins = np.asarray(st["bins_per_month"], np.int64)
        static = set(st.get("static_chans") or [])
        norm = np.asarray(st["norm"], np.float64)
        gres = {}
        pubs = {}
        for v in vers:
            rec = cidx["files"][v][gname]
            p = os.path.join(clim_dir, v, gname, "clim.npy")
            if not os.path.exists(p):
                raise SystemExit(f"{p}: the published climatology is not here "
                                 f"(pass --fetch)")
            pubs[v] = np.load(p, mmap_mode="r")
            if list(pubs[v].shape) != [12, C, H, W]:
                raise SystemExit(f"{p}: shape {list(pubs[v].shape)} vs the "
                                 f"monthly export's [12, {C}, {H}, {W}]")
            keep = year_set(v, years)
            n_rows = int(bins[keep].sum())
            if n_rows != int(rec["n_train_bins"]):
                failures.append(f"{v}/{gname}: Σ bins_per_month over the "
                                f"version's years = {n_rows}, the climatology "
                                f"has n_train_bins {rec['n_train_bins']}")
            gres[v] = dict(n_train_bins=n_rows,
                           n_train_bins_published=int(rec["n_train_bins"]),
                           years_kept=[int(y) for y, k in zip(years, keep)
                                       if k],
                           max_abs_z=0.0, max_abs_phys=0.0, max_ulp=0.0,
                           max_over_bound=0.0, n_cells=0, ok=True)
        t0 = time.time()
        for m in range(12):
            for c in range(C):
                S = np.asarray(S_all[m, c])               # [Y, H, W]
                N = np.asarray(N_all[m, c])
                for v in vers:
                    keep = year_set(v, years)
                    mean, n, bound = compose(S, N, keep)
                    r = compare_plane(mean, n, bound,
                                      np.asarray(pubs[v][m, c]),
                                      static=st["chans"][c] in static)
                    g = gres[v]
                    g["max_abs_z"] = max(g["max_abs_z"], r["max_abs"])
                    g["max_abs_phys"] = max(g["max_abs_phys"],
                                            r["max_abs"] * float(norm[c][1]))
                    g["max_ulp"] = max(g["max_ulp"], r["max_ulp"])
                    g["max_over_bound"] = max(g["max_over_bound"],
                                              r["max_over_bound"])
                    g["n_cells"] += r["n_finite"]
                    if not r["ok"]:
                        g["ok"] = False
                        failures.append(f"{v}/{gname} month {m + 1} "
                                        f"{st['chans'][c]}: {r['why']}")
            if verbose and (m == 11 or time.time() - t0 > 60):
                print(f"  verify [{gname}] month {m + 1}/12 "
                      f"({time.time() - t0:.0f} s)", flush=True)
        for v in vers:
            g = gres[v]
            print(f"  verify {v}/{gname}: {g['n_cells']} cell-months · max "
                  f"|Δ| {g['max_abs_z']:.3g} z ({g['max_abs_phys']:.3g} in "
                  f"the channel's unit) · max {g['max_ulp']:.2f} ulp · "
                  f"max Δ/bound {g['max_over_bound']:.3f} · "
                  f"{'OK' if g['ok'] else 'FAIL'}", flush=True)
        result["groups"][gname] = gres
    result["ok"] = not failures
    result["failures"] = failures[:50]
    if report:
        E_write_json(report, result)
    if failures:
        raise SystemExit("VERIFY FAILED — the per-year sums do not reproduce "
                         "the published climatology:\n  " +
                         "\n  ".join(failures[:20]))
    return result


def fetch_clim(cidx, clim_dir, versions, groups):
    """Pull the published clim.npy files (sha256-checked, parallel ranges)."""
    import requests
    from pull_family7_tensor import pull_file
    sess = requests.Session()
    for v in versions:
        for g in groups:
            rec = cidx["files"][v][g]["clim_npy"]
            dest = os.path.join(clim_dir, v, g, "clim.npy")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            pull_file(sess, rec["url"], dest, rec["sha256"], workers=8,
                      chunk=32 << 20, size=rec["bytes"])


# ------------------------------------------------------------------ fixture --
def run_fixture(out=FIXTURE_OUT, index_out=FIXTURE_INDEX_OUT,
                groups=FIXTURE_GROUPS):
    """The whole path on the committed smoke tensor, no network: export,
    the falsifier against the committed climatology fixture, the index."""
    import publish_family7_monthly_index as P
    work = tempfile.mkdtemp(prefix="f7mon_fix_")
    try:
        npz = E.fixture_npz(work)
        for g in groups:
            shutil.rmtree(os.path.join(out, g), ignore_errors=True)
        export(npz, E.FIXTURE_INDEX, out, clim_index=FIXTURE_CLIM_INDEX,
               groups=list(groups), resume=False)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rep = os.path.join(tempfile.mkdtemp(prefix="f7mon_ver_"), "verify.json")
    verify(out, FIXTURE_CLIM_INDEX, FIXTURE_CLIM_DIR, groups=list(groups),
           report=rep)
    rel = os.path.relpath(out, ROOT)
    return P.main(["index", "--local", "--out", out, "--f7-index",
                   E.FIXTURE_INDEX, "--index", index_out, "--no-cors",
                   "--verify", rep, "--fixture-dir", rel + "/"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("export")
    a1.add_argument("--tensor", required=True)
    a1.add_argument("--index", default=os.path.join(ROOT, "data",
                                                    "family7_index.json"))
    a1.add_argument("--clim-index", default=os.path.join(
        ROOT, "data", "family7_clim_index.json"),
        help="the published climatology's index — source of the static "
             "channels ('' to omit)")
    a1.add_argument("--out", required=True)
    a1.add_argument("--groups", default="")
    a1.add_argument("--skip-sha", action="store_true")
    a1.add_argument("--no-resume", action="store_true")
    a2 = sub.add_parser("verify")
    a2.add_argument("--out", required=True)
    a2.add_argument("--clim-index", default=os.path.join(
        ROOT, "data", "family7_clim_index.json"))
    a2.add_argument("--clim-dir", required=True)
    a2.add_argument("--groups", default="")
    a2.add_argument("--versions", default="")
    a2.add_argument("--fetch", action="store_true",
                    help="pull the published clim.npy files first")
    a2.add_argument("--report", default="",
                    help="write verify.json here (default <out>/verify.json)")
    sub.add_parser("fixture")
    a = ap.parse_args(argv)
    lst = (lambda s: [x.strip() for x in s.split(",") if x.strip()])
    if a.cmd == "export":
        export(a.tensor, a.index, a.out, clim_index=a.clim_index or None,
               groups=lst(a.groups) or None, skip_sha=a.skip_sha,
               resume=not a.no_resume)
    elif a.cmd == "verify":
        verify(a.out, a.clim_index, a.clim_dir, groups=lst(a.groups) or None,
               versions=lst(a.versions) or None, fetch=a.fetch,
               report=a.report or os.path.join(a.out, "verify.json"))
    else:
        run_fixture()
    return 0


if __name__ == "__main__":
    sys.exit(main())
