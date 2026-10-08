#!/usr/bin/env python3
"""E-091 · DAY-OF-YEAR CLIMATOLOGY sums of the gridded stores, CUMULATIVE over
the years, so the Data tab can give "the mean of every calendar day over any
span of years" from two small range reads per day instead of every native map.

WHAT THIS WRITES. For one sharded tier-G store, per calendar day (month, day —
366 slots, 29 February its own):

    <out>/sum_MMDD.npy    [C, Y, H, W] float32 — plane (c, y) is the sum of
                          every finite frame of that calendar day in the years
                          year_first … year_first + y (float64 running total,
                          rounded once per plane), in the store's physical units
    <out>/count_MMDD.npy  [C, Y, H, W] uint8 — how many frames that sum holds
    <out>/manifest_MM.json  per file: bytes, sha256, .npy header; the years;
                          the falsifier's numbers

    mean of day d over the years a … b  = (S[b] − S[a−1]) / (N[b] − N[a−1])
                                           (S[a−1], N[a−1] = 0 when a is the first year)
    leaving a year e out                = subtract (S[e] − S[e−1]) and (N[e] − N[e−1])

Plane (c, y) starts at header_len + (c·Y + y)·H·W·itemsize. A year with no
frame on that day (29 February outside leap years, a gap) repeats the plane
before it. The DAY of a frame is the UTC calendar day of its start instant
(`sharded.frame_datetime`); a six-hourly store therefore puts four frames in a
day and the mean is the daily mean.

WHY CUMULATIVE. A 20-year span is then 2 reads per day whatever its length;
per-year planes would be 20. The price is float32 rounding of a running total:
the falsifier below measures it against numpy on the native frames and refuses
the upload when it is over the bound.

One job does ONE calendar month (its 28–31 day files), so twelve hosted
runners build a store in parallel and none needs more than a month of source
and a month of output on disk.

    python3 ml/export_gridded_doy.py month --store family7_2d/oisst025d --month 2 --out DIR [--upload]
    python3 ml/export_gridded_doy.py index [--stores a,b] [--local DIR]
"""
import argparse
import calendar
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import export_gridded_monthly as X                               # noqa: E402
from family1 import sharded as sh                                # noqa: E402

PLAN = "ml/plans/E091_day_of_year_climatology.md"
VERSION = "v1"
REPO_ID = "chfrank/earth-tensors"
INDEX = os.path.join(ROOT, "data", "gridded_doy_index.json")
STORES = ("family7_2d/oisst025d", "family7_2d/ncep100d",
          "family7_2d/glorys025d", "family7_2d/occci025d",
          "family1_2/era5_t", "family1_2/era5_q",
          "family1_2/era5_u", "family1_2/era5_v")
SAMPLE_DAYS = (1, 15, 29)          # days of the month the falsifier keeps


def prefix_of(store):
    return f"tensors/{store}/doy/{VERSION}"


def spans_to_check(Y):
    """(a, b) year-index spans: whole record, a 20-year span, single years."""
    out = [(0, Y - 1), (Y - 1, Y - 1), (0, 0)]
    if Y > 21:
        out.append((Y - 21, Y - 2))
    if Y > 3:
        out.append((Y // 2, Y // 2))
        out.append((1, Y - 2))
    return sorted(set(out))


def export_month(store, month, out, *, base=None, threads=6):
    """Write the day files of calendar `month` (1..12); returns the manifest."""
    from numpy.lib.format import open_memmap
    t0 = time.time()
    base = base or (X.HUB + store)
    src, meta, spec, arr, g = X.open_store(base, store)
    sha_map = meta.get("sha256") or {}
    times = X.frame_times(arr, spec)
    y0, y1 = times[0][2].year, times[-1][2].year
    years = list(range(y0, y1 + 1))
    Y = len(years)
    H, W = spec["H"], spec["W"]
    chans = [c["name"] if isinstance(c, dict) else c for c in spec["channels"]]
    C = len(chans)
    nd = 29 if month == 2 else calendar.monthrange(2001, month)[1]
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "_spec", g), exist_ok=True)
    with open(os.path.join(out, "_spec", g, "tile_grid.json"), "w") as fh:
        json.dump(spec, fh)

    # frames of this month, per year and day
    per = {y: {d: [] for d in range(1, nd + 1)} for y in years}
    for b, f, t in times:
        if t.month == month:
            per[t.year][t.day].append((b, f))

    S = [open_memmap(os.path.join(out, f"sum_{month:02d}{d:02d}.npy"), "w+",
                     np.float32, (C, Y, H, W)) for d in range(1, nd + 1)]
    N = [open_memmap(os.path.join(out, f"count_{month:02d}{d:02d}.npy"), "w+",
                     np.uint8, (C, Y, H, W)) for d in range(1, nd + 1)]
    cs = np.zeros((nd, C, H, W), np.float64)
    cn = np.zeros((nd, C, H, W), np.int32)
    keep = {d: {} for d in SAMPLE_DAYS if d <= nd}     # d -> {yi: [frames]}
    got = 0
    for yi, y in enumerate(years):
        fl = [bf for d in per[y] for bf in per[y][d]]
        if fl:
            tmp = os.path.join(out, f"_bins_{y}")
            bins = sorted({b for b, _ in fl})
            got += X.fetch_bins(base, g, bins, tmp, sha_map, threads)
            os.makedirs(os.path.join(tmp, g), exist_ok=True)
            shutil.copy(os.path.join(out, "_spec", g, "tile_grid.json"),
                        os.path.join(tmp, g, "tile_grid.json"))
            grp = sh.ShardedGroup(os.path.join(tmp, g))
            for d in range(1, nd + 1):
                for b, f in per[y][d]:
                    fr = grp.read_frame(b, f)
                    if fr is None:
                        raise SystemExit(f"{g} {y}-{month}-{d}: a frame the "
                                         f"shard index lists is absent")
                    fr = np.moveaxis(fr, -1, 0)              # [C, H, W]
                    fin = np.isfinite(fr)
                    cs[d - 1] += np.where(fin, fr, 0.0)
                    cn[d - 1] += fin
                    if d in keep:
                        keep[d].setdefault(yi, []).append(fr.copy())
            del grp
            shutil.rmtree(tmp, ignore_errors=True)
        if cn.max() > 255:
            raise SystemExit(f"{g}: a cumulative count passed 255 in {y}")
        for d in range(nd):
            S[d][:, yi] = cs[d].astype(np.float32)
            N[d][:, yi] = cn[d].astype(np.uint8)
        print(f"  {g} month {month:02d} · {y}: {len(fl)} frames "
              f"({got / 1e9:.2f} GB read, {time.time() - t0:.0f} s)", flush=True)
    for a in S + N:
        a.flush()
    del S, N

    # ---- the falsifier: the composed mean against numpy on the native frames
    worst, worst_ratio, checks = 0.0, 0.0, 0
    for d, by_year in keep.items():
        Sd = np.load(os.path.join(out, f"sum_{month:02d}{d:02d}.npy"), mmap_mode="r")
        Nd = np.load(os.path.join(out, f"count_{month:02d}{d:02d}.npy"), mmap_mode="r")
        for a, b in spans_to_check(Y):
            frs = [f for yi in range(a, b + 1) for f in by_year.get(yi, [])]
            s = Sd[:, b].astype(np.float64) - (Sd[:, a - 1] if a else 0.0)
            n = Nd[:, b].astype(np.int64) - (Nd[:, a - 1] if a else 0)
            if not frs:
                if n.any():
                    raise SystemExit(f"falsifier: {month:02d}{d:02d} span {a}-{b} "
                                     f"has counts with no frame")
                continue
            st = np.stack(frs).astype(np.float64)
            fin = np.isfinite(st)
            rn = fin.sum(0)
            if not np.array_equal(rn, n):
                raise SystemExit(f"falsifier: {month:02d}{d:02d} span {a}-{b}: "
                                 f"counts differ from the native frames")
            ok = rn > 0
            ref = np.where(fin, st, 0.0).sum(0)[ok] / rn[ok]
            mean = s[ok] / n[ok]
            err = np.abs(mean - ref)
            # two float32 roundings of running totals — the plane at the
            # span's end and the one before its start — each ≤ ½ ulp of ITS
            # OWN magnitude (a signed field's total can be near zero at one
            # and large at the other)
            ulp = np.spacing(np.abs(Sd[:, b]).astype(np.float32)).astype(np.float64)
            if a:
                ulp = ulp + np.spacing(np.abs(Sd[:, a - 1]).astype(np.float32)).astype(np.float64)
            bound = 0.5 * ulp[ok] / n[ok] * (1 + 1e-9) + 1e-12 * np.abs(ref) + 1e-30
            worst = max(worst, float(err.max()))
            worst_ratio = max(worst_ratio, float((err / bound).max()))
            checks += 1
        del Sd, Nd
    print(f"  falsifier: {checks} spans, max |Δmean| {worst:.3g}, "
          f"max Δ/bound {worst_ratio:.3f}", flush=True)
    if worst_ratio > 1.0:
        raise SystemExit("falsifier FAILED: a composed mean is outside the "
                         "float32 rounding bound — nothing is uploaded")

    from publish_family7_index import parse_npy_header, sha256_file
    files = {}
    for d in range(1, nd + 1):
        for role in ("sum", "count"):
            name = f"{role}_{month:02d}{d:02d}.npy"
            p = os.path.join(out, name)
            with open(p, "rb") as fh:
                head = fh.read(4096)
            hl, shp, dty, _ = parse_npy_header(head)
            files[name] = dict(bytes=os.path.getsize(p), sha256=sha256_file(p),
                               header_len=hl, shape=list(shp), dtype=dty)
    lat, lon, reg = X.grid_axes(spec)
    lic = meta.get("licence") or meta.get("license")
    man = dict(
        _source="ml/export_gridded_doy.py", plan=PLAN, store=store,
        prefix=prefix_of(store), month=month, days=nd, years=years,
        year_first=y0, year_last=y1, chans=chans,
        channels=spec["channels"], levels_hpa=spec.get("levels_hpa") or meta.get("levels_hpa"),
        H=H, W=W, lat0=float(lat[0]), dlat=float(lat[1] - lat[0]),
        lon0=float(lon[0]), dlon=float(lon[1] - lon[0]), registration=reg,
        frame_seconds=spec["frame_seconds"],
        record_first=times[0][2].strftime("%Y-%m-%dT%H:%M:%SZ"),
        record_last=times[-1][2].strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_store=f"tensors/{store}", licence=lic,
        falsifier=dict(spans=checks, max_abs_mean=worst,
                       max_mean_over_bound=worst_ratio,
                       days_sampled=sorted(keep)),
        files=files, git_sha=X.git_sha(), generated_utc=X.now_utc(),
        seconds=round(time.time() - t0), bytes_read=got)
    X.write_json(os.path.join(out, f"manifest_{month:02d}.json"), man)
    shutil.rmtree(os.path.join(out, "_spec"), ignore_errors=True)
    return man


def upload_month(store, month, out):
    """One Hub commit of the month's files, then sizes and sha256 read back
    from the Hub's own file records."""
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise SystemExit("no HF_TOKEN in the environment")
    api = HfApi(token=tok)
    man = json.load(open(os.path.join(out, f"manifest_{month:02d}.json")))
    pre = prefix_of(store)
    t0 = time.time()
    gb = sum(f["bytes"] for f in man["files"].values()) / 1e9
    print(f"  uploading {len(man['files'])} files, {gb:.2f} GB to {pre} …", flush=True)
    for attempt in range(5):
        try:
            api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=out,
                              path_in_repo=pre, allow_patterns=[
                                  f"sum_{month:02d}*.npy", f"count_{month:02d}*.npy",
                                  f"manifest_{month:02d}.json"],
                              commit_message=f"E-091: day-of-year sums of {store}, month {month:02d}")
            break
        except Exception as e:                                # noqa: BLE001
            print(f"    upload attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 4:
                raise
            time.sleep(30 * (attempt + 1))
    infos = {i.path: i for i in api.get_paths_info(
        REPO_ID, [f"{pre}/{n}" for n in man["files"]], repo_type="dataset")}
    for n, f in man["files"].items():
        i = infos.get(f"{pre}/{n}")
        if i is None or i.size != f["bytes"] or (i.lfs and i.lfs.sha256 != f["sha256"]):
            raise SystemExit(f"read-back: {pre}/{n} is not what was written")
    print(f"  committed and read back in {time.time() - t0:.0f} s", flush=True)


def build_index(stores, fetch=None):
    """data/gridded_doy_index.json from the twelve manifests of each store."""
    import requests
    out = dict(_source="ml/export_gridded_doy.py index — do not hand-edit",
               plan=f"https://blauewelt.github.io/earth/docs.html?f={PLAN}",
               base=f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/",
               layout="sum_MMDD.npy [channel, year, lat, lon] float32, "
                      "count_MMDD.npy the same shape uint8, both CUMULATIVE over "
                      "the years: plane (c, y) = every frame of that calendar "
                      "day in years year_first … year_first + y. Plane offset = "
                      "header_len + (c * n_years + y) * H * W * itemsize.",
               combine="mean of a day over years a…b = (S[b] − S[a−1]) / "
                       "(N[b] − N[a−1]); a year e left out subtracts "
                       "(S[e] − S[e−1]) and (N[e] − N[e−1])",
               generated_utc=X.now_utc(), stores={})
    for store in stores:
        pre = prefix_of(store)
        blk, days, fals = None, {}, []
        for m in range(1, 13):
            url = f"{out['base']}{pre}/manifest_{m:02d}.json"
            if fetch:
                man = fetch(store, m)
            else:
                r = requests.get(url, timeout=120)
                if r.status_code == 404:
                    man = None
                else:
                    r.raise_for_status()
                    man = r.json()
            if man is None:
                continue
            if blk is None:
                blk = {k: man[k] for k in (
                    "store", "prefix", "years", "year_first", "year_last", "chans",
                    "channels", "levels_hpa", "H", "W", "lat0", "dlat", "lon0",
                    "dlon", "registration", "frame_seconds", "record_first",
                    "record_last", "source_store", "licence")}
                blk["n_years"] = len(man["years"])
            elif man["years"] != blk["years"] or man["chans"] != blk["chans"]:
                raise SystemExit(f"{store}: month {m} was built over a different "
                                 f"record than month 1 — rebuild the store")
            for d in range(1, man["days"] + 1):
                s = man["files"][f"sum_{m:02d}{d:02d}.npy"]
                c = man["files"][f"count_{m:02d}{d:02d}.npy"]
                days[f"{m:02d}{d:02d}"] = dict(
                    sum_bytes=s["bytes"], sum_header=s["header_len"],
                    count_bytes=c["bytes"], count_header=c["header_len"])
                if s["shape"] != [len(blk["chans"]), blk["n_years"], blk["H"], blk["W"]]:
                    raise SystemExit(f"{store} {m:02d}{d:02d}: shape {s['shape']}")
            fals.append(man["falsifier"])
        if blk is None:
            print(f"  {store}: nothing published yet")
            continue
        blk["days"] = days
        blk["complete"] = len(days) == 366
        blk["falsifier"] = dict(
            max_abs_mean=max(f["max_abs_mean"] for f in fals),
            max_mean_over_bound=max(f["max_mean_over_bound"] for f in fals))
        out["stores"][store] = blk
        print(f"  {store}: {len(days)} of 366 days")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("month")
    m.add_argument("--store", required=True)
    m.add_argument("--month", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--base")
    m.add_argument("--upload", action="store_true")
    i = sub.add_parser("index")
    i.add_argument("--stores", default=",".join(STORES))
    i.add_argument("--index", default=INDEX)
    a = ap.parse_args(argv)
    if a.cmd == "month":
        export_month(a.store, a.month, a.out, base=a.base)
        if a.upload:
            upload_month(a.store, a.month, a.out)
    else:
        idx = build_index([s for s in a.stores.split(",") if s])
        X.write_json(a.index, idx)
        print(f"wrote {a.index}")


if __name__ == "__main__":
    main()
