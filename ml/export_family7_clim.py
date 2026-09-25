#!/usr/bin/env python3
"""E-083 §3 · Export the model's climatology of family 7, in three versions.

WHAT THIS WRITES. Every forecaster in this programme sees a channel as a
departure from its own per-calendar-month climatology — the mean of that
channel, at that cell, over the training bins that fall in that calendar
month — computed by the ONE function that does it,
`ml/trainprobe.py::anomaly_transform`, and handed back through its `stats`
argument as `stats["clim"]` ([12, H, W, C] float32, NaN where a (month, cell,
channel) had no training sample, 0.0 on static channels). That field is also
the baseline `msss_clim` scores every forecast against. It has only ever
existed in a rented box's RAM; this script writes it to disk for the four
family-7 groups (`g025`, `g100`, `oc025`, `rg100`) under three answers to
"which bins are training" (the `VERSIONS` table, plan §2):

    <out>/<version>/<group>/clim.npy    [12, C, H, W] float32, C order — one
                                        (month, channel) plane is one
                                        contiguous slab, one Range: read
    <out>/<version>/<group>/clim.nc     the same field as CF NetCDF, one
                                        variable per channel in PHYSICAL units
                                        (z * sd + mean from the index `norm`)
    <out>/<version>/<group>/stats.json  mu, sd, den, dynamic, channels, norm,
                                        the version's rule, n_train_bins, the
                                        tensor's sha256s, git sha, UTC time

NO SECOND IMPLEMENTATION. The climatology is `stats["clim"]` from a call to
`anomaly_transform` on a writable copy of each group — the function is
imported, or (without torch) lifted out of `ml/trainprobe.py` by `ast` through
`ml/export_cone_sample.py::_anomaly_transform`, byte for byte.
`tests/test_one_anomaly_transform.py` forbids a copy and
`tests/test_export_family7_clim.py` pins that `clim.npy` IS the function's
array, bit for bit, after the one transpose.

EACH GROUP ON ITS OWN MONTHS. `rg100` holds one row per month and `oc025`
starts at bin 1145 (1997-09-04), so each group's (moy, t_hold) comes from
`ml/cone_sampler.py::group_time` over the master calendar — the same
derivation the trainer and the cone export use.

THE CHECKS (ml/CLAUDE.md §0): every group `.npy` is sha256-verified against
the family-7 index BEFORE anything is computed (the climatology of the wrong
bytes looks exactly like a climatology; `--skip-sha` only for toy tensors);
the tensor's stem, shapes and channel lists must match the index; NetCDF is
importable before the first hour is spent; a (version, group) with no
training bin, or a climatology with no finite value, is refused rather than
written.

Run (on the box, beside the tensor — `ml/pull_family7_tensor.py` fetches it):
    python3 ml/export_family7_clim.py \\
        --tensor /data/family7_global025_pentad_l2.npz \\
        --index data/family7_index.json --out /data/clim \\
        [--versions all,dev,paper] [--groups g025,g100,oc025,rg100] \\
        [--chunk N] [--scratch DIR | --copy ram]
    python3 ml/publish_family7_clim_index.py upload --out /data/clim
Fixture (no network; writes data/family7_clim/fixture/ + its index):
    python3 ml/export_family7_clim.py --fixture
"""
import argparse
import ast
import datetime as _dt
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PLAN = "ml/plans/E083_model_climatology.md"
PLAN_URL = f"https://blauewelt.github.io/earth/docs.html?f={PLAN}"
TRANSFORM_SRC = "ml/trainprobe.py::anomaly_transform"

# ------------------------------------------------------------------ versions --
# Plan §2, verbatim. A version is a YEAR RULE: a bin is held out of the
# climatology (and of every pooled statistic) when its calendar year is in
# `holdout_years`, or is >= `holdout_from` when that is set. `holdout_from` is
# the contiguous retrospective period of the paper's evaluation contract
# (ml/paper/paper.tex, "Training is 1982–2020 less the development years 2009
# and 2017 … Retrospective evaluation is 2021–2024").
VERSIONS = (
    dict(key="all",
         name="All years (1982–2024)",
         holdout_years=[],
         holdout_from=None,
         train_span="1982–2024",
         rule="Every bin of the record is a training bin; nothing is held "
              "out."),
    dict(key="dev",
         name="Development holdout — 2009, 2017 and 2023 held out",
         holdout_years=[2009, 2017, 2023],
         holdout_from=None,
         train_span="1982–2024 less 2009, 2017 and 2023",
         rule="Every bin whose calendar year is 2009, 2017 or 2023 is held "
              "out of the climatology and of every statistic; all other "
              "years are training (the E-059 regime, "
              "ml/export_cone_sample.py::HOLDOUT_YEARS)."),
    dict(key="paper",
         name="The paper's split — trained on 1982–2020 less 2009 and 2017",
         holdout_years=[2009, 2017],
         holdout_from=2021,
         train_span="1982–2020 less 2009 and 2017",
         rule="Training is 1982–2020 less the development years 2009 and "
              "2017; those two years and every bin from 2021-01-01 on (the "
              "retrospective evaluation period 2021–2024) are held out of "
              "the climatology and of every statistic."),
)
VERSION_BY_KEY = {v["key"]: v for v in VERSIONS}

FIXTURE_SRC = os.path.join(ROOT, "data", "family7", "fixture")
FIXTURE_INDEX = os.path.join(FIXTURE_SRC, "family7_index.json")
FIXTURE_OUT = os.path.join(ROOT, "data", "family7_clim", "fixture")
FIXTURE_CLIM_INDEX = os.path.join(FIXTURE_OUT, "family7_clim_index.json")
# The fixture ships the two groups that exercise both grids and both row
# layouts the page has to handle — `g100` (1°-class grid, 15 channels, bin-
# aligned) and `oc025` (0.25°-class grid, OFFSET rows) — in all three
# versions. All four groups in all three versions is 8.1 MB of float32 for a
# tensor that covers five pentads of January 2010; these two are 2.2 MB.
FIXTURE_GROUPS = ("g100", "oc025")
# And no `clim.nc` in git: 0.34 MB of NetCDF for the six (version, group)
# pairs would take the fixture from 2.25 to 2.6 MB for a file the page only
# LINKS to. The fixture index says `clim_nc: null`; the NetCDF writer is
# exercised by tests/test_export_family7_clim.py on the toy tensor instead.
FIXTURE_NC = False


def version_block(v):
    """The version as it is written into stats.json, the NetCDF and the index."""
    return {k: v[k] for k in ("key", "name", "rule", "holdout_years",
                              "holdout_from", "train_span")}


def held_out(version, years):
    """[n] bool — True where a bin of calendar year `years[i]` is held out."""
    years = np.asarray(years, np.int64)
    v = VERSION_BY_KEY[version] if isinstance(version, str) else version
    hold = np.isin(years, np.asarray(v["holdout_years"], np.int64))
    if v["holdout_from"] is not None:
        hold |= years >= int(v["holdout_from"])
    return hold


def master_calendar(d):
    """(moy, years) per MASTER row, from the npz's own `months` ("YYYY-MM").

    The same parse `ml/export_cone_sample.py::_open_family7` makes: the month
    a row belongs to is the month its pentad OPENS in."""
    months = [str(x) for x in np.asarray(d["months"])]
    moy = np.array([int(m[5:7]) - 1 for m in months], np.int64)
    years = np.array([int(m[:4]) for m in months], np.int64)
    return moy, years


def group_masks(d, raw, version):
    """{group: (moy, t_hold)} for one version, each on the group's OWN rows."""
    from cone_sampler import group_time
    moy, years = master_calendar(d)
    hold = held_out(version, years)
    return {g.name: group_time(g, moy, hold) for g in raw.groups}


# ------------------------------------------------------------------ helpers --
def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha():
    """HEAD of the checkout this runs from, or "unknown" outside a repo."""
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        sha = r.stdout.strip()
        if r.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20)
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:                                         # noqa: BLE001
        return "unknown"


def sha256_file(path, label=None, buf=1 << 24):
    """sha256 of a file, printing progress on anything over 2 GB."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    done, t0, last = 0, time.time(), time.time()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(buf), b""):
            h.update(blk)
            done += len(blk)
            if size > 2e9 and time.time() - last > 30:
                last = time.time()
                print(f"    sha256 {label or os.path.basename(path)}: "
                      f"{done / 1e9:.1f}/{size / 1e9:.1f} GB "
                      f"({done / 1e6 / max(last - t0, 1e-9):.0f} MB/s)",
                      flush=True)
    return h.hexdigest()


def transform_source_sha():
    """sha256 of `anomaly_transform`'s own source text — which version of THE
    function produced these bytes, independent of unrelated edits to the
    file around it."""
    src = open(os.path.join(HERE, "trainprobe.py"), encoding="utf-8").read()
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name == "anomaly_transform":
            seg = ast.get_source_segment(src, n) or ""
            return hashlib.sha256(seg.encode("utf-8")).hexdigest()
    raise ImportError("ml/trainprobe.py no longer defines anomaly_transform")


def finite_or_none(a):
    """A float list with every non-finite value as JSON null (never NaN)."""
    return [float(x) if np.isfinite(x) else None
            for x in np.asarray(a, np.float64).ravel()]


def need_netcdf():
    """Fail at dispatch, not after the first hour: `clim.nc` needs netCDF4."""
    try:
        import netCDF4  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "netCDF4 is not importable — `pip install netCDF4` (or pass "
            "--no-nc to write only clim.npy + stats.json). Checked BEFORE "
            f"any group is copied so no hour is spent first. ({e})")


# ------------------------------------------------------------------- opening --
def open_tensor(tensor, index, groups, skip_sha):
    """Load the tensor, check it IS the tensor the index describes, and
    sha256-verify every group that will be computed. Returns (d, raw, idx,
    shas, verified)."""
    from tensor_io import load_tensor
    from cone_sampler import GroupSet

    idx = json.load(open(index, encoding="utf-8"))
    stem = os.path.basename(tensor)
    stem = stem[:-4] if stem.endswith(".npz") else stem
    if stem != idx["stem"]:
        raise SystemExit(
            f"REFUSING: the tensor's stem is {stem!r} and {index} describes "
            f"{idx['stem']!r}. The channel norms, labels and sha256s would all "
            f"be another tensor's.")
    d = load_tensor(tensor, allow_pickle=False)
    names = [str(g) for g in getattr(d, "groups", [])]
    if not names:
        raise SystemExit(f"{tensor}: no group sidecars — family 7 is a "
                         f"multi-group tensor (<stem>_X_<group>.npy)")
    want = list(groups) if groups else names
    unknown = [g for g in want if g not in names]
    if unknown:
        raise SystemExit(f"--groups {unknown} are not in {tensor} "
                         f"(have {names})")
    missing = [g for g in want if g not in idx["groups"]]
    if missing:
        raise SystemExit(f"{index} has no block for {missing}")

    shas, verified = {}, {}
    for g in want:
        blk = idx["groups"][g]
        path = d.x_paths[g]
        arr = d[f"X_{g}"]
        if list(arr.shape) != list(blk["shape"]):
            raise SystemExit(f"{g}: array shape {list(arr.shape)} but the "
                             f"index says {blk['shape']}")
        chans = [str(c) for c in d[f"chan_{g}"]]
        if chans != list(blk["chans"]):
            raise SystemExit(f"{g}: channel list differs from the index — "
                             f"the physical units would be applied to the "
                             f"wrong channels")
        shas[g] = blk["sha256"]
        if skip_sha:
            print(f"  {g}: sha256 NOT verified (--skip-sha)", flush=True)
            verified[g] = False
            continue
        t0 = time.time()
        print(f"  {g}: verifying sha256 of {os.path.basename(path)} "
              f"({os.path.getsize(path) / 1e9:.2f} GB) …", flush=True)
        got = sha256_file(os.path.realpath(path), g)
        if got != blk["sha256"]:
            raise SystemExit(
                f"REFUSING: {path} has sha256 {got}, the index says "
                f"{blk['sha256']}. The climatology of the wrong bytes looks "
                f"exactly like a climatology — re-pull the file "
                f"(ml/pull_family7_tensor.py) before computing anything.")
        verified[g] = True
        print(f"  {g}: sha256 ✓ ({time.time() - t0:.0f} s)", flush=True)
    raw = GroupSet.from_tensor(d)
    return d, raw, idx, shas, verified, want


# ------------------------------------------------------------------- writers --
def write_npy(path, clim):
    """`stats["clim"]` [12, H, W, C] -> [12, C, H, W] float32, C order."""
    out = np.ascontiguousarray(np.transpose(clim, (0, 3, 1, 2)),
                               dtype=np.float32)
    np.save(path, out)
    return out


def write_nc(path, clim_mchw, g, blk, vblock, attrs):
    """CF NetCDF: dims (month, lat, lon), one variable per channel in
    PHYSICAL units, z * sd + mean from the index's `norm` rows."""
    import netCDF4
    chans = list(blk["chans"])
    norm = np.asarray(blk["norm"], np.float64)
    static = set(attrs.get("static_channels", []))
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("month", 12)
        nc.createDimension("lat", len(g.lats))
        nc.createDimension("lon", len(g.lons))
        m = nc.createVariable("month", "i4", ("month",))
        m[:] = np.arange(1, 13, dtype=np.int32)
        m.long_name = "calendar month (1 = January)"
        m.units = "1"
        la = nc.createVariable("lat", "f8", ("lat",))
        la[:] = g.lats
        la.standard_name, la.long_name = "latitude", "latitude"
        la.units = "degrees_north"
        lo = nc.createVariable("lon", "f8", ("lon",))
        lo[:] = g.lons
        lo.standard_name, lo.long_name = "longitude", "longitude"
        lo.units = "degrees_east"
        H, W = len(g.lats), len(g.lons)
        for c, name in enumerate(chans):
            v = nc.createVariable(name, "f4", ("month", "lat", "lon"),
                                  zlib=True, complevel=4, shuffle=True,
                                  chunksizes=(1, H, W),
                                  fill_value=np.float32(np.nan))
            mean, sd = norm[c]
            phys = (clim_mchw[:, c].astype(np.float64) * sd + mean)
            v[:] = phys.astype(np.float32)
            v.long_name = blk["labels"].get(name, name)
            v.units = blk["units"].get(name, "")
            v.tensor_norm_mean = float(mean)
            v.tensor_norm_sd = float(sd)
            v.comment = (
                "per-calendar-month climatology of this channel over the "
                "version's training bins, in physical units "
                "(z * tensor_norm_sd + tensor_norm_mean); NaN where the "
                "(month, cell) had no training sample")
            if name in static:
                v.comment = (
                    "STATIC channel: anomaly_transform found no temporal "
                    "variance and subtracts nothing on it, so its stored "
                    "climatology is 0.0 in z-units and this variable is "
                    "tensor_norm_mean everywhere — not a climatology")
        nc.Conventions = "CF-1.8"
        nc.title = (f"Model climatology — {attrs.get('tensor_stem', '')} "
                    f"group {attrs.get('group', '')} — {vblock['name']}")
        for k, val in attrs.items():
            if isinstance(val, (list, tuple, dict)):
                val = json.dumps(val, sort_keys=True)
            elif val is None:
                val = "none"
            elif isinstance(val, bool):
                val = int(val)
            setattr(nc, k, val)


# ------------------------------------------------------------------- export ---
def export(tensor, index, out, versions=("all", "dev", "paper"), groups=None,
           chunk=None, scratch=None, copy="disk", skip_sha=False,
           write_netcdf=True):
    """Compute and write every (version, group). Returns {version: {group:
    summary}} with the paths written and the counts the stats carry."""
    for v in versions:
        if v not in VERSION_BY_KEY:
            raise SystemExit(f"unknown version {v!r} — have "
                             f"{[x['key'] for x in VERSIONS]}")
    if write_netcdf:
        need_netcdf()
    from export_cone_sample import _anomaly_transform
    from tensor_io import anomaly_chunk, anomaly_peak_bytes, writable_copy

    t_start = time.time()
    print(f"export_family7_clim: {tensor} · versions {list(versions)} · "
          f"copy {copy}", flush=True)
    d, raw, idx, shas, verified, want = open_tensor(tensor, index, groups,
                                                    skip_sha)
    transform = _anomaly_transform()
    src_sha = transform_source_sha()
    sha_git = git_sha()
    stem = idx["stem"]
    by_name = {g.name: g for g in raw.groups}
    scratch_dir = scratch or os.path.dirname(os.path.abspath(tensor))
    masks = {v: group_masks(d, raw, v) for v in versions}
    summary = {}

    for gname in want:
        g = by_name[gname]
        blk = idx["groups"][gname]
        itemsize = np.dtype(g.X.dtype).itemsize
        ch = int(chunk) if chunk else anomaly_chunk(g.X.shape, itemsize)
        peak = anomaly_peak_bytes(g.X.shape, ch, itemsize) / 1e9
        for vkey in versions:
            vb = version_block(VERSION_BY_KEY[vkey])
            moy, t_hold = masks[vkey][gname]
            n_train = int((~np.asarray(t_hold, bool)).sum())
            print(f"[{vkey}/{gname}] {g.T} rows [{g.T}x{g.H}x{g.W}x{g.C}] · "
                  f"{n_train} training row(s), {g.T - n_train} held out · "
                  f"chunk {ch} (peak ~{peak:.1f} GB)", flush=True)
            if n_train == 0:
                raise SystemExit(
                    f"REFUSING {vkey}/{gname}: no training row under this "
                    f"version — the climatology would be NaN everywhere")
            t0 = time.time()
            dst = None
            if copy == "ram":
                A = np.array(g.X)
            else:
                os.makedirs(scratch_dir, exist_ok=True)
                dst = os.path.join(scratch_dir,
                                   f"{stem}_clim_{vkey}_{gname}.npy")
                print(f"  writable copy -> {dst}", flush=True)
                A = writable_copy(g.X, dst, verbose=g.X.nbytes > 1e9)
            stats = {}
            try:
                A, dyn = transform(A, moy, t_hold, np.zeros(g.W, bool),
                                   chunk=ch, verbose=A.nbytes > 1e9,
                                   stats=stats)
            finally:
                del A
                gc.collect()
                if dst and os.path.exists(dst):
                    os.remove(dst)
            clim = stats["clim"]
            fin = np.isfinite(clim)
            if not fin.any():
                raise SystemExit(f"REFUSING {vkey}/{gname}: the climatology "
                                 f"has no finite value at all")
            months_cov = [int(m) + 1 for m in range(12) if fin[m].any()]
            static = [blk["chans"][c] for c in range(g.C) if c not in dyn]
            print(f"  anomaly_transform done ({time.time() - t0:.0f} s): "
                  f"{len(dyn)}/{g.C} dynamic, finite {fin.mean():.1%} of "
                  f"cells, months with data {months_cov}", flush=True)
            if static:
                print(f"  ::warning:: static channel(s) {static} — the "
                      f"transform subtracts nothing on them and their "
                      f"climatology is 0.0 in z-units", flush=True)

            where = os.path.join(out, vkey, gname)
            os.makedirs(where, exist_ok=True)
            npy = os.path.join(where, "clim.npy")
            mchw = write_npy(npy, clim)
            common = dict(
                version=vkey, version_name=vb["name"],
                version_rule=vb["rule"], holdout_years=vb["holdout_years"],
                holdout_from=vb["holdout_from"], train_span=vb["train_span"],
                tensor_stem=stem, group=gname,
                tensor_sha256=shas[gname],
                tensor_sha256_verified=bool(verified[gname]),
                tensor_sha256_by_group={k: idx["groups"][k]["sha256"]
                                        for k in idx["groups"]},
                git_sha=sha_git, generated_utc=now_utc(),
                anomaly_transform=TRANSFORM_SRC,
                anomaly_transform_src_sha256=src_sha,
                n_train_bins=n_train, n_bins=int(g.T),
                static_channels=static, plan=PLAN_URL)
            nc = None
            if write_netcdf:
                nc = os.path.join(where, "clim.nc")
                write_nc(nc, mchw, g, blk, vb, common)
            st = dict(
                _source="ml/export_family7_clim.py — do not hand-edit",
                version=vb, group=gname, chans=list(blk["chans"]),
                norm=[[float(a), float(b)] for a, b in blk["norm"]],
                mu=finite_or_none(stats["mu"]), sd=finite_or_none(stats["sd"]),
                den=finite_or_none(stats["den"]),
                dynamic=[int(c) for c in stats["dynamic"]],
                static_channels=static,
                n_train_bins=n_train, n_bins=int(g.T),
                months_with_data=months_cov,
                finite_fraction=float(fin.mean()),
                shape=list(mchw.shape), dtype=str(mchw.dtype),
                layout="[month, channel, lat, lon], C order; month 0 = "
                       "January; z-units (physical = z * norm_sd + "
                       "norm_mean)",
                lats=[float(x) for x in (g.lats[0], g.lats[-1])],
                lons=[float(x) for x in (g.lons[0], g.lons[-1])],
                tensor_stem=stem, tensor_sha256=shas[gname],
                tensor_sha256_verified=bool(verified[gname]),
                tensor_sha256_by_group=common["tensor_sha256_by_group"],
                git_sha=sha_git, generated_utc=common["generated_utc"],
                anomaly_transform=TRANSFORM_SRC,
                anomaly_transform_src_sha256=src_sha,
                chunk=ch, copy=copy, plan=PLAN_URL,
                note_nulls="mu/sd/den are null where the pool was empty "
                           "(never NaN in this file)")
            sj = os.path.join(where, "stats.json")
            with open(sj, "w", encoding="utf-8") as fh:
                json.dump(st, fh, indent=1, sort_keys=True, allow_nan=False)
                fh.write("\n")
            print(f"  wrote {npy} ({os.path.getsize(npy) / 1e6:.2f} MB)"
                  + (f", clim.nc ({os.path.getsize(nc) / 1e6:.2f} MB)"
                     if nc else "") + ", stats.json", flush=True)
            summary.setdefault(vkey, {})[gname] = dict(
                npy=npy, nc=nc, stats=sj, n_train_bins=n_train,
                dynamic=list(dyn), t_hold=np.asarray(t_hold, bool),
                moy=np.asarray(moy))
            del clim, stats, mchw
            gc.collect()
    print(f"export_family7_clim: done in {(time.time() - t_start) / 60:.1f} "
          f"min → {out}", flush=True)
    return summary


# ------------------------------------------------------------------ fixture --
def fixture_npz(work, index=FIXTURE_INDEX, src=FIXTURE_SRC):
    """A metadata `.npz` for `data/family7/fixture/`, whose npz is not in git.

    The fixture ships only the decimated group sidecars and its own index, so
    the axes the loader needs are rebuilt FROM THAT INDEX (bins, grids,
    channels, norms, the rg100 live bins, the oc025 offset) and the sidecars
    are symlinked beside it — the real `load_tensor` + `GroupSet` path then
    runs unchanged. Returns the npz path."""
    from export_cone_sample import date_of_bin
    idx = json.load(open(index, encoding="utf-8"))
    stem = idx["stem"]
    bins = np.arange(int(idx["bin_first"]), int(idx["bin_last"]) + 1,
                     dtype=np.int64)
    months = np.array([date_of_bin(b)[:7] for b in bins])
    gr = idx["groups"]

    def axis(g):
        gg = gr[g]["grid"]
        return (gg["lat0"] + gg["step"] * np.arange(gg["ny"]),
                gg["lon0"] + gg["step"] * np.arange(gg["nx"]))

    lats, lons = axis("g025")
    lat1, lon1 = axis("g100")
    meta = dict(bin_index=bins, months=months, lats=lats, lons=lons,
                lat1=lat1, lon1=lon1, epoch=np.array(str(idx["epoch"])),
                pentad_days=np.array(int(idx["pentad_days"])),
                groups=np.array(list(gr)), recipe=np.array(str(idx["recipe"])))
    for g, blk in gr.items():
        meta[f"chan_{g}"] = np.array(blk["chans"])
        meta[f"norm_{g}"] = np.asarray(blk["norm"], np.float32)
    if "rg100" in gr:
        meta["rg_bin_index"] = np.asarray(gr["rg100"]["bin_index"], np.int64)
    if "oc025" in gr:
        b0 = int(gr["oc025"]["bin_first"])
        meta["oc_bin_first"] = np.array(b0, np.int64)
        meta["oc025_bin_index"] = np.arange(b0, b0 + int(gr["oc025"]["n_bins"]),
                                            dtype=np.int64)
    npz = os.path.join(work, f"{stem}.npz")
    np.savez(npz, **meta)
    for g, blk in gr.items():
        os.symlink(os.path.abspath(os.path.join(src, blk["file"])),
                   os.path.join(work, blk["file"]))
    return npz


def run_fixture(out=FIXTURE_OUT, clim_index=FIXTURE_CLIM_INDEX,
                groups=FIXTURE_GROUPS, versions=("all", "dev", "paper"),
                write_netcdf=FIXTURE_NC):
    """`--fixture`: the whole export + the index, on the committed smoke
    tensor, no network. Rewrites `out` from scratch."""
    import publish_family7_clim_index as P
    work = tempfile.mkdtemp(prefix="f7clim_fix_")
    try:
        npz = fixture_npz(work)
        if os.path.isdir(out):
            for v in VERSIONS:
                shutil.rmtree(os.path.join(out, v["key"]), ignore_errors=True)
        export(npz, FIXTURE_INDEX, out, versions=versions, groups=groups,
               copy="ram", skip_sha=False, write_netcdf=write_netcdf)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rel = os.path.relpath(out, ROOT)
    return P.main(["index", "--out", out, "--f7-index", FIXTURE_INDEX,
                   "--index", clim_index, "--no-cors", "--local",
                   "--fixture-dir", rel + "/"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tensor", help="<stem>.npz beside its _X_<group>.npy")
    ap.add_argument("--index", default=os.path.join(ROOT, "data",
                                                    "family7_index.json"))
    ap.add_argument("--out", help="output directory (<version>/<group>/…)")
    ap.add_argument("--versions", default="all,dev,paper")
    ap.add_argument("--groups", default="",
                    help="comma list; default every group in the tensor "
                         "(the fixture: " + ",".join(FIXTURE_GROUPS) + ")")
    ap.add_argument("--chunk", type=int, default=0,
                    help="anomaly_transform chunk in timesteps (default "
                         "tensor_io.anomaly_chunk)")
    ap.add_argument("--scratch", default="",
                    help="where the writable copy goes (default: beside the "
                         "tensor); deleted after each version")
    ap.add_argument("--copy", choices=("disk", "ram"), default="disk",
                    help="ram = np.array(group) instead of a scratch .npy")
    ap.add_argument("--skip-sha", action="store_true",
                    help="do not verify the group sha256s (toy tensors only)")
    ap.add_argument("--no-nc", action="store_true",
                    help="skip clim.nc")
    ap.add_argument("--fixture", action="store_true",
                    help="run on data/family7/fixture/ and write "
                         "data/family7_clim/fixture/ + its index")
    a = ap.parse_args(argv)
    versions = [v.strip() for v in a.versions.split(",") if v.strip()]
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    if a.fixture:
        run_fixture(out=a.out or FIXTURE_OUT,
                    clim_index=(os.path.join(a.out, "family7_clim_index.json")
                                if a.out else FIXTURE_CLIM_INDEX),
                    groups=groups or FIXTURE_GROUPS, versions=versions,
                    write_netcdf=FIXTURE_NC and not a.no_nc)
        return 0
    if not a.tensor or not a.out:
        ap.error("--tensor and --out are required (or --fixture)")
    export(a.tensor, a.index, a.out, versions=versions, groups=groups or None,
           chunk=a.chunk or None, scratch=a.scratch or None, copy=a.copy,
           skip_sha=a.skip_sha, write_netcdf=not a.no_nc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
