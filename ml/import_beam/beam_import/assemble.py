"""Stage B — assemble the Stage-A day records into the training set.

    <output>/*/*.tfrecord  ->  Map to (bin, day record)  ->  GroupByKey(bin)
        ->  per bin: the pentad mean / sigma / live-month rule per channel,
            plus a days_present count per channel
        ->  one Example per (bin, group)  ->  WriteToTFRecord, N shards/group
        ->  spec.json + coverage.json

A **bin** is `floor(day_index / cadence_days)` from 1982-01-01 — the same
clock `aggregate_cadence.bin_index` uses, imported. `--cadence pentad` (the
default) makes `cadence_days = 5`; `--cadence daily` makes it 1, which is the
daily sidecar the multi-rate inner cone of E-070 §7 reads, and there a bin IS
a day and `date_start == date_end`. The three channel groups, their names and their
order are `build_family7`'s `CHAN_G025` / `CHAN_G100` / `CHAN_RG100`,
imported. The regridding is `build_family3`'s `lin_weights` / `interp2_nan`,
imported. The `min_days = 3` rule is `build_family7.MIN_DAYS`, imported, and
it is applied HERE and only here — Stage A keeps every day it got.

Nothing in this file hardcodes a grid: the target axes are derived from the
records themselves with `aggregate_cadence.axis_for`, so the same code runs
on the tiny test fixtures and on the real archive.

Stage A is the DAILY ARCHIVE; Stage B is a derived view of it at a chosen
cadence. Running Stage B twice, at two cadences, is the supported way to get
both — nothing is recomputed upstream and the two views never share a
directory.

    python -m beam_import.assemble --output /data/import --runner DirectRunner
    python -m beam_import.assemble --output /data/import --cadence daily \\
        --runner DirectRunner
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import apache_beam as beam
import numpy as np

from . import tfrecord
from .example import (make_example, one_int, one_str, parse_example, str_list)
from .transforms import _ml_on_path

# The pentad default. `cadence_days` is threaded through everything below;
# this constant is only the default value of that parameter, so the Tier-0
# flow is unchanged when `--cadence` is absent.
PENTAD_DAYS = 5
CADENCES = {"pentad": 5, "daily": 1}
GROUPS = ("g025", "g100", "rg100")

# Which groups a cadence emits unless `--groups` says otherwise. At daily
# cadence rg100 is skipped: Roemmich-Gilson Argo is a MONTHLY product, so a
# daily record of it would be four fifths invented — the same reason the
# pentad path writes it only on the live-month bin.
DEFAULT_GROUPS = {"pentad": ("g025", "g100", "rg100"),
                  "daily": ("g025", "g100")}

# min_days per cadence. At daily cadence a bin holds ONE day, so the pentad
# rule (at least three of the five days) cannot apply: a day is present iff it
# was observed.
DEFAULT_MIN_DAYS = {"daily": 1}          # pentad takes build_family7.MIN_DAYS

# The two stress sigma channels are renamed at daily cadence, because the
# quantity is genuinely different: within-PENTAD sigma over ~20 six-hourly
# samples versus within-DAY sigma over 4. Nobody should be able to mistake one
# for the other by reading a channel name.
STD_CHANNELS = ("tau_x_std", "tau_y_std")
DAILY_STD_SUFFIX = "_day"

# NCEP file stem -> (g100 channel name, how to convert AFTER regridding).
# The conversions are build_family7's, transcribed from its `flush()` and
# checked against it in tests/test_assemble.py.
NCEP_TO_G100 = {
    "uflx": "tau_x", "vflx": "tau_y", "air": "t2m", "uwnd": "u10",
    "vwnd": "v10", "pres": "sp", "prate": "log_prate", "weasd": "log_swe",
    "soilw": "soilw", "tmp": "tsoil", "lhtfl": "lhtfl", "shtfl": "shtfl",
    # build_family7's CHAN_G100 carries a 15th channel: the shared surface
    # temperature, one instrument over land, sea and ice alike.
    "skt": "skt",
}
LAND_ONLY = ("soilw", "tmp")            # masked on the gaussian grid first


def f7():
    _ml_on_path()
    import build_family7 as mod                            # noqa: PLC0415
    return mod


def f3():
    _ml_on_path()
    import build_family3 as mod                            # noqa: PLC0415
    return mod


def agg():
    _ml_on_path()
    import aggregate_cadence as mod                        # noqa: PLC0415
    return mod


# --------------------------------------------------------------------------
# reading a Stage-A record
# --------------------------------------------------------------------------
def record_cube(rec: Dict[str, List[Any]]) -> np.ndarray:
    shape = [int(x) for x in rec.get("shape", [])]
    raw = rec["values"][0]
    return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float64)


def record_axes(rec) -> Tuple[np.ndarray, np.ndarray]:
    return (np.asarray(rec.get("lat_values", []), dtype=np.float64),
            np.asarray(rec.get("lon_values", []), dtype=np.float64))


def to_keyed(payload: bytes, cadence_days: int = PENTAD_DAYS):
    """(bin, a light dict) — the shuffle key is the bin the day falls in."""
    rec = parse_example(payload)
    grid = one_str(rec, "grid")
    if grid in ("opaque", "series"):
        return                                # not part of the pentad tensor
    day_index = one_int(rec, "day_index")
    if day_index < 0:
        return
    b = day_index // cadence_days
    yield b, {
        "source": one_str(rec, "source"),
        "item_id": one_str(rec, "item_id"),
        "date": one_str(rec, "date"),
        "day_index": day_index,
        "grid": grid,
        "var_names": str_list(rec, "var_names"),
        "shape": [int(x) for x in rec.get("shape", [])],
        "values": rec["values"][0],
        "lat": [float(x) for x in rec.get("lat_values", [])],
        "lon": [float(x) for x in rec.get("lon_values", [])],
    }


def _cube(d: Dict[str, Any]) -> np.ndarray:
    return np.frombuffer(d["values"], dtype="<f4").reshape(
        d["shape"]).astype(np.float64)


# --------------------------------------------------------------------------
# the per-bin assembly
# --------------------------------------------------------------------------
class Assemble(beam.DoFn):
    """One pentad bin in, up to three group Examples out."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self) -> None:
        self.f7, self.f3, self.agg = f7(), f3(), agg()
        self.cadence_days = int(self.cfg.get("cadence_days", PENTAD_DAYS))
        self.daily = self.cadence_days == 1
        self.min_days = int(self.cfg.get("min_days") or self.f7.MIN_DAYS)
        self.groups = tuple(self.cfg.get("groups") or GROUPS)
        self.chan = {"g025": list(self.f7.CHAN_G025),
                     "g100": list(self.f7.CHAN_G100),
                     "rg100": list(self.f7.CHAN_RG100)}
        self.std_suffix = DAILY_STD_SUFFIX if self.daily else ""
        if self.daily:
            self.chan["g100"] = [n + DAILY_STD_SUFFIX if n in STD_CHANNELS
                                 else n for n in self.chan["g100"]]
        self.flip = set(self.f7.NCEP_FLIP)
        # DESIGN §4 calls g025's channel 5 `skin_t`; build_family7's own
        # CHAN_G025 calls it `sst`. The IMPORTED table is the truth (the
        # brief says import the tables, do not restate them), so the name is
        # taken from it and DESIGN's name is treated as a synonym.
        self.skin = self.chan["g025"][5]
        # Likewise build_family7's CHAN_G100 carries a 15th channel, the
        # shared surface temperature `skt` at C_SKT = 14, which DESIGN's
        # "g100 14 ch" does not mention. It is filled here when the imported
        # list has it, and simply absent when it does not.
        self.g100_skt = (self.chan["g100"][14]
                         if len(self.chan["g100"]) > 14 else None)

    # -- helpers ----------------------------------------------------------
    def _target_axes(self, days, deg):
        """Axes at `deg` covering what the records actually cover.

        `axis_for` is aggregate_cadence's — the same function that defined
        the 0.25 degree point grid — so the 1 degree grid here is the same
        rule one spacing coarser, not a second definition of "a grid".
        """
        lats = [np.asarray(d["lat"]) for d in days if len(d["lat"]) > 1]
        lons = [np.asarray(d["lon"]) for d in days if len(d["lon"]) > 1]
        if not lats:
            return None, None
        lo_a = min(float(a.min()) for a in lats)
        hi_a = max(float(a.max()) for a in lats)
        lo_o = min(float(a.min()) for a in lons)
        hi_o = max(float(a.max()) for a in lons)
        return (self.agg.axis_for(lo_a, hi_a, deg, "point"),
                self.agg.axis_for(lo_o, hi_o, deg, "point"))

    def _regrid(self, field, src_lat, src_lon, dst_lat, dst_lon):
        wy = self.f3.lin_weights(src_lat, dst_lat)
        wx = self.f3.lin_weights(
            src_lon, np.where(dst_lon < 0, dst_lon + 360.0, dst_lon),
            wrap_period=360.0)
        return self.f3.interp2_nan(field, wy, wx)

    @staticmethod
    def _mean_stack(planes: List[np.ndarray]):
        """NaN-aware mean and per-cell count over a list of same-shape days."""
        stack = np.stack(planes)
        finite = np.isfinite(stack)
        count = finite.sum(axis=0)
        with np.errstate(invalid="ignore"):
            total = np.where(finite, stack, 0.0).sum(axis=0)
            mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        return mean, count

    # -- the DoFn ---------------------------------------------------------
    def process(self, element):
        b, days = element
        days = list(days)
        epoch = self.agg.EPOCH
        d0 = epoch + dt.timedelta(days=int(b) * self.cadence_days)
        d1 = d0 + dt.timedelta(days=self.cadence_days - 1)

        by_grid: Dict[str, List[Dict[str, Any]]] = {}
        for d in days:
            by_grid.setdefault(d["grid"], []).append(d)

        lat025, lon025 = self._target_axes(by_grid.get("point025", []), 0.25)
        if lat025 is None:
            # No 0.25 degree source in this bin: fall back to whatever
            # OISST covers, so a bin with only OISST still produces g025.
            lat025, lon025 = self._target_axes(
                by_grid.get("oisst_center025", []), 0.25)

        g025 = self._build_g025(by_grid, lat025, lon025)
        lat1, lon1 = (None, None)
        if lat025 is not None:
            lat1 = self.agg.axis_for(float(lat025.min()), float(lat025.max()),
                                     1.0, "point")
            lon1 = self.agg.axis_for(float(lon025.min()), float(lon025.max()),
                                     1.0, "point")
        g100 = self._build_g100(by_grid, lat1, lon1)
        rg100 = self._build_rg100(by_grid, lat1, lon1, d0, d1)

        for group, built in (("g025", g025), ("g100", g100),
                             ("rg100", rg100)):
            if built is None or group not in self.groups:
                continue
            cube, days_present, lat, lon, srcs = built
            cube = np.asarray(cube, dtype=np.float32)
            # min_days is applied HERE and only here (DESIGN §2).
            for c in range(cube.shape[0]):
                if days_present[c] < self.min_days:
                    cube[c, :, :] = np.nan
            feat = {
                "bin": int(b), "date_start": d0.isoformat(),
                "date_end": d1.isoformat(), "group": group,
                "cadence": "daily" if self.daily else "pentad",
                "cadence_days": int(self.cadence_days),
                "chan_names": self.chan[group],
                "values": np.ascontiguousarray(cube).tobytes(),
                "mask": np.packbits(np.isfinite(cube).ravel()).tobytes(),
                "shape": list(cube.shape),
                "lat0": float(lat[0]), "lat_step": float(lat[1] - lat[0])
                if len(lat) > 1 else 0.0, "nlat": int(len(lat)),
                "lon0": float(lon[0]), "lon_step": float(lon[1] - lon[0])
                if len(lon) > 1 else 0.0, "nlon": int(len(lon)),
                "lat_values": [float(x) for x in lat],
                "lon_values": [float(x) for x in lon],
                "days_present": [int(x) for x in days_present],
                "sources": sorted(set(srcs)),
            }
            yield group, make_example(feat)

    # -- g025 -------------------------------------------------------------
    def _build_g025(self, by_grid, lat, lon):
        """cur_speed, log_mld, ssh, cur_u, cur_v, skin_t, sea_ice.

        `cur_speed` is the hypot of the pentad-MEAN u and v (the mean first,
        then the magnitude — the other order is a different quantity), and
        `log_mld` is log10 of the binned mean mixed-layer depth, matching
        family 4. `skin_t` is OISST's sst where OISST has one and NCEP's skt
        elsewhere, per DESIGN §4.
        """
        if lat is None:
            return None
        names = self.chan["g025"]
        cube = np.full((len(names), len(lat), len(lon)), np.nan)
        present = [0] * len(names)
        srcs: List[str] = []

        ocean = by_grid.get("point025", [])
        per_var: Dict[str, List[np.ndarray]] = {}
        for d in ocean:
            arr = _cube(d)
            for ci, v in enumerate(d["var_names"]):
                per_var.setdefault(v, []).append(
                    self._to_axes(arr[ci], d, lat, lon))
            srcs.append(d["item_id"])

        def put(name, field, ndays):
            i = names.index(name)
            cube[i] = field
            present[i] = ndays

        means = {}
        for v, planes in per_var.items():
            mean, _cnt = self._mean_stack(planes)
            means[v] = (mean, len(planes))
        if "uo" in means and "vo" in means:
            u, nu = means["uo"]
            v, nv = means["vo"]
            put("cur_u", u, nu)
            put("cur_v", v, nv)
            put("cur_speed", np.hypot(u, v), min(nu, nv))
        if "mlotst" in means:
            ml, n = means["mlotst"]
            with np.errstate(invalid="ignore", divide="ignore"):
                put("log_mld", np.where(ml > 0,
                                        np.log10(np.maximum(ml, 1e-6)),
                                        np.nan), n)
        if "zos" in means:
            put("ssh", means["zos"][0], means["zos"][1])

        # OISST -> our point grid, with f3's NaN-aware bilinear.
        sst_planes, ice_planes = [], []
        for d in by_grid.get("oisst_center025", []):
            arr = _cube(d)
            slat, slon = np.asarray(d["lat"]), np.asarray(d["lon"])
            for ci, v in enumerate(d["var_names"]):
                grid = self._regrid(arr[ci], slat, slon, lat, lon)
                (sst_planes if v == "sst" else ice_planes).append(grid)
            srcs.append(d["item_id"])
        skin = None
        if sst_planes:
            skin, _ = self._mean_stack(sst_planes)
            put(self.skin, skin, len(sst_planes))
        if ice_planes:
            ice, _ = self._mean_stack(ice_planes)
            put("sea_ice", np.clip(ice, 0.0, 1.0), len(ice_planes))

        # NCEP skt fills skin_t wherever OISST has nothing (DESIGN §4).
        skt_days = [d for d in by_grid.get("ncep_t62", [])
                    if any(n.startswith("skt") for n in d["var_names"])]
        if skt_days:
            planes = []
            for d in skt_days:
                arr = _cube(d)
                planes.append(self._regrid(arr[0], np.asarray(d["lat"]),
                                           np.asarray(d["lon"]), lat, lon))
                srcs.append(d["item_id"])
            fill, _ = self._mean_stack(planes)
            fill = fill - 273.15                  # K -> degC, as build_family7
            i = names.index(self.skin)
            if skin is None:
                cube[i] = fill
                present[i] = len(planes)
            else:
                cube[i] = np.where(np.isfinite(skin), skin, fill)
                present[i] = max(present[i], len(planes))
        return cube, present, lat, lon, srcs

    def _to_axes(self, field, d, lat, lon):
        """A 0.25 degree plane already on our axes, or regridded onto them."""
        slat, slon = np.asarray(d["lat"]), np.asarray(d["lon"])
        if (len(slat) == len(lat) and len(slon) == len(lon)
                and np.allclose(slat, lat) and np.allclose(slon, lon)):
            return field
        return self._regrid(field, slat, slon, lat, lon)

    # -- g100 -------------------------------------------------------------
    def _build_g100(self, by_grid, lat, lon):
        """The 14 NCEP channels at 1 degree.

        Order of operations follows `build_family7.flush()` exactly: the
        pentad mean is taken on the GAUSSIAN grid, the land mask for
        soilw/tsoil is applied there too (so `interp2_nan` renormalises at
        the coast), then the field is regridded, and only then are the units
        converted. The sign flip on uflx/vflx is linear, so applying it once
        to the pentad mean is the same number `build_family7` gets by
        applying it to every 6-hourly sample.
        """
        if lat is None:
            return None
        days = by_grid.get("ncep_t62", [])
        if not days:
            return None
        names = self.chan["g100"]
        cube = np.full((len(names), len(lat), len(lon)), np.nan)
        present = [0] * len(names)
        srcs: List[str] = []

        per_var: Dict[str, List[np.ndarray]] = {}
        axes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        land = None
        for d in days:
            arr = _cube(d)
            slat, slon = np.asarray(d["lat"]), np.asarray(d["lon"])
            srcs.append(d["item_id"])
            for ci, v in enumerate(d["var_names"]):
                stem = v.split(".")[0]
                if stem == "land":
                    land = arr[ci] >= 0.5
                    continue
                per_var.setdefault(stem, []).append(arr[ci])
                axes[stem] = (slat, slon)

        def to1(field, stem):
            slat, slon = axes[stem]
            return self._regrid(field, slat, slon, lat, lon)

        def put(name, field, ndays):
            if name not in names:
                return
            i = names.index(name)
            cube[i] = field
            present[i] = ndays

        for stem, planes in per_var.items():
            if stem.endswith("_sq"):
                continue
            base = stem
            mean, _cnt = self._mean_stack(planes)
            n = len(planes)
            if base in self.flip:
                mean = -mean
            if base in LAND_ONLY and land is not None:
                mean = np.where(land, mean, np.nan)
            chan = NCEP_TO_G100.get(base)
            if chan is None:
                continue
            g = to1(mean, stem)
            if chan in ("t2m", "tsoil", "skt"):
                g = g - 273.15
            elif chan == "sp":
                g = g / 100.0                       # Pa -> hPa
            elif chan == "log_prate":
                g = np.log1p(np.maximum(g * 86400.0, 0.0))
            elif chan == "log_swe":
                g = np.log1p(np.maximum(g, 0.0))
            put(chan, g, n)

            # tau_x_std / tau_y_std: the POPULATION sigma over the 6-hourly
            # samples IN THIS BIN, recovered exactly from the daily mean and
            # the daily mean of squares (see transforms.ncep_var_year).
            #
            # At pentad cadence that is the within-pentad sigma, which is what
            # family 4 and family 7 mean by `tau_*_std`. At DAILY cadence the
            # same formula over one day's four samples is the within-DAY
            # sigma — a different, well-defined quantity — so the channel is
            # named `tau_x_std_day` / `tau_y_std_day` there and nobody can
            # confuse the two.
            #
            # E-070 §7 wants a CENTRED 5-day sigma for the daily-rate family.
            # That is a rolling window over these daily records and belongs on
            # the trainer's side; Stage B does not bake in a window it would
            # then have to be trusted about.
            sq = per_var.get(f"{stem}_sq")
            if sq is not None and base in self.flip:
                e_x2, _ = self._mean_stack(sq)
                e_x = -mean if base in self.flip else mean
                with np.errstate(invalid="ignore"):
                    var = np.maximum(e_x2 - e_x ** 2, 0.0)
                axis = "x" if base == "uflx" else "y"
                put(f"tau_{axis}_std{self.std_suffix}",
                    to1(np.sqrt(var), stem), n)
        return cube, present, lat, lon, srcs

    # -- rg100 ------------------------------------------------------------
    def _build_rg100(self, by_grid, lat, lon, d0, d1):
        """Roemmich-Gilson temperature and salinity, on the LIVE bin only.

        A monthly product has one value per month, so it is written on the
        single bin that contains that month's 15th and left missing on the
        others; spreading it over the month would invent most of the data.
        At daily cadence that means one bin per month — which is why rg100 is
        not in the daily default set at all, and is emitted only if somebody
        asks for it explicitly with `--groups`.
        """
        if lat is None:
            return None
        days = [d for d in by_grid.get("rg_1deg_center", [])
                if d0 <= dt.date.fromisoformat(d["date"]) <= d1]
        if not days:
            return None
        names = self.chan["rg100"]
        cube = np.full((len(names), len(lat), len(lon)), np.nan)
        present = [0] * len(names)
        srcs = []
        for d in days:
            arr = _cube(d)
            slat, slon = np.asarray(d["lat"]), np.asarray(d["lon"])
            srcs.append(d["item_id"])
            for ci, v in enumerate(d["var_names"]):
                if v not in names:
                    continue
                i = names.index(v)
                cube[i] = self._regrid(arr[ci], slat, slon, lat, lon)
                present[i] = self.min_days      # a live month counts as full
        return cube, present, lat, lon, srcs


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def run(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage B — assemble the Stage-A day records at a cadence")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--output", required=True, help="the Stage-A output root")
    ap.add_argument("--cadence", default="pentad", choices=sorted(CADENCES),
                    help="pentad (5 days, the default and the Tier-0 flow) "
                         "or daily (1 day, the multi-rate cone's sidecar)")
    ap.add_argument("--pentad-out", default=None,
                    help="where the shards go (default <output>/<cadence>)")
    ap.add_argument("--min-days", type=int, default=None,
                    help="override the per-cadence default (pentad 3, daily 1)")
    ap.add_argument("--groups", default=None,
                    help="comma-separated subset of g025,g100,rg100 "
                         "(default: all at pentad, g025,g100 at daily)")
    ap.add_argument("--num-shards", type=int, default=None)
    args, beam_argv = ap.parse_known_args(argv)

    from . import registry as reg_mod
    reg = reg_mod.load(args.registry)
    cadence = args.cadence
    cadence_days = CADENCES[cadence]
    groups = tuple(g.strip() for g in args.groups.split(",")
                   if g.strip()) if args.groups else DEFAULT_GROUPS[cadence]
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        print(f"unknown group(s) {bad}; known: {list(GROUPS)}", file=sys.stderr)
        return 2
    min_days = args.min_days
    if min_days is None:
        min_days = DEFAULT_MIN_DAYS.get(cadence)      # None -> f7.MIN_DAYS

    # Stage-A shards only: never re-read a cadence's own output.
    shards = tfrecord.list_uris(args.output, ".tfrecord")
    shards = [s for s in shards
              if not any(f"/{c}/" in s for c in CADENCES)]
    if not shards:
        print(f"no Stage-A shards under {args.output}", file=sys.stderr)
        return 1
    out_root = args.pentad_out or tfrecord.join(args.output, cadence)
    num_shards = args.num_shards or reg.num_shards_per_group()
    print(f"stage B: {len(shards)} shard(s) -> {out_root} "
          f"(cadence {cadence}, {cadence_days} day(s) per bin, "
          f"groups {','.join(groups)}, {num_shards} shards per group)")

    from apache_beam.options.pipeline_options import (PipelineOptions,
                                                      SetupOptions)
    options = PipelineOptions(beam_argv)
    options.view_as(SetupOptions).save_main_session = True

    cfg = {"cadence_days": cadence_days, "min_days": min_days,
           "groups": list(groups)}
    with beam.Pipeline(options=options) as p:
        keyed = (p | "Shards" >> beam.Create(shards)
                 | "Read" >> beam.io.ReadAllFromTFRecord()
                 | "Key" >> beam.FlatMap(to_keyed, cadence_days=cadence_days)
                 | "Group" >> beam.GroupByKey()
                 | "Assemble" >> beam.ParDo(Assemble(cfg)))
        for group in groups:
            (keyed
             | f"Pick{group}" >> beam.Filter(lambda kv, g=group: kv[0] == g)
             | f"Drop{group}" >> beam.Map(lambda kv: kv[1])
             | f"Write{group}" >> beam.io.WriteToTFRecord(
                 tfrecord.join(out_root, group, "part"),
                 file_name_suffix=".tfrecord", num_shards=num_shards))

    write_spec_and_coverage(out_root, reg, cadence=cadence,
                            cadence_days=cadence_days, groups=groups,
                            min_days=min_days)
    return 0


def write_spec_and_coverage(pentad_out: str, reg, cadence: str = "pentad",
                            cadence_days: int = PENTAD_DAYS,
                            groups: Sequence[str] = GROUPS,
                            min_days: Optional[int] = None) -> Dict[str, Any]:
    """spec.json (what a trainer needs to z-score) and coverage.json.

    Both are written UNDER THE CADENCE's own directory, so a daily run can
    never overwrite a pentad run's spec. The per-channel mean/sd are computed
    over what was actually written and kept OUT of the records, because which
    years are "train" is a trainer decision (DESIGN §4).
    """
    mod = f7()
    eff_min_days = int(min_days if min_days is not None else mod.MIN_DAYS)
    daily = cadence_days == 1
    spec: Dict[str, Any] = {
        "epoch": "1982-01-01", "cadence": cadence,
        "cadence_days": int(cadence_days),
        "pentad_days": PENTAD_DAYS, "min_days": eff_min_days,
        "groups": {}, "written_at": sinks_utcnow(),
        "notes": _spec_notes(daily, eff_min_days),
    }
    coverage: Dict[str, Any] = {"cadence": cadence,
                                "cadence_days": int(cadence_days),
                                "min_days": eff_min_days,
                                "groups": {}, "notes": {}}
    for group in groups:
        names = {"g025": list(mod.CHAN_G025), "g100": list(mod.CHAN_G100),
                 "rg100": list(mod.CHAN_RG100)}[group]
        if daily and group == "g100":
            names = [n + DAILY_STD_SUFFIX if n in STD_CHANNELS else n
                     for n in names]
        if daily and group == "rg100":
            coverage["notes"]["rg100"] = (
                "Roemmich-Gilson Argo is a MONTHLY product. At daily cadence "
                "it is emitted ONLY on the day containing each month's 15th; "
                "every other day is deliberately absent. It is not in the "
                "daily default group set for exactly this reason.")
        uris = tfrecord.list_uris(tfrecord.join(pentad_out, group),
                                  ".tfrecord")
        bins, shape, sums, sqs, cnts = [], None, None, None, None
        for uri in uris:
            for payload in tfrecord.read_records(uri):
                rec = parse_example(payload)
                bins.append(one_int(rec, "bin"))
                shp = [int(x) for x in rec["shape"]]
                cube = np.frombuffer(rec["values"][0],
                                     dtype="<f4").reshape(shp).astype(
                                         np.float64)
                shape = shp
                if sums is None:
                    sums = np.zeros(shp[0])
                    sqs = np.zeros(shp[0])
                    cnts = np.zeros(shp[0])
                fin = np.isfinite(cube)
                sums += np.where(fin, cube, 0.0).sum(axis=(1, 2))
                sqs += np.where(fin, cube, 0.0).__pow__(2).sum(axis=(1, 2))
                cnts += fin.sum(axis=(1, 2))
        stats = []
        if cnts is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
                var = np.where(cnts > 0, sqs / np.maximum(cnts, 1) - mean ** 2,
                               np.nan)
            stats = [{"channel": n, "mean": _f(mean[i]),
                      "sd": _f(math.sqrt(max(float(var[i]), 0.0))
                               if np.isfinite(var[i]) else float("nan")),
                      "finite_cells": int(cnts[i])}
                     for i, n in enumerate(names[:len(cnts)])]
        spec["groups"][group] = {"channels": names, "shape": shape,
                                 "shards": len(uris), "norm": stats}
        bins_sorted = sorted(set(bins))
        coverage["groups"][group] = {
            "bins_present": len(bins_sorted),
            "bin_min": bins_sorted[0] if bins_sorted else None,
            "bin_max": bins_sorted[-1] if bins_sorted else None,
            "bins_missing_in_range": (
                (bins_sorted[-1] - bins_sorted[0] + 1 - len(bins_sorted))
                if bins_sorted else 0),
        }
    tfrecord.write_bytes(tfrecord.join(pentad_out, "spec.json"),
                         json.dumps(spec, indent=1, sort_keys=True,
                                    default=str).encode("utf-8"))
    tfrecord.write_bytes(tfrecord.join(pentad_out, "coverage.json"),
                         json.dumps(coverage, indent=1,
                                    sort_keys=True).encode("utf-8"))
    for group, cov in coverage["groups"].items():
        print(f"coverage {cadence} {group}: "
              f"bins_present={cov['bins_present']} "
              f"range=[{cov['bin_min']}, {cov['bin_max']}] "
              f"missing_in_range={cov['bins_missing_in_range']}")
    return coverage


def _spec_notes(daily: bool, min_days: int) -> Dict[str, str]:
    """The two things a reader of these shards has to be told."""
    notes = {
        "bin": "bin = floor(day_index / cadence_days) from 1982-01-01",
        "min_days": (f"a channel with fewer than {min_days} contributing "
                     "day(s) is written all-NaN; days_present carries the "
                     "true count either way"),
        "norm": ("mean/sd over what was written, kept OUT of the records: "
                 "which years are `train` is a trainer decision"),
    }
    if daily:
        notes["tau_x_std_day"] = notes["tau_y_std_day"] = (
            "WITHIN-DAY population sigma over the four 6-hourly NCEP samples "
            "of this day — NOT the within-pentad sigma the pentad shards call "
            "`tau_x_std`. E-070 §7's centred 5-day sigma is a rolling window "
            "over these daily records and belongs on the trainer's side.")
    return notes


def _f(x) -> Optional[float]:
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


def sinks_utcnow() -> str:
    from .sinks import utcnow
    return utcnow()


if __name__ == "__main__":
    raise SystemExit(run())
