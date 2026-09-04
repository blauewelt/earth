#!/usr/bin/env python3
"""Generate the offline test fixtures and the test registry.

Everything the smoke test and the unit tests read is GENERATED here, not
committed as binary blobs: a NetCDF file in git is a thing nobody can review.

    python tests/fixtures/make_fixtures.py [outdir]

Every fixture covers the SAME pentad — 2003-07-15 … 2003-07-19, which is bin
1573 counting from 1982-01-01 — so Stage B has all three channel groups in one
bin and the smoke test can assert a real coverage line. The 15th is in that
window on purpose: Roemmich-Gilson is monthly and lands on the bin holding its
month's 15th.

Written under `outdir` (default tests/fixtures/generated):

    tiny/hello.dat                     an ordinary small file (opaque)
    ocean/ocean_200307.nc              5 days on a small 0.25° point grid
                                       (uo, vo, mlotst, zos) — a GLORYS stand-in
    oisst/oisst-...{YYYYMMDD}.nc       4 of the 5 days; 2003-07-17 is DELIBERATELY
                                       absent, to exercise the missing-day queue
    ncep/{uflx,skt}.2003.nc            4x-daily on a T62-ish gaussian grid
    ncep/land.sfc.gauss.nc             the land/sea mask
    rg/RG_ArgoClim_200307.nc           one month of T/S at the 16 AMOC levels
    sources_test.yaml                  a registry whose URLs are file:// URLs
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

DAYS = ["2003-07-15", "2003-07-16", "2003-07-17", "2003-07-18", "2003-07-19"]
MISSING_DAY = "2003-07-17"          # never written: the queue must chase it
BIN = 1573                          # (2003-07-15 - 1982-01-01).days // 5

EPOCH1970 = dt.date(1970, 1, 1)


def _days_since(day: str) -> float:
    y, m, d = (int(x) for x in day.split("-"))
    return float((dt.date(y, m, d) - EPOCH1970).days)


def _nc(path):
    import netCDF4
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return netCDF4.Dataset(path, "w", format="NETCDF4")


def _time(ds, days):
    t = ds.createVariable("time", "f8", ("time",))
    t.units = "days since 1970-01-01"
    t[:] = [_days_since(d) for d in days]
    return t


def write_ocean(path: str) -> None:
    """A GLORYS stand-in: already on the 0.25 degree point grid, 5 days."""
    rng = np.random.default_rng(3)
    lat = np.arange(-1.0, 1.001, 0.25)
    lon = np.arange(-1.0, 1.001, 0.25)
    with _nc(path) as ds:
        ds.createDimension("time", len(DAYS))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        _time(ds, DAYS)
        for name, scale, offset in (("uo", 0.3, 0.0), ("vo", 0.3, 0.0),
                                    ("mlotst", 20.0, 60.0), ("zos", 0.1, 0.0)):
            v = ds.createVariable(name, "f4", ("time", "lat", "lon"))
            v.units = {"mlotst": "m", "zos": "m"}.get(name, "m s-1")
            v[:] = (rng.normal(offset, scale,
                               (len(DAYS), len(lat), len(lon)))
                    .astype("f4"))


def write_oisst_day(path: str, day: str, seed: int) -> None:
    """OISST's real shape: (time, zlev, lat, lon), CELL-CENTRE axes, and the
    `ice` trap kept intact — units say percent, valid_range is 0..1."""
    rng = np.random.default_rng(seed)
    lat = np.arange(-0.875, 1.0, 0.25)
    lon = np.arange(-0.875, 1.0, 0.25)
    with _nc(path) as ds:
        ds.createDimension("time", 1)
        ds.createDimension("zlev", 1)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f4", ("lat",))[:] = lat
        ds.createVariable("lon", "f4", ("lon",))[:] = lon
        ds.createVariable("zlev", "f4", ("zlev",))[:] = [0.0]
        _time(ds, [day])
        sst = ds.createVariable("sst", "f4", ("time", "zlev", "lat", "lon"))
        sst.units = "Celsius"
        sst[0, 0] = rng.normal(20.0, 2.0, (len(lat), len(lon))).astype("f4")
        ice = ds.createVariable("ice", "f4", ("time", "zlev", "lat", "lon"))
        ice.units = "percent"                     # the trap, faithfully
        ice.valid_range = np.array([0.0, 1.0], dtype="f4")
        ice[0, 0] = rng.uniform(0.0, 1.0, (len(lat), len(lon))).astype("f4")


def write_ncep(path: str, var: str, seed: int, units: str,
               offset: float) -> None:
    """4x-daily on a descending gaussian-ish latitude, longitudes 0..360."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(2.0, -2.0, 9)               # DESCENDING, like the real one
    lon = np.arange(0.0, 360.0, 30.0)
    steps = [f"{d}" for d in DAYS for _ in range(4)]
    with _nc(path) as ds:
        ds.createDimension("time", len(steps))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f4", ("lat",))[:] = lat
        ds.createVariable("lon", "f4", ("lon",))[:] = lon
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 1970-01-01"
        t.delta_t = "0000-00-00 06:00:00"
        t[:] = [_days_since(d) + 0.25 * (k % 4) for k, d in enumerate(steps)]
        v = ds.createVariable(var, "f4", ("time", "lat", "lon"))
        v.units = units
        v[:] = rng.normal(offset, 1.0,
                          (len(steps), len(lat), len(lon))).astype("f4")
        ds.title = "4x daily NMC reanalysis (synthetic fixture)"


def write_ncep_land(path: str) -> None:
    lat = np.linspace(2.0, -2.0, 9)
    lon = np.arange(0.0, 360.0, 30.0)
    land = np.zeros((1, len(lat), len(lon)), dtype="f4")
    land[0, :, :4] = 1.0                          # a slab of land
    with _nc(path) as ds:
        ds.createDimension("time", 1)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f4", ("lat",))[:] = lat
        ds.createVariable("lon", "f4", ("lon",))[:] = lon
        _time(ds, [DAYS[0]])
        v = ds.createVariable("land", "f4", ("time", "lat", "lon"))
        v.units = "1"
        v[:] = land


def write_rg(path: str, levels) -> None:
    """One month of Roemmich-Gilson: MEAN + ANOMALY at the 16 AMOC levels."""
    rng = np.random.default_rng(9)
    lat = np.arange(-1.5, 2.0, 1.0)
    lon = np.arange(-1.5, 2.0, 1.0)
    nl = len(levels)
    with _nc(path) as ds:
        ds.createDimension("time", 1)
        ds.createDimension("pressure", nl)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("pressure", "f8", ("pressure",))[:] = levels
        _time(ds, ["2003-07-15"])
        for kind, base in (("TEMPERATURE", 10.0), ("SALINITY", 35.0)):
            m = ds.createVariable(f"ARGO_{kind}_MEAN", "f4",
                                  ("pressure", "lat", "lon"))
            m[:] = (base + rng.normal(0, 0.5,
                                      (nl, len(lat), len(lon)))).astype("f4")
            a = ds.createVariable(f"ARGO_{kind}_ANOMALY", "f4",
                                  ("time", "pressure", "lat", "lon"))
            a[:] = rng.normal(0, 0.1,
                              (1, nl, len(lat), len(lon))).astype("f4")


REGISTRY_TEMPLATE = """# GENERATED by tests/fixtures/make_fixtures.py — do not edit by hand.
# A registry whose "upstream" is a set of local file:// URLs, so the whole
# pipeline can be exercised with no network at all.
version: 1

output:
  layout: "<output>/<item_id>.tfrecord  +  <item_id>.done"
  num_shards_per_group: 2

hosts:
  testhost:
    max_lanes: 2
    min_gap_s: 0
    backoff_ladder_s: [0, 0, 0, 0]
    serves: the fixture files
    evidence: local
  testncei:
    max_lanes: 2
    min_gap_s: 0
    backoff_ladder_s: [0, 0, 0, 0]
    serves: the synthetic OISST days
    evidence: local
  testpsl:
    max_lanes: 1
    min_gap_s: 0
    backoff_ladder_s: [0, 0, 0, 0]
    serves: the synthetic NCEP files
    evidence: local
  testflaky:
    max_lanes: 1
    min_gap_s: 0
    backoff_ladder_s: [0, 0, 0, 0]
    serves: nothing — it always fails, to exercise the circuit breaker
    evidence: local

sources:
  - name: tiny
    tier: 0
    host: testhost
    mode: fetch
    fetcher: http
    chunk: file
    files: [hello.dat]
    url: "file://{root}/tiny/{{file}}"
    filename: "{{file}}"
    transform: opaque
    grid: opaque
    expected_items: 1
    bytes_wire: 32
    bytes_stored: 32
    notes: ["an ordinary small file over the http fetcher's file:// path"]

  - name: ocean
    tier: 0
    host: testhost
    mode: fetch
    fetcher: http
    chunk: file
    files: [ocean_200307.nc]
    url: "file://{root}/ocean/{{file}}"
    filename: "{{file}}"
    variables: [uo, vo, mlotst, zos]
    transform: nc025_days
    grid: point025
    expected_items: 1
    bytes_wire: 20000
    bytes_stored: 20000
    notes: ["a GLORYS stand-in already on the 0.25 degree point grid"]

  - name: oisst
    tier: 0
    host: testncei
    mode: fetch
    fetcher: ncei_oisst_days
    chunk: year
    years: [2003, 2003]
    days: {days}
    expected_items: 1
    url: "file://{root}/oisst/oisst-avhrr-v02r01.{{ymd}}.nc"
    filename: oisst_daily_{{year}}.nc
    transform: oisst_days
    grid: oisst_center025
    bytes_wire: 30000
    bytes_stored: 30000
    notes:
      - "FIVE days are asked for and FOUR exist: {missing} is deliberately
         absent upstream. The item must still be written with the four days it
         got, and {missing} must appear in the retry queue as a day-level item."

  - name: ncep
    tier: 0
    host: testpsl
    mode: fetch
    fetcher: http
    chunk: var_year
    vars: [uflx, skt]
    years: [2003, 2003]
    extra_files: [land.sfc.gauss.nc]
    expected_items: 3
    url: "file://{root}/ncep/{{file}}"
    filename: "{{var}}.{{year}}.nc"
    transform: ncep_var_year
    grid: ncep_t62
    bytes_wire: 20000
    bytes_stored: 20000
    notes: ["4x-daily gaussian files plus the land mask"]

  - name: rg
    tier: 0
    host: testhost
    mode: fetch
    fetcher: http
    chunk: file
    files: [RG_ArgoClim_200307.nc]
    url: "file://{root}/rg/{{file}}"
    filename: "{{file}}"
    transform: rg_months
    grid: rg_1deg_center
    expected_items: 1
    bytes_wire: 20000
    bytes_stored: 20000
    notes: ["one month of T/S at the 16 AMOC pressure levels"]

  - name: flaky
    tier: 0
    host: testflaky
    mode: fetch
    fetcher: transient_test
    chunk: file
    files: [a, b, c, d, e, f, g]
    url: "file://{root}/nowhere/{{file}}"
    filename: "{{file}}"
    transform: opaque
    grid: opaque
    expected_items: 7
    bytes_wire: 1
    bytes_stored: 1
    notes:
      - "ALWAYS fails, transiently. Five consecutive failures trip the lane's
         circuit breaker; every remaining item must reach the RETRY QUEUE and
         the pipeline must still finish. Nothing is ever dropped."
"""


def main(outdir: str) -> int:
    root = os.path.abspath(outdir)
    # Start clean: a stale fixture from an earlier revision is a test that
    # passes for the wrong reason.
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)

    os.makedirs(os.path.join(root, "tiny"), exist_ok=True)
    with open(os.path.join(root, "tiny", "hello.dat"), "w",
              encoding="utf-8") as fh:
        fh.write("beam_import fixture: an ordinary file\n")

    write_ocean(os.path.join(root, "ocean", "ocean_200307.nc"))

    for i, day in enumerate(DAYS):
        if day == MISSING_DAY:
            continue                       # the gap the retry queue must chase
        write_oisst_day(
            os.path.join(root, "oisst",
                         f"oisst-avhrr-v02r01.{day.replace('-', '')}.nc"),
            day, seed=100 + i)

    write_ncep(os.path.join(root, "ncep", "uflx.2003.nc"), "uflx", 11,
               "N/m^2", 0.05)
    write_ncep(os.path.join(root, "ncep", "skt.2003.nc"), "skt", 12,
               "degK", 292.0)
    write_ncep_land(os.path.join(root, "ncep", "land.sfc.gauss.nc"))

    levels = _levels()
    write_rg(os.path.join(root, "rg", "RG_ArgoClim_200307.nc"), levels)

    with open(os.path.join(root, "sources_test.yaml"), "w",
              encoding="utf-8") as fh:
        fh.write(REGISTRY_TEMPLATE.format(root=root, days=str(DAYS),
                                          missing=MISSING_DAY))

    print(f"fixtures written to {root} (pentad bin {BIN}, "
          f"{DAYS[0]}..{DAYS[-1]}, {MISSING_DAY} deliberately absent)")
    return 0


def _levels():
    """The 16 AMOC pressure levels — imported from build_family3, not typed."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    if root not in sys.path:
        sys.path.insert(0, root)
    from beam_import.transforms import _ml_on_path
    _ml_on_path()
    import build_family3 as f3
    return list(f3.LEVELS)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else os.path.join(here, "generated")))
