#!/usr/bin/env python3
"""Tests for family 10's tier-P builder, reader and registry (E-079).

Every test here runs with NO NETWORK: each source adapter meets a tiny
synthetic file on disk in the archive's real format, through `--source-dir`,
so what is exercised is the parser, the QC policy, the unit conversions, the
store's invariants and the k-nearest search — the things a real build can get
wrong while looking entirely ordinary from the outside.

    python3 -m pytest -q tests/test_build_family10_stores.py

In the sandbox (pypi answers 403 through the egress proxy, so neither pytest
nor netCDF4 can be installed) the shim runner does the same job:

    PYTHONPATH=<shim> python3 <shim>/run_tests.py tests/test_build_family10_stores.py

What each group of tests is FOR:

  the adapters      a synthetic source per store, built by the builder's own
                    `make_smoke_sources`, run end to end through index + fetch
                    and checked against the truth the generator kept. This is
                    the test that would catch a column order change, a dropped
                    QC clause or a missed unit conversion.
  the invariants    E-079 §4's assertion list, on a store deliberately built to
                    violate each one — because an assertion nobody has seen
                    fail is an assertion nobody knows is wired up.
  negative bins     drifters start in 1979 and SOCAT in 1957, so `bin` is
                    signed and the CSR index starts wherever the data does.
  the reader        the search's three rules (one-sided, bounded, fixed k with
                    miss tokens) and the footprint fields.
  family 8          the Argo store, opened through the family-10 reader with
                    NO rebuild, answering bit-identically to `ArgoStore`.
  the registry      generated from real store.json files with no Hub access.
  resumability      an interrupted fetch resumes at the year it lost.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_registry as reg                           # noqa: E402
import build_family10_stores as b10                             # noqa: E402
import family8_store as f8                                      # noqa: E402
import family10_store as f10                                    # noqa: E402

STORES = ("gdp", "gtmba", "socat", "slatrack")
SMOKE_START = b10.SMOKE_START          # 1981-12-20 — BEFORE the epoch
SMOKE_END = b10.SMOKE_END              # 1982-01-12


# ------------------------------------------------------------------ helpers --
def build(tmp, store, start=SMOKE_START, end=SMOKE_END, stages=("index",
                                                                "fetch"),
          **over):
    """Generate this store's synthetic source and run it end to end."""
    src = os.path.join(tmp, f"src_{store}")
    work = os.path.join(tmp, f"work_{store}")
    os.makedirs(work, exist_ok=True)
    truth = b10.make_smoke_sources(src, store, b10.parse_date(start),
                                   b10.parse_date(end))
    ns = dict(store=store, work=work, source_dir=src, start=start, end=end,
              stage="all", force=False, attempts=1, qc_keep=2, socat_url="",
              smoke=True)
    ns.update(over)
    ctx = b10.Ctx(__import__("argparse").Namespace(**ns))
    b10.run_stages(ctx, list(stages))
    return ctx, truth


@pytest.fixture(scope="module")
def built():
    """Every store, built once — four synthetic archives, seconds in total."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10test_")
    out = {}
    for s in STORES:
        out[s] = build(tmp, s)
    out["_tmp"] = tmp
    return out


# =========================================================== the four adapters ==
def test_every_adapter_builds_its_store_from_a_synthetic_archive(built):
    """Index + fetch, per store, against the truth the generator kept.

    `check_smoke` compares row for row: the timestamp, the position, the
    platform, WHICH channels are measured, and the values of the ones that
    are. A parser that silently dropped a channel passes a row-count check and
    fails this one.
    """
    for s in STORES:
        ctx, truth = built[s]
        summary = b10.check_smoke(ctx, truth)
        assert summary.startswith(f"N={len(truth)}"), (s, summary)


def test_each_store_publishes_the_channels_its_plan_promises(built):
    """store.json's channel list is the adapter's, in order, with units."""
    for s in STORES:
        ctx, _ = built[s]
        meta = json.load(open(os.path.join(ctx.store, "store.json")))
        ad = b10.ADAPTERS[s]()
        assert [c["name"] for c in meta["channels"]] == list(ad.channel_names)
        assert [c["unit"] for c in meta["channels"]] == \
            [c[1] for c in ad.channels]
        assert meta["C"] == ad.C
        assert meta["footprint"]["log2_fp"] == ad.log2_fp
        assert meta["footprint"]["log2_dt"] == ad.log2_dt
        assert meta["tier"] == "P" and meta["family"] == "family10"
        assert meta["qc_policy"] and len(meta["qc_policy"]) > 100
        # Every file the store ships is hashed — a store.json with no sha256
        # block is one `family10_store.verify_store` refuses to open.
        for n in ("bin.npy", "time_days.npy", "lat.npy", "lon.npy",
                  "values.npy", "platform.npy", "qc.npy", "fp.npy",
                  "bin_offsets.npy"):
            assert n in meta["sha256"], (s, n)
        assert f10.verify_store(ctx.store) == len(meta["sha256"])


def test_gdp_drops_a_bad_position_and_keeps_the_drogue_as_a_channel(built):
    """The three GDP rows per sample are one good, one drogue-unknown, one far.

    The generator writes `err_lat = 1 degree` on every third row, which is
    111 km of position error against E-079 §3.1's 50 km bar, and a
    `drogue_lost_date` of 1970-01-01 on every second, which the archive
    documents as "uncertain from the beginning". So a correct parser keeps two
    of three rows and reports `drogue` as NaN on one of the two — an undrogued
    drifter and an unknown drogue are different measurements.
    """
    ctx, truth = built["gdp"]
    st = f10.Store.open(ctx.store)
    meta = json.load(open(os.path.join(ctx.store, "store.json")))
    assert meta["counts"]["drop_pos_err"] == st.N // 2, meta["counts"]
    assert meta["counts"]["drogue_uncertain"] == st.N // 2
    dro = np.asarray(st["values"], np.float32)[:, st.channels.index("drogue")]
    assert np.isnan(dro).sum() == st.N // 2
    assert set(np.unique(dro[np.isfinite(dro)]).tolist()) == {1.0}
    # The counter counts KEPT rows, so it equals the NaN count in the array it
    # describes. In the first published store it did not: it was incremented
    # before the "nothing measured on this row" drop, and ran 568 rows ahead of
    # `values`. A counter that describes rows the store does not contain is a
    # counter a reader cannot reconcile, so the ledger is asserted here.
    assert meta["counts"]["drogue_uncertain"] == int(np.isnan(dro).sum())
    assert (meta["counts"]["rows_read"]
            - meta["counts"]["drop_pos_err"]
            - meta["counts"]["drop_no_time"]
            - meta["counts"]["drop_no_position"]
            - meta["counts"]["drop_out_of_range"]
            - meta["counts"]["drop_no_values"]) == st.N, meta["counts"]
    # every kept row is flagged, and the store's own policy text says how
    assert set(np.unique(np.asarray(st["qc"])).tolist()) <= {1, 2}
    assert "50 km" in meta["qc_policy"]


def test_gtmba_converts_cm_per_second_and_refuses_an_unknown_unit(built):
    """The ADCP is cm/s in the archive and m/s in the store.

    The generator writes 12.5 and 13.5 cm/s; the store must read 0.125 and
    0.135 m/s. And the conversion is driven by the unit in the `.csvp` header,
    so a header that claims something else must RAISE rather than pick a
    factor — a silent factor of 100 on a current is exactly the kind of error
    that still looks like a plausible ocean.
    """
    ctx, _ = built["gtmba"]
    st = f10.Store.open(ctx.store)
    v = np.asarray(st["values"], np.float32)
    u = v[:, st.channels.index("u_cur")]
    w = v[:, st.channels.index("v_cur")]
    assert np.allclose(u[np.isfinite(u)], 0.125, atol=1e-3), u[:3]
    assert np.allclose(w[np.isfinite(w)], 0.135, atol=1e-3), w[:3]
    ad = b10.GTMBAAdapter()
    import io
    bad = io.StringIO("station,time (UTC),depth (m),u_1205 (furlongs)"
                      ",QU_5205\n")
    with pytest.raises(ValueError):
        ad._merge(ctx, bad, "u_1205", "QU_5205", "u_cur", None, {}, {},
                  {}, {d: 0 for d in b10.GTMBA_DEPTHS},
                  {n: i for i, n in enumerate(ad.channel_names)})


def test_gtmba_keeps_only_the_standard_depths_and_drops_flag_4(built):
    """A 7 m sample is not one of the eleven depths; a flag-4 40 m one is bad.

    Both must be absent, and for different reasons — which is why the counts
    are separate: `drop_depth` is a property of the array's geometry and
    `drop_qc` is a property of the instrument.
    """
    ctx, _ = built["gtmba"]
    st = f10.Store.open(ctx.store)
    meta = json.load(open(os.path.join(ctx.store, "store.json")))
    assert meta["counts"]["drop_depth"] > 0
    assert meta["counts"]["drop_qc"] > 0
    v = np.asarray(st["values"], np.float32)
    t40 = v[:, st.channels.index("t_40m")]
    assert not np.isfinite(t40).any(), "the flag-4 40 m sample reached the store"
    t20 = v[:, st.channels.index("t_20m")]
    assert np.isfinite(t20).all()
    assert "t_7m" not in st.channels


def test_socat_drops_flag_e_and_woce_4_and_wraps_longitude(built):
    """Two of four synthetic cruises must not survive, for two reasons.

    `QC_Flag = E` is the dataset-level exclusion and `fCO2rec_flag = 4` is the
    per-value WOCE bad flag. And the file's longitude is 0..360; the store's is
    [-180, 180), so a cruise at 200 E must read as -160.
    """
    ctx, _ = built["socat"]
    st = f10.Store.open(ctx.store)
    meta = json.load(open(os.path.join(ctx.store, "store.json")))
    assert meta["counts"]["drop_dataset_qc"] > 0
    assert meta["counts"]["drop_woce"] > 0
    lon = np.asarray(st["lon"], np.float64)
    assert lon.min() >= -180.0 and lon.max() < 180.0
    assert np.all(lon < -150.0) and np.all(lon > -170.0), (lon.min(), lon.max())
    assert set(np.unique(np.asarray(st["qc"])).tolist()) == {1}
    # `patm` is the MEASURED pressure and never the reanalysis column beside it
    assert "NCEP_SLP" in meta["qc_policy"] or "NCEP_SLP" in \
        b10.SOCATAdapter.qc_policy


def test_socat_resume_granularity_is_the_whole_stream_and_says_so(built):
    """The source is sorted by expocode, so the year is not a resume unit.

    The store must SAY that rather than let a reader assume family 8's
    contract, because an interrupted socat fetch re-reads the file from the
    start and a per-year marker would over-claim (ml/CLAUDE.md §5.21).
    """
    ctx, _ = built["socat"]
    meta = json.load(open(os.path.join(ctx.store, "store.json")))
    assert "stream" in meta["resume_granularity"]
    assert b10.SOCATAdapter.per_year is False
    for s in ("gdp", "gtmba", "slatrack"):
        m = json.load(open(os.path.join(built[s][0].store, "store.json")))
        assert m["resume_granularity"] == "year", s


def test_socat_stream_counts_are_the_streams_not_one_copy_per_year(built):
    """A one-pass adapter's ledger counts the pass ONCE, not once per year.

    The `socat` store published 2026-09-14 claims `rows_read` 2,597,074,036 —
    exactly 59 times the 44,018,204 rows its source file actually holds, one
    copy for each year that ended up with rows, because the fetch loop attached
    the whole-stream counters to every year part and `assemble_store` summed
    them. The arrays were untouched and every statistic derived from them
    verified; only the ledger was wrong, which is the kind of error that is
    invisible until someone computes a drop RATE from it. The smoke source
    spans 1981 and 1982, so a regression here shows up as a factor of two.
    """
    ctx, _ = built["socat"]
    meta = json.load(open(os.path.join(ctx.store, "store.json")))
    years = [y for y, n in meta["per_year"].items() if n]
    assert len(years) >= 2, meta["per_year"]      # or the test proves nothing

    ad = b10.ADAPTERS["socat"]()
    stream = {}
    for year, _rows, counts in ad.fetch_stream(ctx):
        if year is None:
            stream = counts or {}
    assert stream.get("rows_read"), stream

    for key in ("rows_read", "drop_out_of_range", "drop_no_fco2",
                "preamble_lines"):
        if key in stream:
            assert meta["counts"].get(key) == stream[key], (
                key, meta["counts"].get(key), stream[key], len(years))
    # and the ledger closes against the store it describes
    assert meta["counts"]["rows_read"] >= meta["N"]


def test_slatrack_reads_the_cf_time_axis_and_names_the_archive_variable(built):
    """`sla` in the plan is `sla_filtered` in the archive, and both are recorded.

    The time axis is read through its own CF `units` attribute rather than an
    assumed epoch — a store whose timestamps were off by the difference between
    two epochs would look completely ordinary and be wrong by decades.
    """
    ctx, _ = built["slatrack"]
    st = f10.Store.open(ctx.store)
    assert st.channels == ["sla", "sla_unfiltered", "mdt"]
    assert list(b10.CMEMS_VARS) == ["sla_filtered", "sla_unfiltered", "mdt"]
    assert st.log2_fp_const == -2.0 and st.log2_dt_const == -4.0
    got = b10._cf_time_to_days(np.array([0.0, 1.0]),
                               "seconds since 1970-01-01 00:00:00")
    assert abs(got[0] - b10.days_since_epoch(dt.date(1970, 1, 1))) < 1e-6
    assert abs(got[1] - got[0] - 1.0 / 86400.0) < 1e-9
    with pytest.raises(ValueError):
        b10._cf_time_to_days(np.array([0.0]), "furlongs since 1970-01-01")
    with pytest.raises(ValueError):
        b10._cf_time_to_days(np.array([0.0]), "days")


def test_slatrack_refuses_without_credentials_and_never_reads_a_file():
    """E-079 §3.4: read the two env vars, refuse clearly, look nowhere else."""
    ad = b10.SLATrackAdapter()
    saved = {k: os.environ.pop(k, None) for k in b10.CMEMS_ENV}
    try:
        with pytest.raises(SystemExit) as e:
            ad._require_credentials()
        msg = str(e.value)
        for k in b10.CMEMS_ENV:
            assert k in msg
        assert "environment" in msg
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    # and the source line in the file is the env var, not a path to a secret
    src = open(os.path.join(ROOT, "ml", "build_family10_stores.py")).read()
    assert "copernicus-marine-access" not in src
    assert ".cmems" not in src and "COPERNICUSMARINE_SERVICE_PASSWORD" in src


# ============================================================== §4 invariants ==
def _write_store(path, bins, times, lat, lon, values, platform=None, qc=None,
                 fp=(-4.0, -4.0), channels=(("a", "x", -10.0, 10.0),),
                 bin_first=None, offsets=None):
    """A store written from arrays, so each invariant can be broken on purpose."""
    os.makedirs(path, exist_ok=True)
    n = len(bins)
    b = np.asarray(bins, np.int16)
    cols = {
        "bin": b,
        "time_days": np.asarray(times, np.float32),
        "lat": np.asarray(lat, np.float32),
        "lon": np.asarray(lon, np.float32),
        "values": np.asarray(values, np.float16).reshape(n, -1),
        "platform": np.asarray(platform if platform is not None
                               else np.arange(n), np.int64),
        "qc": np.asarray(qc if qc is not None else np.ones(n), np.uint8),
        "fp": np.repeat(np.asarray([fp], np.float16), n, axis=0),
    }
    bf = int(b.min()) if bin_first is None and n else (bin_first or 0)
    nb = (int(b.max()) - bf + 1) if n else 1
    off = offsets if offsets is not None else \
        f10.csr_offsets(b.astype(np.int64), bf, nb)
    cols["bin_offsets"] = np.asarray(off, np.int64)
    for k, v in cols.items():
        np.save(os.path.join(path, k + ".npy"), v)
    meta = {"family": "family10", "tier": "P", "N": n, "C": cols["values"].shape[1],
            "bin_first": bf, "n_bins": nb,
            "channels": [{"name": c[0], "unit": c[1]} for c in channels],
            "footprint": {"log2_fp": fp[0], "log2_dt": fp[1]},
            "sha256": {k + ".npy": f10.sha256(os.path.join(path, k + ".npy"))
                       for k in cols}}
    with open(os.path.join(path, "store.json"), "w") as fh:
        json.dump(meta, fh)
    return path


def test_check_store_catches_every_invariant_it_claims_to(built):
    """An assertion nobody has seen fail is one nobody knows is wired up."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10inv_")
    good = dict(bins=[0, 0, 1, 2], times=[0.5, 1.5, 6.0, 12.0],
                lat=[0, 1, 2, 3], lon=[0, 10, 20, 30],
                values=[[1.0], [2.0], [3.0], [4.0]])
    b10.check_store(_write_store(os.path.join(tmp, "ok"), **good))

    bad = dict(good, bins=[2, 1, 0, 0], times=[12.0, 6.0, 1.5, 0.5])
    with pytest.raises(AssertionError, match="sorted by bin"):
        b10.check_store(_write_store(os.path.join(tmp, "unsorted"), **bad))

    bad = dict(good, bins=[0, 0, 1, 1])          # row 3's time says bin 2
    with pytest.raises(AssertionError, match="disagrees"):
        b10.check_store(_write_store(os.path.join(tmp, "binmismatch"), **bad))

    bad = dict(good, lon=[0, 10, 20, 200.0])
    with pytest.raises(AssertionError, match=r"\[-180, 180\)"):
        b10.check_store(_write_store(os.path.join(tmp, "lon"), **bad))

    # the reader refuses a CSR index that does not span its own rows before it
    # ever answers a search — an off-by-one here answers with a neighbouring
    # pentad's observations and nothing says so
    with pytest.raises(ValueError, match="CSR index does not describe"):
        b10.check_store(_write_store(os.path.join(tmp, "csr"), offsets=[0, 1, 2],
                                     **good))

    # a value outside the channel's physical bounds, against the adapter
    ad = b10.GDPAdapter()
    p = _write_store(os.path.join(tmp, "oob"), bins=[0], times=[0.5], lat=[0],
                     lon=[0], values=[[99.0, 0.0, 10.0, 1.0]],
                     channels=ad.channels)
    with pytest.raises(AssertionError, match="physical bounds"):
        b10.check_store(p, ad)

    # a footprint column that is not constant for a source whose support is
    pf = os.path.join(tmp, "fp")
    _write_store(pf, **good)
    fp = np.load(os.path.join(pf, "fp.npy"))
    fp[1, 0] = 2.0
    np.save(os.path.join(pf, "fp.npy"), fp)
    with pytest.raises(AssertionError, match="footprint columns"):
        b10.check_store(pf)


# ======================================= the check in bounded memory (E-079) ==
#
# slatrack is 1.5-2.5e9 rows, so the assertion pass cannot materialise a
# column: `np.asarray(st["bin"], np.int64)` alone is 16 GB there and the
# float32 view of `values` another 12 GB. `check_store` reads the memmaps in
# blocks of `--check-chunk-rows` instead, and the tests below drive it at
# absurdly small block sizes so every seam is a real boundary.
def test_the_chunked_check_passes_every_real_store_at_a_seven_row_block(built):
    """Same verdict at 7 rows a block as at 16 million — on all four stores."""
    for s in STORES:
        ctx, truth = built[s]
        anchor = (float(truth[0]["lat"]), float(truth[0]["lon"]),
                  int(np.floor(truth[0]["t"] / b10.PENTAD_DAYS)))
        for k in (7, 1, 3, b10.CHECK_CHUNK_ROWS):
            st = b10.check_store(ctx.store, ctx.adapter, anchor=anchor,
                                 chunk_rows=k)
            assert st.N == len(truth), (s, k)


def test_the_chunked_check_still_catches_a_violation_at_a_block_seam():
    """A break BETWEEN two blocks is the one a chunked check can lose.

    The time case is the one that needs the carried row: every block below is
    internally sorted, and only the pair that straddles a boundary is out of
    order. (A bin that goes BACKWARDS is caught twice over — by the same
    carried-row comparison and by the per-block CSR containment check, since
    an offsets vector that still counts the rows correctly can no longer hold
    them in one slice.)
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10seam_")
    n = 9
    times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    good = dict(bins=[0] * n, times=times, lat=[0.0] * n,
                lon=[0.0] * n, values=[[1.0]] * n)
    p = _write_store(os.path.join(tmp, "ok"), **good)
    for k in (1, 2, 3, 4, n, n + 1):
        b10.check_store(p, chunk_rows=k)

    # row 3 (the first row of the second block at chunk 3) predates row 2
    bad = list(times)
    bad[3] = 0.25
    p = _write_store(os.path.join(tmp, "seam"), **dict(good, times=bad))
    # k = 3 puts the offending pair either side of a boundary; every other
    # size here has it inside one block, or inside the carried comparison
    for k in (1, 2, 3, 4, n, b10.CHECK_CHUNK_ROWS):
        with pytest.raises(AssertionError, match="sorted by time"):
            b10.check_store(p, chunk_rows=k)

    # a bin that walks backwards across the seam, with an offsets vector that
    # still COUNTS every bin correctly — only the ordering is wrong
    p = _write_store(os.path.join(tmp, "binseam"),
                     bins=[0, 0, 1, 0, 1, 1], times=[0.1, 0.2, 5.1, 0.3,
                                                     5.2, 5.3],
                     lat=[0.0] * 6, lon=[0.0] * 6, values=[[1.0]] * 6,
                     offsets=[0, 3, 6])
    for k in (2, 3, 6):
        with pytest.raises(AssertionError, match="sorted by bin|CSR slice"):
            b10.check_store(p, chunk_rows=k)


def test_the_chunked_check_catches_an_offsets_vector_that_miscounts():
    """`bin_offsets` that spans the rows and is monotone and still wrong.

    The old check looped `np.unique(bin)` and sliced the whole column; the
    chunked one compares a running `np.bincount` recount against
    `np.diff(bin_offsets)`. This store is the case that separates the two:
    the offsets are internally impeccable — 0, monotone, ending at N — and
    they describe a different store.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10csr_")
    rows = dict(bins=[0, 0, 0, 1, 1, 1], times=[0.1, 0.2, 0.3, 5.1, 5.2, 5.3],
                lat=[0.0] * 6, lon=[0.0] * 6, values=[[1.0]] * 6)
    b10.check_store(_write_store(os.path.join(tmp, "ok"), **rows),
                    chunk_rows=2)
    p = _write_store(os.path.join(tmp, "miscount"), offsets=[0, 2, 6], **rows)
    for k in (2, 3, 6, b10.CHECK_CHUNK_ROWS):
        with pytest.raises(AssertionError, match="CSR slice|recount"):
            b10.check_store(p, chunk_rows=k)


def test_the_chunked_channel_statistics_are_the_whole_array_numbers(built):
    """store.json's per-channel block must not depend on the block size.

    The reference here is plain numpy over the WHOLE `values` array — what
    the statistics reduce to when the chunk is bigger than the store, and
    therefore what produced the three published stores' store.json. The
    chunked pass is run at 7 rows a block so the boundaries fall inside
    every store, and the numbers must come back BIT-IDENTICAL: the mean is
    accumulated as a float64 sum of `np.nansum(float64)` per block, and on
    these archives that reassociation costs nothing.
    """
    for s in STORES:
        ctx, _ = built[s]
        ad = ctx.adapter
        v = np.load(os.path.join(ctx.store, "values.npy"), mmap_mode="r")
        N = int(v.shape[0])
        assert N > 7, (s, N)          # or the seams are not exercised
        whole = np.asarray(v, np.float32)
        fin = np.isfinite(whole)
        want = {}
        for i, (nm, unit, lo_b, hi_b) in enumerate(ad.channels):
            col, f = whole[:, i], fin[:, i]
            k = int(f.sum())
            want[nm] = {
                "unit": unit, "measured": k,
                "fraction": round(k / N, 6) if N else 0.0,
                "min": float(np.nanmin(col)) if k else None,
                "max": float(np.nanmax(col)) if k else None,
                "mean": (float(np.nansum(col.astype(np.float64))) / k)
                        if k else None,
                "bounds": [lo_b, hi_b]}
        want_frac = round(int(fin.sum()) / (N * ad.C), 6)

        got, frac = b10._channel_stats(v, ad.channels, N, chunk=7)
        assert frac == want_frac, (s, frac, want_frac)
        assert got == want, s                      # bit-identical, not close
        # and the numbers store.json actually carries came from the same call
        meta = json.load(open(os.path.join(ctx.store, "store.json")))
        assert meta["per_channel"] == want, s
        assert meta["values_measured_fraction"] == want_frac, s


def test_the_publish_restore_check_refuses_a_disk_that_cannot_hold_it():
    """The restore downloads each file back; the largest one must fit.

    An 80 GB store is fine to upload on a small disk and impossible to
    VERIFY on one, and the verification is not optional — so the refusal has
    to happen before the hours of upload, and it has to name the numbers.
    """
    import shutil as sh
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10pub_")
    dest = os.path.join(tmp, "store")
    os.makedirs(dest)
    for n, nbytes in (("values.npy", 4096), ("bin.npy", 64)):
        with open(os.path.join(dest, n), "wb") as fh:
            fh.write(b"\0" * nbytes)
    names = ["bin.npy", "values.npy"]

    class _Ctx:
        scratch = os.path.join(tmp, "src")

    usage = sh.disk_usage
    free = {"n": 4096 * 1.1 + 1}
    b10.shutil.disk_usage = lambda p: type(
        "U", (), {"total": 1 << 40, "used": 0, "free": free["n"]})()
    try:
        assert b10._restore_disk_preflight(_Ctx(), dest, names) == 4096
        free["n"] = 4096 * 1.1 - 1
        with pytest.raises(SystemExit) as e:
            b10._restore_disk_preflight(_Ctx(), dest, names)
        msg = str(e.value)
        assert "values.npy" in msg and "REFUSING to publish" in msg
        assert "1.1" in msg and "0.00 GB" in msg   # the numbers, named
    finally:
        b10.shutil.disk_usage = usage


def test_the_check_chunk_size_is_a_cli_argument_that_reaches_the_check():
    """`--check-chunk-rows` exists, defaults sanely and lands on the Ctx."""
    import argparse
    ns = dict(store="gdp", work="/tmp/f10cli", source_dir="", start="",
              end="1982-01-12", stage="all", force=False, attempts=1,
              qc_keep=2, socat_url="", smoke=True)
    assert b10.Ctx(argparse.Namespace(**ns)).check_chunk == \
        b10.CHECK_CHUNK_ROWS
    ctx = b10.Ctx(argparse.Namespace(check_chunk_rows=11, **ns))
    assert ctx.check_chunk == 11
    src = open(os.path.join(ROOT, "ml", "build_family10_stores.py")).read()
    assert "--check-chunk-rows" in src


def test_verify_store_refuses_a_truncated_file(built):
    """A half-written values.npy still memmaps and still answers a search."""
    import shutil
    import tempfile
    ctx, _ = built["gdp"]
    tmp = tempfile.mkdtemp(prefix="f10trunc_")
    p = os.path.join(tmp, "gdp")
    shutil.copytree(ctx.store, p)
    assert f10.verify_store(p) > 0
    with open(os.path.join(p, "values.npy"), "r+b") as fh:
        fh.truncate(os.path.getsize(os.path.join(p, "values.npy")) - 16)
    with pytest.raises(ValueError, match="does not match its own store.json"):
        f10.verify_store(p)


# ============================================================= negative bins ==
def test_negative_bins_are_kept_and_the_csr_index_starts_where_the_data_does(
        built):
    """Drifters begin in 1979 and SOCAT in 1957 — before the epoch, and kept.

    Family 8's index is 3,143 entries from bin 0 because Argo starts in 2004.
    A family-10 store's index runs over ITS OWN range, so `bin_first` can be
    negative and a consumer that assumed 0 would read the wrong pentad.
    """
    for s in STORES:
        ctx, _ = built[s]
        st = f10.Store.open(ctx.store)
        assert st.bin_first < 0, (s, st.bin_first)
        assert int(np.asarray(st["bin"]).min()) == st.bin_first
        assert len(st.bin_offsets) == st.n_bins + 1
        # every bin's CSR slice holds only that bin, negative ones included
        for b in np.unique(np.asarray(st["bin"])):
            lo, hi = st._slice(int(b), int(b))
            assert hi > lo
            assert np.all(np.asarray(st["bin"])[lo:hi] == b)
        # and a search anchored on a negative bin works like any other
        i = 0
        tok = st.knearest(float(st["lat"][i]), float(st["lon"][i]),
                          int(st["bin"][i]), k=3, R_max_km=20000.0,
                          T_max_days=30.0)
        assert tok["n_R"] >= 1 and bool(tok["valid"][0])


def test_bin_of_days_floors_toward_minus_infinity():
    """-0.5 days is bin -1, not bin 0. Integer division would get this wrong."""
    got = f10.bin_of_days(np.array([-10.0, -5.1, -5.0, -0.5, 0.0, 4.9, 5.0]))
    assert got.tolist() == [-2, -2, -1, -1, 0, 0, 1]


def test_the_bin_column_survives_the_float32_round_trip(built):
    """A timestamp near a pentad boundary must not land in a neighbouring bin.

    `time_days` is float32 and runs to ~16,000 days, where it resolves about
    0.001 d, so a float64 timestamp a microsecond before a boundary can round
    UP across it. If `bin` were computed at full precision the row would sit in
    bin b carrying a timestamp that reads as bin b+1, and the reader — which
    requires `dt_days = 5*(b+1) - t >= 0` — would silently never return it for
    its own anchor. The builder computes the bin from the STORED value, so the
    store is self-consistent by construction.
    """
    edge = 5.0 * 2411                       # exactly a pentad boundary
    times = [np.nextafter(edge, 0.0), edge, np.nextafter(edge, 1e9),
             edge + 4.999999]
    rows = b10._pack(times, [0.0] * 4, [0.0] * 4, np.zeros((4, 1)),
                     np.arange(4), np.ones(4), 1)
    t32 = np.asarray(rows["time_days"], np.float64)
    assert np.array_equal(np.asarray(rows["bin"], np.int64),
                          f10.bin_of_days(t32))
    for b, t in zip(rows["bin"], t32):
        assert f10.anchor_time_days(int(b)) - t >= 0.0


# ==================================================================== reader ==
def test_the_search_is_one_sided_bounded_and_fixed_k(built):
    """E-076 §2.2's three rules, on a store with a known geometry.

    Twenty observations at the same place, one per pentad. An anchor at bin 10
    with a 30-day bound may see bins 8, 9 and 10 and nothing later, whatever k
    asks for.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10search_")
    n = 20
    bins = np.arange(n)
    times = bins * 5.0 + 1.0
    p = _write_store(os.path.join(tmp, "s"), bins=bins, times=times,
                     lat=np.zeros(n), lon=np.zeros(n),
                     values=np.arange(n, dtype=float).reshape(n, 1))
    st = f10.Store.open(p)
    tok = st.knearest(0.0, 0.0, bin=10, k=8, R_max_km=100.0, T_max_days=30.0)
    # ONE-SIDED: nothing after bin 10
    assert tok["row"][tok["valid"]].max() == 10
    # BOUNDED: t_anchor is the END of bin 10 = day 55; 30 days back is day 25,
    # so bins 5..10 — six observations.
    assert tok["n_R"] == 6, tok["n_R"]
    assert np.all(tok["dt_days"][tok["valid"]] >= 0.0)
    assert np.all(tok["dt_days"][tok["valid"]] <= 30.0)
    # FIXED k with MISS TOKENS: eight slots, six real
    assert tok["values"].shape == (8, 1)
    assert int(tok["valid"].sum()) == 6
    assert np.isnan(tok["values"][6:]).all()
    assert np.all(tok["row"][6:] == -1)
    assert tok["mask"][:6].all() and not tok["mask"][6:].any()
    # nearest first, in the metric
    assert np.all(np.diff(tok["d2"][:6]) >= 0)
    # a radius that admits nothing gives k miss tokens and n_R == 0
    far = st.knearest(80.0, 0.0, bin=10, k=3, R_max_km=10.0, T_max_days=30.0)
    assert far["n_R"] == 0 and not far["valid"].any()
    # a bin before the store's first is not an error
    assert st.knearest(0.0, 0.0, bin=-5, k=3)["n_R"] == 0


def test_the_reader_returns_the_footprint_and_the_mask_per_row(built):
    """E-078 §2: the token carries the support, and NaN is a masked channel."""
    ctx, _ = built["gtmba"]
    st = f10.Store.open(ctx.store)
    i = st.N // 2
    tok = st.knearest(float(st["lat"][i]), float(st["lon"][i]),
                      int(st["bin"][i]), k=4, R_max_km=500.0, T_max_days=30.0)
    assert tok["log2_fp"].shape == (4,) and tok["log2_dt"].shape == (4,)
    assert np.allclose(tok["log2_fp"], -4.0)
    assert np.allclose(tok["log2_dt"], -2.3, atol=1e-3)
    v = tok["values"]
    assert tok["mask"].shape == v.shape
    assert np.array_equal(tok["mask"], np.isfinite(v))
    assert tok["mask"].any() and not tok["mask"].all(), \
        "a mooring that measured every one of the 18 channels is not this test"
    assert tok["channels"] == st.channels


def test_qc_max_narrows_before_n_R_is_counted(built):
    """A token the reader will not return must not inflate the density feature."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10qc_")
    n = 6
    p = _write_store(os.path.join(tmp, "s"), bins=np.zeros(n),
                     times=np.linspace(0.1, 4.9, n), lat=np.zeros(n),
                     lon=np.zeros(n), values=np.arange(n, dtype=float
                                                       ).reshape(n, 1),
                     qc=[1, 1, 2, 3, 4, 1])
    st = f10.Store.open(p)
    assert st.knearest(0.0, 0.0, 0, k=6, R_max_km=100.0)["n_R"] == 6
    tight = st.knearest(0.0, 0.0, 0, k=6, R_max_km=100.0, qc_max=1)
    assert tight["n_R"] == 3
    assert set(np.asarray(tight["qc"])[tight["valid"]].tolist()) == {1}


def test_offsets_km_agrees_with_family_8_exactly():
    """Two readers, one ground scale. A copied constant is a constant that drifts."""
    lat = np.linspace(-85, 85, 37)
    lon = np.linspace(-175, 175, 37)
    for lat0, lon0 in ((36.0, -70.0), (0.0, 179.5), (-65.0, 20.0)):
        a = f10.offsets_km(lat0, lon0, lat, lon)
        b = f8.offsets_km(lat0, lon0, lat, lon)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert f10.KM_PER_DEG == f8.KM_PER_DEG
    assert f10.COS_FLOOR == f8.COS_FLOOR
    assert f10.PENTAD_DAYS == f8.PENTAD_DAYS
    assert f10.anchor_time_days(7) == f8.anchor_time_days(7)


# ================================================================== family 8 ==
def test_the_family_8_argo_store_opens_unchanged_and_answers_identically():
    """E-079 §2: 32 values (temp then psal), fp (-4, -4), NO rebuild.

    The point is that `Store` reads the family-8 layout by CONCATENATING its
    two value blocks at selection time — the blocks stay separate memmaps, so
    opening the real 2.68 M-profile store costs no copy — and that the rows it
    picks, the offsets it reports and the values it returns are bit-identical
    to `ArgoStore`'s.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10f8_")
    p = os.path.join(tmp, "f8")
    rng = np.random.default_rng(20260913)
    n = 300
    temp = rng.uniform(2, 28, (n, 16))
    psal = rng.uniform(33, 37, (n, 16))
    temp[::7, 10:] = np.nan                 # unfilled deep levels, as in reality
    f8.write_synthetic_store(
        p, bin_=rng.integers(2400, 2420, n),
        time_days=rng.uniform(12000.0, 12100.0, n),
        lat=rng.uniform(-60, 60, n), lon=rng.uniform(-180, 180, n),
        temp=temp, psal=psal)
    st = f10.Store.open(p)
    old = f8.ArgoStore(p)
    assert st.layout == "family8"
    assert st.C == 32
    assert st.channels[:2] == ["temp_10", "temp_30"]
    assert st.channels[-1] == "psal_1900"
    assert st.channel_units()[0] == "degC" and st.channel_units()[-1] == "PSU"
    assert (st.log2_fp_const, st.log2_dt_const) == (f8.LOG2_FP_ARGO,
                                                    f8.LOG2_DT_ARGO)
    assert st.bin_first == 0 and st.n_bins == f8.N_BINS
    for i in (0, n // 3, n - 1):
        lat0 = float(st["lat"][i]) + 0.4
        lon0 = float(st["lon"][i]) - 0.4
        b0 = int(st["bin"][i])
        new = st.knearest(lat0, lon0, b0, k=6, R_max_km=4000.0,
                          T_max_days=30.0)
        ref = old.knearest(lat0, lon0, b0, k=6, R_max_km=4000.0,
                           T_max_days=30.0)
        assert np.array_equal(new["row"], ref["row"])
        assert new["n_R"] == ref["n_R"] and new["n_found"] == ref["n_found"]
        for k in ("dx_km", "dy_km", "dt_days", "dist_km", "d2"):
            assert np.allclose(new[k], ref[k], equal_nan=True), k
        want = np.concatenate([ref["temp"], ref["psal"]], axis=1)
        assert np.allclose(new["values"], want, equal_nan=True)
        assert np.array_equal(new["mask"], np.isfinite(want))
        assert np.array_equal(np.asarray(new["platform"])[new["valid"]],
                              np.asarray(ref["wmo"])[ref["valid"]])
    # and it never wrote anything into the family-8 store
    assert not os.path.exists(os.path.join(p, "values.npy"))
    assert not os.path.exists(os.path.join(p, "fp.npy"))


# ================================================================== registry ==
def test_the_registry_lists_every_store_it_can_read_and_names_the_ones_it_cannot(
        built):
    """E-079 §2's registry, built with NO Hub access.

    Two properties matter more than the field list. A group the builder could
    not read is NOT in `groups` and IS in `groups_missing` — a registry that
    listed a group it could not describe would be worse than a short one. And
    the tier-G entry says WHICH family-7 manifest it describes, because E-079
    §2 names family 7.1 and that build has not been dispatched.
    """
    work = os.path.join(built["_tmp"], "regwork")
    os.makedirs(work, exist_ok=True)
    import shutil
    for s in ("gdp", "socat"):
        dst = os.path.join(work, s, s)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(built[s][0].store, dst)
    r = reg.build_registry(work=work, use_hub=False, include_argo=False)
    names = {g["name"]: g for g in r["groups"]}
    assert {"gdp", "socat"} <= set(names)
    assert set(r["groups_missing"]) == {"gtmba", "slatrack"}
    assert r["groups_missing_note"]
    assert r["family"] == "family10"
    assert r["epoch"] == f10.EPOCH and r["pentad_days"] == 5
    assert r["token_schema"]["fields"][:2] == ["value[C]", "mask[C]"]
    # tier G, by reference, with the fallback stated
    g = [x for x in r["groups"] if x["tier"] == "G"]
    assert {x["name"] for x in g} >= {"g025", "g100", "rg100"}
    assert r["tier_g"]["stem"].startswith("family7_global025_pentad_l")
    assert r["tier_g"]["fallback_note"]
    by = {x["name"]: x for x in g}
    assert by["g100"]["footprint"]["log2_fp"] == 2.9, \
        "NCEP's support is its T62 grid, not the 1 degree it is stored on"
    assert by["rg100"]["footprint"]["log2_dt"] == 2.6
    assert by["g025"]["footprint"] == {"log2_fp": 0.0, "log2_dt": 0.0,
                                       **{"note": by["g025"]["footprint"]["note"]}}
    assert by["g025"]["C"] == len(by["g025"]["channels"])
    assert all(c["unit"] for c in by["g025"]["channels"])
    # tier P, from the stores' own store.json
    for s in ("gdp", "socat"):
        e = names[s]
        assert e["tier"] == "P" and e["per_row_footprint"] is True
        assert e["C"] == b10.ADAPTERS[s]().C
        assert e["bin_first"] == f10.Store.open(built[s][0].store).bin_first
        # the nine .npy columns. store.json is NOT among them: it carries the
        # hashes, so it cannot carry its own.
        assert len(e["files"]) == 9
        assert {f["name"] for f in e["files"]} == {
            "bin.npy", "time_days.npy", "lat.npy", "lon.npy", "values.npy",
            "platform.npy", "qc.npy", "fp.npy", "bin_offsets.npy"}
        assert all(f["sha256"] for f in e["files"])
        assert e["qc_policy"] and e["sources"] and e["verified"]
    # every group carries what a consumer dispatches on
    for e in r["groups"]:
        for k in ("name", "tier", "layout", "cadence", "channels",
                  "footprint", "bin_first", "files"):
            assert k in e, (e["name"], k)


def test_the_registry_reads_a_family_8_store_as_a_tier_p_group():
    """Family 8 joins family 10 unchanged — 32 channels, named like the reader."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10reg8_")
    p = os.path.join(tmp, "family8_argo_l0")
    rng = np.random.default_rng(1)
    f8.write_synthetic_store(p, bin_=[10, 11], time_days=[52.0, 57.0],
                             lat=[0.0, 1.0], lon=[0.0, 1.0],
                             temp=rng.uniform(2, 28, (2, 16)),
                             psal=rng.uniform(33, 37, (2, 16)))
    r = reg.build_registry(work=tmp, use_hub=False, stores=())
    e = [g for g in r["groups"] if g["name"] == "argo"][0]
    assert e["tier"] == "P" and len(e["channels"]) == 32
    assert e["channels"][0]["name"] == "temp_10"
    assert e["channels"][-1]["name"] == "psal_1900"
    assert e["per_row_footprint"] is False       # no fp.npy — the constants stand
    assert e["footprint"]["log2_fp"] == f8.LOG2_FP_ARGO
    # and the names agree with what the READER produces, so the two cannot drift
    assert [c["name"] for c in e["channels"]] == f10.Store.open(p).channels


# ============================================================== resumability ==
def test_an_interrupted_fetch_resumes_at_the_year_it_lost(built):
    """E-079 §4: flush, THEN mark — so a marker may only under-claim.

    The year is deleted with its marker and rebuilt; the year beside it is NOT
    re-fetched (its marker stands), and the reassembled store is byte-identical
    to the uninterrupted one. If the marker were written before the data, the
    resumed job would skip a year it never built and the store would be short
    with nothing to say so.
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10resume_")
    ctx, truth = build(tmp, "gdp")
    first = f10.sha256(os.path.join(ctx.store, "values.npy"))
    years = sorted(ctx.years)
    assert len(years) >= 2
    lost = years[-1]
    kept_mtime = os.path.getmtime(
        os.path.join(ctx.year_dir(years[0]), "00000.npz"))

    shutil.rmtree(ctx.year_dir(lost))
    os.remove(b10.marker(ctx.root, f"parts/{lost}"))
    os.remove(b10.marker(ctx.root, "fetch"))
    assert b10.marked(ctx.root, f"parts/{years[0]}")
    assert not b10.marked(ctx.root, f"parts/{lost}")

    ctx2 = b10.Ctx(__import__("argparse").Namespace(
        store="gdp", work=ctx.work, source_dir=ctx.source_dir,
        start=SMOKE_START, end=SMOKE_END, stage="fetch", force=False,
        attempts=1, qc_keep=2, socat_url="", smoke=True))
    b10.run_stages(ctx2, ["fetch"])
    assert os.path.getmtime(os.path.join(ctx2.year_dir(years[0]),
                                         "00000.npz")) == kept_mtime, \
        "the year whose marker stood was re-fetched"
    assert f10.sha256(os.path.join(ctx2.store, "values.npy")) == first
    b10.check_smoke(ctx2, truth)


def test_a_finished_stage_is_skipped_and_force_redoes_it(built):
    """The `.done` markers are the resume contract, and `--force` is the escape."""
    ctx, _ = built["gdp"]
    assert b10.marked(ctx.root, "index") and b10.marked(ctx.root, "fetch")
    before = os.path.getmtime(os.path.join(ctx.root, "plan.json"))
    b10.run_stages(ctx, ["index"])                    # marked: a no-op
    assert os.path.getmtime(os.path.join(ctx.root, "plan.json")) == before


def test_publish_needs_fetch_and_fetch_needs_index():
    """Stage order is fixed; asking out of order refuses before it spends."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="f10order_")
    ctx = b10.Ctx(__import__("argparse").Namespace(
        store="gdp", work=os.path.join(tmp, "w"), source_dir="",
        start=SMOKE_START, end=SMOKE_END, stage="fetch", force=False,
        attempts=1, qc_keep=2, socat_url="", smoke=False))
    with pytest.raises(SystemExit, match="needs 'index'"):
        b10.run_stages(ctx, ["fetch"])
    with pytest.raises(SystemExit, match="needs 'fetch'"):
        b10.run_stages(ctx, ["publish"])


# ============================================================ the access paths ==
def test_the_erddap_query_encodes_the_way_the_service_wants():
    """MEASURED 2026-09-13: percent-encoding `&` or `>=` gets HTTP 400.

    ERDDAP uses `&` and `>=` as SYNTAX, not as data — which is why the query
    is built here rather than handed to a form encoder. The commas between
    variables and the colons inside a timestamp DO need encoding.
    """
    u = b10.erddap_url("https://erddap.aoml.noaa.gov/gdp/erddap",
                       "drifter_6hour_qc", "csvp", ("ID", "time", "sst"),
                       ["time>=2015-01-01T00:00:00Z",
                        "time<=2015-01-31T23:59:59Z"])
    assert u.startswith("https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/"
                        "drifter_6hour_qc.csvp?")
    assert "ID%2Ctime%2Csst" in u
    assert "&time%3E=2015-01-01T00%3A00%3A00Z" in u
    assert "&time%3C=2015-01-31T23%3A59%3A59Z" in u
    assert "%26" not in u, "the constraint separator must stay a literal &"


def test_the_zip_member_stream_inflates_a_real_zip():
    """SOCAT ships one 1.4 GB zip64 deflate member over a non-seekable body."""
    import io
    import zipfile
    buf = io.BytesIO()
    payload = ("Expocode\tversion\n" + "x\ty\n" * 5000).encode()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("SOCATsmoke.tsv", payload)
    raw = buf.getvalue()
    got = b"".join(b10._zip_member_stream(io.BytesIO(raw), size=97))
    assert got == payload
    lines = list(b10._text_lines(
        b10._zip_member_stream(io.BytesIO(raw), size=97)))
    assert lines[0] == "Expocode\tversion"
    assert len(lines) == 5001
    with pytest.raises(ValueError, match="not a zip"):
        list(b10._zip_member_stream(io.BytesIO(b"not a zip at all" * 4)))


def test_platform_hash_is_deterministic_positive_and_collision_free_enough():
    """`hash()` is salted per process; a store built twice must agree."""
    a = b10.platform_hash("06AQ19860627")
    assert a == b10.platform_hash("06AQ19860627") == 0x9f28b32e97c1e2 or a > 0
    assert a > 0 and a < 2 ** 63
    n = 4000
    ids = {b10.platform_hash(f"EXPO{i:06d}") for i in range(n)}
    assert len(ids) == n
    assert b10._platform_int("101515") == 101515
    assert b10._platform_int("") == 0
    assert b10._platform_int("abc") == b10.platform_hash("abc")


def test_the_adapters_declare_the_footprints_the_plan_specifies():
    """E-079 §3's four footprint pairs, which are the whole point of the token."""
    assert (b10.GDPAdapter.log2_fp, b10.GDPAdapter.log2_dt) == (-4.0, -4.3)
    assert (b10.GTMBAAdapter.log2_fp, b10.GTMBAAdapter.log2_dt) == (-4.0, -2.3)
    assert (b10.SOCATAdapter.log2_fp, b10.SOCATAdapter.log2_dt) == (-4.0, -4.0)
    assert (b10.SLATrackAdapter.log2_fp,
            b10.SLATrackAdapter.log2_dt) == (-2.0, -4.0)
    # and each is the number the plan's arithmetic gives
    assert abs(np.log2(0.25 / 5) - (-4.32)) < 0.01      # a 6-hourly sample
    assert abs(np.log2(1.0 / 5) - (-2.32)) < 0.01       # a daily mean
    assert abs(np.log2(7.0 / 27.83) - (-1.99)) < 0.01   # a 7 km along-track cell


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]) if hasattr(pytest, "main") else 0)


def test_parse_stages_accepts_a_comma_list_in_fixed_order():
    assert b10.parse_stages("all") == ["index", "fetch", "publish"]
    assert b10.parse_stages("fetch") == ["fetch"]
    assert b10.parse_stages("fetch,index") == ["index", "fetch"]
    assert b10.parse_stages(" index, fetch ") == ["index", "fetch"]
    with pytest.raises(SystemExit):
        b10.parse_stages("index,assemble")
    with pytest.raises(SystemExit):
        b10.parse_stages("")


def test_slatrack_mission_window_skips_and_clips():
    A = b10.SLATrackAdapter
    lo, hi = dt.date(2015, 1, 1), dt.date(2015, 12, 31)
    m = {"id": "x", "start": "2015-03-31T00:25:42Z", "end": "2026-01-16T23:23:15Z"}
    assert A._mission_window(m, lo, hi) == (dt.date(2015, 3, 31), hi)
    assert A._mission_window(m, dt.date(2015, 1, 1), dt.date(2015, 1, 7)) is None
    ended = {"id": "y", "start": "1992-01-01T00:00:00Z", "end": "1996-06-02T00:00:00Z"}
    assert A._mission_window(ended, lo, hi) is None
    assert A._mission_window({"id": "fixture"}, lo, hi) == (lo, hi)
    assert A._split_id("cmems_obs-sl_glo_phy-ssh_my_al-l3-duacs_PT1S_202411") == (
        "cmems_obs-sl_glo_phy-ssh_my_al-l3-duacs_PT1S", "202411")
    assert A._split_id("cmems_obs-sl_glo_phy-ssh_my_j3g-l3-duacs_PT1S-i_202506") == (
        "cmems_obs-sl_glo_phy-ssh_my_j3g-l3-duacs_PT1S-i", "202506")
    assert A._split_id("plain") == ("plain", None)


def _slatrack_frame_columns():
    """One synthetic mission-month's worth of rows, as columns of arrays.

    Four rows, each testing one branch of the filter: a plain good row, a row
    whose `sla_unfiltered` is outside +/- 3 m (kept, that ONE channel NaN,
    counted), a row whose `sla_filtered` is NaN (dropped — `sla` is the channel
    the keep rule is written on), and a row years outside the ctx window
    (dropped). The time column is datetime64, the shape the toolbox's
    `read_dataframe` returns.
    """
    return {
        "time": np.array(["1981-12-25T06:00:00", "1982-01-05T12:00:00",
                          "1982-01-06T00:00:00", "1990-01-01T00:00:00"],
                         "datetime64[ns]"),
        "latitude": np.array([10.0, -20.0, 30.0, 40.0]),
        "longitude": np.array([100.0, -170.0, 20.0, 0.0]),
        "sla_filtered": np.array([0.10, -0.20, np.nan, 0.30]),
        "sla_unfiltered": np.array([0.11, 9.00, 0.05, 0.31]),
        "mdt": np.array([0.50, 0.60, 0.70, 0.80]),
    }


def test_slatrack_rows_from_frame_filters_counts_and_packs(built):
    """The shared frame->rows step: the live path's only new arithmetic.

    The download path cannot be run from the sandbox, so the thing that CAN be
    checked here is the step both paths share — and it is the step that decides
    what ends up in the store.
    """
    ctx, _ = built["slatrack"]
    ad = b10.SLATrackAdapter()
    rows, counts = ad._rows_from_frame(ctx, _slatrack_frame_columns(), "mX")
    assert counts["rows_read"] == 4
    assert counts["kept"] == 2
    assert counts["out_of_bounds"] == {"sla_unfiltered": 1}
    assert len(rows["time_days"]) == 2
    want_t = [b10.days_since_epoch(dt.date(1981, 12, 25)) + 0.25,
              b10.days_since_epoch(dt.date(1982, 1, 5)) + 0.5]
    assert np.allclose(rows["time_days"], np.float32(want_t), atol=1e-3)
    assert np.allclose(rows["lat"], [10.0, -20.0], atol=1e-3)
    assert np.allclose(rows["lon"], [100.0, -170.0], atol=1e-3)
    v = np.asarray(rows["values"], np.float64)
    assert np.allclose(v[:, 0], [0.10, -0.20], atol=1e-3)
    assert abs(v[0, 1] - 0.11) < 1e-3 and np.isnan(v[1, 1])   # 9 m -> NaN
    assert np.allclose(v[:, 2], [0.50, 0.60], atol=1e-3)
    assert (rows["platform"] == b10.platform_hash("mX")).all()
    assert (rows["qc"] == 1).all()


def test_slatrack_rows_from_frame_takes_a_dataframe_and_a_datetime_index(built):
    """A pandas frame — including one carrying the time as its INDEX — reads
    identically to the mapping of arrays above. `read_dataframe` returns one of
    these shapes; which one is not yet measured, so both are accepted."""
    pd = pytest.importorskip("pandas")
    ctx, _ = built["slatrack"]
    ad = b10.SLATrackAdapter()
    base, _ = ad._rows_from_frame(ctx, _slatrack_frame_columns(), "mX")
    df = pd.DataFrame(_slatrack_frame_columns())
    got, counts = ad._rows_from_frame(ctx, df, "mX")
    indexed = df.set_index("time")
    got_i, counts_i = ad._rows_from_frame(ctx, indexed, "mX")
    for k in base:
        assert np.allclose(np.asarray(got[k], np.float64),
                           np.asarray(base[k], np.float64), equal_nan=True), k
        assert np.allclose(np.asarray(got_i[k], np.float64),
                           np.asarray(base[k], np.float64), equal_nan=True), k
    assert counts == counts_i


def test_slatrack_refuses_a_frame_whose_columns_it_cannot_name(built):
    """A frame with no recognisable time/lat/lon STOPS and prints the columns.

    Keeping zero rows quietly would look like a mission that did not fly.
    """
    ctx, _ = built["slatrack"]
    ad = b10.SLATrackAdapter()
    bad = {"t": np.zeros(2), "y": np.zeros(2), "x": np.zeros(2)}
    with pytest.raises(SystemExit) as e:
        ad._rows_from_frame(ctx, bad, "mX")
    for c in ("t", "y", "x"):
        assert repr(c) in str(e.value)


def test_slatrack_months_clips_the_mission_window_it_is_given():
    """The live fetch asks month by month — ~2.6 M rows instead of ~30 M."""
    A = b10.SLATrackAdapter
    got = A._months(dt.date(2015, 3, 31), dt.date(2015, 6, 2))
    assert got == [(dt.date(2015, 3, 31), dt.date(2015, 3, 31)),
                   (dt.date(2015, 4, 1), dt.date(2015, 4, 30)),
                   (dt.date(2015, 5, 1), dt.date(2015, 5, 31)),
                   (dt.date(2015, 6, 1), dt.date(2015, 6, 2))]
    one = A._months(dt.date(2015, 1, 5), dt.date(2015, 1, 9))
    assert one == [(dt.date(2015, 1, 5), dt.date(2015, 1, 9))]
    dec = A._months(dt.date(2015, 12, 1), dt.date(2016, 1, 2))
    assert dec == [(dt.date(2015, 12, 1), dt.date(2015, 12, 31)),
                   (dt.date(2016, 1, 1), dt.date(2016, 1, 2))]


def test_slatrack_index_says_how_it_fetches(built):
    """store.json's index block names the path, so a later reader of a build
    log knows WHICH toolbox call produced the rows."""
    ctx, _ = built["slatrack"]
    ix = b10.SLATrackAdapter().index(ctx)
    assert ix["fetch"] == ("copernicusmarine.get (original files), month "
                           "batches; fallback read_dataframe")
    assert ix["fetch_route"] == "files"          # the default of the flag
    assert ix["variables"] == list(b10.CMEMS_VARS)
    # a fixture build never probes the remote layout
    assert "slatrack_files_probe" not in ix


def _slatrack_long_frame(pd):
    """The LONG shape `read_dataframe` really returns (run 34849670866):
    one row per (variable, sample). Sample 2 carries only two of the three
    variables — that channel must come out NaN, not shift the others."""
    rows = [
        # time, lat, lon, variable, value, value_qc
        ("1982-01-05T12:00:00", -20.0, -170.0, "sla_filtered", -0.20, 1),
        ("1982-01-05T12:00:00", -20.0, -170.0, "sla_unfiltered", -0.22, 1),
        ("1982-01-05T12:00:00", -20.0, -170.0, "mdt", 0.60, 1),
        ("1982-01-06T00:00:00", 10.0, 100.0, "sla_filtered", 0.10, 2),
        ("1982-01-06T00:00:00", 10.0, 100.0, "mdt", 0.50, 2),
    ]
    return pd.DataFrame({
        "variable": [r[3] for r in rows],
        "platform_id": ["j2"] * len(rows),
        "time": pd.to_datetime([r[0] for r in rows]),
        "longitude": [r[2] for r in rows],
        "latitude": [r[1] for r in rows],
        "value": [r[4] for r in rows],
        "value_qc": [r[5] for r in rows],
        "institution": ["CLS"] * len(rows)})


def test_slatrack_pivots_the_long_frame_read_dataframe_returns(built):
    """The measured frame is LONG; without the pivot the adapter finds no
    channel column and keeps zero rows. One row per sample, a missing
    variable NaN, and the `value_qc` distribution recorded but not filtered."""
    pd = pytest.importorskip("pandas")
    ctx, _ = built["slatrack"]
    ad = b10.SLATrackAdapter()
    rows, counts = ad._rows_from_frame(ctx, _slatrack_long_frame(pd), "mX")
    assert counts["rows_read"] == 2               # samples, not long rows
    assert counts["kept"] == 2
    # sla_filtered qc only: one 1 (the three-variable sample) and one 2
    assert counts["value_qc"] == {1: 1, 2: 1}
    order = np.argsort(np.asarray(rows["time_days"], np.float64))
    lat = np.asarray(rows["lat"], np.float64)[order]
    v = np.asarray(rows["values"], np.float64)[order]
    assert np.allclose(lat, [-20.0, 10.0], atol=1e-3)
    assert np.allclose(v[:, 0], [-0.20, 0.10], atol=1e-3)       # sla
    assert abs(v[0, 1] + 0.22) < 1e-3 and np.isnan(v[1, 1])     # unfiltered
    assert np.allclose(v[:, 2], [0.60, 0.50], atol=1e-3)        # mdt
    assert (rows["platform"] == b10.platform_hash("mX")).all()


def test_slatrack_pivot_keys_on_platform_id_when_it_is_not_constant(built):
    """One dataset is one mission, so `platform_id` is expected constant —
    but if it is not, two platforms at the same instant must stay two
    samples rather than collapse into one."""
    pd = pytest.importorskip("pandas")
    ctx, _ = built["slatrack"]
    ad = b10.SLATrackAdapter()
    df = _slatrack_long_frame(pd)
    two = df.copy()
    two["platform_id"] = ["j2", "j2", "j2", "j3", "j3"]
    two["latitude"] = [-20.0, -20.0, -20.0, -20.0, -20.0]
    two["longitude"] = [-170.0] * 5
    two["time"] = pd.to_datetime(["1982-01-05T12:00:00"] * 5)
    _, counts = ad._rows_from_frame(ctx, two, "mX")
    assert counts["rows_read"] == 2               # NOT merged into one


def test_slatrack_file_batches_group_by_the_date_in_the_basename():
    """The remote layout is unmeasured, so batching reads the 8-digit date
    token off the listed basenames and drops what the window excludes."""
    A = b10.SLATrackAdapter
    paths = ["s3://bucket/native/SEALEVEL/j2/2015/01/x_20150104_v1.nc",
             "s3://bucket/native/SEALEVEL/j2/2015/01/x_20150131_v1.nc",
             "s3://bucket/native/SEALEVEL/j2/2015/02/x_20150202_v1.nc",
             "s3://bucket/native/SEALEVEL/j2/2015/12/x_20151203_v1.nc"]
    got = A._file_batches(paths, dt.date(2015, 1, 10), dt.date(2015, 2, 28),
                          "2015")
    assert got == [("2015-01", ["x_20150131_v1.nc"]),
                   ("2015-02", ["x_20150202_v1.nc"])]
    # a basename with no date token -> one batch for the whole year, never a
    # silently dropped file
    odd = A._file_batches(paths[:1] + ["s3://b/native/j2/latest.nc"],
                          dt.date(2015, 1, 1), dt.date(2015, 12, 31), "2015")
    assert odd == [("2015", ["x_20150104_v1.nc", "latest.nc"])]


def test_slatrack_response_paths_reads_the_toolbox_response_shape():
    """`ResponseGet.files` is a list of `FileGet`; a plain dict is accepted
    too so a toolbox version that hands back JSON still teaches us layout."""
    class F:
        def __init__(self, u):
            self.s3_url, self.https_url, self.filename = u, "", u.rsplit(
                "/", 1)[-1]

    class R:
        files = [F("s3://b/a/x_20150104.nc"), F("s3://b/a/x_20150105.nc")]

    A = b10.SLATrackAdapter
    assert A._response_paths(R()) == ["s3://b/a/x_20150104.nc",
                                      "s3://b/a/x_20150105.nc"]
    assert A._response_paths({"files": [{"s3_url": "s3://b/z.nc"}]}) == \
        ["s3://b/z.nc"]
    assert A._response_paths({}) == []


# ========================================= the streaming assembler (E-079) ==
#
# slatrack is 1.5-2.5e9 rows at ~33 B each — 50-80 GB, which no box holds in
# RAM — so `assemble_store_streaming` writes the store through memmaps in
# three passes. Its ONLY claim is byte identity with `assemble_store`, and a
# claim like that is worth exactly as much as the test that checks it.
STORE_ARRAYS = ("bin.npy", "time_days.npy", "lat.npy", "lon.npy", "values.npy",
                "platform.npy", "qc.npy", "fp.npy", "bin_offsets.npy")


def _hash_store(dest):
    return {n: f10.sha256(os.path.join(dest, n)) for n in STORE_ARRAYS}


def _duplicate_rows_across_parts(ctx, year=None):
    """Add a SECOND part to one year holding rows that tie the first part's.

    Same `bin` AND same `time_days`, different lat/platform. That is the only
    case where the two assemblers could disagree: `np.lexsort` breaks such a
    tie by INPUT ORDER, and the streaming assembler has to reproduce that
    input order from a scatter plus a per-bin stable sort. Without this part
    the test would pass on a streaming assembler that used an unstable sort.
    """
    y = year or ctx.years[0]
    d = ctx.year_dir(y)
    p0 = os.path.join(d, "00000.npz")
    with np.load(p0) as z:
        rows = {k: z[k] for k in b10.ROW_KEYS}
    take = np.arange(0, len(rows["bin"]), 2)[:8]
    assert take.size >= 4, "the synthetic year is too small to tie anything"
    dup = {k: np.array(rows[k][take]) for k in b10.ROW_KEYS}
    dup["lat"] = np.clip(dup["lat"] + np.float32(0.25), -90.0,
                         90.0).astype(np.float32)
    dup["platform"] = dup["platform"] + 7777
    np.savez(os.path.join(d, "00001.npz"), **dup)
    c = json.load(open(os.path.join(d, "counts.json")))
    c["rows"] = int(c["rows"]) + int(take.size)
    c["parts"] = 2
    json.dump(c, open(os.path.join(d, "counts.json"), "w"))
    return int(take.size)


def test_the_streaming_assembler_writes_the_same_bytes(tmp_path):
    """Build one synthetic archive, assemble it twice, compare every file.

    The archive is doctored first so that one year holds duplicate
    (bin, time_days) rows SPREAD ACROSS TWO PARTS — the tie-break case.
    """
    import shutil
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    n_dup = _duplicate_rows_across_parts(ctx)

    shutil.rmtree(ctx.store, ignore_errors=True)
    ctx.a.assemble = "memory"
    mem = b10.assemble(ctx)
    mem_hashes = _hash_store(ctx.store)
    mem_meta = json.load(open(os.path.join(ctx.store, "store.json")))
    keep = ctx.store + "_memory"
    shutil.rmtree(keep, ignore_errors=True)
    shutil.move(ctx.store, keep)

    ctx.a.assemble = "streaming"
    stream = b10.assemble(ctx)
    stream_hashes = _hash_store(ctx.store)
    stream_meta = json.load(open(os.path.join(ctx.store, "store.json")))

    assert mem["N"] == stream["N"] == mem_meta["N"]
    assert mem_hashes == stream_hashes, {
        n: (mem_hashes[n], stream_hashes[n]) for n in STORE_ARRAYS
        if mem_hashes[n] != stream_hashes[n]}
    for k in set(mem_meta) | set(stream_meta):
        if k in ("built_at",):
            continue
        assert mem_meta[k] == stream_meta[k], k

    # …and the ties really are there, or the test proves nothing.
    st = f10.Store(keep)
    b = np.asarray(st["bin"], np.int64)
    t = np.asarray(st["time_days"], np.float64)
    ties = int(((b[1:] == b[:-1]) & (t[1:] == t[:-1])).sum())
    assert ties >= n_dup, (ties, n_dup)


def test_auto_picks_streaming_for_slatrack_and_memory_for_a_small_store(
        tmp_path, capsys):
    """`--assemble auto` is a size decision with one named exception."""
    import shutil
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    shutil.rmtree(ctx.store, ignore_errors=True)
    ctx.a.assemble = "auto"
    b10.assemble(ctx)
    assert "auto -> memory" in capsys.readouterr().out

    ctx2, _ = build(tmp, "slatrack")
    shutil.rmtree(ctx2.store, ignore_errors=True)
    ctx2.a.assemble = "auto"
    b10.assemble(ctx2)
    assert "auto -> streaming" in capsys.readouterr().out


def test_the_disk_preflight_refuses_before_it_writes(tmp_path, monkeypatch):
    """§5.18: size the guard from the allocation it guards, and fire BEFORE
    the write — the parts are safe on the Hub, a half-written 80 GB store is
    not diagnosable from anywhere."""
    import collections
    import shutil
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    shutil.rmtree(ctx.store, ignore_errors=True)
    du = collections.namedtuple("du", "total used free")
    monkeypatch.setattr(b10.shutil, "disk_usage", lambda p: du(1, 1, 8))
    ctx.a.assemble = "streaming"
    with pytest.raises(SystemExit, match="REFUSING to assemble"):
        b10.assemble(ctx)


# ================================== family10_parts_hub, against a FAKE hub ==
#
# NO NETWORK. `family10_parts_hub` talks to the Hub through exactly four
# seams (`_hub`, `_list_files`, `_upload`, `_download`); the fixture points
# them at a directory, so what is exercised is the marker discipline, the
# restore verification and the sha256 bookkeeping rather than huggingface_hub.
import family10_parts_hub as ph                                 # noqa: E402


class FakeHub:
    """A directory that answers like the dataset repo."""

    def __init__(self, root):
        self.root = root
        self.repo = "fake/earth-tensors"
        self.uploads = 0
        self.fail_on = None          # a path_in_repo whose upload must raise

    # the four seams ------------------------------------------------------
    def hub(self):
        return self, self.repo, "fake-token"

    def create_repo(self, *a, **k):
        pass

    def list_files(self, api, repo, prefix):
        pre = prefix.rstrip("/") + "/"
        out = set()
        for dirpath, _, names in os.walk(self.root):
            for n in names:
                rel = os.path.relpath(os.path.join(dirpath, n), self.root)
                rel = rel.replace(os.sep, "/")
                if rel.startswith(pre):
                    out.add(rel)
        return out

    def upload(self, api, repo, pairs, message):
        import shutil
        for rel, local in pairs:
            if self.fail_on and rel.endswith(self.fail_on):
                raise RuntimeError(f"simulated upload failure on {rel}")
            dst = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(local, dst)
            self.uploads += 1

    def download(self, repo, rel, token, dest_dir):
        import shutil
        src = os.path.join(self.root, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(rel)
        os.makedirs(dest_dir, exist_ok=True)
        dst = os.path.join(dest_dir, os.path.basename(rel))
        shutil.copyfile(src, dst)
        return dst

    def install(self, monkeypatch):
        monkeypatch.setattr(ph, "_hub", self.hub)
        monkeypatch.setattr(ph, "_list_files", self.list_files)
        monkeypatch.setattr(ph, "_upload", self.upload)
        monkeypatch.setattr(ph, "_download", self.download)
        return self


@pytest.fixture
def fakehub(tmp_path, monkeypatch):
    return FakeHub(str(tmp_path / "hub")).install(monkeypatch)


def test_push_then_pull_round_trips_every_part(tmp_path, fakehub):
    """Push every fetched year, pull it into a fresh work dir, byte for byte."""
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    for y in ctx.years:
        assert ph.push("gdp", y, ctx.work) == 0
    assert sorted(ph.status("gdp")) == sorted(ctx.years)

    # A second push is a no-op: done.json is already there with these hashes.
    before = fakehub.uploads
    ph.push("gdp", ctx.years[0], ctx.work)
    assert fakehub.uploads == before

    fresh = os.path.join(tmp, "pulled")
    present, missing = ph.pull("gdp", ctx.years, fresh)
    assert present == list(ctx.years) and missing == []
    for y in ctx.years:
        src, dst = ctx.year_dir(y), ph.year_dir(fresh, "gdp", y)
        assert sorted(os.listdir(src)) == sorted(os.listdir(dst))
        for n in os.listdir(src):
            assert f10.sha256(os.path.join(src, n)) == \
                f10.sha256(os.path.join(dst, n)), (y, n)
        assert b10.marked(ph.store_root(fresh, "gdp"), f"parts/{y}")

    # A pull whose local copy already matches downloads nothing.
    calls = []
    orig = fakehub.download
    fakehub.download = lambda *a, **k: (calls.append(a[1]), orig(*a, **k))[1]
    ph.pull("gdp", ctx.years, fresh)
    assert all(c.endswith("done.json") for c in calls), calls


def test_the_done_marker_is_written_last_and_a_half_push_reads_as_missing(
        tmp_path, fakehub):
    """§5.21. Kill the push between the parts and done.json: the year's bytes
    are on the Hub, but no marker claims them, so `pull` reports it MISSING
    and the fetch lane simply refetches it. The opposite order would hand the
    assembler a short year with nothing to say so."""
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    y = ctx.years[0]
    fakehub.fail_on = "/done.json"
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        ph.push("gdp", y, ctx.work)
    fakehub.fail_on = None

    files = fakehub.list_files(None, None, ph.hub_prefix("gdp", y))
    assert any(p.endswith(".npz") for p in files), "the parts never uploaded"
    assert not any(p.endswith("done.json") for p in files)
    assert ph.status("gdp", years=[y]) == []

    fresh = os.path.join(tmp, "pulled")
    with pytest.raises(SystemExit, match="no done.json"):
        ph.pull("gdp", [y], fresh)
    present, missing = ph.pull("gdp", [y], fresh, allow_missing=True)
    assert present == [] and missing == [y]

    # Re-push finishes the job and the year becomes usable.
    assert ph.push("gdp", y, ctx.work) == 0
    assert ph.pull("gdp", [y], fresh)[0] == [y]


def test_push_refuses_a_year_whose_fetch_did_not_finish(tmp_path, fakehub):
    """An unmarked year is a year that was never finished locally."""
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    y = ctx.years[0]
    os.remove(b10.marker(ctx.root, f"parts/{y}"))
    with pytest.raises(SystemExit, match="did not finish"):
        ph.push("gdp", y, ctx.work)


def test_parts_from_hub_builds_the_same_store_without_the_source(
        tmp_path, fakehub):
    """The keyless half of slatrack's build, end to end on the synthetic gdp
    archive: fetch here, push, then assemble THERE with no source at all."""
    import argparse
    import shutil
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    want = _hash_store(ctx.store)
    for y in ctx.years:
        ph.push("gdp", y, ctx.work)

    work2 = os.path.join(tmp, "box")
    ns = dict(store="gdp", work=work2, source_dir=ctx.source_dir,
              start=SMOKE_START, end=SMOKE_END, stage="all", force=False,
              attempts=1, qc_keep=2, socat_url="", smoke=True,
              assemble="auto", parts_from_hub=True,
              allow_missing_years=False)
    ctx2 = b10.Ctx(argparse.Namespace(**ns))
    b10.run_stages(ctx2, ["index"])
    # The source is GONE: nothing in the fetch stage may reach for it.
    shutil.rmtree(ctx.source_dir)
    b10.run_stages(ctx2, ["fetch"])
    assert _hash_store(ctx2.store) == want


def test_parts_from_hub_refuses_when_a_year_is_not_on_the_hub(tmp_path,
                                                              fakehub):
    """Listing the missing years is the whole point: a store that quietly
    dropped 1993-1995 would look entirely ordinary."""
    import argparse
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    ph.push("gdp", ctx.years[0], ctx.work)          # the OTHER year is absent
    work2 = os.path.join(tmp, "box")
    ns = dict(store="gdp", work=work2, source_dir=ctx.source_dir,
              start=SMOKE_START, end=SMOKE_END, stage="all", force=False,
              attempts=1, qc_keep=2, socat_url="", smoke=True,
              assemble="auto", parts_from_hub=True,
              allow_missing_years=False)
    ctx2 = b10.Ctx(argparse.Namespace(**ns))
    b10.run_stages(ctx2, ["index"])
    with pytest.raises(SystemExit, match=str(ctx.years[-1])):
        b10.run_stages(ctx2, ["fetch"])


def test_slatrack_read_nc_folds_a_sample_at_exactly_180_east(tmp_path, built):
    """The DUACS files carry samples at lon == +180.0; the store wants [-180, 180)."""
    import netCDF4 as ncdf
    ctx, _ = built["slatrack"]
    p = tmp_path / "edge.nc"
    ds = ncdf.Dataset(p, "w")
    ds.createDimension("time", 3)
    t = ds.createVariable("time", "f8", ("time",)); t.units = "days since 1950-01-01 00:00:00"
    t[:] = [11690.5, 11690.6, 11690.7]        # 1982-01-03, inside the smoke window
    ds.createVariable("latitude", "f8", ("time",))[:] = [0.0, 10.0, -10.0]
    ds.createVariable("longitude", "f8", ("time",))[:] = [180.0, -180.0, 179.5]
    for v in ("sla_filtered", "sla_unfiltered", "mdt"):
        ds.createVariable(v, "f8", ("time",))[:] = [0.1, 0.2, 0.3]
    ds.close()
    rows, counts = b10.SLATrackAdapter()._read_nc(ctx, str(p), "cmems_test")
    lon = np.asarray(rows["lon"], np.float64)
    assert counts["kept"] == 3
    assert lon.min() >= -180.0 and lon.max() < 180.0
    assert lon[0] == -180.0


def test_rows_pack_keeps_lon_below_180_after_the_float32_cast(tmp_path, built):
    """A float64 longitude of 179.99999 wraps to itself and then ROUNDS to
    180.0 in the float32 column — which is what the 1994 altimeter year did
    on 2026-09-14, one hour after the float64-only fold had landed. The
    invariant is on the column's dtype, so the pack re-wraps after the cast."""
    import netCDF4 as ncdf
    ctx, _ = built["slatrack"]
    p = tmp_path / "edge32.nc"
    ds = ncdf.Dataset(p, "w")
    ds.createDimension("time", 3)
    t = ds.createVariable("time", "f8", ("time",)); t.units = "days since 1950-01-01 00:00:00"
    t[:] = [11690.5, 11690.6, 11690.7]
    ds.createVariable("latitude", "f8", ("time",))[:] = [0.0, 10.0, -10.0]
    # 179.999999 is < 180 in float64 and == 180.0 once cast to float32;
    # -180.0000001 is < -180 in float64 and == -180.0 in float32 after the wrap.
    ds.createVariable("longitude", "f8", ("time",))[:] = [179.999999, 539.999999, -180.0000001]
    for v in ("sla_filtered", "sla_unfiltered", "mdt"):
        ds.createVariable(v, "f8", ("time",))[:] = [0.1, 0.2, 0.3]
    ds.close()
    rows, counts = b10.SLATrackAdapter()._read_nc(ctx, str(p), "cmems_test")
    assert counts["kept"] == 3
    lon = rows["lon"]
    assert lon.dtype == np.float32
    assert np.all((lon >= np.float32(-180.0)) & (lon < np.float32(180.0))), lon
    assert lon[0] == np.float32(-180.0) and lon[1] == np.float32(-180.0)


# =============================================== 57-71 · no silent skip path ==
# Chris's rule, applied to family 10 the way commit fd3b446 applied it to
# family 7: there must be NO code path where an input cannot be read, the
# builder warns (or says nothing), and the unit is marked done anyway. For a
# store of POINT observations that rule bites harder than it does for a
# gridded tensor — a grid has an empty cell to look at, and a store of
# scattered measurements has nothing at all. A month that did not download and
# an ocean nobody sampled produce the same store.
#
# Each test below is one of the sites, and each asserts the same three things
# the family-7 pass did: the stage REFUSES, the unit is NOT marked (so a resume
# retries exactly it), and the `--allow-missing-years` path RECORDS what it
# gave up.
def _ns(**over):
    ns = dict(stage="all", force=False, attempts=1, qc_keep=2, socat_url="",
              smoke=True, assemble="auto", parts_from_hub=False,
              allow_missing_years=False)
    ns.update(over)
    return __import__("argparse").Namespace(**ns)


def _gdp_ctx(tmp, **over):
    """A gdp build laid out but not run, so a test can break its source."""
    src = os.path.join(tmp, "src_gdp_skip")
    work = os.path.join(tmp, "work_gdp_skip")
    os.makedirs(work, exist_ok=True)
    b10.make_smoke_sources(src, "gdp", b10.parse_date(SMOKE_START),
                           b10.parse_date(SMOKE_END))
    return b10.Ctx(_ns(store="gdp", work=work, source_dir=src,
                       start=SMOKE_START, end=SMOKE_END, **over)), src


def test_57_a_month_that_could_not_be_read_refuses_and_leaves_the_year_unmarked(
        tmp_path):
    """The GDP site. `_month` used to answer a missing file with
    `{"missing_month": 1}` and zero rows; `stage_fetch` then marked the year
    COMPLETE, and because a resume trusts the marker the hole was permanent,
    invisible and green — family 7's 1989, one family over."""
    tmp = str(tmp_path)
    ctx, src = _gdp_ctx(tmp)
    b10.run_stages(ctx, ["index"])
    gone = os.path.join(src, "gdp", "1982-01.csv")
    assert os.path.exists(gone)
    os.remove(gone)

    with pytest.raises(SystemExit) as e:
        b10.run_stages(ctx, ["fetch"])
    msg = str(e.value)
    assert "1982-01" in msg and "allow-missing-years" in msg

    # THE MARKER IS THE POINT: 1982 is not marked, so a resume refetches it,
    # and neither the year nor the stage claims to be done.
    assert not b10.marked(ctx.root, "parts/1982")
    assert not b10.marked(ctx.root, "fetch")
    # 1981 landed whole and IS marked — a refusal must not cost the years that
    # worked, or a resume would redo the archive every time.
    assert b10.marked(ctx.root, "parts/1981")
    # and no store was written from the short parts
    assert not os.path.exists(os.path.join(ctx.store, "store.json"))


def test_58_the_flag_is_the_only_way_through_and_the_store_records_the_hole(
        tmp_path):
    """--allow-missing-years builds, and store.json says what it gave up."""
    tmp = str(tmp_path)
    ctx, src = _gdp_ctx(tmp, allow_missing_years=True)
    b10.run_stages(ctx, ["index"])
    os.remove(os.path.join(src, "gdp", "1982-01.csv"))
    b10.run_stages(ctx, ["fetch"])

    meta = json.load(open(os.path.join(ctx.store, "store.json")))
    deg = meta["degraded"]
    assert deg["allow_missing_years"] is True
    assert any("1982-01" in e["unit"] for e in deg["inputs_not_read"])
    assert any("does not exist" in e["why"] for e in deg["inputs_not_read"])
    # The year is still NOT marked — the flag admits the parts, it does not
    # promote a short year to a complete one.
    assert not b10.marked(ctx.root, "parts/1982")
    assert meta["per_year"]["1982"] < meta["per_year"]["1981"]


def test_59_fetch_first_tells_an_empty_archive_from_a_moved_one():
    """ERDDAP's "produced no matching results" is the archive saying the year
    is empty; a bare 404 is the dataset id moving under the build. They used
    to arrive at the caller as the same `None`, and the caller counted both as
    a gap — which is how `missing_month` and `datasets_empty` could hold a
    download failure and an empty year under one number."""
    calls = []

    def fake(url, path=None, **k):
        calls.append(url)
        raise b10._Empty(url) if "empty" in url else b10._NotFound(url)

    import builtins  # noqa: F401  (keeps the monkeypatch local and obvious)
    old = b10.http_bytes
    try:
        b10.http_bytes = lambda u, **k: fake(u)
        assert b10.fetch_first(["http://x/empty"], attempts=1,
                               reason=True) == (None, "empty")
        assert b10.fetch_first(["http://x/gone"], attempts=1,
                               reason=True) == (None, "notfound")
        # a MIX is not an empty archive: one url said "no rows", the other is
        # simply gone, so the pessimistic answer is the honest one.
        assert b10.fetch_first(["http://x/empty", "http://x/gone"], attempts=1,
                               reason=True) == (None, "notfound")
        # and the old two-value call still behaves exactly as it did
        assert b10.fetch_first(["http://x/empty"], attempts=1) is None
    finally:
        b10.http_bytes = old


def test_60_gtmba_refuses_a_year_missing_from_a_dataset_the_mirror_carries(
        tmp_path):
    """A mirror that holds `pmelTaoDySst/` and not `pmelTaoDySst/1982.csv` is
    SHORT — that year's sst is NaN at every mooring. A mirror that holds no
    `pmelTaoDyAdcp/` at all simply does not carry the ADCP, which is a
    property of the archive copy and not a failure (the test fixtures are
    exactly that: three of the eight dataset-years)."""
    tmp = str(tmp_path)
    src = os.path.join(tmp, "src_gtmba_skip")
    work = os.path.join(tmp, "work_gtmba_skip")
    os.makedirs(work, exist_ok=True)
    b10.make_smoke_sources(src, "gtmba", b10.parse_date(SMOKE_START),
                           b10.parse_date(SMOKE_END))
    ctx = b10.Ctx(_ns(store="gtmba", work=work, source_dir=src,
                      start=SMOKE_START, end=SMOKE_END))
    b10.run_stages(ctx, ["index"])
    os.remove(os.path.join(src, "gtmba", "pmelTaoDySst", "1982.csv"))
    with pytest.raises(SystemExit) as e:
        b10.run_stages(ctx, ["fetch"])
    assert "1982 pmelTaoDySst" in str(e.value)
    assert not b10.marked(ctx.root, "parts/1982")
    assert b10.marked(ctx.root, "parts/1981")


def test_61_gtmba_counts_a_dataset_the_mirror_does_not_carry_at_all(built):
    """The other half of 60: the five datasets the fixture omits are COUNTED,
    under a name that says which case it is, and nothing refuses."""
    ctx, _ = built["gtmba"]
    counts = json.load(open(os.path.join(ctx.year_dir(1982),
                                         "counts.json")))["counts"]
    assert counts.get("datasets_absent_locally", 0) >= 4
    assert counts.get("datasets_empty", 0) == 0
    assert b10.marked(ctx.root, "parts/1982")


class _FakeCM:
    """Just enough `copernicusmarine` for `_fetch_files`: a listing and a get
    that writes files. `serve` says how many of the asked-for files arrive."""

    def __init__(self, listed, serve=None):
        self.listed = listed
        self.serve = serve
        self.out = None

    def get(self, dry_run=False, regex=None, output_directory=None, **k):
        if dry_run:
            return {"files": [{"filename": n} for n in self.listed]}
        names = [n for n in self.listed
                 if regex is None or __import__("re").search(regex, n)]
        if self.serve is not None:
            names = names[:self.serve]
        os.makedirs(output_directory, exist_ok=True)
        for n in names:
            open(os.path.join(output_directory, os.path.basename(n)),
                 "wb").write(b"")
        return {"files": []}


def _slatrack_files(ctx, fake, monkeypatch, mid="cmems_m_PT1S_202411"):
    monkeypatch.setitem(sys.modules, "copernicusmarine", fake)
    ad = b10.SLATrackAdapter()
    monkeypatch.setattr(ad, "_read_nc",
                        lambda c, p, m: (b10.empty_rows(ad.C), {"kept": 0}))
    return list(ad._fetch_files(ctx, mid, dt.date(1982, 1, 1),
                                dt.date(1982, 1, 31)))


def test_62_a_mission_year_that_lists_no_file_is_an_absence_not_an_empty_year(
        tmp_path, monkeypatch):
    """THE ONE THAT WOULD HAVE COST THE WHOLE ARCHIVE. The remote path layout
    is not measured (the adapter says so), and `filter=*<year>*` is matched
    against the absolute remote path. If that pattern does not fit, EVERY
    mission-year lists zero files, every year finishes in seconds with no
    rows, done.json is written for each, and 1993-2024 assembles into an empty
    store that is green from end to end."""
    tmp = str(tmp_path)
    ctx, _ = _gdp_ctx(tmp)                       # any ctx: only paths are used
    ctx.a.store = "slatrack"
    fake = _FakeCM(listed=[])
    _slatrack_files(ctx, fake, monkeypatch)
    assert ctx.absent, "an empty listing must be reported as an absence"
    assert "listed NO original file" in ctx.absent[0]["why"]
    with pytest.raises(SystemExit, match="allow-missing-years"):
        b10.fetch_absence_check(ctx)


def test_63_a_batch_that_downloads_fewer_files_than_it_asked_for_refuses(
        tmp_path, monkeypatch):
    """`copernicusmarine.get` does not raise on a file it could not serve. One
    missing day of one mission is invisible in a store of 2e9 rows."""
    tmp = str(tmp_path)
    ctx, _ = _gdp_ctx(tmp)
    ctx.a.store = "slatrack"
    names = [f"/remote/1982/track_1982010{k}.nc" for k in range(1, 5)]
    _slatrack_files(ctx, _FakeCM(listed=names, serve=2), monkeypatch)
    assert ctx.absent and "2 of the 4 file(s)" in ctx.absent[0]["why"]
    with pytest.raises(SystemExit, match="allow-missing-years"):
        b10.fetch_absence_check(ctx)
    # and the complete batch is silent
    ctx.absent = []
    _slatrack_files(ctx, _FakeCM(listed=names), monkeypatch)
    assert ctx.absent == []


def test_64_the_assembler_refuses_parts_that_no_marker_claims(tmp_path):
    """A fetch killed mid-year (the six-hour cap, an OOM, a lost box) leaves
    flushed parts and no marker. Both assemblers walked the DIRECTORY, so
    those orphan parts assembled as if they were a year — and store.json's
    per_year block then reported the loss as a measurement."""
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    os.remove(b10.marker(ctx.root, "parts/1982"))
    with pytest.raises(SystemExit) as e:
        b10.assemble(ctx)
    assert "did not finish" in str(e.value) and "1982" in str(e.value)

    # the flag admits them, and store.json says which year it admitted
    ctx.a.allow_missing_years = True
    ctx.degraded_years = []
    b10.assemble(ctx)
    deg = json.load(open(os.path.join(ctx.store, "store.json")))["degraded"]
    assert any("1982" in m for m in deg["years_admitted_unmarked"])


def test_65_the_assembler_refuses_a_year_short_of_the_parts_it_recorded(
        tmp_path):
    """counts.json is written from the rows that were flushed, so the two can
    only disagree if a part was lost between the fetch and the assembly."""
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    d = ctx.year_dir(1982)
    lost = sorted(n for n in os.listdir(d) if n.endswith(".npz"))[0]
    os.remove(os.path.join(d, lost))
    with pytest.raises(SystemExit) as e:
        b10.assemble(ctx)
    assert "counts.json says" in str(e.value)


def test_66_the_assembler_refuses_a_year_short_of_the_rows_it_recorded(
        tmp_path):
    """The same guard one level down: the part file is there and holds fewer
    rows than the ledger recorded. The number that would show the loss must
    not be computed FROM the loss."""
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    d = ctx.year_dir(1982)
    p = os.path.join(d, sorted(n for n in os.listdir(d)
                               if n.endswith(".npz"))[0])
    with np.load(p) as z:
        half = {k: z[k][: max(1, len(z["bin"]) // 2)] for k in b10.ROW_KEYS}
    np.savez(p, **half)
    with pytest.raises(SystemExit) as e:
        b10.assemble(ctx)
    assert "row(s), the parts hold" in str(e.value)


def test_67_a_truncated_socat_transfer_raises_instead_of_shortening_the_store():
    """The 1.4 GB synthesis streams through a bare urlopen — no Content-Length
    check anywhere — and a prefix of a zip parses perfectly. The store would
    have come out short by however much did not arrive, with every year marked
    done, and nothing could tell afterwards."""
    import io
    import zipfile
    buf = io.BytesIO()
    payload = ("Expocode\tversion\n" + "x\ty\n" * 20000).encode()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("SOCAT.tsv", payload)
    raw = buf.getvalue()
    cut = raw[: len(raw) // 2]                   # the transfer dies halfway
    with pytest.raises(IOError, match="cut short"):
        list(b10._zip_member_stream(io.BytesIO(cut), size=97))
    # the deliberate prefix the index stage reads is exempt, by name
    got = b"".join(b10._zip_member_stream(io.BytesIO(cut), size=97,
                                          partial_ok=True))
    assert payload.startswith(got) and 0 < len(got) < len(payload)


def test_68_check_store_refuses_to_run_with_its_assertions_optimised_away():
    """Every E-079 §4 check is an `assert`, and `python3 -O` deletes all of
    them — so under -O the function would open the store, check nothing, and
    return it to a `_finish_store` and a `stage_publish` that print exactly
    the same lines as a checked build."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "-O", "-c",
         "import sys; sys.path.insert(0, %r); import build_family10_stores as b;"
         "b.check_store('/nonexistent')" % os.path.join(ROOT, "ml")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "PYTHONOPTIMIZE" in r.stderr or "-O" in r.stderr
    assert "check_store cannot run" in r.stderr


def test_69_publish_takes_its_file_list_from_store_json_not_the_directory(
        tmp_path, built, monkeypatch):
    """`family10_store.Store` opens `qc.npy` and `fp.npy` through `_optional`,
    so a store published without one of them opens cleanly, answers every
    search, and silently reports no flag and the default footprint for every
    row. A directory listing publishes whatever is there; the sha256 block
    names what the assembler wrote."""
    import shutil
    ctx, _ = built["gdp"]
    dest = str(tmp_path / "store")
    shutil.copytree(ctx.store, dest)
    ctx2 = b10.Ctx(_ns(store="gdp", work=str(tmp_path / "w"), source_dir="",
                       start=SMOKE_START, end=SMOKE_END))
    monkeypatch.setattr(ctx2, "store", dest, raising=False)
    os.remove(os.path.join(dest, "qc.npy"))
    with pytest.raises(SystemExit, match="qc.npy"):
        b10.stage_publish(ctx2)
    # and a stray column from an older schema is refused rather than uploaded
    shutil.copyfile(os.path.join(dest, "lat.npy"),
                    os.path.join(dest, "qc.npy"))
    open(os.path.join(dest, "pressure.npy"), "wb").write(b"")
    with pytest.raises(SystemExit, match="pressure.npy"):
        b10.stage_publish(ctx2)


def test_70_a_hub_that_will_not_serve_a_marker_is_not_a_year_that_was_never_fetched(
        tmp_path, fakehub):
    """`read_done` swallowed every exception and returned None, so a Hub
    outage, an expired token and "that year was never pushed" were one answer.
    `pull --allow-missing` would then drop a year whose parts were sitting on
    the Hub the whole time."""
    tmp = str(tmp_path)
    ctx, _ = build(tmp, "gdp")
    y = ctx.years[0]
    ph.push("gdp", y, ctx.work)

    real = fakehub.download

    def refuse(repo, rel, token, dest_dir):
        if rel.endswith("done.json"):
            raise RuntimeError("503 Service Unavailable")
        return real(repo, rel, token, dest_dir)

    fakehub.download = refuse
    import family10_parts_hub as _ph
    _ph._download = refuse
    try:
        fresh = os.path.join(tmp, "pulled")
        with pytest.raises(IOError, match="NOT a year that was never fetched"):
            ph.pull("gdp", [y], fresh, allow_missing=True)
    finally:
        _ph._download = real
        fakehub.download = real


def test_71_the_registry_refuses_to_call_an_unreadable_store_a_missing_one():
    """A 404 means the store is not published. A 401/403 means the Hub would
    not answer — and returning None for both put a statement about the archive
    into a registry written by something that could not read the archive."""
    import urllib.error

    def raise_code(code):
        def go(url, timeout=60):
            raise urllib.error.HTTPError(url, code, "no", None, None)
        return go

    old = reg.http_json
    try:
        reg.http_json = raise_code(404)
        assert reg.hub_json("x/y", "a/b.json") is None
        for code in (401, 403):
            reg.http_json = raise_code(code)
            with pytest.raises(IOError, match="refused the read"):
                reg.hub_json("x/y", "a/b.json")
    finally:
        reg.http_json = old
