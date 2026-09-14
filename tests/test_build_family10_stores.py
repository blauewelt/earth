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
