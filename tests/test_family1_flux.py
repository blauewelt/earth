#!/usr/bin/env python3
"""The `flux` adapter's smoke (family 1.0.tf, FLUXNET Shuttle) — no network.

    python3 -m pytest -q tests/test_family1_flux.py

The synthetic archive (`flux.make_smoke_sources`) is the real layout: a hub
listing plus one real zip per site whose members are named the way the
product names them — `*_FLUXNET_FLUXMET_HH_*.csv`, `*_FLUXNET_BIF_*.csv`,
`*_FLUXNET_BIFVARINFO_*` (which must NOT be mistaken for the BADM file) —
and whose half-hourly CSV carries the FULLSET column names. Three sites on
three hubs and three UTC offsets (-5, +1, +10), one of them missing NETRAD;
one row carries the product's -9999 and one an out-of-bound temperature.

What the tests pin is the reasoning: the local-standard-time conversion, the
refusal when a site has no UTC_OFFSET, the refusal when a hub serves HTML
instead of a zip, and the qc packing.
"""
import datetime as dt
import os
import sys
import zipfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import flux as fx                         # noqa: E402

ENV = ("FLUX_HUBS",)


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("flux_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("flux", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Ctx:
    """The three things the listing and the archive ask a context for."""

    def __init__(self, root):
        self.source_dir = root

        class _A:
            attempts = 1
        self.a = _A()

    def count_bytes(self, n):
        return None


# ================================================================ facts ====
def test_registered_and_needs_no_credential(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["flux"] is fx.FluxAdapter
    ad = fx.FluxAdapter()
    assert not b1.is_grid(ad)
    assert ad.family == "1tf" and ad.distribution == "public"
    # MEASURED: all three hubs answer anonymously (see the adapter's
    # `verified`), so the plan's FLUXNET_USERNAME / PASSWORD are not declared
    assert ad.credentials == ()
    assert ad.licence["name"] == "CC BY 4.0"
    assert ad.channel_names == ["nee", "gpp", "reco", "le", "h", "rn", "ta",
                                "vpd", "swc", "p"]
    assert ad.C == 10 and ad.platform_meta is True
    assert ad.per_year is False
    assert ad.first_year == 1991
    assert ad.log2_fp == -4.0                      # a moving flux footprint
    assert abs(ad.log2_dt - (-7.907)) < 1e-3       # a half hour of five days
    assert list(ad.hubs) == ["amf", "icos", "tern"]


def test_the_three_hub_endpoints_are_the_ones_that_were_measured():
    assert fx.AMF_SITES.endswith("/api/v2/site_info_display/AmeriFlux")
    assert fx.AMF_SHUTTLE.endswith("/amf_shuttle_data_files_and_manifest")
    assert fx.AMF_USER_ID == "fluxnetshuttle"
    assert fx.ICOS_SPARQL == "https://meta.icos-cp.eu/sparql"
    assert "licence_accept" in fx.ICOS_ACCEPT
    assert fx.TERN_CATALOGUE.endswith("TERN_THREDDS_catalogue.csv")
    # the SPARQL query really asks for the FLUXNET archive product spec
    q = fx.icos_query()
    assert fx.ICOS_SPEC in q and "hasSizeInBytes" in q
    assert "isNextVersionOf" in q                 # only the newest version


def test_hubs_env_refuses_an_unknown_hub(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("FLUX_HUBS", "ameriflux")
    with pytest.raises(SystemExit) as e:
        fx.FluxAdapter()
    assert "the hubs are" in str(e.value)
    monkeypatch.setenv("FLUX_HUBS", "icos")
    assert list(fx.FluxAdapter().hubs) == ["icos"]


def test_local_standard_time_is_converted_with_the_sites_own_offset():
    """The whole reason the BADM file is read: 2018-06-01 00:00 local at a
    site on -5 is 05:00 UTC, and on +10 it is the previous day at 14:00."""
    raw = fx._hh_csv("US-Smk", -5, None).encode()
    cols, _ = fx.parse_hh(raw, "US-Smk", "amf", -5.0, {})
    t0 = int(cols["t"][0])
    assert (dt.datetime(1982, 1, 1) + dt.timedelta(seconds=t0)) == \
        dt.datetime(2018, 6, 1, 5, 0)
    cols, _ = fx.parse_hh(raw, "AU-Smk", "tern", 10.0, {})
    t0 = int(cols["t"][0])
    assert (dt.datetime(1982, 1, 1) + dt.timedelta(seconds=t0)) == \
        dt.datetime(2018, 5, 31, 14, 0)
    # the rows are half an hour apart and the time is the START of the slot
    d = np.diff(cols["t"][:4])
    assert (d == 1800).all()


def test_a_site_without_a_utc_offset_is_refused_not_guessed():
    bad = fx._bif_csv("ZZ-Bad", None, 0.0, 0.0, 0.0, "WSA", "no_offset")
    with pytest.raises(fx.FormatError) as e:
        fx.read_bif(bad.encode(), "ZZ-Bad")
    assert "UTC_OFFSET" in str(e.value)
    assert "longitude" in str(e.value)
    good = fx._bif_csv("US-Smk", -5, 42.5, -72.2, 340.0, "DBF", None)
    off, meta = fx.read_bif(good.encode(), "US-Smk")
    assert off == -5.0 and meta["IGBP"] == "DBF"
    # an offset outside -14..14 is a refusal too
    with pytest.raises(fx.FormatError):
        fx.read_bif(fx._bif_csv("X", 99, 0, 0, 0, "WSA", None).encode(), "X")


def test_an_absent_column_leaves_its_channel_nan_and_is_counted():
    raw = fx._hh_csv("IT-Smk", 1, "no_netrad").encode()
    cols, counts = fx.parse_hh(raw, "IT-Smk", "icos", 1.0, {})
    i = [c for (c, _s) in fx.COLUMNS].index("rn")
    assert np.isnan(cols["values"][:, i]).all()
    assert counts["channels_absent"] == {"rn": 1}
    # every other channel is there
    j = [c for (c, _s) in fx.COLUMNS].index("ta")
    assert np.isfinite(cols["values"][:, j]).any()


def test_the_missing_value_becomes_nan_and_the_qc_is_packed():
    raw = fx._hh_csv("US-Smk", -5, None).encode()
    cols, _ = fx.parse_hh(raw, "US-Smk", "amf", -5.0, {})
    nee = cols["values"][:, 0]
    assert np.isnan(nee).any()                  # the -9999 row
    q = cols["qc"]
    assert set(int(x) for x in np.unique(q & 0b111)) <= {0, 1, 2, 3}
    assert set(int(x) for x in np.unique((q >> 3) & 0b111)) <= {0, 1, 2, 3}
    assert ((q >> 6) & 1 == 0).all()            # a half-hourly site
    # an HOURLY site sets bit 6, and nothing else moves
    cols2, _ = fx.parse_hh(raw, "AU-Otw", "tern", 10.0, {}, hourly=True)
    assert ((cols2["qc"] >> 6) & 1 == 1).all()
    assert ((cols2["qc"] & 0b111) == (q & 0b111)).all()
    # the HUB is NOT in qc: it is a property of the site, not of the row
    cols3, _ = fx.parse_hh(raw, "IT-Smk", "icos", 1.0, {})
    assert (cols3["qc"] == q).all()


def test_an_hourly_site_is_read_and_marked_not_dropped():
    """TERN's AU-Otw publishes `_FLUXMET_HR_` and no HH file at all."""
    import io as _io
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("TERN_AU-Otw_FLUXNET_FLUXMET_HR_2007-2010_v1.3_r1.csv",
                   fx._hh_csv("AU-Otw", 10, None))
        z.writestr("TERN_AU-Otw_FLUXNET_FLUXMET_DD_2007-2010_v1.3_r1.csv",
                   "TIMESTAMP,TA_F\n20180601,12.0\n")
        z.writestr("TERN_AU-Otw_FLUXNET_BIF_2007-2010_v1.3_r1.csv",
                   fx._bif_csv("AU-Otw", 10, -38.5, 142.8, 90.0, "GRA", None))
    cols, off, meta, counts = fx.read_zip(buf.getvalue(), "AU-Otw", "tern",
                                          {})
    assert off == 10.0
    assert meta["_resolution"] == "HR"
    assert counts["sites_hourly"] == 1
    assert ((cols["qc"] >> 6) & 1 == 1).all()
    # two FLUXMET averaging files in one zip is still a refusal
    buf2 = _io.BytesIO()
    with zipfile.ZipFile(buf2, "w") as z:
        z.writestr("X_A-B_FLUXNET_FLUXMET_HR_2018-2018_v1.3_r1.csv",
                   fx._hh_csv("A-B", 0, None))
        z.writestr("X_A-B_FLUXNET_FLUXMET_HH_2018-2018_v1.3_r1.csv",
                   fx._hh_csv("A-B", 0, None))
        z.writestr("X_A-B_FLUXNET_BIF_2018-2018_v1.3_r1.csv",
                   fx._bif_csv("A-B", 0, 0, 0, 0, "GRA", None))
    with pytest.raises(fx.FormatError) as e:
        fx.read_zip(buf2.getvalue(), "A-B", "amf", {})
    assert "expected exactly one" in str(e.value)


def test_a_zip_that_is_not_a_zip_and_a_zip_with_no_badm_are_refusals():
    with pytest.raises(fx.FormatError) as e:
        fx.read_zip(b"<html>accept the licence</html>", "IT-Smk", "icos", {})
    assert "not a zip" in str(e.value)
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("XX_A-B_FLUXNET_FLUXMET_HH_2018-2018_v1.3_r1.csv",
                   fx._hh_csv("A-B", 0, None))
        # BIFVARINFO is NOT the BADM file, and must not be taken for it
        z.writestr("XX_A-B_FLUXNET_BIFVARINFO_HH_2018-2018_v1.3_r1.csv", "x\n")
    with pytest.raises(fx.FormatError) as e:
        fx.read_zip(buf.getvalue(), "A-B", "amf", {})
    assert "no BADM" in str(e.value)


def test_the_tern_csv_parser_handles_a_quoted_citation():
    line = ('AU-Ade,https://x/y.zip,https://doi/1,"Beringer, J., Hutley, L. '
            '(2026): Adelaide River"')
    f = fx._split_csv(line)
    assert len(f) == 4
    assert f[3].startswith("Beringer, J.")


def test_the_year_split_is_the_calendar_year_of_the_utc_second():
    """One part per calendar year; the vectorised year must agree with the
    calendar at both ends of a year and before 1982."""
    for d in (dt.datetime(1991, 1, 1), dt.datetime(1999, 12, 31, 23, 30),
              dt.datetime(2000, 1, 1), dt.datetime(2018, 6, 1, 5, 0),
              dt.datetime(1981, 12, 31, 23, 30), dt.datetime(2026, 2, 28)):
        t = b10.seconds_since_epoch(d)
        got = int(fx.FluxAdapter._years_of(np.array([t], np.int64))[0])
        assert got == d.year, (d, got)


# ============================================================== the smoke ==
def test_smoke_stores_every_half_hour_of_every_site(smoke):
    truth = smoke["truth"]
    # three sites x two months x two days x 48 half hours
    assert len(truth) == 3 * 2 * 2 * 48
    sm_dir = os.path.join(smoke["work"], "flux", "flux")
    import family10_store as f10
    st = f10.open_store(sm_dir)
    assert st.N == len(truth)
    assert st.C == 10


def test_platforms_json_carries_the_site_metadata(smoke):
    import json
    p = os.path.join(smoke["work"], "flux", "flux", "platforms.json")
    assert os.path.exists(p)
    d = json.load(open(p))
    assert len(d) == 3
    by_id = {v["id"]: v for v in d.values()}
    assert sorted(by_id) == ["AU-Smk", "IT-Smk", "US-Smk"]
    us = by_id["US-Smk"]
    assert us["igbp"] == "DBF" and us["hub"] == "amf"
    assert us["elev_m"] == 340.0 and us["hub_code"] == 1
    # the cadence and the offset the archive itself declared
    assert us["resolution"] == "HH" and us["utc_offset"] == -5.0
    assert by_id["AU-Smk"]["hub"] == "tern"
    assert by_id["IT-Smk"]["hub"] == "icos"


def test_the_probe_measured_the_month(smoke):
    p = smoke["probe"]
    assert p["rows"] == 3 * 2 * 48          # June only
    assert p["counts_scope"] == "month"
    assert p["bytes_fetched"] > 0
    assert p["platforms"]["probe_platforms_without_entry"] == 0
    c = p["counts"]
    assert c["sites_read"] == 3
    assert c["channels_absent"] == {"rn": 1}
    assert c["out_of_bounds"]["ta"] > 0     # the 999 degree row


def test_a_site_whose_archive_will_not_read_is_an_absence(tmp_path,
                                                          monkeypatch):
    """A listed site whose BADM file has no UTC_OFFSET stops the pass."""
    clear_env(monkeypatch)
    root = str(tmp_path)
    fx.make_smoke_sources(root, dt.date(2018, 6, 1), dt.date(2018, 7, 31),
                          extra=(fx.SMOKE_BAD_SITE,))
    ad = fx.FluxAdapter()
    ctx = _Ctx(root)
    sites, _ = ad.sites(ctx)
    assert "ZZ-Bad" in sites
    raw, why = ad.archive(ctx, sites["ZZ-Bad"])
    assert raw is not None and why is None
    with pytest.raises(fx.FormatError) as e:
        fx.read_zip(raw, "ZZ-Bad", "amf", {})
    assert "UTC_OFFSET" in str(e.value)


def test_an_empty_hub_listing_is_a_refusal(tmp_path, monkeypatch):
    clear_env(monkeypatch)
    import json
    root = str(tmp_path)
    base = os.path.join(root, "flux")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "sites.json"), "w") as fh:
        json.dump({"sites": {}, "counts": {}}, fh)
    ad = fx.FluxAdapter()
    with pytest.raises(SystemExit) as e:
        ad.sites(_Ctx(root))
    assert "no site at all" in str(e.value)
