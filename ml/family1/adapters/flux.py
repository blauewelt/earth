"""FLUXNET Shuttle — half-hourly land–air exchange of CO2, water and heat
from eddy-covariance towers (family 1.0.tf, tier P, E-082 wave 4).

PLAIN ENGLISH. A flux tower is a mast that measures, ten times a second, the
air's vertical velocity and its CO2 and water content, and multiplies them
together to get how much carbon and water the ecosystem under it is
exchanging with the atmosphere. Averaged to half-hours, that is the only
DIRECT measurement of the land carbon sink there is: every satellite estimate
of photosynthesis is calibrated against these towers. FLUXNET is the
coalition that standardises them; the Shuttle (2026-04-30 release) is how the
processed product is published.

THE DOWNLOAD NEEDS NO LOGIN, AND THAT IS MEASURED, NOT ASSUMED. The plan
allowed for FLUXNET_USERNAME / FLUXNET_PASSWORD. They are not needed: the
Shuttle is a Python library (github.com/fluxnet/shuttle) over THREE PUBLIC
DATA HUBS, and all three answered this sandbox anonymously on 2026-09-18. The
library's own `config.yaml` names them, and its AmeriFlux plugin posts the
fixed literal `user_id: "fluxnetshuttle"` — there is no credential anywhere
in the code path. So `credentials = ()` and the adapter calls the three hubs
directly rather than depending on a `pip install git+https://...` of a
library that is not on PyPI:

  AmeriFlux (the Americas; 844 sites listed, 407 publishing FLUXNET)
    GET  https://amfcdn.lbl.gov/api/v2/site_info_display/AmeriFlux
         -> 1.69 MB of JSON; per site `site_id`, `grp_location`
         (lat / lon / elev), `grp_igbp`, `grp_publish_fluxnet` (the years),
         `doi`, `grp_team_member`
    POST https://amfcdn.lbl.gov/api/v2/amf_shuttle_data_files_and_manifest
         {"user_id": "fluxnetshuttle", "data_product": "FLUXNET",
          "data_variant": "FULLSET", "site_ids": [...]}
         -> `data_urls`: per site a `ftp.fluxdata.org` zip URL with its
         `download_size` and `download_checksum` (MD5), plus a `manifest`
         listing every file inside. Measured: AMF_US-Ha1_FLUXNET_1991-2025
         is 163,629,147 bytes, AMF_AR-Bal_FLUXNET_2012-2013 22,077,723, and
         the MD5 of the downloaded AR-Bal zip matched the one the API gave.
  ICOS (Europe and beyond; 352 FLUXNET archive products)
    POST https://meta.icos-cp.eu/sparql with the Shuttle's own query for
         `cpmeta:miscFluxnetArchiveProduct` -> 8.4 MB of JSON, 3,012
         bindings over 352 data objects, each with `fileName`, `size`,
         `station`, `lat`, `lon`, `ecosystemType` (an IGBP URI) and a
         citation. The download is a TWO-STEP LICENCE ACCEPT:
         GET https://data.icos-cp.eu/licence_accept?ids=%5B%22<id>%22%5D
         sets a `CpLicenseAcceptedFor` cookie, after which
         GET https://data.icos-cp.eu/objects/<id> serves the zip
         (measured: 206, `application/zip`, `PK\\x03\\x04`, 29,093,732
         bytes). Without the cookie both URLs return the licence page as
         HTML, which is why the adapter keeps a cookie jar and REFUSES a
         body that is not a zip.
  TERN (Australia; 53 sites)
    GET  https://dap.tern.org.au/thredds/fileServer/ecosystem_process/
         fluxnet/TERN_THREDDS_catalogue.csv  -> SITE_ID, PRODUCT_URL,
         PRODUCT_ID, PRODUCT_CITATION
    GET  .../fluxnet/BIF_all_sites.csv -> the BADM metadata for all of them

WHAT IS STORED: C = 10, one row per site half-hour, from the FULLSET
half-hourly file inside each site's zip (`*_FLUXNET_FLUXMET_HH_*.csv`, or
`*_FLUXNET_FULLSET_HH_*.csv` at the sites AmeriFlux still publishes under the
FLUXNET2015-era naming — the columns are identical and both are read):

  nee   umolCO2/m2/s  NEE_VUT_REF        net ecosystem exchange
  gpp   umolCO2/m2/s  GPP_NT_VUT_REF     gross primary production
  reco  umolCO2/m2/s  RECO_NT_VUT_REF    ecosystem respiration
  le    W/m2          LE_F_MDS           latent heat (evaporation)
  h     W/m2          H_F_MDS            sensible heat
  rn    W/m2          NETRAD             net radiation
  ta    degC          TA_F               air temperature
  vpd   hPa           VPD_F              vapour-pressure deficit
  swc   percent       SWC_F_MDS_1        soil water content, shallowest slot
  p     mm            P_F                precipitation in the half hour

Columns are taken BY NAME. A column a site's file does not carry — NETRAD
and SWC_F_MDS_1 are the usual ones, because not every tower has a net
radiometer or a soil probe — leaves that channel NaN for that site and is
counted (`channels_absent`, one tally per channel). -9999 is the product's
missing value and
becomes NaN. A value outside its physical bound becomes NaN and is counted,
never clipped.

THE TIMESTAMPS ARE LOCAL STANDARD TIME AND ARE CONVERTED WITH THE SITE'S OWN
OFFSET. `TIMESTAMP_START` is `YYYYMMDDHHMM` in LOCAL STANDARD TIME (no
daylight saving) — the FLUXNET convention — and the store's time axis is UTC
seconds since 1982-01-01. The offset is not guessed from the longitude: every
zip carries a BADM file (`*_FLUXNET_BIF_*.csv`) whose `UTC_OFFSET` row gives
it (AR-Bal reads -3), and a site whose BIF has no `UTC_OFFSET` is
`ctx.note_absent` — a REFUSAL that stops the pass — rather than a site placed
in the wrong hour. The row's time is the START of the half hour.

`qc`, ONE uint8 PER ROW:
  bits 0-2   NEE_VUT_REF_QC: 0 measured, 1 good-quality gap-fill, 2 medium,
             3 poor, 7 not reported by this file
  bits 3-5   LE_F_MDS_QC, on the same scale as the NEE one
  bit  6     the site averages HOURLY rather than half-hourly

THE CADENCE IS PER SITE AND IT IS IN THE ROW. Most towers publish
`_FLUXMET_HH_` (half-hourly); some publish `_FLUXMET_HR_` (hourly) and no HH
file at all — TERN's AU-Otw is one, measured on a runner. Dropping those would
throw away real towers over a naming convention, so both are read and the
cadence is recorded twice: in qc bit 6, so a consumer of the arrays alone can
see it, and as `resolution` in platforms.json. The store's `log2_dt` is the
half-hourly value, -7.907; an hourly row's support is log2((1/24)/5) = -6.907,
which is the same arrangement `fire` uses for its two footprints. The HUB is
NOT in qc: it is a property of the SITE, and platforms.json is where a site's
properties belong.

`platform` is `platform_hash(site_id)` and `platforms.json` carries each
site's id, latitude, longitude, elevation, IGBP vegetation class, hub, the
years the hub publishes and its citation — the `platform_meta` the ledger row
asks for.

THE FOOTPRINT. A tower's flux footprint is a few hundred metres of upwind
ground that moves with the wind, so there is no fixed pixel: `log2_fp` is
E-078's point label, -4, and `log2_dt` is log2((1/48)/5) = -7.907 — a half
hour of a five-day bin.

SIZE. The note's ledger row is 700+ sites, about 6,000 site-years, roughly
1.05e8 rows and 5 GB. The three hubs together list 812 products, which is
where the "700+" comes from. The probe measures a real month;
`ml/family1/probes/flux_2018-06.json` is the number that replaces the
estimate.

`FLUX_HUBS` ("amf,icos,tern") selects which hubs a run reads, so the build
can be split three ways if one hub is slow; the default is all three.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.parse
import zipfile

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

AMF_BASE = "https://amfcdn.lbl.gov/api/v2/"
AMF_SITES = AMF_BASE + "site_info_display/AmeriFlux"
AMF_SHUTTLE = AMF_BASE + "amf_shuttle_data_files_and_manifest"
AMF_USER_ID = "fluxnetshuttle"
ICOS_SPARQL = "https://meta.icos-cp.eu/sparql"
ICOS_SPEC = "http://meta.icos-cp.eu/resources/cpmeta/miscFluxnetArchiveProduct"
ICOS_ACCEPT = "https://data.icos-cp.eu/licence_accept?ids=%5B%22{}%22%5D"
ICOS_OBJECT = "https://data.icos-cp.eu/objects/{}"
TERN_BASE = ("https://dap.tern.org.au/thredds/fileServer/ecosystem_process/"
             "fluxnet/")
TERN_CATALOGUE = TERN_BASE + "TERN_THREDDS_catalogue.csv"
TERN_BIF = TERN_BASE + "BIF_all_sites.csv"
HUBS = ("amf", "icos", "tern")
HUB_CODE = {"amf": 1, "icos": 2, "tern": 3}
FILL = -9999.0
QC_NOT_REPORTED = 7

# channel -> the FULLSET half-hourly column it comes from
COLUMNS = (("nee", "NEE_VUT_REF"), ("gpp", "GPP_NT_VUT_REF"),
           ("reco", "RECO_NT_VUT_REF"), ("le", "LE_F_MDS"),
           ("h", "H_F_MDS"), ("rn", "NETRAD"), ("ta", "TA_F"),
           ("vpd", "VPD_F"), ("swc", "SWC_F_MDS_1"), ("p", "P_F"))
QC_COLUMNS = (("nee", "NEE_VUT_REF_QC"), ("le", "LE_F_MDS_QC"))
# SOME TOWERS AVERAGE HOURLY, NOT HALF-HOURLY, and FLUXNET names their file
# `_FLUXMET_HR_` instead of `_FLUXMET_HH_`. Measured on a runner: TERN's
# AU-Otw publishes HR, DD, WW, MM and YY and no HH at all. A site is one or
# the other, never both; dropping the hourly ones would throw away real towers
# over a naming convention, so both are read and the CADENCE is recorded --
# per row in qc bit 6, and per site in platforms.json.
# AND THE HALF-HOURLY FILE HAS TWO NAMES, because AmeriFlux still publishes
# some sites under the FLUXNET2015-era naming. Measured on a runner:
#   AMF_US-Ha1_FLUXNET_FLUXMET_HH_1991-2025_v1.3_r1.csv   ONEFlux v1.3+
#   AMF_CA-ER1_FLUXNET_FULLSET_HH_2015-2021_4-7.csv       the older shape
# The columns are the same in both (TA_F, NEE_VUT_REF, GPP_NT_VUT_REF, ...),
# which is the whole point of FULLSET, so both are read — BY COLUMN NAME, as
# everything here is — and a zip carrying two of them is a refusal.
HH_NAME = re.compile(r"_FLUXNET_(FLUXMET|FULLSET)_(HH|HR)_.*\.csv$", re.I)
HR_NAME = re.compile(r"_FLUXNET_(FLUXMET|FULLSET)_HR_.*\.csv$", re.I)
BIF_NAME = re.compile(r"_FLUXNET_BIF_(?!VARINFO).*\.csv$", re.I)
IGBP_URI = re.compile(r"igbp_([A-Za-z]+)$")


class FormatError(ValueError):
    """A hub answer or a site archive that does not match what was measured."""


# ============================================================ the archives =
def _json_get(url, attempts=4, timeout=None):
    raw, why = cm.get_bytes(url, attempts=attempts, timeout=timeout)
    if raw is None:
        raise IOError(f"{url}: {why}")
    return json.loads(raw), len(raw)


def _json_post(url, payload, attempts=4, form=False):
    import urllib.error
    import urllib.request
    if form:
        body = urllib.parse.urlencode(payload).encode()
        ctype = "application/x-www-form-urlencoded"
    else:
        body = json.dumps(payload).encode()
        ctype = "application/json"
    err = None
    for i in range(max(1, attempts)):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={**f10b.UA, "Content-Type": ctype,
                         "Accept": "application/json"})
            with urllib.request.urlopen(req,
                                        timeout=f10b.SOCKET_TIMEOUT) as r:
                raw = r.read()
            f10b.count_bytes(len(raw))
            return json.loads(raw), len(raw)
        except (IOError, OSError, ValueError) as e:      # noqa: PERF203
            err = e
            if i < attempts - 1:
                time.sleep(3.0 * (2 ** i))
    raise IOError(f"POST {url}: {type(err).__name__}: {err}")


def amf_listing(attempts=4):
    """AmeriFlux -> ({site_id: record}, counts). Two documented calls."""
    sites, n1 = _json_get(AMF_SITES, attempts)
    vals = sites.get("values") or []
    if not vals:
        raise FormatError(f"{AMF_SITES} listed no site at all")
    pub = {}
    for s in vals:
        sid = s.get("site_id")
        years = s.get("grp_publish_fluxnet")
        if not sid or not years:
            continue
        loc = s.get("grp_location") or {}
        igbp = (s.get("grp_igbp") or {}).get("igbp")
        pub[sid] = {"hub": "amf", "site_id": sid,
                    "name": s.get("site_name"),
                    "lat": _f(loc.get("location_lat")),
                    "lon": _f(loc.get("location_long")),
                    "elev_m": _f(loc.get("location_elev")),
                    "igbp": igbp, "country": s.get("country"),
                    "years": sorted(int(y) for y in years),
                    "doi": (s.get("doi") or {}).get("FLUXNET")
                    if isinstance(s.get("doi"), dict) else None}
    if not pub:
        raise FormatError(f"{AMF_SITES} listed {len(vals)} site(s) and none "
                          f"with grp_publish_fluxnet")
    d, n2 = _json_post(AMF_SHUTTLE,
                       {"user_id": AMF_USER_ID, "data_product": "FLUXNET",
                        "data_variant": "FULLSET",
                        "site_ids": sorted(pub)}, attempts)
    urls = d.get("data_urls") or []
    if not urls:
        raise FormatError(f"{AMF_SHUTTLE} returned no data_urls for "
                          f"{len(pub)} site(s)")
    got = 0
    for u in urls:
        sid = u.get("site_id")
        if sid not in pub:
            continue
        pub[sid].update(url=u.get("url"), bytes=int(u.get("download_size")
                                                    or 0),
                        md5=u.get("download_checksum"))
        got += 1
    out = {k: v for k, v in pub.items() if v.get("url")}
    counts = {"amf_sites_listed": len(vals), "amf_publish_fluxnet": len(pub),
              "amf_with_download": got, "amf_bytes_listing": n1 + n2,
              "amf_bytes_products": int(sum(v["bytes"]
                                            for v in out.values()))}
    return out, counts


def icos_query():
    """The Shuttle's own object-spec query, reduced to what this store uses.

    Only the fields the store actually needs are asked for, so the answer is
    one binding per data object instead of the Shuttle's 3,012 (its query
    also pulls every team member, which multiplies the rows).
    """
    return f"""
prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
prefix prov: <http://www.w3.org/ns/prov#>
select ?dobj ?station ?stationName ?fileName ?size ?timeStart ?timeEnd
       ?lat ?lon ?ecosystemType ?citationString
where {{
    VALUES ?spec {{<{ICOS_SPEC}>}}
    ?dobj cpmeta:hasObjectSpec ?spec .
    ?dobj cpmeta:wasAcquiredBy/prov:wasAssociatedWith ?station .
    ?dobj cpmeta:hasSizeInBytes ?size .
    ?dobj cpmeta:hasName ?fileName .
    ?dobj cpmeta:hasStartTime | (cpmeta:wasAcquiredBy / prov:startedAtTime)
          ?timeStart .
    ?dobj cpmeta:hasEndTime | (cpmeta:wasAcquiredBy / prov:endedAtTime)
          ?timeEnd .
    OPTIONAL {{ ?station cpmeta:hasName ?stationName . }}
    OPTIONAL {{ ?station cpmeta:hasLatitude ?lat .
                ?station cpmeta:hasLongitude ?lon . }}
    OPTIONAL {{ ?station cpmeta:hasEcosystemType ?ecosystemType . }}
    OPTIONAL {{ ?dobj cpmeta:hasCitationString ?citationString . }}
    FILTER NOT EXISTS {{[] cpmeta:isNextVersionOf ?dobj}}
}}
order by ?fileName
"""


def icos_listing(attempts=4):
    """ICOS -> ({site_id: record}, counts) from the SPARQL endpoint."""
    d, n = _json_post(ICOS_SPARQL, {"query": icos_query()}, attempts,
                      form=True)
    binds = ((d.get("results") or {}).get("bindings") or [])
    if not binds:
        raise FormatError(f"{ICOS_SPARQL} returned no binding for {ICOS_SPEC}")
    out = {}
    skipped = {}
    for b in binds:
        def g(k):
            return (b.get(k) or {}).get("value")
        name = g("fileName") or ""
        m = re.match(r"^[A-Za-z0-9]+_([A-Za-z]{2}-[A-Za-z0-9]+)_FLUXNET_", name)
        if not m:
            # NOT A REFUSAL: the `miscFluxnetArchiveProduct` spec also holds
            # the OLDER FLUXNET2015 archives (measured on a runner:
            # `FLX_DE-Dgw_FLUXNET2015_FULLSET_2015-2018_beta-3.zip`), whose
            # half-hourly file is a different product with a different column
            # set. The Shuttle's own `validate_fluxnet_filename_format` skips
            # exactly these, so the adapter skips them too and COUNTS them by
            # name rather than refusing the whole hub over a product it was
            # not asked for.
            skipped[name] = skipped.get(name, 0) + 1
            continue
        sid = m.group(1)
        oid = (g("dobj") or "").rsplit("/", 1)[-1]
        if not oid:
            raise FormatError(f"ICOS binding for {name} has no data-object id")
        eco = g("ecosystemType") or ""
        mi = IGBP_URI.search(eco)
        y0 = _year(g("timeStart"))
        y1 = _year(g("timeEnd"))
        rec = {"hub": "icos", "site_id": sid, "name": g("stationName"),
               "lat": _f(g("lat")), "lon": _f(g("lon")), "elev_m": None,
               "igbp": mi.group(1).upper() if mi else None,
               "country": None,
               "years": list(range(y0, y1 + 1)) if (y0 and y1) else [],
               "doi": None, "citation": g("citationString"),
               "url": ICOS_OBJECT.format(oid),
               "accept_url": ICOS_ACCEPT.format(oid),
               "object_id": oid, "filename": name,
               "bytes": int(float(g("size") or 0)), "md5": None}
        old = out.get(sid)
        if old is None or rec["filename"] > old["filename"]:
            out[sid] = rec
    if not out:
        raise FormatError(
            f"{ICOS_SPARQL} returned {len(binds)} object(s) and none in the "
            f"ONEFlux form <network>_<site>_FLUXNET_<years>_<version> "
            f"(skipped: {sorted(skipped)[:6]})")
    return out, {"icos_objects": len(binds), "icos_sites": len(out),
                 "icos_not_oneflux": int(sum(skipped.values())),
                 "icos_not_oneflux_names": sorted(skipped)[:20],
                 "icos_bytes_listing": n,
                 "icos_bytes_products": int(sum(v["bytes"]
                                                for v in out.values()))}


def tern_listing(attempts=4):
    """TERN -> ({site_id: record}, counts) from its two published CSVs."""
    raw, why = cm.get_bytes(TERN_CATALOGUE, attempts=attempts)
    if raw is None:
        raise IOError(f"{TERN_CATALOGUE}: {why}")
    rows = _csv_rows(raw, ("SITE_ID", "PRODUCT_URL"))
    braw, why = cm.get_bytes(TERN_BIF, attempts=attempts)
    if braw is None:
        raise IOError(f"{TERN_BIF}: {why}")
    bif = _csv_rows(braw, ("SITE_ID", "VARIABLE", "DATAVALUE"))
    meta = {}
    for r in bif:
        meta.setdefault(r["SITE_ID"], {})[r["VARIABLE"]] = r["DATAVALUE"]
    out = {}
    for r in rows:
        sid = r["SITE_ID"].strip()
        url = r["PRODUCT_URL"].strip()
        if not sid or not url:
            continue
        m = meta.get(sid, {})
        y = re.search(r"_FLUXNET_(\d{4})-(\d{4})_", url)
        out[sid] = {"hub": "tern", "site_id": sid,
                    "name": m.get("SITE_NAME"),
                    "lat": _f(m.get("LOCATION_LAT")),
                    "lon": _f(m.get("LOCATION_LONG")),
                    "elev_m": _f(m.get("LOCATION_ELEV")),
                    "igbp": (m.get("IGBP") or None),
                    "country": "Australia",
                    "years": (list(range(int(y.group(1)), int(y.group(2)) + 1))
                              if y else []),
                    "doi": r.get("PRODUCT_ID"),
                    "citation": r.get("PRODUCT_CITATION"),
                    "url": url, "bytes": 0, "md5": None,
                    "utc_offset": _f(m.get("UTC_OFFSET"))}
    if not out:
        raise FormatError(f"{TERN_CATALOGUE} listed no site at all")
    return out, {"tern_sites": len(out), "tern_bif_sites": len(meta),
                 "tern_bytes_listing": len(raw) + len(braw)}


def _csv_rows(raw, needed):
    text = raw.decode("utf-8", "replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FormatError("an empty CSV")
    head = _split_csv(lines[0])
    missing = [k for k in needed if k not in head]
    if missing:
        raise FormatError(f"the CSV header is missing {missing}: {head}")
    out = []
    for ln in lines[1:]:
        f = _split_csv(ln)
        if len(f) < len(head):
            continue
        out.append(dict(zip(head, f)))
    return out


def _split_csv(line):
    """A CSV line with quoted fields (TERN's citations carry commas)."""
    out, buf, q = [], [], False
    for ch in line:
        if ch == '"':
            q = not q
        elif ch == "," and not q:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf).strip())
    return out


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x == FILL else x


def _year(s):
    try:
        return int(str(s)[:4])
    except (TypeError, ValueError):
        return 0


# ============================================================== one archive =
def read_zip(raw, site_id, hub, counts):
    """A site's FLUXNET zip -> (columns, utc offset, counts). Raises."""
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as e:
        raise FormatError(f"{site_id}: not a zip ({e}) — the hub served "
                          f"{raw[:80]!r}") from None
    names = z.namelist()
    hh = [n for n in names if HH_NAME.search(n)]
    bif = [n for n in names if BIF_NAME.search(n)]
    if len(hh) != 1:
        raise FormatError(f"{site_id}: {len(hh)} half-hourly or hourly "
                          f"FLUXMET file(s) in the zip ({names[:8]}), "
                          f"expected exactly one")
    if not bif:
        raise FormatError(f"{site_id}: no BADM (BIF) file in the zip "
                          f"({names[:8]}) — the UTC offset lives there and is "
                          f"never guessed from the longitude")
    off, meta = read_bif(z.read(bif[0]), site_id)
    hourly = bool(HR_NAME.search(hh[0]))
    cols, counts = parse_hh(z.read(hh[0]), site_id, hub, off, counts,
                            hourly=hourly)
    counts["site_zip_members"] = len(names)
    counts["sites_hourly" if hourly else "sites_half_hourly"] = \
        counts.get("sites_hourly" if hourly else "sites_half_hourly", 0) + 1
    meta = dict(meta)
    meta["_resolution"] = "HR" if hourly else "HH"
    return cols, off, meta, counts


def read_bif(raw, site_id):
    """The BADM file -> (UTC offset in hours, {variable: value})."""
    rows = _csv_rows(raw, ("SITE_ID", "VARIABLE", "DATAVALUE"))
    meta = {}
    for r in rows:
        meta.setdefault(r["VARIABLE"], r["DATAVALUE"])
    if "UTC_OFFSET" not in meta:
        raise FormatError(
            f"{site_id}: the BADM file has no UTC_OFFSET row. FLUXNET "
            f"timestamps are LOCAL STANDARD TIME, so without it the rows "
            f"cannot be placed on a UTC axis and a longitude guess would put "
            f"the site in the wrong hour")
    try:
        off = float(meta["UTC_OFFSET"])
    except ValueError:
        raise FormatError(f"{site_id}: UTC_OFFSET is "
                          f"{meta['UTC_OFFSET']!r}") from None
    if not -14.0 <= off <= 14.0:
        raise FormatError(f"{site_id}: UTC_OFFSET {off} is outside -14..14")
    return off, meta


def parse_hh(raw, site_id, hub, utc_offset, counts, hourly=False):
    """The FULLSET half-hourly CSV -> columns, BY COLUMN NAME."""
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        raise FormatError(f"{site_id}: the half-hourly file is empty")
    head = [h.strip() for h in lines[0].split(",")]
    col = {h: i for i, h in enumerate(head)}
    if "TIMESTAMP_START" not in col:
        raise FormatError(f"{site_id}: no TIMESTAMP_START column "
                          f"({head[:8]})")
    idx = [col.get(src) for (_ch, src) in COLUMNS]
    absent = [ch for (ch, src), i in zip(COLUMNS, idx) if i is None]
    if absent:
        # A COUNT, NOT A LIST OF IDS: `_merge_counts` sums the leaves of a
        # counts dict across every part, and a list under a numeric key makes
        # the assembler raise. The channel names are what matter; which sites
        # lack a net radiometer is in platforms.json.
        counts.setdefault("channels_absent", {})
        for ch in absent:
            counts["channels_absent"][ch] = \
                counts["channels_absent"].get(ch, 0) + 1
    qidx = [col.get(src) for (_ch, src) in QC_COLUMNS]
    n = len(lines) - 1
    t = np.zeros(n, np.int64)
    vals = np.full((n, len(COLUMNS)), np.nan, np.float64)
    qc = np.zeros(n, np.uint8)
    off_s = int(round(utc_offset * 3600))
    cadence_bit = (1 << 6) if hourly else 0
    ncol = len(head)
    k = bad = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split(",")
        if len(f) != ncol:
            bad += 1
            continue
        s = f[col["TIMESTAMP_START"]].strip()
        if len(s) != 12 or not s.isdigit():
            bad += 1
            continue
        try:
            local = dt.datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                                int(s[8:10]), int(s[10:12]))
        except ValueError:
            bad += 1
            continue
        t[k] = f10b.seconds_since_epoch(local) - off_s
        for j, i in enumerate(idx):
            if i is not None:
                v = f[i].strip()
                if v:
                    try:
                        x = float(v)
                    except ValueError:
                        continue
                    if x != FILL:
                        vals[k, j] = x
        q = cadence_bit
        for shift, i in zip((0, 3), qidx):
            cls = QC_NOT_REPORTED
            if i is not None:
                w = f[i].strip()
                try:
                    cls = int(float(w))
                except ValueError:
                    cls = QC_NOT_REPORTED
                if cls < 0 or cls > 3:
                    cls = QC_NOT_REPORTED
            q |= (cls & 0b111) << shift
        qc[k] = q
        k += 1
    counts["rows_read"] = counts.get("rows_read", 0) + (len(lines) - 1)
    if bad:
        counts["rows_unparsable"] = counts.get("rows_unparsable", 0) + bad
    counts["site_years"] = counts.get("site_years", 0) + 1
    return {"t": t[:k], "values": vals[:k], "qc": qc[:k]}, counts


# ============================================================== the adapter =
class FluxAdapter(f10b.SourceAdapter):
    store = "flux"
    title = ("Half-hourly land-air CO2, water and heat exchange from "
             "eddy-covariance towers, FLUXNET Shuttle (2026-04-30)")
    family = "1tf"
    distribution = "public"
    licence = {
        "name": "CC BY 4.0",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("FLUXNET Shuttle (2026-04-30 release), assembled by "
                        "the FLUXNET coalition from the AmeriFlux, ICOS and "
                        "TERN data hubs. Each site's own citation and DOI are "
                        "in platforms.json; the data-use licence document "
                        "inside every site archive is part of the product."),
        "terms": ("fluxnet.org/data/download-data (read 2026-09-18) and the "
                  "Shuttle library's README: 'The FLUXNET data are shared "
                  "under a CC-BY-4.0 data use license which requires "
                  "attribution for each data use. See the data use license "
                  "document contained within the FLUXNET data product "
                  "(archive zip file) for details.'"),
    }
    channels = (("nee", "umolCO2/m2/s", -100.0, 100.0),
                ("gpp", "umolCO2/m2/s", -50.0, 200.0),
                ("reco", "umolCO2/m2/s", -20.0, 100.0),
                ("le", "W/m2", -200.0, 1000.0),
                ("h", "W/m2", -500.0, 1000.0),
                ("rn", "W/m2", -500.0, 1500.0),
                ("ta", "degC", -90.0, 60.0),
                ("vpd", "hPa", 0.0, 200.0),
                ("swc", "percent", 0.0, 100.0),
                ("p", "mm", 0.0, 500.0))
    # a tower's flux footprint moves with the wind: E-078's point label
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 48.0) / 5.0))    # -7.907, a half hour
    per_year = False
    first_year = 1991
    time_dtype = "int32"
    credentials = ()          # MEASURED: all three hubs answer anonymously
    platform_meta = True
    qc_policy = (
        "qc is one uint8 per row: bits 0-2 NEE_VUT_REF_QC (0 measured, 1 "
        "good-quality gap-fill, 2 medium, 3 poor, 7 not reported by this "
        "file), bits 3-5 LE_F_MDS_QC on the same scale, and bit 6 set when "
        "the site averages HOURLY rather than half-hourly (some towers "
        "publish _FLUXMET_HR_ and no HH file; the store's log2_dt is the "
        "half-hourly value and platforms.json carries each site's own "
        "`resolution`). The hub is NOT in qc -- it is a property of the site "
        "and platforms.json is where a site's properties belong. The "
        "product's own -9999 becomes NaN; "
        "a column a site's file does not carry (NETRAD and SWC_F_MDS_1 are "
        "the usual ones, because not every tower has a net radiometer or a "
        "soil probe) leaves that channel NaN for the whole site and is "
        "counted in `channels_absent`. A value outside its physical "
        "bound becomes NaN and is counted, never clipped. Two things are "
        "REFUSALS rather than guesses: a site archive whose BADM file has no "
        "UTC_OFFSET (FLUXNET timestamps are local standard time, and a "
        "longitude guess would put the site in the wrong hour), and a hub "
        "body that is not a zip -- ICOS answers its licence page as HTML "
        "when the accept cookie is missing, and an HTML page parsed as a zip "
        "would be a silent empty site")
    sources = (AMF_SITES, AMF_SHUTTLE, ICOS_SPARQL, TERN_CATALOGUE)
    verified = (
        "2026-09-18 from the sandbox, ANONYMOUSLY -- no FLUXNET credential "
        "was used or needed: the AmeriFlux site_info endpoint (200, 1.69 MB, "
        "844 sites of which 407 carry grp_publish_fluxnet), the AmeriFlux "
        "shuttle POST with the library's own fixed user_id 'fluxnetshuttle' "
        "(200, per-site zip URLs on ftp.fluxdata.org with sizes and MD5s; "
        "AMF_AR-Bal_FLUXNET_2012-2013_v1.3_r1.zip downloaded whole, "
        "22,077,723 bytes, MD5 96c666d12987ddcc1ec64a992e90ad10 matching the "
        "API's, and its FLUXMET_HH member read for the column list); the "
        "ICOS SPARQL endpoint (200, 3,012 bindings over 352 FLUXNET archive "
        "products) and its two-step licence-accept download (the accept URL "
        "sets CpLicenseAcceptedFor, after which the object answers 206 "
        "application/zip, PK magic, 29,093,732 bytes); and TERN's two "
        "published CSVs (200, 53 sites in the catalogue, BADM for all of "
        "them). The Shuttle library's own config.yaml and ameriflux.py were "
        "read to get the endpoints and to confirm there is no credential in "
        "the code path")
    notes = (
        "The download needs NO login: the FLUXNET Shuttle is a library over "
        "three public hubs (AmeriFlux, ICOS, TERN) and its AmeriFlux plugin "
        "posts the fixed literal user_id 'fluxnetshuttle'. The adapter calls "
        "the three hubs directly rather than depending on a "
        "`pip install git+https://github.com/fluxnet/shuttle.git` of a "
        "library that is not on PyPI. ICOS needs a two-step licence accept "
        "whose cookie the adapter keeps, and a body that is not a zip is a "
        "refusal. Timestamps are LOCAL STANDARD TIME and are converted with "
        "each site's own UTC_OFFSET from the BADM file inside its archive; a "
        "site without one is an absence, never a guess. The store's time is "
        "the START of the half hour.")
    smoke_window = ("2018-01-01", "2018-12-31")
    smoke_probe_month = "2018-06"
    HUBS_ENV = "FLUX_HUBS"
    YIELD_ROWS = 1_000_000

    def __init__(self):
        want = (os.environ.get(self.HUBS_ENV) or "").strip()
        if want:
            picked = [h.strip() for h in want.split(",") if h.strip()]
            bad = [h for h in picked if h not in HUBS]
            if bad or not picked:
                sys.exit(f"REFUSING flux: {self.HUBS_ENV}={want!r} names "
                         f"{bad or 'nothing'}; the hubs are {list(HUBS)}")
            self.hubs = picked
            self.notes = (f"{self.notes}\nHUBS: {self.HUBS_ENV} selected "
                          f"{picked} instead of all of {list(HUBS)}.")
        else:
            self.hubs = list(HUBS)
        self._sites = None
        self._jar = None

    # ------------------------------------------------------------- listing --
    def sites(self, ctx):
        """{site_id: record} over every enabled hub, and the counts."""
        if self._sites is not None:
            return self._sites
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "flux", "sites.json")
            if not os.path.exists(p):
                sys.exit(f"REFUSING flux: no {p} — the smoke's synthetic hub "
                         f"listing is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            d = json.loads(raw)
            out = {k: v for k, v in d["sites"].items()
                   if v["hub"] in self.hubs}
            counts = dict(d.get("counts") or {})
        else:
            out, counts = {}, {}
            fns = {"amf": amf_listing, "icos": icos_listing,
                   "tern": tern_listing}
            for hub in self.hubs:
                try:
                    got, c = fns[hub](attempts=ctx.a.attempts)
                except (FormatError, IOError) as e:
                    sys.exit(f"REFUSING flux: the {hub} hub: {e} — a listing "
                             f"that comes back empty or unreadable is a "
                             f"refusal (ml/CLAUDE.md, the 2026-09-14 rule)")
                counts.update(c)
                for sid, rec in got.items():
                    if sid in out:
                        counts["sites_in_two_hubs"] = \
                            counts.get("sites_in_two_hubs", 0) + 1
                        counts.setdefault("sites_in_two_hubs_ids", [])
                        if len(counts["sites_in_two_hubs_ids"]) < 20:
                            counts["sites_in_two_hubs_ids"].append(
                                f"{sid}: {out[sid]['hub']} kept, "
                                f"{hub} dropped")
                        continue
                    out[sid] = rec
        if not out:
            sys.exit(f"REFUSING flux: the hubs {self.hubs} listed no site at "
                     f"all — an empty listing is a refusal")
        counts["sites"] = len(out)
        self._sites = (out, counts)
        return self._sites

    # --------------------------------------------------------------- index --
    def index(self, ctx):
        sites, counts = self.sites(ctx)
        years = sorted({y for s in sites.values() for y in (s["years"] or ())})
        by_hub = {}
        for s in sites.values():
            by_hub[s["hub"]] = by_hub.get(s["hub"], 0) + 1
        out = {"dataset": "FLUXNET Shuttle FULLSET half-hourly, 2026-04-30",
               "url": "https://data.fluxnet.org/",
               "hubs": self.hubs, "sites": len(sites),
               "sites_by_hub": by_hub,
               "site_years_listed": int(sum(len(s["years"] or ())
                                            for s in sites.values())),
               "record": [str(min(years)) if years else None,
                          str(max(years)) if years else None],
               "years_listed": [min(years), max(years)] if years else [],
               "bytes_products": int(sum(s.get("bytes") or 0
                                         for s in sites.values())),
               "listing": counts,
               "columns": {ch: src for (ch, src) in COLUMNS},
               "qc_columns": {ch: src for (ch, src) in QC_COLUMNS}}
        # ONE REAL SITE ARCHIVE, downloaded and read
        small = min(sites.values(),
                    key=lambda s: (s.get("bytes") or 1 << 62, s["site_id"]))
        raw, why = self.archive(ctx, small)
        if raw is None:
            sys.exit(f"REFUSING flux: the smallest listed archive "
                     f"({small['site_id']}, {small.get('bytes')} bytes) could "
                     f"not be read: {why}")
        try:
            cols, off, meta, c = read_zip(raw, small["site_id"],
                                          small["hub"], {})
        except FormatError as e:
            sys.exit(f"REFUSING flux: {small['site_id']}: {e}")
        out["first_archive"] = {
            "site_id": small["site_id"], "hub": small["hub"],
            "bytes": len(raw), "rows": int(cols["t"].size),
            "utc_offset": off, "igbp": meta.get("IGBP"),
            "counts": c,
            "first_time_utc": (str(dt.datetime(1982, 1, 1)
                                   + dt.timedelta(
                                       seconds=int(cols["t"][0])))
                               if cols["t"].size else None)}
        return out

    # ------------------------------------------------------- one archive ----
    def jar(self):
        if self._jar is None:
            import http.cookiejar
            import urllib.request
            cj = http.cookiejar.CookieJar()
            self._jar = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(cj))
        return self._jar

    def archive(self, ctx, rec):
        """A site's zip as bytes -> (bytes, None) or (None, why)."""
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "flux", "archives",
                             f"{rec['site_id']}.zip")
            if not os.path.exists(p):
                return None, f"{p}: listed and absent"
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            return raw, None
        import urllib.request
        op = self.jar()
        attempts = max(int(ctx.a.attempts), 4)
        err = None
        for i in range(attempts):
            try:
                if rec.get("accept_url"):
                    # ICOS: accept the licence FIRST, in the same cookie jar
                    r = op.open(urllib.request.Request(rec["accept_url"],
                                                       headers=f10b.UA),
                                timeout=f10b.SOCKET_TIMEOUT)
                    r.read()
                    r.close()
                r = op.open(urllib.request.Request(rec["url"],
                                                   headers=f10b.UA),
                            timeout=f10b.SOCKET_TIMEOUT)
                want = r.headers.get("Content-Length")
                ctype = r.headers.get("Content-Type", "")
                raw = r.read()
                r.close()
                f10b.count_bytes(len(raw))
                if want is not None and len(raw) != int(want):
                    raise IOError(f"{len(raw):,} of {int(want):,} bytes — "
                                  f"truncated")
                if not raw.startswith(b"PK"):
                    raise IOError(
                        f"the body is {ctype!r} and does not start with a zip "
                        f"signature ({raw[:40]!r}) — ICOS answers its licence "
                        f"page as HTML when the accept cookie is missing, and "
                        f"an HTML page parsed as a zip is a silent empty site")
                if rec.get("md5"):
                    import hashlib
                    got = hashlib.md5(raw).hexdigest()
                    if got != str(rec["md5"]).lower():
                        raise IOError(f"md5 {got} != the hub's "
                                      f"{rec['md5']}")
                return raw, None
            except Exception as e:                              # noqa: BLE001
                err = e
                if i < attempts - 1:
                    time.sleep(3.0 * (2 ** i))
        return None, f"{type(err).__name__}: {err}"

    # --------------------------------------------------------- the contract -
    def fetch_stream(self, ctx):
        yield from self._stream(ctx, ctx.t_lo, ctx.t_hi)

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        parts, counts = [], {}
        for y, rows, c in self._stream(ctx, lo, hi):
            if y is None:
                counts = c or {}
            elif rows is not None:
                parts.append(rows)
        rows = cm.concat_rows(parts, self.C, self.time_dtype)
        yield from cm.batch_rows(self, rows, counts, f"{year}-{month:02d}")

    fetch_month_scope = "month"

    def _stream(self, ctx, t_lo, t_hi):
        sites, _ = self.sites(ctx)
        counts = {"sites_wanted": len(sites)}
        y_lo = (dt.datetime(1982, 1, 1)
                + dt.timedelta(seconds=int(t_lo))).year
        y_hi = (dt.datetime(1982, 1, 1)
                + dt.timedelta(seconds=int(t_hi))).year
        t0 = time.time()
        for i, sid in enumerate(sorted(sites), 1):
            rec = sites[sid]
            # A SITE THE HUB SAYS HAS NO YEAR IN THIS WINDOW IS NOT
            # DOWNLOADED. Each site is one archive covering its whole record,
            # so a month probe would otherwise pull every one of the 812 —
            # about 36 GB — to keep thirty days of a few hundred. The years
            # come from the hub's own listing; a site that publishes none is
            # read rather than skipped.
            yrs = rec.get("years") or []
            if yrs and (max(yrs) < y_lo or min(yrs) > y_hi):
                counts["sites_outside_window"] = \
                    counts.get("sites_outside_window", 0) + 1
                continue
            raw, why = self.archive(ctx, rec)
            if raw is None:
                ctx.note_absent(sid, f"{rec.get('url')}: {why}")
                continue
            try:
                cols, off, meta, counts = read_zip(raw, sid, rec["hub"],
                                                   counts)
            except FormatError as e:
                ctx.note_absent(sid, str(e))
                continue
            rec["resolution"] = meta.get("_resolution")
            rec["utc_offset"] = off
            del raw
            t = cols["t"]
            keep = (t >= t_lo) & (t <= t_hi)
            nout = int((~keep).sum())
            if nout:
                counts["rows_outside_window"] = \
                    counts.get("rows_outside_window", 0) + nout
            t = t[keep]
            v = cols["values"][keep]
            q = cols["qc"][keep]
            oob = self.mask_bounds(v)
            for k, m in (oob or {}).items():
                counts.setdefault("out_of_bounds", {})
                counts["out_of_bounds"][k] = \
                    counts["out_of_bounds"].get(k, 0) + m
            counts["sites_read"] = counts.get("sites_read", 0) + 1
            counts["rows_kept"] = counts.get("rows_kept", 0) + int(t.size)
            if t.size:
                lat = np.full(t.size, rec["lat"] if rec["lat"] is not None
                              else np.nan, np.float64)
                lon = np.full(t.size, rec["lon"] if rec["lon"] is not None
                              else np.nan, np.float64)
                plat = np.full(t.size, f10b.platform_hash(sid), np.int64)
                # one PART PER CALENDAR YEAR, because the framework's
                # one-stream branch writes a part per year
                yy = self._years_of(t)
                for y in np.unique(yy):
                    m = yy == y
                    rows = self.pack(t[m], lat[m], lon[m], v[m], plat[m],
                                     q[m])
                    yield int(y), rows, None
            if i % 10 == 0:
                ctx.prog.item(f"flux {sid}", None,
                              {"sites": i, "of": len(sites),
                               "rows": counts.get("rows_kept", 0),
                               "elapsed_s": round(time.time() - t0, 1)})
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        yield None, None, counts

    @staticmethod
    def _years_of(t):
        """The calendar year of each UTC second, vectorised."""
        days = np.floor_divide(np.asarray(t, np.int64), 86400)
        # 1982-01-01 is day 0; go through the proleptic Gregorian calendar
        z = days + 719468 + cm.EPOCH_DAYS_1970
        era = np.floor_divide(z, 146097)
        doe = z - era * 146097
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        y = yoe + era * 400
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        return y + (mp >= 10)

    def platforms(self, ctx):
        sites, _ = self.sites(ctx)
        out = {}
        for sid, s in sites.items():
            out[f10b.platform_hash(sid)] = {
                "id": sid, "lat": s["lat"], "lon": s["lon"],
                "elev_m": s.get("elev_m"),
                "igbp": s.get("igbp"),
                "name": s.get("name"), "country": s.get("country"),
                "hub": s["hub"], "hub_code": HUB_CODE[s["hub"]],
                "resolution": s.get("resolution"),
                "utc_offset": s.get("utc_offset"),
                "years_published": s.get("years") or [],
                "doi": s.get("doi"), "citation": s.get("citation"),
                "product_bytes": s.get("bytes") or None}
        return out

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi):
        truth = make_smoke_sources(root, d_lo, d_hi)
        self.__init__()
        return truth


# ================================================================== smoke ==
SMOKE_SITES = (
    # site, hub, lat, lon, elev, igbp, utc offset, years, quirk
    ("US-Smk", "amf", 42.5, -72.2, 340.0, "DBF", -5, (2018,), None),
    ("IT-Smk", "icos", 45.5, 11.0, 60.0, "GRA", 1, (2018,), "no_netrad"),
    ("AU-Smk", "tern", -33.5, 150.5, 20.0, "SAV", 10, (2018,), None),
)
# A SITE WITHOUT A UTC_OFFSET IS AN ABSENCE, which stops the pass — so it is
# not in the main smoke (the fetch would refuse, correctly). The test writes
# one explicitly to exercise that path, the way lst05's listed-and-missing
# granule is exercised.
SMOKE_BAD_SITE = ("ZZ-Bad", "amf", 0.0, 0.0, 0.0, "WSA", None, (2018,),
                  "no_offset")
SMOKE_MONTHS = (6, 7)
HH_HEAD_FULL = (
    "TIMESTAMP_START,TIMESTAMP_END,TA_F,TA_F_QC,VPD_F,P_F,NETRAD,"
    "SWC_F_MDS_1,LE_F_MDS,LE_F_MDS_QC,H_F_MDS,H_F_MDS_QC,NEE_VUT_REF,"
    "NEE_VUT_REF_QC,RECO_NT_VUT_REF,GPP_NT_VUT_REF")
HH_HEAD_NO_NETRAD = HH_HEAD_FULL.replace("NETRAD,", "")


def _hh_csv(site, off, quirk):
    head = HH_HEAD_NO_NETRAD if quirk == "no_netrad" else HH_HEAD_FULL
    cols = head.split(",")
    lines = [head]
    for month in SMOKE_MONTHS:
        d0 = dt.date(2018, month, 1)
        for step in range(96):                 # two days of half hours
            start = dt.datetime(d0.year, d0.month, d0.day) \
                + dt.timedelta(minutes=30 * step)
            end = start + dt.timedelta(minutes=30)
            k = step % 48
            rec = {
                "TIMESTAMP_START": start.strftime("%Y%m%d%H%M"),
                "TIMESTAMP_END": end.strftime("%Y%m%d%H%M"),
                "TA_F": round(12.0 + 8.0 * np.sin(k / 48 * 2 * np.pi), 3),
                "TA_F_QC": k % 4,
                "VPD_F": round(4.0 + 3.0 * abs(np.cos(k / 48 * 2 * np.pi)), 3),
                "P_F": 0.0 if k % 7 else 0.4,
                "NETRAD": round(-40.0 + 500.0 * max(
                    0.0, float(np.sin((k - 12) / 24 * np.pi))), 3),
                "SWC_F_MDS_1": round(20.0 + 0.1 * k, 3),
                "LE_F_MDS": round(20.0 * k % 180, 3),
                "LE_F_MDS_QC": k % 4,
                "H_F_MDS": round(10.0 * k % 120, 3),
                "H_F_MDS_QC": k % 3,
                "NEE_VUT_REF": round(-6.0 + 0.2 * (k % 30), 3),
                "NEE_VUT_REF_QC": k % 4,
                "RECO_NT_VUT_REF": round(2.0 + 0.05 * (k % 20), 3),
                "GPP_NT_VUT_REF": round(8.0 + 0.3 * (k % 25), 3),
            }
            if k == 3:
                rec["NEE_VUT_REF"] = int(FILL)         # the missing value
            if k == 5:
                rec["TA_F"] = 999.0                    # outside the bound
            lines.append(",".join(str(rec[c]) for c in cols))
    return "\n".join(lines) + "\n"


def _bif_csv(site, off, lat, lon, elev, igbp, quirk):
    rows = [("HEADER", "SITE_NAME", f"{site} smoke site"),
            ("LOCATION", "LOCATION_LAT", lat),
            ("LOCATION", "LOCATION_LONG", lon),
            ("LOCATION", "LOCATION_ELEV", elev),
            ("IGBP", "IGBP", igbp)]
    if quirk != "no_offset":
        rows.append(("UTC_OFFSET", "UTC_OFFSET", off))
    out = ["SITE_ID,GROUP_ID,VARIABLE_GROUP,VARIABLE,DATAVALUE"]
    for i, (grp, var, val) in enumerate(rows, start=1000):
        out.append(f"{site},{i},{grp},{var},{val}")
    return "\n".join(out) + "\n"


def make_smoke_sources(root, d_lo, d_hi, extra=()):
    """A synthetic hub listing plus one real zip per site, in the real
    layout: a FLUXMET HH member and a BIF member with UTC_OFFSET."""
    base = os.path.join(root, "flux")
    adir = os.path.join(base, "archives")
    os.makedirs(adir, exist_ok=True)
    every = tuple(SMOKE_SITES) + tuple(extra)
    sites = {}
    for (sid, hub, lat, lon, elev, igbp, off, years, quirk) in every:
        p = os.path.join(adir, f"{sid}.zip")
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"XX_{sid}_FLUXNET_FLUXMET_HH_2018-2018_v1.3_r1.csv",
                       _hh_csv(sid, off, quirk))
            z.writestr(f"XX_{sid}_FLUXNET_FLUXMET_DD_2018-2018_v1.3_r1.csv",
                       "TIMESTAMP,TA_F\n20180601,12.0\n")
            z.writestr(f"XX_{sid}_FLUXNET_BIF_2018-2018_v1.3_r1.csv",
                       _bif_csv(sid, off, lat, lon, elev, igbp, quirk))
            z.writestr(f"XX_{sid}_FLUXNET_BIFVARINFO_HH_2018-2018_v1.3_r1.csv",
                       "SITE_ID,GROUP_ID,VARIABLE_GROUP,VARIABLE,DATAVALUE\n")
            z.writestr("DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt", "CC BY 4.0")
        sites[sid] = {"hub": hub, "site_id": sid, "name": f"{sid} smoke site",
                      "lat": lat, "lon": lon, "elev_m": elev, "igbp": igbp,
                      "country": None, "years": list(years), "doi": None,
                      "citation": f"{sid} citation",
                      "url": f"file://{p}", "bytes": os.path.getsize(p),
                      "md5": None}
    with open(os.path.join(base, "sites.json"), "w") as fh:
        json.dump({"sites": sites,
                   "counts": {"amf_sites_listed": 2, "icos_objects": 1,
                              "tern_sites": 1}}, fh)
    os.environ.pop("FLUX_HUBS", None)

    ad = FluxAdapter()
    truth = []
    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    # SORTED BY SITE ID, because `_stream` walks `sorted(sites)` and three
    # sites on whole-hour offsets share most of their half-hour grid points:
    # the store breaks a (bin, time_s) tie by the order the adapter yielded
    # the rows, so the truth has to be in that order too (the contract's
    # smoke rule).
    for (sid, hub, lat, lon, elev, igbp, off, years, quirk) in sorted(
            every, key=lambda r: r[0]):
        if quirk == "no_offset":
            continue                       # an absence, not a row
        with open(os.path.join(adir, f"{sid}.zip"), "rb") as fh:
            cols, _off, _meta, _c = read_zip(fh.read(), sid, hub, {})
        v = cols["values"]
        ad.mask_bounds(v)
        for i in range(cols["t"].size):
            t = int(cols["t"][i])
            if not (t_lo <= t <= t_hi):
                continue
            truth.append({"t": t, "lat": lat, "lon": lon,
                          "platform": f10b.platform_hash(sid),
                          "v": [float(x) for x in v[i]]})
    return truth


ADAPTER = FluxAdapter
