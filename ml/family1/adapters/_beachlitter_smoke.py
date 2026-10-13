"""The synthetic archive of family 1.3's `beachlitter` store (E-092 D13).

Every source in its REAL on-disk layout, mirrored by URL exactly as the
adapter fetches it (`mplastic.mirror_name`): the NCEI Hub CSV with the real
37-column header (Nurdle Patrol, Beach, and rows of other settings that must
be ignored); accession 0259486 behind its landing page, archive listing and
version directory, as a workbook with the real Data and Data_Dictionary
sheets; the EMODnet beach-litter zip with its two CSVs (every survey listing
EVERY code of its reference list from the committed materials table); the
MDMAP site list and one export zip per site (README, survey CSV, sometimes a
back-barrier CSV); and four tiny Natural Earth layers that put each synthetic
beach on a coast, a lake, a river, a minor island or nowhere.

Hostile on purpose — each case is a named expectation in the tests: Nurdle
Patrol rows whose accession copy has the wrong longitude sign or a different
fourth decimal (joined, counted), a row with no accession copy (dropped), a
zero-minute patrol (dropped), a published rate that disagrees (qc bit 6), a
two-decimal position and a position in open water (qc bit 7), a river beach
(qc bit 5); EMODnet surveys under the LS and UN policies (excluded by licence),
a survey with no length (100 m assumed, bit 6), one of length 0 and one with
no lines (dropped), a clean-up (dropped), a two-list survey, a survey whose
start is 8 km from its beach and a location code with two coordinates (bit 7),
an all-zero survey; MDMAP transects with a 9999 width, a zero width and an
identical duplicate, an accumulation survey of site length 0, a Great Lakes
site (bit 5), ends 6 km apart (bit 7), missing coordinates, a file without
four of the material columns, a back-barrier file, a site with no surveys, a
site list that counts one survey more than the export, and an MDMAP survey that
copies an EMODnet one (dropped by precedence). The truth is computed HERE from
the generator's own parameters, never by calling the adapter.

`NURDLE_BULK` ordinary patrols are written as well, because the plan's
falsifier asks for at least 7,275 rows on which the published rate is
reproduced.
"""
import csv
import datetime as dt
import io
import json
import math
import os
import zipfile

import numpy as np

import build_family10_stores as f10b
from family1.adapters import beachlitter as bl
from family1.adapters import mplastic as mp

NAN = float("nan")
NURDLE_BULK = 7300

NCEI_HDR = ['Unique ID', 'Sample Date', 'Latitude (degree)',
            'Longitude (degree)', 'Ocean', 'Region', 'Subregion', 'Country',
            'State', 'Beach Location', 'Marine Setting',
            'Ocean Bottom Depth (m)', 'Water Sample Depth (m)',
            'Sediment Sample Depth (m)', 'Sampling Method', 'Mesh Size (mm)',
            'Transect Number', 'Sampling Point on Beach', 'Volunteers Number',
            'Collecting Time (min)', 'Standardized Nurdle  Amount',
            'Microplastics Measurement', 'Unit', 'Concentration Class Range',
            'Concentration Class', 'Short Reference', 'Long Reference', 'DOI',
            'Organization', 'Keywords', 'NCEI Accession Number',
            'NCEI Accession Link', 'Study Type', 'ObjectId', 'Symbology', 'x',
            'y']
EMODNET_META_HDR = [
    "country_code", "location_code", "location_name", "location_latitude",
    "location_longitude", "location_SRID", "location_measurement_date",
    "survey_code", "date", "survey_start_latitude", "survey_start_longitude",
    "survey_end_latitude", "survey_end_longitude", "survey_SRID",
    "survey_length", "survey_width", "filling_date", "survey_type",
    "originator", "data policy", "beach_notes"]
EMODNET_LITTER_HDR = [
    "country_code", "location_code", "location_name", "location_latitude",
    "location_longitude", "location_SRID", "survey_code", "data policy",
    "date", "survey_start_latitude", "survey_start_longitude",
    "survey_end_latitude", "survey_end_longitude", "survey_SRID",
    "survey_length", "survey_width", "survey_type", "litter_reference_list",
    "type_code", "type_name", "abundance"]
MDMAP_ITEMS = ["plastic:fragments-hard", "plastic:bags", "metal:fragments",
               "glass:fragments"]
MDMAP_HDR = (["survey_protocol", "shoreline_site_id", "survey_date",
              "survey_id", "transect_id", "shoreline_site_name",
              "beach_width_at_transect", "transect_width", "site_length",
              "num_people_searching_for_debris", "debris_search_start_time",
              "total_debris_items"] + list(bl.MDMAP_MATERIALS) + MDMAP_ITEMS +
             ["total_survey_duration", "country", "state_province_territory",
              "site_back_barrier_left_lat", "site_back_barrier_left_lon",
              "site_back_barrier_right_lat", "site_back_barrier_right_lon",
              "site_waters_edge_left_lat", "site_waters_edge_left_lon",
              "site_waters_edge_right_lat", "site_waters_edge_right_lon",
              "site_notes"])
README = ("Contact md.monitoring@noaa.gov with any questions about this "
          "dataset.\n\nProtocol versions are:\n\nMDMAP 2.0\nMDMAP Standing "
          "Stock Protocol\nMDMAP Accumulation Protocol\n")


def L(x):
    return NAN if x is None else math.log10(1.0 + x)


def _w(root, url, data):
    p = os.path.join(root, "beachlitter", mp.mirror_name(url))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(data if isinstance(data, bytes) else data.encode("utf-8"))
    return p


def _t(y, m, d):
    return int((np.datetime64(f"{y:04d}-{m:02d}-{d:02d}") -
                np.datetime64("1982-01-01")).astype(int)) * 86400 + 43200


def _xlsx(sheets):
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, (hdr, rows) in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(list(hdr))
        for r in rows:
            ws.append(list(r))
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def _csv(hdr, rows, bom=False):
    bio = io.StringIO()
    w = csv.DictWriter(bio, fieldnames=hdr, lineterminator="\r\n",
                       quoting=csv.QUOTE_NONNUMERIC)
    w.writeheader()
    for r in rows:
        w.writerow({h: r.get(h, "") for h in hdr})
    return ("﻿" if bom else "") + bio.getvalue()


def entry(src, t, lat, lon, plat, qc, v):
    out = [NAN] * len(bl.NAMES)
    for k, x in v.items():
        out[bl.COL[k]] = NAN if x is None else float(x)
    return {"src": src, "t": int(t), "lat": float(lat), "lon": float(lon),
            "platform": f10b.platform_hash(plat), "qc": int(qc), "v": out}


def litter(total, plastic, ambig, length, area=None):
    v = {"plastic_frac": plastic / total if total > 0 else None,
         "log_items_100m": L(total * 100.0 / length),
         "log_plastic_100m": L(plastic * 100.0 / length),
         "log_ambig_100m": L(ambig * 100.0 / length)}
    if area:
        v["log_items_m2"] = L(total / area)
    return v


# ================================================================ geography ==
#: synthetic shores: the sea along lon -97.3 (25..31 N, "Texas"), along
#: lat 54.0 (5..15 E, "North Sea"), along lat 38.13 (20..21 E, "Greece"); a
#: minor island ring at 24.64 N 82.87 W; a lake (Lake Michigan's box) and a
#: river at lon -98.1, 29.5..30 N.
TX_LON = -97.3


def write_natural_earth(root):
    def fc(geoms):
        return json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": g}
            for g in geoms]})
    ring = [[-82.87 + 0.01 * math.cos(a), 24.64 + 0.01 * math.sin(a)]
            for a in np.linspace(0, 2 * math.pi, 13)]
    layers = {
        "ne_10m_coastline": fc([
            {"type": "LineString", "coordinates": [[TX_LON, 25.0],
                                                   [TX_LON, 31.0]]},
            {"type": "LineString", "coordinates": [[5.0, 54.0], [15.0, 54.0]]},
            {"type": "LineString", "coordinates": [[20.0, 38.13],
                                                   [21.0, 38.13]]}]),
        "ne_10m_minor_islands_coastline": fc([
            {"type": "LineString", "coordinates": ring}]),
        "ne_10m_lakes": fc([
            {"type": "Polygon", "coordinates": [[[-88.0, 42.0], [-86.0, 42.0],
                                                 [-86.0, 46.0], [-88.0, 46.0],
                                                 [-88.0, 42.0]]]}]),
        "ne_10m_rivers_lake_centerlines": fc([
            {"type": "MultiLineString", "coordinates": [[[-98.1, 29.5],
                                                         [-98.1, 30.0]]]}]),
    }
    for n, txt in layers.items():
        _w(root, bl.NE_URL.format(name=n), txt)


# ==================================================================== NCEI ==
def _hub_row(uid, y, m, d, la, lo, setting, **kw):
    r = {h: "" for h in NCEI_HDR}
    r.update({"Unique ID": uid, "Sample Date": f"{m}/{d}/{y} 6:00:00 AM",
              "Latitude (degree)": la, "Longitude (degree)": lo,
              "Marine Setting": setting, "Country": "United States"})
    r.update(kw)
    return r


def write_ncei(root, corrupt=None):
    """The Hub CSV, accession 0259486 and the truth of the rows kept."""
    hub, xl, truth = [], [], []
    xid = [0]

    def nurdle(y, m, d, la, lo, n, mins, vol, pub, tide="No", xl_lat=None,
               xl_lon=None, in_xlsx=True, fresh=False, far=False):
        """la/lo are the Hub's published STRINGS."""
        uid = f"0259486-{len(hub) + 1:07d}"
        hub.append(_hub_row(uid, y, m, d, la, lo, "Nurdle Patrol",
                            **{"Volunteers Number": str(vol),
                               "Collecting Time (min)": str(mins),
                               "Standardized Nurdle  Amount": str(pub),
                               "Unit": "pieces/10 mins",
                               "Sampling Method": "Hand picking",
                               "Short Reference": "Tunnell et al. 2020",
                               "NCEI Accession Number": "259486",
                               "Beach Location": "Somewhere"}))
        if in_xlsx:
            xid[0] += 1
            xl.append([xid[0], dt.datetime(y, m, d), dt.datetime(y, m, d),
                       float(la) if xl_lat is None else xl_lat,
                       float(lo) if xl_lon is None else xl_lon,
                       "United States", "Texas", "Somewhere", tide, vol, mins,
                       n, pub, None])
        if not in_xlsx or mins == 0:
            return
        rate = n * 10.0 / (mins * vol)
        qc = 11 | bl.QC_NO_TIME
        if pub not in (math.floor(rate), math.ceil(rate)):
            qc |= bl.QC_CAVEAT
        if min(bl.decimals(la), bl.decimals(lo)) < 3 or far:
            qc |= bl.QC_POSITION
        if fresh:
            qc |= bl.QC_FRESH
        fla, flo = float(la), float(lo)
        truth.append(("nurdle", len(hub), entry(
            "nurdle", _t(y, m, d), fla, flo,
            f"beachlitter|nurdle|{fla:.3f},{flo:.3f}", qc,
            {"log_nurdles_10pmin": L(rate), "log_nurdle_count": L(n),
             "collect_min": mins, "volunteers": vol,
             "effort_pmin": mins * vol,
             "new_tide_line": 1.0 if tide == "Yes" else 0.0})))

    # the bulk: one ordinary patrol a day from 2001-01-01, on the Texas coast
    d0 = dt.date(2001, 1, 1)
    for i in range(NURDLE_BULK):
        day = d0 + dt.timedelta(days=i)
        n = i % 50
        pub = n
        if corrupt == "nurdle_rule" and i % 4:
            pub = n + 7           # a different rule: refused by the falsifier
        nurdle(day.year, day.month, day.day, f"{26.0 + (i % 400) * 0.01:.4f}",
               f"{TX_LON:.4f}", n, 10, 1, pub)
    # the special cases, 2021-05-01 ..
    nurdle(2021, 5, 1, "27.2000", "-97.3000", 120, 20, 3, 20, tide="Yes")
    nurdle(2021, 5, 2, "27.3000", "-97.3000", 30, 10, 1, 30,
           xl_lon=97.3)                                 # sign differs
    nurdle(2021, 5, 3, "27.1333", "-97.3000", 15, 10, 1, 15,
           xl_lat=27.13336)                             # 4th decimal
    nurdle(2021, 5, 4, "27.4000", "-97.3000", 0, 0, 1, 0)  # zero minutes
    nurdle(2021, 5, 5, "27.5000", "-97.3000", 47, 5, 1, 47)  # rate 94 != 47
    nurdle(2021, 5, 6, "27.6000", "-97.3000", 59, 25, 1, 23)  # 23.6 -> floor
    nurdle(2021, 5, 7, "27.14", "-97.30", 5, 10, 1, 5)    # two decimals
    nurdle(2021, 5, 8, "27.7000", "-97.3000", 9, 10, 1, 9,
           in_xlsx=False)                               # no accession copy
    nurdle(2021, 5, 9, "20.2637", "-94.8305", 3, 10, 2, 2,
           far=True)                                    # open water
    nurdle(2021, 5, 10, "29.7500", "-98.0800", 12, 10, 1, 12,
           fresh=True)                                  # a river beach
    nurdle(2021, 5, 11, "24.6400", "-82.8600", 0, 10, 1, 0)  # minor island
    # an accession row the Hub does not carry
    xid[0] += 1
    xl.append([xid[0], dt.datetime(2021, 6, 1), dt.datetime(2021, 6, 1),
               28.0, -97.3, "United States", "Texas", "x", "No", 1, 10, 1, 1,
               None])
    # sand cores, and settings that are not this store's
    for i, (val, la, lo, y, m, d) in enumerate(
            ((352000, "28.123456", "-97.300000", 2019, 6, 15),
             (0, "28.223456", "-97.300000", 2019, 6, 16))):
        hub.append(_hub_row(f"0278059-{i + 1:07d}", y, m, d, la, lo, "Beach",
                            **{"Microplastics Measurement": str(val),
                               "Unit": "pieces/m3",
                               "Short Reference": "Bissen and Chawchai, 2020",
                               "Beach Location": "Hat Ban Chuen",
                               "Sampling Method": "Metal spoon"}))
        truth.append(("ncei_sand", i, entry(
            "ncei_sand", _t(y, m, d), float(la), float(lo),
            "beachlitter|ncei_sand|Bissen and Chawchai, 2020|Hat Ban Chuen",
            12 | bl.QC_NO_TIME, {"log_n_m3_sand": L(val)})))
    hub.append(_hub_row("0211007-0000001", 2010, 1, 1, "30.0", "-40.0",
                        "Ocean water", Unit="pieces/m3",
                        **{"Microplastics Measurement": "0.1"}))
    hub.append(_hub_row("0300000-0000001", 2010, 1, 2, "30.0", "-40.0",
                        "Ocean sediment", Unit="pieces kg-1 d.w.",
                        **{"Microplastics Measurement": "3"}))
    _w(root, bl.NCEI_CSV, _csv(NCEI_HDR, hub, bom=True))
    # accession 0259486: landing page -> archive -> version -> data
    acc, arc = bl.NURDLE_ACC, "arc0207"
    _w(root, bl.NCEI_LANDING.format(acc7=acc),
       f'<html><a href="ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/{arc}/'
       f'{acc}/">ftp</a></html>')
    base = bl.NCEI_ARCHIVE.format(arc=arc, acc7=acc)
    _w(root, base, '<a href="?C=N;O=D">Name</a><a href="/data/oceans/">'
                   'Parent Directory</a><a href="2.1/">2.1/</a>'
                   '<a href="2.2/">2.2/</a>')
    d0p = f"{base}2.2/data/0-data/"
    name = "NurdlePatrolDatabase_S2N_v2.xlsx"
    _w(root, d0p, f'<a href="/data/oceans/">Parent</a>'
                  f'<a href="2M20XB-ISO-19115-2.xml">x</a>'
                  f'<a href="{name}">{name}</a>')
    defs = [["ID", "serial number"],
            ["Standardized Nurdle Amount",
             "the number of nurdles collected by one person in a 10-minute "
             "period. The standardized number allows us to compare."]]
    _w(root, d0p + name, _xlsx({
        "Data_Dictionary": (["Column Name", "Definition"], defs),
        "Data": (list(bl.NURDLE_HEADER), xl)}))
    return truth


# ================================================================= EMODnet ==
def write_emodnet(root, corrupt=None):
    doc = json.load(open(bl.MATERIALS_JSON, encoding="utf-8"))
    codes = {}
    for r in doc["codes"]:
        codes.setdefault(r["list"], []).append((r["code"], r["name"]))
    meta, lines, truth = [], [], []

    def survey(sc, pol, lists, y, m, d, loc, la, lo, length, width, counts,
               start=None, stype="MSFD_monitoring", with_lines=True,
               expect=None):
        row = {"country_code": "DE", "location_code": loc,
               "location_name": loc, "location_latitude": f"{la}",
               "location_longitude": f"{lo}", "location_SRID": 4326,
               "survey_code": sc, "date": f"{y:04d}-{m:02d}-{d:02d}",
               "survey_start_latitude": f"{(start or (la, lo))[0]}",
               "survey_start_longitude": f"{(start or (la, lo))[1]}",
               "survey_end_latitude": "NULL", "survey_end_longitude": "NULL",
               "survey_SRID": 4326, "survey_length": length,
               "survey_width": width, "survey_type": stype,
               "data policy": pol, "originator": "x", "filling_date": "NULL",
               "location_measurement_date": "NULL", "beach_notes": "NULL"}
        meta.append(row)
        if with_lines:
            for L_ in lists:
                for c, nm in codes[L_]:
                    if corrupt == "emodnet_lines" and sc == 101 and \
                            c == codes[L_][-1][0]:
                        continue
                    lines.append({**{k: row[k] for k in EMODNET_LITTER_HDR
                                     if k in row},
                                  "litter_reference_list": L_,
                                  "type_code": c, "type_name": nm,
                                  "abundance": counts.get((L_, c), 0)})
            if corrupt == "emodnet_unknown_code" and sc == 101:
                lines.append({**{k: row[k] for k in EMODNET_LITTER_HDR
                                 if k in row},
                              "litter_reference_list": "OSPAR",
                              "type_code": "777", "type_name": "New thing",
                              "abundance": 1})
        if expect is not None:
            truth.append(("emodnet", sc, entry(
                "emodnet", _t(y, m, d), la, lo, f"beachlitter|emodnet|{loc}",
                expect[0] | bl.QC_NO_TIME, expect[1])))

    P = 1
    # 101: OSPAR, 100 m: 10 plastic, 5 cigarette butts (ambiguous), 2 glass
    survey(101, "CB", ["OSPAR"], 2021, 3, 1, "DE001", 54.0, 8.0, 100, "NULL",
           {("OSPAR", "1"): 10, ("OSPAR", "64"): 5, ("OSPAR", "91"): 2},
           expect=(P, {**litter(17, 10, 5, 100.0), "survey_len_m": 100.0}))
    # 102: TSG-ML, no length -> 100 m assumed (bit 6), no area
    survey(102, "CB", ["TSG-ML"], 2021, 3, 2, "DE002", 54.0, 9.0, "NULL",
           "NULL", {("TSG-ML", "G1"): 4, ("TSG-ML", "G144"): 1,
                    ("TSG-ML", "G200"): 3},
           expect=(3 | bl.QC_CAVEAT, {**litter(8, 4, 1, 100.0),
                                      "survey_len_m": 100.0}))
    # 103: Joint List + TSG-ML in one survey, 100 m x 20 m
    survey(103, "CB", ["JLIST", "TSG-ML"], 2021, 3, 3, "LT001", 54.0, 10.0,
           100, 20, {("JLIST", "J3"): 6, ("JLIST", "J137"): 2,
                     ("TSG-ML", "G27"): 1},
           expect=(5, {**litter(9, 7, 0, 100.0, 2000.0),
                       "survey_len_m": 100.0}))
    # 104: UNEP/MARLIN, 320 m x 15 m
    survey(104, "CB", ["UNEP_MARLIN"], 2021, 3, 4, "FI001", 54.0, 11.0, 320,
           15, {("UNEP_MARLIN", "PL01"): 32, ("UNEP_MARLIN", "OT02"): 8,
                ("UNEP_MARLIN", "ME03"): 16},
           expect=(6, {**litter(56, 32, 8, 320.0, 4800.0),
                       "survey_len_m": 320.0}))
    # 105: length 0 -> dropped
    survey(105, "CB", ["OSPAR"], 2021, 3, 5, "DE001", 54.0, 8.0, 0, "NULL",
           {("OSPAR", "1"): 3})
    # 106: start 8 km from the beach (bit 7); every count zero
    survey(106, "CB", ["OSPAR"], 2021, 3, 6, "DE003", 54.0, 12.0, 100, "NULL",
           {}, start=(54.072, 12.0),
           expect=(P | bl.QC_POSITION, {**litter(0, 0, 0, 100.0),
                                        "survey_len_m": 100.0}))
    # 107 LS, 108 UN: excluded by licence (their lines are not even read)
    survey(107, "LS", ["OSPAR"], 2021, 3, 7, "DE001", 54.0, 8.0, 100, "NULL",
           {("OSPAR", "1"): 99})
    survey(108, "UN", ["TSG-ML"], 2021, 3, 8, "DE001", 54.0, 8.0, 100, "NULL",
           {("TSG-ML", "G1"): 99})
    # 109: no litter lines at all -> dropped
    survey(109, "CB", ["OSPAR"], 2021, 3, 9, "DE001", 54.0, 8.0, 100, "NULL",
           {}, with_lines=False)
    # 110: Greece; the MDMAP survey of the same day and place is a copy
    survey(110, "CB", ["OSPAR"], 2021, 4, 10, "GR001", 38.13, 20.5, 100,
           "NULL", {("OSPAR", "46"): 40},
           expect=(P, {**litter(40, 40, 0, 100.0), "survey_len_m": 100.0}))
    # 111, 112: one location code, two coordinates (bit 7)
    for sc, dd, lo in ((111, 11, 13.0), (112, 12, 13.001)):
        survey(sc, "CB", ["OSPAR"], 2021, 3, dd, "DE004", 54.0, lo, 100,
               "NULL", {("OSPAR", "117"): 7},
               expect=(P | bl.QC_POSITION, {**litter(7, 7, 0, 100.0),
                                            "survey_len_m": 100.0}))
    # 113: a clean-up -> dropped
    survey(113, "CB", ["OSPAR"], 2021, 3, 13, "DE001", 54.0, 8.0, 100,
           "NULL", {("OSPAR", "1"): 3}, stype="Cleaning")
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(bl.EMODNET_META, _csv(EMODNET_META_HDR, meta))
        zf.writestr(bl.EMODNET_LITTER, _csv(EMODNET_LITTER_HDR, lines))
    _w(root, bl.EMODNET_ZIP, z.getvalue())
    return truth


# =================================================================== MDMAP ==
def write_mdmap(root, corrupt=None):
    sites, truth = [], []

    def tr(proto, sid, svid, date, tid, bw, tw, sl, t, p, ends, back=None,
           drop_cols=False):
        r = {"survey_protocol": proto, "shoreline_site_id": sid,
             "survey_date": date, "survey_id": svid, "transect_id": tid,
             "shoreline_site_name": f"site {sid}",
             "beach_width_at_transect": "" if bw is None else bw,
             "transect_width": "" if tw is None else tw,
             "site_length": "" if sl is None else sl,
             "total_debris_items": t, "total_plastic_items": p,
             "total_metal_items": t - p, "total_glass_items": 0,
             "total_rubber_items": 0, "total_processed-lumber_items": 0,
             "total_cloth-fabric_items": 0, "total_other_items": 0,
             "plastic:fragments-hard": p, "plastic:bags": 0,
             "metal:fragments": t - p, "glass:fragments": 0,
             "country": "United States", "state_province_territory": "Texas"}
        (l1, l2), (r1, r2) = ends
        r.update({"site_waters_edge_left_lat": l1,
                  "site_waters_edge_left_lon": l2,
                  "site_waters_edge_right_lat": r1,
                  "site_waters_edge_right_lon": r2})
        b = back or (("", ""), ("", ""))
        r.update({"site_back_barrier_left_lat": b[0][0],
                  "site_back_barrier_left_lon": b[0][1],
                  "site_back_barrier_right_lat": b[1][0],
                  "site_back_barrier_right_lon": b[1][1]})
        if corrupt == "mdmap_sum" and svid == 1101 and tid == "B":
            r["total_glass_items"] = 1
        return r

    def site(sid, rows, listed, legacy=None, back=None, drop_cols=False):
        sites.append({"id": sid, "name": f"site {sid}",
                      "surveys": listed, "legacy": bool(legacy),
                      "legacyCount": legacy, "verified": True})
        url = bl.MDMAP_EXPORT.format(id=sid)
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", README)
            if rows:
                hdr = [h for h in MDMAP_HDR if not (
                    drop_cols and h in bl.MDMAP_MATERIALS[1:-1] + (
                        "total_other_items",))]
                zf.writestr(f"MDMAP_site{sid}_20261013.csv",
                            _csv(hdr, rows, bom=True))
            if back:
                zf.writestr(f"MDMAP_site{sid}_20261013_backbarrier.csv",
                            _csv(MDMAP_HDR, back, bom=True))
        _w(root, url, z.getvalue())

    def T(svid, date, proto_code, la, lo, v, qc=0, sid=None):
        y, m, d = (int(x) for x in date.split("-"))
        truth.append(("mdmap", svid, entry(
            "mdmap", _t(y, m, d), la, lo, f"beachlitter|mdmap|{sid}",
            proto_code | qc | bl.QC_NO_TIME, v)))

    two = "MDMAP 2.0 Protocol"
    ss = "MDMAP Standing Stock Protocol"
    acc = "MDMAP Accumulation Protocol"
    E10 = (("27.0", f"{TX_LON}"), ("27.0", f"{TX_LON}"))
    # site 10: MDMAP 2.0, two surveys; a 9999 width; a duplicated transect;
    # the site list names three surveys (recorded, not refused)
    rows = [tr(two, 10, 1001, "2022-05-01", "A", 20, 5, 100, 10, 8, E10),
            tr(two, 10, 1001, "2022-05-01", "B", 25, 5, 100, 5, 5, E10),
            tr(two, 10, 1001, "2022-05-01", "C", 9999, 5, 100, 7, 3, E10),
            tr(two, 10, 1001, "2022-05-01", "D", 30, 5, 100, 0, 0, E10)]
    rows.append(dict(rows[1]))
    rows += [tr(two, 10, 1002, "2022-06-01", t_, 10, 5, 100, a, b, E10)
             for t_, a, b in (("A", 2, 1), ("B", 0, 0), ("C", 4, 4),
                              ("D", 2, 0))]
    site(10, rows, [1001, 1002, 1003])
    T(1001, "2022-05-01", 8, 27.0, TX_LON,
      {**litter(22, 16, 0, 20.0, None), "log_items_m2": L(15 / 375.0),
       "survey_len_m": 20.0}, sid=10)
    T(1002, "2022-06-01", 8, 27.0, TX_LON,
      {**litter(8, 5, 0, 20.0, 200.0), "survey_len_m": 20.0}, sid=10)
    # site 11: standing stock, one zero width
    E11 = (("27.5", f"{TX_LON}"), ("27.5", f"{TX_LON}"))
    rows = [tr(ss, 11, 1101, "2015-07-01", t_, bw, 5, 100, a, b, E11)
            for t_, bw, a, b in (("A", 0, 3, 2), ("B", 12, 1, 1),
                                 ("C", 12, 0, 0), ("D", 12, 2, 2))]
    site(11, rows, [1101])
    T(1101, "2015-07-01", 9, 27.5, TX_LON,
      {**litter(6, 5, 0, 20.0, None), "log_items_m2": L(3 / 180.0),
       "survey_len_m": 20.0}, sid=11)
    # site 12: accumulation (a legacy site): 30 and 31 days; one of length 0
    E12 = (("28.0", f"{TX_LON}"), ("28.0", f"{TX_LON}"))
    rows = [tr(acc, 12, 1201, "2016-01-10", "A", 30, None, 100, 50, 40, E12),
            tr(acc, 12, 1202, "2016-02-09", "A", 30, None, 0, 20, 10, E12),
            tr(acc, 12, 1203, "2016-03-11", "A", 9999, None, 100, 25, 20,
               E12)]
    site(12, rows, None, legacy=3)
    T(1201, "2016-01-10", 10, 28.0, TX_LON,
      {**litter(50, 40, 0, 100.0, 3000.0), "survey_len_m": 100.0}, sid=12)
    T(1203, "2016-03-11", 10, 28.0, TX_LON,
      {**litter(25, 20, 0, 100.0, None), "survey_len_m": 100.0,
       "accum_days": 31.0}, sid=12)
    # site 13: the Great Lakes (freshwater)
    E13 = (("43.0", "-87.99"), ("43.0", "-87.99"))
    site(13, [tr(acc, 13, 1301, "2017-05-05", "A", 40, None, 200, 10, 9,
                 E13)], [1301])
    T(1301, "2017-05-05", 10, 43.0, -87.99,
      {**litter(10, 9, 0, 200.0, 8000.0), "survey_len_m": 200.0},
      qc=bl.QC_FRESH, sid=13)
    # site 14: no surveys
    site(14, [], [])
    # site 15: a back-barrier file; ends 6 km apart (bit 7)
    E15 = (("26.5", f"{TX_LON}"), ("26.554", f"{TX_LON}"))
    rows = [tr(two, 15, 1501, "2023-08-08", t_, 10, 5, 100, 1, 1, E15)
            for t_ in "ABCD"]
    back = [tr(two, 15, 1501, "2023-08-08", t_, 10, 5, 100, 9, 9, E15)
            for t_ in "ABC"]
    site(15, rows, [1501], back=back)
    T(1501, "2023-08-08", 8, 26.527, TX_LON,
      {**litter(4, 4, 0, 20.0, 200.0), "survey_len_m": 20.0},
      qc=bl.QC_POSITION, sid=15)
    # site 16: the left end missing -> the right end alone
    E16 = (("null", "null"), ("29.0", f"{TX_LON}"))
    rows = [tr(ss, 16, 1601, "2014-04-04", t_, 8, 5, 100, 2, 1, E16)
            for t_ in "ABCD"]
    site(16, rows, [1601])
    T(1601, "2014-04-04", 9, 29.0, TX_LON,
      {**litter(8, 4, 0, 20.0, 160.0), "survey_len_m": 20.0}, sid=16)
    # site 17: no water's-edge coordinate -> the back barrier; width blank
    E17 = (("", ""), ("", ""))
    site(17, [tr(acc, 17, 1701, "2013-03-03", "A", None, None, 100, 6, 6,
                 E17, back=(("29.5", f"{TX_LON}"), ("29.5", f"{TX_LON}")))],
         None, legacy=1)
    T(1701, "2013-03-03", 10, 29.5, TX_LON,
      {**litter(6, 6, 0, 100.0, None), "survey_len_m": 100.0}, sid=17)
    # site 18: four material columns absent from the file
    E18 = (("30.0", f"{TX_LON}"), ("30.0", f"{TX_LON}"))
    rows = [tr(two, 18, 1801, "2024-02-02", t_, 10, 5, 100, 3, 3, E18)
            for t_ in "ABCD"]
    site(18, rows, [1801], drop_cols=True)
    T(1801, "2024-02-02", 8, 30.0, TX_LON,
      {**litter(12, 12, 0, 20.0, 200.0), "survey_len_m": 20.0}, sid=18)
    # site 19: Greece, a copy of EMODnet survey 110 (dropped by precedence)
    E19 = (("38.135", "20.505"), ("38.135", "20.505"))
    rows = [tr(two, 19, 1901, "2021-04-10", t_, 10, 5, 100, 1, 1, E19)
            for t_ in "ABCD"]
    site(19, rows, [1901])
    _w(root, bl.MDMAP_SITES, json.dumps(sites))
    return truth


# ==================================================================== all ==
def write_all(root, corrupt=None):
    write_natural_earth(root)
    tn = write_ncei(root, corrupt)
    te = write_emodnet(root, corrupt)
    tm = write_mdmap(root, corrupt)
    return tn + te + tm


def make_smoke_sources(root, d_lo, d_hi, corrupt=None):
    entries = write_all(root, corrupt)
    rank = {s: i for i, s in enumerate(bl.SOURCES)}
    lo = f10b.seconds_since_epoch(d_lo)
    hi = f10b.seconds_since_epoch(d_hi) + 86399
    entries = sorted(entries, key=lambda e: (e[2]["t"], rank[e[0]], e[1]))
    return [e[2] for e in entries if lo <= e[2]["t"] <= hi]
