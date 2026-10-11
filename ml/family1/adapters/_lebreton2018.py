"""The Ocean Cleanup / Lebreton et al. 2018 — the published files, parsed ONCE.

Shared by the globe's bake (`scripts/refresh_data.py gpgp`, which writes
`data/gpgp_plastic.json`) and by family 1.3's two stores (`mplastic`,
`mobjects`, E-092). It is a helper module (leading underscore), so the
family-1 adapter registry does not try to read an ADAPTER from it, and it
imports nothing beyond the standard library: `refresh_data.py` loads it by
file path, without importing the adapter package.

SOURCE. figshare article 5873142 (doi:10.6084/m9.figshare.5873142), CC BY 4.0,
the data of Lebreton, L. et al. (2018) "Evidence that the Great Pacific Garbage
Patch is rapidly accumulating plastic", Sci. Rep. 8:4666. The figshare WEB
page answers 403 to scripts; the API (`api.figshare.com/v2/articles/<id>/
files`) and the ndownloader links are keyless, and file ids are resolved
through the API BY NAME, never typed.

TWO TRAPS IN THE PUBLISHED FILES, both ASSERTED here (seen 2026-10-08/09):
  * the Megaplastics sheet has its Longitude and Latitude columns SWAPPED
    ("Longitude" ~ 33, "Latitude" ~ -141). Every position comes from
    StationInfo; each concentration row must agree with its StationInfo
    midpoint within 0.01 deg — the swap is allowed for that one sheet only,
    and every one of its rows must be swapped; anything else raises.
  * event IDs are FLOATS ("1.2000000000000002" in StationInfo, 1.2 in the
    concentration sheets), so a naive join silently drops ~7 % of rows. IDs
    are normalised by rounding to 6 decimals (`event_id`) and every join —
    the four concentration sheets and MosaicDebrisInfo — must be 100 %.

Values are returned RAW (the cells as xlrd reads them); the globe bake
rounds them to four significant figures for the web, a store keeps them.
"""
import math

ARTICLE = "5873142"
DOI = "10.6084/m9.figshare.5873142"
PAPER_DOI = "10.1038/s41598-018-22939-w"
API_FILES = f"https://api.figshare.com/v2/articles/{ARTICLE}/files"
FILES = ("Lebreton2018_Concentration.xls",
         "Lebreton2018_SamplingInformation.xls",
         "Lebreton2018_HistoricalDataset.xls")
SHEETS = [   # (size-class key, sheet name, expected rows, sampling type)
    ("micro", "Microplastics (0.05-0.5 cm)", 501, "Manta trawl"),
    ("meso", "Mesoplastics (0.5-5 cm)", 501, "Manta trawl"),
    ("macro", "Macroplastics (5-50 cm)", 151, "Mega trawl"),
    ("mega", "Megaplastics (>50 cm)", 31, "RGB mosaic"),
]
TYPE = {"Manta trawl": "manta", "Mega trawl": "mega", "RGB mosaic": "aerial"}
#: every event carries exactly the size classes its gear samples
CLASSES_OF = {"manta": {"micro", "meso"}, "mega": {"macro"},
              "aerial": {"mega"}}
EXPECT_EVENTS = {"manta": 501, "mega": 151, "aerial": 31}
EXPECT_OBJECTS = 1595
EXPECT_HISTORICAL = 3532
#: the four estimates of every concentration sheet, in this order
ESTIMATES = ("midpoint", "lower", "higher", "raw")


class FormatError(AssertionError):
    """The published files are not the layout verified above."""


def require(cond, msg):
    if not cond:
        raise FormatError(msg)


def event_id(v):
    """Event IDs arrive as floats with representation noise."""
    x = round(float(v), 6)
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def num(v):
    """A cell -> float, or None for blanks, text and NaN ('nan' included)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v or v in (math.inf, -math.inf) else v


def txt(v):
    s = " ".join(str(v).split())
    return None if s.lower() in ("", "nan", "none") else s


def resolve(files_listing):
    """The API's file listing -> {name: file record} for the three files.

    A listing without one of them is a REFUSAL, never a skip.
    """
    by_name = {f["name"]: f for f in files_listing}
    miss = [n for n in FILES if n not in by_name]
    require(not miss, f"figshare {ARTICLE} lists no {miss} — the listing "
                      f"holds {sorted(by_name)}")
    return {n: by_name[n] for n in FILES}


def rows(sheet):
    """An xlrd sheet -> one dict per data row, keyed by the stripped header."""
    hdr = [str(h).strip() for h in sheet.row_values(0)]
    for r in range(1, sheet.nrows):
        yield dict(zip(hdr, sheet.row_values(r)))


# ============================================================== StationInfo ==
def read_events(samp_wb):
    """StationInfo -> (events {id: dict}, order [id, ...]).

    Each event carries the raw start/end positions, the midpoint (`lon`,
    `lat`, rounded to 4 decimals as the globe has always stored it — the
    0.01-degree join below compares against that), the start date and the
    start hour (None where the sheet has none), and the sampling geometry.
    """
    events, order = {}, []
    for r in rows(samp_wb.sheet_by_name("StationInfo")):
        eid = event_id(r["Sampling event ID"])
        require(eid not in events,
                f"duplicate event id {eid} after normalisation")
        require(r["Sampling type"] in TYPE,
                f"event {eid}: sampling type {r['Sampling type']!r}")
        typ = TYPE[r["Sampling type"]]
        y = int(r["Start year (UTC)"])
        mo = int(r["Start month (UTC)"])
        d = int(r["Start day (UTC)"])
        h = num(r["Start UTC time (hours)"])
        if h is not None:
            require(0 <= h < 24, f"event {eid}: start hour {h}")
        la0 = num(r["Start latitude (degrees)"])
        lo0 = num(r["Start longitude (degrees)"])
        la1 = num(r["End latitude (degrees)"])
        lo1 = num(r["End longitude (degrees)"])
        if la1 is None or lo1 is None:
            la1, lo1 = la0, lo0
        events[eid] = {
            "id": eid, "type": typ,
            "platform": str(r["Platform name"]).strip(),
            "y": y, "mo": mo, "d": d, "hour": h,
            "lon0": lo0, "lat0": la0, "lon1": lo1, "lat1": la1,
            "lon": round((lo0 + lo1) / 2, 4), "lat": round((la0 + la1) / 2, 4),
            "area_km2": r["Sampling area (km2)"],
            "dist_km": r["Sampling distance (km)"],
            "dur_h": r["Sampling duration (hours)"],
            "depth_km": r.get("Sampling depth (km)"),
            "width_km": r.get("Sampling width (km)"),
            "wind_kn": r["Wind speed (Knots)"],
            "bft": r["Sea state (Beaufort)"],
            "c": {},
        }
        order.append(eid)
    n_by_type = {}
    for e in events.values():
        n_by_type[e["type"]] = n_by_type.get(e["type"], 0) + 1
    require(len(events) == sum(EXPECT_EVENTS.values())
            and n_by_type == EXPECT_EVENTS,
            f"StationInfo: {len(events)} events {n_by_type}, expected "
            f"{EXPECT_EVENTS}")
    return events, order


def start_iso(ev):
    """'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MMZ' — the globe's start stamp."""
    ts = f"{ev['y']:04d}-{ev['mo']:02d}-{ev['d']:02d}"
    if ev["hour"] is not None:
        mins = int(round(ev["hour"] * 60))
        if mins >= 1440:
            mins = 1439
        ts += f"T{mins // 60:02d}:{mins % 60:02d}Z"
    return ts


# ======================================================= concentration sheets ==
def read_concentrations(conc_wb, events):
    """The four concentration sheets, joined to `events` IN PLACE.

    `events[id]["c"][class] = {"n": [mid, lower, higher, raw], "g": [...]}`
    with the raw cell values. Returns {class: (rows, swapped rows)}.
    """
    stats = {}
    for key, sheet, expect_n, stype in SHEETS:
        sh = conc_wb.sheet_by_name(sheet)
        n = swapped = 0
        for r in rows(sh):
            eid = event_id(r["Sampling event ID"])
            require(eid in events,
                    f"{sheet}: event {eid} has no StationInfo row")
            ev = events[eid]
            require(r["Sampling type"] == stype and ev["type"] == TYPE[stype],
                    f"{sheet}: event {eid} type mismatch")
            clon = num(r["Longitude (degrees)"])
            clat = num(r["Latitude (degrees)"])
            ok = abs(clon - ev["lon"]) < 0.01 and abs(clat - ev["lat"]) < 0.01
            if not ok and key == "mega":
                ok = abs(clat - ev["lon"]) < 0.01 and \
                    abs(clon - ev["lat"]) < 0.01
                swapped += ok
            require(ok, f"{sheet}: event {eid} at ({clon},{clat}) disagrees "
                        f"with StationInfo ({ev['lon']},{ev['lat']}) by more "
                        f"than 0.01 deg")

            def col(prefix, unit, r=r, sheet=sheet):
                for k, v in r.items():
                    kk = " ".join(k.lower().split())
                    if kk.startswith(prefix) and kk.endswith(f"({unit} km-2)"):
                        return v
                raise FormatError(f"{sheet}: no column {prefix} {unit}")
            require(key not in ev["c"], f"{sheet}: event {eid} twice")
            ev["c"][key] = {"n": [col(p, "#") for p in ESTIMATES],
                            "g": [col(p, "g") for p in ESTIMATES]}
            n += 1
        require(n == expect_n, f"{sheet}: {n} rows, expected {expect_n}")
        if key == "mega":
            require(swapped == n, f"Megaplastics swap: only {swapped}/{n} "
                                  f"rows matched swapped")
        else:
            require(swapped == 0, f"{sheet}: {swapped} swapped rows outside "
                                  f"the Megaplastics sheet")
        stats[key] = (n, swapped)
    for e in events.values():
        require(set(e["c"]) == CLASSES_OF[e["type"]],
                f"event {e['id']}: classes {set(e['c'])}")
    return stats


# ========================================================= MosaicDebrisInfo ==
def read_objects(samp_wb, events):
    """MosaicDebrisInfo -> one dict per object (raw cells), 100 % joined."""
    out = []
    for r in rows(samp_wb.sheet_by_name("MosaicDebrisInfo")):
        eid = event_id(r["Sampling event ID"])
        require(eid in events and events[eid]["type"] == "aerial",
                f"object on event {eid}")
        lo0 = num(r["Start longitude debris (degrees)"])
        la0 = num(r["Start latitude debris (degrees)"])
        lo1 = num(r["End longitude debris (degrees)"])
        la1 = num(r["End latitude debris (degrees)"])
        if lo1 is None or la1 is None:
            lo1, la1 = lo0, la0
        lt50 = str(r["< 50 cm?"]).strip().lower()
        out.append({
            "event": eid, "lon0": lo0, "lat0": la0, "lon1": lo1, "lat1": la1,
            "lon": round((lo0 + lo1) / 2, 4), "lat": round((la0 + la1) / 2, 4),
            "type": str(r["Object type"]).strip(),
            "colour": str(r["Color"]).strip(),
            "length_m": r["Length (m)"], "width_m": r["Width (m)"],
            "lt50": 1 if lt50 == "yes" else 0 if lt50 == "no" else None,
            "area_m2": r["Topview area (m^2)"],
            "kg_lo": r["Lower weight (kg)"], "kg_mid": r["Mid weight (kg)"],
            "kg_hi": r["Upper weight (kg)"],
        })
    require(len(out) == EXPECT_OBJECTS,
            f"MosaicDebrisInfo: {len(out)} objects, expected {EXPECT_OBJECTS}")
    return out


# ======================================================== HistoricalDataset ==
def read_historical(hist_wb):
    """The 1972-2015 compilation of earlier net tows (first sheet).

    One dict per tow with the raw cells: origin (cruise / data origin),
    reference (study), year, month (None where blank), lon, lat, gear
    (lower-case), mesh (microns), net depth (m), n (#/km2), g (g/km2),
    inside-GPGP flag.
    """
    hs = hist_wb.sheets()[0]
    out = []
    for r in rows(hs):
        y, mo = num(r["Sampling Year"]), num(r["Sampling Month"])
        lon, lat = num(r["Longitude (degrees)"]), num(r["Latitude (degrees)"])
        require(y is not None and lon is not None and lat is not None,
                f"historical row without year or position: {r}")
        gear = txt(r["Sampling type"])
        out.append({
            "origin": r["Data Origin"], "ref": r["Data Reference"],
            "y": int(y), "mo": int(mo) if mo else None,
            "lon": lon, "lat": lat,
            "gear": gear.lower() if gear else None,
            "mesh_um": r["Mesh Size (microns)"],
            "depth_m": r["Net tow Depth (in m)"],
            "n": r["Microplastic Numerical Concentration (#/km²)"],
            "g": r["Microplastic Mass Concentration (g/km²)"],
            "inside": num(r["Inside GPGP?"]),
        })
    require(len(out) == EXPECT_HISTORICAL,
            f"historical: {len(out)} tows, expected {EXPECT_HISTORICAL}")
    require(all(-180 <= h["lon"] <= 180 and -90 <= h["lat"] <= 90
                for h in out), "historical: a position outside the globe")
    return out


def read_all(paths):
    """{file name: local path} -> everything the three files hold."""
    import xlrd
    samp = xlrd.open_workbook(paths["Lebreton2018_SamplingInformation.xls"])
    conc = xlrd.open_workbook(paths["Lebreton2018_Concentration.xls"])
    hist = xlrd.open_workbook(paths["Lebreton2018_HistoricalDataset.xls"])
    events, order = read_events(samp)
    stats = read_concentrations(conc, events)
    objects = read_objects(samp, events)
    historical = read_historical(hist)
    return {"events": events, "order": order, "sheet_stats": stats,
            "objects": objects, "historical": historical}
