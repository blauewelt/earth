#!/usr/bin/env python3
"""Build docs/PRIMER.pdf — the background knowledge behind the earth globe app.

Run from repo root: python3 scripts/build_primer.py
Requires: reportlab (pip install reportlab)
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "PRIMER.pdf")

ACCENT = colors.HexColor("#1a5fa8")
DIM = colors.HexColor("#555555")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=ACCENT, spaceBefore=18, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=ACCENT, spaceBefore=12, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=ss["Normal"], fontSize=10, leading=14.5,
                      spaceAfter=7, alignment=4)  # justified
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=9, leading=13, textColor=DIM,
                      leftIndent=6 * mm, spaceAfter=7)
TITLE = ParagraphStyle("Ti", parent=ss["Title"], textColor=ACCENT, spaceAfter=2)
SUB = ParagraphStyle("Sub", parent=ss["Normal"], fontSize=11, textColor=DIM, spaceAfter=18)
CELL = ParagraphStyle("Cell", parent=ss["Normal"], fontSize=8.5, leading=11)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")

def P(text, style=BODY):
    return Paragraph(text, style)

def table(headers, rows, widths):
    data = [[Paragraph(h, CELLB) for h in headers]] + \
           [[Paragraph(c, CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c6d6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fb")]),
    ]))
    return t

story = []

# ----------------------------------------------------------------- title page
story += [
    Spacer(1, 30 * mm),
    P("Open Climate Data on a Globe", TITLE),
    P("The background knowledge behind the <i>earth</i> project: GIBS, tiles, "
      "colormaps, product levels, climatologies and the datasets in the app", SUB),
    Spacer(1, 6 * mm),
    P("This primer collects, in one place, the concepts needed to understand how "
      "open climate data travels from a satellite or a rain gauge onto an interactive "
      "3-D globe — and why the app is built the way it is. It is written for a "
      "technically-minded reader; no remote-sensing background is assumed."),
    P("Companion documents in the repository: <b>docs/CATALOG.md</b> (241+ dataset "
      "catalog), <b>docs/COMBINING_DATASETS.md</b> (which datasets measure the same "
      "quantity and how they may be combined), <b>docs/SPECIES_AND_CLIMATE.md</b>."),
]
story.append(PageBreak())

# ------------------------------------------------------------------- 1. GIBS
story.append(P("1 · What is GIBS?", H1))
story += [
    P("<b>GIBS</b> — the <b>Global Imagery Browse Services</b> — is NASA's public "
      "tile server for satellite imagery. It serves over 1,000 visualised data "
      "products (true-colour mosaics, sea-surface temperature, precipitation, snow "
      "cover, aerosols, night lights, …) as small pre-rendered map tiles that any "
      "web map client can request over plain HTTPS, with no API key or account. It "
      "is the engine behind NASA Worldview, and it is where most of the raster "
      "layers on the earth globe come from."),
    P("The crucial idea is that GIBS serves <b>pictures, not numbers</b>. A GIBS "
      "tile is an ordinary PNG or JPEG in which each data value has been mapped "
      "through a published colour table. That makes display trivially fast — the "
      "browser just draws images — but it means that recovering the physical value "
      "at a pixel requires inverting the colormap (section 4), and that any "
      "computation (differences, averages) must happen on the client after that "
      "inversion. The alternative — downloading the underlying NetCDF/HDF granules "
      "and rendering from raw numbers — is more exact but orders of magnitude "
      "heavier; GIBS is the pragmatic middle ground that makes a globe of daily "
      "global imagery feasible in a browser."),
    P("A GIBS tile URL is fully deterministic:", BODY),
    P("<font face='Courier' size='8'>https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
      "{Layer}/default/{Time}/{TileMatrixSet}/{Zoom}/{Row}/{Col}.png</font>", NOTE),
    P("<b>Layer</b> is an identifier such as "
      "<font face='Courier' size='8'>GHRSST_L4_MUR_Sea_Surface_Temperature</font>. "
      "<b>Time</b> is an ISO date — GIBS layers are time-dimensioned, so the same "
      "URL pattern reaches two decades of daily history. <b>best</b> means \"the best "
      "available version of each date\" (near-real-time at first, replaced by the "
      "final science product when it arrives, typically days later). The "
      "<b>GetCapabilities</b> XML document lists every layer with its valid time "
      "ranges, tile matrix set, format, and colormap — the app reads it to know "
      "what exists."),
]

# --------------------------------------------------------- 2. tiles & WMTS
story.append(P("2 · Tiles, WMTS and the map pyramid", H1))
story += [
    P("<b>WMTS</b> (Web Map Tile Service) is the OGC standard GIBS implements. The "
      "world is cut into a pyramid of square tiles: at each zoom level the map is "
      "divided into a grid twice as fine as the level above, and a client fetches "
      "only the handful of tiles its viewport touches. This is the same mechanism "
      "behind every slippy web map since Google Maps."),
    P("GIBS's geographic (EPSG:4326) pyramid has a quirk worth knowing: it does "
      "<i>not</i> start from one world-spanning tile. Level 0 is already <b>2×1</b> "
      "tiles, level 1 is <b>3×2</b>, and the resolution per level is 0.5625/2<super>L</super> "
      "degrees per pixel with 512-pixel tiles. Because 3 is not a power of two, the "
      "right-most and bottom-most tiles at some levels extend past the edge of the "
      "map — they are <i>partial</i> tiles. A subtle but critical rule: a tiling "
      "scheme must declare the <b>full nominal extent</b> of such tiles, not the "
      "clamped visible portion, or the renderer maps pixels to the wrong longitudes. "
      "(Getting this wrong blanked the Pacific in an early version of the app — "
      "hence the custom GIBSGeographicTilingScheme and a regression test pinned to "
      "the published matrix definitions.)"),
    P("<b>Projections.</b> EPSG:4326 (\"geographic\") maps longitude/latitude "
      "directly to x/y — natural for draping onto a 3-D globe. Most 2-D web maps "
      "instead use EPSG:3857 (Web Mercator), which GIBS also offers. National "
      "products often use a local projection — MeteoSwiss grids are delivered in "
      "the Swiss LV95 metre grid, with per-cell longitude/latitude arrays included "
      "so consumers can resample without a projection library."),
]

# ---------------------------------------------------- 3. product levels
story.append(P("3 · How a measurement becomes a map: product levels", H1))
story += [
    P("Satellite data is published in <b>processing levels</b>, and knowing them "
      "explains most of what you see:"),
    table(
        ["Level", "What it is", "Example in the app"],
        [
            ["L0–L1", "Raw instrument counts; calibrated, geolocated radiances.", "—"],
            ["L2", "A geophysical quantity along the satellite's swath. Gaps between "
                   "orbits, stripes, cloud masks.", "MODIS chlorophyll (visible swath seams)"],
            ["L3", "Swaths binned onto a regular global grid for a day / 8 days / a "
                   "month.", "SMAP salinity monthly, MODIS L3 SST"],
            ["L4", "A gap-free <i>analysis</i>: observations from one or more sensors "
                   "blended with interpolation or a model.", "MUR SST 1 km (the default layer)"],
        ],
        [22 * mm, 82 * mm, 62 * mm]),
    Spacer(1, 3 * mm),
    P("Two practical consequences. First, <b>latency</b>: daily products appear "
      "hours to days after observation (the app defaults to \"two days ago\" for "
      "safety), and <b>monthly composites only exist for completed months</b> — "
      "requesting the current month returns nothing, which is why the salinity "
      "layer snaps back to the last complete month. Second, <b>gaps are real</b>: "
      "instruments fail and missions age. SMAP salinity has a 2024 data gap; AMSR2 "
      "sea ice lags its mission's availability windows. A blank layer is more often "
      "a data-availability fact than a bug."),
]

# ---------------------------------------------------- 4. colormaps
story.append(P("4 · Colormaps, legends, and reading values back", H1))
story += [
    P("Every colormapped GIBS layer publishes an XML colour table mapping RGB "
      "triples to physical value ranges (e.g. <font face='Courier' size='8'>"
      "rgb=\"255,0,0\" value=\"[30,31)\"</font> Kelvin). The app uses these tables "
      "three ways: to draw <b>interactive legends</b> (hover a colour, see its "
      "value); to run the <b>value probe</b> (click the globe → sample the rendered "
      "pixel → invert RGB back to a value); and to power <b>computed differences</b> "
      "— invert both dates to numbers per pixel, subtract, and re-colour on a "
      "diverging blue↔red scale."),
    P("Inversion has limits worth understanding. It recovers the <i>bin centre</i>, "
      "not the exact value — a colormap with 250 bins over 40 °C quantises to "
      "≈0.16 °C. It only works where the mapping is one-to-one and the field is "
      "<b>continuous</b> (SST, sea-ice fraction, land temperature). It fails "
      "conceptually for <b>instantaneous, sparse fields</b>: precipitation rate is "
      "log-scaled, transparent below 0.1 mm/hr, and mostly \"no rain right now\" — "
      "differencing two snapshots measures whether it happened to be raining at "
      "overpass time, not a climate signal. That is why the app offers differences "
      "for continuous rasters only, and answers \"how has rainfall changed?\" with "
      "climatologies instead (next section)."),
]

# ---------------------------------------------------- 5. climatology etc
story.append(P("5 · Instantaneous fields, anomalies and climatologies", H1))
story += [
    P("Climate datasets answer three different questions, and mixing them up is "
      "the most common source of confusion:"),
    table(
        ["Kind", "Question it answers", "Examples in the app"],
        [
            ["Instantaneous / daily state", "What is happening <i>now / on date X</i>?",
             "IMERG rain rate, MUR SST, sea ice, chlorophyll, salinity"],
            ["Anomaly", "How does date X differ from the long-term average for that "
             "place and season?", "SST anomalies (MUR25); GISTEMP series (vs 1951–1980)"],
            ["Climatology / normal", "What is <i>typical</i> here, averaged over "
             "decades?", "GPCP precip (global), E-OBS (Europe), MeteoSwiss normal "
             "1991–2020 (Swiss), OISST 1991–2020 SST mean"],
        ],
        [40 * mm, 60 * mm, 66 * mm]),
    Spacer(1, 3 * mm),
    P("An anomaly is always <i>relative to a stated baseline</i> — GISTEMP's "
      "+1.3 °C is \"versus the 1951–1980 mean\", the WMO's current normal period is "
      "1991–2020, and comparing numbers across different baselines silently shifts "
      "them. A climatology, conversely, deliberately hides weather: GPCP's Amazon "
      "cell says ~2,300 mm/yr falls there <i>on average</i>, not whether it rained "
      "yesterday. The app pairs each live field with its baseline where possible: "
      "MUR (today's SST) sits alongside OISST (the 30-year mean it is judged "
      "against); IMERG (rain now) alongside GPCP/E-OBS/MeteoSwiss (rain normally)."),
]

# ---------------------------------------------------- 6. beyond tiles
story.append(P("6 · Beyond tiles: how the non-GIBS data gets in", H1))
story += [
    P("Not everything is a picture on a tile server. The rest of the app's data "
      "arrives through a handful of standard access patterns worth recognising "
      "across the field:"),
    table(
        ["Pattern", "What it is", "Used for"],
        [
            ["NetCDF / HDF", "Self-describing binary arrays with named dimensions "
             "and units; the lingua franca of climate science.", "GPCP, E-OBS, OISST, "
             "MeteoSwiss grids; RAPID transport series"],
            ["OPeNDAP / THREDDS", "Protocols for subsetting remote NetCDF without "
             "downloading whole files.", "OC-CCI ocean colour (catalogued)"],
            ["ERDDAP", "NOAA's data server; slices many datasets into CSV/JSON via "
             "URL queries.", "Argo float positions"],
            ["STAC", "SpatioTemporal Asset Catalog — JSON catalogs of assets with "
             "bounding boxes and times.", "MeteoSwiss open data (data.geo.admin.ch)"],
            ["REST APIs", "Ordinary JSON web services.", "Climate TRACE emitters, GBIF "
             "occurrences"],
            ["Bulk archives", "Tarballs / parquet on plain HTTP.", "RGI v7 glacier "
             "outlines, Hugonnet 2021 elevation-change rates"],
        ],
        [34 * mm, 76 * mm, 56 * mm]),
    Spacer(1, 3 * mm),
    P("The app converts each of these once, offline, into small static JSON "
      "snapshots (<font face='Courier' size='8'>scripts/refresh_data.py</font>), so "
      "the browser never depends on a third-party API being up. Grids with no tile "
      "service are painted client-side by a small canvas renderer (GridProvider) "
      "from those snapshots."),
]

# ---------------------------------------------------- 7. layer table
story.append(PageBreak())
story.append(P("7 · The app's layers at a glance", H1))
story += [
    table(
        ["Layer", "Source & type", "Cadence / period", "Resolution"],
        [
            ["True colour", "VIIRS mosaic (GIBS tiles)", "daily, ~3 h latency", "250 m"],
            ["Sea surface temperature", "MUR L4 analysis (GIBS)", "daily, 2002–", "1 km"],
            ["SST anomalies", "MUR25 L4 (GIBS)", "daily, 2002–", "25 km"],
            ["SST climatology", "NOAA OISST v2.1 (grid)", "1991–2020 mean", "0.25°→1°"],
            ["Precipitation rate", "GPM IMERG V07 (GIBS)", "daily + 30-min, 2000–", "~10 km"],
            ["Precip climatology (global)", "GPCP v2.3 (grid)", "mean annual, 1979–", "2.5°"],
            ["Precip climatology (Europe)", "E-OBS v31 (grid)", "mean annual, 1950–", "0.25°"],
            ["Precip normal (Switzerland)", "MeteoSwiss OGD (grid)", "1991–2020 normal", "~2 km"],
            ["Sea ice concentration", "AMSR2 (GIBS)", "daily, 2012–", "12 km"],
            ["Snow cover", "MODIS NDSI (GIBS)", "daily, 2000–", "500 m"],
            ["Land surface temperature", "MODIS Terra (GIBS)", "daily", "1 km"],
            ["Aerosol optical depth", "MODIS combined (GIBS)", "daily, 2017–", "10 km"],
            ["Chlorophyll-a", "PACE/OCI, NASA Ocean Color (GIBS)", "daily, 2024–", "1 km"],
            ["Sea surface salinity", "SMAP L3 (GIBS)", "monthly, 2015– (2024 gap)", "~60 km"],
            ["Night lights", "VIIRS Black Marble (GIBS)", "annual composite", "500 m"],
            ["Glaciers", "RGI v7 + Hugonnet 2021 (points)", "~2000 outlines; 2000–2020 dh/dt", "274k glaciers"],
            ["Emitters / floats / species", "Climate TRACE · Argo · GBIF (points)", "snapshots", "—"],
        ],
        [44 * mm, 56 * mm, 42 * mm, 24 * mm]),
]

# ---------------------------------------------------- 8. glossary
story.append(P("8 · Small glossary", H1))
story += [
    table(
        ["Term", "Meaning"],
        [
            ["WMTS / WMS", "OGC standards for serving maps as pre-cut tiles (WMTS) or "
             "on-demand rendered images (WMS)."],
            ["EPSG code", "Numeric identifier of a coordinate reference system "
             "(4326 = lon/lat; 3857 = Web Mercator; 2056 = Swiss LV95)."],
            ["Granule", "One file of satellite data — typically one orbit segment or "
             "one day-tile of a product."],
            ["Analysis (L4)", "A gap-free gridded field produced by blending "
             "observations, e.g. with optimal interpolation (the OI in OISST)."],
            ["Climatology / normal", "A multi-decade average; WMO normals span 30 "
             "years, currently 1991–2020."],
            ["Anomaly", "Departure of an observation from a stated climatological "
             "baseline."],
            ["Composite", "A product built by aggregating a period (8-day, monthly); "
             "exists only once the period is complete."],
            ["NRT vs science quality", "Near-real-time products arrive in hours but "
             "are later replaced by re-processed, better-calibrated versions — "
             "GIBS's 'best' endpoint always shows the best one available."],
            ["Ensemble", "Several independent estimates of the same field; their "
             "spread is a measure of structural uncertainty (the app's SST "
             "ensemble layer)."],
            ["Geodetic mass balance", "Glacier change measured by differencing "
             "elevation models over time (Hugonnet 2021's dh/dt, m/yr)."],
        ],
        [42 * mm, 124 * mm]),
    Spacer(1, 6 * mm),
    P("<i>earth · open climate data on a globe · "
      "github.com/blauewelt/earth · July 2026</i>", NOTE),
]

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=18 * mm, bottomMargin=18 * mm,
                        title="Open Climate Data on a Globe — Primer",
                        author="earth project")
doc.build(story)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
