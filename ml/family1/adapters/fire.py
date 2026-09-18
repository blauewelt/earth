"""FIRMS active fire — every satellite detection of a burning surface, MODIS
Collection 6.1 at 1 km and VIIRS at 375 m (family 1.0.tf, tier P, E-082
wave 4).

PLAIN ENGLISH. When a satellite's thermal bands see a pixel much hotter than
its neighbours, NASA's Fire Information for Resource Management System records
a DETECTION: a point, an instant, how bright the pixel was, how much power the
fire radiated, and how sure the algorithm is. Half a billion of them exist
since November 2000. They are the land family's carbon read-out — a fire is
the fastest way a landscape loses the carbon a forest spent decades
accumulating — and the only store in this family whose rows are events rather
than samples of a continuing quantity.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (the documentation and the
endpoints, keyless; the data needs FIRMS_MAP_KEY and was read on a hosted
runner). THE ARCHIVE DOWNLOAD PAGE IS NOT SCRIPTABLE and the API is:
  `https://firms.modaps.eosdis.nasa.gov/download/` (read 2026-09-18) offers
  shapefiles, CSV and JSON "older than the last 7 days" behind an Earthdata
  or e-mail-code login, and answers a request by EMAILING a link — "Once the
  request has been processed, you will receive an email with instructions on
  how to download your data". A human is in that loop by design, so it is not
  what this adapter uses.
  The AREA API is, and it covers the whole record:
  `https://firms.modaps.eosdis.nasa.gov/api/area/csv/<MAP_KEY>/<SOURCE>/
  world/<DAY_RANGE>/<DATE>` returns the detections of [DATE, DATE +
  DAY_RANGE - 1] for a bounding box, where `world` is [-180, -90, 180, 90]
  and DAY_RANGE is 1..5. Which dates each source holds is not guessed either:
  `https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/<MAP_KEY>/
  all` answers one row per source with its own `min_date` and `max_date`
  (the documented example reads MODIS_SP 2000-11-01 .. 2025-01-31 and
  MODIS_NRT 2025-02-01 .. 2025-06-06), and the index stage reads it.
  The MAP_KEY is free and its documented limit is 5,000 transactions per
  ten-minute window, which a year of five-day requests (73 per source) is
  nowhere near.

  Temporal coverage, from the download page's own text: MODIS Collection 6.1
  from 2000-11-01, VIIRS Suomi-NPP from 2012-01-20, VIIRS NOAA-20 from
  2018-04-01, VIIRS NOAA-21 from 2024-01-17.

SP AND NRT ARE ONE RECORD, SPLIT AT A DATE THE ARCHIVE NAMES. Each instrument
publishes a Standard-Processing series (science quality, several months
behind) and a Near-Real-Time one (the tail). They are consecutive, not
overlapping, so the adapter gives EVERY DAY TO EXACTLY ONE SOURCE — SP where
SP has it, NRT after SP's `max_date` — and refuses if the two leave a gap
between them. No detection is therefore fetched twice, and none is quietly
dropped.

WHAT IS STORED: C = 4, one row per detection.
  brightness   K     MODIS `brightness` (the 4 um channel's brightness
                     temperature) or VIIRS `bright_ti4` (the I-4 band's)
  frp          MW    fire radiative power
  confidence   %     MODIS' 0..100 confidence. NaN FOR VIIRS — see below
  daynight     flag  1 for a daytime overpass, 0 for a night one

THE CONFIDENCE CHANNEL IS EMPTY FOR VIIRS ON PURPOSE. MODIS publishes a
confidence PERCENTAGE; the VIIRS 375 m product publishes three CLASSES, `l`,
`n` and `h`, and there is no percentage to put in the channel. Placing the
classes at 0, 50 and 100 would be inventing three numbers the producer never
published, which §5 of the note forbids ("nothing is homogenised"), so the
channel is NaN for a VIIRS row — "a channel not measured is NaN" — and the
class is kept exactly, in `qc`. A consumer that wants one usable number for
both instruments therefore has to make that choice itself, with the class in
front of it.

`qc`, ONE uint8 PER DETECTION:
  bits 0-2   the platform code: 1 Terra/MODIS, 2 Aqua/MODIS, 3 Suomi-NPP,
             4 NOAA-20, 5 NOAA-21 (0 is never written)
  bits 3-4   VIIRS confidence: 1 low (`l`), 2 nominal (`n`), 3 high (`h`),
             0 not applicable — a MODIS row, whose percentage is in the
             `confidence` channel
  bit  5     the row came from the NRT series rather than from Standard
             Processing, so a consumer can hold the tail to a different
             standard than the science-quality record

`platform` is `platform_hash(<canonical satellite> + " " + instrument)` and
`platforms.json` carries, for each one, the satellite, the instrument, every
spelling the archive uses for it, the nominal pixel size and ITS OWN
`log2_fp`. That last field is why the store can hold both instruments
honestly: the footprint of a MODIS detection is 1 km
(log2(1/27.83) = -4.798) and of a VIIRS one is 375 m
(log2(0.375/27.83) = -6.214), the store's single `log2_fp` is the MODIS
value, and the per-platform table is where the real number lives. The hash is
of the CANONICAL name rather than of the row's own text because FIRMS writes
`Terra` in one product and `N20` in another: hashing the raw string would
make one satellite two platforms and leave the second with nothing in
platforms.json to describe it. A satellite string the table does not know at
all is a REFUSAL, and the spellings actually seen are counted
(`rows_by_spelling`).

THE TRUE log2_fp IS USED, BELOW THE -4 POINT LABEL. E-078's token grid labels
point supports at -4; a 1 km MODIS pixel is -4.798 and a 375 m VIIRS pixel
-6.214, and both are written as measured. A real support that is finer than
the finest label is a fact about the instrument, not something to round up to
the nearest tick.

SIZE. The note's ledger row is MODIS 1.1e8 detections, VIIRS about 4e8 (2023
measured: 4.66 M MODIS and 21.75 M VIIRS S-NPP), 20 GB stored. The probe
measures a real month; `ml/family1/probes/fire_2023-08.json` is the number
that replaces the estimate.

`FIRE_INSTRUMENTS` ("modis,snpp,noaa20,noaa21" — SOURCE FAMILIES, because
`MODIS_SP` carries Terra and Aqua in one CSV) selects which of them a run
fetches; the default is all of them, and it is read at construction time so
the probe's own adapter sees it.
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

BASE = "https://firms.modaps.eosdis.nasa.gov/api/"
AREA = BASE + "area/csv/"
AVAIL = BASE + "data_availability/csv/"
WORLD = "world"
MAX_DAY_RANGE = 5
FILL = -9999.0

# THE UNIT OF FETCHING IS A SOURCE, NOT A SATELLITE. `MODIS_SP` carries Terra
# AND Aqua in one CSV, so asking for it once per satellite would fetch every
# MODIS window twice and store every MODIS detection twice. A "chain" is
# therefore one SOURCE PAIR (Standard Processing plus its Near-Real-Time
# tail), and the satellites it can carry are listed beside it so a row whose
# satellite column does not belong to the source it came from is caught.
CHAINS = {
    "modis": {"sp": "MODIS_SP", "nrt": "MODIS_NRT", "instrument": "MODIS",
              "pixel_km": 1.0, "satellites": ("Terra", "Aqua"),
              "first": "2000-11-01"},
    "snpp": {"sp": "VIIRS_SNPP_SP", "nrt": "VIIRS_SNPP_NRT",
             "instrument": "VIIRS", "pixel_km": 0.375,
             "satellites": ("N", "1", "NPP", "SNPP", "Suomi-NPP"),
             "first": "2012-01-20"},
    "noaa20": {"sp": "VIIRS_NOAA20_SP", "nrt": "VIIRS_NOAA20_NRT",
               "instrument": "VIIRS", "pixel_km": 0.375,
               "satellites": ("N20", "NOAA-20", "J1"),
               "first": "2018-04-01"},
    "noaa21": {"sp": "VIIRS_NOAA21_SP", "nrt": "VIIRS_NOAA21_NRT",
               "instrument": "VIIRS", "pixel_km": 0.375,
               "satellites": ("N21", "NOAA-21", "J2"),
               "first": "2024-01-17"},
}
# THE PLATFORM TABLE: one row per (canonical satellite name, instrument),
# with every spelling the archive has been seen to use for it.
#
# THE HASH IS OF THE CANONICAL NAME, NOT OF THE ROW'S OWN TEXT. FIRMS writes
# `Terra` in one product and `N20` in another, and a future file could write
# `TERRA`; hashing the raw string would then make one satellite two platforms
# and leave the second with no entry in platforms.json. The adapter therefore
# resolves the spelling to its canonical name (case-insensitively), hashes
# THAT, and counts the raw spellings it saw — so the hash is stable across
# products and every platform in the store has an entry describing it.
PLATFORM_LIST = (
    # canonical, instrument, qc code, source family, spellings seen
    ("Terra", "MODIS", 1, "modis", ("Terra", "T")),
    ("Aqua", "MODIS", 2, "modis", ("Aqua", "A")),
    ("Suomi-NPP", "VIIRS", 3, "snpp", ("N", "1", "NPP", "SNPP",
                                       "Suomi-NPP")),
    ("NOAA-20", "VIIRS", 4, "noaa20", ("N20", "NOAA-20", "J1")),
    ("NOAA-21", "VIIRS", 5, "noaa21", ("N21", "NOAA-21", "J2")),
)
# (spelling, instrument) upper-cased -> (canonical, code, source family)
PLATFORMS = {(s.upper(), inst.upper()): (name, code, chain)
             for (name, inst, code, chain, spellings) in PLATFORM_LIST
             for s in spellings}
# MODIS and VIIRS name the 4 um brightness-temperature column differently
BRIGHT = ("brightness", "bright_ti4")
CONF_CLASS = {"l": 1, "n": 2, "h": 3}
NEEDED = ("latitude", "longitude", "acq_date", "acq_time", "satellite",
          "instrument", "confidence", "frp", "daynight")


class FormatError(ValueError):
    """A FIRMS answer that does not look like the API documented."""


def platform_id(satellite, instrument):
    return f"{satellite} {instrument}"


def parse_csv(raw, want_days, chain, is_nrt, counts):
    """A FIRMS area CSV -> columns, BY COLUMN NAME and never by position.

    `want_days` is the set of dates this request asked for; a row outside it
    is counted and dropped (the API answers a whole DAY_RANGE and a window
    can be clipped at a source boundary or at the year's edge).
    """
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        raise FormatError("an empty body — not even a header")
    head = [h.strip() for h in lines[0].split(",")]
    if "latitude" not in head:
        raise FormatError(f"the first line is not a FIRMS header: "
                          f"{lines[0][:200]!r}")
    col = {h: i for i, h in enumerate(head)}
    bright = next((b for b in BRIGHT if b in col), None)
    if bright is None:
        raise FormatError(f"no brightness column in {head} (expected one of "
                          f"{BRIGHT})")
    missing = [k for k in NEEDED if k not in col]
    if missing:
        raise FormatError(f"the header is missing {missing}: {head}")
    n = len(lines) - 1
    t = np.zeros(n, np.int64)
    lat = np.zeros(n, np.float64)
    lon = np.zeros(n, np.float64)
    vals = np.full((n, 4), np.nan, np.float64)
    qc = np.zeros(n, np.uint8)
    plat = np.zeros(n, np.int64)
    k = 0
    bad_day = bad_row = 0
    sats = {}
    ci, cb, cf = col["confidence"], col[bright], col["frp"]
    cd, ct = col["acq_date"], col["acq_time"]
    cs, cin, cdn = col["satellite"], col["instrument"], col["daynight"]
    clat, clon = col["latitude"], col["longitude"]
    ncol = len(head)
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split(",")
        if len(f) != ncol:
            bad_row += 1
            continue
        try:
            y, mo, d = (int(x) for x in f[cd].split("-"))
            day = dt.date(y, mo, d)
        except (ValueError, IndexError):
            bad_row += 1
            continue
        if day not in want_days:
            bad_day += 1
            continue
        sat = f[cs].strip()
        inst = f[cin].strip()
        got = PLATFORMS.get((sat.upper(), inst.upper()))
        if got is None:
            raise FormatError(
                f"satellite/instrument {sat!r}/{inst!r} is not in the "
                f"adapter's platform table {sorted(PLATFORMS)} — a new "
                f"platform is a REFUSAL, because platforms.json must "
                f"describe every platform the store holds")
        canon, code, name = got
        if name != chain:
            raise FormatError(
                f"source {chain!r} answered a {sat!r}/{inst!r} row, which "
                f"belongs to {name!r} — two sources would then both carry it "
                f"and the store would hold it twice")
        c = CHAINS[name]
        try:
            hhmm = int(f[ct])
            lat[k] = float(f[clat])
            lon[k] = float(f[clon])
        except ValueError:
            bad_row += 1
            continue
        t[k] = f10b.seconds_since_epoch(day) + (hhmm // 100) * 3600 \
            + (hhmm % 100) * 60
        vals[k, 0] = _num(f[cb])
        vals[k, 1] = _num(f[cf])
        conf = f[ci].strip()
        q = code & 0b111
        if c["instrument"] == "MODIS":
            vals[k, 2] = _num(conf)
        else:
            cls = CONF_CLASS.get(conf.lower())
            if cls is None:
                raise FormatError(
                    f"VIIRS confidence {conf!r} is not one of l / n / h — "
                    f"the class vocabulary changed and the qc packing must "
                    f"be re-read before anything is built")
            q |= cls << 3
        dn = f[cdn].strip().upper()[:1]
        if dn == "D":
            vals[k, 3] = 1.0
        elif dn == "N":
            vals[k, 3] = 0.0
        elif dn:
            raise FormatError(f"daynight {f[cdn]!r} is neither D nor N")
        if is_nrt:
            q |= 1 << 5
        qc[k] = q
        plat[k] = f10b.platform_hash(platform_id(canon, inst))
        seen = f"{sat}/{inst} -> {canon}"
        sats[seen] = sats.get(seen, 0) + 1
        k += 1
    counts["rows_read"] = counts.get("rows_read", 0) + (len(lines) - 1)
    if bad_day:
        counts["rows_outside_requested_days"] = \
            counts.get("rows_outside_requested_days", 0) + bad_day
    if bad_row:
        counts["rows_unparsable"] = counts.get("rows_unparsable", 0) + bad_row
    for pid, m in sats.items():
        counts.setdefault("rows_by_spelling", {})
        counts["rows_by_spelling"][pid] = \
            counts["rows_by_spelling"].get(pid, 0) + m
    return {"t": t[:k], "lat": lat[:k], "lon": lon[:k], "values": vals[:k],
            "qc": qc[:k], "platform": plat[:k]}, counts


def _num(s):
    s = s.strip()
    if not s:
        return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return np.nan if v == FILL else v


class FireAdapter(f10b.SourceAdapter):
    store = "fire"
    title = ("Active fire detections, FIRMS MODIS C6.1 (1 km) and VIIRS "
             "S-NPP / NOAA-20 / NOAA-21 (375 m)")
    family = "1tf"
    distribution = "public"
    licence = {
        "name": "NASA open data (no restrictions)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("NASA FIRMS. MODIS Collection 6.1 and VIIRS 375 m "
                        "active fire products, "
                        "doi:10.5067/FIRMS/MODIS/MCD14DL.NRT.0061 and "
                        "doi:10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002. We "
                        "acknowledge the use of data products from NASA's "
                        "Fire Information for Resource Management System "
                        "(FIRMS), part of NASA's Earth Observing System Data "
                        "and Information System (EOSDIS)."),
        "terms": ("earthdata.nasa.gov's data-information policy, read "
                  "2026-09-18: NASA's Earth science data are open, full and "
                  "without restriction and free of charge; FIRMS asks that "
                  "its products be acknowledged in the sentence above"),
    }
    channels = (("brightness", "K", 200.0, 600.0),
                ("frp", "MW", 0.0, 100000.0),
                ("confidence", "percent", 0.0, 100.0),
                ("daynight", "flag", 0.0, 1.0))
    # THE TRUE SUPPORT, below E-078's -4 point label: a 1 km MODIS pixel.
    # Every platform's own value is in platforms.json.
    log2_fp = float(np.log2(1.0 / 27.83))          # -4.798
    log2_dt = -4.0                                 # an instantaneous overpass
    per_year = True
    first_year = 2000
    time_dtype = "int32"
    credentials = ("FIRMS_MAP_KEY",)
    platform_meta = True
    qc_policy = (
        "qc is one uint8 per detection: bits 0-2 the platform code (1 "
        "Terra/MODIS, 2 Aqua/MODIS, 3 Suomi-NPP/VIIRS, 4 NOAA-20/VIIRS, 5 "
        "NOAA-21/VIIRS; 0 is never written), bits 3-4 the VIIRS confidence "
        "class (1 low, 2 nominal, 3 high, 0 not applicable), bit 5 set when "
        "the row came from the Near-Real-Time series rather than from "
        "Standard Processing. The `confidence` CHANNEL carries MODIS' 0..100 "
        "percentage and is NaN for a VIIRS row, because the VIIRS 375 m "
        "product publishes three classes and no percentage: placing them at "
        "0, 50 and 100 would invent three numbers the producer never "
        "published (family1tf.tex §5 -- nothing is homogenised), so the "
        "class is kept exactly in qc instead. A value outside a channel's "
        "physical bound becomes NaN and is counted, never clipped. A "
        "satellite/instrument pair the adapter's platform table does not "
        "know, and a VIIRS confidence that is not l / n / h, are both "
        "REFUSALS: platforms.json must describe every platform the store "
        "holds, and a changed vocabulary must be read by a human before a "
        "store is built on it")
    sources = (AREA + "<MAP_KEY>/<SOURCE>/world/<DAY_RANGE>/<DATE>",
               AVAIL + "<MAP_KEY>/all")
    verified = (
        "2026-09-18 from the sandbox, keyless (the documentation and the "
        "endpoint shapes; the data itself needs FIRMS_MAP_KEY and was read "
        "on a GitHub-hosted runner): firms.modaps.eosdis.nasa.gov/api/area/ "
        "for the URL form, the 1..5 DAY_RANGE limit, the `world` bounding "
        "box and the eight source ids; "
        "firms.modaps.eosdis.nasa.gov/content/academy/data_api/"
        "firms_api_use.html for the data_availability endpoint (whose "
        "documented output reads MODIS_SP 2000-11-01..2025-01-31 and "
        "MODIS_NRT 2025-02-01..) and for the exact CSV columns of the MODIS "
        "and VIIRS products; and firms.modaps.eosdis.nasa.gov/download/ for "
        "the archive route this adapter does NOT use, whose own page says a "
        "request is answered by e-mail. The 5,000-transactions-per-ten-"
        "minutes MAP_KEY limit is quoted on the API page")
    notes = (
        "Every day belongs to exactly one source: Standard Processing where "
        "it has it, Near-Real-Time after SP's own max_date, both read from "
        "the data_availability endpoint rather than assumed, so no detection "
        "is fetched twice and a gap between the two is a refusal. The "
        "`confidence` channel is MODIS-only by design (VIIRS publishes three "
        "classes, which are in qc). The store's log2_fp is the MODIS 1 km "
        "pixel (-4.798); a VIIRS detection is 375 m (-6.214) and every "
        "platform's own value is in platforms.json, which is also where the "
        "satellite, the instrument and the nominal pixel size live.")
    smoke_window = ("2023-08-01", "2023-08-31")
    smoke_probe_month = "2023-08"
    fetch_month_scope = "month"

    YIELD_ROWS = 1_000_000
    INSTRUMENTS_ENV = "FIRE_INSTRUMENTS"

    def __init__(self):
        want = (os.environ.get(self.INSTRUMENTS_ENV) or "").strip()
        if want:
            picked = [x.strip() for x in want.split(",") if x.strip()]
            bad = [x for x in picked if x not in CHAINS]
            if bad or not picked:
                sys.exit(f"REFUSING fire: {self.INSTRUMENTS_ENV}={want!r} "
                         f"names {bad or 'nothing'}; the instruments are "
                         f"{sorted(CHAINS)}")
            self.instruments = picked
            self.notes = (f"{self.notes}\nINSTRUMENTS: "
                          f"{self.INSTRUMENTS_ENV} selected {picked} instead "
                          f"of all of {sorted(CHAINS)}.")
        else:
            self.instruments = sorted(CHAINS)
        self._avail = None

    # ------------------------------------------------------- availability --
    def key(self):
        k = os.environ.get("FIRMS_MAP_KEY", "")
        if not k:
            sys.exit("REFUSING fire: FIRMS_MAP_KEY is not set (it comes from "
                     "a repository secret on a GitHub-hosted runner only, "
                     "ml/CLAUDE.md §6)")
        return k

    def availability(self, ctx):
        """{source id: (min date, max date)} — the archive's own answer."""
        if self._avail is not None:
            return self._avail
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "fire", "data_availability.csv")
            if not os.path.exists(p):
                sys.exit(f"REFUSING fire: no {p} — the smoke's synthetic "
                         f"availability table is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
        else:
            raw, why = cm.get_bytes(AVAIL + self.key() + "/all",
                                    attempts=ctx.a.attempts)
            if raw is None:
                sys.exit(f"REFUSING fire: the data_availability endpoint "
                         f"answered {why} — without it the record's own ends "
                         f"are unknown and nothing may be guessed")
        out = {}
        lines = raw.decode("utf-8", "replace").splitlines()
        if not lines or "data_id" not in lines[0]:
            sys.exit(f"REFUSING fire: data_availability did not answer a CSV "
                     f"with a data_id column: {lines[:1]}")
        head = [h.strip() for h in lines[0].split(",")]
        col = {h: i for i, h in enumerate(head)}
        for k in ("data_id", "min_date", "max_date"):
            if k not in col:
                sys.exit(f"REFUSING fire: data_availability has no {k!r} "
                         f"column ({head})")
        for line in lines[1:]:
            f = [x.strip() for x in line.split(",")]
            if len(f) < len(head) or not f[col["data_id"]]:
                continue
            try:
                lo = f10b.parse_date(f[col["min_date"]])
                hi = f10b.parse_date(f[col["max_date"]])
            except (ValueError, IndexError):
                continue
            out[f[col["data_id"]]] = (lo, hi)
        if not out:
            sys.exit("REFUSING fire: data_availability listed no source at "
                     "all — an empty listing is a refusal")
        self._avail = out
        return out

    def coverage(self, ctx):
        """{instrument: [(source id, lo, hi, is_nrt), ...]}, gaps REFUSED."""
        av = self.availability(ctx)
        out = {}
        for name in self.instruments:
            c = CHAINS[name]
            spans = []
            sp = av.get(c["sp"])
            nrt = av.get(c["nrt"])
            if sp is None and nrt is None:
                sys.exit(f"REFUSING fire: neither {c['sp']} nor {c['nrt']} is "
                         f"in the data_availability table ({sorted(av)}) — "
                         f"the archive no longer serves {name} the way this "
                         f"adapter was written")
            if sp is not None:
                spans.append((c["sp"], sp[0], sp[1], False))
            if nrt is not None:
                lo = nrt[0]
                if sp is not None:
                    lo = max(lo, sp[1] + dt.timedelta(days=1))
                    if nrt[0] > sp[1] + dt.timedelta(days=1):
                        sys.exit(
                            f"REFUSING fire: {c['sp']} ends {sp[1]} and "
                            f"{c['nrt']} starts {nrt[0]} — the two leave "
                            f"{(nrt[0] - sp[1]).days - 1} day(s) of the "
                            f"record with no source. A hole in the middle of "
                            f"a published record is a listing problem until "
                            f"proven otherwise")
                if lo <= nrt[1]:
                    spans.append((c["nrt"], lo, nrt[1], True))
            out[name] = spans
        return out

    # --------------------------------------------------------------- index --
    def index(self, ctx):
        cov = self.coverage(ctx)
        av = self.availability(ctx)
        first = min(s[1] for v in cov.values() for s in v)
        last = max(s[2] for v in cov.values() for s in v)
        out = {"dataset": "FIRMS active fire, MODIS C6.1 and VIIRS 375 m",
               "url": AREA + "<MAP_KEY>/<SOURCE>/world/<DAY_RANGE>/<DATE>",
               "availability": {k: [str(v[0]), str(v[1])]
                                for k, v in sorted(av.items())},
               "instruments": {},
               "record": [str(first), str(last)],
               "record_days": [str(first), str(last)],
               "day_range": MAX_DAY_RANGE,
               "requests_per_year_per_instrument":
                   -(-366 // MAX_DAY_RANGE)}
        for name, spans in sorted(cov.items()):
            c = CHAINS[name]
            out["instruments"][name] = {
                "satellites": list(c["satellites"]),
                "instrument": c["instrument"],
                "pixel_km": c["pixel_km"],
                "log2_fp": round(float(np.log2(c["pixel_km"] / 27.83)), 4),
                "qc_codes": sorted({code for (_n, _i, code, ch, _s)
                                    in PLATFORM_LIST if ch == name}),
                "documented_first_day": c["first"],
                "spans": [{"source": s[0], "from": str(s[1]),
                           "to": str(s[2]), "nrt": s[3]} for s in spans]}
        # ONE REAL REQUEST, parsed, so the columns are checked at index time
        probe = self.index_probe(ctx, cov)
        out["first_request"] = probe
        return out

    def index_probe(self, ctx, cov):
        """THE FIRST WINDOW `windows()` would fetch, fetched and parsed.

        Not a window of its own: asking for one the fetch never asks for
        would test a request shape nothing else uses, and on a local archive
        (the smoke) it would not exist at all.
        """
        name = sorted(cov)[0]
        src, lo, hi, is_nrt = cov[name][0]
        wins = [w for w in self.windows(ctx, lo.year) if w[0] == src]
        if not wins:
            sys.exit(f"REFUSING fire: {src} covers {lo}..{hi} and "
                     f"windows({lo.year}) produced nothing for it")
        src, day, n, name, is_nrt = wins[0]
        raw, url, why = self.request(ctx, src, day, n)
        if raw is None:
            sys.exit(f"REFUSING fire: the first request {url} answered "
                     f"{why} — nothing has been built")
        counts = {}
        want = {day + dt.timedelta(days=k) for k in range(n)}
        cols, counts = parse_csv(raw, want, name, is_nrt, counts)
        return {"url": url, "source": src, "date": str(day), "days": n,
                "bytes": len(raw),
                "rows": int(cols["t"].size), "counts": counts,
                "columns_checked": list(NEEDED) + ["brightness|bright_ti4"]}

    # ----------------------------------------------------------- one window -
    def request(self, ctx, source, day, days):
        """(bytes, url, why) for one area-API window."""
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, "fire",
                             f"{source}_{day}_{days}.csv")
            if not os.path.exists(p):
                return None, p, "the smoke's synthetic window is missing"
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            return raw, p, "ok"
        url = f"{AREA}{self.key()}/{source}/{WORLD}/{int(days)}/{day}"
        raw, why = cm.get_bytes(url, attempts=ctx.a.attempts)
        shown = url.replace(self.key(), "<MAP_KEY>")
        if raw is None:
            return None, shown, why
        return raw, shown, "ok"

    def windows(self, ctx, year):
        """[(source, first day, n days, instrument, is_nrt)] for `year`."""
        cov = self.coverage(ctx)
        y0, y1 = dt.date(year, 1, 1), dt.date(year, 12, 31)
        out = []
        for name in sorted(cov):
            for (src, lo, hi, is_nrt) in cov[name]:
                a, b = max(lo, y0), min(hi, y1)
                d = a
                while d <= b:
                    n = min(MAX_DAY_RANGE, (b - d).days + 1)
                    out.append((src, d, n, name, is_nrt))
                    d += dt.timedelta(days=n)
        return out

    # --------------------------------------------------------- the contract -
    def fetch_year(self, ctx, year):
        yield from self._fetch(ctx, self.windows(ctx, year), str(year))

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        d0 = dt.date(year, month, 1)
        d1 = (dt.date(year + (month == 12), month % 12 + 1, 1)
              - dt.timedelta(days=1))
        wins = [w for w in self.windows(ctx, year)
                if w[1] <= d1 and w[1] + dt.timedelta(days=w[2] - 1) >= d0]
        clipped = []
        for (src, d, n, name, is_nrt) in wins:
            a = max(d, d0)
            b = min(d + dt.timedelta(days=n - 1), d1)
            clipped.append((src, a, (b - a).days + 1, name, is_nrt))
        yield from self._fetch(ctx, clipped, f"{year}-{month:02d}")

    def _fetch(self, ctx, wins, label):
        """Data batches with NO counts, then ONE counts-only batch at the end.

        The framework merges whatever `counts` a batch carries, once. Handing
        them over with the FIRST batch — which is what most adapters do,
        because they compute a whole year before yielding — would publish the
        tallies of the first five-day window and call them the year's. This
        adapter learns as it goes, so the ledger is yielded last.
        """
        counts = {"windows": len(wins)}
        t0 = time.time()
        for i, (src, day, n, name, is_nrt) in enumerate(wins, 1):
            want = {day + dt.timedelta(days=k) for k in range(n)}
            raw, url, why = self.request(ctx, src, day, n)
            if raw is None:
                ctx.note_absent(f"{src} {day} +{n}d", f"{url}: {why}")
                continue
            try:
                cols, counts = parse_csv(raw, want, name, is_nrt, counts)
            except FormatError as e:
                sys.exit(f"REFUSING fire: {url}: {e}")
            counts["bytes_windows"] = counts.get("bytes_windows", 0) + len(raw)
            if cols["t"].size == 0:
                counts["windows_with_no_detection"] = \
                    counts.get("windows_with_no_detection", 0) + 1
                continue
            t = cols["t"]
            inside = (t >= ctx.t_lo) & (t <= ctx.t_hi)
            nout = int((~inside).sum())
            if nout:
                counts["rows_outside_window"] = \
                    counts.get("rows_outside_window", 0) + nout
            v = cols["values"][inside]
            oob = self.mask_bounds(v)
            if oob:
                for k, m in oob.items():
                    counts.setdefault("out_of_bounds", {})
                    counts["out_of_bounds"][k] = \
                        counts["out_of_bounds"].get(k, 0) + m
            rows = self.pack(t[inside], cols["lat"][inside],
                             cols["lon"][inside], v,
                             cols["platform"][inside], cols["qc"][inside])
            counts["rows_kept"] = counts.get("rows_kept", 0) \
                + int(t[inside].size)
            counts["fetch_seconds"] = round(time.time() - t0, 1)
            yield f"{label} {src} {day}+{n}d", rows, None
            if i % 10 == 0:
                ctx.prog.item(f"fire {label} {src} {day}", None,
                              {"windows": i, "of": len(wins),
                               "rows": counts.get("rows_kept", 0)})
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        yield f"{label} ledger", None, counts

    def platforms(self, ctx):
        """Every platform the store CAN hold, with its own footprint."""
        out = {}
        for (name, inst, code, chain, spellings) in PLATFORM_LIST:
            c = CHAINS[chain]
            pid = platform_id(name, inst)
            out[f10b.platform_hash(pid)] = {
                "id": pid, "lat": None, "lon": None,
                "source_family": chain,
                "satellite": name, "instrument": inst,
                "spellings_accepted": list(spellings),
                "pixel_km": c["pixel_km"],
                "log2_fp": round(float(np.log2(c["pixel_km"] / 27.83)), 4),
                "qc_code": code,
                "documented_first_day": c["first"],
                "note": ("the store's own log2_fp is the MODIS 1 km pixel; "
                         "THIS platform's footprint is the log2_fp above. "
                         "The hash is platform_hash(canonical satellite + "
                         "' ' + instrument), so every spelling the archive "
                         "uses for one satellite resolves to one platform")}
        return out

    def extra_meta(self, ctx, dest, N, values):
        return {"footprint_by_platform": {
            str(k): {"id": v["id"], "pixel_km": v["pixel_km"],
                     "log2_fp": v["log2_fp"]}
            for k, v in self.platforms(ctx).items()}}

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi):
        truth = make_smoke_sources(root, d_lo, d_hi)
        # `make_smoke_sources` sets FIRE_INSTRUMENTS, which this instance read
        # at construction time — before the archive existed. Re-read it, the
        # way chirps05 re-reads its smoke grid.
        self.__init__()
        return truth


# ================================================================== smoke ==
SMOKE_AVAIL = {
    "MODIS_SP": ("2023-08-01", "2023-08-20"),
    "MODIS_NRT": ("2023-08-21", "2023-08-31"),
    "VIIRS_SNPP_SP": ("2023-08-01", "2023-08-31"),
}
SMOKE_INSTRUMENTS = ("modis", "snpp")


def _rows_for(source, day, n):
    """The deterministic detections of one window."""
    rows = []
    for k in range(n):
        d = day + dt.timedelta(days=k)
        seed = d.toordinal()
        if source.startswith("MODIS"):
            for j, (sat, conf) in enumerate((("Terra", 77), ("Aqua", 42))):
                rows.append({
                    "latitude": -20.0 + 0.5 * ((seed + j) % 40),
                    "longitude": 100.0 + 0.25 * ((seed + 2 * j) % 60),
                    "brightness": 310.0 + (seed + j) % 40,
                    "scan": 1.03, "track": 1.02,
                    "acq_date": str(d),
                    "acq_time": f"{(600 + 13 * j) % 2400:04d}",
                    "satellite": sat, "instrument": "MODIS",
                    "confidence": conf, "version": "6.1",
                    "bright_t31": 292.0, "frp": 10.0 + j,
                    "daynight": "D" if j == 0 else "N"})
        else:
            for j, cls in enumerate(("l", "n", "h")):
                rows.append({
                    "latitude": 5.0 + 0.1 * ((seed + j) % 50),
                    "longitude": -60.0 - 0.2 * ((seed + j) % 30),
                    "bright_ti4": 330.0 + (seed + j) % 20,
                    "scan": 0.4, "track": 0.37,
                    "acq_date": str(d),
                    "acq_time": f"{(100 + 7 * j) % 2400:04d}",
                    "satellite": "N", "instrument": "VIIRS",
                    "confidence": cls, "version": "2.0",
                    "bright_ti5": 295.0, "frp": 2.0 + j,
                    "daynight": "N" if j else "D"})
    return rows


MODIS_HEAD = ("latitude,longitude,brightness,scan,track,acq_date,acq_time,"
              "satellite,instrument,confidence,version,bright_t31,frp,"
              "daynight")
VIIRS_HEAD = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
              "satellite,instrument,confidence,version,bright_ti5,frp,"
              "daynight")


def window_csv(source, day, n):
    head = MODIS_HEAD if source.startswith("MODIS") else VIIRS_HEAD
    cols = head.split(",")
    lines = [head]
    for r in _rows_for(source, day, n):
        lines.append(",".join(str(r[c]) for c in cols))
    return "\n".join(lines) + "\n"


class _SmokeCtx:
    """The two things `availability` and `windows` ask a context for."""

    def __init__(self, root):
        self.source_dir = root

        class _A:
            attempts = 1
        self.a = _A()

    def count_bytes(self, n):
        return None


def make_smoke_sources(root, d_lo, d_hi):
    """The archive in its real shape: an availability CSV and one CSV per
    window, in the API's own column order.

    The windows are `FireAdapter.windows()`' OWN answer, so the synthetic
    archive and the fetch can never disagree about where a source boundary
    falls or how a five-day window is clipped at it.
    """
    base = os.path.join(root, "fire")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "data_availability.csv"), "w") as fh:
        fh.write("data_id,min_date,max_date\n")
        for k, (a, b) in SMOKE_AVAIL.items():
            fh.write(f"{k},{a},{b}\n")
    os.environ["FIRE_INSTRUMENTS"] = ",".join(SMOKE_INSTRUMENTS)
    ad = FireAdapter()
    ctx = _SmokeCtx(root)
    wins = []
    for y in range(d_lo.year, d_hi.year + 1):
        wins += [w for w in ad.windows(ctx, y)
                 if w[1] <= d_hi and
                 w[1] + dt.timedelta(days=w[2] - 1) >= d_lo]
    for (src, day, n, name, is_nrt) in wins:
        with open(os.path.join(base, f"{src}_{day}_{n}.csv"), "w") as fh:
            fh.write(window_csv(src, day, n))
    truth = []
    for (src, day, n, name, is_nrt) in wins:
        p = os.path.join(base, f"{src}_{day}_{n}.csv")
        with open(p, "rb") as fh:
            cols, _ = parse_csv(fh.read(),
                                {day + dt.timedelta(days=k)
                                 for k in range(n)}, name, is_nrt, {})
        for i in range(cols["t"].size):
            t = int(cols["t"][i])
            if not (f10b.seconds_since_epoch(d_lo) <= t
                    <= f10b.seconds_since_epoch(d_hi) + 86399):
                continue
            truth.append({
                "t": t,
                "lat": float(cols["lat"][i]),
                "lon": float(cols["lon"][i]),
                "platform": int(cols["platform"][i]),
                "v": [float(x) for x in cols["values"][i]]})
    return truth


ADAPTER = FireAdapter
