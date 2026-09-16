#!/usr/bin/env python3
"""Write the COMMITTED PLACEHOLDER data/loitering.json.

The globe's "Loitering vessels" layer reads a baked snapshot of the Global
Fishing Watch Events API (`scripts/refresh_data.py loitering`). That API is
keyed, so until the `GFW_API_TOKEN` secret exists there is nothing to bake —
and a layer that cannot be switched on is a layer nobody reviews. This writes a
plausible stand-in instead: 40 drifts at real at-sea transshipment grounds,
with `fixture: true` so the layer says out loud that it is a placeholder and
the browser tests have something deterministic to assert against.

It is deterministic (fixed seed and a fixed "now"), so re-running it produces
byte-identical output and never churns the file. The first real run of
`.github/workflows/refresh-loitering.yml` overwrites it.

    python3 scripts/make_loitering_fixture.py
"""
import json
import os
import random
from datetime import datetime, timedelta, timezone

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")

random.seed(20260916)

# Real at-sea transshipment / waiting grounds, so the picture is plausible
# rather than a uniform scatter over the ocean.
HOTSPOTS = [
    ("NW Pacific, east of Japan",         (150.0, 175.0), (34.0, 44.0)),
    ("Sea of Okhotsk approaches",         (146.0, 158.0), (46.0, 52.0)),
    ("Eastern tropical Pacific",          (-130.0, -100.0), (-8.0, 6.0)),
    ("SW Atlantic, Argentine shelf edge", (-62.0, -55.0), (-47.0, -40.0)),
    ("West Africa, Gulf of Guinea",       (-6.0, 6.0), (-2.0, 6.0)),
    ("Arabian Sea",                       (58.0, 70.0), (8.0, 20.0)),
    ("Indian Ocean, Seychelles basin",    (50.0, 68.0), (-12.0, -2.0)),
    ("SW Indian Ocean ridge",             (36.0, 48.0), (-40.0, -30.0)),
    ("Peruvian offshore",                 (-84.0, -76.0), (-16.0, -6.0)),
    ("Mid-Atlantic, Canaries offshore",   (-25.0, -16.0), (18.0, 28.0)),
]
FLAGS = ["PAN", "RUS", "KOR", "CHN", "LBR", "VUT", "TWN", "JPN", "SGP", "CYP"]
STEM = ["FRIO", "OCEAN", "HAI DA", "SILVER", "LEO", "ORIENT", "ZHE LING",
        "MAESTRO", "NORDIC", "SUN", "VIKING", "BLUE", "AURORA", "TAI",
        "SEA", "CORAL", "ATLANTIC", "PACIFIC", "MERIDIAN", "KOTA"]
TAIL = ["STAR", "PEARL", "REEFER", "EXPRESS", "GLORY", "NO.7", "NO.23",
        "MARU", "TRADER", "WIND", "MOON", "SPIRIT"]

NOW = datetime(2026, 9, 16, 6, 0, 0, tzinfo=timezone.utc)


def build():
    start_day = (NOW - timedelta(days=30)).strftime("%Y-%m-%d")
    end_day = NOW.strftime("%Y-%m-%d")
    events = []
    for i in range(40):
        _, lonr, latr = HOTSPOTS[i % len(HOTSPOTS)]
        lon = round(random.uniform(*lonr), 4)
        lat = round(random.uniform(*latr), 4)
        begin = NOW - timedelta(days=random.uniform(0.3, 29.5))
        hours = round(random.choice([
            random.uniform(3, 9), random.uniform(9, 20), random.uniform(20, 60),
        ]), 2)
        end = begin + timedelta(hours=hours)
        if end > NOW:                      # a snapshot holds no future
            end = NOW - timedelta(minutes=5)
            hours = round((end - begin).total_seconds() / 3600, 2)
        events.append({
            "lat": lat, "lon": lon,
            "start": begin.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": hours,
            "speed_kn": round(random.uniform(0.2, 1.9), 2),
            "shore_km": round(random.uniform(60, 1400), 1),
            "name": f"{random.choice(STEM)} {random.choice(TAIL)}",
            "flag": random.choice(FLAGS),
            "mmsi": str(random.choice([2, 3, 4, 5, 6, 7]) * 100000000
                        + random.randrange(10000000, 99999999)),
        })
    events.sort(key=lambda r: (r["start"], r["mmsi"]))
    return {
        "id": "loitering",
        "title": "Loitering vessels (Global Fishing Watch)",
        "fixture": True,
        "fixture_note": ("A committed PLACEHOLDER, not real events: 40 plausible "
                         "drifts at real at-sea transshipment grounds, written by "
                         "scripts/make_loitering_fixture.py so the layer and its "
                         "tests work before the GFW_API_TOKEN secret exists. The "
                         "first run of .github/workflows/refresh-loitering.yml "
                         "overwrites this file with the real snapshot."),
        "fetched_at": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": {"start": start_day, "end": end_day},
        "count": len(events),
        "api_total": len(events),
        "source": "https://gateway.api.globalfishingwatch.org/v3/events",
        "dataset": "public-global-loitering-events:latest",
        "attribution": "Powered by Global Fishing Watch",
        "attribution_url": "https://globalfishingwatch.org",
        "licence": "CC BY-NC 4.0",
        "licence_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "note": ("A loitering event is a vessel drifting at sea at low speed for "
                 "hours, derived from AIS by Global Fishing Watch. Rolling 30-day "
                 "window, rebaked daily."),
        "events": events,
    }


if __name__ == "__main__":
    payload = build()
    out = os.path.join(DATA, "loitering.json")
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {out}: {payload['count']} placeholder events, "
          f"window {payload['window']['start']} -> {payload['window']['end']}")
