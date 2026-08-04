#!/usr/bin/env python3
"""Ask GIBS, for every timed layer we draw, what its NEWEST served date is.

Diagnostic only — run it when a layer goes silently blank. It walks backwards
from the date the app would ask for and reports the first date that answers,
so the answer is measured from the archive rather than guessed from a product's
documented latency (which is a promise, not an observation).

The app itself no longer needs this: it reads each layer's published time domain
(.../1.0.0/{layer}/default/{tms}/all/all.xml) and snaps to it. But a domain
document is a CLAIM about what is served and a tile is the FACT, and the two can
disagree. This script checks the fact. Keep it for that.

  python3 scripts/probe_gibs_lag.py https://gibs.earthdata.nasa.gov
"""
import datetime as dt
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8081"
LAYERS = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/timed_layers.json"))
TILES = [(2, 1, 1), (2, 2, 1), (2, 3, 1), (2, 0, 0)]   # a few, in case one is empty ocean


def served(layer, tms, ext, date):
    for lvl, x, y in TILES:
        url = f"{BASE}/wmts/epsg4326/best/{layer}/default/{date}/{tms}/{lvl}/{y}/{x}.{ext}"
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                if r.status == 200 and len(r.read(64)) > 0:
                    return True
        except Exception:
            continue
    return False


def back(date, monthly, n):
    d = dt.date.fromisoformat(date)
    if monthly:
        y, m = d.year, d.month - n
        while m < 1:
            m += 12
            y -= 1
        return dt.date(y, m, 1).isoformat()
    return (d - dt.timedelta(days=n)).isoformat()


today = dt.date.today()
asked = (today - dt.timedelta(days=2)).isoformat()
print(f"# app would ask for {asked} (today {today})\n")
for l in LAYERS:
    monthly = bool(l.get("monthly"))
    start = asked[:8] + "01" if monthly else asked
    steps = 8 if monthly else 21
    newest = None
    for n in range(steps):
        d = back(start, monthly, n)
        if served(l["layer"], l["tms"], l["ext"], d):
            newest = d
            break
    lag = "—" if newest is None else (dt.date.fromisoformat(asked) - dt.date.fromisoformat(newest)).days
    flag = "" if (newest == start) else "   <-- BLANK at the date the app asks for"
    print(f"{l['id']:<14} asks {start}  newest {newest or 'none in range'}  lag {lag}d{flag}")
