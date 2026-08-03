#!/usr/bin/env python3
"""Draw the app icon from the app's own data.

The icon is not artwork: it is the globe the app actually draws, rendered by
the same numbers and the same colour ramp. Ocean colour is `data/oisst.json`
-- NOAA's 1991-2020 sea-surface-temperature climatology, the very grid the
pixel inspector reads as "SST annual mean" -- run through the `sst` ramp
copied verbatim from RAMPS in src/app.js. Land is wherever that grid has no
value, because OISST's own land mask is the honest coastline for this data.
Nothing here is invented or traced by hand, which is the same rule the rest of
the project follows: what the icon shows is a fact about the dataset.

The view is centred on the North Atlantic (30 deg W, 20 deg N). That is not a
neutral choice and is not meant to be: the Atlantic overturning circulation is
what this project is ultimately built to watch, and centring it puts the Gulf
Stream's warm tongue and the subpolar cold patch on the face of the icon.

Run:  python3 scripts/make_icons.py
Writes icon-192.png, icon-512.png, icon-512-maskable.png in the repo root.
Deterministic: same inputs -> byte-identical outputs, so re-running it in CI
would produce no diff.
"""

import json
import math
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Copied from RAMPS.sst in src/app.js -- the icon must be the colours the app
# uses, so if that ramp ever changes this file is the one place to follow it.
SST_RAMP = [(0.00, (49, 54, 149)), (0.25, (116, 173, 209)), (0.50, (255, 255, 191)),
            (0.75, (244, 109, 67)), (1.00, (165, 0, 38))]

BG = (13, 17, 23)        # --bg   #0d1117, the app's page background
LAND = (47, 54, 64)      # dark slate: reads as land against both ocean and --bg
RIM = (68, 147, 248)     # --accent #4493f8, a one-pixel limb so the sphere has an edge

VIEW_LON, VIEW_LAT = -30.0, 20.0     # North Atlantic: the AMOC in the frame
SS = 4                                # supersampling factor, downsampled with Lanczos


def ramp(t):
    """Piecewise-linear colour lookup, same maths as rampColor() in src/app.js."""
    t = np.clip(t, 0.0, 1.0)
    xs = np.array([s[0] for s in SST_RAMP])
    out = np.zeros(t.shape + (3,), dtype=np.float64)
    for c in range(3):
        out[..., c] = np.interp(t, xs, [s[1][c] for s in SST_RAMP])
    return out


def _load_grid():
    """OISST as (values, ocean-fraction) arrays ready for bilinear reading.

    Land cells are holes in the SST field, so two things happen here. The
    land/ocean mask is kept as its own 0/1 array and read bilinearly, which
    turns a 1-degree staircase into a smooth coastline instead of the blocky
    one a nearest-neighbour lookup gives. And the holes in the temperature
    field are flood-filled outwards from the coast first, so interpolating
    near a shoreline never mixes a real temperature with a NaN-turned-zero
    and paints a cold fringe around every continent. The fill only ever
    touches pixels the mask then hides.
    """
    g = json.load(open(os.path.join(ROOT, "data", "oisst.json")))
    vals = np.array([np.nan if v is None else v for v in g["values"]], dtype=np.float64)
    vals = vals.reshape(g["ny"], g["nx"])       # row 0 = southernmost (see sampleGrid)
    ocean = np.isfinite(vals).astype(np.float64)

    filled = vals.copy()
    mean = np.nanmean(vals)
    for _ in range(200):
        holes = ~np.isfinite(filled)
        if not holes.any():
            break
        nb = np.stack([np.roll(filled, 1, 0), np.roll(filled, -1, 0),
                       np.roll(filled, 1, 1), np.roll(filled, -1, 1)])
        good = np.isfinite(nb)
        tot = np.where(good, nb, 0.0).sum(0)
        cnt = good.sum(0)
        avg = np.divide(tot, cnt, out=np.full_like(tot, np.nan), where=cnt > 0)
        filled = np.where(holes, avg, filled)
    filled = np.nan_to_num(filled, nan=mean)     # fully-enclosed basins, if any
    return g, filled, ocean


def _bilinear(g, arr, fx, fy):
    """Read `arr` at fractional cell coords, wrapping in longitude."""
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    dx, dy = fx - x0, fy - y0
    xa, xb = x0 % g["nx"], (x0 + 1) % g["nx"]
    ya, yb = np.clip(y0, 0, g["ny"] - 1), np.clip(y0 + 1, 0, g["ny"] - 1)
    return ((arr[ya, xa] * (1 - dx) + arr[ya, xb] * dx) * (1 - dy) +
            (arr[yb, xa] * (1 - dx) + arr[yb, xb] * dx) * dy)


def globe(size, disc_frac):
    """One icon: a sphere of diameter `disc_frac * size` on the app background."""
    n = size * SS
    r = n * disc_frac / 2.0

    # Pixel centres in globe-radius units, y up.
    ax = (np.arange(n) + 0.5 - n / 2.0) / r
    x, y = np.meshgrid(ax, -ax)
    rr = x * x + y * y
    inside = rr <= 1.0
    z = np.sqrt(np.clip(1.0 - rr, 0.0, None))     # towards the viewer

    # Orthographic un-projection: rotate the view-space point back to Earth
    # coordinates, then read its longitude and latitude.
    lat0, lon0 = math.radians(VIEW_LAT), math.radians(VIEW_LON)
    ex = z * math.cos(lat0) - y * math.sin(lat0)   # towards (lon0, lat0)
    ey = x                                          # east
    ez = z * math.sin(lat0) + y * math.cos(lat0)    # north
    lat = np.degrees(np.arcsin(np.clip(ez, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(ey, ex)) + math.degrees(lon0)
    lon = (lon + 180.0) % 360.0 - 180.0

    g, filled, ocean = _load_grid()
    # -0.5 because a grid value belongs to its cell's CENTRE, not its corner.
    fx = (lon - g["west"]) / g["dlon"] - 0.5
    fy = (lat - g["south"]) / g["dlat"] - 0.5
    sst = _bilinear(g, filled, fx, fy)
    wet = np.clip(_bilinear(g, ocean, fx, fy), 0.0, 1.0)[..., None]

    t = (sst - g["vmin"]) / (g["vmax"] - g["vmin"])
    img = np.zeros((n, n, 3), dtype=np.float64)
    img[...] = BG
    img[inside] = (wet * ramp(t) + (1 - wet) * np.array(LAND))[inside]

    # The limb: a thin ring so the disc still reads as a sphere on a dark
    # home screen, where ocean-blue against page-dark is a soft edge.
    ring = inside & (rr >= (1.0 - 2.4 * SS / r))
    img[ring] = 0.45 * img[ring] + 0.55 * np.array(RIM)

    im = Image.fromarray(img.round().clip(0, 255).astype(np.uint8), "RGB")
    return im.resize((size, size), Image.LANCZOS)


def main():
    # Full-bleed for the plain icons; the maskable one keeps the globe inside
    # Android's safe zone (the inner 80% circle), so a round or squircle mask
    # never clips a continent off.
    for name, size, frac in [("icon-192.png", 192, 0.86),
                             ("icon-512.png", 512, 0.86),
                             ("icon-512-maskable.png", 512, 0.60)]:
        path = os.path.join(ROOT, name)
        globe(size, frac).save(path, optimize=True)
        print(f"wrote {name} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
