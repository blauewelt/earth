#!/usr/bin/env python3
"""Draw the app icon: the BLUE planet, for blauewelt.

The icon is not artwork. It is NASA's Blue Marble (shaded relief +
bathymetry) — the app's own base globe — with the ocean deepened toward the
brand blue, in an orthographic view centred on 14 deg E, 34 deg N so Europe
and Africa face the viewer with the eastern Atlantic on the western limb (the
Atlantic overturning is what this project is ultimately built to watch). The
source raster is a snapshot in data/icon/, written by
`python3 scripts/refresh_data.py icon_sources` straight from the GIBS WMS.
Nothing is traced, painted or invented beyond the ocean tint, which is the
brand: an earth visualiser named "blauewelt" (blue world) is represented by a
blue earth.

History, so the choice isn't relitigated blind: the icon was an SST
climatology through the sst ramp (read as a heat map — too red), then MODIS
NDVI over a greyscale base (August 2026, and good at 48 px — but green, and
the brand is blue). The green alternatives live in the git history of this
file if the question comes up again.

Treatment ("blue-accent", the user's pick from the August 2026 contact
sheet): the WHOLE globe is Blue Marble luminance mapped onto a single ramp
from near-black navy to the app's accent blue (#4493f8 family) — a
monochrome blue planet, logo-like on purpose, where land reads as brighter
relief inside the same blue. RAMP_LO/RAMP_HI are the two ends; the deepened
"real-colour land" variant this replaced is in git history.

Run:  python3 scripts/make_icons.py
Writes icon-192.png, icon-512.png, icon-512-maskable.png in the repo root.
Deterministic: same snapshot in -> byte-identical PNGs out, so re-running it
in CI produces no diff.
"""

import math
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "icon")

BG = (13, 17, 23)        # --bg   #0d1117, the app's page background
RIM = (68, 147, 248)     # --accent #4493f8, a one-pixel limb so the sphere has an edge
RAMP_LO = (5, 13, 41)    # darkest ocean: a near-black navy, not pure black
RAMP_HI = (68, 147, 248)  # brightest relief: the app accent itself

VIEW_LON, VIEW_LAT = 14.0, 34.0
SS = 4                   # supersampling factor, downsampled with Lanczos


def _load():
    """The source raster as float RGB in [0,1]: luminance on the accent ramp."""
    rgb = np.asarray(Image.open(os.path.join(SRC, "base.png")).convert("RGB"),
                     dtype=np.float64) / 255.0
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # Normalise so the ramp's full span is used regardless of the snapshot's
    # own black/white points, then map monochrome: dark sea → RAMP_LO,
    # bright relief/ice → RAMP_HI. One ramp, no mask — the coastline emerges
    # from Blue Marble's own contrast.
    t = np.clip((lum - lum.min()) / max(lum.max() - lum.min(), 1e-9), 0.0, 1.0)[..., None]
    lo = np.array(RAMP_LO) / 255.0
    hi = np.array(RAMP_HI) / 255.0
    return lo + (hi - lo) * t


def _sample(src, lon, lat):
    """Bilinear read of an equirectangular raster at lon/lat in degrees."""
    H, W = src.shape[:2]
    fx = (lon + 180.0) / 360.0 * W - 0.5
    fy = (90.0 - lat) / 180.0 * H - 0.5
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    dx = (fx - x0)[..., None]
    dy = (fy - y0)[..., None]
    xa, xb = x0 % W, (x0 + 1) % W                       # wrap: longitude is a circle
    ya, yb = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)   # clamp: latitude is not
    return ((src[ya, xa] * (1 - dx) + src[ya, xb] * dx) * (1 - dy) +
            (src[yb, xa] * (1 - dx) + src[yb, xb] * dx) * dy)


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

    img = np.zeros((n, n, 3), dtype=np.float64)
    img[...] = np.array(BG) / 255.0
    img[inside] = _sample(_load(), lon, lat)[inside]

    # The limb: a thin accent ring so the disc still reads as a sphere on a
    # dark home screen.
    ring = inside & (rr >= (1.0 - 2.4 * SS / r))
    img[ring] = 0.45 * img[ring] + 0.55 * np.array(RIM) / 255.0

    im = Image.fromarray((img * 255).round().clip(0, 255).astype(np.uint8), "RGB")
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
