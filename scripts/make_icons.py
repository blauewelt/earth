#!/usr/bin/env python3
"""Draw the app icon as a globe the app could actually draw.

The icon is not artwork. It is the same composite the app puts on screen when
you switch the vegetation layer on: MODIS Terra's monthly NDVI over a
desaturated Blue Marble base. Both rasters are snapshots in data/icon/, written
by `python3 scripts/refresh_data.py icon_sources` straight from NASA GIBS --
the same GIBS the running app tiles from. Nothing here is traced, painted or
invented, which is the rule the rest of the project follows: what the icon
shows is a fact about the data.

Why NDVI. The green is the biosphere at the northern growing season's peak
(June -- the month is pinned in refresh_data.py, not "latest", so the icon
cannot drift under the user). Land the sensor found bare reads pale: the Sahara
is a bright band across the middle of the disc, and that band is most of what
makes the icon legible at 48 px.

Why greyscale underneath. src/app.js sets `baseImageryLayer.saturation = 0` the
moment a colormapped layer goes on, so a grey base IS what the app looks like
here -- see setBaseGrey(). The one deliberate departure is brightness: the app
dims to 0.6 behind a full-screen layer on a lit page, and at icon size on a
dark home screen that sinks the ocean into the background and costs the disc
its edge. BASE_GAIN below is the value that keeps the coastline readable when
the whole globe is 48 px across.

The view is centred on 14 deg E, 34 deg N: Europe and Africa face the viewer,
with the eastern Atlantic on the western limb. The Atlantic overturning is what
this project is ultimately built to watch, so it stays in frame rather than in
the middle.

Run:  python3 scripts/make_icons.py
Writes icon-192.png, icon-512.png, icon-512-maskable.png in the repo root.
Deterministic: same snapshots in -> byte-identical PNGs out, so re-running it
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
BASE_GAIN = 0.95         # see the note on brightness in the module docstring

VIEW_LON, VIEW_LAT = 14.0, 34.0
SS = 4                   # supersampling factor, downsampled with Lanczos


def _load():
    """The two source rasters as float RGBA in [0,1], plate carree, north-up."""
    grey = np.asarray(Image.open(os.path.join(SRC, "base_grey.png")).convert("L"),
                      dtype=np.float64) / 255.0
    base = np.zeros(grey.shape + (4,), dtype=np.float64)
    base[..., :3] = (grey * BASE_GAIN)[..., None]
    base[..., 3] = 1.0
    ndvi = np.asarray(Image.open(os.path.join(SRC, "ndvi.png")).convert("RGBA"),
                      dtype=np.float64) / 255.0
    return base, ndvi


def _sample(src, lon, lat):
    """Bilinear read of an equirectangular raster at lon/lat in degrees.

    Bilinear rather than nearest because the disc is sampled at 4x and then
    Lanczos-reduced; a nearest lookup would carry a source-pixel staircase all
    the way down into the 48 px icon, and it shows most along the coastline.
    """
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
    face = np.zeros((n, n, 3), dtype=np.float64)
    for src in _load():                              # base first, NDVI over it
        s = _sample(src, lon, lat)
        a = s[..., 3:4]                              # NDVI is transparent over water
        face = face * (1 - a) + s[..., :3] * a
    img[inside] = face[inside]

    # The limb: a thin ring so the disc still reads as a sphere on a dark home
    # screen, where a grey-blue ocean against page-dark is a soft edge.
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
