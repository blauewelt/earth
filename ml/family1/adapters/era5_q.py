"""ERA5 specific humidity in g/kg (1000 x the archive's kg/kg) — 13 pressure
levels, six-hourly, 1 degree (family 1.2, E-085; sharded tier G, exception E4).

PLAIN ENGLISH. One of the four atmosphere stores family 1.2 adds to family
1.gf: the channel axis is the pressure level (`q_50` .. `q_1000`,
50 hPa ~20 km up to 1000 hPa at the surface), every frame is one analysis
instant at 00/06/12/18 UTC, and each value is the area-weighted mean of
ERA5's 0.25-degree field over the 1-degree box around the point. Everything
about the source, the seam, the regrid and the encoding is in `_era5.py`;
this module only names the variable. The g/kg unit is the one
transform: in kg/kg every stratospheric value would sit below float16's
smallest normal number (6.1e-5).
"""
from family1.adapters import _era5


class ERA5QAdapter(_era5.ERA5Base):
    store = "era5_q"
    var = "q"
    channels = _era5.channels_for("q")


ADAPTER = ERA5QAdapter
