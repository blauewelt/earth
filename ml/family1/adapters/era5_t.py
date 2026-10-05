"""ERA5 air temperature, in K — 13 pressure levels, six-hourly, 1 degree
(family 1.2, E-085; sharded tier G, exception E4).

PLAIN ENGLISH. One of the four atmosphere stores family 1.2 adds to family
1.gf: the channel axis is the pressure level (`t_50` .. `t_1000`,
50 hPa ~20 km up to 1000 hPa at the surface), every frame is one analysis
instant at 00/06/12/18 UTC, and each value is the area-weighted mean of
ERA5's 0.25-degree field over the 1-degree box around the point. Everything
about the source, the seam, the regrid and the encoding is in `_era5.py`;
this module only names the variable.
"""
from family1.adapters import _era5


class ERA5TAdapter(_era5.ERA5Base):
    store = "era5_t"
    var = "t"
    channels = _era5.channels_for("t")


ADAPTER = ERA5TAdapter
