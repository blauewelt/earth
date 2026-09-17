"""GESLA-4 tide gauges, PUBLIC track — one row per gauge reading (family 1.0.tf).

PLAIN ENGLISH. Sea level measured every hour or more often at the world's
coastal tide gauges, from the GESLA-4 compilation, keeping only the
contributors whose terms allow redistribution. The research-only
contributors go to the private sibling `tide_private`; both are written by the
same reader, `_gesla.py`, whose module docstring holds the source, the
server's measured quirks, the sort rule and the row rules.
"""
from family1.adapters import _gesla


class TideAdapter(_gesla.GESLAAdapter):
    store = "tide"
    title = ("GESLA-4 tide gauges (public contributors), relative sea level, "
             "one row per gauge reading")
    track = "public"
    distribution = "public"
    licence = {"name": "GESLA-4 contributor terms: free use and "
                       "redistribution (UHSLC ERDDAP: 'may be used and "
                       "redistributed for free')",
               "redistribution": "attribution", "derived_works": "free",
               "attribution": "Haigh, I. D. et al. (2023), GESLA version 3, "
                              "Geosci. Data J. 10, 293-314; GESLA-4.1 via "
                              "the UHSLC ERDDAP (global_hourly_gesla); the "
                              "record's contributor (platforms.json)"}


ADAPTER = TideAdapter
