"""GESLA-4 tide gauges, PRIVATE track — the research-only contributors.

PLAIN ENGLISH. The same tide-gauge readings as `tide`, for the GESLA-4
contributors whose terms allow research use but not redistribution (CMEMS,
CV, UZ) and the South African gauges (SANHO's). Built by the same reader
(`_gesla.py`, which documents the source, the sort rule and the row rules),
published only to the private repository, never republished.
"""
from family1.adapters import _gesla


class TidePrivateAdapter(_gesla.GESLAAdapter):
    store = "tide_private"
    title = ("GESLA-4 tide gauges (research-only contributors), relative sea "
             "level, one row per gauge reading — PRIVATE track")
    track = "private"
    distribution = "private"
    licence = {"name": "GESLA-4 research-only contributor terms (CMEMS, CV, "
                       "UZ; SANHO for the South African gauges)",
               "redistribution": "no", "derived_works": "restricted",
               "attribution": "Haigh, I. D. et al. (2023), GESLA version 3, "
                              "Geosci. Data J. 10, 293-314; the record's "
                              "contributor (platforms.json)"}


ADAPTER = TidePrivateAdapter
