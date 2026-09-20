"""GBIF species occurrences, PUBLIC track — the CC0 and CC BY 4.0 records.

PLAIN ENGLISH. GBIF — the Global Biodiversity Information Facility — is the
world's pooled record of "an organism of this kind was here on this day".
This store holds the records whose own licence lets them be redistributed:
Creative Commons Zero (a waiver of copyright) and Creative Commons
Attribution 4.0. Its private sibling `gbif_nc` holds the non-commercial ones.
Both are built by the same reader (`_gbif.py`, which documents the source,
the row rules, the code tables and the two-track split) from the same monthly
snapshot, and together they read it once.
"""
from family1.adapters import _gbif


class GBIFAdapter(_gbif.GBIFBase):
    store = "gbif"
    title = ("GBIF occurrence records, CC0 and CC BY 4.0 (the AWS Open Data "
             "parquet snapshot): one row per dated, georeferenced record of "
             "an organism")
    track = "public"
    distribution = "public"
    licence = {"name": "CC0 1.0 and CC BY 4.0, per record (the snapshot's "
                       "own `license` column decides the track)",
               "redistribution": "attribution",
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": ("GBIF.org, the GBIF occurrence snapshot on "
                               "the Amazon Open Data registry "
                               "(gbif-open-data-us-east-1); each record "
                               "carries its own licence, publisher and "
                               "dataset, and the snapshot's citation.txt "
                               "names the form GBIF asks for"),
               "terms": ("measured on GBIF's own search API 2026-09-20: of "
                         "the 3,668,229,700 records with coordinates, no "
                         "geospatial issue and a year, CC BY 4.0 is 72.57 % "
                         "and CC0 1.0 is 11.05 % — 83.62 % on this track — "
                         "and CC BY-NC 4.0 is the remaining 16.38 %, which "
                         "goes to `gbif_nc`")}


ADAPTER = GBIFAdapter
