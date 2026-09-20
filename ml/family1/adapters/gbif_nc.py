"""GBIF species occurrences, PRIVATE track — the CC BY-NC 4.0 records.

PLAIN ENGLISH. The same species records as `gbif`, for the publishers who
release under Creative Commons Attribution-NonCommercial 4.0: free to use for
research, not free to redistribute on a public track under this project's own
two-track rule (E-082 §1.3). Built by the same reader (`_gbif.py`, which
documents the source, the row rules and the split), published only to the
private repository, never republished. 16.38 % of the dated, georeferenced
snapshot, measured on GBIF's own search API on 2026-09-20.
"""
from family1.adapters import _gbif


class GBIFNonCommercialAdapter(_gbif.GBIFBase):
    store = "gbif_nc"
    title = ("GBIF occurrence records, CC BY-NC 4.0 (the AWS Open Data "
             "parquet snapshot): one row per dated, georeferenced record of "
             "an organism — PRIVATE track")
    track = "private"
    distribution = "private"
    licence = {"name": "CC BY-NC 4.0, per record (the snapshot's own "
                       "`license` column decides the track)",
               "redistribution": "no",
               "redistribution_confirmed": True,
               "derived_works": "non-commercial",
               "attribution": ("GBIF.org, the GBIF occurrence snapshot on "
                               "the Amazon Open Data registry "
                               "(gbif-open-data-us-east-1); each record "
                               "carries its own licence, publisher and "
                               "dataset, and the snapshot's citation.txt "
                               "names the form GBIF asks for"),
               "terms": ("CC BY-NC 4.0 forbids commercial use and this "
                         "project does not republish these rows: "
                         "`distribution` is private and "
                         "`redistribution` is 'no', which the registry "
                         "enforces. Chris's decision 3 in "
                         "ml/plans/E082_biosphere_wave.md is whether to keep "
                         "this track at all or drop the 16.38 % it holds")}


ADAPTER = GBIFNonCommercialAdapter
