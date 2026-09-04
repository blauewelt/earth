"""beam_import — the polite parallel import of the Earth training data (E-073).

Read DESIGN.md first, then README_FOR_GEMINI.md. The modules in reading order:

    registry.py    load and validate sources.yaml
    manifest.py    expand sources into work items
    hosts.py       pacing, backoff, circuit breaker (one LaneState per lane)
    fetchers.py    get the bytes for one work item
    transforms.py  put the bytes on our grid (imports the repo's bin rule)
    publish.py     commit to the Hub, download back, sha256 compare
    pipeline.py    the Apache Beam pipeline that wires all of the above
    verify_hub.py  compare the Hub against the manifest; write MANIFEST_tierN
    report.py      summary.md from report.jsonl
"""

__version__ = "1.0.0"
