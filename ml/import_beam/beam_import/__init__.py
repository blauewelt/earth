"""beam_import — the polite parallel import of the Earth training data (E-073).

Read DESIGN.md first, then README_FOR_GEMINI.md. The modules in reading order:

    registry.py       load and validate sources.yaml
    manifest.py       expand sources into work items
    hosts.py          pacing, backoff, circuit breaker (one LaneState per lane)
    fetchers.py       get the bytes for one work item
    transforms.py     bytes -> tf.train.Example records (imports the bin rule)
    tfrecord.py       a pure-Python TFRecord reader/writer over FileSystems
    example.py        make_example / parse_example, with no TensorFlow import
    sinks.py          write-verify-mark, the retry queue, the absent evidence
    pipeline.py       STAGE A: the Beam pipeline that wires all of the above
    assemble.py       STAGE B: day records -> one Example per pentad x group
    verify_output.py  compare --output against the manifest
    report.py         summary.md, and the live mid-run view
"""

__version__ = "1.0.0"
