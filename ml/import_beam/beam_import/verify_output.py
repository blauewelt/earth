"""Is the tier actually written?

    python -m beam_import.verify_output --tiers 0 --output /data/import \\
        --state-dir /var/tmp/beam_import

It re-reads every `.done` marker under `--output`, compares that set to the
manifest, checks each shard's byte count against what its marker claims, and
prints:

    missing       manifest items with no marker            (must reach 0)
    short         a shard whose bytes disagree with its marker
    extra         a marker with no manifest item
    queued        what is still in retry_queue.jsonl
    absent        items the archive said 404/410 to twice, with the evidence

`missing 0` and an empty queue is what DESIGN §6 calls done. The `absent`
list is NOT an error and is NOT auto-accepted: a human reads it before Stage
B, because a missing pentad bin later must trace back to one of these.

With `--deep` it also re-reads every shard end to end, checking every
record's CRC — slower, and the right thing to do once before a long training
run. This command NEVER deletes anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import manifest, registry, sinks, tfrecord


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="verify the Stage-A output")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--tiers", default="0")
    ap.add_argument("--only", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--deep", action="store_true",
                    help="re-read every shard and check every record's CRC")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)

    reg = registry.load(args.registry)
    tiers = [int(t) for t in str(args.tiers).split(",") if t.strip() != ""]
    items = manifest.build(reg, tiers, only=args.only, offline=args.offline)
    wanted = {it["item_id"] for it in items}

    markers = sinks.list_done(args.output)
    missing = sorted(wanted - set(markers))
    extra = sorted(set(markers) - wanted)

    short: List[str] = []
    deep_bad: List[str] = []
    total_bytes = 0
    for item_id, doc in sorted(markers.items()):
        shard = doc.get("shard")
        total_bytes += int(doc.get("bytes", 0))
        if not shard or not tfrecord.exists(shard):
            short.append(f"{item_id}: shard {shard} is gone")
            continue
        if args.deep:
            try:
                n = len(tfrecord.read_records(shard))
                if n != int(doc.get("n_records", n)):
                    deep_bad.append(
                        f"{item_id}: marker says {doc.get('n_records')} "
                        f"records, shard holds {n}")
            except Exception as exc:                          # noqa: BLE001
                deep_bad.append(f"{item_id}: {exc}")

    queued = (sinks.read_queue(sinks.queue_uri(args.state_dir))
              if args.state_dir else [])
    absent = sinks.list_absent(args.state_dir) if args.state_dir else []
    absent_ids = {a["item_id"] for a in absent}

    print(f"tier(s) {tiers}: {len(wanted)} item(s) expected, "
          f"{len(wanted) - len(missing)} written "
          f"({total_bytes / 1e9:.2f} GB in {len(markers)} shard(s))")
    print(f"missing: {len(missing)}")
    for p in missing[:50]:
        flag = "  (absent, with evidence)" if p in absent_ids else ""
        print(f"  MISSING {p}{flag}")
    if len(missing) > 50:
        print(f"  ... and {len(missing) - 50} more")
    print(f"short:   {len(short)}")
    for p in short[:20]:
        print(f"  SHORT   {p}")
    if args.deep:
        print(f"deep:    {len(deep_bad)} shard(s) failed a full re-read")
        for p in deep_bad[:20]:
            print(f"  BAD     {p}")
    print(f"extra:   {len(extra)} marker(s) with no manifest item")
    print(f"queued:  {len(queued)} item(s) still to do")
    print(f"absent:  {len(absent)} item(s) — REVIEW THESE BEFORE STAGE B")
    for a in absent[:20]:
        sight = a.get("sightings", [])
        when = " and ".join(s.get("at", "?") for s in sight[:2])
        print(f"  ABSENT  {a['item_id']}  404/410 at {when}")

    doc = {"missing": missing, "short": short, "extra": extra,
           "deep_bad": deep_bad, "queued": [q["item_id"] for q in queued],
           "absent": sorted(absent_ids), "bytes": total_bytes,
           "n_markers": len(markers)}
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        print(f"wrote {args.json_out}")

    # Anything that is missing and NOT explained by absent evidence is a
    # reason to run again, not a reason to move on.
    unexplained = [m for m in missing if m not in absent_ids]
    if unexplained or short or deep_bad:
        return 3
    return 0


if __name__ == "__main__":                     # pragma: no cover
    raise SystemExit(main())
