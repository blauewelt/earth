"""Is the tier actually on the Hub?

    python -m beam_import.verify_hub --tiers 0
    python -m beam_import.verify_hub --tiers 0 --report out/tier0/report.jsonl \
        --publish

Without `--publish` it only LOOKS: it lists the Hub, compares that listing to
the manifest for the tier, and prints what is missing and what is on the Hub
under a tier prefix but not in the manifest ("extra"). `missing = 0` is the
definition of the tier being done (DESIGN §5.3).

With `--publish` it also writes `sources/MANIFEST_tier<N>.json` — one record
per file, with the Hub path, the byte count, the sha256 the restore-verify
compared, the upstream URL or product id, and the fetch timestamp — and
commits it in ONE commit.

This command NEVER deletes anything on the Hub. Nothing in this package does.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import manifest, publish, registry
from .report import read_records


def build_manifest_json(items: List[Dict[str, Any]],
                        records: List[Dict[str, Any]],
                        tier: int) -> Dict[str, Any]:
    """One record per file, joining the manifest with what the run reported."""
    by_id = {r["item_id"]: r for r in records if r.get("status") != "counters"}
    files = []
    for it in items:
        r = by_id.get(it["item_id"], {})
        files.append({
            "hub_path": it["hub_path"],
            "item_id": it["item_id"],
            "source": it["source"],
            "host": it["host"],
            "status": r.get("status", "unknown"),
            "bytes": r.get("bytes", 0),
            "sha256": r.get("sha256"),
            "upstream": (it["urls"][0] if it.get("urls")
                         else it.get("dataset_id")),
            "transform": it["transform"],
        })
    return {
        "tier": tier,
        "written_at": dt.datetime.now(dt.timezone.utc)
                        .replace(microsecond=0).isoformat(),
        "n_files": len(files),
        "files": sorted(files, key=lambda f: f["hub_path"]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="compare the Hub to the manifest")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--tiers", default="0")
    ap.add_argument("--only", default=None)
    ap.add_argument("--hub", default="hub")
    ap.add_argument("--report", default=None,
                    help="report.jsonl, for the sha256 column of the manifest")
    ap.add_argument("--publish", action="store_true",
                    help="write and commit sources/MANIFEST_tier<N>.json")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)

    reg = registry.load(args.registry)
    tiers = [int(t) for t in str(args.tiers).split(",") if t.strip() != ""]
    items = manifest.build(reg, tiers, only=args.only, offline=args.offline)
    pub = publish.make_publisher(args.hub, reg.hub["repo_id"],
                                 reg.hub.get("repo_type", "dataset"))
    on_hub = set(pub.list_paths())

    wanted = {it["hub_path"] for it in items}
    missing = sorted(wanted - on_hub)
    prefixes = {it["hub_path"].rsplit("/", 1)[0] for it in items}
    extra = sorted(p for p in on_hub
                   if p.rsplit("/", 1)[0] in prefixes and p not in wanted)

    print(f"tier(s) {tiers}: {len(wanted)} file(s) expected, "
          f"{len(wanted) - len(missing)} on the Hub")
    print(f"missing: {len(missing)}")
    for p in missing[:50]:
        print(f"  MISSING {p}")
    if len(missing) > 50:
        print(f"  ... and {len(missing) - 50} more")
    print(f"extra (on the Hub under a tier prefix, not in the manifest): "
          f"{len(extra)}")
    for p in extra[:20]:
        print(f"  EXTRA   {p}")

    if args.publish:
        records = read_records(args.report) if args.report else []
        if not records:
            print("warning: no --report, so the manifest will carry no sha256 "
                  "column", file=sys.stderr)
        for tier in tiers:
            tier_items = [it for it in items if int(it["tier"]) == tier]
            if not tier_items:
                continue
            doc = build_manifest_json(tier_items, records, tier)
            os.makedirs(args.out_dir, exist_ok=True)
            local = os.path.join(args.out_dir, f"MANIFEST_tier{tier}.json")
            with open(local, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True)
            hub_path = f"sources/MANIFEST_tier{tier}.json"
            pub.publish_verified([(local, hub_path)], args.out_dir,
                                 f"beam_import: manifest for tier {tier}")
            print(f"published {hub_path} ({doc['n_files']} file records)")

    return 0 if not missing else 3


if __name__ == "__main__":                     # pragma: no cover
    raise SystemExit(main())
