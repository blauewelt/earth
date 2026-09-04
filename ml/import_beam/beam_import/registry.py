"""Load and validate sources.yaml.

The registry is the single place a connection budget is written down. This
module's whole job is to refuse a registry that could let a source run without
one: every source must name a host, and every host must carry both numbers
(`max_lanes` and `min_gap_s`).

There is no Hub, no account and no destination in here: where the output goes
is `--output <uri>` on the command line, so the same registry can be run into
a local directory, a bucket, or a colleague's filesystem (DESIGN, revision 2).

Run the check on its own:

    python -m beam_import.registry --check
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import yaml

# Where sources.yaml lives when nobody says otherwise: next to this package.
DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sources.yaml")

# Every chunk kind manifest.py knows how to expand.
CHUNK_KINDS = {"year", "month", "var_year", "var_month", "file",
               "scrape_month", "day"}

# Every fetcher kind fetchers.py knows how to run.
FETCHERS = {"http", "cmems", "cds", "ncei_oisst_days", "psl_fallback",
            "transient_test"}

# Every transform kind transforms.py knows how to apply.
TRANSFORMS = {"bin025_days", "nc025_days", "oisst_days", "ncep_var_year",
              "rg_months", "series", "opaque"}

MODES = {"fetch"}   # revision 2 removed `verify`: everything is imported


class RegistryError(ValueError):
    """A registry that cannot be trusted. Always fatal — never retried."""


class Registry:
    """sources.yaml, parsed and checked.

    Attributes:
        output:  the `output:` block (shard layout, Stage-B shard count)
        hosts:   {host name: {max_lanes, min_gap_s, ...}}
        sources: [source dict], in file order (so the manifest is deterministic)
        path:    where it was loaded from
    """

    def __init__(self, doc: Dict[str, Any], path: str) -> None:
        self.path = path
        self.output: Dict[str, Any] = doc.get("output") or {}
        self.hosts: Dict[str, Dict[str, Any]] = doc.get("hosts") or {}
        self.sources: List[Dict[str, Any]] = doc.get("sources") or []

    # -- lookups -----------------------------------------------------------
    def host(self, name: str) -> Dict[str, Any]:
        return self.hosts[name]

    def source(self, name: str) -> Dict[str, Any]:
        for s in self.sources:
            if s["name"] == name:
                return s
        raise KeyError(f"no source named {name!r} in {self.path}")

    def total_lanes(self) -> int:
        """Sum of every host's lane budget — the upper bound on concurrency."""
        return sum(int(h["max_lanes"]) for h in self.hosts.values())

    def num_shards_per_group(self) -> int:
        """How many shards Stage B writes per channel group."""
        return int(self.output.get("num_shards_per_group", 64))

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        """Raise RegistryError on anything fatal; return non-fatal warnings."""
        errors: List[str] = []
        warnings: List[str] = []

        if not self.hosts:
            errors.append("no `hosts:` block")
        if not self.sources:
            errors.append("no `sources:` block")

        for name, h in self.hosts.items():
            for key in ("max_lanes", "min_gap_s"):
                if key not in h:
                    errors.append(f"host {name!r}: missing {key}")
            if "max_lanes" in h and int(h["max_lanes"]) < 1:
                errors.append(f"host {name!r}: max_lanes must be >= 1")
            if "min_gap_s" in h and float(h["min_gap_s"]) < 0:
                errors.append(f"host {name!r}: min_gap_s must be >= 0")

        seen = set()
        for s in self.sources:
            n = s.get("name")
            if not n:
                errors.append("a source has no name")
                continue
            if n in seen:
                errors.append(f"source {n!r}: duplicate name")
            seen.add(n)

            if s.get("host") not in self.hosts:
                errors.append(f"source {n!r}: host {s.get('host')!r} is not "
                              "in the hosts table")
            if s.get("mode") not in MODES:
                errors.append(f"source {n!r}: mode must be one of {sorted(MODES)}")
            if s.get("fetcher") not in FETCHERS:
                errors.append(f"source {n!r}: unknown fetcher "
                              f"{s.get('fetcher')!r}")
            if s.get("chunk") not in CHUNK_KINDS:
                errors.append(f"source {n!r}: unknown chunk "
                              f"{s.get('chunk')!r}")
            if s.get("transform") not in TRANSFORMS:
                errors.append(f"source {n!r}: unknown transform "
                              f"{s.get('transform')!r}")
            if "tier" not in s:
                errors.append(f"source {n!r}: no tier")

            if s.get("unverified_url"):
                warnings.append(f"source {n!r}: URL pattern is UNVERIFIED")
            if s.get("enabled") is False:
                warnings.append(f"source {n!r}: disabled (enabled: false)")

        if errors:
            raise RegistryError(
                f"{self.path} is not usable:\n  - " + "\n  - ".join(errors))
        return warnings


def load(path: Optional[str] = None) -> Registry:
    """Read sources.yaml, validate it, return it. Warnings go to stderr."""
    path = path or os.environ.get("BEAM_IMPORT_REGISTRY") or DEFAULT_REGISTRY
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    reg = Registry(doc, path)
    for w in reg.validate():
        print(f"warning: {w}", file=sys.stderr)
    return reg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="check sources.yaml")
    ap.add_argument("--registry", default=None, help="path to sources.yaml")
    ap.add_argument("--check", action="store_true",
                    help="validate and print a summary")
    args = ap.parse_args(argv)

    try:
        reg = load(args.registry)
    except RegistryError as exc:
        print(f"REGISTRY INVALID\n{exc}", file=sys.stderr)
        return 2

    print(f"registry  {reg.path}")
    print(f"output    {reg.output.get('layout', '<source>/<item_id>.tfrecord')} "
          f"({reg.num_shards_per_group()} shards per group in Stage B)")
    print(f"hosts     {len(reg.hosts)}   total lanes {reg.total_lanes()}")
    print(f"sources   {len(reg.sources)}")
    for tier in sorted({int(s['tier']) for s in reg.sources}):
        names = [s["name"] for s in reg.sources if int(s["tier"]) == tier]
        print(f"  tier {tier}: {len(names)}  {' '.join(names)}")
    print("OK")
    return 0


if __name__ == "__main__":                     # pragma: no cover
    raise SystemExit(main())
