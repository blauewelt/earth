"""Load and validate sources.yaml.

The registry is the single place a connection budget is written down. This
module's whole job is to refuse a registry that could let a source run without
one: every source must name a host, and every host must carry both numbers
(`max_lanes` and `min_gap_s`).

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
CHUNK_KINDS = {"year", "month", "var_year", "var_month", "file", "scrape_month"}

# Every fetcher kind fetchers.py knows how to run.
FETCHERS = {"http", "cmems", "cds", "ncei_oisst_year", "psl_fallback",
            "transient_test"}

# Every transform kind transforms.py knows how to apply.
TRANSFORMS = {"passthrough", "bin025", "oisst_year_fold"}

MODES = {"fetch", "verify"}


class RegistryError(ValueError):
    """A registry that cannot be trusted. Always fatal — never retried."""


class Registry:
    """sources.yaml, parsed and checked.

    Attributes:
        hub:     the `hub:` block (Hub repo id, commit budget)
        hosts:   {host name: {max_lanes, min_gap_s, ...}}
        sources: [source dict], in file order (so the manifest is deterministic)
        path:    where it was loaded from
    """

    def __init__(self, doc: Dict[str, Any], path: str) -> None:
        self.path = path
        self.hub: Dict[str, Any] = doc.get("hub") or {}
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

    def commit_budget_per_hour(self) -> int:
        return int(self.hub.get("commit_budget_per_hour", 60))

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        """Raise RegistryError on anything fatal; return non-fatal warnings."""
        errors: List[str] = []
        warnings: List[str] = []

        if not self.hosts:
            errors.append("no `hosts:` block")
        if not self.sources:
            errors.append("no `sources:` block")
        if not self.hub.get("repo_id"):
            errors.append("hub.repo_id is missing")

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
            if int(s.get("batch_files", 0)) < 1:
                errors.append(f"source {n!r}: batch_files must be >= 1")

            # A big item must not be batched with anything else: one failed
            # commit should not cost several hundred megabytes of re-upload.
            big = int(s.get("bytes_stored", 0)) > 200 * 1024 * 1024
            if big and int(s.get("batch_files", 1)) != 1:
                errors.append(f"source {n!r}: items are > 200 MB, so "
                              "batch_files must be 1")

            if not s.get("hub_prefix") and not s.get("hub_path"):
                errors.append(f"source {n!r}: needs hub_prefix or hub_path")
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
    print(f"hub       {reg.hub.get('repo_id')} "
          f"({reg.commit_budget_per_hour()} commits/h)")
    print(f"hosts     {len(reg.hosts)}   total lanes {reg.total_lanes()}")
    print(f"sources   {len(reg.sources)}")
    for tier in sorted({int(s['tier']) for s in reg.sources}):
        names = [s["name"] for s in reg.sources if int(s["tier"]) == tier]
        print(f"  tier {tier}: {len(names)}  {' '.join(names)}")
    print("OK")
    return 0


if __name__ == "__main__":                     # pragma: no cover
    raise SystemExit(main())
