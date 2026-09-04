"""Expand sources.yaml into WORK ITEMS.

A work item is the smallest unit that is fetched, verified and published as
one: a year of OISST, a (variable, year) file of NCEP, a month of GLORYS, a
single file for the small series. Every item is a plain dict — it has to be,
because Beam pickles it and sends it to a worker process.

The expansion is DETERMINISTIC: run it twice and you get the same list in the
same order with the same lane numbers. That is what makes a re-run idempotent.

    python -m beam_import.manifest --tiers 0 --print
    python -m beam_import.manifest --tiers 0,1,2 --print --only duacs
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .registry import Registry, load

# A browser-ish User-Agent. Some public servers answer 403 to the default
# python-requests string; this is politeness, not evasion.
USER_AGENT = ("Mozilla/5.0 (compatible; blauewelt-earth-import/1.0; "
              "+https://github.com/blauewelt/earth)")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def month_range(first: str, last: str) -> List[str]:
    """['1993-01', ... , '2024-12'] inclusive."""
    y0, m0 = (int(x) for x in first.split("-"))
    y1, m1 = (int(x) for x in last.split("-"))
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def days_of_month(ym: str) -> List[str]:
    """['1993-01-01', ...] for one 'YYYY-MM'."""
    y, m = (int(x) for x in ym.split("-"))
    d = dt.date(y, m, 1)
    out = []
    while d.month == m:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def days_of_year(year: int, start_month: Optional[str] = None) -> List[str]:
    """Every day of `year`; from `start_month` (YYYY-MM) if the record begins
    part-way through it. OISST's first day is 1981-09-01."""
    d = dt.date(year, 1, 1)
    if start_month:
        sy, sm = (int(x) for x in start_month.split("-"))
        if sy == year:
            d = dt.date(year, sm, 1)
    out = []
    while d.year == year:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def lane_of(item_id: str, max_lanes: int) -> int:
    """Stable lane number for an item.

    sha1, not Python's hash(): hash() is randomised per process, so a
    multi-process runner would put the same item in different lanes in
    different workers and the lane cap would stop meaning anything.
    """
    digest = hashlib.sha1(item_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, int(max_lanes))


def _fmt(pattern: str, **kw: Any) -> str:
    """Fill a URL/filename pattern. Unknown placeholders are left alone."""
    out = pattern
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _urls(src: Dict[str, Any], **kw: Any) -> List[str]:
    """Candidate URLs for one item, in the order they should be tried:
    the main pattern, then per-item fallbacks, then host mirrors."""
    out: List[str] = []
    for key in ("url",):
        if src.get(key):
            out.append(_fmt(src[key], **kw))
    for pat in src.get("fallback_urls") or []:
        out.append(_fmt(pat, **kw))
    for pat in src.get("mirrors") or []:
        out.append(_fmt(pat, **kw))
    return out


def _hub_path(src: Dict[str, Any], filename: str, **kw: Any) -> str:
    """Where this item lives on the Hub.

    `hub_path` is an explicit full path (GLORYS keeps its historical home
    outside sources/). Otherwise it is sources/<hub_prefix>/<filename>.
    """
    if src.get("hub_path"):
        return _fmt(src["hub_path"], **kw)
    return f"sources/{src.get('hub_prefix', src['name'])}/{filename}"


# --------------------------------------------------------------------------
# the expansion
# --------------------------------------------------------------------------
def _item(reg: Registry, src: Dict[str, Any], key: str, filename: str,
          **kw: Any) -> Dict[str, Any]:
    """One work item. `key` is the part of the id after the source name."""
    host_name = src["host"]
    host = reg.host(host_name)
    item_id = f"{src['name']}/{key}"
    item: Dict[str, Any] = {
        "item_id": item_id,
        "source": src["name"],
        "tier": int(src["tier"]),
        "host": host_name,
        "lane": lane_of(item_id, host["max_lanes"]),
        "mode": src["mode"],
        "fetcher": src["fetcher"],
        "transform": src["transform"],
        "filename": filename,
        "hub_path": _hub_path(src, filename, **kw),
        "urls": _urls(src, **kw),
        "batch_files": int(src.get("batch_files", 1)),
        "bytes_wire": int(src.get("bytes_wire", 0)),
        "bytes_stored": int(src.get("bytes_stored", 0)),
        "unverified_url": bool(src.get("unverified_url", False)),
    }
    # Everything a fetcher might need, passed through verbatim.
    for key_name in ("dataset_id", "product_id", "variables", "bbox", "depth",
                     "subset_by", "requires_env", "resolve_dataset_id",
                     "url_preliminary", "fallback"):
        if src.get(key_name) is not None:
            item[key_name] = src[key_name]
    item.update(kw)
    return item


def _scrape_months(src: Dict[str, Any], offline: bool) -> Optional[List[str]]:
    """Roemmich-Gilson: read the monthly extension list off the index page.

    Returns None when the scrape could not be done, and the caller falls back
    to the declared `months:` range. A month the server does not actually have
    then marks its item `absent`, never `failed`.
    """
    if offline:
        return None
    try:
        import requests
        r = requests.get(src["scrape_url"], timeout=60,
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        stamps = sorted(set(re.findall(src["scrape_re"], r.text)))
        if not stamps:
            return None
        return [f"{s[:4]}-{s[4:6]}" for s in stamps]
    except Exception as exc:                                  # noqa: BLE001
        print(f"warning: scrape of {src['scrape_url']} failed ({exc}); "
              "falling back to the declared month range", file=sys.stderr)
        return None


def expand_source(reg: Registry, src: Dict[str, Any],
                  offline: bool = False) -> List[Dict[str, Any]]:
    """All work items for one source, in a deterministic order."""
    chunk = src["chunk"]
    items: List[Dict[str, Any]] = []

    if chunk == "year":
        for year in range(src["years"][0], src["years"][1] + 1):
            fn = _fmt(src.get("filename", "{year}.nc"), year=year)
            # `days:` is an optional override of the day list a per-day
            # fetcher would compute. Its real use is a partial re-fetch; the
            # test fixtures use it to make a "year" three days long.
            items.append(_item(reg, src, str(year), fn, year=year,
                               start_month=src.get("start_month"),
                               days=src.get("days")))

    elif chunk == "month":
        for ym in month_range(src["months"][0], src["months"][1]):
            compact = ym.replace("-", "")
            fn = _fmt(src.get("filename", "{ym}.nc"), ym=compact, month=ym)
            items.append(_item(reg, src, ym, fn, ym=compact, month=ym))

    elif chunk == "var_year":
        for var in src["vars"]:
            for year in range(src["years"][0], src["years"][1] + 1):
                fn = _fmt(src.get("filename", "{var}.{year}.nc"),
                          var=var, year=year)
                items.append(_item(reg, src, f"{var}/{year}", fn,
                                   var=var, year=year, file=fn))
        for extra in src.get("extra_files") or []:
            items.append(_item(reg, src, extra, extra, file=extra,
                               var=extra, year=0))

    elif chunk == "var_month":
        for var in src["vars"]:
            for ym in month_range(src["months"][0], src["months"][1]):
                compact = ym.replace("-", "")
                fn = _fmt(src.get("filename", "{var}_{ym}.nc"),
                          var=var, ym=compact)
                items.append(_item(reg, src, f"{var}/{ym}", fn,
                                   var=var, ym=compact, month=ym, file=fn))

    elif chunk == "file":
        for f in src["files"]:
            fn = _fmt(src.get("filename", "{file}"), file=f)
            items.append(_item(reg, src, f, fn, file=f))

    elif chunk == "scrape_month":
        for f in src.get("files") or []:                     # the base files
            fn = _fmt(src.get("filename", "{file}"), file=f)
            items.append(_item(reg, src, f, fn, file=f))
        months = _scrape_months(src, offline)
        scraped = months is not None
        if months is None:
            months = month_range(src["months"][0], src["months"][1])
        for ym in months:
            compact = ym.replace("-", "")
            f = _fmt(src["month_filename"], ym=compact)
            fn = _fmt(src.get("filename", "{file}"), file=f)
            it = _item(reg, src, f, fn, file=f, ym=compact, month=ym)
            # An unscraped month is a guess; a 404 on it is `absent`.
            it["absent_ok"] = not scraped
            items.append(it)

    else:                                                     # pragma: no cover
        raise ValueError(f"source {src['name']}: unknown chunk {chunk!r}")

    return items


def build(reg: Registry, tiers: Sequence[int], only: Optional[str] = None,
          include_disabled: bool = False,
          offline: bool = False) -> List[Dict[str, Any]]:
    """The whole manifest for the requested tiers."""
    items: List[Dict[str, Any]] = []
    for src in reg.sources:
        if only and src["name"] != only:
            continue
        if int(src["tier"]) not in tiers and not only:
            continue
        # `--only` is an explicit request, so it wakes a disabled source too.
        if src.get("enabled") is False and not (include_disabled or only):
            continue
        items.extend(expand_source(reg, src, offline=offline))
    return items


def counts(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts and byte totals for --print.

    VERIFY-ONLY items contribute ZERO to the wire and stored totals. They are
    already on the Hub and the pipeline only checks they are there — counting
    their 2.2 GB-per-month download in a "how much will this fetch" figure
    would overstate Tier 0 by a factor of four and make every capacity
    decision downstream of it wrong.
    """
    by_source: Dict[str, int] = {}
    by_host: Dict[str, int] = {}
    by_lane: Dict[str, int] = {}
    fetch_items = fetch_wire = fetch_stored = 0
    verify_items = verify_stored = 0
    for it in items:
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1
        by_host[it["host"]] = by_host.get(it["host"], 0) + 1
        key = f"{it['host']}/{it['lane']}"
        by_lane[key] = by_lane.get(key, 0) + 1
        if it["mode"] == "verify":
            verify_items += 1
            verify_stored += it["bytes_stored"]
        else:
            fetch_items += 1
            fetch_wire += it["bytes_wire"]
            fetch_stored += it["bytes_stored"]
    return {"by_source": by_source, "by_host": by_host, "by_lane": by_lane,
            "fetch_items": fetch_items, "fetch_wire": fetch_wire,
            "fetch_stored": fetch_stored, "verify_items": verify_items,
            "verify_stored": verify_stored}


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return str(n)                                             # pragma: no cover


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="expand sources.yaml into items")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--tiers", default="0", help="e.g. 0 or 0,1 or 0,1,2")
    ap.add_argument("--only", default=None, help="one source name")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="print every item, then the counts")
    ap.add_argument("--json", default=None, help="write the items to a file")
    ap.add_argument("--include-disabled", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="never make a network request (skips the RG scrape)")
    args = ap.parse_args(argv)

    reg = load(args.registry)
    tiers = [int(t) for t in str(args.tiers).split(",") if t.strip() != ""]
    items = build(reg, tiers, only=args.only,
                  include_disabled=args.include_disabled, offline=args.offline)

    if args.do_print:
        for it in items:
            flag = " UNVERIFIED_URL" if it["unverified_url"] else ""
            print(f"{it['item_id']:52s} {it['host']}/{it['lane']:<2d} "
                  f"{it['mode']:6s} {_human(it['bytes_stored']):>9s}  "
                  f"{it['hub_path']}{flag}")

    c = counts(items)
    print("\n--- counts by source " + "-" * 40)
    for src in reg.sources:
        n = c["by_source"].get(src["name"])
        if n is None:
            continue
        exp = src.get("expected_items")
        mark = "" if exp in (None, n) else f"   (registry says {exp})"
        print(f"  {src['name']:16s} tier {src['tier']}  {n:6d} items"
              f"  host {src['host']}{mark}")
    print("--- counts by host " + "-" * 42)
    for host, n in sorted(c["by_host"].items()):
        lanes = reg.host(host)["max_lanes"]
        print(f"  {host:14s} {n:6d} items over {lanes} lane(s)")
    unver = sum(1 for i in items if i["unverified_url"])
    print("-" * 61)
    if c["verify_items"]:
        print(f"  already on Hub (verify only): {c['verify_items']} items, "
              f"{_human(c['verify_stored'])}")
    print(f"  to fetch: {c['fetch_items']} items · "
          f"{_human(c['fetch_wire'])} wire · "
          f"{_human(c['fetch_stored'])} stored")
    print(f"  TOTAL {len(items)} items · {unver} with an unverified URL")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=1, sort_keys=True)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":                     # pragma: no cover
    raise SystemExit(main())
