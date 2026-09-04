"""summary.md from report.jsonl, and the live mid-run view.

One page: what landed, how many bytes each host served, how long it took, and
— the column that matters — how many backoffs and breaker trips each host
produced. A host with many trips means the budget in sources.yaml is TOO
GENEROUS and should be LOWERED before the next run. It is never a reason to
raise anything.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def read_records(path: str) -> List[Dict[str, Any]]:
    """One JSON object per line. Used for report.jsonl and progress files.

    A half-written last line (the run was killed mid-append) is skipped rather
    than raising: a progress file is read while it is being written to.
    """
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                       # a torn final line; ignore it
    return out


def read_progress(state_dir: str) -> List[Dict[str, Any]]:
    """Every record every lane has appended so far, in file order."""
    import glob
    out: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(state_dir, "progress",
                                              "*.jsonl"))):
        out.extend(read_records(path))
    return out


def trips_per_host(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Breaker trips per host, taking the largest reading per lane."""
    per_lane: Dict[tuple, int] = {}
    for r in records:
        key = (r.get("host"), r.get("lane"))
        if r.get("status") == "counters":
            n = int((r.get("counters") or {}).get("trips", 0))
        else:
            n = int(r.get("trips_so_far") or 0)
        per_lane[key] = max(per_lane.get(key, 0), n)
    out: Dict[str, int] = {}
    for (host, _lane), n in per_lane.items():
        out[host] = out.get(host, 0) + n
    return {h: n for h, n in out.items() if h and n}


def _dedupe_key(rec: Dict[str, Any]) -> str:
    """One key per thing a record can describe.

    Item records are keyed by item_id. The per-lane counters record is not an
    item, so it is keyed by its lane — otherwise every lane's counters would
    collapse onto one another under the id `_lane`.
    """
    if rec.get("status") == "counters":
        return f"_lane/{rec.get('host')}/{rec.get('lane')}"
    return str(rec.get("item_id"))


def collect(report_path: Optional[str] = None,
            state_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """The union of the progress files and report.jsonl, LAST RECORD WINS.

    The two sources overlap by design — a lane appends as it goes, and Beam
    writes the same records again at the end — so they are merged rather than
    chosen between. The final report is read last, so where both have a
    record for the same item the final one is the one that counts.
    """
    records: List[Dict[str, Any]] = []
    if state_dir:
        records.extend(read_progress(state_dir))
    if report_path:
        records.extend(read_records(report_path))
    merged: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        merged[_dedupe_key(rec)] = rec
    return list(merged.values())


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return str(n)                                             # pragma: no cover


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    items = [r for r in records if r.get("status") != "counters"]
    lanes = [r for r in records if r.get("status") == "counters"]

    by_status: Dict[str, int] = {}
    by_source: Dict[str, Dict[str, int]] = {}
    by_host: Dict[str, Dict[str, float]] = {}

    for r in items:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        s = by_source.setdefault(r["source"], {})
        s[r["status"]] = s.get(r["status"], 0) + 1
        h = by_host.setdefault(r["host"], {"bytes": 0.0, "seconds": 0.0,
                                           "items": 0.0, "backoffs": 0.0,
                                           "trips": 0.0, "requests": 0.0})
        h["bytes"] += r.get("bytes") or 0
        h["seconds"] += r.get("seconds") or 0.0
        h["items"] += 1

    # Backoffs and trips come from the per-lane counters record, which only
    # exists once a lane has FINISHED. Mid-run they come instead from the
    # running snapshot each item record carries, taken per lane as the maximum
    # (the snapshot only ever grows). Whichever is larger is used, so a
    # half-finished run and a finished one both report honestly.
    per_lane: Dict[tuple, Dict[str, float]] = {}
    for r in items:
        key = (r["host"], r.get("lane"))
        cur = per_lane.setdefault(key, {"backoffs": 0.0, "trips": 0.0})
        cur["backoffs"] = max(cur["backoffs"], r.get("backoffs_so_far") or 0)
        cur["trips"] = max(cur["trips"], r.get("trips_so_far") or 0)
    for r in lanes:
        c = r.get("counters") or {}
        key = (r["host"], r.get("lane"))
        cur = per_lane.setdefault(key, {"backoffs": 0.0, "trips": 0.0})
        cur["backoffs"] = max(cur["backoffs"], c.get("backoffs", 0))
        cur["trips"] = max(cur["trips"], c.get("trips", 0))
        h = by_host.setdefault(r["host"], {"bytes": 0.0, "seconds": 0.0,
                                           "items": 0.0, "backoffs": 0.0,
                                           "trips": 0.0, "requests": 0.0})
        h["requests"] += c.get("requests", 0)
    for (host, _lane), cur in per_lane.items():
        h = by_host.setdefault(host, {"bytes": 0.0, "seconds": 0.0,
                                      "items": 0.0, "backoffs": 0.0,
                                      "trips": 0.0, "requests": 0.0})
        h["backoffs"] += cur["backoffs"]
        h["trips"] += cur["trips"]

    problems = [r for r in items if r["status"] in ("queued", "absent")]
    stamps = sorted(r["at"] for r in records if r.get("at"))
    return {"by_status": by_status, "by_source": by_source,
            "by_host": by_host, "problems": problems,
            "n_items": len(items), "n_lanes": len(lanes),
            "newest_at": stamps[-1] if stamps else None,
            "oldest_at": stamps[0] if stamps else None}


def write_summary(report_path: str, out_path: str, reg=None,
                  state_dir: Optional[str] = None) -> str:
    """summary.md from the union of the progress files and report.jsonl."""
    records = collect(report_path, state_dir)
    s = summarise(records)
    lines: List[str] = []
    a = lines.append

    a("# Import summary")
    a("")
    a(f"`{os.path.abspath(report_path)}` — {s['n_items']} item record(s), "
      f"{s['n_lanes']} lane(s).")
    if state_dir:
        a(f"Merged with the live progress files under "
          f"`{os.path.join(os.path.abspath(state_dir), 'progress')}`.")
    if s["newest_at"]:
        a(f"First record {s['oldest_at']}, newest {s['newest_at']} (UTC).")
    a("")
    a("## Counts by status")
    a("")
    a("| status | items | what it means |")
    a("|---|---:|---|")
    meaning = {
        "written": "shard written, read back, CRC + sha256 matched, marked",
        "present": "a `.done` marker already said so; nothing was fetched",
        "queued": "in retry_queue.jsonl — run again; NOT lost",
        "absent": "the archive said 404/410 twice, >= 6 h apart, with evidence",
    }
    for st, n in sorted(s["by_status"].items(), key=lambda kv: -kv[1]):
        a(f"| `{st}` | {n} | {meaning.get(st, '')} |")
    a("")

    a("## Per host — the politeness audit")
    a("")
    a("| host | lanes | items | requests | bytes | wall | backoffs | trips |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for host, h in sorted(s["by_host"].items()):
        lanes = (reg.host(host)["max_lanes"]
                 if reg and host in reg.hosts else "-")
        a(f"| `{host}` | {lanes} | {int(h['items'])} | {int(h['requests'])} | "
          f"{_human(h['bytes'])} | {h['seconds'] / 3600:.2f} h | "
          f"{int(h['backoffs'])} | {int(h['trips'])} |")
    a("")
    if state_dir:
        from .sinks import list_absent, queue_uri, read_queue
        n_queued = len(read_queue(queue_uri(state_dir)))
        absent = list_absent(state_dir)
        a(f"**Queue: {n_queued} item(s)** in `{queue_uri(state_dir)}`. "
          + ("The tier is complete." if n_queued == 0 else
             "Run again — nothing is lost, it is deferred."))
        a("")
        a(f"**Absent: {len(absent)} item(s)** — the archive said 404/410 "
          "twice, at least six hours apart, with both responses kept in "
          f"`{os.path.join(state_dir, 'absent_evidence')}`. A human reviews "
          "this list before Stage B.")
        a("")
    trips = sum(h["trips"] for h in s["by_host"].values())
    if trips:
        a(f"**{int(trips)} breaker trip(s).** A host that trips repeatedly is "
          "being asked for too much: LOWER its `max_lanes` or RAISE its "
          "`min_gap_s` in sources.yaml before the next run. Never the other "
          "way round.")
    else:
        a("No breaker trips. Every lane ran to the end of its items.")
    a("")

    a("## Per source")
    a("")
    a("| source | " + " | ".join(sorted(s["by_status"])) + " |")
    a("|---|" + "---:|" * len(s["by_status"]))
    for src, counts in sorted(s["by_source"].items()):
        row = " | ".join(str(counts.get(st, 0)) for st in sorted(s["by_status"]))
        a(f"| `{src}` | {row} |")
    a("")

    a("## Missing days kept and re-queued")
    a("")
    md = [(r["item_id"], r.get("missing_dates") or []) for r in
          [x for x in records if x.get("status") in ("written", "present")]]
    md = [(i, d) for i, d in md if d]
    if md:
        a("| item | days the archive did not serve |")
        a("|---|---|")
        for item_id, dates in md[:100]:
            shown = ", ".join(dates[:8]) + (" …" if len(dates) > 8 else "")
            a(f"| `{item_id}` | {len(dates)}: {shown} |")
    else:
        a("None. Every day of every item that was written was served.")
    a("")

    if s["problems"]:
        a("## Everything that is not `written` or `present`")
        a("")
        a("| item | status | reason |")
        a("|---|---|---|")
        for r in s["problems"][:200]:
            err = (r.get("reason") or "").replace("|", "/").replace("\n", " ")
            a(f"| `{r['item_id']}` | {r['status']} | {err[:180]} |")
        if len(s["problems"]) > 200:
            a(f"| … | | {len(s['problems']) - 200} more, see report.jsonl |")
        a("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path


def live(state_dir: str, reg=None) -> str:
    """A mid-run summary from the progress files alone. Prints, returns text.

    Safe to run at any moment, including while lanes are writing: torn last
    lines are skipped and nothing is locked.
    """
    records = collect(None, state_dir)
    s = summarise(records)
    out: List[str] = []
    a = out.append
    prog = os.path.join(os.path.abspath(state_dir), "progress")
    a(f"live progress from {prog}")
    if not records:
        a("  (no records yet — no lane has finished its first item)")
        return "\n".join(out)
    a(f"  newest record  {s['newest_at']} UTC   "
      f"(first {s['oldest_at']})")
    a(f"  records        {s['n_items']} item(s), {s['n_lanes']} lane(s) done")
    a("")
    a("  status per source")
    statuses = sorted(s["by_status"])
    a("    " + f"{'source':16s}" + "".join(f"{st:>11s}" for st in statuses))
    for src, counts in sorted(s["by_source"].items()):
        a("    " + f"{src:16s}"
          + "".join(f"{counts.get(st, 0):>11d}" for st in statuses))
    a("    " + f"{'TOTAL':16s}"
      + "".join(f"{s['by_status'].get(st, 0):>11d}" for st in statuses))
    a("")
    a("  per host")
    a(f"    {'host':14s}{'items':>8s}{'bytes':>12s}{'wall':>10s}"
      f"{'backoffs':>10s}{'trips':>7s}")
    for host, h in sorted(s["by_host"].items()):
        a(f"    {host:14s}{int(h['items']):>8d}{_human(h['bytes']):>12s}"
          f"{h['seconds'] / 3600:>9.2f}h{int(h['backoffs']):>10d}"
          f"{int(h['trips']):>7d}")
    from .sinks import list_absent, queue_uri, read_queue
    a("")
    a(f"  queue          {len(read_queue(queue_uri(state_dir)))} item(s)")
    a(f"  absent         {len(list_absent(state_dir))} item(s) "
      "(404/410 twice, >= 6 h apart)")
    trips = int(sum(h["trips"] for h in s["by_host"].values()))
    if trips:
        a("")
        a(f"  {trips} breaker trip(s) so far. Two on one host in one run "
          "means STOP and report.")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="summary.md from report.jsonl, or a live mid-run summary")
    ap.add_argument("--report", default=None,
                    help="report.jsonl (written when the pipeline finishes)")
    ap.add_argument("--live", default=None, metavar="STATE_DIR",
                    help="summarise the live progress files under a "
                         "--state-dir, mid-run")
    ap.add_argument("--state-dir", default=None,
                    help="merge the live progress files into the summary")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.live:
        print(live(args.live))
        return 0
    if not args.report:
        ap.error("one of --report or --live is required")
    out = args.out or os.path.join(os.path.dirname(args.report) or ".",
                                   "summary.md")
    print(write_summary(args.report, out, state_dir=args.state_dir))
    return 0


if __name__ == "__main__":                     # pragma: no cover
    raise SystemExit(main())
