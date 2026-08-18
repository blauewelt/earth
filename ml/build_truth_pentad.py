#!/usr/bin/env python3
"""E-034 step 1: the transport truth records at 5-DAY cadence.

Why this is the first thing built. `probe_state_ceiling.py` measured that the
transport probe sits at a STATE/LABEL ceiling and not a representation tax —
pooled embeddings score 0.627 where the *true uncompressed fields* score
0.631. No model change can move that number. More labels can, and the labels
already exist at far finer cadence than we use them:

    RAPID 26.5 N   12-HOURLY   14,599 samples since 2004-04-01  (we use 240)
    Florida cable  DAILY       since 1982                       (we use 516)
    MOVE 16 N      ~daily      2000-2022
    SAMBA 34.5 S   daily       2009-2017

Both headline cadences were MEASURED from the live archives on 2026-08-16
(root CLAUDE.md: never guess what an archive serves — ask it), not read from
documentation: RAPID's netCDF reports `days since 2004-4-1` with a median
step of exactly 0.5, and the cable's per-year file announces itself as
`DAILY FLORIDA CURRENT TRANSPORT` with 365 rows for 2015.

So monthly-meaning throws away roughly 6x the labels at pentad and 30x at
daily. This script stops throwing them away.

WHAT A PENTAD IS HERE, stated once because a calendar choice this small is
exactly the kind that silently disagrees between two scripts. Pentads are
fixed 5-day bins counted from 1982-01-01, index = floor(days_since_epoch/5).
Not month-relative (which would make pentads of unequal length and put a
seam at every month boundary), not ISO weeks (7 days breaks the 73-per-year
alignment that makes seasonal statistics clean). A year is 73 pentads, with
the 366th day of a leap year falling in the last bin of that year — the one
irregularity, and it is confined to one bin per four years rather than
smeared.

Output: ml/cache/truth_pentad.npz with, per record, a [n, 2] float32 array
of (pentad_index, transport) plus the epoch and a per-bin sample count, so a
downstream consumer can weight or reject thin bins rather than discovering
them.

Run:  python3 ml/build_truth_pentad.py
"""
import datetime as dt
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TRUTH = os.path.join(CACHE, "truth")
OUT = os.path.join(CACHE, "truth_pentad.npz")

EPOCH = dt.date(1982, 1, 1)        # the tensor's own axis start
PENTAD_DAYS = 5

RAPID_URL = "https://rapid.ac.uk/sites/default/files/rapid_data/moc_transports.nc"
FC_BASE = "https://www.aoml.noaa.gov/ftp/phod/WBTS/cable"
UA = {"User-Agent": "blauewelt-earth/1.0 (research; contact via github)"}


def pentad_of(d):
    """Fixed 5-day bin index from EPOCH. See the module docstring."""
    return (d - EPOCH).days // PENTAD_DAYS


def pentad_start(idx):
    return EPOCH + dt.timedelta(days=int(idx) * PENTAD_DAYS)


def fetch(url, name, attempts=3):
    os.makedirs(TRUTH, exist_ok=True)
    path = os.path.join(TRUTH, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, \
                    open(path, "wb") as f:
                f.write(r.read())
            return path
        except Exception as e:                       # noqa: BLE001
            last = e
            print(f"  attempt {i + 1} failed: {e}")
    raise RuntimeError(f"could not fetch {url}: {last}")


# Physical bounds per record, in Sverdrups. These are GUARDS, not filters:
# a value outside them means a fill value or a unit error reached the
# arithmetic, and the script refuses rather than writing a plausible-looking
# number. Ranges are deliberately generous — they are meant to catch -99999,
# not to police oceanography.
BOUNDS = {"rapid": (-15.0, 45.0), "fc": (5.0, 60.0),
          "move": (-60.0, 20.0), "samba": (-40.0, 40.0)}


def bin_to_pentads(dates, vals, label):
    """(pentad_index, mean transport, n samples) — bins with NO samples are
    simply absent rather than written as NaN. A results file full of NaN is
    loud enough to notice and quiet enough to misattribute (ml/CLAUDE.md
    §5.22); an absent bin cannot be mistaken for a measurement of zero."""
    idx = np.array([pentad_of(d) for d in dates])
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    idx, v = idx[ok], v[ok]
    if not len(idx):
        raise RuntimeError(f"{label}: no finite samples")
    lo, hi = BOUNDS.get(label, (-1e9, 1e9))
    bad = (v < lo) | (v > hi)
    if bad.any():
        ex = np.unique(v[bad])[:5]
        raise RuntimeError(
            f"{label}: {int(bad.sum())} of {len(v)} samples fall outside the "
            f"physical range [{lo}, {hi}] Sv — e.g. {ex}. This is a fill "
            f"value or a unit error reaching the arithmetic, not ocean; "
            f"refusing to write it (ml/CLAUDE.md §5.22).")
    uniq = np.unique(idx)
    out = np.zeros((len(uniq), 2), np.float32)
    cnt = np.zeros(len(uniq), np.int32)
    for i, u in enumerate(uniq):
        m = idx == u
        out[i] = (u, v[m].mean())
        cnt[i] = int(m.sum())
    span = f"{pentad_start(uniq[0])} .. {pentad_start(uniq[-1])}"
    print(f"  {label}: {len(v):,} samples -> {len(uniq):,} pentads "
          f"({span}) · median {int(np.median(cnt))} samples/bin")
    return out, cnt


def rapid():
    import netCDF4
    path = fetch(RAPID_URL, "moc_transports.nc")
    d = netCDF4.Dataset(path)
    t = d.variables["time"]
    # units string is authoritative — parse it rather than assuming 2004-04-01
    units = t.units
    base = units.split("since", 1)[1].strip().split()[0]
    y, m, dd = (int(x) for x in base.replace("/", "-").split("-"))
    t0 = dt.date(y, m, dd)
    days = np.array(t[:], float)
    step = float(np.median(np.diff(days)))
    print(f"  rapid: units {units!r} · median step {step} d "
          f"({'12-hourly' if abs(step - 0.5) < 1e-6 else 'UNEXPECTED'})")
    if abs(step - 0.5) > 1e-6:
        print("  ::warning:: RAPID cadence is not the 0.5 d this script was "
              "written against — check before trusting the bin counts")
    # MASK HANDLING, and it is not optional. netCDF4 returns a MASKED array;
    # `np.array(x, float)` silently strips the mask and hands back the file's
    # raw fill value, which for this product is -99999. The first run of this
    # script did exactly that and produced a mean RAPID transport of
    # -195 Sv — wrong by four orders of magnitude, and wrong in a way that
    # still "looked like data". Same defect the Argo bake documents
    # (np.ma.filled, never np.array), now in a second place.
    raw = d.variables["moc_mar_hc10"][:]
    vals = np.ma.filled(np.ma.masked_invalid(raw).astype(float), np.nan)
    dates = [t0 + dt.timedelta(days=float(x)) for x in days]
    return bin_to_pentads(dates, vals, "rapid")


def fc():
    """Per-year .dat: columns y m d transport [flag]. Header lines start %."""
    dates, vals = [], []
    this_year = dt.date.today().year
    for y in range(1982, this_year + 1):
        path = None
        for suffix in (f"_{y}_v3.dat", f"_{y}_v2.dat", f"_{y}.dat"):
            try:
                path = fetch(f"{FC_BASE}/FC_cable_transport{suffix}",
                             f"fc_{y}{suffix}", attempts=1)
                break
            except Exception:
                path = None
        if not path:
            continue
        for line in open(path):
            if line.lstrip().startswith("%"):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            try:
                yy, mm, dd, tv = int(p[0]), int(p[1]), int(p[2]), float(p[3])
                dates.append(dt.date(yy, mm, dd))
                vals.append(tv)
            except (ValueError, TypeError):
                continue
    return bin_to_pentads(dates, vals, "fc")


def main():
    # --days 1 is the family-5 axis (E-038 Phase B): same epoch, same code,
    # bin = the calendar day. The knob mutates the module global because
    # pentad_of/pentad_start close over it — one binning definition, not two.
    import argparse
    global PENTAD_DAYS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, choices=(5, 1),
                    help="bin width: 5 -> truth_pentad.npz, 1 -> truth_daily.npz")
    a = ap.parse_args()
    PENTAD_DAYS = a.days
    if a.days == 1:
        OUT = os.path.join(CACHE, "truth_daily.npz")
    os.makedirs(CACHE, exist_ok=True)
    print(f"bins: {PENTAD_DAYS}-day bins from {EPOCH} "
          f"({365 // PENTAD_DAYS} per common year)")
    store = {"epoch": np.array(str(EPOCH)),
             "pentad_days": np.array(PENTAD_DAYS)}
    got = []
    for name, fn in (("rapid", rapid), ("fc", fc)):
        try:
            arr, cnt = fn()
            store[f"truth_{name}"] = arr
            store[f"count_{name}"] = cnt
            got.append(name)
        except Exception as e:                       # noqa: BLE001
            # Best effort is a promise about DELIVERY, never about REPORTING
            # (ml/CLAUDE.md §4.6): say which record failed and why.
            print(f"  ::warning:: {name} failed and is ABSENT from the "
                  f"output: {e}")
    if not got:
        sys.exit("no truth record could be built — nothing written")
    np.savez_compressed(OUT, **store)
    print(f"\nwrote {OUT} ({os.path.getsize(OUT):,} bytes) — {', '.join(got)}")
    # The comparison that justifies the whole exercise, printed rather than
    # claimed, so the label multiplier is a measurement in the run log.
    for name in got:
        n_p = len(store[f"truth_{name}"])
        months = len({(pentad_start(i).year, pentad_start(i).month)
                      for i in store[f"truth_{name}"][:, 0]})
        print(f"  {name}: {n_p:,} pentad labels vs {months:,} monthly "
              f"labels = {n_p / max(months, 1):.1f}x")


if __name__ == "__main__":
    main()
