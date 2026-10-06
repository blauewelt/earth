"""Family 7.2d — family 7.2's channels at DAILY resolution (E-087).

PLAIN ENGLISH. Family 7.2 (`family7_global025_pentad_l2`) is the programme's
global input tensor of five-day means. Every one of its channels is built
from DAILY inputs, so family 7.2d is the same channels on the same grids,
one frame per DAY, in the sharded tier-G layout (`ml/family1/sharded.py`):
one compressed shard per five-day bin holding that bin's five daily frames.
One store per SOURCE, so a slow source never blocks the others:

  glorys025d   GLORYS12 ocean reanalysis, 0.25°, 1993 ->: currents, sea-
               surface height, mixed-layer depth
  oisst025d    NOAA OISST v2.1, 0.25°, 1982 ->: sea-surface temperature and
               sea-ice concentration
  ncep100d     NCEP/NCAR Reanalysis 1, 1°, 1982 ->: the 15 atmosphere and
               land channels
  occci025d    ESA OC-CCI v6.0 ocean colour, 0.25°, 1997-09-04 ->

THE DERIVATION IS FAMILY 7'S. Every frame comes from
`ml/family7_daily.py`, which calls the family-7 builder's own helpers in
its own order of operations; the January-2015 probe (E-087 §4) showed the
five-day mean of the daily frames reproduces the published pentad value in
every cell of every channel.

THE LANE REFUSES A BIN THAT DOES NOT REPRODUCE THE PENTAD TENSOR. Before a
bin's five frames are yielded, they are rounded to float16 exactly as the
store will hold them, averaged by each channel's own rule
(`family7_daily.pentad_from_daily`) and compared with the published f7l2
value, read from the Hub by one range request per bin (offsets and norm from
`data/family7_index.json`). A NaN-pattern difference or a difference beyond
the rounding both values have been through (`family7_daily.check_bin`) is
`ctx.note_absent(...)`: the year is NOT marked and the lane fails. The
largest difference and the largest margin per channel go into the year's
counts (`max_pentad_absdiff_<ch>`, `max_pentad_excess_<ch>`; an excess is
<= 0 when the check passes). `F7D_PENTAD_CHECK=off` disables it, and only a
synthetic `--source-dir` build (the smoke) does that by default — it has no
published tensor to compare with.

RECORD. Each store from its source's first day to 2024-12-31, family 7.2's
end (decision Q4); frames outside are `before_record` / `after_record`. A
source file that cannot be read is an ABSENCE (the year is refused); a day a
readable file does not hold is `absent_upstream` (counted).
"""
import datetime as dt
import json
import os
import shutil
import sys
import time

import numpy as np

from family1 import sharded as sh
import build_family10_stores as f10b
import build_family7 as f7
import family7_daily as fd

HUB = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main"
PSL = "https://downloads.psl.noaa.gov/Datasets"
EPOCH = dt.date(1982, 1, 1)
RECORD_END = dt.date(2024, 12, 31)
FAMILY = "72d"
PLAN = "ml/plans/E087_family7_daily.md"
DISK_MARGIN = 1.2
DISK_SPARE_BYTES = 4_000_000_000

LICENCES = {
    "glorys025d": {
        "name": "Copernicus Marine Service data licence (free, attribution)",
        "redistribution": "attribution", "derived_works": "free",
        "attribution": ("E.U. Copernicus Marine Service Information; "
                        "GLORYS12V1 global ocean physics reanalysis, "
                        "cmems_mod_glo_phy_my_0.083deg_P1D-m, "
                        "doi:10.48670/moi-00021 (daily means, binned here "
                        "to 0.25 degrees)")},
    "oisst025d": {
        "name": "NOAA public domain",
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("NOAA OI SST V2.1 High Resolution data provided by "
                        "the NOAA PSL, Boulder, Colorado, USA, "
                        "https://psl.noaa.gov (Huang et al. 2021, "
                        "doi:10.1175/JCLI-D-20-0166.1)")},
    "ncep100d": {
        "name": "NOAA public domain",
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("NCEP-NCAR Reanalysis 1 data provided by the NOAA "
                        "PSL, Boulder, Colorado, USA, https://psl.noaa.gov "
                        "(Kalnay et al. 1996, doi:10.1175/1520-0477(1996)"
                        "077<0437:TNYRP>2.0.CO;2)")},
    "occci025d": {
        "name": "ESA CCI Data Policy: free and open access, cite",
        "redistribution": "attribution", "derived_works": "free",
        "attribution": ("Ocean Colour Climate Change Initiative dataset, "
                        "Version 6.0, European Space Agency, available "
                        "online at http://www.oceancolour.org/ (Sathyendranath "
                        "et al. 2019, doi:10.3390/s19194285)")},
}

# MEASURED stored bytes per frame, January 2015 (E-087 §5.1) — the lane's
# disk preflight sizes its parts from these.
PROBE_BYTES_PER_FRAME = {"glorys025d": 6_180_237, "oisst025d": 893_181,
                         "ncep100d": 1_530_706, "occci025d": 606_547}


def frame_day(b, f):
    return EPOCH + dt.timedelta(days=5 * int(b) + int(f))


class PentadRef:
    """The published family-7.2 pentad slab for one bin, by range read."""

    def __init__(self, store, index_path=fd.INDEX_JSON):
        self.cfg = fd.STORES[store]
        with open(index_path) as fh:
            self.index = json.load(fh)
        self.g = self.index["groups"][self.cfg["group"]]
        self.bytes = 0

    def bin(self, b):
        g = self.g
        H, W, C = g["shape"][1:]
        row = int(b) - int(g.get("bin_first", 0))
        if row < 0 or row >= int(g["shape"][0]):
            return None
        slab = int(g["slab_bytes"])
        blob = f7.http_range(g["url"], int(g["header_len"]) + row * slab,
                             slab)
        self.bytes += len(blob)
        z = np.frombuffer(blob, "<f2").reshape(H, W, C)
        idx = self.cfg["pentad_index"]
        z = z[..., idx].astype(np.float64)
        norm = np.asarray(g["norm"], np.float64)[idx]
        return z * norm[:, 1] + norm[:, 0], z, norm


class F7DailyBase(sh.GridAdapter):
    """One family-7.2 source as a daily sharded tier-G store."""

    family = FAMILY
    distribution = "public"
    credentials = ()
    time_dtype = "int32"
    per_year = True
    frames_per_bin = fd.F
    frame_seconds = fd.FRAME_SECONDS
    dtype = "float16"
    zstd_level = sh.DEFAULT_LEVEL
    log2_dt = float(np.log2(1.0 / 5.0))          # one day
    record_start = None                          # dt.date
    smoke_window = ("2009-12-28", "2010-02-08")
    smoke_probe_month = "2010-01"
    plan = PLAN

    def __init_subclass__(cls, **kw):
        # the registry validates the CLASS, so the per-store constants are
        # class attributes, read once from family7_daily.STORES
        super().__init_subclass__(**kw)
        if getattr(cls, "store", "") in fd.STORES:
            cfg = fd.STORES[cls.store]
            cls.channels = tuple(cfg["channels"])
            cls.tile = int(cfg["tile"])
            cls.licence = dict(LICENCES[cls.store])

    def __init__(self):
        cfg = fd.STORES[self.store]
        self.cfg = cfg
        self.note_estimate = {
            "bytes": int(PROBE_BYTES_PER_FRAME[self.store]
                         * self.record_days()),
            "what": (f"E-087 §6: the January-2015 probe's "
                     f"{PROBE_BYTES_PER_FRAME[self.store] / 1e6:.3f} MB a "
                     f"frame x {self.record_days()} days")}
        self._ref = None

    @property
    def grid(self):
        return dict(self.cfg["grid"])

    def grid_latlon(self, group, x, y):
        return np.asarray(y, np.float64), np.asarray(x, np.float64)

    def specs(self):
        out = super().specs()
        sp = out[self.store]
        sp["frame_rule"] = ("frame f of bin b is the calendar DAY 5b + f "
                            "after 1982-01-01: that day's value, not a mean "
                            "over days")
        sp["family7_group"] = self.cfg["group"]
        sp["family7_channels"] = list(self.cfg["pentad_index"])
        sp["pentad_rule"] = (
            "the five-day mean of a bin's frames by each channel's rule "
            "(ml/family7_daily.py :: pentad_from_daily) reproduces "
            "family7_global025_pentad_l2 to float16 rounding, NaN exactly "
            "where it is NaN; every lane checks every bin (ml/family7_daily."
            "py :: check_bin) and refuses a year that does not")
        return out

    # ------------------------------------------------------------- record --
    def record_days(self):
        return (RECORD_END - self.record_start).days + 1

    def record_frames(self, ctx, group):
        return self.record_days()

    def in_record(self, d):
        if d < self.record_start:
            return "before_record"
        if d > RECORD_END:
            return "after_record"
        return None

    # ------------------------------------------------------------ helpers --
    def check_on(self, ctx):
        v = os.environ.get("F7D_PENTAD_CHECK", "")
        if v:
            return v.lower() not in ("off", "0", "no", "false")
        return not ctx.source_dir

    def ref(self):
        if self._ref is None:
            self._ref = PentadRef(self.store)
        return self._ref

    def get(self, ctx, rel_local, urls, dest):
        """A source file: local under --source-dir, else downloaded (size
        verified) from the first URL that serves. None if no URL has it."""
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, rel_local)
            if os.path.exists(p):
                ctx.count_bytes(os.path.getsize(p))
                return p
            return None
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            return dest
        got = f10b.fetch_first(list(urls), dest, attempts=max(
            1, int(getattr(ctx.a, "attempts", 3) or 3)))
        return dest if got else None

    def drop(self, ctx, path):
        if path and not ctx.source_dir and os.path.exists(path):
            os.remove(path)

    def fetch_preflight(self, ctx):
        bins = sum(len(v) for v in (getattr(ctx, "grid_bins", None)
                                    or {}).values())
        frames = bins * self.frames_per_bin
        per = PROBE_BYTES_PER_FRAME[self.store]
        need = int(frames * per * DISK_MARGIN) + DISK_SPARE_BYTES
        os.makedirs(ctx.parts, exist_ok=True)
        free = shutil.disk_usage(ctx.parts).free
        print(f"  disk: {free / 1e9:.1f} GB free at {ctx.parts}; {frames} "
              f"frames project to {frames * per / 1e9:.1f} GB of parts, "
              f"{need / 1e9:.1f} GB with margin and the source in flight",
              flush=True)
        if not ctx.source_dir and free < need:
            sys.exit(f"REFUSING {self.store}: the lane needs ~"
                     f"{need / 1e9:.1f} GB and the disk has "
                     f"{free / 1e9:.1f} GB free. Split the window into "
                     f"shorter year ranges. Nothing has been fetched.")
        print(f"  pentad consistency check: "
              f"{'ON' if self.check_on(ctx) else 'OFF'}", flush=True)

    def index(self, ctx):
        return {"dataset": self.title, "plan": PLAN,
                "record": [str(self.record_start), str(RECORD_END)],
                "sources": list(self.sources),
                "pentad_reference": fd.load_index()["groups"][
                    self.cfg["group"]]["url"],
                "pentad_check": self.check_on(ctx),
                "grid": self.grid}

    def check_allow(self, b):
        """{channel: absolute allowance} for this bin (check_bin `allow`)."""
        return {}

    def check_extra(self, b):
        """Further check_bin keywords for this bin (glorys: mld_nonpos)."""
        return {}

    # ------------------------------------------------------------ frames ---
    def days_frames(self, ctx, days):
        """Yield (day, frame | None, why | None) for `days`, chronological.
        `why` is None for a frame; "absent_upstream" for a day the readable
        source does not hold; the string "ABSENT:<what>" for an input that
        could not be read (the caller notes it absent)."""
        raise NotImplementedError

    def fetch_frames(self, ctx, wanted):
        groups = {g for g, _b, _f in wanted}
        if groups - {self.store}:
            sys.exit(f"{self.store}: asked for groups {sorted(groups)}")
        F = self.frames_per_bin
        want = {(b, f) for _g, b, f in wanted}
        bins = sorted({b for b, _f in want})
        days = [frame_day(b, f) for b in bins for f in range(F)
                if (b, f) in want]
        need = [d for d in days if self.in_record(d) is None]
        got = {}
        it = self.days_frames(ctx, need)
        check = self.check_on(ctx)
        H, W, C = self.cfg["grid"]["H"], self.cfg["grid"]["W"], \
            len(self.channels)
        for b in bins:
            frames, why, counts = [], [], {}
            whole = all((b, f) in want for f in range(F))
            for f in range(F):
                d = frame_day(b, f)
                r = self.in_record(d)
                if (b, f) not in want:              # a probe's partial bin
                    frames.append(None)
                    why.append("not_wanted")
                    continue
                if r:
                    frames.append(None)
                    why.append(r)
                    continue
                while d not in got:
                    try:
                        dd, arr, w = next(it)
                    except StopIteration:
                        sys.exit(f"{self.store}: the source iterator ended "
                                 f"before {d}")
                    got[dd] = (arr, w)
                arr, w = got.pop(d)
                if w and w.startswith("ABSENT:"):
                    frames.append(None)
                    why.append(w)
                    continue
                if arr is not None:
                    arr = np.array(arr, np.float32)
                    oob = self.mask_bounds(arr.reshape(-1, C))
                    if oob:
                        f10b._merge_counts(counts, {"out_of_bounds": oob})
                frames.append(arr)
                why.append(w)
            absent = [w for w in why if w and w.startswith("ABSENT:")]
            if absent:
                ctx.note_absent(f"{frame_day(b, 0).year} bin {b}",
                                absent[0][7:])
                continue
            if check and whole and any(a is not None for a in frames):
                rr = self.ref().bin(b)
                if rr is not None:
                    P, z, norm = rr
                    stored = [np.full((H, W, C), np.nan, np.float32)
                              if a is None else
                              a.astype(np.float16).astype(np.float32)
                              for a in frames]
                    allow = self.check_allow(b)
                    extra = self.check_extra(b)
                    res = fd.check_bin(self.store, stored, P, z, norm,
                                       allow=allow, **extra)
                    c2 = {"pentad_bins_checked": 1}
                    for k, v in extra.items():
                        c2[f"pentad_{k}_cell_days"] = int(np.sum(v))
                    for nm, x in allow.items():
                        c2[f"max_pentad_allow_{nm}"] = float(x)
                    for nm, v in res["channels"].items():
                        c2[f"max_pentad_absdiff_{nm}"] = v["max_abs_diff"]
                        c2[f"max_pentad_excess_{nm}"] = v["max_excess"]
                        if v["nan_mismatch"]:
                            c2.setdefault("pentad_nan_mismatch", {})[nm] = \
                                v["nan_mismatch"]
                    f10b._merge_counts(counts, c2)
                    if not res["ok"]:
                        bad = {nm: v for nm, v in res["channels"].items()
                               if v["nan_mismatch"] or v["max_excess"] > 0}
                        ctx.note_absent(
                            f"{frame_day(b, 0).year} bin {b}",
                            f"PENTAD CONSISTENCY REFUSED: the five daily "
                            f"frames of bin {b} ({frame_day(b, 0)}) do not "
                            f"reproduce family7_global025_pentad_l2 within "
                            f"float16 rounding: {json.dumps(bad)[:600]}")
                        continue
            first = True
            for f in range(F):
                if (b, f) not in want:
                    continue
                c = dict(counts) if first else {}
                first = False
                if frames[f] is None:
                    c["frame_missing"] = why[f]
                    yield self.store, b, f, None, c
                else:
                    yield self.store, b, f, frames[f], c
        it.close()

    # ------------------------------------------------------------- smoke ---
    def smoke_sources(self, root, d_lo, d_hi):
        """f7's own synthetic sources, and the truth from this adapter's
        derivation (stored dtype, bounds masked) — the framework's smoke then
        tests the PLUMBING (writer, parts, assembly, check); the derivation
        itself is pinned against the real family-7 builder in
        tests/test_family7_daily.py."""
        days = [d_lo + dt.timedelta(days=k)
                for k in range((d_hi - d_lo).days + 1)]
        f7.make_smoke_sources(root, d_lo, d_hi)
        if self.store == "occci025d":
            f7.make_smoke_oc_sources(root, days, dt.date(2010, 1, 18))

        class _C:                                   # a minimal ctx
            source_dir = root
            scratch = os.path.join(root, "_scratch")
            absent = []

            class a:
                attempts = 1

            @staticmethod
            def count_bytes(n):
                pass

            def note_absent(self, unit, why):
                self.absent.append((unit, why))
        c = _C()
        lo = sh.bins_overlapping(f10b.seconds_since_epoch(d_lo),
                                 f10b.seconds_since_epoch(d_hi) + 86399)
        want = [frame_day(b, f) for b in lo for f in range(self.frames_per_bin)]
        need = [d for d in want if self.in_record(d) is None]
        got = {d: (a, w) for d, a, w in self.days_frames(c, need)}
        truth = {}
        C = len(self.channels)
        for b in lo:
            for f in range(self.frames_per_bin):
                d = frame_day(b, f)
                r = self.in_record(d)
                if r:
                    truth[(self.store, b, f)] = (None, r)
                    continue
                a, w = got[d]
                if a is None:
                    truth[(self.store, b, f)] = (None, w)
                    continue
                a = np.array(a, np.float32)
                self.mask_bounds(a.reshape(-1, C))
                truth[(self.store, b, f)] = (a.astype(np.float16), None)
        return truth
