"""NOAA OISST v2.1 at DAILY resolution, 0.25° — family 7.2d (E-087).

PLAIN ENGLISH. OISST is NOAA's daily optimum-interpolation analysis of
satellite and in-situ sea-surface temperature, with a sea-ice concentration
field beside it. Family 7.2's `sst` and `sea_ice` channels are five-day means
of these days; this store keeps each day, interpolated bilinearly from
OISST's cell centres onto the 0.25° point grid exactly as family 7.2 does,
1982-01-01 to 2026-10-04, the last day NOAA PSL's yearly file held on
2026-10-06. NCEI publishes the most recent days as PRELIMINARY and replaces
them with final ones about two weeks later (final through 2026-09-20 when
measured); each preliminary day is listed by date in the year's counts
(`preliminary_days`), read from NCEI's own per-day file names. `sst` is NaN where OISST does not observe (land);
`sea_ice` is NaN below 15 % ice (OISST's own mask), so open water is NaN,
not 0.
"""
import datetime as dt
import os
import re
import urllib.request

from family1.adapters import _f7d
import family7_daily as fd

MIRROR = f"{_f7d.HUB}/mirrors/psl/Datasets/noaa.oisst.v2.highres"
PSL = f"{_f7d.PSL}/noaa.oisst.v2.highres"
DAYS_PER_READ = 10
NCEI = ("https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-"
        "interpolation/v2.1/access/avhrr")


class OISST025DAdapter(_f7d.F7DailyBase):
    store = "oisst025d"
    title = ("NOAA OISST v2.1 sea-surface temperature and sea-ice "
             "concentration, DAILY, 0.25 degrees — family 7.2d")
    first_year = 1982
    record_start = dt.date(1982, 1, 1)
    record_end = dt.date(2026, 10, 4)          # NOAA PSL, measured 2026-10-06
    log2_fp = 0.0
    qc_policy = (
        "OISST publishes no per-value flag in these files. PSL's missing "
        "value (-9.97e36) is dropped as a sentinel; land is NaN; the `icec` "
        "file's 0..1 valid range is trusted over its 'percent' units string "
        "(build_family7.ice_divisor), and below 0.15 OISST itself reports "
        "no ice (NaN). Bilinear from the cell centres with the seam wrapped; "
        "an all-NaN neighbourhood stays NaN. Out-of-bounds values become NaN "
        "and are counted, never clipped.")
    sources = (f"{MIRROR}/{{sst,icec}}.day.mean.<YYYY>.nc (the Hub's "
               f"round-trip-verified mirror of NOAA PSL)",
               f"{PSL}/{{sst,icec}}.day.mean.<YYYY>.nc (fallback)")
    verified = ("2026-10-06: 43 + 43 yearly files 1982-2024 on the Hub "
                "mirror (23.5 GB); 2015's opened (720 x 1440, 365 days)")
    notes = ("Family 7.2d: family 7.2's g025 channels 5-6 one frame per "
             "day; the pentad is the mean of the finite days, >= 3 of them.")

    def year_paths(self, ctx, y):
        out = []
        for k in ("sst", "icec"):
            n = f"{k}.day.mean.{y}.nc"
            out.append(self.get(ctx, f"oisst/{n}", [f"{MIRROR}/{n}",
                                                    f"{PSL}/{n}"],
                                os.path.join(ctx.scratch, "oisst", n)))
        return out

    def source_segments(self):
        return [{"from": "1982-01-01", "to": "2024-12-31",
                 "source": "NOAA OISST v2.1 (PSL yearly files)",
                 "falsifier": "pentad"},
                {"from": "2025-01-01", "to": str(self.record_end),
                 "source": "NOAA OISST v2.1 (PSL yearly files)",
                 "falsifier": "source-readback",
                 "note": "the most recent days are NCEI-preliminary; they are "
                         "listed per year in counts_by_year.preliminary_days"}]

    def preliminary(self, d):
        """True / False from NCEI's per-day file names for d's month (a day
        with only a `_preliminary` file is preliminary); None if NCEI did not
        answer."""
        key = (d.year, d.month)
        cache = getattr(self, "_prelim", None)
        if cache is None:
            cache = self._prelim = {}
        if key not in cache:
            try:
                req = urllib.request.Request(f"{NCEI}/{d.year}{d.month:02d}/",
                                             headers={"User-Agent": "earth"})
                html = urllib.request.urlopen(req, timeout=120).read().decode()
                fin = set(re.findall(r"oisst-avhrr-v02r01\.(\d{8})\.nc", html))
                pre = set(re.findall(
                    r"oisst-avhrr-v02r01\.(\d{8})_preliminary\.nc", html))
                cache[key] = (fin, pre)
            except Exception:                                # noqa: BLE001
                cache[key] = None
        got = cache[key]
        if got is None:
            return None
        ymd = f"{d:%Y%m%d}"
        return ymd in got[1] and ymd not in got[0]

    def day_counts(self, d):
        if d <= _f7d.PENTAD_END:
            return {}
        p = self.preliminary(d)
        if p is None:
            return {"preliminary_status_unknown_days": [str(d)]}
        return {"preliminary_days": [str(d)]} if p else {}

    def days_frames(self, ctx, days):
        years = sorted({d.year for d in days})
        for y in years:
            ydays = [d for d in days if d.year == y]
            ps, pi = self.year_paths(ctx, y)
            if ps is None or pi is None:
                for d in ydays:
                    yield d, None, f"ABSENT:OISST {y} could not be read"
                continue
            for i in range(0, len(ydays), DAYS_PER_READ):
                chunk = ydays[i:i + DAYS_PER_READ]
                fr = fd.oisst_daily(ps, pi, chunk)
                if self.check_on_flag:
                    from family1 import f7d_hub_check as hc
                    for d in chunk:
                        if self.wants_reference(d) and d in fr:
                            try:
                                self.stash_reference(
                                    d, hc.ref_oisst_from(ps, pi, d))
                            except Exception as e:           # noqa: BLE001
                                self.stash_reference(d, e)
                for d in chunk:
                    if d in fr:
                        yield d, fr.pop(d), None
                    else:
                        yield d, None, "absent_upstream"
            self.drop(ctx, ps)
            self.drop(ctx, pi)


ADAPTER = OISST025DAdapter
