"""NOAA OISST v2.1 at DAILY resolution, 0.25° — family 7.2d (E-087).

PLAIN ENGLISH. OISST is NOAA's daily optimum-interpolation analysis of
satellite and in-situ sea-surface temperature, with a sea-ice concentration
field beside it. Family 7.2's `sst` and `sea_ice` channels are five-day means
of these days; this store keeps each day, interpolated bilinearly from
OISST's cell centres onto the 0.25° point grid exactly as family 7.2 does,
1982-01-01 to 2024-12-31. `sst` is NaN where OISST does not observe (land);
`sea_ice` is NaN below 15 % ice (OISST's own mask), so open water is NaN,
not 0.
"""
import datetime as dt
import os

from family1.adapters import _f7d
import family7_daily as fd

MIRROR = f"{_f7d.HUB}/mirrors/psl/Datasets/noaa.oisst.v2.highres"
PSL = f"{_f7d.PSL}/noaa.oisst.v2.highres"
DAYS_PER_READ = 10


class OISST025DAdapter(_f7d.F7DailyBase):
    store = "oisst025d"
    title = ("NOAA OISST v2.1 sea-surface temperature and sea-ice "
             "concentration, DAILY, 0.25 degrees — family 7.2d")
    first_year = 1982
    record_start = dt.date(1982, 1, 1)
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
                for d in chunk:
                    if d in fr:
                        yield d, fr.pop(d), None
                    else:
                        yield d, None, "absent_upstream"
            self.drop(ctx, ps)
            self.drop(ctx, pi)


ADAPTER = OISST025DAdapter
