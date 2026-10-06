"""GLORYS12 ocean reanalysis at DAILY resolution, 0.25° — family 7.2d (E-087).

PLAIN ENGLISH. Mercator Ocean's GLORYS12 reanalysis (an ocean model pulled
towards the observations) gives the surface currents, the sea-surface height
and the mixed-layer depth once a day. Family 7.2's `g025` group averages
them over five days; this store keeps each day: `cur_speed`, `log_mld`
(log10 of the mixed-layer depth in metres), `ssh`, `cur_u`, `cur_v`, on the
0.25° point grid pole to pole (NaN south of 80° S and over land), 1993-01-01
to 2024-12-31. The inputs are the 1/12° daily fields already binned to 0.25°
and parked on the Hub (`daily025_global/`, 384 monthly chunks) — the bytes
family 7.2 was built from — so no Copernicus credential is used.
"""
import datetime as dt
import os

import numpy as np

from family1.adapters import _f7d
import family7_daily as fd


class GLORYS025DAdapter(_f7d.F7DailyBase):
    store = "glorys025d"
    title = ("GLORYS12 ocean reanalysis surface currents, sea-surface height "
             "and mixed-layer depth, DAILY, 0.25 degrees — family 7.2d")
    first_year = 1992        # the year folder 1992 holds bin 803 (1992-12-29
    #                          .. 1993-01-02), whose two 1993 days are data
    record_start = dt.date(1993, 1, 1)
    log2_fp = 0.0            # a 0.25-degree box mean, ~27.8 km
    qc_policy = (
        "GLORYS12 publishes no per-value flag. NaN over land, under ice "
        "shelves and south of 80 S (rows 0..39, outside the reanalysis). "
        "`log_mld` is log10 of a positive depth, NaN otherwise. A value "
        "outside its sanity bounds becomes NaN and is counted, never "
        "clipped; the January-2015 probe counted none.")
    sources = (f"{_f7d.HUB}/daily025_global/glorys025_global_<YYYYMM>.nc "
               f"(GLORYS12 cmems_mod_glo_phy_my_0.083deg_P1D-m, daily, "
               f"binned to 0.25 deg at fetch by ml/fetch_glorys_daily.py)",)
    verified = ("2026-10-06: 384 of 384 chunks listed on the Hub (1993-01 .. "
                "2024-12, 96,800,818,463 bytes); 2015-01 and 2015-02 opened "
                "(681 x 1440, -80..90, one value a day, uo vo mlotst zos)")
    notes = ("Family 7.2d: family 7.2's g025 channels 0-4 one frame per day. "
             "cur_speed = hypot(cur_u, cur_v) of the day; the pentad rule is "
             "hypot of the MEAN u and v, log10 of the MEAN depth "
             "(ml/family7_daily.py :: pentad_from_daily).")

    def days_frames(self, ctx, days):
        months = []
        for d in days:
            if (d.year, d.month) not in months:
                months.append((d.year, d.month))
        for (y, m) in months:
            name = f"glorys025_global_{y}{m:02d}.nc"
            mdays = [d for d in days if (d.year, d.month) == (y, m)]
            p = self.get(ctx, f"daily025_global/{name}",
                         [f"{_f7d.HUB}/daily025_global/{name}"],
                         os.path.join(ctx.scratch, "glorys", name))
            if p is None:
                for d in mdays:
                    yield d, None, f"ABSENT:{name} could not be read"
                continue
            fr = fd.glorys_chunk(p, mdays)
            for d in mdays:
                if d in fr:
                    yield d, fr.pop(d), None
                else:
                    yield d, None, "absent_upstream"
            self.drop(ctx, p)


ADAPTER = GLORYS025DAdapter
