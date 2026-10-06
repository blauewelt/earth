"""NCEP/NCAR Reanalysis 1 at DAILY resolution, 1° — family 7.2d (E-087).

PLAIN ENGLISH. The NCEP/NCAR reanalysis is the US weather model run over the
observational record, four analyses a day. Family 7.2's 15 `g100` channels —
wind stress and its variability, 2 m air temperature, 10 m wind, surface
pressure, rain, snow, soil moisture and temperature, the two turbulent heat
fluxes and the skin temperature — are five-day means of them; this store
keeps each DAY: the mean of the day's four 6-hourly samples on the model's
own grid, then family 7.2's bilinear to the 1° point grid and its unit
transforms, 1982-01-01 to 2026-03-17 — the last day NOAA PSL's yearly files
hold (measured 2026-10-06: all thirteen 2026 files stop at 2026-03-17 18Z and
were last modified 2026-03-19; PSL has not updated them since).

`tau_x_std` / `tau_y_std` ARE NOT A ONE-DAY VALUE: they are the population
standard deviation of the 6-hourly wind stress over the FIVE DAYS CENTRED on
the day (decision Q3), so frame 2 of every bin equals family 7.2's pentad
value exactly; written where >= 3 of those five days are in the record.
"""
import datetime as dt
import glob
import os

import numpy as np

from family1.adapters import _f7d
import build_family7 as f7
import family7_daily as fd

MIRROR = f"{_f7d.HUB}/mirrors/psl/Datasets/ncep.reanalysis/surface_gauss"
PSL = f"{_f7d.PSL}/ncep.reanalysis/surface_gauss"


class NCEP100DAdapter(_f7d.F7DailyBase):
    store = "ncep100d"
    title = ("NCEP/NCAR Reanalysis 1 surface and land channels (family "
             "7.2's g100), DAILY means, 1 degree — family 7.2d")
    first_year = 1982
    record_start = dt.date(1982, 1, 1)
    record_end = dt.date(2026, 3, 17)          # NOAA PSL, measured 2026-10-06
    log2_fp = 2.0            # a 1-degree point from a T62 (~1.9 deg) model
    qc_policy = (
        "The reanalysis has no per-value flag. The day is the NaN-aware "
        "mean of its 6-hourly samples on the T62 gaussian grid; soilw and "
        "tsoil are NaN over sea (land.sfc.gauss applied before the "
        "bilinear). uflx/vflx are sign-flipped to the stress ON the surface. "
        "Out-of-bounds values become NaN and are counted, never clipped.")
    sources = (f"{MIRROR}/<var>.gauss.<YYYY>.nc + land.sfc.gauss.nc (the "
               f"Hub's round-trip-verified mirror of NOAA PSL)",
               f"{PSL}/<var>.gauss.<YYYY>.nc (fallback)")
    verified = ("2026-10-06: 13 variables x 43 years + the land mask (560 "
                "files, 21.4 GB) on the Hub mirror; 2015's opened (94 x 192, "
                "1,460 six-hourly steps)")
    notes = ("Family 7.2d: family 7.2's g100 channels one frame per day. "
             "tau_x_std and tau_y_std are the CENTRED FIVE-DAY sigma of the "
             "6-hourly stress (named so in their units), so frame 2 of every "
             "bin is family 7.2's within-pentad sigma.")

    def path(self, ctx, name):
        return self.get(ctx, f"ncep/{name}", [f"{MIRROR}/{name}",
                                              f"{PSL}/{name}"],
                        os.path.join(ctx.scratch, "ncep", name))

    def days_frames(self, ctx, days):
        if not days:
            return
        lo = min(days) - dt.timedelta(days=2)
        hi = min(max(days) + dt.timedelta(days=2), self.record_end)
        years = [y for y in range(lo.year, hi.year + 1)
                 if self.record_start.year <= y <= self.record_end.year]
        if not ctx.source_dir:          # keep only what this year can use
            for p in glob.glob(os.path.join(ctx.scratch, "ncep",
                                            "*.gauss.*.nc")):
                try:
                    yy = int(os.path.basename(p).split(".")[-2])
                except ValueError:
                    continue
                if yy < years[0]:
                    os.remove(p)
        land_p = self.path(ctx, f"{f7.NCEP_LAND}.nc")
        paths, missing = {}, []
        for v in fd.NCEP_ORDER:
            paths[v] = []
            for y in years:
                n = f"{f7.NCEP_FILES[v]}.{y}.nc"
                p = self.path(ctx, n)
                if p is None:
                    missing.append(n)
                else:
                    paths[v].append(p)
        if land_p is None or missing:
            what = ", ".join(missing[:4]) or f"{f7.NCEP_LAND}.nc"
            for d in days:
                yield d, None, f"ABSENT:NCEP {what} could not be read"
            return
        import netCDF4 as ncdf
        dl = ncdf.Dataset(land_p)
        land = f7.squeeze_level(np.ma.filled(
            np.asarray(f7.pick_var(dl, "land")[:]), 0.0)) >= 0.5
        dl.close()
        self._neg = {}
        fr = fd.ncep_daily(paths, land, days, negmin=self._neg)
        if self.check_on_flag:
            from family1 import f7d_hub_check as hc
            for d in days:
                if not (self.wants_reference(d) and d in fr):
                    continue
                try:
                    yp = {v: [p for p in paths[v]
                              if p.endswith(f".{d.year}.nc")][0]
                          for v in fd.NCEP_ORDER}
                    self.stash_reference(d, hc.ref_ncep_from(yp, land_p, d))
                except Exception as e:                       # noqa: BLE001
                    self.stash_reference(d, e)
        for d in days:
            if d in fr:
                yield d, fr.pop(d), None
            else:
                yield d, None, "absent_upstream"


    def source_segments(self):
        return [{"from": "1982-01-01", "to": "2024-12-31",
                 "source": "NCEP/NCAR Reanalysis 1 (PSL surface_gauss yearly "
                           "files)", "falsifier": "pentad"},
                {"from": "2025-01-01", "to": str(self.record_end),
                 "source": "NCEP/NCAR Reanalysis 1 (PSL surface_gauss yearly "
                           "files)", "falsifier": "source-readback",
                 "note": "PSL's 2026 files end 2026-03-17 18Z (last modified "
                         "2026-03-19)"}]

    def check_allow(self, b):
        """The per-day clamp at zero in `log1p_channel` (ncep_daily
        `negmin`): a day whose prate/weasd samples dip below zero by x can
        differ from the pentad's single clamp by at most |x| times the unit
        scale (86400 for prate, 1 for weasd) — d log1p <= d at zero."""
        neg = getattr(self, "_neg", {}) or {}
        out = {}
        for f in range(self.frames_per_bin):
            for v, x in neg.get(_f7d.frame_day(b, f), {}).items():
                ch, sc = (("log_prate", 86400.0) if v == "prate"
                          else ("log_swe", 1.0))
                out[ch] = max(out.get(ch, 0.0), -x * sc)
        return out


ADAPTER = NCEP100DAdapter
