"""GLORYS12 ocean reanalysis at DAILY resolution, 0.25° — family 7.2d (E-087).

PLAIN ENGLISH. Mercator Ocean's GLORYS12 reanalysis (an ocean model pulled
towards the observations) gives the surface currents, the sea-surface height
and the mixed-layer depth once a day. Family 7.2's `g025` group averages
them over five days; this store keeps each day: `cur_speed`, `log_mld`
(log10 of the mixed-layer depth in metres), `ssh`, `cur_u`, `cur_v`, on the
0.25° point grid pole to pole (NaN south of 80° S and over land), 1993-01-01
to 2026-08-18, the last day Copernicus Marine serves (measured 2026-10-06).
The inputs are the 1/12° daily fields binned to 0.25° and parked on the Hub
(`daily025_global/`, monthly chunks) — the bytes family 7.2 was built from
through 2024-12. A month the Hub does not yet hold (2025-01 on) is fetched
IN THE LANE from Copernicus Marine by `ml/fetch_glorys_daily.py`'s own
request-and-bin code — the SAME dataset id, `cmems_mod_glo_phy_my_0.083deg_
P1D-m` version 202311, the one family 7.2's chunks came from — and parked on
the Hub beside them before it is used (a month the record ends inside goes to
`daily025_global_tail/`, so a short month never sits under the name of a whole
one). The Copernicus credentials are repository secrets on a GitHub-HOSTED
runner only (ml/CLAUDE.md §6); a box assembles from Hub parts and never sees
them.

THE SEAM, STATED. Copernicus lists ONE dataset and ONE version for the whole
record (`describe`: part "default", released 2023-11-30; no separate interim
`myint` id exists in the catalogue today) and the native files are one family
(`mercatorglorys12v1_gl12_mean_<day>_R<run>.nc`) from 1993 to 2026. The
catalogue's `field_date` is 2021-06-30 — the boundary at which GLORYS12 was
historically continued as a near-real-time "interim" run of the same system;
those 2021-07 .. 2024-12 days are ALREADY in family 7.2. What changes at
2025-01-01 here is the FALSIFIER, not the product: through 2024-12-31 every
bin reproduced family 7.2's pentad; after it every day is checked against an
independent read of its own chunk (the weaker guarantee, `_f7d`).
"""
import calendar
import datetime as dt
import os
import shutil
import subprocess
import sys

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
    record_end = dt.date(2026, 8, 18)        # Copernicus Marine, 2026-10-06
    credentials = ("COPERNICUSMARINE_SERVICE_USERNAME",
                   "COPERNICUSMARINE_SERVICE_PASSWORD")
    log2_fp = 0.0            # a 0.25-degree box mean, ~27.8 km
    qc_policy = (
        "GLORYS12 publishes no per-value flag. NaN over land, under ice "
        "shelves and south of 80 S (rows 0..39, outside the reanalysis). "
        "`log_mld` is log10 of a positive depth, NaN otherwise. A value "
        "outside its sanity bounds becomes NaN and is counted, never "
        "clipped; the January-2015 probe counted none.")
    sources = (f"{_f7d.HUB}/daily025_global/glorys025_global_<YYYYMM>.nc "
               f"(GLORYS12 cmems_mod_glo_phy_my_0.083deg_P1D-m version "
               f"202311, daily, binned to 0.25 deg at fetch by "
               f"ml/fetch_glorys_daily.py; 2025-01 on fetched in the lane by "
               f"the same code and parked beside the others)",
               f"{_f7d.HUB}/daily025_global_tail/glorys025_global_<YYYYMM>.nc "
               f"(the month the record ends inside, partial)")
    verified = ("2026-10-06: 384 of 384 chunks listed on the Hub (1993-01 .. "
                "2024-12, 96,800,818,463 bytes); 2015-01 and 2015-02 opened "
                "(681 x 1440, -80..90, one value a day, uo vo mlotst zos)")
    notes = ("Family 7.2d: family 7.2's g025 channels 0-4 one frame per day. "
             "cur_speed = hypot(cur_u, cur_v) of the day; the pentad rule is "
             "hypot of the MEAN u and v, log10 of the MEAN depth "
             "(ml/family7_daily.py :: pentad_from_daily).")

    FIRST_LANE = "d19921229-19961231"         # the lane the record opens with

    def declared_lanes(self, years):
        """1992..1996 were fetched by ONE named lane whose window starts on
        the bin boundary 1992-12-29; every other year is the unnamed lane."""
        return {y: [self.FIRST_LANE] for y in years if 1992 <= int(y) <= 1996}

    def source_segments(self):
        ds = "cmems_mod_glo_phy_my_0.083deg_P1D-m version 202311"
        return [
            {"from": "1993-01-01", "to": "2021-06-30", "source": ds,
             "kind": "GLORYS12V1 reanalysis (before the catalogue's "
                     "field_date)", "falsifier": "pentad"},
            {"from": "2021-07-01", "to": "2024-12-31", "source": ds,
             "kind": "GLORYS12V1, the same dataset id, after the catalogue's "
                     "field_date 2021-06-30 (historically the near-real-time "
                     "'interim' continuation of the same system); already in "
                     "family 7.2", "falsifier": "pentad"},
            {"from": "2025-01-01", "to": str(self.record_end), "source": ds,
             "kind": "GLORYS12V1, the same dataset id, fetched 2026-10-06 in "
                     "the lane", "falsifier": "source-readback"}]

    # -------------------------------------------------- the tail months ---
    def tail_chunk(self, ctx, y, m):
        """A month the Hub does not hold: fetch it from Copernicus Marine with
        fetch_glorys_daily's request-and-bin code, park it on the Hub
        (restore-verified), return the local path. A month the record ends
        inside is partial and goes to daily025_global_tail/."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        import fetch_glorys_daily as fg
        try:
            import copernicusmarine as cm
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "copernicusmarine"], check=True)
            import copernicusmarine as cm
        last = calendar.monthrange(y, m)[1]
        if dt.date(y, m, last) > self.record_end:
            last = self.record_end.day
        whole = last == calendar.monthrange(y, m)[1]
        name = f"glorys025_global_{y}{m:02d}.nc"
        work = os.path.join(ctx.scratch, "glorys_tail")
        os.makedirs(work, exist_ok=True)
        out = os.path.join(work, name)
        raw = os.path.join(work, "_raw_" + name)
        w = fg.BinnedChunk(out, 0.25, "point", fg.DATASET, "global")
        for d0 in range(1, last + 1):
            for i in range(4):
                try:
                    cm.subset(dataset_id=fg.DATASET, variables=fg.VARIABLES,
                              start_datetime=f"{y}-{m:02d}-{d0:02d}T00:00:00",
                              end_datetime=f"{y}-{m:02d}-{d0:02d}T23:59:59",
                              minimum_depth=0, maximum_depth=1,
                              output_filename=raw, disable_progress_bar=True,
                              **fg.GLOBAL_WINDOW)
                    break
                except Exception:                            # noqa: BLE001
                    if os.path.exists(raw):
                        os.remove(raw)
                    if i == 3:
                        raise
            ctx.count_bytes(os.path.getsize(raw))
            w.append(raw)
            os.remove(raw)
        w.close()
        if w.n != last:
            raise RuntimeError(f"{name}: {w.n} daily slices for {last} days")
        folder = "daily025_global/" if whole else "daily025_global_tail/"
        from huggingface_hub import HfApi, hf_hub_download
        tok = os.environ.get("HF_TOKEN")
        api = HfApi(token=tok)
        rp = folder + name
        api.upload_file(path_or_fileobj=out, path_in_repo=rp,
                        repo_id="chfrank/earth-tensors", repo_type="dataset",
                        commit_message=f"glorys daily {y}-{m:02d} "
                                       f"({'whole' if whole else 'to day ' + str(last)}, E-087)")
        vf = os.path.join(work, "vf")
        back = hf_hub_download("chfrank/earth-tensors", rp,
                               repo_type="dataset", token=tok, cache_dir=vf)
        if fg.sha256(back) != fg.sha256(out):
            raise RuntimeError(f"{rp} restored with a different sha256")
        shutil.rmtree(vf, ignore_errors=True)
        print(f"  glorys tail {y}-{m:02d}: {w.n} day(s) fetched from "
              f"Copernicus Marine, parked at {rp} (restore-verified)",
              flush=True)
        return out

    def check_extra(self, b):
        days = [_f7d.frame_day(b, f) for f in range(self.frames_per_bin)]
        m = [self._nonpos.pop(d) for d in days if d in self._nonpos]
        if not m:
            return {}
        cnt = np.sum(m, axis=0).astype(np.int64)
        self._last_nonpos = int(cnt.sum())
        return {"mld_nonpos": cnt}

    def days_frames(self, ctx, days):
        self._nonpos = {}
        months = []
        for d in days:
            if (d.year, d.month) not in months:
                months.append((d.year, d.month))
        for (y, m) in months:
            name = f"glorys025_global_{y}{m:02d}.nc"
            mdays = [d for d in days if (d.year, d.month) == (y, m)]
            p = self.get(ctx, f"daily025_global/{name}",
                         [f"{_f7d.HUB}/daily025_global/{name}",
                          f"{_f7d.HUB}/daily025_global_tail/{name}"],
                         os.path.join(ctx.scratch, "glorys", name))
            if p is not None and dt.date(y, m, 1) > _f7d.PENTAD_END:
                # a tail chunk parked by an earlier lane must cover what this
                # lane wants; a short one is refetched, not trusted
                import netCDF4 as ncdf
                with ncdf.Dataset(p) as dd:
                    n_have = len(dd.variables["time"])
                need_n = max(d.day for d in mdays)
                if n_have < need_n:
                    os.remove(p)
                    p = None
            if p is None and not ctx.source_dir and \
                    dt.date(y, m, 1) > _f7d.PENTAD_END:
                try:
                    p = self.tail_chunk(ctx, y, m)
                except Exception as e:                       # noqa: BLE001
                    for d in mdays:
                        yield d, None, (f"ABSENT:{name}: not on the Hub and "
                                        f"the Copernicus fetch failed "
                                        f"({type(e).__name__}: "
                                        f"{str(e)[:200]})")
                    continue
            if p is None:
                for d in mdays:
                    yield d, None, f"ABSENT:{name} could not be read"
                continue
            fr = fd.glorys_chunk(p, mdays, nonpos=self._nonpos)
            if self.check_on_flag:
                from family1 import f7d_hub_check as hc
                for d in mdays:
                    if self.wants_reference(d) and d in fr:
                        try:
                            self.stash_reference(d, hc.ref_glorys_from(p, d))
                        except Exception as e:               # noqa: BLE001
                            self.stash_reference(d, e)
            for d in mdays:
                if d in fr:
                    yield d, fr.pop(d), None
                else:
                    yield d, None, "absent_upstream"
            self.drop(ctx, p)


ADAPTER = GLORYS025DAdapter
