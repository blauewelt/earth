"""ESA OC-CCI v6.0 ocean colour at DAILY resolution, 0.25° — family 7.2d (E-087).

PLAIN ENGLISH. The Ocean Colour Climate Change Initiative merges every
civilian ocean-colour satellite since SeaWiFS into one daily 4 km map of
chlorophyll-a. Family 7.2's `oc025` group is its five-day mean on the 0.25°
grid; this store keeps each DAY: `log_chl`, the mean of log10(chlorophyll)
over the clear 4 km cells of the 0.25° block (the very term family 7.2 adds
to its accumulator), and `chl_cov`, the fraction of the block's 4 km cells
that were clear. Both are NaN where no cell was clear that day (cloud, night,
ice, land). 1997-09-04 to 2024-12-31: CEDA's per-file archive to 2022, PML's
per-day NetcdfSubset of the aggregate for 2023-2024 — the files family 7.2
used. About 790 GB of producer transfer for ~7 GB of store (E-087 §6), so it
is built LAST (planning session, 2026-10-06).
"""
import datetime as dt
import os
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from family1.adapters import _f7d
import build_family7 as f7
import family7_daily as fd

CEDA = ("https://dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/v6.0-release/"
        "geographic/netcdf/chlor_a/daily/v6.0")
PML_DDS = "https://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-DAILY.dds"
PML_ASCII = ("https://www.oceancolour.org/thredds/dodsC/"
             "CCI_ALL-v6.0-DAILY.ascii?time[0:1:{last}]")
PML_FILE = ("https://www.oceancolour.org/thredds/ncss/grid/CCI_ALL-v6.0-DAILY"
            "?var=chlor_a&time={date}T00:00:00Z&accept=netcdf4")
CEDA_LAST_YEAR = 2022
NAME_RE = re.compile(r"ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-"
                     r"(\d{8})-fv6\.0\.nc")
WORKERS = int(os.environ.get("OC_WORKERS") or 12)


class OCCCI025DAdapter(_f7d.F7DailyBase):
    store = "occci025d"
    title = ("ESA OC-CCI v6.0 chlorophyll-a (log10) and clear-sky coverage, "
             "DAILY, 0.25-degree block means of the 4 km field — family 7.2d")
    first_year = 1997
    record_start = dt.date(1997, 9, 4)
    log2_fp = 0.0
    smoke_window = ("2010-01-16", "2010-02-08")
    smoke_probe_month = "2010-01"
    qc_policy = (
        "The product's only per-pixel flag is its fill value: fill, "
        "non-positive and sentinel values are not clear (they drop out of "
        "both sums and lower chl_cov). log10 is taken per 4 km cell and "
        "averaged over the block (never the log of a linear mean). NaN in "
        "both channels where no cell of the block was clear.")
    sources = (f"{CEDA}/<YYYY>/ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_"
               f"PML_OCx-<YYYYMMDD>-fv6.0.nc (1997-2022)",
               PML_FILE.replace("{date}", "<YYYY-MM-DD>") + " (2023-2024)")
    verified = ("2026-10-06: CEDA's 2015 listing and 30 days of 2015 fetched "
                "and reduced (E-087 probe: identical NaN pattern and within "
                "float16 rounding of f7l2's oc025 in every cell)")
    notes = ("Family 7.2d: family 7.2's oc025 one frame per day. The pentad "
             "rule is the mean of the days with a clear cell (>= 1, not 3) "
             "and the sum of chl_cov / 5.")

    def __init__(self):
        super().__init__()
        self._listing = {}
        self._axis = None

    # ------------------------------------------------------------ listing --
    def listing(self, ctx, year):
        """{date: url} out of the archive's own answer for one year."""
        if year in self._listing:
            return self._listing[year]
        out = {}
        if ctx.source_dir:
            d = os.path.join(ctx.source_dir, "occci")
            for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                m = NAME_RE.fullmatch(n)
                if m and m.group(1)[:4] == str(year):
                    day = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
                    out[day] = os.path.join(d, n)
        elif year <= CEDA_LAST_YEAR:
            html = f10_get(f"{CEDA}/{year}/").decode("utf-8", "replace")
            for ymd in sorted(set(NAME_RE.findall(html))):
                day = dt.datetime.strptime(ymd, "%Y%m%d").date()
                out[day] = f"{CEDA}/{year}/{f7.oc_canonical_name(day)}"
        else:
            if self._axis is None:
                n = f7.parse_dds_time_n(f10_get(PML_DDS))
                self._axis = f7.parse_opendap_ascii_ints(
                    f10_get(PML_ASCII.format(last=int(n) - 1)))
            for v in self._axis:
                day = f7.OC_TIME_EPOCH + dt.timedelta(days=int(v))
                if day.year == year:
                    out[day] = PML_FILE.format(date=day.isoformat())
        if not out and not ctx.source_dir:
            ctx.note_absent(str(year), f"OC-CCI listing for {year} is EMPTY")
        self._listing[year] = out
        return out

    # ------------------------------------------------------------- frames --
    def days_frames(self, ctx, days):
        jobs = []
        for d in days:
            url = self.listing(ctx, d.year).get(d)
            jobs.append((d, url))
        dest = os.path.join(getattr(ctx, "scratch", "/tmp"), "occci")
        os.makedirs(dest, exist_ok=True)
        lock = threading.Lock()

        def fetch(job):
            d, url = job
            if url is None:
                return d, None, "absent_upstream"
            if ctx.source_dir:
                return d, url, None
            p = os.path.join(dest, f7.oc_canonical_name(d))
            for i in range(4):
                try:
                    if f7.oc_generated_url(url):
                        f10_to_file(url, p, sized=False)
                    else:
                        f10_to_file(url, p)
                    return d, p, None
                except Exception as e:                       # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:200]}"
            return d, None, f"ABSENT:OC-CCI {d} listed at {url} and not " \
                            f"readable after 4 attempts ({err})"

        workers = 1 if ctx.source_dir else WORKERS
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fetch, j) for j in jobs[:2 * workers]]
            nxt = len(futs)
            for i in range(len(jobs)):
                d, p, why = futs[i].result()
                if nxt < len(jobs):
                    futs.append(ex.submit(fetch, jobs[nxt]))
                    nxt += 1
                if p is None:
                    yield d, None, why
                    continue
                try:
                    if ctx.source_dir:
                        ctx.count_bytes(os.path.getsize(p))
                    arr, _g = fd.occci_daily(p)
                except SystemExit:
                    raise
                except Exception as e:                       # noqa: BLE001
                    yield d, None, f"ABSENT:OC-CCI {d} {p} would not open " \
                                   f"({type(e).__name__}: {str(e)[:200]})"
                    continue
                finally:
                    if not ctx.source_dir:
                        with lock:
                            if os.path.exists(p):
                                os.remove(p)
                yield d, arr, None


def f10_get(url, attempts=4):
    import time
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=_f7d.f10b.UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
            _f7d.f10b.count_bytes(len(b))
            return b
        except Exception as e:                               # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise IOError(f"{url}: {last}")


def f10_to_file(url, path, sized=True):
    if sized:
        _f7d.f10b.http_to_file(url, path)
        return
    req = urllib.request.Request(url, headers=_f7d.f10b.UA)
    with urllib.request.urlopen(req, timeout=600) as r:
        b = r.read()
    if len(b) < f7.OC_MIN_GENERATED_BYTES or not b.startswith(
            f7.HDF5_SIGNATURE):
        raise IOError(f"{url}: {len(b)} bytes, not a netCDF4 file")
    _f7d.f10b.count_bytes(len(b))
    with open(path + ".part", "wb") as fh:
        fh.write(b)
    os.replace(path + ".part", path)


ADAPTER = OCCCI025DAdapter
