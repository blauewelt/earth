# E-040 — daily SST since 1991, read two bytes at a time from Hugging Face

**Status:** design PROVEN end to end, bake not started. Written 2026-08-18.

> Chris: "Can this data be pulled from hugging face 'live' when a user clicks?
> If so (and this is not too complex to do) then let's do daily OISST since
> 1991. I also wonder how our daily data preparation that we're running now
> would be dependent on this data. Don't we need it to compute the proper
> embedding?"

Yes to the first, with one standing-rule decision attached. The second has a
sharper answer than it looks: we should share the **field** and must never
share the **climatology**.

---

## 1. The measurement that settles the design

Everything below is measured, not estimated.

| | |
|---|---|
| Hugging Face CORS | open — echoes our Origin, exposes `Accept-Ranges` / `Content-Range` |
| Ranged read from a browser on a foreign origin | **HTTP 206**, exact bytes returned |
| One value | **2 bytes**, ~740 ms through a proxy |
| One pixel-year (365 daily values) | **730 bytes**, ~630 ms, ONE request |
| Full 1991-2026 series for a pixel | ~26 KB, 36 parallel reads |
| Value returned vs source NetCDF | 24.22 °C vs 24.22 °C — bit-identical |

The trick is the LAYOUT. Store **pixel-major** — `value(px, day)` at byte
`(px*days + day)*2` — not day-major. The app needs one pixel's value, not one
day's map; the map is already GIBS tiles. Pixel-major makes a point query a
single contiguous read, so **the transfer is bounded by the question, not by
the archive**: 2 bytes out of a 757 MB file.

Day-major would have forced a whole-frame download per click and killed the
idea. This is the entire reason it is cheap.

## 2. Cost

| | |
|---|---|
| Source | 476 MB/year NetCDF (PSL), 36 years ≈ **17 GB** to pull |
| Container headroom | 18 GB — so stream year-by-year and delete (the baker does) |
| Bake time | one year in **< 1 min** measured; 36 years ≈ an hour plus upload |
| Output | 757 MB/year, 36 years ≈ **27 GB** on the Hub |
| Hub quota | PRO, 10 TB public — 0.3% of it |

`scripts/bake_sst_daily.py` does one year and is the slice already run and
verified. Remaining work is a loop, an upload, and the client read.

## 3. The standing-rule decision this needs

CLAUDE.md §3 restricts the browser to **GIBS + GBIF + the Open-Meteo family**.
`huggingface.co` would be a fourth. It clears the same bar the Open-Meteo
exception was granted on, and arguably more cleanly:

- no key, no account, no quota to manage,
- CORS-open (verified, not assumed),
- click-triggered and single-point — a 2-byte range read, never tile streaming,
- degrades to an omitted row: if it fails we still have the monthly value.

**Recommendation: add it, with those four conditions written into §3 as they
were for Open-Meteo, and with the monthly path kept as the fallback** so a
Hub outage costs precision rather than the feature.

## 4. What it buys

- **Uncapped anomalies at 0.25°, daily.** The current correction is 1° monthly
  because that is the only SST we ship as NUMBERS. Off Peru a 1° cell averages
  the coastal upwelling tongue away — measured: MUR 18.8 °C vs OISST-1° 20.9 °C
  at the same point and date, a 2.1 °C gap, in exactly the region this gets
  used for.
- **A time series per click, not a value.** 730 bytes gets the whole year;
  26 KB gets 1991-2026. The pixel card could draw a daily SST history with the
  climatology behind it, which is a better answer to "is this unusual" than any
  single number.
- **A path off the palette entirely** for any layer we later bake this way.

## 5. The ML question: share the field, never the climatology

Chris asked whether the daily pipeline depends on this. Two halves, opposite
answers, and the second is a correctness trap.

**The field: this is not an ADDITIONAL import — it is the import E-034
already owes.** Resolved with Chris 2026-08-18. The pentad plan's cadence
table already declares `OISST SST: daily → pentad mean`, but nothing in the
repo fetches daily OISST yet: the only script touching `sst.day.mean` is
`scripts/bake_sst_daily.py`, written for this plan. So E-040 and E-034 have
the SAME upstream download, and it should be pulled once.

Two consumers, two artifacts, shared at the DOWNLOAD level — not the artifact
level, because the layouts are transposes of each other:

    sst.day.mean.YYYY.nc  (PSL, 476 MB/yr, cached then deleted)
        ├── day-major pentad means, NA window → the E-034 tensor channel
        └── pixel-major int16 → HF → the app's 2-byte point reads

Building the tensor from the pixel-major file would mean reading all 757 MB
back and transposing; building the app file from pentad means would throw away
the daily axis. Each consumer takes its natural cut of one download.

Worth saying plainly: this is also a QUALITY fix for training, not just
plumbing. The current tensor's SST arrives via the 1° monthly bake, upsampled
to 0.25° — carrying no sub-degree structure, exactly like the RG channels the
build notes already flag. Daily 0.25° OISST is native-resolution signal on the
tensor's own grid. (OISST's effective resolution is coarser than its 0.25°
posting — it is an AVHRR analysis, not MUR — but it is real sub-degree
information against a 1° bin.)

**The climatology: must stay separate.** The ML pipeline computes its anomaly
baseline INSIDE the pipeline, from **train years only**:

    clim[m] = np.nanmean(X[(moy == m) & ~t_hold], axis=0)     # temporal.py

The app's baseline is the WMO 1991-2020 normal, which **includes the holdout
years**. Substituting it into the pipeline would leak test-period information
into training through the baseline — the model would be centred using data it
is meant to be scored against. The two baselines look interchangeable and are
not. Keep them apart, and keep this paragraph next to any future refactor that
notices "we compute the climatology twice".

## 6. Order of work

1. Loop the baker over 1991-2026, uploading each year (script exists, one year
   verified).
2. Client: `sstDailyAt(lon, lat, date)` — one range read, cached per pixel-year,
   falling back to the monthly value on any failure.
3. Point the pixel card and the probe at it, keeping the monthly path as the
   fallback and the ±3 palette bound as the stated colour meaning.
4. §3 amendment + a test that the daily read degrades to monthly when the Hub
   is unreachable.
5. The E-034 pentad builder consumes the same cached NetCDFs (its natural,
   day-major cut) — resolved above; the leakage rule travels with it: the
   pipeline's climatology stays train-years-only, computed inside temporal.py,
   never the app's 1991-2020 normal.
