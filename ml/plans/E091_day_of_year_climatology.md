# E-091 — Day-of-year climatology: the mean of every calendar day over any span of years

**What this is for.** Chris, 2026-10-08: *"Say I want average values for every
single day of year, across 20 years."* The Data tab could give a climatology
per calendar month (12 maps) but not per calendar day (366 maps): twenty years
of daily maps is about 7,300 maps per channel, over the tab's read limit for
anything but a small box. This experiment publishes, per gridded store, sums
that make any span of years cost two small reads per day.

## 1. What is published

Per store, under `tensors/<store>/doy/v1/` on the project's Hugging Face
dataset, for each of the 366 calendar days (29 February its own):

- `sum_MMDD.npy` — `[channel, year, lat, lon]` float32, **cumulative over the
  years**: plane `(c, y)` is the sum of every frame of that calendar day from
  the record's first year through year `y`, in the store's physical units.
- `count_MMDD.npy` — the same shape, uint8: how many frames that sum holds.
- `manifest_MM.json` — one per calendar month: sizes, sha256, headers, the
  years, and the falsifier's numbers.

```
mean of a day over years a…b = (S[b] − S[a−1]) / (N[b] − N[a−1])
leaving year e out           : subtract (S[e] − S[e−1]) and (N[e] − N[e−1])
```

The day of a frame is the UTC calendar day of its start instant. A six-hourly
store (ERA5) puts four frames in each day, so its climatology is of daily
means. A year with no frame on a day (29 February outside leap years) repeats
the plane before it.

## 2. Why cumulative

Per-year planes would cost one read per year and day (20 years × 366 days =
7,320 reads); cumulative planes cost two per day whatever the span. The price
is float32 rounding of a running total, which the falsifier measures.

## 3. The falsifier

In every month job, before the upload: for days 1, 15 and 29 of the month,
every native frame is kept, and the composed mean over six spans of years (the
whole record, a 20-year span, single years at both ends and the middle, all but
the end years) is compared with numpy's mean of those frames. Counts must be
equal exactly; the mean must lie within one float32 unit-in-the-last-place of
the running total, divided by the count. A failure uploads nothing.

## 4. How it is built

`ml/export_gridded_doy.py month` on GitHub-hosted runners, twelve jobs per
store (`.github/workflows/gridded-doy.yml`, manual dispatch only). Each job
needs one month of source and one month of output on disk. `index` then reads
the twelve manifests and writes `data/gridded_doy_index.json`, which the Data
tab reads.

## 5. Sizes (dense, float32 sum + uint8 count)

| store | grid | channels | years | size |
|---|---|---|---|---|
| OISST sea-surface temperature, daily | 0.25° | 1–2 | 45 | ≈ 85 GB per channel |
| ERA5 temperature / humidity / winds (four stores) | 1° | 13 levels each | 45 | ≈ 70 GB each |
| NCEP air reanalysis, daily | 1° | see registry | 45 | ≈ 5 GB per channel |
| GLORYS ocean reanalysis, OC-CCI ocean colour | 0.25° | several | 30–34 | ≈ 60 GB per channel — decided after OISST |

## 6. Status

- 2026-10-08: exporter, workflow and plan written; fixture run green; OISST
  dispatched first.
