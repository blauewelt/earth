# E-081 · Family 10.2 — the fishing fleet as an observation store, and its two globe layers

*Written 2026-09-16. Chris: "add what you propose to family 10.2 (and build the
family). At the same time make sure to display the data on blauewelt.org. I'm
also curious where the loitering ships are — let's add a layer." This is the
build spec for the fifth tier-P store of family 10 and for the two globe
layers that show the fleet, written before anything is fetched.*

**What family 10.2 is.** Family 10.1 (E-079 §10.1) is the four observation
stores — surface drifters, the tropical moored arrays, ship CO₂ tracks and
altimeter along-track sea level — with time in integer seconds. Family 10.2
adds one store, `fishing`, and changes nothing else: the four 10.1 stores are
inherited by reference exactly as tier G inherits family 7.2, and the 10.2
registry lists five tier-P groups (plus Argo) with each group's own Hub path.
No 10.1 byte is rebuilt.

## 1 · The source

Global Fishing Watch, *Global AIS-based Apparent Fishing Effort Dataset*,
version 3.0 (2025-03-11), Zenodo record 14982712, licence **CC BY-NC 4.0**
(non-commercial, attribution required; redistribution permitted with
attribution — the store's `store.json` and the data handover carry the
recommended citation and the licence, the globe layer carries "Powered by
Global Fishing Watch"). No credentials: Zenodo serves the files anonymously.

The table this store is built from is **`mmsi-daily-csvs-10-v3-<year>.zip`,
2012–2024** — one row per (day, 0.1° cell, vessel), columns `date`,
`cell_ll_lat`, `cell_ll_lon` (the cell's lower-left corner), `mmsi`, `hours`
(hours the vessel was broadcasting inside the cell that day) and
`fishing_hours` (the part of those hours a neural network classed as
fishing). 13 zips, 5.4 GB in total (0.06 GB for 2012, 0.75 GB for 2024).
Beside it, `fishing-vessels-v3.csv` (0.11 GB) gives each MMSI its year, flag
(`flag_gfw`), gear class (`vessel_class_gfw`), length, engine power and
tonnage. The 0.01° daily table by flag and gear (21 GB) is NOT taken: it has
no vessel identity, and 0.1° is already four times finer than the family-7
grid.

Known issues the source states (verbatim in `README-known-issues-v3.txt`):
2024 is provisional; AIS reception is uneven in space and time, so absence of
effort is not absence of fishing; MMSI is not always one vessel (spoofing,
reflagging, recycling); one gear class per MMSI over the whole record.

## 2 · The store

`tensors/family10_2/fishing/`, the tier-P schema of `ml/family10_store.py`
(schema version 2, nine columns, CSR `bin_offsets`, `[-180, 180)` longitude),
built by a `FishingAdapter` in `ml/build_family10_stores.py` next to the four
existing adapters. Row = one (day, cell, vessel):

| column | value |
|---|---|
| `time_s` | the day at 00:00:00 UTC in seconds since 1982-01-01T00:00Z (int32). The source is daily; every row of a day carries the same second and the pentad bin is `floor_divide(time_s, 432000)` as everywhere else |
| `lat`, `lon` | the cell **centre**: lower-left corner + 0.05°, longitude wrapped to `[-180, 180)` after the float32 cast (the fb5d5ab rule) |
| `values` (C = 2) | `fishing_hours`, `hours` — float16. The builder asserts `0 ≤ fishing_hours ≤ hours` and refuses otherwise. **Measured on the 2012 zip before the build: `hours` exceeds 24 in 1.6 % of rows (to 47.6 h; 2024 reaches 48.0 h)** — the source's own first known issue, one MMSI broadcast by more than one vessel — so the ceiling is not 24 but a `--max-hours` sanity bound (default 168 h; a misread longitude still trips it), and rows over 24 h are counted in `store.json` (`hours_over_24h`, `max_hours`) rather than refused |
| `platform` | `platform_hash(mmsi)` — the same hash the other stores use, so a vessel is one id across days |
| `qc` | the vessel's **gear class** as a small integer from a code table written into `store.json` (`qc_codes`: e.g. 1 trawlers, 2 drifting longlines, 3 purse seines, 4 squid jiggers, … 0 unknown) — a categorical channel a consumer may use or ignore |
| `fp` | footprint: `log2_fp = log2(11.1 km / 27.83 km)` for a 0.1° cell, `log2_dt = log2(1 day / 5 days)` — the same convention as E-079 §2 |

`store.json` additionally carries `source` (the Zenodo DOI, the version, the
citation), `licence: "CC BY-NC 4.0"`, `attribution`, `qc_codes`, the
per-year row counts, `hours_total` and `fishing_hours_total` (float64 sums
over the whole store), and the vessel table's summary (vessels per year, gear
class histogram). The vessel table itself is published beside the store as
`vessels.csv.gz` — small, and the only way a consumer can turn a platform
hash back into a flag or a length.

**Build path.** A rented box, not a hosted runner (measured: 0.1116 rows per byte of zip → ≈ 596 M rows, ≈ 18.5 GB of store plus ≈ 16 GB of parts = ≈ 35 GB while building; a hosted runner has ~14 GB). `family10-build.yml`, `store: fishing`, no credentials: each year's zip is streamed through `zipfile` without
extraction, parsed in chunks, packed per year into parts under the existing
`parts/<year>/` layout, then assembled and published exactly like `gdp`
and `socat` (streaming assembly; the store is about a quarter of `slatrack`). Every year is marked only when its zip's row count equals
the rows read (the audit rules of 7e6b14f apply unchanged: a short download
or a year with zero rows is an absence, never a gap).

**The registry.** `FAMILY_VERSION = "10.2"` in `ml/family10_store.py`;
`HF_ROOT`/`HF_PARTIALS` follow it (`tensors/family10_2`, `partials/family10_2`).
`ml/build_family10_registry.py` gains a per-store root: the four 10.1 stores
are read from `tensors/family10_1/<store>/store.json` and listed with that
`path`; `fishing` from `tensors/family10_2/fishing/`. The registry is written
to `tensors/family10_2/family10.json` with `family_version: "10.2"`,
`inherits: {"10.1": [...]}` and, per group, its `path` and `schema_version`.
The store-side check that refuses a 10.0-schema part stays.

## 3 · The gridded product for the globe

The globe cannot range-read a point store. Beside the store the builder
writes **`tensors/family10_2/fishing_grid/fishing_grid_monthly_025.npy`**:
shape `[156, 721, 1440, 2]` **float32** (planned as float16 — measured before the build: the largest 0.25° cell-month sum is 595,726 vessel-hours in 2024 against float16's 65,504, so the fleet's own cells would overflow; months 2012-01 … 2024-12; the family-7
0.25° grid, latitude −90 → +90, longitude −180 → +180; channels
`fishing_hours`, `hours`), each cell the SUM of the store's rows whose cell
centre falls in it that month, month-major in C order so one month of one
channel pair is one contiguous range read (721 × 1440 × 2 × 4 = 8.3 MB, still under family 7's 14.5 MB).
Zero is a real value (no broadcasting fishing vessel) and is stored as 0, not
NaN. `ml/publish_family7_index.py`'s pattern writes
`data/fishing_index.json` — header length, shape, dtype, the month list, the
grid geometry, the channel labels and units, the measured CORS headers — so
`src/app.js` contains none of the arithmetic (root `CLAUDE.md` §3, the
family-7 rule). The grid's column sums must equal the store's totals to
float16 accumulation error, and the builder asserts it.

## 4 · The two globe layers

**(a) "Fishing effort (AIS), 0.25° monthly" — timed raster from the Hub.**
Reads one month by range from the grid above through the same slab reader
and LRU the family-7 layer uses (E-070), paints `log10(1 + fishing_hours)`
on a sequential ramp with zero transparent, snaps the date to the month
(`monthlyGrid` semantics, §4b), and the probe/pixel card print the two
sums in hours with the month stamp. Complete per root `CLAUDE.md` §2: title
with the pixel size, hover card (gist: what AIS is, what "apparent fishing"
means, the coverage caveat), legend, probe, aggregation posture `✗/✗` with
the reason (a monthly sum already; and a range read per month — the same
byte-count argument as the family-7 row), catalog record with the CC BY-NC
licence, chip, tests against a git-committed decimated fixture, provenance
stamp = the month.

**(b) "Loitering vessels (last 30 days)" — points from a baked snapshot.**
Loitering events (a vessel drifting at sea for hours at low speed, the
signature of transshipment at sea and of waiting) come only from the
Global Fishing Watch Events API, which needs a token — so they are NEVER
fetched by the browser (§3: no keyed host). A scheduled hosted workflow
`.github/workflows/refresh-loitering.yml` (daily, `ubuntu-latest`, secret
`GFW_API_TOKEN`) pages `GET /v3/events?datasets[0]=public-global-loitering-events:latest`
for the last 30 days, and bakes `data/loitering.json`: per event the
position, start, end, duration in hours, median speed, distance from shore,
vessel name, flag and MMSI, plus `fetched_at`, the window, the count and the
attribution string. Until the secret exists the workflow refuses with a clear
message and the layer shows the committed fixture. The layer draws one point
per event (colour by duration), shows events whose interval overlaps the
selected date and toasts when the date is outside the snapshot's window
(§4b — date-driven, not dateless), hover card, catalog record (CC BY-NC,
"Powered by Global Fishing Watch"), chip, tests. The footer carries the
attribution line the licence requires.

## 5 · Hypothesis, falsifier, cost

*Hypothesis.* The fleet is a distributed sensor of the ocean: apparent
fishing effort follows fronts, upwelling and productivity, so it is both a
read-out target for "predict everything" and a weak extra input. This plan
builds the store; the test of the hypothesis is E-078a's ablation with
`fishing` tokens in and out, not this build.

*Falsifier for the build.* (1) For every year, `N_year` equals the number of
data rows in that year's zip; (2) `fishing_hours_total` is RECORDED, not gated — the 2012 zip alone holds 7.7 M fishing hours and the whole record projects to ≈ 0.8 billion, so the "nearly 370 million hours" in Global Fishing Watch's release note is not this table's sum (it is presumably the fleet table's or a different accounting) and cannot serve as a band; the RESULT states both numbers; (3) every
`fishing_hours ≤ hours`; (4) the monthly grid's global sums equal the store's
per-month sums; (5) the registry lists nine tier-P/G groups plus `fishing`
and every 10.1 sha256 is unchanged.

*Cost.* One verified box (`ml/CLAUDE.md` §7: read `inet_up` first): ~20 min of parsing plus 5.3 GB down and ~20 GB up; probe of 2012 on a hosted runner first, free. Globe work: one
frontend session. Expected store size ≈ 596 million rows (projected from the 2012 and 2024 zips; the RESULT records the measurement).

## 6 · Status

DISPATCHED 2026-09-16 — builder and globe layers in implementation; run
numbers and the falsifier's numbers to follow in the RESULT.
