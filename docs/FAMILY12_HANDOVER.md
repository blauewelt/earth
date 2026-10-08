# Family 1.2 — handover for an agent picking up the data (2026-10-08)

Written for an agent who has not seen the sessions that built this. Every
code name is explained where it first appears.

## 1. What family 1.2 is

- **Family 1.gf** is the project's set of global *observation* stores at 10 km
  or finer and five days or finer, each kept at its own resolution: satellite
  maps (ocean colour, sea-surface temperature, cloud tops) and point records
  (ships, floats, bottles, moorings, altimeters, column CO₂).
- **Family 1.2** is family 1.gf **unchanged, inherited by reference** (no byte
  copied — its registry points at 1.gf's), **plus the atmosphere in three
  dimensions**: four stores from ERA5, the European weather centre's
  reanalysis (a weather model pulled towards every observation; not an
  observation itself).
- Everything is public on the Hugging Face dataset `chfrank/earth-tensors`.
  Family 1.2's own stores are under `tensors/family1_2/`, the inherited ones
  under `tensors/family1_gf/`.

## 2. The stores (measured from the live registries, 2026-10-08)

**Added by family 1.2 — ERA5 on 13 pressure levels** (50, 100, 150, 200, 250,
300, 400, 500, 600, 700, 850, 925, 1000 hPa; channel `t_500` is temperature
at 500 hPa):

| store | what | unit | grid | cadence | record |
|---|---|---|---|---|---|
| `era5_t` | air temperature | K | 1°, 181 × 360 | six-hourly instants (00, 06, 12, 18 UTC) | 1982-01-01 → 2026-06-30 |
| `era5_q` | specific humidity | g/kg (1000 × ERA5's kg/kg; small negatives are real and kept) | same | same | same |
| `era5_u` | eastward wind | m/s | same | same | same |
| `era5_v` | northward wind | m/s | same | same | same |

Each value is the area-weighted mean of ERA5's 0.25° field over the 1° box
around the grid point. Row 0 is −90°, column 0 is −180°. Licence CC BY 4.0
(Copernicus Climate Change Service) — the attribution must travel with any
file. Only final ERA5 is included; the preliminary stream (ERA5T, the most
recent ~3 months) is not.

**Inherited from family 1.gf:**

| store | what | kind | cadence | record |
|---|---|---|---|---|
| `oc4k` | ocean colour (3 channels), 4 km | map | daily | 1997-09-04 → 2022-12-31 (source closed) |
| `pace4k` | PACE satellite ocean colour (7 channels), 4 km | map | daily | 2024-03-05 → 2026-08-31 |
| `sst_acspo02` | NOAA ACSPO sea-surface temperature (2 channels), 2 km | map | daily | 2000-02-26 → 2026-10-07 |
| `irtb` | geostationary cloud-top brightness temperature, ±30° band | map | 3-hourly as served | 1998-01-02 → 2026-10-06 |
| `icoads` | ship and buoy surface reports | points | per report | 1662 → 2026-08-31 |
| `wod` | World Ocean Database profiles (128 channels) | points | per profile | 1772 → 2026-02-12 |
| `glodap` | bottle chemistry | points | per sample | 1972 → 2023-09-10 |
| `bgcargo` | biogeochemical Argo floats | points | per profile | 2002 → 2026-09-17 |
| `oceansites` | moorings | points | hourly | 1980 → 2026-09-16 |
| `xco2` | column CO₂ from satellites | points | per sounding | 2009 → 2026-07-31 |
| `swh` | significant wave height (altimeters) | points | per observation | 1991 → 2023-12-31 (source closed) |
| `swot` | SWOT sea-surface height | points | per observation | 2023-07-26 → 2026-09-15 |

`seaice_asi` is listed in the 1.gf registry but not built.

## 3. How the bytes are laid out

- **Time axis for everything:** five-day bins counted from 1982-01-01
  (`bin = floor(seconds since epoch / 432000)`); bins before 1982 are negative.
- **Map stores ("tier G", sharded):** one file per five-day bin holding
  independently zstd-compressed tiles (256 × 256 for the fine maps, 64 × 64
  for ERA5), float16 in physical units, with an index of (offset, length) per
  tile and frame; offset −1 means the frame is absent. ERA5 has 20 frames per
  bin. Reader and writer: `ml/family1/sharded.py`.
- **Point stores ("tier P"):** columns sorted by bin and time, with per-bin row
  offsets, so a period is a row range.
- **Registries:** `tensors/family1_2/family12.json` and
  `tensors/family1_gf/family1gf.json`. `record_span` is where the data really
  starts and ends; `requested_window` is what the build asked for.

## 4. Ways to read it

- **In a browser:** the Data tab on blauewelt.org (group "Atmosphere on
  pressure levels (family 1.2 — ERA5 reanalysis)" and the family 1.gf group):
  period, months, days, hours, box, resolution; NetCDF or CSV.
- **In Python:** `ml/family1/sharded.py` (`ShardedGroup.read_frame(bin, frame)`
  → `[H, W, C]` float32 with NaN for missing). `ml/export_gridded_monthly.py`
  has `open_store` / `fetch_bins`, which download bins with sha256 checks.
- **In JavaScript:** `src/f1data.js` (`window.F1Data`, also a node module) —
  the single reader the tab uses.

## 5. Precomputed averages

- **Monthly sums (E-088 — per-year, per-calendar-month sum, count and squared
  deviations):** published for the four ERA5 stores under
  `tensors/family1_2/<store>/monthly/`, indexed by
  `data/gridded_monthly_index.json`. Any multi-year mean or standard deviation
  of whole months is a handful of range reads.
- **Fine-grid monthly sums (E-089 — the same for the four fine 1.gf maps, in
  native tiles plus an exact 0.25° pooled layer):** `pace4k`, `oc4k`, `irtb`
  published and verified; `sst_acspo02` was being re-run on 2026-10-08. Not yet
  wired into the Data tab.
- **Day-of-year climatology (E-091 — sums cumulative over the years, one file
  per calendar day):** being built 2026-10-08, OISST first, then ERA5.

## 6. Keeping it current

`.github/workflows/data-refresh.yml` (E-090 — the daily refresh, 05:23 UTC, on
GitHub's hosted runners): asks each producer how far its record reaches,
re-fetches only the newest stretch, verifies, and only then updates the
registry. Scheduled for the ERA5 stores and for `pace4k`, `sst_acspo02`,
`irtb`. ERA5 has not gained data since 2026-06-30 (the source's final stream
ends there). Point stores are **not** yet automated (they need a rebuild into
a versioned folder); `icoads`, `swot` need a rented box.

## 7. Things that will bite

- ERA5 is a **reanalysis**; say so wherever its numbers appear.
- Reading one ERA5 level still reads all 13: the levels sit side by side in
  each tile.
- The last ERA5 bin is only two-fifths full (the record ends inside it).
- `sst_acspo02`: NOAA reissues each daily file about ten weeks later under the
  same name, so recent days can change.
- A build that reads a store while the daily refresh rewrites it fails its
  sha256 checks (safe, but costs a re-run): hold one while the other runs.
- The Hub throttles thousands of small range reads (HTTP 429): wait as long
  as its headers say, do not retry in a tight loop.
- Rented boxes never hold credentials; scheduled jobs never touch rented boxes.

## 8. Where to read more

- [The family 1.2 plan (E-085)](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E085_family12_atmosphere.md)
- [The Data tab guide](https://blauewelt.github.io/earth/docs.html?f=docs/DATA_TAB.md)
- [The daily refresh plan (E-090)](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E090_data_refresh.md)
- [Monthly sums (E-088)](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E088_monthly_sums_gridded.md)
- [Fine-grid monthly sums (E-089)](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E089_fine_grid_monthly_sums.md)
- [Day-of-year climatology (E-091)](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E091_day_of_year_climatology.md)
- [The family 1.2 registry on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/blob/main/tensors/family1_2/family12.json)
