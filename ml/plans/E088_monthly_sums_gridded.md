# E-088 · Long-period averages made cheap: per-year monthly sums of the gridded stores

Written 2026-10-07 (Fable plans, Opus implements — `ml/CLAUDE.md` §0b). The
sibling of `ml/plans/E086_monthly_by_year.md`, which did the same for the
five-day tensor.

Chris, 2026-10-07: *"please go ahead with the 'make long period averages
cheap' plan"*.

## 1. In plain English

The Data tab (`ml/plans/E084_data_tab.md` — the page that filters and
downloads every public store in the browser) can already average any period
of any gridded store, but it does it by reading every native frame: a
thirty-year July normal of the daily sea-surface temperature is ~930 daily
frames, and of the six-hourly ERA5 temperature ~3,700. That is hundreds of
megabytes and minutes for a question whose answer is one map.

An average is a sum divided by a count, and sums and counts add up across
months and years. So, once, on a rented box, this experiment computes for
every **calendar month of every year**, every channel and every cell: the
**sum** of that month's frames, how many of them had a value (the
**count**), and the **sum of squared deviations from that month's own mean**
(**m2**). Then the browser composes, from a handful of range reads:

- the mean over ANY set of months and years = Σ sum / Σ count;
- the standard deviation over the same set, by Chan's combination of the
  per-month pieces (§3) — the one thing a set of means alone cannot give.

The stores (phase 1, all in the same sharded tile layout,
`ml/family1/sharded.py`):

| store | what it is | grid | channels | frames | record |
|---|---|---|---|---|---|
| `family7_2d/glorys025d` | GLORYS12 ocean reanalysis: surface current speed and direction, sea-surface height, mixed-layer depth (log₁₀), daily | 0.25°, 721 × 1440 | 5 | 12,283 days | 1993-01-01 → 2026-08-18 |
| `family7_2d/oisst025d` | NOAA OISST v2.1 sea-surface temperature and sea-ice concentration, daily | 0.25° | 2 | 16,348 days | 1982-01-01 → 2026-10-04 |
| `family7_2d/ncep100d` | NCEP/NCAR Reanalysis 1 surface and land channels (wind stress, 2 m air temperature, wind, pressure, rain, snow, soil, heat fluxes), daily | 1°, 181 × 360 | 15 | 16,147 days | 1982-01-01 → 2026-03-17 |
| `family7_2d/occci025d` | ESA OC-CCI chlorophyll-a (log₁₀) and its clear-sky coverage, daily | 0.25° | 2 | 10,501 days | 1997-09-04 → 2026-06-30 |
| `family1_2/era5_t`, `_q`, `_u`, `_v` | ECMWF ERA5 reanalysis temperature, specific humidity (g/kg), eastward and northward wind on 13 pressure levels (50…1000 hPa, folded into channels `t_500` etc.), six-hourly instants | 1° | 13 each | 65,008 each | 1982-01-01 → 2026-06-30 |

Family 7.2d is the global five-day tensor's channels kept at one frame per
day (`ml/plans/E087_family7_daily.md`); family 1.2 is the fine observation
stores plus ERA5 on pressure levels (`ml/plans/E085_family12_atmosphere.md`).

## 2. Decisions

1. **The month is the CALENDAR month (UTC) of each frame's own start
   instant** (`sharded.frame_datetime`). This is deliberately NOT E-086's
   rule, under which a five-day bin belongs to the month it opens in — that
   rule exists only to reproduce the model's climatology. A consequence worth
   stating: a July normal of the daily SST store here and the five-day
   tensor's July normal (E-086) differ slightly at month edges, because a
   bin opening on 30 June is June to the tensor and five-sixths July here.
2. **Physical units as stored** (°C, K, m/s, g/kg, …); the stores are not
   z-scored, so no `norm` is involved.
3. **Same axis order and offset formula as E-086**: `[month, channel, year,
   lat, lon]`, C order, so every February of a period is one contiguous range
   and the website's E-086 reader works unchanged. Plane (m, c, y):

       offset(m, c, y) = header_len + ((m·C + c)·Y + (y − year_first))·H·W·itemsize

   itemsize 4 for `sum.npy` and `m2.npy`, 1 for `count.npy`. Each store has
   its own year axis (its first to its last year with a frame); the
   incomplete final year simply has smaller (or zero) counts.
4. **sum: float64 accumulation, stored float32.** A float16 value is a
   multiple of 2⁻²⁴ below 2¹⁶, so the float64 sum of up to 124 of them is
   EXACT (≤ 47 significant bits); the only rounding is the single float32
   cast. The rigorous bound per cell is |Σsum/Σcount − mean| ≤ Σ ½ulp₃₂(sum)/Σcount
   + 4·eps₆₄·|mean| — attained at rounding ties, never exceeded. float32
   is measured adequate (§5); float64 would double 25 GB for digits nobody
   reads.
5. **count: uint8, asserted** ≤ 31 daily or ≤ 124 six-hourly frames (and
   ≤ 255). It counts FINITE frames per cell; NaN (land, cloud, ice) does not
   count. sum and m2 are 0.0 where the count is 0.
6. **m2, not a sum of squares — shipped.** A plain Σx² of 300 K temperatures
   loses everything to cancellation in float32 (and a fixed per-channel
   offset does not save it: the mean varies by 100 K across the globe). The
   per-(year, month) **centred** m2 = Σ (x − mean_ym)², two-pass in float64,
   needs no offset at all, is independent per year (so years are computed in
   parallel), and combines exactly by Chan et al.:

       var(K) = (Σ_k m2_k + Σ_k n_k (sum_k / n_k − mean_K)²) / Σ_k n_k

   (population, ddof 0). Error analysis: the float32 roundings of sum_k and
   m2_k give |Δstd| ≲ 2⁻²⁴(|mean| + std); the falsifier requires
   |Δstd| ≤ 10⁻⁶ (|mean| + std) — a ~4× margin over that — against numpy's
   nanstd of the native frames. It costs 25.4 GB on the Hub and one more
   plane read per (month, channel) when a std is asked for.
7. **Frames present and frames possible per (year, month)** in stats.json
   and the index: present = frames in the store's shards; possible =
   epoch-aligned frame instants of that month inside the store's measured
   record (first to last present frame). So the page can say "count 28 of
   31 days" and tell a gap upstream from a cloudy pixel.
8. **Streaming, one store at a time, published per store** (`ml/CLAUDE.md`
   §5.20, §5.26). Each worker takes one calendar year: it downloads that
   year's bins (each file **sha256-checked against the store's own
   store.json**), decodes every frame with `sharded.ShardedGroup.read_frame`
   (the reader, not a second decoder), writes its year's planes into the
   output memmaps and frees its disk. When a store is done it is verified
   and committed to the Hub at once with its manifest; a store whose
   `monthly/stats.json` on the Hub names the same source sha256 block is
   skipped, so a lost box redoes at most one store.
9. **Restore from a hosted runner** (E-083's lesson, as in E-086): the box
   uploads; a GitHub-hosted runner streams every file back, hashing in
   order, and only a full match writes the index (`restore_verified: true`).

## 3. Files

Per store, under `tensors/<store>/monthly/` on `chfrank/earth-tensors`:
`sum.npy`, `count.npy`, `m2.npy` (`[12, C, Y, H, W]`), `stats.json`,
`verify.json` (the falsifier's report), `manifest.json` (every file's
sha256, bytes and parsed header).

One committed index, `data/gridded_monthly_index.json`
(`ml/publish_gridded_monthly_index.py`, never hand-edited), same key style as
`data/family7_monthly_index.json` — per store and per file `url`, `bytes`,
`sha256`, `header_len`, `shape`, `dtype`, `itemsize`, `axes`, `plane_bytes`,
`month_channel_bytes`; per store the year axis, `frames_present` and
`frames_possible` tables, channels (units, ranges) and `levels_hpa`, the grid
(pixel positions `lat0 + row·dlat`, `lon0 + col·dlon`, and the store's own
registration sentence — family 7.2d and ERA5 are POINT-registered:
coordinate = the pixel's centre), licence and attribution, and the
falsifier's numbers; at the top the measured CORS (`https://blauewelt.org`,
206), the formulas and the tolerances. Fixture `data/gridded_monthly/fixture/`
(two tiny stores built with the real tier-G framework, < 1 MB).

## 4. Storage

| store | sum | count | m2 | total | one (month, channel), all years, all three files |
|---|---|---|---|---|---|
| `glorys025d` (5 ch, 34 y) | 8.47 GB | 2.12 GB | 8.47 GB | **19.06 GB** | 318 MB |
| `oisst025d` (2 ch, 45 y) | 4.49 | 1.12 | 4.49 | **10.09** | 421 MB |
| `ncep100d` (15 ch, 45 y) | 2.11 | 0.53 | 2.11 | **4.75** | 26 MB |
| `occci025d` (2 ch, 30 y) | 2.99 | 0.75 | 2.99 | **6.73** | 280 MB |
| `era5_t/q/u/v` (13 ch, 45 y) | 1.83 each | 0.46 | 1.83 | **4.12 each** | 26 MB |
| total | | | | **57.1 GB** | |

At 0.25° a 30-year mean of one channel-month reads 125 MB of sums plus 31 MB
of counts (and 125 MB more for a std) for the whole globe, in one range each;
a box is a row band per year (the E-086 reader's per-selection rule decides
between band-by-year and one stretch). At 1° it is under 10 MB. Against the
native path — ~930 daily 0.25° frames for a 30-year July — this is the cheap
version Chris asked for.

The pull: 122 GB (family 7.2d) + 349 GB (ERA5) ≈ 471 GB, on one verified
box chosen by transfer price as well as speed.

## 5. Falsifiers

- **In the job, refusing the upload** (`export_gridded_monthly.py verify`):
  for every store, **sampled planes** — the first and last month with data
  (both partial for most stores) and one month per decade the record touches,
  every channel — must equal numpy's nanmean / nanstd of the native frames
  read back through `sharded.ShardedGroup` from the kept, sha256-checked
  bins: counts EXACTLY, means within the rigorous bound of §2.4, std within
  10⁻⁶ (|mean| + std); and a **period box** — the whole of 2010–2012, a
  4° × 4° box at 38–42° N, 42–38° W — composed from its 36 per-month planes
  must equal the mean and std of ALL native frames in those three years,
  read tile by tile.
- **In the suite** (`tests/test_export_gridded_monthly.py`, 18 checks): on
  two tiny stores built with the real framework (daily across two year
  boundaries with land, clouds and a day absent upstream; six-hourly across
  month boundaries inside bins) EVERY plane equals numpy; four composed sets
  (all, every January, a scattered set, one year) equal numpy over the union;
  a bin straddling a month boundary is split by calendar month; frames
  present / possible; `verify` passes and FAILS on a nudged sum or m2; a
  tampered shard is refused before anything is summed; the publisher's
  manifest, commit, Hub restore, refusal of a byte that does not come back
  and of a failed verify report; the box loop end to end incl. the skip of a
  published store; the committed fixture.
- **Live, after publish, from the sandbox** (independent reads over HTTP):
  one plane of each store against numpy over its native frames on the Hub;
  ERA5 `t_500` January 2015 against its 124 native frames; a 1991–2020 July
  SST normal of `oisst025d` for a small box against the mean of the native
  July frames of those years.

Measured before dispatch on the real `ncep100d` 2015 (sandbox, 564 MB): 15
planes, max |Δmean| 7.9 × 10⁻⁶ (inside the bound, which is met exactly at a
float32 rounding tie), max |Δstd| 5.1 × 10⁻⁶ = 0.03 of the tolerance.

## 6. Phase 2 — the fine 1.gf grids (probe only, no build)

`oc4k` (ocean colour, 4 km daily), `pace4k` (PACE, 4 km daily),
`sst_acspo02` (ACSPO SST, 2 km daily), `irtb` (geostationary cloud-top
brightness temperature, 3-hourly, ±30°). A dense monthly plane is 37–162 M
cells, and monthly coverage is still partial (clouds, night, swaths). The
probe (`export_gridded_monthly.py probe`, in the same box job) streams one
month of each, accumulates per-cell sums and counts, and measures the union
coverage and the zstd size of the month's sum (float32) and count (uint8)
stored as the store's own 256 × 256 tiles. §8 holds the table and the
recommendation.

## 7. What exists

- `ml/export_gridded_monthly.py` — `export` / `verify` / `run` / `probe` /
  `fixture`.
- `ml/publish_gridded_monthly_index.py` — `upload_store` (box) and `index`
  (hosted restore, or `--local`).
- `.github/workflows/gridded-monthly.yml` — `workflow_dispatch` only; job
  `export` on the rented box (per store: export → verify → publish → free;
  then the probe), job `restore` on `ubuntu-latest`.
- `tests/make_gridded_monthly_fixture.py`, `tests/test_export_gridded_monthly.py`.

## 8. Status

- 2026-10-07: plan, exporter, publisher, workflow, 18 toy checks, the fixture
  and a real one-year trial on `ncep100d` written and green.
