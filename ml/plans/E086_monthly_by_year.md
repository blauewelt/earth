# E-086 · A climatology over any period: per-year monthly sums and counts of family 7

Written 2026-10-06 (Fable plans, Opus implements — `ml/CLAUDE.md` §0b). The
successor of `ml/plans/E083_model_climatology.md`.

Chris, on the three fixed "holdout versions" of the model climatology:
*"just setting a period is enough"* — and, 2026-10-06: *"Please build the
per-year monthly files."*

## 1. In plain English

The **model climatology** is what the forecaster calls *normal*: for each
calendar month, each map cell and each channel, the average of the stored
value over the training years' five-day bins that fall in that month. It is
computed by one function, `ml/trainprobe.py::anomaly_transform` (the one place
in the code that turns a channel into a departure from its own normal before
the model sees it). E-083 published that field for **family 7** (the global
input tensor: every 0.25° point from pole to pole, 1982–2024, one frame per
five days) in three fixed versions — all years, the 2009/2017/2023
development holdout, and the paper's 1982–2020 split.

An average is a **sum divided by a count**, and sums and counts add up across
years. So instead of publishing one climatology per choice of years, this
experiment publishes, once, for every year, every calendar month, every
channel and every cell: the **sum** of that year's bins in that month and how
many of them had a value (the **count**). Then the climatology over ANY set of
years — "1991 to 2020", "everything except 2023", one single year — is

    mean = (sum of the chosen years' sums) / (sum of the chosen years' counts)

which is exactly the function's definition, computed in the browser with no
precomputed version at all. That is what lets the page offer a free period
instead of three buttons.

**Family 7.2** is the published state of family 7 (stem
`family7_global025_pentad_l2` on the Hub): four channel groups, each on its own
grid and its own time axis —

| group | what it is | grid | channels | rows | years on its own axis |
|---|---|---|---|---|---|
| `g025` | the ocean at 0.25°: current speed and direction, mixed-layer depth, sea-surface height, sea-surface temperature (OISST), sea ice | 721 × 1440 | 7 | 3,142 five-day bins | 1982–2024 (43) |
| `g100` | the reanalysis surface at 1°: wind stress, 2 m air and skin temperature, wind, pressure, rain, snow, soil, heat fluxes | 181 × 360 | 15 | 3,142 bins | 1982–2024 (43) |
| `oc025` | ocean colour (chlorophyll and its clear-sky coverage) at 0.25° | 721 × 1440 | 2 | 1,997 bins from 1997-09-04 | 1997–2024 (28) |
| `rg100` | Argo temperature and salinity at 16 depths, 1° | 181 × 360 | 32 | 252 monthly rows from 2004-01 | 2004–2024 (21) |

## 2. Decisions

1. **One sum and one count per (year, calendar month, channel, cell), per
   group, nothing else.** The sum is in **z-units** — the tensor's stored
   units, exactly like E-083's `clim.npy` — so the page converts a composed
   mean with the same `norm` (mean, sd) it already uses everywhere:
   physical = z × sd + mean.
2. **The month of a bin is the month its five-day window OPENS in**, from the
   npz's own `months` key, parsed by `ml/export_family7_clim.py::master_calendar`
   and charged to each group's OWN rows by `ml/cone_sampler.py::group_time` —
   the same two calls the E-083 exporter, the trainer and the cone export
   make. No second calendar exists. Two consequences worth knowing: the master
   axis's first row opens on 1982-01-01 (no December-1981 row), and a group
   whose first bin opens in December (on the toy, the pentad holding
   2015-01-01 opens on 2014-12-29) starts its year axis in that December's
   year — the rule, not an off-by-one. `rg100` is summed on its monthly rows
   (one per month, count ≤ 1); `oc025` from its own first bin.
3. **Each group has its own year axis** (`year_first`…`year_last`, contiguous):
   no group stores a block of zeros for years before its record starts — that
   would be 1.5 GB of zeros for `oc025` and 2.1 GB for `rg100`. The index
   carries the axis; the page indexes `y − year_first`.
4. **Axis order `[month, channel, year, lat, lon]`, C order** — *not* year-major.
   The two reads the browser makes:
   - "one month, one channel, a run of years" (every February from 1991 to
     2020 — the period climatology): in this order it is ONE contiguous range
     per file; year-major (`[year, month, channel, …]`) would make it one
     request per year, 30 instead of 1;
   - "one year-month" (2015-02, one channel): one plane in either order.
   The only read year-major makes cheaper is "one year-month, every channel"
   (one range instead of C). The page shows one channel at a time, so that is
   not a common read, and C ≤ 32 small requests is still cheap. A second file
   in the other order would double 27.4 GB for that rare case: not built.
   Byte offset of plane (m, c, y), for both files (itemsize 4 for the sum, 1
   for the count; m = 0 is January):

       offset(m, c, y) = header_len + ((m·C + c)·Y + (y − year_first))·H·W·itemsize

   and years y_a…y_b of one (m, c) are the single range
   `[offset(m, c, y_a), offset(m, c, y_b) + H·W·itemsize)`.
5. **The sum is float32, not float16, and not float64.** A float16 value is a
   multiple of 2⁻²⁴ below 2¹⁶, so the float64 sum of up to seven of them is
   exact; rounding it once to float32 loses at most half a float32 ulp of the
   SUM (it is not exact in general — 3.5 + 0.0001 needs 26 significant bits).
   That rounding is the only difference from the function's own float64
   accumulation, and §4 bounds it exactly. float64 would make the composition
   bit-exact at 30 GB for `g025` alone; float32's error is ~10⁻⁷ relative,
   eight orders below anything a reader of a climatology can see.
6. **Where the count is 0 the stored sum is 0.0, never NaN**, so a period sum
   is a plain sum; the composed mean is NaN exactly where the summed count is
   0 — the function's "no training sample" (it divides 0 by 0 there).
7. **count is uint8, asserted ≤ 7**: a calendar month opens at most
   ⌈31/5⌉ = 7 five-day bins. The exporter refuses a month with more rows, and a
   time axis that revisits a (year, month).
8. **Static channels are copied, not recomputed.** `anomaly_transform`
   subtracts nothing on a channel with no temporal variance (its climatology is
   0.0 there). Its test runs over every row whatever is held out, so the answer
   does not depend on the year set; `stats.json` and the index copy it from
   `data/family7_clim_index.json` (`static_chans`; none on the real tensor).
   Recomputing it here would be a second copy of part of the function, which
   `tests/test_one_anomaly_transform.py` exists to prevent.
9. **Streaming, read-only.** The exporter reads each group through the
   read-only memmap `ml/tensor_io.py::load_tensor` returns, one (year, month)
   run of ≤ 7 bins at a time (≤ 102 MB of float16 at 0.25°), and never makes a
   writable copy. RAM is not a constraint; disk is (tensor + output).
10. **Restore from a hosted runner.** E-083's box (Japan) died during its own
    download-back of 2.5 GB. Here the box uploads and leaves a manifest (every
    file's sha256); a GitHub-hosted runner streams every file back with
    parallel ranged GETs hashed in order (no disk), and only a full match
    writes the index (`restore_verified: true`).
11. **Considered and not built: cumulative (prefix) sums**, which would make a
    contiguous period two plane reads instead of Y. In float32 the difference of
    two large prefix sums loses the exactness this design is for (a one-year
    period would read ~10⁻⁵ z of error out of a sum of ~900); in float64 it
    doubles the bytes. A run of years is already ONE range request here;
    revisit if the 0.25° bytes per period (§3) turn out to matter in the page.

## 3. Storage

| group | sum.npy | count.npy | one plane (sum) | one (month, channel), all years |
|---|---|---|---|---|
| `g025` | 12×7×43×721×1440×4 B = **15.00 GB** | **3.75 GB** | 4.15 MB | 178.6 MB (+44.6 MB counts) |
| `g100` | 12×15×43×181×360×4 B = **2.02 GB** | **0.50 GB** | 0.26 MB | 11.2 MB |
| `oc025` | 12×2×28×721×1440×4 B = **2.79 GB** | **0.70 GB** | 4.15 MB | 116.3 MB |
| `rg100` | 12×32×21×181×360×4 B = **2.10 GB** | **0.53 GB** | 0.26 MB | 5.5 MB |
| total | **21.91 GB** | **5.48 GB** | | **27.39 GB** on the Hub |

Each file is 128 bytes of `.npy` header plus that. At 0.25° a thirty-year
period of one channel-month is 125 MB of sums + 31 MB of counts in two range
requests — inside the Data tab's 600 MB read cap, but heavy for a globe paint;
at 1° it is under 8 MB. The website decides how to present that.

Box: verified host, any GPU, ≥ 150 GB disk (61 GB tensor + 27.4 GB output +
1.8 GB of the published climatology for the falsifier); RAM irrelevant.

## 4. Falsifiers

- **The real-data one, a step of the export job that refuses the upload when
  it fails** (`ml/export_family7_monthly.py verify`): for every published E-083
  version — `all`; `dev` = all years minus 2009/2017/2023; `paper` = 1982–2020
  minus 2009/2017 — and every group, Σsum/Σcount over that version's training
  years must reproduce the published `clim.npy`: NaN exactly where it is NaN,
  and per cell

      |Σsum/Σcount − clim| ≤ Σ_y ½·ulp₃₂(sum_y)/Σcount + ½·ulp₃₂(clim) + 4·eps₆₄·|clim|

  That is a rigorous bound, not a tuned threshold: the function's float64 sum
  of float16 values is exact (multiples of 2⁻²⁴ below 2²⁵), so is the float64
  sum of the float32 per-year sums, so the only errors are the float32
  roundings of each stored sum and of the published mean. Also exact: Σ of the
  bins-per-month table over the version's years must equal the version's
  `n_train_bins`. Reported per (version, group): max |Δ| in z and in the
  channel's unit, max |Δ| in float32 ulps of the mean, max Δ/bound.
- **The toy one** (`tests/test_export_family7_monthly.py`, 40 checks): on a
  four-group toy tensor, five year sets (all, a scattered six, one year, dev,
  paper) agree with a DIRECT `anomaly_transform` call within the bound and to
  < 4 ulps of the mean; counts are uint8, ≤ 7 and equal an independent
  recount; sums are bit-identical to an independent float64 sum rounded once;
  `rg100` and `oc025` live on their own axes; one plane and one run of years
  read at the stated offset are that slice; `verify` passes against the
  toy's E-083 climatology and FAILS when one stored sum is nudged past the
  bound; the publisher refuses a byte that does not come back, another stem,
  a failed verify report; the committed fixture matches its own files.
- **The live one, after publish** (§6): range-read the 2015-02 SST plane of
  `g025` sum + count from the Hub and compare sum/count (in °C) with the mean
  of the same bins range-read straight out of the tensor `.npy` on the Hub (an
  independent path); and rebuild the `all` February SST climatology from its
  43 planes and compare with `clim/all/g025/clim.npy`'s plane.

## 5. What exists (files)

- `ml/export_family7_monthly.py` — `export` / `verify` / `fixture`.
- `ml/publish_family7_monthly_index.py` — `upload` (box: manifest + one Hub
  commit per group) / `index` (hosted runner: stream back, CORS, index) /
  `manifest`; `--local` for the fixture.
- `.github/workflows/family7-monthly.yml` — `workflow_dispatch` only; job
  `export` on the rented box, job `restore` on `ubuntu-latest`.
- On the Hub: `tensors/family7_global025_pentad_l2/monthly/<group>/{sum.npy,count.npy,stats.json}`
  on `chfrank/earth-tensors`.
- `data/family7_monthly_index.json` — the site's address book (urls, header
  lengths, shapes, dtypes, axes, plane bytes, sha256, bytes, years, bins per
  month, static channels, channel metadata and `norm` copied from
  `data/family7_index.json`, the measured CORS from `https://blauewelt.org`,
  the falsifier's numbers).
- `data/family7_monthly/fixture/` — the same schema over the smoke tensor
  (`g100` + `oc025`, January 2010 only, 0.95 MB), for the website's tests.

## 6. Status

- 2026-10-06: plan, exporter, publisher, workflow, 40 toy checks and the
  fixture written; the fixture path ran end to end (export → falsifier
  against the committed E-083 fixture → index). Commit `4a481b4`.
- 2026-10-06: **BUILT AND PUBLISHED** — family7-monthly run 1 (the export of
  the per-year monthly sums and counts of all four groups, with the
  falsifier and a hosted restore). Box: Vast instance 54452966 (offer
  47927849, Quebec, verified, 129 GB RAM, 24 CPUs, 3.2 Gbps up/down,
  $0.116/h), rented 08:01 UTC, job 08:05–08:46, destroyed 08:47 — ≈ 46 min,
  ≈ $0.09. Pull 61.16 GB in 10.4 min (98 MB/s); sha256 of every group ✓;
  export 15.3 min (g025 516 year-months; max count 7 in every pentad group,
  1 in `rg100`); falsifier 10.7 min; upload 4.3 min (one commit per group).
  The restore on a GitHub-hosted runner streamed all 27.39 GB back in 71 s
  (the 15 GB `g025` sum at 389 MB/s) and every sha256 matched; CORS from
  `https://blauewelt.org`: 206, `access-control-allow-origin: *`, on
  `g025/sum.npy` and `g025/count.npy`.
- **The falsifier on the real data** — Σsum/Σcount over each published
  version's training years against that version's `clim.npy`, every cell of
  every month and channel: NaN pattern identical everywhere, Σ bins per month
  = `n_train_bins` exactly (3142/2923/2703 for `g025` and `g100`,
  1997/1778/1558 for `oc025`, 252/216/180 for `rg100`), and max |Δ|/bound
  ≤ **0.73** (all twelve version × group pairs). In absolute terms max |Δ| is
  4.8e-7 z for `g025` (6.9e-7 in the channel's unit), 2.4e-7 z for `g100`
  (2.3e-5 in its largest-unit channel), 2.4e-7 z for `oc025`, 4.5e-7 z for
  `rg100` (1.1e-6). Measured in float32 ulps OF THE MEAN the worst cells read
  up to 65,536 ulps — every one of them a cell whose mean is close to 0 in
  z-units, where an ulp of the mean is tiny while the rounding error scales
  with the yearly sums being added; that is why the tolerance is the bound
  and not a ulp count. On a channel whose mean is away from zero it is a few
  ulps or less: February SST (below) is ≤ 0.50 ulp everywhere.
- **Live, from the sandbox against the Hub** (§4's third falsifier):
  (1) the 2015-02 SST plane of `g025` sum + count (5.2 MB) against the mean of
  the six bins opening in February 2015 (2015-02-02 … 02-27) range-read
  straight out of the tensor `.npy` (87 MB): counts identical, NaN pattern
  identical over 703,902 cells, **max |Δ| = 0 °C** (global mean 13.7885 °C
  both ways; 23.9484 °C at 26.5° N 70° W, n = 6). (2) The `all` February SST
  climatology rebuilt from its 43 planes (223 MB, two byte ranges) against
  `clim/all/g025/clim.npy`'s plane: NaN pattern identical, **max |Δ| =
  6.9e-7 °C = 0.50 float32 ulp** (median 0.25 ulp); 23.3677 °C at the RAPID
  line both ways.
- `data/family7_monthly_index.json` written by the restore job
  (`restore_verified: true`) and committed. Nothing in flight.
