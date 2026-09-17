# Family 1 build log — E-082 wave 1

*One row per published store of families 1.gf and 1.0.tf. **Family 1.gf** is
the global ocean and atmosphere observations at ≤ 10 km and ≤ 5 days (ship
reports, ocean profiles, floats, moorings, bottles, sea ice). **Family 1.0.tf**
is land and coast at the same scale (weather stations, radiosondes, buoys, tide
gauges, gliders). The plan is
[E-082](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E082_family1_builds.md).
Every row here was checked on the Hub after its run went green: the numbers
come from the published `store.json` and `manifest.json`, not from the job's
own report.*

**How a store is built.**
- **Small stores:** one GitHub-hosted job runs all five stages (list the
  archive, fetch, assemble, publish, check).
- **Large stores:** several hosted **lanes** each fetch a window of years and
  park them on the Hub under `partials/family1_<x>/<store>/<year>/`. Then one
  job assembles the store from those parts (`--parts-from-hub`) and publishes
  it.

Before the store is marked published, the publish stage downloads every file
back and compares its sha256.

**Columns.**
- **N** is the number of rows. For the sharded sea-ice store it is the number
  of daily grids (frames) per hemisphere.
- **Stored bytes** is the sum over the store's `manifest.json`.
- **Years** is the first and last year that holds rows.
- **Run(s)** are family1-build workflow runs:
  [the Actions list](https://github.com/blauewelt/earth/actions/workflows/family1-build.yml).

| store | family | run(s) | N | stored bytes | years | verified | notes |
|---|---|---|---|---|---|---|---|
| `glodap` | 1.gf | #2 (all five stages, hosted, 3 min) | 1,460,215 | 77,433,120 (77 MB) | 1972–2023 (49 years with rows) | 2026-09-17 | GLODAPv3 bottle samples; public; schema 2 |
| `tide_private` | 1.0.tf | #5 (all five stages, hosted, 9 min) | 179,314,808 | 5,917,632,868 (5.92 GB) | 1821–2026 (180 years with rows) | 2026-09-17 | GESLA-4 research-only gauges → **private** repository (anonymous read of its store.json answers 401); schema 3. The adapter projected ≈ 1.7 × 10⁸ rows and 5.5 GB, and the measurement is 5 % above. 538 records fetched in 371 s on the hosted runner, about 13 times faster than the ~1.6 h sandbox estimate |

| `igra` | 1.0.tf | lanes #18 (1905–1984), #27 (1985–2026); assembly #41 (hosted, `--parts-from-hub --assemble streaming`; #40 refused by the window guard, fixed in 071bb5c) | 44,842,766 | 8,565,529,637 (8.57 GB) | 1905–2026 (117 years with rows) | 2026-09-17 | IGRA v2.2 radiosonde soundings; public; schema 3. The adapter estimated 39–48 M soundings and 7.4–9.2 GB |
| `seaice_asi` | 1.gf (tier G, sharded) | #31 (all five stages, hosted, 58 min, `--distribution private`; #6 refused the WGS 84 relabel, fixed in 9b54479) | 5,187 daily frames per hemisphere (n, s) | 1,657,059,360 (1.66 GB, 4,156 files) | 2012-07 → 2026-09 (bins 2228–3265) | 2026-09-17 | Bremen ASI v5.4 AMSR2 6.25 km. Published to the **private** repository because Bremen has not yet answered on redistribution: store.json carries `distribution: private`, a `licence_pending` note and `distribution_override`, the public path answers 404 and an anonymous private read answers 401. 3 frames absent upstream per hemisphere (2013-05-11..13). 8,326 frames declare the Hughes 1980 ellipsoid and 2,048 (from 2018-11-02) declare WGS 84 on the identical grid. The probe estimated 1.42 GB, and the store is 17 % above that |

| `gliders` | 1.0.tf | lanes #20, #38, #60, #64, #67, #69, #71 (7 year windows); assembly #76 (hosted, 4 min) | 2,808,590 | 436,057,724 (436 MB) | 2003–2026 (24 years with rows) | 2026-09-17 | IOOS Glider DAC profiles; public; schema 2. The adapter projected ≈ 5 M profiles and 0.8 GB, and the measurement is **44 % fewer rows** — worth checking against the archive's own count before the store is used |
| `wod` | 1.gf | lanes #16, #25, #57, #62 (4 windows); assembly #75 (hosted, `--assemble streaming`; #73 was killed by the memory assembler) | 15,281,373 | 4,385,949,864 (4.39 GB) | 1772–2026 (195 years with rows) | 2026-09-17 | World Ocean Database casts, all instruments but Argo; public; schema 3. The adapter estimated ≈ 14.7 M casts and 4.2 GB |
| `ndbc` | 1.0.tf | lanes #44, #45, #58, #63 (4 windows, each re-run once after the parser fixes); assembly #74 (hosted, `--assemble streaming`, 45 min) | 628,157,329 | 29,523,701,726 (29.52 GB) | 1970–2025 (56 years with rows) | 2026-09-17 | NDBC buoy and C-MAN standard meteorology; public; schema 2. The adapter estimated ≈ 6.3 × 10⁸ rows and 30 GB — both hit. Ends 2025: NDBC's historical listing has no 2026 file, and an unlisted year is an absence |

| `tide` | 1.0.tf | lanes #19 (1800–1989), #32 (1990–2009), #59 (2010–2026); assembly #82 (hosted, `--assemble streaming`, 21 min; #77, #79 and #80 died on the UHSLC record list timing out from runner IPs) | 1,080,914,415 | 35,671,451,441 (35.67 GB) | 1800–2026 (227 years with rows) | 2026-09-17 | GESLA-4 tide gauges, public track; schema 3. The reader's whole-record sample projected ≈ 8.6 × 10⁸ rows and 28 GB, so the store is **26 % more rows** than projected |
| `bgcargo` | 1.gf | lanes #34 (2002–2012), #35–#54 and #81 (one per year), 2022 / 2025 / 2026 fetched **from the sandbox**; assembly #84 (hosted, 1 min) | 335,231 | 73,467,590 (73.5 MB) | 2002–2026 (25 years with rows) | 2026-09-17 | BGC-Argo synthetic profiles; public; schema 2. The adapter projected ≈ 336 k rows and 80 MB — both hit |

| `oceansites` | 1.gf | lanes #8, #13, #23, #29, #30, #85, #86, #87 (8 windows; #65's 1989–2006 window was cancelled at 4 h 20 m and split); assembly #88 (hosted, `--assemble streaming`, 6 min) | 66,483,202 | 5,518,209,460 (5.52 GB) | 1980–2026 (39 years with rows) | 2026-09-17 | OceanSITES moored time series, tropical arrays excluded; public; schema 2. The adapter's ceiling was ≤ 1.3 × 10⁸ rows and ≤ 11 GB, and the measurement sits under both. The index lists files from 1950, but no year before 1980 holds a kept row |
| `ghcnd` | 1.0.tf | lanes #7, #15, #24, #56 — **all 264 years parked** (1,143,728,366 rows, 42.32 GB) | — | — | 1763–2026 | **not assembled** | Needs the rented box: 42.3 GB of parts plus ≈ 47 GB of store is past a hosted runner's ≈ 90 GB. Blocked on `/home/claude/.vast_key`, which is absent |
| `icoads` | 1.gf | lanes #14, #21, #55, #61, #66, #68, #70, #72 — **all 365 years parked** (1,107,951,670 rows, 47.64 GB) | — | — | 1662–2026 | **not assembled** | Same: 47.6 GB of parts plus ≈ 52 GB of store. Blocked on the Vast key |

## Notes

- **Measured on hosted runners, 2026-09-17.** A GitHub-hosted
  `ubuntu-latest` job had **86.4 GB free** under the workspace (run #5's
  disk line). The ~14 GB that lanes were first sized for is the nominal SSD,
  not what the image leaves free. Lanes are therefore sized by time, and disk
  is only a secondary limit.
- **Hub commit budget.** The Hub allows about 256 commits a repository an
  hour. `--push-parts` now pushes a whole lane with `push_many`: parts in
  commits of ≤ 400 files or ≤ 4 GB, every file downloaded back and compared,
  then every `done.json` in one commit. Run #7 (ghcnd 1763–1899) parked 137
  years (831 MB) in two commits and 105 s.
- **Assembly from parked parts is `stage=all --parts-from-hub`.** With that
  flag, `fetch` is the pull, and `assemble` requires `fetch`'s marker, so
  `stage=assemble,publish,check` refuses on a fresh machine (run #39,
  cancelled).
- **Upstream shapes the lanes met, each fixed with a test before the lane was
  re-run:**
  - **NDBC:** a whole gzip of an empty file (42008h1980, #17); a value
    written `02,4` (#22); a header column no line carries (42otph2000, #28);
    `MM` for a missing value (#26). A sandbox pass of all 17,049 stdmet files
    through the fixed parser then raised nothing, with 5 empty files, 59,936
    `MM` values, 566,451 lines without the unread trailing column, 4
    bad-length lines and 1 non-numeric line.
  - **OceanSITES:** a file name with a space (#12), and a refused-layout
    counter that could not be summed (#10, #11).
  - **Gliders:** an aggregate with neither leading axis of length 1
    (bass-20150827T1909, #33). It is now skipped by name.
  - **Bremen sea ice:** the WGS 84 relabel of the same grid from 2018-11-02
    (#6).
  - **IGRA:** the 30 M-sounding window guard also fired on a parts pull
    (#40).
- **The assembler's `auto` threshold counts ROWS, not width.** wod is
  15.3 M rows — under the 50 M that switches to streaming — but 128
  channels, so the memory assembler needed about 8 GB for the value block
  alone and run #73 was killed (exit 143, "the runner has received a
  shutdown signal"). Every wide store is assembled with `--assemble
  streaming`. ndbc's streaming assembly of 628 M rows still peaked at
  14.58 GB RSS against the runner's 16 GB.
- **What actually fits a hosted runner.** ndbc assembled and published there:
  29.52 GB of store, 64.57 GB free before it started, and the restore needs
  only the largest single file (values.npy, 12.56 GB, deleted between
  files). Values are float16, so a store's largest file is 2 bytes a value.
  ghcnd (42.3 GB of parts + ≈ 47 GB of store) and icoads (47.6 + ≈ 52 GB)
  still exceed it and need the rented box.
- **bgcargo cannot be fetched in one hosted job.** Run #4 was cancelled
  after 62 min: from GitHub's runners the Ifremer GDAC gave 2.2 GB of the
  74.7 GB in that time (≈ 0.6 MB/s, against 23 MB/s from the sandbox
  probe). bgcargo is fetched in lanes instead: 2002–2012, then one lane per
  year. Even one lane at a time with ten attempts, Ifremer then answered
  `Connection refused` for tens to hundreds of floats a run (481 in run #54),
  while the same archive serves this sandbox at 23 MB/s: 2022, 2025 and 2026
  were fetched here and pushed with `--push-parts` in 8–10 minutes each,
  against 136 and 187 minutes of failing on runners. A lane's stage markers
  live under `<work>/<store>/`, not per window, so each sandbox year needs
  its own `--work` or it silently skips ("already done").
- **UHSLC throttles the runners after a build.** Three tide assemblies
  (#77, #79, #80) failed in the index stage with the GESLA record list
  timing out, while the same URL answered this sandbox in 2 s. It cleared
  about 45 minutes after the last tide lane finished, and #82 went through.
