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
- **bgcargo cannot be fetched in one hosted job.** Run #4 was cancelled
  after 62 min: from GitHub's runners the Ifremer GDAC gave 2.2 GB of the
  74.7 GB in that time (≈ 0.6 MB/s, against 23 MB/s from the sandbox
  probe). bgcargo is fetched in lanes instead: 2002–2012, then one lane per
  year.
