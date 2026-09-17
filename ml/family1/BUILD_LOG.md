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
