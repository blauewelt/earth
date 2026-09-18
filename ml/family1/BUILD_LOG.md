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
- **N** is the number of rows. For a sharded tier-G store it is the number of
  stored FRAMES (daily grids for `seaice_asi`, pentad grids for `chirps05`),
  with the bins and tiles beside it.
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
| `ghcnd` | 1.0.tf | lanes #7, #15, #24, #56; assembly #89 (**the rented box**, `--parts-from-hub --assemble streaming`, 63 min) | 1,143,728,366 | 46,914,117,496 (46.91 GB) | 1763–2026 (264 years with rows) | 2026-09-17 | GHCN-Daily station-days; public; schema 3. The adapter projected ≈ 40 GB. Peak RSS 48.90 GB — no hosted runner could have assembled it |
| `icoads` | 1.gf | lanes #14, #21, #55, #61, #66, #68, #70, #72; assembly #93 (the same box, 72 min; #91 and #92 could not commit through Xet) | 1,107,951,670 | 52,073,957,276 (52.07 GB) | 1662–2026 (286 years with rows) | 2026-09-18 | ICOADS marine reports (IMMA1), 1662 onward; public; schema 3. The adapter projected ≈ 1.37 × 10⁹ rows and ≈ 64 GB from the listings’ bytes — the measurement is **19 % fewer rows** |

**The box.** One rented Vast instance did both: offer 48942875, instance
51358803, a *verified* Quebec host with 129 GB of RAM, 32 CPUs, 2.2 Gbps up
and a 250 GB disk, at $0.107/h plus $0.069/h of storage. Created
2026-09-17 23:00Z, destroyed 2026-09-18 04:15Z: **5.26 h for $0.93**. It was
not the cheapest qualifying offer ($0.096/h, 16 GB of RAM) and the 1.7 ¢/h
went on memory, which the measurement justified: ghcnd's streaming assembly
peaked at **48.90 GB RSS**.

| `chirps05` | 1.0.tf (tier G, sharded) | lanes #95 (1981–1990), #96 (1991–2000), #97 (2001–2010), #98 (2011–2020), #99 (2021–2026), 15–33 min each; assembly #127 (hosted, `--parts-from-hub`, 66 min) | 3,288 pentad frames in 3,336 bins (628,008 tiles) | 19,064,019,352 (19.06 GB, 6,674 files) | 1981-01 → 2026-08 (bins −73–3262) | 2026-09-18 | CHIRPS v3.0 pentad precipitation, 0.05°, land 60°S–60°N; public; CC BY 4.0, confirmed from chc.ucsb.edu. Exception E4: **F = 2** half-bins of 2.5 days, the pentad filed by its MIDPOINT — F = 1 is impossible because 14 bins of the record hold two pentad midpoints (measured, all in February) — and a `frame_table` in tile_grid.json gives every (bin, frame) its pentad's own dates. 3,384 half-bins hold no pentad and are counted `no_pentad_in_slot`; nothing is absent upstream. Pentad lengths came out 2,922 × 5 days, 320 × 6, 35 × 3, 11 × 4, which is the producer's calendar exactly. 57.68 GB of GeoTIFF read; valid fraction 0.28056. The probe projected 21.18 GB and the store is **10 % under** it |

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
- **The Xet uploader could not publish a 52 GB store; classic LFS could.**
  On that one box, ghcnd's 46.91 GB went up first time, and icoads' 52.07 GB
  died in `huggingface_hub/_commit_api.py::_upload_xet_files` ->
  `session.new_upload_commit` on every attempt of two runs — #91 as one
  commit (four attempts, 63 of the 75 allowed backoff minutes) and #92 in
  batches of three files (four attempts on batch 1/4). A 1 KB NDJSON commit
  to the same repository from the sandbox answered 200 in 1.0 s while #92 was
  still retrying, and ghcnd's twelve files were already on the Hub, so
  neither the repository nor the commit endpoint was at fault.
  `HF_HUB_DISABLE_XET=1` on the build step put the publish back on the
  classic LFS path and #93 published and restore-verified all ten files in
  27 minutes. The publish also now commits `PUBLISH_BATCH` (3) files at a
  time with store.json alone and last, which is the tier-G rule.
- **The Earthdata account is GOOD, and the first check could not see it.**
  Re-run #90, with IPv4 forced and PO.DAAC asked for bytes behind the login:
  LP DAAC (MOD11C1), GES DISC (MERGIR) and PO.DAAC (MUR SST) each answered
  **HTTP 206 with 1,024 bytes after 4–5 redirects through
  urs.earthdata.nasa.gov**, no approval page and no EULA. Run #3's two
  `[Errno 101] Network is unreachable` failures were the runner's missing
  IPv6 route (both hosts publish AAAA records), and its PO.DAAC "ok" was a
  CloudFront 200 that never passed through Earthdata Login — reported as
  `ok_without_login` now rather than as a pass.
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

## Notes — E-082 wave 2, the sharded tier-G stores (added 2026-09-18)

- **A GeoTIFF source means `rasterio` on the runner.** `lossyear` (Hansen
  Global Forest Change) and `chirps05` (CHIRPS pentads) both read GeoTIFFs,
  so `family1-build.yml`'s install step carries `rasterio` and prints its
  GDAL version. Neither source needs an account.
- **`adapter_env` is how a tier-G adapter's construction-time knobs reach a
  runner.** A space-separated list of NAME=VALUE pairs, exported for the
  build step only, echoed into the log, and REFUSED for anything that is not
  a plain upper-case identifier or that sits in a credential's namespace
  (`HF_`, `GITHUB_`, `EARTHDATA_`, `FIRMS_`, `FLUXNET_`, `COPERNICUS`).
  `lossyear`'s probe needs it: the store has 4,480 groups and a probe
  measures five Hansen tiles (`LOSSYEAR_TILES=...`).
- **A tier-G group's frame size is capped by the PROBE, not by the store.**
  `stage_probe_grid` converts each frame to float64 to check the channel
  bounds, so a group of H × W × C elements costs 8 H W C bytes there plus the
  boolean temporaries. Hansen's native 40,000 × 40,000 × 2 tile would need
  25.6 GB of that and could not be probed at all; cut into 2.5-degree
  sub-tiles of 10,000 × 10,000 it needs 1.6 GB and the whole probe fits this
  sandbox's 7 GB. **Size a tier-G group so its probe fits, and read the
  source in whatever shape the FILES want** — Hansen's LZW GeoTIFFs are
  striped one row at a time, so the fetch reads a full-width 10,000-row band
  once per variable and slices four sub-tiles out of it; sixteen square
  windows would decompress every tile four times over.
- **Two one-line gaps in `ml/build_family1_stores.py` that only a float16 or
  sparse tier-G store reaches.** `stage_probe_grid`'s `compression_ratio`
  divided by a zero mean frame size, which is exactly what a probed frame
  with no stored tile legitimately has (an all-ocean lossyear sub-tile), and
  lost the whole probe to a ZeroDivisionError; it is None now.
  `check_grid_smoke` compared frames with `np.array_equal` and no
  `equal_nan`, so no float16 group with a missing pixel could pass its own
  smoke — `stage_publish_grid` and `http_verify` already compared with
  `equal_nan` and it now does too.
- **A hosted runner's SIX-HOUR limit is what decides hosted-or-box for a
  tier-G store, not its size.** `lossyear` projects to 60.7 GB — which fits
  a runner's disk — and needs about 11 h of fetch, because `write_bin` was
  402 s of its probe's 672 and the store is ONE bin with no year axis to
  spread over. Splitting by tile across two lanes does not work either: each
  lane's `counts.json` declares only its own groups' grids and
  `stage_assemble_grid` refuses parts written for a different grid
  declaration. So `lossyear` waits for one box run; `chirps05` (21 GB, 3,288
  pentads, five decade lanes at 15–33 min each) did not.
- **CEDA serves a GitHub runner at 5.0 MB/s.** Measured on `oc4k`'s probe
  (#100): 4.99 GB of OC-CCI in 1,004 s, the same order as the 2.0 MB/s this
  sandbox sees. That is what sizes the `oc4k` lanes at ONE YEAR each — 58.7
  GB and about 3.3 h — rather than by disk.
- **Twenty concurrent jobs is the ceiling, and a queued assembly behind
  nineteen lanes is a self-inflicted delay.** Dispatching all 26 `oc4k` year
  lanes put the `chirps05` assembly in the queue; cancelling the newest
  eleven lanes (which had done the least) let it start. Dispatch the job you
  need FIRST, then fill the pool.
- **The Hugging Face tree API pages at 50 entries and refuses `limit=1000`.**
  A parts directory that looks like 50 files is a page, not a listing;
  `?limit=100` with the `Link` header's cursor is what reads the real 149.
