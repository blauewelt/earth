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
| `oc4k` | 1.gf (tier G, sharded) | 26 one-year fetch lanes on hosted runners, ~3.3 h each at CEDA's 5 MB/s (#101–#115, #128–#132, #136, #138 and #255–#257 for 2017–2019; the lanes in between were cancelled by the concurrency group and re-run); assembly #278 on **the rented box** (Vast 51415980, runner `gpu-box-31947967`, RTX 3060, 420 GB disk, $0.131/h; `--parts-from-hub --assemble streaming`: pull 11,449 s, assemble 14,955 s, publish's own re-check 3.8 h, then the download-back died at file 1,220 of 3,703 on a dropped Hub connection — fixed in 300905a, `_download` retries transient failures); #287 (`stage=publish,check` on the same box, resuming from the assembled store: publish 15,519 s, check 14,155 s) | 9,224 daily frames in 1,850 bins (3,642,817 tiles) | 175,875,077,967 (175.88 GB, 3,702 files) | 1997-09 → 2022-12 (bins 1145–2994) | 2026-09-20 | ESA Ocean Colour CCI v6.0 daily 4 km chlorophyll-a and diffuse attenuation, C = 3 (`log_chl` — log10 of chlorophyll-a, `kd_490`, `total_nobs`), 4320 × 8640; public; ESA CCI free-and-open with attribution, confirmed from the files' own `license` attribute and CEDA's README. 26 frames absent upstream and 245 before the record (the record opens 1997-09-04; 49 bins of the requested window lie outside it); the daily record on CEDA ends 2022-12-31, not the ledger's 2026-06-30. 1.569 TB of netCDF read (18,448 files, 332,259 s of fetch across the lanes); valid fraction 0.1157 — ocean pixels seen through cloud. The upload was almost free: the parts are byte-identical to the store files, so the Hub deduplicated 176 GB into eight metadata commits in ~200 s. **The three `check_store` passes (assemble, publish, check) cost ~3.8 h each on this box's CPU** — every one of the 3.64 M tiles is decompressed each time — so a ~200 GB tier-G store is a ~12 h box job even with the parts already fetched |
| `swh` | 1.gf | six fetch lanes #153–#158 (1991–1996 … 2019–2023, 21–35 min each; #143–#148 were the same lanes before the fix) parked 33 years of parts; assembly #290 on **the rented box** (gpu-box-31947967, `--parts-from-hub --assemble streaming`: pull 2,348 s, assemble 1,012 s at peak RSS 54.24 GB, publish 2,196 s with every file downloaded back, check 450 s; 1 h 41 min in all) | 1,680,274,586 | 55,449,091,761 (55.45 GB, 10 files) | 1991–2023 (33 years with rows) | 2026-09-20 | ESA CCI Sea State v4 along-track significant wave height, one row per 1 Hz altimeter sample, eleven missions from TOPEX/Poseidon to Sentinel-3B; C = 3 (`swh`, `swh_denoised`, `swh_uncertainty`, metres); public; ESA CCI free-and-open / Copernicus Marine with attribution, confirmed from the files' own `license` attribute. Read anonymously from Copernicus Marine's native object store — no account, the toolbox never imported. Schema 2. 11,832 daily files read, 7 days absent upstream (1992-07-20..22, 2001-01-19..22) and named; 8 of 1,680,274,594 samples dropped (out-of-bounds `swh` 2,774 and `swh_denoised` 3,047 became NaN, never clipped). The lanes' 48.73 GB of parts assembled to a 55.45 GB store (the CSR bin offsets and the platform column are the difference), and the row count sits between the lanes' ≈ 1.5e9 projection and the probe's 1.665e9 extrapolation |

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
| `fire` | 1.0.tf | hosted year lanes #233 (2000–2006), **#264 (2007–2013), #284 (2014–2017), #286 (2018–2020), #288 (2021–2023), #289 (2024-01-01..2026-09-30)** — the last five run one at a time on 2026-09-19/20, 26.0 h of fetch in all; assembly **#291** (hosted, `stage=all --parts-from-hub --assemble streaming` 2000-01-01..2026-09-30, 43 min: 27 years pulled, 631.8 M rows scattered in 209 s, 22 GB published and restore-verified) | 631,836,727 | 22,114,314,679 (22.11 GB, 11 files) | 2000–2026 (bins 1375–3266, 1,889 of 1,892 live; 27 years with rows) | 2026-09-20 | NASA FIRMS active-fire detections; public; schema 2, C = 4. `rows_read` = `rows_kept` = N, from 50.26 GB of source CSV over 3,792 windows, 14 of which held no detection. **Five satellites, and the record's shape is their launch dates**: Terra 40,799,565 rows and Aqua 73,853,160 (MODIS, 2000 →), Suomi-NPP 294,864,805 (VIIRS, 2012 →, which is why the year count jumps from 4.7 M in 2011 to 26.0 M in 2012), NOAA-20 170,628,082 (2018 →) and NOAA-21 51,691,115 (2024 →, which is why 2024 is the largest year at 69.2 M). Each satellite's own spelling in the source is mapped by name and counted in `rows_by_spelling`, never guessed. Every day belongs to exactly ONE source — Standard Processing where it exists, Near-Real-Time after SP's own `max_date`, both read from the `data_availability` endpoint — so no detection is fetched twice and a gap between the two is a refusal. `confidence` is MODIS-only by design (VIIRS publishes three classes, which live in `qc`), and `log2_fp` is the MODIS 1 km pixel (−4.798) with each platform's own value, satellite, instrument and nominal pixel size in `platforms.json` — a VIIRS detection is 375 m (−6.214). 143 rows have a `brightness` and 77 an `frp` outside bounds; both are NaN and counted. **The lane length is set by the source count, not the year count**: 7 years fit one lane in the MODIS era (#233, 3 h 03 m) and only 3 fit once four sources overlap (#289, 5 h 29 m for 2024–2026) — see the wave-5 notes |

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

## Notes — E-082 wave 4, the read-out targets and the credentialed ocean/atmosphere stores (added 2026-09-18)

**What this wave was asked for and what it measured.** Eight stores:
`burned500`, `alerts`, `cat_gfm` (family 1.0.tf read-out targets) and
`sst_acspo02`, `irtb`, `xco2`, `swot`, `swh` (family 1.gf). Five adapters
landed with smokes; three did not, and the reason in each case is a
MEASUREMENT about the archive rather than a shortage of code. The access
findings below are the wave's main result, because each one is a decision the
note cannot make from a product page.

### Access findings, each measured rather than transcribed

- **`swh` NEEDS NO ACCOUNT.** The ledger reaches ESA CCI Sea State v4
  along-track wave height through Copernicus Marine "(account, as
  `slatrack`)". The same daily files sit on Copernicus Marine's OWN native
  object store, `https://s3.waw3-1.cloudferro.com/mdl-native-05/native/
  WAVE_GLO_PHY_SWH_L3_MY_014_005/...`, which answers an ANONYMOUS
  ListObjectsV2 and an anonymous GET (measured 2026-09-18 from the sandbox:
  12 paginated pages, 11,832 daily files, 116,351,837,853 bytes). So
  `credentials = ()`, the `copernicusmarine` toolbox is never imported, and
  the whole store can be built on a hosted runner, a box or the sandbox. The
  prefix is read out of the product's public STAC each run and a href that
  leaves that bucket is a refusal.
- **EARTHDATA DOWNLOADS NEED `requests` AND A NETRC, NOT `urllib`.** The first
  `swot` probe (#152) listed cycle 010 correctly from CMR and then failed on
  all twelve passes with `HTTP 401 — HTTP Basic: Access denied.`
  `f10b.http_to_file` is urllib, urllib follows the 302 to
  `urs.earthdata.nasa.gov` WITHOUT credentials, and the workflow's `.netrc`
  is never consulted. `requests`' `Session.rebuild_auth` looks the host up in
  the netrc on every redirect hop and strips the Authorization header on a
  cross-host hop, so a netrc naming Earthdata Login alone sends the password
  there and nowhere else. `ml/family1/adapters/_common.py` now carries
  `earthdata_session` / `earthdata_download` / `netrc_has_urs` /
  `force_ipv4_once`, and every Earthdata adapter of this wave REFUSES in
  `fetch_preflight` when no such netrc is visible — #152 spent its whole
  listing to discover a page of 401s. `_modis_cmg.py` has the same trio for
  the MODIS CMG stores; the two copies should be consolidated.
- **NO ANONYMOUS ROUTE TO A NASA FILE'S VARIABLE LIST.** A GES DISC `.nc4`,
  its `.nc4.xml` sidecar, a PO.DAAC L3S granule and a SWOT pass all answer an
  unauthenticated GET with 302 → 401, and CMR publishes no UMM-Var for these
  collections (`variables.json?concept_id=...` returns 0 hits). Writing a
  variable-name table from memory is what ADAPTER_CONTRACT rule 4 forbids, so
  `xco2`, `irtb` and `sst_acspo02` all RESOLVE each quantity against a
  candidate list, refuse with the file's own variable list when a required one
  matches nothing, and write ONE real file's WHOLE inventory (path, dtype,
  shape, dims, units, scale/offset, fill, flag table) into `plan.json`. One
  runner round trip settles the format whichever way it goes, and the
  resolution that was used is recorded in the store.
- **THE GFM FLOOD CATALOGUE HAS NO ANONYMOUS LISTING, so `cat_gfm` is
  OPERA-only.** `https://gfm.eodc.eu/` answers 404; the service's API is
  `https://api.gfm.eodc.eu/v2/` and its Swagger (`/v2/swagger.json`, "GFM JRC
  API", version 24.01) shows the shape of the thing: every product path is
  scoped to a subscribed AREA OF INTEREST (`/aoi/{aoi_id}/products`,
  `/download/product/{product_id}/{user_id}`) behind `/auth/login` and
  `/auth/get_bearer_token`, with `/aoi/create` and an allowance per user.
  There is no global scene listing at all, anonymous or otherwise — the
  Copernicus Global Flood Monitoring Sentinel-1 archive is an
  alert-subscription service, not a catalogue. So the `cat_gfm` store is
  buildable from OPERA DSWx-HLS (C2617126679-POCLOUD, 2016-01 →, 10,703
  granules on 2024-06-01 alone) and DSWx-S1 (C2949811996-POCLOUD, 2023-12 →),
  both anonymously LISTABLE through CMR, and the GFM half needs a decision
  from Chris: register a GFM account and subscribe areas of interest, or drop
  the GFM column from the ledger row.
- **RADD, GLAD-L AND GLAD-S2 ARE CC BY 4.0 AND HAVE NO KEYLESS BULK
  DOWNLOAD.** The ledger's `alerts` row says "CC BY 4.0 (GLAD terms
  unverified)". The terms are now VERIFIED, from Global Forest Watch's own
  data API (`https://data-api.globalforestwatch.org/dataset/<name>`):
  `wur_radd_alerts`, `umd_glad_landsat_alerts` and `umd_glad_sentinel2_alerts`
  each declare `license: "[CC by 4.0](https://creativecommons.org/licenses/
  by/4.0/)"` with the citation to quote. **So the GLAD terms do not force the
  private track.** What blocks them is ACCESS, not licence: the native
  `epsg-4326` date_conf tile sets are
  `s3://gfw-data-lake/<dataset>/<version>/raster/epsg-4326/10/100000/
  date_conf/geotiff/{tile_id}.tif`, and that bucket is REQUESTER PAYS —
  an anonymous `list-type=2` answers `AccessDenied: Anonymous users cannot
  invoke requests against Requester Pays buckets`. The API's own
  `/download/geotiff` proxy answers `403 Request is missing valid API key`.
  UMD's own page (`glad.umd.edu/dataset/glad-forest-alerts`) links only Earth
  Engine apps. The public GCS prefix `earthenginepartners-hansen/alert/` holds
  a 2017-era partial snapshot (261 objects, 242 of them under `2017/`) and is
  not a maintained archive. So RADD and the two GLAD systems need one of: a
  free GFW data-API key (a new repository secret), an AWS account willing to
  pay the requester-pays egress, or Earth Engine — each a decision, none of
  them a keyless route. OPERA DIST-ALERT via CMR/LPCLOUD is the one system
  `alerts` can be built from today.
- **DIST-ALERT'S VOLUME IS THE REAL OBSTACLE, AND IT IS ARITHMETIC.** CMR
  lists **11,333 DIST-ALERT granules for 2024-03-01 alone** (one per HLS tile
  per acquisition, each ~10 layers of 3,660 x 3,660 uint8). One row per
  alerted pixel-date means reading every granule of every day: order a
  terabyte a day, against a store the note sizes at 30 GB. The cheap
  alternative is the ANNUAL product, OPERA DIST-ANN, which settles each tile's
  disturbance date once a year — a few thousand granules a year instead of
  four million — and `family1tf.tex` §4.5 already files an annual target under
  the bin of its period's end. Whether `alerts` is built from DIST-ANN
  (cheap, annual) or from DIST-ALERT over a bounded REGION (expensive, dated
  to the pass) is a decision for the main session; neither is a code problem.
- **`burned500` IS HDF4 AND THE TIER-G BIN AXIS IS THE PROBLEM, NOT THE
  FORMAT.** MCD64A1 v6.1 (C2565786756-LPCLOUD, 2000-11-01 →) publishes 268
  HDF-EOS2 tiles for 2019-08 at ~0.15 MB each through LP DAAC's cloud
  archive, and `pyhdf`'s manylinux wheel reads HDF4 (rasterio's GDAL has no
  HDF4 driver — the wave's MODIS CMG stores hit this first and the install
  step now carries `pyhdf`). What is not settled is the FILE COUNT. A monthly
  frame filed under the bin holding the month's 15th means ~310 frames spread
  over ~1,890 five-day bins, and `fetch_grid_year` writes a shard and an index
  for EVERY bin of the window unless all its frames are `before_record` or
  `after_record` (`OUTSIDE_RECORD` in `ml/build_family1_stores.py`). With one
  group per MODIS tile row (18 groups, 2,400 x 86,400, the largest frame whose
  probe fits a runner) that is ~11,000 useful files and ~57,000 empty-bin
  files; with one group per tile (268 groups) it is 166,000 files, past the
  Hub's 100k-per-repository limit. **The clean fix is a third skip reason in
  `OUTSIDE_RECORD` — a one-line framework change this wave deliberately did
  not make**, since `ml/build_family1_stores.py` belongs to the framework and
  three other waves were editing the same tree. Until it lands, a monthly
  tier-G store cannot be filed on the five-day bin axis without paying tens of
  thousands of empty shards.

### What ran

- **`swh` is keyless and the sandbox measured its whole record**, so the probe
  ran BOTH places and agreed to the byte: 2015-01 gave 4,352,091 rows from
  4,352,093 samples over 31 days, 304,162,654 bytes, 69.889 source bytes a
  row and 33 stored, NaN fractions swh 0 / swh_denoised 0.0115 /
  swh_uncertainty 0.0249, zero out of bounds, three missions (cryosat-2
  1,255,788, jason-2 1,719,307, saral 1,376,996). The sandbox did it in
  14.7 s (20.7 MB/s) and the hosted runner in 22.6 s (13.4 MB/s) — the only
  difference between the two reports. **116.35 GB / 69.889 B a row projects
  1.665e9 rows and about 55 GB stored, 17 % under the note's ~2e9 rows and
  ~70 GB.**
- **`swh`'s six fetch lanes** — #153 (1991-1996, 20.7 min), #154
  (1997-2002, 23.2), #155 (2003-2008, 30.9), #156 (2009-2013, 21.5), #157
  (2014-2018, 34.3) and #158 (2019-2023, 35.4) — all went green and parked
  **all 33 years, 1,626 files, 48,730,861,888 bytes (48.73 GB), every year
  carrying its `done.json`**, under `partials/family1_gf/swh/<year>/`. The
  runner's own index reproduced the sandbox's listing exactly: 11,832 files,
  116,351,837,853 bytes, 1991-08-03 → 2023-12-31, 7 days absent, and the
  first file's own `flag_meanings` naming twelve missions (cryosat-2, envisat,
  ers-1, ers-2, jason-1/2/3, saral, sentinel-3_a, sentinel-3_b,
  sentinel-6_a, topex-poseidon). Rows a year, measured: 39,462,435 (1996),
  59,609,148 (2002), 42,950,441 (2008), 45,484,441 (2013) — about 47 M a year,
  so **≈ 1.5e9 rows** over the record, against the probe's 1.665e9
  extrapolation and the note's ~2e9.

### `sst_acspo02`: the note is 4.6x high, and one year fits a hosted runner

The probe (#172, 2020-01, `ml/family1/probes/sst_acspo02_2020-01.json`) read
eighteen of the month's thirty-one days and measured, per daily 18,000 x 9,000
x 2 frame:

| what | measured | the note / brief |
|---|---|---|
| valid fraction after the grade-2 floor | **0.3492** | "v ≈ 0.3" |
| compressed bytes a frame | **51,703,095** (mean; 50.07-53.88 MB) | — |
| raw bytes a frame | 648,000,000 | — |
| compression ratio | **12.53x** | — |
| bytes a VALID pixel | **0.457** | the note's formula charges 2C x 1.2 = 4.8 |
| tiles stored / empty a frame | 1,701 / 855 of 2,556 | — |
| encode seconds a frame | **49.4** (of 59.7 s wall) | — |
| **the whole record (9,700 days)** | **≈ 501 GB** | the note's **2.3 TB**, the brief's **1.5 TB** |
| **one year (366 days)** | **≈ 18.9 GB** | — |

**The note's own arithmetic is 4.6x high and the brief's 3.0x**, and the reason
is the one the note's formula cannot see: the field COMPRESSES. SST is smooth
and the quality grade is constant over large areas, so zstd gets 12.5x rather
than the 1.2x overhead the formula assumes — 0.457 bytes a valid pixel instead
of 4.8. **So one year is 19 GB, not 85, and 19 GB of parts plus a 19 GB store
fits a hosted runner with room to spare; the whole record is ~500 GB, which is
a box job but not a "contact us about terabytes" job.** That changes the phase-C
conversation and is the probe's main result.

What the probe also settled: the grade histogram over 18 frames is 0 →
1,865,625,546, 5 → 1,018,298,454 and absent → 32,076,000, i.e. **the DY
super-collation publishes grade 0 or grade 5 and nothing in between** — so the
"grade 2..5 stored with its grade" rule stores a constant 5 in practice and
the quality channel is, for this collection, a mask. That is worth knowing
before paying two bytes a pixel for it.

**Two cautions on this probe.** It ran on the code that refused a granule on
CMR's DECLARED size, so thirteen days of January 2020 are in
`inputs_not_read` rather than measured (see the fix above) — the per-frame
numbers are from eighteen real days and are what matter, and a re-probe is
dispatched. And 49.4 s of encoding a frame means **5.0 h for 366 frames**,
which is inside a hosted runner's six hours only just: size the year as TWO
six-month lanes, not one.

### Still in flight at the end of wave 4, and what each one settles

Four dispatches were queued and had not started when this wave stopped. The
account's concurrency ceiling is the reason and it is worth recording as
operational lore: **twenty concurrent jobs is the ceiling and it is SHARED
across every wave working the repository at once.** With four waves dispatching
together the queue reached 34 jobs behind 9 running, so a probe dispatched at
the wrong minute waits half an hour for a two-minute job. The wave-2 note
already says "dispatch the job you need FIRST, then fill the pool"; the
addition is that the pool is not yours.

| run | what it settles |
|---|---|
| #177 `check_credentials` | whether Earthdata still answers this account for LP DAAC, GES DISC and PO.DAAC — the definitive confirmation for the GES DISC 401 diagnosis above |
| #178 `irtb` probe 2015-01 | the MERGIR file's internal layout (written into plan.json as a whole inventory), whether an hourly file really holds two time steps, its lat/lon axes against the declared grid, and the compressed bytes a tile and a frame |
| #179 `xco2` probe 2019-06 | the OCO-2/OCO-3 Lite file's internal layout, likewise, plus rows a day and the NaN fraction per channel |
| #199 `sst_acspo02` probe 2020-01 | the same month with the declared-size refusal removed, i.e. all 31 days rather than 18 |

Each was dispatched AFTER the Earthdata session fix, which is the change #170
and #169 needed. `irtb` and `xco2` were probed once before it and failed in
the index with `HTTP 401 after 2 redirect(s)` from GES DISC; nothing else
about either adapter has changed.

### `swot`: the storage decision, with measured numbers

The probe (#168, 2024-01, twelve passes of cycle 010,
`ml/family1/probes/swot_2024-01.json`) read 409,912,719 bytes in 9.86 s
(41.6 MB/s) and STOPPED, as the brief asked. Every number below is measured
except the orbit geometry, which is the note's.

| what | measured |
|---|---|
| pixels a pass | **680,754** — every one of the twelve passes is exactly 9,866 x 69, so the swath geometry is fixed |
| `ssha_karin` present | 50.21 % of pixels (the nadir gap and land are the rest) |
| VALID (present, and not graded `bad`) | **51.52 %**, i.e. **350,751 pixels a pass** |
| `ssh_karin_qual` grades seen | good 4,209,008 · bad 3,960,040 — the product's top-byte summary is effectively binary here; no `suspect` or `degraded` pixel appeared in twelve passes |
| bytes a pass | 32.39 MB (CMR's declared size, confirmed) |
| variables a granule | 103; `time_tai` is present beside `time`, and `time`'s units are "seconds since 2000-01-01 00:00:00.0" |
| cycle 010 in full | **581 granules**, passes 1..584 with three absent, 2024-01-25T00:19 .. 2024-02-14T21:04, 18,807.9 MB |

**THE TWO OPTIONS, per year (584 passes a 20.86-day cycle = 10,225.6 passes):**

| | tier P (rows) | tier G (one sharded group a pass) |
|---|---|---|
| rows / valid pixels a year | 3,586,635,748 | the same pixels |
| bytes a valid pixel | 33 (27 + 2C) | ≈ 7.2 (2C x 1.2) |
| **bytes a year** | **118.4 GB** | **25.8 GB** |
| files a year | the store's nine arrays | ≈ 20,452 (a shard + an index a pass) plus 584 `tile_grid.json` + `shard_index.npy` |
| reads a cone | rows by (bin, position), as `slatrack` | 39 tile ranges a pass |

**Tier G is 4.58x cheaper in bytes and much worse in FILES**, and the probe
turned up a third consideration the note does not mention: a pass is **69
pixels wide**, so on the layout's 256 x 256 tiles a frame is 39 x 1 tiles and
**73 % of every tile is pad**. zstd compresses the pad to almost nothing, so
the byte column above is not wrong — but a reader still pays 39 range reads to
cross one swath, and the tile shape is simply a poor fit. If the G form is
chosen, the tile should be re-shaped (64 or 128 columns) rather than left at
256, which is a `family1/sharded.py` change and not an adapter setting.

Two more things the main session should weigh, both measured:
- **The note's arithmetic is about 40 % high.** It projects ~16e9 rows and
  ~0.6 TB for 55 cycles; 3.59e9 rows a year over the ~3.1 years of science
  orbit those 55 cycles are is **1.13e10 rows and ~372 GB**.
- **The channel bounds this adapter declared are too tight, and the probe
  said so instead of clipping**: 47,390 `sig0_karin` values (1.1 %) fell
  outside [-10, 40] dB and 33,675 `ssha_karin` values (0.8 %) outside
  +/- 3 m. Those are NaN in the probe's own frame and COUNTED
  (`out_of_bounds`), and nothing reached the writer out of bounds
  (`out_of_bounds_stored` 0). Whichever tier is chosen, the bounds want
  widening from a distribution rather than from a guess — which is exactly
  what the probe is for.

**`swh`'s ASSEMBLY IS PARKED FOR THE BOX.** 48.73 GB of parts plus a store of
about 54 GB (the ghcnd and icoads precedent is store ≈ parts × 1.1) is ~103 GB
on one disk, past the 86.4 GB a hosted runner leaves free, so it goes the way
ghcnd's and icoads' did. The exact command, for a *verified* Vast box with
≥ 150 GB of disk whose runner name is `<runner>`:

    node scripts/../<dispatch> family1-build.yml '{"store":"swh",
      "stage":"all","runner":"<runner>","start":"1991-01-01",
      "end":"2023-12-31",
      "extra_args":"--parts-from-hub --assemble streaming"}'

i.e. dispatch `family1-build.yml` with store `swh`, stage `all`, the box's
runner name, start 1991-01-01, end 2023-12-31 and
`extra_args=--parts-from-hub --assemble streaming`. Streaming because the
store is past the assembler's 50 M-row `auto` threshold by thirty times; the
box needs no credential of any kind for this store, since even the FETCH is
keyless. `HF_HUB_DISABLE_XET=1` is already set on the build step, which is
what let icoads' 52 GB publish at all.

## Notes — E-082 wave 4, the CREDENTIALED land stores (added 2026-09-18)

`lst05`, `snow05`, `refl05`, `fire`, `flux` and `static_fine`. Everything
here was measured on 2026-09-18, from this sandbox where the source needs no
account and on GitHub-hosted runners where it does.

- **rasterio's GDAL HAS NO HDF4 DRIVER, and the three MODIS fields are all
  HDF4.** Measured in the sandbox: 155 drivers, HDF5, HDF5Image and netCDF
  among them, and nothing that reads HDF-EOS2. `pyhdf` 0.11.7 publishes a
  manylinux wheel that BUNDLES the HDF4 library, so `pip install pyhdf` is the
  whole fix and it is in `family1-build.yml`'s install step beside `rasterio`.
  One trap in it: `_FillValue` is a PREDEFINED HDF attribute, so a plain
  `setattr` on an SDS is silently ignored and `SDS.setfillvalue()` is what
  writes it — which the smokes' synthetic granules need and the first run of
  them found.
- **A MODIS GRANULE'S URL CANNOT BE DERIVED FROM ITS DATE**, so CMR is not a
  convenience but the only honest way to name the file (contract rule 4 in its
  strongest form). `MOD11C1.A2015182.061.2021358223019.hdf` carries the
  PRODUCTION timestamp. NASA's Common Metadata Repository lists every granule
  of a collection with its https link and byte size, needs NO account, and
  pages 2,000 at a time behind the `CMR-Search-After` header. Measured, the
  whole MOD10C1 collection came back in **12.7 s and 39 MB of JSON**: 9,634
  granules. The other counts are 9,587 (MOD11C1 v061), 9,622 (MOD09CMG v061)
  and 8,815 (MYD11C1 v061, Aqua).
- **A CMR COLLECTION CAN LIST TWO GRANULES FOR ONE DAY.** A reprocessing that
  did not retire its predecessor shows up as two entries with different
  production timestamps. `_modis_cmg.parse_cmr` keeps the newest and counts
  the rest as `granules_superseded`; it never silently takes the first.
- **The Earthdata login through a `.netrc` works, and `requests` is what makes
  it safe.** `Session.rebuild_auth` looks the request's host up in the netrc on
  EVERY redirect hop and strips any Authorization header on a cross-host hop,
  so a netrc naming `urs.earthdata.nasa.gov` alone sends the password there and
  nowhere else. With `force_ipv4` (the wave-1 lesson: the runners resolve AAAA
  and have no IPv6 route) the lst05 and snow05 probes each downloaded a month
  of protected granules first time.
- **A STALLED EARTHDATA HANDSHAKE COST A RUN, AND A SINGLE TIMEOUT IS WHY.**
  Run #151 (the first refl05 probe) sat on one 300-second `requests` timeout
  three times over, because a single timeout covers connect AND read;
  `urs.earthdata.nasa.gov` had simply not answered the TCP handshake. Eight
  minutes of the job went on one granule's login and nothing was measured. The
  connect timeout is now 30 s separately from the 300 s read (a 540 MB granule
  legitimately takes minutes), and a granule download gets **at least six
  attempts whatever `--attempts` says** — a lane is hours long and losing the
  year to one refused handshake costs far more than fifteen minutes of backoff.
- **The install step is SHARED, and a bad version check in it takes every
  store down.** The line I added asked for `pyhdf.__version__`, which does not
  exist, so the step exited 1 before anything else ran and killed runs #140,
  #141 and #142 — and would have killed any other store dispatched in that
  window. `importlib.metadata.version("pyhdf")` is the version that exists.
- **THE PROBES, and what they did to the note's estimates.**

  | store | run | month | frames | bytes/frame | valid fraction | projection | the note | verdict |
  |---|---|---|---|---|---|---|---|---|
  | `snow05` | #149, 2 min | 2015-02 | 28 of 28 | 2,442,346 | 0.986 both channels | **23.60 GB** over 9,634 frames | 60 GB | **61 % under** |
  | `lst05` | #150, 5 min | 2015-07 | 31 of 31 | 15,357,302 | 0.2334 day, 0.2357 night, **1.0** both QC | **147.30 GB** over 9,587 frames | 350 GB | **58 % under** |

  `lst05`'s numbers also size its lanes: 46.4 MB a granule at 6.9 MB/s, and
  the ENCODE is 5.2 s of the 6.9 s a frame costs, so the lane length is set by
  CPU and not by the network — four years (1,460 frames) is about 2.8 h of
  fetch plus 22 GB of parts to push.
- **`lst05` keeps its quality bits EVERYWHERE, so every tile of every frame is
  stored.** MOD11C1's QC encodes "not produced because of cloud" versus "for
  another reason" — information about the ABSENCE — and its file specification
  says in as many words that the SDS has no fill value, so 0 is a real
  reading. The measured consequence is `valid_fraction` exactly 1.0 on both QC
  channels against 0.233 on the temperatures, and `tiles_stored` equal to
  frames × 435. `refl05` does the opposite for a reason in the product: its
  state word describes a RETRIEVAL and has no not-produced code, and its
  all-zero value is an ordinary reading (clear, shallow ocean, climatological
  aerosol), so both QA bytes are NaN exactly where all seven bands are fill and
  an all-fill tile costs nothing.
- **`refl05` carries C = 9 where `family1tf.tex` §4.3 plans C = 8.** The state
  QA is a uint16 bit field and float16 represents CONSECUTIVE integers exactly
  only to 2,048 (2,049, 4,097 and 65,535 all change), so one channel cannot
  hold the producer's quality word without losing bits. §5's rule — the
  source's own quality flag is kept and nothing is homogenised — decides it:
  the word is split into its two bytes and `state_qa_hi << 8 | state_qa_lo`
  reconstructs it bit for bit.
- **`lst05` stores CELSIUS, not kelvin, and the reason is measured.** float16
  spaces its values **0.25 K apart at 300 K** and **0.03 K apart near zero
  Celsius**, so in kelvin the store's own quantisation would be a quarter of
  MOD11C1's 1 K accuracy.
- **`snow05` keeps the producer's CLASS CODES, not just its percentages.** Both
  channels are 0..100 percent OR one of 107 lake ice, 111 night, 237 inland
  water, 239 ocean, 250 cloud-obscured water, 252 the Antarctica mask, 253 not
  mapped, 255 fill. The bounds are 0..253, so 255 becomes the layout's missing
  value and 254 — which the producer's key does not define — would be counted
  out of bounds rather than stored. The probe's own tally of the 2015-02 month:
  459,253,879 ocean, 68,250,748 Antarctica-mask, 23,757,474 not-mapped,
  8,694,086 night, 1,237,800 cloud-obscured-water, 444,619 lake-ice pixels.
- **FIRMS' ARCHIVE DOWNLOAD PAGE IS NOT SCRIPTABLE, AND ITS AREA API IS.**
  `firms.modaps.eosdis.nasa.gov/download/` (read 2026-09-18) puts a human in
  the loop by design: it answers a request by e-mailing a link ("Once the
  request has been processed, you will receive an email with instructions on
  how to download your data"). `/api/area/csv/<MAP_KEY>/<SOURCE>/world/
  <DAY_RANGE>/<DATE>` covers the whole record instead — DAY_RANGE 1..5, `world`
  = [-180,-90,180,90] — and `/api/data_availability/csv/<MAP_KEY>/all` says
  which dates each source holds, so the record's ends are read rather than
  assumed. A year is 73 requests a source against a documented limit of 5,000
  per ten minutes.
- **`MODIS_SP` CARRIES TERRA AND AQUA IN ONE CSV**, so a fetch keyed by
  satellite would have asked for every MODIS window twice and stored every
  MODIS detection twice. `fire`'s unit of fetching is a SOURCE PAIR (Standard
  Processing plus its Near-Real-Time tail); every day belongs to exactly one
  source, and a gap between SP's `max_date` and NRT's `min_date` is a refusal.
- **THE VIIRS CONFIDENCE IS THREE CLASSES AND NO PERCENTAGE.** MODIS publishes
  0..100; VIIRS publishes `l`, `n`, `h`. Placing them at 0, 50 and 100 would
  invent three numbers the producer never published, so `fire`'s `confidence`
  channel is NaN for a VIIRS row and the class is kept exactly in qc bits 3-4.
  A class that is not one of the three is a refusal.
- **A PLATFORM HASH MUST BE OF A CANONICAL NAME, NOT OF THE ARCHIVE'S OWN
  TEXT.** FIRMS writes `Terra` in one product and `N20` in another; hashing the
  raw string made one satellite two platforms and left the second with no entry
  in platforms.json, which the smoke caught as
  `probe_platforms_without_entry: 2`. Every spelling now resolves to one
  canonical platform and the spellings seen are counted. platforms.json is also
  where each platform's OWN `log2_fp` lives — a MODIS detection is 1 km
  (-4.798) and a VIIRS one 375 m (-6.214) — because a tier-P store has one
  store-level footprint and two instruments.
- **FLUXNET NEEDS NO LOGIN AT ALL, measured on all three hubs.** The plan
  allowed for `FLUXNET_USERNAME` / `FLUXNET_PASSWORD`; the Shuttle
  (github.com/fluxnet/shuttle, not on PyPI) is a library over three PUBLIC
  hubs and its AmeriFlux plugin posts the fixed literal
  `user_id: "fluxnetshuttle"`. Anonymously from this sandbox: AmeriFlux's
  `site_info_display/AmeriFlux` (844 sites, **407** with
  `grp_publish_fluxnet`, each with lat/lon/elev and IGBP) and its
  `amf_shuttle_data_files_and_manifest` POST (a `ftp.fluxdata.org` zip per
  site with its size and MD5 — `AMF_AR-Bal_FLUXNET_2012-2013_v1.3_r1.zip`
  downloaded whole, 22,077,723 bytes, MD5 matching the API's); ICOS' SPARQL
  endpoint (**352** FLUXNET archive products with lat, lon and IGBP); TERN's
  two published CSVs (**53** sites). 812 products in all, which is the ledger
  row's "700+ sites". So `flux` declares `credentials = ()`.
- **ICOS NEEDS A LICENCE COOKIE, AND WITHOUT IT SERVES HTML THAT LOOKS LIKE A
  DOWNLOAD.** `GET https://data.icos-cp.eu/licence_accept?ids=%5B%22<id>%22%5D`
  sets `CpLicenseAcceptedFor`, after which the object answers **206
  application/zip** with `PK\x03\x04` magic (measured: 29,093,732 bytes).
  Without the cookie BOTH that URL and `/objects/<id>` answer the licence page
  as `text/html` with HTTP 200 — a body an unguarded reader would hand to
  `zipfile` and get an empty site from. `flux` keeps a cookie jar and refuses a
  body that does not start with a zip signature.
- **FLUXNET TIMESTAMPS ARE LOCAL STANDARD TIME AND THE OFFSET IS IN THE
  ARCHIVE.** `TIMESTAMP_START` is `YYYYMMDDHHMM` local, no daylight saving.
  Every site zip carries a BADM file (`*_FLUXNET_BIF_*.csv`) whose
  `UTC_OFFSET` row gives the offset (AR-Bal reads -3); a site whose BADM has
  none is an ABSENCE that stops the pass, never a longitude guess that puts the
  tower in the wrong hour. `*_FLUXNET_BIFVARINFO_*` is NOT that file.

### `static_fine` — DESIGNED AND MEASURED, NOT BUILT, and exactly why

Its three sources were verified on 2026-09-18 and none of them needs an
account:

- **GEBCO_2026** (the 15-arc-second global terrain model) is at
  `https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/
  netcdf/GEBCO_2026.zip?download=1` — **4,252,016,343 bytes**, `application/
  zip`, range requests honoured, keyless. (The `bodc.ac.uk/data/open_download/
  gebco/gebco_2026/zip/` path the older releases used answers 404; CEDA is
  where 2025 and 2026 live, and CEDA serves a GitHub runner at 5.0 MB/s per the
  wave-2 note, so the pull is about 14 minutes.) Uncompressed the grid is
  43,200 × 86,400 int16 = 7.46 GB.
- **ESA WorldCover 2021 v200** is on `s3://esa-worldcover`, whose HTTP listing
  is keyless: **2,651 `*_Map.tif` tiles, 124.03 GB** (measured by paging the
  bucket). Each is a 3° × 3° tile at 36,000 × 36,000.
- **GSHHG** (for `dist_coast`) is at
  `https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-bin-2.3.7.zip`, HTTP 200,
  `application/zip`, keyless. (NOAA's `ngdc.noaa.gov/mgg/shorelines/data/
  gshhg/latest/` path answers 404.) `scipy` is now in the workflow's install
  step for the nearest-shoreline search.

**TWO THINGS STOP IT BEING BUILT IN THIS WAVE, and both are findings rather
than excuses.**

1. **The sharded tier-G layout carries ONE `channels` and ONE `dtype` PER
   STORE, and `static_fine` needs three of each.** `GridAdapter.specs()` hands
   `sharded.make_spec` the ADAPTER's `channels` and `dtype` for every group, and
   `stage_assemble_grid` writes `ad.schema()`, `ad.C` and `ad.dtype` as
   store.json's own top-level declaration. That is right for every store built
   so far, whose groups differ only in GRID (seaice_asi's two hemispheres,
   lossyear's tiles). `static_fine` is the first store whose groups differ in
   WHAT THEY MEASURE: elevation in metres (float16, C = 1) at 15″, eleven
   WorldCover class fractions (C = 11) at 0.05°, and distance to coast (C = 1)
   at 0.05°. A store.json whose `C` and `channels` describe one of the three
   and are published as the store's own is not something to ship quietly. The
   minimal change is small and belongs to whoever owns
   `ml/build_family1_stores.py`: let `specs()` supply per-group `channels` and
   `dtype` (`check_sharded` ALREADY reads the bounds and the dtype from each
   group's own `tile_grid.json`, so the store's self-check needs nothing), have
   `stage_probe_grid` take its bounds and channel names from the group's spec
   rather than from `ad.bounds()` / `ad.channel_names`, and write store.json's
   channel table PER GROUP instead of once at the top.
2. **The WorldCover half is a multi-lane build in its own right, and the
   measurement is what says so.** 124 GB of 10 m tiles to download and, per
   tile, eleven masked reductions over 1.3 × 10⁹ pixels to get the class
   fractions of its 3,600 0.05° cells — about 1.4 × 10¹⁰ element operations a
   tile, i.e. of the order of ten seconds each even done band by band, so tens
   of hours over the 2,651 tiles. That is the "heavy compute: lanes by 3° tile"
   the plan names, and it wants its own wave with the per-group channels
   question settled first — not a corner of this one.

Until then the store is designed and nothing is published under its name,
which is the honest state: a family with a store missing and a note saying so
is better than a store whose store.json misdescribes two thirds of itself.

## E-082 wave 3 — the SCENE CATALOGUES, tier T (added 2026-09-18)

*Eight stores that hold no pixels: one row per satellite scene, with the
second it was taken, its footprint's centre and area, how cloudy it was, how
high the sun stood, which instrument took it and which processing baseline
produced it. The pixels — between 0.01 and 3 petabytes a year each — stay at
the producer, and a sidecar `assets.parquet` maps each row's hashed scene
identifier back to the producer's own identifier and download URLs. The
design note is `ml/paper/notes/family1tf.tex`, "The image catalogue, tier T";
the shared machinery is `ml/family1/adapters/_stac.py`. **None of the eight
needs an account**: every listing endpoint answers anonymously.*

| store | family | run(s) | N | stored bytes | record | verified | notes |
|---|---|---|---|---|---|---|---|
| `cat_nisar` | 1.0.tf (tier T) | built and published **from the sandbox** (`--stage all`, 2025-10-01..2026-09-30) because the hosted pool was saturated by another wave | 138,804 | 8,907,651 (8.9 MB; `assets.parquet` 3.8 MB) | 2025-10 → 2026-09 (bins 3198–3266) | 2026-09-18 | NISAR L2 GCOV, ASF; public; schema 2. **The record starts 2025-10, not the ledger's 2026-06**: 1,719 frames in 2025-10 under the BETA collection, none in 2025-09. Three sampled rows were re-read against ASF and their time, footprint centre, sensor code and asset URL all match the producer's own record |
| `cat_viirs` | 1.0.tf (tier T) | the sandbox, `--stage all` 2012-01-01..2026-09-30; the first attempt died on its last year on a CMR-Hits inconsistency (below) and the re-run skipped the fourteen finished years | 2,341,071 | 128,806,565 (128.8 MB; `assets.parquet` 42.2 MB) | 2012-01 → 2026-09 (bins 2195–3266) | 2026-09-18 | VIIRS I-band L1B, LAADS; public; schema 2. **THREE satellites, not the ledger's two**: VJ202IMG (NOAA-21) has been flying since 2023-02 and is catalogued with its own sensor code. Three of the five channels are NaN for every row — an L1B radiance granule publishes no cloud fraction, no valid fraction and no mean angle — and that is the honest answer rather than an invented one |
| `cat_olci` | 1.0.tf (tier T) | the sandbox, `--stage all` 2016-04-01..2026-09-30, 11 years in 43 min | 1,591,637 | 126,776,002 (126.8 MB; `assets.parquet` 64.7 MB) | 2016-04 → 2026-09 (bins 2506–3266) | 2026-09-18 | Sentinel-3 OLCI L2 WFR, CDSE; public; schema 2; **C = 6** — the only catalogue with a sixth channel, `coastal`. **From 2026 every overpass is published twice**, once NR and once NT, and the store keeps ONE row per overpass (measured on 2026-09: 15,033 products → 7,766 rows). Every WFR granule is catalogued, not only the coastal ones, because the GSHHG ±100 km band static the note's "coastal band" refers to is not built yet; `coastal` is a coarse stand-in computed from the repository's own `data/family7_sphere.json`, and store.json says so |
| `cat_landsat` | 1.0.tf (tier T) | built and published **from the sandbox** (`--stage all`, 1982-08-01..2026-09-30; 16,132 day-windows, 127,084 USGS STAC pages, 63 request retries, 2.85 h of fetch) — the hosted pool was saturated | 10,315,132 | 544,580,767 (544.6 MB; `assets.parquet` 162.9 MB, 45 row groups) | 1982-08 → 2026-09 (bins 46–3265; 3,131 of 3,220 bins live) | 2026-09-18 | Landsat Collection 2 Level-2, USGS STAC (`landsatlook.usgs.gov`); public; schema 2. The catalogue walked exactly the producer's count (`producer_count` = rows = 10,315,132). Five cameras carry their own sensor codes (Landsat 4, 5, 7, 8, 9); `qc` is the USGS processing level and tier (L2SP/L2SR × T1/T2/RT). Per year: 1,177 scenes in 1982, ~120 k a year through the 1990s, ~380 k a year 2014–2021, ~600 k in 2022–2023 (Landsat 9 fully on), 337,608 in 2026 to date. `angle` is filled for every row (mean 45.9°, range 20–76°) |
| `cat_s1` | 1.0.tf (tier T) | built and published **from the sandbox** (`--stage all`, 2014-10-01..2026-09-30; 9,978 CDSE OData pages, 2.54 h of fetch) — same reason | 6,203,597 | 464,222,496 (464.2 MB; `assets.parquet` 234.7 MB, 13 row groups) | 2014-10 → 2026-09 (bins 2392–3266, all 875 live) | 2026-09-18 | Sentinel-1 IW GRDH **and** SLC, CDSE (`catalogue.dataspace.copernicus.eu`); public; schema 2. Walked exactly the producer's count. Sensor codes name satellite × product (S1A/S1B/S1C/S1D × GRDH/SLC); the polarisation and processing baseline are kept under `counts.qc_unlisted` (VV+VH dominates at 3.65 M strips). The per-year curve is the constellation's own history: 28,671 strips in the 2014 quarter, 734 k in 2020, the fall to ~396 k in 2022–2024 after Sentinel-1B failed (2021-12), back to 628 k in 2025 with Sentinel-1C. Three sampled rows were re-read against CDSE and match |
| `cat_ecostress` | 1.0.tf (tier T) | hosted year lanes #200 (2018-07-01..2019-12-31) and #201–#207 (one per year 2020–2026), each `--push-parts`; assembly **#267** (hosted, `stage=all --parts-from-hub --assemble streaming` 2018-01-01..2026-09-30, **3 min**: 54 s to pull the 9 years, 77 s to assemble streaming at 2.89 GB peak RSS, 40 s to publish 11 files, 3.5 s to check) | 19,525,578 | 1,009,092,509 (1.01 GB, 11 files; `assets.parquet` 286.6 MB) | 2018-07 → 2026-09 (bins 2667–3266, 571 of 600 live; 9 years with rows) | 2026-09-19 | ECOSTRESS L2T LSTE 70 m thermal tiles from the ISS, LP DAAC via CMR; public; schema 2, C = 5. **Both processing versions are catalogued and `qc` says which** — v002 15,617,001 rows, v003 3,908,577; they do not overlap in time today because v003 is working backwards from the present, so an overpass that appears twice is a reprocessing rather than a duplicate and a consumer selects on `qc`. The walk accounts for the producer exactly: `producer_count` 19,534,789 − `granule_starts_outside_window` 9,211 = N, over 6,028 windows and 56,414 pages with 4 retries and 12.5 h of fetch across the lanes. 876 rows MORE than the whole-archive count taken from the producer's per-year numbers on 2026-09-18 (19,524,702) — the mission is live and the archive grew between the count and the build. `cloud`, `valid` and `angle` are NaN for every row: ECOSTRESS ships a cloud RASTER rather than a scene cloud fraction, and publishes no valid fraction and no mean angle. One sensor code, ISS/ECOSTRESS. The footprint is a rectangle, not a polygon, and the store has no cadence at all — the ISS orbit drifts through the local day, which is the reason to catalogue it |
| `cat_hls` | 1.0.tf (tier T) | hosted year lanes #219 (2013-04-01..2015-12-31), #220 (2016–2017), #240 (2019), #249 (2020), #251 (2021), #252 (2018), #226 (2023), #227 (2024) from wave 3, plus **#261 (2022), #262 (2025), #263 (2026)** re-run on 2026-09-19 after the three failures below; assembly **#274** (hosted, `stage=all --parts-from-hub --assemble streaming` 2013-01-01..2026-09-30, **9 min**: 233 s to pull 14 years, 1.40 GB of arrays assembled streaming, publish and check) | 37,951,341 | 1,941,103,957 (1.94 GB, 11 files; `assets.parquet` 536.9 MB) | 2013-04 → 2026-09 (bins 2284–3266, all 983 live; 14 years with rows) | 2026-09-19 | NASA's harmonised Landsat and Sentinel-2 30 m tiles (HLS v2.0), LP DAAC via CMR, listed anonymously; public; schema 2, C = 5. S30 21,934,730 rows + L30 16,016,611 = N, and `producer_count` 37,951,343 − 2 `granule_starts_outside_window` = N exactly, over 9,862 windows and 90,150 pages with 5 retries and 15.7 h of fetch across the eleven lanes. Five sensor codes: S2A, S2B, S2C, Landsat-8, Landsat-9. **The only one of the eight catalogues whose producer publishes a valid fraction** — `valid` is SPATIAL_COVERAGE, `cloud` is CLOUD_COVERAGE and `angle` is MEAN_SUN_ZENITH_ANGLE. 204 rows have a `log2_area` outside its channel bounds and are NaN, never clipped. **The two wave-3 adapter fixes are measured here**: 30 granules have no download url at all and are kept as rows with an empty asset entry (`granules_without_asset_url`, `c7cab20`) and 108 asset urls are published under a data type AND a browse/metadata type and are excluded from the template (`asset_urls_also_browse_or_metadata`, `bd7f666`). Either one, before those commits, killed a whole year lane |
| `cat_s2` | 1.0.tf (tier T) | hosted year lanes #245 (2015-07-01..2016-12-31), #246 (2017), #248 (2019), #254 (2020) from wave 3, plus **#268 (2018), #269 (2021), #272 (2022), #276 (2023), #277 (2024), #279 (2025), #283 (2026-01-01..2026-09-15)** on 2026-09-19 — the six lanes the concurrency group had cancelled, plus 2018, whose #247 died on a CDSE HTTP 429; assembly **#285** (hosted, `stage=all --parts-from-hub --assemble streaming` 2015-01-01..2026-09-15, 11 min) | 41,197,086 | 2,906,157,640 (2.91 GB, 11 files; `assets.parquet` 1.38 GB — the largest sidecar of the eight) | 2015-01 → **2026-09-15** (bins 2447–3265, 816 of 819 live; 12 years with rows) | 2026-09-19 | Sentinel-2 L2A tiles from the CDSE OData catalogue; public; schema 2, C = 5. `producer_count` = `rows_kept` = N exactly, over 4,095 windows and 44,282 pages in 21.9 h of fetch across the eleven lanes. **Hour windows, and 2,197 of them had to be halved again**: a day cannot be paged inside CDSE's `$skip` cap of 10,000, and a window whose own count exceeds the reach is split until it fits. **The record stops 2026-09-15, not 2026-09-30**, and that is a deliberate cut: the lane that ran to 09-30 (#281) died on TODAY's day window with `page 10 returned 717 of 721 product(s) with 4 still to come — a truncated page`, the OData form of the CMR-Hits race, because CDSE was ingesting into the window while the walk paged it. A top-up lane extends it once those days settle. **Four sensor codes** — S2A, S2B, S2C and S2D — and the ledger's 'S2A retired 2026-03' stays wrong: the index re-measures it on every build. 3,163 rows carry no `cloudCover` attribute, 701 have a `log2_area` outside bounds and are NaN, and only 15,080 rows (0.04 %) fall outside the qc table. `asset_sets` is empty for the same reason as `cat_s1`: a CDSE product is one `Products({id})/$value` URL, so `assets.parquet` carries `base_url` and no template |

**What each store is, in one line.** `cat_landsat` every Landsat Collection 2
Level-2 scene since 1982 (all five cameras) · `cat_hls` NASA's harmonised
Landsat and Sentinel-2 30 m tiles · `cat_s2` every Sentinel-2 L2A tile ·
`cat_s1` every Sentinel-1 IW strip, ground-range AND single-look complex ·
`cat_nisar` every NISAR L-band covariance frame · `cat_olci` every Sentinel-3
ocean-colour granule · `cat_viirs` every VIIRS 375 m six-minute swath ·
`cat_ecostress` every 70 m thermal tile from the Space Station.

**The probes.** One real month per store, through the real adapter, with the
producer's own count for that month beside it
(`ml/family1/probes/cat_*.json`):

| store | month | rows | the producer's count | rate | MB fetched | B/row |
|---|---|---|---|---|---|---|
| `cat_hls` | 2024-06 | 413,796 | 413,796 | 1,169/s | 5,540 | 13,389 |
| `cat_landsat` | 2024-06 | 42,339 | 42,339 | 859/s | 173 | 4,087 |
| `cat_s2` | 2024-06 | 375,674 | 375,674 | 848/s | 1,647 | 4,384 |
| `cat_s1` | 2024-06 | 30,326 | 30,326 | 671/s | 142 | 4,677 |
| `cat_ecostress` | 2024-06 | 126,788 | 126,788 | 983/s | 2,422 | 19,105 |
| `cat_viirs` | 2024-06 | 20,796 | 20,970 | 1,359/s | 51 | 2,465 |
| `cat_nisar` | 2026-08 | 29,992 | 30,229 | 171/s | 505 | 16,834 |
| `cat_olci` | 2026-09 | 7,766 | 15,033 | 528/s | 65 | 8,334 |

Six of the eight walked exactly the producer's number. `cat_viirs`' 174 and
`cat_nisar`' 237 are `granule_starts_outside_window` (below);
`cat_olci`'s gap is the NR/NT de-duplication. `out_of_bounds_stored` is 0
everywhere and every store's `distinct_platforms` equals its rows — **no hash
collision in 1.05 million probed scenes**.

**Whole-archive sizes, from the producers' own per-year counts, measured
2026-09-18:** `cat_s2` 41,232,005 · `cat_hls` 37,936,283 (S30 21,924,986 +
L30 16,011,297) · `cat_ecostress` 19,524,702 (v002 15,616,410 + v003
3,908,292) · `cat_landsat` 10,315,132 · `cat_s1` 6,203,352 (GRDH 3,184,957 +
SLC 3,018,395) · `cat_viirs` 2,341,106 · `cat_olci` ≈ 1,608,336 ·
`cat_nisar` 138,762. About **1.19 × 10⁸ rows** and, at 37–39 stored bytes a
row, roughly 4.5 GB of arrays plus the sidecars.

### What the producers do that a careless adapter would get wrong

Each of these was MEASURED, and four of them contradict the ledger.

- **Sentinel-2A is not retired.** The note says "S2B + S2C (S2A retired
  2026-03)". By acquisition time: S2A produced 24,411 L2A products in 2026-02,
  **86,193 in 2026-04** and **48,637 in 2026-09**. `cat_s2` carries three
  satellites and its index re-measures the claim on every build.
- **Every Sentinel-1 GRDH scene is in CDSE twice.** `IW_GRDH_1S` and
  `IW_GRDH_1S-COG` hold the same scenes — 529/529 on 2024-06-01, 922/922 on
  2026-09-01, 1,110/1,110 on 2019-06-01 — so `contains(Name,'IW_GRDH')`
  returns exactly double. The filter is on `productType` and the index counts
  the mirror beside the store.
- **From 2026 every OLCI overpass is published twice**, once near-real-time
  and once consolidated: 846 products on 2026-09-01 are 423 NT + 423 NR, and
  NR is 0 for every year 2016–2025. The store keeps the NT where both exist
  and the NR where NT does not exist yet, so the newest weeks are not lost.
- **NISAR GCOV starts 2025-10**, not 2026-06.
- **VIIRS I-band L1B has three satellites**, not two.
- **Both CMR and ASF match on OVERLAP.** `temporal=a,b` returns every granule
  whose own time RANGE overlaps `[a, b]`, both ends inclusive, so a six-minute
  VIIRS granule or a 30-second NISAR frame that spans a window boundary is
  returned for both windows. A granule belongs to the window its START falls
  in; without that rule 169 of 20,796 VIIRS granules and 237 of 30,229 NISAR
  frames came back twice. (ASF's inclusive end is separately visible in its own
  counts: 6,675 + 23,318 over two halves of August 2026 against 29,992 for the
  whole month.)
- **CMR's own `CMR-Hits` header can over-count.** On 2026-08-02 the VNP02IMG
  window answers `CMR-Hits: 243` and serves 242 distinct granules — with any
  sort key, with none, and at `page_size=2000` where the window is ONE response
  with no cursor and no paging at all. A window may be up to 8 short of its own
  header; the shortfall is counted by name, a walk that returns MORE is still a
  refusal, and the three structural truncation checks are untouched.
- **A short LAST page that still offers a cursor is normal.** landsatlook ends
  a window with 66 of 100 items and a `next` link. What is truncation is a
  short page FOLLOWED by more items, and that is what is refused.
- **ASF rate-limits at 250 requests a minute and says so in a 429.** Eight
  workers at two requests a window were issuing about 800 a minute and killed
  a whole-archive fetch 34 minutes in. `_stac.RATE_LIMITS` holds that host at
  240 through a token bucket every thread shares; no other producer throttled
  us (CMR served 2,000-granule pages to eight workers for an hour, CDSE's
  OData the same, landsatlook answered 859 items a second).

### Three decisions the note did not anticipate

- **`area` is stored as `log2(km²)`, not km².** A store's `values` is
  **float16**, whose largest finite number is 65,504; a VIIRS six-minute
  granule is about 7 × 10⁶ km² and a Sentinel-3 OLCI granule 1.5 × 10⁶, so
  both round to **infinity** on the way in — which is exactly how `cat_s1`'s
  and `cat_olci`'s first smokes failed (`values.npy holds an infinity`). The
  price is float16's ten-bit mantissa: the round trip is 0.007 % at a
  Sentinel-2 tile's 12,364 km², 0.31 % at a VIIRS granule's 7 × 10⁶, never
  worse than about 1.2 %. The note's "area" is `2 ** log2_area`.
- **A footprint that encloses a pole is not the whole planet.** The
  spherical-excess formula returns one of the two regions a ring divides the
  sphere into, depending on the winding, so 1,239 of 20,796 VIIRS granules
  (6 %) came back at the area of the Earth. `sphere_centre_area` takes the
  smaller of the two.
- **The CDSE STAC endpoint serves no Sentinel collection at all.**
  `catalogue.dataspace.copernicus.eu/stac/collections` answers 200 with ten
  collections — five Contributing Missions groups and five CLMS burnt-area
  products — and not one Sentinel among them. For `cat_s2`, `cat_s1` and
  `cat_olci` the OData API is not a fallback, it is the only catalogue CDSE
  offers.

### The sidecar, and how it travels with a lane's parts

`assets.parquet` has one row per catalogue row: `platform` (the row's hash),
`stac_id`, `collection`, `base_url` (the directory the scene's files live in)
and `asset_set` (the file names with `{id}` standing for the identifier, so a
URL is `base_url + "/" + name.replace("{id}", stac_id)`). The names are the
PRODUCER's own — nothing is derived from a pattern — and parquet's dictionary
encoding collapses the column to one entry per distinct set (two for the whole
Landsat archive: TM/ETM+ and OLI/TIRS, verified on 360 scenes across six eras).

A fetch writes its asset rows beside that year's `.npz` parts as
`assets-NNNNN.npy`, because `family10_parts_hub` already carries `.npy` files
out of a year directory to the Hub and back and neither assembler reads them —
so a lane's sidecar travels with its parts for free and a
`--parts-from-hub` assembly finds it. The bytes inside are a zstd-compressed
Arrow IPC stream. `extra_files` concatenates them one row group at a time
(Sentinel-2's table is 41 million rows whose `stac_id` and `base_url` are
unique per row, so building it in memory first would need about 8 GB), checks
that the `platform` column is unique across the whole table, and refuses
unless it has exactly as many rows as the store.

### Why three stores were built in the sandbox

The hosted pool was saturated by another wave of this experiment — 30 jobs in
flight and 49 queued, the oldest of them 3.3-hour `oc4k` lanes — so the three
small catalogues were built and published from the sandbox instead
(`HF_TOKEN` from `/home/claude/.hf_token`, `--stage all`), which is the
`bgcargo` precedent from wave 1. Each restore-verified every file and its Hub
`store.json` agrees with the local sha256 block.

### TWO ADJACENT LANES WRITE THE SAME YEAR, and the last one to push wins

The wave's most expensive finding, and it is about the FRAMEWORK rather than
about any source. `--start` / `--end` choose BINS, and the bin that straddles
a lane boundary is inside BOTH lanes' windows. That bin belongs to the year of
its FIRST day, so:

- the earlier lane writes year *N* complete (73 or 74 bins), and
- the later lane writes a ONE-BIN copy of the same year *N*,

both under `partials/<family>/<store>/<N>/`. `family10_parts_hub.push_many`
skips a year only when its `done.json` already matches what it is about to
upload, and otherwise OVERWRITES — so whichever lane finishes last decides
which of the two versions the assembly will read. Nothing warns.

**Measured on snow05.** Lane #159 (2000-2008) and lane #160 (2009-2017) both
contain bin 1972 (2008-12-30 .. 2009-01-03), whose first day is in 2008.
#160 reached its 2008 first and pushed a one-bin year at 07:54; #159 pushed
its 74-bin 2008 two hours later, downloaded the file back to check it, and got
**#160's** copy:

    RESTORE MISMATCH partials/family1_tf/snow05/2008/terra__shard_index.npy:
    uploaded 4e861afe…, downloaded 6b169f4b… — the push is not trustworthy

That looked like a Hub flake for ten minutes. It is not: the restore check
caught a genuine concurrent overwrite of one path by two jobs, which is
exactly what it is for. Without it the store would have assembled with 2008
holding five frames instead of 366, and the only sign would have been a frame
count nobody had a reason to distrust.

**The rule that removes it: a lane boundary must fall on a BIN boundary that
is also a YEAR boundary** — a lane starts on the first day *d* of a year with
`(d − 1982-01-01) mod 5 == 0`, and the previous lane ends on *d − 1*. Then
every year belongs to exactly one lane and no two lanes ever write the same
path. For the four-year lanes this wave used:

| lane starts | (bin) | previous lane ends |
|---|---|---|
| 2000-01-02 | 1315 | — |
| 2004-01-01 | 1607 | 2003-12-31 |
| 2008-01-05 | 1900 | 2008-01-04 |
| 2012-01-04 | 2192 | 2012-01-03 |
| 2016-01-03 | 2484 | 2016-01-02 |
| 2020-01-02 | 2776 | 2020-01-01 |
| 2024-01-01 | 3068 | 2023-12-31 |

(The first bin-start of a year is always inside that year, because a bin is
five days — so this always exists and never pulls the previous year in.)

**How to check a store's parked parts before assembling one**, which is now
the thing to do after any multi-lane tier-G fetch:
`/tmp/.../scratchpad/years.py <store>` reads every parked
`<year>/counts.json` off the Hub and prints its bin count beside the number
the family's calendar says that year holds. A year short of its calendar is
either a lane collision or the record's own end — the record's ends are
legitimately short (snow05's 2000 holds 63 bins of 73 because MOD10C1 starts
on 2000-02-24, and 2026 holds 51 because the record ends in September), and
everything between them should be exact.

## Notes — E-082 wave 5, the hosted assemblies and what the live tail of a record does to a lane (added 2026-09-19)

*Four stores landed from hosted runners on 2026-09-19/20: the three tier-T
catalogues `cat_ecostress` (#267), `cat_hls` (#274) and `cat_s2` (#285) —
98.7 million catalogue rows between them — and `fire` (#291), 631.8 million
active-fire detections over 27 years, the largest 1.0.tf store after `ghcnd`
and `tide`. Their rows are in the tables above. What follows is what the wave measured
about OPERATING the builder, which is the part a later wave pays for twice
if it is not written down.*

### The assembly form is `stage=all`, never `stage=assemble,…`

A dispatch of `{"stage":"assemble,publish,check","extra_args":"--parts-from-hub …"}`
dies in one minute:

    stage 'assemble' needs 'fetch' first (E-079 §4: stage order is fixed) —
    …/ml/cache/family1_tf/cat_ecostress/fetch.done is missing

`--parts-from-hub` is read INSIDE the fetch stage — it is the branch of
`build_family10_stores.py::stage_fetch` that calls `family10_parts_hub.pull`
— so a stage list without `fetch` never pulls the parts and never writes the
marker `assemble` demands. Every assembly that has ever worked used
`stage=all`: #41 `igra`, #75 `wod`, #82 `tide`, #84 `bgcargo`, #88
`oceansites`, #89 `ghcnd`, #93 `icoads`, #127 `chirps05`, #161 `snow05`, and
now #267, #274 and #285. And **`start` and `end` must both be passed**: the
workflow defaults `end` to `2024-12-31` and `ctx.years` is
`range(start.year, end.year + 1)`, so an assembly dispatched without them
silently drops 2025 and 2026 from the store.

A hosted assembly is cheap. `cat_ecostress`, 19.5 M rows: 54 s to pull nine
years of parts, 77 s to assemble streaming at 2.89 GB peak RSS, 40 s to
publish eleven files, 3.5 s to check — three minutes end to end. `cat_hls`,
38.0 M rows: nine minutes. `cat_s2`, 41.2 M rows and a 1.38 GB sidecar:
eleven minutes. Nothing about these needed a box.

### THE LIVE TAIL OF A RECORD REFUSES A LANE, and both dialects say so

A catalogue that is still being ingested cannot be walked to today. Measured
twice, on two different producers, with the same shape:

    cat_hls  S30 2026-09-14T00:  CMR-Hits changed from 7920 to 7921 on page 8
                                 — the catalogue moved under the walk        (#243)
    cat_s2   l2a 2026-09-19T00:00: page 10 returned 717 of 721 product(s)
                                 with 4 still to come — a truncated page     (#281)

Both guards are right: a listing whose total moves under a paged walk has not
been walked completely, and neither adapter may pretend otherwise. The rule
that follows is about the WINDOW, not the code: **a lane's `end` must be
several days behind the producer's ingest front.** `cat_hls`' 2026 lane
re-ran to 2026-09-30 cleanly at 11:53Z (#263) because those days had settled
overnight; `cat_s2`' 2026 lane died on TODAY and was re-cut to
`end 2026-09-15` (#283), which is why that store's record stops on 09-15
where the other two reach 09-30. A top-up lane extends it once the days
settle; what must not happen is a lane that cannot finish.

### One key, one lane: FIRMS counts transactions per MAP_KEY

`fire` fetches through `firms.modaps.eosdis.nasa.gov/api/area/csv/<MAP_KEY>/…`
and the transaction limit is per KEY, not per job. Two lanes dispatched
together (#264, #265) both reported
`the FIRMS transaction limit is exceeded (2/8); … waiting 130s` within the
same minute; the adapter gives up at 8. A SOLO lane reaches 5/8 on its own
(#233, 2000–2006). So `fire` lanes run **one at a time**, and #265 was
cancelled at 12:32Z to let #264 have the whole budget — which it then used,
finishing 2007–2013 in 5 h 00 m.

### A `fire` lane is four years from 2014 on, because the sources multiply

FIRMS gains a source part-way through the record — `MODIS_SP`, then
`VIIRS_SNPP_SP` (2012 →), then `VIIRS_NOAA20_SP` (2018 →) — so a year costs
about 1.5× once the third one appears, and the seven-year lane that worked
for 2000–2006 does not fit 2014–2020. #282 was at three of seven years in
2 h 18 m with its own ETA at 22:29Z against a hard 23:05Z deadline, and was
cancelled rather than allowed to run into it. **A lane pushes its parts only
after the whole fetch stage finishes**, so an overrun loses every year in the
lane, not the tail: the cancellation cost three fetched years and the
alternative risked seven.

**The lane length is set by the SOURCE COUNT, not the year count.** FIRMS
windows are per (source, five days), so a year costs a window per source and
the whole record was fetched in six lanes of shrinking span:

| lane | sources | windows a year | wall clock | run |
|---|---|---|---|---|
| 2000–2006 | MODIS | 73 | 3 h 03 m for 7 years | #233 |
| 2007–2013 | + SNPP from 2012 | 73 → 146 | 5 h 00 m for 7 years | #264 |
| 2014–2017 | MODIS + SNPP | 146 | 4 h 01 m for 4 years | #284 |
| 2018–2020 | + NOAA-20 | 201 | 4 h 17 m for 3 years | #286 |
| 2021–2023 | three | 219 | 4 h 29 m for 3 years | #288 |
| 2024–2026 | **+ NOAA-21** | 292 | 5 h 29 m for 2.75 years | #289 |

The fourth source appears in 2024 and was not anticipated by the split; #289
was given an explicit tripwire before it ran (if 2024 were not complete by
10:35Z the lane could not fit six hours and would be cancelled and halved),
it cleared it at 09:55Z, and it finished with 27 minutes to spare. Write the
deadline down before the evidence closes — ml/CLAUDE.md §4.13.

### `flux` cannot be built: 27 towers publish no BADM file at all

`flux`'s fetch is ONE stream over all 806 site archives, and its resumable
unit is therefore the whole pass. Two runs (#266, #275) each spent ~90
minutes and ~90 GB of download, read 779 of the 806 towers and 108,292,080
rows, and kept nothing. The 27 that could not be read all fail the same way:

    CA-ER1: no BADM (BIF) file in the zip
            (['AMF_CA-ER1_FLUXNET_FULLSET_HH_2015-2021_4-7.csv', …])

and it is **not** the naming regex `2850b90` fixed for the data file. Measured
against AmeriFlux directly, anonymously, on 2026-09-19 — `POST
amfcdn.lbl.gov/api/v2/amf_shuttle_data_files_and_manifest` with
`data_product: "FLUXNET"`, `data_variant: "FULLSET"`:

| site | files in the archive | of `file_type` BADM |
|---|---|---|
| `US-Ha1` (ONEFlux v1.3) | 18 | **6**, incl. `AMF_US-Ha1_FLUXNET_BIF_1991-2025_v1.3_r1.csv` |
| `CA-ER1` (FULLSET 4-7) | 12 | **0** |
| `US-Wi0` (FULLSET 5-7) | 12 | **0** |

The same three sites under `data_product: "FLUXNET2015"` also ship twelve
files with no BADM, and the enum accepts only those two products. Nor is the
offset in the site metadata: `api/v2/site_info_display/AmeriFlux` carries
`grp_location` (lat / lon / elev) and no offset, `api/v1/site_display/AmeriFlux`
carries `GRP_LOCATION`, `IGBP` and `TOWER_BEGAN`/`TOWER_END` and no offset,
and `api/v{1,2}/badm/AmeriFlux` is 404. The adapter needs `UTC_OFFSET`
because FLUXNET timestamps are LOCAL STANDARD TIME and it refuses — rightly —
to derive one from the longitude. One of the 27, `ID-PaD`, fails differently:
it HAS a BADM file and that file has no `UTC_OFFSET` row.

**And `--allow-missing-years` cannot rescue it, because a one-stream store
never marks its years.** #275 passed the absence check and still stopped:

    ::warning::flux: 27 input(s) could not be read (CA-ER1, CA-NS1, …) —
    --allow-missing-years says that is deliberate; the degrade goes into store.json
    --push-parts: flux 1991 is not marked done locally — refusing to push a partial year

`build_family10_stores.py`'s one-stream fetch branch decides on `ctx.absent`
alone and never consults `allow_missing_years`: with any absence it FLUSHES
every writer and marks nothing, where the clean path CLOSES them and writes
`parts/<year>.done`. So for a one-stream adapter (`flux`, `socat`) the flag
stops the stage exiting non-zero and changes nothing else — no year is
marked, `--push-parts` refuses the first one, and an `assemble` would find
nothing either. That is not what the flag's own refusal message promises, and
it is the second thing to fix here. Neither fix was made in this wave: the
first needs a second source for those sites' UTC offsets, and the second is
framework code that four waves and every family-10 store run through.

### `xco2` and `irtb` are blocked on ONE Earthdata click

Both probes fail on the same host with the same status, and did so again when
re-run on 2026-09-19 (#259, #260):

    REFUSING irtb: the first file GPM_MERGIR.1:merg_2015010100_4km-pixel.nc4 could not be read:
    https://data.gesdisc.earthdata.nasa.gov/data/MERGED_IR/GPM_MERGIR.1/2015/001/merg_2015010100_4km-pixel.nc4:
    HTTP 401 after 2 redirect(s) — Earthdata Login refused this account for this archive.

The 401 comes from URS's own `/oauth/authorize` for client
`e2WVk8Pw6weeLUKZYOxvTQ` — the GES DISC application — which is the signature
of an account that has not approved that application, not of a bad password.
LP DAAC and PO.DAAC are genuinely fine on the same credentials. **There is no
`probe.json` for either store**, so every size in the notes for `xco2` and
`irtb` is still the ledger's estimate. The credential check said `ges_disc ok`
throughout, on a 206 whose `final_url` was `disc.gsfc.nasa.gov/earthdata-login`
— a login page read 1,024 bytes at a time; `3a791df` and `1c67c38` add the
data host as its own target and make the unapproved application a verdict.

## Notes — E-082 wave 6, the two orbital LiDARs (added 2026-09-20)

`gedi` and `icesat2` are the biosphere wave's two laser stores. Both landed
with an adapter, a synthetic smoke and a runnable `--stage probe`; NEITHER has
a `probe.json` yet, so every size below is a listing measurement and every
per-row number is still the ledger's estimate. Everything here was verified by
request from the sandbox on 2026-09-20 against NASA's Common Metadata
Repository, which is anonymous; the granules are Earthdata-Login protected, so
the bytes are a hosted runner's job.

### What the archives actually hold

| collection | concept id | record | 2022-06 granules | 2022-06 bytes |
|---|---|---|---:|---:|
| GEDI02_A v003 | C3974616071-LPCLOUD | 2019-04-04 → ongoing | 1,428 | 1,516,813 MB |
| GEDI02_B v003 | C3974616135-LPCLOUD | 2019-04-04 → ongoing | 1,428 | 886,429 MB |
| GEDI L4A **v3** | C4212593885-ORNL_CLOUD | 2019-04-04 → ongoing | 1,428 | 238,860 MB |
| GEDI L4A v2.1 | C2237824918-ORNL_CLOUD | 2019-04-17 → 2025-07-09 | 1,427 | 271,839 MB |
| ATL08 release 007 | C3565574177-NSIDC_CPRD | 2018-10-14 → ongoing | 4,641 | 385,160 MB |

So **one month of the three GEDI products this store joins is 2.58 TB** and
one month of ATL08 is **376 GB** (mean 83.0 MB a granule, smallest 16, largest
344). One DAY of GEDI, read through the adapter's own index, is 56.0 GB.

**Three things the ledger did not say, each measured.**

- **GEDI L4A has a VERSION 3**, whose record starts 2019-04-04 — the same day
  as L2A/L2B version 3 — and whose own user guide says it "uses GEDI02_A
  Version 3 as input". The ledger names v2.1, which was produced from GEDI02_A
  version 2, so joining v003 shots to it crosses a reprocessing. The adapter
  defaults to version 3 and `GEDI_L4A_VERSION=2.1` selects the older one.
- **The granule ids line up one-to-one.** On 2022-06-01 all four products list
  exactly 27 granules and the key `(acquisition YYYYDDDHHMMSS, orbit,
  sub-orbit granule, reference track)` matches across all four with nothing
  missing either way. The adapter's `index` re-measures that on the window's
  first day and REFUSES if L2A and L2B ever disagree — verified live through
  the adapter: `{"granules": {"l2a": 27, "l2b": 27, "l4a": 27},
  "l2a_without_l2b": 0, "l2b_without_l2a": 0}`.
- **Version 3 renamed the quality flag.** CMR publishes UMM-Var for the three
  GEDI collections (3,336 variables for L2A v003, 1,672 for L2B, 1,056 for
  L4A v3) — which is a *contrast* with wave 4's finding that there is "no
  anonymous route to a NASA file's variable list", and it was used. There is
  no `/BEAMXXXX/quality_flag` in v003: it is `l2a_quality_flag_rel3`
  (`quality_flag` survives only under `rx_assess/`), L2B has
  `l2b_quality_flag_rel3`, and L4A v3 has `l4a_quality_flag_rel3` where v2.1
  had `l4_quality_flag`. Both adapters resolve every quantity against a
  candidate list and refuse with the group's own dataset list. ATL08 is the
  other way round: **zero** variable associations, so its names come from the
  release-007 data dictionary PDF, fetched anonymously the same day.

### OPeNDAP: available for exactly one of the five collections

Asked of CMR's service associations rather than of a product page:

| collection | services associated |
|---|---|
| GEDI02_A v003, GEDI02_B v003, GEDI L4A v3 | Harmony Trajectory Subsetter only |
| GEDI L4A v2.1 | Harmony Trajectory Subsetter, Harmony OPeNDAP SubSetter (HOSS), **OPeNDAP** |
| ATL08 007 | Harmony Trajectory Subsetter (generic and NSIDC's), an HTTPS file-system service, an S3 service |

An anonymous GET of
`opendap.earthdata.nasa.gov/collections/<c>/granules/<g>.dmr` answers **HTTP
401 after two redirects to Earthdata Login** for all five, so the endpoint
cannot be probed further from here; the service association is the part that
can be read anonymously and it says variable subsetting over OPeNDAP is not
offered for the collections these stores read.

### The fetch chosen, and why

Whole-granule downloading is impossible for `gedi` (31 TB for 2022 against a
hosted lane's ~86 GB of disk and six hours) and merely tight for `icesat2`
(4.6 TB, about 31 h at 40 MB/s, i.e. twelve monthly lanes with no margin). So
both adapters read **only the datasets they need, over HTTP byte ranges**:
`ml/family1/adapters/_h5range.py` gives `h5py` a seekable file-like object
whose reads are `Range:` requests behind a 1 MiB block cache, resolving the
Earthdata redirect once and reusing the signed url. A host that answers 200 to
a range request is a REFUSAL, not a slow success — that is the whole granule
arriving. `GEDI_FETCH=whole` / `ICESAT2_FETCH=whole` is the fallback, and the
Harmony Trajectory Subsetter is the third option, deliberately not built
before a probe says it is needed.

**What the probe must come back with.** Both adapters' counts carry
`storage_and_fetch`, whose `projected_one_year_2022` extrapolates the measured
`read_fraction` (bytes read against the granules' own sizes) and
`bytes_per_second` onto the measured 2022-06 listing. One year fits twelve
hosted monthly lanes when **`fetch_hours` ≤ 72**; equivalently the probe must
show `read_fraction / bytes_per_second ≤ 8.2e-9 s/byte` for `gedi` (at
100 MB/s that is a read fraction of 0.82, at 30 MB/s it is 0.25) and
`≤ 5.6e-8 s/byte` for `icesat2` (whole granules at 18 MB/s already clear it).
Assembly is the other half: parts plus store is about 2.1× the store, so a
year whose `store_bytes` passes **≈ 45 GB** — 9.2e8 rows at `gedi`'s 49 B a
row, 1.15e9 at `icesat2`'s 39 B — goes to a verified box the way `ghcnd`,
`icoads` and `swh` did.

### The size gate

Phase A builds **one year, 2022**, and both adapters enforce it where the
inputs are all it has cost: every stage past `probe` refuses a window that
crosses a calendar-year boundary unless `GEDI_ALLOW_BUILD=1` /
`ICESAT2_ALLOW_BUILD=1` reaches the build step through `adapter_env`. The
Earthdata credential guard sits beside it, for the reason run #152 taught.

## Notes — E-082 wave 6, the biosphere: what landed, what was measured, what is dispatched (added 2026-09-20)

Design: `ml/plans/E082_biosphere_wave.md`. Fourteen adapters landed in one
wave, each with a synthetic smoke and a runnable probe; the tests are
`tests/test_family1_{sif,gbif,gedi,icesat2,canopy30,pheno500,lai500,pace4k,burned500,cat_biosphere,registry}.py`
(607 passed, 1 skipped over the whole family-10 + family-1 suite). Three
probes RAN FROM THE SANDBOX because their sources are anonymous, and their
numbers replace the notes' estimates now:

| store | probe | measured | projected store |
|---|---|---|---|
| `sif` (TROPOMI, S5P-PAL `L2B_SIF___`, one netCDF a day) | 2023-07, all 31 days: 101,722,256 rows, 12.31 GB fetched in 577 s (21.3 MB/s), zero NaN, zero out of bounds, 3 undeclared fills | 121 source bytes a row, 39 stored; footprint median 5.05–5.12 km | **46.8 GB a year uncut**, 245.5 GB for 2018-05-01 → 2023-07-31, ≈ 390 GB to today; 63.9 % kept at cloud ≤ 0.2 (156.8 GB) |
| `gbif` / `gbif_nc` (2026-09-01 snapshot, eight evenly spaced parts) | 3,028,011 records through 55.2 MB in 54 s (column projection reads 22 % of a part); public keeps 69.18 %, private 19.15 % | 99.98 % of kept rows carry a day, 31.3 % a coordinate uncertainty; 35,206 taxa in the sample; no kingdom or class outside the committed code tables | **2.592e9 rows / 101.1 GB public; 7.174e8 rows / 28.0 GB private**; 68.3 GB of transfer to build either |
| `canopy30` (GLAD forest height 2020, `gladxfer.umd.edu`) | tile 50N_000E, 16 groups: 303.8 MB fetched, 286 s (17.9 s a frame, 122.6 s of zstd-6), 281.4 MB stored, valid fraction 1.0, compression 9.8× | the 261 tiles are a strict subset of `lossyear`'s 280 (19 Hansen tiles have no canopy tile); no water or no-data code exists — 0 is "no woody vegetation ≥ 3 m", the store is DENSE | **≈ 33 GB** (stored/native 0.926 over the listing's 35.95 GB; the by-pixel projection says 73 GB and is the worse estimate); ≈ 11–21 h for 261 tiles in one bin → a box, or `CANOPY30_TILES` lanes once a shared-year parts merge exists |

`pace4k` was measured on three real days through the adapter's own code
path (the framework's probe needs ~6 GB for its float64 bounds check at
C = 7 and was OOM-killed here): 46.6–53.5 MB fetched a day through OB.DAAC's
**anonymous** OPeNDAP (no credential at all — the per-variable file paths
404 there, so the adapter reads the CMR-listed bundles with variable
subsetting), 25.4–27.0 MB stored a frame, compression 19–21×, valid fraction
0.084–0.091 for the four global channels and 0.0235 for the three MOANA
channels, whose file is REGIONAL (70 S–70 N, 85 W–25 E; a sub-block at global
row 480, column 2280, verified on every file read). Projected **≈ 9.3 GB a
year**, ≈ 23 GB for the record — five times below the note's 50 GB.

Hosted probes dispatched 2026-09-20 (`check_credentials:"false"` — with
`"true"` the workflow runs only the Earthdata check and no probe):

| store | dispatch | what the number must be for phase A to fit hosted lanes |
|---|---|---|
| `pheno500` | `probe_month 2020-06`, `PHENO500_TILES=h18v04,h12v09,h29v11,h20v08,h11v08` (5 frames) | ≤ 68 s and ≤ 273 MB stored a frame (315 frames a year in one 6 h lane; 25 lanes) |
| `lai500` | `probe_month 2020-07`, `LAI500_TILES=…` `LAI500_GROUPS=terra` (20 frames) | ≤ 1.6 s a frame for one year a lane (13,340 frames), else split the year or use a box |
| `pace4k` | `probe_month 2024-03` (27 frames) | already fits: 35 s a frame → 3.4 h a year, one lane a year |
| `gedi` | `probe_month 2022-06`, `GEDI_FETCH=range GEDI_L4A_VERSION=3 GEDI_MAX_GRANULES=6` | `read_fraction / bytes_per_second ≤ 8.2e-9 s/byte` (a quarter of the granule at 30 MB/s); whole granules need a box whatever happens (2.58 TB a month for the three products, 31.7 TB for 2022) |
| `icesat2` | `probe_month 2022-06`, `ICESAT2_FETCH=range ICESAT2_MAX_GRANULES=12` | `≤ 5.6e-8 s/byte`; whole granules already clear it above 18 MB/s (376 GB a month) |

Assembly of either laser year goes to a verified box above ≈ 45 GB of store
(9.2e8 GEDI rows, 1.15e9 ATL08 segments), which both ledgers' estimates
exceed.

**What the sources contradicted in the design, in one place** (each is
already in the adapter's docstring): GEDI L4A is now **version 3** (from
2019-04-04, built from L2A v003; v2.1 selectable) and v003 has no
`quality_flag` — the flags are `l2a_quality_flag_rel3`, `l2b_quality_flag_rel3`,
`l4a_quality_flag_rel3`; GEDI's eleventh channel is `sensitivity` (a flag
cannot be a float16 channel); OPeNDAP is offered for none of the collections
read; the daily TROPOMI record starts **2018-05-01** (04-30 is the per-orbit
product); TROPOSIF ships no per-sounding quality flag, so `qc` grades cloud
fraction against the file's own thresholds; GBIF's `taxonkey` is a string,
a geospatial issue is **seven** issue names, and the snapshot's kingdom
names (`Pseudomonadati`, `Bacillati`, …) disagree with GBIF's own species
API, so the code tables are measured from the snapshot and falsified at
`index`; `lai500` composites hold 274–290 tiles (union 290), not 293, and
VNP15A2H v002 is HDF5 with unsuffixed dataset names; MCD12Q2 2025
double-lists 311 tiles (highest production kept) and CMR declares a constant
274.756 MB for every granule; MOANA data run past the producer's `valid_max`
(kept and counted) and carry an undocumented int32-minimum sentinel (NaN,
counted); ESA BIOMASS **is** publicly listed (FedEO: 27 collections, L1A
417,521 · L1B 313,329 · L1C 126,785 · L2A 25,792 · L2B 0 products on
2026-09-20 — `eocat.esa.int` resets, `biomass-pdgs` 502s, CDSE has no BIOMASS
collection); ETH's 10 m tile index is public (2,651 tiles of 3° × 3°); the
Meta 1 m maps are decade composites (2009–2020, measured from their per-tile
sidecars), so their rows carry the collection's reference epoch; JAXA's
ScanSAR L2.2 archive is anonymous on AWS Open Data with STAC items already
written (5,716 for 2025), and only the 25 m mosaics need the JAXA account
(their directory answers 401).

**Two framework facts worth keeping.** (1) Tier P has no per-row footprint or
per-row `log2_dt`: both assemblers fill `fp.npy` from the adapter's constants
and `check_store` asserts the columns are constant. `gbif` therefore puts each
row's own footprint in `qc`'s upper six bits (`k = round(log2 metres)`, real
`log2_fp = k − 14.7639`; bit 0 uncertainty unknown, bit 1 month-only date)
and `sif` stores the nominal −2.49 while the probe measures the corner
distribution. A `row_fp` hook is the right follow-up; it touches `ROW_KEYS`,
`_pack`, `PartWriter`, both assemblers and every part on the Hub, so it is not
a wave-6 change. (2) `burned500`'s groups are blocks of nine sinusoidal tiles
in one row (46 groups, 28,520 files, a 0.83 GB probe array): one group per
tile is 166,160 files, past the Hub's 100,000-per-repository limit, and the
new `no_frame_in_bin` skip reason does not change that arithmetic.
`_modis_sin.py` (the sinusoidal-tile machinery `pheno500` and `lai500` share)
and `burned500`'s own grid constants agree to the digit and differ in group
shape; consolidating them is named in both docstrings.

### The hosted probes, read 2026-09-20 17:41–17:50Z (files in `ml/family1/probes/`)

| store | run | measured | verdict |
|---|---|---|---|
| `pheno500` | #293, 2020-06, five tiles | 15.0 s a frame (288 MB fetched a granule — CMR's constant 274.756 MB was close to real), 27–50 MB stored a frame for full tiles (compression 2.3–4.3×), valid fraction 0.35–0.73 on land, 0.002 on a sea tile; ≈ 0.77 GB per tile over the 25-year record → **≈ 240 GB for the store, ≈ 9.7 GB a year** (the ledger's 10 GB a year holds) | 315 frames × 15 s = 79 min a year: **one hosted lane a year**. Lanes #299–#310 (2014–2025) dispatched; 2001–2013 after them |
| `lai500` | #294, 2020-07, four Terra composites × five tiles | 4.74 s a frame, 11 MB fetched a granule, 3.5–9.5 MB stored a frame (compression 5–13×), valid 0.70–1.00; 0.43 B a valid pixel → **≈ 90 GB a year** (the ledger's 60 GB was low) | 13,340 frames × 4.74 s = 17.6 h a year: **four quarter-lanes a year**. #311–#314 (2020) dispatched; 2021–2024 after their verdict, because 450 GB for five Terra years is a decision, not a default |
| `pace4k` | #295, 2024-03, 23 frames (the record starts 03-05) | 14.6 s a frame, 20 MB stored a frame (0.2–30 MB), valid 0.068 global channels, 0.017 MOANA; the framework's estimate 5.8 GB for the record | **one lane a year**: #328–#330 (2024, 2025, 2026 to September) dispatched, hosted assembly after |
| `icesat2` | #297 range mode, #298 whole mode, 2022-06, 12 granules | range: 1.6 MB/s over 20 requests a granule (read fraction 0.18, 105 s) — too slow; **whole: 36.6 MB/s, 25.7 s for 940 MB**, 32,250 rows a granule, 386,995 rows kept of 438,022 segments (23,199 inland water, 13,274 sea, 14,554 outside the window); h_canopy is NaN on 88 % of segments (bare ground), terrain height on none | 2022: 55,692 granules, **1.80 × 10⁹ rows, 70 GB store, 35 h of fetch** → twelve monthly lanes of ≈ 3 h in whole mode, #325–#339 dispatched (nine re-dispatched after a malformed end date cancelled #316–#324); assembly on the box (parts + store ≈ 150 GB) |
| `gedi` | #296 FAILED: `ConnectTimeout` to `urs.earthdata.nasa.gov` while resolving the first L2A granule's redirect, after the index had listed 1,428 granules a product; re-dispatched as #315 | | if it fails the same way twice, the redirect resolve needs a retry with back-off — URS is the login host every Earthdata download passes through, and the other four Earthdata probes of the same minute succeeded |

### Two things the first lanes taught, 2026-09-20 18:00Z

1. **A fetch lane parks nothing without `--push-parts`.** Fifteen lanes
   (pheno500 2014–2025, pace4k 2024–2026) were dispatched with `stage=index,fetch`
   and no `extra_args`, fetched, and discarded; cancelled and re-dispatched
   with `extra_args: "--push-parts"` (#340–#354).
2. **A lane is a year, never part of one — and the laser adapters' probe
   cap must not reach the build.** ICESat-2 2022 was dispatched as twelve
   monthly lanes; two "succeeded" in 38 s, having read twelve granules of
   4,884 under `ICESAT2_MAX_GRANULES`'s default of 12 (the probe's cap) and
   marked 2022 done — and even without the cap, the parts layout is one
   folder per year with one index, one ledger and one `done.json`, so the
   twelve lanes would have overwritten each other's index and the assembler
   would have built a short year with nothing to say so. Cancelled; nothing
   reached the Hub (no `--push-parts`). Fixed in the adapters the same hour:
   the cap applies in `fetch_month` (the probe) only, `fetch_year` reads every
   granule, and a cap set by name for a build refuses unless
   `*_ALLOW_CAPPED_BUILD=1` (tests in `test_family1_icesat2.py` /
   `test_family1_gedi.py`). The lai500 quarter-lanes were cancelled for the
   same layout reason. What unblocks sub-year lanes is a **lane-aware parts
   layout**: `partials/<family>/<store>/<year>/<lane>/` with the lane named by
   its window (`m06`, `q3`) or its group subset (`tiles-50N_000E`), each lane
   its own parts, shard index, ledger and `done.json`; the assembler merges the
   lanes of a year, asserts their bins (tier G) or their time windows (tier P)
   are disjoint, sums their ledgers, and refuses when a lane the plan declares
   is missing; a year folder with parts at its top level stays a single lane,
   so every store on the Hub reads unchanged. That is the next framework
   task; until it lands, `icesat2`, `gedi`, `lai500` beyond one lane-year and
   the single-bin `canopy30` / `lossyear` are box jobs or wait.

### The lane-aware parts layout, landed 2026-09-20 ~19:30Z (E-082 wave 7)

A lane inside a year is now a NAMED lane. `start=2022-06-01 end=2022-06-30`
writes lane `m06` of 2022 under `partials/<family>/<store>/2022/m06/` — its
own parts, shard index, `counts.json` (with `lane`, `lane_window`,
`lane_groups`) and `done.json` last; quarters are `q1`–`q4`, any window
`d0701-0930`, a group subset `g-<sha1[:8] of the sorted group list>`
(`CANOPY30_TILES`, `LOSSYEAR_TILES` implement the `group_subset()` hook). A
whole-year lane is the unnamed lane and writes exactly what was written
before, so every part and store on the Hub reads unchanged (pinned:
`tests/test_family1_lanes.py`, 15 tests; the whole suite 624 passed). The
assembler merges the lanes of a year — union of shard indices, concatenated
parts, summed ledgers — and REFUSES a declared lane that never arrived
(`--lanes months|quarters|<list>` declares; without a declaration store.json
says `declared: false` and the log warns), two lanes holding the same
(group, bin) shard, overlapping windows over intersecting groups, or a named
lane beside the unnamed one. `store.json` gains `lanes_by_year` only when a
year was laned. Two facts the code taught: a five-day bin does not respect a
month boundary, so a named tier-G lane owns the bins whose FIRST day falls in
its window (the four 2020 quarters own bins 2776–2848, disjoint, exactly the
year's 73), and therefore a laned tier-G assembly starts on the year's first
bin start (`--start 2020-01-02` for 2020) or carries `--allow-missing-years`;
and a product-selecting knob (`LST05_GROUPS`, `LAI500_GROUPS`) is not a group
subset and does not name a lane.
