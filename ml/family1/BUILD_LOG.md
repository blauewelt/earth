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

## Notes — E-082 wave 4, the read-out targets and the credentialed
## ocean/atmosphere stores (added 2026-09-18)

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
- **`swh`'s six fetch lanes** (1991-1996, 1997-2002, 2003-2008, 2009-2013,
  2014-2018, 2019-2023) each ran in 21-30 min on hosted runners and parked
  their years under `partials/family1_gf/swh/<year>/`.

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
