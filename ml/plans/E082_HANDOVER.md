# E-082 handover — building families 1.gf, 1.0.tf and 0.9.tf (state on 2026-09-19 11:00Z)

*For any agent picking this up. E-082 is the build of three new tensor
families designed on 2026-09-16: **family 1.gf** — global ocean and
atmosphere observations at ≤ 10 km and ≤ 5 days; **family 1.0.tf** — land and
coast at the same scale; **family 0.9.tf** — 1.0.tf with the raw satellite
imagery replaced by Google's AlphaEarth embedding, everything else inherited
by reference. Read this first, then `ml/plans/E082_family1_builds.md` (the
plan), `ml/family1/BUILD_LOG.md` (every landing, every measured quirk, in
detail), `ml/family1/ADAPTER_CONTRACT.md` (the code interface), and
`ml/CLAUDE.md` §0–§7 (the working rules; §0b says the main session plans and
Opus sub-agents implement).*

## 0 · The one thing that changed the operating mode

**The Opus weekly quota is exhausted until 2026-09-22 04:00 UTC.** Three
implementation agents were cut off mid-task by the limit on 2026-09-18/19
(wave 3 catalogues, wave 4a credentialed land stores, the box operator), and
their partial state is described below from the artefacts they left — the Hub,
the workflow runs, `BUILD_LOG.md` — not from their reports, which never
arrived. Until the quota resets, the main session can dispatch, monitor,
verify and write; adapter code and new stores wait for Opus (or for Chris's
say-so to spend Fable on implementation).

## 1 · What is on the Hub (verified from `store.json` on 2026-09-19)

Public repository `chfrank/earth-tensors`, `tensors/family1_gf/` and
`tensors/family1_tf/`; private repository `chfrank/earth-tensors-private`
(anonymous read = 401).

| store | family | N (rows or frames) | stored | record | notes |
|---|---|---|---|---|---|
| `glodap` | 1.gf | 1,460,215 bottles | 77 MB | 1972–2023 | GLODAPv3 |
| `wod` | 1.gf | 15,281,373 casts | 4.39 GB | 1772–2026 | schema 3, C = 128 |
| `icoads` | 1.gf | 1,107,951,670 reports | 52.07 GB | 1662–2026 | schema 3; 19 % below the July-based projection |
| `bgcargo` | 1.gf | 335,231 profiles | 73 MB | 2002–2026 | |
| `oceansites` | 1.gf | 66,483,202 rows | 5.52 GB | 1980–2026 | tropical arrays excluded |
| `seaice_asi` | 1.gf, tier G | 5,187 daily frames × 2 hemispheres | 1.66 GB | 2012-07 → 2026-09 | **private** with `licence_pending` until Bremen answers |
| `ghcnd` | 1.0.tf | 1,143,728,366 station-days | 46.91 GB | 1763–2026 | schema 3; assembled on a box (48.9 GB RSS peak) |
| `igra` | 1.0.tf | 44,842,766 soundings | 8.57 GB | 1905–2026 | schema 3, C = 80 |
| `tide` | 1.0.tf | 1,080,914,415 readings | 35.67 GB | 1800–2026 | GESLA-4 public track; 52 Great Lakes gauges excluded by the ±50 m bound (decision open) |
| `tide_private` | 1.0.tf | 179,314,808 | 5.92 GB | 1821–2026 | **private**: CMEMS / CV / UZ / SANHO contributors |
| `ndbc` | 1.0.tf | 628,157,329 | 29.52 GB | 1970–2025 | |
| `gliders` | 1.0.tf | 2,808,590 profiles | 436 MB | 2003–2026 | 44 % below the June-based projection |
| `chirps05` | 1.0.tf, tier G | 3,288 pentad frames (F = 2 half-bins) | 19.06 GB | 1981-01 → 2026-08 | exception E4 |
| `lossyear` | 1.0.tf, tier G | one bin (3215), 280 Hansen land tiles as 2.5° sub-tile groups | see BUILD_LOG | 2001–2025 loss year + tree cover 2000 | built on the box, run #237 |
| `cat_landsat` | 1.0.tf, tier T | catalogue | | 1982 → | scene rows + `assets.parquet` |
| `cat_s1` | 1.0.tf, tier T | catalogue | | 2014-10 → | |
| `cat_nisar` | 1.0.tf, tier T | 138,804 scenes | 8.9 MB | 2025-10 → 2026-09 | built from the sandbox |
| `cat_viirs` | 1.0.tf, tier T | 2,341,071 scenes | 128.8 MB | 2012 → 2026-09 | built from the sandbox |
| `cat_olci` | 1.0.tf, tier T | 1,591,637 scenes | 126.8 MB | 2016-04 → 2026-09 | C = 6 with a `coastal` flag |

Nineteen stores. `cat_landsat` and `cat_s1` are on the Hub but their rows in
`BUILD_LOG.md` were never written (the wave-3 agent was cut off) — read their
`store.json` and add the rows.

## 2 · What is PARKED on the Hub (parts complete, no published store)

Checked with `node scripts/family1_runs.mjs --parts <family> <store>` on
2026-09-19. Every year listed carries its `done.json`.

| store | family | parked years | what remains | where it fits |
|---|---|---|---|---|
| `oc4k` | 1.gf, tier G | 1997–2022 (23 years done; **2017, 2018, 2019 re-dispatched as runs #255–#257 at 10:33Z** after their lanes were cancelled twice by the concurrency group) | assembly of ≈ 159 GB of parts + a ≈ 160 GB store | **a rented box** (≥ 400 GB disk, ≥ 64 GB RAM) |
| `swh` | 1.gf | 1991–2023, all 33 years, 48.73 GB | assembly (≈ 103 GB of disk) | a box, or a hosted runner if the assembly stays under ~86 GB — it will not; use the box |
| `lst05` | 1.0.tf, tier G | 1999–2026, 28 year folders | assembly; size unknown — read the parts' `counts.json` totals first | box if > 60 GB of parts |
| `snow05` | 1.0.tf, tier G | 1999–2026, 28 year folders | assembly (note estimate ≈ 60 GB) | box |
| `fire` | 1.0.tf | 2000–2006 only (lane #233) | the remaining year lanes 2007–2026, then assembly | hosted lanes, hosted assembly |
| `cat_hls` | 1.0.tf, tier T | 2013–2024 | lanes for 2025 and 2026 (their runs #228, #243 failed), then assembly | hosted |
| `cat_s2` | 1.0.tf, tier T | 2015–2020 | lanes 2021–2026 (cancelled by the concurrency group, never re-run), then assembly | hosted |
| `cat_ecostress` | 1.0.tf, tier T | 2018–2026, complete | assembly | hosted |

The exact assembly form, used for every store so far:
`node scripts/family1_dispatch.mjs family1-build.yml '{"store":"<s>","stage":"assemble,publish,check","extra_args":"--parts-from-hub --assemble streaming","runner":"<ubuntu-latest | gpu-box-NNNN>"}'`
(for tier-G stores the wave-2 agent used `stage=all` with `--parts-from-hub`;
both routes pull the parts and skip the fetch). The workflow already sets
`HF_HUB_DISABLE_XET=1`, which the 52 GB ICOADS publish needed.

## 3 · The box

A Vast instance **51415980** (RTX 3060 host, $0.131/h, runner name
`gpu-box-31947967`) built `lossyear` (run #237, green 09:21Z on 2026-09-19)
and was then left RUNNING IDLE by the cut-off operator; the main session
**stopped it at 10:35Z** (stop keeps its disk and registration). Either
`node scripts/gpu_box.mjs start 51415980` when the oc4k lanes finish and
dispatch the oc4k, swh, lst05 and snow05 assemblies to `runner=gpu-box-31947967`
one after another, or — if `start` sits `offline` for more than ten minutes and
the host is absent from `offers` (ml/CLAUDE.md §7, "a parked box's host can go
dark") — destroy it and rent a fresh **verified** box with ≥ 400 GB disk,
≥ 64 GB RAM and ≥ 1 Gbps up. Destroy the box when the last assembly verifies;
never leave it running without a job. The Vast key is at `/home/claude/.vast_key`
(from the project doc `claude/vast-access.md`; recreate the file with the Write
tool on a fresh container, never via argv).

## 4 · Not built, and the measured reason (each needs a decision or a code change)

- **`sst_acspo02`** (NOAA 0.02° SST): probed — 0.457 bytes per valid pixel, ≈ 501 GB for the record, one year ≈ 19 GB in two six-month hosted lanes. The adapter refuses more than one year without `SST_ACSPO02_MAX_YEARS`; nothing dispatched beyond the probe (#199). Decision: build 2020 as a first year, then decide the record.
- **`swot`**: probed; the storage decision is written up with numbers in BUILD_LOG.md ("swot: the storage decision") — tier P 118 GB/yr vs a per-pass tier-G group 26 GB/yr, which needs a narrower tile shape in `sharded.py`. Decision for Chris; the adapter refuses any stage past `probe` without `SWOT_ALLOW_BUILD=1`.
- **`irtb`**: adapter + smoke landed; `sharded.py`'s frame mask is a uint64, so the finest the layout holds is two-hourly (F = 60), not half-hourly (F = 240) — a layout change for phase C. Probe #178 was queued at cut-off; read its artifact.
- **`xco2`**: adapter + smoke landed; probe #179 queued at cut-off; read its artifact, then build (small).
- **`burned500`**: HDF4 solved with `pyhdf`; blocked on a one-line framework change — monthly frames on the five-day axis write a shard for every bin; a third skip reason in `OUTSIDE_RECORD` is needed (the agent deliberately did not touch the framework while four waves shared the tree).
- **`alerts`**: RADD, GLAD-L and GLAD-S2 are CC BY 4.0 (licence settled), but their native tiles are requester-pays S3 / Earth Engine; DIST-ALERT via CMR is ~11,000 granules a day (a TB/day of source for a 30 GB store). Proposal: DIST-ANN (annual) first, DIST-ALERT later with a smarter granule filter.
- **`cat_gfm`**: the Copernicus flood service has no global listing (per-AOI, bearer token). Build the OPERA DSWx half only, or drop the GFM column.
- **`static_fine`**: designed and measured, not built — BUILD_LOG.md §"static_fine — DESIGNED AND MEASURED, NOT BUILT" says exactly why (the WorldCover fractions computed from 10 m tiles are our own derived product and were not to be shipped quietly). Decision for Chris.
- **`flux`**: the AmeriFlux/FLUXNET download was scripted (probe #234 green; some sites still under FLUXNET2015 naming, commit 2850b90) — check whether its lanes ran (none parked on the Hub, so no) and dispatch `stage=all`.
- **`refl05`**: probe #232 measured **2.47 TB** for the Terra record against the note's 700 GB (valid fraction), commit 6c0b3ba. Do not build without a decision: 2015–2026 only would be ≈ 900 GB.
- **`sst_cci05`**: no v3 on CEDA; super-collating 14 sensors × day/night reads ~5 TB for a ~1.2 TB store. Phase B/C.
- **`ismn`, `river_grdc`, `ecad`** (private track, phase B): ISMN credentials are in place; GRDC needs Chris to submit the request form (there is no account — see §6).
- **Family 0.9.tf's own stores** (`aef1k`, `aef_dots`, `aef100r`): nothing started. First step is one UTM zone-year through the pooling code (throughput, cost, unit-length and clip checks), then `aef1k` in-region (AWS keys not provided; a US Vast box is the fallback).
- **The three registries** (`family1gf.json`, `family1tf.json`, `family09tf.json` with `inherits`) and the `siblings` line in family 10.2's registry: not written. `ml/build_family10_registry.py` is the model.

## 5 · The new requirement (Chris, 2026-09-19): the biosphere, especially forests

> "We want to make this a globally useful model that can predict biosphere
> (forests). Lidar is very important for those. But maybe many other
> datasets. Can you make sure all families have such data?"

What the families already hold for this: `lossyear` (built), `chirps05`,
`lst05`/`snow05` (parked), `fire` (partly parked), `flux` (adapter ready),
`cat_landsat`/`cat_s1`/`cat_nisar`/`cat_olci`/`cat_viirs` (built), the
scene catalogues of Sentinel-2 and HLS (parked), and in the design notes but
not yet built: `gedi` (phase B), `icesat2` (phase C), `biomass100` (phase C),
`landcover10` (phase C), `alerts`, `cat_palsar` (phase B). Family 1.gf's
biosphere is the ocean's: `oc4k` (parked), `bgcargo`, `glodap`, `wod` (built).
Family 0.9.tf inherits every 1.0.tf store, and its embedding was itself
trained against GEDI.

**Proposed wave 6 — "biosphere" (the main session's design, to be written into
the three notes and their summaries as a new section, then built):**

1. **Pull the LiDAR forward to phase A.** `gedi` (L2A/L2B relative-height profile + L4A biomass, 0.2–0.5 TB, Earthdata — credentials are in place) and `icesat2` ATL08 (≈ 1.1 TB; build 2018–2026 at 100 m segments). These are the only direct measurements of canopy structure and they are what "3D canopy" means.
2. **Canopy-height and biomass maps as annual targets**, tier G sharded: GLAD 30 m canopy height 2020 (Potapov, CC BY), ETH 10 m canopy height 2020 (Lang et al., CC BY 4.0), Meta/WRI 1 m canopy height (CC BY 4.0, AWS), ESA CCI Biomass v7 100 m (2005–2024), GEDI L4B 1 km biomass. Same filing rule as `lossyear`.
3. **Photosynthesis observed from orbit**: `sif` — solar-induced chlorophyll fluorescence, TROPOMI (7 × 3.5 km, daily soundings, 2018 →, Caltech/GES DISC) and OCO-2/OCO-3 SIF Lite (2014 →) as tier-P rows; passes the ≤ 10 km rule exactly and is the most direct global measure of plant activity.
4. **Leaf area and phenology**: MODIS/VIIRS LAI and FPAR 500 m 8-day (MOD15A2H, VNP15A2H; an 8-day exception, E7) and the MCD12Q2 annual phenology metrics as targets; the daily 0.05° reflectance already gives NDVI daily.
5. **Species occurrences as a point network**: GBIF's monthly occurrence snapshot (parquet on AWS Open Data; ~3 billion dated, georeferenced records; licence per record — CC0/CC BY public, CC BY-NC private track). Tier P, platform = taxon key. This is the biosphere's ICOADS.
6. **Radar biomass**: `cat_palsar` pulled to phase A as a catalogue; ESA BIOMASS (P-band, launched 2025) — verify whether L2 products are public yet; NISAR is already catalogued.
7. **For 1.gf**: PACE OCI hyperspectral ocean colour (1.2 km daily, 2024 →, Earthdata) as `pace1k` beside `oc4k`; marine GBIF rides item 5.

Every one of these needs a design row (source, granularity, licence, size
estimate) in the notes before an adapter — the notes' rule is unchanged: raw
observations first, derived products only where they carry otherwise
inaccessible observations (the height/biomass maps are targets, not inputs).

## 6 · Credentials and people

Repository secrets in place: `EARTHDATA_USERNAME/PASSWORD` (verified against
LP DAAC, GES DISC, PO.DAAC through Earthdata Login, run #90; no EULA),
`FIRMS_MAP_KEY`, `FLUXNET_USERNAME/PASSWORD`, `ISMN_USERNAME/PASSWORD`,
`HF_PRIVATE_READ_TOKEN`, plus the older `HF_TOKEN`, `COPERNICUSMARINE_*`,
`GFW_API_TOKEN`, Cloudflare, GCP. Sandbox files (recreate with the Write tool
from the project docs `claude/github-access.md`, `claude/huggingface-access.md`,
`claude/vast-access.md`): `/home/claude/.gh_pat`, `/home/claude/.hf_token`,
`/home/claude/.vast_key`.

Waiting on Chris: the **GRDC** request form (no account exists; the portal
takes a per-request form with name, affiliation and purpose — either he fills
it, or he tells the agent what to enter); the **Bremen** email (drafted in
English and German in the chat of 2026-09-17) whose answer decides whether
`seaice_asi` is re-published publicly; the **Great Lakes** decision for
`tide`; the `swot`, `refl05`, `sst_acspo02` size decisions; `static_fine`.

## 7 · How to operate

- Dispatch: `node scripts/family1_dispatch.mjs family1-build.yml '<inputs json>'` — inputs `store`, `stage`, `start`, `end`, `runner`, `extra_args`, `probe_month`, `check_credentials`, `adapter_env`.
- Watch: `node scripts/family1_runs.mjs [--n 40] [--status in_progress|queued]`; parked parts: `node scripts/family1_runs.mjs --parts <family> <store>`.
- Verify a landing by its `store.json` on the Hub, never by the run's colour; then add the BUILD_LOG.md row.
- Concurrency: ~20 jobs for the whole account; lanes with the same store are in one concurrency group keyed by window, so a new lane with the SAME window cancels the pending one.
- Hosted runners: 6 h, ~86 GB free; assemblies above ~60 GB of parts go to a box. Streaming assemblies of wide stores need `--assemble streaming`.
- Tests: `python3 -m pytest -q tests/test_build_family10_stores.py tests/test_build_family1_stores.py tests/test_family1_*.py` (337 passing at commit 42f0b81); `tests/test_workflow_config.py` case 3 fails on an unrelated ml-train recipe and did before E-082.
- Push: `git pull --rebase origin main` then `node scripts/git_api_push.mjs --branch main --token-file /home/claude/.gh_pat`; gh-pages by a `force:false` PATCH.

## 8 · Order of work for the next agent

1. Watch runs #255–#257 (oc4k 2017–2019). When green, start the box (or rent one), assemble `oc4k`, then `swh`, `lst05`, `snow05` on it; destroy it; rows into BUILD_LOG.md.
2. Hosted: assemble `cat_ecostress`; re-run the missing `cat_hls` (2025, 2026) and `cat_s2` (2021–2026) lanes, then assemble both; run the `fire` lanes 2007–2026 and assemble; dispatch `flux` and `xco2`; read #178/#179's artifacts.
3. Write the BUILD_LOG.md rows for `cat_landsat` and `cat_s1` from their store.json.
4. When Opus is back (2026-09-22): the biosphere wave (§5) — notes first, then adapters; the three registries; `burned500`'s framework fix; the `alerts` DIST-ANN route; 0.9.tf's pooling probe.
5. Keep `claude/e082-build-status.md` in the project current with each landing.
