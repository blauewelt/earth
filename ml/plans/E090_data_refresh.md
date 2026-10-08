# E-090 · Every store the Data tab serves, kept current with its upstream

*Written 2026-10-08. Chris: "I guess we need a github workflow that always
updates all data as well?" Decisions handed down by the planning session are
marked **(given)**; every other pick is reversible and says so.*

## 0 · In plain English

The Data tab (the page on blauewelt.org/earth that filters and downloads
every public store in the browser, `docs/DATA_TAB.md`) reads stores that were
each built ONCE, to the last day their producer served on the build day.
Producers keep publishing: NOAA adds a sea-surface-temperature day every day,
Copernicus Marine a week of ocean reanalysis every week, ECMWF a month of
final ERA5 every month. This experiment is the job that follows them: every
day, on free GitHub-hosted runners, it asks each producer how far its record
reaches, does nothing where the store already ends there, and otherwise
re-fetches only the store's last few weeks to months (its last *lane*),
splices them into the published store, checks every byte it wrote, and only
then lets the tab offer the new days.

- **The code:** [`ml/data_refresh.py`](https://github.com/blauewelt/earth/blob/main/ml/data_refresh.py)
- **The workflow:** [`.github/workflows/data-refresh.yml`](https://github.com/blauewelt/earth/blob/main/.github/workflows/data-refresh.yml)
- **The toy tests:** [`tests/test_data_refresh.py`](https://github.com/blauewelt/earth/blob/main/tests/test_data_refresh.py)
- **The reader check:** [`scripts/refresh_reader_check.mjs`](https://github.com/blauewelt/earth/blob/main/scripts/refresh_reader_check.mjs)
- **The status line of every store:** [`tensors/refresh/status.json`](https://huggingface.co/datasets/chfrank/earth-tensors/blob/main/tensors/refresh/status.json) on the Hub

Words used below: a **store** is one dataset on the Hub; **tier G** is the
gridded layout (one file of compressed 256 × 256 tiles per five-day *bin*,
plus a small index per bin — `ml/family1/sharded.py`); **tier P** is the
point layout (rows sorted by time with per-bin offsets); a **lane** is one
hosted-runner fetch of a window of the record (a whole year, or a named part
of one such as `d0701-0915`), whose output is parked on the Hub as *parts*
with a *ledger* (`counts.json`) of what it counted.

## 1 · Decisions

| # | decision | pick | why / reversible how |
|---|---|---|---|
| D1 | what is frozen | **(given)** the five-day training tensor family 7.2 (`g025`, `g100`, `oc025`, `rg100`), the paper's climatology files (`clim/{paper,dev,all}`) and E-086's tensor normals | the paper must keep reproducing; they are listed in `POLICY` as `frozen` so `plan --stores all` names them |
| D2 | where it runs | **(given)** ubuntu-latest only, every job; `schedule` + `workflow_dispatch` triggers only | ml/CLAUDE.md §6 — a scheduled job must never reach a rented machine |
| D3 | the unit of re-fetch | the store's **last lane**, re-run with the builder's own `index` + `fetch` stages: a whole calendar year for an unnamed-lane store (all of family 7.2d and 1.2, `pace4k`); for a named-lane year the last named lane extended to 31 December (`d0701-0915` → `d0701-1231`, a name then stable across refreshes) | a lane's ledger is an AGGREGATE (counts, maxima, lists): it cannot be split by bin, so the smallest unit whose counts can be replaced exactly is the lane. Reversible: a finer unit needs per-bin ledgers |
| D4 | how store.json is made | by the assembler's own function (`build_family1_stores.grid_store_meta`, extracted for this) from the published shard indices + published ledgers of the kept lanes + the refetched lane's | no second definition of a store |
| D5 | **the self-check** | before anything changes, the PUBLISHED store.json is rebuilt from its own published shard indices and ledgers; every structural field (groups, per-year frames, counts, counts by year, missing frames, lanes, date range, bins requested, channels, dtype, degraded) must be equal, else the store is refused | a splice keeps what it does not touch; this proves it can. Measured 2026-10-08 from the sandbox: all eight family-7.2d / ERA5 stores and `pace4k` rebuild exactly (45, 35, 30 and 3 lane ledgers) |
| D6 | **mid-read safety** (shards) | a rewritten bin must be **prefix-preserving** — the old shard is the first bytes of the new one and the old frames' index entries are unchanged — except a bin holding days the published store calls provisional; any other change to a published bin is a **revision** and is refused (`--allow-revisions` after a human looked) | the reader computes a shard's URL from its bin number (`src/f1data.js`), so a partly filled last bin is rewritten under its own name. Prefix-preserving means a reader holding the old index and the new shard (or the reverse) reads the old frames exactly or fails loudly on a short or misaligned read — never a silent mix. Picked over "new file names referenced from the registry", which needs a reader change |
| D7 | **the commit points** | (1) ONE Hub commit per store carries its changed shards + indices + `shard_index.npy` + `tile_grid.json` + `store.json` + `manifest.json` (atomic at the repository); (2) the family **registry** — only it makes new days selectable (`record_span`); (3) for the sums, the committed **`data/gridded_monthly_index.json`** | data before announcement; the tab never offers a day whose bytes have not been read back |
| D8 | verification | after the data commit: every committed file read back **at that commit's revision** and sha256-compared; the new frames (up to 40, evenly sampled, first and last included) re-read over HTTP through `sharded.ShardedGroup` and compared with the local bytes; a failure **reverts** the commit. After the registry: the newest day read through the tab's own reader (`scripts/refresh_reader_check.mjs`, node + `src/f1data.js`) against `sharded.py`, exactly — a mismatch republishes the previous registry | the stated order: verify, then announce |
| D9 | the registry guard | `ml/registry_guard.py --refresh NAME`: the refreshed store's DATA fields may change (files, `store_*`, counts, span fields, built_at, builder sha, source_segments, bins); everything that DESCRIBES it (title, channels, licence, adapter, path, tier, layout, sources, qc, probe, estimate) is held to equality | a refresh must not be a way to edit a store's description |
| D10 | provisional data | OISST's preliminary days are listed by the lane (`preliminary_days`, from NCEI's per-day file names); each day the plan asks NCEI which of the store's preliminary days are now final and re-fetches from the first of them. ERA5T (ERA5's preliminary stream) is **not admitted**, as the adapter decided (E-085); GLORYS has one dataset id and no interim stream today | the final value replaces the preliminary one within a day of NCEI finalising it |
| D11 | **sums only for settled months** | E-088's sums are written for a (year, month) only when the month is complete in the store and holds no provisional day; every other changed month is **blanked** (planes 0, `frames_present` 0). New sums go to a NEW folder `monthly/r<UTC>/`; the old version's files stay | the tab chooses sums vs native maps by comparing frame COUNTS (`monthlyPath`), so a preliminary day replaced by its final value would leave an equal count over a stale sum. Blanked = "read the native maps". A new folder keeps the committed index consistent with the files it names until the site redeploys. Bounds the rewrites to about one per store per month |
| D12 | failure isolation and reporting | one matrix job per store, `fail-fast: false`; one status line per store on the Hub (`tensors/refresh/status.json`: state, last checked, upstream newest, record end, gap, last updated, commit); ONE open GitHub issue ("E-090 data refresh failing") created or commented on when any job fails | cheap; the Data tab can show the status file later (another agent's UI work) |
| D13 | concurrency | `concurrency: data-refresh`, `cancel-in-progress: false` — two runs never overlap | a second run would race the first's splice |
| D14 | secrets per job | plan: none; update: `HF_TOKEN` (not in a dry run), the Copernicus Marine pair only for `glorys025d`, the Earthdata pair only for `sst_acspo02` / `irtb`; registry, sums, status: `HF_TOKEN`; finish: the job token with `contents/actions/issues: write` | least privilege per store |
| D15 | the parts follow the store | after a verified commit the refetched lane's parts are pushed to `partials/` (the build's own `push_parts`), and a superseded named lane's folder is deleted | a later from-scratch assembly from parts reproduces the refreshed store, and the NEXT refresh's self-check (D5) can explain it |
| D16 | batching the long lanes | `pace4k`, `sst_acspo02`, `irtb` refresh when ≥ 7 new days are upstream (`min_gap_days`); every store is still checked daily | their lane re-fetch is 0.5–3 h |
| D17 | schedule | daily, `cron: 23 5 * * *` (05:23 UTC: after PSL's overnight OISST update, off the hour) — enabled only after a real dispatch run was green | the instruction |
| D18 | recovery | workflow input `sums_from_run`: remake only the sums of the stores an earlier run updated, from its result artifacts; the failure issue closes itself on the next green real run | a sums failure must not need a data re-fetch |

## 2 · The inventory — every store the tab's registries list

Measured 2026-10-08 07:27 UTC from the sandbox (`python3 ml/data_refresh.py
plan --stores all`), and on the hosted runner by the dry run (§8). "Ours" is
the live registry's `record_span` end. Lag = how far behind today the
producer's own newest day is.

### 2.1 Gridded (tier G) — refreshed by this experiment

| store | what | source and access path | credential | cadence · lag | ours | upstream today | gap | method | hosted? |
|---|---|---|---|---|---|---|---|---|---|
| `family7_2d/oisst025d` | NOAA OISST v2.1 sea-surface temperature and sea ice, daily, 0.25° | PSL yearly `sst/icec.day.mean.<Y>.nc` (THREDDS time axis for the probe; `downloads.psl.noaa.gov` first in a refresh, the Hub's PSL mirror as fallback); NCEI's per-day file names for preliminary vs final | none | daily · 2 days; the newest ~14 days preliminary, final ~2 weeks later | 2026-10-04 | **2026-10-06** (279 days in the 2026 file); 09-21 and 09-22 now final | 2 + 2 replaced | year lane 2026 (~280 frames, ~10 min) | **yes**, scheduled |
| `family7_2d/ncep100d` | NCEP/NCAR Reanalysis 1 surface and land, daily, 1° | PSL `surface_gauss/<var>.gauss.<Y>.nc`, 13 stems (THREDDS) | none | daily in principle · **stalled**: every 2026 file ends 2026-03-17 18Z, last modified 03-19 | 2026-03-17 | 2026-03-17 | 0 | year lane | **yes**, scheduled (no-op until PSL resumes) |
| `family7_2d/glorys025d` | GLORYS12 ocean reanalysis currents, SSH, mixed layer, daily, 0.25° | Copernicus Marine `cmems_mod_glo_phy_my_0.083deg_P1D-m` v202311 (STAC `end_datetime`); months to 2026-07 from the Hub's parked 0.25° chunks, the tail month subset from Copernicus Marine in the lane and parked | Copernicus Marine (Actions secrets) | weekly extension · ~6 weeks | 2026-08-18 | **2026-08-25** | 7 | year lane 2026 | **yes**, scheduled |
| `family7_2d/occci025d` | ESA OC-CCI v6.0 chlorophyll, daily, 0.25° | PML THREDDS `CCI_ALL-v6.0-DAILY` (10,501 days) | none | irregular · ~3.3 months | 2026-06-30 | 2026-06-30 | 0 | year lane (PML per-day subsets, ~8 s a day) | **yes**, scheduled |
| `family1_2/era5_{t,q,u,v}` | ERA5 temperature, humidity, wind on 13 pressure levels, six-hourly, 1° | Google's ARCO-ERA5 zarr `full_37-1h-0p25deg-chunk-1.zarr-v3` root attribute `valid_time_stop` (final ERA5); ERA5T to `valid_time_stop_era5t` 2026-10-02 not admitted (D10) | none | monthly (final ERA5) · ~3 months | 2026-06-30 | 2026-06-30 | 0 | year lane 2026 per store (~0.3 s a frame, ~5–8 min + push) | **yes**, scheduled |
| `family1_gf/pace4k` | NASA PACE OCI ocean colour and phytoplankton, daily, 4 km | OB.DAAC OPeNDAP (anonymous); CMR newest granule of `PACE_OCI_L3M_BGC` / `_AOP` / `L4M_MOANA` | none | daily · ~5 weeks | 2026-07-31 | **2026-08-31** | 31 | year lane 2026 (14.6–35 s a frame ≈ 1–2.4 h) | **yes**, dispatch-only until a real refresh is green, then batched ≥ 7 days |
| `family1_gf/sst_acspo02` | NOAA ACSPO L3S-LEO sea-surface temperature, daily, 2 km | PO.DAAC `L3S_LEO_DY-STAR-v2.81` (CMR C2805339147-POCLOUD) | Earthdata (Actions secrets) | daily · 1 day | 2026-09-16 | **2026-10-07** | 21 | last lane `d0701-0915` → `d0701-1231` (58 s a frame: ~1.6 h now, ~3 h by December; ~10 GB of parts) | **yes**, dispatch-only until green, batched ≥ 7 days |
| `family1_gf/irtb` | NCEP/CPC merged geostationary IR (cloud tops), 3-hourly, 4 km, ±30° | GES DISC `GPM_MERGIR.1` (CMR C1432254058-GES_DISC) | Earthdata | hourly files · 1 day | 2026-09-16 | **2026-10-06** | 20 | last lane `d0701-0915` → `d0701-1231` (13–18 min a quarter measured) | **yes**, dispatch-only until green, batched ≥ 7 days |
| `family1_gf/oc4k` | ESA OC-CCI v6.0 4 km daily | CEDA `v6.0-release/…/chlor_a/daily/v6.0/` — the tree ends at 2022 | none | **closed at this source** | 2022-12-31 | 2022-12-31 | 0 | — (2023 on is PML's subset: a new source for the adapter, a build decision, not a refresh) | — |

### 2.2 Points (tier P) and the family-10 stores — designed, not automated (§5)

| store | source | credential | cadence · lag | ours | upstream today | size | hosted re-assembly? |
|---|---|---|---|---|---|---|---|
| `family1_gf/bgcargo` | Argo GDAC synthetic-profile index (`data-argo.ifremer.fr`) | none | daily · 0 days (index of 2026-10-08) | 2026-09-17 | 2026-10-08 | 0.07 GB | yes |
| `family1_gf/oceansites` | OceanSITES GDAC index | none | daily | 2026-09-16 | (living) | 5.5 GB | yes |
| `family1_gf/wod` | NOAA World Ocean Database yearly files | none | quarterly | 2026-02-12 | (living) | 4.4 GB | yes |
| `family1_gf/xco2` | OCO-2 11.3r / OCO-3 11r / ACOS GOSAT Lite (GES DISC) | Earthdata | daily · 1–2 months | 2026-07-31 | OCO-2 2026-07-28, OCO-3 2026-08-31, GOSAT 2026-07-31 | 16.9 GB | yes (34 GB of disk) |
| `family1_gf/icoads` | ICOADS R3.0.2/3 NRT monthly (NCEI) | none | monthly · ~1 month | 2026-08-31 | (living) | 52.1 GB | **no** — pull + rewrite ≈ 104 GB; a dispatch box job |
| `family1_gf/swot` | SWOT L2 LR SSH Expert (PO.DAAC) | Earthdata | daily · 3 days | 2026-09-15 | 2026-10-05 | 301 GB | **no** — hosted lanes, box assembly (as built) |
| `family1_gf/glodap` | GLODAP v3 merged CSV (NCEI 0315582) | none | annual release | 2023-09-10 | (next release) | 0.08 GB | yes, on release |
| `family1_gf/swh` | ESA CCI Sea State v4 L3 | none | **closed** (STAC to 2023-12-31) | 2023-12-31 | 2023-12-31 | 55 GB | — |
| `family8_argo_l0` (`argo`) | Argo GDAC core profiles | none | daily | 2024-12-31 | (living) | 0.24 GB | yes |
| `family10_1/gdp` | AOML ERDDAP `drifter_6hour_qc` | none | quarterly-ish | 2024-12-31 | 2025-06-18 | 1.7 GB | yes |
| `family10_1/gtmba` | PMEL ERDDAP `pmelTaoDyT` | none | daily | 2024-12-31 | 2026-09-28 | 0.06 GB | yes |
| `family10_1/socat` | SOCAT, one release a year (v2026 in hand) | none | annual (June) | 2024-12-31 | v2026 | 1.5 GB | yes, on release |
| `family10_1/slatrack` | CMEMS L3 along-track altimetry (MY) | Copernicus Marine | ~monthly | 2024-12-31 | 2026-05-15 | 67 GB | **no** — a box assembly |
| `family10_2/fishing` | Global Fishing Watch v3 (Zenodo, 2025-03) | none | **closed** until a v4 | 2024-12-31 | 2024 | 19 GB | — |
| `family7_global025_pentad_l2` `g025/g100/oc025/rg100`, `clim/*`, E-086 normals | — | — | **frozen (D1)** | 2024-12-31 | — | — | — |

The family-10 stores end 2024-12-31 by design (the training window of the
tensor they sit beside); extending them changes what a family-10 training run
can read past 2024. The selection code windows by date, so earlier rows are
unchanged except where the producer revised them (Argo's delayed-mode
profiles arrive late) — a decision for the planning session before §5 is
built for them.

## 3 · The design, step by step (`ml/data_refresh.py`)

1. **plan** (no secret). Per store: the live registry entry → our end and
   our provisional days; the producer's probe → its newest day; NCEI → which
   preliminary days are now final. No new day, nothing final, no
   `--force-from` → **no-op**, no job. Otherwise the first day to re-fetch,
   and the lanes (D3).
2. **update** (one job per store). The lanes are run in process by the
   builder's own stages (`F1_RECORD_END` = the producer's newest day for the
   7.2d adapters, `F7D_UPSTREAM_FIRST=1` so the live year is read from PSL
   rather than the Hub's snapshot). Every falsifier the lanes already had
   runs unchanged (family 7.2d's source read-back on every post-2024 day; the
   pentad check before 2025; bounds; absences refuse the year).
3. **splice.** Self-check (D5); the kept rows + the lane's rows; store.json
   by `grid_store_meta` with the kept + new ledgers; the sha256 block (old
   entries kept, the lane's files hashed); the revision guard and the prefix
   rule (D6); every changed or new bin decoded tile by tile through
   `sharded.check_sharded` before anything is uploaded.
4. **commit + verify** (D7, D8); then the lane's parts to `partials/` (D15).
   A store whose re-fetched lane is byte-identical commits nothing.
5. **registry** (the commit point for the tab): rebuild all family-1
   registries (`build_family1_registry.py --check`), guard each refreshed
   family with `--refresh` (D9), publish; the reader check per store.
6. **sums** (D11): `export_gridded_monthly.update` downloads the published
   version (sha256 against the index), copies it (growing the year axis when
   a new year begins), recomputes the settled touched months with E-088's own
   `year_worker` (sha256-checked bins, `ShardedGroup.read_frame`), blanks the
   others, runs E-088's falsifier on every written plane, uploads to
   `monthly/r<UTC>/`, streams every file back; the finish job merges the new
   block into `data/gridded_monthly_index.json`, commits it to main and
   redeploys the site.
7. **status** (D12) and, on any failure, the issue.

### 3.1 What has to be rewritten when a partial last bin gains frames

| layout | what a new day rewrites | readers mid-read |
|---|---|---|
| sharded tier G (7.2d, 1.2, 1.gf grids) | the last bin's `.zst` and `.idx.npy` (same names), the group's `shard_index.npy`, `store.json`, `manifest.json`; `tile_grid.json` when the record end it states moves; new bins are new files | prefix rule (D6); one atomic commit |
| bin-major `.npy` (the family-7.2 tensor groups) | the whole file (the bin axis is the first axis of one array) | frozen (D1) |
| month-major (fishing grid) | the whole file | closed upstream |
| E-088 sums `[month, channel, year, lat, lon]` | the whole file (a new year changes the shape) | a new version folder (D11), the index is the commit point |
| tier P (rows sorted by time, CSR offsets) | every column file (rows append at the end, and a late row for an old bin shifts every later row) | a NEW versioned prefix and a registry flip (§5) |

## 4 · Per-run cost on a hosted runner (4 vCPU, 15 GiB, ~85 GB free, 6 h)

| store | lane | wall time | disk | Hub traffic |
|---|---|---|---|---|
| `oisst025d` | 2026, ~280 frames | ~10 min (two PSL files, 0.4 GB, 1.2 s a frame) | ~1 GB | ~10 MB of shards; the year's parts (~0.25 GB) |
| `glorys025d` | 2026, ~237 frames | ~30–45 min (7 Hub chunks, the tail month's daily subsets) | ~4 GB | the tail chunk (~0.2 GB); ~45 MB of shards; parts ~1.5 GB |
| `ncep100d` / `occci025d` | 2026 | ~5 min / ~30 min | < 2 GB | — while upstream is still |
| `era5_*` | 2026, 724+ frames each | ~5–8 min each + parts push | ~2 GB | ~1–2 GB of parts per store |
| `pace4k` | 2026 | 1–2.4 h | ~6 GB | ~6 GB of parts |
| `sst_acspo02` | Jul → today | 1.6 h (Oct) → 3 h (Dec) | ~15 GB | ~10 GB of parts |
| `irtb` | Jul → today | ~40 min | ~10 GB | ~5 GB of parts |
| E-088 sums | one store | download 4–19 GB, recompute ≤ a few months, upload 4–19 GB, stream back | ≤ 40 GB (GLORYS) | one new version per store per settled month |

The plan job and the no-op checks cost ~1–2 minutes a day.

## 5 · Point stores (tier P) — the design, not built

A point store is one file per column for the whole record (`time_s.npy`,
`lat.npy`, …) with per-bin row offsets. A new day appends rows; a late
observation for an old bin (Argo delayed mode) inserts rows and shifts every
later one. Rewriting under the same URLs would let a reader that read the
offsets before the commit read the wrong rows AFTER it, silently (columns are
raw arrays; any range reads). So the refresh of a point store is: re-run its
last lane(s) as a hosted fetch with `--push-parts`, re-assemble the whole
store on a hosted runner from the Hub's parts (`--parts-from-hub`), publish
it to a NEW versioned prefix `tensors/<family>/<store>/v<YYYYMMDD>/`, verify
(the builder's own restore check), and flip the registry's `path` (the
reader takes each store's folder from the registry, `relPath(reg, g.path)`,
so no reader change is needed); the previous version is deleted by the next
refresh. That needs the family-1 builder and registry builder to accept a
versioned prefix — the next step of this experiment. Stores that fit a hosted
runner (≤ ~35 GB, twice over): `bgcargo`, `oceansites`, `wod`, `xco2`,
`glodap` (on release), and family 10's `argo`, `gdp`, `gtmba`, `socat` (on
release). `icoads` (52 GB), `slatrack` (67 GB) and `swot` (301 GB) cannot be
re-assembled on a hosted runner: a `workflow_dispatch` box job, never
scheduled (D2).

## 6 · Dry runs

Every store's updater has a dry run: `python3 ml/data_refresh.py update
--store S --work DIR --dry-run` reports the lanes it would fetch and runs the
self-check (D5) — the index stage only, nothing written to the Hub. The
workflow's `dry_run` input runs it for every tier-G store asked for, current
or not (`plan --all-grid`), with no `HF_TOKEN` in the update jobs.

## 7 · The E-089 hook (the fine-grid sums)

E-089 (another agent, in progress) gives `oc4k`, `pace4k`, `sst_acspo02`
and `irtb` per-(year, month) native sum tiles plus an exact pooled 0.25°
layer, indexed by `data/gridded_monthly_fine_index.json`. Its native layer is
one file per month, so the refresh writes the touched SETTLED months' files
only (D11's rule), the pooled layer as a new version folder, and the index is
the commit point. `data_refresh.sums_hook_e089` is the place: it reports
"waiting" until both the index and an `update` entry point in
`ml/export_fine_monthly.py` are on main, then is wired there.

## 8 · Status (2026-10-08)

The run-by-run record is
[E-090 in the log](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-090).
In short:

- **Dry run of all eleven tier-G stores** (data-refresh #2): every
  self-check passed on the hosted runner, five stores had new days.
- **The first real refresh** (#3): `oisst025d` 2026-10-04 → **2026-10-06**
  (2 new days, 09-21 and 09-22 replaced by NCEI's final values, 3 bins
  rewritten, Hub commit `f089598d`) and `glorys025d` 2026-08-18 →
  **2026-08-25** (7 new days, the August tail from Copernicus Marine, commit
  `fc69e5c3`); every committed file read back, the new frames re-read; the
  registry published under `--refresh`; the tab's reader read both newest
  days equal to `sharded.py`, 0 differences in 9,801 cells each.
- **ERA5 forced** (#4): `era5_t`'s 2026 lane re-fetched byte-identical to
  the published store — nothing committed. ARCO-ERA5's final stream has not
  moved past 2026-06-30, so ERA5's append path waits for its next monthly
  extension.
- **Sums** (#5, the recovery of #3's failed sums job): OISST's September
  2026 blanked into `monthly/r20261008T0820/`, index committed by the job.
- **Schedule** enabled after #5: daily 05:23 UTC, the eight `auto` stores.

## 9 · What is NOT verified

- The point-store refresh (§5) is a design.
- `sst_acspo02`, `irtb` and `pace4k` have passed their dry runs only; their
  first real refresh (and the named-lane replacement it performs) is a
  dispatch, not yet run.
- ERA5's append path has not run on new data: ARCO-ERA5's final stream still
  ends 2026-06-30 (D10); its forced byte-identical refresh is the evidence
  until the next monthly extension.
