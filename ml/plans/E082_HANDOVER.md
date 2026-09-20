# E-082 handover — building families 1.gf, 1.0.tf and 0.9.tf (state on 2026-09-20 18:30Z)

*For any agent picking this up. E-082 is the build of three new tensor
families designed on 2026-09-16: **family 1.gf** — global ocean and
atmosphere observations at ≤ 10 km and ≤ 5 days; **family 1.0.tf** — land and
coast at the same scale; **family 0.9.tf** — 1.0.tf with the raw satellite
imagery replaced by Google's AlphaEarth embedding, everything else inherited
by reference. Wave 6 (2026-09-20) added the biosphere — forests, photosynthesis,
species — to all three (`ml/plans/E082_biosphere_wave.md`). Read this first,
then `ml/plans/E082_family1_builds.md` (the plan), `ml/family1/BUILD_LOG.md`
(every landing, every measured quirk, in detail — the wave-6 section at the
end is the newest), `ml/family1/ADAPTER_CONTRACT.md` (the code interface), and
`ml/CLAUDE.md` §0–§7 (the working rules; §0b says the main session plans and
Opus sub-agents implement).*

## 0 · How the last two days went, in one paragraph

The Opus quota outage of 2026-09-19 is over. On 2026-09-19/20 a hosted operator
finished the four catalogues and `fire`, a parallel session assembled `oc4k`,
`swh` and (in flight) `lst05` on the box, and on 2026-09-20 four Opus agents
landed the whole biosphere wave — fourteen adapters, two framework fixes, the
three registries — in one afternoon, with three probes run from the sandbox
and five on hosted runners. Nothing in this file is taken from an agent's
report alone: every "built" below is a `store.json` read from the Hub, and
every probe number is a committed `ml/family1/probes/<store>_<month>.json`.

## 1 · What is on the Hub (verified from `store.json` on 2026-09-20)

Public repository `chfrank/earth-tensors`, `tensors/family1_gf/` and
`tensors/family1_tf/`; private repository `chfrank/earth-tensors-private`
(anonymous read = 401). `python3 ml/build_family1_registry.py --check` prints
this table live (built / probed per store) and writes the three registries.

| store | family | N (rows or frames) | stored | record | notes |
|---|---|---|---|---|---|
| `glodap` | 1.gf | 1,460,215 bottles | 77 MB | 1972–2023 | GLODAPv3 |
| `wod` | 1.gf | 15,281,373 casts | 4.39 GB | 1772–2026 | schema 3, C = 128 |
| `icoads` | 1.gf | 1,107,951,670 reports | 52.07 GB | 1662–2026 | schema 3 |
| `bgcargo` | 1.gf | 335,231 profiles | 73 MB | 2002–2026 | |
| `oceansites` | 1.gf | 66,483,202 rows | 5.52 GB | 1980–2026 | tropical arrays excluded |
| `seaice_asi` | 1.gf, tier G | 5,187 daily frames × 2 hemispheres | 1.66 GB | 2012-07 → 2026-09 | **private** with `licence_pending` until Bremen answers |
| `oc4k` | 1.gf, tier G | 1997-09 → 2022-12 | ≈ 160 GB | | assembled on the box, run #287 |
| `swh` | 1.gf | 1,680,274,586 samples | see BUILD_LOG | 1991-08 → 2023-12 | run #290; 1 % above the probe's projection |
| `ghcnd` | 1.0.tf | 1,143,728,366 station-days | 46.91 GB | 1763–2026 | schema 3 |
| `igra` | 1.0.tf | 44,842,766 soundings | 8.57 GB | 1905–2026 | schema 3, C = 80 |
| `tide` | 1.0.tf | 1,080,914,415 readings | 35.67 GB | 1800–2026 | 52 Great Lakes gauges excluded by the ±50 m bound (decision open) |
| `tide_private` | 1.0.tf | 179,314,808 | 5.92 GB | 1821–2026 | **private** |
| `ndbc` | 1.0.tf | 628,157,329 | 29.52 GB | 1970–2025 | |
| `gliders` | 1.0.tf | 2,808,590 profiles | 436 MB | 2003–2026 | |
| `chirps05` | 1.0.tf, tier G | 3,288 pentad frames | 19.06 GB | 1981-01 → 2026-08 | exception E4 |
| `lossyear` | 1.0.tf, tier G | one bin, 280 Hansen land tiles as 2.5° groups | see BUILD_LOG | 2001–2025 loss year + tree cover 2000 | built on the box, run #237 |
| `fire` | 1.0.tf | 631,836,727 detections | 22.11 GB | 2000-11 → 2026-09 | hosted lanes + assembly, run #291 |
| `cat_landsat` | 1.0.tf, tier T | catalogue | | 1982 → | |
| `cat_s1` | 1.0.tf, tier T | 6,203,597 scenes | | 2014-10 → | |
| `cat_s2` | 1.0.tf, tier T | 41,197,086 scenes | 2.91 GB | 2015-01 → 2026-09-15 | top-up lane 2026-09-16..30 still owed |
| `cat_hls` | 1.0.tf, tier T | 37,951,341 granules | 1.94 GB | 2013-04 → 2026-09 | |
| `cat_ecostress` | 1.0.tf, tier T | 19,525,578 granules | 1.01 GB | 2018-07 → 2026-09 | |
| `cat_nisar` | 1.0.tf, tier T | 138,804 scenes | 8.9 MB | 2025-10 → 2026-09 | |
| `cat_viirs` | 1.0.tf, tier T | 2,341,071 scenes | 128.8 MB | 2012 → 2026-09 | |
| `cat_olci` | 1.0.tf, tier T | 1,591,637 scenes | 126.8 MB | 2016-04 → 2026-09 | C = 6 with a `coastal` flag |

Twenty-six stores.

## 2 · Parked or in flight

| store | family | state on 2026-09-20 18:30Z | what remains |
|---|---|---|---|
| `lst05` | 1.0.tf, tier G | assembly **in flight** on the box, run #292 (`stage=all --parts-from-hub`, 1999 → 2026-09) | verify its `store.json`, add the row |
| `snow05` | 1.0.tf, tier G | parts 1999–2026 on the Hub | assembly on the box after `lst05` |
| `pheno500`, `lai500`, `pace4k`, `gedi`, `icesat2` | wave 6 | hosted **probes** #293–#297 dispatched 17:39Z | read `probe.json` from each run's artefact, write the numbers into BUILD_LOG and the registry, then decide phase-A builds (the thresholds are in BUILD_LOG "wave 6") |

The exact assembly form for anything parked: `stage=all` with
`extra_args: "--parts-from-hub --assemble streaming"` and explicit `start`/`end`
(`stage=assemble,publish,check` fails on a missing `fetch.done`).

## 3 · The box

Vast instance **51415980** (RTX 3060 host, $0.131/h, runner name
`gpu-box-31947967`) is RUNNING and busy with #292 (`lst05`). After `snow05`,
the wave-6 assemblies that need a box are `canopy30` (one bin, 261 tiles,
11–21 h, ≈ 33 GB) and, on their probes' numbers, one year each of `gedi` and
`icesat2` (assembly above ≈ 45 GB of store), `sif` (46.8 GB a year) and
`gbif` (101 GB public + 28 GB private). Stop the box when nothing is queued
for it; destroy it and rent a verified box (≥ 400 GB disk, ≥ 64 GB RAM,
≥ 1 Gbps up) if `start` sits `offline` for ten minutes and the host is absent
from `offers` (ml/CLAUDE.md §7). The Vast key is at `/home/claude/.vast_key`.

## 4 · Not built, and the measured reason

- **`sst_acspo02`**: probed — ≈ 501 GB for the record, one year ≈ 19 GB in two six-month lanes; the adapter refuses more than one year without `SST_ACSPO02_MAX_YEARS`. Decision: build 2020 first.
- **`swot`**: probed; storage decision in BUILD_LOG ("swot: the storage decision") — tier P 118 GB/yr vs per-pass tier G 26 GB/yr. Decision for Chris; refuses past `probe` without `SWOT_ALLOW_BUILD=1`.
- **`irtb`**: adapter + smoke; the layout's frame mask is uint64, so two-hourly (F = 60) is the finest, not half-hourly; probe #178 FAILED on the unapproved **GES DISC** application.
- **`xco2`**: adapter + smoke; probe #179 FAILED for the same reason. **One click by Chris** (approve the GES DISC application in the Earthdata profile) unblocks `xco2`, `irtb` and the OCO half of `sif`.
- **`flux`**: FIXED 2026-09-20 (longitude-derived UTC offsets; `--allow-missing-years` honoured for one-stream stores). Dispatch `stage=all`.
- **`burned500`**: adapter landed 2026-09-20 with the `no_frame_in_bin` skip reason; 46 nine-tile groups. Probe, then build (≈ 10 GB).
- **`alerts`**: DIST-ANN (annual) route first; not written yet.
- **`cat_gfm`**: the Copernicus flood service has no global listing. Build the OPERA DSWx half only, or drop the GFM column.
- **`static_fine`**: designed and measured, not built (BUILD_LOG says why). Decision for Chris.
- **`refl05`**: 2.47 TB for the Terra record. Decision: 2015–2026 only (≈ 900 GB) or nothing.
- **`sst_cci05`**: no v3 on CEDA; phase B/C.
- **`ismn`, `river_grdc`, `ecad`** (private track, phase B): ISMN credentials in place; GRDC needs Chris's request form.
- **Family 0.9.tf's own stores** (`aef1k`, `aef_dots`, `aef100r`): nothing started; first step is one UTM zone-year through the pooling code.
- **`cat_palsar`**: the ScanSAR half lists anonymously and is in the adapter; the 25 m mosaics need a JAXA G-Portal account (Chris).
- **`biomass100`**: designed (phase A for 2020 + GEDI L4B); no adapter yet.

## 5 · The biosphere wave — where it stands

Design: `ml/plans/E082_biosphere_wave.md` (rows, phases, decisions);
measurements and contradictions: BUILD_LOG "Notes — E-082 wave 6" (two
sections). Adapters landed with smokes: `sif`, `gbif` + `gbif_nc`, `gedi`,
`icesat2`, `canopy30`, `pheno500`, `lai500`, `pace4k`, `burned500`,
`canopy_ref`, `cat_palsar`, `cat_biomass_esa`. Probes run: `sif` (46.8 GB a
year uncut), `gbif` (101 GB public / 28 GB private), `canopy30` (≈ 33 GB, a
box); `pace4k` measured on three days (≈ 9.3 GB a year, no credential).
Probes in flight: #293–#297. Not written: `biomass100`, `alerts` (DIST-ANN).

**Phase-A builds to dispatch once the probes are read** (all `stage=all`,
one lane per year, hosted unless noted): `pace4k` 2024, 2025, 2026 (fits); `sif`
2018–2026 as nine year-lanes then a box assembly; `gbif` / `gbif_nc` by part
range (`GBIF_PARTS`) then a box assembly (both tracks from ONE snapshot —
`GBIF_SNAPSHOT`); `canopy30` on the box; `pheno500` 2001–2025 one year a lane
if ≤ 68 s a frame; `lai500` Terra 2020–2024 if ≤ 1.6 s a frame; one year
(2022) of `gedi` and `icesat2` as monthly lanes if the byte-range read
fraction clears the thresholds in BUILD_LOG, assembly on the box;
`burned500` and `flux` after their probes.

## 6 · Credentials and people

Repository secrets in place: `EARTHDATA_USERNAME/PASSWORD` (verified against
LP DAAC, PO.DAAC, OB.DAAC; GES DISC **unapproved**), `FIRMS_MAP_KEY`,
`FLUXNET_USERNAME/PASSWORD`, `ISMN_USERNAME/PASSWORD`, `HF_PRIVATE_READ_TOKEN`,
plus `HF_TOKEN`, `COPERNICUSMARINE_*`, `GFW_API_TOKEN`, Cloudflare, GCP.
Sandbox files (recreate with the Write tool from the project docs
`claude/github-access.md`, `claude/huggingface-access.md`,
`claude/vast-access.md`): `/home/claude/.gh_pat`, `/home/claude/.hf_token`,
`/home/claude/.vast_key`.

Waiting on Chris: the **GES DISC approval** (one click); a **JAXA G-Portal
account** for the PALSAR mosaics; the **GRDC** request form; the **Bremen**
email (decides whether `seaice_asi` is re-published publicly); the **Great
Lakes** decision for `tide`; the `swot`, `refl05`, `sst_acspo02`,
`static_fine` decisions; GBIF's non-commercial records (private track as
built, or drop); the CCI Biomass track (public with attribution is the pick).

## 7 · How to operate

- Dispatch: `node scripts/family1_dispatch.mjs family1-build.yml '<inputs json>'` — inputs `store`, `stage`, `start`, `end`, `runner`, `extra_args`, `probe_month`, `check_credentials`, `adapter_env`. **`check_credentials:"true"` runs ONLY the Earthdata check and no probe or build.** **A fetch lane parks nothing unless `extra_args` carries `--push-parts`** (fifteen lanes were dispatched without it on 2026-09-20 and had to be cancelled and re-dispatched). **A lane is one YEAR or more, never part of a year**: the parts layout is one folder per year with one index, one ledger and one `done.json`, so two lanes writing the same year overwrite each other's index and the assembler would build a short store (ICESat-2's monthly lanes of 2022-09-20, cancelled). Splitting a year across lanes needs the lane-aware parts layout (below) first.
- **The laser stores' granule cap is the probe's, never the build's** (`GEDI_MAX_GRANULES` / `ICESAT2_MAX_GRANULES`, defaults 6 / 12): a build reads every granule in its window, and a cap set by name for a build refuses unless `*_ALLOW_CAPPED_BUILD=1`. The first ICESat-2 lanes ran under the probe's default cap, read twelve granules of 4,884 and marked 2022 done in 38 seconds — fixed the same evening, with a test.
- Watch: `node scripts/family1_runs.mjs [--n 40] [--status in_progress|queued]`; parked parts: `node scripts/family1_runs.mjs --parts <family> <store>`.
- Verify a landing by its `store.json` on the Hub, never by the run's colour; then add the BUILD_LOG.md row and re-run `python3 ml/build_family1_registry.py --check`.
- Concurrency: ~20 jobs for the whole account; lanes with the same store are in one concurrency group keyed by window, so a new lane with the SAME window cancels the pending one.
- Hosted runners: 6 h, ~86 GB free, 16 GB RAM (the tier-G probe's float64 bounds check needs ~6 GB at C = 7); assemblies above ~60 GB of parts go to a box.
- Tests: `python3 -m pytest -q tests/test_build_family10_stores.py tests/test_build_family1_stores.py tests/test_family1_*.py` (607 passing, 1 skipped at commit 0d30803); `tests/test_workflow_config.py` case 3 fails on an unrelated ml-train recipe and did before E-082.
- Push: `git pull --rebase origin main` then `node scripts/git_api_push.mjs --branch main --token-file /home/claude/.gh_pat`; gh-pages by a `force:false` PATCH. A parallel session may be writing to main — always rebase first.

## 8 · Order of work for the next agent

1. Probes read and dispatched (see BUILD_LOG "the hosted probes"): `pheno500` 2014–2025 year-lanes (#340–#351) and `pace4k` 2024–2026 (#352–#354) are fetching with `--push-parts`; assemble each with `stage=all --parts-from-hub --assemble streaming` when its lanes are green (pheno500 2001–2013 lanes after that); `gedi` probe #315.
1b. **The lane-aware parts layout** (an Opus task, specified in BUILD_LOG "wave 6" and the handover §7): `partials/<family>/<store>/<year>/<lane>/` with the lane named by its window or its group subset, each lane its own parts, index, ledger and `done.json`, and an assembler that merges the lanes of a year, asserts their bins or groups are disjoint, sums their ledgers and refuses when a declared lane is missing. It unblocks `icesat2` 2022 (twelve monthly lanes of three hours), `gedi`, `lai500` (four quarter-lanes a year) and `canopy30` / `lossyear` tile lanes on hosted runners. Until it lands, those stores are box jobs or wait.
2. When `lst05` (#292) verifies: `snow05` assembly on the box, then `canopy30` (`stage=all`, runner `gpu-box-31947967`), then the wave-6 assemblies as their lanes finish; BUILD_LOG rows from each `store.json`.
3. `flux` `stage=all` (hosted); `burned500` probe then build; the `cat_s2` top-up lane 2026-09-16..30 and re-assembly.
4. Write `biomass100` (CCI v7 2020 + GEDI L4B) and `alerts` (DIST-ANN) adapters — Opus.
5. After the GES DISC approval: re-run probes #178/#179 (`irtb`, `xco2`) and the OCO half of `sif` (`SIF_OCO=1`).
6. Publish the three registries to the Hub (`tensors/<slug>/<name>.json`) once the phase-A stores of this wave are on it.
7. Keep `claude/e082-build-status.md` in the project current with each landing.
