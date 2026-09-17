# E-082 · Building families 1.gf, 1.0.tf and 0.9.tf — phase A

*Written 2026-09-17, before anything is fetched. Chris, on reading the three
design notes: "Great work. Let's start building these families. Let me know
what credentials / accounts you need." This is the build plan: what is built
in which order, on which machine, from what code, and what each step must
measure before the next is dispatched. The design itself is in the three
notes and is not restated here.*

- [Family 1.gf](https://blauewelt.github.io/earth/ml/paper/notes/family1gf.pdf) — the global fine-granularity observation family (≤ 10 km, ≤ 5 days): ocean colour, level-3 sea-surface temperature, sea ice, SWOT swaths, cloud-top temperature, ship reports, ocean profiles, carbon bottles, floats, moorings, column CO₂.
- [Family 1.0.tf](https://blauewelt.github.io/earth/ml/paper/notes/family1tf.pdf) — land and coast at the same scale: station networks, tide gauges, buoys, radar, gliders, flux towers, fires, rivers, orbital LiDAR, daily 0.05° land fields, forty years of imagery as a scene catalogue, and the land-change targets.
- [Family 0.9.tf](https://blauewelt.github.io/earth/ml/paper/notes/family09tf.pdf) — the same land and coast through Google's published AlphaEarth embedding instead of the pixels.

## 1 · What is reused, and what is new

**Reused: the family-10 builder.** `ml/build_family10_stores.py` already does
everything a tier-P store needs — an adapter interface (`index`,
`fetch_year`, optional preflight and extra files), per-year column parts
written by `PartWriter`, resume by year, the streaming assembler, the
restore-verified publish, `check_store`, and a synthetic smoke per adapter.
Family 1's point stores are new **adapters** against that interface, not a
new builder. The generic stages are parametrised, with defaults that leave
family 10's behaviour and its tests byte-identical:

| what | family 10 today | family 1 |
|---|---|---|
| Hub prefix | `tensors/family10_2/<store>` | `tensors/family1_gf/`, `tensors/family1_tf/`, `tensors/family09_tf/` |
| parts prefix | `partials/family10_1/<store>/<year>` | `partials/family1_<x>/<store>/<year>` |
| cache | `ml/cache/family10_2` | `ml/cache/family1_<x>` |
| repo | `chfrank/earth-tensors` | the same, or `chfrank/earth-tensors-private` when the adapter's `distribution` is `private` |
| `time_s` | int32 (schema 2) | int32, or **int64 (schema 3)** when the adapter's `first_year` is before 1914 (`icoads`, `wod`, `ghcnd`, `igra`, `tide`, `river`, `ghcnh`, `ecad`) |

`ml/family10_store.py` learns schema 3 (the reader checks `store.json`'s
`time_dtype`); nothing else about a row changes.

**New: three things the notes designed that family 10 lacks.**

1. **The sharded tier-G layout** (`ml/family1/sharded.py`) for grids finer
   than 0.25°: one zstd shard per (group, five-day bin), `frames_per_bin`
   frames inside, 256 × 256 tiles, an int64 `(offset, length)` index per
   tile, `tile_grid.json`, and a reader that returns one tile of one frame
   by two range reads. Registry: `tier: "G"`, `layout: "sharded"`.
2. **The probe stage** (`--stage probe`): one calendar month through the
   real adapter on a hosted runner, writing `probe.json` — rows per day,
   distinct platforms, bytes fetched, seconds per month, and for gridded
   stores the valid fraction and compressed bytes per tile. Every size in
   the notes is an estimate until this file replaces it, and no full build
   is dispatched without one.
3. **The two-track publish.** An adapter declares `distribution =
   "private"` and `licence = {"redistribution": "no", "derived_works":
   ...}`; the publish stage then targets the private repository and refuses
   the public one by construction, and the registry lists the group with its
   prefix so the design stays public.

## 2 · Machines and credentials

Unchanged from E-079: **credentials reach hosted runners only** (repository
secrets); **rented boxes are keyless** and assemble from Hub parts. The one
addition, from family 1.gf §4: a box that assembles a *private* store holds
`HF_PRIVATE_READ_TOKEN` — a fine-grained token that can read only
`chfrank/earth-tensors-private` — for the job, and the job's hygiene deletes
the pulled parts. The private repository exists (created 2026-09-17,
anonymous read returns 401).

| credential | secret name | stores it unblocks | status |
|---|---|---|---|
| Hugging Face write | `HF_TOKEN` | every publish | in place |
| Copernicus Marine | `COPERNICUSMARINE_SERVICE_*` | `swh` (1.gf) | in place |
| NASA Earthdata Login | `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | `lst05`, `snow05`, `refl05`, `gedi`, `icesat2`, `fire` (archive), `cat_hls`, `cat_viirs`, `cat_ecostress`, `alerts` (OPERA), `burned500`, `cat_gfm` (DSWx), `ims1k`, `swot`, `irtb`, `xco2`, IMERG | **asked 2026-09-17** |
| FIRMS map key | `FIRMS_MAP_KEY` | `fire` (API) | asked |
| Hugging Face private read | `HF_PRIVATE_READ_TOKEN` | box assembly of private stores | asked |
| FLUXNET / AmeriFlux | `FLUXNET_USERNAME` / `FLUXNET_PASSWORD` | `flux` | asked |
| ISMN | `ISMN_USERNAME` / `ISMN_PASSWORD` | `ismn` (private, phase B) | asked |
| GRDC portal | (account; request by form) | `river_grdc` (private, phase B) | asked |
| Bremen sea-ice group | (an email) | decides `seaice_asi`'s track | asked |
| AWS IAM (optional) | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | `aef1k` in-region build | optional |

Needs **no account** (verified against the archives on 2026-09-16, the
listing endpoints on 2026-09-17): NCEI (GHCN-Daily, IGRA, ICOADS, WOD),
NDBC, GESLA/UHSLC, the IOOS and OceanGliders glider archives, GLODAP, the
Argo and OceanSITES GDACs, CEDA (OC-CCI, SST CCI), NOAA CoastWatch (ACSPO),
Bremen (sea ice), Hansen GFC on Google Cloud Storage, GEBCO, ESA WorldCover on
AWS, CHIRPS, HFRNet, the source.coop mirror of the AlphaEarth embedding, and
— for *listings* only — the CDSE STAC (Sentinel-1/2/3), the USGS STAC
(Landsat), NASA CMR (HLS, VIIRS, ECOSTRESS) and ASF (NISAR).

## 3 · Order of work

Each wave is dispatched only after the previous one's probe has been read.
Store names are the notes'; the family is in brackets.

**Wave 1 — the framework, and the keyless point stores.** The parametrised
builder, schema 3, the probe stage, the workflow, and adapters for
`ghcnd`, `ndbc`, `igra`, `tide` + `tide_private`, `gliders` [1.0.tf] and
`icoads`, `wod`, `glodap`, `bgcargo`, `oceansites` [1.gf]. Each adapter
ships with a smoke test (synthetic source, exact expected counts) before its
probe runs. Probes on hosted runners; full builds of the small stores
(`glodap`, `oceansites`, `bgcargo`, `gliders`, `igra`) on hosted runners
too; `ghcnd`, `ndbc`, `tide`, `icoads`, `wod` fetch per year on hosted lanes
and assemble on one verified box, the `slatrack` pattern.

**Wave 2 — the sharded layout and its two keyless grids.** `seaice_asi`
[1.gf] (6.25 km polar stereographic, daily, `F = 5`) is the smallest real
sharded store and the one the layout is proven on; `lossyear` [1.0.tf]
(Hansen 30 m annual, sparse) is the first annual target. Then `oc4k` and
`sst_cci05` [1.gf] from CEDA, `sst_acspo02` from CoastWatch, `chirps05`.

**Wave 3 — the scene catalogues** [1.0.tf]: eight anonymous listings
(`cat_landsat`, `cat_s2`, `cat_s1`, `cat_hls`, `cat_nisar`, `cat_olci`,
`cat_viirs`, `cat_ecostress`), one row per scene-date with the sensor code
table and the `assets.parquet` sidecar; each probed against the producer's
own count for one month.

**Wave 4 — the credentialed stores**, as the secrets land: `lst05`,
`snow05`, `fire`, `flux`, `burned500`, `alerts`, `cat_gfm`, `static_fine`
[1.0.tf]; `swot`, `irtb` (tropics, three-hourly first), `xco2`, `swh`
[1.gf]. Then the three registries, and the `siblings` line in family 10.2's.

**Wave 5 — 0.9.tf.** One UTM zone-year of the embedding through the pooling
code first (throughput, cost, unit-length and clip checks), then `aef1k`
in-region on AWS spot if keys arrive, else on a US Vast box; `aef_dots`
beside it.

Phase B (GEDI, GHCNh, HFR, rivers, ISMN, the 0.05° reflectance, PALSAR) and
phase C wait for phase A's verdicts, as the notes say.

## 4 · What every store must show before it is called built

Unchanged from E-079/E-081: `check_store` on the published files after a
restore (sha256 of every array against `store.json`; `N` from the file
sizes; `bin` monotone and equal to `floor_divide(time_s, 432000)`; the CSR
index reproduced; per-year counts equal to the fetch ledger); the probe's
numbers written into the registry beside the note's estimate; `verdict`
empty until the falsifier of the note has run. A day absent upstream is a
zero-length frame with a counter, never a silent gap; a fetch lane refuses
an empty listing or a short download (ml/CLAUDE.md, the 2026-09-14 rule).
For a private store, additionally: the publish target is the private
repository (asserted by the URL, not the intention), and the box's
post-job listing shows no part left on disk.

## 5 · Cost

Hosted runners: $0. Boxes: family 10.1's measured rates put the phase-A
assemblies at $10–20 across both families (the largest, `ghcnd` at ≈ 40 GB
and `icoads` at ≈ 60 GB, each one verified-host job of 1–3 hours). 0.9.tf's
pooling: $50–150 in-region. Hub storage: public within the PRO 10 TB; private
tens of GB in phase A, within the plan.
