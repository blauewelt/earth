# E-077 · Family 7.1 — ocean colour joins the global tensor as a fourth group (recipe `f7l1`)

*Written 2026-09-11. Decision by Chris the same day: "let's do some averaging for
now (and provide a new family7 file that I can download)". This is the spec the
`occci` stage of `ml/build_family7.py` is written to; every rule below is a
build-time assertion or a documented choice.*

**In one sentence.** Family 7.1 is family 7 — the first input tensor covering
the whole globe pole to pole, 0.25°, five-day bins 1982–2024, 54 channels in
three native-resolution groups — plus a fourth group `oc025` holding the
observed surface chlorophyll-a concentration from the ESA Ocean Colour Climate
Change Initiative (OC-CCI, the merged SeaWiFS / MERIS / MODIS / VIIRS / OLCI
record, 4 km, 4 Sep 1997 → 31 Dec 2024), box-averaged to the same 0.25° point
grid and binned to the same pentads. The three existing groups are **not
rebuilt and not changed**: they are hard-linked from the family-7 build and
are byte-identical to what is on the Hub as `f7l0`.

## 1 · Why (and why now, and why this way)

- E-072 (the Earth foundation model plan) lists *surface colour* as one of the
  six read-outs the programme has promised, and E-070 §"Surface colour" decided
  colour enters first as a **decoder target and probe** — does the physical
  embedding predict next-season chlorophyll? — before it enters as an input.
  Neither is possible without a colour channel on the tensor's grid. This is
  that channel.
- "Averaging for now": the 4 km product is box-averaged to 0.25°. That throws
  away a factor of 36 in pixels, deliberately, so that the channel drops into
  the existing three-group layout with no new machinery. The design that keeps
  fine-grained channels at their own granularity is a separate note
  (`ml/plans/E078_multi_granularity.md`, in preparation) — this build must not
  wait for it.
- A **new recipe, a new folder on the Hub.** `f7l0`'s files, hashes and
  handover are cited by other agents (the Gemini handover); rewriting its
  `.npz` to say four groups would change a published dataset under its own
  name. So 7.1 is `f7l1`, stem `family7_global025_pentad_l1`, its own folder,
  its own manifest — and its three inherited group files hash-identical to
  `f7l0`'s, which the manifest states and the build asserts.

## 2 · Source

**Primary — OC-CCI v6.0, geographic projection, daily, chlorophyll-a only.**
Two hosts, and **MEASURED 2026-09-14 from a GitHub-hosted runner** (workflow
`probe-urls.yml`, runs 34823048523 and 34824765373) rather than assumed. The
CEDA archive (`dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/v6.0-release/
geographic/netcdf/chlor_a/daily/v6.0/<YYYY>/`) serves one file per day and its
directory listing runs **1997–2022**; 2023 and 2024 are 404 there. Plymouth
Marine Laboratory serves **no per-year 4 km file directory at all** — every
`thredds/catalog/cci/v6.0-release/…` path, chlor_a or all_products, any year
including 2022, answers HTTP 404, so the earlier sentence here claiming PML's
per-year copy runs to the end of 2024 was wrong and the two `pml-thredds*`
entries it produced have been removed from `OC_SOURCES`. What PML does serve
(catalog `https://www.oceancolour.org/thredds/catalog-cci.xml`) is ONE
aggregated dataset per cadence, `urlPath="CCI_ALL-v6.0-DAILY"`, with OPENDAP,
HTTPServer and NetcdfSubset services. Its time axis is days since 1970-01-01
(`.dds` → `Int32 time[time = 10501]`; `.ascii?time[0:1:1]` → `10108, 10110`,
i.e. 1997-09-04 and 1997-09-06; the axis ends at 20154 = 2025-03-07), and a
day absent from that axis is *missing* in exactly the sense a file absent from
a CEDA directory is. A per-day NetcdfSubset request returns a real netCDF4
file: `…/thredds/ncss/grid/CCI_ALL-v6.0-DAILY?var=chlor_a&time=2023-01-01T00:
00:00Z&accept=netcdf4` → HTTP 200, `application/x-netcdf4`, HDF5 signature,
**20,118,883 bytes in 7.5 s**; 2024-12-31 → **19,659,419 bytes in 9.6 s**.
So **CEDA stays FIRST** — 1997–2022 keep coming from the same per-file archive
the already-published partials used, and provenance does not change — and
2023–2024 fall through to `pml-ncss`. Two consequences for the stage: the NCSS
response is GENERATED, so it declares no Content-Length and cannot be
size-verified — it is verified by being opened and read for a `chlor_a` field
instead — and it is a subset, so the variable arrives as
`chlor_a(time=1, lat, lon)`, which `oc_open` reduces to the same two
dimensions CEDA's `chlor_a(lat, lon)` gives. The file
name pattern is expected to be
`ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-<YYYYMMDD>-fv6.0.nc`
(~30–60 MB each), **but the stage must not assume it**: rule from the root
CLAUDE.md — *never guess what an archive serves, ask it*. The stage's `index`
step lists each year's directory (THREDDS `catalog.xml`, or the CEDA `?json`
listing) and records the file names it will fetch, exactly as the family-8
builder indexed the Argo GDAC before fetching. The sandbox's egress proxy
blocks both hosts (measured 2026-09-11: `CONNECT tunnel failed, 403`), so this
listing can only be verified from the box — which is why the **preflight
fetches one real file** (any day of 2015) and opens it before anything is
spent.

Variable: `chlor_a` (mg m⁻³), float32 with a fill value, grid `lat[4320]`
descending from +89.979 to −89.979 (cell centres at odd multiples of 1/48°)
and `lon[8640]` from −179.979 to +179.979. Also read, for the record only:
`chlor_a_log10_rmsd` is NOT stored (an uncertainty of a merged algorithm is
not an observation; revisit when colour becomes an input).

**Fallback, only if the preflight fails on both hosts:** Copernicus Marine
GlobColour L3 multi-sensor 4 km daily
(`cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D`, variable `CHL`), with
the CMEMS credentials from `claude/copernicus-marine-access.md` as
environment variables. Same grid family, same averaging rule. This is a
different merge (GlobColour's, not the CCI bias-corrected chain), so it is a
**decision for the session, not an automatic fallback** — the stage fails at
preflight and says which host refused; a human (or the planning session)
re-dispatches with `--oc-source globcolour`.

## 3 · Channels (group `oc025`, 0.25° point grid, pentad bins)

| idx | name | unit (after un-z-scoring) | rule |
|---|---|---|---|
| 0 | `log_chl` | log10(mg m⁻³) | For each 0.25° point, the 6 × 6 block of 4 km cells whose centres lie within ±0.125° of the point (exactly six per axis, because a point at a multiple of 0.25° sits on a cell boundary of the 1/24° grid; at the two pole rows the block is truncated, and it wraps in longitude). Per DAY: mean of `log10(chlor_a)` over the finite cells of the block (chlorophyll is log-normal — E-072 §3 already scores colour in log space — so the mean is taken in log space, never on the raw concentration). Per PENTAD: mean of the daily block means over the days that had ≥ 1 finite cell. NaN if no day did. |
| 1 | `chl_cov` | fraction in [0, 1] | The coverage: finite 4 km cell-days in the block during the pentad ÷ (cells in the block × days in the bin). Says how much a `log_chl` value rests on — one clear pixel on one day, or thirty-six pixels on five. Stored un-logged; z-scored like everything else. |

**NaN means what it means everywhere in family 7 — "not observed here at this
time"** — and for colour it is the rule rather than the exception: cloud,
sun glint, sea ice, polar night, and land. Before bin 1145 (the bin containing
1997-09-04) there is no observation at all. `chl_cov` is NaN exactly where
`log_chl` is (a block that saw nothing has no coverage to report; do not store
0 there, because the consumer's "0 vs NaN" distinction is the missing-token
design — see the family-7 handover §5 on `sea_ice`).

**Not merged, not filled.** No gap filling, no L4 product, no climatological
fill: the L3 observed field is the only honest one, and E-071 §6.1's rule
("reflectance is the input, colour the target") wants the target to be what
was seen.

## 4 · Layout

`family7_global025_pentad_l1_X_oc025.npy` — shape **`[T_oc, 721, 1440, 2]`**
float16, C order, 128-byte NumPy header, z-scored per channel exactly as the
other groups (`norm_oc025 [2, 2]` in the npz). Two layouts are admissible;
the builder picks the first the **consumers already support**:

1. **Preferred — an offset axis:** `T_oc` = 3142 − 1145 = 1997 bins, the row
   for bin `b` being `b − oc_bin_first`, with `oc_bin_first = 1145` stored in
   the npz. That saves the 4.7 GB of NaN slabs before 1997 (8.2 GB instead of
   12.9 GB). It needs `ml/cone_sampler.py`'s multi-group gather,
   `ml/publish_family7_index.py` and the app's slab reader
   (`src/app.js`, `bin_first` arithmetic) to accept a per-group first bin.
   If that is a one-line generalisation in each, do it and add the tests.
2. **Otherwise — the full axis:** `T_oc` = 3142, rows 0–1144 NaN, no offset.
   Simpler for every consumer; 12.9 GB. Acceptable.

The npz gains `chan_oc025`, `norm_oc025`, `count_oc025` (finite values per
channel), `oc_bin_first`, `n_occci_days` (days actually fetched and read),
`groups = ["g025","g100","rg100","oc025"]`, `recipe = "f7l1"`. Everything
else in the npz is regenerated by the `meta` stage from the inherited state
and must be **equal** to `f7l0`'s (a test compares every shared key).

## 5 · Build mechanics — inherit, don't rebuild

- New stage `occci`, in `STAGES` between `rg` and `static`; `DEPS`: `norm`
  additionally needs `occci`. `STAGE_OWNS["occci"] = "oc025"`,
  `STAGE_CHANNELS["occci"] = CHAN_OC025`, `SPEC_VERSION["occci"] = 1`.
  Written at float32 to `<stem>_X_oc025.f32.npy` like the coarse groups, and
  converted by `norm` (the RAW_F32 rule at the top of the builder).
- Per-year markers and resumability exactly as `sst`: a year is one item;
  the daily files are streamed and **deleted as they are read** (the Argo
  builder's discipline; 10,000 × ~40 MB must never sit on the disk at once);
  a carry-over accumulator handles the pentad that straddles a year
  boundary. `disk_guard` sized for the f32 group (1997 × 721 × 1440 × 2 × 4 B
  = 16.6 GB) plus one year of sources in flight.
- **Recipe vs base recipe.** `RECIPE = "f7l1"`, `STEM =
  "family7_global025_pentad_l1"`, and a new `BASE_RECIPE = "f7l0"` which is
  what `stage_spec` folds into the digest of the *inherited* stages
  (`glorys`, `sst`, `ncep`, `rg`, `static`, `truth`), so that their spec
  files, copied from the `f7l0` work directory, still match and **nothing is
  discarded**. The `occci`, `norm`, `meta` and `publish` digests use `RECIPE`.
- **Seeding.** A new builder flag `--seed-from <old work dir>` (and a matching
  workflow input) hard-links (`os.link`, same filesystem — `/opt/earth-cache`)
  the three finished group files under the new stem, copies the `.done`
  markers, `.spec` files, per-stage directories (`sst/`, `ncep/`, `rg/`,
  `rg/live.npz`, `oisst_seen.npy`, the `static`/`truth` outputs) and
  `norm.npz` into the new work directory, and then **asserts** the three
  linked files have the sizes the shapes imply. It never writes into, renames
  or deletes anything under the old directory. If the old directory is
  absent the flag is an error, not a silent full rebuild.
- **`norm` becomes group-idempotent.** Today it is one marker for all groups.
  It must (a) skip a group whose `norm_<g>` is already in `norm.npz` AND whose
  final float16 file exists, (b) process only the missing group(s), (c) be
  re-enterable when `norm.done` exists but a group in `GROUPS` has no norm —
  without ever re-z-scoring an already converted group (which would square the
  transform, as the docstring warns). The stale-spec rule for `norm`
  ("cannot be discarded") stays; add the group-awareness beside it.
- **A build never publishes an empty group** (2026-09-14, family7-build #10:
  a box that could not read the Argo cubes wrote `rg100` with zero rows,
  normalised it over zero values and published 128 bytes, green) — `rg`
  refuses unless `--allow-empty-rg` says a no-subsurface tensor is what was
  asked for. **`--redo-group <g>` is how ONE group is rebuilt in place**: it
  deletes exactly that group's fill, float16, norm statistics and markers,
  its fill stage's markers, carries and spec, and `meta`/`publish`, so the
  next run rebuilds that group alone and every other stage is skipped by its
  own marker (`g025` is refused — its z-score was in place).
- `meta` regenerates the npz (all inherited keys equal to `f7l0`'s, plus §4's
  new keys). `publish` uploads all five files to
  `tensors/family7_global025_pentad_l1/`, restore-verifies each, and writes
  `manifest.json` with, per file, `sha256` and — for the three inherited ones
  — `same_as_f7l0: true` after **comparing against `f7l0`'s manifest hashes**
  (fetch `tensors/family7_global025_pentad_l0/manifest.json`; a mismatch fails
  the publish, because then the file is not the one the handover describes).
- **Smoke.** `run_smoke` gains a second leg: build a tiny `l0`-style smoke
  work dir, then run the `l1` path with `--seed-from` it and a synthetic
  OC-CCI source directory (`--source-dir`, a few generated daily files on the
  real 4320 × 8640 grid, or a decimated one if the test declares so), and
  assert: the three seeded groups are byte-identical to the seed; `oc025`
  has the shape of §4; a block whose 36 cells are all `10^0.5` mg m⁻³ on all
  five days reads `log_chl = 0.5`, `chl_cov = 1.0`; a block with one finite
  cell on one day reads that cell's log10 and `chl_cov = 1/180`; an all-fill
  block is NaN in both; longitude wrap and the pole rows are exercised;
  `hypot`/pole/NaN assertions of `tests/test_build_family7.py` still pass.
- **Workflow.** `family7-build.yml`: `stage` input accepts `occci`; new
  inputs `seed_from` (default `ml/cache/family7`) and `work` default
  `ml/cache/family7_l1`; the preflight adds the OC-CCI one-file fetch; the
  disk guard's numbers updated. `workflow_dispatch` only — unchanged.
- **Index & app.** `ml/publish_family7_index.py` learns the fourth group and
  writes `data/family7_index.json` for `l1` (the app then paints `log_chl`
  and `chl_cov` as two more channels of the "Global tensor" layer; units and
  labels in `UNITS`; the Cones tab's live mode reads the new group through
  the same LRU). Registration: `docs.html`'s `DOCS` list gets this plan;
  `tests/docs.spec.js` enforces it.

## 6 · Cost and time

Sources: ~10,000 daily files × ~40 MB ≈ 400 GB streamed at the box's ~44 MB/s
≈ 2.5–3 h, CPU-bound on the NetCDF decode and the 36 → 1 block mean (numpy
reshape-mean over `[4320, 8640] → [720, 1440]` is milliseconds; the pole row
and the wrap are the only special cases). `norm` on 16.6 GB: minutes.
Publish + restore-verify of 62 GB: ~1 h. Box `gpu-box-31299601` (Vast
49102182, Ontario, 300 GB, $0.33/h): **≈ 4–5 h, ≈ $1.5–2.** Disk: 89 GB used
before; + 16.6 GB f32 + 8.2 GB f16 + ≤ 3 GB sources in flight + the verify
downloads one file at a time (≤ 46 GB transiently) — fits with ~130 GB to
spare.

**The PSL inputs are Hub-mirrored (2026-09-13).** Measured that day, the UK
box reads `downloads.psl.noaa.gov` at **0.17 MB/s** — one 477 MB OISST year in
47 minutes, against 33 s from another box — so the ~45 GB the `sst` and `ncep`
stages need could not arrive inside the 24 h timeout, while the same box pulls
the Hub at full speed. Those files now live under `mirrors/psl/Datasets/…`
(written by `ml/mirror_psl.py`, round-trip verified), PSL stays the fallback,
and `build_family3.fetch` aborts any transfer averaging under 1 MB/s after 90 s
instead of waiting it out. This costs the colour stage nothing — CEDA is a
different host — but it is why a rebuild of the three inherited groups is
affordable again.

**The colour stage does not fit on one box, measured (2026-09-13).**
`dap.ceda.ac.uk` serves **1.2–1.4 MB/s PER CONNECTION** and parallel
connections scale (4 parallel range reads → 4.6 MB/s aggregate), and the files
are ~79 MB rather than the ~40 MB assumed above: **9,980 × 79 MB = 790 GB**.
Sequentially that is 169 h, and 48 h even at the four-connection rate, against
this workflow's 24 h timeout — so the arithmetic at the top of this section is
superseded, and no rentable box fixes it because the limit is per-connection.
Two changes follow. `stage_occci` now downloads days AHEAD of the consumer with
`EARTH_OC_WORKERS` threads (default 8, ≤ 2× workers in flight and ≤ 3 GB on
disk) while still reducing them in date order. And the reduction splits by
CALENDAR YEAR onto free hosted runners — `--stage occci-partial --years Y`,
`.github/workflows/occci-partials.yml`, 20 lanes at once, ~29 GB in and
~780 MB out per lane inside the 6 h hosted limit — publishing
`partials/f7l1/occci/<Y>.npz` (restore-verified) that the box then FOLDS: the
accumulators are sums over days, so a bin straddling 31 December takes a
contribution from each year's file. The year is the unit of the accumulation on
both paths, so the fold is bit-identical rather than merely close
(`tests/test_build_family7.py::test_30`); the partials are re-usable across
rebuilds, and the box's cost drops from 790 GB of CEDA to ~22 GB of Hub.
Details: `docs/FAMILY7_DATA_HANDOVER.md` §10, "OC-CCI on hosted runners".

## 7 · What is asserted before the tensor is trusted

1. The three inherited group files are hash-identical to `f7l0`'s manifest.
2. `oc025` shape, dtype and header length; `oc_bin_first` = the bin of
   1997-09-04 computed from the epoch, not typed.
3. Every finite `chl_cov` is in (0, 1]; `log_chl` is finite exactly where
   `chl_cov` is; no finite value anywhere on `sphere == 1` (land) cells
   beyond the coastal 0.25° fringe (the 4 km product has coastal pixels the
   0.25° mask calls land — measure and record the count rather than mask it).
4. On one pentad (bin 2411, 2015-01-03, the family-7 comparison bin) the
   0.25° `log_chl` field's global mean and the fraction of observed ocean
   cells are printed into the build log and copied into
   `ml/EXPERIMENTS.md#e-077` — the number a reader compares against the
   PACE/MODIS chlorophyll layer on the globe by eye.
5. Resumability across a simulated interruption inside the `occci` stage.

## 8 · Holdout

Unchanged: 2009, 2017 and 2023 development holdout years; terminal split
train ≤ 2020, test 2021–2024. Colour's record starts in 1997-09, so a
climatology fitted on training years has 1998–2008 + 2010–2016 + 2018–2020
= 21 years to fit from.

## 9 · Not in this build, on purpose

Reflectance bands (`Rrs_412…670`) as inputs — a second group when colour
becomes an input rather than a target; PACE OCI (too short for a tensor;
stays a globe layer); any gap-filled L4; the finer-than-0.25° representation
(E-078).
