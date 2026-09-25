# E-083 · The model's climatology, inspectable and downloadable on blauewelt.org

Written 2026-09-25 (Fable plans, Opus implements — `ml/CLAUDE.md` §0b).

Chris, 2026-09-25: *"What would it take to make the climatology (means?)
baseline inspectable and downloadable on blauewelt.org for various
channels?"* — and, on which climatology: *"I would not holdout anything for
now? (but offer certain holdout periods?). Let's do 3 versions: 1) All data
2) The holdout regime with 2009 / 2017 / 2023 held out. 3) The contiguous
holdout (what's in the paper)."* — *"And yes, let's do it."*

## 1. What the climatology IS

Every forecaster in this programme is trained in **anomaly space**: before a
channel reaches the model it is turned into a departure from its own
**per-calendar-month climatology** (the mean of that channel, at that cell,
over every training-year bin whose date falls in that calendar month), and
then z-scored per channel. The one function that does this is
`ml/trainprobe.py::anomaly_transform` — `tests/test_one_anomaly_transform.py`
fails the suite if a second implementation appears — and since E-076 it
hands the climatology it subtracted back to its caller through its `stats`
argument: `stats["clim"]` is `[12, H, W, C]` float32, NaN where a (month,
cell, channel) had no training sample and 0.0 on static channels; `mu`, `sd`
and `den` are the per-channel z-score constants applied afterwards.

That climatology is also the **baseline every skill number is measured
against**: the rollout skill `msss_clim` (mean-square skill score against
climatology — 1 is perfect, 0 is "no better than saying *normal*") scores
the model's forecast against exactly this field. So "what the model calls
normal" is a first-class artefact of the programme, and it has never been
looked at. It exists only for the seconds between the transform's first and
second pass, in the RAM of a rented box.

**Family 7** is the tensor whose climatology this plan publishes — the first
input tensor covering the whole globe rather than the North Atlantic window,
every 0.25° point from pole to pole, 1982 to 2024, five-day bins
(`ml/plans/E070_family7_build.md`, stem `family7_global025_pentad_l2` on the
Hub). It has four channel groups, each with its own grid and time axis:

| group | grid | channels | bins | note |
|---|---|---|---|---|
| `g025` | 0.25°, 721×1440 | 7 — current speed, u, v, mixed-layer depth (log₁₀), sea-surface height, sea-surface temperature (OISST), sea ice | 3,142 pentads, 1982-01-01 → 2024-12 | 45.7 GB float16 on the Hub |
| `g100` | 1°, 181×360 | 15 — the reanalysis land/ocean channels: skin and 2 m air temperature, wind, pressure, precipitation, snow, soil, heat fluxes | 3,142 pentads | 6.1 GB |
| `oc025` | 0.25° | 2 — ocean colour | 1,997 pentads from bin 1,145 (1997-09-04) | 8.3 GB |
| `rg100` | 1° | 32 — Argo temperature and salinity at 16 depths | 252 monthly rows, 2004 → | 1.0 GB |

The values in the tensor are stored **z-scored by the builder** (`norm` =
per-channel (mean, sd) in `data/family7_index.json`), so a climatology
computed on the stored values is in z-units and is converted back to °C, m,
m/s at read-out time exactly as the "Global tensor" layer already does.

## 2. The three versions

Which bins count as "training" decides the climatology, and the three
versions Chris asked for are three answers to that question. Each is
computed by the same function with a different `t_hold` mask (a boolean per
bin: True = held out of the climatology and of every statistic).

| version key | name on the page | training bins | held out | where it comes from |
|---|---|---|---|---|
| `all` | **All years (1982–2024)** — the default | every bin | none | Chris: *"I would not holdout anything for now"* |
| `dev` | **Development holdout — 2009, 2017 and 2023 held out** | every bin whose year ∉ {2009, 2017, 2023} | those three years, wherever they fall | `CONE_HOLDOUT_YEARS` / `ml/export_cone_sample.py::HOLDOUT_YEARS`, the regime `ml/plans/E059_holdout_window.md` (E-059, the run that fixed a training pool which had been teacher-forcing the held-out years into the weights) trains under |
| `paper` | **The paper's split — trained on 1982–2020 less 2009 and 2017** | every bin whose year ≤ 2020 and ∉ {2009, 2017} | 2009, 2017, and everything from 2021-01-01 on (the contiguous retrospective period 2021–2024) | `ml/paper/paper.tex` §"The evaluation contract": *"Training is 1982–2020 less the development years 2009 and 2017, which are held out of training, climatology and every statistic … Retrospective evaluation is 2021–2024: no model is trained on it and no statistic is fitted on it"* |

Each group is masked on **its own** months — `rg100` holds one row per
month and `oc025` starts in 1997, so the master pentad calendar is the wrong
one for both (`ml/cone_sampler.py::group_time` already does this for the
cone export). A version therefore has one `t_hold` per group, all derived
from the same year rule.

What differs between the versions is small and worth seeing: the `paper`
climatology is fitted to 37 years and the `all` one to 43, so under a
warming trend the `paper` normal is cooler and an anomaly read against it is
warmer — the difference IS the trend of the last four years, and the page
lets a reader flip between them at one cell.

## 3. Export (`ml/export_family7_clim.py`)

Runs on a rented box beside the tensor; the sandbox has neither the disk nor
the bandwidth (the Hub serves one stream at ≈ 2 MB/s; the box pulls the
61 GB with parallel range reads).

1. **Pull** the four group `.npy` sidecars and the `.npz` metadata half of
   `tensors/family7_global025_pentad_l2/` from the Hub (`ml/tensor_io.py`'s
   sidecar layout; read with `load_tensor`, memory-mapped). Verify each
   file's sha256 against `data/family7_index.json`'s `sha256` before
   computing anything — the climatology of the wrong bytes looks exactly
   like a climatology.
2. **For each group × each version**: build `t_hold` from the group's own
   months and the version's year rule; take a writable copy of the group
   (`tensor_io.writable_copy`, or an in-RAM copy on a box with the room —
   `anomaly_transform` writes in place and the canonical memmap must stay
   state-space); call `anomaly_transform(A, moy, t_hold,
   np.zeros(W, bool), chunk=anomaly_chunk(...), stats=stats)`; keep
   `stats`. No second implementation of the climatology, anywhere — the
   test in §5 pins that the export's bytes ARE the function's.
3. **Write, per (version, group)**:
   - `clim.npy` — `[12, C, H, W]` float32, **month-major then channel-major**,
     so one (month, channel) plane is one contiguous slab the page can fetch
     with a single `Range:` read: 4.15 MB at 0.25°, 0.26 MB at 1°. (The
     function returns `[12, H, W, C]`; the export transposes once. Float32,
     not float16: a mean of thousands of float16 samples deserves its
     digits, and 349 MB per version for `g025` is nothing beside the tensor.)
   - `clim.nc` — the same field as a CF-conventional NetCDF for people who
     want a file rather than a layer: dimensions `(month, lat, lon)`, one
     variable per channel **in physical units** (z × sd + mean from `norm`),
     `long_name`/`units` from the index's labels, zlib-compressed, global
     attributes naming the version, the training years, the tensor stem and
     sha256, the git sha and the generation time.
   - `stats.json` — `mu`, `sd`, `den`, `dynamic`, the channel list, `norm`,
     the version's year rule and held-out years, the number of training bins
     per group, the tensor's sha256 per group, the export's git sha, UTC time.
4. **Publish** under `tensors/family7_global025_pentad_l2/clim/<version>/<group>/`
   on `chfrank/earth-tensors` with `ml/hf_mirror.py`'s rule: every file is
   downloaded BACK and its sha256 must match before the index is written.
5. **Write `data/family7_clim_index.json`** (by `ml/publish_family7_clim_index.py`,
   the sibling of `ml/publish_family7_index.py`, never hand-edited): for
   every (version, group) the resolve URL, the parsed `.npy` header length,
   shape, dtype, plane bytes, sha256 and byte count; the versions with their
   names, year rules and held-out years; the channel labels, units, ramps,
   signs and `norm` copied from `data/family7_index.json` (one source of
   truth for channel metadata — the clim index carries the family-7 index's
   `stem` and refuses to be built against a different one); the measured
   CORS headers (`Origin: https://blauewelt.github.io`, a ranged GET
   answering 206 — the family-7 measurement repeated on the new files, not
   assumed from them).

**As built (2026-09-25) — where the code differs from, or pins down, the
list above:**

- **Pull** is its own script, `ml/pull_family7_tensor.py --dest <dir>
  [--workers 16] [--chunk-mb 64] [--method range|hub]`: parallel `Range:`
  reads (each must answer 206 with the exact Content-Range), resumable per
  chunk (`<name>.part` + `<name>.part.done`, marked only after the bytes
  land), `HF_TOKEN` from the env on every request, MB/s printed per file and
  overall. The group `.npy` sha256s come from `data/family7_index.json`; the
  index carries none for the `.npz`, so its sha256 comes from the tensor's
  own `manifest.json` on the Hub (and the two must agree on every group).
- **Export CLI**: `python3 ml/export_family7_clim.py --tensor <dir>/<stem>.npz
  --index data/family7_index.json --out <dir> [--versions all,dev,paper]
  [--groups g025,g100,oc025,rg100] [--chunk N] [--scratch DIR | --copy ram]
  [--skip-sha] [--no-nc]`. The versions are the module-level `VERSIONS`
  table; `paper` is written as `holdout_years: [2009, 2017]` plus
  `holdout_from: 2021` (every bin of a year ≥ 2021), a field the other two
  versions carry as `null`. The transform is reached through
  `ml/export_cone_sample.py::_anomaly_transform` (import, or the `ast` lift).
- **Box install line**: `pip install numpy netCDF4 huggingface_hub requests`
  (netCDF4 is checked before the first group is copied; `--no-nc` skips
  `clim.nc`).
- **Static channels.** `anomaly_transform` writes 0.0 into the climatology of
  any channel it finds static (no temporal variance of the spatial mean).
  `clim.npy` keeps those bytes (it IS the function's array); `stats.json` and
  the index list them as `static_channels` / `static_chans`, and the NetCDF
  variable's `comment` says it is not a climatology. None is expected on the
  real tensor; the toy test has one on purpose.
- **Publish + index**: `python3 ml/publish_family7_clim_index.py upload --out
  <dir>` (one `hub_commit` per version, every file downloaded back and
  sha256-matched, CORS measured on `all/<first group>/clim.npy`, then
  `data/family7_clim_index.json` with `restore_verified: true`).
  `… index --out <dir>` rewrites the index from files already on the Hub
  (still restored); `--local --no-cors` is the fixture's no-network form.
- **The fixture** (`python3 ml/export_family7_clim.py --fixture`) runs the
  real loader on `data/family7/fixture/`, whose metadata `.npz` is not in git
  — it is rebuilt from that fixture's own index into a temp dir beside
  symlinked sidecars. It ships **`g100` and `oc025` only, in all three
  versions, without `clim.nc`** (2.27 MB; all four groups would be 8.1 MB),
  and its index says `clim_nc: null`. The smoke tensor covers five pentads of
  January 2010, so only month 1 is finite and the three versions are
  byte-identical (2010 is a training year under all three rules).

Cost: one verified box with ≥ 128 GB RAM and ≥ 200 GB disk for a few hours
(pull ≈ 61 GB; then 3 versions × 3 passes over each group, page-cache
resident); well under $5. The box is destroyed afterwards; nothing secret
touches it (the `HF_TOKEN` is an env var for the life of the job, per the
family-7 workflows).

## 4. The globe layer and the downloads

A new layer in the layer list, beside "Global tensor":

**"Model climatology — what the forecaster calls normal (family 7), 0.25° / 1°"**

- Reuses the family-7 range-read machinery in `src/app.js` (`data/family7_index.json`
  → slab LRU → decoded plane → `GridProvider`-style paint through the
  channel's `ramp`, `sign`, `units` and `norm`), with the clim index as its
  address book. The month is the **calendar month of the globe's date**
  (`state.date`), so stepping ±1 m walks through the seasonal cycle and the
  Play tab animates it with no playback code; the year is ignored and the
  toast says so (it is a `datelessToast` variant: *"the month matters, the
  year doesn't"*).
- Two selectors on the layer's row: the **channel** (grouped by tensor group,
  labelled in plain English from the index) and the **version** (§2's three
  names; `all` selected by default). Changing either is a decode, or one
  4 MB read, never more.
- **Legend** in physical units (z × sd + mean), the same convention as the
  tensor layer; the probe and the pixel card print the normal in the
  channel's unit AND as the tensor stores it (`z = …`), and — when the
  "Global tensor" slab for the globe's date is already in the LRU, or that
  layer is on — a second row **"departure from normal"** = tensor value −
  normal, in the unit and as the trainer's z-score ((value − clim − mu)/den).
  Never an extra 14.5 MB read on a click for the departure alone: a cell with
  no tensor slab resident prints the normal and a hint.
- **Hover card**: gist ("the per-calendar-month mean every forecast is scored
  against"), Recorded = the version's training years, Interval = "one
  calendar month; the year does not matter", Spatial = 0.25° / 1° by group,
  and a **Downloads** block: for the selected version and group the direct
  Hub links to `clim.nc` and `clim.npy`, `stats.json`, and the whole index;
  plus **"this channel, this month as CSV"**, generated client-side from the
  plane already in memory (`lat,lon,value` in physical units, NaN as empty).
- Catalogue exception as `amoc-eval` and the tensor layer take it (root
  `CLAUDE.md` §2.6): a picture of our own work, no `data/catalog.json`
  record; `doc` links to this plan. Chip-registered, opacity row, `period`
  stamp = the version's training span (a fixed span, no age).
- `data/family7_clim/fixture/` holds the same schema decimated to 5°/10°
  from the smoke tensor (written by the export script's `--fixture` path
  from `data/family7/fixture/`), so `tests/app.spec.js` drives the layer
  under MIRROR with a real 206 on sliced bytes, and `tests/data.spec.js`
  pins the index against the family-7 index (same stem, same channels, same
  `norm`).

`docs/FAMILY7_CLIM.md` explains all of it for a reader and says how to
regenerate; the "Global tensor" hover card gains one line pointing at the
new layer.

## 5. Falsifiers and tests

- `tests/test_export_family7_clim.py`: on a toy tensor the exported
  `clim.npy` is bit-identical (after the transpose) to `stats["clim"]` from a
  direct `anomaly_transform` call under the same `t_hold`; the three
  versions' `t_hold` masks are exactly the year rules of §2 (2009 and 2017 bins held
  out in `dev` and `paper`; 2023 held out in both — by name in `dev`, by the
  2021-onward rule in `paper`; 2021, 2022 and 2024 held out only in `paper`;
  nothing in `all`); `rg100` is masked on its own months; the NetCDF's
  physical values equal `clim × sd + mean`; `stats.json` carries the tensor
  sha256s and git sha; an index built against a different family-7 stem is
  refused.
- `tests/data.spec.js`: the committed index and fixture agree (files,
  shapes, header lengths, plane bytes = C-order product).
- `tests/app.spec.js`: enabling the layer under MIRROR issues exactly one
  ranged request answered 206, the probe reads the cell the fixture holds,
  switching version re-reads and switching channel within a fetched month
  does not, the legend is in physical units, the CSV download has the right
  row count.
- On the live site: at one ocean cell, `all` minus `paper` for
  sea-surface temperature has the sign of the 2021–2024 warming (positive
  over most of the ocean) — a measurement, printed in §6 when it exists.

## 6. Status

- 2026-09-25: plan written. Nothing exported yet.
- 2026-09-25, later: export script + tests landed (d7e3fc5d), workflow
  `family7-clim.yml` (054b6c70). **EXPORTED AND PUBLISHED**: family7-clim run #1
  on a rented verified box (Japan, 129 GB RAM, $0.163/h) — pull 15 min for
  61 GB, export 70 min for four groups × three versions, upload 30 s; the box's
  container exited during its own restore check, so the index was written from
  the sandbox: all 36 files under
  `tensors/family7_global025_pentad_l2/clim/<version>/<group>/` downloaded
  back and sha256-matched (each npy/nc sha256 equals the LFS oid the box
  uploaded), CORS 206 with `allow-origin *`. Index committed (60a18039); the
  layer, the fixture tests and `docs/FAMILY7_CLIM.md` deployed with it. Box
  destroyed. Cost ≈ $0.50.
- **The §5 live measurement holds.** Sea-surface temperature, `all` minus
  `paper`, over every ocean cell × month of the 0.25° group: mean **+0.044 °C**,
  median +0.042 °C, positive on **77.7 %** of cell-months — the sign of the
  2021–2024 warming the paper's split leaves out of its normal. At the RAPID
  line (26.5° N, 70° W) the July normal reads 28.33 °C (`all`), 28.28 (`dev`),
  28.25 (`paper`); Niño 3.4 (0°, 150° W) 26.68 / 26.62 / 26.65; the
  North Atlantic at 55° N, 30° W 11.75 / 11.75 / 11.71. The page's range-read
  arithmetic replayed from node against the live file returns the same
  28.33 °C at RAPID as the local export.
- Follow-up: the box's publish step ran `hf_hub_download` for the 2.5 GB
  restore from Japan and never finished before the container died — the next
  run of the workflow should restore with `ml/pull_family7_tensor.py`'s ranged
  puller (parallel, resumable) rather than one stream per file.
