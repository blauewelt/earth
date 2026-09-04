# E-070 · Family 7 build spec — the global tensor, to the poles, with the shared land/ocean channels (recipe `f7l0`)

**Written 2026-09-04**, from Chris's instruction of that day: *"proceed with
building the new global tensor with all the data (including Antarctica, common
channels for land and water, as proposed there)"* — "there" being
[E-071 §1 and §6](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E071_cone_v2.md#6-land-ice-and-air-nothing-is-dark-by-design-and-the-channel),
the previous session's design. This document is the executable specification
that [E-070](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_global_tensor.md)
Phases B–D and E-071 §6.5 (Phase L0) become when they are built **once, as
one tensor** — the "two builds are two experiments" trap of E-070 §5 is why
the ocean-only 681-row `f7r3` layout is *not* built first.

**Plain English.** Family 7 is the first input tensor covering the whole
globe rather than the North Atlantic window: every 0.25° grid point from the
South Pole to the North Pole, one value per channel per five-day bin from
1982 to 2024. It carries the ocean channels the North Atlantic tensor
already had (currents, mixed-layer depth, sea-surface height, sea-surface
temperature, wind stress, the Argo depth column) and, new, a set of
*shared* channels that exist over both land and water — surface temperature,
2 m air temperature, 10 m wind, surface pressure, precipitation, snow, soil
moisture and temperature, turbulent heat fluxes, sea ice — so that a land
cell is no longer told "the world was not observed here" in every channel.

**Status: specified and implemented in this session; the build job is
dispatched on Chris's fleet (see the E-070 log entry for the run number).**

---

## 0 · The decisions, in one table

| # | decision | why | reverses E-070's |
|---|---|---|---|
| B1 | **Grid 721 × 1440**, lat −90 … 90, lon −180 … 179.75, point-aligned 0.25°, south-first | E-071 §1: the −80° floor was GLORYS12's extent, not the world's; Antarctica is filled by the shared channels | D2 (681 rows) |
| B2 | **Three channel groups at their native resolution**, not one dense tensor: `g025` (0.25°, 7 ch), `g100` (1°, 15 ch), `rg100` (1°, 32 ch, live bins only) | a 1.9° reanalysis upsampled to 0.25° is 60 copies of every number; E-070 D3 already put the 1° Argo product in a 1° sidecar for exactly this reason, and E-071 §6.4 needs ~65 channels, which at 0.25° dense would be 425 GB. The cone reads dots at 0.25° positions and looks each coarse channel up at the nearest coarse cell ("served as the same cell", D3) | D3 (one sidecar) |
| B3 | **Shared-channel source for Phase L0 = NCEP/NCAR Reanalysis 1** (T62 gaussian, daily, 1948→, key-free from NOAA PSL) plus **OISST v2.1 sea-ice concentration** (0.25°, daily, 1981→, same files family 4's SST comes from) | E-071 §6.5 named ERA5, which needs the free CDS account only Chris can create; NCEP R1 is the same physics at 1.9°, is what the tensor's wind stress already comes from, and needs nothing. ERA5 swaps in as Phase L1 by replacing the `g100` fetch — the layout does not change | — |
| B4 | **Two temperature channels, split by INSTRUMENT**: `sst` (g025) is OISST sea-surface temperature and is missing wherever OISST does not observe; `skt` (g100) is NCEP skin temperature over every surface — land, sea, ice alike | a channel is shared only when the measurand AND the instrument match on both sides (E-071 §6.1, "Correction, 4 Sep"). `sst` stays bit-identical to family 4's `sst`, which is what the G1 gate compares, and `skt` is the shared field ERA5 replaces at Phase L1 with no layout change | — |
| B4c | **The correction that produced B4's present form**, recorded because the first build ran under the old rule | The original B4 merged the two into one `skin_t` channel — one measurand, but an infrared/microwave analysis over the sea and a reanalysis everywhere else, spliced at a coastline the model would have had to learn was an instrument boundary rather than a physical one. `ml/build_family7.py` now carries a `repair_sst_channel` pass that runs automatically at the head of the `ncep` stage and clears that fill from any work dir built under the old rule, plus a per-stage `.spec` digest that makes a stale stage discard its own markers, carry and float32 fill file and restart | B4 |
| B4b | OISST's `icec` file declares `units "percent"` but its `valid_range` is 0 … 1 — the reader trusts the range, not the string (measured on the 2020 file's DAS, 2026-09-04) | a percent-divide would have silently scaled the whole channel by 1/100 | — |
| B5 | **Sea ice and snow stay two channels** (`sea_ice` from OISST, `log_swe` from NCEP), not the merged "frozen fraction" of E-071 §6.1 | NCEP reports snow as water equivalent, not a cover fraction; merging a fraction with a mass would need an invented constant. The merged channel arrives with MOD10A1 in Phase L1 | — |
| B6 | **Statics in the tensor**: `sphere` (0 ocean · 1 land · 2 ice sheet · 3 inland water) and `elev` (surface elevation, m; negative = ocean depth) | E-071 §6.1 needs the sphere code as a codec coordinate; elevation is the single most informative static over both spheres (E-070 §6, DATA_LADDER §2) | — |
| B7 | **Build runs as one resumable job on a Vast box** (`family7-build.yml`, `workflow_dispatch` only, self-hosted runner), streaming every source through and publishing to the Hub | E-070's Phase C on GitHub-hosted runners would need four year-lanes for 37 GB of memmaps that do not fit a runner; the box has the disk and the job resumes per stage and per year | Phase C's runner lanes |
| B8 | **Recipe `f7l0`, output stem `family7_global025_pentad_l0`**, published to `chfrank/earth-tensors` under `tensors/family7_global025_pentad_l0/` with a manifest of sha256s, restore-verified | the Hub's 10 TB PRO tier vs the 2 GiB-per-asset release cap | Phase D's release parts |

Everything else in E-070 §2 stands: pentad cadence, epoch 1982-01-01, the
frozen protocol, the NA sub-block as the comparison surface, stage 2 Atlantic.

---

## 1 · Grid, time, and the coarse lookup

```
lats  = -90  + 0.25 * arange(721)      # south-first, ascending, as every family
lons  = -180 + 0.25 * arange(1440)
lat1  = arange(-90, 91)                # 181, point-aligned 1°
lon1  = arange(-180, 180)              # 360
epoch = 1982-01-01 · pentad_days = 5 · T = 3142 bins (bin b = days [5b, 5b+5))
```

The coarse lookup for a 0.25° index `(y, x)` is `y1 = round(y / 4)`,
`x1 = round(x / 4) mod 360` — every fourth 0.25° point IS a 1° point, and
the two points on either side of it round to it. `x = 1438, 1439` round to
360 and wrap to `lon1 = -180`, which is the same meridian as `+180`.

The GLORYS12 chunks (`daily025_global/`, 681 × 1440, lat −80 … 90, verified
on `202006`) land at rows **40 … 720**; rows 0 … 39 of every ocean channel
are NaN, because there is no ocean there, and `sphere` says so.

The NA sub-block of family 4 (lat 0 … 70, lon −100 … 20) is
`g025[:, 360:641, 320:801, :]`.

## 2 · Channels

### `g025` — 0.25°, `[3142, 721, 1440, 7]` float16, 46 GB

| idx | name | source | rule |
|---|---|---|---|
| 0 | `cur_speed` | GLORYS12 `uo`,`vo` | pentad mean of the daily bins, `hypot`, ≥ 3 days in the bin (family 4's `min_days`) |
| 1 | `log_mld` | GLORYS12 `mlotst` | `log10(mld)` where `mld > 0`, else NaN |
| 2 | `ssh` | GLORYS12 `zos` | pentad mean |
| 3 | `cur_u` | GLORYS12 `uo` | pentad mean — written in the same pass as ch 0 so `hypot(cur_u, cur_v) == cur_speed` |
| 4 | `cur_v` | GLORYS12 `vo` | pentad mean |
| 5 | `sst` | OISST v2.1 `sst.day.mean` | OISST bilinear to the point grid (`fetch_sst_na.weights_for` / `f3.interp2_nan`, `wrap_period` 360), pentad mean, ≥ 3 days. **An OBSERVED channel: NaN wherever OISST does not observe** — no reanalysis fill (B4c). Bit-identical to family 4's `sst` |
| 6 | `sea_ice` | OISST v2.1 `icec.day.mean` | bilinear as above, pentad mean, 0 … 1; NaN wherever OISST has no sea (land, ice shelves) |

### `g100` — 1°, `[3142, 181, 360, 15]` float16, 6.1 GB

All from NCEP/NCAR R1 daily gaussian files
(`https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/<var>.<year>.nc`,
THREDDS mirror as `build_family4.fill_wind_pentad` uses), bilinear from the
T62 grid (192 × 94, descending latitude, 0 … 358.125 longitude) to the 1°
point grid with `f3.lin_weights(..., wrap_period=360)` + `f3.interp2_nan`,
pentad mean with ≥ 3 days. Variables with a `level` dimension are squeezed.

| idx | name | file | transform |
|---|---|---|---|
| 0 | `tau_x` | `uflx.sfc.gauss` | **sign-flipped** (stress on the surface), as family 4 |
| 1 | `tau_y` | `vflx.sfc.gauss` | sign-flipped |
| 2 | `tau_x_std` | `uflx.sfc.gauss` | within-pentad population σ of the dailies, as family 4 |
| 3 | `tau_y_std` | `vflx.sfc.gauss` | same |
| 4 | `t2m` | `air.2m.gauss` | K → °C |
| 5 | `u10` | `uwnd.10m.gauss` | m/s, no flip |
| 6 | `v10` | `vwnd.10m.gauss` | m/s |
| 7 | `sp` | `pres.sfc.gauss` | Pa → hPa |
| 8 | `log_prate` | `prate.sfc.gauss` | kg m⁻² s⁻¹ → mm/day (× 86400), then `log1p` |
| 9 | `log_swe` | `weasd.sfc.gauss` | kg m⁻² (= mm water equivalent), `log1p` |
| 10 | `soilw` | `soilw.0-10cm.gauss` | volumetric fraction; **NaN over sea** by `land.sfc.gauss.nc` applied on the gaussian grid *before* regridding (so `interp2_nan` renormalises at the coast) |
| 11 | `tsoil` | `tmp.0-10cm.gauss` | K → °C, land-masked as `soilw` |
| 12 | `lhtfl` | `lhtfl.sfc.gauss` | W m⁻², NCEP sign (positive upward), unchanged |
| 13 | `shtfl` | `shtfl.sfc.gauss` | W m⁻², unchanged |
| 14 | `skt` | `skt.sfc.gauss` | K → °C. **The shared surface temperature**: every surface, land, sea and ice, no land mask and no OISST splice (B4/B4c). This is the channel ERA5 replaces at Phase L1 |

The four `tau_*` channels move here from the dense tensor: at 0.25° they
were 60 copies of every T62 number.

### `rg100` — 1°, `[n_live, 181, 360, 32]` float16 + `rg_bin_index [n_live]`, ~1 GB

The 32 Roemmich–Gilson channels of family 4, same names and order
(`rg_t10 … rg_t1900`, `rg_s10 … rg_s1900`), written **once per month into
the pentad that contains the 15th** (`build_family4.live_row`, E-034 §4),
bilinear from RG's 1° cell-centred grid (lat −64.5 … 79.5, lon 20.5 …
379.5) to the 1° point grid with `wrap_period` 360, and **explicitly NaN
outside RG's latitude band** — `f3.lin_weights` clamps at the axis ends and
would otherwise replicate the edge rows into the Southern Ocean and the
Arctic. `rg_bin_index` lists the live bins in order; a consumer that wants
the value at bin `b` looks up `b` in it and gets a miss token otherwise.

### Statics (in the meta npz)

| key | dtype/shape | rule |
|---|---|---|
| `sphere` | int8 `[721, 1440]` | **2** if the cell centre falls in a Natural Earth 10 m *glaciated areas* polygon (ice sheets and glaciers); else **0** if OISST or GLORYS ever reports a finite value there; else **3** if the centre falls in a Natural Earth 10 m *lakes* polygon; else **1** land. Ice before ocean because the surface an instrument sees on an ice shelf is ice. Rasterised with `shapely` (already a dependency of `refresh_data.py islands`) |
| `elev` | float32 `[721, 1440]` | ETOPO 2022 60-arc-second **ice-surface** elevation (NCEI, key-free), mean over the 0.25° cell centred on each grid point (15 × 15 source cells); negative under the sea |
| `lats`, `lons`, `lat1`, `lon1` | float64 | the axes above |

### Normalisation, truth, meta

- `norm_g025`, `norm_g100`, `norm_rg100`: `[C, 2]` float32 `(mean, sd)` per
  channel over **every finite value in every bin**, the family-4 convention
  (`build_family4.py:913-935`); the train-years-only baseline remains
  `trainprobe.anomaly_transform`'s job. Values are z-scored in place before
  the float16 write. **The `slab[~ocean] = NaN` line of family 4 is gone**:
  land is observed now.
- Truth keys exactly as family 4 (`build_family4.truth_pentad`, `rapid`
  alias) — the labels are Atlantic and the stage-2 gates need them; the
  builder refuses without them, as family 4's `missing_truth_keys` does.
- `bin_index`, `months`, `epoch`, `pentad_days`, `chan_g025`, `chan_g100`,
  `chan_rg100`, `groups = ["g025","g100","rg100"]`, `window = "global025"`,
  `recipe = "f7l0"`, `cadence = "pentad"`, per-stage counts
  (`n_glorys_bins`, `n_sst_days`, `n_ncep_days`, `n_rg_live`), and
  `sources` (the URL/product string of every input, for provenance).

## 3 · Files and the loader

```
family7_global025_pentad_l0.npz            meta, statics, norms, truth (small)
family7_global025_pentad_l0_X_g025.npy     memmap [3142,721,1440,7]  float16
family7_global025_pentad_l0_X_g100.npy     memmap [3142,181,360,15]  float16
family7_global025_pentad_l0_X_rg100.npy    memmap [n_live,181,360,32] float16
```

`ml/tensor_io.load_tensor(stem.npz)` learns the `_X_<group>.npy` form: the
returned object answers `d["X_g025"]`, `d["X_g100"]`, `d["X_rg100"]`, and
`d["X"]` stays an alias of the dense group so a family-4 consumer that only
wants the surface channels keeps working. `save_tensor` renames memmaps as
today. The single-group `_X.npy` form is unchanged.

## 4 · The job — `ml/build_family7.py`, resumable stage by stage

```
python3 ml/build_family7.py --work <dir> --stage all
       [--stage glorys|sst|ncep|rg|static|truth|norm|meta|publish]
       [--source-dir <dir>]   # tests: read every source from local files, no network
       [--smoke]              # tiny synthetic sources, whole path, seconds
```

Stage order is fixed (`ncep` needs `sst` for the `repair_sst_channel`
pass and the `oisst_seen` mask; `norm`
needs everything; `publish` last). Every stage writes its rows to the
memmap, **flushes, then** writes `<work>/<stage>.done` (§5.21: a marker may
only under-claim); `sst`, `ncep` and `rg` also keep per-year / per-cube
markers so a killed job resumes at the year it lost, not at 1982. Sources
are streamed: one GLORYS chunk (257 MB) or one OISST year (1.6 GB + 0.6 GB)
on disk at a time, deleted after use. Nothing is held in RAM larger than a
few bins of one group.

`publish` uploads the four files to the Hub with `HfApi.upload_file`,
downloads each back and compares sha256 (the `pentad-aggregate.yml` rule),
and writes `manifest.json` (name, bytes, sha256, recipe, git sha of the
builder, the `sources` block) beside them. A publish that cannot verify
fails the job.

**Sizing on the box** (`/opt/earth-cache` → `ml/cache`, the persistent
runner directory): 46 + 5.7 + 1 GB of memmaps, ≤ 3 GB of sources in flight,
plus the verify downloads streamed through a temp file. The coarse groups
are filled in **float32** and cast to float16 only at the z-score pass (raw
surface pressure written straight into float16 would lose 0.5 hPa ≈ 0.08 sd
before normalisation; `g025` stays float16 because its float32 twin would be
92 GB — worst case there is `sst` at 0.031 °C, family 4's own `sst`
precedent), so the on-disk **peak is 66 GB** (45.7 g025 + 12.3 g100 f32 +
6.1 g100 f16 + 2.1 rg100 f32; 66.2 GB, printed by `--dry-run`) and **≈ 75 GB free is what the preflight
demands**, checked before anything is written (over 90 % disk is
unusable, not a warning: `ml/CLAUDE.md` §7). Expected wall time: GLORYS 384
chunks ≈ 40 min · OISST 43 years ≈ 2 h (download-bound) · NCEP 14 vars ×
43 years ≈ 40 min · RG minutes · statics minutes · norm ≈ 30 min · publish +
verify ≈ 30 min — **about 5 h**, $1.5 at the 4090 box rate.

## 5 · What is asserted before the tensor is trusted (`tests/test_build_family7.py`, CPU, no network)

1. Axes: 721 / 1440 / 181 / 360, the values above, ascending; the coarse
   lookup maps `x = 1439 → 0` and `y = 2 → 1`, `y = 1 → 0`.
2. `bin_index` and `months` agree with `build_family4`'s for the same epoch.
3. The GLORYS chunk lands at row 40 (its `latitude[0] == -80.0` asserted at
   read time, not assumed).
4. On a synthetic OISST year the pentad means and the `min_days` rule
   reproduce `aggregate_cadence`'s arithmetic; `sst` is the OISST value
   wherever OISST is finite and MISSING everywhere else, and the shared
   surface temperature `skt` is finite over every surface in `g100`.
5. NCEP rules: `uflx`/`vflx` flipped, nothing else flipped; K → °C, Pa →
   hPa, the two `log1p`s; `soilw`/`tsoil` NaN over the gaussian sea mask;
   `skt` K → °C with no mask and no flip.
6. RG: NaN outside −64.5 … 79.5, the live bin is the one holding the 15th.
7. `sphere` priority on toy polygons; `elev` block mean on a toy raster.
8. `norm` round-trip: un-z-scoring a slab reproduces the source to float16.
9. `load_tensor` on a three-group stem returns all three groups and the
   `X` alias; the single-group form still loads.
10. `--smoke` runs the whole path end to end on synthetic sources in a temp
    directory and produces every file and key of §2–§3.

**And one measurement after the build, recorded in the E-070 log:** the NA
sub-block of `cur_u` against family 4 r3's `cur_u`, both un-z-scored — the
plan's bit-identity question (the global chunks were binned at fetch, the
NA chunks in the aggregator; the accumulation order differs). Report the
max |Δ| and the fraction of exactly-equal cells; do not assume zero.

## 6 · What this does NOT do (Phase E's list)

- **Consumers are not converted.** `cone_sampler` still refuses the
  dateline wrap (E-070 D4), reads one dense array, and `cone.channel_family`
  raises on the new names. The ~90-line sidecar change the 09-03 handoff
  priced becomes a *group* change: the sampler gathers each group by its own
  `(H, W, C)` and coarse lookup, then concatenates by channel. That is the
  first task of Phase E, with E-071 §1's destination-point dots and §2's
  harmonic climatology, and it is where `train_cone.load_data`'s
  `ocean = isfinite(X[..., 0])` becomes "any channel ever observed".
- **No ERA5, no MODIS, no GRACE, no SMAP, no altimetry** — Phase L1, after
  the CDS account (ERA5) and after the harmonic climatology has shown it
  handles a channel that starts in 2000.
- **No family-7 branch in `ml-train.yml`** (the file is at the 25-input
  ceiling; a `tensor` VALUE plus a Hub pull block is the shape, written when
  Phase E dispatches).
