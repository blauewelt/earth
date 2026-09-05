# Family 7 — the global input tensor: a self-contained data handover

*Written 2026-09-05 for whoever picks the data up next — a person or a model
with no access to the rest of this repository. Everything needed to load,
decode and correctly interpret the files is in this one document. Where a
statement is a measurement it says so; where it is a design decision it names
the decision.*

**What this is.** Family 7 (recipe id `f7l0`) is a gridded record of the
Earth's surface state — ocean, atmosphere, land and ice — on a regular 0.25°
latitude/longitude grid covering the whole planet from the South Pole to the
North Pole, in five-day time steps from 1982-01-01 to the end of 2024. It is
the input tensor of a forecasting model: the model reads it, learns to
predict how the state evolves, and is scored against the two "truth" series
that ship inside the same file. Everything in it comes from public sources
(GLORYS12 ocean reanalysis, NOAA OISST satellite sea-surface temperature and
sea ice, NCEP/NCAR Reanalysis 1 for the atmosphere and land surface,
Roemmich–Gilson Argo for the ocean interior, ETOPO 2022 for elevation, Natural
Earth for the ice-sheet and lake outlines).

---

## 1 · The files

Four files, published in the Hugging Face dataset repository
`chfrank/earth-tensors` under the folder
`tensors/family7_global025_pentad_l0/`. They are public, need no login, and
the host answers HTTP `Range:` requests (status 206), so a slice can be read
without downloading the whole file.

| file | what | size | sha256 |
|---|---|---|---|
| `family7_global025_pentad_l0.npz` | **the small file**: axes, channel names, normalisation, statics, truth series, provenance | 5.4 MB | `27cd45eb71e8c22a817e3ce06ca4dbdb8e25724ed5706f1a9e7bfc3d3d5de8fd` |
| `family7_global025_pentad_l0_X_g025.npy` | group **g025** — 7 ocean channels at 0.25° | 45.67 GB (45,670,101,248 B) | `dcc4eaf1ab4425d7ac0e95a22a103f7e752036ac3e3ea746db3662f34ff9d61d` |
| `family7_global025_pentad_l0_X_g100.npy` | group **g100** — 15 atmosphere/land channels at 1° | 6.14 GB (6,141,981,728 B) | `fa8966bbebc528eb20234e17ee46dee958fbd8dc77bf615081cb7bcdccbbef7a` |
| `family7_global025_pentad_l0_X_rg100.npy` | group **rg100** — 32 ocean-interior channels at 1°, monthly | 1.05 GB (1,050,900,608 B) | `55c49f1bbc55f9e82204d5d4ff2751cab5625c62003debe56df8cb5f1acaf2d5` |

Base URL for all four (append the file name):
`https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l0/`

A `manifest.json` in the same folder repeats the sizes and hashes and records
the builder's git commit (`56d07f4`, built 2026-09-04 17:00 UTC).

The three `.npy` files are ordinary NumPy array files: a 128-byte header
(`header_len` = 128 for all three, measured) followed by little-endian
**float16** data in C (row-major) order, shape as in §3. `numpy.load(path,
mmap_mode="r")` opens them without reading them; a raw reader needs only the
offset arithmetic in §7.

**Do not try to hold g025 in RAM** (45.67 GB stored, 91 GB as float32).
Everything below is designed to be read one time step at a time.

---

## 2 · Grid, time axis, and the two resolutions

**Latitude/longitude.** Point-aligned (the grid points ARE the multiples of
the step; no half-cell offset), **south first**:

```
0.25° grid (g025):  lats = -90 + 0.25*arange(721)   → -90.00 … +90.00   (721 rows)
                    lons = -180 + 0.25*arange(1440)  → -180.00 … 179.75 (1440 columns)
1°    grid (g100, rg100): lat1 = arange(-90, 91)     → 181 rows
                          lon1 = arange(-180, 180)   → 360 columns
```

Row 0 is the South Pole, the last row the North Pole. Longitude wraps: column
1439 (179.75° E) is adjacent to column 0 (−180°). These axes are stored in the
npz as `lats`, `lons`, `lat1`, `lon1`.

**Time.** A "pentad" here is a fixed five-day bin counted from the epoch
1982-01-01: bin `b` covers days `[5b, 5b+5)` after the epoch, i.e.

```
bin_start_date(b) = date(1982,1,1) + timedelta(days=5*b)
bin_of(date)      = (date - date(1982,1,1)).days // 5
```

There are **3,142 bins** (0 … 3141), the last opening on 2024-12-31. A
five-day bin is a calendar-free unit: bins do not align with months or years,
and a year holds 73 of them. The npz carries `bin_index` (0 … 3141) and
`months` (the `YYYY-MM` of each bin's opening day, for grouping).

**The 1° groups are looked up from a 0.25° position, not upsampled.** The
design decision (E-070 B2) is that a 1.9° reanalysis stored at 0.25° would be
sixty copies of every number, so the coarse channels live on their own grid
and a consumer reads them at the nearest coarse cell:

```
y1 = min(round_half_up(y / 4), 180)          # round(y/4) with .5 rounding up
x1 = round_half_up(x / 4) mod 360
```

Every fourth 0.25° point IS a 1° point; the two points on either side round
to it. `x = 1438, 1439` round to 360 and wrap to column 0 (−180°, the same
meridian as +180°).

---

## 3 · The three groups and their channels

Each group is one array `[T, H, W, C]`, float16, **z-scored per channel**
(§4). `NaN` means "not observed here at this time" and is meaningful (§5).

### g025 — `[3142, 721, 1440, 7]`, 0.25°, ocean only

| idx | name | unit (after un-z-scoring) | source · rule | mean | sd | present where |
|---|---|---|---|---|---|---|
| 0 | `cur_speed` | m/s | GLORYS12 `uo`,`vo` daily → pentad mean of the daily speeds, `hypot`; needs ≥ 3 days in the bin | 0.1580 | 0.1594 | ocean, rows 40+ (GLORYS ends at 80° S), from 1993 (bin 803) |
| 1 | `log_mld` | log₁₀(m) | GLORYS12 `mlotst` mixed-layer depth, `log10` where > 0 | 1.4785 | 0.3587 | as above |
| 2 | `ssh` | m | GLORYS12 `zos` sea-surface height above geoid, pentad mean | −0.1668 | 0.7129 | as above |
| 3 | `cur_u` | m/s | GLORYS12 `uo` eastward current, pentad mean (same pass as ch 0, so `hypot(cur_u,cur_v) == cur_speed`) | −0.0007 | 0.1793 | as above |
| 4 | `cur_v` | m/s | GLORYS12 `vo` northward current | 0.0059 | 0.1349 | as above |
| 5 | `sst` | °C | NOAA OISST v2.1 daily sea-surface temperature, bilinear to the point grid, pentad mean, ≥ 3 days. **Observed; NaN over land and wherever OISST does not report** — never filled from a model | 13.615 | 11.623 | ocean, all bins from 1982 |
| 6 | `sea_ice` | fraction 0.15–1 | NOAA OISST v2.1 daily `icec` sea-ice concentration (the file says "percent" but its valid range is 0–1 and the reader trusts the range), pentad mean | 0.8271 | 0.2392 | **only where ice concentration ≥ 0.15**; NaN elsewhere — open water, land, ice shelves |

**`sea_ice` is NaN wherever there is less than 15 % ice, not zero.** Measured
on the published bytes (bin 2411, 2015-01-03): 155,141 finite cells of
1,038,240, every one of them in [0.150, 1.000], no zeros; a cell in the
mid-Atlantic (30° N 40° W) reads `sst` 21.3 °C and `sea_ice` NaN. That is how
OISST reports ice (it masks below 15 %), and it is why the channel's mean is
0.83. For "ice fraction with zeros" use `where(isfinite(sst) & isnan(sea_ice),
0, sea_ice)`.

### g100 — `[3142, 181, 360, 15]`, 1°, every surface (land, sea, ice)

All from NCEP/NCAR Reanalysis 1 daily Gaussian-grid files (T62 ≈ 1.9°),
bilinear to the 1° point grid, pentad mean with ≥ 3 days, all bins from 1982.

| idx | name | unit | source · transform | mean | sd | notes |
|---|---|---|---|---|---|---|
| 0 | `tau_x` | N/m² | `uflx.sfc` momentum flux, **sign-flipped** so it is the stress the wind exerts ON the surface, eastward | 0.0103 | 0.1246 | |
| 1 | `tau_y` | N/m² | `vflx.sfc`, sign-flipped, northward | 0.0042 | 0.0951 | |
| 2 | `tau_x_std` | N/m² | population σ of the daily `tau_x` within the pentad | 0.0825 | 0.0724 | a measure of storminess |
| 3 | `tau_y_std` | N/m² | same, northward | 0.0808 | 0.0721 | |
| 4 | `t2m` | °C | `air.2m`, K → °C | 4.807 | 21.999 | |
| 5 | `u10` | m/s | `uwnd.10m`, eastward wind at 10 m (no sign flip) | 0.0027 | 4.368 | |
| 6 | `v10` | m/s | `vwnd.10m`, northward | 0.1557 | 3.167 | |
| 7 | `sp` | hPa | `pres.sfc`, Pa → hPa | 965.05 | 94.76 | surface pressure, so low over plateaus and ice sheets |
| 8 | `log_prate` | log1p(mm/day) | `prate.sfc` kg m⁻² s⁻¹ × 86400 → mm/day, then `log1p` | 0.8662 | 0.7683 | undo: `expm1(x)` mm/day |
| 9 | `log_swe` | log1p(mm w.e.) | `weasd.sfc` snow water equivalent, kg m⁻² (= mm), `log1p` | 2.0701 | 3.5659 | undo: `expm1(x)` mm |
| 10 | `soilw` | volumetric fraction | `soilw.0-10cm`; **NaN over sea** (land mask applied before regridding, so coasts renormalise correctly) | 0.3040 | 0.0827 | |
| 11 | `tsoil` | °C | `tmp.0-10cm`, K → °C, land-masked like `soilw` | −9.684 | 33.661 | the mean is low because the 1° land mask includes Antarctica and Greenland |
| 12 | `lhtfl` | W/m² | `lhtfl.sfc` latent heat flux, NCEP sign (positive = upward, surface losing heat) | 62.30 | 62.61 | |
| 13 | `shtfl` | W/m² | `shtfl.sfc` sensible heat flux, positive = upward | 5.719 | 40.15 | |
| 14 | `skt` | °C | `skt.sfc` skin temperature, K → °C. **The one shared surface temperature: defined over every surface, no land mask, no splice with `sst`** | 5.088 | 22.72 | |

**Two temperatures, on purpose (decision B4 / B4c).** `sst` (g025) is the
observed satellite sea-surface temperature and is NaN over land; `skt` (g100)
is the reanalysis skin temperature everywhere. They are two different
instruments and were deliberately NOT merged into one channel, because a
splice at the coastline would be an instrument boundary the model would have
to learn as if it were physics. A land cell's temperature comes from `skt`,
`t2m`, `tsoil`; an ocean cell has both `sst` and `skt`.

### rg100 — `[252, 181, 360, 32]`, 1°, monthly, ocean interior

Roemmich–Gilson Argo climatology + monthly extensions: ocean temperature (°C)
and salinity (PSU) at sixteen pressure levels. Channel order:

```
 0-15  rg_t10  rg_t30  rg_t50  rg_t100 rg_t150 rg_t200 rg_t300 rg_t400
       rg_t500 rg_t700 rg_t900 rg_t1100 rg_t1300 rg_t1500 rg_t1700 rg_t1900   (°C, level in dbar)
16-31  rg_s10 … rg_s1900  (same levels, PSU)
```

Means run from 17.87 °C at 10 dbar to 2.45 °C at 1900 dbar and 34.8–35.0 PSU;
the exact (mean, sd) per channel is `norm_rg100` in the npz.

**This group has its own time axis.** It holds one row per MONTH, 252 of
them, 2004-01 … 2024-12, and each monthly field is assigned to the pentad
that contains that month's 15th. `rg_bin_index` (int64 [252]) lists those
bins in order (1609, 1616, 1621, …, 3137) and `rg_months` the `YYYY-MM`. A
consumer wanting the ocean interior at bin `b` looks `b` up in
`rg_bin_index`; any bin not listed has no observation. Values are NaN outside
the product's latitude band (about 64.5° S … 79.5° N) — the builder masks the
band explicitly rather than replicating the edge rows toward the poles.

### The channel → group map, for a flat 54-channel view

Consumers that want one channel list use the concatenation
`chan_g025 + chan_g100 + chan_rg100` (7 + 15 + 32 = 54) and remember that
indices 7–21 are looked up on the 1° grid and 22–53 on the 1° grid at monthly
bins only.

---

## 4 · Values are stored z-scored — undo it before printing a unit

Every channel was standardised before the float16 write:

```
stored = (value_in_unit − mean_c) / sd_c
value_in_unit = stored * sd_c + mean_c
```

with `(mean_c, sd_c)` computed over **every finite value in every bin** of
that channel (not over training years only) and stored as
`norm_g025 [7,2]`, `norm_g100 [15,2]`, `norm_rg100 [32,2]` — column 0 the
mean, column 1 the sd, indexed within the group. A stored value of 0.31 in
the `sst` channel is not 0.31 °C; it is 13.6 + 0.31 × 11.6 ≈ 17.2 °C.

float16 keeps about 3 significant digits, so a z-score near 1 is resolved to
~0.001 sd (≈ 0.01 °C for `sst`, ≈ 0.0002 m/s for currents). Values beyond
±65,504 sd cannot occur.

**Anomalies are NOT stored.** The training code derives anomalies at load
time (`ml/trainprobe.py::anomaly_transform`: a per-calendar-month climatology
fitted on training years only, subtracted, then re-standardised per channel
over the training pool). If you need "departure from normal" you compute it
yourself; the tensor holds the un-anomalised state.

---

## 5 · What NaN means, and where to expect it

`NaN` is the tensor's word for "no observation here at this time". It is
information, not damage, and the pattern is systematic:

- **Ocean channels over land** (`cur_*`, `ssh`, `log_mld`, `sst`, `sea_ice`):
  always NaN. Use `sphere` (§6) to know which cells are land.
- **GLORYS channels south of 80° S** (rows 0–39 of the 0.25° grid) and
  **before 1993** (bins 0–802): NaN — the ocean reanalysis does not cover
  them. `sst` and `sea_ice` do exist from 1982.
- **`soilw`, `tsoil` over sea**: NaN by construction (land-masked).
- **`sea_ice` below 15 % concentration**: NaN (OISST's own mask; see the
  g025 table) — open water is NaN, not 0.
- **`rg100` outside 2004–2024, outside the Argo latitude band, or in any bin
  not listed in `rg_bin_index`**: absent.
- **Isolated pentads with fewer than 3 days of source data**: NaN for the
  channels whose rule says "≥ 3 days".

The build counted **~13.8 billion finite values** in total (10.69 B in g025,
2.82 B in g100, 0.26 B in rg100), against the ~23 B slots of g025 alone —
i.e. roughly half of g025 is NaN, mostly land and the pre-1993 ocean.

---

## 6 · Statics and the truth series (all inside the npz)

| key | dtype / shape | meaning |
|---|---|---|
| `sphere` | int8 `[721, 1440]` | surface class of each 0.25° cell: **0 ocean** (702,642 cells) · **1 land** (226,495) · **2 ice sheet or glacier** (107,074; from Natural Earth 10 m glaciated areas — includes Antarctica, Greenland, ice shelves) · **3 inland water** (2,029; Natural Earth lakes). Ice wins over ocean where both apply, because the surface an instrument sees on an ice shelf is ice |
| `elev` | float32 `[721, 1440]` | ETOPO 2022 ice-surface elevation in metres, mean over the 0.25° cell; **negative under the sea** (ocean depth) |
| `truth_rapid` (alias `rapid`) | float32 `[1459, 2]` | column 0 = pentad bin, column 1 = the RAPID array's Atlantic meridional overturning transport at 26.5° N in Sverdrups (1 Sv = 10⁶ m³/s), averaged into that bin. Bins 1626 (2004-04) … 3084 (2024-03). This is the programme's headline target |
| `truth_fc` | float32 `[2490, 2]` | column 0 = bin, column 1 = Florida Current cable transport (Sv) through the Florida Straits. Bins 15 (1982-03) … 3141, with gaps |

Both truth series are Atlantic; they exist so a global model can be scored on
the same instrument the North Atlantic models were.

Other npz keys: `bin_index`, `months`, `epoch` (`"1982-01-01"`),
`pentad_days` (5), `lats`, `lons`, `lat1`, `lon1`, `chan_g025`, `chan_g100`,
`chan_rg100`, `groups` (`["g025","g100","rg100"]`), `window`
(`"global025"`), `recipe` (`"f7l0"`), `cadence` (`"pentad"`),
`rg_bin_index`, `rg_months`, `norm_*`, per-stage counts (`n_glorys_bins` 2339,
`n_sst_days` 15341, `n_ncep_days` 15706, `n_rg_live` 252), `sources` (a JSON
string with every input URL), `builder_git_sha`, `built_at`.

---

## 7 · Reading it — three recipes

**(a) Everything small, one call.**

```python
import numpy as np
m = np.load("family7_global025_pentad_l0.npz", allow_pickle=True)
chans = {g: list(m["chan_" + g]) for g in m["groups"]}
norm  = {g: m["norm_" + g] for g in m["groups"]}       # [C,2] (mean, sd)
sphere, elev = m["sphere"], m["elev"]
rapid = m["truth_rapid"]                                # [N,2] (bin, Sv)
```

**(b) One time step of one group from a local copy (memmap; nothing else is
read).**

```python
X = np.load("family7_global025_pentad_l0_X_g025.npy", mmap_mode="r")  # [3142,721,1440,7] float16
b = (np.datetime64("2015-01-03") - np.datetime64("1982-01-01")).astype(int) // 5   # → 2411
frame = np.asarray(X[b], dtype=np.float32)              # [721,1440,7], 29 MB
c = chans["g025"].index("sst")
sst_C = frame[..., c] * norm["g025"][c, 1] + norm["g025"][c, 0]   # °C, NaN over land
```

**(c) One time step straight from the Hub with an HTTP range request (no
download of the 46 GB file).** Offsets are exact because the array is
bin-major C order:

```python
import numpy as np, urllib.request
H, W, C, header = 721, 1440, 7, 128                    # g025; g100: 181,360,15,128; rg100: 181,360,32,128
slab = H * W * C * 2                                    # bytes per bin (14,535,360 for g025)
url = ("https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/"
       "tensors/family7_global025_pentad_l0/family7_global025_pentad_l0_X_g025.npy")
b = 2411
req = urllib.request.Request(url, headers={"Range": f"bytes={header + b*slab}-{header + (b+1)*slab - 1}"})
with urllib.request.urlopen(req) as r:
    assert r.status == 206, "the host ignored the Range header — do not read on"
    buf = r.read()
frame = np.frombuffer(buf, dtype="<f2").reshape(H, W, C).astype(np.float32)
```

For `rg100` the first axis is the 252 live rows, not the 3,142 bins: use the
position of your bin in `rg_bin_index`.

**Reading a coarse channel at a 0.25° position** (the lookup of §2):

```python
def coarse_index(y, x):                                 # 0.25° row/col → 1° row/col
    y1 = min(int(np.floor(y / 4 + 0.5)), 180)
    x1 = int(np.floor(x / 4 + 0.5)) % 360
    return y1, x1
```

---

## 8 · Provenance and what was checked before the tensor was trusted

- Built by `ml/build_family7.py` at commit `56d07f4` in
  `github.com/blauewelt/earth`, on a rented GPU box, 2026-09-04, resumable
  stage by stage (GLORYS ~40 min, OISST 75 min, NCEP 50 min, Argo 28 s,
  statics 2 min, normalisation 11 min, publish + restore-verify 46 min).
  Every published file was downloaded back and re-hashed before the manifest
  was written.
- Spec with every decision and its reason:
  `ml/plans/E070_family7_build.md` in the same repository
  ([rendered](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md)).
- The ocean channels were compared against the earlier North Atlantic tensor
  (family 4, same sources, built separately) on one pentad (bin 2411,
  2015-01-03) over the 86,698 common ocean cells: currents, SSH and
  mixed-layer depth agree to two float16 quanta on 99.2–99.5 % of cells
  (r ≥ 0.99995); a ~1 % tail differs by up to 0.19 m/s because the daily
  fields were binned in a different order at partially masked coastal blocks;
  `sst` agrees to 0.03 °C. So: equal to float16 almost everywhere, not
  bit-identical. Family 7 also observes 1,756 more coastal `sst` cells than
  family 4 (a NaN-aware regridding reaches cells the old one left empty).
- Ten build-time assertions (`tests/test_build_family7.py`): axis orientation
  (row 0 is the South Pole, asserted on the bytes), `hypot(cur_u, cur_v) ==
  cur_speed`, the NaN pattern south of 80° S, the icec unit, the sphere class
  counts, the norm shapes, the truth keys, resumability.

**Known caveats.**

1. **The near-pole geometry of the forecasting model's stencil is
   approximate** — nothing in the tensor, but if you build a sampler that
   offsets dots in kilometres from an anchor, a flat-earth offset at 85° N
   lands 47 % of dots more than one cell off (max 206 km). Use a
   destination-point (great-circle) formula near the poles.
2. **NCEP/NCAR Reanalysis 1 is a stand-in** for ERA5 (which needs a
   Copernicus CDS account). The layout is designed so ERA5 can replace the
   g100 fetch without changing any shape.
3. **`sea_ice` (a fraction) and `log_swe` (a water mass) are two channels**,
   not a merged "frozen fraction"; merging a fraction with a mass would need
   an invented constant.
4. The twelve exported "cone sample" JSON files on the Hub
   (`cone_samples_f7/`, 24 pentads at 12 anchor cells, a demo artefact) carry
   a wrong placeholder unit string for the g100 and rg100 channels; the units
   in §3 of this document and in `data/family7_index.json` are the correct
   ones. The tensor itself carries no unit strings at all — units are this
   document's and the spec's.

---

## 9 · Holdout convention, if you train on it

The programme's evaluation protocol holds out whole years so that no five-day
neighbour of a test bin is ever a training bin: **2009, 2017 and 2023 are
development holdout years**, and everything after **2020** is the terminal
test period for the final read-out (`terminal_train_last_year = 2020`).
Whatever model you fit, keep those years out of the fit if you want its
numbers to be comparable with the ones in `ml/EXPERIMENTS.md`.
