# Family 7.1 — ingestion handover for an agent that already holds family 7

*Written 2026-09-14 for an agent (a Gemini instance, or any other) that has
already ingested family 7 (`f7l0`) from `docs/FAMILY7_DATA_HANDOVER.md` into
its own infrastructure and now needs family 7.1 (`f7l1`). It is a DELTA
document: it assumes the reader knows the grid, the time axis, the z-scoring,
the NaN convention and the three reading recipes of that handover, and it
describes only what is different, what to keep, what to replace, and how to
check the result. Every number is measured against the files published on
the Hub on 2026-09-14 unless it says otherwise.*

**One paragraph.** Family 7.1 is family 7 with a fourth channel group,
`oc025` — satellite chlorophyll-a (ESA OC-CCI v6.0) on the same 0.25° grid
and five-day bins, from September 1997 — stored with an OFFSET time axis
(1,997 rows, not 3,142). The earlier handover's §10 promised the three
inherited groups would be the same bytes as `f7l0`'s. **That is not what was
published.** The build was an unseeded rebuild, and the two things that
follow from it are the reason this document exists: (1) `f7l0`'s `sst` and
`sea_ice` channels were **all-NaN for the whole of calendar 1989** (73
pentads, bins 511–583) and `f7l1` fills that year, so `g025` is a strict
improvement but its z-score constants moved and the file must be replaced,
not reused; (2) the `f7l1` small file ships an **all-NaN `elev` static** —
take elevation from the `f7l0` npz you already have (same grid, same
values by construction) until it is republished. Everything else is either
byte-identical or differs at the last float16 bit.

---

## 1 · Files, side by side

Both folders are public, need no login, and answer HTTP `Range:` with 206.

```
https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l1/
```

| file (`family7_global025_pentad_l1…`) | bytes | sha256 | relation to your `f7l0` copy | action |
|---|---|---|---|---|
| `.npz` | 5,411,116 | `91a8d86f9c822b3e88ca0a47dca46fc396e266e18af67a9647641e265a2e5937` | 40 keys (was 32); `norm_g025` changed; `elev` all NaN (§3) | **fetch**, then patch `elev` from `f7l0` |
| `_X_g025.npy` | 45,670,101,248 | `1cb1e1af4132e6acdec76a7341a49453b32f9b982eee09d3ad577c5dd147d293` | **differs** — 1989 filled, sst/ice re-z-scored (§2.1) | **replace** the f7l0 file |
| `_X_g100.npy` | 6,141,981,728 | `ba59ed7c133bce3756f7aeb30ce37c6ce64fb2b86063967f74e0e70a247123a2` | differs in 1–2 cells per bin, 1 float16 ULP, two channels (§2.2) | replace if you key caches by hash; reusing f7l0's is numerically harmless |
| `_X_rg100.npy` | 1,050,900,608 | `55c49f1bbc55f9e82204d5d4ff2751cab5625c62003debe56df8cb5f1acaf2d5` | **byte-identical** to f7l0's | reuse; rename or symlink |
| `_X_oc025.npy` | 8,293,461,248 | `c50c79b5760103cdda1233fe6d6bf776fa03caf732a59b24cd27bd6a5cdd1700` | **new** — `[1997, 721, 1440, 2]` float16 | **fetch** |

`manifest.json` in the folder carries these values plus, per inherited
file, `same_as_f7l0` (false / false / true) and the f7l0 hash it was
compared against; `seeded_from_base: false`; builder commit `3a13f8a`;
`built_at` 2026-09-14T15:19:34Z. The npy headers are 128 bytes in all four
files (measured), little-endian float16, C order, bin-major — the §7(c)
range-read arithmetic of the family-7 handover is unchanged.

Total to download if you keep rg100 and g100: 5.4 MB + 45.67 GB + 8.29 GB.

---

## 2 · The three inherited groups — what actually changed

Measured 2026-09-14 by range-reading one pentad per calendar year (43 bins)
from both builds for `g025`, six bins for `g100`, and comparing raw
`uint16` views as well as values.

### 2.1 `g025` (0.25°, ocean): 1989 was missing from family 7

- The five GLORYS-derived channels — `cur_speed`, `log_mld`, `ssh`, `cur_u`,
  `cur_v` — are **bit-identical** in every sampled bin.
- `sst` and `sea_ice`: in `f7l0`, **bins 511–583 inclusive are all-NaN** for
  both channels — 73 pentads × 5 days = 365 days, 1988-12-30 → 1989-12-29
  (a truncated source transfer on the day f7l0 was built; the npz's
  `n_sst_days` reads 15,341 against 15,706 = every day 1982–2024). `f7l1`
  has the year: bin 547 (July 1989) reads 703,902 finite `sst` cells and
  153,002 finite `sea_ice` cells, exactly the counts of its neighbours.
  Bin 511 straddles into 1988 (two December days were also void) and bin
  584's f7l0 mean was taken over 1990 days only.
- Because the record is now complete, the z-score constants were recomputed:

  | channel | `f7l0` (mean, sd) | `f7l1` (mean, sd) |
  |---|---|---|
  | `sst` | (13.615001, 11.622964) | (13.609584, 11.620165) |
  | `sea_ice` | (0.8271252, 0.23922408) | (0.82670176, 0.2397769) |

  The other five rows of `norm_g025` are identical. Consequence: in every
  non-1989 bin about two thirds of finite `sst`/`sea_ice` cells differ by
  exactly one float16 ULP (≤ 9.77e-4 z ≈ 0.0113 K; ≤ 9.77e-3 z ≈ 0.0023 ice
  fraction). The NaN mask is bit-for-bit the same everywhere except 1989.

**What this means for you.** A model or statistic derived from `f7l0` saw a
year-long hole in its two most important observed surface channels; anything
that depends on 1989 (climatologies, training folds, persistence baselines)
should be recomputed on `f7l1`. Do not mix the files: `f7l1`'s `g025` must be
decoded with `f7l1`'s `norm_g025`. If your loader keys a cache on the file
hash, the new hash forces the refresh you want.

### 2.2 `g100` (1°, atmosphere and land): equal within float noise

`norm_g100` is bit-identical between builds. Thirteen of fifteen channels are
byte-identical in every sampled bin (`tau_x`, `tau_y`, `tau_x_std`,
`tau_y_std`, `t2m`, `u10`, `v10`, `sp`, `soilw`, `tsoil`, `lhtfl`, `shtfl`,
`skt`). `log_prate` differs in 1–2 of 65,160 cells per bin and `log_swe` in
one cell of one bin, by one float16 ULP (≤ 9.77e-4 z, ≤ 7.5e-4 in log units)
— accumulation-order noise in the log transform, no NaN-payload or signed-zero
differences. Scientifically nothing; the hash differs, which is why the
manifest says `same_as_f7l0: false`. Either file is correct to use.

### 2.3 `rg100` (1°, monthly ocean interior): identical

Same sha256 as `f7l0`. Same 252 live rows, same `rg_bin_index`. Nothing to do.

---

## 3 · The small file (`.npz`): new keys, one changed array, one defect

Everything in `f7l0`'s 32 keys is present with the same value except
`builder_git_sha`, `built_at`, `groups`, `recipe`, `sources`, `n_sst_days`
(15,706), `norm_g025` (§2.1) — and `elev`.

**New keys** (all about `oc025`):

| key | dtype / shape | value |
|---|---|---|
| `chan_oc025` | `<U8 [2]` | `["log_chl", "chl_cov"]` |
| `norm_oc025` | float32 `[2, 2]` | `[[-0.7764689, 0.4566048], [0.28230456, 0.21373105]]` — (mean, sd) per channel |
| `count_oc025` | int64 `[2]` | `[855742985, 855742985]` finite values per channel (41.3 % of 1997 × 721 × 1440) |
| `oc_bin_first` | int64 | **1145** — the pentad holding 1997-09-04 |
| `oc025_bin_index` | int64 `[1997]` | `1145, 1146, …, 3141` — contiguous, so `row = bin − 1145` is exact |
| `n_occci_days` | int64 | 9,955 daily 4 km fields read |
| `n_occci_absent` | int64 | 26 days the archive's own listing did not offer |
| `n_oc_inland` | int64 | 4,588,447 finite `log_chl` cell-pentads on land cells that touch no sea (measured, not masked — §4) |
| `groups` | `<U5 [4]` | `["g025", "g100", "rg100", "oc025"]` |
| `recipe` | | `"f7l1"` |

**Unchanged and verified equal:** `lats`, `lons`, `lat1`, `lon1`,
`bin_index` (3142), `rg_bin_index` (252), `months`, `rg_months`, `epoch`,
`cadence`, `pentad_days`, `window`, `sphere` (int8 mask, 4 classes),
`truth_rapid` `[1459, 2]`, `truth_fc` `[2490, 2]`, `rapid`, `norm_g100`,
`norm_rg100`, `n_glorys_bins`, `n_ncep_days`, `n_rg_live`.

**The defect: `elev` is all NaN.** `f7l1.npz["elev"]` is `[721, 1440]`
float32 with **0 finite values**; `f7l0`'s has 1,038,240 (every cell,
−9,688 m to 6,004 m). The rebuild's `sources` record has no `etopo` entry —
the ETOPO 2022 fetch did not happen on the box that built it (the same
throughput-guard failure class that voided `rg100` in run #10, caught for
`rg100` and not for the static). Elevation is a static, grid-only field, so
**use `f7l0.npz["elev"]` verbatim** — it is the correct array for this grid —
and treat `f7l1`'s as absent. A republished npz will be announced in
`ml/EXPERIMENTS.md#e-077`; when it lands, `elev` should hash equal to
`f7l0`'s.

---

## 4 · `oc025` — the new group

`[1997, 721, 1440, 2]` float16, z-scored like everything else, ocean colour
from ESA OC-CCI v6.0 (merged SeaWiFS / MERIS / MODIS / VIIRS / OLCI, 4 km,
1997-09-04 → 2024-12-31), box-averaged onto the 0.25° point grid: the 6 × 6
block of 4 km cells within ±0.125° of each grid point.

| idx | name | unit after un-z-scoring | rule |
|---|---|---|---|
| 0 | `log_chl` | log₁₀(mg m⁻³) | per day: mean of `log10(chlor_a)` over the block's finite cells; per pentad: mean of those daily means over days with ≥ 1 finite cell |
| 1 | `chl_cov` | fraction in (0, 1] | finite 4 km cell-days in the block during the pentad ÷ (cells in block × 5). NaN exactly where `log_chl` is NaN, never 0 |

**Three things that are different from every other group.**

1. **The time axis is offset, not padded.** Row `r` holds pentad bin
   `1145 + r`. Bins < 1145 (before 1997-09-04) have no row — that is the
   period before any of these satellites flew, not a gap. Use
   `oc025_bin_index` as the lookup if your loader is table-driven, or
   `bin − oc_bin_first` if it is arithmetic; they agree at every row.
2. **The value is a logarithm, then z-scored.** Full inversion:
   `log10_value = stored × 0.4566048 + (−0.7764689)`, then
   `mg_per_m3 = 10 ** log10_value`. A stored z of 0 is 0.167 mg m⁻³.
3. **`chl_cov` is the honesty channel.** A `log_chl` averaged from one clear
   pixel on one day (`chl_cov` = 1/180 ≈ 0.0056) and one from all 36 pixels on
   all 5 days (1.0) are the same kind of number in the array and mean very
   different things. Use it to weight, threshold, or mask. At the two pole
   rows only 18 source cells exist and the denominator counts the cells that
   exist.

**Coverage, so you know what "normal" looks like.** Measured on bin 2411
(2015-01-03, row 1266): 421,137 finite cells of 1,038,240 (40.6 %); on
`sphere == 0` (ocean) cells 59.9 % finite, on land cells 0.11 % finite;
`log_chl` range −2.24 … +1.98 (0.006 … 95 mg m⁻³), median 0.143 mg m⁻³;
`chl_cov` range 0.0055 … 0.9999; `chl_cov` NaN exactly where `log_chl` is
NaN — verified on the whole frame. Over the record 41.3 % of all
cell-pentads are finite. January at 40° N 30° W is NaN (cloud); the
equatorial Pacific and the Southern Ocean bloom are finite.

**Colour over land is real and was left in.** The 4 km product resolves
estuaries, lagoons and shelf water that the 0.25° `sphere` mask calls land.
Those values are not masked; `n_oc_inland` = 4,588,447 counts the finite
cell-pentads on land cells that do not touch the sea, so you can mask them
yourself with `sphere` if your use wants open ocean only.

**Where the colour came from** (recorded in `sources`): 1997–2022 from
CEDA's per-file v6.0 archive, 2023–2024 from Plymouth Marine Laboratory's
`CCI_ALL-v6.0-DAILY` aggregate subset one day at a time; both reduced by the
same code, folded from per-year partials with a bit-identity test against
the sequential path.

---

## 5 · Ingesting it — the minimal loader changes

If your family-7 loader is the one the earlier handover's §7 describes (a
per-group memmap or range reader plus `bin → row`), the whole delta is:

```python
import numpy as np

m  = np.load("family7_global025_pentad_l1.npz", allow_pickle=True)
m0 = np.load("family7_global025_pentad_l0.npz", allow_pickle=True)   # you already have it

groups = list(m["groups"])                       # ["g025","g100","rg100","oc025"]
chans  = {g: list(m["chan_" + g]) for g in groups}
norm   = {g: m["norm_" + g]       for g in groups}   # USE f7l1's norm_g025, not f7l0's

# statics: sphere from f7l1 (equal to f7l0's); elev from f7l0 until republished
sphere = m["sphere"]
elev   = m0["elev"]                              # f7l1's elev is all-NaN (2026-09-14)
assert np.isfinite(elev).all()

# bin -> row, per group. f7l1 adds exactly one case.
def row_of(group, b):
    if group == "rg100":
        idx = np.searchsorted(m["rg_bin_index"], b)
        return int(idx) if idx < len(m["rg_bin_index"]) and m["rg_bin_index"][idx] == b else None
    if group == "oc025":
        r = b - int(m["oc_bin_first"])           # 1145
        return r if 0 <= r < 1997 else None       # None: before 1997-09-04
    return b                                      # g025, g100: row == bin

# shapes for range reads (header 128 in all four files, measured)
SHAPE = {"g025": (721, 1440, 7), "g100": (181, 360, 15),
         "rg100": (181, 360, 32), "oc025": (721, 1440, 2)}
```

Then read `oc025` exactly like `g025` with `SHAPE["oc025"]` and
`row_of("oc025", b)`, un-z-score with `norm["oc025"]`, and exponentiate
channel 0 if you want mg m⁻³. Nothing about `g100`/`rg100` reading changes;
`g025` reading is unchanged too — only its bytes and its norm did.

**If your infrastructure re-normalises or standardises from training
years**, recompute those statistics on `f7l1`'s `g025` — the 1989 hole in
`f7l0` biased nothing visibly (NaN is excluded), but 1989 now exists and any
per-year table or fold definition that skipped it should stop skipping it.

**If you keep a flat channel view** (the 54-channel map of the earlier
handover's §3), `oc025` appends channels 54 and 55 (`log_chl`, `chl_cov`);
a flat row for a bin below 1145 has those two as "absent", which is a
different state from NaN inside the group (NaN = observed nothing this
pentad; absent = no satellite yet). Represent that distinction if your token
design has a missing-vs-unobserved split; otherwise NaN is the conservative
choice.

---

## 6 · Validate after ingest — the numbers to reproduce

1. sha256 of every file equals §1's table (the manifest is the same list).
2. `np.load(npz)`: 40 keys; `groups` has 4 entries; `recipe == "f7l1"`;
   `oc_bin_first == 1145`; `len(oc025_bin_index) == 1997` and
   `oc025_bin_index[-1] == 3141`; `n_sst_days == 15706`.
3. `g025`, bin 547 (1989-07): `sst` finite count **703,902**, `sea_ice`
   **153,002** (in `f7l0` both are 0 — that is the difference you are
   ingesting).
4. `oc025`, row 1266 (bin 2411, 2015-01-03): **421,137** finite `log_chl`
   cells; global mean `log_chl` after un-z-scoring **−0.8295**
   (0.148 mg m⁻³); mean `chl_cov` **0.249**; `isnan(chl_cov) == isnan(log_chl)`
   everywhere.
5. `elev` in your store: 1,038,240 finite values (i.e. you took `f7l0`'s).
6. `rg100` hash `55c49f1b…` — unchanged, so any rg-derived cache is valid.

---

## 7 · Provenance and caveats, briefly

- Built 2026-09-14 on a rented box from builder commit `3a13f8a`, run #10
  (all four groups, unseeded) plus run #11 (`rg100` rebuilt in place after
  #10 published it as a 128-byte empty array — caught by reading byte
  counts, not the run's colour). The publish step restore-verified every
  file; the sandbox re-verified the npz hash and the range reads above.
- The earlier handover's §10 "same bytes, hard-linked" description is the
  DESIGN; the published artefact is the unseeded rebuild described here.
  The inherited-group differences are fully characterised in §2 and are
  either an improvement (1989) or float noise.
- Open defect: the all-NaN `elev` static (§3). Tracked in
  `ml/EXPERIMENTS.md#e-077`; workaround is exact, not approximate.
- Holdout convention (earlier handover §9) is unchanged; colour has 21 full
  training years to fit a climatology from.
- Specification with every decision: `ml/plans/E077_family7_ocean_colour.md`
  ([rendered](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E077_family7_ocean_colour.md)).
  The family-7 base document this one extends:
  `docs/FAMILY7_DATA_HANDOVER.md`
  ([rendered](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY7_DATA_HANDOVER.md)).
