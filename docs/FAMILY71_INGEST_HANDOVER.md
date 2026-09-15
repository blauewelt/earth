# Family 7.1 — ingestion handover for an agent that already holds family 7

PDF design note: https://blauewelt.github.io/earth/ml/paper/notes/family72.pdf

*Written 2026-09-14 for an agent (a Gemini instance, or any other) that has
already ingested family 7 (`f7l0`) from `docs/FAMILY7_DATA_HANDOVER.md` into
its own infrastructure and now needs family 7.1. It is a DELTA document: it
assumes the reader knows the grid, the time axis, the z-scoring, the NaN
convention and the three reading recipes of that handover, and it describes
only what is different, what to keep, what to replace, and how to check the
result. Every number is measured against the files published on the Hub on
2026-09-14 unless it says otherwise.*

**WHICH BUILD TO TAKE: `f7l2`, stem `family7_global025_pentad_l2`.** Family
7.1 was published twice on 2026-09-14. The first build, `f7l1` (15:19Z),
shipped an **empty elevation static** — every one of its
1,038,240 cells was NaN — and this document used to tell you to patch around
it. **That workaround is gone: take `f7l2` (19:58Z) and nothing is missing.**
If you already ingested `f7l1`, §1.1 says exactly what to re-fetch (the 5.4 MB
small file, and the 1° atmosphere group only if you key caches by hash).

**One paragraph.** Family 7.1 is family 7 with a fourth channel group,
`oc025` — satellite chlorophyll-a (ESA OC-CCI v6.0) on the same 0.25° grid
and five-day bins, from September 1997 — stored with an OFFSET time axis
(1,997 rows, not 3,142). The earlier handover's §10 promised the three
inherited groups would be the same bytes as `f7l0`'s. **That is not what was
published**, and the difference is an improvement: (1) `f7l0`'s `sst` and
`sea_ice` channels were **all-NaN for the whole of calendar 1989** (73
pentads, bins 511–583) and family 7.1 fills that year, so `g025` is strictly
better but its z-score constants moved and the file must be replaced, not
reused; (2) the 1° atmosphere group `g100` was rebuilt in `f7l2` with its two
logarithmic channels evaluated in float64, so it now REPRODUCES — at the cost
of differing from both `f7l0` and `f7l1` at the last float16 bit in about ten
cells of each 65,160-cell bin (§2.2). Everything else is either
byte-identical or differs at the last float16 bit.

---

## 1 · Files, side by side

The folder is public, needs no login, and answers HTTP `Range:` with 206.

```
https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/
```

| file (`family7_global025_pentad_l2…`) | bytes | sha256 | relation to your `f7l0` copy | action |
|---|---|---|---|---|
| `.npz` | 5,412,430 | `0837c450eba9e2cda4fb29ea81530987d5a8b5814d7722813fd100bc5595233f` | 41 keys (was 32); `norm_g025` changed; `elev` complete; new `static_n_finite` (§3) | **fetch** |
| `_X_g025.npy` | 45,670,101,248 | `1cb1e1af4132e6acdec76a7341a49453b32f9b982eee09d3ad577c5dd147d293` | **differs** — 1989 filled, sst/ice re-z-scored (§2.1) | **replace** the f7l0 file |
| `_X_g100.npy` | 6,141,981,728 | `79c8075eedf68529736fff6c1f4c3cff0fef091322971919c4e400b9ed825b06` | differs in ~10 cells per bin, 1 float16 ULP, two channels (§2.2) | replace if you key caches by hash; reusing f7l0's is numerically harmless |
| `_X_rg100.npy` | 1,050,900,608 | `55c49f1bbc55f9e82204d5d4ff2751cab5625c62003debe56df8cb5f1acaf2d5` | **byte-identical** to f7l0's | reuse; rename or symlink |
| `_X_oc025.npy` | 8,293,461,248 | `c50c79b5760103cdda1233fe6d6bf776fa03caf732a59b24cd27bd6a5cdd1700` | **new** — `[1997, 721, 1440, 2]` float16 | **fetch** |

`manifest.json` in the folder carries these values plus, per file,
`inherited`, `same_as_base` and — where that is false — `base_sha256`, the
hash of the `f7l1` file it was compared against; `inherited_groups`
(`g025`, `rg100`, `oc025`) and `rebuilt_groups` (`g100`) at the top;
`seeded_from_base: true`; `static_n_finite` `{"elev": 1038240, "sphere":
1038240}`; builder commit `f25f1a6`; `built_at` 2026-09-14T19:58:44Z. The
npy headers are 128 bytes in all four files (measured), little-endian
float16, C order, bin-major — the §7(c) range-read arithmetic of the
family-7 handover is unchanged.

Total to download if you keep rg100: 5.4 MB + 45.67 GB + 6.14 GB + 8.29 GB.

### 1.1 · If you already ingested `f7l1`

Three of the four big files are **hard-linked from `f7l1` and hash
identical** — `g025`, `rg100`, `oc025` are the same bytes under a new name,
which the manifest states as `same_as_base: true`. So the delta is:

| | |
|---|---|
| re-fetch | the 5.4 MB `.npz` (its `elev` is now real, and `static_n_finite` is new) |
| re-fetch only if you key caches by file hash | `_X_g100.npy` — 6.14 GB for a change of one float16 step in ~10 of 65,160 cells per bin |
| do nothing | `_X_g025.npy`, `_X_rg100.npy`, `_X_oc025.npy` — rename or symlink your `f7l1` copies |
| delete | any `elev` you patched in from `f7l0` — it is identical to `f7l2`'s, so this is bookkeeping, not a correction |

Only five keys of the small file changed between `f7l1` and `f7l2`:
`recipe`, `builder_git_sha`, `built_at`, `sources` and `elev` — plus the new
`static_n_finite`. `norm_g025`, `norm_g100`, `norm_rg100`, `norm_oc025`,
`sphere`, every `chan_*`, every count and every bin index are bit-identical,
so nothing derived from them needs recomputing.

---

## 2 · The three inherited groups — what actually changed

Measured 2026-09-14 by range-reading one pentad per calendar year (43 bins)
for `g025` and five bins (200, 1500, 2411, 3000, 3141) for `g100`, from each
published build, and comparing raw `uint16` views as well as values.
`f7l2`'s `g025`, `rg100` and `oc025` are hard links to `f7l1`'s, so for those
three a comparison against `f7l1` is a file compared with itself; what follows
compares `f7l2` with the `f7l0` you hold.

### 2.1 `g025` (0.25°, ocean): 1989 was missing from family 7

- The five GLORYS-derived channels — `cur_speed`, `log_mld`, `ssh`, `cur_u`,
  `cur_v` — are **bit-identical** in every sampled bin.
- `sst` and `sea_ice`: in `f7l0`, **bins 511–583 inclusive are all-NaN** for
  both channels — 73 pentads × 5 days = 365 days, 1988-12-30 → 1989-12-29
  (a truncated source transfer on the day f7l0 was built; the npz's
  `n_sst_days` reads 15,341 against 15,706 = every day 1982–2024). `f7l2`
  has the year: bin 547 (July 1989) reads 703,902 finite `sst` cells and
  153,002 finite `sea_ice` cells, exactly the counts of its neighbours.
  Bin 511 straddles into 1988 (two December days were also void) and bin
  584's f7l0 mean was taken over 1990 days only.
- Because the record is now complete, the z-score constants were recomputed:

  | channel | `f7l0` (mean, sd) | `f7l2` (mean, sd) |
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
should be recomputed on `f7l2`. Do not mix the files: `f7l2`'s `g025` must be
decoded with `f7l2`'s `norm_g025`. If your loader keys a cache on the file
hash, the new hash forces the refresh you want.

### 2.2 `g100` (1°, atmosphere and land): rebuilt so that it REPRODUCES

This is the one group `f7l2` recomputed rather than inherited, and the reason
is a property rather than a number. Two of the fifteen channels, `log_prate`
and `log_swe`, are the only ones whose transform contains a transcendental,
and the builder was evaluating `log1p` on a float32 array. Float32 `log1p` is
not correctly rounded — the answer depends on the numpy build and on which
vector loop the processor dispatches to — so the same input files gave
slightly different output on different machines. `f7l2` evaluates both in
float64 and casts once, which is IEEE-reproducible: rebuild it anywhere and
the bytes are the same.

What that costs you, measured on the five sampled bins:

| | vs `f7l0` | vs `f7l1` |
|---|---|---|
| the other 13 channels | byte-identical in every sampled bin | byte-identical in every sampled bin |
| `log_prate` cells differing per 65,160-cell bin | 0, 9, 6, 11, 10 | 0, 9, 9, 12, 9 |
| `log_swe` cells differing per 65,160-cell bin | 0, 0, 1, 1, 0 | 0, 1, 2, 0, 0 |
| size of every difference | exactly one float16 step | exactly one float16 step |

(the five bins are 3141, 200, 2411, 3000, 1500; bin 3141 is the record's
one-day last bin and is identical in every channel). One float16 step is
≤ 4.9e-4 in the stored z, about 4e-5 mm/day of precipitation and 1.7e-3 of a
snow-water-equivalent unit. No NaN-payload and no signed-zero differences;
the NaN mask is bit-identical.

Note the direction, because it is the opposite of what "a correction" usually
implies: `f7l0` and `f7l1` differ from EACH OTHER in only 1–3 `log_prate`
cells per bin — they were two float32 evaluations that mostly agreed — while
`f7l2` differs from both in about ten. That is expected. The float64 result
is the correctly rounded one and the float32 pair were both drifting around
it; `f7l2` is not "closer to `f7l1`", it is the value the arithmetic is
supposed to produce, and it is the one that will come back identical the next
time anybody builds it.

`norm_g100` is bit-identical across all three builds — the shift is ten cells
in 204 million and does not move a float32 mean or standard deviation.
Scientifically the whole difference is nil; the hash differs, which is why the
manifest says `same_as_base: false` for this one file. **Any of the three
files is correct to use**, and `f7l2`'s is the one that reproduces.

### 2.3 `rg100` (1°, monthly ocean interior): identical

Same sha256 as `f7l0`. Same 252 live rows, same `rg_bin_index`. Nothing to do.

---

## 3 · The small file (`.npz`): nine new keys, seven changed values

All 32 of `f7l0`'s keys are present, and seven of them changed value:
`builder_git_sha`, `built_at`, `groups`, `recipe`, `sources`, `n_sst_days`
(15,341 → 15,706) and `norm_g025` (§2.1). The other 25 — `elev` among them —
are bit-identical to `f7l0`'s, checked key by key in the sandbox.

**The nine new keys.** Eight of them are the colour group; the last two
rows of the table are existing keys whose value changed, listed here because
they are what a loader branches on:

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
| `recipe` | `<U4` | `"f7l2"` |

**One more new key, and it is not about colour:** `static_n_finite`, a JSON
string, reads `{"elev": 1038240, "sphere": 1038240}` — how many cells of each
static carry a number. It exists because the previous build published an
elevation static with none and nothing anywhere said so; it is also in
`manifest.json`, so the question is answerable from the Hub without
downloading anything.

**Unchanged and verified equal:** `lats`, `lons`, `lat1`, `lon1`,
`bin_index` (3142), `rg_bin_index` (252), `months`, `rg_months`, `epoch`,
`cadence`, `pentad_days`, `window`, `sphere` (int8 mask, 4 classes),
`truth_rapid` `[1459, 2]`, `truth_fc` `[2490, 2]`, `rapid`, `norm_g100`,
`norm_rg100`, `n_glorys_bins`, `n_ncep_days`, `n_rg_live`.

**`elev` is complete, and identical to `f7l0`'s.** `f7l2.npz["elev"]` is
`[721, 1440]` float32 with **1,038,240 finite values** — every cell,
−9,687.95 m to 6,004.33 m, metres relative to sea level with the sea floor
negative — and it is **bit-identical to `f7l0`'s array**, verified cell by
cell in the sandbox on 2026-09-14. So if you patched `f7l0`'s elevation in
while `f7l1` was the current build, you patched in exactly the right numbers;
take `f7l2`'s and delete the special case.

(Both are box means of ETOPO 2022 over the 0.25° cell, which is why the
highest cell on Earth reads 6,004 m rather than Everest's 8,849 m: a
quarter-degree box in the Himalaya is about 28 × 25 km and averages the
summit with its valleys. Nothing about that changed in `f7l2`; it is the
convention the earlier handover already describes.)

`sphere` is unchanged and complete as well — 702,642 ocean · 226,495 land ·
107,074 ice sheet · 2,029 inland water, the same histogram in all three
builds.

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

import json

m = np.load("family7_global025_pentad_l2.npz", allow_pickle=True)

groups = list(m["groups"])                       # ["g025","g100","rg100","oc025"]
chans  = {g: list(m["chan_" + g]) for g in groups}
norm   = {g: m["norm_" + g]       for g in groups}   # USE f7l2's norm_g025, not f7l0's

# statics: both come from this file, and both are complete. The build states
# how complete, so check it rather than assuming — this is the key that did
# not exist when an all-NaN elevation was published.
sphere = m["sphere"]
elev   = m["elev"]
n_fin  = json.loads(str(m["static_n_finite"]))   # {"elev": 1038240, "sphere": 1038240}
assert n_fin["elev"] == elev.size == int(np.isfinite(elev).sum())

# bin -> row, per group. family 7.1 adds exactly one case.
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
years**, recompute those statistics on `f7l2`'s `g025` — the 1989 hole in
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
2. `np.load(npz)`: 41 keys; `groups` has 4 entries; `recipe == "f7l2"`;
   `oc_bin_first == 1145`; `len(oc025_bin_index) == 1997` and
   `oc025_bin_index[-1] == 3141`; `n_sst_days == 15706`;
   `static_n_finite == {"elev": 1038240, "sphere": 1038240}`.
3. `g025`, bin 547 (1989-07): `sst` finite count **703,902**, `sea_ice`
   **153,002** (in `f7l0` both are 0 — that is the difference you are
   ingesting).
4. `oc025`, row 1266 (bin 2411, 2015-01-03): **421,137** finite `log_chl`
   cells; global mean `log_chl` after un-z-scoring **−0.8295**
   (0.148 mg m⁻³); mean `chl_cov` **0.249**; `isnan(chl_cov) == isnan(log_chl)`
   everywhere.
5. `elev`: 1,038,240 finite values, min −9,687.95 m (the Mariana Trench),
   max 6,004.33 m; `sphere` codes 0/1/2/3 in counts 702,642 / 226,495 /
   107,074 / 2,029. Both statics come from this npz — no patching.
6. `g100`, any full bin: thirteen channels byte-identical to your `f7l0`
   copy, `log_prate` differing in of order ten of 65,160 cells and `log_swe`
   in none or one, every difference exactly one float16 step (§2.2).
7. `rg100` hash `55c49f1b…` — unchanged, so any rg-derived cache is valid.

---

## 7 · Provenance and caveats, briefly

- Built 2026-09-14 on rented boxes, in three published attempts. Run #10
  (builder `3a13f8a`) built all four groups unseeded; run #11 rebuilt
  `rg100` in place after #10 published it as a 128-byte empty array — caught
  by reading byte counts, not the run's colour — and the two together are
  `f7l1`. Run #12 (builder `f25f1a6`, `built_at` 19:58:44Z) is `f7l2`: it was
  SEEDED from the finished `f7l1` work directory, hard-linked `g025`,
  `rg100` and `oc025` unchanged, re-read the elevation and land-cover
  statics, and rebuilt `g100` alone. The publish step restore-verified five
  files in each build; the sandbox independently re-verified the npz hash,
  the elevation array against `f7l0` cell by cell, and the `g100` range
  reads quoted in §2.2.
- The earlier handover's §10 "same bytes, hard-linked" description is the
  DESIGN, and `f7l2` is the build that finally satisfies it — against
  `f7l1` rather than against `f7l0`. The `f7l0` differences are fully
  characterised in §2 and are either an improvement (1989) or float noise.
- The `f7l1` defect this document used to work around — an elevation static
  with 0 of 1,038,240 finite cells — is CLOSED in `f7l2` (§3). What made it
  publishable was a build that filled the static with NaN when the source
  could not be fetched and recorded nothing about it; the builder now
  refuses, and `static_n_finite` in both the npz and the manifest states the
  count either way.
- Known and deliberately NOT fixed: `oc025`'s `log_chl` is computed with a
  float32 `log10` and has the same non-reproducibility `g100`'s two log
  channels had. Its bytes are inherited unchanged by `f7l2`, so the colour
  group is not expected to reproduce bit-for-bit across machines. The
  difference is far below the float16 storage; it is recorded in
  `ml/plans/E077_family7_ocean_colour.md` §10.3 and will be closed the next
  time the colour group is rebuilt for another reason.
- Holdout convention (earlier handover §9) is unchanged; colour has 21 full
  training years to fit a climatology from.
- Specification with every decision: `ml/plans/E077_family7_ocean_colour.md`
  ([rendered](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E077_family7_ocean_colour.md)).
  The family-7 base document this one extends:
  `docs/FAMILY7_DATA_HANDOVER.md`
  ([rendered](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY7_DATA_HANDOVER.md)).
