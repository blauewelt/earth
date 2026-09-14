# Family 10 — four observation stores and the registry: a self-contained data handover

**For an agent that has not seen this repository.** Everything needed to
download, open, validate and search family 10's tier-P observation stores, and
to read the registry that ties them to the gridded tensors, is on this page;
nothing below requires reading another document. Where a section says "see
also", it is optional background. Written 2026-09-13, the day the builder
landed; updated 2026-09-14, the day all four stores were published.

**Read §9 before you trust a number.** All four stores — `gdp` (surface
drifters), `gtmba` (the tropical moored arrays), `socat` (ship CO₂) and
`slatrack` (along-track sea level from 29 altimeter missions) — were built,
published and then **independently verified on 2026-09-14** by sessions that
did not build them: **2,122,112,905 observations**, every file re-hashed against
its own record, and every count, range and per-channel statistic recomputed
from the arrays rather than read out of the metadata that claims them. Those
sentences are measurements now, including `slatrack`'s, which was checked by
streaming all 67 GB rather than downloading it. §9 says what each check
returned and what it could not do; the full evidence is
`docs/FAMILY10_VERIFICATION_2026-09-14.md`.

Family 8's Argo store has its own self-contained handover and is **part of this
family unchanged** — §7 says how (optional background:
`docs/FAMILY8_DATA_HANDOVER.md`). The gridded tensor it sits beside has one too
(`docs/FAMILY7_DATA_HANDOVER.md`).

---

## 1 · What this is, in one paragraph

Family 10 is the programme's storage contract for observations of **mixed
granularity**. Nothing is resampled onto a common grid when it is stored: every
source keeps its own resolution and cadence, and the *token* a model reads
carries the measurement's **footprint** — how much area and how much time it
averaged — beside its offset from the place and time being asked about. That
matters because a 1.9°-wide reanalysis average 30 km away and a point
measurement 30 km away say different things, and a grid cell hides the
difference. The family has three **tiers**: **G**, gridded dense (the family-7
channel groups, one bin-major array per group at its native grid, read by a
byte range); **P**, points, profiles and tracks (columns sorted by five-day bin,
read by a k-nearest search); and **T**, tiles, which is designed and not built.
This page is about the four new tier-P stores and the registry.

Terms used below, once. **Pentad / bin** — a five-day period; this project
counts them from 1982-01-01, so `bin = floor((date − 1982-01-01) / 5 days)`, and
a bin is **negative** before that date. **CSR** — "compressed sparse row": an
offsets array saying where each bin's rows start and end in a sorted table.
**Footprint** — the pair `(log2_fp, log2_dt)`: `log2(footprint_km / 27.83)`, the
spatial support in units of a 0.25° cell, and `log2(support_days / 5)`, the
temporal support in pentads. **Drogue** — the underwater sail a surface drifter
tows at 15 m; with it the buoy follows the 15 m current, without it the
surface. **fCO₂** — the fugacity of carbon dioxide in the surface water,
microatmospheres; what the ocean carbon sink is actually measured as.
**Expocode** — the unique identifier of a research cruise. **SLA** — sea-level
anomaly, the departure of the sea surface from its mean, metres. **MDT** — mean
dynamic topography, so `adt = sla + mdt` is one addition. **ERDDAP** — a NOAA
data server that answers a tabular query over HTTP.

## 2 · The four stores, and where they are

Hugging Face dataset repository **`chfrank/earth-tensors`**, one directory per
store under `tensors/family10/`. Public, no token needed, plain HTTPS:

```
https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family10/<store>/<file>
```

(`resolve/main/...` answers a 302 redirect to a CDN URL — follow redirects.)

| store | what it measures | C | record | footprint `(log2_fp, log2_dt)` |
|---|---|---|---|---|
| `gdp` | surface drifters: 15 m velocity, sea-surface temperature, drogue state, every 6 h | 4 | 1979-02 → | (−4, −4.3) — a point, a quarter-day sample |
| `gtmba` | the tropical moored arrays: one row per mooring per day, 18 quantities | 18 | 1977-11 → | (−4, −2.3) — a point, a daily mean |
| `socat` | ship and mooring surface-water CO₂ fugacity with SST, salinity, pressure | 4 | 1957 → | (−4, −4) — an underway measurement, minutes |
| `slatrack` | along-track sea-level anomaly, 29 altimeter missions at 1 Hz (≈ 7 km) | 3 | 1993 → | (−2.0, −4) — a 7 km along-track cell |

### 2.1 · The files — nine arrays plus `store.json`, the same in every store

| file | dtype · shape | what |
|---|---|---|
| `bin.npy` | int16 [N] | five-day bin, `floor((date − 1982-01-01) / 5 days)`. **May be NEGATIVE**: drifters begin in 1979 and SOCAT in 1957, and those rows are kept, so the store is the whole record and the *loader* clips to whatever axis a tensor has |
| `time_days.npy` | float32 [N] | days since 1982-01-01 00:00 UTC, fractional, negative before it |
| `lat.npy` | float32 [N] | degrees north |
| `lon.npy` | float32 [N] | degrees east, in **[−180, 180)** |
| `values.npy` | float16 [N, C] | the channels of §3, in **raw units**, **NaN = not measured** |
| `platform.npy` | int64 [N] | the platform: a drifter's AOML id, a mooring's WMO code, a cruise's expocode hashed, an altimeter mission hashed. §4.3 gives the rule |
| `qc.npy` | uint8 [N] | 0 not assessed, 1 good, 2 probably good, 3+ the source's worse grades. The builder keeps ≤ 2 by default and records every drop |
| `fp.npy` | float16 [N, 2] | `(log2_fp, log2_dt)` per row. Constant down the column for all four sources — but **stored per row**, so a store that later mixes supports reads identically |
| `bin_offsets.npy` | int64 [B + 1] | CSR offsets over **this store's own** bin range: the rows of bin `b` are `[off[b − bin_first], off[b − bin_first + 1])`, and `bin_first` is in `store.json` |
| `store.json` | JSON | the schema, the footprint constants, the QC policy, the provenance (URLs, verification dates, builder commit), the per-year counts, the per-channel measured fraction and value range, and the **sha256 of every file above** |

**Rows are sorted by `(bin, time_days)` ascending.** That order is what makes
`bin_offsets` valid at all, and every consumer may rely on it.

**`bin_first` is not zero.** Family 8's Argo index runs from bin 0 over 3,143
entries because Argo starts in 2004, inside the epoch. A family-10 store's index
runs over its own range, which for `socat` begins around bin −1825 (1957). A
consumer that assumes 0 reads the wrong pentad and nothing says so — use
`bin_first`, or use the reader in §5, which does.

### 2.2 · The registry

`tensors/family10/family10.json`, written by `ml/build_family10_registry.py`,
lists **every group of the family in one file** — the tier-G channel groups of
the gridded tensor *by reference to its manifest*, family 8's Argo store, and
these four. Per group: `tier`, `layout`, `cadence`, `channels` with units, the
footprint constants, `bin_first`, the files with their sha256, the sources and
the builder commit. **A consumer dispatches on `tier` and needs nothing else.**

The registry as it stands — regenerated **2026-09-14T22:04:04Z** by the
`slatrack` build's last step — carries **nine groups and an empty
`groups_missing`**: four tier-G groups (`g025`, `g100`, `oc025`, `rg100`) at
recipe `f7l2`, and five tier-P stores (`argo`, `gdp`, `gtmba`, `socat`,
`slatrack`). That is the family complete for the first time.

Two properties are load-bearing, and both were exercised before they were
retired by the world catching up. A group the builder could not read is **not**
in `groups` and **is** in `groups_missing` — a registry that listed a group it
could not describe would be worse than a short one; `slatrack` was the group in
`groups_missing` for most of 2026-09-14 and is in `groups` now. And `tier_g`
says *which* family-7 manifest it describes: E-079 §2 names family 7.1 (the
global gridded tensor with ocean colour added), whose build had not published
when the morning's registry was written, so that copy fell back to `f7l0` (the
same tensor without ocean colour, three groups) and **stated the fallback in
`tier_g.fallback_note`** rather than silently describing a different tensor.
The corrected build published at 19:58Z as recipe `f7l2`, and the registry now
names it with `fallback_note: null`.

## 3 · The channels

Raw units throughout. Nothing is z-scored and nothing is anomalised — the store
is an archive of measurements, and a measurement that has already been divided
by a training-set sigma is no longer one. Standardise in the loader, from
training years only (§8.2).

### 3.1 · `gdp` — surface drifters (C = 4)

| idx | name | unit | what |
|---|---|---|---|
| 0 | `u` | m s⁻¹ | eastward velocity at drogue depth (15 m), the GDP's kriged 6-hourly value |
| 1 | `v` | m s⁻¹ | northward velocity, same |
| 2 | `sst` | °C | sea-surface bulk temperature from the buoy's own thermistor |
| 3 | `drogue` | 1 = attached | 1 where the observation precedes the drifter's `drogue_lost_date`, 0 after it, **NaN** where the archive records that date as 1970-01-01, which it documents as "drogue status uncertain from the beginning" |

`drogue` is a **channel, not a filter**, deliberately: a drogued drifter measures
the 15 m current and an undrogued one measures the surface, and those are
different quantities the model must be able to tell apart. An *unknown* drogue
is a third state and is NaN, not 0.

### 3.2 · `gtmba` — the tropical moored arrays (C = 18)

`sst`, `sss`, `airt`, `wind_u`, `wind_v`, `u_cur`, `v_cur`, then subsurface
temperature at eleven standard depths: `t_1m`, `t_20m`, `t_40m`, `t_60m`,
`t_80m`, `t_100m`, `t_120m`, `t_140m`, `t_180m`, `t_300m`, `t_500m`. Units:
°C, PSU, °C, m s⁻¹, m s⁻¹, m s⁻¹, m s⁻¹, then °C throughout.

One row is **one mooring on one day**, joined across the six PMEL datasets that
publish those quantities. A site that does not carry a depth leaves that channel
NaN; nothing is interpolated in. `t_1m` and `sst` come from two different
datasets and normally agree — that is a free cross-check, not redundancy.

**The ADCP velocities are centimetres per second in the archive and metres per
second here.** The factor is not hardcoded: the service puts the unit in the
column header and the parser reads it and converts, so a unit change upstream
raises rather than being silently absorbed. A silent factor of 100 on a current
still looks like a plausible ocean.

The consumer computes the **warm-water volume** — the El Niño read-out's
six-month lead — from the 20 °C isotherm depth at the equatorial sites. It is
not stored, because it is a derived quantity of a choice of sites.

### 3.3 · `socat` — surface CO₂ (C = 4)

| idx | name | unit | what |
|---|---|---|---|
| 0 | `fco2` | µatm | `fCO2rec`, the recomputed fugacity — SOCAT's own harmonised value, not one of the six raw variants beside it |
| 1 | `sst` | °C | the ship's measured sea-surface temperature |
| 2 | `sss` | PSU | measured practical salinity |
| 3 | `patm` | hPa | `PPPP`, the **measured** atmospheric pressure |

**`patm` is never `NCEP_SLP`.** The file carries both; the second is a
reanalysis value interpolated to the ship — a model output wearing a
measurement's column. A store of observations that substituted it wherever the
barometer was missing would be making up data. `patm` is NaN there instead.

### 3.4 · `slatrack` — along-track sea level (C = 3)

`sla` (m), `sla_unfiltered` (m), `mdt` (m). The plan's short name `sla` is the
archive's `sla_filtered`; both names are in `store.json` so nothing has to be
guessed. `adt = sla + mdt` is one addition rather than a second product.

## 4 · How the values were made (so you can trust or reject it)

Each store's full policy text is in its own `store.json` under `qc_policy`, in
one paragraph, so a consumer never has to find this page. The summary:

### 4.1 · What is dropped, and what is merely NaN

A **row** is dropped when it has no usable position, no usable time, or nothing
measured at all. A **channel** is set to NaN when its own value is missing, its
own flag is bad, or it falls outside the physical bounds of §4.4 — the rest of
the row survives. Every drop is counted by reason into `store.json:counts`,
because a guard that never fires and a guard that silently removes a third of
the archive look identical from outside.

### 4.2 · Per store

- **`gdp`.** The source *is* the quality-controlled product — the GDP's kriging
  onto 00/06/12/18 UTC — and it publishes no per-row flag, only per-sample
  error estimates. So the flag is **derived**, here rather than in a consumer:
  position error = `hypot(err_lat, err_lon · cos lat) × 111.32` km; over **50 km**
  the row is dropped and counted, ≤ 5 km is `qc = 1`, the rest `qc = 2`.
- **`gtmba`.** PMEL publishes a quality code beside every daily value: 0
  missing, 1 highest, 2 default, 3 adjusted, 4 lower, 5 sensor failed. A
  **sample** is kept only at 1 or 2; a **row** survives if any channel does, and
  its `qc` is the **worst** code among the samples that survived. Wind
  components share the wind-speed code, which is the only one the archive
  publishes for them.
- **`socat`.** Two flags, both applied. The dataset-level `QC_Flag` keeps A–D
  and drops E. The per-value WOCE `fCO2rec_flag` maps 2 → 1, 3 → 3, 4 → 4,
  9 → 0, and rows worse than 2 are dropped. The published synthesis file states
  it contains only flag-2 values, so this is a guard rather than a filter — and
  the counts say how many rows each clause actually removed.
- **`slatrack`.** DUACS level 3 is edited upstream and publishes no per-sample
  flag, so every kept row carries `qc = 1` and the store says so rather than
  inventing a grade.

### 4.3 · `platform`

A numeric source id where the archive has one (a drifter's AOML id, a mooring's
WMO code); otherwise `int(sha1(id)[:15], 16)` — deterministic across machines
and Python versions (`hash()` is not), 60 bits, never negative, never colliding
with a real WMO number. The rule is in every `store.json`. A consumer can hold
out a platform the way it holds out a year.

### 4.4 · Physical bounds

`|u|, |v| < 5 m s⁻¹` · `−3 < SST < 45 °C` · `0 < fCO₂ < 2000 µatm` ·
`|sla| < 3 m` · `0 ≤ sss ≤ 45 PSU` · `|wind| < 120 m s⁻¹` · `−60 < airt < 60 °C`.
A value outside is set to **NaN and counted** — never clipped. Clipping would
put a fabricated number where a broken instrument was.

### 4.5 · Every store is asserted before it is trusted

`ml/build_family10_stores.py::check_store` runs on every assembled store and
again before any publish. It asserts: rows sorted by `(bin, time_days)`; the
`bin` column equal to `floor(time_days / 5)`; the CSR offsets spanning exactly
the rows and each bin's slice holding only that bin; `lon` in [−180, 180) and
`|lat| ≤ 90`; no infinity anywhere in `values`; the footprint columns constant;
every channel inside its bounds; and a nearest-neighbour search returning
nothing from the future. Each of those is a property a broken build can have
while looking completely ordinary from outside, which is the only reason they
are worth the seconds.

One of them is subtle and worth stating. `time_days` is float32 and runs to
~16,000 days, where it resolves about 0.001 d — so a timestamp a microsecond
before a pentad boundary can round **up** across it. The builder therefore
computes `bin` from the float32 value that is actually stored, not from the
float64 the parser had. Otherwise a row would sit in bin *b* carrying a
timestamp that reads as bin *b+1*, and the search — which requires
`dt_days = 5(b+1) − t ≥ 0` — would silently never return it for its own anchor.

## 5 · Opening it (numpy only, no dependency on this repository)

```python
import json, numpy as np

d = "tensors/family10/gdp"                 # a local copy of the directory
meta = json.load(open(f"{d}/store.json"))
bin_        = np.load(f"{d}/bin.npy",         mmap_mode="r")   # int16  [N]
time_days   = np.load(f"{d}/time_days.npy",   mmap_mode="r")   # f32    [N]
lat         = np.load(f"{d}/lat.npy",         mmap_mode="r")
lon         = np.load(f"{d}/lon.npy",         mmap_mode="r")
values      = np.load(f"{d}/values.npy",      mmap_mode="r")   # f16    [N, C]
platform    = np.load(f"{d}/platform.npy",    mmap_mode="r")   # i64    [N]
qc          = np.load(f"{d}/qc.npy",          mmap_mode="r")   # u8     [N]
fp          = np.load(f"{d}/fp.npy",          mmap_mode="r")   # f16    [N, 2]
off         = np.load(f"{d}/bin_offsets.npy")                  # i64    [B+1]
bin_first   = meta["bin_first"]
channels    = [c["name"] for c in meta["channels"]]

# every observation in the pentad that contains 2015-01-03
b  = (np.datetime64("2015-01-03") - np.datetime64("1982-01-01")).astype(int) // 5
lo, hi = off[b - bin_first], off[b - bin_first + 1]
print(hi - lo, "observations in bin", b)
```

**Verify before you use it.** `ml/family10_store.verify_store(dir)` re-hashes
every file against `store.json`'s own record and raises on any mismatch. A
truncated `values.npy` still memmaps and still answers a search, with whatever
its tail happens to hold.

### 5.1 · The reader, and the search

`ml/family10_store.py` is the only thing in this repository that opens a store,
so the layout is defined once.

```python
from family10_store import Store

st  = Store.open("tensors/family10/gdp")                       # a directory
st  = Store.open("chfrank/earth-tensors:tensors/family10/gdp") # or the Hub
tok = st.knearest(36.0, -70.0, bin=2411, k=8,
                  R_max_km=300.0, T_max_days=10.0)
```

`Store.open` takes a directory or a Hub prefix; the Hub form downloads
`store.json` first and then exactly the files it names, never a whole-repo
snapshot. `knearest` returns a dict of **fixed length k** — always k slots, so a
batch is never ragged:

| key | shape | what |
|---|---|---|
| `values` | (k, C) float32 | raw units, NaN where a channel or a slot is missing |
| `mask` | (k, C) bool | True where a real number was measured |
| `dx_km`, `dy_km` | (k,) | signed kilometres east and north **from the anchor**, so the model never has to learn the cosine of latitude |
| `dt_days` | (k,) | age in days, always ≥ 0 |
| `dist_km`, `d2` | (k,) | great-circle distance, and the metric the ranking used |
| `log2_fp`, `log2_dt` | (k,) | the footprint of each returned observation |
| `platform`, `qc` | (k,) | provenance and the source's grade |
| `n_R` | scalar | how many observations lay inside `(R_max_km, T_max_days)` — the **local density** feature |
| `valid` | (k,) bool | False marks a **miss token** (NaN values, `row = −1`) |
| `n_found`, `row`, `lat`, `lon`, `time_days`, `channels` | | |

Three rules, all structural rather than applied afterwards:

- **One-sided in time.** Only observations at or before the anchor's pentad are
  candidates; the CSR slice stops at the anchor's own bin and never reads a
  later row, so a future observation cannot be returned however close it is. A
  forecaster that has seen the next pentad is not a forecaster.
- **Bounded** by `R_max_km` and `T_max_days`. Where fewer than k exist inside
  them the remaining slots are miss tokens — the same token the gridded tensor
  uses at an empty cell, made rare instead of usual.
- **The metric** is `d² = (dx² + dy²)/L² + dt²/T²`, with `L` and `T` defaulting
  to the search bounds, which makes the two axes commensurate at the boundary.
  Pass them explicitly to use a channel's own correlation length.

`dt` is measured from the **end** of the anchor's pentad, `5(bin + 1)` days
after the epoch, so an observation inside the anchor's own bin is 0 to 5 days
old rather than arriving from the future.

`qc_max=1` narrows to the source's best grade, **before** `n_R` is counted: a
token the reader will not return must not inflate the density feature.

## 6 · Search radii — measured

Until 2026-09-14 this section held four guesses derived from each observing
system's sampling geometry. Three of them have now been measured on the
published stores, and all three were **too loose by a factor of three to ten**
at anchors where the store has data. `slatrack` is the one still unmeasured —
not because it is unbuilt, but because the measurement wants a machine that can
memory-map 67 GB. The measured numbers:

| store | k | k-th neighbour distance (median / p90 / p99) | k-th neighbour age (median / p99) | measured (R_max, T_max) covering the 99th percentile ON-TRACK | the old suggestion |
|---|---|---|---|---|---|
| `gdp` | 8 | **19 km** / 68 km / **104 km** | **2.0 d** / 3.8 d | **≈ 110 km, ≈ 4 d** | 300 km, 10 d |
| `gtmba` | 4 | **0 km** / 334 km / **334 km** | **2.5 d** / 3.5 d | **≈ 350 km, ≈ 5 d** | 1,500 km, 15 d |
| `socat` | 8 | **4.5 km** / 46 km / **111 km** | **1.8 d** / 4.9 d | **≈ 150 km**; keep **30 d** for T | 500 km, 30 d |
| `slatrack` | 16 | *not measured — the store is built and published, but `measure_knn.py` memory-maps it and it is 67 GB* | — | *suggested only:* 200 km, 10 d, one repeat cycle. Expect far smaller: consecutive along-track samples are **6.2–6.5 km** apart (measured), so k = 16 at an anchor the satellite flew over is ~100 km of one pass | 200 km, 10 d |

Read the third column before fixing anything. `gtmba`'s 4th neighbour is at
0 km at the median because the array is a fixed lattice and the same mooring
contributes several daily rows inside the window — the temporal bound, not the
spatial one, is what k = 4 actually spends. `socat`'s T is the one bound worth
leaving generous: the 8th neighbour is under 5 days old at the 99th percentile,
but that is one cruise's own track, and a ship revisiting a piece of ocean is
rare, so a shorter window would not cost much and a longer one is the only way
a *second* cruise can ever enter the search.

**Off-track, the radius is not the binding constraint — the observing system
is.** At anchors drawn uniformly over the globe rather than from the store's own
rows, the search returns **nothing at all** inside the suggested bounds on
**71 %** of anchors for `gdp`, **81 %** for `gtmba` and **89 %** for `socat`.
Widening R does not rescue them; it only makes the search slower. The miss
token is the normal answer over most of the ocean, and `n_R` (§5.1) is the
feature that says so honestly — this is §8.3, measured.

**The k tokens come from ONE platform.** Over 600 on-track anchors per store,
the median number of *distinct* platforms among the k returned observations is
**1 of 8** for `gdp`, **1 of 8** for `socat`, and **2 of 4** for `gtmba`. The
eight `gdp` slots are one drifter at 6-hourly steps; the eight `socat` slots are
one research cruise, seconds apart along one line. Generous bounds do not change
it. The consequence for a consumer: **k buys one track's redundancy, not k
independent measurements of the ocean.** If independent information is what the
tokens are meant to carry, the search has to be de-duplicated by `platform`, or
k made much larger than 8. That is an open design question, not a decision this
page makes.

**Method, so the numbers can be reproduced or disputed.** Two draws, following
`ml/build_family8_argo.py::index_stats`: a **catalogue draw** whose anchors are
the store's own rows with `bin ≥ 0`, stratified equally across the years present
so no dense recent year dominates, and a **globe draw** of lat ~ U(−60, 60),
lon ~ U(−180, 180) at a random live bin, land included, where the honest answer
is that nothing is measured. 2,000 anchors per draw per store, seed 20260914.
The search itself ran with **generous bounds** (R_max = 5,000 km,
T_max = 60 d) so the k-th neighbour is nearly always found, while the ranking
**metric scales `L_km` and `T_days` stayed pinned to the old suggested (R, T)**
— so the ordering is the one a consumer would actually get, and `dist_km[k−1]`
is exactly "the R_max that would have retained the k-th token". `n_R` was
measured separately at the suggested bounds. The scripts are
`ml/tools/family10_verify/measure_knn.py` and
`ml/tools/family10_verify/platform_diversity.py`; the full result is
`docs/FAMILY10_VERIFICATION_2026-09-14.md` §5 and §8.6.

## 7 · Family 8's Argo store is part of this family, unchanged

`ml/family10_store.Store` opens the family-8 store **with no rebuild and no
copy**: it reads that store's `temp` and `psal` blocks as one 32-column value
matrix (temperature at the 16 Roemmich–Gilson pressures, then salinity at the
same 16), names the channels `temp_10 … temp_1900, psal_10 … psal_1900`, and
takes the footprint constants (−4, −4) out of that store's own `store.json`. The
blocks stay separate memmaps and only the k selected rows are ever
concatenated, so opening the real 2.68 M-profile store costs nothing.

The test suite asserts this is **bit-identical** to `ArgoStore.knearest` — the
same rows, the same offsets, the same values — on every anchor it tries. The
registry lists it as a tier-P group named `argo` with the channel names the
reader produces, so the two cannot drift.

## 8 · How to use it in training

### 8.1 · The unit of data

An **anchor** (a place, and a pentad) plus, per store, its k nearest
observations as §5.1's token fields. That is E-078 §2's schema, and it is the
same schema for a gridded group — a 1° cell's value is one token located at the
cell centre with `log2_fp = +2.9`, seen from sixteen different offsets by the
sixteen 0.25° anchors inside it, rather than sixteen copies claiming sixteen
positions.

### 8.2 · Standardise in the loader, from training years only

The stores are raw on purpose. Compute the mean and standard deviation of each
channel over **training bins only** and apply them in the loader. A statistic
computed over the whole store carries the test years into the weights; that leak
is invisible and it is the one this layout is designed to let you avoid.

### 8.3 · Missingness is a feature

`mask` and `n_R` are inputs, not defects. `n_R = 0` means "nothing was measured
within reach", which is a true statement about the observing system at that
place and time and is exactly what the gridded representation could not say. Do
not impute; do not fill NaN with zero — zero is a real fCO₂ of zero and a real
velocity of zero.

### 8.4 · The holdout protocol is unchanged

2009, 2017 and 2023 are development years; train ≤ 2020, test 2021–2024. The
stores carry the **whole** record, pre-1982 included; the loader restricts. That
is why negative bins are kept rather than dropped at build time — dropping
cannot be undone.

## 9 · What is verified (2026-09-14) and what is pending

**All four stores are built, published and independently verified.** The three
keyless ones were built on 2026-09-14 from builder commit `c5bc2ce` on
GitHub-hosted runners; `slatrack` followed that night from commit `6d67855`,
fetched on six hosted lanes and assembled on a keyless box (§11). Each was
checked the same day by a session that did not build it: every file fetched
from the Hub and re-hashed against `store.json`'s own record, then every
structural property and every statistic **recomputed from the arrays** rather
than read out of the metadata claiming it.

| store | N | live bins / total | overall measured fraction | sha256 | `verify_store` | per-channel statistics |
|---|---|---|---|---|---|---|
| `gdp` | 48,480,798 | **3,353 / 3,353 (100 %)** | 0.963865 | **9 / 9 match** | returns 9, no raise | 4 of 4 reproduced exactly |
| `gtmba` | 1,001,282 | 3,386 / 3,446 (98.3 %) | 0.568374 | **9 / 9 match** | returns 9, no raise | 18 of 18 reproduced exactly |
| `socat` | 41,830,675 | 3,266 / 4,910 (66.5 %) | 0.923146 | **9 / 9 match** | returns 9, no raise | 4 of 4 reproduced exactly |
| `slatrack` | **2,030,800,150** | **2,339 / 2,339 (100 %)** | 0.999052 | **9 / 9 match** | not run — 67 GB does not fit the checking sandbox; **streamed instead** and every property it asserts recomputed row by row | 3 of 3 reproduced exactly, means to 17 digits |

"Reproduced exactly" means the measured count matched to the row and the
fraction, minimum, maximum and mean to float precision. Also recomputed and
passing in all three: N and the bin range, the live-bin count, ascending sort by
`(bin, time_days)`, the CSR offsets reproduced by `searchsorted`,
`bin == floor(time_days / 5)` on every row, longitude in [−180, 180), constant
footprint columns, quality codes within `qc_keep_max`, and per-year counts
summing to N.

**`slatrack` was checked by STREAMING, and the checks are therefore
exhaustive.** 67.02 GB against 14 GB of free disk means `Store.open` and
`verify_store()` are not available, so all nine arrays were read over HTTPS in
row-aligned lockstep — one thread per file, each hashing its own bytes, one
consumer recomputing every check on the chunk in flight — in one pass, 837 s at
80 MB/s. Because it is a pass over every row rather than a sample, "sorted by
`(bin, time_days)`" means 2,030,800,149 adjacent comparisons and "lon ∈
[−180, 180)" means 2.03 billion of them. All nine sha256 match; 0 descents in
`bin`; 0 descents in `time_days` inside a bin; `bin_offsets` reproduced exactly;
`bin == floor(time_days/5)` on every row; `fp` one pair, (−2, −4); `qc` = 1
everywhere; every `platform` one of the 29 mission-id hashes; and **the
per-year counts equal the six fetch lanes' own `done.json` row counts to the
row, 32 of 32 years** — which is E-079's stated falsifier, not met. The files:

- [the `gdp` store on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/tree/main/tensors/family10/gdp) — surface drifters, 6-hourly, 1.6 GB.
- [the `gtmba` store on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/tree/main/tensors/family10/gtmba) — the tropical moored arrays, daily, 63 MB.
- [the `socat` store on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/tree/main/tensors/family10/socat) — ship and mooring surface CO₂, 1.37 GB.
- [the `slatrack` store on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/tree/main/tensors/family10/slatrack) — along-track sea level, 29 altimeter missions, **67.02 GB**.
- [the family-10 registry on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/blob/main/tensors/family10/family10.json) — **nine groups, `groups_missing: []`**, regenerated 2026-09-14T22:04:04Z by the `slatrack` build's last step; tier G is recipe `f7l2` with four groups and no fallback note, and every tier-P entry's N, `bin_first`, file count and sha256 agrees with the `store.json` verified above.

The evidence, number by number, is in
`docs/FAMILY10_VERIFICATION_2026-09-14.md`.

**Two stores were rebuilt after that verification, and the verification still
holds for both.** `socat`'s first build wrote a wrong ledger — the counters in
`counts` were 59× too large (§10) — so it was rebuilt the same day at
**08:46Z** from builder commit `0d5860d` with the fix; `gdp`'s first build
counted 568 drogue-uncertain rows it then dropped (§10), so it was rebuilt from
the same commit, landing **09:47Z**. In both cases only the ledger changed:
**all nine array sha256 values in each new `store.json` are byte-identical to
the ones verified that morning** (`gdp`'s compared digest by digest against
the verified `store.json` fetched before the publish), so every number in the
table above is a number about the files on the Hub today. `socat`'s
`rows_read` now reads 44,018,204 with `drop_out_of_range` 2,160,005 and
`drop_no_fco2` 27,520; `gdp`'s `drogue_uncertain` reads 1,416,828 — exactly
the `drogue` channel's NaN count — beside a new `drop_no_values` 568, and
`rows_read − drop_pos_err − drop_no_values − N` = 0.

- [the `socat` rebuild](https://github.com/blauewelt/earth/actions/runs/34822844466) — re-streamed the SOCAT v2026 synthesis file with the corrected counter and republished the store, ~17 min, arrays unchanged.
- [the `gdp` rebuild](https://github.com/blauewelt/earth/actions/runs/34822846450) — re-fetched the 46 years of drifter data from AOML with the corrected counter and republished the store, 69 min, arrays unchanged; its last step rewrote the registry.

**Historical context: the access paths, verified against the live archive on
2026-09-13** — from a sandbox that can reach these hosts, by listing and
fetching rather than by reading documentation. This is how the parsers were
written; the builds above are what they produced.

| store | the access path, exactly | what was confirmed |
|---|---|---|
| `gdp` | `https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc.csvp?<vars>&time>=<ISO>&time<=<ISO>` (mirror: `https://osmc.noaa.gov/erddap/...`) | the dataset listing (7 datasets; `drifter_6hour_qc` is "Global Drifter Program - 6 Hour Interpolated QC Drifter Data"); every variable's name, type and unit; the time coverage 1979-02-02 → 2025-06-18; and **a real one-month fetch**: January 2015 = 17.2 MB of CSV, **149,658 rows**, 1,275 drifters, 124 six-hourly times, parsed end to end into a store of 149,658 rows in 7 live bins |
| `gtmba` | `https://osmc.noaa.gov/erddap/tabledap/pmelTaoDy<X>.csvp?array,station,wmo_platform_code,longitude,latitude,time,depth,<var>,<qc>&time>=<ISO>&time<=<ISO>` — which answers a **302 to `https://coastwatch.pfeg.noaa.gov/erddap`**, and that host serves it | the nine `pmelTaoDy*` datasets on the listing; **every variable name read from each dataset's own metadata**, not transcribed (`T_25`/`QT_5025`, `T_20`/`QT_5020`, `S_41`/`QS_5041`, `AT_21`/`QAT_5021`, `WU_422`/`WV_423`/`QWS_5401`, `u_1205`/`QU_5205`, `v_1206`/`QV_5206`); the `pmelTaoDyT` coverage 1977-11-03 → 2026-09-02 and its depth axis 1–2000 m; **a real one-month fetch**: January 2015 = **2,905 mooring-days over 96 sites**, every one of the eleven standard depths present |
| `socat` | release page `https://socat.info/latest` → v2026 → `https://socat.info/socat_files/v2026/SOCATv2026.tsv.zip` (NCEI OCADS accession 0315110, DOI 10.25921/8dba-fr90) | the release (v2026, 2026-06-16); the file — HEAD 200, **1,411,801,422 bytes**, `accept-ranges: bytes`, a single zip64 deflate member `SOCATv2026.tsv`; the **32-column data header** read out of the first 50 MB by a Range request, after an 8,453-line preamble; and **a real end-to-end pass**: the whole 1.4 GB streamed and parsed in **357 s**, 44,018,204 source rows, **138,373 kept for January 2015** over 55 cruises |
| `slatrack` | `https://stac.marine.copernicus.eu/metadata/SEALEVEL_GLO_PHY_L3_MY_008_062/product.stac.json` → 29 per-mission `dataset.stac.json` | **PUBLIC metadata, no credentials**: the 29 per-mission dataset ids (`cmems_obs-sl_glo_phy-ssh_my_<mission>-l3-duacs_PT1S_<ver>`, missions ERS-1/2, TOPEX/Poseidon, Jason-1/2/3, Envisat, GFO, CryoSat-2, Saral/AltiKa, HaiYang-2A/2B, Sentinel-3A/3B, Sentinel-6A, SWOT nadir), each mission's time span, and the variable names `sla_filtered`, `sla_unfiltered`, `mdt`, all in metres |

**Real one-month value ranges, measured** (January 2015, so `bins 2410..2416`):

| store | N | channel ranges |
|---|---|---|
| `gdp` | 149,658 over 1,275 drifters | `u` −1.71 … 2.04, `v` −1.45 … 1.63 m s⁻¹ (99.9 % measured) · `sst` −2.03 … 32.00 °C (96.3 %) · `drogue` 0/1 with 124 unknowns · `qc` 148,924 good / 734 probably-good · 0 rows dropped for position error, 0 values outside bounds |
| `gtmba` | 2,905 over 96 sites | `sst` 20.47 … 31.23 °C (88 %) · `sss` 32.31 … 37.34 PSU (59 %) · `airt` 19.20 … 30.19 °C · `wind_u` −10.80 … 11.70, `wind_v` −8.60 … 6.60 m s⁻¹ (79 %) · `u_cur` −0.18 … 0.35 m s⁻¹ (7.5 %, ADCP sites only) · `t_1m` 20.47 … 31.23, `t_100m` 12.18 … 29.95, `t_300m` 8.54 … 16.45, `t_500m` 6.11 … 12.39 °C · dropped: 39,999 fills, 9,698 non-standard depths, 1,091 on quality flags |
| `socat` | 138,373 over 55 cruises | `fco2` 103.56 … 741.50 µatm (100 %) · `sst` −1.89 … 31.27 °C (100 %) · `sss` 17.48 … 37.16 PSU (99.9 %) · `patm` 937.0 … 1040.0 hPa (88.5 %) · every row WOCE flag 2 |

`t_1m` and `sst` agreeing to the last digit is the free cross-check of §3.2:
they come from two different PMEL datasets joined on `(station, day)`.

**WHAT `slatrack` MEASURED, now that it exists.** Everything this page says
about its N, its live bins, its per-year counts, its size and its sha256 block
was the contract until 2026-09-14T21:42:24Z; it is a measurement now.
**2,030,800,150 rows**, C = 3, bins 803..3141 with all 2,339 live,
1993-01-01 → 2024-12-31, 67.02 GB, 28 of its 29 missions carrying rows, and
0.999052 of its value slots measured. Three findings a consumer should carry,
none of them a build error:

- **`time_days` is float32 and the sampling is 1 Hz, so the column is 84×
  coarser than the data.** Its resolution is 21 s in 1993 and **84.4 s from
  2015 on** (2⁻¹⁰ d), while the altimeter samples once a second: 28 to 316
  consecutive rows carry the identical timestamp, which at 6.5 km between
  samples is up to **550 km of one satellite's track**. `knearest`'s `dt_days`
  cannot order inside that window. `bin` is exact
  (`bin == floor(time_days/5)` on every row), so **derive calendar dates from
  `bin`** — the same rule `socat` already earned in §10, one order of magnitude
  more consequential here.
- **January to March 1994 is one satellite.** ERS-1's 35-day repeat stops
  mid-December 1993 and its geodetic phase begins 1994-04-10; DUACS publishes
  no level-3 product for the 3-day ice-phase orbit in between, so those three
  months carry TOPEX/Poseidon alone. That is the whole of 1994's 3.8 M-row
  deficit and the record's thinnest bin (152,741 rows). Nothing is missing from
  the build; the constellation was.
- **GFO thins from 2003 and has two empty months inside its own mission
  window** (2004-03 and 2006-09, against a 1.44 M median month), never
  recovering past 900 k a month after 2007-02. Treat GFO-era coverage as
  uneven.

And two provenance strings in `slatrack`'s `store.json` are stale — `verified`
still says the download path cannot be run from a sandbox, and `sources` names
the `copernicusmarine subset` call rather than the
`copernicusmarine.get`-on-original-files route that actually ran. Both are
class constants; neither affects a byte of any array.

**Historical, and left standing because it is why the build looks the way it
does: as of 2026-09-13 the two Copernicus repository secrets were believed not
to exist** in this repository. The workflow's preflight says so in plain words
and refuses before anything is fetched. They had in fact existed since
2026-08-16 and nobody had checked, which is why `slatrack` was the last store
built rather than the first. The builder reads
`COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`
from the **environment only** — never a file, never a command line.

**Smaller notes, all still current:**

- **`gdp`'s ledger is closed since the 09:47Z rebuild.** The store published
  at 08:05Z carried `counts.drogue_uncertain` 568 higher than the NaN count in
  the `drogue` channel (§10 explains why and why the arrays are fine); the
  store on the Hub now reads 1,416,828 and records the 568 as
  `drop_no_values`. A copy fetched before 09:47Z has the old ledger and the
  same arrays.
- **The `socat` resume granularity is the whole stream, not the year.** The
  synthesis file is sorted by expocode, not by time, so there is no per-year
  request to make and an interrupted fetch re-reads the file from the start.
  Each store's `store.json` states its own `resume_granularity`; the other three
  say `"year"`.
- **The registry's tier-G half is `f7l2`, four groups, no fallback note —
  since 22:04Z on 2026-09-14.** Family 7.1 is the global 0.25° gridded tensor
  with ocean colour added as a fourth channel group, `oc025`. Its manifest was
  still a 404 when the morning's registry was written, so that copy described
  `f7l0` — the same tensor without ocean colour, three groups — and **said so in
  `tier_g.fallback_note`** rather than silently describing a different tensor.
  The corrected build (recipe `f7l2`, E-077) published at 19:58Z, and the
  `slatrack` build's last step regenerated the registry from it: `stem`
  `family7_global025_pentad_l2`, `recipe` `f7l2`, `fallback_note` **null**, and
  the four tier-G file sha256 values match that manifest digest by digest. The
  fallback mechanism was exercised and then retired by the tensor arriving,
  which is the outcome it was written for.
- **`slatrack`'s suggested search radii in §6 are still a guess**, and it is
  now the ONLY store of the four for which that is true — the other three were
  measured on the published stores the same day. The measurement is not cheap
  here: `measure_knn.py` memory-maps a store, and this one is 67 GB, so it wants
  a box with the disk rather than a sandbox. Measure the k-th-neighbour distance
  before fixing them, with `ml/tools/family10_verify/measure_knn.py`. Expect the
  answer to be small: consecutive along-track samples are 6.2–6.5 km apart, so
  k = 8 at any anchor the satellite flew over is ~50 km of one pass — the same
  "the k tokens are one platform" property §6 already measured for the other
  three, in its most extreme form.
- **`time_days` cannot order `slatrack` rows inside 84 seconds** (§10, and §9
  above). Every consumer of the k-nearest search on this store should know that
  its `dt_days` ties are not ties in the world.

## 10 · Known limits and gotchas

- **`values` is float16.** About three decimal digits. That is ~0.002 °C at
  20 °C and ~0.5 µatm at 700 µatm — below every instrument's own uncertainty,
  and it halves the store. Cast to float32 before arithmetic; the reader
  already does.
- **NaN means "not measured", never zero.** Every channel can be NaN
  independently of the others on the same row.
- **A negative bin is normal.** `bin.npy` is signed and `bin_first` can be
  −1825. Code that indexes an array by `bin` without subtracting `bin_first`
  will either crash or, worse, wrap.
- **Longitude is [−180, 180) and the dateline is not a wall.** The reader wraps
  the longitude difference; a consumer computing offsets by hand must too.
- **`gtmba` coverage is very uneven by channel.** Currents exist at 7 % of
  mooring-days (only the ADCP sites) and salinity at 59 %. That is the array,
  not the build; `mask` says it per row.
- **`socat`'s platform is a hashed expocode**, so it identifies a cruise, not a
  ship. Two cruises of the same vessel are two platforms, which is the right
  unit for holding out a calibration.
- **The GDP product is interpolated**, not raw fixes: positions and velocities
  are kriged onto 00/06/12/18 UTC. The footprint says "a point, a quarter-day
  sample"; it does not say "an instantaneous measurement".
- **`slatrack` is 67.02 GB** (measured; the estimate was 50–80 GB) and is the
  only store that needs credentials AND a
  large machine — and, as of 2026-09-14, the only one whose build runs on two
  machines, because no single machine may have both. `ml/CLAUDE.md` §6 forbids
  the Copernicus credentials on a rented box, so the fetch may only happen on a
  GitHub-hosted runner; a hosted runner has ~14 GB of disk and a six-hour job
  cap, so the assembly may not happen there. The two halves are joined by the
  Hub (see §11). The other three stores are keyless and fit a hosted runner
  whole. It is also 22× the other three put together, so anything that opens
  "all the tier-P stores" should size for this one alone.
- **`time_days` is float32, so derive calendar dates from `bin`, not from it.**
  At the end of the record (t ≈ 15,700 days) float32 spacing is 0.00098 d =
  **84 seconds**. Measured on the published `socat` store, three rows late on
  2024-12-31 round up across midnight and read as 2025-01-01 — 3 rows in
  41.8 M. The five-day `bin` is computed from the stored float32 and is
  unaffected, which is exactly what §4.5's last paragraph was written to
  guarantee, so `store.json`'s per-year counts (taken from the source dates) are
  the correct ones and a year recomputed from `time_days` is the imprecise side.
- **In `slatrack` that float32 column is COARSER THAN THE SAMPLING, by 84×, and
  this is the store's single most important gotcha.** The altimeter samples once
  a second; the column resolves 21 s in 1993 and 84.4 s from 2015 on. Measured
  on the published store: 28 rows per distinct timestamp in the first bin, 184
  in bin 2411 (2015-01-03), **316 in the last** — i.e. up to 316 rows, and 21 to
  84 consecutive samples of any ONE mission, carrying the identical
  `time_days`. At 6.5 km between samples that is **550 km of track with one
  time on it**. Consequences: `knearest`'s `dt_days` cannot order inside that
  window and the tie-break falls entirely to distance; the store is nonetheless
  correctly sorted (`time_days` is non-decreasing inside every one of the 2,339
  bins, verified row by row); and `bin == floor(time_days / 5)` holds on all
  2.03 billion rows, so `bin` remains exact. Nothing is hidden — `store.json`'s
  `schema` declares float32 and the family-10 contract fixes the dtype — but
  this is the first store whose time column is coarser than its own sampling
  interval.
- **`slatrack`'s per-mission coverage is uneven, and two of the holes are
  real.** January to March 1994 carries TOPEX/Poseidon **alone** (ERS-1's 35-day
  repeat stops mid-December 1993, its geodetic phase starts 1994-04-10, and
  DUACS publishes no level-3 product for the 3-day ice-phase orbit in between),
  which is the whole of 1994's 3.8 M-row shortfall and contains the record's
  thinnest bin at 152,741 rows. GFO has two calendar months inside its own
  mission window with **no rows at all** (2004-03, 2006-09), two more far below
  its 1.44 M median, and never exceeds 900 k a month after 2007-02. Neither is
  a build gap — no year was marked in which a listing came back empty or a
  download came back short — but both are thin patches a model should not be
  told are ocean without observations.
- **`gdp`'s `counts.drogue_uncertain` runs 568 ahead of the NaNs in the
  `drogue` channel**, in the store published 2026-09-14. The cause is known and
  the arrays are not affected: those 568 source rows had nothing measured on
  them at all — no velocity, no temperature, an uncertain drogue — so they were
  counted and then dropped, and the same 568 is why
  `rows_read − drop_pos_err − N` does not close. Every array-side identity does
  close (drogue is exactly three-state, 1 + 0 = the `measured` count, and the
  claimed mean reproduces). The builder now counts only rows it keeps and
  records that drop as `drop_no_values`; **the rebuild landed 2026-09-14
  09:47Z and closed both halves** — `drogue_uncertain` 1,416,828,
  `drop_no_values` 568, arrays byte-identical (§9). A copy fetched before
  09:47Z carries the old ledger; its arrays are the same bytes.
- **`socat`'s counters were inflated 59× in the first build, and are correct in
  the store on the Hub now.** One pass over the file, its counters copied into
  every year part and then summed across the 59 years that held rows. The
  published ledger reads `rows_read` 44,018,204 → 41,830,675 kept (95.0 %)
  since the **08:46Z rebuild** from builder commit `0d5860d`, whose nine arrays
  are byte-identical to the verified build (§9). If you hold a copy fetched
  before that, its `counts` are the inflated ones — divide by 59, or re-fetch.
- **`slatrack`'s `store.json` carries two stale provenance strings.** Its
  `verified` field still ends "Neither route can be run from this sandbox … NOT
  YET MEASURED: the remote path layout of the original files", and its `sources`
  names the `copernicusmarine subset` call. Both were written before any
  `slatrack` fetch had succeeded; the build that produced the store ran the
  `copernicusmarine.get`-on-original-files route end to end over 32 years. The
  product and the dataset ids are right, the call named is not, and no array is
  affected.
- **The k observations a search returns are usually ONE platform.** Median 1 of
  8 for `gdp` and `socat`, 2 of 4 for `gtmba`, measured on the published stores
  (§6). `slatrack` is not yet measured and will be the extreme case: at 6.5 km
  between consecutive samples, any k under a few hundred is one overflight. A consumer that treats k as k independent looks at the ocean is wrong
  about its own inputs: it has one drifter's or one cruise's track, sampled k
  times. De-duplicate by `platform` if independence is what is wanted.

## 11 · Provenance

Built by `ml/build_family10_stores.py` (`--store gdp|gtmba|socat|slatrack`,
stages `index | fetch | publish`), read by `ml/family10_store.py`, registered by
`ml/build_family10_registry.py`, dispatched by
`.github/workflows/family10-build.yml` (`workflow_dispatch` only), tested by
`tests/test_build_family10_stores.py`. Every `store.json` carries the builder's
git commit, the build time, the source URLs and the verification sentence above.

**`slatrack`'s build path is different, and this is what ran on 2026-09-14.**
It is the one store built on two machines, for the reason in §9: the
credentials may only live on a GitHub-hosted runner and the 67 GB assembly may
only happen on a box. The seam is a Hub prefix.

1. `.github/workflows/family10-slatrack-fetch.yml` — hosted lanes
   (`runs-on: ubuntu-latest`, hard-coded), six of them over weighted year
   ranges, `schedule:` every six hours as the resume mechanism. Each lane
   fetches ONE YEAR (`--stage index,fetch --start Y-01-01 --end Y-12-31`),
   pushes that year's column parts to
   `partials/family10/slatrack/<year>/` on the dataset repo with
   `ml/family10_parts_hub.py push`, deletes the local copy and moves on. The
   push writes `done.json` LAST and only after every file has been downloaded
   back with a matching sha256 (`ml/CLAUDE.md` §5.21 — a marker may only
   under-claim; §0.2 — a 200 is not evidence the bytes are retrievable), so a
   year killed halfway is simply refetched.
2. `.github/workflows/family10-build.yml` on a box, with
   `stage=fetch,publish` and `extra_args=--parts-from-hub`. The builder pulls
   every requested year back, verifies each file against `done.json`, refuses
   (naming them) if any year is missing, and assembles. The box carries
   `HF_TOKEN` and nothing else — the workflow does not pass the Copernicus
   secrets to a self-hosted runner at all, and its Preflight step refuses a
   self-hosted `slatrack` build that did not ask for `--parts-from-hub`.

`python3 ml/family10_parts_hub.py status --store slatrack` lists which years are
on the Hub, with their rows and bytes.

**What it cost, measured.** The six lanes ran 10–16 minutes per year and are
GitHub-hosted, so the ~47 hours of credentialed downloading cost **$0**. The
assembly run —
[#11 (E-079 `slatrack` — assembly of 32 Hub year-parts, streaming sort, publish, registry)](https://github.com/blauewelt/earth/actions/runs/34894902246) —
took **1 h 18 m** on a keyless 250 GB box for **≈ $0.55**: 2,855 s to pull the
32 year-parts back and re-verify them, 12.9 s for the counting pass, 240.9 s to
scatter 2,030,800,150 rows, 112.1 s to sort the 2,315 bins that needed it,
~350 s for the chunked `check_store`, and 1,003 s to publish ten files and
verify each by restore, at a peak RSS of 66.06 GB. The registry followed in
four seconds. The estimate at dispatch was one box-day and ≈ $8; splitting the
credentialed half onto free parallel lanes is where the difference went.

**The assembler that runs there is `assemble_store_streaming`**, selected by
`--assemble auto` (streaming above 50 M part rows, and always for `slatrack`).
It writes the same store as the in-RAM `assemble_store` **byte for byte**, in
three passes over the parts: count rows per bin to get the CSR offsets; scatter
each part's rows into `offsets[bin] + cursor[bin]` through `open_memmap`, in the
order `read_parts` yields them; then sort each bin's slice by `time_days` with a
stable argsort. That is the same permutation `np.lexsort((time_days, bin))`
produces, because lexsort is stable and its final tie-break is input order —
which is what the scatter reproduces.
`tests/test_build_family10_stores.py::test_the_streaming_assembler_writes_the_same_bytes`
proves it by hashing both assemblers' output over one synthetic archive doctored
to hold duplicate `(bin, time_days)` rows across two parts. Peak RAM is one part
plus one bin's slice; a disk preflight computes the store's size from the dtypes
and refuses before pass 2 if free space is under 1.2× it.

Specification: `ml/plans/E079_family10_point_stores.md`. The design it
implements: `ml/plans/E078_multi_granularity.md`. The footprint token it
carries: `ml/plans/E076_family8_nearest_observations.md` §2.6.

Data citations, all CC-BY or equivalent — **cite them when you use the data**:

- **GDP** — Elipot, S. et al., *A global surface drifter data set at hourly
  resolution*, and the NOAA AOML Global Drifter Program's 6-hourly
  quality-controlled interpolated product. NOAA/AOML, Miami.
- **GTMBA** — the TAO Project Office of NOAA/PMEL: TAO/TRITON, PIRATA and RAMA.
  McPhaden, M. J. et al., *The Tropical Ocean-Global Atmosphere observing
  system*, JGR 103(C7), 1998.
- **SOCAT** — Bakker, D. C. E. et al., *A multi-decade record of high-quality
  fCO₂ data in version 3 of the Surface Ocean CO₂ Atlas (SOCAT)*, ESSD 8(2),
  383–413, 2016. Release v2026, DOI 10.25921/8dba-fr90, NCEI accession 0315110.
  CC-BY 4.0; the SOCAT Fair Data Use Statement applies.
- **slatrack** — Copernicus Marine Service product
  `SEALEVEL_GLO_PHY_L3_MY_008_062` (DUACS reprocessed level-3 along-track sea
  level), E.U. Copernicus Marine Service Information.
