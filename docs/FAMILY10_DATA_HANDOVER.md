# Family 10.1 — four observation stores and the registry: a self-contained data handover

PDF design note: https://blauewelt.github.io/earth/ml/paper/notes/family10.pdf

**For an agent that has not seen this repository.** Everything needed to
download, open, validate and search family 10's tier-P observation stores, and
to read the registry that ties them to the gridded tensors, is on this page;
nothing below requires reading another document. Where a section says "see
also", it is optional background. Written 2026-09-13, the day the builder
landed; the first four stores published 2026-09-14; **rewritten 2026-09-15 and
completed 2026-09-16 for family 10.1, which is the version to ingest.**

**What 10.1 is, in one sentence.** The same four stores, the same rows, the
same nine arrays and the same registry as family 10.0, with **one column
changed**: the time is now `time_s.npy`, **int32 seconds** since
1982-01-01T00:00:00Z, in place of 10.0's `time_days.npy`, float32 days. §2.3
says why that was worth a rebuild, and §2.4 says what a reader has to change.

**Read §9 before you trust a number.** As of 2026-09-16, **all four 10.1
stores are published and verified** — `gdp` (surface drifters, 48,480,798
rows), `gtmba` (the tropical moored arrays, 1,001,282), `socat` (ship CO₂,
41,830,675) and `slatrack` (along-track sea level from 29 altimeter missions,
2,030,800,176), **2,122,112,931 observations** — and the registry is complete
again at nine groups with `groups_missing: []`. Each store's row count and
per-year counts were recomputed from the new `time_s` column and match its own
ledger exactly. The rows themselves are the ones independently verified on
2026-09-14 under 10.0 — every file re-hashed against its own record, every
count, range and per-channel statistic recomputed from the arrays rather than
read out of the metadata claiming them — and that evidence is
`docs/FAMILY10_VERIFICATION_2026-09-14.md`, whose §10 adds the 10.1 checks.

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

Terms used below, once. **Epoch** — 1982-01-01T00:00:00 UTC, the instant every
time in these stores is counted from. **Pentad / bin** — a five-day period;
this project counts them from the epoch, so in family 10.1
`bin = floor(time_s / 432000)` — 432,000 being five days in seconds — and a bin
is **negative** before 1982. **Schema version** — which time column a store
carries: schema 1 is 10.0's float32 `time_days`, schema 2 is 10.1's int32
`time_s`; every `store.json` states its own. **CSR** — "compressed sparse row": an
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
store under **`tensors/family10_1/`** — note the `_1`, which is the family
version `10.1` with the dot written as an underscore. Public, no token needed,
plain HTTPS:

```
https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family10_1/<store>/<file>
```

(`resolve/main/...` answers a 302 redirect to a CDN URL — follow redirects.)
One file, for example:

```
curl -L -o store.json \
  https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family10_1/gdp/store.json
```

The three Hub prefixes family 10.1 uses, all derived from one constant
(`ml/family10_store.FAMILY_VERSION`):

| what | where |
|---|---|
| the four stores | `tensors/family10_1/<store>/` |
| the registry | `tensors/family10_1/family10.json` |
| the `slatrack` build's per-year column parts | `partials/family10_1/slatrack/<year>/` |

The `partials/` prefix is build scaffolding, not data to ingest — §11 explains
what it is for. A reader wants the first two.

| store | what it measures | C | record | footprint `(log2_fp, log2_dt)` |
|---|---|---|---|---|
| `gdp` | surface drifters: 15 m velocity, sea-surface temperature, drogue state, every 6 h | 4 | 1979-02 → | (−4, −4.3) — a point, a quarter-day sample |
| `gtmba` | the tropical moored arrays: one row per mooring per day, 18 quantities | 18 | 1977-11 → | (−4, −2.3) — a point, a daily mean |
| `socat` | ship and mooring surface-water CO₂ fugacity with SST, salinity, pressure | 4 | 1957 → | (−4, −4) — an underway measurement, minutes |
| `slatrack` | along-track sea-level anomaly, 29 altimeter missions at 1 Hz (≈ 7 km) | 3 | 1993 → | (−2.0, −4) — a 7 km along-track cell |

### 2.1 · The files — nine arrays plus `store.json`, the same in every store

| file | dtype · shape | what |
|---|---|---|
| `bin.npy` | int16 [N] | five-day bin, `floor_divide(time_s, 432000)` — exact integer arithmetic. **May be NEGATIVE**: drifters begin in 1979 and SOCAT in 1957, and those rows are kept, so the store is the whole record and the *loader* clips to whatever axis a tensor has |
| `time_s.npy` | int32 [N] | **seconds** since 1982-01-01T00:00:00 UTC, negative before it. int32 spans 1913-12-13T20:45:52Z to 2050-01-19T03:14:07Z, and the builder refuses a row outside that rather than letting it wrap |
| `lat.npy` | float32 [N] | degrees north |
| `lon.npy` | float32 [N] | degrees east, in **[−180, 180)** |
| `values.npy` | float16 [N, C] | the channels of §3, in **raw units**, **NaN = not measured** |
| `platform.npy` | int64 [N] | the platform: a drifter's AOML id, a mooring's WMO code, a cruise's expocode hashed, an altimeter mission hashed. §4.3 gives the rule |
| `qc.npy` | uint8 [N] | 0 not assessed, 1 good, 2 probably good, 3+ the source's worse grades. The builder keeps ≤ 2 by default and records every drop |
| `fp.npy` | float16 [N, 2] | `(log2_fp, log2_dt)` per row. Constant down the column for all four sources — but **stored per row**, so a store that later mixes supports reads identically |
| `bin_offsets.npy` | int64 [B + 1] | CSR offsets over **this store's own** bin range: the rows of bin `b` are `[off[b − bin_first], off[b − bin_first + 1])`, and `bin_first` is in `store.json` |
| `store.json` | JSON | the schema, the footprint constants, the QC policy, the provenance (URLs, verification dates, builder commit), the per-year counts, the per-channel measured fraction and value range, and the **sha256 of every file above** |

**Rows are sorted by `(bin, time_s)` ascending.** That order is what makes
`bin_offsets` valid at all, and every consumer may rely on it. Under 10.1 the
sort inside a bin is a real ordering of the measurements: two rows a second
apart are two distinct timestamps, where 10.0 could only say they were within
84 seconds of each other.

**`bin_first` is not zero.** Family 8's Argo index runs from bin 0 over 3,143
entries because Argo starts in 2004, inside the epoch. A family-10 store's index
runs over its own range, which for `socat` begins around bin −1825 (1957). A
consumer that assumes 0 reads the wrong pentad and nothing says so — use
`bin_first`, or use the reader in §5, which does.

### 2.2 · The registry

`tensors/family10_1/family10.json`, written by `ml/build_family10_registry.py`,
lists **every group of the family in one file** — the tier-G channel groups of
the gridded tensor *by reference to its manifest*, family 8's Argo store, and
these four. Per group: `tier`, `layout`, `cadence`, `channels` with units, the
footprint constants, `bin_first`, its own `schema_version`, the files with
their sha256, the sources and the builder commit. **A consumer dispatches on
`tier` and needs nothing else.**

The registry at the top of the file says `family_version` **"10.1"** and
`schema_version` **2**; each group repeats its own, because they are not all
the same — family 8's Argo store is a tier-P group at **schema 1** and stays
that way (§7), while `gdp`, `gtmba` and `socat` are schema 2. A consumer can
therefore see from the registry alone which groups carry the seconds.

The registry as it stands — regenerated **2026-09-16T04:13:38Z**, 47,181 bytes,
sha256 `abf08f61c226b6f0a85591acbb0cad970f98fc085a7c7b1acf0cb0b3648ca90e` —
carries **nine groups, with `groups_missing: []`**: four tier-G groups (`g025`,
`g100`, `oc025`, `rg100`) at recipe `f7l2`, and five tier-P stores (`argo`,
`gdp`, `gtmba`, `socat`, `slatrack`). Every tier-P entry's `N`, `bin_first`,
`bin_last`, file list and nine sha256 values agree with that store's own
`store.json`, digest for digest.

Two properties are load-bearing. A group the builder could not read is **not**
in `groups` and **is** in `groups_missing` — a registry that listed a group it
could not describe would be worse than a short one, and `slatrack` was in
exactly that state for the hours between the three keyless builds and its own.
And `tier_g` says *which* family-7 manifest it
describes: it names **family 7.2, recipe `f7l2`** — the global gridded tensor
with ocean colour added as a fourth channel group, built 2026-09-14T19:58:44Z —
with `fallback_note: null`, and it is **unchanged by 10.1**. Tier G is
referenced by manifest, not rebuilt: family 10.1 changes the tier-P time column
and nothing on the gridded side. For ingesting the gridded half, the pointer is
still `docs/FAMILY7_DATA_HANDOVER.md`. (The `fallback_note` field exists
because an earlier registry was written while that manifest was still a 404 and
fell back to `f7l0`, the same tensor without ocean colour — it **said so** in
that field rather than silently describing a different tensor.)

### 2.3 · 10.1 supersedes 10.0, and the 10.0 directories stay on the Hub

`tensors/family10/` — the four stores published on 2026-09-14 — is still
there, still readable, and still exactly what the 2026-09-14 verification
checked. It is kept as history. **Ingest 10.1 instead**, for one reason.

10.0's time column is `time_days.npy`, float32 days since the epoch. A float32
carries 24 bits of mantissa, so the gap between two representable values grows
with the magnitude: at the start of the altimeter record (1993, ≈ 4,000 days)
it is **21 seconds**, and at the end (2024, ≈ 15,700 days) it is **84
seconds**. That is harmless for a store sampled every six hours or once a day,
and it was chosen when those were the only stores in view.

`slatrack` samples **once a second**. Measured on the published 10.0 store:
21 to 84 consecutive along-track samples carried the *identical* timestamp —
**up to 316 rows to one value**, which at 6.5 km between samples is 550 km of
one satellite's ground track with a single time on it. Nothing in 10.0 is
wrong: the rows are in the right order and the bins agree with the column. The
column simply cannot express a distinction the archive published, so every
consumer had to be told "use the row order, not the timestamp". A format that
needs that instruction alongside it is the wrong format.

An integer second is the finest thing any of these four archives reports. It
is exact everywhere, it costs the same four bytes, and it takes the float
rounding out of the bin derivation at the same time. Three smaller faults go
with it:

- **`socat`: 908 rows sat in the wrong pentad.** A timestamp within ~84 s
  *below* a pentad boundary rounded **up** across it in float32, so the row was
  filed in the next bin. Measured on the two published stores: 908 of
  41,830,675 rows move, and 10.1 puts each in the pentad it was measured in.
  (Row 232,712, for instance, is 356,399,987 s — thirteen seconds before the
  boundary at 4,125 days — and 10.0 stored it as exactly 4125.0.) No row is
  added or lost: `N` and the per-year counts are identical.
- **`socat`: ≤ 3 rows a year read as the wrong calendar year.** The same
  rounding at midnight on 31 December — three rows late on 2024-12-31 read as
  2025-01-01 when the year was recomputed from `time_days`. Under 10.1 a year
  recomputed from `time_s` reproduces `store.json`'s `per_year` block exactly,
  in every year, for every store.
- **`slatrack`: 26 rows the year lanes had declined.** The same float rounding
  at the other edge of the year. A mission publishes a sample a fraction of a
  second *under* midnight on 1 January; 10.0's per-year window test compared in
  float64 days and put it outside the year it was fetching, so the row was
  never written. 10.1 rounds the timestamp to a whole second with `np.rint`,
  which places it at exactly `YYYY-01-01T00:00:00`, inside the year. The 26
  rows are spread over **20 years, one to three each, one per mission with a
  sample at the year's first second**, and **no year loses a row** — the
  10.1 per-year block is the 10.0 one plus those 26 and identical everywhere
  else. This is the one place where 10.1's `N` is not 10.0's: 2,030,800,176
  against 2,030,800,150 (§9.1).

### 2.4 · What a reader has to change, coming from 10.0

Three things, and nothing else:

1. Read `time_s.npy` (int32) where you read `time_days.npy` (float32).
2. `bin = floor_divide(time_s, 432000)` where you had `floor(time_days / 5)`.
   Integer arithmetic, no float anywhere in it.
3. Point your URLs at `tensors/family10_1/` instead of `tensors/family10/`.

Days, if you want them, are `time_s / 86400.0`. Every value int32 can hold is
exactly representable in float64, so that is one correctly-rounded division and
`round(days * 86400)` gives the second back unchanged. A calendar timestamp is
one line:

```python
np.datetime64("1982-01-01T00:00:00") + time_s.astype("timedelta64[s]")
```

Everything else is untouched: the nine arrays, the CSR `bin_offsets`, the
negative bins before 1982, longitude in [−180, 180), the channels and their
order and units, the quality-control policy and its `qc_keep_max`, the
platform-id rule, the footprint constants, and the fact that nothing is
z-scored or anomalised. `bin_first` and `bin_last` are the same integers per
store. For `gdp` and `gtmba` the other **eight arrays are byte-identical to
10.0's** — same sha256, digest for digest; only `socat`'s differ, because the
908 rows that changed pentad also changed the sort order and therefore every
row-ordered array (`fp.npy` and `qc.npy` are constant enough down the column
that they still hash the same).

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
again before any publish. It asserts: rows sorted by `(bin, time_s)`; the `bin`
column equal to `floor_divide(time_s, 432000)` on every row; `time_s`
non-decreasing inside every bin, the seam between two blocks included; every
`time_s` inside the source window `store.json` declares in `date_range`; the
CSR offsets spanning exactly the rows and each bin's slice holding only that
bin; `lon` in [−180, 180) and `|lat| ≤ 90`; no infinity anywhere in `values`;
the footprint columns constant; every channel inside its bounds; and a
nearest-neighbour search returning nothing from the future. Each of those is a
property a broken build can have while looking completely ordinary from
outside, which is the only reason they are worth the seconds.

**Three of those checks became exact in 10.1, and one of them became possible.**
The bin check is now integer against integer rather than a comparison against a
float floor. The window check is now a comparison of whole seconds. And
"`time_s` non-decreasing inside every bin" now means something: under 10.0 the
same assertion could only ever be checked to 84 seconds, and a `slatrack` store
passed it trivially because equal timestamps are non-decreasing — which is
precisely the weakness this rebuild removes.

Gone with it is a whole class of fault the 10.0 builder had to guard rather
than prevent. `time_days` was float32 and ran to ~16,000 days, where it resolves
about 0.001 d, so a timestamp a moment before a pentad boundary could round
**up** across it; the builder therefore had to compute `bin` from the float32
value it was about to *store* rather than from the float64 the parser had, or
a row would sit in bin *b* carrying a timestamp that read as bin *b+1* and the
search — which requires `dt_days = 5(b+1) − t ≥ 0` — would silently never
return it for its own anchor. Under 10.1 there is no float in the derivation at
all, so there is nothing to guard.

## 5 · Opening it (numpy only, no dependency on this repository)

```python
import json, numpy as np

d = "tensors/family10_1/gdp"               # a local copy of the directory
meta = json.load(open(f"{d}/store.json"))
assert meta["schema_version"] == 2         # 10.1: the time column is time_s
bin_        = np.load(f"{d}/bin.npy",         mmap_mode="r")   # int16  [N]
time_s      = np.load(f"{d}/time_s.npy",      mmap_mode="r")   # i32    [N]
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
t0 = (np.datetime64("2015-01-03T00:00:00")
      - np.datetime64("1982-01-01T00:00:00")).astype("int64")   # seconds
b  = t0 // 432_000                                              # 432000 s = 5 d
lo, hi = off[b - bin_first], off[b - bin_first + 1]
print(hi - lo, "observations in bin", b)

# the time of those rows, as calendar timestamps
when = (np.datetime64("1982-01-01T00:00:00")
        + np.asarray(time_s[lo:hi]).astype("timedelta64[s]"))
print(when[0], "→", when[-1])
```

Two notes on the arithmetic. `floor_divide` on a negative `time_s` floors
toward minus infinity, which is what a pre-1982 bin needs — Python's `//` and
numpy's `floor_divide` both do this, C's integer division does **not**, so a
port to C or Rust must floor explicitly. And days, where a reader wants them,
are `time_s / 86400.0` — one correctly-rounded division from which
`round(days * 86400)` recovers the second exactly, so it is a change of unit
and not a loss.

**Verify before you use it.** `ml/family10_store.verify_store(dir)` re-hashes
every file against `store.json`'s own record and raises on any mismatch. A
truncated `values.npy` still memmaps and still answers a search, with whatever
its tail happens to hold.

**If you already downloaded a 10.0 store**, it still opens: it carries
`time_days.npy` (float32 days) instead of `time_s.npy`, and `store.json` says
`schema_version: 1`. Multiply by 86,400 to put the two in one unit if you must
— but the precision is **not recovered**, and doing so produces an integer
column that looks exact and is wrong by up to 84 seconds, which is the single
outcome this rebuild exists to prevent. Re-fetch from `tensors/family10_1/`
instead. `ml/family10_store.py` reads both schemas and answers `st.time_s(...)`
on either, converting a schema-1 store's days on the fly.

### 5.1 · The reader, and the search

`ml/family10_store.py` is the only thing in this repository that opens a store,
so the layout is defined once.

```python
from family10_store import Store

st  = Store.open("tensors/family10_1/gdp")                       # a directory
st  = Store.open("chfrank/earth-tensors:tensors/family10_1/gdp") # or the Hub
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
| `n_found`, `row`, `lat`, `lon`, `time_s`, `time_days`, `channels` | | |

`dt_days` is float64 **days** under both schemas — that did not change. The
provenance fields carry the time twice: `time_s` as float64 seconds and
`time_days` as float64 days, float64 rather than int32 so that a miss slot can
be NaN like every other field in the dict.

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
at anchors where the store has data. **The measurement was made on the 10.0
stores and carries over to 10.1 unchanged**: it is a statement about how far
apart observations are, the rows are the same rows in the same order, and a
k-th-neighbour distance in kilometres does not depend on the unit the time
column is written in. `slatrack` is the one still unmeasured —
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

**Argo is still schema 1, deliberately.** It carries `time_days.npy`, float32
days, and joins family 10.1 with no rebuild at all. That is sound because Argo
profiles roughly every ten days: at a resolution of 84 seconds the column can
still order every measurement the archive publishes, which is the whole thing
10.1 was built to fix and the one property Argo never lacked. The reader
converts its days to seconds on the way past, so `st.time_s(...)` answers on it
like on any other store, and the registry states `schema_version: 1` for that
group so a consumer can see the difference rather than assume it away.

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

## 9 · What is published and verified (2026-09-16)

**All four 10.1 stores are on the Hub and verified.** The three keyless ones
were rebuilt from source on 2026-09-15 from builder commit `d7fc30b` on
GitHub-hosted runners — `socat` landing 10:58Z, `gtmba` 11:10Z, `gdp` 11:56Z —
and `slatrack` was re-fetched the same morning on six hosted lanes and
assembled on a rented box overnight, its store written 2026-09-16T03:22:50Z
from builder commit `df788d8`. Each was checked before it was published and
again after: `check_store` over every row (§4.5), then every file downloaded
back from the Hub and re-hashed against `store.json`'s own record, then `N` and
the **per-year counts recomputed from the new `time_s` column**.

| store | N | live bins / total | measured fraction | per-year from `time_s` | vs 10.0 |
|---|---|---|---|---|---|
| `gdp` | 48,480,798 | **3,353 / 3,353 (100 %)** | 0.963865 | 46 of 46 years exact | N and per-year identical |
| `gtmba` | 1,001,282 | 3,386 / 3,446 (98.3 %) | 0.568374 | 48 of 48 years exact | N and per-year identical |
| `socat` | 41,830,675 | 3,266 / 4,910 (66.5 %) | 0.923146 | 68 of 68 years exact | N identical; 908 rows changed pentad (§2.3) |
| `slatrack` | **2,030,800,176** | **2,339 / 2,339 (100 %)** | 0.999052 | 32 of 32 years exact, and equal to the 32 lane ledgers | **N + 26** (§2.3, §9.1) |

**E-079's falsifier said no row may be added, dropped or moved. Three of the
four stores met that exactly; `slatrack` gained 26 rows, and those 26 are the
reason the rebuild was worth doing rather than an argument against it.** Each
is a real archive sample that 10.0's float64-days window test placed outside
the year it was fetching, one per mission at exactly `YYYY-01-01T00:00:00`,
spread over 20 years at one to three each, with no year losing a row (§2.3).
Everything else the falsifier named holds: each store's `bin_first`,
`bin_last`, `C`, channel names and order, footprint constants, `qc_keep_max`
and overall measured fraction are the values 10.0 published.

**The falsifier's other clause was WRONG, and the 10.1 bins are the correct
ones.** E-079 §10.1 asserted that "the bin of a timestamp is the same integer
whether it is derived from exact seconds or from the float32 days that
timestamp rounded to … already forced to agree by v1's cast-then-bin rule".
That is false. The cast-then-bin rule made 10.0 *self-consistent* — its `bin`
always agreed with its own stored `time_days` — but the value it agreed with
had already moved: a timestamp within 84 s below a pentad boundary rounds *up*
across it in float32, and both the column and the bin then carry the rounded
value. Measured on the two published `socat` stores by differencing their CSR
`bin_offsets`: **1,139 bins changed count and 908 rows changed pentad**, with
`N` unchanged and `gdp` and `gtmba` at zero. `slatrack`, sampling once a
second, will have moved far more; that has not been counted.

**Two things did change beside the time column, and both are corrections.**
`socat`'s 908 misfiled rows moved to the pentad they were measured in (§2.3),
which is why its CSR offsets and row-ordered arrays differ from 10.0's while
`N` does not. And `socat`'s ledger now closes: `drop_out_of_range` reads
**2,160,009** where 10.0 read 2,160,005, so
`rows_read − drop_out_of_range − drop_no_fco2 = N` exactly (44,018,204 −
2,160,009 − 27,520 = 41,830,675), where the 10.0 ledger was four short. `gdp`'s
ledger is unchanged — `drogue_uncertain` 1,416,828, exactly the `drogue`
channel's NaN count, beside `drop_no_values` 568 — except that one counter is
spelled `empty_month` where 10.0 spelled it `missing_month`.

### 9.1 · The sha256 of every published file

Take these as the record of what is on the Hub as this was written. A reader
that re-hashes a download and gets a different digest has a corrupted copy, not
a newer store — the stores are not rewritten in place.

**`gdp` — surface drifters, 1.70 GB in ten files.**

| file | bytes | sha256 |
|---|---|---|
| `bin.npy` | 96,961,724 | `889d05d91d6877fcb7ed06511347568de47048098dfca7f1bffc7988706b009c` |
| `time_s.npy` | 193,923,320 | `0ea90b287e94963bc23e40e601b70dbcf2d88b7274150aeea5afc675247e100b` |
| `lat.npy` | 193,923,320 | `da1598d9c89034d3598899ab7954465929473d584af8ea274465ac87aef88ef3` |
| `lon.npy` | 193,923,320 | `66ea4b18e5bb0b01d6c30e856025d38e818e4ed5bf0a1e77f4914cbf1d5d3aa9` |
| `values.npy` | 387,846,512 | `0f230f2e5712c3a0723417d94c1d5943befc0d04552e330d0ea90fdc9f042838` |
| `platform.npy` | 387,846,512 | `ff7690825f4f32b8e359e9bf5dfae4a2c4f285addeccd42c692f7cb4f6cac462` |
| `qc.npy` | 48,480,926 | `3169df79e7880c6088369d0153c7fb04b93e95e19d4e6017fb376c9471f36cac` |
| `fp.npy` | 193,923,320 | `318f11b3ed354cbc68bc7a396010beb2af76b9b95302b2338aad5ecbf86054e8` |
| `bin_offsets.npy` | 26,960 | `f84f5e53f6631055c797003ce329960479fc5f48df9f72f06aef534e2574f6a2` |
| `store.json` | 7,501 | *the file carrying the nine digests above* |

**`gtmba` — the tropical moored arrays, 63.1 MB in ten files.**

| file | bytes | sha256 |
|---|---|---|
| `bin.npy` | 2,002,692 | `92f922d87f371d0bfa3eecd0f8f2d89a4b1f43c6f5db01b2a42814cb09437e88` |
| `time_s.npy` | 4,005,256 | `e57e5369d5d5c7bc77779b31214abf4c41efd9ecd5f332cb02ea9f3cd5271137` |
| `lat.npy` | 4,005,256 | `b7c17c6609455c1d57ca498d42acab392da9ad2356e4a48140bf4f670a975188` |
| `lon.npy` | 4,005,256 | `2e748c084f963f63daf71cabcdc37613eff82232940d470a742f40959f987e54` |
| `values.npy` | 36,046,280 | `96b89786090b0cfd3448fcf10ea89e91522d1999d7f64b0dfb5e66ad25a56488` |
| `platform.npy` | 8,010,384 | `c75ea083a1424df177223f446e9ecb2a29e936668ba32b9a384089e2586af630` |
| `qc.npy` | 1,001,410 | `49e7051022fb84b32c0a15335a1048db0e3811c8ac3f5f561619793313ee2ed1` |
| `fp.npy` | 4,005,256 | `4c8014d9562a3b7cf64dc61951ab446e37aa39eab94d06c67f295c85dc74c656` |
| `bin_offsets.npy` | 27,704 | `c7bd546dec03b75aab45f7248aa6eb7184f0e88c3a823ba6b629c8bc4b2c68ac` |
| `store.json` | 11,969 | *the file carrying the nine digests above* |

**`socat` — ship and mooring surface CO₂, 1.46 GB in ten files.**

| file | bytes | sha256 |
|---|---|---|
| `bin.npy` | 83,661,478 | `f50b9ccfee28bc96a92cf4d52d81cb5206d4f448f8f67c651a681876685b7574` |
| `time_s.npy` | 167,322,828 | `8de56c67e5185722141a70a1e2180cfbb6cb9634d2fbeb2c90b764bbc3a7396d` |
| `lat.npy` | 167,322,828 | `c79cb63ecf7fb4cfb6f833f61530e95780d4ee271917f285a7468c2d8e6d4a05` |
| `lon.npy` | 167,322,828 | `fef6f87b37faffed580a9cc9070d733738ebe2007db0408d6abb61d948ad21b1` |
| `values.npy` | 334,645,528 | `6f72d4d99aa930e7d465eef49e5f402de13c1f7796db7303b43c1fce740308ea` |
| `platform.npy` | 334,645,528 | `f7c06897acc4624660503a31450c520f346895e201a15706c0b2f1175d80682f` |
| `qc.npy` | 41,830,803 | `bed0417211c2fca1094e5a492f7433b0be38f5178341ef707264ae5272d5947f` |
| `fp.npy` | 167,322,828 | `d583a63e591f6acb7ffbce61494c6dfada7fbbc39e70e9a58ff18a892df3ec5a` |
| `bin_offsets.npy` | 39,416 | `988f7e7fbc37941dd15feb86aa3831d6ca7ed368f6c328fc1b56011fcb6a73d2` |
| `store.json` | 8,531 | *the file carrying the nine digests above* |

**`slatrack` — along-track sea level, 67.02 GB in ten files.**

| file | bytes | sha256 |
|---|---|---|
| `bin.npy` | 4,061,600,480 | `9ee41d57a7d7b86f008fedbb0b65e0120504c8332e8bf0a1cc566ff48da2f49e` |
| `time_s.npy` | 8,123,200,832 | `e5a2032124182da1ceed4b5f61b8561b7881471eb192c5e5ab20eb8c2379ee44` |
| `lat.npy` | 8,123,200,832 | `5bcac25a8f6229e638f60714b1c503b846d476d753de91e3c28883f20397e75b` |
| `lon.npy` | 8,123,200,832 | `95be81ef42b5a82c7187f76f12c9e06142d3246c09df7c1e731bb7f3e87a8b66` |
| `values.npy` | 12,184,801,184 | `1e6de6e5e966354d55310193ccca30e489b32b402301ef0efce544dfc8321f9f` |
| `platform.npy` | 16,246,401,536 | `3c0b416ea3333d8c5e16a08251d1756b327b1df2af622042375441b8691aa624` |
| `qc.npy` | 2,030,800,304 | `57655d198d89d1127bcbf2c57602eb24c7cf9e8c46dcd10bd234986457eed43b` |
| `fp.npy` | 8,123,200,832 | `9f29c5f53c4792f772830fd10a95bca98443b3e8e489ba4c0e9648430e4fa1c2` |
| `bin_offsets.npy` | 18,848 | `d773f67a3ce4e628e2f5868e73f737883cafb586ee9f8ced82048c6836eaef49` |
| `store.json` | 13,986 | *the file carrying the nine digests above* |

Every byte count is `128 + N × itemsize × columns`, the 128 being the `.npy`
header, so a size that does not match means a truncated file. Its layout is
identical to the other three — the same nine arrays, the same dtypes, the same
CSR index — with `C = 3` and channels `sla`, `sla_unfiltered`, `mdt`, footprint
(−2.0, −4), `qc = 1` on every row, `bin_first` 803, `bin_last` 3141.

**Its `N` is 2,030,800,176**, which is 10.0's 2,030,800,150 plus **26 rows**,
and those 26 are a correction rather than new data (§2.3). The count is the sum
of the 32 fetch lanes' own `done.json` row counts on the Hub, 32 of 32 years
present, and it equals the store's own `per_year` block entry for entry.

### 9.2 · What the 2026-09-14 verification established, and why it carries over

The rows in these stores are the rows that were independently verified on
2026-09-14 under 10.0 by sessions that did not build them. That verification
re-hashed every file, then recomputed from the arrays — not from the metadata
claiming them — `N` and the bin range, the live-bin count, the ascending sort,
the CSR offsets reproduced by `searchsorted`, the bin/time agreement, longitude
in [−180, 180), constant footprint columns, quality codes within `qc_keep_max`,
per-year counts summing to `N`, and every per-channel measured fraction,
minimum, maximum and mean: 4 of 4 channels reproduced exactly for `gdp`, 18 of
18 for `gtmba`, 4 of 4 for `socat`, 3 of 3 for `slatrack`.

`slatrack` could not be opened at all in that sandbox — 67.02 GB against 14 GB
of free disk — so it was checked by **streaming**: all nine arrays read over
HTTPS in row-aligned lockstep, one thread per file hashing its own bytes, one
consumer recomputing every check on the chunk in flight, in one pass, 837 s at
80 MB/s. Being a pass over every row rather than a sample, "sorted" meant
2,030,800,149 adjacent comparisons and "lon ∈ [−180, 180)" meant 2.03 billion
of them. It also found the fault this rebuild answers: up to 316 consecutive
rows sharing one `time_days` value.

It carries over because 10.1 changes one column and nothing those checks were
about. What it does **not** cover is 10.1's own arrays — those were checked by
`check_store` and by the re-hash on publish, then from outside on 2026-09-16 by
the checks in §10 of `docs/FAMILY10_VERIFICATION_2026-09-14.md`, and the
numbers in §9.1 are the files on the Hub today. The full 10.0 evidence, number
by number, is that same page's §1–§9.

**The 84-second collision is gone, and the honest statement of what replaced it
is not "one row per timestamp".** E-079 §10.1 expected `slatrack`'s largest
group of rows sharing a timestamp to fall from **316 to 1**. Measured on the
published 10.1 `time_s` column, in windows of two million consecutive rows read
by HTTP range: the first two million rows hold at most **3** rows in any one
second, two million from the middle of the store at most **6**, the last two
million at most **8**, and a window 95 % of the way through the store **9**.
That is not a residual artefact — it is the number of altimeters flying at that
second, each contributing one 1 Hz sample. One row per second per mission is
the physical floor, and up to nine missions were in orbit in 2022–24. So the
right sentence is **316 → the number of missions aloft**, and `dt_days` now
separates two samples of the *same* mission, which is what a consumer needs.

The published directories:

- [the `gdp` store on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/tree/main/tensors/family10_1/gdp) — surface drifters, 6-hourly, 1.70 GB.
- [the `gtmba` store on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/tree/main/tensors/family10_1/gtmba) — the tropical moored arrays, daily, 63.1 MB.
- [the `socat` store on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/tree/main/tensors/family10_1/socat) — ship and mooring surface CO₂, 1.46 GB.
- [the `slatrack` store on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/tree/main/tensors/family10_1/slatrack) — along-track sea level, 29 altimeter missions, 2,030,800,176 rows, 67.02 GB, published 2026-09-16.
- [the family-10 registry on the Hub](https://huggingface.co/datasets/chfrank/earth-tensors/blob/main/tensors/family10_1/family10.json) — `family_version` "10.1", `schema_version` 2, **nine groups with `groups_missing: []`**, regenerated 2026-09-16T04:13:38Z; tier G is recipe `f7l2` with four groups and no fallback note, and every tier-P entry's N, `bin_first`, file count and sha256 agrees with the `store.json` above.

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

**WHAT `slatrack` MEASURED, and what still holds.** The 10.1 store reads
**2,030,800,176 rows** (26 more than 10.0, §2.3), C = 3, bins 803..3141 with
all 2,339 live, 1993-01-01 → 2024-12-31, 67.02 GB, 28 of its 29 missions
carrying rows, and 0.999052 of its value slots measured — the same figures the
10.0 store published apart from the 26 rows. Three findings a consumer should
carry, none of them a build error:

- **The first one is what 10.1 fixes, and it is gone.** Under 10.0 the time
  column was float32 days against 1 Hz sampling, so it was 84× coarser than the
  data: 21 s of resolution in 1993 and 84.4 s from 2015 on, with 28 to 316
  consecutive rows carrying the identical timestamp — at 6.5 km between samples,
  up to 550 km of one satellite's track with one time on it, inside which
  `knearest`'s `dt_days` could not order anything. Under 10.1 the column is
  integer seconds, so each along-track sample carries its own timestamp
  wherever the archive published distinct seconds, and `dt_days` orders them.
  Measured on the published column, the largest group of rows sharing one
  second is **3, 6, 8 and 9** in four separate two-million-row windows — one
  sample per mission aloft at that second, which is the physical floor rather
  than an artefact (§9.2). **Calendar dates may now be derived from `time_s`
  directly** — the "derive dates from `bin`, not from the timestamp" rule that
  10.0 needed is retired.
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

**The 1994 lane carries its gap as a MEASUREMENT, and that is new in 10.1.**
`partials/family10_1/slatrack/1994/counts.json` records
`mission_year_gap_measured: 1` and a `gaps_measured` entry naming the mission
whose 1994 listing was empty: the ERS-1 35-day dataset
`cmems_obs-sl_glo_phy-ssh_my_e1-l3-duacs_PT1S_202411`, whose whole archive was
then listed and found to hold **468 files spanning 1992-10-23 to 1995-05-15
with none in 1994**. Its published STAC window over-declares; the 1994 geodetic
phase is the separate `…_my_e1g-…` dataset, which lists and downloads
normally. Before the builder fix `caa00d3` an empty mission-year listing was a
flat refusal, so this real archive gap failed the lane; it now measures the
mission's whole archive once and records what it found, and a filter that
missed files the archive holds is still a refusal. The year's own ledger reads
`rows_read` 29,147,518 and `kept` 29,146,039 over 29 parts.

And two provenance strings in `slatrack`'s `store.json` are stale, in 10.1 as
in 10.0 — checked on the published 10.1 file, not assumed. `verified` still
ends "Neither route can be run from this sandbox … NOT YET MEASURED: the
remote path layout of the original files", and `sources` still names the
`copernicusmarine subset` call rather than the
`copernicusmarine.get`-on-original-files route that actually ran, twice now,
end to end over 32 years. Both are class constants in the builder; neither
affects a byte of any array, and the product and dataset identifiers they name
are right.

**Historical, and left standing because it is why the build looks the way it
does: as of 2026-09-13 the two Copernicus repository secrets were believed not
to exist** in this repository. The workflow's preflight says so in plain words
and refuses before anything is fetched. They had in fact existed since
2026-08-16 and nobody had checked, which is why `slatrack` was the last store
built rather than the first. The builder reads
`COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`
from the **environment only** — never a file, never a command line.

**Smaller notes, all still current:**

- **Both ledgers are closed in 10.1.** `gdp` reads `drogue_uncertain`
  1,416,828 — exactly the `drogue` channel's NaN count — beside
  `drop_no_values` 568, and `rows_read − drop_pos_err − drop_no_values − N` is
  zero. `socat` reads `rows_read` 44,018,204, `drop_out_of_range` 2,160,009 and
  `drop_no_fco2` 27,520, which subtract to `N` exactly. Both were faults in the
  first 10.0 builds that the rebuilds and then 10.1 closed; §10 keeps the
  mechanics, because a consumer holding an older copy still needs them.
- **The `socat` resume granularity is the whole stream, not the year.** The
  synthesis file is sorted by expocode, not by time, so there is no per-year
  request to make and an interrupted fetch re-reads the file from the start.
  Each store's `store.json` states its own `resume_granularity`; the other three
  say `"year"`.
- **The registry's tier-G half is `f7l2`, four groups, no fallback note, and
  10.1 did not touch it.** Family 7.2 is the global 0.25° gridded tensor with
  ocean colour added as a fourth channel group, `oc025`, built
  2026-09-14T19:58:44Z (recipe `f7l2`, E-077). The registry names it by
  reference — `stem` `family7_global025_pentad_l2`, `recipe` `f7l2`,
  `fallback_note` **null** — and the four tier-G file sha256 values match that
  manifest digest by digest. Family 10.1 changes the tier-P time column and
  nothing on the gridded side, so a consumer already ingesting tier G has
  nothing to redo; the pointer for that half is
  `docs/FAMILY7_DATA_HANDOVER.md`. (The `fallback_note` field exists because an
  earlier registry was written while that manifest was still a 404 and fell
  back to `f7l0`, the same tensor without ocean colour — it **said so** in that
  field rather than silently describing a different tensor.)
- **`slatrack`'s suggested search radii in §6 are still a guess**, and it is
  now the ONLY store of the four for which that is true — the other three were
  measured on the published stores. The measurement is not cheap
  here: `measure_knn.py` memory-maps a store, and this one is 67 GB, so it wants
  a box with the disk rather than a sandbox. Measure the k-th-neighbour distance
  before fixing them, with `ml/tools/family10_verify/measure_knn.py`. Expect the
  answer to be small: consecutive along-track samples are 6.2–6.5 km apart, so
  k = 8 at any anchor the satellite flew over is ~50 km of one pass — the same
  "the k tokens are one platform" property §6 already measured for the other
  three, in its most extreme form.
- **A 10.0 copy of `slatrack` cannot order its rows inside 84 seconds** (§10,
  and §9 above). Anyone still holding one should know that its `dt_days` ties
  are not ties in the world. This is the whole reason 10.1 exists, and it is
  the one thing re-fetching actually buys.

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
- **`slatrack` is 67.02 GB** (67,016,439,666 bytes over its ten files,
  measured on the published 10.1 store) and is the only store
  that needs credentials AND a large machine — the only one whose build runs on
  two machines, because no single machine may have both. `ml/CLAUDE.md` §6 forbids
  the Copernicus credentials on a rented box, so the fetch may only happen on a
  GitHub-hosted runner; a hosted runner has ~14 GB of disk and a six-hour job
  cap, so the assembly may not happen there. The two halves are joined by the
  Hub (see §11). The other three stores are keyless and fit a hosted runner
  whole. It is also 22× the other three put together, so anything that opens
  "all the tier-P stores" should size for this one alone.
- **`time_s` is int32, and int32 ends on 2050-01-19T03:14:07Z.** That is 25
  years of headroom and the builder refuses a row past it rather than letting
  one wrap into 1913, but it is a real edge: widening the column to int64 —
  which doubles it — is the change to make when the archives get there. The
  other end, 1913-12-13T20:45:52Z, is comfortably before SOCAT's 1957.
- **A `bin` derivation must FLOOR toward minus infinity.** `bin =
  floor_divide(time_s, 432000)`, and `time_s` is negative before 1982. Python's
  `//` and numpy's `floor_divide` floor; C, C++, Java, Go and Rust integer
  division truncates **toward zero**, which for a pre-1982 row gives the bin
  above the right one. A port must floor explicitly.
- **The 10.0 time gotchas are gone, and a 10.0 copy still has them.** If you
  are holding `tensors/family10/` rather than `tensors/family10_1/`: its
  `time_days` is float32 with 84 s of spacing at the end of the record, so
  three `socat` rows late on 2024-12-31 read as 2025-01-01 and, in `slatrack`,
  up to 316 consecutive 1 Hz samples carry one timestamp — 550 km of track with
  one time on it, inside which `dt_days` cannot order anything. On such a copy,
  derive calendar dates from `bin` rather than from the timestamp, and treat
  `store.json`'s per-year counts (taken from the source dates) as the correct
  side. On a 10.1 store none of that applies: the column is exact seconds, so
  dates come from `time_s` and the per-year counts recompute from it exactly.
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
- **A `counts` ledger you are holding may be an old one; the 10.1 ledgers
  close.** Three faults were found and fixed in the `counts` block over
  2026-09-14, all of them ledger-only — no array was ever affected — and the
  10.1 stores carry the corrected form. In `gdp`, `drogue_uncertain` once ran
  568 ahead of the NaN count in the `drogue` channel: those 568 source rows had
  nothing measured on them at all, so they were counted and then dropped, and
  the builder now records that drop as `drop_no_values`. In `socat`, the
  counters were once inflated **59×** — one pass over the file, its counters
  copied into every year part and then summed across the 59 years that held
  rows — and separately ran four short on `drop_out_of_range`. On a 10.1 store
  both ledgers subtract to `N` exactly (§9). If a `counts` block you hold does
  not, you have an early 10.0 copy; re-fetch rather than reasoning from it.
- **`slatrack`'s `store.json` carries two stale provenance strings, in 10.1 as
  in 10.0.** Checked on the published 10.1 file: `verified` still ends "Neither
  route can be run from this sandbox … NOT YET MEASURED: the remote path
  layout of the original files", and `sources` still names the
  `copernicusmarine subset` call. Both were written before any `slatrack` fetch
  had succeeded; both builds that produced a store ran the
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
git commit, the build time, the source URLs and the verification sentence
above, plus its `family_version` and `schema_version`.

**Every 10.1 store is rebuilt from the source archive, never converted from a
10.0 store.** That is deliberate. Multiplying float32 days by 86,400 produces
an integer column that *looks* exact and is wrong by up to 84 seconds — the
single outcome this change exists to prevent — and the same trap sits inside
the build, because the `slatrack` lanes park column parts on the Hub and a
different machine assembles them, so the two halves of one build could be a
schema apart. `done.json` records a part's names, bytes and sha256 and nothing
about the layout inside it, so a 10.0 part would verify perfectly and be
unusable. Three guards: `ml/family10_parts_hub.py` refuses a `time_days` part
on push and on pull, the assembler refuses one before either assembler reads
it, and the fresh `partials/family10_1/` prefix and `ml/cache/family10_1` work
directory mean a resume cannot find one in the first place.

**`slatrack`'s build path is different.** It is the one store built on two
machines, for the reason in §9: the credentials may only live on a
GitHub-hosted runner and the 67 GB assembly may only happen on a box. The seam
is a Hub prefix. This ran for 10.0 on 2026-09-14 and again for 10.1 on
2026-09-15/16.

1. `.github/workflows/family10-slatrack-fetch.yml` — hosted lanes
   (`runs-on: ubuntu-latest`, hard-coded), six of them over weighted year
   ranges, `schedule:` every six hours as the resume mechanism. Each lane
   fetches ONE YEAR (`--stage index,fetch --start Y-01-01 --end Y-12-31`),
   pushes that year's column parts to
   `partials/family10_1/slatrack/<year>/` on the dataset repo with
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

**What it cost, measured — 10.0, on 2026-09-14.** The six lanes ran 10–16
minutes per year and are GitHub-hosted, so the ~47 hours of credentialed
downloading cost **$0**. The assembly run —
[#11 (E-079 `slatrack` — assembly of 32 Hub year-parts, streaming sort, publish, registry)](https://github.com/blauewelt/earth/actions/runs/34894902246) —
took **1 h 18 m** on a keyless 250 GB box for **≈ $0.55**: 2,855 s to pull the
32 year-parts back and re-verify them, 12.9 s for the counting pass, 240.9 s to
scatter 2,030,800,150 rows, 112.1 s to sort the 2,315 bins that needed it,
~350 s for the chunked `check_store`, and 1,003 s to publish ten files and
verify each by restore, at a peak RSS of 66.06 GB. The registry followed in
four seconds. The estimate at dispatch was one box-day and ≈ $8; splitting the
credentialed half onto free parallel lanes is where the difference went.

**And for 10.1, on 2026-09-15/16.** The three keyless stores rebuilt on
GitHub-hosted runners at **$0** —
[family10-build #12 (E-079 10.1 `gdp` rebuild on integer seconds)](https://github.com/blauewelt/earth/actions/runs/34959255275)
76 min,
[#13 (E-079 10.1 `gtmba` rebuild)](https://github.com/blauewelt/earth/actions/runs/34959261348)
30 min and
[#14 (E-079 10.1 `socat` rebuild)](https://github.com/blauewelt/earth/actions/runs/34959267464)
19 min, all dispatched together at 10:41Z. The `slatrack` lanes re-fetched all
32 years the same morning
([family10-slatrack-fetch run 34959274326](https://github.com/blauewelt/earth/actions/runs/34959274326),
six hosted lanes, 10:41Z), one of which refused 1994 on an empty mission-year
listing and was re-run after the builder fix `caa00d3`
([run 34968675755](https://github.com/blauewelt/earth/actions/runs/34968675755)
— see the measured-gap note in §9). The assembly then took three attempts, and
**the second one is the operational lesson of the day**: the box matters more
than the code. `#15` was a four-minute false start (`stage=fetch,publish`
without `index`). `#16` pulled and sorted the whole store correctly and then
uploaded at ~0.3 MB/s — 15.6 GB of 67 in seven hours — and was cancelled at
02:33Z; the host had advertised 332 Mbps of uplink. `#17`
([run 35047822726](https://github.com/blauewelt/earth/actions/runs/35047822726)),
on a *verified* host with 129 GB of RAM and 2 Gbps up, did the pull, the
streaming sort, the checks and the publish in **92 minutes** (02:33:31Z →
04:04:54Z), writing the store at 03:22:50Z. Its registry step then failed on an
HTTP 429 from the Hub's rate limiter, so the registry was regenerated and
restore-verified from a sandbox at 04:15Z with `ml/build_family10_registry.py
--publish`. **Rent a verified host for a 67 GB upload** — the whole of the
difference between #16 and #17 was the host, not the build.

**The assembler that runs there is `assemble_store_streaming`**, selected by
`--assemble auto` (streaming above 50 M part rows, and always for `slatrack`).
It writes the same store as the in-RAM `assemble_store` **byte for byte**, in
three passes over the parts: count rows per bin to get the CSR offsets; scatter
each part's rows into `offsets[bin] + cursor[bin]` through `open_memmap`, in the
order `read_parts` yields them; then sort each bin's slice by `time_s` with a
stable argsort. That is the same permutation `np.lexsort((time_s, bin))`
produces, because lexsort is stable and its final tie-break is input order —
which is what the scatter reproduces.
`tests/test_build_family10_stores.py::test_the_streaming_assembler_writes_the_same_bytes`
proves it by hashing both assemblers' output over one synthetic archive doctored
to hold duplicate `(bin, time_s)` rows across two parts. Peak RAM is one part
plus one bin's slice; a disk preflight computes the store's size from the dtypes
and refuses before pass 2 if free space is under 1.2× it.

Specification: `ml/plans/E079_family10_point_stores.md`, whose §10.1 is the
integer-seconds change this page describes. The design it implements:
`ml/plans/E078_multi_granularity.md`. The footprint token it carries:
`ml/plans/E076_family8_nearest_observations.md` §2.6.

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
