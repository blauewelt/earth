# Family 8 — the Argo observation store: a self-contained data handover

**For an agent that has not seen this repository.** Everything needed to
download, open, validate and search the family-8 Argo store is on this page;
nothing below requires reading another document. Where a section says "see
also", it is optional background. Written 2026-09-07, the day the store was
built (`family8-build #3`).

Family 8 is the dense gridded tensor called family 7 **plus** this store;
§7 says how the two fit together, and §8 says how the store is meant to be
used in training. The family-7 tensor has its own self-contained handover
(optional background: `docs/FAMILY7_DATA_HANDOVER.md` in the same
repository).

---

## 1 · What this is, in one paragraph

Family 8 changes how a *sparse* observing system enters a model. The earlier
representation (family 7) held the ocean interior as a gridded, monthly,
optimally-interpolated product (the Roemmich–Gilson Argo climatology,
`rg100`), live in one five-day bin out of six and a "missing" token
everywhere else. Family 8 replaces that grid with **the raw Argo float
profiles themselves** — every profile from 2004-01-01 to 2024-12-31, quality
controlled, interpolated onto sixteen fixed pressure levels, sorted by
five-day bin — so that a model can ask, for any place and time, *"what are
the k nearest measurements, and how far away are they?"* rather than *"is
there a value in this cell?"*. The store is **2,678,439 profiles**, 234 MB,
raw units (°C, PSU), and every one of the 1,535 five-day bins from 2004 has
between 395 and 2,642 profiles in it.

Terms used below, once: **Argo** — the global array of ~4,000 free-drifting
floats that each dive to 2,000 m and surface every ~10 days measuring
temperature and salinity against pressure. **Profile** — one such dive: a
column of (pressure, temperature, salinity) samples at one place and time.
**Pentad / bin** — a five-day period; this project counts them from
1982-01-01. **dbar** — decibar, the pressure unit; 1 dbar ≈ 1 m of depth.
**PSU** — practical salinity unit. **WMO number** — a float's unique ID.
**CSR** — "compressed sparse row": an offsets array that says where each
bin's rows start and end in a sorted table. **GDAC** — the Argo Global Data
Assembly Centres, the two mirrored archives the raw files come from.

---

## 2 · Where it is and how to fetch it

Hugging Face dataset repository **`chfrank/earth-tensors`**, directory
`tensors/family8_argo_l0/`. Public, no token needed, plain HTTPS:

```
https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family8_argo_l0/<file>
```

(`resolve/main/...` answers a 302 redirect to a CDN URL — follow redirects.)
Thirteen files, 234 MB in total:

| file | dtype · shape | bytes | what |
|---|---|---|---|
| `bin.npy` | int16 [N] | 5,357,006 | five-day bin index, `floor((date − 1982-01-01) / 5 days)` |
| `time_days.npy` | float32 [N] | 10,713,884 | days since 1982-01-01 00:00 UTC, fractional |
| `lat.npy` | float32 [N] | 10,713,884 | degrees north |
| `lon.npy` | float32 [N] | 10,713,884 | degrees east, in **[−180, 180)** |
| `temp.npy` | float16 [N, 16] | 85,710,176 | temperature, °C, at the 16 levels of §3; NaN = level not filled |
| `psal.npy` | float16 [N, 16] | 85,710,176 | practical salinity, PSU, same layout |
| `wmo.npy` | int32 [N] | 10,713,884 | the float's WMO number |
| `cycle.npy` | int16 [N] | 5,357,006 | the float's cycle (dive) number |
| `nlev.npy` | int16 [N] | 5,357,006 | how many good raw samples the profile contributed |
| `maxpres.npy` | float16 [N] | 5,357,006 | deepest good pressure of the profile, dbar |
| `mode.npy` | \|S1 [N] | 2,678,567 | `b'R'` real-time, `b'A'` adjusted, `b'D'` delayed-mode (the Argo data-mode flag; D is the best-calibrated) |
| `bin_offsets.npy` | int64 [3143] | 25,272 | CSR offsets: the rows of bin `b` are `[off[b], off[b+1])` |
| `store.json` | JSON | 7,636 | schema, QC policy, provenance, per-year counts, **sha256 of every file above** |

`manifest.json` beside them is the publisher's record (builder commit,
built-at, the same hashes). **N = 2,678,439.** All eleven column files share
that first dimension and are aligned row for row; the rows are sorted by
`(bin, time_days)` ascending, which is what makes `bin_offsets` valid.

Fetch and verify (bash):

```bash
BASE=https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family8_argo_l0
mkdir -p family8_argo_l0 && cd family8_argo_l0
for f in bin time_days lat lon temp psal wmo cycle nlev maxpres mode bin_offsets; do
  curl -sSL -o $f.npy "$BASE/$f.npy"
done
curl -sSL -o store.json "$BASE/store.json"
python3 - <<'EOF'
import json, hashlib
s = json.load(open("store.json"))
for name, want in s["sha256"].items():
    got = hashlib.sha256(open(name, "rb").read()).hexdigest()
    print(name, "OK" if got == want else f"MISMATCH {got}")
EOF
```

Every line must print `OK`. A mismatch means a truncated download; re-fetch
that file. The hashes in `store.json` were computed on the box that built the
store and re-checked by downloading each file back from the Hub before the
build was declared done.

---

## 3 · The sixteen pressure levels

Column `j` of `temp` and `psal` is the value at pressure `LEVELS[j]`:

```
LEVELS = [10, 30, 50, 100, 150, 200, 300, 400, 500, 700, 900, 1100, 1300, 1500, 1700, 1900]   # dbar
```

These are the sixteen levels of the family-7 `rg100` group, in the same
order, so `temp[:, j]` is the observational counterpart of family-7 channel
`rg_t<LEVELS[j]>` and `psal[:, j]` of `rg_s<LEVELS[j]>`.

A level is **NaN when the profile could not fill it honestly** (§4). This is
common at depth: floats park at ~1,000 dbar and profile to ~2,000 dbar, but
many shallow-cycle profiles stop earlier. The fill histogram is in
`store.json` (`filled_levels_hist_t` / `_s`, index = number of filled
levels): 1,402,053 profiles have all 16 temperature levels; 105,774 have
none (kept because their salinity is good, or vice versa — 308,519 have no
usable salinity). **Treat NaN as "not measured", never as zero**, and treat a
temperature NaN and a salinity NaN independently.

---

## 4 · How the values were made (the QC policy, so you can trust or reject it)

Source: the GDAC daily "geo" files, one NetCDF per basin per day
(`https://data-argo.ifremer.fr/geo/{atlantic_ocean,indian_ocean,pacific_ocean}/YYYY/MM/YYYYMMDD_prof.nc`),
7,671 days × 3 basins, 0 files missing, index date 2026-09-07.

A profile is **kept** only if its position quality flag and its time-stamp
quality flag are both "good" or "probably good" (Argo flags 1 or 2), its
position and time are not fill values, and it has at least two filled levels
in temperature *or* in salinity. Drops from the 3,000,564 profiles in the
day files: 145,455 on time-stamp quality, 60,993 on position quality,
115,677 with too few good levels. 0 duplicates were found across basin
boundaries.

Per sample: for data mode `D` (delayed) or `A` (adjusted), the `*_ADJUSTED`
pressure/temperature/salinity are used with their adjusted flags; for mode
`R` (real-time) the raw variables with their raw flags. A sample counts only
if its value and its pressure both carry flag 1 or 2 and the value is finite.
Temperature and salinity are screened **independently** (a profile can have
good T and bad S).

Interpolation onto the sixteen levels is **linear in pressure between two
good samples that bracket the level**, and only if the gap between those
samples is at most 25 dbar for levels ≤ 200 dbar, 50 dbar for ≤ 500 dbar,
100 dbar above — a 1,500 dbar hole is not allowed to invent the thermocline.
One exception: the 10 dbar level may take the shallowest good sample **as
is** (never extrapolated) when that sample is within 20 dbar of the surface,
because a float's first bin is often at 4–15 dbar. Everything else is NaN.

Values are stored as float16 — a precision of ~0.01 °C and ~0.02 PSU at
oceanic magnitudes, adequate for the purpose (Argo's own accuracy is
~0.002 °C / ~0.01 PSU, so float16 is the coarser of the two; if that
matters for your use, the raw files are at the URL above).

---

## 5 · Opening it (numpy only, no dependency on this repository)

```python
import numpy as np, json, datetime as dt

D = "family8_argo_l0"
cols = {n: np.load(f"{D}/{n}.npy", mmap_mode="r")
        for n in ["bin", "time_days", "lat", "lon", "temp", "psal",
                  "wmo", "cycle", "nlev", "maxpres", "mode"]}
off = np.load(f"{D}/bin_offsets.npy")          # int64 [3143]
N = len(cols["bin"]); assert N == off[-1] == 2_678_439

EPOCH = dt.date(1982, 1, 1)
def bin_of(date):  return (date - EPOCH).days // 5
def date_of(b):    return EPOCH + dt.timedelta(days=5 * int(b))   # the bin's first day

# all profiles in the pentad that contains 2015-01-03
b = bin_of(dt.date(2015, 1, 3))                # 2411
rows = slice(off[b], off[b + 1])
print(b, date_of(b), off[b + 1] - off[b], "profiles")
print(cols["lat"][rows][:3], cols["lon"][rows][:3])
print(cols["temp"][rows][0].astype(np.float32))   # 16 values, NaN where unfilled
```

Sanity checks worth running once: `np.all(np.diff(cols["bin"]) >= 0)` (sorted);
`off[1607] == 0` and `off[1607 + 1] > 0` (the first live bin is 2004-01-01,
bin 1607; bins 0–1606 are empty because the float array did not exist);
`(np.diff(off)[1607:] > 0).all()` (no empty bin after that); `lon.min() >= -180`
and `lon.max() < 180`; `time_days` inside `[5*bin, 5*(bin+1)]` for every row
(closed at the top: 209 rows whose time-stamp is exactly midnight ending the
bin round up to `5*(bin+1)` in float32 — harmless, their age at the bin's
end is 0 days).

`store.json["per_year"]` gives `kept` and `n_prof` per year (36,128 kept in
2004 rising to 160–167 k per year from 2015), useful for checking that a
partial download or a filter did what you expect.

---

## 6 · The search: the k nearest observations, one-sided in time

This is the operation the store exists for. The reference implementation is
`ml/family8_store.py::ArgoStore.knearest` in the repository
(`git clone https://github.com/blauewelt/earth`, pure numpy, no other
dependency), but it is short enough to restate here so you need not clone
anything. Three rules, from the design note:

1. **One-sided in time, as a hard constraint of the search.** Only
   observations at or before the anchor's bin are candidates. The CSR slice
   simply stops at `off[bin + 1]`, so a later observation cannot be returned
   however close it is. A forecaster that has seen next week is not a
   forecaster.
2. **Bounded** by a radius `R_max_km` and a window `T_max_days`. Inside them
   the k nearest are taken; if fewer than k exist, the remaining slots are a
   **miss token** (NaN values and offsets, `valid = False`) — a fixed k per
   anchor so batches are never ragged.
3. **The metric** is space–time: `d² = (Δx² + Δy²) / L² + Δt² / T²`, with
   `L` the correlation length and `T` the decorrelation time; by default
   `L = R_max_km` and `T = T_max_days`.

The age `Δt` is measured back from the **end** of the anchor's bin
(`5 * (bin + 1)` days after the epoch), so an observation inside the anchor's
own bin is 0–5 days old and `T_max_days = 30` sees exactly six bins.
Offsets are in kilometres, east and north, on the small-offset approximation:
`dy = Δlat · 111.32`, `dx = Δlon · 111.32 · cos(mean latitude)` with the
cosine floored at 0.05 near the poles and `Δlon` wrapped into [−180, 180) so
the dateline is not a wall.

```python
KM = 111.32
def knearest(lat0, lon0, b, k=5, R_max_km=1000.0, T_max_days=30.0, L=None, T=None):
    L = L or R_max_km; T = T or T_max_days
    t_anchor = 5.0 * (b + 1)
    b_lo = max(0, b - int(np.ceil(T_max_days / 5.0)))     # oldest bin that can still qualify
    rows = slice(off[b_lo], off[b + 1])                   # one-sided by construction
    lat, lon, t = cols["lat"][rows], cols["lon"][rows], cols["time_days"][rows]
    dt_ = t_anchor - t                                    # >= 0 always
    dy = (lat - lat0) * KM
    dlon = np.mod(lon - lon0 + 180.0, 360.0) - 180.0
    dx = dlon * KM * np.maximum(np.cos(np.radians(0.5 * (lat + lat0))), 0.05)
    dist = np.hypot(dx, dy)
    ok = (dist <= R_max_km) & (dt_ <= T_max_days)
    n_R = int(ok.sum())                                   # the local-density feature
    d2 = np.where(ok, (dist / L) ** 2 + (dt_ / T) ** 2, np.inf)
    order = np.argsort(d2)[:min(k, n_R)]
    idx = np.arange(off[b_lo], off[b + 1])[order]
    out = dict(n_R=n_R, n_found=len(idx), row=idx,
               temp=cols["temp"][idx].astype(np.float32), psal=cols["psal"][idx].astype(np.float32),
               dx_km=dx[order], dy_km=dy[order], dt_days=dt_[order], dist_km=dist[order],
               wmo=cols["wmo"][idx], cycle=cols["cycle"][idx], mode=cols["mode"][idx])
    return out   # pad to k slots with NaN / valid=False if you need a fixed shape

r = knearest(36.0, -70.0, bin_of(dt.date(2015, 1, 3)))   # the Gulf Stream, first pentad of 2015
print(r["n_R"], r["dist_km"].round(), r["dt_days"].round(1))
```

**What the numbers look like** (measured on the store's own index): within a
30-day window the nearest profile to a random Argo-sampled point is on
average 34 km away, and the 3rd / 5th / 7th nearest are at 114 / 150 / 200
km; from a point drawn uniformly over the sphere (land included) they are
514 / 572 / 617 km. The design choice this supports is `k = 5, T_max = 30 d,
R_max = 1,000 km` — the fifth neighbour sits inside a surface correlation
length (~200 km) where the array is dense — and `n_R` is what tells a model
which regime an anchor is in. Before 2004 every search returns `n_R = 0`,
which is the true state of the observing system, not a gap to paper over.

**The token, per returned observation** (§7.1 gives the concrete encoding):
`(value vector [32: 16 T + 16 S], Δx_km, Δy_km, Δt_days, n_R, log2_fp, log2_dt)`.
The last two are the *footprint* — the spatial and temporal support of the
measurement in log units of a 0.25° cell (27.83 km) and of a pentad (5 days).
For an Argo profile both are constants at the clamp floor, **−4** (a point
measurement taken over minutes); they exist so that gridded sources
(0.25° satellite cells, ~1.9° reanalysis, monthly means) can share the same
token type with their own footprint stated. If you only ever ingest Argo, you
may drop them.

---

## 7 · How to use it in training

This section is the point of the file. The store is an archive; the model
sees it only through the gather below, and the gather is where every
correctness property lives.

### 7.1 · The unit of data: an anchor and its k nearest profiles

Training examples are built per **anchor** `(lat, lon, bin)` — a place and a
five-day period. For each anchor, run the §6 search once per sparse channel
family (here: Argo) and turn each returned observation into one token:

```
token_i = concat( value_i[32],            # 16 T then 16 S, standardised, NaN -> 0 with a mask
                  mask_i[32],             # 1 where the level is measured, 0 where NaN
                  dx_km_i / 1000, dy_km_i / 1000,   # offsets, scaled to O(1)
                  dt_days_i / 30,                   # age, scaled to O(1)
                  log(1 + n_R),                     # local density (same for all k slots)
                  log2_fp_i, log2_dt_i )            # footprint constants, -4 and -4 for Argo
```

and, when fewer than k profiles are found, fill the remaining slots with a
**miss token** — all zeros plus a `valid = 0` flag that the attention mask
reads. `k` is fixed per channel family (5 for Argo, chosen so the fifth
neighbour typically lies within one surface correlation length where the
array is dense); a fixed k is what keeps batches rectangular.

Do not round a profile onto the anchor's grid cell. The whole information
content of family 8 over the gridded product is that the model sees *where*
and *when* the measurement was made relative to the anchor; the offsets are
features, not an inconvenience to be interpolated away.

### 7.2 · Standardise in the loader, from training years only

The store is raw °C and PSU. Compute per-level mean and standard deviation
of `temp[:, j]` and `psal[:, j]` **over rows whose bin lies in the training
years only** (ignoring NaN), and apply them in the loader. Salinity varies
by ~1 PSU where temperature varies by ~10 °C; unstandardised, the model
learns the units. If you also use the family-7 tensor, its `norm_rg100`
array (32 pairs of mean, sd in the order `rg_t10 … rg_t1900, rg_s10 …
rg_s1900`) was computed on the gridded version of the same quantities and
is a serviceable substitute for a first pass.

Anomalies (value minus a seasonal climatology) help a forecaster because the
seasonal cycle is the largest and least interesting signal. If you compute
one, evaluate the climatology at the **observation's own latitude, longitude
and day of year**, not the anchor's — the anomaly belongs to where the
measurement was made — and fit it on training years only. A per-level
harmonic fit (annual + semi-annual sine/cosine per 1°–2° cell) is enough.

### 7.3 · The two leaks, and which one the store prevents

**The future leak** is prevented by construction: the search reads rows up
to `bin_offsets[bin + 1]` and no further, so no observation later than the
anchor's own pentad can ever appear in its input. Keep it that way — do not
"look ahead one bin to fill gaps".

**The held-out-year leak is the loader's responsibility.** Whatever years
are held out for testing (this project's terminal protocol trains on
≤ 2020 and tests on 2021–2024), an anchor from a training year whose search
window reaches into a held-out year must not receive those profiles, and no
standardisation or climatology constant may be fitted on them. The search
above has no notion of held-out years; filter its output by `time_days`, or
build the training loader over a store view whose rows are restricted to
training bins.

### 7.4 · Missingness is a feature, not a defect

Three distinct kinds, and a model should be able to tell them apart:

- **No profile in range** — `n_R = 0`, all k slots are miss tokens. True of
  every anchor before 2004 and of most of the Southern Ocean in any year.
  This is the honest state of the observing system; do not fill it, and do
  not drop the anchor — a forecaster deployed today will meet it constantly.
- **Fewer than k in range** — some slots are miss tokens; `n_R` says how
  many were there. Common in marginal seas and near ice.
- **A level not measured in an otherwise present profile** — NaN inside
  `value_i`, carried by `mask_i`. Common below 1,000 dbar and for salinity
  (§8). Never average across the NaN, never set it to zero without the mask.

Density (`n_R`) is the feature that lets the model calibrate its confidence
by region and era: an anchor with two floats within 1,000 km is not the same
input as one with forty, even if the five nearest look alike.

### 7.5 · Objectives that use the store well

Two objectives come naturally from the structure, and both are cheap
because the targets are already in the store:

- **Masked-profile reconstruction.** Withhold one of the k profiles (the
  nearest is the hardest and most useful choice) and predict its 32 values
  from the other k − 1 plus whatever dense context the model has
  (sea-surface temperature, sea-surface height, currents on the family-7
  grid). Score per level, in °C and PSU, on the levels the withheld profile
  actually measured. This teaches the model to interpolate the interior
  from displaced measurements, which is exactly what it must do at inference
  time.
- **Next-pentad prediction.** With the anchor at bin `b`, the target is the
  nearest profile at bins `b + h` (a search run *from* `b + h` with its own
  one-sided window, then restricted to `dt ≤ 5h` so it is genuinely later
  than the anchor). This is the forecast objective; the same per-level score
  applies.

Compare either against two baselines before believing a number:
**predict-the-mean** (the standardised zero) and **nearest-profile
persistence** (copy the nearest profile's values as the prediction). A
model that does not beat both has not used the displacement information.

### 7.6 · Using it beside the family-7 tensor

The store and the dense tensor share the time axis (five-day bins from
1982-01-01; family 7 holds bins 0–3141, the store's rows lie in bins
1607–3141) and the level list (§3). A family-7 anchor `(y, x, bin)` on the
0.25° grid has latitude `−90 + 0.25·y` and longitude `−180 + 0.25·x`
(721 × 1440 cells); pass those to the search with the same `bin`. Family 7's
`rg100` group is the *mapped* version of the same measurements — a
reasonable baseline, and redundant as an input once the store is used.

---

## 8 · Known limits and gotchas

- **Coverage is the observing system's, not a gap.** The Southern Ocean,
  marginal seas and ice-covered regions are thin; pre-2007 the array was
  still growing (36 k profiles in 2004 against 160 k+ from 2015). `n_R` is
  the honest measure — do not fill.
- **Salinity is missing far more often than temperature** (308,519 profiles
  with no usable salinity level versus 105,774 with no temperature) because
  conductivity sensors drift and are flagged. Model T and S with separate
  missingness.
- **`float16` values and `int16` bins/cycles**: cast to float32/int64 before
  arithmetic; `temp.astype(np.float32)` is enough. Cycle numbers fit int16
  (Argo cycles rarely exceed ~400).
- **`lon` is in [−180, 180)**, the family-7 grid too; some Argo sources use
  [0, 360). Wrap before differencing (the snippet above does).
- **The same float can appear twice in one bin** (a float cycling every ~5
  days, or ascending and descending profiles from one cycle). That is real
  data, not a duplicate; true duplicates (same WMO, cycle and direction in two
  day files) were removed at build time and there were none.
- **The store ends 2024-12-31** with the family-7 tensor. Newer profiles are
  at the GDAC URL in §4; the builder (`ml/build_family8_argo.py`, stages
  `index | profiles | publish`, resumable per year) extends it with
  `--start 2025-01-01` under a new stem.
- **Nothing here is anomalised or standardised** (§7). A model fed raw PSU
  (~34–37) next to raw °C (−2–30) without scaling will learn the units, not
  the ocean.

---

## 9 · Provenance

Built 2026-09-07 19:21Z by `ml/build_family8_argo.py` at repository commit
`30aafcc` (`family8-build #3`, a rented RTX 4090 box, 70 minutes for 185.8 GB
of source files streamed at 44 MB/s and deleted as they were parsed; the
store itself is the only thing kept). Source archive: Argo GDAC daily geo
files via the Ifremer host and its AWS S3 mirror (`argo-gdac-sandbox`,
eu-west-3), index `ar_index_global_prof.txt.gz` dated 2026-09-07 17:03Z
(3,366,590 profiles listed, of which 35,824 carry no position and years
before 2004 are not in the store). Design and measurements:
[E-076 · family 8](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E076_family8_nearest_observations.md)
(§2 the token and its rules, §2.6 the footprint fields, §3.1 the measured
index); build record:
[EXPERIMENTS.md#e-076](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-076).
Argo data are free and open under the Argo data policy; cite
"These data were collected and made freely available by the International
Argo Program and the national programs that contribute to it
(https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program is part
of the Global Ocean Observing System."
