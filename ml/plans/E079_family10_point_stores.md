# E-079 · Family 10, first data pull — four point-and-track stores and the registry

*Written 2026-09-13. Chris: "let's pull the family 10 data." This is the build
spec for the first tier-P stores of family 10 beyond Argo, and for the
registry that makes family 7.1, family 8 and these stores one family. Every
rule is a build assertion or a stated decision; where a source's file layout
has not yet been seen from this programme it says so, and the builder lists
the archive before it fetches.*

**What was done before, and what was not.** The sources below were ranked in
`ml/figures/geofm_survey/CONE_DATA_AND_ENSO.md` Part 3 (drifters, the
biosphere, SOCAT) and §4.5 (TAO/TRITON, altimetry, daily winds) and named in
E-076 §4 as "the door to moorings, altimeter tracks and ship data". They are
catalogued (`docs/CATALOG.md` §2.8). None has been fetched, parsed or stored:
the only observation store this programme has built is family 8's Argo
profiles (`ml/build_family8_argo.py`, 2026-09-07). This plan builds the next
four the same way.

## 1 · Why these four, against the four goals

Chris tracks four prediction goals: El Niño, the ocean carbon sink, the
Atlantic overturning (AMOC), and sea-surface temperature. The first pull is
the set of *observations* — not gridded products of them — that serve those
goals and can be reached with credentials the programme has:

| store | what it is | El Niño | carbon | AMOC | SST | access |
|---|---|---|---|---|---|---|
| `gdp` | Global Drifter Program: surface buoys, 15 m velocity and sea-surface temperature, 6-hourly, 1979→ | equatorial currents at onset | surface transport | **direct velocity** (the quantity the cone codec could not learn from grids, E-069) | an independent SST instrument | keyless (NOAA AOML / NCEI) |
| `gtmba` | Global Tropical Moored Buoy Array (TAO/TRITON in the Pacific, PIRATA in the Atlantic, RAMA in the Indian Ocean): daily subsurface temperature to 500 m, SST, salinity, winds, currents, 1980→ | **the warm-water volume and thermocline depth** — the six-month lead | — | PIRATA: tropical Atlantic | equatorial SST | keyless (NOAA PMEL) |
| `socat` | Surface Ocean CO₂ Atlas: ship and mooring measurements of surface-water CO₂ fugacity with SST and salinity, 1957→, tens of millions of rows | — | **the only observation of the quantity itself**; everything else is a reconstruction | — | ship SST | keyless (SOCAT / NCEI, CC-BY) |
| `slatrack` | along-track sea-level anomaly from every altimeter mission, ~7 km along track, 1993→, with SWOT swath samples 2023→ | Kelvin waves in sea level | — | geostrophic transport as measured, not modelled | fronts | Copernicus Marine (credentials held) |

Scatterometer winds, 5 km foundation SST and native-resolution ocean colour
are the next pull, as tier-G groups at their native grids (E-078 §3); they
are not point data and do not belong in this builder.

## 2 · The store — one schema for every tier-P source

Family 8's layout, generalised. A store is a directory
`tensors/family10/<store>/` on `chfrank/earth-tensors` with:

| file | dtype · shape | meaning |
|---|---|---|
| `bin.npy` | int16 [N] | five-day bin, `floor((time − 1982-01-01) / 5 d)`; observations before 1982 are **kept with negative bins** (drifters from 1979, SOCAT from 1957) so the store is the whole record — the reader clips to the tensor's axis |
| `time_days.npy` | float32 [N] | days since 1982-01-01 00:00 UTC, fractional (negative before the epoch) |
| `lat.npy`, `lon.npy` | float32 [N] | degrees; lon in [−180, 180) |
| `values.npy` | float16 [N, C] | the channels of §3, raw units, NaN = not measured |
| `platform.npy` | int64 [N] | the platform id (drifter WMO/AOML id, mooring site code hashed, SOCAT expocode hashed, satellite mission code) |
| `qc.npy` | uint8 [N] | the source's own quality flag, mapped to 0 = not assessed, 1 = good, 2 = probably good, 3+ = the source's worse grades; the builder keeps only ≤ 2 by default and records the drops |
| `fp.npy` | float16 [N, 2] | **per-row footprint** `(log2_fp, log2_dt)` from E-078 §2 — constant within a source (§3 gives the values) but stored per row so every store reads identically |
| `bin_offsets.npy` | int64 [B + 1] | CSR offsets over the bin range the store covers; `store.json` carries `bin_first` |
| `store.json` | JSON | schema (channel names, units), footprint constants, QC policy, provenance (URLs, index dates, builder commit), per-year counts, and **sha256 of every file above** |

Rows sorted by `(bin, time_days)`. The reader is `ml/family10_store.py`:
`Store.open(dir)`, `Store.knearest(lat, lon, bin, k, R_max_km, T_max_days)`
— family 8's search, one-sided in time, bounded, fixed k with miss tokens,
returning the §2 token fields of E-078 (values, mask, Δx, Δy, Δt, n_R,
log2_fp, log2_dt, platform). Family 8's Argo store is readable through the
same class unchanged (its `temp`/`psal` columns are two value blocks; the
reader concatenates them; no rebuild).

**The registry.** `tensors/family10/family10.json` lists every group of the
family: the four tier-G groups of family 7.1 (by reference to its manifest
— not copied), the Argo store (tier P), and the four stores of this plan;
per group: `tier`, `layout`, `cadence`, `channels` with units, the
footprint constants, `bin_first`, files with sha256, sources and builder
commit. A consumer dispatches on `tier`. This is E-078b's registry; the
sampler-side dispatch is a later change and is not in this build.

## 3 · The four stores

### 3.1 `gdp` — surface drifters

Source: the Global Drifter Program's quality-controlled **6-hourly
interpolated** product (positions and velocities interpolated to 00/06/12/18
UTC by the GDP's kriging, with error estimates), 1979-02→, from NOAA
AOML/NCEI. Access path to be verified by listing (the ERDDAP `tabledap`
service at `erddap.aoml.noaa.gov/gdp/` and the NCEI archive are the two
candidates; the builder records which answered). Channels (C = 4): `u`, `v`
(m s⁻¹, 15 m drogue-depth velocity), `sst` (°C), `drogue` (1 if the drogue
was still attached — a drogued and an undrogued drifter measure different
things; the flag is a channel so the model can tell them apart). Footprint:
`log2_fp = −4` (a point), `log2_dt = log2(0.25/5) = −4.3` (a 6-hour
sample). Expected N: a few × 10⁷ rows over 45 years, ~1–2 GB. QC: the GDP's
own flags; rows with position error > 50 km dropped and counted.

### 3.2 `gtmba` — the tropical moored arrays

Source: NOAA PMEL's Global Tropical Moored Buoy Array, **daily** averages
per site (the daily product is what the array is designed to deliver; the
high-resolution files add nothing at pentad cadence). Sites: every TAO/TRITON,
PIRATA and RAMA mooring with a record. Access to be verified (PMEL's data
delivery service and its OceanSITES NetCDF mirror are the candidates).
Channels (C = 12 + depth levels): `sst`, `sss`, `airt`, `wind_u`, `wind_v`,
`u_cur`, `v_cur` (where an ADCP or current meter exists), and subsurface
temperature at the array's standard depths **1, 20, 40, 60, 80, 100, 120,
140, 180, 300, 500 m** (NaN where a site does not carry the level). Footprint:
`log2_fp = −4`, `log2_dt = log2(1/5) = −2.3`. One row per site per day; N ≈
70 sites × 40 y × 365 ≈ 10⁶ — tiny. This store also yields the **warm-water
volume** proxy the El Niño read-out needs, computed by the consumer from the
20 °C isotherm depth at the equatorial sites, not stored.

### 3.3 `socat` — surface CO₂

Source: the latest SOCAT synthesis release (v2025 or, if released, v2026;
the builder reads the release page and records the version), the global
synthesis file (tab-separated, tens of millions of rows). Channels (C = 4):
`fco2` (µatm, the recomputed fugacity `fCO2rec`), `sst` (°C), `sss` (PSU),
`patm` (hPa, where present). QC: SOCAT flags A–D kept (the synthesis file's
inclusion criterion), E dropped; the WOCE flag column kept where present.
Footprint: `log2_fp = −4`, `log2_dt = −4` (an underway measurement, minutes).
N ≈ 4 × 10⁷, ~1 GB. Pre-1982 rows kept with negative bins.

### 3.4 `slatrack` — along-track sea-level anomaly

Source: Copernicus Marine `SEALEVEL_GLO_PHY_L3_MY_008_062` (reprocessed
level-3 along-track, all missions, 1 Hz ≈ 7 km) and its SWOT counterpart
where available, via the `copernicusmarine` toolbox with the credentials in
`claude/copernicus-marine-access.md` as environment variables. Channels
(C = 3): `sla` (m, filtered), `sla_unfiltered` (m), `mdt` (m, the mean
dynamic topography at the point, so `adt = sla + mdt` is one addition).
Footprint: `log2_fp = log2(7/27.83) = −2.0`, `log2_dt = −4`. N ≈ 3 × 10⁹
over 32 years — **tens of GB**; this store is built per year, published per
year, and needs the box, not a hosted runner. Everything else in this plan
runs on a hosted runner.

## 4 · Build mechanics

- One builder, `ml/build_family10_stores.py`, with a source adapter per
  store (`--store gdp|gtmba|socat|slatrack`), stages `index | fetch | publish`
  as in the family-8 builder: `index` lists the archive and writes the plan
  of files/years; `fetch` streams year by year, parses, filters, appends to
  per-year column parts under `<work>/<store>/parts/<year>/` and marks each
  year done (flush, then mark); `publish` concatenates parts in bin order,
  writes CSR offsets, computes hashes, uploads to the Hub, downloads each
  file back and re-hashes (a publish that cannot verify fails the job).
- **List, then fetch; preflight one real file** per store before spending
  anything. The sandbox this plan was written in can reach the NOAA, PMEL and
  SOCAT hosts (measured 2026-09-13), so parsers are verified on real samples
  before the workflow runs.
- **A hosted-runner workflow**, `family10-build.yml`, `workflow_dispatch`
  only, `runs-on: ubuntu-latest` by default for `gdp`, `gtmba`, `socat`
  (each fits in 6 h and 14 GB when streamed; the per-year parts are the
  resume points if not), with a `runner` input so `slatrack` can be sent to a
  Vast box. HF_TOKEN from the repo secret as an environment variable, never
  argv. Never add any trigger but `workflow_dispatch`.
- **Assertions before a store is trusted:** rows sorted by (bin, time);
  offsets consistent with `bin`; lon in [−180, 180); every value finite or
  NaN, none outside physical bounds (|u|,|v| < 5 m s⁻¹, −3 < SST < 45 °C,
  0 < fCO₂ < 2000 µatm, |sla| < 3 m); per-year counts within 30 % of the
  source's own published counts where the source publishes them; the
  footprint columns constant per store; the nearest-neighbour search on one
  known anchor (36° N, 70° W, bin 2411) returns a plausible drifter set; and
  resumability across a simulated interruption.
- **Holdout:** unchanged (2009, 2017, 2023 development years; train ≤ 2020,
  test 2021–2024). The stores carry the whole record; the loader restricts.

## 5 · Cost

`gdp`, `gtmba`, `socat`: hosted runners, $0, under an hour each once the
parsers are right. `slatrack`: one box-day, ≈ $8 at $0.33/h, tens of GB on
the Hub. Registry: minutes.

## 6 · What decides whether it was worth it

Per goal, one number each, on the cone codec of E-076a's twin with and
without the new tokens, same seeds: El Niño — Niño-3.4 at 6-month lead
against the persistence-of-warm-water-volume null; carbon — held-out SOCAT
fCO₂ reconstruction against the SeaFlux climatological null; AMOC — the
held-out per-family loss on velocity with drifter tokens vs. without (the
E-069 H1 question, now answerable against observed velocity); SST — the
held-out OSTIA-cell loss with drifter SST tokens vs. without. Falsifier for
each: no change inside the tier's replicate band.

<a id="10-1-integer-seconds"></a>
## 10.1 — integer seconds

Chris, 2026-09-15: family 10's tier-P stores store time as **integer seconds**,
not float32 days. The rebuilt stores are **family 10.1**. Family 10 — the four
stores now published under `tensors/family10/` — stays published and untouched
until 10.1 is built, verified and handed over.

### Why

The published stores carry `time_days.npy`, float32 days since 1982-01-01.
float32 has a 24-bit mantissa, so the spacing between representable values is
proportional to the magnitude: at the start of the altimeter record (1993,
≈ 4,000 days) it is **21 seconds**, and at the end (2024, ≈ 15,700 days) it is
**84 seconds**. That was known when the format was chosen and was judged
harmless, because the stores it was chosen for sample every six hours or every
day.

`slatrack` samples at **1 Hz**. The independent verification of 2026-09-14
([docs/FAMILY10_VERIFICATION_2026-09-14.md](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY10_VERIFICATION_2026-09-14.md)
§9.7, §10) measured the consequence on the published store: consecutive
along-track samples share a timestamp, **up to 316 rows to one value** — at
6.5 km between samples, 550 km of ground track with a single time on it. The
column cannot order the rows it stores. Nothing in the store is *wrong*: the
rows are in the right order, the bins are right, and `bin == floor(time_days /
5)` holds on all 2,030,800,150 rows — the column simply cannot express the
distinction the archive published, so a consumer that wants "which sample came
first" has to be told to use the row order and not the timestamp. A format that
has to be accompanied by that instruction is the wrong format.

An integer second is the finest thing any of the four archives reports. It is
exact everywhere, it costs the same four bytes float32 did, and it removes the
float rounding from the bin derivation at the same time.

### What changes

- **Schema 2.** `time_days.npy` (float32 days) → **`time_s.npy`, int32 seconds
  since 1982-01-01T00:00:00Z**, negative before the epoch. int32 spans
  1913-12-13T20:45:52Z .. **2050-01-19T03:14:07Z**; the builder refuses a row
  past it rather than letting it wrap into 1913, and widening to int64 (twice
  the column) is the change to make when the archives get there.
- **`bin = floor(time_s / 432000)`**, integer `floor_divide`, no float anywhere
  in the derivation. Under v1 the builder had to compute the bin from the
  float32 value it was about to *store* rather than from the float64 it had
  parsed, because a timestamp a microsecond before a pentad boundary could
  round up across it; that whole class of fault is gone rather than guarded.
- **`store.json`** gains `schema_version: 2` and `family_version: "10.1"`.
- **The reader reads BOTH.** `ml/family10_store.py` opens a schema-1 store
  (family 8's Argo store, family 10's published four) by converting its days to
  seconds on the fly — recovering nothing, only putting the two in one unit —
  so the family-10 verification tooling and every existing consumer keep
  working. `dt` out of `knearest` is float64 **days** under both.
- **Prefixes.** `tensors/family10_1/` for the stores and the registry
  (`tensors/family10_1/family10.json`, `family_version` "10.1",
  `schema_version` 2), `partials/family10_1/` for the slatrack lanes,
  `ml/cache/family10_1` for the build directories. One constant,
  `family10_store.FAMILY_VERSION`, derives all of them.
- **A v1 column part cannot be upgraded, so every store is rebuilt from the
  source.** The lanes park `.npz` column parts on the Hub and a box assembles
  them, so the two halves of a slatrack build can be a schema apart — and
  `done.json` records names, bytes and sha256, nothing about the layout inside
  a part. A v1 part therefore verifies perfectly and is unusable: multiplying
  float32 days by 86,400 produces an integer column that *looks* exact and is
  wrong by up to 84 s, which is the single outcome this change exists to
  prevent. `family10_parts_hub` refuses a `time_days` part on push and on pull,
  and the assembler refuses one before either assembler reads it. The fresh
  `partials/family10_1/` prefix and the fresh `ml/cache/family10_1` work
  directory are what stop a resume from finding one.

### What does NOT change

The **bins** (same rule, same numbers, same `bin_first` per store), the
**footprints** (`fp.npy`, same constants), the **channels** (names, order,
units, physical bounds), the **QC policy** (same flags, same `qc_keep_max`),
the **platform-id rule**, the **[−180, 180) longitude invariant and its
float32-cast lesson**, the CSR index, the nine-array layout, the streaming
assembler's byte-identity claim, and **tier G** (the registry still points at
the same `f7l2` family-7.1 manifest, by reference). Family 8's Argo store joins
family 10.1 unchanged and is still schema 1; the registry states each group's
own `schema_version` so a consumer can see which groups have the seconds.

### The falsifier

Only the time column's *precision* changes. No row is added, dropped or moved
by it: the QC clauses, the bounds, the window and the bin rule are all
untouched, and the bin of a timestamp is the same integer whether it is derived
from exact seconds or from the float32 days that timestamp rounded to (the one
case where they could differ — a row within 84 s of a pentad boundary — was
already forced to agree by v1's cast-then-bin rule).

**So each 10.1 store's `N` and its per-year row counts must equal v1's
exactly.** A difference is a bug in the rebuild, not an improvement, and it
must be explained before the store is published. v1's numbers, from the
published `store.json` of each:

| store | v1 `N` | v1 `bin_first` | C |
|---|---|---|---|
| `gdp` | 48,480,798 | −211 | 4 |
| `gtmba` | 1,001,282 | −304 | 18 |
| `socat` | 41,830,675 | −1768 | 4 |
| `slatrack` | **2,030,800,150** | 803 | 3 |

Per-year counts: v1's `per_year` block, store by store, must reproduce entry
for entry. One documented exception is *expected to disappear*, not to persist:
v1's `socat` per-year counts recomputed **from `time_days`** differ from the
builder's by ≤ 3 rows/year at year boundaries (§8.5 of the verification), which
is precisely the float32 artefact — 10.1's counts recomputed from `time_s` must
match its `per_year` block **exactly**, in every year, for every store.

And the measurement that prompted this: `slatrack`'s largest group of rows
sharing one timestamp must fall from **316** to **1** wherever the archive
published distinct seconds.
