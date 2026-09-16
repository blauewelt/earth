# E-081 · Family 10.2 — the fishing fleet as an observation store, and its two globe layers

*Written 2026-09-16. Chris: "add what you propose to family 10.2 (and build the
family). At the same time make sure to display the data on blauewelt.org. I'm
also curious where the loitering ships are — let's add a layer." This is the
build spec for the fifth tier-P store of family 10 and for the two globe
layers that show the fleet, written before anything is fetched.*

**What family 10.2 is.** Family 10.1 (E-079 §10.1) is the four observation
stores — surface drifters, the tropical moored arrays, ship CO₂ tracks and
altimeter along-track sea level — with time in integer seconds. Family 10.2
adds one store, `fishing`, and changes nothing else: the four 10.1 stores are
inherited by reference exactly as tier G inherits family 7.2, and the 10.2
registry lists five tier-P groups (plus Argo) with each group's own Hub path.
No 10.1 byte is rebuilt.

## 1 · The source

Global Fishing Watch, *Global AIS-based Apparent Fishing Effort Dataset*,
version 3.0 (2025-03-11), Zenodo record 14982712, licence **CC BY-NC 4.0**
(non-commercial, attribution required; redistribution permitted with
attribution — the store's `store.json` and the data handover carry the
recommended citation and the licence, the globe layer carries "Powered by
Global Fishing Watch"). No credentials: Zenodo serves the files anonymously.

The table this store is built from is **`mmsi-daily-csvs-10-v3-<year>.zip`,
2012–2024** — one row per (day, 0.1° cell, vessel), columns `date`,
`cell_ll_lat`, `cell_ll_lon` (the cell's lower-left corner), `mmsi`, `hours`
(hours the vessel was broadcasting inside the cell that day) and
`fishing_hours` (the part of those hours a neural network classed as
fishing). 13 zips, 5.4 GB in total (0.06 GB for 2012, 0.75 GB for 2024).
Beside it, `fishing-vessels-v3.csv` (0.11 GB) gives each MMSI its year, flag
(`flag_gfw`), gear class (`vessel_class_gfw`), length, engine power and
tonnage. The 0.01° daily table by flag and gear (21 GB) is NOT taken: it has
no vessel identity, and 0.1° is already four times finer than the family-7
grid.

Known issues the source states (verbatim in `README-known-issues-v3.txt`):
2024 is provisional; AIS reception is uneven in space and time, so absence of
effort is not absence of fishing; MMSI is not always one vessel (spoofing,
reflagging, recycling); one gear class per MMSI over the whole record.

## 2 · The store

`tensors/family10_2/fishing/`, the tier-P schema of `ml/family10_store.py`
(schema version 2, nine columns, CSR `bin_offsets`, `[-180, 180)` longitude),
built by a `FishingAdapter` in `ml/build_family10_stores.py` next to the four
existing adapters. Row = one (day, cell, vessel):

| column | value |
|---|---|
| `time_s` | the day at 00:00:00 UTC in seconds since 1982-01-01T00:00Z (int32). The source is daily; every row of a day carries the same second and the pentad bin is `floor_divide(time_s, 432000)` as everywhere else |
| `lat`, `lon` | the cell **centre**: lower-left corner + 0.05°, longitude wrapped to `[-180, 180)` after the float32 cast (the fb5d5ab rule) |
| `values` (C = 2) | `fishing_hours`, `hours` — float16. The builder asserts `0 ≤ fishing_hours ≤ hours` and refuses otherwise. **Measured on the 2012 zip before the build: `hours` exceeds 24 in 1.6 % of rows (to 47.6 h; 2024 reaches 48.0 h)** — the source's own first known issue, one MMSI broadcast by more than one vessel — so the ceiling is not 24 but a `--max-hours` sanity bound (default 168 h; a misread longitude still trips it), and rows over 24 h are counted in `store.json` (`hours_over_24h`, `max_hours`) rather than refused |
| `platform` | `platform_hash(mmsi)` — the same hash the other stores use, so a vessel is one id across days |
| `qc` | the vessel's **gear class** as a small integer from a code table written into `store.json` (`qc_codes`: e.g. 1 trawlers, 2 drifting longlines, 3 purse seines, 4 squid jiggers, … 0 unknown) — a categorical channel a consumer may use or ignore |
| `fp` | footprint: `log2_fp = log2(11.1 km / 27.83 km)` for a 0.1° cell, `log2_dt = log2(1 day / 5 days)` — the same convention as E-079 §2 |

`store.json` additionally carries `source` (the Zenodo DOI, the version, the
citation), `licence: "CC BY-NC 4.0"`, `attribution`, `qc_codes`, the
per-year row counts, `hours_total` and `fishing_hours_total` (float64 sums
over the whole store), and the vessel table's summary (vessels per year, gear
class histogram). The vessel table itself is published beside the store as
`vessels.csv.gz` — small, and the only way a consumer can turn a platform
hash back into a flag or a length.

**Build path.** A rented box, not a hosted runner (measured: 0.1116 rows per byte of zip → ≈ 596 M rows, ≈ 18.5 GB of store plus ≈ 16 GB of parts = ≈ 35 GB while building; a hosted runner has ~14 GB). `family10-build.yml`, `store: fishing`, no credentials: each year's zip is streamed through `zipfile` without
extraction, parsed in chunks, packed per year into parts under the existing
`parts/<year>/` layout, then assembled and published exactly like `gdp`
and `socat` (streaming assembly; the store is about a quarter of `slatrack`). Every year is marked only when its zip's row count equals
the rows read (the audit rules of 7e6b14f apply unchanged: a short download
or a year with zero rows is an absence, never a gap).

**The registry.** `FAMILY_VERSION = "10.2"` in `ml/family10_store.py`;
`HF_ROOT`/`HF_PARTIALS` follow it (`tensors/family10_2`, `partials/family10_2`).
`ml/build_family10_registry.py` gains a per-store root: the four 10.1 stores
are read from `tensors/family10_1/<store>/store.json` and listed with that
`path`; `fishing` from `tensors/family10_2/fishing/`. The registry is written
to `tensors/family10_2/family10.json` with `family_version: "10.2"`,
`inherits: {"10.1": [...]}` and, per group, its `path` and `schema_version`.
The store-side check that refuses a 10.0-schema part stays.

## 3 · The gridded product for the globe

The globe cannot range-read a point store. Beside the store the builder
writes **`tensors/family10_2/fishing_grid/fishing_grid_monthly_025.npy`**:
shape `[156, 721, 1440, 2]` **float32** (planned as float16 — measured before the build: the largest 0.25° cell-month sum is 595,726 vessel-hours in 2024 against float16's 65,504, so the fleet's own cells would overflow; months 2012-01 … 2024-12; the family-7
0.25° grid, latitude −90 → +90, longitude −180 → +180; channels
`fishing_hours`, `hours`), each cell the SUM of the store's rows whose cell
centre falls in it that month, month-major in C order so one month of one
channel pair is one contiguous range read (721 × 1440 × 2 × 4 = 8.3 MB, still under family 7's 14.5 MB).
Zero is a real value (no broadcasting fishing vessel) and is stored as 0, not
NaN. `ml/publish_family7_index.py`'s pattern writes
`data/fishing_index.json` — header length, shape, dtype, the month list, the
grid geometry, the channel labels and units, the measured CORS headers — so
`src/app.js` contains none of the arithmetic (root `CLAUDE.md` §3, the
family-7 rule). The grid's column sums must equal the store's totals to
float16 accumulation error, and the builder asserts it.

## 4 · The two globe layers

**(a) "Fishing effort (AIS), 0.25° monthly" — timed raster from the Hub.**
Reads one month by range from the grid above through the same slab reader
and LRU the family-7 layer uses (E-070), paints `log10(1 + fishing_hours)`
on a sequential ramp with zero transparent, snaps the date to the month
(`monthlyGrid` semantics, §4b), and the probe/pixel card print the two
sums in hours with the month stamp. Complete per root `CLAUDE.md` §2: title
with the pixel size, hover card (gist: what AIS is, what "apparent fishing"
means, the coverage caveat), legend, probe, aggregation posture `✗/✗` with
the reason (a monthly sum already; and a range read per month — the same
byte-count argument as the family-7 row), catalog record with the CC BY-NC
licence, chip, tests against a git-committed decimated fixture, provenance
stamp = the month.

**(b) "Loitering vessels (last 30 days)" — points from a baked snapshot.**
Loitering events (a vessel drifting at sea for hours at low speed, the
signature of transshipment at sea and of waiting) come only from the
Global Fishing Watch Events API, which needs a token — so they are NEVER
fetched by the browser (§3: no keyed host). A scheduled hosted workflow
`.github/workflows/refresh-loitering.yml` (daily, `ubuntu-latest`, secret
`GFW_API_TOKEN`) pages `GET /v3/events?datasets[0]=public-global-loitering-events:latest`
for the last 30 days, and bakes `data/loitering.json`: per event the
position, start, end, duration in hours, median speed, distance from shore,
vessel name, flag and MMSI, plus `fetched_at`, the window, the count and the
attribution string. Until the secret exists the workflow refuses with a clear
message and the layer shows the committed fixture. The layer draws one point
per event (colour by duration), shows events whose interval overlaps the
selected date and toasts when the date is outside the snapshot's window
(§4b — date-driven, not dateless), hover card, catalog record (CC BY-NC,
"Powered by Global Fishing Watch"), chip, tests. The footer carries the
attribution line the licence requires.

## 5 · Hypothesis, falsifier, cost

*Hypothesis.* The fleet is a distributed sensor of the ocean: apparent
fishing effort follows fronts, upwelling and productivity, so it is both a
read-out target for "predict everything" and a weak extra input. This plan
builds the store; the test of the hypothesis is E-078a's ablation with
`fishing` tokens in and out, not this build.

*Falsifier for the build.* (1) For every year, `N_year` equals the number of
data rows in that year's zip; (2) `fishing_hours_total` is RECORDED, not gated — the 2012 zip alone holds 7.7 M fishing hours and the whole record projects to ≈ 0.8 billion, so the "nearly 370 million hours" in Global Fishing Watch's release note is not this table's sum (it is presumably the fleet table's or a different accounting) and cannot serve as a band; the RESULT states both numbers; (3) every
`fishing_hours ≤ hours`; (4) the monthly grid's global sums equal the store's
per-month sums; (5) the registry lists nine tier-P/G groups plus `fishing`
and every 10.1 sha256 is unchanged.

*Cost.* One verified box (`ml/CLAUDE.md` §7: read `inet_up` first): ~20 min of parsing plus 5.3 GB down and ~20 GB up; probe of 2012 on a hosted runner first, free. Globe work: one
frontend session. Expected store size ≈ 596 million rows (projected from the 2012 and 2024 zips; the RESULT records the measurement).
*Measured, §6: 617,164,038 rows — the projection was 3.5 % low — and
695,155,362.20 apparent fishing hours against the "≈ 0.8 billion" projected here.*

## 6 · RESULT — BUILT, PUBLISHED AND VERIFIED, 2026-09-16

TL;DR: the store is real and it is bigger than the plan thought. **617,164,038
rows** — one per day × 0.1° cell × vessel, 2012–2024 — are published at
`tensors/family10_2/fishing/` on the public Hugging Face dataset repository
`chfrank/earth-tensors`, every year reconciled one-for-one against its own
source zip, and the whole record holds **1,985,162,793.77 broadcasting hours**
of which **695,155,362.20** are apparent fishing. Family 10.2's registry lists
ten groups and **2,741,955,408 tier-P observations**. The build took 84 minutes
on one rented box and cost about **$0.70**. Every clause of §5's falsifier
holds; the two numbers §5 and §2 projected — 596 million rows and ≈ 0.8 billion
fishing hours — were both corrected by measurement, which is what a projection
is for.

**The runs.** All three are `family10-build`, the `workflow_dispatch`-only
workflow that builds one family-10 store end to end:

- **[#18](https://github.com/blauewelt/earth/actions/runs/35126550455) (E-081 · probe of the `fishing` store's index and fetch stages over 2012 alone, on a GitHub-hosted `ubuntu-latest` runner)** — failed, and not on our code: every URL of Zenodo record 14982712 answered **HTTP 504** from about 17:10Z to about 18:06Z on 2026-09-16. Nothing was spent; the archive was down.
- **[#19](https://github.com/blauewelt/earth/actions/runs/35132684369) (E-081 · the same 2012-only probe, re-dispatched after the outage)** — green in **3 minutes** at 18:09Z: **6,257,384 rows in 7 parts**, 97.5 s of fetch, reconciled against the zip's own row count. That is the whole of the plan's "probe first, free" clause, and it is what licensed the box.
- [**#20** (E-081 · the full `fishing` build — `store=fishing stage=all start=2012-01-01 end=2024-12-31`, on the rented box `gpu-box-49553569`)](https://github.com/blauewelt/earth/actions/runs/35134413191) — dispatched 18:26Z, success 19:50Z, **84 minutes**. Stage times: index 132.5 s; fetch **4,105.8 s** (streaming assembly, peak resident memory 18.98 GB); grid 328.0 s; publish 398.5 s for 11 items, the registry step ending `verified by restore` — the publish downloads back what it has just uploaded and re-hashes it before it declares success.

**The box, in one sentence.** The Estonia box parked from the 10.1 work
(instance 51232297, offer 51008173, a *verified* host) never came back after a
`start` — `actual_status: offline` for 13 minutes and the machine gone from the
offer list — so it was destroyed and a new verified box was rented instead
(instance 51237466 from offer 49553569, Texas, 926 Mbps up / 925 down,
reliability 98.2 %, $0.467/h with 100 GB of disk; runner `gpu-box-49553569`),
stopped at 19:50Z and destroyed at 19:55Z. **Cost of the build ≈ $0.70**,
against §5's "one verified box" estimate. The lore that came out of it is in
`ml/CLAUDE.md` §7: a stopped box's host can go dark, and the answer is to rent,
not to wait.

**The store.** `tensors/family10_2/fishing/`, schema 2 (the tier-P column
layout whose time column is `time_s`, int32 seconds since
1982-01-01T00:00:00Z), C = 2 (`fishing_hours`, `hours`, float16, unit h),
`bin_first` **2191**, `bin_last` **3141**, **951 bins and every one of them
live**, footprint (`log2_fp` −1.32, `log2_dt` −2.32), built 2026-09-16T19:35:41Z
from builder commit `e70270a`. Ten files, ≈ 24.1 GB: the nine arrays at
617,164,038 × 39 B ≈ 24.07 GB plus `bin_offsets` (952 × 8 B), and beside them
the vessel table as `vessels.csv.gz`, 19 MB.

| year | rows | | year | rows |
|---|---|---|---|---|
| 2012 | 6,257,384 | | 2019 | 54,201,316 |
| 2013 | 19,099,705 | | 2020 | 55,991,204 |
| 2014 | 23,126,797 | | 2021 | 60,693,166 |
| 2015 | 26,407,087 | | 2022 | 72,195,082 |
| 2016 | 34,864,177 | | 2023 | 84,786,933 |
| 2017 | 43,858,727 | | 2024 | 84,800,259 |
| 2018 | 50,882,201 | | **total** | **617,164,038** |

`rows_read` = `rows_packed` = 617,164,038, with `drop_bad_number` 0,
`drop_no_position` 0 and `drop_out_of_range` 0 — the source table is already
clean, so the guards are guards rather than filters, and the ledger closes with
no residual. `days_expected` = `days_found` = **4,749**, every calendar day of
the thirteen years.

**THE FALSIFIER, clause by clause.**

1. **`N_year` equals the number of data rows in that year's zip — HOLDS, for all thirteen years.** Each year is marked only when the two agree, and each did.
2. **`fishing_hours_total` is RECORDED, not gated — recorded.** float64 sums over the float16 values: **695,155,362.20 apparent fishing hours** and **1,985,162,793.77 broadcasting hours**. So the whole record is ≈ **0.70 billion** fishing hours, where §2 projected ≈ 0.8 billion; and Global Fishing Watch's release-note figure of "nearly 370 million hours" is confirmed **not** to be this table's sum, which is exactly what §5 said could not serve as a band.
3. **`fishing_hours ≤ hours` on every row — HOLDS.** The builder's assertion `0 ≤ fishing_hours ≤ hours ≤ 168` never tripped in 617 million rows, and a sandbox range-read of 1,000,000 rows at offset 300,000,000 confirmed it independently (maximum `hours` in that window 47.3125, no NaN). Per-channel maximum over the store is **49.6875 h**, the float16 of the measured `max_hours` 49.6755 — over 24 h, as §2 measured on the 2012 zip before the build and therefore expected: `hours_over_24h` is **10,688,165 rows (1.73 %)**. `rows_zero_hours` is 4,702,031 (0.76 %) and `qc_unknown` 17,809 rows, an MMSI (Maritime Mobile Service Identity, a vessel's radio call number) with no vessel-table row in that year. No NaN anywhere.
4. **The monthly grid's global sums equal the store's — HOLDS at 2.42e-9.** `grid_sum` reads [695,155,362.31, 1,985,162,793.78] against the store's two totals, and the worst per-month relative disagreement is **2.42e-9**.
5. **The registry lists the tier-P groups with every 10.1 sha256 unchanged — HOLDS.** `tensors/family10_2/family10.json`, `family_version` 10.2, **ten groups**: four tier-G groups from family 7.1 (`g025`, `g100`, `oc025`, `rg100`) and six tier-P (`argo` 2,678,439 · `gdp` 48,480,798 · `gtmba` 1,001,282 · `socat` 41,830,675 · `slatrack` 2,030,800,176 · `fishing` 617,164,038 = **2,741,955,408 observations**), `groups_missing: []`, and an `inherits` block `{"10.1": {groups: [gdp, gtmba, socat, slatrack], root: tensors/family10_1, registry: tensors/family10_1/family10.json}}`. The four inherited entries, **every sha256 included, are byte-identical to the 10.1 registry's**. Published `verified by restore`.

**The `qc` channel is the gear class, and it is a code table, not a grade.**
16 classes plus 0 for unknown: 1 `dredge_fishing`, 2 `drifting_longlines`,
3 `fishing`, 4 `fixed_gear`, 5 `other_purse_seines`, 6 `other_seines`,
7 `pole_and_line`, 8 `pots_and_traps`, 9 `purse_seines`, 10 `seiners`,
11 `set_gillnets`, 12 `set_longlines`, 13 `squid_jigger`, 14 `trawlers`,
15 `trollers`, 16 `tuna_purse_seines`. Over the vessel table's 773,165
vessel-years: trawlers 335,979 · fishing 171,310 · set_gillnets 64,659 ·
set_longlines 45,538 · drifting_longlines 41,544 · fixed_gear 38,021 ·
other_purse_seines 26,504 · pole_and_line 12,733 · squid_jigger 9,626 ·
dredge_fishing 9,617 · tuna_purse_seines 6,630 · pots_and_traps 5,294 ·
purse_seines 2,092 · seiners 1,541 · trollers 1,190 · other_seines 887 — which
sums to 773,165 exactly, the vessel table's own row count. The fleet grows
almost tenfold across the record: 10,447 vessels in 2012, then 31,896 · 35,985
· 38,123 · 47,484 · 58,925 · 64,898 · 68,538 · 68,128 · 75,153 · 82,681 ·
96,450, and **94,457 in 2024**.

**The sha256 of the ten published files**, as the record of what is on the Hub:

| file | sha256 |
|---|---|
| `bin.npy` | `4f075137dd93ae6160671a18cb03e64bfa5c35961489f4a2dc2b9af1816e6c89` |
| `bin_offsets.npy` | `03631d4821c9738f84d90f257aff5e9d14157f3423875680413f8ad1aed60134` |
| `fp.npy` | `bb33a307a73a19af11cb7276c0450fe736cbe63c192196a49b4477090a85b002` |
| `lat.npy` | `d3ed01b6ad8f0c014468ed9748daf21781f13df75d9d3c6db2b09cf217d8bc1a` |
| `lon.npy` | `f0b6003f624f6414eef15485f23de082868d48af589c9171a3a3e1065b7b046a` |
| `platform.npy` | `6e86126e2b3fb6d20d7bd3f4f881246402d95f2ec97ffbd175e0c41835397e21` |
| `qc.npy` | `fb05c17d317564588a76a735d3aabd3265f6a875ec7e2121cca7faf169d485c5` |
| `time_s.npy` | `2a11428a70f29aa27ff186489fbf25191a55200f997f530cd66f5dfeb502fd77` |
| `values.npy` | `4aa2aa70c5a7edf37bb8f5e6ae4f9fb2cbff14982274f1b96bd23e9b7695a9ba` |
| `vessels.csv.gz` | `8892c856cdcc84296f435d48063e03132599c5f2d039016787b6181e98c0697a` |

**The source, exactly.** Global Fishing Watch, *Global AIS-based Apparent
Fishing Effort Dataset* v3.0 (2025-03-11), Zenodo record 14982712, licence
**CC BY-NC 4.0** — AIS being the Automatic Identification System, the
collision-avoidance radio every large vessel broadcasts. The daily table
`mmsi-daily-csvs-10-v3-<year>.zip` for 2012–2024 is **13 zips,
5,336,047,844 bytes**; the vessel table `fishing-vessels-v3.csv` is
**114,823,860 bytes over 773,165 rows**, md5
`b5ba27cedd5426c0bcb8e6009e911cf0`, and it is published beside the store as
`vessels.csv.gz` because it is the only way a consumer can turn a platform hash
back into a flag, a gear class or a length.

**§3's grid, and what it is for.**
`tensors/family10_2/fishing_grid/fishing_grid_monthly_025.npy`, shape
[156, 721, 1440, 2], **float32** rather than the planned float16 — measured
before the build and confirmed by it, the largest 0.25° cell-month is
**595,726 vessel-hours** against float16's 65,504 ceiling, so the fleet's own
busiest cells would have overflowed. 1,295,723,648 bytes, 128 of them the
`.npy` header, one month of both channels a contiguous **8,305,920-byte** slab,
months 2012-01 … 2024-12, `complete: true`, manifest at
`fishing_grid/grid.json`. `data/fishing_index.json` is written by
`ml/publish_fishing_index.py` from the published bytes (CORS measured against
the Hub: HTTP 206, `access-control-allow-origin: *`, final host
`us.aws.cdn.hf.co`) and is committed in `79952cc`, so the
"Fishing effort (AIS), 0.25° monthly" globe layer goes live with that deploy.
Checked from a sandbox at month 2020-01 (index 96), the slab's own sums are
3,295,177 fishing hours and 11,832,883 broadcasting hours, equal to the
manifest. The grid is **for the globe, not for training**: a model reads the
store.

**One thing is still waiting on a secret.** The second layer of §4,
"Loitering vessels (last 30 days)", shows the committed fixture until the
repository secret `GFW_API_TOKEN` exists; the daily workflow
`.github/workflows/refresh-loitering.yml` then bakes `data/loitering.json`.
That is the designed behaviour, not a failure.

**Deviations from the plan: none, beyond two numbers the plan asked to have
measured** — N 596 M → **617,164,038**, and ≈ 0.8 billion → **0.70 billion**
apparent fishing hours.

Result: [E-081 in the experiment log](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-081).
Reader contract: [the family-10 data handover](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY10_DATA_HANDOVER.md).
