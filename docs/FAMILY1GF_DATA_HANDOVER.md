# Family 1.gf — global fine-scale ocean and atmosphere observations: a self-contained data handover

PDF design note: [family1gf.pdf](https://blauewelt.github.io/earth/ml/paper/notes/family1gf.pdf)

**For an agent that has not seen this repository.** Everything needed to
download, open, validate and search the thirteen stores of family 1.gf, and to
read the registry that lists them, is on this page. Nothing below needs another
document, and nothing below imports code from this repository: the recipes are
numpy, plus the `zstandard` package for the one gridded store. Where a section
says "see also", it is optional background. Written 2026-09-22. Every number on
this page was either **measured from the public Hugging Face dataset today**
(a `store.json` or `manifest.json` downloaded, a `.npy` header or a first and
last element read by an HTTP range request, one whole small store downloaded
and re-hashed, one gridded frame read and decoded) or is quoted from a
committed `store.json`, the build log or the registry, and says so. What could
not be measured says "not measured".

> **Rules for the reader — read-only, everywhere.**
>
> 1. **Copy what you need into your own storage and work from your copy.** The
>    source is the public Hugging Face dataset repository
>    [`chfrank/earth-tensors`](https://huggingface.co/datasets/chfrank/earth-tensors),
>    read anonymously over plain HTTPS. No account and no token are needed for
>    anything public on this page.
> 2. **Never write to anything public.** Not to the Hugging Face dataset, not to
>    the GitHub repository
>    [`blauewelt/earth`](https://github.com/blauewelt/earth), not to the globe
>    site. No uploads, no pull requests, no issues, no discussion posts, no
>    comments. Report what you find to the person who gave you this document.
> 3. **One store is private.** `seaice_asi` lives in the private repository
>    `chfrank/earth-tensors-private` (an anonymous read answers HTTP 401). It is
>    **available only with a read token from the owner**. This page does not
>    contain a token and does not ask for one; if you are not given one, skip
>    that store — the registry lists it so you can skip it by its field rather
>    than by a failed download.
> 4. **Read-only applies to tokens too.** If you hold any Hugging Face token for
>    any reason, do not use it against these repositories for anything but a
>    read.

**What 1.gf is, in one sentence.** Family 1.gf ("gf" for *global, fine*) is
the set of observation stores covering the whole globe at **10 km or finer in
space and 5 days or finer in time** — ship reports back to 1662, ocean casts
back to 1772, carbon bottles, biogeochemical floats, open-ocean moorings,
along-track wave height, daily 4 km ocean colour, and six more that are parked
or not yet built — each stored at its own resolution and cadence, beside the
programme's 0.25° gridded tensor rather than resampled onto it.

**What you can download today, at a glance.** Seven stores are built and
public, one is built and private, and five are not built (one of those has
parts parked on the Hub). The "N" column is rows for a point store and daily
frames for a gridded store.

| store | tier | state (2026-09-22) | N | stored bytes | record, measured | licence |
|---|---|---|---|---|---|---|
| `glodap` | P | **BUILT AND VERIFIED**, public | 1,460,215 bottle samples | 77,433,120 | 1972-07-24 → 2023-09-10 | CC BY 4.0 |
| `wod` | P (schema 3) | **BUILT AND VERIFIED**, public | 15,281,373 casts | 4,385,949,864 | 1772-12-15 → 2026-02-12 | US government work, no restriction |
| `icoads` | P (schema 3) | **BUILT AND VERIFIED**, public | 1,107,951,670 reports | 52,073,957,276 | 1662-12-01 → 2026-08-31 | NOAA open data, attribution |
| `bgcargo` | P | **BUILT AND VERIFIED**, public | 335,231 profiles | 73,467,590 | 2002-09-08 → 2026-09-17 | CC BY 4.0 |
| `oceansites` | P | **BUILT AND VERIFIED**, public | 66,483,202 rows | 5,518,209,460 | 1980-10-17 → 2026-09-16 | free, with acknowledgement |
| `swh` | P | **BUILT AND VERIFIED**, public | 1,680,274,586 samples | 55,449,091,761 | 1991-08-03 → 2023-12-31 | free and open, attribution |
| `oc4k` | G (sharded) | **BUILT AND VERIFIED**, public | 9,224 daily frames | 175,961,577,397 (manifest) | 1997-09-04 → 2022-12-31 | free and open, attribution |
| `seaice_asi` | G (sharded) | **BUILT, PRIVATE** — licence pending | 5,187 daily frames per hemisphere (build log) | 1,657,059,360 (build log) | 2012-07 → 2026-09 (build log) | free for science, non-commercial; redistribution **pending** |
| `pace4k` | G (sharded) | **PARKED** — three year-lanes of parts on the Hub, no store | — | parts: 21,374,472,626 | parts: 2024-03 → 2026-08 | NASA open data |
| `sst_acspo02` | G (sharded) | **NOT BUILT** — probed, waiting on a decision | — | — | — | NOAA open data |
| `swot` | P | **NOT BUILT** — probed, waiting on a storage decision | — | — | — | NASA open data |
| `irtb` | G (sharded) | **NOT BUILT** — blocked: Earthdata application unapproved | — | — | — | NASA open data |
| `xco2` | P | **NOT BUILT** — blocked: the same unapproved application | — | — | — | NASA open data |

"Stored bytes" for a point store is the nine arrays plus `store.json` (the
small `manifest.json` beside them is extra); for `oc4k` it is the sum over its
`manifest.json`, which counts shards, shard indices, `tile_grid.json` and
`shard_index.npy`. §9 has the runs, the dates and what each check was.

---

## 1 · What this is, in one paragraph

Family 1.gf is the fine half of the programme's observation ledger. The
programme's gridded input tensor, family 7.2, puts every quantity onto one
0.25° grid (27.83 km) at five-day steps; family 10.2 added point observations
(drifters, moorings, ship CO₂, along-track sea level, fishing effort) under a
**granularity-aware storage contract** in which nothing is resampled when it is
stored and every value carries its **footprint** — how much area and how much
time it averaged. Family 1.gf applies that same contract to everything global
at ≤ 10 km and ≤ 5 days. It has two tiers in use: **P** (points, profiles and
tracks: nine column arrays sorted by five-day bin, read by a k-nearest search,
exactly family 10's layout) and **G** in a new **sharded** form (a grid finer
than 0.25° cut into compressed 256 × 256 tiles, one file per five-day bin, read
two HTTP range requests at a time). The third tier, **T** (scene catalogues),
exists in the sibling land family 1.0.tf and is not used here. The family is
designed around four prediction goals — El Niño, the ocean carbon sink, the
Atlantic overturning circulation and sea-surface temperature — and each store
was admitted because one of them needs detail finer than the 0.25° grid.

Terms used below, once. **Epoch** — 1982-01-01T00:00:00 UTC, the instant every
time is counted from. **Pentad / bin** — a five-day period counted from the
epoch: `bin = floor_divide(time_s, 432000)`, 432,000 being five days in
seconds; a bin is **negative** before 1982. **Frame** — one time step of a
gridded store: frame `f` of bin `b` covers
`[b·432000 + f·frame_seconds, + frame_seconds)` seconds after the epoch, so a
daily store has five frames per bin. **Schema version** — which time column a
point store carries: schema 2 is `time_s` as int32 seconds (spans
1913-12-13T20:45:52Z to 2050-01-19T03:14:07Z), schema 3 is the identical layout
with `time_s` as **int64**, used only where a record starts before 1914.
**CSR** (compressed sparse row) — an offsets array saying where each bin's rows
start and end in the sorted table. **Footprint** — the pair
`(log2_fp, log2_dt)`: `log2(footprint_km / 27.83)` and
`log2(support_days / 5)`; a point instrument is labelled `log2_fp = −4`.
**zstd** — Zstandard, the lossless compressor the tiles use. **Probe** — one
calendar month put through a store's real adapter on a build machine, whose
measured sizes replace the design note's estimates before a full build is
dispatched. **Lane** — one build job that fetches a window of years and parks
the result on the Hub under `partials/…` for a later assembly job.

Acronyms, spelled out once: **GLODAP** — the Global Ocean Data Analysis
Project (interior carbon and nutrient bottle samples); **WOD** — the World
Ocean Database (NOAA's archive of every ocean profile); **ICOADS** — the
International Comprehensive Ocean-Atmosphere Data Set (ship, buoy and platform
weather reports); **BGC-Argo** — biogeochemical Argo, profiling floats that
also measure oxygen, nitrate, pH and chlorophyll; **OceanSITES** — the global
network of open-ocean reference moorings; **SWH** — significant wave height;
**OC-CCI** — the ocean-colour record of the European Space Agency's Climate
Change Initiative; **ASI** — the ARTIST Sea Ice algorithm of the University of
Bremen; **AMSR2** — the Advanced Microwave Scanning Radiometer 2 (a Japanese
satellite radiometer); **PACE** — NASA's Plankton, Aerosol, Cloud, ocean
Ecosystem satellite; **OCI** — its Ocean Color Instrument; **MOANA** — the
phytoplankton-community product derived from it; **ACSPO** — NOAA's Advanced
Clear-Sky Processor for Ocean (satellite sea-surface temperature); **SWOT** —
the Surface Water and Ocean Topography satellite; **KaRIn** — its Ka-band
radar interferometer; **IRTB** — infrared brightness temperature (cloud-top
temperature from the NCEP/CPC merged geostationary infrared product, whose
NASA name is GPM_MERGIR); **XCO₂** — the column-averaged CO₂ mole fraction;
**OCO-2 / OCO-3** — NASA's Orbiting Carbon Observatories; **GOSAT** — Japan's
Greenhouse gases Observing Satellite, read through NASA's **ACOS** retrieval;
**NCEI** — NOAA's National Centers for Environmental Information; **GDAC** — a
Global Data Assembly Centre; **CEDA** — the UK Centre for Environmental Data
Analysis; **CMR** — NASA's Common Metadata Repository (its catalogue);
**GES DISC** — NASA's Goddard Earth Sciences Data and Information Services
Center; **PO.DAAC** and **OB.DAAC** — NASA's physical-oceanography and
ocean-biology archives; **WOCE** — the World Ocean Circulation Experiment,
whose 1–9 quality-flag scale GLODAP uses; **IMMA1** — ICOADS' International
Maritime Meteorological Archive record format; **DIC** — dissolved inorganic
carbon; **TA** — total alkalinity; **CFC** — chlorofluorocarbon;
**SF₆** — sulphur hexafluoride; **PAR** — photosynthetically available
radiation; **BBP700** — particle backscatter at 700 nm; **PSS-78** — the
Practical Salinity Scale of 1978.

## 2 · Where it is

Hugging Face dataset repository **`chfrank/earth-tensors`**, public, anonymous,
plain HTTPS. Every file is at

```
https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/<path>/<file>
```

`resolve/main/…` answers a 302 redirect to a content-delivery URL; follow
redirects (`curl -L`). A `Range:` request is answered with HTTP 206 and exactly
the bytes asked for — every range read on this page was checked for that. A
listing of a folder, with each file's size and, for the large ones, its sha256
as the Hub stores it, is at
`https://huggingface.co/api/datasets/chfrank/earth-tensors/tree/main/<path>?limit=100`
(it pages at 100 entries; follow the `Link: …; rel="next"` header).

| what | where |
|---|---|
| the registry | `tensors/family1_gf/family1gf.json` |
| a built store | `tensors/family1_gf/<store>/` |
| parked build parts (scaffolding, not data to ingest) | `partials/family1_gf/<store>/<year>/` |
| the private store `seaice_asi` | `chfrank/earth-tensors-private`, `tensors/family1_gf/seaice_asi/` — **token from the owner only** |

### 2.1 · The registry

`tensors/family1_gf/family1gf.json`, measured today: **680,669 bytes**, sha256
`9ec1920f0e1b459ba5589904ed2354b3dd31f6fc6efd45da18dd88116951bf4f`,
`generated_utc` **2026-09-22T09:25:05Z**, `builder_git_sha` `19f29336…`,
written by `ml/build_family1_registry.py` and published with a restore check
(the publisher uploads, downloads the file back and compares sha256 before it
declares success). It reads `family_version` "1.gf", `n_groups` **13**,
`n_built` **7**, `built` = `bgcargo, glodap, icoads, oc4k, oceansites, swh, wod`
and `not_built` = `irtb, pace4k, seaice_asi, sst_acspo02, swot, xco2`.

Per store it carries `name`, `title`, `tier`, `built`, `path`, `repo`,
`distribution` (public or private), `C`, `channels` with units and bounds,
`cadence`, `footprint`, `licence`, `credentials` (which the BUILD needed — a
reader needs none), `sources`, `verified` (what was checked against the
archive, and when), `probe` (one month's measured numbers, or null),
`note_estimate` (the design note's guess, kept beside the measurement so the
two can be compared), and, for a built store, `N`, `bin_first`, `bin_last`,
`files` with sha256, `counts`, `record_span`, `built_at` and
`builder_git_sha`. A gridded store also carries `frames_per_bin`,
`frame_seconds` and `dtype`. **A consumer dispatches on `tier`, resolves files
under `path` in `repo`, and needs nothing else.**

Three things in today's registry a reader must correct for (§10 repeats them):
`seaice_asi` is listed `built: false` with `distribution: "public"`, because
the registry builder looked for it only in the public repository — it IS built,
privately (§2.5); `record_span` is the build's *requested window*, not the data
span (`glodap` reads 1972-01-01 → 2026-12-31 where its rows run 1972-07-24 →
2023-09-10 — the "record" column of the table above is the measured one); and
the registry carries no `token_schema` block although family 10.2's does — the
token schema is the same (§8.1).

### 2.2 · A tier-P store: nine arrays plus `store.json`

Six of the seven public stores (`glodap`, `wod`, `icoads`, `bgcargo`,
`oceansites`, `swh`) are tier P, and every one of them ships the same files.
Measured today on all six: each array's byte count is exactly
`128 + N × itemsize × columns` (the 128 is the `.npy` header), so a size that
does not match means a truncated file.

| file | dtype · shape | what |
|---|---|---|
| `bin.npy` | int16 [N] | `floor_divide(time_s, 432000)`; **negative before 1982** (`icoads` starts at bin −23,309) |
| `time_s.npy` | int32 [N] (schema 2) or **int64 [N] (schema 3: `wod`, `icoads`)** | seconds since 1982-01-01T00:00:00Z, negative before it |
| `lat.npy` | float32 [N] | degrees north |
| `lon.npy` | float32 [N] | degrees east, in **[−180, 180)** |
| `values.npy` | float16 [N, C] | the channels of §3 in **raw units**, **NaN = not measured** |
| `platform.npy` | int64 [N] | the instrument or cruise identity: a numeric source id where the archive has one, else `int(sha1(id)[:15], 16)` — deterministic, 60 bits, never negative |
| `qc.npy` | uint8 [N] | a per-row quality code **whose meaning differs per store** — §4 gives each; three of the six are bitmasks, not grades |
| `fp.npy` | float16 [N, 2] | `(log2_fp, log2_dt)` per row |
| `bin_offsets.npy` | int64 [B + 1] | CSR over **this store's own** bin range: the rows of bin `b` are `[off[b − bin_first], off[b − bin_first + 1])` |
| `store.json` | JSON | schema, channels, footprint, QC policy, licence, sources, verification sentence, per-year counts, per-channel statistics, the ledger of every drop (`counts`) and **the sha256 of every file above** |
| `manifest.json` | JSON | a small publish record beside the store (2–12 KB); not part of the sha256 block |

`oceansites` ships one more file, `platforms.json` (10,326 bytes), naming each
mooring site behind a `platform` value; its sha256 is in `store.json` like the
rest. **Rows are sorted by `(bin, time_s)` ascending**, and `bin_first` is the
store's own first bin, never zero by assumption.

### 2.3 · The six point stores' files, measured today

The sha256 of every array was compared today against the Hub's own record of
the file (the Hub stores each large file's sha256 as its content address), and
all **54 of 54** agree with the `store.json` of the store they belong to;
`oceansites`' `platforms.json`, stored as a small file, was downloaded and
hashed: also equal. For `glodap` the whole store was downloaded and every file
re-hashed locally (§5.1).

| store | N | C | schema | `bin_first … bin_last` | bins (live) | measured fraction of value slots | files · bytes |
|---|---|---|---|---|---|---|---|
| `glodap` | 1,460,215 | 13 | 2 | −690 … 3,045 | 3,736 (2,656) | 0.588174 | 10 · 77,433,120 |
| `wod` | 15,281,373 | 128 | **3** | −15,271 … 3,222 | 18,494 (11,434) | 0.052263 | 10 · 4,385,949,864 |
| `icoads` | 1,107,951,670 | 8 | **3** | −23,309 … 3,262 | 26,572 (20,428) | 0.473497 | 10 · 52,073,957,276 |
| `bgcargo` | 335,231 | 96 | 2 | 1,511 … 3,266 | 1,756 (1,720) | 0.300502 | 10 · 73,467,590 |
| `oceansites` | 66,483,202 | 28 | 2 | −89 … 3,265 | 3,355 (2,519) | 0.194674 | 11 · 5,518,209,460 |
| `swh` | 1,680,274,586 | 3 | 2 | 700 … 3,067 | 2,368 (2,368) | 0.985315 | 10 · 55,449,091,761 |

The record span in the first table was measured by range-reading the header,
the **first** and the **last** element of each store's `time_s.npy` (three
small requests per store), and the first and last `bin` the same way: in all
six stores the first and last bin equal `store.json`'s `bin_first` and
`bin_last` and equal `floor_divide` of the first and last `time_s`.

The sha256 of the **small** stores, for a reader who wants to check a download
by eye (the large stores' digests are in their own `store.json`):

**`glodap`** — `tensors/family1_gf/glodap/`

| file | bytes | sha256 |
|---|---|---|
| `bin.npy` | 2,920,558 | `589dd7669e3a8ed0b1c0d040cfd82431696bfa67571bb61c5d365c8f4109b384` |
| `time_s.npy` | 5,840,988 | `c1b60e9506fd85507f2ff52bf02ec613768320e5401d68414d333559972818de` |
| `lat.npy` | 5,840,988 | `efb52237d2c57071c0fa98ee543c0f03c797065e82505d5c7edb0526e171eb2d` |
| `lon.npy` | 5,840,988 | `02efd32e3031b17e2d7047fbfbcb987cbabcf9ffd9419b49581565376c06e3c7` |
| `values.npy` | 37,965,718 | `8f17a76c590a3932adee63ef5a22eb837cb089e460b3a3781414ce4c4909b1b8` |
| `platform.npy` | 11,681,848 | `62916d592635e2c7f032abd84a1805f791073a9d28d03745828145395fc3766e` |
| `qc.npy` | 1,460,343 | `331de9258e62813494b3505313e3063ccae1503058c2d246e82511490b570fc1` |
| `fp.npy` | 5,840,988 | `32038896aa35db494fdfa0e59c5ae1cefe9a54a840f0e1f3d6ad865f67d2ddb6` |
| `bin_offsets.npy` | 30,024 | `49fc03abee600dd6adbb24a025b67acc0cebff5e2f308fb3b4aad1b94d423b20` |
| `store.json` | 10,677 | *the file carrying the nine digests above* |

**`bgcargo`** — `tensors/family1_gf/bgcargo/`

| file | bytes | sha256 |
|---|---|---|
| `bin.npy` | 670,590 | `b4a6e0469200738e1e39b7446e2c4d1d9da17c1653326891cc4f80acb4f80c88` |
| `time_s.npy` | 1,341,052 | `ad406039df8ad24306ca95845a37a75b85af87571931261fe7ffecc0193513d9` |
| `lat.npy` | 1,341,052 | `d0772cf52284121427c31971af20d33747788f856af1be4d7b3d66e9f9b09428` |
| `lon.npy` | 1,341,052 | `bd6a45b72ee9cd3b22c7d2d57cc6d5828c71c16f29dde327fd5f9c2cca1030f0` |
| `values.npy` | 64,364,480 | `8a2f1c3c4514384757444984025be5f65972978dbddd5476438c46477f5599c4` |
| `platform.npy` | 2,681,976 | `19701d49e7cd7c10a94347e057e1b076fd0930259d9e1b990c669ef587de5b9e` |
| `qc.npy` | 335,359 | `d4e7e05c5709a348ce64baca2f07feaea01d135b568cc391853710d8aa9e1aa0` |
| `fp.npy` | 1,341,052 | `3941c21c3450536d7067f8174afd2792ee4d72a2cf73a845c8a7c52b0c2cfa0d` |
| `bin_offsets.npy` | 14,184 | `833f69b807aa48e4ba295e4e526fd5b064ee9b27faadc77504a600b72cd006eb` |
| `store.json` | 36,793 | *the file carrying the nine digests above* |

**`wod`**, the store most readers will want next, for its `values.npy` alone:
3,912,031,616 bytes, sha256
`c05a3cf9bfcb951747ac5dba226e35e9e0a1536b89217b49643beebf511c61a7`;
`time_s.npy` (int64) 122,251,112 bytes,
`257a47c4e16ced4dee6500c310a303b5cdf505ddb166328062e64d6c7aadcc8b`.

### 2.4 · A tier-G sharded store: `oc4k`

`tensors/family1_gf/oc4k/` holds `store.json` (400,140 bytes), `manifest.json`
(552,020 bytes) and one folder per **group** — here a single group, also named
`oc4k`. A group is one grid:

```
oc4k/tile_grid.json                what a tile IS (sizes, grid, channels, codec, index shape)
oc4k/shard_index.npy               one row per bin: frames present, frame bitmask, bytes, valid pixels
oc4k/<yyyy>/bin_<NNNN>.zst         ONE SHARD PER BIN: every stored tile of the bin's 5 frames, concatenated
oc4k/<yyyy>/bin_<NNNN>.idx.npy     that shard's index: int64 [F, n_tiles_y, n_tiles_x, 2] of (offset, length)
```

`<yyyy>` is the calendar year of the bin's **first** day (so the 2004 folder
holds 74 bins), and `<NNNN>` is the absolute bin number. Measured today:

| what | value |
|---|---|
| grid | 4,320 rows × 8,640 columns, EPSG:4326 (plain latitude/longitude), 1/24° = 4.64 km at the equator, rows **north first** |
| pixel centre | `lon = −180 + (col + 0.5)/24`, `lat = 90 − (row + 0.5)/24` |
| tiles | 256 × 256 × C, 17 tile rows × 34 tile columns = **578 tiles a frame**; the last tile row is padded by 32 rows and the last column by 64 columns, pad = NaN |
| channels · dtype | C = 3, float16, NaN = missing (§3.7) |
| frames per bin · frame length | 5 · 86,400 s (one day) |
| bins | 1,145 … 2,994 = **1,850 shards**, 1997-09-04 → 2022-12-31 |
| frames | **9,224 present**, 26 absent upstream (named in `store.json` → `missing_frames`), 245 before the record (49 whole bins outside it, not written) |
| tiles stored | 3,642,817 (a tile with no valid pixel is never stored) |
| valid fraction | 0.1157 of all pixel-days — ocean seen through cloud, night and ice |
| index header | 128 bytes; each `.idx.npy` is 46,368 bytes (128 + 5 × 17 × 34 × 2 × 8) |
| compression | zstd level 15, one independent zstd frame per tile |
| shard bytes | 175,875,077,967 over the 1,850 shards (mean 95.1 MB, largest 177.2 MB); **bin 1153 is a 0-byte shard** — all five of its days (1997-10-14 → 18) are absent upstream and its index is all `(−1, 0)` |
| files | **3,702** in `manifest.json` (1,850 shards + 1,850 indices + `tile_grid.json` + `shard_index.npy`) summing to **175,961,577,397 bytes**; `store.json` and `manifest.json` are two more |

Checked today: `manifest.json`'s 3,702 file names and sha256 values are
**identical** to `store.json`'s `sha256` block (same names, same digests);
`tile_grid.json` and `shard_index.npy` were downloaded and re-hashed, both
equal; the 2015 shard `bin_2411.zst` (86,004,228 bytes) and its index were
downloaded and re-hashed, both equal; and `shard_index.npy` sums to the
store's own totals (1,850 bins, 9,224 frames present, 26 missing, 3,642,817
tiles, 175,875,077,967 shard bytes). The `tile_grid.json` sha256 is
`store.json["sha256"]["oc4k/tile_grid.json"]`.

The index semantics, which carry the rule that a missing day is never a silent
gap:

| `(offset, length)` | meaning |
|---|---|
| `offset ≥ 0`, `length > 0` | a stored tile: `length` bytes of zstd at `offset` in the shard |
| `offset ≥ 0`, `length = 0` | the frame exists and this tile has no valid pixel (land, or all cloud); nothing stored |
| `offset = −1`, `length = 0` | **the frame is not in the shard** — absent upstream or outside the record; `shard_index.npy`'s `frame_mask` says the same |

Within a shard, tiles are written frame by frame, row-major, with **no gaps**:
each tile's offset is the previous tile's offset plus its length. So one
frame's stored tiles are **one contiguous byte span** of the shard — which is
what makes a frame two range reads (§5.2).

### 2.5 · The private store: `seaice_asi`

Built 2026-09-17 by run #31 (all five build stages on one hosted runner, 58
minutes, with `--distribution private`) and published to
**`chfrank/earth-tensors-private`**, because the University of Bremen has not
yet answered whether its sea-ice concentration may be redistributed. Measured
today: the public path `tensors/family1_gf/seaice_asi/store.json` answers
**404** and the private path answers **401** anonymously. **Its `store.json`
and `manifest.json` could not be downloaded here and nothing about it was
measured today**; the numbers that follow are the build log's, taken from its
`store.json` when it was built. Two groups, `n` and `s` (north and south polar
stereographic, 6.25 km, 1,792 × 1,216 and 1,328 × 1,264 pixels, EPSG:3411 and
EPSG:3412), one channel `sic` (sea-ice concentration, %, stored as uint8 whole
percent with **255 = missing**), five daily frames per bin; **5,187 daily frames
per hemisphere**, 2012-07 → 2026-09 (bins 2,228 … 3,265), **1,657,059,360
bytes in 4,156 files**, 3 days per hemisphere absent upstream (2013-05-11 →
13). Its `store.json` carries `distribution: private`, a `licence_pending`
note and a `distribution_override`. If the owner gives you a read token, the
layout and the two-range recipe of §5.2 apply unchanged, with uint8 in place of
float16. Its probe (2020-03, 62 frames, 54,951,401 bytes fetched) had
projected 1.42 GB, and the store is 17 % above that.

### 2.6 · Parked parts: `pace4k`

`partials/family1_gf/pace4k/` holds three year folders written by three hosted
fetch lanes on 2026-09-20 and **no assembled store**. Measured today by
paging each folder's listing and reading its `done.json` and `counts.json`:

| year folder | files · bytes | shards (bins) | what its `done.json` describes |
|---|---|---|---|
| `2024/` | 127 · 6,989,975,821 | 62 (3,080 … 3,141) | **one bin only (3,141), 4 frames, 68,938,999 bytes** — see below |
| `2025/` | 149 · 9,229,697,368 | 73 (3,142 … 3,214) | all 73 bins, 360 frames |
| `2026/` | 87 · 5,154,799,437 | 42 (3,215 … 3,256) | all 42 bins, 205 frames |

**The 2024 folder is not assemblable as it stands.** The 2025 lane's window
(2025-01-01 → 2025-12-31) contains bin 3,141, which starts on 2024-12-31 and is
therefore filed under 2024; the 2025 lane finished last (19:15:48Z), pushed its
one-bin copy of "2024", and overwrote the 2024 lane's `done.json` and
`pace4k__shard_index.npy` (818 bytes, one row, bin 3,141). The 61 shards of
bins 3,080 … 3,140 are still in the folder, orphaned: nothing indexes them. The
same collision cost the 2025 folder four real days — its bin 3,214
(2025-12-31 → 2026-01-04) marks 2026-01-01 → 04 `after_record` because they
lay outside that lane's window. The build log records this failure mode for
another store ("two adjacent lanes write the same year, and the last one to
push wins"); the lane-aware parts layout that prevents it landed about fifteen
minutes after these lanes finished. What remains: re-fetch 2024 (and bin
3,214) under the new layout, then one assembly. **A reader should not ingest
anything under `partials/`.**

## 3 · The channels

Raw units throughout; nothing is z-scored or anomalised (each `store.json`
says `normalisation: RAW`). Standardise in your loader, from training years
only (§8.2). Channel order is the column order of `values.npy`, and the
authoritative list, with bounds, is `store.json → channels`.

### 3.1 · `glodap` — carbon, nutrients and tracers in ship bottle samples (C = 13)

One row is one bottle: `depth` (m, channel 0 — a bottle store puts depth in a
channel, not in the footprint), `temperature` (°C), `salinity` (PSS-78),
`oxygen`, `nitrate`, `phosphate`, `silicate`, `dic`, `ta` (all µmol kg⁻¹),
`ph` (unitless), `cfc11`, `cfc12` (pmol kg⁻¹), `sf6` (fmol kg⁻¹). From
`store.json`: DIC is measured on 540,672 rows (37.0 %), alkalinity on 501,062,
pH on 330,119, oxygen on 1,270,230, CFC-12 on 410,691. Source: GLODAPv3 merged
master file, 1,498,120 lines from 1,181 cruises.

### 3.2 · `wod` — every non-Argo ocean cast at 16 pressures (C = 128)

**128 = 8 quantities × 16 pressure levels.** The levels are the 16 pressures of
the Roemmich–Gilson Argo climatology the programme uses everywhere: **10, 30,
50, 100, 150, 200, 300, 400, 500, 700, 900, 1,100, 1,300, 1,500, 1,700 and
1,900 dbar**. The quantities, in column blocks of 16 in this order:

| columns | name prefix | quantity | unit | bounds |
|---|---|---|---|---|
| 0–15 | `T_` | temperature | °C | −2.5 … 40 |
| 16–31 | `S_` | salinity | PSU | 0 … 42 |
| 32–47 | `O2_` | dissolved oxygen | µmol kg⁻¹ | 0 … 600 |
| 48–63 | `NO3_` | nitrate | µmol kg⁻¹ | 0 … 60 |
| 64–79 | `PO4_` | phosphate | µmol kg⁻¹ | 0 … 5 |
| 80–95 | `SIO4_` | silicate | µmol kg⁻¹ | 0 … 300 |
| 96–111 | `PH_` | pH | unitless | 6.5 … 9 |
| 112–127 | `CHL_` | chlorophyll | mg m⁻³ | 0 … 100 |

So `T_300` is temperature at 300 dbar, column 6. Each cast was converted from
depth to pressure (Saunders 1981) and interpolated onto the 16 levels; a level
the cast did not reach is NaN. Only 5.2 % of the 1.96 billion slots are
measured — many casts carry temperature alone (the expendable and mechanical
bathythermographs, 4,527,706 casts together, measure nothing else). This is the same
16-level vertical axis as family 8's Argo store, which is why `wod` excludes
Argo profiles (the WOD `PFL` files, 34 of them): those are family 8's.

### 3.3 · `icoads` — ship, buoy and platform weather reports since 1662 (C = 8)

`sst` (°C), `airt` (°C), `slp` (sea-level pressure, hPa), `wind_u`, `wind_v`
(m s⁻¹, derived from the reported direction and speed), `dewpt` (dew point,
°C — the humidity the core record carries), `wave_h` (wind-wave height, m,
reported in half metres), `cloud` (total cloud, oktas). ICOADS Release 3.0
final-untrim for 1662-12 → 2014-12 plus the near-real-time final product from
2015. 3,372 monthly files read, 1,392,492,734 lines, 1,107,951,670 rows kept.

### 3.4 · `bgcargo` — biogeochemical float profiles at 16 pressures (C = 96)

**96 = 6 quantities × the same 16 pressures** as `wod`, in blocks of 16:
`DOXY_` (oxygen, µmol kg⁻¹), `CHLA_` (chlorophyll-a, mg m⁻³), `BBP700_`
(particle backscatter at 700 nm, m⁻¹), `NITRATE_` (µmol kg⁻¹),
`PH_IN_SITU_TOTAL_` (pH, unitless), `DOWNWELLING_PAR_` (µmol m⁻² s⁻¹). One row
is one profile, from the Argo GDAC's synthetic ("Sprof") per-float files;
10,080 floats read, 335,231 profiles kept.

### 3.5 · `oceansites` — open-ocean moorings, hourly or finer (C = 28)

Surface: `airt` (°C), `sst` (°C), `sss` (PSU), `wind_u`, `wind_v` (m s⁻¹),
`pres` (air pressure, hPa), `sw` (downwelling shortwave, W m⁻²), `precip`
(mm h⁻¹). Then temperature `T_` and salinity `S_` at ten depths: **10, 20, 30,
50, 75, 100, 150, 200, 300 and 500 m** (depths in metres here, not pressures).
One row per site per time step. The tropical mooring arrays (TAO/TRITON,
PIRATA, RAMA) are **excluded** — 20,907 files — because they are family 10.1's
`gtmba` store; the ALOHA cabled observatory, entirely below 525 m, is skipped.

### 3.6 · `swh` — along-track significant wave height (C = 3)

`swh`, `swh_denoised`, `swh_uncertainty`, all metres. One row per 1 Hz
altimeter sample, eleven missions: TOPEX/Poseidon 210,886,123 samples,
Jason-2 207,109,982, CryoSat-2 197,045,648, Jason-1 192,222,612, SARAL
167,181,491, Envisat 144,196,581, Jason-3 131,275,935, Sentinel-3A
130,762,699, ERS-2 129,648,727, Sentinel-3B 93,727,610, ERS-1 76,217,178. The
mission is the `platform`, so a consumer can hold out an altimeter.

### 3.7 · `oc4k` — daily 4 km ocean colour (C = 3, tier G)

| idx | name | unit | bounds | what |
|---|---|---|---|---|
| 0 | `log_chl` | log₁₀(mg m⁻³) | −4 … 2.5 | the **base-10 logarithm** of chlorophyll-a — stored as a logarithm because float16 would space the raw concentration 0.0625 apart near 100 mg m⁻³; `10 ** log_chl` gives mg m⁻³ |
| 1 | `kd_490` | m⁻¹ | 0 … 20 | diffuse attenuation of light at 490 nm |
| 2 | `total_nobs` | count | 0 … 2,048 | the merged observation count behind the pixel (a float in the source) |

Source: ESA OC-CCI v6.0, merged level-3 daily 4 km geographic files from CEDA.
The six reflectance bands and fourteen water-class memberships are a later
phase and not in this store.

### 3.8 · The not-built stores' declared channels (from the registry)

`seaice_asi`: `sic` (%, uint8). `pace4k` (C = 7, on `oc4k`'s own 4 km grid):
`log_chl` (log₁₀ mg m⁻³, as `oc4k`), `poc` (particulate organic carbon,
mg m⁻³), `carbon_phyto` (phytoplankton carbon, mg m⁻³), `avw_400` (apparent
visible wavelength **minus 400 nm**), and three MOANA cell abundances
`pro_moana`, `syn_moana`, `pico_moana` in **thousands** of cells per mL (a
level-4 derived product, regional 70°S–70°N, 85°W–25°E). `sst_acspo02`: `sst`
(°C), `quality_level` (ACSPO grade). `swot`: `ssha_karin` (m), `sig0_karin`
(LINEAR backscatter, the product's own unit "1" — not dB; stored bounds
−1,000 … 65,504, decided 2026-09-23 from the collection's CMR metadata),
`ssh_karin_uncert` (m). `irtb`: `tb` (cloud-top brightness temperature,
uint8 as **K − 160**). `xco2`: `xco2` (ppm), `xco2_uncertainty` (ppm),
`surface` (0 land, 1 water).

## 4 · How the values were made (so you can trust or reject them)

Each store's full policy is one paragraph in its own `store.json` under
`qc_policy`, and every drop is counted by reason under `counts`. The common
rules: a **row** is dropped only when it has no usable time, no usable position
or nothing measured; a **value** outside its channel's physical bounds becomes
**NaN and is counted — never clipped**; the source's own quality flags are
applied as written, not re-derived.

### 4.1 · What `qc` means, per store — it is NOT one scale

| store | `qc` is | how to read it |
|---|---|---|
| `glodap` | a **bitmask** | bit value 1: a kept value has GLODAP secondary-QC 0 (not yet adjusted); bit value 2: the time of day was missing and the sample was placed at 12:00 UTC (177,630 bottles). Only WOCE flag-2 values are kept at all |
| `wod` | a **bitmask plus an instrument code** | low bits: 1 a value flag was seen, 2 a profile flag, 4 a depth flag; **high nibble = the instrument**: 1 OSD (bottle), 2 CTD, 3 MBT, 4 XBT, 5 MRB (moored buoy), 6 DRB (drifting buoy), 7 UOR (undulating recorder), 8 APB (animal-borne), 9 GLD (glider). `instrument = qc >> 4`. Only WOD flag-0 values are used |
| `icoads` | a class 0–3, worst kept value | 0 all flags "correct", 1 some "correctable", 2 some unassessed, 3 some "suspect"; values flagged erroneous (7–9) are already NaN |
| `bgcargo` | a **bitmask over the six quantities** | bit `j` set: quantity `j` (in §3.4's order) came from the float's delayed-mode adjusted values rather than the real-time ones |
| `oceansites` | a bitmask | bit value 1: a kept value had no quality flag; bit value 2: a kept value was flagged 5, 7 or 8. Flags 3, 4 and 9 are already NaN |
| `swh` | always 1 | the product publishes no per-sample flag; the store says so rather than inventing one |

`store.json` says `qc_keep_max: 2` in every store, inherited from family 10's
builder. **For the four bitmask stores that number means nothing; filtering
`qc <= 2` on `wod` throws away every cast except the bottle casts with no flag
seen.** Dispatch on the store, or ignore the column.

### 4.2 · Per store, the source and the ledger

- **`glodap`.** One streamed pass over GLODAPv3's merged master file from NCEI
  (1,029,335,273 bytes, 1,498,120 lines). 29,872 bottles without depth and
  8,033 with no kept value dropped; 198 values out of bounds set to NaN (181 of
  them phosphate). Read in 116 s on the hosted runner.
- **`wod`.** WOD 2023 netCDF per instrument and year, 519 files read (34 Argo
  files excluded). 16,445,102 casts seen, 15,281,373 kept; 1,118,757 casts had
  no value on any of the 16 levels and were dropped. Kept by instrument: glider
  3,279,782 · bottle 2,966,691 · MBT 2,344,680 · XBT 2,183,026 ·
  animal-borne 1,967,616 · CTD 1,131,496 · moored buoy 1,063,697 · drifting buoy
  231,111 · undulating recorder 113,274.
- **`icoads`.** 1,392,492,734 IMMA1 lines. Dropped and counted: 243,939,323
  reports with no ICOADS attachment, 35,080,785 with no value, 4,161,424 with an
  erroneous position, 1,266,304 with no hour. Values flagged erroneous: 32.4 M
  wave heights, 5.2 M SSTs, 2.1 M air temperatures. The trimming flags are
  **not** applied — outliers against climatology are left for the model.
- **`bgcargo`.** Per parameter and profile, delayed-mode or adjusted values
  where the float's data mode says so, real-time values otherwise, never mixed;
  samples with QC in {1, 2, 5, 8} interpolated onto the 16 levels; profiles need
  good time and position flags (21,302 dropped for position, 190 for time).
- **`oceansites`.** Where one site publishes the same quantity in several files
  (different cadences or data modes), the finest cadence and best mode win and
  the others' overlapping values are dropped, so one site-time is one row.
- **`swh`.** 11,832 daily files read anonymously from the Copernicus Marine
  native object store (the product is ESA CCI Sea State v4, distributed as
  `WAVE_GLO_PHY_SWH_L3_MY_014_005`); 7 days absent upstream (1992-07-20 → 22,
  2001-01-19 → 22), named in `store.json`; 8 of 1,680,274,594 samples dropped,
  2,774 `swh` and 3,047 `swh_denoised` values out of bounds and NaN.
- **`oc4k`.** The only per-pixel flag in the source is the fill value (land,
  cloud, night, ice → NaN). A chlorophyll at or below 10⁻⁴ mg m⁻³ is NaN too,
  because its logarithm is not a measurement. 18,448 netCDF files, 1.569 TB read
  from CEDA, 332,259 s of fetch across 26 one-year hosted lanes. Every frame
  records `product_version` 6.0.

### 4.3 · Every store was checked before it was trusted

The builder's `check_store` runs on the assembled store and again after
publishing: every file re-hashed against `store.json`; `N` from the file
sizes; `bin` equal to `floor_divide(time_s, 432000)` on every row and
monotone; the CSR offsets reproduced; per-year counts equal to the fetch
ledger; longitude in [−180, 180); every channel inside its bounds. For a
sharded store it decompresses **every stored tile** and checks its pad is all
missing, its values in bounds and its valid-pixel count equal to
`shard_index.npy`'s; for `oc4k` that pass (3.64 million tiles) took about 3.8
hours and was run three times. The publish downloads every file back and
compares its sha256 before the store is declared published.

## 5 · Opening it (numpy only for tier P; numpy and `zstandard` for tier G)

### 5.1 · A point store, end to end — measured on `glodap`

A header can be checked before anything is downloaded. One range read of
`bgcargo`'s value matrix answered 206 with 128 bytes, which decode as the magic
`\x93NUMPY`, version 1.0, a header length of 118 and the dictionary below:

```
$ curl -sL -r 0-127 https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family1_gf/bgcargo/values.npy | od -c
{'descr': '<f2', 'fortran_order': False, 'shape': (335231, 96), }
```

The same three-request trick (header, first element, last element of
`time_s.npy`) is how §2.3's record spans were measured. Then the whole of
`glodap` (77 MB) was downloaded into a scratch directory with plain `curl -L`
of the eleven files, and this ran against it:

```python
import hashlib, json, numpy as np

d = "glodap"                                   # a local copy of tensors/family1_gf/glodap
meta = json.load(open(f"{d}/store.json"))

# 1. verify every file against the store's own record
for name, want in meta["sha256"].items():
    h = hashlib.sha256()
    with open(f"{d}/{name}", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == want, name

# 2. open the nine arrays; schema 2 = int32 seconds, schema 3 = int64
tdt = {2: np.int32, 3: np.int64}[meta["schema_version"]]
bin_   = np.load(f"{d}/bin.npy",        mmap_mode="r")
time_s = np.load(f"{d}/time_s.npy",     mmap_mode="r")
lat    = np.load(f"{d}/lat.npy",        mmap_mode="r")
lon    = np.load(f"{d}/lon.npy",        mmap_mode="r")
values = np.load(f"{d}/values.npy",     mmap_mode="r")   # float16 [N, C]
plat   = np.load(f"{d}/platform.npy",   mmap_mode="r")
qc     = np.load(f"{d}/qc.npy",         mmap_mode="r")
fp     = np.load(f"{d}/fp.npy",         mmap_mode="r")
off    = np.load(f"{d}/bin_offsets.npy")
assert time_s.dtype == tdt
N, C = values.shape
b0, b1 = meta["bin_first"], meta["bin_last"]

# 3. the invariants a reader can check in seconds
assert N == meta["N"] and off[0] == 0 and off[-1] == N
assert np.array_equal(bin_, np.floor_divide(time_s.astype(np.int64), 432000))
assert (np.diff(bin_) >= 0).all()
assert ((lon >= -180) & (lon < 180)).all()

# 4. one pentad through the CSR index: the pentad holding 2015-01-03
EPOCH = np.datetime64("1982-01-01T00:00:00")
t0 = (np.datetime64("2015-01-03T00:00:00") - EPOCH).astype("int64")
b = int(t0 // 432000)
lo, hi = int(off[b - b0]), int(off[b - b0 + 1])

# 5. one row, decoded in units: the first row of that bin with DIC measured
names = [c["name"] for c in meta["channels"]]
units = [c["unit"] for c in meta["channels"]]
rows = np.arange(lo, hi)
r = int(rows[~np.isnan(values[lo:hi, names.index("dic")].astype(np.float32))][0])
print(EPOCH + np.timedelta64(int(time_s[r]), "s"), float(lat[r]), float(lon[r]),
      int(plat[r]), int(qc[r]), fp[r].astype(np.float32).tolist())
for n, u, v in zip(names, units, values[r].astype(np.float32)):
    print(n, "NaN (not measured)" if np.isnan(v) else f"{v:.4g}", u)
```

What it printed, today:

```
sha256: all 9 files match store.json
N 1460215 C 13 schema 2 time dtype int32
bins -690 .. 3045 = 3736 bins; 3737 offsets; 2656 live
bin 2411 starts 2015-01-03T00:00:00 rows 1210891 .. 1211671 = 780
row 1210942: time 2015-01-03T08:50:00  lat -66.5615  lon 119.1797  platform 1037830617728969941  qc 1  fp [-4.0, -6.90625]
   depth        7 m
   temperature  -0.9756 degC
   salinity     33.97 PSS-78
   oxygen       346.2 umol/kg
   nitrate      28.62 umol/kg
   phosphate    NaN (not measured) umol/kg
   silicate     NaN (not measured) umol/kg
   dic          2186 umol/kg
   ta           2296 umol/kg
   ph           NaN (not measured) 1
   cfc11        NaN (not measured) pmol/kg
   cfc12        NaN (not measured) pmol/kg
   sf6          NaN (not measured) fmol/kg
```

A surface bottle off the Antarctic coast south of Australia, at −0.98 °C,
7 m, with DIC and alkalinity measured and pH not. Every invariant held. The
recipe is identical for the other five point stores; only `schema_version`
changes the time dtype, and `mmap_mode="r"` keeps a 52 GB store from being
read into memory. The footprint `(−4, −6.91)` is the point label and a
one-hour support (`log2(1/24/5)`).

Two notes on the arithmetic. `floor_divide` on a negative `time_s` floors
toward minus infinity, which a pre-1982 bin needs; C, C++, Java, Go and Rust
integer division truncates toward zero, so a port must floor explicitly. And a
schema-3 store's `time_s` is int64 because 1662 is 10,069,268,400 s before the
epoch, far past int32; `bin` stays int16, which spans the years 1533 to 2430.

### 5.2 · A sharded frame in two range reads — executed on `oc4k`

**The recipe.** Given a group prefix `P` (here
`tensors/family1_gf/oc4k/oc4k`), its `tile_grid.json` (read once, 4,146
bytes) and a day:

1. `b = floor(seconds_since_epoch / 432000)`, `f = (seconds − b·432000) //
   frame_seconds`, `yyyy` = the year of the bin's first day (the epoch plus
   `5·b` days — **not** the year of the day you asked for).
2. **Range read 1**, the frame's slab of the index: `n = n_tiles_y × n_tiles_x`
   pairs, `16·n` bytes at offset `index_header_bytes + 16·f·n` of
   `P/<yyyy>/bin_<b>.idx.npy`. Reshape as int64 `[n_tiles_y, n_tiles_x, 2]`. If
   every offset is −1, the frame is absent — stop and say so.
3. The frame's stored tiles are one contiguous span of the shard: from
   `slab[0, 0, 0]` to `slab[−1, −1, 0] + slab[−1, −1, 1]`.
4. **Range read 2**, that span of `P/<yyyy>/bin_<b>.zst`. Each tile with
   `length > 0` is `blob[offset − start : offset − start + length]`, one zstd
   frame that decompresses to exactly `256 × 256 × C × 2` bytes of
   little-endian float16 in C order `[row, col, channel]`. A tile with length 0
   is all NaN.
5. Place tile `(ty, tx)` at rows `256·ty …`, columns `256·tx …`, and crop the
   pad to `H × W`.

For one **tile** rather than a frame, the same two reads shrink to 16 bytes at
`index_header_bytes + 16·((f·n_tiles_y + ty)·n_tiles_x + tx)` and then that one
tile's `length` bytes — the form the programme's own reader uses. A place maps
to a tile by `row = floor((90 − lat)·24)`, `col = floor((lon + 180)·24)`,
`ty = row // 256`, `tx = col // 256` on this grid.

**The code that ran:**

```python
import json, urllib.request, datetime as dt
import numpy as np, zstandard

BASE = ("https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/"
        "tensors/family1_gf/oc4k/oc4k")

def get(rel):
    with urllib.request.urlopen(f"{BASE}/{rel}") as r:
        return r.read()

def get_range(rel, off, n):
    req = urllib.request.Request(f"{BASE}/{rel}",
                                 headers={"Range": f"bytes={off}-{off + n - 1}"})
    with urllib.request.urlopen(req) as r:
        assert r.status == 206, "the host ignored the Range"   # never accept a 200
        b = r.read()
    assert len(b) == n
    return b

tg = json.loads(get("tile_grid.json"))
F, nty, ntx, T, C, H, W = (tg["frames_per_bin"], tg["n_tiles_y"], tg["n_tiles_x"],
                           tg["tile"], tg["C"], tg["H"], tg["W"])
day = dt.datetime(2015, 1, 3)
sec = int((day - dt.datetime(1982, 1, 1)).total_seconds())
b = sec // tg["bin_seconds"]
f = (sec - b * tg["bin_seconds"]) // tg["frame_seconds"]
yyyy = (dt.date(1982, 1, 1) + dt.timedelta(days=5 * b)).year
idx, shard = f"{yyyy}/bin_{b:04d}.idx.npy", f"{yyyy}/bin_{b:04d}.zst"

n = nty * ntx                                                    # range read 1
ent = np.frombuffer(get_range(idx, tg["index_header_bytes"] + 16 * f * n, 16 * n),
                    "<i8").reshape(nty, ntx, 2)
if (ent[..., 0] == -1).all():
    raise SystemExit("frame not in the shard")
lo, hi = int(ent[0, 0, 0]), int(ent[-1, -1, 0] + ent[-1, -1, 1])
blob = get_range(shard, lo, hi - lo)                             # range read 2

dz = zstandard.ZstdDecompressor()
frame = np.full((nty * T, ntx * T, C), np.nan, np.float16)
for ty in range(nty):
    for tx in range(ntx):
        o, ln = int(ent[ty, tx, 0]), int(ent[ty, tx, 1])
        if ln:
            t = dz.decompress(blob[o - lo:o - lo + ln], max_output_size=T * T * C * 2)
            frame[ty*T:(ty+1)*T, tx*T:(tx+1)*T] = np.frombuffer(t, "<f2").reshape(T, T, C)
frame = frame[:H, :W].astype(np.float32)                         # drop the pad
```

**What it returned, today** (2015-01-03, 9.7 s wall from this sandbox):

```
day 2015-01-03 -> bin 2411 frame 0 -> 2015/bin_2411.zst
index slab: 9248 bytes; 364 stored tiles of 578 ; shard bytes 0 .. 15296016 = 15296016
ranges 2 bytes transferred 15305264
log_chl    valid 8.8951%  min -2.512  median -0.8545  max 1.972
kd_490     valid 8.8952%  min 0.01196  median 0.03671  max 8.773
total_nobs valid 8.8952%  min 0.1111  median 19.56  max 222.9
Benguela upwelling           (-24.0, 13.0): 556 of 576 pixels valid, median log_chl -0.406 -> chlorophyll-a 0.393 mg m-3, median kd_490 0.0528 m-1
subtropical N Pacific gyre   (25.0, -150.0): no valid pixel in the 1-degree box (cloud)
Patagonian shelf             (-45.0, -60.0): 334 of 576 pixels valid, median log_chl -0.289 -> chlorophyll-a 0.514 mg m-3, median kd_490 0.0751 m-1
```

**Two range reads, 15,305,264 bytes** (9,248 of index, 15,296,016 of tiles)
for one global day of three channels, against 223,948,800 bytes for the same
frame uncompressed: 364 of the 578 tiles hold at least one valid pixel, and
3,320,094 of 37,324,800 pixels (8.9 %) are valid — the rest is land, cloud and
polar night. The global median chlorophyll that day is 10^−0.85 ≈ 0.14 mg m⁻³.

**Independently checked:** the whole shard `2015/bin_2411.zst` (86,004,228
bytes) and its index were then downloaded, both re-hashed equal to
`store.json`, and all five frames decoded tile by tile. Frame 0's valid-pixel
counts reproduced the two-read result exactly (3,320,094 / 3,320,097 /
3,320,098 for the three channels), the five frames' spans tiled the shard end
to end with no gap (0 → 15,296,016 → 32,809,182 → 49,284,952 → 66,911,137 →
86,004,228), and their valid-pixel sums (18,873,790 / 18,873,798 / 18,873,831)
equal `shard_index.npy`'s row for bin 2,411 digit for digit.

A reader that wants a region rather than a globe reads the index slab and then
only the tiles it needs (one range each), since a tile's bytes are
`[offset, offset + length)` of the shard; the contiguous span above is the
cheapest form when most tiles are wanted.

## 6 · Search and indexing

**Tier P** is searched exactly as family 10's stores are: pick the anchor's
bin, walk the CSR slices of that bin and the bins before it (never after — a
forecast must not see the future), compute great-circle offsets, keep the k
nearest inside a radius and an age bound, and return **miss tokens** for empty
slots so a batch is never ragged. The time offset `dt` is measured from the
**end** of the anchor's pentad, so an observation in the anchor's own bin is 0
to 5 days old. **No search radii have been measured for any family-1.gf store
yet.** Family 10's measurement on its own stores found the k nearest tokens
usually come from **one platform** (one drifter, one cruise); expect the same
here in its strongest form for `swh` (1 Hz along one ground track), `wod` and
`glodap` (one cruise's casts) and `oceansites` (one mooring, hourly). If
independent information is wanted, de-duplicate by `platform`.

The bin ranges that matter for a loader: `icoads` has 26,572 bins of which
6,144 are empty; `wod` 18,494 with 7,060 empty; `glodap`
3,736 with 1,080 empty (1975, 1976 and 1979 hold no bottle at all). An empty
bin is `off[i] == off[i+1]`, not an error.

**Tier G** is addressed, not searched: an anchor's latitude and longitude give
the tile (§5.2), its date gives the bin and frame, and a cone of a few hundred
kilometres touches a handful of 256-pixel tiles (one `oc4k` tile is 10.7° on a
side). `shard_index.npy` (714,484 bytes for `oc4k`) says, per bin, which frames
exist (`frame_mask`), how many tiles are stored and the valid fraction per
channel, so a loader can skip a bin without touching its shard. A gridded value
becomes one token at the **pixel centre** with its own footprint —
`oc4k`: `(log2_fp, log2_dt) = (−2.585, −2.322)`, i.e. 4.64 km and one day —
and is never copied into neighbouring cells.

## 7 · Relationship to the other families

### 7.1 · Family 10.2 — same contract, same tier-P layout, a `siblings` line

Family 10.2 is the programme's granularity-aware family at the 0.25° scale:
four point stores of family 10.1 (drifters, tropical moorings, ship CO₂,
along-track sea level), a fishing-effort store, family 8's Argo store and the
family-7.2 gridded tensor by reference, all listed in
`tensors/family10_2/family10.json`. Family 1.gf **inherits its contract by
reference**: the epoch, the bin rule, the footprint token
(`log2(footprint_km / 27.83)`, `log2(support_days / 5)`), the tier letters and
dispatch-on-`tier`, and — for tier P — the nine arrays of §2.2 byte-layout for
byte-layout. A family-10 point store and a family-1.gf point store open with
the same code. The one widening is **schema 3**: `wod` and `icoads` start
before 1914, where int32 seconds end, so their `time_s` is int64 and their
`store.json` says `schema_version: 3`, `time_dtype: "int64"`; the other four
point stores here are **schema 2**, exactly family 10.1's layout. A reader that
already handles family 10.2 needs one dtype branch.

`wod`'s C = 128 is the family-8 vertical axis carried across: the 16
Roemmich–Gilson pressures, for eight quantities instead of Argo's two (§3.2). It
is the complement of family 8's Argo store, not an overlap — WOD's Argo files
are excluded at build time — so `argo` (family 8, via family 10.2) plus `wod`
plus `bgcargo` is every profile on one vertical axis. Likewise `oceansites`
excludes exactly the tropical sites family 10.1's `gtmba` holds, and `swh` is
the wave-height companion of family 10.1's `slatrack` sea-level store from the
same altimeters.

**The `siblings` line.** Family 10.2's registry carries one additive key,
`siblings`, naming the three family-1 registries
(`tensors/family1_gf/family1gf.json`, `tensors/family1_tf/family1tf.json`,
`tensors/family09_tf/family09tf.json`), so a consumer that knows only family 10
discovers these. **Measured 2026-09-22 10:10Z:** `tensors/family10_2/family10.json`
is 54,478 bytes, sha256
`07bbb4c1` (prefix), `generated_utc` 2026-09-22T10:09:04Z, and its
`siblings.registries` list is exactly those three paths; no group entry of the
registry changed when the block was added (the earlier copy of 2026-09-16,
54,052 bytes, lacked the key). Either entry point works: family 10's registry
or this family's own path above.

### 7.2 · Family 7.2 — the 0.25° grid, and the two stores that have a coarse twin in it

Family 7.2 (recipe `f7l2`, `tensors/family7_global025_pentad_l2/`) is the
gridded input tensor: 721 × 1,440 point-aligned cells, **south first**
(`lats = −90 + 0.25·i`, `lons = −180 + 0.25·j`), one row per pentad, four
groups — `g025` (ocean surface at 0.25°, including OISST sea-ice fraction),
`g100` (atmosphere and land at 1°), `rg100` (ocean interior at 1°) and
`oc025` (ocean colour at 0.25°). Its values are **z-scored**; family 1.gf's are
raw.

Two stores here are the fine form of something it already holds, and both stay
beside it rather than replacing it:

- **`oc4k` ↔ `oc025`.** Same source (ESA OC-CCI v6.0 daily 4 km chlorophyll).
  `oc025`'s `log_chl` at grid point (φ, λ) is the mean of `log10(chl)` over the
  **6 × 6 block of 4 km cells whose centres lie within ±0.125°** of the point,
  averaged per day and then per pentad. On `oc4k`'s grid that block is rows
  `24·(90 − φ) − 3 … + 2` and columns `24·(λ + 180) − 3 … + 2`. Note the axis
  flip: `oc4k` rows run **north first**, family 7.2 rows south first. `oc4k`
  stops at 2022-12-31 because CEDA's per-file daily archive does; `oc025` runs
  to 2024-12 because its last two years were read from Plymouth Marine
  Laboratory's aggregate service.
- **`seaice_asi` ↔ `g025`'s `sea_ice`.** `g025`'s sea ice is OISST's
  concentration at 0.25°; `seaice_asi` is Bremen's AMSR2 retrieval at 6.25 km on
  two polar stereographic grids (private, §2.5).

The point stores have no grid relationship at all: a point is a token at its
own position, located against a 0.25° anchor by its offset in kilometres.

### 7.3 · Families 1.0.tf and 0.9.tf

The same programme built two more fine families at the same scale over land and
coast: **1.0.tf** (station networks, tide gauges, buoys, radiosondes, gliders,
fire, daily land fields and scene catalogues, `tensors/family1_tf/`) and
**0.9.tf** (1.0.tf with the raw satellite imagery replaced by Google's
AlphaEarth embedding, everything else inherited by reference,
`tensors/family09_tf/`). Their stores and 1.gf's never overlap: every store is
registered in exactly one of the three registries, which the registry builder
asserts.

## 8 · How to use it in training

### 8.1 · The unit of data

An **anchor** (a place and a pentad) plus, per store, its k nearest
observations as tokens. The token is family 10's: `value[C]`, `mask[C]`,
`dx_km`, `dy_km` (signed kilometres east and north of the anchor), `dt_days`
(age, ≥ 0), `log2_fp`, `log2_dt`, `n_R` (how many observations lay inside the
search bounds — the local density), a source id and a channel id, with the
footprint fields clamped to ±12. A gridded pixel is one token at its centre;
the 36 `oc4k` pixels inside one 0.25° cell are 36 tokens with their own
offsets, not one value repeated.

### 8.2 · Standardise in the loader, from training years only

The stores are raw on purpose. Compute each channel's mean and standard
deviation over **training bins only**. A statistic computed over the whole
store carries the test years into the weights. For `oc4k`, standardise
`log_chl` in log space (it is already a logarithm).

### 8.3 · Missingness is a feature

NaN is "not measured", never zero: zero is a real chlorophyll-free pixel, a
real calm sea and a real 0 °C. `mask` and `n_R` are inputs. In `wod`, 95 % of
the slots are NaN, largely because many casts measure temperature only; in `oc4k`,
88 % of pixel-days are NaN because the sea was under cloud. Do not impute.

### 8.4 · The holdout protocol

The programme's protocol: 2009, 2017 and 2023 are development years; train on
bins up to 2020, test on 2021–2024. The stores carry the whole record — 1662
onward in `icoads` — and the loader restricts. `swh` and `oc4k` end before the
test window closes (2023-12-31 and 2022-12-31), so a model evaluated on 2024
sees neither.

## 9 · What is published and verified (as of 2026-09-22)

Every run below is a run of the family1-build workflow (the job that lists an
archive, fetches it, assembles the store, publishes it with a restore check and
checks it). "Verified" dates are when the store was read back from the Hub
after its run; the "today" column is what this page measured on 2026-09-22.

| store | built by | built at (UTC) | verified | measured today |
|---|---|---|---|---|
| `glodap` | [#2 (glodap: all five stages on one hosted runner, 3 min)](https://github.com/blauewelt/earth/actions/runs/35231063813) | 2026-09-17T14:05:53 | 2026-09-17 | whole store downloaded, 9 of 9 sha256 re-hashed equal, invariants and recipe (§5.1) |
| `wod` | four hosted year-lanes then [#75 (wod: streaming assembly on a hosted runner)](https://github.com/blauewelt/earth/actions/runs/35258593390) | 2026-09-17T18:28:18 | 2026-09-17 | 9 of 9 sha256 against the Hub's record, sizes vs N, first/last `time_s` and `bin` |
| `icoads` | eight hosted lanes then [#93 (icoads: assembly on a rented box, 72 min)](https://github.com/blauewelt/earth/actions/runs/35303946084) | 2026-09-18T00:27:00 | 2026-09-18 | as `wod` |
| `bgcargo` | per-year lanes (three years fetched from a sandbox) then [#84 (bgcargo: hosted assembly, 1 min)](https://github.com/blauewelt/earth/actions/runs/35274263100) | 2026-09-17T21:02:01 | 2026-09-17 | as `wod`, plus a header range read |
| `oceansites` | eight hosted lanes then [#88 (oceansites: hosted streaming assembly, 6 min)](https://github.com/blauewelt/earth/actions/runs/35283054594) | 2026-09-17T22:44:42 | 2026-09-17 | as `wod`, plus `platforms.json` hashed |
| `swh` | six hosted lanes then [#290 (swh: assembly on a rented box, 1 h 41 min)](https://github.com/blauewelt/earth/actions/runs/35510367574) | 2026-09-20T13:10:14 | 2026-09-20 | as `wod` |
| `oc4k` | 26 one-year hosted lanes, assembly [#278 (oc4k: assembly on a rented box; its download-back died on a dropped connection)](https://github.com/blauewelt/earth/actions/runs/35450494024), then [#287 (oc4k: publish and check resumed on the same box)](https://github.com/blauewelt/earth/actions/runs/35486384014) | 2026-09-19T18:12:13 | 2026-09-20 | manifest = store.json sha256 (3,702 of 3,702), `tile_grid.json`, `shard_index.npy` and one shard re-hashed, one frame decoded two ways (§5.2) |
| `seaice_asi` (private) | [#31 (seaice_asi: all five stages, hosted, private track)](https://github.com/blauewelt/earth/actions/runs/35240236311); the first attempt [#6](https://github.com/blauewelt/earth/actions/runs/35236468875) refused the source's 2018 relabelling of its ellipsoid | 2026-09-17 | 2026-09-17 | **not measured** — public 404, private 401 |

The lanes, for completeness: `wod`
[#16 (1772–1987)](https://github.com/blauewelt/earth/actions/runs/35237444678),
[#25 (1988–2008)](https://github.com/blauewelt/earth/actions/runs/35239792464),
[#57 (2009–2017)](https://github.com/blauewelt/earth/actions/runs/35250568979),
[#62 (2018–2026)](https://github.com/blauewelt/earth/actions/runs/35253668879),
and its first assembly
[#73 (killed by the in-memory assembler at 128 channels)](https://github.com/blauewelt/earth/actions/runs/35256711798);
`icoads`' two failed assemblies
[#91](https://github.com/blauewelt/earth/actions/runs/35288902698) and
[#92](https://github.com/blauewelt/earth/actions/runs/35300588775)
(the Hub's Xet uploader could not commit 52 GB; the classic upload path
worked); `swh`'s lanes
[#153 (1991–1996)](https://github.com/blauewelt/earth/actions/runs/35321135386),
[#154 (1997–2002)](https://github.com/blauewelt/earth/actions/runs/35321139708),
[#155 (2003–2008)](https://github.com/blauewelt/earth/actions/runs/35321144248),
[#156 (2009–2013)](https://github.com/blauewelt/earth/actions/runs/35321148553),
[#157 (2014–2018)](https://github.com/blauewelt/earth/actions/runs/35321153353),
[#158 (2019–2023)](https://github.com/blauewelt/earth/actions/runs/35321158105);
`oc4k`'s probe
[#100 (oc4k: one month, 2015-01, which measured CEDA at 5.0 MB/s)](https://github.com/blauewelt/earth/actions/runs/35311630922).

**Sizes against the estimates.** The builder records the adapter's projection
beside the measurement: `glodap` and `bgcargo` hit theirs; `wod` 15.3 M casts
against ≈ 14.7 M projected; `icoads` **19 % fewer rows** than the 1.37 × 10⁹
projected from listing bytes; `oceansites` under its ceiling of 1.3 × 10⁸;
`swh` between the lanes' 1.5 × 10⁹ and the probe's 1.665 × 10⁹; `oc4k`
176 GB against the design note's 0.3–0.4 TB.

### 9.1 · The not-built stores, and the measured reason

| store | what it would be | state | the measured reason |
|---|---|---|---|
| `pace4k` | PACE OCI daily 4 km ocean colour and phytoplankton community, 2024-03 → | **PARKED** | probe [#295 (pace4k: one month, 2024-03, 23 frames)](https://github.com/blauewelt/earth/actions/runs/35526540231): 14.6 s a frame, ≈ 20 MB stored a frame, valid 0.068 of pixels, ≈ 5.8 GB for the record by the framework's estimate (≈ 9.3 GB a year from three real days). Year-lanes [#352 (2024)](https://github.com/blauewelt/earth/actions/runs/35527775713), [#353 (2025)](https://github.com/blauewelt/earth/actions/runs/35527777954), [#354 (2026 to September)](https://github.com/blauewelt/earth/actions/runs/35527780354) parked 21.37 GB — but 2024's index was overwritten (§2.6). Remaining: re-fetch 2024 and assemble |
| `sst_acspo02` | NOAA ACSPO super-collated daily SST at 0.02°, 2000 → | **NOT BUILT**, probed | probe [#172 (sst_acspo02: 2020-01, 18 of 31 days read)](https://github.com/blauewelt/earth/actions/runs/35323368382): 51.7 MB a compressed frame, valid 0.349, 12.5× compression, **≈ 501 GB for the whole record, ≈ 18.9 GB a year** — the design note had 2.3 TB. The re-probe [#199 (all 31 days)](https://github.com/blauewelt/earth/actions/runs/35326316307) went green, but the committed probe file is #172's. Waiting on a decision; the plan is 2020 first, as two six-month lanes (49 s of encoding a frame) |
| `swot` | SWOT KaRIn 2 km swath sea level, 2023 → | **NOT BUILT**, probed | probe [#168 (swot: twelve passes of cycle 010, 2024-01)](https://github.com/blauewelt/earth/actions/runs/35321954861): 680,754 pixels a pass, 51.5 % valid; as rows **118.4 GB a year**, as one sharded group a pass **25.8 GB a year** but ≈ 20,000 files a year and 73 % of every 256-pixel tile is padding on a 69-pixel-wide swath. A storage decision for the programme's owner; the adapter refuses past the probe without an explicit switch |
| `irtb` | merged geostationary infrared cloud-top temperature, 4 km, tropics three-hourly first | **NOT BUILT**, never probed | probes [#178 (irtb: 2015-01)](https://github.com/blauewelt/earth/actions/runs/35325158677) and [#260 (the same, re-run 2026-09-19)](https://github.com/blauewelt/earth/actions/runs/35441348054) failed with HTTP 401 from NASA's Earthdata login for the GES DISC application, which the account has not approved. Also: the layout's 64-bit frame mask caps a bin at 64 frames, so two-hourly is the finest cadence the current layout can hold, not half-hourly |
| `xco2` | OCO-2 / OCO-3 (and GOSAT through ACOS) column CO₂, one row per good sounding | **NOT BUILT**, never probed | probes [#179 (xco2: 2019-06)](https://github.com/blauewelt/earth/actions/runs/35325160394) and [#259 (re-run)](https://github.com/blauewelt/earth/actions/runs/35441347351) failed on the same GES DISC 401. One approval in the Earthdata profile unblocks this and `irtb` |

For every not-built store the registry still carries the design row, the
channels, the licence and the probe (or null), so a consumer can plan around
it; none has a `store.json` on the Hub (all five answered 404 today).

**To build these five stores yourself**, from the NASA and NOAA source
archives, on your own infrastructure and with your own Earthdata account, and
to certify your bytes against the programme's reference years, see
[the family 1.gf build handover](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY1GF_BUILD_HANDOVER.md).

## 10 · Known limits and gotchas

- **`qc` is four different things.** A class in `icoads`, a constant in `swh`,
  bitmasks in `glodap`, `bgcargo` and `oceansites`, and a bitmask with the
  instrument in the high nibble in `wod`. `qc_keep_max: 2` in every
  `store.json` is a family-10 inheritance that means nothing for the bitmask
  stores (§4.1).
- **`wod`'s pressures are dbar; `oceansites`' depths are metres.** `wod` and
  `bgcargo` share the 16 Roemmich–Gilson pressures (10 … 1,900 dbar);
  `oceansites` uses ten depths in metres (10 … 500 m); `glodap` carries depth
  as its channel 0 in metres.
- **Schema 3 is int64.** `wod` and `icoads` read `time_s` as int64; code that
  assumes int32 will overflow silently or refuse. Check `store.json →
  schema_version` / `time_dtype`, or the `.npy` header.
- **A negative bin is normal**, down to −23,309 in `icoads`. Index by
  `b − bin_first`, and floor toward minus infinity (§5.1).
- **The registry's `record_span` is the requested window, not the data.**
  `glodap` says 1972-01-01 → 2026-12-31; its rows run 1972-07-24 → 2023-09-10.
  Read the first and last `time_s`, or `store.json → per_year`.
- **The registry lists `seaice_asi` as not built and public.** It is built and
  private (§2.5); the registry builder only looked in the public repository.
- **`oc4k` ends 2022-12-31**, not at the design ledger's 2026-06-30: CEDA's
  per-file daily archive stops there. The 0.25° `oc025` twin in family 7.2 runs
  to 2024.
- **`oc4k`'s `log_chl` is a logarithm.** `10 ** log_chl` is mg m⁻³. So is
  family 7.2's `oc025`, which is additionally z-scored.
- **`oc4k` rows are north first**, family 7.2's south first.
- **A shard's year folder is the year of the bin's first day**, so bin 2,411
  (2015-01-03 → 07) is in `2015/` but bin 2,410, which starts 2014-12-29 and
  holds 2015-01-01 and 02, is in `2014/`. `store.json → per_year` counts frames by that same
  rule (2004 shows 370 frames).
- **A frame can be absent inside a present shard**, and a whole shard can be 0
  bytes (`oc4k` bin 1,153). Check for offset −1 before reading tiles.
- **Refuse an HTTP 200 where you asked for 206.** A 200 means the host ignored
  the range and is sending the whole file — up to 177 MB for one `oc4k` shard,
  17.7 GB for `icoads`' `values.npy`.
- **The design note's sizes are estimates; the probes replaced them.** Use
  §9.1 and the registry's `probe` block, not the note's storage table.
- **`icoads` is 52 GB and `swh` 55 GB.** Anything that opens "all the point
  stores" should size for these two, and memory-map rather than load.
- **The k nearest tokens are usually one platform** (§6). Not yet measured on
  these stores.
- **`partials/` is scaffolding.** The `pace4k` parts in particular are not a
  consistent record today (§2.6).
- **Family 10.2's registry does not yet name this family** (§7.1).

## 11 · Provenance and licences

Built by `ml/build_family1_stores.py` (stages index, fetch, assemble, publish,
check, plus probe) from one adapter per store in `ml/family1/adapters/`, the
sharded layout in `ml/family1/sharded.py`, the point layout and reader in
`ml/family10_store.py`, registered by `ml/build_family1_registry.py`,
dispatched by the `family1-build` workflow (`workflow_dispatch` only).
Credentials, where a fetch needed them, lived only on GitHub-hosted runners;
the rented boxes that assembled the large stores held no credential beyond the
Hub write token. Every `store.json` carries the builder's git commit, the build
time, the source URLs and a `verified` sentence recording what was checked
against the live archive before the adapter was written. Design:
[the family 1.gf note (PDF)](https://blauewelt.github.io/earth/ml/paper/notes/family1gf.pdf);
build plan:
[E-082](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E082_family1_builds.md);
build log:
[family 1 build log](https://blauewelt.github.io/earth/docs.html?f=ml/family1/BUILD_LOG.md).

**Licences, as each store's `licence` block states them.** "Redistribution"
and "derived works" are the registry's two fields; cite each source when you
use its data.

| store | licence (the block's `name`) | redistribution · derived works | cite |
|---|---|---|---|
| `glodap` | CC BY 4.0 (GLODAP) | with attribution · free | "GLODAPv3 (Lauvset, Key, Olsen et al.), NOAA NCEI OCADS accession 0315582, glodap.info" |
| `wod` | World Ocean Database — US government work (NOAA NCEI), no restriction (CC0) | yes · free | "Boyer et al. (2024), World Ocean Database 2023, NOAA NCEI" |
| `icoads` | NOAA NCEI open data (ICOADS R3.0; CC BY 4.0 at GDEX) | with attribution · free | "ICOADS Release 3.0 (Freeman et al. 2017), NOAA NCEI" |
| `bgcargo` | CC BY 4.0 (Argo data policy) | with attribution · free | "Argo (2000). Argo float data and metadata from Global Data Assembly Centre (Argo GDAC). SEANOE. doi:10.17882/42182" |
| `oceansites` | OceanSITES data policy (free and unrestricted, with acknowledgement) | with attribution · free | "OceanSITES (www.oceansites.org), the site PIs and data assembly centres; GDAC at NOAA NDBC" |
| `swh` | ESA CCI Data Policy (free and open) / Copernicus Marine | with attribution · free | ESA CCI Sea State project (Ifremer/CERSAT, ESA), Copernicus Marine `WAVE_GLO_PHY_SWH_L3_MY_014_005`; Dodet, G. et al. (2020), ESSD 12, 1929–1951 |
| `oc4k` | ESA CCI Data Policy: free and open access, cite | with attribution · free | "Ocean Colour Climate Change Initiative dataset, Version 6.0, European Space Agency, available online at http://www.oceancolour.org/" (Plymouth Marine Laboratory; CEDA) |
| `seaice_asi` | free for scientific use, cite Spreen et al. 2008 | **pending** (asked of the Bremen group 2026-09-17) · **non-commercial** | Spreen, G., L. Kaleschke and G. Heygster (2008), JGR 113, C02S03 — **private, licence pending; non-commercial** |
| `pace4k` | NASA open data (no restrictions) | with attribution · free | NASA Ocean Biology Processing Group, PACE OCI L3M BGC / AOP and L4M MOANA v3.2; MOANA: Lange et al. (2020) |
| `sst_acspo02` | NOAA open data | yes · free | NOAA/STAR ACSPO L3S-LEO v2.81, NASA PO.DAAC |
| `swot` | NASA open data (CC0-equivalent) | yes · free | NASA/CNES SWOT L2 LR SSH Expert, version D, PO.DAAC |
| `irtb` | NASA open data (CC0-equivalent) | yes · free | Janowiak, Joyce, Xie (2017), NCEP/CPC merged IR V1, GES DISC |
| `xco2` | NASA open data (CC0-equivalent) | yes · free | OCO-2/OCO-3 Science Team (2024), GES DISC; ACOS Science Team for GOSAT |

What each permits, in plain words: **CC BY 4.0** and the "with attribution"
policies permit any use, including commercial, provided the source is credited;
**CC0 / US government work** and NASA's open-data policy permit any use with no
condition (credit is still customary); **`seaice_asi`'s terms permit scientific
use only, forbid commercial use of anything derived from it, and have not
confirmed redistribution** — which is why its bytes are private. No store in
this family is under a share-alike or no-derivatives clause.
