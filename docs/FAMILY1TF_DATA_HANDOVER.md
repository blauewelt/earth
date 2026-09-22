# Family 1.0.tf — land and coast observations at ten kilometres and five days: a self-contained data handover

PDF design note: [family1tf.pdf](https://blauewelt.github.io/earth/ml/paper/notes/family1tf.pdf)

**For an agent that has not seen this repository.** Everything needed to
download, open, validate and search the stores of family 1.0.tf, to read the
registry that lists all thirty-three of them, and to tell at a glance which
ones exist today, is on this page. Nothing below needs another document, and
nothing below imports code from this repository: the recipes are numpy, plus
the `zstandard` package for the two gridded stores and `pyarrow` (or any other
Parquet reader) for the catalogues' asset table. Where a section says "see
also", it is optional background. Written 2026-09-22. Every number on this
page was either **measured from the public Hugging Face dataset today** (a
`store.json`, `manifest.json`, `done.json` or `counts.json` downloaded, a
`.npy` header or a first and last element read by an HTTP range request, one
whole point store and one whole catalogue downloaded and re-hashed, one
gridded frame and one gridded tile read and decoded) or is quoted from a
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
> 3. **Two stores are private.** `tide_private` and `gbif_nc` belong in the
>    private repository `chfrank/earth-tensors-private` (an anonymous read
>    answers HTTP 401). They are **available only with a read token from the
>    owner**. This page does not contain a token and does not ask for one; if
>    you are not given one, skip them — the registry lists them so you can skip
>    them by their `distribution` field rather than by a failed download.
> 4. **Read-only applies to tokens too.** If you hold any Hugging Face token for
>    any reason, do not use it against these repositories for anything but a
>    read.
> 5. **The imagery behind the catalogues is the producers', not ours.** A
>    catalogue row points at a scene on a NASA, USGS or Copernicus server that
>    needs your own free account there (§5.4). Use your own account, read-only;
>    this page carries no producer credential either.

**What 1.0.tf is, in one sentence.** Family 1.0.tf ("tf" for *terrestrial,
fine*) is the set of observation stores covering the **land and its coasts at
10 km or finer in space and 5 days or finer in time** — weather stations back
to 1763, radiosondes, tide gauges back to 1800, coastal buoys, ocean gliders,
flux towers, satellite fire detections, pentad rainfall at 0.05°, 30 m forest
loss, and eight **catalogues** that list every Landsat, Sentinel, HLS,
ECOSTRESS, NISAR, VIIRS and OLCI scene without copying a pixel — each stored
at its own resolution and cadence, beside the programme's 0.25° gridded tensor
rather than resampled onto it.

**What you can download today, at a glance.** Seventeen stores are built,
public and verified; one is built and private (`tide_private`); one more
(`gbif_nc`) is private and could not be verified from here (§2.5); seven are
**parked** — parts on the Hub, no assembled store; seven are **not built**. The
"N" column is rows for a point or catalogue store and frames for a gridded
store. "Record" is measured from the first and last row, not taken from the
registry (§10).

| store | tier | state (2026-09-22) | N | stored bytes | record, measured | licence |
|---|---|---|---|---|---|---|
| `ghcnd` | P (schema 3) | **BUILT AND VERIFIED**, public | 1,143,728,366 station-days | 46,914,117,496 | 1763-01-01 → 2026-09-14 | CC0 1.0 |
| `igra` | P (schema 3) | **BUILT AND VERIFIED**, public | 44,842,766 soundings | 8,565,529,637 | 1905-04-04 → 2026-09-16 | US government work, no restriction |
| `tide` | P (schema 3) | **BUILT AND VERIFIED**, public | 1,080,914,415 gauge readings | 35,671,451,441 | 1800-01-01 → 2026-07-15 | GESLA-4 contributor terms, free with attribution |
| `ndbc` | P | **BUILT AND VERIFIED**, public | 628,157,329 reports | 29,523,701,726 | 1970-02-26 → 2025-12-31 | US government work, no restriction |
| `gliders` | P | **BUILT AND VERIFIED**, public | 2,808,590 profiles | 436,057,724 | 2003-10-28 → 2026-09-16 | IOOS Glider DAC, no restriction |
| `fire` | P | **BUILT AND VERIFIED**, public | 631,836,727 detections | 22,114,314,679 | 2000-11-01 → 2026-09-20 | NASA open data, attribution |
| `flux` | P | **BUILT AND VERIFIED**, public — **degraded**: 26 of 806 towers not read | 108,923,184 half-hours | 5,119,786,981 | 1991-01-01 → 2026-01-01 | CC BY 4.0 |
| `chirps05` | G (sharded) | **BUILT AND VERIFIED**, public | 3,288 pentad frames | 19,065,120,937 | 1981-01-01 → 2026-08-31 | CC BY 4.0 |
| `lossyear` | G (sharded) | **BUILT AND VERIFIED**, public | 4,480 sub-tile frames, one bin | 56,059,897,569 | one annual map, loss 2001–2025, filed in the bin of 2026-01-05 | CC BY 4.0 |
| `cat_landsat` | T | **BUILT AND VERIFIED**, public | 10,315,132 scenes | 544,578,219 | 1982-08-22 → 2026-09-16 | rows CC0; imagery public domain |
| `cat_s1` | T | **BUILT AND VERIFIED**, public | 6,203,597 products | 464,219,958 | 2014-10-03 → 2026-09-18 | rows CC0; imagery Copernicus, attribution |
| `cat_s2` | T | **BUILT AND VERIFIED**, public | 41,197,086 tiles | 2,906,157,640 | 2015-07-04 → 2026-09-15 | rows CC0; imagery Copernicus, attribution |
| `cat_hls` | T | **BUILT AND VERIFIED**, public | 37,951,341 granules | 1,941,103,957 | 2013-04-11 → 2026-09-17 | rows CC0; imagery NASA open data |
| `cat_ecostress` | T | **BUILT AND VERIFIED**, public | 19,525,578 granules | 1,009,092,509 | 2018-07-09 → 2026-09-17 | rows CC0; imagery NASA open data |
| `cat_nisar` | T | **BUILT AND VERIFIED**, public | 138,804 frames | 8,907,651 | 2025-10-12 → 2026-09-17 | rows CC0; imagery NASA open data |
| `cat_viirs` | T | **BUILT AND VERIFIED**, public | 2,341,071 granules | 128,806,565 | 2012-01-19 → 2026-09-17 | rows CC0; imagery NASA open data |
| `cat_olci` | T | **BUILT AND VERIFIED**, public | 1,591,637 granules | 126,776,002 | 2016-04-25 → 2026-09-18 | rows CC0; imagery Copernicus, attribution |
| `tide_private` | P (schema 3) | **BUILT, PRIVATE** — research-only licence | 179,314,808 (build log) | 5,917,632,868 (build log) | 1821–2026 (build log) | research only, **no redistribution** |
| `gbif_nc` | P | **PRIVATE, NOT VERIFIABLE HERE** — no build is recorded in the repository (§2.5) | — | — | — | CC BY-NC 4.0, **non-commercial** |
| `lst05` | G (sharded) | **PARKED** — 28 year folders of parts, two assemblies failed; 2007 overwritten | — | parts: 138,598,164,834 | parts: 2000 → 2026-09 | NASA open data |
| `snow05` | G (sharded) | **PARKED** — 28 year folders of parts, complete, never assembled | — | parts: 22,866,010,257 | parts: 2000-02 → 2026-09 | NASA open data |
| `pheno500` | G (sharded) | **PARKED** — 2014–2025 year lanes; 2017 and 2019 not usable | — | parts: 40,153,791,221 | parts: 2014 → 2025 | NASA open data |
| `lai500` | G (sharded) | **PARKED** — two of four quarter-lanes of 2020 | — | parts: 14,261,879,125 | parts: 2020-01 → 2020-06 | NASA open data |
| `canopy30` | G (sharded) | **PARKED** — all fourteen tile lanes of the 2020 map | — | parts: 32,738,357,598 | parts: the 2020 map | CC BY |
| `icesat2` | P | **PARKED** — 9 of 12 monthly lanes of 2022 | — | parts: 65,781,451,215 | parts: 2022 minus Jan, Apr, Sep | NASA open data |
| `gedi` | P | **PARKED** — 5 of 10 three-day lanes of 2022-06 | — | parts: 6,547,144,303 | parts: 15 days of 2022-06 | NASA open data |
| `burned500` | G (sharded) | **NOT BUILT** — probed, no probe file committed | — | — | — | NASA open data |
| `refl05` | G (sharded) | **NOT BUILT** — probed: 2.47 TB for the record, waiting on a decision | — | — | — | NASA open data |
| `sif` | P | **NOT BUILT** — probed: 46.8 GB a year, never dispatched | — | — | — | Copernicus open data |
| `gbif` | P | **NOT BUILT** — probed: 101 GB, never dispatched | — | — | — | CC0 / CC BY 4.0 per record |
| `canopy_ref` | T | **NOT BUILT** — adapter only, never probed | — | — | — | rows CC0; maps CC BY 4.0 |
| `cat_palsar` | T | **NOT BUILT** — adapter only; half of it needs a JAXA account | — | — | — | rows CC0; imagery JAXA terms, pixels not redistributable |
| `cat_biomass_esa` | T | **NOT BUILT** — adapter only, never probed | — | — | — | rows CC0; products ESA data policy |

Four more designed stores are **not in the registry at all** because no
adapter carries them: `alerts`, `biomass100`, `static_fine` and `cat_gfm`
(§9.3). "Stored bytes" for a point or catalogue store is the arrays, the
sidecar (`platforms.json` or `assets.parquet`) and `store.json` (the small
`manifest.json` beside them, 2–9 KB, is extra); for the two gridded stores it
is the sum over `manifest.json` plus `store.json`. Parked "bytes" are the sum
of every lane's `done.json` (§2.7). §9 has the runs, the dates and what each
check was.

---

## 1 · What this is, in one paragraph

Family 1.0.tf is the land half of the programme's fine observation ledger. The
programme's gridded input tensor, family 7.2, puts every quantity onto one
0.25° grid (27.83 km) at five-day steps; family 10.2 added point observations
under a **granularity-aware storage contract** in which nothing is resampled
when it is stored and every value carries its **footprint** — how much area
and how much time it averaged. Family 1.gf applies that contract to the global
ocean and atmosphere at ≤ 10 km and ≤ 5 days; family 1.0.tf applies it to the
continents and a coastal band. It uses all **three tiers**: **P** (points,
profiles and tracks: nine column arrays sorted by five-day bin, read by a
k-nearest search, exactly family 10's layout), **G** in the **sharded** form
(a grid finer than 0.25° cut into compressed 256 × 256 tiles, one file per
five-day bin, read two HTTP range requests at a time), and **T**, the **scene
catalogue**, which is a tier-P store in which one row is one satellite scene —
its time, footprint centre, footprint area, cloudiness and instrument — plus a
Parquet table mapping each row to the producer's own download URLs; **no pixel
is copied** (the imagery is 0.01 to 3 petabytes a year per sensor). The family
is designed for what a model of the land surface has to predict — where forest
will be cleared, what burns, how much carbon a canopy exchanges — and for the
land forcing the programme's four ocean goals (El Niño, the ocean carbon sink,
the Atlantic overturning circulation, sea-surface temperature) need from the
coast.

Terms used below, once. **Epoch** — 1982-01-01T00:00:00 UTC, the instant every
time is counted from. **Pentad / bin** — a five-day period counted from the
epoch: `bin = floor_divide(time_s, 432000)`, 432,000 being five days in
seconds; a bin is **negative** before 1982. **Frame** — one time step of a
gridded store: frame `f` of bin `b` covers
`[b·432000 + f·frame_seconds, + frame_seconds)` seconds after the epoch.
**Schema version** — which time column a point store carries: schema 2 is
`time_s` as int32 seconds (spans 1913-12-13T20:45:52Z to 2050-01-19T03:14:07Z),
schema 3 is the identical layout with `time_s` as **int64**, used only where a
record starts before 1914. **CSR** (compressed sparse row) — an offsets array
saying where each bin's rows start and end in the sorted table.
**Footprint** — the pair `(log2_fp, log2_dt)`: `log2(footprint_km / 27.83)`
and `log2(support_days / 5)`; a point instrument is labelled `log2_fp = −4`.
**zstd** — Zstandard, the lossless compressor the tiles use. **Probe** — one
calendar month put through a store's real adapter on a build machine, whose
measured sizes replace the design note's estimates before a full build is
dispatched. **Lane** — one build job that fetches a window and parks the
result on the Hub under `partials/…` for a later assembly job; a lane inside
one year is a **named lane** (`m06`, `q1`, `d0601-0603`, `g-<hash>`).
**Assembly** — the job that pulls every lane's parts, builds the store and
publishes it. **Sidecar** — the one extra file beside a store's arrays:
`platforms.json` (what each `platform` value is) or, for a catalogue,
`assets.parquet`.

Acronyms, spelled out once: **GHCN-D** — the Global Historical Climatology
Network daily (NOAA's archive of station-day weather); **NOAA** — the US
National Oceanic and Atmospheric Administration; **NCEI** — its National
Centers for Environmental Information; **IGRA** — the Integrated Global
Radiosonde Archive (weather balloons); **GESLA** — the Global Extreme Sea
Level Analysis tide-gauge archive; **UHSLC** — the University of Hawaii Sea
Level Center, which serves it; **NDBC** — NOAA's National Data Buoy Center;
**C-MAN** — its Coastal-Marine Automated Network stations; **GTMBA** — the
Global Tropical Moored Buoy Array (excluded here, it belongs to family 10.1);
**IOOS** — the US Integrated Ocean Observing System; **DAC** — its Glider Data
Assembly Center; **QARTOD** — IOOS's Quality Assurance of Real-Time
Oceanographic Data flags; **FIRMS** — NASA's Fire Information for Resource
Management System; **MODIS** — the Moderate Resolution Imaging
Spectroradiometer on Terra and Aqua; **VIIRS** — the Visible Infrared Imaging
Radiometer Suite on Suomi-NPP, NOAA-20 and NOAA-21; **FRP** — fire radiative
power; **FLUXNET** — the global network of eddy-covariance flux towers;
**AmeriFlux**, **ICOS** (Integrated Carbon Observation System) and **TERN**
(Terrestrial Ecosystem Research Network) — its three data hubs; **BADM** —
FLUXNET's Biological, Ancillary, Disturbance and Metadata file; **NEE** — net
ecosystem exchange of CO₂; **GPP** — gross primary production; **LE** — latent
heat flux; **VPD** — vapour-pressure deficit; **SWC** — soil water content;
**CHIRPS** — the Climate Hazards Center InfraRed Precipitation with Stations;
**GFC** — Hansen's Global Forest Change; **HLS** — NASA's Harmonized Landsat
and Sentinel-2; **ECOSTRESS** — the ECOsystem Spaceborne Thermal Radiometer
Experiment on the Space Station; **ISS** — the International Space Station;
**LSTE** — land-surface temperature and emissivity; **NISAR** — the NASA–ISRO
Synthetic Aperture Radar satellite; **ISRO** — the Indian Space Research
Organisation; **GCOV** — its geocoded polarimetric covariance product;
**SAR** — synthetic aperture radar; **ASF** — the Alaska Satellite Facility;
**DAAC** — a NASA Distributed Active Archive Center (**LP DAAC** land
processes, **LAADS** the Level-1 and Atmosphere Archive and Distribution
System, **NSIDC** the National Snow and Ice Data Center, **ORNL** Oak Ridge);
**CMR** — NASA's Common Metadata Repository (its catalogue); **USGS** — the US
Geological Survey; **STAC** — the SpatioTemporal Asset Catalog standard;
**CDSE** — the Copernicus Data Space Ecosystem; **OData** — the Open Data
Protocol its catalogue speaks; **GRDH** — ground-range detected,
high-resolution (a Sentinel-1 product); **SLC** — single-look complex;
**IW** — Interferometric Wide swath; **IPF** — Sentinel-1's Instrument
Processing Facility; **L2A** — Level-2A surface reflectance; **OLCI** — the
Ocean and Land Colour Instrument on Sentinel-3; **WFR** — its full-resolution
water product; **NT / NR / ST** — non-time-critical, near-real-time and
short-time-critical timeliness; **L1B** — Level-1B radiances; **LST** —
land-surface temperature; **LAI** — leaf-area index; **FPAR** — the fraction
of photosynthetically active radiation absorbed; **EVI2** — the two-band
enhanced vegetation index; **GEDI** — the Global Ecosystem Dynamics
Investigation lidar on the Space Station; **ICESat-2** — NASA's Ice, Cloud and
land Elevation Satellite 2, whose **ATL08** product is land and vegetation
height; **LiDAR** — laser ranging; **SIF** — solar-induced chlorophyll
fluorescence; **TROPOMI** — the Tropospheric Monitoring Instrument on
Sentinel-5P; **OCO** — NASA's Orbiting Carbon Observatory; **GES DISC** —
NASA's Goddard Earth Sciences Data and Information Services Center;
**GBIF** — the Global Biodiversity Information Facility; **JAXA** — the Japan
Aerospace Exploration Agency; **PALSAR** — its Phased Array L-band SAR;
**ESA** — the European Space Agency; **CCI** — its Climate Change Initiative;
**GLAD** — the University of Maryland's Global Land Analysis and Discovery
lab; **ETH** — ETH Zurich; **GSHHG** — the Global Self-consistent,
Hierarchical, High-resolution Geography shoreline; **GEBCO** — the General
Bathymetric Chart of the Oceans; **DIST-ANN** — OPERA's annual vegetation
disturbance product; **OPERA** — NASA JPL's Observational Products for
End-Users from Remote Sensing Analysis; **GFM** — Copernicus' Global Flood
Monitoring.

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
| the registry | `tensors/family1_tf/family1tf.json` |
| a built store | `tensors/family1_tf/<store>/` |
| a gridded store's groups | `tensors/family1_tf/<store>/<group>/` (`chirps05/chirps05/`, `lossyear/00N_000E_r0c0/` …) |
| parked build parts (scaffolding, not data to ingest) | `partials/family1_tf/<store>/<year>/` or `…/<year>/<lane>/` |
| the private stores `tide_private`, `gbif_nc` | `chfrank/earth-tensors-private`, `tensors/family1_tf/<store>/` — **token from the owner only** |

### 2.1 · The registry

`tensors/family1_tf/family1tf.json`, measured today: **6,699,677 bytes**,
sha256 `4ded38a238fc663d3d67ba6185c1d409de168e9f8914fd81f7158b9c03c0782a`,
`generated_utc` **2026-09-22T09:25:14Z**, `builder_git_sha` `19f29336…`,
written by `ml/build_family1_registry.py` and published with a restore check
(the publisher uploads, downloads the file back and compares sha256 before it
declares success). It reads `family_version` "1.0.tf", `n_groups` **33**,
`n_built` **17** (the seventeen public stores of the first table) and
`not_built` = the other sixteen, the two private stores among them. The tiers
split **13 P, 9 G, 11 T**. It is large because `lossyear`'s entry alone
carries 2.2 MB of file digests and a 2.1 MB table of its 4,480 groups.

Per store it carries `name`, `title`, `tier`, `built`, `built_note`, `path`,
`repo`, `distribution` (public or private), `C`, `channels` with units and
bounds, `cadence`, `footprint`, `licence`, `credentials` (which the BUILD
needed — a reader needs none), `sources`, `verified` (what was checked against
the archive, and when), `probe` (one month's measured numbers, or null),
`note_estimate` (the design note's guess, kept beside the measurement), and,
for a built store, `N`, `bin_first`, `bin_last`, `files` with sha256, `counts`,
`record_span`, `store_per_year` and `store_schema_version`; a gridded store
also `frames_per_bin`, `frame_seconds` and `dtype`. **A consumer dispatches on
`tier`, resolves files under `path` in `repo`, and needs nothing else.**
Checked today: for all 17 built stores the registry's `files` list is
**identical** to the store's own `store.json` sha256 block, and its `N` equals
`store.json`'s `N`.

Five things in today's registry a reader must correct for, each measured and
each repeated in §10: **`record_span` is the build's requested window, not the
data** (`ghcnd` reads → 2026-12-31 and ends 2026-09-14; `lossyear`'s is the
bin its one map is filed in); **the private stores read `built: false`**
(`tide_private` is built, §2.5); **the two gridded stores read
`schema_version: 2`, `time_dtype: "int32"` and `N: null`** although they have
no time column — count frames from `store.json → groups`; **`qc_keep_max: 2`**
rides in every `store.json` and is meaningless for most stores (§4.1); and
there is no `token_schema` block, though the token is family 10.2's (§8.1).

### 2.2 · A tier-P store: nine arrays, a sidecar and `store.json`

The seven public point stores and the eight catalogues all ship the same nine
arrays. Measured today on all fifteen: each array's byte count is exactly
`128 + N × itemsize × columns` (128 is the `.npy` header), so a size that does
not match means a truncated file.

| file | dtype · shape | what |
|---|---|---|
| `bin.npy` | int16 [N] | `floor_divide(time_s, 432000)`; **negative before 1982** (`ghcnd` starts at bin −15,998) |
| `time_s.npy` | int32 [N] (schema 2) or **int64 [N] (schema 3: `ghcnd`, `igra`, `tide`, and `tide_private` per the build log)** | seconds since 1982-01-01T00:00:00Z, negative before it |
| `lat.npy` | float32 [N] | degrees north |
| `lon.npy` | float32 [N] | degrees east, in **[−180, 180)** |
| `values.npy` | float16 [N, C] | the channels of §3 in **raw units**, **NaN = not measured** |
| `platform.npy` | int64 [N] | the station, gauge, deployment, tower, satellite or scene: a numeric source id where the archive has one, else `int(sha1(id)[:15], 16)` — deterministic, 60 bits, never negative |
| `qc.npy` | uint8 [N] | a per-row quality code **whose meaning differs per store** — §4.1 gives each; two are bitmasks and eight are a processing-baseline code, not a grade |
| `fp.npy` | float16 [N, 2] | `(log2_fp, log2_dt)` per row — **constant down the column** (the builder asserts it; checked today on `gliders` and `cat_nisar`) |
| `bin_offsets.npy` | int64 [B + 1] | CSR over **this store's own** bin range: the rows of bin `b` are `[off[b − bin_first], off[b − bin_first + 1])` |
| `platforms.json` | JSON | point stores: one entry per `platform` value (station name and position, gauge contributor, glider deployment and institution, satellite and its own pixel size, tower and its UTC offset) |
| `assets.parquet` | Parquet | catalogues only: one row per catalogue row, mapping it to the producer's files (§2.4) |
| `store.json` | JSON | schema, channels, footprint, QC policy, licence, sources, verification sentence, per-year counts, per-channel statistics, the ledger of every drop (`counts`) and **the sha256 of every file above** |
| `manifest.json` | JSON | a small publish record beside the store (2–9 KB); not part of the sha256 block |

**Rows are sorted by `(bin, time_s)` ascending**, and `bin_first` is the
store's own first bin, never zero by assumption.

### 2.3 · The fifteen point and catalogue stores, measured today

The sha256 of every file was compared today against the Hub's own record of
it (the Hub stores each large file's sha256 as its content address) and, for
the six small `platforms.json` files the Hub does not hash that way, by
downloading and hashing them: **all 150 of 150** agree with the `store.json`
of the store they belong to. `gliders` and `cat_nisar` were downloaded whole
and every file re-hashed locally (§5.1, §5.4); three more catalogue sidecars
(`cat_viirs`, `cat_olci`, `cat_landsat`) were downloaded and re-hashed, all
equal.

| store | N | C | schema | `bin_first … bin_last` | bins (live) | measured fraction of value slots | built (UTC) |
|---|---|---|---|---|---|---|---|
| `ghcnd` | 1,143,728,366 | 5 | **3** | −15,998 … 3,265 | 19,264 (19,264) | 0.471711 | 2026-09-17T23:34 |
| `igra` | 44,842,766 | 80 | **3** | −5,607 … 3,265 | 8,873 (8,412) | 0.531345 | 2026-09-17T16:20 |
| `tide` | 1,080,914,415 | 1 | **3** | −13,295 … 3,253 | 16,549 (16,519) | 1.0 | 2026-09-17T20:12 |
| `ndbc` | 628,157,329 | 10 | 2 | −866 … 3,214 | 4,081 (4,030) | 0.481711 | 2026-09-17T18:34 |
| `gliders` | 2,808,590 | 64 | 2 | 1,594 … 3,265 | 1,672 (1,402) | 0.170414 | 2026-09-17T18:29 |
| `fire` | 631,836,727 | 4 | 2 | 1,375 … 3,266 | 1,892 (1,889) | 0.795365 | 2026-09-20T14:25 |
| `flux` | 108,923,184 | 10 | 2 | 657 … 3,214 | 2,558 (2,558) | 0.84457 | 2026-09-20T19:34 |
| `cat_landsat` | 10,315,132 | 5 | 2 | 46 … 3,265 | 3,220 (3,131) | 0.8 | 2026-09-18T12:26 |
| `cat_s1` | 6,203,597 | 5 | 2 | 2,392 … 3,266 | 875 (875) | 0.4 | 2026-09-18T12:07 |
| `cat_s2` | 41,197,086 | 5 | 2 | 2,447 … 3,265 | 819 (816) | 0.599981 | 2026-09-19T21:07 |
| `cat_hls` | 37,951,341 | 5 | 2 | 2,284 … 3,266 | 983 (983) | 0.999999 | 2026-09-19T13:00 |
| `cat_ecostress` | 19,525,578 | 5 | 2 | 2,667 … 3,266 | 600 (571) | 0.4 | 2026-09-19T11:58 |
| `cat_nisar` | 138,804 | 5 | 2 | 3,198 … 3,266 | 69 (67) | 0.4 | 2026-09-18T09:21 |
| `cat_viirs` | 2,341,071 | 5 | 2 | 2,195 … 3,266 | 1,072 (1,072) | 0.4 | 2026-09-18T09:28 |
| `cat_olci` | 1,591,637 | 6 | 2 | 2,506 … 3,266 | 761 (761) | 0.496463 | 2026-09-18T09:31 |

**Which schema each store is** — read from its own `store.json`
(`schema_version`, and `time_dtype` where it is written) and confirmed by the
`.npy` header of `time_s.npy`: **schema 3, int64** for `ghcnd`, `igra` and
`tide` (records from 1763, 1905 and 1800); **schema 2, int32** for the other
twelve. `ndbc` starts in 1970, after int32's 1913 limit, and is schema 2.

The record span in the first table was measured by range-reading the header
and the first and last element of each `time_s.npy` and `bin.npy`: in all
fifteen stores the header's shape is `N`, and the first and last bin equal
`bin_first`, `bin_last` and `floor_divide` of the first and last `time_s`.

The sha256 of the smallest point store, for a reader who wants to check a
download by eye (every other store's digests are in its own `store.json`):

**`gliders`** — `tensors/family1_tf/gliders/`

| file | bytes | sha256 |
|---|---|---|
| `bin.npy` | 5,617,308 | `a5dd0752f5c5f66312fb6c57bd40795144fcf2da0e2bc99c0b8c8defe559b022` |
| `time_s.npy` | 11,234,488 | `6b32f0c9c445b30e9a626d8895d73851ba861cf6c049fe9f431e37ebeed57e06` |
| `lat.npy` | 11,234,488 | `2163132b94dbd31ac18627c2559d9877e66637739479149f910ac0b15d2773c7` |
| `lon.npy` | 11,234,488 | `780539138d870686a2e54ee32c02851105fd6cd1a6ccaad4a9f89a0065576223` |
| `values.npy` | 359,499,648 | `6c2de5aa7efba408b935e9538ee61198d3df95741ae6215804d639572629e6f3` |
| `platform.npy` | 22,468,848 | `4fc52d829e378e8efe34e588e6dbd151f124a5a73b7bc152b430e270d5316bdf` |
| `qc.npy` | 2,808,718 | `971133217dcb17c8f28259f0153c449cdf7796352f428c625255e2bb07884f59` |
| `fp.npy` | 11,234,488 | `ac42e8635444cb3147778a09c87dd09828ec09a7d8280c6ebf1ebe84fc0fc9ce` |
| `bin_offsets.npy` | 13,512 | `21606c2afeb3e202f151f8bb2427e56e0e8e8afea2fe2cb3b1161c2a21e49ee7` |
| `platforms.json` | 678,232 | `31c0fde2673fd3a8e757a04e816c8b208418b08b422e24bc34f2a81348b76a13` |
| `store.json` | 33,506 | *the file carrying the ten digests above* |

`cat_nisar`'s ten digests are in its own `store.json` and were all re-hashed
equal from a whole download (§5.4).

### 2.4 · A tier-T catalogue: what a row is, and the asset table

A catalogue is a tier-P store (the nine arrays of §2.2) whose rows are
**scenes, not measurements**: one row says "this producer holds a scene that
started at `time_s`, whose footprint is centred at `lat`, `lon`, covers
`2 ** log2_area` km², was this cloudy, and was taken by this instrument in
this mode, processed at this baseline". `platform` is the scene's own
identifier hashed to 60 bits, so it is **unique per row** (checked on
`cat_nisar`: 138,804 distinct values in 138,804 rows). `fp` is the scene's
nominal footprint and duration (for `cat_nisar` 180 km and 40 s:
`(2.693, −13.399)`). The pixels stay at the producer.

`assets.parquet` has one row per catalogue row with five columns: `platform`
(int64, the join key), `stac_id` (the producer's own scene identifier),
`collection`, `base_url` (the directory the scene's files live in) and
`asset_set` (string). **What `asset_set` holds, measured on four of the eight
tables today:** a space-separated list of file names in which `{id}` stands
for `stac_id` (`cat_landsat`: sixteen names from `{id}_ANG.txt` to
`{id}_thumb_small.jpeg` under `landsatlook.usgs.gov/data/collection02/…`;
`cat_nisar`: the single `{id}.h5` under `nisar.asf.earthdatacloud.nasa.gov`),
or a literal file name (`cat_viirs`, under
`data.laadsdaac.earthdatacloud.nasa.gov`), or `$value` (`cat_olci`, a CDSE
product at `catalogue.dataspace.copernicus.eu/odata/v1/Products(<uuid>)`). In
every case a file's URL is

```
base_url + "/" + name.replace("{id}", stac_id)     for each name in asset_set.split()
```

`store.json → assets_note` describes `asset_set` as "a key of `asset_sets`",
and `asset_sets` is empty in seven of the eight stores (`cat_landsat` names
`oli-tirs` and `tm`, but its rows carry the expanded list). Trust the column.
The sidecars weigh 3.8 MB (`cat_nisar`), 42 MB (`cat_viirs`), 65 MB
(`cat_olci`), 163 MB (`cat_landsat`), 235 MB (`cat_s1`), 287 MB
(`cat_ecostress`), 537 MB (`cat_hls`) and 1.38 GB (`cat_s2`). §5.4 decodes a
row and follows it to the producer.

### 2.5 · The private stores: `tide_private` and `gbif_nc`

Measured today: `chfrank/earth-tensors-private` answers an anonymous GET of
both stores' `store.json` with **401** — and answers a path that cannot exist
(`tensors/family1_tf/no_such_store_xyz/store.json`) with **401** too, so a 401
says nothing about whether a store is there. The public paths of both answer
**404**. **Nothing about either was measured today.**

`tide_private` is the half of the GESLA-4 archive whose contributors allow
research use only. The build log records it as built 2026-09-17 by run #5 (all
five stages on one hosted runner, 9 minutes): **179,314,808 rows,
5,917,632,868 bytes, 1821–2026, schema 3**, `tide`'s layout and channel.
`tide` and `tide_private` read disjoint record sets.

`gbif_nc` is the CC BY-NC 4.0 part of the GBIF occurrence snapshot. **The
repository holds a probe for it (2020-12, one part of eight: 1,255 kept
records; 28.0 GB projected for the private track) and no record of a build**:
no build-log row, no `gbif_nc` build in the workflow's run list, and a
`built_note` saying only that the private repository was not read. The
programme's internal status notes list it as built; treat that as
**unconfirmed** until the owner says otherwise. With a read token, §5.1 applies unchanged.

### 2.6 · The two tier-G sharded stores

`tensors/family1_tf/<store>/` holds `store.json`, `manifest.json` and one
folder per **group** — one grid. `chirps05` has one group (also named
`chirps05`); `lossyear` has **4,480** (each Hansen 10° tile cut into sixteen
2.5° sub-tiles, `<TILE>_r<row>c<col>`, row 0 northernmost). Each group is:

```
<group>/tile_grid.json                what a tile IS (sizes, grid, channels, codec, index shape)
<group>/shard_index.npy               one row per bin: frames present, frame bitmask, bytes, valid pixels
<group>/<yyyy>/bin_<NNNN>.zst         ONE SHARD PER BIN: every stored tile of the bin's F frames, concatenated
<group>/<yyyy>/bin_<NNNN>.idx.npy     that shard's index: int64 [F, n_tiles_y, n_tiles_x, 2] of (offset, length)
```

`<yyyy>` is the calendar year of the bin's **first** day and `<NNNN>` the
absolute bin number (negative before 1982: `chirps05/1981/bin_-001.zst`).
Measured today:

| what | `chirps05` | `lossyear` |
|---|---|---|
| grid | 2,400 × 7,200, EPSG:4326, 0.05°, 60 °S … 60 °N, rows **north first** | 10,000 × 10,000 a group, EPSG:4326, 0.00025° (27.8 m at the equator), rows **north first**, 280 tiles × 16 |
| pixel centre | `lon = −180 + (col + 0.5)·0.05`, `lat = 60 − (row + 0.5)·0.05` | `lon = west + (col + 0.5)·0.00025`, `lat = north − (row + 0.5)·0.00025`, from each group's `grid.extent` |
| tiles | 256 × 256 × 1; 10 × 29 = 290 a frame; pad 160 rows, 224 columns | 256 × 256 × 2; 40 × 40 = 1,600 a frame; pad 240 rows, 240 columns |
| channels · dtype · missing | C = 1 float16, NaN | C = 2 **uint8**, **255** |
| frames per bin · frame length | **2 · 216,000 s (half-bins of 2.5 days)** — see §3.8 | 1 · 432,000 s |
| bins | −73 … 3,262 = 3,336 shards, of which **62 are 0-byte** (no pentad in either half) | **one bin, 3,215** (2026-01-05 → 09), one shard a group, **1,329 of 4,480 shards are 0-byte** (no mapped land) |
| frames | **3,288 present**; 3,384 half-bins hold no pentad by construction, 50 after the record | 4,480 present (one per group), none missing |
| tiles stored | 628,008 | not summed today |
| valid fraction | 0.28056 of all pixel-frames (land 60 °S–60 °N) | per group, in `store.json → groups` |
| index file | 9,408 bytes (128 + 2 × 10 × 29 × 16) | 25,728 bytes (128 + 1 × 40 × 40 × 16) |
| compression | zstd level 15, one zstd frame per tile | zstd level **6**, one zstd frame per tile |
| shard bytes | 19,031,089,462 (mean 5.70 MB, largest 11,443,933) | 55,910,847,477 (largest 77,788,666, `30N_100E_r3c0`) |
| files | **6,674** in `manifest.json`, summing to **19,064,019,352 bytes**; plus `store.json` (1,101,585) and `manifest.json` | **17,920** in `manifest.json`, summing to **56,055,343,541 bytes**; plus `store.json` (4,554,028) and `manifest.json` |

Checked today: both `manifest.json` files list exactly the names and digests
of their `store.json`'s `sha256` block; `chirps05`'s `tile_grid.json` (336,786
bytes), `shard_index.npy` (1,208,016 bytes), the shard `2015/bin_2447.zst`
(6,979,549 bytes) and its index were downloaded and re-hashed, all equal; and
`chirps05`'s `shard_index.npy` sums to the store's own totals (3,336 bins,
3,288 frames present, 3,384 missing, 628,008 tiles, 19,031,089,462 shard
bytes).

The index semantics carry the rule that a missing frame is never a silent gap:

| `(offset, length)` | meaning |
|---|---|
| `offset ≥ 0`, `length > 0` | a stored tile: `length` bytes of zstd at `offset` in the shard |
| `offset ≥ 0`, `length = 0` | the frame exists and this tile has no valid pixel (sea for a land field); nothing stored |
| `offset = −1`, `length = 0` | **the frame is not in the shard** — for `chirps05` usually "no pentad in this half-bin"; `shard_index.npy`'s `frame_mask` says the same |

Within a shard, tiles are written frame by frame, row-major, with **no gaps**,
so one frame's stored tiles are **one contiguous byte span** of the shard —
which is what makes a frame two range reads (§5.2).

### 2.7 · Parked parts: what is under `partials/family1_tf/` today

Seven stores exist only as lane parts. A reader **should not ingest anything
under `partials/`** — it is the builder's scaffolding, and three of the seven
contain a year whose record was overwritten (below). Measured today by
listing each store's folders, reading every folder's `done.json` (the lane's
own statement of what it pushed: file list with sha256, bytes, rows or frames,
lane window) and, for the tier-G years, its `counts.json`:

| store | folders on the Hub | files · bytes (from `done.json`) | what the parts hold | what remains |
|---|---|---|---|---|
| `lst05` | 28 year folders, 1999–2026 | 3,792 · 138,598,164,834 | 9,227 daily frames of MODIS Terra LST at 0.05°, one group `terra`, 2000 → 2026-09 | **2007 is a one-bin record** (below); two assemblies failed (§9.1); re-fetch 2007, then assemble on a box |
| `snow05` | 28 year folders, 1999–2026 | 3,938 · 22,866,010,257 | **9,634 daily frames** — exactly the 9,634 MOD10C1 granules CMR lists; 2000 holds 63 bins (the record starts 2000-02-24), 2026 holds 51, every year between is complete | one assembly; never dispatched |
| `pheno500` | 13 year folders, 2013–2025 | 10,408 · 40,153,791,221 | 315 tile-frames (one per sinusoidal tile) a year for 2014–2016, 2018 and 2020–2025; **2017 and 2019 declare 0 frames** (below); 2013 is an empty spill | re-fetch 2017 and 2019, then lanes 2001–2013, then assembly |
| `lai500` | `2020/q1`, `2020/q2` | 13,922 · 14,261,879,125 | 6,638 group-frames of Terra LAI/FPAR, 2020-01-01 → 2020-06-30 | quarters `q3`, `q4` (both lanes failed), then assembly |
| `canopy30` | 14 named tile-lanes `2020/g-…` | 12,542 · 32,738,357,598 | 4,176 group-frames = 261 tiles × 16 sub-tiles, the whole 2020 map (13 lanes of 320 frames, one of 16) | one assembly, declaring the fourteen lanes |
| `icesat2` | 9 month lanes `2022/m02, m03, m05, m06, m07, m08, m10, m11, m12` | 1,879 · 65,781,451,215 | 1,879,375,764 land segments (m06 alone: 200,811,690 rows in 200 parts) | lanes `m01`, `m04`, `m09` (all three failed), then a box assembly |
| `gedi` | 5 day lanes `2022/d0601-0603, d0604-0606, d0613-0615, d0619-0621, d0625-0627` | 149 · 6,547,144,303 | 145,485,108 quality shots over 15 days of June 2022 | lanes `d0607-0609`, `d0610-0612`, `d0616-0618`, `d0622-0624`, `d0628-0630`, then assembly |

**Five of `gedi`'s ten lanes are on the Hub, not six**, and **two of
`lai500`'s four**, whatever an internal status note says: the Hub listing and
the run list both say five and two.

**Three collisions, measured.** A lane's window is chosen in bins, and the
bin that straddles a year boundary belongs to the year of its first day, so a
lane can write a one-bin or empty copy of the previous year and overwrite that
year's `done.json` and shard index if it finishes last. The build log records
the failure mode on `snow05` (where it was repaired) and the lane-aware layout
that prevents it landed on 2026-09-20; these three predate or escaped it:

- **`lst05/2007`**: its `done.json` (written 2026-09-18T12:08:46, the same
  second as the 2008–2011 lane's years) declares **4 files, one bin, 5
  frames, 72,993,038 bytes**, while the folder itself holds **149 entries and
  5,660,021,033 bytes** — the 2004–2007 lane's real 2007, orphaned. An
  assembly would read one bin of 2007. The 360 missing days are exactly the
  difference between the parts' 9,227 frames and the 9,587 granules CMR lists
  for MOD11C1.
- **`pheno500/2017` and `pheno500/2019`**: each `done.json` declares **0
  frames, 316 files, 3,234,850 bytes** and was written in the same second as
  the following year's lane, while each folder holds **947 entries** (630
  shard and index files of one bin plus 315 shard indices: 3,961,985,140 bytes
  for 2017, 3,926,672,463 for 2019). The runs that fetched those two years
  are recorded as failed (§9.1); the cause was not read.

## 3 · The channels

Raw units throughout (`normalisation: RAW`); standardise in your loader from
training years only (§8.2). Channel order is `values.npy`'s column order; the
authoritative list, with bounds, is `store.json → channels`. A percentage after
a channel is `store.json → per_channel → fraction` (rows measured / all rows).

### 3.1 · `ghcnd` — daily station weather since 1763 (C = 5)

`TMAX`, `TMIN` (°C, −90 … 65; measured on 40.6 % and 40.5 % of rows), `PRCP`
(mm, 0 … 2,000; 95.8 %), `SNOW` (snowfall, mm, 0 … 2,000; 32.2 %), `SNWD`
(snow depth, mm, 0 … 15,000; 26.8 %). One row is one station-day; `time_s`
is the day at 00:00 UTC (the station's observation hour is ignored);
positions are the station list's, one per station. 3,193,962,548 element
lines read, 490,404,449 of other elements skipped, 461,863 station-days with
all five values blanked dropped. `platforms.json` names every station.

### 3.2 · `igra` — radiosonde soundings at 16 mandatory levels (C = 80)

**80 = 5 quantities × 16 pressure levels: 1,000, 925, 850, 700, 500, 400,
300, 250, 200, 150, 100, 70, 50, 30, 20 and 10 hPa**, in blocks of 16 in this
order: `T_` temperature (°C, −100 … 60), `DPD_` dew-point depression (°C,
0 … 80), `U_` and `V_` wind components (m s⁻¹, ±150), `Z_` geopotential
height (m, −1,000 … 40,000). So `T_500` is temperature at 500 hPa, column 4.
`time_s` is the launch time; where the release time is missing the nominal
hour is used (54,808,612 soundings), and a release more than 12 h from the
nominal hour is moved across midnight (5,087,590). 70,540,492 soundings read,
7,409,207 with no mandatory level dropped, 4,515 stations streamed.

### 3.3 · `tide` — hourly-or-finer relative sea level at tide gauges (C = 1)

`sea_level` (m, −50 … 50), relative to each record's own datum, which the
archive does not expose per record (`platforms.json` says `null`). One row is
one gauge reading from GESLA-4's public contributors, read from the UHSLC
data server; 16,842 records requested, 8,369 with rows in the window.
Co-located records of different contributors are kept apart. **The Great
Lakes gauges are excluded by the ±50 m bound** (their levels are 74–184 m
above a lake datum, which float16 would quantise to 6–12 cm); 199 records fell
out whole and are named in `store.json`.

### 3.4 · `ndbc` — buoy and coastal-station meteorology (C = 10)

`WDIR` (wind direction, ° true), `WSPD` (m s⁻¹), `GST` (gust, m s⁻¹), `WVHT`
(significant wave height, m), `DPD` (dominant wave period, s), `APD` (average
wave period, s), `MWD` (mean wave direction, ° true), `PRES` (hPa, 850 …
1,090), `ATMP` and `WTMP` (air and water temperature, °C). Wind and pressure
are measured on 77–78 % of rows, wave height on 9.3 % (most stations carry no
wave sensor). One row per buoy or C-MAN station report; one position per station. The
tropical moored arrays are excluded (they are family 10.1's). **The record
ends 2025-12-31**: NDBC's historical archive has no current-year file.

### 3.5 · `gliders` — ocean glider profiles at 16 pressures (C = 64)

**64 = 4 quantities × the 16 Roemmich–Gilson pressures of family 8's Argo
store: 10, 30, 50, 100, 150, 200, 300, 400, 500, 700, 900, 1,100, 1,300,
1,500, 1,700 and 1,900 dbar**, in blocks of 16: `T_` (°C, −3 … 40), `S_`
(PSU, 0 … 42), `O2_` (µmol kg⁻¹, 0 … 600), `CHL_` (chlorophyll, mg m⁻³,
−0.5 … 100 — small negative values are the sensors' own offsets, kept). One
row is one dive or climb of one deployment from the IOOS Glider DAC, in
delayed mode where the DAC has the file (689 deployments) and real time
otherwise (1,487). Gliders rarely pass 1,000 dbar, so the deep levels are
mostly NaN: only 17.0 % of all value slots are measured. 2,985,372 profiles in
the files, 2,808,590 kept. The build log notes the store has **44 % fewer rows**
than the adapter projected, "worth checking against the archive's own count".

### 3.6 · `fire` — satellite active-fire detections (C = 4)

`brightness` (K, 200 … 600), `frp` (MW, 0 … 100,000), `confidence` (%, 0 …
100 — **MODIS only**, NaN for every VIIRS row, measured on 18.1 % of rows),
`daynight` (0 or 1). One row per detection from NASA FIRMS across five
satellites: Terra and Aqua MODIS (1 km, 2000 →), Suomi-NPP VIIRS (375 m,
2012 →), NOAA-20 (2018 →) and NOAA-21 (2024 →). `platforms.json` (2,539
bytes) holds each satellite's own `log2_fp` (−4.7986 for a 1 km MODIS pixel,
−6.2136 for a 375 m VIIRS one) because the store's `fp` column carries the
MODIS value for every row. 50.26 GB of source CSV over 3,792 windows,
`rows_read` = `rows_kept` = N.

### 3.7 · `flux` — half-hourly land–air exchange at flux towers (C = 10)

`nee`, `gpp`, `reco` (µmol CO₂ m⁻² s⁻¹; ecosystem respiration is `reco`),
`le`, `h`, `rn` (latent heat, sensible heat and net radiation, W m⁻²), `ta`
(°C), `vpd` (hPa), `swc` (soil water, %), `p` (precipitation, mm). One row is
one tower half-hour (hourly at 8 towers, flagged in `qc`); the time is the
**start** of the half-hour in UTC, converted from the product's local standard
time with each tower's UTC offset. 780 towers read (381 AmeriFlux, 347 ICOS,
52 TERN) of 806 wanted; **26 were not read** (§4.2). Measured fractions: `ta`
and `p` ≈ 1.0, `h` 0.90, `le` 0.89, `nee` 0.81, `gpp` and `reco` 0.73, `rn`
0.71, `swc` 0.69 (not every tower has a soil probe or a net radiometer).

### 3.8 · `chirps05` — pentad rainfall at 0.05° (C = 1, tier G)

`precip` (mm per pentad, 0 … 5,000), CHIRPS v3.0 over land 60 °S–60 °N.
**A CHIRPS pentad is not five days**: each month has six, of 5, 5, 5, 5, 5
and 3–6 days (measured over the record: 2,922 pentads of 5 days, 320 of 6, 35
of 3, 11 of 4). So the store has **two half-bin frames per bin** (F = 2,
216,000 s each) and files each pentad in the half-bin holding its
**midpoint**; one frame per bin is impossible because 14 bins hold two
pentad midpoints (all in February). `tile_grid.json → frame_table` gives
every `(bin, frame)` its pentad's first day, last day and length — **read the
pentad's real dates from it, never from the bin arithmetic**. The first pentad
is 1981-01-01 → 05 (bin −73, frame 1) and the last 2026-08-26 → 31 (bin
3,262, frame 0). Fill (−9,999, outside the land mask) is NaN.

### 3.9 · `lossyear` — the year each 30 m pixel lost its forest (C = 2, tier G)

`lossyear` (uint8: **0** = mapped land that did not lose canopy 2001–2025,
**k = 1 … 25** = loss in year 2000 + k) and `treecover2000` (uint8, canopy
closure in 2000, %). **255 = not mapped land** (ocean, permanent water, the
no-data collar); a pixel is valid exactly where Hansen's datamask is 1. Source
GFC-2025-v1.13; only the 280 of 504 Hansen tiles that carry a `lossyear`
granule (the ones with mapped land) are groups. This is an **annual target**
— one map — filed in bin 3,215 (2026-01-05 → 09), the first bin that begins
after the product's 2025-12-31 period end, so a model never reads it as an
input from inside its own years.

### 3.10 · The eight catalogues (C = 5, `cat_olci` C = 6, tier T)

| idx | name | unit | bounds | what |
|---|---|---|---|---|
| 0 | `cloud` | % | 0 … 100 | the scene's cloud fraction as the producer reports it |
| 1 | `valid` | % | 0 … 100 | the fraction of the footprint with data |
| 2 | `angle` | ° | 0 … 90 | sun **zenith** for an optical scene, radar **incidence** for a SAR one |
| 3 | `log2_area` | log₂(km²) | −10 … 26 | the footprint's area on the sphere — **stored as a logarithm** because float16 overflows at 65,504 and a VIIRS granule is ≈ 7 × 10⁶ km²; `2 ** log2_area` is km² (round-trip error ≤ ≈ 1.2 %) |
| 4 | `sensor` | code | 0 … 255 | which instrument on which platform, and for radar the mode: a key of `store.json → sensor_table`; **255 = a value the table does not list** |
| 5 | `coastal` | 0/1 | 0 … 1 | `cat_olci` only: a coarse coastal flag from a 1° land/ocean mask, a stand-in until the GSHHG ±100 km coast band exists |

**A channel a producer does not publish is NaN for every row**, measured
today from `per_channel`: `cloud` exists in `cat_landsat`, `cat_s2` (99.99 %)
and `cat_hls`; `valid` only in `cat_hls`; `angle` in `cat_landsat` and
`cat_hls`; `cat_s1`, `cat_ecostress`, `cat_nisar` and `cat_viirs` carry
`log2_area` and `sensor` only. What the sensor codes are:

| store | `sensor_table` |
|---|---|
| `cat_landsat` | 4 LANDSAT_4, 5 LANDSAT_5, 7 LANDSAT_7, 8 LANDSAT_8, 9 LANDSAT_9 |
| `cat_s1` | 11/12 S1A IW GRDH/SLC, 21/22 S1B, 31/32 S1C, 41/42 S1D |
| `cat_s2` | 1 S2A, 2 S2B, 3 S2C, 4 S2D |
| `cat_hls` | 1–3 Sentinel-2A/2B/2C MSI, 4 LANDSAT-8/OLI, 5 LANDSAT-9/OLI |
| `cat_ecostress` | 1 ISS/ECOSTRESS |
| `cat_nisar` | 1 … 81: the **polarimetric mode**, two characters per frequency band (SH, SV, DH, DV, QP, QQ, CL, CR, NA), e.g. 21 DHDH, 74 NASV |
| `cat_viirs` | 1 Suomi-NPP, 2 NOAA-20, 3 NOAA-21 |
| `cat_olci` | 1 S3A, 2 S3B, 3 S3C, 4 S3D |

### 3.11 · The parked and not-built stores' declared channels (from the registry)

`lst05` (C = 4): `lst_day`, `lst_night` (°C — float16 is 0.25 K apart at
300 K and 0.03 K near 0 °C), `qc_day`, `qc_night` (bit fields). `snow05`
(C = 2): `snow_cover`, `cloud` — percent 0–100 **or** a class code (107 lake
ice, 111 night, 237 inland water, 239 ocean, 250 cloud-obscured water, 252
Antarctica mask, 253 not mapped). `pheno500` (C = 10): six phenological dates
(`greenup` … `dormancy`, day numbers), `evi_minimum`, `evi_amplitude`,
`evi_area` (EVI2), `qa_overall`. `lai500` (C = 4): `lai`, `fpar`,
`lai_stddev`, `fpar_lai_qc`. `canopy30` (C = 1, GLAD's 2020 map): `canopy_height`, whole
metres of woody vegetation ≥ 3 m. `icesat2` (C = 6): `h_te_best_fit`,
`h_canopy`, `canopy_openness`, `n_ca_photons`, `n_te_photons`,
`segment_landcover`. `gedi` (C = 11): `elev_lowestmode`, `rh25` … `rh98`
(relative heights, m), `cover`, `pai`, `agbd`, `agbd_se` (above-ground biomass
density, Mg ha⁻¹), `sensitivity`. `burned500` (C = 2): `burn_doy`, `qa`.
`refl05` (C = 9): bands `b1`–`b7`, `state_qa_lo`, `state_qa_hi`. `sif`
(C = 6): `sif_740`, `sif_740_daily`, `sif_740_error`, `cloud_fraction`,
`solar_zenith`, `viewing_zenith`. `gbif` / `gbif_nc` (C = 4): `kingdom_code`,
`class_code`, `basis_of_record_code`, `individual_count`. `canopy_ref`,
`cat_palsar`, `cat_biomass_esa`: the catalogue channels of §3.10.

## 4 · How the values were made (so you can trust or reject them)

Each store's full policy is one paragraph in its own `store.json` under
`qc_policy`, and every drop is counted by reason under `counts`. The common
rules: a **row** is dropped only when it has no usable time, no usable
position or nothing measured; a **value** outside its channel's physical
bounds becomes **NaN (255 for a uint8 grid) and is counted — never clipped**;
the source's own quality flags are applied as written, not re-derived; a
channel the source does not publish is NaN, never a guess.

### 4.1 · What `qc` means, per store — it is NOT one scale

| store | `qc` is | how to read it |
|---|---|---|
| `ghcnd` | a flag, 0 or 1 | 0: no used element carried an NCEI quality flag — **"passed the producer's QC", not "not assessed"** as in family 10; 1: at least one did, and that element's value is NaN |
| `igra` | a flag, 0 or 1 | 1: IGRA's quality assurance removed at least one value of the sounding (now NaN) |
| `tide` | GESLA's `flag1` | 0 not assessed, 1 correct; flags 3 and 4 (doubtful, wrong) were dropped because they exceed `qc_keep_max` 2, so this is the one store where that field acts; flag 2 (interpolated) and 5 (missing) and `flag2 = 0` (do not use) rows are dropped too |
| `ndbc` | always 0 | NDBC's historical archive is already quality-controlled and carries no per-value flag |
| `gliders` | a flag, 0 or 1 | 1: the file carries QARTOD flags and samples flagged 3/4/9 were removed before interpolation; 0: no such flags (not assessed). Measured today: 1,703,639 rows at 1, 1,104,951 at 0 |
| `fire` | a **bitmask** | bits 0–2 the platform (1 Terra, 2 Aqua, 3 Suomi-NPP, 4 NOAA-20, 5 NOAA-21); bits 3–4 the VIIRS confidence class (1 low, 2 nominal, 3 high, 0 n/a); bit 5 set for a Near-Real-Time row |
| `flux` | a **bitmask** | bits 0–2 the NEE gap-fill grade (0 measured, 1 good, 2 medium, 3 poor, 7 not reported), bits 3–5 the LE grade on the same scale, bit 6 set for a tower that averages hourly |
| `cat_landsat` | a **baseline code** | 0 L2SP/T1, 1 L2SP/T2, 2 L2SP/RT, 3 L2SR/T1, 4 L2SR/T2, 5 L2SR/RT (USGS processing level and tier) |
| `cat_s1` | a baseline code | IPF version and polarisation pair, e.g. `003.71/VV&VH`, through `store.json → qc_table` |
| `cat_s2` | a baseline code | ESA processing baseline: 0 = 05.00, 1 = 05.10 … 4 = 05.13, 10–30 older baselines |
| `cat_hls` | a version code | 0 = v2.0, 1 = v2.1 |
| `cat_ecostress` | a version code | 2 = v002, 3 = v003 — the **same overpass can appear once per version**; select on it |
| `cat_nisar` | a tier code | 0 BETA/PR, 2 PROVISIONAL/PR, 7 URGENT/UR … (`PR` routine, `UR` urgent response) |
| `cat_viirs` | a version code | 0 = v2 (Suomi-NPP), 1 = v2.1 (NOAA-20 and NOAA-21) |
| `cat_olci` | baseline/timeliness | e.g. 9 = `004/NT`, 10 = `004/NR`; since 2026 an overpass exists as NR and NT, and the store keeps one row, NT where it exists |

In every catalogue a code the table does not list is **255** and counted by
name under `counts → qc_unlisted`. `store.json` says `qc_keep_max: 2` in every
store, inherited from family 10's builder. **It means nothing for `fire` and
`flux` (bitmasks) and for the eight catalogues** (each catalogue's
`store.json` says so itself under `qc_keep_note`); filtering `qc <= 2` on
`fire` throws away every row but Standard-Processing Terra and Aqua, and on
`cat_s2` every baseline but 05.00–05.11. Dispatch on the store, or ignore the column.

### 4.2 · Per store, the source and the ledger

- **`ghcnd`** — NCEI's by-year files, 3.19 billion element lines; `-9999`
  and flagged values NaN and counted. **`igra`** — IGRA v2.2 station archives,
  4,515 of 5,862 stations streamed (1,347 outside the window); `-8888`
  (removed by IGRA's QA) and `-9999` NaN and counted. **`tide`** — per-record
  requests to the UHSLC server, 1,184,136,014 rows read, 79,822,967 flagged
  "do not use" dropped. **`ndbc`** — 16,830 files (5 empty upstream), columns
  read by header name; 1,297,722 all-blank reports, 526,602 repeated
  station-seconds and 1,772,181 rows from unlisted stations dropped.
- **`gliders`** — each deployment's NetCDF read by range requests, only the
  needed variables; 2,176 of 2,349 deployments read (173 with no file, 178
  without the variables, 42 in a refused layout). **`fire`** — FIRMS' area API
  in five-day windows; every day from exactly one source (Standard Processing,
  else Near-Real-Time), both read from FIRMS' own availability endpoint.
- **`flux`** — the FLUXNET Shuttle's three public hubs, no login. **A degraded
  build, recorded in `store.json → degraded`**: 26 towers (CA-ER1, CA-NS1 …
  CA-NS7, US-BZS, US-GL1, US-Ho1, US-Me2, US-ORv, US-PFe, US-Pnp, US-UM3,
  US-UMB, US-Wi0, US-Wi1, US-Wi3 … US-Wi9) ship no BADM file, hence no UTC
  offset, and were **not read at all**; the store was built past them with
  `--allow-missing-years`. One tower, `ID-PaD`, takes `round(lon / 15)` hours
  (`platforms.json → utc_offset_source: "longitude"`); 779 use their BADM
  offset. **The store's own `qc_policy` says 27 towers take the longitude
  offset; the measured state is 26 absent and 1 longitude-derived.**
- **`chirps05`** — 3,288 GeoTIFFs, 57.68 GB in 3,755 s, no gap in the
  listing. **`lossyear`** — 3,360 GeoTIFFs (70.48 GB) read row-band by
  row-band in 43,743 s on a rented machine; 225,141,085,546 mapped-land pixels.
- **The catalogues** walk the producer's listing window by window and
  **refuse a count that does not match the producer's**. `cat_landsat`,
  `cat_s1` and `cat_s2` walked it exactly; `cat_hls`, `cat_ecostress`,
  `cat_viirs` and `cat_nisar` differ by granules starting outside the window
  (2, 9,211, 19,518, 184) and, for `cat_nisar`, 229 scenes listed twice;
  `cat_olci` by the NR/NT merge. A catalogue still being ingested cannot be
  walked to today, which is why `cat_s2` stops 2026-09-15.

### 4.3 · Every store was checked before it was trusted

The builder's `check_store` runs on the assembled store and again after
publishing: every file re-hashed against `store.json`; `N` from the file
sizes; `bin` equal to `floor_divide(time_s, 432000)` on every row and
monotone; the CSR offsets reproduced; per-year counts equal to the fetch
ledger; longitude in [−180, 180); every channel inside its bounds; for a
catalogue, `assets.parquet` holding exactly N rows with a unique `platform`.
For a sharded store it decompresses **every stored tile** and checks pad,
bounds and valid-pixel count against `shard_index.npy`. The publish downloads
every file back and compares its sha256 (`manifest.json → restore`: 6,675
files for `chirps05`, 17,921 for `lossyear`, each with five HTTP range checks
and 50 tiles decompressed).

## 5 · Opening it (numpy for tier P and T; numpy and `zstandard` for tier G)

### 5.1 · A point store, end to end — measured on `gliders`

`gliders` is the smallest public point store (436 MB). Its eleven files were
downloaded with plain `curl -L` in 15.6 s, and this ran against the copy:

```python
import hashlib, json, numpy as np

d = "gliders"                                   # a local copy of tensors/family1_tf/gliders
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

# 4. one pentad through the CSR index: the pentad holding 2020-08-15
EPOCH = np.datetime64("1982-01-01T00:00:00")
t0 = (np.datetime64("2020-08-15T00:00:00") - EPOCH).astype("int64")
b = int(t0 // 432000)
lo, hi = int(off[b - b0]), int(off[b - b0 + 1])

# 5. one row, decoded in units: the first row of that bin with oxygen at 100 dbar
names = [c["name"] for c in meta["channels"]]
units = [c["unit"] for c in meta["channels"]]
rows = np.arange(lo, hi)
r = int(rows[~np.isnan(values[lo:hi, names.index("O2_100")].astype(np.float32))][0])
who = json.load(open(f"{d}/platforms.json"))[str(int(plat[r]))]
print(EPOCH + np.timedelta64(int(time_s[r]), "s"), float(lat[r]), float(lon[r]),
      who["id"], who["institution"], int(qc[r]), fp[r].astype(np.float32).tolist())
v = values[r].astype(np.float32)
for q in ("T", "S", "O2", "CHL"):
    cols = [i for i, n in enumerate(names) if n.startswith(q + "_")]
    print(q, units[cols[0]], [None if np.isnan(v[i]) else round(float(v[i]), 4) for i in cols])
```

What the full script printed today (the listing is its core; it also tallied
`qc` and `fp`):

```
sha256: all 10 files match store.json
N 2808590 C 64 schema 2 time dtype int32
bins 1594 .. 3265 = 1672 bins; 1673 offsets; 1402 live
qc values {0: 1104951, 1: 1703639}
fp distinct [[-4.0, -6.90625]]
bin 2821 starts 2020-08-14T00:00:00 rows 1428910 .. 1437171 = 8261
row 1428950: time 2020-08-14T00:31:23  lat 44.5990  lon -124.9469  platform 992996620819710254 = ce_311-20200708T1723 (OOI Coastal Endurance)  qc 1  fp [-4.0, -6.90625]
   T    [degC] 10:10.74 30:8.367 50:7.961 100:8.086 150:7.75 200:NaN 300:NaN ... 1900:NaN
   S    [PSU] 10:32.38 30:32.47 50:32.88 100:33.78 150:33.94 200:NaN ... 1900:NaN
   O2   [umol/kg] 10:299.2 30:246 50:211.5 100:131.4 150:97.44 200:NaN ... 1900:NaN
   CHL  [mg/m3] 10:6.184 30:0.6177 50:0.2659 100:0.1132 150:0.05905 200:NaN ... 1900:NaN
```

A glider of the Ocean Observatories Initiative's Coastal Endurance array off
Oregon in August: 10.7 °C at 10 dbar, oxygen falling from 299 to 97 µmol kg⁻¹
over 150 dbar, chlorophyll highest at the surface, nothing deeper than 150
dbar in this profile. Every invariant held. The recipe is identical for the
other point stores and the catalogues; only `schema_version` changes the time
dtype, and `mmap_mode="r"` keeps a 47 GB store out of memory. `floor_divide`
on a negative `time_s` floors toward minus infinity, which a pre-1982 bin
needs (C, Java, Go and Rust division truncates toward zero; a port must floor
explicitly); a schema-3 `time_s` is int64 because 1763 is 6,910,963,200 s
before the epoch, and `bin` stays int16, which spans 1533 to 2430.

### 5.2 · A sharded frame in two range reads — executed on `chirps05`

**The recipe.** Given a group prefix `P` (here
`tensors/family1_tf/chirps05/chirps05`) and its `tile_grid.json` (read once;
336,786 bytes because it carries the pentad table):

1. Find the frame — for `chirps05` look the pentad up in `frame_table`
   (§3.8); for ordinary frames `b = floor(seconds / 432000)`,
   `f = (seconds − b·432000) // frame_seconds`. `yyyy` is the year of the
   bin's **first** day, not of the day you asked for.
2. **Range read 1**: `16·n` bytes (`n = n_tiles_y × n_tiles_x`) at
   `index_header_bytes + 16·f·n` of `P/<yyyy>/bin_<b>.idx.npy`, as int64
   `[n_tiles_y, n_tiles_x, 2]`. All offsets −1: the frame is absent — stop.
3. **Range read 2**: the contiguous span from `slab[0, 0, 0]` to
   `slab[−1, −1, 0] + slab[−1, −1, 1]` of `P/<yyyy>/bin_<b>.zst`. Each tile
   with `length > 0` is one zstd frame that decompresses to exactly
   `256 × 256 × C × itemsize` bytes, C order `[row, col, channel]`; a tile of
   length 0 is all missing. Place tile `(ty, tx)` at `256·ty`, `256·tx` and
   crop the pad to `H × W`.

**The code that ran:**

```python
import json, urllib.request, datetime as dt
import numpy as np, zstandard

BASE = ("https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/"
        "tensors/family1_tf/chirps05/chirps05")

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
cols = tg["frame_table_columns"]      # bin, frame, year, month, pentad, first_day, last_day, days
row = next(r for r in tg["frame_table"] if r[cols.index("first_day")] == "2015-07-01")
b, f = row[cols.index("bin")], row[cols.index("frame")]
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
rain = frame[:H, :W, 0].astype(np.float32)                       # mm per pentad; pad dropped

g = tg["grid"]                                   # extent = [west, south, east, north], dy < 0
def cell(lat, lon):
    return int((lat - g["extent"][3]) / g["dy"]), int((lon - g["extent"][0]) / g["dx"])
```

**What it returned, today** (pentad 2015-07-01 → 05, 3.4 s wall from this
sandbox):

```
pentad 2015-07-01 .. 2015-07-05 (5 days) -> bin 2447 frame 0 -> 2015/bin_2447.zst
index slab 4640 bytes; 191 stored tiles of 290; shard bytes 0 .. 6,979,549 = 6,979,549
valid 4,848,282 of 17,280,000 pixels (0.2806); min 0 median 3.58 p99 98.56 max 601 mm
Mumbai / Western Ghats   (19.0, 73.2) row 820 col 5063: pixel 26.44 mm; 0.5-degree box: 375 valid, mean 16.85 mm
Sahel, Niamey            (13.5, 2.1) row 930 col 3641: pixel 16.05 mm; 0.5-degree box: 400 valid, mean 13.94 mm
Central Amazon, Manaus   (-3.1, -60.0) row 1262 col 2400: pixel 5.777 mm; 0.5-degree box: 400 valid, mean 3.248 mm
Sahara, Tamanrasset      (22.8, 5.5) row 744 col 3710: pixel 0 mm; 0.5-degree box: 400 valid, mean 0.01355 mm
Zurich                   (47.4, 8.5) row 252 col 3770: pixel 3.367 mm; 0.5-degree box: 400 valid, mean 6.146 mm
ranges 2 bytes transferred 6984189
```

**Two range reads, 6,984,189 bytes** (4,640 of index, 6,979,549 of tiles) for
one global pentad, against 34,560,000 bytes uncompressed: 191 of 290 tiles
hold land, 28.1 % of pixels are valid. The monsoon is where it should be: 26 mm
on the Western Ghats in five days, 16 mm over the Sahel, none over the central
Sahara (Mumbai's box has 375 valid pixels; the rest is sea). This shard is the
whole frame because its second half-bin holds no pentad.

**Independently checked:** the whole shard and its index were downloaded and
re-hashed equal to `store.json`; frame 0 decoded tile by tile is
**bit-identical** to the two-read result (NaN-aware), its pad is all NaN,
frame 1 is absent, and `shard_index.npy`'s row for bin 2,447 reads
`frames_present 1`, `frame_mask 1`, `tiles_stored 191`, `nbytes 6,979,549`,
`valid_pixels 4,848,282` — digit for digit.

### 5.3 · One tile rather than a frame — executed on `lossyear`

For one **tile**, the two reads shrink to 16 bytes at
`index_header_bytes + 16·((f·n_tiles_y + ty)·n_tiles_x + tx)` and then that
tile's `length` bytes; a place maps to `row = floor((lat − north)/dy)`,
`col = floor((lon − west)/dx)`, `ty = row // 256`, `tx = col // 256`, from the
group's `tile_grid.json → grid`. Executed on group `10S_070W_r0c3` (12.5–10 °S,
62.5–60 °W, Rondônia) at 11 °S, 61.5 °W, §5.2's helpers, `BASE` = `…/lossyear/10S_070W_r0c3`:

```python
tg = json.loads(get("tile_grid.json"))
nty, ntx, T, C, g = tg["n_tiles_y"], tg["n_tiles_x"], tg["tile"], tg["C"], tg["grid"]
row = int((-11.0 - g["extent"][3]) / g["dy"]); col = int((-61.5 - g["extent"][0]) / g["dx"])
ty, tx, f = row // T, col // T, 0
o, ln = np.frombuffer(get_range("2026/bin_3215.idx.npy",
                                tg["index_header_bytes"] + 16 * ((f * nty + ty) * ntx + tx), 16), "<i8")
tile = np.frombuffer(zstandard.ZstdDecompressor().decompress(
    get_range("2026/bin_3215.zst", int(o), int(ln)), max_output_size=T * T * C), np.uint8).reshape(T, T, C)
ly, tc = tile[..., 0], tile[..., 1]
valid = ly != tg["missing"]                                   # 255 = not mapped land
yrs, n = np.unique(ly[valid], return_counts=True)
lost = {2000 + int(y): int(c) for y, c in zip(yrs, n) if y > 0}   # int() first: uint8 overflows
```

```
grid extent [-62.5, -12.5, -60.0, -10.0]; place row 4000 col 4000 -> tile (15, 15); entry offset 11,460,294 length 39,478
valid (mapped land) 65,536 of 65,536; tree cover 2000 mean 40.7 %
never lost: 51,205 px; lost by year: {2001: 2381, 2002: 1666, 2003: 889, 2004: 556, 2005: 959, 2006: 601, 2007: 37, 2008: 1298, 2009: 101, 2010: 354, 2011: 122, 2012: 311, 2013: 28, 2014: 81, 2015: 48, 2016: 319, 2017: 205, 2018: 91, 2019: 1051, 2020: 604, 2021: 546, 2022: 1163, 2023: 340, 2024: 153, 2025: 427}
lost in total 14,331 px = 21.9% of the tile; pixel at the place: lossyear 22 treecover 97 %
```

**Two range reads, 39,494 bytes** for 7 × 7 km of 30 m forest history: 21.9 %
of the tile lost its canopy since 2001, the pixel at the place in 2022. The
first run of this code failed with `OverflowError: Python integer 2000 out of
bounds for uint8` — `2000 + y` on a numpy uint8 — which is why the last line
converts first (§10).

### 5.4 · A catalogue, end to end — measured on `cat_nisar`, and followed to the producer

The whole of `cat_nisar` (8.9 MB, eleven files) was downloaded and the §5.1
invariants held unchanged (the recipe is the same; `time_s` is int32). Then:

```python
import numpy as np, json, pyarrow.parquet as pq

meta = json.load(open("cat_nisar/store.json"))
values = np.load("cat_nisar/values.npy").astype(np.float32)          # [N, 5]
plat, qc = np.load("cat_nisar/platform.npy"), np.load("cat_nisar/qc.npy")
lat, lon = np.load("cat_nisar/lat.npy"), np.load("cat_nisar/lon.npy")
names = [c["name"] for c in meta["channels"]]
sensor = {int(k): v for k, v in meta["sensor_table"].items()}
qct = {int(k): v for k, v in meta["qc_table"].items()}
assert len(np.unique(plat)) == len(plat)                           # one hash per scene

# every frame whose footprint centre is over Switzerland
m = (lat > 45.8) & (lat < 47.8) & (lon > 5.9) & (lon < 10.5)
r = int(np.flatnonzero(m)[0])

# join to the asset table and build the producer's URL
t = pq.read_table("cat_nisar/assets.parquet")
i = int(np.flatnonzero(t.column("platform").to_numpy() == plat[r])[0])
a = {k: t.column(k)[i].as_py() for k in t.column_names}
urls = [a["base_url"] + "/" + n.replace("{id}", a["stac_id"]) for n in a["asset_set"].split()]
```

What the full script printed today (it also tallied each channel, `qc` and
the sensor modes):

```
sha256: all 10 files match store.json
N 138804 C 5 channels ['cloud', 'valid', 'angle', 'log2_area', 'sensor']
  cloud      finite       0
  valid      finite       0
  angle      finite       0
  log2_area  finite 138,804  min 11.64 median 15.88 max 16.48
  sensor     finite 138,804  min 1 median 21 max 255
qc   {'BETA/PR': 23527, 'PROVISIONAL/PR': 114613, 'URGENT/UR': 664}
mode {'SHSH': 15277, 'SHSV': 387, 'SHNA': 8353, 'SVSH': 3802, 'SVSV': 8, 'SVNA': 6, 'DHDH': 67133, 'DHNA': 14,
      'DVDV': 620, 'DVNA': 48, 'QPDH': 1803, 'NASH': 170, 'NASV': 36854, 'NADV': 4267, '255 (unlisted)': 62}
footprint area km2: median 60,423  p5 13,409  p95 66,250
46 frames centred in 45.8-47.8 N, 5.9-10.5 E; the first:
row 2: time 2025-10-12T18:26:04  bin 3198  centre 46.4722, 10.3204  platform 852937653616436894  qc 0 = BETA/PR  fp [2.693359375, -13.3984375]
   cloud nan  valid nan  angle nan  area 64,132 km2 (log2 15.97)  sensor 21 = DHDH
{'platform': 852937653616436894,
 'stac_id': 'NISAR_L2_PR_GCOV_002_109_D_065_4005_DHDH_A_20251012T182604_20251012T182640_X05010_N_F_J_001',
 'collection': 'NISAR_L2_GCOV_BETA_V1',
 'base_url': 'https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GCOV_BETA_V1/NISAR_L2_PR_GCOV_..._001',
 'asset_set': '{id}.h5'}
url: https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GCOV_BETA_V1/<stac_id>/<stac_id>.h5
```

**What a catalogue row is, then:** not a measurement of the ground but a
record that *a scene exists* — here a 64,000 km² L-band frame over the eastern
Alps, acquired 2025-10-12 at 18:26 UTC in polarimetric mode DHDH, from the
beta processing tier. Three of its five channels are NaN because a radar has
no cloud fraction and ASF publishes neither a valid fraction nor an incidence
angle for this product; 62 rows carry sensor 255 (modes the table lacks).

**How a consumer uses it to fetch imagery.** Select rows by bin (time),
footprint centre and area (place), `sensor`, `qc` and, for optical stores,
`cloud`; join the selected `platform` values to `assets.parquet`; build each
URL by the rule of §2.4; download from the producer **with your own account
there**. What the producers answered an anonymous request today (one request
each, the first kilobyte):

| store's host | anonymous answer | what you need |
|---|---|---|
| `nisar.asf.earthdatacloud.nasa.gov` (`cat_nisar`) | **401** after a redirect to `urs.earthdata.nasa.gov/oauth/authorize` | a NASA Earthdata Login account |
| `data.laadsdaac.earthdatacloud.nasa.gov` (`cat_viirs`) | **401** after the same redirect | a NASA Earthdata Login account |
| `landsatlook.usgs.gov` (`cat_landsat`) | **200 with an HTML page** — the redirect lands on `ers.cr.usgs.gov/login` | a USGS EROS Registration System account; **a 200 here is a login page, not the file** |
| `catalogue.dataspace.copernicus.eu …/$value` (`cat_olci`, and by the build log `cat_s1`, `cat_s2`) | **401** from `download.dataspace.copernicus.eu` | a Copernicus Data Space account |
| LP DAAC (`cat_hls`, `cat_ecostress`) | not measured | by the build log, an Earthdata Login account |

The catalogues exist so a model can ask "which images exist for this cell in
this week, how cloudy, from which sensor" without anyone copying a petabyte;
the scenes a model finally reads are to be encoded into tokens by a separate,
later step (the design note's "local-codec tokens"), which is not built.

## 6 · Search and indexing

**Tier P** (and T, which is tier P) is searched exactly as family 10's stores
are: pick the anchor's bin, walk the CSR slices of that bin and the bins
before it (never after — a forecast must not see the future), compute
great-circle offsets, keep the k nearest inside a radius and an age bound, and
return **miss tokens** for empty slots so a batch is never ragged. `dt` is
measured from the **end** of the anchor's pentad. **No search radii have been
measured for any family-1.0.tf store yet.** Family 10 found the k nearest
tokens usually come from **one platform**; expect it strongest in `flux` (48
half-hours a day at one tower), `tide` and `ndbc` (hourly at one site) and
`fire` (one pass lights adjacent pixels), and de-duplicate by `platform` if
independent information is wanted — except in a catalogue, where `platform`
is the scene. Empty bins: `tide` 30 of 16,549, `igra` 461 of 8,873, `gliders`
270 of 1,672, `cat_landsat` 89 of 3,220, `ghcnd` none. An empty bin is
`off[i] == off[i+1]`, not an error.

**Tier G** is addressed, not searched: latitude and longitude give the tile
(§5.3), the date gives the bin and frame (for `chirps05`, through the frame
table), and a cone of a few hundred kilometres touches a handful of tiles
(one `chirps05` tile is 12.8° on a side; one `lossyear` tile 0.064°, about
7 km). `shard_index.npy` says per bin which frames exist, how many tiles are
stored and the valid fraction per channel, so a loader can skip a bin without
touching its shard. A gridded value becomes one token at the **pixel centre**
with its own footprint — `chirps05` `(−2.322, 0.0)`, 5.57 km and five days;
`lossyear` `(−9.966, 6.190)`, 30 m and one year — never copied into
neighbouring cells. For `lossyear`, resolve the group from the anchor by the
Hansen tile name (the 10° tile's north-west corner, `00N_060W`) and the
sub-tile's row and column; each of the 4,480 groups has its own
`tile_grid.json`.

## 7 · Relationship to the other families

### 7.1 · Family 10.2 — same contract, same tier-P layout, a `siblings` line

Family 10.2 (`tensors/family10_2/family10.json`) is the programme's
granularity-aware family at the 0.25° scale: drifters, tropical moorings, ship
CO₂, along-track sea level, fishing effort, family 8's Argo store and the
family-7.2 tensor by reference. Family 1.0.tf **inherits its contract by
reference**: the epoch, the bin rule, the footprint token, the tier letters
and dispatch-on-`tier`, and — for tiers P and T — the nine arrays of §2.2 byte
for byte, so a family-10 point store and a 1.0.tf point store open with the
same code. The widenings are **schema 3** (`ghcnd`, `igra`, `tide` start
before 1914: int64 `time_s`), a Parquet sidecar for the catalogues and the
sharded reader for tier G. `gliders` uses family 8's 16 Argo pressures;
`ndbc` excludes the tropical moorings family 10.1's `gtmba` holds.

**The `siblings` line.** Family 10.2's registry is designed to gain a
`siblings` key naming the three family-1 registries. **Measured today, the
published family-10.2 registry does not carry it** (`generated_utc`
2026-09-16T19:49:38Z); reach this family through its own registry path.

### 7.2 · Family 1.gf — the ocean-and-atmosphere sibling at the same scale

Family 1.gf (`tensors/family1_gf/family1gf.json`, 13 stores) is the global
ocean and atmosphere at the same scale, with identical tier-P and tier-G
layouts; tier T exists only here. No store is in both registries (the
registry builder asserts it); the two meet at the coast, where `ndbc`, `tide`
and `gliders` sit beside 1.gf's `wod`, `oceansites` and `icoads`.

### 7.3 · Family 0.9.tf — 1.0.tf with the imagery replaced by an embedding

`tensors/family09_tf/family09tf.json` (measured: 3,589 bytes, sha256
`874574a47677cd8dd0fa08fcd17765534796fa6c5df9a6d0795f7f9d31f368fc`, generated
2026-09-22T09:24:59Z, `n_groups` **0**) lists **all 33 stores of 1.0.tf by
reference** in its `inherits_block` — "the SAME BYTES under the same path" —
and builds nothing of its own yet; its pooled AlphaEarth embeddings (Google's
learned summaries of satellite imagery) are designed and not started.

### 7.4 · Family 7.2 — the 0.25° grid

Family 7.2 (`tensors/family7_global025_pentad_l2/`) is the gridded input
tensor: 721 × 1,440 cells, **south first**, one row per pentad, values
**z-scored**; this family's values are raw and its grids north first. Its land
quantities (2 m air temperature, precipitation, snow, soil, skin temperature)
live in its 1° group `g100`, from reanalysis; `ghcnd`, `chirps05`, `snow05`
and `lst05` are independent measurements at their own resolution and stay
beside the grid rather than replacing it.

## 8 · How to use it in training

### 8.1 · The unit of data

An **anchor** (a place and a pentad) plus, per store, its k nearest
observations as tokens. The token is family 10's: `value[C]`, `mask[C]`,
`dx_km`, `dy_km`, `dt_days` (age, ≥ 0), `log2_fp`, `log2_dt`, `n_R` (how many
observations lay inside the search bounds), a source id and a channel id,
footprint fields clamped to ±12. A gridded pixel is one token at its centre;
a catalogue row is a token that says a scene exists and what kind it is.

### 8.2 · Standardise in the loader, from training years only

The stores are raw on purpose. Compute each channel's mean and standard
deviation over **training bins only**. Code channels (`sensor`, catalogue
`qc`, `snow05`'s class codes, `lossyear`'s year index) are categories to
embed, not to standardise; `log2_area` is already a logarithm.

### 8.3 · Missingness is a feature

NaN is "not measured", never zero: zero is a real dry pentad and a real "did
not lose forest". `mask` and `n_R` are inputs. `gliders` is 83 % NaN (gliders
stay shallow), `chirps05` 72 % (sea), and in the catalogues whole channels
are NaN because the producer does not publish them. Do not impute.

### 8.4 · The holdout protocol, and the targets

The programme's protocol: 2009, 2017 and 2023 are development years; train on
bins up to 2020, test on 2021–2024. The stores carry the whole record — 1763
onward in `ghcnd` — and the loader restricts. `lossyear` is a **target**: its
one frame sits in bin 3,215 so that no input window inside 2001–2025 can read
it; to score year-by-year loss, read `lossyear == k` against inputs that end
before year 2000 + k. `flux`'s towers are the design's held-out carbon target,
and the 26 unread towers are simply absent. `ndbc` ends 2025, `cat_s2`
2026-09-15.

## 9 · What is published and verified (as of 2026-09-22)

Every run below is a run of the family1-build workflow (list, fetch,
assemble, publish with a restore check, check), identified from its run list
as it stood on 2026-09-22. "Verified" is when the build log read the store
back from the Hub; the last column is what this page measured.

| store | built by | built at (UTC) | verified | measured today |
|---|---|---|---|---|
| `ghcnd` | four hosted year-lanes, then [#89 (ghcnd: streaming assembly on a rented machine, 63 min, 48.9 GB peak memory)](https://github.com/blauewelt/earth/actions/runs/35285296155) | 2026-09-17T23:34 | 2026-09-17 | 10 of 10 sha256 against the Hub's record, sizes vs N, first/last `time_s` and `bin` |
| `igra` | two hosted lanes, then [#41 (igra: hosted streaming assembly from parked parts)](https://github.com/blauewelt/earth/actions/runs/35245039767) | 2026-09-17T16:20 | 2026-09-17 | as `ghcnd`, plus `platforms.json` downloaded and hashed |
| `tide` | three hosted lanes, then [#82 (tide: hosted streaming assembly, 21 min)](https://github.com/blauewelt/earth/actions/runs/35267283645) | 2026-09-17T20:12 | 2026-09-17 | as `igra` |
| `ndbc` | four hosted lanes, then [#74 (ndbc: hosted streaming assembly, 45 min)](https://github.com/blauewelt/earth/actions/runs/35256713945) | 2026-09-17T18:34 | 2026-09-17 | as `igra` |
| `gliders` | seven hosted lanes, then [#76 (gliders: hosted assembly, 4 min)](https://github.com/blauewelt/earth/actions/runs/35258595643) | 2026-09-17T18:29 | 2026-09-17 | whole store downloaded, 10 of 10 sha256 re-hashed equal, invariants and recipe (§5.1) |
| `fire` | six hosted lanes, then [#291 (fire: hosted assembly from 27 years of parts, 43 min)](https://github.com/blauewelt/earth/actions/runs/35515309907) | 2026-09-20T14:25 | 2026-09-20 | as `igra` |
| `flux` | [#355 (flux: all five stages on one hosted runner, past 26 unreadable towers)](https://github.com/blauewelt/earth/actions/runs/35527954902) | 2026-09-20T19:34 | not in the build log | as `igra`, plus `degraded` and `platforms.json` read (§4.2) |
| `chirps05` | five decade lanes, then [#127 (chirps05: hosted assembly from parts, 66 min)](https://github.com/blauewelt/earth/actions/runs/35313457413) | 2026-09-18T06:48 | 2026-09-18 | manifest = store.json sha256 (6,674 of 6,674), `tile_grid.json`, `shard_index.npy` and one shard re-hashed, one frame decoded two ways (§5.2) |
| `lossyear` | [#237 (lossyear: all five stages on the rented machine, one bin, 43,743 s of fetch)](https://github.com/blauewelt/earth/actions/runs/35333590380), after probe [#94 (lossyear: five Hansen tiles)](https://github.com/blauewelt/earth/actions/runs/35310891330) | 2026-09-18T22:23 | not in the build log's table | manifest = store.json sha256 (17,920 of 17,920), one tile decoded (§5.3) |
| `cat_landsat` | built and published from a sandbox (the hosted pool was full), no run | 2026-09-18T12:26 | 2026-09-18 | 10 of 10 sha256, `assets.parquet` downloaded and hashed |
| `cat_s1` | from a sandbox, no run | 2026-09-18T12:07 | 2026-09-18 | 10 of 10 sha256 against the Hub's record |
| `cat_s2` | eleven hosted year lanes, then [#285 (cat_s2: hosted assembly, 11 min)](https://github.com/blauewelt/earth/actions/runs/35469018416) | 2026-09-19T21:07 | 2026-09-19 | as `cat_s1` |
| `cat_hls` | eleven hosted year lanes, then [#274 (cat_hls: hosted assembly, 9 min)](https://github.com/blauewelt/earth/actions/runs/35444141756) | 2026-09-19T13:00 | 2026-09-19 | as `cat_s1` |
| `cat_ecostress` | eight hosted year lanes, then [#267 (cat_ecostress: hosted assembly, 3 min)](https://github.com/blauewelt/earth/actions/runs/35441461507) | 2026-09-19T11:58 | 2026-09-19 | as `cat_s1` |
| `cat_nisar` | from a sandbox, no run | 2026-09-18T09:21 | 2026-09-18 | whole store downloaded and re-hashed, decoded and joined to its assets (§5.4) |
| `cat_viirs` | from a sandbox, no run | 2026-09-18T09:28 | 2026-09-18 | as `cat_landsat` |
| `cat_olci` | from a sandbox, no run | 2026-09-18T09:31 | 2026-09-18 | as `cat_landsat` |
| `tide_private` (private) | [#5 (tide_private: all five stages, hosted, private track, 9 min)](https://github.com/blauewelt/earth/actions/runs/35235971748) | 2026-09-17 | 2026-09-17 | **not measured** — private 401 |

The lanes, for completeness (each a hosted job that fetched a window and parked it under `partials/`):

| store | lanes |
|---|---|
| `ghcnd` | [#7 (1763–1899)](https://github.com/blauewelt/earth/actions/runs/35236978566), [#15 (1900–1967)](https://github.com/blauewelt/earth/actions/runs/35237442229), [#24 (1968–1998)](https://github.com/blauewelt/earth/actions/runs/35239789928), [#56 (1999–2026)](https://github.com/blauewelt/earth/actions/runs/35250566582) |
| `igra` | [#18 (1905–1984)](https://github.com/blauewelt/earth/actions/runs/35237449146), [#27 (1985–2026)](https://github.com/blauewelt/earth/actions/runs/35239797586) |
| `tide` | [#19 (1800–1989)](https://github.com/blauewelt/earth/actions/runs/35237451670), [#32 (1990–2009)](https://github.com/blauewelt/earth/actions/runs/35241322028), [#59 (2010–2026)](https://github.com/blauewelt/earth/actions/runs/35252648245); failed assemblies [#77 (record list timed out)](https://github.com/blauewelt/earth/actions/runs/35259639466), [#79 (same)](https://github.com/blauewelt/earth/actions/runs/35260684413), [#80 (same)](https://github.com/blauewelt/earth/actions/runs/35263339110) |
| `ndbc` | [#44 (1970–2011)](https://github.com/blauewelt/earth/actions/runs/35246491348), [#45 (2012–2017)](https://github.com/blauewelt/earth/actions/runs/35246493763), [#58 (2018–2021)](https://github.com/blauewelt/earth/actions/runs/35250571411), [#63 (2022–2025)](https://github.com/blauewelt/earth/actions/runs/35253670963) |
| `gliders` | [#20 (2003–2012)](https://github.com/blauewelt/earth/actions/runs/35238596351), [#38 (2013–2015)](https://github.com/blauewelt/earth/actions/runs/35243928352), [#60 (2016–2017)](https://github.com/blauewelt/earth/actions/runs/35252650442), [#64 (2018–2019)](https://github.com/blauewelt/earth/actions/runs/35253672868), [#67 (2020–2021)](https://github.com/blauewelt/earth/actions/runs/35254666153), [#69 (2022–2023)](https://github.com/blauewelt/earth/actions/runs/35255568644), [#71 (2024–2026)](https://github.com/blauewelt/earth/actions/runs/35256636498) |
| `fire` | [#233 (2000–2006)](https://github.com/blauewelt/earth/actions/runs/35332447511), [#264 (2007–2013)](https://github.com/blauewelt/earth/actions/runs/35441355313), [#284 (2014–2017)](https://github.com/blauewelt/earth/actions/runs/35464419111), [#286 (2018–2020)](https://github.com/blauewelt/earth/actions/runs/35476628610), [#288 (2021–2023)](https://github.com/blauewelt/earth/actions/runs/35488018089), [#289 (2024-01 → 2026-09)](https://github.com/blauewelt/earth/actions/runs/35499764487) |
| `flux` | failed one-stream fetches [#266 (towers without a BADM file stopped the pass)](https://github.com/blauewelt/earth/actions/runs/35441356887), [#275 (same)](https://github.com/blauewelt/earth/actions/runs/35446637521) |
| `chirps05` | [#95 (1981–1990)](https://github.com/blauewelt/earth/actions/runs/35310896703), [#96 (1991–2000)](https://github.com/blauewelt/earth/actions/runs/35310902020), [#97 (2001–2010)](https://github.com/blauewelt/earth/actions/runs/35310908258), [#98 (2011–2020)](https://github.com/blauewelt/earth/actions/runs/35310914521), [#99 (2021–2026)](https://github.com/blauewelt/earth/actions/runs/35310920088) |
| `cat_ecostress` | [#200 (2018-07 → 2019)](https://github.com/blauewelt/earth/actions/runs/35327140301) and one lane a year 2020–2026, #201–#207, e.g. [#207 (2026)](https://github.com/blauewelt/earth/actions/runs/35327164011) |
| `cat_hls` | [#219 (2013-04 → 2015)](https://github.com/blauewelt/earth/actions/runs/35327216095), [#220 (2016–2017)](https://github.com/blauewelt/earth/actions/runs/35327219262), [#252 (2018)](https://github.com/blauewelt/earth/actions/runs/35341317021), [#240 (2019)](https://github.com/blauewelt/earth/actions/runs/35334205877), [#249 (2020)](https://github.com/blauewelt/earth/actions/runs/35340437345), [#251 (2021)](https://github.com/blauewelt/earth/actions/runs/35341313364), [#261 (2022)](https://github.com/blauewelt/earth/actions/runs/35441349993), [#226 (2023)](https://github.com/blauewelt/earth/actions/runs/35327246074), [#227 (2024)](https://github.com/blauewelt/earth/actions/runs/35327250693), [#262 (2025)](https://github.com/blauewelt/earth/actions/runs/35441350723), [#263 (2026)](https://github.com/blauewelt/earth/actions/runs/35441351582) |
| `cat_s2` | [#245 (2015-07 → 2016)](https://github.com/blauewelt/earth/actions/runs/35340202778), [#246 (2017)](https://github.com/blauewelt/earth/actions/runs/35340206551), [#268 (2018)](https://github.com/blauewelt/earth/actions/runs/35441550259), [#248 (2019)](https://github.com/blauewelt/earth/actions/runs/35340214945), [#254 (2020)](https://github.com/blauewelt/earth/actions/runs/35345007326), [#269 (2021)](https://github.com/blauewelt/earth/actions/runs/35441551105), [#272 (2022)](https://github.com/blauewelt/earth/actions/runs/35443153465), [#276 (2023)](https://github.com/blauewelt/earth/actions/runs/35449073690), [#277 (2024)](https://github.com/blauewelt/earth/actions/runs/35449074912), [#279 (2025)](https://github.com/blauewelt/earth/actions/runs/35450588708), [#283 (2026-01-01 → 09-15)](https://github.com/blauewelt/earth/actions/runs/35463314353) |

### 9.1 · The parked stores: runs and what remains

| store | what ran | what remains |
|---|---|---|
| `lst05` | probe [#150 (2015-07, 31 frames, 147.3 GB projected for the record)](https://github.com/blauewelt/earth/actions/runs/35321012014); seven lanes, the last [#176 (2024–2026)](https://github.com/blauewelt/earth/actions/runs/35323865674); assembly [#292 (on the rented machine; died when the disk filled)](https://github.com/blauewelt/earth/actions/runs/35522721838) and [#425 (the same with the cache freed; failed after 52 min, cause not read)](https://github.com/blauewelt/earth/actions/runs/35535980334) | re-fetch 2007 (§2.7), then a box assembly (≈ 139 GB of parts) |
| `snow05` | probe [#149 (2015-02, 23.6 GB projected)](https://github.com/blauewelt/earth/actions/runs/35321006304); lanes [#241 (2000–2008, the re-run that repaired 2008)](https://github.com/blauewelt/earth/actions/runs/35334296920), [#160 (2009–2017)](https://github.com/blauewelt/earth/actions/runs/35321266318), [#161 (2018–2026)](https://github.com/blauewelt/earth/actions/runs/35321271198) | one assembly (≈ 23 GB); parts are complete |
| `pheno500` | probe [#293 (2020-06, five tiles, 15 s a frame)](https://github.com/blauewelt/earth/actions/runs/35526534209); year lanes #340–#351 (2014–2025), of which [#343 (2017)](https://github.com/blauewelt/earth/actions/runs/35527754684) and [#345 (2019)](https://github.com/blauewelt/earth/actions/runs/35527759161) failed | re-fetch 2017 and 2019; lanes 2001–2013; assembly (≈ 240 GB for 25 years by the probe) |
| `lai500` | probe [#294 (2020-07, 4.7 s a frame, ≈ 90 GB a year)](https://github.com/blauewelt/earth/actions/runs/35526536880); quarter lanes [#369 (q1)](https://github.com/blauewelt/earth/actions/runs/35530244515), [#370 (q2)](https://github.com/blauewelt/earth/actions/runs/35530247076) green, [#371 (q3)](https://github.com/blauewelt/earth/actions/runs/35530249596) and [#372 (q4)](https://github.com/blauewelt/earth/actions/runs/35530252107) failed | q3, q4, then an assembly starting `2020-01-02`; 2021–2024 are a decision (≈ 90 GB a year) |
| `canopy30` | probe from a sandbox (one tile, ≈ 33 GB projected); fourteen tile-subset lanes #389–#402, all green, e.g. [#389](https://github.com/blauewelt/earth/actions/runs/35530534541) … [#402](https://github.com/blauewelt/earth/actions/runs/35530573641) | one assembly declaring the fourteen lanes |
| `icesat2` | probes [#297 (range reads, too slow)](https://github.com/blauewelt/earth/actions/runs/35526546790) and [#298 (whole granules, 36.6 MB/s)](https://github.com/blauewelt/earth/actions/runs/35526905112); month lanes green: [#404 m02](https://github.com/blauewelt/earth/actions/runs/35530581163), [#411 m03](https://github.com/blauewelt/earth/actions/runs/35530664577), [#405 m05](https://github.com/blauewelt/earth/actions/runs/35530584180), [#413 m06](https://github.com/blauewelt/earth/actions/runs/35530670483), [#414 m07](https://github.com/blauewelt/earth/actions/runs/35530672848), [#406 m08](https://github.com/blauewelt/earth/actions/runs/35530586756), [#408 m10](https://github.com/blauewelt/earth/actions/runs/35530593382), [#409 m11](https://github.com/blauewelt/earth/actions/runs/35530596399), [#410 m12](https://github.com/blauewelt/earth/actions/runs/35530599100); failed [#403 m01](https://github.com/blauewelt/earth/actions/runs/35530577658), [#412 m04](https://github.com/blauewelt/earth/actions/runs/35530667540), [#407 m09](https://github.com/blauewelt/earth/actions/runs/35530590292) | m01, m04, m09; a box assembly (≈ 70 GB store for 2022) |
| `gedi` | probe [#315 (2022-06, six granules, read fraction 0.111 at 2.0 MB/s → 222 GB and 492 h for 2022)](https://github.com/blauewelt/earth/actions/runs/35527223947); three-day lanes green: [#415](https://github.com/blauewelt/earth/actions/runs/35530713885), [#416](https://github.com/blauewelt/earth/actions/runs/35530716356), [#419](https://github.com/blauewelt/earth/actions/runs/35530725607), [#421](https://github.com/blauewelt/earth/actions/runs/35530732851), [#423](https://github.com/blauewelt/earth/actions/runs/35530738602); failed [#418 (06-10 → 12)](https://github.com/blauewelt/earth/actions/runs/35530722444), [#422 (06-22 → 24)](https://github.com/blauewelt/earth/actions/runs/35530735890), [#424 (06-28 → 30)](https://github.com/blauewelt/earth/actions/runs/35530740955); cancelled [#417 (06-07 → 09)](https://github.com/blauewelt/earth/actions/runs/35530719394), [#420 (06-16 → 18)](https://github.com/blauewelt/earth/actions/runs/35530729202) | the five missing lanes, then a hosted assembly (≈ 18.5 GB for the month); the year waits for faster granule reads |

### 9.2 · The not-built stores, and the measured reason

| store | what it would be | the measured reason it does not exist |
|---|---|---|
| `burned500` | MODIS burned-area date, 500 m, monthly | probe [#387 (2019-08)](https://github.com/blauewelt/earth/actions/runs/35530456612) went green after [#356](https://github.com/blauewelt/earth/actions/runs/35527956523) failed, but **no probe file is committed and the registry's `probe` is null**, so its size is still the note's ≈ 10 GB; no lane dispatched |
| `refl05` | MODIS daily surface reflectance, 0.05°, bands 1–7 | probe [#232 (2015-07, 31 frames, 586 MB fetched a frame)](https://github.com/blauewelt/earth/actions/runs/35332441752): **2.47 TB for the Terra record** against the note's 700 GB. A decision for the owner: 2015–2026 only (≈ 900 GB) or nothing |
| `sif` | TROPOMI chlorophyll fluorescence, one row per sounding | probed from a sandbox (2023-07, all 31 days: 101,722,256 rows, 39 bytes a stored row): **46.8 GB a year**, ≈ 390 GB to today; no lane has been dispatched. Its OCO half is blocked on NASA's GES DISC application approval |
| `gbif` | GBIF occurrences, CC0 and CC BY 4.0 | probed from a sandbox (one part of eight): **2.59 × 10⁹ rows, 101.1 GB**; needs a machine with room for the assembly; no lane has been dispatched |
| `canopy_ref` | a catalogue of ETH's 10 m and Meta's ~1 m canopy-height tiles | adapter and smoke landed 2026-09-20; never probed, never dispatched |
| `cat_palsar` | JAXA PALSAR-2 ScanSAR scenes and 25 m mosaics | the ScanSAR half lists anonymously; the mosaics' directory answers 401 and needs a JAXA G-Portal account (the owner's); never probed |
| `cat_biomass_esa` | ESA BIOMASS P-band product catalogue | adapter and smoke landed; the listing (ESA's federated catalogue) is public; never probed |

The registry still carries each not-built store's design row, channels,
licence and probe; none has a `store.json` on the Hub (all seven, and the
seven parked stores, answered 404 today).

### 9.3 · Designed, not in the registry

`alerts` (OPERA DIST-ANN annual disturbance maps, a second forest-loss
target) and `biomass100` (ESA CCI Biomass 2020 with GEDI's gridded biomass)
have no adapter. `static_fine` (GEBCO terrain, ESA WorldCover fractions,
GSHHG distance to coast) was measured keyless and is not built for two
measured reasons: the sharded layout carries one channel list and dtype per
store where it needs three, and its land-cover half is 124 GB of 10 m tiles
and tens of hours of reduction. `cat_gfm`'s flood service has no global
listing. Three private-track stores (soil-moisture networks, river discharge,
European station climate) are a later phase with no adapter.

## 10 · Known limits and gotchas

The four quirks found on family 1.gf today, checked here:

- **The registry's `record_span` is the requested window, not the data** —
  holds for all 17 built stores (`ghcnd` says → 2026-12-31 and ends
  2026-09-14; `cat_s2` says 2015-01-01 → and starts 2015-07-04). Read the
  first and last `time_s`, or `store.json → per_year`.
- **`qc_keep_max: 2` is meaningless where `qc` is a bitmask** — holds: `fire`
  and `flux` are bitmasks, the catalogues carry baseline codes (and say so in
  `qc_keep_note`), `ndbc` is always 0; only `tide` dropped rows by it. And
  `ghcnd`'s `qc = 0` means "passed", the opposite of family 10's "not
  assessed" (§4.1).
- **Schema 3 is int64** — holds, for `ghcnd`, `igra` and `tide` (and
  `tide_private` per the build log); the other twelve public tier-P and T
  stores are schema 2, int32. The registry's `schema_version: 2` for the two
  gridded stores means nothing — they have no time column.
- **Private stores show `built: false`** — holds: `tide_private` is built,
  `gbif_nc` is unconfirmed, and the private repository answers 401 for any
  path, real or not.

And the ones particular to this family:

- **`uint8` arithmetic overflows** (`2000 + lossyear` raised today); convert
  first. `lossyear`'s missing is **255**, not NaN; it is 4,480 groups in one
  bin, 1,329 of them 0-byte (sea). `chirps05` has two frames a bin and 3–6-day
  pentads: use `frame_table`; 62 of its shards are 0 bytes.
- **A catalogue's `log2_area` is a logarithm**, most of its other channels
  are NaN by design, `sensor = 255` is a code the table lacks, and
  `assets.parquet → asset_set` is the file list itself, not a key of
  `asset_sets` as `assets_note` says (§2.4).
- **A producer's 200 can be a login page.** `landsatlook.usgs.gov` answers an
  anonymous file request with 200 and an HTML login form; check the bytes.
  NASA's hosts and CDSE answer 401 (§5.4).
- **Records end where the source ends**: `ndbc` 2025-12-31, `cat_s2`
  2026-09-15, `tide` without the Great Lakes (±50 m bound); `cat_ecostress`
  lists an overpass once per processing version — select on `qc`.
- **`flux` is degraded**: 26 towers absent, one on a longitude-derived clock,
  while its `qc_policy` text says 27 on the longitude clock; its last row is
  2026-01-01 08:30 UTC (847 rows in "2026") — the record ends in 2025.
- **`fire`'s footprint is per satellite** in `platforms.json`; the `fp`
  column carries MODIS' 1 km for every row. `confidence` is NaN for VIIRS.
- **A negative bin is normal** (−15,998 in `ghcnd`): index by `b − bin_first`
  and floor toward minus infinity. A shard's year folder is the year of the
  bin's first day (`chirps05/1981/` holds −73 … −1).
- **Refuse an HTTP 200 where you asked for 206** — the host is sending the
  whole file, up to 12.6 GB for `ndbc/values.npy`.
- **`partials/` is scaffolding**: `lst05/2007`, `pheno500/2017` and
  `pheno500/2019` are overwritten records with orphaned files (§2.7).
- **`ghcnd` is 47 GB, `tide` 36 GB, `ndbc` 30 GB, `fire` 22 GB** — size for
  these and memory-map. Search radii are not measured (§6), and family 10.2's
  registry does not yet name this family (§7.1).

## 11 · Provenance and licences

Built by `ml/build_family1_stores.py` (stages index, fetch, assemble, publish,
check, plus probe) from one adapter per store in `ml/family1/adapters/` (the
catalogues share `_stac.py`), the sharded layout in `ml/family1/sharded.py`,
the point layout and reader in `ml/family10_store.py`, registered by
`ml/build_family1_registry.py`, dispatched by the `family1-build` workflow
(`workflow_dispatch` only). Credentials, where a fetch needed them (NASA
Earthdata for the MODIS fields and the lasers, a FIRMS map key for `fire`),
lived only on GitHub-hosted runners; five catalogues were built from a
sandbox with no producer credential at all, because every listing endpoint
answers anonymously. Every `store.json` carries the builder's git commit, the
build time, the source URLs and a `verified` sentence recording what was
checked against the live archive before the adapter was written. Design:
[the family 1.0.tf note (PDF)](https://blauewelt.github.io/earth/ml/paper/notes/family1tf.pdf);
build plan:
[E-082](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E082_family1_builds.md);
build log:
[family 1 build log](https://blauewelt.github.io/earth/docs.html?f=ml/family1/BUILD_LOG.md).

**Licences, as each store's `licence` block states them.** "Redistribution"
and "derived works" are the registry's two fields; cite each source when you
use its data.

| store | licence (the block's `name`) | redistribution · derived works | cite |
|---|---|---|---|
| `ghcnd` | CC0 1.0 (NOAA NCEI) | yes · free | Menne et al. (2012), GHCN-Daily v3, NOAA NCEI, doi:10.7289/V5D21VHZ |
| `igra` | US government work (NOAA NCEI), no restriction | yes · free | Durre, Yin, Vose, Applequist, Arnfield (2016), IGRA version 2, NOAA NCEI, doi:10.7289/V5X63K0Q |
| `tide` | GESLA-4 contributor terms: free use and redistribution | with attribution · free | Haigh, I. D. et al. (2023), GESLA version 3, Geosci. Data J. 10, 293–314; GESLA-4.1 via the UHSLC ERDDAP; each record's contributor (`platforms.json`) |
| `tide_private` | GESLA-4 research-only contributor terms | **no** · **restricted** | as `tide` — **private; research only; never redistribute** |
| `ndbc` | US government work (NOAA NDBC), no restriction | yes · free | NOAA National Data Buoy Center |
| `gliders` | IOOS Glider DAC: "may be redistributed and used without restriction" | yes · free | U.S. IOOS National Glider Data Assembly Center; the deployment's institution (`platforms.json`) |
| `fire` | NASA open data (no restrictions) | with attribution · free | NASA FIRMS; MODIS C6.1 and VIIRS 375 m active fire products, doi:10.5067/FIRMS/MODIS/MCD14DL.NRT.0061, doi:10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002 |
| `flux` | CC BY 4.0 | with attribution · free | FLUXNET Shuttle (2026-04-30 release) from AmeriFlux, ICOS and TERN; each site's own citation and DOI in `platforms.json` |
| `chirps05` | CC BY 4.0 | with attribution · free | Funk, Peterson, Harrison et al. (2026), CHIRPS version 3, Sci. Data 13, 718, doi:10.1038/s41597-026-07096-4 |
| `lossyear` | CC BY 4.0 | with attribution · free | Hansen, M. C. et al. (2013), Science 342, 850–853; GFC-2025-v1.13 |
| `cat_landsat` | rows CC0; imagery a US government work, public domain | yes · free | Landsat Collection 2 Level-2 courtesy of the U.S. Geological Survey |
| `cat_s1`, `cat_s2`, `cat_olci` | rows CC0; imagery Copernicus data policy (Regulation (EU) No 1159/2013), redistribution with attribution | yes · free | "Contains modified Copernicus Sentinel data"; product list from the Copernicus Data Space Ecosystem |
| `cat_hls`, `cat_ecostress`, `cat_viirs` | rows CC0; imagery NASA open data, free with attribution | yes · free | HLS v2.0 / ECOSTRESS L2T LSTE (LP DAAC), VIIRS L1B (LAADS DAAC); lists from CMR |
| `cat_nisar` | rows CC0; imagery NASA/ISRO NISAR L2 GCOV, NASA open data, free with attribution; **BETA or PROVISIONAL, calibration may change** | yes · free | NISAR L2 GCOV, NASA/ISRO, Alaska Satellite Facility DAAC |
| `gbif_nc` | CC BY-NC 4.0, per record | **no** · **non-commercial** | GBIF.org occurrence snapshot — **private; non-commercial** |
| `gbif` | CC0 1.0 and CC BY 4.0, per record | with attribution · free | GBIF.org occurrence snapshot (AWS Open Data) |
| `lst05`, `snow05`, `pheno500`, `lai500`, `burned500`, `refl05` | NASA open data (no restrictions) | with attribution · free | the MODIS product DOI in each block (MOD11C1, MOD10C1, MCD12Q2, MOD15A2H, MCD64A1, MOD09CMG v061) |
| `icesat2`, `gedi` | NASA open data (EOSDIS, no restriction on use) | yes · free | ATL08 v7 (NSIDC); GEDI L2A/L2B v3 (LP DAAC) and L4A (ORNL DAAC) |
| `canopy30` | CC BY | with attribution · free | Potapov, P. et al. (2022), Front. Remote Sens. 3 |
| `sif` | Copernicus open data (Sentinel Data Legal Notice) | with attribution · free | S5P-PAL TROPOSIF L2B; Koehler, Frankenberg et al. (2018), GRL 45 |
| `canopy_ref` | rows CC0; maps CC BY 4.0 | yes · free | Lang et al. (2023), Nat. Ecol. Evol. 7; Tolan et al. (2024) |
| `cat_palsar` | rows CC0; imagery JAXA terms of use — **pixels not redistributable** | with attribution · **restricted** | © JAXA, ALOS-2 PALSAR-2 |
| `cat_biomass_esa` | rows CC0; products ESA Earth Observation data policy, free with attribution | with attribution · free | "Contains modified ESA BIOMASS data" |

What each permits, in plain words: **CC BY 4.0**, the Copernicus policy and
the "with attribution" policies permit any use, including commercial, with
credit; **CC0 / US government work** and NASA's open-data policy permit any
use with no condition; the catalogues' own rows are **CC0** and the imagery
they point at carries its producer's terms; **`tide_private` is research-only
and not redistributable**, **`gbif_nc` is non-commercial**, and both are
private for that reason; **`cat_palsar`'s** JAXA pixels may not be
redistributed. No store here is under a share-alike or no-derivatives clause.
