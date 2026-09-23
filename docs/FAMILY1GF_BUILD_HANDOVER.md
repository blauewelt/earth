# Family 1.gf — building the remaining stores on your own infrastructure: a self-contained build handover

PDF design note: [family1gf.pdf](https://blauewelt.github.io/earth/ml/paper/notes/family1gf.pdf)

**For an agent that has not seen this repository.** This page tells you how to
build the five family-1.gf stores that are not yet on the public dataset —
satellite sea-surface temperature at 2 km, cloud-top temperature at 4 km,
SWOT swath sea level, satellite column CO₂ and PACE ocean colour — **from the
source archives, on your own machines, with your own storage and your own
NASA account**, in parallel with the programme's own build fleet, and how to
prove that your bytes and ours are the same bytes. You need a read-only clone
of the builder's code (a public GitHub repository), Python, a free NASA
Earthdata Login account and disk; you need no other document, and no account
or token with this programme. Its companion,
[the family 1.gf data handover](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY1GF_DATA_HANDOVER.md),
describes how to READ the stores once they exist; this page is how to MAKE
them. Written 2026-09-23 against commit
`1913ce4c315cde4f77c0956f7fc9177dad30083a`. Every number on this page was
either **measured in the sandbox today** (a store built from its source and
hashed, a published file downloaded and compared, a range read, a listing
counted) or **read from a build run's own log or a committed probe file**, and
says which. What could not be measured says "not measured".

> **Rules for the reader — read-only toward everything public.**
>
> 1. **Build from the source archives, into your own storage, and keep it
>    there.** The sources are NASA's and NOAA's archives (§3.3); the results
>    live on your disks. How the bytes reach the programme later is decided by
>    the person who gave you this document, not by you.
> 2. **Never write to anything public.** Not to the Hugging Face dataset
>    [`chfrank/earth-tensors`](https://huggingface.co/datasets/chfrank/earth-tensors)
>    (no uploads, no `--push-parts`, no `publish` stage, no `--stage all`,
>    which includes `publish`), not to the GitHub repository
>    [`blauewelt/earth`](https://github.com/blauewelt/earth) (no push, no pull
>    request, no issue, no comment), not to the globe site. You clone the
>    repository; you never write to it. Report everything to the person who
>    gave you this document (§8).
> 3. **Your Earthdata account is your own.** Register one (free), approve the
>    archives' applications in it (§3.3), and use it. Never ask for, look for
>    or use the programme's credentials; this page contains none and asks for
>    none.
> 4. **Read-only applies to tokens too.** You need no Hugging Face account.
>    Unset `HF_TOKEN`, `HF_PRIVATE_TOKEN` and `HUGGING_FACE_HUB_TOKEN` before
>    you start (§3.2). The only things you read from the Hub are our
>    published `store.json` files and, to diagnose a difference, our
>    published tiles — anonymously, for the certification of §7.
> 5. **Derive, never copy.** Never run with `--parts-from-hub`: it pulls the
>    programme's own parked build parts from the Hub, which would make your
>    store a copy of ours and the cross-check of §7 meaningless.
> 6. **The source archives' terms** are quoted in §10; all five are open
>    (NASA open data, NOAA open data) and permit what you are asked to do.

**What you are asked to build, at a glance.** "Reference" is the part the
programme's own fleet builds; you build it too, independently, so the two
pipelines can be compared file for file (§7). You build everything else.

| store | what it is | tier | record, measured on CMR today | stored size | the fleet builds (the reference) | you build |
|---|---|---|---|---|---|---|
| `sst_acspo02` | NOAA ACSPO 2 km daily sea-surface temperature | G | 2000-02-24 → 2026-09-22 | ≈ 18.9 GB a year, ≈ 501 GB the record (probe) | **2020** (two half-year lanes, running) | 2000 → 2019, 2021 → present, **and 2020** |
| `irtb` | NOAA/NASA merged geostationary infrared cloud-top temperature, 4 km, 30° S–30° N, three-hourly | G | 1998-01-01 → 2026-09-21 | 17.52 GB for 2015 (built) | **2015** (built and published today) | 1998 → 2014, 2016 → present, **and 2015** |
| `swot` | SWOT KaRIn 2 km swath sea-surface height anomaly | P | 2023-03-27 → 2026-09-20 (science orbit from 2023-07-26) | 1.53 × 10⁹ rows for January–June 2024 (lanes) | **2024** (January–June fetched, July–December to come) | 2023-07-25 → 2023-12-31, 2025 → present, **and 2024** |
| `xco2` | OCO-2, OCO-3 and GOSAT column CO₂, one row per good sounding | P | 2009 → 2026 | 511,716,245 rows over the record (lanes) | **the whole record** (lanes done, assembly pending) | **one year of your choice** as a cross-check (§7) |
| `pace4k` | PACE daily 4 km ocean colour and phytoplankton community | G | 2024-03-05 → | 21.37 GB of parts for 2024-03 → 2026-08 | **the whole record** (assembly pending) | **one year of your choice** as a cross-check (§7) |

"Tier G" is a gridded store in the programme's sharded layout (one compressed
file per five-day bin); "tier P" is a point store of nine column arrays. Both
are described byte for byte in the data handover; §4 below says what the
build writes where.

---

## 1 · Why these stores, and why build them twice

Family 1.gf is the programme's set of global observation stores at 10 km or
finer and five days or finer, each kept at its own resolution rather than
resampled onto the programme's 0.25° grid. Seven of its thirteen stores are
built and public. What remains is dominated by three stores, and they are
large because they are fine: **`sst_acspo02`** is NOAA's Advanced Clear-Sky
Processor for Ocean (ACSPO) level-3 "super-collated" product — every polar
orbiter's clear-sky sea-surface temperature merged onto one 0.02° grid a day,
18,000 × 9,000 pixels, the finest satellite temperature field there is;
**`irtb`** is the infrared brightness temperature (IRTB) of cloud tops from
the NCEP/CPC merge of every geostationary satellite (NASA's `GPM_MERGIR`),
4 km, cut to 30° S–30° N and every third hour, because deep tropical
convection is what El Niño grows out of; **`swot`** is the Surface Water and
Ocean Topography (SWOT) satellite's Ka-band radar interferometer (KaRIn),
which measures sea level across two 50 km swaths on a 2 km grid, one row per
valid pixel. **`xco2`** (the column-averaged CO₂ mole fraction from NASA's
Orbiting Carbon Observatories OCO-2 and OCO-3, and Japan's GOSAT through
NASA's ACOS retrieval) and **`pace4k`** (NASA's Plankton, Aerosol, Cloud,
ocean Ecosystem satellite, its Ocean Color Instrument, daily 4 km) are small
and are included for completeness.

The fleet that builds for the programme is GitHub-hosted runners, one "lane"
at a time: six hours and about 86 GB of disk each. The large stores take
about 210 lanes between them in the forms of §5 (54 half-years of
`sst_acspo02`, 116 quarters of `irtb`, about 39 months of `swot`). Building
them in parallel on independent infrastructure shortens the calendar time, and **building the same year in both places turns
the parallel build into a test**: if two machines, two operators and two
downloads of the same source produce the same bytes, both pipelines are
right about that year, and every other year each of them built alone
inherits that confidence. §2 measures how far "the same bytes" holds; §7 says
how to compare.

Acronyms and terms, spelled out once: **CMR** — NASA's Common Metadata
Repository, the catalogue every adapter lists its granules from (anonymous);
**granule** — one file of a NASA collection; **Earthdata Login** — NASA's
single sign-on at `urs.earthdata.nasa.gov`; **GES DISC** — NASA's Goddard
Earth Sciences Data and Information Services Center (serves `irtb` and
`xco2`); **PO.DAAC** — NASA's Physical Oceanography Distributed Active Archive
Center (serves `sst_acspo02` and `swot`); **OB.DAAC** — the Ocean Biology
archive (serves `pace4k`); **LP DAAC** — the Land Processes archive (only in
the credential check); **OPeNDAP** — a protocol that serves a subset of a
file's variables over HTTP; **NOAA/STAR** — NOAA's Center for Satellite
Applications and Research; **NCEP/CPC** — NOAA's National Centers for
Environmental Prediction, Climate Prediction Center; **MUR** — the
Multi-scale Ultra-high Resolution sea-surface-temperature analysis (only in
the credential check); **NCEI** — NOAA's National Centers for Environmental
Information; **zstd** — Zstandard, the compressor of every tier-G tile;
**AVX2 / AVX-512** — two generations of the x86 vector instruction set that
numpy chooses between at run time; **bin** — a five-day period counted from
1982-01-01, `bin = floor(seconds since 1982-01-01 / 432,000)`; **frame** —
one time step of a gridded store inside a bin (5 a bin for a daily store, 40
for `irtb`'s three-hourly); **shard** — the one compressed file holding every
tile of one bin; **lane** — one build process covering part of a year (§4);
**UTC** — Coordinated Universal Time.

## 2 · Determinism: what is byte-identical, and what it takes (measured today)

**The claim.** The arrays a store ships are byte-identical when built from
the same source files by the same commit on any machine — **provided three
things are fixed** that the build itself does not fix: the zstd library
version that compresses tiles, the CPU code path numpy uses for logarithms,
and, for `irtb` only, the lane form. Everything below was tested; nothing
else was found to matter.

### 2.1 · The tests

| test | what was built, where | compared with | result |
|---|---|---|---|
| tier P, same machine, lanes vs one pass | `bgcargo` (biogeochemical Argo floats, keyless) 2003 as one whole-year build, and again as two half-year lanes (`d0101-0630`, `d0701-1231`) in separate work directories, merged and assembled | each other | **9 of 9 arrays identical** (`bin`, `bin_offsets`, `fp`, `lat`, `lon`, `platform`, `qc`, `time_s`, `values`); N = 90 both ways |
| tier P, sandbox vs the published store | the same 2003 build (this sandbox, Python 3.11.15, numpy 2.4.4, commit `1913ce4`) | the rows of 2003 inside the published whole-record `bgcargo` store (built 2026-09-17, commit `0bdbfb79`), read by HTTP range requests through its `bin_offsets.npy` | **8 of 8 row arrays byte-identical** over rows 11 to 100 (90 rows; 17,280 bytes of `values`) |
| tier G, sandbox vs the published store, default install | `oc4k` (the published 4 km ocean-colour store, keyless) bin 2411 (2015-01-03 … 07), fetched here as a one-bin lane | the published `oc4k/2015/bin_2411.zst` (86,004,228 bytes) and its index | **different**: 85,988,779 bytes. Decoding every tile of both: 1,856 stored tiles each, same tiles present, **950 of 18,873,790 `log_chl` values differ by exactly one float16 step**, `kd_490` and `total_nobs` identical |
| … the same with numpy's AVX-512 code path switched off | as above, `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` | as above | **every decoded value identical** (56.6 M values); the compressed bytes of 184 tiles still differ |
| … and with `zstandard==0.23.0` | as above, both settings | as above | **`bin_2411.zst` and `bin_2411.idx.npy` byte-identical** (sha256 `f511514a…`, `efda2a9e…`) |
| tier G, two lanes merged and assembled | `oc4k` bins 2411 and 2412 as two one-bin lanes in separate work directories, merged, assembled as 2015, both settings above | the published store | **4 of 4 shard and index files identical, `tile_grid.json` identical**; `shard_index.npy` differs as a file (it covers 2 bins, the published one 1,850) and its rows for bins 2411 and 2412 are byte-identical |
| `glodap` (the bottle store) from scratch | `python3 ml/build_family1_stores.py --store glodap --stage index,fetch,assemble,check` | the published `glodap` store.json | **not measured**: NOAA NCEI's accession directory answered HTTP 503 ("Service Unavailable") or timed out on every one of thirteen attempts between 09:32Z and 10:15Z |

### 2.2 · The two causes, and why they are pinned rather than fixed

**numpy's float32 logarithm depends on the CPU.** `oc4k` and `pace4k` store
chlorophyll as `log10` of a float32 array. numpy dispatches that function to
the widest vector instructions the CPU has. Measured here on 20,000,000
random chlorophyll values: the AVX-512 path and the AVX2 path disagree in the
last bit of **9,148,858** float32 results, and after rounding to the stored
float16, in **1,181** (0.006 %) — the same rate as the 950 of 18.9 M found in
the real shard. numpy's AVX2 path reproduces the published `oc4k` values
exactly; this sandbox's Xeon has AVX-512, and the programme's runners' CPU
model is not recorded in their logs.
**Only `pace4k` among your five stores computes a logarithm**; `sst_acspo02`
subtracts 273.15 in float32, `irtb` subtracts 160 in float64, and the two
point stores scale integers — IEEE-754 floating-point arithmetic, which no
vector path changes
(confirmed: `oc4k`'s two channels without a logarithm were identical on both
paths).

**zstd's output changed between library versions.** The Python package
`zstandard` bundles its own zstd library: 0.22.0 bundles zstd 1.5.5, 0.23.0
bundles 1.5.6, 0.24.0 and 0.25.0 bundle 1.5.7. Re-compressing the published
`oc4k` tiles at the store's level 15: **0.22.0 and 0.23.0 reproduce 400 of
400 tiles byte for byte, 0.24.0 only 369 of 400**, and 0.25.0 (what a plain
`pip install` gives today) produced 184 differing tiles in the one-bin
rebuild — same decoded values, different compressed bytes, so a different
sha256. At `irtb`'s level 9, 0.23.0 and 0.25.0 both reproduce **1,092 of
1,092** tiles of today's published `irtb/2015/bin_2450.zst` (four frames).
`sst_acspo02` and `pace4k` compress at level 15 and are therefore
version-sensitive. **Which zstd the running `sst_acspo02` 2020 lanes use is
not measured** (their parts were not yet on the Hub when this was written);
§7.3 gives the one-minute test you run against our published tiles before
comparing.

**`store.json` itself always differs** — it carries `built_at`, the builder's
git commit and the ledger counters of however many lanes built the year (a
tier-P lane reads every float or file active in its year and keeps only its
window's rows, so pass-level counters such as `floats_read` double when a year
is built in two lanes while `profiles_kept` does not). It is compared field by
field (§7), never by hash. `manifest.json` is written only by `publish`, which
you never run.

**So the certification can be sha256-for-sha256** — for every array of a
tier-P store and every shard, shard index and `tile_grid.json` of a tier-G
store — once §3's pins are in place and, for `irtb`, the lane form of §6.2 is
matched. Where either pin cannot be met, compare decoded values (§7.3), which
the tests above show are identical.

## 3 · The environment

### 3.1 · Code, Python, packages

```bash
git clone https://github.com/blauewelt/earth.git       # a read-only clone
cd earth
git checkout 1913ce4c315cde4f77c0956f7fc9177dad30083a    # the commit the reference years were built with
python3 -m venv .venv && . .venv/bin/activate
# the programme's install line, verbatim from .github/workflows/family1-build.yml (line 233):
pip install --quiet numpy huggingface_hub zstandard netCDF4 requests rasterio pyhdf scipy pyarrow h5py
# the two pins that reproduced the published bytes (§2):
pip install "zstandard==0.23.0" "numpy==2.4.4"
python3 -c "import zstandard, numpy; print(zstandard.__version__, zstandard.ZSTD_VERSION, numpy.__version__)"
#   must print: 0.23.0 (1, 5, 6) 2.4.4
```

The programme's workflow has **no Python-setup step**: it runs the image's
own `python3` on GitHub's `ubuntu-24.04` runner image (the image name is
printed in every job log; the Python version is not printed, so it is not
measured). The bytes above were reproduced with **Python 3.11.15**, so the
interpreter's minor version is not what matters; the two pins are. For the
five stores on this page the builder needs only numpy, zstandard, netCDF4,
requests and huggingface_hub (the registry and all builds here ran in a
sandbox that has no rasterio, pyhdf or h5py); install the full line anyway so
your environment matches the programme's. The job log of today's runs prints
the rest of the programme's versions: rasterio 1.5.1 on GDAL 3.12.4,
pyhdf 0.11.7, scipy 1.18.1, pyarrow 25.0.1, h5py 3.16.0.

**For `pace4k` on a CPU with AVX-512**, also:

```bash
export NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
python3 -c "import numpy; numpy.show_runtime()"   # X86_V4 and AVX512_* must now be under not_found
```

The names are numpy 2.4's; on a CPU without AVX-512 there is nothing to do.
Setting it for the other four stores is harmless.

### 3.2 · What the programme's build step sets, and what you set instead

The build step of
[`.github/workflows/family1-build.yml`](https://github.com/blauewelt/earth/blob/main/.github/workflows/family1-build.yml)
sets these variables. Your column is the whole point:

| variable | what it does in the programme's job | you |
|---|---|---|
| `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | the Earthdata Login account; given to hosted runners only, from repository secrets | **set to your own account** — the builder refuses a credentialed store before its first byte if either is unset |
| `HF_TOKEN`, `HF_PRIVATE_TOKEN` | the Hugging Face write tokens used by `publish` and `--push-parts` | **never set**; `unset HF_TOKEN HF_PRIVATE_TOKEN HUGGING_FACE_HUB_TOKEN` |
| `HF_HUB_DISABLE_XET=1` | forces the Hub's classic upload path (the newer one timed out on a 52 GB store) | irrelevant — you upload nothing; harmless if set |
| `GITHUB_TOKEN` | lets the job publish its live progress to a branch of the repository | **never set**; you do not write to the repository |
| `FIRMS_MAP_KEY`, `FLUXNET_USERNAME`/`_PASSWORD`, `COPERNICUSMARINE_SERVICE_USERNAME`/`_PASSWORD` | credentials of other families' stores | not needed for these five stores |
| `STORE`, `STAGE`, `START`, `END`, `EXTRA`, `PROBE_MONTH`, `ADAPTER_ENV`, `WORK`, `LIVE_BRANCH` | the workflow's plumbing: they become the command line of §5 | you type the command line directly |
| `adapter_env` pairs (e.g. `SWOT_ALLOW_BUILD=1`) | the adapter's own knobs, exported one by one after a name check | exported by you, per §6 |

The workflow's separate `.netrc` step writes, with mode 600, and deletes at
the end of every job:

```
machine urs.earthdata.nasa.gov login <your Earthdata username> password <your Earthdata password>
```

Write the same file with **your** account (`chmod 600 ~/.netrc`). The builder
prefers the two environment variables (they are the measured route: its
session sends the password to `urs.earthdata.nasa.gov` only, never to an
archive or a signed download URL) and falls back to the `.netrc`; the
credentials preflight, however, checks the **variables**, so set both.

### 3.3 · Your Earthdata account: the applications to approve

Register at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/users/new).
Each NASA archive is an "application" your account must authorise once. What
was measured today, by following each archive's own redirect to the login
host:

| store | data host (from each granule's CMR entry) | Earthdata application (`client_id`) | approval link |
|---|---|---|---|
| `irtb`, `xco2` | `data.gesdisc.earthdata.nasa.gov` | **"NASA GESDISC DATA ARCHIVE"**, `e2WVk8Pw6weeLUKZYOxvTQ` | [approve](https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ) |
| `sst_acspo02` | `archive.podaac.earthdata.nasa.gov` | PO.DAAC's cloud archive, `HrBaq4rVeFgp2aOo6PoUxA` | [approve](https://urs.earthdata.nasa.gov/approve_app?client_id=HrBaq4rVeFgp2aOo6PoUxA) |
| `swot` | `archive.swot.podaac.earthdata.nasa.gov` | PO.DAAC's SWOT archive, a **second** application, `YW7gqWudlIG9X4pcwYNn-g` | [approve](https://urs.earthdata.nasa.gov/approve_app?client_id=YW7gqWudlIG9X4pcwYNn-g) |
| `pace4k` | `oceandata.sci.gsfc.nasa.gov/opendap/` | none — the adapter reads NASA's Ocean Biology archive through its OPeNDAP server, which answered without a login (measured 2026-09-20) | — |

The GES DISC application's display name is the one recorded in the builder
(`earthdata_check.py`); **the display names of the two PO.DAAC applications
could not be read here** (Earthdata Login shows them only to a logged-in
user) — identify them by `client_id` on your account's "Applications →
Authorized Apps" page. The programme's account needed an explicit click for
GES DISC (its probes failed with HTTP 401 until it was approved on
2026-09-22) and never saw an approval page for either PO.DAAC host (the
[credential check #90 (one authenticated kilobyte from LP DAAC, GES DISC and PO.DAAC, 2026-09-17)](https://github.com/blauewelt/earth/actions/runs/35285585294)
and today's six SWOT lanes); whether a **new** account
is authorised for PO.DAAC automatically is not measured.

### 3.4 · The first command, and what its answer means

```bash
export EARTHDATA_USERNAME=... EARTHDATA_PASSWORD=...     # your own
python3 ml/build_family1_stores.py --check-credentials --check-credentials-out earthdata_check.json
echo "exit $?"
```

It makes one small authenticated request to each of four targets and prints
one line each, then the whole report as JSON:

| target | what it reads | answers for |
|---|---|---|
| `lp_daac` | a land-temperature granule's metadata file from the Moderate Resolution Imaging Spectroradiometer (MODIS), at LP DAAC | nothing on this page — informational |
| `ges_disc` | the first KB of a `GPM_MERGIR` file on `disc2.gesdisc.eosdis.nasa.gov` | GES DISC, older host |
| `ges_disc_data` | the first KB of `merg_2020010100_4km-pixel.nc4` on `data.gesdisc.earthdata.nasa.gov`, its URL read from CMR | **`irtb` and `xco2`** |
| `podaac` | the first KB of a protected MUR sea-surface-temperature granule | **`sst_acspo02`** (same host) |

Each line's verdict is one of: **`ok`** (HTTP 206 and the path went through
Earthdata Login — the account works for that archive); **`needs_approval`**
or **`refused`** (a 401 from Earthdata Login: the application is not approved
— open the printed `approve_app` link while logged in — or, if every target
refuses, the password is wrong); **`needs_eula`** (a licence page; accept
it at the printed link); **`bad_credentials`**; **`ok_without_login`** (bytes
arrived but the login was never exercised — the account is untested, not
passed); **`error`** (a timeout, a 5xx, a moved archive — not a verdict;
re-run later). **Exit 1** means a definite refusal or unset variables;
**exit 0** means no definite refusal (read the lines: `error` also exits 0).

**The check does not test the SWOT host.** Before the first `swot` lane, run
this (it uses the builder's own session; it was run here with deliberately
wrong credentials and answered `refused 401` with exactly the approval links
of §3.3 — with a working account both lines must read `ok 206`):

```bash
python3 - <<'PY'
import os, sys; sys.path.insert(0, "ml")
from family1 import earthdata_check as e
e.force_ipv4()
s = e.session(os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"])
for u in ("https://archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_D/SWOT_L2_LR_SSH_Expert_008_496_20231231T231538_20240101T000707_PGD0_02.nc",
          "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/L3S_LEO_DY-STAR-v2.81/20200101120000-STAR-L3S_GHRSST-SSTsubskin-LEO_Daily-ACSPO_V2.81-v02.0-fv01.0.nc"):
    r = e.require_urs(e._request(s, "GET", u), "podaac")
    print(r["verdict"], r["status"], r["final_host"], r.get("approval_url"))
PY
```

## 4 · How the builder works, in the part you touch

`ml/build_family1_stores.py --store <name>` runs **stages** in a fixed order
— `index` (list the archive, read one real file, write `plan.json`), `fetch`
(download, convert, write per-year **parts**), `assemble` (the store from the
parts), `publish` (upload — **never**), `check` (re-hash every file against
`store.json`; for tier G, decompress every tile) — and marks each finished
stage with a `<stage>.done` file. **Everything lives under `--work`:**

```
<work>/<store>/plan.json, index.done, fetch.done, assemble.done, check.done, check.json
<work>/<store>/parts/<year>.done           a whole-year ("unnamed") lane's marker
<work>/<store>/parts/<year>/…              its parts
<work>/<store>/parts/<year>/<lane>.done    a named lane's marker
<work>/<store>/parts/<year>/<lane>/…       its parts, counts.json (the lane's ledger), shard index
<work>/<store>/<store>/                    THE STORE: tier P = nine .npy + store.json;
                                           tier G = <group>/<yyyy>/bin_NNNN.zst + .idx.npy,
                                           <group>/tile_grid.json, <group>/shard_index.npy, store.json
<work>/<store>/src/                        downloads in flight (each deleted once converted)
```

A **lane** is one process that fetches part of a year. Its name is derived
from `--start`/`--end`: a whole calendar year is the unnamed lane; one
calendar month is `m01` … `m12`; one quarter is `q1` … `q4`; any other window
inside a year is `dMMDD-MMDD` (so the two halves of a year are `d0101-0630`
and `d0701-1231`); a window across New Year is `dYYYYMMDD-YYYYMMDD`. A tier-G
store's "year" is the **five-day bins whose first day falls in it**, and a
lane owns exactly the bins whose first day is in its window, whole — so the
`sst_acspo02` "2020" is bins 2776 (opens 2020-01-02) … 2848 (opens 2020-12-27,
ends 2020-12-31); 2020-01-01 belongs to bin 2775 and to 2019. `irtb`'s 2015 is
bins 2411 (2015-01-03) … 2483 (2015-12-29, ends 2016-01-02). A tier-P lane
owns the rows whose own second lies in its window.

`--lanes months | quarters | <comma list>` **declares** the lanes a year must
have; it is written into `plan.json` at `index`, and the assembler then
refuses a year whose declared lanes did not all arrive (measured here: "the
build declared lane(s) d0101-0630, d0701-1231 and 1 of them never arrived").
Declare them on every lane command; it costs nothing.

## 5 · The commands, run locally

Every command below uses `--stage index,fetch` for a lane and
`--stage assemble,check` for an assembly, and **never** `publish`, `all`,
`--push-parts` or `--parts-from-hub`. Pass `--work` explicitly (the default
is `ml/cache/family1_gf` inside the clone). **The lane-and-merge form of §5.1
was run here end to end on two keyless stores; the per-store commands of
§5.2 – §5.6 use that form unchanged but were not run here**, because four of
the five stores need an Earthdata account and `pace4k` needs more memory than
this sandbox has; the fleet ran the same windows as hosted lanes, linked
beside each command.

### 5.1 · The lane procedure — verified here, twice

**One work directory per lane, then merge.** Lanes can then run in parallel
on different machines. Measured here on `bgcargo` 2003 (tier P) and on `oc4k`
2015 (tier G), both keyless; the commands that worked:

```bash
W=/data/f1          # your storage
# lane 1 — also becomes the assembly directory
python3 ml/build_family1_stores.py --store bgcargo --stage index,fetch \
    --start 2003-01-01 --end 2003-06-30 --lanes d0101-0630,d0701-1231 --work $W/bgcargo-A
# lane 2 — its own directory (another machine is fine)
python3 ml/build_family1_stores.py --store bgcargo --stage index,fetch \
    --start 2003-07-01 --end 2003-12-31 --lanes d0101-0630,d0701-1231 --work $W/bgcargo-B
# merge: the lane's folder AND its marker
cp -a $W/bgcargo-B/bgcargo/parts/2003/d0701-1231      $W/bgcargo-A/bgcargo/parts/2003/
cp -a $W/bgcargo-B/bgcargo/parts/2003/d0701-1231.done $W/bgcargo-A/bgcargo/parts/2003/
# assemble the whole year (the window must be whole calendar years)
python3 ml/build_family1_stores.py --store bgcargo --stage assemble,check \
    --start 2003-01-01 --end 2003-12-31 --work $W/bgcargo-A
```

It printed `store: 90 row(s), C=96, bins 1534..1606 (50 live)` and `check:
bgcargo — 9 file(s) verified, N=90, schema 2 (not published yet)`, and
`store.json` recorded `lanes_by_year: {"2003": {"declared": true, "expected":
["d0101-0630", "d0701-1231"], "lanes": ["d0101-0630", "d0701-1231"]}}`. The
tier-G form is identical (`oc4k`, lanes `d0103-0107` and `d0108-0112`,
assembled with `--start 2015-01-01 --end 2015-12-31`; §2.1 has the result).

**One work directory for several lanes, in sequence, works only with
`--force` on the second and later lanes** — without it the second lane prints
`stage fetch: already done — skipping (--force to redo)` and **exits 0 having
fetched nothing** (measured). `--force` re-runs `index` and fetches the new
lane; it never re-fetches a different lane's marked parts. The separate-
directory form above avoids the trap entirely.

**The assembly window must be whole calendar years**, e.g. `--start
2020-01-01 --end 2020-12-31`. A window like `2015-01-03 … 2015-01-12` would
name itself a lane and look for a marker of that name.

**Multi-year stores:** copy every year's lanes into one directory and
assemble with `--start <first year>-01-01 --end <last year>-12-31`. Declare
the lanes of every year by running, in that directory before assembling,
`--stage index --force --start … --end … --lanes <form>` with the full window
(`index` lists the archive and reads one real file, so it needs the network
and, for credentialed stores, your account). A directory that has assembled
before carries `assemble.done` and `check.done`, so a re-assembly needs
`--force` too. Measured here: `bgcargo` 2003 and 2004, four half-year lanes
in four directories, merged with `cp -a <lane dir>/bgcargo/parts/<year>/.
<assembly dir>/bgcargo/parts/<year>/`, declared with `--stage index --force
--start 2003-01-01 --end 2004-12-31 --lanes d0101-0630,d0701-1231` ("lanes
expected per year: d0101-0630, d0701-1231") and assembled with `--stage
assemble,check --force --start 2003-01-01 --end 2004-12-31`: N = 417
(`per_year` 90 and 327, the published store's own counts for those years),
and all eight row arrays byte-identical to the published store's rows of
2003–2004.

### 5.2 · `sst_acspo02` — two half-year lanes a year

```bash
export SST_ACSPO02_COLLECTION=DY          # the default; stated so it is recorded
Y=2020
python3 ml/build_family1_stores.py --store sst_acspo02 --stage index,fetch \
    --start $Y-01-01 --end $Y-06-30 --lanes d0101-0630,d0701-1231 --work $W/sst-$Y-A
python3 ml/build_family1_stores.py --store sst_acspo02 --stage index,fetch \
    --start $Y-07-01 --end $Y-12-31 --lanes d0101-0630,d0701-1231 --work $W/sst-$Y-B
cp -a $W/sst-$Y-B/sst_acspo02/parts/$Y/d0701-1231{,.done} $W/sst-$Y-A/sst_acspo02/parts/$Y/
python3 ml/build_family1_stores.py --store sst_acspo02 --stage assemble,check \
    --start $Y-01-01 --end $Y-12-31 --work $W/sst-$Y-A
```

The record opens 2000-02-24, which is inside bin 1325 (opening 2000-02-21):
keep the `d0101-0630` window for 2000 too — a lane owns only the bins that
open inside its window, so a lane starting 2000-02-24 would drop 2000-02-24
… 25 with bin 1325, while `d0101-0630` records the days before the record as
`before_record` and skips the bins wholly before it. The current year's last
lane ends on the last day CMR lists. This is the form the fleet uses for 2020
([#455, the January–June lane](https://github.com/blauewelt/earth/actions/runs/35836499348)
and [#456, the July–December lane](https://github.com/blauewelt/earth/actions/runs/35836505278)).

### 5.3 · `irtb` — four quarter lanes a year

```bash
Y=2015
for q in "01-01 03-31" "04-01 06-30" "07-01 09-30" "10-01 12-31"; do set -- $q
  python3 ml/build_family1_stores.py --store irtb --stage index,fetch \
      --start $Y-$1 --end $Y-$2 --lanes quarters --work $W/irtb-$Y-$1
done
for d in 04-01 07-01 10-01; do
  L=$(ls -d $W/irtb-$Y-$d/irtb/parts/$Y/q?); cp -a $L $L.done $W/irtb-$Y-01-01/irtb/parts/$Y/
done
python3 ml/build_family1_stores.py --store irtb --stage assemble,check \
    --start $Y-01-01 --end $Y-12-31 --work $W/irtb-$Y-01-01
```

This is the fleet's form for 2015: lanes
[#451 (q1)](https://github.com/blauewelt/earth/actions/runs/35836476924),
[#452 (q2)](https://github.com/blauewelt/earth/actions/runs/35836482579),
[#453 (q3)](https://github.com/blauewelt/earth/actions/runs/35836488104),
[#454 (q4)](https://github.com/blauewelt/earth/actions/runs/35836493954), and
the assembly [#460 (irtb 2015: pull, assemble, publish, check on a hosted runner, 23 min)](https://github.com/blauewelt/earth/actions/runs/35838870612).
**Use exactly this form for 2015** — §6.2 explains why the lane form changes
`irtb`'s bytes. The record opens 1998-01-01.

### 5.4 · `swot` — one lane a month

```bash
export SWOT_ALLOW_BUILD=1                 # required: a swot build is opt-in
Y=2024
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  last=$(python3 -c "import calendar; print(calendar.monthrange($Y, int('$m'))[1])")
  python3 ml/build_family1_stores.py --store swot --stage index,fetch \
      --start $Y-$m-01 --end $Y-$m-$last --lanes months --work $W/swot-$Y-$m
done
for m in 02 03 04 05 06 07 08 09 10 11 12; do
  cp -a $W/swot-$Y-$m/swot/parts/$Y/m$m{,.done} $W/swot-$Y-01/swot/parts/$Y/
done
python3 ml/build_family1_stores.py --store swot --stage assemble,check \
    --start $Y-01-01 --end $Y-12-31 --assemble streaming --work $W/swot-$Y-01
```

The fleet's 2024 lanes so far:
[#461 (m01)](https://github.com/blauewelt/earth/actions/runs/35838877599),
[#462 (m02)](https://github.com/blauewelt/earth/actions/runs/35838885401),
[#463 (m03)](https://github.com/blauewelt/earth/actions/runs/35838892217),
[#464 (m04)](https://github.com/blauewelt/earth/actions/runs/35838900538),
[#465 (m05)](https://github.com/blauewelt/earth/actions/runs/35838908066),
[#466 (m06)](https://github.com/blauewelt/earth/actions/runs/35838915813).
**Where 2023 starts is a choice to confirm with the person who gave you this
document**: CMR lists this collection (SWOT L2 LR SSH Expert, version D) from
**2023-03-27**, but until 2023-07-10 that is the one-day-repeat calibration
orbit (cycles 473 … 578); the science orbit's cycle 001 opens
**2023-07-26T00:01:55Z**, and nothing is listed in between. The window this
page assumes is `--start 2023-07-25 --end 2023-07-31` (lane `d0725-0731`) and
then `m08` … `m12` (declare `--lanes d0725-0731,m08,m09,m10,m11,m12` for
2023), which contains the science orbit only; a `m07` lane would also take
ten days of calibration passes. Granules CMR lists per year:
6,988 (2023), 9,692 (2024), 9,509 (2025), 7,641 (2026 so far).

### 5.5 · `xco2` — one lane a year

```bash
Y=2019
python3 ml/build_family1_stores.py --store xco2 --stage index,fetch,assemble,check \
    --start $Y-01-01 --end $Y-12-31 --work $W/xco2-$Y
```

The fleet fetched the record as eighteen whole-year lanes, 2009 … 2026
([#433](https://github.com/blauewelt/earth/actions/runs/35836377380) for 2009
through [#450](https://github.com/blauewelt/earth/actions/runs/35836471183)
for 2026-01-01 … 2026-09-30); 2011 and 2022 failed once on a connection
timeout to Earthdata Login and succeeded on re-run
([#457](https://github.com/blauewelt/earth/actions/runs/35838841295),
[#458](https://github.com/blauewelt/earth/actions/runs/35838848279)). A
failure of that kind stops the lane before any part is marked; re-run the
same command. For one-year certification a single process is enough.

### 5.6 · `pace4k` — one lane a year

```bash
export NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"   # on an AVX-512 CPU (§3.1)
Y=2025
python3 ml/build_family1_stores.py --store pace4k --stage index,fetch,assemble,check \
    --start $Y-01-01 --end $Y-12-31 --work $W/pace4k-$Y
```

No credential. The record opens 2024-03-05, inside bin 3080 (which opens
2024-03-01), so use whole calendar years for every `pace4k` year, 2024
included: the four days before the record are counted `before_record`, and
a window starting 2024-03-05 would own only the bins opening on or after it
(3081 opens 2024-03-06) and lose 2024-03-05. The same holds for any tier-G
record start that is not a bin start.

## 6 · Knobs, sizes and times

### 6.1 · Adapter knobs: what to set, what never to set

| knob | store | default | set it? |
|---|---|---|---|
| `SST_ACSPO02_COLLECTION` | `sst_acspo02` | `DY` | leave at `DY` — the one-granule-a-day super-collation that fills five frames a bin; `PM` and `AM` publish a day and a night granule each and are a different store shape |
| `SST_ACSPO02_MAX_YEARS` | `sst_acspo02` | 1 | **raise it to the number of calendar years of the window on any command that runs the `fetch` stage over more than one calendar year** — a lane crossing New Year (2), or an assembly run as `index,fetch,assemble,check` over several years (the fleet's multi-year box assemblies run `fetch` to pull parts and need it). The guard sits in the adapter's fetch preflight and refuses otherwise ("REFUSING sst_acspo02: the window covers N year(s)"); `--stage index` and `--stage assemble,check` do not consult it. It changes the `notes` text of `store.json` and no array |
| `IRTB_LAT_BAND` | `irtb` | 30 | **never change.** 30° S–30° N is the design; the field is 60° S–60° N. A different band is a different store |
| `IRTB_EVERY` | `irtb` | 6 | **never change.** Every sixth half-hour = three-hourly, 40 frames a bin; the layout cannot hold the full half-hourly field (240 frames a bin against a 64-frame limit) |
| `SWOT_ALLOW_BUILD=1` | `swot` | unset | **required** for any `swot` lane; without it the adapter refuses before its first byte |
| `SWOT_CYCLE`, `SWOT_MAX_PASSES`, `SWOT_ALLOW_CAPPED_BUILD` | `swot` | unset | **never set on a build.** They are the probe's; a build that names one refuses, and `SWOT_ALLOW_CAPPED_BUILD=1` would mark a lane done with a fraction of its passes |
| `XCO2_SENSORS` | `xco2` | all four collections | never set; it narrows the store to named sensors and says so in `store.json` |
| `*_SMOKE_GRID` | all | unset | never set — they shrink the grid for synthetic tests |
| `NPY_DISABLE_CPU_FEATURES` | numpy | unset | set on an AVX-512 CPU (§3.1); essential for `pace4k` |

### 6.2 · The one lane-form dependence: `irtb`'s last bin in every lane

At commit `1913ce4` the `irtb` adapter asks the NASA catalogue (CMR) for the
hours up to the end of its window with a timestamp built from a calendar
**date** rather than a date-time, so the time of day is dropped: the lane
whose last bin runs to 2015-04-02T23:59:59 asks for hours up to
2015-04-02T**00:00**. The last bin of **every** lane therefore loses its
last seven three-hourly frames (03:00 … 21:00 UTC of its last day), which are
recorded honestly as `after_record` in `store.json`'s
`frames_missing_by_reason` and `missing_frames`, never silently. In the
published 2015 store this is **28 frames** — bins 2428, 2446, 2465 and 2483,
seven each, one per quarter lane — out of 2,920 wanted (2,892 present). The
hours exist upstream: CMR lists all 2,208 hours of 2015-01-01 … 2015-04-02
when asked with the full time. Consequences: **(a)** to reproduce the
reference 2015 byte for byte, build it as four quarter lanes (§5.3) — a
whole-year lane would lose only bin 2483's seven frames and its bins 2428,
2446 and 2465 would hold 40 frames instead of 33; **(b)** for your other
years, whatever lane form you use, report the `after_record` count per year
(§8) — it is the size of this defect in your store. A corrected adapter is a
decision for the programme; **do not patch your clone**, or your bytes stop
being comparable. `sst_acspo02` uses the same construction but is not
affected: its granules are daily and start at 00:00, which the truncated time
still includes (measured: a query ending 2020-07-04T00:00:00Z returns the
2020-07-04 granule). `swot`'s and `xco2`'s listings keep the time of day.

### 6.3 · Time, disk and memory per year

Hosted-runner figures are from the fleet's own runs (a GitHub-hosted
`ubuntu-24.04` runner: 4 cores, 16 GB of memory, about 86 GB of free disk);
yours will scale with your bandwidth to NASA.

| store · unit | source read | parts / store written | wall time measured | memory |
|---|---|---|---|---|
| `sst_acspo02` · one half-year lane | CMR declares 422.9 MB a day (2020-01-01); ≈ 77 GB a half-year, each file deleted after its frame is encoded | 51.7 MB a frame (probe, mean of 18 frames; 50.1–53.9 MB) → ≈ 9.5 GB a half, ≈ 18.9 GB a year | probe #172: 49.4 s of encoding a frame, 59.7 s wall; lanes #455/#456 (from their live progress files): **100 frames in 5,763 s and 5,730 s = 57.3–57.6 s a frame** at 10:01Z, i.e. **≈ 3.0 h for a half-year's 180–185 frames** — still running when this was written | the adapter holds a bin's five 18,000 × 9,000 × 2 float16 frames: ≈ 3.24 GB, ≈ 4 GB peak (adapter's own figure); not measured here |
| `irtb` · one quarter lane | 81.3 GB for 2015 (2,892 hourly files, `bytes_files`), ≈ 20 GB a quarter; probe #429: 7.06 GB for 2015-01 | 4.29–4.45 GB of parts a quarter (4,286.1 / 4,347.9 / 4,424.6 / 4,446.3 MB) | **fetch 507–689 s**, job 11.5–15.3 min (#451–#454) | one 1,649 × 9,896 uint8 frame is 16 MB; not a constraint |
| `irtb` · one year's assembly | — | **17,517,692,613 bytes** of shards and indices (146 files) + `tile_grid.json` + `shard_index.npy` (31,215 bytes) | #460: 377 s to gather the parts, 296 s to assemble, 254 s to check (plus 380 s of publishing you do not do) | — |
| `swot` · one month lane | 25.7–27.9 GB of granules a month on CMR (27.9 GB for 2024-01, 865 granules) | 2024-01: 290,659,530 rows, **8,429,570,742 bytes of parts** (241 files) | fetch 320–528 s (143 s for the short May), job 6–13 min (#461–#466) | not measured; not a constraint on a 16 GB runner |
| `swot` · one year's assembly | — | January–June 2024: **1,527,688,830 rows** (May is short upstream: 369 granules against 809–869); at 33 bytes a row that half-year is 50.4 GB of store | not measured | not measured. The streaming assembler peaked at 54.24 GB for 1.68 × 10⁹ rows (`swh`) and 48.90 GB for 1.14 × 10⁹ (`ghcnd`); a SWOT year is about twice those rows, so plan a machine with ≥ 128 GB of memory and ≈ 2.2 × the year's parts in free disk (parts + store) — an extrapolation, not a measurement |
| `xco2` · one year lane | 2019-06 probe: 1.61 GB fetched for 2,279,542 rows | rows a year: 88,047 (2009) … 60,236,900 (2020); **511,716,245 over 2009–2026** ≈ 16.9 GB at 33 bytes a row | fetch 47–682 s a year (#433–#458) | not measured |
| `pace4k` · a year | 46.2 MB a day through OPeNDAP (the adapter's measurement) | ≈ 20 MB stored a frame (probe #295); parts 6.99 / 9.23 / 5.15 GB for 2024 / 2025 / 2026-to-August (data handover §2.6) | probe #295: 14.6 s a frame | **a one-bin fetch was killed for lack of memory in this 7.8 GB sandbox**; hosted 16 GB runners completed whole years. Give it ≥ 16 GB |

## 7 · Certification: comparing your reference year with ours

### 7.1 · What to compare, per tier

Download our `store.json` anonymously (no token):

```bash
curl -sL -o ours.json https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family1_gf/<store>/store.json
```

Today it exists for `irtb` (the 2015 store: 148 file digests, built
2026-09-23T08:51:40Z, commit `1913ce4`); for the other four it answers 404
until the fleet assembles them — the person who gave you this document will
tell you when. Then:

- **Tier G (`sst_acspo02`, `irtb`, `pace4k`).** Every file named
  `<group>/<yyyy>/bin_NNNN.zst` and `<group>/<yyyy>/bin_NNNN.idx.npy` in both
  `sha256` blocks must carry **the same digest**, and so must
  `<group>/tile_grid.json`. A shard is one five-day bin, so this holds even
  when your store covers more years than ours. `<group>/shard_index.npy`
  covers the store's whole bin range and is compared **row by row** for the
  reference bins (measured equal for `oc4k` bins 2411 and 2412 in §2.1).
- **Tier P (`swot`, `xco2`).** Build the reference year **as its own
  store** (assemble that year alone) and compare all **nine** array digests.
  If you compare a year inside a longer store instead, compare the eight row
  arrays over that year's rows, as §2.1 did for `bgcargo`: rows are sorted by
  (bin, time), so a year is one contiguous block starting at
  `bin_offsets[bin(Jan 1) − bin_first]`, and it **ends before the next year's
  rows inside the year's last bin** (a bin straddling New Year holds both;
  in `bgcargo` bin 1680 holds four rows of 2005-01-02 … 04 after the last of
  2004). `bin_offsets.npy` itself differs whenever the bin range does.
- **`store.json`**, field by field: `N` (tier P) or `groups.<g>.frames_present`
  / `frames_missing` / `tiles_stored` / `bytes` / `valid_fraction` (tier G),
  `bin_first`, `bin_last`, `per_year`, `frames_missing_by_reason`,
  `missing_frames`, `counts.out_of_bounds`, and `degraded` (absent in a clean
  build). `built_at`, `builder_git_sha` and lane-dependent counters
  (`fetch_seconds`, files or floats read) are expected to differ.

A short script for the tier-G comparison:

```python
import json, sys
ours, yours = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
a, b = ours["sha256"], yours["sha256"]
ref = sorted(k for k in a if k.endswith((".zst", ".idx.npy", "tile_grid.json")))
same = [k for k in ref if b.get(k) == a[k]]
print(f"{len(same)} of {len(ref)} reference files identical")
for k in ref:
    if b.get(k) != a[k]:
        print("DIFFER" if k in b else "MISSING", k)
```

### 7.2 · The reference years, in numbers you can check before any hash

| store | reference | what our store says (or will say) |
|---|---|---|
| `irtb` | 2015, four quarter lanes | 73 bins (2411 … 2483), **2,892 frames present, 28 missing, all `after_record`** (§6.2), 778,555 tiles, 17,504,928,709 bytes of shards, valid fraction 0.96586, `out_of_bounds.tb` 3,148, `files_read` 2,892, `bytes_files` 81,312,781,313 |
| `sst_acspo02` | 2020, `d0101-0630` + `d0701-1231` | 73 bins (2776 … 2848), 365 frames wanted — not yet assembled |
| `swot` | 2024, `m01` … `m12` | months 01–06 fetched: 290,659,530 · 280,455,011 · 290,411,115 · 279,865,057 · 116,369,732 · 269,928,385 rows — not yet assembled |
| `xco2` | 2009 … 2026 | 511,716,245 rows over eighteen year lanes (per year in §6.3) — not yet assembled |
| `pace4k` | 2024-03 → 2026-08 | not yet assembled; 2024's parts were repaired on 2026-09-23 ([#430, pace4k: rebuild the 2024 ledger from its shards](https://github.com/blauewelt/earth/actions/runs/35836360975)) |

### 7.3 · If a hash differs

Decode before you conclude anything. For tier G, decompress every tile of the
differing shard in both stores (ours by one download or by range reads, as
the data handover's §5.2 shows) and compare the float16 bit patterns per
channel: **identical values with different compressed bytes** means your
zstd is not 1.5.6 — check `zstandard.ZSTD_VERSION`; **one-float16-step
differences in a logarithmic channel only** means numpy used its AVX-512 path;
**anything else is a finding** — report it with the bin, the frame, the
channel and the first differing pixel. To check your zstd against ours
before a long build, re-compress some of our published tiles:

```python
import numpy as np, urllib.request, io, zstandard
B = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family1_gf/irtb/irtb/2015/"
idx = np.load(io.BytesIO(urllib.request.urlopen(B + "bin_2450.idx.npy").read()))
ent = idx.reshape(-1, 2); ent = ent[ent[:, 1] > 0][:273]          # the first 273 stored tiles
lo, hi = int(ent[0, 0]), int(ent[-1, 0] + ent[-1, 1])
blob = urllib.request.urlopen(urllib.request.Request(B + "bin_2450.zst",
        headers={"Range": f"bytes={lo}-{hi - 1}"})).read()
cz, dz = zstandard.ZstdCompressor(level=9), zstandard.ZstdDecompressor()   # level: tile_grid.json → compression
ok = sum(cz.compress(dz.decompress(blob[o - lo:o - lo + n], max_output_size=1 << 20)) == blob[o - lo:o - lo + n]
         for o, n in ent)
print(zstandard.ZSTD_VERSION, ok, "of", len(ent), "tiles re-compress to our bytes")
```

Run here with zstandard 0.25.0 it printed `(1, 5, 7) 273 of 273 tiles
re-compress to our bytes`: at `irtb`'s level 9, zstd 1.5.6 and 1.5.7 agree
(1,092 of 1,092 tiles tested). For `sst_acspo02` and `pace4k` run the same
against their published tiles at level 15 once they exist — that is where the
versions part.

## 8 · What to report back

Send this, per store and per year (or per lane), to the person who gave you
this document — as files in your own storage that they can fetch, never as an
upload to anything of ours:

```text
store:            <name>            year(s): <YYYY[..YYYY]>        lane form: <unnamed | d0101-0630,d0701-1231 | quarters | months>
commit:           1913ce4c315cde4f77c0956f7fc9177dad30083a (or say which)
machine:          <CPU model, AVX-512 yes/no, cores, memory, disk, network>
environment:      python <x.y.z> · numpy <v> · zstandard <v> (ZSTD_VERSION <v>) · netCDF4 <v> · NPY_DISABLE_CPU_FEATURES=<value or unset>
knobs:            <every adapter environment variable set, with its value>
wall time:        index <s> · fetch <s per lane> · assemble <s> · check <s>
source bytes:     <counts.bytes_files or the sum of your downloads>
store:            store.json attached; path <work>/<store>/<store>/
files:            one line per file: <relative path>  <bytes>  <sha256>
N / frames:       tier P: N, bin_first, bin_last, per_year · tier G: per group frames_present, frames_missing, tiles_stored, bytes, valid_fraction
missing:          frames_missing_by_reason (all reasons, with counts) · missing_frames for irtb's after_record (§6.2)
counters:         counts.out_of_bounds · degraded (must be absent) · inputs_not_read · swot: rows_kept, rows_outside_window, outside_product_valid_range, qual_grade · xco2: soundings_kept, soundings_bad_quality, soundings_per_sensor
check:            the last line of the check stage ("check: <store> — N file(s) verified …")
certification:    reference year <YYYY>: <k> of <n> files identical by sha256 · store.json fields that differ and why · any hash that differs, with §7.3's diagnosis
problems:         every refusal, retry and warning, verbatim
```

## 9 · Known limits and gotchas

- **`--stage all` publishes.** Never use it; type the stages.
- **A second lane in the same work directory is silently skipped** unless
  `--force` is passed (§5.1). Separate directories avoid it.
- **`irtb` loses seven frames at the end of every lane** at this commit
  (§6.2). Report the count; match the quarter form for 2015.
- **A tier-G "year" is the bins that open in it**, so `sst_acspo02`'s
  2020-01-01 is in 2019's store (bin 2775 opens 2019-12-28), and `irtb`'s
  2015 store runs to 2016-01-02 (its last bin, 2483, opens 2015-12-29).
- **Transient Earthdata failures stop a lane without marking it** (seen on
  two `xco2` lanes today: a connection timeout to `urs.earthdata.nasa.gov`).
  Re-run the same command with the same `--work`; marked years are skipped.
- **An archive can be down for hours.** NOAA NCEI's ocean-carbon accession
  answered 503 all morning of 2026-09-23 (which is why `glodap` could not be
  rebuilt for §2); the five stores here read from NASA's archives, CMR and
  OB.DAAC, none of which failed today.
- **`sst_acspo02`'s CMR sizes are wrong for many granules** (most of
  2020-01 is declared as 108–114 bytes; the real files are ≈ 423 MB). The
  adapter lists them as `granules_suspiciously_small` at `index` and
  downloads them anyway; a file that really is short is refused at read time
  and leaves its year unmarked.
- **`sst_acspo02`'s quality channel is constant 5 in practice**: the DY
  super-collation published grades 0 and 5 only in the probe month.
- **The check stage's Hub comparison runs only after a publish**, which you
  never do, so your `check.json` carries `"hub": null` and the log says
  "(not published yet)". That is the correct state.
- **The data handover's §9.1 still lists `irtb` and `xco2` as blocked**; the
  GES DISC approval landed on 2026-09-22 and `irtb` 2015 was published
  today.

## 10 · Provenance and the source archives' terms

Builder: `ml/build_family1_stores.py` (stages, lanes, the tier-G writer in
`ml/family1/sharded.py`, the tier-P layout in `ml/family10_store.py`), one
adapter per store in `ml/family1/adapters/`, the credential check
`ml/family1/earthdata_check.py`, the adapter contract
[`ml/family1/ADAPTER_CONTRACT.md`](https://blauewelt.github.io/earth/docs.html?f=ml/family1/ADAPTER_CONTRACT.md),
the build log
[`ml/family1/BUILD_LOG.md`](https://blauewelt.github.io/earth/docs.html?f=ml/family1/BUILD_LOG.md),
the probe files `ml/family1/probes/sst_acspo02_2020-01.json`,
`irtb_2015-01.json`, `xco2_2019-06.json` and `swot_2024-01.json`. The probes
that measured the sizes above:
[#172 (sst_acspo02: 2020-01, 18 frames read)](https://github.com/blauewelt/earth/actions/runs/35323368382),
[#429 (irtb: 2015-01, 241 frames)](https://github.com/blauewelt/earth/actions/runs/35751395044),
[#428 (xco2: 2019-06, 2,279,542 soundings)](https://github.com/blauewelt/earth/actions/runs/35750741673),
[#168 (swot: twelve passes of cycle 010)](https://github.com/blauewelt/earth/actions/runs/35321954861),
[#295 (pace4k: 2024-03, 23 frames)](https://github.com/blauewelt/earth/actions/runs/35526540231).

The terms, quoted from the `licence` block each store carries in the
published registry `tensors/family1_gf/family1gf.json` (generated
2026-09-22T16:10:49Z):

| store | `name` | `redistribution` · `derived_works` | `attribution` (cite this) |
|---|---|---|---|
| `sst_acspo02` | NOAA open data | yes · free | "NOAA/STAR (2022), Advanced Clear-Sky Processor for Ocean (ACSPO) L3S-LEO super-collated SST, version 2.81, distributed by NASA PO.DAAC" |
| `irtb` | NASA open data (CC0-equivalent) | yes · free | "Janowiak, J., B. Joyce, P. Xie (2017), NCEP/CPC L3 half hourly 4 km global (60S-60N) merged IR V1, Goddard Earth Sciences Data and Information Services Center (GES DISC), doi:10.5067/P4HZB9N27EKU" |
| `swot` | NASA open data (CC0-equivalent) | yes · free | "NASA/CNES (2025). SWOT Level 2 KaRIn Low Rate Sea Surface Height, Expert, Version D. PO.DAAC, doi:10.5067/SWOT-L2-LR-SSH-EXPERT-2.0" |
| `xco2` | NASA open data (CC0-equivalent) | yes · free | "OCO-2/OCO-3 Science Team, Vivienne Payne, Abhishek Chatterjee (2024), OCO-2 / OCO-3 Level 2 bias-corrected XCO2 and other select fields from the full-physics retrieval aggregated as daily files, Goddard Earth Sciences Data and Information Services Center (GES DISC); ACOS Science Team for the GOSAT retrieval" |
| `pace4k` | NASA open data (no restrictions) | attribution · free | NASA Ocean Biology Processing Group (2024-2026), PACE OCI L3M BGC (doi:10.5067/PACE/OCI/L3M/BGC/3.2), AOP (doi:10.5067/PACE/OCI/L3M/AOP/3.2) and L4M MOANA (doi:10.5067/PACE/OCI/L4M/MOANA/3.2); MOANA algorithm Lange et al. (2020), Optics Express 28, 25682–25705. Its `terms`: "NASA's Earth science data are open, full and without restriction, free of charge, with no period of exclusive access" |

In plain words: NASA's open-data policy and NOAA's open data permit
downloading, converting, storing and redistributing these data for any
purpose, commercial included, with credit customary (and, for `pace4k`,
asked for). Your Earthdata account's own terms of use apply to your
downloads. Nothing here is share-alike, non-commercial or no-derivatives.
