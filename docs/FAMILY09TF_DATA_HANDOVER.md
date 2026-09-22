# Family 0.9.tf — land and coast through a published embedding: a self-contained data handover

PDF design note: [family09tf.pdf](https://blauewelt.github.io/earth/ml/paper/notes/family09tf.pdf)

**For an agent that has not seen this repository.** This is a DELTA document.
Family 0.9.tf is family 1.0.tf with one substitution, so everything it shares
with 1.0.tf — the tier-P point stores, the sharded tier-G grids, their files,
checksums, channels and quality flags — is described once, in
[`docs/FAMILY1TF_DATA_HANDOVER.md`](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY1TF_DATA_HANDOVER.md),
and is not repeated here. This page says what 0.9.tf adds, how its registry
points at 1.0.tf's, what exists on the Hub today (only the registry) and where
the embedding it is designed around can be read. Nothing below imports code from
this repository; the one recipe is the Python standard library. Written
2026-09-22. Every number was either **measured today** — the registry
downloaded and hashed, a Hub folder listed, Google's bucket and its mirror listed
and range-read anonymously, the tile index downloaded and counted — or is quoted
from the design note and says so. What could not be measured says "not
measured".

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
> 3. **Two inherited stores are private.** `tide_private` and `gbif_nc` live in
>    the private repository `chfrank/earth-tensors-private` (an anonymous read
>    answers HTTP 401). They are **available only with a read token from the
>    owner**. This page does not contain a token and does not ask for one; if
>    you are not given one, skip them by their `distribution` field.
> 4. **Read-only applies to tokens too.** If you hold any Hugging Face token for
>    any reason, do not use it against these repositories for anything but a
>    read.
> 5. **The embedding is Google's, not ours.** If you read AlphaEarth tiles
>    yourself (§11), read them from Google's bucket or its public mirror,
>    anonymously and read-only, under their CC BY 4.0 licence. Never re-host
>    them anywhere public on this programme's behalf.

**What 0.9.tf is, in one sentence.** Family 0.9.tf ("tf" for *terrestrial,
fine*; 0.9 because its image source is a learned summary rather than an
observation) is family 1.0.tf — land and coast observations at 10 km or finer
and 5 days or finer — with 1.0.tf's raw-imagery scene catalogues replaced by
Google's **AlphaEarth Foundations Satellite Embedding** (64 numbers per 10 m
of land per year, 2017–2025), and every other store **inherited by reference**
from 1.0.tf: the same bytes under the same path, never copied.

**What you can download today, at a glance.**

| what | state (2026-09-22) | where |
|---|---|---|
| the registry `family09tf.json` | **PUBLISHED**, 3,589 bytes, 0 own groups | `tensors/family09_tf/family09tf.json` |
| `aef1k` (1 km pooled embedding + coherence) | **NOT STARTED** — no adapter, no probe, no parts, not in the registry | — |
| `aef_dots` (16 raw 10 m samples per 0.25° cell per year) | **NOT STARTED** — same | — |
| `aef100r` (100 m pooled embedding over chosen regions) | **NOT STARTED** — same | — |
| `aef10` (Google's 10 m files referenced by byte range) | **NOT STARTED** — a registry group of layout `external-cog` in the note; not in the registry | Google's bucket and its mirror (§11) |
| the inherited 1.0.tf stores | exactly as 1.0.tf publishes them: 9 of the 24 non-imagery stores are built and public today | `tensors/family1_tf/<store>/` — see §7 |

**Do not assume an embedding store exists until the registry lists it with
`built: true`.** Today the registry's `groups` is empty, `n_built` is 0 and
`not_built` is empty: the planned stores are not listed even as unbuilt.

---

## 1 · What this is, in one paragraph

Family 1.0.tf holds the land's raw fine record, including a tier-T catalogue of
satellite scenes (one row per scene, pointing at the producer's own files),
which becomes model input only once a small learned image encoder — the
programme's own "local codec" — turns tiles into tokens. Google DeepMind's
AlphaEarth Foundations model has already done that job at planetary scale: it
reads Sentinel-2 (the European optical satellites), Sentinel-1 (European C-band
radar) and Landsat 8 and 9 (US optical), was trained to reproduce L-band radar,
spaceborne LiDAR (laser ranging), climate, gravity, elevation and land-cover
targets, and publishes one unit-length 64-dimensional vector per 10 m pixel per
calendar year. Family 0.9.tf takes that output as its image source and keeps
every other 1.0.tf store byte for byte, so that the two families differ in
exactly one thing and a model trained on each measures whether the programme's
own codec is worth its compute for land. The price, stated by the note: an
annual cadence (73 times coarser than the five-day rule), a closed model that
cannot be re-run on new sensors such as NISAR (the NASA–ISRO L-band radar
launched in 2025), and a record that begins in 2017.

## 2 · Where it is

Hugging Face dataset repository **`chfrank/earth-tensors`**, public, anonymous,
plain HTTPS; every file is at
`https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/<path>`
(follow the 302 redirect, `curl -L`).

**Listed today**, `…/api/datasets/chfrank/earth-tensors/tree/main/tensors/family09_tf`
returns exactly one file, the registry. There is **no `partials/family09_tf/`**
folder (the listing answers 404; the `partials/` root holds `f7l1`,
`family10`, `family10_1`, `family10_2`, `family1_gf` and `family1_tf` only), so
no build of this family has ever started, not even a parked one.

### 2.1 · The registry, measured

`tensors/family09_tf/family09tf.json`: **3,589 bytes**, sha256
`874574a47677cd8dd0fa08fcd17765534796fa6c5df9a6d0795f7f9d31f368fc`,
`generated_utc` **2026-09-22T09:24:59Z**, `builder_git_sha` `19f29336…`,
written by `ml/build_family1_registry.py` from the same builder commit as 1.gf's
(09:25:05Z) and 1.0.tf's (09:25:14Z) registries, and published with a restore
check (the publisher downloads the file back and compares sha256 before
declaring success; the run that did it is not recorded in any document read for
this page). It reads:

| key | value |
|---|---|
| `family` / `family_version` / `family_code` | `family09_tf` / `0.9.tf` / `09tf` |
| `hf_root` | `tensors/family09_tf` |
| `groups`, `built`, `not_built` | `[]`, `[]`, `[]` |
| `n_groups`, `n_built` | 0, 0 |
| `inherits` | `"family1tf"` — the parent registry's name |
| `inherits_block` | `{"family_code": "1tf", "family_version": "1.0.tf", "registry": "tensors/family1_tf/family1tf.json", "root": "tensors/family1_tf", "groups": [33 names], "note": …}` |
| `epoch`, `pentad_days`, `bin_rule` | 1982-01-01, 5, `bin = floor(time_s / 432000 s)` — family 10's rule |
| `tiers` | G (sharded grids), P (points), T (scene catalogues), same text as 1.0.tf |
| `private_repo` | `chfrank/earth-tensors-private` |

The `inherits_block.note`, verbatim: *"family 0.9.tf is family 1.0.tf with the
raw satellite imagery replaced by Google's AlphaEarth embedding. Every other
store is the SAME BYTES under the same path, so it is listed BY REFERENCE: read
the named registry for them. `groups` below holds only what 0.9.tf builds
itself — a copy of the parent's rows is the thing that would go stale."*

The parent it names, `tensors/family1_tf/family1tf.json`, measured the same
day: 6,699,677 bytes, sha256
`4ded38a238fc663d3d67ba6185c1d409de168e9f8914fd81f7158b9c03c0782a`,
`generated_utc` 2026-09-22T09:25:14Z, 33 groups, 17 built.

## 3 · The channels (designed, none built)

Quoted from the design note §5; none of these exists as a file.

| store | tier | channels | grid / rows | footprint (log₂ f_p, log₂ f_t) | note's size | phase |
|---|---|---|---|---|---|---|
| `aef1k` | G sharded | C = 65: the 64 components of the **renormalised mean of the unit vectors** in each cell, int8 under the producer's companding, plus a **coherence** channel (length of the un-normalised mean, 0 to 1, uint8) | 1/120° (≈ 0.93 km), 43,200 × 21,600, one frame per year | (−4.9, +6.2) | ≈ 0.2 TB for 2017–2025 | A |
| `aef_dots` | P | C = 64, float16: sixteen **unpooled** 10 m vectors per 0.25° land cell per year at fixed sub-cell positions (the same sixteen every year); `platform` = hashed tile id, `qc` = 1 | ≈ 43 million rows | (−11.4, +6.2) | ≈ 7 GB | A |
| `aef100r` | G sharded | C = 65, as `aef1k` | 1/1,200° (≈ 93 m) over declared regions: the Amazon arc, the Congo basin, insular Southeast Asia first; the ±100 km coastal band later | (−8.2, +6.2) | ≈ 0.35 TB (regions); ≈ 5.7 TB (band) | B / C |
| `aef10` | external | none copied: a registry group of layout `external-cog` carrying the mirror's base URL, the tile index with its sha256, the UTM (Universal Transverse Mercator) projection per zone, the de-quantisation formula and the orientation flag | Google's 10 m tiles, ≈ 1.4 × 10¹² land pixels a year | (−11.4, +6.2) | 0 bytes (≈ 577 TB referenced) | A |

A fifth, `aef_custom` (sub-annual embeddings from Google's Custom Satellite
Embeddings academic programme), exists in the note only as a decision: it would
be private-track, for this programme's use only, and only if an application is
granted. The note records the programme's application deadline as 2026-10-15.

## 4 · How the values would be made

From the note, not from code: no AlphaEarth adapter exists under
`ml/family1/adapters/` and the registry builder has no AlphaEarth group
definitions. **Time**: a year-Y embedding is filed under the first five-day bin
that begins *after* 31 December Y, so an anchor inside year Y sees Y−1 and
earlier, never its own year; the note records that the producer does not state
the observation window behind each annual layer, so this cannot rule out
observations past 31 December. **Pooling**: de-quantise each int8 value q as
`(q / 127.5)² · sign(q)`, sum the unit vectors, renormalise; keep the
pre-normalisation length as coherence. Google's own read-me prescribes the same
three steps for downsampling and warns that averaging the raw integers is wrong.
**Missing values**: −128 in the source is NaN in every store, never a zero
vector; a pooled cell with under 10 % valid pixels is NaN. **Checks each tile
must pass** (note §8): de-quantised vectors unit length within 10⁻², the tile's
corner against the index polygon, and the per-year tile count and byte total
against the listing.

**First step, not taken:** one UTM zone of one year (≈ 280 tiles, ≈ 0.5 TB)
through the pooling code to measure throughput, valid fraction, coherence and
shard bytes before the nine-year build (`ml/plans/E082_family1_builds.md` wave
5; `ml/plans/E082_HANDOVER.md` §4: "nothing started").

## 5 · Opening it — resolving the inheritance

A consumer holding `family09tf.json` resolves **every** store through the parent
registry: take `inherits_block.registry`, download it, and look each name in
`inherits_block.groups` up in the parent's `groups`; each parent row carries
`tier`, `built`, `distribution`, `repo` and `path`, and the files live under
`path` in `repo`. A request under `tensors/family09_tf/<store>/` is wrong and
answers 404. Executed in the sandbox on 2026-09-22:

```python
import json, urllib.request

HUB = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/"

def get_json(path):
    with urllib.request.urlopen(HUB + path) as r:      # anonymous, read-only
        return json.load(r)

reg = get_json("tensors/family09_tf/family09tf.json")
assert reg["family_version"] == "0.9.tf"
own = {g["name"]: g for g in reg["groups"]}              # what 0.9.tf builds itself
blk = reg["inherits_block"]
parent = get_json(blk["registry"])                       # tensors/family1_tf/family1tf.json
assert parent["family_version"] == blk["family_version"] == "1.0.tf"
rows = {g["name"]: g for g in parent["groups"]}

IMAGERY = {"cat_landsat", "cat_hls", "cat_s2", "cat_s1", "cat_nisar",
           "cat_palsar", "cat_olci", "cat_viirs", "cat_ecostress"}   # replaced by design (note §2)

for name in blk["groups"]:
    g = rows[name]                                       # every inherited name resolves
    state = ("built" if g["built"] else "not built") + \
            (", PRIVATE: token from the owner only" if g["distribution"] == "private" else "")
    role = "replaced in 0.9.tf by design" if name in IMAGERY else "inherited"
    print(f"{name:16s} tier {g['tier']}  {state:40s} {g['repo']}/{g['path']}  [{role}]")
print("own groups:", sorted(own) or "none", "| own built:", reg["n_built"])
```

All 33 names resolved; the last line printed `own groups: none | own built: 0`.
Once a file is located, open it with the recipes of the 1.0.tf handover (§5
there: nine arrays plus `store.json` for a point store, CSR — compressed sparse
row — offsets per bin, two range reads per
frame for a sharded grid). An embedding store, when one exists, will be one more
row in `reg["groups"]` with its own `path` under `tensors/family09_tf/` and
`built: true`; until then `own` is empty and there is nothing 0.9.tf-specific
to open.

## 6 · Search and indexing

Nothing 0.9.tf-specific exists. The inherited stores are searched exactly as in
1.0.tf (CSR `bin_offsets` per five-day bin, then a k-nearest search in space;
shard index per bin for the grids). `aef_dots` would be an ordinary tier-P
store under the same rule; `aef1k` and `aef100r` would carry one frame per year
in the bin after 31 December.

## 7 · Relationship to the other families

**1.0.tf, the parent.** The published `inherits_block.groups` is **all 33 of
1.0.tf's registered stores**, including the nine image catalogues the design
note says 0.9.tf does *not* inherit (`cat_landsat`, `cat_hls`, `cat_s2`,
`cat_s1`, `cat_nisar`, `cat_palsar`, `cat_olci`, `cat_viirs`, `cat_ecostress`;
the note's tenth, `cat_submetre`, is not in 1.0.tf's registry). The builder
writes the parent's full store list (`stores_of(parent)` in
`ml/build_family1_registry.py`), so the exclusion lives only in the note today:
**a consumer building the 0.9.tf experiment drops those nine itself**, as the
`IMAGERY` set in §5 does. `canopy_ref` and `cat_biomass_esa` are catalogues of
canopy maps and radar biomass products, not raw imagery, and stay inherited. Of
the 24 inherited non-imagery stores, **9 are built and public today** —
`ghcnd`, `igra`, `tide`, `ndbc`, `gliders`, `fire`, `flux` (tier P),
`chirps05`, `lossyear` (tier G) — and 15 are not built, among them the two
private ones (the 1.0.tf handover §2.5: the build log records `tide_private` as
built on 2026-09-17, while the public registry, which cannot read the private
repository, lists it `built: false`). Their row counts, bytes, spans and runs are in the 1.0.tf
handover §2.3 and §9; they are the same files.

**Family 10.2.** Its registry (`tensors/family10_2/family10.json`, measured
today: 54,478 bytes, `generated_utc` 2026-09-22T10:09:04Z) gained a `siblings`
block on 2026-09-22 naming `tensors/family1_gf/family1gf.json`,
`tensors/family1_tf/family1tf.json` and this registry, so a consumer that
starts from family 10 finds 0.9.tf. The tier-P contract is family 10's (epoch,
bin rule, nine arrays, `time_s` as schema 2 int32 or schema 3 int64); the
sharded tier-G layout is family 1's.

**Family 1.gf** is the ocean-and-atmosphere sibling at the same scale
([`docs/FAMILY1GF_DATA_HANDOVER.md`](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY1GF_DATA_HANDOVER.md));
0.9.tf neither inherits from it nor shares a store with it.

## 8 · How to use it in training

Today, 0.9.tf and 1.0.tf present the same data to a model: the registry adds no
embedding. The comparison the family exists for (note §10) needs `aef1k` and
`aef_dots`: tokens from the embedding against the programme's own codec on the
annual land targets (loss year, land cover, canopy height at held-out GEDI laser
footprints — GEDI is NASA's Global Ecosystem Dynamics Investigation LiDAR on the
International Space Station), family 7.2 alone against family 7.2 plus the
embedding, and the embedding with and without the inherited daily fields. Keep
the strict availability rule of §4 in any loader you write: a year's embedding
is never visible to an anchor inside that year.

## 9 · What is published and verified (as of 2026-09-22)

Published: the registry, 3,589 bytes, sha256 as in §2.1, re-hashed from a fresh
anonymous download today and equal to the value the 1.0.tf handover recorded.
Verified today: its 33 inherited names all resolve to rows of the 1.0.tf
registry of the same day. Not published: every embedding store; no probe, no
adapter, no build run, no parked parts. There is no run number to cite because
no job of this family has run.

## 10 · Known limits and gotchas

**The registry is thinner than the note describes.** The note (§5) specifies an
`inherits` object keyed `"1.0.tf"` whose groups each carry their 1.0.tf `path`
and every file's sha256, and four AlphaEarth groups including `aef10`. The
published file carries `inherits` as a plain string, an `inherits_block` with
names only, and no groups at all; paths and digests come from the parent
registry at read time. The builder's docstring names the list
`inherited_groups`; the published key is `inherits_block.groups`.

**The note's inherited table is wider than 1.0.tf's registry.** It lists
`river`, `river_grdc`, `ismn`, `ecad`, `ghcnh`, `hfr`, `alticap`, `alerts`,
`cat_gfm`, `biomass100`, `landcover10` and `static_fine`, and "four private
stores"; none of the twelve is in the 1.0.tf registry today, which holds two
private stores. A store enters 0.9.tf only when 1.0.tf registers it.

**Annual, closed, from 2017.** A five-day event is invisible; NISAR and any
sensor Google did not train on cannot be added; if Google stops the product the
copied stores would freeze at their last year. The note cites reported transfer
weaknesses across regions and years.

## 11 · Provenance and licences

**Inherited stores**: each keeps its 1.0.tf licence, quoted per store in the
1.0.tf handover §11 and in each parent row's `licence` block.

**The AlphaEarth Foundations Satellite Embedding, version 1**: licence
**CC BY 4.0** — free to use, adapt and redistribute, including commercially,
with attribution. The required attribution, quoted verbatim from both Google's
bucket read-me and the Earth Engine catalogue page (fetched today): *"The
AlphaEarth Foundations Satellite Embedding dataset is produced by Google and
Google DeepMind."* Paper: Brown et al. (2025), arXiv:2507.22291.

**Access routes, what the note says and what was measured anonymously today:**

| route | the note (dated 2026-09-16) | measured 2026-09-22, no account |
|---|---|---|
| Google Cloud Storage (GCS), `gs://alphaearth_foundations/satellite_embedding/v1/annual/<year>/<zone>/` | tiles plus index files; "provider-pays since July 2026" | **readable anonymously**: the JSON listing API answers 200 with the nine year prefixes 2017–2025 and `README.md`, `aef_index.{parquet,gpkg,csv}`, `manifest.txt`; `aef_index.parquet` downloaded whole (69,600,467 bytes, updated 2026-02-27); one 2024 tile range-read with HTTP 206. Bucket metadata itself answers 401. **The read-me still says the bucket is "requester pays", which the anonymous reads contradict** — report it if it changes |
| Source Cooperative mirror `tge-labs/aef`, `…/aef/v1/annual/` | complete mirror, anonymous S3 (Amazon's object-storage protocol) and HTTPS, one VRT (a small GDAL virtual-raster file saying how to read a tile) per tile, tiles stored bottom-up | **readable anonymously** through `https://data.source.coop/tge-labs/aef/v1/annual/…`: an S3-style listing shows the same nine year prefixes plus its own index files (`aef_index.parquet` 77,829,744 bytes); the same 2024 tile answered 206 with the same length (1,099,061,844 bytes) and a first kilobyte identical to Google's; its `.vrt` answered 206. The host `us-west-2.opendata.source.coop` did not connect from this sandbox |
| Earth Engine, `ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")` | the same layers as floats in [−1, 1], bands `A00`–`A63`; needs an account | the public catalogue page answers 200 and carries the licence; the collection itself **not verified** (needs a Google account) |
| Hugging Face `Major-TOM/Core-AlphaEarth-Embeddings` | a 62.5 k-cell sample, 2017–2023, the unit-test fixture | the dataset API answers 200: public, not gated, `license:cc-by-4.0`; contents not measured |

**The index, counted today** from Google's `aef_index.parquet` (sha256
`5d216665ff2e4b8871157d217230cc42d542f1ac9a4b224c16d5c1dbff9cb947`):
**302,466 tiles**, 120 UTM zones in every year, per year 33,148 · 33,291 ·
33,385 · 33,206 · 33,281 · 33,700 · 34,151 · 34,149 · 34,155 (2017 → 2025) —
identical to the note's table. **One tile's header**, range-read: a cloud-optimised BigTIFF (a GeoTIFF readable piece by piece over HTTP),
8,192 × 8,192 pixels, 64 samples per pixel, 8-bit signed integers, band-separate,
1,024 × 1,024 internal tiles, zstd compression, nodata −128, 1,099,061,844
bytes. The note's byte totals (≈ 64 TB a year, 576.6 TB in all) were not
re-measured: summing ≈ 302,000 object sizes is more requests than this page
spends.

**Not available**: sub-annual embeddings (Google Maps Platform's Custom
Satellite Embeddings are a private preview with no public licence), the model
and its training code, any version but 1.
