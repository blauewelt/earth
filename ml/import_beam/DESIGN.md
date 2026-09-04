# Parallel import of the training data — the design (E-073, revision 2)

**Written 2026-09-04 by the Fable session, for Chris and for the agent that
will run it. Revision 2 (same day, on Chris's correction):** the output is a
**sharded `tf.train.Example` dataset** written to wherever the operating
agent chooses (`--output <uri>`: a local directory, a `gs://` bucket, or any
filesystem the runner supports); the Hugging Face Hub is no longer part of
the design; and the throttling is **never allowed to drop data** — a work
item is either written and verified, or it is still in the queue.

Plain English first: this is a program that downloads every upstream
dataset the Earth model's input tensors are built from, in parallel across
many sources at once, converts each one into a standard machine-learning
record format, and writes it as sharded files — *without ever hammering any
of the public servers it reads from, and without ever losing a day of
data.* It is built on Apache Beam (the open-source form of Google's Flume: a
framework for data pipelines that runs the same code on one machine, on a
cluster, or on Cloud Dataflow). Beam is the **orchestrator**, not the
accelerator: the speed limit is set by politeness towards each host, and the
design's whole point is that the limit is enforced by construction, not by
good intentions.

Companion files: `README_FOR_GEMINI.md` (the step-by-step howto),
`sources.yaml` (the registry — every source, every host budget),
`CREDENTIALS.md`, and the code under `beam_import/`.

---

## 0 · Why this exists, in one paragraph

Today every tensor build fetches its sources straight from the upstream
archives, one file at a time, on whatever machine the build runs on
(`ml/build_family7.py` streams OISST, NCEP and Argo through a rented box;
`ml/fetch_glorys_daily.py` pulled GLORYS in four GitHub-Actions lanes over a
day). That works, but every rebuild re-downloads ~100 GB from NOAA and
Scripps, the NOAA PSL server throttles us after two files, the next tensor
family needs six more sources (altimetry, ERA5, sea level, soil moisture,
ENSO series …), the tensors are bespoke memmaps that only our own loader
reads, and nothing records which upstream bytes a tensor was built from.
So: **separate IMPORT from BUILD, and make the import's output a dataset
any training framework can read.** Import once, politely, in parallel, into
sharded TFRecord files of `tf.train.Example` records; every build — ours or
a foundation-model trainer's — reads those.

## 1 · The shape of the pipeline — two stages, both Beam

```
STAGE A · import (one Example per source-day)

sources.yaml ─► manifest (work items) ─► skip items whose shard is DONE
      │                                          │
      │            key = (host, lane)  lane = sha1(item) mod host.max_lanes
      ▼                                          ▼
   GroupByKey  ──►  LaneWorker (ONE lane = ONE sequential polite stream)
                      │  pace: sleep ≥ host.min_gap_s between requests
                      │  fetch: HEAD → size known → stream to disk → size check
                      │  transform: bin/fold onto the declared grid
                      │             (imported code only) → Examples
                      │  write: <output>/<source>/<item_id>.tfrecord
                      │         → read back, CRC + sha256 → write .done marker
                      │         → delete the raw download
                      │  on error: backoff ladder → circuit breaker → the item
                      │            goes to the RETRY QUEUE, never to /dev/null
                      ▼
                   result records ─► report.jsonl · retry_queue.jsonl · summary.md

STAGE B · assemble (one Example per pentad × channel group — the training set)

<output>/*/ *.tfrecord ─► Map to (bin, day-record) ─► GroupByKey(bin)
        ─► per bin: pentad mean / σ / live-month rule per channel, days_present
           mask ─► Example per (bin, group) ─► WriteToTFRecord, N shards/group
        ─► spec.json (channels, grid, norms, bins) + coverage.json
```

**A work item** (Stage A) is the smallest unit that is fetched, verified and
written as one shard: for GLORYS/DUACS a *month* (fetched day by day,
binned to 0.25°); for OISST a *year* (365 per-day files from NCEI); for NCEP
a *(variable, year)* file; for Argo RG one file; for the small series
(RAPID, cable, indices, WWV) one file each. One item → one shard →
one `.done` marker; the marker is written after the shard has been read back
and checked, so it can only under-claim (ml/CLAUDE.md §5.21).

**A lane** is one polite sequential stream to one host. Beam's `GroupByKey`
guarantees that all items sharing a key are handed to one worker as one
iterable, processed in order — so *the number of lanes per host is the
maximum number of simultaneous connections that host will ever see from
us*, whatever machine count or worker count the runner decides on. This is
the standard pattern for bounding load on an external service; it needs no
shared counter and no lock, which is why it survives autoscaling and
multiprocessing alike. **The same guarantee holds in Flume**: a key's values
are iterated by one worker; only the worker count and the sink change.

**Sources on the same host share that host's lanes.** PSL serves OISST
monthly means, NCEP R1, GODAS and OLR; they all draw on `psl`'s budget. A
new source never creates a new connection budget by accident: it must name
a host, and the host's budget is in one table.

## 2 · Politeness without loss — the rules that are load-bearing

The two goals pull against each other only if you let a throttle turn into a
drop. They are reconciled by one rule: **a throttle changes WHEN an item is
fetched, never WHETHER.** Every item ends in exactly one of three durable
states — `written` (shard verified, marker present), `queued` (in
`retry_queue.jsonl`, to be picked up by the next run), or `absent` (the
source itself does not have it — recorded with the evidence, see below).
There is no fourth state.

| rule | mechanism | why it is here |
|---|---|---|
| **Per-host lane cap** | `max_lanes` in `sources.yaml` → number of keys | the only thing that bounds concurrent connections; measured values below |
| **Minimum gap between requests** | `min_gap_s` per host; the lane sleeps before every request | a lane that gets fast answers must not turn into a tight loop |
| **Backoff ladder** | on 429/5xx/timeout/connection error: sleep 60 s, 5 min, 15 min, 60 min (× jitter 0.8–1.2), honouring `Retry-After` when sent | measured on PSL: after two back-to-back years it answers 504 then nothing for ~15 min, then lets one more through |
| **Circuit breaker, then the queue** | 5 consecutive failed items in a lane → the lane stops for this run; every remaining item of the lane is appended to `retry_queue.jsonl` | "a month that fails five times stops the lane by design" — but the month is not lost, it is *deferred to a later run*, and the queue is what makes that a promise |
| **Run until complete** | `run_until_complete.sh`: run → if the queue is non-empty, sleep (1 h, then 2, 4, 8 — the host is being given a night off) → run again with the queue as the manifest; stop only when the queue is empty or a human intervenes | a throttle is a delay, never a decision to skip |
| **`absent` needs evidence, twice** | a 404/410 on two separate runs at least 6 h apart, with both responses' status/date/headers recorded; anything less stays `queued` | "a truncated transfer raised no exception" — a source saying no once is not the same as the data not existing |
| **Gaps are masked, never skipped** | a pentad with 3–4 days is written with its `days_present` mask (and dropped to missing below `min_days` *in Stage B*, by the imported rule) — Stage A writes every day it got | Stage A's job is to lose nothing; deciding what counts as a valid pentad is a modelling rule and lives in one place |
| **Never raise out of a DoFn on a transient error** | every fetch is `try/except` → a result record | Beam **retries a failed bundle** (Dataflow: 4×; DirectRunner: fails the job; Flume: likewise) — an uncaught 504 would re-run the whole lane, i.e. re-download everything it already did, i.e. hammer the host. Only programming errors may raise |
| **Skip before fetch, twice** | the manifest is filtered against the `.done` markers once at start; the lane checks the marker again before each item | a retried bundle, or a second run, must not re-download what is already written |
| **Bytes proportional to the question** | GLORYS/DUACS fetched **one day per request** and binned to 0.25° on arrival; a whole month in one request was 5.95 GB and OOM-killed a 7 GB runner | RAM and wire are both bounded per item |
| **Write-verify-mark-delete** | shard written to a temp name → read back (CRC on every record, sha256 of every array against the in-memory copy) → renamed → `.done` written → raw download deleted | a write returning is not evidence the bytes come back (`ml/CLAUDE.md` §0.2) |
| **No credentials in argv or pipeline options** | read from env in `DoFn.setup()`; Dataflow: Secret Manager | pipeline options are logged and shown in the job UI |
| **Concurrency is small by construction** | the busy hosts are cmems 4 + ncei 6 + psl 1 ≈ 11 streams; the other thirteen hosts are one-lane and hold a handful of files each | one 4–8 vCPU machine saturates this; there is no reason to scale workers beyond it |

### The host table (the numbers, with their provenance)

| host key | what it serves | `max_lanes` | `min_gap_s` | evidence |
|---|---|---|---|---|
| `cmems` | Copernicus Marine via the `copernicusmarine` toolbox (GLORYS12, DUACS, GREP, OSTIA, statics) | **4** | 2 | Phase A ran four lanes for a day at one day-request each without a refusal; the toolbox reads from ARCO object storage built for parallel access |
| `psl` | `downloads.psl.noaa.gov` + THREDDS mirror (NCEP R1, OISST monthly, GODAS, OLR) | **1** | 20 | measured 2026-08-18: 504 + ~15 min silence after two back-to-back 477 MB files; `PACE_S = 20` and a 900 s HEAD-poll are what worked |
| `ncei` | `www.ncei.noaa.gov` OISST per-day files (sst + ice in one file) | **6** | 0.5 | measured: 12 threads did a year in under a minute; 6 leaves headroom for other users |
| `scripps` | `sio-argo.ucsd.edu` (Roemmich–Gilson) | **1** | 10 | a small academic server; ~0.7 GB base files |
| `rapid` | `rapid.ac.uk` | 1 | 10 | one file |
| `aoml` | `aoml.noaa.gov` (Florida cable, SAMBA) | 1 | 5 | small files |
| `gatech` / `ndbc` | OSNAP (`repository.gatech.edu`), MOVE (`dods.ndbc.noaa.gov`) | 1 each | 5 | one file each |
| `ncei_etopo` | `ngdc.noaa.gov` THREDDS (ETOPO 2022) | 1 | 5 | one 450 MB file |
| `github_raw` | Natural Earth GeoJSON | 2 | 1 | CDN |
| `pmel` | `pmel.noaa.gov` (warm-water volume, TAO) | 1 | 5 | tiny |
| `cpc` / `bom` | index series (ONI, RMM) | 1 each | 5 | tiny |
| `metoffice` | EN4 (`metoffice.gov.uk/hadobs`) | 1 | 10 | ~30 MB zips |
| `cds` | Copernicus Climate Data Store (ERA5, ORAS5) — **blocked on the account** | 2 | 5 | CDS queues requests server-side; two in flight is the polite maximum for one user |
| `ceda` | ESA CCI soil moisture (needs free registration) | 2 | 5 | — |

Change a number here and nowhere else — and only downward without a
measurement. The registry is validated at start.

## 3 · What is imported, in tiers

**Tier 0 — what the current global tensor (family 7, recipe `f7l0`) is built
from.** No new account needed. GLORYS12 global daily, binned to 0.25° at
fetch (384 months; the Copernicus login is in `CREDENTIALS.md`); OISST v2.1
daily SST + sea-ice concentration 1981-09 → 2024-12 (from NCEI's per-day
files; PSL yearly files as a separately-enabled fallback); NCEP/NCAR
Reanalysis 1 surface gaussian, 13 files per year × 1982–2024 plus the land
mask (560 files; the archive is frozen since March 2026, so this import is
also its last chance); Roemmich–Gilson Argo (2 base files + every monthly
extension); the statics (ETOPO 2022 60 s, Natural Earth glaciated areas +
lakes); the labels (RAPID 26.5° N, Florida cable, OSNAP, MOVE, SAMBA).
Measured from the registry: **1,131 items, ~950 GB on the wire** (840 of it
GLORYS), **~200 GB written**. Wall clock is set by the slowest lane: GLORYS
at four CMEMS lanes ≈ 13 h; PSL's single lane ≈ 6.5 h (derived, not yet
measured). *If GLORYS is already mirrored somewhere the agent can read
(it is, as NetCDF on the Hub under `daily025_global/`), the registry's
`glorys_from_mirror` source converts that copy instead of refetching.*

**Tier 1 — the next family's channels that need no new account.** DUACS L4
altimetry (`SEALEVEL_GLO_PHY_L4_MY_008_047`: `sla`, `adt`, `ugos`, `vgos`,
0.125° daily 1993 →, fetched day by day and binned to 0.25° per month,
~1.1 TB wire → ~95 GB written — the first *observed* circulation channel);
the CMEMS static bathymetry/mask for the GLORYS grid; RAPID's unused
`moc_vertical` and `ts_gridded`; EN4.2.2 monthly 1900 → (the
backward-extension measurement); PMEL warm-water volume; NOAA interpolated
OLR; GODAS pentad (PSL); the ENSO index series (ONI, PSL indices, MEI v2,
RMM).

**Tier 2 — gated.** ERA5 daily statistics via CDS (14 variables, the
momentum + buoyancy + shared land/ocean set; **blocked on the free CDS
account only Chris can create** — fully specified, the fetcher refuses
politely until `CDSAPI_KEY` exists); ESA CCI soil moisture (free
registration); OSTIA SST + sea-ice fraction (0.05°: ~3 TB on the wire for a
sea-ice fraction OISST already supplies — off by default); GREP 3-D at 8
levels (~200 GB written globally; E-070 §6 says the streaming loader comes
first — off by default, North-Atlantic window available as an option).

Every item carries `tier`, so `--tiers 0` and `--tiers 0,1` are the two
ordinary invocations.

## 4 · The output — two `tf.train.Example` schemas

Everything is float32 little-endian raw bytes in `[C, H, W]` order with a
bit-packed validity mask; units are the source's own (no z-scoring in Stage
A — normalisation is a training decision and lives in Stage B's `spec.json`).

**Stage A — one Example per source-day** (`<output>/<source>/<item_id>.tfrecord`,
records in date order inside a shard; `<item_id>.done` beside it carrying the
shard's byte count, sha256 and record count):

| feature | type | meaning |
|---|---|---|
| `source`, `item_id` | bytes | registry name, e.g. `oisst`, `oisst/1993` |
| `date` | bytes | ISO day (`1993-01-01`); for monthly products the 15th |
| `day_index` | int64 | days since 1982-01-01 (the tensor epoch); `bin = day_index // 5` |
| `grid` | bytes | `point025` (our 0.25° point grid, −90…90 × −180…179.75) · `oisst_center025` · `ncep_t62` · `rg_1deg_center` · `point0125` … — the name of a grid the reader can rebuild from the four axis features |
| `lat0`, `lat_step`, `nlat`, `lon0`, `lon_step`, `nlon` | float / int64 | the axes, explicit, so no reader needs a lookup table |
| `var_names`, `var_units` | bytes list | channel names and units, in `C` order (`uo`, `vo`, `mlotst`, `zos` …) |
| `values` | bytes | float32 LE `[C, H, W]`; NaN where the source has no value |
| `mask` | bytes | bit-packed `[C, H, W]`, 1 = finite in `values` |
| `shape` | int64 list | `[C, H, W]` |
| `source_url`, `source_bytes`, `source_sha256`, `fetched_at` | bytes / int64 | provenance of the upstream file the record came from |
| `transform` | bytes | `none` · `bin025:nearest-scatter:aggregate_cadence.bin_slice` · `oisst_year_fold` — what was done to the source bytes |

Non-gridded sources (labels, indices, WWV) use the same schema with
`grid = series`, `nlat = nlon = 1`, and `date` per record.

**Stage B — one Example per (pentad bin, channel group)** — the training set
in the family-7 layout (`g025` 7 ch at 0.25°, `g100` 14 ch at 1°, `rg100`
32 ch at 1° on live months only), `<output>/pentad/<group>/part-SSSSS-of-NNNNN.tfrecord`:

| feature | meaning |
|---|---|
| `bin`, `date_start`, `date_end` | pentad index from 1982-01-01 and its five days |
| `group`, `chan_names` | `g025` / `g100` / `rg100` and the channel order of E-070's build spec |
| `values`, `mask`, `shape`, axes | as Stage A; values are the pentad mean (σ for `tau_*_std`, `log10`/`log1p` where the spec says, the live-month rule for `rg100`) |
| `days_present` | int64 list, per channel: how many of the five days contributed — the `min_days = 3` rule is applied here and only here |
| `sources` | bytes list: the Stage-A `item_id`s the bin was built from |

Beside the shards: `spec.json` (groups, channels, transforms, grid, bins,
per-channel mean/sd over train years — the numbers a trainer needs to
z-score, kept OUT of the data so the split is a trainer decision) and
`coverage.json` (per group: bins present, bins missing and why). A reader
in TensorFlow is `tf.data.TFRecordDataset(glob)` + `parse_single_example`;
in JAX/NumPy the pure-Python reader in `beam_import/tfrecord.py` does the
same with no TensorFlow import at all.

## 5 · Where it runs, and why not the GPU boxes

Recommended: **one Linux VM with 4–8 vCPU, 16 GB RAM, 200 GB disk, running
the DirectRunner in multi-process mode** (`--direct_running_mode
multi_processing --direct_num_workers 8`), writing to a local directory or
straight to `gs://`. The lane count is what bounds throughput; a single
machine saturates it, and a single machine is where credentials are easiest
to keep out of logs.

Dataflow (or Flume, for an agent inside Google) is a runner change: the
pipeline is the same code, the keys give the same guarantee; set the worker
count so that `workers × threads` matches the registry’s total lane count (29 today) and no higher, and
pass credentials through the runner's secret mechanism, never options.
Stage B is where a distributed runner helps at all — it is a wide reduce
over ~3 TB of Stage-A records — and it needs no politeness, only bandwidth
to the output store.

**Never on a rented GPU box** (Vast) for anything that needs the CMEMS or
CDS credentials: `ml/CLAUDE.md` §6 — the host has root, and nothing that
outlives a job may be stored there.

## 6 · What "done" means, and how it is checked

1. `manifest --tiers 0 --print` lists every work item with its shard path,
   host, lane and expected size; the counts per source match the README.
2. After `run_until_complete.sh`: `retry_queue.jsonl` is **empty**, every
   manifest item has a `.done` marker, and `report.jsonl` carries the
   sha256 the read-back compared for each.
3. `verify_output --tiers 0` re-reads every shard's marker, checks byte
   counts, and reports `missing 0`, plus the list of `absent` records with
   their two-run evidence — that list is reviewed by a human before Stage B.
4. `summary.md`: bytes per host, wall time per host, backoffs and breaker
   trips per host — the politeness audit; a host with trips means the
   budget in `sources.yaml` is too generous and should be lowered.
5. Stage B: `coverage.json` shows every bin from 1982-01-01 to 2024-12-31
   present for `g025` and `g100` (a missing bin must trace to an `absent`
   record), and `rg100` present for every live month; a spot-check
   un-z-scores one `g025` bin's `cur_u` North-Atlantic sub-block against
   family 4's and reports max |Δ|.

## 7 · What this deliberately does not do

- It does not z-score, split or shuffle. Those are training decisions;
  `spec.json` carries what a trainer needs to make them.
- It does not re-implement a rule: binning is `ml/aggregate_cadence
  .bin_plan/bin_slice`, the pentad rules are `build_family7`'s, the bin
  definition is `floor(day_index / 5)` from 1982-01-01, all **imported**.
- It does not run on a schedule; every run is a dispatch, every run is
  idempotent, and `run_until_complete.sh` is the only loop.
- It does not delete anything it did not write, anywhere.
- It does not create accounts. ERA5 waits for Chris.
