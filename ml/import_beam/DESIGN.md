# Parallel import of the training data — the design (E-073)

**Written 2026-09-04 by the Fable session, for Chris and for the agent that
will run it.** Plain English first: this is a program that downloads every
upstream dataset the Earth model's input tensors are built from, in parallel
across many sources at once, and puts a verified copy of each on our
Hugging Face dataset repo — *without ever hammering any of the public
servers it reads from.* It is built on Apache Beam (a Python framework for
data pipelines that runs the same code on one laptop, one VM, or Google
Cloud Dataflow) — but Beam is the **orchestrator**, not the accelerator: the
speed limit is set by politeness towards each host, and the design's whole
point is that the limit is enforced by construction, not by good intentions.

Companion documents in this package: `README_FOR_GEMINI.md` (the step-by-step
howto), `sources.yaml` (the registry — every source, every host budget),
`CREDENTIALS.md`, and the code under `beam_import/`.

---

## 0 · Why this exists, in one paragraph

Today every tensor build fetches its sources straight from the upstream
archives, one file at a time, on whatever machine the build runs on
(`ml/build_family7.py` streams OISST, NCEP and Argo through a Vast box;
`ml/fetch_glorys_daily.py` pulled GLORYS in four GitHub-Actions lanes over a
day). That works, but every rebuild re-downloads ~100 GB from NOAA and
Scripps, the NOAA PSL server throttles us after two files, the next tensor
family needs six more sources (altimetry, ERA5, sea level, soil moisture,
ENSO series …), and nothing records which upstream bytes a tensor was built
from. So: **separate IMPORT from BUILD.** Import once, politely, in parallel,
into a verified mirror on the Hub (`chfrank/earth-tensors` under
`sources/`); every build reads the mirror. The Hub is the resume point, the
provenance record, and the thing that makes two builds "the same bytes".

## 1 · The shape of the pipeline

```
sources.yaml ─► manifest (work items) ─► skip what the Hub already has
      │                                          │
      │            key = (host, lane)  lane = hash(item) mod host.max_lanes
      ▼                                          ▼
   GroupByKey  ──►  LaneWorker (ONE lane = ONE sequential polite stream)
                      │  pace: sleep ≥ host.min_gap_s between requests
                      │  fetch: HEAD → size known → stream to disk → size check
                      │  transform: bin/fold onto our grid (imported code only)
                      │  publish: batch commit to the Hub → download back →
                      │           sha256 equal → delete local copy
                      │  on error: backoff ladder, then circuit breaker
                      ▼
                   result records ─► report.jsonl + summary.md
```

**A work item** is the smallest unit that is fetched, verified and published
as one: for GLORYS/DUACS a *month* (fetched day by day, binned, one NetCDF);
for OISST a *year* (365 daily files from NCEI folded into one file); for
NCEP a *(variable, year)* file; for Argo RG one file; for the small series
(RAPID, cable, indices, WWV) one file each.

**A lane** is one polite sequential stream to one host. Beam's `GroupByKey`
guarantees that all items sharing a key are handed to one worker as one
iterable, processed in order — so *the number of lanes per host is the
maximum number of simultaneous connections that host will ever see from
us*, whatever machine count or worker count the runner decides on. This is
the standard Beam pattern for bounding load on an external service; it
needs no shared counter and no lock, which is why it survives Dataflow's
autoscaling and DirectRunner's multiprocessing alike.

**Sources on the same host share that host's lanes.** PSL serves OISST
monthly means, NCEP R1, GODAS and OLR; they all draw on `psl`'s budget of 2
lanes. A new source never creates a new connection budget by accident: it
must name a host, and the host's budget is in one table.

## 2 · Politeness — the rules that are load-bearing

| rule | mechanism | why it is here |
|---|---|---|
| **Per-host lane cap** | `max_lanes` in `sources.yaml` → number of keys | the only thing that bounds concurrent connections; measured values below |
| **Minimum gap between requests** | `min_gap_s` per host; the lane sleeps before every request | a lane that gets fast answers must not turn into a tight loop |
| **Backoff ladder** | on 429/5xx/timeout/connection error: sleep 60 s, 5 min, 15 min, 60 min (× jitter 0.8–1.2), honouring `Retry-After` when sent | measured on PSL: after two back-to-back years it answers 504 then nothing for ~15 min, then lets one more through |
| **Circuit breaker** | 5 consecutive failures in a lane → the lane stops and marks the rest of its items `deferred` (not failed); a re-run picks them up | "a month that fails five times stops the lane by design" — the rule Phase A already ran under |
| **Never raise out of a DoFn on a transient error** | every fetch is `try/except` → a result record | Beam **retries a failed bundle** (Dataflow: 4×; DirectRunner: fails the job) — an uncaught 504 would re-run the whole lane, i.e. re-download everything it already did, i.e. hammer the host. Only programming errors may raise |
| **Skip before fetch, twice** | the manifest is filtered against a Hub listing taken once at start; the lane HEADs the Hub path again before each item | a retried bundle, or a second run, must not re-download what is already mirrored |
| **Bytes proportional to the question** | GLORYS/DUACS fetched **one day per request** and binned to 0.25° on arrival; a whole month in one request was 5.95 GB and OOM-killed a 7 GB runner | RAM and wire are both bounded per item |
| **Commit budget on the Hub** | lanes batch `batch_files` items per commit; the pipeline as a whole stays under ~60 commits/hour; 429 → `Retry-After` | the Hub enforces an hourly commit quota ("You have exceeded our hourly quotas for action: commit") and per-file commits at 600 files would trip it |
| **Restore-verify** | after upload: download back, sha256 equal, then and only then delete the local copy | an upload returning 200 is not evidence the bytes come back (`ml/CLAUDE.md` §0.2); every prior pipeline in this repo does this |
| **No credentials in argv or pipeline options** | read from env in `DoFn.setup()`; Dataflow: Secret Manager | pipeline options are logged and shown in the job UI; the permission classifier blocks tokens on command lines for the same reason |
| **Concurrency is small by construction** | the busy hosts are cmems 4 + ncei 6 + psl 1 ≈ 11 streams; the other thirteen hosts are one-lane and hold a handful of files each (27 keys in all) | one 4–8 vCPU machine saturates this; there is no reason to scale workers beyond it |

### The host table (the numbers, with their provenance)

| host key | what it serves | `max_lanes` | `min_gap_s` | evidence |
|---|---|---|---|---|
| `cmems` | Copernicus Marine via the `copernicusmarine` toolbox (GLORYS12, DUACS, GREP, OSTIA, statics) | **4** | 2 | Phase A ran four lanes for a day at one day-request each without a refusal; the toolbox reads from ARCO object storage built for parallel access |
| `psl` | `downloads.psl.noaa.gov` + THREDDS mirror (NCEP R1, OISST monthly, GODAS, OLR) | **1** | 20 | measured 2026-08-18: 504 + ~15 min silence after two back-to-back 477 MB files; `PACE_S = 20` and a 900 s HEAD-poll are what worked |
| `ncei` | `www.ncei.noaa.gov` OISST per-day files (sst + ice in one file) | **6** | 0.5 | measured: 12 threads did a year in under a minute; 6 leaves headroom for other users |
| `scripps` | `sio-argo.ucsd.edu` (Roemmich–Gilson) | **1** | 10 | a small academic server; ~2 GB base files |
| `rapid` | `rapid.ac.uk` | 1 | 10 | one file |
| `aoml` | `aoml.noaa.gov` (Florida cable, SAMBA) | 1 | 5 | small files |
| `gatech` / `ndbc` | OSNAP (`repository.gatech.edu`), MOVE (`dods.ndbc.noaa.gov`) | 1 each | 5 | one file each |
| `ncei_etopo` | `ngdc.noaa.gov` THREDDS (ETOPO 2022) | 1 | 5 | one 450 MB file |
| `github_raw` | Natural Earth GeoJSON | 2 | 1 | CDN |
| `pmel` | `pmel.noaa.gov` (warm-water volume, TAO) | 1 | 5 | tiny |
| `cpc` / `bom` | index series (ONI, MEI, RMM) | 1 | 5 | tiny |
| `metoffice` | EN4 (`metoffice.gov.uk/hadobs`) | 1 | 10 | ~30 MB zips |
| `cds` | Copernicus Climate Data Store (ERA5, ORAS5) — **blocked on the account** | 2 | 5 | CDS queues requests server-side; two in flight is the polite maximum for one user |
| `ceda` | ESA CCI soil moisture (needs free registration) | 2 | 5 | — |
| `hf` (publish) | `huggingface.co` commits | (per lane) `batch_files`, ≤ 60 commits/h total | — | hourly commit quota; verified restore |

Change a number here and nowhere else. The registry is validated at start:
every source names a host, every host has both numbers.

## 3 · What is imported, in tiers

**Tier 0 — what the current global tensor (family 7, recipe `f7l0`) is built
from, mirrored so the next build reads the Hub.** No new account needed.
GLORYS12 global monthly chunks (384, **already on the Hub** — the pipeline
only verifies their presence); OISST v2.1 daily SST + sea-ice concentration
1981-09 → 2024-12, one file per year (from NCEI's per-day files, PSL yearly
files as fallback); NCEP/NCAR Reanalysis 1 surface gaussian, 13 files per year (the 14
`g100` channels come from 13 variables — the two stress σ channels reuse the
stress files) × 1982–2024 plus the land mask (560 files; the archive is
frozen since March 2026, so this mirror is also its last chance); Roemmich–Gilson Argo (2 base
files + every monthly extension); the statics (ETOPO 2022 60 s, Natural
Earth glaciated areas + lakes); the labels (RAPID 26.5° N, Florida cable,
OSNAP, MOVE, SAMBA). Measured from the registry: **747 items to fetch,
104 GB on the wire, 104 GB stored**, plus the 384 GLORYS months verified in
place. Wall clock ≈ 6.5 h, set entirely by PSL's single lane (derived, not
yet measured).

**Tier 1 — the next family's channels that need no new account.** DUACS L4
altimetry (`SEALEVEL_GLO_PHY_L4_MY_008_047`: `sla`, `adt`, `ugos`, `vgos`,
0.125° daily 1993 →, fetched day by day and binned to 0.25° per month, ~1.1 TB
wire → ~95 GB stored — the first *observed* circulation channel); the CMEMS
static bathymetry/mask for the GLORYS grid; RAPID's unused `moc_vertical`
and `ts_gridded`; EN4.2.2 monthly 1900 → (the backward-extension
measurement); PMEL warm-water volume; NOAA interpolated OLR; GODAS pentad
(PSL); the ENSO index series (ONI, PSL indices, MEI v2, RMM).

**Tier 2 — gated.** ERA5 daily statistics via CDS (12 variables, the
momentum + buoyancy + shared land/ocean set; **blocked on the free CDS
account only Chris can create** — the source is fully specified, the fetcher
refuses politely until `CDSAPI_KEY` exists); ESA CCI soil moisture (free
registration); OSTIA SST + sea-ice fraction (0.05°: ~3 TB on the wire for a
sea-ice fraction OISST already supplies — off by default); GREP 3-D at 8
levels (~200 GB stored globally; E-070 §6 says the streaming loader comes
first — off by default, North-Atlantic window available as an option).

Every item carries `tier`, so `--tiers 0` and `--tiers 0,1` are the two
ordinary invocations.

## 4 · Where it runs, and why not the GPU boxes

Recommended: **one Linux VM with 4–8 vCPU, 16 GB RAM, 100 GB disk, running
the DirectRunner in multi-process mode** (`--direct_running_mode
multi_processing --direct_num_workers 8`). The total lane count (~12) is what
bounds throughput; a single machine saturates it, and a single machine is
where credentials are easiest to keep out of logs. A Google Compute Engine
`e2-standard-8` is about $0.27/h; Tier 0 is a few hours, Tier 1 (DUACS) a
couple of days at four CMEMS lanes.

Cloud Dataflow is a flag change (`--runner DataflowRunner --max_num_workers 3
--number_of_worker_harness_threads 4`) and buys nothing here except
managed retries — which, per §2, we do not want at the bundle level. Use it
only if the machine, not the hosts, turns out to be the bottleneck (the
binning step is CPU-cheap: 37 s per GLORYS month).

The GitHub-hosted-runner lane pattern of Phase A remains a zero-cost
fallback: the pipeline is resumable from the Hub, so a 350-minute cap just
means more firings.

**Never on a rented Vast box** for anything that needs the CMEMS or CDS
credentials: `ml/CLAUDE.md` §6 — the host has root, and nothing that
outlives a job may be stored there. Tier 0 needs no credential except the
Hub token and could run anywhere.

## 5 · What "done" means, and how it is checked

1. `python -m beam_import.manifest --tiers 0 --print` lists every work item
   with its Hub path, host, lane and expected size; the counts per source
   match the table in the README (86 OISST years, 603 NCEP files, …).
2. The pipeline's `report.jsonl` has one record per item with status in
   {`present`, `published`, `deferred`, `failed`} and, for `published`, the
   sha256 that the restore-verify compared.
3. `python -m beam_import.verify_hub --tiers 0` lists the Hub and reports
   *missing* = 0 for the tier.
4. `sources/MANIFEST_tier0.json` on the Hub: one record per file — Hub path,
   bytes, sha256, upstream URL/product, fetch timestamp — written in a
   single final commit from the report.
5. A one-page summary (`summary.md`) with bytes per host, wall time per
   host, number of backoffs and breaker trips per host — that last column
   is the politeness audit; a host with many trips means the budget in
   `sources.yaml` is too generous and should be lowered before the next
   run.

## 6 · What this deliberately does not do

- It does not build tensors. `ml/build_family7.py` and its successors read
  the mirror; the binning inside the import uses `ml/aggregate_cadence
  .bin_plan/bin_slice` **imported**, never re-implemented — a second
  implementation of a bin rule is the defect class this repo has already
  paid for.
- It does not touch the North-Atlantic GLORYS chunks at the dataset root
  or the `daily025_global/` chunks: those are family-4/family-7 inputs with
  their own provenance and stay where they are.
- It does not run on a schedule. Every run is a dispatch, and every run is
  idempotent.
- It does not create accounts. ERA5 waits for Chris.
