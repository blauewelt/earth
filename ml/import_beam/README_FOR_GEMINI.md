# How to run the import — the step-by-step

**Read this file top to bottom before you type anything.** Every command in it
is meant to be copied exactly. Where a command has an expected output, the
expected output is written underneath it; if what you see differs in a way this
document does not explain, **stop and write down what you saw** rather than
improvising a fix.

Companion files in this directory:

- [`DESIGN.md`](DESIGN.md) — why the thing is built this way (revision 2). Read it once.
- [`sources.yaml`](sources.yaml) — the registry: every dataset, every host budget.
- [`CREDENTIALS.md`](CREDENTIALS.md) — the two credentials and the rules about them.
- `beam_import/` — the code. `tests/` — the tests.

Background in the main repository (all three render on a phone):

- [E-070 family-7 build plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md)
- [E-070 global tensor plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_global_tensor.md)
- [The data ladder](https://blauewelt.github.io/earth/docs.html?f=ml/plans/DATA_LADDER.md)

---

## 0 · What you are doing, and what done looks like

You are going to **copy a set of public climate datasets into one place,
carefully, without overloading any of the servers you copy them from, and
without ever losing a day of data.** That is the whole job. You are not
training anything and not analysing anything. You download files, you check
each one arrived intact, you convert it into a standard machine-learning record
format, you write it out, you check it reads back unchanged, and you write down
what happened. Anything that did not work goes into a queue and is tried again
later — it is never dropped.

The datasets are the inputs a machine-learning model of the ocean is built
from, and they live on a dozen public servers run by national weather and ocean
agencies. Today every rebuild of the model downloads them all again, which is
wasteful and rude, so the plan is: **import once, into a dataset any training
framework can read; every future build reads that.** The output format is
**TFRecord** — a file format that is just a sequence of length-prefixed,
checksummed byte blobs — holding **`tf.train.Example`** records, which are the
standard key/value record protocol buffer. You do not need TensorFlow to read
them; this package ships a pure-Python reader. Where the output goes is
**yours to choose**: `--output /some/directory` or `--output gs://some-bucket/path`.

The tool that runs the copying is **Apache Beam**, a Python framework for data
pipelines and the open-source form of Google's Flume. **DirectRunner** is
Beam's mode for running on one ordinary machine and is what you will use;
**Dataflow** is Google's paid cloud mode, and **Flume** is the internal
equivalent — §9 maps the concepts across if you are running there.

The names you will meet, each spelled out once. **GLORYS12** — a global model
reconstruction of ocean currents, sea-surface height and mixed-layer depth from
Copernicus, at 1/12 of a degree. **OISST** — the Optimum Interpolation Sea
Surface Temperature record from NOAA: daily sea-surface temperature and sea-ice
concentration since September 1981. **NCEP R1** — the NCEP/NCAR Reanalysis 1, a
decades-long reconstruction of surface winds, heat fluxes and soil conditions;
its archive was frozen in March 2026, so this copy is the last chance to take
one. **Roemmich–Gilson** — a climatology of ocean temperature and salinity with
depth, built from the Argo float network at Scripps. **DUACS** — the system
that turns satellite altimeter passes into daily maps of sea-surface height;
the Tier-1 heavyweight, about 1.1 terabytes on the wire. **ERA5** — the
European Centre's global atmospheric reanalysis, reached through the **CDS**
(Climate Data Store), which needs a free account that does not exist yet.
**CMEMS** — Copernicus Marine Environment Monitoring Service, the ocean
download service, for which we *do* have credentials. **Family 7** — the name
of the current input data cube the model trains on, the first one covering the
whole globe. A **pentad** is a five-day period, the time step that cube uses; a
**bin** is one pentad, numbered from 1982-01-01.

**Done is DESIGN §6, five checks, in order.**

1. `manifest --tiers 0 --print` lists every work item with its shard path,
   host, lane and expected size, and the counts per source match §3(b) below.
2. After `run_until_complete.sh`: **`retry_queue.jsonl` is empty**, every
   manifest item has a `.done` marker, and `report.jsonl` carries the sha256
   that the read-back compared for each.
3. `verify_output --tiers 0` reports **`missing 0`**, plus the list of
   `absent` items with their two-run evidence — **a human reviews that list
   before Stage B.**
4. `summary.md` shows bytes, wall time, backoffs and breaker trips per host —
   the politeness audit. A host with trips means its budget in `sources.yaml`
   is too generous and should come DOWN.
5. Stage B: `coverage.json` shows every bin from 1982-01-01 to 2024-12-31
   present for `g025` and `g100`, and `rg100` present for every live month.
   **A missing bin must trace to an `absent` record.**

---

## 1 · Hard rules

These are not style preferences. Each one exists because breaking it has cost
this project real time or real money.

1. **Never drop data.** Every item ends `written`, `present`, `queued` or
   `absent`. There is no `failed`, and you may not add one. If something did
   not work, it belongs in `retry_queue.jsonl`.
2. **`absent` needs evidence, twice.** A 404 or 410 on two separate runs at
   least six hours apart, with both responses recorded. One "no" is a bad
   afternoon, not a fact about the archive.
3. **Never put a credential on a command line, in a Beam pipeline option, or
   in anything you print.** Options are logged and shown in job UIs. The code
   reads credentials from environment variables inside `DoFn.setup()`.
4. **Never raise a host budget.** `max_lanes` and `min_gap_s` in
   `sources.yaml` may be lowered, never raised, and never on your own
   initiative. If a run is slow, it is supposed to be slow.
5. **Never run this on a rented or borrowed machine** (a Vast box, a shared CI
   runner, anyone's GPU box) while the CMEMS or CDS credentials are in the
   environment. The host has root and can read everything.
6. **Never re-implement a rule.** The binning is `bin_plan`/`bin_slice`, the
   pentad clock is `EPOCH`/`bin_index`, the channel lists and the pentad rules
   are `build_family7`'s, the regridding is `build_family3`'s — all imported
   from the earth checkout. If an import fails, fix the checkout; do not write
   your own version, not even a small one, not even temporarily.
7. **If a host's circuit breaker trips twice in one run, stop and report.**
   `run_until_complete.sh` stops by itself in that case and exits 4. Once is
   normal and self-healing; twice is a decision for a human.
8. **Never delete anything you did not write.** Nothing in this package
   deletes anything else; keep it that way.
9. **Never schedule this.** Every run is a manual dispatch.
   `run_until_complete.sh` is the only loop there is.
10. **Never edit `beam_import/*.py` to make a stubborn item pass.** Report the
    item with its exact error text instead.
11. **Run `bash run_smoke.sh` before every real run.** It is offline and takes
    about a minute.
12. **If you are unsure, stop and write down what you observed.** A stopped
    run costs an hour; a wrong run costs a week and looks like it worked.

---

## 2 · The machine and the environment

**The machine.** One ordinary Linux virtual machine: 4–8 virtual CPUs, 16 GB of
RAM, **200 GB of free disk**. Not a GPU box (rule 5). Throughput is set by how
politely the pipeline talks to the servers, not by the machine, so a bigger
machine buys nothing for Stage A. Stage B is the one part that would use a
distributed runner (§9).

### 2.1 Get the code

```bash
cd ~
git clone https://github.com/blauewelt/earth.git
export EARTH_REPO=~/earth
cd ~/handover           # this directory
```

`EARTH_REPO` is how the code finds the rules it imports. If it is unset, the
code looks for `../earth` next to this directory.

### 2.2 Build the virtual environment

```bash
bash setup_env.sh
source beamenv/bin/activate
```

**Why the script and not a bare `pip install`.** `setup_env.sh` installs
`setuptools<70` **first, on its own**. Three of Apache Beam's dependencies —
`crcmod`, `dill` 0.3.1.1 and `hdfs` — are still source distributions that call
setuptools APIs removed in setuptools 70. With a current setuptools their
builds fail part-way and leave a virtual environment where `import apache_beam`
works and running a pipeline does not. This was measured, not guessed.
(`crcmod` is also what computes the TFRecord checksums, so it is not optional.)

Expected tail of the output:

```
apache-beam        2.68.0
netCDF4            1.7.4
numpy              2.2.6
xarray             2026.7.0
copernicusmarine   2.4.1
crcmod             present (the TFRecord checksums)
```

TensorFlow is **not** required and **not** installed by default: the package
reads and writes TFRecord and `tf.train.Example` with its own code. If you want
to cross-check with TensorFlow, `pip install tensorflow-cpu` — but be aware it
upgrades `protobuf` past what apache-beam pins. Beam kept working in testing,
but the default venv leaves it out on purpose.

### 2.3 Put the credentials in the environment

Open [`CREDENTIALS.md`](CREDENTIALS.md) and copy the export block. **The
Copernicus Marine password contains a `%` and must be in single quotes** — that
is the single most common way this setup goes wrong:

```bash
export COPERNICUSMARINE_SERVICE_USERNAME='cfrank1'
export COPERNICUSMARINE_SERVICE_PASSWORD='<the Copernicus password from CREDENTIALS.md — it contains a %>'   # single quotes!
export CDSAPI_URL='https://cds.climate.copernicus.eu/api'
```

There is **no Hugging Face token and no credential file on disk** in revision 2.

### 2.4 Choose where the output goes

```bash
export OUTPUT=/data/import            # or gs://your-bucket/earth-import
export STATE_DIR=/var/tmp/beam_import
```

`--output` accepts anything Beam has a filesystem for. Everything under it is
written by this package and nothing else lives there.

---

## 3 · Preflight

Do all five. They take about two minutes together and they are the difference
between finding a problem now and finding it three hours in.

### (a) The registry parses and every source names a host

```bash
python -m beam_import.registry --check
```

Expected — some `warning:` lines about unverified URLs and disabled sources
(those are correct), then:

```
registry  /home/…/handover/sources.yaml
output    <output>/<item_id>.tfrecord  +  <item_id>.done (64 shards per group in Stage B)
hosts     17   total lanes 29
sources   27
  tier 0: 13  glorys glorys_from_mirror oisst oisst_psl ncep rg rapid florida_cable osnap move samba etopo natural_earth
  tier 1: 10  duacs cmems_static rapid_extra en4 pmel_wwv olr godas oni mei rmm
  tier 2: 4  era5 cci_sm ostia grep3d
OK
```

If it prints `REGISTRY INVALID`, read the reasons; each names a source or host
and what is missing. Do not run anything else until it says `OK`.

### (b) The manifest expands and the counts are right

```bash
python -m beam_import.manifest --tiers 0 --print | tail -30
```

**The expected Tier-0 table, measured 2026-09-04:**

| source | items | host | what one item is |
|---|---:|---|---|
| `glorys` | 384 | cmems | one month, fetched **one day per request** and binned to 0.25° |
| `oisst` | 44 | ncei | one year, ~365 per-day files, one record each |
| `ncep` | 560 | psl | one (variable, year) file: 13 variables × 43 years + the land mask |
| `rg` | 93 | scripps | 2 base files + one per monthly extension (**scraped**, see below) |
| `rapid` | 1 | rapid | one file |
| `florida_cable` | 43 | aoml | one year |
| `osnap` · `move` · `samba` | 1 each | gatech · ndbc · aoml | one file each |
| `etopo` | 1 | ncei_etopo | one file (~450 MB) |
| `natural_earth` | 2 | github_raw | two GeoJSON files |

and the last lines of the output are:

```
  to fetch: 1131 items · 925.1 GB wire · 199.9 GB stored
  TOTAL 1131 items · 0 with an unverified URL
```

**925 GB, of which about 840 is GLORYS.** If you enable `glorys_from_mirror`
instead (§4.6) that drops to about **185 GB**. `glorys_from_mirror` and
`oisst_psl` are `enabled: false`, so they do not appear in the counts unless
you ask for them with `--only`.

Two numbers move legitimately and neither is an error:

- **`rg` is 93 only if the scrape worked.** The Roemmich–Gilson monthly
  extension list is read off `https://sio-argo.ucsd.edu/RG_Climatology.html`.
  On 2026-09-04 that gave 91 extensions plus 2 base files = 93. If the page
  cannot be read, the code warns and falls back to the declared range
  2019-01 … 2025-12, giving **86** and a Tier-0 total of **1,124 items,
  920.5 GB**. Add `--offline` to force the fallback.
- **`ncep` is 560, and older notes say 603 with 14 variables.** The registry is
  the truth: the repository's own variable table (`ml/build_family7.py`) has
  **thirteen** stems, so 13 × 43 + 1 = 560. Report what the manifest prints.

For all three tiers:

```bash
python -m beam_import.manifest --tiers 0,1,2 --offline | tail -40
```

Expected: Tier 1 adds **519** items (`duacs` 384, `en4` 126, the rest small),
Tier 2 adds **7,224** (ERA5: 14 variables × 516 months), grand total
**8,867 items · 2.0 TB wire · 369.6 GB stored**, of which **518 carry
`unverified_url: true`** — listed on purpose so a human can see what *would*
be attempted; they are not ready to run.

### (c) The offline smoke test

```bash
bash run_smoke.sh
```

It generates its own fixtures, uses `file://` URLs instead of the internet and
a local directory as the output, and exercises Stage A, the retry queue,
`run_until_complete` (with the sleep overridden to one second), Stage B and
`verify_output`. It must end with:

```
  ok   Stage A wrote every good source (opaque, 0.25° ocean, OISST, NCEP, RG)
  ok   every marker carries the sha256 its read-back compared
  ok   the OISST year kept its four days and recorded the missing one
  ok   the shard holds the four days that WERE served
  ok   there is no `failed` status anywhere
  ok   the missing day was queued as a day-level item
  ok   all seven flaky items reached the queue (breaker + ladder), none lost
  ok   one circuit-breaker trip per round was recorded
  ok   run_until_complete rotated round 1's queue and re-ran from it
  ok   Stage B produced all three channel groups for the fixture's one bin
  ok   the bin is 1573 — floor(day_index/5) from 1982-01-01, imported
  ok   one g025 record for one bin
  ok   days_present is per channel and honest: [5, 5, 5, 5, 5, 5, 4]
  ok   the channel order is build_family7's, imported

SMOKE TEST: PASS
```

Anything other than `SMOKE TEST: PASS` means **do not start a real run.** The
finer-grained version is:

```bash
python -m pytest tests -q
```

Expected: `90 passed`.

### (d) The dry run

```bash
python -m beam_import.pipeline --tiers 0 --dry-run \
    --output "$OUTPUT" --state-dir "$STATE_DIR" \
    --report-dir out/dryrun --runner DirectRunner
head -2 out/dryrun/report.jsonl
```

Every record comes back with `"status": "present"` and a `reason` saying what
would have been fetched. A dry run makes no requests of any kind and needs no
credentials.

### (e) One HEAD request per host

The only network check you make by hand, and **exactly one request per host**.
Do not loop, and do not retry a host that refuses.

```bash
python - <<'PY'
import requests
URLS = {
 "psl":        "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/land.sfc.gauss.nc",
 "ncei":       "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/198109/oisst-avhrr-v02r01.19810901.nc",
 "scripps":    "https://sio-argo.ucsd.edu/pub/www-argo/RG/RG_ArgoClim_Temperature_2019.nc.gz",
 "rapid":      "https://rapid.ac.uk/sites/default/files/rapid_data/moc_transports.nc",
 "aoml":       "https://www.aoml.noaa.gov/ftp/phod/WBTS/cable/FC_cable_transport_2020_v3.dat",
 "ncei_etopo": "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc",
 "github_raw": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_lakes.geojson",
 "gatech":     "https://repository.gatech.edu/server/api/core/bitstreams/597db471-e2ea-4109-b1a1-b94451f1b884/content",
 "ndbc":       "https://dods.ndbc.noaa.gov/thredds/fileServer/oceansites/DATA_GRIDDED/MOVE/OS_MOVE_20000206-20221014_DPR_VOLUMETRANSPORT.nc",
 "cmems":      "https://data.marine.copernicus.eu/",
}
for k, u in URLS.items():
    try:
        r = requests.head(u, timeout=25, allow_redirects=True)
        print(f"{k:12s} {r.status_code}  len={r.headers.get('Content-Length','-')}")
    except Exception as e:
        print(f"{k:12s} ERROR {type(e).__name__}: {str(e)[:90]}")
PY
```

Expected, measured 2026-09-04 — **every Tier-0 host answered 200**:

```
psl          200  len=25213
ncei         200  len=1714749
scripps      200  len=695480508
rapid        200  len=1182284
aoml         200  len=18607
ncei_etopo   200  len=-            (this THREDDS server sends no Content-Length on HEAD; normal)
github_raw   200  len=1570533
gatech       200  len=-
ndbc         200  len=172492
cmems        200  len=329182
```

A 403, a 404 or a proxy error is worth reporting; it is not worth retrying.

---

## 4 · Running Tier 0 (Stage A)

### 4.1 The command

```bash
cd ~/handover && source beamenv/bin/activate
export EARTH_REPO=~/earth OUTPUT=/data/import STATE_DIR=/var/tmp/beam_import
bash run_until_complete.sh --tiers 0
```

That is the normal way to run it: one round, then — if anything is still
queued — a sleep of 1 h, then 2, 4, 8, and another round with the queue as the
manifest, until the queue is empty. One round on its own is:

```bash
python -m beam_import.pipeline \
    --tiers 0 --output "$OUTPUT" \
    --state-dir "$STATE_DIR" --report-dir out/tier0 \
    --runner DirectRunner --direct_running_mode multi_processing \
    --direct_num_workers 8
```

Exit codes: **0** the queue is empty; **3** the queue is not empty (run again —
this is normal and not an error); **4** a host's breaker tripped twice, stop
and report.

### 4.2 The never-drop semantics — the whole point

**A throttle changes WHEN an item is fetched, never WHETHER.** Every item ends
in exactly one of four durable states, and there is no fifth:

```
                        ┌──────────────────────────────────────┐
                        │            a work item               │
                        └──────────────────┬───────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
              .done marker           fetched, written        404 or 410
              already there          read back, sha256       from the server
                    │                 matched, marked               │
                    ▼                      │                        │
                PRESENT   ◄────────────────┘                        │
             nothing fetched            WRITTEN                     │
                                                                    ▼
                                                        first sighting? ──yes──► QUEUED
                                                                │                (ask again
                                                               no                 in ≥ 6 h)
                                                                ▼
                                                             ABSENT
                                                    (two 404s, ≥ 6 h apart,
                                                     both responses on disk;
                                                     a human reviews the list)

          anything else — a 504, a timeout, a short transfer, the circuit
          breaker stopping the lane, a missing credential, even an
          unclassified bug:                        ────────────────► QUEUED
                                                   (retry_queue.jsonl;
                                                    run again, nothing lost)
```

Two consequences worth stating plainly.

**A partly-served month or year is written, not rejected.** If OISST 1993 has
360 of its 365 days upstream, Stage A writes the 360 it got, records the five
dates in the `.done` marker's `missing_dates` and in the report, and puts those
five dates on the queue as **day-level items** (`oisst/1993/1993-03-04`). A
later run fetches each missing day into a **fill shard** beside the parent's
(`oisst/1993.fill-19930304.tfrecord`) rather than rewriting a shard that may be
gigabytes; Stage B reads every shard it finds. Only the two-sighting rule ever
turns one of those days into `absent`.

**`queued` is not a failure and should not be reported as one.** The run is
finished when the queue is empty, and `run_until_complete.sh` is what gets it
there.

### 4.3 What to watch

`report.jsonl` is written by Beam only when the pipeline finishes, so each
lane also appends every result to `<state-dir>/progress/<host>-<lane>.jsonl`
the moment the record exists — before it is even handed back to Beam. Read
those, from a second terminal:

```bash
watch -n 60 "python -m beam_import.report --live /var/tmp/beam_import"
```

Example output (from the smoke test, so the names are the fixtures'):

```
live progress from /tmp/smk/state/progress
  newest record  2026-09-04T14:48:58+00:00 UTC   (first 2026-09-04T14:48:58+00:00)
  records        14 item(s), 4 lane(s) done

  status per source
    source              queued    written
    flaky                    7          0
    ncep                     0          3
    oisst                    0          1
    …

  per host
    host             items       bytes      wall  backoffs  trips
    testflaky            7         0 B     0.00h        20      1
    testpsl              3     27.6 KB     0.00h         0      0

  queue          8 item(s)
  absent         0 item(s) (404/410 twice, >= 6 h apart)
  1 breaker trip(s) so far. Two on one host in one run means STOP and report.
```

`newest record` is the honesty check: if it stops advancing for longer than a
host's slowest expected item, something is wedged. The pipeline's own stdout is
the other live signal — one line per lane event, prefixed `[host/lane]`:
`wrote <shard> (N records, B bytes, sha …)` and `attempt N failed (…); slept Ms`.

### 4.4 Expected wall time per host

**These numbers are DERIVED, not measured.** They come from the registry's own
per-item counts and each host's `min_gap_s` and `max_lanes`, plus one
assumption per host: 50 MB/s of usable throughput per lane for HTTP, and the
**measured 411 s per GLORYS month** for CMEMS. Nobody has run Tier 0 end to
end. Use them to tell "slow because polite" from "wedged".

| host | lanes | items | requests | wire | derived total | why |
|---|---:|---:|---:|---:|---:|---|
| `cmems` (GLORYS) | 4 | 384 | 11,688 | 881 GB | **~11 h** | 384 months × 411 s ÷ 4 lanes |
| `psl` (NCEP) | 1 | 560 | 1,120 | 23.5 GB | **6.35 h** | 1 lane, 20 s apart — pace, not bandwidth |
| `scripps` (Roemmich–Gilson) | 1 | 93 | 186 | 65 GB † | **0.88 h** | |
| `ncei` (OISST) | 6 | 44 | 31,656 | 22.9 GB | **0.75 h** | ~16,000 daily files over 6 lanes |
| `aoml` (cable + SAMBA) | 1 | 44 | 88 | ~0 | **0.12 h** | |
| `ncei_etopo` | 1 | 1 | 2 | 0.5 GB | **0.01 h** | |
| `rapid`, `gatech`, `ndbc`, `github_raw` | 1–2 | 5 | 10 | ~0 | **minutes** | |

† over-estimated: the registry gives every Roemmich–Gilson item the base
files' ~0.7 GB and the monthly extensions are far smaller.

**Lanes run in parallel, so the run's wall clock is the longest lane: about
11 hours, set by GLORYS** — or about 6.5 hours, set by PSL, if you use
`glorys_from_mirror` (§4.6).

### 4.5 Re-running after an interruption

**Run exactly the same command again.** The `.done` markers are listed once at
the start and re-checked per item, so everything already written is skipped.
This is also what you do after a crash, a reboot or a lost connection. If the
queue is what you want to work through, `run_until_complete.sh` does it for
you; by hand it is:

```bash
python -m beam_import.pipeline --output "$OUTPUT" --state-dir "$STATE_DIR" \
    --report-dir out/tier0 --from-queue "$STATE_DIR/retry_queue.jsonl" \
    --runner DirectRunner --direct_running_mode multi_processing \
    --direct_num_workers 8
```

The old queue file is rotated to `retry_queue.1.jsonl` (then `.2`, `.3` …) and
a fresh one is written; nothing is deleted, so the rotated files are the record
of how many rounds it took.

### 4.6 The 840 GB shortcut

GLORYS is 90% of Tier 0's traffic, and a copy of exactly those monthly chunks —
already binned to 0.25° by the same imported rule — exists as NetCDF and can be
read anonymously. Converting that copy costs about **96 GB** and about an hour
instead of 840 GB and eleven hours:

```bash
# instead of the `glorys` source
python -m beam_import.pipeline --tiers 0 --only glorys_from_mirror \
    --output "$OUTPUT" --state-dir "$STATE_DIR" --report-dir out/mirror \
    --runner DirectRunner --direct_running_mode multi_processing \
    --direct_num_workers 4
```

It is `enabled: false` on purpose, because the default has to be the path that
does not depend on somebody else's storage still being there. The files are
**not re-binned** — binning them again would be a second decision about the
same bytes. To read the same chunks from a different filesystem, edit the
`url:` in the `glorys_from_mirror` block of `sources.yaml`. If you use this
source, do **not** also run `glorys`, and say in your report which one you ran.

### 4.7 What a breaker trip looks like

```
[psl/0] attempt 1 failed (HEAD …: HTTP 504); slept 61s
[psl/0] attempt 2 failed (HEAD …: HTTP 504); slept 297s
…
[psl/0] CIRCUIT BREAKER TRIPPED — the rest of this lane is queued for the next run.
```

Five items in a row failed, so the lane stopped on purpose and everything it
had not reached went to the queue. `run_until_complete.sh` sleeps and tries
again. **If a host trips twice in one run the script stops and exits 4** — do
not re-run it; write the report in §10.

---

## 5 · Verifying Stage A

```bash
python -m beam_import.verify_output --tiers 0 \
    --output "$OUTPUT" --state-dir "$STATE_DIR" --deep \
    --json-out out/tier0/verify.json
```

Expected when the tier is complete:

```
tier(s) [0]: 1131 item(s) expected, 1131 written (199.90 GB in 1131 shard(s))
missing: 0
short:   0
deep:    0 shard(s) failed a full re-read
extra:   0 marker(s) with no manifest item
queued:  0 item(s) still to do
absent:  0 item(s) — REVIEW THESE BEFORE STAGE B
```

`--deep` re-reads every shard and checks every record's CRC; it is slower and
it is the right thing to do once, before a long training run. The command
exits 0 only when nothing is missing that is not explained by absent evidence.

Then read the summary:

```bash
cat out/tier0/summary.md
```

It has counts by status, the **per-host politeness audit** (requests, bytes,
wall time, backoffs, breaker trips), a per-source table, a table of the days
that were missing upstream and re-queued, and the queue and absent counts. The
politeness audit is what Chris reads first: **a host with many backoffs or any
trips means its budget is too generous and should come DOWN before the next
run** — never up.

---

## 6 · Stage B — assembling the training set

Stage A wrote one record per source-day. Stage B turns those into one record
per **(pentad bin, channel group)** — the family-7 layout — and that is the
thing a trainer reads.

```bash
python -m beam_import.assemble \
    --output "$OUTPUT" \
    --pentad-out "$OUTPUT/pentad" \
    --num-shards 64 \
    --runner DirectRunner --direct_running_mode multi_processing \
    --direct_num_workers 8
```

Expected output — the coverage line is the thing to read:

```
stage B: 1131 shard(s) -> /data/import/pentad (64 shards per group)
coverage g025: bins_present=3139 range=[0, 3138] missing_in_range=0
coverage g100: bins_present=3139 range=[0, 3138] missing_in_range=0
coverage rg100: bins_present=252 range=[1606, 3138] missing_in_range=1281
```

(On the smoke test's one-bin fixture the same three lines read
`bins_present=1 range=[1573, 1573] missing_in_range=0`.)

`missing_in_range` must be **0** for `g025` and `g100` — a missing bin has to
trace back to an `absent` record, and if it does not, something was dropped and
that is a stop-and-report. For `rg100` a large `missing_in_range` is **correct
and expected**: Roemmich–Gilson is monthly, so it is written only on the bin
holding each month's 15th and is deliberately absent on the other four bins of
the month.

Beside the shards you get two files:

- **`spec.json`** — the groups, their channel names in order, the shape, and
  the per-channel mean and standard deviation over what was written. Those
  numbers are kept OUT of the records on purpose: which years count as "train"
  is a trainer's decision, not the import's.
- **`coverage.json`** — the machine-readable form of the lines above.

### 6.1 Reading the shards — pure Python, no TensorFlow

```python
import numpy as np
from beam_import import tfrecord
from beam_import.example import one_int, one_str, parse_example, str_list

ROOT = "/data/import/pentad/g025"
for uri in tfrecord.list_uris(ROOT, ".tfrecord"):
    for payload in tfrecord.read_records(uri):     # CRCs checked in here
        rec = parse_example(payload)
        shape = [int(x) for x in rec["shape"]]                  # [C, H, W]
        values = np.frombuffer(rec["values"][0], dtype="<f4").reshape(shape)
        names = str_list(rec, "chan_names")
        print("bin", one_int(rec, "bin"), one_str(rec, "date_start"),
              "shape", shape, "cur_u mean",
              float(np.nanmean(values[names.index("cur_u")])))
```

Output on the smoke fixture:

```
bin 1573 2003-07-15 shape [7, 9, 9] cur_u mean 0.009829164482653141
```

### 6.2 Reading the shards — `tf.data`

```python
import numpy as np, tensorflow as tf

FEATURES = {
    "bin":          tf.io.FixedLenFeature([], tf.int64),
    "date_start":   tf.io.FixedLenFeature([], tf.string),
    "chan_names":   tf.io.VarLenFeature(tf.string),
    "shape":        tf.io.FixedLenFeature([3], tf.int64),
    "values":       tf.io.FixedLenFeature([], tf.string),
    "days_present": tf.io.VarLenFeature(tf.int64),
}
files = tf.io.gfile.glob("/data/import/pentad/g025/*.tfrecord")
for raw in tf.data.TFRecordDataset(files):
    ex = tf.io.parse_single_example(raw, FEATURES)
    shape = ex["shape"].numpy()
    values = tf.reshape(tf.io.decode_raw(ex["values"], tf.float32),
                        shape).numpy()
    names = [b.decode() for b in tf.sparse.to_dense(ex["chan_names"]).numpy()]
    print("bin", int(ex["bin"]), ex["date_start"].numpy().decode(),
          "shape", list(shape), "cur_u mean",
          float(np.nanmean(values[names.index("cur_u")])))
```

Output on the same shard:

```
bin 1573 2003-07-15 shape [7, 9, 9] cur_u mean 0.009829164482653141
```

**Both snippets above were run and produce that identical number.** The
`mask` feature is a bit-packed `[C, H, W]` array of "is this finite" —
`np.unpackbits(np.frombuffer(rec["mask"][0], dtype=np.uint8))[:C*H*W]
.reshape(shape)` — and `days_present` says, per channel, how many of the five
days contributed. The `min_days = 3` rule has already been applied: a channel
below it is all-NaN with its true `days_present`, so you can tell "no data"
from "not enough data".

---

## 7 · Tier 1

Tier 1 is everything the next model family needs that requires no new account.
Most of it is small. One source is not.

**Run DUACS on its own.** It is ~1.1 TB on the wire — 384 monthly items, each
fetched one day at a time and binned to 0.25° — and at four CMEMS lanes it
takes **two to three days**:

```bash
bash run_until_complete.sh --tiers 1 --only duacs
```

It may be interrupted and resumed at will. **Before the first DUACS run**,
resolve the real dataset identifier — the one in `sources.yaml` is marked
`unverified_url` because it is a best guess:

```bash
copernicusmarine describe --product-id SEALEVEL_GLO_PHY_L4_MY_008_047
```

Find the daily 0.125° delayed-time (`_my_`) dataset, paste its `dataset_id`
into the `duacs:` block, and delete the `unverified_url: true` and
`resolve_dataset_id: true` lines from that block. Do not hardcode a guess.
Delayed-time DUACS is produced with a centred ±6-week window, so the last ~6
weeks of the record do not exist in this product.

Everything else in Tier 1:

```bash
bash run_until_complete.sh --tiers 1
```

**Six Tier-1 sources carry `unverified_url: true` and will probably not work
until a human checks them.** That is expected and it is why they are flagged:
`en4` (a HEAD on the 2020 file returned **404** on 2026-09-04 — the path has
moved; open <https://www.metoffice.gov.uk/hadobs/en4/download-en4-2-2.html>),
`pmel_wwv` (the configured URL is an index page, not a file), `godas` (a
directory — it needs rewriting as a `var_year` source), `oni` (this sandbox's
proxy refused the host, so it is untested), `mei`, and `rmm` (a HEAD returned
**403** on 2026-09-04). Report what each one does; do not invent replacement
URLs. Note what they will do meanwhile: **`queued`, not lost.**

---

## 8 · Tier 2

### ERA5, once Chris provides the CDS key

Today every ERA5 item is reported `queued` with the reason "no CDS
credentials", and **nothing is requested**. When the key exists:

```bash
export CDSAPI_URL='https://cds.climate.copernicus.eu/api'
export CDSAPI_KEY='<the key Chris gives you>'
pip install cdsapi

# start with ONE month of ONE variable and look at the file before
# committing to 7,224 requests
python -m beam_import.pipeline --tiers 2 --only era5 --dry-run \
    --output "$OUTPUT" --state-dir "$STATE_DIR" --report-dir out/era5dry \
    --runner DirectRunner
```

14 variables × 516 months = **7,224 requests**, two in flight at a time. That
is a lot of queueing and it should be discussed with Chris before it starts.
Check the variable names against the dataset's own form first — `sources.yaml`
lists ERA5 **short** names (`metss`, `u10`, `t2m` …) and some CDS datasets want
the long ones.

### ESA CCI soil moisture

`enabled: false`, and its path pattern is a guess, because the real directory
listing can only be read after registering free at
<https://services.ceda.ac.uk>. Register, browse to the v09.1 COMBINED daily
0.25° product, put the real URL pattern into the `cci_sm:` block, remove
`unverified_url: true`, set `enabled: true`, and run `--tiers 2 --only cci_sm`.

### Why OSTIA and GREP-3D are off

**OSTIA** is a 0.05° sea-surface-temperature and sea-ice product: roughly
**3 TB on the wire**, for a sea-ice fraction OISST already gives us at a
fraction of the cost. **GREP-3D** is the depth-resolved ocean ensemble
reanalysis: about **200 GB written** globally, and E-070 §6 says the streaming
loader that would use it has to be built first. Both are off by default and
should stay off unless somebody has a reason. `sources.yaml` carries a
North-Atlantic window for GREP (`bbox_na_option`) to swap into `bbox`
deliberately, and its depth levels must be **read from the first file the
server sends**, never hardcoded.

---

## 9 · Beam → Flume, Dataflow, and the zero-cost fallback

### The mapping, if you are running inside Google

The pipeline is written in Apache Beam, which is the open-source form of
Flume. The concepts line up one to one, and **the part that must survive the
move is the key-count guarantee**, not the API:

| Beam (this code) | Flume | what it must keep doing |
|---|---|---|
| `beam.GroupByKey()` on `(host, lane)` | `MakeUnique` / `GroupByKey` on the same key | **THE POLITENESS GUARANTEE.** All the values of one key are iterated by ONE worker, in order. That is why "number of keys per host" *is* "maximum simultaneous connections to that host". Nothing else in the design bounds load. |
| `beam.DoFn` with `setup()` | `DoFn` with `startBundle` / a lazily-built member | Clients and credentials are created per worker, inside the DoFn — never captured in the constructor, never a pipeline option. |
| `beam.pvalue.AsList(...)` side input | a side input / broadcast | The `.done` listing, taken once at the start. It is an optimisation; the per-item marker check is the correctness half and must stay. |
| `beam.io.WriteToTFRecord` | the TFRecord sink | Same file format, same records. `beam_import/tfrecord.py` is a pure-Python fallback that writes the identical bytes, if a sink is easier to avoid than to configure. |
| `apache_beam.io.filesystems.FileSystems` | the filesystem abstraction | Everything the package writes goes through it, so the output URI is the only thing that changes. |
| `--direct_num_workers` | the worker count | **Set `workers × threads ≥ 29` and no higher** — 29 is the sum of `max_lanes` over all hosts. More workers cannot make it faster (the lanes are the limit) and only risk someone raising a budget to "use" them. |
| environment variables in `setup()` | the runner's secret mechanism | Credentials never travel as options or flags, on any runner. |

**Stage B is the only part a distributed runner helps with.** It is a wide
reduce over the whole Stage-A output and it needs no politeness at all, only
bandwidth to the output store. Stage A's speed is set by `min_gap_s`, and no
amount of hardware changes that.

### Dataflow

```bash
python -m beam_import.assemble --output gs://bucket/import \
    --pentad-out gs://bucket/import/pentad \
    --runner DataflowRunner --project <gcp-project> --region <region> \
    --temp_location gs://bucket/tmp \
    --max_num_workers 8 --number_of_worker_harness_threads 4
```

**The warning that matters: pipeline options are logged and displayed in the
Dataflow job UI, so a credential passed as an option is a published
credential.** On Dataflow, secrets come from Secret Manager, read inside
`DoFn.setup()`. For Stage A also keep `max_num_workers × threads ≤ 29`, and
remember that Dataflow retries a failed bundle four times — which is exactly
why nothing transient is allowed to escape a DoFn.

### The GitHub-Actions fallback

Run the same command inside a GitHub-hosted runner, on `workflow_dispatch`.
Because the pipeline is resumable from the `.done` markers and the retry queue,
the 350-minute job cap just means more firings — each picks up where the last
left off. It costs nothing. The output has to be a bucket the runner can write
to; no credential goes on a command line.

---

## 10 · What to report back to Chris

Send **one markdown message**. Use this template, filled in with real numbers
read out of `summary.md`, `verify.json` and `coverage.json` — never from
memory, and never a number you did not read out of an artefact.

```markdown
## Import status — Tier <N> — <date, UTC>

**TL;DR:** <one plain sentence: is the tier done, and if not, what is left.>

### Counts per tier
| tier | items | written | present | queued | absent |
|---|---:|---:|---:|---:|---:|
| 0 | 1131 | | | | |

### Wire GB fetched vs the manifest's estimate
Fetched <X> GB against the manifest's estimate of <Y> GB (<Z>%).
<One sentence if they differ by more than ~20%, naming the source.>
<Say whether `glorys` or `glorys_from_mirror` was used — it is a 9× difference.>

### Bytes and wall time per host
| host | lanes | items | requests | bytes | wall | backoffs | breaker trips |
|---|---:|---:|---:|---:|---:|---:|---:|
| cmems | 4 | | | | | | |
| psl | 1 | | | | | | |
| ... | | | | | | | |

### Output
`--output` was <uri>. <n> shards, <n> `.done` markers, <X> GB.
`verify_output --tiers <N> --deep` reports missing <n>, short <n>, deep_bad <n>.

### Stage B coverage
coverage g025: bins_present=<n> range=[<a>, <b>] missing_in_range=<n>
coverage g100: bins_present=<n> range=[<a>, <b>] missing_in_range=<n>
coverage rg100: bins_present=<n> range=[<a>, <b>] missing_in_range=<n>
<Every missing g025/g100 bin traced to an `absent` record — or say it did not.>

### Still queued, and absent
| item | status | reason / error text (verbatim) |
|---|---|---|
| | | |
<For each `absent`: both sightings' timestamps and status codes.>

### What I could not verify
- <e.g. "the `en4` URL pattern still 404s; I did not look for a replacement">

### What I would do next
- <one or two sentences>
```

Rules for the report: **paste error text verbatim**, do not paraphrase.
**Never write a bare identifier** — every source name gets a short
plain-English gloss the first time it appears. **Every number comes from a
file**, and say which. If a breaker tripped twice and you stopped, that goes in
the TL;DR, not a footnote. And **`queued` is not "failed"** — report it as
"still to do", with what the next round will try.

Where the numbers come from: the per-tier and per-host tables are the two
tables in `summary.md`; the wire GB is the sum of that file's per-host `bytes`
column; the manifest's estimate is the `to fetch:` line from
`python -m beam_import.manifest --tiers <N>` (925.1 GB for Tier 0 with
`glorys`, ~185 GB with `glorys_from_mirror`); the coverage lines are printed by
Stage B and stored in `coverage.json`.

---

## 11 · Troubleshooting

| symptom | cause | what to do |
|---|---|---|
| `HTTP 504` from `downloads.psl.noaa.gov`, then silence for ~15 min | PSL throttles after two large files. This is the measured behaviour the whole backoff ladder exists for. | Nothing. The lane sleeps 60 s, 5 min, 15 min, 60 min. If the breaker trips, `run_until_complete.sh` waits an hour and tries again. **Do not raise `max_lanes` for `psl`.** |
| the run exits **3** | the queue is not empty | Not an error. Run again, or let `run_until_complete.sh` do it. Nothing was lost. |
| the run exits **4** | a host's breaker tripped twice in one run | **Stop.** Hard rule 7. Read `summary.md`, write the §10 report. Do not re-run. |
| `NetCDF: HDF error` when something opens a file | the download was **truncated**; a short transfer raises nothing at the socket | The HEAD `Content-Length` vs bytes-on-disk check should have caught it. If it did not, the server sent no `Content-Length` — report the URL. Delete the item's directory under `<state-dir>/work` and run again. |
| `payload CRC mismatch` / `truncated payload` reading a shard | the shard on disk is corrupt | The write path checks this before the marker is written, so a corrupt shard means storage trouble after the fact. Delete the shard AND its `.done` marker, then re-run: it will be rewritten. |
| `came back with sha256 …` during a write | the bytes did not survive the round trip | The shard is deleted and the item is queued automatically. If it repeats for the same item, report it — that is a storage problem, not a network one. |
| an item is `queued` forever | it genuinely cannot be fetched, or its URL is wrong | Read the `reason` in `report.jsonl`. If it is a 404, the two-sighting rule will make it `absent` on the next run six hours later. If the URL is marked `unverified_url`, a human has to fix the registry. |
| an item is `absent` | the archive said 404/410 twice, ≥ 6 h apart | Nothing automatic. Read `<state-dir>/absent_evidence/<item>.json` — both responses are there — and put it in the report. A missing Stage-B bin must trace to one of these. |
| `copernicusmarine` authentication error | almost always the `%` in the password being eaten by the shell | Re-export it in **single quotes**, then check with the set/NOT-set loop in §2.3. |
| everything in a tier is `queued` with "no credentials" | a gated source: ERA5 without `CDSAPI_KEY`, CMEMS without the two `COPERNICUSMARINE_*` variables | Read the reason; it names the variables. For ERA5 this is correct until Chris makes the account. |
| the process is killed; `dmesg` says `Out of memory` | an item is too big for one machine — a whole GLORYS month in one request is 5.95 GB | Fix the chunking (the registry fetches one DAY per request), **not** the worker count. Adding workers makes it worse. Report it before changing `sources.yaml`. |
| `PicklingError` / `Can't pickle …` at start-up | Beam pickles the DoFn and everything it holds | Clients belong in `DoFn.setup()`, configuration in the plain dict passed to `__init__`. Do not put an open file, a session or a client in a constructor. |
| the DirectRunner hangs at start or spawns endless processes | a missing `if __name__ == '__main__':` guard — in `multi_processing` mode every child re-imports the module | Both entry points have the guard. A new one needs it too. |
| `No space left on device` | raw downloads that were not cleaned up, from a run killed between fetch and write | `du -sh <state-dir>/work/*`, then delete those subdirectories. Raw files are removed only after the shard is written and verified, so anything left over is safe to delete: it will be fetched again. |
| `no earth checkout at …` | `EARTH_REPO` unset or pointing somewhere without `ml/` | `git clone https://github.com/blauewelt/earth.git` and export it. **Do not** write your own binning or pentad rule (hard rule 6). |
| Stage B says `missing_in_range` > 0 for `g025` or `g100` | some bin has no source day | Trace each missing bin to an `absent` record. If it does not trace, something was dropped — **stop and report**, do not proceed to training. |
| Stage B says `missing_in_range` is large for `rg100` | Roemmich–Gilson is monthly | Correct and expected: it is written only on the bin holding each month's 15th. |
