# How to run the import — the step-by-step

**Read this file top to bottom before you type anything.** Every command in it
is meant to be copied exactly. Where a command has an expected output, the
expected output is written underneath it; if what you see differs in a way this
document does not explain, **stop and write down what you saw** rather than
improvising a fix.

Companion files in this directory:

- [`DESIGN.md`](DESIGN.md) — why the thing is built this way. Read it once.
- [`sources.yaml`](sources.yaml) — the registry: every dataset, every host budget.
- [`CREDENTIALS.md`](CREDENTIALS.md) — the passwords and the rules about them.
- `beam_import/` — the code. `tests/` — the tests.

Background in the main repository (all three render on a phone):

- [E-070 family-7 build plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md)
- [E-070 global tensor plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_global_tensor.md)
- [The data ladder](https://blauewelt.github.io/earth/docs.html?f=ml/plans/DATA_LADDER.md)

---

## 0 · What you are doing, and what done looks like

You are going to **copy a set of public climate datasets onto one storage
location, carefully, without overloading any of the servers you copy them
from.** That is the whole job. You are not training anything, not building
anything, and not analysing anything. You download files, you check each one
arrived intact, you upload it to a shared archive, you check it comes back out
of that archive unchanged, and you write down what happened.

The datasets are the inputs a machine-learning model of the ocean is built
from, and they live on a dozen different public servers run by national weather
and ocean agencies. Some of them are enormous. Today every rebuild of the model
downloads them all again from those agencies, which is wasteful and rude, so
the plan is: **import once, into a mirror; every future build reads the
mirror.** The mirror is a repository on the **Hugging Face Hub** ("the Hub" —
a public website that hosts large datasets and model files, like GitHub but
for data), specifically the dataset repository `chfrank/earth-tensors`. The
tool that runs the copying is **Apache Beam** (a Python framework for data
pipelines); **DirectRunner** is Beam's mode for running on one ordinary machine
and is the mode you will use, and **Dataflow** is Google's paid cloud mode
which you will almost certainly not need.

The names you will see, each spelled out once. **GLORYS12** — a global model
reconstruction of ocean currents, sea-surface height and mixed-layer depth from
Copernicus, at 1/12 of a degree; already on the Hub, so you only check it is
there. **OISST** — the Optimum Interpolation Sea Surface Temperature record
from NOAA, daily sea-surface temperature and sea-ice concentration since
September 1981. **NCEP R1** — the NCEP/NCAR Reanalysis 1, a decades-long
reconstruction of surface winds, heat fluxes and soil conditions; its archive
was frozen in March 2026, so this copy is the last chance to take one.
**Roemmich–Gilson** — a climatology of ocean temperature and salinity with
depth, built from the Argo float network at the Scripps Institution.
**DUACS** — the processing system that turns satellite altimeter passes into
daily maps of sea-surface height; the Tier-1 heavyweight, about 1.1 terabytes
on the wire. **ERA5** — the European Centre's global atmospheric reanalysis,
reached through the **CDS** (Climate Data Store, Copernicus's download service
for atmosphere data), which needs a free account that does not exist yet.
**CMEMS** — Copernicus Marine Environment Monitoring Service, the download
service for ocean data, for which we *do* have credentials. **Family 7** — the
name of the current input data cube the model trains on, the first one covering
the whole globe rather than just the North Atlantic. A **pentad** is a five-day
period, the time step that cube uses.

**Done looks like this.** `python -m beam_import.verify_hub --tiers 0` prints
`missing: 0`. A file `out/tier0/summary.md` exists and shows, per server, how
many bytes were fetched, how long it took, how many times we backed off after
an error, and how many times a lane's circuit breaker tripped. A file
`sources/MANIFEST_tier0.json` exists on the Hub listing every mirrored file
with its checksum. And you have sent Chris the status report from §9. If some
items come back `deferred` (a server got tired and we stopped talking to it),
that is not failure — re-run the same command later and they get picked up.
Tier 1 is the same procedure with `--tiers 1`, and Tier 2 is blocked until
Chris creates a CDS account.

---

## 1 · Hard rules

These are not style preferences. Each one exists because breaking it has cost
this project real time or real money.

1. **Never put a credential on a command line, in a Beam pipeline option, or
   in anything you print.** Pipeline options are logged and shown in job UIs.
   The code reads credentials from environment variables inside
   `DoFn.setup()`; keep it that way.
2. **Never raise a host budget.** `max_lanes` and `min_gap_s` in
   `sources.yaml` may be lowered, never raised, and never by you on your own
   initiative. If a run is slow, it is supposed to be slow.
3. **Never run this on a rented or borrowed machine (a Vast box, a shared CI
   runner, anyone's GPU box) while CMEMS or CDS credentials are in the
   environment.** The host has root and can read everything.
4. **Never re-implement the binning rule.** The 0.25-degree binning is
   `bin_plan` / `bin_slice`, imported from `ml/aggregate_cadence.py` in the
   earth checkout. If it will not import, fix the checkout — do not write your
   own version, not even a small one, not even temporarily.
5. **Never delete anything on the Hub.** Nothing in this package deletes; do
   not add anything that does.
6. **Never schedule this.** Every run is a manual dispatch. No cron, no
   scheduled task, no "I will kick it off again automatically in an hour".
7. **If a host's circuit breaker trips twice in one run, stop the run and
   report.** Once is normal and self-healing. Twice means the budget in
   `sources.yaml` is too generous and a human has to decide, not you.
8. **Never edit `beam_import/*.py` to make a failing item pass.** Report the
   failure with its exact error text instead.
9. **Run `bash run_smoke.sh` before every real run.** It is offline and takes
   under a minute.
10. **Do not retry a host the network refused you.** If a request is blocked
    by a proxy or a firewall, record it as blocked and move on. Retrying is
    how a block becomes a ban.
11. **Anything you did not verify, say you did not verify.** "The URL pattern
    is marked `unverified_url` and I did not test it" is a complete and useful
    sentence.
12. **If you are unsure, stop and write down what you observed.** A stopped
    run costs an hour. A wrong run costs a week and looks like it worked.

---

## 2 · The machine and the environment

**The machine.** One ordinary Linux virtual machine: 4–8 virtual CPUs, 16 GB
of RAM, 100 GB of free disk. Not a GPU box (rule 3). Tier 1's DUACS run wants
the disk more than the CPU. The whole pipeline's speed is set by how politely
it talks to the servers, not by the machine, so a bigger machine buys nothing.

### 2.1 Get the code

```bash
cd ~
git clone https://github.com/blauewelt/earth.git
export EARTH_REPO=~/earth
# the handover package (this directory) — copy it to ~/handover if it is not
# already there, then:
cd ~/handover
```

`EARTH_REPO` is how `beam_import/transforms.py` finds the binning rule. If it
is unset, the code looks for `../earth` next to this directory.

### 2.2 Build the virtual environment

```bash
bash setup_env.sh
source beamenv/bin/activate
```

**Why the script and not a bare `pip install`.** `setup_env.sh` installs
`setuptools<70` **first, on its own**, before anything else. Three of Apache
Beam's dependencies — `crcmod`, `dill` 0.3.1.1 and `hdfs` — are still shipped
as source distributions that call setuptools APIs which were removed in
setuptools 70. With a current setuptools their builds fail part-way through the
install and leave you with a virtual environment where `import apache_beam`
works and running a pipeline does not. This was measured, not guessed. If you
build the environment by hand, pin setuptools first.

Expected tail of the output:

```
apache-beam        2.68.0
huggingface_hub    1.30.0
netCDF4            1.7.4
numpy              2.2.6
xarray             2026.7.0
copernicusmarine   2.4.1
cdsapi             present (Tier 2 only)
```

### 2.3 Put the credentials in the environment

Open [`CREDENTIALS.md`](CREDENTIALS.md) and copy the export block from it.
**The Copernicus Marine password contains a `%` and must be in single
quotes** — that is the single most common way this setup goes wrong:

```bash
export COPERNICUSMARINE_SERVICE_USERNAME='cfrank1'
export COPERNICUSMARINE_SERVICE_PASSWORD='<password from CREDENTIALS.md — has a %>'   # single quotes!
export CDSAPI_URL='https://cds.climate.copernicus.eu/api'
```

The Hugging Face token may live in the environment **or** in one file, which is
the only credential allowed on disk:

```bash
printf '%s' '<token from CREDENTIALS.md>' > ~/.hf_token
chmod 600 ~/.hf_token
```

Check without printing any of them:

```bash
for v in COPERNICUSMARINE_SERVICE_USERNAME COPERNICUSMARINE_SERVICE_PASSWORD \
         HF_TOKEN CDSAPI_URL CDSAPI_KEY; do
  printf '%-38s %s\n' "$v" "$([ -n "${!v:-}" ] && echo set || echo 'NOT set')"
done
```

Expected: the first two `set`, `CDSAPI_URL` `set`, `CDSAPI_KEY` **`NOT set`**
(that is correct — ERA5 is blocked on purpose), and `HF_TOKEN` either `set` or
`NOT set` with `~/.hf_token` present instead.

---

## 3 · Preflight

Do all five checks. They take about two minutes together and they are the
difference between finding a problem now and finding it three hours in.

### (a) The registry parses and every source names a host

```bash
python -m beam_import.registry --check
```

Expected — some `warning:` lines about unverified URLs and disabled sources
(those are correct and expected), then:

```
registry  /home/…/handover/sources.yaml
hub       chfrank/earth-tensors (60 commits/h)
hosts     16   total lanes 27
sources   26
  tier 0: 12  glorys oisst oisst_psl ncep rg rapid florida_cable osnap move samba etopo natural_earth
  tier 1: 10  duacs cmems_static rapid_extra en4 pmel_wwv olr godas oni mei rmm
  tier 2: 4  era5 cci_sm ostia grep3d
OK
```

If it prints `REGISTRY INVALID`, read the list of reasons; every one of them
names a source or a host and what is missing. Do not run anything else until
this says `OK`.

### (b) The manifest expands and the counts are right

```bash
python -m beam_import.manifest --tiers 0 --print | tail -30
```

The last lines are a counts table. **The expected Tier-0 table, measured
2026-09-04:**

| source | items | host | what one item is |
|---|---:|---|---|
| `glorys` | 384 | cmems | one month, 1993-01 … 2024-12 — **verify only** |
| `oisst` | 44 | ncei | one year, 1981 … 2024, folded from ~365 daily files |
| `ncep` | 560 | psl | one (variable, year) file: 13 variables × 43 years + the land mask |
| `rg` | 93 | scripps | 2 base files + one file per monthly extension (**scraped** — see below) |
| `rapid` | 1 | rapid | one file |
| `florida_cable` | 43 | aoml | one year |
| `osnap` | 1 | gatech | one file |
| `move` | 1 | ndbc | one file |
| `samba` | 1 | aoml | one file |
| `etopo` | 1 | ncei_etopo | one file (~450 MB) |
| `natural_earth` | 2 | github_raw | two GeoJSON files |
| **TOTAL** | **1131** | | 384 verify-only + **747 to fetch** |

and the last three lines of the output are:

```
  already on Hub (verify only): 384 items, 96.4 GB
  to fetch: 747 items · 104.3 GB wire · 103.5 GB stored
  TOTAL 1131 items · 0 with an unverified URL
```

**Read the "to fetch" line, not the total.** The 384 GLORYS months are already
on the Hub and are only checked for presence — nothing is downloaded for them.
Counting the 2.2 GB per month they *would* have cost would put Tier 0 at
~925 GB of wire, which is four times what this run actually transfers. The
real figures are **747 items, 104.3 GB on the wire, 103.5 GB stored** — which
is DESIGN §3's "about 110 GB on the wire, about the same stored".

For all three tiers the same two lines read `already on Hub (verify only):
384 items, 96.4 GB` and `to fetch: 8483 items · 1.2 TB wire · 273.2 GB
stored`.

Two of these numbers move legitimately and neither is an error:

- **`rg` is 93 only if the scrape worked.** The list of Roemmich–Gilson monthly
  extension files is read off
  `https://sio-argo.ucsd.edu/RG_Climatology.html`. On 2026-09-04 that gave 91
  extensions plus the 2 base files = 93. If the page cannot be read the code
  prints a warning and falls back to the declared range 2019-01 … 2025-12,
  giving **86**, and months the server does not actually have come back
  `absent`, which is fine. Add `--offline` to force the fallback.
- **`ncep` is 560, and `DESIGN.md` §3 says 603 with 14 variables.** The
  registry is the truth: the repository's own variable table
  (`ml/build_family7.py`) has **thirteen** variables, so 13 × 43 + 1 = 560.
  Report the number the manifest prints, do not "fix" it to match the design
  document.

For all three tiers:

```bash
python -m beam_import.manifest --tiers 0,1,2 --offline | tail -40
```

Expected totals: Tier 1 adds **519** items (of which `duacs` is 384 and `en4`
is 126) and Tier 2 adds **7,224** (ERA5: 14 variables × 516 months). With
`--offline` — so `rg` falls back to 86 and Tier 0 is 1,124 — the grand total
is **8,867 items**, of which **8,483 are to fetch (1.2 TB wire, 273.2 GB
stored)** and **518 carry `unverified_url: true`**. The unverified ones are
listed on purpose, so a human can see what *would* be attempted; they are not
ready to run.

### (c) The offline smoke test

```bash
bash run_smoke.sh
```

It generates its own fixtures, uses `file://` URLs instead of the internet and
a local directory instead of the Hub. It must end with:

```
  ok   three sources published (file:// http, OISST year-fold, NCEP-like)
  ok   every published item carries the sha256 the restore-verify compared
  ok   the flaky source failed exactly 5 items (the breaker threshold)
  ok   the rest of that lane is `deferred`, not `failed`
  ok   exactly one circuit-breaker trip was recorded
  ok   the folded OISST year is on the fake Hub
  ok   the Hub round-trip preflight ran
  ok   summary.md was written

SMOKE TEST: PASS
```

Anything other than `SMOKE TEST: PASS` means **do not start a real run.** The
unit tests are the finer-grained version of the same thing:

```bash
python -m pytest tests -q
```

Expected: `42 passed`.

### (d) The Hub round trip, and the dry run

The pipeline does the Hub round trip itself at the start of every real run:
it uploads a 37-byte file to `sources/_preflight/roundtrip.txt`, downloads it
back, compares the sha256, and deletes the local copy. You will see

```
preflight: Hub round trip OK (sha256 0178c24b0dfc)
```

as the first line of a real run. If instead it raises, the credentials or the
namespace are wrong — go to §10.

To see the plan without touching a single upstream server:

```bash
python -m beam_import.pipeline --tiers 0 --dry-run \
    --report-dir out/dryrun --state-dir /var/tmp/beam_import \
    --runner DirectRunner
head -3 out/dryrun/report.jsonl
```

Every record comes back with `"status": "planned"` and the URL that would have
been fetched. A dry run makes no Hub calls at all, so it works with no
credentials.

### (e) One HEAD request per host

This is the only network check you make by hand, and you make **exactly one
request per host**. Do not loop, do not retry a host that refuses.

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

Expected, measured 2026-09-04 from this sandbox — **every one of these Tier-0
hosts answered 200**:

```
psl          200  len=25213
ncei         200  len=1714749
scripps      200  len=695480508
rapid        200  len=1182284
aoml         200  len=18607
ncei_etopo   200  len=-            (this THREDDS server sends no Content-Length on HEAD; that is normal)
github_raw   200  len=1570533
gatech       200  len=-
ndbc         200  len=172492
cmems        200  len=329182
```

A `403`, a `404` or a proxy error on one of these is worth reporting; it is
not worth retrying.

---

## 4 · Running Tier 0

```bash
cd ~/handover
source beamenv/bin/activate
export EARTH_REPO=~/earth
bash run_tier0.sh
```

or, if you want the pipeline directly:

```bash
python -m beam_import.pipeline \
    --tiers 0 \
    --report-dir out/tier0 \
    --state-dir /var/tmp/beam_import \
    --runner DirectRunner \
    --direct_running_mode multi_processing \
    --direct_num_workers 8
```

### What to watch

**`report.jsonl` is written by Beam only when the whole pipeline finishes.**
So each lane also appends every result record to its own file under
`<state-dir>/progress/<host>-<lane>.jsonl` the moment that record exists —
before it is even handed back to Beam. Read those, in a second terminal:

```bash
python -m beam_import.report --live /var/tmp/beam_import
```

or, to keep it on screen:

```bash
watch -n 60 "python -m beam_import.report --live /var/tmp/beam_import"
```

Example output (this is the smoke test's, so the names are the fixtures'; a
real run has `ncep`, `oisst`, `rg` and the rest):

```
live progress from /tmp/smk/state/progress
  newest record  2026-09-04T14:48:58+00:00 UTC   (first 2026-09-04T14:48:58+00:00)
  records        10 item(s), 4 lane(s) done

  status per source
    source             deferred     failed  published
    flaky                     2          5          0
    ncep                      0          0          1
    oisst                     0          0          1
    tiny                      0          0          1
    TOTAL                     2          5          3

  per host
    host             items       bytes      wall  backoffs  trips
    testflaky            7         0 B     0.00h        20      1
    testhost             1        38 B     0.00h         0      0
    testncei             1     90.7 KB     0.00h         0      0
    testpsl              1     27.6 KB     0.00h         0      0

  1 breaker trip(s) so far. Two on one host in one run means STOP and report.
```

`newest record` is the honesty check: if it stops advancing for longer than a
host's slowest expected item, something is wedged. The `backoffs` and `trips`
columns are live — you do **not** have to wait for a lane to finish to see
that it is in trouble.

The pipeline's own stdout is the other live signal: one line per lane event,
prefixed `[host/lane]` — `verified <hub path> (<sha prefix>)` for each file
that passed the upload-and-download-back check, and `attempt N failed (…);
slept Ms` for each backoff.

When the run ends, `summary.md` is built from the **union** of the progress
files and `report.jsonl`, deduplicated by item with the final record winning —
so a run that was killed still summarises correctly from the progress files
alone:

```bash
python -m beam_import.report --report out/tier0/report.jsonl \
    --state-dir /var/tmp/beam_import --out out/tier0/summary.md
```

### Expected wall time per host

**These numbers are DERIVED, not measured.** They come from the registry's own
per-item byte counts and each host's `min_gap_s` and `max_lanes`, plus one
assumption: 50 MB/s of usable throughput per lane. Nobody has yet run Tier 0
end to end. Use them to tell "slow because polite" from "wedged", not as a
promise.

The derivation is: *pace* = requests × `min_gap_s` ÷ `max_lanes` (two requests,
a HEAD and a GET, per file — and OISST is ~365 files per item), *transfer* =
wire bytes ÷ 50 MB/s ÷ `max_lanes`.

| host | lanes | items | requests | wire | pace | transfer | **derived total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `psl` (NCEP) | 1 | 560 | 1,120 | 23.5 GB | 6.22 h | 0.13 h | **6.35 h** |
| `scripps` (Roemmich–Gilson) | 1 | 93 | 186 | 65.1 GB † | 0.52 h | 0.36 h | **0.88 h** |
| `ncei` (OISST) | 6 | 44 | 31,656 | 22.9 GB | 0.73 h | 0.02 h | **0.75 h** |
| `aoml` (cable + SAMBA) | 1 | 44 | 88 | ~0 | 0.12 h | 0.00 h | **0.12 h** |
| `ncei_etopo` | 1 | 1 | 2 | 0.5 GB | 0.00 h | 0.00 h | **0.01 h** |
| `rapid`, `gatech`, `ndbc`, `github_raw` | 1–2 | 5 | 10 | ~0 | — | — | **minutes** |
| `cmems` (GLORYS) | 4 | 384 | 0 | 0 | — | — | **minutes** (verify only) |

† The registry gives every Roemmich–Gilson item the base files' ~0.7 GB, so
65 GB is an over-estimate: the monthly extensions are much smaller.

**Lanes run in parallel, so the run's wall clock is the longest lane: about
6.4 hours, set entirely by PSL.** That is deliberate — PSL is one lane with a
twenty-second gap because two large files back to back made it answer 504 and
go silent for a quarter of an hour. If your run takes considerably longer,
look at the `backoffs` column before assuming anything is broken.

### Re-running after an interruption

**Run exactly the same command again.** The pipeline lists the Hub once at the
start and skips everything already there, and it checks each item's Hub path
again immediately before fetching it. A second run over a completed tier
reports every item as `present` and downloads nothing. This is also what you
do after a crash, a reboot, or a lost connection.

### What a breaker trip looks like, and what to do

```
[psl/0] attempt 1 failed (HEAD …: HTTP 504); slept 61s
[psl/0] attempt 2 failed (HEAD …: HTTP 504); slept 297s
…
[psl/0] CIRCUIT BREAKER TRIPPED — the rest of this lane is deferred. Wait at least an hour, then re-run.
```

Five items in a row failed, so that lane stopped on purpose. Everything it had
not reached comes back with status `deferred` — **that is not a failure**, it
is a note that the work is still to do.

- **First trip on a host:** wait **at least one hour**, then re-run the same
  command. The deferred items get picked up.
- **Second trip on the same host in one run, or a trip on the re-run:**
  **stop.** Do not re-run a third time. Write the report in §9, say which host,
  how many trips, and paste the exact error text. Chris decides whether the
  budget in `sources.yaml` comes down.

---

## 5 · Verifying Tier 0 and publishing the manifest

```bash
python -m beam_import.verify_hub --tiers 0
```

Expected when the tier is complete:

```
tier(s) [0]: 1131 file(s) expected, 1131 on the Hub
missing: 0
extra (on the Hub under a tier prefix, not in the manifest): 0
```

The command exits 0 when `missing` is 0 and 3 otherwise, so it can be used in a
shell condition. Anything listed as `MISSING` should also appear in
`report.jsonl` with a status of `failed`, `deferred`, `absent` or `blocked` —
if a file is missing on the Hub and `published` in the report, say so loudly;
that would mean the restore-verify is lying and it is the most serious thing
that can go wrong here.

Then write the manifest onto the Hub, in one commit:

```bash
python -m beam_import.verify_hub --tiers 0 \
    --report out/tier0/report.jsonl --out-dir out/tier0 --publish
```

Expected:

```
published sources/MANIFEST_tier0.json (1131 file records)
```

Each record carries the Hub path, the source, the host, the status, the byte
count, the sha256 the restore-verify compared, the upstream URL or product id,
and the transform that was applied.

Finally read the summary:

```bash
cat out/tier0/summary.md
```

It has three tables: counts by status; **per host — the politeness audit**
(requests, bytes, wall time, backoffs, breaker trips); and per source. Then a
table of everything that is not `published` or `present`, with its error text.
The politeness audit is the part Chris reads first. **A host with many
backoffs or any trips means its budget is too generous and should come DOWN
before the next run** — never up.

---

## 6 · Running Tier 1

Tier 1 is everything the next model family needs that requires no new account.
Most of it is small. One source is not.

**Run DUACS on its own.** It is ~1.1 TB on the wire — 384 monthly items, each
fetched one day at a time from Copernicus Marine and binned down to 0.25° —
and at four CMEMS lanes it takes **two to three days**:

```bash
python -m beam_import.pipeline \
    --tiers 1 --only duacs \
    --report-dir out/duacs --state-dir /var/tmp/beam_import \
    --runner DirectRunner --direct_running_mode multi_processing \
    --direct_num_workers 8
```

It may be interrupted and resumed at will: kill it, reboot, run the same
command again. Finished months are on the Hub and are skipped.

**Before the first DUACS run**, resolve the real dataset identifier. The one in
`sources.yaml` is marked `unverified_url` because it is a best guess:

```bash
copernicusmarine describe --product-id SEALEVEL_GLO_PHY_L4_MY_008_047
```

Find the daily 0.125° delayed-time (`_my_`) dataset in the output, paste its
`dataset_id` into the `duacs:` block of `sources.yaml`, and remove the
`unverified_url: true` and `resolve_dataset_id: true` lines from that block. Do
not hardcode a guess. Note also that delayed-time DUACS is produced with a
centred ±6-week window, so the last ~6 weeks of the record simply do not exist
in this product.

Everything else in Tier 1:

```bash
python -m beam_import.pipeline --tiers 1 \
    --report-dir out/tier1 --state-dir /var/tmp/beam_import \
    --runner DirectRunner --direct_running_mode multi_processing \
    --direct_num_workers 8
```

**Six Tier-1 sources carry `unverified_url: true` and will probably fail until
a human checks them.** That is expected and it is why they are flagged. They
are `en4` (a HEAD on the 2020 file returned **404** on 2026-09-04 — the path
has moved; open <https://www.metoffice.gov.uk/hadobs/en4/download-en4-2-2.html>
and copy the real one), `pmel_wwv` (the configured URL is an index page, not a
file), `godas` (a directory, not a file — it needs rewriting as a `var_year`
source), `oni` (this sandbox's proxy refused the host, so it is untested),
`mei`, and `rmm` (a HEAD returned **403** on 2026-09-04). Report what each one
does; do not invent replacement URLs.

---

## 7 · Tier 2

### ERA5, once Chris provides the CDS key

Today every ERA5 item is reported `blocked` and **nothing is requested**. When
the key exists:

```bash
export CDSAPI_URL='https://cds.climate.copernicus.eu/api'
export CDSAPI_KEY='<the key Chris gives you>'
pip install cdsapi                       # already in requirements.txt

# start with ONE month of ONE variable and look at the file before
# committing to 7,224 requests
python -m beam_import.pipeline --tiers 2 --only era5 --dry-run \
    --report-dir out/era5dry --state-dir /var/tmp/beam_import \
    --runner DirectRunner
```

The full source is 14 variables × 516 months = **7,224 requests**, two in
flight at a time. That is a lot of queueing and it should be discussed with
Chris before it is started. Check the variable names against the dataset's own
form on the CDS website first — `sources.yaml` lists ERA5 **short** names
(`metss`, `u10`, `t2m` …) and some CDS datasets want the long ones.

### ESA CCI soil moisture

It is `enabled: false` and its path pattern is a guess, because the real
directory listing can only be read after registering a free account at
<https://services.ceda.ac.uk>. To enable it: register, browse to the v09.1
COMBINED daily 0.25° product, put the real URL pattern into the `cci_sm:` block
of `sources.yaml`, remove `unverified_url: true`, set `enabled: true`, and run
`--tiers 2 --only cci_sm`.

### Why OSTIA and GREP-3D are off

**OSTIA** is a 0.05° sea-surface-temperature and sea-ice product: roughly
**3 TB on the wire**, for a sea-ice fraction that OISST already gives us at a
fraction of the cost. It is off by default and should stay off unless somebody
has a reason.

**GREP-3D** is the three-dimensional (depth-resolved) ocean ensemble
reanalysis: about **200 GB stored** globally, and the plan document E-070 §6
says the streaming data loader that would use it has to be built first. It is
off by default. If a smaller version is ever wanted, `sources.yaml` carries a
North-Atlantic window (`bbox_na_option`, longitude −100…20, latitude 0…70) to
swap into `bbox` deliberately. Its depth levels must be **read from the first
file the server sends**, never hardcoded.

---

## 8 · Optional: Dataflow, and the GitHub Actions fallback

You almost certainly do not need either.

**Dataflow** (Google Cloud's managed Beam runner) is a flag change:

```bash
python -m beam_import.pipeline --tiers 0 \
    --report-dir gs://<bucket>/tier0 --state-dir /var/tmp/beam_import \
    --runner DataflowRunner \
    --project <gcp-project> --region <region> \
    --temp_location gs://<bucket>/tmp \
    --max_num_workers 3 --number_of_worker_harness_threads 4
```

It buys nothing here except managed retries, which — per `DESIGN.md` §2 — we
specifically do not want at the bundle level. **The warning that matters:
pipeline options are logged and displayed in the Dataflow job UI, so a
credential passed as an option is a published credential.** On Dataflow the
credentials must come from Google Secret Manager, read inside `DoFn.setup()`,
never from `--flags`. Use Dataflow only if the machine, and not the hosts,
turns out to be the bottleneck.

**The GitHub Actions fallback** is the pattern an earlier phase used: run the
same command inside a GitHub-hosted runner, several lanes at a time, on
`workflow_dispatch`. Because the pipeline is resumable from the Hub, the
350-minute job cap just means more firings — each one picks up where the last
left off. It costs nothing. If you use it, the Hugging Face token goes in a
repository secret and is exported into the environment by the workflow step,
never onto a command line.

---

## 9 · What to report back to Chris

Send **one markdown message**. Use this template, filled in with real numbers
taken from `summary.md` and `report.jsonl` — never from memory, and never a
number you did not read out of an artefact.

```markdown
## Import status — Tier <N> — <date, UTC>

**TL;DR:** <one plain sentence: is the tier done, and if not, what is missing.>

### Counts per tier
| tier | items | published | present | absent | deferred | failed | blocked |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1131 (747 to fetch + 384 verify-only) | | | | | | |

### Bytes and wall time per host
| host | lanes | items | requests | bytes | wall | backoffs | breaker trips |
|---|---:|---:|---:|---:|---:|---:|---:|
| psl | 1 | | | | | | |
| ncei | 6 | | | | | | |
| ... | | | | | | | |

### Wire GB fetched vs the manifest's estimate
Fetched <X> GB against the manifest's "to fetch" estimate of <Y> GB
(<Z>%). <One sentence if they differ by more than ~20%, saying which source
accounts for it.>

### Hub manifest
`sources/MANIFEST_tier<N>.json` — <n> file records.
`verify_hub --tiers <N>` reports missing: <n>.

### Anything absent, deferred, failed or blocked
| item | status | error text (verbatim) |
|---|---|---|
| | | |

### What I could not verify
- <e.g. "the `en4` URL pattern still 404s; I did not look for a replacement">

### What I would do next
- <one or two sentences>
```

Rules for the report: **paste error text verbatim**, do not paraphrase it.
**Never write a bare identifier** — every source name gets a short plain-English
gloss the first time it appears. **Every number comes from a file**, and say
which file. And if a breaker tripped twice and you stopped, that goes in the
TL;DR, not in a footnote.

Where the numbers come from: the per-tier and per-host tables are the two
tables in `summary.md`. The **wire GB fetched** is the sum of that file's
per-host `bytes` column; the **manifest's estimate** is the `to fetch:` line
printed by `python -m beam_import.manifest --tiers <N>` — 104.3 GB for Tier 0.
A large gap between them is worth a sentence: it usually means a source's
`bytes_wire` in `sources.yaml` is a bad estimate (the Roemmich–Gilson rows are
known to over-estimate), which is useful for the next run's planning.

---

## 10 · Troubleshooting

| symptom | cause | what to do |
|---|---|---|
| `HTTP 504` from `downloads.psl.noaa.gov`, then nothing at all for ~15 min | PSL throttles after two large files. This is the measured behaviour the whole backoff ladder was built for. | Nothing. The lane sleeps 60 s, 5 min, 15 min, 60 min and carries on. If it trips the breaker, wait an hour and re-run. **Do not raise `max_lanes` for `psl`.** |
| `NetCDF: HDF error` when something opens a file | The download was **truncated**. A short transfer raises no exception at the socket; it surfaces here, much later. | The size check in `fetchers.py` (HEAD `Content-Length` vs bytes on disk) should have caught it. If it did not, the server sent no `Content-Length` — report the URL. Delete the local state directory and re-run. |
| `Hub rate limit … hourly quota` / HTTP 429 on a commit | The Hugging Face hourly commit quota. | Nothing. The code honours `Retry-After` and backs off. If it keeps happening, lower `hub.commit_budget_per_hour` in `sources.yaml` (yes, lower) and re-run. |
| `Hub commit conflict` / HTTP 409 or 412 | Someone or something else committed to the repository while we were preparing our commit. | Nothing. It is retried; our operations are additions at distinct paths. If it repeats, check nobody else is writing to `chfrank/earth-tensors` right now. |
| `403 You don't have the rights to create a model under the namespace "blauewelt"` | The **namespace trap**: `blauewelt` is the GitHub organisation and does not exist on the Hugging Face Hub. | The Hub account is the user **`chfrank`** and the repository is `chfrank/earth-tensors`. Check `sources.yaml`'s `hub.repo_id`. |
| `copernicusmarine` authentication error | Almost always the `%` in the password being eaten by the shell. | Re-export it in **single quotes**: `export COPERNICUSMARINE_SERVICE_PASSWORD='<password from CREDENTIALS.md — has a %>'`. Then check with the "set / NOT set" loop in §2.3. |
| The process is killed; `dmesg` says `Out of memory` | An item is too big for one machine. This means **the registry's `chunk` for that source is wrong** — e.g. a whole month of GLORYS in one request is 5.95 GB. | Fix the chunking (fetch per day), **not** the worker count. Adding workers makes it worse. Report it before changing `sources.yaml`. |
| `PicklingError` / `Can't pickle …` when the pipeline starts | Beam pickles the DoFn and everything it holds. Something unpicklable got into `LaneWorker.__init__`. | Clients belong in `DoFn.setup()`, configuration in the plain dict passed to `__init__`. Do not add an open file, a session or a client to the constructor. |
| The DirectRunner hangs forever at start, or spawns endless processes | A missing `if __name__ == '__main__':` guard. In `multi_processing` mode every child re-imports the module, and without the guard each child starts its own pipeline. | `beam_import/pipeline.py` has the guard. If you write a new entry point, it needs one too. |
| `No space left on device` | Downloads that were not cleaned up — usually because a run was killed between fetching and publishing. | `du -sh /var/tmp/beam_import/*` to see where it went, then delete the per-item subdirectories. Files are removed only after the restore-verify passes, so anything left over is from an interrupted run and is safe to delete: it will be fetched again. |
| Everything in a tier comes back `blocked` | A gated source with no credentials — ERA5 without `CDSAPI_KEY`, CMEMS without the two `COPERNICUSMARINE_*` variables. | Read the error text; it names the variables. For ERA5 this is the correct behaviour until Chris makes the account. |
| An item is `absent` | The archive genuinely does not have that file — a Florida-cable year that was never measured, a Roemmich–Gilson month that was never published. | Nothing. `absent` is a fact about the archive, it does not count towards the circuit breaker, and it is reported as such. |
| `no earth checkout at …` | `EARTH_REPO` is unset or points somewhere without an `ml/` directory. | `git clone https://github.com/blauewelt/earth.git` and `export EARTH_REPO=…`. **Do not** write your own binning function (hard rule 4). |
