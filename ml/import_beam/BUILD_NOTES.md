# Build notes — what was built, what was measured, what was not

Written 2026-09-04 by the session that implemented `DESIGN.md`. This is the
record for whoever has to trust or repair the package; `README_FOR_GEMINI.md`
is the howto, this is the audit.

---

## 1 · What is in the package

```
handover/
  README_FOR_GEMINI.md   the howto (sections 0-10 as specified)
  DESIGN.md              unchanged — not one substantive edit, no typo found
                         worth fixing
  CREDENTIALS.md         the credentials plus the rules and the namespace trap
  sources.yaml           the registry: 16 hosts, 26 sources, 3 tiers
  requirements.txt       pinned to what this was tested against
  setup_env.sh           venv with setuptools<70 installed FIRST
  run_smoke.sh           offline smoke on the DirectRunner, 2 worker processes
  run_tier0.sh           the real Tier-0 invocation, credentials from env
  beam_import/           registry, manifest, hosts, fetchers, transforms,
                         publish, pipeline, verify_hub, report  (1,692 lines
                         of code excluding comments and docstrings)
  tests/                 7 test modules + a fixture generator (739 lines);
                         every fixture is GENERATED, no binary blobs in the
                         package
```

The design's §2 rules are implemented where the design puts them: per-host lane
cap (`sources.yaml` `hosts:` → the GroupByKey key), `min_gap_s` (`LaneState.pace`),
the 60 s / 5 min / 15 min / 60 min ladder with 0.8–1.2 jitter and `Retry-After`
(`LaneState.backoff_seconds`), the five-consecutive-failure circuit breaker
(`LaneState.record_failure`, `deferred` records in `LaneWorker.process`), skip
before fetch twice (a side-input listing taken once plus a per-item
`exists()`), one day per CMEMS request, the Hub commit budget
(`publish.commit_min_interval_s`), restore-verify then delete
(`BasePublisher.publish_verified`), no credentials in argv or options, and a
`try/except` around everything transient so only a programming error can
propagate out of the DoFn.

Added after the first review pass (2026-09-04), and reflected in the numbers
below:

- **Verify-only items no longer count as wire.** `manifest.counts()` returns
  separate fetch and verify totals and the CLI prints `already on Hub (verify
  only): …` and `to fetch: …` lines. Counting GLORYS's 2.2 GB-per-month
  download in a "how much will this fetch" figure had Tier 0 reading 925 GB
  when the run actually transfers 104 GB.
- **Live progress** (`ml/CLAUDE.md` §5.25). `LaneWorker.process` appends every
  result record to `<state-dir>/progress/<host>-<lane>.jsonl` — one file per
  lane, open-append-fsync-close per record, **written before the record is
  yielded** — so a run killed at hour six still has everything it did.
  `python -m beam_import.report --live <state-dir>` summarises them mid-run,
  and `summary.md` is now built from the union of the progress files and
  `report.jsonl`, deduped by item with the final record winning. Every record
  carries `at` (UTC) plus a running `backoffs_so_far` / `trips_so_far`
  snapshot, so the politeness audit is readable before a lane finishes.
- **A derived wall-time table** in README §4, computed from the registry's own
  per-item bytes and each host's `min_gap_s` / `max_lanes`, and labelled as
  derived rather than measured.

No budget in `sources.yaml` was changed.

---

## 2 · Test results, verbatim

### `python -m pytest tests -q`

```
...............................................                          [100%]
47 passed in 16.84s
```

(Environment: `/home/claude/beamenv/bin/python`, `EARTH_REPO=/home/claude/earth`.)

### `bash run_smoke.sh` — offline, DirectRunner `multi_processing`, 2 workers

```
== 3/4  the manifest
--- counts by source ----------------------------------------
  tiny             tier 0       1 items  host testhost
  oisst            tier 0       1 items  host testncei
  ncep             tier 0       1 items  host testpsl
  flaky            tier 0       7 items  host testflaky
--- counts by host ------------------------------------------
  testflaky           7 items over 1 lane(s)
  testhost            1 items over 2 lane(s)
  testncei            1 items over 2 lane(s)
  testpsl             1 items over 1 lane(s)
-------------------------------------------------------------
  to fetch: 10 items · 48.9 KB wire · 48.9 KB stored
  TOTAL 10 items · 0 with an unverified URL

== 4/4  the pipeline on the DirectRunner, 2 worker processes
preflight: Hub round trip OK (sha256 0178c24b0dfc)
manifest: 10 item(s); Hub already holds 1 file(s)
[testflaky/0] CIRCUIT BREAKER TRIPPED — the rest of this lane is deferred. …
[testhost/0] verified sources/tiny/hello.dat (8a2537b134ac)
[testncei/0] verified sources/oisst/oisst_daily_2001.nc (22d298f0a40c)
[testpsl/0] verified sources/ncep/uflx.sfc.gauss.2001.nc (94beedc94f8b)

== checking what the run produced
  ok   three sources published (file:// http, OISST year-fold, NCEP-like)
  ok   every published item carries the sha256 the restore-verify compared
  ok   the flaky source failed exactly 5 items (the breaker threshold)
  ok   the rest of that lane is `deferred`, not `failed`
  ok   exactly one circuit-breaker trip was recorded
  ok   the folded OISST year is on the fake Hub
  ok   the Hub round-trip preflight ran
  ok   summary.md was written
  ok   live progress: one file per lane (4), 14 records appended during the run

SMOKE TEST: PASS
```

The smoke run's own report, by status: `published 3`, `failed 5`,
`deferred 2`, plus 4 lane-counter records — 14 records, and all 14 were on
disk in the per-lane progress files while the run was still going. Re-running
the identical command against the same fake Hub gave `present 3, failed 5,
deferred 2` and fetched nothing — the idempotence claim is measured, not
asserted.

`python -m beam_import.report --live <state-dir>` on that run's state
directory printed:

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

Three tests pin the live path: a record is on disk after the FIRST item even
when the generator is thrown away mid-lane (`LaneWorker.process` called
directly on a 2-item iterable, `next()` once, `close()`); `report.live()`
reads it; and `summary.md` from the union of a progress file and a
`report.jsonl` that disagree about one item keeps the final record and counts
the item once.

### Manifest counts — `--tiers 0 --print`

| source | items | host | registry `expected_items` |
|---|---:|---|---|
| glorys | 384 | cmems | 384 |
| oisst | 44 | ncei | 44 |
| ncep | 560 | psl | 560 |
| rg | **93** | scripps | scraped (null) |
| rapid | 1 | rapid | 1 |
| florida_cable | 43 | aoml | 43 |
| osnap | 1 | gatech | 1 |
| move | 1 | ndbc | 1 |
| samba | 1 | aoml | 1 |
| etopo | 1 | ncei_etopo | 1 |
| natural_earth | 2 | github_raw | 2 |
| **total** | **1131** | | see the totals lines below |

The totals, as printed:

```
  already on Hub (verify only): 384 items, 96.4 GB
  to fetch: 747 items · 104.3 GB wire · 103.5 GB stored
  TOTAL 1131 items · 0 with an unverified URL
```

**104.3 GB is what a Tier-0 run actually transfers**, and it matches DESIGN §3's
"about 110 GB on the wire, about the same stored". The first draft of this
document quoted 925.1 GB wire / 199.9 GB stored, which counted the 384
verify-only GLORYS months as though they would be downloaded; they are already
on the Hub and only their presence is checked. Fixed in `manifest.counts()`,
printed as two separate lines, and pinned by
`tests/test_manifest.py::test_verify_only_items_are_not_counted_as_wire`.

`rg` = 93 came from a live scrape of `https://sio-argo.ucsd.edu/RG_Climatology.html`
on 2026-09-04: 2 base files + **91** monthly extensions. With `--offline` the
fallback range 2019-01…2025-12 gives 86 and Tier 0 totals 1,124.

`--tiers 0,1,2 --offline`: **8,867 items**, of which **8,483 to fetch —
1.2 TB wire, 273.2 GB stored** — plus the same 384 verify-only months
(96.4 GB), and **518 items flagged `unverified_url: true`**. Tier 1 = 519
items (duacs 384, en4 126, the rest small); Tier 2 = 7,224 (ERA5 only —
`cci_sm`, `ostia` and `grep3d` are `enabled: false` and are excluded unless
named with `--only`).

### Derived Tier-0 wall time — NOT measured

Computed from the registry's per-item byte counts and each host's `min_gap_s`
and `max_lanes`, assuming 50 MB/s of usable throughput per lane, two requests
(HEAD + GET) per file, and ~365 files per OISST item. Nobody has run Tier 0
end to end, so these are planning figures. Reproduced in README §4 with the
same "derived, not measured" warning.

| host | lanes | items | requests | wire | pace | transfer | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| psl | 1 | 560 | 1,120 | 23.5 GB | 6.22 h | 0.13 h | **6.35 h** |
| scripps | 1 | 93 | 186 | 65.1 GB * | 0.52 h | 0.36 h | **0.88 h** |
| ncei | 6 | 44 | 31,656 | 22.9 GB | 0.73 h | 0.02 h | **0.75 h** |
| aoml | 1 | 44 | 88 | ~0 | 0.12 h | 0.00 h | **0.12 h** |
| ncei_etopo | 1 | 1 | 2 | 0.5 GB | 0.00 h | 0.00 h | **0.01 h** |
| rapid, gatech, ndbc, github_raw | 1–2 | 5 | 10 | ~0 | — | — | minutes |
| cmems | 4 | 384 | 0 | 0 | — | — | minutes (verify only) |

\* over-estimated: the registry gives every Roemmich–Gilson item the base
files' ~0.7 GB and the monthly extensions are far smaller. Lanes run in
parallel, so the run's wall clock is the longest lane — **about 6.4 h, set
entirely by PSL's single 20-second-gap lane.**

### `registry --check`

```
registry  /home/claude/handover/sources.yaml
hub       chfrank/earth-tensors (60 commits/h)
hosts     16   total lanes 27
sources   26
  tier 0: 12  glorys oisst oisst_psl ncep rg rapid florida_cable osnap move samba etopo natural_earth
  tier 1: 10  duacs cmems_static rapid_extra en4 pmel_wwv olr godas oni mei rmm
  tier 2: 4  era5 cci_sm ostia grep3d
OK
```

### Binning equality

`tests/test_transforms.py` asserts (a) that `bin_plan`/`bin_slice` come from
the module `aggregate_cadence` in the earth checkout, (b) that a toy 1/12°
patch with an injected NaN bins to a bit-identical array through
`transforms.bin025`'s exact call, and (c) that `bin025` on the CMEMS-shaped
fixture equals a direct `bin_slice` on the same raw array, element for element,
on every finite cell. It also asserts that the axes land on multiples of 0.25
(the `point` alignment) and that `oisst_year_fold` does **not** regrid — the
folded year's latitudes are still OISST's cell centres.

---

## 3 · Host HEAD results, measured 2026-09-04

One HEAD per host, from this sandbox, through its egress proxy.

| host | URL probed | result |
|---|---|---|
| `cmems` | `https://data.marine.copernicus.eu/` | **200**, 329,182 B |
| `psl` | `…/ncep.reanalysis/surface_gauss/land.sfc.gauss.nc` | **200**, 25,213 B, `application/x-netcdf` |
| `ncei` | `…/avhrr/198109/oisst-avhrr-v02r01.19810901.nc` | **200**, 1,714,749 B |
| `scripps` | `…/RG/RG_ArgoClim_Temperature_2019.nc.gz` | **200**, **695,480,508 B** (≈0.7 GB, not the ~2 GB the brief estimated) |
| `rapid` | `…/rapid_data/moc_transports.nc` | **200**, 1,182,284 B |
| `aoml` | `…/WBTS/cable/FC_cable_transport_2020_v3.dat` | **200**, 18,607 B |
| `ncei_etopo` | `…/ETOPO_2022_v1_60s_N90W180_surface.nc` | **200**, **no `Content-Length`** on HEAD |
| `github_raw` | `…/geojson/ne_10m_lakes.geojson` | **200**, 1,570,533 B |
| `gatech` (OSNAP) | the bitstream `…/597db471-…/content` | **200**, no `Content-Length` |
| `ndbc` (MOVE) | `…/MOVE/OS_MOVE_…_VOLUMETRANSPORT.nc` | **200**, 172,492 B |
| `pmel` | `https://www.pmel.noaa.gov/tao/wwv/data/` | **200**, 799 B — it is an **index page**, not a file |
| `cds` | `https://cds.climate.copernicus.eu/api` | **202**, JSON (reachable; no key, so nothing else was tried) |
| `ceda` | `https://dap.ceda.ac.uk/` | **200** |
| `metoffice` | `…/en4-2-2/EN.4.2.2.analyses.g10.2020.zip` | **404** — the EN4 path pattern is WRONG or has moved |
| `bom` | `…/mjo/graphics/rmm.74toRealtime.txt` | **403** — the host refuses a plain HEAD |
| `cpc` | `…/ensostuff/ONI_v5.php` | **blocked by the sandbox proxy** (`ProxyError`); not retried |
| `hf` | `https://huggingface.co/api/whoami-v2` | **401** without a token — i.e. the endpoint is reachable |

Every Tier-0 host answered 200. The three problems (`metoffice` 404, `bom` 403,
`cpc` proxy-blocked) are all Tier 1, all already carry `unverified_url: true`
in the registry, and each has a note saying exactly what a human must check.

---

## 4 · What I could not verify

- **The EN4 URL pattern.** A HEAD on the 2020 zip returns 404. I did not hunt
  for the replacement — that needs a human reading the Met Office download
  page. Registry note says so.
- **`cpc` (the ONI index).** The sandbox proxy refused the host. Untested by
  request, per the one-probe-per-host rule.
- **`bom` (the RMM index).** 403 on HEAD. The fetchers send a browser-like
  `User-Agent`, which often fixes this, but I did not spend a second request
  finding out.
- **`pmel_wwv` and `godas`.** Both configured URLs are directory/index pages,
  confirmed for `pmel` by the 799-byte HTML response. They need a human to pick
  the actual files; the registry rows say so and are flagged.
- **The DUACS `dataset_id`** and the two other CMEMS ids
  (`cmems_mod_glo_phy_my_0.083deg_static`, the GREP and OSTIA products). I did
  not run `copernicusmarine describe` — that is an authenticated call against a
  production service and the brief limits me to one probe per host. All three
  are marked `resolve_dataset_id: true` and `unverified_url: true`, and the
  README tells the operator to resolve them first.
- **The CMEMS, CDS and Hugging Face credentials themselves.** Nothing in this
  session authenticated anywhere. The Hub round-trip preflight, the commit
  path, the download-back path and the 429/409 handling were exercised only
  against the `local:` publisher.
- **ERA5's exact CDS request body.** `cds_fetch` builds a
  `derived-era5-single-levels-daily-statistics` request with what that dataset
  documents, but with no account it has never been sent. Treat it as a first
  draft; the README says to try one variable-month before 7,224.
- **Real upstream throughput.** Every wall-time figure in the README is derived
  from the design's measured per-item numbers and the registry's item counts,
  not from a run.

---

## 5 · Deviations from DESIGN.md, and why

1. **`ncep` is 13 variables and 560 items, not 14 and 603.** DESIGN §3 says
   "14 variables × 1982–2024 plus the land mask (603 files)". The repository's
   own `NCEP_FILES` table in `ml/build_family7.py` has **thirteen** stems, so
   13 × 43 + 1 = 560. The brief said to make the registry the truth and print
   the real count; that is what happens, and the discrepancy is written into
   the registry's own `notes:` and into the README's expected-counts table.

2. **Two hosts added that DESIGN §2's table does not list:** `gatech`
   (repository.gatech.edu, the OSNAP series) and `ndbc` (dods.ndbc.noaa.gov,
   the MOVE series). DESIGN names both sources but gives neither a host row,
   and every source must name a host. Both were given `max_lanes: 1` — the
   floor, so nothing is raised — and `min_gap_s: 5`, and both rows say
   "ADDED (not in DESIGN.md §2)" in their `evidence:` field.

3. **`cpc` and `bom` are two host rows, not one.** DESIGN's table has a single
   `cpc / bom` row. They are different domains, so they are two rows of one
   lane each — which is stricter than sharing one budget, not looser.

4. **Total lanes sum to 27, where DESIGN §2 says "total lanes ≤ ~12".** That
   is DESIGN's own table summing to 25 before my two additions; I did not
   change any budget to make the sentence true. In practice concurrency is far
   below 27 because eleven of the sixteen hosts have one lane and a handful of
   items. Flagged here rather than silently reconciled.

5. **No automatic NCEI→PSL fallback for OISST.** DESIGN §3 says PSL yearly
   files are the fallback, and the brief asks for a `psl_fallback` fetcher.
   Falling back *inside an NCEI lane* would have opened a second concurrent
   stream to PSL without PSL's one-lane budget agreeing to it — the exact thing
   §2 says must not happen by accident. So the fallback is a separate source,
   `oisst_psl`, on the `psl` host, `enabled: false`, run explicitly with
   `--only oisst_psl`. The `psl_fallback` fetcher exists and is what that
   source uses. Both the registry note and the fetcher docstring explain it.

6. **A `days:` knob on year-chunked sources.** An optional explicit day list
   that overrides the computed one. Its honest use is a partial re-fetch; its
   immediate use is letting the test fixture make a "year" three days long
   instead of committing 365 synthetic NetCDF files.

7. **`--dry-run` builds no Hub client at all**, so it runs with no credentials.
   DESIGN does not say either way; this makes the "show me the plan" path
   usable on a machine that has nothing configured.

8. **Code size.** The brief targeted ~1,500 lines; the package is **1,692
   lines of code** after the live-progress and byte-accounting additions, plus
   a comparable volume of comments and docstrings. The
   comments are deliberate — the reader is a junior operator and several of
   the rules (the `%` in the password, the OISST middle-gap rule, the
   `__main__` guard) are only obvious once explained.

---

## 6 · Things a future maintainer should know

- **`report.jsonl` is written by Beam only at the end of the run**, so the
  live artefact is elsewhere: every lane appends its records to
  `<state-dir>/progress/<host>-<lane>.jsonl` as it goes — one file per lane so
  nothing has to be locked, fsynced per record, and **written before the
  record is yielded to the runner**. `report --live <state-dir>` reads them
  mid-run; `summary.md` is the union of those files and `report.jsonl`,
  deduped by item id with the final record winning. This is `ml/CLAUDE.md`
  §5.25 ("progress is an artefact, not a log line"). A torn last line in a
  progress file is skipped rather than raising, because the file is read while
  it is being appended to.
- **Lane assignment is `sha1(item_id) mod max_lanes`, not `hash()`.** Python's
  `hash()` is salted per process, so under a multi-process runner the same
  item would land in different lanes in different workers and the lane cap
  would stop meaning anything. `tests/test_manifest.py` pins this.
- **`publish_verified` deletes the local file only after the sha256 of the
  downloaded-back copy matches.** Anything left in the state directory after a
  run is from an interrupted one and is safe to delete.
- **The circuit breaker counts consecutive failed ITEMS, not failed requests.**
  Each item gets its own five-attempt ladder first. `absent` does not count.
- **`verify_hub` exits 3 when anything is missing**, so it composes into a
  shell condition; `--publish` writes `sources/MANIFEST_tier<N>.json` in one
  commit and nothing else.
- **Nothing in this package deletes anything on the Hub**, and there is no
  code path that could. Keep it that way.
