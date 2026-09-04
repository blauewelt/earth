# Build notes — what was built, what was measured, what was not

Written 2026-09-04 by the session that implemented `DESIGN.md`, and rewritten
the same day for **revision 2** (no Hugging Face Hub; never drop data; a
sharded `tf.train.Example` output in two stages). This is the record for
whoever has to trust or repair the package; `README_FOR_GEMINI.md` is the
howto, this is the audit.

---

## 1 · What is in the package

```
handover/
  README_FOR_GEMINI.md   the howto, rewritten for revision 2
  DESIGN.md              revision 2 — unchanged, not one substantive edit
  CREDENTIALS.md         CMEMS + the CDS placeholder. The HF token is GONE.
  sources.yaml           the registry: 17 hosts, 27 sources, 3 tiers
  requirements.txt       pinned; tensorflow deliberately NOT among them
  setup_env.sh           venv with setuptools<70 installed FIRST
  run_smoke.sh           offline: Stage A + queue + run_until_complete + Stage B
  run_tier0.sh           one Tier-0 round
  run_until_complete.sh  the loop: run, sleep 1/2/4/8 h, run from the queue
  beam_import/
    registry.py    manifest.py   hosts.py      fetchers.py   transforms.py
    tfrecord.py    example.py    sinks.py      pipeline.py   assemble.py
    verify_output.py  report.py
                         2,662 lines of code (comments and docstrings excluded)
  tests/                 10 test modules + a fixture generator (TEST2,662 lines);
                         every fixture is GENERATED, no binary blobs
```

### What changed from revision 1

| revision 1 | revision 2 |
|---|---|
| `publish.py` — Hugging Face commits, restore-verify, commit quota | **deleted.** `sinks.py` (write-verify-mark to `--output`) + `tfrecord.py` + `example.py` |
| `verify_hub.py` | **deleted.** `verify_output.py` — markers, byte counts, deep CRC re-read, the absent list |
| statuses `published/present/planned/absent/missing/deferred/blocked/failed` | statuses **`written` / `present` / `queued` / `absent`**, and no fifth |
| a `failed` terminal state | **gone.** Everything not written and not proven absent is appended to `retry_queue.jsonl` |
| `absent` on one 404 | `absent` needs **two** 404/410s, ≥ 6 h apart, both responses kept in `<state-dir>/absent_evidence/` |
| a middle gap in an OISST year was fatal | the days that arrived are written, the missing dates go in the marker AND on the queue as day-level items |
| output = one NetCDF per item on the Hub | output = one TFRecord shard of `tf.train.Example` per item under `--output`, plus a `.done` marker |
| (no second stage) | **Stage B** — `assemble.py`, pentad × channel group, `spec.json` + `coverage.json` |
| `hub:` in the registry | `output:` (shapes only); `--output <uri>` on the command line |
| GLORYS was `mode: verify` | GLORYS is fetched; `glorys_from_mirror` is the optional 9×-cheaper path |

**Added after the second review pass (same day):** Stage B takes
`--cadence pentad|daily` (default `pentad`; the Tier-0 flow is unchanged),
plus `--min-days` and `--groups`. Stage A is the daily archive and Stage B is
a derived view of it; the two views live in `<output>/pentad/…` and
`<output>/daily/…`, each with its own `spec.json` and `coverage.json`. At
daily cadence a bin IS a day (`date_start == date_end`), `min_days` is 1,
`rg100` is skipped by default (Roemmich-Gilson Argo is monthly, so a daily
record of it would be mostly invented) and the two stress sigma channels are
renamed `tau_x_std_day` / `tau_y_std_day`, because a within-DAY sigma over
four 6-hourly samples is a different quantity from the within-pentad sigma
over twenty. E-070 §7's centred 5-day sigma is a rolling window over those
daily records and is deliberately NOT baked in here. DESIGN §1, §4 and §7
were amended to say all of this, including the one line §7 needed: Stage A
reduces the 4x-daily NCEP samples to a daily mean plus a daily
mean-of-squares, so DAILY is the floor for everything downstream — fine for
the multi-rate cone, cheap to undo for the frozen NCEP archive, and a
decision to take BEFORE importing hourly ERA5.

DESIGN §2's rules are implemented where the design puts them: per-host lane cap
(`sources.yaml` → the GroupByKey key), `min_gap_s` (`LaneState.pace`), the
60 s / 5 min / 15 min / 60 min ladder with 0.8–1.2 jitter and `Retry-After`
(`LaneState.backoff_seconds`), the five-consecutive-failure breaker followed by
the queue (`LaneState.record_failure` + `sinks.enqueue`), run-until-complete
(`run_until_complete.sh`), absent-needs-evidence-twice
(`sinks.record_not_found`), gaps masked never skipped (`missing_dates` +
day-level items + Stage B's `days_present`), skip-before-fetch twice (a
`.done` side input plus a per-item marker check), one day per CMEMS request,
write-verify-mark-delete (`sinks.write_verify_mark`), no credentials in argv or
options, and a `try/except` around everything transient so only a programming
error can propagate out of the DoFn.

---

## 2 · Test results, verbatim

### `python -m pytest tests -q`

```
........................................................................ [ 72%]
............................                                             [100%]
100 passed in 41.26s
```

(Environment: `/home/claude/beamenv/bin/python`, `EARTH_REPO=/home/claude/earth`.)

The ten modules: `test_registry`, `test_manifest`, `test_hosts`,
`test_tfrecord`, `test_example`, `test_sinks`, `test_transforms`,
`test_pipeline_smoke`, `test_no_raise`, `test_assemble`. What the new ones pin:

- **TFRecord** — the frame round-trips; a flipped payload bit, a flipped length
  bit and a truncated tail are each caught with a distinct error; TensorFlow
  reads what we wrote and we read what TensorFlow wrote; `masked_crc32c` is
  pinned to the format's own constants.
- **Example** — round trip; the SAME dict always produces the SAME bytes (a
  shard's sha256 is meaningless otherwise); TensorFlow parses ours and we parse
  TensorFlow's, field for field; **and both readers see the identical values on
  a real shard.**
- **sinks** — a wrong sha256 refuses and leaves NOTHING behind; killing the
  process between the rename and the marker leaves the item not-done and the
  next run rewrites it; a fill shard sits beside its parent with its own
  marker; the queue appends, dedupes and rotates; one 404 is not absent, a
  second 404 inside six hours is still one sighting, and two 404s seven hours
  apart are `absent` with both responses on disk.
- **no_raise** — a transient fetcher yields `queued` and the job still
  succeeds; a 404 is `queued` then `absent` across two runs; the breaker queues
  the rest of its lane rather than dropping it; an unclassified `KeyError`
  becomes a `queued` record carrying its traceback; a short transfer is caught
  by the size check; a record is durable on disk after the FIRST item even when
  the generator is thrown away.
- **assemble (cadence)** — the pentad default is unchanged: a run with no
  flag and a run with `--cadence pentad` produce byte-identical records, and
  the fixture's `cur_u` pentad mean is still exactly `0.009829164482653141`,
  the number this package printed and documented BEFORE the flag existed. The
  only difference from the pre-flag records is two ADDED metadata features,
  `cadence` and `cadence_days`, which the same change specified. The daily run
  gives one bin per observed day with `bin == day_index` and
  `date_start == date_end`; `days_present` is 0 or 1 and a channel at 0 is
  all-NaN; `rg100` is absent by default and, when forced with `--groups`,
  appears only on the day containing the 15th with `coverage.json` saying why;
  `tau_x_std_day` sits at the same index `tau_x_std` sits at in the pentad
  view and equals the population sigma computed directly from that day's four
  raw samples; `--min-days 6` blanks a five-day pentad while `days_present`
  still reports the true 5; and the two cadences never share a directory.
- **assemble** — all three groups in one bin; the bin is the imported rule; the
  channel names and order are `build_family7`'s; `days_present` is per channel
  and `min_days` blanks a channel below it; `cur_speed` is the hypot of the
  pentad means; `log_mld` is log10 of the mean; the stress sign is flipped
  exactly once; **`tau_x_std` equals the population sigma computed directly
  from the raw 6-hourly samples** (to 2e-3 after regridding); `rg100` is
  written on the live-month bin only; `spec.json` carries the norms; opaque and
  series records never become pentads; `tf.data` reads the pentad shards.

### `bash run_smoke.sh` — offline, DirectRunner `multi_processing`, 2 workers

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
  ok   the default cadence is pentad, and coverage.json says so
  ok   the daily sidecar emits g025 and g100 and skips rg100 (Argo is monthly)
  ok   the daily sidecar has one bin per observed day (5 of them)
  ok   a daily bin holds one day, so min_days is 1
  ok   a daily record's date_start equals its date_end
  ok   the daily stress sigma is named `tau_x_std_day`, never `tau_x_std`
  ok   daily days_present is 0 or 1 per channel
  ok   one g025 record for one bin
  ok   days_present is per channel and honest: [5, 5, 5, 5, 5, 5, 4]
  ok   the channel order is build_family7's, imported

SMOKE TEST: PASS
```

The smoke run's Stage-B coverage line, verbatim:

```
stage B: 7 shard(s) -> /tmp/smk/out/pentad (2 shards per group)
coverage pentad g025: bins_present=1 range=[1573, 1573] missing_in_range=0
coverage pentad g100: bins_present=1 range=[1573, 1573] missing_in_range=0
coverage pentad rg100: bins_present=1 range=[1573, 1573] missing_in_range=0
coverage daily  g025: bins_present=5 range=[7865, 7869] missing_in_range=0
coverage daily  g100: bins_present=5 range=[7865, 7869] missing_in_range=0
```

The smoke test now runs Stage B twice — the pentad default, then the daily
sidecar over the same Stage-A output — and asserts both.

Every fixture covers 2003-07-15 … 2003-07-19 — pentad bin 1573 — on purpose,
so all three groups land in one bin and the coverage line means something. One
OISST day (2003-07-17) is deliberately never written, which is what exercises
the missing-day path end to end.

### Manifest counts — `--tiers 0 --print`

| source | items | host | registry `expected_items` |
|---|---:|---|---|
| glorys | 384 | cmems | 384 |
| oisst | 44 | ncei | 44 |
| ncep | 560 | psl | 560 |
| rg | **93** | scripps | scraped (null) |
| rapid · osnap · move · samba | 1 each | rapid · gatech · ndbc · aoml | 1 each |
| florida_cable | 43 | aoml | 43 |
| etopo | 1 | ncei_etopo | 1 |
| natural_earth | 2 | github_raw | 2 |

The totals line, as printed:

```
  to fetch: 1131 items · 925.1 GB wire · 199.9 GB stored
  TOTAL 1131 items · 0 with an unverified URL
```

DESIGN §3 says "1,131 items, ~950 GB on the wire (840 of it GLORYS), ~200 GB
written" — the registry agrees to within 3%. With `--offline` the
Roemmich–Gilson scrape falls back to its declared range and the total is
**1,124 items · 920.5 GB**. `glorys_from_mirror` (`enabled: false`) would put
the same 384 months in for **103.5 GB**, taking Tier 0 to about **185 GB**.

`--tiers 0,1,2 --offline`: **8,867 items · 2.0 TB wire · 369.6 GB stored**,
518 flagged `unverified_url`. Tier 1 = 519 items, Tier 2 = 7,224 (ERA5 only;
`cci_sm`, `ostia`, `grep3d` are `enabled: false`).

### Derived Tier-0 wall time — NOT measured

From the registry's per-item counts and each host's `min_gap_s`/`max_lanes`,
with 50 MB/s per lane assumed for HTTP and the **measured 411 s per GLORYS
month** for CMEMS. Nobody has run Tier 0 end to end.

| host | lanes | items | requests | wire | derived total |
|---|---:|---:|---:|---:|---:|
| cmems (GLORYS) | 4 | 384 | 11,688 | 881 GB | **~11 h** |
| psl (NCEP) | 1 | 560 | 1,120 | 23.5 GB | **6.35 h** |
| scripps | 1 | 93 | 186 | 65 GB * | **0.88 h** |
| ncei (OISST) | 6 | 44 | 31,656 | 22.9 GB | **0.75 h** |
| aoml | 1 | 44 | 88 | ~0 | **0.12 h** |
| ncei_etopo | 1 | 1 | 2 | 0.5 GB | **0.01 h** |
| rapid, gatech, ndbc, github_raw | 1–2 | 5 | 10 | ~0 | minutes |

\* over-estimated: the registry gives every Roemmich–Gilson item the base
files' ~0.7 GB. Lanes run in parallel, so the wall clock is the longest lane:
**~11 h with `glorys`, ~6.5 h with `glorys_from_mirror`.**

---

## 3 · Which path was taken for the Example protos

**The pure-Python path, with TensorFlow used only to certify it.**

`pip install tensorflow-cpu` **succeeded** in the venv, in about four minutes
(tensorflow-cpu 2.21.0). But it **upgraded `protobuf` to 7.36.1**, and
apache-beam 2.68.0 pins `protobuf<6.0.0.dev0`; pip printed the conflict as an
error line and installed it anyway. Beam kept working in testing (a
DirectRunner pipeline ran, and the whole test suite passes with TensorFlow
present), but shipping a required dependency that breaks another dependency's
pin is not something to hand to a junior operator.

So `beam_import/example.py` encodes and decodes `tf.train.Example` directly —
it is three nested messages, all varints and length-delimited blocks, about 130
lines — and `beam_import/tfrecord.py` writes the frame with `crcmod`, which
apache-beam already depends on. That was going to be necessary anyway: DESIGN
§4 promises "the pure-Python reader in `beam_import/tfrecord.py` does the same
with no TensorFlow import at all", so the reader existed either way and the
writer is its mirror image.

**The certification is the point.** `tests/test_example.py` and
`tests/test_tfrecord.py` assert, whenever TensorFlow happens to be installed
(`importorskip`), that TensorFlow parses our bytes field for field, that we
parse TensorFlow's, that `tf.data.TFRecordDataset` reads our shards and our
reader reads its shards, and that on a real Stage-B shard both readers produce
the same numbers. Both README §6 snippets were run: they print
`bin 1573 2003-07-15 shape [7, 9, 9] cur_u mean 0.009829164482653141` —
the identical float.

`tensorflow` is **not** in `requirements.txt`. The tests skip their
cross-checks when it is absent and everything else still passes.

---

## 4 · Host HEAD results, measured 2026-09-04

One HEAD per host, from this sandbox, through its egress proxy. Unchanged from
revision 1 except that the Hugging Face row is now about the optional
`glorys_from_mirror` source rather than about a publishing destination.

| host | result |
|---|---|
| `cmems` (`data.marine.copernicus.eu`) | **200**, 329,182 B |
| `psl` (`land.sfc.gauss.nc`) | **200**, 25,213 B, `application/x-netcdf` |
| `ncei` (an OISST day) | **200**, 1,714,749 B |
| `scripps` (`RG_ArgoClim_Temperature_2019.nc.gz`) | **200**, **695,480,508 B** (≈0.7 GB, not the ~2 GB earlier notes estimated) |
| `rapid` (`moc_transports.nc`) | **200**, 1,182,284 B |
| `aoml` (cable 2020 v3) | **200**, 18,607 B |
| `ncei_etopo` | **200**, **no `Content-Length` on HEAD** |
| `github_raw` | **200**, 1,570,533 B |
| `gatech` (OSNAP bitstream) | **200**, no `Content-Length` |
| `ndbc` (MOVE) | **200**, 172,492 B |
| `pmel` | **200**, 799 B — an **index page**, not a file |
| `cds` | **202**, JSON (reachable; no key, nothing else tried) |
| `ceda` | **200** |
| `metoffice` (EN4 2020 zip) | **404** — the pattern is WRONG or has moved |
| `bom` (RMM) | **403** — the host refuses a plain HEAD |
| `cpc` (ONI) | **blocked by the sandbox proxy**; not retried |
| `hf_public` (`huggingface.co`) | reachable (a `whoami` probe returned 401 without a token, i.e. the host answers) |

Every Tier-0 host answered 200. The three problems are all Tier 1, all already
carry `unverified_url: true`, and each has a registry note saying what a human
must check.

---

## 5 · What I could not verify

- **Any credentialed path.** Nothing in this session authenticated anywhere.
  CMEMS `subset` calls, the CDS request body, and the `glorys_from_mirror` URL
  pattern have never been executed against the real services.
- **`rg_months` against a real Roemmich–Gilson file.** It is written against
  the variable names `ml/fetch_rg_channels.py` uses
  (`ARGO_TEMPERATURE_ANOMALY` / `_MEAN`, `PRESSURE`) and tested against a
  synthetic file of that shape. The real base file is 0.7 GB gzipped and was
  not downloaded.
- **The `series` reader on the real label files.** RAPID, MOVE and OSNAP are
  read as NetCDF with a `time` axis; the Florida cable, SAMBA and the indices
  are read as whitespace-numeric tables (year + day-of-year, or year + month +
  day). Neither branch has seen a real file. A file neither branch understands
  is stored **opaque** rather than dropped, so the failure mode is "less
  useful", never "lost".
- **The EN4 URL pattern** (404), **`cpc`** (proxy-blocked), **`bom`** (403),
  **`pmel_wwv`** and **`godas`** (index pages, not files).
- **The DUACS `dataset_id`** and the other CMEMS ids. `copernicusmarine
  describe` was not run — it is an authenticated call and the brief allows one
  probe per host. All are marked `resolve_dataset_id: true`.
- **ERA5's exact CDS request body.** Never sent; treat it as a first draft.
- **Real upstream throughput.** Every wall-time figure is derived (§2).
- **The daily cadence at full scale.** Like the pentad view, it has only been
  run on the fixtures (five bins, 9×9 cells). At Tier-0 scale it is ~15,700
  bins per group rather than ~3,140, so its shard count and memory profile are
  unmeasured — start it with `--num-shards` raised in proportion.
- **Stage B at full scale.** It was run only on the fixtures — one bin, 9×9
  cells. The arithmetic is asserted against `build_family7`'s own tables and
  against directly-computed sigmas, but a 3,139-bin run over ~200 GB has not
  happened and its memory profile is unmeasured.

---

## 6 · Deviations from DESIGN.md, and why

1. **`lat_values` / `lon_values` on every gridded record.** DESIGN §4 lists
   `lat0`, `lat_step`, `nlat` (and the same for longitude). Those cannot
   describe NCEP's gaussian latitudes or Roemmich–Gilson's grid. Both forms are
   written: the triple as specified, plus the explicit axes. A reader that has
   to guess is a reader that will guess wrong.

2. **`grid = opaque` for files that are neither a grid nor a series.** DESIGN
   §4's schema has no place for a GeoJSON coastline, an EN4 zip or ETOPO's
   relief. Those get `grid: opaque`, an empty `values`, and the file verbatim
   in a `raw` feature with its own sha256. It exists so Stage A can promise it
   lost nothing; Stage B ignores them.

3. **NCEP records carry a `<var>_sq` channel for the two stress variables.**
   `tau_*_std` is the within-pentad population sigma over the 6-HOURLY samples,
   and a sigma is not aggregable from a mean — but it is exactly recoverable as
   `sqrt(E[x²] − E[x]²)`, and every day has the same four samples, so the
   pentad mean of the daily square-means is the pentad `E[x²]` exactly. Stage A
   therefore stores the daily mean AND the daily mean of squares, and Stage B
   recovers the sigma. Pinned by
   `test_assemble.py::test_tau_std_is_the_within_pentad_population_sigma`,
   which compares against the raw samples directly.

4. **The stress sign flip happens in Stage B, not Stage A.**
   `build_family7` negates `uflx`/`vflx` per 6-hourly sample before
   accumulating. Negation is linear, so negating the pentad mean once gives the
   identical number, and Stage A stays a pure format conversion.

5. **`skin_t` vs `sst`, and g100's 15th channel.** DESIGN §4 calls g025's
   channel 5 `skin_t`; `build_family7.CHAN_G025` calls it `sst`. DESIGN says
   g100 has 14 channels; `build_family7.CHAN_G100` has 15, the extra one being
   `skt` at `C_SKT = 14`. **The imported tables are the truth** (the brief says
   import the tables rather than restate them), so the code takes the names and
   the count from `build_family7` and treats DESIGN's names as synonyms. The
   fill rule is DESIGN's: OISST's sst where OISST has one, NCEP's skt
   (K → °C) elsewhere.

6. **Fill shards are named by DATE, not by a counter.**
   `<parent>.fill-<YYYYMMDD>.tfrecord`. A counter would need a read-modify-write
   across lanes; the date is deterministic and two lanes filling two days of
   the same month can never collide. The day item keeps its own `.done` marker
   so it is separately resumable.

7. **Two hosts DESIGN §2's table names but does not budget** (`gatech`,
   `ndbc`) get `max_lanes: 1` — the floor — and say so in their `evidence`
   field. A third, `hf_public`, is ADDED for the optional
   `glorys_from_mirror` source at 2 lanes / 1 s, the same budget `github_raw`
   gets for the same reason (a CDN-backed anonymous public read).

8. **`cpc` and `bom` are two host rows**, where DESIGN's table has one
   `cpc / bom` row. They are different domains, so they get one lane each —
   stricter than sharing, not looser.

9. **Total lanes sum to 29**, which is what DESIGN §5 tells a distributed
   runner to size against; DESIGN §2's own prose says "the busy hosts are
   cmems 4 + ncei 6 + psl 1 ≈ 11 streams", which is also true and is the
   number that matters in practice. No budget was changed.

10. **No automatic NCEI→PSL fallback for OISST.** Falling back inside an NCEI
    lane would open a second concurrent stream on PSL without PSL's one-lane
    budget agreeing. The fallback is a separate source, `oisst_psl`, on the
    `psl` host, `enabled: false`, run explicitly.

11. **A `days:` knob on year-chunked sources** — an explicit day list that
    overrides the computed one. Its real use is a partial re-fetch; the
    fixtures use it to make a "year" five days long.

12. **`--dry-run` builds nothing and needs no credentials**, and reports
    `present` with a `reason` naming what would have been fetched (there is no
    `planned` status any more — the four statuses are the whole vocabulary).

13. **The daily cadence adds two features to the PENTAD records too.**
    `cadence` and `cadence_days` are written at every cadence, so a pentad
    shard made today is two features larger than one made before the flag
    existed. That is the one respect in which the default output is not
    byte-identical to the previous version, it was specified by the same
    change (DESIGN §4's feature table), and every value, mask, shape, axis,
    channel name, `days_present` and bin is unchanged — pinned by
    `test_assemble.py::test_the_pentad_default_is_unchanged`.

14. **Code size.** The brief targeted ≤ ~2,500 lines; the package is
    **2,662** lines of code plus a comparable volume of comments. The
    overshoot is `assemble.py` — Stage B has three channel groups with
    different rules and two cadences, and each needed its own path.

---

## 7 · Things a future maintainer should know

- **The four statuses are load-bearing.** Adding a fifth — especially a
  `failed` — breaks the promise DESIGN §2 is built on. If something cannot be
  fetched, it goes on the queue with a reason.
- **`.done` is written LAST and can only under-claim.** A run killed between
  the rename and the marker simply rewrites the shard next time. Pinned by
  `test_sinks.py::test_mark_is_written_after_the_shard_so_it_can_only_underclaim`.
- **Lane assignment is `sha1(item_id) mod max_lanes`, not `hash()`.** Python's
  `hash()` is salted per process, so under a multi-process runner the same item
  would land in different lanes in different workers and the lane cap would
  stop meaning anything.
- **`report.jsonl` is written by Beam only at the end.** The live artefact is
  `<state-dir>/progress/<host>-<lane>.jsonl` — one file per lane, fsynced per
  record, written BEFORE the record is yielded. `report --live` reads it, and
  `summary.md` is the union of those files and `report.jsonl`, deduped by item
  with the final record winning.
- **The circuit breaker counts consecutive failed ITEMS**, each of which has
  already exhausted its own five-attempt ladder. A 404 does not count (it is
  evidence, not a failure of ours).
- **Everything reaching the output goes through `tfrecord.py`, which goes
  through Beam's `FileSystems`.** That is why `--output gs://…` needs no code
  change. If you add a write path, add it there.
- **`--cadence` is a Stage-B parameter, never a Stage-A one.** Stage A always
  writes one record per source-day; changing the cadence re-reads that same
  archive and costs no upstream traffic. If a third cadence is ever wanted,
  add it to `CADENCES` and give it a `DEFAULT_GROUPS` and a `DEFAULT_MIN_DAYS`
  entry — nothing else in the package needs to know.
- **Stage B derives its grids from the records**, using
  `aggregate_cadence.axis_for` — the same function that defined the 0.25°
  point grid, one spacing coarser for the 1° grids. Nothing is hardcoded,
  which is why the tests can run on 9×9 fixtures.
- **Nothing in this package deletes anything it did not write**, and there is
  no code path that could. Keep it that way.
