# Family 1 adapters — the contract every store follows

Read `ml/plans/E082_family1_builds.md` first, then the design note of the
family the store belongs to (`ml/paper/notes/family1gf.tex`, `family1tf.tex`,
`family09tf.tex` — the ledger rows give source, channels, cadence, record,
licence and the note's size estimate). This file fixes the code interface so
adapters can be written in parallel with the framework.

## Where things live

- `ml/build_family1_stores.py` — the CLI. Same stages and flags as
  `ml/build_family10_stores.py` (`--store`, `--stage`, `--start`, `--end`,
  `--work`, `--smoke`, …) plus `--stage probe` / `--probe-month YYYY-MM`.
  It reuses the family-10 machinery (`Ctx`, `PartWriter`, `stage_fetch`,
  `assemble*`, `stage_publish`, `check_store`, `run_smoke`) by import; it
  does not copy it.
- `ml/family1/adapters/<store>.py` — one module per store, exposing
  `ADAPTER` (the class). `ml/family1/adapters/__init__.py` imports every
  module in the directory and builds `REGISTRY = {store: class}`.
- `ml/family1/sharded.py` — the sharded tier-G layout (wave 2).
- `tests/test_family1_<store>.py` — the store's smoke test; the framework's
  own tests are `tests/test_build_family1_stores.py`.

## The adapter class

Subclass `ml.build_family10_stores.SourceAdapter` (import the module as
`f10b`). Everything family 10 defines keeps its meaning (`store`, `title`,
`channels` as `((name, unit, lo, hi), …)`, `log2_fp`, `log2_dt`, `per_year`,
`first_year`, `qc_policy`, `sources`, `verified`, `notes`, `index(ctx)`,
`fetch_year(ctx, year)` yielding row batches, `fetch_stream(ctx)` for
`per_year = False`, the optional hooks). New class attributes:

```python
family = "1tf"            # "1gf" | "1tf" | "09tf" -> tensors/family1_tf/, partials/family1_tf/, ml/cache/family1_tf
distribution = "public"   # "public" | "private"; private publishes to chfrank/earth-tensors-private and NEVER to the public repo
licence = {"name": "CC0", "redistribution": "yes", "derived_works": "free", "attribution": ""}
                          # redistribution: yes | attribution | no  (no  => distribution must be "private"; the framework refuses otherwise)
                          # derived_works: free | non-commercial | restricted
time_dtype = "int32"      # "int32" (schema 2) | "int64" (schema 3); MUST be "int64" when first_year < 1914
platform_meta = False     # True => the adapter's `platforms(ctx)` returns {hash: {...}} written to platforms.json beside the arrays
```

Rows are what family 10 rows are: a dict/tuple batch the framework packs
with `f10b._pack(t, lat, lon, values, platform, qc, C)` — look at
`GDPAdapter.fetch_year` for the shape. `t` is seconds since 1982-01-01 (may
be negative and may exceed int32 when `time_dtype == "int64"`); `lat`,
`lon` float32 with `lon` in [-180, 180); `values` float64 `(n, C)` in raw
units with NaN for not-measured; `platform` int64 from
`f10b.platform_hash(text)`; `qc` uint8 per the adapter's own policy.
Profiles put the vertical axis into `C` (the 16 Roemmich–Gilson pressures
per variable for ocean profiles; the 16 mandatory levels for radiosondes).

## Rules that are not optional

1. **Every adapter ships a smoke** (a synthetic source written by the test,
   exact expected row counts and per-year counts) before its probe runs.
2. **A listing that comes back empty or a download that comes back short is
   a refusal**, counted in `degraded`, never a skip (ml/CLAUDE.md, the
   2026-09-14 rule; `f10b.fetch_first` / `http_to_file` already do this).
3. **Values outside the channel's `[lo, hi]` become NaN and are counted**,
   never clipped.
4. **Nothing is hard-coded that the archive can tell you**: station lists,
   file names and years come from the archive's own listing at `index`
   time; a URL pattern is a hypothesis the index verifies.
5. **Credentials come from the environment only** (`EARTHDATA_USERNAME`,
   `FIRMS_MAP_KEY`, …); an adapter that needs one declares
   `credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")` and the
   framework refuses at preflight if any is unset.
6. **Measure, then estimate.** The probe report is the number; the note's
   estimate is what it replaces.

## Additions from the framework build (E-082, 2026-09-17)

Everything above still holds. These are ADDITIONS — optional hooks, one
helper, and one naming convention the probe relies on.

- **Packing a schema-3 row.** `f10b._pack(...)` now takes
  `time_dtype="int32"` as an eighth keyword. An adapter with
  `time_dtype = "int64"` either calls `self.pack(t, lat, lon, values,
  platform, qc)` (C and the dtype filled in) or passes
  `time_dtype=self.time_dtype`. The int32 default still REFUSES a second
  before 1913-12-13 or after 2050-01-19; `PartWriter` widens int32 rows to
  int64 for an int64 store, so a record that never leaves int32's range works
  either way. `bin` stays int16 (1533 .. 2430).
- **Out of bounds, counted by name.** `self.mask_bounds(values)` sets every
  value outside its channel's `[lo, hi]` to NaN IN PLACE and returns
  `{channel_name: n}`. Put that dict under `counts["out_of_bounds"]` — the
  probe reports it per channel, and separately counts any stored value that is
  still out of bounds (`out_of_bounds_stored`, which must be 0; the probe
  exits non-zero otherwise).
- **`fetch_month(ctx, year, month)` — optional, for the probe.** Yields
  `(label, rows, counts)` for ONE calendar month. The default (on
  `SourceAdapter`) runs `fetch_year` (or `fetch_stream`) and keeps the rows
  whose `time_s` lies in the month — always correct, costs the whole year, and
  its counts describe the year (`fetch_month_scope = "year"`). Override it when
  the source can be asked for one month, or is ordered so the adapter can stop
  after it (then set `fetch_month_scope = "month"` and record the stop in
  `counts`, as `ghcnd` does with `probe_stopped_early`). The probe measures
  `ctx.bytes_fetched` around this call, so an override that stops early gives
  an honest bytes-per-row.
- **Bytes.** `f10b.http_bytes`, `f10b.http_to_file` and the new
  `f10b.CountingStream(url)` (a file-like GET that counts every byte and
  RAISES if the body ends short of Content-Length) add to the counter
  `ctx.bytes_fetched` reads. An adapter reading a `--source-dir` file calls
  `ctx.count_bytes(n)` for what the network would have carried; the smoke
  asserts the probe counted more than zero.
- **`platforms(ctx)`** (when `platform_meta = True`) returns
  `{platform_hash: {"id": ..., "lat": ..., "lon": ..., ...}}` for AT LEAST
  every platform the store can hold; the framework keeps only the hashes that
  occur in `platform.npy`, writes them as `platforms.json` (keys as decimal
  strings, sorted) beside the arrays, puts the file in store.json's sha256
  block, and records `platforms: {entries, source_entries, in_store,
  in_store_without_entry}` in store.json.
- **The smoke hooks.** `smoke_sources(self, root, d_lo, d_hi)` writes the
  synthetic archive under `root` in the source's real on-disk layout (the
  adapter reads it when `ctx.source_dir` is set) and returns the truth rows
  `f10b.check_smoke` compares against — dicts with `t` (int seconds), `lat`,
  `lon`, `platform`, `v` (C floats, NaN = absent). `smoke_window = (start,
  end)` and `smoke_probe_month = "YYYY-MM"` choose the window and the month
  `python3 ml/build_family1_stores.py --store <s> --smoke` probes; the probe's
  row count must equal the truth's rows in that month. The truth must be
  ordered the way the store breaks ties among equal `(bin, time_s)`: the order
  the adapter yields them in (`ghcnd` sorts its synthetic stations by id for
  exactly this reason).
- **`credentials`** is checked by `build_family1_stores.credentials_preflight`
  before the index, the fetch and the probe — except with `--source-dir`, and
  except on a `--parts-from-hub` box, which never reads the source.
- **Store-level keys.** store.json carries `family` (`family1_tf`, …),
  `family_code` (the adapter's `1tf`), `family_version` (`1.0.tf`, …),
  `distribution`, `licence`, and on an int64 store `schema_version: 3` and
  `time_dtype: "int64"`. (The existing `schema` key is the per-file dtype
  table, so the schema NUMBER is `schema_version`, as in family 10.)

## Tier G — the sharded gridded layout (E-082 wave 2, 2026-09-17)

A gridded store finer than 0.25° subclasses `family1.sharded.GridAdapter`
(itself a `SourceAdapter`) instead of producing rows. The layout is
`ml/family1/sharded.py`'s docstring; `seaice_asi` is the worked example.

```python
tier = "G"; layout = "sharded"   # set by GridAdapter
frames_per_bin = 5               # F; F * frame_seconds must be 432,000; F <= 64
frame_seconds = 86400
dtype = "float16"                # or "uint8" (255 = missing, 0..254 stored)
tile = 256; zstd_level = 15
grid = {...}                     # one group named after the store, OR
grids = {"n": {...}, "s": {...}} # several; each JSON-able: H, W, x0, dx, y0,
                                 # dy, crs / projection text, row_order
def grid_latlon(self, group, x, y): ...   # optional: tile-corner lat/lon
def record_frames(self, ctx, group): ...  # optional: the probe's extrapolation
def fetch_frames(self, ctx, wanted):      # wanted = [(group, bin, frame), ...]
    yield group, bin, frame, array_or_None, counts
note_estimate = {"bytes": ..., "what": ...}
```

- `array` is `[H, W, C]` (or `[H, W]` when C = 1), raw units, NaN for
  missing, bounds already masked (`self.mask_bounds`). Row 0 is whatever the
  grid's `row_order` says — state it.
- `None` is a frame that is not in the source; `counts["frame_missing"]`
  names why (`absent_upstream`, `before_record`, `after_record`). It is
  stored as a zero-length frame and counted by reason in store.json. A bin
  whose frames are ALL `before_record`/`after_record` is not written.
- An input that could not be READ is `ctx.note_absent(...)` and nothing is
  yielded for it — the year is then not marked. A frame that is neither
  yielded nor noted absent is an adapter bug and the fetch refuses.
- The framework's `fetch_year` hands `fetch_frames` every frame of the
  year's bins, where a year is the bins whose FIRST day falls in it
  (`--start/--end` choose bins, not days). `ctx.grid_specs`, `ctx.grid_bins`
  and `ctx.grid_wanted(year)` are set by `prepare_grid_ctx`.
- The smoke hook `smoke_sources(root, d_lo, d_hi)` returns
  `{(group, bin, frame): (stored-dtype array [H, W, C] or None, reason or
  None)}` for every frame of every bin overlapping the record; `--smoke`
  reads every frame back tile by tile and compares exactly.
- A licence the producer has not confirmed carries
  `"redistribution_confirmed": False`; a PUBLIC publish or parts push is then
  refused unless `--allow-unconfirmed-licence` is passed.
