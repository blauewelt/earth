# The **Global tensor** layer — the model's own input, on the globe

Every other layer in this app is an observation of the world. This one is a
picture of **what our forecaster reads**: family 7, the first input tensor
covering the whole globe rather than the North Atlantic window — every 0.25°
grid point from the South Pole to the North Pole, one value per channel per
five-day bin, 1982 to 2024.

Switch it on in the layer list ("Global tensor — what the model reads
(family 7), 0.25° / 1°"), pick a channel in the row underneath, and move the
date. Each frame is **one HTTP range read** of a 46-gigabyte archive.

This page says what you are looking at, where the bytes come from, exactly how
the read is addressed, how to make the index again, and what the layer
deliberately does not show.

---

## What you are looking at

A few words first, because none of them are ordinary English.

- **A pentad** is a fixed five-day bin. The tensor has one frame per pentad,
  counted from 1982-01-01, so bin = ⌊days since then ÷ 5⌋ and it holds bins
  0–3141. Bin *b* covers days [5b, 5b+5), and every date the layer prints is
  the day its bin **opens** on. Four days in five therefore change nothing —
  the toast on enable names the bin you are looking at, and the ±1 day stepper
  will move four times before the picture does.
- **A value is a five-day mean**, not a snapshot. "Sea-surface temperature on
  2015-01-03" here means the average of 2015-01-03 → 01-07.
- **Land cells carry the shared channels.** This is the whole reason family 7
  exists. The North Atlantic tensor told a land cell "the world was not
  observed here" in every channel; family 7 carries a set of channels that
  exist over both land and water — 2 m air temperature, 10 m wind, surface
  pressure, precipitation, snow water equivalent, soil moisture and
  temperature, the two turbulent heat fluxes, and the reanalysis **skin
  temperature `skt`** — so the Sahara and the Southern Ocean are both
  observed, in different channels.
- **Two temperatures, and they are not the same measurement** (E-071 §6.1,
  corrected 2026-09-04). `sst` is OISST: an *observed* sea-surface temperature
  at 0.25°, and NaN over land, which is the honest answer rather than a gap.
  `skt` is the NCEP reanalysis skin temperature at 1°: *every* surface — land,
  ice and sea alike. A land cell reads its temperature from `skt`, `t2m` and
  `tsoil`, never from `sst`, which has nothing to say there.
- **Two channel groups, at their own resolutions.** `g025` is 0.25° and holds
  the seven ocean-surface channels (current speed, the two current components,
  mixed-layer depth, sea-surface height, `sst`, sea ice). `g100` is 1° and
  holds the fifteen atmospheric and land-surface channels — `skt` among them —
  at the
  native resolution of the reanalysis they come from — upsampling a 1.9°
  product to 0.25° would be sixty copies of every number.
- **The values are stored z-scored.** The builder's last pass writes
  (x − mean) / sd per channel, so the number in the archive is in *standard
  deviations*. The layer multiplies it back — `raw = z·sd + mean`, using the
  (mean, sd) the builder recorded — before it paints or prints anything. The
  probe shows both: the value in the channel's own unit, and `z = …` as the
  tensor stores it, because the model reads the second and you think in the
  first.
- **Two statics, not channels.** `sphere` (0 ocean · 1 land · 2 ice sheet ·
  3 inland water) and `elev` (metres, negative under the sea) describe the
  grid rather than a date. They are in the channel picker because that is
  where you look for them, but they ignore the date selector and say so, and
  they are read from two small baked files rather than from the archive.
- **The ocean channels start in 1993**, where the GLORYS reanalysis does; the
  shared channels reach back to the 1982 epoch. Ask for 1985 and the ocean
  channels are empty — that is data truth, not a broken layer.

---

## Where the bytes come from

The tensor is built by `ml/build_family7.py` (recipe `f7l0`, specified in
[E-070](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md))
and published to the Hugging Face Hub, in the dataset repo
`chfrank/earth-tensors` under `tensors/family7_global025_pentad_l0/`:

| file | what | size |
|---|---|---|
| `…_X_g025.npy` | `[3142, 721, 1440, 7]` float16, C order | 46 GB |
| `…_X_g100.npy` | `[3142, 181, 360, 15]` float16, C order | 6.1 GB |
| `…_X_rg100.npy` | `[n_live, 181, 360, 32]` float16 — the Argo depth column | ~1 GB |
| `….npz` | meta: axes, channel names, norms, statics, truth | small |
| `manifest.json` | name, bytes, sha256 per file | tiny |

`huggingface.co` is the second live endpoint this app is allowed to touch
(root `CLAUDE.md` §3), and this use clears the same bar the daily-SST read was
admitted on: **no key**, **CORS measured** with our own `Origin` and recorded
in the index, **bounded reads triggered by a click or a date step** rather than
streaming, and a **degrade path** — every failure ends in a hint toast and an
empty layer, never a broken globe.

---

## The arithmetic, exactly

The three arrays are stored **bin-major in C order**, so one pentad of one
group is a single contiguous slab and the whole layer is one range request:

```
row    = bin − bin_first            (0 for the real tensor; the fixture is
                                     five bins out of the middle of the record)
offset = header_len + row × slab_bytes
length = slab_bytes  =  H × W × C × 2
```

`Range: bytes=<offset>-<offset+length-1>`, and the response must be **HTTP
206**. A 200 means the host ignored the Range and is sending the whole 46 GB
file; the reader refuses it rather than consuming it.

That is **14,535,360 bytes** for a 0.25° frame (721 × 1440 × 7 × 2) and
**1,954,800** for a 1° one (181 × 360 × 15 × 2).

`header_len` is the one number that cannot be guessed — a `.npy` header is a
Python-literal dict padded to a 64-byte boundary, so it is 128 bytes for these
files but is *read from the file* rather than assumed. It, the shape, the
dtype and the slab size all come out of `data/family7_index.json`, which is
why there is no 721, no 1440 and no 128 anywhere in `src/app.js`.

**Decoding.** The slab is float16; `Float16Array` is not available in the
browsers this app supports, so the reader converts each `Uint16` by hand
(sign · 2^(e−15) · (1 + f/1024), with the subnormal and NaN cases spelled out —
NaN is how the builder says "never observed", and a NaN that reached the colour
ramp would paint the palette's top colour on an unobserved cell).

**Caching.** The LRU holds **raw slabs, not decoded planes** — the bytes are
what the network paid for, and every channel of a bin lives in the same
14.5 MB. So switching channel inside one pentad is a decode and no request, and
the ±1 pentad steppers walk a cache of eight per group. The ceiling is eight
slabs per group *and* 128 MB in total; per-group counting matters because a run
of cheap 1.8 MB coarse reads must not evict the 14.5 MB fine slab the layer is
painting from.

**The date scrub is coalesced.** A held-down date key would otherwise be one
14.5 MB read per keystroke, so the refresh goes through `scrubApply` — one
generation in flight at a time, newest pending date applied when the globe
reports its tiles settled (`docs/TILE_BUDGET.md`). A held key costs one read
per *settled* date.

---

## The cone, live

The [Cones tab](https://blauewelt.github.io/earth/docs.html?f=docs/CONE_DATA_DEMO.md)'s
Data mode has a second source: **live from the global tensor**. It reads the
same slabs through the same cache, so moving the layer's date and moving the
cone's anchor pay for each other's fetches.

What it gains: **any cell on the planet is an anchor** — Antarctica, the
Sahara, the middle of the Pacific — and the dots **wrap across the dateline**,
because on a globe there is no eastern edge. What it gives up, and the hint in
the tab says all of it out loud:

- **No anomaly.** The anomaly is a harmonic climatology fitted over the whole
  tensor on training years only (`ml/trainprobe.py`), not something seven
  pentads can produce. Raw measurements only.
- **No depth column.** The 32 Roemmich–Gilson channels are written once per
  month, into the pentad that holds the 15th, so six lags in seven would be
  empty.
- **The outer stencil is geometry, not values.** Stage 2's rings run to lag
  143; one slab per lag would be about two gigabytes.
- **A coarse channel is read at its coarse cell**, so the nine cells of the
  3×3 patch all show the same 1° number. That is not the drawing rounding —
  it is the model's own view of a coarse channel.

The dots' positions come from the same JS port of `ml/cone.py` the geometry
mode uses, and that port is certified against Python's own reference dot sets —
including the global block's wrapped ones — in `tests/data.spec.js`.

---

## Exported family-7 anchors

The Cones tab's Data mode has a **third** source, between the two above:
**exported anchors (family 7, global — 12 cells)**. These are pre-sampled files
rather than a live read, so they carry everything live mode has to give up —
above all the **anomaly** — while still covering the whole globe.

The twelve anchors, in the index's order:

| id | what it is |
| --- | --- |
| `acc` | Antarctic Circumpolar Current, 55 °S 60 °E |
| `antarctica_ice` | Antarctic ice sheet, 80 °S 60 °E |
| `dateline` | the tensor's last column, 0 °N 179.75 °E |
| `equator` | Equator, 0 °N 30 °W |
| `greenland` | Greenland ice sheet, 72 °N 40 °W |
| `gulf_stream` | Gulf Stream, 36 °N 70 °W |
| `ionian_edge` | Ionian Sea — the North Atlantic window's east edge, 36 °N 19 °E |
| `kuroshio` | Kuroshio, 35 °N 145 °E |
| `labrador` | Labrador Sea, 58 °N 52 °W |
| `nino34` | Niño 3.4, 0 °N 150 °W |
| `rapid` | the RAPID array's latitude, 26.5 °N 70 °W |
| `sahara` | Sahara, 23 °N 10 °E |

The index is **`data/cone_samples_f7.json`** (anchors, dates, URLs, sizes, the
tensor's sha256, the exporter's commit, the grid block and the three channel
groups); the files themselves are ~5 MB each on the Hugging Face Hub under
`cone_samples_f7/`, read one anchor at a time. `data/cone_samples_f7/fixture.json`
is the trimmed in-repo copy the browser tests run against with no network — the
dateline anchor cut to two dates and five channels — one per group, a second
ocean one, and one channel of every **cone family** the Cones tab can select
(`cur_speed` B, `sst` and `skt` C, `tau_x` A, `rg_t10` the depth column), so a
browser test can check that the family select really does follow the channel.
Rebuilt from a downloaded anchor with:

```bash
python3 ml/export_cone_sample.py \
    --trim-file dateline.json \
    --fixture data/cone_samples_f7/fixture.json \
    --fixture-channels 0,5,7,21,22 --fixture-dates 2
```

The trim also **recomputes `meta.units`**. Every anchor on the Hub was exported
while the exporter's unit table knew only family 4's ten surface channels and
gave everything else the Argo depth column's composite string, so those files
say air temperature at 2 m is measured in `dbar-level (°C / PSU)`. The exporter
now takes its vocabulary from `ml/publish_family7_index.py::CHANNELS` — the
same table this index publishes — and the page reads that index for a family-7
sample rather than the file's own `meta.units`, so a reader gets `°C` without
re-exporting twelve 5 MB anchors off a 46 GB tensor.

**They come out of the production sampler, not out of this page.** Each file is
written by `ml/export_cone_sample.py --tensor …` — the same script that made
the North Atlantic set, calling `ml/cone_sampler.py::ConeSampler.sample` for the
codec's inner stencil and `ml/cone.py::outer_spiral` for stage 2's outer one —
run on the GPU box by
[`.github/workflows/family7-export-cones.yml`](https://github.com/blauewelt/earth/blob/main/.github/workflows/family7-export-cones.yml)
and uploaded with `ml/upload_cone_samples.py`, which also writes the index.

Three things to know when reading this mode:

- **The anomaly IS available here**, unlike live mode. The exporter computed it
  with the trainer's own function — `ml/trainprobe.py::anomaly_transform`, called
  on a writable copy of each group: departure from a per-calendar-month
  climatology built on **training years only**, then z-scored per channel over
  the training pool. Each group is transformed with its own calendar, because
  `rg100` holds one row per month and the master's would be the wrong one.
- **The `rg_*` channels are populated only in month-holding pentads.** The 32
  Roemmich–Gilson depth channels are written once per month, into the pentad
  that contains the 15th, so on every other date their dots are dimmed — a
  *missing* token, which is exactly what the model is handed there.
- **The raw values are z-scores, per group.** `meta.value_space.tensor_norm` is
  the npz's own `norm_g025` / `norm_g100` / `norm_rg100`, keyed by group and
  indexed *within* the group; `meta.channel_group` says which group a channel is
  in and `meta.groups.channels[g]` is that group's order. The read-out puts the
  unit back with that two-hop lookup and prints the stored σ beside it.

---

## What is NOT shown, and why

- **`rg100`, the Argo depth column** (32 channels, 10–1900 dbar). Live bins
  only: a globe layer that is blank on four dates in five reads as broken.
- **No aggregation window and no computed difference.** The fields are
  continuous and would average and difference perfectly well; the reason is the
  byte count. Each frame is one 14.5 MB read, so a 12-day window would be three
  of them per paint and a 365-day one would be most of the archive; a computed
  difference doubles whatever the window costs. The question this layer answers
  — "what does the model read at this pentad" — has no window in it.
- **No catalog record.** `CLAUDE.md` §2.6 catalogues open *datasets*; this is a
  picture of our own tensor, the same exception the AMOC eval mask and the city
  labels take. Its documentation link points at the build spec instead.
- **The truth series** (RAPID and the two pentad labels) are in the meta npz
  and are not a map.

---

## Regenerating the index

`data/family7_index.json` is written by `ml/publish_family7_index.py`, **after**
the build job has published. It reads `manifest.json` and the meta npz from the
Hub, parses each `.npy` header for `header_len` / shape / dtype by a **range
read** (the file is never downloaded to be understood), verifies every
published file's sha256 by downloading it back, measures the CORS headers with
`Origin: https://blauewelt.github.io` exactly as `ml/upload_cone_samples.py`
does, and writes the index plus the two static grids:

```bash
python3 ml/publish_family7_index.py
#   data/family7_index.json     the index (urls, bytes, sha256, header_len,
#                               shape, dtype, chans, labels, units, norm,
#                               grids, bin range, epoch, recipe, cors_measured)
#   data/family7_sphere.json    classGrid, packed, palette in the file
#   data/family7_elev.json      regular grid, int16 metres
python3 scripts/stamp_assets.py     # the deploy is invisible without it
```

`--trust-manifest` skips the 100 GB re-download when the build job has already
restore-verified in the same session; the index records which kind it is
(`restore_verified`), because a reader should be able to tell an index that
checked the bytes from one that believed a claim.

### The in-repo fixture

Until the real files exist, `data/family7/fixture/` holds the same schema over
the T=5 tensor `ml/build_family7.py --smoke` writes in about twenty seconds,
**decimated to a 5° / 10° grid** so it can live in git (one 0.25° bin is
14.5 MB). The decimation lands exactly on point-aligned 5° and 10° grids that
are still south-first and still wrap, so every piece of arithmetic the browser
does — the offset, the float16 decode, the un-z-scoring, the cell bounds, the
dateline wrap — runs identically on it:

```bash
python3 ml/build_family7.py --smoke --smoke-dir /tmp/f7fix
python3 ml/publish_family7_index.py --fixture /tmp/f7fix/work
```

The fixture index carries the **real Hub URLs**, so the browser code path is
the production one; `tests/app.spec.js` routes those URLs to the local files
and answers with the *sliced* bytes and a real 206, which is what makes the
offset arithmetic genuinely tested rather than hidden behind a whole-file 200.

---

## Orientation — the mistake that looks fine

The tensor's rows run **south-first** (`lats = −90 + 0.25·arange(721)`), and
this app's grid convention is row-major from the south, so there is **no
flip**. That sentence is worth nothing on its own: the AMOC eval mask once
shipped with a `[::-1]` row flip copied from another bake, which put the Gulf
Stream at the latitude of the Norwegian Sea and still looked like a perfectly
plausible ocean. So `tests/data.spec.js` reads a known cell out of the
fixture's own bytes and asserts the equator is warmer than the southern edge.
A picture of a model is exactly the artefact whose mistakes look fine.
