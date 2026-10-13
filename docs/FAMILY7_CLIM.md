# The **Model climatology** layer — what the forecaster calls normal

Every other layer in this app shows the world, or (the "Global tensor" layer)
what the forecaster *reads*. This one shows what the forecaster calls
**normal**: the per-calendar-month average it is trained against and scored
against, for every channel of family 7, on the globe and as files you can
download.

Switch it on in the layer list ("Model climatology — what the forecaster calls
normal (family 7), 0.25° / 1°"), pick a **channel** in the row underneath, and
move the date by months. The layer paints the **all-years normal**
(1982–2024). Each frame is **one HTTP range read** of a file on the Hugging
Face Hub.

**For any other period, use the Data tab.** The layer used to offer three
fixed versions (all years, a development holdout, the paper's split); they
were retired on 2026-10-06 in favour of a free period (Chris: *"No 'paper
holdout' or similar anymore, just setting a period is enough"*). The Data
tab's **Monthly normals** stores (under *Derived maps*, one per channel group)
average over any years you choose, from per-year monthly sums and counts
([E-086](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E086_monthly_by_year.md)):

- **the paper's split** — period **1982 to 2020**, *leave out* **2009, 2017**;
- the development holdout — 1982 to 2024, leave out 2009, 2017, 2023;
- a WMO-style normal — 1991 to 2020; or a single year, which gives that
  year's own monthly mean.

Those reproduce the retired versions' files: E-086's falsifier rebuilt every
one of them from the sums and counts, every cell, within a rigorous float32
rounding bound (max 4.8 × 10⁻⁷ in z-units). The row's **⤓ downloads** fold
has a button that opens the tab on the group of the channel on screen.

The plan behind it is
[E-083](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E083_model_climatology.md);
the tensor it is computed from is described in
[docs/FAMILY7_GLOBE.md](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY7_GLOBE.md).

---

## What you are looking at

- **Family 7** is the global input tensor: every 0.25° grid point from pole to
  pole, one value per channel per five-day bin ("pentad"), 1982–2024. Its
  channels come in groups, each on its own native grid — 0.25° for the ocean
  surface and ocean colour, 1° for the atmosphere-and-land channels and the
  Argo depth column.
- **The climatology** is, for each channel, cell and calendar month, the mean
  over every *training-year* pentad that opens in that month. It is computed
  by the one function that does this for training,
  `ml/trainprobe.py::anomaly_transform`, and never by a second implementation.
  The model is handed the **departure** from this field (then z-scored), and
  the rollout skill `msss_clim` — "how much better than saying *normal*" —
  scores every forecast against the same field.
- **The month of the date selector is the whole of the date.** January 2010
  and January 2023 show the same map; the day and the year do nothing. The
  ±1 month stepper walks the seasonal cycle, and the Play tab animates it as a
  twelve-frame loop.
- **Values are shown in the channel's own unit** (°C, m/s, hPa …). The files
  store them *z-scored*, like the tensor: physical = z × sd + mean, with the
  per-channel (mean, sd) the tensor builder recorded. The probe prints both —
  the normal in its unit and `z = …` as stored — and uses the same colours
  and legend range as the Global tensor layer, so one channel looks identical
  on the two layers.
- **A blank cell means "no training sample"** for that month at that cell
  (NaN in the file) — for example ocean-only channels over land.

### The departure from normal

With the **Global tensor** layer also on (so that pentad's bytes are already
in the page), the probe and the pixel card add a second row: **departure from
normal** — the tensor's value minus the normal, in the unit, and as the
trainer's own z-score `(value − normal − mu) / den`, with `mu` and `den` read
from the all-years `stats.json`. This is the number the model is actually
handed. The trainer charges a pentad to the calendar month it *opens* in, so
the departure uses that month's normal even when the selected date is in the
next month.

The departure is never a reason to download the tensor: without the Global
tensor's pentad already in memory, a click prints the normal and a one-line
hint instead of fetching 14.5 MB.

---

## The three versions (retired 2026-10-06)

Which years count as "training" decides the climatology. Until 2026-10-06 the
row offered three versions, three answers to that question computed by the
same function with a different hold-out mask:

| version | training years | the same normal in the Data tab |
|---|---|---|
| `all` — **All years (1982–2024)**, what the layer paints | every year | period 1982–2024 |
| `dev` — development holdout | every year except 2009, 2017, 2023 | 1982–2024, leave out 2009, 2017, 2023 |
| `paper` — the paper's split | 1982–2020 except 2009 and 2017 | 1982–2020, leave out 2009, 2017 |

The selector is gone; the `dev` and `paper` files stay on the Hub and in the
index (the downloads fold links only `all`'s), and anything else is a period
in the Data tab. The layer's numbers are the `all` file's, unchanged.

---

## How the bytes are addressed

Per (version, group) there is one `clim.npy` on the Hub under
`tensors/<stem>/clim/<version>/<group>/`, shaped **`[12, C, H, W]` float32,
C order** — month-major, then channel-major. So one (month, channel) plane is
one contiguous run of bytes:

```
offset = header_len + (month × C + c) × plane_bytes
length = plane_bytes  =  H × W × 4
```

`Range: bytes=<offset>-<offset+length-1>`, and the answer must be **HTTP 206**;
a 200 (the host ignoring the range and sending the whole file) is refused.
That is **4.15 MB** for a 0.25° plane (721 × 1440 × 4) and **0.26 MB** at 1°
(181 × 360 × 4).

Because the C planes of one month sit side by side, a whole month of a group is
*also* one contiguous range (`header_len + month × C × plane_bytes`,
`C × plane_bytes` long). When that is small — every 1° group, well under
16 MB — the page reads the month whole, so switching channel within it costs
no request. A 0.25° month (7 × 4.15 MB = 29 MB) is not read whole: there a
channel switch is one more 4 MB plane. Either way a change of channel or month
is a decode or **one** read, never more.

Every number in this arithmetic — `header_len`, `shape`, `dtype`,
`plane_bytes`, the grid, the channel names, labels, units, colour ramps and
(mean, sd) — comes from **`data/family7_clim_index.json`**, which is why the
app contains no 721, no 1440, no 12 and no channel name. Decoded planes are
kept in a 24-plane cache keyed by (version, group, month, channel), so stepping
back to a month already seen is free; a held date key is coalesced
through `scrubApply` into one read per *settled* date.

The index's channel metadata and `norm` are **copied** from
`data/family7_index.json`, and the index carries that tensor's `stem`; the
publisher refuses to write one against a different tensor.

---

## Downloads

The layer's row has a **⤓ downloads** fold (the hover card lists the same
links), for the all-years normal of the selected channel's group, and a button
that opens the Data tab on that group's **Monthly normals** store for any
other period:

- **`clim.nc`** — the same field as CF-conventional NetCDF, one variable per
  channel **in physical units**, dimensions (month, lat, lon). Where it was
  not published (the in-repo fixture ships none) the page says "NetCDF not
  published for this group".
- **`clim.npy`** — `[12, C, H, W]` float32, z-scored — see `stats.json`.
- **`stats.json`** — the z-score constants `anomaly_transform` applied after
  subtracting the climatology (`mu`, `sd`, `den`, `dynamic`), the channel list
  and `norm`, the version's rule and held-out years, the number of training
  bins, the tensor's sha256 per group, the exporter's git sha.
- **the index** — `data/family7_clim_index.json`, every version and group with
  URLs, byte counts and sha256.
- **"this channel, this month as CSV"** — built in the browser from the plane
  already in memory: `lat,lon,value` in physical units, south to north, one
  row per grid point, an empty value where there is no training sample. Named
  `family7_clim_<version>_<group>_<channel>_m<MM>.csv`.

---

## What is NOT shown, and why

- **No aggregation window and no computed difference.** The field is already a
  multi-decade average, one calendar month per frame; averaging it over an
  N-day window would blend two months' normals into a number no trainer ever
  subtracted, and a per-pixel difference between two dates is the seasonal
  cycle, which the ±1 month stepper already shows.
- **No catalog record.** Root `CLAUDE.md` §2.6 catalogues open *datasets*; this
  is a picture of our own work, the same exception the Global tensor layer and
  the AMOC eval mask take. Its title links the plan instead.

---

## Regenerating it

On a rented box with the tensor's disk and memory (E-083 §3):

```bash
python3 ml/pull_family7_tensor.py --dest <dir>            # parallel, resumable range reads, sha256-checked
python3 ml/export_family7_clim.py --tensor <dir>/<stem>.npz \
    --index data/family7_index.json --out <out>           # clim.npy + clim.nc + stats.json per (version, group)
python3 ml/publish_family7_clim_index.py upload --out <out>
#   uploads every file, downloads each BACK and checks its sha256, measures CORS
#   with Origin: https://blauewelt.github.io, then writes data/family7_clim_index.json
python3 scripts/stamp_assets.py                           # the deploy is invisible without it
```

Until `data/family7_clim_index.json` exists the layer stays switchable and
paints nothing, and a toast names this chain. The moment the file lands beside
the page the layer is live, with no code change.

### The in-repo fixture

`data/family7_clim/fixture/` holds the same schema, written by
`python3 ml/export_family7_clim.py --fixture` from the family-7 smoke fixture
(`data/family7/fixture/`, decimated to 5° / 10°): the `g100` and `oc025`
groups in all three versions, without `clim.nc`. The smoke tensor covers five
pentads of January 2010, so only January is finite and the three versions are
byte-identical. `tests/app.spec.js` routes the Hub URLs to it and answers with
the sliced bytes and a real 206, so the browser's offset arithmetic is tested
rather than assumed; `tests/data.spec.js` pins the index against the files and
against the family-7 fixture index (same stem, channels, grids and norms).

## Beside it: the **Pentad climatology** layer — the exact five-day climatology (E-093)

Chris, 2026-10-13: *"Can you make sure that climatology can also be displayed
as a layer? (This should be the exact pentad averaged over years data)"*. The
layer **"Pentad climatology — each five-day bin averaged over the years
(family 7), 0.25° / 1°"** (`poy7`) is the five-day counterpart of the layer
above: not one calendar month per frame but one of the 73 five-day slots of
the year, averaged over a span of years you choose.

**Which slot.** Family 7 is a continuous run of five-day bins from 1982-01-01
(bin *b* covers days [5b, 5b+5)). The globe's date falls in one bin; the bin
belongs to slot `min(72, floor((day_of_year(bin start) − 1) / 5))` — slot 0 is
the bins opening on days 1–5, slot 72 those opening on days 361–366. Because
365 is a multiple of five, the bins keep their place against the calendar in
ordinary years and slide by one day at every leap day, so the same calendar
date can fall one slot over in another year. The year of the date never
chooses which years are averaged; the toast says so.

**Which years.** The row has a channel picker, a *years* pair (default: the
group's whole record) and a *leave out* field ("2009, 2017"). The mean is
composed in the browser from E-093's files (`data/gridded_poy_index.json`,
plan [`ml/plans/E093_pentad_climatology.md`](../ml/plans/E093_pentad_climatology.md)):
per group and slot, `sum_pSS.npy` float32 `[C, Y, H, W]` in physical units and
`count_pSS.npy` uint8, both cumulative over the years, so

    mean over a…b = (S[b] − S[a−1]) / (N[b] − N[a−1])

and a year *e* left out subtracts `(S[e] − S[e−1])` and `(N[e] − N[e−1])`.
Each year-plane is one `Range:` read at
`header_len + (c·Y + y)·H·W·itemsize` — 4.15 MB of sums plus 1.04 MB of counts
at 0.25°, 0.26 + 0.07 MB at 1°. The whole record is two reads; a span starting
later is four; each year left out adds up to four. Planes are cached per
(group, slot, channel, kind, year-plane), so a date that stays in its slot, and
a span change that reuses planes, read nothing new. A cell with count 0 is
blank.

**Read-outs.** The probe and the pixel card print the mean in the channel's
unit, the number of five-day bins behind it at that cell, the span and the
slot; the stamp is the span (a fixed period, no age). Labels, units, ramps and
legend ranges come from `data/family7_index.json`, so a channel looks the same
here as on the Global tensor and the Model climatology layers. The ⤓ downloads
fold links the slot's two `.npy` files and the index, and opens the Data tab on
the same group (`10/<group>`) at the step *Climatology — mean per pentad of
year over the years* — the same numbers as a file, for any box.

No aggregation window and no computed difference: the field is already a
multi-year mean of one five-day slot (its five days ARE its window), a
difference of two dates is the seasonal cycle the ±5-day stepper walks, and
the comparison that means something — one span of years against another — is
the row's own years and leave-out fields. Tests: `tests/app.spec.js`
"Pentad climatology", over `data/poy_layer_fixture/`
(`tests/make_poy_layer_fixture.py`).
