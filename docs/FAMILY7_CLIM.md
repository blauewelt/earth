# The **Model climatology** layer — what the forecaster calls normal

Every other layer in this app shows the world, or (the "Global tensor" layer)
what the forecaster *reads*. This one shows what the forecaster calls
**normal**: the per-calendar-month average it is trained against and scored
against, for every channel of family 7, on the globe and as files you can
download.

Switch it on in the layer list ("Model climatology — what the forecaster calls
normal (family 7), 0.25° / 1°"), pick a **channel** and a **version** in the
row underneath, and move the date by months. Each frame is **one HTTP range
read** of a file on the Hugging Face Hub.

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
  (NaN in the file) — for example ocean-only channels over land, or a group
  whose record starts after the version's training years.

### The departure from normal

With the **Global tensor** layer also on (so that pentad's bytes are already
in the page), the probe and the pixel card add a second row: **departure from
normal** — the tensor's value minus the normal, in the unit, and as the
trainer's own z-score `(value − normal − mu) / den`, with `mu` and `den` read
from that version's `stats.json`. This is the number the model is actually
handed. The trainer charges a pentad to the calendar month it *opens* in, so
the departure uses that month's normal even when the selected date is in the
next month.

The departure is never a reason to download the tensor: without the Global
tensor's pentad already in memory, a click prints the normal and a one-line
hint instead of fetching 14.5 MB.

---

## The three versions

Which years count as "training" decides the climatology. The three versions
are three answers to that question, computed by the same function with a
different hold-out mask:

| version | name on the page | training years | held out |
|---|---|---|---|
| `all` | **All years (1982–2024)** — the default | every year | nothing |
| `dev` | **Development holdout — 2009, 2017 and 2023 held out** | every year except 2009, 2017, 2023 | those three years (the E-059 regime) |
| `paper` | **The paper's split — trained on 1982–2020 less 2009 and 2017** | 1982–2020 except 2009 and 2017 | 2009, 2017, and all of 2021–2024 (the retrospective evaluation period) |

The names, rules and spans on the page are read from the index, not typed into
the app. Flipping between `all` and `paper` at one cell is a direct view of the
last four years' trend: `paper`'s normal is fitted to 37 years, `all`'s to 43.

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
channel switch is one more 4 MB plane. Either way a change of channel, version
or month is a decode or **one** read, never more.

Every number in this arithmetic — `header_len`, `shape`, `dtype`,
`plane_bytes`, the grid, the channel names, labels, units, colour ramps and
(mean, sd) — comes from **`data/family7_clim_index.json`**, which is why the
app contains no 721, no 1440, no 12 and no channel name. Decoded planes are
kept in a 24-plane cache keyed by (version, group, month, channel), so stepping
back to a month or version already seen is free; a held date key is coalesced
through `scrubApply` into one read per *settled* date.

The index's channel metadata and `norm` are **copied** from
`data/family7_index.json`, and the index carries that tensor's `stem`; the
publisher refuses to write one against a different tensor.

---

## Downloads

The layer's row has a **⤓ downloads** fold (the hover card lists the same
links), for the selected version and the selected channel's group:

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
