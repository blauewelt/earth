# The Cones tab's **Data** mode — what the model actually reads

The [Cones tab](https://blauewelt.github.io/earth/) draws the *dependency
cone*: the space-time window our own forecaster reads in order to predict one
pixel. Until now it drew the cone's **shape** — which cells, how far away, how
long ago. **Data** mode puts the cone's **values** in it: the real numbers,
out of the real training tensor, gathered by the real code.

This page says what those files are and how to make them again.

---

## What you are looking at

A few words first, because none of them are ordinary English.

- **A pentad** is a fixed five-day bin. The tensor has one frame per pentad,
  counted from 1982-01-01, so bin = ⌊days since then ÷ 5⌋ and it holds bins
  0–3141 (1982-01-01 → 2024-12-31). "Lag 3" means three bins — fifteen days —
  earlier. Every date the tab prints is the day its bin *opens* on.
- **The codec stencil** is what the *encoder* reads in order to compress one
  pixel into 32 numbers: a 3×3 patch of its neighbours at lag 0, plus a
  sunflower of dots at lags 1–6 (up to 30 days back), for each of the tensor's
  42 channels. 42 patch tokens + 706 dot tokens = **748 tokens per anchor**.
- **The stage-2 stencil** is what the *forecaster* reads on top of that: a ring
  of 24 dots per lag, from lag 7 out to lag 143 (715 days back), over the
  compressed codes rather than the raw field. It is **empty for lags 0–6 by
  construction** — the codec has already read that whole disc, so stage 2 keeps
  only the anchor's own column there. Its first ring is at lag 7.
- **Raw** is the measurement in its own unit (°C, m/s, m, N/m²).
  **Anomaly** is that value minus its own calendar month's average — computed
  on *training years only*, so the held-out years cannot leak into the
  climatology — and then divided by the channel's spread over the training
  pool. The anomaly is what the model is actually given, which is why it runs
  about −3…+3 rather than 15 °C.
- **Hollow** dot: the cell is outside the tensor's window (0–70° N,
  100° W–20° E). The model reads it as missing and never wraps it round the
  globe — this window is a basin, not a planet, so a wrap would put the Iberian
  shelf one cell west of Florida.
- **Dimmed** dot: inside the window, but never observed there — land, cloud, or
  an Argo float that did not profile that pentad. The model gets a "missing"
  token, not a zero.

Tap a dot and the read-out names the lag, **that dot's own date** (the anchor's
date minus its lag — for stage 2 that can be two years earlier than the pixel
being explained), the offset in kilometres east and north, the raw value with
its unit, the anomaly, and whether the cell was observed, missing or off the
window.

---

## Where the data comes from, and what runs

Nothing about which cell is read, or what is in it, is decided in JavaScript.
The exporter calls the production code:

| what | called | file |
|---|---|---|
| inner cone (patch, dots, future targets, `valid`/`obs`) | `ConeSampler.sample` | `ml/cone_sampler.py` |
| outer stencil (stage 2's rings) | `cone.outer_spiral` | `ml/cone.py` |
| the anomaly the codec sees | `anomaly_transform`'s three passes | `ml/trainprobe.py` |

- **`ml/export_cone_sample.py`** writes one JSON per anchor.
- **`tests/test_export_cone_sample.py`** asserts the exported `valid`/`obs`
  flags and values are **bit-identical** to a direct `ConeSampler.sample` call,
  that the dot counts equal `data/cone_geometry.json`'s own `counts` (706 dots
  + 42 patch = 748), and that the streaming anomaly is **bit-identical** to
  `trainprobe.anomaly_transform` on a toy tensor.
- **`tests/data.spec.js`** pins the index and the in-repo fixture's schema.
- **`tests/app.spec.js`** drives the tab against that fixture, with no network.

### Why the anomaly is streamed rather than transformed in place

`ml/train_cone.py` loads the tensor and lets `anomaly_transform` rewrite it in
place — 35.7 GB of writes. A sandbox with 30 GB of disk and 7 GB of RAM cannot
do that. `streaming_anomaly` is the same three passes over a *stream*
(the `.npz` member decompressed twice at ~480 MB/s instead of stored once),
keeping only the ~169 bins the cone touches. Same float64 accumulators, same
Chan parallel variance combination, same float16 round-trip between passes 2
and 3 — and the equality is a test, not a claim.

---

## Regenerating the samples

```bash
# 1. the tensor (5.3 GB compressed, 35.7 GB inside), from the public release
mkdir -p ~/tensors && cd ~/tensors
for a in aa ab ac ad; do
  curl -fsSL -o part.$a \
    "https://github.com/blauewelt/earth/releases/download/data-cache-v1/family4_na025_pentad_r3_fa460837fa.npz.$a"
done
cat part.a? > family4_na025_pentad_r3_fa460837fa.npz && rm part.a?
sha256sum family4_na025_pentad_r3_fa460837fa.npz
# must be fa460837fa172825ee76c8fc6fc4da75fa7b96d64519a2e2186f5c306cf03ea9

# 2. export (two streaming passes over the tensor; ~1 h on two CPU cores)
cd ~/earth
python3 ml/export_cone_sample.py \
    --stream-npz ~/tensors/family4_na025_pentad_r3_fa460837fa.npz \
    --scratch ~/tensors/scratch \
    --out ~/cone_out \
    --anomaly trainer --outer-stride 2 \
    --fixture data/cone_samples/fixture.json

# 3. publish + write the index (uploads, downloads back, sha256-compares,
#    and measures the CORS headers a browser would get)
export HF_TOKEN=$(cat ~/.hf_token)          # never in argv
python3 ml/upload_cone_samples.py --dir ~/cone_out

# 4. the tests
python3 -m pytest -q tests/test_export_cone_sample.py
MIRROR=1 scripts/run_tests.sh tests/data.spec.js -g "cone_samples"
MIRROR=1 scripts/run_tests.sh tests/app.spec.js -g "Cones data mode"
```

`--outer-stride 2` keeps every second outer lag (7, 9, …, 143 — 69 of the 137).
Without it each anchor file is roughly twice the size; the budget is ~6 MB per
anchor, and the stride is recorded in each file's `meta.outer.stride` so the
tab can say what it is showing.

### The anchors

Four are the presets the tab already had; the fifth exists so the page can show
what an unreadable neighbour looks like.

| id | where | why |
|---|---|---|
| `gulf_stream` | 36° N 70° W | the western boundary current |
| `rapid` | 26.5° N 70° W | the RAPID array's latitude — the AMOC section |
| `labrador` | 58° N 52° W | deep convection |
| `equator` | 0° N 30° W | the window's southern edge: part of the cone falls off it |
| `ionian_edge` | 36° N 19° E | the window's eastern edge, and coastal — hollow dots and missing dots in one anchor |

Dates: 24 consecutive pentads (120 days) from 2015-01-03. 2015 is a training
year under *both* protocols — it is none of the three development holdout years
(2009, 2017, 2023) and it is before the terminal split's 2020 cut — so nothing
shown here is a held-out year the model was never allowed to see.

---

## Where the files live

The index is in the repo: **`data/cone_samples.json`** — anchors, dates, URLs,
sizes, sha256 of each file, the tensor's own sha256, the exporter's commit, and
the measured CORS headers. The samples themselves are on the Hugging Face Hub,
in the dataset repo `chfrank/earth-tensors` under `cone_samples/`, which is
`CLAUDE.md` §3's second approved live endpoint: keyless, CORS-open, and read
once per anchor rather than streamed. A failure degrades to a hint and the
geometry mode — never to a broken tab.

`data/cone_samples/fixture.json` is a small trimmed copy (one anchor, three
dates, outer lags 7/35/143) with the *same schema*, so the browser tests run
with no network at all.

---

## Related

- [The E-069 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_cone_codec.md)
  — why the cone is split in two, and what the codec is for.
- [What is a training example](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_TRAINING_EXAMPLE.md)
  — one anchor, 748 tokens, 6,579 numbers, and the three boundaries.
- [`ml/cone.py`](https://github.com/blauewelt/earth/blob/main/ml/cone.py)
  — the geometry's one definition.
- [`ml/cone_sampler.py`](https://github.com/blauewelt/earth/blob/main/ml/cone_sampler.py)
  — the loader the trainer uses and this exporter calls.
