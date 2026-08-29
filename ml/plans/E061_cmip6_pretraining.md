# E-061 · A pretraining corpus, because 43 years is the constraint

**Dispatched as #514 on 2026-08-29 ~11:0xZ.** This plan was written and
committed before the dispatch; §5 is pre-registration.

---

## 1. Why

E-059 established that a 206.66M-parameter stage-2 head trained under a clean
pool reaches its best held-out one-step loss at **step 2,000** and gets worse
for the remaining 198,000 — and that its RAPID probe **degrades** from 0.616
to ~0.52 while two contaminated runs hold flat near 0.60. E-060a reproduces
the same plateau at **7,597,856 parameters**, 27x smaller.

Neither of those is a capacity result. They are a data result. The pool's
headline `209,549,066` train windows is exactly **2,417 end-bins × 86,698
ocean pixels**: the temporal diversity of the training set is 2,417 windows,
against 206.66M parameters. Memorisation is the path of least resistance and
a clean pool removes the reward for it, not the incentive.

Daily cadence does not fix this. It gives 5x the values inside **the same 43
years**, and at fixed K=144 it shortens the context window from 1.97 years to
0.39. The binding constraint is that the ocean record is 43 years long and
RAPID is 20.

The one source of more ocean that is not capped by 43 years is **model
output**: free-running control runs, which generate physically consistent
ocean states indefinitely.

## 2. What is being built

**800 model-years**, against the 43 the reanalysis gives us.

| | source_id | member | months | model-years | grid | calendar |
|---|---|---|---|---|---|---|
| a | `HadGEM3-GC31-MM` | r1i1p1f1 | 6,000 | 500 | eORCA025 (1205×1440) | 360_day |
| b | `CNRM-CM6-1-HR` | r1i1p1f2 | 3,600 | 300 | eORCA025 (1050×1442) | gregorian |

CMIP6 `piControl`, table `Omon`, from the public Google Cloud zarr store over
**anonymous HTTPS** — no credential reaches the box for the source. Both
models are NEMO eORCA025 at nominal 25 km, the **same nominal resolution as
the target**, so nothing is upsampled. That is why these two and not the ~1°
multi-member population.

**Variables: `mlotst`, `zos`, `tos` — all scalars, deliberately.** `uo`/`vo`
on a tripolar C-grid sit on staggered points with a local rotation angle;
regridding them without unrotating produces a systematically wrong subpolar
gyre north of the fold — inside our window, over the Labrador Sea. That is
the silent-error class `check_grid()` exists to refuse, so currents wait for
the ESGF OPeNDAP surface-hyperslab route (~5 MB per model-year per variable,
against 196 MB per month through the zarr store, which chunks all 75 depth
levels globally).

Channel map into family 3's 40-channel layout: `mlotst`→`log_mld` (1),
`zos`→`ssh` (2), `tos`→`sst` (39). **`cur_speed` (0) is missing**, and so are
the 32 Argo and 4 NCEP channels — those are an observing system and a forcing
product, not model fields. Three of forty is a real limitation, stated here
rather than discovered later.

## 3. The regrid, which is the part that must be right

A fixed 1-nearest-neighbour map, built once per model:

- source and target cell centres converted to **3-D unit vectors** (not raw
  degrees — that is what makes the dateline and the poles correct),
- `cKDTree`, restricted to a NA bounding box (lat −8…78, lon −110…30),
- the map **refuses** a maximum nearest-neighbour distance over **40 km**.

Measured, identically for both models: mean **5.91 km**, p99 **12.96**, max
**15.02**. Byte-identical statistics across the two is the expected answer and
a check in itself — in the North Atlantic these are literally the same
eORCA025 points; only the southern extent and the halo columns differ.

Nearest-neighbour rather than bilinear, on purpose: at equal nominal
resolution bilinear buys little and needs a land mask to avoid bleeding land
into coastal ocean. NN transports the model's own mask exactly.

**Validated against GREP** on the same axes (HadGEM3 `tos` vs the ORAS5
January climatology — a control run and a reanalysis share a seasonal cycle
but no year):

| | |
|---|---|
| spatial correlation | **0.9927** |
| RMSE / bias | 1.37 °C / −0.72 °C |
| ocean cells | 84,745 vs GREP's 84,405 |
| mask disagreement | **0.46%** |

and the disagreeing cells are almost entirely the **Great Lakes**, which
HadGEM3's ocean carries and a reanalysis does not. A model-vs-reanalysis
difference, not a geometry error. Cross-check on CNRM (July): corr 0.9833,
mask disagreement 0.12%.

## 4. Storage — sparse, on purpose

The naive dense `[9600, 281, 481, 40]` float16 is **103.8 GB**, of which
**92.5% is the literal absence of Argo and NCEP over a model run that never
had them**. This programme already carries one 165.6 GB millstone (the daily
tensor fits on no box we can rent and `temporal.py` cannot open it at all),
and this corpus is meant to grow — more models, and currents later.

So the sidecar stores **only the live channels**: `[9600, 281, 481, 3]`
float16 = **7.8 GB**, with `channel_index = [1, 2, 39]` and
`n_channels_full = 40` in the index, and `expand_to_full()` scattering it
back into the 40-channel layout as a three-line NaN scatter. Both directions
are pinned in the module's self-test.

## 5. Pre-registered checks

| check | expected |
|---|---|
| rows | **9,600** |
| shape | `[9600, 281, 481, 3]` float16 sparse |
| size | **7.8 GB** |
| `max_nn_km` (both models) | **15.02** |
| `labelled` | **False**, and no truth keys in the index |
| `check_grid` against `base025_na.npz` | verified, not warned |

`preflight_write` allocates and round-trips the whole sidecar **before a
single CMIP6 byte is fetched** (`ml/CLAUDE.md` §0.3), and the HF publish
downloads each file back and compares sha256 (§0.2).

## 6. What this does NOT settle

It builds a corpus. It pretrains nothing.

The experiment that follows has a confound that must be designed for, not
discovered: a 3-of-40-channel corpus fed to a 40-channel codec is a
**distribution shift** as well as a transfer. The control is to embed the
REANALYSIS with the same 37 channels masked out, so "pretraining helped" can
be told apart from "the fine-tune adapted to a channel set it had been
pretrained on". Without that control the result is uninterpretable in either
direction.

Second open question, stated now: a coarse or non-eddying model teaches
large-scale structure — geostrophy, Sverdrup balance, the seasonal MLD
cycle — and does **not** teach the mesoscale. Whether the memorisation
problem is about large-scale structure is exactly what is unknown.

## 7. Licence

CMIP6 output is **CC BY-SA 4.0 per file**, and the terms of use require a
table naming every model and institution, the WCRP acknowledgement, and
citation per the CMIP6 Data Citation Guidelines. `paper/paper.tex` has no
acknowledgements or data-availability section today; that gap has to close
before submission regardless of this corpus. The ShareAlike clause deserves a
sentence of caution if weights are ever published under a chosen licence.
