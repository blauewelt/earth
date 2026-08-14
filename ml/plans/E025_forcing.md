# E-025 · Forcing: CO₂, the energy balance, and the 20-year projection

**Status: PLAN — nothing here has been dispatched.** Written 2026-08-14 in
answer to Chris: *"How should the co2 / the energy balance (see the layer on
the globe) be modeled as part of the prediction? Can you thoroughly
investigate and make a proposal that can be implemented by a less capable
model?"*

Read this file in order. §1–§3 are the investigation and they change the
design; do not skip to §5 and implement "add CO₂ as an input", because §1
shows why that specific thing is a trap.

---

## 0 · What you need to know before touching anything

- **Governing rules:** `ml/CLAUDE.md`. Rule 1: the EXPERIMENTS.md entry is
  written AT DISPATCH, hypothesis first. Rule: replicates, not arms — three
  seeds minimum, because the RAPID probe's seed sd is ~0.12.
- **The system:** a frozen 41 M PixelMAE codec maps each pixel-month to a
  64-d embedding `z`; `ml/temporal.py` trains a transformer that predicts
  `z_{t+1}` from a K=24-month window of `z`. `ml/rollout_spatial.py` rolls it
  forward over all 84,405 window pixels and scores it.
- **Where the programme stands.** Unroll: closed (E-010, E-020). Local
  spatial coupling: closed and *negative* (E-022 — touching neighbours make
  it worse). A decorrelated 222 km ring: **helps**, −3.9 % one-step error
  (E-023). Bigger pixel arrays: measured and rejected without spending GPU
  (E-024). Forcing is the last of E-021's two named missing ingredients and
  the only one never tested.
- **What E-021 found that this experiment exists to fix:** rolled forward
  without forcing, the model settles onto a seasonal limit cycle near
  −0.8 Sv with lag-12 autocorrelation 0.99. It has no channel through which
  anything outside the ocean can push it, so a 240-month projection is
  "what the learned dynamics does unforced", which is not what anyone means
  by a climate projection.

---

## 1 · The investigation, part 1: CO₂ is a clock

`ml/measure_forcing_info.py` (run 2026-08-14, no GPU). Correlation of each
candidate forcing series with a **linear time index**, over the model's
training window 1982-01 … 2024-12:

| series | source | r with linear time |
|---|---|---|
| CO₂, Mauna Loa monthly | NOAA GML, keyless CSV | **+0.9905** |
| CO₂ deseasonalised | same file | **+0.9946** |
| ERF anthropogenic (AR6) | already baked in `data/eei.json` | **+0.9921** |
| ERF natural (volcanic+solar) | same | +0.3851 |
| sunspot number | SILSO monthly | −0.2519 |
| ONI (ENSO index) | NOAA PSL monthly | **−0.0901** |

**Over this window CO₂ and anthropogenic ERF *are* the calendar**, to within
1 %. A model handed CO₂ has been handed `t`. It will fit the warming trend,
score better on held-out years that sit *inside* 1982–2024, and be
indistinguishable from a model that simply learned the date — which is
precisely the memorisation E-021 already caught this model doing. Naming that
input "forcing" would be a naming choice, not a finding.

This is the degeneracy `ml/CLAUDE.md` §4.9b says must be closed or measured
before dispatch, and everything in §4 and §10 below is built around it.

## 2 · The investigation, part 2: how much is there to gain at one step?

Same script: incremental held-out variance explained on top of the centre
pixel's own three-month history, pooled ridge, 600 pixels, held-out years
2009/2017/2023.

| input | gain |
|---|---|
| linear time alone | +0.00165 |
| CO₂ alone | +0.00196 |
| **CO₂ beyond a linear clock** | **+0.00043** |
| CO₂ *shuffled across years* (control) | −0.00034 |
| ONI alone | +0.00005 |
| sunspots alone | +0.00044 |
| ERF anthropogenic alone | +0.00164 |
| everything together | +0.00243 |

Three things follow.

1. **All of it is tiny.** The best combination is +0.0024 where E-023's ring
   is +0.0112 and E-022's 3×3 is +0.0063. For *one-step* prediction, forcing
   is worth about a fifth of a spatial input we already rejected as too
   small to bother with.
2. **Because the state already contains it.** The centre pixel's own recent
   embeddings encode the warming that CO₂ would announce. A covariate that
   tells the model what its own input already says cannot add much.
3. **The instrument works** — the shuffled control goes *negative*, so the
   small gain that exists is genuine temporal alignment and not a
   distributional artefact.

**Do not read the ONI row as "ENSO is useless".** A pooled linear model gives
a global scalar one shared coefficient, so it can only add a
spatially-uniform term, while ENSO's actual signature is a pattern. This
probe is structurally weak for exactly the channel most likely to matter at
seasonal range. Treat +0.00005 as "not measurable this way", not as zero.

## 3 · The investigation, part 3: the energy balance is an OUTPUT, not an input

Chris's question names "the co2 / the energy balance (see the layer on the
globe)". They are not the same kind of quantity and must not enter the model
the same way.

- **Radiative forcing (CO₂, ERF)** is *external*: the atmosphere's
  composition pushes on the ocean. It is a legitimate input.
- **Earth's energy imbalance and ocean heat content** (`data/eei.json`,
  the Energy tab: NOAA NCEI OHC 0–700 m and 0–2000 m) are the *ocean's
  response*. OHC is a vertical integral of the same temperature field the
  codec embeds and the model predicts. Feeding OHC in as a covariate feeds
  the model a smoothed summary of its own target.

That is leakage, and it is the seductive kind: it would improve every metric
and mean nothing. **EEI and OHC belong on the other side of the experiment —
as validation targets.** A forced model that projects the North Atlantic
forward should reproduce the observed OHC accumulation over the held-out era;
that is a test the model can fail, which is what makes it worth running.

## 4 · Settled design decisions

**4.1 — Two channels, two experiments.** They differ in what can be
validated, so they must not be entangled in one arm.

| channel | series | r with t | testable on the current split? |
|---|---|---|---|
| **SLOW** (anthropogenic) | CO₂, ERF anthro | ~0.99 | **No** — needs an era split (§4.4) |
| **FAST** (natural) | ONI, ERF natural (volcanic), sunspots | ≤ 0.39 | Yes |

**4.2 — How forcing enters: extend the month-feature vector.** The model
already takes `month_seq [B, K, 2]` (sin/cos of calendar month) alongside
`z_seq`. Forcing is exactly the same kind of thing — a global, time-varying
covariate known for every step — so it extends that vector to
`[B, K, 2+F]`. This is a one-line change to the input width plus a wider
`Mt` table, it touches no gather site, and it composes with the E-023 ring
because that lives on the `z` side. **Do not** invent a separate encoder.

**4.3 — Forcing is supplied at the step being predicted, not lagged.** For a
window ending at t predicting t+1, step j carries the forcing at that step's
own month. During a roll this is what makes the future scenario-conditional.

**4.4 — The ERA SPLIT is the only honest test of the slow channel.** Train on
1982–2010, hold out **2011–2024 entirely**. This is the only split where
"knowing the forcing" and "knowing the date" come apart, because the test era
lies *outside* the fitted trend. Interpolated held-out years cannot separate
them, no matter how many seeds are run.

**4.5 — Controls, pre-registered.** Every slow-channel arm ships with a
`time` arm that feeds a bare linear index in place of CO₂. **If the CO₂ arm
does not beat the time arm, the correct report is "we added a clock", and it
gets written that way.**

**4.6 — Scenarios are the deliverable.** In a hindcast the roll gets observed
forcing. In a *future* roll it gets a pathway the caller names, which turns
E-021's unforced fan into a scenario-conditioned projection. Two pathways to
ship: `flat` (forcing frozen at the last observed value — the counterfactual)
and `ssp245` (CO₂ continuing at its recent +2.4 ppm/yr with the AR6 ERF
relation). The comparison *between* the two is the scientific output; neither
absolute trajectory should be quoted alone.

**4.7 — Volcanic forcing is deferred, and why.** The GISS stratospheric AOD
file that would give a proper volcanic series returns 404 (checked
2026-08-14). `data/eei.json`'s `erf_natural` is yearly and carries the same
information at coarse resolution; use that. Do not hand-code eruption pulses
— a hand-placed exponential decay is a hand-picked threshold wearing physics
clothes.

## 5 · Implementation A — bake the forcing table

New file `scripts/refresh_forcing.py`, writing `data/forcing.json` **and**
`ml/cache/forcing.npz`. All three sources are keyless HTTPS, verified
reachable 2026-08-14:

| series | URL | cadence |
|---|---|---|
| CO₂ | `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv` | monthly, 1958-03→ |
| ONI | `https://psl.noaa.gov/data/correlation/oni.data` | monthly, 1950→ |
| sunspots | `https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt` | monthly, 1749→ |
| ERF anthro/natural | `data/eei.json` (already baked) | **yearly**, 1955–2024 |

Parsing for each is already written and working in
`ml/measure_forcing_info.py` (`load_co2`, `load_oni`, `load_sun`,
`load_erf`) — **import or copy those functions rather than re-writing
them.** Requirements:

1. Output covers exactly the tensor's months (`f3_small.npz["months"]`,
   1982-01…2024-12, T=516) in that order.
2. Yearly series (ERF) are held constant within their year. Record that in
   the file as `"erf_interp": "step"`; do not linearly interpolate, which
   would invent monthly structure that the source does not have.
3. **Standardise on TRAIN months only** and store `mean`/`sd` per series in
   the file. A statistic computed over held-out months is leakage, and this
   is the cheapest place in the pipeline to leak.
4. Write both `raw` and `z` (standardised) arrays, plus `sources` with each
   URL and the fetch date, plus the measured `r_with_time` per series — so
   the file itself carries the §1 warning to whoever reads it next.
5. Refuse (exit 1) if any series has a gap longer than 2 months inside the
   window, rather than silently interpolating across it.

## 6 · Implementation B — `ml/temporal.py`

**6.1** Load the table next to the month features (~line 915, where
`ctx_all` is built):

```python
FORCE = np.zeros((T, 0), np.float32)
if a.forcing:
    fz = np.load(a.forcing_file)          # ml/cache/forcing.npz
    names = [s.strip() for s in a.forcing.split(",")]
    FORCE = np.stack([fz["z"][fz["names"].tolist().index(n)]
                      for n in names], 1).astype(np.float32)
    assert len(FORCE) == T, "forcing table is not the tensor's months"
ctx_all = np.concatenate([ctx_all, FORCE], 1)     # [T, 2+F]
```

Everything downstream already reads `Mt[base + j]`, so **no gather site
changes.** That is the whole point of choosing this route.

**6.2** Model width, in `TemporalTransformer.__init__`, mirroring exactly how
`stencil` was handled in E-022:

```python
self.n_force = n_force
self.inp = nn.Linear(stencil * d_z + 2 + n_force, d_model)
```

with `n_force=0` reproducing today's shapes **exactly**, so every published
head keeps loading `strict=True`.

**6.3** CLI: `--forcing co2,oni,...` (default `""`), `--forcing-file`
(default `ml/cache/forcing.npz`). Record BOTH in `stage2_config` and in
`temporal.json:scale` — E-022 had to infer an arm's identity from its
parameter count once, which is a check, not a record.

**6.4** Workflow: a `force:co2-oni` token in the `window` input, dashes for
commas (the `direct:` token's convention, because commas separate window
fields). Parse it beside `stencil:`/`ring:` (~line 895). **Measure the Probes
`run:` block afterwards — the ceiling is 21,000 characters and it currently
sits at 20,670.** Inputs must stay at exactly 25.

## 7 · Implementation C — tests, before any GPU

New `tests/test_e025_forcing.py`:

1. **Zero-weight equivalence** (the one that matters, as in E-022): a model
   with `n_force=2` whose forcing input columns are zeroed and whose other
   columns are copied from an `n_force=0` model must be `allclose` to it at
   1e-6. This pins the column layout with an identity rather than a
   threshold.
2. **Back-compat**: `n_force=0` builds the exact legacy shapes; a published
   head loads `strict=True` through the new code.
3. **Alignment**: build a forcing table that is a known ramp, run one
   training step, and assert the value the model saw at window step j is the
   table's value at that step's month — an off-by-one here is invisible in
   every metric and would silently shift the whole experiment by a month.
4. **Standardisation uses train months only**: construct a table whose
   held-out months are wildly out of distribution and assert the stored
   `mean`/`sd` are unchanged by them.
5. **Refusal**: `--forcing` naming a series absent from the table exits with
   a message naming the series, not a KeyError.

## 8 · Implementation D — scenario-conditioned projection

`ml/project_amoc.py` and `ml/rollout_spatial.py` build month features for
rolled steps. Extend both to append forcing:

- **Hindcast** (context ends in the past): observed forcing for the rolled
  months. This is what makes the hindcast a fair test of a forced model.
- **Future roll**: `--scenario flat|ssp245`, implemented as a function
  producing the forcing vector for each future month. `flat` holds the last
  observed value; `ssp245` continues CO₂ at +2.4 ppm/yr and maps it to ERF
  with the AR6 logarithmic relation ΔF = 5.35 · ln(C/C₀). Record the
  scenario in the output JSON — a projection whose forcing pathway is not
  recorded is not reproducible.
- **The headline output is the DIFFERENCE between scenarios**, not either
  trajectory: the difference is the model's forced response, while the
  absolute level still carries all the biases E-021 measured (the ~5×
  under-dispersion, the contraction to a limit cycle).

## 9 · Dispatch plan

Run in this order; each step's result decides whether the next is worth it.

- **R0 (free)** — `python3 -m pytest tests/test_e025_forcing.py -q`, all
  green. Bake `data/forcing.json` and eyeball the CO₂ series against
  `data/eei.json`'s ERF for sanity.
- **R1 (free)** — extend `ml/measure_forcing_info.py` with a *per-pixel*
  ENSO regression (a global scalar times a per-pixel coefficient) to close
  the §2 caveat about the pooled probe being blind to patterned responses.
  If ENSO shows nothing even there, drop the fast channel from R2 and say so.
- **R2 (~4.5 GPU-h, ~$1.5)** — the FAST channel on the standard split, three
  seeds: `force:oni-erfnat-sunspots`, 60 k steps, everything else the e017
  recipe. **Control:** the existing e017 trio; no new baseline runs needed.
  **Falsifier:** three seeds whose forecast ratio sits inside the e017 band
  0.19216 ± 3 × 0.00205 → the fast channel does not help at one step.
- **R3 (~13.5 GPU-h, ~$3.7)** — the SLOW channel on the ERA SPLIT. **NINE
  runs, three arms × three seeds**, all trained on 1982–2010 only
  (`--holdout-years` covering 2011–2024) and scored on the 2011–2024 era:
  1. `force:co2` — the hypothesis;
  2. `force:time` — the clock control (§4.5);
  3. **no forcing at all** — the era-split baseline.
  The third arm is not optional and is the easiest thing in this plan to
  forget: changing the training split changes the data, so e017's numbers
  are NOT a valid control for R3 (§11, last bullet). Without an unforced arm
  trained on the same 29 years there is nothing to compare against.
  **This is the experiment that answers Chris's question**, and the time arm
  is what makes its answer interpretable.
- **R4 (~1.5 GPU-h)** — roll the R3 heads with `sroll`, plus the
  scenario-conditioned projection (§8) for `flat` vs `ssp245`, and check the
  projected OHC accumulation against `data/eei.json` over the held-out era
  (§3 — the validation target, not an input).
- **R5** — harvest, table, RESULT, **stop both boxes**.

Total ≈ 20 GPU-h, ≈ $5.4 — R3 is three quarters of it, and it is the
only part that answers the question as asked.

## 10 · Pre-registered decision rules

- **R3 CO₂ arm beats the time arm outside seed spread** → forcing is a real
  covariate for this model; ship the scenario projection and say so.
- **R3 CO₂ arm ≈ time arm** → we added a clock. Report it that way, keep the
  scenario machinery (it is still the honest way to project), and stop
  claiming CO₂ as physics.
- **Both beat the unforced baseline on the held-out ERA** → the useful thing
  is knowing *that* there is a trend, not *what* drives it, which is worth
  writing down plainly because it is a statement about the data record's
  length, not about the ocean.
- **Neither beats it** → forcing does not reach the ocean state through this
  architecture at monthly cadence, and the projection stays explicitly
  unforced with E-021's caveat attached.

## 11 · Pitfalls that will bite

- **The clock degeneracy (§1).** Every result must be read against the time
  control. This is the single largest risk in the whole experiment.
- **Leakage through standardisation** — statistics over all months, not just
  train months (§5.3).
- **Leakage through OHC** — it is the ocean's own heat, not a forcing (§3).
- **Off-by-one in forcing alignment** — invisible in metrics; test 7.3.
- **21,000-char Probes block, 25 workflow inputs** — measure after editing.
- **Commit before running.** Three separate measurements were lost to a
  container restart tonight because their output lived in `/tmp` or the
  gitignored `ml/runs`.
- **A rerun is not a resample** — the era split changes the training data, so
  R3's numbers are NOT comparable to e017's; its baseline must be an
  unforced arm trained on the same 1982–2010 split. **Budget for that arm**
  (three more seeds, ~4.5 GPU-h) or the comparison has no control.
