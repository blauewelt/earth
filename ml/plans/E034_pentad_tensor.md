# E-034 · The pentad tensor — build plan

**Status: IN PROGRESS.** Chris approved the E-033 recommendation on
2026-08-16: *"Let's do E-34."* — build the 5-day tensor first, then the daily
one.

This document is the recipe decisions and their justifications. It is
written before the build so the decisions cannot be reverse-engineered from
whatever the build happened to do.

---

## 1 · What this buys, restated as the two testable claims

**Claim A — six times the timesteps.** Stage 2 consumes timesteps and there
are 516 of them. Pentad gives 3,096 over the same 1982–2024 axis.

**Claim B — six times the LABELS, which is the one that matters.** The
transport probe was measured to sit at a *state/label* ceiling, not a
representation tax: pooled embeddings score 0.627 where the true
uncompressed fields score 0.631 (`probe_state_ceiling.py`). No model change
can move that. More labels can, and finer cadence is the only way to get
them from records that already exist.

**Both truth records were probed today rather than assumed** (root
`CLAUDE.md`: *never guess what an archive serves — ask it*):

| record | native cadence | measured | at pentad | today |
|---|---|---|---|---|
| **RAPID 26.5°N** | **12-hourly** | 14,599 samples, median Δt = 0.5 d, from 2004-04-01 | ~1,460 | **240 monthly means** |
| **Florida cable** | **daily** | file header reads `DAILY FLORIDA CURRENT TRANSPORT`, 365 rows for 2015 | ~3,100 (1982–2024) | 516 monthly means |
| MOVE 16°N | ~daily | per fetcher docs | ~1,600 | monthly |
| SAMBA 34.5°S | daily | per fetcher docs | ~580 | monthly |

RAPID's file also carries its constituent terms (`t_ek10` Ekman, `t_gs10`
Gulf Stream, `t_umo10` upper-mid-ocean) at the same 12-hourly cadence, which
the classical-baseline comparison in the paper currently reconstructs from
monthly means — a free improvement to §Making them comparable, noted here so
it is not forgotten.

---

## 2 · Cadence audit, per channel — the decision that defines the recipe

| channel group | native cadence | pentad policy |
|---|---|---|
| NCEP R1 wind stress τx, τy | **daily** — already read as dailies today, to build the within-month σ | pentad mean of the dailies |
| NCEP R1 storminess τ_std | daily | **within-PENTAD** σ — the same statistic over a 5-day window, which is a strictly better storminess measure than a monthly σ |
| OISST SST | daily | pentad mean |
| GLORYS currents, MLD, SSH | **daily** (GLORYS12 `cmems_mod_glo_phy_my_0.083deg_P1D-m`, verified reachable 2026-08-16) | pentad mean of the dailies, subsetted to the NA window and binned to 0.25° |
| RG-Argo T/S, 16 levels | **monthly gridded** (floats profile ~10-daily) | see §4 — the missing-token decision |

---

## 3 · RESOLVED: the GLORYS base channels come from GLORYS12 daily

*This section originally read "Open item ... I cannot verify this without
Chris", on the belief that CMEMS credentials were deleted after each use by
policy. That belief was wrong — it came from a stale line in the root
`CLAUDE.md` which had frozen one 2026-08-04 episode into an apparent standing
rule. Chris: "I don't have such a policy." The credentials are in the project
doc `claude/copernicus-marine-access.md`, and the root CLAUDE.md line has
been corrected so the next session does not repeat the mistake.*

**Verified 2026-08-16**, by calling the API rather than reading about it:
`cmems_mod_glo_phy_my_0.083deg_P1D-m` — GLORYS12 reanalysis, **daily**, 1/12°
— is reachable with the account. So the highest-fidelity option is available
and is the one to take:

- **Fetch GLORYS12 daily for the NA window and pentad-mean it.** The base
  channels (`cur_speed`, `log_mld`, `ssh`) then move at true pentad cadence
  like the wind and SST channels, and only Argo remains a slow channel.
- Download cost is the real constraint, not access: 1/12° is 16× the pixels
  of the 1/4° ensemble we used for the monthly bake, and we bin down to 0.25°
  anyway. Fetch **subsetted to the NA window** (100 W–20 E, 0–70 N) and
  daily-to-pentad average server-side where the toolbox allows it, or
  stream year-by-year as the monthly bake already does — that pattern is
  written and resume-friendly.
- The interim (`myint`) stream for the most recent months carries a different
  dataset ID than the one guessed here; look it up in the catalogue at build
  time rather than hardcoding, the same way the monthly bake resolves it.

Consequence for §6: the pentad tensor no longer has to ship with stale base
channels, and the "option 3" fallback (build from the daily-native channels
only) is retired.

## 4 · The Argo missing-token decision (explicit, per E-033)

Argo's gridded product is monthly. Inside a pentad axis a month is ~6
timesteps. Two candidate policies:

- **Forward-fill** — repeat the monthly value across its pentads. **Rejected.**
  It tells the model the subsurface was observed on days when it was not, and
  the whole architecture is built on the opposite claim (*missingness is
  information*; the `missing` token is distinct from the `mask` token by
  design and the distinction was measured to matter).
- **One live pentad per month, `missing` elsewhere.** **Chosen.** The Argo
  channels carry a value in the pentad containing the month's nominal
  timestamp and a learned `missing` token in the other five. This is
  truthful, it exercises machinery that already exists and is tested, and it
  makes the eventual daily rung a change of one constant rather than a
  redesign.

Consequence to watch: the observed-value count per pentad drops, so the
Chinchilla anchor must be recomputed from **observed values**, not from
tensor volume. The build already prints that count; it must be re-read
rather than scaled.

---

## 5 · Storage and the recipe guard

- **float16 for the tensor.** Fields are anomaly-space and normalised to
  ~N(0,1); fp16 carries ~3 decimal digits there, far below observational
  error. Free 2×: **65.3 GB → 32.6 GB**. Do it here rather than as a
  separate change.
- **Box-disk arithmetic**, which decides whether this fits at all:
  32.6 GB tensor + 33.4 GB embedding cache + ~15 GB torch image =
  **81 GB of a 100 GB disk.** It fits, with no room to spare, and a Vast
  disk cannot be resized. Any *further* rung (daily = 170 GB) requires the
  object-storage work in E-033 §5 first.
- **A NEW RECIPE, never an edit to the monthly one.** `build_family3.py`
  carries `RECIPE_REV = "f3r1"` as its skip guard, and the monthly tensor is
  sha-pinned (`adcbe700…`) and verified on every box for cross-run
  comparability. The pentad build is `build_family4.py` with its own
  `RECIPE_REV`, its own output name, and its own sha — so a pentad and a
  monthly tensor can never be silently mixed, and every result to date stays
  reproducible.
- Provenance must record the cadence explicitly, so a run's own manifest
  says which axis it trained on.

**The aggregator's output contract, settled 2026-08-16 — `build_family4.py`
reads this, so it is written down before that script exists.** The first
`ml/aggregate_cadence.py` could not have completed this step at all: it kept
one accumulator per (bin, variable) resident for the whole run and
materialised a single stacked `.npz` at the end, which over the 2,339 pentad
bins of 1993-01..2024-12 at 1/12° is **227 GB of RAM and a 45.4 GB file**.
Measured, not modelled — one real month peaked at 0.93 GB for 7 bins and the
arithmetic scales exactly. It would have failed *after* the ~8-hour GLORYS12
pull, not before it. Three consequences now hold:

- **Streaming.** A bin is flushed the instant the date crosses out of it, so
  memory is O(one bin) and does not grow with the run. Measured over three
  real months: peak open bins 1, peak RSS **0.16 GB** at 0.25° and 0.56 GB on
  the native grid.
- **Spatial binning happens HERE** (`--bin-deg 0.25`), as §2 of this document
  already specified for the GLORYS channels and nothing was yet doing. Full
  archive at 0.25°: **5.0 GB**, which fits; at 1/12° it does not. Each DAILY
  slice is binned and only then averaged in time, so the day count that
  `--min-days` guards stays a count of *days* rather than of source-cell-days.
- **Output is one memmapped `.npy` per variable plus `index.npz`** (`bin_index`
  = the full contiguous bin range, `has_data` marking bins no chunk covered,
  plus `lat`/`lon` when binned). Row *i* is `bin_index[i]`; nothing has to
  exist in RAM before it can be written.
- **The dailies stream back from the Hub** (`--hf-repo earth-tensors`), because
  `fetch_glorys_daily.py` deletes each chunk once it is restore-verified —
  after the pull there is no local archive to aggregate, by design.

`tests/test_e034_aggregate.py` pins all of it, and the pin that matters is
**bit-identity**: with no `--bin-deg` the streaming version reproduces the
pre-streaming one exactly — same means, same NaN mask, same dropped straddle
bin — verified both on a three-month synthetic fixture and on three real
GLORYS12 months at full 841×1441. That test is three months rather than one
on purpose: the rewrite's first version re-created the output memmap once per
chunk and silently erased every bin already written, and a single-chunk test
cannot see that.

---

## 6 · Order of work

1. ~~**Pentad truth first**~~ (`ml/build_truth_pentad.py`) — done.
2. ~~Chris's decision on §3~~ — resolved: GLORYS12 daily, credentials in the project doc.
3. **The GLORYS12 daily fetcher** — running unattended as
   `.github/workflows/glorys-pull.yml` (the Cowork sandbox kills a
   backgrounded job while the session is idle; the Vast fleet is ruled out by
   `ml/CLAUDE.md` §6, which forbids CMEMS credentials on a rented box). The
   6-hourly schedule IS the resume mechanism — the fetcher treats the Hub as
   its resume source, so the runner's 6-hour cap costs nothing.
4. ~~**The pentad tensor builder**~~ — **built 2026-08-16**:
   `ml/build_family4.py`, `RECIPE_REV = "f4r1"`, output
   `ml/cache/family4_na025_pentad.npz`. See below.
5. Re-anchor Chinchilla from observed-value counts; re-size the codec. The
   build prints the inventory; **re-read it, do not scale the monthly one** —
   the Argo policy in §4 drops the observed count per pentad by construction.
6. **Train four codecs FROM SCRATCH before embedding anything** — see
   `ml/plans/E038_codec_matrix.md`. Chris, 2026-08-17: the pentad and daily
   fields are *out of domain* for the existing codec, so re-encoding with it
   would measure the codec's extrapolation rather than the cadence's value.
   The sharpest form of that argument is the missingness pattern: RG is 32 of
   39 channels, and §4's one-live-timestep-per-month policy takes the share of
   steps carrying a `missing` token on those channels from ~0% at monthly to
   **~83% at pentad and ~97% at daily**. A codec that has never had to
   represent that is being asked to spend most of its capacity on it.
   This REPLACES the old ordering, in which the codec was frozen and only the
   embeddings were recomputed.
7. Then daily — which needs a large-disk box, not new engineering: the daily
   tensor is **165.6 GB** at fp16 over the full 1982-2024 axis (123.2 GB from
   1993), against a 100 GB fleet. E-038 §3 has the staging.

### 4 · What the builder does, and the three things it refuses

**The grid is family-3's, and that is a precondition rather than an
aspiration.** `base025_na.npz` was opened on 2026-08-16: lats 0.0..70.0 (281),
lons −100.0..20.0 (481) — samples **on** multiples of 0.25°, not cell centres.
`aggregate_cadence.py` was emitting centres (280×480, a half-cell off), so it
gained `--grid-align point` (now the default) which reproduces those axes
exactly. `build_family4.py` **asserts** it against the artefact and refuses
otherwise, because a half-cell offset would leave every stencil geometry, the
AMOC eval mask and the corridor definitions quietly describing different
pixels than they name — and would be invisible in every plot anyone would
draw.

**Channels are family-3's 39, imported from that module** so there is one
definition, with the cadence policy of §2 applied per group:

- **base** — `cur_speed = hypot(mean uo, mean vo)` from the *binned
  components*, never a mean of magnitudes. `log_mld` is **log10**, measured
  against family 3 rather than assumed: January 1993 peaks at 2439 m under
  log10 (Labrador Sea convection); natural log would read 30 m, which is not
  a January.
- **rg** — one live pentad per month, `missing` in the other five, per §4
  above. The nominal timestamp is the **15th**: RG carries no within-month
  time and a monthly mean is most nearly centred on the midpoint.
- **wind** — pentad mean, and `tau_std` as the **within-pentad** σ computed
  from the dailies. This is the one channel strictly better at this cadence:
  family 3's is a within-*month* σ, which mixes the storm band with the
  seasonal cycle. A σ is not aggregable from a mean, so it cannot be derived
  from family 3.

**float16, and it is a BOX build.** [3142, 281, 481, 39] is 66.2 GB at float32
and **33.1 GB at float16** — measured by `--dry-run`, and close to this
section's 32.6 GB estimate. It does not fit the sandbox; `--max-bins` builds a
prefix for exercising the path.

`tests/test_e034_family4.py` runs the real builder end to end over a complete
miniature of every source — a pentad aggregation, RG cubes in SIO's schema,
NCEP dailies in PSL's schema, a pentad truth record — and pins seven
properties, including the two that no summary statistic would reveal: that RG
is **not** forward-filled, and that `tau_std` is distinguishable from a
within-month σ on the fixture (a check that cannot fail is not a check).

---

*Written 2026-08-16. Cadences in §1 measured from the live archives the same
day, not taken from documentation.*
