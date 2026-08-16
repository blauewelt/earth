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

---

## 6 · Order of work

1. **Pentad truth first** (`ml/build_truth_pentad.py`) — no credentials
   needed, and it is the claim worth testing. Re-derive RAPID, FC, MOVE and
   SAMBA at 5-day cadence from their native archives.
2. ~~Chris's decision on §3~~ — resolved: GLORYS12 daily, credentials in the project doc.
3. The GLORYS12 daily → pentad fetcher (the long pole: 1/12° subsetted to the NA window, year-by-year, resume-friendly).
4. The pentad tensor builder.
5. Re-anchor Chinchilla from observed-value counts; re-size the codec.
6. Then daily, once object storage exists.

---

*Written 2026-08-16. Cadences in §1 measured from the live archives the same
day, not taken from documentation.*
