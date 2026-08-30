# The data ladder · what to import next, and in what order

**Written 2026-08-30.** Chris asked what additional data to bring in, naming
three priorities: **(A) smaller pixels · (B) more depths · (C) whatever is
useful in representing each Earth pixel faithfully.** This document answers
those three, prices every candidate against the window and the disk, and ranks
them. Every volume figure is arithmetic from the project's own grid, stated so
it can be checked.

Companion documents:
[the protocol reset](https://blauewelt.github.io/earth/docs.html?f=ml/plans/PROTOCOL_RESET.md)
(what may count as a result) and
[the E-061 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E061_cmip6_pretraining.md)
(why 43 years is the constraint).

---

## 0. The one thing to know before reading the list

Every candidate falls into exactly one of three classes, and only one of them
touches the measured bottleneck:

| class | what it adds | does it move the wall? |
|---|---|---|
| **new temporal samples** | more distinct end-bins | **yes — this is the constraint** |
| **new spatial detail** | more pixels inside the same 43 years | **no** |
| **new physical variables** | channels the tensor does not have | orthogonal, and cheap |

The pool's `209,549,066` train windows is `2,417 end-bins × 86,698 pixels`, and
heads from 7.6M to 400M all reach their best held-out loss at ~step 2,000.
**Going to 1/12° makes the pool `2,417 × 780,282` — 1.9 billion windows from
the same 2,417 bins.** Nine times the pixels of the same 43 years is nine
correlated views of one dataset, at nine times the disk. Expect the wall to sit
exactly where it sits now.

So: (A) is the expensive question and the one least likely to pay. (B) is
cheap and largely unexploited. (C) contains the highest information-per-byte
items in the whole list.

## 1. The unit of account

One channel, dense over the window (lat 0–70 N, lon −100…+20 E), float16:

| resolution | grid | cells | pentad, 43 y (3,142 bins) | daily (15,706) |
|---|---|---|---|---|
| 1° | 71 × 121 | 8,591 | **0.054 GB** | 0.27 GB |
| **0.25° (current)** | 281 × 481 | 135,161 | **0.849 GB** | **4.25 GB** |
| 0.125° | 561 × 961 | 539,121 | 3.39 GB | 16.9 GB |
| 1/12° | 841 × 1441 | 1,211,881 | 7.62 GB | 38.1 GB |
| 0.05° | 1401 × 2401 | 3,363,801 | 21.1 GB | 105.7 GB |

Checks: 39 × 4.25 = 165.6 GB (the daily millstone) · 40 × 0.849 = 34.0 GB (the
pentad tensor). 1993–2024 is 2,339 of 3,142 pentads = **×0.744**.

**Moving today's 40 channels from 0.25° to 1/12° costs 34 GB → 305 GB.** Not on
a 100 GB box. That is the answer to (A) in one line.

## 2. A finding that pays for the next three items

**32 of the 40 channels are `rg_*`** — Argo Roemmich–Gilson temperature and
salinity at 16 pressures. They are **1° native, monthly, and start in 2004**.
The builder fills nothing: 1982-01 → 2003-12 (22 of 43 years) carries *no*
subsurface data at all, and inside 2004+ the channels are live one pentad in
six. Stored at 0.25° on the pentad axis they cost **27.2 GB — 80 % of the
tensor** — to carry about **0.28 GB** of actual information, a ~97× redundancy
(16× spatial upsample × 6× temporal replication of `missing`).

**Holding the coarse channels at native resolution in a sidecar and upsampling
in the loader frees ~27 GB** — more than the entire cost of the depth expansion
below. The depth upgrade can be made storage-neutral. This is the cheapest
single change in the document and it is not a data import at all.

## 3. (B) More depths — the leverage, and cheaper than expected

### B1 · GREP 3-D at 0.25° — the product the repo already downloads

`GLOBAL_MULTIYEAR_PHY_ENS_001_031`, datasets
`cmems_mod_glo_phy-all_my_0.25deg_P1D-m` (per-member) and
`cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m` (ensemble mean + sd).
`ml/fetch_cmems025.py` already pulls this product — **and passes
`minimum_depth=0, maximum_depth=1` on every call.** The project has never
downloaded a 3-D field from it. The water column has been one argument away
the whole time.

- **Grid: regular 0.25°, the project's exact grid** (`base025_na.npz` was
  built from this very product).
- **75 depth levels, ~1 m surface layer to ~5,902 m.** RG stops at 1,900 dbar.
- **Daily, 1993-01-01 → 2024-12-31** — it ends exactly where the tensor ends.
- Variables: `thetao`, `so`, **`uo`, `vo`**, plus `zos`, `mlotst`, `siconc`.
- Existing credentials, existing code path.

Cost, `glor` member, NA window, fp16, 1993–2024: **8 levels ≈ 20.2 GB · 16
levels ≈ 40.5 GB · all 75 ≈ 190 GB** (infeasible). Start at 8.

What it buys over RG, concretely:

1. **11 extra years** of subsurface T/S (1993–2003), where there is none today.
2. **Every pentad live**, not one in six.
3. **Velocity at depth.** The overturning is a velocity integral and the tensor
   has never seen a subsurface velocity.
4. **Below 2,000 m** — the NADW/AABW limb is currently invisible.
5. **Ensemble spread** (`*_std`) as an honest per-pixel uncertainty channel.

Transfer caution: honest pentad means require pulling all 11,688 days
(~150–250 GB on the wire with CMEMS's int16 packing). Stream month-by-month and
reduce; peak disk stays small. One-day-per-pentad (~40 GB on the wire) is
defensible for `thetao`/`so` and weak for `uo`/`vo`.

**Class: new physical variables + more live bins for the depth channels.** Not
new end-bins, but it converts ~5/6 of the depth channels from `missing` to real
and extends them back 11 years.

### B2 · ARMOR3D — the observation-based second opinion

`MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012`, 0.125°, 50 levels, daily+monthly,
`my` 1993→2024-12-31 and `nrt` 2024-04→2026-08 with no gap. Observation-driven
(satellite SLA + SST projected onto Argo), so it is *independent* of GLORYS —
the pair (GREP, ARMOR3D) is a free model-vs-observation discrepancy channel.
Downsampled to 0.25° at 16 levels it costs the same 40.5 GB as GREP and largely
duplicates it. **Second priority behind GREP**; its own value is independence,
not resolution.

### B3 · The long records — the only real temporal lever

| product | grid | levels | cadence | coverage | access |
|---|---|---|---|---|---|
| **EN4.2.2** | 1° | 42, to 5,350 m | monthly | **1900 → 2026** | anonymous HTTPS |
| **IAPv4** | 1° | 119, to 6,000 m | monthly | **1940 → 2026** | free, temperature only |
| **ORAS5** | 0.25° | 75 | monthly | **1958 → 2026** | CDS account (not held) |

**EN4 T+S at 42 levels, 1900–2026 monthly, on its own 1° grid = 2.2 GB.** At 16
levels, 0.84 GB. This is the only entry in the document that adds end-bins:
extending the axis to 1958 is **+~1,750 pentads, +56 %**; to 1940, **+127 %**.

Two honesty points, and they decide how to spend on it:

- **The labels do not extend back.** The Florida Current cable starts 1982,
  RAPID 2004. Pre-1982 frames give the *field model* new temporal windows —
  it is trained self-supervised on frames — and give the *transport probe*
  nothing. If the step-2,000 wall is the field model's, this is the highest-
  value import here. If it is the probe's, it is worthless.
- **Effective resolution before Argo is far below 1° nominal.** IAPv4's own
  paper reports 1°×1°×1-year sampling coverage above 30 % only from 1960 and
  above 70 % in the late 1960s. Pre-1950 the fields are climatology plus a
  handful of casts; 1950–1990 the upper 700 m is informative along shipping
  lanes and below that is largely background; 2004+ is the only period where a
  1° analysis is a real observation of a 1° cell.

**Which is exactly why it should be MEASURED before it is committed to.** It is
3 GB and a short small-tier run: build a 1958-start tensor holding only the 1°
T/S channels, train, and see whether the step-2,000 wall moves. The asymmetry is
extreme — 3 GB settles whether a 300 GB re-grid could ever be worth anything.

## 4. (A) Smaller pixels — priced, and mostly declined

- **GLORYS12V1 at 1/12°** (`GLOBAL_MULTIYEAR_PHY_001_030`, 50 levels,
  1993 → 2026-06): the obvious answer, and **305 GB** for the current channel
  set. Its `my`/`myint` split date could not be resolved from public pages —
  the repo's existing try-`myint`-then-`my` pattern handles it either way.
  **Declined basin-wide.** If resolution must be tested, test it on a **strip**:
  23–30 N, −80…−10 E at 1/12° is 85 × 841 = 71,485 cells ⇒ **0.45 GB per
  channel pentad**, forty channels = 18 GB. That is an affordable way to ask
  "does resolution at the RAPID section matter?" and it is the experiment to
  run before any basin re-grid.
- **OSTIA reprocessed SST** (`SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`, 0.05°,
  daily, 1981-10 → 2026-03): **take it at 0.25°, for reasons that are not
  resolution.** It is foundation SST (diurnal-free), gap-free by construction,
  starts before the axis, runs to 2026 — and **`sea_ice_fraction` comes free in
  the same file**, a channel the tensor lacks entirely and one that matters for
  Labrador and Irminger convection. 0.85 GB for both.
- **DUACS altimetry** (`SEALEVEL_GLO_PHY_L4_MY_008_047`, now **0.125°**, daily,
  1993 → 2026-01): `adt`, `sla`, `ugos`, `vgos` and their error fields. At
  0.25°, ~3.8 GB. These are the **observed** surface expression of the
  overturning, model-independent — unlike every current base channel. Note
  `sla` is referenced to a 1993–2012 mean, a different baseline from GLORYS
  `zos`, so they are two channels and not one.
- **MUR SST at 1 km — skip.** 528 GB per channel, and it covers 24 of 43 years.

## 5. (C) Faithful pixel state — the cheapest real gains

### C1 · ERA5 forcing, replacing NCEP R1

NCEP R1 is **2.5°**, upsampled 10× onto our grid, and its archive is frozen.
ERA5 is 0.25° native — the project's own grid — 1940 → present. What matters
for AMOC, by mechanism:

- **Momentum:** `metss`/`mntss` (turbulent surface stress, the direct NCEP
  replacement), `u10`/`v10` for a within-pentad storminess σ as today's
  `tau_*_std` channels do.
- **Buoyancy — entirely absent today:** `msnswrf`, `msnlwrf`, `msshf`, `mslhf`
  (net heat flux is their sum, one derived channel), and `mer`/`mtpr` for E−P.
  The subpolar salinity anomaly is the classic AMOC precursor and the tensor
  cannot see its forcing.
- **Free, derived:** wind-stress **curl** and **Ekman pumping**
  `w_Ek = curl(τ/ρf)` — the physically correct forcing for gyre spin-up, a
  finite difference of channels we would already hold.

**Access decides the cost, by an order of magnitude.** The CDS
`derived-era5-single-levels-daily-statistics` API subsets server-side by
`area`: ~69 GB float32 for eight variables over the window, less compressed.
The anonymous ARCO-ERA5 zarr chunks one global slice per hour, so extracting
our 4 % of the globe still downloads the globe — **~1.56 TB per variable**.
**Get the free CDS account.** It costs minutes, saves an order of magnitude,
and it also unlocks ORAS5 for §3-B3. Net storage after retiring the four NCEP
channels: **+5 GB**.

### C2 · The overturning observations we already download and do not use

The RAPID distribution ships three files. The project uses one.
**`moc_vertical` is the overturning streamfunction as a function of depth** —
a depth-resolved *label*, which is exactly the supervision that pairs with
depth-resolved *state* from §3-B1, at zero marginal download. **`ts_gridded`**
gives observed full-depth T/S at the boundaries, which is how the GREP depth
channels get validated rather than trusted. Also already wired and worth
carrying as labels: MOVE 16 N (the longest deep-limb record, 2000→), the
Florida Current cable (**1982→, the only label spanning the whole axis**),
OSNAP, SAMBA.

**State plainly in any writeup:** there is no gridded *observational* product
below 2,000 m. Argo's core mission stops there; Deep Argo is regional pilots.
Every abyssal channel we add will be model output, and must be labelled as
such.

### C3 · Static fields — the highest information per byte in the document

One static channel is 135,161 × 2 B = **0.27 MB**. Ten cost 2.7 MB.

**Bathymetry** (`cmems_mod_glo_phy_my_0.083deg_staticbathy`, or GEBCO_2026,
public domain) and its **gradient**; an explicit **land/ocean mask** (so "land"
and "not observed" stop sharing a token); **Coriolis f** and **β**; **distance
to coast** and **to the 1000 m isobath**; **mean dynamic topography** (which
closes `sla` → `adt`). The western boundary, the Mid-Atlantic Ridge and the
Greenland–Scotland sills *are* the overturning's geometry, f/H contours are its
barotropic waveguide, and none of it is inferable from the channels we have.

### C4 · Looked at, and skipped

SMOS/SMAP sea-surface salinity (11–16 of 43 years — too short) · ocean colour
(no mechanistic path at these timescales) · 20CRv3 (early-period ensemble
spread exceeds the signal) · ECCO V4r4 as a *training* channel (0.5°, ends
2017) — though it is worth using as a **diagnostic**, since its adjoint says
which pixels the 26.5 N transport actually depends on, and that could justify a
small high-resolution window instead of a basin-wide one.

## 6. Storage, end to end

| configuration | fp16 | fits 100 GB? |
|---|---|---|
| current pentad tensor, 40 ch @ 0.25° | 34.0 GB | yes |
| same 40 ch re-gridded to 1/12° | **305 GB** | **no** |
| + GREP 4 vars × 8 levels @ 0.25° | +20.2 GB | yes |
| + GREP 4 vars × 16 levels | +40.5 GB | tight |
| + DUACS 6 ch @ 0.25° | +3.8 GB | yes |
| + ERA5 8 ch (net of dropping NCEP 4) | +5 GB | yes |
| + OSTIA sst & ice @ 0.25° | +0.85 GB | yes |
| + statics (10 ch) | +0.003 GB | yes |
| **− RG block moved to a 1° sidecar** | **−26.9 GB** | frees the budget |

**Target: 0.25°, pentad, ~55 dense channels + a 1° sidecar ≈ 55 GB** — deeper,
better forced, physically richer than today's, and *smaller* than today's plus
GREP because §2's waste is removed.

## 7. Ranked

1. **GREP 3-D at 0.25°, 4 variables × 8 levels.** Same product, credentials,
   grid and code path; two arguments change. +20 GB. Subsurface velocity and
   abyssal levels the tensor has never had, 11 extra years, 6× the live
   subsurface bins. Best ratio of new physics to new risk in the document.
2. **The statics, and RAPID's unused files.** ~3 MB and one already-made
   download. Bathymetry, slope, mask, f, β, distance-to-coast; `moc_vertical`
   as a depth-resolved label and `ts_gridded` to validate item 1 against.
   Afternoons, not projects.
3. **A CDS account, then ERA5 forcing.** Buoyancy forcing is entirely absent
   today and is the mechanism the subpolar precursor runs through. Net +5 GB,
   and the account also unlocks ORAS5 for item 5.
4. **DUACS + OSTIA at 0.25°.** +4.7 GB, existing credentials. The observed
   surface overturning signature, and the missing sea-ice channel.
5. **The backward extension to 1958 — as a measurement first.** ~3 GB. The only
   candidate that attacks the measured constraint, and cheap enough to test
   before committing: a 1° T/S-only tensor, a small-tier run, and the question
   is whether the step-2,000 wall moves. Whichever way it answers is the most
   valuable thing on this list.
6. **The 2025–2026 continuation** (GLORYS12 to 2026-06, OSTIA, DUACS, RG
   extensions). +4.5 % of end-bins — small, nearly free, and it is also what
   would let the terminal holdout run to 2026 instead of 2024.

**Skip:** MUR at 1 km · basin-wide GLORYS12 at 1/12° (test the 23–30 N strip
instead) · ARMOR3D at native 0.125° · SMOS/SMAP · ocean colour.

**And before any of it: §2.** Moving the RG block to a 1° sidecar frees 27 GB,
which is more than items 1, 3, 4 and 6 cost put together.

## 8. Unverified, flagged rather than guessed

- The GLORYS12 `my`/`myint` split date — the catalogue and the third-party
  documentation disagree. Resolve with `copernicusmarine describe`.
- The 50 GLORYS12 and 75 GREP depth *values* — counts and ranges are confirmed,
  the level lists are not. **Read them from the first served file; do not
  hardcode.**
- The CDS ORAS5 3-D dataset name and whether it arrives regridded or on the
  ORCA025 tripolar grid. If tripolar, `uo`/`vo` need rotation — the one
  candidate here where that gotcha genuinely bites.
- Latest RAPID and OSNAP releases (both form-gated): RAPID's README describes
  v2023.1 to 2023-02-11 while a v2024.1 is registered elsewhere.
- Licences: CMEMS requires registration and per-product citation; GEBCO is
  public domain; **EN4 is a non-commercial government licence** — read it
  before redistributing any derived tensor; RG-Argo requires citing Roemmich &
  Gilson (2009). Whether the project's existing published tensors carry these
  attributions has not been audited.
