# E-080 · Should the learnable cone be per channel or per channel group?

*Research note, 19 September 2026. Chris's question: "Should the learnable
cone shape vary with every channel, or all channels in the ocean group at
once (but what about deep currents going the other way)? I'm inclined to
leave things per channel — speed, direction, … learned per channel."*
Nothing here is measured; it is the argument and the recommendation, and the
paper (`ml/paper/paper.tex` §3) now states the design this note recommends.

## 1 · What E-080 revision 3 chose, and why

Slide 9: the drift **d** and the ellipses Σ₀, Σ_v are shared per channel
*group* (ocean surface / atmosphere / land) and per 1° cell; only a scale s_c
and a memory τ_c are per channel. The stated reason: everything the water
carries is carried the same way, so one arrow per flow; per-channel arrows
would multiply parameters and let the model learn different "upstreams" for
quantities that arrive from the same place. Depth was handled by dropping the
Argo ellipses (revision 2): profile tokens sit where floats are, and
depth-dependent sourcing was left to attention, checked as a diagnostic (R4).

## 2 · The physics says the group's channels do NOT share a carrier

Within the ocean-surface group of family 7.2 (currents, SSH, SST, sea ice,
mixed-layer depth, and colour beside it), the thing that moves each signal is
different, and often points a different way from the mean current:

| channel | what carries its anomalies | direction relative to the surface current | speed |
|---|---|---|---|
| `cur_u`, `cur_v`, `cur_speed` | the flow itself (advection, boundary currents) | along the current | 0.1–3.6 m/s |
| `ssh` | mesoscale eddies and Rossby waves | **westward regardless of the current** — nonlinear eddies "propagate nearly due west" at close to the long baroclinic Rossby phase speed, cyclones deflecting poleward and anticyclones equatorward ([Chelton, Schlax & Samelson 2011](https://www.sciencedirect.com/science/article/abs/pii/S0079661111000036)); ~10 cm/s or less, and more than 2× faster than linear theory above 35° ([Chelton & Schlax 1996](http://www-po.coas.oregonstate.edu/research/po/research/rossby_waves/chelton.html)) | 0.02–0.1 m/s |
| `sst` | at lags 0–1 the atmosphere (fluxes, Ekman drift, downwind); at longer lags the mean flow — SST anomalies propagate along the Gulf Stream / North Atlantic Current over years ([Sutton & Allen 1997](https://www.nature.com/articles/41523)) | downwind first, then along the current | two regimes |
| `sea_ice` | the wind: ice moves at ~0.8–2 % of the geostrophic wind speed, 5–30° to its right, with correlations 0.6–0.8 across the Arctic (Thorndike & Colony 1982 as quoted in [Polar Research](https://polarresearch.net/index.php/polar/article/download/3370/11914)) | with the wind, not the current | ~0.1–0.2 m/s |
| `log_mld` | local: wind stirring and buoyancy flux | ≈ no drift | — |
| `log_chl` | advected by the flow, grown locally, mixed by wind | along the current, weakly | — |
| `rg_t*`, `rg_s*` at depth | the deep flow: at 26.5° N the Antilles Current runs **north** in the upper 800 m and the Deep Western Boundary Current **south** from 800 to 4,800 dbar at 10–15 cm/s ([Meinen, Garzoli et al.](https://www.aoml.noaa.gov/phod/wbts/publications/meinen_garzoli_dsr.pdf)) | reversed below ~800 m at the western boundary | 0.02–0.15 m/s |

So "one direction per flow" is right only if the group is defined by the
*carrier*, and the carriers cut across the file groups: SSH's carrier is the
Rossby/eddy field, sea ice's is the wind, SST's changes with lag, and the
interior's reverses with depth. A shared group arrow would average a
south-west (current) arrow with an eastward (SSH source) and a downwind (ice,
SST) arrow, and at the western boundary a northward upper-limb arrow with a
southward deep one. The average is the drift ≈ 0 that the 10° map was
rejected for.

## 3 · Prior art: shared departure points, per-variable extents unmeasured

- PARADIS ([Pereira et al. 2026](https://arxiv.org/html/2601.21151v2)) learns
  one velocity field per location, shared across a compressed set of
  "transportable modes"; it does not learn a departure point per variable.
- Chen et al. ([arXiv 2405.06590](https://arxiv.org/html/2405.06590))
  decompose forecasting into advection and convection with an attention
  network whose weights are shared across variables.
- Deformable DETR learns offsets per query and per attention head — many
  arrows per location, none tied to a physical variable.

No published Earth-system model learns a per-variable receptive-field shape;
the closest is per-variable *tokenisation* (Aurora, ClimaX), which changes
what a variable is embedded as, not where it is read from. Per-channel
geometry is therefore a hypothesis this programme would test, not a practice
it would import.

## 4 · What per-channel costs, and what it does not

- **Tokens: nothing**, if the per-channel geometry lives in the aperture.
  A per-location token carries every channel of its group; the aperture
  weight w_{c,i,ℓ} already takes a per-channel Σ_c(ℓ); giving it a per-channel
  centre ℓ·**d**_c as well is the same elementwise multiply. The dots are
  shared, the reads are unchanged, the loss reaches **d**_c and S_c through
  every dot at once (slide 10's argument holds channel by channel).
- **Parameters: ≈ 65 channels × 8 numbers × 64,800 cells ≈ 34 M nominal**,
  about half live, against 1.6 M per group. Still no compute. Data per number
  is unchanged — each channel's numbers are informed by that channel's
  contribution at every anchor — so the 50 k anchor-pentads per 1° cell
  argument of slide 12 holds per channel. Cells where a channel is sparse
  (colour under cloud, ice outside the ice zone) are exactly where the
  coarse-plus-residual prior does the work.
- **Sampling** is the one place per-channel is not free. The hardened dot
  table is one sunflower per group and lag; if channels' ellipses diverge,
  one sunflower cannot sit inside all of them. Two answers, cheap first:
  (i) the gate phase already reads a fixed log-radial candidate set (36 per
  lag) that covers every bearing, so per-channel geometry learns there
  without any change; (ii) at hardening, draw the group's 24 points per lag
  from the **mixture** of its channels' Gaussians (equal weights, or weights
  ∝ each channel's loss share) rather than from one ellipse, and raise the
  budget to 36 if the mixture is multimodal. Per-channel sunflowers (24 dots
  × C channels) would multiply the group's tokens by C and are not proposed.
- **The depth column needs no sampling ellipse, and gets its direction
  anyway.** Profile tokens (family 8, tier P) sit where floats were; the
  k-nearest search is unchanged (revision 2 stands: no Argo ellipse to
  *sample*). But the aperture is a weight over any token whose offset is
  known, and a profile token's (Δx, Δy, Δt, depth) is known — so each
  pressure-level channel gets its own **d**_c and Σ_c as a weight over the
  same k tokens. Deep levels can learn a northward source at the western
  boundary and shallow levels a southward one, from the same tokens,
  without one more read. The gridded interior (`rg100`, read through E-071
  §3's 12-dot sunflower at two live lags) samples from the mixture like the
  surface group.
- **Identifiability and noise.** Two channels of the same carrier learning
  two arrows is not a failure, it is a measurement (if `cur_u` and `ssh`
  agree, fine; if they disagree, §2 says they should). The risk is noise in
  channels with weak gradient signal (`log_mld`, `chl_cov`); the fix is
  hierarchical: **d**_c = **d**_group + δ_c and S_c = S_group + ΔS_c, with the
  per-channel residual under L2 shrinkage toward the group. A channel that
  shares the flow costs nothing extra; a channel that does not pays for its
  departure with evidence. This is slide 12's coarse-plus-residual prior
  applied along the channel axis instead of the space axis.
- **Cheating guards** are unchanged (stop-gradient on targets, mirror tie,
  frozen eval geometry, aperture never in the loss weights, ring ungated);
  the mirror tie is per channel (a channel's future cone is its own past cone
  reflected), which is also physically right: SSH targets sit west of the
  anchor, current targets downstream, ice targets downwind.

## 5 · Recommendation

**Per channel, hierarchically.** All eight numbers (**d**, Σ₀, Σ_v) plus s_c
and τ_c per channel, stored as a group value plus a per-channel residual
under shrinkage, applied through the aperture over shared dots; hardening
samples from the group's mixture. The group arrow becomes a prior, not a
constraint. The Argo decision of revision 2 stands: no sampling ellipse for
profiles, but the per-level aperture over the k nearest profiles.

**What changes in E-080's experiment.** One arm is added: **A1g** (learned,
per group — revision 3 as written) beside **A1** (learned, per channel,
hierarchical) and **A2** (primed). Verdict on R1 paired by seed as before.
R3 becomes per channel with pre-registered signs: the current channels' drift
against −**u**_clim (positive); `ssh`'s zonal drift positive (source to the
east, Rossby/eddy propagation westward) where |**u**_clim| is small;
`sea_ice`'s drift against the reversed 10 m wind rotated 5–30° (positive);
`sst`'s lag-0–1 aperture wide and its drift against the reversed wind at
short lags. R4 becomes a **geometry** check as well as an attention check:
at the anchor north of the 26.5° N array, the drift of the ≥ 1,000 dbar
temperature channels points north and that of the ≤ 400 dbar channels
south-west; and over the world ocean the deep channels' drift correlates
with the reversed GLORYS velocity at their own depth (GLORYS carries the
velocities; the tensor holds only the surface, so the check reads the
reanalysis directly). If A1 beats A1g on R1 at leads ≤ 3 at all seeds, per
channel is adopted; if they tie, the shrinkage has collapsed the residuals
and the per-group form is kept as the cheaper equal; if A1g wins, the
per-channel residuals are noise and the prior weight is raised before the
form is abandoned.

**What not to do.** Do not sample per-channel sunflowers; do not add an Argo
sampling ellipse; do not let the per-channel apertures into the loss
weights; do not read the ice or SST arrows as currents — their carriers are
the wind and the fluxes, and that is the point.
