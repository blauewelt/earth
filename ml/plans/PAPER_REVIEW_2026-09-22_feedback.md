**Review and handoff note for the agent developing the Earth prediction paper**

Reviewed 22 September 2026. Source: [Predicting any observed quantity of the Earth](https://blauewelt.github.io/earth/ml/paper/paper.pdf), dated 22 September 2026, 16 pages. PDF SHA-256: `78ca125e40c23c8a153117e685417ac60e881ca20749970b3c92e508e3f180a9`.

Purpose: improve the rate of scientific learning and eventual climate predictive skill, while keeping the next implementation small. This is a review of the design and evaluation logic, not a reproduction or code audit. I read the paper and inspected the abstract and relevant results/discussion of the [archived North Atlantic pilot](https://github.com/blauewelt/earth/tree/main/ml/paper/archive/v8-2026-09-18-north-atlantic-results). Findings below distinguish mathematical/protocol corrections from experiments whose outcomes remain unknown.

**Overall assessment**

The strongest elements are the observation/query interface, explicit footprints, separation of representation from forecasting, serious null ladder, and willingness to publish negative results. Preserve them. The pilot's disappointing forecast results are scientifically useful: they make the next question much sharper.

The design currently fixes too many architectural details before demonstrating the central benefit: does this representation improve forecasts of the physical quantities of interest beyond inexpensive models using the same information? A successful reconstruction model, a physically plausible sampling pattern, and billions of measurements do not establish that benefit.

My main recommendation is to bring a small end-to-end forecasting comparison forward, fix the evaluation contract first, and postpone most geometry refinement and scaling. This changes the order of work more than the architecture.

**1. Repair the evaluation contract before running comparisons — highest priority**

Sections 3.5 and 5.1 put 2023 in the development set and 2021–2024 in a supposedly once-only terminal test. Once 2023 informs architecture, training choices, or source admission, that test is no longer untouched. Training only through 2020 does not remove model-selection leakage. The archived pilot already reports development using 2023, so merely deleting 2023 from the new protocol cannot erase prior knowledge.

Action: record which periods have already influenced the project; label them development or retrospective evaluation. Reserve a genuinely unused period where available, or explicitly state that the remaining evidence is retrospective and register a prospective evaluation. Do not promise a pristine terminal test that no longer exists. Include every learned component in this rule: codecs, tile caches, LIM fitting, read-outs, loss weights, and calibration.

Separate two legitimate tasks:

| Evaluation | What may be used | What it establishes |
|---|---|---|
| Retrospective state forecasting | Declared retrospective analysis products with audited support | Skill conditional on that reconstructed state |
| Forecast as issued at time t | Only products actually available by t | Deployable forecast skill |

The existing bin-level exclusion is useful but insufficient. Section 3.1 assigns monthly Argo analyses to the pentad containing the 15th. The [RG product documentation](https://sio-argo.ucsd.edu/RG_Climatology.html) describes fitting profiles within a month, a multiyear mean/annual background, and monthly extensions released after the month. A midmonth timestamp therefore does not make the full field available midmonth. This is a concrete support/availability problem; its effect on current scores needs an audit, not an assumed numerical correction.

Add `support_start`, `support_end`, `available_at`, and product version to the source contract where relevant. Audit annual maps, retrospective smoothing, reanalyses, climatological backgrounds, and pretrained embedding caches similarly. Recomputing the final anomaly climatology on training years does not remove future information already embedded upstream.

Use lineage to group duplicate measurements and derived products during holdouts. A hidden Argo profile may still contribute to a gridded Argo analysis or an assimilating reanalysis; a hidden laser footprint may have supervised a canopy map/cache. This is not automatically invalid for forecasting later independent observations, but it invalidates a claim of independent reconstruction or source generalization if the held-out truth returns through another route. Test the specific claimed separation.

Clarify whether longitude holes exclude targets only, all local inputs, or underlying observations in derived products. These measure different kinds of generalization. Verify representative boundary cases deterministically; a large random anchor draw is a diagnostic, not a proof that no prohibited window exists.

Acceptance condition: one machine-readable split/source manifest, with issue-time and full-support checks, and explicit labels for retrospective versus as-issued scores.

**2. Move the simplest forecasting experiment ahead of learned geometry**

Section 5.2 puts geometry, task mix, aperture, maps, and hardening before the forecaster. This risks optimizing a codec property that never improves the downstream scientific task. The pilot reports near parity between raw and encoded current-state transport read-outs, and stronger simple forecast baselines at longer leads. That is a reason to test forecasting value early.

Start with existing data, fixed sampling, and the smallest practical model. Use a small common information set to compare:

| Arm | Question |
|---|---|
| Damped persistence and data-space LIM | What skill is already available cheaply? |
| Direct multilead regularized regression on compact raw/EOF features and selected lags | Does using history and fitting each lead directly explain the apparent gain? |
| Fixed codec plus embedding-space LIM | Does the representation improve useful compression? |
| Same codec plus LIM and a small nonlinear residual | Is there additional predictable nonlinear signal? |

Match forecast origins, available variables, history, targets, and preprocessing. Keep the existing analogue baseline where affordable. An anchor-only LIM is a valid baseline, but a history-reading neural model also needs a simple history-reading comparator before a gain is attributed to nonlinear dynamics or cone geometry.

Score in decoded physical space. Use SST at a few monthly/seasonal leads plus one subsurface quantity to begin; keep ENSO and transport as scientifically important secondary checks. Add one land task only when there is a sufficiently long usable record. For data-space comparisons, permit sensible compression rather than forcing a huge raw regression.

Only after this result, compare fixed geometry with per-group geometry at equal token/compute budgets. Per-channel maps should earn their additional estimation burden. If the codec does not improve forecasting, test a smaller codec or modest decoded forecast supervision before importing more sources.

**3. Treat the cone as a sampling prior, and fix training targets independently**

Sections 2.1–2.4 present the cone and mirrored future as a physical fact. Finite propagation motivates locality, but predictive relevance is not identical to direct causal influence. A remote observation can reveal a shared driver or hidden state without being a parcel travelling to the target. Coupled pathways, spatially varying flow, forcing, and observation density complicate one channel/one drift interpretations. A dissipative forward system does not imply a point-mirrored statistical dependency geometry.

Suggested wording: "We test a transport-inspired space–time sampling prior, supplemented by global context. Its learned geometry represents predictive relevance and is not interpreted as a recovered causal transport law."

Keep the ungated far field as a useful design choice, but remove the guarantee that 24 points exclude no driver: sparse global reach is not exhaustive information coverage. A compact coarse global summary is a reasonable equal-budget comparator, not a required new subsystem.

The existing stop-gradient, fixed evaluation geometry, and target-variance flag are good safeguards. However, stop-gradient does not stop the training target distribution from changing after geometry updates/rebaking. Target variance alone cannot detect a move toward easier, cleaner, or denser observations.

Simplest fix: draw target queries from a fixed distribution independent of learned input geometry, shared across arms and seeds. Learn where to read while holding what is predicted constant. This also makes the mirror an optional auxiliary objective and removes a substantial confound with less machinery. Compare the mirrored objective later if it buys skill.

Interpret drift agreement with known currents as a descriptive diagnostic. A disagreement could reflect useful information routing; agreement could reflect redundant source structure. Neither establishes causality. Avoid interpreting A2 beating A1 as proof that gradients "cannot find upstream"; it may reflect optimization, regularization, or initialization.

**4. Correct the linear backbone and test whether its restrictions help**

Equation 6 uses `G(l) = U diag(exp(-l/tau_j)) U^-1`. With positive finite time constants it has decaying eigenmodes, but unrestricted U does not guarantee contraction in the ordinary Euclidean norm. A non-orthogonal, ill-conditioned eigenbasis can permit transient amplification. State the norm if contraction is intended; otherwise call the backbone asymptotically stable under the stated assumptions. Stability of G also does not imply stability of the nonlinear residual model or its multiyear rollout.

There is a second issue: this real positive decay spectrum cannot reproduce a general LIM, which may have damped oscillatory modes. The claim that the step-zero forecaster equals the embedding-space LIM requires either a restricted LIM fit or an explicit projection and a measured projection error.

Lowest-cost next step: retain the fitted, regularized LIM unchanged and add a zero-initialized residual. Compare that with the proposed decay-only backbone. If guaranteed contraction is desired, use an appropriate constrained parameterization and state the tradeoff: suppressing transient growth or oscillation may remove useful predictability. A few damped 2×2 rotation blocks would accommodate oscillation if evidence warrants them, but need not precede the baseline experiment.

Report the backbone's decoded skill before training the residual, its finite-lead amplification, and whether the residual improves it. This is more informative than imposing a stability adjective on the equation.

**5. Align the objective and primary metrics with climate developments**

Dense-field scores have more information than a short transport series, but they can favor local, short-lead improvements while obscuring degradation in slow or spatially integrated quantities. More pixels do not create more independent ENSO events or overturning transitions.

Keep dense-field evaluation, with explicit physical area weights and fixed channel/region weights. Add a small predefined panel of decoded quantities at relevant leads: regional SST, upper-ocean heat content where coverage supports it, and an index such as Niño-3.4. For one supported event, score threshold probability, duration, or accumulated anomaly. Do not require an extensive event suite before the first result.

Distinguish proxies from the scientific quantity named. Surface CO2 error is useful but does not establish air–sea carbon flux skill; surface-current improvements do not establish overturning skill; static canopy-height accuracy does not establish vegetation-change prediction. Either score the named quantity with its observation/aggregation assumptions or label the result as a precursor/proxy. Preserve limited-power read-outs without letting an inconclusive result automatically veto a useful representation.

Section 2.6 trains primarily in latent space and adds decoded losses at three leads. A latent Euclidean metric depends on the codec's coordinate choices, and decoding the mean latent need not produce the mean physical forecast through a nonlinear decoder. Similarly, a read-out trained on true embeddings can encounter a changed input distribution at forecast time. Measure these effects using the complete forecasting pipeline. A small decoded loss at the primary scored leads is preferable to adding model size if these mismatches dominate.

Use exact area-weighted integration for quantities whose definition is known, such as Niño-3.4 or a specified heat-content integral. An attention read-out is appropriate when learning an observation mapping, but should not replace a known integral merely for interface uniformity.

**6. Make uncertainty claims narrower and check joint behavior directly**

Section 2.7 says an unsupported query will receive appropriately high variance because the model was never rewarded for confidence there. That conclusion does not follow: behavior outside supervised support is unconstrained. SST over land should be marked invalid; unfamiliar but physically valid queries should carry an explicit extrapolation/support flag. Neither case needs a new uncertainty model.

Variation across three to five training seeds measures some fitting variability, not the full conditional variability of the Earth system. Initially report it as seed sensitivity. A cheap forecast-distribution baseline is lead/region-conditioned residual calibration on a separate calibration split. Evaluate coverage and proper scores in physical units or with the stated transform consistently applied.

Shared noise can encourage coherent samples, but accurate marginal scores do not identify a joint distribution. [FGN](https://arxiv.org/abs/2506.10772) demonstrates good joint structure for its particular architecture and training setup; it is evidence for trying the approach, not a guarantee for this one. Score an area-average distribution and a temporal accumulation/event-duration distribution alongside marginal CRPS. Permuting ensemble members independently across cells preserves marginals while damaging dependence: this is a useful diagnostic showing that the chosen aggregate scores can detect the problem.

Specify whether Gaussian-head variance and seed spread are combined, and how latent uncertainty reaches the decoded query. Avoid counting the same residual variability twice. For transformed quantities, distinguish transformed mean, physical median, and physical mean. State the exact ensemble score estimator and its sampling assumptions when comparing unequal ensemble sizes.

**7. Distinguish initialized prediction from long-term climate projection**

The present objective supports testing forecasts from recent observations out to a year. Multiyear rollout is explicitly not the main objective. It therefore cannot yet support claims about scenario-conditioned climate change, tipping times, or century-scale behavior. The title's broad query interface should not be mistaken for demonstrated predictability.

Use a scope sentence such as: "The first experiments address initialized forecasts of observed Earth-system quantities over five days to one year; long-term forced responses and scenario projections remain unvalidated."

For seasonal/interannual evaluation, add a training-only evolving-climatology or trend-plus-anomaly baseline. Report skill with and without the estimated slow background, so trend reproduction and internal-variability prediction are visible separately. Do not simply remove all trends: the forced background is part of the scientific target.

If scenario-conditioned projection becomes an explicit goal, provide a forcing pathway and evaluate response under changed forcing. Longer rollout of an observation-conditioned model is not that experiment. A low-dimensional forcing/slow-state input may eventually be enough to begin testing this; a full new climate model is not the immediate requirement.

**8. Use the limited verification record more effectively**

Three starts per year under-sample seasonality and forecast-origin dependence. Once a model exists, evaluate more regularly spaced starts where inference cost permits. Do not count overlapping forecasts as independent. Use paired comparisons and blocks that preserve the relevant temporal dependence; report sensitivity to block length and the number of distinct years/events. With only a few years, a bootstrap cannot manufacture robust long-timescale evidence. Training seeds address optimization variability, not additional climate histories.

Before new comparisons, specify one primary weighted score and a practical improvement threshold, with a small set of non-degradation checks. Requiring A1 to beat A0 at every short lead and every seed is brittle and short-horizon biased. Define an equivalence margin: failure to detect a difference is not evidence that two designs are equal. If evidence is inconclusive, retain the simpler arm operationally while reporting the uncertainty.

Make evaluation outputs identical across arms. If fixed evaluation geometry means switching the learned model's input sampler back to A0, that tests behavior under an input change rather than its intended deployment. Freeze the query/target set; allow each trained model to use its intended admissible input sampler. Report a forced-common-input diagnostic separately if desired.

The FGN comparison is useful context, but equal output support does not equal equal input information. ERA5 instantaneous initial states and a pentad observation stream differ. Define the issue time at the end of the input averaging window, the exact valid interval for each forecast mean, and precipitation accumulation/averaging rules. Label an unmatched-information comparison as a system benchmark. Do not let it delay the internal climate-horizon experiment.

**9. Make data admission and scaling evidence-driven without overinterpreting null results**

The pilot supports concern about generalization; it does not uniquely diagnose data limitation. Width-insensitive performance and early validation minima can also reflect representation, objective, optimization, target noise, or rollout mismatch. A fixed 2,000-step cutoff depends on batch size and learning-rate schedule; it is not a portable test of overparameterization.

Before moving from millions to billions of parameters, compare two data extents at fixed size and two sizes at fixed data/compute. Record examples/tokens processed and validation learning curves. Treat the language-model-derived parameter ceiling and four-epoch rule as planning heuristics, not Earth-system scaling laws.

An observation count is not independent information. A long altimeter track can overwhelm a loss or nearest-neighbor set without adding independent climate regimes. Use platform/track-aware sampling and verification. Record equal-budget source ablations and the cost of each store. Where appropriate, inspect source benefits conditional on the companion variables that make the source useful.

The rule that every store must improve one of five read-outs or be dropped is too absolute when those read-outs have low power. Distinguish "useful," "not useful at this budget," and "inconclusive." Retain provenance while deferring expensive inconclusive sources; do not confuse absence of measured benefit with absence of information.

Likewise, the rule requiring a target channel to appear somewhere in every input prevents some important cross-variable inference tasks. Keep it as a copy/persistence control, then include a small explicitly scored task where the whole target channel is absent. Otherwise claims of general observation completion are narrower than they sound.

For the tile codec, same-place/different-date invariance can suppress precisely the changes the forecast should retain. Test sensitivity to change before adding a contrastive objective, or use augmentations that preserve the relevant state. Defer GBIF occurrence prediction until an observation-effort/absence interpretation exists; a generic Gaussian channel is not enough to distinguish species absence from lack of reporting.

**10. Resolve these implementation-facing ambiguities in the text**

These are checks for the next agent, not assertions that the code contains the same errors.

| Location | Issue and smallest correction |
|---|---|
| Figure 1, abstract, §2.6 | The figure shows ±143 pentads; text uses past lag 144 and future lead 73, while the caption suggests a 7–143 range. Make the drawing and one authoritative lag list agree. |
| §2.4 | A growth floor described as the design speed appears incompatible with much smaller initial growth. Specify units and distinguish a minimum radius, maximum reach, and default growth rate. |
| §§2.2, 2.5 | "No placeholders" conflicts with padded miss tokens/hidden markers unless their attention semantics differ. Specify whole-token padding, channel masking, and hidden versus never-observed entries. |
| §§2.5–2.6 | Stage 2 consumes mean/log-variance of z, but Stage 1 does not clearly define a stochastic latent distribution. Define it or use deterministic z initially. |
| §2.6 | Per-channel geometry cannot be applied to mixed latent coordinates without defining how channel-specific sampling produces/reads embedding tokens. Document the actual operation and token budget. |
| §2.5, Eq. 5 | A reference model's error is not irreducible error. Fixed inverse-reference-MSE weights are not generally equivalent to optimizing a sum of log current MSEs. State the actual objective; fit reference weights on a training calibration fold, freeze them, and cap extreme weights. |
| §2.6 | The MSSS identity assumes the relevant bias and variance normalization conditions. With nonzero mean bias, include its normalized squared contribution. Define whether ACC, amplitude, and MSSS share identical aggregation weights. |
| §§2.2, 2.7 | A scalar footprint scale cannot encode every anisotropic sensor support. Start with documented source-specific support assumptions. Test fine-to-coarse consistency where measurements genuinely average the same physical quantity; do not invent fine-scale information from the coordinate query alone. |
| §3 | "Nothing is resampled" conflicts with the stated chlorophyll block means and pentad aggregation. Say that sources need not share one common analysis grid, and document existing aggregation honestly. |
| §3.3 | "Nothing on land advects" is too categorical. Replace with a statement about the first model's assumed dominant land predictors; transport, dispersal, hydrology, disturbance, and management need not fit that assumption. |
| §4 | Verify cost with measured anchor throughput, encoder/decoder tokens, loader time, and cache construction. `6NT` with ambiguous T does not establish the stated GPU-hours for overlapping cone examples. |
| References | The pilot source is dated 18 September but reference [1] says 2 September. Check the exact relation of Eq. 4 to [Seitzer et al.'s beta-NLL](https://arxiv.org/abs/2203.09168); label this detached-mean variant accurately. Verify the citation for the precise fair-CRPS estimator rather than relying only on a general proper-scoring reference. |

**Suggested next-agent work order and deliverables**

1. Amend the scope, split contract, source availability rules, and mathematical claims. Preserve existing results and distinguish retrospective evidence from prospective commitments.
2. Implement a fixed-target benchmark with existing sources. Produce a single table for the four small forecast arms above, in physical units, on identical origins and targets. Include runtime and uncertainty on paired differences.
3. Produce one error decomposition: lead, season, region, slow background versus anomaly, and one scientific aggregate. Determine whether the limiting issue is representation, dynamics, calibration, or data coverage.
4. Run one equal-budget fixed-versus-per-group geometry comparison only if it addresses the observed limitation. Keep input learning independent of the target distribution.
5. Add one source family chosen to address a documented error, with a provenance-safe holdout and an incremental-cost comparison. Replicate a promising effect before scaling.

The next paper revision should make it possible to learn something important even if the neural residual, learned cone, or additional store fails. A smaller experiment that cleanly identifies where predictable signal is lost is more valuable now than completing the full architectural roadmap.
