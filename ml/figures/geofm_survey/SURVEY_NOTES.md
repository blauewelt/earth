# Geospatial representation-model survey (stencil level) — 31 Aug 2026

Deck delivered in session: `geospatial-representation-models.pptx` (v9, 34 slides + notes, fact-checked; build script `build.js`, pptxgenjs).
Purpose: compare AlphaEarth, TESSERA, OlmoEarth, TerraMind, Prithvi-EO 2.0 and IBM's Prithvi-based ocean model at the level of
*what area in space and what span of time feeds one embedding*, plus what each project actually releases.

## Stencil table (verified against papers / model configs)

| Model | Inputs | Spatial stencil | Temporal stencil | Output |
|---|---|---|---|---|
| AlphaEarth (DeepMind, arXiv 2507.22291) | S-2, Landsat 8/9, S-1, PALSAR-2, GEDI, ERA5-Land, GRACE, GLO-30, NLCD, text | 10 m output pixel; multi-scale attention, extent not disclosed | period summary; arbitrary valid window in model, calendar year in product (2017–2024) | 64-D unit vector, int8, per pixel-year |
| TESSERA (Cambridge, arXiv 2506.20380, CVPR 2026) | S-2 L2A 10 bands + S-1 RTC VV/VH | single 10 m pixel, no spatial context | 1 calendar year, 40 sampled dates/modality, DOY embedding (2017–2025) | 128-D int8 per pixel-year |
| OlmoEarth (Ai2, arXiv 2511.13655; v1.1 May 2026, v1.2 later 2026) | S-1, S-2, Landsat-8 monthly + static maps | 2.56 km tile = 256 px @10 m; 1–8 px tokens | ≤12 monthly steps over 1 yr, each step tokenised | 768-D (Base) / 1024-D (Large) per patch per step + pooled |
| TerraMind (IBM/ESA, arXiv 2504.11171) | S-2 L2A/L1C, S-1 GRD/RTC, DEM, RGB, NDVI, LULC, captions, geoloc | 2.24 km tile = 224 px @10 m; 16 px = 160 m tokens (14×14) | single co-registered timestamp | 196×768-D per modality; merged 768-D; generated modalities |
| Prithvi-EO 2.0 (IBM/NASA, arXiv 2412.02732) | HLS v2 30 m, 6 bands | 6.72 km tile = 224 px @30 m; 16 px = 480 m (300M) / 14 px = 420 m (600M) tokens | 4 dated frames 1–6 months apart, tubelet depth 1 | 1024-D / 1280-D per token per frame |
| Granite-Geospatial-Ocean (IBM + STFC Hartree + PML + Exeter, arXiv 2509.21273; github.com/ibm-granite/geospatial) | Sentinel-3 OLCI L2 16 bands + SLSTR SST = 17 ch @300 m | 12.6 km tile = 42 px @300 m; 2 px = 600 m tokens (21×21) | single acquisition (num_frames=1) | 512-D per token (embed 512, depth 12, 8 heads, ~50M params) |

## What each hands you
- AlphaEarth: precomputed global 10 m annual embeddings only (Earth Engine / GCS, CC-BY 4.0); weights + code unreleased. Land + coastal/inland water, no open ocean.
- TESSERA: weights CC0, code MIT, precomputed 10 m annual embeddings via GeoTessera (tile-sampled). Land only.
- OlmoEarth: open weights (4 sizes, custom licence with use restrictions); embeddings on demand via Studio, no global archive. Land-biased sampling.
- TerraMind: Apache 2.0 weights (tiny/small/base/large) + tokenizers; no precomputed embeddings. Land-stratified TerraMesh.
- Prithvi-EO 2.0: open weights 300M/600M ±TL; no precomputed embeddings; sea-only tiles removed from pretraining.
- Granite-Geospatial-Ocean: Apache 2.0 weights on HF + TerraTorch fine-tuning notebook; only model of the six trained on sea pixels — surface ocean colour + SST, one snapshot, fine-tuned to chl-a and primary production on ~100–200 in-situ points. Not an ocean-state model.
- Also checked: Prithvi WxC is atmosphere-only (no ocean downstream tasks found); NASA marine-debris work is not Prithvi-based.

## Takeaways recorded for Earth 2
1. Pixel vs patch: TESSERA shows zero-spatial-context per-pixel temporal encoders are competitive on land. (v1 suggested a 1×1 control; superseded — see v2 correction below: the paper already has it.)
2. Snapshot vs period: TerraMind/Granite = snapshot; Prithvi/OlmoEarth = stacked dated frames; AlphaEarth/TESSERA = year folded into one vector. Our stage-1 codec is a snapshot of a monthly/pentad mean; the year-summary design is untested on ocean fields.
3. Product vs model: nothing precomputed exists for the ocean; whatever we release would be the first embedding-field product for the ocean interior.
4. Gap: to our knowledge no published model embeds the gridded ocean state (T, S, SSH, currents) through time.

## v2 additions (same day)
- **AlphaEarth STP encoder** (slide 4): three operators per block — spatial ViT-like self-attention at ≈1/16 L, time-axial attention at ≈1/8 L (sinusoidal timecodes carry the 'valid period' t_s,t_e), 3×3 convs at ≈1/2 L — re-synchronised by learned Laplacian-pyramid rescaling after every block; input projectors downscale to 1/2 L; final learned resampling to 10 m; 64-D unit-norm, int8 in the product. Not stated: number of blocks, pyramid levels, frame size L → receptive field in metres cannot be quoted. Spatial attention is global within the frame.
- **Ocean-including representation models** (slide 13), ranked: Granite-Geospatial-Ocean (S-3 OLCI+SST, ocean-only); OceanSAR-1/-2 (Galeio, S-1 wave-mode SAR, ~12 M vignettes, DINO/DINOv2, 256 px @ 50 m, ResNet50 2048-D / ViT 384-D, Apache 2.0, arXiv 2504.06962 / 2601.07392); Copernicus-FM (TUM, 0.25° cells, 'whole land surface and near-land ocean' — partial); weather FMs carry ocean cells only as reanalysis background; location encoders effectively no. Forecast-only ocean systems (GLONET, WenHai, AI-GOMS, Xihe, ORCA-DL) publish forecasts, not embeddings.
- **How systems are compared** (slide 14): protocols = frozen kNN/linear probe (AlphaEarth k=1,3; OlmoEarth), frozen encoder + light decoder (PANGAEA UPerNet, TESSERA), full fine-tune (GEO-Bench ≤16 trials, ≥10 seeds, train ratios); benchmarks = GEO-Bench (12 tasks, accuracy / IoU, normalised bootstrapped IQM), PANGAEA (~11 datasets, mIoU/F1/RMSE, 9 GFMs vs UNet), model suites (AlphaEarth 15 evals; OlmoEarth 24/29; Copernicus-Bench 15); metrics = (balanced) accuracy, macro-F1, κ/BERκ, mIoU, R²/RMSE/MAE (log10 for biogeochem), error-reduction-vs-next-best %, win counts.
- **Correction**: the "1×1 pixel-only control" I proposed in v1 already exists in paper.tex (attribution matrix: single-pixel raw 0.613 / codec 0.617; 3×3 raw 0.659 / codec 0.672 — probe numbers, inside or at the edge of the seed noise band). What TESSERA adds that is untested is the *temporal fold*.
- **Proposed arm** (slide 16): pixel-year codec — p=1, encoder over 12 (or 24) pixel-months of one ¼° cell with month-of-year tokens → one z per pixel-year (≈310 tokens at 24 ch, same order as the 282-token month-block codec); Arm B swaps masked reconstruction for Barlow Twins on two random date-subsamples. Controls: mean of 12 monthly z (pooling), stage-2 over monthly z, raw 12-month stack into the same attention head; ≥2 codec seeds. Read-outs: ladder → RAPID r/RMSE, round-trip variance lost, rolled skill under the corrected window-scope protocol (MSSS per lead; 'corridor AUC' retired 2 Sep).

## v3 additions (same day)
- **STP walk-through** (slides 5–8, build sequence): illustrative L = 128 px @ 10 m (1.28 km), so ½ L = 64 px (20 m), ⅛ L = 16 px (80 m), ⅟₁₆ L = 8 px (160 m), N = 4 blocks. Ratios are the paper's; L and N are not stated.
- **Head-to-head tables transcribed** (slides 18–22), replacing the taxonomy-only metrics slide:
  - OlmoEarth Table 2 (frozen kNN/LP) and Table 3 (fine-tune) — OlmoEarth, TerraMind, Prithvi-EO 2.0 (+ TESSERA on CropHarvest-Togo 81.0). Frozen classification: OlmoEarth > TerraMind > Prithvi (So2Sat 68/47/35; EuroSAT 96/90/82); frozen segmentation split (TerraMind cashew 50.4, SA-crop 31.2; OlmoEarth PASTIS 51.8, MADOS 67.2). Fine-tuned: TerraMind-L ≈ OlmoEarth-L > Prithvi 600M by 1–3 pp; random-init row = pretraining worth +10 to +37 pp.
  - TerraMind Table 6 / PANGAEA (frozen + UPerNet mIoU): TerraMind-L 59.57 > TerraMind-B 58.35 > U-Net scratch 57.58 > CROMA 55.29 > … > Prithvi-EO 1.0 51.00.
  - TESSERA Table 1: TESSERA > AlphaEarth on 5/6 tasks (TreeSatAI F1 77.96 vs 76.90; Austrian crop F1@30 % 82.09 vs 56.36; Austrian seg mIoU 53.12 vs 25.70; BioMassters RMSE 27.43 vs 29.59; Borneo CHM 12.21 vs 16.11 m; AlphaEarth wins PASTIS-R 51.08 vs 50.68). Both ≫ frozen Prithvi.
  - OlmoEarth Table 7: fine-tuned OlmoEarth > AlphaEarth on 5/5 partner tasks (Nandi 82.2 vs 66.0, solar farm 86.7 vs 77.5); AlphaEarth cannot be fine-tuned. Independent LCZ Bern study (arXiv 2606.20034): TESSERA 0.90/0.82, AlphaEarth 0.89/0.81, S1S2 0.87/0.77 (acc/IoU).
  - GEO-Bench-2 (arXiv 2511.15658, independent, ranks of 14): Prithvi 600M-TL core 5, TerraMind-L 6, Prithvi 300M-TL 9, TerraMind-B 11; TerraMind-L rank 1 multi-temporal & multi-spectral.
  - Granite-Ocean Table 1: chl-a RMSE FM 0.14 vs scratch 0.16 vs RF 0.16; PP 0.39 vs 0.42 vs 0.40 (log10; gain inside fold spread). OceanSAR-1 kNN: ViT-B/8 83.6 % / 0.63 m / 1.37 m s⁻¹ vs best land-trained SAR FM 74.8 % / 0.78 / 1.95.
  - Structural caveats: no benchmark contains all six; the widest (OlmoEarth's) has five and omits AlphaEarth; only independent multi-model source (GEO-Bench-2) has two; no public benchmark has an ocean-state task.

## v4 addition — velocity-specific dependency cone (Chris's proposal, slides 25–26)
Model: for process p with signal speed v_p and memory τ_p, inputs to a prediction Δt ahead matter only inside C_p(Δt) = {(Δx, ℓ): Δx ≤ v_p (Δt+ℓ), Δt+ℓ ≤ τ_p}. A channel's cone is the union over the processes forcing it: C_SST = C_atm ∪ C_ocean (wide-shallow ∪ narrow-deep = L-shape, not a cone).
Illustrative (v, τ): synoptic atmosphere 10 m/s, ~10 d; surface currents / mixed layer 0.1–0.3 m/s, 3–12 mo; interior & midlatitude Rossby 0.02–0.05 m/s, years; Kelvin/coastal waves 1–3 m/s along waveguides (anisotropic).
Our stencils on that map: ring-8 @222 km per monthly step = 0.085 m/s (ocean advection); sunflower-89 @4444 km = 1.7 m/s (fast-waveguide speed, inside the atmospheric cone at Δt = 1 mo); pentad codec at the same reach = 10 m/s per step.
Key inversion at monthly Δt: τ_atm < Δt, so monthly wind is not self-predictable; its useful cone is the ocean's (via SST/teleconnections) — the atmosphere is a spatially-correlated stochastic forcing at our cadence.
Predictions (ablations, none run yet): (1) reach can be cut to ~500 km for interior T/S slots without loss of rolled skill (MSSS per lead under the corrected protocol; 'corridor AUC' retired 2 Sep), not for wind/SST slots — the answer to the open 'can reach come down?' question in dependency-cone-and-window.md §5; (2) wind slots beyond lag 1 add nothing, interior slots gain from 24→48 mo; (3) a cone-shaped stencil (reach ∝ Δt+lag) matches the 4444 km cylinder with fewer slots; (4) pentad stage-2 shows one-step wind skill monthly cannot.
Caveats: v = fastest relevant signal (waves), not mean flow; τ truncates by predictability, so wind's area is v·τ not v·horizon; cones are tilted (upstream, westward Rossby, along-boundary); eddy diffusion √(Kt) ≈ 50 km/mo negligible at ¼°.

## v5 additions — four spheres and related work (slides 17, 18, 28, 29)
- Slide 17 retitled "Who embeds the ocean — and how deep?": the published ocean encoders (Granite, OceanSAR) are surface-only; Earth 2 row added — gridded ocean state incl. Argo T/S at depth, the only embedding model with the interior.
- **Four spheres** (Chris's framing): a) atmospheric weather, b) ocean weather (incl. interior; much larger heat capacity), c) ocean biosphere, d) land biosphere. Coverage today: a) forecast FMs only; b) surface by OceanSAR/Granite/AIFS-ocean 2026, interior only by Earth 2; c) Granite + task models (SOM-FFN, CSIR-ML6, CANYON-B); d) the whole mainstream. Nobody represents two spheres jointly.
- **Sequencing (agreed)**: 1 ocean interior (anchor) → 2 + ocean colour (same grid, Granite recipe, MLD/light/SST coupling) → 3 + atmosphere as forcing (already in channels; wide-shallow at monthly cadence) → 4 + land biosphere (last; slow coupling).
- Design consequences: per-sphere (v, τ) cones with couplings as L-shaped unions; period tokens for biospheres vs dated-state tokens for physics; masked-vs-never-measured token as the primitive for heterogeneous observing geometries.
- **Related work, cone idea**: precedents for 'speed × time' — OceanNet (Chattopadhyay 2024: Rossby ≈1 cm/s at 40°N → eddies travel 30–120 km over the horizon, used to drop lateral BCs) and PARADIS (Pereira 2026, arXiv 2601.21151: neural semi-Lagrangian gather at x−Δt·u, CFL→Lipschitz, 20-param layer beats U-Net receptive field); 'one medium as forcing' standard in Samudra/ACE2-SOM; LAMs use fixed halos (Neural-LAM 10 nodes); global models same reach for all channels; per-variable tokenisation ≠ per-variable extent. Not found: τ-truncation + union → per-channel stencil as a stated principle (hypothesis, not result).
- **Related work, four spheres**: nearest physics lineage Samudra → SamudrACE (ocean-interior emulator first, atmosphere later; emulates a simulator); AIFS surface ocean (ECMWF 2026, arXiv 2604.25559) forecasts atmosphere + SST/SSS/SSH/currents/waves/ice jointly with larger loss-scaling for slow ocean/ice fields; Aurora = one backbone, sphere fine-tunes; observation-native Aardvark/GraphDOP/4DVarNet; ESFM (2026, arXiv 2605.00850) and Copernicus-FM = heterogeneous sources + missingness tokens but no ocean; biosphere ML = task models; coupled DA (Penny & Hamill 2017) names the timescale problem. Name collisions to disambiguate: NVIDIA 'Earth-2', Aurora's 'A foundation model for the Earth system'.

## v6 — Google Slides compatibility (format only, no content change)
Heading font Cambria → **Georgia** (Cambria is Microsoft-only and gets substituted on import to Google Slides; Georgia is in both Office and the Google Slides font list). Body stays **Calibri**, which Google Slides does have — important because every dense table is Calibri and a substitution there would reflow them. `title()` now sizes by string length (30/26/23/21 pt) so long titles stop wrapping into the subtitle — that was a real defect on ~6 slides.
Import path: Drive → upload → open with Google Slides → File ▸ Save as Google Slides (or File ▸ Import slides from an existing deck). No Google Slides connector exists in the registry, so the deck cannot be authored natively from here; Google Drive is connected at org level but toggled off for this chat.

## v7 — worked example + Dependency-Cone Explorer (slides 28–31; artifact "Dependency-Cone Explorer")
Target: surface current (u, v) at a ¼° pixel, Δt = 1 month. Drivers with illustrative (v, τ, L_corr): wind stress (10 m/s, 10 d, 1,500 km; Ekman; τ<Δt ⇒ lag 0 only), air temperature/heat flux (fast driver integrated by the mixed layer ⇒ τ_ML 6 mo, reach = 1,500 km + 0.15 m/s·(Δt+ℓ)), ocean surface T (0.15 m/s, 8 mo; thermal wind ⇒ needs 3×3), interior T/S (0.03 m/s, 5 yr), SSH (0.03 m/s Rossby + 2.5 m/s Kelvin arm, 12 mo), own history (0.15 m/s, 3 mo).
Rules: reach(ℓ) = min(max(v·(Δt+ℓ), L_corr), 10,000 km); lag useful iff Δt+ℓ ≤ τ (lag 0 always). Slots(r) = clamp(round(89·(r/4444)²), 9, 89) on the sunflower-89; cylinder = 89 × 24 = 2,136 per channel.
Budgets: wind 89 · air T 232 · surface T 157 · interior 238 · SSH 108 · own 27 → 851 vs 12,816 (93.4 % fewer). Union in (lag, distance) is an L: lag-0 spike (atmosphere), shoulder to 6–8 months (mixed layer), low tail to 24 months (interior ≤ 1,900 km).
Explorer artifact (published, private): editable v/τ/L_corr per driver, Δt 5 d / 1 month, presets (current / SST / interior T), anisotropy sketch, Play (lag sweep) and Roll (h = 1…12 recursion), wedge chart + map view with the 89 sunflower slots + slot-budget bars. Source: deck/cone-explorer.html (36 KB, vanilla JS/SVG).
Ablations specialised to this target (hypotheses): interior slots at 500 km ⇒ no loss; wind/heat-flux lags > 1 ⇒ no loss, same cut on surface T ⇒ loss; cone stencil 851 vs cylinder 12,816 ⇒ equal rolled skill; pentad Δt ⇒ one-step wind skill (all untested; scored under the corrected protocol only).
Drive: the deck (2.8 MB) is too large for the connector's inline upload; convert manually (Drive → open with Slides → Save as Google Slides).

## v8 — formatting pass + plain-English speaker notes (1 Sep)
- Formatting: full visual QA over all 34 slides (subagent + manual). Fixed: title slide text running under the motif (text column now 8.5 in, title 32 pt single line); tick-less timeline rows removed on snapshot slides (TerraMind, Granite) so the marker line no longer strikes through a caption; two titles shortened; the 24-month marker on the cone slide shortened so it clears the INTERIOR label; label halos on the cone slide. Second QA pass: 34/34 clean.
- Speaker notes: `deck/notes.js` holds one plain-English note per slide (terms and symbols explained: embedding, stencil, token, frozen probe vs fine-tune, mIoU/F1/RMSE, Δt, ℓ, τ, v, Ekman, thermal wind, Rossby/Kelvin, RAPID, rolled skill…). `build.js` replaces the old technical notes with these in the PPTX notes pane and writes `titles.json`.
- PDF with notes: `notes_deck.js` renders a landscape notes page per slide (two balanced columns) → `notes-pages.pdf`; `qpdf --empty --collate --pages slides.pdf notes.pdf -- out.pdf` interleaves slide N / notes N → `geospatial-representation-models-with-notes.pdf` (68 pages). Plain `geospatial-representation-models.pdf` (34 pages) also delivered.
- Rebuild recipe: `node build.js && node notes_deck.js && soffice → pdf (both) && qpdf --collate`.

## Published on GitHub (1 Sep 2026, commit e1188b20 on blauewelt/earth main)
- Explorer (live): https://blauewelt.github.io/earth/ml/figures/dependency_cone_explorer.html
- Deck folder: https://github.com/blauewelt/earth/tree/main/ml/figures/geofm_survey (PDF with notes, PPTX, build.js, notes.js, notes_deck.js, SURVEY_NOTES.md = this note)
- Rows added to ml/figures/README.md. gh-pages fast-forwarded (force:false). Deploy job green; Playwright test job pending at time of writing (previous commit's test job had failed before this change).

## v9 — fact-check pass (1 Sep 2026, commit 1ffc3cc on blauewelt/earth)
Six parallel checks against primary sources; ~360 benchmark cells re-read from page images, 0 table mismatches, 1 summary-line slip. Corrections applied to deck + notes:
- **AlphaEarth frame size IS published** (supplement S2.1/S8.2): 128 × 128 px = 1.28 km frames for training and inference (960 m tiles + 160 m buffer, outer 80 m trimmed); 15 STP blocks, widths D_S 1024 / D_T 512 / D_P 128 (S2.4); vMF κ = 8e3. Receptive field ≤ ~640 m from any pixel. Earlier statements "L and N not stated" were wrong (main text only). Slide 14 point moved to 1.28 km; summary table updated.
- Removed unprintable numbers: AlphaEarth LCMAP 92.4 %, Prithvi GEO-Bench 75.6 %.
- Granite: RMSE loss; 6-day window centred (~±3 d); IBM Research Europe; 512k vs 470k+50k inconsistency shown; chl-a spread 0.03–0.10.
- TerraMind: tiny/small Jun–Jul 2025; Large 1024-D; Univ. of Iceland; no DeCUR in PANGAEA claim. OlmoEarth: token merging v1.1, RoPE v1.2.
- Physics: Rossby at 10° ~0.3 m/s; own-history τ = Eulerian decorrelation (eddies live 4+ months); additive reach rule for air T stated on slide 28; Earth 2 stencil ~7× (not 10×) wider than Granite's tile.
- Head-to-head wording: Prithvi 8–13 pp behind on PASTIS/MADOS; OceanSAR wind-speed error −30–45 %.
- Verified as correct (no change): all OlmoEarth/TerraMind/TESSERA/Granite/OceanSAR/GEO-Bench-2/LCZ table cells; TESSERA CC0 weights; Aurora 1.5 exists (MSR, 2026); heat-capacity 2.5 m; Ekman/thermal-wind/geostrophy formulas and signs; slot arithmetic 851 vs 12,816.

## v10 — the generic-embedding input proposal (2 Sep 2026, commit a68f364 on blauewelt/earth)

Twelve slides added before Sources (34–45; 46 slides, 92-page with-notes PDF), backed by the in-depth note
`ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md` (project copy: `claude/generic-earth-embedding-inputs-2026-09-02.md`):

- 34 the question + ladder at a glance and the three tests for a derived product · 35–37 the 13-rung input ladder in three tables
  (plain-English content, O/D, "adds") · 38 observed / derived / derivable and the leakage trap (DUACS delayed-time ±6-week window, NRT past 7
  weeks; GLORYS 7-day cycle; ERA5 12 h; → time-shift derived dots, train the forecasting objective on NRT streams, two embeddings per anchor
  like ERA5T/ERA5) · 39 five cone families with glyphs (A fast-wide-short, B slow-narrow-long, C L-shaped, D column-only land, E static) + depth
  axis · 40 the cone-native codec (dot sampler → tokens → Perceiver-style encoder → e_dyn; masked dots incl. future) · 41 where velocity comes
  from (Emery et al. 1986 MCC; flow as maskable channels; advected-sampling prior; velocity probe) · 42 zero-sum pros/cons table · 43 the
  asymmetry (B contains A; data-processing inequality), the physics-vs-task split, the local-codec → cone-codec → thin-stage-2 hybrid ·
  44 Phase 0 + phases 1–5 + ablations · 45 data continuity 2026–27.
- Dataset specs verified 2 Sep 2026 against agency pages (four parallel passes; spec sheet = Appendix A of the note). Notable: S1A terminated
  29 Jun 2026 (S1C+S1D); S2 three-satellite until end-2026; MODIS retiring late 2026/27, SNPP data end 1 Nov 2026; SSMIS→AMSR2/AMSR3
  hand-over; DUACS L4 now 0.125°; OceanOPS 2 Sep: 4,372 Argo (219 deep, 989 BGC), 1,317 drifters; ERA5 SST/sea ice are prescribed inputs;
  ERA5-Land has no DA; Copernicus waves are MFWAM; Copernicus carbon is 0.25°; OPERA CIRRUS 1 km/5 min.
- Fact-check of 15 method/physics claims: 13 verified, 2 corrected (DUACS window; SST anomaly memory 3–6 months per F&H 1977 / Deser 2003).
- **Aligned with the paper reset of the same day (v8, commit a17b189):** every rolled number from heads trained under the endpoint pool is
  withdrawn and "corridor AUC" is retired. The deck no longer quotes any of them (slides 25, 27, 31 reworded; new slides never did); Phase 0
  is scored under the corrected window-scope protocol (MSSS per lead vs climatology and damped persistence, trained/held-out longitudes
  separately, block-bootstrap intervals, LIM null in pixel and embedding space, n ≥ 3 seeds; probes are diagnostics, not verdicts); the
  attribution-matrix contrast 0.672 vs 0.659 is stated as a single-seed probe number inside the §3b noise band (parity is the honest reading).
- Links: deck README https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/README.md · proposal
  https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md · with-notes PDF
  https://github.com/blauewelt/earth/blob/main/ml/figures/geofm_survey/geospatial-representation-models-with-notes.pdf

## v11 — name and provenance clean-up (2 Sep 2026)

- The account nickname removed from every deck source, note, README and the explorer's provenance comment; where a name is
  needed the deck says "Chris". Nothing else changed on those slides.
- Second pass against the reset paper's appendix ("claims of the earlier versions and their status"): the last laundered readings are
  gone — the takeaways slide no longer states the +0.05 / +0.01 attribution-matrix effects as levels (probe numbers, seed spread 0.04–0.25),
  the RAPID-cone note no longer cites the pooled-vs-unpooled gap as evidence (a two-interval comparison, `ml/CLAUDE.md` §3), and the
  "pentad stage-2 shows one-step wind skill" line is now a prediction marked untested (the pentad rolls were withdrawn). The one Earth 2
  result the deck still quotes is the retained probe contrast (0.672 vs 0.659), always with its caveat.

## v12 — sources on every slide (2 Sep 2026)

- Chris asked whether the paper links were in the slide notes; they were not (prose only; the Sources slide had plain-text short URLs).
  Now `sources.js` holds 161 verified full URLs mapped to the 46 slides; every speaker note ends with a "Sources for this slide" block
  (plain text in the PPTX notes pane), every notes page in the with-notes PDF carries the same links as clickable labels (355 link
  annotations), and every URL / DOI on the Sources slide is clickable (70 links). A subagent resolved all 161 URLs by fetch before use;
  four were corrected (Copernicus carbon product id, MOD16A2GF page, Black Marble, RGI user guide).
