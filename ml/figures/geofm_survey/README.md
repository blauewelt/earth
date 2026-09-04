# Geospatial representation-model survey (stencil level) — 31 Aug – 4 Sep 2026

A 52-slide deck comparing AlphaEarth, TESSERA, OlmoEarth, TerraMind, Prithvi-EO 2.0 and
IBM's Granite-Geospatial-Ocean at the level of *what area in space and what span of time feeds
one embedding*, then applying that lens to Earth 2: the velocity-specific dependency cone, a
worked example for the surface current, the four-sphere plan, head-to-head benchmark tables
transcribed from the papers, and related work; since 2 Sep also the *generic-embedding input proposal* (slides 34–50): the 13-rung input ladder, observed-vs-derived rule and leakage trap, cone families per input, the cone-native codec ("dots in, embedding out"), where velocity comes from, the zero-sum question, phases, 2026–27 data continuity; and since 3 Sep five slides about the data, the geometry and the other spheres, revised on 4 Sep with E-069's results and Chris's four cone-v2 decisions — **44**, two stencils, one cone: the E-069 geometry as built *and* the cone-v2 geometry beside it on one (lag, distance) map, with what five seeds said (H1 refuted — the 30-day history buys persistence and a 7–8 % better 1–2-pentad forecast, not velocity); **45**, the derivation graph of everything that feeds the cone codec today (four products, 42 channels) with the ranked ocean-and-biosphere additions beside it; **46**, the four boundaries the sampler meets (the window edge, land, the archive ends and the holdout shadow, and the calendar month), with the measured cost of each — plus the two DECISIONS that remove two of them: a global dataset, now stated as running to the POLES (721 × 1,440, −90 … 90°), and a day-of-year harmonic anomaly baseline (E-071 §1–2); **47**, the speeds the cone did not have and the speeds it will: a log-scale ruler of measured maxima (0.02 m/s deep-boundary-current advection to the Somali Current's 3.6 m/s, each with its source) against the single 0.3 m/s the cone implements, the design rule "maximum measured speed × 1.5" giving 5.4 m/s, reach-vs-lag out to the antipode, and Argo going from a zero-reach column to 26 profile tokens (E-071 §3–4); **48** (new on 4 Sep), *land, ice and air — nothing is dark by design*: the thirteen-row SHARED-CHANNEL table that answers Chris's *"why not add some data on land, too?"* — one quantity, one instrument, two surfaces, with the ocean and land/ice cells tinted so a shared row reads across both (skin temperature, near-surface air, air aloft, fluxes, frozen fraction, albedo, ASCAT σ⁰, MODIS reflectance, SMAP brightness temperature, GRACE, altimetry, velocity, the profile token) — beside three boxes: Antarctica was never unobserved, the GRID was (the grid now runs to −90°, 721 × 1,440, and what fills the ice-sheet rows); the cone families for the new spheres (atmosphere global-at-once, land at v = 0, ice-sheet column only, with the sphere code as a coordinate); and the cost plus the token change it forces (~163 GB of new channels, and a token = one (lag, location) carrying every channel, ≈ 300 per anchor) with Phase L0, the "sparse for now" six-channel set (E-071 §6); and **51**, El Niño 2026: what the 274-record open-data catalog can predict it with and what to add. Made at Chris's request across four sessions.

| file | what |
|---|---|
| `geospatial-representation-models-with-notes.pdf` | **Read this one.** 104 pages: every slide followed by a plain-English speaker-notes page that explains the terms and symbols (embedding, stencil, token, frozen probe vs fine-tune, mIoU/F1/RMSE, Δt, ℓ, τ, v, Ekman, thermal wind, RAPID, corridor AUC…) |
| `geospatial-representation-models.pptx` | The deck with the same notes in the speaker-notes pane. Fonts chosen to survive a Google Slides import (Georgia headings, Calibri body) |
| `GENERIC_EMBEDDING_INPUTS.md` | The in-depth proposal behind slides 34–50: input ladder with plain-English descriptions, cone parameters per input, the cone-native codec, the zero-sum analysis, phased plan, and a dataset spec sheet verified against agency pages on 2 Sep 2026 |
| `SURVEY_NOTES.md` | The running research note: verified stencil table, transcribed benchmark numbers, the cone model, the proposed pixel-year arm, related work — with sources |
| `build.js` | pptxgenjs generator for the deck (`node build.js`; needs `notes.js` beside it, writes `titles.json`) |
| `sources.js` | Verified URLs mapped to the 52 slides — appended to every note, clickable on the notes pages and the Sources slide |
| `slides_inputs.js`, `notes_inputs.js` | Slides 34–51 and their notes (including the data slides 45 and 51, the two geometry slides 46 and 47, and the land/ice/air slide 48), required by `build.js` / `notes.js` |
| `notes.js` | One plain-English note per slide, in deck order |
| `notes_deck.js` | Renders one landscape notes page per slide → `notes-pages.pptx` |
| `CONE_DATA_AND_ENSO.md` | Companion page to slides 45, 46, 47, 48 and 51: the dataset inventory of the cone codec with `file:line` references, the derivation graph as an edge list, the ranked additions with a paragraph each, the El Niño section (index table computed from `data/oisst_y`, the CPC status quoted, the catalog audit, and the three framings of "predict this year's El Niño"), and since 3 Sep two more parts — **5**, every boundary the sampler meets and the measured cost of each (with §5.7, added 4 Sep, on the fifth edge nobody had called one: land as a `miss` token in all 42 channels, and E-071 §6's answer), and **6**, the propagation speeds the cone can and cannot represent; both parts carry a *Decision, 4 Sep 2026* section (E-071 §1–2 and §3–4.5) and Part 6 also records where E-069's refuted H1 leaves the case for cone v2 |
| `../dependency_cone_explorer.html` | Interactive companion to slides 26–31 — open via the Pages URL |

**2 Sep 2026 — protocol note.** The paper was reset the same day (v8: every rolled number from
heads trained under the endpoint pool withdrawn; "corridor AUC" retired as a metric name). The
deck and `GENERIC_EMBEDDING_INPUTS.md` were corrected in the same commit: no withdrawn rolled
number is quoted, the Phase-0 experiment is scored under the corrected window-scope protocol
(MSSS per lead vs climatology and damped persistence, LIM null, n ≥ 3 seeds), and the
attribution-matrix probe contrast (0.672 vs 0.659) is stated as a single-seed number inside the
probe noise band (`ml/CLAUDE.md` §3b), i.e. consistent with parity.

Rebuild (both scripts write beside themselves, into this directory): `node build.js && node notes_deck.js`, convert both `.pptx` to PDF with LibreOffice,
then `qpdf --empty --collate --pages geospatial-representation-models.pdf notes-pages.pdf -- geospatial-representation-models-with-notes.pdf`.

Everything quantitative about *our* codec in the deck (attribution matrix 0.613/0.617 vs
0.659/0.672, stencil shapes) is quoted from `ml/paper/paper.tex` (v8 appendix, retained
probe numbers) and `ml/EXPERIMENTS.md`; the cone parameters (speeds, memories) are
order-of-magnitude illustrations and the four ablations are hypotheses, not results.
