# Geospatial representation-model survey (stencil level) — 31 Aug – 3 Sep 2026

A 51-slide deck comparing AlphaEarth, TESSERA, OlmoEarth, TerraMind, Prithvi-EO 2.0 and
IBM's Granite-Geospatial-Ocean at the level of *what area in space and what span of time feeds
one embedding*, then applying that lens to Earth 2: the velocity-specific dependency cone, a
worked example for the surface current, the four-sphere plan, head-to-head benchmark tables
transcribed from the papers, and related work; since 2 Sep also the *generic-embedding input proposal* (slides 34–49): the 13-rung input ladder, observed-vs-derived rule and leakage trap, cone families per input, the cone-native codec ("dots in, embedding out"), where velocity comes from, the zero-sum question, phases, 2026–27 data continuity; and since 3 Sep four slides about the data and the geometry — **45**, the derivation graph of everything that feeds the cone codec today (four products, 42 channels) with the ranked ocean-and-biosphere additions beside it; **46**, the four boundaries the sampler meets (the window edge, land, the archive ends and the holdout shadow, and the calendar month — the one edge that is NOT masked), with the measured cost of each; **47**, the speeds the cone does not have (a log-scale speed axis from 0.02 m/s deep-boundary-current advection to 10 m/s wind stress, against the single 0.3 m/s the cone implements, plus the Argo channels' zero spatial reach and three ranked proposals); and **50**, El Niño 2026: what the 274-record open-data catalog can predict it with and what to add. Made at Chris's request across three sessions.

| file | what |
|---|---|
| `geospatial-representation-models-with-notes.pdf` | **Read this one.** 102 pages: every slide followed by a plain-English speaker-notes page that explains the terms and symbols (embedding, stencil, token, frozen probe vs fine-tune, mIoU/F1/RMSE, Δt, ℓ, τ, v, Ekman, thermal wind, RAPID, corridor AUC…) |
| `geospatial-representation-models.pptx` | The deck with the same notes in the speaker-notes pane. Fonts chosen to survive a Google Slides import (Georgia headings, Calibri body) |
| `GENERIC_EMBEDDING_INPUTS.md` | The in-depth proposal behind slides 34–49: input ladder with plain-English descriptions, cone parameters per input, the cone-native codec, the zero-sum analysis, phased plan, and a dataset spec sheet verified against agency pages on 2 Sep 2026 |
| `SURVEY_NOTES.md` | The running research note: verified stencil table, transcribed benchmark numbers, the cone model, the proposed pixel-year arm, related work — with sources |
| `build.js` | pptxgenjs generator for the deck (`node build.js`; needs `notes.js` beside it, writes `titles.json`) |
| `sources.js` | Verified URLs mapped to the 51 slides — appended to every note, clickable on the notes pages and the Sources slide |
| `slides_inputs.js`, `notes_inputs.js` | Slides 34–50 and their notes (including the data slides 45 and 50 and the two geometry slides 46 and 47), required by `build.js` / `notes.js` |
| `notes.js` | One plain-English note per slide, in deck order |
| `notes_deck.js` | Renders one landscape notes page per slide → `notes-pages.pptx` |
| `CONE_DATA_AND_ENSO.md` | Companion page to slides 45, 46, 47 and 50: the dataset inventory of the cone codec with `file:line` references, the derivation graph as an edge list, the ranked additions with a paragraph each, the El Niño section (index table computed from `data/oisst_y`, the CPC status quoted, the catalog audit, and the three framings of "predict this year's El Niño"), and since 3 Sep two more parts — **5**, every boundary the sampler meets and the measured cost of each, and **6**, the propagation speeds the cone can and cannot represent, with the three ranked (unmeasured) fixes |
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
