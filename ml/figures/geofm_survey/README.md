# Geospatial representation-model survey (stencil level) — 31 Aug – 2 Sep 2026

A 46-slide deck comparing AlphaEarth, TESSERA, OlmoEarth, TerraMind, Prithvi-EO 2.0 and
IBM's Granite-Geospatial-Ocean at the level of *what area in space and what span of time feeds
one embedding*, then applying that lens to Earth 2: the velocity-specific dependency cone, a
worked example for the surface current, the four-sphere plan, head-to-head benchmark tables
transcribed from the papers, and related work; since 2 Sep also the *generic-embedding input proposal* (slides 34–45): the 13-rung input ladder, observed-vs-derived rule and leakage trap, cone families per input, the cone-native codec ("dots in, embedding out"), where velocity comes from, the zero-sum question, phases, 2026–27 data continuity. Made at Chris's request across two sessions.

| file | what |
|---|---|
| `geospatial-representation-models-with-notes.pdf` | **Read this one.** 92 pages: every slide followed by a plain-English speaker-notes page that explains the terms and symbols (embedding, stencil, token, frozen probe vs fine-tune, mIoU/F1/RMSE, Δt, ℓ, τ, v, Ekman, thermal wind, RAPID, corridor AUC…) |
| `geospatial-representation-models.pptx` | The deck with the same notes in the speaker-notes pane. Fonts chosen to survive a Google Slides import (Georgia headings, Calibri body) |
| `GENERIC_EMBEDDING_INPUTS.md` | The in-depth proposal behind slides 34–45: input ladder with plain-English descriptions, cone parameters per input, the cone-native codec, the zero-sum analysis, phased plan, and a dataset spec sheet verified against agency pages on 2 Sep 2026 |
| `SURVEY_NOTES.md` | The running research note: verified stencil table, transcribed benchmark numbers, the cone model, the proposed pixel-year arm, related work — with sources |
| `build.js` | pptxgenjs generator for the deck (`node build.js`; needs `notes.js` beside it, writes `titles.json`) |
| `slides_inputs.js`, `notes_inputs.js` | Slides 34–45 and their notes, required by `build.js` / `notes.js` |
| `notes.js` | One plain-English note per slide, in deck order |
| `notes_deck.js` | Renders one landscape notes page per slide → `notes-pages.pptx` |
| `../dependency_cone_explorer.html` | Interactive companion to slides 26–31 — open via the Pages URL |

**2 Sep 2026 — protocol note.** The paper was reset the same day (v8: every rolled number from
heads trained under the endpoint pool withdrawn; "corridor AUC" retired as a metric name). The
deck and `GENERIC_EMBEDDING_INPUTS.md` were corrected in the same commit: no withdrawn rolled
number is quoted, the Phase-0 experiment is scored under the corrected window-scope protocol
(MSSS per lead vs climatology and damped persistence, LIM null, n ≥ 3 seeds), and the
attribution-matrix probe contrast (0.672 vs 0.659) is stated as a single-seed number inside the
probe noise band (`ml/CLAUDE.md` §3b), i.e. consistent with parity.

Rebuild: `node build.js && node notes_deck.js`, convert both `.pptx` to PDF with LibreOffice,
then `qpdf --empty --collate --pages geospatial-representation-models.pdf notes-pages.pdf -- geospatial-representation-models-with-notes.pdf`.

Everything quantitative about *our* codec in the deck (attribution matrix 0.613/0.617 vs
0.659/0.672, stencil shapes) is quoted from `ml/paper/paper.tex` (v8 appendix, retained
probe numbers) and `ml/EXPERIMENTS.md`; the cone parameters (speeds, memories) are
order-of-magnitude illustrations and the four ablations are hypotheses, not results.
