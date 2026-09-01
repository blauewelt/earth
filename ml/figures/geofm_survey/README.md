# Geospatial representation-model survey (stencil level) — 31 Aug / 1 Sep 2026

A 34-slide deck comparing AlphaEarth, TESSERA, OlmoEarth, TerraMind, Prithvi-EO 2.0 and
IBM's Granite-Geospatial-Ocean at the level of *what area in space and what span of time feeds
one embedding*, then applying that lens to Earth 2: the velocity-specific dependency cone, a
worked example for the surface current, the four-sphere plan, head-to-head benchmark tables
transcribed from the papers, and related work. Made at Yannick's request across one session.

| file | what |
|---|---|
| `geospatial-representation-models-with-notes.pdf` | **Read this one.** 68 pages: every slide followed by a plain-English speaker-notes page that explains the terms and symbols (embedding, stencil, token, frozen probe vs fine-tune, mIoU/F1/RMSE, Δt, ℓ, τ, v, Ekman, thermal wind, RAPID, corridor AUC…) |
| `geospatial-representation-models.pptx` | The deck with the same notes in the speaker-notes pane. Fonts chosen to survive a Google Slides import (Georgia headings, Calibri body) |
| `SURVEY_NOTES.md` | The running research note: verified stencil table, transcribed benchmark numbers, the cone model, the proposed pixel-year arm, related work — with sources |
| `build.js` | pptxgenjs generator for the deck (`node build.js`; needs `notes.js` beside it, writes `titles.json`) |
| `notes.js` | One plain-English note per slide, in deck order |
| `notes_deck.js` | Renders one landscape notes page per slide → `notes-pages.pptx` |
| `../dependency_cone_explorer.html` | Interactive companion to slides 26–31 — open via the Pages URL |

Rebuild: `node build.js && node notes_deck.js`, convert both `.pptx` to PDF with LibreOffice,
then `qpdf --empty --collate --pages geospatial-representation-models.pdf notes-pages.pdf -- geospatial-representation-models-with-notes.pdf`.

Everything quantitative about *our* codec in the deck (attribution matrix 0.613/0.617 vs
0.659/0.672, stencil shapes, corridor AUC) is quoted from `ml/paper/paper.tex` and
`ml/EXPERIMENTS.md` as of 31 Aug 2026; the cone parameters (speeds, memories) are
order-of-magnitude illustrations and the four ablations are hypotheses, not results.
