# E-080 deck build

Everything needed to rebuild the three E-080 deliverables from the spec.
The spec, `../E080_hourglass_cone.md`, is the single source of truth: the
scripts read it, nothing here restates its content.

## Run it

```bash
bash ml/plans/E080_build/rebuild.sh
```

Then **look at every page** of `build/qa-*.png` (slides) and
`build/qa-notes-*.png` (slides + notes) before shipping. The layout is
hand-tuned inch geometry, so any text change can push a card into a figure
or a bullet past the bottom of its box, and only a rendered page shows it.
Fix by tightening non-numeric wording, or by adjusting the `x/y/w/h` of the
image and the card on that slide in `build_deck.js`.

Outputs, next to the spec in `ml/plans/`:

| file | what |
|---|---|
| `E080_hourglass_cone_deck.pptx` | 20 slides, speaker notes in each slide's notes pane |
| `E080_hourglass_cone_deck.pdf` | the slides only |
| `E080_hourglass_cone_deck_with_notes.pdf` | slide page + notes page, interleaved (2 × slides) |
| `E080_figures/*.png` | the five matplotlib figures, 200 dpi |

Derived files (`content.json`, the intermediate PDF, the 150 dpi slide
rasters, the QA renders) all land in `build/`, which is disposable.

## Requirements

- **node** with **pptxgenjs**. It is resolved from a *global* npm install
  (`npm root -g` → `/home/claude/.npm-global/lib/node_modules` on this
  machine); there is no `node_modules` here and no `package.json`. If
  `node -e "require('pptxgenjs')"` fails, run `npm install -g pptxgenjs`.
- **python3** with `matplotlib`, `numpy`, `python-pptx`, `reportlab`,
  and `pdfplumber` for the verification snippets.
- **LibreOffice** (`soffice`) for the PPTX → PDF conversion and
  **poppler** (`pdftoppm`, `pdfinfo`).
- Optional: the pptx skill at `/mnt/skills/public/pptx/scripts/office/`.
  `rebuild.sh` uses its `soffice.py` wrapper when present and falls back to
  plain `soffice`; `validate.py` here delegates to its OOXML schema
  validator when present and skips that layer when it is not. That
  validator is **not** vendored, because it needs its sibling modules and
  the XSD schema set.

## The files

| file | what it does |
|---|---|
| `extract_notes.py` | parses the spec into `build/content.json` — slide titles and the `*Notes.*` text, whitespace collapsed |
| `figures.py` | all five figures into `../E080_figures/` at 200 dpi, dark background |
| `build_deck.js` | the pptxgenjs deck: house style, layout, the slide body text |
| `fix_pptx.py` | post-processes the pptx (see the gotcha below) |
| `validate.py` | schema check (delegated) + deck checks: slide count, footer, page number, notes verbatim, no emoji |
| `make_notes_pdf.py` | the interleaved notes PDF, with reportlab |
| `rebuild.sh` | runs all of the above in order |

## Spec conventions the scripts rely on

- A slide is a section headed `## Slide N · Kicker`. `N` runs 1..20 with no
  gaps; `extract_notes.py` fails loudly otherwise. The kicker becomes the
  small uppercase blue line at the top of the slide.
- The **first bold paragraph** of the section is the headline — the big
  serif title. `build_deck.js` carries it as the `sub` argument of
  `newSlide`, and `titleSize()` picks one line at ≥ 23 pt or two smaller
  ones.
- `[FIGURE: name — description]` is a note to the deck builder, not
  content. `figures.py` draws `name.png`; `build_deck.js` places it.
- `*Notes.* …` up to the next blank line is the speaker note. It reaches
  the notes pane and the notes PDF **verbatim** — `build_deck.js` never
  edits it, and `validate.py` fails the build if it drifts.
- Bullets and numbered items are re-typed in `build_deck.js`, because they
  have to fit fixed-size cards. Keep them faithful; tighten only
  non-numeric wording, and say so when reporting a rebuild.

## Inline markers in `build_deck.js` body text

These are this builder's own micro-markup, handled by `rt()`:

| marker | renders as |
|---|---|
| `**bold**` | bold in the accent colour (blue by default, `accent` overrides) |
| `{{italic}}` | italic gold — for the one-line aphorisms |
| `~{x}` | subscript |
| `^{x}` | superscript |
| `[label](url)` | a real PPTX hyperlink: accent blue, thin underline |
| `` `code` `` | backticks are stripped; the text stays plain |

Write `exp(s~{c})` rather than `e^{s_c}` — the underscore and the caret are
not markers, and a bare `_x_` used to match across them.

## Links

Chris's rule: any reference another reader would not know links to its
original definition.

- **On a slide**, `[label](url)` becomes an `<a:hlinkClick>` with an
  external relationship, styled accent blue with a thin underline.
  `ahyp:hlinkClr val="tx"` keeps that colour instead of the theme's
  hyperlink colour. LibreOffice carries these into the slides PDF as real
  link annotations — check with `pdfplumber` that the count matches the
  pptx's hyperlink relationships.
- **In the notes pane**, plain text: `notesText()` rewrites
  `[label](url)` as `label — url` so the URL is visible to whoever is
  reading the notes.
- **In the notes PDF**, `make_notes_pdf.py` turns the same markdown into
  reportlab `<a href>` — clickable, underlined, accent blue. The spec's
  notes currently contain no links, so that PDF has no link annotations;
  that is expected, not a dropped URL. Its slide pages are 150 dpi
  rasters, so the slides' own links do not survive there either.

## Gotchas, all of them learned the hard way

- **pptxgenjs starts a new paragraph at every run carrying `bullet`.** So
  paragraph properties (`bullet`, `paraSpaceAfter`) go on the **first** run
  of an item only, and `breakLine` on the last. `para()` does this.
- **Duplicate `<a:pPr>`.** pptxgenjs writes one per run; LibreOffice honours
  the last, which erases the bullet on any item containing inline bold.
  `fix_pptx.py` keeps only the first per `<a:p>`. Never skip it.
- **Georgia is substituted by LibreOffice** and renders wider than the
  metrics suggest, so titles wrap into figures. `titleSize()`'s constants
  (1300 / 2600) were calibrated from a real render, not from font metrics.
- **The notes PDF shrinks its body font** from 16 pt down to 10 pt until the
  paragraph fits its card. A note that suddenly renders small means the
  spec's note grew, not that something broke.
- **Always re-extract before building.** `rebuild.sh` does; running
  `node build_deck.js` by hand against a stale `content.json` silently
  ships the previous revision's notes.

## House style

From the E-069 deck. 16:9 (13.333 × 7.5 in). Background `#0D1117`, cards
`#161B22` with `#30363D` borders, body `#E6EDF3`, muted `#7D8590`, footer
`#4A5460`, accents blue `#4493F8`, gold `#E3B341`, orange `#E8734A`, red
`#F85149`. Georgia for titles, Calibri for body, Cambria for formulas. Every
slide carries a small uppercase blue kicker, a bold serif headline, the
footer "E-080 · the cut-off mirrored double cone" on the left and the slide
number on the right. No emoji.
