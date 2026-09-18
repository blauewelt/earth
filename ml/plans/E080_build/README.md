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
| `E080_hourglass_cone_deck.pptx` | 22 slides, speaker notes in each slide's notes pane |
| `E080_hourglass_cone_deck.pdf` | the slides only |
| `E080_hourglass_cone_deck_with_notes.pdf` | slide page + notes page, interleaved (2 × slides) |
| `E080_hourglass_cone_summary.pptx` | slide 21 on its own — same layout, no slide number, same notes |
| `E080_hourglass_cone_summary.pdf` | that one slide |
| `E080_hourglass_two_scales.pptx` | slide 22 on its own — same layout, no slide number, same notes |
| `E080_hourglass_two_scales.pdf` | that one slide |
| `E080_hourglass_two_scales_light.pptx` | the same slide 22 on WHITE — for printing and for light-theme documents; same kicker, headline, footer, legend, design-intent line, links and notes |
| `E080_hourglass_two_scales_light.pdf` | that one slide |
| `E080_figures/*.png` | the eight matplotlib figures, 200 dpi, plus the light-theme `two_scales_3d_light.png` |

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
| `figures.py` | all eight figures into `../E080_figures/` at 200 dpi, dark background. Slide 21 has two of its own, `summary_cone` and `summary_hourglass` — compact redraws of `warped_sunflower` and `hourglass_spacetime` with a label budget for a 3.65 in column (no ticks, axis words, a few large labels), both at 4.0 × 2.52 in = 1.587 : 1 so they land 2.30 in tall beside the results panel. `two_scales_3d` (slide 22) draws the mirrored double cone in 3-D at both scales — sunflower footprints stacked along the time axis, mplot3d with the surfaces drawn as ring + rib LINES (a translucent `plot_surface` sorts wrong against the dots at some azimuths) — its reach read from `ml/cone.py::reach_km` / `outer_reach_km` rather than a second copy of the formula; stage 2's rings are sampled densely below the 4,444 km cap at lag 33 and sparsely above it, or the taper disappears into a drum. `_hourglass_shape` is the one definition of the cone polygons, shared by figure 1 and both redraws. `two_scales_3d(th)` takes a THEME — `THEME_DARK` (the deck) or `THEME_LIGHT` (white background, ink titles and ticks, the three region colours darkened to hold on white) — and writes the file the theme names; the background colour is also what the stage-1 cut-out and the opaque label patches are filled with, so one definition draws both. The dark output must stay byte-identical: `sha256(two_scales_3d.png)` is `47d8af54…985f`. Two further themes, `THEME_PAPER_LIGHT` / `THEME_PAPER_DARK`, draw the same figure for the paper (`ml/paper/make_hourglass_fig.py` → `ml/paper/figs*/fig_hourglass.png`): a `fs` / `ms` factor of 1.3 on every font and mark, so the 11 in drawing still reads at the paper's 15.8 cm text width, the dark page's own colours, and a `pos` table that moves four labels which would otherwise cover the `km` axis word, the lowest past ring and the widest future ring at that size. Both factors default to 1 and `pos` to the deck's places, so the two deck files are unchanged; the generator calls sit under `if __name__ == "__main__"` so the paper script can import the module without redrawing the deck |
| `build_deck.js` | the pptxgenjs deck: house style, layout, the slide body text; also the standalone one-slide exports, whose bodies are `summaryBody()` (slide 21) and `twoScalesBody()` (slide 22), each shared with its deck slide through `standalone()`. `standalone()` and `twoScalesBody()` take a THEME object (`THEME_DARK` / `THEME_LIGHT` — background, kicker, title, body, mut, foot, the three region colours, the link colour and the figure file), so the white-background slide 22 is the same builder with a different palette rather than a second copy of it; `rt()` takes optional `link` / `ital` colours for the same reason |
| `fix_pptx.py` | post-processes the pptx (see the gotcha below) |
| `validate.py` | schema check (delegated) + deck checks: slide count, footer, page number, notes verbatim, no emoji |
| `make_notes_pdf.py` | the interleaved notes PDF, with reportlab |
| `rebuild.sh` | runs all of the above in order |

## Spec conventions the scripts rely on

- A slide is a section headed `## Slide N · Kicker`. `N` runs 1..22 with no
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
- **At most ONE emphasised run may follow a hyperlink in the same paragraph.**
  With two (`**3 seeds each**` *and* `**Verdict**` after the link on slide 21)
  LibreOffice renders the link in the body colour instead of accent blue,
  underline and link annotation intact — the PPTX markup is byte-identical to
  every other link in the deck, so this is a renderer quirk, not a builder bug.
  Measured on eight variants; the fix is to move the link so only one `**…**`
  follows it. Check the colour in the render, not in the XML.
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
