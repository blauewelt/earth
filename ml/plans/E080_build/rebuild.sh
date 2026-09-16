#!/usr/bin/env bash
# Rebuild the three E-080 deliverables from the spec. Run from anywhere:
#   bash ml/plans/E080_build/rebuild.sh
# Everything derived lands in E080_build/build/; the deliverables land in
# ml/plans/ next to the spec.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export E080_PLANS="${E080_PLANS:-$(dirname "$HERE")}"
export E080_BUILD="${E080_BUILD:-$HERE/build}"
export E080_FIG_OUT="${E080_FIG_OUT:-$E080_PLANS/E080_figures}"

DECK="$E080_PLANS/E080_hourglass_cone_deck"
SUMMARY="$E080_PLANS/E080_hourglass_cone_summary"
SOFFICE="/mnt/skills/public/pptx/scripts/office/soffice.py"

# one entry point for the PPTX -> PDF conversion, used by the deck and by the
# standalone summary slide
topdf() {
  if [ -f "$SOFFICE" ]; then
    python3 "$SOFFICE" --headless --convert-to pdf \
      --outdir "$E080_BUILD" "$1" >/dev/null 2>&1
  else
    soffice --headless --convert-to pdf \
      --outdir "$E080_BUILD" "$1" >/dev/null 2>&1
  fi
}

mkdir -p "$E080_BUILD"

echo "== figures"
python3 "$HERE/figures.py"

echo "== notes and titles from the spec"
python3 "$HERE/extract_notes.py"

echo "== pptx"
node "$HERE/build_deck.js"
# pptxgenjs writes one <a:pPr> per RUN; LibreOffice honours the last one,
# which silently drops the bullet on any item containing inline bold.
python3 "$HERE/fix_pptx.py" "$DECK.pptx"
rm -f "$DECK.pptx.orig"
python3 "$HERE/fix_pptx.py" "$SUMMARY.pptx"
rm -f "$SUMMARY.pptx.orig"

echo "== validate"
python3 "$HERE/validate.py" "$DECK.pptx"

echo "== slides pdf"
rm -f "$E080_BUILD"/*.pdf "$E080_BUILD"/qa-*.png "$E080_BUILD"/slidepage-*.png \
      "$E080_BUILD"/summary-*.png
topdf "$DECK.pptx"
cp "$E080_BUILD/E080_hourglass_cone_deck.pdf" "$DECK.pdf"

echo "== summary pdf (one slide, standalone)"
topdf "$SUMMARY.pptx"
cp "$E080_BUILD/E080_hourglass_cone_summary.pdf" "$SUMMARY.pdf"

echo "== notes pdf"
# the notes PDF embeds each slide as a 150 dpi raster, then a dark notes page
pdftoppm -png -r 150 "$DECK.pdf" "$E080_BUILD/slidepage"
python3 "$HERE/make_notes_pdf.py"

echo "== qa renders (look at every one of these before shipping)"
pdftoppm -png -r 50 "$DECK.pdf" "$E080_BUILD/qa"
pdftoppm -png -r 50 "${DECK}_with_notes.pdf" "$E080_BUILD/qa-notes"
pdftoppm -png -r 100 "$SUMMARY.pdf" "$E080_BUILD/summary"

echo
echo "slides : $(pdfinfo "$DECK.pdf" | awk '/^Pages/{print $2}') pages  -> $DECK.pdf"
echo "notes  : $(pdfinfo "${DECK}_with_notes.pdf" | awk '/^Pages/{print $2}') pages  -> ${DECK}_with_notes.pdf"
echo "summary: $(pdfinfo "$SUMMARY.pdf" | awk '/^Pages/{print $2}') page   -> $SUMMARY.pdf"
echo "pptx   : $DECK.pptx"
echo "pptx   : $SUMMARY.pptx"
