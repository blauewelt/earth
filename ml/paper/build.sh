#!/bin/bash
# Build BOTH paper PDFs. This script exists because the dark-variant
# transformation lived only in a chat transcript and had to be re-derived
# by diffing the two .tex files (2026-08-08) — the third artifact this
# programme lost that way. paper_dark.tex is GENERATED; never hand-edit it.
#
# Usage:  cd ml/paper && bash build.sh [--figs]
#   --figs  regenerate figs/ and figs_dark/ first (make_figs.py both modes)
set -e
cd "$(dirname "$0")"

if [ "$1" = "--figs" ]; then
  python3 make_figs.py
  python3 make_figs.py --dark
fi

# Dark variant: same source, dark page + light ink + dark-mode figures.
python3 - <<'EOF'
src = open("paper.tex").read()
src = src.replace(
    r"\definecolor{ink2}{HTML}{6F6E66}",
    "\\pagecolor[HTML]{14140F}\n"
    "\\AtBeginDocument{\\color[HTML]{E8E6DF}}\n"
    "\\definecolor{ink2}{HTML}{A5A396}")
src = src.replace(r"\graphicspath{{figs/}}", r"\graphicspath{{figs_dark/}}")
open("paper_dark.tex", "w").write(src)
print("paper_dark.tex regenerated")
EOF

pdflatex -interaction=batchmode paper.tex >/dev/null
pdflatex -interaction=batchmode paper.tex >/dev/null
pdflatex -interaction=batchmode paper_dark.tex >/dev/null
pdflatex -interaction=batchmode paper_dark.tex >/dev/null
echo "built: $(ls -la paper.pdf paper_dark.pdf | awk '{print $NF, $5}')"
