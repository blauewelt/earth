#!/bin/bash
# Build the archived v7 report (light + dark). Same transformation as
# ../../build.sh; paper_dark.tex is GENERATED, never hand-edit it.
# Usage: cd ml/paper/archive/v7-2026-08-27-contaminated && bash build.sh [--figs]
set -e
cd "$(dirname "$0")"
if [ "$1" = "--figs" ]; then
  python3 make_figs_addendum.py
  python3 make_figs_addendum.py --dark
fi
python3 - <<'PY'
src = open("paper.tex").read()
src = src.replace(
    r"\definecolor{ink2}{HTML}{6F6E66}",
    "\\pagecolor[HTML]{14140F}\n"
    "\\definecolor{darkink}{HTML}{E8E6DF}\n"
    "\\AtBeginDocument{\\color{darkink}\n"
    "  \\renewcommand{\\normalcolor}{\\color{darkink}}}\n"
    "\\definecolor{ink2}{HTML}{A5A396}")
src = src.replace(r"\graphicspath{{figs/}}", r"\graphicspath{{figs_dark/}}")
open("paper_dark.tex", "w").write(src)
PY
for t in paper paper_dark; do
  pdflatex -interaction=batchmode $t.tex >/dev/null || true
  pdflatex -interaction=batchmode $t.tex >/dev/null || true
done
echo "built: $(ls -la paper.pdf paper_dark.pdf | awk '{print $NF, $5}')"
