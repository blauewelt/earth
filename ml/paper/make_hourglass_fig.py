#!/usr/bin/env python3
"""Figure for the design section (\\ref{fig:hourglass}): the mirrored double
cone at stage 1's and stage 2's scale, drawn by the E-080 deck's own generator
(ml/plans/E080_build/figures.py::two_scales_3d) under the two paper themes.

Writes figs/fig_hourglass.png (light page) and figs_dark/fig_hourglass.png
(dark page, the page and ink colours of build.sh's dark variant). The deck's
own two files are not touched: the paper themes differ only in scale and
colour, and figures.py leaves the deck outputs byte-identical.

Usage:  cd ml/paper && python3 make_hourglass_fig.py   (build.sh --figs runs it)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plans", "E080_build"))
import figures  # noqa: E402  (the deck's generator; import has no side effects)

for sub, theme in (("figs", figures.THEME_PAPER_LIGHT),
                   ("figs_dark", figures.THEME_PAPER_DARK)):
    figures.OUT = os.path.join(HERE, sub)
    os.makedirs(figures.OUT, exist_ok=True)
    figures.two_scales_3d(theme)
    print("wrote", os.path.join(figures.OUT, theme["out"]))
