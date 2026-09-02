# Archived: technical report v7 (27 August 2026) — RETIRED

This is the last version of the report written before the protocol reset of
2026-08-29. It is kept for the record and must not be cited as current.

**Why it is retired.** Nearly every rolled-forecast number in it — every
"corridor AUC", every long-hindcast correlation, the capacity, noise and
longitude-holdout effects — was measured on stage-2 heads trained under a
contaminated pool: the training loss was dense over the 144-frame window
while the pool only excluded windows whose *final* target was held out, so
windows straddling a held-out year teacher-forced that year's transitions
into the weights (21,018 of 400,176 scored frame-targets were held-out bins).
The report's authors (Claude sessions) stated repeatedly that the held-out
years were held out of training. They were not. The memorisation signatures
that established this are in `ml/plans/PROTOCOL_RESET.md` §2, and the
replacement report at `ml/paper/paper.tex` carries only results measured
under the corrected protocol.

Also retired with it: the metric name "corridor AUC" (the quantity is a mean
MSSS against climatology, not an AUC), the capacity ladder as a skill
question, and every pentad and daily rolled number.

What in it is not contaminated: the data description (§Data), the codec
architecture and its reconstruction curves, the JAX port's parity gates, and
the replication-discipline arithmetic. Those are carried into the new report
where they are needed and summarised rather than reproduced.

Files: `paper.tex` / `paper.pdf` (light), `paper_dark.tex` / `paper_dark.pdf`
(dark, generated), `make_figs.py` and the JSON inputs it read, `figs/` and
`figs_dark/`.
