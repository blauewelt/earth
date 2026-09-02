# E-067 / E-068 · Rolling past a year: held-out BLOCKS, and the terminal codec

**Written 2026-09-02 ~17:2xZ, at dispatch of E-068.** Chris: *"Can you
continue to roll forward the results in Figure 5? It seems that at exactly
365 days, the transformer heads seem to catch up with LIM."* — and, on the
first draft of this plan: *"I thought we already established that we're
holding out the last 3 years. This was part of an earlier suggestion of
yours for a programme reboot."* He was right: the frozen protocol
([PROTOCOL_RESET §3.1](PROTOCOL_RESET.md), [REBOOT_PLAN §3 and step 4](REBOOT_PLAN.md),
decided 2026-08-30) is **train ≤ 2020, test 2021–2024, no gap**, and it
begins with a codec trained on ≤ 2020 — "the long pole; start it first".
It had never been dispatched. This plan is that dispatch, plus the
development-side reading Chris's question actually asks for.

## 1. Why Figure 5 stops at 365 days, and why the end of it is noisy

The battery scores a roll only while its target row is inside the held-out
year (`scored_horizon`: the roll breaks at the year boundary). With three
starts per year near 1 Jan / 1 May / 1 Sep, nine starts feed leads to
~125 d, six to ~245 d, and only the three year-start starts feed
250–365 d (n falls 2,171,138 → 1,447,424 → 723,710 in `probes-520`). Every
target past 365 d from those starts lies in 2010 / 2018 / 2024 — training
years for the head and the LIM alike — so a longer roll of the existing
heads is not a skill measurement.

Over 250–365 d (three starts) the 7.6M head's per-lead corridor ACC averages
0.062 with a lead-to-lead sd of 0.067 (mean |Δ| between neighbouring leads
0.044); the LIM's averages 0.182, sd 0.035. The head's last five leads read
0.015, 0.15, 0.20, 0.12, 0.12 against the LIM's 0.14, 0.17, 0.18, 0.17,
0.18. The "catch-up" is the head's jitter touching the LIM's curve at the
end of a three-start segment, not a crossing. Whether the head really holds
~0.1 while the LIM keeps decaying below it is an open question — this plan
is how it gets measured.

## 2. The mechanism: contiguous held-out years form BLOCKS

Groundwork landed on `main` (`72eade6`): consecutive held-out years group
into blocks (`ml/temporal.hold_blocks`); the evaluator picks starts per
block and truncates a roll at the **block's** end (`starts_for_block`,
block-aware `scored_horizon`); single-year behaviour is byte-identical
(`tests/test_roll_monthly_identity.py`, the LIM smoke artefact). A stage-2
run may hold out MORE years than its codec (`ml/temporal.py
--holdout-years`, the `hold:` window token) and is refused if it holds out
fewer; the embed cache then carries `_hold-<blocks>` in its name so a Z
embedded under other anomaly statistics can never be pulled by mistake. A
codec's own holdout is a recipe key (`holdout_years` → `ml/train.py
--holdout-years`).

Without this, a roll from a 2021 start would have been truncated at
2021-12 — the terminal protocol needed it as much as the development
blocks do.

## 3. E-068 · the terminal codec — DISPATCHED #532, 2026-09-02 17:1xZ

**E-068 · run-415's architecture trained with 2021–2024 held out, plus the
two-year development blocks 2008–09 and 2016–17 · params 37.976M · stage
encoder · data family4_na025_pentad_r2 · arch 512×12 d_z 32 patch 1 · steps
200k×512 · resume none · recipe `f4r2-40M-terminal` (holdout_years
2008,2009,2016,2017,2021,2022,2023,2024) · runner gpu-box-49401037 (Vast
49633408, Virginia, 63 GB, tensor cached from #523).**

One codec carries both holdouts because a development roll and the
terminal test must share a frozen encoder, and REBOOT_PLAN step 4 offers
exactly this ("2021–2024 plus the development fold-years"). Eight of the
axis's 43 years (1982–2024) are held out, ~19 % of the bins.

Readings, in order: (i) the codec's own probe ladder against run-415's
(`loss_rec`, the unpooled RAPID head) — the width-tax control is now a
data-tax control; (ii) the Z embedding (published under its own key);
(iii) the 7.6M stage-2 head (E-060a's configuration, torch, K 144, window
pool, 20k steps with milestones) — its pool is smaller by the eight years;
(iv) **E-067**: the head and a refitted LIM (the same eight years excluded
from the fit) rolled 146 pentads (730 d) from starts spread over the
2008–09 and 2016–17 blocks, truncated at each block's end. Two blocks × 3
starts; only the block-start starts reach 730 d, so the far end is again a
2-start reading — quoted as such.

**2021–2024 is NOT rolled by this plan.** It is the test set the protocol
opens once; it is rolled when Chris says so, with whatever head and
baseline are then the programme's best, and reported as the terminal
number of the paper.

## 4. Pre-registered readings for E-067

- Lead-decay must PASS on both blocks (a flat profile is a replay).
- The head's corridor ACC over 400–730 d against the LIM's over the same
  leads, same starts. If the head holds ≥ 0.10 and the LIM decays below it,
  Chris's reading of Figure 5 is right and the transformer carries a slow
  component the linear model does not; if the LIM stays above the head to
  730 d, the 365 d touch was noise. Either is a consistency at two starts
  per block until a second seed and a second block reading agree.
- The 0–365 d segment of the block rolls must reproduce the single-year
  battery's shape on this codec (a check that the block mechanism did not
  move the near leads).

## 5. Cost

Codec ~19 h on a 4090 (~$6; run-415 took 19 h at this size); Z embedding
~8 h on a 4090 (or ~4 h on an H100); head ~2 h; LIM ~1 h; the two block
rolls ~4 h at the 7.6M head's 9 s/unit (146 + 122 + 98 leads per block ×
2). ≈ $12 and ~1.5 days end to end, with the terminal roll (~6 h more)
held for Chris.

## 6. What must be verified, not assumed

- `#532`'s first minutes: the Train banner must print `--holdout-years
  2008,2009,2016,2017,2021,2022,2023,2024` and the pool certificate; the
  anomaly statistics must exclude the eight years (the `std_stats` line).
- The codec checkpoint's `args["holdout_years"]` must read the eight years
  before any stage-2 run names it in `resume` — every downstream script
  reads the years from there.
- The Z published for it must NOT carry a `_hold-` suffix (the codec's own
  years are the default; a suffix would mean a stage-2 override was on).
