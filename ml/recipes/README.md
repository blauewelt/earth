# Recipes — a dispatch names an experiment, not twenty-four knobs

A recipe is the full parameter set for one kind of run, checked in, so a
dispatch can say **what it is** instead of restating **how it is built**.

## Why this exists

On 2026-08-18 the same failure happened twice in one day:

- **#395** passed the stage-2 fields and `resume: !run-62` but left the codec
  fields at their workflow defaults, which describe the 0.92M *pilot*. It
  loaded 576-wide weights into a 128-wide model and died after 90 s with sixty
  `size mismatch` lines.
- **#387** raised `codec_d_model` to 1024 but left `codec_heads` at the default
  4 — head_dim 256 — and the 202M codec's embedding collapsed between step 10k
  and 15k. Nothing noticed for nine more hours.

Neither was carelessness. On #358, an entirely ordinary run, **17 of 24 inputs
had to be overridden**. A configuration where the defaults are wrong 71% of the
time is one where every dispatch is a 24-field copying exercise, and copying
exercises produce exactly these two bugs. A default that is never correct is
not a default.

## Using one

Put `recipe:<name>` at the front of the `window` input:

    window: recipe:f4-40M
    window: recipe:f3-anchor-41M,stencil:234,seed:0    # recipe first, then the rest

`scripts/resolve_recipe.sh` reads `ml/recipes/<name>.json`, exports every key
it holds, and strips the token from `window` before the job uses it. Any input
the recipe does **not** name keeps the dispatch's own value, so a recipe fixes
the architecture while `steps`, `seed` and the runner stay per-run.

The resolved configuration is printed in full at the top of the job, and the
recipe's name is recorded in `metrics.jsonl` — a reader of the curves can tell
which recipe produced them without finding the dispatch.

## Writing one

One JSON object of `input: value` pairs, both strings and numbers accepted,
plus two reserved keys:

    {
      "_description": "one line — what this configuration IS",
      "_provenance":  "which run established it",
      "codec_d_model": 576, ...
    }

Keys must be real `ml-train.yml` inputs; `resolve_recipe.sh` refuses an unknown
one rather than exporting a variable nothing reads. Add a recipe when a
configuration has been **run and measured**, and say in `_provenance` which run
measured it — a recipe is a record of something that worked, not a wish.
