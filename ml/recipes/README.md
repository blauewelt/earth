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

Keys must be real `ml-train.yml` inputs — or **recipe-only keys**, below;
`resolve_recipe.sh` refuses an unknown one rather than exporting a variable
nothing reads. Add a recipe when a configuration has been **run and measured**,
and say in `_provenance` which run measured it — a recipe is a record of
something that worked, not a wish.

## Recipe-only keys

`workflow_dispatch` takes at most 25 inputs and `ml-train.yml` has exactly 25;
a 26th does not fail gracefully — GitHub refuses to parse the file and every
dispatch in the repo 422s. So a knob with nowhere to go may instead be declared
a **recipe-only key**, in the `RECIPE-ONLY KEYS` comment block of
`ml-train.yml`, and set only by naming a recipe.

The exemption is narrow, and it is the harmless half. What a recipe-only key
does NOT escape is the check that actually protects anything: the job must read
it as `$RECIPE_<KEY>` somewhere, or the dispatch is refused. A key that appears
to apply and does nothing is still impossible. What it gives up is the ability
to be overridden from the dispatch form — which is the direction §1 wants
anyway ("name a recipe; never hand-assemble an architecture").

Currently declared:

| key | reaches | values |
|---|---|---|
| `holdout_lon` | `ml/train.py --holdout-lon` (Train step) | `lo,hi` \| `none` — but pass **`0,0`** for "no holdout", see below |
| `train_lon_hold` | `ml/temporal.py --train-lon-hold` (`scripts/probes_run.sh`) | `inherit` \| `none` \| `lo,hi` |

`holdout_lon` is saved verbatim into the checkpoint's `args`, and **twelve
eval scripts under `ml/` still re-read it as
`lo, hi = (float(v) for v in ...split(","))`** — `ablate_channels`,
`dip_check`, `probe_head`, `probe_kfold`, `probe_sequence`, `project_amoc`,
`recon_decoder`, `recon_eval`, `rollout`, `rollout_spatial`, `train_joint`,
`trainprobe`: the whole probe ladder and the roll. A codec whose args say
`none` trains perfectly and then loses every one of them to `ValueError:
could not convert string to float`, each raise swallowed by a best-effort
guard — a green run with an empty probe archive. So a recipe asks for "no
holdout" as **`0,0`**: `[0, 0)` is the empty half-open interval, the mask is
bit-identical, and all twelve can parse it. `train.py` accepts `none` and
`""` and prints a `::warning::` naming this. `tests/test_lon_holdout_optional.py`
check 6 pins it. Routing those twelve through `train.lon_holdout_mask` is the
follow-up that retires the workaround.

## Three fields a recipe still cannot carry

Not every input is reachable, and the two guards that say so are correct — do
not weaken them to make a recipe tidier.

- **`resume`** and **`temporal_steps`** are read in `if:` conditions, which are
  YAML expression contexts evaluated before any shell exists. A recipe setting
  either would govern the trainer and not the gate — half-applied, which is the
  failure recipes exist to abolish. `tests/test_workflow_config.py` case 3
  refuses them.
- **`runner`** and **`job_timeout`** are read by `runs-on` / `timeout-minutes`
  at job start, before a step can set an env var.
- **stage-2 window tokens** (`stencil:`, `ring:`, `seed:`, `direct:`,
  `uprobs:`, `sched:`, `unroll:`) are not inputs at all — they are parsed out of
  `window` in `scripts/probes_run.sh`, and the `recipe:` token is stripped from
  the front of that same string, so a recipe and its window tokens travel
  together: `window: recipe:xl144-nolonhold,stencil:145,ring:spiral:...`.

A recipe whose arm needs any of these must say so in its `_description`, with
the exact string to dispatch — see `xl144-nolonhold.json`.


