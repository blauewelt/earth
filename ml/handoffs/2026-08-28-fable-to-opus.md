# HANDOFF: Fable → Opus, 2026-08-28 ~06:30Z — the next 2–3 days

Chris: *"We're almost out of tokens for you (Fable)... this should be
roughly 2-3 days of instructions for Opus."* Fable returns Sunday
(2026-08-30). Chris has standing-authorized aggressive parallelization and
auto-top-up ("Any additional boxes and parallelization is welcome...
let's start them", 2026-08-27) — spend on more boxes/TPU when it buys
calendar time, report the arithmetic, never park the fleet.

**Read before acting, in this order:** `ml/CLAUDE.md` (every § applies —
especially §0b: plan in the main session, implement via Opus subagents
with verified diffs; §0c/§0d/§0f report forms; §2 first-minutes; §3
unpooled-is-the-verdict; §3b replication), `ml/OVERVIEW.md`,
`claude/expectations.md` (project doc — the ops ledger; UPDATE IT at every
check-in), and this file. Push via
`node scripts/git_api_push.mjs --token-file /home/claude/.gh_pat --branch main`
(git proxy refuses; ALWAYS `git pull --rebase origin main` first — three
other sessions push to main). Every chat message to Chris: markdown links
one per line, run numbers never without a summary, PLAIN-ENGLISH gloss for
any term of art (he flagged "battery" — spell such things out).

**Scheduled wakes already armed:** 08:15Z 08-28 morning harvest
(trig_01TA7fgQUhYxc3vMnbuw9Xv1 — its checklist is mostly superseded by
this doc; follow this doc where they differ). Schedule your own wakes with
send_later as work demands; keep the cadence: check-ins ~2 h around
critical landings, quiet otherwise.

---

## 0 · IMMEDIATE (first hour)

1. **E-054b is NOT running — relaunch it.** The gradient-accumulation fix
   is BUILT, CERTIFIED, PUSHED (6ec09b9-class commit; tests
   `tests/test_grad_accum_jax.py` 6/6). The relaunch retry loop DIED when
   its shell exited (sandbox background processes do not survive between
   Bash invocations — root CLAUDE.md §4; do long waits with repeated
   short checks, or nohup started and re-checked within the SAME
   session's later commands). Startup file ready:
   `/tmp/e054b_startup_v2.sh` (GRAD_ACCUM=4, D_MODEL 1280, LAYERS 20,
   STEPS 200000, TAG e054b, placeholders substituted). Launch:
   `cd /home/claude/earth && GCP_SA_KEY=/home/claude/.gcp/sa.json GCP_PROJECT=earth-tpu-blauewelt GCP_ZONE=us-west1-c python3 scripts/tpu_box.py create e054b-400m --spot --startup-file /tmp/e054b_startup_v2.sh`
   — expect LEMON DRAWS in us-west1-c (one refused 06:04Z; the guard
   auto-deletes, each retry is a fresh host); retry 2–3×, then walk
   us-west4-b / us-central1-a / us-east5-b, then ON-DEMAND any zone
   (authorized). Append every incident to `ml/SPOT_LEDGER.md`.
   First-minutes: config echo `grad_accum: 4, micro_batch: 64`, NO OOM
   through the first training steps, LR 4e-4, fresh step 0. If it OOMs
   even at accum 4: try GRAD_ACCUM=8 via sed (still exact); if that
   fails, HBM fixed costs dominate → record and hold for remat (the
   fallback in EXPERIMENTS e-054's options). ~32 h once running.
   Registered reading (vs E-054a's 0.02981 @400k, 206.6M): at 200k
   this run trains HALF the steps of E-054a — the honest comparison is
   vs E-051's 0.0330@200k at 206.6M: materially below ⇒ capacity pays;
   ≈0.033 ⇒ capacity null at this rung. **When it finishes (~08-29
   late): the trainer has NO post-loop final save and its success-path
   self-reap has hung once — the durable final is the last CKPT_EVERY
   multiple (open it, read `_step`), and delete the node by hand if the
   log shows the "checkpoint unchanged" ship pattern.**

2. **#494 (E-056a, tokens K=24 on gpu-box-32966687) — live metrics
   possibly STALE**: last visible record step 2800 at both 05:04Z and
   06:21Z, but the run should be at ~15k+ (0.34 s/step). Diagnose
   BEFORE trusting: check the run's Actions status + `ml-live-494`
   freshness + Vast `disk_usage` on instance 48520137 (was 75/100
   BEFORE the ~3 GB token-Z embed — **a full-disk box computes fine and
   reports nothing**, ml/CLAUDE.md §7). If it's just the publisher dead
   of disk: the run finishes anyway (~20k ≈ done by ~06:45Z if healthy);
   the verdict then comes from the archived `run-494.jsonl` on
   ml-metrics (or hand-harvest from the box via a follow-up job if the
   archive step also failed). NEVER stop this box — until #494's
   publish step lands, its disk holds E-050's ONLY 260k FSQ codec.

---

## 1 · IN FLIGHT — what to harvest, when, and how to read it

| run | what | lands | how to read it |
|---|---|---|---|
| **#494** (E-056a) | stage-2 K=24 head ON 16-bit FSQ tokens (d_z 6) — is the token alphabet a competitive forecasting substrate? | ~07Z 08-28 (if healthy) | final one-step ratio = last `stage2_val_zmse` / monitor `val_persistence` **0.63451** (token-z scale!). Controls: continuous d_z-32 **0.5056**, lattice d_z-32 **0.4394** (both 20k K=24). **≲0.44 ⇒ token road OPENS at ~5% state size** (gates the next $100-class head onto tokens). **PRE-REGISTERED CONFOUND (already in e-056): znoise rel_pers measured 0.87878 on the token scale vs 0.15116 on d_z-32 — ~6× the intended relative dose.** A LOSS is confounded (dispatch the clean arm: same JSON `/tmp/d_e056a.json` with `--input-znoise 0.12` in the window tail, ~$1.5); a WIN stands a-fortiori. Also VERIFY after #494: `run-485__pixelmae.pt` on model-checkpoints-v1 (OPEN it, step must be 260000) + token-Z cache (~3 GB) on embed-cache-v1 — then note in expectations for the E-050 session. |
| **#495** (E-056b) | same on K=144 — does the dense 2-y slab hold on tokens, and what does a K=144 step COST on d_z 6? | starts after #494; ~13 h; watch the embed reuse (token-Z cache should make its embed ~free) | first-minutes: `resumed at_step 260000`, `fsq_levels 8,8,8,5,5,5`. Verdict vs #478's **0.0820**; the measured s/step vs torch K=144 pace is the efficiency datapoint. Same znoise confound applies — read rel_pers, same clean-arm rule. |
| **#503** (E-051-roll, gpu-box-38116559) | THE DECISIVE READING: 365-day roll of the best pentad head ever (398k, one-step 0.0298) — does it forecast, where the 20k K=24 head collapsed to −0.499? | skill phase partials ~10:30Z 08-28; full run ~81 h; **job token EXPIRES 23:52Z 08-28** | read `rollout_spatial.json` partials on `ml-live-503` (`in_progress: true` — quote only as partial). Compare `horizon_auc_daymatched` vs monthly `_trainlon` 0.939/0.939 and pentad-K24 −0.499. **The replay battery (the long/future rolls' tracking-vs-lead profile — flat = calendar memorisation) is MANDATORY before any number is called forecast** — it's in the ~70 h tail. DECISION at ~11Z: if the skill read is decisively positive or negative, consider cutting the run after the long/future rolls needed for the battery complete rather than riding 70 h of animation dumps — but check what fraction of the tail IS battery vs dumps first (the dumps are `--dump-roll` npz writes; the battery needs the rolls themselves). After token expiry the box computes blind (live pushes 401) — plan the hand-harvest: the artefacts stay on the box; a short follow-up job on the SAME runner can upload them. |
| **#500 / #502** (E-057.1 FGN pair, gpu-box-42005419 / 46292015) | learned-noise ensemble heads, seeds 0/1 — does eps-FiLM + fair CRPS replace hand-dosed znoise? | #500 ~22:30Z 08-28; #502 ~03:30Z 08-29; **both exceed the 24 h token — HAND-HARVEST both** (pull run-50{0,2}.jsonl + heads from live branches before expiry / box after) | F2 during: `stage2_val_member_var` → 0 = ε-collapse (so far 0.31–0.77, healthy). The corridor verdict needs the ENSEMBLE ROLL (torch evaluator BUILT by the FGN session — rollout_spatial +779; dispatch an sroll-style eval after both heads land, vs znoise pair **0.7235** with the clean pair 0.6781). This is primarily the FGN session's harvest — coordinate via expectations; don't double-dispatch. Cross-box caveat recorded in e-057(j). |
| **E-052.1 / E-052.1b** | diffusion session's field heads (on-demand + us-west4-a spot) | ~04Z 08-28 (may be done) | THEIRS — do not touch, do not reap their nodes. |

Fleet rules that bite: never stop gpu-box-32966687 with jobs queued;
gpu-box-40623952 (Vast 48478310) is a confirmed dead-GPU lemon
(destroy-candidate, stopped) and gpu-box-30257785 (47726876) is full-disk
(stopped) — do NOT restart either; Vast API v1 for instances (v0 for
users/invoices); api.github.com via Node fetch only.

## 2 · THE DIMENSIONS — strongest next steps for each

**(a) Capacity (E-054 line).** E-054a settled: 2× steps bought −9.7%
(0.0330→0.0298), decelerating ⇒ capacity is the cheaper axis. E-054b
(400M, launch above) is the test. If it lands materially below 0.033 at
200k: the axis is open — the follow-up is NOT an immediate 800M (HBM
ceiling measured: 4.03G free; 400M only fits with accumulation) but
E-054b→400k continuation (bucket-resume, same node name — and FIX THE
LAUNCHER FIRST: add a post-loop final save + trainer-exit reap to
tpu_train_s2.sh; small subagent job, spec in EXPERIMENTS e-054's lesson
paragraph). If ≈0.033: capacity null at 400M — write it as n=1 with the
§3b caveat and pivot GPU-hours to the token road / rolls.

**(b) Codecs incl. FSQ (E-056 line).** The E-053 synthesis ("frames, not
placement, bind — make frames cheaper") makes this the strategic axis.
#494/#495 are the gate. If tokens pass at K=24 (≲0.44): dispatch the
znoise-matched clean arm anyway (removes the confound from the record,
~$1.5), then the BIG move — the next full-budget pentad head (200k×K144
class, ~$100) trains ON TOKENS; design it as E-056c, register two-sided
vs E-051's 0.0330, and consider training it on TPU (jaxport supports
d_z 6; check fsq handling in embed path — the Z comes pre-embedded so
the head side is d_z-agnostic). If tokens fail even in the clean arm:
the road closes for now; record, and capacity + rolls carry the
programme. Separate codec follow-up owed to the E-050 session: verify
their 260k final went durable (item 1.#494 above).

**(c) Rolls (E-051-roll + forced rolls).** #503 is the reading that
matters most this week. THEN the approved-but-unbuilt item: **forced
rolls** — Chris approved feeding OBSERVED wind (tau channels) into the
roll at each step instead of letting the model free-run its own weather,
because unforced rolls relax to climatology (wind-only ridge 0.568 is
the bar that shows how much wind alone carries). Build spec: eval-side
only, a `--force-channels tau_x,tau_y,...` mode in
`ml/rollout_spatial.py` that, at each rolled step, re-encodes the
window with the FORCED channels' true values spliced into the input
state before the codec (or — cheaper and cleaner given z-space rolling —
substitutes the true z of a wind-only... NO: z mixes channels, so
channel-splicing must happen in pixel space pre-encode; that makes each
rolled step cost one encode — price it first on a toy). Write the plan
doc (ml/plans/, REGISTER in docs.html DOCS — the suite enforces it),
falsifiers first (forced ⋙ unforced at long lead = the model uses
forcing correctly; forced ≈ unforced = wind wasn't the binding
information), subagent-build with tests, then one eval dispatch on the
#503 head. This is 1–2 days of the build queue and is the natural
complement to whatever #503 says.

**(d) FGN.** Pair lands within ~24 h. After hand-harvest: the ensemble
roll eval (torch, built) answers F1 vs 0.7235. If FGN wins: it becomes
the default stage-2 objective and the JAX FGN port (phase 1 CERTIFIED,
d9ae181) makes TPU training available for the follow-ups; phase 2 (the
ensemble roll in jaxport, spec §2 of ml/plans/FGN_JAX_PORT.md) becomes
worth building (M=8 rolls are 8× compute — TPU's natural ground). If
FGN loses: znoise stays the default; phase 2 waits. Either way the
pre-registered rule stands: the first JAX-trained FGN arm at a
torch-measured config buys a same-seed torch twin (plan §4).

**(e) NEW DIRECTIVE — multi-target read-outs (Chris, this morning):**
*"I'd love to predict ocean surface temperature as a secondary
downstream target (next to AMOC). It's important to have multiple such
targets (maybe even a third one) to ensure the embedding representation
is comprehensive and not just AMOC tailored."* This slots perfectly
into the programme statement (§"What this programme is building", item
5: predicting embeddings predicts everything; the decoder already
exists). Execute in three rungs, cheapest first:

1. **SST field skill from ROLLED embeddings (the flagship, ~1 day,
   mostly eval-side build).** The codec's decoder answers coordinate
   queries and SST is channel 40 (E-042; the tensor carries it, truth
   exists as `sst` in the tensor + the daily `sst_na025` artifacts on
   the HF Hub). Build: in `ml/rollout_spatial.py`, decode the rolled z
   at each step to the SST channel over the corridor/window and score
   MSSS vs climatology and vs persistence, per horizon band — NEW KEYS
   ONLY (`sst_field_msss_*`), never touching read_sv/GATE_REF (the §3
   exception-1 discipline; E-055 is the template for additive keys).
   Then read it retroactively: re-run the eval on the #503 head (and
   the monthly champion #422/#429) — no retraining, the embedding
   either carries SST or it doesn't, WHICH IS EXACTLY CHRIS'S QUESTION.
   Register the falsifier before running: rolled SST skill far below
   what the SAME roll's z-space skill implies ⇒ the embedding is
   AMOC-shaped, not comprehensive.
2. **SST as a probe target in the standard ladder (hours).**
   `head_targets` (recipe-only key) already extends probe_head beyond
   RAPID; add an SST-section/region target so every future run reports
   it alongside rapid_r. One recipe edit + verify output keys appear.
3. **Third target — Florida Current transport (data task, ~1 day).**
   Daily cable transport since 1982, public (NOAA/AOML), physically
   distinct from RAPID (western boundary vs basin-wide) yet related —
   the ideal "is it comprehensive" triangulation point. Bake it as a
   truth series (scripts/refresh_data.py pattern; catalog entry per
   root CLAUDE.md §2 if it becomes a globe layer later), add as a
   head_targets probe + a rolled read-out. Alternatives if AOML access
   fails: OSNAP (lower cadence), or OHC-700m computed from the rg_t
   channels (internal, weaker as an external check).
   Propose the design to Chris in the §0f report before the big eval
   spend; rungs 1–2 need no permission (additive, cheap).

## 3 · BUILD QUEUE (subagent jobs, in priority order)

1. E-054b launch + babysit (item 0.1) — not a build, do it first.
2. SST rolled read-out (e.1 above) — flagship for Chris's directive.
3. Forced rolls (c) — plan doc + build + tests.
4. tpu_train_s2.sh: post-loop final save + trainer-exit reap +
   (from the FGN session's list) snapshot_head.sh delete-then-upload.
5. FGN JAX phase 2 (only if the FGN pair wins).
6. Housekeeping: pixelmae-472/-473/-477 Actions artifacts expire
   2026-09-24 (rescue to releases if still wanted); ml/CLAUDE.md §3/§8
   cite stale temporal.py line numbers (real sites :3490/:3302); T&D
   CI flakes (tide-live app.spec:3476 + slow-source :3754) — if
   tide-live fails on EVERY run since 08-27 it may be date/moon-geometry
   sensitive: reproduce locally with a frozen date before touching it.

## 4 · REPORTING & PAPER

§0f four-section reports to Chris at each substantive landing (#494
verdict, #503 skill read, FGN pair, E-054b). Plain English first,
numbers after; links one per line. When #503's verdict + battery land,
update `ml/paper/paper.tex` (build `bash build.sh --figs`, deliver BOTH
PDFs via SendUserFile + the permalink block — root CLAUDE.md §6b;
project backup via project_write). The E-053 wave synthesis and E-054a
are not yet in the paper either — one paper pass covering E-053/E-054/
E-056/#503 is due once #503 reads out. Keep OVERVIEW.md + expectations
stamped at every touch; extend §3b's replicate table if any pair lands.

## 5 · RESERVED FOR CHRIS (do not decide unilaterally)

Cutting #503's tail early (propose with the battery-completeness
arithmetic); any E-056c $100-class dispatch (present the gate verdict
first); the 800M question; cancelling any other session's run. Chris
answers in chat — options in prose, name your pick, proceed reversibly;
NEVER AskUserQuestion widgets (root CLAUDE.md §0).
