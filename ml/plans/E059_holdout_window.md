# E-059 · The memorization-controlled head

**Retrain E-051 bit-for-bit except for one thing: actually hold out the
held-out years.** Written 2026-08-28, before the node launched; the
predicted numbers below are pre-registration, and §5 records what the node
then printed.

> **Provenance note (2026-08-28 19:4xZ).** This file was committed EMPTY in
> `fc83585` and stayed 0 bytes through `2d1b20a` — the write never landed,
> and neither commit noticed because `docs.html` registers the path, not its
> size. The content below was reconstructed from the dispatch record, the
> node's own log and `ml/EXPERIMENTS.md#e-059`. Every number in §3 and §5 is
> read back from the run's artefacts, not from memory.

---

## 1. The bug

Stage 2 is a causal transformer over a window of K = 144 frames. Its loss is
**dense**: `win_ztgt` (`ml/temporal.py:2819`) supervises, at *every one of
the K frames*, the embedding one bin after that frame's own time. A window
anchored at end-bin `t` therefore scores targets at roughly `t-142 … t+1`.

The training pool did not know that. `ok_t` (the legacy expression, now
preserved verbatim inside `build_window_pool`, `ml/temporal.py:619`) admitted
a window if and only if its **final scored bin** `t + reach` was not held
out:

```python
ok_t = np.array([t + reach[-1] < T and t >= CTX_BACK
                 and not any(t_hold[t + r] for r in reach)
                 for t in range(T)])
```

So a window whose endpoint sits safely in 2010 but whose body straddles 2009
was admitted, and every 2009 transition inside it was **teacher-forced into
the weights**. The model saw the held-out years' one-step dynamics as
training signal, then was evaluated on them.

This is not the roll leaking. `ml/rollout_spatial.py:880` breaks at the year
boundary (`ax.year[s + h] != int(Y)`), proven by the artefact's own
per-horizon `n` falling 3:2:1 — 2,171,138 / 1,447,424 / 723,710. The roll
protocol is clean. **The training pool was not.** Every archived head, the
monthly champion included, was trained this way.

Chris, on being shown the mechanism: *"let's fix training. I can't believe we
had this bug for so long. The agent was always saying to me: 2009 is _never_
used as a target. Not just if the last t ends up in 2009."*

## 2. What #503 looks like in that light

#503 rolled `head-weights-e051-398k-xl144zn-pentad-s0` to a **day-matched
corridor AUC of 0.944** with a **flat skill-vs-lead profile**. Flat is the
signature to be suspicious of: genuine forward physics decays with lead,
calendar/sequence recall does not. The dense-loss/pool mismatch supplies a
mechanism for exactly that shape, so 0.944 is under suspicion until a
memorization-controlled head is rolled the same way.

## 3. The three scopes and their measured cost

`--holdout-scope`, `ml/temporal.py` + mirrored in `ml/jaxport/train_stage2.py`
(which *imports* `build_window_pool` and `frame_target_keep` rather than
copying them, so the two paths cannot drift):

| scope | rule | end-bins | windows | scored frame-targets | held out among them |
|---|---|---|---|---|---|
| `endpoint_contaminated` | legacy: only the final scored bin must be clean | 2,779 | 240,933,742 | 400,176 | **21,018** |
| `target` | held-out **targets** masked out of the loss; held-out bins may still appear as **context** | 2,779 | 240,933,742 | 379,158 (−5.25%) | 0 |
| `window` | nothing the forward pass **touches** — the 144 frames, each frame's teacher-forced target, the scored reach — may be held out | 2,417 | 209,549,066 | 348,048 (−13.03%) | 0 |

Measured on this experiment's own axis (T = 3142 pentad bins, K = 144,
219 held-out bins, 86,698 ocean pixels), by the code, not by hand.

**The 21,018 is the bug as a number**: that many per-frame targets in
E-051's own training were bins its own evaluation then scored.

Strictness costs **7.8 percentage points** of supervision, not a factor —
this was Chris's open question (*"not sure what fraction of training data
we're throwing out out of principle"*), and 5.25% vs 13.03% is the answer.

**Defaults (amended 2026-08-28, commit 58eb286, at Chris's instruction —
*"you don't want training data contamination to be the default"*):** the
legacy value is named `endpoint_contaminated`, never selected implicitly;
the default everywhere, launcher included, is **`window`**.

E-059 runs at `window`. The claim under test is forecast skill on the
held-out years; `target`'s residual context channel would have to be argued
rather than shown. An arm at `target` sits exactly between E-051 and E-059
and would split the memorization term into its teacher-forcing and context
halves — one dispatch away if the headline warrants it.

**Non-vacuous bit-identity, proven twice** (31/31 parameter tensors,
identical loss curves): tree at `endpoint_contaminated` ≡ HEAD's old default
run, and tree's new default `window` ≡ HEAD's explicit `--holdout-scope
window` run. The identities mean something because contaminated and window
differ in **all 31** tensors.

## 4. The run

**E-059 · params 206.659 M · stage 2 · data `family4_na025_pentad_r2`
(`37e146384b`) · arch 1024×16, K 144, stencil 145 ring
`spiral:111-4444-0.71-0.5`, znoise 0.7, grad-clip 128, seed 0, frozen
run-415 codec, published Z · two phases matching the rolled artefact's own
records (200k @ lr 1e-3 hl 40k, then → 400k @ 4e-4 hl 100k) · resume none
(fresh) · JAX / v5litepod-4, node `e059-window`, us-west4-a SPOT · THE ONE
CHANGE: `--holdout-scope window`.**

Phase 2 is a relaunch under the **same node name** with `STEPS=400000`,
`LR=4e-4`, `LR_HALFLIFE=100000` sed'd into `/tmp/e059_startup.sh`; the
resume is exact.

## 5. Pre-registered first-minutes checks — ALL PASSED

Predicted before launch, printed by the node at 17:5xZ, **every one to the
digit**:

| check | predicted | printed |
|---|---|---|
| pooled end-bins | 2,417 | `2,417 end-bins remain in the pool` |
| pool certificate | 0 violations | `0 of 2,417 pooled end-bins touch a held-out bin (350,465 bin checks)` |
| train windows | 209,549,066 | `train windows: 209,549,066` |
| `stage2_config.holdout_scope` | `window` | `window` |
| `stage2_monitor.val_persistence` | 21.44621 | 21.44621 |

`val_persistence` matching E-051's **to all six digits** is the load-bearing
one: it certifies that the validation set is unchanged, so E-051 and E-059
are scored against the same windows and the same baseline. Only the
training pool moved.

## 6. Registered readings

1. **One-step ratio** (`stage2_val_zmse / val_persistence`) at 200k and 400k
   vs E-051's **0.03304** and **0.02981** on the identical
   `val_persistence` 21.44621. The gap **IS the memorization term at h = 1**
   — E-051's val targets are holdout bins whose transitions its pool trained
   on. A worse ratio here is the honest number, not a regression.
2. **The roll**, same protocol as #503. The **shape** is the headline:
   decay ⇒ forward skill; flat ⇒ the E-051 head's signature reproduced under
   a clean pool, which would clear 0.944 of the memorization charge and send
   the search elsewhere.
3. The standard battery.

## 7. Falsifiers

- **The fix is inert** if E-059's val curve tracks E-051's within noise.
  Then the 21,018 contaminated targets, 5.25% of the supervision, carried no
  information the model did not already have, and 0.944 needs another
  explanation.
- **The fix is the whole story** if E-059's roll decays where #503's was
  flat, at a materially lower AUC.
- **Something else is wrong** if E-059's *training* curve also departs from
  E-051's. The pool changed; the optimisation did not. A train-side
  divergence beyond the ~13% supervision cut would mean the change did more
  than it was supposed to.
- **The roll, not the loss, is the leak** if E-059's one-step ratio recovers
  to E-051's by 400k *and* the roll stays flat.

## 8. The companion experiment

**#510 (E-051-roll-B)** rolls the OLD head again with the battery shortened
to 36 months each way (`longm:36,futm:36` → 219 + 219 axis steps) so the job
fits inside its 24 h token. Two independent things come out of one run:

- its **441 skill steps** must reproduce #503's harvested partial
  (day-matched corridor AUC 0.944) — the **protocol-determinism
  certificate**;
- its **future roll** runs past the end of the record, where **nothing
  existed to memorize**. A sharp break there from the in-record tracking
  convicts 0.944 as recall, independently of anything E-059 does.

## 9. Downstream, if the fix bites

The **monthly champion** must be retrained at `scope=window`
(`xl144-nolonhold` on the monthly tensor). Every published skill number in
this programme descends from a head trained under the contaminated pool.
