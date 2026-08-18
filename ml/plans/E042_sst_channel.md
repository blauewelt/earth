# E-042 · SST becomes channel 40 — the first channel that covers the tensor's own axis

**Status: BUILT, NOT DISPATCHED (2026-08-18).** The fetcher, the bake
workflow and both r2 recipes are on `main` and tested; the SST artifact is
published and its bytes are verified on the Hub. Nothing has been trained on
it. The A/B in §5 is waiting on a box with the disk for a 34.0 GB tensor
build, not on a decision.

**A numbering note, because git history cannot be fixed.** Four commit
messages from 2026-08-18 (`bcacf89`, `e2438c7`, and the two follow-ups) and
several code comments label this work **E-041**. That number was already
spent on the globe playback feature (`ml/plans/E041_playback.md`, shipped the
same day). The work is **E-042**; the commit messages stay wrong, because
rewriting published history to fix a label is a worse trade than this
paragraph. A future reader who greps `E-041` in the SST code is in the right
place.

---

## 1 · The hole this fills, in coverage percentages

The 39 channels of families 3/4/5 are almost entirely AMOC plumbing: 3 GLORYS
surface fields, 32 Roemmich–Gilson Argo levels, 4 NCEP wind-stress channels.
Chris, 2026-08-18, on reading the channel table: the embedding *"should
represent a holistic view on any point of the world"*, and the table looked
like it was assembled to answer one question.

**The only temperature in the tensor is `rg_t`, and it is absent for half the
axis.** `build_family4.fill_rg_pentad` walks the RG cubes from
`y, m = 2004 + k // 12, k % 12 + 1` (`ml/build_family4.py:294`) — **Argo
starts in 2004**. So:

| | `rg_t` (16 levels, the only temperature) | `sst` (this channel) |
|---|---|---|
| record | 2004-01 onward | 1982-01-01 onward |
| years of the 1982–2024 axis with NO temperature at all | **22 of 43** (1982–2003) | none |
| native grid | $1^\circ$, lifted to $0.25^\circ$ by bilinear | **$0.25^\circ$, the tensor's own grid** |
| native cadence | monthly | daily |
| missing-token share, pentad bins (Argo era only) | **83.6%** | ~0% |
| missing-token share, daily bins (Argo era only) | **96.7%** | ~0% |
| missing-token share over the WHOLE axis, pentad / daily | **92% / 98%** | ~0% |

The Argo shares follow from E-034 §4's policy — one live bin per month, the
bin containing the 15th, a learned `missing` token everywhere else — times the
Argo era's 21 of 43 years. That policy is right and is not what this
experiment questions: forward-filling would tell the model the subsurface was
observed on days it was not. The point is narrower and it is arithmetic. **On
the temperature axis the tensor is 92–98% missing token, and it is 100%
missing before 2004** — which is the decade the Florida cable makes most
valuable, and the decade E-038 §3 argued hardest to keep.

SST fills exactly that hole and nothing else: daily, $0.25^\circ$, live in
essentially every bin from 1982-01-01 to 2024-12-31.

## 2 · What was built

Three pieces, all on `main`, all with tests that run offline in seconds.

**`ml/fetch_sst_na.py`** streams `sst.day.mean.YYYY.nc` from PSL one year at a
time (~476 MB), crops to the E-034 window and interpolates each day onto the
tensor's own axes, deleting each source before pulling the next. Measured on
1993: 365 rows folded in **14.5 s at 0.26 GB peak RSS**, so a 43-year run is
bounded by download, not arithmetic. Output contract:
`sst_daily_na.npy` int16 at 0.01 °C, nodata −32768, day-major
`(NDAYS, 281, 481)`, plus `index.npz` carrying `bin_index` (days since
1982-01-01), `has_data`, `epoch`, `cadence_days=1`, `scale`, `nodata`, `lat`,
`lon`.

**`.github/workflows/sst-na-bake.yml`** publishes it (`workflow_dispatch`
only, `runs-on: ubuntu-latest` — never `gpu`, ml/CLAUDE.md §6). **Run #1
completed and the bytes are verified on the Hub**, `chfrank/earth-tensors`:

| file | bytes |
|---|---|
| `sst_na025/index.npz` | 146,802 |
| `sst_na025/sst_daily_na.npy` | **4,245,677,460** |

That second number is the artefact checking itself: $15{,}706 \times 281
\times 481 \times 2 = 4{,}245{,}677{,}332$, plus numpy's 128-byte header, is
4,245,677,460 exactly. The file is the **whole** 1982-01-01…2024-12-31 axis at
the window's full grid, not a truncated run that finished early.

**Recipes `f4r2` / `f5r2`.** `CHANS_R2 = list(f3.CHANS) + ["sst"]` —
SST is channel 40 (index 39). Each `(cadence, rev)` pair carries its own
recipe string *and its own output name*
(`family4_na025_pentad_r2.npz`, `family5_na025_daily_r2.npz`), because
`ml-train.yml` derives `$TENSOR` from the `tensor` input verbatim: the file
stems **are** the dispatch values. Two new VALUES on an existing input, never
a 26th input — the workflow sits exactly at the 25-input ceiling and a 26th
makes every dispatch 422 (ml/CLAUDE.md §7). Both r2 branches seed
`sst_na025/` from the Hub and are **fatal** on a failed fetch, because
`build_family4.py` warns and leaves the channel entirely missing when the
artefact is absent — a silent fetch failure would produce a run that trains,
goes green, and reports E-042 numbers for an experiment in which SST never
appeared.

## 3 · The design decisions, and the alternatives rejected

**Bilinear, not nearest — and this one is load-bearing.** OISST's `lat`/`lon`
are cell **centres**, $0.125 + k\cdot 0.25$; the ML window samples **on**
multiples of $0.25$. Every target point therefore falls exactly halfway
between two source centres in *each* axis, so nearest-indexing would displace
the entire field by half a cell in both directions — consistently, and in a
way that looks like a perfectly ordinary SST map. It would be invisible in
every plot and fatal to every stencil and to the AMOC eval mask. The
interpolation reuses `build_family3.lin_weights` / `interp2_nan`, the same
machinery the wind channel already uses, so the two channels cannot disagree
about what "on this grid" means; the 360/0 seam is wrap-interpolated. Test 2
of `tests/test_sst_na.py` measures reproduction of an analytic ramp at
**7.6e-07 °C**.

**Appended, never inserted.** `build_family3.py` is untouched, channels 1–39
keep the exact indices every published result was measured at, and an r1 and
an r2 tensor are diffable channel by channel. A test pins r2's channels 0–38
**bit-identical** to the r1 tensor built from the same fixtures — float16 for
float16, NaN pattern, `norm` and axis included. Inserting SST in a physically
tidy place (beside `rg_t10`, say) would have renumbered 32 channels and
invalidated the index every result quotes. Channel *order* carries no
information the model uses: identity comes from `chan_emb`, an
`nn.Embedding(n_chan, d_model)` trained from scratch per run, so "last" costs
nothing.

**A mean, not a sample, at pentad.** Unlike RG, SST is observed *every* day,
so a pentad bin has five real observations and the NaN-aware mean is the
bin's state; taking one day would throw four fifths away and alias the storm
band. At daily the same code path runs at `days=1` and the bin is the day.
Unlike the wind σ channels, a mean is aggregable from dailies, so no second
cadence-specific estimator is needed.

**The rev is a FLAG, never inferred from disk.** `--rev r2` is passed
explicitly; the recipe guard refuses to reuse an r1 cache under r2's name.
"39 channels and no sst" is exactly the artefact that would train cleanly and
mean something else. A recipe the filesystem decides is not a recipe.

**The climatology is deliberately NOT shared with the app's SST bake.** This
file emits the **field**. The pipeline's anomaly baseline is computed in
`temporal.py`/`trainprobe.py` from **train years only**; the app uses the WMO
1991–2020 normal, which **includes the holdout years**. They look
interchangeable and are not — substituting one for the other leaks
test-period information into training through the baseline, so the model
would be centred using data it is meant to be scored against. This is E-040
§5's rule, restated here because this is where a future refactor will notice
"we compute the climatology twice".

**Rejected: reuse the app's pixel-major file.** `scripts/bake_sst_daily.py`
writes the same download pixel-major for the browser's 2-byte range reads.
The two layouts are transposes; building the tensor from the pixel-major file
means reading 757 MB/year back and transposing it. Sharing happens at the
**download** level, not the artefact level — each consumer takes its natural
cut of one stream.

**Rejected: upsampling the existing 1° monthly SST bake.** That is what the
tensor's SST would have been under the old plan and it carries no sub-degree
structure at all. OISST is an AVHRR analysis, so its *effective* resolution is
coarser than its 0.25° posting — but it is real sub-degree information on the
tensor's own grid, against a 1° bin that has none.

## 4 · The consequence that must be stated before dispatch

**The frozen 39-channel anchor CANNOT encode a 40-channel tensor.** `f3_anchor41M`
(run #62/#80) has a `chan_emb` of 39 rows and a per-channel input projection;
there is no honest way to hand it a fortieth. So **E-038's frozen-codec
baseline retires on r2**. On r2 the only surviving baseline is the wind-only
ridge — **0.670** at pentad, which is the bar the frozen anchor itself failed
to clear (#390: codec 0.660 [0.593, 0.722] against wind-only 0.670 [0.601,
0.733]).

This is not a reason to avoid r2; it is a reason the A/B below is between two
*trained* codecs rather than against the frozen control. A number without its
baseline is not a result (ml/CLAUDE.md §3), and on r2 the baseline is #386
itself.

## 5 · The A/B: one difference, and it is the tensor

A matched pair against run **#386** (E-038a, f4-40M on r1). Every dispatch
field identical, copied verbatim from #386's own `INPUTS_JSON` block per
ml/CLAUDE.md §1, with `tensor` the **only** change:

| field | #386 (control) | E-042 arm |
|---|---|---|
| `tensor` | `family4_na025_pentad` | **`family4_na025_pentad_r2`** |
| steps | 200,000 | 200,000 |
| batch | 512 | 512 |
| `d_z` | 32 | 32 |
| `patch` | 1 | 1 |
| `d_model` / `n_layers` / `n_heads` / `d_dec` | 512 / 12 / 4 / 256 | same |
| params | 37,975,889 | **37,976,465** (+576: `chan_emb` +512 and `q_chan` +64, one row each — measured by instantiating the real class) |
| `anomaly` | true | true |
| `eval_every` / `light_probe_every` | 7,500 / 10,000 | same |
| `resume` | null (fresh codec) | null |

The arm is dispatched as `window: recipe:f4r2-40M` — not hand-assembled, and
**not** as `recipe:f4-40M` with the tensor overridden, which is what an
earlier version of this paragraph said and which does not work. `f4-40M.json`
PINS `"tensor": "family4_na025_pentad"`, and every consumption site
(`ml-train.yml:762`, `:881`, `scripts/probes_run.sh:43`, plus the build
branches at 378-412 and 518-591) reads `${RECIPE_TENSOR:-$IN_TENSOR}`. The
`:-` fallback fires only when `RECIPE_TENSOR` is UNSET, so a recipe that names
`tensor` discards `inputs.tensor` entirely — while `provenance.json`, which
records raw `toJSON(inputs)`, would faithfully report `family4_na025_pentad_r2`.
That run would train **r1**, go green, and publish E-042 numbers for an
experiment in which SST never appeared: intent and reality disagreeing in
exactly the way the manifest exists to catch. A recipe's tensor is not
overridable from a dispatch, so the r2 arm gets its own recipe
(`ml/recipes/f4r2-40M.json`, identical but for the tensor).
`head_probe: "true"`, because at pentad cadence the unpooled head is the
primary read-out and the pooled ridge is the comparable-to-history number,
never the verdict (ml/CLAUDE.md §3).

**One arm is one seed, and one seed is not a result.** The stage-2 seed sd is
0.123 on the RAPID head k-fold (E-010), so a codec-level difference smaller
than that band cannot be read off a single pair. What the pair CAN establish
is whether SST moves the number at all and in which direction; a difference
worth quoting needs replicates, and this entry should not be closed on n=1.

## 6 · Hypothesis and falsifier

**Hypothesis.** A codec of the same capacity trained on the 40-channel r2
pentad tensor produces embeddings that read the RAPID transport better than
the same codec trained on r1 — because 22 of 43 years gain their first
temperature field, and because the added channel is the only one native to
the tensor's grid.

**Falsified if** the r2 arm's `probe_kfold` rapid r does not exceed #386's,
or exceeds it by less than the seed band, at the same step count. Two
readings of that outcome, and they are distinguishable:

- **SST is redundant given GLORYS.** `cur_speed`/`ssh`/`log_mld` cover
  1993–2024 and are dynamically tied to the surface temperature field, so
  post-1993 SST may add little. **The discriminating check is the 1982–92
  decade**, where GLORYS is absent and only wind and the cable exist: score
  the two arms on that block separately. If r2 wins there and nowhere else,
  the channel is a coverage fix rather than an information gain, and it is
  still worth keeping for the daily arm's ~4,000 cable labels in that decade.
- **The transport probe cannot see it.** The probe reads the codec through a
  26.5°N section; a basin-scale surface-temperature field may be genuinely
  present in `z` and invisible to that instrument. `probe_sequence` and the
  reconstruction loss on the sst channel itself distinguish "not encoded"
  from "not read out" — and if the second is the story, that is an argument
  about the read-out, not about the channel.

**A result that would retire the channel outright:** r2 measurably WORSE than
r1 beyond the seed band, at matched steps. That would say the fortieth
channel costs capacity it does not repay — the same budget spread over
2.6% more input tokens — and the honest response is to drop it rather than to
grow the codec until it pays.

## 7 · What blocks it, and it is disk

The r2 pentad tensor is $[3142, 281, 481, 40]$ float16 = **34.0 GB** (against
r1's 33.1). The builder's own guard refuses to start unless the dense memmap
fits in 95% of free space, and the build wants roughly **42 GB** free while
the memmap and the written archive coexist. No box in the fleet has that
headroom today — #386/#387 are in flight on r1 on the 126 GB boxes and their
own tensors and caches occupy them.

So the order of work is:

1. Free or rent a box with ~50 GB spare, and build `family4_na025_pentad_r2`
   there (`--rev r2 --sst-dir`, seeded from the Hub). Read the printed
   Chinchilla inventory: 40 channels change the observed-value count and
   therefore the anchor.
2. Dispatch the matched arm in §5. Nothing else changes.
3. Report against **#386**, with the wind-only 0.670 stated beside it as the
   floor, and the frozen anchor explicitly marked *not applicable at r2* (§4).
4. Only then consider `family5_na025_daily_r2` — 169.8 GB, which needs the
   600 GB+ box E-038 §3 already argued for, and which is the arm where the
   1982–92 decade's ~4,000 daily cable labels sit beside a live temperature
   field for the first time.

The daily track has its own reasons for waiting, all fixed today and all
recorded in `ml/EXPERIMENTS.md`: `anomaly_transform`'s 249 full-extent
traversals (which hung #389 for 7 h), two hand-inlined copies of that
transform that zeroed 100% of dynamic channels at float16, and
`probe_head.py`'s 82.8 GB transient inside `np.nan_to_num`. None of them is
E-042's hypothesis; all of them are why the daily rung is not the next thing
dispatched.

---

*Written 2026-08-18. Coverage percentages derived from E-034 §4's Argo policy
and the 2004 start in `build_family4.fill_rg_pentad`; artefact sizes read from
the published Hub files; the interpolation error and the timing figures are
this session's measurements, quoted from `ml/fetch_sst_na.py`'s own tests and
its 1993 run.*
