#!/usr/bin/env python3
"""E-057: rolling an FGN head as an ENSEMBLE, and proving nothing else moved.

    python3 tests/test_fgn_roll.py

Spec: `ml/plans/E057_roll_spec.md`. Plan: `ml/plans/E057_fgn_head.md`. The
evaluator under test is `ml/rollout_spatial.py`; the scoring functions are
`ml/probscore.py`'s and are never re-derived here — where a number is
asserted, it is asserted against a DIRECT probscore call on the same arrays,
so this file pins the WIRING and probscore's own suite pins the estimators.

The seven checks, in the spec's order:

  1. **Deterministic-path purity — the acceptance bar.** Two independent
     forms, because one of them can rot and the other cannot. (a) `roll_step`
     with `eps=None` is BITWISE the pre-patch function on random tensors, at
     stencil 1 and 9 and at three chunk sizes — against a verbatim inline copy
     of the pre-E-057 body, which no future commit can quietly update. (b) the
     whole evaluator, run on the deterministic toy, produces a results tree
     DEEP-EQUAL to the one the pristine `ml/rollout_spatial.py` produces from
     the same inputs — the pristine file being recovered from git history as
     the newest blob of that path with no `E-057` in it, so the check survives
     the main session committing this diff. ONE key is excluded from that
     compare and it is not a clock: E-058's `per_channel`, which the pristine
     evaluator does not write at all. It is removed by its own named strip,
     COUNTED against the scored scopes and PINNED against the pooled rows and
     the E-026b audit block before it is believed — `pin_per_channel`.
     (c) not one E-057 key appears
     anywhere in a deterministic head's entry, and no label carries the token.
  2. **Member determinism** — same `--ens-seed` twice ⇒ deep-equal results;
     a different `--ens-seed` ⇒ different members, and (the control) a head
     that ignores ε is unmoved by the seed.
  3. **Shared ε within a member-step** — a recording wrapper around the model
     proves every chunk of one roll step saw ONE ε, expanded to that chunk's
     rows; and the field is chunk-size invariant to float error, with a
     deterministic control establishing that the residual is BLAS batching and
     not the noise path (see the note at `test_shared_eps`).
  4. **CRPS wiring** — members == truth ⇒ 0 exactly; identical members ⇒ the
     MAE identity exactly; and `accumulate_ens`/`ens_block` aggregated over
     two starts equal a direct `probscore` call on the pooled cells.
  5. **`--ens-members 1` refuses**, at argv time, before the codec is opened.
  6. **`--ens-members` is a no-op for a deterministic head** — byte-identical
     results with the flag set to an arbitrary value.
  7. **Dispersion arrays** — right length, finite, ≥ 0, floor reported; and a
     ZERO-FILM head (ε ignored at init, the collapse signature) returns a
     transport spread of exactly 0 and a field variance at the float32 floor.

No GPU: the 79-pixel monthly synthetic ocean of tests/test_rollout_spatial.py,
whose `build_fixture` is imported rather than re-implemented.
"""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)
from test_rollout_spatial import build_fixture, DZ, K            # noqa: E402
import probscore                                                 # noqa: E402
import rollout_spatial as RS                                     # noqa: E402
from temporal import TemporalTransformer                         # noqa: E402

EPS_DIM = 6
M_ENS = 3
HORIZON, STARTS, N_LONG, N_FUT = 3, 2, 5, 3


# ---------------------------------------------------------------------------
# 1(a): the PRE-PATCH roll_step, verbatim.
# ---------------------------------------------------------------------------
# Copied unchanged from `ml/rollout_spatial.py` at the commit before E-057
# (`git show HEAD~:ml/rollout_spatial.py`, function `roll_step`), comments
# stripped, body byte-for-byte otherwise. It lives HERE, in the test, on
# purpose: a reference read out of the tree under test can be edited by the
# same change it is supposed to police, and then it certifies nothing.
def roll_step_prepatch(model, Zwin, NBR_t, static_ctx, mfeat, chunk,
                       amp=False, quant=None):
    if quant is not None:
        Zwin = quant(Zwin)
    P = Zwin.shape[0]
    if NBR_t is not None:
        S, Kw, dz = NBR_t.shape[1], Zwin.shape[1], Zwin.shape[2]
        row_bytes = S * Kw * dz * 4
        chunk = max(256, min(chunk, (1 << 30) // max(1, row_bytes)))
    outs = []
    for i in range(0, P, chunk):
        sl = slice(i, min(i + chunk, P))
        if NBR_t is None:
            zin = Zwin[sl]
        else:
            nbr = NBR_t[sl]
            miss = nbr < 0
            zj = Zwin[nbr.clamp(min=0)]
            zj[miss] = 0.0
            zin = zj.permute(0, 2, 1, 3).reshape(
                zj.shape[0], Zwin.shape[1], -1)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp):
            pred, _ = model(zin,
                            mfeat[None].expand(zin.shape[0], -1, -1),
                            static_ctx[sl])
        outs.append(pred[:, -1].float())
    return torch.cat(outs, 0)


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def fgn_head(tmp, name, seed, zero_film, eps_dim=EPS_DIM, stencil=1):
    """A toy FGN head checkpoint: `args.fgn_eps > 0` is the ONLY thing that
    puts the evaluator on the ensemble path.

    `zero_film=True` leaves `_CondLayer.film` at its zero init, which is the
    state ml/temporal.py documents as "the ε-path is the identity" — so the
    head is an FGN head by its args and a DETERMINISTIC function of its input
    in fact. That is exactly the ε-collapse the dispersion instrument exists
    to make visible, and check 7 uses it as the collapse control.
    """
    torch.manual_seed(10 + seed)
    hm = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                             k_max=K if stencil == 1 else max(K, 36),
                             stencil=stencil, eps_dim=eps_dim)
    if not zero_film:
        # A trained head's film is not zero; a randomly perturbed one stands
        # in for it. Everything else keeps the deterministic head's init.
        with torch.no_grad():
            for lay in hm.encoder.layers:
                lay.film.weight.normal_(0.0, 0.5)
                lay.film.bias.normal_(0.0, 0.5)
    hp = os.path.join(tmp, f"{name}.pt")
    torch.save({"model": hm.state_dict(),
                "args": {"K": K, "d_model": 8, "layers": 1, "unroll": 1,
                         "seed": seed, "stencil": stencil,
                         "fgn_eps": eps_dim}}, hp)
    return hp


def run(script, f, out, heads, extra=(), expect_ok=True):
    """The evaluator (patched or pristine) on the toy. Returns (obj, stdout)."""
    env = dict(os.environ,
               PYTHONPATH=ML + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-u", script,
           "--x", f["x"], "--npz-small", f["npz"], "--z", f["z"],
           "--ckpt", f["ckpt"], "--out", out,
           "--horizon", str(HORIZON), "--starts-per-year", str(STARTS),
           "--long-start", "1991-12", "--long-months", str(N_LONG),
           "--future-months", str(N_FUT), "--cache-dir", f["cache"],
           "--no-gate", *extra, "--heads", *heads]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                       cwd=ROOT, env=env)
    if expect_ok and r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f"{os.path.basename(script)} failed (rc "
                         f"{r.returncode})")
    if not expect_ok:
        return r, r.stdout + r.stderr
    return json.load(open(out)), r.stdout


# Clock readings, and nothing else. Every one of these is a duration or a
# timestamp; none is a number the roll computed. They are stripped RECURSIVELY
# because `probe_unpooled.fit_wall_s` sits three levels down and is the one
# field that made the first purity comparison fail for a reason that was not a
# behaviour change.
CLOCKS = ("wall_s", "created_utc", "elapsed_min", "fit_wall_s", "device")


def strip_clocks(o):
    if isinstance(o, dict):
        return {k: strip_clocks(v) for k, v in o.items()
                if not (k in CLOCKS or k.endswith("_wall_s"))}
    if isinstance(o, list):
        return [strip_clocks(v) for v in o]
    return o


# E-058's `per_channel` — a channel-by-channel skill read-out, `sst` finally
# visible instead of pooled into a 40-channel average — is the one key the
# purity compare at 1b must ignore, because the PRISTINE evaluator predates it
# and does not write it at all. It gets its own named strip rather than a line
# in CLOCKS: CLOCKS is a promise that everything it drops is a reading of the
# wall clock, and `per_channel` is a RESULT. A result may only be dropped from
# a purity compare once someone has said what it must contain — which is what
# `pin_per_channel` says, and why the two functions arrive together.
E058_KEY = "per_channel"


def strip_per_channel(o, found=None):
    """(tree without `per_channel`, the blocks that were removed).

    Recursive for `strip_clocks`' reason: the key sits at
    `heads.<label>.<scope>`, and a strip that hardcoded that depth would stop
    policing on the day the depth changed."""
    found = [] if found is None else found
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k == E058_KEY:
                found.append(v)
                continue
            out[k] = strip_per_channel(v, found)[0]
        return out, found
    if isinstance(o, list):
        return [strip_per_channel(v, found)[0] for v in o], found
    return o, found


def pin_per_channel(res):
    """The three statements that make the strip above a removal of an ADDITION
    rather than a hole in the compare. Returns (scored scopes, channel rows
    checked, max |delta| against the audit block).

    (i) COVERAGE — every scope with pooled rows carries the key, so a harvest
        can never find a scope where only the 40-channel pool exists.
    (ii) THE SAME CELLS — a pooled row's `n` is its channels' `n` added up.
        `accumulate` fills the pooled sums and the per-channel sums from the
        same [n_pixels, C] arrays on the same early-exit branch, so counts
        that do not add up mean the two were accumulated over different cells.
    (iii) THE SAME NUMBERS — tests/test_per_channel_skill.py check 2 proves
        the pooled msss_clim IS the climatology-mass-weighted mean of the
        per-channel ones, to 1.11e-16. Those weights (`mse_c` per channel) are
        not published, so this file checks the two consequences of that same
        relation a reader of the ARTEFACT can also check: the pooled value
        never leaves its own channels' [min, max] (a mean with positive
        weights cannot leave the convex hull of its terms), and on the
        corridor scope it agrees with the E-026b audit block — a per-channel
        decomposition of the SAME quantity that predates E-058 and is
        accumulated independently in `aud["ch_m"]/["ch_c"]`. The two divide by
        `n` at different points (`1 - (m/n)/(c/n)` against `1 - m/c`), so they
        can differ by at most one unit in the last PUBLISHED place; anything
        larger is a different accumulation, not a rounding step.
    """
    n_scope = n_row = 0
    dev = 0.0
    for lab, e in res["heads"].items():
        au = e["audit"]
        for name, blk in e.items():
            if not (isinstance(blk, dict) and blk.get("chan_skill")):
                continue
            per = blk.get(E058_KEY)
            assert per, (f"{lab}/{name} has pooled rows but no `{E058_KEY}` — "
                         f"a scope whose sst read-out no harvest can ask for")
            assert list(per) == au["channels"], (
                f"{lab}/{name}: `{E058_KEY}` is keyed {list(per)} but the "
                f"artefact's own channel list is {au['channels']} — the names "
                f"must be the TENSOR's, or the rows below are pinned against "
                f"the wrong channel")
            n_scope += 1
            for r in blk["chan_skill"]:
                mine = [p for c in per.values() for p in c if p["h"] == r["h"]]
                n_row += len(mine)
                tot = sum(p["n"] for p in mine)
                assert tot == r["n"], (
                    f"{lab}/{name} h{r['h']}: pooled n {r['n']} but the "
                    f"channels count {tot} — the two sum sets were "
                    f"accumulated over different cells")
                v = [p["msss_clim"] for p in mine]
                assert min(v) <= r["msss_clim"] <= max(v), (
                    f"{lab}/{name} h{r['h']}: pooled msss_clim "
                    f"{r['msss_clim']} lies outside its own channels' "
                    f"[{min(v)}, {max(v)}] — it is then not a pooling of them")
            if name != "corridor":
                continue
            for ci, c in enumerate(per):
                for p in per[c]:
                    a_ = au["per_channel_msss_clim_corridor"][p["h"] - 1][ci]
                    assert a_ is not None \
                        and abs(a_ - p["msss_clim"]) <= 1e-3, (
                            f"{lab}/corridor {c} h{p['h']}: `{E058_KEY}` says "
                            f"{p['msss_clim']} where the E-026b audit block's "
                            f"independent decomposition of the same quantity "
                            f"says {a_}")
                    dev = max(dev, abs(a_ - p["msss_clim"]))
    return n_scope, n_row, dev


def all_keys(o, acc=None):
    acc = set() if acc is None else acc
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(k)
            all_keys(v, acc)
    elif isinstance(o, list):
        for v in o:
            all_keys(v, acc)
    return acc


def first_diff(a, b, path="$"):
    """Where two result trees stop agreeing — a path, not a 300 kB dump."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: missing on the left"
            if k not in b:
                return f"{path}.{k}: missing on the right"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def pristine_evaluator(tmp):
    """The newest `ml/rollout_spatial.py` in git history with no E-057 in it,
    written beside the toy so it can be run and imported.

    Deliberately not `HEAD~1`: the main session commits this diff, and a
    reference pinned to a commit OFFSET would then compare the patched file
    against itself and pass for the worst possible reason. Searching by
    CONTENT ("does this blob know about E-057") keeps the comparison honest
    after the commit lands. Returns None where git cannot answer, in which
    case check 1 falls back to (a) + (c), loudly.
    """
    try:
        shas = subprocess.check_output(
            ["git", "log", "--format=%H", "--", "ml/rollout_spatial.py"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL).split()
    except Exception:                                          # noqa: BLE001
        return None
    for sha in shas:
        try:
            blob = subprocess.check_output(
                ["git", "show", f"{sha}:ml/rollout_spatial.py"],
                cwd=ROOT, stderr=subprocess.DEVNULL)
        except Exception:                                      # noqa: BLE001
            continue
        if b"E-057" in blob:
            continue
        path = os.path.join(tmp, "pristine_rollout_spatial.py")
        with open(path, "wb") as fh:
            fh.write(blob)
        return path, sha
    return None


# ---------------------------------------------------------------------------
# 1. deterministic-path purity
# ---------------------------------------------------------------------------
def test_purity_roll_step(pristine_mod):
    """(a) `roll_step(..., eps=None)` is the pre-patch function, bitwise."""
    n_case = 0
    for stencil in (1, 9):
        torch.manual_seed(100 + stencil)
        m = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                                k_max=K, stencil=stencil)
        m.eval()
        P = 61
        Zwin = torch.randn(P, K, DZ)
        mfeat = torch.randn(K, 2)
        NBR_t = None
        sctx = torch.randn(P, DZ + 2)
        if stencil > 1:
            NBR_t = torch.randint(-1, P, (P, stencil))
            sctx = torch.randn(P, DZ + 2 + stencil)
        for chunk in (P, P // 3, 7):
            with torch.no_grad():
                got = RS.roll_step(m, Zwin.clone(), NBR_t, sctx, mfeat,
                                   chunk, False, quant=None, eps=None)
                want = roll_step_prepatch(m, Zwin.clone(), NBR_t, sctx,
                                          mfeat, chunk, False, quant=None)
            assert torch.equal(got, want), \
                (f"roll_step diverged from its pre-patch self at stencil "
                 f"{stencil}, chunk {chunk}: max |Δ| "
                 f"{(got - want).abs().max().item():.3e}")
            n_case += 1
            if pristine_mod is not None:
                with torch.no_grad():
                    old = pristine_mod.roll_step(m, Zwin.clone(), NBR_t, sctx,
                                                 mfeat, chunk, False,
                                                 quant=None)
                assert torch.equal(got, old), \
                    (f"roll_step diverged from the PRISTINE TREE's roll_step "
                     f"at stencil {stencil}, chunk {chunk}")
    print(f"1a. roll_step(eps=None) bitwise identical to the pre-patch body "
          f"in {n_case} (stencil, chunk) cases"
          + ("" if pristine_mod is None else " and to the pristine module")
          + " OK")


def test_purity_end_to_end(f, tmp, pristine):
    """(b) the whole evaluator, det toy, patched vs pristine — deep equal.
    (c) not one E-057 key, and no fgn token in any label."""
    new, _ = run(os.path.join(ML, "rollout_spatial.py"), f,
                 os.path.join(tmp, "det_new.json"), f["heads"])
    forbidden = {"fgn", "ens_prob", "amoc_bands_ens", "amoc_bands_ens_unpooled",
                 "long_dispersion", "future_dispersion", "dispersion",
                 "field_var", "sv_spread", "field_var_floor"}
    got = all_keys(new) & forbidden
    assert not got, f"E-057 keys leaked into a deterministic run: {sorted(got)}"
    for lab in new["heads"]:
        assert "fgn" not in lab, f"deterministic label carries a token: {lab}"
    print(f"1c. deterministic entry carries none of the {len(forbidden)} "
          f"E-057 keys; labels {sorted(new['heads'])} OK")

    if pristine is None:
        print("::warning::1b. SKIPPED — no pre-E-057 blob of "
              "ml/rollout_spatial.py in git history, so the end-to-end "
              "deep compare could not be built. 1a and 1c still hold.")
        return new
    ppath, sha = pristine
    old, _ = run(ppath, f, os.path.join(tmp, "det_old.json"), f["heads"])
    a, b = strip_clocks(new), strip_clocks(old)
    # E-058: `per_channel` exists on the patched side only, so it is removed
    # from the compare — but only after it has been COUNTED against the scored
    # scopes and PINNED against the pooled rows it was computed beside. A
    # silent key skip here would leave one key in the artefact that nothing
    # compares and nothing checks, which is exactly where a real divergence
    # would sit unnoticed.
    a, blocks = strip_per_channel(a)
    base_pc = strip_per_channel(b)[1]
    assert not base_pc, (
        f"the pristine evaluator at {sha[:8]} already writes `{E058_KEY}` — "
        f"the strip would then be hiding a DIFFERENCE, not a new key")
    n_scope, n_row, dev = pin_per_channel(new)
    assert len(blocks) == n_scope > 0, (
        f"{len(blocks)} `{E058_KEY}` blocks for {n_scope} scored scopes — the "
        f"key must be on every scope that has rows")
    d = first_diff(a, b)
    assert d is None, f"patched vs pristine ({sha[:8]}) differ at {d}"
    assert a == b
    print(f"1b. patched evaluator == pristine {sha[:8]} on the deterministic "
          f"toy, deep-equal over {len(all_keys(a))} distinct keys "
          f"(clock readings {CLOCKS} stripped) OK")
    print(f"1b. the one non-clock exclusion: {len(blocks)} `{E058_KEY}` "
          f"blocks (one per scored scope, {n_scope} of them) holding {n_row} "
          f"channel rows, removed and PINNED — the channel `n`s sum to the "
          f"pooled `n`, the pooled msss_clim sits inside its channels' "
          f"[min, max], and the corridor values equal the E-026b audit "
          f"block's own per-channel decomposition (max |Δ| {dev:.1e}) OK")
    return new


# ---------------------------------------------------------------------------
# 2. member determinism
# ---------------------------------------------------------------------------
def test_member_determinism(f, tmp, heads):
    a, _ = run(os.path.join(ML, "rollout_spatial.py"), f,
               os.path.join(tmp, "fgn_s0a.json"), heads,
               extra=("--ens-members", str(M_ENS), "--ens-seed", "0"))
    b, _ = run(os.path.join(ML, "rollout_spatial.py"), f,
               os.path.join(tmp, "fgn_s0b.json"), heads,
               extra=("--ens-members", str(M_ENS), "--ens-seed", "0"))
    dumpdir = os.path.join(tmp, "dump_s1")
    c, _ = run(os.path.join(ML, "rollout_spatial.py"), f,
               os.path.join(tmp, "fgn_s1.json"), heads,
               extra=("--ens-members", str(M_ENS), "--ens-seed", "1",
                      "--dump-roll", dumpdir))
    d = first_diff(strip_clocks(a), strip_clocks(b))
    assert d is None, f"same --ens-seed 0 twice disagreed at {d}"
    print("2. --ens-seed 0 twice: deep-equal results OK")

    live = [k for k in a["heads"] if k.endswith("_s0")][0]
    zero = [k for k in a["heads"] if k.endswith("_s1")][0]

    def numbers(entry):
        """Everything the roll COMPUTED. `meta` is dropped because it records
        `ens_seed` itself, so comparing it would answer "did the seed change"
        with the seed — which is not the question either assertion asks."""
        return strip_clocks({k: v for k, v in entry.items() if k != "meta"})

    assert numbers(a["heads"][live]) != numbers(c["heads"][live]), \
        "a different --ens-seed produced the SAME members — the seed is not " \
        "reaching the noise stream"
    print(f"2. --ens-seed 1 moves the live head ({live}): "
          f"long sv_spread "
          f"{a['heads'][live]['long_dispersion']['sv_spread'][:3]} -> "
          f"{c['heads'][live]['long_dispersion']['sv_spread'][:3]} OK")
    dz_ = first_diff(numbers(a["heads"][zero]), numbers(c["heads"][zero]))
    assert dz_ is None, \
        (f"the zero-film head moved with the seed at {dz_} — it must not: "
         f"its ε path is the identity, so every member is the same "
         f"trajectory whatever ε it is handed")
    print(f"2. the zero-film control ({zero}) is seed-invariant OK")

    # The seed formula itself, and that it is a per-member stream.
    assert RS.member_seed(0, 0) == 59 and RS.member_seed(0, 3) == 62
    assert RS.member_seed(2, 1) == 2 * 1000003 + 60
    g0, g0b, g1 = (RS.member_gen(0, 0), RS.member_gen(0, 0),
                   RS.member_gen(0, 1))
    d0 = torch.randn(4, EPS_DIM, generator=g0)
    assert torch.equal(d0, torch.randn(4, EPS_DIM, generator=g0b))
    assert not torch.equal(d0, torch.randn(4, EPS_DIM, generator=g1))
    assert g0.device.type == "cpu", "member generators must be CPU streams"
    print("2. member_seed/member_gen: reproducible, per-member, CPU OK")

    # SPEC §5: --dump-roll writes MEMBER 0 ONLY, and says so in its meta. The
    # animation is one trajectory; M x the bytes buys nothing for it, and a
    # dump that did not name its member would be M trajectories' worth of
    # ambiguity in a file meant to outlive the run.
    man = json.load(open(os.path.join(dumpdir, "dump_manifest.json")))
    assert man["files"], "an FGN head with --dump-roll wrote no trajectory"
    for rec in man["files"]:
        z = np.load(os.path.join(dumpdir, rec["file"]))
        meta = json.loads(str(z["meta_json"]))
        assert meta["member"] == 0, meta
        assert meta["members"] == M_ENS and meta["fgn_eps"] == EPS_DIM
        assert meta["ens_seed"] == 1, meta
        assert z["z"].shape[0] == rec["n_states"] and z["z"].dtype == \
            np.float16
    n_lab = len({r["head"] for r in man["files"]})
    print(f"2. --dump-roll on {n_lab} FGN head(s): {len(man['files'])} "
          f"trajectories, every one tagged member 0 of {M_ENS} "
          f"(ens_seed 1, fgn_eps {EPS_DIM}) OK")
    return a, live, zero


# ---------------------------------------------------------------------------
# 3. shared eps within a member-step
# ---------------------------------------------------------------------------
class Recorder(torch.nn.Module):
    """Wraps a head and records the `eps` of every forward — which is the
    only direct way to ask "did every pixel of this step see one draw?"."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = []

    def forward(self, z, mo, sc, eps=None):
        self.calls.append((int(z.shape[0]),
                           None if eps is None else eps.detach().clone()))
        return self.inner(z, mo, sc, eps=eps)


def test_shared_eps():
    torch.manual_seed(7)
    inner = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                                k_max=K, stencil=1, eps_dim=EPS_DIM)
    with torch.no_grad():
        for lay in inner.encoder.layers:
            lay.film.weight.normal_(0.0, 0.5)
            lay.film.bias.normal_(0.0, 0.5)
    inner.eval()
    m = Recorder(inner)
    P = 79
    Zwin, mfeat = torch.randn(P, K, DZ), torch.randn(K, 2)
    sctx = torch.randn(P, DZ + 2)
    eps = torch.randn(1, EPS_DIM)

    with torch.no_grad():
        full = RS.roll_step(m, Zwin, None, sctx, mfeat, P, False, eps=eps)
    assert len(m.calls) == 1
    m.calls.clear()
    with torch.no_grad():
        split = RS.roll_step(m, Zwin, None, sctx, mfeat, P // 3, False,
                             eps=eps)
    n_chunk = -(-P // (P // 3))
    assert len(m.calls) == n_chunk, \
        f"expected {n_chunk} chunks, saw {len(m.calls)}"
    seen = 0
    for n_rows, got in m.calls:
        assert got is not None, "eps did not reach the model"
        assert got.shape == (n_rows, EPS_DIM), \
            f"eps arrived as {tuple(got.shape)} for {n_rows} rows"
        assert torch.equal(got, eps.expand(n_rows, -1)), \
            "a chunk saw an eps that is not the step's single draw"
        assert torch.equal(got[0].expand(n_rows, -1), got), \
            "eps varies BETWEEN PIXELS inside one chunk"
        seen += n_rows
    assert seen == P
    print(f"3. one draw per (member, step): {len(m.calls)} chunks covering "
          f"{seen} pixels all saw the identical eps [k={EPS_DIM}] OK")

    # Chunk-size invariance. NOT bitwise, and the control says why: the SAME
    # comparison on the deterministic path (eps=None, a head with no noise at
    # all) leaves a residual of the same size, because torch dispatches a
    # different GEMM for a different batch height. Bitwise chunk invariance is
    # therefore not a property this evaluator has ever had, with or without
    # E-057 — the property being tested is that the ε path adds NOTHING to it.
    det = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                              k_max=K, stencil=1)
    det.eval()
    with torch.no_grad():
        d_full = RS.roll_step(det, Zwin, None, sctx, mfeat, P, False)
        d_split = RS.roll_step(det, Zwin, None, sctx, mfeat, P // 3, False)
    dev_fgn = (full - split).abs().max().item()
    dev_det = (d_full - d_split).abs().max().item()
    scale = full.abs().max().item()
    assert dev_fgn <= max(1e-5, 20 * dev_det + 1e-7), \
        (f"chunking moved the FGN field by {dev_fgn:.3e} against a "
         f"deterministic control of {dev_det:.3e} — the eps path is "
         f"chunk-dependent")
    print(f"3. chunk P vs P//3: |Δ| {dev_fgn:.2e} (field scale {scale:.3f}); "
          f"deterministic control {dev_det:.2e} — same float-dispatch floor, "
          f"no eps contribution OK")


# ---------------------------------------------------------------------------
# 4. CRPS wiring
# ---------------------------------------------------------------------------
def _ens_from(mem, truth, obs):
    """The evaluator's own masking, in three lines, so the reference below is
    computed from the same arrays the accumulator is fed."""
    obs_n = np.where(obs, truth, np.nan)
    ens_n = np.where(obs[None], mem, np.nan)
    return ens_n, obs_n


def test_crps_wiring():
    rng = np.random.default_rng(4)
    P, C, M = 40, 5, 4
    truth = rng.standard_normal((P, C))
    obs = rng.random((P, C)) > 0.25
    scope = np.zeros(P, bool)
    scope[: P // 2] = True

    # (i) members == truth exactly -> fair CRPS 0, exactly.
    mem = np.repeat(truth[None], M, 0)
    ens_n, obs_n = _ens_from(mem, truth, obs)
    su = RS.new_ens_sums(1)
    cf = probscore.crps_ensemble(ens_n, obs_n)["crps_field"]
    n_ok = int(obs[scope].sum())
    RS.accumulate_ens(su, 1, float(np.nansum(cf[scope])),
                      int(np.isfinite(cf[scope]).sum()),
                      probscore.spread_error(ens_n[:, scope], obs_n[scope]),
                      probscore.ensemble_decomposition(ens_n[:, scope],
                                                       obs_n[scope]), n_ok)
    blk = RS.ens_block(su, 1, M, n_px=int(scope.sum()))
    row = blk["rows"][0]
    assert row["crps"] == 0.0, row
    assert row["spread"] == 0.0 and row["mean_var"] == 0.0, row
    assert row["n"] == n_ok, (row, n_ok)
    print(f"4. members == truth: crps {row['crps']}, spread {row['spread']}, "
          f"mean_var {row['mean_var']} over n={row['n']} cells OK")

    # (ii) M identical (but wrong) members -> the MAE identity, exactly.
    one = rng.standard_normal((P, C))
    mem = np.repeat(one[None], M, 0)
    ens_n, obs_n = _ens_from(mem, truth, obs)
    cf = probscore.crps_ensemble(ens_n, obs_n)["crps_field"]
    mae = float(np.abs(one - truth)[obs].mean())
    got = float(np.nansum(cf[obs]) / np.isfinite(cf[obs]).sum())
    assert abs(got - mae) < 1e-12, (got, mae)
    assert abs(probscore.crps_ensemble(ens_n[:1], obs_n)["crps"] - mae) < 1e-12
    print(f"4. identical members: fair CRPS {got:.9f} == MAE {mae:.9f}, and "
          f"== the M=1 CRPS OK")

    # (iii) the ACCUMULATOR over two starts equals a direct probscore call on
    # the pooled cells — which is the property that makes `ens_prob` a number
    # about the same element population `accumulate`/`skill_block` score.
    su = RS.new_ens_sums(1)
    pool_e, pool_o = [], []
    for start in range(2):
        mem = (truth[None] + 0.6 * rng.standard_normal((M, P, C)))
        ens_n, obs_n = _ens_from(mem, truth + 0.1 * start, obs)
        cf = probscore.crps_ensemble(ens_n, obs_n)["crps_field"]
        sub, ob = ens_n[:, scope], obs_n[scope]
        RS.accumulate_ens(su, 1, float(np.nansum(cf[scope])),
                          int(np.isfinite(cf[scope]).sum()),
                          probscore.spread_error(sub, ob),
                          probscore.ensemble_decomposition(sub, ob),
                          int(obs[scope].sum()))
        pool_e.append(sub.reshape(M, -1)[:, obs[scope].reshape(-1)])
        pool_o.append(ob.reshape(-1)[obs[scope].reshape(-1)])
    blk = RS.ens_block(su, 1, M, n_px=int(scope.sum()))
    row = blk["rows"][0]
    PE = np.concatenate(pool_e, 1)
    PO = np.concatenate(pool_o, 0)
    ref_c = probscore.crps_ensemble(PE, PO)
    ref_s = probscore.spread_error(PE, PO)
    ref_d = probscore.ensemble_decomposition(PE, PO)
    for key, ref in (("crps", ref_c["crps"]), ("spread", ref_s["spread"]),
                     ("rmse", ref_s["rmse"]), ("spread_ratio", ref_s["ratio"]),
                     ("mse_mean", ref_d["mse_mean"]),
                     ("mean_var", ref_d["mean_var"]),
                     ("mse_sample", ref_d["mse_sample"])):
        assert abs(row[key] - round(float(ref), 5)) <= 1e-5, \
            f"{key}: accumulated {row[key]} vs direct probscore {ref}"
    assert abs(blk["crps_mean"] - row["crps"]) <= 1e-9
    assert abs(row["mse_sample"] - (row["mse_mean"] + row["mean_var"])) <= 2e-5
    print(f"4. two starts accumulated == one direct probscore call on the "
          f"pooled {PE.shape[1]} cells, all 7 fields OK "
          f"(crps {row['crps']}, spread/rmse {row['spread_ratio']})")

    # (iv) the SERIES block, through the same functions.
    ens = rng.standard_normal((M, 30)) * 0.5 - 0.2
    obs_s = rng.standard_normal(30)
    thr = -0.8
    got = RS.ens_series_block(ens, obs_s, thr, M)
    assert abs(got["crps"] - round(float(
        probscore.crps_ensemble(ens, obs_s)["crps"]), 4)) < 1e-9
    sp = probscore.spread_error(ens, obs_s)
    assert abs(got["spread_ratio"] - round(float(sp["ratio"]), 4)) < 1e-9
    br = probscore.brier_dip(ens, obs_s, thr, below=True)
    assert got["dip"]["threshold"] == round(thr, 4), got["dip"]
    assert abs(got["dip"]["brier"] - round(float(br["brier"]), 5)) < 1e-9
    assert got["dip"]["n"] == br["n"]
    print(f"4. ens_series_block == probscore directly (crps {got['crps']}, "
          f"dip brier {got['dip']['brier']} at recorded threshold "
          f"{got['dip']['threshold']}) OK")


# ---------------------------------------------------------------------------
# 5 / 6. the refusals and the no-op
# ---------------------------------------------------------------------------
def test_m1_refusal(f, tmp, heads):
    r, txt = run(os.path.join(ML, "rollout_spatial.py"), f,
                 os.path.join(tmp, "never.json"), heads,
                 extra=("--ens-members", "1"), expect_ok=False)
    assert r.returncode != 0, "an FGN head at --ens-members 1 was ACCEPTED"
    assert "--ens-members 1" in txt and "fgn_eps" in txt, txt[-1500:]
    # ARGV TIME: the refusal must land before anything expensive. The codec is
    # opened right after this block, and the Z verify after that.
    assert "Z cache verified" not in txt and "rolled" not in txt, \
        "the M<2 refusal fired AFTER the roll had started paying"
    assert not os.path.exists(os.path.join(tmp, "never.json")), \
        "a refused run still wrote a results file"
    print("5. --ens-members 1 on an FGN head refuses at argv time, before "
          "the codec is opened OK")


def test_flag_is_noop_for_det(f, tmp, det_new):
    got, _ = run(os.path.join(ML, "rollout_spatial.py"), f,
                 os.path.join(tmp, "det_flag.json"), f["heads"],
                 extra=("--ens-members", "5", "--ens-seed", "7"))
    d = first_diff(strip_clocks(got), strip_clocks(det_new))
    assert d is None, f"--ens-members moved a deterministic run at {d}"
    print("6. --ens-members 5 --ens-seed 7 on deterministic heads: results "
          "deep-equal to the run without them OK")


# ---------------------------------------------------------------------------
# 7. the dispersion battery
# ---------------------------------------------------------------------------
def test_dispersion(res, live, zero):
    for lab, n in ((live, "live"), (zero, "zero-film")):
        e = res["heads"][lab]
        for key, want in (("long_dispersion", N_LONG),
                          ("future_dispersion", N_FUT)):
            d = e[key]
            assert d["steps"] == want and len(d["sv_spread"]) == want \
                and len(d["field_var"]) == want and len(d["roll_ym"]) == want, \
                f"{lab}/{key}: {d['steps']} steps, " \
                f"{len(d['sv_spread'])}/{len(d['field_var'])} values"
            assert d["members"] == M_ENS and d["ddof"] == 1
            assert all(np.isfinite(v) and v >= 0 for v in d["sv_spread"])
            assert all(np.isfinite(v) and v >= 0 for v in d["field_var"])
            assert "field_var_floor" in d and d["field_var_floor"] >= 0
        print(f"7. {n} head {lab}: long/future dispersion arrays are "
              f"{N_LONG}/{N_FUT} long, finite and >= 0, floor "
              f"{e['long_dispersion']['field_var_floor']:.2e} OK")

    liv = res["heads"][live]["long_dispersion"]
    zer = res["heads"][zero]["long_dispersion"]
    assert max(liv["sv_spread"]) > 0, \
        "the live FGN head produced NO transport dispersion at any step — " \
        "either eps is not reaching the roll or the members are being reduced"
    assert all(v == 0.0 for v in zer["sv_spread"]), \
        f"the zero-film head dispersed: {zer['sv_spread']}"
    assert all(v <= zer["field_var_floor"] for v in zer["field_var"]), \
        (f"zero-film field_var {zer['field_var']} exceeds its own float32 "
         f"cancellation floor {zer['field_var_floor']}")
    assert max(liv["field_var"]) > 10 * liv["field_var_floor"], \
        (f"the live head's field dispersion {max(liv['field_var']):.3e} is "
         f"inside its own float32 floor {liv['field_var_floor']:.3e} — the "
         f"instrument cannot tell dispersion from cancellation noise here")
    print(f"7. COLLAPSE SIGNATURE: zero-film sv_spread {zer['sv_spread']} "
          f"(exactly 0 at every step) and field_var at/below its floor "
          f"{zer['field_var_floor']:.2e}; the live head reaches "
          f"{max(liv['sv_spread']):.5f} OK")

    # dispersion_block's own arithmetic, against the textbook two-pass form.
    rng = np.random.default_rng(11)
    M, n_steps, npx, nch = 4, 6, 9, 3
    fields = rng.standard_normal((M, n_steps, npx * nch)).astype(np.float32)
    sv = rng.standard_normal((M, n_steps))
    s1 = fields.sum(0).astype(np.float32)
    s2 = (fields.astype(np.float64) ** 2).sum(0).sum(1)
    blk = RS.dispersion_block(sv, s1, s2, npx, nch, M, "long",
                              [f"r{i}" for i in range(n_steps)])
    ref_v = fields.astype(np.float64).var(axis=0, ddof=1).mean(axis=1)
    ref_s = sv.std(axis=0, ddof=1)
    # rtol 1e-5, not tighter: both arrays are written at SIX SIGNIFICANT
    # DIGITS (`%.6g`, so a collapsed head's 1e-8 survives instead of rounding
    # to 0.0), which is itself a relative error of up to 5e-7.
    assert np.allclose(blk["field_var"], ref_v, rtol=1e-5, atol=1e-9), \
        (blk["field_var"], ref_v)
    assert np.allclose(blk["sv_spread"], ref_s, rtol=1e-5, atol=1e-12), \
        (blk["sv_spread"], ref_s)
    rel = float(np.max(np.abs(np.array(blk["field_var"]) - ref_v)
                       / np.maximum(ref_v, 1e-12)))
    print(f"7. dispersion_block's one-pass identity == the two-pass "
          f"variance over {M} members x {n_steps} steps x {npx * nch} cells "
          f"(max rel dev {rel:.2e}) OK")


# ---------------------------------------------------------------------------
def main():
    tmp = tempfile.mkdtemp(prefix="fgnroll_")
    try:
        # `build_fixture` seeds numpy and re-seeds torch per head, but builds
        # its codec off the AMBIENT torch stream — so without this line the
        # toy codec's weights (and therefore every number below) differ run to
        # run, and a tolerance that passed today could fail tomorrow for no
        # reason anyone could reconstruct.
        torch.manual_seed(0)
        f = build_fixture(tmp)
        f["cache"] = os.path.join(tmp, "cache")
        os.makedirs(f["cache"], exist_ok=True)
        live = fgn_head(tmp, "toy_fgn_live", seed=0, zero_film=False)
        zerof = fgn_head(tmp, "toy_fgn_zero", seed=1, zero_film=True)
        fgn_heads = [live, zerof]

        pristine = pristine_evaluator(tmp)
        pmod = None
        if pristine is not None:
            spec = importlib.util.spec_from_file_location(
                "pristine_rollout_spatial", pristine[0])
            pmod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pmod)

        print("--- 1. deterministic-path purity (the acceptance bar) ---")
        test_purity_roll_step(pmod)
        det_new = test_purity_end_to_end(f, tmp, pristine)

        print("--- 2. member determinism ---")
        res, live_lab, zero_lab = test_member_determinism(f, tmp, fgn_heads)

        print("--- 3. shared eps within a member-step ---")
        test_shared_eps()

        print("--- 4. CRPS wiring ---")
        test_crps_wiring()

        print("--- 5. the M=1 refusal ---")
        test_m1_refusal(f, tmp, fgn_heads)

        print("--- 6. --ens-members is a no-op for a deterministic head ---")
        test_flag_is_noop_for_det(f, tmp, det_new)

        print("--- 7. the dispersion battery ---")
        test_dispersion(res, live_lab, zero_lab)

        e = res["heads"][live_lab]
        assert e["meta"]["fgn"]["members"] == M_ENS
        assert e["meta"]["fgn"]["mode"] == "ensemble_mean"
        assert e["meta"]["fgn"]["eps_dim"] == EPS_DIM
        assert e["meta"]["fgn"]["ens_seed"] == 0
        assert "dip" in e["meta"]["fgn"] and "threshold" in e["meta"]["fgn"]["dip"]
        assert set(e["ens_prob"]) - {"note"} == {n for n in e if n in
                                                 ("gate", "corridor", "window",
                                                  "gate_trainlon",
                                                  "gate_holdlon",
                                                  "corridor_trainlon",
                                                  "corridor_holdlon",
                                                  "window_trainlon",
                                                  "window_holdlon")}
        assert "amoc_bands_ens" in e and "amoc_bands_ens_unpooled" in e
        for bn, v in e["amoc_bands_ens"].items():
            assert v["members"] == M_ENS and "crps" in v
            assert v["dip"]["threshold"] == e["meta"]["fgn"]["dip"]["threshold"]
        print(f"+. meta.fgn, ens_prob over "
              f"{len(set(e['ens_prob']) - {'note'})} scopes, amoc_bands_ens"
              f"{'/_unpooled' if 'amoc_bands_ens_unpooled' in e else ''} "
              f"with the recorded dip threshold "
              f"{e['meta']['fgn']['dip']['threshold']} OK")
        print("\ntests/test_fgn_roll.py: all 7 checks green")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
