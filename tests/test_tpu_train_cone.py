#!/usr/bin/env python3
"""`ml/jaxport/tpu_train_cone.sh` still carries the safety kit, and its knob
block is not a lie.

A TPU launcher is the one file in this repo whose bugs are billed by the hour.
Two classes of them have already been paid for and are what this test pins:

  · **The kit that stops a node outliving its session.** The `__BUCKET__` /
    `__NODE__` / `__TPUZONE__` placeholders (`scripts/tpu_box.py:cmd_create`
    refuses an unbaked file — three nodes once ran the whole launcher blind,
    uploading to a bucket literally named `__BUCKET__`), the boot beacon at
    the banner and its retirement by `kill "${BEACON_PID}"`, `self_delete`,
    `gcs_publish`'s verified two-step, and `gcs_put`'s `curl -T` (a
    `--data-binary @file` buffered a 1.5 GiB chunk in memory and died of it,
    every time, for a day). None of these are cone-specific; all of them are
    what a clone is allowed to get wrong.
  · **The knob block, which IS the experiment** (SPOT_LEDGER 08-28: a
    template rebake silently reverted every knob and spent 62 min running a
    configuration nobody had chosen). Two properties: every knob appears in a
    `resolved ...` echo, because that line is what a launch is read back
    from; and every RUN knob is passed EXPLICITLY on the trainer's own
    invocation, because a knob the trainer defaults silently is a knob the
    resolved-knobs line lies about.

Plus the refusal the cone adds: `TENSOR_SHA` has no default and the file must
refuse the placeholder and the empty string, because the sha256 IS the
tensor's identity (the box effect, ml/CLAUDE.md §7).

No network, no shell execution: this reads the file. `bash -n` is the
syntax check and it is run separately.

    python3 tests/test_tpu_train_cone.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SH = os.path.join(ROOT, "ml", "jaxport", "tpu_train_cone.sh")

# The knob block of ml/plans/E069_HANDOVER.md §8.8, transcribed. Split the way
# the launcher uses them: DATA knobs reach the trainer as the staged file
# path (`--tensor "${TF}"`), RUN knobs are trainer flags and every one of them
# has to be on the invocation.
DATA_KNOBS = ["TENSOR_NAME", "TENSOR_SHA", "TENSOR_PARTS", "GCS_TENSOR",
              "EST_TENSOR_BYTES"]
RUN_KNOBS = ["STEPS", "BATCH", "LR", "SEED", "D_Z", "D_MODEL", "HEADS",
             "N_LATENTS", "LAYERS", "D_DEC", "DEC_LAYERS", "L_IN",
             "FUTURE_LAGS", "AUX_W", "DOT_QUERIES", "HOLDOUT_YEARS",
             "EVAL_EVERY", "CKPT_EVERY", "VELOCITY_PROBE",
             "SNAPSHOT_ABLATION", "TAG", "EXTRA_ARGS"]
KNOBS = DATA_KNOBS + RUN_KNOBS

# The knobs the field launcher carried that the cone must NOT: there is no Z,
# no codec asset and no pixels object, and a knob that does nothing is a knob
# somebody will one day read as if it did.
DROPPED = ["GCS_Z", "GCS_PIXELS", "Z_ASSET", "EST_Z_BYTES", "CODEC_ASSET",
           "FIELD_MODE", "NFE", "MEMBERS", "INPUT_ZNOISE", "COND_CHUNK",
           "D_COND", "COND_LAYERS", "PATCH", "EVAL_WINDOWS"]


def read():
    with open(SH) as f:
        return f.read()


def assign_re(name):
    """`NAME=` as an ASSIGNMENT, not as a mention in prose or a sed recipe.

    The knob block packs several assignments onto one line
    (`STEPS=20000 BATCH=256 …`, the handover's own formatting), so the anchor
    is start-of-line-or-whitespace rather than start-of-line.
    """
    return re.compile(r"(?m)(?:^|[ \t])%s=" % re.escape(name))


def echo_blocks(src, prefix):
    """Every `echo "<prefix>…"` statement, continuations included.

    A shell echo continued with `\\` spans lines; grepping one line at a time
    would find `resolved knobs:` and miss two thirds of the knobs it names.
    """
    out, lines, i = [], src.splitlines(), 0
    while i < len(lines):
        if lines[i].lstrip().startswith('echo "%s' % prefix):
            buf = []
            while i < len(lines):
                buf.append(lines[i])
                if not lines[i].rstrip().endswith("\\"):
                    break
                i += 1
            out.append("\n".join(buf))
        i += 1
    return out


def func_body(src, name):
    """The body of a shell function `name() { … }`, by brace depth."""
    m = re.search(r"(?m)^%s\(\)\s*\{" % re.escape(name), src)
    assert m, "no %s() in %s" % (name, SH)
    depth, i = 0, m.end() - 1
    start = i
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    raise AssertionError("unbalanced braces in %s()" % name)


def main():
    assert os.path.exists(SH), SH
    src = read()

    # ---- 1 · the three placeholders, as ASSIGNMENTS ------------------------
    # tpu_box.py's guard is `^\s*\w+="(__[A-Z]+__)"` — assignments only,
    # because the header legitimately mentions the tokens when documenting the
    # sed. Reproduce it here so the launcher can never drift out of the shape
    # the create command refuses.
    ph = sorted(set(re.findall(r'^\s*\w+="(__[A-Z]+__)"', src, re.MULTILINE)))
    assert ph == ["__BUCKET__", "__NODE__", "__TPUZONE__"], ph
    for tok, var in (("__BUCKET__", "BUCKET"), ("__NODE__", "NODE_NAME"),
                     ("__TPUZONE__", "TPUZONE")):
        assert re.search(r'(?m)^%s="%s"\s*(?:#.*)?$' % (var, tok), src), (var, tok)
    print("1 · placeholders: %s assigned to BUCKET / NODE_NAME / TPUZONE — "
          "tpu_box.py's own regex refuses this file until the bake fills them"
          % ", ".join(ph))

    # ---- 2 · the boot beacon: upload_log BEFORE the shipper loop -----------
    # The beacon is what makes "no object under runs/<node>/ within ~6 min of
    # READY ⇒ zombie" a valid verdict for this launcher (ml/CLAUDE.md §7: a
    # launcher earns the verdict by shipping a beacon). It must fire at the
    # banner — before the 10-minute shipper, which sleeps FIRST and is armed
    # only behind the staging.
    banner = src.index("=== tpu_train_cone ${STAMP}")
    beacon = src.index("upload_log\n( while true; do sleep 180;")
    shipper = src.index("( while true; do sleep $(( SHIP_EVERY_MIN * 60 ))")
    assert banner < beacon < shipper, (banner, beacon, shipper)
    # …and it is RETIRED when the 10-minute shipper takes over the same object
    # name, so two uploaders never race on one log.
    assert "BEACON_PID=$!" in src
    # `src.index` would find the header's prose mention of the retirement; the
    # one that matters is the statement, which is after the shipper.
    retire = src.index('kill "${BEACON_PID}"', shipper)
    assert shipper < retire, (shipper, retire)
    print("2 · beacon: upload_log at the banner, then a 3-min shipper, "
          "retired by kill \"${BEACON_PID}\" once the %s-min shipper arms"
          % "SHIP_EVERY_MIN")

    # ---- 3 · the rest of the safety kit -----------------------------------
    # self_delete asserts the 404 rather than trusting the DELETE (a DELETE
    # issued while CREATE is still settling is silently dropped).
    sd = func_body(src, "self_delete")
    assert '"404"' in sd and "for _i in" in sd, sd[:200]
    assert "trap 'code=$?;" in src and "ship_final; upload_log; self_delete' EXIT" in src
    # gcs_put streams with -T; gcs_publish verifies the size the BUCKET
    # reports before the artefact name exists.
    put = func_body(src, "gcs_put")
    assert "curl -sf -X POST" in put and " -T " in put, put
    pub = func_body(src, "gcs_publish")
    for need in (".uploading", "gcs_size", "gcs_copy", "gcs_delete"):
        assert need in pub, need
    # gcs_size retries TRANSPORT failures — a resolver hiccup once read as
    # "object absent" and reaped a healthy node.
    assert "transport failure" in func_body(src, "gcs_size")
    # the four exits
    for need in ("STALL_MIN", "MAX_HOURS", "VERIFY_MAX_HOURS",
                 "PROGRESS WATCHDOG", "HARD CAP"):
        assert need in src, need
    # the disk guard and its /dev/shm fallback, sized for the anomaly scratch
    assert "mount -o remount,size=170G /dev/shm" in src
    assert re.search(r'(?m)^NEED_GB="\$\{NEED_GB:-120\}"', src)
    assert "36 GB" in src, "the disk-guard comment must name the ~36 GB " \
                           "anomaly scratch copy that sizes NEED_GB"
    print("3 · kit: EXIT trap -> ship_final -> upload_log -> self_delete "
          "(asserts 404) · gcs_put curl -T · gcs_publish via .uploading + "
          "size check · gcs_size transport retry · STALL_MIN / MAX_HOURS / "
          "VERIFY_MAX_HOURS · disk guard 120 GB + /dev/shm remount")

    # ---- 4 · TENSOR_SHA is required and the placeholder is refused ---------
    assert 'TENSOR_SHA="${TENSOR_SHA:-<the Phase-A sha256>}"' in src, \
        "the template must ship the placeholder, not a stale r2 sha"
    ref = re.search(
        r'if \[ -z "\$\{TENSOR_SHA\}" \] \|\| '
        r'\[ "\$\{TENSOR_SHA\}" = "<the Phase-A sha256>" \]; then\n'
        r'(.{0,2000}?)\n\s*exit 1\n\s*fi', src, re.S)
    assert ref, "no refusal branch on an empty/placeholder TENSOR_SHA"
    assert "REFUSING" in ref.group(1)
    # The refusal must live BELOW the EXIT trap, or a refusing node bills
    # until somebody notices — the one failure this whole file prevents.
    assert src.index("trap 'code=$?;") < ref.start(), \
        "the TENSOR_SHA refusal is above the EXIT trap: that node never reaps"
    # …and it is refused before a single byte is staged.
    assert ref.start() < src.index("--- stall arithmetic ---")
    # a 64-hex length check, so a truncated pin cannot be turned into a prefix
    assert '[ "${#TENSOR_SHA}" != "64" ]' in src
    # train mode refuses an unpinned GIT_SHA; verify mode allows it
    assert '[ "${MODE}" = "train" ] && [ -z "${GIT_SHA}" ]' in src
    print("4 · refusals: empty/placeholder TENSOR_SHA, a non-64-hex sha, and "
          "MODE=train with an empty GIT_SHA — all below the EXIT trap and "
          "above the first staged byte")

    # ---- 5 · every knob is echoed -----------------------------------------
    assigned = [k for k in KNOBS if assign_re(k).search(src)]
    assert assigned == KNOBS, sorted(set(KNOBS) - set(assigned))
    echoed = "\n".join(echo_blocks(src, "resolved"))
    assert echoed, "no `echo \"resolved …\"` block at all"
    missing = [k for k in KNOBS if ("${%s}" % k) not in echoed]
    assert not missing, ("knobs assigned but never echoed: %s — the "
                         "`resolved` lines are what a launch is read back "
                         "from" % missing)
    # and the knobs that must NOT be here
    still = [k for k in DROPPED if assign_re(k).search(src)]
    assert not still, ("the cone launcher still assigns field/Z knobs: %s"
                       % still)
    print("5 · resolved echo: all %d knobs of §8.8 named in a `resolved …` "
          "line; none of the %d dropped Z/pixels/diffusion knobs survive"
          % (len(KNOBS), len(DROPPED)))

    # ---- 6 · every RUN knob is on the trainer's invocation -----------------
    inv = func_body(src, "run_trainer")
    assert "ml/jaxport/train_cone.py" in inv
    for flag in ("--tensor", "--steps", "--out", "--metrics metrics.jsonl",
                 "--ckpt-every"):
        assert flag in inv, flag
    assert "${RESUME_ARG}" in inv and "${EXTRA_ARGS}" in inv
    # The three CADENCE knobs are the only ones the verify leg is allowed to
    # override, so they reach run_trainer as its positional arguments; the
    # train-mode call site is where their knob values are read. Everything
    # else is read inside the function, from the knob itself.
    CADENCE = ["STEPS", "EVAL_EVERY", "CKPT_EVERY"]
    train_call = re.search(r'(?m)^run_trainer "\$\{STEPS\}" "\$\{EVAL_EVERY\}" '
                           r'"\$\{CKPT_EVERY\}" "\$\{OUT\}"$', src)
    assert train_call, "no train-mode run_trainer call passing the knobs"
    missing = [k for k in RUN_KNOBS
               if k not in CADENCE and ("${%s}" % k) not in inv]
    assert not missing, ("run knobs not passed to the trainer: %s — a knob "
                         "the trainer defaults silently is a knob the "
                         "resolved-knobs line lies about" % missing)
    for k in CADENCE:
        assert ("${%s}" % k) in train_call.group(0), k
    # the staged tensor is how the DATA knobs reach the trainer
    assert '--tensor "${TF}"' in inv
    assert 'TF="${WORK}/cache/$(basename "${GCS_TENSOR}")"' in src
    # one invocation, used by BOTH legs, so verify and train can differ only
    # in steps / eval cadence / ckpt cadence
    calls = re.findall(r"(?m)^\s*(?:if )?run_trainer ", src)
    assert len(calls) == 2, calls
    print("6 · invocation: one run_trainer() called by both legs, %d run "
          "knobs passed explicitly, --tensor from the staged GCS_TENSOR"
          % len(RUN_KNOBS))

    # ---- 7 · the artefacts it protects ------------------------------------
    for name in ("cone_codec.pt", "ckpt_latest.npz", "velocity_probe.json",
                 "results.json", "metrics.jsonl", "metrics_snapshot.jsonl",
                 "snapshot_codec.pt", "ckpt_latest_snapshot.npz"):
        assert name in src, name
    ship = func_body(src, "ship_state")
    for need in ("${CKPT_NPZ_NAME}", "${CKPT_PT_NAME}", "${PROBE_NAME}",
                 "metrics.jsonl", "results.json", "LAST_CKPT_MARK"):
        assert need in ship, need
    # the resume pulls all four and removes curl's zero-byte 404 leftovers
    resume = src[src.index("--- resume ---"):src.index("9 · the shipper")]
    for need in ("${CKPT_NPZ_NAME}", "${CKPT_PT_NAME}", "metrics.jsonl",
                 "results.json", 'RESUME_ARG="--resume"', "FRESH run"):
        assert need in resume, need
    assert resume.count("rm -f") >= 5, resume.count("rm -f")
    print("7 · artefacts: ship_state ships the ckpt (watchdog marker), the "
          ".pt, metrics, results and the probe; the resume pulls all four "
          "and rm -f's curl's zero-byte 404 leftovers")

    # ---- 8 · verify mode measures the s/step STEP_EST_S stands in for ------
    assert 'VERIFY_STEPS="${VERIFY_STEPS:-300}"' in src
    assert 'VERIFY_MAX_HOURS="${VERIFY_MAX_HOURS:-2}"' in src
    assert 'STEP_EST_S="${STEP_EST_S:-0.5}"' in src
    msps = func_body(src, "measured_s_per_step")
    assert "wall_s" in msps and "measured: %.4f s/step" in msps, msps[:300]
    assert 'measured_s_per_step "${VERIFY_OUT}/metrics.jsonl"' in src
    # the stall arithmetic still refuses, and can still be overridden
    assert "REFUSING: this configuration self-reaps while healthy" in src
    assert '[ "${ALLOW_STALL_RISK}" = "1" ]' in src
    print("8 · verify: %s steps at the real geometry, cap %s h, and it PRINTS "
          "`measured: <s/step>` off the trainer's own wall_s so STEP_EST_S "
          "(%s, a stand-in) stops being one" % (300, 2, 0.5))

    print("\ntpu_train_cone.sh: all 8 checks hold ✓")


if __name__ == "__main__":
    sys.exit(main())
