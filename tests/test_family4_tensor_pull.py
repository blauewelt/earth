#!/usr/bin/env python3
"""The family-4 branch of the Build step: pull the pinned tensor, or say why not.

WHY THE PULL EXISTS ON THIS FAMILY, and it is a sharper reason than family 3's.
`np.savez` stamps zip timestamps, so a REBUILT family-4 tensor never hashes the
same as the published one even from byte-identical inputs. The embed cache is
keyed by (codec weight hash, TENSOR sha256) — `ml/temporal.py` codec_weight_hash
+ data_fingerprint — so the 16.24 GiB pentad Z published as
`Z_8b639abe36_37e146384b` is reachable only by a box holding the tensor whose
sha256 starts 37e146384b. Without this pull a second box:

  * re-spends 8.5 h of a 4090 on the embed it could have downloaded, or
  * on the `sroll:` path, fails outright ("no embed cache published for
    8b639abe36/<its own hash> — run the embed pass first") AFTER building a
    33 GB tensor and extracting a 34 GB X.

So the pull is not a convenience here, it is the thing that makes the published
Z and every cross-box pentad number reachable at all (E-044 §4).

WHAT IS PINNED. The Build step's `run:` body is LIFTED OUT OF THE SHIPPED
WORKFLOW and executed for real (the tests/test_embed_cache_push_marker.py
convention), with `curl`, `sha256sum` and `python` stubbed, so what is tested is
the shell the boxes run and not a paraphrase of it. Four branches:

  1. the tensor is already present AND matches the pin -> no curl at all, no
     Hub fetch, no build;
  2. three chunks pull and the sha verifies -> the file is assembled, and the
     pentad025/truth/SST fetches and build_family4.py are ALL skipped, because
     they are BUILD inputs only and the npz is self-describing;
  3. the pull FAILS -> a ::warning:: that names the consequence (this box will
     build its own bytes and re-spend the embed), then today's path exactly:
     the Hub fetch and build_family4.py run;
  4. the pull succeeds but the sha DISAGREES -> the file is discarded rather
     than used, with a warning, and the build runs.

3 and 4 are the ones worth having a test for: a pull that silently half-worked
would leave a box holding a tensor nobody can name, which is the b40f5b0b /
adcbe700 divergence with an extra step (ml/CLAUDE.md §0.2).

    python3 tests/test_family4_tensor_pull.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WF = os.path.join(ROOT, ".github", "workflows", "ml-train.yml")
TPIN = "37e146384b6f622fefe3c7e18ad9bab0389c9538be79536899fe8729bb2d0826"
TENSOR = "family4_na025_pentad_r2"


def build_step_body(cache_root):
    """The shipped step, with two substitutions and no other edit:

      * the workflow's own `${{ inputs.* }}` expressions bound to env vars,
        which is the substitution GitHub itself performs;
      * `/opt/earth-cache` redirected into the sandbox. That path is the BOX's
        persistent cache and holds tensors, codecs and 16 GiB embeddings; a
        test that writes stub files into it on a developer machine is a test
        that can poison a real cache, so it is rewritten rather than trusted.
    """
    d = yaml.safe_load(open(WF))
    body = next(s["run"] for s in d["jobs"]["train"]["steps"]
                if s.get("name", "").startswith("Build dataset"))
    body = body.replace("${{ inputs.tensor || 'family2' }}", "${IN_TENSOR}")
    body = body.replace("${{ inputs.tensor }}", "${IN_TENSOR}")
    body = body.replace("${{ inputs.sst_channel }}", "${IN_SST}")
    assert "${{" not in body, "an unbound workflow expression survived"
    assert "/opt/earth-cache" in body, \
        "the step no longer names the box cache — re-read this substitution"
    body = body.replace("/opt/earth-cache", os.path.join(cache_root, "cache"))
    return body


def sandbox(curl_rc=0, sha_out=TPIN):
    """A box-shaped temp dir. `curl` writes a chunk, and exits `curl_rc` for
    the data-cache RELEASE only — the Hub fetches must keep working, or case 3
    would be testing the truth-npz refusal instead of the fallback.
    `sha256sum` answers `sha_out`, `python` records argv."""
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "ml", "cache"))
    os.makedirs(os.path.join(tmp, "scripts"))
    bin_ = os.path.join(tmp, "bin")
    os.makedirs(bin_)
    log = os.path.join(tmp, "calls.log")
    with open(os.path.join(bin_, "curl"), "w") as f:
        f.write("#!/bin/sh\n"
                f'echo "curl $*" >> {log}\n'
                'o=""; u=""\nwhile [ $# -gt 0 ]; do case "$1" in '
                '-o) o="$2"; shift 2;; http*) u="$1"; shift;; *) shift;; '
                'esac; done\n'
                'case "$u" in *data-cache-v1*) '
                f'[ {curl_rc} -eq 0 ] || exit {curl_rc};; esac\n'
                '[ -n "$o" ] && echo chunk > "$o"\nexit 0\n')
    with open(os.path.join(bin_, "sha256sum"), "w") as f:
        f.write("#!/bin/sh\n"
                f'echo "sha256sum $*" >> {log}\n'
                f'echo "{sha_out}  $1"\n')
    # BOTH names: the step calls `python -u ml/build_family4.py` and, in
    # seed_sst, a `python3 - <<PY` heredoc that would try to np.load the stub
    # chunk. Stubbing one and not the other would make case 3 fail on the
    # verifier instead of on the branch under test.
    for name in ("python", "python3"):
        with open(os.path.join(bin_, name), "w") as f:
            f.write(f'#!/bin/sh\necho "{name} $*" >> {log}\ncat >/dev/null '
                    f'2>/dev/null || true\nexit 0\n')
    for n in ("curl", "sha256sum", "python", "python3"):
        os.chmod(os.path.join(bin_, n), 0o755)
    return tmp, bin_, log


def run(tmp, bin_, present=False, tensor=TENSOR):
    """The step, rebuilt against THIS sandbox's cache root and run under
    `bash -e` — the same failure semantics the workflow gives it."""
    body = build_step_body(tmp)
    tf = os.path.join(tmp, "ml", "cache", f"{tensor}.npz")
    if present:
        open(tf, "w").write("already here")
    e = dict(os.environ)
    e["PATH"] = bin_ + os.pathsep + e["PATH"]
    e.update({"IN_TENSOR": TENSOR, "IN_SST": "false", "RECIPE_TENSOR": TENSOR,
              "GITHUB_REPOSITORY": "blauewelt/earth", "WINDOW": ""})
    p = os.path.join(tmp, "step.sh")
    open(p, "w").write(body)
    r = subprocess.run(["bash", "-e", p], cwd=tmp, env=e, capture_output=True,
                       text=True, timeout=300)
    calls = open(os.path.join(tmp, "calls.log")).read() \
        if os.path.exists(os.path.join(tmp, "calls.log")) else ""
    return r, calls, tf


def main():
    tmps = [tempfile.mkdtemp()]
    probe = build_step_body(tmps[0])
    assert TPIN in probe, "the pin is not in the shipped workflow"
    # the asset name is assembled from the pin at run time, so the literal
    # request is checked below against what the stub curl was actually asked for
    assert f"{TENSOR}_${{TPIN:0:10}}.npz" in probe, \
        "the asset name the release actually holds is not the one requested"
    try:
        # ---- 1. already present and matching: nothing is fetched ---------
        tmp, bin_, _ = sandbox()
        tmps.append(tmp)
        r, calls, tf = run(tmp, bin_, present=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert "matches pin 37e146384b" in r.stdout, r.stdout[-1500:]
        assert "curl" not in calls, calls
        assert "build_family4.py" not in calls, calls
        assert open(tf).read() == "already here", "an existing pinned tensor was overwritten"
        print("1. a tensor already on the box that matches the pin costs one "
              "sha256sum and nothing else — no curl, no Hub fetch, no build")

        # ---- 2. the pull, and everything it makes unnecessary ------------
        tmp, bin_, _ = sandbox()
        tmps.append(tmp)
        r, calls, tf = run(tmp, bin_)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        got = [c for c in calls.splitlines() if c.startswith("curl")]
        assert len(got) == 3, got
        for sfx in ("aa", "ab", "ac"):
            assert any(f"{TENSOR}_{TPIN[:10]}.npz.{sfx}" in c for c in got), sfx
        assert "data-cache-v1" in got[0], got[0]
        assert "sha verified 37e146384b" in r.stdout, r.stdout[-1500:]
        assert os.path.exists(tf) and open(tf).read() == "chunk\nchunk\nchunk\n"
        assert not any(p.endswith(".part") for p in
                       os.listdir(os.path.join(tmp, "ml", "cache")))
        assert "build_family4.py" not in calls, calls
        assert "huggingface" not in calls, "the Hub was hit for a build that " \
                                           "is not going to happen"
        assert "skipping the pentad025/truth/SST fetch and the build" in r.stdout
        print("2. three chunks pull from data-cache-v1, the sha is verified, "
              "the parts are cleaned up — and the pentad025 means, the truth "
              "npz, the SST artifact and build_family4.py are ALL skipped, "
              "because they are build inputs and there is no build")

        # ---- 3. a FAILED pull falls back, loudly, to today's path --------
        tmp, bin_, _ = sandbox(curl_rc=22)
        tmps.append(tmp)
        r, calls, tf = run(tmp, bin_)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert "::warning::tensor pull failed" in r.stdout, r.stdout[-1500:]
        assert "8.5 h embed" in r.stdout, \
            "the warning must name the CONSEQUENCE, not just the failure"
        assert not os.path.exists(tf), "a failed pull left a partial tensor"
        assert not any(p.endswith(".part") for p in
                       os.listdir(os.path.join(tmp, "ml", "cache")))
        assert "build_family4.py" in calls, calls
        assert "huggingface" in calls, "the fallback must fetch its inputs"
        print("3. a pull that 404s warns with its consequence named (this box "
              "builds its own bytes AND re-spends the 8.5 h embed), leaves no "
              "partial file, and falls through to the Hub fetch + "
              "build_family4.py — today's path exactly")

        # ---- 4. a pull whose sha disagrees is DISCARDED ------------------
        tmp, bin_, _ = sandbox(sha_out="d" * 64)
        tmps.append(tmp)
        r, calls, tf = run(tmp, bin_)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert "!= pin — discarded" in r.stdout, r.stdout[-1500:]
        assert not os.path.exists(tf), \
            "a tensor whose sha does not match the pin was KEPT — this is the " \
            "b40f5b0b/adcbe700 divergence with an extra step"
        assert "build_family4.py" in calls, calls
        print("4. a pulled tensor whose sha256 disagrees with the pin is "
              "discarded, not used: the run builds its own and says so, which "
              "is the one outcome that cannot silently mislabel a box")

        # ---- 5. the family-3 branch is untouched ------------------------
        tmp, bin_, _ = sandbox()
        tmps.append(tmp)
        e = dict(os.environ)
        e["PATH"] = bin_ + os.pathsep + e["PATH"]
        e.update({"IN_TENSOR": "family3_na025", "IN_SST": "false",
                  "RECIPE_TENSOR": "family3_na025", "WINDOW": "",
                  "GITHUB_REPOSITORY": "blauewelt/earth"})
        p = os.path.join(tmp, "step.sh")
        open(p, "w").write(build_step_body(tmp))
        r = subprocess.run(["bash", "-e", p], cwd=tmp, env=e,
                           capture_output=True, text=True, timeout=300)
        calls = open(os.path.join(tmp, "calls.log")).read()
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        assert "adcbe700fb" in calls and "build_family3.py" in calls, calls
        assert "family4" not in calls, calls
        print("5. the family-3 branch still pulls adcbe700fb in two chunks and "
              "still runs build_family3.py — the new branch is beside it, not "
              "in front of it")

        print("\nfamily-4 pinned-tensor pull: all 5 checks hold ✓")
    finally:
        for t in tmps:
            shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    main()
