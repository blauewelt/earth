#!/usr/bin/env python3
"""Mirror artifacts from the GitHub release to the Hugging Face Hub.

Chris, 2026-08-16: *"I created the HuggingFace account, this should get you
started."* — the off-GitHub half of `docs/ARCHIVE.md`.

WHY THIS EXISTS. `model-checkpoints-v1` measured 290 assets / 135.7 GB
against a hard 2 GiB-per-asset cap, which is the only reason a 205M head
ships as a weights-only file PLUS split `.full.partNN` pieces PLUS a
`.sha256` manifest. Hugging Face allows 500 GB per file, so the split
machinery becomes unnecessary the moment the mirror is trusted.

THE RULE THIS SCRIPT ENFORCES, and the reason it is slower than it could be:
**a backup is only real if the restore works.** Every file is uploaded, then
DOWNLOADED BACK, then sha256-compared against the source bytes. An upload
that returns 200 is not evidence the bytes are retrievable (ml/CLAUDE.md
§0.2 — assert the effect, not the invocation), and for a backup the restore
path is the only part that matters. `--no-verify` exists but prints a loud
warning, because the one time it is used will be the time it mattered.

DISK. This sandbox had 9 GB free when the script was written, and the assets
run 0.8–2.5 GB, so files are streamed ONE AT A TIME through a scratch path
and deleted immediately — never a bulk download. That is also why the source
is the GitHub release rather than a local cache: nothing here needs to hold
the collection.

Credentials: `HF_TOKEN` env var, or `/home/claude/.hf_token`. Never argv.
See the project doc `claude/huggingface-access.md`.

Run:
  python3 ml/hf_mirror.py --list
  python3 ml/hf_mirror.py --crown            # the checkpoints worth losing sleep over
  python3 ml/hf_mirror.py --match 'e031xl.*'
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

# NAMESPACE: the HF account is the USER `chfrank`, not an org. `blauewelt` is
# the GitHub org and does not exist on the Hub, so creating there returns
# "403 You don't have the rights to create a model under the namespace".
# Resolved from whoami() at run time rather than hardcoded, so this keeps
# working if Chris later creates a `blauewelt` HF organisation and passes
# --repo.
REPO_NAME = "earth-checkpoints"
REPO_ID = None          # filled from whoami() unless --repo is given
GH_REPO = "blauewelt/earth"
GH_RELEASE_ID = 367193764

# The artifacts that represent GPU time nobody will pay for twice. Order is
# deliberate: the current best models first, so an interrupted run has still
# saved the things that matter most.
CROWN = [
    r"^e031xl_u1_s[01]__temporal\.pt$",          # xl89 200k — project best
    r"^e032xl_u1_s[01]__temporal\.pt$",          # xl144 200k
    r"^e028xl55_u1_s[012]__temporal\.pt$",       # xl55, the roll champion tier
    r"^e029dsun89x200_u1_s[12]__temporal\.pt$",  # sun89 200k
    r"^f3_anchor41M__pixelmae\.pt$",             # the frozen codec everything rides
    r"^e017_u1_s0__temporal\.pt$",               # the validation gate head
]


def gh_assets(token):
    """Every asset on the release, via node fetch (python is proxy-blocked)."""
    js = """
const fs=require('fs');
const pat=fs.readFileSync('/home/claude/.gh_pat','utf8').trim();
(async()=>{
  const out=[];
  for(let p=1;p<=6;p++){
    const r=await fetch('https://api.github.com/repos/%s/releases/%d/assets?per_page=100&page='+p,
      {headers:{Authorization:'Bearer '+pat}});
    const d=await r.json();
    if(!Array.isArray(d)||!d.length) break;
    for(const a of d) out.push({name:a.name,size:a.size,id:a.id});
  }
  console.log(JSON.stringify(out));
})();
""" % (GH_REPO, GH_RELEASE_ID)
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("listing release assets failed: " + r.stderr[-400:])
    import json
    return json.loads(r.stdout)


def gh_download(asset_id, dest):
    js = """
const fs=require('fs');
const {Readable}=require('stream');
const pat=fs.readFileSync('/home/claude/.gh_pat','utf8').trim();
(async()=>{
  const r=await fetch('https://api.github.com/repos/%s/releases/assets/%d',
    {headers:{Authorization:'Bearer '+pat, Accept:'application/octet-stream'},redirect:'follow'});
  if(!r.ok){console.error('HTTP '+r.status);process.exit(1);}
  await require('stream/promises').pipeline(Readable.fromWeb(r.body), fs.createWriteStream('%s'));
})();
""" % (GH_REPO, asset_id, dest)
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("download failed: " + r.stderr[-300:])


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="")
    ap.add_argument("--match", default="")
    ap.add_argument("--crown", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scratch", default="/tmp/hfmirror")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    tok = os.environ.get("HF_TOKEN") or (
        open("/home/claude/.hf_token").read().strip()
        if os.path.exists("/home/claude/.hf_token") else "")
    if not tok:
        sys.exit("no HF_TOKEN (env or /home/claude/.hf_token) — see "
                 "claude/huggingface-access.md")
    os.environ["HF_TOKEN"] = tok
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    if not a.repo:
        who = api.whoami()
        a.repo = f"{who['name']}/{REPO_NAME}"
        print(f"namespace resolved from the token: {who['name']}")

    assets = gh_assets(tok)
    if a.list:
        tot = sum(x["size"] for x in assets)
        print(f"{len(assets)} assets, {tot/1e9:.1f} GB on the GitHub release")
        for x in sorted(assets, key=lambda z: -z["size"])[:15]:
            print(f"  {x['size']/1e6:8.0f} MB  {x['name']}")
        return

    if a.crown:
        pats = [re.compile(p) for p in CROWN]
        want, seen = [], set()
        for p in pats:
            for x in assets:
                if p.match(x["name"]) and x["name"] not in seen:
                    seen.add(x["name"])
                    want.append(x)
    elif a.match:
        p = re.compile(a.match)
        want = [x for x in assets if p.match(x["name"])]
    else:
        sys.exit("give --crown, --match <regex>, or --list")
    if a.limit:
        want = want[:a.limit]
    if not want:
        sys.exit("nothing matched")

    print(f"target  https://huggingface.co/{a.repo}")
    print(f"mirror  {len(want)} file(s), {sum(x['size'] for x in want)/1e9:.1f} GB")
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=False)

    existing = set()
    try:
        existing = set(api.list_repo_files(a.repo, repo_type="model"))
    except Exception:                                # noqa: BLE001
        pass

    os.makedirs(a.scratch, exist_ok=True)
    ok = skipped = failed = 0
    for x in want:
        name, size = x["name"], x["size"]
        if name in existing:
            print(f"  = {name} already on the hub")
            skipped += 1
            continue
        tmp = os.path.join(a.scratch, name)
        try:
            st = os.statvfs(a.scratch)
            free = st.f_bavail * st.f_frsize
            if free < size * 2.2:
                print(f"  ::warning:: skipping {name}: needs "
                      f"{size*2.2/1e9:.1f} GB of scratch (download + verify "
                      f"copy), {free/1e9:.1f} GB free")
                failed += 1
                continue
            print(f"  > {name} ({size/1e6:,.0f} MB) downloading …", flush=True)
            gh_download(x["id"], tmp)
            src_sha = sha256(tmp)
            print(f"    sha256 {src_sha[:16]}… uploading …", flush=True)
            api.upload_file(path_or_fileobj=tmp, path_in_repo=name,
                            repo_id=a.repo, repo_type="model",
                            commit_message=f"mirror {name} from GitHub release")
            if a.no_verify:
                print("    ::warning:: --no-verify: the RESTORE path was not "
                      "exercised, so this is a copy, not yet a backup")
            else:
                # THE point of the script: prove the bytes come back.
                from huggingface_hub import hf_hub_download
                back = hf_hub_download(a.repo, name, repo_type="model",
                                       token=tok,
                                       cache_dir=os.path.join(a.scratch, "dl"))
                got = sha256(back)
                if got != src_sha:
                    raise RuntimeError(
                        f"RESTORE MISMATCH {name}: uploaded {src_sha}, "
                        f"downloaded {got} — the mirror is not trustworthy")
                print(f"    verified: restored sha256 matches ✓")
                shutil.rmtree(os.path.join(a.scratch, "dl"), ignore_errors=True)
            ok += 1
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  ::warning:: {name} FAILED: {type(e).__name__}: "
                  f"{str(e)[:200]}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    print(f"\n{ok} mirrored+verified · {skipped} already present · {failed} failed")
    print(f"https://huggingface.co/{a.repo}/tree/main")


if __name__ == "__main__":
    main()
