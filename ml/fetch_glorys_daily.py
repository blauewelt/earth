#!/usr/bin/env python3
"""E-034: GLORYS12 DAILY base channels for the North Atlantic window.

Chris, 2026-08-16: *"Go ahead with the GLORYS12 daily fetcher, and more
generally, with preparing all data both daily and 5-day."*

THE ONE DESIGN DECISION THIS FILE ENCODES, because everything else follows
from it: **fetch DAILY, derive pentad by aggregation — never two pipelines.**
A 5-day mean is a pure reduction of the daily series, so building the two
cadences independently would be two implementations of one rule, and the two
would eventually disagree about a leap day or a bin edge. That is the same
defect class as the status page re-deriving the LR schedule (which drew a
cosine for a run that used expdecay) and the corridor being traced by hand in
the frontend. So: this script writes DAILY files, and `--cadence pentad`
aggregates the same downloaded bytes with `ml/aggregate_cadence.py`. The
daily files are the single source of truth for both tensors.

Pentads are the fixed 5-day bins of `ml/build_truth_pentad.py` — counted from
1982-01-01, index = floor(days_since_epoch / 5) — so the state axis and the
label axis land in the same bins by construction rather than by coincidence.

WHAT IS FETCHED. `cmems_mod_glo_phy_my_0.083deg_P1D-m` (GLORYS12 reanalysis,
daily, 1/12 degree), verified reachable 2026-08-16. Variables `uo`, `vo`
(surface current -> speed), `mlotst` (mixed-layer depth), `zos` (sea surface
height), depth slice 0-1 m only. Subsetted server-side to the NA window
(100 W..20 E, 0..70 N) — 1/12 degree globally is 16x the pixels of the 1/4
degree ensemble the monthly bake used, and we bin down to 0.25 degrees
anyway, so subsetting is what makes this affordable at all.

RESUME. One file per (year, month) chunk, skipped when already present and
non-empty. A month of the subsetted window is ~30 slices; the loop can be
killed and restarted without losing work, which matters because the full
1993-2024 pull is hours of transfer.

DISK. The caller must have room: the daily NA tensor is ~165 GB in fp16 and
does NOT fit a 100 GB Vast box or this sandbox (measured 8.7 GB free on
2026-08-16). Run this where the disk is — see ml/plans/E034_pentad_tensor.md
section 5. `--dry-run` reports the plan and the byte estimate without
fetching, and `--months N` limits the pull for a smoke test.

Credentials: env vars COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD, from the
project doc claude/copernicus-marine-access.md. Never written to disk here.

Run:
  python3 ml/fetch_glorys_daily.py --dry-run
  python3 ml/fetch_glorys_daily.py --start 1993-01 --end 2024-12 --out /data/glorys_daily
"""
import argparse
import calendar
import datetime as dt
import os
import shutil
import sys

DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
# The interim stream covers the most recent months under a different id; it is
# resolved from the catalogue at run time rather than hardcoded, because a
# guessed id is how the first credential check earned a DatasetNotFound.
INTERIM_HINT = "myint"
VARIABLES = ["uo", "vo", "mlotst", "zos"]

# The family-3 window, verbatim. Any drift here silently makes a tensor that
# cannot be compared with anything already measured.
WINDOW = dict(minimum_longitude=-100.0, maximum_longitude=20.0,
              minimum_latitude=0.0, maximum_latitude=70.0)

EPOCH = dt.date(1982, 1, 1)
PENTAD_DAYS = 5


def months_between(start, end):
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def sha256(path, buf=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def hf_connect(repo_name):
    """Resolve the namespace and ensure the dataset repo exists."""
    import os as _os
    tok = _os.environ.get("HF_TOKEN") or (
        open("/home/claude/.hf_token").read().strip()
        if _os.path.exists("/home/claude/.hf_token") else "")
    if not tok:
        sys.exit("no HF_TOKEN (env or ~/.hf_token) and backup is enabled — "
                 "see claude/huggingface-access.md, or pass --no-backup and "
                 "accept that a dead box costs the whole pull")
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    repo = repo_name if "/" in repo_name else f"{api.whoami()['name']}/{repo_name}"
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    return api, repo, tok


def hf_preflight(api, repo, tok, scratch):
    """PROVE THE BACKUP WORKS BEFORE FETCHING ANYTHING.

    Chris, 2026-08-16: *"I would build the HF backup into that script (test
    it in the beginning). Otherwise we will have built the training data and
    then lost it."* Exactly right, and the repo already has the scar: the
    embed-cache push sat AFTER `wait $S2_PID`, so a cache that existed for
    sixteen hours was published at the very end — and when the upload failed
    for lack of room, the run reported success (ml/CLAUDE.md §5.20, §4.6).

    So this uploads a few bytes, downloads them BACK, compares, and deletes —
    the entire round trip, exercised while it has cost nothing. If the token,
    the namespace, the quota or the network is wrong, the job dies here
    instead of after 110 GB."""
    import os as _os
    from huggingface_hub import hf_hub_download
    probe = _os.path.join(scratch, ".preflight")
    payload = b"earth/E-034 backup preflight\n"
    with open(probe, "wb") as f:
        f.write(payload)
    want = sha256(probe)
    api.upload_file(path_or_fileobj=probe, path_in_repo=".preflight",
                    repo_id=repo, repo_type="dataset",
                    commit_message="backup preflight")
    back = hf_hub_download(repo, ".preflight", repo_type="dataset", token=tok,
                           cache_dir=_os.path.join(scratch, "pf"))
    got = sha256(back)
    _os.remove(probe)
    shutil.rmtree(_os.path.join(scratch, "pf"), ignore_errors=True)
    if got != want:
        sys.exit(f"BACKUP PREFLIGHT FAILED: uploaded {want}, got back {got}. "
                 f"Refusing to fetch — the point of the preflight is that we "
                 f"find this out now and not after 110 GB.")
    print("backup   preflight OK: uploaded, downloaded back, sha256 matched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1993-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache", "glorys_daily"))
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--months", type=int, default=0,
                    help="stop after N month-chunks (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="refuse to start below this much free disk")
    ap.add_argument("--hf-repo", default="earth-tensors",
                    help="Hugging Face dataset repo for the running backup; "
                         "namespace resolved from the token. '' disables, "
                         "which you should not do — see --no-backup.")
    ap.add_argument("--no-backup", action="store_true",
                    help="fetch without mirroring. Prints a loud warning: a "
                         "110 GB pull with no backup is one dead box away "
                         "from being done twice.")
    ap.add_argument("--keep-local", action="store_true",
                    help="keep chunks on local disk after they are backed up "
                         "(default: delete, so the pull needs ~1 chunk of "
                         "disk rather than 110 GB)")
    a = ap.parse_args()

    chunks = list(months_between(a.start, a.end))
    if a.months:
        chunks = chunks[:a.months]
    ndays = sum(calendar.monthrange(y, m)[1] for y, m in chunks)

    # Size the job against the disk BEFORE spending anything on it: a guard
    # that depends only on the inputs must fire while the inputs are all it
    # has cost (ml/CLAUDE.md section 0.3). The window is 1440x840 cells at
    # 1/12 degree; 4 variables, float32, one depth level.
    cells = int(120 / (1 / 12)) * int(70 / (1 / 12))
    per_day = cells * len(VARIABLES) * 4
    est_gb = per_day * ndays / 1e9
    # MEASURED, not modelled: one real 2015-01 chunk came back at 287 MB for
    # 31 days of the subsetted window (smoke test, 2026-08-16), i.e. NetCDF
    # compression buys ~2x over the raw arithmetic below. Both numbers are
    # printed because the raw one bounds peak decompressed memory and the
    # measured one bounds the disk.
    meas_gb = 0.287 * len(chunks)
    print(f"dataset   {a.dataset}")
    print(f"window    {WINDOW['minimum_longitude']}..{WINDOW['maximum_longitude']} E, "
          f"{WINDOW['minimum_latitude']}..{WINDOW['maximum_latitude']} N "
          f"({cells:,} cells at 1/12 deg)")
    print(f"span      {a.start}..{a.end} = {len(chunks)} month-chunks, {ndays:,} days")
    print(f"variables {', '.join(VARIABLES)} (surface, depth 0-1 m)")
    print(f"estimate  ~{est_gb:.1f} GB uncompressed · "
          f"~{meas_gb:.0f} GB on disk (measured 287 MB/month-chunk)")

    os.makedirs(a.out, exist_ok=True)
    st = os.statvfs(a.out)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    print(f"disk      {free_gb:.1f} GB free at {a.out}")
    if a.dry_run:
        have = sum(1 for y, m in chunks
                   if os.path.exists(os.path.join(a.out, f"glorys_{y}{m:02d}.nc")))
        print(f"resume    {have}/{len(chunks)} chunks already present")
        print("\n--dry-run: nothing fetched.")
        if free_gb < meas_gb:
            print(f"NOTE: {meas_gb:.0f} GB needed vs {free_gb:.1f} GB free — this "
                  f"must run where the disk is (E-034 section 5).")
        return
    if free_gb < a.min_free_gb:
        sys.exit(f"refusing to start: {free_gb:.1f} GB free < --min-free-gb "
                 f"{a.min_free_gb}. Fetching into a full disk is how a box "
                 f"goes metrics-blind (ml/CLAUDE.md section 7).")

    if not os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"):
        sys.exit("set COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD "
                 "(claude/copernicus-marine-access.md)")
    try:
        import copernicusmarine as cm
    except ImportError:
        sys.exit("pip install copernicusmarine")

    api = repo = tok = None
    hf_have = set()
    if a.no_backup:
        print("::warning:: --no-backup: a 110 GB pull with no mirror is one "
              "dead box away from being done twice")
    else:
        api, repo, tok = hf_connect(a.hf_repo)
        print(f"backup   https://huggingface.co/datasets/{repo}")
        hf_preflight(api, repo, tok, a.out)
        try:
            hf_have = {f for f in api.list_repo_files(repo, repo_type="dataset")}
        except Exception:                            # noqa: BLE001
            pass
        if hf_have:
            print(f"resume   {len(hf_have)} file(s) already backed up — "
                  f"those chunks are skipped")

    done = fail = skip = 0
    for y, m in chunks:
        fname = f"glorys_{y}{m:02d}.nc"
        out = os.path.join(a.out, fname)
        # HF is the resume source, not the local disk: a box that dies takes
        # its disk with it, and the whole point of backing up per chunk is
        # that the next box starts where this one stopped.
        if fname in hf_have:
            skip += 1
            continue
        if os.path.exists(out) and os.path.getsize(out) > 0 and a.keep_local:
            skip += 1
            continue
        last = calendar.monthrange(y, m)[1]
        try:
            cm.subset(dataset_id=a.dataset, variables=VARIABLES,
                      start_datetime=f"{y}-{m:02d}-01",
                      end_datetime=f"{y}-{m:02d}-{last:02d}",
                      minimum_depth=0, maximum_depth=1,
                      output_filename=out, **WINDOW)
            sz = os.path.getsize(out) / 1e6
            if api is not None:
                # PUBLISH WHEN THE ARTEFACT EXISTS, not when the job ends
                # (ml/CLAUDE.md §5.20) — and verify the restore, because an
                # upload returning 200 is not evidence the bytes come back.
                src = sha256(out)
                api.upload_file(path_or_fileobj=out, path_in_repo=fname,
                                repo_id=repo, repo_type="dataset",
                                commit_message=f"glorys daily {y}-{m:02d}")
                from huggingface_hub import hf_hub_download
                back = hf_hub_download(repo, fname, repo_type="dataset",
                                       token=tok,
                                       cache_dir=os.path.join(a.out, "vf"))
                if sha256(back) != src:
                    shutil.rmtree(os.path.join(a.out, "vf"), ignore_errors=True)
                    raise RuntimeError(f"{fname} restored with a DIFFERENT "
                                       f"sha256 — backup not trustworthy")
                shutil.rmtree(os.path.join(a.out, "vf"), ignore_errors=True)
                if not a.keep_local:
                    os.remove(out)
            done += 1
            print(f"  {y}-{m:02d}: {sz:,.0f} MB"
                  + ("" if api is None else " · backed up + restore-verified")
                  + f" ({done} fetched, {skip} skipped)", flush=True)
        except Exception as e:                       # noqa: BLE001
            # Say WHY it gave up — best effort is a promise about delivery,
            # never about reporting (ml/CLAUDE.md section 4.6).
            if os.path.exists(out):
                os.remove(out)
            fail += 1
            print(f"  ::warning:: {y}-{m:02d} FAILED, chunk removed: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            if fail >= 5 and done == 0:
                sys.exit("five consecutive failures with nothing fetched — "
                         "stopping rather than hammering the service")
    print(f"\n{done} fetched · {skip} already present · {fail} failed")
    print(f"next: python3 ml/aggregate_cadence.py --in {a.out} "
          f"--cadence pentad   (daily stays the source of truth)")


if __name__ == "__main__":
    main()
