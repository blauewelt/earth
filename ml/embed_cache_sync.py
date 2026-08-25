#!/usr/bin/env python3
"""Move the frozen-codec embedding cache between a box and a GitHub release.

The embedding is the most expensive derived artefact in the project: ~95
minutes of a 4090 for 43.5M encoder forwards over the quarter-degree tensor,
and every stage-2 run on the same frozen codec needs exactly the same one. It
was also the only expensive artefact we did NOT ship anywhere — checkpoints go
to `model-checkpoints-v1`, tensors to `data-cache-v1`, and Z lived on one
rented box's disk, where the hygiene step deleted it whenever space ran short.
Chris, 2026-08-10: *"getting more disk space to keep training data long term
(or backing it up somewhere) is the better choice."* Vast will not sell us the
disk (it accepts a resize and ignores it), so this is the "somewhere".

Two properties matter more than the transfer itself:

  1. **The asset is named by the CODEC'S WEIGHT HASH**, via the same function
     temporal.py uses. A cache keyed by run name poisoned runs #10/#11 — the
     shape check passed, the embeddings belonged to a different codec, and two
     stage-2 models trained on z their decoder did not speak. Publishing that
     mistake to a release would spread it to every box instead of one.
  2. **A pull VERIFIES before it is trusted**: the reassembled file must parse
     as a .npy, carry the expected dtype, and have the exact byte length the
     header implies. A truncated chunk that still "loads" is the failure this
     guards against — half a cache is not a slow cache, it is a wrong one.
  3. **A push LEAVES NO TAIL BEHIND.** The upload replaces the chunks it
     uploads and, until 2026-08-25, deleted nothing else — so a SHORTER
     re-publish under the same key left the previous publish's higher-suffix
     chunks on the release and the next pull concatenated a chimera. Measured
     on run #462: six fresh chunks of an 8.72 GB strided Z followed by six
     stale chunks of a 16.7 GB full one, header claiming 8,716,963,840 bytes.
     `verify()` refused it — correctly — which is why every run since then
     rebuilt a four-hour embed instead of pulling one. So every push now
     deletes the chunks of the key it did NOT just upload, and does it AFTER
     the upload succeeds, never before: a failed upload must not leave the
     release emptier than it found it.
  4. **A push and a pull both know the SHAPE they expect.** `--expect-t` is
     the tensor's own first dimension, and a Z whose header disagrees is
     refused at the push and discarded at the pull. This is the other half of
     the #462 story: the strided Z was published under the UNSTRIDED name
     because the sidecar loop in scripts/probes_run.sh ships whatever
     /opt/earth-cache/Z_*.npy holds, and temporal.py's "the embed cache is
     DISABLED for this run" governed only temporal.py. A name check cannot
     see a wrong axis; a shape check can. The companion is the completeness
     marker `<cache>.done`, written by temporal.py only after the final flush
     (ml/CLAUDE.md §5.21, flush THEN mark) and required by push — because a
     half-FILLED cache of the right full shape has the right T, and T alone
     would wave it through.

Usage:
  python3 ml/embed_cache_sync.py pull --run actions --data D --expect-t 3142
  python3 ml/embed_cache_sync.py push --run actions --data D --expect-t 3142
  python3 ml/embed_cache_sync.py tensor-t --data D      # what to pass above

`push` needs GITHUB_TOKEN (the job token is enough: `contents: write`).
`pull` needs nothing — the repo is public.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal import (CACHE_DTYPE, codec_weight_hash, data_fingerprint,
                      embed_cache_path)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("GITHUB_REPOSITORY", "blauewelt/earth")
TAG = "embed-cache-v1"
# GitHub caps a release asset at 2 GiB. The fp16 quarter-degree cache is
# 5.2 GiB, so it ships in chunks — the same shape as the data-cache seed,
# which has moved far larger tars this way since #47.
CHUNK = 1500 * 1024 * 1024


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def cache_name(run, data):
    """(local path, asset base name, label) for this run's codec AND tensor.

    Both hashes are in the name. A codec-only key was enough while every box
    built the same tensor; they do not — b40f5b0b against adcbe700, measured
    2026-08-11 — and with a codec-only key a box pulls embeddings of the wrong
    dataset while the shape check, the dtype check and the length check all
    pass. Same failure as #10/#11, with "codec" replaced by "data".
    """
    ck = torch.load(os.path.join(HERE, "runs", run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    whash = codec_weight_hash(ck)
    dhash = data_fingerprint(data)
    return (embed_cache_path(run, whash, dhash),
            f"Z_{whash}_{dhash}.npy", f"{whash}/{dhash}")


def chunk_suffix(i):
    """`aa`, `ab`, … — the split(1) naming the release has used since #47.

    One function, because the pull loop, the push loop, the "already
    published" check and the stale-tail sweep must agree on it; four copies of
    a naming rule are four chances to disagree and the disagreement is a
    chimera nobody sees until verify() refuses one.
    """
    return f"{chr(97 + i // 26)}{chr(97 + i % 26)}"


def stale_chunk_assets(names, asset, n):
    """The chunks of `asset` on the release that a fresh `n`-chunk publish
    ORPHANS — i.e. every `asset.<suffix>` whose suffix is past the end.

    THE BUG THIS IS. Nothing here ever deleted an asset it had not just
    replaced, so publishing a shorter Z under a key that already held a longer
    one left the tail in place. `pull` concatenates suffixes until one 404s,
    so it happily glued six fresh chunks to six stale ones (#462: 8.72 GB of
    strided Z + the tail of a 16.7 GB full Z). The result parses, maps, and is
    wrong — caught only because the assembled length disagreed with the
    header, which is luck rather than design: two publishes whose lengths
    happened to differ by a whole number of chunks would agree.

    Pure, and separated from the API calls, so the arithmetic is testable
    without a network (tests/test_embed_cache_sync.py).
    """
    keep = {chunk_suffix(i) for i in range(n)}
    out = []
    for name in names:
        if not name.startswith(asset + "."):
            continue
        suffix = name[len(asset) + 1:]
        # Only OUR chunks. A two-lowercase-letter tail is the whole naming
        # convention; anything else on the release under this prefix belongs
        # to someone else's scheme and a sweep must not eat it.
        if len(suffix) != 2 or not suffix.isalpha() or not suffix.islower():
            continue
        if suffix not in keep:
            out.append(name)
    return sorted(out)


def done_path(path):
    """The completeness marker beside a finished cache."""
    return path + ".done"


def write_done(path):
    """Mark a cache complete: the byte size, after the bytes are on disk.

    FLUSH, THEN MARK (ml/CLAUDE.md §5.21). The size is in the file rather
    than the marker being empty so that the attestation is about THESE bytes:
    a marker left behind by a different file at the same path fails the
    comparison instead of vouching for a stranger.
    """
    with open(done_path(path), "w") as f:
        f.write(str(os.path.getsize(path)) + "\n")


def check_done(path):
    """(ok, why) for the completeness marker.

    T alone cannot catch a half-filled cache: `open_memmap` allocates the FULL
    (T, P, d_z) shape up front, so a cache abandoned at month 900 of 3142 has
    the right length, the right dtype and the right T, and every other check
    in this file passes it. Only something written after the last write can
    tell the difference, and this is it.
    """
    d = done_path(path)
    if not os.path.exists(d):
        return False, (
            f"no completeness marker {os.path.basename(d)} — this cache was "
            f"not written by a run that finished its embedding pass (or it "
            f"predates the marker). If you have checked the file yourself, "
            f"attest to it: printf '%s\\n' {os.path.getsize(path)} > {d}")
    try:
        with open(d) as f:
            want = int(f.read().strip() or -1)
    except (OSError, ValueError) as e:
        return False, f"unreadable completeness marker {os.path.basename(d)}: {e}"
    have = os.path.getsize(path)
    if want != have:
        return False, (f"completeness marker says {want:,} bytes, the cache is "
                       f"{have:,} — the marker belongs to a different file")
    return True, f"complete ({have:,} bytes, marked)"


def tensor_t(path):
    """The tensor's OWN first dimension, from headers — never a whole load.

    Both of our layouts (ml/tensor_io.py): X inside the single-file `.npz`
    for families 2-4, and X beside it as a bare `_X.npy` for family 5's
    165.6 GB. Either way this reads ~128 bytes; `np.load(...)["X"].shape`
    would decompress 33 GB to learn one integer, and on family 5 the box
    cannot hold it at all.
    """
    x_path = path[:-4] + "_X.npy" if path.endswith(".npz") else path + "_X.npy"
    if os.path.exists(x_path):
        return int(npy_shape(x_path)[0])
    with zipfile.ZipFile(path) as z:
        member = next((m for m in ("X.npy", "X") if m in z.namelist()), None)
        if member is None:
            raise ValueError(f"{os.path.basename(path)} has no X member "
                             f"(has {z.namelist()[:6]})")
        with z.open(member) as f:
            return int(npy_shape(f)[0])


def npy_shape(f):
    """The shape a .npy header declares, from a path OR an open binary file."""
    if isinstance(f, str):
        with open(f, "rb") as fh:
            return npy_shape(fh)
    major, minor = np.lib.format.read_magic(f)
    reader = {(1, 0): np.lib.format.read_array_header_1_0,
              (2, 0): np.lib.format.read_array_header_2_0}[(major, minor)]
    return reader(f)[0]


def npy_expected_bytes(path):
    """Byte length the .npy header implies, so truncation is detectable."""
    # The private _read_array_header was renamed between numpy versions, so go
    # through the public per-version readers. Getting this wrong would make
    # verify() raise on every call, and since the caller treats an exception as
    # "best effort, carry on", the guard would be silently absent — the exact
    # shape of failure this whole file exists to prevent.
    with open(path, "rb") as f:
        major, minor = np.lib.format.read_magic(f)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}[(major, minor)]
        shape, _, dtype = reader(f)
        return f.tell() + int(np.prod(shape)) * dtype.itemsize, shape, dtype


def verify(path, expect_t=None):
    """A cache is trusted only if the file is exactly as long as its own
    header says, carries the dtype we write, and — when the caller knows what
    the axis should be — covers the SAME NUMBER OF TIME BINS as the tensor.

    `expect_t` is optional so the two shell call sites that import this
    function (scripts/sroll_run.sh, ml/jaxport/tpu_train_s2.sh) keep working
    unchanged; where the caller knows the tensor it should pass it, because
    the length check below is blind to a Z that is SELF-CONSISTENTLY the
    wrong shape. #462's strided Z was exactly that: 8.72 GB of real
    embeddings of one bin in two, with a header that agreed with itself.

    Measured rather than assumed: on this numpy a SHORT file raises
    ("mmap length is greater than file size"), so pure truncation would not
    silently return garbage. Two things still justify the check. A file that is
    the wrong length in the other direction — a chunk uploaded twice, or
    reassembled out of order — maps cleanly and returns embeddings that are
    real numbers belonging to the wrong months, which is the failure mode with
    no symptom. And catching it here turns an exception thrown halfway through
    a sixteen-hour job into "discard, rebuild, carry on"."""
    want, shape, dtype = npy_expected_bytes(path)
    have = os.path.getsize(path)
    if have != want:
        return False, (f"{os.path.basename(path)} is {have:,} bytes, header "
                       f"says {want:,} — truncated or corrupt")
    if dtype != np.dtype(CACHE_DTYPE):
        return False, f"dtype {dtype}, expected {np.dtype(CACHE_DTYPE)}"
    if expect_t is not None and int(shape[0]) != int(expect_t):
        return False, (f"{os.path.basename(path)} covers T={int(shape[0])} "
                       f"time bins, the tensor has T={int(expect_t)} — a "
                       f"strided or partial Z must never be published under "
                       f"the unstrided key, nor trained on as if it were one")
    return True, f"{shape} {dtype}, {have / (1 << 30):.2f} GiB"


def pull(run, a_data, expect_t):
    path, asset, whash = cache_name(run, a_data)
    if os.path.exists(path):
        ok, why = verify(path, expect_t)
        if ok:
            print(f"embed cache already local and valid: {why}")
            # NO MARKER IS WRITTEN HERE, deliberately. This branch found a
            # file it did not produce; every check it can run is a check of
            # the header, and the marker's whole job is to say something the
            # header cannot. temporal.py writes it when it finishes the pass
            # (or when it re-enters and the shape is the one it wanted), and
            # the branch below writes it when it downloaded every byte itself.
            return 0
        print(f"local cache rejected ({why}) — removing and re-pulling")
        os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = f"https://github.com/{REPO}/releases/download/{TAG}"
    tmp = path + ".pull"
    if os.path.exists(tmp):
        os.remove(tmp)
    got = 0
    for i in range(64):                       # 64 x 1.5 GiB is far past need
        suffix = f"{chr(97 + i // 26)}{chr(97 + i % 26)}"
        url = f"{base}/{asset}.{suffix}"
        r = sh(f'curl -fsSL --max-time 1800 --retry 3 --retry-delay 5 '
               f'-o "{tmp}.part" "{url}"')
        if r.returncode != 0:
            if i == 0:
                print(f"no embed cache published for codec {whash} — "
                      f"it will be built and, if push runs, uploaded")
                return 1
            break
        with open(tmp, "ab") as out, open(f"{tmp}.part", "rb") as chunk:
            while True:
                b = chunk.read(1 << 24)
                if not b:
                    break
                out.write(b)
        os.remove(f"{tmp}.part")
        got += 1
        print(f"  pulled chunk {suffix} ({os.path.getsize(tmp) / (1<<30):.2f} GiB)")
    os.replace(tmp, path)
    ok, why = verify(path, expect_t)
    if not ok:
        os.remove(path)
        # DISCARDED BEFORE ANY RUN TOUCHES IT. Both halves of this matter and
        # they catch different things: the length check catches a chimera
        # (#462's six fresh chunks glued to six stale ones), and the T check
        # catches a Z that is internally consistent and still the wrong axis.
        # Training on the second is worse than rebuilding, by four hours
        # against a whole run's results.
        print(f"::warning::published embed cache failed verification ({why}) — "
              f"discarded; it will be rebuilt")
        return 1
    write_done(path)
    print(f"embed cache restored from the release in {got} chunk(s): {why}")
    print("this is the ~95 minutes of GPU that stage 2 no longer has to spend")
    return 0


def _release(hdr, api):
    """The release JSON, or None. Called again AFTER an upload, because the
    asset list this sweep works from must be the one the upload left behind."""
    r = sh(f'curl -fsSL {hdr} "{api}/repos/{REPO}/releases/tags/{TAG}"')
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def prune_stale_chunks(hdr, api, asset, n, when):
    """Delete the chunks of `asset` that this publish orphaned. LOUDLY.

    AFTER the upload, never before: a failed upload must not leave the
    release emptier than it found it, and a pull that races this sweep gets
    either the old complete publish or the new one — never a hole in the
    middle of the wanted range, because the wanted range is exactly what is
    NOT swept.

    Best-effort by design. A failed DELETE leaves the tail in place, which is
    the situation we were already in for two weeks; it must not turn a
    successful 8.7 GiB upload into a failed push.
    """
    rel = _release(hdr, api)
    if rel is None:
        print(f"::warning::could not re-list {TAG} after {when} — a stale "
              f"chunk tail may remain under {asset}")
        return
    assets = {a["name"]: a["id"] for a in rel.get("assets", [])}
    stale = stale_chunk_assets(list(assets), asset, n)
    if not stale:
        print(f"stale-chunk sweep after {when}: {asset} has exactly its "
              f"{n} chunk(s) on the release, nothing orphaned")
        return
    print(f"::warning::stale-chunk sweep after {when}: {len(stale)} orphaned "
          f"chunk(s) from a LONGER previous publish of {asset} — "
          f"{', '.join(s.rsplit('.', 1)[1] for s in stale)}. Left in place "
          f"they would be concatenated onto this publish by every pull "
          f"(#462's chimera). Deleting them.")
    for name in stale:
        r = sh(f'curl -fsSL -X DELETE {hdr} '
               f'"{api}/repos/{REPO}/releases/assets/{assets[name]}"')
        print(f"  {'deleted' if r.returncode == 0 else 'FAILED to delete'} "
              f"{name}")


def push(run, a_data, expect_t):
    path, asset, whash = cache_name(run, a_data)
    if not os.path.exists(path):
        # The RAM path writes no cache. That is a legitimate outcome, not a
        # failure, and saying so is the difference between "nothing to do" and
        # a silent no-op that looks like success.
        print(f"no embed cache on disk for codec {whash} — nothing to publish "
              f"(the run built Z in RAM because the disk could not hold it)")
        return 0
    # THE SHAPE CHECK COMES BEFORE EVERYTHING IT WOULD COST. It is a 128-byte
    # header read, and what it prevents is publishing a Z of the wrong axis
    # under the key every box pulls — the #462 failure, which cost the fleet
    # two weeks of four-hour rebuilds and would have cost a run its results
    # had verify() not refused the chimera on the way back in.
    ok, why = verify(path, expect_t)
    if not ok:
        print(f"::warning::refusing to publish an invalid cache: {why}")
        return 1
    # AND THE COMPLETENESS MARKER, which is the half T cannot see. The cache
    # is a memmap allocated at its full (T, P, d_z) shape before the first
    # month is written, so an abandoned pass leaves a file of exactly the
    # right length, dtype and T with zeros where the last two thousand months
    # should be. Only a mark written after the final flush separates them.
    ok, why = check_done(path)
    if not ok:
        print(f"::warning::refusing to publish an unattested cache: {why}")
        return 1
    print(f"  {os.path.basename(path)}: T={expect_t} matches the tensor, {why}")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("::warning::no GITHUB_TOKEN — cannot publish the embed cache")
        return 1

    api = "https://api.github.com"
    hdr = f'-H "Authorization: token {token}" -H "Accept: application/vnd.github+json"'
    r = sh(f'curl -fsSL {hdr} "{api}/repos/{REPO}/releases/tags/{TAG}"')
    if r.returncode != 0:
        r = sh(f'curl -fsSL -X POST {hdr} "{api}/repos/{REPO}/releases" '
               f"-d '{json.dumps({'tag_name': TAG, 'name': 'Embedding caches', 'body': 'Frozen-codec embedding caches (Z), named by codec weight hash. Derived data: deleting these costs GPU time, never correctness.'})}'")
        if r.returncode != 0:
            print(f"::warning::could not create release {TAG}: {r.stderr[:200]}")
            return 1
    rel = json.loads(r.stdout)
    rid = rel["id"]
    assets = {a["name"]: a for a in rel.get("assets", [])}
    existing = {k: v["id"] for k, v in assets.items()}

    total = os.path.getsize(path)
    n = (total + CHUNK - 1) // CHUNK

    # ALREADY PUBLISHED AND COMPLETE? Then do nothing.
    #
    # The upload replaces each chunk by DELETING it and re-POSTing, so a
    # re-publish of an identical cache is not merely 5.2 GiB of wasted
    # bandwidth per run — it opens a window in which the release holds a
    # PARTIAL cache. Watched live on 2026-08-11: the asset count went 4 → 3
    # while #142 replaced a cache byte-identical to the one already there, and
    # every queued arm would have done it again. A puller in that window gets
    # a short file; `verify()` catches it and discards, which turns a wasted
    # upload into someone else's wasted 95 minutes.
    #
    # The check is on the exact chunk names and their total size, because the
    # asset name already carries the codec's weight hash — so matching names
    # plus matching bytes means the same cache for the same codec.
    want = [f"{asset}.{chunk_suffix(i)}" for i in range(n)]
    have_all = all(w in assets for w in want)
    have_bytes = sum(assets[w]["size"] for w in want) if have_all else -1
    if have_all and have_bytes == total:
        print(f"embed cache for codec {whash} is already published and complete "
              f"({n} chunk(s), {total / (1<<30):.2f} GiB) — nothing to do")
        # …EXCEPT THE SWEEP, which is exactly the case that kept #462's
        # chimera alive. The wanted chunks were all present with the right
        # total, so this branch returned 0 and never looked at the six stale
        # ones sitting past the end. "Nothing to do" was true about the
        # upload and false about the release.
        prune_stale_chunks(hdr, api, asset, n, "the no-op (already complete)")
        return 0
    if have_all:
        print(f"republishing: {n} chunk(s) present but {have_bytes:,} bytes "
              f"against {total:,} on disk")

    # DO NOT START A WRITE THE DISK CANNOT HOLD. Chunking to a temporary file
    # needs CHUNK bytes free, and on 2026-08-10 this ran on a box with under
    # 1.5 GiB left: the write failed with ENOSPC, the exception left the part
    # file behind, and the disk went to 50/50. Every subsequent job on that
    # box failed in "Set up job" — before any step, so the hygiene step that
    # would have cleaned up could never run. One unchecked write cost three
    # queued runs and took the box out of the fleet.
    #
    # ml/CLAUDE.md §5.18: size a guard from the allocation it guards. The
    # allocation is CHUNK, and it is knowable here.
    free = shutil.disk_usage(os.path.dirname(path)).free
    need = CHUNK + (256 << 20)
    if free < need:
        print(f"::warning::refusing to publish: chunking needs "
              f"{need / (1<<30):.2f} GiB free and the disk has "
              f"{free / (1<<30):.2f} GiB. Publishing would fill it and take "
              f"the box out of the fleet, which is worse than not publishing.")
        return 1

    print(f"publishing {asset} as {n} chunk(s), {total / (1<<30):.2f} GiB total")
    with open(path, "rb") as f:
        for i in range(n):
            suffix = chunk_suffix(i)
            name = f"{asset}.{suffix}"
            part = f"{path}.{suffix}.up"
            # try/finally, so a part file NEVER outlives the attempt that made
            # it. Previously an ENOSPC while writing raised straight past the
            # os.remove below and left up to 1.5 GiB of garbage on a disk that
            # had just proved it had no room.
            try:
                with open(part, "wb") as o:
                    left = min(CHUNK, total - i * CHUNK)
                    while left:
                        b = f.read(min(1 << 24, left))
                        if not b:
                            break
                        o.write(b)
                        left -= len(b)
                if name in existing:          # replace, never duplicate
                    sh(f'curl -fsSL -X DELETE {hdr} '
                       f'"{api}/repos/{REPO}/releases/assets/{existing[name]}"')
                # `-T`, NEVER `--data-binary @file`. THIS is why the embed
                # cache had never once published, through every other fix
                # tonight: --data-binary reads the entire body into memory
                # before sending, and a 1.5 GiB chunk made curl die with
                # "option --data-binary: out of memory" on the first chunk,
                # every time, since the day the feature was written.
                #
                # Measured rather than assumed, on a 300 MiB file:
                #   --data-binary @file   peak RSS 226 MiB
                #   -T file               peak RSS  10 MiB
                # -T streams from the file and sets Content-Length from its
                # size, which is what the release upload endpoint wants.
                up = sh(f'curl -fsSL -X POST {hdr} '
                        f'-H "Content-Type: application/octet-stream" '
                        f'-T "{part}" '
                        f'"https://uploads.github.com/repos/{REPO}/releases/{rid}/'
                        f'assets?name={name}"')
            finally:
                if os.path.exists(part):
                    os.remove(part)
            if up.returncode != 0:
                print(f"::warning::chunk {name} failed: {up.stderr[:200]}")
                return 1
            print(f"  uploaded {name}")
    # ALL N CHUNKS ARE UP. Only now is it safe to remove what they superseded.
    prune_stale_chunks(hdr, api, asset, n, f"uploading {n} chunk(s)")
    print(f"embed cache for codec {whash} is now durable: any box can pull it "
          f"instead of spending 95 minutes rebuilding it")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pull", "push", "tensor-t"))
    ap.add_argument("--run")
    # The tensor is part of the cache's identity, so it is a required
    # input rather than a default nobody passes.
    ap.add_argument("--data", required=True,
                    help="the tensor .npz these embeddings belong to")
    # REQUIRED, not optional-with-a-default. An optional shape check is one
    # that is absent exactly where it was needed: the sidecar loop that
    # published #462's strided Z would simply not have passed it. A caller
    # that has not been taught the argument now fails loudly at argparse
    # instead of publishing unchecked, and the callers' own `|| echo
    # ::warning::` turns that into a visible line rather than a dead job.
    ap.add_argument("--expect-t", metavar="T",
                    help="time bins the tensor has — a Z whose header "
                         "disagrees is refused (push) or discarded (pull). "
                         "'auto' reads it from --data's own header.")
    a = ap.parse_args()
    if a.mode == "tensor-t":
        # The one-liner the shell callers use to learn T without loading a
        # 33 GB tensor. Prints one integer and nothing else.
        print(tensor_t(a.data))
        sys.exit(0)
    if not a.run:
        ap.error("--run is required for pull and push")
    if not a.expect_t:
        ap.error("--expect-t is required for pull and push (pass the tensor's "
                 "T, or 'auto' to read it from --data)")
    try:
        expect_t = (tensor_t(a.data) if a.expect_t == "auto"
                    else int(a.expect_t))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
        print(f"::warning::embed cache {a.mode}: could not establish the "
              f"tensor's T from --expect-t {a.expect_t!r}: "
              f"{type(e).__name__}: {e}")
        sys.exit(1)
    # THE EXIT CODE MUST MEAN SOMETHING. This read
    #     sys.exit(0 if push(...) == 0 else 0)
    # which is zero either way — so a failed push looked like a successful one,
    # the caller's `&& touch /tmp/embed-cache-pushed` fired, and the cache was
    # never uploaded and never retried. Written, on 2026-08-10, into the very
    # file whose docstring is about steps that report success while doing
    # nothing. Best-effort is the CALLER's decision (`|| true`), never a lie
    # told by the callee.
    try:
        rc = (pull if a.mode == "pull" else push)(a.run, a.data, expect_t)
    except Exception as e:                    # noqa: BLE001
        print(f"::warning::embed cache {a.mode} failed: {type(e).__name__}: {e}")
        rc = 1
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
