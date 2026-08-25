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

  5. **A PUBLISH DOES NOT WAIT FOR THE EMBEDDING TO FINISH.** Chris,
     2026-08-25: *"Publishing should happen 'during' the embedding
     computation, for example, after each 1/100th of the data. A new job that
     needs the same embedding can choose to continue the computation (if
     32/100 are already complete it will start with chunk 33)."* The publish
     unit is the 1.5 GiB release chunk this file has always used (~9% of a
     pentad Z), not a hundredth: a chunk is the smallest thing that can be
     published at all, and chunk-aligned partials need no change to the
     format or to how a complete pull reads it. `push --partial` ships the
     whole chunks that lie inside the flushed prefix and a
     `Z_<w>_<d>.manifest.json` marked `complete: false`; `pull` reads that
     manifest first and, finding one, downloads the prefix, leaves it
     resumable and lets ml/temporal.py embed from the first missing row.
     The ordering rule is the same one everywhere in this file: chunks, THEN
     manifest, so the release can only ever under-claim what it holds.

Usage:
  python3 ml/embed_cache_sync.py pull --run actions --data D --expect-t 3142
  python3 ml/embed_cache_sync.py push --run actions --data D --expect-t 3142
  python3 ml/embed_cache_sync.py push --partial --run actions --data D \\
      --expect-t 3142                       # the finished chunks, mid-embed
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
import time
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


def manifest_name(asset):
    """`Z_<w>_<d>.manifest.json` — the small asset that says how much of the
    cache is on the release, and whether it is all of it.

    It is a SEPARATE asset rather than a field of the chunks because a puller
    must be able to learn "this key is a partial, 6 chunks of 12" for the
    price of one small GET, before it decides how to spend the next twenty
    minutes of bandwidth. It is also the only thing on the release that can
    distinguish a partial publish from a complete one: the chunks themselves
    are byte-identical either way — a partial IS the prefix of the complete
    file — and `pull`'s "concatenate suffixes until one 404s" reads the two
    the same way, which is how it would silently hand back a truncated Z.
    """
    return (asset[:-4] if asset.endswith(".npy") else asset) + ".manifest.json"


def progress_path(path):
    """The in-progress marker ml/temporal.py writes beside a growing cache."""
    return path + ".progress"


def read_progress(path):
    """The `<cache>.progress` dict, or None. Never raises: an unreadable
    marker means "publish nothing yet", not "fail the job"."""
    p = progress_path(path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            mark = json.load(f)
        if not isinstance(mark, dict):
            raise ValueError("not an object")
        for k in ("rows_flushed", "bytes_flushed", "T", "P", "d_z"):
            mark[k] = int(mark[k])
        return mark
    except (OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as e:
        print(f"::warning::unreadable progress marker {os.path.basename(p)}: "
              f"{e} — nothing will be published from it")
        return None


def partial_source(path):
    """The file that HOLDS the flushed bytes for the cache named `path`.

    Two shapes, one function. While ml/temporal.py is embedding, the bytes
    live in `<cache>.partial` and the final name does not exist yet; after a
    `pull --partial` seeded a resume, they live under the final name itself.
    Both are complete .npy files of the FULL (T, P, d_z) shape — `open_memmap`
    and the pull's truncate-to-length both allocate the whole thing up front —
    so the chunk arithmetic below is identical either way and only the path
    differs.
    """
    tmp = path + ".partial"
    return tmp if os.path.exists(tmp) else path


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


def fetch_manifest(base, asset, dest):
    """The published manifest for this key, or None if there is not one.

    One small GET before any bandwidth is committed. `None` covers both "this
    key predates manifests" and "the network said no", and both mean the same
    thing to the caller: fall back to the complete-pull path, which is exactly
    what it did before this file knew about partials.
    """
    url = f"{base}/{manifest_name(asset)}"
    r = sh(f'curl -fsSL --max-time 120 --retry 2 --retry-delay 3 '
           f'-o "{dest}" "{url}"')
    if r.returncode != 0:
        return None
    try:
        with open(dest) as f:
            man = json.load(f)
        return man if isinstance(man, dict) else None
    except (OSError, ValueError) as e:
        # ValueError covers both a bad JSON body and a BINARY one — a proxy
        # error page, or a chunk served under the manifest's name. Neither is
        # a reason to fail a pull; both mean "this key has no manifest I can
        # read", and the complete-pull path below is the honest fallback.
        print(f"::warning::the published manifest for {asset} is unreadable "
              f"({e}) — treating this key as having none")
        return None
    finally:
        if os.path.exists(dest):
            os.remove(dest)


def pull_partial(path, asset, base, expect_t, man):
    """Download the PREFIX a manifest names and leave it resumable.

    The other half of Chris's sentence: *"A new job that needs the same
    embedding can choose to continue the computation (if 32/100 are already
    complete it will start with chunk 33)."* What lands is a full-length .npy
    — the header, the chunks the release actually holds, and zeros to the end
    (`os.truncate`, so the tail costs no bytes on a sparse filesystem) — plus
    a `<cache>.progress` beside it saying how much of it is real. ml/temporal.py
    reads that marker, adopts the file as its own `.partial`, and embeds from
    the first missing row.

    NO `.done` IS WRITTEN, ever, on this path. That marker means "every row is
    real" and is what `push` requires; a partial that could attest to itself
    would republish its own zeros as a complete cache.
    """
    n_total = ((int(man.get("total_bytes_expected", 0)) + CHUNK - 1) // CHUNK)
    chunks = int(man.get("chunks_done", 0))
    # EVERY NUMBER IN THE MANIFEST IS CHECKED AGAINST SOMETHING ELSE. It is a
    # file on a public release; the run that believes it spends sixteen hours
    # on what it says.
    if int(man.get("header_t", -1)) != int(expect_t):
        print(f"::warning::the published partial covers T="
              f"{man.get('header_t')} time bins and the tensor has "
              f"T={int(expect_t)} — a strided or foreign Z, discarded")
        return 1
    if int(man.get("chunk_bytes", 0)) != CHUNK:
        print(f"::warning::the published partial was cut at "
              f"{man.get('chunk_bytes')} bytes per chunk and this build cuts "
              f"at {CHUNK} — refusing to assemble across two chunk sizes")
        return 1
    if not (1 <= chunks < max(n_total, 1)):
        print(f"::warning::the published partial claims {chunks} of "
              f"{n_total} chunk(s) — not a usable prefix, discarded")
        return 1
    tmp = path + ".pull"
    for junk in (tmp, f"{tmp}.part"):
        if os.path.exists(junk):
            os.remove(junk)
    for i in range(chunks):
        suffix = chunk_suffix(i)
        r = sh(f'curl -fsSL --max-time 1800 --retry 3 --retry-delay 5 '
               f'-o "{tmp}.part" "{base}/{asset}.{suffix}"')
        if r.returncode != 0:
            print(f"::warning::the manifest names {chunks} chunk(s) and "
                  f"{asset}.{suffix} is not there — the publish is still in "
                  f"flight or was swept; nothing is assembled from a hole")
            for junk in (tmp, f"{tmp}.part"):
                if os.path.exists(junk):
                    os.remove(junk)
            return 1
        got = os.path.getsize(f"{tmp}.part")
        if got != CHUNK:
            # A SHORT CHUNK IN THE MIDDLE IS THE WHOLE #462 CLASS. Every chunk
            # of a partial is a FULL one by construction (the publisher floors
            # to whole chunks), so a short one is a truncated download or a
            # foreign asset, and gluing it in shifts every row after it.
            print(f"::warning::{asset}.{suffix} is {got:,} bytes, not the "
                  f"{CHUNK:,} a partial chunk must be — discarded")
            for junk in (tmp, f"{tmp}.part"):
                if os.path.exists(junk):
                    os.remove(junk)
            return 1
        with open(tmp, "ab") as out, open(f"{tmp}.part", "rb") as chunk:
            while True:
                b = chunk.read(1 << 24)
                if not b:
                    break
                out.write(b)
        os.remove(f"{tmp}.part")
        print(f"  pulled chunk {suffix} "
              f"({os.path.getsize(tmp) / (1<<30):.2f} GiB)")
    have = os.path.getsize(tmp)
    if have != chunks * CHUNK:
        print(f"::warning::assembled {have:,} bytes from {chunks} chunk(s), "
              f"expected {chunks * CHUNK:,} — discarded")
        os.remove(tmp)
        return 1
    # THE TAIL IS ZEROS AND THE HEADER SAYS SO. The array has to be its full
    # declared length before anything can memmap it, and `.progress` is what
    # keeps those zeros from being mistaken for embeddings.
    os.truncate(tmp, int(man["total_bytes_expected"]))
    os.replace(tmp, path)
    ok, why = verify(path, expect_t)
    if not ok:
        os.remove(path)
        print(f"::warning::the assembled partial failed verification ({why}) "
              f"— discarded; the embedding will be built here")
        return 1
    shape = tuple(int(x) for x in npy_shape(path))
    itemsize = np.dtype(CACHE_DTYPE).itemsize
    header = int(man["total_bytes_expected"]) - int(np.prod(shape)) * itemsize
    row = int(np.prod(shape[1:])) * itemsize
    # ROWS ARE BOUNDED BY THE BYTES THAT ARRIVED, never by the claim. The
    # manifest is data, and the arithmetic that contradicts it wins.
    rows = min(int(man.get("rows_flushed", 0)),
               max(0, (chunks * CHUNK - header) // row), shape[0])
    if rows < 1:
        os.remove(path)
        print("::warning::the published partial covers no complete row — "
              "discarded")
        return 1
    mark = {"rows_flushed": int(rows), "T": int(shape[0]), "P": int(shape[1]),
            "d_z": int(shape[2]),
            "bytes_flushed": int(header + rows * row),
            "dtype": str(np.dtype(CACHE_DTYPE)),
            "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "pull"}
    with open(progress_path(path) + ".part", "w") as f:
        json.dump(mark, f)
    os.replace(progress_path(path) + ".part", progress_path(path))
    print(f"embed cache PARTIAL restored from the release: {chunks}/{n_total} "
          f"chunk(s), {rows}/{shape[0]} row(s) real. The embedding continues "
          f"at row {rows} instead of row 0 — {rows / shape[0] * 100:.0f}% of "
          f"the pass is already done.")
    return 0


def pull(run, a_data, expect_t):
    path, asset, whash = cache_name(run, a_data)
    if os.path.exists(path):
        ok, why = verify(path, expect_t)
        if ok:
            local = read_progress(path)
            if local:
                # A LOCAL PARTIAL BEATS A PUBLISHED ONE and is never
                # overwritten by it: this box is either computing those rows
                # or has already pulled them, and re-pulling could only move
                # the resume point backwards.
                print(f"embed cache already local, valid and PARTIAL "
                      f"({local['rows_flushed']}/{local['T']} row(s) real): "
                      f"{why} — the embedding will continue from it")
                return 0
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
    # THE MANIFEST FIRST, because the chunks cannot tell the two cases apart.
    # A partial publish IS the prefix of the complete file, so the loop below
    # — which concatenates suffixes until one 404s — would assemble a short
    # array with a header claiming the full length, and `verify()` would
    # discard it. That is safe and it throws away the four hours the partial
    # was published to save. One small GET buys the difference.
    man = fetch_manifest(base, asset, path + ".manifest")
    if man is not None and not man.get("complete", False):
        print(f"the release holds a PARTIAL publish of {asset}: "
              f"{man.get('chunks_done')} chunk(s), "
              f"{man.get('rows_flushed')} row(s), last updated "
              f"{man.get('updated_iso')}")
        return pull_partial(path, asset, base, expect_t, man)
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
    # A COMPLETE PULL LEAVES NO `.progress` BEHIND. A stale one from an
    # earlier partial pull of the same key would tell ml/temporal.py to treat
    # a finished cache as a prefix and re-embed most of it.
    for junk in (progress_path(path), progress_path(path) + ".part"):
        if os.path.exists(junk):
            os.remove(junk)
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


def prune_stale_chunks(hdr, api, asset, n, when, rel=None):
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
    if rel is None:
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


def chunk_prefix_on_release(assets, asset, n, total):
    """How many chunks of `asset`, FROM THE FRONT, the release actually holds
    at their right sizes — the only honest source for a manifest.

    Derived from the release rather than from our own upload count on purpose.
    Two boxes may embed the same key at once (the name is the codec's weight
    hash and the tensor's, so they are computing bit-identical bytes), and the
    one that publishes second must not write a manifest that RETRACTS the
    other's chunks. Counting the leading run of present, correctly-sized
    chunks can only move forward, and it is exactly what a puller can use:
    `pull` concatenates from `aa` upward, so a hole makes everything past it
    unreachable however many chunks sit beyond.
    """
    prefix = 0
    for i in range(n):
        a = assets.get(f"{asset}.{chunk_suffix(i)}")
        if a is None or int(a["size"]) != min(CHUNK, total - i * CHUNK):
            break
        prefix += 1
    return prefix


def manifest_for(src, asset, chunks_done, n, total, expect_t):
    """The manifest a puller reads: what is up there, and is it all of it."""
    want, shape, dtype = npy_expected_bytes(src)
    header = want - int(np.prod(shape)) * dtype.itemsize
    row = int(np.prod(shape[1:])) * dtype.itemsize
    complete = chunks_done >= n
    # ROWS THE CHUNKS COVER, not rows the local marker claims. A row that
    # straddles a chunk boundary is not on the release until the NEXT chunk
    # is, so the division floors — the manifest under-claims for the same
    # reason every other marker in this file does (ml/CLAUDE.md §5.21).
    rows = (int(shape[0]) if complete
            else max(0, (chunks_done * CHUNK - header) // row))
    return {"header_t": int(shape[0]), "total_bytes_expected": int(total),
            "chunk_bytes": int(CHUNK), "chunks_done": int(chunks_done),
            "rows_flushed": int(min(rows, int(shape[0]))),
            "complete": bool(complete),
            "expect_t": int(expect_t),
            "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def put_asset(hdr, api, rid, existing, name, part):
    """Replace-or-create one release asset from a local file. True on success.

    `-T`, NEVER `--data-binary @file`. THIS is why the embed cache had never
    once published, through every other fix on 2026-08-10: --data-binary reads
    the entire body into memory before sending, and a 1.5 GiB chunk made curl
    die with "option --data-binary: out of memory" on the first chunk, every
    time, since the day the feature was written. Measured on a 300 MiB file:
    --data-binary @file peaks at 226 MiB RSS, -T at 10 MiB.
    """
    if name in existing:                      # replace, never duplicate
        sh(f'curl -fsSL -X DELETE {hdr} '
           f'"{api}/repos/{REPO}/releases/assets/{existing[name]}"')
    up = sh(f'curl -fsSL -X POST {hdr} '
            f'-H "Content-Type: application/octet-stream" '
            f'-T "{part}" '
            f'"https://uploads.github.com/repos/{REPO}/releases/{rid}/'
            f'assets?name={name}"')
    if up.returncode != 0:
        print(f"::warning::{name} failed: {up.stderr[:200]}")
        return False
    return True


def write_manifest(hdr, api, rid, existing, path, asset, man):
    """Publish the manifest — AFTER the chunks it describes, always.

    Same ordering rule as every marker in this file: the statement about the
    data is written after the data, so it can only ever under-claim. A
    manifest that lands before its chunks tells a puller to fetch bytes that
    are not there yet, and a puller that believes it assembles a short file
    with a header that says otherwise.
    """
    tmp = f"{path}.manifest.up"
    try:
        with open(tmp, "w") as f:
            json.dump(man, f, indent=1)
        ok = put_asset(hdr, api, rid, existing, manifest_name(asset), tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    state = "COMPLETE" if man["complete"] else "partial"
    print(f"  manifest {manifest_name(asset)}: {state}, "
          f"{man['chunks_done']}/{(man['total_bytes_expected'] + CHUNK - 1) // CHUNK}"
          f" chunk(s), {man['rows_flushed']}/{man['header_t']} row(s)"
          f"{'' if ok else ' — UPLOAD FAILED'}")
    return ok


def push(run, a_data, expect_t, partial=False):
    """Publish the cache. `partial` ships the finished PREFIX of one still
    being computed.

    Chris, 2026-08-25: *"Publishing should happen 'during' the embedding
    computation… A new job that needs the same embedding can choose to
    continue the computation (if 32/100 are already complete it will start
    with chunk 33)."* The publish unit is the release chunk the format has
    always used (1.5 GiB, ~9% of a pentad Z) rather than a hundredth: a chunk
    is the smallest thing that can be published at all, and chunk-aligned
    partials need no change to the format, to the naming, or to how a
    complete pull reads it.

    DETERMINISM IS WHAT MAKES THIS SAFE. An embedding is a deterministic
    function of (codec weights, tensor) and the asset name carries a hash of
    both, so two boxes publishing the same key are publishing the same bytes.
    Last-writer-wins per chunk is therefore not a race with a wrong outcome —
    it is two writers agreeing. That is also why a chunk already on the
    release at its right size is SKIPPED rather than re-uploaded.
    """
    path, asset, whash = cache_name(run, a_data)
    # While the embedding runs the bytes are in `<cache>.partial`; after a
    # partial pull they are under the final name. Both are full-length .npy
    # files, so only the path differs.
    src = partial_source(path) if partial else path
    if not os.path.exists(src):
        # The RAM path writes no cache. That is a legitimate outcome, not a
        # failure, and saying so is the difference between "nothing to do" and
        # a silent no-op that looks like success.
        print(f"no embed cache on disk for codec {whash} — nothing to publish "
              f"(the run built Z in RAM because the disk could not hold it)")
        return 0
    # THE SHAPE CHECK COMES BEFORE EVERYTHING IT WOULD COST, and it applies to
    # a partial IDENTICALLY. It is a 128-byte header read, and what it
    # prevents is publishing a Z of the wrong axis under the key every box
    # pulls — the #462 failure, which cost the fleet two weeks of four-hour
    # rebuilds and would have cost a run its results had verify() not refused
    # the chimera on the way back in. A strided Z cannot publish nine percent
    # of itself either.
    ok, why = verify(src, expect_t)
    if not ok:
        print(f"::warning::refusing to publish an invalid cache: {why}")
        return 1
    total = os.path.getsize(src)
    n = (total + CHUNK - 1) // CHUNK
    n_pub, prog = n, None
    if partial:
        # THE PROGRESS MARKER IS THE PARTIAL'S COMPLETENESS MARKER, one level
        # down: `.done` says every row is real, `.progress` says the first
        # `rows_flushed` are. Without it a partial publish would be exactly
        # the thing `check_done` exists to refuse — a full-shape memmap whose
        # tail is zeros — with the zeros shipped to every box in the fleet.
        prog = read_progress(path)
        if prog is None:
            print(f"no {os.path.basename(progress_path(path))} beside the "
                  f"cache — nothing to publish partially yet (a partial "
                  f"publish is only as good as the marker that says which "
                  f"rows are real)")
            return 0
        shape = tuple(int(x) for x in npy_shape(src))
        if (prog["T"], prog["P"], prog["d_z"]) != shape:
            print(f"::warning::refusing a partial publish: the progress marker "
                  f"describes {(prog['T'], prog['P'], prog['d_z'])} and the "
                  f"cache on disk is {shape} — one of them belongs to another "
                  f"run")
            return 1
        flushed = max(0, min(prog["bytes_flushed"], total))
        n_pub = min(flushed // CHUNK, n)
        print(f"  {os.path.basename(src)}: T={expect_t} matches the tensor; "
              f"{prog['rows_flushed']}/{prog['T']} row(s) flushed = "
              f"{flushed / (1<<30):.2f} GiB = {n_pub} whole chunk(s) of {n}")
        if n_pub < 1:
            print(f"embed cache partial: less than one "
                  f"{CHUNK / (1<<30):.2f} GiB chunk is finished — nothing to "
                  f"publish yet; the next window will look again")
            return 0
    else:
        # AND THE COMPLETENESS MARKER, which is the half T cannot see. The
        # cache is a memmap allocated at its full (T, P, d_z) shape before the
        # first month is written, so an abandoned pass leaves a file of exactly
        # the right length, dtype and T with zeros where the last two thousand
        # months should be. Only a mark written after the final flush separates
        # them.
        ok, why = check_done(path)
        if not ok:
            print(f"::warning::refusing to publish an unattested cache: {why}")
            return 1
        print(f"  {os.path.basename(path)}: T={expect_t} matches the tensor, "
              f"{why}")
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
        if partial:
            # A PARTIAL MUST NOT DEMOTE A COMPLETE PUBLISH. The chunks are
            # already all there; rewriting the manifest as complete:false
            # would tell every puller to fetch a prefix of a cache that is
            # entirely on the release, and the four hours it saves are the
            # whole point of the feature.
            print("  (a complete publish stands — the partial push leaves the "
                  "release and its manifest exactly as they are)")
            return 0
        # …EXCEPT THE SWEEP, which is exactly the case that kept #462's
        # chimera alive. The wanted chunks were all present with the right
        # total, so this branch returned 0 and never looked at the six stale
        # ones sitting past the end. "Nothing to do" was true about the
        # upload and false about the release.
        prune_stale_chunks(hdr, api, asset, n, "the no-op (already complete)")
        # The manifest may still say complete:false from the last partial
        # push of this key, so it is rewritten here too: the no-op is about
        # the BYTES, and the manifest is a separate claim about them.
        write_manifest(hdr, api, rid, existing, path, asset,
                       manifest_for(src, asset, n, n, total, expect_t))
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

    print(f"publishing {asset} as {n_pub} of {n} chunk(s), "
          f"{total / (1<<30):.2f} GiB total")
    sent = 0
    with open(src, "rb") as f:
        for i in range(n_pub):
            suffix = chunk_suffix(i)
            name = f"{asset}.{suffix}"
            size = min(CHUNK, total - i * CHUNK)
            # CHEAP IDEMPOTENCE. A chunk already up at its right size is the
            # same bytes as the one we would send (the key is a hash of the
            # codec and the tensor, and the embedding is a deterministic
            # function of both), so re-sending it buys nothing and costs
            # 1.5 GiB — which, ten minutes apart for a whole run, is the
            # difference between a cadence and a flood.
            if name in assets and int(assets[name]["size"]) == size:
                print(f"  {name} already on the release at {size:,} bytes — "
                      f"skipped")
                f.seek((i + 1) * CHUNK)
                continue
            part = f"{path}.{suffix}.up"
            # try/finally, so a part file NEVER outlives the attempt that made
            # it. Previously an ENOSPC while writing raised straight past the
            # os.remove below and left up to 1.5 GiB of garbage on a disk that
            # had just proved it had no room.
            try:
                f.seek(i * CHUNK)
                with open(part, "wb") as o:
                    left = size
                    while left:
                        b = f.read(min(1 << 24, left))
                        if not b:
                            break
                        o.write(b)
                        left -= len(b)
                ok = put_asset(hdr, api, rid, existing, name, part)
            finally:
                if os.path.exists(part):
                    os.remove(part)
            if not ok:
                return 1
            sent += 1
            print(f"  uploaded {name}")
    # THE CHUNKS ARE UP. Only now is it safe to say anything about them — and
    # only a re-listing of the release can say it. The count we THINK we
    # uploaded is an intention (ml/CLAUDE.md §0.1); the assets the API returns
    # are the artefact.
    rel2 = _release(hdr, api)
    if rel2 is None:
        print(f"::warning::uploaded {sent} chunk(s) but could not re-list "
              f"{TAG} to verify them — treating this push as NOT durable so "
              f"the caller retries; the upload itself is idempotent")
        return 1
    assets2 = {a["name"]: a for a in rel2.get("assets", [])}
    prefix = chunk_prefix_on_release(assets2, asset, n, total)
    if partial:
        # NO SWEEP ON A PARTIAL. The chunks past our prefix are not orphans —
        # they are either another writer's progress on the identical bytes, or
        # the tail of a COMPLETE publish of this same key. Deleting those to
        # "tidy up" would replace a finished cache with a fragment.
        if prefix < n_pub:
            print(f"::warning::partial publish: {prefix} chunk(s) verified on "
                  f"the release, {n_pub} expected — the manifest will claim "
                  f"only what is actually there")
        write_manifest(hdr, api, rid, existing, path, asset,
                       manifest_for(src, asset, prefix, n, total, expect_t))
        print(f"embed cache for codec {whash}: {prefix}/{n} chunk(s) durable "
              f"{(prefix * CHUNK) / (1<<30):.2f} GiB — a job that needs this "
              f"embedding can start from chunk {prefix + 1} instead of month 0")
        return 0
    prune_stale_chunks(hdr, api, asset, n, f"uploading {sent} chunk(s)",
                       rel=rel2)
    # DURABLE MEANS THE RELEASE SAYS SO. `prefix == n` is every chunk present,
    # from `aa` upward, each at exactly the size this file would have cut it
    # to — which is also, by construction, `total` bytes in the order a pull
    # concatenates them. Anything less is reported as a failed push so the
    # caller retries, even though the upload may have half worked: a retry is
    # idempotent (the skip above) and a false "durable" is what let #462 sit
    # unnoticed.
    if prefix < n:
        print(f"::warning::the release does not hold all {n} chunk(s) of "
              f"{asset} after the upload ({prefix} verified from the front) — "
              f"NOT durable; the caller should retry")
        return 1
    write_manifest(hdr, api, rid, existing, path, asset,
                   manifest_for(src, asset, n, n, total, expect_t))
    print(f"embed cache for codec {whash} is now durable: VERIFIED {n} "
          f"chunk(s), {total:,} bytes on the release — any box can pull it "
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
    # PARTIAL IS A PUSH MODE, NOT A PULL ONE. A pull cannot be told which of
    # the two it is getting — it has to LOOK, because a partial publish is
    # byte-for-byte the prefix of a complete one and only the manifest tells
    # them apart. So `pull` handles both with no flag, and a caller that has
    # not been taught about partials keeps working unchanged.
    ap.add_argument("--partial", action="store_true",
                    help="push: publish the finished 1.5 GiB chunks of a "
                         "cache that is STILL BEING COMPUTED, and a manifest "
                         "marked complete:false. The shape gate and the "
                         "progress marker both apply; a strided Z cannot "
                         "publish nine percent of itself either.")
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
    if a.partial and a.mode != "push":
        ap.error("--partial is a push mode; pull reads the manifest and takes "
                 "whichever it finds")
    try:
        rc = (pull(a.run, a.data, expect_t) if a.mode == "pull"
              else push(a.run, a.data, expect_t, partial=a.partial))
    except Exception as e:                    # noqa: BLE001
        print(f"::warning::embed cache {a.mode} failed: {type(e).__name__}: {e}")
        rc = 1
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
