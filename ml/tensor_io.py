#!/usr/bin/env python3
"""One loader for our tensors, so a 166 GB one can be opened at all.

WHY THIS EXISTS. Every reader in `ml/` opens a tensor the same way:

    d = np.load(a.data, allow_pickle=False)
    X = d["X"]

and `np.load` on an `.npz` DECOMPRESSES the member into RAM. At family 3's
10.9 GB nobody noticed. At family 4's 33.1 GB it fits a 63 GB box, but only
after today's memory work. At family 5's **165.6 GB it cannot be done on any
box we can rent** — there is no 4090 host with that much RAM at a sane price,
and the number is not close.

So the daily arm is not a disk problem, it is a FORMAT problem, and E-038 §3's
"blocked on renting the right box" was wrong about that. `np.load` grew a
`mmap_mode` for exactly this case and it works only on a bare `.npy`. Hence
the sidecar layout:

    family5_na025_daily.npz      small: months, lats, lons, chan, norm, truths
    family5_na025_daily_X.npy    the tensor itself, uncompressed, memmappable

`load_tensor` returns the same mapping interface either way — `d["X"]`,
`d["months"]`, `k in d`, `d.files` — so a reader changes ONE line and stops
caring which layout it got. Families 2, 3 and 4 keep their single-file `.npz`
and behave exactly as before; there is no migration and no rebuild.

WHY A MEMMAP IS ENOUGH, and where it is NOT. Every consumer of `X` either
walks it in chunks (`obs_any_chunked`) or indexes a batch of pixels out of it
(`LazyPixels`, `gather_px`), so the resident cost becomes the working set and
the page cache absorbs the rest — evictable, unlike a decompressed array.

The exception is `trainprobe.anomaly_transform`, which WRITES into `X` in
place. That is deliberate and it is what keeps the pentad path affordable, but
it cannot be pointed at a read-only map, and pointing it at an `r+` map would
rewrite the canonical tensor — so the next run would z-score already-z-scored
data and nothing would say so. `writable_copy` is the answer: one chunked pass
into a scratch `.npy` the caller owns and deletes. It costs a second copy on
disk (166 GB, hence a >= 400 GB box) and no RAM at all.

    from tensor_io import load_tensor, writable_copy
"""
import os

import numpy as np

SIDECAR = "_X.npy"


def sidecar_path(npz_path):
    """Where `X` lives when it is stored beside the npz rather than inside."""
    stem = npz_path[:-4] if npz_path.endswith(".npz") else npz_path
    return stem + SIDECAR


class _Tensor:
    """A read-only mapping over (small metadata npz, memmapped X)."""

    def __init__(self, meta, X, path, x_path):
        self._meta, self._X = meta, X
        self.path, self.x_path = path, x_path

    def __getitem__(self, k):
        if k == "X":
            return self._X
        return self._meta[k]

    def __contains__(self, k):
        return k == "X" or k in self._meta.files

    @property
    def files(self):
        return ["X"] + [k for k in self._meta.files if k != "X"]

    def get(self, k, default=None):
        return self[k] if k in self else default

    def close(self):
        self._meta.close()


def load_tensor(path, mmap=True, allow_pickle=False):
    """`np.load` for our tensors, memory-mapping X when it is stored beside.

    Returns something that answers `d["X"]`, `d["months"]`, `"rapid" in d` and
    `d.files` — the same four things every reader in ml/ uses — so a call site
    changes from `np.load(a.data, allow_pickle=False)` to `load_tensor(a.data)`
    and nothing else.

    `mmap=False` forces the array into memory, which is what a consumer that
    intends to MUTATE X wants if it is small enough to afford it. For anything
    that must mutate a large one, see `writable_copy`.
    """
    x_path = sidecar_path(path)
    if os.path.exists(x_path):
        X = np.load(x_path, mmap_mode="r" if mmap else None)
        meta = np.load(path, allow_pickle=allow_pickle)
        return _Tensor(meta, X, path, x_path)
    return np.load(path, allow_pickle=allow_pickle)


def writable_copy(X, scratch, chunk=64, verbose=True):
    """A WRITABLE `.npy` copy of X, one chunk at a time. Returns the memmap.

    For `anomaly_transform`, which writes into its input. Copying rather than
    opening the canonical tensor `r+` is not caution, it is correctness: an
    in-place transform on the stored file would leave an anomaly-space tensor
    where a state-space one is documented, and the NEXT run would z-score it
    again and train happily on the result.

    Chunked, so peak RSS is one chunk regardless of size: 337 MB at C=39,
    H=281, W=481 whether the tensor is 33 GB or 166. The caller owns `scratch`
    and should delete it; on a box that is the whole point of `--keep-memmap`
    style hygiene.
    """
    out = np.lib.format.open_memmap(scratch, mode="w+", dtype=X.dtype,
                                    shape=X.shape)
    n = X.shape[0]
    for i in range(0, n, chunk):
        out[i:i + chunk] = X[i:i + chunk]
        if verbose and (i // chunk) % 20 == 0:
            print(f"  writable_copy {min(i + chunk, n)}/{n}", flush=True)
    out.flush()
    return out


def save_tensor(path, X, **meta):
    """Write the sidecar layout: X as a bare .npy, everything else in the npz.

    `np.savez_compressed` is right for a tensor a reader can hold in memory
    and wrong for one it cannot — compression is exactly what makes a file
    unmappable. The saving it buys is small here anyway: these tensors are
    dense float16 anomalies, not sparse integers.

    X may be a memmap the builder has been filling, in which case this only
    renames it — no second 166 GB copy, which is the difference between a
    build that fits a 400 GB box and one that does not.
    """
    x_path = sidecar_path(path)
    if isinstance(X, np.memmap) and os.path.exists(X.filename or ""):
        X.flush()
        src = X.filename
        del X
        if os.path.abspath(src) != os.path.abspath(x_path):
            os.replace(src, x_path)
    else:
        np.save(x_path, X)
    np.savez(path, **meta)
    return x_path
