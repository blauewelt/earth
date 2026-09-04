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

MULTI-GROUP TENSORS (E-070 §3, family 7). Family 7 is not one dense array: it
is THREE arrays at three native resolutions — `g025` (0.25 degree, 7 ocean and
surface channels), `g100` (1 degree, 14 NCEP reanalysis channels) and `rg100`
(1 degree, 32 Argo depth channels on the live bins only) — because a 1.9-degree
reanalysis upsampled to 0.25 degrees is sixty copies of every number. Each
group is its own memmappable sidecar:

    family7_global025_pentad_l0.npz            meta, statics, norms, truth
    family7_global025_pentad_l0_X_g025.npy     [3142, 721, 1440,  7] float16
    family7_global025_pentad_l0_X_g100.npy     [3142,  181,  360, 14] float16
    family7_global025_pentad_l0_X_rg100.npy    [n_live, 181, 360, 32] float16

`load_tensor` answers `d["X_g025"]`, `d["X_g100"]`, `d["X_rg100"]`, and keeps
`d["X"]` as an alias of the FIRST group (the dense one, `g025`) so a family-4
consumer that only wants the surface channels changes nothing. The
single-group `_X.npy` form is untouched and takes precedence, so no existing
tensor moves.
"""
import glob as _glob
import os

import numpy as np

SIDECAR = "_X.npy"
GROUP_PREFIX = "_X_"


def sidecar_path(npz_path):
    """Where `X` lives when it is stored beside the npz rather than inside."""
    return stem_of(npz_path) + SIDECAR


def stem_of(npz_path):
    return npz_path[:-4] if npz_path.endswith(".npz") else npz_path


def group_path(npz_path, group):
    """Where group `g` lives in the multi-group sidecar layout."""
    return f"{stem_of(npz_path)}{GROUP_PREFIX}{group}.npy"


def group_names(npz_path, meta=None):
    """The groups stored beside `npz_path`, in the order a reader should see.

    The npz's own `groups` key is authoritative when it is there — the builder
    writes it, and it is what fixes which group `d["X"]` aliases. Without it
    the files themselves are enumerated, sorted, so a hand-assembled stem
    still loads rather than looking empty.
    """
    on_disk = {}
    pat = f"{stem_of(npz_path)}{GROUP_PREFIX}*.npy"
    for p in _glob.glob(pat):
        name = os.path.basename(p)
        head = os.path.basename(stem_of(npz_path)) + GROUP_PREFIX
        on_disk[name[len(head):-4]] = p
    if not on_disk:
        return []
    declared = []
    if meta is not None and "groups" in getattr(meta, "files", ()):
        declared = [str(g) for g in np.atleast_1d(meta["groups"])]
    ordered = [g for g in declared if g in on_disk]
    ordered += sorted(g for g in on_disk if g not in ordered)
    return ordered


class _Tensor:
    """A read-only mapping over (small metadata npz, memmapped group arrays).

    `groups` is an ordered {name: array} dict. The single-group layout passes
    `{"X": array}` and behaves exactly as it always did; a multi-group tensor
    passes {"g025": ..., "g100": ...} and additionally answers `X_<group>`,
    with `X` aliasing the first entry.
    """

    def __init__(self, meta, groups, path, single):
        self._meta = meta
        self._groups = dict(groups)
        self._single = single
        self.path = path
        self.groups = list(groups)
        self.x_paths = {}

    # -- the keys the group arrays answer to -------------------------------
    def _group_key(self, k):
        if self._single:
            return "X" if k == "X" else None
        if k == "X":
            return self.groups[0]           # the dense group, by declaration
        if k.startswith("X_") and k[2:] in self._groups:
            return k[2:]
        return None

    def __getitem__(self, k):
        g = self._group_key(k)
        if g is not None:
            return self._groups[g]
        return self._meta[k]

    def __contains__(self, k):
        return self._group_key(k) is not None or k in self._meta.files

    @property
    def files(self):
        if self._single:
            keys = ["X"]
        else:
            keys = ["X"] + [f"X_{g}" for g in self.groups]
        return keys + [k for k in self._meta.files
                       if k != "X" and k not in keys]

    def get(self, k, default=None):
        return self[k] if k in self else default

    def close(self):
        self._meta.close()

    # Back-compat: the single-group form exposed `.x_path`.
    @property
    def x_path(self):
        return self.x_paths.get("X") or next(iter(self.x_paths.values()), None)


def load_tensor(path, mmap=True, allow_pickle=False):
    """`np.load` for our tensors, memory-mapping X when it is stored beside.

    Returns something that answers `d["X"]`, `d["months"]`, `"rapid" in d` and
    `d.files` — the same four things every reader in ml/ uses — so a call site
    changes from `np.load(a.data, allow_pickle=False)` to `load_tensor(a.data)`
    and nothing else.

    A MULTI-GROUP tensor (family 7) additionally answers `d["X_g025"]`,
    `d["X_g100"]`, `d["X_rg100"]`; `d["X"]` is the first declared group.

    `mmap=False` forces the array into memory, which is what a consumer that
    intends to MUTATE X wants if it is small enough to afford it. For anything
    that must mutate a large one, see `writable_copy`.
    """
    mode = "r" if mmap else None
    x_path = sidecar_path(path)
    if os.path.exists(x_path):
        X = np.load(x_path, mmap_mode=mode)
        meta = np.load(path, allow_pickle=allow_pickle)
        t = _Tensor(meta, {"X": X}, path, single=True)
        t.x_paths = {"X": x_path}
        return t
    meta_probe = np.load(path, allow_pickle=allow_pickle) \
        if os.path.exists(path) else None
    names = group_names(path, meta_probe)
    if names and meta_probe is None:
        raise FileNotFoundError(
            f"{path} is missing but its group sidecars are there "
            f"({', '.join(names)}) — the metadata npz is where the axes, the "
            f"norms and the group ORDER live, and `X` cannot be resolved "
            f"without it")
    if names:
        arrs, paths = {}, {}
        for g in names:
            p = group_path(path, g)
            arrs[g] = np.load(p, mmap_mode=mode)
            paths[g] = p
        t = _Tensor(meta_probe, arrs, path, single=False)
        t.x_paths = paths
        return t
    if meta_probe is not None:
        meta_probe.close()
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

    MULTI-GROUP (family 7): pass a dict {group: array} and each group is
    renamed (or written) to `<stem>_X_<group>.npy` by the same rule, and the
    group ORDER is recorded as `groups` in the npz unless the caller passed
    its own — that key is what decides which group `d["X"]` aliases.
    Returns the sidecar path, or {group: path} for the dict form.
    """
    if isinstance(X, dict):
        out = {}
        for g, arr in X.items():
            out[g] = _write_sidecar(group_path(path, g), arr)
        meta.setdefault("groups", np.array(list(X)))
        np.savez(path, **meta)
        return out
    x_path = _write_sidecar(sidecar_path(path), X)
    np.savez(path, **meta)
    return x_path


def _write_sidecar(x_path, X):
    """One group's array into `x_path`: rename a builder memmap, else save."""
    if isinstance(X, np.memmap) and os.path.exists(X.filename or ""):
        X.flush()
        src = X.filename
        del X
        if os.path.abspath(src) != os.path.abspath(x_path):
            os.replace(src, x_path)
    else:
        np.save(x_path, X)
    return x_path
