#!/usr/bin/env python3
"""Extract the raw X memmap + small members from the box's verified tensor.

recon_decoder.py wants X as a bare .npy it can memmap (the npz member is
deflate-compressed and cannot be mapped) plus the small metadata members in
their own npz. This runs once per box; both outputs live in ml/cache next to
the tensor and survive between dectrain dispatches.

The disk guard is sized from the allocation it guards (ml/CLAUDE.md §5.18):
X.npy is ~10.9 GiB and an ENOSPC mid-write is exactly the "50/50 disk,
every run pays an 80-minute rebuild" treadmill of 2026-08-11.

Usage: python3 scripts/dectrain_extract.py --tensor ml/cache/family3_na025.npz
Idempotent: exits 0 doing nothing if both outputs exist and X.npy's size
matches the zip entry (a partial extract from a killed job re-extracts).
"""
import argparse
import os
import shutil
import sys
import zipfile

import numpy as np

SMALL = ["months", "lats", "lons", "chan", "norm", "rapid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--out-x", default="ml/cache/family3_X.npy")
    ap.add_argument("--out-small", default="ml/cache/f3_dec_small.npz")
    a = ap.parse_args()

    z = zipfile.ZipFile(a.tensor)
    need = z.getinfo("X.npy").file_size
    if (os.path.exists(a.out_x) and os.path.getsize(a.out_x) == need
            and os.path.exists(a.out_small)):
        print(f"already extracted: {a.out_x} ({need:,} bytes) + {a.out_small}")
        return 0

    free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.tensor))).free
    if free < need + (2 << 30):
        print(f"::error::extracting X.npy needs {need >> 30} GiB "
              f"(+2 headroom); disk has {free >> 30} GiB free")
        return 1

    with z.open("X.npy") as src, open(a.out_x, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 22)
    d = np.load(a.tensor, allow_pickle=False)
    np.savez(a.out_small, **{k: d[k] for k in SMALL})
    print(f"extracted {a.out_x} ({os.path.getsize(a.out_x):,} bytes) "
          f"+ {a.out_small}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
