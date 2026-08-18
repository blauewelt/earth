#!/usr/bin/env python3
"""Family-5 tensor: the 0.25-degree North Atlantic at DAILY cadence.

E-038 Phase B. This file is deliberately almost empty: E-038 §3 specified
family 5 as "`build_family4.py` with `PENTAD_DAYS = 1` and its own
`RECIPE_REV`, not a copy" — so the builder IS build_family4.py, parameterised
by `--days`, and this wrapper only pins the parameter. Everything
cadence-specific (recipe `f5r1`, output `family5_na025_daily.npz` + its
memmappable `_X.npy` sidecar, `truth_daily.npz`, the centred 5-day rolling
wind sigma) hangs off that one flag inside the shared builder, where the
pentad path stays byte-identical to what built E-038a/b's tensors.

E-041 adds the second axis, `--rev`: `--rev r2` builds
`family5_na025_daily_r2.npz` under recipe `f5r2`, the same 39 channels plus
the appended `sst`. It passes straight through to the shared builder like
every other flag — this wrapper still pins nothing but the cadence.

Run:
  python3 ml/build_family5.py --dry-run
  python3 ml/build_family5.py --pentad-dir ml/cache/glorys_daily025
  python3 ml/build_family5.py --pentad-dir ... --max-bins 40   # smoke
  python3 ml/build_family5.py --rev r2 --pentad-dir ...        # E-041, +sst

Sizes, so nobody re-derives them: [15706, 281, 481, 39] float16 = 165.6 GB.
The sidecar layout exists because np.load on a compressed npz member
decompresses the WHOLE array into RAM, and no rentable 4090 host has 166 GB
of it — see ml/tensor_io.py. The anomaly transform additionally needs a
writable scratch copy, so the box wants >= 400 GB of disk.
"""
import sys

from build_family4 import main                     # noqa: F401

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--days", "1", *sys.argv[1:]]
    sys.exit(main())
