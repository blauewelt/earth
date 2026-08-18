#!/usr/bin/env python3
"""Subprocess half of tests/test_stage2_float16_anomaly.py check 4.

Runs one script's transform path on a SIDECAR tensor and prints what the
parent cannot see from outside the process: whether the scratch copy existed
while the run was alive (the parent only ever sees it after atexit has removed
it), and the resulting dynamic-channel statistics.

    _sidecar_driver.py <module> <npz> <tmp> <scratch> [extra argv...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_stage2_float16_anomaly as t          # noqa: E402


def main():
    mod, data, tmp, scratch = sys.argv[1:5]
    extra = sys.argv[5:]
    seen = {"scratch": False}

    # The scratch is created inside main() and only deleted at process exit,
    # so look for it at the one moment the transform is finished and the
    # process is still alive: the codec_from_ckpt interception.
    real = t.run_transform

    def watched(*a, **k):
        try:
            return real(*a, **k)
        finally:
            seen["scratch"] = os.path.exists(scratch)

    X = watched(mod, data, tmp, extra)
    sd, zero = t.report(mod, X)
    print("SCRATCH=%d" % int(seen["scratch"]))
    print("SD=%.6f ZERO=%.6f" % (sd, zero))


if __name__ == "__main__":
    main()
