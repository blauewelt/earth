#!/usr/bin/env python3
"""Run ACOLITE with GDAL reading FULL-RESOLUTION JPEG 2000 data.

WHY (E-092 s2fdi_coast, falsifier (i) diagnosed 2026-10-14). ACOLITE puts
every Sentinel-2 band on the output grid with `gdal.Warp(...,
resampleAlg='average')` (acolite/shared/read_band.py, called from
acolite/sentinel2/l1_convert.py). GDAL's default `overviewLevel='AUTO'` then
answers a 10 m -> 20 m request from the JPEG 2000 file's OWN 20 m resolution
level — the codec's wavelet low-pass — and not from the 10 m pixels. Measured
on S2B_16PCC_20180611 B08: ACOLITE's 20 m B08 equalled the file's overview 0
exactly and differed from the true 2x2 mean by a median of 20 digital numbers
(max 5,868); its kernel put 0.635 of the weight on the north-west sub-pixel
with negative lobes, centred ~7 m north-west of the cell. B04 and B08 then
sit on a different footprint from the native 20 m B06 and B11, and FDI
decorrelates within a scene (Spearman against MARIDA 0.82-0.97; with full-
resolution reads 0.97-0.999). `docs`: ml/family1/probes/
s2fdi_coast_diagnosis_i.json.

WHAT THIS DOES. It wraps `osgeo.gdal.Warp` so that every call that does not
name an `overviewLevel` gets 'NONE' (read the base resolution, then
resample), prints ONE marker line (`READ_PATH_MARKER`) the adapter checks in
the log, and then runs ACOLITE's own `launch_acolite.py` unchanged in-process.
Upsampling (60 m -> 20 m) and nearest-neighbour geometry warps are unaffected
by overviews except that they too now read the base level, which is the
correct source.

usage: python3 acolite_full_res.py <ACOLITE checkout> --cli --settings=FILE
"""
import os
import sys

READ_PATH = "gdal.Warp overviewLevel=NONE (full-resolution JPEG 2000 reads)"
READ_PATH_MARKER = "s2fdi_coast read path: " + READ_PATH


def patch_gdal():
    """Make gdal.Warp default to overviewLevel='NONE'. Idempotent."""
    from osgeo import gdal
    if getattr(gdal.Warp, "_s2fdi_full_res", False):
        return gdal
    orig = gdal.Warp

    def warp(*a, **k):
        k.setdefault("overviewLevel", "NONE")
        return orig(*a, **k)
    warp._s2fdi_full_res = True
    warp._orig = orig
    gdal.Warp = warp
    return gdal


def main(argv):
    if len(argv) < 2:
        sys.exit("usage: acolite_full_res.py <ACOLITE dir> --cli "
                 "--settings=FILE")
    aco = os.path.abspath(argv[1])
    launcher = os.path.join(aco, "launch_acolite.py")
    if not os.path.isfile(launcher):
        sys.exit(f"{aco}: no launch_acolite.py")
    patch_gdal()
    print(READ_PATH_MARKER, flush=True)
    os.chdir(aco)
    sys.path.insert(0, aco)
    sys.argv = [launcher] + list(argv[2:])
    code = compile(open(launcher).read(), launcher, "exec")
    exec(code, {"__name__": "__main__", "__file__": launcher})


if __name__ == "__main__":
    main(sys.argv)
