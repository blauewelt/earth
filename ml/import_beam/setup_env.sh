#!/usr/bin/env bash
# Build the virtual environment for the import.
#
# THE ORDER MATTERS. `setuptools<70` is pinned and installed FIRST, on its own,
# because three of apache-beam's dependencies — crcmod, dill 0.3.1.1 and hdfs —
# are still source distributions that use setuptools APIs removed in version 70.
# With a current setuptools the build of those sdists fails and you are left
# with a venv that imports apache_beam and cannot run a pipeline. This was
# measured, not guessed.
#
#   bash setup_env.sh [venv-dir]      # default: ./beamenv
#
# Afterwards:
#   source beamenv/bin/activate
#   python -m beam_import.registry --check
set -euo pipefail

VENV="${1:-$(cd "$(dirname "$0")" && pwd)/beamenv}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== creating $VENV"
python3 -m venv "$VENV"

echo "== step 1/3: pip, wheel, and setuptools<70 (before anything else)"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "setuptools<70" wheel

echo "== step 2/3: the rest of requirements.txt"
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

echo "== step 3/3: checking the install"
"$VENV/bin/python" - <<'PY'
import apache_beam, crcmod, netCDF4, numpy, requests, yaml, xarray
print("apache-beam       ", apache_beam.__version__)
print("netCDF4           ", netCDF4.__version__)
print("numpy             ", numpy.__version__)
print("xarray            ", xarray.__version__)
try:
    import copernicusmarine
    print("copernicusmarine  ", copernicusmarine.__version__)
except Exception as exc:
    print("copernicusmarine   MISSING:", exc)
print("crcmod             present (the TFRecord checksums)")
try:
    import cdsapi
    print("cdsapi             present (Tier 2 only)")
except Exception:
    print("cdsapi             missing (fine until Tier 2)")
try:
    import tensorflow
    print("tensorflow         present — optional; the package does not need it")
except Exception:
    print("tensorflow         absent — correct: TFRecord is read and written"
          " by beam_import itself")
PY

cat <<EOF

Done. Next:

  export EARTH_REPO=/path/to/earth        # the blauewelt/earth checkout
  source $VENV/bin/activate
  python -m beam_import.registry --check
  bash run_smoke.sh
  OUTPUT=/data/import bash run_until_complete.sh --tiers 0

EOF
