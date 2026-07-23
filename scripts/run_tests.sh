#!/usr/bin/env bash
# Local sandbox test runner. The headless browser can't reach NASA GIBS / GBIF /
# the Cesium CDN directly, so MIRROR=1 routes them to the vendored Cesium copy
# (_vendor/cesium) and two small forwarding proxies that fetch upstream from the
# container (which does have egress). CI uses the real network and skips all this.
set -e
cd "$(dirname "$0")/.."
python3 scripts/test_proxy.py 8081 https://gibs.earthdata.nasa.gov >/tmp/gibs_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8082 https://api.gbif.org         >/tmp/gbif_proxy.log 2>&1 &
python3 -m http.server 8080 >/tmp/http.log 2>&1 &
sleep 2
export MIRROR=1 CHROMIUM_PATH="${CHROMIUM_PATH:-/opt/pw-browsers/chromium}"
npx playwright test "$@"
