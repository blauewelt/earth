#!/usr/bin/env bash
# Local sandbox test runner. The headless browser can't reach NASA GIBS / GBIF /
# the Cesium CDN directly, so MIRROR=1 routes them to the vendored Cesium copy
# (_vendor/cesium) and two small forwarding proxies that fetch upstream from the
# container (which does have egress). CI uses the real network and skips all this.
set -e
cd "$(dirname "$0")/.."
python3 scripts/test_proxy.py 8081 https://gibs.earthdata.nasa.gov >/tmp/gibs_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8082 https://api.gbif.org         >/tmp/gbif_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8083 https://api.open-meteo.com   >/tmp/meteo_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8084 https://air-quality-api.open-meteo.com >/tmp/aq_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8085 https://flood-api.open-meteo.com       >/tmp/flood_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8086 https://marine-api.open-meteo.com      >/tmp/marine_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8087 https://climate-api.open-meteo.com     >/tmp/climate_proxy.log 2>&1 &
# the third backend (CLAUDE.md §3): four keyless tile hosts
python3 scripts/test_proxy.py 8088 https://wmts.terrascope.be              >/tmp/terrascope_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8089 https://storage.googleapis.com          >/tmp/gcs_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8090 https://tiles.maps.eox.at               >/tmp/eox_proxy.log 2>&1 &
python3 scripts/test_proxy.py 8091 https://wmts.geo.admin.ch               >/tmp/swisstopo_proxy.log 2>&1 &
python3 -m http.server 8080 >/tmp/http.log 2>&1 &
sleep 2
export MIRROR=1 CHROMIUM_PATH="${CHROMIUM_PATH:-/opt/pw-browsers/chromium}"
npx playwright test "$@"
