"""Forwarding proxy for MIRROR=1 test runs (see CLAUDE.md §4).

The sandbox browser cannot reach external hosts; these proxies stand in for
GIBS, GBIF and the Open-Meteo family. The sandbox's own egress is flaky —
measured 2026-08-10, roughly one connection in ten to api.open-meteo.com
fails outright — and because each Open-Meteo call owns a whole section of the
pixel card, one dropped connection reads in the test report as "the app
stopped rendering the forecast". So the proxy absorbs transport failures
here, where they belong, rather than the app growing retries to paper over a
broken sandbox. Upstream HTTP statuses (404, 429, …) are passed through
untouched — those are answers, and the tests must see them.
"""
import sys, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]); UPSTREAM = sys.argv[2]
TRIES = 4          # transport-level only
BACKOFF = 0.6      # seconds, doubling


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        url = UPSTREAM + self.path
        req = urllib.request.Request(url, headers={"User-Agent": "earth-test-proxy"})
        last = None
        for attempt in range(TRIES):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    body = r.read()
                    ct = r.headers.get("Content-Type", "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            except urllib.error.HTTPError as e:
                # A real answer from upstream. Never retry it.
                self.send_response(e.code)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                return
            except Exception as e:
                last = e
                if attempt < TRIES - 1:
                    time.sleep(BACKOFF * (2 ** attempt))
        sys.stderr.write(f"{PORT} give up after {TRIES}: {self.path[:80]} · {last}\n")
        sys.stderr.flush()
        self.send_response(502)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
