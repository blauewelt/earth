import sys, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]); UPSTREAM = sys.argv[2]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        url = UPSTREAM + self.path
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "earth-test-proxy"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read(); ct = r.headers.get("Content-Type", "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        except urllib.error.HTTPError as e:
            self.send_response(e.code); self.end_headers()
        except Exception as e:
            self.send_response(502); self.end_headers()

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
