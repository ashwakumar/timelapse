"""Local jury demo: sourcing story, held-out inputs, OpenTSLM generation."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"


def _json_bytes(payload: object, status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload).encode("utf-8"), "application/json"


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print(f"[demo] {args[0]}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_file(STATIC / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/runs":
            from demo.runs import list_runs

            self._send(*_json_bytes({"runs": list_runs(ROOT)}))
            return
        if parsed.path == "/api/story":
            from demo.story import build_story

            self._send(*_json_bytes(build_story(ROOT)))
            return
        if parsed.path == "/api/samples":
            from demo.infer import list_samples

            self._send(*_json_bytes({"samples": list_samples()}))
            return
        if parsed.path == "/api/input":
            from demo.infer import sample_input

            query = parse_qs(parsed.query)
            index = int(query.get("index", ["0"])[0])
            run_id = query.get("run", [None])[0]
            self._send(*_json_bytes(sample_input(index, run_id)))
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/generate":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        from demo.infer import generate

        self._send(
            *_json_bytes(
                generate(int(body.get("index", 0)), body.get("run") or body.get("run_id"))
            )
        )

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send(404, b"missing static file", "text/plain")
            return
        self._send(200, path.read_bytes(), content_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    if args.preload:
        from demo.infer import load_runtime

        print("[demo] loading trained best.pt …", flush=True)
        load_runtime()
        print("[demo] model ready", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
