"""Serve the Adaptive Traffic Lab web app on localhost.

Usage:  python -m backend.app [--port 8000] [--no-browser]
"""
import argparse
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    handler = functools.partial(Handler, directory=str(WEB_DIR))
    socketserver.TCPServer.allow_reuse_address = True
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = socketserver.TCPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"No free port found starting at {args.port}.")

    url = f"http://localhost:{port}"
    print(f"Adaptive Traffic Lab running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
