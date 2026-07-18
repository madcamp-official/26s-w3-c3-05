"""``jarvis-monitor`` — render the pipeline debug dashboard.

Until the runtime is wired end to end, this renders the representative demo
snapshot. ``--serve`` re-renders on every request (with an auto-refresh header)
so the page updates once a live snapshot builder replaces the demo one.
"""

from __future__ import annotations

import argparse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from jarvis.monitoring.demo import build_demo_snapshot
from jarvis.monitoring.render import render_html
from jarvis.monitoring.snapshot import MonitorSnapshot


def current_snapshot() -> MonitorSnapshot:
    """The snapshot to render. Swap this for a live builder once wired."""
    return build_demo_snapshot()


def _serve(port: int, refresh_s: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            html = render_html(current_snapshot())
            if refresh_s > 0:
                html = html.replace(
                    "<head>", f'<head>\n<meta http-equiv="refresh" content="{refresh_s}">', 1
                )
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:  # silence per-request logging
            return

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"serving JARVIS monitor at {url} (refresh {refresh_s}s) — Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-monitor", description=__doc__)
    parser.add_argument(
        "-o", "--output", default="monitor.html", help="HTML output path (default: monitor.html)"
    )
    parser.add_argument("--open", action="store_true", help="open the rendered file in a browser")
    parser.add_argument(
        "--serve", type=int, metavar="PORT", help="serve a live-refreshing dashboard on PORT"
    )
    parser.add_argument(
        "--refresh", type=int, default=2, help="auto-refresh seconds in --serve mode (default: 2)"
    )
    args = parser.parse_args(argv)

    if args.serve is not None:
        _serve(args.serve, args.refresh)
        return 0

    output = Path(args.output)
    output.write_text(render_html(current_snapshot()), encoding="utf-8")
    print(f"wrote {output.resolve()}")
    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
