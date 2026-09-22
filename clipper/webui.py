"""A local web UI — pick options, watch the run, review what came out.

Deliberately stdlib-only. It adds no dependency to the project, starts in
milliseconds, and is the same surface a desktop shell (Tauri) would wrap later,
so the browser and the eventual app never diverge.

    python run.py --ui            # http://127.0.0.1:8765

Binds to loopback by default: this exposes a control surface that downloads and
publishes video, and it should not be reachable from the network by accident.
"""
from __future__ import annotations

import html
import json
import logging
import mimetypes
import os
import queue
import subprocess
import threading
import time
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

log = logging.getLogger("webui")
WEB = Path(__file__).resolve().parent / "web"

CAPTION_CHOICES = ["boxed", "pop", "karaoke", "lyric", "none"]
HOOK_CHOICES = ["clean", "glow", "boxed", "none"]
LAYOUT_CHOICES = ["auto", "card", "smart", "center", "blur_bg"]
PICK_CHOICES = ["llm", "scenes", "even"]


# ── live log fan-out ──────────────────────────────────────────

class _Bus(logging.Handler):
    """Fans log records out to every connected browser.

    Each listener gets its own bounded queue: a browser that stops reading must
    not be able to block the pipeline, so a full queue drops the oldest line
    rather than applying backpressure to the run.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._listeners: list[queue.Queue] = []

    def listen(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=400)
        with self._lock:
            self._listeners.append(q)
        return q

    def drop(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def publish(self, kind: str, text: str) -> None:
        msg = {"kind": kind, "text": text, "t": time.time()}
        with self._lock:
            targets = list(self._listeners)
        for q in targets:
            try:
                q.put_nowait(msg)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except queue.Empty:
                    pass

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "webui":
            return
        try:
            self.publish("log", f"{record.name:<11} {record.getMessage()}")
        except Exception:  # noqa: BLE001 — logging must never break the run
            pass


BUS = _Bus()


# ── the run, on its own thread ────────────────────────────────

class Runner:
    """One pipeline pass at a time, so two browser tabs cannot start two runs."""

    def __init__(self, one_pass):
        self._one_pass = one_pass
        self._thread: threading.Thread | None = None
        self.state = "idle"          # idle | running | done | failed
        self.started_at: float | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, opts: dict) -> bool:
        if self.busy:
            return False
        self.state, self.started_at = "running", time.time()
        BUS.publish("state", "running")

        def work():
            try:
                self._one_pass(
                    stages=opts["stages"], dry_run=False,
                    clips=opts.get("clips"), url=opts.get("url") or None,
                    captions=opts.get("captions"), hook=opts.get("hook"),
                    pick=opts.get("pick"), layout=opts.get("layout"),
                )
                self.state = "done"
            except Exception as e:  # noqa: BLE001
                self.state = "failed"
                BUS.publish("log", f"run failed: {e}")
                log.exception("run failed")
            finally:
                BUS.publish("state", self.state)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        return True


# ── reading what came out ─────────────────────────────────────

def list_runs(cfg, limit: int = 12) -> list[dict]:
    """Newest-first runs that actually produced clips."""
    root = cfg.data_dir / "runs"
    out = []
    if not root.exists():
        return out
    for d in sorted((p for p in root.iterdir() if p.is_dir()),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        clips = sorted(d.glob("*.mp4"))
        if not clips:
            continue
        meta = {}
        mf = d / "manifest.json"
        if mf.exists():
            try:
                meta = {c["id"]: c for c in json.loads(mf.read_text()).get("clips", [])}
            except (json.JSONDecodeError, KeyError, TypeError):
                meta = {}
        campaign = next((c.get("campaign") for c in meta.values() if c.get("campaign")), "")
        out.append({
            "name": d.name,
            "path": str(d),
            "campaign": campaign,
            "clips": [{
                "file": c.name,
                "size": c.stat().st_size,
                "id": c.stem,
                "hook": (meta.get(c.stem) or {}).get("hook", ""),
                "title": (meta.get(c.stem) or {}).get("title", ""),
                "seconds": round((meta.get(c.stem) or {}).get("end_s", 0)
                                 - (meta.get(c.stem) or {}).get("start_s", 0), 1),
            } for c in clips],
        })
        if len(out) >= limit:
            break
    return out


def thumbnail(video: Path, cfg=None) -> Path:
    """A poster frame for a clip, extracted once and cached beside it.

    Browsers will not paint a frame from preload="metadata", and preloading whole
    clips to get one would download every video in the grid. A small JPEG is
    cheaper than either.
    """
    out = video.with_suffix(".jpg")
    if out.exists() and out.stat().st_mtime >= video.stat().st_mtime:
        return out
    from .binaries import resolve
    try:
        subprocess.run(
            [resolve("ffmpeg", cfg), "-v", "error", "-y", "-ss", "0.6", "-i", str(video),
             "-frames:v", "1", "-vf", "scale=360:-2", str(out)],
            check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        log.warning("no thumbnail for %s: %s", video.name, e)
    return out


# ── HTTP ──────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "verticlip"

    def __init__(self, *a, cfg_loader=None, runner=None, **kw):
        self.cfg_loader, self.runner = cfg_loader, runner
        super().__init__(*a, **kw)

    def log_message(self, fmt, *args):      # quiet: the UI shows the pipeline log
        pass

    def handle_one_request(self):
        # A browser closing a video stream or an SSE tab resets the socket. That is
        # normal here and must not print a traceback for every closed tab.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    # -- helpers ------------------------------------------------
    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _file(self, path: Path):
        """Serve a file, honouring Range so <video> can seek."""
        if not path.is_file():
            return self._send(404, b"not found", "text/plain")
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng and rng.startswith("bytes="):
            try:
                s, _, e = rng[6:].partition("-")
                start = int(s) if s else 0
                end = int(e) if e else size - 1
                end = min(end, size - 1)
                if start > end:
                    raise ValueError
                code = 206
            except ValueError:
                return self._send(416, b"bad range", "text/plain")
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass        # the browser seeked away mid-stream; normal for video

    @staticmethod
    def _under_runs(cfg, rel: str) -> Path | None:
        """Resolve `rel` inside data/runs, or None if it escapes."""
        root = (cfg.data_dir / "runs").resolve()
        try:
            target = (root / rel).resolve()
            return target if target.is_relative_to(root) else None
        except (OSError, ValueError):
            return None

    # -- routes -------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)

        if p in ("/", "/index.html"):
            f = WEB / "index.html"
            if not f.exists():
                return self._send(500, b"web/index.html missing", "text/plain")
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")

        if p == "/api/options":
            cfg = self.cfg_loader()
            r = cfg.get("render", default={}) or {}
            cap = r.get("captions", {}) or {}
            return self._json({
                "captions": CAPTION_CHOICES, "hooks": HOOK_CHOICES,
                "layouts": LAYOUT_CHOICES, "picks": PICK_CHOICES,
                "defaults": {
                    "clips": int((cfg.get("limits", default={}) or {}).get("clips_per_source", 1)),
                    "captions": "none" if not cap.get("enabled", True) else cap.get("style", "boxed"),
                    "hook": (r.get("hook_title", {}) or {}).get("style", "clean"),
                    "layout": r.get("crop_mode", "auto"),
                    "pick": (cfg.get("highlighter", default={}) or {}).get("strategy", "llm"),
                },
                "busy": self.runner.busy, "state": self.runner.state,
            })

        if p == "/api/runs":
            return self._json({"runs": list_runs(self.cfg_loader())})

        if p == "/api/events":
            return self._sse()

        if p.startswith("/thumb/"):
            cfg = self.cfg_loader()
            rel = p[len("/thumb/"):]
            video = self._under_runs(cfg, rel)
            if video is None:
                return self._send(403, b"forbidden", "text/plain")
            return self._file(thumbnail(video, cfg))

        if p.startswith("/media/"):
            cfg = self.cfg_loader()
            target = self._under_runs(cfg, p[len("/media/"):])
            if target is None:
                return self._send(403, b"forbidden", "text/plain")
            return self._file(target)

        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if unquote(u.path) != "/api/run":
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "bad request body"}, 400)

        url = (body.get("url") or "").strip()
        stages = ["ingest", "transcribe", "highlight", "render", "export", "prune"]
        if not url:
            stages = ["render", "export"]          # re-render what is already planned
        opts = {
            "url": url, "stages": stages,
            "clips": max(1, int(body.get("clips") or 1)),
            "captions": body.get("captions"), "hook": body.get("hook"),
            "pick": body.get("pick"), "layout": body.get("layout"),
        }
        if not self.runner.start(opts):
            return self._json({"error": "a run is already in progress"}, 409)
        return self._json({"ok": True})

    def _sse(self):
        q = BUS.listen()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15)
                    payload = json.dumps(msg)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")   # keeps proxies from closing us
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            BUS.drop(q)


def serve(cfg_loader, one_pass, host="127.0.0.1", port=8765, open_browser=True):
    logging.getLogger().addHandler(BUS)
    runner = Runner(one_pass)
    handler = partial(Handler, cfg_loader=cfg_loader, runner=runner)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    url = f"http://{host}:{port}"
    print(f"\n  verticlip  →  {url}\n  ctrl-c to stop\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
