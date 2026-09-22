"""Stage 1 — discover and download source footage for every active campaign.

* YouTube videos / playlists / channels via yt-dlp (720p max — plenty for 9:16 crops)
* Campaign-provided footage dropped into campaigns/<slug>/footage/
* Public Google Drive / Dropbox folder links via gdown
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from pathlib import Path

from .config import Campaign, Config
from .binaries import resolve
from .ledger import Ledger

log = logging.getLogger("ingest")
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}


def sid_for(origin: str) -> str:
    return hashlib.sha1(origin.encode()).hexdigest()[:12]


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [resolve("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


# ── discovery ─────────────────────────────────────────────────

def _expand_youtube(url: str) -> list[dict]:
    """Return [{id,url,title}] — expands playlists/channels to individual videos.

    A URL carrying `v=` names one specific video, so it is treated as one video
    even when it also carries `list=`. YouTube appends a Mix (`list=RD...`) to
    ordinary watch links, and expanding one of those turns a single-clip request
    into hundreds of downloads. Playlist- and channel-only URLs still expand.
    """
    single = bool(re.search(r"[?&]v=", url))
    cmd = [resolve("yt-dlp"), "--flat-playlist", "-J", "--no-warnings"]
    if single:
        cmd.append("--no-playlist")
    cmd.append(url)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:  # binary missing — must not kill discovery for other sources
        log.error("cannot run yt-dlp (%s); skipping YouTube source %s", e, url)
        return []
    if out.returncode != 0:
        log.warning("yt-dlp listing failed for %s: %s", url, out.stderr[-300:])
        return []
    info = json.loads(out.stdout)
    entries = info.get("entries") or [info]
    vids = []
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        if not vid:
            continue
        vids.append({"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "title": e.get("title", "")})
    return vids


def discover(cfg: Config, ledger: Ledger) -> int:
    n = 0
    for camp in cfg.campaigns:
        # An [EDITS] campaign inverts the usual arrangement: the campaign supplies
        # the AUDIO and you supply the visuals. So the track is the one source that
        # gets downloaded and transcribed (into lyric timings), and footage/ is a
        # pool the renderer draws from — not a set of sources in its own right.
        if camp.mode == "edits":
            if not camp.track:
                log.error("campaign %s is mode: edits but has no sources.track", camp.slug)
                continue
            kind = "youtube" if camp.track.startswith(("http://", "https://")) else "footage"
            ledger.upsert_source(sid_for(camp.track), camp.slug, kind, camp.track)
            n += 1
            continue
        for url in camp.youtube_sources:
            for v in _expand_youtube(url):
                ledger.upsert_source(sid_for(v["url"]), camp.slug, "youtube", v["url"])
                ledger.update_source(sid_for(v["url"]), title=v["title"])
                n += 1
        if camp.footage_dir and camp.footage_dir.exists():
            for f in sorted(camp.footage_dir.iterdir()):
                if f.suffix.lower() in VIDEO_EXT:
                    ledger.upsert_source(sid_for(str(f)), camp.slug, "footage", str(f))
                    n += 1
        for link in camp.drive_links:
            ledger.upsert_source(sid_for(link), camp.slug, "drive", link)
            n += 1
    log.info("discovered %d source(s)", n)
    return n


# ── download ──────────────────────────────────────────────────

def _download_youtube(url: str, dest_dir: Path, sid: str, max_h: int = 1080) -> Path:
    """Download at up to `max_h` vertical pixels.

    Resolution matters more here than for normal playback: a 9:16 slice of a
    16:9 frame is only height*0.5625 wide, so a 720p source gives a 405px-wide
    crop that then has to be upscaled 2.7x to reach 1080x1920. At 1080p that
    upscale drops to 1.8x, and a 4K source needs none at all.
    """
    out_tmpl = str(dest_dir / f"{sid}.%(ext)s")
    cmd = [
        resolve("yt-dlp"), "--no-warnings", "--no-playlist",
        "-f", (f"bv*[height<={max_h}][ext=mp4]+ba[ext=m4a]/"
               f"bv*[height<={max_h}]+ba/b[height<={max_h}]/b"),
        "--merge-output-format", "mp4",
        "--newline",                     # one progress line per update, not \r
        "-o", out_tmpl, url,
    ]
    # stream yt-dlp's own progress instead of swallowing it: a 300 MB download
    # is otherwise several silent minutes with no sign it is working
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    for line in proc.stdout:
        m = re.search(r"\[download\]\s+([0-9.]+)%.*?of\s+~?([0-9.]+\w+)(?:.*?at\s+([0-9.]+\w+/s))?", line)
        if m:
            pct = float(m.group(1))
            bucket = f"{int(pct // 10) * 10}%"
            if bucket != last:
                last = bucket
                log.info("  downloading %s of %s%s", bucket, m.group(2),
                         f" at {m.group(3)}" if m.group(3) else "")
        elif "Merging formats" in line:
            log.info("  merging video + audio")
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    matches = list(dest_dir.glob(f"{sid}.*"))
    if not matches:
        raise RuntimeError("yt-dlp produced no file")
    return matches[0]


def _download_drive(link: str, dest_dir: Path, sid: str) -> list[Path]:
    """Folder or file share links. Uses gdown for Google Drive; direct URL otherwise."""
    target = dest_dir / sid
    target.mkdir(exist_ok=True)
    if "drive.google.com" in link:
        subprocess.run([resolve("gdown"), "--folder", "--remaining-ok", "-O", str(target), link],
                       check=True, capture_output=True, text=True)
    else:
        # Dropbox: force direct download
        dl = link.replace("dl=0", "dl=1")
        subprocess.run([resolve("curl"), "-L", "-o", str(target / "download.bin"), dl], check=True,
                       capture_output=True, text=True)
    return [p for p in target.rglob("*") if p.suffix.lower() in VIDEO_EXT]


def download_pending(cfg: Config, ledger: Ledger) -> int:
    dest = cfg.data_dir / "downloads"
    n = 0
    for s in ledger.sources_with_status("new"):
        sid, kind, origin = s["id"], s["kind"], s["origin"]
        try:
            if kind == "youtube":
                max_h = int(cfg.get("download", "max_height", default=1080))
                path = _download_youtube(origin, dest, sid, max_h)
            elif kind == "footage":
                path = Path(origin)
            elif kind == "drive":
                files = _download_drive(origin, dest, sid)
                if not files:
                    raise RuntimeError("no video files in drive link")
                # register every file as its own 'footage' source, retire the link itself
                for f in files:
                    ledger.upsert_source(sid_for(str(f)), s["campaign"], "footage", str(f))
                ledger.update_source(sid, status="done")
                continue
            else:
                raise RuntimeError(f"unknown kind {kind}")
            dur = _ffprobe_duration(path)
            ledger.update_source(sid, status="downloaded", video_path=str(path), duration_s=dur,
                                 title=s["title"] or path.stem)
            log.info("downloaded %s (%.0fs) → %s", sid, dur, path.name)
            n += 1
        except Exception as e:  # noqa: BLE001
            log.exception("download failed for %s", origin)
            ledger.update_source(sid, status="failed", error=str(e)[:500])
    return n


def prune_sources(cfg: Config, ledger: Ledger) -> int:
    """Delete downloaded source videos once every clip from them is rendered.

    Sources are by far the biggest thing on disk — a 30-minute 1080p video is
    ~300 MB, against ~20 MB for the clips cut from it. Transcripts are kept, so
    re-picking highlights stays free; only re-rendering would need the download
    back. Campaign footage under campaigns/ is never touched, only files this
    pipeline downloaded itself.
    """
    if not bool(cfg.get("retention", "delete_source_after_render", default=True)):
        return 0
    keep_days = float(cfg.get("retention", "keep_days", default=0) or 0)
    cutoff = time.time() - keep_days * 86400
    downloads = (cfg.data_dir / "downloads").resolve()
    freed = 0
    rows = ledger.conn.execute(
        "SELECT s.id, s.video_path, s.title FROM sources s WHERE s.video_path IS NOT NULL"
    ).fetchall()
    for r in rows:
        pending = ledger.conn.execute(
            "SELECT COUNT(*) FROM clips WHERE source_id=? AND status NOT IN ('rendered','failed')",
            (r["id"],)).fetchone()[0]
        total = ledger.conn.execute(
            "SELECT COUNT(*) FROM clips WHERE source_id=?", (r["id"],)).fetchone()[0]
        if total == 0 or pending:
            continue                      # nothing cut yet, or still work to do
        f = Path(r["video_path"])
        try:
            if not f.resolve().is_relative_to(downloads):
                continue                  # campaign footage — not ours to delete
        except (OSError, ValueError):
            continue
        if not f.exists() or f.stat().st_mtime > cutoff:
            continue
        size = f.stat().st_size
        f.unlink()
        ledger.update_source(r["id"], video_path=None)
        freed += size
        log.info("pruned source %s (%.0f MB) — %d clip(s) already rendered",
                 (r["title"] or r["id"])[:40], size / 1e6, total)
    if freed:
        log.info("freed %.0f MB of source video", freed / 1e6)
    return freed


def run(cfg: Config, ledger: Ledger) -> None:
    discover(cfg, ledger)
    download_pending(cfg, ledger)
