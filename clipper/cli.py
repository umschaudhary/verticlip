#!/usr/bin/env python3
"""verticlip — pipeline orchestrator and CLI.

    verticlip --url URL --clips 6 --no-publish   # 6 shorts from one YouTube link, nothing posted
    verticlip                   # one full pass: ingest → transcribe → highlight → render → publish → export
    python run.py --dry-run     # same, but publish logs what it would post instead of posting
    python run.py --no-publish  # stop after render; posts stay queued
    python run.py --stage render
    python run.py --skip transcribe
    python run.py --status      # ledger summary
    python run.py --loop 20     # keep running every 20 minutes (alternative to launchd)

Every stage is idempotent (state lives in data/ledger.db), so this can be
fired by launchd/cron as often as you like with no human in the loop.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import logging
import re
import sys
import time
from pathlib import Path

from . import highlight, ingest, publish, render, submissions, transcribe
from .config import load_config
from .ledger import Ledger

STAGES = {
    "ingest": ingest.run,
    "transcribe": transcribe.run,
    "highlight": highlight.run,
    "render": render.run,
    "publish": publish.run,
    "export": submissions.run,
    "prune": ingest.prune_sources,      # reclaim disk once clips exist
}


def setup_logging(data_dir: Path) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)-12s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(data_dir / "logs" / "clipper.log")])


ADHOC_SLUG = "adhoc"
ADHOC_YAML = """# Auto-created by `python run.py --url ...`.
# Ad-hoc sources land here so a one-off YouTube link needs no campaign of its own.
# Edit freely — it is a normal campaign.
name: Ad-hoc
platform: direct
active: true
rate_per_1k: 0
sources:
  youtube: []
  footage_dir: footage
rules:
  platforms: [youtube, instagram, tiktok]
highlight_brief: "Self-contained moments that make sense with zero context."
"""


def add_url(url: str, campaigns_dir: Path) -> str:
    """Register a one-off source, creating the ad-hoc campaign if needed."""
    cdir = campaigns_dir / ADHOC_SLUG
    f = cdir / "campaign.yaml"
    if not f.exists():
        (cdir / "footage").mkdir(parents=True, exist_ok=True)
        f.write_text(ADHOC_YAML)
    import yaml
    y = yaml.safe_load(f.read_text()) or {}
    urls = list((y.get("sources") or {}).get("youtube") or [])
    if url not in urls:
        urls.append(url)
        y.setdefault("sources", {})["youtube"] = urls
        y["active"] = True
        f.write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True))
    return ADHOC_SLUG


def _run_dir(data_dir: Path, url: str | None) -> Path:
    """A fresh folder per pass: data/runs/<timestamp>[_<video-id>]."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    label = ""
    if url:
        m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", url)
        if m:
            label = "_" + m.group(1)
    d = data_dir / "runs" / f"{stamp}{label}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_manifest(cfg, ledger, run_dir: Path, started: str) -> int:
    """List what this run produced, next to the files themselves."""
    rows = [dict(r) for r in ledger.conn.execute(
        "SELECT id, campaign, start_s, end_s, score, hook, title, caption, render_path "
        "FROM clips WHERE render_path LIKE ? ORDER BY id", (f"{run_dir}%",))]
    (run_dir / "manifest.json").write_text(
        json.dumps({"started": started, "run_dir": str(run_dir), "clips": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        lines = [f"{len(rows)} clip(s) — {started}", ""]
        for r in rows:
            lines.append(f"{Path(r['render_path']).name}  "
                         f"{r['end_s'] - r['start_s']:.1f}s  score {r['score']:.0f}")
            lines.append(f"    hook   {r['hook']}")
            lines.append(f"    title  {r['title']}")
            lines.append("")
        (run_dir / "clips.txt").write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


CAPTION_CHOICES = ("karaoke", "pop", "boxed", "lyric", "none")
HOOK_CHOICES = ("boxed", "clean", "glow", "none")
PICK_CHOICES = ("llm", "scenes", "even")
LAYOUT_CHOICES = ("auto", "smart", "card", "center", "blur_bg")


def one_pass(stages: list[str], dry_run: bool = False, clips: int | None = None,
             url: str | None = None, captions: str | None = None,
             hook: str | None = None, pick: str | None = None,
             layout: str | None = None) -> None:
    cfg = load_config()
    if url:
        add_url(url, cfg.campaigns_dir)
        cfg = load_config()          # reload so the new source is picked up
    if captions:
        cap = cfg.raw.setdefault("render", {}).setdefault("captions", {})
        if captions == "none":
            cap["enabled"] = False          # hook title and credit overlay still render
        else:
            cap["enabled"] = True
            cap["style"] = captions
    if hook:
        cfg.raw.setdefault("render", {}).setdefault("hook_title", {})["style"] = hook
    if pick:
        cfg.raw.setdefault("highlighter", {})["strategy"] = pick
    if layout:
        cfg.raw.setdefault("render", {})["crop_mode"] = layout
    if clips:
        lim = cfg.raw.setdefault("limits", {})
        lim["clips_per_source"] = clips
        # never let the per-run ceiling silently cap what was explicitly asked for
        lim["max_clips_per_run"] = max(int(lim.get("max_clips_per_run", 12)), clips)
    setup_logging(cfg.data_dir)
    log = logging.getLogger("run")
    if not cfg.campaigns:
        log.warning("no active campaigns in %s — copy campaigns/example and edit it", cfg.campaigns_dir)
    ledger = Ledger(cfg.data_dir / "ledger.db")
    lock = (cfg.data_dir / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("another pass is running — exiting")
        return
    started = dt.datetime.now().isoformat(timespec="seconds")
    cfg.run_dir = _run_dir(cfg.data_dir, url)
    log.info("output → %s", cfg.run_dir)
    t0 = time.time()
    for name in stages:
        log.info("── %s ──", name)
        try:
            # dry_run only means anything to publish — it's the only stage that
            # touches the outside world.
            if name == "publish" and dry_run:
                STAGES[name](cfg, ledger, dry_run=True)
            else:
                STAGES[name](cfg, ledger)
        except Exception:  # noqa: BLE001
            log.exception("stage %s crashed; continuing", name)
    n = _write_manifest(cfg, ledger, cfg.run_dir, started)
    if n == 0:
        # nothing rendered this pass — don't leave an empty folder behind
        for f in cfg.run_dir.iterdir():
            f.unlink()
        cfg.run_dir.rmdir()
        log.info("no clips rendered this pass — removed empty %s", cfg.run_dir.name)
    else:
        log.info("%d clip(s) → %s", n, cfg.run_dir)
    log.info("pass finished in %.0fs  %s", time.time() - t0, json.dumps(ledger.summary()))


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="clipper — ingest, transcribe, pick highlights, render, publish.",
        epilog="""examples:
  python run.py --url URL --clips 6   6 shorts from one link (add --no-publish to keep them local)
  python run.py                       full pass, posts for real
  python run.py --dry-run             full pass, logs what it would post, uploads nothing
  python run.py --no-publish          full pass, stops after render (queue left pending)
  python run.py --stage render        just re-render planned clips
  python run.py --skip transcribe     everything except transcription
  python run.py --status              ledger summary
  python run.py --loop 20 --dry-run   every 20 min, never posting
""")
    ap.add_argument("--stage", choices=list(STAGES), action="append", metavar="STAGE",
                    help=f"run only these stage(s); repeatable. one of: {', '.join(STAGES)}")
    ap.add_argument("--skip", choices=list(STAGES), action="append", metavar="STAGE",
                    help="run everything except these stage(s); repeatable")
    ap.add_argument("--no-publish", action="store_true",
                    help="shorthand for --skip publish: render and queue, upload nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="run publish without uploading — logs the exact file, title and "
                         "caption it would send and leaves the queue untouched")
    ap.add_argument("--url", metavar="URL",
                    help="clip this YouTube URL directly — no campaign file needed. "
                         "It is added to the auto-created 'adhoc' campaign and run immediately.")
    ap.add_argument("--clips", type=int, metavar="N",
                    help="how many clips to cut per source (default: limits.clips_per_source "
                         "in config.yaml, which ships as 1)")
    ap.add_argument("--captions", choices=list(CAPTION_CHOICES), metavar="STYLE",
                    help="caption style for this run: " + ", ".join(CAPTION_CHOICES) +
                         ". 'none' turns burned-in captions off entirely "
                         "(default: render.captions.style in config.yaml)")
    ap.add_argument("--hook", choices=list(HOOK_CHOICES), metavar="STYLE",
                    help="opening hook-title style: " + ", ".join(HOOK_CHOICES) +
                         ". 'none' removes the title card entirely "
                         "(default: render.hook_title.style in config.yaml)")
    ap.add_argument("--pick", choices=list(PICK_CHOICES), metavar="HOW",
                    help="how to choose clips: llm (read the transcript, default), "
                         "scenes (start each clip on a shot change), even (space them out). "
                         "scenes/even need no transcription at all when captions are off")
    ap.add_argument("--layout", choices=list(LAYOUT_CHOICES), metavar="MODE",
                    help="framing: auto (crop, split when two faces), smart (crop, "
                         "never split), card (whole frame inset over a blurred backdrop, "
                         "nothing cropped), center, blur_bg")
    ap.add_argument("--ui", action="store_true",
                    help="open the local web UI instead of running once "
                         "(http://127.0.0.1:8765 — pick options, watch progress, review clips)")
    ap.add_argument("--port", type=int, default=8765, metavar="N", help="port for --ui")
    ap.add_argument("--status", action="store_true", help="print ledger summary and exit")
    ap.add_argument("--loop", type=int, metavar="MINUTES", help="run forever, every N minutes")
    a = ap.parse_args()

    if a.clips is not None and a.clips < 1:
        ap.error("--clips must be at least 1")
    if a.url and a.stage and "ingest" not in a.stage:
        ap.error("--url needs the ingest stage; drop --stage or add --stage ingest")
    if a.dry_run and (a.no_publish or (a.skip and "publish" in a.skip)):
        ap.error("--dry-run needs the publish stage to run; drop --no-publish/--skip publish")

    if a.ui:
        from .webui import serve
        cfg = load_config()
        setup_logging(cfg.data_dir)
        serve(load_config, one_pass, port=a.port)
        return

    if a.status:
        cfg = load_config()
        print(json.dumps(Ledger(cfg.data_dir / "ledger.db").summary(), indent=2))
        return
    stages = a.stage or list(STAGES)
    skip = set(a.skip or [])
    if a.no_publish:
        skip.add("publish")
    stages = [s for s in stages if s not in skip]
    if not stages:
        ap.error("every stage was skipped — nothing to do")

    if a.loop:
        while True:
            one_pass(stages, dry_run=a.dry_run, clips=a.clips, url=a.url, captions=a.captions, hook=a.hook, pick=a.pick, layout=a.layout)
            a.url = None          # register the URL on the first pass only
            time.sleep(a.loop * 60)
    else:
        one_pass(stages, dry_run=a.dry_run, clips=a.clips, url=a.url, captions=a.captions, hook=a.hook, pick=a.pick, layout=a.layout)


if __name__ == "__main__":
    main()
