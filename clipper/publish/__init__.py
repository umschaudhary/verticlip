"""Publishers — one module per platform, all exposing:

    upload(cfg, video_path: Path, title: str, description: str, hashtags: list[str]) -> (remote_id, url)

`publish.run()` drains the post queue respecting per-platform daily caps and
spacing so the account never looks like a bot.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path

from ..config import Config
from ..ledger import Ledger

log = logging.getLogger("publish")
PLATFORMS = ("youtube", "instagram", "tiktok")


def build_caption(cfg: Config, camp, clip_caption: str, hashtags: list[str]) -> str:
    parts = [clip_caption.strip()]
    if camp.caption_suffix:
        parts.append(camp.caption_suffix)
    if hashtags:
        parts.append(" ".join(hashtags))
    return "\n\n".join(p for p in parts if p)


def run(cfg: Config, ledger: Ledger, only: list[str] | None = None,
        dry_run: bool = False) -> int:
    """Drain the post queue. With dry_run=True nothing is uploaded and the queue is
    left untouched — every gate is still evaluated and the exact title/caption that
    would be sent is logged, so the whole path short of the network call is checked."""
    camps = {c.slug: c for c in cfg.campaigns}
    lim = cfg.get("limits", default={}) or {}
    per_day = int(lim.get("max_posts_per_platform_per_day", 6))
    gap = timedelta(minutes=int(lim.get("posting_gap_minutes", 45)))
    posted = 0
    for plat in (only or PLATFORMS):
        if not cfg.get("publish", plat, "enabled", default=False):
            continue
        try:
            mod = import_module(f"clipper.publish.{plat}")
        except ImportError as e:
            log.error("publisher %s unavailable: %s", plat, e)
            continue
        if ledger.posts_today(plat) >= per_day:
            log.info("%s: daily cap reached", plat)
            continue
        last = ledger.last_post_time(plat)
        if last and datetime.now(timezone.utc) - last < gap:
            log.info("%s: waiting for posting gap", plat)
            continue
        queue = ledger.queued_posts(plat)
        if not queue:
            continue
        p = queue[0]  # one post per platform per run; scheduler runs often
        camp = camps.get(p["campaign"])
        if not camp:
            if dry_run:
                log.info("%s: DRY RUN would skip %s (campaign inactive)", plat, p["clip_id"])
                continue
            ledger.mark_post(p["id"], "skipped", error="campaign inactive")
            continue
        desc = build_caption(cfg, camp, p["caption"] or "", camp.hashtags)
        if dry_run:
            render = Path(p["render_path"]) if p["render_path"] else None
            missing = "" if render and render.exists() else "  [!] render file missing"
            log.info("%s: DRY RUN would post %s%s", plat, p["clip_id"], missing)
            log.info("    file    : %s", render)
            log.info("    title   : %s", p["title"] or p["hook"] or "Clip")
            log.info("    caption : %s", desc.replace("\n", " ⏎ "))
            if len(queue) > 1:
                log.info("    (%d more queued for %s behind this one)", len(queue) - 1, plat)
            posted += 1
            continue
        try:
            rid, url = mod.upload(cfg, Path(p["render_path"]), p["title"] or p["hook"] or "Clip", desc, camp.hashtags)
            ledger.mark_post(p["id"], "posted", remote_id=rid, url=url)
            log.info("%s: posted %s → %s", plat, p["clip_id"], url)
            posted += 1
        except Exception as e:  # noqa: BLE001
            log.exception("%s: upload failed for %s", plat, p["clip_id"])
            ledger.mark_post(p["id"], "failed", error=str(e)[:500])
    return posted
