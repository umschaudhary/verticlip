"""Stage 6 — export every newly-posted link to submissions.csv, grouped by
campaign, ready to paste into the Whop / Vyro campaign submission form.

Whop and Vyro have no public submission API, so this is the one place a human
touches the loop: open the CSV (or the campaign page), paste links, done.
Each row also carries an estimated payout at the campaign's rate so you can
see what's worth chasing.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from .config import Config
from .ledger import Ledger

log = logging.getLogger("submissions")


def run(cfg: Config, ledger: Ledger) -> int:
    rows = ledger.unsubmitted_posts()
    if not rows:
        return 0
    out = Path(cfg.get("submissions", "export_csv", default="./data/submissions.csv"))
    if not out.is_absolute():
        out = (cfg.root / out).resolve()
    new_file = not out.exists()
    camps = {c.slug: c for c in cfg.campaigns}
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["posted_at", "campaign", "campaign_url", "platform", "clip_id", "title", "post_url", "rate_per_1k"])
        for r in rows:
            c = camps.get(r["campaign"])
            w.writerow([r["posted_at"], r["campaign"], c.campaign_url if c else "", r["platform"],
                        r["clip_id"], r["title"], r["url"], c.rate_per_1k if c else ""])
    ledger.mark_submitted([r["id"] for r in rows])
    log.info("exported %d link(s) → %s", len(rows), out)
    return len(rows)
