"""SQLite ledger — the single source of truth for what has been downloaded,
transcribed, rendered, posted and submitted. Makes every stage idempotent so
the scheduler can re-run safely with no human in the loop."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  id          TEXT PRIMARY KEY,            -- sha1 of url/path
  campaign    TEXT NOT NULL,
  kind        TEXT NOT NULL,               -- youtube | footage | drive
  origin      TEXT NOT NULL,               -- url or path
  title       TEXT,
  video_path  TEXT,
  duration_s  REAL,
  status      TEXT NOT NULL DEFAULT 'new', -- new | downloaded | transcribed | highlighted | done | failed
  error       TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
  id          TEXT PRIMARY KEY,            -- <source_id>_<n>
  source_id   TEXT NOT NULL,
  campaign    TEXT NOT NULL,
  start_s     REAL NOT NULL,
  end_s       REAL NOT NULL,
  score       REAL,
  hook        TEXT,
  title       TEXT,
  caption     TEXT,
  render_path TEXT,
  status      TEXT NOT NULL DEFAULT 'planned', -- planned | rendered | failed
  meta        TEXT,                        -- json
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_id     TEXT NOT NULL,
  campaign    TEXT NOT NULL,
  platform    TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'queued', -- queued | posted | failed | skipped
  remote_id   TEXT,
  url         TEXT,
  error       TEXT,
  attempts    INTEGER NOT NULL DEFAULT 0,
  submitted   INTEGER NOT NULL DEFAULT 0,  -- 1 once exported to submissions.csv
  posted_at   TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  UNIQUE(clip_id, platform)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Additive migrations. Existing databases must keep working untouched."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(clips)")}
        if "decision" not in cols:
            # NULL = undecided. Set by the review gate; rejected clips never post.
            self.conn.execute("ALTER TABLE clips ADD COLUMN decision TEXT")
            self.conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ── sources ────────────────────────────────────────────────
    def upsert_source(self, sid: str, campaign: str, kind: str, origin: str) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO sources(id,campaign,kind,origin,created_at,updated_at)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING""",
                (sid, campaign, kind, origin, now(), now()),
            )

    def sources_with_status(self, status: str) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM sources WHERE status=?", (status,)).fetchall()

    def update_source(self, sid: str, **fields: Any) -> None:
        fields["updated_at"] = now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE sources SET {cols} WHERE id=?", (*fields.values(), sid))

    # ── clips ──────────────────────────────────────────────────
    def add_clip(self, cid: str, source_id: str, campaign: str, start: float, end: float,
                 score: float, hook: str, title: str, caption: str, meta: dict | None = None) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR IGNORE INTO clips(id,source_id,campaign,start_s,end_s,score,hook,title,caption,meta,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (cid, source_id, campaign, start, end, score, hook, title, caption,
                 json.dumps(meta or {}), now(), now()),
            )

    def clips_with_status(self, status: str, limit: int | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM clips WHERE status=? ORDER BY score DESC, created_at"
        if limit:
            q += f" LIMIT {int(limit)}"
        return self.conn.execute(q, (status,)).fetchall()

    def update_clip(self, cid: str, **fields: Any) -> None:
        fields["updated_at"] = now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE clips SET {cols} WHERE id=?", (*fields.values(), cid))

    def clips_for_source(self, source_id: str) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM clips WHERE source_id=?", (source_id,)).fetchone()[0]

    # ── posts ──────────────────────────────────────────────────
    def decide_clip(self, cid: str, decision: str) -> int:
        """Approve or reject a clip.

        Rejecting drops its queued posts, so a clip you said no to can never be
        picked up by a later publish pass. Already-posted rows are left alone —
        rejecting after the fact cannot un-post anything.
        """
        removed = 0
        with self.tx() as c:
            c.execute("UPDATE clips SET decision=?, updated_at=? WHERE id=?",
                      (decision, now(), cid))
            if decision == "rejected":
                removed = c.execute(
                    "DELETE FROM posts WHERE clip_id=? AND status='queued'", (cid,)).rowcount
        return removed

    def decisions_for(self, clip_ids: list[str]) -> dict[str, str]:
        if not clip_ids:
            return {}
        q = ",".join("?" * len(clip_ids))
        return {r["id"]: r["decision"] for r in self.conn.execute(
            f"SELECT id, decision FROM clips WHERE id IN ({q})", clip_ids) if r["decision"]}

    def queue_post(self, clip_id: str, campaign: str, platform: str) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR IGNORE INTO posts(clip_id,campaign,platform,created_at,updated_at)
                   VALUES(?,?,?,?,?)""",
                (clip_id, campaign, platform, now(), now()),
            )

    def queued_posts(self, platform: str, max_attempts: int = 3) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT p.*, c.render_path, c.title, c.caption, c.hook
               FROM posts p JOIN clips c ON c.id=p.clip_id
               WHERE p.platform=? AND p.status IN ('queued','failed') AND p.attempts<?
                 AND c.status='rendered'
                 AND COALESCE(c.decision,'') <> 'rejected'
               ORDER BY c.score DESC, p.created_at""",
            (platform, max_attempts),
        ).fetchall()

    def posts_today(self, platform: str) -> int:
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
        return self.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE platform=? AND status='posted' AND posted_at>=?",
            (platform, since),
        ).fetchone()[0]

    def last_post_time(self, platform: str) -> datetime | None:
        r = self.conn.execute(
            "SELECT MAX(posted_at) FROM posts WHERE platform=? AND status='posted'", (platform,)
        ).fetchone()[0]
        return datetime.fromisoformat(r) if r else None

    def mark_post(self, post_id: int, status: str, remote_id: str | None = None,
                  url: str | None = None, error: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                """UPDATE posts SET status=?, remote_id=COALESCE(?,remote_id), url=COALESCE(?,url),
                   error=?, attempts=attempts+1, posted_at=CASE WHEN ?='posted' THEN ? ELSE posted_at END,
                   updated_at=? WHERE id=?""",
                (status, remote_id, url, error, status, now(), now(), post_id),
            )

    def unsubmitted_posts(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT p.*, c.title FROM posts p JOIN clips c ON c.id=p.clip_id
               WHERE p.status='posted' AND p.submitted=0 ORDER BY p.posted_at"""
        ).fetchall()

    def mark_submitted(self, ids: list[int]) -> None:
        if not ids:
            return
        with self.tx() as c:
            c.executemany("UPDATE posts SET submitted=1, updated_at=? WHERE id=?", [(now(), i) for i in ids])

    # ── reporting ──────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        q = lambda sql: self.conn.execute(sql).fetchall()  # noqa: E731
        return {
            "sources": {r[0]: r[1] for r in q("SELECT status, COUNT(*) FROM sources GROUP BY status")},
            "clips": {r[0]: r[1] for r in q("SELECT status, COUNT(*) FROM clips GROUP BY status")},
            "posts": {f"{r[0]}/{r[1]}": r[2] for r in q("SELECT platform, status, COUNT(*) FROM posts GROUP BY platform, status")},
        }
