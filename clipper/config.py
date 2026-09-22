"""Load config.yaml, .env, and all campaign.yaml files into plain dataclasses."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_env(root: Path) -> None:
    """Minimal .env loader (no extra dependency)."""
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Campaign:
    slug: str
    dir: Path
    name: str
    active: bool
    mode: str            # "clip" — cut the campaign's own video (a [CLIPPING] brief)
                         # "edits" — lay the campaign's track under your own footage ([EDITS])
    platform: str
    campaign_url: str
    rate_per_1k: float
    payout_cap_usd: float
    youtube_sources: list[str]
    footage_dir: Path | None
    drive_links: list[str]
    track: str           # edits mode: the campaign's song (YouTube URL or local file)
    rules: dict[str, Any]
    highlight_brief: str

    @property
    def platforms(self) -> list[str]:
        return list(self.rules.get("platforms", ["youtube", "instagram", "tiktok"]))

    @property
    def hashtags(self) -> list[str]:
        return list(self.rules.get("required_hashtags", []))

    @property
    def credit_overlay(self) -> str:
        return self.rules.get("required_credit_overlay", "") or ""

    @property
    def caption_suffix(self) -> str:
        return self.rules.get("required_caption_text", "") or ""


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path
    data_dir: Path
    campaigns_dir: Path
    campaigns: list[Campaign] = field(default_factory=list)
    run_dir: Path | None = None    # set per pass; renders land here instead of data/renders

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def _resolve(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def load_campaign(cdir: Path) -> Campaign | None:
    f = cdir / "campaign.yaml"
    if not f.exists():
        return None
    y = yaml.safe_load(f.read_text()) or {}
    sources = y.get("sources", {}) or {}
    footage = sources.get("footage_dir")
    footage_path = (cdir / footage).resolve() if footage else None
    return Campaign(
        slug=cdir.name,
        dir=cdir,
        name=y.get("name", cdir.name),
        active=bool(y.get("active", True)),
        mode=str(y.get("mode", "clip")).lower(),
        platform=y.get("platform", "direct"),
        campaign_url=y.get("campaign_url", ""),
        rate_per_1k=float(y.get("rate_per_1k", 0) or 0),
        payout_cap_usd=float(y.get("payout_cap_usd", 0) or 0),
        youtube_sources=list(sources.get("youtube", []) or []),
        footage_dir=footage_path,
        drive_links=list(sources.get("drive_links", []) or []),
        track=(sources.get("track") or "").strip(),
        rules=y.get("rules", {}) or {},
        highlight_brief=(y.get("highlight_brief", "") or "").strip(),
    )


def load_config(root: Path | None = None) -> Config:
    root = root or ROOT
    _load_env(root)
    raw = yaml.safe_load((root / "config.yaml").read_text()) or {}
    data_dir = _resolve(root, raw.get("paths", {}).get("data_dir", "./data"))
    campaigns_dir = _resolve(root, raw.get("paths", {}).get("campaigns_dir", "./campaigns"))
    for sub in ("downloads", "transcripts", "renders", "logs"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    cfg = Config(raw=raw, root=root, data_dir=data_dir, campaigns_dir=campaigns_dir)
    if campaigns_dir.exists():
        for cdir in sorted(campaigns_dir.iterdir()):
            if cdir.is_dir() and cdir.name != "example":
                c = load_campaign(cdir)
                if c and c.active:
                    cfg.campaigns.append(c)
    return cfg
