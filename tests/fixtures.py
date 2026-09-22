"""Generated test fixtures — no binary files committed, no downloads needed.

Everything here is synthesised with ffmpeg on first use and cached under
tests/fixtures/ (gitignored), so the suite runs on a fresh clone.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"
TRUE_BPM = 120.0


def _ff() -> str:
    from clipper.binaries import resolve
    return resolve("ffmpeg")


def click_track(bpm: float = TRUE_BPM, seconds: int = 30) -> Path:
    """Audio click at a known tempo — ground truth for the beat tracker."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / f"click_{int(bpm)}bpm.wav"
    if out.exists():
        return out
    period = 60.0 / bpm
    expr = (f"sin(2*PI*1000*t)*exp(-24*mod(t\\,{period:.6f}))"
            f"*lt(mod(t\\,{period:.6f})\\,0.05)")
    subprocess.run([_ff(), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"aevalsrc='{expr}':s=22050:d={seconds}",
                    "-ac", "1", str(out)], check=True)
    return out


def music_video(seconds: int = 30) -> Path:
    """A click track with moving picture — stands in for a downloaded source."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / f"track_{seconds}s.mp4"
    if out.exists():
        return out
    a = click_track(seconds=seconds)
    subprocess.run([_ff(), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}",
                    "-i", str(a), "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


def footage_pool(n: int = 3, seconds: int = 20) -> Path:
    """A directory of visually distinct clips for [EDITS] mode."""
    d = FIXTURES / "footage"
    d.mkdir(parents=True, exist_ok=True)
    patterns = ["testsrc2", "smptebars", "rgbtestsrc"]
    for i in range(n):
        f = d / f"clip{i}.mp4"
        if f.exists():
            continue
        subprocess.run([_ff(), "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"{patterns[i % len(patterns)]}=size=1280x720:rate=30:duration={seconds}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        str(f)], check=True)
    return d


def demo_campaign(tmp: Path):
    """An [EDITS] campaign pointed at the generated footage pool."""
    import yaml
    from clipper.config import load_campaign
    cdir = tmp / "edits-fixture"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "campaign.yaml").write_text(yaml.safe_dump({
        "name": "Fixture", "mode": "edits", "active": True, "rate_per_1k": 0,
        "sources": {"track": str(music_video()), "footage_dir": str(footage_pool())},
        "rules": {"vad_filter": False, "min_clip_seconds": 5, "max_clip_seconds": 58,
                  "platforms": ["tiktok"],
                  "render": {"captions": {"enabled": True, "style": "boxed"},
                             "hook_title": {"enabled": False},
                             "edit": {"beats_per_shot": 4, "punch": 0.08,
                                      "transition": "auto", "transition_duration": 0.22,
                                      "effects": {}}}},
        "highlight_brief": "fixture",
    }, sort_keys=False))
    return load_campaign(cdir)


def clip_campaign(tmp: Path):
    """A `mode: clip` campaign that overrides render settings.

    Mirrors how a real clipping campaign is configured — captions off, a single
    hook line held for the whole clip — so the per-campaign override path is
    exercised, not just the defaults in config.yaml.
    """
    import yaml
    from clipper.config import load_campaign
    cdir = tmp / "clip-fixture"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "campaign.yaml").write_text(yaml.safe_dump({
        "name": "Clip Fixture", "mode": "clip", "active": True, "rate_per_1k": 0,
        "sources": {"youtube": [], "footage_dir": "footage"},
        "rules": {"min_clip_seconds": 5, "max_clip_seconds": 58,
                  "platforms": ["tiktok"],
                  "render": {"captions": {"enabled": False},
                             "hook_title": {"enabled": True, "hold_seconds": "full",
                                            "uppercase": False, "style": "clean"}}},
        "highlight_brief": "fixture",
    }, sort_keys=False))
    return load_campaign(cdir)
