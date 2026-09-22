"""End-to-end smoke test with a synthetic transcript and a mocked LLM.
Exercises: config → ledger → ingest(footage) → highlight parsing → ASS captions
→ ffmpeg render → post queue → submissions export. No network, no API keys.

    python -m pytest tests/ -q     (or)    python tests/test_pipeline.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clipper import highlight, ingest, render, submissions  # noqa: E402
from clipper.binaries import resolve  # noqa: E402
from clipper.config import load_config  # noqa: E402
from clipper.ledger import Ledger  # noqa: E402


def fake_transcript(duration: float = 90.0) -> dict:
    """~2 words/sec of filler, one segment per 5 s."""
    words_pool = "so here is the thing nobody tells you about building apps at night".split()
    segs, t, i = [], 0.0, 0
    while t < duration:
        ws, tt = [], t
        for _ in range(9):
            w = words_pool[i % len(words_pool)]; i += 1
            ws.append({"word": w, "start": round(tt, 2), "end": round(tt + 0.45, 2)})
            tt += 0.55
        segs.append({"start": round(t, 2), "end": round(min(tt, duration), 2),
                     "text": " ".join(w["word"] for w in ws), "words": ws})
        t += 5.0
    return {"language": "en", "duration": duration, "segments": segs}


def fake_llm(system: str, user: str) -> str:
    # segments are 5 s each → segs 2..7 = 30 s clip, segs 10..17 = 40 s clip, one junk item, one too short
    return json.dumps([
        {"start_seg": 2, "end_seg": 7, "score": 9, "hook": "Nobody tells you this", "title": "The thing nobody tells you",
         "caption": "Wild take on building at night.", "why": "hook"},
        {"start_seg": 10, "end_seg": 17, "score": 8, "hook": "Build apps at night?", "title": "Night builder",
         "caption": "Second moment.", "why": "payoff"},
        {"start_seg": 3, "end_seg": 6, "score": 10, "hook": "overlap", "title": "dup", "caption": "", "why": ""},  # overlaps #1 → dropped
        {"start_seg": 12, "end_seg": 12, "score": 9, "hook": "short", "title": "too short", "caption": "", "why": ""},
    ])


def ensure_sample() -> None:
    """Generate a 90 s synthetic 16:9 video (test pattern + tone) if it's not there."""
    import subprocess
    dst = ROOT / "campaigns" / "_test-campaign" / "footage" / "sample.mp4"
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([resolve("ffmpeg"), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=90",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=90",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(dst)], check=True)


def main() -> None:
    ensure_sample()
    cfg = load_config(ROOT)
    # Isolate the run. Previously this pointed at the live data_dir and deleted
    # data/ledger.db outright, and it asserted on a campaigns list that any second
    # active campaign would change. Both are the user's real state — the suite
    # gets its own scratch dir and only ever sees _test-campaign.
    tmp = Path(tempfile.mkdtemp(prefix="clipper-test-"))
    cfg.data_dir = tmp
    for sub in ("downloads", "transcripts", "renders", "logs"):
        (tmp / sub).mkdir(parents=True, exist_ok=True)
    cfg.raw.setdefault("submissions", {})["export_csv"] = str(tmp / "submissions.csv")
    # pin the clip count: config.yaml ships clips_per_source: 1 and the operator
    # is expected to override it, so the fixture must not inherit whatever is set
    cfg.raw.setdefault("limits", {})["clips_per_source"] = 4
    # pin the publish targets too: config.yaml is the operator's, and disabling a
    # platform there should not silently change what this fixture asserts
    pub = cfg.raw.setdefault("publish", {})
    for plat in ("youtube", "instagram", "tiktok"):
        pub.setdefault(plat, {})["enabled"] = True
    cfg.campaigns = [c for c in cfg.campaigns if c.slug == "_test-campaign"]
    assert cfg.campaigns, "campaigns/_test-campaign missing or inactive"
    ledger = Ledger(tmp / "ledger.db")

    # 1. ingest local footage
    ingest.run(cfg, ledger)
    srcs = ledger.sources_with_status("downloaded")
    assert len(srcs) == 1, ledger.summary()
    sid = srcs[0]["id"]
    print("ingest ok:", sid, f"{srcs[0]['duration_s']:.0f}s")

    # 2. fake transcript (skip whisper)
    (cfg.data_dir / "transcripts" / f"{sid}.json").write_text(json.dumps(fake_transcript(srcs[0]["duration_s"])))
    ledger.update_source(sid, status="transcribed")

    # 3. highlight with mocked LLM
    n = highlight.run(cfg, ledger, llm=fake_llm)
    clips = ledger.clips_with_status("planned")
    assert n == 2 and len(clips) == 2, (n, [dict(c) for c in clips])
    for c in clips:
        d = c["end_s"] - c["start_s"]
        assert 18 <= d <= 58, d
    print("highlight ok:", [(c["id"], round(c["end_s"] - c["start_s"], 1), c["hook"]) for c in clips])

    # 4. render
    n = render.run(cfg, ledger)
    rendered = ledger.clips_with_status("rendered")
    assert n == 2 and len(rendered) == 2, ledger.summary()
    for c in rendered:
        p = Path(c["render_path"])
        assert p.exists() and p.stat().st_size > 50_000, p
        info = render.probe(p)
        assert (info["w"], info["h"]) == (1080, 1920), info
        ass = p.with_suffix(".ass").read_text()
        assert "Dialogue:" in ass and "@testpod" in ass and "NOBODY" in ass.upper()
    print("render ok:", [Path(c["render_path"]).name for c in rendered])

    # 5. post queue: 2 clips × 3 platforms
    q = ledger.conn.execute("SELECT platform, COUNT(*) FROM posts GROUP BY platform").fetchall()
    assert {r[0]: r[1] for r in q} == {"youtube": 2, "instagram": 2, "tiktok": 2}, q
    print("queue ok:", dict(q))

    # 6. simulate a successful post and export submissions
    pid = ledger.queued_posts("youtube")[0]["id"]
    ledger.mark_post(pid, "posted", remote_id="abc123", url="https://youtube.com/shorts/abc123")
    assert ledger.posts_today("youtube") == 1
    assert submissions.run(cfg, ledger) == 1
    csv_path = tmp / "submissions.csv"
    assert "abc123" in csv_path.read_text()
    assert submissions.run(cfg, ledger) == 0  # idempotent
    print("export ok:", csv_path.name)
    print("SUMMARY", json.dumps(ledger.summary()))
    ledger.conn.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL OK")


if __name__ == "__main__":
    main()
    print("ALL OK")
