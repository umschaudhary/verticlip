"""Stage 2 — transcribe downloaded sources with word-level timestamps.

Uses faster-whisper (CTranslate2) — runs locally, free, ~10x realtime for the
`small` model on an M-series Mac. Output: data/transcripts/<sid>.json
    {"language": "en", "segments": [{"start","end","text","words":[{"word","start","end"}]}]}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Config
from .ledger import Ledger

log = logging.getLogger("transcribe")
_model = None


def _get_model(cfg: Config):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # lazy import: heavy

        t = cfg.get("transcription", default={}) or {}
        _model = WhisperModel(
            t.get("model", "small"),
            device=t.get("device", "auto"),
            compute_type=t.get("compute_type", "int8"),
        )
    return _model


def transcribe_file(cfg: Config, video: Path, out: Path, vad: bool | None = None) -> dict:
    model = _get_model(cfg)
    lang = cfg.get("transcription", "language")
    if vad is None:
        vad = bool(cfg.get("transcription", "vad_filter", default=True))
    kw = {"vad_parameters": {"min_silence_duration_ms": 400}} if vad else {}
    segments, info = model.transcribe(
        str(video), language=lang, word_timestamps=True, vad_filter=vad, **kw,
    )
    data = {"language": info.language, "duration": info.duration, "segments": []}
    total = float(info.duration or 0)
    next_mark = 10.0
    for seg in segments:
        # faster-whisper yields lazily, so segment end time is a real progress bar
        if total:
            pct = 100.0 * seg.end / total
            if pct >= next_mark:
                log.info("  transcribed %d%%  (%.0fs of %.0fs)", int(pct), seg.end, total)
                next_mark = (int(pct // 10) + 1) * 10.0
        data["segments"].append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": [
                {"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in (seg.words or [])
            ],
        })
    out.write_text(json.dumps(data, ensure_ascii=False))
    return data


def run(cfg: Config, ledger: Ledger) -> int:
    tdir = cfg.data_dir / "transcripts"
    camps = {c.slug: c for c in cfg.campaigns}
    # Transcription exists to serve two consumers: the LLM picker and burned-in
    # captions. If neither wants words, it is minutes of work for nothing.
    strategy = str(cfg.get("highlighter", "strategy", default="llm")).lower()
    caps_on = bool(cfg.get("render", "captions", "enabled", default=True))
    if strategy != "llm" and not caps_on:
        log.info("skipping transcription — picking by %s and captions are off", strategy)
        return 0
    n = 0
    for s in ledger.sources_with_status("downloaded"):
        out = tdir / f"{s['id']}.json"
        # Whisper's VAD is tuned for speech and throws away sung vocals over a
        # backing track — a music source comes back with zero segments. Music
        # campaigns set `vad_filter: false` in their rules to transcribe lyrics.
        camp = camps.get(s["campaign"])
        vad = None
        if camp is not None and "vad_filter" in camp.rules:
            vad = bool(camp.rules["vad_filter"])
        try:
            if not out.exists():
                log.info("transcribing %s (vad=%s) …", s["id"],
                         vad if vad is not None else cfg.get("transcription", "vad_filter", default=True))
                transcribe_file(cfg, Path(s["video_path"]), out, vad=vad)
            ledger.update_source(s["id"], status="transcribed")
            n += 1
        except Exception as e:  # noqa: BLE001
            log.exception("transcription failed for %s", s["id"])
            ledger.update_source(s["id"], status="failed", error=str(e)[:500])
    return n


def load_transcript(cfg: Config, sid: str) -> dict:
    """Transcript for a source, or an empty one when it was never transcribed."""
    f = cfg.data_dir / "transcripts" / f"{sid}.json"
    if not f.exists():
        return {"language": None, "duration": 0.0, "segments": []}
    return json.loads(f.read_text())
