"""Stage 3 — pick clip-worthy moments from the transcript with an LLM.

The transcript is chunked into ~10-minute windows (with segment indices), each
window is sent to an LLM (a local Ollama model by default, no key and no network)
along with the campaign's highlight brief, and the model returns
candidate clips as JSON. Candidates are snapped to sentence boundaries,
de-duplicated, scored, and the top N per source are written to the ledger.
"""
from __future__ import annotations

import json
import logging
import os
import re

import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import Campaign, Config
from .ledger import Ledger
from .transcribe import load_transcript

log = logging.getLogger("highlight")

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

SYSTEM = """You are an elite short-form video editor who finds viral clips inside long videos.
You will receive a transcript window as numbered segments with timestamps, plus a brief
describing the audience. Return ONLY a JSON array (no prose, no markdown fences) of the best
self-contained moments in this window. Each item:
{
  "start_seg": <int first segment index>,
  "end_seg": <int last segment index (inclusive)>,
  "score": <int 1-10 how likely this goes viral as a vertical clip>,
  "hook": "<max 8 words on-screen hook, punchy, no clickbait lies>",
  "title": "<max 70 char video title>",
  "caption": "<1-2 sentence post caption, no hashtags>",
  "why": "<one line>"
}
Rules: clips must make sense with zero context, start on a strong line, end on a payoff.
Target %(min)d-%(max)d seconds. Skip intros, ad reads, housekeeping, and anything requiring
the previous 10 minutes to understand. Prefer fewer great clips over many mediocre ones.
Return [] if nothing qualifies."""


def _windows(segments: list[dict], window_s: float):
    """Yield (start_index, end_index_exclusive) covering the transcript."""
    i = 0
    while i < len(segments):
        t0 = segments[i]["start"]
        j = i
        while j < len(segments) and segments[j]["end"] - t0 <= window_s:
            j += 1
        yield i, max(j, i + 1)
        i = max(j, i + 1)


def _trim_to_words(seg: dict, start: float, max_s: float) -> float:
    """End time no more than `max_s` after `start`, landing on a word boundary.

    Used when one transcript segment is longer than the maximum clip length —
    without this the candidate is discarded and a source made of few long
    segments produces no clips at all.
    """
    limit = start + max_s
    best = None
    for w in seg.get("words") or []:
        if w["end"] <= limit:
            best = w["end"]
        else:
            break
    return best if best is not None else limit


def _fmt_window(segments: list[dict], i0: int, i1: int) -> str:
    lines = []
    for k in range(i0, i1):
        s = segments[k]
        lines.append(f"[{k}] {s['start']:.1f}-{s['end']:.1f}: {s['text']}")
    return "\n".join(lines)


def _call_llm(cfg: Config, system: str, user: str) -> str:
    # env wins over config so a personal setup can differ from what the repo ships
    provider = str(os.environ.get("CLIPPER_LLM_PROVIDER")
                   or cfg.get("highlighter", "provider", default="ollama")).lower()
    model = (os.environ.get("CLIPPER_LLM_MODEL")
             or cfg.get("highlighter", "model", default=DEFAULT_MODEL))
    max_tokens = int(cfg.get("highlighter", "max_tokens", default=2000))

    if provider == "ollama":
        base = (os.environ.get("OLLAMA_HOST")
                or cfg.get("highlighter", "base_url", default=OLLAMA_URL)).rstrip("/")
        try:
            r = requests.post(
                f"{base}/api/chat",
                json={"model": model, "stream": False,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "options": {"temperature": 0.4, "num_predict": max_tokens}},
                timeout=float(cfg.get("highlighter", "timeout_s", default=300)),
            )
        except requests.RequestException as e:
            raise RuntimeError(
                f"cannot reach Ollama at {base} ({e}). Install it from https://ollama.com, "
                f"then run:  ollama pull {model}"
            ) from e
        if r.status_code == 404:
            raise RuntimeError(f"Ollama has no model {model!r} — run:  ollama pull {model}")
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    if provider == "anthropic":
        try:
            import anthropic  # lazy: an optional extra, not a core dependency
        except ImportError as e:
            raise RuntimeError(
                "provider 'anthropic' needs the anthropic package, which is an optional "
                "extra. Install it with:  uv sync --extra anthropic   (or --extra all). "
                "The default provider is local Ollama and needs nothing."
            ) from e

        key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or cfg.get(
            "highlighter", "base_url", default=None)
        if not key and not base_url:
            raise RuntimeError(
                "provider 'anthropic' needs ANTHROPIC_API_KEY in .env, or "
                "highlighter.base_url pointing at a gateway of your own")
        client = anthropic.Anthropic(base_url=base_url, api_key=key or "not-required")
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content)

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("provider 'openai' needs OPENAI_API_KEY in .env")
        base = (os.environ.get("OPENAI_BASE_URL")
                or cfg.get("highlighter", "base_url", default="https://api.openai.com/v1")).rstrip("/")
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=float(cfg.get("highlighter", "timeout_s", default=300)),
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    raise RuntimeError(
        f"unknown highlighter.provider {provider!r} — use ollama, anthropic or openai")


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else []
    except json.JSONDecodeError:
        return []


def pick_highlights(cfg: Config, camp: Campaign, transcript: dict, llm=None) -> list[dict[str, Any]]:
    """Return list of clips: {start,end,score,hook,title,caption}. `llm` is injectable for tests."""
    llm = llm or (lambda s, u: _call_llm(cfg, s, u))
    segs = transcript["segments"]
    lim = cfg.get("limits", default={}) or {}
    min_s = max(int(lim.get("min_clip_seconds", 18)), int(camp.rules.get("min_clip_seconds", 0) or 0))
    max_s = min(int(lim.get("max_clip_seconds", 58)), int(camp.rules.get("max_clip_seconds", 60) or 60))
    window_s = float(cfg.get("highlighter", "window_seconds", default=600))
    min_score = float(cfg.get("highlighter", "min_score", default=7))
    banned = [b.lower() for b in camp.rules.get("banned_words", []) or []]

    system = SYSTEM % {"min": min_s, "max": max_s}
    passed: list[dict] = []   # cleared min_score
    below: list[dict] = []    # valid but under the threshold, kept as fallback
    windows = list(_windows(segs, window_s))
    failures: list[str] = []

    def _ask(win: tuple[int, int]) -> str:
        i0, i1 = win
        user = (f"AUDIENCE BRIEF:\n{camp.highlight_brief or 'General audience.'}\n\n"
                f"TRANSCRIPT WINDOW:\n{_fmt_window(segs, i0, i1)}")
        try:
            return llm(system, user)
        except Exception as e:  # noqa: BLE001
            # one line, not a traceback per window — the same cause repeats for each
            failures.append(str(e))
            log.error("window %d-%d: %s", i0, i1, str(e).splitlines()[0][:200])
            return ""

    # The windows are independent, and each call is minutes of waiting on the
    # network — an hour-long source is six of them back to back. Run them at once
    # and the stage costs about as long as its slowest window instead of their sum.
    workers = int(cfg.get("highlighter", "max_parallel", default=4))
    if len(windows) > 1 and workers > 1:
        log.info("  %d windows, %d at a time", len(windows), min(workers, len(windows)))
        with ThreadPoolExecutor(max_workers=min(workers, len(windows))) as ex:
            raws = list(ex.map(_ask, windows))
    else:
        raws = [_ask(w) for w in windows]

    if failures and len(failures) == len(windows):
        # every window failed for the same reason; returning [] would look like
        # "nothing was worth clipping" when the model was never reached at all
        raise RuntimeError(failures[0])

    for wi, ((i0, i1), raw) in enumerate(zip(windows, raws), 1):
        log.info("  window %d/%d (%.0f–%.0fs) → %d candidate(s)",
                 wi, len(windows), segs[i0]["start"], segs[i1 - 1]["end"],
                 len(_parse_json_array(raw)))
        for item in _parse_json_array(raw):
            try:
                a, b = int(item["start_seg"]), int(item["end_seg"])
                a, b = max(a, 0), min(b, len(segs) - 1)
                if b < a:
                    continue
                start, end = segs[a]["start"], segs[b]["end"]
                dur = end - start
                # trim/extend to bounds by dropping/adding whole segments
                while dur > max_s and b > a:
                    b -= 1; end = segs[b]["end"]; dur = end - start
                if dur > max_s:
                    # A single segment longer than max_clip_seconds cannot be
                    # trimmed by dropping segments, so the whole source used to
                    # yield nothing. Whisper emits these on music and on long
                    # unbroken speech. Fall back to word timestamps and cut on a
                    # word boundary inside the segment.
                    end = _trim_to_words(segs[b], start, max_s)
                    dur = end - start
                if dur < min_s or dur > max_s:
                    continue
                text = " ".join(segs[k]["text"] for k in range(a, b + 1)).lower()
                if any(w in text for w in banned):
                    continue
                score = float(item.get("score", 0))
                (passed if score >= min_score else below).append({
                    "start": round(start - 0.15, 2) if start > 0.15 else 0.0,  # tiny lead-in
                    "end": round(end + 0.25, 2),
                    "score": score,
                    "hook": str(item.get("hook", ""))[:60].strip(),
                    "title": str(item.get("title", ""))[:90].strip(),
                    "caption": str(item.get("caption", ""))[:400].strip(),
                    "why": str(item.get("why", ""))[:200],
                })
            except (KeyError, ValueError, TypeError):
                continue

    # de-dup overlapping candidates, keep highest score
    def _pack(pool: list[dict], into: list[dict]) -> None:
        for c in sorted(pool, key=lambda c: -c["score"]):
            if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in into):
                into.append(c)

    per_source = int(lim.get("clips_per_source", 1))
    chosen: list[dict] = []
    _pack(passed, chosen)
    # Asking for N clips should return N. The score threshold is a quality
    # preference, not a hard gate — if too few candidates clear it, top up with
    # the best of the rest rather than handing back an empty run.
    if len(chosen) < per_source and below:
        before = len(chosen)
        _pack(below, chosen)
        if len(chosen) > before:
            log.info("only %d clip(s) scored >= %g; topped up to %d with the next best",
                     before, min_score, min(len(chosen), per_source))
    return chosen[:per_source]


def pick_without_speech(cfg: Config, camp: Campaign, video: Path, duration: float,
                        strategy: str = "scenes") -> list[dict[str, Any]]:
    """Choose clips without a transcript at all.

    For music, ambience, or anything where the words are not the point — and for
    sources whose transcript comes back as noise. `scenes` starts each clip on a
    detected shot change so cuts land where the picture already changes;
    `even` just spaces them across the runtime.
    """
    lim = cfg.get("limits", default={}) or {}
    min_s = max(int(lim.get("min_clip_seconds", 18)),
                int(camp.rules.get("min_clip_seconds", 0) or 0))
    max_s = min(int(lim.get("max_clip_seconds", 58)),
                int(camp.rules.get("max_clip_seconds", 60) or 60))
    want = int(lim.get("clips_per_source", 1))
    target = (min_s + max_s) / 2.0

    starts: list[float] = []
    if strategy == "scenes":
        from .render import scene_cuts
        cuts = [0.0] + scene_cuts(video, 0.0, duration, cfg=cfg)
        # keep cuts far enough apart that consecutive clips do not overlap
        for c in cuts:
            if c + min_s <= duration and (not starts or c - starts[-1] >= target):
                starts.append(c)
        if len(starts) < want:
            log.info("only %d usable scene cut(s) — spacing the rest evenly", len(starts))
    if len(starts) < want:
        room = max(duration - target, 0.0)
        n = max(want, 1)
        starts = sorted({round(room * i / max(n - 1, 1), 2) for i in range(n)})

    clips = []
    for st in starts[:want]:
        en = min(st + target, duration)
        if en - st < min_s:
            continue
        clips.append({"start": round(st, 2), "end": round(en, 2), "score": 5.0,
                      "hook": "", "title": camp.name or "", "caption": "",
                      "why": f"{strategy} pick (no transcript)"})
    log.info("picked %d clip(s) by %s, no speech analysis", len(clips), strategy)
    return clips


def run(cfg: Config, ledger: Ledger, llm=None) -> int:
    camps = {c.slug: c for c in cfg.campaigns}
    strategy = str(cfg.get("highlighter", "strategy", default="llm")).lower()
    n = 0
    # without speech analysis a source is ready as soon as it is downloaded
    ready = (ledger.sources_with_status("transcribed") if strategy == "llm"
             else list(ledger.sources_with_status("transcribed"))
                  + list(ledger.sources_with_status("downloaded")))
    for s in ready:
        camp = camps.get(s["campaign"])
        if not camp:
            ledger.update_source(s["id"], status="failed", error="campaign inactive")
            continue
        try:
            if strategy == "llm":
                clips = pick_highlights(cfg, camp, load_transcript(cfg, s["id"]), llm=llm)
            else:
                clips = pick_without_speech(cfg, camp, Path(s["video_path"]),
                                            float(s["duration_s"] or 0), strategy)
            base = ledger.clips_for_source(s["id"])
            for k, c in enumerate(clips):
                cid = f"{s['id']}_{base + k:02d}"
                ledger.add_clip(cid, s["id"], camp.slug, c["start"], c["end"], c["score"],
                                c["hook"], c["title"], c["caption"], meta={"why": c["why"]})
                n += 1
            ledger.update_source(s["id"], status="highlighted")
            log.info("%s → %d clip(s)", s["id"], len(clips))
        except Exception as e:  # noqa: BLE001
            log.exception("highlighting failed for %s", s["id"])
            ledger.update_source(s["id"], status="failed", error=str(e)[:500])
    return n
