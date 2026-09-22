"""Beat grid for a music track — pure numpy, no librosa.

An [EDITS] post is judged on whether the cuts land with the song, so the editor
needs to know where the beats are. This decodes the track to mono PCM with
ffmpeg and runs the standard three steps:

    1. spectral flux    — per-frame sum of positive magnitude change; percussive
                          onsets show up as spikes
    2. tempo            — autocorrelate the onset envelope, take the strongest
                          lag inside a plausible BPM range
    3. phase            — slide a grid of that period over the envelope and keep
                          the offset with the most onset energy under it

Good enough for cutting: on a 4-on-the-floor pop track it lands within a frame
or two. It is not a general-purpose beat tracker — it assumes roughly constant
tempo, which is what these campaign tracks are.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from .binaries import resolve

log = logging.getLogger("beats")

SR = 22050
HOP = 512
WIN = 1024
BPM_MIN, BPM_MAX = 70.0, 180.0


def _decode_mono(path: Path, start: float, dur: float, cfg=None) -> np.ndarray:
    """Decode [start, start+dur) to mono float32 in [-1, 1]."""
    cmd = [
        resolve("ffmpeg", cfg), "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
        "-vn", "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SR), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def onset_envelope(x: np.ndarray) -> np.ndarray:
    """Spectral flux, one value per HOP samples."""
    if x.size < WIN:
        return np.zeros(0, dtype=np.float32)
    n = 1 + (x.size - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(WIN).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    # log compression keeps loud sections from dominating the flux
    mag = np.log1p(mag * 10.0)
    flux = np.maximum(mag[1:] - mag[:-1], 0.0).sum(axis=1)
    if flux.size == 0:
        return flux
    # subtract a local mean so the envelope is centred and quiet passages still peak
    k = 16
    pad = np.pad(flux, (k, k), mode="edge")
    local = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    env = np.maximum(flux - local, 0.0)
    peak = env.max()
    return env / peak if peak > 0 else env


def _tempo_period(env: np.ndarray) -> float | None:
    """Dominant inter-beat period, in envelope frames."""
    fps = SR / HOP
    lo, hi = int(fps * 60.0 / BPM_MAX), int(fps * 60.0 / BPM_MIN)
    if env.size < hi * 2 or hi <= lo:
        return None
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[e.size - 1:]
    window = ac[lo:hi + 1]
    if window.size == 0 or not np.isfinite(window).any():
        return None
    return float(lo + int(np.argmax(window)))


def beat_times(path: Path, start: float, dur: float, cfg=None) -> list[float]:
    """Beat offsets in seconds, relative to `start`. Empty list if undetectable."""
    try:
        x = _decode_mono(path, start, dur, cfg)
    except subprocess.CalledProcessError as e:
        log.warning("could not decode %s for beat tracking: %s", path.name, e)
        return []
    env = onset_envelope(x)
    period = _tempo_period(env)
    if period is None or period <= 0:
        log.warning("no tempo found in %s [%.1f,+%.1f]", path.name, start, dur)
        return []

    # phase: try every offset within one period, keep the grid with most energy
    grid = np.arange(0, env.size, period)
    best_off, best_score = 0.0, -1.0
    for off in np.arange(0.0, period, 1.0):
        idx = np.clip((grid + off).astype(int), 0, env.size - 1)
        score = float(env[idx].sum())
        if score > best_score:
            best_off, best_score = float(off), score

    fps = SR / HOP
    times = [float((g + best_off) / fps) for g in grid]
    times = [t for t in times if 0.0 <= t <= dur]
    log.info("%s [%.1f,+%.1fs] → %.1f BPM, %d beats",
             path.name, start, dur, 60.0 * fps / period, len(times))
    return times


RAMP_FROM, RAMP_TO = 8, 1


def shot_boundaries(beats: list[float], clip_len: float, beats_per_shot,
                    min_shot: float = 0.45) -> list[float]:
    """Cut points (including 0.0 and clip_len).

    `beats_per_shot` is either an int (cut every Nth beat) or "ramp", which
    accelerates from RAMP_FROM beats per shot down to RAMP_TO across the clip.
    The reference posts Clipster ships do exactly this — ~3.8s shots through the
    verse, ~0.5s by the end — and a uniform grid reads as flat next to them.

    Falls back to an even split when there is no usable beat grid, so an edit is
    still produced for a track the tracker cannot read.
    """
    ramp = isinstance(beats_per_shot, str) and beats_per_shot.lower() == "ramp"
    if not beats or (not ramp and int(beats_per_shot) <= 0):
        n = max(int(clip_len // 2.0), 1)
        return [round(clip_len * i / n, 3) for i in range(n + 1)]

    cuts = [0.0]
    if ramp:
        i = 0
        while i < len(beats):
            t = beats[i]
            if t - cuts[-1] >= min_shot and clip_len - t >= min_shot:
                cuts.append(round(t, 3))
            prog = min(t / clip_len, 1.0) if clip_len > 0 else 1.0
            i += max(RAMP_TO, int(round(RAMP_FROM - (RAMP_FROM - RAMP_TO) * prog)))
    else:
        for t in beats[::int(beats_per_shot)]:
            if t - cuts[-1] >= min_shot and clip_len - t >= min_shot:
                cuts.append(round(t, 3))
    cuts.append(round(clip_len, 3))
    return cuts
