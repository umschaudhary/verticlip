"""Stage 4 — cut, reframe to 9:16, burn animated word-by-word captions, hook
title and campaign credit, and encode a platform-ready MP4.

Everything is a single ffmpeg pass:
    crop/scale → ASS subtitles (captions + hook + credit) → h264/aac
Face-aware cropping ("smart") samples frames with OpenCV's Haar cascade and
centers the crop on the median face position; falls back to center crop.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from .config import Campaign, Config
from .binaries import resolve
from .ledger import Ledger
from .transcribe import load_transcript
from .beats import beat_times, shot_boundaries

log = logging.getLogger("render")


# ── probing ───────────────────────────────────────────────────

def probe(path: Path) -> dict:
    out = subprocess.run(
        [resolve("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    return {"w": int(st["width"]), "h": int(st["height"])}


# ── smart crop ────────────────────────────────────────────────

def scene_cuts(video: Path, start: float, end: float, threshold: float = 0.30,
               cfg=None) -> list[float]:
    """Shot-change times inside [start, end), relative to `start`.

    Uses ffmpeg's own scene score rather than PySceneDetect: it decodes in C at
    a fraction of the cost, needs no extra dependency, and this only has to find
    hard cuts — the crop must not glide across one, and that is all the precision
    a step boundary needs.
    """
    cmd = [resolve("ffmpeg", cfg), "-v", "error", "-ss", f"{start:.3f}",
           "-to", f"{end:.3f}", "-i", str(video),
           "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-",
           "-an", "-f", "null", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return []
    cuts = []
    for line in out.splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            t = float(m.group(1))
            if 0.25 < t < (end - start) - 0.25:      # ignore cuts at the very edges
                cuts.append(round(t, 3))
    if cuts:
        log.info("%d scene cut(s) inside the clip: %s", len(cuts),
                 ", ".join(f"{c:.1f}s" for c in cuts))
    return cuts


def _segment_track(pts, cuts: list[float]):
    """Split face samples into per-shot runs so nothing is smoothed across a cut."""
    if not cuts:
        return [pts] if pts else []
    segs, i = [], 0
    bounds = list(cuts) + [float("inf")]
    cur = []
    for t, x in pts:
        while t >= bounds[i]:
            i += 1
            if cur:
                segs.append(cur)
            cur = []
        cur.append((t, x))
    if cur:
        segs.append(cur)
    return segs


def _face_track(video: Path, start: float, end: float, fps: float = 2.0):
    """Sample face-centre x over the clip. Returns [(t_rel, cx), ...] or [].

    Haar cascades false-positive freely on textured backgrounds — grass, foliage,
    cloud — so detections are filtered three ways before they are trusted:
    a minimum face size relative to the frame, a high `minNeighbors`, and a
    median-absolute-deviation pass that drops points far from the run of the
    others. A clip whose detections never settle returns [] and gets a centre crop,
    which is wrong less often than a crop anchored to a patch of sky.
    """
    try:
        import cv2  # lazy
    except ImportError:
        return []
    if not hasattr(cv2, "CascadeClassifier"):
        log.warning("cv2 has no CascadeClassifier (opencv %s) — centre crop",
                    getattr(cv2, "__version__", "?"))
        return []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    import numpy as np
    dur = max(end - start, 0.1)
    n = max(int(dur * fps), 4)
    pts = []
    fh = None
    for i in range(n):
        t = start + dur * i / (n - 1 if n > 1 else 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        if fh is None:
            fh = frame.shape[0]
        faces = _faces_in_frame(frame, cascade, fh)
        if faces:
            # the biggest face is the subject; bystanders are further away
            pts.append((t - start, max(faces, key=lambda f: f[1])[0]))
    cap.release()
    if len(pts) < max(3, n // 5):                    # too sparse to trust
        return []

    xs = np.array([p[1] for p in pts])
    med = float(np.median(xs))
    mad = float(np.median(np.abs(xs - med))) or 1.0
    keep = [p for p in pts if abs(p[1] - med) <= 4.0 * mad]
    if len(keep) < max(3, len(pts) // 2):
        log.info("face detections too scattered (mad=%.0fpx) — centre crop", mad)
        return []
    return keep


_MP_MODEL = Path(__file__).resolve().parent.parent / "assets" / "models" / "blaze_face_short_range.tflite"
_mp_detector = None
_mp_tried = False


def _mediapipe_detector():
    """BlazeFace detector, or None if mediapipe/the model isn't available.

    Preferred over Haar: Haar only fires on frontal, upright, evenly-lit faces,
    so it both misses people (killing split detection) and false-positives on
    foliage and cloud (dragging the crop onto grass). Cached — construction is
    the expensive part, inference is not.
    """
    global _mp_detector, _mp_tried
    if _mp_tried:
        return _mp_detector
    _mp_tried = True
    if not _MP_MODEL.exists():
        log.info("no BlazeFace model at %s — falling back to Haar cascades", _MP_MODEL)
        return None
    if os.environ.get("CLIPPER_MEDIAPIPE", "").lower() not in ("1", "true", "yes"):
        return None
    # mediapipe fails by calling abort(), not by raising — on this machine 1.0.1
    # dies in DrishtiMetalHelper during graph init, GPU or CPU delegate alike, and
    # a fatal abort inside a render takes the whole pass down. So prove it survives
    # in a throwaway process before trusting it in ours.
    probe = subprocess.run(
        [sys.executable, "-c",
         "from mediapipe.tasks import python as p;from mediapipe.tasks.python import vision;"
         f"vision.FaceDetector.create_from_options(vision.FaceDetectorOptions("
         f"base_options=p.BaseOptions(model_asset_path={str(_MP_MODEL)!r})))"],
        capture_output=True)
    if probe.returncode != 0:
        log.warning("mediapipe aborted during a probe (rc=%s) — using Haar cascades",
                    probe.returncode)
        return None
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MP_MODEL)),
            min_detection_confidence=0.4,
        )
        _mp_detector = vision.FaceDetector.create_from_options(opts)
        log.info("face detection: mediapipe BlazeFace")
    except Exception as e:  # noqa: BLE001
        log.warning("mediapipe unavailable (%s) — falling back to Haar cascades", e)
        _mp_detector = None
    return _mp_detector


_profile = None
_profile_tried = False


def _profile_cascade():
    global _profile, _profile_tried
    if not _profile_tried:
        _profile_tried = True
        import cv2
        path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        c = cv2.CascadeClassifier(path)
        _profile = None if c.empty() else c
    return _profile


def _faces_in_frame(frame, cascade, fh: int) -> list[float]:
    """Face-centre x for one BGR frame, via mediapipe when present, else Haar."""
    det = _mediapipe_detector()
    if det is not None:
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = det.detect(img)
        out = []
        for d in res.detections:
            bb = d.bounding_box
            if bb.height >= fh * 0.06:          # ignore specks in the background
                out.append((float(bb.origin_x + bb.width / 2.0), float(bb.height)))
        return out
    import cv2
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    min_side = max(int(fh * 0.10), 48)
    found = list(cascade.detectMultiScale(gray, 1.15, 8, minSize=(min_side, min_side)))
    # Frontal-only Haar misses anyone turned even slightly, which is most of a
    # two-person scene. Sweep the profile cascade over the frame and its mirror
    # (the cascade is trained on one facing direction only), then merge.
    prof = _profile_cascade()
    if prof is not None:
        found += list(prof.detectMultiScale(gray, 1.15, 6, minSize=(min_side, min_side)))
        flipped = cv2.flip(gray, 1)
        w_img = gray.shape[1]
        for x, y, w, h in prof.detectMultiScale(flipped, 1.15, 6, minSize=(min_side, min_side)):
            found.append((w_img - x - w, y, w, h))
    faces = sorted(((float(x + w / 2), float(h)) for x, y, w, h in found), key=lambda f: f[0])
    # merge detections of the same face by the two cascades, keeping the bigger box
    merged: list[tuple[float, float]] = []
    for c, h in faces:
        if merged and c - merged[-1][0] <= min_side:
            if h > merged[-1][1]:
                merged[-1] = (c, h)
        else:
            merged.append((c, h))
    return merged


def _detect_faces(video: Path, start: float, end: float, fps: float = 2.0):
    """Sample every plausible face per frame: [(t_rel, [cx, ...]), ...].

    Same false-positive guards as the single-face path — minimum size relative to
    the frame, high `minNeighbors`, histogram equalisation — but keeps all
    survivors so a two-person scene can be recognised as one.
    """
    try:
        import cv2
    except ImportError:
        return []
    if not hasattr(cv2, "CascadeClassifier"):
        return []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    dur = max(end - start, 0.1)
    n = max(int(dur * fps), 4)
    out, fh = [], None
    for i in range(n):
        t = start + dur * i / (n - 1 if n > 1 else 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        if fh is None:
            fh = frame.shape[0]
        out.append((t - start, _faces_in_frame(frame, cascade, fh)))
    cap.release()
    return out


def split_centres(samples, sw: int, sh: int = 1080, min_frac: float = 0.40,
                  min_sep: float = 0.15, min_face: float = 0.11):
    """Two stable face columns, or None.

    Uses the two LARGEST faces in each frame, not the leftmost and rightmost:
    in a crowd the extremes are bystanders, and cropping to them yields two tiles
    of the same background. Both must also be big enough to be subjects rather
    than onlookers — a real two-hander has two faces close to camera.
    """
    if not samples:
        return None
    import numpy as np
    sep_px = sw * min_sep
    min_h = sh * min_face
    lefts, rights, hits = [], [], 0
    for _, faces in samples:
        if len(faces) < 2:
            continue
        big = sorted(faces, key=lambda f: -f[1])[:2]
        if big[1][1] < min_h:            # second-biggest is a bystander
            continue
        lo, hi = sorted(f[0] for f in big)
        if hi - lo < sep_px:
            continue
        lefts.append(lo)
        rights.append(hi)
        hits += 1
    if hits < max(3, int(len(samples) * min_frac)):
        return None
    l, r = float(np.median(lefts)), float(np.median(rights))
    log.info("split layout: %d/%d frame(s) with two faces, centres %.0f and %.0f",
             hits, len(samples), l, r)
    return l, r


def _split_filter(l_cx: float, r_cx: float, sw: int, sh: int, W: int, H: int, tail_fx: str) -> str:
    """Two tiles stacked into WxH, each cropped around one face.

    Each tile is W x H/2, so its source crop keeps the full frame height and
    height*(W/(H/2)) of the width — a downscale on a 1080p source rather than the
    upscale a full-width 9:16 crop needs, which is why split reads sharper.
    """
    th = H // 2
    cw = min(int(sh * (W / th)) // 2 * 2, sw)
    def x_for(cx):
        return int(min(max(cx - cw / 2, 0), sw - cw)) // 2 * 2
    fx = f"scale={W}:{th}:flags=lanczos,{tail_fx}"
    return (f"split=2[sa][sb];"
            f"[sa]crop={cw}:{sh}:{x_for(l_cx)}:0,{fx}[st];"
            f"[sb]crop={cw}:{sh}:{x_for(r_cx)}:0,{fx}[sd];"
            f"[st][sd]vstack=2")


def _smooth_track(pts, keyframes: int = 10, deadzone: float = 0.0):
    """EMA-smooth the samples, then thin to at most `keyframes` points.

    The EMA stops the crop chasing per-frame detector noise; the deadzone holds
    the crop still while the subject moves less than a few percent of the width,
    which is what separates a deliberate reframe from a drifting one.
    """
    if not pts:
        return []
    alpha = 0.25
    sm, cur = [], pts[0][1]
    for t, x in pts:
        if abs(x - cur) > deadzone:
            cur = alpha * x + (1 - alpha) * cur
        sm.append((t, cur))
    if len(sm) <= keyframes:
        return sm
    step = (len(sm) - 1) / (keyframes - 1)
    return [sm[int(round(i * step))] for i in range(keyframes)]


def _x_expr(segments, crop_w: int, sw: int, clip_len: float) -> str:
    """ffmpeg crop-x expression: smooth inside a shot, a hard step at every cut.

    Written as a flat sum of `between()`-gated terms rather than nested ifs —
    nesting a dozen conditionals hits the expression parser's limits, a sum does
    not. Each segment owns its own time window, so the crop jumps at a shot
    change instead of sliding across it, which is what an editor would do.
    Commas are escaped for the filtergraph.
    """
    lo, hi = 0.0, float(max(sw - crop_w, 0))

    def clamp(cx):
        return min(max(cx - crop_w / 2.0, lo), hi)

    terms = []
    for si, seg in enumerate(segments):
        if not seg:
            continue
        t_lo = 0.0 if si == 0 else seg[0][0]
        t_hi = clip_len + 1000 if si == len(segments) - 1 else segments[si + 1][0][0]
        if len(seg) == 1:
            terms.append(f"between(t\\,{t_lo:.2f}\\,{t_hi:.2f})*{clamp(seg[0][1]):.1f}")
            continue
        for i in range(len(seg) - 1):
            t0, x0 = seg[i]
            t1, x1 = seg[i + 1]
            a, b = clamp(x0), clamp(x1)
            g0 = t_lo if i == 0 else t0
            g1 = t_hi if i == len(seg) - 2 else t1
            span = max(t1 - t0, 1e-3)
            terms.append(f"between(t\\,{g0:.2f}\\,{g1:.2f})*"
                         f"({a:.1f}+({b - a:.1f})*(t-{t0:.2f})/{span:.3f})")
    return "+".join(terms) if terms else f"{clamp(sw / 2):.1f}"


# Where the platform UI sits, as a fraction of frame height. TikTok, Reels and
# Shorts all stack a caption, handle and action rail over the bottom of the video
# and put their own chrome at the top, so anything you care about belongs between.
SAFE_TOP, SAFE_BOTTOM = 0.10, 0.22


def _card_filter(cfg: Config, W: int, H: int, sw: int = 0, sh: int = 0) -> str:
    """Whole frame as an inset panel over a blurred backdrop.

    A 9:16 crop of a 16:9 source discards 44% of the width. This keeps all of it
    and gives the picture breathing room instead — margins at the sides, rounded
    corners, and a vertical position that clears the app's own buttons rather
    than running edge to edge underneath them.
    """
    r = cfg.get("render", default={}) or {}
    card = r.get("card", {}) or {}
    margin = int(W * float(card.get("margin_pct", 6)) / 100) // 2 * 2
    radius = int(card.get("corner_radius", 28))
    blur = int(card.get("backdrop_blur", 40))
    dim = float(card.get("backdrop_dim", 0.45))          # 0 = untouched, 1 = black
    fw = W - 2 * margin
    fill = float(card.get("fill_pct", 0) or 0)

    # How tall the panel ends up is decided by the source's own shape: at full
    # width a 16:9 frame is only 608px, a third of the screen. fill_pct asks for a
    # taller panel, and the only way to get one is to crop the sides — so crop
    # exactly as much as the requested height needs and no more.
    pre = ""
    if fill > 0 and sw and sh:
        want_ar = fw / (H * fill / 100.0)
        if sw / sh > want_ar:
            cw = int(sh * want_ar) // 2 * 2
            pre = f"crop={cw}:{sh}:(iw-{cw})/2:0,"
            log.info("card fills %.0f%% of height — keeping %.0f%% of the source width",
                     fill, 100.0 * cw / sw)
        else:
            log.info("card fills %.0f%% of height — no crop needed", fill)

    # centre of the usable band, not of the frame
    band_top, band_bottom = H * SAFE_TOP, H * (1 - SAFE_BOTTOM)
    y = int((band_top + band_bottom) / 2)

    fg = f"{pre}scale={fw}:-2:flags=lanczos,unsharp=5:5:0.5:3:3:0.2"
    if radius > 0:
        # round the corners by masking alpha: opaque everywhere except outside the
        # quarter-circle at each corner
        fg += (f",format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
               f"a='if(gt(abs(X-W/2),W/2-{radius})*gt(abs(Y-H/2),H/2-{radius}),"
               f"if(lte(hypot({radius}-(W/2-abs(X-W/2)),{radius}-(H/2-abs(Y-H/2))),{radius}),255,0),255)'")
    return (f"split=2[cbg][cfg];"
            f"[cbg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"gblur=sigma={blur},eq=brightness=-{dim:.2f}[cb];"
            f"[cfg]{fg}[cf];"
            f"[cb][cf]overlay=(W-w)/2:{y}-h/2:format=auto,format=yuv420p")


def _crop_filter(cfg: Config, camp: Campaign, video: Path, start: float, end: float, src: dict) -> str:
    W, H = int(cfg.get("render", "width", default=1080)), int(cfg.get("render", "height", default=1920))
    mode = cfg.get("render", "crop_mode", default="smart")
    sw, sh = src["w"], src["h"]
    target_ratio = W / H
    if sw / sh <= target_ratio:  # already vertical-ish: just scale + pad
        return f"scale={W}:-2,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black"
    crop_w = int(sh * target_ratio) // 2 * 2
    if mode == "card":
        return _card_filter(cfg, W, H, sw, sh)
    if mode == "blur_bg":
        return (f"split[bg][fg];[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},boxblur=30:8[bgb];[fg]scale={W}:-2[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2")
    # lanczos + a light unsharp: the crop is always upscaled to reach 1080x1920
    # (2.7x from a 720p source, 1.8x from 1080p), and bicubic makes that visibly soft
    sharpen = "unsharp=5:5:0.6:3:3:0.3"
    tail = f"scale={W}:{H}:flags=lanczos,{sharpen}"
    if mode in ("smart", "auto", "split"):
        samples = _detect_faces(video, start, end)
        pair = split_centres(samples, sw)
        if pair and mode != "smart":
            return _split_filter(pair[0], pair[1], sw, sh, W, H, sharpen)
    if mode in ("smart", "auto"):
        pts = _face_track(video, start, end)
        if pts:
            # Smooth each shot separately. An EMA run straight through a cut drags
            # the crop from where the last shot ended toward where the next begins,
            # which reads as a slow pan nobody asked for.
            segs = [sg for sg in (_smooth_track(seg, keyframes=6, deadzone=sw * 0.03)
                                  for seg in _segment_track(pts, scene_cuts(video, start, end, cfg=cfg)))
                    if sg]
            if segs:
                flat = [x for sg in segs for _, x in sg]
                log.info("tracking crop: %d shot(s), %d keyframe(s), x %.0f..%.0f",
                         len(segs), len(flat), min(flat), max(flat))
                expr = _x_expr(segs, crop_w, sw, end - start)
                return f"crop={crop_w}:{sh}:x='{expr}':y=0,{tail}"
    x = int(min(max(sw / 2 - crop_w / 2, 0), sw - crop_w)) // 2 * 2
    return f"crop={crop_w}:{sh}:{x}:0,{tail}"


# ── ASS subtitles ─────────────────────────────────────────────

def _ass_time(t: float) -> str:
    t = max(t, 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def _inline_color(c: str) -> str:
    """Convert a style colour to the inline-override form.

    Style lines use &HAABBGGRR (alpha first, no trailing &). Inline overrides
    like \\1c want &HBBGGRR& — 6 digits and a trailing ampersand. Feeding the
    8-digit style form straight into \\1c makes libass misread the alpha and the
    text renders invisible.
    """
    h = c.strip().lstrip("&").lstrip("Hh").rstrip("&")
    if len(h) == 8:  # AABBGGRR -> BBGGRR
        h = h[2:]
    return f"&H{h.upper()}&"


CAPTION_PRESETS = ("karaoke", "pop", "boxed", "lyric")
HOOK_PRESETS = ("boxed", "clean", "glow", "none")

# Per-preset font-size multiplier — a single popped word carries far more size
# than a three-word line can.
_PRESET_SCALE = {"karaoke": 1.0, "pop": 1.30, "boxed": 0.95, "lyric": 0.58}


def _cap_karaoke(words, clip_start, clip_len, cx, cy, hi, base, wpl, glow, blur):
    """Whole line stays put; the active word changes colour with an eased fade.

    Deliberately no per-word \\fscx: scaling one word inside a centred line
    changes the line's total width, which shifts every other word sideways. The
    emphasis comes from colour plus an outline glow, which are metric-neutral.
    """
    out = []
    lines = [words[i:i + wpl] for i in range(0, len(words), wpl)]
    for line in lines:
        for wi, w in enumerate(line):
            t0 = w["start"] - clip_start
            t1 = (line[wi + 1]["start"] if wi + 1 < len(line) else line[-1]["end"] + 0.12) - clip_start
            t1 = min(max(t1, t0 + 0.05), clip_len)
            if t0 >= clip_len:
                break
            parts = []
            for wj, x in enumerate(line):
                txt = _esc(x["word"].upper())
                if wj == wi:
                    # ease into the highlight colour over 90 ms instead of snapping
                    parts.append(f"{{\\1c{base}\\t(0,90,0.7,\\1c{hi}\\3c{glow}\\blur{blur})}}{txt}"
                                 f"{{\\1c{base}\\3c&H000000&\\blur0}}")
                else:
                    parts.append(txt)
            fade = "\\fad(70,50)" if wi == 0 else ""
            out.append((t0, t1, f"{{\\an5\\pos({cx},{cy}){fade}}}" + " ".join(parts)))
    return out


def _cap_pop(words, clip_start, clip_len, cx, cy, hi, base, wpl, glow, blur):
    """One word at a time, centred, scaling up as it lands.

    Each word is its own centred Dialogue, so scaling can never reflow a
    neighbour — which is what makes the punchier animation safe here.
    """
    out = []
    for i, w in enumerate(words):
        t0 = w["start"] - clip_start
        nxt = words[i + 1]["start"] - clip_start if i + 1 < len(words) else None
        t1 = nxt if nxt is not None else (w["end"] - clip_start + 0.14)
        t1 = min(max(t1, t0 + 0.06), clip_len)
        if t0 >= clip_len:
            break
        txt = _esc(w["word"].upper())
        out.append((t0, t1,
                    f"{{\\an5\\pos({cx},{cy})\\1c{base}\\3c{glow}"
                    f"\\fscx78\\fscy78\\t(0,80,0.6,\\fscx100\\fscy100)\\fad(18,0)}}{txt}"))
    return out


def _cap_boxed(words, clip_start, clip_len, cx, cy, hi, base, wpl, glow, blur):
    """Line sits on an opaque band (BorderStyle 3) — most legible over busy footage."""
    out = []
    lines = [words[i:i + wpl] for i in range(0, len(words), wpl)]
    for line in lines:
        for wi, w in enumerate(line):
            t0 = w["start"] - clip_start
            t1 = (line[wi + 1]["start"] if wi + 1 < len(line) else line[-1]["end"] + 0.12) - clip_start
            t1 = min(max(t1, t0 + 0.05), clip_len)
            if t0 >= clip_len:
                break
            parts = []
            for wj, x in enumerate(line):
                txt = _esc(x["word"].upper())
                parts.append(f"{{\\1c{base}\\t(0,80,0.7,\\1c{hi})}}{txt}{{\\1c{base}}}"
                             if wj == wi else txt)
            fade = "\\fad(60,40)" if wi == 0 else ""
            out.append((t0, t1, f"{{\\an5\\pos({cx},{cy}){fade}}}" + " ".join(parts)))
    return out


def _cap_lyric(words, clip_start, clip_len, cx, cy, hi, base, wpl, glow, blur):
    """Small lowercase phrase with a soft white bloom, held dead centre.

    Modelled on the reference posts Clipster ships with its music [EDITS]
    campaigns: one quiet 1-3 word fragment, no box, no outline, no per-word
    colour change. The footage carries the clip; the text just tracks the lyric.
    """
    out = []
    lines = [words[i:i + wpl] for i in range(0, len(words), wpl)]
    for line in lines:
        t0 = line[0]["start"] - clip_start
        t1 = min(line[-1]["end"] - clip_start + 0.18, clip_len)
        if t0 >= clip_len or t1 <= t0:
            continue
        txt = _esc(" ".join(x["word"].lower() for x in line))
        out.append((t0, t1, f"{{\\an5\\pos({cx},{cy})\\1c{base}\\3c{glow}"
                            f"\\blur{blur}\\fad(90,110)}}{txt}"))
    return out


_CAP_BUILDERS = {"karaoke": _cap_karaoke, "pop": _cap_pop, "boxed": _cap_boxed,
                 "lyric": _cap_lyric}


def _merge(base: dict, over: dict) -> dict:
    """Deep-merge `over` onto `base`, returning a new dict (neither is mutated)."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_ass(cfg: Config, camp: Campaign, words: list[dict], clip_start: float, clip_len: float,
              hook: str) -> str:
    # A campaign may override any render setting under `rules.render:` — music
    # campaigns typically want lyric captions off and a single reaction line held
    # for the whole clip, which is the opposite of the podcast default.
    r = _merge(cfg.get("render", default={}) or {}, camp.rules.get("render", {}) or {})
    cap = r.get("captions", {}) or {}
    W, H = int(r.get("width", 1080)), int(r.get("height", 1920))
    font = cap.get("font", "Arial")
    preset = str(cap.get("style", "boxed")).lower()
    if preset not in _CAP_BUILDERS:
        log.warning("unknown caption style %r — falling back to 'boxed' (choices: %s)",
                    preset, ", ".join(CAPTION_PRESETS))
        preset = "boxed"
    fs = int(int(cap.get("font_size", 78)) * _PRESET_SCALE[preset])
    hi_style = cap.get("highlight_color", "&H0000E5FF")
    base_style = cap.get("base_color", "&H00FFFFFF")
    # builders emit override tags, which need the 6-digit inline form
    hi = _inline_color(hi_style)
    base = _inline_color(base_style)
    glow = _inline_color(cap.get("glow_color", hi_style))
    glow_blur = float(cap.get("glow_blur", 0.9))
    outline = int(cap.get("outline", 4))
    shadow = int(cap.get("shadow", 3))
    wpl = int(cap.get("words_per_line", 3))
    ypos = int(H * float(cap.get("position_pct", 68)) / 100)
    hook_cfg = r.get("hook_title", {}) or {}
    hook_fs = int(hook_cfg.get("font_size", 84))
    hook_style = str(hook_cfg.get("style", "boxed")).lower()
    if hook_style not in HOOK_PRESETS:
        log.warning("unknown hook style %r — using 'boxed' (choices: %s)",
                    hook_style, ", ".join(HOOK_PRESETS))
        hook_style = "boxed"
    # BorderStyle 3 boxes each wrapped line separately, so a two-line hook gets two
    # ragged black slabs. 'clean' and 'glow' drop the box for an outline instead.
    hk_border, hk_outline, hk_shadow, hk_back = {
        "boxed": (3, 10, 0, "&H99000000"),      # softer than fully opaque
        "clean": (1, 5, 3, "&H00000000"),
        "glow":  (1, 4, 0, "&H00000000"),
    }[hook_style if hook_style != "none" else "clean"]
    # how long the hook stays on screen; "full" (or <= 0) pins it to the whole clip
    hold = hook_cfg.get("hold_seconds", 2.6)
    hook_end = clip_len if (str(hold).lower() == "full" or float(hold or 0) <= 0) else min(float(hold), clip_len)

    # 'boxed' paints an opaque band behind the line instead of an outline;
    # 'lyric' wants only a soft bloom, so a thin outline that \blur can smear.
    if preset == "boxed":
        cap_border, cap_outline, cap_shadow = 3, max(outline, 12), 0
    elif preset == "lyric":
        cap_border, cap_outline, cap_shadow = 1, 3, 0
    else:
        cap_border, cap_outline, cap_shadow = 1, outline, shadow

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{fs},{base_style},{base_style},&H00000000,&HC0000000,-1,0,0,0,100,100,0,0,{cap_border},{cap_outline},{cap_shadow},5,60,60,0,1
Style: Hook,{font},{hook_fs},&H00FFFFFF,&H00FFFFFF,&H00000000,{hk_back},-1,0,0,0,100,100,0,0,{hk_border},{hk_outline},{hk_shadow},8,80,80,{int(H*0.16)},1
Style: Credit,{font},{int(fs*0.48)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,60,60,{int(H*0.09)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    if cap.get("enabled", True) and words:
        for t0, t1, text in _CAP_BUILDERS[preset](words, clip_start, clip_len,
                                                  W // 2, ypos, hi, base, wpl, glow, glow_blur):
            events.append(f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Cap,,0,0,0,,{text}")

    if hook and hook_cfg.get("enabled", True) and hook_style != "none":
        # slide up a touch while fading in, instead of appearing flat
        events.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(hook_end)},Hook,,0,0,0,,"
                      f"{{\\fad(140,220)\\fscx92\\fscy92\\t(0,160,0.6,\\fscx100\\fscy100)"
                      f"{'\\\\blur3' if hook_style == 'glow' else ''}}}"
                      f"{_esc(hook.upper() if hook_cfg.get('uppercase', True) else hook)}")

    credit = camp.credit_overlay or r.get("watermark_text", "")
    if credit:
        events.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(clip_len)},Credit,,0,0,0,,{_esc(credit)}")

    return header + "\n".join(events) + "\n"


# ── edits mode ────────────────────────────────────────────────

def footage_pool(camp: Campaign) -> list[Path]:
    """Visual clips an [EDITS] campaign draws from. Empty list = misconfigured."""
    if not camp.footage_dir or not camp.footage_dir.exists():
        return []
    return sorted(p for p in camp.footage_dir.rglob("*")
                  if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm", ".m4v"})


def pick_footage(camp: Campaign, clip_id: str, clip_len: float) -> tuple[Path, float]:
    """Choose a footage file and an in-point for one clip.

    Deterministic in clip_id so a re-render reproduces the same edit, but spread
    across the pool and across each file's runtime so two posts from the same
    campaign never open on the same frame — Clipster's rules disqualify duplicates.
    """
    pool = footage_pool(camp)
    if not pool:
        raise RuntimeError(
            f"campaign {camp.slug} is mode: edits but {camp.footage_dir} holds no video files — "
            "drop the clips you are editing with into that folder")
    h = int(hashlib.sha1(clip_id.encode()).hexdigest(), 16)
    path = pool[h % len(pool)]
    dur = _duration(path)
    room = max(dur - clip_len, 0.0)
    # quantise the in-point to whole seconds so the offset is legible in logs
    start = float(int((h // len(pool)) % int(room + 1))) if room >= 1 else 0.0
    return path, start


_SCORE_CACHE: dict[str, "list[float]"] = {}
DARK_FLOOR = 28.0      # mean 8-bit luma below this reads as black on a phone


def motion_scores(path: Path, cfg=None, fps: float = 2.0) -> list[float]:
    """Per-second "is anything happening here" score for a footage file.

    Decodes the whole file once at `fps` into 64x64 greyscale and scores each
    second by mean absolute frame-to-frame change. This is what separates usable
    footage from the things a blind in-point keeps landing on: intertitle cards,
    credits and letterboxed black are all static, so they score ~0, while a
    moving shot scores high. Cached per path.
    """
    key = str(path)
    if key in _SCORE_CACHE:
        return _SCORE_CACHE[key]
    import numpy as np
    cmd = [resolve("ffmpeg", cfg), "-v", "error", "-i", str(path),
           "-vf", f"fps={fps},scale=64:64,format=gray", "-f", "rawvideo", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        log.warning("could not score %s; falling back to blind in-points", path.name)
        return []
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = frames.size // 4096
    if n < 2:
        return []
    f = frames[: n * 4096].reshape(n, 4096).astype(np.float32)
    diff = np.abs(f[1:] - f[:-1]).mean(axis=1)
    # A shot also has to be *visible*. Old prints are full of near-black scenes
    # that still register motion; opening on one wastes the shot, so anything
    # under DARK_FLOOR mean luma is scored as unusable.
    luma = f[1:].mean(axis=1)
    diff = np.where(luma < DARK_FLOOR, 0.0, diff)
    # collapse to one score per second of source
    per_s = max(int(round(fps)), 1)
    secs = [float(diff[i:i + per_s].mean()) for i in range(0, diff.size, per_s)]
    _SCORE_CACHE[key] = secs
    return secs


def _duration(path: Path) -> float:
    out = subprocess.run(
        [resolve("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def build_shots(camp: Campaign, clip_id: str, cuts: list[float], cfg=None) -> list[dict]:
    """One footage file + in-point per shot, spread across the pool.

    Consecutive shots always come from different files when the pool allows it,
    so an edit reads as an edit rather than a jump-cut inside one clip.
    """
    pool = footage_pool(camp)
    if not pool:
        raise RuntimeError(
            f"campaign {camp.slug} is mode: edits but {camp.footage_dir} holds no video files — "
            "drop the clips you are editing with into that folder")
    h = int(hashlib.sha1(clip_id.encode()).hexdigest(), 16)
    durs = {p: _duration(p) for p in pool}
    used: dict[Path, set[int]] = {p: set() for p in pool}
    shots = []
    for i in range(len(cuts) - 1):
        path = pool[(h + i) % len(pool)]
        span = cuts[i + 1] - cuts[i]
        room = max(durs[path] - span, 0.0)
        if room < 1:
            offset = 0.0
        else:
            offset = _pick_offset(path, int(room), span, (h >> (i * 3 + 1)), used[path], cfg)
        used[path].add(int(offset))
        shots.append({"path": path, "in": float(offset), "dur": round(span, 3)})
    return shots


def _pick_offset(path: Path, room: int, span: float, seed: int,
                 used: set[int], cfg=None) -> float:
    """In-point with real motion in it, deterministic in `seed`.

    Walks candidate offsets in a seed-dependent order and takes the first whose
    window scores above the file's own median motion — so a shot never opens on
    an intertitle card or a black frame while a moving alternative exists.
    Falls back to the blind offset when the file cannot be scored.
    """
    import numpy as np
    scores = motion_scores(path, cfg)
    blind = float(seed % (room + 1))
    if not scores:
        return blind
    need = max(int(round(span)), 1)
    thresh = float(np.median([s for s in scores if s > 0.5]) if any(s > 0.5 for s in scores) else 0)
    best, best_score = blind, -1.0
    for k in range(room + 1):
        off = (seed + k * 7) % (room + 1)          # stride 7 to spread candidates
        if off in used:
            continue
        window = scores[off:off + need]
        if not window:
            continue
        m = float(np.mean(window))
        if m > best_score:
            best, best_score = float(off), m
        if m >= thresh:
            return float(off)
    return best


# xfade names grouped into styles an editor would actually reach for. "auto"
# cycles a mixed set so consecutive cuts differ and two clips from one campaign
# never share a transition sequence.
TRANSITION_STYLES = {
    "auto":     ("smoothleft", "dissolve", "zoomin", "smoothright", "pixelize", "circleopen"),
    "whip":     ("smoothleft", "smoothright", "hlwind", "hrwind"),
    "slide":    ("slideleft", "slideright", "slideup", "slidedown"),
    "dissolve": ("dissolve", "fade", "fadegrays"),
    "zoom":     ("zoomin", "circleopen", "circleclose"),
    "glitch":   ("pixelize", "hlslice", "vuslice", "squeezeh"),
    "flash":    ("fadewhite",),
}

GRADES = {
    "warm":  "eq=contrast=1.08:saturation=1.20:gamma_r=1.06:gamma_b=0.96",
    "cool":  "eq=contrast=1.08:saturation=1.10:gamma_r=0.96:gamma_b=1.07",
    "crush": "eq=contrast=1.22:saturation=0.88:brightness=-0.03",
    "vhs":   "eq=contrast=1.12:saturation=1.35,noise=alls=6:allf=t",
}


def transition_kinds(style: str, clip_id: str, n: int) -> list[str]:
    """One xfade name per cut. An explicit xfade name is used verbatim."""
    pool = TRANSITION_STYLES.get(style)
    if pool is None:
        return [style] * n          # caller passed a raw xfade name
    h = int(hashlib.sha1(clip_id.encode()).hexdigest(), 16)
    return [pool[(h + i) % len(pool)] for i in range(n)]


def _xfade_graph(labels: list[str], lens: list[float], td: float,
                 kinds: list[str]) -> tuple[list[str], str]:
    """Chain xfades across shots. Returns (filter parts, final label).

    Each xfade overlaps its two inputs by `td`, so the running length is
    cum = cum - td + len(next). Offsets are cumulative, not per-shot.
    """
    if len(labels) == 1:
        return [], labels[0]
    parts, prev, cum = [], labels[0], lens[0]
    for i in range(1, len(labels)):
        out = f"x{i}"
        parts.append(f"[{prev}][{labels[i]}]xfade=transition={kinds[i - 1]}"
                     f":duration={td:.3f}:offset={max(cum - td, 0):.3f}[{out}]")
        cum = cum - td + lens[i]
        prev = out
    return parts, prev


def _shot_chain(idx: int, W: int, H: int, fps: int, punch: float, flash: bool = False) -> str:
    """Filter chain turning input `idx` into one WxH shot, with a beat punch."""
    chain = (f"[{idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
             f"crop={W}:{H},fps={fps}")
    if punch > 0:
        # ease a zoom back to 1.0 over 0.35s. Alternating the direction per shot
        # keeps a long edit from feeling metronomic.
        z = (f"({1 + punch}-{punch}*min(t/0.35\\,1))" if idx % 2 == 0
             else f"(1+{punch}*min(t/0.35\\,1))")
        chain += (f",crop=w='trunc(iw/{z}/2)*2':h='trunc(ih/{z}/2)*2':"
                  f"x='(iw-ow)/2':y='(ih-oh)/2',scale={W}:{H}")
    if flash and idx > 0:
        chain += ",fade=t=in:st=0:d=0.07:color=white"
    # setsar LAST: the animated crop rounds w and h independently, so the scale
    # above lands on a near-but-not-1:1 SAR, and concat/xfade both refuse inputs
    # whose parameters differ. format too — xfade needs matching pix_fmt.
    return chain + f",setsar=1,format=yuv420p[v{idx}]"


# ── render ────────────────────────────────────────────────────

_has_subtitles: bool | None = None


def has_subtitles_filter(cfg: Config | None = None) -> bool:
    """True if this ffmpeg was built with libass (needed to burn in the .ass)."""
    global _has_subtitles
    if _has_subtitles is None:
        out = subprocess.run([resolve("ffmpeg", cfg), "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
        _has_subtitles = any(line.split()[1:2] == ["subtitles"]
                             for line in out.splitlines() if line.strip())
    return _has_subtitles


def _ass_has_events(ass_text: str) -> bool:
    return any(l.startswith("Dialogue:") for l in ass_text.splitlines())



def render_clip(cfg: Config, camp: Campaign, video: Path, transcript: dict, start: float, end: float,
                hook: str, out: Path, footage: Path | None = None,
                footage_start: float = 0.0) -> Path:
    """Cut `video` [start,end) to a 9:16 MP4 with burned-in text.

    In [EDITS] mode `footage` carries the picture and `video` carries only the
    campaign's track, so the two are read from separate inputs and muxed: the
    lyrics stay locked to the song while the visuals come from your own pool.
    """
    picture = footage or video
    src = probe(picture)
    clip_len = end - start
    words = [w for s in transcript["segments"] for w in s.get("words", [])
             if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]
    ass_path = out.with_suffix(".ass")
    ass_text = build_ass(cfg, camp, words, start, clip_len, hook)
    ass_path.write_text(ass_text, encoding="utf-8")

    vf = (_crop_filter(cfg, camp, picture, footage_start, footage_start + clip_len, src)
          if footage else _crop_filter(cfg, camp, video, start, end, src))
    if _ass_has_events(ass_text):
        if not has_subtitles_filter(cfg):
            raise RuntimeError(
                "this ffmpeg has no 'subtitles' filter (built without libass), so captions, "
                "hook titles and credit overlays cannot be burned in. Install an ffmpeg with "
                "libass, or turn off render.captions.enabled + render.hook_title.enabled and "
                "clear the campaign's required_credit_overlay to render bare clips.")
        ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        vf = f"{vf},subtitles='{ass_escaped}',format=yuv420p"
    else:
        vf = f"{vf},format=yuv420p"
    fps = int(cfg.get("render", "fps", default=30))

    cmd = [resolve("ffmpeg", cfg), "-y", "-hide_banner", "-loglevel", "error"]
    if footage:
        # input 0 = your footage (video only), input 1 = the campaign's track (audio only).
        # -stream_loop -1 so footage shorter than the clip repeats rather than
        # ending the output early; -t bounds it back to the clip length.
        cmd += ["-stream_loop", "-1", "-ss", f"{footage_start:.3f}", "-t", f"{clip_len:.3f}",
                "-i", str(picture),
                "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
                "-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video)]
    cmd += [
        "-vf", vf, "-r", str(fps),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-profile:v", "high",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-shortest", "-dn", "-sn", "-map_metadata", "-1",
            "-dn", "-sn", "-map_metadata", "-1", "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def render_edit(cfg: Config, camp: Campaign, track: Path, transcript: dict, start: float,
                end: float, hook: str, out: Path, clip_id: str) -> Path:
    """[EDITS] render — your footage, cut to the track's beat, lyrics burned over.

    One ffmpeg input per shot (each pre-seeked, so decoding stays cheap), all
    normalised to WxH and concatenated; audio comes from the track alone.
    """
    r = _merge(cfg.get("render", default={}) or {}, camp.rules.get("render", {}) or {})
    ed = r.get("edit", {}) or {}
    W, H = int(r.get("width", 1080)), int(r.get("height", 1920))
    fps = int(r.get("fps", 30))
    punch = float(ed.get("punch", 0.08))
    bps = ed.get("beats_per_shot", 4)          # int, or "ramp" for accelerating cuts
    style = str(ed.get("transition", "auto")).lower()
    td = 0.0 if style == "cut" else float(ed.get("transition_duration", 0.22))
    fx = ed.get("effects", {}) or {}
    clip_len = end - start

    ramp = isinstance(bps, str) and bps.lower() == "ramp"
    beats = beat_times(track, start, clip_len, cfg) if (ramp or int(bps) > 0) else []
    cuts = shot_boundaries(beats, clip_len, bps)
    shots = build_shots(camp, clip_id, cuts, cfg)
    log.info("%s → %d shot(s) over %.1fs (%d beat(s) found)", clip_id, len(shots), clip_len, len(beats))

    words = [w for sg in transcript["segments"] for w in sg.get("words", [])
             if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]
    ass_path = out.with_suffix(".ass")
    ass_text = build_ass(cfg, camp, words, start, clip_len, hook)
    ass_path.write_text(ass_text, encoding="utf-8")

    # Every shot but the last must carry `td` extra footage for the next
    # transition to dissolve into; sum(lens) - (n-1)*td then lands on clip_len.
    lens = [sh["dur"] + td for sh in shots[:-1]] + [shots[-1]["dur"]]

    cmd = [resolve("ffmpeg", cfg), "-y", "-hide_banner", "-loglevel", "error"]
    for sh, ln in zip(shots, lens):
        # loop each source so a shot longer than its file still fills its slot
        cmd += ["-stream_loop", "-1", "-ss", f"{sh['in']:.3f}", "-t", f"{ln:.3f}",
                "-i", str(sh["path"])]
    cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(track)]

    flash = bool(fx.get("flash_on_cut", False))
    chains = [_shot_chain(i, W, H, fps, punch, flash) for i in range(len(shots))]
    labels = [f"v{i}" for i in range(len(shots))]
    if td > 0 and len(shots) > 1:
        kinds = transition_kinds(style, clip_id, len(shots) - 1)
        parts, last = _xfade_graph(labels, lens, td, kinds)
        log.info("%s transitions: %s", clip_id, ", ".join(kinds))
    else:
        parts = ["".join(f"[{l}]" for l in labels) + f"concat=n={len(shots)}:v=1:a=0[vc]"]
        last = "vc"
    chains += parts

    grade = str(fx.get("grade", "none")).lower()
    post = []
    if grade in GRADES:
        post.append(GRADES[grade])
    if float(fx.get("grain", 0)):
        post.append(f"noise=alls={int(fx['grain'])}:allf=t+u")
    if fx.get("vignette", False):
        post.append("vignette=PI/5")

    tail = f"[{last}]" + ("".join(p + "," for p in post))
    if _ass_has_events(ass_text):
        if not has_subtitles_filter(cfg):
            raise RuntimeError(
                "this ffmpeg has no 'subtitles' filter (built without libass); an [EDITS] post "
                "needs the lyrics burned in. Install an ffmpeg with libass.")
        esc = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")  # noqa: E501
        tail += f"subtitles='{esc}',"
    tail += "format=yuv420p[vout]"
    cmd += ["-filter_complex", ";".join(chains + [tail]),
            "-map", "[vout]", "-map", f"{len(shots)}:a:0",
            "-r", str(fps),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-profile:v", "high",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
            "-shortest", "-dn", "-sn", "-map_metadata", "-1",
            "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def run(cfg: Config, ledger: Ledger) -> int:
    camps = {c.slug: c for c in cfg.campaigns}
    rdir = cfg.run_dir or (cfg.data_dir / "renders")
    rdir.mkdir(parents=True, exist_ok=True)
    max_run = int(cfg.get("limits", "max_clips_per_run", default=12))
    n = 0
    planned = list(ledger.clips_with_status("planned", limit=max_run))
    for idx, clip in enumerate(planned, 1):
        log.info("  clip %d/%d …", idx, len(planned))
        camp = camps.get(clip["campaign"])
        src = ledger.conn.execute("SELECT * FROM sources WHERE id=?", (clip["source_id"],)).fetchone()
        if not camp or not src or not src["video_path"]:
            ledger.update_clip(clip["id"], status="failed")
            continue
        out = rdir / f"{clip['id']}.mp4"
        try:
            tr = load_transcript(cfg, src["id"])
            if camp.mode == "edits":
                render_edit(cfg, camp, Path(src["video_path"]), tr, clip["start_s"],
                            clip["end_s"], clip["hook"] or "", out, clip["id"])
            else:
                render_clip(cfg, camp, Path(src["video_path"]), tr, clip["start_s"],
                            clip["end_s"], clip["hook"] or "", out)
            ledger.update_clip(clip["id"], status="rendered", render_path=str(out))
            for plat in camp.platforms:
                if cfg.get("publish", plat, "enabled", default=False):
                    ledger.queue_post(clip["id"], camp.slug, plat)
            log.info("rendered %s (%.1fs)", clip["id"], clip["end_s"] - clip["start_s"])
            n += 1
        except subprocess.CalledProcessError as e:
            log.error("ffmpeg failed for %s: %s", clip["id"], e.stderr[-400:])
            ledger.update_clip(clip["id"], status="failed")
        except Exception:  # noqa: BLE001
            log.exception("render failed for %s", clip["id"])
            ledger.update_clip(clip["id"], status="failed")
    return n
