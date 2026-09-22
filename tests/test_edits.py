"""Unit tests for the code added for Clipster: beats, edits mode, render overrides."""
import sys, tempfile, subprocess, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from clipper import beats, render
from clipper.config import load_config, load_campaign
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name}")
    else:    fail += 1; print(f"  FAIL  {name}  {detail}")


cfg = load_config(ROOT)
TRUE_BPM = fixtures.TRUE_BPM


print("\n── beats (against a generated %g BPM click) ──" % TRUE_BPM)
TRACK = fixtures.click_track()
b = beats.beat_times(TRACK, 0.0, 28.0, cfg)
check("finds beats", len(b) > 20, len(b))
iv = np.diff(b) if len(b) > 1 else np.array([float("nan")])
check("tempo steady (std < 5ms)", iv.std() < 0.005, f"std={iv.std()*1000:.1f}ms")
check("beats within window", all(0 <= t <= 28.0 for t in b))
check("beats monotonic", all(b[i] < b[i+1] for i in range(len(b)-1)))
bpm = 60/iv.mean() if len(b) > 1 else float("nan")
check("BPM in plausible range", beats.BPM_MIN <= bpm <= beats.BPM_MAX, f"{bpm:.1f}")
# the detector may lock to half or double time; both are musically correct
ratio = bpm / TRUE_BPM if bpm == bpm else 0
check("recovers the true tempo (1x, 1/2x or 2x)",
      any(abs(ratio - m) < 0.04 for m in (0.5, 1.0, 2.0)), f"{bpm:.1f} vs {TRUE_BPM}")

print("\n── beats: degenerate input ──")
check("silence -> no crash", isinstance(beats.onset_envelope(np.zeros(44100, np.float32)), np.ndarray))
check("too-short signal -> empty", beats.onset_envelope(np.zeros(100, np.float32)).size == 0)
check("missing file -> [] not raise", beats.beat_times(Path("/nope.mp4"), 0, 5, cfg) == [])

print("\n── shot_boundaries ──")
c4 = beats.shot_boundaries(b, 24.24, 4)
check("starts at 0", c4[0] == 0.0)
check("ends at clip_len", abs(c4[-1] - 24.24) < 1e-6, c4[-1])
check("monotonic", all(c4[i] < c4[i+1] for i in range(len(c4)-1)), c4)
check("no shot shorter than min_shot", min(np.diff(c4)) >= 0.45, min(np.diff(c4)))
check("more beats/shot -> fewer shots",
      len(beats.shot_boundaries(b,24.24,8)) < len(beats.shot_boundaries(b,24.24,2)))
fb = beats.shot_boundaries([], 20.0, 4)
check("no beats -> even fallback", len(fb) > 1 and abs(fb[-1]-20.0) < 1e-6, fb)
check("beats_per_shot=0 -> fallback", len(beats.shot_boundaries(b, 20.0, 0)) > 1)

print("\n── build_shots ──")
camp = fixtures.demo_campaign(Path(tempfile.mkdtemp(prefix="clipper-camp-")))
shots = render.build_shots(camp, "abc_00", c4)
check("one shot per interval", len(shots) == len(c4)-1, (len(shots), len(c4)-1))
check("durations match cuts", all(abs(s["dur"]-(c4[i+1]-c4[i])) < 1e-3 for i,s in enumerate(shots)))
pool = render.footage_pool(camp)
check("consecutive shots differ (pool>1)",
      len(pool) < 2 or all(shots[i]["path"] != shots[i+1]["path"] for i in range(len(shots)-1)),
      f"pool={len(pool)} " + str([s["path"].name for s in shots]))
check("single-file pool still varies in-points",
      len(pool) > 1 or len({s["in"] for s in shots}) > 1, [s["in"] for s in shots])
check("deterministic", [ (s["path"],s["in"]) for s in render.build_shots(camp,"abc_00",c4)] ==
                       [ (s["path"],s["in"]) for s in shots])
other = render.build_shots(camp, "abc_01", c4)
check("different clip -> different edit",
      [s["in"] for s in other] != [s["in"] for s in shots] or
      [s["path"] for s in other] != [s["path"] for s in shots])
check("in-points inside file", all(s["in"] >= 0 for s in shots))

print("\n── render overrides ──")
words = [{"word":"hello","start":1.0,"end":1.4},{"word":"world","start":1.4,"end":1.9}]
a_edits = render.build_ass(cfg, camp, words, 0.0, 10.0, "some hook")
check("edits: hook suppressed", "some hook" not in a_edits.lower())
check("edits: lyrics present", "HELLO" in a_edits.upper())
clip_camp = fixtures.clip_campaign(Path(tempfile.mkdtemp(prefix="clipper-camp-")))
a_clip = render.build_ass(cfg, clip_camp, words, 0.0, 10.0, "some hook")
check("clip: hook present", "some hook" in a_clip)
check("clip: captions suppressed", "HELLO" not in a_clip.upper())
check("clip: hook held full clip", ",0:00:10.00," in a_clip.replace("0:00:10.0,","0:00:10.00,"), 
      [l for l in a_clip.splitlines() if l.startswith("Dialogue")])

print("\n── _merge ──")
check("deep merge", render._merge({"a":{"x":1,"y":2},"b":3},{"a":{"y":9}}) == {"a":{"x":1,"y":9},"b":3})
base={"a":{"x":1}}; render._merge(base,{"a":{"x":2}})
check("does not mutate base", base == {"a":{"x":1}})

print("\n── error paths ──")
import copy, dataclasses
empty = dataclasses.replace(camp, footage_dir=Path("/tmp/definitely-not-here"))
try:
    render.build_shots(empty, "x", [0.0, 2.0]); check("empty pool raises", False)
except RuntimeError as e:
    check("empty pool raises with actionable message", "drop the clips" in str(e), str(e))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
