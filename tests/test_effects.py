import sys, copy, dataclasses, subprocess, json, shutil, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clipper.config import load_config, load_campaign
from clipper.transcribe import load_transcript
from clipper.render import render_edit
from clipper.binaries import resolve
import logging; logging.basicConfig(level=logging.ERROR)


sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures
cfg=load_config(ROOT)
base=fixtures.demo_campaign(Path(tempfile.mkdtemp(prefix="clipper-camp-")))
track=fixtures.music_video()
tr={"language":"en","duration":30.0,"segments":[]}
OUT=Path(tempfile.mkdtemp(prefix="clipper-fx-"))
S,E = 2.0, 12.0    # 10s, inside the 30s fixture

def variant(**edit):
    r=copy.deepcopy(base.rules); r["render"]["edit"].update(edit)
    return dataclasses.replace(base, rules=r)

def dur(p):
    o=subprocess.run([resolve("ffprobe"),"-v","error","-show_entries","format=duration",
                      "-of","json",str(p)],capture_output=True,text=True).stdout
    return float(json.loads(o)["format"]["duration"])

cases = [
  ("cut",        dict(transition="cut")),
  ("whip",       dict(transition="whip")),
  ("slide",      dict(transition="slide")),
  ("dissolve",   dict(transition="dissolve")),
  ("zoom",       dict(transition="zoom")),
  ("glitch",     dict(transition="glitch")),
  ("flashstyle", dict(transition="flash")),
  ("raw-xfade",  dict(transition="circleclose")),
  ("long-td",    dict(transition="auto", transition_duration=0.6)),
  ("no-punch",   dict(punch=0.0)),
  ("one-shot",   dict(beats_per_shot=0, transition="cut")),
  ("fx-flash",   dict(effects={"flash_on_cut":True,"grade":"none"})),
  ("fx-grain",   dict(effects={"grain":12,"grade":"crush"})),
  ("fx-vhs",     dict(effects={"grade":"vhs","vignette":True})),
  ("fx-cool",    dict(effects={"grade":"cool","vignette":False})),
  ("fx-bad-grade",dict(effects={"grade":"nonexistent"})),
]
ok=fail=0
for name, ed in cases:
    p = OUT/f"{name}.mp4"
    try:
        render_edit(cfg, variant(**ed), track, tr, S, E, "", p, f"case_{name}")
        d = dur(p)
        good = abs(d-(E-S)) < 0.35 and p.stat().st_size > 100_000
        print(f"  {'PASS' if good else 'FAIL'}  {name:<13} {d:5.2f}s  {p.stat().st_size//1024:>5} KB")
        ok += good; fail += (not good)
    except Exception as e:
        print(f"  FAIL  {name:<13} {type(e).__name__}: {str(e)[-160:]}")
        fail += 1
shutil.rmtree(OUT, ignore_errors=True)
print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
