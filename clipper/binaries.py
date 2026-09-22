"""Locate the external binaries the pipeline shells out to.

Bare names like "ffmpeg" or "yt-dlp" are not enough in the situations this
pipeline actually hits:

* yt-dlp and gdown are installed as console scripts inside .venv/bin, so they
  are only on PATH while the venv is activated. `python run.py` via an explicit
  interpreter path (or launchd) does not activate anything.
* schedule_mac.sh runs the pass under launchd, whose PATH is typically just
  /usr/bin:/bin:/usr/sbin:/sbin — no /opt/homebrew/bin at all.
* Homebrew's ffmpeg 9 formula ships without libass, so the build that can burn
  in captions is often a keg-only one (ffmpeg@7) that is never on PATH.

Resolution order, first hit wins:
    1. $FFMPEG_BIN / $YT_DLP_BIN / ...   (tool name upper-cased, '-' → '_')
    2. the `tools:` section of config.yaml
    3. the directory holding the running interpreter (.venv/bin)
    4. known install prefixes on disk
    5. whatever the bare name resolves to on PATH
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Searched in order. Keg-only ffmpeg builds come first because they are the ones
# that still carry libass; /opt/homebrew/bin is the plain Homebrew prefix.
_PREFIXES = (
    "/opt/homebrew/opt/ffmpeg@7/bin",
    "/opt/homebrew/opt/ffmpeg@6/bin",
    "/usr/local/opt/ffmpeg@7/bin",
    "/usr/local/opt/ffmpeg@6/bin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
)


def _executable(p: Path) -> bool:
    return p.is_file() and os.access(p, os.X_OK)


def resolve(tool: str, cfg=None) -> str:
    """Absolute path to `tool`, or the bare name as a last resort."""
    env = os.environ.get(f"{tool.upper().replace('-', '_')}_BIN")
    if env:
        return env

    if cfg is not None:
        configured = cfg.get("tools", tool, default=None)
        if configured:
            return str(configured)

    # console scripts live next to the interpreter that is running us
    venv_bin = Path(sys.executable).parent / tool
    if _executable(venv_bin):
        return str(venv_bin)

    for prefix in _PREFIXES:
        p = Path(prefix) / tool
        if _executable(p):
            return str(p)

    return shutil.which(tool) or tool
