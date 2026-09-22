#!/usr/bin/env python3
"""Thin shim so `python run.py ...` keeps working.

The CLI itself lives in clipper/cli.py, which is what the installed `verticlip`
entry point calls. Both paths run exactly the same code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clipper.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
