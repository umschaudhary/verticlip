#!/usr/bin/env bash
# install.sh — set clipper up on macOS or Linux.
#
# Installs ffmpeg (with libass), a Python venv and the Python deps, then checks
# everything the pipeline actually needs at runtime. It never fails silently:
# anything it could not do is listed at the end with the command to fix it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
WARN=()
note()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()    { printf '  ok    %s\n' "$*"; }
warn()  { printf '  MISS  %s\n' "$*"; WARN+=("$*"); }

case "$(uname -s)" in
  Darwin) OS=mac ;;
  Linux)  OS=linux ;;
  *)      echo "unsupported platform: $(uname -s)"; exit 1 ;;
esac
note "platform: $OS"

# ── ffmpeg, with libass ───────────────────────────────────────
has_libass() {
  command -v "$1" >/dev/null 2>&1 || return 1
  local filters
  filters="$("$1" -hide_banner -filters 2>/dev/null)" || return 1
  printf '%s\n' "$filters" | awk '{print $2}' | grep -x subtitles >/dev/null 2>&1
}

note "ffmpeg"
FF=""
for cand in ffmpeg /opt/homebrew/opt/ffmpeg@7/bin/ffmpeg /usr/local/opt/ffmpeg@7/bin/ffmpeg; do
  if has_libass "$cand"; then FF="$cand"; break; fi
done
if [ -z "$FF" ]; then
  if [ "$OS" = mac ] && command -v brew >/dev/null 2>&1; then
    # Homebrew's current ffmpeg ships without libass; ffmpeg@7 is bottled and has it
    echo "  installing ffmpeg@7 (the current formula has no libass)…"
    brew install ffmpeg@7 >/dev/null 2>&1
    has_libass /opt/homebrew/opt/ffmpeg@7/bin/ffmpeg && FF=/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg
    has_libass /usr/local/opt/ffmpeg@7/bin/ffmpeg    && FF=/usr/local/opt/ffmpeg@7/bin/ffmpeg
  elif [ "$OS" = linux ] && command -v apt-get >/dev/null 2>&1; then
    echo "  installing ffmpeg via apt…"
    sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg >/dev/null 2>&1
    has_libass ffmpeg && FF=ffmpeg
  fi
fi
if [ -n "$FF" ]; then ok "ffmpeg with libass: $FF"
else warn "ffmpeg with libass — captions cannot be burned in. macOS: brew install ffmpeg@7 · Debian/Ubuntu: sudo apt install ffmpeg"; fi

# ── python ────────────────────────────────────────────────────
note "python"
PYBIN=""
for c in python3.12 python3.11 python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,11) else 1)' && { PYBIN="$c"; break; }
done
if [ -z "$PYBIN" ]; then
  warn "python 3.11+ — required"
else
  ok "$($PYBIN --version)"
  [ -d .venv ] || "$PYBIN" -m venv .venv
  ./.venv/bin/python -m pip install --quiet --upgrade pip
  echo "  installing dependencies…"
  if ./.venv/bin/python -m pip install --quiet -r requirements.txt; then ok "python dependencies"
  else warn "pip install -r requirements.txt failed — run it yourself to see why"; fi
fi

# ── fonts ─────────────────────────────────────────────────────
note "fonts"
if command -v fc-list >/dev/null 2>&1; then
  FAMILIES="$(fc-list : family 2>/dev/null)"
  if printf '%s\n' "$FAMILIES" | grep -i montserrat >/dev/null 2>&1; then ok "Montserrat present"
  else warn "Montserrat — captions fall back to a default face. Get it free at https://fonts.google.com/specimen/Montserrat and install the Black weight"; fi
else
  # macOS has no fc-list unless fontconfig is installed; ffmpeg bundles its own
  [ "$OS" = mac ] && ok "fontconfig not present (normal on macOS); ffmpeg resolves fonts itself" \
                  || warn "fontconfig — install fontconfig so ffmpeg can find fonts by name"
fi

# ── ollama (the default highlight picker) ─────────────────────
note "highlight model"
if command -v ollama >/dev/null 2>&1; then
  ok "ollama installed"
  MODELS="$(ollama list 2>/dev/null)"
  if printf '%s\n' "$MODELS" | grep "qwen2.5:7b-instruct" >/dev/null 2>&1; then ok "qwen2.5:7b-instruct pulled"
  else warn "the model — run: ollama pull qwen2.5:7b-instruct"; fi
else
  warn "ollama — the default highlight picker. Install from https://ollama.com, then: ollama pull qwen2.5:7b-instruct  (or set highlighter.provider to anthropic/openai in config.yaml)"
fi

# ── config scaffolding ────────────────────────────────────────
note "config"
[ -f .env ] || { cp .env.example .env; ok "created .env from .env.example"; }
[ -f .env ] && ok ".env present"
mkdir -p data/{downloads,transcripts,runs,logs} secrets
ok "data/ and secrets/ created"

# ── result ────────────────────────────────────────────────────
if [ ${#WARN[@]} -eq 0 ]; then
  note "ready"
  echo "  ./clip                 walk through it step by step"
  echo "  ./clip <url> 6         six clips from one link"
else
  note "installed, but ${#WARN[@]} thing(s) still need you:"
  for w in "${WARN[@]}"; do echo "  · $w"; done
  echo
  echo "Re-run ./install.sh once those are sorted."
fi
