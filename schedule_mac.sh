#!/usr/bin/env bash
# Install / remove a launchd agent that runs the pipeline every 20 minutes
# whenever the Mac is awake. Logs → data/logs/launchd.{out,err}
#   ./schedule_mac.sh install
#   ./schedule_mac.sh uninstall
#   ./schedule_mac.sh status
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LABEL="net.narva.clipper"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
INTERVAL_MIN="${INTERVAL_MIN:-20}"

case "${1:-}" in
  install)
    mkdir -p "$HERE/data/logs" "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$HERE/.venv/bin/python</string>
    <string>$HERE/run.py</string>
  </array>
  <key>WorkingDirectory</key><string>$HERE</string>
  <key>StartInterval</key><integer>$((INTERVAL_MIN*60))</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$HERE/data/logs/launchd.out</string>
  <key>StandardErrorPath</key><string>$HERE/data/logs/launchd.err</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict></plist>
EOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "✔ scheduled every $INTERVAL_MIN min ($LABEL). Tip: System Settings → Battery → prevent sleep on power, or use 'caffeinate'."
    ;;
  uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"; echo "✔ removed" ;;
  status)
    launchctl list | grep "$LABEL" || echo "not loaded"
    tail -n 20 "$HERE/data/logs/clipper.log" 2>/dev/null || true ;;
  *) echo "usage: $0 install|uninstall|status"; exit 1 ;;
esac
