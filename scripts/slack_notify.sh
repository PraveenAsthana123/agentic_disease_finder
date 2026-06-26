#!/usr/bin/env bash
# Slack notifier — posts project events (watchdog down, health errors, failover,
# build status) to a Slack channel via an Incoming Webhook. Graceful no-op when
# not configured (§57.7 — never crash a caller if Slack isn't set up).
#
# Setup (one-time, operator):
#   1. Create a Slack Incoming Webhook: https://api.slack.com/messaging/webhooks
#   2. Put the URL in ~/.config/agenticfinder/slack_webhook  (chmod 600)
#      OR export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
#
# Usage:
#   scripts/slack_notify.sh "message text"            # info
#   scripts/slack_notify.sh --level warn "message"    # warn (amber)
#   scripts/slack_notify.sh --level error "message"   # error (red)
#   scripts/slack_notify.sh --test                    # send a test message
#   scripts/slack_notify.sh --check                   # report config status (no send)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${HOME}/.config/agenticfinder/slack_webhook"

# resolve webhook: env wins, then config file
WEBHOOK="${SLACK_WEBHOOK_URL:-}"
[ -z "$WEBHOOK" ] && [ -f "$CFG" ] && WEBHOOK="$(tr -d '[:space:]' < "$CFG" 2>/dev/null)"

level="info"; [ "${1:-}" = "--level" ] && { level="$2"; shift 2; }

if [ "${1:-}" = "--check" ]; then
  if [ -n "$WEBHOOK" ]; then echo "✓ Slack configured (${WEBHOOK:0:30}…)"; exit 0
  else echo "✗ Slack NOT configured. Set SLACK_WEBHOOK_URL or write $CFG"; exit 1; fi
fi

if [ -z "$WEBHOOK" ]; then
  # honest no-op — don't crash callers (watchdog/health) when Slack isn't set up
  echo "slack: not configured (skipped)"; exit 0
fi

[ "${1:-}" = "--test" ] && set -- "🧪 agenticfinder Slack integration test — $(date '+%Y-%m-%d %H:%M %Z')"
msg="${*:-（empty）}"
host="$(hostname 2>/dev/null || echo host)"
case "$level" in
  error) emoji="🔴"; color="#e01e5a" ;;
  warn)  emoji="🟠"; color="#ecb22e" ;;
  *)     emoji="🔵"; color="#2eb67d" ;;
esac
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
payload=$(cat <<JSON
{"attachments":[{"color":"$color","blocks":[
  {"type":"section","text":{"type":"mrkdwn","text":"$emoji *agenticfinder* · $level\n$msg"}},
  {"type":"context","elements":[{"type":"mrkdwn","text":"host: $host · $ts"}]}
]}]}
JSON
)
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
  -d "$payload" "$WEBHOOK" -m 10 2>/dev/null)
mkdir -p "$ROOT/jobs/logs"
printf '{"ts":"%s","level":"%s","http":"%s","msg":%s}\n' "$ts" "$level" "$code" "$(printf '%s' "$msg" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' 2>/dev/null || echo '""')" >> "$ROOT/jobs/logs/slack.jsonl"
[ "$code" = "200" ] && echo "slack: sent ($level)" || echo "slack: failed (http=$code)"
