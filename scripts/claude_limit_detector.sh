#!/usr/bin/env bash
# Claude-Code Notification hook — auto-detects the session/token-limit message and
# prints the Claude→Ollama failover banner so the operator sees the switch path
# the moment Claude becomes unavailable. Wired as a "Notification" hook in
# .claude/settings.local.json. Reads the hook JSON from stdin.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
input="$(cat 2>/dev/null)"
msg="$(printf '%s' "$input" | tr 'A-Z' 'a-z')"

# limit-signal keywords
if printf '%s' "$msg" | grep -qE 'session limit|token limit|usage limit|limit reached|rate limit|out of (tokens|credits)|quota'; then
  printf '\n\033[33m═══════════════════════════════════════════════════════════════\033[0m\n'
  printf '\033[33m⚠  CLAUDE LIMIT DETECTED → LOCAL OLLAMA FAILOVER AVAILABLE\033[0m\n'
  printf '\033[33m═══════════════════════════════════════════════════════════════\033[0m\n'
  printf '   Keep working locally (no cloud, no limit):\n\n'
  printf '   CLI agent : \033[36mbash %s/scripts/claude_ollama_failover.sh\033[0m\n' "$ROOT"
  printf '   VS Code   : open Continue.dev (already wired to Ollama)\n'
  printf '   Check     : bash %s/scripts/claude_ollama_failover.sh --check\n\n' "$ROOT"
  # log the event for the record
  mkdir -p "$ROOT/jobs/logs"
  printf '{"ts":"%s","event":"claude_limit_detected"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ROOT/jobs/logs/failover.jsonl" 2>/dev/null
fi
exit 0
