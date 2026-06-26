#!/usr/bin/env bash
# Claude → Ollama failover. Prints a clear banner + drops you into the local
# Ollama coding agent so work continues when Claude hits its session/token limit.
#
#   scripts/claude_ollama_failover.sh                 # banner + interactive agent
#   scripts/claude_ollama_failover.sh "fix bug in X"  # banner + one-shot task
#   scripts/claude_ollama_failover.sh --check         # just report readiness
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY=/home/praveen/venv-ardupilot/bin/python3
[ -x "$VENV_PY" ] || VENV_PY=python3
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
AGENT_MODEL="${OLLAMA_AGENT_MODEL:-qwen2.5-coder:14b}"

banner() {
  printf '\n'
  printf '\033[33m╔══════════════════════════════════════════════════════════════╗\033[0m\n'
  printf '\033[33m║  ⚠  CLAUDE SESSION LIMIT REACHED — SWITCHING TO LOCAL OLLAMA  ║\033[0m\n'
  printf '\033[33m╚══════════════════════════════════════════════════════════════╝\033[0m\n'
  printf '  Cloud Claude is unavailable. Continuing locally — zero cloud, zero limit.\n'
  printf '  Model   : \033[36m%s\033[0m  (override: OLLAMA_AGENT_MODEL=...)\n' "$AGENT_MODEL"
  printf '  Editor  : VS Code → Continue.dev is already wired to Ollama (~/.continue/config.yaml)\n'
  printf '  CLI     : this terminal agent (read/write/bash loop)\n\n'
}

# readiness check
ollama_up() { curl -s -o /dev/null -w '%{http_code}' "$OLLAMA_URL/api/tags" -m 5 2>/dev/null | grep -q 200; }

if [ "${1:-}" = "--check" ]; then
  if ollama_up; then
    n=$(curl -s "$OLLAMA_URL/api/tags" -m 5 | "$VENV_PY" -c "import sys,json;print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null)
    printf '✓ Ollama up at %s · %s models · failover READY\n' "$OLLAMA_URL" "${n:-?}"
    exit 0
  else
    printf '✗ Ollama not reachable at %s — run: ollama serve\n' "$OLLAMA_URL"; exit 1
  fi
fi

banner
if ! ollama_up; then
  printf '\033[31m✗ Ollama not running.\033[0m Start it in another terminal:  ollama serve\n'
  printf '   then re-run:  scripts/claude_ollama_failover.sh\n'
  exit 1
fi

exec "$VENV_PY" "$ROOT/scripts/ollama_agent.py" "$@"
