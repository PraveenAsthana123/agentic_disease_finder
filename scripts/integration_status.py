#!/usr/bin/env python3
"""Integration status collector — live state of every local-AI integration.

Powers the Integration Hub UI. Checks: Ollama, OpenClaw gateway, agentic editor
extensions (VS Code + Antigravity), Claude→Ollama failover, Slack notifier, MCP,
and the project backend. Pure reads — no mutation.
"""
import glob
import json
import os
import subprocess
import urllib.request
from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).resolve().parent.parent


def _http(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode()
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:120]


def _ollama():
    code, body = _http("http://localhost:11434/api/tags", 4)
    if code != 200:
        return {"status": "down", "detail": "Ollama not reachable on :11434"}
    models = [m["name"] for m in json.loads(body).get("models", [])]
    coders = [m for m in models if "coder" in m or "code" in m]
    code2, ps = _http("http://localhost:11434/api/ps", 3)
    loaded = [m["name"] for m in json.loads(ps).get("models", [])] if code2 == 200 else []
    return {"status": "live", "n_models": len(models), "coder_models": coders[:8],
            "loaded_now": loaded, "endpoint": "http://localhost:11434"}


def _openclaw():
    code, body = _http("http://localhost:18789/health", 4)
    mj = HOME / ".openclaw/agents/main/agent/models.json"
    oll_models = []
    if mj.exists():
        try:
            d = json.load(open(mj))
            oll_models = [m["id"] for m in d.get("providers", {}).get("ollama", {}).get("models", [])]
        except Exception:  # noqa: BLE001
            pass
    return {"status": "live" if code == 200 else "down",
            "gateway": "http://localhost:18789", "health": body[:40] if code == 200 else None,
            "ollama_models_wired": [m for m in oll_models if "coder" in m or "qwen3" in m or "deepseek" in m]}


def _extensions():
    def scan(d):
        try:
            names = os.listdir(d)
        except OSError:
            return {}
        def has(pat):
            return any(pat in n.lower() for n in names)
        return {"continue": has("continue"), "cline": has("claude-dev"),
                "roo": has("roo-cline"), "kilo": has("kilo-code"),
                "openclaw": has("openclaw")}
    return {"vscode": scan(HOME / ".vscode/extensions"),
            "antigravity": scan(HOME / ".antigravity/extensions")}


def _failover():
    sh = ROOT / "scripts/claude_ollama_failover.sh"
    hook = (ROOT / ".claude/settings.local.json")
    hook_wired = False
    if hook.exists():
        try:
            hook_wired = "Notification" in json.load(open(hook)).get("hooks", {})
        except Exception:  # noqa: BLE001
            pass
    return {"status": "ready" if sh.exists() else "missing",
            "cli": "scripts/claude_ollama_failover.sh", "auto_detect_hook": hook_wired,
            "continue_config": (HOME / ".continue/config.yaml").exists()}


def _slack():
    cfg = HOME / ".config/agenticfinder/slack_webhook"
    configured = bool(os.environ.get("SLACK_WEBHOOK_URL")) or cfg.exists()
    return {"status": "active" if configured else "needs_webhook",
            "notifier": "scripts/slack_notify.sh",
            "setup": None if configured else f"write webhook to {cfg} (see docs/SLACK_INTEGRATION.md)"}


def _mcp():
    have = (ROOT / "mcp_server.py").exists() or (ROOT / "mcp" / "mcp_server.py").exists() or bool(__import__("glob").glob(str(ROOT / "**" / "mcp_server.py"), recursive=True))
    return {"status": "present" if have else "missing",
            "server": "mcp_server.py", "wired_to_editors": False,
            "note": "MCP infra exists; not yet exposed to Cline/Roo/Continue (see TOP_1PCT plan #3)"}


def _backend():
    code, _ = _http("http://localhost:8010/api/data-manager", 6)
    return {"status": "live" if code == 200 else "down", "endpoint": "http://localhost:8010",
            "frontend_3003": _http("http://localhost:3003", 3)[0] == 200}


def collect():
    integ = {"ollama": _ollama(), "openclaw": _openclaw(), "agentic_extensions": _extensions(),
             "claude_ollama_failover": _failover(), "slack": _slack(), "mcp": _mcp(),
             "project_backend": _backend()}
    # health rollup
    up = sum(1 for k, v in integ.items()
             if isinstance(v, dict) and v.get("status") in ("live", "ready", "active", "present"))
    from datetime import datetime, timezone
    return {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "integrations": integ, "summary": {"total": len(integ), "healthy": up}}


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
