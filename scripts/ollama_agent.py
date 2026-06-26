#!/usr/bin/env python3
"""Local Ollama coding agent — the fallback when Claude hits its session limit.

A self-contained terminal coding agent powered by local Ollama (qwen2.5-coder:14b
by default). Supports a real tool loop: read files, write/patch files, run shell
commands, and chat — so work continues with zero cloud dependency.

Usage:
  scripts/ollama_agent.py                       # interactive REPL
  scripts/ollama_agent.py "fix the bug in X"    # one-shot task
  scripts/ollama_agent.py --model qwen2.5-coder:14b

No Claude / no cloud — everything runs against http://localhost:11434.
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_AGENT_MODEL", "qwen2.5-coder:14b")
ROOT = Path(__file__).resolve().parent.parent

SYSTEM = """You are a local coding agent running on Ollama (no cloud). You help edit code in this repo.
You have these tools — emit a SINGLE json object on its own line to call one, then stop and wait:
  {"tool":"read","path":"relative/path"}
  {"tool":"write","path":"relative/path","content":"full file content"}
  {"tool":"bash","cmd":"shell command"}
  {"tool":"done","summary":"what you did"}
Rules: read before you write; keep changes minimal; prefer the project's existing patterns;
after a tool result is returned, continue or call done. Never invent file paths — read to confirm."""


def _post(path, payload, timeout=300):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(messages):
    out = _post("/api/chat", {"model": MODEL, "messages": messages, "stream": False})
    return out.get("message", {}).get("content", "")


def run_tool(call):
    t = call.get("tool")
    try:
        if t == "read":
            p = ROOT / call["path"]
            return p.read_text()[:8000] if p.exists() else f"ERROR: {call['path']} not found"
        if t == "write":
            p = ROOT / call["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(call["content"])
            return f"wrote {call['path']} ({len(call['content'])} bytes)"
        if t == "bash":
            r = subprocess.run(call["cmd"], shell=True, cwd=ROOT, capture_output=True,
                               text=True, timeout=120)
            return (r.stdout + r.stderr)[-4000:]
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    return "ERROR: unknown tool"


def _extract_call(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"tool"' in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def agent_loop(task, max_steps=20):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    for step in range(max_steps):
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        call = _extract_call(reply)
        if not call or call.get("tool") == "done":
            print("\n" + (call.get("summary", reply) if call else reply))
            return
        print(f"  → {call['tool']} {call.get('path', call.get('cmd', ''))[:70]}")
        result = run_tool(call)
        messages.append({"role": "user", "content": f"TOOL RESULT:\n{result}"})
    print("(max steps reached)")


def main():
    args = [a for a in sys.argv[1:]]
    global MODEL
    if "--model" in args:
        i = args.index("--model"); MODEL = args[i + 1]; del args[i:i + 2]
    # health
    try:
        tags = _post("/api/tags", {}, timeout=5) if False else json.loads(
            urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5).read())
        names = [m["name"] for m in tags.get("models", [])]
    except Exception:
        print(f"✗ Ollama not reachable at {OLLAMA}. Start it: `ollama serve`"); sys.exit(1)
    if MODEL not in names:
        print(f"⚠ model {MODEL} not pulled. Available: {names[:6]}\n  pull: ollama pull {MODEL}")
        MODEL = next((n for n in names if "coder" in n), names[0] if names else MODEL)
        print(f"  → using {MODEL}")
    print(f"🦙 Ollama agent · model={MODEL} · repo={ROOT.name}  (Ctrl-C to exit)")
    if args:
        agent_loop(" ".join(args)); return
    while True:
        try:
            task = input("\nollama> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return
        if task in ("exit", "quit"):
            return
        if task:
            agent_loop(task)


if __name__ == "__main__":
    main()
