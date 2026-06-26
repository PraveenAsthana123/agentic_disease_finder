# Top-1% Local-AI + Agentic Integration Plan (multi-approach)

> Goal: a local-first AI coding + project-ops setup that rivals cloud agents —
> zero session limits, agents that call REAL project tools, GPU-aware routing,
> verified output, full observability. Grounded in THIS machine's actual stack.

## Current stack (verified 2026-06-25)
| Have | Detail |
|---|---|
| Ollama | 29 models (qwen3-coder:30b, qwen2.5-coder:14b/3b, deepseek-coder-v2, **llama3-groq-tool-use**, bge-m3) |
| Agentic editors | Continue.dev + Cline + Roo Code + Kilo Code — in **VS Code AND Antigravity** |
| Claude→Ollama failover | auto-detect hook + banner + CLI agent (`scripts/ollama_agent.py`) |
| Slack | notifier wired to watchdog + failover (needs webhook URL) |
| MCP infra | `mcp_server.py` + `mcp_client.py` exist — **NOT wired to the editors** |
| GPU | **GTX 1080 Ti, 11 GB** — fits ≤14B comfortably; 30B spills to CPU (slow) |
| Backend | FastAPI, 38 endpoints, single venv, watchdog, 18 crons |

## The gap: pieces exist but aren't connected. Top-1% = wiring them into one loop.

---

## The 10 integrations (each with MULTIPLE approaches)

### 1. 🥇 MCP — agents call REAL project tools (the #1 differentiator)
Text-editing agents are common; agents that query your DB, run drills, hit your
health API, and read your RAG are top-1%. You already have `mcp_server.py`.
- **Approach A (recommended):** expose project tools as an MCP server (DB query,
  `/api/health`, drill runner, EEG analysis, RAG search) → register in
  Cline/Roo/Continue MCP config. Agents then *act on real data*.
- **Approach B:** native function-calling via `llama3-groq-tool-use` (installed) —
  for the CLI `ollama_agent.py`, give it the same tools as JSON functions.
- **Approach C:** the `§50` dispatch scripts as a fallback tool lane.
- **Effort:** A ≈ 1 day · B ≈ 2 hr · C ≈ done.

### 2. 🥇 GPU-aware model routing (tiering, §111)
30B on an 11GB GPU spills to CPU → slow. Top-1% routes by task + fits the GPU.
- **Approach A:** tier map — autocomplete→`qwen2.5-coder:3b`, edit/chat→
  `qwen2.5-coder:14b` (fits 11GB), heavy-reason→`qwen3-coder:30b` (accept slower)
  or `deepseek-coder-v2`. Wire into a router used by CLI agent + Continue.
- **Approach B:** Ollama `keep_alive` + `num_gpu` tuning so the 14B stays resident
  (no reload cost). Set `OLLAMA_KEEP_ALIVE=30m`.
- **Approach C:** quantization — pull `qwen2.5-coder:14b-instruct-q4_K_M` to fit
  more headroom.
- **Effort:** A ≈ 3 hr · B ≈ 30 min · C ≈ 30 min.

### 3. 🥈 Pre-configure Cline / Roo / Kilo / Continue for Ollama
Right now they're installed but unconfigured — you'd click through setup.
- **Approach A:** write each editor's settings (provider=ollama, apiBase=
  localhost:11434, model per tier) into their globalStorage/settings.
- **Approach B:** a one-shot `scripts/configure_local_agents.sh` that seeds all
  four so any fresh machine is ready in one command.
- **Effort:** ≈ 2 hr.

### 4. 🥈 Codebase RAG / retrieval (small-context survival)
Local models have small context; they need retrieval to work on a big repo.
- **Approach A:** Continue `@codebase` (already wired to bge-m3) — index now.
- **Approach B:** a project RAG MCP tool (chunk repo → bge-m3 → vector store →
  search) shared by ALL agents (compose with the project's existing vector infra).
- **Approach C:** `aider`-style repo-map (ctags) for cheap structural context.
- **Effort:** A ≈ done (index) · B ≈ 1 day · C ≈ 2 hr.

### 5. 🥈 Failover continuity (Claude → Ollama with memory)
Today failover starts cold — Ollama doesn't know what Claude was doing.
- **Approach A:** on limit-detect, dump the last task/plan/files to
  `.agent/handoff.md`; `ollama_agent.py` reads it on launch.
- **Approach B:** shared session log both agents append to (`.agent/session.jsonl`).
- **Effort:** ≈ 3 hr.

### 6. 🥉 Local output verification (§50 council / drill gate)
Local models hallucinate more — top-1% verifies before trusting.
- **Approach A:** §50 council (author/reviewer/advisor across 3 models) on
  non-trivial diffs.
- **Approach B:** drill-gate — every agent edit must pass the project's drills
  before commit (deterministic verify, §43/§57).
- **Effort:** A ≈ done (scripts exist) · B ≈ 3 hr.

### 7. 🥉 Slack / notifications (built — activate)
- **Approach A:** add the webhook URL (1 min) → watchdog + failover post to Slack.
- **Approach B:** extend to build/health/eval alerts + daily digest.
- **Effort:** A ≈ 1 min (operator) · B ≈ 2 hr.

### 8. 🥉 Observability triad (§112) for the backend
- **Approach A:** OpenTelemetry → Jaeger (traces) + Prometheus (metrics) + Grafana.
- **Approach B:** lightweight — the existing `/api/health` + a Grafana JSON
  dashboard reading the cron reports.
- **Effort:** A ≈ 1 day · B ≈ 3 hr.

### 9. Frontend up + auth
Frontend `:3003` is down; backend endpoints are open.
- **Approach A:** `npm run dev` + an API-key middleware (global §4.1).
- **Effort:** ≈ 3 hr.

### 10. CI/CD + push
40 commits unpushed; tests/drills run locally only.
- **Approach A:** push + a GitHub Actions workflow (lint + pytest + drills).
- **Approach B:** local pre-push hook running the drill suite.
- **Effort:** ≈ 3 hr.

---

## Priority order (highest leverage first)
1. **#2 GPU-aware routing** (30 min–3 hr) — makes everything faster *today*.
2. **#3 Pre-configure agents** (2 hr) — one command → all 4 editors ready.
3. **#1 MCP tools** (1 day) — the real top-1% leap: agents act on your data.
4. **#5 Failover continuity** (3 hr) — seamless Claude→Ollama handoff.
5. **#6 Drill-gate verification** (3 hr) — trust local output.
6. **#7 Slack webhook** (1 min, operator) — activate the built integration.
7. **#4 RAG** → **#8 Observability** → **#9 Frontend/auth** → **#10 CI/CD**.

## Hardware-honest note
On the GTX 1080 Ti (11 GB), the **top practical model is `qwen2.5-coder:14b`**
(fits in VRAM, fast). `qwen3-coder:30b` works but offloads to CPU — keep it for
"hard reasoning, willing to wait." This is why routing (#2) is priority #1.

## What "top-1%" means here
Not "more extensions" — you already have too many. It's: **one routed model set,
agents that call real project tools via MCP, verified output via drills, seamless
Claude↔Ollama failover with memory, and Slack/observability on the ops loop.**
Connected, not collected.
