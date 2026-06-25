
### [2026-06-25 10:03:29 MDT] ASSISTANT
`[2026-06-25 10:05 MDT]` — Done + pushed (`d7190bc`). The server-down problem is now self-healing.

## ✅ Server uptime — answered + fixed
| Your concern | Solution |
|---|---|
| **How do I know server is down / was down?** | `bash scripts/uptime_status.sh` → NOW status + up/down history. Auto-shown at session start. |
| **Server should not go down** | **WATCHDOG cron (every 2 min)** — if down, auto-restarts + verifies. Self-healing, <2-3 min recovery. |
| **On terminal** | `uptime_status.sh` + `health_check.sh` both terminal, both in SessionStart hook. |
| **More than one server** | Honest: watchdog auto-recovery is the dev-grade answer. True zero-downtime HA needs nginx + 2 instances (noted in §164 as the upgrade path). |
| **Global policy** | **§164** Server Uptime Watchdog — mandatory every project. |

## Where it all is (terminal — even with editor closed)
```bash
bash scripts/uptime_status.sh        # server up? was it down? history
bash scripts/health_check.sh         # frontend/backend/API/DB error counts (live, since restart)
bash scripts/show_requests.sh        # your inputs + status + where-built
bash scripts/show_pending.sh         # full pending backlog
bash scripts/automation_status.py    # plan/crons/count/crash-survival/completion
bash scripts/tracklog.sh             # timestamped event log
```
All five also auto-run at **SessionStart** (you'll see them when you open a session).

## Full automation stack now (all editor-independent, crash-surviving cron)
**18 crons** including: WATCHDOG (*/2, self-heal) · AUTO-BUILD (*/15) · AUTO-PLAN · HEALTH · CONSISTENCY · DRIFT · FAIRNESS · DATA-QUALITY · STATUS-REPORT + the data pipelines.

## Global policies added this session (mandatory, every project)
§158 full enumeration · §159 autonomous completion · §160 unattended build + tracking · §161 input inbox · §162 complete I/O chain · §163 terminal error/health · **§164 server watchdog**

## ⏳ Pending (1 unaddressed input)
- Multiple-server HA (nginx + 2 instance

### OPERATOR
have one agent allocated who can file list of issue which I a mnot aware off and guiding

### [2026-06-25 10:10:33 MDT] ASSISTANT
Push rejected due to GitHub email privacy. That's a config issue, not a code issue. Let me track and report.
