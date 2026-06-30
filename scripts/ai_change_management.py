"""
AI Change Management — parses real git history + track.jsonl + uptime.jsonl
to produce overview / breakdown / definitions for the Change Management dashboard.

Stages: change request → impact analysis → approval → deploy → rollback
Data sources:
  - git log: commits as change requests (type, impact, files changed)
  - track.jsonl: deploy/autobuild events
  - uptime.jsonl: rollback triggers (DOWN events after deploys)
"""

import json, os, pathlib, subprocess
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict

MDT = timezone(timedelta(hours=-6))
BASE = pathlib.Path(__file__).resolve().parent.parent
TRACK_LOG = BASE / "jobs" / "logs" / "track.jsonl"
UPTIME_LOG = BASE / "jobs" / "logs" / "uptime.jsonl"


def _parse_ts(ts_str: str) -> datetime:
    ts_str = ts_str.strip()
    if ts_str.endswith(" MDT"):
        ts_str = ts_str[:-4]
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=MDT)
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def _load_git_log(max_commits=200):
    """Load recent git commits with stats."""
    commits = []
    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={max_commits}",
             "--format=%H|%ai|%an|%s", "--shortstat"],
            capture_output=True, text=True, cwd=str(BASE), timeout=15
        )
        lines = result.stdout.strip().split("\n")
        current = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "|" in line and len(line.split("|")) >= 4:
                parts = line.split("|", 3)
                if len(parts[0]) == 40:
                    if current:
                        commits.append(current)
                    try:
                        dt = datetime.fromisoformat(parts[1].strip())
                    except (ValueError, TypeError):
                        dt = datetime.now(MDT)
                    current = {
                        "hash": parts[0][:8],
                        "date": dt,
                        "author": parts[2].strip(),
                        "message": parts[3].strip(),
                        "files_changed": 0,
                        "insertions": 0,
                        "deletions": 0,
                    }
            elif current and ("file" in line or "insertion" in line or "deletion" in line):
                # parse shortstat: " 3 files changed, 120 insertions(+), 5 deletions(-)"
                for token in line.replace(",", "").split():
                    if token.isdigit():
                        num = int(token)
                    elif token.startswith("file"):
                        current["files_changed"] = num
                    elif token.startswith("insertion"):
                        current["insertions"] = num
                    elif token.startswith("deletion"):
                        current["deletions"] = num
        if current:
            commits.append(current)
    except Exception:
        pass
    return commits


def _load_track():
    events = []
    if not TRACK_LOG.exists():
        return events
    for line in TRACK_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "ts_utc" in obj:
                obj["_dt"] = datetime.fromisoformat(obj["ts_utc"].replace("Z", "+00:00"))
            elif "ts_local" in obj:
                obj["_dt"] = _parse_ts(obj["ts_local"])
            events.append(obj)
        except Exception:
            pass
    return events


def _load_uptime():
    events = []
    if not UPTIME_LOG.exists():
        return events
    for line in UPTIME_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            obj["_dt"] = _parse_ts(obj["ts"])
            events.append(obj)
        except Exception:
            pass
    return events


def _classify_change(msg: str):
    """Classify commit type from message prefix."""
    msg_lower = msg.lower()
    if msg_lower.startswith("feat"):
        return "feature"
    if msg_lower.startswith("fix"):
        return "bugfix"
    if msg_lower.startswith("refactor"):
        return "refactor"
    if msg_lower.startswith("docs") or msg_lower.startswith("doc:"):
        return "documentation"
    if msg_lower.startswith("test"):
        return "test"
    if msg_lower.startswith("chore") or msg_lower.startswith("ci"):
        return "infrastructure"
    if msg_lower.startswith("style"):
        return "style"
    if msg_lower.startswith("perf"):
        return "performance"
    return "other"


def _assess_risk(files_changed, insertions, deletions):
    """Assess change risk based on scope."""
    total_lines = insertions + deletions
    if files_changed >= 20 or total_lines >= 500:
        return "high"
    if files_changed >= 5 or total_lines >= 100:
        return "medium"
    return "low"


# ── overview ─────────────────────────────────────────────────────
def overview():
    commits = _load_git_log(200)
    track = _load_track()
    uptime = _load_uptime()
    now = datetime.now(MDT)

    total_changes = len(commits)

    # change type distribution
    type_counts = Counter()
    for c in commits:
        type_counts[_classify_change(c["message"])] += 1
    change_type_distribution = [
        {"type": k, "count": v, "percent": round(v / total_changes * 100, 1) if total_changes else 0}
        for k, v in type_counts.most_common()
    ]

    # risk distribution
    risk_counts = Counter()
    for c in commits:
        risk_counts[_assess_risk(c["files_changed"], c["insertions"], c["deletions"])] += 1
    risk_distribution = [
        {"level": k, "count": v, "percent": round(v / total_changes * 100, 1) if total_changes else 0}
        for k, v in sorted(risk_counts.items(), key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x[0], 3))
    ]

    # deploy events from track
    deploy_events = [t for t in track if t.get("level") in ("autobuild", "deploy", "git")]
    total_deploys = len(deploy_events)

    # rollback indicators: DOWN events from uptime
    downs = [e for e in uptime if e.get("event") == "DOWN"]
    rollback_count = len(downs)

    # success rate: deploys that weren't immediately followed by DOWN
    deploy_success_count = total_deploys - rollback_count if total_deploys > rollback_count else max(total_deploys - rollback_count, 0)
    deploy_success_rate = round(deploy_success_count / total_deploys * 100, 1) if total_deploys else 100.0

    # avg files per change
    avg_files = round(sum(c["files_changed"] for c in commits) / len(commits), 1) if commits else 0
    avg_lines = round(sum(c["insertions"] + c["deletions"] for c in commits) / len(commits), 0) if commits else 0

    # changes in last 24h / 7d
    changes_24h = 0
    changes_7d = 0
    for c in commits:
        dt = c["date"]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=MDT)
        delta = (now - dt).total_seconds()
        if delta < 86400:
            changes_24h += 1
        if delta < 86400 * 7:
            changes_7d += 1

    # daily change counts – last 14 days
    daily = {}
    for i in range(14):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        daily[day] = 0
    for c in commits:
        dt = c["date"]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=MDT)
        day = dt.astimezone(MDT).strftime("%Y-%m-%d")
        if day in daily:
            daily[day] += 1
    daily_change_counts = [{"date": k, "count": v} for k, v in sorted(daily.items())]

    # top contributors
    author_counts = Counter(c["author"] for c in commits)
    top_contributors = [{"author": k, "commits": v} for k, v in author_counts.most_common(10)]

    # recent changes (last 15)
    recent = []
    for c in commits[:15]:
        recent.append({
            "hash": c["hash"],
            "date": c["date"].isoformat() if hasattr(c["date"], "isoformat") else str(c["date"]),
            "author": c["author"],
            "message": c["message"][:120],
            "type": _classify_change(c["message"]),
            "risk": _assess_risk(c["files_changed"], c["insertions"], c["deletions"]),
            "files_changed": c["files_changed"],
            "lines_changed": c["insertions"] + c["deletions"],
        })

    return {
        "available": True,
        "total_changes": total_changes,
        "total_deploys": total_deploys,
        "rollback_count": rollback_count,
        "deploy_success_rate": deploy_success_rate,
        "avg_files_per_change": avg_files,
        "avg_lines_per_change": avg_lines,
        "changes_last_24h": changes_24h,
        "changes_last_7d": changes_7d,
        "change_type_distribution": change_type_distribution,
        "risk_distribution": risk_distribution,
        "daily_change_counts": daily_change_counts,
        "top_contributors": top_contributors,
        "recent_changes": recent,
    }


# ── breakdown ────────────────────────────────────────────────────
def breakdown():
    commits = _load_git_log(200)
    track = _load_track()
    uptime = _load_uptime()
    now = datetime.now(MDT)

    # impact analysis: files changed per change type
    type_impact = defaultdict(lambda: {"files": 0, "insertions": 0, "deletions": 0, "count": 0})
    for c in commits:
        ct = _classify_change(c["message"])
        type_impact[ct]["files"] += c["files_changed"]
        type_impact[ct]["insertions"] += c["insertions"]
        type_impact[ct]["deletions"] += c["deletions"]
        type_impact[ct]["count"] += 1
    impact_by_type = []
    for ct, d in sorted(type_impact.items(), key=lambda x: -x[1]["count"]):
        impact_by_type.append({
            "type": ct,
            "changes": d["count"],
            "total_files": d["files"],
            "avg_files": round(d["files"] / d["count"], 1) if d["count"] else 0,
            "total_insertions": d["insertions"],
            "total_deletions": d["deletions"],
            "avg_lines": round((d["insertions"] + d["deletions"]) / d["count"], 0) if d["count"] else 0,
        })

    # hourly heatmap: 7 days x 24 hours for commits
    heatmap = [[0] * 24 for _ in range(7)]
    day_labels = []
    for i in range(6, -1, -1):
        d = now - timedelta(days=i)
        day_labels.append(d.strftime("%a %m/%d"))
    for c in commits:
        dt = c["date"]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=MDT)
        dt = dt.astimezone(MDT)
        day_offset = (now.date() - dt.date()).days
        if 0 <= day_offset <= 6:
            row_idx = 6 - day_offset
            heatmap[row_idx][dt.hour] += 1
    hourly_heatmap = {
        "day_labels": day_labels,
        "hour_labels": [f"{h:02d}" for h in range(24)],
        "matrix": heatmap,
    }

    # deploy timeline from track (autobuild/deploy/git events)
    deploy_levels = {"autobuild", "deploy", "git"}
    deploy_events = sorted(
        [t for t in track if t.get("level") in deploy_levels],
        key=lambda t: t.get("_dt", datetime.min.replace(tzinfo=MDT)),
        reverse=True
    )
    deploy_timeline = []
    for t in deploy_events[:30]:
        deploy_timeline.append({
            "ts": t.get("ts_local", t.get("ts_utc", "")),
            "level": t.get("level", ""),
            "event": t.get("event", ""),
            "host": t.get("host", ""),
        })

    # risk trend: risk distribution per day (last 14 days)
    risk_trend = []
    for i in range(13, -1, -1):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        day_commits = []
        for c in commits:
            dt = c["date"]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=MDT)
            if dt.astimezone(MDT).strftime("%Y-%m-%d") == day:
                day_commits.append(c)
        risk_day = {"date": day, "total": len(day_commits), "high": 0, "medium": 0, "low": 0}
        for c in day_commits:
            r = _assess_risk(c["files_changed"], c["insertions"], c["deletions"])
            risk_day[r] += 1
        risk_trend.append(risk_day)

    # change velocity: cumulative changes over time (last 14 days)
    velocity = []
    cumulative = 0
    for rt in risk_trend:
        cumulative += rt["total"]
        velocity.append({"date": rt["date"], "daily": rt["total"], "cumulative": cumulative})

    # rollback events from uptime (DOWN followed by AUTO-RESTARTED)
    sorted_uptime = sorted(uptime, key=lambda e: e["_dt"])
    rollback_events = []
    for i, ev in enumerate(sorted_uptime):
        if ev.get("event") != "DOWN":
            continue
        recovery_ts = None
        for j in range(i + 1, len(sorted_uptime)):
            if sorted_uptime[j].get("event") in ("AUTO-RESTARTED", "RECOVERED"):
                recovery_ts = sorted_uptime[j]["ts"]
                break
        rollback_events.append({
            "ts": ev["ts"],
            "http": ev.get("http", ""),
            "recovery_ts": recovery_ts,
        })
    rollback_events = rollback_events[-20:]  # last 20

    return {
        "available": True,
        "impact_by_type": impact_by_type,
        "hourly_heatmap": hourly_heatmap,
        "deploy_timeline": deploy_timeline,
        "risk_trend": risk_trend,
        "change_velocity": velocity,
        "rollback_events": rollback_events,
    }


# ── definitions ──────────────────────────────────────────────────
def definitions():
    return {
        "available": True,
        "stages": [
            {"stage": "Change Request", "description": "A proposed modification to the codebase, captured as a git commit. Each commit represents a discrete, reviewed change with author, timestamp, and impact metrics."},
            {"stage": "Impact Analysis", "description": "Assessment of change scope: files modified, lines inserted/deleted, and risk level (low/medium/high) based on change magnitude."},
            {"stage": "Approval", "description": "Change accepted into the main branch. In this project, approval is implicit via commit (single-operator model)."},
            {"stage": "Deploy", "description": "Automated deployment via autobuild pipeline or manual restart. Tracked via track.jsonl events with level=autobuild/deploy/git."},
            {"stage": "Rollback", "description": "System recovery after a deployment causes a DOWN event. Detected from uptime.jsonl DOWN→AUTO-RESTARTED pairs."},
        ],
        "metrics": [
            {"term": "Total Changes", "definition": "Number of git commits in the analysis window (last 200 commits)."},
            {"term": "Total Deploys", "definition": "Count of autobuild, deploy, and git events in track.jsonl representing code deployments."},
            {"term": "Rollback Count", "definition": "Number of DOWN events from uptime.jsonl, each representing a deployment that required system recovery."},
            {"term": "Deploy Success Rate", "definition": "Percentage of deployments not immediately followed by a system DOWN event. Higher is better."},
            {"term": "Change Type", "definition": "Classification based on commit message prefix: feat→feature, fix→bugfix, refactor, docs→documentation, test, chore/ci→infrastructure, perf→performance, other."},
            {"term": "Risk Level", "definition": "Impact assessment: LOW (<5 files, <100 lines), MEDIUM (5-19 files or 100-499 lines), HIGH (≥20 files or ≥500 lines)."},
            {"term": "Avg Files/Change", "definition": "Mean number of files modified per commit, indicating change granularity."},
            {"term": "Avg Lines/Change", "definition": "Mean total lines (insertions + deletions) per commit, indicating change magnitude."},
            {"term": "Change Velocity", "definition": "Cumulative and daily commit counts over time, showing development pace and rhythm."},
            {"term": "Impact by Type", "definition": "Aggregate file and line changes grouped by change type, showing which categories produce the most code churn."},
            {"term": "Hourly Heatmap", "definition": "7-day × 24-hour matrix of commit counts, showing when development activity concentrates."},
            {"term": "Deploy Timeline", "definition": "Chronological log of deployment-related events from the system tracker."},
            {"term": "Risk Trend", "definition": "Daily breakdown of high/medium/low risk changes over the last 14 days."},
        ],
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN ===")
    pprint.pprint(breakdown())
