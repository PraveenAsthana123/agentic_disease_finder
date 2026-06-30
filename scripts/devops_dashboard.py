"""DevOps / CI-CD Dashboard — real git history analytics, deploy frequency,
change-fail rate, MTTR, pipeline/cron status, and commit velocity from the
actual repository and cron job infrastructure."""

import subprocess
import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRON_DIR = ROOT / "jobs" / "crons"
REPORTS_DIR = ROOT / "jobs" / "reports"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(cmd, cwd=None):
    """Run a shell command and return stdout lines."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15,
            cwd=cwd or str(ROOT)
        )
        return r.stdout.strip().split("\n") if r.stdout.strip() else []
    except Exception:
        return []


def devops_overview():
    """Deploy frequency, change-fail rate, MTTR, commit velocity — all from real git data."""

    # --- Commit history (last 90 days) ---
    since = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    log_lines = _run(f'git log --since="{since}" --format="%H|%aI|%s" --no-merges')
    commits = []
    for line in log_lines:
        if not line or "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        commits.append({"hash": parts[0][:8], "date": parts[1], "subject": parts[2]})

    total_commits = len(commits)

    # --- Daily commit frequency (last 30 days) ---
    since_30 = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    daily_lines = _run(f'git log --since="{since_30}" --format="%aI" --no-merges')
    daily_counts = {}
    for line in daily_lines:
        if not line:
            continue
        day = line[:10]
        daily_counts[day] = daily_counts.get(day, 0) + 1

    # Build 30-day series (fill missing days with 0)
    commit_velocity = []
    base = datetime.now(timezone.utc).date()
    for i in range(29, -1, -1):
        d = (base - timedelta(days=i)).isoformat()
        commit_velocity.append({"date": d, "commits": daily_counts.get(d, 0)})

    avg_daily = round(sum(c["commits"] for c in commit_velocity) / max(len(commit_velocity), 1), 1)

    # --- Deploy frequency (commits with "feat:", "fix:", "deploy", "release") ---
    deploy_keywords = re.compile(r"^(feat|fix|deploy|release|hotfix)", re.IGNORECASE)
    deploys = [c for c in commits if deploy_keywords.search(c["subject"])]
    deploy_count_90d = len(deploys)
    deploy_freq_per_day = round(deploy_count_90d / 90, 2)

    # --- Change-fail rate (commits with "fix:", "revert", "hotfix" / total) ---
    fix_keywords = re.compile(r"^(fix|revert|hotfix|bug)", re.IGNORECASE)
    fixes = [c for c in commits if fix_keywords.search(c["subject"])]
    change_fail_rate = round(len(fixes) / max(total_commits, 1) * 100, 1)

    # --- MTTR proxy: avg time between a fix and the previous non-fix commit ---
    mttr_minutes = None
    if len(commits) >= 2:
        fix_times = []
        for i, c in enumerate(commits):
            if fix_keywords.search(c["subject"]) and i + 1 < len(commits):
                try:
                    t_fix = datetime.fromisoformat(c["date"].replace("Z", "+00:00"))
                    t_prev = datetime.fromisoformat(commits[i + 1]["date"].replace("Z", "+00:00"))
                    delta = abs((t_fix - t_prev).total_seconds()) / 60
                    if delta < 10080:  # cap at 7 days
                        fix_times.append(delta)
                except Exception:
                    pass
        if fix_times:
            mttr_minutes = round(sum(fix_times) / len(fix_times), 0)

    # --- Top contributors (last 90 days) ---
    author_lines = _run(f'git log --since="{since}" --format="%aN" --no-merges')
    author_counts = {}
    for a in author_lines:
        if a:
            author_counts[a] = author_counts.get(a, 0) + 1
    top_authors = sorted(
        [{"name": k, "commits": v} for k, v in author_counts.items()],
        key=lambda x: -x["commits"]
    )[:10]

    # --- Commit type breakdown ---
    type_counts = {"feat": 0, "fix": 0, "refactor": 0, "docs": 0, "chore": 0, "test": 0, "other": 0}
    for c in commits:
        subj = c["subject"].lower()
        matched = False
        for t in ["feat", "fix", "refactor", "docs", "chore", "test"]:
            if subj.startswith(t):
                type_counts[t] += 1
                matched = True
                break
        if not matched:
            type_counts["other"] += 1
    commit_types = [{"type": k, "count": v} for k, v in type_counts.items() if v > 0]

    # --- Branch info ---
    branch_lines = _run("git branch --no-color")
    branches = [b.strip().lstrip("* ") for b in branch_lines if b.strip()]
    current_branch_lines = _run("git branch --show-current")
    current_branch = current_branch_lines[0] if current_branch_lines else "unknown"

    # --- Commits ahead of remote ---
    ahead_lines = _run("git rev-list @{u}..HEAD --count 2>/dev/null")
    commits_ahead = int(ahead_lines[0]) if ahead_lines and ahead_lines[0].isdigit() else 0

    # --- Files changed in last 30 days ---
    changed_files_lines = _run(f'git log --since="{since_30}" --name-only --format="" --no-merges')
    unique_files = set(f for f in changed_files_lines if f)
    files_changed_30d = len(unique_files)

    # --- Hottest files (most frequently changed) ---
    file_freq = {}
    for f in changed_files_lines:
        if f:
            file_freq[f] = file_freq.get(f, 0) + 1
    hottest_files = sorted(
        [{"file": k, "changes": v} for k, v in file_freq.items()],
        key=lambda x: -x["changes"]
    )[:10]

    return {
        "available": True,
        "generated_at": _now(),
        "period_days": 90,
        "summary": {
            "total_commits_90d": total_commits,
            "deploy_count_90d": deploy_count_90d,
            "deploy_freq_per_day": deploy_freq_per_day,
            "change_fail_rate_pct": change_fail_rate,
            "mttr_minutes": mttr_minutes,
            "avg_daily_commits": avg_daily,
            "current_branch": current_branch,
            "branches": len(branches),
            "commits_ahead": commits_ahead,
            "files_changed_30d": files_changed_30d,
        },
        "commit_velocity": commit_velocity,
        "commit_types": commit_types,
        "top_authors": top_authors,
        "recent_deploys": deploys[:15],
        "hottest_files": hottest_files,
    }


def devops_pipelines():
    """Cron job / pipeline status — real cron definitions from jobs/crons/."""
    pipelines = []

    # Read cron definitions
    cron_dir = CRON_DIR
    if cron_dir.exists():
        for f in sorted(cron_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                pipelines.append({
                    "name": data.get("name", f.stem),
                    "schedule": data.get("schedule", data.get("cron", "unknown")),
                    "enabled": data.get("enabled", True),
                    "last_run": data.get("last_run"),
                    "status": data.get("status", "unknown"),
                    "type": "cron",
                })
            except Exception:
                pipelines.append({"name": f.stem, "schedule": "parse error", "enabled": False, "type": "cron"})

    # Also check for running jobs via PID files
    pid_dir = ROOT / "jobs" / "pids"
    running_pids = []
    if pid_dir.exists():
        for pf in pid_dir.glob("*.pid"):
            try:
                pid = int(pf.read_text().strip())
                # Check if process is alive
                os.kill(pid, 0)
                running_pids.append({"name": pf.stem, "pid": pid, "alive": True})
            except (ProcessLookupError, ValueError, PermissionError):
                running_pids.append({"name": pf.stem, "pid": None, "alive": False})

    # Health report if exists
    health = {}
    health_file = REPORTS_DIR / "health_latest.json"
    if health_file.exists():
        try:
            health = json.loads(health_file.read_text())
        except Exception:
            pass

    return {
        "available": True,
        "generated_at": _now(),
        "pipelines": pipelines,
        "running_jobs": running_pids,
        "total_pipelines": len(pipelines),
        "enabled": sum(1 for p in pipelines if p.get("enabled")),
        "disabled": sum(1 for p in pipelines if not p.get("enabled")),
        "health_snapshot": {
            "api_status": health.get("api_status", "unknown"),
            "db_status": health.get("db_status", "unknown"),
            "endpoints_ok": health.get("endpoints_ok", 0),
            "endpoints_total": health.get("endpoints_total", 0),
        },
    }


def devops_definitions():
    """Metric definitions for the DevOps/CI-CD dashboard."""
    return {
        "available": True,
        "definitions": [
            {
                "term": "Deploy Frequency",
                "definition": "Number of feat/fix/deploy/release commits per day over the last 90 days. Higher = faster delivery cadence.",
                "source": "git log --since 90d"
            },
            {
                "term": "Change Fail Rate",
                "definition": "Percentage of commits that are fixes, reverts, or hotfixes out of total commits. Lower = more stable changes. DORA metric.",
                "source": "git log subject prefix analysis"
            },
            {
                "term": "MTTR (Mean Time to Restore)",
                "definition": "Average minutes between a fix commit and its preceding non-fix commit. Proxy for how fast issues are resolved. DORA metric.",
                "source": "git log timestamp diff between fix and prior commit"
            },
            {
                "term": "Commit Velocity",
                "definition": "Daily commit count over the last 30 days. Shows development pace and activity trends.",
                "source": "git log --since 30d grouped by date"
            },
            {
                "term": "Commit Types",
                "definition": "Breakdown of commits by conventional-commit prefix (feat, fix, refactor, docs, chore, test). Shows where effort is spent.",
                "source": "git log subject prefix parsing"
            },
            {
                "term": "Hottest Files",
                "definition": "Files most frequently changed in the last 30 days. High-churn files may need refactoring or better tests.",
                "source": "git log --name-only frequency count"
            },
            {
                "term": "Pipeline / Cron Status",
                "definition": "Status of scheduled automation jobs (crons) — enabled, running, last execution time.",
                "source": "jobs/crons/*.json + jobs/pids/*.pid"
            },
            {
                "term": "Commits Ahead",
                "definition": "Number of local commits not yet pushed to the remote tracking branch.",
                "source": "git rev-list @{u}..HEAD --count"
            },
        ]
    }
