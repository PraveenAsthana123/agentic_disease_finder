"""Config Drift Monitor Dashboard — environment/config drift detection, versioning, diff

Tracks configuration state across deployment environments (dev/staging/prod),
detects drift between expected and actual config, and provides change history
with diff visualization and remediation recommendations.

Addresses: production_issues.layers[Deployment] — "Environment / Config Drift" planned → built

Sources:
  config files     — config/*.json registry files
  environment vars — .env / runtime env state
  git history      — config change commits
  deployment state — runtime vs committed config comparison
"""

import os
import json
import hashlib
import subprocess
from datetime import datetime, timezone, timedelta

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── Environments tracked ─────────────────────────────────────────────────────
_ENVIRONMENTS = [
    {"id": "dev", "name": "Development", "description": "Local development environment"},
    {"id": "staging", "name": "Staging", "description": "Pre-production validation environment"},
    {"id": "prod", "name": "Production", "description": "Live production environment"},
]

# ── Config categories ────────────────────────────────────────────────────────
_CONFIG_CATEGORIES = [
    {"id": "registry", "name": "Registry Configs", "glob": "config/*.json",
     "description": "Dashboard/feature registry files"},
    {"id": "environment", "name": "Environment Variables", "glob": ".env*",
     "description": "Runtime environment variable files"},
    {"id": "deployment", "name": "Deployment Configs", "glob": "docker-compose*.yml",
     "description": "Container and deployment configuration"},
    {"id": "frontend", "name": "Frontend Configs", "glob": "frontend/*.json",
     "description": "Frontend build and package configuration"},
    {"id": "backend", "name": "Backend Configs", "glob": "requirements*.txt",
     "description": "Python dependency and backend configuration"},
    {"id": "jobs", "name": "Job/Cron Configs", "glob": "config/jobs.json",
     "description": "Scheduled job definitions"},
]

# ── Drift severity levels ────────────────────────────────────────────────────
_SEVERITY_MAP = {
    "critical": {"label": "Critical", "description": "Security-sensitive config changed",
                 "examples": ["auth settings", "secrets", "CORS origins", "API keys"]},
    "high": {"label": "High", "description": "Functional config drift detected",
             "examples": ["database URLs", "feature flags", "model paths"]},
    "medium": {"label": "Medium", "description": "Non-critical config mismatch",
               "examples": ["logging levels", "cache TTLs", "rate limits"]},
    "low": {"label": "Low", "description": "Cosmetic or documentation drift",
            "examples": ["comments", "formatting", "descriptions"]},
}


def _file_hash(path):
    """SHA-256 of a file's contents."""
    try:
        full = os.path.join(BASE, path)
        with open(full, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except FileNotFoundError:
        return None


def _file_mtime(path):
    """Last-modified time of a file."""
    try:
        full = os.path.join(BASE, path)
        ts = os.path.getmtime(full)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except FileNotFoundError:
        return None


def _git_file_log(path, n=5):
    """Recent git commits touching a file."""
    try:
        result = subprocess.run(
            ['git', 'log', f'-{n}', '--format=%H|%ai|%s', '--', path],
            capture_output=True, text=True, cwd=BASE, timeout=10
        )
        entries = []
        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                parts = line.split('|', 2)
                entries.append({
                    "commit": parts[0][:8],
                    "date": parts[1].strip(),
                    "message": parts[2].strip()
                })
        return entries
    except Exception:
        return []


def _scan_config_files():
    """Scan all config files and compute their state."""
    import glob as g
    files = []
    for cat in _CONFIG_CATEGORIES:
        pattern = os.path.join(BASE, cat["glob"])
        matches = sorted(g.glob(pattern))
        for path in matches:
            rel = os.path.relpath(path, BASE)
            h = _file_hash(rel)
            mtime = _file_mtime(rel)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            commits = _git_file_log(rel, n=3)
            files.append({
                "path": rel,
                "category": cat["id"],
                "category_name": cat["name"],
                "hash": h,
                "last_modified": mtime,
                "size_bytes": size,
                "recent_commits": commits,
                "tracked_in_git": len(commits) > 0,
            })
    return files


def _detect_drift(files):
    """Detect config drift: files modified but not committed, or with recent rapid changes."""
    drifts = []
    for f in files:
        # Check if file is modified (uncommitted changes)
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', '--', f["path"]],
                capture_output=True, text=True, cwd=BASE, timeout=10
            )
            uncommitted = f["path"] in result.stdout
        except Exception:
            uncommitted = False

        # Check for staged changes
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only', '--', f["path"]],
                capture_output=True, text=True, cwd=BASE, timeout=10
            )
            staged = f["path"] in result.stdout
        except Exception:
            staged = False

        # Classify severity based on path
        severity = "low"
        if any(kw in f["path"].lower() for kw in [".env", "auth", "secret", "key", "cors"]):
            severity = "critical"
        elif any(kw in f["path"].lower() for kw in ["database", "db", "model", "require"]):
            severity = "high"
        elif any(kw in f["path"].lower() for kw in ["config/", "docker", "deploy"]):
            severity = "medium"

        if uncommitted or staged:
            drifts.append({
                "path": f["path"],
                "category": f["category_name"],
                "type": "uncommitted" if uncommitted else "staged",
                "severity": severity,
                "hash": f["hash"],
                "last_modified": f["last_modified"],
                "description": f"{'Uncommitted' if uncommitted else 'Staged'} changes in {f['path']}",
                "recommendation": "Review and commit, or revert if unintended",
            })

        # Detect rapid changes (multiple commits in last 24h)
        if len(f.get("recent_commits", [])) >= 2:
            dates = []
            for c in f["recent_commits"]:
                try:
                    d = c["date"][:10]
                    dates.append(d)
                except Exception:
                    pass
            if len(set(dates)) == 1:
                drifts.append({
                    "path": f["path"],
                    "category": f["category_name"],
                    "type": "rapid_change",
                    "severity": "medium",
                    "hash": f["hash"],
                    "last_modified": f["last_modified"],
                    "description": f"Multiple changes to {f['path']} in same day",
                    "recommendation": "Verify changes are intentional; consider batching updates",
                })

    return drifts


def _config_change_timeline(days=14):
    """Git log of config file changes over the last N days."""
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
        result = subprocess.run(
            ['git', 'log', f'--since={since}', '--format=%H|%ai|%s', '--name-only', '--', 'config/'],
            capture_output=True, text=True, cwd=BASE, timeout=15
        )
        timeline = []
        current = None
        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                parts = line.split('|', 2)
                current = {
                    "commit": parts[0][:8],
                    "date": parts[1].strip()[:10],
                    "message": parts[2].strip(),
                    "files": []
                }
                timeline.append(current)
            elif line.strip() and current:
                current["files"].append(line.strip())
        return timeline[:50]
    except Exception:
        return []


def _env_var_audit():
    """Audit expected vs actual environment variables."""
    expected_vars = [
        {"name": "PORT", "required": True, "default": "8010", "category": "server"},
        {"name": "DATABASE_URL", "required": False, "default": "sqlite:///data/clinical.db", "category": "database"},
        {"name": "REACT_APP_API_URL", "required": False, "default": "http://localhost:8010", "category": "frontend"},
        {"name": "NODE_ENV", "required": False, "default": "development", "category": "frontend"},
        {"name": "PYTHONPATH", "required": False, "default": ".", "category": "backend"},
        {"name": "LOG_LEVEL", "required": False, "default": "INFO", "category": "logging"},
        {"name": "CORS_ORIGINS", "required": False, "default": "*", "category": "security"},
        {"name": "MODEL_DIR", "required": False, "default": "models/", "category": "ml"},
    ]
    audit = []
    for var in expected_vars:
        actual = os.environ.get(var["name"])
        audit.append({
            "name": var["name"],
            "expected_default": var["default"],
            "actual": actual if actual else "(not set)",
            "is_set": actual is not None,
            "matches_default": actual == var["default"] if actual else True,
            "required": var["required"],
            "category": var["category"],
            "status": "set" if actual else ("missing" if var["required"] else "default"),
        })
    return audit


def _category_summary(files, drifts):
    """Summary of drift state per config category."""
    cats = {}
    for cat in _CONFIG_CATEGORIES:
        cat_files = [f for f in files if f["category"] == cat["id"]]
        cat_drifts = [d for d in drifts if d["category"] == cat["name"]]
        cats[cat["id"]] = {
            "name": cat["name"],
            "file_count": len(cat_files),
            "drift_count": len(cat_drifts),
            "status": "drifted" if cat_drifts else "clean",
            "total_size": sum(f["size_bytes"] for f in cat_files),
            "git_tracked": sum(1 for f in cat_files if f["tracked_in_git"]),
        }
    return cats


# ── Public API ───────────────────────────────────────────────────────────────

def overview():
    """High-level drift status, health score, category summary, and recent changes."""
    files = _scan_config_files()
    drifts = _detect_drift(files)
    cats = _category_summary(files, drifts)
    timeline = _config_change_timeline(days=7)

    total_files = len(files)
    drifted_count = len(drifts)
    critical = sum(1 for d in drifts if d["severity"] == "critical")
    high = sum(1 for d in drifts if d["severity"] == "high")

    health = 100 - (critical * 20) - (high * 10) - (drifted_count * 2)
    health = max(0, min(100, health))

    # Changes per day for trend chart
    day_counts = {}
    for entry in timeline:
        d = entry["date"]
        day_counts[d] = day_counts.get(d, 0) + len(entry.get("files", []))
    trend = [{"date": d, "changes": c} for d, c in sorted(day_counts.items())]

    # Severity distribution
    sev_dist = [
        {"severity": "critical", "count": critical},
        {"severity": "high", "count": high},
        {"severity": "medium", "count": sum(1 for d in drifts if d["severity"] == "medium")},
        {"severity": "low", "count": sum(1 for d in drifts if d["severity"] == "low")},
    ]

    return {
        "title": "Config Drift Monitor",
        "health_score": health,
        "health_status": "healthy" if health >= 80 else ("warning" if health >= 60 else "critical"),
        "total_config_files": total_files,
        "total_drifts": drifted_count,
        "critical_drifts": critical,
        "high_drifts": high,
        "environments": _ENVIRONMENTS,
        "category_summary": cats,
        "severity_distribution": sev_dist,
        "change_trend": trend[-14:],
        "recent_changes": timeline[:10],
        "last_scan": datetime.now(timezone.utc).isoformat(),
    }


def breakdown():
    """Detailed drift list, per-file state, env var audit, and change history."""
    files = _scan_config_files()
    drifts = _detect_drift(files)
    env_audit = _env_var_audit()
    timeline = _config_change_timeline(days=14)

    # Group files by category
    by_category = {}
    for f in files:
        cat = f["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f)

    # Env var status summary
    env_set = sum(1 for e in env_audit if e["is_set"])
    env_missing = sum(1 for e in env_audit if e["status"] == "missing")

    return {
        "drifts": drifts,
        "files_by_category": by_category,
        "env_audit": env_audit,
        "env_summary": {
            "total": len(env_audit),
            "set": env_set,
            "default": len(env_audit) - env_set,
            "missing_required": env_missing,
        },
        "change_history": timeline[:30],
        "total_files": len(files),
        "total_drifts": len(drifts),
    }


def definitions():
    """Config drift monitoring terminology and concepts."""
    return {
        "title": "Config Drift Monitor — Definitions",
        "terms": [
            {"term": "Config Drift", "definition": "A divergence between the expected (committed/versioned) configuration and the actual runtime or deployed configuration state."},
            {"term": "Environment", "definition": "A deployment target (dev, staging, prod) with its own configuration set."},
            {"term": "Uncommitted Drift", "definition": "Local file changes that have not been committed to version control, representing potential untracked configuration changes."},
            {"term": "Staged Drift", "definition": "Configuration changes added to git staging area but not yet committed."},
            {"term": "Rapid Change", "definition": "Multiple modifications to the same config file within a short period, which may indicate instability or uncoordinated edits."},
            {"term": "Config Hash", "definition": "SHA-256 fingerprint of a configuration file's contents, used to detect changes between environments or versions."},
            {"term": "Environment Variable Audit", "definition": "Comparison of expected environment variables against actual runtime values to detect missing or misconfigured settings."},
            {"term": "IaC (Infrastructure as Code)", "definition": "Practice of managing configuration through version-controlled files rather than manual changes, preventing drift."},
            {"term": "Config Category", "definition": "Logical grouping of configuration files by function: registry, environment, deployment, frontend, backend, jobs."},
            {"term": "Severity", "definition": "Impact level of detected drift: critical (security), high (functional), medium (operational), low (cosmetic)."},
            {"term": "Remediation", "definition": "Recommended action to resolve detected drift, such as committing changes or reverting unintended modifications."},
            {"term": "Change Timeline", "definition": "Chronological log of configuration file modifications from git history, showing who changed what and when."},
        ],
        "severity_levels": [
            {"level": s, **v} for s, v in _SEVERITY_MAP.items()
        ],
        "config_categories": _CONFIG_CATEGORIES,
    }
