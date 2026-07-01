"""
AI FinOps Dashboard — real cost tracking from track.jsonl, model files,
git history, and system resource data.

Tracks: GPU/compute hours, model inference costs, agent/autobuild runs,
storage costs per model, daily cost trends, cost-per-build metrics.

Data sources:
  - track.jsonl: autobuild/ops/health/git events (real timestamps, durations)
  - models/*.joblib: real model file sizes → storage cost
  - git log: commit frequency → compute proxy
  - uptime.jsonl: downtime events → wasted compute
"""

import json, os, pathlib, subprocess
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict

MDT = timezone(timedelta(hours=-6))
BASE = pathlib.Path(__file__).resolve().parent.parent
TRACK_LOG = BASE / "jobs" / "logs" / "track.jsonl"
UPTIME_LOG = BASE / "jobs" / "logs" / "uptime.jsonl"
MODELS_DIR = BASE / "models"

# ── Cost rates (realistic cloud pricing) ─────────────────────────
# GPU: T4 on-demand ~$0.35/hr, A100 ~$3.00/hr; we use T4-equivalent
GPU_RATE_PER_HOUR = 0.35
# Model inference: ~$0.002 per prediction (EEG feature extraction + ML)
INFERENCE_COST = 0.002
# Agent/autobuild: Claude API ~$0.015/1K tokens, avg build ~8K tokens
AGENT_COST_PER_RUN = 0.12
# Storage: S3-equivalent $0.023/GB/month
STORAGE_RATE_GB_MONTH = 0.023
# Health check: minimal cost per check
HEALTH_CHECK_COST = 0.001


def _load_track_events():
    """Load all track.jsonl events."""
    events = []
    if not TRACK_LOG.exists():
        return events
    with open(TRACK_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                ts_str = evt.get("ts_local", "")
                if ts_str.endswith(" MDT"):
                    ts_str = ts_str[:-4]
                try:
                    evt["_dt"] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=MDT)
                except (ValueError, TypeError):
                    evt["_dt"] = datetime.now(MDT)
                events.append(evt)
            except json.JSONDecodeError:
                continue
    return events


def _load_uptime_events():
    """Load uptime.jsonl events."""
    events = []
    if not UPTIME_LOG.exists():
        return events
    with open(UPTIME_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _get_model_storage():
    """Calculate real model storage costs from .joblib files."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for fp in sorted(MODELS_DIR.glob("*.joblib")):
        size_bytes = fp.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_mb / 1024
        monthly_cost = size_gb * STORAGE_RATE_GB_MONTH
        models.append({
            "name": fp.stem.replace("_model", "").replace("_", " ").title(),
            "file": fp.name,
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "monthly_cost": round(monthly_cost, 6),
        })
    return models


def _compute_build_sessions(events):
    """Pair autobuild START/END events to compute session durations."""
    sessions = []
    starts = []
    for evt in events:
        if evt.get("level") == "autobuild":
            msg = evt.get("event", "")
            if "START" in msg:
                starts.append(evt)
            elif "END" in msg:
                if starts:
                    start = starts.pop(0)
                    duration_sec = (evt["_dt"] - start["_dt"]).total_seconds()
                    if duration_sec < 0:
                        duration_sec = 0
                    sessions.append({
                        "start": start["_dt"],
                        "end": evt["_dt"],
                        "duration_min": round(duration_sec / 60, 1),
                        "cost": round(duration_sec / 3600 * GPU_RATE_PER_HOUR + AGENT_COST_PER_RUN, 4),
                    })
    return sessions


def overview():
    """Overview KPIs, cost breakdown by category, daily cost trend, model storage."""
    events = _load_track_events()
    if not events:
        return {"available": False}

    now = datetime.now(MDT)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)

    # Count events by level
    level_counts = Counter(e.get("level", "unknown") for e in events)

    # Build sessions
    sessions = _compute_build_sessions(events)
    total_build_minutes = sum(s["duration_min"] for s in sessions)
    total_build_cost = sum(s["cost"] for s in sessions)

    # Autobuild events with "built+" = successful builds
    successful_builds = [e for e in events
                         if e.get("level") == "autobuild" and "built+" in e.get("event", "")]

    # Health checks
    health_events = [e for e in events if e.get("level") == "health"]
    health_cost = len(health_events) * HEALTH_CHECK_COST

    # Watchdog events
    watchdog_events = [e for e in events if e.get("level") == "watchdog"]
    watchdog_cost = len(watchdog_events) * HEALTH_CHECK_COST * 0.5

    # Git pushes
    git_events = [e for e in events if e.get("level") == "git"]
    git_cost = len(git_events) * 0.005  # minimal CI trigger cost

    # Ops events (restarts etc.)
    ops_events = [e for e in events if e.get("level") == "ops"]
    ops_cost = len(ops_events) * 0.01

    # Ollama inference
    ollama_events = [e for e in events if e.get("level") == "ollama"]
    ollama_cost = len(ollama_events) * 0.05  # local inference, electricity only

    # Model storage
    model_storage = _get_model_storage()
    total_storage_mb = sum(m["size_mb"] for m in model_storage)
    total_storage_cost = sum(m["monthly_cost"] for m in model_storage)

    # Total cost
    total_cost = (total_build_cost + health_cost + watchdog_cost +
                  git_cost + ops_cost + ollama_cost + total_storage_cost)

    # Cost breakdown by category
    cost_breakdown = [
        {"category": "Autobuild (GPU + Agent)", "cost": round(total_build_cost, 2),
         "events": len(sessions), "percent": 0},
        {"category": "Health Monitoring", "cost": round(health_cost, 2),
         "events": len(health_events), "percent": 0},
        {"category": "Watchdog", "cost": round(watchdog_cost, 2),
         "events": len(watchdog_events), "percent": 0},
        {"category": "Git/CI Triggers", "cost": round(git_cost, 2),
         "events": len(git_events), "percent": 0},
        {"category": "Ops (Restarts)", "cost": round(ops_cost, 2),
         "events": len(ops_events), "percent": 0},
        {"category": "Ollama Inference", "cost": round(ollama_cost, 2),
         "events": len(ollama_events), "percent": 0},
        {"category": "Model Storage", "cost": round(total_storage_cost, 4),
         "events": len(model_storage), "percent": 0},
    ]
    for c in cost_breakdown:
        c["percent"] = round(c["cost"] / total_cost * 100, 1) if total_cost > 0 else 0

    # Daily cost trend (last 14 days)
    daily_costs = []
    for d in range(13, -1, -1):
        day = (now - timedelta(days=d)).date()
        day_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=MDT)
        day_end = day_start + timedelta(days=1)
        day_sessions = [s for s in sessions if day_start <= s["start"] < day_end]
        day_health = len([e for e in health_events if day_start <= e["_dt"] < day_end])
        day_ops = len([e for e in ops_events if day_start <= e["_dt"] < day_end])
        day_git = len([e for e in git_events if day_start <= e["_dt"] < day_end])
        dc = (sum(s["cost"] for s in day_sessions) +
              day_health * HEALTH_CHECK_COST +
              day_ops * 0.01 + day_git * 0.005)
        daily_costs.append({
            "date": str(day),
            "cost": round(dc, 2),
            "builds": len(day_sessions),
            "health_checks": day_health,
        })

    # Recent 24h / 7d costs
    recent_24h = sum(d["cost"] for d in daily_costs[-1:])
    recent_7d = sum(d["cost"] for d in daily_costs[-7:])

    # Cost per successful build
    cost_per_build = round(total_build_cost / len(successful_builds), 2) if successful_builds else 0

    return {
        "available": True,
        "total_events": len(events),
        "total_cost": round(total_cost, 2),
        "cost_last_24h": round(recent_24h, 2),
        "cost_last_7d": round(recent_7d, 2),
        "total_builds": len(sessions),
        "successful_builds": len(successful_builds),
        "cost_per_build": cost_per_build,
        "total_build_minutes": round(total_build_minutes, 1),
        "total_storage_mb": round(total_storage_mb, 1),
        "cost_breakdown": cost_breakdown,
        "daily_costs": daily_costs,
        "model_storage": model_storage,
    }


def breakdown():
    """Breakdown: build session log, hourly cost heatmap, cost velocity,
    storage breakdown, cost efficiency metrics."""
    events = _load_track_events()
    if not events:
        return {"available": False}

    now = datetime.now(MDT)
    sessions = _compute_build_sessions(events)

    # Build session log (recent 25)
    build_log = []
    for s in sessions[-25:]:
        build_log.append({
            "start": s["start"].strftime("%Y-%m-%d %H:%M"),
            "duration_min": s["duration_min"],
            "cost": f"${s['cost']:.4f}",
        })

    # Hourly cost heatmap (7 days x 24 hours)
    week_ago = now - timedelta(days=7)
    matrix = [[0]*24 for _ in range(7)]
    day_labels = []
    for d in range(6, -1, -1):
        day = (now - timedelta(days=d)).date()
        day_labels.append(day.strftime("%a %m/%d"))
    for s in sessions:
        if s["start"] >= week_ago:
            day_idx = 6 - (now.date() - s["start"].date()).days
            if 0 <= day_idx < 7:
                hour = s["start"].hour
                matrix[day_idx][hour] += s["cost"]
    # Round values
    matrix = [[round(v, 3) for v in row] for row in matrix]

    hourly_heatmap = {
        "matrix": matrix,
        "day_labels": day_labels,
        "hour_labels": [f"{h:02d}" for h in range(24)],
    }

    # Cost velocity (last 14 days: daily + cumulative)
    cost_velocity = []
    cumulative = 0
    daily_costs_map = defaultdict(float)
    for s in sessions:
        d = s["start"].date()
        daily_costs_map[d] += s["cost"]
    for d in range(13, -1, -1):
        day = (now - timedelta(days=d)).date()
        daily = round(daily_costs_map.get(day, 0), 3)
        cumulative += daily
        cost_velocity.append({
            "date": str(day),
            "daily": daily,
            "cumulative": round(cumulative, 3),
        })

    # Model storage breakdown
    model_storage = _get_model_storage()
    total_mb = sum(m["size_mb"] for m in model_storage)
    storage_breakdown = []
    for m in model_storage:
        pct = round(m["size_mb"] / total_mb * 100, 1) if total_mb > 0 else 0
        storage_breakdown.append({
            "model": m["name"],
            "size_mb": m["size_mb"],
            "percent": pct,
            "monthly_cost": f"${m['monthly_cost']:.6f}",
        })

    # Cost efficiency: cost per event by level
    level_counts = Counter(e.get("level", "unknown") for e in events)
    efficiency = []
    cost_map = {
        "autobuild": AGENT_COST_PER_RUN,
        "health": HEALTH_CHECK_COST,
        "watchdog": HEALTH_CHECK_COST * 0.5,
        "git": 0.005,
        "ops": 0.01,
        "ollama": 0.05,
        "system": 0.001,
    }
    for lv, cnt in sorted(level_counts.items(), key=lambda x: -x[1]):
        cpe = cost_map.get(lv, 0.001)
        efficiency.append({
            "level": lv,
            "events": cnt,
            "cost_per_event": f"${cpe:.4f}",
            "total_cost": f"${cnt * cpe:.2f}",
        })

    # Top 5 most expensive build sessions
    sorted_sessions = sorted(sessions, key=lambda s: -s["cost"])[:5]
    top_expensive = [{
        "start": s["start"].strftime("%Y-%m-%d %H:%M"),
        "duration_min": s["duration_min"],
        "cost": f"${s['cost']:.4f}",
    } for s in sorted_sessions]

    # Downtime cost (wasted compute from uptime events)
    uptime_events = _load_uptime_events()
    down_events = [e for e in uptime_events if "DOWN" in str(e.get("status", ""))]
    downtime_cost = len(down_events) * GPU_RATE_PER_HOUR * 0.033  # ~2 min avg downtime

    return {
        "available": True,
        "build_log": build_log,
        "hourly_heatmap": hourly_heatmap,
        "cost_velocity": cost_velocity,
        "storage_breakdown": storage_breakdown,
        "efficiency": efficiency,
        "top_expensive_builds": top_expensive,
        "downtime_events": len(down_events),
        "downtime_cost": round(downtime_cost, 2),
    }


def definitions():
    """FinOps metric definitions, cost model, optimization strategies."""
    return {
        "available": True,
        "cost_model": [
            {"resource": "GPU Compute (T4)", "rate": "$0.35/hour",
             "description": "On-demand NVIDIA T4 GPU for model training and inference"},
            {"resource": "Agent/Autobuild", "rate": "$0.12/run",
             "description": "Claude API tokens per autonomous build cycle (~8K tokens avg)"},
            {"resource": "Model Inference", "rate": "$0.002/prediction",
             "description": "EEG feature extraction + ML prediction pipeline per sample"},
            {"resource": "Health Monitoring", "rate": "$0.001/check",
             "description": "Periodic health check (API + DB + endpoint validation)"},
            {"resource": "Model Storage", "rate": "$0.023/GB/month",
             "description": "S3-equivalent object storage for .joblib model artifacts"},
            {"resource": "Git/CI Triggers", "rate": "$0.005/push",
             "description": "CI pipeline trigger cost per git push event"},
        ],
        "metrics": [
            {"term": "Total Cost", "definition": "Aggregate spend across all tracked resource categories (compute, storage, agent runs, monitoring)"},
            {"term": "Cost per Build", "definition": "Average cost of one autonomous build cycle including GPU time and agent API tokens"},
            {"term": "Build Minutes", "definition": "Total GPU-minutes consumed by autobuild sessions (START→END paired events)"},
            {"term": "Storage Cost", "definition": "Monthly cost of storing trained model artifacts (.joblib files) in cloud object storage"},
            {"term": "Cost Velocity", "definition": "Daily and cumulative cost trend showing spend acceleration or deceleration over time"},
            {"term": "Downtime Cost", "definition": "Wasted compute from system downtime events where GPU/resources were allocated but unused"},
            {"term": "Cost Efficiency", "definition": "Cost per event by operational category — lower is better for routine operations"},
            {"term": "FinOps", "definition": "Financial Operations — discipline of managing cloud AI costs with engineering, finance, and business collaboration"},
            {"term": "Unit Economics", "definition": "Cost per unit of value delivered (e.g., cost per successful model build, cost per prediction)"},
            {"term": "Right-sizing", "definition": "Matching resource allocation to actual workload needs — avoiding over-provisioning GPU/memory"},
            {"term": "Spot/Preemptible", "definition": "Discounted GPU instances that can be interrupted — up to 70% savings for fault-tolerant workloads"},
            {"term": "Reserved Capacity", "definition": "Pre-committed GPU/compute capacity at reduced rates — suitable for predictable baseline workloads"},
        ],
        "optimization_strategies": [
            {"strategy": "Batch Scheduling", "savings": "20-40%",
             "description": "Group autobuild runs during off-peak hours when GPU spot prices are lower"},
            {"strategy": "Model Compression", "savings": "50-80%",
             "description": "Quantize and prune models to reduce storage and inference costs"},
            {"strategy": "Caching", "savings": "30-60%",
             "description": "Cache intermediate EEG features and predictions to avoid redundant compute"},
            {"strategy": "Auto-scaling", "savings": "25-50%",
             "description": "Scale GPU instances to zero when idle; spin up on-demand for builds"},
            {"strategy": "Monitoring Optimization", "savings": "10-20%",
             "description": "Reduce health check frequency during low-traffic periods"},
        ],
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    ov = overview()
    pprint.pprint({k: v for k, v in ov.items() if k != "daily_costs"})
    print("\n=== BREAKDOWN (summary) ===")
    bd = breakdown()
    print(f"Build log entries: {len(bd.get('build_log', []))}")
    print(f"Top expensive: {bd.get('top_expensive_builds', [])}")
    print(f"Efficiency: {bd.get('efficiency', [])}")
