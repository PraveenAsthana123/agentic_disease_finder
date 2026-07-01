"""
Cloud Ops Dashboard
Infrastructure monitoring: compute/storage/network costs, region health,
autoscaling events, uptime SLA, and resource utilisation.

Registry item: cloud (admin_module.ops_dashboards)
Purpose: infra cost, regions, autoscaling, uptime
"""

import hashlib
import json
import math
import os
import time

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Reference data — regions, services, thresholds
# ---------------------------------------------------------------------------

REGIONS = [
    {"id": "us-east-1",      "label": "US East (N. Virginia)",   "provider": "AWS"},
    {"id": "us-west-2",      "label": "US West (Oregon)",        "provider": "AWS"},
    {"id": "eu-west-1",      "label": "EU West (Ireland)",       "provider": "AWS"},
    {"id": "ap-south-1",     "label": "Asia Pacific (Mumbai)",   "provider": "AWS"},
    {"id": "ca-central-1",   "label": "Canada (Central)",        "provider": "AWS"},
]

SERVICES = [
    {"id": "compute",  "label": "Compute (EC2/ECS)",        "icon": "server"},
    {"id": "storage",  "label": "Storage (S3/EBS)",         "icon": "database"},
    {"id": "network",  "label": "Network (VPC/ALB/CDN)",    "icon": "globe"},
    {"id": "database", "label": "Database (RDS/DynamoDB)",  "icon": "table"},
    {"id": "ml",       "label": "ML/AI (SageMaker/GPU)",    "icon": "cpu"},
    {"id": "logging",  "label": "Logging (CloudWatch)",     "icon": "file-text"},
]

COST_THRESHOLDS = {
    "budget_monthly_usd":  2500,
    "alert_50_pct":        1250,
    "alert_80_pct":        2000,
    "alert_100_pct":       2500,
}

UPTIME_SLA_TARGET = 99.95  # percent

AUTOSCALE_POLICIES = [
    {"name": "api-gateway",    "min": 2,  "max": 8,  "metric": "CPU > 70%",     "cooldown_s": 300},
    {"name": "ml-workers",     "min": 1,  "max": 4,  "metric": "Queue > 50",    "cooldown_s": 600},
    {"name": "rag-service",    "min": 1,  "max": 3,  "metric": "Latency > 2s",  "cooldown_s": 300},
    {"name": "web-frontend",   "min": 2,  "max": 6,  "metric": "RPS > 200",     "cooldown_s": 180},
]

# ---------------------------------------------------------------------------
# Deterministic seed helper
# ---------------------------------------------------------------------------

def _seed(key: str) -> float:
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


# ---------------------------------------------------------------------------
# Data generation — deterministic from region/service/date
# ---------------------------------------------------------------------------

def _generate_region_health():
    """Per-region health metrics."""
    results = []
    for reg in REGIONS:
        s = _seed(f"region:{reg['id']}:health")
        uptime = round(_lerp(99.80, 100.0, s), 3)
        latency_ms = round(_lerp(8, 120, _seed(f"region:{reg['id']}:latency")))
        instances = max(1, round(_lerp(2, 12, _seed(f"region:{reg['id']}:instances"))))
        error_rate = round(_lerp(0.0, 0.5, _seed(f"region:{reg['id']}:errors")), 2)
        status = "healthy" if uptime >= 99.95 and error_rate < 0.3 else "degraded" if uptime >= 99.5 else "down"
        results.append({
            "region_id": reg["id"],
            "label": reg["label"],
            "provider": reg["provider"],
            "uptime_pct": uptime,
            "latency_ms": latency_ms,
            "active_instances": instances,
            "error_rate_pct": error_rate,
            "status": status,
            "meets_sla": uptime >= UPTIME_SLA_TARGET,
        })
    return results


def _generate_service_costs():
    """Per-service monthly cost breakdown."""
    results = []
    total = 0
    for svc in SERVICES:
        s = _seed(f"cost:{svc['id']}:monthly")
        # ML/compute are expensive; logging/network cheaper
        if svc["id"] == "ml":
            cost = round(_lerp(500, 900, s), 2)
        elif svc["id"] == "compute":
            cost = round(_lerp(400, 700, s), 2)
        elif svc["id"] == "database":
            cost = round(_lerp(200, 450, s), 2)
        elif svc["id"] == "storage":
            cost = round(_lerp(100, 250, s), 2)
        else:
            cost = round(_lerp(50, 200, s), 2)
        prev_cost = round(cost * _lerp(0.85, 1.15, _seed(f"cost:{svc['id']}:prev")), 2)
        delta_pct = round((cost - prev_cost) / max(prev_cost, 1) * 100, 1)
        total += cost
        results.append({
            "service_id": svc["id"],
            "label": svc["label"],
            "icon": svc["icon"],
            "cost_usd": cost,
            "prev_month_usd": prev_cost,
            "delta_pct": delta_pct,
        })
    return results, round(total, 2)


def _generate_autoscale_events():
    """Recent autoscaling events."""
    events = []
    base_ts = 1751331600  # 2025-06-30 approx
    for i, pol in enumerate(AUTOSCALE_POLICIES):
        n_events = max(1, round(_lerp(1, 6, _seed(f"autoscale:{pol['name']}:count"))))
        for j in range(n_events):
            s = _seed(f"autoscale:{pol['name']}:event:{j}")
            ts = base_ts - round(s * 86400 * 7)  # within last 7 days
            direction = "scale-up" if s > 0.35 else "scale-down"
            from_count = pol["min"] + round(_lerp(0, pol["max"] - pol["min"], _seed(f"asc:{pol['name']}:{j}:from")))
            delta = 1 if direction == "scale-up" else -1
            to_count = max(pol["min"], min(pol["max"], from_count + delta))
            events.append({
                "policy": pol["name"],
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "direction": direction,
                "from_count": from_count,
                "to_count": to_count,
                "trigger_metric": pol["metric"],
                "cooldown_s": pol["cooldown_s"],
            })
    events.sort(key=lambda e: e["timestamp_utc"], reverse=True)
    return events


def _generate_resource_utilisation():
    """CPU, memory, disk, network utilisation per region."""
    results = []
    for reg in REGIONS:
        cpu = round(_lerp(15, 85, _seed(f"util:{reg['id']}:cpu")), 1)
        mem = round(_lerp(30, 90, _seed(f"util:{reg['id']}:mem")), 1)
        disk = round(_lerp(20, 75, _seed(f"util:{reg['id']}:disk")), 1)
        net_mbps = round(_lerp(10, 500, _seed(f"util:{reg['id']}:net")), 1)
        results.append({
            "region_id": reg["id"],
            "label": reg["label"],
            "cpu_pct": cpu,
            "memory_pct": mem,
            "disk_pct": disk,
            "network_mbps": net_mbps,
            "cpu_status": "critical" if cpu > 80 else "warning" if cpu > 60 else "ok",
            "memory_status": "critical" if mem > 85 else "warning" if mem > 70 else "ok",
        })
    return results


def _generate_uptime_history():
    """30-day uptime history."""
    days = []
    for d in range(30):
        s = _seed(f"uptime:day:{d}")
        uptime = round(_lerp(99.50, 100.0, s), 3)
        incidents = 1 if uptime < 99.9 else 0
        days.append({
            "day_offset": d,
            "uptime_pct": uptime,
            "incidents": incidents,
        })
    avg_uptime = round(sum(d["uptime_pct"] for d in days) / len(days), 3)
    total_incidents = sum(d["incidents"] for d in days)
    return days, avg_uptime, total_incidents


# ---------------------------------------------------------------------------
# Public API — overview / breakdown / definitions
# ---------------------------------------------------------------------------

def overview():
    """Cloud Ops overview: KPIs, region health, cost summary, autoscale summary."""
    regions = _generate_region_health()
    costs, total_cost = _generate_service_costs()
    events = _generate_autoscale_events()
    _, avg_uptime, total_incidents = _generate_uptime_history()

    healthy_count = sum(1 for r in regions if r["status"] == "healthy")
    degraded_count = sum(1 for r in regions if r["status"] == "degraded")
    sla_met = all(r["meets_sla"] for r in regions)
    budget_pct = round(total_cost / COST_THRESHOLDS["budget_monthly_usd"] * 100, 1)
    mean_latency = round(sum(r["latency_ms"] for r in regions) / len(regions), 1)

    # Region status distribution
    region_status_dist = {}
    for r in regions:
        region_status_dist[r["status"]] = region_status_dist.get(r["status"], 0) + 1

    # Cost by service
    cost_by_service = [{"service": c["label"], "cost_usd": c["cost_usd"], "delta_pct": c["delta_pct"]} for c in costs]

    # Autoscale summary
    scale_up_count = sum(1 for e in events if e["direction"] == "scale-up")
    scale_down_count = sum(1 for e in events if e["direction"] == "scale-down")

    return {
        "kpis": {
            "total_regions": len(regions),
            "healthy_regions": healthy_count,
            "degraded_regions": degraded_count,
            "avg_uptime_pct": avg_uptime,
            "total_cost_usd": total_cost,
            "budget_pct": budget_pct,
            "mean_latency_ms": mean_latency,
            "sla_met": sla_met,
            "total_incidents_30d": total_incidents,
        },
        "region_status_distribution": region_status_dist,
        "cost_by_service": cost_by_service,
        "autoscale_summary": {
            "total_events": len(events),
            "scale_up": scale_up_count,
            "scale_down": scale_down_count,
        },
        "region_summary": [
            {
                "region": r["label"],
                "status": r["status"],
                "uptime_pct": r["uptime_pct"],
                "latency_ms": r["latency_ms"],
                "instances": r["active_instances"],
                "error_rate_pct": r["error_rate_pct"],
                "meets_sla": r["meets_sla"],
            }
            for r in regions
        ],
    }


def breakdown():
    """Cloud Ops breakdown: resource utilisation, cost detail, autoscale events,
    uptime history."""
    regions = _generate_region_health()
    costs, total_cost = _generate_service_costs()
    events = _generate_autoscale_events()
    utilisation = _generate_resource_utilisation()
    uptime_days, avg_uptime, total_incidents = _generate_uptime_history()

    return {
        "resource_utilisation": utilisation,
        "cost_breakdown": [
            {
                "service": c["label"],
                "service_id": c["service_id"],
                "icon": c["icon"],
                "cost_usd": c["cost_usd"],
                "prev_month_usd": c["prev_month_usd"],
                "delta_pct": c["delta_pct"],
            }
            for c in costs
        ],
        "total_cost_usd": total_cost,
        "budget_usd": COST_THRESHOLDS["budget_monthly_usd"],
        "budget_pct": round(total_cost / COST_THRESHOLDS["budget_monthly_usd"] * 100, 1),
        "autoscale_events": events[:20],
        "autoscale_policies": AUTOSCALE_POLICIES,
        "uptime_history": uptime_days,
        "uptime_avg_30d": avg_uptime,
        "uptime_sla_target": UPTIME_SLA_TARGET,
        "incidents_30d": total_incidents,
    }


def definitions():
    """Cloud Ops definitions, thresholds, policies, reference info."""
    return {
        "regions": REGIONS,
        "services": SERVICES,
        "cost_thresholds": COST_THRESHOLDS,
        "uptime_sla_target_pct": UPTIME_SLA_TARGET,
        "autoscale_policies": AUTOSCALE_POLICIES,
        "status_levels": {
            "healthy":  {"description": "All SLA targets met, error rate < 0.3%, uptime >= 99.95%"},
            "degraded": {"description": "Partial SLA miss, uptime >= 99.5% but < 99.95%, or error rate elevated"},
            "down":     {"description": "Uptime < 99.5%, region experiencing significant outage"},
        },
        "cost_alert_levels": {
            "50%": {"threshold_usd": COST_THRESHOLDS["alert_50_pct"], "action": "Informational alert to team"},
            "80%": {"threshold_usd": COST_THRESHOLDS["alert_80_pct"], "action": "Warning alert, review spend"},
            "100%": {"threshold_usd": COST_THRESHOLDS["alert_100_pct"], "action": "Critical alert, freeze non-essential"},
        },
        "resource_thresholds": {
            "cpu":    {"warning": 60, "critical": 80, "unit": "%"},
            "memory": {"warning": 70, "critical": 85, "unit": "%"},
            "disk":   {"warning": 70, "critical": 90, "unit": "%"},
        },
        "clinical_relevance": (
            "Cloud infrastructure underpins the NeuroAI platform. Downtime or "
            "degraded performance directly impacts EEG classification latency, "
            "model inference availability, and clinical decision support. "
            "Autoscaling ensures ML workloads handle burst demand (e.g., batch "
            "EEG processing). Cost monitoring prevents budget overruns on GPU "
            "instances used for deep learning inference."
        ),
    }
