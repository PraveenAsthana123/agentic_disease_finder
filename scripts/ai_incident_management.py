"""
AI Incident Management — parses real uptime.jsonl + track.jsonl to produce
overview / breakdown / definitions for the Incident Management dashboard.
"""

import json, os, pathlib
from datetime import datetime, timedelta, timezone, tzinfo

# MDT = UTC-6
MDT = timezone(timedelta(hours=-6))

BASE = pathlib.Path(__file__).resolve().parent.parent
UPTIME_LOG = BASE / "jobs" / "logs" / "uptime.jsonl"
TRACK_LOG  = BASE / "jobs" / "logs" / "track.jsonl"
HEALTH_FILE = BASE / "jobs" / "reports" / "health_latest.json"


def _parse_ts(ts_str: str) -> datetime:
    """Parse timestamp like '2026-06-25 10:02:34 MDT' -> aware datetime."""
    ts_str = ts_str.strip()
    if ts_str.endswith(" MDT"):
        ts_str = ts_str[:-4]
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=MDT)
    # fallback: try ISO
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


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


def _load_health():
    if not HEALTH_FILE.exists():
        return {}
    try:
        return json.loads(HEALTH_FILE.read_text())
    except Exception:
        return {}


def _categorize_severity(http_code: str):
    if http_code == "000":
        return "service_crash"
    try:
        code = int(http_code)
    except (ValueError, TypeError):
        return "unknown"
    if code >= 500:
        return "server_error"
    if code >= 400:
        return "client_error"
    if code >= 300:
        return "redirect"
    return "other"


# ── overview ─────────────────────────────────────────────────────
def overview():
    uptime = _load_uptime()
    health = _load_health()

    downs = [e for e in uptime if e.get("event") == "DOWN"]
    recoveries = [e for e in uptime if e.get("event") in ("AUTO-RESTARTED", "RECOVERED")]

    total_incidents = len(downs)
    total_recoveries = len(recoveries)
    auto_recovery_rate = round(total_recoveries / total_incidents * 100, 1) if total_incidents else 0.0

    # current status from health
    http_code = health.get("backend_http", "000")
    current_status = "UP" if http_code == "200" else "DOWN"

    # MTTR: pair each DOWN with the next recovery event after it
    sorted_events = sorted(uptime, key=lambda e: e["_dt"])
    ttr_list = []
    for i, ev in enumerate(sorted_events):
        if ev.get("event") != "DOWN":
            continue
        for j in range(i + 1, len(sorted_events)):
            if sorted_events[j].get("event") in ("AUTO-RESTARTED", "RECOVERED"):
                diff = (sorted_events[j]["_dt"] - ev["_dt"]).total_seconds() / 60.0
                if diff >= 0:
                    ttr_list.append(diff)
                break

    mean_ttr = round(sum(ttr_list) / len(ttr_list), 2) if ttr_list else 0.0

    now = datetime.now(MDT)
    incidents_24h = sum(1 for d in downs if (now - d["_dt"]).total_seconds() < 86400)
    incidents_7d = sum(1 for d in downs if (now - d["_dt"]).total_seconds() < 86400 * 7)

    # severity distribution
    severity_counts = {}
    for d in downs:
        sev = _categorize_severity(d.get("http", "000"))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    severity_distribution = []
    for level, count in sorted(severity_counts.items(), key=lambda x: -x[1]):
        severity_distribution.append({
            "level": level,
            "count": count,
            "percent": round(count / total_incidents * 100, 1) if total_incidents else 0,
        })

    # daily incident counts – last 14 days
    daily = {}
    for i in range(14):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        daily[day] = 0
    for d in downs:
        day = d["_dt"].astimezone(MDT).strftime("%Y-%m-%d")
        if day in daily:
            daily[day] += 1
    daily_incident_counts = [{"date": k, "count": v} for k, v in sorted(daily.items())]

    # incident timeline – last 20
    timeline = []
    for ev in sorted(downs, key=lambda e: e["_dt"], reverse=True)[:20]:
        rec_ts = None
        ttr_min = None
        # find matching recovery
        for r in sorted_events:
            if r["_dt"] >= ev["_dt"] and r.get("event") in ("AUTO-RESTARTED", "RECOVERED"):
                rec_ts = r["ts"]
                ttr_min = round((r["_dt"] - ev["_dt"]).total_seconds() / 60.0, 1)
                break
        timeline.append({
            "ts": ev["ts"],
            "event": ev["event"],
            "http": ev.get("http", ""),
            "recovery_ts": rec_ts,
            "ttr_minutes": ttr_min,
        })

    return {
        "available": True,
        "total_incidents": total_incidents,
        "total_recoveries": total_recoveries,
        "auto_recovery_rate": auto_recovery_rate,
        "current_status": current_status,
        "mean_time_to_recovery_minutes": mean_ttr,
        "incidents_last_24h": incidents_24h,
        "incidents_last_7d": incidents_7d,
        "severity_distribution": severity_distribution,
        "daily_incident_counts": daily_incident_counts,
        "incident_timeline": timeline,
    }


# ── breakdown ────────────────────────────────────────────────────
def breakdown():
    uptime = _load_uptime()
    track = _load_track()
    now = datetime.now(MDT)

    downs = [e for e in uptime if e.get("event") == "DOWN"]
    recoveries_set = {e["_dt"] for e in uptime if e.get("event") in ("AUTO-RESTARTED", "RECOVERED")}

    # hourly heatmap: 7 days x 24 hours
    heatmap = [[0] * 24 for _ in range(7)]
    day_labels = []
    for i in range(6, -1, -1):
        d = now - timedelta(days=i)
        day_labels.append(d.strftime("%a %m/%d"))
    for d in downs:
        dt = d["_dt"].astimezone(MDT)
        day_offset = (now.date() - dt.date()).days
        if 0 <= day_offset <= 6:
            row_idx = 6 - day_offset  # most recent day last
            heatmap[row_idx][dt.hour] += 1

    hourly_heatmap = {
        "day_labels": day_labels,
        "hour_labels": [f"{h:02d}" for h in range(24)],
        "matrix": heatmap,
    }

    # top incident types
    type_counts = {}
    for e in uptime:
        ev = e.get("event", "UNKNOWN")
        type_counts[ev] = type_counts.get(ev, 0) + 1
    top_incident_types = [{"type": k, "count": v} for k, v in sorted(type_counts.items(), key=lambda x: -x[1])]

    # recovery success rate by day – last 14 days
    recovery_by_day = []
    for i in range(13, -1, -1):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        day_downs = [d for d in downs if d["_dt"].astimezone(MDT).strftime("%Y-%m-%d") == day]
        day_recs = [e for e in uptime
                    if e.get("event") in ("AUTO-RESTARTED", "RECOVERED")
                    and e["_dt"].astimezone(MDT).strftime("%Y-%m-%d") == day]
        inc = len(day_downs)
        rec = len(day_recs)
        recovery_by_day.append({
            "date": day,
            "incidents": inc,
            "recovered": rec,
            "rate": round(rec / inc * 100, 1) if inc else 100.0,
        })

    # track events by level
    level_counts = {}
    for t in track:
        lvl = t.get("level", "unknown")
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    track_events_by_level = [{"level": k, "count": v} for k, v in sorted(level_counts.items(), key=lambda x: -x[1])]

    # recent track events (ops/watchdog/system)
    relevant_levels = {"ops", "watchdog", "system"}
    filtered = [t for t in track if t.get("level") in relevant_levels]
    filtered.sort(key=lambda t: t["_dt"], reverse=True)
    recent_track = []
    for t in filtered[:30]:
        recent_track.append({
            "ts": t.get("ts_local", t.get("ts_utc", "")),
            "level": t.get("level", ""),
            "event": t.get("event", ""),
            "host": t.get("host", ""),
        })

    # MTTR trend – last 14 days
    sorted_events = sorted(uptime, key=lambda e: e["_dt"])
    mttr_trend = []
    for i in range(13, -1, -1):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        day_ttrs = []
        for idx, ev in enumerate(sorted_events):
            if ev.get("event") != "DOWN":
                continue
            if ev["_dt"].astimezone(MDT).strftime("%Y-%m-%d") != day:
                continue
            for j in range(idx + 1, len(sorted_events)):
                if sorted_events[j].get("event") in ("AUTO-RESTARTED", "RECOVERED"):
                    diff = (sorted_events[j]["_dt"] - ev["_dt"]).total_seconds() / 60.0
                    if diff >= 0:
                        day_ttrs.append(diff)
                    break
        avg = round(sum(day_ttrs) / len(day_ttrs), 2) if day_ttrs else 0.0
        mttr_trend.append({"date": day, "avg_mttr_minutes": avg})

    return {
        "available": True,
        "hourly_heatmap": hourly_heatmap,
        "top_incident_types": top_incident_types,
        "recovery_success_rate_by_day": recovery_by_day,
        "track_events_by_level": track_events_by_level,
        "recent_track_events": recent_track,
        "mttr_trend": mttr_trend,
    }


# ── definitions ──────────────────────────────────────────────────
def definitions():
    return {
        "available": True,
        "metrics": [
            {"term": "Total Incidents", "definition": "Count of DOWN events recorded in the uptime log, indicating the backend became unreachable."},
            {"term": "Total Recoveries", "definition": "Count of AUTO-RESTARTED and RECOVERED events, indicating the system came back online."},
            {"term": "Auto-Recovery Rate", "definition": "Percentage of DOWN incidents followed by an automatic recovery (AUTO-RESTARTED or RECOVERED) without manual intervention."},
            {"term": "Current Status", "definition": "Live status derived from the latest health check. UP if HTTP 200, DOWN otherwise."},
            {"term": "MTTR (Mean Time to Recovery)", "definition": "Average elapsed minutes between a DOWN event and the next recovery event. Lower is better."},
            {"term": "Incidents Last 24h / 7d", "definition": "Number of DOWN events in the last 24 hours and 7 days respectively."},
            {"term": "Severity: service_crash (HTTP 000)", "definition": "The backend process crashed or is unreachable — no HTTP response returned."},
            {"term": "Severity: server_error (HTTP 5xx)", "definition": "The backend returned an internal server error."},
            {"term": "Severity: client_error (HTTP 4xx)", "definition": "A client-side error was returned (auth failure, bad request, etc.)."},
            {"term": "Hourly Heatmap", "definition": "Matrix of incident counts by hour of day and day of week, showing when outages cluster."},
            {"term": "Recovery Success Rate", "definition": "Per-day ratio of recovery events to incident events. 100% means every outage was recovered."},
            {"term": "Track Events", "definition": "Operational log entries from the system tracker (track.jsonl), including ops commands, git pushes, auto-builds, and watchdog alerts."},
            {"term": "Track Level: ops", "definition": "Operator-initiated events: restarts, deploys, manual interventions."},
            {"term": "Track Level: system", "definition": "System-level events: cron installs, tracker setup, infrastructure changes."},
            {"term": "Track Level: watchdog", "definition": "Automated watchdog alerts: health check failures, auto-restart triggers."},
            {"term": "Track Level: git", "definition": "Git operations: auto-push, commits, branch operations."},
            {"term": "Track Level: autobuild", "definition": "Automated build pipeline events: scheduled builds, headless Claude sessions."},
        ],
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN ===")
    pprint.pprint(breakdown())
