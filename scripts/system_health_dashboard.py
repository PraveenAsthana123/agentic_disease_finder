"""
Neuro AI Ecosystem — System Health Monitoring Dashboard
=======================================================
Real-time infrastructure health monitoring from system_health_log table.

Components tracked: API, Cache, Database, Frontend, ML Pipeline, Queue, Storage
Status levels: healthy, degraded, down
Metrics: response_time_ms, cpu_pct, memory_pct, disk_pct, error_count

Real data: system_health_log (30 rows, 7 components) in clinical.db.
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """System health overview — overall uptime, component status, resource utilization,
    response time trends, error distribution."""
    conn = _conn()
    cur = conn.cursor()

    # Total checks and status distribution
    cur.execute("SELECT COUNT(*) FROM system_health_log")
    total_checks = cur.fetchone()[0]

    cur.execute("SELECT status, COUNT(*) FROM system_health_log GROUP BY status ORDER BY status")
    status_dist = {r[0]: r[1] for r in cur.fetchall()}

    healthy_pct = round(status_dist.get("healthy", 0) / max(total_checks, 1) * 100, 1)

    # Resource averages
    cur.execute("""
        SELECT ROUND(AVG(cpu_pct), 1) avg_cpu,
               ROUND(AVG(memory_pct), 1) avg_mem,
               ROUND(AVG(disk_pct), 1) avg_disk,
               ROUND(AVG(response_time_ms), 0) avg_rt,
               SUM(error_count) total_errors
        FROM system_health_log
    """)
    r = cur.fetchone()
    avg_cpu, avg_mem, avg_disk, avg_rt, total_errors = r

    # Per-component health summary
    cur.execute("""
        SELECT component,
               COUNT(*) checks,
               SUM(CASE WHEN status = 'healthy' THEN 1 ELSE 0 END) healthy_count,
               SUM(CASE WHEN status = 'degraded' THEN 1 ELSE 0 END) degraded_count,
               SUM(CASE WHEN status = 'down' THEN 1 ELSE 0 END) down_count,
               ROUND(AVG(response_time_ms), 0) avg_rt,
               ROUND(AVG(cpu_pct), 1) avg_cpu,
               ROUND(AVG(memory_pct), 1) avg_mem,
               ROUND(AVG(disk_pct), 1) avg_disk,
               SUM(error_count) errors
        FROM system_health_log
        GROUP BY component
        ORDER BY component
    """)
    component_summary = []
    for row in cur.fetchall():
        uptime = round(row[2] / max(row[1], 1) * 100, 1)
        latest_status = "healthy"
        # Get the latest status for each component
        cur2 = conn.cursor()
        cur2.execute("""
            SELECT status FROM system_health_log
            WHERE component = ? ORDER BY timestamp DESC LIMIT 1
        """, (row[0],))
        latest = cur2.fetchone()
        if latest:
            latest_status = latest[0]
        component_summary.append({
            "component": row[0],
            "checks": row[1],
            "healthy": row[2],
            "degraded": row[3],
            "down": row[4],
            "uptime_pct": uptime,
            "current_status": latest_status,
            "avg_response_ms": int(row[5]) if row[5] else 0,
            "avg_cpu": row[6],
            "avg_mem": row[7],
            "avg_disk": row[8],
            "error_count": row[9] or 0,
        })

    # Timeline trend (checks over time)
    cur.execute("""
        SELECT timestamp, component, status, response_time_ms,
               cpu_pct, memory_pct, disk_pct, error_count
        FROM system_health_log
        ORDER BY timestamp
    """)
    timeline = _dict_rows(cur)

    # Resource utilization distribution
    cur.execute("""
        SELECT
            SUM(CASE WHEN cpu_pct < 30 THEN 1 ELSE 0 END) cpu_low,
            SUM(CASE WHEN cpu_pct BETWEEN 30 AND 70 THEN 1 ELSE 0 END) cpu_mid,
            SUM(CASE WHEN cpu_pct > 70 THEN 1 ELSE 0 END) cpu_high,
            SUM(CASE WHEN memory_pct < 40 THEN 1 ELSE 0 END) mem_low,
            SUM(CASE WHEN memory_pct BETWEEN 40 AND 75 THEN 1 ELSE 0 END) mem_mid,
            SUM(CASE WHEN memory_pct > 75 THEN 1 ELSE 0 END) mem_high,
            SUM(CASE WHEN disk_pct < 50 THEN 1 ELSE 0 END) disk_low,
            SUM(CASE WHEN disk_pct BETWEEN 50 AND 80 THEN 1 ELSE 0 END) disk_mid,
            SUM(CASE WHEN disk_pct > 80 THEN 1 ELSE 0 END) disk_high
        FROM system_health_log
    """)
    rd = cur.fetchone()
    resource_distribution = {
        "cpu": {"low": rd[0], "moderate": rd[1], "high": rd[2]},
        "memory": {"low": rd[3], "moderate": rd[4], "high": rd[5]},
        "disk": {"low": rd[6], "moderate": rd[7], "high": rd[8]},
    }

    conn.close()
    return {
        "kpis": {
            "total_checks": total_checks,
            "overall_uptime_pct": healthy_pct,
            "avg_response_ms": int(avg_rt) if avg_rt else 0,
            "avg_cpu_pct": avg_cpu,
            "avg_memory_pct": avg_mem,
            "avg_disk_pct": avg_disk,
            "total_errors": total_errors or 0,
            "components_monitored": len(component_summary),
        },
        "status_distribution": status_dist,
        "component_summary": component_summary,
        "resource_distribution": resource_distribution,
        "timeline": timeline,
    }


def breakdown():
    """Per-component detail — recent checks, resource trends, error log,
    response time percentiles."""
    conn = _conn()
    cur = conn.cursor()

    # Per-component detailed records
    cur.execute("""
        SELECT check_id, timestamp, component, status,
               response_time_ms, cpu_pct, memory_pct, disk_pct, error_count
        FROM system_health_log
        ORDER BY timestamp DESC
    """)
    all_checks = _dict_rows(cur)

    # Component-level response time percentiles
    components = {}
    cur.execute("SELECT DISTINCT component FROM system_health_log ORDER BY component")
    for (comp,) in cur.fetchall():
        cur.execute("""
            SELECT response_time_ms FROM system_health_log
            WHERE component = ? ORDER BY response_time_ms
        """, (comp,))
        rts = [r[0] for r in cur.fetchall()]
        n = len(rts)
        components[comp] = {
            "p50": rts[n // 2] if n else 0,
            "p90": rts[int(n * 0.9)] if n else 0,
            "p99": rts[-1] if n else 0,
            "min": rts[0] if n else 0,
            "max": rts[-1] if n else 0,
        }

    # Error timeline (only checks with errors)
    cur.execute("""
        SELECT check_id, timestamp, component, error_count, status
        FROM system_health_log
        WHERE error_count > 0
        ORDER BY timestamp DESC
    """)
    error_events = _dict_rows(cur)

    # Degraded/down incidents
    cur.execute("""
        SELECT check_id, timestamp, component, status,
               response_time_ms, error_count
        FROM system_health_log
        WHERE status != 'healthy'
        ORDER BY timestamp DESC
    """)
    incidents = _dict_rows(cur)

    conn.close()
    return {
        "all_checks": all_checks,
        "response_percentiles": components,
        "error_events": error_events,
        "incidents": incidents,
        "total_incidents": len(incidents),
    }


def definitions():
    """System health monitoring glossary — status levels, resource thresholds,
    component descriptions."""
    return {
        "glossary": [
            {"term": "Healthy", "definition": "Component responding within SLA, no errors detected."},
            {"term": "Degraded", "definition": "Component responding but with elevated latency or partial errors."},
            {"term": "Down", "definition": "Component unresponsive or returning critical errors."},
            {"term": "Response Time (ms)", "definition": "Round-trip latency from health check probe to component response."},
            {"term": "CPU %", "definition": "Percentage of CPU capacity utilized by the component at check time."},
            {"term": "Memory %", "definition": "Percentage of allocated memory consumed by the component."},
            {"term": "Disk %", "definition": "Percentage of disk storage used on the component's volume."},
            {"term": "Error Count", "definition": "Number of errors logged by the component since the previous health check."},
            {"term": "Uptime %", "definition": "Fraction of health checks where the component reported healthy status."},
            {"term": "P50 / P90 / P99", "definition": "Response time percentiles — median, 90th, and 99th percentile latency."},
            {"term": "SLA", "definition": "Service Level Agreement — contractual uptime and latency guarantees."},
            {"term": "Incident", "definition": "Any health check returning degraded or down status."},
        ],
    }
