#!/usr/bin/env python3
"""
Tool Execution Dashboard — real tool/component execution data from clinical_db
===============================================================================

Reads transaction_log from clinical_db to produce real execution metrics:
  - tool_execution_overview   — total executions, success rate, top tools,
                                actor breakdown, throughput, 14-day trend
  - tool_execution_breakdown  — per-component execution details with action mix
  - tool_execution_definitions — metric definitions, clinical relevance
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

# ── path setup ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clinical_db as cdb


def _load_transactions() -> List[Dict]:
    """Fetch all transactions from clinical_db."""
    try:
        result = cdb.list_transactions(limit=9999)
        return result.get("items", [])
    except Exception:
        return []


def _parse_ts(ts_raw: str) -> datetime | None:
    if not ts_raw:
        return None
    try:
        clean = ts_raw.replace("Z", "+00:00")
        if "+" in clean[10:]:
            clean = clean[: clean.rindex("+")]
        elif "-" in clean[11:]:
            idx = clean.rindex("-")
            if idx > 10:
                clean = clean[:idx]
        return datetime.fromisoformat(clean)
    except Exception:
        return None


def _daily_trend(transactions: List[Dict], days: int = 14) -> List[Dict]:
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days)
    buckets: Dict[str, Dict] = defaultdict(lambda: {"executions": 0, "components": Counter()})

    for tx in transactions:
        ts = _parse_ts(tx.get("ts_utc") or tx.get("ts_local") or "")
        if ts is None or ts < cutoff:
            continue
        day = ts.strftime("%Y-%m-%d")
        buckets[day]["executions"] += 1
        buckets[day]["components"][tx.get("component", "unknown")] += 1

    result = []
    for d in range(days):
        day_str = (now - timedelta(days=days - 1 - d)).strftime("%Y-%m-%d")
        b = buckets.get(day_str, {"executions": 0, "components": Counter()})
        result.append({
            "date": day_str,
            "executions": b["executions"],
            "unique_tools": len(b["components"]),
        })
    return result


# ── public API ────────────────────────────────────────────────────────

def tool_execution_overview() -> Dict[str, Any]:
    transactions = _load_transactions()
    total = len(transactions)

    if total == 0:
        return {
            "available": False,
            "note": "No transactions in clinical_db. Run clinical operations first.",
            "generated_at": datetime.utcnow().isoformat(),
        }

    by_component: Counter = Counter()
    by_action: Counter = Counter()
    by_actor: Counter = Counter()
    hourly: Counter = Counter()

    for tx in transactions:
        by_component[tx.get("component") or "unknown"] += 1
        by_action[tx.get("action") or "unknown"] += 1
        by_actor[tx.get("actor") or "unknown"] += 1
        ts = _parse_ts(tx.get("ts_utc") or tx.get("ts_local") or "")
        if ts:
            hourly[ts.hour] += 1

    # Date range
    timestamps = []
    for tx in transactions:
        ts = _parse_ts(tx.get("ts_utc") or tx.get("ts_local") or "")
        if ts:
            timestamps.append(ts)

    first_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None
    span_days = (last_ts - first_ts).days + 1 if first_ts and last_ts else 1

    # Peak hour
    peak_hour = hourly.most_common(1)[0] if hourly else (0, 0)

    # Error proxy: "blocked" actions
    blocked = sum(1 for tx in transactions if tx.get("action") == "blocked")

    trend = _daily_trend(transactions, 14)

    return {
        "available": True,
        "summary": {
            "total_executions": total,
            "unique_tools": len(by_component),
            "unique_actions": len(by_action),
            "unique_actors": len(by_actor),
            "avg_daily_executions": round(total / max(span_days, 1), 1),
            "peak_hour": peak_hour[0],
            "peak_hour_count": peak_hour[1],
            "blocked_count": blocked,
            "success_rate_pct": round((1 - blocked / max(total, 1)) * 100, 2),
        },
        "top_tools": [{"tool": c, "count": n} for c, n in by_component.most_common(10)],
        "top_actions": [{"action": a, "count": n} for a, n in by_action.most_common(10)],
        "actors": [{"actor": a, "count": n} for a, n in by_actor.most_common(10)],
        "daily_trend": trend,
        "date_range": {
            "first": first_ts.isoformat() if first_ts else None,
            "last": last_ts.isoformat() if last_ts else None,
            "span_days": span_days,
        },
        "generated_at": datetime.utcnow().isoformat(),
    }


def tool_execution_breakdown() -> Dict[str, Any]:
    transactions = _load_transactions()

    if not transactions:
        return {
            "available": False,
            "note": "No transaction data available.",
            "generated_at": datetime.utcnow().isoformat(),
        }

    comp_data: Dict[str, Dict] = defaultdict(lambda: {
        "count": 0,
        "actions": Counter(),
        "actors": Counter(),
        "first_seen": None,
        "last_seen": None,
    })

    for tx in transactions:
        comp = tx.get("component") or "unknown"
        act = tx.get("action") or "unknown"
        actor = tx.get("actor") or "unknown"
        comp_data[comp]["count"] += 1
        comp_data[comp]["actions"][act] += 1
        comp_data[comp]["actors"][actor] += 1

        ts = _parse_ts(tx.get("ts_utc") or tx.get("ts_local") or "")
        if ts:
            if comp_data[comp]["first_seen"] is None or ts < comp_data[comp]["first_seen"]:
                comp_data[comp]["first_seen"] = ts
            if comp_data[comp]["last_seen"] is None or ts > comp_data[comp]["last_seen"]:
                comp_data[comp]["last_seen"] = ts

    total = sum(d["count"] for d in comp_data.values())
    tools = []
    for comp, data in comp_data.items():
        tools.append({
            "tool": comp,
            "executions": data["count"],
            "pct_of_total": round(data["count"] / max(total, 1) * 100, 1),
            "top_actions": dict(data["actions"].most_common(5)),
            "top_actors": dict(data["actors"].most_common(3)),
            "first_seen": data["first_seen"].isoformat() if data["first_seen"] else None,
            "last_seen": data["last_seen"].isoformat() if data["last_seen"] else None,
        })

    tools.sort(key=lambda x: x["executions"], reverse=True)

    return {
        "available": True,
        "tools": tools,
        "total_executions": total,
        "unique_tools": len(tools),
        "generated_at": datetime.utcnow().isoformat(),
    }


def tool_execution_definitions() -> Dict[str, Any]:
    return {
        "available": True,
        "metrics": [
            {
                "name": "Total Executions",
                "description": "Count of all tool/component invocations recorded in the transaction log",
                "unit": "count",
            },
            {
                "name": "Unique Tools",
                "description": "Number of distinct components (e.g., assessment, seizure_diary, cv_pipeline) that have been invoked",
                "unit": "count",
            },
            {
                "name": "Success Rate",
                "description": "Percentage of executions that were not blocked or errored. Computed as (total - blocked) / total * 100",
                "unit": "%",
            },
            {
                "name": "Avg Daily Executions",
                "description": "Total executions divided by the span of days covered by the transaction log",
                "unit": "executions/day",
            },
            {
                "name": "Peak Hour",
                "description": "UTC hour with the highest number of tool executions, useful for capacity planning",
                "unit": "hour (0-23 UTC)",
            },
            {
                "name": "Execution Share (%)",
                "description": "Per-tool percentage of total executions, showing which tools dominate usage",
                "unit": "%",
            },
        ],
        "data_sources": {
            "transaction_log": {
                "source": "clinical_db SQLite — transaction_log table",
                "real_data": True,
                "description": "Every component invocation (create, process, query, monitor, etc.) across the clinical AI system",
            },
        },
        "clinical_relevance": (
            "Monitoring tool execution volume and patterns is essential for clinical AI "
            "post-market surveillance (FDA SaMD, EU AI Act Article 72). It reveals: "
            "(1) which AI tools clinicians actually use vs. deploy; "
            "(2) peak usage windows for capacity planning; "
            "(3) failure/blocked rates for reliability monitoring; "
            "(4) actor distribution to verify human-in-the-loop governance."
        ),
    }


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    fn_map = {
        "overview": tool_execution_overview,
        "breakdown": tool_execution_breakdown,
        "definitions": tool_execution_definitions,
    }
    target = sys.argv[1] if len(sys.argv) > 1 else "overview"
    func = fn_map.get(target, tool_execution_overview)
    print(json.dumps(func(), indent=2, default=str))
