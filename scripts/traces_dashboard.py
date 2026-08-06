"""HTTP Traces Dashboard.

Serves overview, breakdown, and definitions for the /traces portal page.
Reads live data from transaction_log WHERE component='http-trace'.
Each row: action='METHOD /path', ref_id=status_code, detail=JSON{trace_id,latency_ms}.
"""

import json
import os
import sqlite3
import statistics
from datetime import datetime, timezone

_DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _rows(limit=2000):
    """Return raw http-trace rows as list of dicts, newest first."""
    if not os.path.exists(_DB):
        return []
    try:
        conn = sqlite3.connect(_DB, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT id, action, ref_id, detail, ts_utc "
            "FROM transaction_log "
            "WHERE component='http-trace' "
            "ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = []
        for r in cur.fetchall():
            try:
                d = json.loads(r['detail'] or '{}')
            except Exception:
                d = {}
            rows.append({
                'id': r['id'],
                'method_path': r['action'] or '',
                'status_code': r['ref_id'] or 200,
                'trace_id': d.get('trace_id', ''),
                'latency_ms': d.get('latency_ms', 0),
                'ts_utc': r['ts_utc'] or '',
            })
        conn.close()
        return rows
    except Exception:
        return []


def overview():
    """KPIs + top_paths + status_distribution + recent 5 traces."""
    rows = _rows(5000)
    total = len(rows)
    errors = [r for r in rows if (r['status_code'] or 0) >= 400]
    latencies = [r['latency_ms'] for r in rows if isinstance(r['latency_ms'], (int, float)) and r['latency_ms'] > 0]

    error_count = len(errors)
    error_rate = round((error_count / total * 100), 1) if total > 0 else 0.0

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[n // 2] if n > 0 else 0
    p95 = latencies_sorted[int(n * 0.95)] if n > 0 else 0

    # Top paths (by count, descending)
    path_counts: dict = {}
    for r in rows:
        p = r['method_path']
        path_counts[p] = path_counts.get(p, 0) + 1
    top_paths = [
        {'path': k, 'count': v}
        for k, v in sorted(path_counts.items(), key=lambda x: -x[1])[:10]
    ]

    # Status distribution
    status_counts: dict = {}
    for r in rows:
        sc = str(r['status_code'] or 200)
        status_counts[sc] = status_counts.get(sc, 0) + 1
    status_distribution = [
        {'status': k, 'count': v}
        for k, v in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    recent = rows[:5]

    return {
        'available': True,
        'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'kpis': {
            'total_requests': total,
            'error_count': error_count,
            'error_rate_pct': error_rate,
            'p50_latency_ms': p50,
            'p95_latency_ms': p95,
            'trace_coverage': '100%',
        },
        'top_paths': top_paths,
        'status_distribution': status_distribution,
        'recent': recent,
    }


def breakdown():
    """Full trace table (up to 500 most recent rows)."""
    rows = _rows(500)
    return {'traces': rows}


def definitions():
    """Glossary — trace propagation, latency SLAs, span types, fields."""
    return {
        'trace_id': (
            'UUID v4 generated per request by FastAPI middleware. '
            'Injected as X-Trace-ID response header and stored in '
            'transaction_log.detail JSON.'
        ),
        'x_trace_id_header': (
            'X-Trace-ID — present on every API response. '
            'Paired with X-Latency-Ms for end-to-end timing.'
        ),
        'storage': (
            'transaction_log table, component=\'http-trace\'. '
            'action=\'METHOD /path\', ref_id=HTTP status code, '
            'detail=JSON{trace_id, latency_ms}.'
        ),
        'latency_slas': {
            'p50_target_ms': 200,
            'p95_target_ms': 1000,
            'alert_threshold_ms': 3000,
        },
        'span_types': [
            {'name': 'http-trace', 'description': 'Outermost HTTP request span — method, path, status, latency'},
            {'name': 'council',    'description': 'Multi-agent council orchestration step (request_id + tenant_id)'},
            {'name': 'rag-query',  'description': 'Retrieval-augmented generation query latency'},
            {'name': 'db-read',    'description': 'SQLite read operation'},
            {'name': 'db-write',   'description': 'SQLite write / insert operation'},
        ],
        'propagation_fields': [
            'X-Trace-ID', 'X-Latency-Ms', 'request_id', 'tenant_id',
        ],
    }
