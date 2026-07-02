"""MCP Governance Dashboard — real agent/tool governance metrics.

Sources:
- config/agent_tasks.json (60 agents: id, task, module, status)
- jobs/reports/agentic_mcp_report.json (MCP telemetry, A2A health, votes)
- config/enterprise_pipelines.json (pipeline registry: status per pipeline)
- data/clinical.db transaction_log (agent actions, blocked events)
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, 'data', 'clinical.db')
AGENT_TASKS = os.path.join(BASE, 'config', 'agent_tasks.json')
MCP_REPORT = os.path.join(BASE, 'jobs', 'reports', 'agentic_mcp_report.json')
PIPELINES = os.path.join(BASE, 'config', 'enterprise_pipelines.json')


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _conn():
    return sqlite3.connect(DB)


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def mcp_governance_overview():
    """Aggregate MCP governance posture: agent inventory, tool certification,
    access control events, monitoring health, retirement status."""

    # ── Agent inventory from config/agent_tasks.json ────────────────
    agents_data = _load_json(AGENT_TASKS)
    agents = agents_data.get('agents', []) if isinstance(agents_data, dict) else []
    total_agents = len(agents)
    status_counts = {}
    module_groups = {}
    for a in agents:
        st = a.get('status', 'unknown')
        status_counts[st] = status_counts.get(st, 0) + 1
        mod = a.get('module', 'unknown')
        group = mod.split('.')[0] if '.' in mod else mod
        module_groups[group] = module_groups.get(group, 0) + 1

    built_agents = status_counts.get('built', 0)
    planned_agents = status_counts.get('planned', 0)
    partial_agents = status_counts.get('partial', 0)
    certified_pct = round(built_agents / total_agents * 100) if total_agents else 0

    # ── MCP protocol health from agentic_mcp_report.json ────────────
    mcp = _load_json(MCP_REPORT) or {}
    mcp_summary = mcp.get('mcp_summary', {})
    a2a_summary = mcp.get('a2a_summary', {})
    telemetry = mcp.get('telemetry', {})
    mcp_health = mcp_summary.get('status', 'unknown')
    mcp_issues = mcp_summary.get('issues', [])
    mcp_recommendations = mcp_summary.get('recommendations', [])
    a2a_health = a2a_summary.get('status', 'unknown')
    a2a_issues = a2a_summary.get('issues', [])
    n_messages = telemetry.get('messages', 0)
    n_audit_entries = telemetry.get('audit_entries', 0)
    n_disease_agents = mcp.get('n_agents', 0)
    pattern = mcp.get('pattern', 'unknown')

    # ── Pipeline governance from enterprise_pipelines.json ──────────
    pipe_data = _load_json(PIPELINES) or {}
    pipe_status = {'built': 0, 'partial': 0, 'planned': 0}
    total_pipelines = 0
    for g in pipe_data.get('groups', []):
        for p in g.get('pipelines', []):
            st = p.get('status', 'unknown')
            pipe_status[st] = pipe_status.get(st, 0) + 1
            total_pipelines += 1
    pipe_governance_pct = round(pipe_status['built'] / total_pipelines * 100) if total_pipelines else 0

    # ── Access control from transaction_log ─────────────────────────
    blocked_events = 0
    daily_activity = []
    action_dist = []
    if os.path.exists(DB):
        conn = _conn()
        cur = conn.cursor()
        blocked_events = _safe_count(cur,
            "SELECT count(*) FROM transaction_log WHERE action = 'blocked'")
        # Daily agent activity trend
        try:
            cur.execute("""
                SELECT date(ts_utc) as d, count(*) as cnt
                FROM transaction_log
                GROUP BY d ORDER BY d DESC LIMIT 14
            """)
            daily_activity = [{'date': r[0], 'events': r[1]}
                              for r in cur.fetchall()][::-1]
        except Exception:
            pass
        # Action distribution (top actions)
        try:
            cur.execute("""
                SELECT action, count(*) as cnt
                FROM transaction_log
                GROUP BY action ORDER BY cnt DESC LIMIT 10
            """)
            action_dist = [{'action': r[0], 'count': r[1]}
                           for r in cur.fetchall()]
        except Exception:
            pass
        total_events = _safe_count(cur, "SELECT count(*) FROM transaction_log")
        conn.close()
    else:
        total_events = 0

    return {
        'available': True,
        'summary': {
            'total_agents': total_agents,
            'built_agents': built_agents,
            'planned_agents': planned_agents,
            'partial_agents': partial_agents,
            'certified_pct': certified_pct,
            'total_pipelines': total_pipelines,
            'built_pipelines': pipe_status['built'],
            'pipeline_governance_pct': pipe_governance_pct,
            'mcp_health': mcp_health,
            'a2a_health': a2a_health,
            'blocked_events': blocked_events,
            'total_events': total_events,
            'n_messages': n_messages,
            'n_audit_entries': n_audit_entries,
            'n_disease_agents': n_disease_agents,
            'mcp_pattern': pattern,
        },
        'agent_status_distribution': [
            {'status': k, 'count': v}
            for k, v in sorted(status_counts.items(), key=lambda x: -x[1])
        ],
        'module_groups': [
            {'group': k, 'count': v}
            for k, v in sorted(module_groups.items(), key=lambda x: -x[1])
        ],
        'pipeline_status': [
            {'status': k, 'count': v}
            for k, v in sorted(pipe_status.items(), key=lambda x: -x[1])
        ],
        'mcp_issues': mcp_issues,
        'mcp_recommendations': mcp_recommendations,
        'a2a_issues': a2a_issues,
        'daily_activity': daily_activity,
        'action_distribution': action_dist,
    }


def mcp_governance_breakdown():
    """Per-agent inventory with governance details; pipeline-level status;
    MCP protocol voting/consensus detail."""

    # Agent inventory
    agents_data = _load_json(AGENT_TASKS) or {}
    agents = agents_data.get('agents', []) if isinstance(agents_data, dict) else []
    agent_inventory = []
    for a in agents:
        mod = a.get('module', 'unknown')
        group = mod.split('.')[0] if '.' in mod else mod
        agent_inventory.append({
            'id': a.get('id', ''),
            'task': a.get('task', ''),
            'module': mod,
            'group': group,
            'status': a.get('status', 'unknown'),
            'certified': a.get('status') == 'built',
        })

    # Pipeline breakdown
    pipe_data = _load_json(PIPELINES) or {}
    pipeline_breakdown = []
    for g in pipe_data.get('groups', []):
        group_pipes = []
        for p in g.get('pipelines', []):
            group_pipes.append({
                'name': p.get('name', ''),
                'status': p.get('status', 'unknown'),
                'stages': p.get('stages', []),
                'maps_to': p.get('maps_to', ''),
            })
        pipeline_breakdown.append({
            'group': g.get('group', ''),
            'pipelines': group_pipes,
            'total': len(group_pipes),
            'built': sum(1 for p in group_pipes if p['status'] == 'built'),
        })

    # MCP voting/consensus
    mcp = _load_json(MCP_REPORT) or {}
    votes = mcp.get('votes', [])
    consensus_records = []
    for v in votes:
        consensus_records.append({
            'disease': v.get('disease', ''),
            'label': v.get('label', ''),
            'probability': v.get('disease_prob', 0),
        })

    return {
        'available': True,
        'agent_inventory': agent_inventory,
        'pipeline_breakdown': pipeline_breakdown,
        'consensus_records': consensus_records,
    }


def mcp_governance_definitions():
    """Definitions for MCP governance terms, metrics, and compliance refs."""
    return {
        'available': True,
        'definitions': [
            {
                'term': 'MCP (Model Context Protocol)',
                'definition': 'Standard protocol for agent-to-tool communication, enabling structured tool registration, access control, and audit trailing across multi-agent systems.',
            },
            {
                'term': 'Agent Certification',
                'definition': 'Status indicating an agent has been verified as functional (built) with correct module bindings, task specifications, and output contracts.',
            },
            {
                'term': 'Tool Registration',
                'definition': 'Process of declaring an agent/tool in the governance registry (agent_tasks.json) with id, task description, module path, and operational status.',
            },
            {
                'term': 'Access Control Event',
                'definition': 'Transaction log entry where an operation was blocked by guardrails, authorization rules, or safety constraints (action=blocked).',
            },
            {
                'term': 'A2A (Agent-to-Agent) Health',
                'definition': 'Overall connectivity and trust level between agents in the multi-agent system, derived from message routing, schema compliance, and coordination patterns.',
            },
            {
                'term': 'Pipeline Governance Coverage',
                'definition': 'Percentage of enterprise pipelines that have been fully built and verified (status=built) out of the total registered pipeline catalog.',
            },
            {
                'term': 'MCP Telemetry',
                'definition': 'Protocol-level metrics including message counts, audit entries, cycle detection, and anomaly flags from the MCP communication layer.',
            },
            {
                'term': 'Consensus Voting',
                'definition': 'Multi-agent decision-making pattern where disease-specialist agents vote on classification outcomes; used in Mixture-of-Agents architecture for clinical confidence.',
            },
            {
                'term': 'Module Group',
                'definition': 'Logical grouping of agents by their Python module prefix (e.g., eeg_ingest, governance, rag), indicating functional domain alignment.',
            },
            {
                'term': 'Residual Risk (ISO 14971)',
                'definition': 'Risk remaining after mitigation controls are applied. In agent governance, this includes unmonitored agent behaviors, uncertified tools, and incomplete access control.',
            },
        ],
        'compliance_refs': [
            'EU AI Act Art. 9 — Risk management system for high-risk AI',
            'ISO 14971 — Medical device risk management',
            'IEC 62304 — Medical device software lifecycle',
            'FDA AI/ML SaMD — Software as Medical Device guidance',
            'NIST AI RMF — AI Risk Management Framework',
        ],
        'remediation': [
            'Certify all planned/partial agents before production deployment',
            'Enable MCP audit logging for all agent-to-tool interactions',
            'Implement role-based access control for agent tool invocations',
            'Monitor A2A trust levels and enforce message schema compliance',
            'Review blocked events for access control policy tuning',
        ],
    }


if __name__ == '__main__':
    import json
    print('=== OVERVIEW ===')
    ov = mcp_governance_overview()
    print(json.dumps(ov.get('summary', {}), indent=2))
    print(f"\nAgent groups: {len(ov.get('module_groups', []))}")
    print(f"Daily activity points: {len(ov.get('daily_activity', []))}")
    print(f"Action types: {len(ov.get('action_distribution', []))}")

    print('\n=== BREAKDOWN ===')
    br = mcp_governance_breakdown()
    print(f"Agent inventory: {len(br.get('agent_inventory', []))}")
    print(f"Pipeline groups: {len(br.get('pipeline_breakdown', []))}")
    print(f"Consensus records: {len(br.get('consensus_records', []))}")

    print('\n=== DEFINITIONS ===')
    df = mcp_governance_definitions()
    print(f"Terms: {len(df.get('definitions', []))}")
