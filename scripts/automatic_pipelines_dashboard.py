"""Automatic Pipelines Dashboard — end-to-end pipeline status, stage counts,
trigger types, and automation rates from config/automatic_pipelines.json."""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: pipeline counts by status, trigger distribution,
    stage statistics, automation rate."""
    cfg = _load('automatic_pipelines.json')
    if not cfg:
        return {"available": False, "note": "automatic_pipelines.json missing"}

    pipelines = cfg.get('pipelines', [])
    total = len(pipelines)

    # Status counts
    status_cnt = Counter(p.get('status', '?') for p in pipelines)
    automatic = status_cnt.get('automatic', 0)
    semi = status_cnt.get('semi', 0)
    planned = status_cnt.get('planned', 0)
    automation_pct = round(automatic / total * 100, 1) if total else 0

    # Trigger distribution
    trigger_cnt = Counter()
    for p in pipelines:
        t = p.get('trigger', 'unknown')
        # Normalize triggers into categories
        tl = t.lower()
        if 'upload' in tl:
            trigger_cnt['upload'] += 1
        elif 'cron' in tl or 'scheduled' in tl:
            trigger_cnt['scheduled'] += 1
        elif 'query' in tl:
            trigger_cnt['query'] += 1
        elif 'stream' in tl:
            trigger_cnt['stream'] += 1
        elif 'on-demand' in tl:
            trigger_cnt['on-demand'] += 1
        else:
            trigger_cnt['other'] += 1

    # Stage statistics
    stage_counts = [len(p.get('stages', [])) for p in pipelines]
    total_stages = sum(stage_counts)
    avg_stages = round(total_stages / total, 1) if total else 0
    max_stages = max(stage_counts) if stage_counts else 0
    min_stages = min(stage_counts) if stage_counts else 0

    # Endpoint type distribution
    endpoint_types = Counter()
    for p in pipelines:
        ep = p.get('endpoint', '')
        if ep.startswith('GET '):
            endpoint_types['GET'] += 1
        elif ep.startswith('POST '):
            endpoint_types['POST'] += 1
        elif ep.startswith('scripts/'):
            endpoint_types['script'] += 1
        else:
            endpoint_types['other'] += 1

    return {
        "available": True,
        "summary": {
            "total_pipelines": total,
            "automatic": automatic,
            "semi": semi,
            "planned": planned,
            "automation_pct": automation_pct,
            "total_stages": total_stages,
            "avg_stages_per_pipeline": avg_stages,
            "max_stages": max_stages,
            "min_stages": min_stages,
        },
        "status_distribution": dict(status_cnt),
        "trigger_distribution": dict(trigger_cnt),
        "endpoint_type_distribution": dict(endpoint_types),
    }


def breakdown():
    """Detailed view: all pipelines with stages, triggers, endpoints."""
    cfg = _load('automatic_pipelines.json')
    if not cfg:
        return {"available": False}

    pipelines = cfg.get('pipelines', [])

    pipeline_detail = []
    for p in pipelines:
        stages = p.get('stages', [])
        pipeline_detail.append({
            "process": p.get('process', '?'),
            "trigger": p.get('trigger', '?'),
            "endpoint": p.get('endpoint', '?'),
            "stages": stages,
            "stage_count": len(stages),
            "status": p.get('status', '?'),
        })

    # Sort: automatic first, then semi, then planned
    order = {'automatic': 0, 'semi': 1, 'planned': 2}
    pipeline_detail.sort(key=lambda x: order.get(x['status'], 9))

    return {
        "available": True,
        "pipelines": pipeline_detail,
        "meta": {
            "title": cfg.get('title', ''),
            "note": cfg.get('note', ''),
            "updated_at": cfg.get('updated_at', ''),
        },
    }


def definitions():
    """Automatic pipelines terminology and definitions."""
    return {
        "available": True,
        "definitions": [
            {"term": "Pipeline", "definition": "An end-to-end automated workflow that runs from trigger to output with no manual steps. Each pipeline has a defined trigger, stages, and an API endpoint or script entry point."},
            {"term": "Automatic", "definition": "Pipeline status indicating the full chain runs end-to-end via a single call or trigger — no human intervention required between stages."},
            {"term": "Semi-automatic", "definition": "Pipeline status indicating the chain exists but requires some manual input, configuration, or is only partially wired (e.g., streaming pipelines awaiting device integration)."},
            {"term": "Planned", "definition": "Pipeline identified in the roadmap but not yet implemented. No trigger, stages, or endpoint is wired."},
            {"term": "Trigger", "definition": "The event that initiates a pipeline run. Types include: file upload, scheduled/cron, on-demand (manual invocation), query (user input), and device stream."},
            {"term": "Stage", "definition": "A discrete processing step within a pipeline. Stages execute sequentially — the output of one feeds the next. Examples: parse, extract features, predict, write report."},
            {"term": "Endpoint", "definition": "The API route (GET/POST) or script path that serves as the entry point for triggering the pipeline."},
            {"term": "Automation Rate", "definition": "Percentage of pipelines with 'automatic' status out of total pipelines. Higher rates indicate more of the system runs without human intervention."},
            {"term": "On-demand", "definition": "A trigger type where the pipeline runs when explicitly requested by a user or another system, rather than on a schedule or event."},
            {"term": "Cron/Scheduled", "definition": "A trigger type where the pipeline runs automatically at fixed intervals (e.g., daily at 08:00) managed by the job scheduler."},
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
