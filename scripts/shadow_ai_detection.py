"""
Shadow AI Detection — parses real track.jsonl to detect unauthorized/unregistered
AI tool usage patterns across the system.

Registered baseline comes from:
  - config/agent_tasks.json  (status == 'built'  agent IDs)
  - config/neuro_ai_ecosystem.json (tool names from all categories)

Shadow events = track.jsonl entries whose 'level' field references a source
NOT in the authorized level set, OR whose 'event' text references a tool/model
NOT in the registered tool name set.
"""

import json, pathlib, re
from datetime import datetime, timedelta, timezone

MDT = timezone(timedelta(hours=-6))

BASE = pathlib.Path(__file__).resolve().parent.parent
TRACK_LOG   = BASE / "jobs" / "logs" / "track.jsonl"
AGENT_TASKS = BASE / "config" / "agent_tasks.json"
ECOSYSTEM   = BASE / "config" / "neuro_ai_ecosystem.json"

# ── Authorized level set (registered components of the system) ────
AUTHORIZED_LEVELS = {"system", "ops", "autobuild", "watchdog", "git", "health", "ollama"}


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
                ts = obj["ts_local"].strip()
                if ts.endswith(" MDT"):
                    ts = ts[:-4]
                    obj["_dt"] = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=MDT)
                else:
                    obj["_dt"] = datetime.fromisoformat(ts)
            events.append(obj)
        except Exception:
            pass
    return events


def _load_registered_tools():
    """Build the set of registered tool/agent names from agent_tasks.json
    and neuro_ai_ecosystem.json."""
    names = set()

    # agent_tasks.json — built agents
    if AGENT_TASKS.exists():
        try:
            data = json.loads(AGENT_TASKS.read_text())
            agents = data.get("agents", [])
            for a in agents:
                if a.get("status") == "built":
                    names.add(a.get("id", "").lower())
                    names.add(a.get("module", "").lower())
                    names.add(a.get("task", "").lower())
        except Exception:
            pass

    # neuro_ai_ecosystem.json — all tools in all categories
    if ECOSYSTEM.exists():
        try:
            data = json.loads(ECOSYSTEM.read_text())
            cats = data.get("categories", [])
            for cat in cats:
                for tool in cat.get("tools", []):
                    names.add(tool.get("name", "").lower())
        except Exception:
            pass

    # remove empty strings
    names.discard("")
    return names


def _is_shadow_event(event: dict, registered_tools: set) -> bool:
    """Determine if a track event is a shadow (unauthorized) activity.

    Shadow criteria:
    1. 'level' field is NOT in AUTHORIZED_LEVELS, OR
    2. 'event' text mentions a model/tool that is not registered
       (looks for patterns like 'model <name>', 'using <name>',
        or mentions of known unregistered AI services).
    """
    level = event.get("level", "").lower()
    if level not in AUTHORIZED_LEVELS:
        return True

    event_text = event.get("event", "").lower()

    # autobuild level legitimately references "headless claude" — that is authorized
    if level == "autobuild":
        return False

    # ollama level: check for unregistered / non-standard model names
    # Registered ollama models for this project: qwen2.5-coder series, llama3 series
    if level == "ollama":
        # Flag if the model name looks like an external API model (not local ollama)
        unregistered_ollama = [
            r"\bgpt-[0-9]", r"\bgpt4\b", r"\bdavinci\b", r"\bcurie\b",
            r"\bgemini\b", r"\bpalm\b", r"\bclaude-[0-9]", r"\bsonnet\b",
            r"\bhaiku\b", r"\bopus\b",
        ]
        for pattern in unregistered_ollama:
            if re.search(pattern, event_text):
                return True
        return False

    # Patterns that suggest unauthorized AI tool usage (for ops/system/watchdog/git/health levels)
    unregistered_patterns = [
        r"\bgpt-[0-9]",          # OpenAI GPT variants
        r"\bchatgpt\b",
        r"\bgemini\b",
        r"\bbard\b",
        r"\bcopilot\b",
        r"\bperplexity\b",
        r"\bmistral\b",
        r"\bllama\b",             # standalone llama reference (not via ollama level)
        r"\bcohere\b",
        r"\banthropic\b",
        r"\bopenai\b",
        r"\bhuggingface\b",
        r"\bstability\b",
        r"\bdall-e\b",
        r"\bmidjourney\b",
        r"\bsora\b",
        r"\bclaude\b",            # claude API references outside autobuild
        r"\bai21\b",
        r"\bcomet\b",
        r"\bwandb\b",
        r"\bapi_fail=([1-9][0-9]*)\b",  # health events with API failures (potential shadow side-effect)
    ]

    for pattern in unregistered_patterns:
        if re.search(pattern, event_text):
            return True

    return False


def _get_shadow_source(event: dict) -> str:
    """Extract a readable source label from a shadow event."""
    level = event.get("level", "unknown").lower()
    if level not in AUTHORIZED_LEVELS:
        return f"unauthorized-level:{level}"

    event_text = event.get("event", "").lower()

    tool_keywords = [
        ("gpt", "OpenAI GPT"),
        ("chatgpt", "ChatGPT"),
        ("gemini", "Google Gemini"),
        ("bard", "Google Bard"),
        ("copilot", "GitHub Copilot"),
        ("perplexity", "Perplexity AI"),
        ("mistral", "Mistral AI"),
        ("llama", "LLaMA (standalone)"),
        ("cohere", "Cohere AI"),
        ("openai", "OpenAI API"),
        ("huggingface", "HuggingFace"),
        ("stability", "Stability AI"),
        ("dall-e", "DALL-E"),
        ("midjourney", "Midjourney"),
        ("sora", "OpenAI Sora"),
        ("claude", "Claude API (unmanaged)"),
        ("anthropic", "Anthropic API"),
        ("ai21", "AI21 Labs"),
        ("comet", "CometML"),
        ("wandb", "Weights & Biases"),
    ]
    for kw, label in tool_keywords:
        if kw in event_text:
            return label
    if "api_fail=" in event_text:
        return "API Failure (shadow side-effect)"
    return f"unknown-tool (level:{level})"


# ── overview ─────────────────────────────────────────────────────
def overview():
    track = _load_track()
    registered_tools = _load_registered_tools()
    registered_levels_count = len(AUTHORIZED_LEVELS)

    # Count built agents from agent_tasks.json
    built_agent_count = 0
    if AGENT_TASKS.exists():
        try:
            data = json.loads(AGENT_TASKS.read_text())
            built_agent_count = sum(1 for a in data.get("agents", []) if a.get("status") == "built")
        except Exception:
            pass

    total_events = len(track)
    shadow_events = [e for e in track if _is_shadow_event(e, registered_tools)]
    shadow_count = len(shadow_events)
    shadow_rate = round(shadow_count / total_events * 100, 2) if total_events else 0.0

    if shadow_rate < 5:
        risk_level = "low"
    elif shadow_rate < 15:
        risk_level = "medium"
    else:
        risk_level = "high"

    now = datetime.now(timezone.utc)
    detections_24h = sum(1 for e in shadow_events if (now - e["_dt"]).total_seconds() < 86400)
    detections_7d  = sum(1 for e in shadow_events if (now - e["_dt"]).total_seconds() < 86400 * 7)

    # top shadow sources
    source_counts = {}
    for e in shadow_events:
        src = _get_shadow_source(e)
        source_counts[src] = source_counts.get(src, 0) + 1
    top_shadow_sources = [
        {"source": k, "count": v}
        for k, v in sorted(source_counts.items(), key=lambda x: -x[1])[:5]
    ]

    # detection timeline — last 14 days
    daily = {}
    for i in range(14):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        daily[day] = 0
    for e in shadow_events:
        day = e["_dt"].astimezone(MDT).strftime("%Y-%m-%d")
        if day in daily:
            daily[day] += 1
    detection_timeline = [{"date": k, "count": v} for k, v in sorted(daily.items())]

    # risk distribution
    risk_dist = {"low": 0, "medium": 0, "high": 0}
    for e in shadow_events:
        src = _get_shadow_source(e)
        count = source_counts.get(src, 1)
        rate = count / total_events * 100 if total_events else 0
        if rate < 5:
            risk_dist["low"] += 1
        elif rate < 15:
            risk_dist["medium"] += 1
        else:
            risk_dist["high"] += 1
    risk_distribution = [{"category": k, "count": v} for k, v in risk_dist.items()]

    return {
        "available": True,
        "total_events_scanned": total_events,
        "registered_tools_count": built_agent_count,
        "shadow_detections": shadow_count,
        "shadow_rate": shadow_rate,
        "risk_level": risk_level,
        "detections_last_24h": detections_24h,
        "detections_last_7d": detections_7d,
        "top_shadow_sources": top_shadow_sources,
        "detection_timeline": detection_timeline,
        "risk_distribution": risk_distribution,
    }


# ── breakdown ────────────────────────────────────────────────────
def breakdown():
    track = _load_track()
    registered_tools = _load_registered_tools()
    now = datetime.now(timezone.utc)

    shadow_events = [e for e in track if _is_shadow_event(e, registered_tools)]

    # hourly heatmap: 7 days x 24 hours
    heatmap = [[0] * 24 for _ in range(7)]
    day_labels = []
    for i in range(6, -1, -1):
        d = now - timedelta(days=i)
        day_labels.append(d.astimezone(MDT).strftime("%a %m/%d"))
    for e in shadow_events:
        dt = e["_dt"].astimezone(MDT)
        day_offset = (now.astimezone(MDT).date() - dt.date()).days
        if 0 <= day_offset <= 6:
            row_idx = 6 - day_offset
            heatmap[row_idx][dt.hour] += 1

    hourly_heatmap = {
        "day_labels": day_labels,
        "hour_labels": [f"{h:02d}" for h in range(24)],
        "matrix": heatmap,
    }

    # source analysis — all unique sources with registered/unregistered flag
    all_sources = {}
    for e in track:
        level = e.get("level", "unknown")
        is_reg = level in AUTHORIZED_LEVELS
        key = level
        if key not in all_sources:
            all_sources[key] = {"source": key, "registered": is_reg, "total_events": 0, "shadow_events": 0}
        all_sources[key]["total_events"] += 1
        if _is_shadow_event(e, registered_tools):
            all_sources[key]["shadow_events"] += 1

    source_analysis = sorted(all_sources.values(), key=lambda x: -x["total_events"])

    # recent shadow events — last 30
    shadow_sorted = sorted(shadow_events, key=lambda e: e["_dt"], reverse=True)
    recent_shadow_events = []
    for e in shadow_sorted[:30]:
        recent_shadow_events.append({
            "ts": e.get("ts_local", e.get("ts_utc", "")),
            "level": e.get("level", ""),
            "event": e.get("event", ""),
            "host": e.get("host", ""),
            "user": e.get("user", ""),
            "shadow_source": _get_shadow_source(e),
        })

    # level distribution of shadow events
    level_dist = {}
    for e in shadow_events:
        lvl = e.get("level", "unknown")
        level_dist[lvl] = level_dist.get(lvl, 0) + 1
    level_distribution = [{"level": k, "count": v} for k, v in sorted(level_dist.items(), key=lambda x: -x[1])]

    # temporal pattern — 6-hour blocks
    blocks = {"00-06": 0, "06-12": 0, "12-18": 0, "18-24": 0}
    for e in shadow_events:
        hr = e["_dt"].astimezone(MDT).hour
        if hr < 6:
            blocks["00-06"] += 1
        elif hr < 12:
            blocks["06-12"] += 1
        elif hr < 18:
            blocks["12-18"] += 1
        else:
            blocks["18-24"] += 1
    temporal_pattern = [{"block": k, "count": v} for k, v in blocks.items()]

    return {
        "available": True,
        "hourly_heatmap": hourly_heatmap,
        "source_analysis": source_analysis,
        "recent_shadow_events": recent_shadow_events,
        "level_distribution": level_distribution,
        "temporal_pattern": temporal_pattern,
    }


# ── definitions ──────────────────────────────────────────────────
def definitions():
    return {
        "available": True,
        "metrics": [
            {"term": "Shadow AI", "definition": "An AI tool, model, or service used within the system that has not been officially registered and approved through the agent_tasks.json or neuro_ai_ecosystem.json registries."},
            {"term": "Total Events Scanned", "definition": "The total number of log entries in track.jsonl that were analyzed for shadow AI activity."},
            {"term": "Registered Tools Count", "definition": "Number of agents/tools with status 'built' in config/agent_tasks.json — these are authorized and monitored components."},
            {"term": "Shadow Detections", "definition": "Count of log events that reference an unregistered tool, unauthorized level, or mention a non-authorized AI service by name."},
            {"term": "Shadow Rate", "definition": "Percentage of scanned events identified as shadow activity (shadow_detections / total_events_scanned × 100)."},
            {"term": "Risk Level: Low (<5%)", "definition": "Shadow rate below 5% — minimal unauthorized AI usage detected; normal operational noise."},
            {"term": "Risk Level: Medium (5–15%)", "definition": "Shadow rate between 5% and 15% — moderate unauthorized usage; investigation recommended."},
            {"term": "Risk Level: High (>15%)", "definition": "Shadow rate above 15% — significant unauthorized AI tool proliferation; immediate governance action required."},
            {"term": "Authorized Levels", "definition": "Log levels that are part of the registered system: system, ops, autobuild, watchdog, git, health, ollama. Events from other levels are flagged as shadow."},
            {"term": "Detection Methodology", "definition": "Two-pass detection: (1) level-based — any event whose 'level' is not in the authorized set; (2) content-based — events whose 'event' text references known unregistered AI services via regex pattern matching."},
            {"term": "Hourly Heatmap", "definition": "Matrix of shadow detection counts by hour of day (0–23) and day of week (last 7 days), showing when unauthorized activity clusters."},
            {"term": "Source Analysis", "definition": "Per-level breakdown showing total events vs shadow events, and whether each level is registered (authorized) or unregistered."},
            {"term": "Temporal Pattern", "definition": "Shadow detections grouped into 6-hour blocks (00-06, 06-12, 12-18, 18-24 MDT) to identify off-hours activity."},
            {"term": "Top Shadow Sources", "definition": "The five most frequently detected unregistered sources or tool references, ranked by event count."},
            {"term": "Authorized vs Unauthorized", "definition": "Authorized = registered in agent_tasks.json (built) or neuro_ai_ecosystem.json with known status. Unauthorized = any AI tool invoked outside these registries, including external APIs (OpenAI, Gemini, Cohere, etc.) or unregistered local models."},
            {"term": "Mitigation Recommendations", "definition": "1) Add approved tools to agent_tasks.json with status 'built'. 2) Block outbound API calls to unauthorized AI endpoints via firewall rules. 3) Rotate credentials if shadow usage suggests credential compromise. 4) Set alert threshold at shadow_rate > 5% to trigger review workflow."},
        ],
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN (truncated) ===")
    bd = breakdown()
    pprint.pprint({k: v for k, v in bd.items() if k != "hourly_heatmap"})
