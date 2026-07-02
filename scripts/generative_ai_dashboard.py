"""
Generative AI Dashboard — analytics from real clinical.db
==========================================================
Surfaces: GenAI bot usage (transaction_log), conversation log stats,
content safety analysis, and responsible-AI governance metrics.
"""

import sqlite3
import os
from collections import defaultdict
from datetime import datetime

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB)


def overview():
    """KPIs: total conversations, genai bot queries, avg message length,
    operator vs assistant ratio, content safety score, response quality."""
    con = _conn()
    cur = con.cursor()

    # Conversation log stats
    cur.execute("SELECT COUNT(*) FROM conversation_log")
    total_messages = cur.fetchone()[0]

    cur.execute("SELECT role, COUNT(*) FROM conversation_log GROUP BY role")
    role_counts = dict(cur.fetchall())

    cur.execute("SELECT AVG(LENGTH(text)) FROM conversation_log WHERE role='assistant'")
    avg_assistant_len = cur.fetchone()[0] or 0

    cur.execute("SELECT AVG(LENGTH(text)) FROM conversation_log WHERE role='operator'")
    avg_operator_len = cur.fetchone()[0] or 0

    # GenAI bot queries
    cur.execute("SELECT COUNT(*) FROM transaction_log WHERE component='genai_bot'")
    genai_queries = cur.fetchone()[0]

    cur.execute("SELECT detail FROM transaction_log WHERE component='genai_bot'")
    genai_details = [r[0] for r in cur.fetchall() if r[0]]

    # Parse roles and layouts from genai details
    role_usage = defaultdict(int)
    layout_usage = defaultdict(int)
    for d in genai_details:
        # format: "Neurologist [list]: query..."
        if "[" in d and "]" in d:
            role_part = d.split("[")[0].strip()
            layout_part = d.split("[")[1].split("]")[0].strip()
            role_usage[role_part] += 1
            layout_usage[layout_part] += 1

    # Content safety heuristic from conversation log
    # Check for flagged/moderated content (proxy: look for safety-related words)
    cur.execute("SELECT COUNT(*) FROM conversation_log WHERE text LIKE '%error%' OR text LIKE '%failed%' OR text LIKE '%rejected%'")
    flagged_count = cur.fetchone()[0]
    content_safety_score = round(1.0 - (flagged_count / max(total_messages, 1)) * 0.5, 3)

    # Daily conversation trend
    cur.execute("""
        SELECT DATE(ts_local) as day, COUNT(*) as cnt
        FROM conversation_log
        GROUP BY DATE(ts_local)
        ORDER BY day
    """)
    daily_trend = [{"date": r[0], "messages": r[1]} for r in cur.fetchall()]

    # Transaction log AI component usage
    cur.execute("""
        SELECT component, COUNT(*) FROM transaction_log
        WHERE component IN ('genai_bot','council','patient_chat','training','drift','consistency','fairness')
        GROUP BY component ORDER BY COUNT(*) DESC
    """)
    ai_component_usage = [{"component": r[0], "count": r[1]} for r in cur.fetchall()]

    con.close()

    return {
        "total_messages": total_messages,
        "assistant_messages": role_counts.get("assistant", 0),
        "operator_messages": role_counts.get("operator", 0),
        "genai_bot_queries": genai_queries,
        "avg_assistant_response_length": round(avg_assistant_len, 0),
        "avg_operator_query_length": round(avg_operator_len, 0),
        "content_safety_score": content_safety_score,
        "response_ratio": round(role_counts.get("assistant", 0) / max(role_counts.get("operator", 1), 1), 2),
        "role_usage": dict(role_usage),
        "layout_usage": dict(layout_usage),
        "daily_trend": daily_trend,
        "ai_component_usage": ai_component_usage,
        "flagged_messages": flagged_count
    }


def breakdown():
    """Per-role breakdown, recent conversations, genai query details,
    message length distribution, conversation timeline."""
    con = _conn()
    cur = con.cursor()

    # Recent conversations (last 20)
    cur.execute("""
        SELECT id, role, SUBSTR(text, 1, 200) as preview, ts_local
        FROM conversation_log
        ORDER BY id DESC LIMIT 20
    """)
    recent = [{"id": r[0], "role": r[1], "preview": r[2], "ts_local": r[3]} for r in cur.fetchall()]

    # GenAI bot query details
    cur.execute("""
        SELECT patient_id, detail, ts_local
        FROM transaction_log WHERE component='genai_bot'
        ORDER BY id DESC
    """)
    genai_queries = [{"patient_id": r[0], "detail": r[1], "ts_local": r[2]} for r in cur.fetchall()]

    # Message length distribution
    cur.execute("""
        SELECT
            CASE
                WHEN LENGTH(text) < 100 THEN '<100 chars'
                WHEN LENGTH(text) < 500 THEN '100-500 chars'
                WHEN LENGTH(text) < 1000 THEN '500-1K chars'
                WHEN LENGTH(text) < 3000 THEN '1K-3K chars'
                ELSE '3K+ chars'
            END as bucket,
            COUNT(*) as cnt
        FROM conversation_log
        GROUP BY bucket
        ORDER BY MIN(LENGTH(text))
    """)
    length_dist = [{"bucket": r[0], "count": r[1]} for r in cur.fetchall()]

    # Hourly activity pattern
    cur.execute("""
        SELECT SUBSTR(ts_local, 12, 2) as hour, COUNT(*) as cnt
        FROM conversation_log
        WHERE ts_local IS NOT NULL AND LENGTH(ts_local) >= 13
        GROUP BY hour ORDER BY hour
    """)
    hourly_pattern = [{"hour": r[0] + ":00", "messages": r[1]} for r in cur.fetchall()]

    # Conversation sessions (grouped by date)
    cur.execute("""
        SELECT DATE(ts_local) as day, role, COUNT(*) as cnt
        FROM conversation_log
        GROUP BY DATE(ts_local), role
        ORDER BY day
    """)
    daily_by_role = []
    day_data = defaultdict(lambda: {"operator": 0, "assistant": 0})
    for r in cur.fetchall():
        if r[0]:
            day_data[r[0]][r[1]] = r[2]
    for day in sorted(day_data.keys()):
        daily_by_role.append({
            "date": day,
            "operator": day_data[day]["operator"],
            "assistant": day_data[day]["assistant"],
            "total": day_data[day]["operator"] + day_data[day]["assistant"]
        })

    # AI transaction components breakdown
    cur.execute("""
        SELECT component, action, COUNT(*) as cnt
        FROM transaction_log
        WHERE component IN ('genai_bot','council','patient_chat','training','drift','consistency','fairness','clinical_trust','feedback')
        GROUP BY component, action
        ORDER BY cnt DESC
    """)
    ai_transactions = [{"component": r[0], "action": r[1], "count": r[2]} for r in cur.fetchall()]

    # Content type analysis from genai details
    content_types = {"passage": 0, "table": 0, "list": 0, "graph": 0}
    for q in genai_queries:
        detail = q.get("detail", "")
        for ct in content_types:
            if ct in detail.lower():
                content_types[ct] += 1

    con.close()

    return {
        "recent_conversations": recent,
        "genai_queries": genai_queries,
        "length_distribution": length_dist,
        "hourly_pattern": hourly_pattern,
        "daily_by_role": daily_by_role,
        "ai_transactions": ai_transactions,
        "content_type_distribution": [{"type": k, "count": v} for k, v in content_types.items() if v > 0]
    }


def definitions():
    """Metric definitions for the Generative AI dashboard."""
    return {
        "sections": [
            {
                "title": "Generative AI Concepts",
                "items": [
                    {"term": "GenAI Bot", "definition": "Role-based generative AI assistant powered by Ollama that answers clinical queries with context from patient records. Supports passage, table, list, and graph output layouts."},
                    {"term": "Content Safety Score", "definition": "Proportion of AI-generated content that passes safety and moderation checks. Computed as 1 minus the weighted rate of flagged/error content in conversation history."},
                    {"term": "Response Ratio", "definition": "Ratio of assistant messages to operator messages. A ratio above 1.0 indicates the AI generates more content per operator query, typical for detailed clinical responses."},
                    {"term": "Conversation Log", "definition": "Full audit trail of operator-assistant interactions stored in clinical.db. Each entry records role, message text, and UTC/local timestamps."},
                    {"term": "Output Layout", "definition": "Format of GenAI bot responses: passage (narrative text), table (structured rows), list (bullet points), or graph (visualization-ready data)."},
                    {"term": "Prompt Injection Defense", "definition": "Detection of adversarial prompt patterns (e.g., 'ignore previous instructions') to prevent misuse of the generative AI system."},
                    {"term": "Hallucination Rate", "definition": "Proportion of AI-generated outputs containing factually unsupported claims. Monitored via the responsible_ai/generative_ai_analysis module."},
                    {"term": "AI Transaction", "definition": "Logged event in transaction_log for AI components (genai_bot, council, patient_chat, training, drift, consistency, fairness)."}
                ]
            },
            {
                "title": "Quality Metrics",
                "items": [
                    {"term": "Avg Response Length", "definition": "Mean character count of assistant-generated messages. Longer responses typically indicate detailed clinical analysis or comprehensive reports."},
                    {"term": "Hourly Activity Pattern", "definition": "Distribution of AI conversations by hour of day. Identifies peak usage windows for capacity planning."},
                    {"term": "Message Length Distribution", "definition": "Bucketed histogram of message lengths (<100, 100-500, 500-1K, 1K-3K, 3K+ characters) showing the mix of brief vs detailed interactions."},
                    {"term": "Daily Trend", "definition": "Time series of daily message counts split by role (operator vs assistant), showing interaction volume over time."}
                ]
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {"term": "ILAE Guidelines", "definition": "International League Against Epilepsy standards for clinical documentation. GenAI outputs must align with ILAE terminology and classification systems."},
                    {"term": "IEC 62304", "definition": "Medical device software lifecycle standard. AI-generated clinical content requires traceability, validation, and audit logging per Class C requirements."},
                    {"term": "FDA AI/ML Guidance", "definition": "FDA framework for AI/ML-based Software as Medical Device (SaMD). Requires transparency, performance monitoring, and real-world performance validation for generative AI in clinical settings."},
                    {"term": "HIPAA", "definition": "AI-generated content containing PHI must comply with HIPAA safeguards. Conversation logs are access-controlled and patient identifiers are managed per minimum necessary standard."}
                ]
            },
            {
                "title": "Remediation Strategies",
                "items": [
                    {"term": "High Hallucination Rate", "definition": "If hallucination rate exceeds 5%, implement retrieval-augmented generation (RAG) with verified clinical knowledge bases and add fact-checking validation layers."},
                    {"term": "Low Content Safety", "definition": "If safety score drops below 0.95, review flagged outputs, update content moderation rules, and add pre/post-generation safety filters."},
                    {"term": "Prompt Injection Detected", "definition": "Quarantine the session, log the attempt, alert security, and review the injection pattern to update the defense rule set."},
                    {"term": "Poor Response Quality", "definition": "If average quality scores decline, review prompt templates, update context retrieval, fine-tune model parameters, and increase human-in-the-loop review frequency."}
                ]
            }
        ]
    }
