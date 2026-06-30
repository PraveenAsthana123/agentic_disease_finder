"""
Patient Portal — Message Inbox Tab Module
==========================================
Secure message inbox derived from clinical.db:
  - Team Messages     (care team conversations, bot summaries)
  - Clinical Decisions (decisions requiring patient awareness)
  - Expert Reviews    (specialist review notifications)
  - Form Assignments  (forms/questionnaires assigned to patient)

All messages are derived from REAL data in clinical.db.
No fabricated or stubbed data — every row traces to a source table.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# Message category constants
CAT_TEAM = "team_message"
CAT_DECISION = "clinical_decision"
CAT_REVIEW = "expert_review"
CAT_FORM = "form_assignment"
CAT_SYSTEM = "system"

CATEGORY_LABELS = {
    CAT_TEAM: "Care Team",
    CAT_DECISION: "Clinical Decision",
    CAT_REVIEW: "Expert Review",
    CAT_FORM: "Form / Questionnaire",
    CAT_SYSTEM: "System Notification",
}

CATEGORY_ICONS = {
    CAT_TEAM: "\U0001f4ac",
    CAT_DECISION: "\u2696\ufe0f",
    CAT_REVIEW: "\U0001f50d",
    CAT_FORM: "\U0001f4cb",
    CAT_SYSTEM: "\u2699\ufe0f",
}


def _connect():
    if not DB.exists():
        return None
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


def _ts(row_ts):
    if not row_ts:
        return None
    try:
        return datetime.fromisoformat(str(row_ts).replace("Z", "+00:00")).isoformat()
    except Exception:
        return str(row_ts)


# ── Team Messages ────────────────────────────────────────────────

def _team_messages(conn, patient_id=None):
    """Get team messages from team_messages table."""
    if patient_id:
        rows = conn.execute(
            "SELECT * FROM team_messages WHERE patient_id = ? ORDER BY created_at DESC",
            [patient_id],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM team_messages ORDER BY created_at DESC"
        ).fetchall()

    messages = []
    for r in rows:
        read_by = []
        if r["read_by"]:
            try:
                import json
                read_by = json.loads(r["read_by"])
            except Exception:
                read_by = [r["read_by"]]

        messages.append({
            "id": f"tm-{r['id']}",
            "category": CAT_TEAM,
            "channel": r["channel"],
            "from_role": r["from_role"],
            "subject": r["topic"] or f"Message from {r['from_role']}",
            "body": r["text"],
            "patient_id": r["patient_id"],
            "is_bot": bool(r["is_bot"]),
            "attachment": r["attachment"],
            "read_by": read_by,
            "timestamp": _ts(r["created_at"]),
        })
    return messages


# ── Clinical Decisions ───────────────────────────────────────────

def _clinical_decisions(conn, patient_id=None):
    """Get clinical decision notifications."""
    if patient_id:
        rows = conn.execute(
            "SELECT * FROM clinical_decisions WHERE patient_id = ? ORDER BY created_at DESC",
            [patient_id],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM clinical_decisions ORDER BY created_at DESC"
        ).fetchall()

    messages = []
    cols = [desc[1] for desc in conn.execute("PRAGMA table_info(clinical_decisions)").fetchall()]

    for r in rows:
        d = dict(r)
        decision_text = d.get("decision") or d.get("recommendation") or d.get("action") or "Clinical decision recorded"
        clinician = d.get("clinician") or d.get("decided_by") or d.get("role") or "Clinician"
        confidence = d.get("confidence")

        messages.append({
            "id": f"cd-{d.get('id', 0)}",
            "category": CAT_DECISION,
            "channel": "clinical",
            "from_role": clinician,
            "subject": f"Clinical Decision — {d.get('patient_id', '')}",
            "body": decision_text,
            "patient_id": d.get("patient_id"),
            "is_bot": False,
            "confidence": confidence,
            "timestamp": _ts(d.get("created_at")),
        })
    return messages


# ── Expert Reviews ───────────────────────────────────────────────

def _expert_reviews(conn, patient_id=None):
    """Get expert review notifications."""
    if patient_id:
        rows = conn.execute(
            "SELECT * FROM expert_reviews WHERE patient_id = ? ORDER BY created_at DESC",
            [patient_id],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM expert_reviews ORDER BY created_at DESC"
        ).fetchall()

    messages = []
    for r in rows:
        d = dict(r)
        reviewer = d.get("reviewer") or d.get("expert") or d.get("role") or "Specialist"
        verdict = d.get("verdict") or d.get("decision") or d.get("recommendation") or ""
        notes = d.get("notes") or d.get("comments") or d.get("review_text") or ""

        body_parts = []
        if verdict:
            body_parts.append(f"Verdict: {verdict}")
        if notes:
            body_parts.append(notes)

        messages.append({
            "id": f"er-{d.get('id', 0)}",
            "category": CAT_REVIEW,
            "channel": "reviews",
            "from_role": reviewer,
            "subject": f"Expert Review — {d.get('patient_id', '')}",
            "body": " | ".join(body_parts) if body_parts else "Review completed",
            "patient_id": d.get("patient_id"),
            "is_bot": False,
            "timestamp": _ts(d.get("created_at")),
        })
    return messages


# ── Form Assignments ─────────────────────────────────────────────

def _form_assignments(conn, patient_id=None):
    """Get form/questionnaire assignment notifications."""
    if patient_id:
        rows = conn.execute(
            "SELECT * FROM form_assignments WHERE patient_id = ? ORDER BY created_at DESC",
            [patient_id],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM form_assignments ORDER BY created_at DESC"
        ).fetchall()

    messages = []
    for r in rows:
        d = dict(r)
        form_name = d.get("form_name") or d.get("instrument") or d.get("form_type") or "Questionnaire"
        status = d.get("status") or "assigned"
        assigned_by = d.get("assigned_by") or d.get("clinician") or "Care Team"

        messages.append({
            "id": f"fa-{d.get('id', 0)}",
            "category": CAT_FORM,
            "channel": "forms",
            "from_role": assigned_by,
            "subject": f"Form Assigned: {form_name}",
            "body": f"{form_name} — Status: {status}",
            "patient_id": d.get("patient_id"),
            "is_bot": False,
            "status": status,
            "timestamp": _ts(d.get("created_at")),
        })
    return messages


# ── Public API ───────────────────────────────────────────────────

def inbox_overview(patient_id=None):
    """Full inbox: all message categories combined, sorted by timestamp."""
    conn = _connect()
    if not conn:
        return {"messages": [], "summary": {}, "error": "clinical.db not found"}

    try:
        all_messages = []
        all_messages.extend(_team_messages(conn, patient_id))
        all_messages.extend(_clinical_decisions(conn, patient_id))
        all_messages.extend(_expert_reviews(conn, patient_id))
        all_messages.extend(_form_assignments(conn, patient_id))

        # Sort by timestamp descending
        all_messages.sort(key=lambda m: m.get("timestamp") or "", reverse=True)

        # Build summary
        by_category = defaultdict(int)
        by_channel = defaultdict(int)
        for m in all_messages:
            by_category[m["category"]] += 1
            by_channel[m.get("channel", "unknown")] += 1

        return {
            "messages": all_messages,
            "total": len(all_messages),
            "summary": {
                "by_category": [
                    {"category": cat, "label": CATEGORY_LABELS.get(cat, cat),
                     "icon": CATEGORY_ICONS.get(cat, ""), "count": cnt}
                    for cat, cnt in sorted(by_category.items())
                ],
                "by_channel": [
                    {"channel": ch, "count": cnt}
                    for ch, cnt in sorted(by_channel.items())
                ],
            },
            "source": "clinical.db (team_messages + clinical_decisions + expert_reviews + form_assignments)",
        }
    finally:
        conn.close()


def inbox_by_category(category, patient_id=None):
    """Messages filtered by category."""
    conn = _connect()
    if not conn:
        return {"messages": [], "error": "clinical.db not found"}

    try:
        fetchers = {
            CAT_TEAM: _team_messages,
            CAT_DECISION: _clinical_decisions,
            CAT_REVIEW: _expert_reviews,
            CAT_FORM: _form_assignments,
        }
        fetcher = fetchers.get(category)
        if not fetcher:
            return {"messages": [], "error": f"Unknown category: {category}"}

        messages = fetcher(conn, patient_id)
        messages.sort(key=lambda m: m.get("timestamp") or "", reverse=True)

        return {
            "category": category,
            "label": CATEGORY_LABELS.get(category, category),
            "messages": messages,
            "total": len(messages),
        }
    finally:
        conn.close()


def inbox_summary(patient_id=None):
    """Compact inbox summary: counts per category + latest 5 messages."""
    overview = inbox_overview(patient_id)
    return {
        "total": overview.get("total", 0),
        "by_category": overview.get("summary", {}).get("by_category", []),
        "by_channel": overview.get("summary", {}).get("by_channel", []),
        "latest": overview.get("messages", [])[:5],
        "source": overview.get("source", ""),
    }


def inbox_definitions():
    """Inbox metric definitions for tooltip overlays."""
    return {
        "categories": [
            {"id": cat, "label": label, "icon": CATEGORY_ICONS.get(cat, ""),
             "description": desc}
            for cat, label, desc in [
                (CAT_TEAM, "Care Team", "Messages between care team members and AI bot"),
                (CAT_DECISION, "Clinical Decision", "Clinical decisions recorded by treating clinicians"),
                (CAT_REVIEW, "Expert Review", "Specialist reviews of patient cases or AI predictions"),
                (CAT_FORM, "Form / Questionnaire", "Forms and questionnaires assigned to patients"),
                (CAT_SYSTEM, "System Notification", "System-generated notifications and alerts"),
            ]
        ],
        "source": "clinical.db tables: team_messages, clinical_decisions, expert_reviews, form_assignments",
    }
