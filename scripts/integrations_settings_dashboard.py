"""Integrations & Delivery Channels Settings Dashboard
from config/integrations.json.
6 integrations (Google Drive, OneDrive, Slack, Google Chat, Gmail, WhatsApp),
6 delivery channels (voice AI, conversational AI, form AI, email form link,
survey link, email campaign)."""

import json
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "integrations.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        raw = json.load(f)
    return raw[0] if isinstance(raw, list) else raw


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: total integrations, delivery channels, status distribution."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "integrations.json missing"}

    integrations = cfg.get("integrations", [])
    channels = cfg.get("delivery_channels", [])
    summary = cfg.get("summary", {})

    total_int = len(integrations)
    total_ch = len(channels)
    total = total_int + total_ch

    # Status counts across both lists
    status_counts = {}
    for item in integrations + channels:
        s = item.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    built = status_counts.get("built", 0)
    partial = status_counts.get("partial", 0)
    needs_creds = status_counts.get("needs-credentials", 0)
    planned = status_counts.get("planned", 0)

    # Category distribution (integrations only)
    cat_counts = {}
    for item in integrations:
        c = item.get("category", "other")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    category_dist = [{"name": k.title(), "value": v} for k, v in cat_counts.items() if v > 0]

    # Status distribution pie
    status_dist = [{"name": k.replace("-", " ").title(), "value": v}
                   for k, v in status_counts.items() if v > 0]

    # Integration status bar chart
    int_status = [{"name": item.get("name", item.get("id", "")),
                   "status": item.get("status", "unknown")}
                  for item in integrations]

    # Channel status bar chart
    ch_status = [{"name": item.get("name", item.get("id", "")),
                  "status": item.get("status", "unknown")}
                 for item in channels]

    return {
        "available": True,
        "title": cfg.get("title", "Integrations & Delivery Channels"),
        "note": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "kpis": {
            "total": total,
            "integrations": total_int,
            "delivery_channels": total_ch,
            "built": built,
            "partial": partial,
            "needs_credentials": needs_creds,
            "planned": planned,
        },
        "charts": {
            "status_distribution": status_dist,
            "category_distribution": category_dist,
            "integration_status": int_status,
            "channel_status": ch_status,
        },
        "honest_note": summary.get("honest_note", ""),
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-item detail: integrations and delivery channels."""
    cfg = _load()
    if not cfg:
        return {"available": False}

    integrations = cfg.get("integrations", [])
    channels = cfg.get("delivery_channels", [])

    int_cards = []
    for item in integrations:
        int_cards.append({
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "category": item.get("category", ""),
            "purpose": item.get("purpose", ""),
            "status": item.get("status", "unknown"),
            "config": item.get("config", ""),
            "scope": item.get("scope", ""),
        })

    ch_cards = []
    for item in channels:
        ch_cards.append({
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "purpose": item.get("purpose", ""),
            "status": item.get("status", "unknown"),
            "config": item.get("config", ""),
            "note": item.get("note", ""),
        })

    return {
        "available": True,
        "integrations": int_cards,
        "delivery_channels": ch_cards,
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "label": "Built", "meaning": "Fully implemented and functional"},
            {"status": "partial", "label": "Partial", "meaning": "Core component exists, integration mapping in progress"},
            {"status": "needs-credentials", "label": "Needs Credentials", "meaning": "Adapter wired, awaiting real OAuth/API credentials to activate"},
            {"status": "planned", "label": "Planned", "meaning": "Architecture defined, implementation not started"},
        ],
        "glossary": [
            {"term": "Integration", "definition": "External service connection (storage, messaging, email) via OAuth or API credentials"},
            {"term": "Delivery Channel", "definition": "Method for delivering clinical assessments to patients (voice, chat, form, email, survey)"},
            {"term": "OAuth2", "definition": "Open standard for access delegation, used for Google Drive and OneDrive integration"},
            {"term": "MSAL", "definition": "Microsoft Authentication Library for Office365/OneDrive integration"},
            {"term": "STT", "definition": "Speech-to-Text conversion, used in Voice AI intake via Whisper"},
            {"term": "SMTP", "definition": "Simple Mail Transfer Protocol, used for Gmail/email integration"},
            {"term": "Adapter", "definition": "Code that translates between the platform and an external service API"},
            {"term": "Scope", "definition": "Permission level requested from an external service (e.g., drive.file, chat:write)"},
            {"term": "Conversational AI", "definition": "LLM-based chat intake that asks assessment items conversationally"},
            {"term": "Form AI", "definition": "Smart web form for structured clinical assessment data entry"},
            {"term": "Survey Link", "definition": "Tokenized public URL for patient self-service assessment completion"},
            {"term": "Meta Cloud API", "definition": "WhatsApp Business API provided by Meta for programmatic messaging"},
        ],
        "clinical_notes": [
            "All integrations must comply with HIPAA when handling PHI (Protected Health Information)",
            "Assessment delivery channels support 10 validated clinical instruments (MoCA, PHQ-9, GAD-7, etc.)",
            "Voice AI intake uses Whisper STT for offline, privacy-preserving speech recognition",
            "Email campaigns require explicit patient consent per HIPAA marketing rules",
        ],
        "references": [
            {"label": "integrations.json", "note": "Source config for all integrations and delivery channels"},
            {"label": "HIPAA", "note": "Health Insurance Portability and Accountability Act — privacy/security framework"},
            {"label": "OAuth 2.0 (RFC 6749)", "note": "Authorization framework for third-party access delegation"},
            {"label": "Whisper (OpenAI)", "note": "Open-source automatic speech recognition model used for Voice AI"},
        ],
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(overview())
