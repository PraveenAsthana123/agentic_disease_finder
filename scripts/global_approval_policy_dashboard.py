"""Global Approval Policy Dashboard — governance approval rules, roles,
risk bands, and audit requirements from config/global_approval_policy.json.
5 approval roles, 8 policy rules, 4 risk bands, HITL workflow."""

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
    """Summary KPIs: rules, roles, risk bands, decision distribution."""
    cfg = _load('global_approval_policy.json')
    if not cfg:
        return {"available": False, "note": "global_approval_policy.json missing"}

    rules = cfg.get('rules', [])
    approval_roles = cfg.get('approval_roles', {})
    risk_bands = cfg.get('risk_bands', {})
    decisions = cfg.get('decisions', {})
    applies_to = cfg.get('applies_to', [])
    audit_req = cfg.get('audit_requirements', {})

    decision_counts = Counter(r.get('decision', 'unknown') for r in rules)
    decision_distribution = [
        {"name": d, "value": c}
        for d, c in sorted(decision_counts.items(), key=lambda x: -x[1])
    ]

    total_approvable_scopes = sum(
        len(v.get('can_approve', []))
        for v in approval_roles.values()
    )

    rules_requiring_role = sum(
        1 for r in rules if r.get('required_role')
    )

    rules_table = [
        {
            "rule_id": r.get('rule_id', ''),
            "name": r.get('name', ''),
            "decision": r.get('decision', ''),
            "required_role": r.get('required_role', ''),
            "required_audit": r.get('required_audit', False),
        }
        for r in rules
    ]

    return {
        "available": True,
        "policy_id": cfg.get('policy_id', ''),
        "policy_name": cfg.get('name', ''),
        "version": cfg.get('version', ''),
        "status": cfg.get('status', ''),
        "classification": cfg.get('classification', ''),
        "default_decision": cfg.get('default_decision', ''),
        "summary": {
            "total_rules": len(rules),
            "total_roles": len(approval_roles),
            "total_risk_bands": len(risk_bands),
            "total_decision_types": len(decisions),
            "applies_to_domains": len(applies_to),
            "total_approvable_scopes": total_approvable_scopes,
            "rules_requiring_role": rules_requiring_role,
            "audit_required_fields": len(audit_req.get('required_fields', [])),
            "audit_approval_fields": len(audit_req.get('approval_fields', [])),
            "retention_days": audit_req.get('retention_days', 0),
        },
        "decision_distribution": decision_distribution,
        "rules_table": rules_table,
        "applies_to": applies_to,
    }


def breakdown():
    """Per-role approval scopes, per-rule match criteria, risk band details."""
    cfg = _load('global_approval_policy.json')
    if not cfg:
        return {"available": False, "note": "global_approval_policy.json missing"}

    approval_roles = cfg.get('approval_roles', {})
    rules = cfg.get('rules', [])
    risk_bands = cfg.get('risk_bands', {})
    audit_req = cfg.get('audit_requirements', {})
    hitl = cfg.get('human_approval', {})
    change_ctrl = cfg.get('change_control', {})

    per_role = []
    for role_id, role_data in approval_roles.items():
        scopes = role_data.get('can_approve', [])
        per_role.append({
            "role": role_id,
            "can_approve": scopes,
            "scope_count": len(scopes),
        })

    per_rule = []
    for r in rules:
        match = r.get('match', {})
        per_rule.append({
            "rule_id": r.get('rule_id', ''),
            "name": r.get('name', ''),
            "decision": r.get('decision', ''),
            "required_role": r.get('required_role', ''),
            "required_audit": r.get('required_audit', False),
            "match_criteria": match,
        })

    per_risk_band = []
    for band_name, band_data in risk_bands.items():
        per_risk_band.append({
            "band": band_name,
            "default_decision": band_data.get('default_decision', ''),
            "examples": band_data.get('examples', []),
        })

    return {
        "available": True,
        "per_role": per_role,
        "per_rule": per_rule,
        "per_risk_band": per_risk_band,
        "audit_required_fields": audit_req.get('required_fields', []),
        "audit_approval_fields": audit_req.get('approval_fields', []),
        "retention_days": audit_req.get('retention_days', 0),
        "hitl": {
            "queue": hitl.get('queue', ''),
            "request_endpoint": hitl.get('request_endpoint', ''),
            "decision_endpoint": hitl.get('decision_endpoint', ''),
            "default_sla_minutes": hitl.get('default_sla_minutes', 0),
            "critical_sla_minutes": hitl.get('critical_sla_minutes', 0),
            "allowed_decisions": hitl.get('allowed_decisions', []),
        },
        "change_control": change_ctrl,
    }


def definitions():
    """Decision types, role descriptions, risk band legend, glossary, references."""
    return {
        "available": True,
        "decision_types": [
            {"decision": "allow", "description": "Action may proceed with audit logging — no human gate."},
            {"decision": "require_human_approval", "description": "Action queued for reviewer approval before execution — HITL gate."},
            {"decision": "deny", "description": "Action must not proceed — hard block, no override without governance_lead."},
        ],
        "role_descriptions": [
            {"role": "clinical_reviewer", "description": "Reviews clinical decision support outputs, patient reports, and high-risk predictions before release."},
            {"role": "data_steward", "description": "Approves dataset ingestion, data exports, retention changes, and de-identification exceptions."},
            {"role": "model_owner", "description": "Approves model training, promotion to production, threshold changes, and performance claim updates."},
            {"role": "security_officer", "description": "Approves external integrations, credential rotations, privileged tool use, and production access."},
            {"role": "governance_lead", "description": "Approves policy changes, overrides, kill-switch disables, and regulatory exceptions. Top-level authority."},
        ],
        "risk_band_legend": [
            {"band": "low", "color": "#22c55e", "description": "Read-only, non-sensitive — auto-allowed with audit logging."},
            {"band": "medium", "color": "#f97316", "description": "Local writes, non-production training — requires human approval."},
            {"band": "high", "color": "#ef4444", "description": "Patient-facing, external sharing, model deployment — requires human approval."},
            {"band": "critical", "color": "#7f1d1d", "description": "Destructive production actions — denied by default, governance_lead override only."},
        ],
        "glossary": [
            {"term": "HITL", "definition": "Human-In-The-Loop — requires human review/approval before AI action proceeds."},
            {"term": "GAP", "definition": "Global Approval Policy — the master ruleset governing all agentic actions."},
            {"term": "SLA", "definition": "Service Level Agreement — max time (minutes) for a reviewer to respond to an approval request."},
            {"term": "PHI", "definition": "Protected Health Information — HIPAA-regulated patient data requiring special handling."},
            {"term": "PII", "definition": "Personally Identifiable Information — data that can identify an individual."},
            {"term": "RBAC", "definition": "Role-Based Access Control — permissions assigned by organizational role."},
            {"term": "MCP", "definition": "Model Context Protocol — standardized tool/integration protocol for AI agents."},
            {"term": "Kill Switch", "definition": "Emergency mechanism to immediately halt all AI agent actions."},
            {"term": "Scope Prefix", "definition": "Hierarchical permission namespace (e.g., public:read, admin:write) for rule matching."},
            {"term": "Risk Band", "definition": "Classification tier (low/medium/high/critical) determining default approval behavior."},
        ],
        "clinical_notes": [
            "All clinical decision support outputs require clinical_reviewer approval before release to patients.",
            "PHI/PII exports require data_steward approval — no automated data egress.",
            "Model deployment requires model_owner sign-off — prevents untested models reaching production.",
            "Policy changes require governance_lead approval — prevents unauthorized rule relaxation.",
        ],
        "references": [
            "FDA AI/ML Software as a Medical Device — Predetermined Change Control Plan",
            "EU AI Act (2024) — High-Risk AI System Requirements",
            "ISO 14971:2019 — Medical Devices Risk Management",
            "NIST AI Risk Management Framework (AI RMF 1.0)",
            "HIPAA Security Rule — 45 CFR Part 164",
        ],
    }
