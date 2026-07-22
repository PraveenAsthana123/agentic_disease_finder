"""IS SOP Compliance Dashboard — procedures + audits from clinical.db."""
import sqlite3
import json
import os

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB)


def _parse_rows(table):
    conn = _conn()
    rows = conn.execute(
        f"SELECT id, patient_id, fields_json, created_at FROM {table} ORDER BY id"
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = json.loads(r[2]) if r[2] else {}
        d["id"] = r[0]
        d["patient_id"] = r[1]
        d["created_at"] = r[3]
        out.append(d)
    return out


def overview():
    procedures = _parse_rows("is_sop_procedures")
    audits = _parse_rows("is_sop_audits")

    # Procedure KPIs
    total_sops = len(procedures)
    status_dist = {}
    category_dist = {}
    scores = []
    overdue_reviews = 0
    for p in procedures:
        s = p.get("status", "unknown")
        status_dist[s] = status_dist.get(s, 0) + 1
        c = p.get("category", "unknown")
        category_dist[c] = category_dist.get(c, 0) + 1
        score = p.get("compliance_score")
        if score is not None:
            scores.append(score)
        nrd = p.get("next_review_due", "")
        if nrd and nrd < "2026-07-22":
            overdue_reviews += 1

    avg_compliance = round(sum(scores) / len(scores), 1) if scores else 0

    # Audit KPIs
    total_audits = len(audits)
    finding_dist = {}
    severity_dist = {}
    audit_status_dist = {}
    open_findings = 0
    for a in audits:
        ft = a.get("finding_type", "unknown")
        finding_dist[ft] = finding_dist.get(ft, 0) + 1
        sv = a.get("severity", "unknown")
        severity_dist[sv] = severity_dist.get(sv, 0) + 1
        st = a.get("status", "unknown")
        audit_status_dist[st] = audit_status_dist.get(st, 0) + 1
        if st in ("open", "in_progress"):
            open_findings += 1

    compliant_count = finding_dist.get("compliant", 0)
    compliance_rate = round(100 * compliant_count / total_audits, 1) if total_audits else 0

    # Monthly audit trend
    monthly = {}
    for a in audits:
        dt = a.get("audit_date", "")[:7]
        if dt:
            monthly.setdefault(dt, {"month": dt, "audits": 0, "findings": 0})
            monthly[dt]["audits"] += 1
            if a.get("finding_type") != "compliant":
                monthly[dt]["findings"] += 1
    monthly_trend = sorted(monthly.values(), key=lambda x: x["month"])

    # Standards coverage from procedures
    standards_set = {}
    for p in procedures:
        for std in p.get("applicable_standards", []):
            standards_set[std] = standards_set.get(std, 0) + 1
    standards_coverage = [{"standard": k, "count": v} for k, v in sorted(standards_set.items(), key=lambda x: -x[1])]

    return {
        "total_sops": total_sops,
        "total_audits": total_audits,
        "avg_compliance_score": avg_compliance,
        "compliance_rate": compliance_rate,
        "open_findings": open_findings,
        "overdue_reviews": overdue_reviews,
        "sop_status_distribution": [{"status": k, "count": v} for k, v in status_dist.items()],
        "category_distribution": [{"category": k, "count": v} for k, v in category_dist.items()],
        "finding_distribution": [{"type": k, "count": v} for k, v in finding_dist.items()],
        "severity_distribution": [{"severity": k, "count": v} for k, v in severity_dist.items()],
        "audit_status_distribution": [{"status": k, "count": v} for k, v in audit_status_dist.items()],
        "monthly_trend": monthly_trend,
        "standards_coverage": standards_coverage,
    }


def breakdown():
    procedures = _parse_rows("is_sop_procedures")
    audits = _parse_rows("is_sop_audits")

    # Patient summary: how many SOPs + audits per patient
    patient_map = {}
    for p in procedures:
        pid = p.get("patient_id", "?")
        patient_map.setdefault(pid, {"patient_id": pid, "sop_count": 0, "audit_count": 0, "open_findings": 0, "avg_compliance": []})
        patient_map[pid]["sop_count"] += 1
        sc = p.get("compliance_score")
        if sc is not None:
            patient_map[pid]["avg_compliance"].append(sc)
    for a in audits:
        pid = a.get("patient_id", "?")
        patient_map.setdefault(pid, {"patient_id": pid, "sop_count": 0, "audit_count": 0, "open_findings": 0, "avg_compliance": []})
        patient_map[pid]["audit_count"] += 1
        if a.get("status") in ("open", "in_progress"):
            patient_map[pid]["open_findings"] += 1

    patients = []
    for v in patient_map.values():
        sc = v.pop("avg_compliance")
        v["avg_compliance_score"] = round(sum(sc) / len(sc), 1) if sc else None
        patients.append(v)
    patients.sort(key=lambda x: x["patient_id"])

    return {
        "procedures": procedures,
        "audits": audits,
        "patients": patients,
    }


def definitions():
    return {
        "concepts": [
            {"term": "SOP", "definition": "Standard Operating Procedure — documented process for consistent, compliant clinical and IT operations."},
            {"term": "Compliance Score", "definition": "0–100 rating of how well an SOP meets applicable regulatory standards."},
            {"term": "Finding Type", "definition": "Audit outcome: compliant, observation, minor_nonconformance, or major_nonconformance."},
            {"term": "Corrective Action", "definition": "Remediation plan required for non-compliant audit findings."},
            {"term": "Overdue Review", "definition": "SOP whose next_review_due date has passed without a recorded review."},
        ],
        "severity_levels": [
            {"level": "low", "description": "Minor documentation gap, no patient safety impact."},
            {"level": "medium", "description": "Process deviation with limited risk; corrective action within 30 days."},
            {"level": "high", "description": "Significant non-compliance; corrective action within 14 days."},
            {"level": "critical", "description": "Immediate patient safety or regulatory risk; corrective action required within 48 hours."},
        ],
        "sop_categories": [
            {"category": "Data Protection", "description": "EEG data handling, storage, encryption, backup procedures."},
            {"category": "Access Control", "description": "User authentication, MFA, role-based access policies."},
            {"category": "Incident Management", "description": "Incident response, escalation, root cause analysis procedures."},
            {"category": "Compliance", "description": "Regulatory compliance tracking, audit preparation, reporting."},
            {"category": "Device Security", "description": "IoT/medical device security, firmware updates, network segmentation."},
            {"category": "Training", "description": "Staff training requirements, competency assessment, certification tracking."},
        ],
        "data_sources": [
            {"table": "is_sop_procedures", "description": "20 SOPs across 6 categories with compliance scores and review schedules."},
            {"table": "is_sop_audits", "description": "30 audit findings with severity, corrective actions, and resolution status."},
        ],
    }
