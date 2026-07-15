"""SOP Compliance Dashboard — Information Security SOP and audit analytics.

Standard Operating Procedures (SOPs) are foundational documents in clinical informatics
that codify repeatable processes for data protection, incident management, access control,
regulatory compliance, AI governance, and clinical operations.  Periodic audits measure
adherence to these procedures against industry standards (HIPAA, GDPR, SOC 2 Type II,
IEC 62443, NIST CSF, ISO 27001, HITRUST).  Tracking SOP compliance scores, review
cadences, audit findings, and corrective actions enables organisations to maintain
continuous compliance posture and satisfy external auditor requirements.

All data from REAL is_sop_procedures and is_sop_audits rows in data/clinical.db.
"""
import sqlite3
import json
from datetime import date
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _parse_procedures():
    """Fetch all is_sop_procedures rows and parse fields_json."""
    raw = _rows("SELECT id, patient_id, fields_json, created_at FROM is_sop_procedures ORDER BY id")
    parsed = []
    today = date.today().isoformat()
    for r in raw:
        fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
        next_review = fields.get("next_review_due", "")
        is_overdue = bool(next_review and next_review < today)
        parsed.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "sop_id": fields.get("sop_id", ""),
            "title": fields.get("title", ""),
            "version": fields.get("version", ""),
            "status": fields.get("status", "draft"),
            "category": fields.get("category", "Unknown"),
            "owner": fields.get("owner", "Unknown"),
            "last_reviewed": fields.get("last_reviewed", ""),
            "next_review_due": next_review,
            "compliance_score": fields.get("compliance_score", 0),
            "applicable_standards": fields.get("applicable_standards", []),
            "approver": fields.get("approver", ""),
            "revision_count": fields.get("revision_count", 0),
            "is_overdue": is_overdue,
            "created_at": r["created_at"],
        })
    return parsed


def _parse_audits():
    """Fetch all is_sop_audits rows and parse fields_json."""
    raw = _rows("SELECT id, patient_id, fields_json, created_at FROM is_sop_audits ORDER BY id")
    parsed = []
    for r in raw:
        fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
        parsed.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "audit_id": fields.get("audit_id", ""),
            "sop_id": fields.get("sop_id", ""),
            "audit_date": fields.get("audit_date", ""),
            "auditor": fields.get("auditor", ""),
            "finding_type": fields.get("finding_type", ""),
            "finding_description": fields.get("finding_description", ""),
            "corrective_action": fields.get("corrective_action"),
            "status": fields.get("status", "open"),
            "severity": fields.get("severity", "low"),
            "created_at": r["created_at"],
        })
    return parsed


# ── Public API ─────────────────────────────────────────────────────────

def overview():
    """Aggregate SOP compliance statistics.

    Returns dict with total_procedures, total_audits, avg_compliance_score,
    status_distribution, category_breakdown, overdue_reviews, standards_coverage,
    finding_type_distribution, severity_distribution, and open_findings.
    """
    procedures = _parse_procedures()
    audits = _parse_audits()

    if not procedures:
        return {"total_procedures": 0, "total_audits": 0, "message": "No SOP data yet"}

    total_procedures = len(procedures)
    total_audits = len(audits)

    # Average compliance score
    scores = [p["compliance_score"] for p in procedures if p["compliance_score"] is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    # Status distribution
    status_dist = {"published": 0, "under_review": 0, "draft": 0, "retired": 0}
    for p in procedures:
        s = p["status"]
        if s in status_dist:
            status_dist[s] += 1

    # Category breakdown
    cat_map = {}
    for p in procedures:
        cat = p["category"]
        if cat not in cat_map:
            cat_map[cat] = {"category": cat, "count": 0, "total_score": 0}
        cat_map[cat]["count"] += 1
        cat_map[cat]["total_score"] += p["compliance_score"] or 0
    category_breakdown = []
    for v in cat_map.values():
        category_breakdown.append({
            "category": v["category"],
            "count": v["count"],
            "avg_score": round(v["total_score"] / v["count"], 1) if v["count"] else 0,
        })
    category_breakdown.sort(key=lambda x: x["count"], reverse=True)

    # Overdue reviews
    overdue_reviews = sum(1 for p in procedures if p["is_overdue"])

    # Standards coverage
    std_counter = {}
    for p in procedures:
        for st in p["applicable_standards"]:
            std_counter[st] = std_counter.get(st, 0) + 1
    standards_coverage = [{"standard": k, "sop_count": v} for k, v in std_counter.items()]
    standards_coverage.sort(key=lambda x: x["sop_count"], reverse=True)

    # Finding type distribution
    finding_dist = {
        "compliant": 0,
        "observation": 0,
        "minor_nonconformance": 0,
        "major_nonconformance": 0,
    }
    for a in audits:
        ft = a["finding_type"]
        if ft in finding_dist:
            finding_dist[ft] += 1

    # Severity distribution
    severity_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for a in audits:
        sv = a["severity"]
        if sv in severity_dist:
            severity_dist[sv] += 1

    # Open findings
    open_findings = sum(1 for a in audits if a["status"] != "closed")

    return {
        "total_procedures": total_procedures,
        "total_audits": total_audits,
        "avg_compliance_score": avg_score,
        "status_distribution": status_dist,
        "category_breakdown": category_breakdown,
        "overdue_reviews": overdue_reviews,
        "standards_coverage": standards_coverage,
        "finding_type_distribution": finding_dist,
        "severity_distribution": severity_dist,
        "open_findings": open_findings,
    }


def breakdown():
    """Per-SOP and per-audit drill-down with SOP-audit join.

    Returns dict with procedures (list), audits (list), and sop_audit_map
    (SOPs joined to their related audits).
    """
    procedures = _parse_procedures()
    audits = _parse_audits()

    # Build procedure list (drop helper fields)
    proc_list = []
    for p in procedures:
        proc_list.append({
            "sop_id": p["sop_id"],
            "title": p["title"],
            "version": p["version"],
            "status": p["status"],
            "category": p["category"],
            "owner": p["owner"],
            "compliance_score": p["compliance_score"],
            "last_reviewed": p["last_reviewed"],
            "next_review_due": p["next_review_due"],
            "applicable_standards": p["applicable_standards"],
            "revision_count": p["revision_count"],
            "is_overdue": p["is_overdue"],
        })

    # Build audit list
    audit_list = []
    for a in audits:
        audit_list.append({
            "audit_id": a["audit_id"],
            "sop_id": a["sop_id"],
            "audit_date": a["audit_date"],
            "auditor": a["auditor"],
            "finding_type": a["finding_type"],
            "finding_description": a["finding_description"],
            "corrective_action": a["corrective_action"],
            "status": a["status"],
            "severity": a["severity"],
        })

    # SOP-audit join map
    sop_title_map = {p["sop_id"]: p["title"] for p in procedures}
    audit_by_sop = {}
    for a in audit_list:
        sid = a["sop_id"]
        if sid not in audit_by_sop:
            audit_by_sop[sid] = []
        audit_by_sop[sid].append(a)

    sop_audit_map = []
    # Include all SOPs, even those with no audits
    seen = set()
    for p in procedures:
        sid = p["sop_id"]
        if sid in seen:
            continue
        seen.add(sid)
        sop_audit_map.append({
            "sop_id": sid,
            "sop_title": sop_title_map.get(sid, ""),
            "audits": audit_by_sop.get(sid, []),
        })
    # Include audits referencing SOPs not in the procedures table
    for sid, alist in audit_by_sop.items():
        if sid not in seen:
            sop_audit_map.append({
                "sop_id": sid,
                "sop_title": sop_title_map.get(sid, "Unknown SOP"),
                "audits": alist,
            })

    return {
        "procedures": proc_list,
        "audits": audit_list,
        "sop_audit_map": sop_audit_map,
    }


def definitions():
    """Clinical informatics glossary for SOP compliance terms.

    Returns dict with term/definition pairs used by tooltip overlays.
    """
    return {
        "terms": [
            {
                "term": "SOP (Standard Operating Procedure)",
                "definition": (
                    "A documented, step-by-step set of instructions that describes how to "
                    "perform a routine activity consistently.  In clinical informatics, SOPs "
                    "govern data handling, system access, incident response, and AI model "
                    "deployment to ensure reproducibility and regulatory compliance."
                ),
            },
            {
                "term": "Compliance Score",
                "definition": (
                    "A quantitative measure (0\u2013100) reflecting the degree to which a "
                    "procedure meets its applicable regulatory and organisational standards.  "
                    "Scores are derived from audit findings, review cadence adherence, and "
                    "documentation completeness."
                ),
            },
            {
                "term": "Finding Type \u2014 Compliant",
                "definition": (
                    "An audit outcome indicating the SOP fully meets the requirements of the "
                    "applicable standard with no deviations observed."
                ),
            },
            {
                "term": "Finding Type \u2014 Observation",
                "definition": (
                    "A non-binding audit note highlighting a potential improvement opportunity "
                    "that does not constitute a formal nonconformance but may become one if "
                    "left unaddressed."
                ),
            },
            {
                "term": "Finding Type \u2014 Minor Nonconformance",
                "definition": (
                    "An audit finding where a single element of the SOP does not meet the "
                    "standard requirement, but the overall process intent is still achieved.  "
                    "Corrective action is expected within a defined timeframe."
                ),
            },
            {
                "term": "Finding Type \u2014 Major Nonconformance",
                "definition": (
                    "An audit finding where the SOP fails to meet a critical standard "
                    "requirement, posing a significant risk to data integrity, patient safety, "
                    "or regulatory standing.  Immediate corrective action and root-cause "
                    "analysis are required."
                ),
            },
            {
                "term": "Corrective Action",
                "definition": (
                    "A documented remediation step taken to address an audit finding.  "
                    "Corrective actions must include a description of the fix, responsible "
                    "party, target completion date, and evidence of effectiveness verification."
                ),
            },
            {
                "term": "HIPAA (Health Insurance Portability and Accountability Act)",
                "definition": (
                    "US federal law establishing national standards for protecting individuals\u2019 "
                    "electronic personal health information.  Requires administrative, physical, "
                    "and technical safeguards including access controls, audit logging, and "
                    "encryption."
                ),
            },
            {
                "term": "GDPR (General Data Protection Regulation)",
                "definition": (
                    "European Union regulation on data protection and privacy.  Mandates lawful "
                    "basis for processing, data subject rights (access, erasure, portability), "
                    "data protection impact assessments, and breach notification within 72 hours."
                ),
            },
            {
                "term": "SOC 2 Type II",
                "definition": (
                    "An auditing framework developed by the AICPA that evaluates an "
                    "organisation\u2019s controls over security, availability, processing integrity, "
                    "confidentiality, and privacy over a sustained period (typically 6\u201312 months)."
                ),
            },
            {
                "term": "IEC 62443",
                "definition": (
                    "International standard series for industrial automation and control system "
                    "(IACS) security.  Defines security levels, zones, conduits, and lifecycle "
                    "requirements applicable to networked medical devices and clinical "
                    "infrastructure."
                ),
            },
            {
                "term": "NIST CSF (Cybersecurity Framework)",
                "definition": (
                    "A voluntary framework by the US National Institute of Standards and "
                    "Technology organising cybersecurity activities into five functions: Identify, "
                    "Protect, Detect, Respond, and Recover.  Widely adopted in healthcare IT."
                ),
            },
            {
                "term": "ISO 27001",
                "definition": (
                    "International standard for information security management systems (ISMS).  "
                    "Specifies requirements for establishing, implementing, maintaining, and "
                    "continually improving an ISMS, including risk assessment and treatment."
                ),
            },
            {
                "term": "HITRUST CSF",
                "definition": (
                    "A certifiable security and privacy framework that harmonises requirements "
                    "from HIPAA, NIST, ISO 27001, PCI-DSS, and other standards into a single "
                    "assessment methodology commonly used in healthcare."
                ),
            },
            {
                "term": "Audit Cycle",
                "definition": (
                    "The recurring process of planning, conducting, reporting, and following up "
                    "on SOP compliance audits.  A typical cycle includes internal audits "
                    "(quarterly or semi-annually) and external certification audits (annually), "
                    "with surveillance audits in interim periods to verify corrective action "
                    "effectiveness."
                ),
            },
        ]
    }
