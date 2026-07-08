"""
Groups / Teams Dashboard
=========================
Provides group-membership and group-permission metrics for the
neuro-AI clinical platform: clinical teams, admin groups,
cross-functional committees.

Pulls real patient/staff data from clinical.db via user_management_dashboard
and organises them into deterministic groups with role-appropriate permissions.

Data Sources:
  - patients              (40 rows)  — registered patient records
  - patient_demographics  (30 rows)  — extended demographics
  - user_management_dashboard._build_user_list() — combined user roster

Author: Research Team
"""

import sqlite3
import json
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# Import user list builder from sibling module
try:
    from scripts.user_management_dashboard import _build_user_list
except ImportError:
    from user_management_dashboard import _build_user_list


def _conn():
    return sqlite3.connect(DB_PATH)


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _fmt(val, decimals=1):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ── Deterministic synthetic helpers ────────────────────────────────

def _seed_rng(seed_str):
    """Deterministic RNG so results are stable across calls."""
    return random.Random(hashlib.md5(seed_str.encode()).hexdigest())


# ── Group definitions ─────────────────────────────────────────────

GROUP_DEFS = [
    # Clinical groups
    {
        "group_id": "GRP-C001",
        "name": "Neurology Team",
        "type": "clinical",
        "description": "Core neurology clinicians responsible for patient diagnosis and treatment planning.",
        "role_filter": ["Clinician"],
        "dept_filter": ["Neurology"],
        "permissions": [
            "view_patient_data", "edit_patient_data", "run_analysis",
            "approve_reports", "order_eeg", "prescribe_medication",
            "view_audit_log",
        ],
    },
    {
        "group_id": "GRP-C002",
        "name": "Epilepsy Monitoring Unit",
        "type": "clinical",
        "description": "Dedicated team for continuous EEG monitoring and seizure detection in admitted patients.",
        "role_filter": ["Clinician", "Technician"],
        "dept_filter": ["Neurology", "EEG Lab"],
        "permissions": [
            "view_patient_data", "run_analysis", "view_eeg_streams",
            "annotate_eeg", "trigger_seizure_alert", "export_data",
        ],
    },
    {
        "group_id": "GRP-C003",
        "name": "EEG Lab Technicians",
        "type": "clinical",
        "description": "Technical staff operating EEG equipment, maintaining electrodes, and performing routine recordings.",
        "role_filter": ["Technician"],
        "dept_filter": ["EEG Lab", "Imaging"],
        "permissions": [
            "view_patient_data", "run_analysis", "operate_eeg_equipment",
            "calibrate_devices", "upload_recordings",
        ],
    },
    {
        "group_id": "GRP-C004",
        "name": "Psychiatry Consult",
        "type": "clinical",
        "description": "Psychiatry liaison team providing consultations for neuropsychiatric comorbidities.",
        "role_filter": ["Clinician"],
        "dept_filter": ["Psychiatry"],
        "permissions": [
            "view_patient_data", "edit_patient_data", "run_analysis",
            "approve_reports", "view_psych_assessments",
        ],
    },
    {
        "group_id": "GRP-C005",
        "name": "Neuro-Research Group",
        "type": "clinical",
        "description": "Research team conducting clinical studies on neurological disorders using platform analytics.",
        "role_filter": ["Researcher"],
        "dept_filter": ["Neuro Research", "Data Science"],
        "permissions": [
            "view_patient_data", "run_analysis", "export_data",
            "access_anonymised_datasets", "submit_publications",
            "manage_study_protocols",
        ],
    },
    # Admin groups
    {
        "group_id": "GRP-A001",
        "name": "Platform Admins",
        "type": "admin",
        "description": "System administrators with full platform control including user management and configuration.",
        "role_filter": ["Admin"],
        "dept_filter": ["IT"],
        "permissions": [
            "manage_users", "admin_settings", "view_audit_log",
            "manage_roles", "system_config", "deploy_updates",
            "manage_integrations",
        ],
    },
    {
        "group_id": "GRP-A002",
        "name": "Data Governance Committee",
        "type": "admin",
        "description": "Oversight committee ensuring data quality, privacy compliance, and ethical data usage.",
        "role_filter": ["Admin", "Clinician"],
        "dept_filter": ["Compliance", "Neurology"],
        "permissions": [
            "view_audit_log", "manage_data_policies", "review_access_requests",
            "audit_data_usage", "manage_consent_records",
        ],
    },
    {
        "group_id": "GRP-A003",
        "name": "IT Operations",
        "type": "admin",
        "description": "Infrastructure and operations team managing servers, backups, and platform uptime.",
        "role_filter": ["Admin"],
        "dept_filter": ["IT", "Hospital Admin"],
        "permissions": [
            "admin_settings", "system_config", "manage_backups",
            "monitor_uptime", "manage_integrations", "view_audit_log",
        ],
    },
    # Cross-functional groups
    {
        "group_id": "GRP-X001",
        "name": "Quality Improvement",
        "type": "cross-functional",
        "description": "Multi-disciplinary team driving continuous quality improvement across clinical and technical workflows.",
        "role_filter": ["Clinician", "Admin", "Researcher"],
        "dept_filter": None,  # any department
        "permissions": [
            "view_patient_data", "run_analysis", "view_audit_log",
            "submit_qi_reports", "track_outcomes",
        ],
    },
    {
        "group_id": "GRP-X002",
        "name": "Patient Safety Board",
        "type": "cross-functional",
        "description": "Cross-functional board reviewing adverse events, near-misses, and safety protocols.",
        "role_filter": ["Clinician", "Admin", "Technician"],
        "dept_filter": None,  # any department
        "permissions": [
            "view_patient_data", "view_audit_log", "review_incidents",
            "submit_safety_reports", "manage_safety_protocols",
        ],
    },
]

# All unique permissions across all groups (for the permission matrix)
ALL_PERMISSIONS = sorted({p for g in GROUP_DEFS for p in g["permissions"]})


def _build_groups():
    """Build full group objects with members assigned deterministically."""
    rng = _seed_rng("groups-teams-v1")
    users = _build_user_list()
    now = datetime.now()

    groups = []
    for gdef in GROUP_DEFS:
        # Filter eligible users by role (and optionally department)
        eligible = [
            u for u in users
            if u["role"] in gdef["role_filter"]
            and (gdef["dept_filter"] is None or u["department"] in gdef["dept_filter"])
        ]

        # If too few eligible, relax department filter
        if len(eligible) < 2:
            eligible = [u for u in users if u["role"] in gdef["role_filter"]]

        # Deterministically pick members (at least 2, up to all eligible)
        if len(eligible) <= 3:
            members = list(eligible)
        else:
            count = rng.randint(max(2, len(eligible) // 2), len(eligible))
            members = rng.sample(eligible, count)

        # Pick a lead from members
        lead = rng.choice(members) if members else None

        # Deterministic created date (90-730 days ago)
        created_date = (now - timedelta(days=rng.randint(90, 730))).strftime("%Y-%m-%d")

        # Status: most groups active, occasionally archived
        status = rng.choices(["Active", "Archived"], weights=[0.9, 0.1])[0]

        member_list = [
            {"user_id": m["user_id"], "name": m["name"], "role": m["role"]}
            for m in members
        ]

        groups.append({
            "group_id": gdef["group_id"],
            "name": gdef["name"],
            "type": gdef["type"],
            "description": gdef["description"],
            "members": member_list,
            "permissions": gdef["permissions"],
            "created": created_date,
            "status": status,
            "lead": {"user_id": lead["user_id"], "name": lead["name"]} if lead else None,
            "member_count": len(member_list),
        })

    return groups


def _membership_trend(groups):
    """Generate 30-day membership trend (cumulative memberships)."""
    rng = _seed_rng("membership-trend-v1")
    now = datetime.now()
    total_memberships = sum(g["member_count"] for g in groups)
    trend = []
    for i in range(30):
        day = (now - timedelta(days=29 - i)).strftime("%Y-%m-%d")
        delta = rng.randint(-2, 3)
        memberships = max(total_memberships - 10 + i + delta, 1)
        trend.append({"date": day, "memberships": memberships})
    return trend


# ── Public API ─────────────────────────────────────────────────────

def overview():
    """KPIs, group distribution, membership trends, permission matrix."""
    groups = _build_groups()
    total = len(groups)
    active = sum(1 for g in groups if g["status"] == "Active")
    archived = sum(1 for g in groups if g["status"] == "Archived")
    total_memberships = sum(g["member_count"] for g in groups)
    avg_members = round(total_memberships / max(total, 1), 1)
    cross_functional = sum(1 for g in groups if g["type"] == "cross-functional")

    type_counts = Counter(g["type"] for g in groups)
    groups_by_type = [{"name": t, "value": c} for t, c in type_counts.most_common()]

    groups_by_size = [{"name": g["name"], "members": g["member_count"]} for g in groups]
    groups_by_size.sort(key=lambda x: x["members"], reverse=True)

    membership_trend = _membership_trend(groups)

    # Permission matrix: group × permission booleans
    permission_matrix = []
    for g in groups:
        row = {"group": g["name"]}
        for perm in ALL_PERMISSIONS:
            row[perm] = perm in g["permissions"]
        permission_matrix.append(row)

    return {
        "kpis": {
            "total_groups": total,
            "active_groups": active,
            "archived_groups": archived,
            "total_memberships": total_memberships,
            "avg_members_per_group": avg_members,
            "cross_functional_count": cross_functional,
        },
        "groups_by_type": groups_by_type,
        "groups_by_size": groups_by_size,
        "membership_trend": membership_trend,
        "permission_matrix": permission_matrix,
    }


def breakdown():
    """Full group list with members and permissions, plus type summary."""
    groups = _build_groups()

    type_summary = {}
    for g in groups:
        t = g["type"]
        if t not in type_summary:
            type_summary[t] = {"total": 0, "active": 0, "avg_members": 0, "_member_sum": 0}
        type_summary[t]["total"] += 1
        type_summary[t]["_member_sum"] += g["member_count"]
        if g["status"] == "Active":
            type_summary[t]["active"] += 1

    for t, info in type_summary.items():
        info["avg_members"] = round(info["_member_sum"] / max(info["total"], 1), 1)
        del info["_member_sum"]

    return {
        "groups": groups,
        "type_summary": type_summary,
        "total": len(groups),
    }


def definitions():
    """Groups & teams glossary — group types, membership, permissions."""
    return {
        "terms": [
            {"term": "Group", "definition": "A named collection of platform users who share a common function, department, or responsibility. Groups are the primary unit for assigning collective permissions."},
            {"term": "Team", "definition": "A working group with an assigned lead, active project scope, and shared clinical or operational objectives. All teams are groups but not all groups are teams."},
            {"term": "Membership", "definition": "The association of a user account with one or more groups. A single user may belong to multiple groups simultaneously."},
            {"term": "Group Permission", "definition": "An access right granted to all members of a group. Group permissions are additive — a user's effective permissions are the union of all their group permissions."},
            {"term": "Clinical Group", "definition": "A group composed of clinical staff (clinicians, technicians) focused on patient care activities such as diagnosis, monitoring, and treatment."},
            {"term": "Admin Group", "definition": "A group of administrative or IT staff responsible for platform configuration, user management, compliance, and infrastructure operations."},
            {"term": "Cross-Functional Group", "definition": "A group spanning multiple departments and roles, typically formed for quality improvement, safety oversight, or strategic initiatives."},
            {"term": "Group Lead", "definition": "The designated point of contact and decision-maker for a group. The lead manages membership requests, reviews permissions, and represents the group in governance meetings."},
            {"term": "Group Status", "definition": "The lifecycle state of a group: Active (operational with current members) or Archived (retained for audit trail but no longer granting permissions)."},
            {"term": "Inherited Permission", "definition": "A permission a user receives automatically by virtue of group membership, as opposed to individually assigned permissions. Revoking group membership removes all inherited permissions."},
            {"term": "Permission Matrix", "definition": "A cross-tabulation of groups against available permissions, showing which capabilities each group grants to its members."},
            {"term": "Role Filter", "definition": "The set of user roles eligible for membership in a given group. Used during group population to ensure appropriate skill-mix."},
            {"term": "Member Count", "definition": "The current number of active user accounts associated with a group. Used for capacity planning and workload distribution."},
            {"term": "Data Governance", "definition": "Policies and processes ensuring data quality, privacy compliance (HIPAA/GDPR), and ethical usage of patient information across the platform."},
        ]
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(overview(), indent=2, default=str))
