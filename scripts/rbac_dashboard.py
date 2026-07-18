"""RBAC Dashboard — role-based access control analytics from admin_users + transaction_log."""

import sqlite3
from collections import Counter
from datetime import datetime

DB = "data/clinical.db"


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def overview():
    conn = _conn()
    cur = conn.cursor()

    # Users
    users = [dict(r) for r in cur.execute("SELECT * FROM admin_users").fetchall()]
    total_users = len(users)
    active_users = sum(1 for u in users if u["status"] == "active")
    inactive_users = total_users - active_users
    mfa_enabled = sum(1 for u in users if u["mfa_enabled"])
    mfa_rate = round(mfa_enabled / total_users * 100, 1) if total_users else 0

    # Role distribution
    role_counts = Counter(u["role"] for u in users)
    role_distribution = [{"role": k, "count": v} for k, v in role_counts.most_common()]

    # Department distribution
    dept_counts = Counter(u["department"] for u in users)
    dept_distribution = [{"department": k, "count": v} for k, v in dept_counts.most_common()]

    # Status distribution
    status_distribution = [
        {"status": "Active", "count": active_users},
        {"status": "Inactive", "count": inactive_users},
    ]

    # Login activity (top users by login_count)
    login_leaders = sorted(users, key=lambda u: u["login_count"], reverse=True)[:10]
    login_data = [{"user": u["full_name"], "role": u["role"], "logins": u["login_count"]} for u in login_leaders]

    # Transaction log — actions per actor role
    txn_rows = cur.execute("SELECT actor, action, component FROM transaction_log").fetchall()
    txn_list = [dict(r) for r in txn_rows]
    actor_actions = Counter(r["actor"] for r in txn_list)
    action_by_actor = [{"actor": k, "actions": v} for k, v in actor_actions.most_common(10)]

    # Components accessed
    comp_counts = Counter(r["component"] for r in txn_list)
    component_access = [{"component": k, "accesses": v} for k, v in comp_counts.most_common(10)]

    # Roles vs departments matrix
    role_dept = {}
    for u in users:
        key = (u["role"], u["department"])
        role_dept[key] = role_dept.get(key, 0) + 1
    roles = sorted(set(u["role"] for u in users))
    depts = sorted(set(u["department"] for u in users))
    matrix = []
    for role in roles:
        row = {"role": role}
        for dept in depts:
            row[dept] = role_dept.get((role, dept), 0)
        matrix.append(row)

    conn.close()
    return {
        "kpis": {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": inactive_users,
            "roles": len(role_counts),
            "departments": len(dept_counts),
            "mfa_enabled": mfa_enabled,
            "mfa_rate": mfa_rate,
            "total_transactions": len(txn_list),
        },
        "role_distribution": role_distribution,
        "dept_distribution": dept_distribution,
        "status_distribution": status_distribution,
        "login_leaders": login_data,
        "action_by_actor": action_by_actor,
        "component_access": component_access,
        "role_dept_matrix": matrix,
        "departments": depts,
    }


def breakdown():
    conn = _conn()
    cur = conn.cursor()

    users = [dict(r) for r in cur.execute("SELECT * FROM admin_users ORDER BY role, full_name").fetchall()]

    # Per-role summary
    role_groups = {}
    for u in users:
        role_groups.setdefault(u["role"], []).append(u)

    role_summaries = []
    for role, members in sorted(role_groups.items()):
        active = sum(1 for m in members if m["status"] == "active")
        avg_logins = round(sum(m["login_count"] for m in members) / len(members), 1)
        mfa_count = sum(1 for m in members if m["mfa_enabled"])
        role_summaries.append({
            "role": role,
            "total": len(members),
            "active": active,
            "inactive": len(members) - active,
            "avg_logins": avg_logins,
            "mfa_enabled": mfa_count,
            "mfa_rate": round(mfa_count / len(members) * 100, 1),
            "departments": sorted(set(m["department"] for m in members)),
        })

    # User list
    user_list = [{
        "user_id": u["user_id"],
        "username": u["username"],
        "full_name": u["full_name"],
        "email": u["email"],
        "role": u["role"],
        "department": u["department"],
        "status": u["status"],
        "mfa_enabled": bool(u["mfa_enabled"]),
        "login_count": u["login_count"],
        "last_login": u["last_login"],
        "created_at": u["created_at"],
    } for u in users]

    # Inactive users (security concern)
    inactive_users = [u for u in user_list if u["status"] == "inactive"]

    # No MFA (security concern)
    no_mfa_users = [u for u in user_list if not u["mfa_enabled"]]

    # Recent transactions per actor
    txn_rows = cur.execute(
        "SELECT actor, action, component, ts_utc FROM transaction_log ORDER BY ts_utc DESC LIMIT 50"
    ).fetchall()
    recent_transactions = [dict(r) for r in txn_rows]

    conn.close()
    return {
        "role_summaries": role_summaries,
        "user_list": user_list,
        "inactive_users": inactive_users,
        "no_mfa_users": no_mfa_users,
        "recent_transactions": recent_transactions,
    }


def definitions():
    return {
        "roles": [
            {"role": "Neurologist", "description": "Consulting physician — full read/write access to patient records, EEG interpretations, treatment decisions."},
            {"role": "EEG Tech", "description": "EEG technician — records EEGs, manages signal quality, flags artifacts. Read access to patient demographics."},
            {"role": "Nurse", "description": "Clinical nurse — medication administration, seizure diary review, patient education. Read/write nursing notes."},
            {"role": "Researcher", "description": "Research staff — de-identified data access, cohort analytics, model training. No direct patient care access."},
            {"role": "Admin", "description": "System administrator — user management, configuration, audit logs, system health. Full platform access."},
            {"role": "Data Scientist", "description": "AI/ML engineer — model development, validation metrics, pipeline management. De-identified data access."},
        ],
        "permissions": [
            {"level": "Read", "description": "View records without modification capability."},
            {"level": "Write", "description": "Create and modify records within role scope."},
            {"level": "Delete", "description": "Remove records (admin only, audit-logged)."},
            {"level": "Export", "description": "Download/export data (requires de-identification for research roles)."},
            {"level": "Configure", "description": "Modify system settings, thresholds, pipeline parameters."},
            {"level": "Audit", "description": "View audit trails and access logs."},
        ],
        "security_policies": [
            {"policy": "MFA Required", "description": "Multi-factor authentication mandatory for all clinical access roles."},
            {"policy": "Session Timeout", "description": "Automatic logout after 30 minutes of inactivity."},
            {"policy": "Least Privilege", "description": "Users receive minimum permissions necessary for their role."},
            {"policy": "Audit Trail", "description": "All access events logged with UTC timestamp, actor, action, component."},
            {"policy": "De-identification", "description": "Research/data-science roles access only de-identified datasets."},
        ],
        "access_matrix": [
            {"resource": "Patient Records", "Neurologist": "RW", "EEG Tech": "R", "Nurse": "RW", "Researcher": "—", "Admin": "RW", "Data Scientist": "—"},
            {"resource": "EEG Signals", "Neurologist": "RW", "EEG Tech": "RW", "Nurse": "R", "Researcher": "R*", "Admin": "R", "Data Scientist": "R*"},
            {"resource": "Medications", "Neurologist": "RW", "EEG Tech": "—", "Nurse": "RW", "Researcher": "—", "Admin": "R", "Data Scientist": "—"},
            {"resource": "AI Models", "Neurologist": "R", "EEG Tech": "R", "Nurse": "—", "Researcher": "RW", "Admin": "RW", "Data Scientist": "RW"},
            {"resource": "Audit Logs", "Neurologist": "—", "EEG Tech": "—", "Nurse": "—", "Researcher": "—", "Admin": "RW", "Data Scientist": "—"},
            {"resource": "System Config", "Neurologist": "—", "EEG Tech": "—", "Nurse": "—", "Researcher": "—", "Admin": "RW", "Data Scientist": "R"},
            {"resource": "Reports", "Neurologist": "RW", "EEG Tech": "R", "Nurse": "R", "Researcher": "RW", "Admin": "RW", "Data Scientist": "RW"},
        ],
        "glossary": [
            {"term": "RBAC", "definition": "Role-Based Access Control — permissions assigned to roles, users inherit permissions from their role."},
            {"term": "MFA", "definition": "Multi-Factor Authentication — requires two or more verification methods."},
            {"term": "Least Privilege", "definition": "Security principle: grant only the minimum access needed for job function."},
            {"term": "Separation of Duties", "definition": "No single user should control all aspects of a critical function."},
            {"term": "De-identification", "definition": "Removal of PHI/PII so data cannot be linked to an individual (HIPAA Safe Harbor)."},
            {"term": "Audit Trail", "definition": "Chronological record of system activities for accountability and compliance."},
            {"term": "Session Timeout", "definition": "Automatic termination of user session after period of inactivity."},
            {"term": "PHI", "definition": "Protected Health Information — any health data linkable to an individual (HIPAA-regulated)."},
            {"term": "R/W/D", "definition": "Read / Write / Delete — standard permission levels in access matrices."},
            {"term": "R*", "definition": "Read with de-identification — access to data with PII/PHI removed."},
        ],
        "clinical_notes": [
            "RBAC is the foundation of HIPAA-compliant access control in clinical systems.",
            "Role assignments should be reviewed quarterly and on personnel changes.",
            "MFA compliance is tracked as a security KPI — target is 100% for clinical roles.",
            "Inactive accounts should be disabled within 24 hours of role termination.",
            "All permission escalations require documented justification and approval.",
        ],
    }
