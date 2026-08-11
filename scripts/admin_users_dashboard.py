"""Admin Users Dashboard — real admin_users table (15 users, 6 roles, 8 departments).
Endpoints: /api/admin-users/overview|breakdown|definitions
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    return sqlite3.connect(DB)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def overview() -> dict:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT user_id, username, full_name, email, role, status, "
            "last_login, created_at, login_count, mfa_enabled, department "
            "FROM admin_users"
        ).fetchall()
    finally:
        conn.close()

    total = len(rows)
    active = sum(1 for r in rows if r[5] == "active")
    inactive = total - active
    mfa_enabled = sum(1 for r in rows if r[9] == 1)

    # Role distribution
    role_counts: dict[str, int] = {}
    for r in rows:
        role_counts[r[4]] = role_counts.get(r[4], 0) + 1

    # Department distribution
    dept_counts: dict[str, int] = {}
    for r in rows:
        dept_counts[r[10]] = dept_counts.get(r[10], 0) + 1

    # Activity — total logins
    total_logins = sum(r[8] for r in rows)
    avg_logins = round(total_logins / total, 1) if total else 0

    # Top active user by login_count
    top_user = max(rows, key=lambda r: r[8]) if rows else None

    role_breakdown = sorted(
        [{"role": k, "count": v, "pct": round(v / total * 100, 1)} for k, v in role_counts.items()],
        key=lambda x: -x["count"],
    )
    dept_breakdown = sorted(
        [{"department": k, "count": v} for k, v in dept_counts.items()],
        key=lambda x: -x["count"],
    )

    return {
        "generated": _now_utc(),
        "kpis": {
            "total_users": total,
            "active_users": active,
            "inactive_users": inactive,
            "mfa_enabled": mfa_enabled,
            "mfa_rate_pct": round(mfa_enabled / total * 100, 1) if total else 0,
            "total_roles": len(role_counts),
            "total_departments": len(dept_counts),
            "avg_logins": avg_logins,
        },
        "role_breakdown": role_breakdown,
        "dept_breakdown": dept_breakdown,
        "status_summary": {"active": active, "inactive": inactive},
        "top_user": {
            "user_id": top_user[0],
            "full_name": top_user[2],
            "role": top_user[4],
            "department": top_user[10],
            "login_count": top_user[8],
        } if top_user else None,
    }


def breakdown() -> dict:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT user_id, username, full_name, email, role, status, "
            "last_login, created_at, login_count, mfa_enabled, department "
            "FROM admin_users ORDER BY login_count DESC"
        ).fetchall()
    finally:
        conn.close()

    users = []
    for r in rows:
        users.append({
            "user_id": r[0],
            "username": r[1],
            "full_name": r[2],
            "email": r[2].lower().replace(" ", ".") + "@neurolab.health",
            "role": r[4],
            "status": r[5],
            "last_login": (r[6] or "")[:10],
            "created_at": (r[7] or "")[:10],
            "login_count": r[8],
            "mfa_enabled": bool(r[9]),
            "department": r[10],
        })

    # Role × Status cross-tab
    role_status: dict[str, dict] = {}
    for u in users:
        role = u["role"]
        if role not in role_status:
            role_status[role] = {"role": role, "active": 0, "inactive": 0, "total": 0}
        role_status[role][u["status"]] = role_status[role].get(u["status"], 0) + 1
        role_status[role]["total"] += 1
    role_status_list = sorted(role_status.values(), key=lambda x: -x["total"])

    # MFA non-compliant users
    mfa_missing = [u for u in users if not u["mfa_enabled"]]

    return {
        "users": users,
        "role_status_matrix": role_status_list,
        "mfa_missing": mfa_missing,
        "total_users": len(users),
        "active_count": sum(1 for u in users if u["status"] == "active"),
        "inactive_count": sum(1 for u in users if u["status"] == "inactive"),
    }


def definitions() -> dict:
    return {
        "terms": [
            {
                "term": "Admin User",
                "definition": "A registered system operator with credentials to access the NeuroLab platform. Each user is assigned a role and department.",
                "example": "Neurologist with full clinical read/write access to EEG analyses.",
            },
            {
                "term": "Role",
                "definition": "Functional position that determines a user's access scope and responsibilities.",
                "levels": {
                    "Researcher": "Read access to datasets and model outputs; no PHI write.",
                    "Neurologist": "Full clinical review, decision override, EEG interpretation.",
                    "Admin": "System configuration, user management, audit log access.",
                    "Data Scientist": "Model training, feature engineering, analytics dashboards.",
                    "EEG Tech": "EEG acquisition, artifact annotation, recording quality checks.",
                    "Nurse": "Patient monitoring, medication adherence tracking, care coordination.",
                },
            },
            {
                "term": "Status",
                "definition": "Account lifecycle state.",
                "levels": {
                    "active": "User can log in and perform role-scoped actions.",
                    "inactive": "Account suspended; login blocked pending review.",
                },
            },
            {
                "term": "MFA",
                "definition": "Multi-Factor Authentication — secondary identity verification beyond username/password. Required for PHI-access roles per HIPAA §164.312(d).",
                "example": "TOTP app or hardware key required at login.",
            },
            {
                "term": "Login Count",
                "definition": "Cumulative successful authentications since account creation. Proxy for user engagement and platform adoption.",
                "example": "Olga Tanaka (EEG Tech) has 332 logins — highest platform engagement.",
            },
            {
                "term": "Department",
                "definition": "Organizational unit the user belongs to. Determines which patient cohorts and reports a user can access by default.",
                "example": "Neurology department users see all active epilepsy patient panels.",
            },
        ],
        "fields": [
            {"field": "user_id", "description": "Unique identifier (USR-XXXX format)"},
            {"field": "username", "description": "Login credential username (firstname.lastname)"},
            {"field": "full_name", "description": "Display name for audit trail and reports"},
            {"field": "role", "description": "Functional role determining permissions scope"},
            {"field": "status", "description": "active | inactive account state"},
            {"field": "last_login", "description": "Most recent successful authentication timestamp"},
            {"field": "created_at", "description": "Account creation date"},
            {"field": "login_count", "description": "Total successful logins since creation"},
            {"field": "mfa_enabled", "description": "Boolean — MFA enrolled and active"},
            {"field": "department", "description": "Organizational unit assignment"},
        ],
        "compliance_note": "User access logs and MFA compliance are required under HIPAA §164.312 and PIPEDA Principle 7. All admin user records are retained for 7 years per audit policy.",
        "source": "clinical.db → admin_users (15 rows)",
    }
