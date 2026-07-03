"""
Data Steward / Privacy Officer Dashboard
=========================================
Real analytics from clinical.db for data governance and privacy oversight:
  - Dataset registration (uploads, file types, patient coverage)
  - PHI assessment (identifiable fields detection)
  - EEG/Video privacy risk (sensitive file type classification)
  - De-identification status (proxy analysis)
  - Access control (actor-based permission distribution)
  - Data sharing (export/share action detection)
  - Audit logs (transaction_log timeline, component/action coverage)
  - Privacy incident detection (multi-actor access, unusual patterns)
  - Data retention analysis (age distribution, policy compliance)
  - Privacy risk scoring (composite score from multiple factors)
"""
import re
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _connect():
    if not DB.exists():
        return None
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


# ════════════════════════════════════════════════════════════════════════
# 1. overview()
# ════════════════════════════════════════════════════════════════════════

def overview():
    """KPI cards + chart data for the Data Steward / Privacy Officer dashboard."""
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found"}

    cur = conn.cursor()

    # ── Basic counts ────────────────────────────────────────────────────
    total_datasets = cur.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    total_patients = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    audit_events = cur.execute("SELECT COUNT(*) FROM transaction_log").fetchone()[0]

    # ── PHI fields detected ─────────────────────────────────────────────
    # Check patients table for identifiable fields: name, age, gender
    phi_fields = []
    patients_with_name = cur.execute(
        "SELECT COUNT(*) FROM patients WHERE name IS NOT NULL AND name != ''"
    ).fetchone()[0]
    if patients_with_name > 0:
        phi_fields.append({"field": "name", "records_exposed": patients_with_name})

    patients_with_age = cur.execute(
        "SELECT COUNT(*) FROM patients WHERE age IS NOT NULL"
    ).fetchone()[0]
    if patients_with_age > 0:
        phi_fields.append({"field": "age", "records_exposed": patients_with_age})

    patients_with_gender = cur.execute(
        "SELECT COUNT(*) FROM patients WHERE gender IS NOT NULL AND gender != ''"
    ).fetchone()[0]
    if patients_with_gender > 0:
        phi_fields.append({"field": "gender", "records_exposed": patients_with_gender})

    # Check if disease field is present (diagnosis = PHI under HIPAA)
    patients_with_disease = cur.execute(
        "SELECT COUNT(*) FROM patients WHERE disease IS NOT NULL AND disease != ''"
    ).fetchone()[0]
    if patients_with_disease > 0:
        phi_fields.append({"field": "disease/diagnosis", "records_exposed": patients_with_disease})

    phi_fields_detected = len(phi_fields)

    # ── De-identification assessment ────────────────────────────────────
    # Proxy: patients whose name looks like a code/ID (e.g., starts with "P" or
    # is alphanumeric without spaces) vs full names (contain space)
    full_name_rows = cur.execute(
        "SELECT COUNT(*) FROM patients WHERE name LIKE '% %'"
    ).fetchone()[0]
    coded_name_rows = cur.execute(
        "SELECT COUNT(*) FROM patients WHERE name NOT LIKE '% %' AND name IS NOT NULL AND name != ''"
    ).fetchone()[0]
    total_named = full_name_rows + coded_name_rows
    deidentified_pct = round(
        (coded_name_rows / max(total_named, 1)) * 100, 1
    )

    # ── Unique actors ──────────────────────────────────────────────────
    unique_actors = cur.execute(
        "SELECT COUNT(DISTINCT actor) FROM transaction_log WHERE actor IS NOT NULL"
    ).fetchone()[0]

    # ── Data sharing actions ───────────────────────────────────────────
    data_sharing_actions = cur.execute(
        "SELECT COUNT(*) FROM transaction_log WHERE "
        "action LIKE '%export%' OR action LIKE '%share%' OR action LIKE '%download%' "
        "OR action LIKE '%send%' OR action LIKE '%transfer%'"
    ).fetchone()[0]

    # ── Privacy risk score ─────────────────────────────────────────────
    # Composite score 0-100 based on multiple risk factors
    risk_factors = []

    # Factor 1: PHI exposure (0-25 points)
    phi_score = min(25, phi_fields_detected * 6.25)
    risk_factors.append({"factor": "PHI Exposure", "score": phi_score, "max": 25})

    # Factor 2: De-identification gap (0-25 points)
    deid_gap_score = round((100 - deidentified_pct) / 4, 1)
    risk_factors.append({"factor": "De-identification Gap", "score": deid_gap_score, "max": 25})

    # Factor 3: Access breadth — many actors = higher risk (0-25 points)
    access_score = min(25, unique_actors * 5)
    risk_factors.append({"factor": "Access Breadth", "score": access_score, "max": 25})

    # Factor 4: Audit coverage gap (0-25 points)
    patients_with_txn = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM transaction_log WHERE patient_id IS NOT NULL"
    ).fetchone()[0]
    audit_coverage = patients_with_txn / max(total_patients, 1)
    audit_gap_score = max(0, round((1 - min(audit_coverage, 1.0)) * 25, 1))
    risk_factors.append({"factor": "Audit Coverage Gap", "score": audit_gap_score, "max": 25})

    privacy_risk_score = round(sum(f["score"] for f in risk_factors), 1)

    # ── KPI cards ───────────────────────────────────────────────────────
    kpis = {
        "total_datasets": total_datasets,
        "total_patients": total_patients,
        "phi_fields_detected": phi_fields_detected,
        "deidentified_pct": deidentified_pct,
        "unique_actors": unique_actors,
        "audit_events": audit_events,
        "data_sharing_actions": data_sharing_actions,
        "privacy_risk_score": privacy_risk_score,
    }

    # ── Chart: PHI exposure by field ───────────────────────────────────
    phi_exposure_by_field = [
        {"field": f["field"], "records_exposed": f["records_exposed"]}
        for f in phi_fields
    ]

    # ── Chart: access by actor ─────────────────────────────────────────
    actor_rows = cur.execute(
        "SELECT actor, COUNT(*) as cnt FROM transaction_log "
        "WHERE actor IS NOT NULL GROUP BY actor ORDER BY cnt DESC"
    ).fetchall()
    access_by_actor = [
        {"actor": r["actor"], "count": r["cnt"]} for r in actor_rows
    ]

    # ── Chart: actions timeline (by month) ─────────────────────────────
    timeline_rows = cur.execute(
        "SELECT strftime('%Y-%m', ts_utc) as month, COUNT(*) as cnt "
        "FROM transaction_log WHERE ts_utc IS NOT NULL "
        "GROUP BY strftime('%Y-%m', ts_utc) ORDER BY month"
    ).fetchall()
    actions_timeline = [
        {"month": r["month"], "count": r["cnt"]} for r in timeline_rows
    ]

    # ── Chart: risk distribution (pie) ─────────────────────────────────
    risk_distribution = [
        {"name": f["factor"], "value": f["score"]} for f in risk_factors
    ]

    conn.close()

    return {
        "kpis": kpis,
        "charts": {
            "phi_exposure_by_field": phi_exposure_by_field,
            "access_by_actor": access_by_actor,
            "actions_timeline": actions_timeline,
            "risk_distribution": risk_distribution,
        },
        "risk_factors": risk_factors,
    }


# ════════════════════════════════════════════════════════════════════════
# 2. breakdown()
# ════════════════════════════════════════════════════════════════════════

def breakdown():
    """Per-patient PHI profiles, access logs, file analysis, incidents."""
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found"}

    cur = conn.cursor()

    # ── Per-patient PHI assessment ─────────────────────────────────────
    patients = [dict(r) for r in cur.execute(
        "SELECT patient_id, name, age, gender, disease, department, created_at FROM patients"
    ).fetchall()]

    per_patient_phi = []
    for pat in patients:
        fields_exposed = []
        if pat.get("name") and str(pat["name"]).strip():
            fields_exposed.append("name")
        if pat.get("age") is not None:
            fields_exposed.append("age")
        if pat.get("gender") and str(pat["gender"]).strip():
            fields_exposed.append("gender")
        if pat.get("disease") and str(pat["disease"]).strip():
            fields_exposed.append("disease/diagnosis")
        if pat.get("department") and str(pat["department"]).strip():
            fields_exposed.append("department")

        # De-identification proxy: name with space = likely full name = not de-identified
        name = pat.get("name") or ""
        is_deidentified = " " not in name and len(name) > 0

        per_patient_phi.append({
            "patient_id": pat["patient_id"],
            "name": pat["name"],
            "fields_exposed": fields_exposed,
            "phi_field_count": len(fields_exposed),
            "is_deidentified": is_deidentified,
            "created_at": pat["created_at"],
        })

    # ── Access log (recent transaction_log entries) ────────────────────
    recent_rows = cur.execute(
        "SELECT id, patient_id, component, action, actor, ref_id, detail, ts_utc, ts_local "
        "FROM transaction_log ORDER BY id DESC LIMIT 50"
    ).fetchall()
    access_log = [
        {
            "id": r["id"],
            "patient_id": r["patient_id"],
            "component": r["component"],
            "action": r["action"],
            "actor": r["actor"],
            "ref_id": r["ref_id"],
            "detail": r["detail"],
            "ts_utc": r["ts_utc"],
            "ts_local": r["ts_local"],
        }
        for r in recent_rows
    ]

    # ── File type analysis ─────────────────────────────────────────────
    file_rows = cur.execute(
        "SELECT file_name FROM uploads WHERE file_name IS NOT NULL"
    ).fetchall()

    file_type_counts = defaultdict(int)
    eeg_video_files = []
    for r in file_rows:
        fname = r["file_name"]
        # Extract extension
        if "." in fname:
            ext = fname.rsplit(".", 1)[-1].lower()
        else:
            ext = "unknown"
        file_type_counts[ext] += 1

        # Flag EEG/video file types as privacy-sensitive
        sensitive_extensions = {
            "edf", "bdf", "eeg", "set", "fdt", "cnt", "vhdr", "vmrk",  # EEG formats
            "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm",           # Video formats
            "wav", "mp3",                                                  # Audio
        }
        if ext in sensitive_extensions:
            eeg_video_files.append({
                "file_name": fname,
                "extension": ext,
                "privacy_risk": "high" if ext in {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"} else "medium",
                "risk_reason": (
                    "Video may contain patient face/identifying features"
                    if ext in {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"}
                    else "EEG/audio data is patient-specific biometric data"
                ),
            })

    file_type_analysis = [
        {"extension": ext, "count": cnt} for ext, cnt in
        sorted(file_type_counts.items(), key=lambda x: -x[1])
    ]

    # ── Actor permissions (who did what) ───────────────────────────────
    actor_action_rows = cur.execute(
        "SELECT actor, action, COUNT(*) as cnt FROM transaction_log "
        "WHERE actor IS NOT NULL GROUP BY actor, action ORDER BY actor, cnt DESC"
    ).fetchall()

    actor_permissions = defaultdict(list)
    for r in actor_action_rows:
        actor_permissions[r["actor"]].append({
            "action": r["action"],
            "count": r["cnt"],
        })
    actor_permissions = dict(actor_permissions)

    # ── Retention analysis ─────────────────────────────────────────────
    # Analyze data age from created_at in patients and uploads
    now = datetime.utcnow()
    retention_buckets = {"0-30_days": 0, "31-90_days": 0, "91-180_days": 0, "181-365_days": 0, "over_1_year": 0}

    patient_dates = cur.execute(
        "SELECT created_at FROM patients WHERE created_at IS NOT NULL"
    ).fetchall()
    upload_dates = cur.execute(
        "SELECT created_at FROM uploads WHERE created_at IS NOT NULL"
    ).fetchall()

    all_dates = [r["created_at"] for r in patient_dates] + [r["created_at"] for r in upload_dates]
    for d in all_dates:
        try:
            # Strip timezone offset (e.g., -06:00, +00:00, Z) to get naive datetime
            raw = re.sub(r'[+-]\d{2}:\d{2}$', '', str(d)).replace('Z', '').strip()
            dt = datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            try:
                dt = datetime.strptime(str(d)[:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue
        age_days = max((now - dt).days, 0)
        if age_days <= 30:
            retention_buckets["0-30_days"] += 1
        elif age_days <= 90:
            retention_buckets["31-90_days"] += 1
        elif age_days <= 180:
            retention_buckets["91-180_days"] += 1
        elif age_days <= 365:
            retention_buckets["181-365_days"] += 1
        else:
            retention_buckets["over_1_year"] += 1

    retention_analysis = [
        {"period": k.replace("_", " "), "record_count": v}
        for k, v in retention_buckets.items()
    ]

    # ── Incident candidates ────────────────────────────────────────────
    # Detect potential privacy incidents:
    # 1. Patients accessed by multiple actors
    multi_actor_rows = cur.execute(
        "SELECT patient_id, COUNT(DISTINCT actor) as n_actors, COUNT(*) as n_events "
        "FROM transaction_log WHERE patient_id IS NOT NULL AND actor IS NOT NULL "
        "GROUP BY patient_id HAVING COUNT(DISTINCT actor) > 1 "
        "ORDER BY n_actors DESC"
    ).fetchall()

    # 2. High-frequency access (patients with > 20 events)
    high_freq_rows = cur.execute(
        "SELECT patient_id, COUNT(*) as n_events "
        "FROM transaction_log WHERE patient_id IS NOT NULL "
        "GROUP BY patient_id HAVING COUNT(*) > 20 "
        "ORDER BY n_events DESC"
    ).fetchall()

    # 3. Unusual action patterns (actors with many distinct actions)
    unusual_actor_rows = cur.execute(
        "SELECT actor, COUNT(DISTINCT action) as n_actions, COUNT(*) as n_events "
        "FROM transaction_log WHERE actor IS NOT NULL "
        "GROUP BY actor HAVING COUNT(DISTINCT action) > 5 "
        "ORDER BY n_actions DESC"
    ).fetchall()

    incident_candidates = {
        "multi_actor_access": [
            {
                "patient_id": r["patient_id"],
                "distinct_actors": r["n_actors"],
                "total_events": r["n_events"],
                "severity": "high" if r["n_actors"] >= 3 else "medium",
                "reason": f"Patient data accessed by {r['n_actors']} distinct actors",
            }
            for r in multi_actor_rows
        ],
        "high_frequency_access": [
            {
                "patient_id": r["patient_id"],
                "total_events": r["n_events"],
                "severity": "high" if r["n_events"] > 50 else "medium",
                "reason": f"Unusually high access frequency ({r['n_events']} events)",
            }
            for r in high_freq_rows
        ],
        "broad_permission_actors": [
            {
                "actor": r["actor"],
                "distinct_actions": r["n_actions"],
                "total_events": r["n_events"],
                "severity": "medium",
                "reason": f"Actor performed {r['n_actions']} distinct action types",
            }
            for r in unusual_actor_rows
        ],
    }

    # ── Dataset Registration ─────────────────────────────────────────────
    upload_rows = cur.execute(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at DESC"
    ).fetchall()

    dataset_records = []
    ds_file_type_dist = defaultdict(int)
    ds_timeline = defaultdict(int)
    ds_unique_patients = set()
    for r in upload_rows:
        fname = r["file_name"] or ""
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "unknown"
        ds_file_type_dist[ext] += 1
        if r["patient_id"]:
            ds_unique_patients.add(r["patient_id"])
        # timeline by month
        ts = r["created_at"] or ""
        month = ts[:7] if len(ts) >= 7 else "unknown"
        ds_timeline[month] += 1
        dataset_records.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "file_name": fname,
            "file_type": ext,
            "disease": r["disease"],
            "department": r["department"],
            "created_at": r["created_at"],
        })

    dataset_registration = {
        "records": dataset_records,
        "summary": {
            "total_uploads": len(dataset_records),
            "unique_patients_with_data": len(ds_unique_patients),
            "file_type_distribution": [
                {"file_type": k, "count": v}
                for k, v in sorted(ds_file_type_dist.items(), key=lambda x: -x[1])
            ],
            "registration_timeline": [
                {"month": k, "count": v}
                for k, v in sorted(ds_timeline.items())
            ],
        },
    }

    # ── De-identification Detail ──────────────────────────────────────────
    # HIPAA Safe Harbor 18 identifiers — check which are present per patient
    SAFE_HARBOR_FIELDS = [
        "name", "geographic_data", "dates", "phone_number", "fax_number",
        "email_address", "ssn", "medical_record_number", "health_plan_beneficiary",
        "account_number", "certificate_license", "vehicle_identifiers",
        "device_identifiers", "web_urls", "ip_address", "biometric_identifiers",
        "facial_photographs", "other_unique_identifier",
    ]

    deid_per_patient = []
    fully_deid = 0
    partially_deid = 0
    not_deid = 0

    for pat in patients:
        identifiers_present = []
        identifiers_removed = []

        # Check fields we can actually detect in the database
        name_val = pat.get("name") or ""
        if name_val.strip() and " " in name_val:
            # Full name present — identifiable
            identifiers_present.append("name")
        else:
            identifiers_removed.append("name")

        # Age/dates — age is a quasi-identifier (dates if >89 are PHI)
        if pat.get("age") is not None:
            identifiers_present.append("dates")
        else:
            identifiers_removed.append("dates")

        # Disease/diagnosis — can be an other_unique_identifier in context
        if pat.get("disease") and str(pat["disease"]).strip():
            identifiers_present.append("other_unique_identifier")
        else:
            identifiers_removed.append("other_unique_identifier")

        # Department — geographic / organizational data
        if pat.get("department") and str(pat["department"]).strip():
            identifiers_present.append("geographic_data")
        else:
            identifiers_removed.append("geographic_data")

        # Gender is a quasi-identifier but not one of the 18; still flag it
        # Check if patient has EEG uploads (biometric identifiers)
        eeg_count = cur.execute(
            "SELECT COUNT(*) FROM uploads WHERE patient_id = ? AND ("
            "file_name LIKE '%.edf' OR file_name LIKE '%.bdf' OR "
            "file_name LIKE '%.eeg' OR file_name LIKE '%.set' OR "
            "file_name LIKE '%.cnt' OR file_name LIKE '%.vhdr')",
            (pat["patient_id"],)
        ).fetchone()[0]
        if eeg_count > 0:
            identifiers_present.append("biometric_identifiers")
        else:
            identifiers_removed.append("biometric_identifiers")

        # Check if patient has video uploads (facial photographs proxy)
        vid_count = cur.execute(
            "SELECT COUNT(*) FROM uploads WHERE patient_id = ? AND ("
            "file_name LIKE '%.mp4' OR file_name LIKE '%.avi' OR "
            "file_name LIKE '%.mov' OR file_name LIKE '%.mkv' OR "
            "file_name LIKE '%.wmv')",
            (pat["patient_id"],)
        ).fetchone()[0]
        if vid_count > 0:
            identifiers_present.append("facial_photographs")
        else:
            identifiers_removed.append("facial_photographs")

        # Remaining Safe Harbor fields — not stored in this DB, mark removed
        checked_fields = {
            "name", "dates", "other_unique_identifier", "geographic_data",
            "biometric_identifiers", "facial_photographs",
        }
        for field in SAFE_HARBOR_FIELDS:
            if field not in checked_fields:
                identifiers_removed.append(field)

        total_checked = len(SAFE_HARBOR_FIELDS)
        removed_count = len(identifiers_removed)
        compliance_pct = round((removed_count / max(total_checked, 1)) * 100, 1)

        # Recommendation
        if compliance_pct == 100.0:
            recommendation = "Fully compliant — no action needed"
            fully_deid += 1
        elif compliance_pct >= 70.0:
            recommendation = f"Partially de-identified — address: {', '.join(identifiers_present)}"
            partially_deid += 1
        else:
            recommendation = f"High PHI exposure — urgent de-identification needed for: {', '.join(identifiers_present)}"
            not_deid += 1

        deid_per_patient.append({
            "patient_id": pat["patient_id"],
            "identifiers_present": identifiers_present,
            "identifiers_removed": identifiers_removed,
            "safe_harbor_compliance_pct": compliance_pct,
            "recommendation": recommendation,
        })

    deidentification_detail = {
        "per_patient": deid_per_patient,
        "summary": {
            "total_patients": len(patients),
            "fully_deidentified": fully_deid,
            "partially_deidentified": partially_deid,
            "not_deidentified": not_deid,
            "safe_harbor_fields_checked": SAFE_HARBOR_FIELDS,
        },
    }

    # ── Data Sharing Detail ───────────────────────────────────────────────
    sharing_rows = cur.execute(
        "SELECT id, patient_id, actor, action, detail, ts_utc "
        "FROM transaction_log WHERE "
        "action LIKE '%export%' OR action LIKE '%share%' OR action LIKE '%download%' "
        "OR action LIKE '%send%' OR action LIKE '%transfer%' "
        "ORDER BY ts_utc DESC"
    ).fetchall()

    sharing_records = []
    sharing_by_actor = defaultdict(int)
    sharing_by_action = defaultdict(int)
    sharing_timeline = defaultdict(int)
    for r in sharing_rows:
        sharing_records.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "actor": r["actor"],
            "action": r["action"],
            "detail": r["detail"],
            "ts_utc": r["ts_utc"],
        })
        if r["actor"]:
            sharing_by_actor[r["actor"]] += 1
        if r["action"]:
            sharing_by_action[r["action"]] += 1
        ts = r["ts_utc"] or ""
        month = ts[:7] if len(ts) >= 7 else "unknown"
        sharing_timeline[month] += 1

    data_sharing_detail = {
        "records": sharing_records,
        "summary": {
            "total_sharing_actions": len(sharing_records),
            "sharing_by_actor": [
                {"actor": k, "count": v}
                for k, v in sorted(sharing_by_actor.items(), key=lambda x: -x[1])
            ],
            "sharing_by_action_type": [
                {"action": k, "count": v}
                for k, v in sorted(sharing_by_action.items(), key=lambda x: -x[1])
            ],
            "sharing_timeline": [
                {"month": k, "count": v}
                for k, v in sorted(sharing_timeline.items())
            ],
        },
    }

    # ── Risk Assessment ───────────────────────────────────────────────────
    # Per-patient privacy risk profile
    risk_level_counts = defaultdict(int)
    risk_per_patient = []

    for pat in patients:
        pid = pat["patient_id"]
        risk_score = 0
        risk_factors_list = []

        # Factor 1: has full name (identifiable)
        name_val = pat.get("name") or ""
        if name_val.strip() and " " in name_val:
            risk_score += 20
            risk_factors_list.append("has_full_name")

        # Factor 2: diagnosis exposed
        if pat.get("disease") and str(pat["disease"]).strip():
            risk_score += 15
            risk_factors_list.append("has_diagnosis_exposed")

        # Factor 3: multiple actors accessed this patient
        actor_count = cur.execute(
            "SELECT COUNT(DISTINCT actor) FROM transaction_log "
            "WHERE patient_id = ? AND actor IS NOT NULL",
            (pid,)
        ).fetchone()[0]
        if actor_count > 1:
            risk_score += 15
            risk_factors_list.append("multiple_actors_accessed")

        # Factor 4: high frequency access (>20 events)
        event_count = cur.execute(
            "SELECT COUNT(*) FROM transaction_log WHERE patient_id = ?",
            (pid,)
        ).fetchone()[0]
        if event_count > 20:
            risk_score += 15
            risk_factors_list.append("high_frequency_access")
        elif event_count > 10:
            risk_score += 8
            risk_factors_list.append("high_frequency_access")

        # Factor 5: sensitive files (EEG/video)
        sensitive_count = cur.execute(
            "SELECT COUNT(*) FROM uploads WHERE patient_id = ? AND ("
            "file_name LIKE '%.edf' OR file_name LIKE '%.bdf' OR "
            "file_name LIKE '%.eeg' OR file_name LIKE '%.set' OR "
            "file_name LIKE '%.mp4' OR file_name LIKE '%.avi' OR "
            "file_name LIKE '%.mov' OR file_name LIKE '%.mkv')",
            (pid,)
        ).fetchone()[0]
        if sensitive_count > 0:
            risk_score += 20
            risk_factors_list.append("sensitive_files")

        # Factor 6: no audit trail
        if event_count == 0:
            risk_score += 15
            risk_factors_list.append("no_audit_trail")

        # Clamp to 0-100
        risk_score = min(100, max(0, risk_score))

        # Determine risk level
        if risk_score >= 70:
            level = "critical"
        elif risk_score >= 45:
            level = "high"
        elif risk_score >= 20:
            level = "medium"
        else:
            level = "low"

        risk_level_counts[level] += 1
        risk_per_patient.append({
            "patient_id": pid,
            "risk_score": risk_score,
            "risk_level": level,
            "risk_factors": risk_factors_list,
        })

    # Sort by risk_score descending
    risk_per_patient.sort(key=lambda x: -x["risk_score"])

    risk_assessment = {
        "per_patient": risk_per_patient,
        "risk_distribution": [
            {"risk_level": lvl, "count": risk_level_counts.get(lvl, 0)}
            for lvl in ["critical", "high", "medium", "low"]
        ],
    }

    conn.close()

    return {
        "per_patient_phi": per_patient_phi,
        "access_log": access_log,
        "file_type_analysis": file_type_analysis,
        "eeg_video_privacy": eeg_video_files,
        "actor_permissions": actor_permissions,
        "retention_analysis": retention_analysis,
        "incident_candidates": incident_candidates,
        "dataset_registration": dataset_registration,
        "deidentification_detail": deidentification_detail,
        "data_sharing_detail": data_sharing_detail,
        "risk_assessment": risk_assessment,
    }


# ════════════════════════════════════════════════════════════════════════
# 3. definitions()
# ════════════════════════════════════════════════════════════════════════

def definitions():
    """Data stewardship / privacy concepts, quality metrics, compliance refs, remediation."""
    return {
        "terms": [
            {
                "term": "Protected Health Information (PHI)",
                "definition": (
                    "Any individually identifiable health information held or transmitted by a "
                    "covered entity. Under HIPAA, PHI includes 18 identifier categories: names, "
                    "dates, phone/fax numbers, email, SSN, medical record numbers, account numbers, "
                    "biometric identifiers (including EEG), facial photographs, and any other unique "
                    "identifying number. In this system, patient names, ages, genders, and diagnoses "
                    "constitute PHI requiring protection."
                ),
            },
            {
                "term": "De-identification",
                "definition": (
                    "The process of removing or transforming PHI so that the resulting data cannot "
                    "be used to identify an individual. HIPAA recognizes two methods: Safe Harbor "
                    "(removal of 18 identifier types) and Expert Determination (statistical/scientific "
                    "assessment). This dashboard uses a proxy metric based on whether patient records "
                    "use coded identifiers vs. full names."
                ),
            },
            {
                "term": "Data Stewardship",
                "definition": (
                    "The management and oversight of an organization's data assets to ensure data "
                    "quality, integrity, security, and compliance with regulatory requirements. "
                    "The Data Steward is responsible for data governance policies, access controls, "
                    "de-identification procedures, and ensuring lawful use of patient data in "
                    "AI-assisted clinical research."
                ),
            },
            {
                "term": "HIPAA Safe Harbor",
                "definition": (
                    "A de-identification method under HIPAA (45 CFR 164.514(b)(2)) that requires "
                    "removal of 18 specific identifiers from health data. Once all identifiers are "
                    "removed and the covered entity has no actual knowledge the remaining data could "
                    "identify an individual, the data is considered de-identified and no longer "
                    "subject to HIPAA privacy protections."
                ),
            },
            {
                "term": "Access Control",
                "definition": (
                    "Security mechanisms that regulate who can view, modify, or interact with "
                    "patient data. In clinical AI systems, access control includes role-based "
                    "permissions (clinician, researcher, administrator), audit logging of all "
                    "data interactions, and the principle of minimum necessary access. The "
                    "transaction log tracks all actor-initiated data events."
                ),
            },
            {
                "term": "Audit Trail",
                "definition": (
                    "A chronological record of system activities that provides documentary "
                    "evidence of the sequence of actions affecting a specific operation, procedure, "
                    "or event. In this system, the transaction_log table serves as the audit trail, "
                    "recording every data access, modification, and clinical action with timestamps, "
                    "actors, and affected patient records."
                ),
            },
            {
                "term": "Data Retention",
                "definition": (
                    "Policies governing how long patient data is stored and when it must be "
                    "securely destroyed. HIPAA requires retention of designated record sets for "
                    "6 years from creation or last effective date. Research data may have longer "
                    "retention requirements per institutional policy and funding agency mandates. "
                    "This dashboard tracks data age to identify records approaching retention limits."
                ),
            },
            {
                "term": "Privacy Impact Assessment (PIA)",
                "definition": (
                    "A systematic process to evaluate the potential effects of a project, system, "
                    "or technology on individual privacy. A PIA identifies privacy risks, assesses "
                    "their likelihood and impact, and recommends mitigations. For AI-assisted "
                    "clinical systems, PIAs must address algorithmic bias, re-identification risk "
                    "from model outputs, and cross-dataset linkage vulnerabilities."
                ),
            },
        ],
        "quality_metrics": [
            {
                "metric": "PHI Exposure Rate",
                "description": (
                    "Percentage of patient records with identifiable PHI fields still present. "
                    "Target: 0% for research datasets, controlled access for clinical datasets."
                ),
                "target": "0% (research) / controlled (clinical)",
            },
            {
                "metric": "De-identification Coverage",
                "description": (
                    "Percentage of patient records that have been successfully de-identified "
                    "(coded identifiers, no full names). Target: 100% for research use."
                ),
                "target": "100%",
            },
            {
                "metric": "Audit Trail Completeness",
                "description": (
                    "Percentage of patient records with corresponding transaction log entries, "
                    "confirming all data interactions are tracked. Target: 100%."
                ),
                "target": "100%",
            },
            {
                "metric": "Privacy Incident Rate",
                "description": (
                    "Number of detected potential privacy incidents (multi-actor access, "
                    "unusual access patterns) per reporting period. Target: 0."
                ),
                "target": "0 incidents",
            },
        ],
        "compliance_references": [
            {
                "standard": "HIPAA (Health Insurance Portability and Accountability Act)",
                "description": (
                    "U.S. federal law establishing national standards for the protection of "
                    "individually identifiable health information. The Privacy Rule (45 CFR 160, 164) "
                    "governs use and disclosure of PHI. The Security Rule mandates administrative, "
                    "physical, and technical safeguards for electronic PHI."
                ),
            },
            {
                "standard": "GDPR (General Data Protection Regulation)",
                "description": (
                    "EU regulation on data protection and privacy. Applies when processing personal "
                    "data of EU residents. Requires lawful basis for processing, data minimization, "
                    "purpose limitation, right to erasure, and Data Protection Impact Assessments "
                    "for high-risk processing such as health data analytics."
                ),
            },
            {
                "standard": "45 CFR 164 (HIPAA Security and Privacy Rules)",
                "description": (
                    "Code of Federal Regulations implementing HIPAA's privacy and security "
                    "requirements. Part 164.512 governs uses and disclosures for research. "
                    "Part 164.514 defines de-identification standards (Safe Harbor and Expert "
                    "Determination methods). Part 164.530 mandates administrative requirements "
                    "including training, safeguards, and complaint procedures."
                ),
            },
            {
                "standard": "HITECH Act (Health Information Technology for Economic and Clinical Health)",
                "description": (
                    "Extends HIPAA's scope to business associates, strengthens enforcement with "
                    "tiered civil monetary penalties (up to $1.5M per violation category per year), "
                    "mandates breach notification within 60 days for breaches affecting 500+ "
                    "individuals, and promotes meaningful use of electronic health records."
                ),
            },
        ],
        "remediation_strategies": [
            {
                "strategy": "PHI De-identification Sprint",
                "description": (
                    "Conduct a systematic review of all patient records to replace identifiable "
                    "names with coded identifiers, generalize ages to ranges, and remove or "
                    "encrypt direct identifiers. Apply HIPAA Safe Harbor checklist to verify "
                    "removal of all 18 identifier categories. Timeline: 2-week sprint with "
                    "verification audit."
                ),
            },
            {
                "strategy": "Access Control Tightening",
                "description": (
                    "Implement role-based access control (RBAC) to enforce minimum necessary "
                    "access. Assign data access roles (clinician, researcher, admin, auditor) "
                    "with specific permitted actions. Require multi-factor authentication for "
                    "PHI access. Review and revoke inactive actor permissions quarterly."
                ),
            },
            {
                "strategy": "Privacy Incident Response Protocol",
                "description": (
                    "Establish a formal incident response plan: (1) detect via automated anomaly "
                    "alerts on multi-actor access and high-frequency patterns, (2) contain by "
                    "temporarily restricting affected accounts, (3) investigate root cause, "
                    "(4) remediate with access control updates, (5) notify per HITECH breach "
                    "notification requirements if PHI was compromised."
                ),
            },
            {
                "strategy": "EEG/Video Data Protection",
                "description": (
                    "Classify EEG recordings as biometric PHI and video files as high-risk "
                    "re-identification sources. Implement: encryption at rest (AES-256), "
                    "face-blurring for video before research use, EEG header stripping to remove "
                    "patient identifiers, and separate storage with heightened access controls "
                    "for multimedia clinical data."
                ),
            },
        ],
    }
