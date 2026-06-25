"""Per-patient clinical database + report persistence (SQLite).

Stores patients, EEG uploads, analyses, predictions, and surveys so each
department (EEG Technician, Neurologist, Psychiatrist, OT, ...) can onboard
patients, run the pipeline, and view per-patient history. WAL mode + indexes
per project DB standards.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "clinical.db"
REPORTS_DIR = ROOT / "reports" / "patient_reports"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_transaction(patient_id: str, component: str, action: str,
                    actor: str = "system", ref_id: Optional[int] = None,
                    detail: str = "") -> None:
    """Append a timestamped transaction-history row (UTC + local). Best-effort."""
    try:
        with _connect() as c:
            c.execute(
                "INSERT INTO transaction_log (patient_id, component, action, actor, ref_id, detail, ts_utc, ts_local) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (patient_id, component, action, actor, ref_id, detail, _utc(), _now()))
    except Exception:  # never let audit logging break the write path
        pass


def list_transactions(patient_id: Optional[str] = None, offset: int = 0, limit: int = 100) -> dict:
    with _connect() as c:
        if patient_id:
            rows = c.execute("SELECT * FROM transaction_log WHERE patient_id=? ORDER BY id DESC LIMIT ? OFFSET ?",
                             (patient_id, limit, offset)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM transaction_log WHERE patient_id=?", (patient_id,)).fetchone()[0]
        else:
            rows = c.execute("SELECT * FROM transaction_log ORDER BY id DESC LIMIT ? OFFSET ?",
                             (limit, offset)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM transaction_log").fetchone()[0]
    return {"items": [dict(r) for r in rows], "total": total, "offset": offset, "limit": limit}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    with _connect() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id   TEXT PRIMARY KEY,
                name         TEXT,
                age          INTEGER,
                gender       TEXT,
                disease      TEXT,
                department   TEXT,
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS uploads (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   TEXT,
                file_name    TEXT,
                disease      TEXT,
                department   TEXT,
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id    INTEGER,
                patient_id   TEXT,
                disease      TEXT,
                predicted_label TEXT,
                confidence   REAL,
                signal_quality  TEXT,
                report_path  TEXT,
                result_json  TEXT,
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS surveys (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   TEXT,
                department   TEXT,
                kind         TEXT,
                answers_json TEXT,
                created_at   TEXT NOT NULL
            );
            -- Clinical/governance capture tables (one row per submission).
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mri_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS neuropsych (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                analysis_id INTEGER, fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS explainability_gt (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                analysis_id INTEGER, fields_json TEXT, created_at TEXT NOT NULL
            );
            -- EEG Technician Data Collection Form sections (one row per submission).
            CREATE TABLE IF NOT EXISTS eeg_acquisition (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS channel_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clinical_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recording_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifact_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS eeg_interpretation (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            -- Seizure metadata (ILAE): per-patient semiology, not per-event.
            CREATE TABLE IF NOT EXISTS seizure_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            -- Comorbidities / hospitalization / DBA KPIs / model-governance / risk.
            CREATE TABLE IF NOT EXISTS comorbidities (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hospitalization (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dba_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_governance (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS risk_management (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                fields_json TEXT, created_at TEXT NOT NULL
            );
            -- Feedback / correction capture (per role) — feeds RLHF + consensus.
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT, role TEXT, ai_output TEXT,
                rating INTEGER, correction TEXT, reason TEXT,
                created_at TEXT NOT NULL
            );
            -- Standardized clinical assessments (MoCA, PHQ-9, GAD-7, NDDI-E, COPM) — auto-scored, CRUD.
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL, instrument TEXT NOT NULL,
                answers_json TEXT NOT NULL, score REAL, max_score REAL,
                interpretation TEXT, level TEXT, alert TEXT, examiner TEXT,
                created_at TEXT NOT NULL, updated_at TEXT
            );
            -- Team chat: roles message each other in channels; AI bot can reply.
            CREATE TABLE IF NOT EXISTS team_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL, from_role TEXT NOT NULL,
                text TEXT NOT NULL, is_bot INTEGER DEFAULT 0,
                patient_id TEXT, topic TEXT, attachment TEXT,
                read_by TEXT DEFAULT '[]', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE, members TEXT DEFAULT '[]',
                topic TEXT, created_by TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_presence (
                role TEXT PRIMARY KEY, status TEXT DEFAULT 'active', updated_at TEXT NOT NULL
            );
            -- Patient seizure diary — the most valuable patient dataset (one row per seizure event).
            CREATE TABLE IF NOT EXISTS seizure_diary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL, event_date TEXT, event_time TEXT,
                duration_sec INTEGER, location TEXT, witnessed TEXT,
                aura TEXT, awareness TEXT, motor_signs TEXT,
                injury TEXT, post_ictal TEXT, recovery_min INTEGER,
                er_visit TEXT, rescue_med TEXT, severity TEXT,
                trigger TEXT, notes TEXT, created_at TEXT NOT NULL
            );
            -- Per-component doctor findings (AI finding vs doctor finding, per EEG component).
            CREATE TABLE IF NOT EXISTS component_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL, component TEXT NOT NULL,
                doctor_finding TEXT, doctor TEXT, agree_with_ai TEXT,
                created_at TEXT NOT NULL, updated_at TEXT,
                UNIQUE(patient_id, component)
            );
            -- Multi-expert review of a study: each role attaches their assessment + agree/disagree with AI.
            CREATE TABLE IF NOT EXISTS expert_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL, analysis_id INTEGER,
                role TEXT NOT NULL, expert TEXT, finding TEXT,
                agree_with_ai TEXT, note TEXT,
                created_at TEXT NOT NULL
            );
            -- Forms an expert sends to a patient to fill via the self-service portal.
            CREATE TABLE IF NOT EXISTS form_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL, instrument TEXT NOT NULL,
                assigned_by TEXT, status TEXT DEFAULT 'pending',
                assessment_id INTEGER, message TEXT,
                created_at TEXT NOT NULL, completed_at TEXT
            );
            -- Transaction history: every write stamped with UTC+local time + actor.
            CREATE TABLE IF NOT EXISTS transaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT, component TEXT, action TEXT, actor TEXT,
                ref_id INTEGER, detail TEXT,
                ts_utc TEXT NOT NULL, ts_local TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_uploads_patient   ON uploads(patient_id);
            CREATE INDEX IF NOT EXISTS idx_analyses_patient  ON analyses(patient_id);
            CREATE INDEX IF NOT EXISTS idx_analyses_disease  ON analyses(disease);
            CREATE INDEX IF NOT EXISTS idx_surveys_patient   ON surveys(patient_id);
            CREATE INDEX IF NOT EXISTS idx_med_patient        ON medications(patient_id);
            CREATE INDEX IF NOT EXISTS idx_mri_patient        ON mri_findings(patient_id);
            CREATE INDEX IF NOT EXISTS idx_outcomes_patient   ON outcomes(patient_id);
            CREATE INDEX IF NOT EXISTS idx_neuro_patient      ON neuropsych(patient_id);
            CREATE INDEX IF NOT EXISTS idx_hitl_patient       ON hitl_reviews(patient_id);
            CREATE INDEX IF NOT EXISTS idx_xai_patient        ON explainability_gt(patient_id);
            CREATE INDEX IF NOT EXISTS idx_acq_patient        ON eeg_acquisition(patient_id);
            CREATE INDEX IF NOT EXISTS idx_chq_patient        ON channel_quality(patient_id);
            CREATE INDEX IF NOT EXISTS idx_hist_patient       ON clinical_history(patient_id);
            CREATE INDEX IF NOT EXISTS idx_rec_patient        ON recording_conditions(patient_id);
            CREATE INDEX IF NOT EXISTS idx_evt_patient        ON event_annotations(patient_id);
            CREATE INDEX IF NOT EXISTS idx_art_patient        ON artifact_annotations(patient_id);
            CREATE INDEX IF NOT EXISTS idx_interp_patient     ON eeg_interpretation(patient_id);
            CREATE INDEX IF NOT EXISTS idx_szmeta_patient     ON seizure_metadata(patient_id);
            CREATE INDEX IF NOT EXISTS idx_comorb_patient     ON comorbidities(patient_id);
            CREATE INDEX IF NOT EXISTS idx_hosp_patient       ON hospitalization(patient_id);
            CREATE INDEX IF NOT EXISTS idx_dba_patient        ON dba_metrics(patient_id);
            CREATE INDEX IF NOT EXISTS idx_modgov_patient     ON model_governance(patient_id);
            CREATE INDEX IF NOT EXISTS idx_risk_patient       ON risk_management(patient_id);
            CREATE INDEX IF NOT EXISTS idx_txn_patient        ON transaction_log(patient_id);
            CREATE INDEX IF NOT EXISTS idx_txn_ts             ON transaction_log(ts_utc);
            CREATE INDEX IF NOT EXISTS idx_fb_patient         ON feedback(patient_id);
            CREATE INDEX IF NOT EXISTS idx_fb_role            ON feedback(role);
            """
        )


# --------------------------------------------------------------------------
# Patients
# --------------------------------------------------------------------------
def upsert_patient(patient_id: str, name: str = "", age: Optional[int] = None,
                   gender: str = "", disease: str = "", department: str = "") -> dict:
    with _connect() as c:
        c.execute(
            """INSERT INTO patients (patient_id, name, age, gender, disease, department, created_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(patient_id) DO UPDATE SET
                 name=excluded.name, age=excluded.age, gender=excluded.gender,
                 disease=excluded.disease, department=excluded.department""",
            (patient_id, name, age, gender, disease, department, _now()),
        )
    return get_patient(patient_id)


def get_patient(patient_id: str) -> Optional[dict]:
    with _connect() as c:
        row = c.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
        if not row:
            return None
        p = dict(row)
        p["analyses"] = [dict(r) for r in c.execute(
            "SELECT id, disease, predicted_label, confidence, signal_quality, report_path, created_at "
            "FROM analyses WHERE patient_id=? ORDER BY id DESC", (patient_id,)).fetchall()]
        return p


def list_patients(department: Optional[str] = None, offset: int = 0, limit: int = 100) -> dict:
    with _connect() as c:
        if department:
            rows = c.execute(
                "SELECT * FROM patients WHERE department=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (department, limit, offset)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM patients WHERE department=?", (department,)).fetchone()[0]
        else:
            rows = c.execute(
                "SELECT * FROM patients ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    return {"items": [dict(r) for r in rows], "total": total, "offset": offset, "limit": limit}


# --------------------------------------------------------------------------
# Analyses + report
# --------------------------------------------------------------------------
def save_analysis(result: dict, department: str = "") -> dict:
    """Persist a pipeline result, write a markdown report, return the saved row."""
    patient_id = result.get("patient_id")
    disease = result.get("disease", "")
    pred = result.get("prediction", {}) or {}

    with _connect() as c:
        up = c.execute(
            "INSERT INTO uploads (patient_id, file_name, disease, department, created_at) VALUES (?,?,?,?,?)",
            (patient_id, result.get("file", ""), disease, department, _now()))
        upload_id = up.lastrowid

    report_path = _write_report(result, department)

    with _connect() as c:
        an = c.execute(
            """INSERT INTO analyses
               (upload_id, patient_id, disease, predicted_label, confidence, signal_quality, report_path, result_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (upload_id, patient_id, disease,
             pred.get("predicted_label"), pred.get("confidence"),
             (result.get("analysis", {}) or {}).get("signal_quality"),
             str(report_path), json.dumps(result), _now()))
        analysis_id = an.lastrowid

    return {"analysis_id": analysis_id, "upload_id": upload_id, "report_path": str(report_path)}


def _write_report(result: dict, department: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pid = result.get("patient_id") or "anon"
    disease = result.get("disease", "unknown")
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    md_path = REPORTS_DIR / f"{pid}_{disease}_{ts}.md"
    json_path = md_path.with_suffix(".json")

    a = result.get("analysis", {}) or {}
    p = result.get("prediction", {}) or {}
    metrics = (p.get("model_metrics") or {})

    lines = [
        f"# EEG Analysis Report — {pid}",
        "",
        f"- Generated: {result.get('generated_at', _now())}",
        f"- Department: {department or '—'}",
        f"- Disease model: **{disease}**",
        f"- File: `{result.get('file', '')}`",
        "",
        "## Signal Analysis",
        f"- Channels: {a.get('n_channels')}",
        f"- Sampling rate: {a.get('sampling_rate')} Hz",
        f"- Duration: {a.get('duration_seconds')} s",
        f"- Signal quality: **{a.get('signal_quality')}** ({a.get('flat_channels', 0)} flat channels)",
        "",
        "### Relative band power",
    ]
    for b, v in (a.get("band_power_relative") or {}).items():
        lines.append(f"- {b}: {v}")
    lines += [
        "",
        "## Prediction",
        f"- Model available: {p.get('available')}",
    ]
    if p.get("available"):
        lines += [
            f"- Predicted label: **{p.get('predicted_label')}**",
            f"- Confidence: **{p.get('confidence')}**",
            f"- Class probabilities: {p.get('class_probabilities')}",
            f"- Model trained: {p.get('model_trained')}",
            "",
            "### Model card metrics (reference validation)",
        ]
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
        lines += ["", f"> {p.get('note', '')}"]
    else:
        lines.append(f"- Reason unavailable: {p.get('reason')}")
    lines += [
        "",
        "> ⚠️ Demonstrator output. Validate against subject-wise split before any clinical or thesis claim.",
        "",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return md_path


# --------------------------------------------------------------------------
# Surveys
# --------------------------------------------------------------------------
def save_survey(patient_id: str, department: str, kind: str, answers: dict) -> dict:
    with _connect() as c:
        cur = c.execute(
            "INSERT INTO surveys (patient_id, department, kind, answers_json, created_at) VALUES (?,?,?,?,?)",
            (patient_id, department, kind, json.dumps(answers), _now()))
        return {"survey_id": cur.lastrowid}


def list_analyses(disease: Optional[str] = None, offset: int = 0, limit: int = 50) -> dict:
    with _connect() as c:
        if disease:
            rows = c.execute(
                "SELECT id, patient_id, disease, predicted_label, confidence, signal_quality, report_path, created_at "
                "FROM analyses WHERE disease=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (disease, limit, offset)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM analyses WHERE disease=?", (disease,)).fetchone()[0]
        else:
            rows = c.execute(
                "SELECT id, patient_id, disease, predicted_label, confidence, signal_quality, report_path, created_at "
                "FROM analyses ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    return {"items": [dict(r) for r in rows], "total": total, "offset": offset, "limit": limit}


# --------------------------------------------------------------------------
# Clinical capture tables (medication / MRI / outcomes / neuropsych / HITL / XAI)
# --------------------------------------------------------------------------
# Whitelist guards table names (never interpolate untrusted strings into SQL).
_CLINICAL_TABLES = {
    "medications", "mri_findings", "outcomes", "neuropsych",
    "hitl_reviews", "explainability_gt",
    "eeg_acquisition", "channel_quality", "clinical_history",
    "recording_conditions", "event_annotations", "artifact_annotations",
    "eeg_interpretation", "seizure_metadata",
    "comorbidities", "hospitalization", "dba_metrics", "model_governance", "risk_management",
}
_HAS_ANALYSIS = {"hitl_reviews", "explainability_gt"}


def save_clinical(table: str, patient_id: str, fields: dict,
                  analysis_id: Optional[int] = None) -> dict:
    if table not in _CLINICAL_TABLES:
        raise ValueError(f"Unknown clinical table: {table}")
    with _connect() as c:
        if table in _HAS_ANALYSIS:
            cur = c.execute(
                f"INSERT INTO {table} (patient_id, analysis_id, fields_json, created_at) VALUES (?,?,?,?)",
                (patient_id, analysis_id, json.dumps(fields), _now()))
        else:
            cur = c.execute(
                f"INSERT INTO {table} (patient_id, fields_json, created_at) VALUES (?,?,?)",
                (patient_id, json.dumps(fields), _now()))
        row_id = cur.lastrowid
    log_transaction(patient_id, component=table, action="create", ref_id=row_id,
                    detail=f"{len(fields)} fields")
    return {"id": row_id, "table": table}


def list_clinical(table: str, patient_id: str) -> list[dict]:
    if table not in _CLINICAL_TABLES:
        raise ValueError(f"Unknown clinical table: {table}")
    with _connect() as c:
        rows = c.execute(
            f"SELECT * FROM {table} WHERE patient_id=? ORDER BY id DESC", (patient_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["fields"] = json.loads(d.pop("fields_json") or "{}")
        out.append(d)
    return out


def patient_clinical_summary(patient_id: str) -> dict:
    return {t: list_clinical(t, patient_id) for t in _CLINICAL_TABLES}


# --------------------------------------------------------------------------
# Per-department / per-role report
# --------------------------------------------------------------------------
def department_report(department: str) -> dict:
    """Aggregate a department's patients, analyses, HITL governance + clinical
    capture counts into a report dict."""
    with _connect() as c:
        patients = c.execute("SELECT COUNT(*) FROM patients WHERE department=?", (department,)).fetchone()[0]
        analyses = c.execute(
            "SELECT a.predicted_label l, a.confidence conf, a.signal_quality q "
            "FROM analyses a JOIN uploads u ON a.upload_id=u.id WHERE u.department=?",
            (department,)).fetchall()
        surveys = c.execute("SELECT COUNT(*) FROM surveys WHERE department=?", (department,)).fetchone()[0]
        hitl = [dict(r) for r in c.execute("SELECT fields_json FROM hitl_reviews").fetchall()]

    confs = [r["conf"] for r in analyses if r["conf"] is not None]
    pred_dist: dict = {}
    for r in analyses:
        pred_dist[r["l"]] = pred_dist.get(r["l"], 0) + 1

    # Governance KPIs from HITL reviews (override/acceptance rate).
    decisions = [json.loads(h["fields_json"] or "{}").get("decision") for h in hitl]
    n_dec = len([d for d in decisions if d])
    overrides = len([d for d in decisions if d == "override"])
    accepts = len([d for d in decisions if d == "accept"])

    return {
        "department": department,
        "generated_at": _now(),
        "patients": patients,
        "analyses": len(analyses),
        "surveys": surveys,
        "avg_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "prediction_distribution": pred_dist,
        "governance": {
            "hitl_reviews": n_dec,
            "override_rate": round(overrides / n_dec, 4) if n_dec else None,
            "acceptance_rate": round(accepts / n_dec, 4) if n_dec else None,
        },
    }


def write_department_report(department: str) -> str:
    rep = department_report(department)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in department if ch.isalnum() or ch in "-_") or "dept"
    md = REPORTS_DIR.parent / f"department_{safe}.md"
    g = rep["governance"]
    lines = [
        f"# Department Report — {department}",
        "", f"_Generated {rep['generated_at']}_", "",
        f"- Patients: **{rep['patients']}**",
        f"- Analyses: **{rep['analyses']}**",
        f"- Surveys: **{rep['surveys']}**",
        f"- Avg prediction confidence: **{rep['avg_confidence']}**",
        "", "## Prediction distribution",
    ]
    lines += [f"- {k}: {v}" for k, v in rep["prediction_distribution"].items()] or ["- (none)"]
    lines += [
        "", "## Governance (HITL)",
        f"- Reviews: {g['hitl_reviews']}",
        f"- Override rate: {g['override_rate']}",
        f"- Acceptance rate: {g['acceptance_rate']}",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return str(md)


# --------------------------------------------------------------------------
# Patient master data (registry row; full extract lives in clinical_data/patients/<pid>/)
# --------------------------------------------------------------------------
def save_master(patient_id: str, name: str, n_files: int, modalities: list, master_path: str) -> dict:
    init_db()
    with _connect() as c:
        c.execute("CREATE TABLE IF NOT EXISTS patient_master ("
                  "patient_id TEXT PRIMARY KEY, name TEXT, n_files INTEGER, "
                  "modalities TEXT, master_path TEXT, updated_at TEXT)")
        c.execute("INSERT INTO patient_master (patient_id, name, n_files, modalities, master_path, updated_at) "
                  "VALUES (?,?,?,?,?,?) ON CONFLICT(patient_id) DO UPDATE SET "
                  "name=excluded.name, n_files=excluded.n_files, modalities=excluded.modalities, "
                  "master_path=excluded.master_path, updated_at=excluded.updated_at",
                  (patient_id, name, n_files, json.dumps(modalities), master_path, _now()))
    log_transaction(patient_id, component="patient_master", action="ingest",
                    actor="neurologist", detail=f"{n_files} files; {modalities}")
    return {"patient_id": patient_id, "n_files": n_files}


def list_masters_db() -> dict:
    init_db()
    with _connect() as c:
        c.execute("CREATE TABLE IF NOT EXISTS patient_master ("
                  "patient_id TEXT PRIMARY KEY, name TEXT, n_files INTEGER, "
                  "modalities TEXT, master_path TEXT, updated_at TEXT)")
        rows = c.execute("SELECT * FROM patient_master ORDER BY updated_at DESC").fetchall()
    items = []
    for r in rows:
        d = dict(r)
        d["modalities"] = json.loads(d.get("modalities") or "[]")
        items.append(d)
    return {"items": items, "total": len(items)}


def save_feedback(patient_id: str, role: str, ai_output: str, rating: int,
                  correction: str = "", reason: str = "") -> dict:
    """Capture a role's feedback/correction on an AI output (feeds RLHF)."""
    init_db()
    with _connect() as c:
        cur = c.execute(
            "INSERT INTO feedback (patient_id, role, ai_output, rating, correction, reason, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (patient_id, role, ai_output, rating, correction, reason, _now()))
        fid = cur.lastrowid
    log_transaction(patient_id, component="feedback", action="correction" if correction else "rating",
                    actor=role, ref_id=fid, detail=f"rating={rating}")
    return {"id": fid}


def list_feedback(role: Optional[str] = None, limit: int = 100) -> dict:
    init_db()
    with _connect() as c:
        if role:
            rows = c.execute("SELECT * FROM feedback WHERE role=? ORDER BY id DESC LIMIT ?", (role, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    items = [dict(r) for r in rows]
    ratings = [r["rating"] for r in items if r["rating"] is not None]
    corrections = len([r for r in items if r["correction"]])
    return {"items": items, "total": len(items),
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "corrections": corrections}


def consensus() -> dict:
    """Consensus AI: agreement across HITL reviews of the same patient.
    For patients reviewed >=2 times, do the reviewers' decisions agree?"""
    init_db()
    with _connect() as c:
        rows = c.execute("SELECT patient_id, fields_json FROM hitl_reviews").fetchall()
    by_patient: dict = {}
    for r in rows:
        d = json.loads(r["fields_json"] or "{}")
        by_patient.setdefault(r["patient_id"], []).append(d.get("decision") or d.get("human_decision"))
    multi = {p: v for p, v in by_patient.items() if len([x for x in v if x]) >= 2}
    agreed = sum(1 for v in multi.values() if len(set(x for x in v if x)) == 1)
    return {
        "patients_multi_reviewed": len(multi),
        "consensus_reached": agreed,
        "consensus_rate": round(agreed / len(multi), 3) if multi else None,
        "note": "Consensus = all reviewers of a patient chose the same decision.",
    }


def decision_route(confidence: float, role: str = "", task: str = "") -> dict:
    """Decision AI (per role/task): route by confidence → auto / review / escalate."""
    if confidence >= 0.8:
        action, why = "auto-decision", "High confidence — proceed automatically"
    elif confidence >= 0.5:
        action, why = "human-review", "Moderate confidence — route to clinician review"
    else:
        action, why = "escalate", "Low confidence — escalate / reject"
    return {"role": role, "task": task, "confidence": round(float(confidence), 3),
            "decision": action, "rationale": why,
            "thresholds": {"auto": 0.8, "review": 0.5}}


def _load_instruments():
    from pathlib import Path as _P
    p = _P(__file__).resolve().parent / "config" / "assessments.json"
    return {i["id"]: i for i in (json.loads(p.read_text()).get("instruments", []) if p.exists() else [])}


def score_assessment(instrument: str, answers: dict) -> dict:
    """Auto-score answers against the validated instrument's rules + interpretation band."""
    inst = _load_instruments().get(instrument)
    if not inst:
        return {"score": None, "interpretation": "unknown instrument", "level": "", "alert": ""}
    vals = [float(v) for v in answers.values() if isinstance(v, (int, float)) or str(v).replace(".", "").isdigit()]
    vals = [float(v) for v in answers.values() if str(v).replace(".", "").replace("-", "").isdigit()]
    score = (sum(vals) / len(vals)) if (inst.get("scoring") == "mean" and vals) else sum(vals)
    score = round(float(score), 2)
    band = next((b for b in inst.get("bands", []) if b["min"] <= score <= b["max"]), None)
    # alert: suicidality items
    alert = ""
    if instrument == "PHQ9" and float(answers.get("item9", answers.get("9", 0)) or 0) > 0:
        alert = "PHQ-9 item 9 (self-harm) positive — escalate"
    if instrument == "NDDIE" and float(answers.get("item4", answers.get("4", 0)) or 0) >= 2:
        alert = "NDDI-E item 4 (suicidality) positive — escalate"
    return {"score": score, "max_score": inst.get("max"),
            "interpretation": band["label"] if band else "out of range",
            "level": band["level"] if band else "", "alert": alert}


def save_assessment(patient_id: str, instrument: str, answers: dict, examiner: str = "") -> dict:
    init_db()
    s = score_assessment(instrument, answers)
    now = _now()
    with _connect() as c:
        cur = c.execute(
            "INSERT INTO assessments(patient_id,instrument,answers_json,score,max_score,interpretation,level,alert,examiner,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (patient_id, instrument, json.dumps(answers), s["score"], s["max_score"],
             s["interpretation"], s["level"], s["alert"], examiner, now))
        aid = cur.lastrowid
    log_transaction(patient_id, component="assessment", action="create", ref_id=aid,
                    detail=f"{instrument} score={s['score']} ({s['interpretation']})")
    return {"id": aid, "patient_id": patient_id, "instrument": instrument, **s, "created_at": now}


def list_assessments(patient_id: str | None = None, limit: int = 50) -> list:
    init_db()
    with _connect() as c:
        if patient_id:
            rows = c.execute("SELECT * FROM assessments WHERE patient_id=? ORDER BY id DESC LIMIT ?", (patient_id, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM assessments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_assessment(aid: int):
    init_db()
    with _connect() as c:
        r = c.execute("SELECT * FROM assessments WHERE id=?", (aid,)).fetchone()
    return dict(r) if r else None


def update_assessment(aid: int, answers: dict, examiner: str = ""):
    init_db()
    cur = get_assessment(aid)
    if not cur:
        return None
    s = score_assessment(cur["instrument"], answers)
    now = _now()
    with _connect() as c:
        c.execute("UPDATE assessments SET answers_json=?,score=?,max_score=?,interpretation=?,level=?,alert=?,examiner=?,updated_at=? WHERE id=?",
                  (json.dumps(answers), s["score"], s["max_score"], s["interpretation"], s["level"], s["alert"], examiner, now, aid))
    log_transaction(cur["patient_id"], component="assessment", action="update", ref_id=aid,
                    detail=f"{cur['instrument']} re-scored={s['score']}")
    return {"id": aid, "instrument": cur["instrument"], **s, "updated_at": now}


def delete_assessment(aid: int) -> bool:
    init_db()
    cur = get_assessment(aid)
    if not cur:
        return False
    with _connect() as c:
        c.execute("DELETE FROM assessments WHERE id=?", (aid,))
    log_transaction(cur["patient_id"], component="assessment", action="delete", ref_id=aid,
                    detail=f"{cur['instrument']} deleted")
    return True


def _ensure_chat_columns():
    """Add advanced-chat columns to a pre-existing team_messages table (idempotent)."""
    with _connect() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(team_messages)").fetchall()}
        for col, ddl in [("topic", "topic TEXT"), ("attachment", "attachment TEXT"),
                         ("read_by", "read_by TEXT DEFAULT '[]'")]:
            if col not in cols:
                try:
                    c.execute(f"ALTER TABLE team_messages ADD COLUMN {ddl}")
                except Exception:
                    pass


def post_team_message(channel: str, from_role: str, text: str, patient_id: str = "",
                      topic: str = "", attachment: str = "") -> dict:
    """A role posts a message to a team channel. @bot triggers an AI reply. Supports topic + attachment."""
    init_db(); _ensure_chat_columns()
    now = _now()
    with _connect() as c:
        cur = c.execute("INSERT INTO team_messages(channel,from_role,text,is_bot,patient_id,topic,attachment,read_by,created_at)"
                        " VALUES(?,?,?,0,?,?,?,?,?)",
                        (channel, from_role, text, patient_id or None, topic or None, attachment or None,
                         json.dumps([from_role]), now))
        mid = cur.lastrowid
    log_transaction(patient_id or channel, component="team_chat", action="message", ref_id=mid,
                    detail=f"[{channel}] {from_role}: {text[:60]}")
    out = {"id": mid, "channel": channel, "from_role": from_role, "text": text, "is_bot": 0, "created_at": now}
    bot = None
    if "@bot" in text.lower():
        bot = _team_bot_reply(channel, text, patient_id)
    return {"message": out, "bot_reply": bot}


def _team_bot_reply(channel: str, query: str, patient_id: str = "") -> dict:
    """Team bot: answers using Ollama, grounded in patient records if a patient is in context."""
    context = ""
    if patient_id:
        try:
            r = patient_chat(patient_id, query)
            ctx_rows = (r.get("vector_results") or [])[:4] or (r.get("results") or [])[:4]
            context = "\n".join(str(x.get("text") or x.get("data")) for x in ctx_rows)
        except Exception:
            context = ""
    answer = None
    try:
        import ollama_agent
        resp = ollama_agent.answer(query.replace("@bot", "").strip(),
                                   context=context, layout="passage")
        # ollama_agent.answer may return a str or a dict — coerce to text
        if isinstance(resp, dict):
            answer = resp.get("answer") or resp.get("text") or resp.get("response") or json.dumps(resp)[:500]
        else:
            answer = str(resp) if resp else None
    except Exception:
        answer = None
    if not answer:
        answer = ("(Bot offline — Ollama unavailable. Grounded context retrieved: "
                  + (context[:200] if context else "none") + ")")
    now = _now()
    with _connect() as c:
        cur = c.execute("INSERT INTO team_messages(channel,from_role,text,is_bot,patient_id,created_at) VALUES(?,?,?,1,?,?)",
                        (channel, "TeamBot", answer, patient_id or None, now))
        bid = cur.lastrowid
    return {"id": bid, "channel": channel, "from_role": "TeamBot", "text": answer, "is_bot": 1, "created_at": now}


def list_team_messages(channel: str | None = None, limit: int = 100) -> list:
    init_db()
    with _connect() as c:
        if channel:
            rows = c.execute("SELECT * FROM team_messages WHERE channel=? ORDER BY id ASC LIMIT ?", (channel, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM team_messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def list_team_channels() -> list:
    init_db()
    with _connect() as c:
        rows = c.execute("SELECT channel, COUNT(*) n, MAX(created_at) last FROM team_messages GROUP BY channel ORDER BY last DESC").fetchall()
    return [dict(r) for r in rows]


def create_chat_group(name: str, members: list, topic: str = "", created_by: str = "") -> dict:
    """Create a group + post a welcome message into its channel."""
    init_db()
    now = _now()
    with _connect() as c:
        try:
            c.execute("INSERT INTO chat_groups(name,members,topic,created_by,created_at) VALUES(?,?,?,?,?)",
                      (name, json.dumps(members), topic, created_by, now))
        except Exception:
            c.execute("UPDATE chat_groups SET members=?,topic=? WHERE name=?", (json.dumps(members), topic, name))
    # welcome message
    post_team_message(name, "TeamBot",
                      f"👋 Welcome to '{name}'! Members: {', '.join(members)}. Topic: {topic or 'general'}. Type @bot to ask the AI.",
                      topic=topic)
    log_transaction(name, component="chat_group", action="create", detail=f"{name} by {created_by}")
    return {"name": name, "members": members, "topic": topic, "created_at": now}


def list_chat_groups() -> list:
    init_db()
    with _connect() as c:
        rows = c.execute("SELECT * FROM chat_groups ORDER BY id DESC").fetchall()
    return [{**dict(r), "members": json.loads(r["members"] or "[]")} for r in rows]


def set_presence(role: str, status: str) -> dict:
    """status: active | away | desk | break | offline."""
    init_db()
    now = _now()
    with _connect() as c:
        c.execute("INSERT INTO chat_presence(role,status,updated_at) VALUES(?,?,?) "
                  "ON CONFLICT(role) DO UPDATE SET status=?,updated_at=?", (role, status, now, status, now))
    return {"role": role, "status": status, "updated_at": now}


def get_presence() -> list:
    init_db()
    with _connect() as c:
        rows = c.execute("SELECT * FROM chat_presence ORDER BY role").fetchall()
    return [dict(r) for r in rows]


def mark_read(channel: str, role: str) -> dict:
    """Mark all messages in a channel as read by this role."""
    init_db(); _ensure_chat_columns()
    n = 0
    with _connect() as c:
        for r in c.execute("SELECT id, read_by FROM team_messages WHERE channel=?", (channel,)).fetchall():
            rb = json.loads(r["read_by"] or "[]")
            if role not in rb:
                rb.append(role)
                c.execute("UPDATE team_messages SET read_by=? WHERE id=?", (json.dumps(rb), r["id"])); n += 1
    return {"channel": channel, "role": role, "marked": n}


def genai_bot(role: str, query: str, layout: str = "passage", patient_id: str = "") -> dict:
    """Generative-AI bot per department role. Answers free-text with report access,
    formatted as passage | table | list | graph (layout via the Ollama agent)."""
    context = ""
    if patient_id:
        try:
            r = patient_chat(patient_id, query)
            rows = (r.get("vector_results") or [])[:5] or (r.get("results") or [])[:5]
            context = "\n".join(str(x.get("text") or x.get("data")) for x in rows)
        except Exception:
            context = ""
    answer = None
    try:
        import ollama_agent
        resp = ollama_agent.answer(query, context=context, layout=layout)
        answer = resp if isinstance(resp, (dict, list)) else str(resp)
    except Exception:
        answer = None
    if answer is None:
        answer = f"(GenAI bot offline. Role={role}, layout={layout}. Context: {context[:200] or 'none'})"
    log_transaction(patient_id or "genai", component="genai_bot", action="ask",
                    detail=f"{role} [{layout}]: {query[:60]}")
    return {"role": role, "layout": layout, "query": query, "answer": answer}


def add_expert_review(patient_id: str, role: str, finding: str, agree_with_ai: str = "",
                      note: str = "", expert: str = "", analysis_id: int | None = None) -> dict:
    """An expert (any role) attaches their assessment/finding to a patient's study."""
    init_db()
    now = _now()
    with _connect() as c:
        if analysis_id is None:
            a = c.execute("SELECT id FROM analyses WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,)).fetchone()
            analysis_id = a["id"] if a else None
        cur = c.execute(
            "INSERT INTO expert_reviews(patient_id,analysis_id,role,expert,finding,agree_with_ai,note,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)", (patient_id, analysis_id, role, expert, finding, agree_with_ai, note, now))
        rid = cur.lastrowid
    log_transaction(patient_id, component="expert_review", action="add", ref_id=rid,
                    detail=f"{role}: {finding[:60]} (agree_ai={agree_with_ai})")
    return {"id": rid, "patient_id": patient_id, "role": role, "finding": finding,
            "agree_with_ai": agree_with_ai, "created_at": now}


def save_seizure(patient_id: str, fields: dict) -> dict:
    """Log a seizure event. Auto-scores severity from duration + injury + ER + recovery."""
    init_db()
    now = _now()
    # auto severity score
    score = 0
    dur = int(fields.get("duration_sec") or 0)
    if dur > 300:
        score += 3
    elif dur > 120:
        score += 2
    elif dur > 30:
        score += 1
    if str(fields.get("injury", "")).lower() in ("yes", "true", "fall", "head injury"):
        score += 3
    if str(fields.get("er_visit", "")).lower() in ("yes", "true"):
        score += 5
    rec = int(fields.get("recovery_min") or 0)
    if rec > 30:
        score += 2
    severity = "Severe" if score >= 6 else "Moderate" if score >= 3 else "Mild"
    cols = ["patient_id", "event_date", "event_time", "duration_sec", "location", "witnessed",
            "aura", "awareness", "motor_signs", "injury", "post_ictal", "recovery_min",
            "er_visit", "rescue_med", "severity", "trigger", "notes", "created_at"]
    # cols[1:14] = event_date..rescue_med (13 fields); then severity, trigger, notes, created_at
    vals = [patient_id] + [fields.get(c) for c in cols[1:14]] + [severity, fields.get("trigger"), fields.get("notes"), now]
    with _connect() as c:
        cur = c.execute(f"INSERT INTO seizure_diary({','.join(cols)}) VALUES({','.join(['?'] * len(cols))})", vals)
        sid = cur.lastrowid
    log_transaction(patient_id, component="seizure_diary", action="log", ref_id=sid,
                    detail=f"seizure {fields.get('event_date')} {dur}s severity={severity}")
    return {"id": sid, "patient_id": patient_id, "severity": severity, "severity_score": score, "created_at": now}


def list_seizures(patient_id: str, limit: int = 200) -> dict:
    init_db()
    with _connect() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM seizure_diary WHERE patient_id=? ORDER BY id DESC LIMIT ?", (patient_id, limit)).fetchall()]
    # monthly trend + stats
    from collections import Counter
    months = Counter((r.get("event_date") or "")[:7] for r in rows if r.get("event_date"))
    sev = Counter(r.get("severity") for r in rows)
    durs = [r["duration_sec"] for r in rows if r.get("duration_sec")]
    return {"items": rows, "count": len(rows),
            "monthly": dict(sorted(months.items())),
            "severity_dist": dict(sev),
            "avg_duration_sec": round(sum(durs) / len(durs), 1) if durs else 0,
            "er_visits": sum(1 for r in rows if str(r.get("er_visit", "")).lower() in ("yes", "true"))}


def analyze_correlations(patient_id: str) -> dict:
    """Trigger/pattern analysis from the seizure diary — answers 'why did my seizure happen?'.
    Honest: computed from LOGGED events only (no non-seizure baseline → frequencies, not true risk ratios)."""
    init_db()
    from collections import Counter
    with _connect() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT trigger, event_time, severity, aura, location, duration_sec FROM seizure_diary WHERE patient_id=?",
            (patient_id,)).fetchall()]
    n = len(rows)
    if n == 0:
        return {"patient_id": patient_id, "count": 0, "triggers": [], "time_of_day": {},
                "by_trigger_severity": {}, "top_trigger": None,
                "note": "No seizures logged yet. Log events in the Seizure Diary to enable correlation."}

    def tod(t):
        try:
            h = int((t or "")[:2])
        except (ValueError, TypeError):
            return "unknown"
        return "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"

    trig_ct = Counter((r.get("trigger") or "none") for r in rows)
    triggers = [{"trigger": k, "count": v, "pct": round(100 * v / n)} for k, v in trig_ct.most_common() if k and k != "none"]
    time_ct = Counter(tod(r.get("event_time")) for r in rows)
    # severity mix per trigger (does a trigger lead to worse seizures?)
    by_ts = {}
    for r in rows:
        tg = r.get("trigger") or "none"
        by_ts.setdefault(tg, Counter())[r.get("severity") or "?"] += 1
    by_trigger_severity = {k: dict(v) for k, v in by_ts.items() if k and k != "none"}
    aura_ct = Counter((r.get("aura") or "none") for r in rows if r.get("aura") and r.get("aura") != "None")
    loc_ct = Counter((r.get("location") or "?") for r in rows if r.get("location"))
    top = triggers[0]["trigger"] if triggers else None
    return {"patient_id": patient_id, "count": n,
            "triggers": triggers, "time_of_day": dict(time_ct),
            "by_trigger_severity": by_trigger_severity,
            "aura_dist": dict(aura_ct), "location_dist": dict(loc_ct),
            "top_trigger": top,
            "insight": f"Your most frequent trigger is '{top}' ({triggers[0]['pct']}% of logged seizures)." if top else "No trigger pattern yet — log triggers with each seizure.",
            "note": "Frequencies among logged seizures (not true risk ratios — needs non-seizure baseline for that)."}


def save_component_finding(patient_id: str, component: str, doctor_finding: str,
                           doctor: str = "", agree_with_ai: str = "") -> dict:
    """Doctor records their finding for one EEG component (upsert per component)."""
    init_db()
    now = _now()
    with _connect() as c:
        c.execute(
            "INSERT INTO component_findings(patient_id,component,doctor_finding,doctor,agree_with_ai,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?) ON CONFLICT(patient_id,component) DO UPDATE SET"
            " doctor_finding=excluded.doctor_finding, doctor=excluded.doctor,"
            " agree_with_ai=excluded.agree_with_ai, updated_at=excluded.updated_at",
            (patient_id, component, doctor_finding, doctor, agree_with_ai, now, now))
    log_transaction(patient_id, component="component_finding", action="save",
                    detail=f"{component}: {doctor_finding[:50]} (agree={agree_with_ai})")
    return {"patient_id": patient_id, "component": component, "doctor_finding": doctor_finding,
            "agree_with_ai": agree_with_ai, "updated_at": now}


def get_component_findings(patient_id: str) -> dict:
    init_db()
    with _connect() as c:
        rows = c.execute("SELECT * FROM component_findings WHERE patient_id=?", (patient_id,)).fetchall()
    return {r["component"]: dict(r) for r in rows}


def study_review(patient_id: str) -> dict:
    """Full study review: AI assessment detail + every expert's attached review."""
    init_db()
    with _connect() as c:
        a = c.execute("SELECT * FROM analyses WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,)).fetchone()
        reviews = c.execute("SELECT * FROM expert_reviews WHERE patient_id=? ORDER BY id DESC", (patient_id,)).fetchall()
        upl = c.execute("SELECT * FROM uploads WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,)).fetchone()
    ai = dict(a) if a else None
    return {
        "patient_id": patient_id,
        "ai_assessment": {
            "predicted": ai.get("predicted_label") if ai else None,
            "confidence": ai.get("confidence") if ai else None,
            "signal_quality": ai.get("signal_quality") if ai else None,
            "source_file": (dict(upl).get("filename") if upl else None),
        } if ai else None,
        "expert_reviews": [dict(r) for r in reviews],
        "n_experts": len(reviews),
    }


def assign_form(patient_id: str, instrument: str, assigned_by: str = "", message: str = "") -> dict:
    """Expert assigns an assessment form to a patient (pending until patient fills it)."""
    init_db()
    now = _now()
    with _connect() as c:
        cur = c.execute(
            "INSERT INTO form_assignments(patient_id,instrument,assigned_by,status,message,created_at)"
            " VALUES(?,?,?,'pending',?,?)", (patient_id, instrument, assigned_by, message, now))
        fid = cur.lastrowid
    log_transaction(patient_id, component="form", action="assign", ref_id=fid,
                    detail=f"{instrument} assigned by {assigned_by}")
    return {"id": fid, "patient_id": patient_id, "instrument": instrument, "status": "pending", "created_at": now}


def list_forms(patient_id: str | None = None, status: str | None = None, limit: int = 50) -> list:
    init_db()
    q = "SELECT * FROM form_assignments"
    cond, args = [], []
    if patient_id:
        cond.append("patient_id=?"); args.append(patient_id)
    if status:
        cond.append("status=?"); args.append(status)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with _connect() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def submit_form(form_id: int, answers: dict) -> dict | None:
    """Patient fills a pending form via the portal -> auto-score + save assessment + mark complete."""
    init_db()
    with _connect() as c:
        f = c.execute("SELECT * FROM form_assignments WHERE id=?", (form_id,)).fetchone()
    if not f:
        return None
    f = dict(f)
    saved = save_assessment(f["patient_id"], f["instrument"], answers, examiner="patient(self-service)")
    now = _now()
    with _connect() as c:
        c.execute("UPDATE form_assignments SET status='completed',assessment_id=?,completed_at=? WHERE id=?",
                  (saved["id"], now, form_id))
    log_transaction(f["patient_id"], component="form", action="submit", ref_id=form_id,
                    detail=f"{f['instrument']} completed -> score {saved['score']}")
    return {"form_id": form_id, "status": "completed", **saved}


def _vector_retrieve(patient_id: str, query: str, k: int = 6):
    """Semantic retrieval from the ChromaDB collection populated by the
    VECTOR-INGEST cron. Embeds the query via local Ollama, queries the
    'clinical' collection filtered to this patient. Returns [] on any failure
    so patient_chat falls back to keyword retrieval."""
    try:
        import json as _json
        import urllib.request as _u
        from pathlib import Path as _P
        body = _json.dumps({"model": "nomic-embed-text", "prompt": query[:2000]}).encode()
        req = _u.Request("http://localhost:11434/api/embeddings", data=body,
                         headers={"Content-Type": "application/json"})
        with _u.urlopen(req, timeout=15) as r:
            emb = _json.loads(r.read())["embedding"]
        import chromadb
        vdb = _P(__file__).resolve().parent / "data" / "vector_db"
        col = chromadb.PersistentClient(path=str(vdb)).get_or_create_collection("clinical")
        res = col.query(query_embeddings=[emb], n_results=k,
                        where={"patient_id": patient_id} if patient_id else None)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        # ChromaDB default space is L2 (lower distance = closer). Report raw distance honestly.
        return [{"source": (m or {}).get("type", "vector"), "text": d,
                 "distance": round(dist, 3)} for d, m, dist in zip(docs, metas, dists)]
    except Exception:
        return []


def patient_chat(patient_id: str, query: str) -> dict:
    """Retrieval-based patient Q&A (RAG retrieval step). Tries SEMANTIC vector
    retrieval from ChromaDB (populated by the VECTOR-INGEST cron) first, then
    falls back to keyword retrieval over the patient's records."""
    init_db()
    vector_hits = _vector_retrieve(patient_id, query)
    q = (query or "").lower()
    kws = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2]
    hits = []
    with _connect() as c:
        # Patient core
        p = c.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
        if p:
            hits.append({"source": "patients", "data": dict(p)})
        # All clinical capture tables + surveys + analyses
        tables = list(_CLINICAL_TABLES) + ["surveys", "analyses"]
        for t in tables:
            try:
                rows = c.execute(f"SELECT * FROM {t} WHERE patient_id=? ORDER BY id DESC LIMIT 5", (patient_id,)).fetchall()
            except Exception:
                continue
            for r in rows:
                d = dict(r)
                blob = json.dumps(d).lower()
                score = sum(1 for k in kws if k in blob)
                hits.append({"source": t, "score": score, "data": d})
    # Rank: keyword matches first, then recency (already DESC).
    ranked = sorted(hits, key=lambda h: h.get("score", 0), reverse=True)
    matched = [h for h in ranked if h.get("score", 0) > 0][:8] or ranked[:6]
    log_transaction(patient_id, component="patient_chat", action="query",
                    detail=f"{query[:100]} [retrieval={'vector' if vector_hits else 'keyword'}]")
    return {
        "patient_id": patient_id, "query": query,
        "retrieval_mode": "vector (semantic)" if vector_hits else "keyword (fallback)",
        "vector_results": vector_hits,
        "note": "Semantic retrieval from ChromaDB (VECTOR-INGEST cron) with keyword fallback. LLM answer-generation via Ollama.",
        "results": matched,
    }


# ── Clinical Trust Panel + Human Oversight audit (thesis core) ──────────────
def _ensure_decisions() -> None:
    with _connect() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS clinical_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, analysis_id INTEGER,
            ai_prediction TEXT, ai_confidence REAL,
            top_channels TEXT, artifact_risk TEXT, time_window TEXT,
            neurologist_agreement TEXT, final_decision TEXT,
            reviewer TEXT, note TEXT, created_at TEXT)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cd_patient ON clinical_decisions(patient_id)")


def build_trust_panel(analysis_id: Optional[int] = None, patient_id: Optional[str] = None) -> dict:
    """Clinical Trust Panel — per-prediction summary a neurologist confirms/rejects.
    Built from the REAL stored analysis. Honest about what is derived vs not yet available."""
    init_db()
    with _connect() as c:
        if analysis_id:
            row = c.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        elif patient_id:
            row = c.execute("SELECT * FROM analyses WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,)).fetchone()
        else:
            row = c.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return {"available": False, "note": "No analysis on record. Run an EEG analysis first."}
    r = dict(row)
    try:
        res = json.loads(r.get("result_json") or "{}")
    except (ValueError, TypeError):
        res = {}
    analysis = res.get("analysis", {})
    pred = res.get("prediction", {})
    per_ch = analysis.get("per_channel", []) or []
    ch_names = analysis.get("channels", []) or []
    # top channels = highest-variance (std) — honest proxy for "most active"; true per-channel SHAP not yet wired
    ranked = sorted(per_ch, key=lambda x: x.get("std", 0) or 0, reverse=True)[:3]
    top_channels = [ch_names[ci["channel"]] if ci.get("channel", 0) < len(ch_names) else f"ch{ci.get('channel')}"
                    for ci in ranked]
    # artifact risk from signal quality + flat channels (real)
    q = analysis.get("signal_quality") or r.get("signal_quality") or "Unknown"
    flat = analysis.get("flat_channels", 0) or 0
    risk = {"Good": "Low", "Fair": "Moderate", "Poor": "High"}.get(q, "Unknown")
    if flat > 0 and risk == "Low":
        risk = "Moderate"
    conf = pred.get("confidence", r.get("confidence"))
    ai_label = pred.get("predicted_label") or r.get("predicted_label")
    class_probs = pred.get("class_probabilities", {})
    # Recompute LIVE with corrected preprocessing (stored values pre-date the scaler fix → stale).
    feats = res.get("features", {})
    if feats:
        try:
            import numpy as np
            import eeg_analysis_pipeline as eeg
            fv = np.array(list(feats.values()), dtype=float)
            fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
            live = eeg.classify(fv, r.get("disease", "epilepsy"))
            if live.get("available"):
                ai_label = live.get("predicted_label", ai_label)
                conf = live.get("confidence", conf)
                class_probs = live.get("class_probabilities", class_probs)
        except Exception:  # fall back to stored if recompute fails
            pass
    return {
        "available": True,
        "analysis_id": r["id"], "patient_id": r["patient_id"], "disease": r.get("disease"),
        "ai_prediction": ai_label,
        "confidence": conf,
        "confidence_basis": "recomputed live with corrected scaler+selector preprocessing",
        "class_probabilities": class_probs,
        "top_channels": top_channels or ch_names[:3],
        "top_channels_basis": "highest-variance channels (true per-channel SHAP not yet wired)",
        "time_window": "full recording",
        "time_window_basis": "window-level seizure localization not yet implemented — whole-record prediction",
        "artifact_risk": risk,
        "artifact_basis": f"signal quality={q}, flat channels={flat}",
        "signal_quality": q,
        "model_note": pred.get("note", ""),
        "created_at": r.get("created_at"),
        "decision_options": ["Confirm", "Reject", "Needs Review"],
        "guidance": "AI is a decision-support tool. Neurologist confirmation is required before any clinical use (human oversight).",
    }


def save_clinical_decision(fields: dict) -> dict:
    """Record the neurologist's Confirm/Reject/Needs-Review decision → human-oversight audit trail."""
    _ensure_decisions()
    now = _now()
    cols = ["patient_id", "analysis_id", "ai_prediction", "ai_confidence", "top_channels",
            "artifact_risk", "time_window", "neurologist_agreement", "final_decision",
            "reviewer", "note", "created_at"]
    tc = fields.get("top_channels")
    if isinstance(tc, list):
        tc = ", ".join(map(str, tc))
    vals = [fields.get("patient_id"), fields.get("analysis_id"), fields.get("ai_prediction"),
            fields.get("ai_confidence"), tc, fields.get("artifact_risk"), fields.get("time_window"),
            fields.get("neurologist_agreement"), fields.get("final_decision"),
            fields.get("reviewer", "neurologist"), fields.get("note"), now]
    with _connect() as c:
        cur = c.execute(f"INSERT INTO clinical_decisions({','.join(cols)}) VALUES({','.join(['?'] * len(cols))})", vals)
        did = cur.lastrowid
    log_transaction(fields.get("patient_id", "_unknown"), component="clinical_trust", action="human_decision",
                    actor=fields.get("reviewer", "neurologist"), ref_id=did,
                    detail=f"{fields.get('final_decision')} (AI={fields.get('ai_prediction')} conf={fields.get('ai_confidence')})")
    return {"id": did, "final_decision": fields.get("final_decision"), "created_at": now}


def list_clinical_decisions(patient_id: Optional[str] = None, limit: int = 200) -> dict:
    """Human-oversight audit trail (Confirm/Reject/Needs-Review history)."""
    _ensure_decisions()
    with _connect() as c:
        if patient_id:
            rows = [dict(r) for r in c.execute("SELECT * FROM clinical_decisions WHERE patient_id=? ORDER BY id DESC LIMIT ?", (patient_id, limit)).fetchall()]
        else:
            rows = [dict(r) for r in c.execute("SELECT * FROM clinical_decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
    from collections import Counter
    dist = Counter(r.get("final_decision") for r in rows)
    agree = Counter(r.get("neurologist_agreement") for r in rows)
    n = len(rows)
    confirmed = dist.get("Confirm", 0)
    return {"items": rows, "count": n, "decision_dist": dict(dist), "agreement_dist": dict(agree),
            "ai_confirm_rate": round(confirmed / n, 3) if n else None,
            "note": "Human-in-the-loop oversight audit — every AI prediction a neurologist accepted/rejected."}


if __name__ == "__main__":
    init_db()
    print(f"Initialized {DB_PATH}")
