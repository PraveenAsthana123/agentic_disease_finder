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


if __name__ == "__main__":
    init_db()
    print(f"Initialized {DB_PATH}")
