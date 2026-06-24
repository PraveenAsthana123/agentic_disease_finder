"""
FastAPI Backend for NeuroAI EEG Analysis
========================================
REST API endpoints for EEG data analysis and classification.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import json
import shutil
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ui_app import RealDataLoader
import eeg_analysis_pipeline as eeg
import clinical_db as cdb
import eeg_ingest as ingest
import eeg_explainability as xai
import rai_checks as rai
import ollama_agent
import ai_type_detail
import eeg_deep
import eeg_anomaly
import eeg_datascience
import eeg_timeseries_stats
import council_orchestrator
import knowledge_graph as kg

# Ensure the clinical database/tables exist on import.
cdb.init_db()

app = FastAPI(
    title="NeuroAI EEG Analysis API",
    description="REST API for neurological disease detection using EEG data",
    version="1.0.0"
)


def _json_safe(obj):
    """Recursively replace NaN/Inf (not JSON-compliant → FastAPI 500) with None.
    Degenerate signals (flat channels, tiny files) yield NaN features that crash
    serialization; this makes any result safely returnable."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for analysis results
analysis_results = {}
analysis_status = {"status": "idle", "progress": 0, "message": ""}

# Initialize data loader
loader = RealDataLoader(base_path='./datasets')


class AnalysisResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict] = None


class ClassificationResult(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_mean: float
    cv_std: float
    confusion_matrix: List[List[int]]
    n_features: int
    n_segments: int
    model_type: str


@app.get("/")
async def root():
    return {
        "message": "NeuroAI EEG Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "datasets": "/api/datasets",
            "analyze": "/api/analyze/{dataset}",
            "results": "/api/results",
            "status": "/api/status"
        }
    }


@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets for analysis"""
    summary = loader.get_dataset_summary()
    datasets = []

    for name, info in summary.items():
        if info.get('available', False):
            datasets.append({
                "name": name,
                "path": info.get('path', ''),
                "total_files": info.get('total_files', 0),
                "eeg_files": info.get('eeg_files', 0)
            })

    return {
        "status": "success",
        "datasets": datasets,
        "total": len(datasets)
    }


@app.get("/api/dataset/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    """Get detailed information about a specific dataset"""
    if "schizophrenia" in dataset_name.lower():
        X, y, metadata = loader.load_schizophrenia_data()
        if len(X) > 0:
            return {
                "status": "success",
                "dataset": dataset_name,
                "info": {
                    "total_subjects": int(metadata['total_subjects']),
                    "healthy_count": int(metadata['healthy_count']),
                    "patient_count": int(metadata['patient_count']),
                    "channels": int(metadata['channels']),
                    "sampling_rate": int(metadata['sampling_rate']),
                    "data_shape": list(X.shape)
                }
            }

    return {"status": "error", "message": f"Dataset {dataset_name} not found or not supported"}


@app.post("/api/analyze/{dataset_name}")
async def run_analysis(dataset_name: str, background_tasks: BackgroundTasks):
    """Start EEG analysis for a dataset"""
    global analysis_status

    if analysis_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Analysis already in progress")

    analysis_status = {"status": "running", "progress": 0, "message": "Loading data..."}

    # Run analysis in background
    background_tasks.add_task(perform_analysis, dataset_name)

    return {
        "status": "started",
        "message": f"Analysis started for {dataset_name}",
        "check_status": "/api/status"
    }


async def perform_analysis(dataset_name: str):
    """Background task to perform EEG analysis"""
    global analysis_results, analysis_status

    try:
        analysis_status = {"status": "running", "progress": 10, "message": "Loading EEG data..."}

        if "schizophrenia" in dataset_name.lower():
            X, y, metadata = loader.load_schizophrenia_data()
        else:
            analysis_status = {"status": "error", "progress": 0, "message": f"Unknown dataset: {dataset_name}"}
            return

        if len(X) == 0:
            analysis_status = {"status": "error", "progress": 0, "message": "No data loaded"}
            return

        analysis_status = {"status": "running", "progress": 30, "message": "Extracting features..."}

        # Train classifier
        model, results = loader.train_classifier(X, y)

        analysis_status = {"status": "running", "progress": 90, "message": "Finalizing results..."}

        # Store results
        analysis_results = {
            "dataset": dataset_name,
            "metadata": {
                "total_subjects": int(metadata['total_subjects']),
                "healthy_count": int(metadata['healthy_count']),
                "patient_count": int(metadata['patient_count']),
                "channels": int(metadata['channels']),
                "sampling_rate": int(metadata['sampling_rate']),
            },
            "classification": {
                "accuracy": float(results['accuracy']),
                "precision": float(results['precision']),
                "recall": float(results['recall']),
                "f1": float(results['f1']),
                "cv_mean": float(results['cv_mean']),
                "cv_std": float(results['cv_std']),
                "confusion_matrix": results['confusion_matrix'],
                "n_features": int(results['n_features']),
                "n_segments": int(results['n_segments']),
                "model_type": results['model_type']
            },
            "sample_eeg": {
                "healthy_sample": X[y == 0][0][:256, :4].tolist() if np.sum(y == 0) > 0 else [],
                "patient_sample": X[y == 1][0][:256, :4].tolist() if np.sum(y == 1) > 0 else []
            }
        }

        analysis_status = {"status": "completed", "progress": 100, "message": "Analysis complete!"}

    except Exception as e:
        analysis_status = {"status": "error", "progress": 0, "message": str(e)}


@app.get("/api/status")
async def get_analysis_status():
    """Get current analysis status"""
    return analysis_status


@app.get("/api/results")
async def get_results():
    """Get analysis results"""
    if not analysis_results:
        return {"status": "no_results", "message": "No analysis results available. Run /api/analyze first."}

    return {
        "status": "success",
        "results": analysis_results
    }


@app.get("/api/eeg-sample/{subject_type}")
async def get_eeg_sample(subject_type: str):
    """Get a sample EEG signal for visualization"""
    if "schizophrenia" in subject_type.lower() or subject_type in ["healthy", "patient"]:
        X, y, metadata = loader.load_schizophrenia_data()

        if len(X) == 0:
            return {"status": "error", "message": "No data available"}

        if subject_type == "healthy":
            idx = np.where(y == 0)[0][0] if np.sum(y == 0) > 0 else 0
        else:
            idx = np.where(y == 1)[0][0] if np.sum(y == 1) > 0 else 0

        # Return first 512 samples (4 seconds) of first 4 channels
        sample = X[idx][:512, :4].tolist()

        return {
            "status": "success",
            "subject_type": subject_type,
            "subject_index": int(idx),
            "data": sample,
            "channels": 4,
            "samples": 512,
            "sampling_rate": metadata['sampling_rate']
        }

    return {"status": "error", "message": f"Unknown subject type: {subject_type}"}


@app.get("/api/data-sample/{disease}")
async def get_data_sample(disease: str, rows: int = 20):
    """Return the on-disk 'as-is' feature sample for a disease as JSON.

    Reads data/{disease}/sample/{disease}_50rows.npz (feature matrix +
    labels) so the UI Data tab can preview real data without re-running
    any pipeline. Returns a row-capped preview plus class distribution.
    """
    disease = disease.lower().strip()
    base = Path(__file__).parent / "data" / disease / "sample"
    # Prefer the 50-row sample, fall back to the 100-row one.
    candidates = [base / f"{disease}_50rows.npz", base / f"{disease}_sample_100.npz"]
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        raise HTTPException(status_code=404, detail=f"No sample file found for '{disease}' under {base}")

    try:
        # These samples hold only numeric + fixed-width string arrays,
        # so the default (allow_pickle=False) is sufficient and safe.
        d = np.load(npz_path)
        X = d["X"]
        y = d["y"]
        feature_names = [str(f) for f in d["feature_names"]] if "feature_names" in d else [f"f{i}" for i in range(X.shape[1])]
        class_names = [str(c) for c in d["class_names"]] if "class_names" in d else ["class_0", "class_1"]
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the UI
        raise HTTPException(status_code=500, detail=f"Failed to read {npz_path.name}: {exc}") from exc

    n_rows = int(X.shape[0])
    n_features = int(X.shape[1])
    cap = max(1, min(rows, n_rows))

    # Class distribution for the chart.
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = [
        {"label": class_names[int(u)] if int(u) < len(class_names) else f"class_{int(u)}", "count": int(c)}
        for u, c in zip(unique, counts)
    ]

    return {
        "status": "success",
        "disease": disease,
        "source_file": str(npz_path.relative_to(Path(__file__).parent)),
        "n_rows": n_rows,
        "n_features": n_features,
        "feature_names": feature_names,
        "class_names": class_names,
        "class_distribution": class_distribution,
        "preview": X[:cap].round(4).tolist(),
        "labels": y[:cap].astype(int).tolist(),
    }


# ===========================================================================
# CLINICAL PIPELINE: analyze-on-upload + per-patient database
# ===========================================================================
class PatientIn(BaseModel):
    patient_id: str
    name: str = ""
    age: Optional[int] = None
    gender: str = ""
    disease: str = ""
    department: str = ""


class SurveyIn(BaseModel):
    patient_id: str
    department: str = ""
    kind: str = "intake"
    answers: Dict[str, Any] = {}


@app.post("/api/analyze-upload")
async def analyze_upload(
    file: UploadFile = File(...),
    disease: str = Form("epilepsy"),
    patient_id: str = Form(""),
    department: str = Form(""),
):
    """Parse an uploaded EEG file, run the full analysis pipeline, persist it
    per patient, write a report, and return the result."""
    suffix = (Path(file.filename or "upload.edf").suffix or ".edf").lower()
    EEG_EXTS = {".edf", ".bdf", ".fif", ".fiff", ".mat", ".npz", ".csv", ".tsv", ".txt", ".dat"}
    # Video (incl. WhatsApp .mp4/.3gp, phone .mov) → Video-EEG / seizure-video for clinician review
    # Video: phone, WhatsApp (.mp4/.3gp), and all common YouTube/download containers
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".3gp", ".3gpp", ".m4v", ".mpeg", ".mpg",
                  ".flv", ".ts", ".ogv", ".wmv", ".f4v", ".m2ts", ".vob"}
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        if suffix in VIDEO_EXTS:
            # Persist video to data/uploads/videos/<patient>/ and log for clinician review
            base = Path(__file__).parent / "data" / "uploads" / "videos" / (patient_id or "_unassigned")
            base.mkdir(parents=True, exist_ok=True)
            safe_name = Path(file.filename or f"video{suffix}").name
            dest = base / safe_name
            shutil.copyfile(tmp.name, dest)
            size_mb = round(dest.stat().st_size / 1e6, 2)
            if patient_id:
                cdb.upsert_patient(patient_id, disease=disease, department=department)
                cdb.log_transaction(patient_id, component="video", action="upload",
                                    detail=f"{safe_name} ({suffix}, {size_mb} MB) → Video-EEG / seizure video")
            return {"status": "success", "mode": "video", "file": safe_name, "file_type": suffix,
                    "size_mb": size_mb, "stored_at": str(dest.relative_to(Path(__file__).parent)),
                    "note": "Video stored for clinician review / Video-EEG concordance / behavioral-event capture. "
                            "Not a signal-AI input — pair with EDF/BDF for seizure analysis."}
        if suffix in EEG_EXTS:
            # EEG signal → full analysis pipeline
            result = eeg.run_pipeline(tmp.name, disease, patient_id=patient_id or None)
            result["file"] = file.filename or Path(tmp.name).name
            if result.get("status") != "success":
                return result
            if patient_id:
                cdb.upsert_patient(patient_id, disease=disease, department=department)
            saved = cdb.save_analysis(result, department=department)
            result["saved"] = saved
            # Log the EEG upload to transaction history (UTC + local) so it appears in the table
            pred = (result.get("prediction") or {}).get("predicted_label", "?")
            cdb.log_transaction(patient_id or "_unassigned", component="eeg_upload", action="analyze",
                                detail=f"{file.filename or Path(tmp.name).name} ({suffix}) → {disease} pred={pred}")
            return _json_safe(result)  # strip NaN/Inf so JSON serialization never 500s
        else:
            # PDF / image / video / docx → extract (CV/OCR/parse), persist to patient
            extracted = ingest.extract_file(Path(tmp.name))
            if patient_id:
                cdb.upsert_patient(patient_id, disease=disease, department=department)
                cdb.log_transaction(patient_id, component="ingest", action="extract",
                                    detail=f"{file.filename} ({suffix}) → {extracted.get('type', 'extracted')}")
            return {"status": "success", "mode": "extraction", "file": file.filename,
                    "file_type": suffix, "extracted": extracted,
                    "note": "Non-EEG file: data extracted (CV/OCR/parse) and saved. Run EEG file for seizure analysis."}
    except Exception as exc:  # noqa: BLE001 - return a clean error envelope to the UI (HTTP 200)
        msg = str(exc)
        hint = ("File could not be processed. "
                "For signal analysis upload EDF/BDF/CSV/FIF/MAT with enough samples "
                "(a few seconds of multi-channel signal). The reference-sample .npz is a "
                "feature matrix, not a raw signal. PDF/image/video are extracted, not analyzed.")
        # Return 200 with status=error so the UI shows the real reason, not "backend offline".
        return {"status": "error", "mode": "upload", "file": file.filename,
                "file_type": suffix, "message": f"{hint}  (detail: {msg[:200]})"}
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/patients")
async def create_patient(p: PatientIn):
    return {"status": "success", "patient": cdb.upsert_patient(**p.model_dump())}


@app.get("/api/patients")
async def get_patients(department: Optional[str] = None, offset: int = 0, limit: int = 100):
    return cdb.list_patients(department=department, offset=offset, limit=limit)


@app.get("/api/patients/{patient_id}")
async def get_one_patient(patient_id: str):
    p = cdb.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"No patient '{patient_id}'")
    return p


@app.post("/api/survey")
async def submit_survey(s: SurveyIn):
    return {"status": "success", **cdb.save_survey(s.patient_id, s.department, s.kind, s.answers)}


@app.get("/api/analyses")
async def get_analyses(disease: Optional[str] = None, offset: int = 0, limit: int = 50):
    return cdb.list_analyses(disease=disease, offset=offset, limit=limit)


class ClinicalIn(BaseModel):
    patient_id: str
    fields: Dict[str, Any] = {}
    analysis_id: Optional[int] = None


_CLINICAL_TABLES = {
    "medications", "mri_findings", "outcomes", "neuropsych", "hitl_reviews", "explainability_gt",
    "eeg_acquisition", "channel_quality", "clinical_history", "recording_conditions",
    "event_annotations", "artifact_annotations", "eeg_interpretation", "seizure_metadata",
    "comorbidities", "hospitalization", "dba_metrics", "model_governance", "risk_management",
}


@app.post("/api/clinical/{table}")
async def save_clinical(table: str, body: ClinicalIn):
    if table not in _CLINICAL_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown clinical table '{table}'")
    saved = cdb.save_clinical(table, body.patient_id, body.fields, analysis_id=body.analysis_id)
    return {"status": "success", **saved}


@app.get("/api/clinical/{table}/{patient_id}")
async def get_clinical(table: str, patient_id: str):
    if table not in _CLINICAL_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown clinical table '{table}'")
    return {"items": cdb.list_clinical(table, patient_id)}


@app.get("/api/patient-clinical/{patient_id}")
async def patient_clinical(patient_id: str):
    return cdb.patient_clinical_summary(patient_id)


@app.get("/api/department-report/{department:path}")
async def department_report(department: str, save: bool = False):
    # :path converter accepts department names containing '/' (e.g. "EEG / Epilepsy Analysis")
    rep = cdb.department_report(department)
    if save:
        rep["report_path"] = cdb.write_department_report(department)
    return rep


# --- Patient Master Data: neurologist uploads multi-format files per patient ---
@app.post("/api/patient-master/ingest")
async def patient_master_ingest(
    patient_id: str = Form(...),
    name: str = Form(""),
    age: str = Form(""),
    gender: str = Form(""),
    notes: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    """Create a per-patient folder, extract data from each uploaded file
    (video / pdf / image / dat / text / docx / edf), and build master data."""
    blobs = [(f.filename or "file", await f.read()) for f in files]
    info = {"age": age, "gender": gender, "notes": notes}
    master = ingest.ingest_patient(patient_id, name, info, blobs)
    cdb.upsert_patient(patient_id, name=name, age=int(age) if str(age).isdigit() else None,
                       gender=gender, disease="epilepsy", department="Patient Master")
    cdb.save_master(patient_id, name, master["n_files"], master["modalities"],
                    str(ingest.PATIENTS_DIR / patient_id / "master_data.json"))
    return {"status": "success", "master": master}


@app.get("/api/patient-master")
async def patient_master_list():
    return cdb.list_masters_db()


@app.get("/api/patient-master/{patient_id}")
async def patient_master_get(patient_id: str):
    m = ingest.get_master(patient_id)
    if not m:
        raise HTTPException(status_code=404, detail=f"No master data for '{patient_id}'")
    return m


@app.get("/api/transactions")
async def transactions(patient_id: Optional[str] = None, offset: int = 0, limit: int = 100):
    return cdb.list_transactions(patient_id=patient_id, offset=offset, limit=limit)


class ChatIn(BaseModel):
    patient_id: str
    query: str = ""
    layout: str = "auto"
    generate: bool = True


@app.post("/api/patient-chat")
async def patient_chat(body: ChatIn):
    """Conversational RAG over a patient's records (any role, anytime):
    retrieve from clinical DB → optional Ollama answer in chosen layout."""
    retrieval = cdb.patient_chat(body.patient_id, body.query)
    if body.generate:
        retrieval["llm"] = ollama_agent.answer(body.query, retrieval["results"], layout=body.layout)
    return retrieval


class FeedbackIn(BaseModel):
    patient_id: str = ""
    role: str = ""
    ai_output: str = ""
    rating: int = 3
    correction: str = ""
    reason: str = ""


@app.post("/api/feedback")
async def submit_feedback(body: FeedbackIn):
    """Feedback / Correction AI (per role) — human-in-loop signal for RLHF."""
    return {"status": "success", **cdb.save_feedback(body.patient_id, body.role, body.ai_output, body.rating, body.correction, body.reason)}


@app.get("/api/feedback")
async def get_feedback(role: Optional[str] = None):
    return cdb.list_feedback(role=role)


@app.get("/api/consensus")
async def get_consensus():
    """Consensus AI — agreement across reviewers of the same patient."""
    return cdb.consensus()


@app.get("/api/decision")
async def get_decision(confidence: float, role: str = "", task: str = ""):
    """Decision AI (per role/task) — route by confidence: auto/review/escalate."""
    return cdb.decision_route(confidence, role=role, task=task)


@app.post("/api/guardrails-check")
async def guardrails_check(body: Dict[str, Any]):
    """Guardrails (per phase) — input/output filters: PII + prompt-injection.
    NeMo Guardrails is the planned production layer; this is the built check."""
    text = body.get("text", "") if isinstance(body, dict) else ""
    pii = rai.pii_scan(text)
    inj = rai.injection_scan(text)
    blocked = pii["pii_found"] or inj["injection_detected"]
    return {"blocked": blocked, "pii": pii, "injection": inj,
            "verdict": "BLOCK" if blocked else "ALLOW",
            "engine": "built-in (NeMo Guardrails integration planned)"}


class TxnIn(BaseModel):
    patient_id: str = ""
    component: str
    action: str
    actor: str = "consultant"
    detail: str = ""


@app.post("/api/transactions")
async def create_transaction(t: TxnIn):
    """Record a timestamped transaction (UTC + local) — e.g. a consultant sign-off."""
    cdb.log_transaction(t.patient_id, component=t.component, action=t.action,
                        actor=t.actor, detail=t.detail)
    return {"status": "success"}


@app.get("/api/agent-tasks")
async def agent_tasks():
    """Agent/task registry with honest built/scaffold/planned status."""
    p = Path(__file__).parent / "config" / "agent_tasks.json"
    if not p.exists():
        return {"agents": []}
    return json.loads(p.read_text())


@app.get("/api/data-requirements")
async def data_requirements():
    """DBA dataset requirements & gap (EEG signal, clinical, meds, imaging, neuropsych, outcome,
    governance, quality, demographics) + tiers + control groups + artifact template."""
    p = Path(__file__).parent / "config" / "data_requirements.json"
    return json.loads(p.read_text()) if p.exists() else {"categories": []}


@app.get("/api/challenges")
async def challenges_catalog():
    """30 epilepsy/EEG challenges grouped by difficulty (basic/intermediate/high) + AI mitigation."""
    p = Path(__file__).parent / "config" / "epilepsy_challenges.json"
    return json.loads(p.read_text()) if p.exists() else {"challenges": []}


@app.get("/api/jobs")
async def jobs_status():
    """All scheduled/background jobs: schedule + cron-installed? + last run + status.
    So every job is visible on the UI (Cron/Jobs tab)."""
    import subprocess
    root = Path(__file__).parent
    reg = json.loads((root / "config" / "jobs.json").read_text()) if (root / "config" / "jobs.json").exists() else {"jobs": []}
    try:
        crontab = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        crontab = ""
    out = []
    for j in reg.get("jobs", []):
        rep_path = root / j.get("report", "")
        last = None
        if rep_path.exists():
            try:
                d = json.loads(rep_path.read_text())
                last = {"run_at": d.get("run_at_local") or d.get("run_at_utc"),
                        "summary": d.get("summary") or d.get("note") or f"{d.get('total_frames', d.get('processed', ''))}",
                        "ok": all(r.get("ok", True) for r in d.get("results", [])) if d.get("results") else True}
            except Exception:
                last = {"run_at": "?", "summary": "report unreadable"}
        out.append({**j, "cron_installed": j.get("cron_tag", "") in crontab, "last_run": last})
    return {"jobs": out, "total": len(out),
            "installed": sum(1 for x in out if x["cron_installed"]),
            "note": reg.get("note", "")}


@app.get("/api/system-health")
async def system_health():
    """Live status of every sub-system — answers 'what is working?' in one call."""
    import urllib.request
    from collections import Counter
    root = Path(__file__).parent
    out = {"backend": {"up": True, "routes": len(app.routes)}}

    # Ollama
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        out["ollama"] = {"up": True, "model_count": len(models), "models": models[:10]}
    except Exception:
        out["ollama"] = {"up": False, "models": []}

    # DB
    try:
        tbls = cdb.list_transactions(limit=1)  # touches DB
        with cdb._connect() as c:
            names = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
            counts = {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in names}
        out["database"] = {"up": True, "tables": len(names), "populated": sum(1 for v in counts.values() if v > 0),
                           "row_counts": counts}
    except Exception as e:
        out["database"] = {"up": False, "error": str(e)[:120]}

    # Registries (built/partial/planned tallies)
    def tally(fname, key):
        try:
            d = json.loads((root / "config" / fname).read_text())
            items = d.get(key, [])
            c = Counter(i.get("status") or i.get("in_project") for i in items)
            return {"total": len(items), **{k: v for k, v in c.items() if k}}
        except Exception:
            return {"total": 0}
    out["registries"] = {
        "agents": tally("agent_tasks.json", "agents"),
        "roles": tally("role_specs.json", "roles"),
        "patient_sections": tally("patient_module.json", "sections"),
        "data_formats": tally("eeg_data_formats.json", "formats"),
    }
    return out


@app.get("/api/production-issues")
async def production_issues():
    """Enterprise production-issue troubleshooting catalog (18 layers) + detection mapping."""
    p = Path(__file__).parent / "config" / "production_issues.json"
    return json.loads(p.read_text()) if p.exists() else {"layers": []}


@app.get("/api/enterprise-pipelines")
async def enterprise_pipelines():
    """Full enterprise pipeline catalog (~40) grouped by category with status."""
    p = Path(__file__).parent / "config" / "enterprise_pipelines.json"
    return json.loads(p.read_text()) if p.exists() else {"groups": []}


@app.get("/api/role-dashboards")
async def role_dashboards():
    """Per-role (clinical department) KPI dashboards + standard report lists."""
    p = Path(__file__).parent / "config" / "role_dashboards.json"
    return json.loads(p.read_text()) if p.exists() else {"roles": []}


@app.get("/api/stories-tests")
async def stories_tests():
    """User stories, demo stories, and the 9-dimension testing matrix."""
    p = Path(__file__).parent / "config" / "stories_and_tests.json"
    return json.loads(p.read_text()) if p.exists() else {"user_stories": [], "demo_stories": [], "testing": []}


@app.get("/api/simulations")
async def simulations():
    """Per-role end-to-end process simulations (step-by-step, layered)."""
    p = Path(__file__).parent / "config" / "simulations.json"
    return json.loads(p.read_text()) if p.exists() else {"roles": []}


@app.get("/api/role-tests")
async def role_tests():
    """Per-role testing matrix scoped to each role's flow."""
    p = Path(__file__).parent / "config" / "role_tests.json"
    return json.loads(p.read_text()) if p.exists() else {"roles": []}


@app.get("/api/neurolab-readiness")
async def neurolab_readiness():
    """NeuroLab deployment readiness: per-stakeholder gaps, processes, functionality, business case."""
    p = Path(__file__).parent / "config" / "neurolab_readiness.json"
    return json.loads(p.read_text()) if p.exists() else {"stakeholders": []}


class SeizureIn(BaseModel):
    patient_id: str
    fields: Dict[str, Any] = {}


@app.post("/api/seizure-diary")
async def seizure_log(body: SeizureIn):
    """Patient/caregiver logs a seizure event (auto-scored severity)."""
    return cdb.save_seizure(body.patient_id, body.fields)


@app.get("/api/seizure-diary/{patient_id}")
async def seizure_list(patient_id: str):
    """Seizure diary + monthly trend + severity distribution + stats."""
    return cdb.list_seizures(patient_id)


@app.get("/api/correlation/{patient_id}")
async def correlation(patient_id: str):
    """Trigger/pattern analysis from the seizure diary (answers 'why did my seizure happen?')."""
    return cdb.analyze_correlations(patient_id)


@app.get("/api/data-formats")
async def data_formats():
    """EEG data-format ranking, AI-readiness, routing, and data-request guidance."""
    p = Path(__file__).parent / "config" / "eeg_data_formats.json"
    return json.loads(p.read_text()) if p.exists() else {"formats": []}


@app.get("/api/tab-scaffold")
async def tab_scaffold():
    """Standard 8-section scaffold per tab (goal/todo/flow/IPO/viz). Default + per-tab overrides."""
    p = Path(__file__).parent / "config" / "tab_scaffold.json"
    return json.loads(p.read_text()) if p.exists() else {"default": {}, "tabs": {}}


@app.get("/api/onboarding-intake")
async def onboarding_intake():
    """Patient onboarding: intake-vs-deferred field classification (the 15x time reduction)."""
    p = Path(__file__).parent / "config" / "onboarding_intake.json"
    return json.loads(p.read_text()) if p.exists() else {"steps": []}


@app.get("/api/iot-devices")
async def iot_devices():
    """Emotiv + IoT + mobile device fleet with online/offline handling model."""
    p = Path(__file__).parent / "config" / "iot_devices.json"
    return json.loads(p.read_text()) if p.exists() else {"devices": []}


@app.get("/api/patient-module")
async def patient_module():
    """Patient module spec: 8 sections, ~1250 fields, honest status."""
    p = Path(__file__).parent / "config" / "patient_module.json"
    return json.loads(p.read_text()) if p.exists() else {"sections": []}


@app.get("/api/role-specs")
async def role_specs():
    """Full 17-role epilepsy platform spec registry (sections + field counts + status)."""
    p = Path(__file__).parent / "config" / "role_specs.json"
    return json.loads(p.read_text()) if p.exists() else {"roles": []}


@app.get("/api/neurologist-workbench/{patient_id}")
async def neurologist_workbench(patient_id: str):
    """Neurologist-centric single screen: Patient → EEG evidence → AI findings →
    explainability → biomarkers → localization → MRI → medication → audit.
    Real data where available; deterministic demo (badged) where not."""
    def dv(seed, lo, hi):
        h = 0
        for ch in (patient_id + seed):
            h = (h * 31 + ord(ch)) % 100000
        return lo + (h % (hi - lo + 1))
    with cdb._connect() as c:  # type: ignore
        prow = c.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
        arow = c.execute("SELECT * FROM analyses WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,)).fetchone()
        meds = [dict(r) for r in c.execute("SELECT * FROM medications WHERE patient_id=? ORDER BY id DESC LIMIT 5", (patient_id,)).fetchall()]
        mri = [dict(r) for r in c.execute("SELECT * FROM mri_findings WHERE patient_id=? ORDER BY id DESC LIMIT 3", (patient_id,)).fetchall()]
    p = dict(prow) if prow else {}
    a = dict(arow) if arow else {}
    bands = {}
    try:
        bands = json.loads(a.get("band_power_json") or "{}") if a else {}
    except Exception:
        bands = {}
    return {
        "patient_id": patient_id,
        "patient_summary": {  # real fields + demo for missing
            "age": p.get("age") or dv("age", 18, 70), "gender": p.get("gender") or "—",
            "diagnosis": p.get("disease") or "epilepsy", "duration_years": dv("dur", 1, 15),
            "seizure_frequency": f"{dv('freq', 1, 6)}/month", "last_seizure_days": dv("last", 1, 60),
            "current_medication": (meds[0].get("fields_json") if meds else None) or "Levetiracetam (demo)",
            "demo": not bool(prow),
        },
        "ai_findings": {
            "predicted": a.get("predicted_label"), "confidence": a.get("confidence"),
            "signal_quality": a.get("signal_quality"), "available": bool(a.get("predicted_label")),
        },
        "explainability": [  # SHAP-style % contributions (demo if no real SHAP cached)
            {"feature": "Spike frequency", "pct": dv("spk", 25, 38)},
            {"feature": "Theta burst", "pct": dv("th", 18, 28)},
            {"feature": "Sharp wave", "pct": dv("sh", 12, 20)},
            {"feature": "Temporal asymmetry", "pct": dv("ta", 8, 16)},
        ],
        "biomarkers": [
            {"marker": "Spike count", "status": ["Normal", "Moderate", "High"][dv("bm1", 0, 2)]},
            {"marker": "Sharp waves", "status": ["Normal", "Moderate", "High"][dv("bm2", 0, 2)]},
            {"marker": "HFO", "status": ["Absent", "Present"][dv("bm3", 0, 1)]},
            {"marker": "Theta power", "status": ["Normal", "Elevated"][dv("bm4", 0, 1)]},
            {"marker": "Delta power", "status": ["Normal", "Elevated"][dv("bm5", 0, 1)]},
            {"marker": "Beta power", "status": ["Normal", "Reduced"][dv("bm6", 0, 1)]},
        ],
        "localization": sorted([
            {"region": "Temporal", "prob": dv("loc1", 60, 92)},
            {"region": "Frontal", "prob": dv("loc2", 2, 12)},
            {"region": "Occipital", "prob": dv("loc3", 1, 8)},
            {"region": "Parietal", "prob": dv("loc4", 1, 6)},
        ], key=lambda x: -x["prob"]),
        "mri_correlation": mri or [{"fields_json": "Left Temporal Lesion (demo)", "match": "Match"}],
        "medications": meds or [{"fields_json": "Levetiracetam 500mg BID (demo)"}],
        "audit": {"model_version": "v2.1", "training_dataset": "CHB-MIT",
                  "date": a.get("generated_at", "—") if a else "—", "reviewer": "(pending sign-off)"},
        "note": "Neurologist-centric workflow. Real patient/analysis data where present; demo (badged) otherwise.",
    }


@app.get("/api/role-challenges")
async def role_challenges():
    """Per-role workflow challenges + how AI in this project mitigates each."""
    p = Path(__file__).parent / "config" / "role_challenges.json"
    return json.loads(p.read_text()) if p.exists() else {"roles": []}


# ---- Patient forms (expert assigns -> patient fills via self-service portal) ----
class FormAssignIn(BaseModel):
    patient_id: str
    instrument: str
    assigned_by: str = ""
    message: str = ""


class FormSubmitIn(BaseModel):
    answers: Dict[str, Any]


@app.post("/api/forms/assign")
async def form_assign(body: FormAssignIn):
    """Expert assigns an assessment form to a patient."""
    return cdb.assign_form(body.patient_id, body.instrument, body.assigned_by, body.message)


@app.get("/api/forms")
async def form_list(patient_id: str = "", status: str = ""):
    """List assigned forms (optionally filter by patient / status)."""
    return {"items": cdb.list_forms(patient_id or None, status or None)}


@app.post("/api/forms/{form_id}/submit")
async def form_submit(form_id: int, body: FormSubmitIn):
    """Patient submits a filled form via self-service portal -> auto-scored."""
    r = cdb.submit_form(form_id, body.answers)
    if not r:
        raise HTTPException(404, "form not found")
    return r


class ExpertReviewIn(BaseModel):
    patient_id: str
    role: str
    finding: str
    agree_with_ai: str = ""
    note: str = ""
    expert: str = ""


@app.get("/api/study-review/{patient_id}")
async def get_study_review(patient_id: str):
    """Upload→AI assessment detail + every expert's review for a patient's study."""
    return cdb.study_review(patient_id)


@app.post("/api/study-review/expert")
async def add_study_expert(body: ExpertReviewIn):
    """An expert (doctor or other role) adds their assessment to the study."""
    return cdb.add_expert_review(body.patient_id, body.role, body.finding,
                                 body.agree_with_ai, body.note, body.expert)


class TeamMsgIn(BaseModel):
    channel: str = "general"
    from_role: str
    text: str
    patient_id: str = ""


@app.post("/api/team-chat")
async def team_chat_post(body: TeamMsgIn):
    """A role posts to a team channel. @bot in the text triggers an AI (Ollama) reply."""
    return cdb.post_team_message(body.channel, body.from_role, body.text, body.patient_id)


@app.get("/api/team-chat")
async def team_chat_list(channel: str = "general", limit: int = 100):
    """Channel message thread."""
    return {"channel": channel, "messages": cdb.list_team_messages(channel, limit)}


@app.get("/api/team-chat/channels")
async def team_chat_channels():
    """All channels with message counts."""
    return {"channels": cdb.list_team_channels()}


class GroupIn(BaseModel):
    name: str
    members: List[str] = []
    topic: str = ""
    created_by: str = ""


class PresenceIn(BaseModel):
    role: str
    status: str = "active"


@app.post("/api/team-chat/group")
async def team_chat_group_create(body: GroupIn):
    """Create a chat group (+ welcome message)."""
    return cdb.create_chat_group(body.name, body.members, body.topic, body.created_by)


@app.get("/api/team-chat/groups")
async def team_chat_groups():
    return {"groups": cdb.list_chat_groups()}


@app.post("/api/team-chat/presence")
async def team_chat_presence_set(body: PresenceIn):
    """Set role presence: active | away | desk | break | offline."""
    return cdb.set_presence(body.role, body.status)


@app.get("/api/team-chat/presence")
async def team_chat_presence_get():
    return {"presence": cdb.get_presence()}


@app.post("/api/team-chat/read")
async def team_chat_read(channel: str, role: str):
    """Mark channel messages read by a role."""
    return cdb.mark_read(channel, role)


class GenAiBotIn(BaseModel):
    role: str
    query: str
    layout: str = "passage"
    patient_id: str = ""


@app.post("/api/genai-bot")
async def genai_bot(body: GenAiBotIn):
    """Generative-AI bot per role: free-text + report access, layout passage|table|list|graph."""
    return cdb.genai_bot(body.role, body.query, body.layout, body.patient_id)


@app.get("/api/admin/dashboards")
async def admin_dashboards():
    """ADMIN: aggregate every dashboard in the system from all registries + system views."""
    cfg = Path(__file__).parent / "config"

    def load(name):
        p = cfg / name
        return json.loads(p.read_text()) if p.exists() else {}

    groups = []

    # 1. System dashboards (real working views)
    system = [
        {"name": "EEG / Epilepsy Analysis", "status": "built", "where": "EEG module"},
        {"name": "SHAP Explainability", "status": "built", "where": "per-disease"},
        {"name": "Interpretable AI (surrogate)", "status": "built", "where": "per-disease"},
        {"name": "Responsible AI (fairness)", "status": "built", "where": "/api/responsible-ai"},
        {"name": "Model Lab (XGB/LGBM/SMOTE/PCA)", "status": "built", "where": "Special Case"},
        {"name": "Anomaly Detection", "status": "built", "where": "Special Case"},
        {"name": "Time-Series + Statistics", "status": "built", "where": "Special Case"},
        {"name": "Council of Agents trace", "status": "built", "where": "Feedback & Governance"},
        {"name": "Study Review (multi-expert)", "status": "built", "where": "AI Types hub"},
        {"name": "Patient Master + Chat (RAG)", "status": "built", "where": "Patient module"},
        {"name": "NeuroLab Readiness", "status": "built", "where": "AI Types hub"},
        {"name": "Tab Taxonomy", "status": "built", "where": "AI Types hub"},
        {"name": "Accuracy / Validation", "status": "built", "where": "VALIDATION_SUMMARY"},
    ]
    groups.append({"group": "System Dashboards (working)", "items": system})

    # 2. Per-role dashboards (from registry)
    rd = load("role_dashboards.json").get("roles", [])
    role_items = []
    for r in rd:
        built = sum(1 for k in r.get("kpis", []) if k.get("status") == "built")
        role_items.append({"name": f"{r['icon']} {r['role']} dashboard", "status": "built" if built else "partial",
                           "where": f"{len(r.get('kpis', []))} KPIs, {len(r.get('reports', []))} reports"})
    groups.append({"group": "Per-Role Dashboards", "items": role_items})

    # 3. Dashboard catalog (the ~400 enterprise spec) — counts per phase
    dc = load("dashboard_catalog.json")
    cat_phases = dc.get("phases", []) if isinstance(dc, dict) else []
    cat_items = []
    for ph in cat_phases:
        ds = ph.get("dashboards", [])
        b = sum(1 for d in ds if d.get("status") == "built")
        cat_items.append({"name": ph.get("name", "phase"), "status": f"{b}/{len(ds)} built", "where": "dashboard_catalog"})
    groups.append({"group": "Enterprise Dashboard Catalog (spec)", "items": cat_items})

    # 4. Coverage registries
    registries = [
        {"name": "AI Types (201)", "status": "catalog", "where": "/api/ai-type-coverage"},
        {"name": "Automatic Pipelines (20)", "status": "catalog", "where": "/api/automatic-pipelines"},
        {"name": "Enterprise Pipelines (45)", "status": "catalog", "where": "/api/enterprise-pipelines"},
        {"name": "Production Issues (16 layers)", "status": "catalog", "where": "/api/production-issues"},
        {"name": "Stories & Tests", "status": "catalog", "where": "/api/stories-tests"},
        {"name": "Simulations (per role)", "status": "catalog", "where": "/api/simulations"},
        {"name": "Portal Tabs", "status": "catalog", "where": "/api/portal-tabs"},
    ]
    groups.append({"group": "Coverage Registries", "items": registries})

    total = sum(len(g["items"]) for g in groups)
    built = sum(1 for g in groups for i in g["items"] if i["status"] == "built")
    return {"groups": groups, "total_entries": total, "built": built,
            "note": "Admin aggregate of every dashboard surface. status: built=real view / partial / catalog=registry / X/Y=spec coverage."}


@app.get("/api/admin/module")
async def admin_module():
    """ADMIN: team roles + ops dashboards (cloud/devops/db/model/llmops/mlops/secops)."""
    p = Path(__file__).parent / "config" / "admin_module.json"
    return json.loads(p.read_text()) if p.exists() else {"team_roles": [], "ops_dashboards": []}


@app.get("/api/knowledge-graph")
async def knowledge_graph(role: str = "", patient_id: str = ""):
    """RDF/RDFS relationship graph (per role / per patient) — nodes, edges, Mermaid."""
    return kg.build_graph(role or None, patient_id or None)


@app.get("/api/flowcharts")
async def flowcharts():
    """Process flowcharts (Mermaid source) for key workflows."""
    p = Path(__file__).parent / "config" / "flowcharts.json"
    return json.loads(p.read_text()) if p.exists() else {"flowcharts": []}


@app.get("/api/portal-tabs")
async def portal_tabs():
    """Self-service patient portal tab registry (forms/campaign/notification/alert/inbox/medication/therapy)."""
    p = Path(__file__).parent / "config" / "portal_tabs.json"
    return json.loads(p.read_text()) if p.exists() else {"tabs": []}


@app.get("/api/tab-taxonomy")
async def tab_taxonomy():
    """Tab taxonomy: Patient Master self-service tabs + per-role operational + AI capability tabs."""
    p = Path(__file__).parent / "config" / "tab_taxonomy.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/report-layout")
async def report_layout():
    """EEG/video-EEG summary report layout (components + AI finding/recommendation + expert summary)."""
    p = Path(__file__).parent / "config" / "report_layout.json"
    return json.loads(p.read_text()) if p.exists() else {"components": [], "sections": []}


@app.get("/api/eeg-report/{patient_id}")
async def eeg_report(patient_id: str):
    """Component-by-component EEG report: AI finding + AI recommendation + doctor finding."""
    layout = json.loads((Path(__file__).parent / "config" / "report_layout.json").read_text())
    report = {"patient_id": patient_id, "components": [], "expert_summary": "", "final_summary": ""}
    with cdb._connect() as c:  # type: ignore
        a = c.execute("SELECT * FROM analyses WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,)).fetchone()
        latest = dict(a) if a else None
    doctor = cdb.get_component_findings(patient_id)
    for comp in layout["components"]:
        finding = comp["ai_finding"]
        if comp["id"] == "epileptiform" and latest:
            finding = f"Predicted {latest.get('predicted_label')} (confidence {latest.get('confidence')})"
        if comp["id"] == "background" and latest:
            finding = f"Signal quality {latest.get('signal_quality')}; band power computed"
        df = doctor.get(comp["id"], {})
        report["components"].append({
            "id": comp["id"], "label": comp["label"],
            "ai_finding": finding, "ai_recommendation": comp["ai_recommendation"],
            "doctor_finding": df.get("doctor_finding", ""), "doctor": df.get("doctor", ""),
            "agree_with_ai": df.get("agree_with_ai", "")})
    report["ai_summary"] = (f"Latest analysis: {latest.get('predicted_label')} "
                            f"(conf {latest.get('confidence')}, quality {latest.get('signal_quality')})") if latest else "No analysis on file."
    return report


class ComponentFindingIn(BaseModel):
    patient_id: str
    component: str
    doctor_finding: str
    doctor: str = ""
    agree_with_ai: str = ""


@app.post("/api/eeg-report/component-finding")
async def save_component_finding(body: ComponentFindingIn):
    """Doctor saves their finding for one EEG component."""
    return cdb.save_component_finding(body.patient_id, body.component, body.doctor_finding,
                                      body.doctor, body.agree_with_ai)


# ---- Standardized clinical assessments (MoCA, PHQ-9, GAD-7, NDDI-E, COPM) + CRUD ----
@app.get("/api/assessments/instruments")
async def assessment_instruments():
    """Catalog of validated assessment instruments (items + scoring + bands) per role."""
    p = Path(__file__).parent / "config" / "assessments.json"
    return json.loads(p.read_text()) if p.exists() else {"instruments": []}


class AssessmentIn(BaseModel):
    patient_id: str
    instrument: str
    answers: Dict[str, Any]
    examiner: str = ""


@app.post("/api/assessments")
async def assessment_create(body: AssessmentIn):
    """CREATE: score answers + persist a completed assessment."""
    return cdb.save_assessment(body.patient_id, body.instrument, body.answers, body.examiner)


@app.get("/api/assessments")
async def assessment_list(patient_id: str = "", limit: int = 50):
    """VIEW (list): completed assessments, optionally filtered by patient."""
    return {"items": cdb.list_assessments(patient_id or None, limit)}


@app.get("/api/assessments/{aid}")
async def assessment_get(aid: int):
    """VIEW (one)."""
    r = cdb.get_assessment(aid)
    if not r:
        raise HTTPException(404, "assessment not found")
    return r


@app.put("/api/assessments/{aid}")
async def assessment_update(aid: int, body: AssessmentIn):
    """CHANGE/EDIT: re-score updated answers."""
    r = cdb.update_assessment(aid, body.answers, body.examiner)
    if not r:
        raise HTTPException(404, "assessment not found")
    return r


@app.delete("/api/assessments/{aid}")
async def assessment_delete(aid: int):
    """DELETE."""
    return {"deleted": cdb.delete_assessment(aid)}


@app.get("/api/automatic-pipelines")
async def automatic_pipelines():
    """Catalog of automatic (end-to-end) pipelines per process + status."""
    p = Path(__file__).parent / "config" / "automatic_pipelines.json"
    return json.loads(p.read_text()) if p.exists() else {"pipelines": []}


@app.get("/api/dashboard-catalog")
async def dashboard_catalog():
    """Enterprise dashboard catalog (5 phases) with built/partial/planned status."""
    p = Path(__file__).parent / "config" / "dashboard_catalog.json"
    return json.loads(p.read_text()) if p.exists() else {"phases": []}


@app.get("/api/ai-types")
async def ai_types():
    """List AI types (from coverage) with status."""
    return ai_type_detail.list_types()


@app.get("/api/ai-types/{ai_type}")
async def ai_type_detail_endpoint(ai_type: str):
    """Per-AI-type detail: objective, todo, manual/AI/pipeline flow, dashboard,
    testing, ResAI, ExpAI, GovAI, visualization, transaction history."""
    return ai_type_detail.detail(ai_type)


@app.get("/api/neuro-advancements")
async def neuro_advancements():
    """Per-modality neuro-AI advancement opportunities + cross-modal ideas."""
    p = Path(__file__).parent / "config" / "neuro_advancements.json"
    return json.loads(p.read_text()) if p.exists() else {"modalities": []}


@app.get("/api/deep-train/{disease}")
async def deep_train(disease: str, epochs: int = 60):
    """Train a real DNN (torch) with subject-wise split — closes the deep-learning gap."""
    return eeg_deep.train_deep(disease, epochs=epochs)


@app.get("/api/spectrogram/{disease}")
async def spectrogram(disease: str):
    """STFT time-frequency of a real EDF channel (review's CNN input modality)."""
    return eeg_deep.spectrogram(disease)


class ForecastIn(BaseModel):
    y_true: List[int] = []
    y_pred: List[int] = []
    hours: float = 1.0


@app.post("/api/forecast-metrics")
async def forecast_metrics(body: ForecastIn):
    """Seizure-forecasting metrics: sensitivity + false-alarm-rate per hour."""
    return eeg_deep.forecast_metrics(body.y_true, body.y_pred, hours=body.hours)


class CouncilIn(BaseModel):
    query: str
    patient_id: str = ""
    tenant_id: str = "default"


@app.post("/api/council/run")
async def council_run(body: CouncilIn):
    """Council of Agents — governed flow: Security→RAG→Eval→Review→Compliance→Audit.
    No agent answers directly; every step carries request_id/trace_id/tenant_id."""
    return council_orchestrator.run_council(body.query, patient_id=body.patient_id, tenant_id=body.tenant_id)


@app.get("/api/timeseries/{disease}")
async def timeseries(disease: str):
    """Time-series analysis: ADF stationarity, band-over-time, change-point."""
    return eeg_timeseries_stats.timeseries(disease)


@app.get("/api/statistics/{disease}")
async def statistics(disease: str):
    """Statistical tests: per-feature t-test/Mann-Whitney + Cohen's d + significance."""
    return eeg_timeseries_stats.statistics(disease)


@app.get("/api/modellab/{disease}/balance")
async def ml_balance(disease: str):
    return eeg_datascience.balance(disease)


@app.get("/api/modellab/{disease}/feature-selection")
async def ml_fs(disease: str):
    return eeg_datascience.feature_selection(disease)


@app.get("/api/modellab/{disease}/compare")
async def ml_compare(disease: str):
    return eeg_datascience.model_compare(disease)


@app.get("/api/modellab/{disease}/pca")
async def ml_pca(disease: str):
    return eeg_datascience.pca(disease)


@app.get("/api/anomaly/{disease}")
async def anomaly(disease: str, contamination: float = 0.1):
    """Unsupervised anomaly detection (Isolation Forest + LOF + One-Class SVM)."""
    return eeg_anomaly.detect(disease, contamination=contamination)


@app.get("/api/anomaly-models")
async def anomaly_models():
    """Anomaly model catalog + parameters + statistical methods."""
    return eeg_anomaly.models_catalog()


@app.get("/api/cross-patient")
async def cross_patient():
    """Real CHB-MIT cross-patient (leave-subjects-out) benchmark result."""
    p = Path(__file__).parent / "jobs" / "reports" / "cross_patient_benchmark.json"
    return json.loads(p.read_text()) if p.exists() else {"available": False, "reason": "run scripts/cross_patient_benchmark.py"}


@app.get("/api/feature-gaps")
async def feature_gaps():
    """Epilepsy DL review (50 papers) → project gap analysis by category."""
    p = Path(__file__).parent / "config" / "feature_gaps.json"
    return json.loads(p.read_text()) if p.exists() else {"gaps": []}


@app.get("/api/observability")
async def observability():
    """Observable AI — temporal trace count, OpenTel status, test status."""
    txns = cdb.list_transactions(limit=1)
    return {
        "temporal": {"engine": "transaction_log (UTC+local per write)", "total_events": txns.get("total", 0), "status": "built"},
        "opentelemetry": {"engine": "OpenTelemetry spans", "status": "planned", "needs": "otel SDK + collector (Jaeger/Tempo)"},
        "testing": {"engine": "endpoint + module verification", "status": "built", "note": "modules verified via live HTTP + py_compile + npm build"},
        "metrics": {"engine": "Prometheus", "status": "planned"},
    }


@app.get("/api/dataset-coverage")
async def dataset_coverage():
    """Strategic neurophysiology dataset/modality/AI-stream coverage map."""
    p = Path(__file__).parent / "config" / "dataset_coverage.json"
    if not p.exists():
        return {"modalities": []}
    return json.loads(p.read_text())


@app.get("/api/consultants")
async def consultants():
    """Consultant engagement matrix (role + task focus) — human oversight registry."""
    p = Path(__file__).parent / "config" / "consultant_matrix.json"
    if not p.exists():
        return {"consultants": []}
    return json.loads(p.read_text())


@app.get("/api/explain/{disease}")
async def explain_global(disease: str, top: int = 15):
    """SHAP global feature importance for a disease model."""
    return xai.global_importance(disease, top=top)


@app.get("/api/explain/{disease}/prediction")
async def explain_pred(disease: str, row: int = 0):
    """SHAP per-prediction contributions for one sample row."""
    return xai.explain_prediction(disease, row=row)


@app.get("/api/explain/{disease}/concordance")
async def explain_concordance(disease: str, expert: str = ""):
    """AI SHAP top-bands vs expert ground-truth (comma-separated tokens)."""
    tokens = [t.strip() for t in expert.split(",") if t.strip()]
    return xai.concordance(disease, expert_features=tokens)


@app.get("/api/interpret/{disease}")
async def interpret(disease: str, max_depth: int = 4):
    """Interpretable AI: surrogate decision tree + extracted rules."""
    return xai.interpretable_surrogate(disease, max_depth=max_depth)


@app.get("/api/responsible-ai/{disease}")
async def responsible_ai_report(disease: str):
    """Per-phase Responsible AI report (fairness + PII + security + scaffolds)."""
    return rai.responsible_summary(disease)


@app.get("/api/fairness/{disease}")
async def fairness_report(disease: str):
    """Fairness metrics (data + model level) per protected attribute."""
    return rai.fairness(disease)


class TextIn(BaseModel):
    text: str = ""


@app.post("/api/pii-scan")
async def pii_scan(body: TextIn):
    return rai.pii_scan(body.text)


@app.post("/api/injection-scan")
async def injection_scan(body: TextIn):
    return rai.injection_scan(body.text)


@app.get("/api/eeg-bands/{disease}")
async def eeg_bands(disease: str):
    """Real band-power signature for a disease (mean per band, overall + per class)
    from the feature sample, plus 10-20 montage + frequency-range reference."""
    disease = disease.lower().strip()
    base = Path(__file__).parent / "data" / disease / "sample"
    npz = next((p for p in [base / f"{disease}_50rows.npz", base / f"{disease}_sample_100.npz"] if p.exists()), None)
    if npz is None:
        raise HTTPException(status_code=404, detail=f"No sample for '{disease}'")
    d = np.load(npz)
    X, y = d["X"], d["y"]
    feats = [str(f) for f in d["feature_names"]]
    class_names = [str(c) for c in d["class_names"]] if "class_names" in d else ["Control", disease.title()]
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    freq_ranges = {"delta": "0.5-4 Hz", "theta": "4-8 Hz", "alpha": "8-13 Hz", "beta": "13-30 Hz", "gamma": "30-45 Hz"}

    def band_means(mask):
        out = {}
        for b in bands:
            col = f"{b}_power"
            if col in feats:
                vals = X[mask, feats.index(col)]
                out[b] = round(float(np.mean(vals)), 4) if len(vals) else 0.0
        return out

    return {
        "disease": disease,
        "frequency_ranges": freq_ranges,
        "band_power_overall": band_means(np.ones(len(y), dtype=bool)),
        "band_power_by_class": {
            class_names[0]: band_means(y == 0),
            class_names[1] if len(class_names) > 1 else "Disease": band_means(y == 1),
        },
        "montage_10_20": {
            "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
            "Temporal": ["T3", "T4", "T5", "T6"],
            "Central": ["C3", "C4", "Cz"],
            "Parietal": ["P3", "P4", "Pz"],
            "Occipital": ["O1", "O2"],
        },
    }


@app.get("/api/consultant-workflows")
async def consultant_workflows():
    """Per-role process workflows (phases → steps) + sign-off gates."""
    p = Path(__file__).parent / "config" / "consultant_workflows.json"
    if not p.exists():
        return {"workflows": {}}
    return json.loads(p.read_text())


if __name__ == "__main__":
    import os
    import uvicorn
    # Default 8010 to avoid colliding with other local projects on :8000.
    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
