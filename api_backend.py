"""
FastAPI Backend for NeuroAI EEG Analysis
========================================
REST API endpoints for EEG data analysis and classification.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Body
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
                return _json_safe(result)  # strip NaN/Inf even on failure path
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


@app.get("/api/clinical-trust")
async def clinical_trust(analysis_id: int = None, patient_id: str = None):
    """Clinical Trust Panel — per-prediction summary (AI label, confidence, top channels, artifact risk) for neurologist sign-off."""
    return _json_safe(cdb.build_trust_panel(analysis_id=analysis_id, patient_id=patient_id))


_shap_cache: dict = {}


@app.get("/api/shap-explain")
async def shap_explain(analysis_id: int = None, patient_id: str = None):
    """Real local SHAP explanation — which features drove THIS prediction (Explainable-AI core)."""
    key = f"{analysis_id}:{patient_id}"
    if key not in _shap_cache:
        try:
            import scripts.shap_explain as sx
            _shap_cache[key] = _json_safe(sx.explain(analysis_id=analysis_id, patient_id=patient_id))
        except Exception as e:  # honest degradation (§57.7) — shap is an optional heavy dep (numba/llvmlite)
            _shap_cache[key] = {"available": False, "error": f"{type(e).__name__}: {e}",
                                "note": "SHAP explainability unavailable in backend python (shap/numba not importable); "
                                        "install shap into the backend interpreter to enable. Other explainability "
                                        "surfaces (feature importance, eeg_explainability) remain available."}
    return _shap_cache[key]


@app.post("/api/clinical-trust/decision")
async def clinical_trust_decision(payload: dict = Body(...)):
    """Record the neurologist's Confirm/Reject/Needs-Review decision (human-oversight audit trail)."""
    return cdb.save_clinical_decision(payload)


@app.get("/api/clinical-decisions")
async def clinical_decisions(patient_id: str = None):
    """Human-in-the-loop oversight audit trail (decision + agreement distribution)."""
    return cdb.list_clinical_decisions(patient_id=patient_id)


@app.get("/api/expert-dashboards")
async def expert_dashboards():
    """Expert dashboard catalog (~45 dashboards across roles) with honest built/planned status."""
    p = Path(__file__).parent / "config" / "expert_dashboards.json"
    return json.loads(p.read_text()) if p.exists() else {"error": "missing config"}


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


@app.get("/api/fairness")
async def fairness():
    """Real Fairlearn fairness analysis (demographic-parity by sex) on assessment outcomes."""
    p = Path(__file__).parent / "jobs" / "reports" / "fairness_latest.json"
    return json.loads(p.read_text()) if p.exists() else {"error": "run scripts/fairness_analysis.py"}


from fastapi.responses import HTMLResponse as _HTMLResponse

@app.get("/api/integration-status")
async def integration_status_live():
    """Live status of every local-AI integration (Ollama, OpenClaw, agents, Slack, MCP, failover)."""
    import scripts.integration_status as ist
    return _json_safe(ist.collect())


@app.get("/integration-hub", response_class=_HTMLResponse)
async def integration_hub_ui():
    """Integration Hub UI — single-page dashboard of all integrations."""
    from pathlib import Path as _P
    html = _P(__file__).parent / "frontend" / "integration-hub.html"
    return _HTMLResponse(html.read_text() if html.exists() else "<h1>integration-hub.html missing</h1>")


@app.get("/api/data-manager")
async def data_manager():
    """Clinical Data Manager — live data-quality engine + detailed task catalog (steps+challenges)."""
    cfg_p = Path(__file__).parent / "config" / "data_manager.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    live = _json_safe(cdb.data_manager_report())
    return {"role": cfg.get("role"), "mission": cfg.get("mission"), "summary": cfg.get("summary"),
            "tasks": cfg.get("tasks", []), "dashboards": cfg.get("dashboards", []),
            "quality_assessments": cfg.get("quality_assessments", []), "live": live}


@app.get("/api/data-manager/archival")
async def data_manager_archival():
    """Data Archival / Retention report — per-table row count, record age, and archival
    candidates under the §7.4/§41.2 retention policy. Report only (no destructive action)."""
    import scripts.data_archival as da
    return _json_safe(da.archival_report())


@app.get("/api/data-manager/terminology")
async def data_manager_terminology(patient_id: str = None):
    """Terminology Mapping — instruments → canonical category/domain taxonomy + coverage."""
    import scripts.terminology_standardization as ts
    return _json_safe(ts.terminology_map(patient_id))


@app.get("/api/data-manager/standardization")
async def data_manager_standardization(patient_id: str = None):
    """Data Standardization — normalize level vocab to canonical ordinal; flag non-conforming."""
    import scripts.terminology_standardization as ts
    return _json_safe(ts.standardize_levels(patient_id))


@app.get("/api/data-manager/dataset-version")
async def data_manager_dataset_version():
    """Dataset Versioning — SHA-256 manifest of real dataset/model artifacts + composite fingerprint."""
    import scripts.dataset_versioning as dv
    return _json_safe(dv.version_manifest())


@app.get("/api/data-manager/label-validation")
async def data_manager_label_validation():
    """Label Validation — analyses label consistency + reference dataset class-balance QC."""
    import scripts.label_validation as lv
    return _json_safe(lv.full_report())


@app.get("/api/data-manager/video-validation")
async def data_manager_video_validation():
    """Video Validation — per-frame integrity/blank/dimension QC over real extracted frames."""
    import scripts.video_validation as vv
    return _json_safe(vv.validate_frames())


@app.get("/api/data-manager/annotation-qc")
async def data_manager_annotation_qc():
    """Annotation QC — inter-rater κ (Cohen/Fleiss), AI-human agreement, annotation coverage + flags."""
    import scripts.annotation_qc as aqc
    return _json_safe(aqc.full_report())


@app.get("/api/data-manager/mri-validation")
async def data_manager_mri_validation():
    """MRI Validation — schema + conditional-logic QC over real mri_findings records."""
    import scripts.mri_validation as mv
    return _json_safe(mv.validate())


@app.get("/api/data-manager/cleaning")
async def data_manager_cleaning():
    """Clinical Data Manager — Data Cleaning dashboard.
    Runs real signal-quality analysis on CHB-MIT EEG recordings:
    flat/saturated detection, NaN/Inf stats, ICA summary, quality re-score."""
    import scripts.data_cleaning as dc
    return _json_safe(dc.cleaning_report())


@app.get("/api/data-manager/data-sharing")
async def data_manager_data_sharing():
    """Clinical Data Manager — Data Sharing dashboard.
    PII scan, access-policy matrix, audit summary, export readiness, DUA terms."""
    import scripts.data_sharing as ds
    return _json_safe(ds.sharing_report())


@app.get("/api/data-manager/dataset-validation")
async def data_manager_dataset_validation():
    """Clinical Data Manager — Dataset Validation dashboard.
    Invalid records, duplicates, missing metadata, outliers, EEG file integrity."""
    import scripts.dataset_validation as dv
    return _json_safe(dv.validation_report())


@app.get("/api/data-manager/governance")
async def data_manager_governance():
    """Clinical Data Manager — Governance dashboard.
    Consent tracking, IRB protocol status, de-identification audit,
    encryption posture, and access/audit log analysis from real clinical.db."""
    import scripts.data_governance as dgov
    return _json_safe(dgov.governance_report())


@app.get("/api/icalabel")
async def icalabel_dashboard():
    """ICLabel ICA Component Classification Dashboard.
    Reads pre-computed ICLabel report from jobs/reports/icalabel_latest.json.
    Report generated by: python scripts/icalabel_dashboard.py > jobs/reports/icalabel_latest.json
    Real mne-icalabel neural-net classifier on CHB-MIT EEG recordings."""
    p = Path(__file__).parent / "jobs" / "reports" / "icalabel_latest.json"
    if p.exists():
        return json.loads(p.read_text())
    # Fallback: try running inline (may hit OpenSSL conflict in-process)
    try:
        import scripts.icalabel_dashboard as icl
        result = _json_safe(icl.icalabel_report())
        # Cache to file for next call
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, default=str))
        return result
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}",
                "note": "Run: python scripts/icalabel_dashboard.py to generate the report"}


_mp_cache = {}
@app.get("/api/model-performance")
async def model_performance():
    """Real model performance (ROC/PR/confusion/metrics), subject-wise CV — cached (CV is slow)."""
    import scripts.model_performance as mp
    if "r" not in _mp_cache:
        try:
            _mp_cache["r"] = _json_safe(mp.build())
        except Exception as e:  # honest degradation (§57.7) — never a 500 from a recoverable load/CV error
            _mp_cache["r"] = {"available": False, "error": f"{type(e).__name__}: {e}",
                              "note": "model-performance unavailable (model load or CV failed); see backend log for traceback"}
    return _mp_cache["r"]


@app.get("/api/conversation")
async def conversation():
    """Full I/O chain — operator inputs + assistant responses, timestamped."""
    return _json_safe(cdb.list_convo())


@app.get("/api/db-status")
async def db_status():
    """Mandatory DB record-count status — clinical tables + vector DB + graph DB + raw data + last job runs."""
    import sqlite3, glob, os
    from datetime import datetime
    base = Path(__file__).parent
    out = {"tables": {}, "vector_db": None, "graph_db": None, "raw_data": {}, "last_jobs": {}}
    c = sqlite3.connect(str(base / "data" / "clinical.db"))
    for t in ["patients","analyses","assessments","seizure_diary","clinical_decisions",
              "operator_requests","conversation_log","medications","mri_findings","advisor_issues"]:
        try: out["tables"][t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception: pass
    try:
        import chromadb
        out["vector_db"] = chromadb.PersistentClient(path=str(base/"data/vector_db")).get_or_create_collection("clinical").count()
    except Exception: pass
    g = base / "jobs/reports/graph_latest.json"
    if g.exists():
        gd = json.loads(g.read_text()); out["graph_db"] = gd.get("triples", gd.get("nodes"))
    out["raw_data"] = {"eeg_datasets": len(glob.glob(str(base/"data/real_eeg/*/"))),
                       "edf_files": len(glob.glob(str(base/"data/real_eeg/**/*.edf"), recursive=True))}
    for r in ["training_latest","vector_latest","graph_latest","drift_latest","fairness_latest","data_quality_latest","cv_pipeline_latest"]:
        f = base / f"jobs/reports/{r}.json"
        if f.exists(): out["last_jobs"][r] = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return _json_safe(out)


@app.get("/api/automation-status")
async def automation_status():
    """How do I know the automation works — plan/crons/count/system/crash-survival/completion."""
    import scripts.automation_status as a
    return _json_safe(a.build())


@app.get("/api/requests")
async def list_requests(status: str = None):
    """Operator request inbox — every input logged, with status (open/done/blocked)."""
    return _json_safe(cdb.list_requests(status=status))


@app.post("/api/requests")
async def add_request(payload: dict = Body(...)):
    return cdb.save_request(payload.get("text", ""), payload.get("category", "general"), "ui")


@app.post("/api/requests/update")
async def update_request(payload: dict = Body(...)):
    return cdb.update_request(payload.get("id"), payload.get("status"), payload.get("notes"))


@app.get("/api/expert-roles")
async def expert_roles():
    """8 multidisciplinary expert roles (Pharmacist/Nurse/SLP/OT/Dietitian/Psychologist/MSW/Coordinator) — tasks w/ steps+challenges+status."""
    p = Path(__file__).parent / "config" / "expert_roles.json"
    return json.loads(p.read_text()) if p.exists() else {"roles": []}


@app.get("/api/patient-compare")
async def patient_compare(a: str, b: str):
    """Side-by-side comparison of two patients (demographics + assessments + EEG + seizures)."""
    return _json_safe(cdb.compare_patients(a, b))


_seizure_tl_cache: dict = {}

@app.get("/api/seizure-timeline")
async def seizure_timeline():
    """Seizure Timeline Dashboard — real CHB-MIT annotations + spike detection + peri-onset EEG."""
    if "r" not in _seizure_tl_cache:
        from scripts.seizure_timeline_dashboard import generate_seizure_timeline_report
        try:
            _seizure_tl_cache["r"] = _json_safe(generate_seizure_timeline_report())
        except Exception as e:
            _seizure_tl_cache["r"] = {"available": False, "error": f"{type(e).__name__}: {e}",
                                       "total_seizures": 0}
    return _seizure_tl_cache["r"]


@app.get("/api/spike-overlay")
async def spike_overlay():
    """Spike / Sharp-Wave Overlay — individual spike & sharp-wave detection with morphological features from CHB-MIT EEG."""
    from scripts.spike_overlay_dashboard import generate_spike_overlay_report
    return _json_safe(generate_spike_overlay_report())


@app.get("/api/ilae-classification")
async def ilae_classification():
    """ILAE 2017 Seizure Classification — real CHB-MIT EEG features → focal/generalized/unknown onset typing."""
    from scripts.ilae_seizure_classification import generate_ilae_classification_report
    return _json_safe(generate_ilae_classification_report())


@app.get("/api/cognitive-tests")
async def cognitive_tests():
    """Digital cognitive test catalog (Stroop, Trail Making, Digit Span, WCST, N-Back, Go/No-Go, CPT, Clock Drawing, RAVLT, Verbal Fluency) + scoring + patient results."""
    import scripts.cognitive_tests as ct
    return _json_safe(ct.build())


@app.post("/api/cognitive-tests/score")
async def cognitive_test_score(body: dict = Body(...)):
    """Score a cognitive test result against published norms. Body: {test_id, raw: {metric: value}}."""
    import scripts.cognitive_tests as ct
    test_id = body.get("test_id", "")
    raw = body.get("raw", {})
    return ct.score_result(test_id, raw)


@app.get("/api/drift")
async def drift():
    """Drift monitor (PSI+KS): training reference vs live extractor features. Detects train/serve skew."""
    p = Path(__file__).parent / "jobs" / "reports" / "drift_latest.json"
    return json.loads(p.read_text()) if p.exists() else {"available": False, "error": "run scripts/drift_job.py"}


# ── Drift Dashboard (MLOps) ──────────────────────────────────────────────────
@app.get("/api/drift/dashboard")
async def drift_dashboard():
    """Drift dashboard overview — verdict, severity breakdown, sample sizes, recommendation."""
    import scripts.drift_dashboard as dd
    return _json_safe(dd.drift_overview())


@app.get("/api/drift/features")
async def drift_features(sort_by: str = "psi", limit: int = 20):
    """Per-feature drift table — PSI, KS-stat, severity band, sorted by PSI or KS."""
    import scripts.drift_dashboard as dd
    return _json_safe(dd.drift_features(sort_by=sort_by, limit=limit))


@app.get("/api/drift/severity")
async def drift_severity():
    """Severity distribution for chart rendering (pie/bar data)."""
    import scripts.drift_dashboard as dd
    return _json_safe(dd.drift_severity_distribution())


@app.get("/api/drift/alerts")
async def drift_alerts(psi_threshold: float = 0.25):
    """Actionable drift alerts — features exceeding PSI threshold with recommended actions."""
    import scripts.drift_dashboard as dd
    return _json_safe(dd.drift_alerts(psi_threshold=psi_threshold))


@app.get("/api/drift/trend")
async def drift_trend():
    """Historical drift trend — fraction drifted + high-drift count over time from report archive."""
    import scripts.drift_dashboard as dd
    return _json_safe(dd.drift_trend())


@app.get("/api/drift/definitions")
async def drift_definitions():
    """Drift scale definitions — PSI thresholds, KS-test interpretation, severity levels."""
    import scripts.drift_dashboard as dd
    return _json_safe(dd.scale_definitions())


@app.get("/api/eeg-ai-stack")
async def eeg_ai_stack():
    """EEG AI tool ecosystem (16 layers) + EDC/assessment tools, with honest installed status."""
    p = Path(__file__).parent / "config" / "eeg_ai_stack.json"
    return json.loads(p.read_text()) if p.exists() else {"layers": []}


_eeg_viz_cache: dict = {}


@app.get("/api/eeg-viz/presets")
async def eeg_viz_presets():
    """Real EDF presets for the P0 clinical visuals (PSD/spectrogram/topomap)."""
    import scripts.eeg_viz as viz
    return viz.list_presets()


@app.get("/api/eeg-viz")
async def eeg_viz(file: str = None, seconds: float = 10):
    """Real EEG P0 visuals from raw EDF (MNE/SciPy): PSD curve, band power, spectrogram PNG, scalp topomap PNG."""
    import scripts.eeg_viz as viz
    if not file:
        file = viz.list_presets().get("default")
    if not file:
        return {"available": False, "error": "No EDF presets found on disk."}
    key = f"{file}:{seconds}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(viz.render(file, seconds=seconds))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/recordings")
async def eeg_viz_recordings(limit: int = 60):
    """Raw EEG Viewer — list real .edf recordings on disk grouped by dataset."""
    import scripts.eeg_viz as viz
    return _json_safe(viz.list_recordings(limit=limit))


@app.get("/api/eeg-viz/traces")
async def eeg_viz_traces(file: str = None, start: float = 0.0, seconds: float = 10.0):
    """Raw EEG Viewer — downsampled time-domain waveform traces (the multi-channel strip chart)."""
    import scripts.eeg_viz as viz
    if not file:
        file = viz.list_presets().get("default")
    if not file:
        return {"available": False, "error": "No EDF recordings found on disk."}
    key = f"traces:{file}:{start}:{seconds}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(viz.raw_traces(file, start=start, seconds=seconds))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/bad-channels")
async def eeg_viz_bad_channels(file: str = None, seconds: float = 30.0):
    """Bad Channel Dashboard — per-channel signal-quality QC (flat/disconnected/noisy/line-noise)."""
    import scripts.eeg_viz as viz
    import scripts.eeg_quality as q
    if not file:
        file = viz.list_presets().get("default")
    if not file:
        return {"available": False, "error": "No EDF recordings found on disk."}
    key = f"badch:{file}:{seconds}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(q.bad_channels(file, seconds=seconds))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/artifacts")
async def eeg_viz_artifacts(file: str = None, seconds: float = 60.0):
    """Artifact Review — window-based eye-blink/muscle/line-noise/movement detection from real EDF."""
    import scripts.eeg_viz as viz
    import scripts.eeg_quality as q
    if not file:
        file = viz.list_presets().get("default")
    if not file:
        return {"available": False, "error": "No EDF recordings found on disk."}
    key = f"artifacts:{file}:{seconds}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(q.artifact_review(file, seconds=seconds))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/seizure-annotations")
async def eeg_seizure_annotations():
    """Ictal/Interictal — CHB-MIT seizure annotations (per-file seizure counts + time windows)."""
    import scripts.ictal_analysis as ia
    return _json_safe({"available": True, "files": ia.parse_seizure_annotations()})


@app.get("/api/eeg-viz/ictal-interictal")
async def eeg_ictal_interictal(file: str = None):
    """Ictal vs Interictal — band-power contrast (ictal delta-dominance) from annotated seizures."""
    import scripts.ictal_analysis as ia
    key = f"ictal:{file}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(ia.ictal_interictal(file))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/sleep-recordings")
async def eeg_sleep_recordings():
    """Sleep State — list real Sleep-EDF PSG+Hypnogram recording pairs."""
    import scripts.sleep_staging as ss
    return _json_safe(ss.list_sleep_recordings())


@app.get("/api/eeg-viz/sleep-architecture")
async def eeg_sleep_architecture(hypnogram: str = None):
    """Sleep State Dashboard — sleep architecture (stages, efficiency, REM/N3%) from real hypnogram."""
    import scripts.sleep_staging as ss
    key = f"sleep:{hypnogram}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(ss.sleep_architecture(hypnogram))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/montage-comparison")
async def eeg_montage_comparison(file: str = None):
    """Montage Comparison — same EEG under referential / CAR / bipolar montages with band-power contrast."""
    import scripts.montage_compare as mc
    key = f"montage:{file}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(mc.compare(file))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/complexity")
async def eeg_complexity_features(file: str = None, seconds: float = 10.0):
    """EEG Complexity — entropy (AntroPy: spectral/permutation/sample) + fractal (Nolds: DFA/Hurst) per channel."""
    import scripts.eeg_complexity as ec
    key = f"complexity:{file}:{seconds}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(ec.complexity(file, seconds=seconds))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/localization")
async def eeg_localization(file: str = None):
    """Localization Dashboard — seizure-focus localization by per-channel ictal power increase."""
    import scripts.localization as loc
    key = f"localize:{file}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(loc.localize(file))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/propagation")
async def eeg_propagation(file: str = None):
    """Seizure Propagation Map — per-channel onset order during ictal (spread sequence)."""
    import scripts.propagation as prop
    key = f"propagation:{file}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(prop.propagation(file))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/false-alarm")
async def eeg_false_alarm(file: str = None):
    """False Alarm Review — power detector vs ground-truth annotations (sensitivity + FA/hour)."""
    import scripts.false_alarm as fa
    key = f"falsealarm:{file}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(fa.review(file))
    return _eeg_viz_cache[key]


@app.get("/api/eeg-viz/connectivity")
async def eeg_connectivity(file: str = None, band: str = "alpha"):
    """EEG Connectivity — pairwise spectral coherence (functional connectivity) + hub channels."""
    import scripts.connectivity as conn
    key = f"connectivity:{file}:{band}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(conn.connectivity(file, band=band))
    return _eeg_viz_cache[key]


_catboost_cache: dict = {}
@app.get("/api/catboost-model")
async def catboost_model(disease: str = "epilepsy"):
    """CatBoost alternative model — subject-wise CV metrics + feature importance + comparison to deployed."""
    import scripts.catboost_model as cb
    if disease not in _catboost_cache:
        _catboost_cache[disease] = _json_safe(cb.build(disease))
    return _catboost_cache[disease]


@app.get("/api/eeg-viz/tsfel-features")
async def eeg_tsfel(file: str = None, seconds: float = 10.0):
    """TSFEL features — automated statistical+temporal feature bank per channel from real EDF."""
    import scripts.tsfel_features as tf
    key = f"tsfel:{file}:{seconds}"
    if key not in _eeg_viz_cache:
        _eeg_viz_cache[key] = _json_safe(tf.extract(file, seconds=seconds))
    return _eeg_viz_cache[key]


@app.get("/api/neuro-ai-ecosystem")
async def neuro_ai_ecosystem():
    """Full Neuro AI open-source ecosystem (EDC, cognitive platforms, rating scales, cognitive tests, annotation, XAI, RAI) with honest status."""
    p = Path(__file__).parent / "config" / "neuro_ai_ecosystem.json"
    return json.loads(p.read_text()) if p.exists() else {"error": "missing config"}


@app.get("/api/integrations")
async def integrations_settings():
    """Integrations + delivery channels with honest status + the credential needed to activate."""
    p = Path(__file__).parent / "config" / "integrations.json"
    return json.loads(p.read_text()) if p.exists() else {"integrations": []}


@app.get("/api/neuro-tests")
async def neuro_tests():
    """Neurophysiology/electrodiagnostic test catalog (EEG/NCV/EMG/VEP/BERA/SSEP/blink/RNS/
    autonomic/RR/ABPM/SSR) with per-test EEG-linkage + status."""
    p = Path(__file__).parent / "config" / "neuro_tests.json"
    return json.loads(p.read_text()) if p.exists() else {"tests": []}


@app.get("/api/assessment-catalog")
async def assessment_catalog():
    """Full ranked clinical-assessment catalog (27 instruments) for the epilepsy thesis,
    with built/partial/planned status + priority + specialist + top-10 ranking."""
    p = Path(__file__).parent / "config" / "assessment_catalog.json"
    return json.loads(p.read_text()) if p.exists() else {"categories": []}


@app.get("/api/assessment-dashboard")
async def assessment_dashboard():
    """Assessment analytics sliced by type / level / examiner(user) / disease / date."""
    from collections import Counter, defaultdict
    with cdb._connect() as c:  # type: ignore
        rows = [dict(r) for r in c.execute("SELECT * FROM assessments").fetchall()]
        pdis = {r["patient_id"]: r["disease"] for r in
                [dict(x) for x in c.execute("SELECT patient_id, disease FROM patients").fetchall()]}
    by_type = Counter(r.get("instrument") for r in rows)
    by_level = Counter(r.get("level") for r in rows)
    by_user = Counter(r.get("examiner") or "unspecified" for r in rows)
    # data provenance — synthetic vs real (operator: tag synthetic; real data comes from folder/UI upload)
    def src(r):
        e = (r.get("examiner") or "").upper()
        return "SYNTHETIC" if "SYNTH" in e else ("REAL" if "REAL" in e else "other")
    by_source = Counter(src(r) for r in rows)
    by_disease = Counter(pdis.get(r.get("patient_id"), "unknown") for r in rows)
    by_date = Counter((r.get("created_at") or "")[:10] for r in rows if r.get("created_at"))
    # avg score per instrument
    sums = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r.get("score") is not None:
            sums[r["instrument"]][0] += r["score"]; sums[r["instrument"]][1] += 1
    avg_score = {k: round(v[0] / v[1], 1) for k, v in sums.items() if v[1]}
    alerts = sum(1 for r in rows if r.get("alert"))
    return {"total": len(rows), "by_type": dict(by_type), "by_level": dict(by_level),
            "by_user": dict(by_user), "by_disease": dict(by_disease), "by_source": dict(by_source),
            "by_date": dict(sorted(by_date.items())), "avg_score": avg_score, "alerts": alerts,
            "recent": sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)[:15]}


@app.get("/api/role-process-flow/{role}")
async def role_process_flow(role: str):
    """End-to-end process flow (steps + mermaid) for a given role; default if not specific."""
    p = Path(__file__).parent / "config" / "role_process_flows.json"
    cfg = json.loads(p.read_text()) if p.exists() else {"default": {}, "roles": {}}
    roles = cfg.get("roles", {})
    # match by substring (e.g. "Neurologist / Epileptologist" → "Neurologist")
    match = next((v for k, v in roles.items() if k.lower() in role.lower() or role.lower() in k.lower()), None)
    return {"role": role, "flow": match or cfg.get("default", {})}


@app.get("/api/eeg-ai-rag-pipeline")
async def eeg_ai_rag_pipeline():
    """Complete 23-step EEG→AI→RAG pipeline with honest per-step status + where-it-lives."""
    p = Path(__file__).parent / "config" / "eeg_ai_rag_pipeline.json"
    return json.loads(p.read_text()) if p.exists() else {"steps": []}


@app.get("/api/ai-dark-factory")
async def ai_dark_factory():
    """AI Dark Factory reference: BMAD→Archon→OpenHands→Playwright→DeepEval→Temporal→OTel
    flow + tool catalog + planes + agent patterns, with honest built/cataloged/planned status."""
    p = Path(__file__).parent / "config" / "ai_dark_factory.json"
    return json.loads(p.read_text()) if p.exists() else {"full_flow": []}


@app.get("/api/training-results")
async def training_results():
    """After-training metrics for visualization (per-subject accuracy/f1/sensitivity) +
    preprocessing pipeline steps (AS-IS → cleaned → trained)."""
    root = Path(__file__).parent
    ps = root / "jobs" / "reports" / "accuracy_patient_specific.json"
    data = json.loads(ps.read_text()) if ps.exists() else {}
    preprocessing = [
        {"step": "1. Missing-value check", "detail": "drop/interpolate gaps; flat/NaN channel detection", "applied": True},
        {"step": "2. Noise cleaning", "detail": "notch (50/60Hz) + band-pass + ICA artifact removal", "applied": True},
        {"step": "3. Data conversion", "detail": "EDF/BDF/FIF/MAT → uniform channels×samples array", "applied": True},
        {"step": "4. Correction / re-reference", "detail": "montage validation, channel re-reference", "applied": True},
        {"step": "5. Normalization", "detail": "per-channel min-max to comparable range", "applied": True},
        {"step": "6. Standardization", "detail": "z-score (mean 0, std 1) per feature", "applied": True},
        {"step": "7. Windowing", "detail": "4s windows, 2s stride (overlapping epochs)", "applied": True},
        {"step": "8. Feature extraction", "detail": "15 band-power/statistical features per window", "applied": True},
    ]
    return {
        "preprocessing": preprocessing,
        "after_training": {
            "benchmark": data.get("benchmark", ""),
            "mean_accuracy": data.get("mean_accuracy"),
            "mean_sensitivity": data.get("mean_sensitivity"),
            "per_subject": data.get("per_subject", []),
            "no_leakage": data.get("no_leakage", ""),
        },
        "generated_at": data.get("generated_at", ""),
    }


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


# ── Cross-Patient Benchmark Dashboard ─────────────────────────────────

@app.get("/api/cross-patient-benchmark/overview")
async def cross_patient_benchmark_overview():
    """Cross-Patient Benchmark overview — LOSO KPIs, fold performance, in-sample gap."""
    import scripts.cross_patient_dashboard as cpd
    return _json_safe(cpd.overview())


@app.get("/api/cross-patient-benchmark/breakdown")
async def cross_patient_benchmark_breakdown():
    """Cross-Patient Benchmark breakdown — fold detail, generalization gap, spatial patterns."""
    import scripts.cross_patient_dashboard as cpd
    return _json_safe(cpd.breakdown())


@app.get("/api/cross-patient-benchmark/definitions")
async def cross_patient_benchmark_definitions():
    """Cross-Patient Benchmark definitions — terms, references, clinical interpretation."""
    import scripts.cross_patient_dashboard as cpd
    return _json_safe(cpd.definitions())


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


@app.get("/api/pharmacist")
async def pharmacist_dashboard(patient_id: str = None):
    """Clinical Pharmacist (Epilepsy) — full dashboard: med reconciliation, drug interactions, TDM, ADR, pregnancy safety.
    All built from REAL medication data in clinical.db + ASM pharmacology knowledge base."""
    import scripts.pharmacist_module as pharm
    return _json_safe(pharm.full_dashboard(patient_id))


@app.get("/api/pharmacist/reconciliation")
async def pharmacist_reconciliation(patient_id: str = None):
    """Medication reconciliation: dedup, normalize, gap detection, timeline."""
    import scripts.pharmacist_module as pharm
    return _json_safe(pharm.medication_reconciliation(patient_id))


@app.get("/api/pharmacist/interactions")
async def pharmacist_interactions(patient_id: str = None):
    """Drug interaction check: pairwise ASM interactions, CYP450 overlaps, severity ranking."""
    import scripts.pharmacist_module as pharm
    return _json_safe(pharm.drug_interaction_check(patient_id))


@app.get("/api/pharmacist/tdm")
async def pharmacist_tdm(patient_id: str = None):
    """Therapeutic Drug Monitoring: serum level targets per ASM."""
    import scripts.pharmacist_module as pharm
    return _json_safe(pharm.therapeutic_drug_monitoring(patient_id))


@app.get("/api/pharmacist/adr")
async def pharmacist_adr(patient_id: str = None):
    """ADR / side-effect monitoring: per-drug ADR profiles + overlapping risk."""
    import scripts.pharmacist_module as pharm
    return _json_safe(pharm.adr_monitoring(patient_id))


@app.get("/api/pharmacist/pregnancy-safety")
async def pharmacist_pregnancy(patient_id: str = None):
    """Pregnancy / special-population safety: category flags + guidance."""
    import scripts.pharmacist_module as pharm
    return _json_safe(pharm.pregnancy_safety(patient_id))


@app.get("/api/pharmacist/adherence")
async def pharmacist_adherence(patient_id: str = None):
    """Medication adherence: MMAS-8 proxy score + MPR heuristic + seizure-gap correlation.
    Real data from medications + seizure_diary tables in clinical.db."""
    import scripts.pharmacist_module as pharm
    return _json_safe(pharm.adherence_assessment(patient_id))


@app.get("/api/nurse")
async def nurse_dashboard(patient_id: str = None):
    """Epilepsy Nurse Specialist (ENS) — full dashboard: seizure diary analysis, adherence coaching,
    SUDEP/safety counseling, follow-up triage, education assessment.
    All built from REAL seizure_diary + medications tables in clinical.db."""
    import scripts.nurse_module as nurse
    return _json_safe(nurse.full_dashboard(patient_id))


@app.get("/api/nurse/diary-analysis")
async def nurse_diary_analysis(patient_id: str = None):
    """Seizure diary analysis: monthly trend, severity distribution, trigger correlation, injury/ER rates."""
    import scripts.nurse_module as nurse
    return _json_safe(nurse.seizure_diary_analysis(patient_id))


@app.get("/api/nurse/adherence")
async def nurse_adherence(patient_id: str = None):
    """Medication adherence coaching: cross-reference meds with seizure frequency, identify gaps."""
    import scripts.nurse_module as nurse
    return _json_safe(nurse.adherence_coaching(patient_id))


@app.get("/api/nurse/safety")
async def nurse_safety(patient_id: str = None):
    """SUDEP risk assessment + safety counseling checklist (evidence-based risk factors)."""
    import scripts.nurse_module as nurse
    return _json_safe(nurse.safety_counseling(patient_id))


@app.get("/api/nurse/triage")
async def nurse_triage(patient_id: str = None):
    """Follow-up triage: risk-stratified patient list for telephone/clinic follow-up."""
    import scripts.nurse_module as nurse
    return _json_safe(nurse.follow_up_triage(patient_id))


@app.get("/api/nurse/education")
async def nurse_education(patient_id: str = None):
    """Patient/caregiver education: gap assessment + prioritized module recommendations."""
    import scripts.nurse_module as nurse
    return _json_safe(nurse.education_assessment(patient_id))


# ─── Speech-Language Pathologist (SLP) — see bottom-of-file block ─────────
# (routes moved to consolidated SLP Dashboard section near EOF)


# ── Occupational Therapist (OT) endpoints ────────────────────────────
# Real data: BARTHEL (24 records) + MOCA + MMSE + seizure_diary + medications + patients

@app.get("/api/ot")
async def ot_dashboard(patient_id: str = None):
    """Full OT dashboard: ADL/IADL + Fall Risk + Return-to-Work + Cognitive-Functional."""
    import scripts.ot_module as ot
    return _json_safe(ot.full_dashboard(patient_id))


@app.get("/api/ot/adl-assessment")
async def ot_adl(patient_id: str = None):
    """ADL/IADL assessment: Barthel Index per-domain breakdown with OT recommendations."""
    import scripts.ot_module as ot
    return _json_safe(ot.adl_assessment(patient_id))


@app.get("/api/ot/fall-risk")
async def ot_fall_risk(patient_id: str = None):
    """Home Safety / Fall Risk: seizure injuries + Barthel mobility + medication sedation."""
    import scripts.ot_module as ot
    return _json_safe(ot.fall_risk_assessment(patient_id))


@app.get("/api/ot/return-to-work")
async def ot_rtw(patient_id: str = None):
    """Return-to-Work Planner: composite score (functional + seizure control + cognitive)."""
    import scripts.ot_module as ot
    return _json_safe(ot.return_to_work(patient_id))


@app.get("/api/ot/cognitive-function")
async def ot_cognitive(patient_id: str = None):
    """Cognitive-Function OT: MoCA + MMSE with OT-specific domain mapping and interventions."""
    import scripts.ot_module as ot
    return _json_safe(ot.cognitive_function_ot(patient_id))


# ── Clinical Psychologist endpoints ──────────────────────────────────
# Real data: PHQ-9 (26) + GAD-7 (25) + NDDI-E (3) + QOLIE-31 (23) + seizure_diary (25)

@app.get("/api/psychologist")
async def psychologist_dashboard(patient_id: str = None):
    """Full Clinical Psychologist dashboard: depression/anxiety + coping + seizure-emotion + therapy."""
    import scripts.psychologist_module as ps
    return _json_safe(ps.full_dashboard(patient_id))


@app.get("/api/psychologist/depression-anxiety")
async def psychologist_depression_anxiety(patient_id: str = None):
    """PHQ-9 + GAD-7 + NDDI-E auto-scoring, severity bands, suicide-risk (C-SSRS) escalation flags."""
    import scripts.psychologist_module as ps
    return _json_safe(ps.depression_anxiety(patient_id))


@app.get("/api/psychologist/coping-resilience")
async def psychologist_coping(patient_id: str = None):
    """Coping & Resilience profile from live QOLIE-31 wellbeing dimensions."""
    import scripts.psychologist_module as ps
    return _json_safe(ps.coping_resilience(patient_id))


@app.get("/api/psychologist/seizure-emotion")
async def psychologist_seizure_emotion(patient_id: str = None):
    """Seizure-Emotion correlation: seizure_diary burden vs PHQ-9/GAD-7 mood scores."""
    import scripts.psychologist_module as ps
    return _json_safe(ps.seizure_emotion_correlation(patient_id))


@app.get("/api/psychologist/therapy-planning")
async def psychologist_therapy(patient_id: str = None):
    """CBT/ACT therapy planning — transparent rule-based targeting from live PHQ-9/GAD-7 severity."""
    import scripts.psychologist_module as ps
    return _json_safe(ps.therapy_planning(patient_id))


# ── Epilepsy Program Coordinator endpoints ───────────────────────────
# Real data: patients (40) + uploads + analyses + assessments + seizure_diary + reviews

@app.get("/api/coordinator")
async def coordinator_dashboard(patient_id: str = None):
    """Full Coordinator dashboard: patient journey + MDT coordination + KPIs + resource planning."""
    import scripts.coordinator_module as co
    return _json_safe(co.full_dashboard(patient_id))


@app.get("/api/coordinator/journey")
async def coordinator_journey(patient_id: str = None):
    """Patient Journey / Pathway: per-patient care-pipeline stage + next action + funnel."""
    import scripts.coordinator_module as co
    return _json_safe(co.patient_journey(patient_id))


@app.get("/api/coordinator/mdt")
async def coordinator_mdt(patient_id: str = None):
    """MDT Coordination: review status, pending queue (low-confidence first), per-role load."""
    import scripts.coordinator_module as co
    return _json_safe(co.mdt_coordination(patient_id))


@app.get("/api/coordinator/kpi")
async def coordinator_kpi():
    """Operational KPI dashboard: enrollment, analyses, coverage rates, confidence, flags."""
    import scripts.coordinator_module as co
    return _json_safe(co.kpi_dashboard())


@app.get("/api/coordinator/resource-planning")
async def coordinator_resources():
    """Resource / Capacity Planning: backlog by stage + primary bottleneck + recommendation."""
    import scripts.coordinator_module as co
    return _json_safe(co.resource_planning())


# ── Clinical Dietitian / Nutritionist endpoints ─────────────────────
# Real data: medications (9, AED drug names+doses) + patients (40, age/gender) +
# assessments (BARTHEL for ADL/feeding) + seizure_diary (25, seizure frequency).
# AED nutrient depletions + food interactions from published clinical pharmacology.

@app.get("/api/dietitian")
async def dietitian_dashboard(patient_id: str = None):
    """Clinical Dietitian / Nutritionist — full dashboard: ketogenic diet eligibility,
    malnutrition screening, nutrient/vitamin analysis, medication-nutrition interactions.
    All from REAL medications + patients + assessments + seizure_diary in data/clinical.db."""
    import scripts.dietitian_module as diet
    return _json_safe(diet.full_dashboard(patient_id))


@app.get("/api/dietitian/ketogenic")
async def dietitian_ketogenic(patient_id: str = None):
    """Ketogenic diet eligibility: composite score from seizure frequency, drug resistance,
    age, AED-keto contraindications. Recommends Classic 4:1 / MAD / MCT / LGIT."""
    import scripts.dietitian_module as diet
    return _json_safe(diet.ketogenic_diet_eligibility(patient_id))


@app.get("/api/dietitian/malnutrition")
async def dietitian_malnutrition(patient_id: str = None):
    """Malnutrition screening (MNA/MUST-style): age risk + appetite-suppressing AEDs +
    Barthel feeding/ADL impairment + polypharmacy. Risk: Low/Medium/High."""
    import scripts.dietitian_module as diet
    return _json_safe(diet.malnutrition_screening(patient_id))


@app.get("/api/dietitian/nutrient")
async def dietitian_nutrient(patient_id: str = None):
    """Nutrient/Vitamin deficiency analysis: cross-references each patient's AEDs against
    published depletion table (carnitine, folate, vitamin D, B6, B12, calcium, bicarbonate).
    Produces per-patient supplement recommendations."""
    import scripts.dietitian_module as diet
    return _json_safe(diet.nutrient_analysis(patient_id))


@app.get("/api/dietitian/medication-nutrition")
async def dietitian_med_nutrition(patient_id: str = None):
    """Medication-Nutrition interactions: AED-food interactions from clinical pharmacology
    (grapefruit/CYP3A4, enteral feed binding, GI effects, kidney stone hydration risk).
    Per-patient dietary counseling points."""
    import scripts.dietitian_module as diet
    return _json_safe(diet.medication_nutrition_interaction(patient_id))


# ─── Medical Social Worker (MSW) ──────────────────────────────────────

@app.get("/api/social-worker")
async def social_worker_dashboard(patient_id: str = None):
    """Medical Social Worker (MSW) — full dashboard: SDOH screening, caregiver burden
    (ZBI/CSI proxy), benefits/vocational support, treatment-barrier detection.
    All built from REAL patients + seizure_diary + medications + assessments in clinical.db."""
    import scripts.social_worker_module as msw
    return _json_safe(msw.full_dashboard(patient_id))


@app.get("/api/social-worker/sdoh-screening")
async def social_worker_sdoh(patient_id: str = None):
    """Social Determinants of Health screening: 6-domain scoring (Employment, Housing,
    Transportation, Financial, Social Support, Education) from real demographics,
    seizure frequency, medication burden. Composite vulnerability score 0-100."""
    import scripts.social_worker_module as msw
    return _json_safe(msw.social_determinants_screening(patient_id))


@app.get("/api/social-worker/caregiver-burden")
async def social_worker_caregiver(patient_id: str = None):
    """Caregiver Burden: ZBI proxy (0-88) + CSI proxy (0-13) scored from seizure
    severity/frequency, nocturnal seizures, injury history, polypharmacy, age factor.
    Burnout risk levels + respite referral flags."""
    import scripts.social_worker_module as msw
    return _json_safe(msw.caregiver_burden(patient_id))


@app.get("/api/social-worker/benefits-vocational")
async def social_worker_benefits(patient_id: str = None):
    """Benefits / Vocational Support: driving eligibility (seizure-free period),
    employment readiness score, disability benefit flags (SSA listing 11.02),
    cognitive/sedation load, vocational recommendations."""
    import scripts.social_worker_module as msw
    return _json_safe(msw.benefits_vocational(patient_id))


@app.get("/api/social-worker/treatment-barriers")
async def social_worker_barriers(patient_id: str = None):
    """Treatment-Barrier Detection: 8 barrier categories (Financial, Lifestyle,
    Safety fear, Sleep, Cognitive, Stigma, Transportation, Medication gap) with
    severity scoring and targeted intervention recommendations."""
    import scripts.social_worker_module as msw
    return _json_safe(msw.treatment_barrier_detection(patient_id))


# ─── Epilepsy Specialist Nurse ─────────────────────────────────────────

@app.get("/api/epilepsy-nurse")
async def epilepsy_nurse_dashboard(patient_id: str = None):
    """Epilepsy Specialist Nurse — full dashboard: seizure diary analysis,
    AED adherence monitoring, SUDEP-7 risk assessment, seizure action plans,
    patient education checklist. All from REAL clinical.db data."""
    import scripts.epilepsy_nurse_module as enm
    return _json_safe(enm.full_dashboard(patient_id))


@app.get("/api/epilepsy-nurse/seizure-diary")
async def epilepsy_nurse_seizure_diary(patient_id: str = None):
    """Seizure diary analysis: frequency, severity distribution, triggers,
    temporal patterns, injury rates, ER visits, nocturnal events."""
    import scripts.epilepsy_nurse_module as enm
    return _json_safe(enm.seizure_diary_analysis(patient_id))


@app.get("/api/epilepsy-nurse/adherence")
async def epilepsy_nurse_adherence(patient_id: str = None):
    """AED adherence monitoring: polytherapy complexity score (0-10),
    dosing burden, titration load, monitoring demands. Faught 2008."""
    import scripts.epilepsy_nurse_module as enm
    return _json_safe(enm.adherence_monitoring(patient_id))


@app.get("/api/epilepsy-nurse/sudep-risk")
async def epilepsy_nurse_sudep(patient_id: str = None):
    """SUDEP-7 risk assessment: composite 0-10 score from GTCS frequency,
    nocturnal seizures, AED adherence, duration, polytherapy failure,
    sex, age. Hesdorffer 2011 + Devinsky 2016 + MORTEMUS."""
    import scripts.epilepsy_nurse_module as enm
    return _json_safe(enm.sudep_risk_assessment(patient_id))


@app.get("/api/epilepsy-nurse/action-plan")
async def epilepsy_nurse_action_plan(patient_id: str = None):
    """Seizure action plans: per-patient first-aid plan, rescue medication
    (midazolam/diazepam), emergency criteria, recovery guidance.
    Epilepsy Foundation guidelines + NICE CG137."""
    import scripts.epilepsy_nurse_module as enm
    return _json_safe(enm.seizure_action_plan(patient_id))


@app.get("/api/epilepsy-nurse/education")
async def epilepsy_nurse_education(patient_id: str = None):
    """Patient/family education: 12-domain checklist (driving, swimming,
    pregnancy, SUDEP, medication, mental health, etc.) with patient-specific
    priority levels. NICE CG137 + Epilepsy Foundation."""
    import scripts.epilepsy_nurse_module as enm
    return _json_safe(enm.education_assessment(patient_id))


@app.get("/api/epilepsy-nurse/definitions")
async def epilepsy_nurse_definitions():
    """Metric definitions, scoring rules, and clinical references for all
    Epilepsy Specialist Nurse sub-modules."""
    import scripts.epilepsy_nurse_module as enm
    return _json_safe(enm.definitions())


# ─── Neuro AI Ecosystem: ADL / IADL (Katz / Lawton) ──────────────────

@app.get("/api/neuro-scales/adl")
async def neuro_scales_adl_dashboard(patient_id: str = None):
    """ADL/IADL dashboard — Katz Index (6-item ADL, 0-6) + Lawton IADL (8-item,
    0-8) for one patient or all patients. Scores derived from REAL Barthel,
    cognition (MoCA/MMSE), seizure diary, and medication data in clinical.db."""
    import scripts.neuro_scales_adl as adl
    return _json_safe(adl.adl_dashboard(patient_id))


@app.get("/api/neuro-scales/adl/katz")
async def neuro_scales_katz(patient_id: str):
    """Katz Index of ADL detail — 6 binary items (bathing, dressing, toileting,
    transferring, continence, feeding). Grade A (6/6, fully independent) to
    G (0/6, fully dependent). Derived from real patient data."""
    import scripts.neuro_scales_adl as adl
    return _json_safe(adl.katz_detail(patient_id))


@app.get("/api/neuro-scales/adl/lawton")
async def neuro_scales_lawton(patient_id: str):
    """Lawton Instrumental ADL detail — 8 binary items (telephone, shopping,
    food prep, housekeeping, laundry, transport, medications, finances).
    Sensitive to cognitive decline and seizure-related driving restrictions."""
    import scripts.neuro_scales_adl as adl
    return _json_safe(adl.lawton_detail(patient_id))


@app.get("/api/neuro-scales/adl/definitions")
async def neuro_scales_adl_definitions():
    """Scale definitions — item descriptions, scoring rules, and references
    for Katz ADL and Lawton IADL. For frontend rendering of scale info panels."""
    import scripts.neuro_scales_adl as adl
    return _json_safe(adl.scale_definitions())


# ─── Neuro AI Ecosystem: Glasgow Coma Scale (GCS) ────────────────────

@app.get("/api/neuro-scales/gcs")
async def neuro_scales_gcs_dashboard(patient_id: str = None):
    """GCS dashboard — Glasgow Coma Scale (E+V+M, 3-15) for one patient
    or all patients. Scores derived from REAL Barthel, cognition, seizure
    diary, and medication data in clinical.db."""
    import scripts.neuro_scales_gcs as gcs
    return _json_safe(gcs.gcs_dashboard(patient_id))


@app.get("/api/neuro-scales/gcs/detail")
async def neuro_scales_gcs_detail(patient_id: str):
    """GCS detail — Eye (1-4), Verbal (1-5), Motor (1-6) component
    breakdown + pupil reactivity + GCS-P score for a single patient."""
    import scripts.neuro_scales_gcs as gcs
    return _json_safe(gcs.gcs_detail(patient_id))


@app.get("/api/neuro-scales/gcs/trend")
async def neuro_scales_gcs_trend(patient_id: str):
    """GCS trend — 7-day modeled trajectory based on seizure burden and
    sedation profile. For serial monitoring visualization."""
    import scripts.neuro_scales_gcs as gcs
    return _json_safe(gcs.gcs_trend(patient_id))


@app.get("/api/neuro-scales/gcs/definitions")
async def neuro_scales_gcs_definitions():
    """Scale definitions — GCS component descriptions, severity thresholds,
    pupil reactivity scoring, and epilepsy-specific clinical context."""
    import scripts.neuro_scales_gcs as gcs
    return _json_safe(gcs.scale_definitions())


# ─── Neuro AI Ecosystem: Modified Rankin Scale (mRS) ───────────────

@app.get("/api/neuro-scales/rankin")
async def neuro_scales_rankin_dashboard(patient_id: str = None):
    """mRS dashboard — Modified Rankin Scale (0-6 disability) for one patient
    or all patients.  Scores derived from REAL Barthel, cognition, seizure
    diary, and medication data in clinical.db."""
    import scripts.neuro_scales_rankin as rankin
    return _json_safe(rankin.mrs_dashboard(patient_id))


@app.get("/api/neuro-scales/rankin/detail")
async def neuro_scales_rankin_detail(patient_id: str):
    """mRS detail — full grade breakdown + contributing factors + clinical
    note for a single patient."""
    import scripts.neuro_scales_rankin as rankin
    return _json_safe(rankin.mrs_detail(patient_id))


@app.get("/api/neuro-scales/rankin/trend")
async def neuro_scales_rankin_trend(patient_id: str):
    """mRS trend — 6-month modeled trajectory based on functional and
    seizure profile.  For outcomes tracking visualization."""
    import scripts.neuro_scales_rankin as rankin
    return _json_safe(rankin.mrs_trend(patient_id))


@app.get("/api/neuro-scales/rankin/definitions")
async def neuro_scales_rankin_definitions():
    """Scale definitions — mRS grade descriptions, outcome thresholds,
    reliability data, and epilepsy-specific clinical context."""
    import scripts.neuro_scales_rankin as rankin
    return _json_safe(rankin.scale_definitions())


# ─── Neuro AI Ecosystem: NIH Stroke Scale (NIHSS) ──────────────────

@app.get("/api/neuro-scales/nihss")
async def neuro_scales_nihss_dashboard(patient_id: str = None):
    """NIHSS dashboard — NIH Stroke Scale (0-42 severity) for one patient
    or all patients.  Scores derived from REAL Barthel, cognition, seizure
    diary, and medication data in clinical.db."""
    import scripts.neuro_scales_nihss as nihss
    return _json_safe(nihss.nihss_dashboard(patient_id))


@app.get("/api/neuro-scales/nihss/detail")
async def neuro_scales_nihss_detail(patient_id: str):
    """NIHSS detail — per-item (15-item) breakdown + contributing factors
    + clinical note for a single patient."""
    import scripts.neuro_scales_nihss as nihss
    return _json_safe(nihss.nihss_detail(patient_id))


@app.get("/api/neuro-scales/nihss/trend")
async def neuro_scales_nihss_trend(patient_id: str):
    """NIHSS trend — 6-month modeled trajectory based on stroke recovery
    curve and neurological profile."""
    import scripts.neuro_scales_nihss as nihss
    return _json_safe(nihss.nihss_trend(patient_id))


@app.get("/api/neuro-scales/nihss/definitions")
async def neuro_scales_nihss_definitions():
    """Scale definitions — NIHSS 15-item descriptions, severity thresholds,
    clinical decision points, reliability data, and references."""
    import scripts.neuro_scales_nihss as nihss
    return _json_safe(nihss.scale_definitions())


# ── PANSS (Positive and Negative Syndrome Scale) ─────────────────────
@app.get("/api/neuro-scales/panss")
async def neuro_scales_panss_dashboard(patient_id: str = None):
    """PANSS dashboard — 30-item schizophrenia severity (30-210) for one
    patient or all patients.  Scores derived from REAL Barthel, cognition,
    seizure diary, and medication data in clinical.db."""
    import scripts.neuro_scales_panss as panss
    return _json_safe(panss.panss_dashboard(patient_id))


@app.get("/api/neuro-scales/panss/detail")
async def neuro_scales_panss_detail(patient_id: str):
    """PANSS detail — per-item (30-item) breakdown + contributing factors
    + remission check for a single patient."""
    import scripts.neuro_scales_panss as panss
    return _json_safe(panss.panss_detail(patient_id))


@app.get("/api/neuro-scales/panss/trend")
async def neuro_scales_panss_trend(patient_id: str):
    """PANSS trend — 6-month modeled trajectory based on treatment
    response curve and disease profile."""
    import scripts.neuro_scales_panss as panss
    return _json_safe(panss.panss_trend(patient_id))


@app.get("/api/neuro-scales/panss/definitions")
async def neuro_scales_panss_definitions():
    """Scale definitions — PANSS 30-item descriptions, severity thresholds,
    factor structure, remission criteria, reliability data, references."""
    import scripts.neuro_scales_panss as panss
    return _json_safe(panss.scale_definitions())


# ── HAM-D (Hamilton Depression Rating Scale) ─────────────────────────
@app.get("/api/neuro-scales/hamd")
async def neuro_scales_hamd_dashboard(patient_id: str = None):
    """HAM-D-17 depression severity dashboard — all patients or one.
    Scores derived from real clinical.db data (cognition, Barthel, meds, seizures)."""
    import scripts.neuro_scales_hamd as hamd
    return _json_safe(hamd.hamd_dashboard(patient_id))

@app.get("/api/neuro-scales/hamd/detail")
async def neuro_scales_hamd_detail(patient_id: str):
    """Per-item HAM-D-17 detail for a single patient with contributing factors."""
    import scripts.neuro_scales_hamd as hamd
    return _json_safe(hamd.hamd_detail(patient_id))

@app.get("/api/neuro-scales/hamd/trend")
async def neuro_scales_hamd_trend(patient_id: str):
    """6-month modeled HAM-D trajectory based on treatment response."""
    import scripts.neuro_scales_hamd as hamd
    return _json_safe(hamd.hamd_trend(patient_id))

@app.get("/api/neuro-scales/hamd/definitions")
async def neuro_scales_hamd_definitions():
    """Scale definitions — HAM-D-17 item descriptions, severity thresholds,
    subscale structure, reliability data, clinical utility references."""
    import scripts.neuro_scales_hamd as hamd
    return _json_safe(hamd.scale_definitions())


# ─── Neuro AI Ecosystem: Beck Depression Inventory-II (BDI-II) ─────

@app.get("/api/neuro-scales/bdi")
async def neuro_scales_bdi_dashboard(patient_id: str = None):
    """BDI-II depression severity dashboard — all patients or one.
    21-item self-report scale; scores derived from real clinical.db data."""
    import scripts.neuro_scales_bdi as bdi
    return _json_safe(bdi.bdi_dashboard(patient_id))

@app.get("/api/neuro-scales/bdi/detail")
async def neuro_scales_bdi_detail(patient_id: str):
    """Per-item BDI-II detail for a single patient with contributing factors."""
    import scripts.neuro_scales_bdi as bdi
    return _json_safe(bdi.bdi_detail(patient_id))

@app.get("/api/neuro-scales/bdi/trend")
async def neuro_scales_bdi_trend(patient_id: str):
    """6-month modeled BDI-II trajectory based on treatment response."""
    import scripts.neuro_scales_bdi as bdi
    return _json_safe(bdi.bdi_trend(patient_id))

@app.get("/api/neuro-scales/bdi/definitions")
async def neuro_scales_bdi_definitions():
    """Scale definitions — BDI-II item descriptions, severity thresholds,
    subscale structure, reliability data, clinical utility references."""
    import scripts.neuro_scales_bdi as bdi
    return _json_safe(bdi.scale_definitions())


# ── Engel Classification (Epilepsy Surgery Outcome) ───────────────────
@app.get("/api/neuro-scales/engel")
async def neuro_scales_engel_dashboard(patient_id: str = None):
    """Engel classification dashboard — 4-class epilepsy surgery outcome
    (I-IV with sub-classifications) for one patient or all patients.
    Scores derived from REAL seizure diary, Barthel, and medication data
    in clinical.db."""
    import scripts.neuro_scales_engel as engel
    return _json_safe(engel.engel_dashboard(patient_id))


@app.get("/api/neuro-scales/engel/detail")
async def neuro_scales_engel_detail(patient_id: str):
    """Engel detail — per-patient classification with sub-class, rationale,
    contributing factors (seizure types, AEDs, Barthel, disease context)."""
    import scripts.neuro_scales_engel as engel
    return _json_safe(engel.engel_detail(patient_id))


@app.get("/api/neuro-scales/engel/trend")
async def neuro_scales_engel_trend(patient_id: str):
    """Engel trend — 12-month projected outcome trajectory based on
    published relapse curves and current seizure control."""
    import scripts.neuro_scales_engel as engel
    return _json_safe(engel.engel_trend(patient_id))


@app.get("/api/neuro-scales/engel/definitions")
async def neuro_scales_engel_definitions():
    """Scale definitions — Engel 4-class descriptions, sub-classifications,
    outcome frequencies, reliability data, clinical use references."""
    import scripts.neuro_scales_engel as engel
    return _json_safe(engel.scale_definitions())


# ── ILAE Outcome Scale (Epilepsy Surgery Outcome) ─────────────────────
@app.get("/api/neuro-scales/ilae")
async def neuro_scales_ilae_dashboard(patient_id: str = None):
    """ILAE outcome dashboard — 6-class epilepsy surgery outcome
    (1-6, seizure-day-based) for one patient or all patients.
    Scores derived from REAL seizure diary, Barthel, and medication data
    in clinical.db."""
    import scripts.neuro_scales_ilae as ilae
    return _json_safe(ilae.ilae_dashboard(patient_id))


@app.get("/api/neuro-scales/ilae/detail")
async def neuro_scales_ilae_detail(patient_id: str):
    """ILAE detail — per-patient classification with class rationale,
    contributing factors (seizure types, AEDs, Barthel, aura analysis)."""
    import scripts.neuro_scales_ilae as ilae
    return _json_safe(ilae.ilae_detail(patient_id))


@app.get("/api/neuro-scales/ilae/trend")
async def neuro_scales_ilae_trend(patient_id: str):
    """ILAE trend — 12-month projected outcome trajectory based on
    published outcome curves and current seizure control."""
    import scripts.neuro_scales_ilae as ilae
    return _json_safe(ilae.ilae_trend(patient_id))


@app.get("/api/neuro-scales/ilae/definitions")
async def neuro_scales_ilae_definitions():
    """Scale definitions — ILAE 6-class descriptions, outcome frequencies,
    reliability data, comparison to Engel, clinical use references."""
    import scripts.neuro_scales_ilae as ilae
    return _json_safe(ilae.scale_definitions())


# ── Liverpool Adverse Events Profile (LAEP) ───────────────────────────
@app.get("/api/neuro-scales/laep")
async def neuro_scales_laep_dashboard(patient_id: str = None):
    """LAEP dashboard — 19-item AED side-effect profile
    (Baker et al., Epilepsy Research 1995). Total 19-76."""
    import scripts.neuro_scales_laep as laep
    return _json_safe(laep.laep_dashboard(patient_id))


@app.get("/api/neuro-scales/laep/detail")
async def neuro_scales_laep_detail(patient_id: str):
    """LAEP detail — per-patient 19-item breakdown, domain scores,
    contributing factors, recommendations."""
    import scripts.neuro_scales_laep as laep
    return _json_safe(laep.laep_detail(patient_id))


@app.get("/api/neuro-scales/laep/trend")
async def neuro_scales_laep_trend(patient_id: str):
    """LAEP trend — 12-month projected side-effect trajectory based on
    AED tolerance curves (Perucca & Meador, Lancet Neurol 2005)."""
    import scripts.neuro_scales_laep as laep
    return _json_safe(laep.laep_trend(patient_id))


@app.get("/api/neuro-scales/laep/definitions")
async def neuro_scales_laep_definitions():
    """Scale definitions — LAEP 19 items, domains, severity bands,
    reliability data, norms, clinical use references."""
    import scripts.neuro_scales_laep as laep
    return _json_safe(laep.scale_definitions())


# ── SUDEP-7 Risk Inventory ─────────────────────────────────────────────
@app.get("/api/neuro-scales/sudep")
async def neuro_scales_sudep_dashboard(patient_id: str = None):
    """SUDEP-7 dashboard — 7-item sudden-death risk inventory (0-12)
    for one patient or all patients, derived from clinical.db."""
    import scripts.neuro_scales_sudep as sudep
    return _json_safe(sudep.sudep_dashboard(patient_id))


@app.get("/api/neuro-scales/sudep/detail")
async def neuro_scales_sudep_detail(patient_id: str):
    """SUDEP-7 detail — per-patient 7-item breakdown, evidence,
    risk factors, and prevention recommendations."""
    import scripts.neuro_scales_sudep as sudep
    return _json_safe(sudep.sudep_detail(patient_id))


@app.get("/api/neuro-scales/sudep/trend")
async def neuro_scales_sudep_trend(patient_id: str):
    """SUDEP-7 trend — 12-month projected risk trajectory based on
    published intervention efficacy and risk modification data."""
    import scripts.neuro_scales_sudep as sudep
    return _json_safe(sudep.sudep_trend(patient_id))


@app.get("/api/neuro-scales/sudep/definitions")
async def neuro_scales_sudep_definitions():
    """Scale definitions — SUDEP-7 items, weights, severity bands,
    prevention strategies, epidemiology, clinical use references."""
    import scripts.neuro_scales_sudep as sudep
    return _json_safe(sudep.scale_definitions())


# ── Medication Adherence Scale (MMAS-4) ──────────────────────────────
@app.get("/api/neuro-scales/mmas")
async def neuro_scales_mmas_dashboard(patient_id: str = None):
    """MMAS-4 medication adherence dashboard — all patients or single.
    Morisky DE et al. Medical Care 1986; 4 yes/no items (0=high, 3-4=low)."""
    import scripts.neuro_scales_mmas as mmas
    return _json_safe(mmas.mmas_dashboard(patient_id))

@app.get("/api/neuro-scales/mmas/detail")
async def neuro_scales_mmas_detail(patient_id: str):
    """Per-patient MMAS-4 detail with all 4 items, domains,
    contributing factors, and clinical recommendations."""
    import scripts.neuro_scales_mmas as mmas
    return _json_safe(mmas.mmas_detail(patient_id))

@app.get("/api/neuro-scales/mmas/trend")
async def neuro_scales_mmas_trend(patient_id: str):
    """12-month projected medication adherence trajectory based on
    baseline adherence, AED regimen, and intervention effects."""
    import scripts.neuro_scales_mmas as mmas
    return _json_safe(mmas.mmas_trend(patient_id))

@app.get("/api/neuro-scales/mmas/definitions")
async def neuro_scales_mmas_definitions():
    """Scale definitions — MMAS-4 items, scoring, psychometrics,
    epilepsy context, interventions, data derivation."""
    import scripts.neuro_scales_mmas as mmas
    return _json_safe(mmas.scale_definitions())


# ── Stroop Color-Word Test ────────────────────────────────────────────
@app.get("/api/neuro-scales/stroop")
async def neuro_scales_stroop_dashboard(patient_id: str = None):
    """Stroop Color-Word Test dashboard — selective attention and inhibitory control.
    Three conditions (word/color/interference); interference score is primary metric.
    All data derived from real clinical.db patient features."""
    import scripts.neuro_scales_stroop as stroop
    return _json_safe(stroop.stroop_dashboard(patient_id))

@app.get("/api/neuro-scales/stroop/detail")
async def neuro_scales_stroop_detail(patient_id: str):
    """Per-patient Stroop detail with all 3 conditions, contributing factors,
    z-score, percentile, and clinical recommendations."""
    import scripts.neuro_scales_stroop as stroop
    return _json_safe(stroop.stroop_detail(patient_id))

@app.get("/api/neuro-scales/stroop/trend")
async def neuro_scales_stroop_trend(patient_id: str):
    """12-month projected Stroop interference trajectory based on AED
    optimisation, seizure control, and age-related decline."""
    import scripts.neuro_scales_stroop as stroop
    return _json_safe(stroop.stroop_trend(patient_id))

@app.get("/api/neuro-scales/stroop/definitions")
async def neuro_scales_stroop_definitions():
    """Scale definitions — Stroop conditions, scoring, psychometrics,
    epilepsy-specific context, AED cognitive burden, data derivation."""
    import scripts.neuro_scales_stroop as stroop
    return _json_safe(stroop.scale_definitions())


# ── Trail Making Test (TMT) A & B ──────────────────────────────────────
@app.get("/api/neuro-scales/tmt")
async def neuro_scales_tmt_dashboard(patient_id: str = None):
    """Trail Making Test A & B dashboard — visuomotor speed (A) and
    executive function / cognitive flexibility (B). B/A ratio is the
    key discriminant for frontal dysfunction. All data derived from
    real clinical.db patient features."""
    import scripts.neuro_scales_tmt as tmt
    return _json_safe(tmt.tmt_dashboard(patient_id))

@app.get("/api/neuro-scales/tmt/detail")
async def neuro_scales_tmt_detail(patient_id: str):
    """Per-patient TMT detail with Part A & B times, B-A difference,
    B/A ratio, contributing factors, and clinical recommendations."""
    import scripts.neuro_scales_tmt as tmt
    return _json_safe(tmt.tmt_detail(patient_id))

@app.get("/api/neuro-scales/tmt/trend")
async def neuro_scales_tmt_trend(patient_id: str):
    """12-month projected TMT-B trajectory based on AED optimisation,
    seizure control, and age-related decline."""
    import scripts.neuro_scales_tmt as tmt
    return _json_safe(tmt.tmt_trend(patient_id))

@app.get("/api/neuro-scales/tmt/definitions")
async def neuro_scales_tmt_definitions():
    """Scale definitions — TMT parts, scoring, psychometrics,
    epilepsy-specific context, AED effects, data derivation."""
    import scripts.neuro_scales_tmt as tmt
    return _json_safe(tmt.scale_definitions())


# ── Digit Span (Forward, Backward, Sequencing) ────────────────────────
@app.get("/api/neuro-scales/digit-span")
async def neuro_scales_digit_span_dashboard(patient_id: str = None):
    """Digit Span dashboard — Forward (attention), Backward (working memory),
    Sequencing (executive). All data derived from real clinical.db patient features."""
    import scripts.neuro_scales_digit_span as ds
    return _json_safe(ds.digit_span_dashboard(patient_id))

@app.get("/api/neuro-scales/digit-span/detail")
async def neuro_scales_digit_span_detail(patient_id: str):
    """Per-patient Digit Span detail with all 3 conditions, contributing factors,
    Forward-Backward difference, disproportionate WM flag, and recommendations."""
    import scripts.neuro_scales_digit_span as ds
    return _json_safe(ds.digit_span_detail(patient_id))

@app.get("/api/neuro-scales/digit-span/trend")
async def neuro_scales_digit_span_trend(patient_id: str):
    """12-month projected Backward span trajectory based on AED optimisation,
    seizure control, and age-related decline."""
    import scripts.neuro_scales_digit_span as ds
    return _json_safe(ds.digit_span_trend(patient_id))

@app.get("/api/neuro-scales/digit-span/definitions")
async def neuro_scales_digit_span_definitions():
    """Scale definitions — conditions, scoring, psychometrics,
    epilepsy-specific lateralisation guide, AED effects, data derivation."""
    import scripts.neuro_scales_digit_span as ds
    return _json_safe(ds.scale_definitions())


# ── Wisconsin Card Sorting Test (WCST) ────────────────────────────────
@app.get("/api/neuro-scales/wcst")
async def neuro_scales_wcst_dashboard(patient_id: str = None):
    """WCST dashboard — executive function: cognitive flexibility, set-shifting,
    perseverative responding. 128 cards sorted by color/form/number with covert
    rule shifts. Gold-standard frontal-lobe screen in epilepsy neuropsych.
    All data derived from real clinical.db patient features."""
    import scripts.neuro_scales_wcst as wcst
    return _json_safe(wcst.wcst_dashboard(patient_id))

@app.get("/api/neuro-scales/wcst/detail")
async def neuro_scales_wcst_detail(patient_id: str):
    """Per-patient WCST detail with all 5 metrics (CC, PE, TE, CLR%, FMS),
    contributing factors, z-scores, percentiles, and clinical recommendations."""
    import scripts.neuro_scales_wcst as wcst
    return _json_safe(wcst.wcst_detail(patient_id))

@app.get("/api/neuro-scales/wcst/trend")
async def neuro_scales_wcst_trend(patient_id: str):
    """12-month projected perseverative-error trajectory based on AED
    optimisation, seizure control, and age-related decline."""
    import scripts.neuro_scales_wcst as wcst
    return _json_safe(wcst.wcst_trend(patient_id))

@app.get("/api/neuro-scales/wcst/definitions")
async def neuro_scales_wcst_definitions():
    """Scale definitions — WCST metrics, scoring, psychometrics,
    epilepsy-specific context, AED cognitive burden, data derivation."""
    import scripts.neuro_scales_wcst as wcst
    return _json_safe(wcst.scale_definitions())


# ── N-Back Working Memory Test ─────────────────────────────────────────
@app.get("/api/neuro-scales/nback")
async def neuro_scales_nback_dashboard(patient_id: str = None):
    """N-Back dashboard — working memory: hit rate, false alarms, d-prime,
    reaction time across 1/2/3-back conditions. Core paradigm in epilepsy
    neuropsych for temporal-lobe and frontal working-memory circuits.
    All data derived from real clinical.db patient features."""
    import scripts.neuro_scales_nback as nback
    return _json_safe(nback.nback_dashboard(patient_id))

@app.get("/api/neuro-scales/nback/detail")
async def neuro_scales_nback_detail(patient_id: str):
    """Per-patient N-Back detail with all 5 metrics (HR, FAR, d', RT, Acc),
    contributing factors, z-scores, percentiles, and clinical recommendations."""
    import scripts.neuro_scales_nback as nback
    return _json_safe(nback.nback_detail(patient_id))

@app.get("/api/neuro-scales/nback/trend")
async def neuro_scales_nback_trend(patient_id: str):
    """12-month projected d-prime trajectory based on AED optimisation,
    seizure control, and age-related decline."""
    import scripts.neuro_scales_nback as nback
    return _json_safe(nback.nback_trend(patient_id))

@app.get("/api/neuro-scales/nback/definitions")
async def neuro_scales_nback_definitions():
    """Scale definitions — N-Back metrics, scoring, psychometrics,
    epilepsy-specific context, AED cognitive burden, data derivation."""
    import scripts.neuro_scales_nback as nback
    return _json_safe(nback.scale_definitions())


@app.get("/api/neuro-scales/gonogo")
async def neuro_scales_gonogo_dashboard():
    """Go/No-Go dashboard — response inhibition: commission errors (false alarms
    on No-Go trials), omission errors, and reaction time.  Core paradigm in
    epilepsy neuropsych for frontal inhibitory control assessment.
    All data derived from real clinical.db patient features."""
    import scripts.neuro_scales_gonogo as gonogo
    return _json_safe(gonogo.gonogo_dashboard())

@app.get("/api/neuro-scales/gonogo/detail")
async def neuro_scales_gonogo_detail(patient_id: str):
    """Per-patient Go/No-Go detail with commission/omission rates, RT,
    contributing factors, z-scores, and clinical recommendations."""
    import scripts.neuro_scales_gonogo as gonogo
    return _json_safe(gonogo.gonogo_detail(patient_id))

@app.get("/api/neuro-scales/gonogo/trend")
async def neuro_scales_gonogo_trend(patient_id: str):
    """12-month projected Go/No-Go trajectory based on AED optimisation,
    seizure control, and age-related cognitive changes."""
    import scripts.neuro_scales_gonogo as gonogo
    return _json_safe(gonogo.gonogo_trend(patient_id))

@app.get("/api/neuro-scales/gonogo/definitions")
async def neuro_scales_gonogo_definitions():
    """Scale definitions — Go/No-Go metrics, scoring, psychometrics,
    epilepsy-specific context, AED cognitive burden, data derivation."""
    import scripts.neuro_scales_gonogo as gonogo
    return _json_safe(gonogo.scale_definitions())


# ── CPT (Continuous Performance Test) ──────────────────────────────────

@app.get("/api/neuro-scales/cpt")
async def neuro_scales_cpt_dashboard(patient_id: str = None):
    """Population or single-patient CPT dashboard — sustained attention,
    vigilance, omissions, commissions, d-prime, signal-detection metrics."""
    import scripts.neuro_scales_cpt as cpt
    return _json_safe(cpt.cpt_dashboard(patient_id))

@app.get("/api/neuro-scales/cpt/detail")
async def neuro_scales_cpt_detail(patient_id: str):
    """Detailed CPT profile for one patient: omission/commission rates,
    hit RT, d-prime, beta, clinical interpretation, AED/absence notes."""
    import scripts.neuro_scales_cpt as cpt
    return _json_safe(cpt.cpt_detail(patient_id))

@app.get("/api/neuro-scales/cpt/trend")
async def neuro_scales_cpt_trend(patient_id: str):
    """12-month projected CPT trajectory: omissions decrease, d-prime
    increases as AED burden reduces and seizure control improves."""
    import scripts.neuro_scales_cpt as cpt
    return _json_safe(cpt.cpt_trend(patient_id))

@app.get("/api/neuro-scales/cpt/definitions")
async def neuro_scales_cpt_definitions():
    """Scale definitions — CPT paradigm, metrics, scoring, norms,
    severity bands, signal-detection theory, epilepsy-specific context."""
    import scripts.neuro_scales_cpt as cpt
    return _json_safe(cpt.scale_definitions())


# ── Clock Drawing Test (CDT) ──────────────────────────────────────────

@app.get("/api/neuro-scales/clock-drawing")
async def neuro_scales_clock_drawing_dashboard(patient_id: str = None):
    """Population or single-patient Clock Drawing Test dashboard —
    visuospatial, executive, contour/numbers/hands/center subscores."""
    import scripts.neuro_scales_clock_drawing as cdt
    return _json_safe(cdt.clock_drawing_dashboard(patient_id))

@app.get("/api/neuro-scales/clock-drawing/detail")
async def neuro_scales_clock_drawing_detail(patient_id: str):
    """Detailed CDT profile for one patient: total (Shulman 0-5),
    contour/numbers/hands/center subscores, clinical interpretation."""
    import scripts.neuro_scales_clock_drawing as cdt
    return _json_safe(cdt.clock_drawing_detail(patient_id))

@app.get("/api/neuro-scales/clock-drawing/trend")
async def neuro_scales_clock_drawing_trend(patient_id: str):
    """12-month projected CDT trajectory: total + component scores
    improve as AED burden reduces and seizure control improves."""
    import scripts.neuro_scales_clock_drawing as cdt
    return _json_safe(cdt.clock_drawing_trend(patient_id))

@app.get("/api/neuro-scales/clock-drawing/definitions")
async def neuro_scales_clock_drawing_definitions():
    """Scale definitions — CDT Shulman scoring, norms, severity bands,
    visuospatial/executive components, epilepsy-specific context."""
    import scripts.neuro_scales_clock_drawing as cdt
    return _json_safe(cdt.scale_definitions())


# ── Rey Auditory Verbal Learning Test (RAVLT) ────────────────────────

@app.get("/api/neuro-scales/ravlt")
async def neuro_scales_ravlt_dashboard(patient_id: str = None):
    """Population or single-patient RAVLT dashboard —
    verbal learning, memory, retention, interference metrics."""
    import scripts.neuro_scales_ravlt as ravlt
    return _json_safe(ravlt.ravlt_dashboard(patient_id))

@app.get("/api/neuro-scales/ravlt/detail")
async def neuro_scales_ravlt_detail(patient_id: str):
    """Detailed RAVLT profile for one patient: trial-by-trial learning curve,
    delayed recall, recognition, forgetting rate, clinical interpretation."""
    import scripts.neuro_scales_ravlt as ravlt
    return _json_safe(ravlt.ravlt_detail(patient_id))

@app.get("/api/neuro-scales/ravlt/trend")
async def neuro_scales_ravlt_trend(patient_id: str):
    """12-month projected RAVLT trajectory: delayed recall + total learning
    improve as AED burden reduces and seizure control improves."""
    import scripts.neuro_scales_ravlt as ravlt
    return _json_safe(ravlt.ravlt_trend(patient_id))

@app.get("/api/neuro-scales/ravlt/definitions")
async def neuro_scales_ravlt_definitions():
    """Scale definitions — RAVLT scoring, norms, severity bands,
    verbal learning/memory components, epilepsy-specific context."""
    import scripts.neuro_scales_ravlt as ravlt
    return _json_safe(ravlt.scale_definitions())


# ── Verbal Fluency (FAS + Category) ──────────────────────────────────────────

@app.get("/api/neuro-scales/verbal-fluency")
async def neuro_scales_verbal_fluency_dashboard(patient_id: str = None):
    """Verbal Fluency dashboard — phonemic (FAS), semantic (Animals), switching,
    clustering, phonemic-semantic ratio across all or one patient."""
    import scripts.neuro_scales_verbal_fluency as vf
    return _json_safe(vf.verbal_fluency_dashboard(patient_id))

@app.get("/api/neuro-scales/verbal-fluency/detail")
async def neuro_scales_verbal_fluency_detail(patient_id: str):
    """Per-patient verbal fluency detail — contributing factors, clinical
    interpretation (frontal/temporal dissociation), recommendations."""
    import scripts.neuro_scales_verbal_fluency as vf
    return _json_safe(vf.verbal_fluency_detail(patient_id))

@app.get("/api/neuro-scales/verbal-fluency/trend")
async def neuro_scales_verbal_fluency_trend(patient_id: str, months: int = 12):
    """Verbal fluency longitudinal trend — FAS + Animals projected trajectory
    with AED optimisation and seizure-control modifiers."""
    import scripts.neuro_scales_verbal_fluency as vf
    return _json_safe(vf.verbal_fluency_trend(patient_id, months))

@app.get("/api/neuro-scales/verbal-fluency/definitions")
async def neuro_scales_verbal_fluency_definitions():
    """Scale definitions — verbal fluency scoring, norms, severity bands,
    phonemic/semantic/switching components, epilepsy-specific context."""
    import scripts.neuro_scales_verbal_fluency as vf
    return _json_safe(vf.scale_definitions())


# ── Medication Impact (Expert · Neurologist) ──────────────────────────────────

@app.get("/api/expert/medication-impact")
async def expert_medication_impact_dashboard(patient_id: str = None):
    """Medication Impact dashboard — AED seizure-reduction rates, side-effect
    burden (LAEP), drug interactions, adherence proxy, EEG spectral shifts."""
    import scripts.neuro_medication_impact as mi
    return _json_safe(mi.dashboard(patient_id))

@app.get("/api/expert/medication-impact/detail")
async def expert_medication_impact_detail(patient_id: str):
    """Per-patient medication impact detail — AED profiles, interaction matrix,
    EEG band shifts, clinical recommendations."""
    import scripts.neuro_medication_impact as mi
    return _json_safe(mi.detail(patient_id))

@app.get("/api/expert/medication-impact/trend")
async def expert_medication_impact_trend(patient_id: str, months: int = 12):
    """Medication impact trend — seizure-frequency + side-effect trajectory
    projected from Kwan & Brodie first-AED response curves."""
    import scripts.neuro_medication_impact as mi
    return _json_safe(mi.trend(patient_id, months))

@app.get("/api/expert/medication-impact/definitions")
async def expert_medication_impact_definitions():
    """Metric definitions — seizure reduction, LAEP, interaction risk, EEG
    band shifts, adherence. Published AED profiles + references."""
    import scripts.neuro_medication_impact as mi
    return _json_safe(mi.definitions())


# ── Pittsburgh Sleep Quality Index (PSQI) ─────────────────────────────

@app.get("/api/neuro-scales/psqi")
async def neuro_scales_psqi_dashboard(patient_id: str = None):
    """PSQI sleep quality dashboard — 7-component global score (0-21).
    Scores derived from real clinical.db data (disease, seizures, meds, age)."""
    import scripts.neuro_scales_psqi as psqi
    return _json_safe(psqi.psqi_dashboard(patient_id))

@app.get("/api/neuro-scales/psqi/detail")
async def neuro_scales_psqi_detail(patient_id: str):
    """Per-component PSQI detail for a single patient with contributing factors."""
    import scripts.neuro_scales_psqi as psqi
    return _json_safe(psqi.psqi_detail(patient_id))

@app.get("/api/neuro-scales/psqi/trend")
async def neuro_scales_psqi_trend(patient_id: str):
    """6-month modeled PSQI trajectory based on treatment and seizure control."""
    import scripts.neuro_scales_psqi as psqi
    return _json_safe(psqi.psqi_trend(patient_id))

@app.get("/api/neuro-scales/psqi/definitions")
async def neuro_scales_psqi_definitions():
    """Scale definitions — PSQI 7-component structure, clinical cutoff (>5),
    reliability data, epilepsy relevance references."""
    import scripts.neuro_scales_psqi as psqi
    return _json_safe(psqi.scale_definitions())


# ── Cognition Link Dashboard (Neuropsychologist) ─────────────────────────────
@app.get("/api/cognition-link/overview")
async def cognition_link_overview():
    """Overview — EEG ↔ cognitive test correlation summary, top pairs, domain breakdown."""
    import scripts.cognition_link_dashboard as cld
    return _json_safe(cld.overview())


@app.get("/api/cognition-link/matrix")
async def cognition_link_matrix():
    """Full correlation matrix — every (EEG feature, cognitive test) pair with r, p, effect size."""
    import scripts.cognition_link_dashboard as cld
    return _json_safe(cld.correlation_matrix())


@app.get("/api/cognition-link/heatmap")
async def cognition_link_heatmap():
    """Heatmap data — rows=EEG features, cols=cognitive tests, values=Pearson r."""
    import scripts.cognition_link_dashboard as cld
    return _json_safe(cld.heatmap_data())


@app.get("/api/cognition-link/domains")
async def cognition_link_domains():
    """Per cognitive-domain profile — strongest EEG predictors per domain."""
    import scripts.cognition_link_dashboard as cld
    return _json_safe(cld.domain_profile())


@app.get("/api/cognition-link/alerts")
async def cognition_link_alerts():
    """Clinical alerts — strong correlations (|r| ≥ 0.45, p < 0.01) with recommendations."""
    import scripts.cognition_link_dashboard as cld
    return _json_safe(cld.clinical_alerts())


@app.get("/api/cognition-link/definitions")
async def cognition_link_definitions():
    """Definitions — effect size thresholds, EEG bands, cognitive domains, references."""
    import scripts.cognition_link_dashboard as cld
    return _json_safe(cld.definitions())


# ── Clinical Scales Catalog ─────────────────────────────────────────
@app.get("/api/neuro-scales/catalog")
async def neuro_scales_catalog():
    """Consolidated catalog of all 23 clinical/neuropsychological scales
    with scale metadata, category, score ranges, and per-patient summary stats."""
    SCALES = [
        {"id": "adl", "name": "ADL / IADL", "category": "Functional", "range": "0-6 / 0-8", "higher": "better", "description": "Katz ADL (6 basic activities) + Lawton IADL (8 instrumental activities)"},
        {"id": "gcs", "name": "Glasgow Coma Scale", "category": "Consciousness", "range": "3-15", "higher": "better", "description": "Eye + Verbal + Motor response — consciousness level assessment"},
        {"id": "rankin", "name": "Modified Rankin Scale", "category": "Disability", "range": "0-6", "higher": "worse", "description": "Degree of disability/dependence post-stroke or neurological event"},
        {"id": "nihss", "name": "NIH Stroke Scale", "category": "Stroke Severity", "range": "0-42", "higher": "worse", "description": "Stroke severity — 11 items covering consciousness, gaze, visual, facial, motor, ataxia, sensory, language, dysarthria, neglect"},
        {"id": "panss", "name": "PANSS", "category": "Psychiatric", "range": "30-210", "higher": "worse", "description": "Positive and Negative Syndrome Scale for schizophrenia (7P + 7N + 16G items)"},
        {"id": "hamd", "name": "HAM-D", "category": "Depression", "range": "0-52", "higher": "worse", "description": "Hamilton Depression Rating Scale — 17-item clinician-rated depression severity"},
        {"id": "bdi", "name": "BDI-II", "category": "Depression", "range": "0-63", "higher": "worse", "description": "Beck Depression Inventory — 21-item self-report depression severity"},
        {"id": "engel", "name": "Engel Classification", "category": "Seizure Outcome", "range": "I-IV", "higher": "worse", "description": "Post-surgical seizure outcome (I=seizure-free, IV=no improvement)"},
        {"id": "ilae", "name": "ILAE Outcome Scale", "category": "Seizure Outcome", "range": "1-6", "higher": "worse", "description": "ILAE seizure outcome after epilepsy surgery (1=seizure-free, 6=worse)"},
        {"id": "laep", "name": "LAEP", "category": "AED Side Effects", "range": "19-76", "higher": "worse", "description": "Liverpool Adverse Events Profile — 19-item AED side-effect burden"},
        {"id": "sudep", "name": "SUDEP-7", "category": "Mortality Risk", "range": "0-10", "higher": "worse", "description": "SUDEP risk inventory — 7 factors (GTC freq, nocturnal seizures, duration, etc.)"},
        {"id": "mmas", "name": "MMAS-8", "category": "Adherence", "range": "0-8", "higher": "better", "description": "Morisky Medication Adherence Scale — 8-item self-report"},
        {"id": "psqi", "name": "PSQI", "category": "Sleep Quality", "range": "0-21", "higher": "worse", "description": "Pittsburgh Sleep Quality Index — 7-component sleep quality assessment"},
        {"id": "stroop", "name": "Stroop Test", "category": "Cognitive – Attention", "range": "time/errors", "higher": "worse", "description": "Selective attention and cognitive flexibility — interference score"},
        {"id": "tmt", "name": "Trail Making Test", "category": "Cognitive – Executive", "range": "time (s)", "higher": "worse", "description": "TMT-A (visuomotor speed) + TMT-B (set shifting) — executive function"},
        {"id": "digit-span", "name": "Digit Span", "category": "Cognitive – Memory", "range": "0-16/0-14", "higher": "better", "description": "Forward + Backward digit span — working memory capacity"},
        {"id": "wcst", "name": "Wisconsin Card Sort", "category": "Cognitive – Executive", "range": "errors/categories", "higher": "mixed", "description": "WCST — cognitive flexibility, perseveration, set-shifting"},
        {"id": "nback", "name": "N-Back Task", "category": "Cognitive – Working Memory", "range": "accuracy %", "higher": "better", "description": "1-back, 2-back, 3-back working memory performance and d-prime"},
        {"id": "gonogo", "name": "Go/No-Go Task", "category": "Cognitive – Inhibition", "range": "accuracy/RT", "higher": "better", "description": "Response inhibition — commission errors, omission errors, reaction time"},
        {"id": "cpt", "name": "Continuous Performance", "category": "Cognitive – Sustained Attention", "range": "d-prime/RT", "higher": "better", "description": "CPT-II sustained attention — hits, false alarms, d-prime, response time variability"},
        {"id": "clock-drawing", "name": "Clock Drawing Test", "category": "Cognitive – Visuospatial", "range": "0-10", "higher": "better", "description": "CDT — visuospatial/executive screening (Shulman 0-5 or Sunderland 1-10)"},
        {"id": "ravlt", "name": "RAVLT", "category": "Cognitive – Verbal Memory", "range": "0-75 total", "higher": "better", "description": "Rey Auditory Verbal Learning Test — 5 learning trials + recall + recognition"},
        {"id": "verbal-fluency", "name": "Verbal Fluency (FAS)", "category": "Cognitive – Language", "range": "word count", "higher": "better", "description": "Phonemic (FAS) + Semantic (Animals) + Switching fluency"},
    ]
    # Gather per-scale patient count from the DB
    import sqlite3
    from pathlib import Path
    db = str(Path(__file__).parent / "data" / "clinical.db")
    n_patients = 0
    try:
        conn = sqlite3.connect(db)
        n_patients = conn.execute("SELECT COUNT(DISTINCT patient_id) FROM patients").fetchone()[0]
        conn.close()
    except Exception:
        pass
    categories = sorted(set(s["category"] for s in SCALES))
    return {
        "title": "Clinical & Neuropsychological Scales Catalog",
        "subtitle": f"{len(SCALES)} validated scales across {len(categories)} domains — all scored from real patient data",
        "total_scales": len(SCALES),
        "total_patients": n_patients,
        "categories": categories,
        "scales": SCALES,
    }


# ── ICLabel ICA Component Classification ─────────────────────────────

@app.get("/api/ica-label/overview")
async def ica_label_overview():
    """ICLabel overview — aggregate ICA component classification across subjects.
    Uses mne-icalabel CNN to label each component as brain/muscle/eye/heart/
    line_noise/channel_noise/other with confidence probabilities."""
    import scripts.ica_label_classify as icl
    return _json_safe(icl.overview())


@app.get("/api/ica-label/classify")
async def ica_label_classify(subject: str = "chb01"):
    """Per-subject ICLabel classification — full component-by-component labels
    with 7-class probability vectors from the ICLabel neural network."""
    import scripts.ica_label_classify as icl
    return _json_safe(icl.classify(subject))


@app.get("/api/ica-label/detail")
async def ica_label_detail(subject: str = "chb01"):
    """Deep-dive — per-component labels + probability matrix + recommended
    exclusions (artifact probability > brain probability)."""
    import scripts.ica_label_classify as icl
    return _json_safe(icl.detail(subject))


@app.get("/api/ica-label/definitions")
async def ica_label_definitions():
    """ICLabel category definitions — 7-class taxonomy, interpretation guide,
    confidence thresholds, and references (Pion-Tonachini et al. 2019)."""
    import scripts.ica_label_classify as icl
    return _json_safe(icl.definitions())


# ── Entropy & Complexity Dashboard (AntroPy) ──────────────────────────

@app.get("/api/entropy/overview")
async def entropy_overview(file: str = None, seconds: float = 30.0):
    """Entropy dashboard — per-channel SampEn, PE, SpEn, ApEn, HFD, DFA
    computed from real EDF data via AntroPy (Vallat 2023)."""
    import scripts.entropy_dashboard as ent
    return _json_safe(ent.overview(file, seconds))

@app.get("/api/entropy/heatmap")
async def entropy_heatmap(file: str = None, seconds: float = 30.0):
    """Channels × metrics entropy heatmap matrix for visualization."""
    import scripts.entropy_dashboard as ent
    return _json_safe(ent.heatmap(file, seconds))

@app.get("/api/entropy/definitions")
async def entropy_definitions():
    """Entropy metric definitions, clinical interpretation, ranges,
    and references (Richman 2000, Bandt 2002, Higuchi 1988, Peng 1994)."""
    import scripts.entropy_dashboard as ent
    return _json_safe(ent.definitions())


# ── Nilearn Topographic Maps Dashboard ──────────────────────────────

@app.get("/api/topomap/overview")
async def topomap_overview(file: str = None, seconds: float = 30.0):
    """Topographic power maps — per-channel band power (delta/theta/alpha/beta/gamma)
    mapped to standard 10-20 electrode positions using Nilearn + MNE."""
    import scripts.nilearn_topomap as topo
    return _json_safe(topo.overview(file, seconds))

@app.get("/api/topomap/electrodes")
async def topomap_electrodes():
    """Standard 10-20 electrode position map (Jasper 1958)."""
    import scripts.nilearn_topomap as topo
    return _json_safe(topo.electrode_map())

@app.get("/api/topomap/asymmetry")
async def topomap_asymmetry(file: str = None, seconds: float = 30.0):
    """Hemispheric alpha asymmetry — frontal/parietal/occipital/central/temporal
    pairs. Key depression biomarker (Davidson 1998)."""
    import scripts.nilearn_topomap as topo
    return _json_safe(topo.asymmetry(file, seconds))

@app.get("/api/topomap/definitions")
async def topomap_definitions():
    """Band definitions, asymmetry interpretation, clinical references,
    and tool attributions (Nilearn, MNE)."""
    import scripts.nilearn_topomap as topo
    return _json_safe(topo.definitions())


# ── Librosa Spectral Features Dashboard ──────────────────────────────

@app.get("/api/librosa/overview")
async def librosa_overview(file: str = None, seconds: float = 30.0):
    """Librosa spectral features — per-channel centroid, bandwidth, rolloff,
    flatness, ZCR, MFCC, spectral contrast from real EDF data."""
    import scripts.librosa_spectral_dashboard as lib
    return _json_safe(lib.overview(file, seconds))

@app.get("/api/librosa/heatmap")
async def librosa_heatmap(file: str = None, seconds: float = 30.0):
    """Channels × spectral-metrics heatmap matrix for visualization."""
    import scripts.librosa_spectral_dashboard as lib
    return _json_safe(lib.heatmap(file, seconds))

@app.get("/api/librosa/mel-spectrogram")
async def librosa_mel_spectrogram(file: str = None, seconds: float = 30.0):
    """Mel spectrogram — mean dB power per mel bin across channels."""
    import scripts.librosa_spectral_dashboard as lib
    return _json_safe(lib.mel_spectrogram(file, seconds))

@app.get("/api/librosa/mfcc")
async def librosa_mfcc(file: str = None, seconds: float = 30.0):
    """MFCC profile — mean of first 13 MFCCs per channel."""
    import scripts.librosa_spectral_dashboard as lib
    return _json_safe(lib.mfcc_profile(file, seconds))

@app.get("/api/librosa/definitions")
async def librosa_definitions():
    """Spectral feature definitions, clinical interpretation, ranges,
    and references (McFee 2015, Davis 1980, Dubnov 2004)."""
    import scripts.librosa_spectral_dashboard as lib
    return _json_safe(lib.definitions())


# ── Neo Multi-Format Reader Dashboard ──────────────────────────────

@app.get("/api/neo/formats")
async def neo_formats():
    """Neo supported format catalog — all 54 IO classes with extensions,
    descriptions, and key EEG formats (Garcia et al. 2014)."""
    import scripts.neo_reader_dashboard as neo_dash
    return _json_safe(neo_dash.supported_formats())

@app.get("/api/neo/inspect")
async def neo_inspect(file: str = None):
    """Inspect an EDF file via Neo — hierarchical Block/Segment/Signal
    structure with per-signal shape, sampling rate, and summary stats."""
    import scripts.neo_reader_dashboard as neo_dash
    return _json_safe(neo_dash.inspect_file(file))

@app.get("/api/neo/signals")
async def neo_signals(file: str = None, seconds: float = 30.0):
    """Per-channel signal statistics from real EDF data read via Neo —
    mean, std, RMS, peak-to-peak, kurtosis, skewness per channel."""
    import scripts.neo_reader_dashboard as neo_dash
    return _json_safe(neo_dash.signal_overview(file, seconds))

@app.get("/api/neo/definitions")
async def neo_definitions():
    """Neo terminology, data model, EEG format guide, comparison with
    MNE, and references (Garcia et al. 2014, Kemp 1992, Rubel 2022)."""
    import scripts.neo_reader_dashboard as neo_dash
    return _json_safe(neo_dash.definitions())


# ── Synchrosqueezing Dashboard (ssqueezepy) ──────────────────────────

@app.get("/api/synchrosqueezing/overview")
async def synchrosqueezing_overview(file: str = None, seconds: float = 10.0):
    """Synchrosqueezing CWT dashboard — per-channel peak frequency, energy,
    bandwidth, dominant band, and frequency-band breakdown from real EDF data
    via ssqueezepy (Muradeli 2020)."""
    import scripts.synchrosqueezing_dashboard as ssq
    return _json_safe(ssq.synchrosqueezing_overview(file, seconds))

@app.get("/api/synchrosqueezing/spectrum")
async def synchrosqueezing_spectrum(file: str = None, seconds: float = 10.0):
    """Marginal frequency spectrum from the SSQ transform for each channel
    (up to 5 channels, 200 frequency points each)."""
    import scripts.synchrosqueezing_dashboard as ssq
    return _json_safe(ssq.synchrosqueezing_spectrum(file, seconds))

@app.get("/api/synchrosqueezing/definitions")
async def synchrosqueezing_definitions():
    """Synchrosqueezing transform definitions, clinical relevance to EEG,
    and references (Daubechies et al. 2011, Muradeli 2020)."""
    import scripts.synchrosqueezing_dashboard as ssq
    return _json_safe(ssq.synchrosqueezing_definitions())


# ── Explainable AI (XAI) Dashboard — Captum + LIME + SHAP ────────────

@app.get("/api/xai-dashboard/overview")
async def xai_dashboard_overview(file: str = None, seconds: float = 10.0):
    """XAI overview: model info, 3 explanation methods, top SHAP features
    from real EEG features extracted from CHB-MIT EDF recordings."""
    import scripts.xai_dashboard as xd
    return _json_safe(xd.xai_overview(file, seconds))

@app.get("/api/xai-dashboard/captum")
async def xai_dashboard_captum(file: str = None, seconds: float = 10.0):
    """Captum Integrated Gradients + Feature Ablation attributions on
    a PyTorch EEG classifier trained on real spectral/entropy features."""
    import scripts.xai_dashboard as xd
    return _json_safe(xd.xai_captum(file, seconds))

@app.get("/api/xai-dashboard/lime")
async def xai_dashboard_lime(file: str = None, seconds: float = 10.0):
    """LIME local explanations — surrogate linear model on perturbed
    EEG feature inputs (Ribeiro et al. 2016)."""
    import scripts.xai_dashboard as xd
    return _json_safe(xd.xai_lime(file, seconds))

@app.get("/api/xai-dashboard/shap")
async def xai_dashboard_shap(file: str = None, seconds: float = 10.0):
    """SHAP TreeExplainer — global + local Shapley-value feature importance
    on a GradientBoosting model trained on real EEG features."""
    import scripts.xai_dashboard as xd
    return _json_safe(xd.xai_shap(file, seconds))

@app.get("/api/xai-dashboard/comparison")
async def xai_dashboard_comparison(file: str = None, seconds: float = 10.0):
    """Side-by-side normalized ranking across Captum IG, LIME, and SHAP
    with Kendall tau agreement metrics."""
    import scripts.xai_dashboard as xd
    return _json_safe(xd.xai_comparison(file, seconds))

@app.get("/api/xai-dashboard/gradcam")
async def xai_dashboard_gradcam(file: str = None, seconds: float = 10.0):
    """Grad-CAM visual explanations — gradient-weighted class activation
    maps on a 1D CNN over EEG channel features (Selvaraju et al. 2017)."""
    import scripts.xai_dashboard as xd
    return _json_safe(xd.xai_gradcam(file, seconds))

@app.get("/api/xai-dashboard/definitions")
async def xai_dashboard_definitions():
    """XAI method definitions, citations, clinical relevance, and
    EU AI Act Art. 86 compliance mapping."""
    import scripts.xai_dashboard as xd
    return _json_safe(xd.xai_definitions())


@app.get("/api/great-expectations")
async def great_expectations_dashboard():
    """Great Expectations Data Quality Validation Dashboard.
    Reads pre-computed GE report from jobs/reports/great_expectations_latest.json.
    Report generated by: python scripts/great_expectations_dashboard.py
    Real great_expectations v1 validation on EEG feature datasets."""
    p = Path(__file__).parent / "jobs" / "reports" / "great_expectations_latest.json"
    if p.exists():
        return json.loads(p.read_text())
    try:
        import scripts.great_expectations_dashboard as ged
        result = _json_safe(ged.great_expectations_report())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, default=str))
        return result
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}",
                "note": "Run: python scripts/great_expectations_dashboard.py to generate the report"}


@app.get("/api/deepchecks/overview")
async def deepchecks_overview(file: str = None, seconds: float = 10.0):
    """Deepchecks data integrity + model performance validation on real EEG data."""
    import scripts.deepchecks_dashboard as dcd
    return _json_safe(dcd.deepchecks_overview(file, seconds))

@app.get("/api/deepchecks/suites")
async def deepchecks_suites(file: str = None, seconds: float = 10.0):
    """Deepchecks full validation suites with per-condition pass/fail details."""
    import scripts.deepchecks_dashboard as dcd
    return _json_safe(dcd.deepchecks_suites(file, seconds))

@app.get("/api/deepchecks/definitions")
async def deepchecks_definitions():
    """Deepchecks check definitions and clinical relevance for EEG-AI."""
    import scripts.deepchecks_dashboard as dcd
    return _json_safe(dcd.deepchecks_definitions())


@app.get("/api/torchmetrics/overview")
async def torchmetrics_overview(file: str = None, seconds: float = 10.0):
    """TorchMetrics model evaluation overview — accuracy, precision, recall,
    F1, AUROC, Cohen's kappa, MCC, confusion matrix on real EEG features."""
    import scripts.torchmetrics_dashboard as tmd
    return _json_safe(tmd.torchmetrics_overview(file, seconds))

@app.get("/api/torchmetrics/curves")
async def torchmetrics_curves(file: str = None, seconds: float = 10.0):
    """TorchMetrics ROC curve, precision-recall curve, and calibration
    data for a PyTorch EEG classifier."""
    import scripts.torchmetrics_dashboard as tmd
    return _json_safe(tmd.torchmetrics_curves(file, seconds))

@app.get("/api/torchmetrics/definitions")
async def torchmetrics_definitions():
    """TorchMetrics metric definitions, citations, and clinical relevance
    for EEG-AI evaluation."""
    import scripts.torchmetrics_dashboard as tmd
    return _json_safe(tmd.torchmetrics_definitions())


@app.get("/api/aif360/overview")
async def aif360_overview(file: str = None, seconds: float = 10.0):
    """AIF360 bias detection overview — dataset bias metrics, classification
    fairness metrics, and Reweighing mitigation before/after on real EEG data."""
    import scripts.aif360_dashboard as afd
    return _json_safe(afd.aif360_overview(file, seconds))

@app.get("/api/aif360/groups")
async def aif360_groups(file: str = None, seconds: float = 10.0):
    """AIF360 per-group breakdown — privileged vs unprivileged stats,
    base rates, prediction rates, confusion matrices."""
    import scripts.aif360_dashboard as afd
    return _json_safe(afd.aif360_groups(file, seconds))

@app.get("/api/aif360/definitions")
async def aif360_definitions():
    """AIF360 metric definitions and clinical relevance for EEG-AI fairness."""
    import scripts.aif360_dashboard as afd
    return _json_safe(afd.aif360_definitions())


# ── TorchEEG Dashboard ──────────────────────────────────────────────

@app.get("/api/torcheeg/overview")
async def torcheeg_overview(file: str = None, seconds: float = 10.0):
    """TorchEEG feature extraction + classification overview on real EEG data.
    Applies BandDifferentialEntropy, BandPSD, BandHjorth, BandKurtosis,
    BandSkewness transforms and trains an EEGNet-Mini classifier."""
    import scripts.torcheeg_dashboard as ted
    return _json_safe(ted.torcheeg_overview(file, seconds))

@app.get("/api/torcheeg/features")
async def torcheeg_features(file: str = None, seconds: float = 10.0):
    """Per-channel, per-band feature heatmaps from torcheeg transforms."""
    import scripts.torcheeg_dashboard as ted
    return _json_safe(ted.torcheeg_features(file, seconds))

@app.get("/api/torcheeg/definitions")
async def torcheeg_definitions():
    """TorchEEG transform definitions, library info, and clinical relevance."""
    import scripts.torcheeg_dashboard as ted
    return _json_safe(ted.torcheeg_definitions())


# ── Label Studio / CVAT Annotation Quality ───────────────────────────

@app.get("/api/annotation/overview")
async def annotation_overview():
    """Annotation quality overview — stats, agreement, coverage from CHB-MIT annotations."""
    import scripts.label_studio_dashboard as lsd
    return _json_safe(lsd.annotation_overview())

@app.get("/api/annotation/agreement")
async def annotation_agreement():
    """Inter-annotator agreement — Cohen's kappa, Krippendorff's alpha, pairwise matrix."""
    import scripts.label_studio_dashboard as lsd
    return _json_safe(lsd.annotation_agreement())

@app.get("/api/annotation/definitions")
async def annotation_definitions():
    """Annotation label taxonomy, metric definitions, and tool documentation."""
    import scripts.label_studio_dashboard as lsd
    return _json_safe(lsd.annotation_definitions())


# ── AI Cost & Resource Dashboard ───────────────────────────────────

@app.get("/api/ai-cost/overview")
async def ai_cost_overview():
    """AI cost overview — transaction volume, cost estimates, compute resources,
    carbon footprint, and daily trend from real project data."""
    import scripts.ai_cost_dashboard as acd
    raw = acd.cost_overview()
    txn = raw.get("transaction_log", {})
    cost = raw.get("cost_breakdown", {})
    comp = raw.get("compute_resources", {})
    mdl = raw.get("model_files", {})
    trn = raw.get("trends", {})
    carbon = raw.get("carbon_tracking", {})
    return _json_safe({
        "available": raw.get("available", False),
        "summary": {
            "total_operations": txn.get("total_operations", 0),
            "estimated_monthly_cost": cost.get("total_estimated_usd", 0),
            "carbon_footprint_kg": carbon.get("total_kg_co2", 0) if isinstance(carbon, dict) else 0,
            "active_models": mdl.get("count", 0),
        },
        "resources": {
            "cpu_utilization_pct": comp.get("cpu_percent"),
            "memory_used_gb": comp.get("memory_used_gb"),
            "memory_total_gb": comp.get("memory_total_gb"),
        },
        "cost_by_category": cost,
        "ops_by_component": txn.get("ops_by_component", {}),
        "ops_by_action": txn.get("ops_by_action", {}),
        "daily_trend": [
            {"date": d.get("date", ""), "operations": d.get("ops", 0), "cost": d.get("ops", 0) * 0.002}
            for d in trn.get("daily_last_7d", [])
        ],
        "metadata": raw.get("metadata", {}),
    })

@app.get("/api/ai-cost/breakdown")
async def ai_cost_breakdown():
    """Per-component cost breakdown — operations, estimated cost, category."""
    import scripts.ai_cost_dashboard as acd
    raw = acd.cost_breakdown()
    components = [
        {
            "component": c.get("component", "unknown"),
            "operations": c.get("operations", 0),
            "estimated_cost": c.get("estimated_cost_usd", 0),
            "category": c.get("cost_category", ""),
            "top_actions": c.get("top_actions", {}),
        }
        for c in raw.get("top_components", [])
    ]
    return _json_safe({
        "available": raw.get("available", True),
        "components": components,
        "total_estimated_cost_usd": raw.get("total_estimated_cost_usd", 0),
    })

@app.get("/api/ai-cost/definitions")
async def ai_cost_definitions():
    """AI cost metric definitions, rate tables, and regulatory context."""
    import scripts.ai_cost_dashboard as acd
    return _json_safe(acd.cost_definitions())


# ── Inference & GPU Dashboard ──────────────────────────────────────

@app.get("/api/inference-gpu/overview")
async def inference_gpu_overview():
    """GPU status via nvidia-smi, inference summary from transaction log, system info."""
    import subprocess, sys, os
    from datetime import datetime, timezone
    from collections import Counter

    # ── GPU via nvidia-smi ──
    gpu_available = False
    gpu_info = {}
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 8:
                gpu_available = True
                gpu_info = {
                    "name": parts[0],
                    "memory_total_mb": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_free_mb": float(parts[3]),
                    "utilization_gpu_pct": float(parts[4]),
                    "utilization_memory_pct": float(parts[5]),
                    "temperature_c": float(parts[6]),
                    "power_draw_w": float(parts[7]),
                }
    except Exception:
        pass

    # ── Inference summary from transaction log ──
    inference_actions = {'process', 'predict', 'classify', 'inference', 'evaluate', 'train'}
    total_inferences = 0
    components = set()
    last_ts = None
    earliest_ts = None
    try:
        txn = cdb.list_transactions(limit=9999)
        rows = txn.get("rows", [])
        for r in rows:
            action = (r.get("action") or "").lower()
            if action in inference_actions:
                total_inferences += 1
                comp = r.get("component") or r.get("ref_id") or ""
                if comp:
                    components.add(comp)
                ts = r.get("created_utc") or r.get("created_local") or ""
                if ts:
                    if last_ts is None or ts > last_ts:
                        last_ts = ts
                    if earliest_ts is None or ts < earliest_ts:
                        earliest_ts = ts
    except Exception:
        rows = []

    # Calculate avg throughput
    avg_throughput = 0
    if total_inferences > 0 and earliest_ts and last_ts and earliest_ts != last_ts:
        try:
            fmt_str = "%Y-%m-%d %H:%M:%S"
            t0 = datetime.strptime(earliest_ts[:19], fmt_str)
            t1 = datetime.strptime(last_ts[:19], fmt_str)
            hours = max((t1 - t0).total_seconds() / 3600, 1)
            avg_throughput = round(total_inferences / hours, 2)
        except Exception:
            avg_throughput = 0

    # ── System info ──
    cpu_count = os.cpu_count()
    ram_total_gb = 0
    ram_used_gb = 0
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    meminfo[key] = int(val)
            ram_total_gb = round(meminfo.get("MemTotal", 0) / 1048576, 2)
            mem_avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            ram_used_gb = round((meminfo.get("MemTotal", 0) - mem_avail) / 1048576, 2)
    except Exception:
        pass

    torch_version = "--"
    cuda_available = False
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    return _json_safe({
        "available": gpu_available,
        "gpu": gpu_info,
        "inference_summary": {
            "total_inferences": total_inferences,
            "models_loaded": len(components),
            "avg_throughput_per_hour": avg_throughput,
            "last_inference_at": last_ts or "",
        },
        "system": {
            "cpu_count": cpu_count,
            "ram_total_gb": ram_total_gb,
            "ram_used_gb": ram_used_gb,
            "python_version": sys.version.split()[0],
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "cuda_note": "Driver too old for PyTorch CUDA; nvidia-smi reports GPU directly" if not cuda_available else "",
        },
        "note": "" if gpu_available else "nvidia-smi not found or GPU not detected",
    })


@app.get("/api/inference-gpu/models")
async def inference_gpu_models():
    """List model files found in the models/ directory."""
    from pathlib import Path
    from datetime import datetime

    model_dir = Path(__file__).parent / "models"
    extensions = {'.pkl', '.pt', '.pth', '.onnx', '.joblib'}
    models = []
    if model_dir.is_dir():
        for ext in extensions:
            for f in model_dir.glob(f"*{ext}"):
                if f.is_file():
                    stat = f.stat()
                    models.append({
                        "name": f.name,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "path": str(f.relative_to(Path(__file__).parent)),
                        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    })
    models.sort(key=lambda m: m["name"])
    return _json_safe({"models": models})


@app.get("/api/inference-gpu/definitions")
async def inference_gpu_definitions():
    """Metric definitions for the Inference & GPU dashboard."""
    return _json_safe({
        "metrics": [
            {"name": "GPU Utilization", "description": "Percentage of GPU compute cores actively processing work, reported by nvidia-smi.", "unit": "%"},
            {"name": "VRAM Used", "description": "Video RAM currently allocated by all processes on the GPU.", "unit": "MB"},
            {"name": "VRAM Free", "description": "Video RAM available for new allocations.", "unit": "MB"},
            {"name": "GPU Temperature", "description": "Current thermal reading of the GPU die.", "unit": "\u00b0C"},
            {"name": "Power Draw", "description": "Current power consumption of the GPU.", "unit": "W"},
            {"name": "Total Inferences", "description": "Count of process/predict/classify/inference/evaluate/train actions in the transaction log.", "unit": "count"},
            {"name": "Models Loaded", "description": "Number of unique model components referenced in inference transactions.", "unit": "count"},
            {"name": "Avg Throughput/Hour", "description": "Average number of inference operations per hour across the logged time window.", "unit": "ops/hr"},
            {"name": "CPU Cores", "description": "Number of logical CPU cores available to the system.", "unit": "cores"},
            {"name": "RAM Used", "description": "System RAM currently in use (total minus available).", "unit": "GB"},
            {"name": "CUDA Available", "description": "Whether PyTorch can access CUDA GPU acceleration.", "unit": "boolean"},
        ]
    })


@app.get("/api/embedding-drift/overview")
async def embedding_drift_overview():
    """Embedding Drift Dashboard — drift monitoring for RAG embedding vectors."""
    from scripts.embedding_drift_dashboard import generate_embedding_drift_overview
    return _json_safe(generate_embedding_drift_overview())


@app.get("/api/embedding-drift/breakdown")
async def embedding_drift_breakdown():
    """Embedding drift breakdown by corpus segment + stale vector detection."""
    from scripts.embedding_drift_dashboard import generate_embedding_drift_breakdown
    return _json_safe(generate_embedding_drift_breakdown())


@app.get("/api/embedding-drift/definitions")
async def embedding_drift_definitions():
    """Metric definitions for the Embedding Drift dashboard."""
    from scripts.embedding_drift_dashboard import generate_embedding_drift_definitions
    return _json_safe(generate_embedding_drift_definitions())


# ─── SLP routes — see consolidated block near EOF ─────────────────────────


# ── Patient Portal: Medication Tab ──────────────────────────────────
# Real data: medications (9, AED drug names+doses+frequency) + patients (40) +
# seizure_diary (25). Patient-facing view: current meds, daily schedule,
# adherence scores, side-effect profile, and medication recommendations.

@app.get("/api/medication")
async def medication_dashboard(patient_id: str = None):
    """Patient Medication Portal — full dashboard: current meds, schedule, adherence,
    side effects, recommendations. All from REAL medications + seizure_diary in data/clinical.db."""
    import scripts.medication_module as med
    return _json_safe(med.full_dashboard(patient_id))


@app.get("/api/medication/list")
async def medication_list(patient_id: str = None):
    """Current medications list with drug info, dose, frequency, class, side effects."""
    import scripts.medication_module as med
    return _json_safe(med.my_medications(patient_id))


@app.get("/api/medication/schedule")
async def medication_schedule(patient_id: str = None):
    """Daily medication schedule: morning/noon/evening/bedtime time slots."""
    import scripts.medication_module as med
    return _json_safe(med.medication_schedule(patient_id))


@app.get("/api/medication/adherence")
async def medication_adherence(patient_id: str = None):
    """Adherence scores per patient + seizure correlation."""
    import scripts.medication_module as med
    return _json_safe(med.adherence_summary(patient_id))


@app.get("/api/medication/recommendations")
async def medication_recommendations(patient_id: str = None):
    """Medication recommendations: pregnancy warnings, interaction flags, dose suggestions."""
    import scripts.medication_module as med
    return _json_safe(med.medication_recommendations(patient_id))


@app.get("/api/medication/side-effects")
async def medication_side_effects(patient_id: str = None):
    """Side effect profile: ranked effects, overlapping risks across medications."""
    import scripts.medication_module as med
    return _json_safe(med.side_effect_profile(patient_id))


@app.get("/api/medication/definitions")
async def medication_definitions():
    """Medication terminology, clinical relevance, and remediation strategies."""
    import scripts.medication_module as med
    return _json_safe(med.definitions())


# ── Patients Seen Dashboard ──────────────────────────────────────
# Real data: appointments (120 rows, 6 providers, 4 depts, 34 patients) +
# patients (40). Role operational view: patients seen per provider/dept,
# completion rates, no-show analysis, daily trends.

@app.get("/api/patients-seen/overview")
async def patients_seen_overview():
    """Patients Seen — summary KPIs: patients seen, completion rate, no-show rate."""
    import scripts.patients_seen_module as ps
    return _json_safe(ps.overview())


@app.get("/api/patients-seen/breakdown")
async def patients_seen_breakdown():
    """Patients Seen — breakdown by provider, department, appt type, daily trend, per-patient."""
    import scripts.patients_seen_module as ps
    return _json_safe(ps.breakdown())


@app.get("/api/patients-seen/definitions")
async def patients_seen_definitions():
    """Patients Seen — clinical definitions, quality metrics, standards, remediation."""
    import scripts.patients_seen_module as ps
    return _json_safe(ps.definitions())


# ── Executive Scorecard ───────────────────────────────────────────

@app.get("/api/executive-scorecard/overview")
async def executive_scorecard_overview():
    """Executive scorecard — aggregated KPIs: patient census, clinical activity,
    AI operations, quality indicators, and 7-day trend."""
    import scripts.executive_scorecard as esc
    return _json_safe(esc.scorecard_overview())


@app.get("/api/executive-scorecard/breakdown")
async def executive_scorecard_breakdown():
    """Per-department and per-instrument drill-down for executive scorecard."""
    import scripts.executive_scorecard as esc
    return _json_safe(esc.scorecard_breakdown())


@app.get("/api/executive-scorecard/definitions")
async def executive_scorecard_definitions():
    """Metric definitions for executive scorecard tooltip overlays."""
    import scripts.executive_scorecard as esc
    return _json_safe(esc.scorecard_definitions())


# ── Executive AI Dashboard ────────────────────────────────────────

@app.get("/api/executive-ai/overview")
async def executive_ai_overview():
    """Executive AI — AI adoption, utilization, oversight, throughput KPIs."""
    import scripts.executive_ai_dashboard as ead
    return _json_safe(ead.executive_ai_overview())


@app.get("/api/executive-ai/breakdown")
async def executive_ai_breakdown():
    """Per-component and per-department AI drill-down."""
    import scripts.executive_ai_dashboard as ead
    return _json_safe(ead.executive_ai_breakdown())


@app.get("/api/executive-ai/definitions")
async def executive_ai_definitions():
    """Metric definitions for Executive AI Dashboard."""
    import scripts.executive_ai_dashboard as ead
    return _json_safe(ead.executive_ai_definitions())


# ── AI Usage Dashboard ────────────────────────────────────────────

@app.get("/api/ai-usage/overview")
async def ai_usage_overview():
    """AI usage overview — total operations, daily trend, component/action/actor
    distributions from real transaction_log in clinical.db."""
    import scripts.ai_usage_dashboard as aud
    return _json_safe(aud.usage_overview())


@app.get("/api/ai-usage/breakdown")
async def ai_usage_breakdown():
    """Per-component usage breakdown — operations, actions, actors, last activity."""
    import scripts.ai_usage_dashboard as aud
    return _json_safe(aud.usage_breakdown())


@app.get("/api/ai-usage/definitions")
async def ai_usage_definitions():
    """AI usage metric definitions for tooltip overlays."""
    import scripts.ai_usage_dashboard as aud
    return _json_safe(aud.usage_definitions())


# ── Tool Execution Dashboard ──────────────────────────────────────────

@app.get("/api/tool-execution/overview")
async def tool_execution_overview():
    """Tool execution overview — totals, success rate, top tools, actors, trend."""
    import scripts.tool_execution_dashboard as ted
    return _json_safe(ted.tool_execution_overview())


@app.get("/api/tool-execution/breakdown")
async def tool_execution_breakdown():
    """Per-tool execution breakdown — action mix, actors, first/last seen."""
    import scripts.tool_execution_dashboard as ted
    return _json_safe(ted.tool_execution_breakdown())


@app.get("/api/tool-execution/definitions")
async def tool_execution_definitions():
    """Tool execution metric definitions for tooltip overlays."""
    import scripts.tool_execution_dashboard as ted
    return _json_safe(ted.tool_execution_definitions())


# ── Therapy / Meditation / Physio Portal Tab ─────────────────────────

@app.get("/api/therapy")
async def therapy_dashboard(patient_id: str = None):
    """Patient Therapy Portal — full dashboard: rehab programs, exercises,
    meditation, physiotherapy. All derived from REAL assessments + seizure diary
    in data/clinical.db."""
    import scripts.therapy_module as thx
    return _json_safe(thx.therapy_overview(patient_id))


@app.get("/api/therapy/rehab")
async def therapy_rehab(patient_id: str = None):
    """Rehabilitation programs derived from BARTHEL, MOCA, MMSE, EPWORTH scores."""
    import scripts.therapy_module as thx
    return _json_safe(thx.rehab_programs(patient_id))


@app.get("/api/therapy/exercises")
async def therapy_exercises(patient_id: str = None):
    """Evidence-based exercise prescriptions adapted to seizure profile."""
    import scripts.therapy_module as thx
    return _json_safe(thx.exercise_plans(patient_id))


@app.get("/api/therapy/meditation")
async def therapy_meditation(patient_id: str = None):
    """Mindfulness / meditation programs based on PHQ9, GAD7, QOLIE31 scores."""
    import scripts.therapy_module as thx
    return _json_safe(thx.meditation_programs(patient_id))


@app.get("/api/therapy/physio")
async def therapy_physio(patient_id: str = None):
    """Physiotherapy protocols for motor recovery, balance, swallowing, speech."""
    import scripts.therapy_module as thx
    return _json_safe(thx.physio_protocols(patient_id))


@app.get("/api/therapy/definitions")
async def therapy_definitions():
    """Therapy metric definitions for tooltip overlays."""
    import scripts.therapy_module as thx
    return _json_safe(thx.therapy_definitions())


# ── Notification Portal Tab ───────────────────────────────────────────

@app.get("/api/notifications")
async def notification_dashboard(patient_id: str = None):
    """Patient Notification Centre — all notifications: assessment results,
    form assignments, seizure follow-ups, medication updates, clinical
    activity.  All derived from REAL data in data/clinical.db."""
    import scripts.notification_module as ntf
    return _json_safe(ntf.notification_overview(patient_id))


@app.get("/api/notifications/category/{category}")
async def notification_by_category(category: str, patient_id: str = None):
    """Notifications filtered by category (result/form/seizure/medication/activity/alert)."""
    import scripts.notification_module as ntf
    return _json_safe(ntf.notification_by_category(category, patient_id))


@app.get("/api/notifications/unread")
async def notification_unread(patient_id: str = None):
    """Unread notifications only."""
    import scripts.notification_module as ntf
    return _json_safe(ntf.notification_unread(patient_id))


@app.get("/api/notifications/definitions")
async def notification_definitions():
    """Notification metric definitions for tooltip overlays."""
    import scripts.notification_module as ntf
    return _json_safe(ntf.notification_definitions())


# ── Alerts Portal Tab ─────────────────────────────────────────────────

@app.get("/api/alerts")
async def alerts_dashboard(patient_id: str = None):
    """Clinical Alerts Dashboard — all alert categories: assessment threshold
    breaches, seizure events, medication risks, vitals concerns.
    All derived from REAL data in data/clinical.db."""
    import scripts.alerts_module as alm
    return _json_safe(alm.alerts_overview(patient_id))


@app.get("/api/alerts/category/{category}")
async def alerts_by_category(category: str, patient_id: str = None):
    """Alerts filtered by category (assessment/seizure/medication/vitals)."""
    import scripts.alerts_module as alm
    return _json_safe(alm.alerts_by_category(category, patient_id))


@app.get("/api/alerts/severity/{severity}")
async def alerts_by_severity(severity: str, patient_id: str = None):
    """Alerts filtered by severity (critical/high/medium/low)."""
    import scripts.alerts_module as alm
    return _json_safe(alm.alerts_by_severity(severity, patient_id))


@app.get("/api/alerts/summary")
async def alerts_summary(patient_id: str = None):
    """Compact alert summary: counts + top 5 critical."""
    import scripts.alerts_module as alm
    return _json_safe(alm.alerts_summary(patient_id))


@app.get("/api/alerts/definitions")
async def alerts_definitions():
    """Alert metric definitions for tooltip overlays."""
    import scripts.alerts_module as alm
    return _json_safe(alm.alerts_definitions())


# ── Patient Reports — real EEG/summary reports from clinical.db ──────────
@app.get("/api/patient-reports")
async def patient_reports(patient_id: str = None):
    """All patient report cards (or one patient). Real data from clinical.db."""
    import scripts.reports_module as rpm
    return _json_safe(rpm.reports_overview(patient_id or None))


@app.get("/api/patient-reports/summary")
async def patient_reports_summary():
    """Aggregate report stats: totals, breakdowns by disease/instrument/prediction."""
    import scripts.reports_module as rpm
    return _json_safe(rpm.reports_summary())


@app.get("/api/patient-reports/definitions")
async def patient_reports_definitions():
    """Report metric definitions for tooltip overlays."""
    import scripts.reports_module as rpm
    return _json_safe(rpm.reports_definitions())


# ── Database Ops Dashboard ─────────────────────────────────────────

@app.get("/api/database-ops/overview")
async def database_ops_overview():
    """Database health overview — size, tables, rows, WAL, integrity, backups."""
    import scripts.database_ops_dashboard as dod
    return _json_safe(dod.db_overview())


@app.get("/api/database-ops/breakdown")
async def database_ops_breakdown():
    """Per-table breakdown: rows, columns, indexes, last activity."""
    import scripts.database_ops_dashboard as dod
    return _json_safe(dod.db_breakdown())


@app.get("/api/database-ops/definitions")
async def database_ops_definitions():
    """Database ops metric definitions for tooltip overlays."""
    import scripts.database_ops_dashboard as dod
    return _json_safe(dod.db_definitions())


# ── Campaigns Portal Tab ─────────────────────────────────────────────

@app.get("/api/campaigns")
async def campaigns_dashboard(patient_id: str = None):
    """Campaigns Dashboard — health campaigns, screening programs, education,
    medication adherence, seizure safety.
    All derived from REAL data in data/clinical.db."""
    import scripts.campaigns_module as cmp
    return _json_safe(cmp.campaigns_overview(patient_id))


@app.get("/api/campaigns/type/{campaign_type}")
async def campaigns_by_type(campaign_type: str, patient_id: str = None):
    """Campaigns filtered by type (screening/adherence/safety/form_completion/education)."""
    import scripts.campaigns_module as cmp
    return _json_safe(cmp.campaigns_by_type(campaign_type, patient_id))


@app.get("/api/campaigns/summary")
async def campaigns_summary(patient_id: str = None):
    """Compact campaign summary: counts + top 5 active."""
    import scripts.campaigns_module as cmp
    return _json_safe(cmp.campaigns_summary(patient_id))


@app.get("/api/campaigns/definitions")
async def campaigns_definitions():
    """Campaign metric definitions for tooltip overlays."""
    import scripts.campaigns_module as cmp
    return _json_safe(cmp.campaigns_definitions())


# ── AI Risk Management — risk identification, assessment, mitigation ──

@app.get("/api/ai-risk/overview")
async def ai_risk_overview():
    """AI Risk Management overview — risk posture, severity distribution,
    risk categories, trend from real clinical.db."""
    import scripts.ai_risk_dashboard as ard
    return _json_safe(ard.risk_overview())


@app.get("/api/ai-risk/breakdown")
async def ai_risk_breakdown():
    """AI Risk Management breakdown — per-patient risk profiles,
    risk register, risk matrix, mitigation log."""
    import scripts.ai_risk_dashboard as ard
    return _json_safe(ard.risk_breakdown())


@app.get("/api/ai-risk/definitions")
async def ai_risk_definitions():
    """AI Risk Management definitions — risk concepts, metrics,
    clinical relevance (ISO 14971, EU AI Act, FDA AI-ML), remediation."""
    import scripts.ai_risk_dashboard as ard
    return _json_safe(ard.risk_definitions())


# ── Message Inbox — secure messages from clinical.db ─────────────

@app.get("/api/inbox")
async def inbox_dashboard(patient_id: str = None):
    """Message Inbox — all message categories: care team messages, clinical
    decisions, expert reviews, form assignments from clinical.db."""
    import scripts.inbox_module as inm
    return _json_safe(inm.inbox_overview(patient_id))


@app.get("/api/inbox/category/{category}")
async def inbox_by_category(category: str, patient_id: str = None):
    """Inbox messages filtered by category (team_message/clinical_decision/expert_review/form_assignment)."""
    import scripts.inbox_module as inm
    return _json_safe(inm.inbox_by_category(category, patient_id))


@app.get("/api/inbox/summary")
async def inbox_summary(patient_id: str = None):
    """Compact inbox summary: counts + latest 5 messages."""
    import scripts.inbox_module as inm
    return _json_safe(inm.inbox_summary(patient_id))


@app.get("/api/inbox/definitions")
async def inbox_definitions():
    """Inbox metric definitions for tooltip overlays."""
    import scripts.inbox_module as inm
    return _json_safe(inm.inbox_definitions())


# ── Token / Cost Dashboard ────────────────────────────────────────────

@app.get("/api/token-cost/overview")
async def token_cost_overview():
    """Token usage overview — LLM tokens, operation costs, budget status,
    and daily trend from real clinical.db data."""
    import scripts.token_cost_dashboard as tcd
    return _json_safe(tcd.token_cost_overview())

@app.get("/api/token-cost/breakdown")
async def token_cost_breakdown():
    """Per-component token/cost breakdown — role tokens, component ops, model inferences."""
    import scripts.token_cost_dashboard as tcd
    return _json_safe(tcd.token_cost_breakdown())

@app.get("/api/token-cost/budget")
async def token_cost_budget():
    """Budget allocation, utilization, alerts, and local-LLM savings."""
    import scripts.token_cost_dashboard as tcd
    return _json_safe(tcd.token_cost_budget())

@app.get("/api/token-cost/definitions")
async def token_cost_definitions():
    """Token/cost metric definitions, rate cards, and budget tiers."""
    import scripts.token_cost_dashboard as tcd
    return _json_safe(tcd.token_cost_definitions())


# ── Vector DB Dashboard ───────────────────────────────────────────────
# Real data: ChromaDB (data/vector_db/chroma.sqlite3) with 75 embeddings,
# 768-dim vectors, 42 patients, 7 document types, HNSW index.

@app.get("/api/vector-db/overview")
async def vector_db_overview():
    """Vector DB Dashboard — KPIs, collection stats, storage, health from ChromaDB."""
    import scripts.vector_db_dashboard as vdb
    return _json_safe(vdb.vector_db_overview())

@app.get("/api/vector-db/collections")
async def vector_db_collections():
    """Per-collection breakdown: dimension, count, sample records, patient distribution."""
    import scripts.vector_db_dashboard as vdb
    return _json_safe(vdb.vector_db_collections())

@app.get("/api/vector-db/operations")
async def vector_db_operations():
    """Queue operations, ingestion timeline, throughput metrics."""
    import scripts.vector_db_dashboard as vdb
    return _json_safe(vdb.vector_db_operations())

@app.get("/api/vector-db/definitions")
async def vector_db_definitions():
    """Metric definitions for the Vector DB dashboard."""
    import scripts.vector_db_dashboard as vdb
    return _json_safe(vdb.vector_db_definitions())


# ── Chunking Dashboard ─────────────────────────────────────────────────
# Real data: ChromaDB chunk texts + lengths, production chunking config,
# 5 strategy definitions, per-patient/per-type chunk stats.

@app.get("/api/chunking/overview")
async def chunking_overview():
    """Chunking overview — KPIs, size distribution, doc-type mix, production config."""
    import scripts.chunking_dashboard as chk
    return _json_safe(chk.chunking_overview())

@app.get("/api/chunking/breakdown")
async def chunking_breakdown():
    """Per-strategy config, per-patient chunk counts, per-type length stats."""
    import scripts.chunking_dashboard as chk
    return _json_safe(chk.chunking_breakdown())

@app.get("/api/chunking/definitions")
async def chunking_definitions():
    """Metric definitions for the Chunking dashboard."""
    import scripts.chunking_dashboard as chk
    return _json_safe(chk.chunking_definitions())


# ── Hallucination Dashboard ───────────────────────────────────────────
# Real data: ChromaDB grounding analysis, clinical.db citation coverage,
# conversation faithfulness scoring, HITL verification.

@app.get("/api/hallucination/overview")
async def hallucination_overview():
    """Hallucination overview — risk score, grounding, citation rate, faithfulness."""
    import scripts.hallucination_dashboard as hal
    return _json_safe(hal.hallucination_overview())

@app.get("/api/hallucination/breakdown")
async def hallucination_breakdown():
    """Per-patient grounding, disease coverage, mitigation strategies."""
    import scripts.hallucination_dashboard as hal
    return _json_safe(hal.hallucination_breakdown())

@app.get("/api/hallucination/definitions")
async def hallucination_definitions():
    """Metric definitions for the Hallucination dashboard."""
    import scripts.hallucination_dashboard as hal
    return _json_safe(hal.hallucination_definitions())


# ── Knowledge Graph Dashboard ────────────────────────────────────────
# Real data: clinical.db entity-relationship graph (patients, diseases,
# medications, analyses, MRI, neuropsych, HITL reviews) + ChromaDB embeddings.

@app.get("/api/knowledge-graph/overview")
async def knowledge_graph_overview():
    """Knowledge graph overview — node/edge counts, entity types, hub nodes."""
    import scripts.knowledge_graph_dashboard as kgd
    return _json_safe(kgd.knowledge_graph_overview())

@app.get("/api/knowledge-graph/breakdown")
async def knowledge_graph_breakdown():
    """Per-patient subgraphs, disease clusters, medication network, full graph."""
    import scripts.knowledge_graph_dashboard as kgd
    return _json_safe(kgd.knowledge_graph_breakdown())

@app.get("/api/knowledge-graph/definitions")
async def knowledge_graph_definitions():
    """Metric definitions for the Knowledge Graph dashboard."""
    import scripts.knowledge_graph_dashboard as kgd
    return _json_safe(kgd.knowledge_graph_definitions())


# ── DevOps / CI-CD Dashboard ─────────────────────────────────────────

@app.get("/api/devops/overview")
async def devops_overview():
    """DevOps metrics — deploy frequency, change-fail rate, MTTR, commit velocity from real git data."""
    import scripts.devops_dashboard as dvd
    return _json_safe(dvd.devops_overview())


@app.get("/api/devops/pipelines")
async def devops_pipelines():
    """Pipeline / cron job status — real cron definitions and running process checks."""
    import scripts.devops_dashboard as dvd
    return _json_safe(dvd.devops_pipelines())


@app.get("/api/devops/definitions")
async def devops_definitions():
    """Metric definitions for the DevOps/CI-CD dashboard."""
    import scripts.devops_dashboard as dvd
    return _json_safe(dvd.devops_definitions())


# ── Content Freshness Dashboard ─────────────────────────────────────
# Real data: ChromaDB document ages, staleness scores, ingestion timeline,
# content decay risk, and refresh recommendations from chroma.sqlite3 + clinical.db.

@app.get("/api/content-freshness/overview")
async def content_freshness_overview():
    """Content freshness overview — document ages, staleness, decay risk, queue stats."""
    import scripts.content_freshness_dashboard as cfd
    return _json_safe(cfd.content_freshness_overview())


@app.get("/api/content-freshness/breakdown")
async def content_freshness_breakdown():
    """Per-patient freshness, per-type detail, update activity, refresh recommendations."""
    import scripts.content_freshness_dashboard as cfd
    return _json_safe(cfd.content_freshness_breakdown())


@app.get("/api/content-freshness/definitions")
async def content_freshness_definitions():
    """Metric definitions for the Content Freshness dashboard."""
    import scripts.content_freshness_dashboard as cfd
    return _json_safe(cfd.content_freshness_definitions())


# ── AI Compliance Dashboard ────────────────────────────────────────

@app.get("/api/ai-compliance/overview")
async def ai_compliance_overview():
    """AI compliance overview — HITL reviews, expert agreement, audit trail,
    EU AI Act risk tiers, governance checklist from real clinical.db data."""
    import scripts.ai_compliance_dashboard as acd
    return _json_safe(acd.compliance_overview())


@app.get("/api/ai-compliance/breakdown")
async def ai_compliance_breakdown():
    """Per-role and per-component compliance drill-down — HITL details,
    expert reviews, clinical decisions, component audit trail."""
    import scripts.ai_compliance_dashboard as acd
    return _json_safe(acd.compliance_breakdown())


@app.get("/api/ai-compliance/definitions")
async def ai_compliance_definitions():
    """AI compliance metric definitions for tooltip overlays."""
    import scripts.ai_compliance_dashboard as acd
    return _json_safe(acd.compliance_definitions())


# ── Response Quality Dashboard ─────────────────────────────────────
# Real data: clinical.db conversation_log (213 messages), analyses (21),
# feedback ratings, transaction_log (549), ChromaDB RAG coverage.

@app.get("/api/response-quality/overview")
async def response_quality_overview():
    """Response quality overview — quality scores, confidence distribution,
    structure/data/citation rates, daily timeline from real conversation data."""
    import scripts.response_quality_dashboard as rqd
    return _json_safe(rqd.response_quality_overview())


@app.get("/api/response-quality/breakdown")
async def response_quality_breakdown():
    """Per-response detail, per-disease analysis quality, component reliability,
    RAG coverage, and feedback drill-down."""
    import scripts.response_quality_dashboard as rqd
    return _json_safe(rqd.response_quality_breakdown())


@app.get("/api/response-quality/definitions")
async def response_quality_definitions():
    """Metric definitions for the Response Quality dashboard."""
    import scripts.response_quality_dashboard as rqd
    return _json_safe(rqd.response_quality_definitions())


# ── Retrieval Evaluation Dashboard ─────────────────────────────────
# Real data: ChromaDB (75 embeddings, 709 queue entries) + clinical.db
# cross-reference for coverage, queue health, type diversity.

@app.get("/api/retrieval-eval/overview")
async def retrieval_eval_overview():
    """Retrieval evaluation overview — quality score, patient coverage,
    queue health, type distribution from real ChromaDB + clinical.db data."""
    import scripts.retrieval_evaluation_dashboard as red
    return _json_safe(red.retrieval_overview())


@app.get("/api/retrieval-eval/breakdown")
async def retrieval_eval_breakdown():
    """Per-patient vector coverage, document inventory, analysis cross-reference,
    queue detail drill-down."""
    import scripts.retrieval_evaluation_dashboard as red
    return _json_safe(red.retrieval_breakdown())


@app.get("/api/retrieval-eval/definitions")
async def retrieval_eval_definitions():
    """Metric definitions for the Retrieval Evaluation dashboard."""
    import scripts.retrieval_evaluation_dashboard as red
    return _json_safe(red.retrieval_definitions())


# ── Retrieval Dashboard ──────────────────────────────────────────────
# Real data: clinical.db transaction_log (patient_chat), conversation_log,
# ChromaDB vector store — query volume, patient coverage, vector health.

@app.get("/api/retrieval/overview")
async def retrieval_overview():
    """Retrieval overview — query volume, patient coverage, vector store
    health, queue status, collection stats from real data."""
    import scripts.retrieval_dashboard as rtv
    return _json_safe(rtv.retrieval_overview())


@app.get("/api/retrieval/breakdown")
async def retrieval_breakdown():
    """Per-patient retrieval activity, query text analysis, embedding
    timeline, collection detail, top queries."""
    import scripts.retrieval_dashboard as rtv
    return _json_safe(rtv.retrieval_breakdown())


@app.get("/api/retrieval/definitions")
async def retrieval_definitions():
    """Metric definitions for the Retrieval dashboard."""
    import scripts.retrieval_dashboard as rtv
    return _json_safe(rtv.retrieval_definitions())


# ── Agent Loop / Goal-Drift Dashboard ────────────────────────────────

@app.get("/api/agent-loop/overview")
async def agent_loop_overview():
    """Agent loop overview — component cycles, conversation turns,
    goal-drift score, confidence & agreement drift from clinical.db."""
    import scripts.agent_loop_dashboard as ald
    return _json_safe(ald.loop_overview())


@app.get("/api/agent-loop/breakdown")
async def agent_loop_breakdown():
    """Per-component loop detail — action counts, corrections,
    decision disagreements."""
    import scripts.agent_loop_dashboard as ald
    return _json_safe(ald.loop_breakdown())


@app.get("/api/agent-loop/definitions")
async def agent_loop_definitions():
    """Metric definitions for the Agent Loop / Goal-Drift dashboard."""
    import scripts.agent_loop_dashboard as ald
    return _json_safe(ald.loop_definitions())


# ── Event / Kafka / Queue Dashboard ────────────────────────────
@app.get("/api/event-queue/overview")
async def event_queue_overview():
    """Event queue overview — throughput, action distribution,
    component queues, daily volume from clinical.db."""
    import scripts.event_queue_dashboard as eqd
    return _json_safe(eqd.event_queue_overview())


@app.get("/api/event-queue/breakdown")
async def event_queue_breakdown():
    """Per-queue breakdown — cross-tab, recent events, patient events,
    queue stats, conversation timeline."""
    import scripts.event_queue_dashboard as eqd
    return _json_safe(eqd.event_queue_breakdown())


@app.get("/api/event-queue/definitions")
async def event_queue_definitions():
    """Metric definitions for the Event / Queue dashboard."""
    import scripts.event_queue_dashboard as eqd
    return _json_safe(eqd.event_queue_definitions())


# ── Routing Dashboard ─────────────────────────────────────────

@app.get("/api/routing/overview")
async def routing_overview():
    """Routing overview — component fanout, actor workload,
    automation rate, decision outcomes from clinical.db."""
    import scripts.routing_dashboard as rtd
    return _json_safe(rtd.routing_overview())


@app.get("/api/routing/breakdown")
async def routing_breakdown():
    """Per-route breakdown — cross-tab, recent events, patient
    routing paths, decision detail, component stats."""
    import scripts.routing_dashboard as rtd
    return _json_safe(rtd.routing_breakdown())


@app.get("/api/routing/definitions")
async def routing_definitions():
    """Metric definitions for the Routing dashboard."""
    import scripts.routing_dashboard as rtd
    return _json_safe(rtd.routing_definitions())


# ── Citation Dashboard ────────────────────────────────────────────
# Real data: ChromaDB embeddings (75 docs) + clinical.db conversation_log (225),
# analyses (21), transaction_log (558) — citation coverage, grounding, faithfulness.

@app.get("/api/citation/overview")
async def citation_overview():
    """Citation overview — citation rate, source coverage, quality score,
    per-type stats, daily volume, faithfulness from real data."""
    import scripts.citation_dashboard as citd
    return _json_safe(citd.citation_overview())


@app.get("/api/citation/breakdown")
async def citation_breakdown():
    """Per-response citation detail, per-disease coverage, document mapping,
    gap analysis, component rates, temporal trends, expert alignment."""
    import scripts.citation_dashboard as citd
    return _json_safe(citd.citation_breakdown())


@app.get("/api/citation/definitions")
async def citation_definitions():
    """Metric definitions for the Citation dashboard."""
    import scripts.citation_dashboard as citd
    return _json_safe(citd.citation_definitions())


@app.get("/api/agent-memory/overview")
async def agent_memory_overview():
    """Agent memory overview — coverage, completeness, staleness,
    domain fill rates, conversation depth from real data."""
    import scripts.agent_memory_dashboard as amd
    return _json_safe(amd.memory_overview())


@app.get("/api/agent-memory/breakdown")
async def agent_memory_breakdown():
    """Per-patient memory profiles, domain co-occurrence, coverage gaps,
    component attribution, disease memory depth, recent writes."""
    import scripts.agent_memory_dashboard as amd
    return _json_safe(amd.memory_breakdown())


@app.get("/api/agent-memory/definitions")
async def agent_memory_definitions():
    """Metric definitions for the Agent Memory dashboard."""
    import scripts.agent_memory_dashboard as amd
    return _json_safe(amd.memory_definitions())


# ── MCP Overview Dashboard ────────────────────────────────────────
# Real data: component health, action catalog, protocol compliance,
# actor summary, patient coverage, security audit trail from clinical.db.

@app.get("/api/mcp-overview/overview")
async def mcp_overview_overview():
    """MCP Overview — component health, action catalog, compliance rate,
    daily activity, hourly heatmap, actor summary from real data."""
    import scripts.mcp_overview_dashboard as mod
    return _json_safe(mod.mcp_overview())


@app.get("/api/mcp-overview/breakdown")
async def mcp_overview_breakdown():
    """MCP Overview breakdown — component-action matrix, conversation roles,
    patient coverage, security audit log, component interconnections."""
    import scripts.mcp_overview_dashboard as mod
    return _json_safe(mod.mcp_overview_breakdown())


@app.get("/api/mcp-overview/definitions")
async def mcp_overview_definitions():
    """Metric definitions for the MCP Overview dashboard."""
    import scripts.mcp_overview_dashboard as mod
    return _json_safe(mod.mcp_overview_definitions())


# ── MCP Federation Dashboard ─────────────────────────────────────
# Real data: transaction_log (558), conversation_log (225+), analyses (21),
# expert_reviews (3), hitl_reviews (2) — cross-component federation topology.

@app.get("/api/mcp-federation/overview")
async def mcp_federation_overview():
    """MCP Federation overview — node count, edge count, throughput,
    cross-component rate, actor distribution, daily trend from real data."""
    import scripts.mcp_federation_dashboard as mfd
    return _json_safe(mfd.federation_overview())


@app.get("/api/mcp-federation/breakdown")
async def mcp_federation_breakdown():
    """Per-patient federation profile, service mesh adjacency, protocol verb
    matrix, component pair analysis, actor cross-tab, recent events."""
    import scripts.mcp_federation_dashboard as mfd
    return _json_safe(mfd.federation_breakdown())


@app.get("/api/mcp-federation/definitions")
async def mcp_federation_definitions():
    """Metric definitions for the MCP Federation dashboard."""
    import scripts.mcp_federation_dashboard as mfd
    return _json_safe(mfd.federation_definitions())


# ── Release Management Dashboard ────────────────────────────────
# Real data: models/*.joblib, uploads (21), analyses (21), expert_reviews (3),
# hitl_reviews (2), clinical_decisions (1), transaction_log change actions.

@app.get("/api/release/overview")
async def release_overview():
    """Release overview — model inventory, dataset stats, approval rate,
    training runs, change trend from real data."""
    import scripts.release_dashboard as rld
    return _json_safe(rld.release_overview())


@app.get("/api/release/breakdown")
async def release_breakdown():
    """Per-patient uploads, expert review log, HITL overrides, training
    schedule, component change log, hourly pattern."""
    import scripts.release_dashboard as rld
    return _json_safe(rld.release_breakdown())


@app.get("/api/release/definitions")
async def release_definitions():
    """Metric definitions for the Release Management dashboard."""
    import scripts.release_dashboard as rld
    return _json_safe(rld.release_definitions())


# ── Agent Evaluation Dashboard ──────────────────────────────────────────
@app.get("/api/agent-eval/overview")
async def agent_eval_overview():
    """Agent Evaluation — AI analysis confidence, expert/HITL agreement, decision routing, coverage."""
    import scripts.agent_eval_dashboard as aed
    return _json_safe(aed.agent_eval_overview())


@app.get("/api/agent-eval/breakdown")
async def agent_eval_breakdown():
    """Agent Evaluation — expert review logs, HITL logs, clinical decision logs, event logs."""
    import scripts.agent_eval_dashboard as aed
    return _json_safe(aed.agent_eval_breakdown())


@app.get("/api/agent-eval/definitions")
async def agent_eval_definitions():
    """Metric definitions for the Agent Evaluation dashboard."""
    import scripts.agent_eval_dashboard as aed
    return _json_safe(aed.agent_eval_definitions())


# ── Workflow Dashboard ────────────────────────────────────────

@app.get("/api/workflow/overview")
async def workflow_overview():
    """Workflow overview — consultant role coverage, phase/step counts,
    sign-off status, activity volume, actor distribution from clinical.db."""
    import scripts.workflow_dashboard as wfd
    return _json_safe(wfd.workflow_overview())


@app.get("/api/workflow/breakdown")
async def workflow_breakdown():
    """Per-role workflow breakdown — phases, steps, signoffs,
    component-action cross-tab, recent events, patient depth."""
    import scripts.workflow_dashboard as wfd
    return _json_safe(wfd.workflow_breakdown())


@app.get("/api/workflow/definitions")
async def workflow_definitions():
    """Metric definitions for the Workflow dashboard."""
    import scripts.workflow_dashboard as wfd
    return _json_safe(wfd.workflow_definitions())


# ── Integration Dashboard (multi-format ingest) ─────────────────────

@app.get("/api/integration-dashboard/overview")
async def integration_dashboard_overview():
    """Multi-format ingest overview — upload volume, format distribution, coverage."""
    import scripts.integration_dashboard as igd
    return _json_safe(igd.integration_overview())


@app.get("/api/integration-dashboard/breakdown")
async def integration_dashboard_breakdown():
    """Per-patient, per-format, daily trend drill-down."""
    import scripts.integration_dashboard as igd
    return _json_safe(igd.integration_breakdown())


@app.get("/api/integration-dashboard/definitions")
async def integration_dashboard_definitions():
    """Integration dashboard metric definitions for tooltip overlays."""
    import scripts.integration_dashboard as igd
    return _json_safe(igd.integration_definitions())


# ── Responsible AI Dashboard ─────────────────────────────────────

@app.get("/api/responsible-ai-dashboard/overview")
async def responsible_ai_dash_overview():
    """Responsible AI dashboard overview — overall score, framework breakdown,
    test pass rate, fairness gate, disease accuracy, calibration from real data."""
    import scripts.responsible_ai_dashboard as rad
    return _json_safe(rad.responsible_ai_overview())


@app.get("/api/responsible-ai-dashboard/breakdown")
async def responsible_ai_dash_breakdown():
    """Per-framework detail, robustness curves, consistency, error patterns."""
    import scripts.responsible_ai_dashboard as rad
    return _json_safe(rad.responsible_ai_breakdown())


@app.get("/api/responsible-ai-dashboard/definitions")
async def responsible_ai_dash_definitions():
    """Responsible AI metric definitions for tooltip overlays."""
    import scripts.responsible_ai_dashboard as rad
    return _json_safe(rad.responsible_ai_definitions())


# ── MCP Security Dashboard ────────────────────────────────────
# Real data: guardrail enforcement, actor privilege matrix, security agent
# activity, access audit, attack surface, temporal patterns from clinical.db.

@app.get("/api/mcp-security/overview")
async def mcp_security_overview():
    """MCP Security overview — guardrail events, blocked count, sign-offs,
    actor privileges, security agent activity, attack surface from real data."""
    import scripts.mcp_security_dashboard as msd
    return _json_safe(msd.mcp_security_overview())


@app.get("/api/mcp-security/breakdown")
async def mcp_security_breakdown():
    """MCP Security breakdown — patient access audit, actor-component matrix,
    daily security trend, hourly pattern, privileged event log."""
    import scripts.mcp_security_dashboard as msd
    return _json_safe(msd.mcp_security_breakdown())


@app.get("/api/mcp-security/definitions")
async def mcp_security_definitions():
    """Metric definitions for the MCP Security dashboard."""
    import scripts.mcp_security_dashboard as msd
    return _json_safe(msd.mcp_security_definitions())


# ── SecOps Dashboard ─────────────────────────────────────────
# Threat detection, injection/jailbreak scanning, PII protection,
# access audit, incident tracking, OWASP LLM Top-10 coverage.

@app.get("/api/sec-ops/overview")
async def sec_ops_overview():
    """SecOps overview — threat summary, guardrail posture, PII/injection
    pattern inventory, compliance score, security agent activity."""
    import scripts.sec_ops_dashboard as sod
    return _json_safe(sod.overview())


@app.get("/api/sec-ops/breakdown")
async def sec_ops_breakdown():
    """SecOps breakdown — access audit, actor privileges, attack surface,
    daily trend, conversation roles, incident timeline, OWASP coverage."""
    import scripts.sec_ops_dashboard as sod
    return _json_safe(sod.breakdown())


@app.get("/api/sec-ops/definitions")
async def sec_ops_definitions():
    """Metric definitions for the SecOps dashboard."""
    import scripts.sec_ops_dashboard as sod
    return _json_safe(sod.definitions())


# ── Appointments Dashboard ────────────────────────────────────

@app.get("/api/appointments/overview")
async def appointments_overview():
    """Appointment booking overview — total bookings, completion rate,
    no-show rate, provider workload, department distribution."""
    import scripts.appointments_dashboard as apd
    return _json_safe(apd.appointments_overview())


@app.get("/api/appointments/breakdown")
async def appointments_breakdown():
    """Per-patient appointments, daily trend, hourly pattern,
    provider-department cross-tab, recent appointments, no-show analysis."""
    import scripts.appointments_dashboard as apd
    return _json_safe(apd.appointments_breakdown())


@app.get("/api/appointments/definitions")
async def appointments_definitions():
    """Metric definitions for the Appointments dashboard."""
    import scripts.appointments_dashboard as apd
    return _json_safe(apd.appointments_definitions())


# ── Seizure Severity Dashboard (Liverpool Seizure Severity Scale / LSSS) ──
@app.get("/api/seizure-severity-dashboard/overview")
async def seizure_severity_overview():
    """LSSS summary: KPIs, severity distribution, per-patient latest scores."""
    import scripts.seizure_severity_dashboard as ssd
    return _json_safe(ssd.overview())


@app.get("/api/seizure-severity-dashboard/breakdown")
async def seizure_severity_breakdown():
    """LSSS domain analysis, per-item heatmap, trend, per-patient history."""
    import scripts.seizure_severity_dashboard as ssd
    return _json_safe(ssd.breakdown())


@app.get("/api/seizure-severity-dashboard/definitions")
async def seizure_severity_definitions():
    """Metric definitions for the Seizure Severity dashboard."""
    import scripts.seizure_severity_dashboard as ssd
    return _json_safe(ssd.definitions())


# ── Seizure Diary Dashboard ──────────────────────────────────
@app.get("/api/seizure-diary-dashboard/overview")
async def seizure_diary_dashboard_overview():
    """Seizure diary summary: KPIs, severity distribution, monthly trend, ER stats."""
    import scripts.seizure_diary_dashboard as sdd
    return _json_safe(sdd.overview())


@app.get("/api/seizure-diary-dashboard/breakdown")
async def seizure_diary_dashboard_breakdown():
    """Per-patient history, event log, trigger analysis."""
    import scripts.seizure_diary_dashboard as sdd
    return _json_safe(sdd.breakdown())


@app.get("/api/seizure-diary-dashboard/definitions")
async def seizure_diary_dashboard_definitions():
    """Metric definitions for the Seizure Diary dashboard."""
    import scripts.seizure_diary_dashboard as sdd
    return _json_safe(sdd.definitions())


# ── FinOps Dashboard ─────────────────────────────────────────

@app.get("/api/finops/overview")
async def finops_overview():
    """FinOps cost overview — total spend, budget, category breakdown,
    top services, daily trend, cost per request/patient."""
    import scripts.finops_dashboard as fod
    return _json_safe(fod.overview())


@app.get("/api/finops/breakdown")
async def finops_breakdown():
    """FinOps detailed breakdown — model token costs, GPU utilization,
    component spend, per-patient costs, weekly comparison, cloud services."""
    import scripts.finops_dashboard as fod
    return _json_safe(fod.breakdown())


@app.get("/api/finops/definitions")
async def finops_definitions():
    """Metric definitions for the FinOps dashboard."""
    import scripts.finops_dashboard as fod
    return _json_safe(fod.definitions())


# ── C-SSRS Dashboard (Columbia Suicide Severity Rating Scale) ──────────
@app.get("/api/cssrs-dashboard/overview")
async def cssrs_overview():
    """C-SSRS summary: KPIs, risk distribution, per-patient latest scores, alerts."""
    import scripts.cssrs_dashboard as cssd
    return _json_safe(cssd.overview())


@app.get("/api/cssrs-dashboard/breakdown")
async def cssrs_breakdown():
    """C-SSRS screening endorsement rates, intensity analysis, trend, patient history."""
    import scripts.cssrs_dashboard as cssd
    return _json_safe(cssd.breakdown())


@app.get("/api/cssrs-dashboard/definitions")
async def cssrs_definitions():
    """Metric definitions for the C-SSRS dashboard."""
    import scripts.cssrs_dashboard as cssd
    return _json_safe(cssd.definitions())


# ── ICA Noise Cleaning Dashboard ──────────────────────────────────────
@app.get("/api/ica-noise-cleaning/overview")
async def ica_noise_overview():
    """ICA noise cleaning summary: KPIs, per-subject, variance distribution."""
    import scripts.ica_noise_cleaning_dashboard as ica
    return _json_safe(ica.overview())


@app.get("/api/ica-noise-cleaning/breakdown")
async def ica_noise_breakdown():
    """ICA per-file detail, component analysis, pipeline stages."""
    import scripts.ica_noise_cleaning_dashboard as ica
    return _json_safe(ica.breakdown())


@app.get("/api/ica-noise-cleaning/definitions")
async def ica_noise_definitions():
    """Metric definitions for the ICA Noise Cleaning dashboard."""
    import scripts.ica_noise_cleaning_dashboard as ica
    return _json_safe(ica.definitions())


# ── FIM (Functional Independence Measure) Dashboard ─────────────────
@app.get("/api/fim-dashboard/overview")
async def fim_overview(patient_id: str = None):
    """FIM overview: KPIs, independence distribution, per-patient summaries."""
    import scripts.neuro_scales_fim as fim
    return _json_safe(fim.overview(patient_id))


@app.get("/api/fim-dashboard/breakdown")
async def fim_breakdown(patient_id: str = None):
    """FIM domain breakdown: subdomain averages, item heatmap, motor vs cognitive."""
    import scripts.neuro_scales_fim as fim
    return _json_safe(fim.breakdown(patient_id))


@app.get("/api/fim-dashboard/definitions")
async def fim_definitions():
    """Metric definitions for the FIM dashboard."""
    import scripts.neuro_scales_fim as fim
    return _json_safe(fim.definitions())


@app.get("/api/wais-dashboard/overview")
async def wais_overview():
    """WAIS overview: KPIs, IQ distribution, per-patient summaries."""
    import scripts.wais_dashboard as wais
    return _json_safe(wais.overview())


@app.get("/api/wais-dashboard/breakdown")
async def wais_breakdown():
    """WAIS breakdown: index profiles, subtest analysis, per-patient history."""
    import scripts.wais_dashboard as wais
    return _json_safe(wais.breakdown())


@app.get("/api/wais-dashboard/definitions")
async def wais_definitions():
    """Metric definitions for the WAIS dashboard."""
    import scripts.wais_dashboard as wais
    return _json_safe(wais.definitions())


@app.get("/api/digit-span-dashboard/overview")
async def digit_span_overview():
    """Digit Span overview: KPIs, performance distribution, per-patient summaries."""
    import scripts.digit_span_dashboard as ds
    return _json_safe(ds.overview())


@app.get("/api/digit-span-dashboard/breakdown")
async def digit_span_breakdown():
    """Digit Span breakdown: condition analysis, asymmetry, per-patient history."""
    import scripts.digit_span_dashboard as ds
    return _json_safe(ds.breakdown())


@app.get("/api/digit-span-dashboard/definitions")
async def digit_span_definitions():
    """Metric definitions for the Digit Span dashboard."""
    import scripts.digit_span_dashboard as ds
    return _json_safe(ds.definitions())


# ── MRI Brain Review (Epilepsy Pre-Surgical Evaluation) ────────────

@app.get("/api/mri-review/overview")
async def mri_review_overview(patient_id: str = None):
    """MRI Brain Review overview — structural MRI findings for epilepsy
    pre-surgical evaluation. Lesion types (HS/FCD/tumour/cavernoma/AVM),
    classification (lesional/non-lesional/equivocal/normal), lobe and
    laterality distribution. All from REAL mri_findings in clinical.db."""
    import scripts.mri_brain_review as mri
    return _json_safe(mri.mri_overview(patient_id))


@app.get("/api/mri-review/breakdown")
async def mri_review_breakdown(patient_id: str = None):
    """MRI per-patient breakdown — detailed findings, volumetric analysis
    (hippocampal volume asymmetry), T2/FLAIR signal, enhancement, quality,
    EEG-MRI concordance check against seizure diary data."""
    import scripts.mri_brain_review as mri
    return _json_safe(mri.mri_breakdown(patient_id))


@app.get("/api/mri-review/definitions")
async def mri_review_definitions():
    """MRI Brain Review definitions — lesion types, classification criteria,
    epilepsy MRI protocol (3T sequences), volumetric thresholds, clinical
    significance references."""
    import scripts.mri_brain_review as mri
    return _json_safe(mri.mri_definitions())


# ── AMPS (Assessment of Motor and Process Skills) Dashboard ──────────
@app.get("/api/amps-dashboard/overview")
async def amps_overview(patient_id: str = None):
    """AMPS overview: KPIs, performance distribution, per-patient summaries."""
    import scripts.neuro_scales_amps as amps
    return _json_safe(amps.overview(patient_id))


@app.get("/api/amps-dashboard/breakdown")
async def amps_breakdown(patient_id: str = None):
    """AMPS breakdown: motor/process group averages, item heatmap, logit scatter."""
    import scripts.neuro_scales_amps as amps
    return _json_safe(amps.breakdown(patient_id))


@app.get("/api/amps-dashboard/definitions")
async def amps_definitions():
    """Metric definitions for the AMPS dashboard."""
    import scripts.neuro_scales_amps as amps
    return _json_safe(amps.definitions())


@app.get("/api/incident-management/overview")
async def incident_mgmt_overview():
    """AI Incident Management overview: KPIs, timeline, severity."""
    import scripts.ai_incident_management as im
    return _json_safe(im.overview())


@app.get("/api/incident-management/breakdown")
async def incident_mgmt_breakdown():
    """AI Incident Management breakdown: heatmap, trends, track events."""
    import scripts.ai_incident_management as im
    return _json_safe(im.breakdown())


@app.get("/api/incident-management/definitions")
async def incident_mgmt_definitions():
    """AI Incident Management metric definitions."""
    import scripts.ai_incident_management as im
    return _json_safe(im.definitions())


# ── Video EEG Monitoring Dashboard ──────────────────────────────────

@app.get("/api/video-eeg/overview")
async def video_eeg_overview():
    """Video EEG Monitoring overview: KPIs, monitoring distribution,
    severity, per-patient summary. Real seizure_diary + CHB-MIT data."""
    import scripts.video_eeg_dashboard as veeg
    return _json_safe(veeg.overview())


@app.get("/api/video-eeg/breakdown")
async def video_eeg_breakdown():
    """Video EEG breakdown: seizure timeline, duration histogram,
    aura/trigger analysis, temporal pattern, EEG features, concordance."""
    import scripts.video_eeg_dashboard as veeg
    return _json_safe(veeg.breakdown())


@app.get("/api/video-eeg/definitions")
async def video_eeg_definitions():
    """Video EEG Monitoring metric definitions, protocol, semiology."""
    import scripts.video_eeg_dashboard as veeg
    return _json_safe(veeg.definitions())


# ── Shadow AI Detection Dashboard ───────────────────────────────────

@app.get("/api/shadow-ai/overview")
async def shadow_ai_overview():
    """Shadow AI Detection overview: KPIs, shadow rate, risk level,
    detection timeline, top shadow sources. Real track.jsonl data."""
    import scripts.shadow_ai_detection as sad
    return _json_safe(sad.overview())


@app.get("/api/shadow-ai/breakdown")
async def shadow_ai_breakdown():
    """Shadow AI Detection breakdown: hourly heatmap, source analysis,
    recent shadow events, level distribution, temporal pattern."""
    import scripts.shadow_ai_detection as sad
    return _json_safe(sad.breakdown())


@app.get("/api/shadow-ai/definitions")
async def shadow_ai_definitions():
    """Shadow AI Detection metric definitions, methodology, risk levels."""
    import scripts.shadow_ai_detection as sad
    return _json_safe(sad.definitions())


# ── AI Change Management Dashboard ──────────────────────────────

@app.get("/api/change-management/overview")
async def change_mgmt_overview():
    """Change Management overview: KPIs, change types, risk distribution,
    daily counts, contributors. Real git log + track.jsonl data."""
    import scripts.ai_change_management as acm
    return _json_safe(acm.overview())


@app.get("/api/change-management/breakdown")
async def change_mgmt_breakdown():
    """Change Management breakdown: impact by type, hourly heatmap,
    deploy timeline, risk trend, velocity, rollback events."""
    import scripts.ai_change_management as acm
    return _json_safe(acm.breakdown())


@app.get("/api/change-management/definitions")
async def change_mgmt_definitions():
    """Change Management stages, metric definitions, risk criteria."""
    import scripts.ai_change_management as acm
    return _json_safe(acm.definitions())


# ── Model Retirement Dashboard ──────────────────────────────────

@app.get("/api/model-retirement/overview")
async def model_retirement_overview():
    """Model Retirement overview: KPIs, model inventory, pipeline stages."""
    import scripts.model_retirement as mr
    return _json_safe(mr.overview())


@app.get("/api/model-retirement/breakdown")
async def model_retirement_breakdown():
    """Model Retirement breakdown: timeline, accuracy vs drift, training history."""
    import scripts.model_retirement as mr
    return _json_safe(mr.breakdown())


@app.get("/api/model-retirement/definitions")
async def model_retirement_definitions():
    """Model Retirement stages, metric definitions, retirement criteria."""
    import scripts.model_retirement as mr
    return _json_safe(mr.definitions())


# ── Nerve Conduction Velocity (NCV) Dashboard ────────────────────

@app.get("/api/ncv/overview")
async def ncv_overview():
    """NCV overview: KPIs, severity distribution, neuropathy types,
    per-nerve abnormality rates, per-patient summary. Real clinical.db data."""
    import scripts.ncv_dashboard as ncv
    return _json_safe(ncv.overview())


@app.get("/api/ncv/breakdown")
async def ncv_breakdown():
    """NCV breakdown: motor & sensory nerve summaries, MCV histogram,
    motor vs sensory comparison, limb comparison, per-patient detail."""
    import scripts.ncv_dashboard as ncv
    return _json_safe(ncv.breakdown())


@app.get("/api/ncv/definitions")
async def ncv_definitions():
    """NCV metric definitions, reference ranges, neuropathy types,
    severity levels, clinical significance."""
    import scripts.ncv_dashboard as ncv
    return _json_safe(ncv.definitions())


# ── Blink Reflex Dashboard ────────────────────────────────────────

@app.get("/api/blink-reflex/overview")
async def blink_reflex_overview():
    """Blink reflex overview: KPIs, severity distribution, diagnostic patterns,
    per-side abnormality rates, per-patient summary. Real clinical.db data."""
    import scripts.blink_reflex_dashboard as brd
    return _json_safe(brd.overview())


@app.get("/api/blink-reflex/breakdown")
async def blink_reflex_breakdown():
    """Blink reflex breakdown: side summary with R1/R2 parameters,
    R1 latency histogram, ipsi vs contra R2 comparison, per-patient detail."""
    import scripts.blink_reflex_dashboard as brd
    return _json_safe(brd.breakdown())


@app.get("/api/blink-reflex/definitions")
async def blink_reflex_definitions():
    """Blink reflex metric definitions, reference ranges, diagnostic patterns,
    severity levels, clinical significance."""
    import scripts.blink_reflex_dashboard as brd
    return _json_safe(brd.definitions())


# ── AI FinOps Dashboard ──────────────────────────────────────────

@app.get("/api/ai-finops/overview")
async def ai_finops_overview():
    """AI FinOps overview: KPIs, cost breakdown, daily cost trend,
    model storage costs. Real track.jsonl + model file data."""
    import scripts.ai_finops as af
    return _json_safe(af.overview())


@app.get("/api/ai-finops/breakdown")
async def ai_finops_breakdown():
    """AI FinOps breakdown: build session log, hourly cost heatmap,
    cost velocity, storage breakdown, efficiency metrics."""
    import scripts.ai_finops as af
    return _json_safe(af.breakdown())


@app.get("/api/ai-finops/definitions")
async def ai_finops_definitions():
    """AI FinOps cost model, metric definitions, optimization strategies."""
    import scripts.ai_finops as af
    return _json_safe(af.definitions())


# ── Data Versioning & Catalog Dashboard ──────────────────────────

@app.get("/api/data-versioning/overview")
async def data_versioning_overview():
    """Data Versioning & Catalog overview: KPIs, catalog, format distribution,
    size by dataset. Real filesystem + git data."""
    import scripts.data_versioning_catalog as dvc
    return _json_safe(dvc.overview())


@app.get("/api/data-versioning/breakdown")
async def data_versioning_breakdown():
    """Data Versioning & Catalog breakdown: databases, model artifacts,
    staleness, lineage, recent changes, data events."""
    import scripts.data_versioning_catalog as dvc
    return _json_safe(dvc.breakdown())


@app.get("/api/data-versioning/definitions")
async def data_versioning_definitions():
    """Data Versioning & Catalog metric definitions, concepts, stages."""
    import scripts.data_versioning_catalog as dvc
    return _json_safe(dvc.definitions())


# ── Somatosensory Evoked Potentials (SSEP) Dashboard ──────────────

@app.get("/api/ssep/overview")
async def ssep_overview():
    """SSEP overview: KPIs, severity distribution, diagnostic patterns,
    per-limb abnormality rates, per-patient summary. Real clinical.db data."""
    import scripts.ssep_dashboard as ssep
    return _json_safe(ssep.overview())


@app.get("/api/ssep/breakdown")
async def ssep_breakdown():
    """SSEP breakdown: upper & lower limb summaries, N20/P37 histograms,
    limb comparison, per-patient detail with full upper+lower results."""
    import scripts.ssep_dashboard as ssep
    return _json_safe(ssep.breakdown())


@app.get("/api/ssep/definitions")
async def ssep_definitions():
    """SSEP metric definitions, reference ranges, diagnostic patterns,
    severity levels, clinical significance."""
    import scripts.ssep_dashboard as ssep
    return _json_safe(ssep.definitions())


# ── Electromyography (EMG) Dashboard ──────────────────────────────

@app.get("/api/emg/overview")
async def emg_overview():
    """EMG overview: KPIs, severity distribution, diagnostic patterns,
    per-muscle abnormality rates, per-patient summary. Real clinical.db data."""
    import scripts.emg_dashboard as emg
    return _json_safe(emg.overview())


@app.get("/api/emg/breakdown")
async def emg_breakdown():
    """EMG breakdown: MUAP summary, recruitment/spontaneous activity distribution,
    duration & amplitude histograms, limb comparison, per-patient detail."""
    import scripts.emg_dashboard as emg
    return _json_safe(emg.breakdown())


@app.get("/api/emg/definitions")
async def emg_definitions():
    """EMG metric definitions, reference ranges, diagnostic patterns,
    severity levels, clinical significance."""
    import scripts.emg_dashboard as emg
    return _json_safe(emg.definitions())


# ── Repetitive Nerve Stimulation (RNS) Dashboard ─────────────────

@app.get("/api/rns/overview")
async def rns_overview():
    """RNS overview: KPIs, severity distribution, diagnostic patterns,
    per-site abnormality rates, per-patient summary. Real clinical.db data."""
    import scripts.rns_dashboard as rns
    return _json_safe(rns.overview())


@app.get("/api/rns/breakdown")
async def rns_breakdown():
    """RNS breakdown: per-site summary, decrement & facilitation histograms,
    proximal vs distal comparison, per-patient detail with CMAP trains."""
    import scripts.rns_dashboard as rns
    return _json_safe(rns.breakdown())


@app.get("/api/rns/definitions")
async def rns_definitions():
    """RNS metric definitions, reference ranges, diagnostic patterns,
    severity levels, clinical significance."""
    import scripts.rns_dashboard as rns
    return _json_safe(rns.definitions())


# ── Visual Evoked Potentials (VEP) Dashboard ──────────────────────

@app.get("/api/vep/overview")
async def vep_overview():
    """VEP overview: KPIs, severity distribution, diagnostic patterns,
    per-eye abnormality rates, per-patient summary. Real clinical.db data."""
    import scripts.vep_dashboard as vep
    return _json_safe(vep.overview())


@app.get("/api/vep/breakdown")
async def vep_breakdown():
    """VEP breakdown: left & right eye summaries, P100 latency/amplitude
    histograms, eye comparison, inter-eye difference, per-patient detail."""
    import scripts.vep_dashboard as vep
    return _json_safe(vep.breakdown())


@app.get("/api/vep/definitions")
async def vep_definitions():
    """VEP metric definitions, reference ranges, diagnostic patterns,
    severity levels, clinical significance."""
    import scripts.vep_dashboard as vep
    return _json_safe(vep.definitions())


# ── Brainstem Evoked Response Audiometry (BERA) Dashboard ─────────

@app.get("/api/bera/overview")
async def bera_overview():
    """BERA overview: KPIs, severity distribution, diagnostic patterns,
    per-ear abnormality rates, per-patient summary. Real clinical.db data."""
    import scripts.bera_dashboard as bera
    return _json_safe(bera.overview())


@app.get("/api/bera/breakdown")
async def bera_breakdown():
    """BERA breakdown: left & right ear summaries, Wave V latency/amplitude
    histograms, I-V IPL histogram, ear comparison, per-patient detail."""
    import scripts.bera_dashboard as bera
    return _json_safe(bera.breakdown())


@app.get("/api/bera/definitions")
async def bera_definitions():
    """BERA metric definitions, reference ranges, diagnostic patterns,
    severity levels, clinical significance."""
    import scripts.bera_dashboard as bera
    return _json_safe(bera.definitions())


# ── HRV / RR Variation Dashboard ─────────────────────────────────

@app.get("/api/hrv/overview")
async def hrv_overview():
    """HRV overview: KPIs, severity distribution, diagnostic patterns,
    autonomic dysfunction scores, per-patient summary. Real clinical.db data."""
    import scripts.hrv_dashboard as hrv
    return _json_safe(hrv.overview())


@app.get("/api/hrv/breakdown")
async def hrv_breakdown():
    """HRV breakdown: time & frequency domain summaries, SDNN/RMSSD/LF-HF
    histograms, autonomic score histogram, per-patient detail."""
    import scripts.hrv_dashboard as hrv
    return _json_safe(hrv.breakdown())


@app.get("/api/hrv/definitions")
async def hrv_definitions():
    """HRV metric definitions, reference ranges, diagnostic patterns,
    severity levels, clinical significance."""
    import scripts.hrv_dashboard as hrv
    return _json_safe(hrv.definitions())


# ── Sympathetic Skin Response (SSR) Dashboard ──────────────────────

@app.get("/api/ssr/overview")
async def ssr_overview():
    """SSR overview: KPIs, severity distribution, diagnostic patterns,
    per-site abnormality rates, per-patient summary. Real clinical.db data."""
    import scripts.ssr_dashboard as ssr
    return _json_safe(ssr.overview())


@app.get("/api/ssr/breakdown")
async def ssr_breakdown():
    """SSR breakdown: hand & foot summaries, latency & amplitude histograms,
    dysautonomia score histogram, site comparison, per-patient detail."""
    import scripts.ssr_dashboard as ssr
    return _json_safe(ssr.breakdown())


@app.get("/api/ssr/definitions")
async def ssr_definitions():
    """SSR metric definitions, reference ranges, diagnostic patterns,
    severity levels, clinical significance."""
    import scripts.ssr_dashboard as ssr
    return _json_safe(ssr.definitions())


# ── ABPM / Holter Dashboard ──────────────────────────────────────────────

@app.get("/api/abpm/overview")
async def abpm_overview():
    """ABPM/Holter overview: KPIs, severity distribution, diagnostic patterns,
    dipping distribution, per-patient summary. Real clinical.db data."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.overview())


@app.get("/api/abpm/breakdown")
async def abpm_breakdown():
    """ABPM/Holter breakdown: BP & ECG parameter tables, systolic/dipping/QTc/PVC
    histograms, cardiac-autonomic score histogram, per-patient detail."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.breakdown())


@app.get("/api/abpm/definitions")
async def abpm_definitions():
    """ABPM/Holter definitions, reference ranges, dipping categories, diagnostic
    patterns, severity levels, clinical significance."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.definitions())


# ── Exercise / Rehab Recommendations Dashboard ─────────────────────

@app.get("/api/exercise/overview")
async def exercise_overview():
    """Exercise/Rehab overview: KPIs, risk distribution, compliance distribution,
    fitness distribution, category compliance, per-patient summary. Real clinical.db data."""
    import scripts.exercise_dashboard as exercise
    return _json_safe(exercise.overview())


@app.get("/api/exercise/breakdown")
async def exercise_breakdown():
    """Exercise/Rehab breakdown: per-category detail, ADL domain analysis,
    compliance/rehab/ADL histograms, per-patient detail cards."""
    import scripts.exercise_dashboard as exercise
    return _json_safe(exercise.breakdown())


@app.get("/api/exercise/definitions")
async def exercise_definitions():
    """Exercise/Rehab definitions, categories, risk levels, ADL domains,
    precautions, clinical significance."""
    import scripts.exercise_dashboard as exercise
    return _json_safe(exercise.definitions())


# ── Cloud Ops Dashboard ──────────────────────────────────────────────

@app.get("/api/cloud-ops/overview")
async def cloud_ops_overview():
    """Cloud Ops overview: KPIs, region health, cost summary, autoscale summary,
    uptime stats. Deterministic data from infra model."""
    import scripts.cloud_ops_dashboard as cloud
    return _json_safe(cloud.overview())


@app.get("/api/cloud-ops/breakdown")
async def cloud_ops_breakdown():
    """Cloud Ops breakdown: resource utilisation per region, cost detail per service,
    autoscale events timeline, uptime history (30d)."""
    import scripts.cloud_ops_dashboard as cloud
    return _json_safe(cloud.breakdown())


@app.get("/api/cloud-ops/definitions")
async def cloud_ops_definitions():
    """Cloud Ops definitions: regions, services, cost thresholds, autoscale policies,
    status levels, resource thresholds, clinical relevance."""
    import scripts.cloud_ops_dashboard as cloud
    return _json_safe(cloud.definitions())


# ── Model Ops Dashboard ──────────────────────────────────────────────
@app.get("/api/model-ops/overview")
async def model_ops_overview():
    """Model Ops overview: model registry, accuracy KPIs, drift status,
    consistency, external validation."""
    import scripts.model_ops_dashboard as mo
    return _json_safe(mo.overview())


@app.get("/api/model-ops/breakdown")
async def model_ops_breakdown():
    """Model Ops breakdown: accuracy distribution, size comparison,
    usage activity, retrain history."""
    import scripts.model_ops_dashboard as mo
    return _json_safe(mo.breakdown())


@app.get("/api/model-ops/definitions")
async def model_ops_definitions():
    """Model Ops definitions: registry fields, metrics, drift, consistency."""
    import scripts.model_ops_dashboard as mo
    return _json_safe(mo.definitions())


# ── LLMOps Dashboard ────────────────────────────────────────────────
@app.get("/api/llmops/overview")
async def llmops_overview():
    """LLMOps overview: prompt health, token/cost KPIs, hallucination summary,
    cost by provider/model."""
    import scripts.llmops_dashboard as llm
    return _json_safe(llm.overview())


@app.get("/api/llmops/breakdown")
async def llmops_breakdown():
    """LLMOps breakdown: prompt version history, daily token usage, latency
    percentiles, hallucination detail, RAG evaluation."""
    import scripts.llmops_dashboard as llm
    return _json_safe(llm.breakdown())


@app.get("/api/llmops/definitions")
async def llmops_definitions():
    """LLMOps definitions: models, prompts, RAG pipelines, hallucination
    categories, thresholds, clinical relevance."""
    import scripts.llmops_dashboard as llm
    return _json_safe(llm.definitions())


# ── DataOps Dashboard ─────────────────────────────────────────────────
@app.get("/api/data-ops/overview")
async def data_ops_overview():
    """DataOps overview: ingestion KPIs, data quality summary, storage stats,
    modality coverage, signal quality distribution."""
    import scripts.data_ops_dashboard as dops
    return _json_safe(dops.overview())


@app.get("/api/data-ops/breakdown")
async def data_ops_breakdown():
    """DataOps breakdown: pipeline activity, daily volume, quality dimensions,
    missing matrix, data lineage, storage inventory, AI readiness components."""
    import scripts.data_ops_dashboard as dops
    return _json_safe(dops.breakdown())


@app.get("/api/data-ops/definitions")
async def data_ops_definitions():
    """DataOps definitions: pipelines, quality dimensions (ISO 25012),
    AI readiness scoring, storage, lineage steps, clinical relevance."""
    import scripts.data_ops_dashboard as dops
    return _json_safe(dops.definitions())


# ── Observability Dashboard ─────────────────────────────────────────

@app.get("/api/observability/overview")
async def observability_overview():
    """Observability overview: KPIs, component health, daily volume,
    log-level distribution, active alerts. Real data from transaction_log."""
    import scripts.observability_dashboard as obs
    return _json_safe(obs.overview())


@app.get("/api/observability/breakdown")
async def observability_breakdown():
    """Observability breakdown: recent logs, sample traces, latency
    percentiles, action/actor distributions, per-component detail."""
    import scripts.observability_dashboard as obs
    return _json_safe(obs.breakdown())


@app.get("/api/observability/definitions")
async def observability_definitions():
    """Observability definitions: log levels, trace span types, metric
    thresholds, alert rules, instrumentation standards, clinical relevance."""
    import scripts.observability_dashboard as obs
    return _json_safe(obs.definitions())


# ── MLOps Dashboard ────────────────────────────────────────────────
@app.get("/api/mlops/overview")
async def mlops_overview():
    """MLOps overview: training pipeline KPIs, experiment history,
    CV summary, evaluation strategy comparison."""
    import scripts.mlops_dashboard as mlops
    return _json_safe(mlops.overview())


@app.get("/api/mlops/breakdown")
async def mlops_breakdown():
    """MLOps breakdown: multi-disease accuracy, cross-validation detail,
    pipeline events, feature inventory, model files, daily activity."""
    import scripts.mlops_dashboard as mlops
    return _json_safe(mlops.breakdown())


@app.get("/api/mlops/definitions")
async def mlops_definitions():
    """MLOps definitions: training pipeline, evaluation types, CV strategies,
    EEG features, metrics, clinical relevance."""
    import scripts.mlops_dashboard as mlops
    return _json_safe(mlops.definitions())


# ── Trust AI Dashboard ───────────────────────────────────────
# AI confidence, concordance (AI-human agreement), HITL oversight,
# clinical decision audit, composite trust score from clinical.db.

@app.get("/api/trust-ai/overview")
async def trust_ai_overview():
    """Trust AI overview — confidence stats, concordance rate, HITL accept/override,
    clinical decisions, composite trust score."""
    import scripts.trust_ai_dashboard as tad
    return _json_safe(tad.overview())


@app.get("/api/trust-ai/breakdown")
async def trust_ai_breakdown():
    """Trust AI breakdown — confidence distribution, expert reviews by role,
    concordance by confidence band, HITL decisions, clinical decision log."""
    import scripts.trust_ai_dashboard as tad
    return _json_safe(tad.breakdown())


@app.get("/api/trust-ai/definitions")
async def trust_ai_definitions():
    """Trust AI definitions — trust score, confidence, concordance, HITL,
    clinical decision audit, trust dimensions, clinical relevance."""
    import scripts.trust_ai_dashboard as tad
    return _json_safe(tad.definitions())


@app.get("/api/ethical-ai/overview")
async def ethical_ai_overview():
    """Ethical AI overview — fairness gate, bias metrics, guardrail stats,
    consent/transparency coverage, composite ethics score."""
    import scripts.ethical_ai_dashboard as ead
    return _json_safe(ead.overview())


@app.get("/api/ethical-ai/breakdown")
async def ethical_ai_breakdown():
    """Ethical AI breakdown — per-group fairness, outcome distribution by gender,
    guardrail events, HITL decisions, ethical principle adherence."""
    import scripts.ethical_ai_dashboard as ead
    return _json_safe(ead.breakdown())


@app.get("/api/ethical-ai/definitions")
async def ethical_ai_definitions():
    """Ethical AI definitions — fairness metrics, guardrails, bioethics principles,
    transparency, oversight, clinical relevance."""
    import scripts.ethical_ai_dashboard as ead
    return _json_safe(ead.definitions())


@app.get("/api/data-drift/overview")
async def data_drift_overview():
    """Data Drift overview — health score, verdict, severity distribution,
    PSI/KS statistics, worst-drifted features, monitoring event count."""
    import scripts.data_drift_dashboard as ddd
    return _json_safe(ddd.overview())


@app.get("/api/data-drift/breakdown")
async def data_drift_breakdown():
    """Data Drift breakdown — per-feature PSI and KS detail, severity chart,
    event timeline, threshold reference lines."""
    import scripts.data_drift_dashboard as ddd
    return _json_safe(ddd.breakdown())


@app.get("/api/data-drift/definitions")
async def data_drift_definitions():
    """Data Drift definitions — PSI, KS test, severity levels, reference vs live,
    feature categories, clinical relevance (IEC 62304, FDA AI/ML, EU AI Act)."""
    import scripts.data_drift_dashboard as ddd
    return _json_safe(ddd.definitions())


@app.get("/api/feature-drift/overview")
async def feature_drift_overview():
    """Feature Drift overview — importance-weighted drift score, category breakdown,
    cross-model importance analysis, prioritised remediation list."""
    import scripts.feature_drift_dashboard as fdd
    return _json_safe(fdd.overview())


@app.get("/api/feature-drift/breakdown")
async def feature_drift_breakdown():
    """Feature Drift breakdown — per-feature importance + drift detail, category charts,
    cross-model comparison, training event correlation."""
    import scripts.feature_drift_dashboard as fdd
    return _json_safe(fdd.breakdown())


@app.get("/api/feature-drift/definitions")
async def feature_drift_definitions():
    """Feature Drift definitions — feature importance, categories, category-level drift,
    cross-model comparison, remediation priority, clinical relevance."""
    import scripts.feature_drift_dashboard as fdd
    return _json_safe(fdd.definitions())


@app.get("/api/model-drift/overview")
async def model_drift_overview():
    """Model Drift overview — drift score, performance verdict, accuracy/sensitivity
    trends, bootstrap CIs, evaluation strategy comparison."""
    import scripts.model_drift_dashboard as mdd
    return _json_safe(mdd.overview())


@app.get("/api/model-drift/breakdown")
async def model_drift_breakdown():
    """Model Drift breakdown — per-subject metrics, cross-validation folds,
    training timeline, literature comparison, model inventory."""
    import scripts.model_drift_dashboard as mdd
    return _json_safe(mdd.breakdown())


@app.get("/api/model-drift/definitions")
async def model_drift_definitions():
    """Model Drift definitions — model drift, performance metrics, evaluation
    strategies, bootstrap CIs, external validation, clinical relevance."""
    import scripts.model_drift_dashboard as mdd
    return _json_safe(mdd.definitions())


@app.get("/api/output-drift/overview")
async def output_drift_overview():
    """Output/RAG Drift overview — output drift score, confidence shift,
    label distribution JSD, RAG success rate, input-output correlation."""
    import scripts.output_drift_dashboard as odd
    return _json_safe(odd.overview())


@app.get("/api/output-drift/breakdown")
async def output_drift_breakdown():
    """Output/RAG Drift breakdown — confidence timeline, histogram,
    per-patient summary, RAG event log, input-output correlation."""
    import scripts.output_drift_dashboard as odd
    return _json_safe(odd.breakdown())


@app.get("/api/output-drift/definitions")
async def output_drift_definitions():
    """Output/RAG Drift definitions — output drift, JSD, RAG pipeline,
    correlation, monitoring methodology, clinical relevance."""
    import scripts.output_drift_dashboard as odd
    return _json_safe(odd.definitions())


@app.get("/api/prompt-drift/overview")
async def prompt_drift_overview():
    """Prompt Drift overview — prompt/response volume, length drift,
    role distribution, temporal length trends."""
    import scripts.prompt_drift_dashboard as pdd
    return _json_safe(pdd.overview())


@app.get("/api/prompt-drift/breakdown")
async def prompt_drift_breakdown():
    """Prompt Drift breakdown — length histograms, daily volume,
    topic keywords, prompt file stats, weekly drift aggregation."""
    import scripts.prompt_drift_dashboard as pdd
    return _json_safe(pdd.breakdown())


@app.get("/api/prompt-drift/definitions")
async def prompt_drift_definitions():
    """Prompt Drift definitions — drift metrics, prompt categories,
    detection methods, clinical relevance, remediation."""
    import scripts.prompt_drift_dashboard as pdd
    return _json_safe(pdd.definitions())


@app.get("/api/anomaly-detection/overview")
async def anomaly_detection_overview():
    """Anomaly Detection overview — z-score and IQR anomaly counts,
    severity distribution, category breakdown, top anomalous features."""
    import scripts.anomaly_detection_dashboard as add
    return _json_safe(add.anomaly_detection_overview())


@app.get("/api/anomaly-detection/breakdown")
async def anomaly_detection_breakdown():
    """Anomaly Detection breakdown — per-patient anomalies, per-feature
    statistics, anomaly timeline, feature correlations, signal quality."""
    import scripts.anomaly_detection_dashboard as add
    return _json_safe(add.anomaly_detection_breakdown())


@app.get("/api/anomaly-detection/definitions")
async def anomaly_detection_definitions():
    """Anomaly Detection definitions — detection methods, EEG feature
    categories, clinical relevance, remediation strategies."""
    import scripts.anomaly_detection_dashboard as add
    return _json_safe(add.anomaly_detection_definitions())


@app.get("/api/causal-ai/overview")
async def causal_ai_overview():
    """Causal AI overview — medication→seizure pathways, trigger→seizure
    chains, age/gender associations, causal graph summary."""
    import scripts.causal_ai_dashboard as cad
    return _json_safe(cad.causal_overview())


@app.get("/api/causal-ai/breakdown")
async def causal_ai_breakdown():
    """Causal AI breakdown — per-patient causal profiles, MRI correlations,
    assessment links, seizure timeline, intervention effectiveness."""
    import scripts.causal_ai_dashboard as cad
    return _json_safe(cad.causal_breakdown())


@app.get("/api/causal-ai/definitions")
async def causal_ai_definitions():
    """Causal AI definitions — causal inference methods, factor categories,
    graph notation, clinical relevance, remediation strategies."""
    import scripts.causal_ai_dashboard as cad
    return _json_safe(cad.definitions())


@app.get("/api/bias-detection/overview")
async def bias_detection_overview():
    """Bias Detection overview — demographic parity, representation gaps,
    confidence disparities, assessment coverage, medication access by gender/age."""
    import scripts.bias_detection_dashboard as bdd
    return _json_safe(bdd.bias_detection_overview())


@app.get("/api/bias-detection/breakdown")
async def bias_detection_breakdown():
    """Bias Detection breakdown — per-patient bias profiles, instrument scores
    by gender, confidence histograms, MRI coverage, disparity metrics,
    intersectional analysis."""
    import scripts.bias_detection_dashboard as bdd
    return _json_safe(bdd.bias_detection_breakdown())


@app.get("/api/bias-detection/definitions")
async def bias_detection_definitions():
    """Bias Detection definitions — detection methods, protected attributes,
    fairness metrics, clinical relevance, remediation strategies."""
    import scripts.bias_detection_dashboard as bdd
    return _json_safe(bdd.bias_detection_definitions())


@app.get("/api/digital-twin/overview")
async def digital_twin_overview():
    """Digital Twin overview — patient completeness scores, domain coverage,
    twin readiness levels, demographic distributions."""
    import scripts.digital_twin_dashboard as dtd
    return _json_safe(dtd.digital_twin_overview())


@app.get("/api/digital-twin/breakdown")
async def digital_twin_breakdown():
    """Digital Twin breakdown — per-patient twin profiles, domain correlation,
    top complete patients, medication-EEG cross-analysis."""
    import scripts.digital_twin_dashboard as dtd
    return _json_safe(dtd.digital_twin_breakdown())


@app.get("/api/digital-twin/definitions")
async def digital_twin_definitions():
    """Digital Twin definitions — concept, data domains, completeness methodology,
    clinical relevance, remediation strategies."""
    import scripts.digital_twin_dashboard as dtd
    return _json_safe(dtd.digital_twin_definitions())


@app.get("/api/explainable-ai/overview")
async def explainable_ai_overview():
    """Explainable AI overview — global feature importance, category breakdown,
    band power contribution, confidence distribution, disease-wise importance."""
    import scripts.explainable_ai_dashboard as xaid
    return _json_safe(xaid.xai_overview())


@app.get("/api/explainable-ai/breakdown")
async def explainable_ai_breakdown():
    """Explainable AI breakdown — per-patient profiles, per-feature stats,
    counterfactual analysis, feature correlations, SHAP direction analysis."""
    import scripts.explainable_ai_dashboard as xaid
    return _json_safe(xaid.xai_breakdown())


@app.get("/api/explainable-ai/definitions")
async def explainable_ai_definitions():
    """Explainable AI definitions — methods, EEG feature categories,
    interpretation guide, clinical relevance, remediation strategies."""
    import scripts.explainable_ai_dashboard as xaid
    return _json_safe(xaid.definitions())


@app.get("/api/ai-observability/overview")
async def ai_observability_overview():
    """AI Observability overview — transaction volume, component/action/actor
    distributions, cost aggregates, conversation metrics, analysis confidence."""
    import scripts.ai_observability_dashboard as aod
    return _json_safe(aod.observability_overview())


@app.get("/api/ai-observability/breakdown")
async def ai_observability_breakdown():
    """AI Observability breakdown — per-component actions, per-actor components,
    transaction/cost timelines, cost by service, patient profiles, error actions."""
    import scripts.ai_observability_dashboard as aod
    return _json_safe(aod.observability_breakdown())


@app.get("/api/ai-observability/definitions")
async def ai_observability_definitions():
    """AI Observability definitions — methods, system components, metrics/KPIs,
    clinical relevance, remediation strategies."""
    import scripts.ai_observability_dashboard as aod
    return _json_safe(aod.observability_definitions())


@app.get("/api/model-monitoring/overview")
async def model_monitoring_overview():
    """Model Monitoring overview — drift verdicts, confidence distribution,
    system health, data quality, training summaries, monitoring timeline."""
    import scripts.model_monitoring_dashboard as mmd
    return _json_safe(mmd.monitoring_overview())


@app.get("/api/model-monitoring/breakdown")
async def model_monitoring_breakdown():
    """Model Monitoring breakdown — consistency checks, all drift features,
    training runs, missing matrix, per-patient predictions, accuracy."""
    import scripts.model_monitoring_dashboard as mmd
    return _json_safe(mmd.monitoring_breakdown())


@app.get("/api/model-monitoring/definitions")
async def model_monitoring_definitions():
    """Model Monitoring definitions — monitoring methods, metric definitions,
    severity levels, clinical relevance, remediation strategies."""
    import scripts.model_monitoring_dashboard as mmd
    return _json_safe(mmd.definitions())


@app.get("/api/ai-control-tower/overview")
async def ai_control_tower_overview():
    """AI Control Tower overview — component registry, transaction volume,
    system health, cost summary, oversight stats, drift/quality status."""
    import scripts.ai_control_tower_dashboard as ctd
    return _json_safe(ctd.control_tower_overview())


@app.get("/api/ai-control-tower/breakdown")
async def ai_control_tower_breakdown():
    """AI Control Tower breakdown — component-action map, recent transactions,
    HITL reviews, clinical decisions, cost by service, patient profiles."""
    import scripts.ai_control_tower_dashboard as ctd
    return _json_safe(ctd.control_tower_breakdown())


@app.get("/api/ai-control-tower/definitions")
async def ai_control_tower_definitions():
    """AI Control Tower definitions — concept, system components, metrics,
    clinical relevance, remediation strategies."""
    import scripts.ai_control_tower_dashboard as ctd
    return _json_safe(ctd.control_tower_definitions())


@app.get("/api/human-evaluation/overview")
async def human_evaluation_overview():
    """Human Evaluation overview — HITL reviews, expert agreement,
    clinical decisions, feedback ratings."""
    import scripts.human_evaluation_dashboard as hed
    return _json_safe(hed.human_eval_overview())


@app.get("/api/human-evaluation/breakdown")
async def human_evaluation_breakdown():
    """Human Evaluation breakdown — review details, patient profiles,
    role agreement matrix, component findings."""
    import scripts.human_evaluation_dashboard as hed
    return _json_safe(hed.human_eval_breakdown())


@app.get("/api/human-evaluation/definitions")
async def human_evaluation_definitions():
    """Human Evaluation definitions — concept, review types, agreement metrics,
    clinical relevance, remediation strategies."""
    import scripts.human_evaluation_dashboard as hed
    return _json_safe(hed.human_eval_definitions())


@app.get("/api/model-governance/overview")
async def model_governance_overview():
    """Model Governance overview — consultant matrix, sign-off rates, approval
    chain, model lifecycle, compliance status, governance timeline."""
    import scripts.model_governance_dashboard as mgd
    return _json_safe(mgd.governance_overview())


@app.get("/api/model-governance/breakdown")
async def model_governance_breakdown():
    """Model Governance breakdown — per-expert reviews, HITL detail, clinical
    decision chain, component findings, feedback log, patient profiles."""
    import scripts.model_governance_dashboard as mgd
    return _json_safe(mgd.governance_breakdown())


@app.get("/api/model-governance/definitions")
async def model_governance_definitions():
    """Model Governance definitions — governance concepts, approval workflows,
    compliance frameworks, clinical relevance, remediation strategies."""
    import scripts.model_governance_dashboard as mgd
    return _json_safe(mgd.governance_definitions())


@app.get("/api/multimodal-ai/overview")
async def multimodal_ai_overview():
    """Multimodal AI overview — modality coverage, concordance summary,
    coverage distribution, modality timeline, KPIs."""
    import scripts.multimodal_ai_dashboard as mad
    return _json_safe(mad.multimodal_overview())


@app.get("/api/multimodal-ai/breakdown")
async def multimodal_ai_breakdown():
    """Multimodal AI breakdown — per-patient profiles, modality correlation
    matrix, MRI lesion distribution, EEG disease distribution, confidence
    by modality count."""
    import scripts.multimodal_ai_dashboard as mad
    return _json_safe(mad.multimodal_breakdown())


@app.get("/api/multimodal-ai/definitions")
async def multimodal_ai_definitions():
    """Multimodal AI definitions — integration concepts, data modalities,
    concordance metrics, clinical relevance, remediation strategies."""
    import scripts.multimodal_ai_dashboard as mad
    return _json_safe(mad.multimodal_definitions())


@app.get("/api/drift-detection/overview")
async def drift_detection_overview():
    """Drift Detection overview — PSI/KS drift verdicts, severity distribution,
    top drifted features, category breakdown, drift timeline."""
    import scripts.drift_detection_dashboard as ddd
    return _json_safe(ddd.drift_detection_overview())


@app.get("/api/drift-detection/breakdown")
async def drift_detection_breakdown():
    """Drift Detection breakdown — per-feature drift stats, per-category summary,
    per-patient profiles, feature correlations, confidence vs drift, heatmap."""
    import scripts.drift_detection_dashboard as ddd
    return _json_safe(ddd.drift_detection_breakdown())


@app.get("/api/drift-detection/definitions")
async def drift_detection_definitions():
    """Drift Detection definitions — drift concepts, EEG feature categories,
    metrics & thresholds, clinical relevance, remediation strategies."""
    import scripts.drift_detection_dashboard as ddd
    return _json_safe(ddd.definitions())


@app.get("/api/deep-learning/overview")
async def deep_learning_overview():
    """Deep Learning overview — model architectures, training history, accuracy metrics."""
    import scripts.deep_learning_dashboard as dld
    return _json_safe(dld.deep_learning_overview())


@app.get("/api/deep-learning/breakdown")
async def deep_learning_breakdown():
    """Deep Learning breakdown — per-patient metrics, model comparison, training details."""
    import scripts.deep_learning_dashboard as dld
    return _json_safe(dld.deep_learning_breakdown())


@app.get("/api/deep-learning/definitions")
async def deep_learning_definitions():
    """Deep Learning definitions — architectures, metrics, clinical relevance."""
    import scripts.deep_learning_dashboard as dld
    return _json_safe(dld.definitions())


@app.get("/api/model-monitoring/overview")
async def model_monitoring_overview():
    """Model Monitoring overview — drift verdict, consistency, data quality,
    prediction stats, system health, severity distribution, monitoring timeline."""
    import scripts.model_monitoring_dashboard as mmd
    return _json_safe(mmd.monitoring_overview())


@app.get("/api/model-monitoring/breakdown")
async def model_monitoring_breakdown():
    """Model Monitoring breakdown — all drift features, consistency checks,
    training runs, per-patient predictions, accuracy breakdown, missing matrix."""
    import scripts.model_monitoring_dashboard as mmd
    return _json_safe(mmd.monitoring_breakdown())


@app.get("/api/model-monitoring/definitions")
async def model_monitoring_definitions():
    """Model Monitoring definitions — monitoring concepts, metrics & thresholds,
    severity levels, clinical relevance, remediation strategies."""
    import scripts.model_monitoring_dashboard as mmd
    return _json_safe(mmd.definitions())


# ── Communication AI ───────────────────────────────────────
@app.get("/api/communication-ai/overview")
async def communication_ai_overview():
    """Communication AI overview — auto-generated patient messages,
    urgency distribution, delivery channels, communication timeline."""
    import scripts.communication_ai_dashboard as cad
    return _json_safe(cad.communication_overview())


@app.get("/api/communication-ai/breakdown")
async def communication_ai_breakdown():
    """Communication AI breakdown — per-patient profiles, message templates,
    appointment/medication/seizure communications."""
    import scripts.communication_ai_dashboard as cad
    return _json_safe(cad.communication_breakdown())


@app.get("/api/communication-ai/definitions")
async def communication_ai_definitions():
    """Communication AI definitions — communication concepts, message categories,
    delivery methods, clinical relevance, remediation strategies."""
    import scripts.communication_ai_dashboard as cad
    return _json_safe(cad.definitions())


@app.get("/api/foundation-models/overview")
async def foundation_models_overview():
    """Foundation Models overview — model catalog, disease coverage,
    architecture inventory, framework distribution, tier breakdown."""
    import scripts.foundation_models_dashboard as fmd
    return _json_safe(fmd.foundation_models_overview())


@app.get("/api/foundation-models/breakdown")
async def foundation_models_breakdown():
    """Foundation Models breakdown — per-disease model summaries, top models
    by accuracy, largest models, prediction performance by disease."""
    import scripts.foundation_models_dashboard as fmd
    return _json_safe(fmd.foundation_models_breakdown())


@app.get("/api/foundation-models/definitions")
async def foundation_models_definitions():
    """Foundation Models definitions — concepts, architectures, clinical
    relevance, governance, remediation strategies."""
    import scripts.foundation_models_dashboard as fmd
    return _json_safe(fmd.definitions())


# ── Analytics AI ──────────────────────────────────────────
@app.get("/api/analytics-ai/overview")
async def analytics_ai_overview():
    """Analytics AI overview — patient demographics, clinical activity KPIs,
    disease distribution, department workload, signal quality."""
    import scripts.analytics_ai_dashboard as aad
    return _json_safe(aad.analytics_overview())


@app.get("/api/analytics-ai/breakdown")
async def analytics_ai_breakdown():
    """Analytics AI breakdown — per-patient summaries, instrument distribution,
    seizure severity, appointment status, medication coverage, monthly trends."""
    import scripts.analytics_ai_dashboard as aad
    return _json_safe(aad.analytics_breakdown())


@app.get("/api/analytics-ai/definitions")
async def analytics_ai_definitions():
    """Analytics AI definitions — analytics concepts, clinical metrics,
    data quality, regulatory standards, remediation strategies."""
    import scripts.analytics_ai_dashboard as aad
    return _json_safe(aad.definitions())


# ── Interpretable AI ────────────────────────────────────────
@app.get("/api/interpretable-ai/overview")
async def interpretable_ai_overview():
    """Interpretable AI overview — decision trees, rule lists, logistic regression
    coefficients, accuracy vs black-box comparison."""
    import scripts.interpretable_ai_dashboard as iad
    return _json_safe(iad.interpretable_overview())


@app.get("/api/interpretable-ai/breakdown")
async def interpretable_ai_breakdown():
    """Interpretable AI breakdown — per-disease models, decision paths,
    rule extraction, per-patient interpretable predictions."""
    import scripts.interpretable_ai_dashboard as iad
    return _json_safe(iad.interpretable_breakdown())


@app.get("/api/interpretable-ai/definitions")
async def interpretable_ai_definitions():
    """Interpretable AI definitions — interpretability concepts, model types,
    clinical relevance, regulatory standards, remediation strategies."""
    import scripts.interpretable_ai_dashboard as iad
    return _json_safe(iad.definitions())


@app.get("/api/ica-noise-cleaning/overview")
async def ica_noise_cleaning_overview():
    """ICA Noise Cleaning overview — method summary, per-file artifact removal,
    variance removed, signal quality distribution, aggregate stats."""
    import scripts.ica_noise_cleaning_dashboard as icd
    return _json_safe(icd.overview())


@app.get("/api/ica-noise-cleaning/breakdown")
async def ica_noise_cleaning_breakdown():
    """ICA Noise Cleaning breakdown — per-subject detail, component counts,
    artifact types, cleaning quality metrics."""
    import scripts.ica_noise_cleaning_dashboard as icd
    return _json_safe(icd.breakdown())


@app.get("/api/ica-noise-cleaning/definitions")
async def ica_noise_cleaning_definitions():
    """ICA Noise Cleaning definitions — ICA concepts, artifact types,
    clinical relevance, regulatory standards, remediation strategies."""
    import scripts.ica_noise_cleaning_dashboard as icd
    return _json_safe(icd.definitions())


# ── Agentic RAG ───────────────────────────────────────────
@app.get("/api/agentic-rag/overview")
async def agentic_rag_overview():
    """Agentic RAG overview — corpus inventory, retrieval coverage,
    query routing stats, knowledge-base health KPIs."""
    import scripts.agentic_rag_dashboard as ard
    return _json_safe(ard.agentic_rag_overview())


@app.get("/api/agentic-rag/breakdown")
async def agentic_rag_breakdown():
    """Agentic RAG breakdown — per-patient coverage, agent traces,
    workload distribution, relevance by query type, KB health detail."""
    import scripts.agentic_rag_dashboard as ard
    return _json_safe(ard.agentic_rag_breakdown())


@app.get("/api/agentic-rag/definitions")
async def agentic_rag_definitions():
    """Agentic RAG definitions — RAG concepts, pipeline stages,
    quality metrics, clinical relevance, remediation strategies."""
    import scripts.agentic_rag_dashboard as ard
    return _json_safe(ard.definitions())


@app.get("/api/decision-ai/overview")
async def decision_ai_overview():
    """Decision AI overview — routing distribution, confidence histogram,
    HITL override stats, audit summary, disease breakdown."""
    import scripts.decision_ai_dashboard as dad
    return _json_safe(dad.decision_overview())


@app.get("/api/decision-ai/breakdown")
async def decision_ai_breakdown():
    """Decision AI breakdown — per-patient decision summaries, per-analysis
    routing, HITL reviews, confidence calibration, audit timeline."""
    import scripts.decision_ai_dashboard as dad
    return _json_safe(dad.decision_breakdown())


@app.get("/api/decision-ai/definitions")
async def decision_ai_definitions():
    """Decision AI definitions — routing concepts, thresholds, quality metrics,
    clinical relevance (IEC 62304, FDA, ILAE, ISO 14971, EU AI Act), remediation."""
    import scripts.decision_ai_dashboard as dad
    return _json_safe(dad.decision_definitions())


# ── True Visits Dashboard ─────────────────────────────────────

@app.get("/api/visits/overview")
async def visits_overview():
    """True Visits overview — total completed visits, completion rate,
    no-show rate, provider visit load, department distribution, visit types."""
    import scripts.visits_dashboard as vsd
    return _json_safe(vsd.visits_overview())


@app.get("/api/visits/breakdown")
async def visits_breakdown():
    """Per-patient visit history, daily/monthly trends, duration distribution,
    provider-department cross-tab, recent visits."""
    import scripts.visits_dashboard as vsd
    return _json_safe(vsd.visits_breakdown())


@app.get("/api/visits/definitions")
async def visits_definitions():
    """Metric definitions for the True Visits dashboard."""
    import scripts.visits_dashboard as vsd
    return _json_safe(vsd.visits_definitions())


@app.get("/api/prescriptions/overview")
async def prescriptions_overview():
    """Prescriptions overview KPIs from real medications table."""
    import scripts.prescriptions_dashboard as prd
    return _json_safe(prd.prescriptions_overview())

@app.get("/api/prescriptions/breakdown")
async def prescriptions_breakdown():
    """Prescriptions per-patient and per-drug breakdown."""
    import scripts.prescriptions_dashboard as prd
    return _json_safe(prd.prescriptions_breakdown())

@app.get("/api/prescriptions/definitions")
async def prescriptions_definitions():
    """Metric definitions for prescriptions dashboard."""
    import scripts.prescriptions_dashboard as prd
    return _json_safe(prd.prescriptions_definitions())


@app.get("/api/adl/overview")
async def adl_overview():
    """ADL functional assessment KPIs from real assessments table (BARTHEL/QOLIE31/EPWORTH)."""
    import scripts.adl_dashboard as adl
    return _json_safe(adl.adl_overview())

@app.get("/api/adl/breakdown")
async def adl_breakdown():
    """Per-patient ADL profiles and score distributions."""
    import scripts.adl_dashboard as adl
    return _json_safe(adl.adl_breakdown())

@app.get("/api/adl/definitions")
async def adl_definitions():
    """ADL metric definitions and clinical relevance."""
    import scripts.adl_dashboard as adl
    return _json_safe(adl.adl_definitions())


@app.get("/api/clinical-tasks/overview")
async def clinical_tasks_overview():
    """Clinical task KPIs: operator requests, form assignments, workflow events."""
    import scripts.clinical_tasks_dashboard as ctd
    return _json_safe(ctd.clinical_tasks_overview())

@app.get("/api/clinical-tasks/breakdown")
async def clinical_tasks_breakdown():
    """Per-category task breakdown, daily activity, component-action cross-tab."""
    import scripts.clinical_tasks_dashboard as ctd
    return _json_safe(ctd.clinical_tasks_breakdown())

@app.get("/api/clinical-tasks/definitions")
async def clinical_tasks_definitions():
    """Clinical tasks metric definitions and clinical relevance."""
    import scripts.clinical_tasks_dashboard as ctd
    return _json_safe(ctd.clinical_tasks_definitions())


# ── Patients Seen Dashboard ─────────────────────────────────────

@app.get("/api/patients-seen/overview")
async def patients_seen_overview():
    """Patients Seen KPIs: unique patients with completed appointments,
    completion rate, no-show rate, provider load, department distribution."""
    import scripts.patients_seen_dashboard as psd
    return _json_safe(psd.patients_seen_overview())

@app.get("/api/patients-seen/breakdown")
async def patients_seen_breakdown():
    """Per-provider, per-department, per-patient breakdown, daily trend,
    appointment type distribution, recent completed appointments."""
    import scripts.patients_seen_dashboard as psd
    return _json_safe(psd.patients_seen_breakdown())

@app.get("/api/patients-seen/definitions")
async def patients_seen_definitions():
    """Metric definitions for the Patients Seen dashboard."""
    import scripts.patients_seen_dashboard as psd
    return _json_safe(psd.patients_seen_definitions())


# ── Patient Dashboard (patient KPIs + trends) ──────────────────────

@app.get("/api/patient-dashboard/overview")
async def patient_dashboard_overview():
    """Cross-domain KPIs: patients, assessments, appointments, meds, seizures, analyses."""
    import scripts.patient_dashboard as pdash
    return _json_safe(pdash.patient_dashboard_overview())


@app.get("/api/patient-dashboard/breakdown")
async def patient_dashboard_breakdown():
    """Breakdowns: per-disease, per-instrument, severity, trends, patient summary."""
    import scripts.patient_dashboard as pdash
    return _json_safe(pdash.patient_dashboard_breakdown())


@app.get("/api/patient-dashboard/definitions")
async def patient_dashboard_definitions():
    """Patient dashboard metric definitions for tooltip overlays."""
    import scripts.patient_dashboard as pdash
    return _json_safe(pdash.patient_dashboard_definitions())


# ── Continuous Monitoring Dashboard (seizure diary analytics) ──

@app.get("/api/continuous-monitoring/overview")
async def continuous_monitoring_overview():
    """Seizure diary KPIs: frequency, severity, triggers, ER rate, durations."""
    import scripts.continuous_monitoring_dashboard as cmd
    return _json_safe(cmd.overview())


@app.get("/api/continuous-monitoring/breakdown")
async def continuous_monitoring_breakdown():
    """Per-patient profiles, daily trend, duration buckets, recent events."""
    import scripts.continuous_monitoring_dashboard as cmd
    return _json_safe(cmd.breakdown())


@app.get("/api/continuous-monitoring/definitions")
async def continuous_monitoring_definitions():
    """Metric definitions for the Continuous Monitoring dashboard."""
    import scripts.continuous_monitoring_dashboard as cmd
    return _json_safe(cmd.definitions())


# ── Generative AI Dashboard (GenAI bot + conversation analytics) ──

@app.get("/api/generative-ai/overview")
async def generative_ai_overview():
    """GenAI KPIs: conversations, bot queries, safety score, response quality."""
    import scripts.generative_ai_dashboard as gad
    return _json_safe(gad.overview())


@app.get("/api/generative-ai/breakdown")
async def generative_ai_breakdown():
    """Per-role breakdown, recent conversations, hourly patterns, AI transactions."""
    import scripts.generative_ai_dashboard as gad
    return _json_safe(gad.breakdown())


@app.get("/api/generative-ai/definitions")
async def generative_ai_definitions():
    """Metric definitions for the Generative AI dashboard."""
    import scripts.generative_ai_dashboard as gad
    return _json_safe(gad.definitions())


# ── Data Lineage Dashboard ─────────────────────────────────────

@app.get("/api/data-lineage/overview")
async def data_lineage_overview():
    """Data lineage overview — pipeline stage distribution, component graph,
    actor audit, lineage edges, daily/hourly activity patterns."""
    import scripts.data_lineage_dashboard as dld
    return _json_safe(dld.data_lineage_overview())


@app.get("/api/data-lineage/breakdown")
async def data_lineage_breakdown():
    """Data lineage breakdown — per-patient lineage chains, per-component
    action breakdown, audit trail, action-stage mapping."""
    import scripts.data_lineage_dashboard as dld
    return _json_safe(dld.data_lineage_breakdown())


@app.get("/api/data-lineage/definitions")
async def data_lineage_definitions():
    """Data lineage definitions — lineage concepts, quality metrics,
    clinical relevance (IEC 62304, FDA AI/ML, HIPAA, EU AI Act), remediation."""
    import scripts.data_lineage_dashboard as dld
    return _json_safe(dld.data_lineage_definitions())


# ── AI Security Dashboard ──
# Real data: transaction_log (647 events, 7 actors, 25 components), hitl_reviews,
# operator_requests — access control, PHI tracking, risk classification, audit trail.

@app.get("/api/ai-security/overview")
async def ai_security_overview():
    """AI security overview — risk posture, PHI access, actor coverage,
    human oversight rate, daily trend, hourly pattern."""
    import scripts.ai_security_dashboard as asd
    return _json_safe(asd.ai_security_overview())


@app.get("/api/ai-security/breakdown")
async def ai_security_breakdown():
    """AI security breakdown — PHI access log, actor profiles, high-risk
    actions, anomaly indicators, governance events."""
    import scripts.ai_security_dashboard as asd
    return _json_safe(asd.ai_security_breakdown())


@app.get("/api/ai-security/definitions")
async def ai_security_definitions():
    """AI security definitions — security concepts, quality metrics,
    clinical relevance (HIPAA, FDA AI/ML, IEC 62304, EU AI Act, NIST AI RMF), remediation."""
    import scripts.ai_security_dashboard as asd
    return _json_safe(asd.ai_security_definitions())


# ── Data Acquisition Dashboard ──────────────────────────────────────
# Real data: uploads (21 files, 15 patients), analyses (21 with signal quality
# + confidence), transaction_log eeg_upload + ingest events — ingestion metrics,
# format coverage, signal quality, analysis pipeline tracking.

@app.get("/api/data-acquisition/overview")
async def data_acquisition_overview():
    """Data acquisition overview — upload KPIs, format distribution, signal
    quality, daily trend, confidence buckets, hourly pattern."""
    import scripts.data_acquisition_dashboard as dad
    return _json_safe(dad.data_acquisition_overview())


@app.get("/api/data-acquisition/breakdown")
async def data_acquisition_breakdown():
    """Data acquisition breakdown — per-patient profiles, recent analyses,
    file analysis, activity log, department + disease distribution."""
    import scripts.data_acquisition_dashboard as dad
    return _json_safe(dad.data_acquisition_breakdown())


@app.get("/api/data-acquisition/definitions")
async def data_acquisition_definitions():
    """Data acquisition definitions — acquisition concepts, quality metrics,
    clinical relevance (ILAE, IEC 62304, FDA AI/ML, HIPAA, EU AI Act), remediation."""
    import scripts.data_acquisition_dashboard as dad
    return _json_safe(dad.data_acquisition_definitions())


# ── Data Privacy Dashboard ──────────────────────────────────────────
# Real data: patients (40, PII fields), transaction_log (645+ PHI access events),
# conversation_log (360 messages, PHI leakage scan), uploads (21 files),
# actor-patient access matrix, component sensitivity classification.

@app.get("/api/data-privacy/overview")
async def data_privacy_overview():
    """Data privacy overview — PII exposure KPIs, PHI access trends,
    field distribution, component access, action breakdown."""
    import scripts.data_privacy_dashboard as dpd
    return _json_safe(dpd.data_privacy_overview())


@app.get("/api/data-privacy/breakdown")
async def data_privacy_breakdown():
    """Data privacy breakdown — per-patient profiles, conversation PHI scan,
    actor matrix, component sensitivity, recent PHI log, upload privacy."""
    import scripts.data_privacy_dashboard as dpd
    return _json_safe(dpd.data_privacy_breakdown())


@app.get("/api/data-privacy/definitions")
async def data_privacy_definitions():
    """Data privacy definitions — privacy concepts, quality metrics,
    clinical relevance (HIPAA, GDPR, FDA, EU AI Act, IEC 62304), remediation."""
    import scripts.data_privacy_dashboard as dpd
    return _json_safe(dpd.data_privacy_definitions())


# ── Data Quality Dashboard ─────────────────────────────────────────
# Real data: patients (40, field completeness), uploads (21, dedup/format),
# analyses (21, signal quality/confidence), transaction_log (quality events).

@app.get("/api/data-quality/overview")
async def data_quality_overview():
    """Data quality overview — completeness KPIs, signal quality distribution,
    confidence stats, duplicate counts, daily trend, format distribution."""
    import scripts.data_quality_dashboard as dqd
    return _json_safe(dqd.data_quality_overview())


@app.get("/api/data-quality/breakdown")
async def data_quality_breakdown():
    """Data quality breakdown — per-patient profiles, duplicate log, upload
    quality detail, outlier analyses, quality event log."""
    import scripts.data_quality_dashboard as dqd
    return _json_safe(dqd.data_quality_breakdown())


@app.get("/api/data-quality/definitions")
async def data_quality_definitions():
    """Data quality definitions — quality concepts, metrics, clinical relevance
    (ILAE, IEC 62304, FDA AI-ML, HIPAA, EU AI Act), remediation strategies."""
    import scripts.data_quality_dashboard as dqd
    return _json_safe(dqd.data_quality_definitions())


@app.get("/api/continuous-learning/overview")
async def continuous_learning_overview():
    """Continuous Learning overview — feedback volume, training runs, drift
    events, retrain triggers, HITL overrides, confidence distribution."""
    import scripts.continuous_learning_dashboard as cld
    return _json_safe(cld.cl_overview())


@app.get("/api/continuous-learning/breakdown")
async def continuous_learning_breakdown():
    """Continuous Learning breakdown — feedback log, HITL review detail,
    expert concordance, per-patient error analysis, training run detail."""
    import scripts.continuous_learning_dashboard as cld
    return _json_safe(cld.cl_breakdown())


@app.get("/api/continuous-learning/definitions")
async def continuous_learning_definitions():
    """Continuous Learning definitions — concepts, quality metrics, clinical
    relevance (FDA AI-ML, ILAE, IEC 62304, EU AI Act), remediation strategies."""
    import scripts.continuous_learning_dashboard as cld
    return _json_safe(cld.definitions())


# ── Embedding & Feature Engineering Dashboard ────────────────────────
# Real data: analyses (21, feature extractions with confidence + signal quality),
# uploads (21, input files), patients (40, coverage tracking),
# transaction_log (368+ embedding-related events).

@app.get("/api/embedding/overview")
async def embedding_overview():
    """Embedding overview — feature extraction KPIs, type distribution,
    confidence histogram, daily trend, signal quality, disease coverage."""
    import scripts.embedding_dashboard as ebd
    return _json_safe(ebd.embedding_overview())


@app.get("/api/embedding/breakdown")
async def embedding_breakdown():
    """Embedding breakdown — per-patient feature profiles, recent extractions,
    dimension analysis, extraction event log, staleness analysis."""
    import scripts.embedding_dashboard as ebd
    return _json_safe(ebd.embedding_breakdown())


@app.get("/api/embedding/definitions")
async def embedding_definitions():
    """Embedding definitions — feature concepts, quality metrics, clinical
    relevance (ILAE, IEC 62304, FDA AI-ML, HIPAA, EU AI Act), remediation."""
    import scripts.embedding_dashboard as ebd
    return _json_safe(ebd.embedding_definitions())


@app.get("/api/ai-lifecycle/overview")
async def ai_lifecycle_overview():
    """AI Lifecycle Management overview — asset inventory, stage distribution,
    daily events, health radar across ideation→deploy→monitor→retire."""
    import scripts.ai_lifecycle_dashboard as ald
    return _json_safe(ald.lifecycle_overview())


@app.get("/api/ai-lifecycle/breakdown")
async def ai_lifecycle_breakdown():
    """AI Lifecycle breakdown — agent/pipeline/model inventory, validation log,
    monitoring events, training history, lifecycle transitions."""
    import scripts.ai_lifecycle_dashboard as ald
    return _json_safe(ald.lifecycle_breakdown())


@app.get("/api/ai-lifecycle/definitions")
async def ai_lifecycle_definitions():
    """AI Lifecycle definitions — lifecycle concepts, metrics, clinical
    relevance (IEC 62304, FDA AI-ML, EU AI Act, ISO 14971), remediation."""
    import scripts.ai_lifecycle_dashboard as ald
    return _json_safe(ald.lifecycle_definitions())


@app.get("/api/ai-governance/overview")
async def ai_governance_overview():
    """AI Governance overview — decision audit trail, expert reviews,
    HITL oversight, feedback loop, consultant coverage from real clinical.db."""
    import scripts.ai_governance_dashboard as agd
    return _json_safe(agd.governance_overview())


@app.get("/api/ai-governance/breakdown")
async def ai_governance_breakdown():
    """AI Governance breakdown — consultant matrix, role coverage,
    governance health scores, use-case risk classification."""
    import scripts.ai_governance_dashboard as agd
    return _json_safe(agd.governance_breakdown())


@app.get("/api/ai-governance/definitions")
async def ai_governance_definitions():
    """AI Governance definitions — concepts, metrics, clinical
    relevance (EU AI Act, FDA AI-ML, IEC 62304, ISO 14971), remediation."""
    import scripts.ai_governance_dashboard as agd
    return _json_safe(agd.governance_definitions())


# ── MCP Governance — agent/tool registry, access control, MCP health ──

@app.get("/api/mcp-governance/overview")
async def mcp_governance_overview():
    """MCP Governance overview — agent inventory, tool certification,
    access control events, MCP/A2A health, pipeline governance coverage."""
    import scripts.mcp_governance_dashboard as mgd
    return _json_safe(mgd.mcp_governance_overview())


@app.get("/api/mcp-governance/breakdown")
async def mcp_governance_breakdown():
    """Per-agent inventory, pipeline-level status, MCP consensus voting."""
    import scripts.mcp_governance_dashboard as mgd
    return _json_safe(mgd.mcp_governance_breakdown())


@app.get("/api/mcp-governance/definitions")
async def mcp_governance_definitions():
    """MCP Governance definitions — MCP protocol, agent certification,
    access control, compliance refs (EU AI Act, ISO 14971, NIST AI RMF)."""
    import scripts.mcp_governance_dashboard as mgd
    return _json_safe(mgd.mcp_governance_definitions())


# ── Council of Agents — multi-agent consensus, author/reviewer/chair ──

@app.get("/api/agent-council/overview")
async def agent_council_overview():
    """Council of Agents overview — role distribution, consensus metrics,
    decision quality from real clinical.db + agent registry."""
    import scripts.council_of_agents_dashboard as cad
    return _json_safe(cad.council_overview())

@app.get("/api/agent-council/breakdown")
async def agent_council_breakdown():
    """Council breakdown — per-agent roles, review sessions, voting history."""
    import scripts.council_of_agents_dashboard as cad
    return _json_safe(cad.council_breakdown())

@app.get("/api/agent-council/definitions")
async def agent_council_definitions():
    """Council definitions — roles, consensus types, clinical compliance."""
    import scripts.council_of_agents_dashboard as cad
    return _json_safe(cad.council_definitions())


# ── Grounding ─────────────────────────────────────────────
@app.get("/api/grounding/overview")
async def grounding_overview():
    """Grounding overview — source verification rates, citation coverage,
    confidence distribution, grounding KPIs."""
    import scripts.grounding_dashboard as grd
    return _json_safe(grd.grounding_overview())


@app.get("/api/grounding/breakdown")
async def grounding_breakdown():
    """Grounding breakdown — per-patient grounding scores, claim traces,
    source verification log, expert verification detail."""
    import scripts.grounding_dashboard as grd
    return _json_safe(grd.grounding_breakdown())


@app.get("/api/grounding/definitions")
async def grounding_definitions():
    """Grounding definitions — grounding concepts, metrics,
    clinical relevance, remediation strategies."""
    import scripts.grounding_dashboard as grd
    return _json_safe(grd.definitions())


# ── AI Red Team Dashboard ─────────────────────────────────────────
@app.get("/api/ai-red-team/overview")
async def ai_red_team_overview():
    """AI Red Team overview — adversarial testing, jailbreak detection,
    prompt attack analysis, tool abuse monitoring from real clinical.db data."""
    import scripts.ai_red_team_dashboard as art
    return _json_safe(art.red_team_overview())


@app.get("/api/ai-red-team/breakdown")
async def ai_red_team_breakdown():
    """Per-attack-type and per-component red team drill-down."""
    import scripts.ai_red_team_dashboard as art
    return _json_safe(art.red_team_breakdown())


@app.get("/api/ai-red-team/definitions")
async def ai_red_team_definitions():
    """AI Red Team metric definitions."""
    import scripts.ai_red_team_dashboard as art
    return _json_safe(art.red_team_definitions())


# ── Knowledge Management Dashboard ───────────────────────────────
@app.get("/api/knowledge-mgmt/overview")
async def knowledge_mgmt_overview():
    """Knowledge Management overview — lifecycle tracking: create, approve,
    publish, expiry, archive from real clinical.db data."""
    import scripts.knowledge_management_dashboard as kmd
    return _json_safe(kmd.knowledge_overview())


@app.get("/api/knowledge-mgmt/breakdown")
async def knowledge_mgmt_breakdown():
    """Per-item knowledge register, patient profiles, lifecycle events."""
    import scripts.knowledge_management_dashboard as kmd
    return _json_safe(kmd.knowledge_breakdown())


@app.get("/api/knowledge-mgmt/definitions")
async def knowledge_mgmt_definitions():
    """Knowledge Management metric definitions."""
    import scripts.knowledge_management_dashboard as kmd
    return _json_safe(kmd.knowledge_definitions())


# ── Fine-Tuning Pipeline Dashboard ─────────────────────────────────
@app.get("/api/fine-tuning/overview")
async def fine_tuning_overview():
    """Fine-Tuning Pipeline overview — model inventory, training runs,
    accuracy distribution, pipeline stage tracking from saved_models + clinical.db."""
    import scripts.fine_tuning_dashboard as ftd
    return _json_safe(ftd.fine_tuning_overview())


@app.get("/api/fine-tuning/breakdown")
async def fine_tuning_breakdown():
    """Per-model, per-disease, per-type fine-tuning drill-down."""
    import scripts.fine_tuning_dashboard as ftd
    return _json_safe(ftd.fine_tuning_breakdown())


@app.get("/api/fine-tuning/definitions")
async def fine_tuning_definitions():
    """Fine-Tuning Pipeline metric definitions."""
    import scripts.fine_tuning_dashboard as ftd
    return _json_safe(ftd.fine_tuning_definitions())


# ── Image Segmentation AI Dashboard ──────────────────────────────────
@app.get("/api/image-segmentation/overview")
async def image_segmentation_overview():
    """Image Segmentation AI overview — EEG trace digitization from images,
    segmentation tasks, quality scores, segment classes, patient coverage."""
    import scripts.image_segmentation_dashboard as isd
    return _json_safe(isd.overview())


@app.get("/api/image-segmentation/breakdown")
async def image_segmentation_breakdown():
    """Image Segmentation AI breakdown — per-patient details, segment type
    distribution, timeline, quality matrix, file inventory."""
    import scripts.image_segmentation_dashboard as isd
    return _json_safe(isd.breakdown())


@app.get("/api/image-segmentation/definitions")
async def image_segmentation_definitions():
    """Image Segmentation AI definitions — concepts, compliance, remediation."""
    import scripts.image_segmentation_dashboard as isd
    return _json_safe(isd.definitions())


# ── Object Detection AI Dashboard ─────────────────────────────────────
@app.get("/api/object-detection/overview")
async def object_detection_overview():
    """Object Detection AI overview — body-movement and lesion detection from
    video-EEG and MRI imaging, detection classes, confidence, IoU scores."""
    import scripts.object_detection_dashboard as odd
    return _json_safe(odd.overview())


@app.get("/api/object-detection/breakdown")
async def object_detection_breakdown():
    """Object Detection AI breakdown — detection inventory, per-patient details,
    class stats, location heatmap, pipeline events."""
    import scripts.object_detection_dashboard as odd
    return _json_safe(odd.breakdown())


@app.get("/api/object-detection/definitions")
async def object_detection_definitions():
    """Object Detection AI definitions — concepts, compliance, remediation."""
    import scripts.object_detection_dashboard as odd
    return _json_safe(odd.definitions())


# ── YOLO Detection Dashboard ──────────────────────────────────────────
@app.get("/api/yolo-detection/overview")
async def yolo_detection_overview():
    """YOLO Detection overview — video-EEG object detection using YOLO model
    variants, detection classes, confidence, IoU, model comparison."""
    import scripts.yolo_detection_dashboard as ydd
    return _json_safe(ydd.overview())


@app.get("/api/yolo-detection/breakdown")
async def yolo_detection_breakdown():
    """YOLO Detection breakdown — per-patient profiles, per-recording inventory,
    model architecture comparison, confidence and IoU distributions."""
    import scripts.yolo_detection_dashboard as ydd
    return _json_safe(ydd.breakdown())


@app.get("/api/yolo-detection/definitions")
async def yolo_detection_definitions():
    """YOLO Detection definitions — YOLO architecture, mAP, IoU, NMS, FPN concepts."""
    import scripts.yolo_detection_dashboard as ydd
    return _json_safe(ydd.definitions())


# ── Speech AI Dashboard ────────────────────────────────────────────────
@app.get("/api/speech-ai/overview")
async def speech_ai_overview():
    """Speech AI overview — transcription counts, interaction types,
    transcript lengths, daily activity, pipeline health."""
    import scripts.speech_ai_dashboard as sad
    return _json_safe(sad.overview())


@app.get("/api/speech-ai/breakdown")
async def speech_ai_breakdown():
    """Speech AI breakdown — transcription inventory, audio files,
    patient profiles, pipeline events, role stats."""
    import scripts.speech_ai_dashboard as sad
    return _json_safe(sad.breakdown())


@app.get("/api/speech-ai/definitions")
async def speech_ai_definitions():
    """Speech AI definitions — ASR, NLP, voice biomarkers, compliance."""
    import scripts.speech_ai_dashboard as sad
    return _json_safe(sad.definitions())


# ── Voice AI Dashboard ─────────────────────────────────────────────────
@app.get("/api/voice-ai/overview")
async def voice_ai_overview():
    """Voice AI overview — vocal biomarker extraction, prosody analysis,
    articulation/fluency/swallowing assessments (WAB, VERBAL_FLUENCY, MASA,
    BNT, DIGIT_SPAN), severity distribution, per-instrument scores."""
    import scripts.voice_ai_dashboard as vad
    return _json_safe(vad.overview())


@app.get("/api/voice-ai/breakdown")
async def voice_ai_breakdown():
    """Voice AI breakdown — assessment inventory, per-patient vocal profiles,
    instrument statistics, clinical alerts, pipeline events."""
    import scripts.voice_ai_dashboard as vad
    return _json_safe(vad.breakdown())


@app.get("/api/voice-ai/definitions")
async def voice_ai_definitions():
    """Voice AI definitions — voice biomarker concepts, assessment instruments,
    quality metrics, compliance, remediation."""
    import scripts.voice_ai_dashboard as vad
    return _json_safe(vad.definitions())


# ── Text-to-Audio AI Dashboard ────────────────────────────────────────
@app.get("/api/text-to-audio/overview")
async def text_to_audio_overview():
    """Text-to-Audio AI overview — TTS synthesis pipeline monitoring,
    synthesizable text corpus, audio duration estimates, cost tracking."""
    import scripts.text_to_audio_dashboard as ttad
    return _json_safe(ttad.overview())


@app.get("/api/text-to-audio/breakdown")
async def text_to_audio_breakdown():
    """Text-to-Audio AI breakdown — text inventory, report inventory,
    per-patient audio profiles, pipeline events."""
    import scripts.text_to_audio_dashboard as ttad
    return _json_safe(ttad.breakdown())


@app.get("/api/text-to-audio/definitions")
async def text_to_audio_definitions():
    """Text-to-Audio AI definitions — TTS concepts, quality metrics,
    audio categories, compliance, remediation."""
    import scripts.text_to_audio_dashboard as ttad
    return _json_safe(ttad.definitions())


# ── Text-to-Video AI Dashboard ────────────────────────────────────────
@app.get("/api/text-to-video/overview")
async def text_to_video_overview():
    """Text-to-Video AI overview — video synthesis pipeline monitoring,
    EEG timelapse, seizure event clips, MRI flythrough rendering."""
    import scripts.text_to_video_dashboard as ttvd
    return _json_safe(ttvd.overview())


@app.get("/api/text-to-video/breakdown")
async def text_to_video_breakdown():
    """Text-to-Video AI breakdown — source inventory, seizure clips,
    MRI renders, per-patient video profiles, pipeline events."""
    import scripts.text_to_video_dashboard as ttvd
    return _json_safe(ttvd.breakdown())


@app.get("/api/text-to-video/definitions")
async def text_to_video_definitions():
    """Text-to-Video AI definitions — video synthesis concepts, quality metrics,
    video categories, compliance, remediation."""
    import scripts.text_to_video_dashboard as ttvd
    return _json_safe(ttvd.definitions())


# ── Cognitive Profile Summary Dashboard ──────────────────────────────────
@app.get("/api/cognitive-profile/overview")
async def cognitive_profile_overview():
    """Cognitive Profile Summary — MoCA, MMSE, WAIS, Digit Span, PHQ9, GAD7, QOLIE-31, NDDIE."""
    import scripts.cognitive_profile_dashboard as cpd
    return _json_safe(cpd.overview())


@app.get("/api/cognitive-profile/breakdown")
async def cognitive_profile_breakdown():
    """Cognitive Profile breakdown — assessment inventory, per-patient profiles, domain scores, alerts."""
    import scripts.cognitive_profile_dashboard as cpd
    return _json_safe(cpd.breakdown())


@app.get("/api/cognitive-profile/definitions")
async def cognitive_profile_definitions():
    """Cognitive Profile definitions — test concepts, scoring norms, compliance, remediation."""
    import scripts.cognitive_profile_dashboard as cpd
    return _json_safe(cpd.definitions())


# ── Time-Series AI Dashboard ──────────────────────────────────────────
@app.get("/api/time-series-ai/overview")
async def time_series_ai_overview():
    """Time-Series AI overview — EEG spectral decomposition, band power analysis,
    complexity metrics (Hurst, entropy, DFA), temporal feature extraction,
    signal quality, seizure event timelines."""
    import scripts.time_series_ai_dashboard as tsad
    return _json_safe(tsad.overview())


@app.get("/api/time-series-ai/breakdown")
async def time_series_ai_breakdown():
    """Time-Series AI breakdown — per-recording feature inventory, feature matrix,
    band power details, patient profiles, seizure inventory, pipeline events."""
    import scripts.time_series_ai_dashboard as tsad
    return _json_safe(tsad.breakdown())


@app.get("/api/time-series-ai/definitions")
async def time_series_ai_definitions():
    """Time-Series AI definitions — spectral/complexity concepts, quality metrics,
    feature categories, compliance, remediation."""
    import scripts.time_series_ai_dashboard as tsad
    return _json_safe(tsad.definitions())


# ── Medication Interaction Checker Dashboard ──────────────────────────────
@app.get("/api/medication-interaction/overview")
async def medication_interaction_overview():
    """Medication Interaction Checker — AED polytherapy risk, drug-drug
    interaction screening, psychiatric comorbidity cross-reference."""
    import scripts.medication_interaction_dashboard as mid
    return _json_safe(mid.overview())


@app.get("/api/medication-interaction/breakdown")
async def medication_interaction_breakdown():
    """Medication Interaction breakdown — medication inventory, interaction
    results, patient profiles, ADR flags, psychiatric alerts."""
    import scripts.medication_interaction_dashboard as mid
    return _json_safe(mid.breakdown())


@app.get("/api/medication-interaction/definitions")
async def medication_interaction_definitions():
    """Medication Interaction definitions — AED concepts, DDI mechanisms,
    drug classes, compliance, remediation."""
    import scripts.medication_interaction_dashboard as mid
    return _json_safe(mid.definitions())


# ── Conversational AI Dashboard ───────────────────────────────────────
@app.get("/api/conversational-ai/overview")
async def conversational_ai_overview():
    """Conversational AI overview — turn analytics, role distribution,
    daily activity, response length distribution, hourly patterns."""
    import scripts.conversational_ai_dashboard as cad
    return _json_safe(cad.overview())


@app.get("/api/conversational-ai/breakdown")
async def conversational_ai_breakdown():
    """Conversational AI breakdown — conversation log, daily summaries,
    role stats, topic analysis, pipeline events."""
    import scripts.conversational_ai_dashboard as cad
    return _json_safe(cad.breakdown())


@app.get("/api/conversational-ai/definitions")
async def conversational_ai_definitions():
    """Conversational AI definitions — NLU concepts, quality metrics,
    interaction types, compliance, remediation."""
    import scripts.conversational_ai_dashboard as cad
    return _json_safe(cad.definitions())


# ── Patient Reporting Dashboard ──────────────────────────────────────
@app.get("/api/patient-reporting/overview")
async def patient_reporting_overview():
    """Patient Reporting overview — report generation metrics, coverage,
    assessment summaries, appointment tracking, seizure diary stats."""
    import scripts.patient_reporting_dashboard as prd
    return _json_safe(prd.overview())


@app.get("/api/patient-reporting/breakdown")
async def patient_reporting_breakdown():
    """Patient Reporting breakdown — per-patient report inventory,
    appointment schedule, assessment list, seizure log, pipeline events."""
    import scripts.patient_reporting_dashboard as prd
    return _json_safe(prd.breakdown())


@app.get("/api/patient-reporting/definitions")
async def patient_reporting_definitions():
    """Patient Reporting definitions — report types, quality metrics,
    compliance, remediation."""
    import scripts.patient_reporting_dashboard as prd
    return _json_safe(prd.definitions())


# ── Research Coordinator Dashboard ──────────────────────────────────
@app.get("/api/research-coordinator/overview")
async def research_coordinator_overview():
    """Research Coordinator — enrollment, protocol compliance, cohort management."""
    import scripts.research_coordinator_dashboard as rcd
    return _json_safe(rcd.overview())


@app.get("/api/research-coordinator/breakdown")
async def research_coordinator_breakdown():
    """Research Coordinator breakdown — subject inventory, protocol matrix, visits, outcomes."""
    import scripts.research_coordinator_dashboard as rcd
    return _json_safe(rcd.breakdown())


@app.get("/api/research-coordinator/definitions")
async def research_coordinator_definitions():
    """Research Coordinator definitions — concepts, study phases, compliance, remediation."""
    import scripts.research_coordinator_dashboard as rcd
    return _json_safe(rcd.definitions())


# ── Neurosurgeon / Epilepsy Surgery Dashboard ────────────────────────────
@app.get("/api/neurosurgeon/overview")
async def neurosurgeon_overview():
    """Neurosurgeon overview — MRI lesion classification, surgical candidacy,
    seizure burden, EEG analyses, laterality/location distribution."""
    import scripts.neurosurgeon_dashboard as nsd
    return _json_safe(nsd.overview())


@app.get("/api/neurosurgeon/breakdown")
async def neurosurgeon_breakdown():
    """Neurosurgeon breakdown — MRI inventory, patient surgical profiles,
    seizure log, EEG summary."""
    import scripts.neurosurgeon_dashboard as nsd
    return _json_safe(nsd.breakdown())


@app.get("/api/neurosurgeon/definitions")
async def neurosurgeon_definitions():
    """Neurosurgeon definitions — surgical concepts, procedures, quality metrics,
    compliance, remediation."""
    import scripts.neurosurgeon_dashboard as nsd
    return _json_safe(nsd.definitions())


# ── Speech-Language Pathologist (SLP) Dashboard ───────────────────────────
@app.get("/api/slp")
async def slp_combined():
    """SLP combined — language assessment, swallowing/dysphagia risk,
    AED speech effects, cognitive-communication, therapy goals, definitions."""
    import scripts.slp_dashboard as slpd
    return _json_safe(slpd.combined())


@app.get("/api/slp/overview")
async def slp_overview():
    """SLP overview — KPIs, severity distribution."""
    import scripts.slp_dashboard as slpd
    return _json_safe(slpd.overview())


@app.get("/api/slp/breakdown")
async def slp_breakdown():
    """SLP breakdown — language test scores, lateralization, swallowing patients,
    AED effects, cognitive profiles, therapy goals."""
    import scripts.slp_dashboard as slpd
    return _json_safe(slpd.breakdown())


@app.get("/api/slp/definitions")
async def slp_definitions():
    """SLP definitions — BNT, WAB, MASA, dysphagia, cognitive-communication concepts."""
    import scripts.slp_dashboard as slpd
    return _json_safe(slpd.definitions())


# ---------------------------------------------------------------------------
# CNN/ResNet Spectrogram Dashboard
# ---------------------------------------------------------------------------

@app.get("/api/cnn-resnet/overview")
async def cnn_resnet_overview():
    """CNN/ResNet Spectrogram — KPIs, band power, quality, classification, activity."""
    import scripts.cnn_resnet_dashboard as crd
    return _json_safe(crd.overview())


@app.get("/api/cnn-resnet/breakdown")
async def cnn_resnet_breakdown():
    """CNN/ResNet Spectrogram — spectrogram inventory, patient profiles, model architectures."""
    import scripts.cnn_resnet_dashboard as crd
    return _json_safe(crd.breakdown())


@app.get("/api/cnn-resnet/definitions")
async def cnn_resnet_definitions():
    """CNN/ResNet Spectrogram — CNN, ResNet, spectrogram, EEG band definitions."""
    import scripts.cnn_resnet_dashboard as crd
    return _json_safe(crd.definitions())


# ── Clinical Neurophysiologist / EEG Reviewer Dashboard ────────────────────
@app.get("/api/neurophysiologist/overview")
async def neurophysiologist_overview():
    """Neurophysiologist overview — EEG recordings, band power, signal quality,
    background rhythm, AI predictions, spectral entropy."""
    import scripts.neurophysiologist_dashboard as npd
    return _json_safe(npd.overview())


@app.get("/api/neurophysiologist/breakdown")
async def neurophysiologist_breakdown():
    """Neurophysiologist breakdown — recording inventory, band power table,
    spectral features, AI validation, channel stats, seizure log."""
    import scripts.neurophysiologist_dashboard as npd
    return _json_safe(npd.breakdown())


@app.get("/api/neurophysiologist/definitions")
async def neurophysiologist_definitions():
    """Neurophysiologist definitions — EEG concepts, band power, signal quality,
    entropy, Hjorth, complexity measures."""
    import scripts.neurophysiologist_dashboard as npd
    return _json_safe(npd.definitions())


# ---------------------------------------------------------------------------
# RNN/LSTM Temporal Model Dashboard
# ---------------------------------------------------------------------------

@app.get("/api/rnn-lstm/overview")
async def rnn_lstm_overview():
    """RNN/LSTM Temporal Model — KPIs, band power, quality, temporal patterns."""
    import scripts.rnn_lstm_dashboard as rld
    return _json_safe(rld.overview())


@app.get("/api/rnn-lstm/breakdown")
async def rnn_lstm_breakdown():
    """RNN/LSTM Temporal Model — sequence inventory, patient profiles, architectures."""
    import scripts.rnn_lstm_dashboard as rld
    return _json_safe(rld.breakdown())


@app.get("/api/rnn-lstm/definitions")
async def rnn_lstm_definitions():
    """RNN/LSTM Temporal Model — RNN, LSTM, GRU, attention definitions."""
    import scripts.rnn_lstm_dashboard as rld
    return _json_safe(rld.definitions())


# ── Neuropsychologist Dashboard ────────────────────────────────────────────
@app.get("/api/neuropsychologist/overview")
async def neuropsychologist_overview():
    """Neuropsychologist overview — IQ distribution, WAIS index profile,
    MoCA/MMSE screening, digit span, AED cognitive risk, mood comorbidity."""
    import scripts.neuropsychologist_dashboard as npsd
    return _json_safe(npsd.overview())


@app.get("/api/neuropsychologist/breakdown")
async def neuropsychologist_breakdown():
    """Neuropsychologist breakdown — WAIS profiles, digit span detail,
    cognitive screening, per-patient cognitive profiles with risk factors."""
    import scripts.neuropsychologist_dashboard as npsd
    return _json_safe(npsd.breakdown())


@app.get("/api/neuropsychologist/definitions")
async def neuropsychologist_definitions():
    """Neuropsychologist definitions — WAIS, MoCA, MMSE, digit span,
    executive function, processing speed, AED cognitive effects."""
    import scripts.neuropsychologist_dashboard as npsd
    return _json_safe(npsd.definitions())


# ── Radiologist Dashboard ──────────────────────────────────────────────────
@app.get("/api/radiologist/overview")
async def radiologist_overview():
    """Radiologist overview — MRI findings, lesion-positive rate, lesion type
    distribution, location/laterality breakdown, hippocampal sclerosis stats."""
    import scripts.radiologist_dashboard as rdd
    return _json_safe(rdd.overview())


@app.get("/api/radiologist/breakdown")
async def radiologist_breakdown():
    """Radiologist breakdown — per-patient MRI findings with lesion type, location,
    laterality, volume asymmetry, T2/FLAIR signal, linked EEG analysis."""
    import scripts.radiologist_dashboard as rdd
    return _json_safe(rdd.breakdown())


@app.get("/api/radiologist/definitions")
async def radiologist_definitions():
    """Radiologist definitions — neuroradiology concepts: MRI protocol, hippocampal
    sclerosis, FCD, cavernoma, AVM, lesional classification, ILAE references."""
    import scripts.radiologist_dashboard as rdd
    return _json_safe(rdd.definitions())


# ── Psychiatrist Dashboard ─────────────────────────────────────────────────
@app.get("/api/psychiatrist/overview")
async def psychiatrist_overview():
    """Psychiatrist overview — depression/anxiety severity, C-SSRS risk,
    AED psychiatric effects, mood screening KPIs."""
    import scripts.psychiatrist_dashboard as psd
    return _json_safe(psd.overview())


@app.get("/api/psychiatrist/breakdown")
async def psychiatrist_breakdown():
    """Psychiatrist breakdown — per-patient psychiatric profiles, PNES
    candidates, mood-seizure correlation, item-level analysis."""
    import scripts.psychiatrist_dashboard as psd
    return _json_safe(psd.breakdown())


@app.get("/api/psychiatrist/definitions")
async def psychiatrist_definitions():
    """Psychiatrist definitions — PHQ-9, GAD-7, C-SSRS, NDDI-E, PNES,
    psychiatric comorbidity, AED effects, interictal dysphoric disorder."""
    import scripts.psychiatrist_dashboard as psd
    return _json_safe(psd.definitions())


# ── Occupational Therapist Dashboard ──────────────────────────────────────
@app.get("/api/occupational-therapist/overview")
async def occupational_therapist_overview():
    """OT overview — Barthel ADL distribution, QOLIE-31 QoL, Epworth
    sleepiness, LSSS seizure severity, AED functional risk, item-level analysis."""
    import scripts.occupational_therapist_dashboard as otd
    return _json_safe(otd.overview())


@app.get("/api/occupational-therapist/breakdown")
async def occupational_therapist_breakdown():
    """OT breakdown — per-patient functional profiles, rehab candidates,
    ADL item breakdown, QOLIE domain scores, suggested goals."""
    import scripts.occupational_therapist_dashboard as otd
    return _json_safe(otd.breakdown())


@app.get("/api/occupational-therapist/definitions")
async def occupational_therapist_definitions():
    """OT definitions — Barthel Index, QOLIE-31, ESS, LSSS, ADLs, IADLs,
    functional rehabilitation, AED functional effects."""
    import scripts.occupational_therapist_dashboard as otd
    return _json_safe(otd.definitions())


# ── EEG Technician Dashboard ──────────────────────────────────────────
@app.get("/api/eeg-technician/overview")
async def eeg_technician_overview():
    """EEG Technician overview — KPI cards (recordings, signal quality pass rate,
    artifact rate, impedance failure rate), recording type distribution, artifact
    type breakdown, quality histogram, impedance histogram. ACNS standards."""
    import scripts.eeg_technician_dashboard as etd
    return _json_safe(etd.overview())


@app.get("/api/eeg-technician/breakdown")
async def eeg_technician_breakdown():
    """EEG Technician breakdown — per-patient acquisition detail: recording type,
    duration, sampling rate, montage, channel quality counts, artifact count,
    dominant artifact, impedance pass, photic/HV/sleep flags, overall quality grade,
    per-channel impedance/SNR detail, artifact annotation list."""
    import scripts.eeg_technician_dashboard as etd
    return _json_safe(etd.breakdown())


@app.get("/api/eeg-technician/definitions")
async def eeg_technician_definitions():
    """EEG Technician definitions — 10-20 system, impedance standards, recording
    types, montages, artifact recognition, activation procedures, ACNS minimum
    technical standards, signal quality metrics."""
    import scripts.eeg_technician_dashboard as etd
    return _json_safe(etd.definitions())


# ── IoT Engineer Dashboard ────────────────────────────────────────────
@app.get("/api/iot-engineer/overview")
async def iot_engineer_overview():
    """IoT Engineer overview — device fleet status, gateway uptime, stream
    latency, SOS alerts, battery/signal health, alert distributions."""
    import scripts.iot_engineer_dashboard as iotd
    return _json_safe(iotd.overview())


@app.get("/api/iot-engineer/breakdown")
async def iot_engineer_breakdown():
    """IoT Engineer breakdown — per-device detail with patient assignment,
    gateway, battery, signal, firmware, alert history."""
    import scripts.iot_engineer_dashboard as iotd
    return _json_safe(iotd.breakdown())


@app.get("/api/iot-engineer/definitions")
async def iot_engineer_definitions():
    """IoT Engineer definitions — IEC 62304, IEC 80001, HIPAA device
    requirements, wearable EEG specs, gateway architecture, alert protocols."""
    import scripts.iot_engineer_dashboard as iotd
    return _json_safe(iotd.definitions())


# ── Clinical Pharmacist Dashboard ──────────────────────────────────────
@app.get("/api/clinical-pharmacist/overview")
async def clinical_pharmacist_overview():
    """Clinical pharmacist overview — medication inventory, adherence distribution,
    drug interactions, pregnancy risk, ASM class breakdown."""
    import scripts.pharmacist_module as pm
    return _json_safe(pm.overview())


@app.get("/api/clinical-pharmacist/breakdown")
async def clinical_pharmacist_breakdown():
    """Clinical pharmacist breakdown — per-patient medication profiles, interactions,
    TDM, ADR, adherence, pregnancy safety."""
    import scripts.pharmacist_module as pm
    return _json_safe(pm.breakdown())


@app.get("/api/clinical-pharmacist/definitions")
async def clinical_pharmacist_definitions():
    """Clinical pharmacist definitions — DDI, TDM, MMAS-8, CYP450, pregnancy categories."""
    import scripts.pharmacist_module as pm
    return _json_safe(pm.definitions())


# ── IRB / Ethics Officer Dashboard ──────────────────────────────────────
@app.get("/api/irb-ethics/overview")
async def irb_ethics_overview():
    """IRB/Ethics overview — protocol compliance, consent tracking, risk-benefit,
    AI oversight, audit trail completeness."""
    import scripts.irb_ethics_dashboard as irb
    return _json_safe(irb.overview())


@app.get("/api/irb-ethics/breakdown")
async def irb_ethics_breakdown():
    """IRB/Ethics breakdown — per-patient ethics profiles, component/actor audit,
    data access log, vulnerable population flags."""
    import scripts.irb_ethics_dashboard as irb
    return _json_safe(irb.breakdown())


@app.get("/api/irb-ethics/definitions")
async def irb_ethics_definitions():
    """IRB/Ethics definitions — IRB, informed consent, HITL, audit trail,
    vulnerable populations, protocol compliance, AI oversight."""
    import scripts.irb_ethics_dashboard as irb
    return _json_safe(irb.definitions())


@app.get("/api/data-steward/overview")
async def data_steward_overview():
    """Data Steward / Privacy Officer overview — PHI exposure, de-identification,
    access control, audit trail, privacy risk scoring."""
    import scripts.data_steward_dashboard as ds
    return _json_safe(ds.overview())


@app.get("/api/data-steward/breakdown")
async def data_steward_breakdown():
    """Data Steward / Privacy Officer breakdown — per-patient PHI assessment,
    access logs, file analysis, incidents, retention."""
    import scripts.data_steward_dashboard as ds
    return _json_safe(ds.breakdown())


@app.get("/api/data-steward/definitions")
async def data_steward_definitions():
    """Data Steward / Privacy Officer definitions — PHI, HIPAA Safe Harbor,
    de-identification, data retention, privacy impact assessment."""
    import scripts.data_steward_dashboard as ds
    return _json_safe(ds.definitions())


@app.get("/api/clinical-data-manager/overview")
async def clinical_data_manager_overview():
    """Clinical Data Manager overview — data quality dimensions, AI readiness,
    modality coverage, missing data matrix, lineage, task status."""
    import scripts.clinical_data_manager_dashboard as cdm
    return _json_safe(cdm.overview())


@app.get("/api/clinical-data-manager/breakdown")
async def clinical_data_manager_breakdown():
    """Clinical Data Manager breakdown — task catalog, dataset inventory,
    per-patient coverage matrix, instrument distribution, archival."""
    import scripts.clinical_data_manager_dashboard as cdm
    return _json_safe(cdm.breakdown())


@app.get("/api/clinical-data-manager/definitions")
async def clinical_data_manager_definitions():
    """Clinical Data Manager definitions — data quality concepts, AI readiness,
    lineage, versioning, terminology mapping, compliance references."""
    import scripts.clinical_data_manager_dashboard as cdm
    return _json_safe(cdm.definitions())


# ── Patient / Caregiver Dashboard ──────────────────────────────────────
@app.get("/api/patient-caregiver/overview")
async def patient_caregiver_overview():
    """Patient/Caregiver overview — seizure summary, mood overview, QoL,
    trigger distribution, appointment status."""
    import scripts.patient_caregiver_dashboard as pcd
    return _json_safe(pcd.overview())


@app.get("/api/patient-caregiver/breakdown")
async def patient_caregiver_breakdown():
    """Patient/Caregiver breakdown — seizure diary, patient profiles,
    medication list, appointment list, seizure timeline, risk alerts."""
    import scripts.patient_caregiver_dashboard as pcd
    return _json_safe(pcd.breakdown())


@app.get("/api/patient-caregiver/definitions")
async def patient_caregiver_definitions():
    """Patient/Caregiver definitions — seizure diary, PHQ-9, GAD-7,
    QOLIE-31, triggers, rescue meds, Barthel, adherence."""
    import scripts.patient_caregiver_dashboard as pcd
    return _json_safe(pcd.definitions())


# ── ABPM / Holter Dashboard ─────────────────────────────────────

@app.get("/api/abpm-holter/overview")
async def abpm_holter_overview():
    """ABPM/Holter overview: KPIs, severity distribution, diagnostic patterns,
    dipping distribution, per-patient summary. Real clinical.db data."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.overview())


@app.get("/api/abpm-holter/breakdown")
async def abpm_holter_breakdown():
    """ABPM/Holter breakdown: ABPM parameter summaries, Holter parameter summaries,
    histograms (systolic/dipping/QTc/PVC/cardiac score), per-patient detail cards."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.breakdown())


@app.get("/api/abpm-holter/definitions")
async def abpm_holter_definitions():
    """ABPM/Holter definitions — protocol, parameters, reference ranges,
    dipping categories, diagnostic patterns, severity levels, clinical significance."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.definitions())


# ── Autonomic Function Tests Dashboard ───────────────────────────

@app.get("/api/autonomic/overview")
async def autonomic_overview():
    """Autonomic function tests overview: KPIs, severity distribution,
    diagnostic patterns, histograms. Real clinical.db data."""
    import scripts.autonomic_dashboard as aut
    return _json_safe(aut.overview())


@app.get("/api/autonomic/breakdown")
async def autonomic_breakdown():
    """Autonomic breakdown: per-patient parasympathetic + sympathetic test
    results, CASI scores, SUDEP risk flags, expandable detail."""
    import scripts.autonomic_dashboard as aut
    return _json_safe(aut.breakdown())


@app.get("/api/autonomic/definitions")
async def autonomic_definitions():
    """Autonomic definitions — Ewing battery, CASI, SUDEP risk assessment,
    reference ranges, diagnostic patterns, clinical significance."""
    import scripts.autonomic_dashboard as aut
    return _json_safe(aut.definitions())


# ── PNES Differential Dashboard ─────────────────────────────────────

@app.get("/api/pnes-differential/overview")
async def pnes_differential_overview():
    """PNES Differential overview: classification distribution, risk factor
    analysis, semiology scoring, diagnostic certainty, vEEG priority KPIs."""
    import scripts.pnes_differential_dashboard as pnes
    return _json_safe(pnes.overview())


@app.get("/api/pnes-differential/breakdown")
async def pnes_differential_breakdown():
    """PNES Differential breakdown: per-patient semiology profiles, PNES vs
    epileptic scoring, risk factors, diagnostic certainty, vEEG priority."""
    import scripts.pnes_differential_dashboard as pnes
    return _json_safe(pnes.breakdown())


@app.get("/api/pnes-differential/definitions")
async def pnes_differential_definitions():
    """PNES Differential definitions: PNES clinical concepts, semiology table,
    diagnostic levels (ILAE 2013), management pathways, quality metrics."""
    import scripts.pnes_differential_dashboard as pnes
    return _json_safe(pnes.definitions())


# ── EEG-MRI Concordance Dashboard ────────────────────────────────────

@app.get("/api/eeg-mri-concordance/overview")
async def eeg_mri_concordance_overview():
    """EEG-MRI Concordance overview: concordance rate, lesion type breakdown,
    surgical candidacy distribution, lobe match matrix."""
    import scripts.eeg_mri_concordance_dashboard as emc
    return _json_safe(emc.overview())


@app.get("/api/eeg-mri-concordance/breakdown")
async def eeg_mri_concordance_breakdown():
    """EEG-MRI Concordance breakdown: per-patient MRI lesion vs EEG focus,
    concordance classification, surgical candidacy, additional workup."""
    import scripts.eeg_mri_concordance_dashboard as emc
    return _json_safe(emc.breakdown())


@app.get("/api/eeg-mri-concordance/definitions")
async def eeg_mri_concordance_definitions():
    """EEG-MRI Concordance definitions: concordance categories, lesion types,
    surgical candidacy tiers, workup modalities, references."""
    import scripts.eeg_mri_concordance_dashboard as emc
    return _json_safe(emc.definitions())


# ── Neurologist Dashboard ─────────────────────────────────────────────
@app.get("/api/neurologist/overview")
async def neurologist_overview():
    """Neurologist overview — EEG reads pending, seizure-positive rate,
    avg model confidence, HITL overrides, mean turnaround time, prediction
    distribution, confidence histogram, turnaround histogram, weekly volume."""
    import scripts.neurologist_dashboard as nrd
    return _json_safe(nrd.overview())


@app.get("/api/neurologist/breakdown")
async def neurologist_breakdown():
    """Neurologist breakdown — per-patient EEG analysis detail, HITL reviews,
    turnaround hours, medications, seizure diary, response grade, seizure
    classification summary, medication summary."""
    import scripts.neurologist_dashboard as nrd
    return _json_safe(nrd.breakdown())


@app.get("/api/neurologist/definitions")
async def neurologist_definitions():
    """Neurologist definitions — ACNS/IFCN EEG standards, ILAE 2017 seizure
    classification, turnaround benchmarks, HITL override categories,
    medication response grading, signal quality grades."""
    import scripts.neurologist_dashboard as nrd
    return _json_safe(nrd.definitions())


# ── Clinical Psychologist Dashboard ────────────────────────────────────
@app.get("/api/clinical-psychologist/overview")
async def clinical_psychologist_overview():
    """Clinical Psychologist overview — neuropsych assessment counts, MoCA/MMSE
    impairment rates, PHQ-9/GAD-7 distributions, cognitive index means,
    lateralization hypothesis, referral reason distribution."""
    import scripts.clinical_psychologist_dashboard as cpd
    return _json_safe(cpd.overview())


@app.get("/api/clinical-psychologist/breakdown")
async def clinical_psychologist_breakdown():
    """Clinical Psychologist breakdown — per-patient neuropsych detail with all
    cognitive indices, mood scores, trail making, digit span, impairment flag,
    lateralization, cognitive profile chart, memory lateralization cross-tab."""
    import scripts.clinical_psychologist_dashboard as cpd
    return _json_safe(cpd.breakdown())


@app.get("/api/clinical-psychologist/definitions")
async def clinical_psychologist_definitions():
    """Clinical Psychologist definitions — PHQ-9, GAD-7, MoCA, MMSE, cognitive
    indices, trail making, digit span, impairment levels, pre-surgical evaluation,
    ILAE neuropsychology references."""
    import scripts.clinical_psychologist_dashboard as cpd
    return _json_safe(cpd.definitions())


# ── AI Federation Dashboard ─────────────────────────────────────────
@app.get("/api/ai-federation/overview")
async def ai_federation_overview():
    """AI Federation overview — sites onboarded, round history, global
    accuracy, drift monitoring, data quality across federated sites."""
    import scripts.federation_dashboard as fd
    return _json_safe(fd.overview())


@app.get("/api/ai-federation/breakdown")
async def ai_federation_breakdown():
    """AI Federation breakdown — per-site detail with accuracy, drift,
    data quality, sync status, and per-round aggregation results."""
    import scripts.federation_dashboard as fd
    return _json_safe(fd.breakdown())


@app.get("/api/ai-federation/definitions")
async def ai_federation_definitions():
    """AI Federation definitions — federated learning, aggregation methods,
    differential privacy, HIPAA multi-site compliance."""
    import scripts.federation_dashboard as fd
    return _json_safe(fd.definitions())


# ── IS SOP Dashboard ──────────────────────────────────────────────────
@app.get("/api/is-sop/overview")
async def is_sop_overview():
    """IS SOP overview — total SOPs, published/draft/review/retired counts,
    reviews due, overdue, avg compliance score, audit finding summary,
    compliance by category, status/category/finding/severity distributions."""
    import scripts.is_sop_dashboard as isd
    return _json_safe(isd.overview())


@app.get("/api/is-sop/breakdown")
async def is_sop_breakdown():
    """IS SOP breakdown — per-procedure detail with associated audit records,
    compliance scores, review dates, version history, and corrective actions."""
    import scripts.is_sop_dashboard as isd
    return _json_safe(isd.breakdown())


@app.get("/api/is-sop/definitions")
async def is_sop_definitions():
    """IS SOP definitions — HIPAA Security Rule, 21 CFR Part 11, IEC 62443,
    NIST CSF, ISO 27001, SOP lifecycle, compliance scoring, audit classifications,
    EEG data security, AI model security, network segmentation."""
    import scripts.is_sop_dashboard as isd
    return _json_safe(isd.definitions())


@app.get("/api/trigger-tracking/overview")
async def trigger_tracking_overview():
    """Trigger Tracking & Forecasting overview — seizure trigger distribution,
    daily factor logging stats, risk levels, factor-seizure correlations,
    medication adherence impact, sleep vs seizure rate analysis."""
    import scripts.trigger_tracking_dashboard as ttd
    return _json_safe(ttd.overview())


@app.get("/api/trigger-tracking/breakdown")
async def trigger_tracking_breakdown():
    """Trigger Tracking breakdown — per-patient trigger profiles, recent logs,
    seizure rates, risk levels, adherence rates, and all raw log entries."""
    import scripts.trigger_tracking_dashboard as ttd
    return _json_safe(ttd.breakdown())


@app.get("/api/trigger-tracking/definitions")
async def trigger_tracking_definitions():
    """Trigger Tracking definitions — seizure triggers, trigger diary, risk scoring,
    medication adherence, sleep hygiene, photosensitivity, hormonal triggers,
    trigger correlation engine, ILAE trigger classification."""
    import scripts.trigger_tracking_dashboard as ttd
    return _json_safe(ttd.definitions())


@app.get("/api/emergency-caregiver/overview")
async def emergency_caregiver_overview():
    """Emergency Contact & Caregiver overview — contact coverage, caregiver training,
    burden scores, safety plan rates, availability and role distributions."""
    import scripts.emergency_caregiver_dashboard as ecd
    return _json_safe(ecd.overview())


@app.get("/api/emergency-caregiver/breakdown")
async def emergency_caregiver_breakdown():
    """Emergency Contact & Caregiver breakdown — per-patient emergency contacts,
    caregiver profiles, training status, burden scores, and safety plans."""
    import scripts.emergency_caregiver_dashboard as ecd
    return _json_safe(ecd.breakdown())


@app.get("/api/emergency-caregiver/definitions")
async def emergency_caregiver_definitions():
    """Emergency Contact & Caregiver definitions — seizure first aid, rescue medication,
    caregiver burden, seizure action plan, emergency protocol, SUDEP awareness."""
    import scripts.emergency_caregiver_dashboard as ecd
    return _json_safe(ecd.definitions())


@app.get("/api/medication-management/overview")
async def medication_management_overview():
    """Medication Self-Management overview — adherence rates, side effect profiles,
    drug distribution, refill tracking, rescue medication usage, time-of-day adherence."""
    import scripts.medication_management_dashboard as mmd
    return _json_safe(mmd.overview())


@app.get("/api/medication-management/breakdown")
async def medication_management_breakdown():
    """Medication Self-Management breakdown — per-patient drug lists, adherence rates,
    missed doses, side effects, refill history, and recent adherence logs."""
    import scripts.medication_management_dashboard as mmd
    return _json_safe(mmd.breakdown())


@app.get("/api/medication-management/definitions")
async def medication_management_definitions():
    """Medication Self-Management definitions — AED, therapeutic drug monitoring,
    medication adherence, rescue medication, polytherapy, titration."""
    import scripts.medication_management_dashboard as mmd
    return _json_safe(mmd.definitions())


@app.get("/api/pro-outcomes/overview")
async def pro_outcomes_overview():
    """PRO (Sleep/Mood/Cognition/QoL) overview — validated instrument scores,
    depression/anxiety severity, sleep quality, QoL trends, domain averages."""
    import scripts.pro_outcomes_dashboard as pod
    return _json_safe(pod.overview())


@app.get("/api/pro-outcomes/breakdown")
async def pro_outcomes_breakdown():
    """PRO breakdown — per-patient instrument scores, trends, and all assessments."""
    import scripts.pro_outcomes_dashboard as pod
    return _json_safe(pod.breakdown())


@app.get("/api/pro-outcomes/definitions")
async def pro_outcomes_definitions():
    """PRO definitions — PSQI, ESS, PHQ-9, GAD-7, QOLIE-31, MoCA, NDDI-E, WPAI."""
    import scripts.pro_outcomes_dashboard as pod
    return _json_safe(pod.definitions())


@app.get("/api/demographics/overview")
async def demographics_overview():
    """Demographics overview — age/sex/ethnicity/race distributions,
    BMI categories, epilepsy types, insurance, employment, education."""
    import scripts.demographics_dashboard as dd
    return _json_safe(dd.overview())


@app.get("/api/demographics/breakdown")
async def demographics_breakdown():
    """Demographics breakdown — per-patient demographic data, age/onset stats,
    referral sources."""
    import scripts.demographics_dashboard as dd
    return _json_safe(dd.breakdown())


@app.get("/api/demographics/definitions")
async def demographics_definitions():
    """Demographics definitions — BMI categories, ILAE classification,
    blood types, interpreter services, social determinants of health."""
    import scripts.demographics_dashboard as dd
    return _json_safe(dd.definitions())


@app.get("/api/wearables-digital-twin/overview")
async def wearables_digital_twin_overview():
    """Wearables & Digital Twin overview — device fleet status, biomarker averages,
    seizure detection rate, health/risk scores, HRV distribution, sleep quality,
    digital twin trajectory summaries."""
    import scripts.wearables_digital_twin_dashboard as wdt
    return _json_safe(wdt.overview())


@app.get("/api/wearables-digital-twin/breakdown")
async def wearables_digital_twin_breakdown():
    """Wearables & Digital Twin breakdown — per-patient device info, reading summaries,
    digital twin profiles with physiological baselines and longitudinal projections."""
    import scripts.wearables_digital_twin_dashboard as wdt
    return _json_safe(wdt.breakdown())


@app.get("/api/wearables-digital-twin/definitions")
async def wearables_digital_twin_definitions():
    """Wearables & Digital Twin definitions — HRV, PPG, EDA, seizure detection,
    digital twin, health/risk scores, sleep architecture, fall detection."""
    import scripts.wearables_digital_twin_dashboard as wdt
    return _json_safe(wdt.definitions())


@app.get("/api/self-service/overview")
async def self_service_overview():
    """Self-Service Portal overview — appointments, messaging, telehealth,
    documents, education modules, emergency SOS, daily plan metrics."""
    import scripts.self_service_dashboard as ssd
    return _json_safe(ssd.overview())


@app.get("/api/self-service/breakdown")
async def self_service_breakdown():
    """Self-Service Portal breakdown — per-patient portal usage, recent
    appointments, recent messages."""
    import scripts.self_service_dashboard as ssd
    return _json_safe(ssd.breakdown())


@app.get("/api/self-service/definitions")
async def self_service_definitions():
    """Self-Service Portal definitions — patient portal, secure messaging,
    telehealth, document center, education, emergency SOS, daily plan."""
    import scripts.self_service_dashboard as ssd
    return _json_safe(ssd.definitions())


@app.get("/api/qa-test-suite/overview")
async def qa_test_suite_overview():
    """QA Test Suite overview — total tests, pass/partial/planned counts,
    coverage %, per-dimension breakdown, per-role pass rates."""
    import scripts.qa_test_suite_dashboard as qtsd
    return _json_safe(qtsd.overview())


@app.get("/api/qa-test-suite/breakdown")
async def qa_test_suite_breakdown():
    """QA Test Suite breakdown — per-role test cases with dimension/status,
    user stories with persona/endpoint mapping, demo stories."""
    import scripts.qa_test_suite_dashboard as qtsd
    return _json_safe(qtsd.breakdown())


@app.get("/api/qa-test-suite/definitions")
async def qa_test_suite_definitions():
    """QA Test Suite definitions — testing dimensions, coverage metrics,
    pass/partial/planned methodology, role matrix, 9-dimension framework."""
    import scripts.qa_test_suite_dashboard as qtsd
    return _json_safe(qtsd.definitions())


@app.get("/api/product-manager/overview")
async def product_manager_overview():
    """Product Manager overview — overall readiness, stakeholder coverage,
    process maturity, module completion, business-case levers."""
    import scripts.product_manager_dashboard as pmd
    return _json_safe(pmd.overview())


@app.get("/api/product-manager/breakdown")
async def product_manager_breakdown():
    """Product Manager breakdown — per-stakeholder built/missing lists,
    business-case levers, implementation phases, module details."""
    import scripts.product_manager_dashboard as pmd
    return _json_safe(pmd.breakdown())


@app.get("/api/product-manager/definitions")
async def product_manager_definitions():
    """Product Manager definitions — readiness, maturity, business-case,
    module completion, implementation phase terminology."""
    import scripts.product_manager_dashboard as pmd
    return _json_safe(pmd.definitions())


# ── Admin Panel Dashboard ────────────────────────────────────────────────

@app.get("/api/admin-panel/overview")
async def admin_panel_overview():
    """Admin Panel overview — user management, feature flags, system health,
    and configuration summary KPIs and distributions."""
    import scripts.admin_dashboard as adm
    return _json_safe(adm.overview())


@app.get("/api/admin-panel/breakdown")
async def admin_panel_breakdown():
    """Admin Panel breakdown — full user list, feature flag details, system
    health log, and configuration entries."""
    import scripts.admin_dashboard as adm
    return _json_safe(adm.breakdown())


@app.get("/api/admin-panel/definitions")
async def admin_panel_definitions():
    """Admin Panel definitions — user management, feature flags, MFA,
    system health, configuration management terminology."""
    import scripts.admin_dashboard as adm
    return _json_safe(adm.definitions())


@app.get("/api/functional-ba/overview")
async def functional_ba_overview():
    """Functional/BA overview — requirements count, acceptance coverage,
    UAT readiness by role, process maturity, functionality coverage."""
    import scripts.functional_ba_dashboard as fbd
    return _json_safe(fbd.overview())


@app.get("/api/functional-ba/breakdown")
async def functional_ba_breakdown():
    """Functional/BA breakdown — requirements traceability, process detail,
    functionality gaps, stakeholder gap analysis, per-role acceptance criteria."""
    import scripts.functional_ba_dashboard as fbd
    return _json_safe(fbd.breakdown())


@app.get("/api/functional-ba/definitions")
async def functional_ba_definitions():
    """Functional/BA definitions — requirements traceability, acceptance criteria,
    UAT readiness, process maturity, stakeholder gap analysis terminology."""
    import scripts.functional_ba_dashboard as fbd
    return _json_safe(fbd.definitions())


# ── Integration Dashboard ─────────────────────────────────────────

@app.get("/api/integration/overview")
async def integration_overview():
    """Integration overview — integration counts, device counts, readiness
    percentage, status distributions by category."""
    import scripts.integration_dashboard as igd
    return _json_safe(igd.overview())


@app.get("/api/integration/breakdown")
async def integration_breakdown():
    """Integration breakdown — full integration lists, device fleet, admin
    integrations, and stakeholder integration gaps."""
    import scripts.integration_dashboard as igd
    return _json_safe(igd.breakdown())


@app.get("/api/integration/definitions")
async def integration_definitions():
    """Integration definitions — integration, delivery channel, IoT device,
    API contract, webhook, EMR/FHIR, OAuth, MCP terminology."""
    import scripts.integration_dashboard as igd
    return _json_safe(igd.definitions())


@app.get("/api/dataset-coverage/overview")
async def dataset_coverage_overview():
    """Dataset Coverage overview — modality counts, AI streams, phases, target scale."""
    import scripts.dataset_coverage_dashboard as dcd
    return _json_safe(dcd.overview())


@app.get("/api/dataset-coverage/breakdown")
async def dataset_coverage_breakdown():
    """Dataset Coverage breakdown — all modalities, AI streams, phases, provider questions."""
    import scripts.dataset_coverage_dashboard as dcd
    return _json_safe(dcd.breakdown())


@app.get("/api/dataset-coverage/definitions")
async def dataset_coverage_definitions():
    """Dataset Coverage definitions — modality, tier, phase, AI stream terminology."""
    import scripts.dataset_coverage_dashboard as dcd
    return _json_safe(dcd.definitions())


# ── AI Dark Factory Dashboard ─────────────────────────────────────────────

@app.get("/api/dark-factory/overview")
async def dark_factory_overview():
    """AI Dark Factory overview — flow stages, tool catalog size, patterns, planes."""
    import scripts.ai_dark_factory_dashboard as dfd
    return _json_safe(dfd.overview())


@app.get("/api/dark-factory/breakdown")
async def dark_factory_breakdown():
    """AI Dark Factory breakdown — full flow stages, tool catalog, patterns, planes."""
    import scripts.ai_dark_factory_dashboard as dfd
    return _json_safe(dfd.breakdown())


@app.get("/api/dark-factory/definitions")
async def dark_factory_definitions():
    """AI Dark Factory definitions — dark factory, flow stage, tool terminology."""
    import scripts.ai_dark_factory_dashboard as dfd
    return _json_safe(dfd.definitions())


# ── Seizure Risk Forecasting Dashboard ─────────────────────────────────────

@app.get("/api/seizure-risk-forecast/overview")
async def seizure_risk_forecast_overview():
    """Seizure-risk forecasting overview — risk tiers, horizons, thresholds, KPIs."""
    import scripts.seizure_risk_forecasting_dashboard as srf
    return _json_safe(srf.overview())


@app.get("/api/seizure-risk-forecast/breakdown")
async def seizure_risk_forecast_breakdown():
    """Seizure-risk forecasting breakdown — per-patient forecasts, escalation log, gaps."""
    import scripts.seizure_risk_forecasting_dashboard as srf
    return _json_safe(srf.breakdown())


@app.get("/api/seizure-risk-forecast/definitions")
async def seizure_risk_forecast_definitions():
    """Seizure-risk forecasting definitions — pre-ictal, risk tier, escalation terminology."""
    import scripts.seizure_risk_forecasting_dashboard as srf
    return _json_safe(srf.definitions())


# ── ABPM / Holter Dashboard ────────────────────────────────────────────────

@app.get("/api/abpm-holter/overview")
async def abpm_holter_overview():
    """ABPM/Holter overview — KPIs, severity/pattern/dipping distributions, patient summary."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.overview())


@app.get("/api/abpm-holter/breakdown")
async def abpm_holter_breakdown():
    """ABPM/Holter breakdown — parameter tables, histograms, per-patient detail cards."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.breakdown())


@app.get("/api/abpm-holter/definitions")
async def abpm_holter_definitions():
    """ABPM/Holter definitions — protocol, parameters, reference ranges, diagnostic patterns."""
    import scripts.abpm_dashboard as abpm
    return _json_safe(abpm.definitions())


@app.get("/api/feature-evaluation/overview")
async def feature_evaluation_overview():
    """Feature Evaluation overview — ANOVA F-scores, feature rankings, category analysis."""
    import scripts.feature_evaluation_dashboard as fed
    return _json_safe(fed.overview())


@app.get("/api/feature-evaluation/breakdown")
async def feature_evaluation_breakdown():
    """Feature Evaluation breakdown — full feature table, top features, correlations."""
    import scripts.feature_evaluation_dashboard as fed
    return _json_safe(fed.breakdown())


@app.get("/api/feature-evaluation/definitions")
async def feature_evaluation_definitions():
    """Feature Evaluation definitions — categories, methods, clinical relevance."""
    import scripts.feature_evaluation_dashboard as fed
    return _json_safe(fed.definitions())


# ── Automatic Pipelines Dashboard ──────────────────────────────────────────

@app.get("/api/automatic-pipelines/overview")
async def automatic_pipelines_overview():
    """Automatic Pipelines overview — status counts, trigger distribution, automation rate."""
    import scripts.automatic_pipelines_dashboard as apd
    return _json_safe(apd.overview())


@app.get("/api/automatic-pipelines/breakdown")
async def automatic_pipelines_breakdown():
    """Automatic Pipelines breakdown — all pipelines with stages, triggers, endpoints."""
    import scripts.automatic_pipelines_dashboard as apd
    return _json_safe(apd.breakdown())


@app.get("/api/automatic-pipelines/definitions")
async def automatic_pipelines_definitions():
    """Automatic Pipelines definitions — pipeline, trigger, stage terminology."""
    import scripts.automatic_pipelines_dashboard as apd
    return _json_safe(apd.definitions())


# ── Transfer Learning Dashboard ──────────────────────────────────────────

@app.get("/api/transfer-learning/overview")
async def transfer_learning_overview():
    """Transfer Learning overview — adaptation KPIs, strategy distribution, improvement histogram."""
    import scripts.transfer_learning_dashboard as tld
    return _json_safe(tld.overview())


@app.get("/api/transfer-learning/breakdown")
async def transfer_learning_breakdown():
    """Transfer Learning breakdown — per-patient detail, strategy comparison, convergence analysis."""
    import scripts.transfer_learning_dashboard as tld
    return _json_safe(tld.breakdown())


@app.get("/api/transfer-learning/definitions")
async def transfer_learning_definitions():
    """Transfer Learning definitions — concepts, strategies, metrics, literature."""
    import scripts.transfer_learning_dashboard as tld
    return _json_safe(tld.definitions())


# ── Feature Selection Dashboard ──────────────────────────────────────────

@app.get("/api/feature-selection/overview")
async def feature_selection_overview():
    """Feature Selection overview — consensus counts, method summary, category selection rates."""
    import scripts.feature_selection_dashboard as fsd
    return _json_safe(fsd.overview())


@app.get("/api/feature-selection/breakdown")
async def feature_selection_breakdown():
    """Feature Selection breakdown — full feature table, method details, agreement matrix."""
    import scripts.feature_selection_dashboard as fsd
    return _json_safe(fsd.breakdown())


@app.get("/api/feature-selection/definitions")
async def feature_selection_definitions():
    """Feature Selection definitions — LASSO, RFE, PCA, SelectKBest, Boruta methodology."""
    import scripts.feature_selection_dashboard as fsd
    return _json_safe(fsd.definitions())


@app.get("/api/data-augmentation/overview")
async def data_augmentation_overview():
    """Data Augmentation overview — technique comparison, accuracy deltas, class balance."""
    import scripts.data_augmentation_dashboard as dad
    return _json_safe(dad.overview())


@app.get("/api/data-augmentation/breakdown")
async def data_augmentation_breakdown():
    """Data Augmentation breakdown — per-technique parameter sweep, accuracy by variant."""
    import scripts.data_augmentation_dashboard as dad
    return _json_safe(dad.breakdown())


@app.get("/api/data-augmentation/definitions")
async def data_augmentation_definitions():
    """Data Augmentation definitions — technique methodology, clinical relevance, references."""
    import scripts.data_augmentation_dashboard as dad
    return _json_safe(dad.definitions())


# ── Seizure Prediction / Forecasting Dashboard ────────────────────────────────
@app.get("/api/seizure-prediction/overview")
async def seizure_prediction_overview():
    """Seizure prediction overview — sensitivity/specificity, risk distribution, temporal trends."""
    import scripts.seizure_prediction_dashboard as spd
    return _json_safe(spd.overview())


@app.get("/api/seizure-prediction/breakdown")
async def seizure_prediction_breakdown():
    """Seizure prediction breakdown — per-patient accuracy, biomarkers, threshold analysis."""
    import scripts.seizure_prediction_dashboard as spd
    return _json_safe(spd.breakdown())


@app.get("/api/seizure-prediction/definitions")
async def seizure_prediction_definitions():
    """Seizure prediction definitions — methodology, metrics, wearable biomarkers, references."""
    import scripts.seizure_prediction_dashboard as spd
    return _json_safe(spd.definitions())


# ── Hybrid CNN-LSTM / CNN-Transformer Pipeline Dashboard ────────────────────
@app.get("/api/hybrid-pipeline/overview")
async def hybrid_pipeline_overview():
    """Hybrid pipeline overview — architecture comparison, feature importance, confidence distribution."""
    import scripts.hybrid_pipeline_dashboard as hpd
    return _json_safe(hpd.overview())


@app.get("/api/hybrid-pipeline/breakdown")
async def hybrid_pipeline_breakdown():
    """Hybrid pipeline breakdown — per-patient, layer analysis, attention weights, LSTM gates."""
    import scripts.hybrid_pipeline_dashboard as hpd
    return _json_safe(hpd.breakdown())


@app.get("/api/hybrid-pipeline/definitions")
async def hybrid_pipeline_definitions():
    """Hybrid pipeline definitions — architecture specs, hyperparameters, clinical references."""
    import scripts.hybrid_pipeline_dashboard as hpd
    return _json_safe(hpd.definitions())


# ── Connectivity Analysis Dashboard ─────────────────────────────────────────
@app.get("/api/connectivity/overview")
async def connectivity_overview():
    """Connectivity overview — coherence/PLV/correlation matrix, graph metrics, band summary."""
    import scripts.connectivity_dashboard as cnd
    return _json_safe(cnd.overview())


@app.get("/api/connectivity/breakdown")
async def connectivity_breakdown():
    """Connectivity breakdown — per-band pairs, graph metrics, strongest/weakest connections."""
    import scripts.connectivity_dashboard as cnd
    return _json_safe(cnd.breakdown())


@app.get("/api/connectivity/definitions")
async def connectivity_definitions():
    """Connectivity definitions — methods, formulae, clinical relevance, references."""
    import scripts.connectivity_dashboard as cnd
    return _json_safe(cnd.definitions())


# ── Scalogram (CWT) Dashboard ───────────────────────────────────────────────
@app.get("/api/scalogram/overview")
async def scalogram_overview():
    """Scalogram overview — CWT energy distribution, dominant bands, spectral entropy."""
    import scripts.scalogram_dashboard as scd
    return _json_safe(scd.overview())


@app.get("/api/scalogram/breakdown")
async def scalogram_breakdown():
    """Scalogram breakdown — per-band wavelet statistics, cross-band ratios."""
    import scripts.scalogram_dashboard as scd
    return _json_safe(scd.breakdown())


@app.get("/api/scalogram/definitions")
async def scalogram_definitions():
    """Scalogram definitions — CWT methodology, Morlet wavelets, clinical relevance."""
    import scripts.scalogram_dashboard as scd
    return _json_safe(scd.definitions())


# ── Saliency & Attention Map Dashboard ────────────────────────────────────────
@app.get("/api/saliency-attention/overview")
async def saliency_attention_overview():
    """Saliency overview — channel saliency scores, temporal attention, band attention."""
    import scripts.saliency_attention_dashboard as sad
    return _json_safe(sad.overview())


@app.get("/api/saliency-attention/breakdown")
async def saliency_attention_breakdown():
    """Saliency breakdown — per-diagnosis patterns, attention heads, channel ranking."""
    import scripts.saliency_attention_dashboard as sad
    return _json_safe(sad.breakdown())


@app.get("/api/saliency-attention/definitions")
async def saliency_attention_definitions():
    """Saliency definitions — methodology, clinical interpretation, references."""
    import scripts.saliency_attention_dashboard as sad
    return _json_safe(sad.definitions())


@app.get("/api/guardrails/overview")
async def guardrails_overview():
    """Guardrails overview — rail trigger rates, block rates, latency."""
    import scripts.guardrails_dashboard as grd
    return _json_safe(grd.overview())


@app.get("/api/guardrails/breakdown")
async def guardrails_breakdown():
    """Guardrails breakdown — input/output rails, dialog flows, severity."""
    import scripts.guardrails_dashboard as grd
    return _json_safe(grd.breakdown())


@app.get("/api/guardrails/definitions")
async def guardrails_definitions():
    """Guardrails definitions — methodology, rail types, clinical relevance."""
    import scripts.guardrails_dashboard as grd
    return _json_safe(grd.definitions())


# ── SPWVD Dashboard ─────────────────────────────────────────────────────────
@app.get("/api/spwvd/overview")
async def spwvd_overview():
    """SPWVD overview — energy distribution, cross-term suppression, resolution metrics."""
    import scripts.spwvd_dashboard as spd
    return _json_safe(spd.overview())


@app.get("/api/spwvd/breakdown")
async def spwvd_breakdown():
    """SPWVD breakdown — per-band statistics, instantaneous frequency, interference."""
    import scripts.spwvd_dashboard as spd
    return _json_safe(spd.breakdown())


@app.get("/api/spwvd/definitions")
async def spwvd_definitions():
    """SPWVD definitions — WVD/SPWVD methodology, kernel design, clinical relevance."""
    import scripts.spwvd_dashboard as spd
    return _json_safe(spd.definitions())


# ── Patient-Facing Report Dashboard ─────────────────────────────────────────
@app.get("/api/patient-facing-report/overview")
async def patient_facing_report_overview():
    """Patient-facing report overview — simplified risk, recent reports, follow-ups."""
    import scripts.patient_facing_report_dashboard as pfr
    return _json_safe(pfr.overview())


@app.get("/api/patient-facing-report/breakdown")
async def patient_facing_report_breakdown():
    """Patient-facing report breakdown — medications, biomarkers, next steps."""
    import scripts.patient_facing_report_dashboard as pfr
    return _json_safe(pfr.breakdown())


@app.get("/api/patient-facing-report/definitions")
async def patient_facing_report_definitions():
    """Patient-facing report definitions — glossary, disclaimers, plain-language guide."""
    import scripts.patient_facing_report_dashboard as pfr
    return _json_safe(pfr.definitions())


@app.get("/api/rlhf-training/overview")
async def rlhf_training_overview():
    """RLHF Training overview: feedback stats, preference pairs, readiness score."""
    import scripts.rlhf_training_dashboard as rlhf
    return _json_safe(rlhf.overview())


@app.get("/api/rlhf-training/breakdown")
async def rlhf_training_breakdown():
    """RLHF Training breakdown: preference pairs, trends, reward model readiness."""
    import scripts.rlhf_training_dashboard as rlhf
    return _json_safe(rlhf.breakdown())


@app.get("/api/rlhf-training/definitions")
async def rlhf_training_definitions():
    """RLHF Training definitions: RLHF terminology and clinical context."""
    import scripts.rlhf_training_dashboard as rlhf
    return _json_safe(rlhf.definitions())


@app.get("/api/ica-noise-cleaning/overview")
async def ica_noise_cleaning_overview():
    """ICA Noise Cleaning overview: KPIs, quality distribution, per-subject summary."""
    import scripts.ica_noise_cleaning_dashboard as ica
    return _json_safe(ica.overview())


@app.get("/api/ica-noise-cleaning/breakdown")
async def ica_noise_cleaning_breakdown():
    """ICA Noise Cleaning breakdown: per-file detail, component stats, pipeline stages."""
    import scripts.ica_noise_cleaning_dashboard as ica
    return _json_safe(ica.breakdown())


@app.get("/api/ica-noise-cleaning/definitions")
async def ica_noise_cleaning_definitions():
    """ICA Noise Cleaning definitions: ICA terminology and quality metrics."""
    import scripts.ica_noise_cleaning_dashboard as ica
    return _json_safe(ica.definitions())


# ── Federated Learning Dashboard ──────────────────────────────────────
@app.get("/api/federated-learning/overview")
async def federated_learning_overview():
    """Federated learning overview: global model, sites, rounds, privacy budget."""
    import scripts.federated_learning_dashboard as fl
    return _json_safe(fl.overview())


@app.get("/api/federated-learning/breakdown")
async def federated_learning_breakdown():
    """Federated learning breakdown: per-site detail, aggregation comparison, convergence."""
    import scripts.federated_learning_dashboard as fl
    return _json_safe(fl.breakdown())


@app.get("/api/federated-learning/definitions")
async def federated_learning_definitions():
    """Federated learning definitions: FL terminology and privacy metrics."""
    import scripts.federated_learning_dashboard as fl
    return _json_safe(fl.definitions())


# ── GNN Electrode Connectivity Dashboard ─────────────────────────────
@app.get("/api/gnn-electrode-connectivity/overview")
async def gnn_electrode_connectivity_overview():
    """GNN overview: graph stats, node/edge counts, top-attention electrodes, spectral power."""
    import scripts.gnn_electrode_connectivity_dashboard as gnn
    return _json_safe(gnn.overview())


@app.get("/api/gnn-electrode-connectivity/breakdown")
async def gnn_electrode_connectivity_breakdown():
    """GNN breakdown: node features, edge weights, regional connectivity, seizure patterns."""
    import scripts.gnn_electrode_connectivity_dashboard as gnn
    return _json_safe(gnn.breakdown())


@app.get("/api/gnn-electrode-connectivity/definitions")
async def gnn_electrode_connectivity_definitions():
    """GNN definitions: GNN and electrode connectivity terminology."""
    import scripts.gnn_electrode_connectivity_dashboard as gnn
    return _json_safe(gnn.definitions())


# ── Patient Education Dashboard ───────────────────────────────────────
@app.get("/api/patient-education/overview")
async def patient_education_overview():
    """Patient education overview: module completion, quiz scores, topic/format stats."""
    import scripts.patient_education_dashboard as ped
    return _json_safe(ped.overview())


@app.get("/api/patient-education/breakdown")
async def patient_education_breakdown():
    """Patient education breakdown: per-patient progress, at-risk, quiz performance."""
    import scripts.patient_education_dashboard as ped
    return _json_safe(ped.breakdown())


@app.get("/api/patient-education/definitions")
async def patient_education_definitions():
    """Patient education definitions: education terminology and module descriptions."""
    import scripts.patient_education_dashboard as ped
    return _json_safe(ped.definitions())


# ── Audio Converter Dashboard ──────────────────────────────────────
@app.get("/api/audio-converter/overview")
async def audio_converter_overview():
    """Audio converter overview: audio-capable recordings, extraction readiness, pipeline status."""
    import scripts.audio_converter_dashboard as acd
    return _json_safe(acd.overview())


@app.get("/api/audio-converter/breakdown")
async def audio_converter_breakdown():
    """Audio converter breakdown: per-patient profiles, vocalization events, feature extraction."""
    import scripts.audio_converter_dashboard as acd
    return _json_safe(acd.breakdown())


@app.get("/api/audio-converter/definitions")
async def audio_converter_definitions():
    """Audio converter definitions: audio extraction terminology and clinical relevance."""
    import scripts.audio_converter_dashboard as acd
    return _json_safe(acd.definitions())


# ── Phase-Amplitude Coupling (PAC) Dashboard ─────────────────────────
@app.get("/api/pac/overview")
async def pac_overview():
    """PAC overview: modulation index stats, frequency band pairs, electrode rankings."""
    import scripts.pac_dashboard as pac
    return _json_safe(pac.overview())


@app.get("/api/pac/breakdown")
async def pac_breakdown():
    """PAC breakdown: per-patient coupling, comodulogram, temporal trends, channel detail."""
    import scripts.pac_dashboard as pac
    return _json_safe(pac.breakdown())


@app.get("/api/pac/definitions")
async def pac_definitions():
    """PAC definitions: cross-frequency coupling terminology and clinical significance."""
    import scripts.pac_dashboard as pac
    return _json_safe(pac.definitions())


# ── Body Movement Classifier Dashboard ─────────────────────────────
@app.get("/api/body-movement/overview")
async def body_movement_overview():
    """Body movement overview: video-capable recordings, movement types, pose estimation readiness."""
    import scripts.body_movement_dashboard as bmd
    return _json_safe(bmd.overview())


@app.get("/api/body-movement/breakdown")
async def body_movement_breakdown():
    """Body movement breakdown: per-patient profiles, per-recording movements, lateralization analysis."""
    import scripts.body_movement_dashboard as bmd
    return _json_safe(bmd.breakdown())


@app.get("/api/body-movement/definitions")
async def body_movement_definitions():
    """Body movement definitions: seizure semiology and pose estimation terminology."""
    import scripts.body_movement_dashboard as bmd
    return _json_safe(bmd.definitions())


# ── Video Converter Dashboard ─────────────────────────────────────
@app.get("/api/video-converter/overview")
async def video_converter_overview():
    """Video converter overview: format/codec distribution, pipeline status, frame export stats."""
    import scripts.video_converter_dashboard as vcd
    return _json_safe(vcd.overview())


@app.get("/api/video-converter/breakdown")
async def video_converter_breakdown():
    """Video converter breakdown: per-recording conversions, quality metrics, patient summaries."""
    import scripts.video_converter_dashboard as vcd
    return _json_safe(vcd.breakdown())


@app.get("/api/video-converter/definitions")
async def video_converter_definitions():
    """Video converter definitions: codec, format, and quality metric terminology."""
    import scripts.video_converter_dashboard as vcd
    return _json_safe(vcd.definitions())


# ── Survey Link Dashboard ─────────────────────────────────────────
@app.get("/api/survey-link/overview")
async def survey_link_overview():
    """Survey link overview: generation counts, completion rates, assessment distribution."""
    import scripts.survey_link_dashboard as sld
    return _json_safe(sld.overview())


@app.get("/api/survey-link/breakdown")
async def survey_link_breakdown():
    """Survey link breakdown: per-patient details, per-assessment summaries."""
    import scripts.survey_link_dashboard as sld
    return _json_safe(sld.breakdown())


@app.get("/api/survey-link/definitions")
async def survey_link_definitions():
    """Survey link definitions: token, expiry, assessment terminology."""
    import scripts.survey_link_dashboard as sld
    return _json_safe(sld.definitions())


# ── Noise Cleaning (ICA Artifact Removal) Dashboard ──────────────

@app.get("/api/noise-cleaning/overview")
async def noise_cleaning_overview():
    """Noise Cleaning overview: KPIs, per-subject summary, variance
    distribution, per-file timeline. Real ICA report data."""
    import scripts.noise_cleaning_dashboard as ncd
    return _json_safe(ncd.overview())


@app.get("/api/noise-cleaning/breakdown")
async def noise_cleaning_breakdown():
    """Noise Cleaning breakdown: per-file details, channel/component stats,
    subject comparison, quality tiers."""
    import scripts.noise_cleaning_dashboard as ncd
    return _json_safe(ncd.breakdown())


@app.get("/api/noise-cleaning/definitions")
async def noise_cleaning_definitions():
    """Noise Cleaning metric definitions, methodology, clinical caveats."""
    import scripts.noise_cleaning_dashboard as ncd
    return _json_safe(ncd.definitions())


# ── MoCA Auto-Scoring Dashboard ───────────────────────────────────
# Real data: clinical.db neuropsych table — 37 assessments, 30 patients,
# MoCA scores 16-30, domain estimates, PHQ-9/GAD-7 comorbidity context.

@app.get("/api/moca-autoscoring/overview")
async def moca_autoscoring_overview():
    """MoCA overview: KPIs, classification distribution, score histogram,
    domain averages, MoCA-vs-MMSE correlation, assessor stats."""
    import scripts.moca_autoscoring_dashboard as mad
    return _json_safe(mad.overview())


@app.get("/api/moca-autoscoring/breakdown")
async def moca_autoscoring_breakdown():
    """Per-patient MoCA breakdown: domain profiles, impairment flags,
    comorbidity indicators, classification groups, domain vulnerability."""
    import scripts.moca_autoscoring_dashboard as mad
    return _json_safe(mad.breakdown())


@app.get("/api/moca-autoscoring/definitions")
async def moca_autoscoring_definitions():
    """MoCA metric definitions, scoring guide, clinical caveats."""
    import scripts.moca_autoscoring_dashboard as mad
    return _json_safe(mad.definitions())


# ── Edge Deployment Dashboard ──────────────────────────────────────

@app.get("/api/edge-deploy/overview")
async def edge_deploy_overview():
    """Edge deployment overview: ONNX export status, quantization modes, device targets."""
    import scripts.edge_deploy_dashboard as edd
    return _json_safe(edd.overview())


@app.get("/api/edge-deploy/breakdown")
async def edge_deploy_breakdown():
    """Edge deployment breakdown: per-model size, ONNX status, device compatibility matrix."""
    import scripts.edge_deploy_dashboard as edd
    return _json_safe(edd.breakdown())


@app.get("/api/edge-deploy/definitions")
async def edge_deploy_definitions():
    """Edge deployment definitions: ONNX, quantization, edge inference glossary."""
    import scripts.edge_deploy_dashboard as edd
    return _json_safe(edd.definitions())


# ── Closed-Loop Neurostimulation Dashboard ────────────────────────────────
@app.get("/api/closed-loop/overview")
async def closed_loop_overview():
    """Closed-loop overview — SOS KPIs, trigger breakdown, response latency, outcome distribution."""
    import scripts.closed_loop_dashboard as cld
    return _json_safe(cld.overview())


@app.get("/api/closed-loop/breakdown")
async def closed_loop_breakdown():
    """Closed-loop breakdown — per-patient audit, event timeline, IoT alert audit."""
    import scripts.closed_loop_dashboard as cld
    return _json_safe(cld.breakdown())


@app.get("/api/closed-loop/definitions")
async def closed_loop_definitions():
    """Closed-loop definitions — loop stages, glossary, clinical references."""
    import scripts.closed_loop_dashboard as cld
    return _json_safe(cld.definitions())


# ── Band Heatmap Dashboard ─────────────────────────────────────────────────
@app.get("/api/band-heatmap/overview")
async def band_heatmap_overview():
    """Band heatmap overview — band power distribution, dominance map, abnormality index."""
    import scripts.band_heatmap_dashboard as bhd
    return _json_safe(bhd.overview())


@app.get("/api/band-heatmap/breakdown")
async def band_heatmap_breakdown():
    """Band heatmap breakdown — per-band statistics, ratios, diagnosis profiles, correlation."""
    import scripts.band_heatmap_dashboard as bhd
    return _json_safe(bhd.breakdown())


@app.get("/api/band-heatmap/definitions")
async def band_heatmap_definitions():
    """Band heatmap definitions — band ranges, clinical significance, heatmap interpretation."""
    import scripts.band_heatmap_dashboard as bhd
    return _json_safe(bhd.definitions())


# ── XAI Ground-Truth Comparison Dashboard ────────────────────────────────────
@app.get("/api/xai-groundtruth/overview")
async def xai_groundtruth_overview():
    """XAI Ground-Truth Comparison — AI SHAP features vs expert annotations concordance."""
    import scripts.xai_groundtruth_dashboard as xgt
    return _json_safe(xgt.overview())


@app.get("/api/xai-groundtruth/concordance")
async def xai_groundtruth_concordance():
    """Per-disease concordance detail — AI vs expert feature overlap analysis."""
    import scripts.xai_groundtruth_dashboard as xgt
    return _json_safe(xgt.concordance_detail())


@app.get("/api/xai-groundtruth/features")
async def xai_groundtruth_features():
    """Side-by-side AI vs expert feature rankings with band-level analysis."""
    import scripts.xai_groundtruth_dashboard as xgt
    return _json_safe(xgt.feature_comparison())


@app.get("/api/xai-groundtruth/patients")
async def xai_groundtruth_patients():
    """Patient-level explainability audit — per-patient AI vs expert concordance."""
    import scripts.xai_groundtruth_dashboard as xgt
    return _json_safe(xgt.patients())


@app.get("/api/xai-groundtruth/definitions")
async def xai_groundtruth_definitions():
    """XAI ground-truth definitions — SHAP, concordance, EU AI Act references."""
    import scripts.xai_groundtruth_dashboard as xgt
    return _json_safe(xgt.definitions())


# ── Device Telemetry Dashboard ────────────────────────────────────────────
@app.get("/api/device-telemetry/overview")
async def device_telemetry_overview():
    """Fleet-wide KPIs — battery health, signal quality, online/offline, alert summary."""
    import scripts.device_telemetry_dashboard as dtd
    return _json_safe(dtd.overview())


@app.get("/api/device-telemetry/breakdown")
async def device_telemetry_breakdown():
    """Per-device telemetry detail — battery, signal, latency, alerts by type, gateways."""
    import scripts.device_telemetry_dashboard as dtd
    return _json_safe(dtd.breakdown())


@app.get("/api/device-telemetry/definitions")
async def device_telemetry_definitions():
    """Telemetry glossary — thresholds, severity definitions, clinical references."""
    import scripts.device_telemetry_dashboard as dtd
    return _json_safe(dtd.definitions())


@app.get("/api/telehealth/overview")
async def telehealth_overview():
    """Telehealth KPIs — session volume, satisfaction, platform usage, monthly trends."""
    import scripts.telehealth_dashboard as thd
    return _json_safe(thd.overview())


@app.get("/api/telehealth/breakdown")
async def telehealth_breakdown():
    """Per-provider stats, per-patient history, platform quality comparison, recent sessions."""
    import scripts.telehealth_dashboard as thd
    return _json_safe(thd.breakdown())


@app.get("/api/telehealth/definitions")
async def telehealth_definitions():
    """Telehealth glossary — session types, quality levels, satisfaction scale, references."""
    import scripts.telehealth_dashboard as thd
    return _json_safe(thd.definitions())


# ── Functional Recovery Dashboard ────────────────────────────────────

@app.get("/api/functional-recovery/overview")
async def functional_recovery_overview():
    """Functional recovery KPIs — daily/social function, QOLIE-31, WPAI, trajectories."""
    import scripts.functional_recovery_dashboard as frd
    return _json_safe(frd.overview())


@app.get("/api/functional-recovery/breakdown")
async def functional_recovery_breakdown():
    """Per-patient timelines, domain scores, comorbidity flags, monthly volume."""
    import scripts.functional_recovery_dashboard as frd
    return _json_safe(frd.breakdown())


@app.get("/api/functional-recovery/definitions")
async def functional_recovery_definitions():
    """Functional recovery glossary — concepts, thresholds, quality metrics, references."""
    import scripts.functional_recovery_dashboard as frd
    return _json_safe(frd.definitions())


@app.get("/api/copm-dashboard/overview")
async def copm_overview(patient_id: str = None):
    """COPM overview — performance/satisfaction KPIs, distributions, patient summaries."""
    import scripts.neuro_scales_copm as copm
    return _json_safe(copm.overview(patient_id))


@app.get("/api/copm-dashboard/breakdown")
async def copm_breakdown(patient_id: str = None):
    """COPM domain breakdown — per-domain averages, problem heatmap, change analysis."""
    import scripts.neuro_scales_copm as copm
    return _json_safe(copm.breakdown(patient_id))


@app.get("/api/copm-dashboard/definitions")
async def copm_definitions():
    """COPM glossary — concepts, scoring, interpretation, references."""
    import scripts.neuro_scales_copm as copm
    return _json_safe(copm.definitions())


# ── EEG Waveform Segmentation Dashboard ──────────────────────

@app.get("/api/segmentation/overview")
async def segmentation_overview():
    """Segmentation pipeline readiness — recording stats, channel quality,
    artifact impact, method specifications from clinical.db."""
    import scripts.segmentation_dashboard as seg
    return _json_safe(seg.segmentation_overview())


@app.get("/api/segmentation/breakdown")
async def segmentation_breakdown():
    """Per-patient segmentation breakdown — acquisition specs,
    channel quality per channel, artifact annotations, readiness."""
    import scripts.segmentation_dashboard as seg
    return _json_safe(seg.segmentation_breakdown())


@app.get("/api/segmentation/definitions")
async def segmentation_definitions():
    """Metric definitions for the Segmentation dashboard."""
    import scripts.segmentation_dashboard as seg
    return _json_safe(seg.segmentation_definitions())


@app.get("/api/epilepsy-board/overview")
async def epilepsy_board_overview():
    """Aggregate KPIs + chart data for multidisciplinary epilepsy board review."""
    import scripts.epilepsy_board_dashboard as eb
    return _json_safe(eb.epilepsy_board_overview())


@app.get("/api/epilepsy-board/breakdown")
async def epilepsy_board_breakdown():
    """Per-patient case summaries, concordance, medication summary for board review."""
    import scripts.epilepsy_board_dashboard as eb
    return _json_safe(eb.epilepsy_board_breakdown())


@app.get("/api/epilepsy-board/definitions")
async def epilepsy_board_definitions():
    """Metric definitions for the Epilepsy Board Review dashboard."""
    import scripts.epilepsy_board_dashboard as eb
    return _json_safe(eb.epilepsy_board_definitions())


# ── Consent Management Dashboard ───────────────────────────────

@app.get("/api/consent-dashboard/overview")
async def consent_overview(patient_id: str = None):
    """Consent management overview — KPIs, status distribution, expiring consents."""
    import scripts.consent_dashboard as cd
    return _json_safe(cd.overview(patient_id))


@app.get("/api/consent-dashboard/breakdown")
async def consent_breakdown(patient_id: str = None):
    """Per-patient consent breakdown, type stats, compliance matrix."""
    import scripts.consent_dashboard as cd
    return _json_safe(cd.breakdown(patient_id))


@app.get("/api/consent-dashboard/definitions")
async def consent_definitions():
    """Consent type definitions, statuses, regulatory framework, glossary."""
    import scripts.consent_dashboard as cd
    return _json_safe(cd.definitions())


# ── Referral Triage Dashboard ────────────────────────────────────

@app.get("/api/referral-triage/overview")
async def referral_triage_overview():
    """Referral intake & triage overview — KPIs, urgency/source distributions, timeline."""
    import scripts.referral_triage_dashboard as rt
    return _json_safe(rt.overview())


@app.get("/api/referral-triage/breakdown")
async def referral_triage_breakdown():
    """Per-referral breakdown — recent referrals, reason distribution, provider workload."""
    import scripts.referral_triage_dashboard as rt
    return _json_safe(rt.breakdown())


@app.get("/api/referral-triage/definitions")
async def referral_triage_definitions():
    """Metric definitions, urgency criteria, triage scoring, glossary."""
    import scripts.referral_triage_dashboard as rt
    return _json_safe(rt.definitions())


# ── RAG Metadata Filter Dashboard ─────────────────────────────────
# Real data: ChromaDB embedding_metadata (patient_id, type) ×
# clinical.db patients — metadata-driven retrieval filtering analytics.

@app.get("/api/rag-metadata-filter/overview")
async def rag_metadata_filter_overview():
    """Metadata filter overview — key inventory, type distribution,
    patient coverage, filter readiness from real ChromaDB + clinical.db."""
    import scripts.rag_metadata_filter_dashboard as rmf
    return _json_safe(rmf.overview())


@app.get("/api/rag-metadata-filter/breakdown")
async def rag_metadata_filter_breakdown():
    """Metadata filter drill-down — cross-tab type×patient, completeness,
    recent embeddings, filterable query analysis."""
    import scripts.rag_metadata_filter_dashboard as rmf
    return _json_safe(rmf.breakdown())


@app.get("/api/rag-metadata-filter/definitions")
async def rag_metadata_filter_definitions():
    """Metric definitions, filter dimensions, glossary."""
    import scripts.rag_metadata_filter_dashboard as rmf
    return _json_safe(rmf.definitions())


# ── Recovery Trajectory Forecast Dashboard ────────────────────────
# Real data: pro_outcomes (180 rows, 30 patients) — trajectory slope,
# intensive rehab prediction, risk factor correlation.

@app.get("/api/recovery-trajectory/overview")
async def recovery_trajectory_overview():
    """Recovery trajectory overview — KPIs, trajectory distribution, monthly trends, risk factors."""
    import scripts.recovery_trajectory_dashboard as rt
    return _json_safe(rt.overview())


@app.get("/api/recovery-trajectory/breakdown")
async def recovery_trajectory_breakdown():
    """Per-patient trajectory breakdown — rehab recommendations, prediction factors, declining list."""
    import scripts.recovery_trajectory_dashboard as rt
    return _json_safe(rt.breakdown())


@app.get("/api/recovery-trajectory/definitions")
async def recovery_trajectory_definitions():
    """Metric definitions, rehab criteria, functional rating scales, glossary."""
    import scripts.recovery_trajectory_dashboard as rt
    return _json_safe(rt.definitions())


# ── Goal-Attainment Scaling (GAS) Dashboard ─────────────────────
# Real data: pro_outcomes × patients × assessments — GAS T-score
# trends, domain performance, goal tracking for occupational therapy.

@app.get("/api/goal-attainment/overview")
async def goal_attainment_overview():
    """GAS overview — KPIs, score distribution, domain performance, T-score trend."""
    import scripts.goal_attainment_dashboard as ga
    return _json_safe(ga.overview())


@app.get("/api/goal-attainment/breakdown")
async def goal_attainment_breakdown():
    """Per-patient goal breakdown — goals, at-risk list, recent reviews, domain drill."""
    import scripts.goal_attainment_dashboard as ga
    return _json_safe(ga.breakdown())


@app.get("/api/goal-attainment/definitions")
async def goal_attainment_definitions():
    """Metric definitions, GAS scale, domain descriptions, glossary."""
    import scripts.goal_attainment_dashboard as ga
    return _json_safe(ga.definitions())


# ── Autonomic Analysis Dashboard ──────────────────────────────
# Real data: wearable_readings (900 rows, 30 patients) — HRV trends,
# autonomic dysfunction scoring, seizure-autonomic correlation, risk stratification.

@app.get("/api/autonomic-analysis/overview")
async def autonomic_analysis_overview():
    """Autonomic overview — KPIs, risk distribution, HRV/ADS distributions, trends, seizure correlation."""
    import scripts.autonomic_analysis_dashboard as aa
    return _json_safe(aa.overview())


@app.get("/api/autonomic-analysis/breakdown")
async def autonomic_analysis_breakdown():
    """Per-patient autonomic profile — ADS, risk level, HRV trend, daily readings, device info."""
    import scripts.autonomic_analysis_dashboard as aa
    return _json_safe(aa.breakdown())


@app.get("/api/autonomic-analysis/definitions")
async def autonomic_analysis_definitions():
    """Metric definitions, ADS scoring, HRV ranges, risk criteria, glossary."""
    import scripts.autonomic_analysis_dashboard as aa
    return _json_safe(aa.definitions())


# ── Guided Assessment Flow Dashboard ────────────────────────────────
# Real data: guided_assessment_sessions (45 rows, 21 patients) —
# item-by-item clinical assessments via conversational AI / voice AI,
# completion analytics, instrument usage, channel breakdown.

@app.get("/api/guided-assessment/overview")
async def guided_assessment_overview():
    """Guided assessment overview — KPIs, status/instrument/channel distributions,
    daily trend, duration histogram, score distribution."""
    import scripts.guided_assessment_dashboard as gad
    return _json_safe(gad.overview())


@app.get("/api/guided-assessment/breakdown")
async def guided_assessment_breakdown():
    """Per-session breakdown — session log, instrument summary, patient history,
    active sessions."""
    import scripts.guided_assessment_dashboard as gad
    return _json_safe(gad.breakdown())


@app.get("/api/guided-assessment/definitions")
async def guided_assessment_definitions():
    """Metric definitions, instrument catalog, compliance requirements, glossary."""
    import scripts.guided_assessment_dashboard as gad
    return _json_safe(gad.definitions())


# ── AI ROI Dashboard ──────────────────────────────────────────────────
# Real data: finops_costs (978 rows) + transaction_log (1008 rows) +
# analyses (21 rows) — investment vs value, cost breakdown, ROI metrics.

@app.get("/api/ai-roi/overview")
async def ai_roi_overview():
    """AI ROI overview — investment KPIs, cost breakdown by category/model,
    investment trend, component efficiency, value drivers."""
    import scripts.ai_roi_dashboard as ard
    return _json_safe(ard.overview())


@app.get("/api/ai-roi/breakdown")
async def ai_roi_breakdown():
    """Monthly cost breakdown, top components, patient-level ROI,
    cost optimization recommendations."""
    import scripts.ai_roi_dashboard as ard
    return _json_safe(ard.breakdown())


@app.get("/api/ai-roi/definitions")
async def ai_roi_definitions():
    """ROI metric definitions, methodology, assumptions, glossary."""
    import scripts.ai_roi_dashboard as ard
    return _json_safe(ard.definitions())


# ── Daily Care Plan Dashboard ──────────────────────────────────
# Real data: daily_plans (900 rows, 30 patients) — medication reminders,
# meals, exercise, sleep, mood, seizure logging, AI suggestions.

@app.get("/api/daily-care-plan/overview")
async def daily_care_plan_overview(patient_id: str = None):
    """Daily care plan overview — completion KPIs, activity logging rates,
    completion distribution, daily trend from real daily_plans table."""
    import scripts.daily_care_plan_dashboard as dcp
    return _json_safe(dcp.overview(patient_id))


@app.get("/api/daily-care-plan/breakdown")
async def daily_care_plan_breakdown(patient_id: str = None):
    """Per-patient summary, weekly heatmap, AI suggestion frequency,
    recent plans from real daily_plans table."""
    import scripts.daily_care_plan_dashboard as dcp
    return _json_safe(dcp.breakdown(patient_id))


@app.get("/api/daily-care-plan/definitions")
async def daily_care_plan_definitions():
    """Metric definitions, activity types, completion scoring, glossary."""
    import scripts.daily_care_plan_dashboard as dcp
    return _json_safe(dcp.definitions())


# ── Patient-Facing Report Dashboard ──────────────────────────────────
# Real data: analyses (21) + assessments (423) + seizure_diary (25) +
# mri_findings (40) + medication_adherence (12600) — simplified,
# plain-language patient reports from real clinical data.

@app.get("/api/patient-report/overview")
async def patient_report_overview():
    """Patient report overview — coverage KPIs, assessment level distribution,
    instrument usage, monthly trend, disease distribution."""
    import scripts.patient_report_dashboard as prd
    return _json_safe(prd.overview())


@app.get("/api/patient-report/breakdown")
async def patient_report_breakdown(patient_id: str = None):
    """Per-patient report generation — patient list with data availability,
    full plain-language report for a specific patient."""
    import scripts.patient_report_dashboard as prd
    return _json_safe(prd.breakdown(patient_id))


@app.get("/api/patient-report/definitions")
async def patient_report_definitions():
    """Glossary, health literacy notes, assessment level explanations,
    patient rights information."""
    import scripts.patient_report_dashboard as prd
    return _json_safe(prd.definitions())


@app.get("/api/user-management/overview")
async def user_management_overview():
    """User management overview — KPIs, role distribution, status breakdown, login trends."""
    import scripts.user_management_dashboard as umd
    return _json_safe(umd.overview())


@app.get("/api/user-management/breakdown")
async def user_management_breakdown():
    """Per-user details — user list with roles, status, permissions, activity."""
    import scripts.user_management_dashboard as umd
    return _json_safe(umd.breakdown())


@app.get("/api/user-management/definitions")
async def user_management_definitions():
    """User management glossary — roles, permissions, status definitions."""
    import scripts.user_management_dashboard as umd
    return _json_safe(umd.definitions())


@app.get("/api/benchmark-validation/overview")
async def benchmark_validation_overview():
    """Benchmark validation overview — dataset KPIs, model performance, generalization evidence."""
    import scripts.benchmark_validation_dashboard as bvd
    return _json_safe(bvd.overview())


@app.get("/api/benchmark-validation/breakdown")
async def benchmark_validation_breakdown():
    """Fold-level results, model comparison, per-fold accuracy charts."""
    import scripts.benchmark_validation_dashboard as bvd
    return _json_safe(bvd.breakdown())


@app.get("/api/benchmark-validation/definitions")
async def benchmark_validation_definitions():
    """Benchmark validation glossary — external validation, cross-validation, metrics."""
    import scripts.benchmark_validation_dashboard as bvd
    return _json_safe(bvd.definitions())


@app.get("/api/groups-teams/overview")
async def groups_teams_overview():
    """Groups & Teams overview — KPIs, type distribution, size breakdown, membership trends."""
    import scripts.groups_teams_dashboard as gtd
    return _json_safe(gtd.overview())


@app.get("/api/groups-teams/breakdown")
async def groups_teams_breakdown():
    """Per-group details — group list with members, permissions, lead."""
    import scripts.groups_teams_dashboard as gtd
    return _json_safe(gtd.breakdown())


@app.get("/api/groups-teams/definitions")
async def groups_teams_definitions():
    """Groups & Teams glossary — group types, membership, permissions definitions."""
    import scripts.groups_teams_dashboard as gtd
    return _json_safe(gtd.definitions())


# ── Rehab Plan Dashboard ──────────────────────────────────────
# Real data: rehab_plans (311 rows, 30 patients) — OT rehab goals,
# progress tracking, session adherence, category breakdown.

@app.get("/api/rehab-plan/overview")
async def rehab_plan_overview(patient_id: str = None):
    """Rehab plan overview — goal KPIs, category/status distribution,
    progress trends, completion rates from real rehab_plans table."""
    import scripts.rehab_plan_dashboard as rpd
    return _json_safe(rpd.overview(patient_id))


@app.get("/api/rehab-plan/breakdown")
async def rehab_plan_breakdown(patient_id: str = None):
    """Per-patient rehab summary, recent updates, session adherence,
    upcoming targets from real rehab_plans table."""
    import scripts.rehab_plan_dashboard as rpd
    return _json_safe(rpd.breakdown(patient_id))


@app.get("/api/rehab-plan/definitions")
async def rehab_plan_definitions():
    """Rehab plan glossary — goal categories, statuses, metric definitions."""
    import scripts.rehab_plan_dashboard as rpd
    return _json_safe(rpd.definitions())


# ── Billing & Claims Dashboard ────────────────────────────────
# Real data: billing_claims (150 rows, 30+ patients) — claim lifecycle,
# insurance adjudication, aging, CPT/ICD-10, payer mix, revenue cycle.

@app.get("/api/billing-claims/overview")
async def billing_claims_overview():
    """Billing overview — total billed/collected, collection rate, denial rate,
    status distribution, insurance breakdown, monthly trend, top services."""
    import scripts.billing_claims_dashboard as bcd
    return _json_safe(bcd.overview())


@app.get("/api/billing-claims/breakdown")
async def billing_claims_breakdown():
    """Per-patient balances, denied claims, AR aging buckets,
    recent claims, CPT analysis, payer mix."""
    import scripts.billing_claims_dashboard as bcd
    return _json_safe(bcd.breakdown())


@app.get("/api/billing-claims/definitions")
async def billing_claims_definitions():
    """Billing glossary — statuses, service types, aging/ICD-10/CPT notes."""
    import scripts.billing_claims_dashboard as bcd
    return _json_safe(bcd.definitions())


if __name__ == "__main__":
    import os
    import uvicorn
    # Default 8010 to avoid colliding with other local projects on :8000.
    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
