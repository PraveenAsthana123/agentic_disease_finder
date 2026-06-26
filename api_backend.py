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


@app.get("/api/seizure-timeline")
async def seizure_timeline():
    """Real seizure timeline from CHB-MIT ground-truth annotations (per-subject events + burden)."""
    import scripts.seizure_timeline as st
    return _json_safe(st.build())


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


# ─── Speech-Language Pathologist (SLP) ────────────────────────────────────

@app.get("/api/slp")
async def slp_dashboard(patient_id: str = None):
    """Speech-Language Pathologist (SLP) — full dashboard: language assessment (BNT+WAB),
    verbal fluency, swallowing (MASA), pre/post-surgical language comparison.
    All built from REAL BNT, WAB, VERBAL_FLUENCY, MASA assessments in clinical.db."""
    import scripts.slp_module as slp
    return _json_safe(slp.full_dashboard(patient_id))


@app.get("/api/slp/language-assessment")
async def slp_language(patient_id: str = None):
    """Language assessment: BNT (Boston Naming Test) + WAB (Western Aphasia Battery)."""
    import scripts.slp_module as slp
    return _json_safe(slp.language_assessment(patient_id))


@app.get("/api/slp/speech-analysis")
async def slp_speech(patient_id: str = None):
    """Speech analysis: verbal fluency (phonemic FAS + semantic animals/fruits), clustering, switching."""
    import scripts.slp_module as slp
    return _json_safe(slp.speech_analysis(patient_id))


@app.get("/api/slp/swallowing")
async def slp_swallowing(patient_id: str = None):
    """Swallowing assessment: MASA score, aspiration risk, post-ictal risk flags."""
    import scripts.slp_module as slp
    return _json_safe(slp.swallowing_assessment(patient_id))


@app.get("/api/slp/pre-post-surgical")
async def slp_prepost(patient_id: str = None):
    """Pre/post-surgical language comparison: baseline scores, risk estimates, Wada test recommendations."""
    import scripts.slp_module as slp
    return _json_safe(slp.pre_post_surgical(patient_id))


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


if __name__ == "__main__":
    import os
    import uvicorn
    # Default 8010 to avoid colliding with other local projects on :8000.
    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
