#!/usr/bin/env python3
"""AI-type coverage checklist — insur_project 201 catalog vs THIS project.

Reads the 201 AI-type names pulled from the insurance project and marks each
with its HONEST status in agenticfinder (epilepsy EEG):
  built       = working code exists in this repo
  scaffold    = partial / capture-only (compute or enforcement still missing)
  planned     = named in agent registry, not yet built
  not-pulled  = exists in insur_project, not applicable / not imported here

Writes: config/ai_type_coverage.json + docs/AI_TYPE_COVERAGE.md + docs/PENDING_TASKS.md
Usage:  python scripts/ai_type_coverage.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES_FILE = ROOT / "config" / "insur_ai_types.txt"
OUT_JSON = ROOT / "config" / "ai_type_coverage.json"
OUT_MD = ROOT / "docs" / "AI_TYPE_COVERAGE.md"
PENDING_MD = ROOT / "docs" / "PENDING_TASKS.md"

# Honest curated status for this epilepsy/EEG project.
BUILT = {
    "eeg-ai": "EEG pipeline: parse + 47 features + analysis (eeg_analysis_pipeline.py)",
    "predictive-ai": "Per-disease classifiers (models/*.joblib)",
    "machine-learning": "RandomForest models trained",
    "supervised-learning": "Labeled disease classification",
    "diagnostic-ai": "Epilepsy diagnosis prediction + confidence",
    "clinical-ai": "Clinical capture tables + per-patient analysis",
    "healthcare-ai": "EEG epilepsy clinical workflow",
    "ocr-ai": "pytesseract OCR in ingest (eeg_ingest.extract_image)",
    "document-ai": "PDF/DOCX/text extraction (eeg_ingest)",
    "agentic-ai": "Mixture-of-Agents ensemble (test_agentic_mcp.py)",
    "multi-agent-systems": "7 disease agents + orchestrator (MCP)",
    "audit-ai": "Transaction log (UTC+local) on every write",
    "anomaly-detection-ai": "IQR+z-score anomaly detection on 47 EEG features (scripts/anomaly_detection_dashboard.py, 3 API endpoints, AnomalyDetectionDashboard.jsx)",
}
SCAFFOLD = {
    "explainable-ai": "Ground-truth capture table; SHAP/Grad-CAM compute planned",
    "responsible-ai": "HITL + governance reports; fairness gates not enforced",
    "computer-vision-ai": "Video frame + motion extraction; classification planned",
    "multimodal-ai": "Multi-format ingest (video/pdf/img/edf); fusion model planned",
    "rag": "agentic_rag_analysis/ modules present; live RAG pipeline not wired",
    "agentic-rag": "Analysis module present; not wired to a live corpus",
    "graphrag": "Analysis module present; graph not built",
    "ai-control-tower": "Department reports + transaction log",
    "ai-observability": "Transaction log + audit reports",
    "model-governance": "Consultant matrix + HITL sign-off",
    "governance-ai": "Governance dept + consultant oversight registry",
    "human-evaluation-ai": "HITL accept/override + reason codes",
    "model-monitoring-ai": "Department report KPIs; drift monitor planned",
}
PLANNED = {
    "deep-learning": "CNN/LSTM/Transformer on EEG — agent registry",
    "speech-ai": "Audio conversion from video-EEG — registry",
    "voice-ai": "Audio markers — registry",
    "text-to-audio-ai": "Audio conversion — registry",
    "text-to-video-ai": "Video conversion — registry",
    "image-segmentation-ai": "EEG trace digitization from images — registry",
    "object-detection-ai": "Body-movement detection in video — registry",
    "drift-detection-ai": "Model drift monitoring — registry",
    "fairness-ai": "Fairness gates — registry",
    "bias-detection-ai": "Bias assessment — registry",
    "federated-learning": "Multi-site federation — registry",
    "digital-twin-ai": "Patient digital twin — registry",
    "foundation-models": "Clinical foundation model — registry",
    "llm": "LLM-backed explanation — registry",
    "neuro-ai": "Neuro-specific modeling — registry",
    "knowledge-graph-ai": "Neurophysiology KG — registry",
    "causal-ai": "Causal seizure analysis — registry",
}


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    if not TYPES_FILE.exists():
        print(f"Missing {TYPES_FILE} — copy insur_project ai_types list first.")
        return 1
    types = [t.strip() for t in TYPES_FILE.read_text().splitlines() if t.strip()]

    rows = []
    for t in types:
        if t in BUILT:
            rows.append({"type": t, "status": "built", "note": BUILT[t]})
        elif t in SCAFFOLD:
            rows.append({"type": t, "status": "scaffold", "note": SCAFFOLD[t]})
        elif t in PLANNED:
            rows.append({"type": t, "status": "planned", "note": PLANNED[t]})
        else:
            rows.append({"type": t, "status": "not-pulled", "note": "in insur_project; not applicable / not imported"})

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    payload = {"generated_at": _now(), "source": "insur_project (201 AI types)",
               "project": "agenticfinder (epilepsy EEG)", "total": len(rows),
               "counts": counts, "rows": rows}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# AI-Type Coverage — insur_project (201) vs agenticfinder (epilepsy)",
        "", f"_Generated {payload['generated_at']}_", "",
        f"**Totals:** built {counts.get('built',0)} · scaffold {counts.get('scaffold',0)} · "
        f"planned {counts.get('planned',0)} · not-pulled {counts.get('not-pulled',0)} (of {len(rows)})",
        "",
    ]
    for status in ["built", "scaffold", "planned"]:
        md.append(f"## {status.title()} ({counts.get(status,0)})")
        md.append("| AI Type | Note |")
        md.append("|---|---|")
        for r in rows:
            if r["status"] == status:
                md.append(f"| {r['type']} | {r['note']} |")
        md.append("")
    md.append(f"## Not-pulled ({counts.get('not-pulled',0)})")
    md.append("_Available in insur_project, not applicable to the epilepsy DBA scope:_")
    md.append("")
    md.append(", ".join(r["type"] for r in rows if r["status"] == "not-pulled"))
    md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"[ai_type_coverage] {payload['generated_at']}")
    print(f"  total={len(rows)} | " + " · ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"  -> {OUT_MD}")
    print(f"  -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
