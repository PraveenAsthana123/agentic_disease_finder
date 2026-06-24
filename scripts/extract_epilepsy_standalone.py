#!/usr/bin/env python3
"""Extract a STANDALONE epilepsy-only project into a sibling directory, ready to
become its own GitHub repo. Excludes other diseases + the 9.4GB raw EDFs (keeps
seizure-annotation summaries + small samples + a download script).

Usage: python scripts/extract_epilepsy_standalone.py [--target ../agentic_epilepsy_finder]
The result is a self-contained epilepsy app: backend + frontend + epilepsy
scripts + epilepsy config + governance/explainability + standalone README.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Shared infra (disease-agnostic — epilepsy uses these). Copied as-is.
SHARED_PY = [
    "api_backend.py", "clinical_db.py", "eeg_analysis_pipeline.py", "eeg_deep.py",
    "eeg_anomaly.py", "eeg_datascience.py", "eeg_timeseries_stats.py", "eeg_ingest.py",
    "eeg_explainability.py", "rai_checks.py", "ollama_agent.py", "ai_type_detail.py",
    "council_orchestrator.py", "ui_app.py",
]
# Epilepsy-specific scripts.
EPI_SCRIPTS = [
    "accuracy_all_options.py", "accuracy_patient_specific.py", "bonn_external_validation.py",
    "bootstrap_ci_baselines.py", "concordance_analysis.py", "cross_patient_benchmark.py",
    "ica_noise_cleaning.py", "generate_synthetic_epilepsy.py", "run_validation_suite.py",
    "vector_ingest.py", "install_vector_cron.sh", "install_validation_cron.sh",
]
CONFIG_ALL = True  # all config/*.json are epilepsy-oriented now


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(ROOT.parent / "agentic_epilepsy_finder"))
    args = ap.parse_args()
    T = Path(args.target)
    print(f"Extracting standalone epilepsy project -> {T}")
    T.mkdir(parents=True, exist_ok=True)

    # 1. shared infra + epilepsy scripts
    for f in SHARED_PY:
        copy(ROOT / f, T / f)
    for f in EPI_SCRIPTS:
        copy(ROOT / "scripts" / f, T / "scripts" / f)

    # 2. config (all epilepsy-oriented)
    copy(ROOT / "config", T / "config")

    # 3. frontend (epilepsy-focused UI)
    if (ROOT / "frontend" / "src").exists():
        copy(ROOT / "frontend" / "src", T / "frontend" / "src")
        for f in ["package.json", "vite.config.js", "index.html"]:
            copy(ROOT / "frontend" / f, T / "frontend" / f)

    # 4. epilepsy data — SMALL only (samples + seizure annotations, NOT 9.4GB EDFs)
    copy(ROOT / "data" / "epilepsy", T / "data" / "epilepsy")
    chb = ROOT / "data" / "real_eeg" / "epilepsy_physionet"
    if chb.exists():
        for summ in chb.rglob("*-summary.txt"):
            copy(summ, T / "data" / "real_eeg" / "epilepsy_physionet" / summ.parent.name / summ.name)
    bonn = ROOT / "data" / "external_validation" / "epilepsy_bonn"
    if bonn.exists():
        copy(bonn, T / "data" / "external_validation" / "epilepsy_bonn")

    # 5. docs + reports (epilepsy)
    for d in ["docs"]:
        if (ROOT / d).exists():
            copy(ROOT / d, T / d)
    rep = T / "jobs" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    for r in (ROOT / "jobs" / "reports").glob("*.json") if (ROOT / "jobs" / "reports").exists() else []:
        if any(k in r.name for k in ["accuracy", "bonn", "ica", "bootstrap", "concordance", "VALIDATION"]):
            copy(r, rep / r.name)

    # 6. standalone scaffolding
    (T / ".gitignore").write_text(
        "*.edf\n*.pkl\n__pycache__/\nnode_modules/\ndata/real_eeg/epilepsy_physionet/**/*.edf\n"
        "data/vector_db/\n*.db\n*.db-shm\n*.db-wal\njobs/logs/\n.env\n", encoding="utf-8")
    (T / "requirements.txt").write_text(
        "fastapi\nuvicorn\npydantic\nnumpy\nscipy\nscikit-learn\nxgboost\nlightgbm\n"
        "mne\nstatsmodels\nshap\nchromadb\npandas\nstreamlit\n", encoding="utf-8")
    (T / "README.md").write_text(
        "# Agentic Epilepsy Finder (standalone)\n\n"
        "Epilepsy-only EEG AI with Responsible-AI governance under human clinical oversight.\n"
        "Extracted from the multi-disease parent for disease isolation.\n\n"
        "## Headline result (honest, leakage-free)\n"
        "- Patient-specific: **0.98** [0.973-0.987]\n"
        "- Cross-patient: **0.73** [0.40-0.93]\n\n"
        "## Run\n```\npip install -r requirements.txt\npython api_backend.py          # backend :8010\n"
        "cd frontend && npm install && npm run dev\n```\n\n"
        "## CHB-MIT data (not committed — 9.4GB)\n"
        "Download EDFs into `data/real_eeg/epilepsy_physionet/chbXX/` (summaries included).\n"
        "Benchmarks auto-discover subjects. Then:\n```\npython scripts/run_validation_suite.py\n```\n\n"
        "## What's inside\nEEG pipeline · SHAP/interpretable/responsible AI · Council of Agents ·\n"
        "multi-expert study review · assessments (MoCA/PHQ-9/etc.) · vector RAG · governance audit.\n",
        encoding="utf-8")

    # count
    nfiles = sum(1 for _ in T.rglob("*") if _.is_file())
    print(f"Done. {nfiles} files in {T}")
    print("Next (you run):")
    print(f"  cd {T} && git init && git add -A && git commit -m 'init: standalone epilepsy'")
    print(f"  gh repo create agentic_epilepsy_finder --private --source . --push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
