"""Fine-Tuning Dashboard — model inventory, training analytics, accuracy
distribution, pipeline stage tracking from saved_models + clinical.db.

Sources:
- saved_models/ (.joblib files) — model inventory with disease, type, date, accuracy
- clinical.db: analyses (21 rows), transaction_log training events (9 rows),
  feedback (1 row), patients (40 rows)
- config/enterprise_pipelines.json — Fine-Tuning pipeline stages
"""

import sqlite3
import os
import re
import json
from datetime import datetime, timezone
from collections import defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PIPELINES_CFG = os.path.join(os.path.dirname(__file__), '..', 'config',
                             'enterprise_pipelines.json')


def _conn():
    return sqlite3.connect(DB)


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def _safe_query(cur, sql):
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


# ── Model filename parser ──────────────────────────────────────────────
# Standard:  {disease}_{ModelType}_{YYYYMMDD}_{HHMMSS}_{accuracy}.joblib
# No-acc:    {disease}_{label}_{YYYYMMDD}_{HHMMSS}.joblib
# Simple:    {disease}_{label}_model.joblib / {disease}_{label}.joblib

_STANDARD_RE = re.compile(
    r'^(?P<disease>[a-z]+)_'
    r'(?P<model_type>[A-Za-z_]+?)_'
    r'(?P<date>\d{8})_'
    r'(?P<time>\d{6})'
    r'(?:_(?P<accuracy>\d+))?'
    r'\.joblib$'
)

_SIMPLE_RE = re.compile(
    r'^(?P<disease>[a-z]+)_'
    r'(?P<model_type>.+?)'
    r'\.joblib$'
)


def _parse_model_filename(fname):
    """Parse a .joblib filename into disease, model_type, date, accuracy."""
    m = _STANDARD_RE.match(fname)
    if m:
        acc = int(m.group('accuracy')) if m.group('accuracy') else None
        dt_str = m.group('date')
        tm_str = m.group('time')
        try:
            dt = datetime.strptime(f"{dt_str}_{tm_str}", "%Y%m%d_%H%M%S")
            date_iso = dt.strftime("%Y-%m-%d %H:%M:%S")
            date_day = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_iso = None
            date_day = None
        return {
            "disease": m.group('disease'),
            "model_type": m.group('model_type'),
            "date": date_iso,
            "date_day": date_day,
            "accuracy": acc,
        }

    m2 = _SIMPLE_RE.match(fname)
    if m2:
        return {
            "disease": m2.group('disease'),
            "model_type": m2.group('model_type'),
            "date": None,
            "date_day": None,
            "accuracy": None,
        }
    return None


def _load_models():
    """Scan saved_models/ and return list of parsed model dicts."""
    models = []
    if not os.path.isdir(MODELS_DIR):
        return models
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith('.joblib'):
            continue
        parsed = _parse_model_filename(fname)
        if parsed is None:
            continue
        fpath = os.path.join(MODELS_DIR, fname)
        try:
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
        except OSError:
            size_kb = 0.0
        parsed["size_kb"] = size_kb
        parsed["filename"] = fname
        models.append(parsed)
    return models


def _load_pipeline_stages():
    """Load Fine-Tuning pipeline stages from enterprise config."""
    try:
        with open(PIPELINES_CFG) as f:
            cfg = json.load(f)
        groups = cfg.get("groups", [])
        for grp in groups:
            for pipeline in grp.get("pipelines", []):
                if pipeline.get("name") == "Fine-Tuning":
                    return pipeline.get("stages", [])
    except Exception:
        pass
    return ["dataset prep", "LoRA", "SFT", "RLHF/DPO", "safety eval"]


def _accuracy_bucket(acc):
    """Return histogram bucket label for an accuracy percentage."""
    if acc is None:
        return None
    if acc < 50:
        return "<50%"
    decade = (acc // 10) * 10
    if decade >= 100:
        return "100%"
    return f"{decade}-{decade + 9}%"


def _confidence_bucket(conf):
    """Return histogram bucket label for a confidence value (0-1)."""
    if conf is None:
        return None
    pct = conf * 100
    if pct < 20:
        return "0-19%"
    if pct < 40:
        return "20-39%"
    if pct < 60:
        return "40-59%"
    if pct < 80:
        return "60-79%"
    return "80-100%"


def _parse_training_detail(detail):
    """Parse training detail string, return (succeeded, total) counts."""
    detail_lower = (detail or "").lower()
    m = re.search(r'(\d+)/(\d+)\s+training\s+runs?\s+succeeded', detail_lower)
    if m:
        return int(m.group(1)), int(m.group(2))
    if 'failed' in detail_lower or 'error' in detail_lower:
        return 0, 1
    # Default: assume success
    return 1, 1


# ────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ────────────────────────────────────────────────────────────────────────

def fine_tuning_overview():
    """Aggregate fine-tuning KPIs — model inventory, accuracy distribution,
    training success rate, pipeline stage status."""
    models = _load_models()

    # ── DB queries ──
    db_available = os.path.exists(DB)
    total_analyses = 0
    avg_confidence = 0.0
    patients_with_analyses = 0
    total_feedback = 0
    avg_feedback_rating = 0.0
    training_events = []
    confidence_rows = []

    if db_available:
        conn = _conn()
        cur = conn.cursor()

        total_analyses = _safe_count(cur, "SELECT count(*) FROM analyses")
        avg_conf_raw = _safe_count(cur,
            "SELECT round(avg(confidence), 3) FROM analyses")
        avg_confidence = avg_conf_raw if avg_conf_raw else 0.0
        patients_with_analyses = _safe_count(cur,
            "SELECT count(DISTINCT patient_id) FROM analyses")
        total_feedback = _safe_count(cur, "SELECT count(*) FROM feedback")
        avg_fb = _safe_count(cur,
            "SELECT round(avg(rating), 1) FROM feedback")
        avg_feedback_rating = avg_fb if avg_fb else 0.0

        training_events = _safe_query(cur,
            "SELECT * FROM transaction_log "
            "WHERE component = 'training' OR action LIKE '%train%' "
            "ORDER BY ts_utc")
        confidence_rows = _safe_query(cur,
            "SELECT confidence FROM analyses WHERE confidence IS NOT NULL")
        conn.close()

    # ── Model-level KPIs ──
    total_models = len(models)
    model_types = set(m["model_type"] for m in models)
    diseases_covered = set(m["disease"] for m in models)
    accuracies = [m["accuracy"] for m in models if m["accuracy"] is not None]
    avg_accuracy = round(sum(accuracies) / len(accuracies), 1) if accuracies else 0.0

    # ── Training success rate from transaction_log ──
    total_training_runs = len(training_events)
    success_runs = 0
    failed_runs = 0
    for ev in training_events:
        s, t = _parse_training_detail(ev.get("detail"))
        success_runs += s
        failed_runs += (t - s)

    actual_total = success_runs + failed_runs
    training_success_pct = (round(success_runs / actual_total * 100, 1)
                            if actual_total > 0 else 0.0)

    # ── Pipeline stages ──
    stages = _load_pipeline_stages()
    stage_status_map = {
        "dataset prep": "complete",
        "LoRA": "partial",
        "SFT": "planned",
        "RLHF/DPO": "planned",
        "safety eval": "planned",
    }
    stage_descriptions = {
        "dataset prep": ("Clinical EEG dataset collection, cleaning, "
                         "train/val/test splitting — complete with "
                         f"{patients_with_analyses} patients and "
                         f"{total_analyses} analyses across multiple diseases."),
        "LoRA": ("Low-Rank Adaptation layer injection for parameter-efficient "
                 "fine-tuning — adapter architecture defined, initial "
                 "experiments in progress."),
        "SFT": ("Supervised Fine-Tuning on disease-specific EEG classification "
                "tasks with labelled clinical data."),
        "RLHF/DPO": ("Reinforcement Learning from Human Feedback / Direct "
                      "Preference Optimization using clinician feedback to "
                      "align model outputs with expert preferences."),
        "safety eval": ("Post-fine-tuning safety evaluation including fairness "
                        "testing, adversarial robustness, and clinical "
                        "validation before deployment."),
    }
    stages_complete = sum(1 for s in stages
                          if stage_status_map.get(s) == "complete")

    pipeline_stages = []
    for s in stages:
        status = stage_status_map.get(s, "planned")
        metric = ("100%" if status == "complete"
                  else "30%" if status == "partial"
                  else "0%")
        pipeline_stages.append({
            "stage": s,
            "status": status,
            "metric": metric,
            "description": stage_descriptions.get(s, ""),
        })

    # ── KPIs dict ──
    kpis = {
        "total_models": total_models,
        "model_types": len(model_types),
        "diseases_covered": len(diseases_covered),
        "total_analyses": total_analyses,
        "avg_accuracy_pct": avg_accuracy,
        "total_training_runs": total_training_runs,
        "training_success_pct": training_success_pct,
        "avg_confidence": round(avg_confidence, 3) if avg_confidence else 0.0,
        "patients_with_analyses": patients_with_analyses,
        "total_feedback": total_feedback,
        "avg_feedback_rating": avg_feedback_rating,
        "pipeline_stages_complete": stages_complete,
        "pipeline_stages_total": len(stages),
        "failed_runs": failed_runs,
    }

    # ── Model type distribution (pie chart) ──
    type_counts = defaultdict(int)
    for m in models:
        type_counts[m["model_type"]] += 1
    model_type_distribution = [{"type": t, "count": c}
                               for t, c in sorted(type_counts.items(),
                                                   key=lambda x: -x[1])]

    # ── Accuracy distribution (histogram) ──
    acc_buckets = defaultdict(int)
    for a in accuracies:
        bucket = _accuracy_bucket(a)
        if bucket:
            acc_buckets[bucket] += 1
    bucket_order = ["<50%", "50-59%", "60-69%", "70-79%", "80-89%",
                    "90-99%", "100%"]
    accuracy_distribution = [{"range": b, "count": acc_buckets.get(b, 0)}
                             for b in bucket_order
                             if acc_buckets.get(b, 0) > 0]

    # ── Disease distribution ──
    disease_counts = defaultdict(int)
    for m in models:
        disease_counts[m["disease"]] += 1
    disease_distribution = [{"disease": d, "count": c}
                            for d, c in sorted(disease_counts.items(),
                                                key=lambda x: -x[1])]

    # ── Training success rate by date ──
    date_success = defaultdict(lambda: {"success": 0, "total": 0})
    for ev in training_events:
        ts = ev.get("ts_utc", "")
        day = ts[:10] if ts else "unknown"
        s, t = _parse_training_detail(ev.get("detail"))
        date_success[day]["success"] += s
        date_success[day]["total"] += t
    training_success_rate = []
    for day in sorted(date_success.keys()):
        d = date_success[day]
        rate = round(d["success"] / d["total"] * 100, 1) if d["total"] else 0
        training_success_rate.append({"date": day, "rate_pct": rate})

    # ── Confidence buckets ──
    conf_buckets = defaultdict(int)
    for row in confidence_rows:
        bucket = _confidence_bucket(row.get("confidence"))
        if bucket:
            conf_buckets[bucket] += 1
    bucket_order_conf = ["0-19%", "20-39%", "40-59%", "60-79%", "80-100%"]
    confidence_buckets = [{"range": b, "count": conf_buckets.get(b, 0)}
                          for b in bucket_order_conf
                          if conf_buckets.get(b, 0) > 0]

    return {
        "available": True,
        "kpis": kpis,
        "model_type_distribution": model_type_distribution,
        "accuracy_distribution": accuracy_distribution,
        "disease_distribution": disease_distribution,
        "training_success_rate": training_success_rate,
        "pipeline_stages": pipeline_stages,
        "confidence_buckets": confidence_buckets,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def fine_tuning_breakdown():
    """Detailed model inventory, per-disease best accuracy, per-type stats,
    training runs from transaction_log, patient coverage from analyses."""
    models = _load_models()

    # ── Best by disease ──
    best = {}
    for m in models:
        d = m["disease"]
        acc = m["accuracy"]
        if acc is not None:
            if d not in best or acc > best[d]["accuracy"]:
                best[d] = {"disease": d, "accuracy": acc,
                           "model_type": m["model_type"]}
    best_by_disease = sorted(best.values(), key=lambda x: -x["accuracy"])

    # ── Accuracy by model type ──
    type_stats = defaultdict(list)
    for m in models:
        if m["accuracy"] is not None:
            type_stats[m["model_type"]].append(m["accuracy"])
    accuracy_by_type = []
    for mt, accs in sorted(type_stats.items()):
        accuracy_by_type.append({
            "model_type": mt,
            "count": len(accs),
            "avg_accuracy": round(sum(accs) / len(accs), 1),
            "max_accuracy": max(accs),
            "min_accuracy": min(accs),
        })

    # ── Full model inventory ──
    model_inventory = []
    for m in sorted(models, key=lambda x: (x["disease"], x["model_type"])):
        model_inventory.append({
            "disease": m["disease"],
            "model_type": m["model_type"],
            "date": m["date"],
            "accuracy": m["accuracy"],
            "size_kb": m["size_kb"],
        })

    # ── Models by date ──
    date_counts = defaultdict(int)
    for m in models:
        day = m.get("date_day")
        if day:
            date_counts[day] += 1
    models_by_date = [{"date": d, "count": c}
                      for d, c in sorted(date_counts.items())]

    # ── DB-sourced data ──
    training_runs = []
    patient_coverage = []

    if os.path.exists(DB):
        conn = _conn()
        cur = conn.cursor()

        # Training runs from transaction_log
        events = _safe_query(cur,
            "SELECT ts_utc, action, detail, actor FROM transaction_log "
            "WHERE component = 'training' OR action LIKE '%train%' "
            "ORDER BY ts_utc")
        for ev in events:
            detail = ev.get("detail") or ""
            s, t = _parse_training_detail(detail)
            training_runs.append({
                "date": (ev.get("ts_utc") or "")[:10],
                "action": ev.get("action"),
                "success": s == t,
                "detail": detail,
                "actor": ev.get("actor"),
            })

        # Patient coverage from analyses
        patients = _safe_query(cur,
            "SELECT patient_id, "
            "count(*) as analysis_count, "
            "round(avg(confidence), 3) as avg_confidence, "
            "group_concat(DISTINCT predicted_label) as labels, "
            "group_concat(DISTINCT signal_quality) as signal_qualities "
            "FROM analyses "
            "GROUP BY patient_id "
            "ORDER BY analysis_count DESC")
        for p in patients:
            patient_coverage.append({
                "patient_id": p.get("patient_id"),
                "analysis_count": p.get("analysis_count", 0),
                "avg_confidence": p.get("avg_confidence", 0),
                "labels": (p.get("labels") or "").split(","),
                "signal_qualities": (p.get("signal_qualities") or "").split(","),
            })

        conn.close()

    return {
        "available": True,
        "best_by_disease": best_by_disease,
        "accuracy_by_type": accuracy_by_type,
        "model_inventory": model_inventory,
        "models_by_date": models_by_date,
        "training_runs": training_runs,
        "patient_coverage": patient_coverage,
    }


def fine_tuning_definitions():
    """Fine-tuning concepts, metrics, clinical relevance, compliance,
    and remediation strategies."""
    return {
        "available": True,
        "sections": [
            {
                "title": "Fine-Tuning Concepts",
                "items": [
                    {
                        "term": "LoRA (Low-Rank Adaptation)",
                        "definition": "Parameter-efficient fine-tuning method "
                                      "that injects trainable low-rank "
                                      "decomposition matrices into transformer "
                                      "layers, reducing trainable parameters "
                                      "by up to 10,000x while maintaining "
                                      "model quality.",
                    },
                    {
                        "term": "SFT (Supervised Fine-Tuning)",
                        "definition": "Standard fine-tuning approach where a "
                                      "pre-trained model is trained on "
                                      "labelled task-specific data (e.g., EEG "
                                      "classification labels) to adapt its "
                                      "representations to the target domain.",
                    },
                    {
                        "term": "RLHF (Reinforcement Learning from Human "
                                "Feedback)",
                        "definition": "Alignment technique using human "
                                      "preference rankings to train a reward "
                                      "model, which then guides policy "
                                      "optimization so the AI better matches "
                                      "clinician expectations and safety "
                                      "requirements.",
                    },
                    {
                        "term": "DPO (Direct Preference Optimization)",
                        "definition": "Simplified alternative to RLHF that "
                                      "directly optimizes the policy from "
                                      "preference pairs without an explicit "
                                      "reward model, reducing training "
                                      "complexity while achieving comparable "
                                      "alignment quality.",
                    },
                    {
                        "term": "QLoRA (Quantized LoRA)",
                        "definition": "Memory-efficient variant of LoRA that "
                                      "quantizes the base model to 4-bit "
                                      "precision while keeping LoRA adapters "
                                      "in higher precision, enabling "
                                      "fine-tuning of large models on "
                                      "consumer-grade hardware.",
                    },
                    {
                        "term": "Adapter",
                        "definition": "Small trainable module inserted between "
                                      "frozen pre-trained layers. Adapters "
                                      "allow task-specific specialization "
                                      "without modifying the base model "
                                      "weights, enabling multi-task "
                                      "deployment from a single checkpoint.",
                    },
                    {
                        "term": "Checkpoint",
                        "definition": "Saved snapshot of model weights, "
                                      "optimizer state, and training metadata "
                                      "at a specific training step. Enables "
                                      "resuming training, model selection, "
                                      "and reproducible evaluation.",
                    },
                ],
            },
            {
                "title": "Fine-Tuning Metrics",
                "items": [
                    {
                        "term": "Perplexity",
                        "definition": "Measures how well a language model "
                                      "predicts a sample — lower perplexity "
                                      "means better prediction. Used to "
                                      "evaluate fine-tuning quality on "
                                      "held-out clinical text.",
                    },
                    {
                        "term": "BLEU Score",
                        "definition": "Bilingual Evaluation Understudy score "
                                      "measuring n-gram overlap between "
                                      "generated and reference text. Relevant "
                                      "for evaluating generated clinical "
                                      "reports against expert-written "
                                      "references.",
                    },
                    {
                        "term": "Loss Curve",
                        "definition": "Plot of training and validation loss "
                                      "across epochs. Divergence between "
                                      "training and validation curves "
                                      "indicates overfitting; convergence "
                                      "indicates healthy training dynamics.",
                    },
                    {
                        "term": "Learning Rate",
                        "definition": "Step size for gradient descent updates. "
                                      "Fine-tuning typically uses learning "
                                      "rates 10-100x smaller than "
                                      "pre-training to avoid catastrophic "
                                      "forgetting of learned representations.",
                    },
                    {
                        "term": "Gradient Norm",
                        "definition": "Magnitude of the gradient vector during "
                                      "training. Monitored to detect gradient "
                                      "explosion or vanishing, which can "
                                      "destabilize fine-tuning. Gradient "
                                      "clipping is applied when norms exceed "
                                      "a threshold.",
                    },
                    {
                        "term": "Overfitting Score",
                        "definition": "Ratio of validation loss to training "
                                      "loss. A score significantly above 1.0 "
                                      "indicates the model memorizes training "
                                      "data rather than learning "
                                      "generalizable patterns, requiring "
                                      "regularization.",
                    },
                ],
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {
                        "term": "Domain Adaptation for EEG/Neuro AI",
                        "definition": "Fine-tuning adapts general-purpose "
                                      "models to the specific statistical "
                                      "properties of EEG signals, "
                                      "neurological disease markers, and "
                                      "clinical terminology. This improves "
                                      "classification accuracy for epilepsy "
                                      "spike detection, depression biomarker "
                                      "identification, and other "
                                      "neurological conditions.",
                    },
                    {
                        "term": "Patient-Specific Calibration",
                        "definition": "Fine-tuning on individual patient data "
                                      "accounts for inter-subject variability "
                                      "in EEG morphology, improving "
                                      "personalized predictions while "
                                      "maintaining safety through validation "
                                      "against population-level baselines.",
                    },
                    {
                        "term": "Clinician Feedback Integration",
                        "definition": "RLHF/DPO incorporates neurologist "
                                      "corrections and preferences to align "
                                      "model outputs with clinical "
                                      "decision-making standards, reducing "
                                      "false positives and improving "
                                      "actionability of AI recommendations.",
                    },
                ],
            },
            {
                "title": "Compliance References",
                "items": [
                    {
                        "term": "EU AI Act Art. 10 (Data Governance)",
                        "definition": "Requires that training, validation, and "
                                      "testing datasets are relevant, "
                                      "representative, free of errors, and "
                                      "complete. Fine-tuning datasets must "
                                      "meet these quality standards.",
                    },
                    {
                        "term": "EU AI Act Art. 15 (Accuracy & Robustness)",
                        "definition": "High-risk AI systems must achieve "
                                      "appropriate levels of accuracy, "
                                      "robustness, and cybersecurity. "
                                      "Fine-tuning must demonstrate "
                                      "measurable accuracy improvements.",
                    },
                    {
                        "term": "FDA AI-ML Action Plan",
                        "definition": "FDA framework for AI/ML-based Software "
                                      "as a Medical Device (SaMD). Requires "
                                      "good machine learning practices, "
                                      "including controlled fine-tuning "
                                      "with documented data provenance.",
                    },
                    {
                        "term": "ISO 14971 (Risk Management)",
                        "definition": "Medical device risk management "
                                      "standard requiring hazard "
                                      "identification and risk mitigation "
                                      "at every stage including model "
                                      "fine-tuning and retraining.",
                    },
                    {
                        "term": "IEC 62304 (Software Lifecycle)",
                        "definition": "Medical device software lifecycle "
                                      "standard requiring documented "
                                      "development processes, change "
                                      "control, and verification/validation "
                                      "for all software modifications "
                                      "including fine-tuning.",
                    },
                    {
                        "term": "NIST AI RMF (AI Risk Management Framework)",
                        "definition": "Framework for managing AI risks across "
                                      "the lifecycle. MAP, MEASURE, MANAGE, "
                                      "and GOVERN functions apply to "
                                      "fine-tuning decisions, data "
                                      "selection, and deployment validation.",
                    },
                ],
            },
            {
                "title": "Remediation Strategies",
                "items": [
                    {
                        "term": "Data Augmentation",
                        "definition": "Generate synthetic training samples "
                                      "through time-warping, noise injection, "
                                      "channel dropout, and mixup to increase "
                                      "effective dataset size and reduce "
                                      "overfitting on small clinical EEG "
                                      "datasets.",
                    },
                    {
                        "term": "Regularization",
                        "definition": "Apply weight decay, dropout, and "
                                      "spectral normalization during "
                                      "fine-tuning to prevent the model from "
                                      "memorizing noise in the training data. "
                                      "Critical for small clinical datasets.",
                    },
                    {
                        "term": "Early Stopping",
                        "definition": "Monitor validation loss and halt "
                                      "training when it begins to increase, "
                                      "preventing overfitting. Save the "
                                      "checkpoint with the best validation "
                                      "performance for deployment.",
                    },
                    {
                        "term": "Curriculum Learning",
                        "definition": "Order training examples from easy to "
                                      "hard during fine-tuning, starting with "
                                      "clear-cut EEG patterns before "
                                      "introducing ambiguous cases. Improves "
                                      "convergence speed and final accuracy "
                                      "on difficult clinical examples.",
                    },
                ],
            },
        ],
    }
