"""Enterprise Pipelines Dashboard — 40-pipeline enterprise AI control-tower catalog
grouped by category (Data/Feature/Training/Evaluation/RAG/Agentic+MCP/Inference/Drift/
Security/ResponsibleAI/GRC/Ops), with stages, status, and maps_to linkage,
from config/enterprise_pipelines.json."""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: total groups, pipelines, stages, status distribution, per-group counts."""
    cfg = _load('enterprise_pipelines.json')
    if not cfg:
        return {"available": False, "note": "enterprise_pipelines.json missing"}

    groups = cfg.get('groups', [])
    total_groups = len(groups)

    all_pipelines = []
    for g in groups:
        for p in g.get('pipelines', []):
            p_copy = dict(p)
            p_copy['group'] = g.get('group', 'Unknown')
            all_pipelines.append(p_copy)

    total_pipelines = len(all_pipelines)
    total_stages = sum(len(p.get('stages', [])) for p in all_pipelines)
    avg_stages = round(total_stages / total_pipelines, 1) if total_pipelines else 0

    status_counts = Counter(p.get('status', 'unknown') for p in all_pipelines)
    built = status_counts.get('built', 0)
    partial = status_counts.get('partial', 0)
    planned = status_counts.get('planned', 0)
    built_pct = round(built / total_pipelines * 100, 1) if total_pipelines else 0

    has_maps_to = sum(1 for p in all_pipelines if p.get('maps_to'))

    status_distribution = [
        {"name": s, "value": c}
        for s, c in [('built', built), ('partial', partial), ('planned', planned)]
        if c > 0
    ]

    pipelines_per_group = []
    for g in groups:
        pipes = g.get('pipelines', [])
        gs = Counter(p.get('status', 'unknown') for p in pipes)
        stages = sum(len(p.get('stages', [])) for p in pipes)
        pipelines_per_group.append({
            "name": g.get('group', ''),
            "value": len(pipes),
            "built": gs.get('built', 0),
            "partial": gs.get('partial', 0),
            "planned": gs.get('planned', 0),
            "total_stages": stages,
        })

    stages_per_group = [
        {"name": g['name'], "value": g['total_stages']}
        for g in pipelines_per_group
    ]

    max_stages_pipeline = max(all_pipelines, key=lambda p: len(p.get('stages', [])))
    min_stages_pipeline = min(all_pipelines, key=lambda p: len(p.get('stages', [])))

    return {
        "available": True,
        "summary": {
            "total_groups": total_groups,
            "total_pipelines": total_pipelines,
            "total_stages": total_stages,
            "avg_stages": avg_stages,
            "built": built,
            "partial": partial,
            "planned": planned,
            "built_pct": built_pct,
            "has_maps_to": has_maps_to,
            "max_stages_pipeline": max_stages_pipeline.get('name', ''),
            "max_stages_count": len(max_stages_pipeline.get('stages', [])),
            "min_stages_pipeline": min_stages_pipeline.get('name', ''),
            "min_stages_count": len(min_stages_pipeline.get('stages', [])),
        },
        "status_distribution": status_distribution,
        "pipelines_per_group": pipelines_per_group,
        "stages_per_group": stages_per_group,
    }


def breakdown():
    """Per-group pipeline details: name, stages, status, maps_to, notes."""
    cfg = _load('enterprise_pipelines.json')
    if not cfg:
        return {"available": False}

    groups = cfg.get('groups', [])

    group_details = []
    flat_table = []

    for g in groups:
        group_name = g.get('group', '')
        pipes = []
        for p in g.get('pipelines', []):
            row = {
                "name": p.get('name', ''),
                "stages": p.get('stages', []),
                "stage_count": len(p.get('stages', [])),
                "status": p.get('status', 'unknown'),
                "maps_to": p.get('maps_to', ''),
                "note": p.get('note', ''),
            }
            pipes.append(row)
            flat_row = dict(row)
            flat_row['group'] = group_name
            flat_table.append(flat_row)
        group_details.append({
            "group": group_name,
            "pipeline_count": len(pipes),
            "pipelines": pipes,
        })

    return {
        "available": True,
        "groups": group_details,
        "flat_table": flat_table,
    }


def definitions():
    """Group descriptions, status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "group_descriptions": [
            {"group": "Data", "description": "End-to-end data lifecycle: acquisition, ingestion, privacy, quality, synthetic generation, lineage, versioning."},
            {"group": "Feature", "description": "Feature engineering, drift monitoring, and embedding pipelines for ML-ready feature stores."},
            {"group": "Training", "description": "Model training, fine-tuning (LoRA/SFT/RLHF), and continuous learning with feedback loops."},
            {"group": "Evaluation", "description": "Model evaluation (accuracy, robustness, bias, explainability) and cross-patient benchmarking."},
            {"group": "RAG", "description": "Retrieval-Augmented Generation: ingest, chunk, embed, search, rerank, generate, cite, hallucination detection."},
            {"group": "Agentic + MCP", "description": "Agentic AI orchestration, multi-agent coordination, MCP governance, and human-in-the-loop."},
            {"group": "Inference", "description": "Production inference: auth, feature retrieval, model selection, prediction, confidence, explainability."},
            {"group": "Drift", "description": "Data drift (PSI/KS), model drift (accuracy/precision/recall), output/RAG drift, and prompt drift monitoring."},
            {"group": "Security & Red Team", "description": "AI security (injection, jailbreak, leakage detection) and adversarial red-team testing."},
            {"group": "Responsible AI & Explainability", "description": "Bias, fairness, explainability (SHAP/LIME/GradCAM), privacy, oversight, and audit."},
            {"group": "Governance, Risk & Compliance", "description": "AI governance, risk management, compliance (regulation/policy mapping), and AI inventory."},
            {"group": "Ops (FinOps / Observability / Incident / Release / Lifecycle)", "description": "Operational pipelines: cost optimization, observability, incident management, change management, release, shadow AI, model retirement, lifecycle management."},
        ],
        "status_legend": [
            {"status": "built", "meaning": "Pipeline implemented and verified — real code exists and endpoints return 200."},
            {"status": "partial", "meaning": "Pipeline partially implemented — some stages exist, others pending."},
            {"status": "planned", "meaning": "Pipeline designed but not yet implemented."},
        ],
        "glossary": [
            {"term": "PSI", "definition": "Population Stability Index — measures distribution shift between training and production data."},
            {"term": "KS Test", "definition": "Kolmogorov-Smirnov test — statistical test for distribution drift detection."},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — game-theoretic feature importance for model explainability."},
            {"term": "LIME", "definition": "Local Interpretable Model-agnostic Explanations — local surrogate model for individual predictions."},
            {"term": "GradCAM", "definition": "Gradient-weighted Class Activation Mapping — CNN visualization for highlighting decision-relevant regions."},
            {"term": "LoRA", "definition": "Low-Rank Adaptation — parameter-efficient fine-tuning method for large models."},
            {"term": "RLHF", "definition": "Reinforcement Learning from Human Feedback — training loop using human preference signals."},
            {"term": "DPO", "definition": "Direct Preference Optimization — simplified alternative to RLHF reward modeling."},
            {"term": "MCP", "definition": "Model Context Protocol — standardized tool/agent communication and governance layer."},
            {"term": "HITL", "definition": "Human-in-the-Loop — clinical expert review, override, and feedback integration."},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — combines vector search with LLM generation for grounded outputs."},
            {"term": "FAR/hr", "definition": "False Alarm Rate per hour — key metric for seizure detection system reliability."},
            {"term": "ICA", "definition": "Independent Component Analysis — artifact removal technique for EEG signal cleaning."},
            {"term": "SMOTE", "definition": "Synthetic Minority Over-sampling Technique — class balancing for imbalanced datasets."},
        ],
        "clinical_notes": [
            "Enterprise pipelines cover the full AI lifecycle from data acquisition through model retirement.",
            "All pipelines enforce clinical governance: HITL review, audit trails, and regulatory compliance.",
            "Drift monitoring operates across data, model, output, and prompt dimensions for continuous safety.",
            "RAG pipelines include hallucination detection and grounding verification for clinical safety.",
        ],
        "references": [
            "IEC 62304 — Medical device software lifecycle processes",
            "FDA AI/ML Software as a Medical Device (SaMD) Action Plan",
            "EU AI Act — High-risk AI system requirements (Art. 9-15)",
            "ISO 14971 — Application of risk management to medical devices",
            "NIST AI Risk Management Framework (AI RMF 1.0)",
        ],
    }
