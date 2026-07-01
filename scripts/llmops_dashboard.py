"""
LLMOps Dashboard
Prompt versioning, token/cost tracking, latency monitoring,
hallucination detection, and RAG evaluation metrics.

Registry item: llmops (admin_module.ops_dashboards)
Purpose: prompt versions, token/cost, latency, hallucination, RAG eval
"""

import hashlib
import json
import os
import time

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Reference data — models, prompts, RAG pipelines
# ---------------------------------------------------------------------------

MODELS = [
    {"id": "gpt-4o",           "label": "GPT-4o",             "provider": "OpenAI",    "cost_per_1k_input": 0.005,  "cost_per_1k_output": 0.015},
    {"id": "claude-sonnet",    "label": "Claude Sonnet 4",    "provider": "Anthropic", "cost_per_1k_input": 0.003,  "cost_per_1k_output": 0.015},
    {"id": "ollama-llama3",    "label": "Llama 3 (Ollama)",   "provider": "Local",     "cost_per_1k_input": 0.0,    "cost_per_1k_output": 0.0},
    {"id": "ollama-mistral",   "label": "Mistral 7B (Ollama)","provider": "Local",     "cost_per_1k_input": 0.0,    "cost_per_1k_output": 0.0},
]

PROMPT_TEMPLATES = [
    {"id": "eeg-classify",     "label": "EEG Classification Prompt",     "category": "clinical",     "model_id": "claude-sonnet"},
    {"id": "report-gen",       "label": "Clinical Report Generator",     "category": "clinical",     "model_id": "gpt-4o"},
    {"id": "rag-query",        "label": "RAG Retrieval Query",           "category": "retrieval",    "model_id": "ollama-llama3"},
    {"id": "patient-summary",  "label": "Patient Summary Prompt",        "category": "clinical",     "model_id": "claude-sonnet"},
    {"id": "drug-interaction", "label": "Drug Interaction Checker",      "category": "clinical",     "model_id": "gpt-4o"},
    {"id": "council-vote",     "label": "AI Council Vote Prompt",        "category": "governance",   "model_id": "claude-sonnet"},
    {"id": "anomaly-explain",  "label": "EEG Anomaly Explanation",       "category": "explainability","model_id": "ollama-mistral"},
    {"id": "intake-parse",     "label": "Patient Intake Parser",         "category": "data-entry",   "model_id": "ollama-llama3"},
]

RAG_PIPELINES = [
    {"id": "clinical-rag",     "label": "Clinical Knowledge RAG",   "vector_db": "ChromaDB", "chunk_size": 512,  "overlap": 50},
    {"id": "drug-rag",         "label": "Drug Reference RAG",       "vector_db": "ChromaDB", "chunk_size": 256,  "overlap": 32},
    {"id": "guideline-rag",    "label": "ILAE Guidelines RAG",      "vector_db": "ChromaDB", "chunk_size": 1024, "overlap": 100},
]

HALLUCINATION_CATEGORIES = [
    "factual_error", "unsupported_claim", "contradicts_source",
    "fabricated_reference", "numerical_error", "temporal_confusion",
]

# Latency thresholds (milliseconds)
LATENCY_THRESHOLDS = {
    "p50_target_ms": 800,
    "p95_target_ms": 3000,
    "p99_target_ms": 8000,
    "timeout_ms":    30000,
}

COST_BUDGET_MONTHLY_USD = 150.0

# ---------------------------------------------------------------------------
# Deterministic seed helper
# ---------------------------------------------------------------------------

def _seed(key: str) -> float:
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


# ---------------------------------------------------------------------------
# Data generation — deterministic from model/prompt/date
# ---------------------------------------------------------------------------

def _generate_prompt_versions():
    """Prompt template version history with metrics."""
    results = []
    for pt in PROMPT_TEMPLATES:
        n_versions = max(2, round(_lerp(2, 7, _seed(f"pv:{pt['id']}:count"))))
        versions = []
        for v in range(1, n_versions + 1):
            s = _seed(f"pv:{pt['id']}:v{v}")
            tokens_avg = round(_lerp(150, 2500, _seed(f"pv:{pt['id']}:v{v}:tokens")))
            accuracy = round(_lerp(0.70, 0.98, _seed(f"pv:{pt['id']}:v{v}:acc")), 3)
            # Later versions tend to be better
            accuracy = min(0.99, round(accuracy + v * 0.015, 3))
            halluc_rate = round(max(0.0, _lerp(0.01, 0.12, _seed(f"pv:{pt['id']}:v{v}:halluc")) - v * 0.008), 4)
            latency_ms = round(_lerp(200, 5000, _seed(f"pv:{pt['id']}:v{v}:lat")))
            versions.append({
                "version": v,
                "avg_tokens": tokens_avg,
                "accuracy": accuracy,
                "hallucination_rate": halluc_rate,
                "avg_latency_ms": latency_ms,
                "status": "active" if v == n_versions else "archived",
                "created_days_ago": (n_versions - v) * round(_lerp(3, 14, _seed(f"pv:{pt['id']}:v{v}:age"))),
            })
        current = versions[-1]
        results.append({
            "prompt_id": pt["id"],
            "label": pt["label"],
            "category": pt["category"],
            "model_id": pt["model_id"],
            "total_versions": n_versions,
            "current_version": n_versions,
            "current_accuracy": current["accuracy"],
            "current_halluc_rate": current["hallucination_rate"],
            "current_latency_ms": current["avg_latency_ms"],
            "versions": versions,
        })
    return results


def _generate_token_cost_tracking():
    """Daily token usage and cost for last 30 days, per model."""
    daily = []
    model_totals = {m["id"]: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "requests": 0} for m in MODELS}

    for day in range(30):
        day_data = {"day_offset": day, "models": {}}
        for m in MODELS:
            s = _seed(f"tokens:{m['id']}:day:{day}")
            requests = round(_lerp(5, 200, s))
            input_tokens = round(_lerp(500, 15000, _seed(f"tokens:{m['id']}:day:{day}:in")))
            output_tokens = round(_lerp(200, 8000, _seed(f"tokens:{m['id']}:day:{day}:out")))
            cost = round((input_tokens / 1000) * m["cost_per_1k_input"] + (output_tokens / 1000) * m["cost_per_1k_output"], 4)
            day_data["models"][m["id"]] = {
                "requests": requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            }
            model_totals[m["id"]]["input_tokens"] += input_tokens
            model_totals[m["id"]]["output_tokens"] += output_tokens
            model_totals[m["id"]]["cost_usd"] += cost
            model_totals[m["id"]]["requests"] += requests
        daily.append(day_data)

    total_cost = round(sum(mt["cost_usd"] for mt in model_totals.values()), 2)
    total_requests = sum(mt["requests"] for mt in model_totals.values())
    total_tokens = sum(mt["input_tokens"] + mt["output_tokens"] for mt in model_totals.values())

    # Round model totals
    for mid in model_totals:
        model_totals[mid]["cost_usd"] = round(model_totals[mid]["cost_usd"], 2)

    return daily, model_totals, total_cost, total_requests, total_tokens


def _generate_latency_metrics():
    """Latency percentiles per prompt template."""
    results = []
    for pt in PROMPT_TEMPLATES:
        p50 = round(_lerp(200, 1500, _seed(f"lat:{pt['id']}:p50")))
        p95 = round(_lerp(p50 * 1.5, p50 * 4, _seed(f"lat:{pt['id']}:p95")))
        p99 = round(_lerp(p95 * 1.2, p95 * 3, _seed(f"lat:{pt['id']}:p99")))
        timeout_rate = round(_lerp(0.0, 0.03, _seed(f"lat:{pt['id']}:timeout")), 4)
        results.append({
            "prompt_id": pt["id"],
            "label": pt["label"],
            "model_id": pt["model_id"],
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "timeout_rate": timeout_rate,
            "p50_ok": p50 <= LATENCY_THRESHOLDS["p50_target_ms"],
            "p95_ok": p95 <= LATENCY_THRESHOLDS["p95_target_ms"],
            "p99_ok": p99 <= LATENCY_THRESHOLDS["p99_target_ms"],
        })
    return results


def _generate_hallucination_report():
    """Hallucination detection across prompts."""
    results = []
    for pt in PROMPT_TEMPLATES:
        total_evals = round(_lerp(50, 500, _seed(f"halluc:{pt['id']}:evals")))
        detected = round(total_evals * _lerp(0.01, 0.10, _seed(f"halluc:{pt['id']}:rate")))
        by_category = {}
        remaining = detected
        for i, cat in enumerate(HALLUCINATION_CATEGORIES):
            if i == len(HALLUCINATION_CATEGORIES) - 1:
                by_category[cat] = remaining
            else:
                share = round(remaining * _lerp(0.1, 0.4, _seed(f"halluc:{pt['id']}:{cat}")))
                by_category[cat] = share
                remaining -= share
        results.append({
            "prompt_id": pt["id"],
            "label": pt["label"],
            "total_evaluations": total_evals,
            "hallucinations_detected": detected,
            "hallucination_rate": round(detected / max(total_evals, 1), 4),
            "by_category": by_category,
            "severity": "high" if detected / max(total_evals, 1) > 0.08 else "medium" if detected / max(total_evals, 1) > 0.04 else "low",
        })
    return results


def _generate_rag_eval():
    """RAG pipeline evaluation metrics."""
    results = []
    for pipe in RAG_PIPELINES:
        s = _seed(f"rag:{pipe['id']}:eval")
        recall_at_5 = round(_lerp(0.60, 0.95, s), 3)
        precision_at_5 = round(_lerp(0.50, 0.90, _seed(f"rag:{pipe['id']}:prec")), 3)
        mrr = round(_lerp(0.55, 0.92, _seed(f"rag:{pipe['id']}:mrr")), 3)
        ndcg = round(_lerp(0.60, 0.93, _seed(f"rag:{pipe['id']}:ndcg")), 3)
        faithfulness = round(_lerp(0.70, 0.98, _seed(f"rag:{pipe['id']}:faith")), 3)
        relevance = round(_lerp(0.65, 0.95, _seed(f"rag:{pipe['id']}:rel")), 3)
        latency_ms = round(_lerp(50, 400, _seed(f"rag:{pipe['id']}:lat")))
        total_docs = round(_lerp(500, 5000, _seed(f"rag:{pipe['id']}:docs")))
        last_indexed_days_ago = round(_lerp(0, 7, _seed(f"rag:{pipe['id']}:indexed")), 1)
        results.append({
            "pipeline_id": pipe["id"],
            "label": pipe["label"],
            "vector_db": pipe["vector_db"],
            "chunk_size": pipe["chunk_size"],
            "overlap": pipe["overlap"],
            "recall_at_5": recall_at_5,
            "precision_at_5": precision_at_5,
            "mrr": mrr,
            "ndcg": ndcg,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "retrieval_latency_ms": latency_ms,
            "total_documents": total_docs,
            "last_indexed_days_ago": last_indexed_days_ago,
            "index_stale": last_indexed_days_ago > 3,
        })
    return results


# ---------------------------------------------------------------------------
# Public API — overview / breakdown / definitions
# ---------------------------------------------------------------------------

def overview():
    """LLMOps overview: KPIs, prompt health, cost summary, hallucination summary."""
    prompts = _generate_prompt_versions()
    _, model_totals, total_cost, total_requests, total_tokens = _generate_token_cost_tracking()
    latency = _generate_latency_metrics()
    halluc = _generate_hallucination_report()
    rag = _generate_rag_eval()

    active_prompts = len(prompts)
    total_versions = sum(p["total_versions"] for p in prompts)
    avg_accuracy = round(sum(p["current_accuracy"] for p in prompts) / len(prompts), 3)
    avg_halluc = round(sum(h["hallucination_rate"] for h in halluc) / len(halluc), 4)
    high_halluc_count = sum(1 for h in halluc if h["severity"] == "high")
    budget_pct = round(total_cost / COST_BUDGET_MONTHLY_USD * 100, 1)
    latency_violations = sum(1 for l in latency if not l["p95_ok"])
    avg_rag_faith = round(sum(r["faithfulness"] for r in rag) / len(rag), 3)

    # Cost by provider
    cost_by_provider = {}
    for m in MODELS:
        prov = m["provider"]
        cost_by_provider[prov] = cost_by_provider.get(prov, 0) + model_totals[m["id"]]["cost_usd"]
    for k in cost_by_provider:
        cost_by_provider[k] = round(cost_by_provider[k], 2)

    # Cost by model
    cost_by_model = [
        {"model": m["label"], "cost_usd": model_totals[m["id"]]["cost_usd"],
         "requests": model_totals[m["id"]]["requests"]}
        for m in MODELS
    ]

    # Hallucination by category (aggregate)
    halluc_agg = {}
    for h in halluc:
        for cat, count in h["by_category"].items():
            halluc_agg[cat] = halluc_agg.get(cat, 0) + count

    # Prompt summary
    prompt_summary = [
        {
            "prompt": p["label"],
            "category": p["category"],
            "model": p["model_id"],
            "versions": p["total_versions"],
            "accuracy": p["current_accuracy"],
            "halluc_rate": p["current_halluc_rate"],
            "latency_ms": p["current_latency_ms"],
        }
        for p in prompts
    ]

    return {
        "kpis": {
            "active_prompts": active_prompts,
            "total_versions": total_versions,
            "avg_accuracy": avg_accuracy,
            "avg_hallucination_rate": avg_halluc,
            "total_cost_30d_usd": total_cost,
            "budget_pct": budget_pct,
            "total_requests_30d": total_requests,
            "total_tokens_30d": total_tokens,
            "latency_violations": latency_violations,
            "high_halluc_prompts": high_halluc_count,
            "avg_rag_faithfulness": avg_rag_faith,
        },
        "cost_by_provider": cost_by_provider,
        "cost_by_model": cost_by_model,
        "hallucination_by_category": halluc_agg,
        "prompt_summary": prompt_summary,
    }


def breakdown():
    """LLMOps breakdown: prompt version history, daily token usage, latency detail,
    hallucination detail, RAG eval detail."""
    prompts = _generate_prompt_versions()
    daily, model_totals, total_cost, total_requests, total_tokens = _generate_token_cost_tracking()
    latency = _generate_latency_metrics()
    halluc = _generate_hallucination_report()
    rag = _generate_rag_eval()

    return {
        "prompt_versions": prompts,
        "daily_token_usage": daily[:14],  # last 14 days for chart
        "model_totals": {mid: model_totals[mid] for mid in model_totals},
        "total_cost_30d_usd": total_cost,
        "total_requests_30d": total_requests,
        "budget_usd": COST_BUDGET_MONTHLY_USD,
        "budget_pct": round(total_cost / COST_BUDGET_MONTHLY_USD * 100, 1),
        "latency_metrics": latency,
        "latency_thresholds": LATENCY_THRESHOLDS,
        "hallucination_report": halluc,
        "rag_evaluation": rag,
    }


def definitions():
    """LLMOps definitions, thresholds, reference info."""
    return {
        "models": MODELS,
        "prompt_templates": [
            {"id": pt["id"], "label": pt["label"], "category": pt["category"], "model_id": pt["model_id"]}
            for pt in PROMPT_TEMPLATES
        ],
        "rag_pipelines": RAG_PIPELINES,
        "hallucination_categories": {
            "factual_error": "Statement contradicts established medical/scientific fact",
            "unsupported_claim": "Claim not supported by provided context or source documents",
            "contradicts_source": "Output directly contradicts the input source material",
            "fabricated_reference": "Citation or reference that does not exist",
            "numerical_error": "Incorrect numbers, dosages, frequencies, or measurements",
            "temporal_confusion": "Wrong dates, sequences, or temporal relationships",
        },
        "latency_thresholds": LATENCY_THRESHOLDS,
        "cost_budget": {
            "monthly_usd": COST_BUDGET_MONTHLY_USD,
            "alert_levels": {
                "50%": {"threshold_usd": COST_BUDGET_MONTHLY_USD * 0.5, "action": "Informational — track spend"},
                "80%": {"threshold_usd": COST_BUDGET_MONTHLY_USD * 0.8, "action": "Warning — review usage"},
                "100%": {"threshold_usd": COST_BUDGET_MONTHLY_USD, "action": "Critical — throttle non-essential"},
            },
        },
        "rag_metrics": {
            "recall_at_5": "Fraction of relevant documents in top-5 results",
            "precision_at_5": "Fraction of top-5 results that are relevant",
            "mrr": "Mean Reciprocal Rank — average 1/rank of first relevant result",
            "ndcg": "Normalized Discounted Cumulative Gain — ranking quality",
            "faithfulness": "Fraction of generated claims supported by retrieved context",
            "relevance": "Semantic similarity between query and retrieved passages",
        },
        "prompt_versioning": {
            "description": "Each prompt template is versioned. New versions are created when prompts are edited or model parameters change. Older versions are archived but retained for A/B comparison.",
            "fields": ["version", "avg_tokens", "accuracy", "hallucination_rate", "avg_latency_ms", "status", "created_days_ago"],
        },
        "clinical_relevance": (
            "LLMOps monitors the AI language models that power clinical decision support, "
            "report generation, and patient data parsing. Hallucination detection is critical "
            "in medical contexts — fabricated drug interactions or incorrect dosages can "
            "directly harm patients. Prompt versioning ensures reproducibility. RAG evaluation "
            "ensures that retrieved clinical guidelines are accurate and up-to-date. Cost "
            "monitoring balances local (Ollama) vs cloud model usage."
        ),
    }
