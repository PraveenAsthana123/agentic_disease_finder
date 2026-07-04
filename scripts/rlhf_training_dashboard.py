"""RLHF Training Dashboard — Reinforcement Learning from Human Feedback analytics.

All data from REAL clinical feedback in data/clinical.db (feedback, hitl_reviews,
expert_reviews, transaction_log tables).

Reinforcement Learning from Human Feedback (RLHF) is a technique for aligning AI
model outputs with human preferences.  In clinical neuro-AI, this means aligning
EEG/seizure predictions with expert neurologist judgment so the model learns to
prefer outputs that clinicians trust and override outputs they reject.

The RLHF pipeline has four stages:
  1. Data Collection — gather human feedback, HITL reviews, and expert opinions
     on AI predictions.
  2. Preference Labeling — convert reviews into preference pairs (chosen vs
     rejected outputs) using override/disagree signals as rejection labels.
  3. Reward Model Training — train a reward model on the preference pairs to
     score AI outputs by clinical acceptability.
  4. Policy Fine-Tuning — use PPO (Proximal Policy Optimization) to fine-tune
     the prediction model against the reward model while constraining divergence
     from the reference policy via KL penalty.

Why RLHF matters for clinical AI:
  - Regulatory alignment: models that incorporate clinician feedback are more
    defensible under FDA/IEC 62304 requirements.
  - Safety: overrides capture near-miss events; training on them reduces future
    misclassifications.
  - Trust calibration: clinicians are more likely to adopt AI tools that
    demonstrably learn from their corrections.

Reference:
  Ouyang L et al. Training language models to follow instructions with human
  feedback. NeurIPS 2022.
  Christiano PB et al. Deep reinforcement learning from human preferences.
  NeurIPS 2017.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# ----- Readiness thresholds -----
MIN_PREFERENCE_PAIRS = 500
MIN_FEEDBACK_ENTRIES = 200
MIN_EXPERT_REVIEWS = 50
MIN_HITL_REVIEWS = 50


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _parse_fields(row):
    """Safely parse fields_json from a hitl_reviews row."""
    try:
        return json.loads(row.get("fields_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


# ─────────────────────────────────────────────────────────────────────
# Preference pair generation helpers
# ─────────────────────────────────────────────────────────────────────

def _hitl_preference_pairs(hitl_rows):
    """Generate preference pairs from HITL reviews where the human overrode AI."""
    pairs = []
    for r in hitl_rows:
        fields = _parse_fields(r)
        if fields.get("decision") == "override":
            pairs.append({
                "chosen": fields.get("human_decision", ""),
                "rejected": fields.get("ai_prediction", ""),
                "source": "hitl_override",
                "reason_code": fields.get("reason_code", ""),
                "patient_id": r.get("patient_id", ""),
                "date": r.get("created_at", ""),
            })
    return pairs


def _expert_preference_pairs(expert_rows):
    """Generate preference pairs from expert reviews that disagree with AI."""
    pairs = []
    for r in expert_rows:
        if (r.get("agree_with_ai") or "").lower() == "disagree":
            pairs.append({
                "chosen": r.get("finding", ""),
                "rejected": f"AI prediction (expert disagreed)",
                "source": "expert_disagree",
                "patient_id": r.get("patient_id", ""),
                "role": r.get("role", ""),
                "expert": r.get("expert", ""),
                "date": r.get("created_at", ""),
            })
    return pairs


# ─────────────────────────────────────────────────────────────────────
# Readiness scoring
# ─────────────────────────────────────────────────────────────────────

def _readiness_score(total_feedback, total_hitl, total_expert, total_pairs):
    """Compute RLHF readiness 0-100 based on data sufficiency."""
    components = [
        min(total_feedback / MIN_FEEDBACK_ENTRIES, 1.0) * 25,
        min(total_hitl / MIN_HITL_REVIEWS, 1.0) * 20,
        min(total_expert / MIN_EXPERT_REVIEWS, 1.0) * 20,
        min(total_pairs / MIN_PREFERENCE_PAIRS, 1.0) * 35,
    ]
    return round(sum(components))


def _pipeline_stages(total_feedback, total_pairs, readiness):
    """Return pipeline stage statuses."""
    data_status = "active" if total_feedback > 0 else "pending"
    pref_status = "active" if total_pairs > 0 else "pending"
    reward_status = "ready" if readiness >= 60 else "insufficient_data"
    policy_status = "ready" if readiness >= 80 else "insufficient_data"
    return [
        {
            "stage": "Data Collection",
            "status": data_status,
            "description": "Gathering clinician feedback, HITL overrides, and expert reviews",
            "progress_pct": min(100, round((total_feedback / max(MIN_FEEDBACK_ENTRIES, 1)) * 100)),
        },
        {
            "stage": "Preference Labeling",
            "status": pref_status,
            "description": "Converting overrides and disagreements into chosen/rejected pairs",
            "progress_pct": min(100, round((total_pairs / max(MIN_PREFERENCE_PAIRS, 1)) * 100)),
        },
        {
            "stage": "Reward Model Training",
            "status": reward_status,
            "description": "Training Bradley-Terry reward model on preference pairs",
            "progress_pct": min(100, readiness) if readiness >= 60 else round(readiness / 60 * 100),
        },
        {
            "stage": "Policy Fine-Tuning",
            "status": policy_status,
            "description": "PPO fine-tuning with KL-constrained reward maximization",
            "progress_pct": min(100, readiness) if readiness >= 80 else round(readiness / 80 * 100),
        },
    ]


# ═════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════

def overview():
    """Summary KPIs: feedback counts, preference pairs, rating distribution,
    agreement rate, RLHF readiness score, and pipeline stage statuses."""

    feedback_rows = _rows("SELECT * FROM feedback ORDER BY created_at DESC")
    hitl_rows = _rows("SELECT * FROM hitl_reviews ORDER BY created_at DESC")
    expert_rows = _rows("SELECT * FROM expert_reviews ORDER BY created_at DESC")

    total_feedback = len(feedback_rows)
    total_hitl = len(hitl_rows)
    total_expert = len(expert_rows)

    # Preference pairs
    hitl_pairs = _hitl_preference_pairs(hitl_rows)
    expert_pairs = _expert_preference_pairs(expert_rows)
    total_pairs = len(hitl_pairs) + len(expert_pairs)

    # Rating distribution from feedback
    rating_dist = Counter()
    for r in feedback_rows:
        rating = r.get("rating")
        if rating is not None:
            rating_dist[int(rating)] += 1
    # Ensure all ratings 1-5 present
    rating_distribution = [{"rating": i, "count": rating_dist.get(i, 0)} for i in range(1, 6)]

    # Role distribution from feedback
    role_dist = Counter()
    for r in feedback_rows:
        role = r.get("role") or "unknown"
        role_dist[role] += 1
    role_distribution = [{"role": k, "count": v} for k, v in sorted(role_dist.items(), key=lambda x: -x[1])]

    # Expert agreement rate
    agree_count = sum(1 for r in expert_rows if (r.get("agree_with_ai") or "").lower() == "agree")
    disagree_count = sum(1 for r in expert_rows if (r.get("agree_with_ai") or "").lower() == "disagree")
    total_rated_expert = agree_count + disagree_count
    agreement_rate = round(agree_count / total_rated_expert * 100, 1) if total_rated_expert > 0 else None

    # HITL accept/override counts
    accept_count = sum(1 for r in hitl_rows if _parse_fields(r).get("decision") == "accept")
    override_count = sum(1 for r in hitl_rows if _parse_fields(r).get("decision") == "override")

    # Readiness
    readiness = _readiness_score(total_feedback, total_hitl, total_expert, total_pairs)
    stages = _pipeline_stages(total_feedback + total_hitl + total_expert, total_pairs, readiness)

    return {
        "total_feedback_entries": total_feedback,
        "total_hitl_reviews": total_hitl,
        "total_expert_reviews": total_expert,
        "total_preference_pairs": total_pairs,
        "hitl_accepts": accept_count,
        "hitl_overrides": override_count,
        "expert_agree": agree_count,
        "expert_disagree": disagree_count,
        "agreement_rate_pct": agreement_rate,
        "rating_distribution": rating_distribution,
        "role_distribution": role_distribution,
        "rlhf_readiness_score": readiness,
        "readiness_thresholds": {
            "min_feedback": MIN_FEEDBACK_ENTRIES,
            "min_hitl": MIN_HITL_REVIEWS,
            "min_expert": MIN_EXPERT_REVIEWS,
            "min_preference_pairs": MIN_PREFERENCE_PAIRS,
        },
        "pipeline_stages": stages,
    }


def breakdown():
    """Detailed preference pairs, monthly trends, reason code distribution,
    per-role quality metrics, reward model readiness, and training config."""

    feedback_rows = _rows("SELECT * FROM feedback ORDER BY created_at ASC")
    hitl_rows = _rows("SELECT * FROM hitl_reviews ORDER BY created_at ASC")
    expert_rows = _rows("SELECT * FROM expert_reviews ORDER BY created_at ASC")

    # ── Preference pairs detail ──
    hitl_pairs = _hitl_preference_pairs(hitl_rows)
    expert_pairs = _expert_preference_pairs(expert_rows)
    all_pairs = hitl_pairs + expert_pairs

    # ── Monthly feedback trend ──
    monthly = defaultdict(lambda: {"feedback": 0, "hitl": 0, "expert": 0})
    for r in feedback_rows:
        month = (r.get("created_at") or "")[:7]
        if month:
            monthly[month]["feedback"] += 1
    for r in hitl_rows:
        month = (r.get("created_at") or "")[:7]
        if month:
            monthly[month]["hitl"] += 1
    for r in expert_rows:
        month = (r.get("created_at") or "")[:7]
        if month:
            monthly[month]["expert"] += 1
    monthly_trend = [{"month": m, **v} for m, v in sorted(monthly.items())]

    # ── Override reason codes ──
    reason_codes = Counter()
    for r in hitl_rows:
        fields = _parse_fields(r)
        if fields.get("decision") == "override":
            code = fields.get("reason_code", "unspecified")
            reason_codes[code] += 1
    reason_code_distribution = [{"reason_code": k, "count": v}
                                for k, v in sorted(reason_codes.items(), key=lambda x: -x[1])]

    # ── Per-role feedback quality ──
    role_ratings = defaultdict(list)
    for r in feedback_rows:
        role = r.get("role") or "unknown"
        rating = r.get("rating")
        if rating is not None:
            role_ratings[role].append(int(rating))
    per_role_quality = []
    for role, ratings in sorted(role_ratings.items()):
        per_role_quality.append({
            "role": role,
            "total_feedback": len(ratings),
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "min_rating": min(ratings) if ratings else None,
            "max_rating": max(ratings) if ratings else None,
        })

    # ── Reward model training readiness ──
    total_pairs = len(all_pairs)
    dataset_readiness = {
        "current_pairs": total_pairs,
        "minimum_required": MIN_PREFERENCE_PAIRS,
        "surplus_deficit": total_pairs - MIN_PREFERENCE_PAIRS,
        "ready": total_pairs >= MIN_PREFERENCE_PAIRS,
        "sources": {
            "hitl_overrides": len(hitl_pairs),
            "expert_disagreements": len(expert_pairs),
        },
    }

    # ── Training configuration (reference/planned, not fabricated results) ──
    training_config = {
        "algorithm": "PPO (Proximal Policy Optimization)",
        "reward_model": {
            "architecture": "Bradley-Terry pairwise preference model",
            "loss": "Binary cross-entropy on preference pairs",
            "epochs": 3,
            "learning_rate": 1e-5,
            "batch_size": 32,
        },
        "policy": {
            "optimizer": "AdamW",
            "learning_rate": 1e-6,
            "kl_coeff": 0.02,
            "clip_range": 0.2,
            "value_coeff": 0.5,
            "entropy_coeff": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 4,
            "minibatch_size": 8,
            "gamma": 0.99,
            "lam": 0.95,
        },
        "safety": {
            "kl_target": 6.0,
            "kl_horizon": 10000,
            "reward_clip": 10.0,
            "early_stopping_kl": 0.1,
            "reference_policy_frozen": True,
        },
        "note": "Planned configuration — training will begin when preference pair "
                "dataset reaches minimum threshold.",
    }

    return {
        "preference_pairs": all_pairs,
        "preference_pair_count": total_pairs,
        "monthly_trend": monthly_trend,
        "reason_code_distribution": reason_code_distribution,
        "per_role_quality": per_role_quality,
        "dataset_readiness": dataset_readiness,
        "training_config": training_config,
    }


def definitions():
    """RLHF terminology definitions with clinical context."""
    return {
        "title": "RLHF Training Dashboard — Terminology & Definitions",
        "definitions": [
            {
                "term": "RLHF (Reinforcement Learning from Human Feedback)",
                "definition": "A training paradigm where a policy model is optimized "
                              "to produce outputs preferred by human evaluators. In "
                              "clinical AI, human evaluators are neurologists and EEG "
                              "technicians whose overrides and corrections form the "
                              "preference signal.",
            },
            {
                "term": "Preference Pair",
                "definition": "A (chosen, rejected) tuple representing one human "
                              "judgment. The chosen output is what the clinician "
                              "preferred; the rejected output is what the AI produced "
                              "and the clinician overrode. These pairs are the training "
                              "data for the reward model.",
            },
            {
                "term": "Reward Model",
                "definition": "A model trained on preference pairs to predict a scalar "
                              "reward score for any given AI output. Higher scores "
                              "correspond to outputs more likely to be accepted by "
                              "clinicians. Typically uses a Bradley-Terry pairwise "
                              "comparison framework.",
            },
            {
                "term": "PPO (Proximal Policy Optimization)",
                "definition": "The RL algorithm used to fine-tune the policy model "
                              "against the reward model. PPO clips the policy ratio "
                              "to prevent destructively large updates, making training "
                              "stable. Widely used in RLHF pipelines (InstructGPT, "
                              "ChatGPT).",
            },
            {
                "term": "KL Divergence (Kullback-Leibler)",
                "definition": "A measure of how much the fine-tuned policy has "
                              "diverged from the reference (pre-RLHF) policy. A KL "
                              "penalty is added to the reward to prevent the model "
                              "from drifting too far from its original behavior, "
                              "preserving general capability while aligning preferences.",
            },
            {
                "term": "Reference Policy",
                "definition": "The frozen copy of the model before RLHF fine-tuning. "
                              "Used to compute the KL penalty. Ensures the fine-tuned "
                              "model does not degenerate or overfit to reward hacking.",
            },
            {
                "term": "Chosen / Rejected",
                "definition": "Labels in a preference pair. 'Chosen' is the output "
                              "the human preferred (human_decision in HITL overrides, "
                              "expert finding when disagreeing with AI). 'Rejected' is "
                              "the AI output that was overridden.",
            },
            {
                "term": "Bradley-Terry Model",
                "definition": "A probabilistic model for pairwise comparisons. Given "
                              "two items, the probability of preferring item A over B "
                              "is sigmoid(r(A) - r(B)), where r is the reward. This is "
                              "the standard loss function for training reward models "
                              "from preference data.",
            },
            {
                "term": "DPO (Direct Preference Optimization)",
                "definition": "An alternative to PPO-based RLHF that skips the "
                              "explicit reward model. DPO directly optimizes the "
                              "policy on preference pairs using a closed-form loss, "
                              "simplifying the pipeline. May be used as a future "
                              "alternative approach.",
            },
            {
                "term": "Constitutional AI (CAI)",
                "definition": "A self-supervision approach where the model critiques "
                              "and revises its own outputs against a set of principles "
                              "(a 'constitution'). In clinical AI, the constitution "
                              "could encode safety rules such as 'never dismiss "
                              "potential seizure activity' or 'flag artifact vs "
                              "pathology uncertainty'.",
            },
            {
                "term": "Clinical RLHF Context",
                "definition": "In neuro-AI, RLHF aligns EEG classification and "
                              "seizure detection models with expert clinical judgment. "
                              "When a neurologist overrides an AI prediction (e.g., "
                              "reclassifying 'Epilepsy' as 'Artifact'), this creates a "
                              "preference signal. Training on these signals reduces "
                              "clinically dangerous misclassifications and builds "
                              "trust in AI-assisted diagnosis.",
            },
        ],
    }
