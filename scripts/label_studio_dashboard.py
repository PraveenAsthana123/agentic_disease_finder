"""
Label Studio / CVAT Annotation Quality Dashboard
=================================================
Analyzes EEG annotation tasks — inter-annotator agreement (Cohen's kappa,
Krippendorff's alpha), annotation coverage, label distribution, and
time-series annotation quality metrics.

Uses REAL CHB-MIT PhysioNet seizure annotations (.seizures summary files)
as the ground-truth annotation layer, then simulates a realistic multi-
annotator scenario to compute agreement metrics that would apply to a
Label Studio / CVAT annotation pipeline.

Data source: data/real_eeg/epilepsy_physionet/chbNN/
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "real_eeg" / "epilepsy_physionet"

# ── Annotation label taxonomy (EEG clinical annotation) ──────────────

ANNOTATION_LABELS = [
    {"id": "seizure", "name": "Seizure", "color": "#ef4444", "hotkey": "S"},
    {"id": "spike", "name": "Spike/Sharp-wave", "color": "#f59e0b", "hotkey": "K"},
    {"id": "artifact", "name": "Artifact", "color": "#94a3b8", "hotkey": "A"},
    {"id": "normal", "name": "Normal Background", "color": "#22c55e", "hotkey": "N"},
    {"id": "slowing", "name": "Focal Slowing", "color": "#6366f1", "hotkey": "W"},
    {"id": "burst_suppression", "name": "Burst Suppression", "color": "#ec4899", "hotkey": "B"},
]

# ── Load real seizure annotations from CHB-MIT ───────────────────────

def _load_chb_annotations() -> list[dict]:
    """Parse CHB-MIT summary files for seizure start/end annotations."""
    annotations = []
    if not _DATA_ROOT.exists():
        return annotations

    for subj_dir in sorted(_DATA_ROOT.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("chb"):
            continue
        summary = subj_dir / f"{subj_dir.name}-summary.txt"
        if not summary.exists():
            # Try alternate name
            candidates = list(subj_dir.glob("*summary*"))
            if candidates:
                summary = candidates[0]
            else:
                continue
        try:
            text = summary.read_text(errors="replace")
            current_file = None
            for line in text.splitlines():
                line_s = line.strip()
                fn_match = re.match(r"File Name:\s*(\S+)", line_s)
                if fn_match:
                    current_file = fn_match.group(1)
                start_match = re.match(
                    r"Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds", line_s
                )
                if start_match and current_file:
                    start = int(start_match.group(1))
                    # Look for end on next relevant line; store start for now
                    annotations.append({
                        "subject": subj_dir.name,
                        "file": current_file,
                        "start_sec": start,
                        "end_sec": None,
                        "label": "seizure",
                        "annotator": "clinical_gold",
                    })
                end_match = re.match(
                    r"Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds", line_s
                )
                if end_match and annotations and annotations[-1]["end_sec"] is None:
                    annotations[-1]["end_sec"] = int(end_match.group(1))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", summary, exc)

    # Fix any missing end times
    for ann in annotations:
        if ann["end_sec"] is None:
            ann["end_sec"] = ann["start_sec"] + 30  # default 30s window

    return annotations


# ── Simulate multi-annotator scenario ────────────────────────────────

def _simulate_annotators(gold: list[dict], n_annotators: int = 3,
                         seed: int = 42) -> list[dict]:
    """
    Given gold-standard seizure annotations, simulate n_annotators with
    realistic disagreement patterns (boundary jitter, missed events,
    false positives, label confusion).
    """
    rng = np.random.RandomState(seed)
    all_annotations = []

    # Gold annotator
    for ann in gold:
        all_annotations.append({**ann, "annotator": "annotator_gold"})

    for i in range(1, n_annotators + 1):
        annotator_id = f"annotator_{i}"
        # Sensitivity: probability of catching each seizure
        sensitivity = 0.75 + rng.random() * 0.20  # 0.75-0.95

        for ann in gold:
            if rng.random() < sensitivity:
                # Boundary jitter: ±5 seconds
                jitter_start = int(rng.normal(0, 3))
                jitter_end = int(rng.normal(0, 3))
                duration = ann["end_sec"] - ann["start_sec"]

                # Occasional label confusion (5% chance)
                label = ann["label"]
                if rng.random() < 0.05:
                    label = rng.choice(["spike", "slowing", "artifact"])

                all_annotations.append({
                    "subject": ann["subject"],
                    "file": ann["file"],
                    "start_sec": max(0, ann["start_sec"] + jitter_start),
                    "end_sec": ann["end_sec"] + jitter_end,
                    "label": label,
                    "annotator": annotator_id,
                })

            # Small chance of false positive nearby
            if rng.random() < 0.08:
                fp_start = ann["start_sec"] + int(rng.normal(60, 30))
                fp_dur = int(rng.exponential(10)) + 3
                all_annotations.append({
                    "subject": ann["subject"],
                    "file": ann["file"],
                    "start_sec": max(0, fp_start),
                    "end_sec": fp_start + fp_dur,
                    "label": rng.choice(["spike", "artifact", "slowing"]),
                    "annotator": annotator_id,
                })

    return all_annotations


# ── Agreement metrics ────────────────────────────────────────────────

def _cohens_kappa(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compute Cohen's kappa for two annotator label arrays."""
    if len(labels_a) == 0:
        return 0.0
    n = len(labels_a)
    classes = list(set(labels_a) | set(labels_b))
    k = len(classes)
    if k < 2:
        return 1.0

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        conf[cls_to_idx[a], cls_to_idx[b]] += 1

    po = np.trace(conf) / n
    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _krippendorff_alpha(annotations_matrix: np.ndarray) -> float:
    """
    Simplified Krippendorff's alpha for nominal data.
    annotations_matrix: (n_annotators, n_items) with -1 for missing.
    """
    n_annotators, n_items = annotations_matrix.shape
    if n_items < 2:
        return 0.0

    # Observed disagreement
    Do = 0.0
    n_pairs = 0
    for j in range(n_items):
        valid = annotations_matrix[:, j]
        valid = valid[valid >= 0]
        m = len(valid)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                Do += (0 if valid[a] == valid[b] else 1)
                n_pairs += 1

    if n_pairs == 0:
        return 0.0
    Do /= n_pairs

    # Expected disagreement
    all_valid = annotations_matrix[annotations_matrix >= 0]
    unique, counts = np.unique(all_valid, return_counts=True)
    total = counts.sum()
    De = 1.0 - np.sum(counts * (counts - 1)) / (total * (total - 1)) if total > 1 else 1.0

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def _compute_agreement(all_annotations: list[dict]) -> dict:
    """Compute inter-annotator agreement metrics."""
    annotators = sorted(set(a["annotator"] for a in all_annotations))
    subjects = sorted(set(a["subject"] for a in all_annotations))

    # Build per-subject label vectors (discretize time into 10s windows)
    label_to_int = {l["id"]: i for i, l in enumerate(ANNOTATION_LABELS)}
    label_to_int["normal"] = label_to_int.get("normal", 3)

    pairwise_kappas = []
    all_items = []
    all_matrix_rows = {ann_id: [] for ann_id in annotators}

    for subj in subjects:
        subj_anns = [a for a in all_annotations if a["subject"] == subj]
        if not subj_anns:
            continue
        max_time = max(a["end_sec"] for a in subj_anns)
        n_windows = max(1, max_time // 10 + 1)

        for ann_id in annotators:
            ann_labels = np.full(n_windows, label_to_int.get("normal", 3), dtype=int)
            for a in subj_anns:
                if a["annotator"] != ann_id:
                    continue
                w_start = a["start_sec"] // 10
                w_end = min(a["end_sec"] // 10 + 1, n_windows)
                lbl_int = label_to_int.get(a["label"], 3)
                ann_labels[w_start:w_end] = lbl_int
            all_matrix_rows[ann_id].extend(ann_labels.tolist())

    # Build matrix (n_annotators x n_items)
    n_items = len(all_matrix_rows[annotators[0]]) if annotators else 0
    if n_items == 0:
        return {"cohens_kappa_mean": 0, "krippendorff_alpha": 0, "pairwise": []}

    matrix = np.array([all_matrix_rows[a] for a in annotators], dtype=int)

    # Pairwise Cohen's kappa
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            k = _cohens_kappa(matrix[i], matrix[j])
            pairwise_kappas.append({
                "annotator_a": annotators[i],
                "annotator_b": annotators[j],
                "kappa": round(k, 4),
            })

    alpha = _krippendorff_alpha(matrix)
    mean_kappa = float(np.mean([p["kappa"] for p in pairwise_kappas])) if pairwise_kappas else 0.0

    return {
        "cohens_kappa_mean": round(mean_kappa, 4),
        "krippendorff_alpha": round(alpha, 4),
        "pairwise": pairwise_kappas,
        "n_annotators": len(annotators),
        "n_items": n_items,
    }


# ── Public API functions ─────────────────────────────────────────────

def annotation_overview() -> dict[str, Any]:
    """Main overview: annotation stats, agreement, coverage."""
    gold = _load_chb_annotations()
    if not gold:
        return {
            "available": False,
            "reason": "No CHB-MIT annotation data found",
            "data_path": str(_DATA_ROOT),
        }

    all_anns = _simulate_annotators(gold, n_annotators=3)
    agreement = _compute_agreement(all_anns)

    # Annotation stats
    subjects = sorted(set(a["subject"] for a in gold))
    total_seizure_time = sum(a["end_sec"] - a["start_sec"] for a in gold)

    # Label distribution across all annotators
    label_counts = {}
    for a in all_anns:
        label_counts[a["label"]] = label_counts.get(a["label"], 0) + 1
    label_dist = [
        {"label": lbl, "count": cnt, "percent": round(100 * cnt / len(all_anns), 1)}
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])
    ]

    # Per-subject stats
    subject_stats = []
    for subj in subjects:
        subj_gold = [a for a in gold if a["subject"] == subj]
        subj_all = [a for a in all_anns if a["subject"] == subj]
        ann_ids = set(a["annotator"] for a in subj_all)
        subject_stats.append({
            "subject": subj,
            "n_seizures": len(subj_gold),
            "total_seconds": sum(a["end_sec"] - a["start_sec"] for a in subj_gold),
            "n_annotations": len(subj_all),
            "n_annotators": len(ann_ids),
        })

    # Per-annotator stats
    annotator_stats = []
    annotators = sorted(set(a["annotator"] for a in all_anns))
    for ann_id in annotators:
        ann_subset = [a for a in all_anns if a["annotator"] == ann_id]
        annotator_stats.append({
            "annotator": ann_id,
            "n_annotations": len(ann_subset),
            "labels_used": len(set(a["label"] for a in ann_subset)),
            "subjects_covered": len(set(a["subject"] for a in ann_subset)),
        })

    return {
        "available": True,
        "total_annotations": len(all_anns),
        "total_subjects": len(subjects),
        "total_seizures_gold": len(gold),
        "total_seizure_seconds": total_seizure_time,
        "agreement": agreement,
        "label_distribution": label_dist,
        "subject_stats": subject_stats,
        "annotator_stats": annotator_stats,
        "annotation_labels": ANNOTATION_LABELS,
        "tools": {
            "label_studio": {
                "name": "Label Studio",
                "version": "1.13.x",
                "role": "Primary annotation UI — time-series labeling, review workflows",
                "features": [
                    "Time-series annotation with waveform display",
                    "Multi-annotator task assignment",
                    "Review / consensus workflow",
                    "Webhook integration for model-assisted labeling",
                    "Export: JSON, COCO, YOLO, Pascal VOC",
                ],
            },
            "cvat": {
                "name": "CVAT",
                "version": "2.x",
                "role": "Computer-vision annotation — spectrogram / topomap labeling",
                "features": [
                    "Spectrogram region annotation",
                    "Interpolation for temporal sequences",
                    "Auto-annotation with ML models",
                    "Team management and quality control",
                ],
            },
        },
    }


def annotation_agreement() -> dict[str, Any]:
    """Detailed agreement metrics with pairwise matrix."""
    gold = _load_chb_annotations()
    if not gold:
        return {"available": False}

    all_anns = _simulate_annotators(gold, n_annotators=3)
    agreement = _compute_agreement(all_anns)

    # Kappa interpretation
    kappa = agreement["cohens_kappa_mean"]
    if kappa >= 0.81:
        interpretation = "Almost perfect agreement"
    elif kappa >= 0.61:
        interpretation = "Substantial agreement"
    elif kappa >= 0.41:
        interpretation = "Moderate agreement"
    elif kappa >= 0.21:
        interpretation = "Fair agreement"
    else:
        interpretation = "Slight agreement"

    alpha = agreement["krippendorff_alpha"]
    alpha_interp = (
        "Reliable (α ≥ 0.80)" if alpha >= 0.80
        else "Tentative (0.67 ≤ α < 0.80)" if alpha >= 0.67
        else "Unreliable (α < 0.67)"
    )

    return {
        "available": True,
        "cohens_kappa_mean": kappa,
        "kappa_interpretation": interpretation,
        "krippendorff_alpha": alpha,
        "alpha_interpretation": alpha_interp,
        "pairwise_kappas": agreement["pairwise"],
        "n_annotators": agreement["n_annotators"],
        "n_items": agreement["n_items"],
        "scale": [
            {"range": "0.81 – 1.00", "label": "Almost perfect"},
            {"range": "0.61 – 0.80", "label": "Substantial"},
            {"range": "0.41 – 0.60", "label": "Moderate"},
            {"range": "0.21 – 0.40", "label": "Fair"},
            {"range": "0.00 – 0.20", "label": "Slight"},
        ],
    }


def annotation_definitions() -> dict[str, Any]:
    """Definitions, label taxonomy, and tool documentation."""
    return {
        "title": "Label Studio / CVAT Annotation Quality Dashboard",
        "description": (
            "Analyzes EEG annotation quality using inter-annotator agreement "
            "metrics (Cohen's kappa, Krippendorff's alpha) computed on real "
            "CHB-MIT PhysioNet seizure annotations with simulated multi-annotator "
            "disagreement patterns."
        ),
        "annotation_labels": ANNOTATION_LABELS,
        "metrics": [
            {
                "name": "Cohen's Kappa (κ)",
                "description": "Measures pairwise agreement between two annotators, corrected for chance agreement. Range: -1 to 1.",
                "reference": "Cohen, 1960",
            },
            {
                "name": "Krippendorff's Alpha (α)",
                "description": "Generalizes to multiple annotators and handles missing data. Recommended threshold: α ≥ 0.80 for reliable annotations.",
                "reference": "Krippendorff, 2011",
            },
            {
                "name": "Annotation Coverage",
                "description": "Percentage of total recording time that has been annotated by at least one annotator.",
            },
            {
                "name": "Label Consistency",
                "description": "Proportion of time windows where all annotators assigned the same label.",
            },
        ],
        "tools": {
            "label_studio": {
                "name": "Label Studio",
                "url": "https://labelstud.io",
                "license": "Apache 2.0",
                "description": "Open-source data labeling platform supporting time-series, image, text, and audio annotation with ML-assisted labeling.",
            },
            "cvat": {
                "name": "CVAT (Computer Vision Annotation Tool)",
                "url": "https://www.cvat.ai",
                "license": "MIT",
                "description": "Open-source annotation tool optimized for computer vision tasks — spectrogram and topomap region labeling for EEG.",
            },
        },
        "data_source": "CHB-MIT Scalp EEG Database (PhysioNet)",
    }
