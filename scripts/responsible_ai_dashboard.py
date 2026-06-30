"""Responsible AI Dashboard — real metrics from responsible AI analysis results.

Sources:
- results/responsible_ai_scores.json (framework scores, 30 frameworks)
- results/responsible_ai_metrics.json (reliability, trustworthiness, fairness analysis)
- results/responsible_ai_test_results.json (consistency, robustness, fairness tests)
- jobs/reports/fairness_latest.json (Fairlearn demographic parity)
"""

import json
import os
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), '..')


def _load(path):
    fp = os.path.join(BASE, path)
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def responsible_ai_overview():
    """Aggregate responsible AI posture: overall score, framework breakdown,
    test pass rate, fairness gate, reliability + trustworthiness scores."""
    scores = _load('results/responsible_ai_scores.json')
    metrics = _load('results/responsible_ai_metrics.json')
    tests = _load('results/responsible_ai_test_results.json')
    fairness = _load('jobs/reports/fairness_latest.json')

    if not scores:
        return {"available": False, "note": "responsible_ai_scores.json not found"}

    meta = scores.get('metadata', {})
    frameworks = scores.get('frameworks', {})

    # Framework score cards
    framework_cards = []
    for fid, fdata in frameworks.items():
        framework_cards.append({
            "id": fid,
            "label": fid.replace('_', ' ').title(),
            "score": fdata.get('score', 0),
            "status": fdata.get('status', 'unknown'),
        })
    framework_cards.sort(key=lambda x: x['score'] if x['score'] is not None else 0, reverse=True)

    # Reliability detail
    reliability = {}
    if metrics:
        rel = metrics.get('reliability_analysis', {})
        reliability = {
            "score": rel.get('score', 0),
            "status": rel.get('status', 'unknown'),
            "sla_targets": rel.get('sla_targets', {}),
            "consistency": rel.get('consistency_scores', {}),
            "robustness": rel.get('robustness', {}),
            "calibration": rel.get('calibration', {}),
        }

    # Trustworthiness detail
    trustworthiness = {}
    if metrics:
        tw = metrics.get('trustworthiness_analysis', {})
        trustworthiness = {
            "score": tw.get('score', 0),
            "status": tw.get('status', 'unknown'),
        }

    # Test results summary
    test_summary = {}
    if tests:
        test_summary = {
            "overall_status": tests.get('overall_status', 'unknown'),
            "tests_passed": tests.get('tests_passed', 0),
            "total_tests": tests.get('total_tests', 0),
            "pass_rate": tests.get('pass_rate', 0),
            "timestamp": tests.get('timestamp', ''),
        }

    # Per-test breakdown
    test_cards = []
    if tests:
        for tid, tdata in tests.get('test_results', {}).items():
            test_cards.append({
                "id": tid,
                "label": tdata.get('test', tid),
                "status": tdata.get('status', 'unknown'),
            })

    # Fairness gate
    fairness_summary = {}
    if fairness:
        fairness_summary = {
            "gate": fairness.get('fairness_gate', 'unknown'),
            "dpd": fairness.get('demographic_parity_difference', None),
            "protected_attribute": fairness.get('protected_attribute', ''),
            "overall_selection_rate": fairness.get('overall_selection_rate', None),
            "interpretation": fairness.get('interpretation', ''),
            "library": fairness.get('library', ''),
            "n": fairness.get('n', 0),
            "by_group": fairness.get('by_group', {}),
        }

    # Disease-level accuracy from scores
    disease_accuracy = []
    perf = frameworks.get('reliable_ai', {}).get('analysis_types', {}).get('model_performance_testing', {})
    perf_metrics = perf.get('metrics', {})
    for key, val in perf_metrics.items():
        if key.endswith('_accuracy'):
            disease_accuracy.append({
                "disease": key.replace('_accuracy', '').title(),
                "accuracy": val,
            })
    disease_accuracy.sort(key=lambda x: x['accuracy'], reverse=True)

    # Calibration ECE per disease
    cal = frameworks.get('reliable_ai', {}).get('analysis_types', {}).get('calibration_analysis', {})
    cal_metrics = cal.get('metrics', {})
    calibration_data = []
    for key, val in cal_metrics.items():
        if key.endswith('_ece'):
            calibration_data.append({
                "disease": key.replace('_ece', '').title(),
                "ece": val,
            })

    return {
        "overall_score": meta.get('overall_score', 0),
        "total_frameworks": meta.get('total_frameworks', 0),
        "applicable_frameworks": meta.get('applicable_frameworks', 0),
        "analysis_date": meta.get('analysis_date', ''),
        "framework_cards": framework_cards,
        "reliability": reliability,
        "trustworthiness": trustworthiness,
        "test_summary": test_summary,
        "test_cards": test_cards,
        "fairness": fairness_summary,
        "disease_accuracy": disease_accuracy,
        "calibration_data": calibration_data,
    }


def responsible_ai_breakdown():
    """Detailed breakdown: per-framework analysis types, robustness curves,
    consistency per disease, error patterns."""
    scores = _load('results/responsible_ai_scores.json')
    tests = _load('results/responsible_ai_test_results.json')
    metrics = _load('results/responsible_ai_metrics.json')

    if not scores:
        return {"available": False}

    frameworks = scores.get('frameworks', {})

    # Per-framework analysis types
    framework_details = []
    for fid, fdata in frameworks.items():
        analyses = fdata.get('analysis_types', {})
        detail = {
            "id": fid,
            "label": fid.replace('_', ' ').title(),
            "score": fdata.get('score', 0),
            "status": fdata.get('status', 'unknown'),
            "analyses": [],
        }
        for aid, adata in analyses.items():
            detail["analyses"].append({
                "id": aid,
                "label": aid.replace('_', ' ').title(),
                "score": adata.get('score', 0),
                "method": adata.get('method', ''),
                "justification": adata.get('justification', ''),
            })
        framework_details.append(detail)

    # Robustness noise curve
    robustness_curve = []
    if tests:
        rob = tests.get('test_results', {}).get('robustness', {})
        for level, rdata in rob.get('results_by_noise', {}).items():
            robustness_curve.append({
                "noise_level": float(level),
                "accuracy": rdata.get('accuracy', 0),
                "accuracy_drop": rdata.get('accuracy_drop', 0),
                "prediction_change_rate": rdata.get('prediction_change_rate', 0),
            })
        robustness_curve.sort(key=lambda x: x['noise_level'])

    # Consistency scores per disease (from metrics)
    consistency_data = []
    if metrics:
        cons = metrics.get('reliability_analysis', {}).get('consistency_scores', {})
        for disease, cdata in cons.items():
            consistency_data.append({
                "disease": disease.title(),
                "variance": cdata.get('variance', 0),
                "status": cdata.get('status', 'unknown'),
            })

    # Robustness noise levels from metrics
    robustness_levels = []
    if metrics:
        rob = metrics.get('reliability_analysis', {}).get('robustness', {})
        for level, rdata in rob.items():
            robustness_levels.append({
                "level": level.replace('noise_', '').replace('_percent', '%'),
                "accuracy_drop": rdata.get('accuracy_drop', ''),
                "status": rdata.get('status', 'unknown'),
            })

    # Error patterns
    error_patterns = {}
    err_analysis = frameworks.get('reliable_ai', {}).get('analysis_types', {}).get('error_analysis', {})
    if err_analysis:
        error_patterns = err_analysis.get('error_patterns', {})

    return {
        "framework_details": framework_details,
        "robustness_curve": robustness_curve,
        "consistency_data": consistency_data,
        "robustness_levels": robustness_levels,
        "error_patterns": error_patterns,
    }


def responsible_ai_definitions():
    """Metric definitions for tooltip overlays."""
    return {
        "overall_score": "Weighted average across all applicable responsible AI frameworks (0-100).",
        "framework_score": "Individual framework assessment score based on quantitative analysis.",
        "reliability_score": "Measures model consistency, calibration, and robustness under perturbation.",
        "trustworthiness_score": "Assesses ground truth quality, uncertainty quantification, and decision audit trail.",
        "fairness_gate": "PASS/FAIL based on demographic parity difference (DPD < 0.2 across protected groups).",
        "demographic_parity_difference": "Absolute difference in selection rates between demographic groups. Lower is fairer.",
        "overall_selection_rate": "Fraction of patients receiving an adverse assessment across all groups.",
        "consistency_variance": "Variance in accuracy across multiple runs. Lower variance = more consistent.",
        "robustness_score": "Model accuracy retention under injected noise at various levels.",
        "calibration_ece": "Expected Calibration Error — measures how well predicted probabilities match actual outcomes.",
        "accuracy_drop": "Percentage decrease in accuracy when noise is injected into input signals.",
        "prediction_change_rate": "Fraction of predictions that change under noise perturbation.",
        "pass_rate": "Fraction of responsible AI tests that passed (tests_passed / total_tests).",
        "sla_targets": "Service-level agreement targets for accuracy, sensitivity, specificity, latency, and availability.",
        "error_patterns": "Systematic misclassification categories: borderline cases, medication effects, comorbidities.",
    }
