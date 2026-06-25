#!/usr/bin/env python3
"""CatBoost model tests — real epilepsy features, subject-wise CV."""
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("catboost")
import scripts.catboost_model as cb  # noqa: E402
NPZ = ROOT / "data" / "epilepsy" / "sample" / "epilepsy_sample_100.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="no reference npz")


def test_metrics_real():
    r = cb.build()
    assert r["available"] is True
    m = r["metrics"]
    assert 0 <= m["auc"] <= 1 and 0 <= m["accuracy"] <= 1


def test_subject_wise_cv_declared():
    """CV must be subject-wise (leakage-free) — the credibility requirement."""
    r = cb.build()
    assert "subject-wise" in r["cv"]


def test_feature_importance_present():
    r = cb.build()
    assert r["top_features"]
    assert all(f["importance"] >= 0 for f in r["top_features"])


def test_comparison_to_deployed():
    r = cb.build()
    assert "catboost_auc" in r["comparison"]
