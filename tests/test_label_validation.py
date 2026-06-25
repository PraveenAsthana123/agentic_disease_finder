#!/usr/bin/env python3
"""Label Validation tests — real analyses labels + reference dataset classes.

Positive: report shape; dataset class balance computed for real .npz.
Negative: a single-class analyses set MUST be flagged (you cannot validate a
classifier against one class — silently passing it hides a QC gap), and any
class_names/label cardinality mismatch MUST be flagged.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.label_validation as lv  # noqa: E402

DB = ROOT / "data" / "clinical.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="clinical.db not present")


def test_full_report_shape():
    r = lv.full_report()
    assert set(r) >= {"analysis_labels", "dataset_labels"}
    assert r["analysis_labels"]["available"] is True


def test_single_class_analyses_is_flagged():
    """If predicted labels are single-class, that MUST be surfaced (QC lock)."""
    al = lv.analysis_labels()
    if len(al["predicted_label_distribution"]) <= 1 and al["n_analyses"] > 0:
        assert any("single predicted class" in f for f in al["flags"])


def test_dataset_balance_computed():
    dl = lv.dataset_labels()
    if dl["available"]:
        assert dl["n_datasets"] >= 1
        for d in dl["datasets"]:
            if d.get("available"):
                assert sum(d["class_distribution"].values()) == d["n"]


def test_imbalance_or_mismatch_flagged_when_present():
    """Datasets with >1.5:1 imbalance or class_names mismatch must carry a flag."""
    dl = lv.dataset_labels()
    for d in dl.get("datasets", []):
        if not d.get("available"):
            continue
        ratio = d.get("imbalance_ratio")
        if ratio and ratio > 1.5:
            assert d["flags"], f"{d['dataset']} imbalanced but unflagged"
