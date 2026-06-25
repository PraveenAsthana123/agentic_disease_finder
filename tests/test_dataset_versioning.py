#!/usr/bin/env python3
"""Dataset Versioning tests — real SHA-256 over on-disk artifacts.

Positive: manifest hashes real datasets + models; fingerprint is stable.
Negative: the composite fingerprint MUST be deterministic across two runs with
no file changes (a regression that makes it non-deterministic breaks lineage),
and every hash MUST be a real 64-hex SHA-256 (no placeholder/None for present
files).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.dataset_versioning as dv  # noqa: E402


def test_manifest_shape():
    m = dv.version_manifest()
    assert m["available"] is True
    assert m["summary"]["n_datasets"] >= 1
    assert m["composite_fingerprint"] is not None


def test_fingerprint_is_deterministic():
    """Same files → same fingerprint (lineage stability lock)."""
    a = dv.version_manifest()["composite_fingerprint"]
    b = dv.version_manifest()["composite_fingerprint"]
    assert a == b


def test_hashes_are_real_sha256():
    """Every present artifact gets a real 64-hex SHA-256 (no placeholder)."""
    m = dv.version_manifest()
    for art in m["datasets"] + m["models"]:
        h = art["sha256"]
        assert h is not None and len(h) == 64 and all(c in "0123456789abcdef" for c in h), art["artifact"]


def test_dataset_shapes_when_present():
    m = dv.version_manifest()
    shaped = [d for d in m["datasets"] if "shape" in d]
    # at least one dataset exposes a real numeric shape
    assert shaped, "expected at least one .npz with a readable X shape"
    for d in shaped:
        assert d["shape"]["n_samples"] > 0 and d["shape"]["n_features"] > 0
