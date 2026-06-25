#!/usr/bin/env python3
"""Video Validation tests — real extracted frames on disk.

Positive: counts match real files; readable frames report dimensions.
Negative: blank/corrupt counts must be honest (sum of per-dir == summary);
validation verdict must be REVIEW (not PASS) whenever any corrupt/blank frame
or >1 resolution is present (a regression that PASSes dirty data is the bug).
"""

import glob
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.video_validation as vv  # noqa: E402

HAS_FRAMES = bool(glob.glob(str(ROOT / "data" / "frames_clean" / "*.jpg")))
pytestmark = pytest.mark.skipif(not HAS_FRAMES, reason="no frames present")


def test_report_shape():
    r = vv.validate_frames()
    assert r["available"] is True
    assert r["summary"]["total_frames"] >= 1


def test_counts_are_consistent():
    r = vv.validate_frames()
    per_dir = sum(d["n_frames"] for d in r["directories"])
    assert per_dir == r["summary"]["total_frames"]


def test_verdict_matches_findings():
    """PASS iff clean; REVIEW iff any corrupt/blank/multi-resolution (QC lock)."""
    r = vv.validate_frames()
    s = r["summary"]
    dirty = s["corrupt_frames"] > 0 or s["blank_frames"] > 0 or s["distinct_resolutions"] > 1
    assert s["validation"] == ("REVIEW" if dirty else "PASS")


def test_readable_frames_have_dimensions():
    r = vv.validate_frames()
    for d in r["directories"]:
        if d["readable"]:
            assert d["dimensions"], f"{d['directory']} readable frames must report dimensions"
