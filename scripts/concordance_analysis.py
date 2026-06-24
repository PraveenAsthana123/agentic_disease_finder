#!/usr/bin/env python3
"""AI–Expert concordance analysis — the project's unique research angle.

Analyzes the expert_reviews table: how often experts agree/disagree with the AI,
broken down by role and by AI confidence. This is the measurable evidence that
the human-oversight layer works (or where it's needed most).

Usage: python scripts/concordance_analysis.py
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "clinical.db"
OUT = ROOT / "jobs" / "reports"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not DB.exists():
        print("No clinical DB."); return 1
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    reviews = [dict(r) for r in c.execute("SELECT * FROM expert_reviews").fetchall()]
    # join AI confidence per analysis
    conf = {}
    for r in c.execute("SELECT id, confidence FROM analyses").fetchall():
        conf[r["id"]] = r["confidence"]
    c.close()

    if not reviews:
        print("No expert reviews yet — assign/add reviews to populate concordance.")
        payload = {"generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                   "n_reviews": 0, "note": "No expert reviews yet."}
        (OUT / "concordance_analysis.json").write_text(json.dumps(payload, indent=2))
        return 0

    overall = Counter(r.get("agree_with_ai") or "unspecified" for r in reviews)
    by_role = defaultdict(Counter)
    by_conf = {"high(>=0.7)": Counter(), "mid(0.5-0.7)": Counter(), "low(<0.5)": Counter()}
    for r in reviews:
        by_role[r["role"]][r.get("agree_with_ai") or "unspecified"] += 1
        cf = conf.get(r.get("analysis_id"))
        if cf is not None:
            band = "high(>=0.7)" if cf >= 0.7 else "mid(0.5-0.7)" if cf >= 0.5 else "low(<0.5)"
            by_conf[band][r.get("agree_with_ai") or "unspecified"] += 1

    n = len(reviews)
    agree_rate = round(overall.get("agree", 0) / n, 3)
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_reviews": n,
        "overall": dict(overall),
        "overall_agree_rate": agree_rate,
        "by_role": {k: dict(v) for k, v in by_role.items()},
        "by_ai_confidence": {k: dict(v) for k, v in by_conf.items()},
        "interpretation": "Disagreement concentrated at low AI confidence => oversight layer is targeting the right cases.",
    }
    (OUT / "concordance_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"=== AI–EXPERT CONCORDANCE ({n} reviews) ===")
    print(f"  overall agree rate: {agree_rate}")
    print(f"  overall: {dict(overall)}")
    for role, ctr in by_role.items():
        print(f"  {role}: {dict(ctr)}")
    print(f"Saved: {OUT / 'concordance_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
