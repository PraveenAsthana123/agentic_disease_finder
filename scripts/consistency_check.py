#!/usr/bin/env python3
"""Scheduled prediction-consistency guard — verifies the SHAP explanation explains the
SAME model that classify()/the Trust Panel uses (the bug fixed in commit 5c2b039).
For each recent analysis: trust-panel confidence == SHAP confidence (within tol).
Catches model-bundle drift / extractor skew regressions automatically.
Writes jobs/reports/consistency_latest.json."""
import json, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
TOL = 0.02


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main():
    import clinical_db as cdb
    import shap_explain as sx

    c = sqlite3.connect(str(ROOT / "data" / "clinical.db"))
    ids = [r[0] for r in c.execute(
        "SELECT id FROM analyses WHERE result_json LIKE '%features%' ORDER BY id DESC LIMIT 5").fetchall()]
    checks, mismatches = [], 0
    for aid in ids:
        try:
            tp = cdb.build_trust_panel(analysis_id=aid)
            sh = sx.explain(analysis_id=aid)
            tc = tp.get("confidence")
            sc = sh.get("confidence")
            ok = (tc is not None and sc is not None and abs(float(tc) - float(sc)) <= TOL
                  and tp.get("ai_prediction") == sh.get("predicted_label"))
            if not ok:
                mismatches += 1
            checks.append({"analysis_id": aid, "trust_conf": tc, "shap_conf": sc,
                           "trust_label": tp.get("ai_prediction"), "shap_label": sh.get("predicted_label"),
                           "consistent": ok})
        except Exception as e:  # noqa: BLE001
            mismatches += 1
            checks.append({"analysis_id": aid, "error": str(e)[:120], "consistent": False})

    verdict = "PASS" if mismatches == 0 else "FAIL"
    report = {"run_at": _now(), "n_checked": len(checks), "mismatches": mismatches,
              "verdict": verdict, "tolerance": TOL, "checks": checks,
              "invariant": "trust-panel confidence == SHAP confidence (same model bundle) — guards commit 5c2b039",
              "note": "FAIL = SHAP explains a different model than the Trust Panel shows (bundle/extractor drift)."}
    out = ROOT / "jobs" / "reports" / "consistency_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    try:
        cdb.log_transaction("_system", component="consistency", action="check",
                            detail=f"{verdict} {len(checks)-mismatches}/{len(checks)} consistent")
    except Exception:
        pass
    print(f"CONSISTENCY: {verdict} — {len(checks)-mismatches}/{len(checks)} predictions match their explanation")
    for ch in checks:
        print(f"  #{ch['analysis_id']}: trust={ch.get('trust_conf')} shap={ch.get('shap_conf')} "
              f"{'✓' if ch['consistent'] else '✗ ' + ch.get('error','MISMATCH')}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
