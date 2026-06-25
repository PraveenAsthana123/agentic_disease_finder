#!/usr/bin/env python3
"""Deterministic 'what to build next' picker for the autonomous pending-completion loop.
Outputs the ordered queue of BUILDABLE pending items (excludes blocked + gated).
The loop: pick top → build → verify → commit → repeat until queue empty / blocked / stop."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

# Curated buildable queue (highest value first). Blocked/gated items excluded by design.
BUILDABLE = [
    {"id": "ictal_interictal", "title": "Ictal/interictal retrain (same-setup, removes dataset confound)", "value": "P0", "effort": "high"},
    {"id": "seizure_timeline", "title": "Seizure timeline from CHB-MIT summary/.seizures annotations", "value": "P0", "effort": "med"},
    {"id": "spike_overlay", "title": "Spike/sharp-wave overlay (threshold on filtered signal)", "value": "P0", "effort": "med"},
    {"id": "lateralization", "title": "Lateralization dashboard (L/R hemisphere asymmetry index)", "value": "P1", "effort": "med"},
    {"id": "patient_compare", "title": "Patient Comparison UI (2 patients EEG+assessments side-by-side)", "value": "P1", "effort": "med"},
    {"id": "cognitive_tests", "title": "Digital cognitive tests (Stroop/Digit-Span/Trail-Making)", "value": "P1", "effort": "med"},
    {"id": "expert_pharmacist", "title": "Expert module: Clinical Pharmacist (meds table is real)", "value": "P1", "effort": "med"},
    {"id": "expert_nurse", "title": "Expert module: Epilepsy Nurse (seizure-diary analytics exist)", "value": "P1", "effort": "med"},
    {"id": "cdm_label_val", "title": "CDM: Label/Annotation validation (κ when multi-rater)", "value": "P2", "effort": "med"},
    {"id": "expert_more", "title": "Expert modules: SLP / OT / Dietitian / Psychologist / Social Worker / Coordinator", "value": "P2", "effort": "high"},
]
BLOCKED = ["Gmail/Slack/Drive live (credentials)", "Multi-user auth/RBAC", "EMR/FHIR + device streaming"]
GATED = ["git push (operator approval, §42)"]


def main():
    as_json = "--json" in sys.argv
    if as_json:
        print(json.dumps({"buildable": BUILDABLE, "blocked": BLOCKED, "gated": GATED}, indent=2)); return 0
    print("════ NEXT BUILDABLE (autonomous loop queue) ════")
    for i, b in enumerate(BUILDABLE):
        print(f"  {i+1:2d}. [{b['value']}] {b['title']}  ({b['effort']})")
    print(f"\n▸ TOP PICK: {BUILDABLE[0]['title']}")
    print(f"\n🔒 BLOCKED (need operator): {' · '.join(BLOCKED)}")
    print(f"🔴 GATED (need go-ahead): {' · '.join(GATED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
