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
]
# Already built (removed from queue):
# seizure_timeline, spike_overlay (in eeg_viz), lateralization (in eeg_viz),
# patient_compare, cognitive_tests (endpoint + panel + scoring)
# expert_pharmacist, expert_nurse, expert_slp, expert_ot, expert_dietitian,
# expert_psychologist, expert_coordinator, expert_social_worker, data_archival
# cdm_label_val (Label/Annotation QC — endpoints live on :8010)
# verbal_fluency (Verbal Fluency FAS+Category dashboard — 4 endpoints + panel)
# neuro_scales (Clinical Scales dashboard — catalog endpoint + 23 scales + Next.js page)
# expert_dashboards (Montage Comparison + Localization + False Alarm Review — ExpertDashboard.jsx + nav wired)
# dataset_validation (Dataset Validation Dashboard — real clinical.db + CHB-MIT validation, endpoint /api/data-manager/dataset-validation verified 200)
# captum_lime_xai (Captum IG+FA + LIME endpoints verified 200 — registry updated cataloged→built)
# torchmetrics_deepchecks (TorchMetrics + Deepchecks dashboards + endpoints verified 200)
# icalabel_ge (ICLabel + Great Expectations dashboards verified 200)
# aif360_bias (AIF360 bias detection dashboard — 3 endpoints verified 200, registry cataloged→built)
# torcheeg (TorchEEG dashboard — 5 transforms + EEGNet-Mini classifier, 3 endpoints verified 200)
# inference_gpu (Inference/GPU Dashboard — real nvidia-smi + model scan + system info, 3 endpoints verified 200)
# medication_dashboard (Medication Dashboard — 6 endpoints + Next.js page + nav wired, real clinical.db prescriptions/schedule/adherence/warnings/side-effects)
# knowledge_graph (Knowledge Graph Dashboard — real clinical.db + ChromaDB entity-relationship graph, 81 nodes 191 edges, 3 endpoints verified 200)
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
