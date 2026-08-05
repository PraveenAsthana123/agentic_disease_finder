# 🧭 Advisor — issues you may not be aware of (2026-06-25 10:40 MDT)

**7 findings**

- **[P1] model** — Drift monitor reports SEVERE drift (live features vs training distribution)
    ↳ Retrain on same-setup ictal/interictal data; until then trust confidence only with human sign-off.
- **[P1] model** — Model has dataset-confound caveat (acc 0.8718, control=motor-imagery)
    ↳ Confidence partly reflects dataset, not only epilepsy. Use ictal/interictal data for clinical claims.
- **[P1] backend** — 6 HTTP-500 since last restart
    ↳ Check jobs/logs/backend.log; wrap NaN/serialization in _json_safe.
- **[P2] data** — Only 4 CHB-MIT subjects on disk
    ↳ Download more PhysioNet subjects (10-20) for stronger cross-patient claims.
- **[P2] security** — No multi-user auth / RBAC (single-operator mode)
    ↳ Fine for research/dev; required before multi-clinician or PHI deployment (see §47.6).
- **[P3] data** — MRI coverage very low (2.5%)
    ↳ Expected — DICOM not ingested. Note as limitation, not a bug.
- **[P3] git** — 2 commit(s) unpushed
    ↳ safe_push.sh auto-pushes on shared repos; or push manually.
