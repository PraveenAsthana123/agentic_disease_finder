# 🧭 Advisor — issues you may not be aware of (2026-08-31 10:55 MDT)

**4 findings**

- **[P1] model** — Drift monitor reports SEVERE drift (live features vs training distribution)
    ↳ Retrain on same-setup ictal/interictal data; until then trust confidence only with human sign-off.
- **[P2] frontend** — Dev server :3003 not reachable (http=000)
    ↳ Run: cd frontend && npm run dev -- --port 3003 (or operator opens the UI).
- **[P2] data** — Only 4 CHB-MIT subjects on disk
    ↳ Download more PhysioNet subjects (10-20) for stronger cross-patient claims.
- **[P2] security** — No multi-user auth / RBAC (single-operator mode)
    ↳ Fine for research/dev; required before multi-clinician or PHI deployment (see §47.6).
