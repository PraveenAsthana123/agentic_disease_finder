# 🧭 Advisor — issues you may not be aware of (2026-08-09 22:34 MDT)

**5 findings**

- **[P1] model** — Drift monitor reports SEVERE drift (live features vs training distribution)
    ↳ Retrain on same-setup ictal/interictal data; until then trust confidence only with human sign-off.
- **[P2] frontend** — Dev server :3003 not reachable (http=000)
    ↳ Run: cd frontend && npm run dev -- --port 3003 (or operator opens the UI).
- **[P2] data** — Only 4 CHB-MIT subjects on disk
    ↳ Download more PhysioNet subjects (10-20) for stronger cross-patient claims.
- **[P2] security** — No multi-user auth / RBAC (single-operator mode)
    ↳ Fine for research/dev; required before multi-clinician or PHI deployment (see §47.6).
- **[P3] git** — 717 commit(s) unpushed
    ↳ safe_push.sh auto-pushes on shared repos; or push manually.
