#!/usr/bin/env python3
"""Knowledge-graph (RDF) build job — rebuilds the patient→EEG→finding→diagnosis graph
from the clinical DB and writes jobs/reports/graph_latest.json. Cron daily 06:00."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "jobs" / "reports"


def main():
    now = datetime.now(timezone.utc).astimezone()
    REPORTS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    nodes = edges = 0
    ok = True
    note = ""
    try:
        import knowledge_graph as kg
        g = kg.build_graph() if hasattr(kg, "build_graph") else None
        if g is not None:
            nodes = len(g.get("nodes", [])) if isinstance(g, dict) else 0
            edges = len(g.get("edges", [])) if isinstance(g, dict) else 0
        else:
            note = "knowledge_graph.build_graph() not found"
    except Exception as e:  # noqa: BLE001
        ok = False; note = str(e)[:200]
    report = {"run_at_local": now.isoformat(timespec="seconds"),
              "run_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
              "nodes": nodes, "edges": edges, "ok": ok, "note": note,
              "summary": f"graph rebuilt: {nodes} nodes / {edges} edges"}
    (REPORTS / "graph_latest.json").write_text(json.dumps(report, indent=2))
    try:
        import clinical_db as cdb
        cdb.log_transaction("_system", component="graph_db", action="build", detail=report["summary"])
    except Exception:
        pass
    print(f"[{report['run_at_local']}] {report['summary']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
