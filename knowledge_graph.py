"""RDF/RDFS knowledge graph over the clinical DB for relationship analysis.

Builds triples (patient -> analysis -> prediction; patient -> assessment -> score;
expert -> reviewed -> patient; role -> relates_to entities) with rdflib + an RDFS
schema, then returns a per-role subgraph as nodes+edges (for Mermaid rendering)
and supports simple relationship queries.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent / "data" / "clinical.db"

# Which entity types each role cares about (relationship lens).
ROLE_LENS = {
    "Neurologist": ["Patient", "Analysis", "Prediction", "ExpertReview", "Seizure"],
    "EEG Technician": ["Patient", "Upload", "ChannelQuality", "Analysis"],
    "Psychiatrist": ["Patient", "Assessment", "Survey", "Comorbidity"],
    "Occupational Therapist": ["Patient", "Outcome", "Assessment"],
    "Clinical Psychologist": ["Patient", "Neuropsych", "Assessment"],
    "Radiologist": ["Patient", "MRI", "Analysis"],
    "IRB / Governance Reviewer": ["Patient", "Analysis", "ExpertReview", "Audit"],
    "IoT Engineer": ["Patient", "Device", "Gateway", "Alert"],
}


def _rows(c, q, args=()):
    try:
        return [dict(r) for r in c.execute(q, args).fetchall()]
    except Exception:
        return []


def build_graph(role: str | None = None, patient_id: str | None = None) -> dict:
    """Return {nodes, edges, schema, triples_count, mermaid} for the (role-filtered) graph."""
    if not DB.exists():
        return {"nodes": [], "edges": [], "triples_count": 0, "mermaid": "graph TD\n  A[No data]"}
    try:
        import rdflib
        from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef
        N = Namespace("http://agenticfinder.local/epilepsy#")
        g = Graph(); g.bind("eeg", N)
        # RDFS schema (classes + properties)
        for cls in ["Patient", "Analysis", "Prediction", "Assessment", "ExpertReview", "Seizure", "Upload"]:
            g.add((N[cls], RDF.type, RDFS.Class))
    except Exception:
        rdflib = None
        g = None
        N = None

    nodes, edges = {}, []

    def node(nid, label, ntype):
        nodes[nid] = {"id": nid, "label": label, "type": ntype}

    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    pat_filter = " WHERE patient_id=?" if patient_id else ""
    pa = (patient_id,) if patient_id else ()

    # Patients
    for p in _rows(c, f"SELECT * FROM patients{(' WHERE patient_id=?' if patient_id else '')}", pa)[:30]:
        pid = p["patient_id"]
        node(f"P:{pid}", f"{pid} ({p.get('disease','epilepsy')})", "Patient")
        if g is not None:
            g.add((N[f"patient_{pid}"], RDF.type, N.Patient))

    # Analyses -> predictions
    for a in _rows(c, f"SELECT * FROM analyses{pat_filter} ORDER BY id DESC LIMIT 40", pa):
        pid, aid = a.get("patient_id"), a["id"]
        node(f"A:{aid}", f"Analysis {aid}", "Analysis")
        node(f"Pred:{aid}", f"{a.get('predicted_label')} ({a.get('confidence')})", "Prediction")
        if pid:
            edges.append((f"P:{pid}", "hasAnalysis", f"A:{aid}"))
        edges.append((f"A:{aid}", "predicts", f"Pred:{aid}"))
        if g is not None and pid:
            g.add((N[f"patient_{pid}"], N.hasAnalysis, N[f"analysis_{aid}"]))

    # Assessments
    for a in _rows(c, f"SELECT * FROM assessments{pat_filter} ORDER BY id DESC LIMIT 40", pa):
        pid, aid = a.get("patient_id"), a["id"]
        node(f"As:{aid}", f"{a.get('instrument')}={a.get('score')}", "Assessment")
        if pid:
            edges.append((f"P:{pid}", "hasAssessment", f"As:{aid}"))

    # Expert reviews
    for r in _rows(c, f"SELECT * FROM expert_reviews{pat_filter} ORDER BY id DESC LIMIT 40", pa):
        pid, rid = r.get("patient_id"), r["id"]
        node(f"E:{rid}", f"{r.get('role')} ({r.get('agree_with_ai')})", "ExpertReview")
        if pid:
            edges.append((f"E:{rid}", "reviewed", f"P:{pid}"))
    c.close()

    # Role lens filter
    lens = ROLE_LENS.get(role) if role else None
    if lens:
        keep = {k for k, v in nodes.items() if v["type"] in lens}
        nodes = {k: v for k, v in nodes.items() if k in keep}
        edges = [e for e in edges if e[0] in nodes and e[2] in nodes]

    # Mermaid
    def safe(s):
        return s.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_").replace("=", "_")
    lines = ["graph LR"]
    for n in nodes.values():
        shape = f'["{n["label"]}"]' if n["type"] == "Patient" else f'("{n["label"]}")'
        lines.append(f"  {safe(n['id'])}{shape}")
    for s, rel, o in edges:
        if s in nodes and o in nodes:
            lines.append(f"  {safe(s)} -->|{rel}| {safe(o)}")
    mermaid = "\n".join(lines) if len(lines) > 1 else "graph TD\n  A[No relationships for this role/patient]"

    return {"role": role, "patient_id": patient_id,
            "nodes": list(nodes.values()), "edges": [{"from": s, "rel": r, "to": o} for s, r, o in edges],
            "triples_count": (len(g) if g is not None else 0),
            "schema": {"classes": list(ROLE_LENS.get(role, [])) or ["Patient", "Analysis", "Assessment", "ExpertReview"],
                       "properties": ["hasAnalysis", "predicts", "hasAssessment", "reviewed"]},
            "mermaid": mermaid,
            "engine": "rdflib" if g is not None else "fallback"}
