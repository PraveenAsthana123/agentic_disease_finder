#!/usr/bin/env python3
"""Multi-agent (Mixture-of-Agents) + MCP analysis job.

Runs all 7 disease models as independent "agents" coordinated by an
orchestrator (MCP-style protocol), on one real EEG feature vector. Captures
the agent messages + decision audit trail, then feeds them through the
project's A2A (agent-to-agent) and MCP (model-control-protocol) analyzers
(agentic_analysis/) to produce a governance report.

Pattern: Mixture-of-Agents (§64.43 #12) — each disease agent votes; the
orchestrator aggregates by highest disease-class probability.

Usage: python scripts/test_agentic_mcp.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic_analysis.a2a_analysis import A2AAnalysisReportGenerator
from agentic_analysis.mcp_analysis import MCPAnalysisReportGenerator
MODELS = ROOT / "models"
DATA = ROOT / "data"
OUT = ROOT / "jobs" / "reports"
DISEASES = ["alzheimer", "parkinson", "schizophrenia", "epilepsy", "autism", "stress", "depression"]


def _now():
    return datetime.now(timezone.utc).astimezone()


def _ts(i):
    return (_now() + timedelta(milliseconds=i * 40)).isoformat()


def load_feature_vector() -> np.ndarray:
    """One real 47-feature vector from the epilepsy sample (any disease row works)."""
    d = np.load(DATA / "epilepsy" / "sample" / "epilepsy_50rows.npz")
    return d["X"][0]


def run_ensemble(features: np.ndarray) -> dict:
    """Orchestrator dispatches to each disease agent; collect votes + telemetry."""
    messages, audit, agent_actions, interactions = [], [], [], []
    requests, circuit_events, state_transitions = [], [], []
    votes = []
    step = 0

    for dz in DISEASES:
        mp = MODELS / f"{dz}_model.joblib"
        if not mp.exists():
            continue
        bundle = joblib.load(mp)
        model = bundle["model"]
        classes = bundle.get("class_names", ["Control", dz.title()])
        agent = f"agent_{dz}"

        # Orchestrator -> agent: dispatch request (+ MCP telemetry)
        # MCP-compliant message schema: type + timestamp + payload.
        messages.append({"sender": "orchestrator", "receiver": agent, "type": "dispatch",
                         "timestamp": _ts(step), "payload": {"disease": dz, "op": "predict"}}); step += 1
        audit.append({"timestamp": _ts(step), "actor": "orchestrator", "action": "dispatch",
                      "resource": dz, "outcome": "dispatched", "target": agent, "request_id": f"req-{dz}"}); step += 1
        requests.append({"timestamp": _ts(step), "client": "orchestrator", "endpoint": "predict",
                         "service": agent, "status": "allowed"})
        circuit_events.append({"timestamp": _ts(step), "service": agent, "state": "closed", "failures": 0})
        state_transitions.append({"timestamp": _ts(step), "from": "idle", "to": "dispatched", "agent": agent})

        X = features.reshape(1, -1)
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        disease_prob = float(proba[1]) if proba is not None and len(proba) > 1 else float(pred)
        label = classes[pred] if pred < len(classes) else str(pred)

        # Agent -> orchestrator: result message + action + audit
        messages.append({"sender": agent, "receiver": "orchestrator", "type": "result",
                         "timestamp": _ts(step), "payload": {"label": label, "p": round(disease_prob, 4)}}); step += 1
        agent_actions.append({"agent": agent, "action": "predict", "resource": dz,
                              "result": label, "confidence": round(disease_prob, 4)})
        interactions.append({"source": agent, "target": "orchestrator", "trust": round(disease_prob, 4),
                             "outcome": "success"})
        audit.append({"timestamp": _ts(step), "actor": agent, "action": "predict",
                      "resource": dz, "outcome": label, "confidence": round(disease_prob, 4),
                      "request_id": f"req-{dz}"}); step += 1

        votes.append({"disease": dz, "label": label, "disease_prob": round(disease_prob, 4)})

    # Mixture-of-Agents aggregation: highest disease-class probability wins.
    ranked = sorted(votes, key=lambda v: v["disease_prob"], reverse=True)
    decision = ranked[0] if ranked else None
    audit.append({"timestamp": _ts(step), "actor": "orchestrator", "action": "aggregate",
                  "resource": decision["disease"] if decision else "none",
                  "target": decision["disease"] if decision else None,
                  "outcome": "consensus", "request_id": "req-final"})

    return {
        "votes": votes, "ranked": ranked, "decision": decision,
        "messages": messages, "audit": audit,
        "agent_actions": agent_actions, "interactions": interactions,
        "requests": requests, "circuit_events": circuit_events,
        "state_transitions": state_transitions,
        "agent_activities": {f"agent_{v['disease']}": [{"action": "predict"}] for v in votes},
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = load_feature_vector()
    ens = run_ensemble(feats)

    a2a = A2AAnalysisReportGenerator().generate_full_report(
        messages=ens["messages"], agent_actions=ens["agent_actions"],
        interactions=ens["interactions"], agent_activities=ens["agent_activities"])
    mcp = MCPAnalysisReportGenerator().generate_full_report(
        interactions=ens["messages"], audit_entries=ens["audit"],
        requests=ens["requests"], circuit_events=ens["circuit_events"],
        state_transitions=ens["state_transitions"])

    report = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "pattern": "Mixture-of-Agents (7 disease agents + orchestrator, MCP protocol)",
        "n_agents": len(ens["votes"]),
        "decision": ens["decision"],
        "votes": ens["ranked"],
        "a2a_summary": a2a.get("summary", {}),
        "mcp_summary": mcp.get("summary", {}),
        "telemetry": {"messages": len(ens["messages"]), "audit_entries": len(ens["audit"])},
    }
    (OUT / "agentic_mcp_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n=== Multi-Agent + MCP Job — {report['generated_at']} ===")
    print(f"Pattern: {report['pattern']}")
    print(f"Agents: {report['n_agents']} | messages: {report['telemetry']['messages']} | audit: {report['telemetry']['audit_entries']}")
    print("\nAgent votes (disease-class probability):")
    for v in ens["ranked"]:
        mark = "  <-- consensus" if v is ens["decision"] else ""
        print(f"  {v['disease']:<15} {v['label']:<12} p={v['disease_prob']}{mark}")
    print(f"\nA2A summary: {report['a2a_summary'].get('status', a2a.get('summary'))}")
    print(f"MCP summary: {report['mcp_summary'].get('status', mcp.get('summary'))}")

    lines = [
        "# Multi-Agent (Mixture-of-Agents) + MCP Analysis Report",
        "", f"_Generated {report['generated_at']}_", "",
        f"- Pattern: **{report['pattern']}**",
        f"- Agents: **{report['n_agents']}** · messages: {report['telemetry']['messages']} · audit entries: {report['telemetry']['audit_entries']}",
        f"- Consensus decision: **{ens['decision']['disease'] if ens['decision'] else '—'}** "
        f"(p={ens['decision']['disease_prob'] if ens['decision'] else '—'})",
        "", "## Agent votes",
        "| Disease agent | Prediction | Disease-class prob |", "|---|---|---|",
    ]
    for v in ens["ranked"]:
        lines.append(f"| {v['disease']} | {v['label']} | {v['disease_prob']} |")
    lines += [
        "", "## A2A summary", "```json", json.dumps(report["a2a_summary"], indent=2, default=str), "```",
        "", "## MCP summary", "```json", json.dumps(report["mcp_summary"], indent=2, default=str), "```",
        "",
        "> ⚠️ Demonstrator: agents run the same input through per-disease models. "
        "Disease-class probabilities reflect the in-sample models (see leakage caveat in the accuracy report).",
        "",
    ]
    (OUT / "agentic_mcp_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {OUT / 'agentic_mcp_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
