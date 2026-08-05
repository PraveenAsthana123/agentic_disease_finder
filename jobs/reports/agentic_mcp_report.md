# Multi-Agent (Mixture-of-Agents) + MCP Analysis Report

_Generated 2026-06-23T17:20:11-06:00_

- Pattern: **Mixture-of-Agents (7 disease agents + orchestrator, MCP protocol)**
- Agents: **7** · messages: 14 · audit entries: 15
- Consensus decision: **autism** (p=0.17)

## Agent votes
| Disease agent | Prediction | Disease-class prob |
|---|---|---|
| autism | Control | 0.17 |
| alzheimer | Control | 0.0 |
| parkinson | Control | 0.0 |
| schizophrenia | Control | 0.0 |
| epilepsy | Control | 0.0 |
| stress | Control | 0.0 |
| depression | Control | 0.0 |

## A2A summary
```json
{
  "status": "healthy",
  "issues": [
    "Low trust level (0.50)"
  ],
  "recommendations": [
    "7 isolated agents - improve connectivity",
    "Many messages missing required fields - enforce message schema",
    "No coordinator role - consider designating a coordinator"
  ]
}
```

## MCP summary
```json
{
  "status": "healthy",
  "issues": [],
  "recommendations": [
    "7 cycles detected - verify intentional and add termination conditions",
    "Investigate 1 anomalies in audit trail"
  ]
}
```

> ⚠️ Demonstrator: agents run the same input through per-disease models. Disease-class probabilities reflect the in-sample models (see leakage caveat in the accuracy report).
