# AgenticFinder Global Approval Policy

**Policy ID**: agenticfinder-global-approval-policy
**Version**: 1.0.0
**Status**: Submitted
**Submitted**: 2026-06-24
**Classification**: Research Use Only

## Purpose

This policy defines when AgenticFinder agents, tools, MCP actions, data operations, model lifecycle actions, and clinical decision-support outputs may proceed automatically, require human approval, or must be denied.

The machine-readable policy is maintained in:

```text
config/global_approval_policy.json
```

## Default Rule

When no explicit rule matches, the action requires human approval.

Allowed actions must still write an audit record. Denied actions must write an audit record with the policy rule and reason.

## Approval Outcomes

| Decision | Meaning |
|---|---|
| allow | Action may proceed with audit logging. |
| require_human_approval | Action must be submitted to the HITL approval queue before execution. |
| deny | Action must not proceed. |

## Role Matrix

| Role | Approval Scope |
|---|---|
| Clinical Reviewer | Clinical decision support, patient reports, high-risk prediction review |
| Data Steward | Dataset ingest, data export, retention changes, de-identification exceptions |
| Model Owner | Model training, promotion, threshold changes, performance claim updates |
| Security Officer | External integrations, credential rotation, privileged tools, production access |
| Governance Lead | Policy changes, overrides, kill-switch disablement, regulatory exceptions |

## Automatic Allow

The following may proceed automatically when audit logging is available:

| Case | Conditions |
|---|---|
| Public read | `public:read`, reversible, low risk |
| Scoped read | `read:*`, no PHI/PII, low risk |

## Human Approval Required

Human approval is required for:

| Case | Required Reviewer |
|---|---|
| Clinical or patient-facing output | Clinical Reviewer |
| Write, submit, send, publish, or release actions | Relevant domain owner |
| Admin and irreversible actions | Governance Lead |
| Protected data export or external sharing | Data Steward |
| Model promotion, deployment, threshold changes, or claim updates | Model Owner |
| Budget, cost, or scope exceptions | Governance Lead |

## Deny

The policy denies actions that match destructive production patterns, including:

```text
production.*\.delete
prod-db.*\.drop
.*force.*push
admin\.dangerous_.*
```

The policy also denies requests to bypass audit logging, bypass approval, disable the kill switch without governance approval, or modify this policy without change control.

## HITL Submission

Actions requiring approval should be submitted through the existing approval path:

```text
POST /api/v1/agent-kernel/hitl/request
```

Review decisions are recorded through:

```text
POST /api/v1/agent-kernel/hitl/{approval_id}/decide
```

The approval queue is backed by `kernel_approval_queue`.

## Audit Requirements

Every policy evaluation must include:

```text
request_id
actor
agent_id
action_kind
target
scope_required
risk_band
policy_rule_id
policy_decision
policy_reason
timestamp
```

Approval records must also include:

```text
approval_id
requested_by
reviewer_role
decision
decision_reason
decided_at
sla_due_at
```

Audit retention is 2555 days unless a stricter legal or institutional rule applies.

## Change Control

Policy changes require Governance Lead approval, rollback planning, and validation before activation. Emergency overrides require a written reason and must remain auditable.
