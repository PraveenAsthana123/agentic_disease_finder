# Responsible AI (RAI) Deep Audit Framework

## 5-Pillar Healthcare AI Governance Model

This comprehensive audit framework provides 100+ audit dimensions across 5 pillars for ensuring responsible, safe, and compliant AI in clinical/healthcare settings.

---

## Pillar 1: Data Responsibility & PHI Governance

### Overview
Ensures proper handling of Protected Health Information (PHI), data quality, and regulatory compliance.

| Audit Dimension | What to Assess | How to Assess | Tools | Framework | Evidence | Gap Indicators | Risk | Remediation |
|-----------------|----------------|---------------|-------|-----------|----------|----------------|------|-------------|
| **Data Inventory** | All data fields used by app & model | Data flow walkthrough + schema review | Manual worksheet, ERD | HIPAA Minimum Necessary | Data dictionary, schema | No complete inventory | HIGH | Create centralized data inventory |
| **Data Lineage** | Source → storage → model → output | Data Flow Diagram (DFD) | Draw.io, Lucidchart | NIST Privacy Framework | DFD diagram | "Data comes from many places" | HIGH | Formal lineage documentation |
| **PHI/PII Classification** | Identify PHI, PII, quasi-identifiers | Field-level tagging | Presidio, manual tagging | HIPAA Safe Harbor | PHI mapping table | No PHI tags | HIGH | PHI classification matrix |
| **Consent Management** | Patient consent & purpose limitation | Consent traceability review | Policy review | HIPAA, PHIPA | Consent forms, policy | Assumed consent | HIGH | Align consent with AI usage |
| **Purpose Limitation** | Data used only for stated purpose | Use-case vs data mapping | Manual | GDPR principle | Use-case justification | Training reuse unclear | MEDIUM | Restrict secondary use |
| **Data Minimization** | Only required features used | Feature necessity review | pandas profiling | Privacy by Design | Feature list | Over-collection | MEDIUM | Remove unused features |
| **De-identification** | PHI removed before training | Re-ID risk review | Presidio, regex | HIPAA De-ID | Sample dataset | PHI in training | HIGH | De-ID pipeline |
| **Free-Text Risk** | Clinical notes handling | Text inspection | NLP scan | HIPAA | Sample notes | Raw notes used | HIGH | PHI scrubbing |
| **Encryption** | Data at rest & in transit | Security configuration review | Cloud KMS | ISO 27001 | Architecture screenshots | Weak encryption | HIGH | Enforce encryption |
| **Access Control** | Who can access what | RBAC review | IAM console | Least Privilege | Access matrix | Broad access | HIGH | Role-based access |
| **Logging & Telemetry** | PHI leakage in logs | Log sampling | SIEM, logs | OWASP ML Top 10 | Log samples | PHI in logs | HIGH | Mask logs |
| **Retention Policy** | Data retention & deletion | Policy check | Manual | HIPAA retention | Retention policy | No deletion plan | MEDIUM | Define retention limits |
| **Vendor/Sub-processors** | Third-party data sharing | Contract review | Manual | HIPAA BAA | Vendor list | Unknown vendors | HIGH | Vendor governance |
| **Representativeness** | Population coverage | Slice analysis | pandas, Evidently | Fairness by design | Slice metrics | Skewed data | HIGH | Balance datasets |
| **Missing Data Bias** | Missingness across groups | Missingness heatmap | pandas | Data quality | Profiling report | Uneven missingness | MEDIUM | Imputation strategy |
| **Site/Hospital Bias** | Multi-site variability | Site-wise analysis | Evidently | Domain shift | Site metrics | Single-site data | MEDIUM | Multi-site validation |
| **Incident Response** | Data breach readiness | IR plan review | Manual | HIPAA Security Rule | IR document | No IR plan | HIGH | Define IR workflow |
| **Audit Readiness** | Evidence traceability | Evidence completeness check | Checklist | Regulator readiness | Evidence pack | Verbal-only | HIGH | Evidence repository |

### Pillar 1 Scoring
- **Total Dimensions**: 18
- **High Risk**: 14 (78%)
- **Medium Risk**: 4 (22%)
- **Critical Focus**: PHI Classification, De-identification, Encryption, Access Control

---

## Pillar 2: Model Responsibility

### Overview
Ensures model quality, fairness, explainability, and robustness for clinical decision support.

| Audit Dimension | What to Assess | How to Assess | Tools | Framework | Evidence | Gap Indicators | Risk | Remediation |
|-----------------|----------------|---------------|-------|-----------|----------|----------------|------|-------------|
| **Model Purpose & Scope** | Intended clinical use & limits | Use-case vs model capability review | Workshop | FDA SaMD (intended use) | Use-case doc | "General purpose" claims | HIGH | Define narrow intended use |
| **Model Selection Rationale** | Why this model/architecture | Comparative benchmark review | Notebooks | Model governance | Benchmark results | "It worked best" | MEDIUM | Document trade-offs |
| **Training Data–Model Fit** | Data supports clinical task | Task–data alignment check | Profiling | ML validation | Data summary | Weak label quality | HIGH | Improve labels/data |
| **Performance Metrics** | Beyond accuracy | Metric coverage review | sklearn | Clinical relevance | Metric report | Accuracy only | HIGH | Add safety metrics |
| **Fairness Metrics** | Group-wise performance | Slice analysis by demographics | Fairlearn | Fair ML | Fairness report | No slicing | HIGH | Add fairness KPIs |
| **Threshold Strategy** | Decision thresholds per group | ROC/PR analysis | sklearn | Risk-based thresholds | Threshold doc | Single threshold | MEDIUM | Group-aware thresholds |
| **Bias Mitigation** | Mitigation attempted? | Technique review | Reweighing | Fairness mitigation | Experiment logs | No mitigation | HIGH | Apply mitigation |
| **Explainability (Global)** | What drives predictions | Global feature importance | SHAP | Explainable AI | Plots/screens | None | HIGH | Add global XAI |
| **Explainability (Local)** | Why this decision | Per-case explanation | SHAP/LIME | Right to explanation | Case examples | "Not available" | HIGH | Add local XAI |
| **Clinician Interpretability** | Human-understandable? | Clinician review | UI review | Human-centered AI | Feedback notes | Too technical | MEDIUM | Simplify explanations |
| **Human-in-the-Loop** | Override capability | Workflow inspection | UI/SOP | Safety by design | SOP/screens | No override | HIGH | Add HITL |
| **Confidence & Uncertainty** | Model confidence exposed | Calibration check | Reliability plots | Safe AI | Calibration plots | Overconfident | HIGH | Calibrate model |
| **Error Analysis** | Failure modes known | Error clustering | Confusion analysis | Robust ML | Error report | Unknown failures | MEDIUM | Error taxonomy |
| **Robustness (OOD)** | Out-of-distribution handling | Stress tests | Evidently | Robust AI | OOD tests | No OOD checks | HIGH | Add OOD detection |
| **Adversarial Risk** | Input manipulation | Threat modeling | Manual | OWASP ML Top 10 | Threat notes | Ignored | MEDIUM | Input validation |
| **Model Drift Readiness** | Degradation detection | Drift plan review | Evidently | MLOps | Monitoring plan | One-time eval | HIGH | Continuous monitoring |
| **Retraining Policy** | When/how retrain | Policy check | SOP | Model lifecycle | Retrain policy | Ad hoc retrain | MEDIUM | Formal retrain rules |
| **Versioning & Rollback** | Safe rollback exists | MLOps review | MLflow | Change mgmt | Version logs | No rollback | HIGH | Version control |
| **Auditability** | Decision traceable | Lineage to decision | Logs | Reg readiness | Audit trail | No trace | HIGH | Decision logs |

### Pillar 2 Scoring
- **Total Dimensions**: 19
- **High Risk**: 14 (74%)
- **Medium Risk**: 5 (26%)
- **Critical Focus**: Fairness, Explainability, Human-in-the-Loop, Calibration

---

## Pillar 3: Output Responsibility & Clinical Safety

### Overview
Ensures safe, interpretable, and actionable AI outputs for clinical decision support.

| Audit Dimension | What to Assess | How to Assess | Tools | Framework | Evidence | Gap Indicators | Risk | Remediation |
|-----------------|----------------|---------------|-------|-----------|----------|----------------|------|-------------|
| **Decision Role** | Advisory vs autonomous output | Workflow walkthrough | UI review | FDA SaMD | Screenshots | Autonomous decisions | HIGH | Enforce advisory-only |
| **Human-in-the-Loop (HITL)** | Mandatory clinician review | SOP & UI check | SOP review | Safety by design | SOP, UI | No override | HIGH | Add clinician approval |
| **Override Logging** | Overrides tracked & reviewed | Log inspection | Logs, DB | Auditability | Override logs | Overrides not logged | HIGH | Log & review overrides |
| **Confidence Disclosure** | Confidence shown to users | UI + calibration review | Reliability plots | Safe AI | UI screenshots | No confidence | HIGH | Expose confidence bands |
| **Uncertainty Handling** | Low-confidence behavior | Scenario testing | Test cases | Risk-based AI | Test results | No fallback | HIGH | Fallback to human |
| **Thresholds & Escalation** | Risk-based thresholds | Threshold review | ROC/PR | Clinical risk mgmt | Threshold doc | Single threshold | MEDIUM | Tiered thresholds |
| **Harm Scenarios** | Known failure harms | Hazard analysis | HAZOP-lite | ISO 14971 | Hazard log | No harm analysis | HIGH | Create harm register |
| **Safety Guardrails** | Prohibited outputs | Rules & filters | Rule engine | Clinical safety | Rules config | No guardrails | HIGH | Add hard constraints |
| **Contraindications** | Unsafe recommendations blocked | Rules review | Rule engine | Clinical guidelines | Rule list | Missing blocks | HIGH | Encode contraindications |
| **Explanation at Output** | Why this output now | Per-output XAI | SHAP/LIME | Right to explanation | Case examples | Generic text | MEDIUM | Contextual explanations |
| **User Comprehension** | Clinician/patient understanding | Usability testing | Surveys | Human-centered AI | Feedback | Confusing outputs | MEDIUM | Simplify language |
| **Actionability** | Clear next steps | UX review | UI review | Clinical workflow | Screens | Ambiguous actions | MEDIUM | Structured actions |
| **Alert Fatigue** | Alert frequency & relevance | Alert analysis | Logs | Patient safety | Alert stats | Excess alerts | MEDIUM | Alert tuning |
| **False Positives Risk** | Overtreatment risk | Error analysis | Confusion matrix | Clinical safety | Error report | High FP rate | HIGH | Adjust thresholds |
| **False Negatives Risk** | Missed diagnosis risk | Sensitivity review | Metrics | Clinical safety | Sensitivity report | High FN rate | HIGH | Safety-first tuning |
| **Edge Cases** | Rare populations | Case review | Case library | Equity | Case list | Ignored edge cases | HIGH | Add edge tests |
| **Temporal Safety** | Time-based misuse | Workflow review | SOP | Clinical ops | SOP | Delayed misuse | MEDIUM | Time validity checks |
| **Explainability Consistency** | Stable explanations | Consistency test | XAI checks | Trustworthy AI | Samples | Inconsistent | MEDIUM | Stabilize XAI |
| **Output Logging** | Full decision trace | Log review | SIEM | Reg readiness | Logs | Partial logs | HIGH | End-to-end logging |
| **Post-Decision Review** | Retrospective safety review | Governance check | Manual | Clinical QA | Review minutes | No reviews | MEDIUM | Monthly reviews |

### Pillar 3 Scoring
- **Total Dimensions**: 20
- **High Risk**: 13 (65%)
- **Medium Risk**: 7 (35%)
- **Critical Focus**: HITL, Confidence, Harm Scenarios, False Negative Risk

---

## Pillar 4: Monitoring & Drift

### Overview
Ensures continuous monitoring, drift detection, and operational excellence in production.

| Audit Dimension | What to Assess | How to Assess | Tools | Framework | Evidence | Gap Indicators | Risk | Remediation |
|-----------------|----------------|---------------|-------|-----------|----------|----------------|------|-------------|
| **Monitoring Coverage** | What is monitored in prod | Monitoring architecture review | Dashboards | MLOps best practice | Monitoring map | "We monitor uptime only" | HIGH | Expand to ML health |
| **Data Drift** | Input distribution changes | Drift checks per feature | Evidently, custom | Robust AI | Drift report | No drift metrics | HIGH | Add PSI/KS drift |
| **Concept Drift** | Relationship changes (X→Y) | Performance decay detection | Eval pipeline | Safe AI | Trend charts | No prod evaluation | HIGH | Scheduled re-eval |
| **Performance Monitoring** | Ongoing metrics in prod | Shadow eval + sampling | MLflow, logs | Clinical QA | KPI trends | One-time validation | HIGH | Continuous KPIs |
| **Bias Drift** | Fairness changing over time | Slice monitoring | Fairlearn + logs | Fair AI | Bias drift charts | No group metrics | HIGH | Track per cohort |
| **Calibration Drift** | Confidence reliability over time | Reliability analysis | sklearn plots | Safety | Calibration plots | Overconfident outputs | HIGH | Recalibration schedule |
| **Alerting Rules** | When alerts trigger | Threshold review | Pager, email | Ops readiness | Alert rules | No thresholds | MEDIUM | Define thresholds |
| **Alert Routing** | Who gets alerts | On-call workflow | Runbooks | ITIL/ops | On-call doc | No owner | HIGH | Define RACI + on-call |
| **Incident Response** | AI incidents handled | IR tabletop test | Playbook | HIPAA security | IR plan | "We'll investigate" | HIGH | Formal AI IR playbook |
| **Rollback Capability** | Safe rollback available | Deployment pipeline review | CI/CD | Change mgmt | Rollback proof | No rollback | HIGH | Versioned rollback |
| **Model Versioning** | Model + data versions tracked | Artifact registry review | MLflow/DVC | Auditability | Registry screenshots | No registry | HIGH | Model registry |
| **Feature Store Controls** | Feature consistency | Training/serving skew check | Feature store logs | MLOps | Skew tests | Training ≠ serving | HIGH | Feature parity tests |
| **Logging Completeness** | Inputs/outputs traced | Log sampling | SIEM | Reg readiness | Log samples | Missing trace fields | HIGH | End-to-end trace logs |
| **PHI-safe Monitoring** | Logs avoid PHI leakage | DLP scan of logs | DLP/Presidio | Privacy | DLP report | PHI in logs | HIGH | Mask/redact logs |
| **Feedback Loop** | Clinician feedback captured | Workflow inspection | UI + ticketing | Human-centered AI | Feedback records | No feedback channel | MEDIUM | Add feedback intake |
| **Ground Truth Pipeline** | Outcome labels obtained | Label pipeline review | ETL | Clinical validation | Label process | No label plan | HIGH | Define GT collection |
| **Retraining Triggers** | When retrain happens | Policy review | SOP | Lifecycle gov | Retrain criteria | Ad hoc retrain | MEDIUM | Define triggers |
| **Re-Validation Gate** | Re-approval before release | Release gate review | CI checks | Governance | Release checklist | No gate | HIGH | Add approval gates |
| **Post-Deploy Audits** | Regular reviews | Monthly/quarterly review | Reports | Governance | Review minutes | No cadence | MEDIUM | Scheduled audits |
| **KPI Ownership** | Single accountable owner | RACI validation | RACI | Accountability | RACI doc | Shared ownership | HIGH | Assign accountable owner |

### Pillar 4 Scoring
- **Total Dimensions**: 20
- **High Risk**: 16 (80%)
- **Medium Risk**: 4 (20%)
- **Critical Focus**: Drift Detection, Incident Response, Rollback, Ground Truth

---

## Pillar 5: Governance & Compliance

### Overview
Ensures organizational accountability, regulatory compliance, and sustainable AI governance.

| Audit Dimension | What to Assess | How to Assess | Tools | Framework | Evidence | Gap Indicators | Risk | Remediation |
|-----------------|----------------|---------------|-------|-----------|----------|----------------|------|-------------|
| **AI Governance Structure** | Formal AI governance exists | Org + role walkthrough | RACI | ISO/IEC 42001 | Org chart | No AI owner | HIGH | Create AI governance body |
| **Accountability** | Single accountable owner | RACI validation | RACI matrix | Accountability principle | RACI doc | Shared ownership | HIGH | Assign accountable exec |
| **Intended Use Definition** | Clear clinical scope | Intended-use review | Checklist | FDA SaMD | Intended use doc | Over-broad scope | HIGH | Narrow use claims |
| **Risk Classification** | AI risk tier defined | Risk scoring | Risk register | ISO 14971 | Risk score | No risk tier | HIGH | Risk-based governance |
| **Regulatory Mapping** | Laws mapped to controls | Regulation-to-control map | Compliance matrix | HIPAA, PHIPA, GDPR | Mapping table | Assumed compliance | HIGH | Formal mapping |
| **Model Approval** | Approval before prod | Gate review | Workflow tool | Change mgmt | Approval records | Informal approval | HIGH | Formal sign-off |
| **Model Card** | Model documented | Artifact review | Model card | Model cards standard | Model card | Missing/incomplete | HIGH | Create model card |
| **Data Sheet** | Dataset documented | Artifact review | Data sheet | Datasheets for datasets | Data sheet | None exists | HIGH | Create datasheet |
| **Risk Register** | AI risks logged | Register review | Risk log | Enterprise risk mgmt | Risk register | Risks undocumented | HIGH | Maintain AI risk log |
| **Bias Register** | Known bias risks tracked | Bias log review | Bias register | Fair AI | Bias log | No bias register | HIGH | Bias tracking |
| **Explainability Evidence** | XAI documented | Artifact check | SHAP reports | Explainable AI | XAI evidence | Verbal only | MEDIUM | Persist XAI artifacts |
| **Clinical Validation** | Clinician sign-off | Validation review | Meeting minutes | Clinical governance | Sign-off docs | No validation | HIGH | Formal validation |
| **Change Management** | Model changes governed | Change review | Change log | ITIL | Change log | Silent changes | HIGH | Change approvals |
| **Version Control** | Model/data versions | Registry review | MLflow/DVC | Auditability | Version logs | No versioning | HIGH | Enforce versioning |
| **Third-Party Models** | Vendor AI governed | Vendor risk review | Vendor assessment | Third-party risk | Vendor reports | Blind trust | HIGH | Vendor governance |
| **Training Records** | Team trained on RAI | Training audit | LMS | Responsible AI | Training logs | No training | MEDIUM | RAI training program |
| **Incident Escalation** | AI incidents escalated | IR flow check | Runbook | HIPAA security | IR playbook | Undefined escalation | HIGH | Define escalation |
| **Audit Trail** | End-to-end traceability | Trace sample | Logs | Reg readiness | Trace logs | Partial trace | HIGH | Complete audit trail |
| **Review Cadence** | Regular RAI reviews | Calendar review | Governance calendar | ISO 42001 | Review schedule | No cadence | MEDIUM | Quarterly reviews |
| **Decommission Policy** | Model retirement | Policy review | SOP | Lifecycle mgmt | Decom policy | No retirement plan | MEDIUM | Define sunset plan |

### Pillar 5 Scoring
- **Total Dimensions**: 20
- **High Risk**: 16 (80%)
- **Medium Risk**: 4 (20%)
- **Critical Focus**: Governance Structure, Model Card, Risk Register, Audit Trail

---

## Summary Statistics

| Pillar | Dimensions | High Risk | Medium Risk | Focus Areas |
|--------|------------|-----------|-------------|-------------|
| **1. Data Responsibility** | 18 | 14 (78%) | 4 (22%) | PHI, De-ID, Encryption |
| **2. Model Responsibility** | 19 | 14 (74%) | 5 (26%) | Fairness, XAI, HITL |
| **3. Output Responsibility** | 20 | 13 (65%) | 7 (35%) | Safety, Confidence, Harm |
| **4. Monitoring & Drift** | 20 | 16 (80%) | 4 (20%) | Drift, IR, Rollback |
| **5. Governance & Compliance** | 20 | 16 (80%) | 4 (20%) | Structure, Risk, Audit |
| **TOTAL** | **97** | **73 (75%)** | **24 (25%)** | -- |

---

## Regulatory Standards Referenced

| Standard | Domain | Pillars |
|----------|--------|---------|
| **HIPAA** | US Healthcare Privacy | 1, 4, 5 |
| **PHIPA** | Ontario Healthcare Privacy | 1, 5 |
| **GDPR** | EU Data Protection | 1, 5 |
| **FDA SaMD** | Medical Device Software | 2, 3, 5 |
| **ISO 14971** | Medical Device Risk Management | 3, 5 |
| **ISO/IEC 42001** | AI Management System | 5 |
| **ISO 27001** | Information Security | 1, 4 |
| **NIST Privacy Framework** | Privacy by Design | 1 |
| **OWASP ML Top 10** | ML Security | 1, 2 |

---

## Tools Referenced

| Category | Tools |
|----------|-------|
| **Data Quality** | pandas, Evidently, Great Expectations |
| **Privacy** | Presidio, DLP tools, Cloud KMS |
| **Fairness** | Fairlearn, AIF360 |
| **Explainability** | SHAP, LIME |
| **MLOps** | MLflow, DVC, Feature Stores |
| **Monitoring** | Evidently, SIEM, Dashboards |
| **Documentation** | Draw.io, Lucidchart, Model Cards |

---

## Implementation Checklist

### Phase 1: Assessment (Weeks 1-2)
- [ ] Complete Pillar 1 data inventory audit
- [ ] Review PHI classification
- [ ] Assess current model documentation

### Phase 2: Gap Analysis (Weeks 3-4)
- [ ] Identify high-risk gaps across all pillars
- [ ] Prioritize remediation based on risk
- [ ] Create remediation roadmap

### Phase 3: Remediation (Weeks 5-12)
- [ ] Implement high-risk fixes
- [ ] Create missing documentation (Model Cards, Datasheets)
- [ ] Deploy monitoring infrastructure

### Phase 4: Validation (Weeks 13-14)
- [ ] Re-audit all pillars
- [ ] Collect evidence for each dimension
- [ ] Prepare compliance package

### Phase 5: Continuous Improvement (Ongoing)
- [ ] Quarterly RAI reviews
- [ ] Drift monitoring
- [ ] Incident response drills

---

## Citation

```bibtex
@misc{rai_audit_framework_2024,
  title={5-Pillar Healthcare AI Governance Audit Framework},
  author={NeuroMCP-Agent Team},
  year={2024},
  note={Version 1.0}
}
```
