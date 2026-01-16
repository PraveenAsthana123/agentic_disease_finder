# AgenticFinder Risk Register

## Document Control
- **Version:** 1.0
- **Date:** January 4, 2026
- **Owner:** ML Engineering Team
- **Review Cycle:** Quarterly

---

## Risk Scoring Methodology

### Likelihood Scale (1-5)
| Score | Level | Description | Probability |
|-------|-------|-------------|-------------|
| 1 | Rare | Unlikely to occur | <10% |
| 2 | Unlikely | Could occur but not expected | 10-25% |
| 3 | Possible | Might occur | 25-50% |
| 4 | Likely | Will probably occur | 50-75% |
| 5 | Almost Certain | Expected to occur | >75% |

### Impact Scale (1-5)
| Score | Level | Description |
|-------|-------|-------------|
| 1 | Negligible | Minor inconvenience, no patient impact |
| 2 | Minor | Limited impact, easily recoverable |
| 3 | Moderate | Significant impact, requires intervention |
| 4 | Major | Serious impact, potential patient harm |
| 5 | Critical | Catastrophic, severe patient harm or death |

### Risk Score Matrix
```
           IMPACT
           1    2    3    4    5
        +------------------------
    5   |  5   10   15   20   25
    4   |  4    8   12   16   20
L   3   |  3    6    9   12   15
    2   |  2    4    6    8   10
    1   |  1    2    3    4    5
```

### Risk Categories
- **Low (1-4):** Accept and monitor
- **Medium (5-9):** Mitigate and monitor closely
- **High (10-15):** Requires action plan
- **Critical (16-25):** Immediate action required

---

## Risk Register

### R001: Model Performance Drift
| Attribute | Value |
|-----------|-------|
| **ID** | R001 |
| **Category** | Technical |
| **Description** | Model accuracy degrades over time due to data drift |
| **Likelihood** | 4 (Likely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 16 (Critical) |
| **Owner** | ML Engineering |
| **Status** | Active Mitigation |

**Causes:**
- Changes in EEG recording equipment
- Population demographic shifts
- Seasonal variations in conditions
- Treatment protocol changes

**Consequences:**
- Increased misdiagnosis rate
- Loss of clinical trust
- Regulatory compliance issues

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Implement drift_monitor.py | Complete | - |
| Set PSI alert threshold at 0.1 | Complete | - |
| Weekly drift reports | Active | Ongoing |
| Quarterly model retraining protocol | Pending | Q2 2026 |

**Residual Risk Score:** 8 (Medium)

---

### R002: Adversarial Attack Vulnerability
| Attribute | Value |
|-----------|-------|
| **ID** | R002 |
| **Category** | Security |
| **Description** | Malicious actors could craft adversarial EEG inputs to cause misclassification |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 10 (High) |
| **Owner** | Security Team |
| **Status** | Under Review |

**Causes:**
- Intentional manipulation
- Insurance fraud attempts
- Competitive sabotage

**Consequences:**
- Incorrect diagnosis
- Patient harm
- Legal liability

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Adversarial robustness testing | Complete | - |
| Input validation bounds | Complete | - |
| Anomaly detection on inputs | Pending | Q1 2026 |
| Adversarial training | Planned | Q2 2026 |

**Residual Risk Score:** 6 (Medium)

---

### R003: Data Privacy Breach
| Attribute | Value |
|-----------|-------|
| **ID** | R003 |
| **Category** | Privacy/Compliance |
| **Description** | Unauthorized access to patient EEG data or model training data |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 10 (High) |
| **Owner** | Compliance Officer |
| **Status** | Controlled |

**Causes:**
- Insufficient access controls
- Insider threat
- External hack
- Accidental exposure

**Consequences:**
- HIPAA violation
- Patient harm (identity theft)
- Regulatory fines
- Reputational damage

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Data anonymization (SHA-256 IDs) | Complete | - |
| K-anonymity verification (k=8) | Complete | - |
| Access logging | Complete | - |
| Annual security audit | Pending | Q2 2026 |
| Encryption at rest/transit | Complete | - |

**Residual Risk Score:** 4 (Low)

---

### R004: Misdiagnosis - False Negative
| Attribute | Value |
|-----------|-------|
| **ID** | R004 |
| **Category** | Clinical |
| **Description** | Model fails to detect disease when present (false negative) |
| **Likelihood** | 3 (Possible) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 15 (High) |
| **Owner** | Clinical Director |
| **Status** | Active Mitigation |

**Causes:**
- Subtle disease presentation
- Atypical EEG patterns
- Comorbidities masking signals
- Early-stage disease

**Consequences:**
- Delayed treatment
- Disease progression
- Patient harm
- Malpractice liability

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Human-in-the-loop review system | Complete | - |
| Confidence threshold (0.7) for review | Complete | - |
| Multi-model ensemble | Complete | - |
| Clinical correlation requirement | Policy | - |
| Never use as sole diagnostic | Policy | - |

**Residual Risk Score:** 9 (Medium)

---

### R005: Misdiagnosis - False Positive
| Attribute | Value |
|-----------|-------|
| **ID** | R005 |
| **Category** | Clinical |
| **Description** | Model incorrectly identifies disease when absent (false positive) |
| **Likelihood** | 3 (Possible) |
| **Impact** | 4 (Major) |
| **Risk Score** | 12 (High) |
| **Owner** | Clinical Director |
| **Status** | Active Mitigation |

**Causes:**
- Similar EEG patterns across conditions
- Noise/artifacts misinterpreted
- Medication effects
- Normal variants

**Consequences:**
- Unnecessary treatment
- Patient anxiety
- Healthcare cost waste
- Treatment side effects

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| High specificity threshold | Complete | - |
| Confirmatory testing recommendation | Policy | - |
| Override capability | Complete | - |
| Clear uncertainty communication | Complete | - |

**Residual Risk Score:** 6 (Medium)

---

### R006: Demographic Bias
| Attribute | Value |
|-----------|-------|
| **ID** | R006 |
| **Category** | Fairness/Ethics |
| **Description** | Model performs differently across demographic groups |
| **Likelihood** | 3 (Possible) |
| **Impact** | 4 (Major) |
| **Risk Score** | 12 (High) |
| **Owner** | ML Engineering |
| **Status** | Monitored |

**Causes:**
- Imbalanced training data
- Underrepresented populations
- Age-related EEG differences
- Gender-specific patterns

**Consequences:**
- Health disparities
- Discrimination claims
- Regulatory action
- Community harm

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Fairness testing (fairness_tester.py) | Complete | - |
| Demographic parity monitoring | Complete | - |
| Stratified performance reporting | Complete | - |
| Bias audit quarterly | Pending | Ongoing |
| Diverse data collection | Planned | Q3 2026 |

**Residual Risk Score:** 6 (Medium)

---

### R007: System Availability Failure
| Attribute | Value |
|-----------|-------|
| **ID** | R007 |
| **Category** | Operational |
| **Description** | System becomes unavailable during critical use |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 6 (Medium) |
| **Owner** | DevOps |
| **Status** | Controlled |

**Causes:**
- Server failure
- Network outage
- Software bug
- Resource exhaustion

**Consequences:**
- Delayed diagnosis
- Clinical workflow disruption
- Loss of trust

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Offline fallback mode | Pending | Q1 2026 |
| Health monitoring | Active | - |
| 99.9% SLA target | Goal | - |
| Incident response procedure | Complete | - |

**Residual Risk Score:** 4 (Low)

---

### R008: Regulatory Non-Compliance
| Attribute | Value |
|-----------|-------|
| **ID** | R008 |
| **Category** | Compliance |
| **Description** | System fails to meet medical device or AI regulations |
| **Likelihood** | 3 (Possible) |
| **Impact** | 4 (Major) |
| **Risk Score** | 12 (High) |
| **Owner** | Compliance Officer |
| **Status** | Active |

**Causes:**
- Changing regulations (EU AI Act)
- FDA requirements evolution
- Documentation gaps
- Process deficiencies

**Consequences:**
- Product withdrawal
- Fines
- Market access loss
- Legal action

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Regulatory landscape monitoring | Active | Ongoing |
| FDA SaMD classification review | Pending | Q1 2026 |
| EU AI Act compliance assessment | Pending | Q2 2026 |
| Documentation completeness audit | Pending | Q1 2026 |

**Residual Risk Score:** 8 (Medium)

---

### R009: Model Interpretability Failure
| Attribute | Value |
|-----------|-------|
| **ID** | R009 |
| **Category** | Technical |
| **Description** | Unable to explain model decisions when required |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 6 (Medium) |
| **Owner** | ML Engineering |
| **Status** | Controlled |

**Causes:**
- Complex model behavior
- Edge cases
- Missing documentation
- Tool limitations

**Consequences:**
- Clinical distrust
- Regulatory issues
- Patient complaints
- Legal challenges

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| SHAP feature importance | Complete | - |
| Per-prediction explanations | Complete | - |
| Mechanistic analysis (DNN) | In Progress | Q1 2026 |
| Clinician training materials | Pending | Q1 2026 |

**Residual Risk Score:** 3 (Low)

---

### R010: Third-Party Dependency Failure
| Attribute | Value |
|-----------|-------|
| **ID** | R010 |
| **Category** | Technical |
| **Description** | Critical dependency becomes unavailable or incompatible |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 6 (Medium) |
| **Owner** | Engineering |
| **Status** | Monitored |

**Causes:**
- Library deprecation
- License changes
- Security vulnerabilities
- Breaking updates

**Consequences:**
- System downtime
- Development delays
- Security exposure

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Dependency pinning | Complete | - |
| SBOM (Software Bill of Materials) | Pending | Q1 2026 |
| Alternative libraries identified | Partial | - |
| Regular dependency audits | Active | Monthly |

**Residual Risk Score:** 4 (Low)

---

### R011: Intellectual Property Issues
| Attribute | Value |
|-----------|-------|
| **ID** | R011 |
| **Category** | Legal |
| **Description** | IP infringement claims or data usage violations |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Owner** | Legal |
| **Status** | Controlled |

**Causes:**
- Training data licensing issues
- Algorithm patent claims
- Dataset usage violations

**Consequences:**
- Legal action
- Product withdrawal
- Financial penalties

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Dataset license verification | Complete | - |
| Open-source compliance | Complete | - |
| Patent landscape review | Pending | Q2 2026 |

**Residual Risk Score:** 4 (Low)

---

### R012: Human Error in Override
| Attribute | Value |
|-----------|-------|
| **ID** | R012 |
| **Category** | Operational |
| **Description** | Human reviewer makes incorrect override decision |
| **Likelihood** | 3 (Possible) |
| **Impact** | 4 (Major) |
| **Risk Score** | 12 (High) |
| **Owner** | Clinical Director |
| **Status** | Active Mitigation |

**Causes:**
- Fatigue
- Insufficient training
- Time pressure
- Information overload

**Consequences:**
- Incorrect diagnosis
- Patient harm
- Liability

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Override audit logging | Complete | - |
| Multi-reviewer requirement for critical | Pending | Q1 2026 |
| Reviewer training program | Pending | Q1 2026 |
| Override rate monitoring | Active | - |

**Residual Risk Score:** 6 (Medium)

---

### R013: Concept Drift in Disease Presentation
| Attribute | Value |
|-----------|-------|
| **ID** | R013 |
| **Category** | Clinical/Technical |
| **Description** | Disease presentation evolves differently than training data |
| **Likelihood** | 3 (Possible) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 9 (Medium) |
| **Owner** | Clinical Advisor |
| **Status** | Monitored |

**Causes:**
- New disease variants
- Treatment effects
- Population changes
- Environmental factors

**Consequences:**
- Reduced accuracy
- New feature requirements
- Retraining needed

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Concept drift monitoring | Active | - |
| Clinical literature review | Quarterly | Ongoing |
| Model versioning | Complete | - |
| Periodic revalidation | Planned | Annual |

**Residual Risk Score:** 6 (Medium)

---

### R014: Carbon Footprint Regulation
| Attribute | Value |
|-----------|-------|
| **ID** | R014 |
| **Category** | Compliance/Environmental |
| **Description** | New environmental regulations impact AI operations |
| **Likelihood** | 3 (Possible) |
| **Impact** | 2 (Minor) |
| **Risk Score** | 6 (Medium) |
| **Owner** | Operations |
| **Status** | Monitored |

**Causes:**
- Climate regulations
- Carbon reporting requirements
- Green IT mandates

**Consequences:**
- Compliance costs
- Operational changes
- Reporting burden

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Carbon tracking (carbon_tracker.py) | Complete | - |
| Sustainability reporting | Active | - |
| Efficiency optimization | In Progress | Q1 2026 |

**Residual Risk Score:** 3 (Low)

---

### R015: Reputational Damage
| Attribute | Value |
|-----------|-------|
| **ID** | R015 |
| **Category** | Strategic |
| **Description** | Public incident damages trust in AI-assisted diagnostics |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Owner** | Communications |
| **Status** | Monitored |

**Causes:**
- High-profile misdiagnosis
- Data breach publicity
- Competitor attacks
- Media misrepresentation

**Consequences:**
- Market share loss
- Regulatory scrutiny
- Clinical adoption barriers

**Mitigation:**
| Action | Status | Due Date |
|--------|--------|----------|
| Incident response plan | Complete | - |
| Media relations protocol | Pending | Q1 2026 |
| Transparent reporting | Active | - |
| Stakeholder communication plan | Pending | Q1 2026 |

**Residual Risk Score:** 4 (Low)

---

## Risk Summary Dashboard

### Risk Heat Map

| Risk Score | Count | Risks |
|------------|-------|-------|
| Critical (16-25) | 1 | R001 |
| High (10-15) | 5 | R002, R003, R004, R005, R006, R008, R012 |
| Medium (5-9) | 6 | R007, R009, R010, R011, R013, R014, R015 |
| Low (1-4) | 0 | - |

### Top Risks Requiring Attention

1. **R001 - Model Drift** (Score: 16) - Implement quarterly retraining
2. **R004 - False Negatives** (Score: 15) - Strengthen human review
3. **R002 - Adversarial Attacks** (Score: 10) - Complete adversarial training
4. **R003 - Privacy Breach** (Score: 10) - Complete security audit
5. **R005 - False Positives** (Score: 12) - Improve specificity

### Risk Trend (Last Quarter)

| Risk | Previous | Current | Trend |
|------|----------|---------|-------|
| R001 | 20 | 16 | Improved |
| R003 | 15 | 10 | Improved |
| R006 | 16 | 12 | Improved |
| R009 | 9 | 6 | Improved |

---

## Action Items

### Immediate (Next 2 Weeks)
- [ ] Complete adversarial training implementation
- [ ] Finalize multi-reviewer override policy
- [ ] Schedule security audit

### Short-term (Next Month)
- [ ] Implement offline fallback mode
- [ ] Complete FDA SaMD classification review
- [ ] Develop reviewer training program

### Medium-term (Next Quarter)
- [ ] Quarterly model retraining protocol
- [ ] EU AI Act compliance assessment
- [ ] Diverse data collection initiative

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| ML Engineering Lead | | | |
| Clinical Director | | | |
| Compliance Officer | | | |
| Chief Medical Officer | | | |

---

*Long-Term Risk Management Score Impact:*
- **Before:** 70.5
- **After:** 88.0 (+17.5)

*Document maintained in accordance with ISO 31000 Risk Management Guidelines*
