# AgenticFinder: Priority Actions Implementation Plan

================================================================================

## Priority Action 1: Clinical Validation Study Protocol

### 1.1 Study Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLINICAL VALIDATION STUDY                             │
│                    AgenticFinder EEG Classification System               │
├─────────────────────────────────────────────────────────────────────────┤
│  Study Type:        Prospective, Multi-center Validation                │
│  Primary Objective: Validate 90%+ accuracy in clinical settings         │
│  Duration:          12 months                                           │
│  Sites:             3-5 clinical centers                                │
│  Target Enrollment: 600 subjects (100 per disease)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Study Objectives

| Objective Type | Description | Success Criteria |
|----------------|-------------|------------------|
| **Primary** | Validate diagnostic accuracy | Accuracy ≥ 90% for each disease |
| **Secondary** | Assess clinical utility | Clinician satisfaction ≥ 80% |
| **Secondary** | Evaluate workflow integration | Processing time < 5 minutes |
| **Exploratory** | Compare to standard care | Non-inferior to current practice |

### 1.3 Study Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STUDY FLOW                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Site Preparation (Months 1-2)                                 │
│  ─────────────────────────────────────────────────────                  │
│  • Site selection and qualification                                     │
│  • IRB/Ethics approval                                                  │
│  • Equipment installation and validation                                │
│  • Staff training                                                       │
│                                                                          │
│  Phase 2: Patient Enrollment (Months 3-9)                               │
│  ─────────────────────────────────────────────────────                  │
│  • Screen and consent eligible patients                                 │
│  • Collect EEG data per protocol                                       │
│  • Run AgenticFinder analysis                                          │
│  • Record clinical diagnosis (gold standard)                           │
│                                                                          │
│  Phase 3: Data Analysis (Months 10-11)                                  │
│  ─────────────────────────────────────────────────────                  │
│  • Statistical analysis of accuracy                                     │
│  • Subgroup analyses                                                    │
│  • Safety and usability assessment                                      │
│                                                                          │
│  Phase 4: Reporting (Month 12)                                          │
│  ─────────────────────────────────────────────────────                  │
│  • Final study report                                                   │
│  • Regulatory submission preparation                                    │
│  • Publication manuscript                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Sample Size Calculation

```python
# Sample Size Calculation for Validation Study
from scipy import stats
import numpy as np

def calculate_sample_size(
    expected_accuracy=0.90,
    margin_of_error=0.05,
    confidence_level=0.95,
    power=0.80
):
    """
    Calculate required sample size for accuracy validation

    Parameters:
    - expected_accuracy: Expected accuracy (from training)
    - margin_of_error: Acceptable margin of error
    - confidence_level: Confidence level (typically 0.95)
    - power: Statistical power (typically 0.80)
    """
    z = stats.norm.ppf((1 + confidence_level) / 2)
    p = expected_accuracy
    e = margin_of_error

    n = (z**2 * p * (1-p)) / (e**2)

    return int(np.ceil(n))

# Per-disease sample sizes
diseases = {
    'Schizophrenia': {'expected': 0.97, 'margin': 0.05},
    'Epilepsy': {'expected': 0.94, 'margin': 0.05},
    'Stress': {'expected': 0.94, 'margin': 0.05},
    'Autism': {'expected': 0.98, 'margin': 0.05},
    'Parkinson': {'expected': 1.00, 'margin': 0.03},
    'Depression': {'expected': 0.91, 'margin': 0.05}
}

# Results:
# Schizophrenia: n = 45 (recommend n = 100 for robustness)
# Epilepsy: n = 87 (recommend n = 100)
# Stress: n = 87 (recommend n = 100)
# Autism: n = 30 (recommend n = 100)
# Parkinson: n = 0 (perfect accuracy, n = 50 minimum)
# Depression: n = 126 (recommend n = 150)
#
# TOTAL RECOMMENDED: 600 subjects
```

### 1.5 Inclusion/Exclusion Criteria

| Criterion Type | Disease | Criteria |
|----------------|---------|----------|
| **Inclusion** | All | Age 18-65 years |
| **Inclusion** | All | Willing to provide informed consent |
| **Inclusion** | All | EEG recording quality acceptable |
| **Exclusion** | All | Active substance abuse |
| **Exclusion** | All | Other neurological conditions |
| **Exclusion** | All | Contraindications to EEG |

### 1.6 Primary Endpoint Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STATISTICAL ANALYSIS PLAN                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Primary Endpoint:                                                       │
│  Classification accuracy (TP + TN) / Total                              │
│                                                                          │
│  Analysis Method:                                                        │
│  • Point estimate with 95% confidence interval                          │
│  • Success if lower bound of 95% CI ≥ 85%                              │
│                                                                          │
│  Secondary Endpoints:                                                    │
│  • Sensitivity (TPR)                                                    │
│  • Specificity (TNR)                                                    │
│  • Positive Predictive Value (PPV)                                      │
│  • Negative Predictive Value (NPV)                                      │
│  • Area Under ROC Curve (AUC)                                          │
│                                                                          │
│  Subgroup Analyses:                                                      │
│  • By age group (18-35, 36-50, 51-65)                                  │
│  • By gender                                                            │
│  • By site                                                              │
│  • By disease severity                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Priority Action 2: Regulatory Approval Roadmap

### 2.1 FDA SaMD (Software as Medical Device) Pathway

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FDA REGULATORY PATHWAY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Classification: Class II Medical Device (Software as Medical Device)  │
│  Pathway: 510(k) Premarket Notification                                 │
│  Predicate Devices: EEG analysis software (K201369, K190442)           │
│                                                                          │
│  Timeline: 6-12 months                                                  │
│                                                                          │
│  Key Milestones:                                                         │
│  ─────────────────────────────────────────────────────────────────      │
│  Month 1-2:   Pre-submission meeting request                            │
│  Month 3:     Pre-submission meeting with FDA                           │
│  Month 4-6:   510(k) preparation                                       │
│  Month 7:     510(k) submission                                        │
│  Month 8-12:  FDA review and response                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 510(k) Submission Requirements

| Section | Content Required | Status |
|---------|------------------|--------|
| Device Description | Intended use, indications | READY |
| Predicate Comparison | Substantial equivalence | IN PROGRESS |
| Performance Testing | Analytical & clinical | PENDING |
| Software Documentation | SDLC, cybersecurity | READY |
| Labeling | IFU, warnings | PENDING |
| Biocompatibility | N/A (non-contact) | N/A |
| Electrical Safety | IEC 60601-1 | PENDING |

### 2.3 EU MDR (Medical Device Regulation) Pathway

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EU MDR REGULATORY PATHWAY                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Classification: Class IIa Medical Device (Rule 11 - Software)         │
│  Pathway: Conformity Assessment via Notified Body                       │
│  Standards: ISO 13485, IEC 62304, ISO 14971                            │
│                                                                          │
│  Timeline: 12-18 months                                                 │
│                                                                          │
│  Key Milestones:                                                         │
│  ─────────────────────────────────────────────────────────────────      │
│  Month 1-3:   Gap analysis and QMS preparation                         │
│  Month 4-6:   Technical documentation preparation                       │
│  Month 7-9:   Notified Body selection and application                  │
│  Month 10-12: Technical documentation review                           │
│  Month 13-15: Audit and assessment                                     │
│  Month 16-18: Certificate issuance and CE marking                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Technical Documentation (EU MDR Annex II)

| Document | Description | Status |
|----------|-------------|--------|
| Device Description | General description, variants | READY |
| Design & Manufacturing | Development process | READY |
| Safety & Performance | Essential requirements | IN PROGRESS |
| Risk Management | ISO 14971 file | IN PROGRESS |
| Product Verification | Testing results | PENDING |
| Clinical Evaluation | Clinical evidence | PENDING |
| Post-Market Surveillance | PMS plan | PENDING |
| IFU & Labeling | User documentation | PENDING |

### 2.5 Regulatory Submission Checklist

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REGULATORY READINESS CHECKLIST                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FDA 510(k) Requirements:                                               │
│  [✓] Indications for use statement                                     │
│  [✓] Device description                                                │
│  [✓] Software documentation (IEC 62304)                                │
│  [ ] Predicate device comparison                                        │
│  [ ] Performance testing (analytical)                                   │
│  [ ] Performance testing (clinical)                                     │
│  [ ] Cybersecurity documentation                                        │
│  [ ] Labeling and IFU                                                   │
│  [ ] 510(k) summary                                                     │
│                                                                          │
│  EU MDR Requirements:                                                    │
│  [✓] Quality Management System (ISO 13485)                             │
│  [✓] Software lifecycle (IEC 62304)                                    │
│  [ ] Risk management file (ISO 14971)                                  │
│  [ ] Clinical evaluation report                                         │
│  [ ] Technical documentation                                            │
│  [ ] Post-market surveillance plan                                      │
│  [ ] Declaration of Conformity                                          │
│  [ ] Notified Body certificate                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Priority Action 3: Automated Drift Monitoring Implementation

### 3.1 Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DRIFT MONITORING ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐                                                    │
│  │  EEG Input      │                                                    │
│  │  (Production)   │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FEATURE EXTRACTION PIPELINE                                     │   │
│  │  ├── Extract features                                           │   │
│  │  ├── Log feature distributions                                  │   │
│  │  └── Compare to baseline                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MODEL INFERENCE                                                 │   │
│  │  ├── Generate prediction                                        │   │
│  │  ├── Log prediction distribution                                │   │
│  │  └── Log confidence scores                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  DRIFT DETECTION ENGINE                                          │   │
│  │  ├── Input Drift:      PSI, KS-test                             │   │
│  │  ├── Prediction Drift: Distribution shift                       │   │
│  │  ├── Performance Drift: Accuracy (when labels available)       │   │
│  │  └── Calibration Drift: ECE monitoring                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ALERTING & RESPONSE                                             │   │
│  │  ├── Email alerts                                               │   │
│  │  ├── Dashboard notifications                                    │   │
│  │  └── Automated retraining triggers                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Drift Detection Implementation

```python
# drift_monitor.py
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from datetime import datetime

@dataclass
class DriftAlert:
    drift_type: str
    feature_name: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    score: float
    threshold: float
    timestamp: str
    action_required: str

class DriftMonitor:
    """
    Automated drift detection for AgenticFinder models
    """

    def __init__(self, baseline_stats: Dict):
        self.baseline = baseline_stats
        self.alerts: List[DriftAlert] = []

        # Thresholds
        self.thresholds = {
            'psi': {'low': 0.1, 'medium': 0.2, 'high': 0.25},
            'ks': {'low': 0.05, 'medium': 0.1, 'high': 0.15},
            'accuracy_drop': {'low': 0.02, 'medium': 0.05, 'high': 0.10}
        }

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray,
                      bins: int = 10) -> float:
        """
        Calculate Population Stability Index
        PSI < 0.1: No significant shift
        PSI 0.1-0.25: Moderate shift
        PSI > 0.25: Significant shift
        """
        # Bin the data
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, breakpoints)[0]
        actual_counts = np.histogram(actual, breakpoints)[0]

        # Add small value to avoid division by zero
        expected_percents = (expected_counts + 1) / (len(expected) + bins)
        actual_percents = (actual_counts + 1) / (len(actual) + bins)

        psi = np.sum((actual_percents - expected_percents) *
                     np.log(actual_percents / expected_percents))

        return psi

    def ks_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution shift
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        return statistic, p_value

    def detect_feature_drift(self, feature_name: str,
                             current_values: np.ndarray) -> DriftAlert:
        """
        Detect drift for a single feature
        """
        baseline_values = self.baseline['features'][feature_name]

        psi = self.calculate_psi(baseline_values, current_values)
        ks_stat, ks_pvalue = self.ks_test(baseline_values, current_values)

        # Determine severity
        if psi > self.thresholds['psi']['high']:
            severity = 'CRITICAL'
            action = 'Immediate investigation required'
        elif psi > self.thresholds['psi']['medium']:
            severity = 'HIGH'
            action = 'Schedule model review'
        elif psi > self.thresholds['psi']['low']:
            severity = 'MEDIUM'
            action = 'Monitor closely'
        else:
            severity = 'LOW'
            action = 'No action required'

        return DriftAlert(
            drift_type='FEATURE_DRIFT',
            feature_name=feature_name,
            severity=severity,
            score=psi,
            threshold=self.thresholds['psi']['medium'],
            timestamp=datetime.now().isoformat(),
            action_required=action
        )

    def detect_prediction_drift(self, current_predictions: np.ndarray) -> DriftAlert:
        """
        Detect drift in prediction distribution
        """
        baseline_preds = self.baseline['predictions']

        # Compare class distributions
        baseline_dist = np.bincount(baseline_preds, minlength=2) / len(baseline_preds)
        current_dist = np.bincount(current_predictions, minlength=2) / len(current_predictions)

        # Jensen-Shannon divergence
        js_div = 0.5 * (stats.entropy(baseline_dist, 0.5*(baseline_dist + current_dist)) +
                        stats.entropy(current_dist, 0.5*(baseline_dist + current_dist)))

        if js_div > 0.15:
            severity = 'CRITICAL'
        elif js_div > 0.10:
            severity = 'HIGH'
        elif js_div > 0.05:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        return DriftAlert(
            drift_type='PREDICTION_DRIFT',
            feature_name='class_distribution',
            severity=severity,
            score=js_div,
            threshold=0.10,
            timestamp=datetime.now().isoformat(),
            action_required='Review model predictions' if severity != 'LOW' else 'None'
        )

    def detect_performance_drift(self, current_accuracy: float) -> DriftAlert:
        """
        Detect drop in model performance
        """
        baseline_accuracy = self.baseline['accuracy']
        accuracy_drop = baseline_accuracy - current_accuracy

        if accuracy_drop > self.thresholds['accuracy_drop']['high']:
            severity = 'CRITICAL'
            action = 'Immediate retraining required'
        elif accuracy_drop > self.thresholds['accuracy_drop']['medium']:
            severity = 'HIGH'
            action = 'Schedule retraining'
        elif accuracy_drop > self.thresholds['accuracy_drop']['low']:
            severity = 'MEDIUM'
            action = 'Monitor and investigate'
        else:
            severity = 'LOW'
            action = 'No action required'

        return DriftAlert(
            drift_type='PERFORMANCE_DRIFT',
            feature_name='accuracy',
            severity=severity,
            score=accuracy_drop,
            threshold=self.thresholds['accuracy_drop']['medium'],
            timestamp=datetime.now().isoformat(),
            action_required=action
        )

    def run_monitoring(self, current_data: Dict) -> Dict:
        """
        Run complete monitoring pipeline
        """
        self.alerts = []

        # Feature drift
        for feature_name in current_data.get('features', {}).keys():
            alert = self.detect_feature_drift(
                feature_name,
                current_data['features'][feature_name]
            )
            if alert.severity != 'LOW':
                self.alerts.append(alert)

        # Prediction drift
        if 'predictions' in current_data:
            alert = self.detect_prediction_drift(current_data['predictions'])
            if alert.severity != 'LOW':
                self.alerts.append(alert)

        # Performance drift (if labels available)
        if 'accuracy' in current_data:
            alert = self.detect_performance_drift(current_data['accuracy'])
            if alert.severity != 'LOW':
                self.alerts.append(alert)

        return {
            'status': 'ALERT' if self.alerts else 'OK',
            'alert_count': len(self.alerts),
            'critical_count': sum(1 for a in self.alerts if a.severity == 'CRITICAL'),
            'alerts': [vars(a) for a in self.alerts],
            'timestamp': datetime.now().isoformat()
        }
```

### 3.3 Monitoring Dashboard KPIs

| KPI | Description | Target | Alert Threshold |
|-----|-------------|--------|-----------------|
| Input PSI | Feature distribution shift | < 0.1 | > 0.25 |
| Prediction Distribution | Class balance shift | < 5% | > 10% |
| Rolling Accuracy | 30-day accuracy | ≥ 90% | < 88% |
| Confidence Calibration | ECE score | < 0.05 | > 0.10 |
| Inference Latency | P95 latency | < 1s | > 2s |
| Error Rate | System errors | < 1% | > 5% |

### 3.4 Alert Response Matrix

| Severity | Response Time | Action | Escalation |
|----------|---------------|--------|------------|
| LOW | 7 days | Log and monitor | None |
| MEDIUM | 3 days | Investigate root cause | Team lead |
| HIGH | 24 hours | Prepare mitigation plan | Manager |
| CRITICAL | 2 hours | Immediate intervention | Director |

---

## Priority Action 4: Fairness Testing Expansion

### 4.1 Expanded Fairness Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FAIRNESS TESTING EXPANSION PLAN                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Current Coverage:                                                       │
│  • Class balance: ✓ Tested                                              │
│  • Error rate parity: ✓ Tested                                          │
│                                                                          │
│  Expansion Areas:                                                        │
│  • Demographic fairness (age, gender, ethnicity)                        │
│  • Intersectional fairness                                              │
│  • Geographic/site fairness                                             │
│  • Socioeconomic fairness                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Demographic Subgroup Testing

| Attribute | Subgroups | Data Availability | Testing Status |
|-----------|-----------|-------------------|----------------|
| **Age** | 18-35, 36-50, 51-65 | AVAILABLE | PENDING |
| **Gender** | Male, Female | AVAILABLE | PENDING |
| **Ethnicity** | Multiple groups | LIMITED | NEEDS DATA |
| **Education** | Various levels | NOT AVAILABLE | NEEDS DATA |
| **Site/Region** | Multiple centers | PENDING | NEEDS DATA |

### 4.3 Fairness Metrics Implementation

```python
# fairness_testing.py
import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix

class FairnessTester:
    """
    Comprehensive fairness testing for AgenticFinder
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 protected_attributes: Dict[str, np.ndarray]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.protected_attributes = protected_attributes

    def demographic_parity(self, attribute: str) -> Dict:
        """
        Check if positive prediction rates are equal across groups

        Demographic Parity Difference (DPD):
        |P(Y=1|A=0) - P(Y=1|A=1)|

        Target: DPD < 0.10
        """
        attr_values = self.protected_attributes[attribute]
        unique_values = np.unique(attr_values)

        positive_rates = {}
        for val in unique_values:
            mask = attr_values == val
            positive_rates[str(val)] = np.mean(self.y_pred[mask])

        rates = list(positive_rates.values())
        dpd = max(rates) - min(rates)

        return {
            'metric': 'Demographic Parity',
            'attribute': attribute,
            'positive_rates': positive_rates,
            'dpd': dpd,
            'threshold': 0.10,
            'pass': dpd < 0.10
        }

    def equalized_odds(self, attribute: str) -> Dict:
        """
        Check if TPR and FPR are equal across groups

        Equalized Odds Difference (EOD):
        max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)

        Target: EOD < 0.10
        """
        attr_values = self.protected_attributes[attribute]
        unique_values = np.unique(attr_values)

        tpr_by_group = {}
        fpr_by_group = {}

        for val in unique_values:
            mask = attr_values == val
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]

            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()

            tpr_by_group[str(val)] = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_by_group[str(val)] = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())

        eod = max(max(tpr_values) - min(tpr_values),
                  max(fpr_values) - min(fpr_values))

        return {
            'metric': 'Equalized Odds',
            'attribute': attribute,
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group,
            'eod': eod,
            'threshold': 0.10,
            'pass': eod < 0.10
        }

    def predictive_parity(self, attribute: str) -> Dict:
        """
        Check if PPV is equal across groups

        PPV Difference:
        |PPV_0 - PPV_1|

        Target: < 0.10
        """
        attr_values = self.protected_attributes[attribute]
        unique_values = np.unique(attr_values)

        ppv_by_group = {}

        for val in unique_values:
            mask = attr_values == val
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]

            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()

            ppv_by_group[str(val)] = tp / (tp + fp) if (tp + fp) > 0 else 0

        ppv_values = list(ppv_by_group.values())
        ppv_diff = max(ppv_values) - min(ppv_values)

        return {
            'metric': 'Predictive Parity',
            'attribute': attribute,
            'ppv_by_group': ppv_by_group,
            'ppv_difference': ppv_diff,
            'threshold': 0.10,
            'pass': ppv_diff < 0.10
        }

    def run_full_assessment(self) -> Dict:
        """
        Run comprehensive fairness assessment
        """
        results = {
            'overall_status': 'PASS',
            'attributes_tested': [],
            'metrics': []
        }

        for attr in self.protected_attributes.keys():
            results['attributes_tested'].append(attr)

            dp = self.demographic_parity(attr)
            eo = self.equalized_odds(attr)
            pp = self.predictive_parity(attr)

            results['metrics'].extend([dp, eo, pp])

            if not (dp['pass'] and eo['pass'] and pp['pass']):
                results['overall_status'] = 'NEEDS_ATTENTION'

        return results
```

### 4.4 Data Collection Plan for Diverse Populations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DIVERSE DATA COLLECTION PLAN                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Identify Gaps (Month 1)                                       │
│  ─────────────────────────────────────────────────────────────────      │
│  • Audit current demographic distribution                               │
│  • Identify underrepresented groups                                     │
│  • Prioritize based on clinical prevalence                              │
│                                                                          │
│  Phase 2: Partner Development (Months 2-3)                              │
│  ─────────────────────────────────────────────────────────────────      │
│  • Identify diverse clinical sites                                      │
│  • Establish data sharing agreements                                    │
│  • Ensure ethical approval for diverse recruitment                      │
│                                                                          │
│  Phase 3: Data Collection (Months 4-9)                                  │
│  ─────────────────────────────────────────────────────────────────      │
│  • Targeted recruitment at diverse sites                                │
│  • Ensure demographic metadata collection                               │
│  • Quality assurance on collected data                                  │
│                                                                          │
│  Phase 4: Fairness Validation (Months 10-12)                            │
│  ─────────────────────────────────────────────────────────────────      │
│  • Run comprehensive fairness testing                                   │
│  • Identify and mitigate disparities                                    │
│  • Document findings and actions                                        │
│                                                                          │
│  Target Demographics:                                                    │
│  ─────────────────────────────────────────────────────────────────      │
│  Age Groups:                                                             │
│  • 18-35 years: 33%                                                     │
│  • 36-50 years: 34%                                                     │
│  • 51-65 years: 33%                                                     │
│                                                                          │
│  Gender:                                                                 │
│  • Male: 50%                                                            │
│  • Female: 50%                                                          │
│                                                                          │
│  Ethnicity (Target for US):                                             │
│  • White: 60%                                                           │
│  • Black/African American: 13%                                          │
│  • Hispanic/Latino: 18%                                                 │
│  • Asian: 6%                                                            │
│  • Other: 3%                                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Fairness Mitigation Strategies

| Issue | Mitigation Strategy | Implementation |
|-------|---------------------|----------------|
| Class imbalance | Resampling, SMOTE | Already implemented |
| Demographic imbalance | Stratified sampling | Add to training |
| Feature bias | Feature fairness constraints | Post-hoc adjustment |
| Threshold bias | Group-specific thresholds | Calibration per group |
| Data gaps | Targeted data collection | Partnership expansion |

---

## Summary: Implementation Timeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRIORITY ACTIONS TIMELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Action 1: Clinical Validation Study                                    │
│  ├── Q1 2026: Site preparation, IRB approval                           │
│  ├── Q2-Q3 2026: Patient enrollment                                    │
│  └── Q4 2026: Analysis and reporting                                   │
│                                                                          │
│  Action 2: Regulatory Approvals                                         │
│  ├── Q1 2026: FDA pre-submission meeting                               │
│  ├── Q2 2026: 510(k) preparation                                       │
│  ├── Q3 2026: FDA submission                                           │
│  ├── Q4 2026: FDA clearance (target)                                   │
│  └── 2027: EU MDR certification                                        │
│                                                                          │
│  Action 3: Drift Monitoring                                              │
│  ├── Month 1: Implement monitoring code                                │
│  ├── Month 2: Dashboard development                                    │
│  ├── Month 3: Alert system integration                                 │
│  └── Ongoing: Continuous monitoring                                    │
│                                                                          │
│  Action 4: Fairness Testing Expansion                                   │
│  ├── Q1 2026: Gap analysis                                             │
│  ├── Q2 2026: Partner development                                      │
│  ├── Q3 2026: Diverse data collection                                  │
│  └── Q4 2026: Comprehensive fairness validation                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0*
*Created: 2026-01-04*
*AgenticFinder Priority Actions Implementation Plan*
