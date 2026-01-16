"""
Compliance AI Analysis Module
==============================

Comprehensive analysis for AI compliance and regulatory alignment.
Implements 20 analysis types for AI compliance governance.

Analysis Types:
1. Compliance Scope & Jurisdiction Mapping
2. Regulatory Risk Classification
3. Legal Basis & Purpose Limitation
4. Data Protection & Privacy Compliance
5. Transparency & Disclosure Compliance
6. Fairness & Non-Discrimination Compliance
7. Safety & Risk Control Compliance
8. Human Oversight Compliance
9. Explainability / Right-to-Explanation Compliance
10. Accuracy, Robustness & Reliability Compliance
11. Monitoring & Post-Market Surveillance
12. Incident & Breach Reporting Compliance
13. Vendor & Third-Party Compliance
14. Documentation & Record-Keeping
15. Audit & Inspection Readiness
16. Change Management & Re-Compliance
17. Training & Awareness Compliance
18. Accountability & Liability Mapping
19. Compliance Drift Detection
20. Compliance Governance & Enforcement
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Jurisdiction:
    """Represents a regulatory jurisdiction."""
    jurisdiction_id: str
    name: str
    region: str
    applicable_regulations: List[str] = field(default_factory=list)
    data_flow_restrictions: bool = False


@dataclass
class ComplianceRequirement:
    """Represents a compliance requirement."""
    requirement_id: str
    regulation: str
    description: str
    category: str  # privacy, safety, fairness, transparency
    compliance_status: str = "unknown"  # compliant, non_compliant, partial, unknown
    evidence: List[str] = field(default_factory=list)


@dataclass
class ComplianceAudit:
    """Represents a compliance audit."""
    audit_id: str
    audit_date: datetime
    auditor: str
    scope: str
    findings: List[str] = field(default_factory=list)
    status: str = "passed"  # passed, failed, conditional


# ============================================================================
# Jurisdiction Mapping (Type 1)
# ============================================================================

class JurisdictionMappingAnalyzer:
    """Analyzes compliance scope and jurisdiction mapping."""

    MAJOR_REGULATIONS = {
        'EU': ['EU_AI_Act', 'GDPR', 'DSA'],
        'US': ['State_AI_Laws', 'FTC_Guidance', 'CCPA'],
        'UK': ['UK_AI_Framework', 'UK_GDPR'],
        'APAC': ['PDPA', 'PIPL'],
        'Global': ['ISO_42001', 'OECD_AI_Principles']
    }

    def analyze_jurisdictions(self,
                             operating_regions: List[str],
                             data_flows: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Map which laws and standards apply."""
        applicable_regulations = []
        cross_border_issues = []

        for region in operating_regions:
            regs = self.MAJOR_REGULATIONS.get(region, [])
            applicable_regulations.extend(regs)

        # Analyze cross-border data flows
        if data_flows:
            for source, destinations in data_flows.items():
                for dest in destinations:
                    if source != dest:
                        cross_border_issues.append({
                            'source': source,
                            'destination': dest,
                            'requires_transfer_mechanism': True
                        })

        return {
            'compliance_scope_jurisdiction_map': {
                'operating_regions': operating_regions,
                'applicable_regulations': list(set(applicable_regulations)),
                'cross_border_data_flows': cross_border_issues,
                'sectors_applicable': ['general']  # Can be customized
            },
            'jurisdiction_count': len(operating_regions),
            'regulation_count': len(set(applicable_regulations))
        }


# ============================================================================
# Regulatory Risk Classification (Type 2)
# ============================================================================

class RegulatoryRiskAnalyzer:
    """Analyzes regulatory risk classification."""

    def analyze_risk_classification(self,
                                   ai_system: Dict[str, Any],
                                   use_case: Dict[str, Any]) -> Dict[str, Any]:
        """Classify how regulated this AI system is."""
        # Determine risk tier based on use case
        high_risk_categories = [
            'employment', 'credit', 'healthcare', 'law_enforcement',
            'education', 'critical_infrastructure', 'biometric'
        ]

        use_case_category = use_case.get('category', 'general')
        intended_use = use_case.get('intended_use', '')

        if use_case_category in high_risk_categories:
            risk_tier = 'high'
        elif ai_system.get('autonomous_decisions', False):
            risk_tier = 'high'
        elif ai_system.get('affects_individuals', False):
            risk_tier = 'medium'
        else:
            risk_tier = 'low'

        # EU AI Act classification
        if use_case_category in ['biometric', 'law_enforcement']:
            eu_ai_act_class = 'prohibited_or_high'
        elif use_case_category in high_risk_categories:
            eu_ai_act_class = 'high_risk'
        elif ai_system.get('user_facing', False):
            eu_ai_act_class = 'limited_risk'
        else:
            eu_ai_act_class = 'minimal_risk'

        return {
            'regulatory_risk_classification': {
                'risk_tier': risk_tier,
                'use_case_category': use_case_category,
                'eu_ai_act_classification': eu_ai_act_class,
                'intended_use': intended_use
            },
            'compliance_obligations': self._determine_obligations(risk_tier, eu_ai_act_class)
        }

    def _determine_obligations(self, risk_tier: str, eu_class: str) -> List[str]:
        obligations = []
        if risk_tier == 'high' or eu_class in ['high_risk', 'prohibited_or_high']:
            obligations.extend([
                'Risk management system',
                'Data governance',
                'Technical documentation',
                'Record-keeping',
                'Transparency',
                'Human oversight',
                'Accuracy and robustness'
            ])
        elif risk_tier == 'medium':
            obligations.extend([
                'Basic documentation',
                'Transparency to users'
            ])
        return obligations


# ============================================================================
# Legal Basis Analysis (Type 3)
# ============================================================================

class LegalBasisAnalyzer:
    """Analyzes legal basis and purpose limitation."""

    def analyze_legal_basis(self,
                           processing_purposes: List[Dict[str, Any]],
                           legal_bases: Dict[str, str]) -> Dict[str, Any]:
        """Analyze if there's a lawful reason to use AI."""
        purpose_analysis = []

        for purpose in processing_purposes:
            purpose_name = purpose.get('name', '')
            legal_basis = legal_bases.get(purpose_name, 'unspecified')

            purpose_analysis.append({
                'purpose': purpose_name,
                'legal_basis': legal_basis,
                'documented': legal_basis != 'unspecified',
                'within_boundaries': purpose.get('within_original_purpose', True)
            })

        documented_count = sum(1 for p in purpose_analysis if p['documented'])
        within_bounds = sum(1 for p in purpose_analysis if p['within_boundaries'])

        return {
            'legal_basis_justification': purpose_analysis,
            'compliance_metrics': {
                'purposes_documented': documented_count,
                'total_purposes': len(processing_purposes),
                'within_purpose_bounds': within_bounds,
                'legal_basis_coverage': documented_count / len(processing_purposes) if processing_purposes else 0
            },
            'legal_basis_types': {
                'consent': sum(1 for p in purpose_analysis if p['legal_basis'] == 'consent'),
                'contract': sum(1 for p in purpose_analysis if p['legal_basis'] == 'contract'),
                'legitimate_interest': sum(1 for p in purpose_analysis if p['legal_basis'] == 'legitimate_interest')
            }
        }


# ============================================================================
# Data Protection Compliance (Type 4)
# ============================================================================

class DataProtectionAnalyzer:
    """Analyzes data protection and privacy compliance."""

    def analyze_data_protection(self,
                               data_practices: Dict[str, Any],
                               pii_handling: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze if personal data is handled lawfully."""
        # Data minimization check
        data_minimization = data_practices.get('data_minimization', False)

        # PII handling
        pii_detected = pii_handling.get('pii_detected', False) if pii_handling else False
        pii_protected = pii_handling.get('pii_protected', False) if pii_handling else False

        # Retention compliance
        has_retention_policy = data_practices.get('retention_policy', False)
        deletion_capability = data_practices.get('deletion_capability', False)

        compliance_score = sum([
            data_minimization,
            not pii_detected or pii_protected,
            has_retention_policy,
            deletion_capability
        ]) / 4

        return {
            'privacy_compliance_report': {
                'data_minimization': data_minimization,
                'pii_handling': {
                    'pii_detected': pii_detected,
                    'pii_protected': pii_protected
                },
                'retention_deletion': {
                    'has_retention_policy': has_retention_policy,
                    'deletion_capability': deletion_capability
                },
                'compliance_score': float(compliance_score)
            }
        }


# ============================================================================
# Transparency Compliance (Type 5)
# ============================================================================

class TransparencyComplianceAnalyzer:
    """Analyzes transparency and disclosure compliance."""

    def analyze_transparency(self,
                            disclosure_config: Dict[str, Any],
                            user_communications: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if users are informed AI is used."""
        ai_disclosed = disclosure_config.get('ai_use_disclosed', False)
        explanation_provided = disclosure_config.get('explanation_provided', False)
        limitations_stated = disclosure_config.get('limitations_stated', False)

        # Check user communications
        disclosure_count = 0
        if user_communications:
            disclosure_count = sum(1 for c in user_communications if c.get('includes_ai_disclosure', False))

        return {
            'transparency_disclosure_statement': {
                'ai_use_disclosed': ai_disclosed,
                'explanation_provided': explanation_provided,
                'limitations_stated': limitations_stated,
                'disclosure_in_communications': disclosure_count
            },
            'compliance_status': 'compliant' if ai_disclosed and explanation_provided else 'non_compliant'
        }


# ============================================================================
# Fairness Compliance (Type 6)
# ============================================================================

class FairnessComplianceAnalyzer:
    """Analyzes fairness and non-discrimination compliance."""

    def analyze_fairness_compliance(self,
                                   fairness_tests: List[Dict[str, Any]],
                                   protected_groups: List[str] = None) -> Dict[str, Any]:
        """Analyze if AI violates equality laws."""
        protected_groups = protected_groups or ['gender', 'race', 'age', 'disability']

        disparate_impact = []
        proxy_discrimination = []

        for test in fairness_tests:
            test_type = test.get('type', '')
            if test_type == 'disparate_impact':
                disparate_impact.append({
                    'group': test.get('group', ''),
                    'impact_ratio': test.get('impact_ratio', 0),
                    'passes_threshold': test.get('impact_ratio', 0) >= 0.8
                })
            elif test_type == 'proxy':
                proxy_discrimination.append({
                    'proxy_feature': test.get('feature', ''),
                    'correlation_with_protected': test.get('correlation', 0)
                })

        # Overall compliance
        passes_disparate_impact = all(d['passes_threshold'] for d in disparate_impact) if disparate_impact else True
        has_proxy_issues = any(p['correlation_with_protected'] > 0.7 for p in proxy_discrimination)

        return {
            'fairness_compliance_assessment': {
                'disparate_impact_tests': disparate_impact,
                'proxy_discrimination_checks': proxy_discrimination,
                'protected_groups_analyzed': protected_groups
            },
            'compliance_status': {
                'passes_disparate_impact': passes_disparate_impact,
                'proxy_discrimination_risk': has_proxy_issues,
                'overall': 'compliant' if passes_disparate_impact and not has_proxy_issues else 'at_risk'
            }
        }


# ============================================================================
# Additional Compliance Analyzers (Types 7-20)
# ============================================================================

class SafetyComplianceAnalyzer:
    """Analyzes safety and risk control compliance (Type 7)."""

    def analyze_safety_compliance(self,
                                 safety_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if safety obligations are met."""
        has_hazard_analysis = safety_analysis.get('hazard_analysis', False)
        failsafe_mechanisms = safety_analysis.get('failsafe_mechanisms', [])

        return {
            'safety_compliance_evidence': {
                'hazard_analysis_conducted': has_hazard_analysis,
                'failsafe_mechanisms': failsafe_mechanisms,
                'compliance_status': 'compliant' if has_hazard_analysis and failsafe_mechanisms else 'needs_work'
            }
        }


class HumanOversightComplianceAnalyzer:
    """Analyzes human oversight compliance (Type 8)."""

    def analyze_oversight_compliance(self,
                                    oversight_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if legally required human oversight is present."""
        hitl_thresholds = oversight_config.get('hitl_thresholds', {})
        override_authority = oversight_config.get('override_authority', '')

        return {
            'human_oversight_compliance_report': {
                'hitl_thresholds_defined': bool(hitl_thresholds),
                'override_authority_assigned': bool(override_authority),
                'compliance_status': 'compliant' if hitl_thresholds and override_authority else 'non_compliant'
            }
        }


class ExplainabilityComplianceAnalyzer:
    """Analyzes explainability compliance (Type 9)."""

    def analyze_explainability_compliance(self,
                                         explanation_capability: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if decisions can be explained when required."""
        can_explain = explanation_capability.get('explanation_available', False)
        audience_appropriate = explanation_capability.get('audience_appropriate', False)

        return {
            'explainability_compliance_artifacts': {
                'explanation_capability': can_explain,
                'audience_appropriate': audience_appropriate,
                'explanation_methods': explanation_capability.get('methods', []),
                'compliance_status': 'compliant' if can_explain else 'non_compliant'
            }
        }


class PerformanceComplianceAnalyzer:
    """Analyzes accuracy, robustness & reliability compliance (Type 10)."""

    def analyze_performance_compliance(self,
                                      performance_metrics: Dict[str, Any],
                                      thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze if performance meets regulatory expectations."""
        thresholds = thresholds or {'accuracy': 0.9, 'robustness': 0.85}

        accuracy = performance_metrics.get('accuracy', 0)
        robustness = performance_metrics.get('robustness', 0)

        meets_accuracy = accuracy >= thresholds['accuracy']
        meets_robustness = robustness >= thresholds['robustness']

        return {
            'performance_compliance_report': {
                'accuracy': float(accuracy),
                'robustness': float(robustness),
                'thresholds': thresholds,
                'meets_accuracy_threshold': meets_accuracy,
                'meets_robustness_threshold': meets_robustness,
                'compliance_status': 'compliant' if meets_accuracy and meets_robustness else 'non_compliant'
            }
        }


class PostMarketSurveillanceAnalyzer:
    """Analyzes monitoring and post-market surveillance (Type 11)."""

    def analyze_surveillance(self,
                            monitoring_config: Dict[str, Any],
                            incident_log: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if compliance is maintained after deployment."""
        drift_monitoring = monitoring_config.get('drift_monitoring', False)
        incident_tracking = monitoring_config.get('incident_tracking', False)

        return {
            'post_deployment_compliance_log': {
                'drift_monitoring_enabled': drift_monitoring,
                'incident_tracking_enabled': incident_tracking,
                'incidents_logged': len(incident_log) if incident_log else 0,
                'compliance_status': 'compliant' if drift_monitoring and incident_tracking else 'needs_improvement'
            }
        }


class IncidentReportingAnalyzer:
    """Analyzes incident and breach reporting compliance (Type 12)."""

    def analyze_incident_reporting(self,
                                  incident_config: Dict[str, Any],
                                  incidents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if incidents are handled per law."""
        notification_timeline_hours = incident_config.get('notification_timeline_hours', 72)
        regulator_communication = incident_config.get('regulator_communication', False)

        reported_in_time = 0
        if incidents:
            reported_in_time = sum(1 for i in incidents
                                  if i.get('reported_within_hours', float('inf')) <= notification_timeline_hours)

        return {
            'incident_breach_reports': {
                'notification_timeline_hours': notification_timeline_hours,
                'regulator_communication_established': regulator_communication,
                'incidents_reported_in_time': reported_in_time,
                'total_incidents': len(incidents) if incidents else 0
            }
        }


class VendorComplianceAnalyzer:
    """Analyzes vendor and third-party compliance (Type 13)."""

    def analyze_vendor_compliance(self,
                                 vendors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if external providers are compliant."""
        compliant_vendors = sum(1 for v in vendors if v.get('compliance_verified', False))
        has_controls = sum(1 for v in vendors if v.get('contractual_controls', False))

        return {
            'vendor_compliance_assessment': {
                'total_vendors': len(vendors),
                'compliance_verified': compliant_vendors,
                'contractual_controls_in_place': has_controls,
                'due_diligence_complete': compliant_vendors == len(vendors) if vendors else True
            }
        }


class RecordKeepingAnalyzer:
    """Analyzes documentation and record-keeping (Type 14)."""

    def analyze_record_keeping(self,
                              documentation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if evidence is retained and accessible."""
        has_model_cards = documentation.get('model_cards', False)
        has_logs = documentation.get('logs', False)
        retention_policy = documentation.get('retention_policies', False)

        return {
            'compliance_documentation_index': {
                'model_cards': has_model_cards,
                'logs': has_logs,
                'retention_policies': retention_policy,
                'completeness': sum([has_model_cards, has_logs, retention_policy]) / 3
            }
        }


class AuditReadinessAnalyzer:
    """Analyzes audit and inspection readiness (Type 15)."""

    def analyze_audit_readiness(self,
                               audit_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if regulators can audit the system."""
        evidence_accessible = audit_artifacts.get('evidence_accessible', False)
        traceability = audit_artifacts.get('traceability', False)

        return {
            'audit_readiness_package': {
                'evidence_accessible': evidence_accessible,
                'traceability_enabled': traceability,
                'artifacts_available': audit_artifacts.get('artifacts', []),
                'readiness_score': (evidence_accessible + traceability) / 2
            }
        }


class ReComplianceAnalyzer:
    """Analyzes change management and re-compliance (Type 16)."""

    def analyze_recompliance(self,
                            changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if changes are re-evaluated for compliance."""
        reevaluated = sum(1 for c in changes if c.get('compliance_reevaluated', False))

        return {
            'change_impact_compliance_review': {
                'total_changes': len(changes),
                'compliance_reevaluated': reevaluated,
                'reevaluation_rate': reevaluated / len(changes) if changes else 1
            }
        }


class TrainingComplianceAnalyzer:
    """Analyzes training and awareness compliance (Type 17)."""

    def analyze_training_compliance(self,
                                   training_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if staff are trained on obligations."""
        trained = sum(1 for r in training_records if r.get('completed', False))
        certified = sum(1 for r in training_records if r.get('certified', False))

        return {
            'training_records': {
                'total_staff': len(training_records),
                'training_completed': trained,
                'certified': certified,
                'compliance_rate': trained / len(training_records) if training_records else 0
            }
        }


class LiabilityMappingAnalyzer:
    """Analyzes accountability and liability mapping (Type 18)."""

    def analyze_liability(self,
                         liability_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze who is legally responsible."""
        return {
            'accountability_register': {
                'liability_owner': liability_config.get('liability_owner', ''),
                'ownership_documented': bool(liability_config.get('liability_owner')),
                'liability_assignment': liability_config.get('liability_assignment', {})
            }
        }


class ComplianceDriftAnalyzer:
    """Analyzes compliance drift detection (Type 19)."""

    def analyze_compliance_drift(self,
                                compliance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if compliance erodes over time."""
        if not compliance_history:
            return {'drift_detected': False, 'report': {}}

        compliance_rates = [h.get('compliance_rate', 0) for h in compliance_history]

        if len(compliance_rates) >= 2:
            first_half = np.mean(compliance_rates[:len(compliance_rates)//2])
            second_half = np.mean(compliance_rates[len(compliance_rates)//2:])
            drift = first_half - second_half
        else:
            drift = 0

        # Check for scope creep
        scope_changes = sum(1 for h in compliance_history if h.get('scope_expanded', False))

        return {
            'compliance_drift_report': {
                'drift_score': float(drift),
                'drift_detected': drift > 0.1,
                'scope_expansions': scope_changes,
                'use_case_creep': scope_changes > len(compliance_history) * 0.2
            }
        }


class ComplianceEnforcementAnalyzer:
    """Analyzes compliance governance and enforcement (Type 20)."""

    def analyze_enforcement(self,
                           enforcement_config: Dict[str, Any],
                           audit_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze who enforces compliance."""
        compliance_owner = enforcement_config.get('compliance_owner', '')
        sanctions = enforcement_config.get('sanctions', [])
        escalation = enforcement_config.get('escalation_path', [])

        # Audit trail
        audit_compliance = 0
        if audit_records:
            audit_compliance = sum(1 for a in audit_records if a.get('compliant', False)) / len(audit_records)

        return {
            'compliance_governance_policy': {
                'compliance_owner': compliance_owner,
                'sanctions_defined': len(sanctions),
                'escalation_path': escalation
            },
            'audit_trail': {
                'total_audits': len(audit_records) if audit_records else 0,
                'compliance_rate': float(audit_compliance)
            }
        }


# ============================================================================
# Report Generator
# ============================================================================

class ComplianceReportGenerator:
    """Generates comprehensive compliance analysis reports."""

    def __init__(self):
        self.jurisdiction_analyzer = JurisdictionMappingAnalyzer()
        self.risk_analyzer = RegulatoryRiskAnalyzer()
        self.legal_analyzer = LegalBasisAnalyzer()
        self.data_analyzer = DataProtectionAnalyzer()
        self.transparency_analyzer = TransparencyComplianceAnalyzer()
        self.fairness_analyzer = FairnessComplianceAnalyzer()
        self.drift_analyzer = ComplianceDriftAnalyzer()
        self.enforcement_analyzer = ComplianceEnforcementAnalyzer()

    def generate_full_report(self,
                            operating_regions: List[str] = None,
                            ai_system: Dict[str, Any] = None,
                            use_case: Dict[str, Any] = None,
                            fairness_tests: List[Dict[str, Any]] = None,
                            compliance_history: List[Dict[str, Any]] = None,
                            enforcement_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            'report_type': 'compliance_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if operating_regions:
            report['jurisdictions'] = self.jurisdiction_analyzer.analyze_jurisdictions(operating_regions)

        if ai_system and use_case:
            report['risk_classification'] = self.risk_analyzer.analyze_risk_classification(ai_system, use_case)

        if fairness_tests:
            report['fairness_compliance'] = self.fairness_analyzer.analyze_fairness_compliance(fairness_tests)

        if compliance_history:
            report['compliance_drift'] = self.drift_analyzer.analyze_compliance_drift(compliance_history)

        if enforcement_config:
            report['enforcement'] = self.enforcement_analyzer.analyze_enforcement(enforcement_config)

        # Calculate overall compliance score
        compliance_statuses = []
        for key, value in report.items():
            if isinstance(value, dict):
                status = value.get('compliance_status')
                if status:
                    compliance_statuses.append(1 if status == 'compliant' else 0)

        report['overall_compliance_score'] = float(np.mean(compliance_statuses)) if compliance_statuses else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
