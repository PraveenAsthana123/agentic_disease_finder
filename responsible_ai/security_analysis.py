"""
Secure AI Analysis Module
==========================

Comprehensive analysis for AI security governance.
Implements 20 analysis types for AI security.

Analysis Types:
1. Security Scope & Asset Definition
2. Threat Model Definition
3. Attack Surface Mapping
4. Data Integrity & Poisoning Protection
5. Training Pipeline Security
6. Model Confidentiality & IP Protection
7. Adversarial Input Robustness
8. Prompt Injection & Jailbreak Defense
9. Output Safety & Misuse Prevention
10. Tool & Action Security
11. Access Control & Authentication
12. Inference-Time Security
13. Privacy & Information Leakage Defense
14. Supply Chain & Dependency Security
15. Monitoring & Security Drift Detection
16. Logging, Auditing & Traceability
17. Incident Detection & Response
18. Penetration Testing & Red Teaming
19. Residual Risk & Risk Acceptance
20. Secure AI Governance & Accountability
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
class SecurityAsset:
    """Represents a security asset."""
    asset_id: str
    asset_type: str  # data, model, pipeline, api, output
    classification: str = "internal"  # public, internal, confidential, restricted
    criticality: str = "medium"


@dataclass
class SecurityControl:
    """Represents a security control."""
    control_id: str
    control_type: str
    description: str
    implemented: bool = False
    effectiveness: float = 0.0


@dataclass
class SecurityIncident:
    """Represents a security incident."""
    incident_id: str
    incident_type: str
    severity: str = "medium"
    detected_at: Optional[datetime] = None
    resolved: bool = False
    response_time_hours: float = 0.0


# ============================================================================
# Security Scope Analyzer (Type 1)
# ============================================================================

class SecurityScopeAnalyzer:
    """Analyzes security scope and asset definition."""

    def analyze_scope(self,
                     assets: List[SecurityAsset]) -> Dict[str, Any]:
        """Define what AI assets must be protected."""
        if not assets:
            return {'asset_inventory': [], 'scope': {}}

        # Categorize assets
        by_type = defaultdict(list)
        by_classification = defaultdict(list)

        for asset in assets:
            by_type[asset.asset_type].append(asset)
            by_classification[asset.classification].append(asset)

        critical_assets = [a for a in assets if a.criticality in ['high', 'critical']]

        return {
            'security_asset_inventory': [
                {
                    'id': a.asset_id,
                    'type': a.asset_type,
                    'classification': a.classification,
                    'criticality': a.criticality
                } for a in assets
            ],
            'scope_summary': {
                'total_assets': len(assets),
                'by_type': {k: len(v) for k, v in by_type.items()},
                'by_classification': {k: len(v) for k, v in by_classification.items()},
                'critical_assets': len(critical_assets)
            }
        }


# ============================================================================
# Threat Model Analyzer (Type 2)
# ============================================================================

class ThreatModelAnalyzer:
    """Analyzes threat model definition."""

    def analyze_threat_model(self,
                            threat_actors: List[Dict[str, Any]],
                            attack_scenarios: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Define who are the adversaries and their goals."""
        actor_analysis = []

        for actor in threat_actors:
            actor_analysis.append({
                'actor_type': actor.get('type', 'unknown'),
                'capability': actor.get('capability', 'medium'),
                'motivation': actor.get('motivation', 'unknown'),
                'target_assets': actor.get('target_assets', [])
            })

        # Analyze attack scenarios
        scenario_analysis = []
        if attack_scenarios:
            for scenario in attack_scenarios:
                scenario_analysis.append({
                    'scenario': scenario.get('description', ''),
                    'likelihood': scenario.get('likelihood', 'medium'),
                    'impact': scenario.get('impact', 'medium'),
                    'mitigated': scenario.get('mitigated', False)
                })

        return {
            'ai_threat_model': {
                'actors': actor_analysis,
                'actor_types': list(set(a['actor_type'] for a in actor_analysis)),
                'scenarios': scenario_analysis
            },
            'threat_summary': {
                'total_actors': len(threat_actors),
                'total_scenarios': len(attack_scenarios) if attack_scenarios else 0,
                'high_capability_actors': sum(1 for a in actor_analysis if a['capability'] == 'high')
            }
        }


# ============================================================================
# Attack Surface Analyzer (Type 3)
# ============================================================================

class AttackSurfaceAnalyzer:
    """Analyzes attack surface mapping."""

    def analyze_attack_surface(self,
                              interfaces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map where the AI can be attacked."""
        if not interfaces:
            return {'attack_surface': [], 'total_vectors': 0}

        attack_vectors = []
        for interface in interfaces:
            vector = {
                'interface': interface.get('name', ''),
                'type': interface.get('type', ''),  # api, input, tool
                'exposed': interface.get('externally_exposed', False),
                'authentication': interface.get('authentication', 'none'),
                'risk_level': 'high' if interface.get('externally_exposed') and interface.get('authentication') == 'none' else 'medium'
            }
            attack_vectors.append(vector)

        high_risk = sum(1 for v in attack_vectors if v['risk_level'] == 'high')

        return {
            'attack_surface_diagram': attack_vectors,
            'summary': {
                'total_interfaces': len(interfaces),
                'externally_exposed': sum(1 for v in attack_vectors if v['exposed']),
                'high_risk_vectors': high_risk,
                'unauthenticated': sum(1 for v in attack_vectors if v['authentication'] == 'none')
            }
        }


# ============================================================================
# Data Integrity Analyzer (Type 4)
# ============================================================================

class DataIntegrityAnalyzer:
    """Analyzes data integrity and poisoning protection."""

    def analyze_data_integrity(self,
                              data_sources: List[Dict[str, Any]],
                              integrity_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if training data can be corrupted."""
        integrity_checks = []

        for source in data_sources:
            check = {
                'source': source.get('name', ''),
                'trust_level': source.get('trust_level', 'unknown'),
                'integrity_verification': source.get('integrity_checks', False),
                'access_control': source.get('access_control', 'none')
            }
            integrity_checks.append(check)

        # Analyze poisoning tests
        poisoning_resistance = 1.0
        if integrity_tests:
            successful_poisoning = sum(1 for t in integrity_tests if t.get('poisoning_successful', False))
            poisoning_resistance = 1 - (successful_poisoning / len(integrity_tests))

        return {
            'data_integrity_audit': integrity_checks,
            'poisoning_protection': {
                'poisoning_resistance_score': float(poisoning_resistance),
                'tests_conducted': len(integrity_tests) if integrity_tests else 0,
                'sources_with_integrity_checks': sum(1 for c in integrity_checks if c['integrity_verification'])
            }
        }


# ============================================================================
# Training Security Analyzer (Type 5)
# ============================================================================

class TrainingSecurityAnalyzer:
    """Analyzes training pipeline security."""

    def analyze_training_security(self,
                                 pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if training process is tamper-proof."""
        access_control = pipeline_config.get('access_control', 'none')
        artifact_signing = pipeline_config.get('artifact_signing', False)
        audit_logging = pipeline_config.get('audit_logging', False)
        environment_isolation = pipeline_config.get('environment_isolation', False)

        security_score = sum([
            access_control != 'none',
            artifact_signing,
            audit_logging,
            environment_isolation
        ]) / 4

        return {
            'training_security_report': {
                'access_control': access_control,
                'artifact_signing': artifact_signing,
                'audit_logging': audit_logging,
                'environment_isolation': environment_isolation,
                'security_score': float(security_score)
            }
        }


# ============================================================================
# Model Confidentiality Analyzer (Type 6)
# ============================================================================

class ModelConfidentialityAnalyzer:
    """Analyzes model confidentiality and IP protection."""

    def analyze_confidentiality(self,
                               model_exposure: Dict[str, Any],
                               extraction_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if model can be stolen."""
        returns_logits = model_exposure.get('returns_logits', False)
        returns_embeddings = model_exposure.get('returns_embeddings', False)
        rate_limited = model_exposure.get('rate_limited', False)

        # Risk factors
        risk_factors = []
        if returns_logits:
            risk_factors.append('Returns prediction logits')
        if returns_embeddings:
            risk_factors.append('Returns embeddings')
        if not rate_limited:
            risk_factors.append('No rate limiting')

        # Analyze extraction tests
        extraction_resistance = 1.0
        if extraction_tests:
            successful = sum(1 for t in extraction_tests if t.get('extraction_successful', False))
            extraction_resistance = 1 - (successful / len(extraction_tests))

        return {
            'model_theft_risk_assessment': {
                'risk_factors': risk_factors,
                'risk_level': 'high' if len(risk_factors) >= 2 else 'medium' if risk_factors else 'low',
                'extraction_resistance': float(extraction_resistance)
            },
            'ip_protection_measures': {
                'rate_limiting': rate_limited,
                'output_obfuscation': not returns_logits
            }
        }


# ============================================================================
# Adversarial Robustness Analyzer (Type 7)
# ============================================================================

class AdversarialRobustnessAnalyzer:
    """Analyzes adversarial input robustness."""

    def analyze_robustness(self,
                          perturbation_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if inputs can manipulate predictions."""
        if not perturbation_tests:
            return {'robustness_score': 0, 'evaluation': {}}

        robust = sum(1 for t in perturbation_tests if not t.get('attack_successful', False))
        robustness_score = robust / len(perturbation_tests)

        # By attack type
        by_type = defaultdict(lambda: {'total': 0, 'successful': 0})
        for test in perturbation_tests:
            attack_type = test.get('attack_type', 'unknown')
            by_type[attack_type]['total'] += 1
            if test.get('attack_successful', False):
                by_type[attack_type]['successful'] += 1

        return {
            'robustness_evaluation_report': {
                'total_tests': len(perturbation_tests),
                'robust_predictions': robust,
                'robustness_score': float(robustness_score),
                'by_attack_type': dict(by_type)
            }
        }


# ============================================================================
# Prompt Security Analyzer (Type 8)
# ============================================================================

class PromptSecurityAnalyzer:
    """Analyzes prompt injection and jailbreak defense."""

    def analyze_prompt_security(self,
                               injection_tests: List[Dict[str, Any]],
                               defenses: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze if instructions can be overridden."""
        if not injection_tests:
            return {'security_score': 0, 'assessment': {}}

        blocked = sum(1 for t in injection_tests if t.get('blocked', False))
        block_rate = blocked / len(injection_tests)

        # Analyze defense coverage
        defense_coverage = {}
        if defenses:
            defense_coverage = {
                'input_sanitization': defenses.get('input_sanitization', False),
                'prompt_hardening': defenses.get('prompt_hardening', False),
                'output_filtering': defenses.get('output_filtering', False)
            }

        return {
            'prompt_security_assessment': {
                'total_attacks': len(injection_tests),
                'blocked': blocked,
                'block_rate': float(block_rate),
                'defense_coverage': defense_coverage
            }
        }


# ============================================================================
# Output Safety Analyzer (Type 9)
# ============================================================================

class OutputSafetyAnalyzer:
    """Analyzes output safety and misuse prevention."""

    def analyze_output_safety(self,
                             output_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if outputs can cause harm."""
        if not output_tests:
            return {'safety_score': 0, 'mitigation': {}}

        harmful = sum(1 for t in output_tests if t.get('harmful', False))
        safety_rate = 1 - (harmful / len(output_tests))

        return {
            'output_risk_mitigation_plan': {
                'total_outputs_tested': len(output_tests),
                'harmful_outputs': harmful,
                'safety_rate': float(safety_rate),
                'abuse_scenarios_tested': list(set(t.get('abuse_type', '') for t in output_tests if t.get('abuse_type')))
            }
        }


# ============================================================================
# Additional Security Analyzers (Types 10-20)
# ============================================================================

class ToolSecurityAnalyzer:
    """Analyzes tool and action security (Type 10)."""

    def analyze_tool_security(self,
                             tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if tools can execute unsafe actions."""
        return {
            'tool_security_policy': {
                'total_tools': len(tools),
                'permission_boundaries': sum(1 for t in tools if t.get('permission_boundary', False)),
                'misuse_tested': sum(1 for t in tools if t.get('misuse_tested', False))
            }
        }


class AccessControlAnalyzer:
    """Analyzes access control and authentication (Type 11)."""

    def analyze_access_control(self,
                              access_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze who can access AI systems."""
        return {
            'access_control_audit': {
                'rbac_implemented': access_config.get('rbac', False),
                'least_privilege': access_config.get('least_privilege', False),
                'mfa_required': access_config.get('mfa', False),
                'access_review_frequency': access_config.get('review_frequency', 'never')
            }
        }


class InferenceSecurityAnalyzer:
    """Analyzes inference-time security (Type 12)."""

    def analyze_inference_security(self,
                                  inference_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if inference is protected from abuse."""
        return {
            'inference_security_controls': {
                'rate_limiting': inference_config.get('rate_limiting', False),
                'input_validation': inference_config.get('input_validation', False),
                'anomaly_detection': inference_config.get('anomaly_detection', False)
            }
        }


class PrivacyLeakageAnalyzer:
    """Analyzes privacy and information leakage defense (Type 13)."""

    def analyze_privacy_leakage(self,
                               leakage_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if private data can leak."""
        if not leakage_tests:
            return {'leakage_score': 0, 'report': {}}

        leaks = sum(1 for t in leakage_tests if t.get('leakage_detected', False))
        protection_rate = 1 - (leaks / len(leakage_tests))

        return {
            'privacy_leakage_report': {
                'total_tests': len(leakage_tests),
                'leaks_detected': leaks,
                'protection_rate': float(protection_rate)
            }
        }


class SupplyChainSecurityAnalyzer:
    """Analyzes supply chain and dependency security (Type 14)."""

    def analyze_supply_chain(self,
                            dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if dependencies are trustworthy."""
        vulnerable = sum(1 for d in dependencies if d.get('known_vulnerabilities', 0) > 0)
        verified = sum(1 for d in dependencies if d.get('provenance_verified', False))

        return {
            'supply_chain_security_register': {
                'total_dependencies': len(dependencies),
                'vulnerable_dependencies': vulnerable,
                'provenance_verified': verified,
                'security_score': verified / len(dependencies) if dependencies else 0
            }
        }


class SecurityDriftAnalyzer:
    """Analyzes monitoring and security drift detection (Type 15)."""

    def analyze_security_drift(self,
                              security_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if attacks are evolving over time."""
        if not security_history:
            return {'drift_detected': False, 'dashboard': {}}

        recent = security_history[-30:] if len(security_history) > 30 else security_history
        older = security_history[:-30] if len(security_history) > 30 else []

        recent_incidents = len(recent)
        older_incidents = len(older)

        trend = 'increasing' if recent_incidents > older_incidents * 1.2 else 'stable'

        return {
            'security_monitoring_dashboard': {
                'recent_incidents': recent_incidents,
                'trend': trend,
                'new_attack_patterns': list(set(
                    h.get('attack_type', '') for h in recent if h.get('attack_type') not in
                    [o.get('attack_type') for o in older]
                ))
            }
        }


class AuditTrailAnalyzer:
    """Analyzes logging, auditing and traceability (Type 16)."""

    def analyze_audit_trail(self,
                           logging_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if actions can be reconstructed."""
        return {
            'audit_trail': {
                'secure_logging': logging_config.get('secure_logging', False),
                'tamper_evidence': logging_config.get('tamper_evidence', False),
                'retention_days': logging_config.get('retention_days', 0),
                'completeness': logging_config.get('completeness', 0)
            }
        }


class IncidentResponseAnalyzer:
    """Analyzes incident detection and response (Type 17)."""

    def analyze_incident_response(self,
                                 incidents: List[SecurityIncident],
                                 response_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze how security incidents are handled."""
        if not incidents:
            return {'response_metrics': {}, 'plan': {}}

        response_times = [i.response_time_hours for i in incidents if i.response_time_hours > 0]
        resolved = sum(1 for i in incidents if i.resolved)

        return {
            'ai_security_incident_response_plan': {
                'total_incidents': len(incidents),
                'resolved': resolved,
                'avg_response_time_hours': float(np.mean(response_times)) if response_times else 0,
                'detection_sla': response_config.get('detection_sla_hours', 0) if response_config else 0
            }
        }


class PenetrationTestAnalyzer:
    """Analyzes penetration testing and red teaming (Type 18)."""

    def analyze_pen_test(self,
                        pen_test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if AI has been actively attacked."""
        if not pen_test_results:
            return {'pen_test_reports': [], 'coverage': 0}

        successful_attacks = sum(1 for r in pen_test_results if r.get('attack_successful', False))

        return {
            'pen_test_reports': pen_test_results,
            'summary': {
                'total_tests': len(pen_test_results),
                'successful_attacks': successful_attacks,
                'defense_rate': 1 - (successful_attacks / len(pen_test_results)) if pen_test_results else 0
            }
        }


class ResidualSecurityRiskAnalyzer:
    """Analyzes residual risk and risk acceptance (Type 19)."""

    def analyze_residual_risk(self,
                             risk_register: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what risks remain acceptable."""
        if not risk_register:
            return {'residual_risks': [], 'acceptance': []}

        unmitigated = [r for r in risk_register if not r.get('mitigated', False)]

        return {
            'residual_risk_register': unmitigated,
            'summary': {
                'total_risks': len(risk_register),
                'mitigated': len(risk_register) - len(unmitigated),
                'residual': len(unmitigated)
            }
        }


class SecurityGovernanceAnalyzer:
    """Analyzes secure AI governance and accountability (Type 20)."""

    def analyze_governance(self,
                          governance_config: Dict[str, Any],
                          audit_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze who enforces security decisions."""
        security_owner = governance_config.get('security_owner', '')
        escalation_authority = governance_config.get('escalation_authority', '')

        # Audit compliance
        compliance = 0
        if audit_records:
            compliance = sum(1 for a in audit_records if a.get('compliant', False)) / len(audit_records)

        return {
            'secure_ai_governance_policy': governance_config.get('policy', {}),
            'ownership': {
                'security_owner': security_owner,
                'escalation_authority': escalation_authority,
                'governance_score': 1.0 if security_owner and escalation_authority else 0.5
            },
            'audit_trail': {
                'total_audits': len(audit_records) if audit_records else 0,
                'compliance_rate': float(compliance)
            }
        }


# ============================================================================
# Report Generator
# ============================================================================

class SecurityReportGenerator:
    """Generates comprehensive security analysis reports."""

    def __init__(self):
        self.scope_analyzer = SecurityScopeAnalyzer()
        self.threat_analyzer = ThreatModelAnalyzer()
        self.surface_analyzer = AttackSurfaceAnalyzer()
        self.integrity_analyzer = DataIntegrityAnalyzer()
        self.robustness_analyzer = AdversarialRobustnessAnalyzer()
        self.prompt_analyzer = PromptSecurityAnalyzer()
        self.governance_analyzer = SecurityGovernanceAnalyzer()

    def generate_full_report(self,
                            assets: List[SecurityAsset] = None,
                            threat_actors: List[Dict[str, Any]] = None,
                            interfaces: List[Dict[str, Any]] = None,
                            perturbation_tests: List[Dict[str, Any]] = None,
                            injection_tests: List[Dict[str, Any]] = None,
                            governance_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            'report_type': 'security_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if assets:
            report['scope'] = self.scope_analyzer.analyze_scope(assets)

        if threat_actors:
            report['threat_model'] = self.threat_analyzer.analyze_threat_model(threat_actors)

        if interfaces:
            report['attack_surface'] = self.surface_analyzer.analyze_attack_surface(interfaces)

        if perturbation_tests:
            report['adversarial_robustness'] = self.robustness_analyzer.analyze_robustness(perturbation_tests)

        if injection_tests:
            report['prompt_security'] = self.prompt_analyzer.analyze_prompt_security(injection_tests)

        if governance_config:
            report['governance'] = self.governance_analyzer.analyze_governance(governance_config)

        # Calculate overall security score
        scores = []
        if 'adversarial_robustness' in report:
            scores.append(report['adversarial_robustness'].get('robustness_evaluation_report', {}).get('robustness_score', 0))
        if 'prompt_security' in report:
            scores.append(report['prompt_security'].get('prompt_security_assessment', {}).get('block_rate', 0))
        if 'governance' in report:
            scores.append(report['governance'].get('ownership', {}).get('governance_score', 0))

        report['overall_security_score'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
