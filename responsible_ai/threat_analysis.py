"""
Threat AI Analysis Module
==========================

Comprehensive analysis for AI threat modeling and security analysis.
Implements 20 analysis types for AI threat governance.

Analysis Types:
1. Threat Scope & Asset Identification
2. Threat Actor Identification
3. Attack Surface Mapping
4. Data Poisoning Threat Analysis
5. Prompt Injection & Input Manipulation
6. Model Extraction & Theft Threat
7. Membership Inference & Privacy Attacks
8. Adversarial Example Threat
9. Hallucination Exploitation Threat
10. Tool Abuse & API Misuse
11. Retrieval Poisoning (RAG)
12. Output Manipulation & Social Engineering
13. Availability & Denial-of-Service Threats
14. Supply Chain & Dependency Threats
15. Monitoring & Threat Drift Detection
16. Detection & Alerting Effectiveness
17. Incident Response & Containment
18. Threat Mitigation Controls
19. Residual Risk & Risk Acceptance
20. Threat Governance & Accountability
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
class ThreatActor:
    """Represents a potential threat actor."""
    actor_id: str
    actor_type: str  # external, insider, competitor, nation_state
    capability_level: str = "medium"  # low, medium, high, advanced
    motivation: str = ""
    resources: str = "medium"


@dataclass
class Asset:
    """Represents an AI asset under protection."""
    asset_id: str
    asset_type: str  # data, model, pipeline, api, output
    criticality: str = "medium"
    classification: str = "internal"


@dataclass
class Threat:
    """Represents a specific threat."""
    threat_id: str
    threat_type: str
    description: str = ""
    likelihood: str = "medium"
    impact: str = "medium"
    status: str = "active"


@dataclass
class Vulnerability:
    """Represents a vulnerability in the AI system."""
    vuln_id: str
    vuln_type: str
    attack_surface: str = ""
    exploitability: str = "medium"
    mitigations: List[str] = field(default_factory=list)


# ============================================================================
# Threat Scope & Asset Analysis (Type 1)
# ============================================================================

class ThreatScopeAnalyzer:
    """Analyzes threat scope and asset identification."""

    def analyze_scope(self,
                     assets: List[Asset],
                     business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Identify assets under attack and their criticality."""
        if not assets:
            return {'threat_scope': {}, 'asset_map': []}

        # Categorize assets
        asset_by_type = defaultdict(list)
        for asset in assets:
            asset_by_type[asset.asset_type].append(asset)

        # Criticality analysis
        critical_assets = [a for a in assets if a.criticality in ['high', 'critical']]

        # Business-critical outputs
        business_outputs = []
        if business_context:
            business_outputs = business_context.get('critical_outputs', [])

        return {
            'threat_scope': {
                'total_assets': len(assets),
                'critical_assets': len(critical_assets),
                'asset_types': {k: len(v) for k, v in asset_by_type.items()}
            },
            'asset_map': [
                {
                    'id': a.asset_id,
                    'type': a.asset_type,
                    'criticality': a.criticality,
                    'classification': a.classification
                } for a in assets
            ],
            'business_critical_outputs': business_outputs,
            'scope_statement': f"Protecting {len(assets)} assets across {len(asset_by_type)} categories"
        }


# ============================================================================
# Threat Actor Identification (Type 2)
# ============================================================================

class ThreatActorAnalyzer:
    """Analyzes potential threat actors."""

    ACTOR_PROFILES = {
        'external': {
            'typical_capability': 'medium',
            'typical_motivation': 'financial_gain',
            'common_techniques': ['phishing', 'social_engineering', 'public_exploits']
        },
        'insider': {
            'typical_capability': 'high',
            'typical_motivation': 'data_theft',
            'common_techniques': ['privilege_abuse', 'data_exfiltration']
        },
        'competitor': {
            'typical_capability': 'high',
            'typical_motivation': 'competitive_advantage',
            'common_techniques': ['model_extraction', 'trade_secret_theft']
        },
        'nation_state': {
            'typical_capability': 'advanced',
            'typical_motivation': 'espionage',
            'common_techniques': ['advanced_persistent_threat', 'zero_day_exploits']
        }
    }

    def analyze_actors(self,
                      identified_actors: List[ThreatActor] = None,
                      industry_context: str = 'general') -> Dict[str, Any]:
        """Identify who could attack the AI."""
        # Default actors based on industry
        if not identified_actors:
            identified_actors = [
                ThreatActor('actor_1', 'external', 'medium', 'financial_gain', 'low'),
                ThreatActor('actor_2', 'insider', 'high', 'data_theft', 'medium')
            ]

        actor_profiles = []
        for actor in identified_actors:
            profile_template = self.ACTOR_PROFILES.get(actor.actor_type, {})
            actor_profiles.append({
                'actor_id': actor.actor_id,
                'actor_type': actor.actor_type,
                'capability_level': actor.capability_level,
                'motivation': actor.motivation or profile_template.get('typical_motivation', 'unknown'),
                'resources': actor.resources,
                'common_techniques': profile_template.get('common_techniques', [])
            })

        # Risk prioritization
        risk_priority = sorted(actor_profiles, key=lambda x: {
            'advanced': 4, 'high': 3, 'medium': 2, 'low': 1
        }.get(x['capability_level'], 0), reverse=True)

        return {
            'threat_actor_profiles': actor_profiles,
            'actor_count_by_type': {
                t: sum(1 for a in actor_profiles if a['actor_type'] == t)
                for t in self.ACTOR_PROFILES.keys()
            },
            'risk_prioritization': [a['actor_id'] for a in risk_priority],
            'highest_capability_actors': [a for a in actor_profiles if a['capability_level'] in ['high', 'advanced']]
        }


# ============================================================================
# Attack Surface Mapping (Type 3)
# ============================================================================

class AttackSurfaceAnalyzer:
    """Analyzes attack surface mapping."""

    def map_attack_surface(self,
                          system_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map where attacks can occur."""
        if not system_components:
            return {'attack_surface': [], 'total_entry_points': 0}

        attack_surfaces = []
        entry_points = 0

        for component in system_components:
            component_name = component.get('name', 'unknown')
            component_type = component.get('type', 'unknown')
            interfaces = component.get('interfaces', [])
            exposed = component.get('externally_exposed', False)

            surfaces = []
            for interface in interfaces:
                surfaces.append({
                    'interface': interface.get('name', ''),
                    'protocol': interface.get('protocol', ''),
                    'authentication': interface.get('authentication', 'none'),
                    'risk_level': 'high' if exposed and interface.get('authentication') == 'none' else 'medium'
                })
                entry_points += 1

            attack_surfaces.append({
                'component': component_name,
                'type': component_type,
                'externally_exposed': exposed,
                'interfaces': surfaces,
                'entry_point_count': len(surfaces)
            })

        # Identify highest risk surfaces
        high_risk = [s for s in attack_surfaces
                    if any(i.get('risk_level') == 'high' for i in s.get('interfaces', []))]

        return {
            'attack_surface_diagram': attack_surfaces,
            'total_entry_points': entry_points,
            'high_risk_surfaces': high_risk,
            'entry_point_types': {
                'data_ingestion': sum(1 for s in attack_surfaces if s['type'] == 'data_ingestion'),
                'api': sum(1 for s in attack_surfaces if s['type'] == 'api'),
                'model_endpoint': sum(1 for s in attack_surfaces if s['type'] == 'model_endpoint'),
                'tool_integration': sum(1 for s in attack_surfaces if s['type'] == 'tool_integration')
            }
        }


# ============================================================================
# Data Poisoning Analysis (Type 4)
# ============================================================================

class DataPoisoningAnalyzer:
    """Analyzes data poisoning threats."""

    def analyze_poisoning_risk(self,
                              data_sources: List[Dict[str, Any]],
                              poisoning_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if training data can be maliciously altered."""
        if not data_sources:
            return {'poisoning_risk': 'unknown', 'source_trust_analysis': []}

        source_analysis = []
        total_trust_score = 0

        for source in data_sources:
            source_name = source.get('name', 'unknown')
            source_type = source.get('type', 'unknown')  # internal, external, user_generated
            access_control = source.get('access_control', 'none')
            integrity_checks = source.get('integrity_checks', False)

            # Calculate trust score
            trust_factors = {
                'internal': 0.3, 'external': 0.1, 'user_generated': 0.0
            }
            access_factors = {
                'strict': 0.3, 'moderate': 0.2, 'none': 0.0
            }

            trust_score = trust_factors.get(source_type, 0.1) + \
                         access_factors.get(access_control, 0) + \
                         (0.3 if integrity_checks else 0)

            total_trust_score += trust_score

            source_analysis.append({
                'source': source_name,
                'type': source_type,
                'trust_score': float(trust_score),
                'vulnerabilities': self._identify_vulnerabilities(source)
            })

        avg_trust = total_trust_score / len(data_sources) if data_sources else 0

        # Analyze poisoning tests
        poisoning_results = {}
        if poisoning_tests:
            successful_poisoning = sum(1 for t in poisoning_tests if t.get('poisoning_successful', False))
            poisoning_results = {
                'tests_run': len(poisoning_tests),
                'successful_poisoning': successful_poisoning,
                'resistance_rate': 1 - (successful_poisoning / len(poisoning_tests))
            }

        return {
            'data_poisoning_risk_report': {
                'overall_risk': 'high' if avg_trust < 0.5 else 'medium' if avg_trust < 0.7 else 'low',
                'avg_trust_score': float(avg_trust)
            },
            'source_trust_analysis': source_analysis,
            'poisoning_test_results': poisoning_results,
            'attack_vectors': {
                'label_poisoning': any(s['type'] == 'user_generated' for s in source_analysis),
                'backdoor_insertion': avg_trust < 0.5,
                'data_manipulation': any(s['trust_score'] < 0.3 for s in source_analysis)
            }
        }

    def _identify_vulnerabilities(self, source: Dict) -> List[str]:
        vulns = []
        if source.get('access_control') == 'none':
            vulns.append('No access control')
        if not source.get('integrity_checks'):
            vulns.append('No integrity verification')
        if source.get('type') == 'external':
            vulns.append('External source dependency')
        return vulns


# ============================================================================
# Prompt Injection Analysis (Type 5)
# ============================================================================

class PromptInjectionAnalyzer:
    """Analyzes prompt injection and input manipulation threats."""

    def analyze_prompt_injection(self,
                                injection_tests: List[Dict[str, Any]],
                                defenses: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze if inputs can override system intent."""
        if not injection_tests:
            return {'injection_risk': 'unknown', 'assessment': {}}

        successful_injections = []
        blocked_injections = []

        for test in injection_tests:
            injection_type = test.get('type', 'unknown')  # jailbreak, instruction_hijack, context_override
            success = test.get('successful', False)
            payload = test.get('payload', '')[:100]

            if success:
                successful_injections.append({
                    'type': injection_type,
                    'payload_preview': payload
                })
            else:
                blocked_injections.append({
                    'type': injection_type,
                    'blocked_by': test.get('blocked_by', 'unknown')
                })

        success_rate = len(successful_injections) / len(injection_tests)

        # Evaluate defenses
        defense_coverage = {}
        if defenses:
            defense_coverage = {
                'input_validation': defenses.get('input_validation', False),
                'prompt_hardening': defenses.get('prompt_hardening', False),
                'output_filtering': defenses.get('output_filtering', False),
                'rate_limiting': defenses.get('rate_limiting', False)
            }

        return {
            'prompt_attack_assessment': {
                'total_tests': len(injection_tests),
                'successful_injections': len(successful_injections),
                'blocked_injections': len(blocked_injections),
                'success_rate': float(success_rate),
                'risk_level': 'critical' if success_rate > 0.3 else 'high' if success_rate > 0.1 else 'medium'
            },
            'successful_attack_types': successful_injections,
            'defense_coverage': defense_coverage,
            'vulnerability_by_type': {
                'jailbreak': sum(1 for s in successful_injections if s['type'] == 'jailbreak'),
                'instruction_hijacking': sum(1 for s in successful_injections if s['type'] == 'instruction_hijack'),
                'context_override': sum(1 for s in successful_injections if s['type'] == 'context_override')
            }
        }


# ============================================================================
# Model Extraction Analysis (Type 6)
# ============================================================================

class ModelExtractionAnalyzer:
    """Analyzes model extraction and theft threats."""

    def analyze_extraction_risk(self,
                               api_config: Dict[str, Any],
                               query_logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if model can be stolen or replicated."""
        # API exposure analysis
        rate_limit = api_config.get('rate_limit', 0)
        query_limit = api_config.get('query_limit_per_user', 0)
        returns_probabilities = api_config.get('returns_probabilities', False)
        returns_embeddings = api_config.get('returns_embeddings', False)

        # Risk factors
        risk_factors = []
        if rate_limit == 0:
            risk_factors.append('No rate limiting')
        if returns_probabilities:
            risk_factors.append('Returns prediction probabilities')
        if returns_embeddings:
            risk_factors.append('Returns embeddings')
        if query_limit == 0 or query_limit > 10000:
            risk_factors.append('High or unlimited query allowance')

        # Analyze query patterns for extraction attempts
        extraction_attempts = []
        if query_logs:
            # Look for systematic querying patterns
            user_queries = defaultdict(list)
            for log in query_logs:
                user_id = log.get('user_id', 'unknown')
                user_queries[user_id].append(log)

            for user_id, queries in user_queries.items():
                if len(queries) > 1000:  # Suspicious volume
                    extraction_attempts.append({
                        'user_id': user_id,
                        'query_count': len(queries),
                        'pattern': 'high_volume'
                    })

        risk_level = 'critical' if len(risk_factors) >= 3 else 'high' if len(risk_factors) >= 2 else 'medium'

        return {
            'model_theft_risk_report': {
                'risk_level': risk_level,
                'risk_factors': risk_factors
            },
            'api_exposure_analysis': {
                'rate_limit': rate_limit,
                'query_limit_per_user': query_limit,
                'exposes_probabilities': returns_probabilities,
                'exposes_embeddings': returns_embeddings
            },
            'extraction_attempts_detected': extraction_attempts,
            'ip_leakage_risk': returns_probabilities or returns_embeddings
        }


# ============================================================================
# Privacy Attack Analysis (Type 7)
# ============================================================================

class PrivacyAttackAnalyzer:
    """Analyzes membership inference and privacy attacks."""

    def analyze_privacy_attacks(self,
                               privacy_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if training data can be inferred."""
        if not privacy_tests:
            return {'privacy_risk': 'unknown', 'assessment': {}}

        membership_inference = [t for t in privacy_tests if t.get('attack_type') == 'membership_inference']
        attribute_inference = [t for t in privacy_tests if t.get('attack_type') == 'attribute_inference']

        # Membership inference analysis
        mi_success = sum(1 for t in membership_inference if t.get('successful', False))
        mi_rate = mi_success / len(membership_inference) if membership_inference else 0

        # Attribute inference analysis
        ai_success = sum(1 for t in attribute_inference if t.get('successful', False))
        ai_rate = ai_success / len(attribute_inference) if attribute_inference else 0

        overall_risk = 'critical' if mi_rate > 0.7 else 'high' if mi_rate > 0.5 else 'medium'

        return {
            'privacy_attack_assessment': {
                'overall_risk': overall_risk
            },
            'membership_inference_tests': {
                'total_tests': len(membership_inference),
                'successful': mi_success,
                'success_rate': float(mi_rate)
            },
            'attribute_inference_tests': {
                'total_tests': len(attribute_inference),
                'successful': ai_success,
                'success_rate': float(ai_rate)
            },
            'recommendations': self._privacy_recommendations(mi_rate, ai_rate)
        }

    def _privacy_recommendations(self, mi_rate: float, ai_rate: float) -> List[str]:
        recs = []
        if mi_rate > 0.5:
            recs.append('Implement differential privacy during training')
        if ai_rate > 0.3:
            recs.append('Apply output perturbation')
        return recs


# ============================================================================
# Additional Threat Analyzers (Types 8-20)
# ============================================================================

class AdversarialExampleAnalyzer:
    """Analyzes adversarial example threats (Type 8)."""

    def analyze_adversarial_robustness(self,
                                      adversarial_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if small perturbations break predictions."""
        if not adversarial_tests:
            return {'robustness_score': 0, 'report': {}}

        successful_attacks = sum(1 for t in adversarial_tests if t.get('attack_successful', False))
        robustness_score = 1 - (successful_attacks / len(adversarial_tests))

        attack_by_type = defaultdict(int)
        for test in adversarial_tests:
            if test.get('attack_successful'):
                attack_by_type[test.get('attack_type', 'unknown')] += 1

        return {
            'adversarial_robustness_report': {
                'total_tests': len(adversarial_tests),
                'successful_attacks': successful_attacks,
                'robustness_score': float(robustness_score)
            },
            'attack_success_by_type': dict(attack_by_type),
            'vulnerability_level': 'high' if robustness_score < 0.7 else 'medium' if robustness_score < 0.9 else 'low'
        }


class HallucinationExploitAnalyzer:
    """Analyzes hallucination exploitation threats (Type 9)."""

    def analyze_hallucination_exploit(self,
                                     exploit_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if hallucinations can be weaponized."""
        if not exploit_tests:
            return {'exploitation_risk': 'unknown', 'report': {}}

        exploitable = sum(1 for t in exploit_tests if t.get('exploitable', False))
        exploitation_rate = exploitable / len(exploit_tests)

        return {
            'hallucination_misuse_risk': {
                'risk_level': 'high' if exploitation_rate > 0.3 else 'medium',
                'exploitation_rate': float(exploitation_rate),
                'total_tests': len(exploit_tests)
            },
            'exploit_types': {
                'false_authority': sum(1 for t in exploit_tests if t.get('exploit_type') == 'false_authority'),
                'fabricated_citations': sum(1 for t in exploit_tests if t.get('exploit_type') == 'fabricated_citations')
            }
        }


class ToolAbuseAnalyzer:
    """Analyzes tool abuse and API misuse (Type 10)."""

    def analyze_tool_abuse(self,
                          tool_usage_logs: List[Dict[str, Any]],
                          tool_policies: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze if tools can be exploited."""
        if not tool_usage_logs:
            return {'abuse_risk': 'unknown', 'report': {}}

        abuse_incidents = [l for l in tool_usage_logs if l.get('flagged_abuse', False)]
        unauthorized_actions = [l for l in tool_usage_logs if l.get('unauthorized', False)]

        return {
            'tool_abuse_threat_model': {
                'total_tool_calls': len(tool_usage_logs),
                'flagged_abuse': len(abuse_incidents),
                'unauthorized_actions': len(unauthorized_actions),
                'abuse_rate': len(abuse_incidents) / len(tool_usage_logs) if tool_usage_logs else 0
            },
            'tool_chaining_abuse': sum(1 for l in abuse_incidents if l.get('abuse_type') == 'chaining'),
            'policy_violations': len(unauthorized_actions)
        }


class RAGPoisoningAnalyzer:
    """Analyzes retrieval poisoning threats (Type 11)."""

    def analyze_rag_poisoning(self,
                             knowledge_base_audit: Dict[str, Any],
                             poisoning_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if knowledge base can be corrupted."""
        doc_count = knowledge_base_audit.get('document_count', 0)
        verified_sources = knowledge_base_audit.get('verified_sources_percent', 0)
        access_control = knowledge_base_audit.get('write_access_control', 'none')

        risk_level = 'high' if verified_sources < 50 or access_control == 'none' else 'medium'

        # Poisoning test results
        if poisoning_tests:
            successful = sum(1 for t in poisoning_tests if t.get('poisoning_successful', False))
            test_results = {
                'tests_run': len(poisoning_tests),
                'successful_poisoning': successful,
                'resistance_rate': 1 - (successful / len(poisoning_tests))
            }
        else:
            test_results = {}

        return {
            'rag_poisoning_audit': {
                'document_count': doc_count,
                'verified_sources_percent': verified_sources,
                'write_access_control': access_control,
                'risk_level': risk_level
            },
            'poisoning_vectors': {
                'malicious_documents': verified_sources < 80,
                'embedding_poisoning': access_control == 'none'
            },
            'test_results': test_results
        }


class SocialEngineeringAnalyzer:
    """Analyzes output manipulation and social engineering (Type 12)."""

    def analyze_social_engineering(self,
                                  manipulation_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if outputs can deceive users."""
        if not manipulation_tests:
            return {'manipulation_risk': 'unknown', 'report': {}}

        persuasive = sum(1 for t in manipulation_tests if t.get('persuasive_manipulation', False))
        fraud_risk = sum(1 for t in manipulation_tests if t.get('fraud_potential', False))

        return {
            'social_engineering_threat_report': {
                'total_tests': len(manipulation_tests),
                'persuasive_manipulation_detected': persuasive,
                'fraud_amplification_risk': fraud_risk,
                'risk_level': 'high' if persuasive > len(manipulation_tests) * 0.3 else 'medium'
            }
        }


class AvailabilityThreatAnalyzer:
    """Analyzes availability and DoS threats (Type 13)."""

    def analyze_availability(self,
                            load_tests: List[Dict[str, Any]],
                            rate_limits: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze if AI can be disrupted."""
        if not load_tests:
            return {'availability_risk': 'unknown', 'assessment': {}}

        failures_under_load = sum(1 for t in load_tests if t.get('failed', False))
        max_handled_qps = max(t.get('qps_handled', 0) for t in load_tests) if load_tests else 0

        # Rate limit protection
        has_rate_limits = rate_limits is not None and rate_limits.get('enabled', False)

        return {
            'availability_risk_assessment': {
                'failure_rate_under_load': failures_under_load / len(load_tests) if load_tests else 0,
                'max_qps_handled': max_handled_qps,
                'has_rate_limiting': has_rate_limits,
                'risk_level': 'high' if not has_rate_limits else 'medium'
            },
            'dos_vectors': {
                'query_flooding': not has_rate_limits,
                'resource_exhaustion': failures_under_load > 0
            }
        }


class SupplyChainThreatAnalyzer:
    """Analyzes supply chain threats (Type 14)."""

    def analyze_supply_chain(self,
                            dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if vendors are a weak point."""
        if not dependencies:
            return {'supply_chain_risk': 'unknown', 'register': []}

        threat_register = []
        high_risk_deps = 0

        for dep in dependencies:
            risk_factors = []
            if dep.get('source') == 'open_source' and not dep.get('vetted', False):
                risk_factors.append('Unvetted open source')
            if dep.get('last_update_months', 0) > 12:
                risk_factors.append('Outdated dependency')
            if not dep.get('provenance_verified', False):
                risk_factors.append('Unverified provenance')

            risk_level = 'high' if len(risk_factors) >= 2 else 'medium' if len(risk_factors) == 1 else 'low'
            if risk_level == 'high':
                high_risk_deps += 1

            threat_register.append({
                'dependency': dep.get('name', 'unknown'),
                'type': dep.get('type', 'unknown'),
                'risk_level': risk_level,
                'risk_factors': risk_factors
            })

        return {
            'supply_chain_threat_register': threat_register,
            'summary': {
                'total_dependencies': len(dependencies),
                'high_risk': high_risk_deps,
                'open_source_risk': sum(1 for d in dependencies if d.get('source') == 'open_source'),
                'model_dependency_risk': sum(1 for d in dependencies if d.get('type') == 'model')
            }
        }


class ThreatDriftMonitor:
    """Monitors threat drift over time (Type 15)."""

    def monitor_threat_drift(self,
                            threat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if threats are evolving over time."""
        if not threat_history:
            return {'drift_detected': False, 'dashboard': {}}

        # Analyze threat trends
        recent_threats = sorted(threat_history, key=lambda x: x.get('detected_at', ''))[-30:]
        older_threats = sorted(threat_history, key=lambda x: x.get('detected_at', ''))[:-30]

        recent_count = len(recent_threats)
        older_count = len(older_threats)

        drift_detected = recent_count > older_count * 1.5 if older_count > 0 else False

        # New attack patterns
        recent_types = set(t.get('threat_type', '') for t in recent_threats)
        older_types = set(t.get('threat_type', '') for t in older_threats)
        new_patterns = recent_types - older_types

        return {
            'drift_detected': drift_detected,
            'threat_monitoring_dashboard': {
                'recent_threat_count': recent_count,
                'trend': 'increasing' if drift_detected else 'stable',
                'new_attack_patterns': list(new_patterns),
                'abuse_trend': 'increasing' if recent_count > older_count else 'stable'
            }
        }


class DetectionEffectivenessAnalyzer:
    """Analyzes detection and alerting effectiveness (Type 16)."""

    def analyze_detection(self,
                         detection_logs: List[Dict[str, Any]],
                         ground_truth: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if attacks can be detected early."""
        if not detection_logs:
            return {'detection_effectiveness': 0, 'report': {}}

        alerts_raised = len(detection_logs)

        # If ground truth available, calculate precision/recall
        if ground_truth:
            true_positives = sum(1 for d in detection_logs if d.get('confirmed_attack', False))
            false_positives = alerts_raised - true_positives
            total_attacks = len(ground_truth)
            false_negatives = total_attacks - true_positives

            precision = true_positives / alerts_raised if alerts_raised > 0 else 0
            recall = true_positives / total_attacks if total_attacks > 0 else 0
        else:
            precision = 0
            recall = 0
            false_positives = 0
            false_negatives = 0

        return {
            'detection_effectiveness_report': {
                'alerts_raised': alerts_raised,
                'precision': float(precision),
                'recall': float(recall),
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'signal_to_noise': precision
            }
        }


class IncidentResponseAnalyzer:
    """Analyzes incident response and containment (Type 17)."""

    def analyze_incident_response(self,
                                 incident_logs: List[Dict[str, Any]],
                                 response_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze what happens during an attack."""
        if not incident_logs:
            return {'response_effectiveness': 0, 'plan': {}}

        response_times = [i.get('response_time_minutes', 0) for i in incident_logs]
        contained = sum(1 for i in incident_logs if i.get('contained', False))

        avg_response = np.mean(response_times) if response_times else 0
        containment_rate = contained / len(incident_logs) if incident_logs else 0

        return {
            'ai_incident_response_plan': {
                'total_incidents': len(incident_logs),
                'contained': contained,
                'containment_rate': float(containment_rate),
                'avg_response_time_minutes': float(avg_response)
            },
            'response_capabilities': {
                'kill_switch': response_config.get('kill_switch', False) if response_config else False,
                'rollback': response_config.get('rollback', False) if response_config else False,
                'user_notification': response_config.get('user_notification', False) if response_config else False
            }
        }


class ThreatMitigationAnalyzer:
    """Analyzes threat mitigation controls (Type 18)."""

    def analyze_mitigations(self,
                           controls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how threats are reduced."""
        if not controls:
            return {'mitigation_coverage': 0, 'matrix': []}

        implemented = sum(1 for c in controls if c.get('implemented', False))
        effective = sum(1 for c in controls if c.get('effectiveness', 0) > 0.7)

        return {
            'threat_mitigation_matrix': controls,
            'summary': {
                'total_controls': len(controls),
                'implemented': implemented,
                'effective': effective,
                'coverage': implemented / len(controls) if controls else 0
            },
            'control_types': {
                'rate_limiting': any(c.get('type') == 'rate_limiting' for c in controls),
                'input_validation': any(c.get('type') == 'input_validation' for c in controls),
                'isolation': any(c.get('type') == 'isolation' for c in controls)
            }
        }


class ResidualRiskAnalyzer:
    """Analyzes residual risk and risk acceptance (Type 19)."""

    def analyze_residual_risk(self,
                             risk_register: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what threats remain acceptable."""
        if not risk_register:
            return {'residual_risks': [], 'acceptance_required': []}

        residual_risks = []
        for risk in risk_register:
            likelihood = risk.get('likelihood', 'medium')
            impact = risk.get('impact', 'medium')
            mitigated = risk.get('mitigated', False)

            if not mitigated:
                risk_score = {'low': 1, 'medium': 2, 'high': 3}.get(likelihood, 2) * \
                            {'low': 1, 'medium': 2, 'high': 3}.get(impact, 2)

                residual_risks.append({
                    'risk': risk.get('description', ''),
                    'likelihood': likelihood,
                    'impact': impact,
                    'risk_score': risk_score,
                    'acceptance_recommended': risk_score <= 4
                })

        return {
            'residual_risk_register': residual_risks,
            'risks_requiring_acceptance': [r for r in residual_risks if r['risk_score'] > 4],
            'total_residual_risks': len(residual_risks)
        }


class ThreatGovernanceAnalyzer:
    """Analyzes threat governance and accountability (Type 20)."""

    def analyze_governance(self,
                          governance_config: Dict[str, Any],
                          audit_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze who owns threat decisions."""
        has_owner = governance_config.get('security_owner') is not None
        has_escalation = governance_config.get('escalation_authority') is not None
        has_review_cadence = governance_config.get('review_cadence') is not None

        governance_score = sum([has_owner, has_escalation, has_review_cadence]) / 3

        # Audit compliance
        if audit_records:
            compliant = sum(1 for a in audit_records if a.get('compliant', False))
            compliance_rate = compliant / len(audit_records)
        else:
            compliance_rate = 0

        return {
            'threat_governance_policy': governance_config.get('policy_document'),
            'governance_elements': {
                'security_owner_assigned': has_owner,
                'escalation_authority_defined': has_escalation,
                'review_cadence_set': has_review_cadence
            },
            'governance_score': float(governance_score),
            'audit_trail': {
                'total_audits': len(audit_records) if audit_records else 0,
                'compliance_rate': float(compliance_rate)
            }
        }


# ============================================================================
# Report Generator
# ============================================================================

class ThreatReportGenerator:
    """Generates comprehensive threat analysis reports."""

    def __init__(self):
        self.scope_analyzer = ThreatScopeAnalyzer()
        self.actor_analyzer = ThreatActorAnalyzer()
        self.surface_analyzer = AttackSurfaceAnalyzer()
        self.poisoning_analyzer = DataPoisoningAnalyzer()
        self.injection_analyzer = PromptInjectionAnalyzer()
        self.governance_analyzer = ThreatGovernanceAnalyzer()

    def generate_full_report(self,
                            assets: List[Asset] = None,
                            system_components: List[Dict[str, Any]] = None,
                            data_sources: List[Dict[str, Any]] = None,
                            injection_tests: List[Dict[str, Any]] = None,
                            governance_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive threat analysis report."""
        report = {
            'report_type': 'threat_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if assets:
            report['scope'] = self.scope_analyzer.analyze_scope(assets)

        report['actors'] = self.actor_analyzer.analyze_actors()

        if system_components:
            report['attack_surface'] = self.surface_analyzer.map_attack_surface(system_components)

        if data_sources:
            report['data_poisoning'] = self.poisoning_analyzer.analyze_poisoning_risk(data_sources)

        if injection_tests:
            report['prompt_injection'] = self.injection_analyzer.analyze_prompt_injection(injection_tests)

        if governance_config:
            report['governance'] = self.governance_analyzer.analyze_governance(governance_config)

        # Calculate overall threat score
        risk_factors = []
        if 'data_poisoning' in report:
            risk_factors.append(report['data_poisoning'].get('data_poisoning_risk_report', {}).get('overall_risk', 'medium'))
        if 'prompt_injection' in report:
            risk_factors.append(report['prompt_injection'].get('prompt_attack_assessment', {}).get('risk_level', 'medium'))

        risk_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        avg_risk = np.mean([risk_scores.get(r, 2) for r in risk_factors]) if risk_factors else 2

        report['overall_threat_level'] = 'critical' if avg_risk >= 3.5 else 'high' if avg_risk >= 2.5 else 'medium' if avg_risk >= 1.5 else 'low'

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
