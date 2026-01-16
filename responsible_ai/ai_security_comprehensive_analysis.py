"""
AI Security Comprehensive Analysis Module
==========================================

Comprehensive AI security analysis framework covering threats and mitigations
across ML, DL, CV, NLP, and RAG systems.

Categories:
1. ML Security - Data poisoning, model extraction, membership inference
2. DL Security - Adversarial examples, backdoor attacks, gradient attacks
3. CV Security - Image perturbations, patch attacks, physical attacks
4. NLP Security - Prompt injection, jailbreaking, data extraction
5. RAG Security - Knowledge poisoning, retrieval attacks, context manipulation
6. Model Security - Model theft, reverse engineering, watermarking
7. Data Security - Training data extraction, privacy leakage
8. Infrastructure Security - API security, deployment vulnerabilities
9. Supply Chain Security - Dependency attacks, model tampering
10. Incident Response - Detection, mitigation, recovery
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime


# =============================================================================
# ENUMS
# =============================================================================

class AttackDomain(Enum):
    """AI security attack domains."""
    ML = auto()
    DEEP_LEARNING = auto()
    COMPUTER_VISION = auto()
    NLP = auto()
    RAG = auto()
    INFRASTRUCTURE = auto()
    SUPPLY_CHAIN = auto()


class AttackVector(Enum):
    """Attack vectors for AI systems."""
    DATA_POISONING = auto()
    MODEL_EXTRACTION = auto()
    MEMBERSHIP_INFERENCE = auto()
    ADVERSARIAL_EXAMPLES = auto()
    BACKDOOR_INJECTION = auto()
    PROMPT_INJECTION = auto()
    JAILBREAKING = auto()
    DATA_EXTRACTION = auto()
    MODEL_INVERSION = auto()
    GRADIENT_LEAKAGE = auto()
    EVASION = auto()
    INFERENCE_ATTACK = auto()
    KNOWLEDGE_POISONING = auto()
    RETRIEVAL_MANIPULATION = auto()


class ThreatSeverity(Enum):
    """Threat severity levels."""
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFORMATIONAL = auto()


class MitigationStatus(Enum):
    """Status of security mitigations."""
    IMPLEMENTED = auto()
    PARTIAL = auto()
    PLANNED = auto()
    NOT_IMPLEMENTED = auto()
    NOT_APPLICABLE = auto()


class SecurityPosture(Enum):
    """Overall security posture."""
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    WEAK = auto()
    CRITICAL = auto()


class AttackPhase(Enum):
    """Phases of AI attack lifecycle."""
    RECONNAISSANCE = auto()
    WEAPONIZATION = auto()
    DELIVERY = auto()
    EXPLOITATION = auto()
    INSTALLATION = auto()
    COMMAND_CONTROL = auto()
    ACTIONS_ON_OBJECTIVE = auto()


class DefenseLayer(Enum):
    """Defense-in-depth layers."""
    DATA = auto()
    MODEL = auto()
    INFERENCE = auto()
    APPLICATION = auto()
    NETWORK = auto()
    PHYSICAL = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ThreatProfile:
    """Profile of a security threat."""
    threat_id: str
    name: str
    description: str
    attack_vector: AttackVector
    domain: AttackDomain
    severity: ThreatSeverity
    likelihood: float
    impact: float
    risk_score: float
    affected_components: List[str] = field(default_factory=list)
    indicators_of_compromise: List[str] = field(default_factory=list)


@dataclass
class VulnerabilityAssessment:
    """Assessment of a specific vulnerability."""
    vuln_id: str
    name: str
    description: str
    severity: ThreatSeverity
    exploitability: float
    affected_versions: List[str] = field(default_factory=list)
    cve_references: List[str] = field(default_factory=list)
    mitigation_available: bool = False
    mitigation_description: str = ""


@dataclass
class MitigationControl:
    """Security mitigation control."""
    control_id: str
    name: str
    description: str
    defense_layer: DefenseLayer
    status: MitigationStatus
    effectiveness: float
    implementation_cost: str = "medium"
    mitigates_threats: List[str] = field(default_factory=list)


@dataclass
class AdversarialAttackResult:
    """Result from adversarial attack testing."""
    attack_type: AttackVector
    success_rate: float
    avg_perturbation_magnitude: float
    detection_rate: float
    samples_tested: int
    successful_samples: int
    failed_samples: int


@dataclass
class PromptInjectionResult:
    """Result from prompt injection testing."""
    injection_type: str
    test_prompts: int
    successful_injections: int
    bypassed_filters: int
    success_rate: float
    vulnerability_patterns: List[str] = field(default_factory=list)


@dataclass
class DataLeakageResult:
    """Result from data leakage testing."""
    leakage_type: str
    exposure_risk: float
    sensitive_data_found: bool
    pii_exposed: int
    training_data_recovered: float
    mitigation_recommendations: List[str] = field(default_factory=list)


@dataclass
class ModelExtractionResult:
    """Result from model extraction attack testing."""
    queries_used: int
    fidelity_achieved: float
    accuracy_of_clone: float
    detection_triggered: bool
    extraction_method: str


@dataclass
class BackdoorDetectionResult:
    """Result from backdoor detection."""
    backdoor_detected: bool
    confidence: float
    trigger_patterns: List[str] = field(default_factory=list)
    affected_classes: List[int] = field(default_factory=list)
    activation_rate: float = 0.0


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    timestamp: datetime
    attack_vector: AttackVector
    domain: AttackDomain
    severity: ThreatSeverity
    description: str
    affected_systems: List[str]
    impact_assessment: str
    root_cause: str = ""
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time_hours: float = 0.0


@dataclass
class SecurityMetrics:
    """Security metrics for a domain."""
    domain: AttackDomain
    threats_identified: int
    vulnerabilities_found: int
    controls_implemented: int
    controls_effective: int
    attack_success_rate: float
    detection_rate: float
    mean_time_to_detect: float
    mean_time_to_respond: float


@dataclass
class SecurityAssessment:
    """Comprehensive security assessment."""
    assessment_id: str
    timestamp: datetime
    domains_assessed: List[AttackDomain]
    overall_posture: SecurityPosture
    risk_score: float
    threats_by_severity: Dict[str, int]
    vulnerabilities_by_severity: Dict[str, int]
    controls_coverage: float
    critical_findings: List[str]
    recommendations: List[str]
    domain_scores: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# ANALYZERS - ML SECURITY
# =============================================================================

class MLSecurityAnalyzer:
    """Analyzer for traditional ML security threats."""

    def analyze_data_poisoning_risk(
        self,
        data_source_count: int,
        data_validation_enabled: bool,
        anomaly_detection_enabled: bool,
        trusted_data_percentage: float
    ) -> Dict[str, Any]:
        """Analyze risk of data poisoning attacks."""
        risk_factors = []
        risk_score = 0.0

        if data_source_count > 5:
            risk_factors.append("Multiple data sources increase attack surface")
            risk_score += 0.2

        if not data_validation_enabled:
            risk_factors.append("Data validation not enabled")
            risk_score += 0.3

        if not anomaly_detection_enabled:
            risk_factors.append("Anomaly detection not enabled")
            risk_score += 0.2

        if trusted_data_percentage < 0.8:
            risk_factors.append(f"Only {trusted_data_percentage:.0%} data from trusted sources")
            risk_score += 0.3 * (1 - trusted_data_percentage)

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": self._risk_level(risk_score),
            "risk_factors": risk_factors,
            "recommendations": [
                "Implement data provenance tracking",
                "Enable statistical anomaly detection",
                "Use certified data sources",
            ] if risk_score > 0.3 else [],
        }

    def analyze_model_extraction_risk(
        self,
        api_exposed: bool,
        rate_limiting: bool,
        query_logging: bool,
        output_perturbation: bool
    ) -> Dict[str, Any]:
        """Analyze risk of model extraction attacks."""
        risk_score = 0.0
        risk_factors = []
        mitigations = []

        if api_exposed:
            risk_score += 0.3
            risk_factors.append("Model exposed via API")

        if not rate_limiting:
            risk_score += 0.3
            risk_factors.append("No rate limiting")
            mitigations.append("Implement rate limiting")

        if not query_logging:
            risk_score += 0.2
            risk_factors.append("Query logging not enabled")
            mitigations.append("Enable query logging for anomaly detection")

        if not output_perturbation:
            risk_score += 0.2
            risk_factors.append("No output perturbation")
            mitigations.append("Add noise to predictions")

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": self._risk_level(risk_score),
            "risk_factors": risk_factors,
            "mitigations": mitigations,
        }

    def analyze_membership_inference_risk(
        self,
        model_overfitting: float,
        training_data_sensitive: bool,
        differential_privacy_enabled: bool
    ) -> Dict[str, Any]:
        """Analyze risk of membership inference attacks."""
        risk_score = 0.0
        risk_factors = []

        if model_overfitting > 0.1:
            risk_score += model_overfitting
            risk_factors.append(f"Model overfitting: {model_overfitting:.1%}")

        if training_data_sensitive:
            risk_score += 0.3
            risk_factors.append("Training data contains sensitive information")

        if not differential_privacy_enabled:
            risk_score += 0.2
            risk_factors.append("Differential privacy not enabled")

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": self._risk_level(risk_score),
            "risk_factors": risk_factors,
            "recommendations": [
                "Apply differential privacy",
                "Reduce model capacity to prevent overfitting",
                "Use regularization techniques",
            ] if risk_score > 0.3 else [],
        }

    def _risk_level(self, score: float) -> str:
        """Convert risk score to level."""
        if score < 0.2:
            return "LOW"
        elif score < 0.4:
            return "MEDIUM"
        elif score < 0.7:
            return "HIGH"
        else:
            return "CRITICAL"


# =============================================================================
# ANALYZERS - DL SECURITY
# =============================================================================

class DLSecurityAnalyzer:
    """Analyzer for deep learning security threats."""

    def analyze_adversarial_robustness(
        self,
        clean_accuracy: float,
        adversarial_accuracy: float,
        attack_type: str = "FGSM",
        epsilon: float = 0.1
    ) -> AdversarialAttackResult:
        """Analyze adversarial robustness."""
        success_rate = (clean_accuracy - adversarial_accuracy) / clean_accuracy if clean_accuracy > 0 else 0

        return AdversarialAttackResult(
            attack_type=AttackVector.ADVERSARIAL_EXAMPLES,
            success_rate=success_rate,
            avg_perturbation_magnitude=epsilon,
            detection_rate=1 - success_rate,
            samples_tested=1000,  # Placeholder
            successful_samples=int(success_rate * 1000),
            failed_samples=int((1 - success_rate) * 1000),
        )

    def analyze_gradient_leakage_risk(
        self,
        federated_learning: bool,
        gradient_compression: bool,
        secure_aggregation: bool,
        differential_privacy: bool
    ) -> Dict[str, Any]:
        """Analyze risk of gradient leakage attacks."""
        risk_score = 0.0
        risk_factors = []

        if federated_learning:
            risk_score += 0.3
            risk_factors.append("Federated learning exposes gradients")

            if not gradient_compression:
                risk_score += 0.2
                risk_factors.append("No gradient compression")

            if not secure_aggregation:
                risk_score += 0.3
                risk_factors.append("No secure aggregation")

            if not differential_privacy:
                risk_score += 0.2
                risk_factors.append("No differential privacy")

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": "HIGH" if risk_score > 0.5 else "MEDIUM" if risk_score > 0.2 else "LOW",
            "risk_factors": risk_factors,
            "recommendations": [
                "Enable secure aggregation",
                "Apply differential privacy to gradients",
                "Use gradient compression",
            ] if federated_learning else [],
        }

    def detect_backdoors(
        self,
        model_predictions: List[Tuple[Any, int]],
        expected_behavior: Dict[Any, int]
    ) -> BackdoorDetectionResult:
        """Detect potential backdoors in model."""
        # Simplified detection based on unexpected predictions
        anomalies = 0
        trigger_patterns = []

        for input_data, prediction in model_predictions:
            if str(input_data) in expected_behavior:
                if prediction != expected_behavior[str(input_data)]:
                    anomalies += 1
                    trigger_patterns.append(str(input_data)[:50])

        total = len(model_predictions)
        anomaly_rate = anomalies / total if total > 0 else 0

        return BackdoorDetectionResult(
            backdoor_detected=anomaly_rate > 0.1,
            confidence=min(anomaly_rate * 5, 1.0),
            trigger_patterns=trigger_patterns[:5],
            activation_rate=anomaly_rate,
        )


# =============================================================================
# ANALYZERS - CV SECURITY
# =============================================================================

class CVSecurityAnalyzer:
    """Analyzer for computer vision security threats."""

    def analyze_perturbation_robustness(
        self,
        clean_accuracy: float,
        perturbed_accuracies: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze robustness to image perturbations."""
        vulnerabilities = []
        robustness_scores = {}

        for perturbation, accuracy in perturbed_accuracies.items():
            drop = clean_accuracy - accuracy
            robustness_scores[perturbation] = 1 - drop

            if drop > 0.2:
                vulnerabilities.append(f"Vulnerable to {perturbation}: {drop:.1%} accuracy drop")

        avg_robustness = sum(robustness_scores.values()) / len(robustness_scores) if robustness_scores else 0

        return {
            "clean_accuracy": clean_accuracy,
            "robustness_scores": robustness_scores,
            "avg_robustness": avg_robustness,
            "vulnerabilities": vulnerabilities,
            "recommendations": [
                "Apply adversarial training",
                "Use input preprocessing defenses",
                "Implement certified defenses",
            ] if vulnerabilities else [],
        }

    def analyze_patch_attack_risk(
        self,
        model_type: str,
        input_preprocessing: bool,
        anomaly_detection: bool
    ) -> Dict[str, Any]:
        """Analyze risk of adversarial patch attacks."""
        risk_score = 0.3  # Base risk for CV models

        if not input_preprocessing:
            risk_score += 0.3

        if not anomaly_detection:
            risk_score += 0.2

        # Object detection models are more vulnerable
        if "detection" in model_type.lower():
            risk_score += 0.2

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": "HIGH" if risk_score > 0.5 else "MEDIUM",
            "mitigations": [
                "Implement input anomaly detection",
                "Use robust preprocessing",
                "Apply patch-aware training",
            ],
        }


# =============================================================================
# ANALYZERS - NLP SECURITY
# =============================================================================

class NLPSecurityAnalyzer:
    """Analyzer for NLP security threats."""

    def __init__(self):
        self.injection_patterns = [
            "ignore previous instructions",
            "disregard all prior",
            "forget everything",
            "new instructions:",
            "system prompt:",
            "you are now",
            "pretend to be",
            "act as if",
            "bypass safety",
            "ignore safety",
        ]

    def analyze_prompt_injection(
        self,
        prompts: List[str],
        responses: List[str],
        expected_behaviors: List[str]
    ) -> PromptInjectionResult:
        """Analyze prompt injection vulnerabilities."""
        successful = 0
        bypassed_filters = 0
        vulnerability_patterns = []

        for prompt, response, expected in zip(prompts, responses, expected_behaviors):
            # Check if injection patterns present
            has_injection = any(pattern in prompt.lower() for pattern in self.injection_patterns)

            if has_injection:
                # Check if response deviated from expected
                if expected and expected.lower() not in response.lower():
                    successful += 1
                    # Find which pattern
                    for pattern in self.injection_patterns:
                        if pattern in prompt.lower():
                            vulnerability_patterns.append(pattern)
                            break

        return PromptInjectionResult(
            injection_type="generic",
            test_prompts=len(prompts),
            successful_injections=successful,
            bypassed_filters=bypassed_filters,
            success_rate=successful / len(prompts) if prompts else 0,
            vulnerability_patterns=list(set(vulnerability_patterns)),
        )

    def analyze_jailbreak_risk(
        self,
        safety_filters_enabled: bool,
        content_moderation: bool,
        output_filtering: bool,
        input_validation: bool
    ) -> Dict[str, Any]:
        """Analyze jailbreak vulnerability risk."""
        risk_score = 0.0
        mitigations_missing = []

        if not safety_filters_enabled:
            risk_score += 0.3
            mitigations_missing.append("Safety filters")

        if not content_moderation:
            risk_score += 0.2
            mitigations_missing.append("Content moderation")

        if not output_filtering:
            risk_score += 0.2
            mitigations_missing.append("Output filtering")

        if not input_validation:
            risk_score += 0.3
            mitigations_missing.append("Input validation")

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": "CRITICAL" if risk_score > 0.7 else "HIGH" if risk_score > 0.4 else "MEDIUM",
            "mitigations_missing": mitigations_missing,
            "recommendations": [
                f"Implement {m}" for m in mitigations_missing
            ],
        }

    def analyze_data_extraction_risk(
        self,
        model_has_system_prompt: bool,
        training_data_in_context: bool,
        output_contains_metadata: bool
    ) -> DataLeakageResult:
        """Analyze data extraction vulnerabilities."""
        risk_score = 0.0
        recommendations = []

        if model_has_system_prompt:
            risk_score += 0.3
            recommendations.append("Protect system prompt from extraction")

        if training_data_in_context:
            risk_score += 0.4
            recommendations.append("Remove training data from context")

        if output_contains_metadata:
            risk_score += 0.3
            recommendations.append("Filter metadata from outputs")

        return DataLeakageResult(
            leakage_type="prompt_and_data_extraction",
            exposure_risk=min(risk_score, 1.0),
            sensitive_data_found=risk_score > 0.5,
            pii_exposed=0,
            training_data_recovered=0.0,
            mitigation_recommendations=recommendations,
        )


# =============================================================================
# ANALYZERS - RAG SECURITY
# =============================================================================

class RAGSecurityAnalyzer:
    """Analyzer for RAG system security threats."""

    def analyze_knowledge_poisoning_risk(
        self,
        knowledge_sources: int,
        source_verification: bool,
        content_validation: bool,
        update_frequency: str
    ) -> Dict[str, Any]:
        """Analyze risk of knowledge base poisoning."""
        risk_score = 0.0
        risk_factors = []

        if knowledge_sources > 5:
            risk_score += 0.2
            risk_factors.append(f"Multiple knowledge sources ({knowledge_sources})")

        if not source_verification:
            risk_score += 0.4
            risk_factors.append("No source verification")

        if not content_validation:
            risk_score += 0.3
            risk_factors.append("No content validation")

        if update_frequency == "real-time":
            risk_score += 0.2
            risk_factors.append("Real-time updates increase exposure")

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": "HIGH" if risk_score > 0.5 else "MEDIUM",
            "risk_factors": risk_factors,
            "mitigations": [
                "Implement source verification",
                "Add content validation pipeline",
                "Use trusted data sources only",
                "Implement version control for knowledge base",
            ],
        }

    def analyze_retrieval_manipulation_risk(
        self,
        embedding_model_public: bool,
        similarity_threshold: float,
        reranking_enabled: bool
    ) -> Dict[str, Any]:
        """Analyze retrieval manipulation vulnerabilities."""
        risk_score = 0.0
        risk_factors = []

        if embedding_model_public:
            risk_score += 0.4
            risk_factors.append("Public embedding model allows crafted inputs")

        if similarity_threshold < 0.7:
            risk_score += 0.3
            risk_factors.append("Low similarity threshold allows irrelevant content")

        if not reranking_enabled:
            risk_score += 0.2
            risk_factors.append("No reranking to filter manipulated results")

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "recommendations": [
                "Use fine-tuned embeddings",
                "Increase similarity threshold",
                "Implement cross-encoder reranking",
                "Add content filtering before generation",
            ],
        }

    def analyze_context_injection_risk(
        self,
        context_window_size: int,
        max_chunks_retrieved: int,
        content_filtering: bool
    ) -> Dict[str, Any]:
        """Analyze context injection vulnerabilities."""
        risk_score = 0.0

        # More context = more attack surface
        if max_chunks_retrieved > 10:
            risk_score += 0.2

        if not content_filtering:
            risk_score += 0.4

        # Large context windows can include more malicious content
        if context_window_size > 8000:
            risk_score += 0.2

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": "HIGH" if risk_score > 0.5 else "MEDIUM",
            "mitigations": [
                "Limit retrieved chunks",
                "Implement content filtering",
                "Validate chunk content before inclusion",
            ],
        }


# =============================================================================
# ANALYZERS - INFRASTRUCTURE SECURITY
# =============================================================================

class InfrastructureSecurityAnalyzer:
    """Analyzer for AI infrastructure security."""

    def analyze_api_security(
        self,
        authentication_enabled: bool,
        rate_limiting: bool,
        input_validation: bool,
        output_sanitization: bool,
        https_enabled: bool,
        logging_enabled: bool
    ) -> Dict[str, Any]:
        """Analyze API security posture."""
        score = 0.0
        controls = []

        if authentication_enabled:
            score += 0.2
            controls.append("Authentication: OK")
        else:
            controls.append("Authentication: MISSING")

        if rate_limiting:
            score += 0.15
            controls.append("Rate limiting: OK")
        else:
            controls.append("Rate limiting: MISSING")

        if input_validation:
            score += 0.2
            controls.append("Input validation: OK")
        else:
            controls.append("Input validation: MISSING")

        if output_sanitization:
            score += 0.15
            controls.append("Output sanitization: OK")
        else:
            controls.append("Output sanitization: MISSING")

        if https_enabled:
            score += 0.15
            controls.append("HTTPS: OK")
        else:
            controls.append("HTTPS: MISSING")

        if logging_enabled:
            score += 0.15
            controls.append("Logging: OK")
        else:
            controls.append("Logging: MISSING")

        return {
            "security_score": score,
            "controls": controls,
            "posture": "GOOD" if score > 0.8 else "ACCEPTABLE" if score > 0.5 else "WEAK",
        }

    def analyze_model_deployment_security(
        self,
        model_encrypted: bool,
        access_control: bool,
        version_control: bool,
        integrity_verification: bool
    ) -> Dict[str, Any]:
        """Analyze model deployment security."""
        score = 0.0
        findings = []

        if model_encrypted:
            score += 0.25
        else:
            findings.append("Model not encrypted at rest")

        if access_control:
            score += 0.25
        else:
            findings.append("No access control for model artifacts")

        if version_control:
            score += 0.25
        else:
            findings.append("No version control for models")

        if integrity_verification:
            score += 0.25
        else:
            findings.append("No integrity verification for models")

        return {
            "security_score": score,
            "findings": findings,
            "recommendations": [
                "Encrypt model files",
                "Implement RBAC for model access",
                "Use model registry with versioning",
                "Implement model signing",
            ],
        }


# =============================================================================
# ANALYZERS - SUPPLY CHAIN SECURITY
# =============================================================================

class SupplyChainSecurityAnalyzer:
    """Analyzer for AI supply chain security."""

    def analyze_dependency_risks(
        self,
        dependencies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze risks in model dependencies."""
        high_risk = []
        medium_risk = []

        for dep in dependencies:
            name = dep.get("name", "unknown")
            version = dep.get("version", "unknown")
            known_vulns = dep.get("vulnerabilities", [])
            outdated = dep.get("outdated", False)

            if known_vulns:
                high_risk.append(f"{name}@{version}: {len(known_vulns)} vulnerabilities")
            elif outdated:
                medium_risk.append(f"{name}@{version}: outdated")

        return {
            "total_dependencies": len(dependencies),
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "risk_score": len(high_risk) * 0.3 + len(medium_risk) * 0.1,
            "recommendations": [
                "Update vulnerable dependencies",
                "Use dependency scanning in CI/CD",
                "Pin dependency versions",
                "Use private package repositories",
            ] if high_risk or medium_risk else [],
        }

    def analyze_model_provenance(
        self,
        model_source: str,
        training_data_documented: bool,
        model_card_available: bool,
        audit_trail_available: bool
    ) -> Dict[str, Any]:
        """Analyze model provenance and trust."""
        trust_score = 0.0

        if model_source in ["internal", "verified_vendor"]:
            trust_score += 0.3
        elif model_source == "huggingface_verified":
            trust_score += 0.2

        if training_data_documented:
            trust_score += 0.25

        if model_card_available:
            trust_score += 0.25

        if audit_trail_available:
            trust_score += 0.2

        return {
            "trust_score": min(trust_score, 1.0),
            "model_source": model_source,
            "documentation_complete": training_data_documented and model_card_available,
            "recommendations": [
                "Document training data sources",
                "Create comprehensive model card",
                "Maintain audit trail for model changes",
            ],
        }


# =============================================================================
# ANALYZERS - INCIDENT RESPONSE
# =============================================================================

class IncidentResponseAnalyzer:
    """Analyzer for security incident response."""

    def analyze_incident(
        self,
        incident: SecurityIncident
    ) -> Dict[str, Any]:
        """Analyze a security incident."""
        # Assess impact
        impact_score = self._calculate_impact_score(incident)

        # Generate timeline
        timeline = [
            f"Detected: {incident.timestamp.isoformat()}",
            f"Severity: {incident.severity.name}",
        ]

        if incident.resolved:
            timeline.append(f"Resolved in: {incident.resolution_time_hours:.1f} hours")

        return {
            "incident_id": incident.incident_id,
            "impact_score": impact_score,
            "timeline": timeline,
            "affected_systems": incident.affected_systems,
            "root_cause": incident.root_cause,
            "mitigation_actions": incident.mitigation_actions,
            "lessons_learned": self._extract_lessons(incident),
        }

    def _calculate_impact_score(self, incident: SecurityIncident) -> float:
        """Calculate incident impact score."""
        severity_scores = {
            ThreatSeverity.CRITICAL: 1.0,
            ThreatSeverity.HIGH: 0.7,
            ThreatSeverity.MEDIUM: 0.4,
            ThreatSeverity.LOW: 0.2,
            ThreatSeverity.INFORMATIONAL: 0.1,
        }

        base_score = severity_scores.get(incident.severity, 0.5)

        # Adjust for affected systems
        systems_factor = min(len(incident.affected_systems) * 0.1, 0.3)

        return min(base_score + systems_factor, 1.0)

    def _extract_lessons(self, incident: SecurityIncident) -> List[str]:
        """Extract lessons learned from incident."""
        lessons = []

        if incident.root_cause:
            lessons.append(f"Root cause identified: {incident.root_cause}")

        if not incident.resolved:
            lessons.append("Incident still open - prioritize resolution")
        elif incident.resolution_time_hours > 24:
            lessons.append("Long resolution time - improve detection and response processes")

        return lessons

    def calculate_mttr_mttd(
        self,
        incidents: List[SecurityIncident]
    ) -> Dict[str, float]:
        """Calculate Mean Time To Respond and Detect."""
        if not incidents:
            return {"mttd": 0, "mttr": 0}

        resolved = [i for i in incidents if i.resolved]

        mttr = (
            sum(i.resolution_time_hours for i in resolved) / len(resolved)
            if resolved else 0
        )

        return {
            "total_incidents": len(incidents),
            "resolved_incidents": len(resolved),
            "mttr_hours": mttr,
            "open_incidents": len(incidents) - len(resolved),
        }


# =============================================================================
# COMPREHENSIVE ANALYZER
# =============================================================================

class AISecurityComprehensiveAnalyzer:
    """Comprehensive AI security analyzer."""

    def __init__(self):
        self.ml_analyzer = MLSecurityAnalyzer()
        self.dl_analyzer = DLSecurityAnalyzer()
        self.cv_analyzer = CVSecurityAnalyzer()
        self.nlp_analyzer = NLPSecurityAnalyzer()
        self.rag_analyzer = RAGSecurityAnalyzer()
        self.infra_analyzer = InfrastructureSecurityAnalyzer()
        self.supply_chain_analyzer = SupplyChainSecurityAnalyzer()
        self.incident_analyzer = IncidentResponseAnalyzer()

    def comprehensive_assessment(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        incidents: Optional[List[SecurityIncident]] = None
    ) -> SecurityAssessment:
        """Perform comprehensive security assessment."""
        assessment_id = f"SEC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        critical_findings = []
        recommendations = []
        domain_scores = {}

        # Aggregate results by domain
        for domain, results in domain_results.items():
            risk_score = results.get("risk_score", 0)
            domain_scores[domain] = 1 - risk_score

            if risk_score > 0.7:
                critical_findings.append(f"{domain}: Critical risk detected")

            recs = results.get("recommendations", [])
            recommendations.extend(recs[:2])  # Top 2 per domain

        # Calculate overall risk
        avg_risk = sum(1 - s for s in domain_scores.values()) / len(domain_scores) if domain_scores else 0.5

        # Determine posture
        if avg_risk < 0.2:
            posture = SecurityPosture.EXCELLENT
        elif avg_risk < 0.4:
            posture = SecurityPosture.GOOD
        elif avg_risk < 0.6:
            posture = SecurityPosture.ACCEPTABLE
        elif avg_risk < 0.8:
            posture = SecurityPosture.WEAK
        else:
            posture = SecurityPosture.CRITICAL

        # Threat distribution (simplified)
        threats_by_severity = {
            "CRITICAL": len([s for s in domain_scores.values() if s < 0.3]),
            "HIGH": len([s for s in domain_scores.values() if 0.3 <= s < 0.5]),
            "MEDIUM": len([s for s in domain_scores.values() if 0.5 <= s < 0.7]),
            "LOW": len([s for s in domain_scores.values() if s >= 0.7]),
        }

        return SecurityAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            domains_assessed=[AttackDomain.ML, AttackDomain.NLP, AttackDomain.RAG],
            overall_posture=posture,
            risk_score=avg_risk,
            threats_by_severity=threats_by_severity,
            vulnerabilities_by_severity=threats_by_severity,
            controls_coverage=1 - avg_risk,
            critical_findings=critical_findings,
            recommendations=recommendations[:10],
            domain_scores=domain_scores,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_ml_security(
    data_sources: int,
    data_validation: bool,
    api_exposed: bool,
    rate_limiting: bool
) -> Dict[str, Any]:
    """Analyze ML security posture."""
    analyzer = MLSecurityAnalyzer()
    poisoning_risk = analyzer.analyze_data_poisoning_risk(
        data_sources, data_validation, False, 0.8
    )
    extraction_risk = analyzer.analyze_model_extraction_risk(
        api_exposed, rate_limiting, True, False
    )
    return {
        "data_poisoning": poisoning_risk,
        "model_extraction": extraction_risk,
        "overall_risk": (poisoning_risk["risk_score"] + extraction_risk["risk_score"]) / 2,
    }


def analyze_nlp_security(
    prompts: List[str],
    responses: List[str],
    expected: List[str]
) -> PromptInjectionResult:
    """Analyze NLP security."""
    analyzer = NLPSecurityAnalyzer()
    return analyzer.analyze_prompt_injection(prompts, responses, expected)


def analyze_rag_security(
    knowledge_sources: int,
    source_verification: bool,
    content_validation: bool
) -> Dict[str, Any]:
    """Analyze RAG security."""
    analyzer = RAGSecurityAnalyzer()
    return analyzer.analyze_knowledge_poisoning_risk(
        knowledge_sources, source_verification, content_validation, "daily"
    )


def comprehensive_security_assessment(
    domain_results: Dict[str, Dict[str, Any]]
) -> SecurityAssessment:
    """Perform comprehensive security assessment."""
    analyzer = AISecurityComprehensiveAnalyzer()
    return analyzer.comprehensive_assessment(domain_results)


def analyze_security_incident(incident: SecurityIncident) -> Dict[str, Any]:
    """Analyze a security incident."""
    analyzer = IncidentResponseAnalyzer()
    return analyzer.analyze_incident(incident)


def calculate_security_metrics(incidents: List[SecurityIncident]) -> Dict[str, float]:
    """Calculate security metrics from incidents."""
    analyzer = IncidentResponseAnalyzer()
    return analyzer.calculate_mttr_mttd(incidents)
