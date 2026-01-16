"""
Lifecycle Governance Analysis Module - Pillar 2: Responsible AI
===============================================================

Comprehensive analysis framework for AI lifecycle governance covering all stages
from Design through Retirement following the 12-Pillar Trustworthy AI Framework.

Lifecycle Stages:
- Design: Requirements, architecture, risk assessment, ethics review
- Build: Development, training, validation, documentation
- Test: Functional, performance, fairness, security testing
- Deploy: Staging, canary, rollout, monitoring setup
- Run: Operations, monitoring, incident response, optimization
- Retire: Deprecation, migration, data handling, documentation

Key Components:
- LifecycleStageAnalyzer: Stage-specific analysis and compliance
- GateReviewAnalyzer: Stage gate review and approval analysis
- RiskClassificationAnalyzer: Risk-based lifecycle controls
- HumanOversightAnalyzer: Human oversight integration
- LifecycleGovernanceAnalyzer: Comprehensive lifecycle governance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod


# ============================================================================
# ENUMS - Lifecycle Classifications
# ============================================================================

class LifecycleStage(Enum):
    """AI system lifecycle stages"""
    DESIGN = "design"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    RUN = "run"
    RETIRE = "retire"


class StageStatus(Enum):
    """Status of a lifecycle stage"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class RiskClassification(Enum):
    """AI system risk classification levels"""
    MINIMAL = "minimal"  # Negligible risk
    LIMITED = "limited"  # Low risk with basic controls
    HIGH = "high"  # Significant risk, strict controls
    UNACCEPTABLE = "unacceptable"  # Prohibited use cases


class GateType(Enum):
    """Types of stage gates"""
    MANDATORY = "mandatory"  # Must pass to proceed
    ADVISORY = "advisory"  # Recommended but not blocking
    CONDITIONAL = "conditional"  # Based on risk level
    AUTOMATED = "automated"  # Automated checks only
    MANUAL = "manual"  # Requires human review


class OversightLevel(Enum):
    """Human oversight levels"""
    FULL_AUTONOMY = "full_autonomy"  # No human oversight
    HUMAN_ON_THE_LOOP = "human_on_the_loop"  # Monitoring, can intervene
    HUMAN_IN_THE_LOOP = "human_in_the_loop"  # Approval required
    HUMAN_IN_COMMAND = "human_in_command"  # Human makes decisions


class DocumentationType(Enum):
    """Types of lifecycle documentation"""
    REQUIREMENTS = "requirements"
    DESIGN_SPEC = "design_spec"
    RISK_ASSESSMENT = "risk_assessment"
    ETHICS_REVIEW = "ethics_review"
    TEST_PLAN = "test_plan"
    TEST_RESULTS = "test_results"
    DEPLOYMENT_PLAN = "deployment_plan"
    OPERATIONS_MANUAL = "operations_manual"
    INCIDENT_REPORT = "incident_report"
    RETIREMENT_PLAN = "retirement_plan"
    AUDIT_TRAIL = "audit_trail"


class ComplianceStatus(Enum):
    """Compliance status for lifecycle governance"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"


# ============================================================================
# DATA CLASSES - Lifecycle Metrics and Results
# ============================================================================

@dataclass
class StageRequirement:
    """Requirement for a lifecycle stage"""
    requirement_id: str
    stage: LifecycleStage
    description: str
    mandatory: bool
    risk_level_threshold: RiskClassification  # Applies at or above this level
    evidence_required: List[str]
    approval_required: bool
    automation_possible: bool


@dataclass
class StageGate:
    """Stage gate definition"""
    gate_id: str
    gate_name: str
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    gate_type: GateType
    requirements: List[StageRequirement]
    approvers: List[str]
    automation_checks: List[str]
    timeout_hours: int
    escalation_path: List[str]


@dataclass
class GateReviewResult:
    """Result of a gate review"""
    gate_id: str
    review_date: datetime
    reviewer: str
    status: StageStatus
    requirements_met: Dict[str, bool]
    evidence_verified: Dict[str, bool]
    comments: str
    conditions: List[str]  # Conditions for conditional approval
    blocking_issues: List[str]
    next_steps: List[str]


@dataclass
class StageMetrics:
    """Metrics for a lifecycle stage"""
    stage: LifecycleStage
    status: StageStatus
    start_date: Optional[datetime]
    completion_date: Optional[datetime]
    duration_days: Optional[float]
    requirements_total: int
    requirements_met: int
    compliance_score: float
    risk_items_identified: int
    risk_items_mitigated: int
    documentation_complete: bool
    gate_passed: bool


@dataclass
class DesignStageMetrics:
    """Specific metrics for Design stage"""
    requirements_documented: bool
    architecture_reviewed: bool
    risk_assessment_complete: bool
    ethics_review_complete: bool
    stakeholder_sign_off: bool
    data_requirements_defined: bool
    privacy_impact_assessed: bool
    fairness_considerations_documented: bool
    use_case_boundaries_defined: bool
    success_criteria_defined: bool


@dataclass
class BuildStageMetrics:
    """Specific metrics for Build stage"""
    code_review_complete: bool
    model_trained: bool
    validation_performed: bool
    bias_testing_complete: bool
    security_scan_complete: bool
    documentation_updated: bool
    version_controlled: bool
    reproducibility_verified: bool
    dependencies_audited: bool
    performance_benchmarked: bool


@dataclass
class TestStageMetrics:
    """Specific metrics for Test stage"""
    functional_tests_passed: bool
    performance_tests_passed: bool
    fairness_tests_passed: bool
    security_tests_passed: bool
    integration_tests_passed: bool
    user_acceptance_tests_passed: bool
    edge_case_tests_passed: bool
    adversarial_tests_passed: bool
    regression_tests_passed: bool
    test_coverage_percentage: float


@dataclass
class DeployStageMetrics:
    """Specific metrics for Deploy stage"""
    staging_validated: bool
    canary_successful: bool
    rollback_tested: bool
    monitoring_configured: bool
    alerting_configured: bool
    documentation_published: bool
    runbook_created: bool
    access_controls_configured: bool
    audit_logging_enabled: bool
    deployment_approval_obtained: bool


@dataclass
class RunStageMetrics:
    """Specific metrics for Run stage"""
    uptime_percentage: float
    incident_count: int
    mean_time_to_recovery: float
    drift_detected: bool
    performance_within_sla: bool
    compliance_maintained: bool
    user_feedback_positive_rate: float
    model_refresh_needed: bool
    security_incidents: int
    escalations_count: int


@dataclass
class RetireStageMetrics:
    """Specific metrics for Retire stage"""
    migration_plan_created: bool
    data_archived: bool
    users_notified: bool
    dependencies_resolved: bool
    documentation_archived: bool
    access_revoked: bool
    audit_trail_preserved: bool
    lessons_learned_documented: bool
    replacement_identified: bool
    retirement_approved: bool


@dataclass
class RiskAssessmentResult:
    """Result of risk classification"""
    assessment_id: str
    assessment_date: datetime
    risk_classification: RiskClassification
    risk_factors: List[Dict[str, Any]]
    impact_assessment: Dict[str, str]
    probability_assessment: Dict[str, str]
    mitigations_required: List[str]
    controls_required: List[str]
    oversight_level_required: OversightLevel
    review_frequency: str


@dataclass
class OversightConfiguration:
    """Human oversight configuration"""
    oversight_level: OversightLevel
    approval_workflows: List[str]
    escalation_triggers: List[str]
    monitoring_requirements: List[str]
    intervention_capabilities: List[str]
    audit_requirements: List[str]
    training_requirements: List[str]


@dataclass
class LifecycleAssessment:
    """Comprehensive lifecycle assessment"""
    assessment_id: str
    assessment_date: datetime
    system_id: str
    current_stage: LifecycleStage
    risk_classification: RiskClassification
    stage_metrics: Dict[LifecycleStage, StageMetrics]
    gate_results: List[GateReviewResult]
    oversight_config: OversightConfiguration
    compliance_status: ComplianceStatus
    documentation_status: Dict[DocumentationType, bool]
    recommendations: List[str]
    risks: List[str]


# ============================================================================
# ANALYZERS - Stage-Specific Analysis
# ============================================================================

class DesignStageAnalyzer:
    """Analyzes Design stage compliance and quality"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_history: List[DesignStageMetrics] = []

    def analyze_design_stage(
        self,
        design_artifacts: Dict[str, Any]
    ) -> DesignStageMetrics:
        """Analyze design stage compliance"""
        metrics = DesignStageMetrics(
            requirements_documented=self._check_requirements(design_artifacts),
            architecture_reviewed=design_artifacts.get('architecture_reviewed', False),
            risk_assessment_complete=self._check_risk_assessment(design_artifacts),
            ethics_review_complete=design_artifacts.get('ethics_review', False),
            stakeholder_sign_off=design_artifacts.get('stakeholder_approval', False),
            data_requirements_defined=self._check_data_requirements(design_artifacts),
            privacy_impact_assessed=design_artifacts.get('pia_complete', False),
            fairness_considerations_documented=design_artifacts.get('fairness_documented', False),
            use_case_boundaries_defined=self._check_boundaries(design_artifacts),
            success_criteria_defined=design_artifacts.get('success_criteria', False)
        )

        self.analysis_history.append(metrics)
        return metrics

    def _check_requirements(self, artifacts: Dict[str, Any]) -> bool:
        """Check if requirements are properly documented"""
        return (
            'requirements' in artifacts and
            len(artifacts.get('requirements', [])) > 0
        )

    def _check_risk_assessment(self, artifacts: Dict[str, Any]) -> bool:
        """Check if risk assessment is complete"""
        risk = artifacts.get('risk_assessment', {})
        return (
            'identified_risks' in risk and
            'mitigations' in risk and
            'classification' in risk
        )

    def _check_data_requirements(self, artifacts: Dict[str, Any]) -> bool:
        """Check if data requirements are defined"""
        data_req = artifacts.get('data_requirements', {})
        return (
            'data_sources' in data_req and
            'data_quality' in data_req and
            'data_privacy' in data_req
        )

    def _check_boundaries(self, artifacts: Dict[str, Any]) -> bool:
        """Check if use case boundaries are defined"""
        boundaries = artifacts.get('boundaries', {})
        return (
            'intended_use' in boundaries and
            'prohibited_use' in boundaries and
            'limitations' in boundaries
        )

    def get_design_recommendations(
        self,
        metrics: DesignStageMetrics
    ) -> List[str]:
        """Generate recommendations for design stage"""
        recommendations = []

        if not metrics.requirements_documented:
            recommendations.append("Document functional and non-functional requirements")

        if not metrics.risk_assessment_complete:
            recommendations.append("Complete risk assessment with mitigations")

        if not metrics.ethics_review_complete:
            recommendations.append("Conduct ethics review for AI use case")

        if not metrics.privacy_impact_assessed:
            recommendations.append("Perform Privacy Impact Assessment (PIA)")

        if not metrics.fairness_considerations_documented:
            recommendations.append("Document fairness considerations and protected attributes")

        if not metrics.use_case_boundaries_defined:
            recommendations.append("Define clear use case boundaries and limitations")

        return recommendations


class BuildStageAnalyzer:
    """Analyzes Build stage compliance and quality"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_history: List[BuildStageMetrics] = []

    def analyze_build_stage(
        self,
        build_artifacts: Dict[str, Any]
    ) -> BuildStageMetrics:
        """Analyze build stage compliance"""
        metrics = BuildStageMetrics(
            code_review_complete=build_artifacts.get('code_review_approved', False),
            model_trained=build_artifacts.get('model_trained', False),
            validation_performed=build_artifacts.get('validation_complete', False),
            bias_testing_complete=build_artifacts.get('bias_tested', False),
            security_scan_complete=build_artifacts.get('security_scanned', False),
            documentation_updated=build_artifacts.get('docs_updated', False),
            version_controlled=build_artifacts.get('version_controlled', False),
            reproducibility_verified=build_artifacts.get('reproducible', False),
            dependencies_audited=build_artifacts.get('deps_audited', False),
            performance_benchmarked=build_artifacts.get('benchmarked', False)
        )

        self.analysis_history.append(metrics)
        return metrics

    def get_build_recommendations(
        self,
        metrics: BuildStageMetrics
    ) -> List[str]:
        """Generate recommendations for build stage"""
        recommendations = []

        if not metrics.code_review_complete:
            recommendations.append("Complete code review with security focus")

        if not metrics.bias_testing_complete:
            recommendations.append("Perform bias testing across protected attributes")

        if not metrics.reproducibility_verified:
            recommendations.append("Verify model reproducibility with fixed seeds")

        if not metrics.dependencies_audited:
            recommendations.append("Audit dependencies for vulnerabilities")

        return recommendations


class TestStageAnalyzer:
    """Analyzes Test stage compliance and quality"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_history: List[TestStageMetrics] = []

    def analyze_test_stage(
        self,
        test_results: Dict[str, Any]
    ) -> TestStageMetrics:
        """Analyze test stage compliance"""
        metrics = TestStageMetrics(
            functional_tests_passed=test_results.get('functional_passed', False),
            performance_tests_passed=test_results.get('performance_passed', False),
            fairness_tests_passed=test_results.get('fairness_passed', False),
            security_tests_passed=test_results.get('security_passed', False),
            integration_tests_passed=test_results.get('integration_passed', False),
            user_acceptance_tests_passed=test_results.get('uat_passed', False),
            edge_case_tests_passed=test_results.get('edge_cases_passed', False),
            adversarial_tests_passed=test_results.get('adversarial_passed', False),
            regression_tests_passed=test_results.get('regression_passed', False),
            test_coverage_percentage=test_results.get('coverage', 0.0)
        )

        self.analysis_history.append(metrics)
        return metrics

    def get_test_recommendations(
        self,
        metrics: TestStageMetrics
    ) -> List[str]:
        """Generate recommendations for test stage"""
        recommendations = []

        if not metrics.fairness_tests_passed:
            recommendations.append("Address fairness test failures before deployment")

        if not metrics.adversarial_tests_passed:
            recommendations.append("Improve robustness against adversarial inputs")

        if metrics.test_coverage_percentage < 80:
            recommendations.append(
                f"Increase test coverage from {metrics.test_coverage_percentage:.0f}% to 80%+"
            )

        return recommendations


class DeployStageAnalyzer:
    """Analyzes Deploy stage compliance and quality"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_history: List[DeployStageMetrics] = []

    def analyze_deploy_stage(
        self,
        deploy_config: Dict[str, Any]
    ) -> DeployStageMetrics:
        """Analyze deployment stage compliance"""
        metrics = DeployStageMetrics(
            staging_validated=deploy_config.get('staging_validated', False),
            canary_successful=deploy_config.get('canary_success', False),
            rollback_tested=deploy_config.get('rollback_tested', False),
            monitoring_configured=deploy_config.get('monitoring_enabled', False),
            alerting_configured=deploy_config.get('alerting_enabled', False),
            documentation_published=deploy_config.get('docs_published', False),
            runbook_created=deploy_config.get('runbook_exists', False),
            access_controls_configured=deploy_config.get('access_configured', False),
            audit_logging_enabled=deploy_config.get('audit_logging', False),
            deployment_approval_obtained=deploy_config.get('approved', False)
        )

        self.analysis_history.append(metrics)
        return metrics


class RunStageAnalyzer:
    """Analyzes Run stage operations and compliance"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_history: List[RunStageMetrics] = []

    def analyze_run_stage(
        self,
        operational_data: Dict[str, Any]
    ) -> RunStageMetrics:
        """Analyze run stage operations"""
        metrics = RunStageMetrics(
            uptime_percentage=operational_data.get('uptime', 99.0),
            incident_count=operational_data.get('incidents', 0),
            mean_time_to_recovery=operational_data.get('mttr', 0.0),
            drift_detected=operational_data.get('drift_detected', False),
            performance_within_sla=operational_data.get('within_sla', True),
            compliance_maintained=operational_data.get('compliant', True),
            user_feedback_positive_rate=operational_data.get('positive_feedback', 0.8),
            model_refresh_needed=operational_data.get('refresh_needed', False),
            security_incidents=operational_data.get('security_incidents', 0),
            escalations_count=operational_data.get('escalations', 0)
        )

        self.analysis_history.append(metrics)
        return metrics


class RetireStageAnalyzer:
    """Analyzes Retire stage compliance"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_history: List[RetireStageMetrics] = []

    def analyze_retire_stage(
        self,
        retirement_data: Dict[str, Any]
    ) -> RetireStageMetrics:
        """Analyze retirement stage compliance"""
        metrics = RetireStageMetrics(
            migration_plan_created=retirement_data.get('migration_plan', False),
            data_archived=retirement_data.get('data_archived', False),
            users_notified=retirement_data.get('users_notified', False),
            dependencies_resolved=retirement_data.get('deps_resolved', False),
            documentation_archived=retirement_data.get('docs_archived', False),
            access_revoked=retirement_data.get('access_revoked', False),
            audit_trail_preserved=retirement_data.get('audit_preserved', False),
            lessons_learned_documented=retirement_data.get('lessons_documented', False),
            replacement_identified=retirement_data.get('replacement_ready', False),
            retirement_approved=retirement_data.get('approved', False)
        )

        self.analysis_history.append(metrics)
        return metrics


# ============================================================================
# ANALYZERS - Gate Review
# ============================================================================

class GateReviewAnalyzer:
    """Analyzes stage gate reviews and approvals"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.gates: Dict[str, StageGate] = self._initialize_gates()
        self.review_history: List[GateReviewResult] = []

    def _initialize_gates(self) -> Dict[str, StageGate]:
        """Initialize default stage gates"""
        return {
            'design_to_build': StageGate(
                gate_id='G1',
                gate_name='Design to Build Gate',
                from_stage=LifecycleStage.DESIGN,
                to_stage=LifecycleStage.BUILD,
                gate_type=GateType.MANDATORY,
                requirements=[
                    StageRequirement(
                        requirement_id='R1.1',
                        stage=LifecycleStage.DESIGN,
                        description='Requirements documented',
                        mandatory=True,
                        risk_level_threshold=RiskClassification.MINIMAL,
                        evidence_required=['requirements_doc'],
                        approval_required=True,
                        automation_possible=False
                    ),
                    StageRequirement(
                        requirement_id='R1.2',
                        stage=LifecycleStage.DESIGN,
                        description='Risk assessment complete',
                        mandatory=True,
                        risk_level_threshold=RiskClassification.LIMITED,
                        evidence_required=['risk_assessment'],
                        approval_required=True,
                        automation_possible=False
                    )
                ],
                approvers=['product_owner', 'tech_lead', 'ethics_officer'],
                automation_checks=['doc_completeness', 'risk_score'],
                timeout_hours=72,
                escalation_path=['director', 'vp']
            ),
            'build_to_test': StageGate(
                gate_id='G2',
                gate_name='Build to Test Gate',
                from_stage=LifecycleStage.BUILD,
                to_stage=LifecycleStage.TEST,
                gate_type=GateType.MANDATORY,
                requirements=[
                    StageRequirement(
                        requirement_id='R2.1',
                        stage=LifecycleStage.BUILD,
                        description='Code review approved',
                        mandatory=True,
                        risk_level_threshold=RiskClassification.MINIMAL,
                        evidence_required=['code_review_approval'],
                        approval_required=True,
                        automation_possible=True
                    )
                ],
                approvers=['tech_lead'],
                automation_checks=['code_quality', 'security_scan'],
                timeout_hours=48,
                escalation_path=['director']
            ),
            'test_to_deploy': StageGate(
                gate_id='G3',
                gate_name='Test to Deploy Gate',
                from_stage=LifecycleStage.TEST,
                to_stage=LifecycleStage.DEPLOY,
                gate_type=GateType.MANDATORY,
                requirements=[
                    StageRequirement(
                        requirement_id='R3.1',
                        stage=LifecycleStage.TEST,
                        description='All tests passed',
                        mandatory=True,
                        risk_level_threshold=RiskClassification.MINIMAL,
                        evidence_required=['test_report'],
                        approval_required=True,
                        automation_possible=True
                    )
                ],
                approvers=['qa_lead', 'product_owner'],
                automation_checks=['test_results', 'coverage'],
                timeout_hours=24,
                escalation_path=['director']
            ),
            'deploy_to_run': StageGate(
                gate_id='G4',
                gate_name='Deploy to Run Gate',
                from_stage=LifecycleStage.DEPLOY,
                to_stage=LifecycleStage.RUN,
                gate_type=GateType.MANDATORY,
                requirements=[
                    StageRequirement(
                        requirement_id='R4.1',
                        stage=LifecycleStage.DEPLOY,
                        description='Deployment validated',
                        mandatory=True,
                        risk_level_threshold=RiskClassification.MINIMAL,
                        evidence_required=['deployment_validation'],
                        approval_required=True,
                        automation_possible=True
                    )
                ],
                approvers=['ops_lead', 'product_owner'],
                automation_checks=['health_check', 'monitoring_active'],
                timeout_hours=12,
                escalation_path=['director']
            ),
            'run_to_retire': StageGate(
                gate_id='G5',
                gate_name='Run to Retire Gate',
                from_stage=LifecycleStage.RUN,
                to_stage=LifecycleStage.RETIRE,
                gate_type=GateType.MANDATORY,
                requirements=[
                    StageRequirement(
                        requirement_id='R5.1',
                        stage=LifecycleStage.RUN,
                        description='Retirement plan approved',
                        mandatory=True,
                        risk_level_threshold=RiskClassification.MINIMAL,
                        evidence_required=['retirement_plan'],
                        approval_required=True,
                        automation_possible=False
                    )
                ],
                approvers=['product_owner', 'director'],
                automation_checks=['data_backup_verified'],
                timeout_hours=168,
                escalation_path=['vp']
            )
        }

    def evaluate_gate(
        self,
        gate_id: str,
        evidence: Dict[str, Any],
        reviewer: str
    ) -> GateReviewResult:
        """Evaluate a stage gate"""
        gate = self.gates.get(gate_id)
        if not gate:
            return self._create_failed_review(gate_id, "Gate not found")

        requirements_met = {}
        evidence_verified = {}
        blocking_issues = []

        for req in gate.requirements:
            # Check if requirement is met
            req_met = self._check_requirement(req, evidence)
            requirements_met[req.requirement_id] = req_met

            # Verify evidence
            for ev in req.evidence_required:
                evidence_verified[ev] = ev in evidence

            if req.mandatory and not req_met:
                blocking_issues.append(f"Mandatory requirement not met: {req.description}")

        # Determine status
        if blocking_issues:
            status = StageStatus.BLOCKED
        elif all(requirements_met.values()):
            status = StageStatus.APPROVED
        else:
            status = StageStatus.PENDING_REVIEW

        result = GateReviewResult(
            gate_id=gate_id,
            review_date=datetime.now(),
            reviewer=reviewer,
            status=status,
            requirements_met=requirements_met,
            evidence_verified=evidence_verified,
            comments=f"Gate review by {reviewer}",
            conditions=[],
            blocking_issues=blocking_issues,
            next_steps=self._determine_next_steps(status, blocking_issues)
        )

        self.review_history.append(result)
        return result

    def _check_requirement(
        self,
        requirement: StageRequirement,
        evidence: Dict[str, Any]
    ) -> bool:
        """Check if a requirement is met"""
        for ev in requirement.evidence_required:
            if ev not in evidence:
                return False
        return True

    def _create_failed_review(
        self,
        gate_id: str,
        reason: str
    ) -> GateReviewResult:
        """Create a failed review result"""
        return GateReviewResult(
            gate_id=gate_id,
            review_date=datetime.now(),
            reviewer="system",
            status=StageStatus.BLOCKED,
            requirements_met={},
            evidence_verified={},
            comments=reason,
            conditions=[],
            blocking_issues=[reason],
            next_steps=["Resolve blocking issue"]
        )

    def _determine_next_steps(
        self,
        status: StageStatus,
        blocking_issues: List[str]
    ) -> List[str]:
        """Determine next steps based on review status"""
        if status == StageStatus.APPROVED:
            return ["Proceed to next stage"]
        elif status == StageStatus.BLOCKED:
            return [f"Resolve: {issue}" for issue in blocking_issues]
        else:
            return ["Complete pending requirements", "Request re-review"]

    def get_gate_summary(self) -> Dict[str, Any]:
        """Get summary of all gates"""
        return {
            'total_gates': len(self.gates),
            'gates': [
                {
                    'id': g.gate_id,
                    'name': g.gate_name,
                    'from': g.from_stage.value,
                    'to': g.to_stage.value,
                    'type': g.gate_type.value,
                    'requirements_count': len(g.requirements)
                }
                for g in self.gates.values()
            ],
            'review_count': len(self.review_history)
        }


# ============================================================================
# ANALYZERS - Risk Classification
# ============================================================================

class RiskClassificationAnalyzer:
    """Analyzes and classifies AI system risk levels"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.assessment_history: List[RiskAssessmentResult] = []

    def assess_risk(
        self,
        system_profile: Dict[str, Any]
    ) -> RiskAssessmentResult:
        """Assess risk classification for AI system"""
        risk_factors = self._identify_risk_factors(system_profile)
        impact = self._assess_impact(system_profile, risk_factors)
        probability = self._assess_probability(system_profile, risk_factors)

        classification = self._determine_classification(risk_factors, impact, probability)
        oversight_level = self._determine_oversight_level(classification)

        result = RiskAssessmentResult(
            assessment_id=f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            assessment_date=datetime.now(),
            risk_classification=classification,
            risk_factors=risk_factors,
            impact_assessment=impact,
            probability_assessment=probability,
            mitigations_required=self._determine_mitigations(classification, risk_factors),
            controls_required=self._determine_controls(classification),
            oversight_level_required=oversight_level,
            review_frequency=self._determine_review_frequency(classification)
        )

        self.assessment_history.append(result)
        return result

    def _identify_risk_factors(
        self,
        profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify risk factors from system profile"""
        factors = []

        # Data sensitivity
        if profile.get('uses_personal_data', False):
            factors.append({
                'factor': 'personal_data',
                'severity': 'high',
                'description': 'System processes personal data'
            })

        # Decision impact
        impact_level = profile.get('decision_impact', 'low')
        if impact_level in ['high', 'critical']:
            factors.append({
                'factor': 'high_impact_decisions',
                'severity': impact_level,
                'description': f'System makes {impact_level} impact decisions'
            })

        # Autonomy level
        if profile.get('autonomous_actions', False):
            factors.append({
                'factor': 'autonomous_operations',
                'severity': 'medium',
                'description': 'System performs autonomous actions'
            })

        # Vulnerable populations
        if profile.get('vulnerable_users', False):
            factors.append({
                'factor': 'vulnerable_populations',
                'severity': 'high',
                'description': 'System interacts with vulnerable populations'
            })

        return factors

    def _assess_impact(
        self,
        profile: Dict[str, Any],
        factors: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Assess potential impact"""
        return {
            'financial': profile.get('financial_impact', 'low'),
            'safety': profile.get('safety_impact', 'low'),
            'rights': profile.get('rights_impact', 'low'),
            'reputation': profile.get('reputation_impact', 'low'),
            'operational': profile.get('operational_impact', 'medium')
        }

    def _assess_probability(
        self,
        profile: Dict[str, Any],
        factors: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Assess probability of negative outcomes"""
        base_probability = len(factors) / 10
        return {
            'error_occurrence': 'medium' if base_probability > 0.3 else 'low',
            'misuse': profile.get('misuse_probability', 'low'),
            'failure': profile.get('failure_probability', 'low')
        }

    def _determine_classification(
        self,
        factors: List[Dict[str, Any]],
        impact: Dict[str, str],
        probability: Dict[str, str]
    ) -> RiskClassification:
        """Determine overall risk classification"""
        high_severity_count = sum(
            1 for f in factors if f.get('severity') in ['high', 'critical']
        )

        high_impact_count = sum(
            1 for v in impact.values() if v in ['high', 'critical']
        )

        if high_severity_count >= 2 or high_impact_count >= 2:
            return RiskClassification.HIGH
        elif high_severity_count >= 1 or high_impact_count >= 1:
            return RiskClassification.LIMITED
        else:
            return RiskClassification.MINIMAL

    def _determine_oversight_level(
        self,
        classification: RiskClassification
    ) -> OversightLevel:
        """Determine required oversight level"""
        mapping = {
            RiskClassification.UNACCEPTABLE: OversightLevel.HUMAN_IN_COMMAND,
            RiskClassification.HIGH: OversightLevel.HUMAN_IN_THE_LOOP,
            RiskClassification.LIMITED: OversightLevel.HUMAN_ON_THE_LOOP,
            RiskClassification.MINIMAL: OversightLevel.FULL_AUTONOMY
        }
        return mapping.get(classification, OversightLevel.HUMAN_ON_THE_LOOP)

    def _determine_mitigations(
        self,
        classification: RiskClassification,
        factors: List[Dict[str, Any]]
    ) -> List[str]:
        """Determine required mitigations"""
        mitigations = []

        if classification in [RiskClassification.HIGH, RiskClassification.UNACCEPTABLE]:
            mitigations.extend([
                "Implement human-in-the-loop for critical decisions",
                "Add explainability for all outputs",
                "Establish continuous monitoring"
            ])

        for factor in factors:
            if factor.get('factor') == 'personal_data':
                mitigations.append("Implement data minimization and privacy controls")
            if factor.get('factor') == 'vulnerable_populations':
                mitigations.append("Add additional safeguards for vulnerable users")

        return mitigations

    def _determine_controls(
        self,
        classification: RiskClassification
    ) -> List[str]:
        """Determine required controls"""
        controls = {
            RiskClassification.MINIMAL: ['basic_logging', 'periodic_review'],
            RiskClassification.LIMITED: ['enhanced_logging', 'bias_monitoring', 'quarterly_review'],
            RiskClassification.HIGH: ['comprehensive_audit', 'real_time_monitoring', 'monthly_review', 'incident_response'],
            RiskClassification.UNACCEPTABLE: ['prohibited_without_exception']
        }
        return controls.get(classification, [])

    def _determine_review_frequency(
        self,
        classification: RiskClassification
    ) -> str:
        """Determine required review frequency"""
        frequencies = {
            RiskClassification.MINIMAL: 'annually',
            RiskClassification.LIMITED: 'quarterly',
            RiskClassification.HIGH: 'monthly',
            RiskClassification.UNACCEPTABLE: 'prohibited'
        }
        return frequencies.get(classification, 'quarterly')


# ============================================================================
# ANALYZERS - Human Oversight
# ============================================================================

class HumanOversightAnalyzer:
    """Analyzes human oversight requirements and implementation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.oversight_configs: List[OversightConfiguration] = []

    def analyze_oversight_requirements(
        self,
        risk_classification: RiskClassification,
        system_profile: Dict[str, Any]
    ) -> OversightConfiguration:
        """Analyze and configure human oversight requirements"""
        oversight_level = self._determine_oversight_level(risk_classification, system_profile)

        config = OversightConfiguration(
            oversight_level=oversight_level,
            approval_workflows=self._define_approval_workflows(oversight_level),
            escalation_triggers=self._define_escalation_triggers(oversight_level),
            monitoring_requirements=self._define_monitoring_requirements(oversight_level),
            intervention_capabilities=self._define_intervention_capabilities(oversight_level),
            audit_requirements=self._define_audit_requirements(oversight_level),
            training_requirements=self._define_training_requirements(oversight_level)
        )

        self.oversight_configs.append(config)
        return config

    def _determine_oversight_level(
        self,
        classification: RiskClassification,
        profile: Dict[str, Any]
    ) -> OversightLevel:
        """Determine appropriate oversight level"""
        if classification == RiskClassification.UNACCEPTABLE:
            return OversightLevel.HUMAN_IN_COMMAND
        elif classification == RiskClassification.HIGH:
            return OversightLevel.HUMAN_IN_THE_LOOP
        elif classification == RiskClassification.LIMITED:
            return OversightLevel.HUMAN_ON_THE_LOOP
        else:
            # Even minimal risk may need oversight based on profile
            if profile.get('autonomous_actions', False):
                return OversightLevel.HUMAN_ON_THE_LOOP
            return OversightLevel.FULL_AUTONOMY

    def _define_approval_workflows(
        self,
        level: OversightLevel
    ) -> List[str]:
        """Define approval workflows for oversight level"""
        workflows = {
            OversightLevel.HUMAN_IN_COMMAND: [
                'all_decisions_require_approval',
                'multi_person_approval_for_critical',
                'audit_trail_for_all_approvals'
            ],
            OversightLevel.HUMAN_IN_THE_LOOP: [
                'high_impact_decisions_require_approval',
                'exception_handling_requires_approval',
                'periodic_batch_approval'
            ],
            OversightLevel.HUMAN_ON_THE_LOOP: [
                'anomaly_based_approval',
                'threshold_triggered_review',
                'sampling_based_review'
            ],
            OversightLevel.FULL_AUTONOMY: []
        }
        return workflows.get(level, [])

    def _define_escalation_triggers(
        self,
        level: OversightLevel
    ) -> List[str]:
        """Define escalation triggers"""
        triggers = {
            OversightLevel.HUMAN_IN_COMMAND: [
                'any_uncertainty',
                'any_error',
                'any_new_situation'
            ],
            OversightLevel.HUMAN_IN_THE_LOOP: [
                'confidence_below_threshold',
                'high_impact_decision',
                'policy_violation',
                'user_complaint'
            ],
            OversightLevel.HUMAN_ON_THE_LOOP: [
                'significant_anomaly',
                'performance_degradation',
                'drift_detected',
                'safety_concern'
            ],
            OversightLevel.FULL_AUTONOMY: [
                'critical_error',
                'safety_violation'
            ]
        }
        return triggers.get(level, [])

    def _define_monitoring_requirements(
        self,
        level: OversightLevel
    ) -> List[str]:
        """Define monitoring requirements"""
        requirements = {
            OversightLevel.HUMAN_IN_COMMAND: [
                'real_time_dashboard',
                'decision_by_decision_logging',
                'immediate_alerting',
                'full_audit_trail'
            ],
            OversightLevel.HUMAN_IN_THE_LOOP: [
                'near_real_time_monitoring',
                'aggregate_metrics',
                'threshold_alerting',
                'daily_review_reports'
            ],
            OversightLevel.HUMAN_ON_THE_LOOP: [
                'periodic_monitoring',
                'trend_analysis',
                'weekly_review_reports',
                'anomaly_detection'
            ],
            OversightLevel.FULL_AUTONOMY: [
                'automated_monitoring',
                'exception_logging',
                'monthly_reports'
            ]
        }
        return requirements.get(level, [])

    def _define_intervention_capabilities(
        self,
        level: OversightLevel
    ) -> List[str]:
        """Define intervention capabilities"""
        capabilities = {
            OversightLevel.HUMAN_IN_COMMAND: [
                'approve_reject_all',
                'modify_output',
                'override_decision',
                'immediate_shutdown',
                'rollback_capability'
            ],
            OversightLevel.HUMAN_IN_THE_LOOP: [
                'approve_reject_flagged',
                'modify_parameters',
                'pause_operations',
                'escalate_to_management'
            ],
            OversightLevel.HUMAN_ON_THE_LOOP: [
                'pause_on_anomaly',
                'trigger_review',
                'adjust_thresholds',
                'request_retraining'
            ],
            OversightLevel.FULL_AUTONOMY: [
                'emergency_shutdown',
                'post_hoc_review'
            ]
        }
        return capabilities.get(level, [])

    def _define_audit_requirements(
        self,
        level: OversightLevel
    ) -> List[str]:
        """Define audit requirements"""
        requirements = {
            OversightLevel.HUMAN_IN_COMMAND: [
                'complete_audit_trail',
                'decision_rationale_logging',
                'human_approval_records',
                'continuous_audit'
            ],
            OversightLevel.HUMAN_IN_THE_LOOP: [
                'detailed_audit_trail',
                'intervention_logging',
                'monthly_audit_review'
            ],
            OversightLevel.HUMAN_ON_THE_LOOP: [
                'standard_audit_trail',
                'exception_logging',
                'quarterly_audit_review'
            ],
            OversightLevel.FULL_AUTONOMY: [
                'basic_audit_trail',
                'annual_audit_review'
            ]
        }
        return requirements.get(level, [])

    def _define_training_requirements(
        self,
        level: OversightLevel
    ) -> List[str]:
        """Define training requirements for human operators"""
        requirements = {
            OversightLevel.HUMAN_IN_COMMAND: [
                'comprehensive_system_training',
                'decision_making_training',
                'ethics_training',
                'regular_refresher_training',
                'simulation_exercises'
            ],
            OversightLevel.HUMAN_IN_THE_LOOP: [
                'system_operation_training',
                'review_process_training',
                'escalation_procedure_training'
            ],
            OversightLevel.HUMAN_ON_THE_LOOP: [
                'monitoring_dashboard_training',
                'alert_response_training'
            ],
            OversightLevel.FULL_AUTONOMY: [
                'basic_system_awareness'
            ]
        }
        return requirements.get(level, [])


# ============================================================================
# COMPREHENSIVE ANALYZER
# ============================================================================

class LifecycleGovernanceAnalyzer:
    """Comprehensive lifecycle governance analyzer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.design_analyzer = DesignStageAnalyzer(config)
        self.build_analyzer = BuildStageAnalyzer(config)
        self.test_analyzer = TestStageAnalyzer(config)
        self.deploy_analyzer = DeployStageAnalyzer(config)
        self.run_analyzer = RunStageAnalyzer(config)
        self.retire_analyzer = RetireStageAnalyzer(config)
        self.gate_analyzer = GateReviewAnalyzer(config)
        self.risk_analyzer = RiskClassificationAnalyzer(config)
        self.oversight_analyzer = HumanOversightAnalyzer(config)
        self.assessments: List[LifecycleAssessment] = []

    def analyze_lifecycle(
        self,
        system_id: str,
        current_stage: LifecycleStage,
        system_profile: Dict[str, Any],
        stage_artifacts: Dict[LifecycleStage, Dict[str, Any]]
    ) -> LifecycleAssessment:
        """Perform comprehensive lifecycle analysis"""
        # Assess risk first
        risk_result = self.risk_analyzer.assess_risk(system_profile)

        # Configure oversight
        oversight_config = self.oversight_analyzer.analyze_oversight_requirements(
            risk_result.risk_classification,
            system_profile
        )

        # Analyze each stage
        stage_metrics = {}

        for stage, artifacts in stage_artifacts.items():
            metrics = self._analyze_stage(stage, artifacts)
            stage_metrics[stage] = metrics

        # Get gate reviews
        gate_results = self._get_gate_results()

        # Determine compliance status
        compliance_status = self._determine_compliance_status(stage_metrics, gate_results)

        # Check documentation status
        doc_status = self._check_documentation_status(stage_artifacts)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_stage, stage_metrics, risk_result, gate_results
        )

        # Identify risks
        risks = self._identify_risks(stage_metrics, risk_result, compliance_status)

        assessment = LifecycleAssessment(
            assessment_id=f"lc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            assessment_date=datetime.now(),
            system_id=system_id,
            current_stage=current_stage,
            risk_classification=risk_result.risk_classification,
            stage_metrics=stage_metrics,
            gate_results=gate_results,
            oversight_config=oversight_config,
            compliance_status=compliance_status,
            documentation_status=doc_status,
            recommendations=recommendations,
            risks=risks
        )

        self.assessments.append(assessment)
        return assessment

    def _analyze_stage(
        self,
        stage: LifecycleStage,
        artifacts: Dict[str, Any]
    ) -> StageMetrics:
        """Analyze a specific lifecycle stage"""
        analyzers = {
            LifecycleStage.DESIGN: self.design_analyzer.analyze_design_stage,
            LifecycleStage.BUILD: self.build_analyzer.analyze_build_stage,
            LifecycleStage.TEST: self.test_analyzer.analyze_test_stage,
            LifecycleStage.DEPLOY: self.deploy_analyzer.analyze_deploy_stage,
            LifecycleStage.RUN: self.run_analyzer.analyze_run_stage,
            LifecycleStage.RETIRE: self.retire_analyzer.analyze_retire_stage
        }

        analyzer = analyzers.get(stage)
        if analyzer:
            specific_metrics = analyzer(artifacts)

            # Convert to generic StageMetrics
            requirements_met = sum(
                1 for v in vars(specific_metrics).values()
                if isinstance(v, bool) and v
            )
            requirements_total = sum(
                1 for v in vars(specific_metrics).values()
                if isinstance(v, bool)
            )

            return StageMetrics(
                stage=stage,
                status=StageStatus.IN_PROGRESS if requirements_met < requirements_total else StageStatus.COMPLETED,
                start_date=artifacts.get('start_date'),
                completion_date=artifacts.get('completion_date'),
                duration_days=artifacts.get('duration'),
                requirements_total=requirements_total,
                requirements_met=requirements_met,
                compliance_score=requirements_met / requirements_total if requirements_total > 0 else 0,
                risk_items_identified=artifacts.get('risks_identified', 0),
                risk_items_mitigated=artifacts.get('risks_mitigated', 0),
                documentation_complete=artifacts.get('docs_complete', False),
                gate_passed=artifacts.get('gate_passed', False)
            )

        return StageMetrics(
            stage=stage,
            status=StageStatus.NOT_STARTED,
            start_date=None,
            completion_date=None,
            duration_days=None,
            requirements_total=0,
            requirements_met=0,
            compliance_score=0,
            risk_items_identified=0,
            risk_items_mitigated=0,
            documentation_complete=False,
            gate_passed=False
        )

    def _get_gate_results(self) -> List[GateReviewResult]:
        """Get all gate review results"""
        return self.gate_analyzer.review_history

    def _determine_compliance_status(
        self,
        stage_metrics: Dict[LifecycleStage, StageMetrics],
        gate_results: List[GateReviewResult]
    ) -> ComplianceStatus:
        """Determine overall compliance status"""
        if not stage_metrics:
            return ComplianceStatus.UNDER_REVIEW

        avg_compliance = sum(
            m.compliance_score for m in stage_metrics.values()
        ) / len(stage_metrics)

        blocked_gates = sum(
            1 for g in gate_results if g.status == StageStatus.BLOCKED
        )

        if blocked_gates > 0:
            return ComplianceStatus.NON_COMPLIANT
        elif avg_compliance >= 0.9:
            return ComplianceStatus.COMPLIANT
        elif avg_compliance >= 0.7:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _check_documentation_status(
        self,
        stage_artifacts: Dict[LifecycleStage, Dict[str, Any]]
    ) -> Dict[DocumentationType, bool]:
        """Check documentation status"""
        doc_status = {doc_type: False for doc_type in DocumentationType}

        # Map stages to expected documentation
        stage_docs = {
            LifecycleStage.DESIGN: [
                DocumentationType.REQUIREMENTS,
                DocumentationType.DESIGN_SPEC,
                DocumentationType.RISK_ASSESSMENT,
                DocumentationType.ETHICS_REVIEW
            ],
            LifecycleStage.TEST: [
                DocumentationType.TEST_PLAN,
                DocumentationType.TEST_RESULTS
            ],
            LifecycleStage.DEPLOY: [
                DocumentationType.DEPLOYMENT_PLAN,
                DocumentationType.OPERATIONS_MANUAL
            ],
            LifecycleStage.RETIRE: [
                DocumentationType.RETIREMENT_PLAN
            ]
        }

        for stage, artifacts in stage_artifacts.items():
            if artifacts.get('docs_complete', False):
                for doc_type in stage_docs.get(stage, []):
                    doc_status[doc_type] = True

        return doc_status

    def _generate_recommendations(
        self,
        current_stage: LifecycleStage,
        stage_metrics: Dict[LifecycleStage, StageMetrics],
        risk_result: RiskAssessmentResult,
        gate_results: List[GateReviewResult]
    ) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []

        # Stage-specific recommendations
        for stage, metrics in stage_metrics.items():
            if metrics.compliance_score < 0.8:
                recommendations.append(
                    f"Improve {stage.value} stage compliance (currently {metrics.compliance_score:.0%})"
                )

            if metrics.risk_items_identified > metrics.risk_items_mitigated:
                unmitigated = metrics.risk_items_identified - metrics.risk_items_mitigated
                recommendations.append(
                    f"Address {unmitigated} unmitigated risks in {stage.value} stage"
                )

        # Risk-based recommendations
        recommendations.extend(risk_result.mitigations_required)

        # Gate-based recommendations
        for gate in gate_results:
            if gate.blocking_issues:
                recommendations.extend(gate.next_steps)

        return list(set(recommendations))

    def _identify_risks(
        self,
        stage_metrics: Dict[LifecycleStage, StageMetrics],
        risk_result: RiskAssessmentResult,
        compliance_status: ComplianceStatus
    ) -> List[str]:
        """Identify lifecycle risks"""
        risks = []

        if compliance_status == ComplianceStatus.NON_COMPLIANT:
            risks.append("System is non-compliant with governance requirements")

        if risk_result.risk_classification in [RiskClassification.HIGH, RiskClassification.UNACCEPTABLE]:
            risks.append(f"System classified as {risk_result.risk_classification.value} risk")

        for stage, metrics in stage_metrics.items():
            if metrics.status == StageStatus.BLOCKED:
                risks.append(f"{stage.value} stage is blocked")

        return risks

    def generate_report(
        self,
        assessment: LifecycleAssessment
    ) -> Dict[str, Any]:
        """Generate comprehensive lifecycle report"""
        return {
            'assessment_id': assessment.assessment_id,
            'assessment_date': assessment.assessment_date.isoformat(),
            'system_id': assessment.system_id,
            'current_stage': assessment.current_stage.value,
            'risk_classification': assessment.risk_classification.value,
            'compliance_status': assessment.compliance_status.value,
            'oversight_level': assessment.oversight_config.oversight_level.value,
            'stage_summary': {
                stage.value: {
                    'status': metrics.status.value,
                    'compliance': metrics.compliance_score,
                    'requirements_met': f"{metrics.requirements_met}/{metrics.requirements_total}",
                    'gate_passed': metrics.gate_passed
                }
                for stage, metrics in assessment.stage_metrics.items()
            },
            'documentation_status': {
                doc_type.value: status
                for doc_type, status in assessment.documentation_status.items()
            },
            'recommendations': assessment.recommendations,
            'risks': assessment.risks
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'LifecycleStage',
    'StageStatus',
    'RiskClassification',
    'GateType',
    'OversightLevel',
    'DocumentationType',
    'ComplianceStatus',
    # Data Classes
    'StageRequirement',
    'StageGate',
    'GateReviewResult',
    'StageMetrics',
    'DesignStageMetrics',
    'BuildStageMetrics',
    'TestStageMetrics',
    'DeployStageMetrics',
    'RunStageMetrics',
    'RetireStageMetrics',
    'RiskAssessmentResult',
    'OversightConfiguration',
    'LifecycleAssessment',
    # Stage Analyzers
    'DesignStageAnalyzer',
    'BuildStageAnalyzer',
    'TestStageAnalyzer',
    'DeployStageAnalyzer',
    'RunStageAnalyzer',
    'RetireStageAnalyzer',
    # Gate Analyzer
    'GateReviewAnalyzer',
    # Risk Analyzer
    'RiskClassificationAnalyzer',
    # Oversight Analyzer
    'HumanOversightAnalyzer',
    # Comprehensive Analyzer
    'LifecycleGovernanceAnalyzer',
]
