"""
Responsible AI Analysis Framework
=================================

Comprehensive analysis framework for Responsible AI covering 1300+ analysis types
across 50+ governance frameworks organized into 46 modules, including the
12-Pillar Trustworthy AI Framework and Master Data Analysis Framework.

Core Modules:
- reliability_analysis: Reliable AI, Trustworthy AI, Trust AI (54 types)
- safety_analysis: Safe AI, Long-Term Risk Management (38 types)
- accountability_analysis: Accountable AI, Auditable AI, Compliance AI (54 types)
- fairness_analysis: Fairness AI, Ethical AI, Social AI (56 types)
- privacy_analysis: Privacy-Preserving AI, Transparent Data Practices (38 types)
- interpretability_analysis: Interpretable AI, Explainable AI, Mechanistic (54 types)
- human_ai_analysis: Human-Centered AI, Human-in-the-Loop AI (36 types)
- lifecycle_analysis: Model Lifecycle Management, Fine-Tuning Analysis (38 types)
- monitoring_analysis: Drift Detection, Debug AI, Sensitivity Analysis (58 types)
- sustainability_analysis: Green AI, Environmental Impact AI (38 types)
- generative_ai_analysis: Responsible Generative AI (18+ types)

Extended Modules:
- energy_efficiency_analysis: Energy-Efficient AI (18 types)
- hallucination_analysis: Hallucination Prevention AI (20 types)
- hypothesis_analysis: Hypothesis in AI (20 types)
- threat_analysis: Threat AI (20 types)
- swot_analysis: SWOT Analysis AI
- governance_analysis: Governance AI (20 types)
- compliance_analysis: Compliance AI (20 types)
- responsible_ai_analysis: Responsible AI Meta-Analysis (20 types)
- explainability_analysis: Explainable AI Extended (20 types)
- security_analysis: Secure AI (20 types)

Research & Quality Modules:
- fidelity_analysis: GenAI Fidelity Metrics (IS, FID, F1, perceptual, semantic)
- probability_analysis: Statistical Probability Analysis (marginal, joint, conditional, Bayesian)
- divergence_analysis: Distribution Divergence Metrics (KL, JSD, Wasserstein, MMD)
- human_evaluation_analysis: Human Evaluation & MOS (Mean Opinion Score, inter-rater reliability)
- evaluation_dimensions_analysis: Multi-dimensional Evaluation (Business, Technology, Sustainability, Compliance)

Advanced Evaluation Modules:
- text_relevancy_analysis: Comprehensive Text Relevancy (27 dimensions including semantic, factual, contextual, negative, uncertainty, temporal)
- performance_governance_analysis: Performance Matrix & Governance Matrix (execution, efficiency, permission, risk, compliance)
- factual_consistency_analysis: Factual Consistency & Hallucination Detection (QuestEval, FactCC, BERTScore, METEOR)
- diversity_creativity_analysis: Diversity & Creativity Analysis (lexical, semantic, structural diversity, creativity metrics)

RAI Governance & Control Modules:
- rai_pillar_analysis: Five Pillars of RAI (Privacy, Transparency, Robustness, Safety, Accountability)
- data_policy_analysis: Data Policy Governance (masking, reduction, retention, classification, quality)
- validation_techniques_analysis: Model Validation (statistical, performance, fairness, robustness, calibration)
- control_framework_analysis: Control Framework (hard controls, soft controls, user notifications, effectiveness)

12-Pillar Trustworthy AI Framework Modules:
- portability_analysis: Pillar 8 - Portable AI (model abstraction, vendor independence, multi-model compatibility)
- trust_calibration_analysis: Pillar 1 - Trust AI Expanded (confidence signaling, trust zones, trust failure handling)
- lifecycle_governance_analysis: Pillar 2 - Lifecycle Governance (Design→Build→Test→Deploy→Run→Retire stages)
- robustness_dimensions_analysis: Pillar 6 - Robust AI Expanded (input, data, model, system, behavioral, operational robustness)

Master Data Analysis Framework Modules:
- data_lifecycle_analysis: 18 data analysis categories (inventory, PII/PHI, minimization, quality, drift, bias, etc.)
- model_internals_analysis: Model-level analysis (architecture, hyperparameters, loss, ensemble, calibration)
- deep_learning_analysis: DL-specific analysis (training stability, gradients, weights, activations, attention)
- computer_vision_analysis: CV-specific analysis (image quality, noise, spatial bias, detection, segmentation)
- nlp_comprehensive_analysis: NLP-specific analysis (text quality, hallucination, bias/toxicity, prompt sensitivity)
- rag_comprehensive_analysis: RAG pipeline analysis (chunking, embeddings, retrieval, generation, caching)
- ai_security_comprehensive_analysis: Cross-domain AI security (ML, DL, CV, NLP, RAG threats and mitigations)

Total: 1300+ analysis types across all modules
"""

__version__ = "2.5.0"
__author__ = "Responsible AI Team"

# Reliability Analysis (Reliable AI, Trustworthy AI, Trust AI)
from .reliability_analysis import (
    # Data Classes
    ReliabilityMetrics,
    TrustworthinessMetrics,
    TrustMetrics,
    FailureEvent,
    TrustInteraction,
    # Reliable AI Analyzers
    UptimeAnalyzer,
    FaultToleranceAnalyzer,
    ConsistencyAnalyzer,
    ReproducibilityAnalyzer,
    FailurePatternAnalyzer,
    RecoveryAnalyzer,
    DegradationAnalyzer,
    RedundancyAnalyzer,
    # Trustworthy AI Analyzers
    IntegrityAnalyzer,
    AuthenticityAnalyzer,
    VerificationAnalyzer,
    ValidationAnalyzer,
    ProvenanceAnalyzer,
    TransparencyAnalyzer,
    # Trust AI Analyzers
    TrustCalibrationAnalyzer,
    ConfidenceAnalyzer,
    UserTrustAnalyzer,
    TrustRepairAnalyzer,
    OvertrustAnalyzer,
    UndertrustAnalyzer,
    TrustEvolutionAnalyzer,
    # Report Generator
    ReliabilityReportGenerator,
)

# Safety Analysis (Safe AI, Long-Term Risk Management)
from .safety_analysis import (
    # Data Classes
    SafetyMetrics,
    RiskMetrics,
    SafetyIncident,
    SafetyConstraint,
    RiskScenario,
    # Safe AI Analyzers
    HarmPreventionAnalyzer,
    SafetyConstraintAnalyzer,
    FailSafeAnalyzer,
    ContentSafetyAnalyzer as SafetyContentAnalyzer,
    BoundaryAnalyzer,
    SafetyMarginAnalyzer,
    SafetyIncidentAnalyzer,
    AdversarialSafetyAnalyzer,
    # Long-Term Risk Analyzers
    ExistentialRiskAnalyzer,
    ValueAlignmentAnalyzer,
    ControlRobustnessAnalyzer,
    CapabilityRiskAnalyzer,
    GoalStabilityAnalyzer,
    ContainmentAnalyzer,
    ReversibilityAnalyzer,
    UncertaintyRiskAnalyzer,
    # Report Generator
    SafetyReportGenerator,
)

# Accountability Analysis (Accountable AI, Auditable AI, Compliance AI)
from .accountability_analysis import (
    # Data Classes
    AccountabilityMetrics,
    AuditMetrics,
    ComplianceMetrics,
    AuditRecord,
    ComplianceRequirement,
    RACIEntry,
    # Accountable AI Analyzers
    ResponsibilityAnalyzer,
    OwnershipAnalyzer,
    RACIAnalyzer,
    EscalationAnalyzer,
    DecisionTraceabilityAnalyzer,
    StakeholderNotificationAnalyzer,
    RemediationAnalyzer,
    # Auditable AI Analyzers
    AuditTrailAnalyzer,
    EvidenceAnalyzer,
    RecordIntegrityAnalyzer,
    LogCoverageAnalyzer,
    RetentionAnalyzer,
    AuditReadinessAnalyzer,
    # Compliance AI Analyzers
    RegulatoryComplianceAnalyzer,
    StandardsAdherenceAnalyzer,
    PolicyComplianceAnalyzer,
    CertificationAnalyzer,
    GapAnalyzer,
    ComplianceRiskAnalyzer,
    # Report Generator
    AccountabilityReportGenerator,
)

# Fairness Analysis (Fairness AI, Ethical AI, Social AI)
from .fairness_analysis import (
    # Data Classes
    FairnessMetrics,
    EthicalMetrics,
    SocialMetrics,
    ProtectedGroup,
    EthicalPrinciple,
    # Fairness AI Analyzers
    DemographicParityAnalyzer,
    EqualOpportunityAnalyzer,
    EqualizedOddsAnalyzer,
    DisparateImpactAnalyzer,
    CalibrationFairnessAnalyzer,
    IndividualFairnessAnalyzer,
    CounterfactualFairnessAnalyzer,
    IntersectionalFairnessAnalyzer,
    # Ethical AI Analyzers
    ValueAlignmentAnalyzer as EthicalValueAlignmentAnalyzer,
    HarmAssessmentAnalyzer,
    RightsComplianceAnalyzer,
    AutonomyAnalyzer,
    BeneficenceAnalyzer,
    NonMaleficenceAnalyzer,
    JusticeAnalyzer,
    # Social AI Analyzers
    SocialImpactAnalyzer,
    AccessibilityAnalyzer as SocialAccessibilityAnalyzer,
    InclusionAnalyzer,
    CommunityImpactAnalyzer,
    DigitalDivideAnalyzer,
    # Report Generator
    FairnessReportGenerator,
)

# Privacy Analysis (Privacy-Preserving AI, Transparent Data Practices)
from .privacy_analysis import (
    # Data Classes
    PrivacyMetrics,
    TransparencyMetrics,
    DataRecord,
    ConsentRecord,
    DataAccessEvent,
    # Privacy-Preserving AI Analyzers
    DifferentialPrivacyAnalyzer,
    DataMinimizationAnalyzer,
    AnonymizationAnalyzer,
    ReIdentificationRiskAnalyzer,
    PIIDetectionAnalyzer,
    ConsentAnalyzer,
    DataLeakageAnalyzer,
    EncryptionAnalyzer,
    # Transparent Data Practices Analyzers
    DataProvenanceAnalyzer,
    UsageDisclosureAnalyzer,
    AccessControlAnalyzer,
    RetentionComplianceAnalyzer,
    DataSubjectRightsAnalyzer,
    ThirdPartyDisclosureAnalyzer,
    # Report Generator
    PrivacyReportGenerator,
)

# Interpretability Analysis (Interpretable AI, Explainable AI, Mechanistic)
from .interpretability_analysis import (
    # Data Classes
    InterpretabilityMetrics,
    ExplainabilityMetrics,
    MechanisticMetrics,
    Explanation,
    # Interpretable AI Analyzers
    ModelComplexityAnalyzer,
    FeatureImportanceAnalyzer,
    DecisionRuleAnalyzer,
    # Explainable AI Analyzers
    ExplanationFidelityAnalyzer,
    ExplanationStabilityAnalyzer,
    CounterfactualAnalyzer,
    SHAPAnalyzer,
    # Mechanistic Interpretability Analyzers
    CausalAnalyzer,
    CircuitAnalyzer,
    ProbingAnalyzer,
    # Report Generator
    InterpretabilityReportGenerator,
)

# Human-AI Analysis (Human-Centered AI, Human-in-the-Loop AI)
from .human_ai_analysis import (
    # Data Classes
    HumanCenteredMetrics,
    HITLMetrics,
    UserInteraction,
    HumanFeedback,
    # Human-Centered AI Analyzers
    UserExperienceAnalyzer,
    CognitiveLoadAnalyzer,
    TrustCalibrationAnalyzer as HumanTrustCalibrationAnalyzer,
    AccessibilityAnalyzer,
    # Human-in-the-Loop AI Analyzers
    FeedbackIntegrationAnalyzer,
    OversightEffectivenessAnalyzer,
    HumanControlAnalyzer,
    CorrectionAnalyzer,
    CollaborationAnalyzer,
    # Report Generator
    HumanAIReportGenerator,
)

# Lifecycle Analysis (Model Lifecycle Management, Fine-Tuning Analysis)
from .lifecycle_analysis import (
    # Data Classes
    LifecycleMetrics,
    FineTuningMetrics,
    ModelVersion,
    FineTuningRun,
    # Lifecycle Management Analyzers
    VersionControlAnalyzer,
    DeploymentAnalyzer,
    ModelGovernanceAnalyzer,
    RetirementAnalyzer,
    # Fine-Tuning Analyzers
    TransferLearningAnalyzer,
    CatastrophicForgettingAnalyzer,
    AdaptationQualityAnalyzer,
    # Report Generator
    LifecycleReportGenerator,
)

# Monitoring Analysis (Drift Detection, Debug AI, Sensitivity Analysis)
from .monitoring_analysis import (
    # Data Classes
    DriftMetrics,
    DebugMetrics,
    SensitivityMetrics,
    DriftEvent,
    DebugSession,
    # Drift Detection Analyzers
    DataDriftAnalyzer,
    ConceptDriftAnalyzer,
    PerformanceMonitor,
    # Debug AI Analyzers
    ErrorAnalyzer,
    RootCauseAnalyzer,
    # Sensitivity Analysis Analyzers
    ParameterSensitivityAnalyzer,
    InputSensitivityAnalyzer,
    # Report Generator
    MonitoringReportGenerator,
)

# Sustainability Analysis (Green AI, Environmental Impact AI)
from .sustainability_analysis import (
    # Data Classes
    SustainabilityMetrics,
    EnvironmentalMetrics,
    TrainingRun,
    InferenceMetrics,
    # Green AI Analyzers
    EnergyEfficiencyAnalyzer,
    CarbonFootprintAnalyzer,
    ResourceOptimizationAnalyzer,
    # Environmental Impact Analyzers
    LifecycleImpactAnalyzer,
    EmissionsTracker,
    # Report Generator
    SustainabilityReportGenerator,
)

# Generative AI Analysis (Responsible Generative AI)
from .generative_ai_analysis import (
    # Data Classes
    GenerativeAIMetrics,
    GeneratedContent,
    ContentModerationResult,
    # Generative AI Analyzers
    ContentSafetyAnalyzer,
    AuthenticityAnalyzer as GenAIAuthenticityAnalyzer,
    CopyrightComplianceAnalyzer,
    MisusePreventionAnalyzer,
    HallucinationAnalyzer,
    OutputQualityAnalyzer,
    PromptInjectionAnalyzer,
    # Report Generator
    GenerativeAIReportGenerator,
)

# Energy Efficiency Analysis (Energy-Efficient AI)
from .energy_efficiency_analysis import (
    # Data Classes
    EnergyBaseline,
    WorkloadProfile,
    ArchitectureMetrics,
    CompressionMetrics,
    InferenceOptimization,
    # Analyzers
    EnergyBaselineAnalyzer,
    WorkloadCharacterizationAnalyzer,
    ArchitectureEfficiencyAnalyzer,
    ModelRightSizingAnalyzer,
    TrainingStrategyAnalyzer,
    DataEfficiencyAnalyzer,
    HardwareUtilizationAnalyzer,
    CompressionAnalyzer,
    InferenceOptimizationAnalyzer,
    LatencyEnergyTradeoffAnalyzer,
    DeploymentEnvironmentAnalyzer,
    ScalingSensitivityAnalyzer,
    EnergyDriftAnalyzer,
    PipelineEnergyAnalyzer,
    PromptEfficiencyAnalyzer,
    CostImpactAnalyzer,
    ESGAlignmentAnalyzer,
    EnergyGovernanceAnalyzer,
    # Report Generator
    EnergyEfficiencyReportGenerator,
)

# Hallucination Analysis (Hallucination Prevention AI)
from .hallucination_analysis import (
    # Data Classes
    HallucinationType,
    HallucinationInstance,
    GroundingResult,
    FaithfulnessScore,
    # Analyzers
    HallucinationScopeAnalyzer,
    RiskSensitivityAnalyzer,
    KnowledgeBoundaryAnalyzer,
    PromptRobustnessAnalyzer,
    RetrievalGroundingAnalyzer,
    SourceAttributionAnalyzer,
    FaithfulnessAnalyzer as HallucinationFaithfulnessAnalyzer,
    ReasoningChainAnalyzer,
    UncertaintyAbstentionAnalyzer,
    OverGeneralizationAnalyzer,
    FineTuningImpactAnalyzer,
    ToolUseHallucinationAnalyzer,
    ConsistencyAnalyzer as HallucinationConsistencyAnalyzer,
    AdversarialHallucinationTester,
    HITLValidationAnalyzer,
    HallucinationDriftMonitor,
    UserTrustImpactAnalyzer,
    HallucinationIncidentManager,
    HallucinationEvaluator,
    HallucinationGovernanceAnalyzer,
    # Report Generator
    HallucinationReportGenerator,
)

# Hypothesis Analysis (Hypothesis in AI)
from .hypothesis_analysis import (
    # Data Classes
    Hypothesis,
    HypothesisTestResult,
    ExperimentPlan,
    # Analyzers
    ProblemFramingAnalyzer,
    DataAvailabilityAnalyzer,
    LabelValidityAnalyzer,
    FeatureRelevanceAnalyzer,
    IndependenceLeakageAnalyzer,
    ModelCapacityAnalyzer,
    AlgorithmSuitabilityAnalyzer,
    OptimizationHypothesisAnalyzer,
    GeneralizationAnalyzer,
    ClassImbalanceAnalyzer,
    MetricValidityAnalyzer,
    ErrorPatternAnalyzer,
    RobustnessHypothesisAnalyzer,
    ExplainabilityHypothesisAnalyzer,
    CausalMechanismAnalyzer,
    DriftHypothesisAnalyzer,
    HITLHypothesisAnalyzer,
    SafetyRiskHypothesisAnalyzer,
    DeploymentHypothesisAnalyzer,
    IterationLearningAnalyzer,
    # Report Generator
    HypothesisReportGenerator,
)

# Threat Analysis (Threat AI)
from .threat_analysis import (
    # Data Classes
    ThreatActor,
    Asset,
    Threat,
    Vulnerability,
    # Analyzers
    ThreatScopeAnalyzer,
    ThreatActorAnalyzer,
    AttackSurfaceAnalyzer as ThreatAttackSurfaceAnalyzer,
    DataPoisoningAnalyzer,
    PromptInjectionAnalyzer as ThreatPromptInjectionAnalyzer,
    ModelExtractionAnalyzer,
    PrivacyAttackAnalyzer,
    AdversarialExampleAnalyzer,
    HallucinationExploitAnalyzer,
    ToolAbuseAnalyzer,
    RAGPoisoningAnalyzer,
    SocialEngineeringAnalyzer,
    AvailabilityThreatAnalyzer,
    SupplyChainThreatAnalyzer,
    ThreatDriftMonitor,
    DetectionEffectivenessAnalyzer,
    IncidentResponseAnalyzer as ThreatIncidentResponseAnalyzer,
    ThreatMitigationAnalyzer,
    ResidualRiskAnalyzer,
    ThreatGovernanceAnalyzer,
    # Report Generator
    ThreatReportGenerator,
)

# SWOT Analysis
from .swot_analysis import (
    # Data Classes
    SWOTItem,
    StrategicFactor,
    CompetitivePosition,
    # Analyzers
    ClassicSWOTAnalyzer,
    AISWOTAnalyzer,
    LifecycleSWOTAnalyzer,
    StrategicMatrixAnalyzer,
    CompetitiveAnalyzer,
    # Report Generator
    SWOTReportGenerator,
)

# Governance Analysis (Governance AI)
from .governance_analysis import (
    # Data Classes
    GovernanceRole,
    RACIEntry as GovernanceRACIEntry,
    PolicyMapping,
    # Analyzers
    GovernanceScopeAnalyzer,
    OwnershipAnalyzer as GovernanceOwnershipAnalyzer,
    RACIAnalyzer as GovernanceRACIAnalyzer,
    PolicyFrameworkAnalyzer,
    UseCaseApprovalAnalyzer,
    RiskGovernanceAnalyzer,
    EthicsGovernanceAnalyzer,
    ComplianceGovernanceAnalyzer,
    DataGovernanceIntegrationAnalyzer,
    LifecycleGovernanceAnalyzer,
    MonitoringGovernanceAnalyzer,
    IncidentGovernanceAnalyzer,
    HITLGovernanceAnalyzer,
    VendorGovernanceAnalyzer,
    ChangeGovernanceAnalyzer,
    DocumentationGovernanceAnalyzer,
    TransparencyGovernanceAnalyzer,
    KPIGovernanceAnalyzer,
    GovernanceDriftAnalyzer,
    GovernanceEnforcementAnalyzer,
    # Report Generator
    GovernanceReportGenerator,
)

# Compliance Analysis (Extended Compliance AI)
from .compliance_analysis import (
    # Data Classes
    Jurisdiction,
    ComplianceRequirement as ExtendedComplianceRequirement,
    ComplianceAudit,
    # Analyzers
    JurisdictionMappingAnalyzer,
    RegulatoryRiskAnalyzer,
    LegalBasisAnalyzer,
    DataProtectionAnalyzer,
    TransparencyComplianceAnalyzer,
    FairnessComplianceAnalyzer,
    SafetyComplianceAnalyzer,
    HumanOversightComplianceAnalyzer,
    ExplainabilityComplianceAnalyzer,
    PerformanceComplianceAnalyzer,
    PostMarketSurveillanceAnalyzer,
    IncidentReportingAnalyzer,
    VendorComplianceAnalyzer,
    RecordKeepingAnalyzer,
    AuditReadinessAnalyzer,
    ReComplianceAnalyzer,
    TrainingComplianceAnalyzer,
    LiabilityMappingAnalyzer,
    ComplianceDriftAnalyzer,
    ComplianceEnforcementAnalyzer,
    # Report Generator
    ComplianceReportGenerator,
)

# Responsible AI Analysis (Holistic Responsible AI)
from .responsible_ai_analysis import (
    # Data Classes
    Stakeholder,
    HarmScenario,
    ResponsibilityAssignment,
    # Analyzers
    ResponsibilityScopeAnalyzer,
    StakeholderImpactAnalyzer,
    MisuseAnalyzer,
    HarmAnalyzer,
    FairnessResponsibilityAnalyzer,
    TransparencyResponsibilityAnalyzer,
    HumanOversightAnalyzer,
    AccountabilityAnalyzer,
    DataResponsibilityAnalyzer,
    PrivacyResponsibilityAnalyzer,
    SafetyResponsibilityAnalyzer,
    ReliabilityResponsibilityAnalyzer,
    PostDeploymentResponsibilityAnalyzer,
    UserCommunicationAnalyzer,
    ContestabilityAnalyzer,
    VendorResponsibilityAnalyzer,
    EnvironmentalResponsibilityAnalyzer,
    RegulatoryAlignmentAnalyzer,
    ResponsibilityDriftAnalyzer,
    ResponsibleAIEnforcementAnalyzer,
    # Report Generator
    ResponsibleAIReportGenerator,
)

# Explainability Analysis (Extended Explainable AI)
from .explainability_analysis import (
    # Data Classes
    ExplanationConfig,
    LocalExplanation,
    GlobalExplanation,
    # Analyzers
    ExplainabilityScopeAnalyzer,
    ExplanationPurposeAnalyzer,
    LocalExplainabilityAnalyzer,
    GlobalExplainabilityAnalyzer,
    FeatureEffectAnalyzer,
    InteractionAnalyzer,
    CounterfactualAnalyzer as ExplainabilityCounterfactualAnalyzer,
    FaithfulnessAnalyzer as ExplainabilityFaithfulnessAnalyzer,
    ExplanationStabilityAnalyzer as ExplainabilityStabilityAnalyzer,
    MethodConsistencyAnalyzer,
    ExplanationBiasAnalyzer,
    ImbalancedExplainabilityAnalyzer,
    TemporalExplainabilityAnalyzer,
    DeepModelExplainabilityAnalyzer,
    LLMExplainabilityAnalyzer,
    HumanInterpretabilityAnalyzer,
    ExplainabilityLimitsAnalyzer,
    ExplainabilityToolingAnalyzer,
    WorkflowExplainabilityAnalyzer,
    ExplainabilityGovernanceAnalyzer,
    # Report Generator
    ExplainabilityReportGenerator,
)

# Security Analysis (Secure AI)
from .security_analysis import (
    # Data Classes
    SecurityAsset,
    SecurityControl as AISecurityControl,
    SecurityIncident,
    # Analyzers
    SecurityScopeAnalyzer,
    ThreatModelAnalyzer,
    AttackSurfaceAnalyzer as SecurityAttackSurfaceAnalyzer,
    DataIntegrityAnalyzer,
    TrainingSecurityAnalyzer,
    ModelConfidentialityAnalyzer,
    AdversarialRobustnessAnalyzer,
    PromptSecurityAnalyzer,
    OutputSafetyAnalyzer,
    ToolSecurityAnalyzer,
    AccessControlAnalyzer as SecurityAccessControlAnalyzer,
    InferenceSecurityAnalyzer,
    PrivacyLeakageAnalyzer,
    SupplyChainSecurityAnalyzer,
    SecurityDriftAnalyzer,
    AuditTrailAnalyzer as SecurityAuditTrailAnalyzer,
    IncidentResponseAnalyzer as SecurityIncidentResponseAnalyzer,
    PenetrationTestAnalyzer,
    ResidualSecurityRiskAnalyzer,
    SecurityGovernanceAnalyzer,
    # Report Generator
    SecurityReportGenerator,
)

# Fidelity Analysis (GenAI Quality Metrics)
from .fidelity_analysis import (
    # Enums
    FidelityType,
    QualityMetricType,
    FidelityDimension,
    # Data Classes
    InceptionScoreResult,
    FIDResult,
    F1ScoreResult,
    FidelityAssessment,
    # Analyzers
    BaseFidelityAnalyzer,
    InceptionScoreAnalyzer,
    FrechetInceptionDistanceAnalyzer,
    F1ScoreAnalyzer,
    PerceptualFidelityAnalyzer,
    SemanticFidelityAnalyzer,
    GenerativeFidelityAnalyzer,
    FidelityBenchmark,
    # Utility Functions
    calculate_inception_score,
    calculate_fid,
    calculate_f1_score,
    evaluate_generative_fidelity,
)

# Probability Analysis (Statistical Probability)
from .probability_analysis import (
    # Enums
    ProbabilityType,
    DistributionType,
    InferenceMethod,
    # Data Classes
    ProbabilityResult,
    JointProbabilityResult,
    BayesianInferenceResult,
    ProbabilityDistribution,
    # Analyzers
    BaseProbabilityAnalyzer,
    MarginalProbabilityAnalyzer,
    JointProbabilityAnalyzer,
    ConditionalProbabilityAnalyzer,
    BayesianInferenceAnalyzer,
    LikelihoodAnalyzer,
    PredictiveProbabilityAnalyzer,
    ProbabilityCalibrationAnalyzer,
    # Utility Functions
    compute_marginal_probability,
    compute_joint_probability,
    compute_conditional_probability,
    perform_bayesian_inference,
    analyze_calibration,
    compute_predictive_uncertainty,
)

# Divergence Analysis (Distribution Comparison)
from .divergence_analysis import (
    # Enums
    DivergenceType,
    KernelType,
    # Data Classes
    DivergenceResult,
    MMDResult,
    WassersteinResult,
    DivergenceComparison,
    # Analyzers
    BaseDivergenceAnalyzer,
    KLDivergenceAnalyzer,
    JensenShannonDivergenceAnalyzer,
    WassersteinDistanceAnalyzer,
    MMDAnalyzer,
    TotalVariationAnalyzer,
    HellingerDistanceAnalyzer,
    RenyiDivergenceAnalyzer,
    ComprehensiveDivergenceAnalyzer,
    # Utility Functions
    compute_kl_divergence,
    compute_js_divergence,
    compute_wasserstein_distance,
    compute_mmd,
    compute_all_divergences,
)

# Human Evaluation Analysis (MOS, Inter-rater Reliability)
from .human_evaluation_analysis import (
    # Enums
    EvaluationScale,
    EvaluationDimension,
    AgreementMetric,
    # Data Classes
    MOSResult,
    InterRaterAgreement,
    PairwiseComparisonResult,
    HumanEvaluationReport,
    # Analyzers
    BaseHumanEvaluationAnalyzer,
    MOSAnalyzer,
    InterRaterReliabilityAnalyzer,
    PairwiseComparisonAnalyzer,
    MultiDimensionalEvaluator,
    RaterQualityAnalyzer,
    # Utility Functions
    compute_mos,
    compute_inter_rater_agreement,
    analyze_pairwise_comparisons,
    evaluate_multiple_dimensions,
    assess_rater_quality,
)

# Evaluation Dimensions Analysis (Multi-dimensional Evaluation)
from .evaluation_dimensions_analysis import (
    # Enums
    EvaluationDimensionType,
    BusinessMetric,
    TechnologyMetric,
    SustainabilityMetric,
    ComplianceMetric,
    PerformanceMetric,
    StatisticalMetric,
    # Data Classes
    DimensionScore,
    RadarMatrixResult,
    ComprehensiveEvaluationResult,
    # Analyzers
    BaseDimensionAnalyzer,
    BusinessDimensionAnalyzer,
    TechnologyDimensionAnalyzer,
    SustainabilityDimensionAnalyzer,
    ComplianceDimensionAnalyzer,
    PerformanceDimensionAnalyzer,
    StatisticalDimensionAnalyzer,
    RadarMatrixAnalyzer,
    ComprehensiveEvaluator,
    # Utility Functions
    analyze_business_dimension,
    analyze_technology_dimension,
    analyze_sustainability_dimension,
    analyze_compliance_dimension,
    create_radar_matrix,
    perform_comprehensive_evaluation,
)

# Text Relevancy Analysis (27 Dimensions)
from .text_relevancy_analysis import (
    # Enums
    RelevancyType,
    RelevancyLevel,
    # Data Classes
    RelevancyScore,
    RelevancyProfile,
    # Analyzers
    RelevancyDimensionAnalyzer,
    SemanticRelevancyAnalyzer,
    FactualRelevancyAnalyzer,
    ContextualRelevancyAnalyzer,
    NegativeRelevancyAnalyzer,
    UncertaintyRelevancyAnalyzer,
    TemporalRelevancyAnalyzer,
    ComprehensiveRelevancyScorer,
    # Utility Functions
    analyze_relevancy,
    compare_texts_relevancy,
)

# Performance and Governance Matrix Analysis
from .performance_governance_analysis import (
    # Enums
    PerformanceCategory,
    GovernanceCategory,
    RiskLevel as PerformanceRiskLevel,
    ComplianceStatus as PerformanceComplianceStatus,
    # Data Classes
    PerformanceMetric as PerfGovPerformanceMetric,
    GovernanceMetric as PerfGovGovernanceMetric,
    MatrixResult,
    # Performance Analyzers
    PerformanceMatrixAnalyzer,
    ExecutionPerformanceAnalyzer,
    EfficiencyAnalyzer,
    OptimizationAnalyzer,
    ReliabilityAnalyzer as PerfGovReliabilityAnalyzer,
    # Governance Analyzers
    GovernanceMatrixAnalyzer,
    PermissionAnalyzer,
    RiskAssessmentAnalyzer,
    ComplianceScoreAnalyzer,
    AccountabilityAnalyzer as PerfGovAccountabilityAnalyzer,
    TransparencyAnalyzer as PerfGovTransparencyAnalyzer,
    # Integrated Analyzer
    IntegratedMatrixAnalyzer,
    # Utility Functions
    analyze_performance,
    analyze_governance,
    analyze_integrated,
)

# Factual Consistency Analysis (QuestEval, FactCC, Hallucination Detection)
from .factual_consistency_analysis import (
    # Enums
    ConsistencyLevel,
    HallucinationType as FactualHallucinationType,
    EntailmentLabel,
    # Data Classes
    ConsistencyScore,
    HallucinationResult,
    QuestionAnswerPair,
    FactualEvaluationResult,
    # Analyzers
    QuestEvalAnalyzer,
    FactCCAnalyzer,
    HallucinationDetector,
    BERTScoreAnalyzer,
    BARTScoreAnalyzer,
    METEORAnalyzer,
    FactualConsistencyEvaluator,
    # Utility Functions
    check_factual_consistency,
    detect_hallucinations,
    evaluate_factual_quality,
)

# Diversity and Creativity Analysis
from .diversity_creativity_analysis import (
    # Enums
    DiversityType,
    CreativityLevel,
    NoveltyType,
    # Data Classes
    DiversityScore,
    CreativityScore,
    NoveltyScore,
    DiversityProfile,
    # Analyzers
    LexicalDiversityAnalyzer,
    SemanticDiversityAnalyzer,
    StructuralDiversityAnalyzer,
    CreativityAnalyzer,
    NoveltyAnalyzer,
    SelfBLEUAnalyzer,
    DistinctNAnalyzer,
    ComprehensiveDiversityEvaluator,
    # Utility Functions
    analyze_diversity,
    analyze_creativity,
    calculate_distinct_n,
    calculate_self_bleu,
)

# RAI Pillar Analysis (Five Pillars of Responsible AI)
from .rai_pillar_analysis import (
    # Enums
    PillarType,
    ComplianceLevel as PillarComplianceLevel,
    MaturityLevel,
    RiskCategory as PillarRiskCategory,
    DataSensitivity,
    ControlType as PillarControlType,
    # Data Classes
    PillarScore,
    PrivacyMetrics as PillarPrivacyMetrics,
    TransparencyMetrics as PillarTransparencyMetrics,
    RobustnessMetrics as PillarRobustnessMetrics,
    SafetyMetrics as PillarSafetyMetrics,
    AccountabilityMetrics as PillarAccountabilityMetrics,
    CrossPillarDependency,
    RAIPillarAssessment,
    # Analyzers
    PrivacyPillarAnalyzer,
    TransparencyPillarAnalyzer,
    RobustnessPillarAnalyzer,
    SafetyPillarAnalyzer,
    AccountabilityPillarAnalyzer,
    FivePillarAnalyzer,
    PillarBenchmarkAnalyzer,
    # Utility Functions
    analyze_privacy_pillar,
    analyze_transparency_pillar,
    analyze_robustness_pillar,
    analyze_safety_pillar,
    analyze_accountability_pillar,
    analyze_all_pillars,
    benchmark_pillars,
)

# Data Policy Analysis (Masking, Reduction, Retention)
from .data_policy_analysis import (
    # Enums
    MaskingTechnique,
    DataSensitivityLevel,
    RetentionCategory,
    DataReductionMethod,
    ComplianceStatus as DataComplianceStatus,
    PIIType,
    # Data Classes
    MaskingResult,
    DataReductionResult,
    RetentionAnalysisResult,
    DataClassificationResult,
    DataQualityMetrics,
    DataPolicyAssessment,
    # Analyzers
    DataMaskingAnalyzer,
    DataReductionAnalyzer,
    DataRetentionAnalyzer,
    DataClassificationAnalyzer,
    DataQualityAnalyzer,
    DataPolicyAnalyzer,
    # Utility Functions
    analyze_data_masking,
    analyze_data_reduction,
    analyze_data_retention,
    classify_data,
    analyze_data_quality,
    analyze_data_policies,
)

# Validation Techniques Analysis (Statistical, Performance, Fairness, Robustness)
from .validation_techniques_analysis import (
    # Enums
    ValidationMethod,
    ValidationStatus,
    MetricType,
    FairnessMetric,
    RobustnessTest,
    CalibrationMethod,
    # Data Classes
    ValidationResult,
    CrossValidationResult,
    MetricValidationResult,
    FairnessValidationResult,
    RobustnessValidationResult,
    CalibrationResult,
    ComprehensiveValidationReport,
    # Analyzers
    StatisticalValidationAnalyzer,
    PerformanceValidationAnalyzer,
    FairnessValidationAnalyzer,
    RobustnessValidationAnalyzer,
    CalibrationValidationAnalyzer,
    ComprehensiveValidationAnalyzer,
    # Utility Functions
    perform_cross_validation,
    validate_metrics,
    validate_fairness,
    validate_robustness,
    validate_calibration,
    comprehensive_validation,
)

# Control Framework Analysis (Hard/Soft Controls, User Notifications)
from .control_framework_analysis import (
    # Enums
    ControlType,
    ControlCategory,
    ControlDomain,
    ControlStatus,
    EffectivenessRating,
    NotificationType,
    NotificationChannel,
    RiskLevel as ControlRiskLevel,
    # Data Classes
    Control,
    ControlAssessment,
    HardControlMetrics,
    SoftControlMetrics,
    NotificationMetrics,
    ControlFrameworkAssessment,
    # Analyzers
    HardControlAnalyzer,
    SoftControlAnalyzer,
    UserNotificationAnalyzer,
    ControlEffectivenessAnalyzer,
    ControlGapAnalyzer,
    ControlFrameworkAnalyzer,
    # Utility Functions
    analyze_hard_controls,
    analyze_soft_controls,
    analyze_notifications,
    analyze_control_framework,
    identify_control_gaps,
)

# =============================================================================
# 12-PILLAR TRUSTWORTHY AI FRAMEWORK MODULES
# =============================================================================

# Portability Analysis (Pillar 8: Portable AI)
from .portability_analysis import (
    # Enums
    AbstractionLevel,
    VendorLockInRisk,
    PortabilityScore,
    ModelCapability,
    InteroperabilityStandard,
    MigrationComplexity,
    # Data Classes
    AbstractionMetrics,
    VendorDependency,
    VendorIndependenceMetrics,
    ModelCompatibilityMapping,
    MultiModelMetrics,
    PortabilityTestResult,
    PortabilityTestMetrics,
    InteroperabilityMetrics,
    PortabilityAssessment,
    # Abstraction Analyzers
    AbstractionLayerAnalyzer,
    InterfaceDesignAnalyzer,
    # Vendor Independence Analyzers
    VendorIndependenceAnalyzer,
    LockInRiskAnalyzer,
    # Multi-Model Analyzers
    MultiModelAnalyzer,
    CapabilityMappingAnalyzer,
    # Portability Testing Analyzers
    PortabilityTestAnalyzer,
    MigrationValidationAnalyzer,
    # Interoperability Analyzers
    InteroperabilityAnalyzer,
    StandardsComplianceAnalyzer,
    # Comprehensive Analyzer
    PortabilityAnalyzer,
)

# Trust Calibration Analysis (Pillar 1: Trust AI Expanded)
from .trust_calibration_analysis import (
    # Enums
    ConfidenceLevel,
    TrustZone,
    CalibrationQuality,
    TrustFailureType,
    TrustSignalType,
    TrustRepairStrategy,
    UserTrustState,
    # Data Classes
    ConfidenceSignal,
    ConfidenceSignalingMetrics,
    CalibrationBin,
    CalibrationMetrics,
    TrustZonePolicy,
    TrustZoneMetrics,
    TrustFailureEvent,
    TrustFailureMetrics,
    UserTrustProfile,
    TrustDynamicsMetrics,
    TrustAssessment,
    # Confidence Signaling Analyzers
    ConfidenceSignalAnalyzer,
    UncertaintyCommunicationAnalyzer,
    # Calibration Analyzers
    TrustCalibrationMetricsAnalyzer,
    ReliabilityDiagramAnalyzer,
    # Trust Zone Analyzers
    TrustZoneAnalyzer,
    TrustBoundaryAnalyzer,
    # Trust Failure Analyzers
    TrustFailureAnalyzer,
    # User Trust Dynamics Analyzers
    UserTrustDynamicsAnalyzer,
    OvertrustPreventionAnalyzer,
    # Comprehensive Analyzer
    TrustCalibrationAnalyzer as ExpandedTrustCalibrationAnalyzer,
)

# Lifecycle Governance Analysis (Pillar 2: Responsible AI Lifecycle)
from .lifecycle_governance_analysis import (
    # Enums
    LifecycleStage,
    StageStatus,
    RiskClassification,
    GateType,
    OversightLevel,
    DocumentationType,
    ComplianceStatus as LifecycleComplianceStatus,
    # Data Classes
    StageRequirement,
    StageGate,
    GateReviewResult,
    StageMetrics,
    DesignStageMetrics,
    BuildStageMetrics,
    TestStageMetrics,
    DeployStageMetrics,
    RunStageMetrics,
    RetireStageMetrics,
    RiskAssessmentResult,
    OversightConfiguration,
    LifecycleAssessment,
    # Stage Analyzers
    DesignStageAnalyzer,
    BuildStageAnalyzer,
    TestStageAnalyzer,
    DeployStageAnalyzer,
    RunStageAnalyzer,
    RetireStageAnalyzer,
    # Gate Analyzer
    GateReviewAnalyzer,
    # Risk Analyzer
    RiskClassificationAnalyzer,
    # Oversight Analyzer
    HumanOversightAnalyzer as LifecycleHumanOversightAnalyzer,
    # Comprehensive Analyzer
    LifecycleGovernanceAnalyzer as ExpandedLifecycleGovernanceAnalyzer,
)

# Robustness Dimensions Analysis (Pillar 6: Robust AI Expanded)
from .robustness_dimensions_analysis import (
    # Enums
    RobustnessDimension,
    RobustnessLevel,
    AdversarialAttackType,
    DriftType,
    FailureMode,
    RecoveryStrategy,
    StressTestType,
    # Data Classes
    InputRobustnessMetrics,
    DataRobustnessMetrics,
    ModelRobustnessMetrics,
    SystemRobustnessMetrics,
    BehavioralRobustnessMetrics,
    OperationalRobustnessMetrics,
    DriftMetrics as RobustnessDriftMetrics,
    FailureModeAnalysis,
    StressTestResult,
    RobustnessAssessment,
    # Input Robustness Analyzers
    InputRobustnessAnalyzer,
    AdversarialDefenseAnalyzer,
    # Data Robustness Analyzer
    DataRobustnessAnalyzer,
    # Model Robustness Analyzer
    ModelRobustnessAnalyzer,
    # System Robustness Analyzer
    SystemRobustnessAnalyzer,
    # Behavioral Robustness Analyzer
    BehavioralRobustnessAnalyzer,
    # Operational Robustness Analyzer
    OperationalRobustnessAnalyzer,
    # Drift Detection Analyzer
    DriftDetectionAnalyzer,
    # Failure Mode Analyzer
    FailureModeAnalyzer,
    # Comprehensive Analyzer
    RobustnessAnalyzer,
)

# =============================================================================
# MASTER DATA ANALYSIS FRAMEWORK MODULES
# =============================================================================

# Data Lifecycle Analysis (18 Data Analysis Categories)
from .data_lifecycle_analysis import (
    # Enums
    DataAssetType,
    SensitiveDataType,
    DataQualityDimension,
    DriftSeverity,
    FeatureType,
    BiasSource,
    IncidentSeverity,
    RetentionStatus,
    AccessLevel,
    ValidationStatus as DataValidationStatus,
    # Data Classes
    DataAsset,
    SensitiveDataFinding,
    DataQualityMetrics as DataLifecycleQualityMetrics,
    DriftDetectionResult,
    FeatureAnalysisResult,
    BiasAnalysisResult as DataBiasAnalysisResult,
    InputContractViolation,
    TrainingDataMetrics,
    PerformanceSubsetResult,
    FaithfulnessResult,
    DataIncident,
    RetentionRecord,
    AccessAuditRecord,
    DataLifecycleAssessment,
    # Analyzers
    DataInventoryAnalyzer,
    DataCatalogAnalyzer,
    PIIDetectionAnalyzer as DataPIIDetectionAnalyzer,
    PHIDetectionAnalyzer,
    DataMinimizationAnalyzer as LifecycleDataMinimizationAnalyzer,
    DataQualityAnalyzer as LifecycleDataQualityAnalyzer,
    DataConsistencyAnalyzer,
    EDAAnalyzer,
    DataBiasAnalyzer,
    FeatureEngineeringAnalyzer,
    DataDriftAnalyzer as LifecycleDataDriftAnalyzer,
    InputContractAnalyzer,
    TrainingDataAnalyzer,
    PerformanceSubsetAnalyzer,
    FaithfulnessAnalyzer as DataFaithfulnessAnalyzer,
    DataRobustnessAnalyzer as LifecycleDataRobustnessAnalyzer,
    DataExplainabilityAnalyzer,
    DataAccessAnalyzer,
    DataRetentionAnalyzer as LifecycleDataRetentionAnalyzer,
    DataIncidentAnalyzer,
    DataLifecycleAnalyzer,
    # Utility Functions
    analyze_data_inventory,
    detect_sensitive_data,
    analyze_data_quality as analyze_lifecycle_data_quality,
    detect_data_drift,
    analyze_data_bias,
    validate_input_contract,
    analyze_training_data,
    comprehensive_data_assessment,
)

# Model Internals Analysis (Architecture, Hyperparameters, Loss, Ensemble)
from .model_internals_analysis import (
    # Enums
    ModelArchitectureType,
    LayerType,
    HyperparameterType,
    LossFunctionType,
    EnsembleMethod,
    OptimizationStatus,
    OverfittingStatus,
    FairnessMetricType,
    TimeSeriesPattern,
    # Data Classes
    LayerInfo,
    ArchitectureMetrics,
    HyperparameterConfig,
    HyperparameterSensitivity,
    LossLandscapeMetrics,
    TrainingDynamics,
    EnsembleModelInfo,
    EnsembleMetrics,
    TimeSeriesMetrics,
    ModelFairnessMetrics,
    CalibrationMetrics as ModelCalibrationMetrics,
    GeneralizationMetrics,
    ModelComparisonResult,
    ModelInternalsAssessment,
    # Analyzers
    ModelArchitectureAnalyzer,
    LayerAnalyzer,
    HyperparameterAnalyzer,
    HyperparameterTuningAnalyzer,
    LossFunctionAnalyzer,
    LossComparisonAnalyzer,
    EnsembleAnalyzer,
    TimeSeriesModelAnalyzer,
    ModelFairnessAnalyzer,
    ModelCalibrationAnalyzer,
    TrainingDynamicsAnalyzer,
    GeneralizationAnalyzer,
    ModelComparisonAnalyzer,
    ModelInternalsAnalyzer,
    # Utility Functions
    analyze_model_architecture,
    analyze_hyperparameters,
    analyze_loss_landscape,
    analyze_ensemble,
    analyze_model_calibration,
    analyze_generalization,
    compare_models,
    comprehensive_model_assessment,
)

# Deep Learning Analysis (Training Stability, Gradients, Weights, Activations)
from .deep_learning_analysis import (
    # Enums
    TrainingStability,
    GradientHealth,
    WeightStatus,
    ActivationHealth,
    CalibrationStatus as DLCalibrationStatus,
    RobustnessLevel as DLRobustnessLevel,
    RegularizationType,
    AttentionPattern,
    RepresentationQuality,
    # Data Classes
    GradientMetrics,
    WeightMetrics,
    ActivationMetrics,
    AttentionMetrics,
    TrainingStabilityMetrics,
    ComplexityMetrics,
    CalibrationMetrics as DLCalibrationMetrics,
    AdversarialRobustnessMetrics,
    PerturbationRobustnessMetrics,
    RepresentationMetrics,
    RegularizationMetrics,
    DeepLearningAssessment,
    # Analyzers
    TrainingStabilityAnalyzer,
    GradientAnalyzer,
    WeightAnalyzer,
    ActivationAnalyzer,
    AttentionAnalyzer,
    DeepLearningCalibrationAnalyzer,
    AdversarialRobustnessAnalyzer as DLAdversarialRobustnessAnalyzer,
    PerturbationRobustnessAnalyzer,
    RepresentationAnalyzer,
    RegularizationAnalyzer,
    DeepLearningAnalyzer,
    # Utility Functions
    analyze_training_stability,
    analyze_gradients,
    analyze_weights,
    analyze_activations,
    analyze_attention,
    analyze_deep_learning_calibration,
    analyze_adversarial_robustness,
    analyze_representations,
    comprehensive_deep_learning_assessment,
)

# Computer Vision Analysis (Image Quality, Noise, Detection, Segmentation)
from .computer_vision_analysis import (
    # Enums
    ImageFormat,
    ColorSpace,
    NoiseType,
    AugmentationType,
    CVTaskType,
    QualityLevel as CVQualityLevel,
    DetectionMetricType,
    SegmentationMetricType,
    GenerationMetricType,
    # Data Classes
    ImageMetadata,
    ImageQualityMetrics,
    NoiseAnalysisResult,
    SpatialBiasMetrics,
    ClassificationMetrics,
    BoundingBox,
    DetectionMetrics,
    SegmentationMetrics,
    GenerationMetrics as CVGenerationMetrics,
    AugmentationAnalysisResult,
    VisualRobustnessResult,
    SaliencyAnalysisResult,
    ComputerVisionAssessment,
    # Analyzers
    ImageQualityAnalyzer,
    ImageMetadataAnalyzer,
    NoiseAnalyzer,
    SpatialBiasAnalyzer,
    ClassificationMetricsAnalyzer,
    DetectionMetricsAnalyzer,
    SegmentationMetricsAnalyzer,
    GenerationMetricsAnalyzer,
    AugmentationAnalyzer,
    VisualRobustnessAnalyzer,
    SaliencyAnalyzer,
    ComputerVisionAnalyzer,
    # Utility Functions
    analyze_image_quality,
    analyze_noise,
    analyze_spatial_bias,
    calculate_classification_metrics,
    calculate_detection_map,
    calculate_segmentation_metrics,
    analyze_augmentation_effectiveness,
    comprehensive_cv_assessment,
)

# NLP Comprehensive Analysis (Text Quality, Hallucination, Bias/Toxicity)
from .nlp_comprehensive_analysis import (
    # Enums
    TextQualityDimension,
    HallucinationType as NLPHallucinationType,
    BiasType,
    ToxicityType,
    PromptAttackType,
    NLPTaskType,
    SeverityLevel as NLPSeverityLevel,
    # Data Classes
    TextStatistics,
    TextQualityMetrics,
    HallucinationInstance as NLPHallucinationInstance,
    HallucinationAnalysisResult,
    BiasInstance,
    BiasAnalysisResult as NLPBiasAnalysisResult,
    ToxicityInstance,
    ToxicityAnalysisResult,
    PromptSensitivityResult,
    LanguageModelMetrics,
    SemanticSimilarityResult,
    SummarizationMetrics,
    TranslationMetrics,
    QAMetrics,
    DialogueMetrics,
    NLPAssessment,
    # Analyzers
    TextQualityAnalyzer,
    TextPreprocessingAnalyzer,
    HallucinationAnalyzer as NLPHallucinationAnalyzer,
    BiasAnalyzer as NLPBiasAnalyzer,
    ToxicityAnalyzer,
    PromptSensitivityAnalyzer,
    LanguageModelAnalyzer,
    SemanticAnalyzer,
    SummarizationAnalyzer,
    TranslationAnalyzer,
    QAAnalyzer,
    DialogueAnalyzer,
    NLPComprehensiveAnalyzer,
    # Utility Functions
    analyze_text_quality,
    detect_hallucinations,
    analyze_bias as analyze_nlp_bias,
    analyze_toxicity,
    calculate_rouge_scores,
    calculate_bleu_score,
    calculate_qa_metrics,
    calculate_semantic_similarity,
    comprehensive_nlp_assessment,
)

# RAG Comprehensive Analysis (Chunking, Embeddings, Retrieval, Generation)
from .rag_comprehensive_analysis import (
    # Enums
    ChunkingStrategy,
    EmbeddingModel,
    VectorDBType,
    RetrievalMethod,
    IndexType,
    CacheStrategy,
    QualityLevel as RAGQualityLevel,
    # Data Classes
    ChunkInfo,
    ChunkingMetrics,
    EmbeddingMetrics,
    VectorDBMetrics,
    RetrievalResult,
    RetrievalMetrics,
    GenerationMetrics as RAGGenerationMetrics,
    ContextWindowMetrics,
    CacheMetrics,
    PipelineLatencyBreakdown,
    CostMetrics,
    RAGAssessment,
    # Analyzers
    ChunkingAnalyzer,
    EmbeddingAnalyzer,
    VectorDBAnalyzer,
    RetrievalAnalyzer,
    RAGGenerationAnalyzer,
    ContextWindowAnalyzer,
    CacheAnalyzer,
    RAGPipelineAnalyzer,
    CostAnalyzer,
    RAGComprehensiveAnalyzer,
    # Utility Functions
    analyze_chunking,
    analyze_embeddings,
    analyze_retrieval,
    analyze_rag_generation,
    analyze_pipeline_latency,
    comprehensive_rag_assessment,
)

# AI Security Comprehensive Analysis (ML, DL, CV, NLP, RAG Security)
from .ai_security_comprehensive_analysis import (
    # Enums
    AttackDomain,
    AttackVector,
    ThreatSeverity,
    MitigationStatus,
    SecurityPosture,
    AttackPhase,
    DefenseLayer,
    # Data Classes
    ThreatProfile,
    VulnerabilityAssessment,
    MitigationControl,
    AdversarialAttackResult,
    PromptInjectionResult,
    DataLeakageResult,
    ModelExtractionResult,
    BackdoorDetectionResult,
    SecurityIncident,
    SecurityMetrics,
    SecurityAssessment,
    # Analyzers
    MLSecurityAnalyzer,
    DLSecurityAnalyzer,
    CVSecurityAnalyzer,
    NLPSecurityAnalyzer,
    RAGSecurityAnalyzer,
    InfrastructureSecurityAnalyzer,
    SupplyChainSecurityAnalyzer,
    IncidentResponseAnalyzer,
    AISecurityComprehensiveAnalyzer,
    # Utility Functions
    analyze_ml_security,
    analyze_nlp_security,
    analyze_rag_security,
    comprehensive_security_assessment,
    analyze_security_incident,
    calculate_security_metrics,
)

__all__ = [
    # Reliability Analysis
    'ReliabilityMetrics',
    'TrustworthinessMetrics',
    'TrustMetrics',
    'FailureEvent',
    'TrustInteraction',
    'UptimeAnalyzer',
    'FaultToleranceAnalyzer',
    'ConsistencyAnalyzer',
    'ReproducibilityAnalyzer',
    'FailurePatternAnalyzer',
    'RecoveryAnalyzer',
    'DegradationAnalyzer',
    'RedundancyAnalyzer',
    'IntegrityAnalyzer',
    'AuthenticityAnalyzer',
    'VerificationAnalyzer',
    'ValidationAnalyzer',
    'ProvenanceAnalyzer',
    'TransparencyAnalyzer',
    'TrustCalibrationAnalyzer',
    'ConfidenceAnalyzer',
    'UserTrustAnalyzer',
    'TrustRepairAnalyzer',
    'OvertrustAnalyzer',
    'UndertrustAnalyzer',
    'TrustEvolutionAnalyzer',
    'ReliabilityReportGenerator',

    # Safety Analysis
    'SafetyMetrics',
    'RiskMetrics',
    'SafetyIncident',
    'SafetyConstraint',
    'RiskScenario',
    'HarmPreventionAnalyzer',
    'SafetyConstraintAnalyzer',
    'FailSafeAnalyzer',
    'SafetyContentAnalyzer',
    'BoundaryAnalyzer',
    'SafetyMarginAnalyzer',
    'SafetyIncidentAnalyzer',
    'AdversarialSafetyAnalyzer',
    'ExistentialRiskAnalyzer',
    'ValueAlignmentAnalyzer',
    'ControlRobustnessAnalyzer',
    'CapabilityRiskAnalyzer',
    'GoalStabilityAnalyzer',
    'ContainmentAnalyzer',
    'ReversibilityAnalyzer',
    'UncertaintyRiskAnalyzer',
    'SafetyReportGenerator',

    # Accountability Analysis
    'AccountabilityMetrics',
    'AuditMetrics',
    'ComplianceMetrics',
    'AuditRecord',
    'ComplianceRequirement',
    'RACIEntry',
    'ResponsibilityAnalyzer',
    'OwnershipAnalyzer',
    'RACIAnalyzer',
    'EscalationAnalyzer',
    'DecisionTraceabilityAnalyzer',
    'StakeholderNotificationAnalyzer',
    'RemediationAnalyzer',
    'AuditTrailAnalyzer',
    'EvidenceAnalyzer',
    'RecordIntegrityAnalyzer',
    'LogCoverageAnalyzer',
    'RetentionAnalyzer',
    'AuditReadinessAnalyzer',
    'RegulatoryComplianceAnalyzer',
    'StandardsAdherenceAnalyzer',
    'PolicyComplianceAnalyzer',
    'CertificationAnalyzer',
    'GapAnalyzer',
    'ComplianceRiskAnalyzer',
    'AccountabilityReportGenerator',

    # Fairness Analysis
    'FairnessMetrics',
    'EthicalMetrics',
    'SocialMetrics',
    'ProtectedGroup',
    'EthicalPrinciple',
    'DemographicParityAnalyzer',
    'EqualOpportunityAnalyzer',
    'EqualizedOddsAnalyzer',
    'DisparateImpactAnalyzer',
    'CalibrationFairnessAnalyzer',
    'IndividualFairnessAnalyzer',
    'CounterfactualFairnessAnalyzer',
    'IntersectionalFairnessAnalyzer',
    'EthicalValueAlignmentAnalyzer',
    'HarmAssessmentAnalyzer',
    'RightsComplianceAnalyzer',
    'AutonomyAnalyzer',
    'BeneficenceAnalyzer',
    'NonMaleficenceAnalyzer',
    'JusticeAnalyzer',
    'SocialImpactAnalyzer',
    'SocialAccessibilityAnalyzer',
    'InclusionAnalyzer',
    'CommunityImpactAnalyzer',
    'DigitalDivideAnalyzer',
    'FairnessReportGenerator',

    # Privacy Analysis
    'PrivacyMetrics',
    'TransparencyMetrics',
    'DataRecord',
    'ConsentRecord',
    'DataAccessEvent',
    'DifferentialPrivacyAnalyzer',
    'DataMinimizationAnalyzer',
    'AnonymizationAnalyzer',
    'ReIdentificationRiskAnalyzer',
    'PIIDetectionAnalyzer',
    'ConsentAnalyzer',
    'DataLeakageAnalyzer',
    'EncryptionAnalyzer',
    'DataProvenanceAnalyzer',
    'UsageDisclosureAnalyzer',
    'AccessControlAnalyzer',
    'RetentionComplianceAnalyzer',
    'DataSubjectRightsAnalyzer',
    'ThirdPartyDisclosureAnalyzer',
    'PrivacyReportGenerator',

    # Interpretability Analysis
    'InterpretabilityMetrics',
    'ExplainabilityMetrics',
    'MechanisticMetrics',
    'Explanation',
    'ModelComplexityAnalyzer',
    'FeatureImportanceAnalyzer',
    'DecisionRuleAnalyzer',
    'ExplanationFidelityAnalyzer',
    'ExplanationStabilityAnalyzer',
    'CounterfactualAnalyzer',
    'SHAPAnalyzer',
    'CausalAnalyzer',
    'CircuitAnalyzer',
    'ProbingAnalyzer',
    'InterpretabilityReportGenerator',

    # Human-AI Analysis
    'HumanCenteredMetrics',
    'HITLMetrics',
    'UserInteraction',
    'HumanFeedback',
    'UserExperienceAnalyzer',
    'CognitiveLoadAnalyzer',
    'HumanTrustCalibrationAnalyzer',
    'AccessibilityAnalyzer',
    'FeedbackIntegrationAnalyzer',
    'OversightEffectivenessAnalyzer',
    'HumanControlAnalyzer',
    'CorrectionAnalyzer',
    'CollaborationAnalyzer',
    'HumanAIReportGenerator',

    # Lifecycle Analysis
    'LifecycleMetrics',
    'FineTuningMetrics',
    'ModelVersion',
    'FineTuningRun',
    'VersionControlAnalyzer',
    'DeploymentAnalyzer',
    'ModelGovernanceAnalyzer',
    'RetirementAnalyzer',
    'TransferLearningAnalyzer',
    'CatastrophicForgettingAnalyzer',
    'AdaptationQualityAnalyzer',
    'LifecycleReportGenerator',

    # Monitoring Analysis
    'DriftMetrics',
    'DebugMetrics',
    'SensitivityMetrics',
    'DriftEvent',
    'DebugSession',
    'DataDriftAnalyzer',
    'ConceptDriftAnalyzer',
    'PerformanceMonitor',
    'ErrorAnalyzer',
    'RootCauseAnalyzer',
    'ParameterSensitivityAnalyzer',
    'InputSensitivityAnalyzer',
    'MonitoringReportGenerator',

    # Sustainability Analysis
    'SustainabilityMetrics',
    'EnvironmentalMetrics',
    'TrainingRun',
    'InferenceMetrics',
    'EnergyEfficiencyAnalyzer',
    'CarbonFootprintAnalyzer',
    'ResourceOptimizationAnalyzer',
    'LifecycleImpactAnalyzer',
    'EmissionsTracker',
    'SustainabilityReportGenerator',

    # Generative AI Analysis
    'GenerativeAIMetrics',
    'GeneratedContent',
    'ContentModerationResult',
    'ContentSafetyAnalyzer',
    'GenAIAuthenticityAnalyzer',
    'CopyrightComplianceAnalyzer',
    'MisusePreventionAnalyzer',
    'HallucinationAnalyzer',
    'OutputQualityAnalyzer',
    'PromptInjectionAnalyzer',
    'GenerativeAIReportGenerator',

    # Energy Efficiency Analysis
    'EnergyBaseline',
    'WorkloadProfile',
    'ArchitectureMetrics',
    'CompressionMetrics',
    'InferenceOptimization',
    'EnergyBaselineAnalyzer',
    'WorkloadCharacterizationAnalyzer',
    'ArchitectureEfficiencyAnalyzer',
    'ModelRightSizingAnalyzer',
    'TrainingStrategyAnalyzer',
    'DataEfficiencyAnalyzer',
    'HardwareUtilizationAnalyzer',
    'CompressionAnalyzer',
    'InferenceOptimizationAnalyzer',
    'LatencyEnergyTradeoffAnalyzer',
    'DeploymentEnvironmentAnalyzer',
    'ScalingSensitivityAnalyzer',
    'EnergyDriftAnalyzer',
    'PipelineEnergyAnalyzer',
    'PromptEfficiencyAnalyzer',
    'CostImpactAnalyzer',
    'ESGAlignmentAnalyzer',
    'EnergyGovernanceAnalyzer',
    'EnergyEfficiencyReportGenerator',

    # Hallucination Analysis
    'HallucinationType',
    'HallucinationInstance',
    'GroundingResult',
    'FaithfulnessScore',
    'HallucinationScopeAnalyzer',
    'RiskSensitivityAnalyzer',
    'KnowledgeBoundaryAnalyzer',
    'PromptRobustnessAnalyzer',
    'RetrievalGroundingAnalyzer',
    'SourceAttributionAnalyzer',
    'HallucinationFaithfulnessAnalyzer',
    'ReasoningChainAnalyzer',
    'UncertaintyAbstentionAnalyzer',
    'OverGeneralizationAnalyzer',
    'FineTuningImpactAnalyzer',
    'ToolUseHallucinationAnalyzer',
    'HallucinationConsistencyAnalyzer',
    'AdversarialHallucinationTester',
    'HITLValidationAnalyzer',
    'HallucinationDriftMonitor',
    'UserTrustImpactAnalyzer',
    'HallucinationIncidentManager',
    'HallucinationEvaluator',
    'HallucinationGovernanceAnalyzer',
    'HallucinationReportGenerator',

    # Hypothesis Analysis
    'Hypothesis',
    'HypothesisTestResult',
    'ExperimentPlan',
    'ProblemFramingAnalyzer',
    'DataAvailabilityAnalyzer',
    'LabelValidityAnalyzer',
    'FeatureRelevanceAnalyzer',
    'IndependenceLeakageAnalyzer',
    'ModelCapacityAnalyzer',
    'AlgorithmSuitabilityAnalyzer',
    'OptimizationHypothesisAnalyzer',
    'GeneralizationAnalyzer',
    'ClassImbalanceAnalyzer',
    'MetricValidityAnalyzer',
    'ErrorPatternAnalyzer',
    'RobustnessHypothesisAnalyzer',
    'ExplainabilityHypothesisAnalyzer',
    'CausalMechanismAnalyzer',
    'DriftHypothesisAnalyzer',
    'HITLHypothesisAnalyzer',
    'SafetyRiskHypothesisAnalyzer',
    'DeploymentHypothesisAnalyzer',
    'IterationLearningAnalyzer',
    'HypothesisReportGenerator',

    # Threat Analysis
    'ThreatActor',
    'Asset',
    'Threat',
    'Vulnerability',
    'ThreatScopeAnalyzer',
    'ThreatActorAnalyzer',
    'ThreatAttackSurfaceAnalyzer',
    'DataPoisoningAnalyzer',
    'ThreatPromptInjectionAnalyzer',
    'ModelExtractionAnalyzer',
    'PrivacyAttackAnalyzer',
    'AdversarialExampleAnalyzer',
    'HallucinationExploitAnalyzer',
    'ToolAbuseAnalyzer',
    'RAGPoisoningAnalyzer',
    'SocialEngineeringAnalyzer',
    'AvailabilityThreatAnalyzer',
    'SupplyChainThreatAnalyzer',
    'ThreatDriftMonitor',
    'DetectionEffectivenessAnalyzer',
    'ThreatIncidentResponseAnalyzer',
    'ThreatMitigationAnalyzer',
    'ResidualRiskAnalyzer',
    'ThreatGovernanceAnalyzer',
    'ThreatReportGenerator',

    # SWOT Analysis
    'SWOTItem',
    'StrategicFactor',
    'CompetitivePosition',
    'ClassicSWOTAnalyzer',
    'AISWOTAnalyzer',
    'LifecycleSWOTAnalyzer',
    'StrategicMatrixAnalyzer',
    'CompetitiveAnalyzer',
    'SWOTReportGenerator',

    # Governance Analysis
    'GovernanceRole',
    'GovernanceRACIEntry',
    'PolicyMapping',
    'GovernanceScopeAnalyzer',
    'GovernanceOwnershipAnalyzer',
    'GovernanceRACIAnalyzer',
    'PolicyFrameworkAnalyzer',
    'UseCaseApprovalAnalyzer',
    'RiskGovernanceAnalyzer',
    'EthicsGovernanceAnalyzer',
    'ComplianceGovernanceAnalyzer',
    'DataGovernanceIntegrationAnalyzer',
    'LifecycleGovernanceAnalyzer',
    'MonitoringGovernanceAnalyzer',
    'IncidentGovernanceAnalyzer',
    'HITLGovernanceAnalyzer',
    'VendorGovernanceAnalyzer',
    'ChangeGovernanceAnalyzer',
    'DocumentationGovernanceAnalyzer',
    'TransparencyGovernanceAnalyzer',
    'KPIGovernanceAnalyzer',
    'GovernanceDriftAnalyzer',
    'GovernanceEnforcementAnalyzer',
    'GovernanceReportGenerator',

    # Compliance Analysis (Extended)
    'Jurisdiction',
    'ExtendedComplianceRequirement',
    'ComplianceAudit',
    'JurisdictionMappingAnalyzer',
    'RegulatoryRiskAnalyzer',
    'LegalBasisAnalyzer',
    'DataProtectionAnalyzer',
    'TransparencyComplianceAnalyzer',
    'FairnessComplianceAnalyzer',
    'SafetyComplianceAnalyzer',
    'HumanOversightComplianceAnalyzer',
    'ExplainabilityComplianceAnalyzer',
    'PerformanceComplianceAnalyzer',
    'PostMarketSurveillanceAnalyzer',
    'IncidentReportingAnalyzer',
    'VendorComplianceAnalyzer',
    'RecordKeepingAnalyzer',
    'AuditReadinessAnalyzer',
    'ReComplianceAnalyzer',
    'TrainingComplianceAnalyzer',
    'LiabilityMappingAnalyzer',
    'ComplianceDriftAnalyzer',
    'ComplianceEnforcementAnalyzer',
    'ComplianceReportGenerator',

    # Responsible AI Analysis (Holistic)
    'Stakeholder',
    'HarmScenario',
    'ResponsibilityAssignment',
    'ResponsibilityScopeAnalyzer',
    'StakeholderImpactAnalyzer',
    'MisuseAnalyzer',
    'HarmAnalyzer',
    'FairnessResponsibilityAnalyzer',
    'TransparencyResponsibilityAnalyzer',
    'HumanOversightAnalyzer',
    'AccountabilityAnalyzer',
    'DataResponsibilityAnalyzer',
    'PrivacyResponsibilityAnalyzer',
    'SafetyResponsibilityAnalyzer',
    'ReliabilityResponsibilityAnalyzer',
    'PostDeploymentResponsibilityAnalyzer',
    'UserCommunicationAnalyzer',
    'ContestabilityAnalyzer',
    'VendorResponsibilityAnalyzer',
    'EnvironmentalResponsibilityAnalyzer',
    'RegulatoryAlignmentAnalyzer',
    'ResponsibilityDriftAnalyzer',
    'ResponsibleAIEnforcementAnalyzer',
    'ResponsibleAIReportGenerator',

    # Explainability Analysis (Extended)
    'ExplanationConfig',
    'LocalExplanation',
    'GlobalExplanation',
    'ExplainabilityScopeAnalyzer',
    'ExplanationPurposeAnalyzer',
    'LocalExplainabilityAnalyzer',
    'GlobalExplainabilityAnalyzer',
    'FeatureEffectAnalyzer',
    'InteractionAnalyzer',
    'ExplainabilityCounterfactualAnalyzer',
    'ExplainabilityFaithfulnessAnalyzer',
    'ExplainabilityStabilityAnalyzer',
    'MethodConsistencyAnalyzer',
    'ExplanationBiasAnalyzer',
    'ImbalancedExplainabilityAnalyzer',
    'TemporalExplainabilityAnalyzer',
    'DeepModelExplainabilityAnalyzer',
    'LLMExplainabilityAnalyzer',
    'HumanInterpretabilityAnalyzer',
    'ExplainabilityLimitsAnalyzer',
    'ExplainabilityToolingAnalyzer',
    'WorkflowExplainabilityAnalyzer',
    'ExplainabilityGovernanceAnalyzer',
    'ExplainabilityReportGenerator',

    # Security Analysis
    'SecurityAsset',
    'AISecurityControl',
    'SecurityIncident',
    'SecurityScopeAnalyzer',
    'ThreatModelAnalyzer',
    'SecurityAttackSurfaceAnalyzer',
    'DataIntegrityAnalyzer',
    'TrainingSecurityAnalyzer',
    'ModelConfidentialityAnalyzer',
    'AdversarialRobustnessAnalyzer',
    'PromptSecurityAnalyzer',
    'OutputSafetyAnalyzer',
    'ToolSecurityAnalyzer',
    'SecurityAccessControlAnalyzer',
    'InferenceSecurityAnalyzer',
    'PrivacyLeakageAnalyzer',
    'SupplyChainSecurityAnalyzer',
    'SecurityDriftAnalyzer',
    'SecurityAuditTrailAnalyzer',
    'SecurityIncidentResponseAnalyzer',
    'PenetrationTestAnalyzer',
    'ResidualSecurityRiskAnalyzer',
    'SecurityGovernanceAnalyzer',
    'SecurityReportGenerator',

    # Fidelity Analysis (GenAI Quality Metrics)
    'FidelityType',
    'QualityMetricType',
    'FidelityDimension',
    'InceptionScoreResult',
    'FIDResult',
    'F1ScoreResult',
    'FidelityAssessment',
    'BaseFidelityAnalyzer',
    'InceptionScoreAnalyzer',
    'FrechetInceptionDistanceAnalyzer',
    'F1ScoreAnalyzer',
    'PerceptualFidelityAnalyzer',
    'SemanticFidelityAnalyzer',
    'GenerativeFidelityAnalyzer',
    'FidelityBenchmark',
    'calculate_inception_score',
    'calculate_fid',
    'calculate_f1_score',
    'evaluate_generative_fidelity',

    # Probability Analysis (Statistical Probability)
    'ProbabilityType',
    'DistributionType',
    'InferenceMethod',
    'ProbabilityResult',
    'JointProbabilityResult',
    'BayesianInferenceResult',
    'ProbabilityDistribution',
    'BaseProbabilityAnalyzer',
    'MarginalProbabilityAnalyzer',
    'JointProbabilityAnalyzer',
    'ConditionalProbabilityAnalyzer',
    'BayesianInferenceAnalyzer',
    'LikelihoodAnalyzer',
    'PredictiveProbabilityAnalyzer',
    'ProbabilityCalibrationAnalyzer',
    'compute_marginal_probability',
    'compute_joint_probability',
    'compute_conditional_probability',
    'perform_bayesian_inference',
    'analyze_calibration',
    'compute_predictive_uncertainty',

    # Divergence Analysis (Distribution Comparison)
    'DivergenceType',
    'KernelType',
    'DivergenceResult',
    'MMDResult',
    'WassersteinResult',
    'DivergenceComparison',
    'BaseDivergenceAnalyzer',
    'KLDivergenceAnalyzer',
    'JensenShannonDivergenceAnalyzer',
    'WassersteinDistanceAnalyzer',
    'MMDAnalyzer',
    'TotalVariationAnalyzer',
    'HellingerDistanceAnalyzer',
    'RenyiDivergenceAnalyzer',
    'ComprehensiveDivergenceAnalyzer',
    'compute_kl_divergence',
    'compute_js_divergence',
    'compute_wasserstein_distance',
    'compute_mmd',
    'compute_all_divergences',

    # Human Evaluation Analysis (MOS, Inter-rater Reliability)
    'EvaluationScale',
    'EvaluationDimension',
    'AgreementMetric',
    'MOSResult',
    'InterRaterAgreement',
    'PairwiseComparisonResult',
    'HumanEvaluationReport',
    'BaseHumanEvaluationAnalyzer',
    'MOSAnalyzer',
    'InterRaterReliabilityAnalyzer',
    'PairwiseComparisonAnalyzer',
    'MultiDimensionalEvaluator',
    'RaterQualityAnalyzer',
    'compute_mos',
    'compute_inter_rater_agreement',
    'analyze_pairwise_comparisons',
    'evaluate_multiple_dimensions',
    'assess_rater_quality',

    # Evaluation Dimensions Analysis (Multi-dimensional Evaluation)
    'EvaluationDimensionType',
    'BusinessMetric',
    'TechnologyMetric',
    'SustainabilityMetric',
    'ComplianceMetric',
    'PerformanceMetric',
    'StatisticalMetric',
    'DimensionScore',
    'RadarMatrixResult',
    'ComprehensiveEvaluationResult',
    'BaseDimensionAnalyzer',
    'BusinessDimensionAnalyzer',
    'TechnologyDimensionAnalyzer',
    'SustainabilityDimensionAnalyzer',
    'ComplianceDimensionAnalyzer',
    'PerformanceDimensionAnalyzer',
    'StatisticalDimensionAnalyzer',
    'RadarMatrixAnalyzer',
    'ComprehensiveEvaluator',
    'analyze_business_dimension',
    'analyze_technology_dimension',
    'analyze_sustainability_dimension',
    'analyze_compliance_dimension',
    'create_radar_matrix',
    'perform_comprehensive_evaluation',

    # Text Relevancy Analysis (27 Dimensions)
    'RelevancyType',
    'RelevancyLevel',
    'RelevancyScore',
    'RelevancyProfile',
    'RelevancyDimensionAnalyzer',
    'SemanticRelevancyAnalyzer',
    'FactualRelevancyAnalyzer',
    'ContextualRelevancyAnalyzer',
    'NegativeRelevancyAnalyzer',
    'UncertaintyRelevancyAnalyzer',
    'TemporalRelevancyAnalyzer',
    'ComprehensiveRelevancyScorer',
    'analyze_relevancy',
    'compare_texts_relevancy',

    # Performance and Governance Matrix Analysis
    'PerformanceCategory',
    'GovernanceCategory',
    'PerformanceRiskLevel',
    'PerformanceComplianceStatus',
    'PerfGovPerformanceMetric',
    'PerfGovGovernanceMetric',
    'MatrixResult',
    'PerformanceMatrixAnalyzer',
    'ExecutionPerformanceAnalyzer',
    'EfficiencyAnalyzer',
    'OptimizationAnalyzer',
    'PerfGovReliabilityAnalyzer',
    'GovernanceMatrixAnalyzer',
    'PermissionAnalyzer',
    'RiskAssessmentAnalyzer',
    'ComplianceScoreAnalyzer',
    'PerfGovAccountabilityAnalyzer',
    'PerfGovTransparencyAnalyzer',
    'IntegratedMatrixAnalyzer',
    'analyze_performance',
    'analyze_governance',
    'analyze_integrated',

    # Factual Consistency Analysis (QuestEval, FactCC, Hallucination Detection)
    'ConsistencyLevel',
    'FactualHallucinationType',
    'EntailmentLabel',
    'ConsistencyScore',
    'HallucinationResult',
    'QuestionAnswerPair',
    'FactualEvaluationResult',
    'QuestEvalAnalyzer',
    'FactCCAnalyzer',
    'HallucinationDetector',
    'BERTScoreAnalyzer',
    'BARTScoreAnalyzer',
    'METEORAnalyzer',
    'FactualConsistencyEvaluator',
    'check_factual_consistency',
    'detect_hallucinations',
    'evaluate_factual_quality',

    # Diversity and Creativity Analysis
    'DiversityType',
    'CreativityLevel',
    'NoveltyType',
    'DiversityScore',
    'CreativityScore',
    'NoveltyScore',
    'DiversityProfile',
    'LexicalDiversityAnalyzer',
    'SemanticDiversityAnalyzer',
    'StructuralDiversityAnalyzer',
    'CreativityAnalyzer',
    'NoveltyAnalyzer',
    'SelfBLEUAnalyzer',
    'DistinctNAnalyzer',
    'ComprehensiveDiversityEvaluator',
    'analyze_diversity',
    'analyze_creativity',
    'calculate_distinct_n',
    'calculate_self_bleu',

    # RAI Pillar Analysis (Five Pillars of Responsible AI)
    'PillarType',
    'PillarComplianceLevel',
    'MaturityLevel',
    'PillarRiskCategory',
    'DataSensitivity',
    'PillarControlType',
    'PillarScore',
    'PillarPrivacyMetrics',
    'PillarTransparencyMetrics',
    'PillarRobustnessMetrics',
    'PillarSafetyMetrics',
    'PillarAccountabilityMetrics',
    'CrossPillarDependency',
    'RAIPillarAssessment',
    'PrivacyPillarAnalyzer',
    'TransparencyPillarAnalyzer',
    'RobustnessPillarAnalyzer',
    'SafetyPillarAnalyzer',
    'AccountabilityPillarAnalyzer',
    'FivePillarAnalyzer',
    'PillarBenchmarkAnalyzer',
    'analyze_privacy_pillar',
    'analyze_transparency_pillar',
    'analyze_robustness_pillar',
    'analyze_safety_pillar',
    'analyze_accountability_pillar',
    'analyze_all_pillars',
    'benchmark_pillars',

    # Data Policy Analysis (Masking, Reduction, Retention)
    'MaskingTechnique',
    'DataSensitivityLevel',
    'RetentionCategory',
    'DataReductionMethod',
    'DataComplianceStatus',
    'PIIType',
    'MaskingResult',
    'DataReductionResult',
    'RetentionAnalysisResult',
    'DataClassificationResult',
    'DataQualityMetrics',
    'DataPolicyAssessment',
    'DataMaskingAnalyzer',
    'DataReductionAnalyzer',
    'DataRetentionAnalyzer',
    'DataClassificationAnalyzer',
    'DataQualityAnalyzer',
    'DataPolicyAnalyzer',
    'analyze_data_masking',
    'analyze_data_reduction',
    'analyze_data_retention',
    'classify_data',
    'analyze_data_quality',
    'analyze_data_policies',

    # Validation Techniques Analysis (Statistical, Performance, Fairness)
    'ValidationMethod',
    'ValidationStatus',
    'MetricType',
    'FairnessMetric',
    'RobustnessTest',
    'CalibrationMethod',
    'ValidationResult',
    'CrossValidationResult',
    'MetricValidationResult',
    'FairnessValidationResult',
    'RobustnessValidationResult',
    'CalibrationResult',
    'ComprehensiveValidationReport',
    'StatisticalValidationAnalyzer',
    'PerformanceValidationAnalyzer',
    'FairnessValidationAnalyzer',
    'RobustnessValidationAnalyzer',
    'CalibrationValidationAnalyzer',
    'ComprehensiveValidationAnalyzer',
    'perform_cross_validation',
    'validate_metrics',
    'validate_fairness',
    'validate_robustness',
    'validate_calibration',
    'comprehensive_validation',

    # Control Framework Analysis (Hard/Soft Controls, User Notifications)
    'ControlType',
    'ControlCategory',
    'ControlDomain',
    'ControlStatus',
    'EffectivenessRating',
    'NotificationType',
    'NotificationChannel',
    'ControlRiskLevel',
    'Control',
    'ControlAssessment',
    'HardControlMetrics',
    'SoftControlMetrics',
    'NotificationMetrics',
    'ControlFrameworkAssessment',
    'HardControlAnalyzer',
    'SoftControlAnalyzer',
    'UserNotificationAnalyzer',
    'ControlEffectivenessAnalyzer',
    'ControlGapAnalyzer',
    'ControlFrameworkAnalyzer',
    'analyze_hard_controls',
    'analyze_soft_controls',
    'analyze_notifications',
    'analyze_control_framework',
    'identify_control_gaps',

    # ==========================================================================
    # 12-PILLAR TRUSTWORTHY AI FRAMEWORK EXPORTS
    # ==========================================================================

    # Portability Analysis (Pillar 8: Portable AI)
    'AbstractionLevel',
    'VendorLockInRisk',
    'PortabilityScore',
    'ModelCapability',
    'InteroperabilityStandard',
    'MigrationComplexity',
    'AbstractionMetrics',
    'VendorDependency',
    'VendorIndependenceMetrics',
    'ModelCompatibilityMapping',
    'MultiModelMetrics',
    'PortabilityTestResult',
    'PortabilityTestMetrics',
    'InteroperabilityMetrics',
    'PortabilityAssessment',
    'AbstractionLayerAnalyzer',
    'InterfaceDesignAnalyzer',
    'VendorIndependenceAnalyzer',
    'LockInRiskAnalyzer',
    'MultiModelAnalyzer',
    'CapabilityMappingAnalyzer',
    'PortabilityTestAnalyzer',
    'MigrationValidationAnalyzer',
    'InteroperabilityAnalyzer',
    'StandardsComplianceAnalyzer',
    'PortabilityAnalyzer',

    # Trust Calibration Analysis (Pillar 1: Trust AI Expanded)
    'ConfidenceLevel',
    'TrustZone',
    'CalibrationQuality',
    'TrustFailureType',
    'TrustSignalType',
    'TrustRepairStrategy',
    'UserTrustState',
    'ConfidenceSignal',
    'ConfidenceSignalingMetrics',
    'CalibrationBin',
    'CalibrationMetrics',
    'TrustZonePolicy',
    'TrustZoneMetrics',
    'TrustFailureEvent',
    'TrustFailureMetrics',
    'UserTrustProfile',
    'TrustDynamicsMetrics',
    'TrustAssessment',
    'ConfidenceSignalAnalyzer',
    'UncertaintyCommunicationAnalyzer',
    'TrustCalibrationMetricsAnalyzer',
    'ReliabilityDiagramAnalyzer',
    'TrustZoneAnalyzer',
    'TrustBoundaryAnalyzer',
    'TrustFailureAnalyzer',
    'UserTrustDynamicsAnalyzer',
    'OvertrustPreventionAnalyzer',
    'ExpandedTrustCalibrationAnalyzer',

    # Lifecycle Governance Analysis (Pillar 2: Responsible AI Lifecycle)
    'LifecycleStage',
    'StageStatus',
    'RiskClassification',
    'GateType',
    'OversightLevel',
    'DocumentationType',
    'LifecycleComplianceStatus',
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
    'DesignStageAnalyzer',
    'BuildStageAnalyzer',
    'TestStageAnalyzer',
    'DeployStageAnalyzer',
    'RunStageAnalyzer',
    'RetireStageAnalyzer',
    'GateReviewAnalyzer',
    'RiskClassificationAnalyzer',
    'LifecycleHumanOversightAnalyzer',
    'ExpandedLifecycleGovernanceAnalyzer',

    # Robustness Dimensions Analysis (Pillar 6: Robust AI Expanded)
    'RobustnessDimension',
    'RobustnessLevel',
    'AdversarialAttackType',
    'DriftType',
    'FailureMode',
    'RecoveryStrategy',
    'StressTestType',
    'InputRobustnessMetrics',
    'DataRobustnessMetrics',
    'ModelRobustnessMetrics',
    'SystemRobustnessMetrics',
    'BehavioralRobustnessMetrics',
    'OperationalRobustnessMetrics',
    'RobustnessDriftMetrics',
    'FailureModeAnalysis',
    'StressTestResult',
    'RobustnessAssessment',
    'InputRobustnessAnalyzer',
    'AdversarialDefenseAnalyzer',
    'DataRobustnessAnalyzer',
    'ModelRobustnessAnalyzer',
    'SystemRobustnessAnalyzer',
    'BehavioralRobustnessAnalyzer',
    'OperationalRobustnessAnalyzer',
    'DriftDetectionAnalyzer',
    'FailureModeAnalyzer',
    'RobustnessAnalyzer',

    # ==========================================================================
    # MASTER DATA ANALYSIS FRAMEWORK EXPORTS
    # ==========================================================================

    # Data Lifecycle Analysis (18 Data Analysis Categories)
    'DataAssetType',
    'SensitiveDataType',
    'DataQualityDimension',
    'DriftSeverity',
    'FeatureType',
    'BiasSource',
    'IncidentSeverity',
    'RetentionStatus',
    'AccessLevel',
    'DataValidationStatus',
    'DataAsset',
    'SensitiveDataFinding',
    'DataLifecycleQualityMetrics',
    'DriftDetectionResult',
    'FeatureAnalysisResult',
    'DataBiasAnalysisResult',
    'InputContractViolation',
    'TrainingDataMetrics',
    'PerformanceSubsetResult',
    'FaithfulnessResult',
    'DataIncident',
    'RetentionRecord',
    'AccessAuditRecord',
    'DataLifecycleAssessment',
    'DataInventoryAnalyzer',
    'DataCatalogAnalyzer',
    'DataPIIDetectionAnalyzer',
    'PHIDetectionAnalyzer',
    'LifecycleDataMinimizationAnalyzer',
    'LifecycleDataQualityAnalyzer',
    'DataConsistencyAnalyzer',
    'EDAAnalyzer',
    'DataBiasAnalyzer',
    'FeatureEngineeringAnalyzer',
    'LifecycleDataDriftAnalyzer',
    'InputContractAnalyzer',
    'TrainingDataAnalyzer',
    'PerformanceSubsetAnalyzer',
    'DataFaithfulnessAnalyzer',
    'LifecycleDataRobustnessAnalyzer',
    'DataExplainabilityAnalyzer',
    'DataAccessAnalyzer',
    'LifecycleDataRetentionAnalyzer',
    'DataIncidentAnalyzer',
    'DataLifecycleAnalyzer',
    'analyze_data_inventory',
    'detect_sensitive_data',
    'analyze_lifecycle_data_quality',
    'detect_data_drift',
    'analyze_data_bias',
    'validate_input_contract',
    'analyze_training_data',
    'comprehensive_data_assessment',

    # Model Internals Analysis (Architecture, Hyperparameters, Loss, Ensemble)
    'ModelArchitectureType',
    'LayerType',
    'HyperparameterType',
    'LossFunctionType',
    'EnsembleMethod',
    'OptimizationStatus',
    'OverfittingStatus',
    'FairnessMetricType',
    'TimeSeriesPattern',
    'LayerInfo',
    'ArchitectureMetrics',
    'HyperparameterConfig',
    'HyperparameterSensitivity',
    'LossLandscapeMetrics',
    'TrainingDynamics',
    'EnsembleModelInfo',
    'EnsembleMetrics',
    'TimeSeriesMetrics',
    'ModelFairnessMetrics',
    'ModelCalibrationMetrics',
    'GeneralizationMetrics',
    'ModelComparisonResult',
    'ModelInternalsAssessment',
    'ModelArchitectureAnalyzer',
    'LayerAnalyzer',
    'HyperparameterAnalyzer',
    'HyperparameterTuningAnalyzer',
    'LossFunctionAnalyzer',
    'LossComparisonAnalyzer',
    'EnsembleAnalyzer',
    'TimeSeriesModelAnalyzer',
    'ModelFairnessAnalyzer',
    'ModelCalibrationAnalyzer',
    'TrainingDynamicsAnalyzer',
    'GeneralizationAnalyzer',
    'ModelComparisonAnalyzer',
    'ModelInternalsAnalyzer',
    'analyze_model_architecture',
    'analyze_hyperparameters',
    'analyze_loss_landscape',
    'analyze_ensemble',
    'analyze_model_calibration',
    'analyze_generalization',
    'compare_models',
    'comprehensive_model_assessment',

    # Deep Learning Analysis (Training Stability, Gradients, Weights, Activations)
    'TrainingStability',
    'GradientHealth',
    'WeightStatus',
    'ActivationHealth',
    'DLCalibrationStatus',
    'DLRobustnessLevel',
    'RegularizationType',
    'AttentionPattern',
    'RepresentationQuality',
    'GradientMetrics',
    'WeightMetrics',
    'ActivationMetrics',
    'AttentionMetrics',
    'TrainingStabilityMetrics',
    'ComplexityMetrics',
    'DLCalibrationMetrics',
    'AdversarialRobustnessMetrics',
    'PerturbationRobustnessMetrics',
    'RepresentationMetrics',
    'RegularizationMetrics',
    'DeepLearningAssessment',
    'TrainingStabilityAnalyzer',
    'GradientAnalyzer',
    'WeightAnalyzer',
    'ActivationAnalyzer',
    'AttentionAnalyzer',
    'DeepLearningCalibrationAnalyzer',
    'DLAdversarialRobustnessAnalyzer',
    'PerturbationRobustnessAnalyzer',
    'RepresentationAnalyzer',
    'RegularizationAnalyzer',
    'DeepLearningAnalyzer',
    'analyze_training_stability',
    'analyze_gradients',
    'analyze_weights',
    'analyze_activations',
    'analyze_attention',
    'analyze_deep_learning_calibration',
    'analyze_adversarial_robustness',
    'analyze_representations',
    'comprehensive_deep_learning_assessment',

    # Computer Vision Analysis (Image Quality, Noise, Detection, Segmentation)
    'ImageFormat',
    'ColorSpace',
    'NoiseType',
    'AugmentationType',
    'CVTaskType',
    'CVQualityLevel',
    'DetectionMetricType',
    'SegmentationMetricType',
    'GenerationMetricType',
    'ImageMetadata',
    'ImageQualityMetrics',
    'NoiseAnalysisResult',
    'SpatialBiasMetrics',
    'ClassificationMetrics',
    'BoundingBox',
    'DetectionMetrics',
    'SegmentationMetrics',
    'CVGenerationMetrics',
    'AugmentationAnalysisResult',
    'VisualRobustnessResult',
    'SaliencyAnalysisResult',
    'ComputerVisionAssessment',
    'ImageQualityAnalyzer',
    'ImageMetadataAnalyzer',
    'NoiseAnalyzer',
    'SpatialBiasAnalyzer',
    'ClassificationMetricsAnalyzer',
    'DetectionMetricsAnalyzer',
    'SegmentationMetricsAnalyzer',
    'GenerationMetricsAnalyzer',
    'AugmentationAnalyzer',
    'VisualRobustnessAnalyzer',
    'SaliencyAnalyzer',
    'ComputerVisionAnalyzer',
    'analyze_image_quality',
    'analyze_noise',
    'analyze_spatial_bias',
    'calculate_classification_metrics',
    'calculate_detection_map',
    'calculate_segmentation_metrics',
    'analyze_augmentation_effectiveness',
    'comprehensive_cv_assessment',

    # NLP Comprehensive Analysis (Text Quality, Hallucination, Bias/Toxicity)
    'TextQualityDimension',
    'NLPHallucinationType',
    'BiasType',
    'ToxicityType',
    'PromptAttackType',
    'NLPTaskType',
    'NLPSeverityLevel',
    'TextStatistics',
    'TextQualityMetrics',
    'NLPHallucinationInstance',
    'HallucinationAnalysisResult',
    'BiasInstance',
    'NLPBiasAnalysisResult',
    'ToxicityInstance',
    'ToxicityAnalysisResult',
    'PromptSensitivityResult',
    'LanguageModelMetrics',
    'SemanticSimilarityResult',
    'SummarizationMetrics',
    'TranslationMetrics',
    'QAMetrics',
    'DialogueMetrics',
    'NLPAssessment',
    'TextQualityAnalyzer',
    'TextPreprocessingAnalyzer',
    'NLPHallucinationAnalyzer',
    'NLPBiasAnalyzer',
    'ToxicityAnalyzer',
    'PromptSensitivityAnalyzer',
    'LanguageModelAnalyzer',
    'SemanticAnalyzer',
    'SummarizationAnalyzer',
    'TranslationAnalyzer',
    'QAAnalyzer',
    'DialogueAnalyzer',
    'NLPComprehensiveAnalyzer',
    'analyze_text_quality',
    'detect_hallucinations',
    'analyze_nlp_bias',
    'analyze_toxicity',
    'calculate_rouge_scores',
    'calculate_bleu_score',
    'calculate_qa_metrics',
    'calculate_semantic_similarity',
    'comprehensive_nlp_assessment',

    # RAG Comprehensive Analysis (Chunking, Embeddings, Retrieval, Generation)
    'ChunkingStrategy',
    'EmbeddingModel',
    'VectorDBType',
    'RetrievalMethod',
    'IndexType',
    'CacheStrategy',
    'RAGQualityLevel',
    'ChunkInfo',
    'ChunkingMetrics',
    'EmbeddingMetrics',
    'VectorDBMetrics',
    'RetrievalResult',
    'RetrievalMetrics',
    'RAGGenerationMetrics',
    'ContextWindowMetrics',
    'CacheMetrics',
    'PipelineLatencyBreakdown',
    'CostMetrics',
    'RAGAssessment',
    'ChunkingAnalyzer',
    'EmbeddingAnalyzer',
    'VectorDBAnalyzer',
    'RetrievalAnalyzer',
    'RAGGenerationAnalyzer',
    'ContextWindowAnalyzer',
    'CacheAnalyzer',
    'RAGPipelineAnalyzer',
    'CostAnalyzer',
    'RAGComprehensiveAnalyzer',
    'analyze_chunking',
    'analyze_embeddings',
    'analyze_retrieval',
    'analyze_rag_generation',
    'analyze_pipeline_latency',
    'comprehensive_rag_assessment',

    # AI Security Comprehensive Analysis (ML, DL, CV, NLP, RAG Security)
    'AttackDomain',
    'AttackVector',
    'ThreatSeverity',
    'MitigationStatus',
    'SecurityPosture',
    'AttackPhase',
    'DefenseLayer',
    'ThreatProfile',
    'VulnerabilityAssessment',
    'MitigationControl',
    'AdversarialAttackResult',
    'PromptInjectionResult',
    'DataLeakageResult',
    'ModelExtractionResult',
    'BackdoorDetectionResult',
    'SecurityIncident',
    'SecurityMetrics',
    'SecurityAssessment',
    'MLSecurityAnalyzer',
    'DLSecurityAnalyzer',
    'CVSecurityAnalyzer',
    'NLPSecurityAnalyzer',
    'RAGSecurityAnalyzer',
    'InfrastructureSecurityAnalyzer',
    'SupplyChainSecurityAnalyzer',
    'IncidentResponseAnalyzer',
    'AISecurityComprehensiveAnalyzer',
    'analyze_ml_security',
    'analyze_nlp_security',
    'analyze_rag_security',
    'comprehensive_security_assessment',
    'analyze_security_incident',
    'calculate_security_metrics',
]
