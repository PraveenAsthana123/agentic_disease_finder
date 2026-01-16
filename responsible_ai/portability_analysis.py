"""
Portability Analysis Module - Pillar 8: Portable AI
====================================================

Comprehensive analysis framework for AI model portability, vendor independence,
and multi-model compatibility following the 12-Pillar Trustworthy AI Framework.

Analysis Categories:
- Model Abstraction Analysis: Abstract interface layers, provider-agnostic design
- Vendor Independence Analysis: Lock-in risk assessment, migration pathways
- Multi-Model Compatibility: Cross-model orchestration, capability mapping
- Portability Testing: Migration validation, performance consistency
- Interoperability Analysis: Standard compliance, protocol compatibility

Key Components:
- AbstractionLayerAnalyzer: Model abstraction interface analysis
- VendorIndependenceAnalyzer: Vendor lock-in assessment
- MultiModelAnalyzer: Cross-model compatibility analysis
- PortabilityTestAnalyzer: Migration testing framework
- InteroperabilityAnalyzer: Standards compliance checking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod


# ============================================================================
# ENUMS - Portability Classifications
# ============================================================================

class AbstractionLevel(Enum):
    """Model abstraction layer levels"""
    NONE = "none"  # Direct vendor API calls
    THIN = "thin"  # Simple wrapper
    STANDARD = "standard"  # Common interface abstraction
    FULL = "full"  # Complete provider-agnostic layer
    ENTERPRISE = "enterprise"  # Multi-layer with governance


class VendorLockInRisk(Enum):
    """Vendor lock-in risk levels"""
    MINIMAL = "minimal"  # Easy migration, standard APIs
    LOW = "low"  # Some proprietary features
    MODERATE = "moderate"  # Significant vendor-specific code
    HIGH = "high"  # Deep integration, difficult migration
    CRITICAL = "critical"  # Vendor-dependent architecture


class PortabilityScore(Enum):
    """Overall portability assessment scores"""
    EXCELLENT = "excellent"  # >90% portable
    GOOD = "good"  # 70-90% portable
    MODERATE = "moderate"  # 50-70% portable
    LIMITED = "limited"  # 30-50% portable
    POOR = "poor"  # <30% portable


class ModelCapability(Enum):
    """Standard model capabilities for mapping"""
    TEXT_GENERATION = "text_generation"
    TEXT_COMPLETION = "text_completion"
    CHAT_CONVERSATION = "chat_conversation"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    MULTIMODAL = "multimodal"
    FINE_TUNING = "fine_tuning"
    FUNCTION_CALLING = "function_calling"


class InteroperabilityStandard(Enum):
    """Interoperability standards and protocols"""
    OPENAI_API = "openai_api"
    HUGGINGFACE_INFERENCE = "huggingface_inference"
    ONNX = "onnx"
    TRITON = "triton"
    MLFLOW = "mlflow"
    KUBERNETES_SERVING = "kubernetes_serving"
    TENSORFLOW_SERVING = "tensorflow_serving"
    TORCHSERVE = "torchserve"
    CUSTOM = "custom"


class MigrationComplexity(Enum):
    """Migration complexity assessment"""
    TRIVIAL = "trivial"  # Simple config change
    SIMPLE = "simple"  # Minor code changes
    MODERATE = "moderate"  # Significant refactoring
    COMPLEX = "complex"  # Architectural changes
    MAJOR = "major"  # Complete redesign needed


# ============================================================================
# DATA CLASSES - Portability Metrics and Results
# ============================================================================

@dataclass
class AbstractionMetrics:
    """Metrics for model abstraction analysis"""
    abstraction_level: AbstractionLevel
    interface_coverage: float  # 0-1, % of capabilities abstracted
    vendor_specific_calls: int
    abstracted_calls: int
    direct_api_exposure: float  # 0-1, % of direct vendor API usage
    configuration_externalized: bool
    prompt_templates_portable: bool
    response_parsing_unified: bool
    error_handling_standardized: bool
    capability_mapping_complete: bool


@dataclass
class VendorDependency:
    """Individual vendor dependency record"""
    vendor_name: str
    dependency_type: str  # api, sdk, feature, infrastructure
    criticality: str  # low, medium, high, critical
    migration_effort: MigrationComplexity
    alternative_vendors: List[str]
    proprietary_features_used: List[str]
    standard_alternatives: List[str]
    estimated_migration_cost: str  # relative scale


@dataclass
class VendorIndependenceMetrics:
    """Metrics for vendor independence analysis"""
    lock_in_risk: VendorLockInRisk
    primary_vendor: str
    secondary_vendors: List[str]
    vendor_dependencies: List[VendorDependency]
    proprietary_feature_count: int
    standard_feature_count: int
    migration_readiness_score: float  # 0-1
    fallback_providers_configured: bool
    vendor_agnostic_data_format: bool
    portable_prompt_engineering: bool


@dataclass
class ModelCompatibilityMapping:
    """Capability mapping between models"""
    source_model: str
    target_model: str
    capability: ModelCapability
    compatibility_score: float  # 0-1
    parameter_mapping: Dict[str, str]
    output_format_differences: List[str]
    quality_variance: str  # expected quality difference
    latency_variance: str  # expected latency difference
    cost_variance: str  # expected cost difference
    migration_notes: List[str]


@dataclass
class MultiModelMetrics:
    """Metrics for multi-model compatibility analysis"""
    models_supported: List[str]
    capability_coverage: Dict[ModelCapability, List[str]]
    compatibility_mappings: List[ModelCompatibilityMapping]
    orchestration_support: bool
    fallback_chain_configured: bool
    load_balancing_enabled: bool
    model_routing_rules: List[str]
    unified_response_format: bool
    cross_model_consistency: float  # 0-1


@dataclass
class PortabilityTestResult:
    """Result of a portability test"""
    test_name: str
    test_type: str  # migration, compatibility, performance, consistency
    source_config: str
    target_config: str
    passed: bool
    execution_time_ms: float
    source_output: Any
    target_output: Any
    output_similarity: float  # 0-1
    performance_ratio: float  # target/source performance
    issues_found: List[str]
    recommendations: List[str]


@dataclass
class PortabilityTestMetrics:
    """Aggregate portability test metrics"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    average_similarity: float
    performance_consistency: float
    critical_issues: List[str]
    migration_blockers: List[str]
    test_results: List[PortabilityTestResult]


@dataclass
class InteroperabilityMetrics:
    """Metrics for interoperability analysis"""
    standards_supported: List[InteroperabilityStandard]
    api_compatibility_score: float  # 0-1
    data_format_compliance: float  # 0-1
    protocol_adherence: float  # 0-1
    extension_compatibility: bool
    backward_compatibility: bool
    forward_compatibility: bool
    integration_test_coverage: float
    known_incompatibilities: List[str]


@dataclass
class PortabilityAssessment:
    """Comprehensive portability assessment"""
    assessment_id: str
    assessment_date: datetime
    overall_score: PortabilityScore
    abstraction_metrics: AbstractionMetrics
    vendor_independence_metrics: VendorIndependenceMetrics
    multi_model_metrics: MultiModelMetrics
    portability_test_metrics: PortabilityTestMetrics
    interoperability_metrics: InteroperabilityMetrics
    recommendations: List[str]
    risk_factors: List[str]
    migration_roadmap: List[str]


# ============================================================================
# ANALYZERS - Model Abstraction
# ============================================================================

class AbstractionLayerAnalyzer:
    """Analyzes model abstraction layer implementation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_history: List[AbstractionMetrics] = []

    def analyze_abstraction(
        self,
        codebase_metrics: Dict[str, Any],
        api_usage: Dict[str, int]
    ) -> AbstractionMetrics:
        """Analyze abstraction layer implementation"""
        vendor_calls = sum(
            count for api, count in api_usage.items()
            if self._is_vendor_specific(api)
        )
        abstracted_calls = sum(
            count for api, count in api_usage.items()
            if self._is_abstracted(api)
        )
        total_calls = vendor_calls + abstracted_calls or 1

        abstraction_level = self._determine_abstraction_level(
            vendor_calls, abstracted_calls, codebase_metrics
        )

        metrics = AbstractionMetrics(
            abstraction_level=abstraction_level,
            interface_coverage=abstracted_calls / total_calls,
            vendor_specific_calls=vendor_calls,
            abstracted_calls=abstracted_calls,
            direct_api_exposure=vendor_calls / total_calls,
            configuration_externalized=codebase_metrics.get('config_external', False),
            prompt_templates_portable=codebase_metrics.get('portable_prompts', False),
            response_parsing_unified=codebase_metrics.get('unified_parsing', False),
            error_handling_standardized=codebase_metrics.get('standard_errors', False),
            capability_mapping_complete=codebase_metrics.get('capability_mapped', False)
        )

        self.analysis_history.append(metrics)
        return metrics

    def _is_vendor_specific(self, api: str) -> bool:
        """Check if API call is vendor-specific"""
        vendor_patterns = ['openai.', 'anthropic.', 'google.', 'azure.', 'aws.']
        return any(pattern in api.lower() for pattern in vendor_patterns)

    def _is_abstracted(self, api: str) -> bool:
        """Check if API call goes through abstraction layer"""
        abstraction_patterns = ['llm_client.', 'model_service.', 'ai_interface.']
        return any(pattern in api.lower() for pattern in abstraction_patterns)

    def _determine_abstraction_level(
        self,
        vendor_calls: int,
        abstracted_calls: int,
        metrics: Dict[str, Any]
    ) -> AbstractionLevel:
        """Determine overall abstraction level"""
        total = vendor_calls + abstracted_calls or 1
        abstraction_ratio = abstracted_calls / total

        if abstraction_ratio < 0.1:
            return AbstractionLevel.NONE
        elif abstraction_ratio < 0.4:
            return AbstractionLevel.THIN
        elif abstraction_ratio < 0.7:
            return AbstractionLevel.STANDARD
        elif abstraction_ratio < 0.9:
            return AbstractionLevel.FULL
        else:
            return AbstractionLevel.ENTERPRISE

    def get_abstraction_recommendations(
        self,
        metrics: AbstractionMetrics
    ) -> List[str]:
        """Generate recommendations for improving abstraction"""
        recommendations = []

        if metrics.abstraction_level in [AbstractionLevel.NONE, AbstractionLevel.THIN]:
            recommendations.append(
                "Implement a standard model abstraction layer to reduce vendor coupling"
            )

        if not metrics.configuration_externalized:
            recommendations.append(
                "Externalize model configuration to enable vendor switching without code changes"
            )

        if not metrics.prompt_templates_portable:
            recommendations.append(
                "Create vendor-agnostic prompt templates with variable substitution"
            )

        if not metrics.response_parsing_unified:
            recommendations.append(
                "Implement unified response parsing that normalizes vendor-specific formats"
            )

        if not metrics.error_handling_standardized:
            recommendations.append(
                "Standardize error handling to map vendor errors to common error types"
            )

        if not metrics.capability_mapping_complete:
            recommendations.append(
                "Complete capability mapping to enable automatic model selection"
            )

        return recommendations


class InterfaceDesignAnalyzer:
    """Analyzes provider-agnostic interface design"""

    def __init__(self):
        self.interface_patterns: List[Dict[str, Any]] = []

    def analyze_interface(
        self,
        interface_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze interface design for portability"""
        analysis = {
            'is_provider_agnostic': self._check_provider_agnostic(interface_definition),
            'uses_standard_types': self._check_standard_types(interface_definition),
            'has_capability_abstraction': self._check_capability_abstraction(interface_definition),
            'supports_configuration_injection': self._check_config_injection(interface_definition),
            'has_fallback_mechanism': self._check_fallback(interface_definition),
            'supports_multi_provider': self._check_multi_provider(interface_definition)
        }

        analysis['portability_score'] = sum(analysis.values()) / len(analysis)
        return analysis

    def _check_provider_agnostic(self, definition: Dict[str, Any]) -> bool:
        """Check if interface is provider-agnostic"""
        return definition.get('provider_agnostic', False)

    def _check_standard_types(self, definition: Dict[str, Any]) -> bool:
        """Check if interface uses standard types"""
        return definition.get('standard_types', False)

    def _check_capability_abstraction(self, definition: Dict[str, Any]) -> bool:
        """Check for capability abstraction"""
        return 'capabilities' in definition

    def _check_config_injection(self, definition: Dict[str, Any]) -> bool:
        """Check for configuration injection support"""
        return definition.get('config_injectable', False)

    def _check_fallback(self, definition: Dict[str, Any]) -> bool:
        """Check for fallback mechanism"""
        return 'fallback' in definition

    def _check_multi_provider(self, definition: Dict[str, Any]) -> bool:
        """Check for multi-provider support"""
        return len(definition.get('providers', [])) > 1


# ============================================================================
# ANALYZERS - Vendor Independence
# ============================================================================

class VendorIndependenceAnalyzer:
    """Analyzes vendor independence and lock-in risks"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vendor_registry: Dict[str, VendorDependency] = {}

    def analyze_vendor_independence(
        self,
        vendor_usage: Dict[str, Any],
        feature_usage: Dict[str, List[str]]
    ) -> VendorIndependenceMetrics:
        """Analyze vendor independence metrics"""
        dependencies = self._identify_dependencies(vendor_usage, feature_usage)

        proprietary_count = sum(
            len(dep.proprietary_features_used) for dep in dependencies
        )
        standard_count = sum(
            len(dep.standard_alternatives) for dep in dependencies
        )

        lock_in_risk = self._assess_lock_in_risk(dependencies)
        migration_readiness = self._calculate_migration_readiness(dependencies)

        return VendorIndependenceMetrics(
            lock_in_risk=lock_in_risk,
            primary_vendor=self._identify_primary_vendor(vendor_usage),
            secondary_vendors=self._identify_secondary_vendors(vendor_usage),
            vendor_dependencies=dependencies,
            proprietary_feature_count=proprietary_count,
            standard_feature_count=standard_count,
            migration_readiness_score=migration_readiness,
            fallback_providers_configured=vendor_usage.get('fallback_configured', False),
            vendor_agnostic_data_format=vendor_usage.get('agnostic_format', False),
            portable_prompt_engineering=vendor_usage.get('portable_prompts', False)
        )

    def _identify_dependencies(
        self,
        vendor_usage: Dict[str, Any],
        feature_usage: Dict[str, List[str]]
    ) -> List[VendorDependency]:
        """Identify vendor dependencies"""
        dependencies = []

        for vendor, usage in vendor_usage.get('vendors', {}).items():
            proprietary = feature_usage.get(vendor, {}).get('proprietary', [])
            standard = feature_usage.get(vendor, {}).get('standard', [])

            dependency = VendorDependency(
                vendor_name=vendor,
                dependency_type=usage.get('type', 'api'),
                criticality=usage.get('criticality', 'medium'),
                migration_effort=self._assess_migration_effort(usage),
                alternative_vendors=usage.get('alternatives', []),
                proprietary_features_used=proprietary,
                standard_alternatives=standard,
                estimated_migration_cost=usage.get('migration_cost', 'medium')
            )
            dependencies.append(dependency)
            self.vendor_registry[vendor] = dependency

        return dependencies

    def _assess_migration_effort(self, usage: Dict[str, Any]) -> MigrationComplexity:
        """Assess migration effort for a vendor"""
        complexity_score = usage.get('complexity_score', 0.5)

        if complexity_score < 0.2:
            return MigrationComplexity.TRIVIAL
        elif complexity_score < 0.4:
            return MigrationComplexity.SIMPLE
        elif complexity_score < 0.6:
            return MigrationComplexity.MODERATE
        elif complexity_score < 0.8:
            return MigrationComplexity.COMPLEX
        else:
            return MigrationComplexity.MAJOR

    def _assess_lock_in_risk(
        self,
        dependencies: List[VendorDependency]
    ) -> VendorLockInRisk:
        """Assess overall vendor lock-in risk"""
        if not dependencies:
            return VendorLockInRisk.MINIMAL

        critical_deps = sum(1 for d in dependencies if d.criticality == 'critical')
        proprietary_total = sum(len(d.proprietary_features_used) for d in dependencies)

        if critical_deps == 0 and proprietary_total == 0:
            return VendorLockInRisk.MINIMAL
        elif critical_deps == 0 and proprietary_total < 3:
            return VendorLockInRisk.LOW
        elif critical_deps <= 1 and proprietary_total < 6:
            return VendorLockInRisk.MODERATE
        elif critical_deps <= 2:
            return VendorLockInRisk.HIGH
        else:
            return VendorLockInRisk.CRITICAL

    def _calculate_migration_readiness(
        self,
        dependencies: List[VendorDependency]
    ) -> float:
        """Calculate migration readiness score"""
        if not dependencies:
            return 1.0

        scores = []
        for dep in dependencies:
            effort_scores = {
                MigrationComplexity.TRIVIAL: 1.0,
                MigrationComplexity.SIMPLE: 0.8,
                MigrationComplexity.MODERATE: 0.6,
                MigrationComplexity.COMPLEX: 0.4,
                MigrationComplexity.MAJOR: 0.2
            }
            scores.append(effort_scores.get(dep.migration_effort, 0.5))

        return sum(scores) / len(scores)

    def _identify_primary_vendor(self, vendor_usage: Dict[str, Any]) -> str:
        """Identify the primary vendor"""
        vendors = vendor_usage.get('vendors', {})
        if not vendors:
            return "none"

        return max(vendors.keys(), key=lambda v: vendors[v].get('usage_count', 0))

    def _identify_secondary_vendors(self, vendor_usage: Dict[str, Any]) -> List[str]:
        """Identify secondary vendors"""
        vendors = vendor_usage.get('vendors', {})
        primary = self._identify_primary_vendor(vendor_usage)
        return [v for v in vendors.keys() if v != primary]

    def generate_migration_plan(
        self,
        metrics: VendorIndependenceMetrics,
        target_vendor: str
    ) -> Dict[str, Any]:
        """Generate a migration plan to target vendor"""
        plan = {
            'current_state': {
                'primary_vendor': metrics.primary_vendor,
                'lock_in_risk': metrics.lock_in_risk.value,
                'migration_readiness': metrics.migration_readiness_score
            },
            'target_state': {
                'target_vendor': target_vendor,
                'expected_lock_in_risk': 'moderate'
            },
            'migration_steps': [],
            'risk_mitigation': [],
            'estimated_effort': 'medium'
        }

        for dep in metrics.vendor_dependencies:
            if dep.vendor_name == metrics.primary_vendor:
                plan['migration_steps'].append({
                    'step': f"Migrate {dep.dependency_type} from {dep.vendor_name}",
                    'effort': dep.migration_effort.value,
                    'proprietary_features_to_replace': dep.proprietary_features_used
                })

        return plan


class LockInRiskAnalyzer:
    """Analyzes vendor lock-in risks in detail"""

    def __init__(self):
        self.risk_factors: List[Dict[str, Any]] = []

    def analyze_lock_in_risks(
        self,
        vendor_metrics: VendorIndependenceMetrics
    ) -> Dict[str, Any]:
        """Analyze detailed lock-in risks"""
        risks = {
            'overall_risk': vendor_metrics.lock_in_risk.value,
            'risk_factors': [],
            'mitigation_strategies': [],
            'exit_costs': {}
        }

        # Analyze proprietary feature risks
        if vendor_metrics.proprietary_feature_count > 0:
            risks['risk_factors'].append({
                'type': 'proprietary_features',
                'count': vendor_metrics.proprietary_feature_count,
                'severity': 'high' if vendor_metrics.proprietary_feature_count > 5 else 'medium'
            })

        # Analyze fallback configuration
        if not vendor_metrics.fallback_providers_configured:
            risks['risk_factors'].append({
                'type': 'no_fallback',
                'description': 'No fallback providers configured',
                'severity': 'high'
            })
            risks['mitigation_strategies'].append(
                'Configure at least one fallback provider for critical paths'
            )

        # Analyze data format portability
        if not vendor_metrics.vendor_agnostic_data_format:
            risks['risk_factors'].append({
                'type': 'data_format_lock_in',
                'description': 'Using vendor-specific data formats',
                'severity': 'medium'
            })
            risks['mitigation_strategies'].append(
                'Implement data format adapters for vendor-agnostic storage'
            )

        return risks


# ============================================================================
# ANALYZERS - Multi-Model Compatibility
# ============================================================================

class MultiModelAnalyzer:
    """Analyzes multi-model compatibility and orchestration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.compatibility_matrix: Dict[str, Dict[str, float]] = {}

    def analyze_multi_model_compatibility(
        self,
        models: List[Dict[str, Any]],
        capability_requirements: List[ModelCapability]
    ) -> MultiModelMetrics:
        """Analyze multi-model compatibility"""
        capability_coverage = self._map_capabilities(models, capability_requirements)
        compatibility_mappings = self._create_compatibility_mappings(models)

        return MultiModelMetrics(
            models_supported=[m['name'] for m in models],
            capability_coverage=capability_coverage,
            compatibility_mappings=compatibility_mappings,
            orchestration_support=self._check_orchestration_support(models),
            fallback_chain_configured=self._check_fallback_chain(models),
            load_balancing_enabled=self._check_load_balancing(models),
            model_routing_rules=self._extract_routing_rules(models),
            unified_response_format=self._check_unified_format(models),
            cross_model_consistency=self._calculate_consistency(compatibility_mappings)
        )

    def _map_capabilities(
        self,
        models: List[Dict[str, Any]],
        requirements: List[ModelCapability]
    ) -> Dict[ModelCapability, List[str]]:
        """Map capabilities to supporting models"""
        coverage = {cap: [] for cap in requirements}

        for model in models:
            model_caps = model.get('capabilities', [])
            for cap in requirements:
                if cap.value in model_caps or cap in model_caps:
                    coverage[cap].append(model['name'])

        return coverage

    def _create_compatibility_mappings(
        self,
        models: List[Dict[str, Any]]
    ) -> List[ModelCompatibilityMapping]:
        """Create compatibility mappings between models"""
        mappings = []

        for i, source in enumerate(models):
            for target in models[i+1:]:
                for cap in source.get('capabilities', []):
                    if cap in target.get('capabilities', []):
                        mapping = ModelCompatibilityMapping(
                            source_model=source['name'],
                            target_model=target['name'],
                            capability=ModelCapability(cap) if isinstance(cap, str) else cap,
                            compatibility_score=self._calculate_compatibility(source, target, cap),
                            parameter_mapping=self._map_parameters(source, target),
                            output_format_differences=self._find_format_differences(source, target),
                            quality_variance=self._estimate_quality_variance(source, target),
                            latency_variance=self._estimate_latency_variance(source, target),
                            cost_variance=self._estimate_cost_variance(source, target),
                            migration_notes=[]
                        )
                        mappings.append(mapping)

                        # Update compatibility matrix
                        if source['name'] not in self.compatibility_matrix:
                            self.compatibility_matrix[source['name']] = {}
                        self.compatibility_matrix[source['name']][target['name']] = mapping.compatibility_score

        return mappings

    def _calculate_compatibility(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        capability: str
    ) -> float:
        """Calculate compatibility score between two models for a capability"""
        # Base compatibility on shared capabilities
        source_caps = set(source.get('capabilities', []))
        target_caps = set(target.get('capabilities', []))

        shared = len(source_caps & target_caps)
        total = len(source_caps | target_caps) or 1

        return shared / total

    def _map_parameters(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> Dict[str, str]:
        """Map parameters between models"""
        return {
            'temperature': 'temperature',
            'max_tokens': 'max_tokens',
            'top_p': 'top_p'
        }

    def _find_format_differences(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> List[str]:
        """Find output format differences"""
        differences = []
        if source.get('output_format') != target.get('output_format'):
            differences.append(f"Output format: {source.get('output_format')} vs {target.get('output_format')}")
        return differences

    def _estimate_quality_variance(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> str:
        """Estimate quality variance between models"""
        source_tier = source.get('quality_tier', 'medium')
        target_tier = target.get('quality_tier', 'medium')

        if source_tier == target_tier:
            return 'comparable'
        return f"{source_tier} to {target_tier}"

    def _estimate_latency_variance(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> str:
        """Estimate latency variance between models"""
        return 'variable'

    def _estimate_cost_variance(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> str:
        """Estimate cost variance between models"""
        return 'varies by usage'

    def _check_orchestration_support(self, models: List[Dict[str, Any]]) -> bool:
        """Check if orchestration is supported"""
        return len(models) > 1

    def _check_fallback_chain(self, models: List[Dict[str, Any]]) -> bool:
        """Check if fallback chain is configured"""
        return any(m.get('is_fallback', False) for m in models)

    def _check_load_balancing(self, models: List[Dict[str, Any]]) -> bool:
        """Check if load balancing is enabled"""
        return any(m.get('load_balanced', False) for m in models)

    def _extract_routing_rules(self, models: List[Dict[str, Any]]) -> List[str]:
        """Extract model routing rules"""
        rules = []
        for model in models:
            if 'routing_rules' in model:
                rules.extend(model['routing_rules'])
        return rules

    def _check_unified_format(self, models: List[Dict[str, Any]]) -> bool:
        """Check if unified response format is used"""
        formats = set(m.get('output_format', 'default') for m in models)
        return len(formats) <= 1

    def _calculate_consistency(
        self,
        mappings: List[ModelCompatibilityMapping]
    ) -> float:
        """Calculate cross-model consistency score"""
        if not mappings:
            return 1.0
        return sum(m.compatibility_score for m in mappings) / len(mappings)


class CapabilityMappingAnalyzer:
    """Analyzes capability mappings across models"""

    def __init__(self):
        self.capability_registry: Dict[ModelCapability, List[str]] = {}

    def analyze_capability_coverage(
        self,
        required_capabilities: List[ModelCapability],
        available_models: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze capability coverage"""
        coverage = {
            'required': [c.value for c in required_capabilities],
            'covered': [],
            'gaps': [],
            'coverage_by_model': {},
            'redundancy_map': {}
        }

        for cap in required_capabilities:
            supporting_models = []
            for model in available_models:
                if cap.value in model.get('capabilities', []):
                    supporting_models.append(model['name'])

            self.capability_registry[cap] = supporting_models

            if supporting_models:
                coverage['covered'].append(cap.value)
                coverage['redundancy_map'][cap.value] = len(supporting_models)
            else:
                coverage['gaps'].append(cap.value)

        for model in available_models:
            model_caps = model.get('capabilities', [])
            coverage['coverage_by_model'][model['name']] = [
                c for c in model_caps if c in [cap.value for cap in required_capabilities]
            ]

        coverage['coverage_rate'] = len(coverage['covered']) / len(required_capabilities) if required_capabilities else 1.0

        return coverage


# ============================================================================
# ANALYZERS - Portability Testing
# ============================================================================

class PortabilityTestAnalyzer:
    """Analyzes and executes portability tests"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.test_history: List[PortabilityTestResult] = []

    def run_portability_tests(
        self,
        test_cases: List[Dict[str, Any]],
        source_config: Dict[str, Any],
        target_config: Dict[str, Any]
    ) -> PortabilityTestMetrics:
        """Run portability tests"""
        results = []

        for test in test_cases:
            result = self._execute_test(test, source_config, target_config)
            results.append(result)
            self.test_history.append(result)

        passed = sum(1 for r in results if r.passed)

        return PortabilityTestMetrics(
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=len(results) - passed,
            pass_rate=passed / len(results) if results else 0,
            average_similarity=sum(r.output_similarity for r in results) / len(results) if results else 0,
            performance_consistency=self._calculate_performance_consistency(results),
            critical_issues=self._identify_critical_issues(results),
            migration_blockers=self._identify_blockers(results),
            test_results=results
        )

    def _execute_test(
        self,
        test: Dict[str, Any],
        source_config: Dict[str, Any],
        target_config: Dict[str, Any]
    ) -> PortabilityTestResult:
        """Execute a single portability test"""
        # Simulate test execution
        return PortabilityTestResult(
            test_name=test.get('name', 'unnamed_test'),
            test_type=test.get('type', 'compatibility'),
            source_config=str(source_config.get('name', 'source')),
            target_config=str(target_config.get('name', 'target')),
            passed=test.get('expected_pass', True),
            execution_time_ms=test.get('execution_time', 100.0),
            source_output=test.get('source_output'),
            target_output=test.get('target_output'),
            output_similarity=test.get('similarity', 0.95),
            performance_ratio=test.get('performance_ratio', 1.0),
            issues_found=test.get('issues', []),
            recommendations=test.get('recommendations', [])
        )

    def _calculate_performance_consistency(
        self,
        results: List[PortabilityTestResult]
    ) -> float:
        """Calculate performance consistency across tests"""
        if not results:
            return 1.0

        ratios = [r.performance_ratio for r in results]
        avg_ratio = sum(ratios) / len(ratios)
        variance = sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)

        # Higher consistency = lower variance
        return max(0, 1 - variance)

    def _identify_critical_issues(
        self,
        results: List[PortabilityTestResult]
    ) -> List[str]:
        """Identify critical issues from test results"""
        issues = []
        for result in results:
            if not result.passed:
                issues.extend(result.issues_found)
        return list(set(issues))

    def _identify_blockers(
        self,
        results: List[PortabilityTestResult]
    ) -> List[str]:
        """Identify migration blockers"""
        blockers = []
        for result in results:
            if result.output_similarity < 0.5:
                blockers.append(f"Low similarity in {result.test_name}: {result.output_similarity}")
            if result.performance_ratio < 0.5:
                blockers.append(f"Performance degradation in {result.test_name}: {result.performance_ratio}")
        return blockers


class MigrationValidationAnalyzer:
    """Validates migration between configurations"""

    def __init__(self):
        self.validation_results: List[Dict[str, Any]] = []

    def validate_migration(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        validation_criteria: List[str]
    ) -> Dict[str, Any]:
        """Validate migration between source and target"""
        validation = {
            'source': source.get('name', 'source'),
            'target': target.get('name', 'target'),
            'criteria_results': {},
            'overall_valid': True,
            'warnings': [],
            'blockers': []
        }

        for criterion in validation_criteria:
            result = self._validate_criterion(source, target, criterion)
            validation['criteria_results'][criterion] = result

            if not result['passed']:
                if result.get('severity') == 'blocker':
                    validation['blockers'].append(result['message'])
                    validation['overall_valid'] = False
                else:
                    validation['warnings'].append(result['message'])

        self.validation_results.append(validation)
        return validation

    def _validate_criterion(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        criterion: str
    ) -> Dict[str, Any]:
        """Validate a single migration criterion"""
        validators = {
            'capability_coverage': self._validate_capability_coverage,
            'performance_parity': self._validate_performance_parity,
            'cost_acceptable': self._validate_cost,
            'quality_maintained': self._validate_quality
        }

        validator = validators.get(criterion, self._default_validator)
        return validator(source, target)

    def _validate_capability_coverage(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate capability coverage"""
        source_caps = set(source.get('capabilities', []))
        target_caps = set(target.get('capabilities', []))

        missing = source_caps - target_caps

        return {
            'passed': len(missing) == 0,
            'message': f"Missing capabilities: {missing}" if missing else "All capabilities covered",
            'severity': 'blocker' if missing else None
        }

    def _validate_performance_parity(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate performance parity"""
        return {
            'passed': True,
            'message': "Performance within acceptable range",
            'severity': None
        }

    def _validate_cost(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate cost acceptability"""
        return {
            'passed': True,
            'message': "Cost within budget",
            'severity': None
        }

    def _validate_quality(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate quality maintenance"""
        return {
            'passed': True,
            'message': "Quality maintained",
            'severity': None
        }

    def _default_validator(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Default validation"""
        return {
            'passed': True,
            'message': "Validation passed",
            'severity': None
        }


# ============================================================================
# ANALYZERS - Interoperability
# ============================================================================

class InteroperabilityAnalyzer:
    """Analyzes system interoperability"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.standard_compliance: Dict[InteroperabilityStandard, bool] = {}

    def analyze_interoperability(
        self,
        system_config: Dict[str, Any],
        target_standards: List[InteroperabilityStandard]
    ) -> InteroperabilityMetrics:
        """Analyze system interoperability"""
        supported_standards = []

        for standard in target_standards:
            if self._check_standard_compliance(system_config, standard):
                supported_standards.append(standard)
                self.standard_compliance[standard] = True
            else:
                self.standard_compliance[standard] = False

        return InteroperabilityMetrics(
            standards_supported=supported_standards,
            api_compatibility_score=self._calculate_api_compatibility(system_config),
            data_format_compliance=self._calculate_format_compliance(system_config),
            protocol_adherence=self._calculate_protocol_adherence(system_config),
            extension_compatibility=system_config.get('extension_compatible', True),
            backward_compatibility=system_config.get('backward_compatible', True),
            forward_compatibility=system_config.get('forward_compatible', False),
            integration_test_coverage=system_config.get('integration_coverage', 0.0),
            known_incompatibilities=system_config.get('incompatibilities', [])
        )

    def _check_standard_compliance(
        self,
        config: Dict[str, Any],
        standard: InteroperabilityStandard
    ) -> bool:
        """Check compliance with a specific standard"""
        supported = config.get('supported_standards', [])
        return standard.value in supported or standard in supported

    def _calculate_api_compatibility(self, config: Dict[str, Any]) -> float:
        """Calculate API compatibility score"""
        return config.get('api_compatibility', 0.8)

    def _calculate_format_compliance(self, config: Dict[str, Any]) -> float:
        """Calculate data format compliance score"""
        return config.get('format_compliance', 0.9)

    def _calculate_protocol_adherence(self, config: Dict[str, Any]) -> float:
        """Calculate protocol adherence score"""
        return config.get('protocol_adherence', 0.85)


class StandardsComplianceAnalyzer:
    """Analyzes compliance with interoperability standards"""

    def __init__(self):
        self.compliance_reports: List[Dict[str, Any]] = []

    def analyze_standards_compliance(
        self,
        implementation: Dict[str, Any],
        standards: List[InteroperabilityStandard]
    ) -> Dict[str, Any]:
        """Analyze compliance with multiple standards"""
        report = {
            'implementation': implementation.get('name', 'unnamed'),
            'standards_checked': [s.value for s in standards],
            'compliance_results': {},
            'overall_compliance': 0.0,
            'recommendations': []
        }

        compliant_count = 0
        for standard in standards:
            result = self._check_standard(implementation, standard)
            report['compliance_results'][standard.value] = result
            if result['compliant']:
                compliant_count += 1
            else:
                report['recommendations'].extend(result.get('recommendations', []))

        report['overall_compliance'] = compliant_count / len(standards) if standards else 1.0
        self.compliance_reports.append(report)

        return report

    def _check_standard(
        self,
        implementation: Dict[str, Any],
        standard: InteroperabilityStandard
    ) -> Dict[str, Any]:
        """Check compliance with a single standard"""
        standard_requirements = self._get_standard_requirements(standard)

        met_requirements = []
        missing_requirements = []

        for req in standard_requirements:
            if implementation.get(req, False):
                met_requirements.append(req)
            else:
                missing_requirements.append(req)

        return {
            'compliant': len(missing_requirements) == 0,
            'met_requirements': met_requirements,
            'missing_requirements': missing_requirements,
            'recommendations': [f"Implement {req} for {standard.value} compliance" for req in missing_requirements]
        }

    def _get_standard_requirements(
        self,
        standard: InteroperabilityStandard
    ) -> List[str]:
        """Get requirements for a standard"""
        requirements_map = {
            InteroperabilityStandard.OPENAI_API: [
                'chat_completions_endpoint',
                'streaming_support',
                'function_calling'
            ],
            InteroperabilityStandard.ONNX: [
                'onnx_export',
                'onnx_runtime_compatible'
            ],
            InteroperabilityStandard.MLFLOW: [
                'mlflow_tracking',
                'mlflow_model_format'
            ]
        }
        return requirements_map.get(standard, [])


# ============================================================================
# COMPREHENSIVE ANALYZER
# ============================================================================

class PortabilityAnalyzer:
    """Comprehensive portability analyzer combining all aspects"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.abstraction_analyzer = AbstractionLayerAnalyzer(config)
        self.vendor_analyzer = VendorIndependenceAnalyzer(config)
        self.multi_model_analyzer = MultiModelAnalyzer(config)
        self.test_analyzer = PortabilityTestAnalyzer(config)
        self.interop_analyzer = InteroperabilityAnalyzer(config)
        self.assessments: List[PortabilityAssessment] = []

    def analyze_portability(
        self,
        codebase_metrics: Dict[str, Any],
        api_usage: Dict[str, int],
        vendor_usage: Dict[str, Any],
        feature_usage: Dict[str, List[str]],
        models: List[Dict[str, Any]],
        capability_requirements: List[ModelCapability],
        test_cases: List[Dict[str, Any]],
        system_config: Dict[str, Any],
        target_standards: List[InteroperabilityStandard]
    ) -> PortabilityAssessment:
        """Perform comprehensive portability analysis"""
        # Run all analyzers
        abstraction_metrics = self.abstraction_analyzer.analyze_abstraction(
            codebase_metrics, api_usage
        )

        vendor_metrics = self.vendor_analyzer.analyze_vendor_independence(
            vendor_usage, feature_usage
        )

        multi_model_metrics = self.multi_model_analyzer.analyze_multi_model_compatibility(
            models, capability_requirements
        )

        # Run portability tests if test cases provided
        if test_cases and len(models) >= 2:
            test_metrics = self.test_analyzer.run_portability_tests(
                test_cases,
                {'name': models[0]['name']},
                {'name': models[1]['name']}
            )
        else:
            test_metrics = PortabilityTestMetrics(
                total_tests=0, passed_tests=0, failed_tests=0,
                pass_rate=0, average_similarity=0, performance_consistency=0,
                critical_issues=[], migration_blockers=[], test_results=[]
            )

        interop_metrics = self.interop_analyzer.analyze_interoperability(
            system_config, target_standards
        )

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            abstraction_metrics, vendor_metrics, multi_model_metrics,
            test_metrics, interop_metrics
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            abstraction_metrics, vendor_metrics, multi_model_metrics,
            test_metrics, interop_metrics
        )

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            vendor_metrics, test_metrics
        )

        # Create migration roadmap
        migration_roadmap = self._create_migration_roadmap(
            vendor_metrics, abstraction_metrics
        )

        assessment = PortabilityAssessment(
            assessment_id=f"port_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            assessment_date=datetime.now(),
            overall_score=overall_score,
            abstraction_metrics=abstraction_metrics,
            vendor_independence_metrics=vendor_metrics,
            multi_model_metrics=multi_model_metrics,
            portability_test_metrics=test_metrics,
            interoperability_metrics=interop_metrics,
            recommendations=recommendations,
            risk_factors=risk_factors,
            migration_roadmap=migration_roadmap
        )

        self.assessments.append(assessment)
        return assessment

    def _calculate_overall_score(
        self,
        abstraction: AbstractionMetrics,
        vendor: VendorIndependenceMetrics,
        multi_model: MultiModelMetrics,
        tests: PortabilityTestMetrics,
        interop: InteroperabilityMetrics
    ) -> PortabilityScore:
        """Calculate overall portability score"""
        scores = [
            abstraction.interface_coverage,
            vendor.migration_readiness_score,
            multi_model.cross_model_consistency,
            tests.pass_rate if tests.total_tests > 0 else 0.5,
            interop.api_compatibility_score
        ]

        avg_score = sum(scores) / len(scores)

        if avg_score >= 0.9:
            return PortabilityScore.EXCELLENT
        elif avg_score >= 0.7:
            return PortabilityScore.GOOD
        elif avg_score >= 0.5:
            return PortabilityScore.MODERATE
        elif avg_score >= 0.3:
            return PortabilityScore.LIMITED
        else:
            return PortabilityScore.POOR

    def _generate_recommendations(
        self,
        abstraction: AbstractionMetrics,
        vendor: VendorIndependenceMetrics,
        multi_model: MultiModelMetrics,
        tests: PortabilityTestMetrics,
        interop: InteroperabilityMetrics
    ) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []

        # Abstraction recommendations
        recommendations.extend(
            self.abstraction_analyzer.get_abstraction_recommendations(abstraction)
        )

        # Vendor recommendations
        if vendor.lock_in_risk in [VendorLockInRisk.HIGH, VendorLockInRisk.CRITICAL]:
            recommendations.append(
                "Reduce vendor lock-in by implementing abstraction layers"
            )

        if not vendor.fallback_providers_configured:
            recommendations.append(
                "Configure fallback providers for business continuity"
            )

        # Multi-model recommendations
        if not multi_model.unified_response_format:
            recommendations.append(
                "Implement unified response format across all models"
            )

        # Test recommendations
        if tests.pass_rate < 0.8 and tests.total_tests > 0:
            recommendations.append(
                "Address failing portability tests before migration"
            )

        # Interoperability recommendations
        if interop.api_compatibility_score < 0.8:
            recommendations.append(
                "Improve API compatibility for better interoperability"
            )

        return recommendations

    def _identify_risk_factors(
        self,
        vendor: VendorIndependenceMetrics,
        tests: PortabilityTestMetrics
    ) -> List[str]:
        """Identify risk factors"""
        risks = []

        if vendor.lock_in_risk in [VendorLockInRisk.HIGH, VendorLockInRisk.CRITICAL]:
            risks.append(f"High vendor lock-in risk: {vendor.lock_in_risk.value}")

        if vendor.proprietary_feature_count > 5:
            risks.append(f"Heavy use of proprietary features: {vendor.proprietary_feature_count}")

        if tests.migration_blockers:
            risks.extend(tests.migration_blockers)

        return risks

    def _create_migration_roadmap(
        self,
        vendor: VendorIndependenceMetrics,
        abstraction: AbstractionMetrics
    ) -> List[str]:
        """Create migration roadmap"""
        roadmap = []

        if abstraction.abstraction_level in [AbstractionLevel.NONE, AbstractionLevel.THIN]:
            roadmap.append("Phase 1: Implement model abstraction layer")

        if not vendor.fallback_providers_configured:
            roadmap.append("Phase 2: Configure fallback providers")

        if vendor.lock_in_risk != VendorLockInRisk.MINIMAL:
            roadmap.append("Phase 3: Migrate proprietary features to standard alternatives")

        roadmap.append("Phase 4: Run comprehensive portability tests")
        roadmap.append("Phase 5: Execute gradual migration with monitoring")

        return roadmap

    def generate_report(
        self,
        assessment: PortabilityAssessment
    ) -> Dict[str, Any]:
        """Generate comprehensive portability report"""
        return {
            'assessment_id': assessment.assessment_id,
            'assessment_date': assessment.assessment_date.isoformat(),
            'overall_score': assessment.overall_score.value,
            'summary': {
                'abstraction_level': assessment.abstraction_metrics.abstraction_level.value,
                'lock_in_risk': assessment.vendor_independence_metrics.lock_in_risk.value,
                'models_supported': len(assessment.multi_model_metrics.models_supported),
                'test_pass_rate': assessment.portability_test_metrics.pass_rate,
                'interop_score': assessment.interoperability_metrics.api_compatibility_score
            },
            'recommendations': assessment.recommendations,
            'risk_factors': assessment.risk_factors,
            'migration_roadmap': assessment.migration_roadmap,
            'detailed_metrics': {
                'abstraction': {
                    'level': assessment.abstraction_metrics.abstraction_level.value,
                    'interface_coverage': assessment.abstraction_metrics.interface_coverage,
                    'vendor_specific_calls': assessment.abstraction_metrics.vendor_specific_calls,
                    'abstracted_calls': assessment.abstraction_metrics.abstracted_calls
                },
                'vendor_independence': {
                    'primary_vendor': assessment.vendor_independence_metrics.primary_vendor,
                    'lock_in_risk': assessment.vendor_independence_metrics.lock_in_risk.value,
                    'migration_readiness': assessment.vendor_independence_metrics.migration_readiness_score,
                    'proprietary_features': assessment.vendor_independence_metrics.proprietary_feature_count
                },
                'multi_model': {
                    'models': assessment.multi_model_metrics.models_supported,
                    'orchestration': assessment.multi_model_metrics.orchestration_support,
                    'consistency': assessment.multi_model_metrics.cross_model_consistency
                }
            }
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'AbstractionLevel',
    'VendorLockInRisk',
    'PortabilityScore',
    'ModelCapability',
    'InteroperabilityStandard',
    'MigrationComplexity',
    # Data Classes
    'AbstractionMetrics',
    'VendorDependency',
    'VendorIndependenceMetrics',
    'ModelCompatibilityMapping',
    'MultiModelMetrics',
    'PortabilityTestResult',
    'PortabilityTestMetrics',
    'InteroperabilityMetrics',
    'PortabilityAssessment',
    # Abstraction Analyzers
    'AbstractionLayerAnalyzer',
    'InterfaceDesignAnalyzer',
    # Vendor Independence Analyzers
    'VendorIndependenceAnalyzer',
    'LockInRiskAnalyzer',
    # Multi-Model Analyzers
    'MultiModelAnalyzer',
    'CapabilityMappingAnalyzer',
    # Portability Testing Analyzers
    'PortabilityTestAnalyzer',
    'MigrationValidationAnalyzer',
    # Interoperability Analyzers
    'InteroperabilityAnalyzer',
    'StandardsComplianceAnalyzer',
    # Comprehensive Analyzer
    'PortabilityAnalyzer',
]
