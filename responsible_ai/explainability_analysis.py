"""
Explainable AI Analysis Module
===============================

Comprehensive analysis for AI explainability and interpretability.
Implements 20 analysis types for explainable AI governance.

Analysis Types:
1. Explainability Scope & Audience Definition
2. Explanation Purpose Analysis
3. Local (Instance-Level) Explainability
4. Global (Model-Level) Explainability
5. Feature Effect Analysis
6. Interaction Explainability Analysis
7. Counterfactual Explainability
8. Explanation Faithfulness Analysis
9. Stability & Robustness of Explanations
10. Consistency Across Methods
11. Explanation Bias & Fairness Analysis
12. Explainability for Imbalanced Data
13. Temporal Explainability (Time-Series)
14. Explainability for Deep Models
15. Explainability for LLMs / GenAI
16. Human Interpretability Evaluation
17. Explainability Limits & Failure Modes
18. Explainability Tooling & Reproducibility
19. Explainability in Decision Workflows
20. Explainability Governance & Accountability
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExplanationConfig:
    """Configuration for explanation generation."""
    explanation_type: str  # local, global
    audience: str  # user, expert, regulator
    method: str  # shap, lime, counterfactual, attention
    decision_criticality: str = "medium"


@dataclass
class LocalExplanation:
    """Local (instance-level) explanation."""
    instance_id: str
    prediction: Any
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    counterfactual: Optional[str] = None
    confidence: float = 0.0


@dataclass
class GlobalExplanation:
    """Global (model-level) explanation."""
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_behavior_summary: str = ""
    interaction_effects: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Scope & Audience Analyzer (Type 1)
# ============================================================================

class ExplainabilityScopeAnalyzer:
    """Analyzes explainability scope and audience definition."""

    def analyze_scope(self,
                     use_case_context: Dict[str, Any],
                     stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Define who needs explanations and why."""
        # Identify audiences
        audiences = []
        for stakeholder in stakeholders:
            audiences.append({
                'stakeholder': stakeholder.get('type', ''),
                'explanation_needs': stakeholder.get('needs', []),
                'technical_level': stakeholder.get('technical_level', 'low'),
                'required_detail': 'high' if stakeholder.get('type') == 'regulator' else 'medium'
            })

        # Determine explanation types needed
        needs_local = any(s.get('needs_instance_explanation', False) for s in stakeholders)
        needs_global = any(s.get('needs_model_understanding', False) for s in stakeholders)

        return {
            'explainability_scope_statement': {
                'decision_criticality': use_case_context.get('criticality', 'medium'),
                'audiences': audiences,
                'explanation_types_needed': {
                    'local': needs_local,
                    'global': needs_global
                }
            },
            'scope_requirements': {
                'regulatory_requirement': any(a['stakeholder'] == 'regulator' for a in audiences),
                'user_transparency': any(a['stakeholder'] == 'end_user' for a in audiences)
            }
        }


# ============================================================================
# Explanation Purpose Analyzer (Type 2)
# ============================================================================

class ExplanationPurposeAnalyzer:
    """Analyzes what explanations should enable."""

    def analyze_purpose(self,
                       explanation_requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what explanations should enable."""
        purposes = {
            'trust_calibration': False,
            'debugging': False,
            'compliance': False,
            'user_understanding': False,
            'contestability': False
        }

        for req in explanation_requirements:
            purpose = req.get('purpose', '')
            if purpose in purposes:
                purposes[purpose] = True

        return {
            'explanation_objective_mapping': purposes,
            'primary_purposes': [p for p, enabled in purposes.items() if enabled],
            'purpose_count': sum(purposes.values())
        }


# ============================================================================
# Local Explainability Analyzer (Type 3)
# ============================================================================

class LocalExplainabilityAnalyzer:
    """Analyzes local (instance-level) explainability."""

    def analyze_local_explanations(self,
                                  explanations: List[LocalExplanation]) -> Dict[str, Any]:
        """Analyze why model made specific decisions."""
        if not explanations:
            return {'instance_explanations': [], 'coverage': 0}

        # Analyze feature contributions
        all_contributions = defaultdict(list)
        for exp in explanations:
            for feature, contribution in exp.feature_contributions.items():
                all_contributions[feature].append(contribution)

        # Summary statistics
        feature_summary = {}
        for feature, contributions in all_contributions.items():
            feature_summary[feature] = {
                'mean_contribution': float(np.mean(contributions)),
                'std_contribution': float(np.std(contributions)),
                'max_contribution': float(np.max(contributions))
            }

        return {
            'instance_explanation_artifacts': [
                {
                    'instance_id': e.instance_id,
                    'top_features': sorted(
                        e.feature_contributions.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5],
                    'counterfactual': e.counterfactual
                } for e in explanations[:10]  # Sample
            ],
            'feature_contribution_summary': feature_summary,
            'explanation_coverage': len(explanations)
        }


# ============================================================================
# Global Explainability Analyzer (Type 4)
# ============================================================================

class GlobalExplainabilityAnalyzer:
    """Analyzes global (model-level) explainability."""

    def analyze_global_explanations(self,
                                   global_explanation: GlobalExplanation) -> Dict[str, Any]:
        """Analyze how model behaves overall."""
        # Sort features by importance
        sorted_features = sorted(
            global_explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return {
            'global_explanation_report': {
                'feature_importance_ranking': sorted_features,
                'top_features': [f[0] for f in sorted_features[:10]],
                'model_behavior_summary': global_explanation.model_behavior_summary
            },
            'importance_distribution': {
                'top_5_contribution': sum(abs(f[1]) for f in sorted_features[:5]) /
                                     sum(abs(f[1]) for f in sorted_features) if sorted_features else 0
            }
        }


# ============================================================================
# Feature Effect Analyzer (Type 5)
# ============================================================================

class FeatureEffectAnalyzer:
    """Analyzes how features influence predictions."""

    def analyze_feature_effects(self,
                               pdp_data: List[Dict[str, Any]],
                               ice_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze feature effects via PDP/ICE."""
        feature_effects = []

        for pdp in pdp_data:
            feature = pdp.get('feature', '')
            values = pdp.get('values', [])
            predictions = pdp.get('predictions', [])

            if values and predictions:
                # Calculate effect direction
                if len(predictions) >= 2:
                    direction = 'positive' if predictions[-1] > predictions[0] else 'negative'
                else:
                    direction = 'unknown'

                feature_effects.append({
                    'feature': feature,
                    'effect_direction': direction,
                    'value_range': [min(values), max(values)] if values else [0, 0],
                    'prediction_range': [min(predictions), max(predictions)] if predictions else [0, 0]
                })

        return {
            'feature_effect_visualizations': feature_effects,
            'pdp_count': len(pdp_data),
            'ice_available': ice_data is not None
        }


# ============================================================================
# Interaction Analyzer (Type 6)
# ============================================================================

class InteractionAnalyzer:
    """Analyzes feature interactions."""

    def analyze_interactions(self,
                            interaction_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if features interact non-linearly."""
        if not interaction_data:
            return {'interactions': [], 'significant_count': 0}

        significant_interactions = []
        for interaction in interaction_data:
            strength = interaction.get('interaction_strength', 0)
            if abs(strength) > 0.1:  # Threshold for significance
                significant_interactions.append({
                    'features': interaction.get('feature_pair', []),
                    'interaction_strength': float(strength),
                    'interaction_type': interaction.get('type', 'unknown')
                })

        return {
            'interaction_explanation_maps': significant_interactions,
            'total_pairs_analyzed': len(interaction_data),
            'significant_interactions': len(significant_interactions)
        }


# ============================================================================
# Counterfactual Analyzer (Type 7)
# ============================================================================

class CounterfactualAnalyzer:
    """Analyzes counterfactual explanations."""

    def analyze_counterfactuals(self,
                               counterfactuals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what minimal change flips decisions."""
        if not counterfactuals:
            return {'counterfactuals': [], 'actionability': 0}

        actionable = 0
        for cf in counterfactuals:
            changes = cf.get('changes', [])
            # Check actionability - changes should be feasible
            if all(c.get('actionable', False) for c in changes):
                actionable += 1

        actionability_rate = actionable / len(counterfactuals) if counterfactuals else 0

        return {
            'counterfactual_examples': counterfactuals[:10],
            'actionability_analysis': {
                'total_counterfactuals': len(counterfactuals),
                'actionable': actionable,
                'actionability_rate': float(actionability_rate)
            }
        }


# ============================================================================
# Faithfulness Analyzer (Type 8)
# ============================================================================

class FaithfulnessAnalyzer:
    """Analyzes if explanations reflect real model logic."""

    def analyze_faithfulness(self,
                            faithfulness_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze explanation faithfulness via perturbation tests."""
        if not faithfulness_tests:
            return {'faithfulness_score': 0, 'validation': {}}

        faithful = sum(1 for t in faithfulness_tests if t.get('faithful', False))
        faithfulness_score = faithful / len(faithfulness_tests)

        # Sanity check results
        passed_sanity = sum(1 for t in faithfulness_tests if t.get('sanity_check_passed', False))

        return {
            'faithfulness_validation_report': {
                'total_tests': len(faithfulness_tests),
                'faithful_explanations': faithful,
                'faithfulness_score': float(faithfulness_score),
                'sanity_checks_passed': passed_sanity
            }
        }


# ============================================================================
# Stability Analyzer (Type 9)
# ============================================================================

class ExplanationStabilityAnalyzer:
    """Analyzes stability and robustness of explanations."""

    def analyze_stability(self,
                         stability_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if explanations change under small input noise."""
        if not stability_tests:
            return {'stability_score': 0, 'analysis': {}}

        stable = sum(1 for t in stability_tests if t.get('stable', False))
        stability_score = stable / len(stability_tests)

        # Attribution variance
        variances = [t.get('attribution_variance', 0) for t in stability_tests]
        avg_variance = np.mean(variances) if variances else 0

        return {
            'explanation_stability_score': float(stability_score),
            'stability_analysis': {
                'total_tests': len(stability_tests),
                'stable_explanations': stable,
                'avg_attribution_variance': float(avg_variance)
            }
        }


# ============================================================================
# Method Consistency Analyzer (Type 10)
# ============================================================================

class MethodConsistencyAnalyzer:
    """Analyzes consistency across XAI methods."""

    def analyze_consistency(self,
                           method_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if different XAI methods agree."""
        if not method_comparisons:
            return {'consistency_score': 0, 'report': {}}

        consistent = sum(1 for c in method_comparisons if c.get('methods_agree', False))
        consistency_score = consistent / len(method_comparisons)

        # Method pairs analyzed
        method_pairs = list(set(
            tuple(sorted([c.get('method1', ''), c.get('method2', '')]))
            for c in method_comparisons
        ))

        return {
            'method_consistency_report': {
                'total_comparisons': len(method_comparisons),
                'consistent': consistent,
                'consistency_score': float(consistency_score),
                'method_pairs_compared': method_pairs
            }
        }


# ============================================================================
# Additional Analyzers (Types 11-20)
# ============================================================================

class ExplanationBiasAnalyzer:
    """Analyzes explanation bias and fairness (Type 11)."""

    def analyze_explanation_bias(self,
                                group_explanations: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze if explanations hide bias."""
        group_analysis = {}

        for group, explanations in group_explanations.items():
            if explanations:
                # Aggregate feature importance by group
                avg_importance = defaultdict(list)
                for exp in explanations:
                    for feature, importance in exp.get('feature_importance', {}).items():
                        avg_importance[feature].append(importance)

                group_analysis[group] = {
                    'avg_feature_importance': {
                        f: float(np.mean(v)) for f, v in avg_importance.items()
                    }
                }

        return {
            'bias_in_explanations_audit': group_analysis,
            'groups_analyzed': list(group_explanations.keys())
        }


class ImbalancedExplainabilityAnalyzer:
    """Analyzes explainability for imbalanced data (Type 12)."""

    def analyze_imbalanced_explanations(self,
                                       class_explanations: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze if minority explanations are meaningful."""
        class_analysis = {}

        for cls, explanations in class_explanations.items():
            if explanations:
                class_analysis[cls] = {
                    'explanation_count': len(explanations),
                    'avg_confidence': float(np.mean([e.get('confidence', 0) for e in explanations]))
                }

        return {
            'minority_class_explanation_report': class_analysis,
            'class_conditional_analysis': True
        }


class TemporalExplainabilityAnalyzer:
    """Analyzes temporal explainability (Type 13)."""

    def analyze_temporal_explanations(self,
                                     temporal_explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how reasoning evolved over time."""
        return {
            'temporal_explanation_plots': temporal_explanations,
            'time_aware_analysis': len(temporal_explanations) > 0,
            'windowed_explanations': True
        }


class DeepModelExplainabilityAnalyzer:
    """Analyzes explainability for deep models (Type 14)."""

    def analyze_deep_model_explanations(self,
                                       attention_maps: List[Dict[str, Any]] = None,
                                       gradient_explanations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if black-box models can be explained."""
        return {
            'deep_model_explanation_visuals': {
                'attention_maps_available': attention_maps is not None,
                'gradient_explanations_available': gradient_explanations is not None,
                'methods_used': [
                    m for m in ['attention', 'grad_cam', 'integrated_gradients']
                    if (attention_maps or gradient_explanations)
                ]
            }
        }


class LLMExplainabilityAnalyzer:
    """Analyzes explainability for LLMs/GenAI (Type 15)."""

    def analyze_llm_explanations(self,
                                token_attributions: List[Dict[str, Any]] = None,
                                rationale_traces: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze why text/code was generated."""
        return {
            'genai_explanation_artifacts': {
                'token_attribution_available': token_attributions is not None,
                'rationale_tracing': rationale_traces is not None,
                'retrieval_grounding': True  # If RAG is used
            }
        }


class HumanInterpretabilityAnalyzer:
    """Analyzes human interpretability evaluation (Type 16)."""

    def analyze_human_interpretability(self,
                                      user_studies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if humans understand explanations."""
        if not user_studies:
            return {'comprehension_score': 0, 'report': {}}

        comprehension_scores = [s.get('comprehension_score', 0) for s in user_studies]
        interpretation_errors = sum(s.get('interpretation_errors', 0) for s in user_studies)

        return {
            'human_comprehension_report': {
                'studies_conducted': len(user_studies),
                'avg_comprehension': float(np.mean(comprehension_scores)) if comprehension_scores else 0,
                'total_interpretation_errors': interpretation_errors
            }
        }


class ExplainabilityLimitsAnalyzer:
    """Analyzes explainability limits and failure modes (Type 17)."""

    def analyze_limits(self,
                      failure_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze where XAI breaks down."""
        failure_types = defaultdict(int)
        for case in failure_cases:
            failure_types[case.get('failure_type', 'unknown')] += 1

        return {
            'explainability_limitation_statement': {
                'failure_cases_identified': len(failure_cases),
                'failure_types': dict(failure_types),
                'ood_failures': sum(1 for c in failure_cases if c.get('is_ood', False)),
                'correlation_failures': sum(1 for c in failure_cases if c.get('highly_correlated', False))
            }
        }


class ExplainabilityToolingAnalyzer:
    """Analyzes explainability tooling and reproducibility (Type 18)."""

    def analyze_tooling(self,
                       tooling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if explanations can be reproduced."""
        return {
            'xai_reproducibility_log': {
                'deterministic': tooling_config.get('deterministic', False),
                'versioned': tooling_config.get('versioned', False),
                'tools_used': tooling_config.get('tools', []),
                'reproducibility_score': 1.0 if tooling_config.get('deterministic') and tooling_config.get('versioned') else 0.5
            }
        }


class WorkflowExplainabilityAnalyzer:
    """Analyzes explainability in decision workflows (Type 19)."""

    def analyze_workflow_usage(self,
                              workflow_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if explanations are actually used."""
        if not workflow_data:
            return {'usage_analysis': {}, 'integration': False}

        override_count = sum(1 for w in workflow_data if w.get('decision_overridden', False))
        explanation_viewed = sum(1 for w in workflow_data if w.get('explanation_viewed', False))

        return {
            'workflow_usage_analysis': {
                'total_decisions': len(workflow_data),
                'explanations_viewed': explanation_viewed,
                'decisions_overridden': override_count,
                'view_rate': explanation_viewed / len(workflow_data) if workflow_data else 0,
                'hitl_integration': True
            }
        }


class ExplainabilityGovernanceAnalyzer:
    """Analyzes explainability governance and accountability (Type 20)."""

    def analyze_governance(self,
                          governance_config: Dict[str, Any],
                          audit_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze who enforces explainability standards."""
        has_owner = governance_config.get('explainability_owner') is not None
        has_raci = governance_config.get('raci') is not None
        has_approval_gates = len(governance_config.get('approval_gates', [])) > 0

        governance_score = sum([has_owner, has_raci, has_approval_gates]) / 3

        # Audit compliance
        audit_compliance = 0
        if audit_records:
            audit_compliance = sum(1 for a in audit_records if a.get('compliant', False)) / len(audit_records)

        return {
            'explainable_ai_governance_policy': governance_config.get('policy', {}),
            'governance_elements': {
                'owner_assigned': has_owner,
                'raci_defined': has_raci,
                'approval_gates': has_approval_gates
            },
            'governance_score': float(governance_score),
            'audit_trail': {
                'total_audits': len(audit_records) if audit_records else 0,
                'compliance_rate': float(audit_compliance)
            }
        }


# ============================================================================
# Report Generator
# ============================================================================

class ExplainabilityReportGenerator:
    """Generates comprehensive explainability analysis reports."""

    def __init__(self):
        self.scope_analyzer = ExplainabilityScopeAnalyzer()
        self.local_analyzer = LocalExplainabilityAnalyzer()
        self.global_analyzer = GlobalExplainabilityAnalyzer()
        self.faithfulness_analyzer = FaithfulnessAnalyzer()
        self.stability_analyzer = ExplanationStabilityAnalyzer()
        self.consistency_analyzer = MethodConsistencyAnalyzer()
        self.governance_analyzer = ExplainabilityGovernanceAnalyzer()

    def generate_full_report(self,
                            use_case_context: Dict[str, Any] = None,
                            stakeholders: List[Dict[str, Any]] = None,
                            local_explanations: List[LocalExplanation] = None,
                            global_explanation: GlobalExplanation = None,
                            faithfulness_tests: List[Dict[str, Any]] = None,
                            stability_tests: List[Dict[str, Any]] = None,
                            governance_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive explainability report."""
        report = {
            'report_type': 'explainability_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if use_case_context and stakeholders:
            report['scope'] = self.scope_analyzer.analyze_scope(use_case_context, stakeholders)

        if local_explanations:
            report['local_explanations'] = self.local_analyzer.analyze_local_explanations(local_explanations)

        if global_explanation:
            report['global_explanations'] = self.global_analyzer.analyze_global_explanations(global_explanation)

        if faithfulness_tests:
            report['faithfulness'] = self.faithfulness_analyzer.analyze_faithfulness(faithfulness_tests)

        if stability_tests:
            report['stability'] = self.stability_analyzer.analyze_stability(stability_tests)

        if governance_config:
            report['governance'] = self.governance_analyzer.analyze_governance(governance_config)

        # Calculate overall explainability score
        scores = []
        if 'faithfulness' in report:
            scores.append(report['faithfulness'].get('faithfulness_validation_report', {}).get('faithfulness_score', 0))
        if 'stability' in report:
            scores.append(report['stability'].get('explanation_stability_score', 0))
        if 'governance' in report:
            scores.append(report['governance'].get('governance_score', 0))

        report['explainability_score'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
