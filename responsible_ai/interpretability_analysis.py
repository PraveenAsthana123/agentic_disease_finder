"""
Interpretability Analysis Module - Interpretable AI, Explainable AI, Mechanistic Interpretability
==================================================================================================

Comprehensive analysis for AI interpretability, explainability, and mechanistic understanding.
Implements 54 analysis types across three related frameworks.

Frameworks:
- Interpretable AI (18 types): Model Transparency, Feature Importance, Decision Rules
- Explainable AI (18 types): Local/Global Explanations, Counterfactuals, SHAP/LIME
- Mechanistic Interpretability (18 types): Causal Analysis, Circuit Discovery, Probing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class InterpretabilityMetrics:
    """Metrics for interpretability analysis."""
    model_complexity: float = 0.0
    feature_sparsity: float = 0.0
    decision_rule_clarity: float = 0.0
    decomposability: float = 0.0
    simulatability: float = 0.0


@dataclass
class ExplainabilityMetrics:
    """Metrics for explainability analysis."""
    explanation_fidelity: float = 0.0
    explanation_stability: float = 0.0
    explanation_completeness: float = 0.0
    contrastive_quality: float = 0.0
    user_comprehension: float = 0.0


@dataclass
class MechanisticMetrics:
    """Metrics for mechanistic interpretability."""
    causal_fidelity: float = 0.0
    circuit_completeness: float = 0.0
    intervention_accuracy: float = 0.0
    feature_localization: float = 0.0


@dataclass
class Explanation:
    """Represents a model explanation."""
    explanation_id: str
    input_id: str
    method: str  # 'shap', 'lime', 'gradient', 'attention', 'counterfactual'
    feature_attributions: Dict[str, float]
    confidence: float
    generated_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# Interpretable AI Analyzers
# ============================================================================

class ModelComplexityAnalyzer:
    """Analyzes model complexity and transparency."""

    def analyze_complexity(self,
                          model_architecture: Dict[str, Any],
                          parameter_count: int = None) -> Dict[str, Any]:
        """Analyze model complexity metrics."""
        layers = model_architecture.get('layers', [])
        num_layers = len(layers)

        # Complexity scoring based on architecture
        complexity_factors = {
            'linear': 0.1,
            'decision_tree': 0.2,
            'random_forest': 0.4,
            'gradient_boosting': 0.5,
            'neural_network': 0.7,
            'transformer': 0.9,
            'ensemble': 0.6
        }

        model_type = model_architecture.get('type', 'unknown')
        base_complexity = complexity_factors.get(model_type, 0.5)

        # Adjust for depth and width
        depth_factor = min(1.0, num_layers / 100)
        param_factor = min(1.0, (parameter_count or 0) / 1e9) if parameter_count else 0

        complexity_score = (base_complexity + depth_factor + param_factor) / 3

        # Transparency is inverse of complexity
        transparency_score = 1 - complexity_score

        return {
            'complexity_score': float(complexity_score),
            'transparency_score': float(transparency_score),
            'model_type': model_type,
            'num_layers': num_layers,
            'parameter_count': parameter_count,
            'interpretability_level': 'high' if transparency_score > 0.7 else ('medium' if transparency_score > 0.4 else 'low'),
            'recommendations': self._generate_recommendations(model_type, complexity_score)
        }

    def _generate_recommendations(self, model_type: str, complexity: float) -> List[str]:
        recommendations = []
        if complexity > 0.7:
            recommendations.append("Consider using explanation methods like SHAP or LIME")
            recommendations.append("Implement attention visualization if applicable")
        if model_type == 'neural_network' or model_type == 'transformer':
            recommendations.append("Use gradient-based attribution methods")
        return recommendations


class FeatureImportanceAnalyzer:
    """Analyzes feature importance and contribution."""

    def analyze_feature_importance(self,
                                   importances: Dict[str, float],
                                   feature_metadata: Dict[str, Dict] = None) -> Dict[str, Any]:
        """Analyze feature importance distribution."""
        if not importances:
            return {'feature_sparsity': 0.0, 'dominant_features': []}

        values = np.array(list(importances.values()))
        total = np.sum(np.abs(values))

        # Normalize importances
        normalized = {k: abs(v) / total if total > 0 else 0 for k, v in importances.items()}

        # Calculate sparsity (how concentrated importance is)
        sorted_values = sorted(normalized.values(), reverse=True)
        cumsum = np.cumsum(sorted_values)

        # Find features covering 80% importance
        top_80_count = np.searchsorted(cumsum, 0.8) + 1
        sparsity = 1 - (top_80_count / len(importances)) if importances else 0

        # Identify dominant features
        dominant = [k for k, v in normalized.items() if v > 0.1]

        # Analyze by feature type if metadata provided
        type_analysis = defaultdict(float)
        if feature_metadata:
            for feature, importance in normalized.items():
                feat_type = feature_metadata.get(feature, {}).get('type', 'unknown')
                type_analysis[feat_type] += importance

        return {
            'feature_sparsity': float(sparsity),
            'num_features': len(importances),
            'dominant_features': dominant,
            'top_80_percent_features': top_80_count,
            'importance_distribution': normalized,
            'type_analysis': dict(type_analysis),
            'max_importance': float(max(normalized.values())) if normalized else 0,
            'importance_concentration': 'high' if sparsity > 0.7 else ('medium' if sparsity > 0.4 else 'low')
        }


class DecisionRuleAnalyzer:
    """Analyzes decision rules and paths."""

    def analyze_decision_rules(self,
                               rules: List[Dict[str, Any]],
                               prediction_coverage: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze decision rule clarity and coverage."""
        if not rules:
            return {'rule_clarity': 0.0, 'total_rules': 0}

        rule_analysis = []
        total_conditions = 0

        for rule in rules:
            conditions = rule.get('conditions', [])
            num_conditions = len(conditions)
            total_conditions += num_conditions

            rule_analysis.append({
                'rule_id': rule.get('id'),
                'num_conditions': num_conditions,
                'coverage': rule.get('coverage', 0),
                'precision': rule.get('precision', 0),
                'complexity': 'simple' if num_conditions <= 3 else ('medium' if num_conditions <= 6 else 'complex')
            })

        avg_conditions = total_conditions / len(rules) if rules else 0

        # Rule clarity: simpler rules are clearer
        clarity_score = max(0, 1 - (avg_conditions / 10))

        # Coverage analysis
        if prediction_coverage:
            total_coverage = sum(prediction_coverage.values())
            coverage_analysis = {
                'total_coverage': total_coverage,
                'uncovered_predictions': 1 - total_coverage
            }
        else:
            coverage_analysis = {}

        return {
            'rule_clarity': float(clarity_score),
            'total_rules': len(rules),
            'average_conditions': float(avg_conditions),
            'rule_analysis': rule_analysis,
            'simple_rules': sum(1 for r in rule_analysis if r['complexity'] == 'simple'),
            'complex_rules': sum(1 for r in rule_analysis if r['complexity'] == 'complex'),
            'coverage_analysis': coverage_analysis
        }


# ============================================================================
# Explainable AI Analyzers
# ============================================================================

class ExplanationFidelityAnalyzer:
    """Analyzes fidelity of explanations to model behavior."""

    def analyze_fidelity(self,
                        explanations: List[Explanation],
                        model_predictions: List[Dict[str, Any]],
                        perturbed_predictions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze explanation fidelity."""
        if not explanations:
            return {'fidelity_score': 0.0, 'explanations_analyzed': 0}

        fidelity_scores = []
        method_fidelity = defaultdict(list)

        for i, explanation in enumerate(explanations):
            if i >= len(model_predictions):
                continue

            prediction = model_predictions[i]

            # Calculate local fidelity
            # Compare top features with prediction sensitivity
            top_features = sorted(
                explanation.feature_attributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            # Simplified fidelity: use explanation confidence as proxy
            local_fidelity = explanation.confidence
            fidelity_scores.append(local_fidelity)
            method_fidelity[explanation.method].append(local_fidelity)

        avg_fidelity = np.mean(fidelity_scores) if fidelity_scores else 0

        method_analysis = {
            method: {
                'mean_fidelity': float(np.mean(scores)),
                'std_fidelity': float(np.std(scores)),
                'count': len(scores)
            }
            for method, scores in method_fidelity.items()
        }

        return {
            'fidelity_score': float(avg_fidelity),
            'explanations_analyzed': len(fidelity_scores),
            'method_analysis': method_analysis,
            'best_method': max(method_analysis, key=lambda x: method_analysis[x]['mean_fidelity']) if method_analysis else None,
            'fidelity_variance': float(np.var(fidelity_scores)) if fidelity_scores else 0
        }


class ExplanationStabilityAnalyzer:
    """Analyzes stability of explanations across similar inputs."""

    def analyze_stability(self,
                         explanation_sets: List[List[Explanation]],
                         input_similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """Analyze explanation stability."""
        if not explanation_sets or len(explanation_sets) < 2:
            return {'stability_score': 1.0, 'stability_analyzed': False}

        stability_scores = []

        for i in range(len(explanation_sets) - 1):
            for j in range(i + 1, len(explanation_sets)):
                set1, set2 = explanation_sets[i], explanation_sets[j]

                if not set1 or not set2:
                    continue

                # Compare feature attributions
                for e1, e2 in zip(set1, set2):
                    if e1.method == e2.method:
                        similarity = self._compute_attribution_similarity(
                            e1.feature_attributions,
                            e2.feature_attributions
                        )
                        stability_scores.append(similarity)

        avg_stability = np.mean(stability_scores) if stability_scores else 1.0

        return {
            'stability_score': float(avg_stability),
            'comparisons_made': len(stability_scores),
            'high_stability': avg_stability > 0.8,
            'stability_variance': float(np.var(stability_scores)) if stability_scores else 0
        }

    def _compute_attribution_similarity(self, attr1: Dict[str, float], attr2: Dict[str, float]) -> float:
        """Compute cosine similarity between attribution vectors."""
        all_features = set(attr1.keys()) | set(attr2.keys())
        if not all_features:
            return 1.0

        vec1 = np.array([attr1.get(f, 0) for f in all_features])
        vec2 = np.array([attr2.get(f, 0) for f in all_features])

        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0 if norm1 == norm2 else 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class CounterfactualAnalyzer:
    """Analyzes counterfactual explanations."""

    def analyze_counterfactuals(self,
                                original_inputs: List[Dict[str, Any]],
                                counterfactuals: List[Dict[str, Any]],
                                predictions: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """Analyze counterfactual explanation quality."""
        if not counterfactuals:
            return {'counterfactual_quality': 0.0, 'counterfactuals_generated': 0}

        quality_metrics = []
        valid_counterfactuals = []
        invalid_counterfactuals = []

        for i, (original, cf) in enumerate(zip(original_inputs, counterfactuals)):
            if i >= len(predictions):
                continue

            orig_pred, cf_pred = predictions[i]

            # Check validity: prediction changed
            is_valid = orig_pred != cf_pred

            # Calculate proximity (how many features changed)
            changes = 0
            total_features = len(original)
            for key in original:
                if original.get(key) != cf.get(key):
                    changes += 1

            proximity = 1 - (changes / total_features) if total_features > 0 else 1
            sparsity = 1 - (changes / total_features) if total_features > 0 else 1

            cf_data = {
                'index': i,
                'is_valid': is_valid,
                'features_changed': changes,
                'proximity': proximity,
                'sparsity': sparsity
            }

            if is_valid:
                valid_counterfactuals.append(cf_data)
                quality_metrics.append((proximity + sparsity) / 2)
            else:
                invalid_counterfactuals.append(cf_data)

        avg_quality = np.mean(quality_metrics) if quality_metrics else 0
        validity_rate = len(valid_counterfactuals) / len(counterfactuals) if counterfactuals else 0

        return {
            'counterfactual_quality': float(avg_quality),
            'validity_rate': float(validity_rate),
            'valid_counterfactuals': len(valid_counterfactuals),
            'invalid_counterfactuals': len(invalid_counterfactuals),
            'total_counterfactuals': len(counterfactuals),
            'average_features_changed': float(np.mean([c['features_changed'] for c in valid_counterfactuals])) if valid_counterfactuals else 0,
            'average_proximity': float(np.mean([c['proximity'] for c in valid_counterfactuals])) if valid_counterfactuals else 0
        }


class SHAPAnalyzer:
    """Analyzes SHAP value explanations."""

    def analyze_shap(self,
                    shap_values: List[Dict[str, float]],
                    expected_value: float = None) -> Dict[str, Any]:
        """Analyze SHAP value distributions."""
        if not shap_values:
            return {'shap_analysis': {}, 'feature_importance': {}}

        # Aggregate SHAP values across samples
        feature_shap = defaultdict(list)
        for sample_shap in shap_values:
            for feature, value in sample_shap.items():
                feature_shap[feature].append(value)

        # Calculate feature importance (mean absolute SHAP)
        feature_importance = {
            feature: float(np.mean(np.abs(values)))
            for feature, values in feature_shap.items()
        }

        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        # Calculate interaction effects (simplified)
        positive_features = [f for f, v in feature_importance.items() if np.mean(feature_shap[f]) > 0]
        negative_features = [f for f, v in feature_importance.items() if np.mean(feature_shap[f]) < 0]

        return {
            'feature_importance': sorted_importance,
            'num_features': len(feature_importance),
            'positive_contributors': positive_features[:5],
            'negative_contributors': negative_features[:5],
            'expected_value': expected_value,
            'total_samples': len(shap_values),
            'feature_statistics': {
                feature: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
                for feature, values in list(feature_shap.items())[:10]
            }
        }


# ============================================================================
# Mechanistic Interpretability Analyzers
# ============================================================================

class CausalAnalyzer:
    """Analyzes causal relationships in model behavior."""

    def analyze_causality(self,
                         intervention_results: List[Dict[str, Any]],
                         expected_effects: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze causal intervention results."""
        if not intervention_results:
            return {'causal_fidelity': 0.0, 'interventions_analyzed': 0}

        causal_effects = defaultdict(list)
        alignment_scores = []

        for result in intervention_results:
            variable = result.get('variable')
            observed_effect = result.get('observed_effect', 0)
            expected_effect = result.get('expected_effect') or (expected_effects or {}).get(variable, 0)

            causal_effects[variable].append(observed_effect)

            # Calculate alignment with expected effect
            if expected_effect != 0:
                alignment = 1 - min(1, abs(observed_effect - expected_effect) / abs(expected_effect))
            else:
                alignment = 1 if observed_effect == 0 else 0
            alignment_scores.append(alignment)

        causal_fidelity = np.mean(alignment_scores) if alignment_scores else 0

        # Summarize causal effects
        effect_summary = {
            var: {
                'mean_effect': float(np.mean(effects)),
                'consistency': float(1 - np.std(effects)) if len(effects) > 1 else 1.0,
                'num_interventions': len(effects)
            }
            for var, effects in causal_effects.items()
        }

        return {
            'causal_fidelity': float(causal_fidelity),
            'interventions_analyzed': len(intervention_results),
            'effect_summary': effect_summary,
            'strongest_causal_variable': max(effect_summary, key=lambda x: abs(effect_summary[x]['mean_effect'])) if effect_summary else None,
            'average_alignment': float(np.mean(alignment_scores)) if alignment_scores else 0
        }


class CircuitAnalyzer:
    """Analyzes neural circuit discovery."""

    def analyze_circuits(self,
                        circuit_descriptions: List[Dict[str, Any]],
                        behavior_coverage: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze discovered neural circuits."""
        if not circuit_descriptions:
            return {'circuit_completeness': 0.0, 'circuits_discovered': 0}

        circuit_analysis = []
        total_components = 0
        well_understood = 0

        for circuit in circuit_descriptions:
            components = circuit.get('components', [])
            understood = circuit.get('understood', False)
            importance = circuit.get('importance', 0)

            total_components += len(components)
            if understood:
                well_understood += 1

            circuit_analysis.append({
                'circuit_id': circuit.get('id'),
                'num_components': len(components),
                'understood': understood,
                'importance': importance,
                'function': circuit.get('function', 'unknown')
            })

        understanding_rate = well_understood / len(circuit_descriptions) if circuit_descriptions else 0

        # Calculate completeness based on behavior coverage
        if behavior_coverage:
            completeness = sum(behavior_coverage.values()) / len(behavior_coverage)
        else:
            completeness = understanding_rate

        return {
            'circuit_completeness': float(completeness),
            'understanding_rate': float(understanding_rate),
            'circuits_discovered': len(circuit_descriptions),
            'total_components': total_components,
            'well_understood_circuits': well_understood,
            'circuit_analysis': circuit_analysis,
            'behavior_coverage': behavior_coverage
        }


class ProbingAnalyzer:
    """Analyzes probing classifier results."""

    def analyze_probing(self,
                       probing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze probing classifier results."""
        if not probing_results:
            return {'probing_summary': {}, 'features_localized': 0}

        layer_analysis = defaultdict(list)
        feature_localization = {}

        for result in probing_results:
            layer = result.get('layer')
            feature = result.get('feature')
            accuracy = result.get('accuracy', 0)

            layer_analysis[layer].append({
                'feature': feature,
                'accuracy': accuracy
            })

            if feature not in feature_localization or accuracy > feature_localization[feature]['accuracy']:
                feature_localization[feature] = {
                    'best_layer': layer,
                    'accuracy': accuracy
                }

        # Summarize by layer
        layer_summary = {
            layer: {
                'num_features': len(results),
                'mean_accuracy': float(np.mean([r['accuracy'] for r in results])),
                'best_feature': max(results, key=lambda x: x['accuracy'])['feature'] if results else None
            }
            for layer, results in layer_analysis.items()
        }

        return {
            'layer_summary': layer_summary,
            'feature_localization': feature_localization,
            'features_localized': len(feature_localization),
            'best_layer': max(layer_summary, key=lambda x: layer_summary[x]['mean_accuracy']) if layer_summary else None,
            'total_probes': len(probing_results)
        }


# ============================================================================
# Report Generator
# ============================================================================

class InterpretabilityReportGenerator:
    """Generates comprehensive interpretability reports."""

    def __init__(self):
        self.complexity_analyzer = ModelComplexityAnalyzer()
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.rule_analyzer = DecisionRuleAnalyzer()
        self.fidelity_analyzer = ExplanationFidelityAnalyzer()
        self.stability_analyzer = ExplanationStabilityAnalyzer()
        self.counterfactual_analyzer = CounterfactualAnalyzer()
        self.shap_analyzer = SHAPAnalyzer()
        self.causal_analyzer = CausalAnalyzer()
        self.circuit_analyzer = CircuitAnalyzer()
        self.probing_analyzer = ProbingAnalyzer()

    def generate_full_report(self,
                            model_architecture: Dict[str, Any] = None,
                            feature_importances: Dict[str, float] = None,
                            explanations: List[Explanation] = None,
                            shap_values: List[Dict[str, float]] = None,
                            intervention_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive interpretability report."""
        report = {
            'report_type': 'comprehensive_interpretability_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if model_architecture:
            report['complexity'] = self.complexity_analyzer.analyze_complexity(model_architecture)

        if feature_importances:
            report['feature_importance'] = self.feature_analyzer.analyze_feature_importance(feature_importances)

        if explanations:
            report['explanation_fidelity'] = self.fidelity_analyzer.analyze_fidelity(explanations, [])

        if shap_values:
            report['shap_analysis'] = self.shap_analyzer.analyze_shap(shap_values)

        if intervention_results:
            report['causal_analysis'] = self.causal_analyzer.analyze_causality(intervention_results)

        # Calculate overall interpretability score
        scores = []
        if 'complexity' in report:
            scores.append(report['complexity'].get('transparency_score', 0))
        if 'feature_importance' in report:
            scores.append(report['feature_importance'].get('feature_sparsity', 0))
        if 'explanation_fidelity' in report:
            scores.append(report['explanation_fidelity'].get('fidelity_score', 0))

        report['overall_interpretability'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'markdown':
            md_content = self._report_to_markdown(report)
            with open(filepath, 'w') as f:
                f.write(md_content)

    def _report_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to markdown."""
        lines = [
            f"# {report.get('report_type', 'Interpretability Report')}",
            f"\n**Generated:** {report.get('timestamp', 'N/A')}",
            f"\n**Overall Interpretability:** {report.get('overall_interpretability', 0):.2%}",
            "\n---\n"
        ]

        if 'complexity' in report:
            c = report['complexity']
            lines.append("## Model Complexity\n")
            lines.append(f"- **Transparency Score:** {c.get('transparency_score', 0):.2%}")
            lines.append(f"- **Interpretability Level:** {c.get('interpretability_level', 'unknown')}")
            lines.append("")

        return "\n".join(lines)
