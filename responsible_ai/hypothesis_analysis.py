"""
Hypothesis in AI Analysis Module
=================================

Comprehensive analysis for AI hypothesis testing and validation.
Implements 20 analysis types for hypothesis-driven AI development.

Analysis Types:
1. Problem Framing Hypothesis
2. Data Availability Hypothesis
3. Label Validity Hypothesis
4. Feature Relevance Hypothesis
5. Independence & Leakage Hypothesis
6. Model Capacity Hypothesis
7. Algorithm Suitability Hypothesis
8. Optimization Hypothesis
9. Generalization Hypothesis
10. Class Imbalance Hypothesis
11. Metric Validity Hypothesis
12. Error Pattern Hypothesis
13. Robustness Hypothesis
14. Explainability Hypothesis
15. Causal Mechanism Hypothesis
16. Drift Hypothesis
17. Human-in-the-Loop Hypothesis
18. Safety & Risk Hypothesis
19. Deployment Hypothesis
20. Iteration & Learning Hypothesis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Hypothesis:
    """Represents a testable hypothesis."""
    hypothesis_id: str
    statement: str
    hypothesis_type: str
    assumptions: List[str] = field(default_factory=list)
    test_method: str = ""
    status: str = "untested"  # untested, testing, confirmed, rejected
    confidence: float = 0.0


@dataclass
class HypothesisTestResult:
    """Result of hypothesis testing."""
    hypothesis_id: str
    test_passed: bool = False
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ExperimentPlan:
    """Plan for testing a hypothesis."""
    experiment_id: str
    hypothesis_id: str
    methodology: str = ""
    metrics: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Problem Framing Hypothesis (Type 1)
# ============================================================================

class ProblemFramingAnalyzer:
    """Analyzes problem framing hypotheses."""

    def analyze_problem_framing(self,
                               problem_definition: Dict[str, Any],
                               validation_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if we're solving the right problem."""
        input_output_relationship = problem_definition.get('input_output_relationship', {})
        causal_assumption = problem_definition.get('is_causal', False)
        target_variable = problem_definition.get('target_variable', '')

        # Validate assumptions
        hypothesis_statement = f"Inputs {list(input_output_relationship.keys())} predict {target_variable}"

        # Check if relationship holds in validation data
        relationship_validated = False
        correlation_strength = 0.0

        if validation_data:
            # Simple correlation analysis
            correlations = []
            for feature in input_output_relationship.keys():
                feature_values = [d.get(feature, 0) for d in validation_data]
                target_values = [d.get(target_variable, 0) for d in validation_data]
                if feature_values and target_values:
                    corr = np.corrcoef(feature_values, target_values)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            if correlations:
                correlation_strength = np.mean(correlations)
                relationship_validated = correlation_strength > 0.3

        return {
            'hypothesis_statement': hypothesis_statement,
            'problem_hypothesis': {
                'is_causal': causal_assumption,
                'assumed_relationship': input_output_relationship,
                'target_variable': target_variable
            },
            'validation': {
                'relationship_validated': relationship_validated,
                'correlation_strength': float(correlation_strength),
                'samples_analyzed': len(validation_data) if validation_data else 0
            },
            'status': 'confirmed' if relationship_validated else 'needs_investigation'
        }


# ============================================================================
# Data Availability Hypothesis (Type 2)
# ============================================================================

class DataAvailabilityAnalyzer:
    """Analyzes data availability hypotheses."""

    def analyze_data_sufficiency(self,
                                dataset_stats: Dict[str, Any],
                                task_complexity: str = 'medium') -> Dict[str, Any]:
        """Analyze if available data is sufficient to learn the task."""
        sample_count = dataset_stats.get('sample_count', 0)
        feature_count = dataset_stats.get('feature_count', 0)
        missing_rate = dataset_stats.get('missing_rate', 0)
        signal_to_noise = dataset_stats.get('signal_to_noise_ratio', 1.0)

        # Heuristics for data sufficiency
        complexity_multipliers = {'simple': 10, 'medium': 100, 'complex': 1000}
        min_samples_per_feature = complexity_multipliers.get(task_complexity, 100)

        required_samples = feature_count * min_samples_per_feature
        sample_sufficiency = sample_count / required_samples if required_samples > 0 else 1

        # Data quality score
        quality_score = (1 - missing_rate) * min(1, signal_to_noise)

        # Overall sufficiency
        is_sufficient = sample_sufficiency >= 1.0 and quality_score >= 0.7

        return {
            'hypothesis': 'Available data is sufficient to learn the task',
            'data_sufficiency_analysis': {
                'sample_count': sample_count,
                'feature_count': feature_count,
                'required_samples_estimate': required_samples,
                'sample_sufficiency_ratio': float(sample_sufficiency),
                'data_quality_score': float(quality_score)
            },
            'assumptions': {
                'signal_to_noise_adequate': signal_to_noise >= 1.0,
                'feature_target_dependence': dataset_stats.get('feature_target_correlation', 0) > 0.1
            },
            'conclusion': {
                'is_sufficient': is_sufficient,
                'confidence': float(min(sample_sufficiency, quality_score))
            }
        }


# ============================================================================
# Label Validity Hypothesis (Type 3)
# ============================================================================

class LabelValidityAnalyzer:
    """Analyzes label validity hypotheses."""

    def analyze_label_validity(self,
                              label_analysis: Dict[str, Any],
                              annotation_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze if labels truly represent the concept."""
        label_type = label_analysis.get('label_type', 'direct')  # direct, proxy
        is_proxy = label_type == 'proxy'

        # Annotation quality metrics
        if annotation_metadata:
            inter_annotator_agreement = annotation_metadata.get('inter_annotator_agreement', 0)
            annotator_count = annotation_metadata.get('annotator_count', 1)
            annotation_guidelines = annotation_metadata.get('has_guidelines', False)
        else:
            inter_annotator_agreement = 0
            annotator_count = 1
            annotation_guidelines = False

        # Label distribution analysis
        label_distribution = label_analysis.get('distribution', {})
        label_noise_rate = label_analysis.get('estimated_noise_rate', 0)

        # Validity score
        validity_score = 0.0
        if not is_proxy:
            validity_score += 0.3
        validity_score += inter_annotator_agreement * 0.4
        validity_score += 0.2 if annotation_guidelines else 0
        validity_score += 0.1 * (1 - label_noise_rate)

        return {
            'hypothesis': 'Labels truly represent the target concept',
            'label_validity_report': {
                'label_type': label_type,
                'is_proxy_label': is_proxy,
                'proxy_vs_ground_truth': 'proxy' if is_proxy else 'ground_truth'
            },
            'annotation_reliability': {
                'inter_annotator_agreement': float(inter_annotator_agreement),
                'annotator_count': annotator_count,
                'has_annotation_guidelines': annotation_guidelines,
                'estimated_noise_rate': float(label_noise_rate)
            },
            'conclusion': {
                'validity_score': float(validity_score),
                'is_valid': validity_score >= 0.6,
                'concerns': self._identify_concerns(is_proxy, inter_annotator_agreement, label_noise_rate)
            }
        }

    def _identify_concerns(self, is_proxy: bool, agreement: float, noise: float) -> List[str]:
        concerns = []
        if is_proxy:
            concerns.append("Using proxy labels - may not fully represent target concept")
        if agreement < 0.7:
            concerns.append("Low inter-annotator agreement suggests label ambiguity")
        if noise > 0.1:
            concerns.append(f"Estimated {noise:.1%} label noise may impact learning")
        return concerns


# ============================================================================
# Feature Relevance Hypothesis (Type 4)
# ============================================================================

class FeatureRelevanceAnalyzer:
    """Analyzes feature relevance hypotheses."""

    def analyze_feature_relevance(self,
                                 feature_hypotheses: List[Dict[str, Any]],
                                 feature_importance_results: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze which features matter and why."""
        if not feature_hypotheses:
            return {'feature_hypothesis_log': [], 'validation_status': 'no_hypotheses'}

        validated_hypotheses = []

        for hypothesis in feature_hypotheses:
            feature_name = hypothesis.get('feature', '')
            expected_importance = hypothesis.get('expected_importance', 'high')
            expected_direction = hypothesis.get('expected_direction', 'positive')  # positive, negative

            # Validate against actual importance
            if feature_importance_results:
                actual_importance = feature_importance_results.get(feature_name, 0)

                importance_thresholds = {'high': 0.1, 'medium': 0.05, 'low': 0.01}
                threshold = importance_thresholds.get(expected_importance, 0.05)

                hypothesis_confirmed = actual_importance >= threshold

                validated_hypotheses.append({
                    'feature': feature_name,
                    'hypothesis': f"{feature_name} has {expected_importance} importance",
                    'expected_importance': expected_importance,
                    'actual_importance': float(actual_importance),
                    'confirmed': hypothesis_confirmed
                })

        confirmation_rate = sum(1 for h in validated_hypotheses if h['confirmed']) / len(validated_hypotheses) if validated_hypotheses else 0

        return {
            'feature_hypothesis_log': validated_hypotheses,
            'validation_summary': {
                'total_hypotheses': len(feature_hypotheses),
                'confirmed': sum(1 for h in validated_hypotheses if h['confirmed']),
                'rejected': sum(1 for h in validated_hypotheses if not h['confirmed']),
                'confirmation_rate': float(confirmation_rate)
            },
            'unexpected_findings': [h for h in validated_hypotheses if not h['confirmed']]
        }


# ============================================================================
# Independence & Leakage Hypothesis (Type 5)
# ============================================================================

class IndependenceLeakageAnalyzer:
    """Analyzes independence and data leakage hypotheses."""

    def analyze_independence(self,
                            data_splits: Dict[str, Any],
                            leakage_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if samples are independent and clean."""
        # Check for train-test leakage
        train_ids = set(data_splits.get('train_ids', []))
        test_ids = set(data_splits.get('test_ids', []))

        overlap = train_ids & test_ids
        has_id_leakage = len(overlap) > 0

        # Check temporal leakage
        is_temporal = data_splits.get('is_temporal_data', False)
        temporal_leakage = False
        if is_temporal:
            train_max_date = data_splits.get('train_max_date')
            test_min_date = data_splits.get('test_min_date')
            if train_max_date and test_min_date:
                temporal_leakage = train_max_date >= test_min_date

        # Check feature leakage
        feature_leakage = []
        if leakage_tests:
            for test in leakage_tests:
                if test.get('leakage_detected', False):
                    feature_leakage.append({
                        'feature': test.get('feature', ''),
                        'leakage_type': test.get('leakage_type', 'unknown')
                    })

        # IID assumption
        iid_assumption = data_splits.get('assumes_iid', True)
        iid_validated = not is_temporal or not temporal_leakage

        # Overall status
        no_leakage = not has_id_leakage and not temporal_leakage and len(feature_leakage) == 0

        return {
            'hypothesis': 'Samples are independent with no train-test leakage',
            'leakage_test_results': {
                'id_leakage': {
                    'detected': has_id_leakage,
                    'overlap_count': len(overlap)
                },
                'temporal_leakage': {
                    'detected': temporal_leakage,
                    'is_temporal_data': is_temporal
                },
                'feature_leakage': {
                    'detected': len(feature_leakage) > 0,
                    'leaky_features': feature_leakage
                }
            },
            'iid_assumption': {
                'assumed': iid_assumption,
                'validated': iid_validated
            },
            'conclusion': {
                'hypothesis_confirmed': no_leakage,
                'leakage_free': no_leakage,
                'requires_action': not no_leakage
            }
        }


# ============================================================================
# Model Capacity Hypothesis (Type 6)
# ============================================================================

class ModelCapacityAnalyzer:
    """Analyzes model capacity hypotheses."""

    def analyze_capacity(self,
                        model_metrics: Dict[str, Any],
                        learning_curves: List[Dict[str, float]] = None) -> Dict[str, Any]:
        """Analyze if model is expressive enough (but not too much)."""
        train_loss = model_metrics.get('train_loss', 0)
        val_loss = model_metrics.get('val_loss', 0)
        train_accuracy = model_metrics.get('train_accuracy', 0)
        val_accuracy = model_metrics.get('val_accuracy', 0)

        # Detect underfitting/overfitting
        loss_gap = val_loss - train_loss
        accuracy_gap = train_accuracy - val_accuracy

        if train_accuracy < 0.7 and val_accuracy < 0.7:
            capacity_status = 'underfitting'
            diagnosis = 'Model lacks capacity to learn the pattern'
        elif accuracy_gap > 0.15 or loss_gap > train_loss * 0.5:
            capacity_status = 'overfitting'
            diagnosis = 'Model has excess capacity, memorizing training data'
        else:
            capacity_status = 'appropriate'
            diagnosis = 'Model capacity appears balanced'

        # Analyze learning curves if available
        curve_analysis = {}
        if learning_curves:
            train_losses = [c.get('train_loss', 0) for c in learning_curves]
            val_losses = [c.get('val_loss', 0) for c in learning_curves]

            # Check for convergence
            if len(train_losses) > 5:
                recent_change = abs(train_losses[-1] - train_losses[-5]) / train_losses[-5] if train_losses[-5] > 0 else 0
                converged = recent_change < 0.05
            else:
                converged = False

            curve_analysis = {
                'converged': converged,
                'final_train_loss': float(train_losses[-1]) if train_losses else 0,
                'final_val_loss': float(val_losses[-1]) if val_losses else 0
            }

        return {
            'hypothesis': 'Model has appropriate capacity for the task',
            'capacity_validation': {
                'status': capacity_status,
                'diagnosis': diagnosis,
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy),
                'accuracy_gap': float(accuracy_gap),
                'loss_gap': float(loss_gap)
            },
            'learning_curve_analysis': curve_analysis,
            'recommendation': self._capacity_recommendation(capacity_status)
        }

    def _capacity_recommendation(self, status: str) -> str:
        recs = {
            'underfitting': 'Increase model complexity or train longer',
            'overfitting': 'Add regularization, reduce capacity, or get more data',
            'appropriate': 'Current capacity is suitable'
        }
        return recs.get(status, 'Unknown status')


# ============================================================================
# Algorithm Suitability Hypothesis (Type 7)
# ============================================================================

class AlgorithmSuitabilityAnalyzer:
    """Analyzes algorithm suitability hypotheses."""

    def analyze_algorithm_suitability(self,
                                     task_characteristics: Dict[str, Any],
                                     algorithm_choice: Dict[str, Any],
                                     benchmark_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if this model class is appropriate."""
        data_type = task_characteristics.get('data_type', 'tabular')  # tabular, sequence, image
        relationship_type = task_characteristics.get('relationship_type', 'linear')  # linear, nonlinear
        data_size = task_characteristics.get('data_size', 'medium')

        algorithm_type = algorithm_choice.get('type', 'unknown')
        algorithm_name = algorithm_choice.get('name', 'unknown')

        # Suitability heuristics
        suitability_matrix = {
            ('tabular', 'linear'): ['linear_regression', 'logistic_regression', 'svm'],
            ('tabular', 'nonlinear'): ['random_forest', 'gradient_boosting', 'neural_network'],
            ('sequence', 'nonlinear'): ['lstm', 'transformer', 'cnn_1d'],
            ('image', 'nonlinear'): ['cnn', 'vision_transformer', 'resnet']
        }

        recommended_types = suitability_matrix.get((data_type, relationship_type), [])
        is_suitable = algorithm_type.lower() in [r.lower() for r in recommended_types]

        # Compare with benchmarks
        benchmark_comparison = {}
        if benchmark_results:
            for result in benchmark_results:
                benchmark_comparison[result.get('algorithm', '')] = result.get('score', 0)

            current_score = benchmark_comparison.get(algorithm_name, 0)
            best_score = max(benchmark_comparison.values()) if benchmark_comparison else 0
            score_gap = best_score - current_score
        else:
            score_gap = 0

        return {
            'hypothesis': f'{algorithm_name} is appropriate for this task',
            'model_choice_justification': {
                'task_characteristics': {
                    'data_type': data_type,
                    'relationship_type': relationship_type,
                    'data_size': data_size
                },
                'chosen_algorithm': algorithm_choice,
                'recommended_algorithms': recommended_types,
                'is_recommended': is_suitable
            },
            'benchmark_comparison': benchmark_comparison,
            'conclusion': {
                'hypothesis_supported': is_suitable and score_gap < 0.05,
                'score_gap_from_best': float(score_gap)
            }
        }


# ============================================================================
# Optimization Hypothesis (Type 8)
# ============================================================================

class OptimizationHypothesisAnalyzer:
    """Analyzes optimization hypotheses."""

    def analyze_optimization(self,
                            training_logs: List[Dict[str, Any]],
                            optimizer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if training will converge meaningfully."""
        if not training_logs:
            return {'hypothesis': 'Training will converge', 'status': 'no_data'}

        losses = [l.get('loss', 0) for l in training_logs]

        # Check convergence
        if len(losses) > 10:
            early_loss = np.mean(losses[:5])
            late_loss = np.mean(losses[-5:])
            loss_reduction = (early_loss - late_loss) / early_loss if early_loss > 0 else 0

            # Check for instability
            loss_std = np.std(losses[-10:])
            is_stable = loss_std < late_loss * 0.1 if late_loss > 0 else True

            converged = loss_reduction > 0.5 and is_stable
        else:
            loss_reduction = 0
            is_stable = True
            converged = False

        # Learning rate analysis
        lr = optimizer_config.get('learning_rate', 0.001)
        lr_schedule = optimizer_config.get('lr_schedule', 'constant')

        # Detect issues
        issues = []
        if not is_stable:
            issues.append('Training instability detected - consider lower learning rate')
        if loss_reduction < 0.1 and len(losses) > 20:
            issues.append('Slow convergence - consider increasing learning rate or changing optimizer')

        return {
            'hypothesis': 'Training will converge meaningfully',
            'training_stability_analysis': {
                'converged': converged,
                'loss_reduction': float(loss_reduction),
                'is_stable': is_stable,
                'final_loss': float(losses[-1]) if losses else 0
            },
            'optimizer_config': {
                'learning_rate': lr,
                'schedule': lr_schedule,
                'optimizer': optimizer_config.get('optimizer', 'unknown')
            },
            'issues_detected': issues,
            'conclusion': {
                'hypothesis_confirmed': converged and is_stable,
                'requires_intervention': len(issues) > 0
            }
        }


# ============================================================================
# Generalization Hypothesis (Type 9)
# ============================================================================

class GeneralizationAnalyzer:
    """Analyzes generalization hypotheses."""

    def analyze_generalization(self,
                              in_distribution_results: Dict[str, Any],
                              ood_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if performance will hold on unseen data."""
        id_accuracy = in_distribution_results.get('accuracy', 0)
        id_loss = in_distribution_results.get('loss', 0)

        # OOD performance
        ood_analysis = []
        if ood_results:
            for ood in ood_results:
                ood_accuracy = ood.get('accuracy', 0)
                performance_drop = (id_accuracy - ood_accuracy) / id_accuracy if id_accuracy > 0 else 0

                ood_analysis.append({
                    'dataset': ood.get('name', 'unknown'),
                    'accuracy': float(ood_accuracy),
                    'performance_drop': float(performance_drop),
                    'maintains_performance': performance_drop < 0.2
                })

        # Generalization score
        if ood_analysis:
            avg_ood_accuracy = np.mean([o['accuracy'] for o in ood_analysis])
            generalization_score = avg_ood_accuracy / id_accuracy if id_accuracy > 0 else 0
        else:
            avg_ood_accuracy = id_accuracy
            generalization_score = 1.0  # Assume generalizes if no OOD data

        return {
            'hypothesis': 'Performance will hold on unseen data',
            'generalization_test_results': {
                'in_distribution': {
                    'accuracy': float(id_accuracy),
                    'loss': float(id_loss)
                },
                'out_of_distribution': ood_analysis,
                'generalization_score': float(generalization_score)
            },
            'conclusion': {
                'generalizes_well': generalization_score > 0.8,
                'ood_robust': all(o['maintains_performance'] for o in ood_analysis) if ood_analysis else True
            }
        }


# ============================================================================
# Class Imbalance Hypothesis (Type 10)
# ============================================================================

class ClassImbalanceAnalyzer:
    """Analyzes class imbalance hypotheses."""

    def analyze_imbalance_impact(self,
                                class_distribution: Dict[str, int],
                                per_class_metrics: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """Analyze if imbalance harms learning."""
        total_samples = sum(class_distribution.values())
        class_ratios = {k: v / total_samples for k, v in class_distribution.items()}

        # Calculate imbalance ratio
        max_ratio = max(class_ratios.values())
        min_ratio = min(class_ratios.values())
        imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')

        # Identify minority classes
        minority_threshold = 0.1  # Less than 10% of data
        minority_classes = [k for k, v in class_ratios.items() if v < minority_threshold]

        # Analyze per-class performance
        minority_performance = {}
        if per_class_metrics:
            for cls in minority_classes:
                if cls in per_class_metrics:
                    minority_performance[cls] = per_class_metrics[cls]

        # Determine impact
        if per_class_metrics and minority_classes:
            minority_accuracies = [per_class_metrics.get(c, {}).get('accuracy', 0) for c in minority_classes]
            majority_accuracies = [per_class_metrics.get(c, {}).get('accuracy', 0)
                                  for c in class_ratios.keys() if c not in minority_classes]

            avg_minority = np.mean(minority_accuracies) if minority_accuracies else 0
            avg_majority = np.mean(majority_accuracies) if majority_accuracies else 0
            performance_gap = avg_majority - avg_minority
            imbalance_harms = performance_gap > 0.1
        else:
            performance_gap = 0
            imbalance_harms = False

        return {
            'hypothesis': 'Class imbalance will not harm learning',
            'imbalance_impact_report': {
                'class_distribution': class_distribution,
                'class_ratios': {k: float(v) for k, v in class_ratios.items()},
                'imbalance_ratio': float(imbalance_ratio),
                'minority_classes': minority_classes
            },
            'separability_assumption': {
                'minority_class_performance': minority_performance,
                'performance_gap': float(performance_gap),
                'imbalance_harms_learning': imbalance_harms
            },
            'mitigation_needed': imbalance_ratio > 5 or imbalance_harms
        }


# ============================================================================
# Metric Validity Hypothesis (Type 11)
# ============================================================================

class MetricValidityAnalyzer:
    """Analyzes metric validity hypotheses."""

    def analyze_metric_validity(self,
                               metrics_used: List[Dict[str, Any]],
                               business_objectives: List[str]) -> Dict[str, Any]:
        """Analyze if metrics reflect real success."""
        metric_alignment = []

        for metric in metrics_used:
            metric_name = metric.get('name', '')
            metric_type = metric.get('type', '')  # accuracy, precision, recall, custom

            # Check alignment with business objectives
            aligned_objectives = []
            for obj in business_objectives:
                if self._metrics_aligns_with_objective(metric_name, obj):
                    aligned_objectives.append(obj)

            metric_alignment.append({
                'metric': metric_name,
                'type': metric_type,
                'aligned_objectives': aligned_objectives,
                'has_alignment': len(aligned_objectives) > 0
            })

        # Overall alignment score
        aligned_count = sum(1 for m in metric_alignment if m['has_alignment'])
        alignment_score = aligned_count / len(metric_alignment) if metric_alignment else 0

        return {
            'hypothesis': 'Metrics reflect real business success',
            'metric_justification': metric_alignment,
            'alignment_analysis': {
                'total_metrics': len(metrics_used),
                'aligned_metrics': aligned_count,
                'alignment_score': float(alignment_score)
            },
            'conclusion': {
                'metrics_valid': alignment_score >= 0.8,
                'gaps': [m['metric'] for m in metric_alignment if not m['has_alignment']]
            }
        }

    def _metrics_aligns_with_objective(self, metric: str, objective: str) -> bool:
        # Simple keyword matching for alignment
        alignment_map = {
            'accuracy': ['classification', 'prediction', 'correct'],
            'precision': ['false positive', 'spam', 'fraud'],
            'recall': ['miss', 'detection', 'catch'],
            'f1': ['balance', 'both'],
            'auc': ['ranking', 'discrimination'],
            'mse': ['regression', 'numeric', 'error'],
            'latency': ['speed', 'performance', 'fast']
        }

        keywords = alignment_map.get(metric.lower(), [])
        return any(kw in objective.lower() for kw in keywords)


# ============================================================================
# Error Pattern Hypothesis (Type 12)
# ============================================================================

class ErrorPatternAnalyzer:
    """Analyzes error pattern hypotheses."""

    def analyze_error_patterns(self,
                              predictions: List[Dict[str, Any]],
                              error_hypotheses: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze where and why the model will fail."""
        if not predictions:
            return {'error_hypothesis_validation': [], 'error_analysis': {}}

        # Identify errors
        errors = [p for p in predictions if p.get('is_error', False)]
        error_rate = len(errors) / len(predictions)

        # Categorize errors
        error_categories = defaultdict(list)
        for error in errors:
            category = error.get('error_category', 'unknown')
            error_categories[category].append(error)

        # Validate hypotheses
        hypothesis_validations = []
        if error_hypotheses:
            for hyp in error_hypotheses:
                expected_category = hyp.get('expected_failure_region', '')
                expected_rate = hyp.get('expected_rate', 0)

                actual_rate = len(error_categories.get(expected_category, [])) / len(predictions) if predictions else 0

                hypothesis_validations.append({
                    'hypothesis': hyp.get('statement', ''),
                    'expected_category': expected_category,
                    'expected_rate': float(expected_rate),
                    'actual_rate': float(actual_rate),
                    'validated': abs(actual_rate - expected_rate) < 0.1
                })

        return {
            'error_hypothesis_validation': hypothesis_validations,
            'error_analysis': {
                'total_predictions': len(predictions),
                'total_errors': len(errors),
                'error_rate': float(error_rate),
                'error_distribution': {k: len(v) for k, v in error_categories.items()},
                'hard_case_regions': list(error_categories.keys())
            }
        }


# ============================================================================
# Robustness Hypothesis (Type 13)
# ============================================================================

class RobustnessHypothesisAnalyzer:
    """Analyzes robustness hypotheses."""

    def analyze_robustness(self,
                          perturbation_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if small perturbations break behavior."""
        if not perturbation_tests:
            return {'robustness_score': 0, 'stress_test': {}}

        robust_count = 0
        perturbation_results = []

        for test in perturbation_tests:
            original_pred = test.get('original_prediction')
            perturbed_pred = test.get('perturbed_prediction')
            perturbation_size = test.get('perturbation_magnitude', 0)

            is_robust = original_pred == perturbed_pred
            if is_robust:
                robust_count += 1

            perturbation_results.append({
                'perturbation_type': test.get('perturbation_type', 'unknown'),
                'magnitude': float(perturbation_size),
                'maintained_prediction': is_robust
            })

        robustness_score = robust_count / len(perturbation_tests)

        return {
            'hypothesis': 'Small perturbations will not break predictions',
            'robustness_stress_test': {
                'total_tests': len(perturbation_tests),
                'robust_predictions': robust_count,
                'robustness_score': float(robustness_score),
                'perturbation_results': perturbation_results
            },
            'noise_tolerance_assumption': {
                'validated': robustness_score >= 0.9,
                'tolerance_level': 'high' if robustness_score >= 0.95 else 'medium' if robustness_score >= 0.8 else 'low'
            }
        }


# ============================================================================
# Explainability Hypothesis (Type 14)
# ============================================================================

class ExplainabilityHypothesisAnalyzer:
    """Analyzes explainability hypotheses."""

    def analyze_explainability(self,
                              feature_importance: Dict[str, float],
                              domain_expectations: Dict[str, str]) -> Dict[str, Any]:
        """Analyze if explanations match domain intuition."""
        # Compare top features with expected
        sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [f[0] for f in sorted_importance[:5]]

        # Check alignment with domain expectations
        expected_important = [f for f, exp in domain_expectations.items() if exp in ['high', 'critical']]
        expected_unimportant = [f for f, exp in domain_expectations.items() if exp in ['low', 'none']]

        # Calculate alignment
        correct_important = sum(1 for f in top_features if f in expected_important)
        incorrect_important = sum(1 for f in top_features if f in expected_unimportant)

        alignment_score = (correct_important - incorrect_important) / len(top_features) if top_features else 0
        alignment_score = max(0, (alignment_score + 1) / 2)  # Normalize to [0, 1]

        return {
            'hypothesis': 'Model explanations match domain intuition',
            'explanation_validation_report': {
                'top_model_features': top_features,
                'expected_important_features': expected_important,
                'alignment_score': float(alignment_score)
            },
            'domain_alignment': {
                'features_match_expectations': alignment_score >= 0.7,
                'unexpected_important': [f for f in top_features if f not in expected_important],
                'missing_expected': [f for f in expected_important if f not in top_features]
            }
        }


# ============================================================================
# Remaining Hypothesis Types (15-20)
# ============================================================================

class CausalMechanismAnalyzer:
    """Analyzes causal mechanism hypotheses (Type 15)."""

    def analyze_causal_hypothesis(self,
                                 causal_graph: Dict[str, List[str]],
                                 intervention_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if there's a causal story behind predictions."""
        # Analyze causal assumptions
        nodes = list(causal_graph.keys())
        edges = sum(len(children) for children in causal_graph.values())

        # Test interventions
        intervention_analysis = []
        if intervention_results:
            for result in intervention_results:
                intervention_analysis.append({
                    'intervention': result.get('intervention', ''),
                    'expected_effect': result.get('expected', ''),
                    'observed_effect': result.get('observed', ''),
                    'matches': result.get('expected') == result.get('observed')
                })

        return {
            'hypothesis': 'Predictions have a causal basis',
            'causal_hypothesis_test': {
                'causal_graph_nodes': len(nodes),
                'causal_edges': edges,
                'intervention_tests': intervention_analysis
            },
            'conclusion': {
                'causal_validated': all(i['matches'] for i in intervention_analysis) if intervention_analysis else False
            }
        }


class DriftHypothesisAnalyzer:
    """Analyzes drift hypotheses (Type 16)."""

    def analyze_drift_hypothesis(self,
                                temporal_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if data or concept will change over time."""
        if not temporal_metrics:
            return {'drift_risk_assessment': {}, 'stability_assumption': False}

        # Analyze temporal patterns
        performance_over_time = [m.get('performance', 0) for m in temporal_metrics]

        if len(performance_over_time) >= 2:
            trend = np.polyfit(range(len(performance_over_time)), performance_over_time, 1)[0]
            declining = trend < -0.01
        else:
            trend = 0
            declining = False

        return {
            'hypothesis': 'Data distribution will remain stable over time',
            'drift_risk_assessment': {
                'temporal_stability_assumed': True,
                'performance_trend': float(trend),
                'shows_decline': declining,
                'samples_analyzed': len(temporal_metrics)
            },
            'stability_assumption': {
                'validated': not declining,
                'monitoring_recommended': True
            }
        }


class HITLHypothesisAnalyzer:
    """Analyzes human-in-the-loop hypotheses (Type 17)."""

    def analyze_hitl_hypothesis(self,
                               hitl_interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if humans improve outcomes."""
        if not hitl_interventions:
            return {'hitl_impact_analysis': {}, 'override_effectiveness': 0}

        successful_overrides = sum(1 for i in hitl_interventions if i.get('improved_outcome', False))
        total_overrides = len(hitl_interventions)
        effectiveness = successful_overrides / total_overrides if total_overrides > 0 else 0

        return {
            'hypothesis': 'Human intervention improves outcomes',
            'hitl_impact_analysis': {
                'total_interventions': total_overrides,
                'successful_overrides': successful_overrides,
                'override_effectiveness': float(effectiveness)
            },
            'conclusion': {
                'hypothesis_confirmed': effectiveness > 0.6,
                'humans_add_value': effectiveness > 0.5
            }
        }


class SafetyRiskHypothesisAnalyzer:
    """Analyzes safety and risk hypotheses (Type 18)."""

    def analyze_safety_hypothesis(self,
                                 risk_scenarios: List[Dict[str, Any]],
                                 safety_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze what harm scenarios are plausible."""
        risk_register = []
        for scenario in risk_scenarios:
            risk_register.append({
                'scenario': scenario.get('description', ''),
                'likelihood': scenario.get('likelihood', 'unknown'),
                'impact': scenario.get('impact', 'unknown'),
                'mitigated': scenario.get('mitigated', False)
            })

        # Analyze safety tests
        safety_passed = 0
        if safety_tests:
            for test in safety_tests:
                if test.get('passed', False):
                    safety_passed += 1

        return {
            'hypothesis': 'Worst-case behaviors are contained',
            'risk_hypothesis_register': risk_register,
            'safety_test_results': {
                'total_tests': len(safety_tests) if safety_tests else 0,
                'passed': safety_passed,
                'pass_rate': safety_passed / len(safety_tests) if safety_tests else 0
            },
            'unmitigated_risks': [r for r in risk_register if not r['mitigated']]
        }


class DeploymentHypothesisAnalyzer:
    """Analyzes deployment hypotheses (Type 19)."""

    def analyze_deployment_hypothesis(self,
                                     training_env: Dict[str, Any],
                                     production_env: Dict[str, Any],
                                     parity_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if training behavior matches production."""
        # Compare environments
        env_differences = []
        for key in set(training_env.keys()) | set(production_env.keys()):
            train_val = training_env.get(key)
            prod_val = production_env.get(key)
            if train_val != prod_val:
                env_differences.append({
                    'parameter': key,
                    'training': train_val,
                    'production': prod_val
                })

        # Parity test results
        parity_passed = 0
        if parity_tests:
            for test in parity_tests:
                if test.get('passed', False):
                    parity_passed += 1

        return {
            'hypothesis': 'Training behavior will match production',
            'deployment_validation_report': {
                'environment_differences': env_differences,
                'has_parity': len(env_differences) == 0
            },
            'parity_tests': {
                'total': len(parity_tests) if parity_tests else 0,
                'passed': parity_passed,
                'pass_rate': parity_passed / len(parity_tests) if parity_tests else 0
            },
            'train_serve_parity': len(env_differences) == 0 and (parity_passed == len(parity_tests) if parity_tests else True)
        }


class IterationLearningAnalyzer:
    """Analyzes iteration and learning hypotheses (Type 20)."""

    def analyze_iteration_hypothesis(self,
                                    improvement_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what will improve performance next."""
        if not improvement_experiments:
            return {'next_iteration_plan': [], 'improvement_beliefs': {}}

        # Categorize experiments
        data_improvements = [e for e in improvement_experiments if e.get('category') == 'data']
        model_improvements = [e for e in improvement_experiments if e.get('category') == 'model']
        process_improvements = [e for e in improvement_experiments if e.get('category') == 'process']

        # Calculate success rates by category
        def success_rate(experiments):
            if not experiments:
                return 0
            successful = sum(1 for e in experiments if e.get('improved', False))
            return successful / len(experiments)

        return {
            'hypothesis': 'Targeted experiments will improve performance',
            'next_iteration_experiment_plan': improvement_experiments,
            'improvement_beliefs': {
                'data_improvements': {
                    'count': len(data_improvements),
                    'success_rate': float(success_rate(data_improvements))
                },
                'model_improvements': {
                    'count': len(model_improvements),
                    'success_rate': float(success_rate(model_improvements))
                },
                'process_improvements': {
                    'count': len(process_improvements),
                    'success_rate': float(success_rate(process_improvements))
                }
            },
            'recommended_focus': max(
                [('data', success_rate(data_improvements)),
                 ('model', success_rate(model_improvements)),
                 ('process', success_rate(process_improvements))],
                key=lambda x: x[1]
            )[0] if improvement_experiments else 'data'
        }


# ============================================================================
# Report Generator
# ============================================================================

class HypothesisReportGenerator:
    """Generates comprehensive hypothesis analysis reports."""

    def __init__(self):
        self.problem_analyzer = ProblemFramingAnalyzer()
        self.data_analyzer = DataAvailabilityAnalyzer()
        self.capacity_analyzer = ModelCapacityAnalyzer()
        self.generalization_analyzer = GeneralizationAnalyzer()
        self.robustness_analyzer = RobustnessHypothesisAnalyzer()

    def generate_full_report(self,
                            problem_definition: Dict[str, Any] = None,
                            dataset_stats: Dict[str, Any] = None,
                            model_metrics: Dict[str, Any] = None,
                            in_distribution_results: Dict[str, Any] = None,
                            perturbation_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive hypothesis analysis report."""
        report = {
            'report_type': 'hypothesis_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if problem_definition:
            report['problem_framing'] = self.problem_analyzer.analyze_problem_framing(problem_definition)

        if dataset_stats:
            report['data_sufficiency'] = self.data_analyzer.analyze_data_sufficiency(dataset_stats)

        if model_metrics:
            report['model_capacity'] = self.capacity_analyzer.analyze_capacity(model_metrics)

        if in_distribution_results:
            report['generalization'] = self.generalization_analyzer.analyze_generalization(in_distribution_results)

        if perturbation_tests:
            report['robustness'] = self.robustness_analyzer.analyze_robustness(perturbation_tests)

        # Count confirmed hypotheses
        confirmed = 0
        total = 0
        for key, value in report.items():
            if isinstance(value, dict):
                conclusion = value.get('conclusion', {})
                if isinstance(conclusion, dict):
                    if 'hypothesis_confirmed' in conclusion:
                        total += 1
                        if conclusion['hypothesis_confirmed']:
                            confirmed += 1

        report['hypothesis_summary'] = {
            'total_hypotheses_tested': total,
            'confirmed': confirmed,
            'confirmation_rate': confirmed / total if total > 0 else 0
        }

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
