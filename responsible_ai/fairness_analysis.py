"""
Fairness Analysis Module - Fairness AI, Ethical AI, Social AI
==============================================================

Comprehensive analysis for AI fairness, ethics, and social impact.
Implements 56 analysis types across three related frameworks.

Frameworks:
- Fairness AI (18 types): Demographic Parity, Equal Opportunity, Calibration, Disparate Impact
- Ethical AI (20 types): Value Alignment, Moral Reasoning, Harm Prevention, Rights Protection
- Social AI (18 types): Social Impact, Community Effects, Accessibility, Inclusion
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FairnessMetrics:
    """Metrics for fairness analysis."""
    demographic_parity: float = 0.0
    equal_opportunity: float = 0.0
    equalized_odds: float = 0.0
    calibration_score: float = 0.0
    disparate_impact_ratio: float = 0.0
    individual_fairness: float = 0.0
    group_fairness: float = 0.0
    counterfactual_fairness: float = 0.0


@dataclass
class EthicalMetrics:
    """Metrics for ethical analysis."""
    value_alignment: float = 0.0
    harm_score: float = 0.0
    rights_compliance: float = 0.0
    autonomy_respect: float = 0.0
    beneficence_score: float = 0.0
    justice_score: float = 0.0
    transparency_score: float = 0.0
    consent_compliance: float = 0.0


@dataclass
class SocialMetrics:
    """Metrics for social impact analysis."""
    social_benefit: float = 0.0
    community_impact: float = 0.0
    accessibility_score: float = 0.0
    inclusion_index: float = 0.0
    equity_score: float = 0.0
    stakeholder_satisfaction: float = 0.0
    digital_divide_impact: float = 0.0


@dataclass
class ProtectedGroup:
    """Represents a protected demographic group."""
    group_id: str
    name: str
    attribute: str  # e.g., 'gender', 'race', 'age'
    size: int
    prevalence: float = 0.0


@dataclass
class EthicalPrinciple:
    """Represents an ethical principle for evaluation."""
    principle_id: str
    name: str
    description: str
    category: str  # 'autonomy', 'beneficence', 'justice', 'non-maleficence'
    weight: float = 1.0


# ============================================================================
# Fairness AI Analyzers (18 Analysis Types)
# ============================================================================

class DemographicParityAnalyzer:
    """Analyzes demographic parity across groups."""

    def analyze_demographic_parity(self,
                                   predictions: List[int],
                                   group_labels: Dict[str, List[int]],
                                   positive_label: int = 1) -> Dict[str, Any]:
        """Analyze demographic parity for predictions across groups."""
        if not predictions or not group_labels:
            return {'demographic_parity': 0.0, 'group_rates': {}}

        predictions = np.array(predictions)
        positive_rates = {}

        for group_name, labels in group_labels.items():
            labels = np.array(labels)
            group_mask = labels == 1
            group_predictions = predictions[group_mask]

            if len(group_predictions) > 0:
                positive_rate = np.mean(group_predictions == positive_label)
                positive_rates[group_name] = float(positive_rate)

        if len(positive_rates) < 2:
            return {
                'demographic_parity': 1.0,
                'group_rates': positive_rates,
                'max_disparity': 0.0
            }

        # Calculate disparity
        rates = list(positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)
        disparity = max_rate - min_rate

        # Demographic parity score (1 = perfect parity)
        parity_score = 1 - disparity

        return {
            'demographic_parity': float(parity_score),
            'group_rates': positive_rates,
            'max_disparity': float(disparity),
            'max_rate_group': max(positive_rates, key=positive_rates.get),
            'min_rate_group': min(positive_rates, key=positive_rates.get),
            'parity_achieved': disparity < 0.1
        }


class EqualOpportunityAnalyzer:
    """Analyzes equal opportunity (true positive rate parity)."""

    def analyze_equal_opportunity(self,
                                  predictions: List[int],
                                  labels: List[int],
                                  group_labels: Dict[str, List[int]],
                                  positive_label: int = 1) -> Dict[str, Any]:
        """Analyze equal opportunity across groups."""
        if not predictions or not labels or not group_labels:
            return {'equal_opportunity': 0.0, 'tpr_by_group': {}}

        predictions = np.array(predictions)
        labels = np.array(labels)

        tpr_by_group = {}

        for group_name, g_labels in group_labels.items():
            g_labels = np.array(g_labels)
            group_mask = g_labels == 1

            group_predictions = predictions[group_mask]
            group_true_labels = labels[group_mask]

            # Calculate TPR for this group
            positive_mask = group_true_labels == positive_label
            if np.sum(positive_mask) > 0:
                tpr = np.mean(group_predictions[positive_mask] == positive_label)
                tpr_by_group[group_name] = float(tpr)

        if len(tpr_by_group) < 2:
            return {
                'equal_opportunity': 1.0,
                'tpr_by_group': tpr_by_group,
                'tpr_disparity': 0.0
            }

        # Calculate disparity
        tprs = list(tpr_by_group.values())
        disparity = max(tprs) - min(tprs)
        equal_opportunity_score = 1 - disparity

        return {
            'equal_opportunity': float(equal_opportunity_score),
            'tpr_by_group': tpr_by_group,
            'tpr_disparity': float(disparity),
            'highest_tpr_group': max(tpr_by_group, key=tpr_by_group.get),
            'lowest_tpr_group': min(tpr_by_group, key=tpr_by_group.get),
            'opportunity_equal': disparity < 0.1
        }


class EqualizedOddsAnalyzer:
    """Analyzes equalized odds (TPR and FPR parity)."""

    def analyze_equalized_odds(self,
                               predictions: List[int],
                               labels: List[int],
                               group_labels: Dict[str, List[int]],
                               positive_label: int = 1) -> Dict[str, Any]:
        """Analyze equalized odds across groups."""
        if not predictions or not labels or not group_labels:
            return {'equalized_odds': 0.0, 'metrics_by_group': {}}

        predictions = np.array(predictions)
        labels = np.array(labels)

        metrics_by_group = {}

        for group_name, g_labels in group_labels.items():
            g_labels = np.array(g_labels)
            group_mask = g_labels == 1

            group_predictions = predictions[group_mask]
            group_true_labels = labels[group_mask]

            # Calculate TPR
            positive_mask = group_true_labels == positive_label
            if np.sum(positive_mask) > 0:
                tpr = np.mean(group_predictions[positive_mask] == positive_label)
            else:
                tpr = 0

            # Calculate FPR
            negative_mask = group_true_labels != positive_label
            if np.sum(negative_mask) > 0:
                fpr = np.mean(group_predictions[negative_mask] == positive_label)
            else:
                fpr = 0

            metrics_by_group[group_name] = {
                'tpr': float(tpr),
                'fpr': float(fpr)
            }

        if len(metrics_by_group) < 2:
            return {
                'equalized_odds': 1.0,
                'metrics_by_group': metrics_by_group,
                'tpr_disparity': 0.0,
                'fpr_disparity': 0.0
            }

        # Calculate disparities
        tprs = [m['tpr'] for m in metrics_by_group.values()]
        fprs = [m['fpr'] for m in metrics_by_group.values()]

        tpr_disparity = max(tprs) - min(tprs)
        fpr_disparity = max(fprs) - min(fprs)

        # Equalized odds: both TPR and FPR should be equal
        equalized_odds_score = 1 - (tpr_disparity + fpr_disparity) / 2

        return {
            'equalized_odds': float(equalized_odds_score),
            'metrics_by_group': metrics_by_group,
            'tpr_disparity': float(tpr_disparity),
            'fpr_disparity': float(fpr_disparity),
            'odds_equalized': tpr_disparity < 0.1 and fpr_disparity < 0.1
        }


class DisparateImpactAnalyzer:
    """Analyzes disparate impact ratio."""

    def analyze_disparate_impact(self,
                                 predictions: List[int],
                                 group_labels: Dict[str, List[int]],
                                 positive_label: int = 1,
                                 threshold: float = 0.8) -> Dict[str, Any]:
        """Analyze disparate impact ratio."""
        if not predictions or not group_labels:
            return {'disparate_impact_ratio': 0.0, 'has_disparate_impact': True}

        predictions = np.array(predictions)
        positive_rates = {}

        for group_name, labels in group_labels.items():
            labels = np.array(labels)
            group_mask = labels == 1
            group_predictions = predictions[group_mask]

            if len(group_predictions) > 0:
                positive_rate = np.mean(group_predictions == positive_label)
                positive_rates[group_name] = float(positive_rate)

        if len(positive_rates) < 2:
            return {
                'disparate_impact_ratio': 1.0,
                'has_disparate_impact': False,
                'group_rates': positive_rates
            }

        # Calculate disparate impact ratio (min/max)
        rates = list(positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        if max_rate > 0:
            di_ratio = min_rate / max_rate
        else:
            di_ratio = 1.0 if min_rate == 0 else 0.0

        # Check 4/5ths rule (80% rule)
        has_disparate_impact = di_ratio < threshold

        return {
            'disparate_impact_ratio': float(di_ratio),
            'has_disparate_impact': has_disparate_impact,
            'threshold': threshold,
            'passes_four_fifths_rule': di_ratio >= 0.8,
            'group_rates': positive_rates,
            'advantaged_group': max(positive_rates, key=positive_rates.get),
            'disadvantaged_group': min(positive_rates, key=positive_rates.get)
        }


class CalibrationFairnessAnalyzer:
    """Analyzes calibration fairness across groups."""

    def analyze_calibration(self,
                           probabilities: List[float],
                           labels: List[int],
                           group_labels: Dict[str, List[int]],
                           n_bins: int = 10) -> Dict[str, Any]:
        """Analyze calibration across groups."""
        if not probabilities or not labels or not group_labels:
            return {'calibration_fairness': 0.0, 'calibration_by_group': {}}

        probabilities = np.array(probabilities)
        labels = np.array(labels)

        calibration_by_group = {}

        for group_name, g_labels in group_labels.items():
            g_labels = np.array(g_labels)
            group_mask = g_labels == 1

            group_probs = probabilities[group_mask]
            group_labels_actual = labels[group_mask]

            if len(group_probs) == 0:
                continue

            # Calculate calibration error for this group
            bin_edges = np.linspace(0, 1, n_bins + 1)
            calibration_errors = []

            for i in range(n_bins):
                bin_mask = (group_probs >= bin_edges[i]) & (group_probs < bin_edges[i + 1])
                if np.sum(bin_mask) > 0:
                    avg_prob = np.mean(group_probs[bin_mask])
                    actual_rate = np.mean(group_labels_actual[bin_mask])
                    calibration_errors.append(abs(avg_prob - actual_rate))

            if calibration_errors:
                ece = np.mean(calibration_errors)
            else:
                ece = 0

            calibration_by_group[group_name] = {
                'expected_calibration_error': float(ece),
                'calibration_score': float(1 - ece)
            }

        if len(calibration_by_group) < 2:
            return {
                'calibration_fairness': 1.0,
                'calibration_by_group': calibration_by_group
            }

        # Calculate calibration disparity
        eces = [c['expected_calibration_error'] for c in calibration_by_group.values()]
        ece_disparity = max(eces) - min(eces)
        calibration_fairness = 1 - ece_disparity

        return {
            'calibration_fairness': float(calibration_fairness),
            'calibration_by_group': calibration_by_group,
            'ece_disparity': float(ece_disparity),
            'best_calibrated_group': min(calibration_by_group, key=lambda x: calibration_by_group[x]['expected_calibration_error']),
            'worst_calibrated_group': max(calibration_by_group, key=lambda x: calibration_by_group[x]['expected_calibration_error'])
        }


class IndividualFairnessAnalyzer:
    """Analyzes individual fairness (similar individuals treated similarly)."""

    def analyze_individual_fairness(self,
                                    features: List[List[float]],
                                    predictions: List[float],
                                    similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """Analyze individual fairness."""
        if not features or not predictions or len(features) != len(predictions):
            return {'individual_fairness': 0.0, 'violations': 0}

        features = np.array(features)
        predictions = np.array(predictions)
        n_samples = len(features)

        # Normalize features
        feature_std = np.std(features, axis=0)
        feature_std[feature_std == 0] = 1
        normalized_features = (features - np.mean(features, axis=0)) / feature_std

        violations = 0
        total_pairs = 0
        violation_examples = []

        # Sample pairs for efficiency
        n_pairs = min(10000, n_samples * (n_samples - 1) // 2)
        pair_indices = np.random.choice(n_samples, size=(n_pairs, 2), replace=True)

        for i, j in pair_indices:
            if i == j:
                continue

            total_pairs += 1

            # Calculate similarity (inverse of distance)
            feature_distance = np.linalg.norm(normalized_features[i] - normalized_features[j])
            prediction_distance = abs(predictions[i] - predictions[j])

            # Similar individuals should have similar predictions
            if feature_distance < similarity_threshold and prediction_distance > similarity_threshold:
                violations += 1
                if len(violation_examples) < 10:
                    violation_examples.append({
                        'pair': (int(i), int(j)),
                        'feature_distance': float(feature_distance),
                        'prediction_distance': float(prediction_distance)
                    })

        violation_rate = violations / total_pairs if total_pairs > 0 else 0
        individual_fairness = 1 - violation_rate

        return {
            'individual_fairness': float(individual_fairness),
            'violation_rate': float(violation_rate),
            'violations': violations,
            'pairs_analyzed': total_pairs,
            'similarity_threshold': similarity_threshold,
            'violation_examples': violation_examples
        }


class CounterfactualFairnessAnalyzer:
    """Analyzes counterfactual fairness."""

    def analyze_counterfactual_fairness(self,
                                        original_predictions: List[float],
                                        counterfactual_predictions: Dict[str, List[float]],
                                        sensitive_attributes: List[str]) -> Dict[str, Any]:
        """Analyze counterfactual fairness."""
        if not original_predictions or not counterfactual_predictions:
            return {'counterfactual_fairness': 0.0, 'attribute_effects': {}}

        original = np.array(original_predictions)
        attribute_effects = {}

        for attribute in sensitive_attributes:
            if attribute not in counterfactual_predictions:
                continue

            counterfactual = np.array(counterfactual_predictions[attribute])

            if len(counterfactual) != len(original):
                continue

            # Calculate effect of changing sensitive attribute
            differences = np.abs(original - counterfactual)
            mean_effect = np.mean(differences)
            max_effect = np.max(differences)

            # Effect should be zero for counterfactual fairness
            fairness_score = 1 - min(1, mean_effect)

            attribute_effects[attribute] = {
                'mean_effect': float(mean_effect),
                'max_effect': float(max_effect),
                'fairness_score': float(fairness_score),
                'causal_effect_detected': mean_effect > 0.05
            }

        if not attribute_effects:
            return {
                'counterfactual_fairness': 0.0,
                'attribute_effects': {},
                'analysis_possible': False
            }

        # Overall counterfactual fairness
        overall_fairness = np.mean([a['fairness_score'] for a in attribute_effects.values()])

        return {
            'counterfactual_fairness': float(overall_fairness),
            'attribute_effects': attribute_effects,
            'most_influential_attribute': max(attribute_effects, key=lambda x: attribute_effects[x]['mean_effect']),
            'fair_attributes': [a for a, v in attribute_effects.items() if v['fairness_score'] > 0.9]
        }


class IntersectionalFairnessAnalyzer:
    """Analyzes fairness across intersectional groups."""

    def analyze_intersectional(self,
                               predictions: List[int],
                               group_memberships: Dict[str, List[int]],
                               labels: List[int] = None) -> Dict[str, Any]:
        """Analyze fairness for intersectional groups."""
        if not predictions or not group_memberships:
            return {'intersectional_fairness': 0.0, 'subgroup_analysis': {}}

        predictions = np.array(predictions)
        if labels is not None:
            labels = np.array(labels)

        # Create intersectional groups
        attributes = list(group_memberships.keys())
        if len(attributes) < 2:
            return {
                'intersectional_fairness': 1.0,
                'subgroup_analysis': {},
                'note': 'Need at least 2 attributes for intersectional analysis'
            }

        # Create all intersections
        n_samples = len(predictions)
        intersections = defaultdict(list)

        for i in range(n_samples):
            intersection_key = tuple(
                group_memberships[attr][i] if i < len(group_memberships[attr]) else 0
                for attr in attributes
            )
            intersections[intersection_key].append(i)

        subgroup_analysis = {}
        positive_rates = {}

        for intersection, indices in intersections.items():
            if len(indices) < 10:  # Minimum sample size
                continue

            subgroup_predictions = predictions[indices]
            positive_rate = np.mean(subgroup_predictions)

            subgroup_name = '_'.join(f"{attr}={intersection[i]}" for i, attr in enumerate(attributes))

            subgroup_data = {
                'size': len(indices),
                'positive_rate': float(positive_rate)
            }

            if labels is not None:
                subgroup_labels = labels[indices]
                if np.sum(subgroup_labels) > 0:
                    tpr = np.mean(subgroup_predictions[subgroup_labels == 1])
                    subgroup_data['tpr'] = float(tpr)

            subgroup_analysis[subgroup_name] = subgroup_data
            positive_rates[subgroup_name] = positive_rate

        if len(positive_rates) < 2:
            return {
                'intersectional_fairness': 1.0,
                'subgroup_analysis': subgroup_analysis
            }

        # Calculate disparity
        rates = list(positive_rates.values())
        disparity = max(rates) - min(rates)
        intersectional_fairness = 1 - disparity

        return {
            'intersectional_fairness': float(intersectional_fairness),
            'subgroup_analysis': subgroup_analysis,
            'max_disparity': float(disparity),
            'most_advantaged_subgroup': max(positive_rates, key=positive_rates.get),
            'most_disadvantaged_subgroup': min(positive_rates, key=positive_rates.get),
            'subgroups_analyzed': len(subgroup_analysis)
        }


# ============================================================================
# Ethical AI Analyzers (20 Analysis Types)
# ============================================================================

class ValueAlignmentAnalyzer:
    """Analyzes alignment with ethical values."""

    def analyze_value_alignment(self,
                                decisions: List[Dict[str, Any]],
                                value_framework: Dict[str, float]) -> Dict[str, Any]:
        """Analyze alignment with specified values."""
        if not decisions or not value_framework:
            return {'value_alignment': 0.0, 'value_scores': {}}

        value_scores = defaultdict(list)

        for decision in decisions:
            decision_values = decision.get('values_demonstrated', {})
            for value, target_weight in value_framework.items():
                demonstrated = decision_values.get(value, 0)
                alignment = 1 - abs(target_weight - demonstrated)
                value_scores[value].append(alignment)

        # Calculate average alignment per value
        alignment_by_value = {}
        for value, scores in value_scores.items():
            alignment_by_value[value] = {
                'mean_alignment': float(np.mean(scores)),
                'min_alignment': float(np.min(scores)),
                'consistency': float(1 - np.std(scores))
            }

        overall_alignment = np.mean([v['mean_alignment'] for v in alignment_by_value.values()]) if alignment_by_value else 0

        return {
            'value_alignment': float(overall_alignment),
            'alignment_by_value': alignment_by_value,
            'best_aligned_value': max(alignment_by_value, key=lambda x: alignment_by_value[x]['mean_alignment']) if alignment_by_value else None,
            'worst_aligned_value': min(alignment_by_value, key=lambda x: alignment_by_value[x]['mean_alignment']) if alignment_by_value else None,
            'alignment_status': 'aligned' if overall_alignment > 0.8 else ('partial' if overall_alignment > 0.5 else 'misaligned')
        }


class HarmAssessmentAnalyzer:
    """Analyzes potential and actual harms."""

    def analyze_harms(self,
                     outcomes: List[Dict[str, Any]],
                     harm_categories: List[str] = None) -> Dict[str, Any]:
        """Analyze harm patterns in outcomes."""
        harm_categories = harm_categories or [
            'physical', 'psychological', 'financial', 'social',
            'privacy', 'autonomy', 'dignity', 'discrimination'
        ]

        if not outcomes:
            return {'harm_score': 0.0, 'harms_detected': 0}

        harms_by_category = defaultdict(list)
        total_harm_instances = 0
        severe_harms = []

        for outcome in outcomes:
            harms = outcome.get('harms', [])
            for harm in harms:
                category = harm.get('category', 'unknown')
                severity = harm.get('severity', 0.5)
                harms_by_category[category].append(severity)
                total_harm_instances += 1

                if severity > 0.7:
                    severe_harms.append({
                        'outcome_id': outcome.get('id'),
                        'category': category,
                        'severity': severity,
                        'description': harm.get('description')
                    })

        # Calculate harm score (lower is better)
        if total_harm_instances == 0:
            harm_score = 0.0
        else:
            all_severities = [s for severities in harms_by_category.values() for s in severities]
            harm_score = np.mean(all_severities)

        # Category analysis
        category_analysis = {}
        for category in harm_categories:
            severities = harms_by_category.get(category, [])
            category_analysis[category] = {
                'count': len(severities),
                'mean_severity': float(np.mean(severities)) if severities else 0,
                'max_severity': float(np.max(severities)) if severities else 0
            }

        return {
            'harm_score': float(harm_score),
            'safety_score': float(1 - harm_score),
            'harms_detected': total_harm_instances,
            'severe_harms': len(severe_harms),
            'harm_rate': total_harm_instances / len(outcomes) if outcomes else 0,
            'category_analysis': category_analysis,
            'most_frequent_harm': max(harms_by_category, key=lambda x: len(harms_by_category[x])) if harms_by_category else None,
            'severe_harm_details': severe_harms[:20]
        }


class RightsComplianceAnalyzer:
    """Analyzes compliance with fundamental rights."""

    def analyze_rights_compliance(self,
                                  operations: List[Dict[str, Any]],
                                  rights_framework: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze compliance with rights framework."""
        if not operations or not rights_framework:
            return {'rights_compliance': 1.0, 'violations': 0}

        violations = []
        compliance_by_right = {}

        for right_name, requirements in rights_framework.items():
            right_violations = []

            for op in operations:
                op_compliance = op.get('rights_compliance', {})
                right_status = op_compliance.get(right_name, {})

                if right_status.get('violated', False):
                    right_violations.append({
                        'operation_id': op.get('id'),
                        'violation_type': right_status.get('violation_type'),
                        'severity': right_status.get('severity', 'unknown')
                    })

            compliance_rate = 1 - (len(right_violations) / len(operations)) if operations else 1
            compliance_by_right[right_name] = {
                'compliance_rate': compliance_rate,
                'violations': len(right_violations),
                'violation_details': right_violations[:5]
            }

            violations.extend(right_violations)

        overall_compliance = np.mean([v['compliance_rate'] for v in compliance_by_right.values()]) if compliance_by_right else 1

        return {
            'rights_compliance': float(overall_compliance),
            'total_violations': len(violations),
            'compliance_by_right': compliance_by_right,
            'most_violated_right': min(compliance_by_right, key=lambda x: compliance_by_right[x]['compliance_rate']) if compliance_by_right else None,
            'critical_violations': [v for v in violations if v.get('severity') == 'critical']
        }


class AutonomyAnalyzer:
    """Analyzes respect for human autonomy."""

    def analyze_autonomy(self,
                        interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze autonomy respect in interactions."""
        if not interactions:
            return {'autonomy_score': 1.0, 'autonomy_violations': 0}

        autonomy_metrics = {
            'informed_consent': [],
            'choice_provided': [],
            'override_available': [],
            'transparency': []
        }

        violations = []

        for interaction in interactions:
            # Check informed consent
            has_consent = interaction.get('informed_consent', False)
            autonomy_metrics['informed_consent'].append(1 if has_consent else 0)

            # Check choice availability
            has_choice = interaction.get('user_choice_available', False)
            autonomy_metrics['choice_provided'].append(1 if has_choice else 0)

            # Check override capability
            can_override = interaction.get('override_available', False)
            autonomy_metrics['override_available'].append(1 if can_override else 0)

            # Check transparency
            is_transparent = interaction.get('decision_explained', False)
            autonomy_metrics['transparency'].append(1 if is_transparent else 0)

            # Record violations
            if not (has_consent and has_choice):
                violations.append({
                    'interaction_id': interaction.get('id'),
                    'missing': ['consent' if not has_consent else None, 'choice' if not has_choice else None]
                })

        # Calculate scores
        metric_scores = {
            metric: float(np.mean(values)) if values else 0
            for metric, values in autonomy_metrics.items()
        }

        overall_autonomy = np.mean(list(metric_scores.values()))

        return {
            'autonomy_score': float(overall_autonomy),
            'metric_scores': metric_scores,
            'autonomy_violations': len(violations),
            'violation_rate': len(violations) / len(interactions) if interactions else 0,
            'weakest_area': min(metric_scores, key=metric_scores.get),
            'violation_details': violations[:20]
        }


class BeneficenceAnalyzer:
    """Analyzes beneficence (doing good)."""

    def analyze_beneficence(self,
                           outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze beneficent outcomes."""
        if not outcomes:
            return {'beneficence_score': 0.0, 'positive_outcomes': 0}

        positive_outcomes = []
        negative_outcomes = []
        neutral_outcomes = []

        benefit_categories = defaultdict(list)

        for outcome in outcomes:
            impact = outcome.get('impact', 0)
            benefits = outcome.get('benefits', [])
            harms = outcome.get('harms', [])

            net_impact = len(benefits) - len(harms)

            outcome_data = {
                'outcome_id': outcome.get('id'),
                'impact_score': impact,
                'benefits': len(benefits),
                'harms': len(harms),
                'net_impact': net_impact
            }

            if impact > 0 or net_impact > 0:
                positive_outcomes.append(outcome_data)
            elif impact < 0 or net_impact < 0:
                negative_outcomes.append(outcome_data)
            else:
                neutral_outcomes.append(outcome_data)

            for benefit in benefits:
                category = benefit.get('category', 'general')
                magnitude = benefit.get('magnitude', 0.5)
                benefit_categories[category].append(magnitude)

        total = len(outcomes)
        beneficence_score = len(positive_outcomes) / total if total > 0 else 0

        # Analyze benefit categories
        category_analysis = {
            cat: {
                'count': len(magnitudes),
                'mean_magnitude': float(np.mean(magnitudes)),
                'total_benefit': float(np.sum(magnitudes))
            }
            for cat, magnitudes in benefit_categories.items()
        }

        return {
            'beneficence_score': float(beneficence_score),
            'positive_outcomes': len(positive_outcomes),
            'negative_outcomes': len(negative_outcomes),
            'neutral_outcomes': len(neutral_outcomes),
            'total_outcomes': total,
            'net_benefit_ratio': (len(positive_outcomes) - len(negative_outcomes)) / total if total > 0 else 0,
            'category_analysis': category_analysis,
            'primary_benefit_area': max(category_analysis, key=lambda x: category_analysis[x]['total_benefit']) if category_analysis else None
        }


class NonMaleficenceAnalyzer:
    """Analyzes non-maleficence (avoiding harm)."""

    def analyze_non_maleficence(self,
                                actions: List[Dict[str, Any]],
                                harm_potential_threshold: float = 0.3) -> Dict[str, Any]:
        """Analyze non-maleficence in actions."""
        if not actions:
            return {'non_maleficence_score': 1.0, 'harmful_actions': 0}

        harmful_actions = []
        safe_actions = []
        harm_prevented = []

        for action in actions:
            harm_potential = action.get('harm_potential', 0)
            harm_realized = action.get('harm_realized', False)
            harm_mitigated = action.get('harm_mitigated', False)

            action_data = {
                'action_id': action.get('id'),
                'harm_potential': harm_potential,
                'harm_realized': harm_realized
            }

            if harm_realized:
                harmful_actions.append(action_data)
            elif harm_potential > harm_potential_threshold:
                if harm_mitigated:
                    harm_prevented.append(action_data)
                else:
                    harmful_actions.append(action_data)
            else:
                safe_actions.append(action_data)

        total = len(actions)
        non_maleficence_score = len(safe_actions) / total if total > 0 else 1

        return {
            'non_maleficence_score': float(non_maleficence_score),
            'safe_actions': len(safe_actions),
            'harmful_actions': len(harmful_actions),
            'harms_prevented': len(harm_prevented),
            'total_actions': total,
            'harm_rate': len(harmful_actions) / total if total > 0 else 0,
            'prevention_effectiveness': len(harm_prevented) / (len(harm_prevented) + len(harmful_actions)) if (len(harm_prevented) + len(harmful_actions)) > 0 else 1,
            'harmful_action_details': harmful_actions[:20]
        }


class JusticeAnalyzer:
    """Analyzes distributive and procedural justice."""

    def analyze_justice(self,
                       distributions: List[Dict[str, Any]],
                       procedures: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze justice in distributions and procedures."""
        results = {
            'justice_score': 0.0,
            'distributive_justice': 0.0,
            'procedural_justice': 0.0
        }

        # Distributive justice analysis
        if distributions:
            equity_scores = []
            for dist in distributions:
                recipients = dist.get('recipients', [])
                values = dist.get('values', [])

                if len(values) >= 2:
                    # Gini coefficient (lower = more equal)
                    values = np.array(sorted(values))
                    n = len(values)
                    index = np.arange(1, n + 1)
                    gini = (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)) if np.sum(values) > 0 else 0
                    equity = 1 - gini
                    equity_scores.append(equity)

            distributive_justice = np.mean(equity_scores) if equity_scores else 0.5
            results['distributive_justice'] = float(distributive_justice)

        # Procedural justice analysis
        if procedures:
            procedural_scores = []
            for proc in procedures:
                score = 0
                if proc.get('consistent_rules', False):
                    score += 0.25
                if proc.get('bias_suppression', False):
                    score += 0.25
                if proc.get('accuracy_of_information', False):
                    score += 0.25
                if proc.get('correctability', False):
                    score += 0.25
                procedural_scores.append(score)

            procedural_justice = np.mean(procedural_scores) if procedural_scores else 0.5
            results['procedural_justice'] = float(procedural_justice)
        else:
            results['procedural_justice'] = 0.5

        results['justice_score'] = (results['distributive_justice'] + results['procedural_justice']) / 2

        return results


# ============================================================================
# Social AI Analyzers (18 Analysis Types)
# ============================================================================

class SocialImpactAnalyzer:
    """Analyzes social impact of AI systems."""

    def analyze_social_impact(self,
                             impact_assessments: List[Dict[str, Any]],
                             stakeholder_groups: List[str] = None) -> Dict[str, Any]:
        """Analyze social impact across stakeholder groups."""
        stakeholder_groups = stakeholder_groups or [
            'end_users', 'employees', 'community', 'society', 'environment'
        ]

        if not impact_assessments:
            return {'social_impact_score': 0.0, 'impact_by_group': {}}

        impact_by_group = defaultdict(list)
        overall_impacts = []

        for assessment in impact_assessments:
            for group in stakeholder_groups:
                group_impact = assessment.get(f'{group}_impact', 0)
                impact_by_group[group].append(group_impact)
                overall_impacts.append(group_impact)

        # Calculate scores
        group_analysis = {}
        for group, impacts in impact_by_group.items():
            if impacts:
                group_analysis[group] = {
                    'mean_impact': float(np.mean(impacts)),
                    'min_impact': float(np.min(impacts)),
                    'max_impact': float(np.max(impacts)),
                    'positive_assessments': sum(1 for i in impacts if i > 0),
                    'negative_assessments': sum(1 for i in impacts if i < 0)
                }

        overall_score = np.mean(overall_impacts) if overall_impacts else 0
        # Normalize to 0-1 scale (assuming impacts range from -1 to 1)
        normalized_score = (overall_score + 1) / 2

        return {
            'social_impact_score': float(normalized_score),
            'raw_impact_score': float(overall_score),
            'impact_by_group': group_analysis,
            'most_positive_group': max(group_analysis, key=lambda x: group_analysis[x]['mean_impact']) if group_analysis else None,
            'most_negative_group': min(group_analysis, key=lambda x: group_analysis[x]['mean_impact']) if group_analysis else None,
            'assessments_analyzed': len(impact_assessments)
        }


class AccessibilityAnalyzer:
    """Analyzes accessibility of AI systems."""

    def analyze_accessibility(self,
                             accessibility_tests: List[Dict[str, Any]],
                             wcag_level: str = 'AA') -> Dict[str, Any]:
        """Analyze accessibility compliance."""
        if not accessibility_tests:
            return {'accessibility_score': 0.0, 'wcag_compliance': 0.0}

        wcag_criteria = {
            'perceivable': [],
            'operable': [],
            'understandable': [],
            'robust': []
        }

        failures = []

        for test in accessibility_tests:
            principle = test.get('principle', 'unknown')
            passed = test.get('passed', False)
            level = test.get('level', 'A')

            if principle in wcag_criteria:
                wcag_criteria[principle].append(1 if passed else 0)

            if not passed:
                failures.append({
                    'criterion': test.get('criterion'),
                    'principle': principle,
                    'level': level,
                    'description': test.get('description')
                })

        # Calculate scores by principle
        principle_scores = {
            principle: float(np.mean(scores)) if scores else 0
            for principle, scores in wcag_criteria.items()
        }

        overall_accessibility = np.mean(list(principle_scores.values())) if principle_scores else 0

        # WCAG compliance (all criteria must pass for compliance)
        level_thresholds = {'A': 0, 'AA': 1, 'AAA': 2}
        threshold = level_thresholds.get(wcag_level, 1)

        compliant_levels = [t for t in accessibility_tests if t.get('passed', False)]
        target_level_tests = [t for t in accessibility_tests if level_thresholds.get(t.get('level', 'A'), 0) <= threshold]
        wcag_compliance = len([t for t in target_level_tests if t.get('passed', False)]) / len(target_level_tests) if target_level_tests else 0

        return {
            'accessibility_score': float(overall_accessibility),
            'wcag_compliance': float(wcag_compliance),
            'wcag_level_target': wcag_level,
            'principle_scores': principle_scores,
            'failures': len(failures),
            'failure_details': failures,
            'weakest_principle': min(principle_scores, key=principle_scores.get) if principle_scores else None
        }


class InclusionAnalyzer:
    """Analyzes inclusion and representation."""

    def analyze_inclusion(self,
                         user_demographics: List[Dict[str, Any]],
                         population_distribution: Dict[str, float],
                         outcomes: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze inclusion across demographic groups."""
        if not user_demographics:
            return {'inclusion_index': 0.0, 'representation_gaps': []}

        # Analyze representation
        user_distribution = defaultdict(int)
        total_users = len(user_demographics)

        for user in user_demographics:
            for attr, value in user.items():
                if attr in population_distribution:
                    user_distribution[f"{attr}_{value}"] += 1

        # Calculate representation index
        representation_analysis = {}
        representation_gaps = []

        for group, expected_rate in population_distribution.items():
            actual_count = user_distribution.get(group, 0)
            actual_rate = actual_count / total_users if total_users > 0 else 0

            representation_ratio = actual_rate / expected_rate if expected_rate > 0 else 0

            representation_analysis[group] = {
                'expected_rate': expected_rate,
                'actual_rate': float(actual_rate),
                'representation_ratio': float(representation_ratio),
                'represented': representation_ratio >= 0.8
            }

            if representation_ratio < 0.8:
                representation_gaps.append({
                    'group': group,
                    'gap': float(expected_rate - actual_rate),
                    'representation_ratio': float(representation_ratio)
                })

        # Calculate inclusion index
        if representation_analysis:
            inclusion_index = np.mean([
                min(1, v['representation_ratio'])
                for v in representation_analysis.values()
            ])
        else:
            inclusion_index = 0

        # Outcome equity analysis
        outcome_equity = 1.0
        if outcomes:
            outcome_by_group = defaultdict(list)
            for i, outcome in enumerate(outcomes):
                if i < len(user_demographics):
                    user = user_demographics[i]
                    for attr, value in user.items():
                        group_key = f"{attr}_{value}"
                        outcome_by_group[group_key].append(outcome.get('success', 0))

            group_outcomes = {
                group: float(np.mean(outcomes)) if outcomes else 0
                for group, outcomes in outcome_by_group.items()
            }

            if group_outcomes:
                outcome_equity = 1 - (max(group_outcomes.values()) - min(group_outcomes.values()))

        return {
            'inclusion_index': float(inclusion_index),
            'outcome_equity': float(outcome_equity),
            'combined_inclusion': float((inclusion_index + outcome_equity) / 2),
            'representation_analysis': representation_analysis,
            'representation_gaps': representation_gaps,
            'underrepresented_groups': len(representation_gaps)
        }


class CommunityImpactAnalyzer:
    """Analyzes impact on communities."""

    def analyze_community_impact(self,
                                 community_metrics: List[Dict[str, Any]],
                                 impact_areas: List[str] = None) -> Dict[str, Any]:
        """Analyze community-level impacts."""
        impact_areas = impact_areas or [
            'employment', 'education', 'healthcare', 'safety',
            'economic', 'social_cohesion', 'cultural'
        ]

        if not community_metrics:
            return {'community_impact_score': 0.0, 'area_analysis': {}}

        area_impacts = defaultdict(list)

        for metrics in community_metrics:
            for area in impact_areas:
                impact = metrics.get(f'{area}_impact', 0)
                area_impacts[area].append(impact)

        area_analysis = {}
        for area, impacts in area_impacts.items():
            if impacts:
                positive = sum(1 for i in impacts if i > 0)
                negative = sum(1 for i in impacts if i < 0)

                area_analysis[area] = {
                    'mean_impact': float(np.mean(impacts)),
                    'positive_impacts': positive,
                    'negative_impacts': negative,
                    'net_positive': positive > negative
                }

        overall_impact = np.mean([
            a['mean_impact'] for a in area_analysis.values()
        ]) if area_analysis else 0

        # Normalize to 0-1 scale
        normalized_score = (overall_impact + 1) / 2

        return {
            'community_impact_score': float(normalized_score),
            'raw_impact_score': float(overall_impact),
            'area_analysis': area_analysis,
            'most_positive_area': max(area_analysis, key=lambda x: area_analysis[x]['mean_impact']) if area_analysis else None,
            'most_negative_area': min(area_analysis, key=lambda x: area_analysis[x]['mean_impact']) if area_analysis else None,
            'areas_with_negative_impact': [a for a, v in area_analysis.items() if v['mean_impact'] < 0]
        }


class DigitalDivideAnalyzer:
    """Analyzes digital divide impact."""

    def analyze_digital_divide(self,
                               user_access: List[Dict[str, Any]],
                               minimum_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze digital divide and accessibility gaps."""
        if not user_access:
            return {'digital_divide_score': 0.0, 'access_gaps': []}

        access_metrics = {
            'internet_speed': [],
            'device_capability': [],
            'digital_literacy': [],
            'affordability': []
        }

        users_below_minimum = []

        for user in user_access:
            for metric in access_metrics:
                value = user.get(metric, 0)
                access_metrics[metric].append(value)

                minimum = minimum_requirements.get(metric, 0)
                if value < minimum:
                    users_below_minimum.append({
                        'user_id': user.get('id'),
                        'metric': metric,
                        'value': value,
                        'minimum': minimum,
                        'gap': minimum - value
                    })

        # Calculate access scores
        metric_scores = {}
        for metric, values in access_metrics.items():
            if values:
                minimum = minimum_requirements.get(metric, 0)
                above_minimum = sum(1 for v in values if v >= minimum)
                metric_scores[metric] = above_minimum / len(values)

        overall_access = np.mean(list(metric_scores.values())) if metric_scores else 0

        # Group analysis
        access_gaps = defaultdict(int)
        for user in users_below_minimum:
            access_gaps[user['metric']] += 1

        return {
            'digital_divide_score': float(overall_access),
            'metric_scores': metric_scores,
            'users_with_gaps': len(set(u['user_id'] for u in users_below_minimum)),
            'total_users': len(user_access),
            'gap_rate': len(set(u['user_id'] for u in users_below_minimum)) / len(user_access) if user_access else 0,
            'access_gaps_by_metric': dict(access_gaps),
            'most_common_gap': max(access_gaps, key=access_gaps.get) if access_gaps else None,
            'gap_details': users_below_minimum[:50]
        }


# ============================================================================
# Report Generator
# ============================================================================

class FairnessReportGenerator:
    """Generates comprehensive fairness, ethics, and social impact reports."""

    def __init__(self):
        self.demographic_parity_analyzer = DemographicParityAnalyzer()
        self.equal_opportunity_analyzer = EqualOpportunityAnalyzer()
        self.equalized_odds_analyzer = EqualizedOddsAnalyzer()
        self.disparate_impact_analyzer = DisparateImpactAnalyzer()
        self.calibration_analyzer = CalibrationFairnessAnalyzer()
        self.individual_fairness_analyzer = IndividualFairnessAnalyzer()
        self.counterfactual_analyzer = CounterfactualFairnessAnalyzer()
        self.intersectional_analyzer = IntersectionalFairnessAnalyzer()

        self.value_alignment_analyzer = ValueAlignmentAnalyzer()
        self.harm_analyzer = HarmAssessmentAnalyzer()
        self.rights_analyzer = RightsComplianceAnalyzer()
        self.autonomy_analyzer = AutonomyAnalyzer()
        self.beneficence_analyzer = BeneficenceAnalyzer()
        self.non_maleficence_analyzer = NonMaleficenceAnalyzer()
        self.justice_analyzer = JusticeAnalyzer()

        self.social_impact_analyzer = SocialImpactAnalyzer()
        self.accessibility_analyzer = AccessibilityAnalyzer()
        self.inclusion_analyzer = InclusionAnalyzer()
        self.community_analyzer = CommunityImpactAnalyzer()
        self.digital_divide_analyzer = DigitalDivideAnalyzer()

    def generate_fairness_report(self,
                                predictions: List[int] = None,
                                labels: List[int] = None,
                                group_labels: Dict[str, List[int]] = None,
                                probabilities: List[float] = None) -> Dict[str, Any]:
        """Generate fairness analysis report."""
        report = {
            'report_type': 'fairness_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if predictions is not None and group_labels is not None:
            report['demographic_parity'] = self.demographic_parity_analyzer.analyze_demographic_parity(
                predictions, group_labels
            )
            report['disparate_impact'] = self.disparate_impact_analyzer.analyze_disparate_impact(
                predictions, group_labels
            )

            if labels is not None:
                report['equal_opportunity'] = self.equal_opportunity_analyzer.analyze_equal_opportunity(
                    predictions, labels, group_labels
                )
                report['equalized_odds'] = self.equalized_odds_analyzer.analyze_equalized_odds(
                    predictions, labels, group_labels
                )

            if probabilities is not None and labels is not None:
                report['calibration'] = self.calibration_analyzer.analyze_calibration(
                    probabilities, labels, group_labels
                )

        return report

    def generate_ethics_report(self,
                              decisions: List[Dict[str, Any]] = None,
                              outcomes: List[Dict[str, Any]] = None,
                              operations: List[Dict[str, Any]] = None,
                              interactions: List[Dict[str, Any]] = None,
                              value_framework: Dict[str, float] = None,
                              rights_framework: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Generate ethics analysis report."""
        report = {
            'report_type': 'ethics_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if decisions and value_framework:
            report['value_alignment'] = self.value_alignment_analyzer.analyze_value_alignment(
                decisions, value_framework
            )

        if outcomes:
            report['harm_assessment'] = self.harm_analyzer.analyze_harms(outcomes)
            report['beneficence'] = self.beneficence_analyzer.analyze_beneficence(outcomes)

        if operations and rights_framework:
            report['rights_compliance'] = self.rights_analyzer.analyze_rights_compliance(
                operations, rights_framework
            )

        if interactions:
            report['autonomy'] = self.autonomy_analyzer.analyze_autonomy(interactions)

        return report

    def generate_social_report(self,
                              impact_assessments: List[Dict[str, Any]] = None,
                              accessibility_tests: List[Dict[str, Any]] = None,
                              user_demographics: List[Dict[str, Any]] = None,
                              population_distribution: Dict[str, float] = None,
                              community_metrics: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate social impact report."""
        report = {
            'report_type': 'social_impact_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if impact_assessments:
            report['social_impact'] = self.social_impact_analyzer.analyze_social_impact(
                impact_assessments
            )

        if accessibility_tests:
            report['accessibility'] = self.accessibility_analyzer.analyze_accessibility(
                accessibility_tests
            )

        if user_demographics and population_distribution:
            report['inclusion'] = self.inclusion_analyzer.analyze_inclusion(
                user_demographics, population_distribution
            )

        if community_metrics:
            report['community_impact'] = self.community_analyzer.analyze_community_impact(
                community_metrics
            )

        return report

    def generate_full_report(self,
                            predictions: List[int] = None,
                            labels: List[int] = None,
                            group_labels: Dict[str, List[int]] = None,
                            decisions: List[Dict[str, Any]] = None,
                            outcomes: List[Dict[str, Any]] = None,
                            impact_assessments: List[Dict[str, Any]] = None,
                            value_framework: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate comprehensive fairness, ethics, and social report."""
        report = {
            'report_type': 'comprehensive_fairness_ethics_social_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        # Fairness analysis
        if predictions is not None and group_labels is not None:
            fairness_report = self.generate_fairness_report(
                predictions, labels, group_labels
            )
            for key, value in fairness_report.items():
                if key not in ['report_type', 'timestamp']:
                    report[f'fairness_{key}'] = value

        # Ethics analysis
        if decisions or outcomes:
            ethics_report = self.generate_ethics_report(
                decisions, outcomes, value_framework=value_framework
            )
            for key, value in ethics_report.items():
                if key not in ['report_type', 'timestamp']:
                    report[f'ethics_{key}'] = value

        # Social analysis
        if impact_assessments:
            social_report = self.generate_social_report(impact_assessments)
            for key, value in social_report.items():
                if key not in ['report_type', 'timestamp']:
                    report[f'social_{key}'] = value

        # Calculate overall scores
        scores = []
        if 'fairness_demographic_parity' in report:
            scores.append(report['fairness_demographic_parity'].get('demographic_parity', 0))
        if 'ethics_harm_assessment' in report:
            scores.append(report['ethics_harm_assessment'].get('safety_score', 0))
        if 'social_social_impact' in report:
            scores.append(report['social_social_impact'].get('social_impact_score', 0))

        report['overall_score'] = float(np.mean(scores)) if scores else 0.0

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
        """Convert report to markdown format."""
        lines = [
            f"# {report.get('report_type', 'Fairness Analysis Report')}",
            f"\n**Generated:** {report.get('timestamp', 'N/A')}",
            f"\n**Overall Score:** {report.get('overall_score', 0):.2%}",
            "\n---\n"
        ]

        if 'fairness_demographic_parity' in report:
            lines.append("## Fairness Analysis\n")
            dp = report['fairness_demographic_parity']
            lines.append(f"- **Demographic Parity:** {dp.get('demographic_parity', 0):.2%}")
            lines.append(f"- **Max Disparity:** {dp.get('max_disparity', 0):.4f}")
            lines.append("")

        if 'ethics_harm_assessment' in report:
            lines.append("## Ethics Analysis\n")
            ha = report['ethics_harm_assessment']
            lines.append(f"- **Safety Score:** {ha.get('safety_score', 0):.2%}")
            lines.append(f"- **Harms Detected:** {ha.get('harms_detected', 0)}")
            lines.append("")

        if 'social_social_impact' in report:
            lines.append("## Social Impact\n")
            si = report['social_social_impact']
            lines.append(f"- **Social Impact Score:** {si.get('social_impact_score', 0):.2%}")
            lines.append("")

        return "\n".join(lines)
