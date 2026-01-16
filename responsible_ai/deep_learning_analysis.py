"""
Deep Learning Analysis Module
=============================

Comprehensive deep learning analysis framework covering training stability,
model complexity, calibration, robustness, and neural network internals.

Categories:
1. Training Stability Analysis - Gradient health, loss convergence
2. Model Complexity Analysis - Parameter efficiency, computational cost
3. Calibration Analysis - Confidence calibration, uncertainty estimation
4. Robustness Analysis - Adversarial robustness, perturbation sensitivity
5. Gradient Analysis - Gradient flow, vanishing/exploding gradients
6. Weight Analysis - Weight distribution, initialization quality
7. Activation Analysis - Activation patterns, dead neurons
8. Attention Analysis - Attention patterns (for transformers)
9. Representation Analysis - Learned representations quality
10. Regularization Analysis - Regularization effectiveness
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math


# =============================================================================
# ENUMS
# =============================================================================

class TrainingStability(Enum):
    """Training stability status."""
    STABLE = auto()
    UNSTABLE = auto()
    OSCILLATING = auto()
    DIVERGING = auto()
    CONVERGED = auto()


class GradientHealth(Enum):
    """Health status of gradients."""
    HEALTHY = auto()
    VANISHING = auto()
    EXPLODING = auto()
    DEAD = auto()
    NOISY = auto()


class WeightStatus(Enum):
    """Status of weight distributions."""
    HEALTHY = auto()
    COLLAPSED = auto()
    EXPLODED = auto()
    SPARSE = auto()
    SATURATED = auto()


class ActivationHealth(Enum):
    """Health status of activations."""
    HEALTHY = auto()
    SATURATED = auto()
    DEAD = auto()
    SPARSE = auto()
    EXPLODING = auto()


class CalibrationStatus(Enum):
    """Model calibration status."""
    WELL_CALIBRATED = auto()
    OVERCONFIDENT = auto()
    UNDERCONFIDENT = auto()
    MISCALIBRATED = auto()


class RobustnessLevel(Enum):
    """Robustness level of model."""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    CRITICAL = auto()


class RegularizationType(Enum):
    """Types of regularization."""
    L1 = auto()
    L2 = auto()
    DROPOUT = auto()
    BATCH_NORM = auto()
    LAYER_NORM = auto()
    WEIGHT_DECAY = auto()
    EARLY_STOPPING = auto()
    DATA_AUGMENTATION = auto()
    MIXUP = auto()
    CUTOUT = auto()


class AttentionPattern(Enum):
    """Types of attention patterns."""
    FOCUSED = auto()
    DISTRIBUTED = auto()
    DIAGONAL = auto()
    SPARSE = auto()
    UNIFORM = auto()
    POSITION_BIASED = auto()


class RepresentationQuality(Enum):
    """Quality of learned representations."""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    COLLAPSED = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GradientMetrics:
    """Metrics about gradient health."""
    layer_name: str
    mean_gradient: float
    std_gradient: float
    max_gradient: float
    min_gradient: float
    gradient_norm: float
    health_status: GradientHealth
    vanishing_ratio: float = 0.0  # Ratio of near-zero gradients
    exploding_ratio: float = 0.0  # Ratio of very large gradients


@dataclass
class WeightMetrics:
    """Metrics about weight distributions."""
    layer_name: str
    mean_weight: float
    std_weight: float
    max_weight: float
    min_weight: float
    sparsity: float
    health_status: WeightStatus
    effective_rank: float = 0.0


@dataclass
class ActivationMetrics:
    """Metrics about activation patterns."""
    layer_name: str
    mean_activation: float
    std_activation: float
    max_activation: float
    dead_neuron_ratio: float
    saturation_ratio: float
    health_status: ActivationHealth


@dataclass
class AttentionMetrics:
    """Metrics about attention patterns."""
    layer_name: str
    head_index: int
    pattern_type: AttentionPattern
    entropy: float
    sparsity: float
    max_attention_weight: float
    position_bias_score: float = 0.0
    diagonal_strength: float = 0.0


@dataclass
class TrainingStabilityMetrics:
    """Metrics about training stability."""
    epoch: int
    loss_value: float
    loss_change: float
    gradient_norm: float
    learning_rate: float
    stability_status: TrainingStability
    issues_detected: List[str] = field(default_factory=list)


@dataclass
class ComplexityMetrics:
    """Model complexity metrics."""
    total_parameters: int
    trainable_parameters: int
    flops: int
    memory_footprint_mb: float
    inference_time_ms: float
    parameter_efficiency: float
    compression_ratio: float = 1.0


@dataclass
class CalibrationMetrics:
    """Deep learning calibration metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    average_confidence: float
    accuracy: float
    calibration_status: CalibrationStatus
    reliability_diagram: List[Tuple[float, float, int]] = field(default_factory=list)
    temperature: float = 1.0


@dataclass
class AdversarialRobustnessMetrics:
    """Adversarial robustness metrics."""
    clean_accuracy: float
    adversarial_accuracy: float
    robustness_gap: float
    attack_success_rate: float
    attack_type: str
    epsilon: float
    robustness_level: RobustnessLevel


@dataclass
class PerturbationRobustnessMetrics:
    """Perturbation robustness metrics."""
    perturbation_type: str
    severity_levels: List[float]
    accuracy_at_levels: List[float]
    degradation_rate: float
    robustness_score: float


@dataclass
class RepresentationMetrics:
    """Metrics about learned representations."""
    layer_name: str
    dimensionality: int
    intrinsic_dimension: float
    cluster_quality: float
    class_separation: float
    representation_quality: RepresentationQuality


@dataclass
class RegularizationMetrics:
    """Metrics about regularization effectiveness."""
    regularization_type: RegularizationType
    strength: float
    train_loss: float
    val_loss: float
    overfitting_gap: float
    effectiveness_score: float
    recommendation: str = ""


@dataclass
class DeepLearningAssessment:
    """Comprehensive deep learning assessment."""
    assessment_id: str
    timestamp: datetime
    training_stability_score: float
    gradient_health_score: float
    weight_health_score: float
    activation_health_score: float
    calibration_score: float
    robustness_score: float
    complexity_efficiency_score: float
    overall_health_score: float
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# ANALYZERS - TRAINING STABILITY
# =============================================================================

class TrainingStabilityAnalyzer:
    """Analyzer for deep learning training stability."""

    def analyze_stability(
        self,
        loss_history: List[float],
        gradient_norms: Optional[List[float]] = None,
        learning_rates: Optional[List[float]] = None
    ) -> List[TrainingStabilityMetrics]:
        """Analyze training stability over epochs."""
        metrics = []

        for i, loss in enumerate(loss_history):
            loss_change = loss - loss_history[i-1] if i > 0 else 0
            grad_norm = gradient_norms[i] if gradient_norms and i < len(gradient_norms) else 0
            lr = learning_rates[i] if learning_rates and i < len(learning_rates) else 0

            # Determine stability
            issues = []
            if i > 0:
                if loss > loss_history[i-1] * 1.5:
                    status = TrainingStability.DIVERGING
                    issues.append("Loss spike detected")
                elif i > 5 and self._is_oscillating(loss_history[max(0, i-5):i+1]):
                    status = TrainingStability.OSCILLATING
                    issues.append("Loss oscillation detected")
                elif i > 10 and abs(loss_change) < 0.0001:
                    status = TrainingStability.CONVERGED
                else:
                    status = TrainingStability.STABLE
            else:
                status = TrainingStability.STABLE

            if grad_norm > 100:
                issues.append("Gradient explosion risk")
            elif grad_norm < 1e-7:
                issues.append("Vanishing gradient risk")

            metrics.append(TrainingStabilityMetrics(
                epoch=i + 1,
                loss_value=loss,
                loss_change=loss_change,
                gradient_norm=grad_norm,
                learning_rate=lr,
                stability_status=status,
                issues_detected=issues,
            ))

        return metrics

    def _is_oscillating(self, values: List[float], threshold: float = 0.1) -> bool:
        """Check if values are oscillating."""
        if len(values) < 3:
            return False

        direction_changes = 0
        for i in range(2, len(values)):
            prev_diff = values[i-1] - values[i-2]
            curr_diff = values[i] - values[i-1]
            if prev_diff * curr_diff < 0:
                direction_changes += 1

        return direction_changes / (len(values) - 2) > 0.5

    def get_stability_summary(
        self,
        metrics: List[TrainingStabilityMetrics]
    ) -> Dict[str, Any]:
        """Get summary of training stability."""
        if not metrics:
            return {}

        status_counts = {}
        all_issues = []

        for m in metrics:
            status_name = m.stability_status.name
            status_counts[status_name] = status_counts.get(status_name, 0) + 1
            all_issues.extend(m.issues_detected)

        return {
            "total_epochs": len(metrics),
            "status_distribution": status_counts,
            "final_status": metrics[-1].stability_status.name,
            "total_issues": len(all_issues),
            "unique_issues": list(set(all_issues)),
        }


# =============================================================================
# ANALYZERS - GRADIENT ANALYSIS
# =============================================================================

class GradientAnalyzer:
    """Analyzer for gradient health in deep learning."""

    def analyze_gradients(
        self,
        layer_gradients: Dict[str, List[float]]
    ) -> List[GradientMetrics]:
        """Analyze gradients for each layer."""
        metrics = []

        for layer_name, gradients in layer_gradients.items():
            if not gradients:
                continue

            mean_grad = sum(gradients) / len(gradients)
            variance = sum((g - mean_grad) ** 2 for g in gradients) / len(gradients)
            std_grad = variance ** 0.5
            max_grad = max(abs(g) for g in gradients)
            min_grad = min(abs(g) for g in gradients)
            grad_norm = (sum(g ** 2 for g in gradients)) ** 0.5

            # Calculate health metrics
            vanishing_ratio = sum(1 for g in gradients if abs(g) < 1e-7) / len(gradients)
            exploding_ratio = sum(1 for g in gradients if abs(g) > 100) / len(gradients)

            # Determine health status
            if vanishing_ratio > 0.5:
                health = GradientHealth.VANISHING
            elif exploding_ratio > 0.1:
                health = GradientHealth.EXPLODING
            elif vanishing_ratio > 0.9:
                health = GradientHealth.DEAD
            elif std_grad / (abs(mean_grad) + 1e-10) > 10:
                health = GradientHealth.NOISY
            else:
                health = GradientHealth.HEALTHY

            metrics.append(GradientMetrics(
                layer_name=layer_name,
                mean_gradient=mean_grad,
                std_gradient=std_grad,
                max_gradient=max_grad,
                min_gradient=min_grad,
                gradient_norm=grad_norm,
                health_status=health,
                vanishing_ratio=vanishing_ratio,
                exploding_ratio=exploding_ratio,
            ))

        return metrics

    def detect_gradient_issues(
        self,
        metrics: List[GradientMetrics]
    ) -> List[str]:
        """Detect gradient-related issues."""
        issues = []

        vanishing_layers = [m.layer_name for m in metrics if m.health_status == GradientHealth.VANISHING]
        exploding_layers = [m.layer_name for m in metrics if m.health_status == GradientHealth.EXPLODING]
        dead_layers = [m.layer_name for m in metrics if m.health_status == GradientHealth.DEAD]

        if vanishing_layers:
            issues.append(f"Vanishing gradients in: {', '.join(vanishing_layers[:3])}")
        if exploding_layers:
            issues.append(f"Exploding gradients in: {', '.join(exploding_layers[:3])}")
        if dead_layers:
            issues.append(f"Dead gradients in: {', '.join(dead_layers[:3])}")

        return issues

    def recommend_fixes(self, metrics: List[GradientMetrics]) -> List[str]:
        """Recommend fixes for gradient issues."""
        recommendations = []

        has_vanishing = any(m.health_status == GradientHealth.VANISHING for m in metrics)
        has_exploding = any(m.health_status == GradientHealth.EXPLODING for m in metrics)

        if has_vanishing:
            recommendations.extend([
                "Use residual/skip connections",
                "Apply batch normalization",
                "Try different activation functions (ReLU, LeakyReLU)",
                "Use gradient clipping",
                "Consider LSTM/GRU for sequence models",
            ])

        if has_exploding:
            recommendations.extend([
                "Apply gradient clipping",
                "Reduce learning rate",
                "Use weight regularization",
                "Check for numerical instability",
            ])

        return recommendations


# =============================================================================
# ANALYZERS - WEIGHT ANALYSIS
# =============================================================================

class WeightAnalyzer:
    """Analyzer for neural network weight distributions."""

    def analyze_weights(
        self,
        layer_weights: Dict[str, List[float]]
    ) -> List[WeightMetrics]:
        """Analyze weight distributions for each layer."""
        metrics = []

        for layer_name, weights in layer_weights.items():
            if not weights:
                continue

            mean_weight = sum(weights) / len(weights)
            variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
            std_weight = variance ** 0.5
            max_weight = max(abs(w) for w in weights)
            min_weight = min(abs(w) for w in weights)

            # Calculate sparsity
            sparsity = sum(1 for w in weights if abs(w) < 1e-6) / len(weights)

            # Determine health status
            if std_weight < 1e-6:
                health = WeightStatus.COLLAPSED
            elif max_weight > 1000:
                health = WeightStatus.EXPLODED
            elif sparsity > 0.9:
                health = WeightStatus.SPARSE
            elif abs(mean_weight) > 10 or std_weight > 10:
                health = WeightStatus.SATURATED
            else:
                health = WeightStatus.HEALTHY

            metrics.append(WeightMetrics(
                layer_name=layer_name,
                mean_weight=mean_weight,
                std_weight=std_weight,
                max_weight=max_weight,
                min_weight=min_weight,
                sparsity=sparsity,
                health_status=health,
            ))

        return metrics

    def check_initialization(
        self,
        weights: List[float],
        expected_std: float,
        tolerance: float = 0.2
    ) -> Dict[str, Any]:
        """Check if weights are properly initialized."""
        if not weights:
            return {"status": "empty", "is_good": False}

        actual_std = (sum((w - sum(weights)/len(weights)) ** 2 for w in weights) / len(weights)) ** 0.5

        is_good = abs(actual_std - expected_std) / expected_std < tolerance

        return {
            "expected_std": expected_std,
            "actual_std": actual_std,
            "deviation": abs(actual_std - expected_std) / expected_std,
            "is_good": is_good,
            "recommendation": "" if is_good else "Consider re-initializing with proper variance",
        }


# =============================================================================
# ANALYZERS - ACTIVATION ANALYSIS
# =============================================================================

class ActivationAnalyzer:
    """Analyzer for neural network activation patterns."""

    def analyze_activations(
        self,
        layer_activations: Dict[str, List[float]]
    ) -> List[ActivationMetrics]:
        """Analyze activation patterns for each layer."""
        metrics = []

        for layer_name, activations in layer_activations.items():
            if not activations:
                continue

            mean_act = sum(activations) / len(activations)
            variance = sum((a - mean_act) ** 2 for a in activations) / len(activations)
            std_act = variance ** 0.5
            max_act = max(activations)

            # Dead neuron ratio (for ReLU-like activations)
            dead_ratio = sum(1 for a in activations if a == 0) / len(activations)

            # Saturation ratio (for sigmoid/tanh)
            saturation_ratio = sum(1 for a in activations if abs(a) > 0.99) / len(activations)

            # Determine health
            if dead_ratio > 0.5:
                health = ActivationHealth.DEAD
            elif saturation_ratio > 0.5:
                health = ActivationHealth.SATURATED
            elif max_act > 1000:
                health = ActivationHealth.EXPLODING
            elif sum(1 for a in activations if a != 0) / len(activations) < 0.1:
                health = ActivationHealth.SPARSE
            else:
                health = ActivationHealth.HEALTHY

            metrics.append(ActivationMetrics(
                layer_name=layer_name,
                mean_activation=mean_act,
                std_activation=std_act,
                max_activation=max_act,
                dead_neuron_ratio=dead_ratio,
                saturation_ratio=saturation_ratio,
                health_status=health,
            ))

        return metrics

    def detect_dying_relu(
        self,
        activations: List[float],
        threshold: float = 0.1
    ) -> bool:
        """Detect dying ReLU problem."""
        if not activations:
            return False
        zero_ratio = sum(1 for a in activations if a == 0) / len(activations)
        return zero_ratio > (1 - threshold)


# =============================================================================
# ANALYZERS - ATTENTION ANALYSIS
# =============================================================================

class AttentionAnalyzer:
    """Analyzer for transformer attention patterns."""

    def analyze_attention(
        self,
        attention_weights: List[List[float]],
        layer_name: str,
        head_index: int
    ) -> AttentionMetrics:
        """Analyze attention pattern for a single head."""
        if not attention_weights or not attention_weights[0]:
            return AttentionMetrics(
                layer_name=layer_name,
                head_index=head_index,
                pattern_type=AttentionPattern.UNIFORM,
                entropy=0.0,
                sparsity=0.0,
                max_attention_weight=0.0,
            )

        # Flatten attention weights
        flat_weights = [w for row in attention_weights for w in row]

        # Calculate entropy
        entropy = -sum(w * math.log(w + 1e-10) for w in flat_weights if w > 0)

        # Calculate sparsity
        sparsity = sum(1 for w in flat_weights if w < 0.01) / len(flat_weights)

        # Max attention weight
        max_weight = max(flat_weights)

        # Determine pattern type
        if sparsity > 0.9:
            pattern = AttentionPattern.SPARSE
        elif max_weight > 0.8:
            pattern = AttentionPattern.FOCUSED
        elif entropy > len(flat_weights) * 0.9 * math.log(len(flat_weights)):
            pattern = AttentionPattern.UNIFORM
        else:
            pattern = AttentionPattern.DISTRIBUTED

        # Check for diagonal pattern
        diagonal_strength = 0.0
        if len(attention_weights) == len(attention_weights[0]):
            n = len(attention_weights)
            diagonal_sum = sum(attention_weights[i][i] for i in range(n))
            diagonal_strength = diagonal_sum / n

        return AttentionMetrics(
            layer_name=layer_name,
            head_index=head_index,
            pattern_type=pattern,
            entropy=entropy,
            sparsity=sparsity,
            max_attention_weight=max_weight,
            diagonal_strength=diagonal_strength,
        )

    def analyze_all_heads(
        self,
        attention_maps: Dict[str, List[List[List[float]]]]
    ) -> List[AttentionMetrics]:
        """Analyze attention patterns for all heads."""
        all_metrics = []

        for layer_name, heads in attention_maps.items():
            for head_idx, weights in enumerate(heads):
                metrics = self.analyze_attention(weights, layer_name, head_idx)
                all_metrics.append(metrics)

        return all_metrics


# =============================================================================
# ANALYZERS - CALIBRATION
# =============================================================================

class DeepLearningCalibrationAnalyzer:
    """Analyzer for deep learning model calibration."""

    def analyze_calibration(
        self,
        predictions: List[float],
        labels: List[int],
        n_bins: int = 10
    ) -> CalibrationMetrics:
        """Analyze model calibration."""
        if not predictions or not labels:
            return CalibrationMetrics(
                expected_calibration_error=0.0,
                maximum_calibration_error=0.0,
                average_confidence=0.0,
                accuracy=0.0,
                calibration_status=CalibrationStatus.MISCALIBRATED,
            )

        # Create bins
        bins = [[] for _ in range(n_bins)]

        for prob, label in zip(predictions, labels):
            bin_idx = min(int(prob * n_bins), n_bins - 1)
            bins[bin_idx].append((prob, label))

        # Calculate ECE and MCE
        ece = 0.0
        mce = 0.0
        reliability = []

        for i, bin_data in enumerate(bins):
            if bin_data:
                avg_conf = sum(p for p, _ in bin_data) / len(bin_data)
                avg_acc = sum(l for _, l in bin_data) / len(bin_data)
                bin_size = len(bin_data)

                error = abs(avg_conf - avg_acc)
                ece += (bin_size / len(predictions)) * error
                mce = max(mce, error)
                reliability.append((avg_conf, avg_acc, bin_size))

        # Overall metrics
        avg_confidence = sum(predictions) / len(predictions)
        correct = sum(1 for p, l in zip(predictions, labels) if (p > 0.5) == l)
        accuracy = correct / len(labels)

        # Determine calibration status
        if ece < 0.05:
            status = CalibrationStatus.WELL_CALIBRATED
        elif avg_confidence > accuracy + 0.1:
            status = CalibrationStatus.OVERCONFIDENT
        elif avg_confidence < accuracy - 0.1:
            status = CalibrationStatus.UNDERCONFIDENT
        else:
            status = CalibrationStatus.MISCALIBRATED

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            average_confidence=avg_confidence,
            accuracy=accuracy,
            calibration_status=status,
            reliability_diagram=reliability,
        )

    def recommend_calibration_method(
        self,
        metrics: CalibrationMetrics
    ) -> str:
        """Recommend calibration method based on metrics."""
        if metrics.calibration_status == CalibrationStatus.WELL_CALIBRATED:
            return "No calibration needed"
        elif metrics.calibration_status == CalibrationStatus.OVERCONFIDENT:
            return "Apply temperature scaling with T > 1.0"
        elif metrics.calibration_status == CalibrationStatus.UNDERCONFIDENT:
            return "Apply temperature scaling with T < 1.0"
        else:
            return "Try Platt scaling or isotonic regression"


# =============================================================================
# ANALYZERS - ROBUSTNESS
# =============================================================================

class AdversarialRobustnessAnalyzer:
    """Analyzer for adversarial robustness."""

    def analyze_adversarial_robustness(
        self,
        clean_predictions: List[int],
        adversarial_predictions: List[int],
        labels: List[int],
        attack_type: str = "FGSM",
        epsilon: float = 0.1
    ) -> AdversarialRobustnessMetrics:
        """Analyze adversarial robustness."""
        if not clean_predictions or not labels:
            return AdversarialRobustnessMetrics(
                clean_accuracy=0.0,
                adversarial_accuracy=0.0,
                robustness_gap=0.0,
                attack_success_rate=0.0,
                attack_type=attack_type,
                epsilon=epsilon,
                robustness_level=RobustnessLevel.CRITICAL,
            )

        # Clean accuracy
        clean_correct = sum(1 for p, l in zip(clean_predictions, labels) if p == l)
        clean_acc = clean_correct / len(labels)

        # Adversarial accuracy
        adv_correct = sum(1 for p, l in zip(adversarial_predictions, labels) if p == l)
        adv_acc = adv_correct / len(labels)

        # Robustness gap
        robustness_gap = clean_acc - adv_acc

        # Attack success rate (on correctly classified samples)
        successful_attacks = sum(
            1 for c, a, l in zip(clean_predictions, adversarial_predictions, labels)
            if c == l and a != l
        )
        attack_success_rate = successful_attacks / max(clean_correct, 1)

        # Determine robustness level
        if robustness_gap < 0.1:
            level = RobustnessLevel.HIGH
        elif robustness_gap < 0.3:
            level = RobustnessLevel.MEDIUM
        elif robustness_gap < 0.5:
            level = RobustnessLevel.LOW
        else:
            level = RobustnessLevel.CRITICAL

        return AdversarialRobustnessMetrics(
            clean_accuracy=clean_acc,
            adversarial_accuracy=adv_acc,
            robustness_gap=robustness_gap,
            attack_success_rate=attack_success_rate,
            attack_type=attack_type,
            epsilon=epsilon,
            robustness_level=level,
        )


class PerturbationRobustnessAnalyzer:
    """Analyzer for perturbation robustness."""

    def analyze_perturbation_robustness(
        self,
        perturbation_type: str,
        severity_levels: List[float],
        accuracy_at_levels: List[float],
        clean_accuracy: float
    ) -> PerturbationRobustnessMetrics:
        """Analyze robustness to various perturbations."""
        if not severity_levels or not accuracy_at_levels:
            return PerturbationRobustnessMetrics(
                perturbation_type=perturbation_type,
                severity_levels=[],
                accuracy_at_levels=[],
                degradation_rate=0.0,
                robustness_score=0.0,
            )

        # Calculate degradation rate (slope of accuracy vs severity)
        if len(severity_levels) >= 2:
            degradation = (clean_accuracy - accuracy_at_levels[-1]) / severity_levels[-1]
        else:
            degradation = 0.0

        # Robustness score (area under accuracy curve)
        robustness_score = sum(accuracy_at_levels) / len(accuracy_at_levels)

        return PerturbationRobustnessMetrics(
            perturbation_type=perturbation_type,
            severity_levels=severity_levels,
            accuracy_at_levels=accuracy_at_levels,
            degradation_rate=degradation,
            robustness_score=robustness_score,
        )


# =============================================================================
# ANALYZERS - REPRESENTATION ANALYSIS
# =============================================================================

class RepresentationAnalyzer:
    """Analyzer for learned representations."""

    def analyze_representations(
        self,
        layer_name: str,
        embeddings: List[List[float]],
        labels: Optional[List[int]] = None
    ) -> RepresentationMetrics:
        """Analyze quality of learned representations."""
        if not embeddings or not embeddings[0]:
            return RepresentationMetrics(
                layer_name=layer_name,
                dimensionality=0,
                intrinsic_dimension=0.0,
                cluster_quality=0.0,
                class_separation=0.0,
                representation_quality=RepresentationQuality.LOW,
            )

        dimensionality = len(embeddings[0])

        # Calculate intrinsic dimension (simplified using variance ratio)
        variances = []
        for dim in range(dimensionality):
            dim_values = [e[dim] for e in embeddings]
            mean_val = sum(dim_values) / len(dim_values)
            variance = sum((v - mean_val) ** 2 for v in dim_values) / len(dim_values)
            variances.append(variance)

        total_var = sum(variances)
        if total_var > 0:
            sorted_vars = sorted(variances, reverse=True)
            cumsum = 0
            intrinsic_dim = 0
            for v in sorted_vars:
                cumsum += v
                intrinsic_dim += 1
                if cumsum / total_var > 0.95:
                    break
        else:
            intrinsic_dim = 0

        # Class separation (if labels provided)
        class_separation = 0.0
        if labels and len(set(labels)) > 1:
            # Simplified: calculate inter-class vs intra-class distance ratio
            class_separation = self._calculate_class_separation(embeddings, labels)

        # Determine quality
        if class_separation > 0.7 and intrinsic_dim > 0:
            quality = RepresentationQuality.HIGH
        elif class_separation > 0.4:
            quality = RepresentationQuality.MEDIUM
        elif sum(variances) < 1e-6:
            quality = RepresentationQuality.COLLAPSED
        else:
            quality = RepresentationQuality.LOW

        return RepresentationMetrics(
            layer_name=layer_name,
            dimensionality=dimensionality,
            intrinsic_dimension=float(intrinsic_dim),
            cluster_quality=class_separation,
            class_separation=class_separation,
            representation_quality=quality,
        )

    def _calculate_class_separation(
        self,
        embeddings: List[List[float]],
        labels: List[int]
    ) -> float:
        """Calculate class separation score."""
        # Group by class
        class_embeddings = {}
        for emb, label in zip(embeddings, labels):
            if label not in class_embeddings:
                class_embeddings[label] = []
            class_embeddings[label].append(emb)

        if len(class_embeddings) < 2:
            return 0.0

        # Calculate class centroids
        centroids = {}
        for label, embs in class_embeddings.items():
            centroid = [sum(e[i] for e in embs) / len(embs) for i in range(len(embs[0]))]
            centroids[label] = centroid

        # Inter-class distance
        inter_class_dist = 0
        count = 0
        labels_list = list(centroids.keys())
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                dist = sum((centroids[labels_list[i]][k] - centroids[labels_list[j]][k]) ** 2
                          for k in range(len(centroids[labels_list[i]]))) ** 0.5
                inter_class_dist += dist
                count += 1

        avg_inter = inter_class_dist / max(count, 1)

        # Intra-class distance
        intra_class_dist = 0
        intra_count = 0
        for label, embs in class_embeddings.items():
            centroid = centroids[label]
            for emb in embs:
                dist = sum((emb[k] - centroid[k]) ** 2 for k in range(len(emb))) ** 0.5
                intra_class_dist += dist
                intra_count += 1

        avg_intra = intra_class_dist / max(intra_count, 1)

        # Separation score
        if avg_intra > 0:
            return avg_inter / (avg_inter + avg_intra)
        return 1.0 if avg_inter > 0 else 0.0


# =============================================================================
# ANALYZERS - REGULARIZATION
# =============================================================================

class RegularizationAnalyzer:
    """Analyzer for regularization effectiveness."""

    def analyze_regularization(
        self,
        reg_type: RegularizationType,
        strength: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> RegularizationMetrics:
        """Analyze regularization effectiveness."""
        train_loss = train_metrics.get("loss", 0)
        val_loss = val_metrics.get("loss", 0)
        overfitting_gap = val_loss - train_loss

        # Effectiveness based on overfitting gap
        if overfitting_gap < 0.1:
            effectiveness = 0.9
            recommendation = "Regularization is effective"
        elif overfitting_gap < 0.3:
            effectiveness = 0.6
            recommendation = "Consider increasing regularization"
        else:
            effectiveness = 0.3
            recommendation = "Regularization may be too weak"

        return RegularizationMetrics(
            regularization_type=reg_type,
            strength=strength,
            train_loss=train_loss,
            val_loss=val_loss,
            overfitting_gap=overfitting_gap,
            effectiveness_score=effectiveness,
            recommendation=recommendation,
        )

    def compare_regularization_methods(
        self,
        results: Dict[str, RegularizationMetrics]
    ) -> Dict[str, Any]:
        """Compare different regularization methods."""
        if not results:
            return {}

        ranked = sorted(
            results.items(),
            key=lambda x: x[1].effectiveness_score,
            reverse=True
        )

        return {
            "ranking": [name for name, _ in ranked],
            "best_method": ranked[0][0],
            "best_effectiveness": ranked[0][1].effectiveness_score,
            "comparison": {name: m.effectiveness_score for name, m in results.items()},
        }


# =============================================================================
# COMPREHENSIVE ANALYZER
# =============================================================================

class DeepLearningAnalyzer:
    """Comprehensive deep learning analyzer."""

    def __init__(self):
        self.stability_analyzer = TrainingStabilityAnalyzer()
        self.gradient_analyzer = GradientAnalyzer()
        self.weight_analyzer = WeightAnalyzer()
        self.activation_analyzer = ActivationAnalyzer()
        self.attention_analyzer = AttentionAnalyzer()
        self.calibration_analyzer = DeepLearningCalibrationAnalyzer()
        self.adversarial_analyzer = AdversarialRobustnessAnalyzer()
        self.representation_analyzer = RepresentationAnalyzer()
        self.regularization_analyzer = RegularizationAnalyzer()

    def comprehensive_assessment(
        self,
        loss_history: List[float],
        layer_gradients: Dict[str, List[float]],
        layer_weights: Dict[str, List[float]],
        layer_activations: Dict[str, List[float]],
        predictions: Optional[List[float]] = None,
        labels: Optional[List[int]] = None
    ) -> DeepLearningAssessment:
        """Perform comprehensive deep learning assessment."""
        assessment_id = f"DLA-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        critical_issues = []
        recommendations = []

        # Training stability
        stability_metrics = self.stability_analyzer.analyze_stability(loss_history)
        stability_score = 1.0 if all(
            m.stability_status in [TrainingStability.STABLE, TrainingStability.CONVERGED]
            for m in stability_metrics
        ) else 0.6

        # Gradient health
        gradient_metrics = self.gradient_analyzer.analyze_gradients(layer_gradients)
        healthy_gradients = sum(1 for m in gradient_metrics if m.health_status == GradientHealth.HEALTHY)
        gradient_score = healthy_gradients / len(gradient_metrics) if gradient_metrics else 0

        gradient_issues = self.gradient_analyzer.detect_gradient_issues(gradient_metrics)
        if gradient_issues:
            critical_issues.extend(gradient_issues)
            recommendations.extend(self.gradient_analyzer.recommend_fixes(gradient_metrics))

        # Weight health
        weight_metrics = self.weight_analyzer.analyze_weights(layer_weights)
        healthy_weights = sum(1 for m in weight_metrics if m.health_status == WeightStatus.HEALTHY)
        weight_score = healthy_weights / len(weight_metrics) if weight_metrics else 0

        # Activation health
        activation_metrics = self.activation_analyzer.analyze_activations(layer_activations)
        healthy_activations = sum(1 for m in activation_metrics if m.health_status == ActivationHealth.HEALTHY)
        activation_score = healthy_activations / len(activation_metrics) if activation_metrics else 0

        # Calibration
        calibration_score = 0.8  # Default
        if predictions and labels:
            cal_metrics = self.calibration_analyzer.analyze_calibration(predictions, labels)
            calibration_score = 1.0 - cal_metrics.expected_calibration_error
            if cal_metrics.calibration_status != CalibrationStatus.WELL_CALIBRATED:
                recommendations.append(self.calibration_analyzer.recommend_calibration_method(cal_metrics))

        # Robustness (placeholder)
        robustness_score = 0.7

        # Complexity efficiency (placeholder)
        complexity_score = 0.8

        # Overall score
        overall = (
            stability_score * 0.2 +
            gradient_score * 0.15 +
            weight_score * 0.15 +
            activation_score * 0.15 +
            calibration_score * 0.15 +
            robustness_score * 0.1 +
            complexity_score * 0.1
        )

        return DeepLearningAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            training_stability_score=stability_score,
            gradient_health_score=gradient_score,
            weight_health_score=weight_score,
            activation_health_score=activation_score,
            calibration_score=calibration_score,
            robustness_score=robustness_score,
            complexity_efficiency_score=complexity_score,
            overall_health_score=overall,
            critical_issues=critical_issues,
            recommendations=recommendations,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_training_stability(loss_history: List[float]) -> List[TrainingStabilityMetrics]:
    """Analyze training stability."""
    analyzer = TrainingStabilityAnalyzer()
    return analyzer.analyze_stability(loss_history)


def analyze_gradients(layer_gradients: Dict[str, List[float]]) -> List[GradientMetrics]:
    """Analyze gradient health."""
    analyzer = GradientAnalyzer()
    return analyzer.analyze_gradients(layer_gradients)


def analyze_weights(layer_weights: Dict[str, List[float]]) -> List[WeightMetrics]:
    """Analyze weight distributions."""
    analyzer = WeightAnalyzer()
    return analyzer.analyze_weights(layer_weights)


def analyze_activations(layer_activations: Dict[str, List[float]]) -> List[ActivationMetrics]:
    """Analyze activation patterns."""
    analyzer = ActivationAnalyzer()
    return analyzer.analyze_activations(layer_activations)


def analyze_attention(
    attention_weights: List[List[float]],
    layer_name: str,
    head_index: int
) -> AttentionMetrics:
    """Analyze attention pattern."""
    analyzer = AttentionAnalyzer()
    return analyzer.analyze_attention(attention_weights, layer_name, head_index)


def analyze_deep_learning_calibration(
    predictions: List[float],
    labels: List[int]
) -> CalibrationMetrics:
    """Analyze deep learning model calibration."""
    analyzer = DeepLearningCalibrationAnalyzer()
    return analyzer.analyze_calibration(predictions, labels)


def analyze_adversarial_robustness(
    clean_predictions: List[int],
    adversarial_predictions: List[int],
    labels: List[int],
    attack_type: str = "FGSM",
    epsilon: float = 0.1
) -> AdversarialRobustnessMetrics:
    """Analyze adversarial robustness."""
    analyzer = AdversarialRobustnessAnalyzer()
    return analyzer.analyze_adversarial_robustness(
        clean_predictions, adversarial_predictions, labels, attack_type, epsilon
    )


def analyze_representations(
    layer_name: str,
    embeddings: List[List[float]],
    labels: Optional[List[int]] = None
) -> RepresentationMetrics:
    """Analyze learned representations."""
    analyzer = RepresentationAnalyzer()
    return analyzer.analyze_representations(layer_name, embeddings, labels)


def comprehensive_deep_learning_assessment(
    loss_history: List[float],
    layer_gradients: Dict[str, List[float]],
    layer_weights: Dict[str, List[float]],
    layer_activations: Dict[str, List[float]]
) -> DeepLearningAssessment:
    """Perform comprehensive deep learning assessment."""
    analyzer = DeepLearningAnalyzer()
    return analyzer.comprehensive_assessment(
        loss_history, layer_gradients, layer_weights, layer_activations
    )
