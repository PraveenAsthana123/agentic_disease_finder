"""
Energy-Efficient AI Analysis Module
====================================

Comprehensive analysis for AI energy efficiency and optimization.
Implements 18 analysis types for energy-efficient AI governance.

Analysis Types:
1. Energy Efficiency Scope & Baseline Definition
2. Workload Characterization Analysis
3. Model Architecture Efficiency Analysis
4. Model Size & Capacity Right-Sizing
5. Training Strategy Energy Analysis
6. Data Efficiency & Sample Utilization
7. Hardware Utilization Efficiency
8. Quantization & Compression Analysis
9. Inference Optimization Analysis
10. Latency-Energy Trade-off Analysis
11. Deployment Environment Analysis
12. Scaling & Load Sensitivity Analysis
13. Monitoring & Energy Drift Detection
14. Dependency & Pipeline Energy Analysis
15. User Behavior & Prompt Efficiency Analysis
16. Cost & Business Impact Analysis
17. Sustainability & ESG Alignment
18. Energy Efficiency Governance & Accountability
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
class EnergyBaseline:
    """Baseline energy metrics for AI system."""
    system_id: str
    baseline_kwh_per_inference: float = 0.0
    baseline_kwh_per_training_epoch: float = 0.0
    measurement_date: Optional[datetime] = None
    workload_type: str = "mixed"  # training, inference, mixed


@dataclass
class WorkloadProfile:
    """Energy workload profile."""
    workload_id: str
    training_energy_kwh: float = 0.0
    inference_energy_kwh: float = 0.0
    idle_energy_kwh: float = 0.0
    peak_energy_kwh: float = 0.0
    avg_utilization: float = 0.0


@dataclass
class ArchitectureMetrics:
    """Model architecture metrics for efficiency analysis."""
    model_name: str
    parameter_count: int = 0
    flops: float = 0.0  # Floating point operations
    layer_count: int = 0
    redundant_layers: int = 0
    memory_footprint_mb: float = 0.0


@dataclass
class CompressionMetrics:
    """Compression and quantization metrics."""
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    quantization_bits: int = 32
    accuracy_drop: float = 0.0
    energy_savings_percent: float = 0.0


@dataclass
class InferenceOptimization:
    """Inference optimization metrics."""
    batch_size: int = 1
    caching_enabled: bool = False
    token_limit: int = 0
    avg_latency_ms: float = 0.0
    energy_per_request_wh: float = 0.0


# ============================================================================
# Energy Efficiency Analyzers
# ============================================================================

class EnergyBaselineAnalyzer:
    """Analyzes energy efficiency scope and baseline (Type 1)."""

    def analyze_baseline(self,
                        energy_measurements: List[Dict[str, Any]],
                        scope_definition: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze energy efficiency scope and establish baseline."""
        if not energy_measurements:
            return {'baseline_established': False, 'baseline_kwh': 0.0}

        scope_definition = scope_definition or {
            'focus': 'mixed',  # training, inference, mixed
            'critical_workloads': []
        }

        # Calculate baseline metrics
        training_energy = [m.get('training_kwh', 0) for m in energy_measurements]
        inference_energy = [m.get('inference_kwh', 0) for m in energy_measurements]

        baseline_training = np.mean(training_energy) if training_energy else 0
        baseline_inference = np.mean(inference_energy) if inference_energy else 0

        return {
            'baseline_established': True,
            'scope': scope_definition,
            'baseline_training_kwh': float(baseline_training),
            'baseline_inference_kwh': float(baseline_inference),
            'baseline_total_kwh': float(baseline_training + baseline_inference),
            'measurement_count': len(energy_measurements),
            'variance_training': float(np.std(training_energy)) if training_energy else 0,
            'variance_inference': float(np.std(inference_energy)) if inference_energy else 0
        }


class WorkloadCharacterizationAnalyzer:
    """Analyzes workload characterization for energy (Type 2)."""

    def analyze_workload(self,
                        workload_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze where energy is actually consumed."""
        if not workload_logs:
            return {'workload_profile': {}, 'energy_distribution': {}}

        # Categorize energy consumption
        training_energy = sum(w.get('training_kwh', 0) for w in workload_logs)
        inference_energy = sum(w.get('inference_kwh', 0) for w in workload_logs)
        idle_energy = sum(w.get('idle_kwh', 0) for w in workload_logs)
        total_energy = training_energy + inference_energy + idle_energy

        # Analyze utilization patterns
        utilization = [w.get('utilization', 0) for w in workload_logs]
        peak_periods = sum(1 for u in utilization if u > 0.8)
        idle_periods = sum(1 for u in utilization if u < 0.2)

        distribution = {}
        if total_energy > 0:
            distribution = {
                'training': training_energy / total_energy * 100,
                'inference': inference_energy / total_energy * 100,
                'idle': idle_energy / total_energy * 100
            }

        return {
            'workload_profile': {
                'total_energy_kwh': float(total_energy),
                'training_energy_kwh': float(training_energy),
                'inference_energy_kwh': float(inference_energy),
                'idle_energy_kwh': float(idle_energy)
            },
            'energy_distribution': distribution,
            'utilization_analysis': {
                'avg_utilization': float(np.mean(utilization)) if utilization else 0,
                'peak_periods': peak_periods,
                'idle_periods': idle_periods,
                'utilization_efficiency': float(1 - (idle_periods / len(workload_logs))) if workload_logs else 0
            },
            'samples_analyzed': len(workload_logs)
        }


class ArchitectureEfficiencyAnalyzer:
    """Analyzes model architecture efficiency (Type 3)."""

    def analyze_architecture(self,
                            architecture: ArchitectureMetrics,
                            benchmark_flops_per_param: float = 2.0) -> Dict[str, Any]:
        """Analyze if architecture is energy-optimal."""
        # Calculate efficiency metrics
        expected_flops = architecture.parameter_count * benchmark_flops_per_param
        flops_efficiency = expected_flops / architecture.flops if architecture.flops > 0 else 0

        # Analyze layer efficiency
        effective_layers = architecture.layer_count - architecture.redundant_layers
        layer_efficiency = effective_layers / architecture.layer_count if architecture.layer_count > 0 else 0

        # Memory efficiency (lower is better for same capability)
        memory_per_param = architecture.memory_footprint_mb / (architecture.parameter_count / 1e6) if architecture.parameter_count > 0 else 0

        # Overall efficiency score
        efficiency_score = (flops_efficiency * 0.4 + layer_efficiency * 0.3 +
                          min(1.0, 4.0 / memory_per_param) * 0.3 if memory_per_param > 0 else 0)

        return {
            'architecture_efficiency_score': float(efficiency_score),
            'flops_efficiency': float(flops_efficiency),
            'layer_efficiency': float(layer_efficiency),
            'memory_per_million_params_mb': float(memory_per_param),
            'redundant_layer_count': architecture.redundant_layers,
            'recommendations': self._generate_recommendations(architecture, efficiency_score)
        }

    def _generate_recommendations(self, arch: ArchitectureMetrics, score: float) -> List[str]:
        recommendations = []
        if arch.redundant_layers > 0:
            recommendations.append(f"Consider pruning {arch.redundant_layers} redundant layers")
        if score < 0.5:
            recommendations.append("Architecture may benefit from efficiency-focused redesign")
        return recommendations


class ModelRightSizingAnalyzer:
    """Analyzes model size and capacity right-sizing (Type 4)."""

    def analyze_right_sizing(self,
                            size_accuracy_curve: List[Dict[str, float]],
                            current_model_size: float) -> Dict[str, Any]:
        """Analyze if model is larger than necessary."""
        if not size_accuracy_curve:
            return {'right_sized': True, 'optimal_size': current_model_size}

        # Find diminishing returns point
        sorted_curve = sorted(size_accuracy_curve, key=lambda x: x.get('size', 0))

        # Calculate marginal gains
        marginal_gains = []
        for i in range(1, len(sorted_curve)):
            size_delta = sorted_curve[i]['size'] - sorted_curve[i-1]['size']
            accuracy_delta = sorted_curve[i]['accuracy'] - sorted_curve[i-1]['accuracy']
            if size_delta > 0:
                marginal_gains.append({
                    'size': sorted_curve[i]['size'],
                    'marginal_gain': accuracy_delta / size_delta
                })

        # Find optimal size (where marginal gain drops below threshold)
        threshold = 0.01  # 1% accuracy per unit size
        optimal_size = current_model_size
        for mg in marginal_gains:
            if mg['marginal_gain'] < threshold:
                optimal_size = mg['size']
                break

        oversized = current_model_size > optimal_size * 1.2  # 20% buffer

        return {
            'right_sized': not oversized,
            'current_size': current_model_size,
            'optimal_size': optimal_size,
            'size_ratio': current_model_size / optimal_size if optimal_size > 0 else 1,
            'potential_reduction': max(0, (current_model_size - optimal_size) / current_model_size * 100),
            'capacity_analysis': {
                'diminishing_returns_point': optimal_size,
                'marginal_gains_analysis': marginal_gains[:5]  # Top 5
            }
        }


class TrainingStrategyAnalyzer:
    """Analyzes training strategy energy efficiency (Type 5)."""

    def analyze_training_strategy(self,
                                 training_config: Dict[str, Any],
                                 training_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if training is energy-efficient."""
        # Evaluate training strategy efficiency
        uses_peft = training_config.get('uses_peft', False)
        uses_lora = training_config.get('uses_lora', False)
        early_stopping = training_config.get('early_stopping', False)
        is_fine_tuning = training_config.get('is_fine_tuning', False)

        # Calculate efficiency score
        efficiency_factors = {
            'peft_usage': 0.25 if uses_peft else 0,
            'lora_usage': 0.25 if uses_lora else 0,
            'early_stopping': 0.25 if early_stopping else 0,
            'fine_tuning_vs_full': 0.25 if is_fine_tuning else 0
        }
        strategy_efficiency = sum(efficiency_factors.values())

        # Analyze training metrics for waste
        if training_metrics:
            total_epochs = len(training_metrics)
            convergence_epoch = next(
                (i for i, m in enumerate(training_metrics)
                 if m.get('validation_loss_delta', 1) < 0.001),
                total_epochs
            )
            wasted_epochs = total_epochs - convergence_epoch
            epoch_efficiency = convergence_epoch / total_epochs if total_epochs > 0 else 1
        else:
            total_epochs = 0
            convergence_epoch = 0
            wasted_epochs = 0
            epoch_efficiency = 1

        return {
            'strategy_efficiency_score': float(strategy_efficiency),
            'optimization_techniques': {
                'peft': uses_peft,
                'lora': uses_lora,
                'early_stopping': early_stopping,
                'fine_tuning': is_fine_tuning
            },
            'epoch_analysis': {
                'total_epochs': total_epochs,
                'convergence_epoch': convergence_epoch,
                'wasted_epochs': wasted_epochs,
                'epoch_efficiency': float(epoch_efficiency)
            },
            'energy_optimization_plan': self._generate_optimization_plan(training_config)
        }

    def _generate_optimization_plan(self, config: Dict) -> List[str]:
        plan = []
        if not config.get('uses_peft'):
            plan.append("Consider PEFT techniques to reduce training compute")
        if not config.get('early_stopping'):
            plan.append("Enable early stopping to prevent wasteful epochs")
        if not config.get('uses_lora'):
            plan.append("LoRA can significantly reduce fine-tuning energy")
        return plan


class DataEfficiencyAnalyzer:
    """Analyzes data efficiency and sample utilization (Type 6)."""

    def analyze_data_efficiency(self,
                               data_metrics: Dict[str, Any],
                               sample_usage_log: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if data is used efficiently."""
        total_samples = data_metrics.get('total_samples', 0)
        unique_samples = data_metrics.get('unique_samples', total_samples)
        epochs = data_metrics.get('epochs', 1)

        # Sample efficiency
        reuse_ratio = (total_samples * epochs) / unique_samples if unique_samples > 0 else 1

        # Curriculum learning usage
        uses_curriculum = data_metrics.get('curriculum_learning', False)
        uses_active_learning = data_metrics.get('active_learning', False)

        # Calculate efficiency score
        efficiency_score = min(1.0, unique_samples / (total_samples * epochs)) if total_samples > 0 else 0
        if uses_curriculum:
            efficiency_score += 0.15
        if uses_active_learning:
            efficiency_score += 0.15
        efficiency_score = min(1.0, efficiency_score)

        return {
            'data_efficiency_score': float(efficiency_score),
            'sample_metrics': {
                'total_samples': total_samples,
                'unique_samples': unique_samples,
                'reuse_ratio': float(reuse_ratio),
                'epochs': epochs
            },
            'techniques_used': {
                'curriculum_learning': uses_curriculum,
                'active_learning': uses_active_learning
            },
            'recommendations': self._generate_data_recommendations(data_metrics)
        }

    def _generate_data_recommendations(self, metrics: Dict) -> List[str]:
        recs = []
        if not metrics.get('curriculum_learning'):
            recs.append("Consider curriculum learning to improve sample efficiency")
        if metrics.get('epochs', 1) > 10:
            recs.append("High epoch count may indicate data inefficiency")
        return recs


class HardwareUtilizationAnalyzer:
    """Analyzes hardware utilization efficiency (Type 7)."""

    def analyze_hardware(self,
                        utilization_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if hardware energy is used effectively."""
        if not utilization_logs:
            return {'hardware_efficiency': 0.0, 'utilization_report': {}}

        # Extract utilization metrics
        gpu_util = [l.get('gpu_utilization', 0) for l in utilization_logs]
        cpu_util = [l.get('cpu_utilization', 0) for l in utilization_logs]
        memory_util = [l.get('memory_utilization', 0) for l in utilization_logs]

        # Calculate efficiency (optimal is 70-85% utilization)
        def calc_efficiency(utils):
            if not utils:
                return 0
            avg = np.mean(utils)
            # Penalty for both under and over utilization
            if avg < 0.3:
                return avg / 0.7  # Under-utilized
            elif avg > 0.95:
                return 0.9  # Over-utilized (risk of throttling)
            else:
                return 1 - abs(avg - 0.75) / 0.75

        gpu_efficiency = calc_efficiency(gpu_util)
        cpu_efficiency = calc_efficiency(cpu_util)
        memory_efficiency = calc_efficiency(memory_util)

        overall_efficiency = (gpu_efficiency * 0.5 + cpu_efficiency * 0.3 + memory_efficiency * 0.2)

        return {
            'hardware_efficiency': float(overall_efficiency),
            'utilization_report': {
                'gpu': {
                    'avg': float(np.mean(gpu_util)) if gpu_util else 0,
                    'max': float(np.max(gpu_util)) if gpu_util else 0,
                    'min': float(np.min(gpu_util)) if gpu_util else 0,
                    'efficiency': float(gpu_efficiency)
                },
                'cpu': {
                    'avg': float(np.mean(cpu_util)) if cpu_util else 0,
                    'efficiency': float(cpu_efficiency)
                },
                'memory': {
                    'avg': float(np.mean(memory_util)) if memory_util else 0,
                    'efficiency': float(memory_efficiency)
                }
            },
            'samples_analyzed': len(utilization_logs)
        }


class CompressionAnalyzer:
    """Analyzes quantization and compression (Type 8)."""

    def analyze_compression(self,
                           original_metrics: Dict[str, Any],
                           compressed_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if model can be compressed safely."""
        original_size = original_metrics.get('size_mb', 0)
        compressed_size = compressed_metrics.get('size_mb', 0)
        original_accuracy = original_metrics.get('accuracy', 0)
        compressed_accuracy = compressed_metrics.get('accuracy', 0)

        # Calculate compression ratio
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
        accuracy_drop = original_accuracy - compressed_accuracy

        # Energy savings estimate (proportional to size reduction)
        energy_savings = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

        # Safe compression threshold (< 2% accuracy drop)
        safe_compression = accuracy_drop < 0.02

        return {
            'compression_safe': safe_compression,
            'compression_ratio': float(compression_ratio),
            'size_reduction_percent': float(energy_savings),
            'accuracy_impact': {
                'original_accuracy': float(original_accuracy),
                'compressed_accuracy': float(compressed_accuracy),
                'accuracy_drop': float(accuracy_drop)
            },
            'energy_savings_estimate_percent': float(energy_savings),
            'techniques_evaluated': {
                'quantization': compressed_metrics.get('quantization_bits', 32),
                'pruning': compressed_metrics.get('pruning_ratio', 0),
                'distillation': compressed_metrics.get('distillation_used', False)
            }
        }


class InferenceOptimizationAnalyzer:
    """Analyzes inference optimization (Type 9)."""

    def analyze_inference(self,
                         inference_config: Dict[str, Any],
                         inference_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if inference is energy-optimal."""
        # Configuration efficiency
        batch_size = inference_config.get('batch_size', 1)
        caching = inference_config.get('caching_enabled', False)
        token_limit = inference_config.get('token_limit', 0)

        config_score = 0
        if batch_size > 1:
            config_score += 0.3
        if caching:
            config_score += 0.4
        if token_limit > 0:
            config_score += 0.3

        # Metrics analysis
        if inference_metrics:
            latencies = [m.get('latency_ms', 0) for m in inference_metrics]
            energies = [m.get('energy_wh', 0) for m in inference_metrics]
            avg_latency = np.mean(latencies)
            avg_energy = np.mean(energies)
            energy_variance = np.std(energies)
        else:
            avg_latency = 0
            avg_energy = 0
            energy_variance = 0

        return {
            'inference_efficiency_score': float(config_score),
            'configuration': {
                'batch_size': batch_size,
                'caching_enabled': caching,
                'token_limit': token_limit
            },
            'performance_metrics': {
                'avg_latency_ms': float(avg_latency),
                'avg_energy_wh': float(avg_energy),
                'energy_variance': float(energy_variance)
            },
            'optimization_opportunities': self._identify_opportunities(inference_config)
        }

    def _identify_opportunities(self, config: Dict) -> List[str]:
        opps = []
        if config.get('batch_size', 1) == 1:
            opps.append("Enable batching to amortize compute overhead")
        if not config.get('caching_enabled'):
            opps.append("Implement caching for repeated queries")
        if not config.get('token_limit'):
            opps.append("Set token limits to prevent runaway generation")
        return opps


class LatencyEnergyTradeoffAnalyzer:
    """Analyzes latency-energy trade-offs (Type 10)."""

    def analyze_tradeoffs(self,
                         tradeoff_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what is traded for efficiency."""
        if not tradeoff_data:
            return {'tradeoff_curve': [], 'optimal_point': None}

        # Build trade-off curve
        curve = []
        for point in tradeoff_data:
            curve.append({
                'latency_ms': point.get('latency_ms', 0),
                'energy_wh': point.get('energy_wh', 0),
                'accuracy': point.get('accuracy', 0)
            })

        # Find Pareto optimal points
        pareto_points = []
        for p in curve:
            is_dominated = False
            for q in curve:
                if (q['latency_ms'] <= p['latency_ms'] and
                    q['energy_wh'] <= p['energy_wh'] and
                    q['accuracy'] >= p['accuracy'] and
                    (q['latency_ms'] < p['latency_ms'] or
                     q['energy_wh'] < p['energy_wh'] or
                     q['accuracy'] > p['accuracy'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_points.append(p)

        # Recommend balanced point
        if pareto_points:
            # Sort by combined normalized score
            for p in pareto_points:
                max_lat = max(x['latency_ms'] for x in pareto_points)
                max_eng = max(x['energy_wh'] for x in pareto_points)
                p['score'] = (1 - p['latency_ms']/max_lat if max_lat else 0) * 0.3 + \
                            (1 - p['energy_wh']/max_eng if max_eng else 0) * 0.4 + \
                            p['accuracy'] * 0.3
            optimal = max(pareto_points, key=lambda x: x['score'])
        else:
            optimal = None

        return {
            'tradeoff_curve': curve,
            'pareto_optimal_points': pareto_points,
            'recommended_operating_point': optimal,
            'analysis': {
                'latency_range_ms': [min(p['latency_ms'] for p in curve), max(p['latency_ms'] for p in curve)] if curve else [0, 0],
                'energy_range_wh': [min(p['energy_wh'] for p in curve), max(p['energy_wh'] for p in curve)] if curve else [0, 0]
            }
        }


class DeploymentEnvironmentAnalyzer:
    """Analyzes deployment environment for energy (Type 11)."""

    # Regional carbon intensity (kg CO2 per kWh)
    REGION_ENERGY_MIX = {
        'us-west-2': {'carbon_intensity': 0.25, 'renewable_percent': 60},
        'us-east-1': {'carbon_intensity': 0.40, 'renewable_percent': 30},
        'eu-west-1': {'carbon_intensity': 0.30, 'renewable_percent': 45},
        'eu-north-1': {'carbon_intensity': 0.05, 'renewable_percent': 95},
        'ap-southeast-1': {'carbon_intensity': 0.50, 'renewable_percent': 20}
    }

    def analyze_deployment(self,
                          deployment_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze where inference should run for energy efficiency."""
        if not deployment_options:
            return {'deployment_comparison': [], 'recommended': None}

        comparisons = []
        for opt in deployment_options:
            region = opt.get('region', 'unknown')
            deployment_type = opt.get('type', 'cloud')  # cloud, edge, on-premise

            region_info = self.REGION_ENERGY_MIX.get(region, {'carbon_intensity': 0.40, 'renewable_percent': 30})

            # Calculate efficiency factors
            latency_factor = 1.0 if deployment_type == 'edge' else 0.8
            energy_factor = region_info['renewable_percent'] / 100

            comparisons.append({
                'option': opt,
                'region': region,
                'deployment_type': deployment_type,
                'carbon_intensity': region_info['carbon_intensity'],
                'renewable_percent': region_info['renewable_percent'],
                'efficiency_score': (latency_factor * 0.3 + energy_factor * 0.7)
            })

        # Recommend best option
        recommended = max(comparisons, key=lambda x: x['efficiency_score']) if comparisons else None

        return {
            'deployment_comparison': comparisons,
            'recommended_deployment': recommended,
            'energy_impact_analysis': {
                'lowest_carbon': min(comparisons, key=lambda x: x['carbon_intensity']) if comparisons else None,
                'highest_renewable': max(comparisons, key=lambda x: x['renewable_percent']) if comparisons else None
            }
        }


class ScalingSensitivityAnalyzer:
    """Analyzes scaling and load sensitivity (Type 12)."""

    def analyze_scaling(self,
                       load_energy_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how energy scales with usage."""
        if not load_energy_data:
            return {'scaling_model': 'unknown', 'scaling_coefficient': 0}

        # Extract load and energy data
        loads = [d.get('load', 0) for d in load_energy_data]
        energies = [d.get('energy_kwh', 0) for d in load_energy_data]

        if len(loads) < 2:
            return {'scaling_model': 'insufficient_data', 'scaling_coefficient': 0}

        # Fit linear model
        loads_arr = np.array(loads)
        energies_arr = np.array(energies)

        # Linear regression
        slope, intercept = np.polyfit(loads_arr, energies_arr, 1)

        # Check for non-linearity
        predicted_linear = slope * loads_arr + intercept
        residuals = energies_arr - predicted_linear
        non_linearity = np.std(residuals) / np.mean(energies_arr) if np.mean(energies_arr) > 0 else 0

        # Determine scaling model
        if non_linearity < 0.1:
            scaling_model = 'linear'
        elif slope > 0 and np.mean(residuals[len(residuals)//2:]) > 0:
            scaling_model = 'super_linear'
        else:
            scaling_model = 'sub_linear'

        # Analyze spike behavior
        max_load_idx = np.argmax(loads_arr)
        peak_energy_ratio = energies_arr[max_load_idx] / np.mean(energies_arr) if np.mean(energies_arr) > 0 else 1

        return {
            'scaling_model': scaling_model,
            'scaling_coefficient': float(slope),
            'base_energy_kwh': float(intercept),
            'non_linearity_score': float(non_linearity),
            'spike_analysis': {
                'peak_energy_ratio': float(peak_energy_ratio),
                'max_load': float(max(loads)),
                'max_energy_kwh': float(max(energies))
            },
            'efficiency_recommendation': 'linear' if scaling_model == 'linear' else 'optimize_for_peaks'
        }


class EnergyDriftAnalyzer:
    """Monitors energy drift over time (Type 13)."""

    def analyze_energy_drift(self,
                            energy_history: List[Dict[str, Any]],
                            threshold: float = 0.15) -> Dict[str, Any]:
        """Detect if energy usage is drifting over time."""
        if not energy_history:
            return {'drift_detected': False, 'drift_score': 0}

        # Sort by timestamp
        sorted_history = sorted(energy_history, key=lambda x: x.get('timestamp', 0))

        # Calculate per-inference energy over time
        energies = [h.get('energy_per_inference_wh', 0) for h in sorted_history]

        if len(energies) < 10:
            return {'drift_detected': False, 'drift_score': 0, 'reason': 'insufficient_data'}

        # Compare first half to second half
        first_half = energies[:len(energies)//2]
        second_half = energies[len(energies)//2:]

        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)

        drift_score = abs(mean_second - mean_first) / mean_first if mean_first > 0 else 0
        drift_detected = drift_score > threshold

        # Identify anomalies
        overall_mean = np.mean(energies)
        overall_std = np.std(energies)
        anomalies = sum(1 for e in energies if abs(e - overall_mean) > 2 * overall_std)

        return {
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'drift_direction': 'increasing' if mean_second > mean_first else 'decreasing',
            'baseline_energy_wh': float(mean_first),
            'current_energy_wh': float(mean_second),
            'anomaly_count': anomalies,
            'monitoring_dashboard': {
                'trend': 'up' if drift_score > 0.05 else 'stable',
                'alert_level': 'high' if drift_score > 0.2 else 'medium' if drift_score > 0.1 else 'low'
            }
        }


class PipelineEnergyAnalyzer:
    """Analyzes dependency and pipeline energy (Type 14)."""

    def analyze_pipeline(self,
                        pipeline_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if upstream/downstream systems waste energy."""
        if not pipeline_components:
            return {'pipeline_efficiency': 0, 'energy_audit': []}

        total_energy = sum(c.get('energy_kwh', 0) for c in pipeline_components)

        # Analyze each component
        audit = []
        redundant_energy = 0
        for comp in pipeline_components:
            energy = comp.get('energy_kwh', 0)
            calls = comp.get('call_count', 1)
            cache_hits = comp.get('cache_hits', 0)

            cache_efficiency = cache_hits / calls if calls > 0 else 0
            redundant = energy * (1 - cache_efficiency) * comp.get('redundancy_factor', 0)
            redundant_energy += redundant

            audit.append({
                'component': comp.get('name', 'unknown'),
                'energy_kwh': energy,
                'call_count': calls,
                'cache_efficiency': float(cache_efficiency),
                'redundant_energy_kwh': float(redundant),
                'efficiency_score': float(cache_efficiency)
            })

        pipeline_efficiency = 1 - (redundant_energy / total_energy) if total_energy > 0 else 1

        return {
            'pipeline_efficiency': float(pipeline_efficiency),
            'total_energy_kwh': float(total_energy),
            'redundant_energy_kwh': float(redundant_energy),
            'energy_audit': audit,
            'recommendations': [a['component'] for a in audit if a['efficiency_score'] < 0.5]
        }


class PromptEfficiencyAnalyzer:
    """Analyzes user behavior and prompt efficiency (Type 15)."""

    def analyze_prompt_efficiency(self,
                                 prompt_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if users drive energy waste."""
        if not prompt_logs:
            return {'prompt_efficiency': 0, 'guidelines': []}

        # Analyze prompt characteristics
        lengths = [p.get('token_count', 0) for p in prompt_logs]
        requery_count = sum(1 for p in prompt_logs if p.get('is_requery', False))
        verbose_count = sum(1 for p in prompt_logs if p.get('token_count', 0) > 500)

        avg_length = np.mean(lengths) if lengths else 0
        requery_rate = requery_count / len(prompt_logs) if prompt_logs else 0
        verbose_rate = verbose_count / len(prompt_logs) if prompt_logs else 0

        # Efficiency score
        efficiency = 1 - (requery_rate * 0.4 + verbose_rate * 0.3 + min(1, avg_length / 1000) * 0.3)

        guidelines = []
        if requery_rate > 0.2:
            guidelines.append("High re-query rate - improve initial response quality")
        if verbose_rate > 0.3:
            guidelines.append("Many verbose prompts - provide prompt optimization tips")
        if avg_length > 300:
            guidelines.append("Average prompt length is high - encourage concise queries")

        return {
            'prompt_efficiency': float(efficiency),
            'metrics': {
                'avg_token_count': float(avg_length),
                'requery_rate': float(requery_rate),
                'verbose_prompt_rate': float(verbose_rate)
            },
            'prompt_efficiency_guidelines': guidelines,
            'samples_analyzed': len(prompt_logs)
        }


class CostImpactAnalyzer:
    """Analyzes cost and business impact (Type 16)."""

    def analyze_cost_impact(self,
                           energy_costs: List[Dict[str, Any]],
                           optimization_investments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if energy efficiency reduces cost."""
        if not energy_costs:
            return {'cost_efficiency': 0, 'roi': 0}

        total_energy_cost = sum(c.get('cost_usd', 0) for c in energy_costs)
        total_requests = sum(c.get('request_count', 0) for c in energy_costs)

        cost_per_request = total_energy_cost / total_requests if total_requests > 0 else 0

        # Calculate ROI of optimizations
        if optimization_investments:
            total_investment = sum(i.get('cost_usd', 0) for i in optimization_investments)
            total_savings = sum(i.get('annual_savings_usd', 0) for i in optimization_investments)
            roi = (total_savings - total_investment) / total_investment if total_investment > 0 else 0
            payback_months = total_investment / (total_savings / 12) if total_savings > 0 else float('inf')
        else:
            total_investment = 0
            total_savings = 0
            roi = 0
            payback_months = 0

        return {
            'cost_efficiency': float(1 - min(1, cost_per_request)),
            'energy_cost_report': {
                'total_cost_usd': float(total_energy_cost),
                'cost_per_request_usd': float(cost_per_request),
                'total_requests': total_requests
            },
            'optimization_roi': {
                'total_investment_usd': float(total_investment),
                'annual_savings_usd': float(total_savings),
                'roi_percent': float(roi * 100),
                'payback_months': float(payback_months) if payback_months != float('inf') else None
            }
        }


class ESGAlignmentAnalyzer:
    """Analyzes sustainability and ESG alignment (Type 17)."""

    def analyze_esg_alignment(self,
                             energy_metrics: Dict[str, Any],
                             esg_targets: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if efficiency supports sustainability goals."""
        carbon_emissions = energy_metrics.get('carbon_kg', 0)
        energy_kwh = energy_metrics.get('energy_kwh', 0)
        renewable_percent = energy_metrics.get('renewable_percent', 0)

        carbon_target = esg_targets.get('carbon_reduction_percent', 0)
        renewable_target = esg_targets.get('renewable_percent', 0)
        efficiency_target = esg_targets.get('efficiency_improvement_percent', 0)

        # Calculate alignment scores
        carbon_baseline = esg_targets.get('carbon_baseline_kg', carbon_emissions)
        carbon_reduction = (carbon_baseline - carbon_emissions) / carbon_baseline * 100 if carbon_baseline > 0 else 0
        carbon_alignment = min(1.0, carbon_reduction / carbon_target) if carbon_target > 0 else 1.0

        renewable_alignment = renewable_percent / renewable_target if renewable_target > 0 else 1.0

        overall_alignment = (carbon_alignment * 0.5 + renewable_alignment * 0.5)

        return {
            'esg_alignment_score': float(overall_alignment),
            'carbon_alignment': {
                'current_carbon_kg': float(carbon_emissions),
                'baseline_kg': float(carbon_baseline),
                'reduction_achieved_percent': float(carbon_reduction),
                'target_percent': float(carbon_target),
                'on_track': carbon_reduction >= carbon_target
            },
            'renewable_alignment': {
                'current_renewable_percent': float(renewable_percent),
                'target_percent': float(renewable_target),
                'on_track': renewable_percent >= renewable_target
            },
            'esg_evidence': {
                'supports_sustainability_goals': overall_alignment >= 0.8,
                'improvement_areas': self._identify_improvements(energy_metrics, esg_targets)
            }
        }

    def _identify_improvements(self, metrics: Dict, targets: Dict) -> List[str]:
        improvements = []
        if metrics.get('renewable_percent', 0) < targets.get('renewable_percent', 0):
            improvements.append("Increase renewable energy usage")
        if metrics.get('carbon_kg', 0) > targets.get('carbon_target_kg', 0):
            improvements.append("Reduce carbon emissions")
        return improvements


class EnergyGovernanceAnalyzer:
    """Analyzes energy efficiency governance (Type 18)."""

    def analyze_governance(self,
                          governance_config: Dict[str, Any],
                          audit_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze who owns energy efficiency decisions."""
        # Check governance elements
        has_owner = governance_config.get('energy_efficiency_owner') is not None
        has_budget = governance_config.get('energy_budget_kwh') is not None
        has_approval_gates = governance_config.get('approval_gates', []) != []
        has_raci = governance_config.get('raci_matrix') is not None

        governance_score = sum([has_owner, has_budget, has_approval_gates, has_raci]) / 4

        # Analyze audit trail
        if audit_records:
            recent_audits = [a for a in audit_records if a.get('days_ago', 365) < 90]
            audit_frequency = len(recent_audits)
            compliance_rate = sum(1 for a in audit_records if a.get('compliant', False)) / len(audit_records)
        else:
            audit_frequency = 0
            compliance_rate = 0

        return {
            'governance_score': float(governance_score),
            'governance_elements': {
                'owner_assigned': has_owner,
                'energy_budget_defined': has_budget,
                'approval_gates_configured': has_approval_gates,
                'raci_defined': has_raci
            },
            'ownership': {
                'energy_efficiency_owner': governance_config.get('energy_efficiency_owner'),
                'raci_matrix': governance_config.get('raci_matrix')
            },
            'audit_trail': {
                'total_audits': len(audit_records) if audit_records else 0,
                'recent_audit_count': audit_frequency,
                'compliance_rate': float(compliance_rate)
            },
            'governance_policy': governance_config.get('policy_document')
        }


# ============================================================================
# Report Generator
# ============================================================================

class EnergyEfficiencyReportGenerator:
    """Generates comprehensive energy efficiency reports."""

    def __init__(self):
        self.baseline_analyzer = EnergyBaselineAnalyzer()
        self.workload_analyzer = WorkloadCharacterizationAnalyzer()
        self.architecture_analyzer = ArchitectureEfficiencyAnalyzer()
        self.hardware_analyzer = HardwareUtilizationAnalyzer()
        self.compression_analyzer = CompressionAnalyzer()
        self.inference_analyzer = InferenceOptimizationAnalyzer()
        self.drift_analyzer = EnergyDriftAnalyzer()
        self.esg_analyzer = ESGAlignmentAnalyzer()
        self.governance_analyzer = EnergyGovernanceAnalyzer()

    def generate_full_report(self,
                            energy_measurements: List[Dict[str, Any]] = None,
                            workload_logs: List[Dict[str, Any]] = None,
                            architecture: ArchitectureMetrics = None,
                            utilization_logs: List[Dict[str, Any]] = None,
                            energy_history: List[Dict[str, Any]] = None,
                            esg_targets: Dict[str, Any] = None,
                            governance_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive energy efficiency report."""
        report = {
            'report_type': 'energy_efficiency_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if energy_measurements:
            report['baseline'] = self.baseline_analyzer.analyze_baseline(energy_measurements)

        if workload_logs:
            report['workload'] = self.workload_analyzer.analyze_workload(workload_logs)

        if architecture:
            report['architecture'] = self.architecture_analyzer.analyze_architecture(architecture)

        if utilization_logs:
            report['hardware'] = self.hardware_analyzer.analyze_hardware(utilization_logs)

        if energy_history:
            report['drift'] = self.drift_analyzer.analyze_energy_drift(energy_history)

        if esg_targets and energy_measurements:
            energy_metrics = {
                'energy_kwh': sum(m.get('energy_kwh', 0) for m in energy_measurements),
                'carbon_kg': sum(m.get('carbon_kg', 0) for m in energy_measurements)
            }
            report['esg_alignment'] = self.esg_analyzer.analyze_esg_alignment(energy_metrics, esg_targets)

        if governance_config:
            report['governance'] = self.governance_analyzer.analyze_governance(governance_config)

        # Calculate overall score
        scores = []
        if 'baseline' in report:
            scores.append(1.0 if report['baseline'].get('baseline_established') else 0)
        if 'workload' in report:
            scores.append(report['workload'].get('utilization_analysis', {}).get('utilization_efficiency', 0))
        if 'architecture' in report:
            scores.append(report['architecture'].get('architecture_efficiency_score', 0))
        if 'hardware' in report:
            scores.append(report['hardware'].get('hardware_efficiency', 0))
        if 'esg_alignment' in report:
            scores.append(report['esg_alignment'].get('esg_alignment_score', 0))
        if 'governance' in report:
            scores.append(report['governance'].get('governance_score', 0))

        report['energy_efficiency_score'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'markdown':
            with open(filepath, 'w') as f:
                f.write(self._to_markdown(report))

    def _to_markdown(self, report: Dict[str, Any]) -> str:
        md = "# Energy Efficiency Analysis Report\n\n"
        md += f"Generated: {report.get('timestamp', 'N/A')}\n\n"
        md += f"**Overall Energy Efficiency Score: {report.get('energy_efficiency_score', 0):.2%}**\n\n"

        for section, data in report.items():
            if isinstance(data, dict) and section not in ['timestamp', 'report_type', 'framework_version', 'energy_efficiency_score']:
                md += f"## {section.replace('_', ' ').title()}\n\n"
                for key, value in data.items():
                    md += f"- **{key}**: {value}\n"
                md += "\n"

        return md
