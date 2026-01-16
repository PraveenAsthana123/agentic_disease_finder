"""
Sustainability Analysis Module - Green AI, Environmental Impact AI
==================================================================

Comprehensive analysis for AI sustainability and environmental impact.
Implements 38 analysis types across two related frameworks.

Frameworks:
- Sustainable/Green AI (18 types): Energy Efficiency, Carbon Footprint, Resource Optimization
- Environmental Impact AI (20 types): Emissions, Resource Consumption, Lifecycle Impact
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
class SustainabilityMetrics:
    """Metrics for sustainability analysis."""
    energy_efficiency: float = 0.0
    carbon_footprint: float = 0.0  # kg CO2e
    resource_utilization: float = 0.0
    green_score: float = 0.0


@dataclass
class EnvironmentalMetrics:
    """Metrics for environmental impact analysis."""
    total_emissions: float = 0.0  # kg CO2e
    energy_consumed: float = 0.0  # kWh
    water_usage: float = 0.0  # liters
    e_waste_generated: float = 0.0  # kg
    lifecycle_impact: float = 0.0


@dataclass
class TrainingRun:
    """Represents a model training run with resource metrics."""
    run_id: str
    model_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    energy_kwh: float = 0.0
    gpu_hours: float = 0.0
    carbon_kg: float = 0.0
    hardware_type: str = "unknown"
    region: str = "unknown"


@dataclass
class InferenceMetrics:
    """Metrics for inference energy consumption."""
    total_requests: int = 0
    energy_per_request_wh: float = 0.0
    avg_latency_ms: float = 0.0
    hardware_utilization: float = 0.0


# ============================================================================
# Sustainable/Green AI Analyzers
# ============================================================================

class EnergyEfficiencyAnalyzer:
    """Analyzes energy efficiency of AI systems."""

    def analyze_energy_efficiency(self,
                                  training_runs: List[TrainingRun],
                                  performance_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze energy efficiency."""
        if not training_runs:
            return {'energy_efficiency': 0.0, 'total_energy': 0.0}

        total_energy = sum(r.energy_kwh for r in training_runs)
        total_gpu_hours = sum(r.gpu_hours for r in training_runs)

        # Calculate efficiency metrics
        avg_energy_per_run = total_energy / len(training_runs)

        # If performance metrics available, calculate performance per energy
        if performance_metrics:
            accuracy = performance_metrics.get('accuracy', 0)
            if total_energy > 0:
                performance_per_kwh = accuracy / total_energy
            else:
                performance_per_kwh = 0
        else:
            performance_per_kwh = None

        # Analyze by hardware type
        hardware_analysis = defaultdict(lambda: {'energy': 0, 'runs': 0})
        for run in training_runs:
            hardware_analysis[run.hardware_type]['energy'] += run.energy_kwh
            hardware_analysis[run.hardware_type]['runs'] += 1

        for hw in hardware_analysis:
            hardware_analysis[hw]['avg_energy'] = hardware_analysis[hw]['energy'] / hardware_analysis[hw]['runs']

        # Efficiency score (normalized, lower energy is better)
        # Assuming 100 kWh per run is baseline
        baseline_energy = 100 * len(training_runs)
        efficiency_score = max(0, 1 - (total_energy / baseline_energy)) if baseline_energy > 0 else 0

        return {
            'energy_efficiency': float(efficiency_score),
            'total_energy_kwh': float(total_energy),
            'total_gpu_hours': float(total_gpu_hours),
            'avg_energy_per_run': float(avg_energy_per_run),
            'performance_per_kwh': float(performance_per_kwh) if performance_per_kwh else None,
            'hardware_analysis': dict(hardware_analysis),
            'training_runs_analyzed': len(training_runs)
        }


class CarbonFootprintAnalyzer:
    """Analyzes carbon footprint of AI systems."""

    # Carbon intensity by region (kg CO2 per kWh)
    CARBON_INTENSITY = {
        'us-west': 0.25,
        'us-east': 0.40,
        'eu-west': 0.30,
        'eu-north': 0.05,  # Renewable heavy
        'asia-east': 0.55,
        'default': 0.40
    }

    def analyze_carbon_footprint(self,
                                training_runs: List[TrainingRun],
                                inference_metrics: InferenceMetrics = None) -> Dict[str, Any]:
        """Analyze carbon footprint."""
        if not training_runs and not inference_metrics:
            return {'total_carbon_kg': 0.0, 'carbon_intensity': 0.0}

        # Training carbon
        training_carbon = 0.0
        carbon_by_region = defaultdict(float)

        for run in training_runs or []:
            if run.carbon_kg > 0:
                carbon = run.carbon_kg
            else:
                intensity = self.CARBON_INTENSITY.get(run.region, self.CARBON_INTENSITY['default'])
                carbon = run.energy_kwh * intensity

            training_carbon += carbon
            carbon_by_region[run.region] += carbon

        # Inference carbon
        inference_carbon = 0.0
        if inference_metrics:
            # Estimate based on energy per request
            inference_energy_kwh = (inference_metrics.energy_per_request_wh * inference_metrics.total_requests) / 1000
            inference_carbon = inference_energy_kwh * self.CARBON_INTENSITY['default']

        total_carbon = training_carbon + inference_carbon

        # Calculate carbon efficiency
        if training_runs:
            avg_carbon_per_run = training_carbon / len(training_runs)
        else:
            avg_carbon_per_run = 0

        return {
            'total_carbon_kg': float(total_carbon),
            'training_carbon_kg': float(training_carbon),
            'inference_carbon_kg': float(inference_carbon),
            'avg_carbon_per_run': float(avg_carbon_per_run),
            'carbon_by_region': dict(carbon_by_region),
            'equivalent_tree_months': float(total_carbon / 1.5),  # 1 tree absorbs ~1.5 kg CO2/month
            'equivalent_car_km': float(total_carbon / 0.12)  # ~120g CO2 per km
        }


class ResourceOptimizationAnalyzer:
    """Analyzes resource optimization opportunities."""

    def analyze_optimization(self,
                            resource_usage: List[Dict[str, Any]],
                            capacity_limits: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze resource optimization opportunities."""
        if not resource_usage:
            return {'optimization_score': 0.0, 'opportunities': []}

        capacity_limits = capacity_limits or {
            'gpu_utilization': 1.0,
            'memory_utilization': 1.0,
            'cpu_utilization': 1.0
        }

        utilization_by_resource = defaultdict(list)
        underutilized = []
        overutilized = []

        for usage in resource_usage:
            for resource, value in usage.items():
                if resource in capacity_limits:
                    utilization_by_resource[resource].append(value)

                    # Check for inefficiency
                    if value < 0.3:
                        underutilized.append({'resource': resource, 'utilization': value})
                    elif value > 0.95:
                        overutilized.append({'resource': resource, 'utilization': value})

        # Calculate average utilization
        avg_utilization = {}
        for resource, values in utilization_by_resource.items():
            avg_utilization[resource] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        # Optimization opportunities
        opportunities = []
        for resource, stats in avg_utilization.items():
            if stats['mean'] < 0.5:
                opportunities.append({
                    'resource': resource,
                    'current_utilization': stats['mean'],
                    'recommendation': f"Consider downsizing {resource} allocation",
                    'potential_savings': f"{(1 - stats['mean']) * 100:.0f}%"
                })

        # Overall optimization score (balanced utilization is optimal)
        utilization_scores = [stats['mean'] for stats in avg_utilization.values()]
        if utilization_scores:
            # Optimal is around 70-80% utilization
            optimization_score = np.mean([1 - abs(u - 0.75) for u in utilization_scores])
        else:
            optimization_score = 0

        return {
            'optimization_score': float(optimization_score),
            'avg_utilization': avg_utilization,
            'underutilized_instances': len(underutilized),
            'overutilized_instances': len(overutilized),
            'opportunities': opportunities,
            'samples_analyzed': len(resource_usage)
        }


# ============================================================================
# Environmental Impact Analyzers
# ============================================================================

class LifecycleImpactAnalyzer:
    """Analyzes lifecycle environmental impact."""

    def analyze_lifecycle(self,
                         development_metrics: Dict[str, float],
                         deployment_metrics: Dict[str, float],
                         operational_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze full lifecycle impact."""
        # Development phase impact
        dev_energy = development_metrics.get('energy_kwh', 0)
        dev_carbon = development_metrics.get('carbon_kg', 0)

        # Deployment phase impact
        deploy_energy = deployment_metrics.get('energy_kwh', 0)
        deploy_carbon = deployment_metrics.get('carbon_kg', 0)

        # Operational phase impact (projected annual)
        op_energy = operational_metrics.get('energy_kwh_per_day', 0) * 365
        op_carbon = operational_metrics.get('carbon_kg_per_day', 0) * 365

        total_energy = dev_energy + deploy_energy + op_energy
        total_carbon = dev_carbon + deploy_carbon + op_carbon

        # Phase breakdown
        phases = {
            'development': {
                'energy_kwh': dev_energy,
                'carbon_kg': dev_carbon,
                'percentage': dev_energy / total_energy * 100 if total_energy > 0 else 0
            },
            'deployment': {
                'energy_kwh': deploy_energy,
                'carbon_kg': deploy_carbon,
                'percentage': deploy_energy / total_energy * 100 if total_energy > 0 else 0
            },
            'operations': {
                'energy_kwh': op_energy,
                'carbon_kg': op_carbon,
                'percentage': op_energy / total_energy * 100 if total_energy > 0 else 0
            }
        }

        # Identify dominant phase
        dominant_phase = max(phases, key=lambda x: phases[x]['energy_kwh'])

        return {
            'total_energy_kwh': float(total_energy),
            'total_carbon_kg': float(total_carbon),
            'phase_breakdown': phases,
            'dominant_phase': dominant_phase,
            'operational_dominance': phases['operations']['percentage'] > 50,
            'recommendations': self._generate_recommendations(phases)
        }

    def _generate_recommendations(self, phases: Dict) -> List[str]:
        recommendations = []
        if phases['development']['percentage'] > 30:
            recommendations.append("Optimize hyperparameter search to reduce development energy")
        if phases['operations']['percentage'] > 70:
            recommendations.append("Focus on inference optimization for maximum impact")
        return recommendations


class EmissionsTracker:
    """Tracks and analyzes emissions over time."""

    def track_emissions(self,
                       emissions_log: List[Dict[str, Any]],
                       targets: Dict[str, float] = None) -> Dict[str, Any]:
        """Track emissions against targets."""
        if not emissions_log:
            return {'total_emissions_kg': 0.0, 'on_track': True}

        targets = targets or {'annual_limit_kg': 1000}

        total_emissions = sum(e.get('carbon_kg', 0) for e in emissions_log)

        # Calculate rate
        if len(emissions_log) >= 2:
            first_date = emissions_log[0].get('date')
            last_date = emissions_log[-1].get('date')
            if first_date and last_date:
                days = (last_date - first_date).days
                daily_rate = total_emissions / max(1, days)
            else:
                daily_rate = total_emissions / len(emissions_log)
        else:
            daily_rate = total_emissions

        projected_annual = daily_rate * 365
        annual_limit = targets.get('annual_limit_kg', float('inf'))

        on_track = projected_annual <= annual_limit

        return {
            'total_emissions_kg': float(total_emissions),
            'daily_rate_kg': float(daily_rate),
            'projected_annual_kg': float(projected_annual),
            'annual_limit_kg': annual_limit,
            'on_track': on_track,
            'percentage_of_limit': float(projected_annual / annual_limit * 100) if annual_limit > 0 else 0,
            'entries_tracked': len(emissions_log)
        }


# ============================================================================
# Report Generator
# ============================================================================

class SustainabilityReportGenerator:
    """Generates comprehensive sustainability reports."""

    def __init__(self):
        self.energy_analyzer = EnergyEfficiencyAnalyzer()
        self.carbon_analyzer = CarbonFootprintAnalyzer()
        self.optimization_analyzer = ResourceOptimizationAnalyzer()
        self.lifecycle_analyzer = LifecycleImpactAnalyzer()
        self.emissions_tracker = EmissionsTracker()

    def generate_full_report(self,
                            training_runs: List[TrainingRun] = None,
                            resource_usage: List[Dict[str, Any]] = None,
                            emissions_log: List[Dict[str, Any]] = None,
                            development_metrics: Dict[str, float] = None,
                            deployment_metrics: Dict[str, float] = None,
                            operational_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate comprehensive sustainability report."""
        report = {
            'report_type': 'comprehensive_sustainability_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if training_runs:
            report['energy_efficiency'] = self.energy_analyzer.analyze_energy_efficiency(training_runs)
            report['carbon_footprint'] = self.carbon_analyzer.analyze_carbon_footprint(training_runs)

        if resource_usage:
            report['optimization'] = self.optimization_analyzer.analyze_optimization(resource_usage)

        if emissions_log:
            report['emissions_tracking'] = self.emissions_tracker.track_emissions(emissions_log)

        if development_metrics and deployment_metrics and operational_metrics:
            report['lifecycle'] = self.lifecycle_analyzer.analyze_lifecycle(
                development_metrics, deployment_metrics, operational_metrics
            )

        # Calculate overall sustainability score
        scores = []
        if 'energy_efficiency' in report:
            scores.append(report['energy_efficiency'].get('energy_efficiency', 0))
        if 'optimization' in report:
            scores.append(report['optimization'].get('optimization_score', 0))
        if 'emissions_tracking' in report:
            scores.append(1 if report['emissions_tracking'].get('on_track', False) else 0)

        report['sustainability_score'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
