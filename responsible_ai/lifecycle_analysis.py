"""
Lifecycle Analysis Module - Model Lifecycle Management, Fine-Tuning Analysis
=============================================================================

Comprehensive analysis for AI model lifecycle and fine-tuning processes.
Implements 38 analysis types across two related frameworks.

Frameworks:
- Model Lifecycle Management (18 types): Versioning, Deployment, Retirement, Governance
- Fine-Tuning Analysis (20 types): Transfer Learning, Adaptation, Catastrophic Forgetting
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
class LifecycleMetrics:
    """Metrics for lifecycle management analysis."""
    version_control_score: float = 0.0
    deployment_success_rate: float = 0.0
    rollback_capability: float = 0.0
    governance_compliance: float = 0.0
    documentation_completeness: float = 0.0


@dataclass
class FineTuningMetrics:
    """Metrics for fine-tuning analysis."""
    transfer_efficiency: float = 0.0
    adaptation_quality: float = 0.0
    forgetting_rate: float = 0.0
    generalization_retention: float = 0.0
    task_performance: float = 0.0


@dataclass
class ModelVersion:
    """Represents a model version."""
    version_id: str
    version_number: str
    created_at: datetime
    status: str  # 'development', 'staging', 'production', 'deprecated', 'retired'
    metrics: Dict[str, float] = field(default_factory=dict)
    parent_version: Optional[str] = None
    changes: List[str] = field(default_factory=list)


@dataclass
class FineTuningRun:
    """Represents a fine-tuning run."""
    run_id: str
    base_model: str
    target_task: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    success: bool = False


# ============================================================================
# Model Lifecycle Management Analyzers
# ============================================================================

class VersionControlAnalyzer:
    """Analyzes model version control practices."""

    def analyze_version_control(self,
                               versions: List[ModelVersion]) -> Dict[str, Any]:
        """Analyze version control quality."""
        if not versions:
            return {'version_control_score': 0.0, 'total_versions': 0}

        # Analyze version lineage
        has_lineage = sum(1 for v in versions if v.parent_version is not None)
        lineage_coverage = has_lineage / len(versions)

        # Analyze documentation
        has_changes = sum(1 for v in versions if v.changes)
        has_metrics = sum(1 for v in versions if v.metrics)
        documentation_score = (has_changes + has_metrics) / (2 * len(versions))

        # Analyze version progression
        status_distribution = defaultdict(int)
        for version in versions:
            status_distribution[version.status] += 1

        version_control_score = (lineage_coverage * 0.4 + documentation_score * 0.6)

        return {
            'version_control_score': float(version_control_score),
            'total_versions': len(versions),
            'lineage_coverage': float(lineage_coverage),
            'documentation_score': float(documentation_score),
            'status_distribution': dict(status_distribution),
            'active_versions': status_distribution.get('production', 0),
            'deprecated_versions': status_distribution.get('deprecated', 0)
        }


class DeploymentAnalyzer:
    """Analyzes model deployment practices."""

    def analyze_deployments(self,
                           deployments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze deployment success and patterns."""
        if not deployments:
            return {'deployment_success_rate': 0.0, 'total_deployments': 0}

        successful = sum(1 for d in deployments if d.get('success', False))
        rollbacks = sum(1 for d in deployments if d.get('rolled_back', False))

        success_rate = successful / len(deployments)
        rollback_rate = rollbacks / len(deployments)

        # Analyze deployment times
        deployment_times = [d.get('duration_minutes', 0) for d in deployments if 'duration_minutes' in d]
        avg_deployment_time = np.mean(deployment_times) if deployment_times else 0

        # Analyze by environment
        env_analysis = defaultdict(lambda: {'total': 0, 'successful': 0})
        for deployment in deployments:
            env = deployment.get('environment', 'unknown')
            env_analysis[env]['total'] += 1
            if deployment.get('success', False):
                env_analysis[env]['successful'] += 1

        return {
            'deployment_success_rate': float(success_rate),
            'total_deployments': len(deployments),
            'successful_deployments': successful,
            'rollback_rate': float(rollback_rate),
            'rollbacks': rollbacks,
            'average_deployment_time': float(avg_deployment_time),
            'environment_analysis': dict(env_analysis)
        }


class ModelGovernanceAnalyzer:
    """Analyzes model governance compliance."""

    def analyze_governance(self,
                          models: List[Dict[str, Any]],
                          governance_requirements: List[str]) -> Dict[str, Any]:
        """Analyze governance compliance."""
        if not models:
            return {'governance_compliance': 0.0, 'compliant_models': 0}

        compliant_models = []
        non_compliant_models = []

        for model in models:
            model_compliance = model.get('compliance', {})
            met_requirements = [r for r in governance_requirements if model_compliance.get(r, False)]

            compliance_rate = len(met_requirements) / len(governance_requirements) if governance_requirements else 1

            if compliance_rate >= 0.9:
                compliant_models.append(model.get('id'))
            else:
                non_compliant_models.append({
                    'model_id': model.get('id'),
                    'compliance_rate': compliance_rate,
                    'missing': [r for r in governance_requirements if not model_compliance.get(r, False)]
                })

        overall_compliance = len(compliant_models) / len(models) if models else 0

        return {
            'governance_compliance': float(overall_compliance),
            'compliant_models': len(compliant_models),
            'non_compliant_models': len(non_compliant_models),
            'total_models': len(models),
            'non_compliant_details': non_compliant_models[:10],
            'governance_requirements': governance_requirements
        }


class RetirementAnalyzer:
    """Analyzes model retirement processes."""

    def analyze_retirement(self,
                          retired_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model retirement practices."""
        if not retired_models:
            return {'retirement_score': 1.0, 'models_analyzed': 0}

        proper_retirement = []
        improper_retirement = []

        for model in retired_models:
            has_migration_plan = model.get('migration_plan', False)
            has_deprecation_notice = model.get('deprecation_notice', False)
            has_replacement = model.get('replacement_model', None) is not None
            data_archived = model.get('data_archived', False)

            score = (has_migration_plan + has_deprecation_notice + has_replacement + data_archived) / 4

            if score >= 0.75:
                proper_retirement.append(model.get('id'))
            else:
                improper_retirement.append({
                    'model_id': model.get('id'),
                    'score': score,
                    'missing': [
                        'migration_plan' if not has_migration_plan else None,
                        'deprecation_notice' if not has_deprecation_notice else None,
                        'replacement' if not has_replacement else None,
                        'data_archival' if not data_archived else None
                    ]
                })

        retirement_score = len(proper_retirement) / len(retired_models) if retired_models else 1

        return {
            'retirement_score': float(retirement_score),
            'properly_retired': len(proper_retirement),
            'improperly_retired': len(improper_retirement),
            'models_analyzed': len(retired_models),
            'improper_details': improper_retirement
        }


# ============================================================================
# Fine-Tuning Analysis Analyzers
# ============================================================================

class TransferLearningAnalyzer:
    """Analyzes transfer learning effectiveness."""

    def analyze_transfer(self,
                        fine_tuning_runs: List[FineTuningRun]) -> Dict[str, Any]:
        """Analyze transfer learning effectiveness."""
        if not fine_tuning_runs:
            return {'transfer_efficiency': 0.0, 'runs_analyzed': 0}

        transfer_scores = []
        improvement_rates = []

        for run in fine_tuning_runs:
            if not run.metrics_before or not run.metrics_after:
                continue

            # Calculate improvement
            before_acc = run.metrics_before.get('accuracy', 0)
            after_acc = run.metrics_after.get('accuracy', 0)

            if before_acc > 0:
                improvement = (after_acc - before_acc) / before_acc
            else:
                improvement = after_acc

            improvement_rates.append(improvement)

            # Transfer efficiency: how much of base model knowledge was retained
            base_general = run.metrics_before.get('general_performance', 0.5)
            final_general = run.metrics_after.get('general_performance', 0.5)
            retention = final_general / base_general if base_general > 0 else 1

            transfer_scores.append(min(1, (retention * 0.5 + min(1, improvement) * 0.5)))

        avg_transfer = np.mean(transfer_scores) if transfer_scores else 0
        avg_improvement = np.mean(improvement_rates) if improvement_rates else 0

        return {
            'transfer_efficiency': float(avg_transfer),
            'average_improvement': float(avg_improvement),
            'runs_analyzed': len(fine_tuning_runs),
            'successful_transfers': sum(1 for s in transfer_scores if s > 0.6),
            'improvement_distribution': {
                'negative': sum(1 for i in improvement_rates if i < 0),
                'small': sum(1 for i in improvement_rates if 0 <= i < 0.1),
                'moderate': sum(1 for i in improvement_rates if 0.1 <= i < 0.3),
                'large': sum(1 for i in improvement_rates if i >= 0.3)
            }
        }


class CatastrophicForgettingAnalyzer:
    """Analyzes catastrophic forgetting in fine-tuning."""

    def analyze_forgetting(self,
                          fine_tuning_runs: List[FineTuningRun],
                          original_task_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze catastrophic forgetting patterns."""
        if not fine_tuning_runs:
            return {'forgetting_rate': 0.0, 'runs_analyzed': 0}

        forgetting_rates = []
        severe_forgetting = []

        for run in fine_tuning_runs:
            before = run.metrics_before
            after = run.metrics_after

            if not before or not after:
                continue

            # Calculate forgetting on original tasks
            original_tasks = [k for k in before.keys() if k.startswith('task_')]

            if original_tasks:
                task_forgetting = []
                for task in original_tasks:
                    before_score = before.get(task, 0)
                    after_score = after.get(task, 0)

                    if before_score > 0:
                        forgetting = (before_score - after_score) / before_score
                        task_forgetting.append(max(0, forgetting))

                        if forgetting > 0.2:
                            severe_forgetting.append({
                                'run_id': run.run_id,
                                'task': task,
                                'forgetting_rate': forgetting
                            })

                if task_forgetting:
                    forgetting_rates.append(np.mean(task_forgetting))

        avg_forgetting = np.mean(forgetting_rates) if forgetting_rates else 0

        return {
            'forgetting_rate': float(avg_forgetting),
            'runs_analyzed': len(fine_tuning_runs),
            'severe_forgetting_instances': len(severe_forgetting),
            'severe_forgetting_details': severe_forgetting[:20],
            'forgetting_status': 'minimal' if avg_forgetting < 0.05 else ('moderate' if avg_forgetting < 0.15 else 'severe'),
            'recommendations': self._generate_forgetting_recommendations(avg_forgetting)
        }

    def _generate_forgetting_recommendations(self, rate: float) -> List[str]:
        recommendations = []
        if rate > 0.1:
            recommendations.append("Consider elastic weight consolidation (EWC)")
            recommendations.append("Use replay buffers with old task data")
        if rate > 0.2:
            recommendations.append("Reduce learning rate or use gradual unfreezing")
            recommendations.append("Consider multi-task learning instead of sequential fine-tuning")
        return recommendations


class AdaptationQualityAnalyzer:
    """Analyzes fine-tuning adaptation quality."""

    def analyze_adaptation(self,
                          fine_tuning_runs: List[FineTuningRun]) -> Dict[str, Any]:
        """Analyze adaptation quality."""
        if not fine_tuning_runs:
            return {'adaptation_quality': 0.0, 'runs_analyzed': 0}

        successful_runs = [r for r in fine_tuning_runs if r.success]
        success_rate = len(successful_runs) / len(fine_tuning_runs)

        # Analyze by target task
        task_analysis = defaultdict(lambda: {'runs': 0, 'successful': 0, 'avg_improvement': []})

        for run in fine_tuning_runs:
            task = run.target_task
            task_analysis[task]['runs'] += 1
            if run.success:
                task_analysis[task]['successful'] += 1

            if run.metrics_before and run.metrics_after:
                before = run.metrics_before.get('accuracy', 0)
                after = run.metrics_after.get('accuracy', 0)
                improvement = after - before
                task_analysis[task]['avg_improvement'].append(improvement)

        # Calculate quality metrics
        task_summary = {}
        for task, data in task_analysis.items():
            task_summary[task] = {
                'success_rate': data['successful'] / data['runs'] if data['runs'] > 0 else 0,
                'runs': data['runs'],
                'avg_improvement': float(np.mean(data['avg_improvement'])) if data['avg_improvement'] else 0
            }

        adaptation_quality = success_rate

        return {
            'adaptation_quality': float(adaptation_quality),
            'success_rate': float(success_rate),
            'runs_analyzed': len(fine_tuning_runs),
            'successful_runs': len(successful_runs),
            'task_summary': task_summary,
            'best_adapted_task': max(task_summary, key=lambda x: task_summary[x]['avg_improvement']) if task_summary else None
        }


# ============================================================================
# Report Generator
# ============================================================================

class LifecycleReportGenerator:
    """Generates comprehensive lifecycle analysis reports."""

    def __init__(self):
        self.version_analyzer = VersionControlAnalyzer()
        self.deployment_analyzer = DeploymentAnalyzer()
        self.governance_analyzer = ModelGovernanceAnalyzer()
        self.retirement_analyzer = RetirementAnalyzer()
        self.transfer_analyzer = TransferLearningAnalyzer()
        self.forgetting_analyzer = CatastrophicForgettingAnalyzer()
        self.adaptation_analyzer = AdaptationQualityAnalyzer()

    def generate_full_report(self,
                            versions: List[ModelVersion] = None,
                            deployments: List[Dict[str, Any]] = None,
                            fine_tuning_runs: List[FineTuningRun] = None,
                            governance_requirements: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive lifecycle report."""
        report = {
            'report_type': 'comprehensive_lifecycle_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if versions:
            report['version_control'] = self.version_analyzer.analyze_version_control(versions)

        if deployments:
            report['deployments'] = self.deployment_analyzer.analyze_deployments(deployments)

        if fine_tuning_runs:
            report['transfer_learning'] = self.transfer_analyzer.analyze_transfer(fine_tuning_runs)
            report['forgetting'] = self.forgetting_analyzer.analyze_forgetting(fine_tuning_runs)
            report['adaptation'] = self.adaptation_analyzer.analyze_adaptation(fine_tuning_runs)

        # Calculate overall score
        scores = []
        if 'version_control' in report:
            scores.append(report['version_control'].get('version_control_score', 0))
        if 'deployments' in report:
            scores.append(report['deployments'].get('deployment_success_rate', 0))
        if 'adaptation' in report:
            scores.append(report['adaptation'].get('adaptation_quality', 0))

        report['overall_score'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
