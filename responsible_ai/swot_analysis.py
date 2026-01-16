"""
SWOT Analysis AI Module
========================

Comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis
for AI systems across multiple dimensions and lifecycle stages.

Analysis Types:
1. Classic SWOT Analysis
2. AI-Specific SWOT Framework
3. Lifecycle Stage SWOT Analysis
4. Strategic Positioning Analysis
5. Competitive Analysis
6. Risk-Opportunity Matrix
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
class SWOTItem:
    """Represents an item in SWOT analysis."""
    item_id: str
    category: str  # strength, weakness, opportunity, threat
    description: str
    impact: str = "medium"  # low, medium, high
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)


@dataclass
class StrategicFactor:
    """Represents a strategic factor in SWOT."""
    factor_id: str
    factor_type: str
    description: str
    weight: float = 1.0
    score: float = 0.0
    is_internal: bool = True


@dataclass
class CompetitivePosition:
    """Represents competitive positioning."""
    dimension: str
    our_score: float = 0.0
    competitor_avg: float = 0.0
    market_benchmark: float = 0.0


# ============================================================================
# Classic SWOT Analyzer
# ============================================================================

class ClassicSWOTAnalyzer:
    """Performs classic SWOT analysis for AI systems."""

    def analyze_swot(self,
                    ai_system_assessment: Dict[str, Any],
                    market_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform classic SWOT analysis."""
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []

        # Internal Analysis - Strengths
        if ai_system_assessment.get('accuracy', 0) > 0.9:
            strengths.append({
                'factor': 'High accuracy',
                'evidence': f"Accuracy: {ai_system_assessment.get('accuracy', 0):.2%}",
                'impact': 'high'
            })

        if ai_system_assessment.get('inference_speed_ms', float('inf')) < 100:
            strengths.append({
                'factor': 'Fast inference',
                'evidence': f"Latency: {ai_system_assessment.get('inference_speed_ms', 0)}ms",
                'impact': 'medium'
            })

        if ai_system_assessment.get('scalability_score', 0) > 0.8:
            strengths.append({
                'factor': 'High scalability',
                'evidence': f"Scalability score: {ai_system_assessment.get('scalability_score', 0):.2%}",
                'impact': 'high'
            })

        if ai_system_assessment.get('proprietary_data', False):
            strengths.append({
                'factor': 'Proprietary data advantage',
                'evidence': 'Access to unique training data',
                'impact': 'high'
            })

        # Internal Analysis - Weaknesses
        if ai_system_assessment.get('explainability_score', 0) < 0.5:
            weaknesses.append({
                'factor': 'Explainability gaps',
                'evidence': 'Black-box behavior limits trust',
                'impact': 'high'
            })

        if ai_system_assessment.get('data_drift_risk', 0) > 0.3:
            weaknesses.append({
                'factor': 'Data dependency risk',
                'evidence': 'Sensitive to data quality/drift',
                'impact': 'medium'
            })

        if ai_system_assessment.get('ood_robustness', 0) < 0.7:
            weaknesses.append({
                'factor': 'Generalization limits',
                'evidence': 'Poor OOD performance',
                'impact': 'medium'
            })

        if ai_system_assessment.get('maintenance_cost', 0) > 0.5:
            weaknesses.append({
                'factor': 'Maintenance burden',
                'evidence': 'High retraining/monitoring cost',
                'impact': 'medium'
            })

        # External Analysis - Opportunities
        if market_context:
            if market_context.get('automation_potential', 0) > 0.5:
                opportunities.append({
                    'factor': 'Process automation',
                    'evidence': 'End-to-end automation potential',
                    'impact': 'high'
                })

            if market_context.get('new_markets', []):
                opportunities.append({
                    'factor': 'New product/services',
                    'evidence': f"Markets: {', '.join(market_context.get('new_markets', [])[:3])}",
                    'impact': 'high'
                })

            if market_context.get('personalization_demand', 0) > 0.5:
                opportunities.append({
                    'factor': 'Scale & personalization',
                    'evidence': 'Mass customization opportunity',
                    'impact': 'medium'
                })

            if market_context.get('ai_first_advantage', False):
                opportunities.append({
                    'factor': 'Strategic differentiation',
                    'evidence': 'AI-first competitive advantage',
                    'impact': 'high'
                })

        # External Analysis - Threats
        if market_context:
            if market_context.get('regulatory_pressure', 0) > 0.5:
                threats.append({
                    'factor': 'Regulatory pressure',
                    'evidence': 'Compliance & liability risks',
                    'impact': 'high'
                })

            if market_context.get('ethics_risk', 0) > 0.3:
                threats.append({
                    'factor': 'Ethical & trust risks',
                    'evidence': 'Bias, harm, misuse concerns',
                    'impact': 'high'
                })

            if market_context.get('commoditization_risk', 0) > 0.5:
                threats.append({
                    'factor': 'Competitive imitation',
                    'evidence': 'Model commoditization risk',
                    'impact': 'medium'
                })

            if market_context.get('security_threat_level', 0) > 0.3:
                threats.append({
                    'factor': 'Security & adversarial risk',
                    'evidence': 'Model abuse, leakage threats',
                    'impact': 'high'
                })

        return {
            'swot_analysis': {
                'strengths': strengths,
                'weaknesses': weaknesses,
                'opportunities': opportunities,
                'threats': threats
            },
            'summary': {
                'strength_count': len(strengths),
                'weakness_count': len(weaknesses),
                'opportunity_count': len(opportunities),
                'threat_count': len(threats)
            },
            'internal_balance': len(strengths) - len(weaknesses),
            'external_balance': len(opportunities) - len(threats)
        }


# ============================================================================
# AI-Specific SWOT Analyzer
# ============================================================================

class AISWOTAnalyzer:
    """AI-specific detailed SWOT analysis."""

    def analyze_ai_swot(self,
                       technical_metrics: Dict[str, Any],
                       operational_metrics: Dict[str, Any],
                       market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-specific SWOT analysis with detailed breakdowns."""
        swot = {
            'strengths': {
                'technical_capability': self._assess_technical_strengths(technical_metrics),
                'data_advantage': self._assess_data_advantage(technical_metrics),
                'operational_efficiency': self._assess_operational_efficiency(operational_metrics),
                'decision_consistency': self._assess_decision_consistency(operational_metrics)
            },
            'weaknesses': {
                'data_dependency': self._assess_data_dependency(technical_metrics),
                'explainability_gaps': self._assess_explainability_gaps(technical_metrics),
                'generalization_limits': self._assess_generalization_limits(technical_metrics),
                'maintenance_burden': self._assess_maintenance_burden(operational_metrics)
            },
            'opportunities': {
                'new_products': self._assess_product_opportunities(market_metrics),
                'scale_personalization': self._assess_scale_opportunities(market_metrics),
                'process_redesign': self._assess_process_opportunities(operational_metrics),
                'strategic_differentiation': self._assess_differentiation(market_metrics)
            },
            'threats': {
                'regulatory_pressure': self._assess_regulatory_threats(market_metrics),
                'ethical_trust_risks': self._assess_ethical_threats(market_metrics),
                'competitive_imitation': self._assess_competitive_threats(market_metrics),
                'security_adversarial': self._assess_security_threats(technical_metrics)
            }
        }

        # Generate artifacts
        artifacts = {
            'model_benchmark_report': self._generate_benchmark_report(technical_metrics),
            'data_moat_assessment': self._generate_data_moat(technical_metrics),
            'roi_metrics': self._generate_roi_metrics(operational_metrics),
            'regulatory_risk_register': self._generate_risk_register(market_metrics)
        }

        return {
            'ai_specific_swot': swot,
            'artifacts': artifacts,
            'analysis_date': datetime.now().isoformat()
        }

    def _assess_technical_strengths(self, metrics: Dict) -> Dict[str, Any]:
        accuracy = metrics.get('accuracy', 0)
        speed = metrics.get('inference_speed_ms', 1000)
        scalability = metrics.get('scalability', 0)

        return {
            'accuracy': {'value': accuracy, 'strength': accuracy > 0.9},
            'speed': {'value': speed, 'strength': speed < 100},
            'scalability': {'value': scalability, 'strength': scalability > 0.8}
        }

    def _assess_data_advantage(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'proprietary_data': metrics.get('proprietary_data', False),
            'data_quality_score': metrics.get('data_quality', 0),
            'data_volume': metrics.get('data_volume', 0)
        }

    def _assess_operational_efficiency(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'cost_reduction': metrics.get('cost_reduction_percent', 0),
            'automation_rate': metrics.get('automation_rate', 0)
        }

    def _assess_decision_consistency(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'variance_reduction': metrics.get('decision_variance_reduction', 0),
            'consistency_score': metrics.get('consistency_score', 0)
        }

    def _assess_data_dependency(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'drift_sensitivity': metrics.get('drift_sensitivity', 0),
            'data_quality_impact': metrics.get('data_quality_impact', 0)
        }

    def _assess_explainability_gaps(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'explainability_score': metrics.get('explainability_score', 0),
            'is_black_box': metrics.get('is_black_box', True)
        }

    def _assess_generalization_limits(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'ood_performance': metrics.get('ood_robustness', 0),
            'robustness_score': metrics.get('robustness_score', 0)
        }

    def _assess_maintenance_burden(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'retraining_frequency': metrics.get('retraining_frequency', 0),
            'monitoring_cost': metrics.get('monitoring_cost', 0)
        }

    def _assess_product_opportunities(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'new_markets': metrics.get('new_markets', []),
            'opportunity_pipeline': metrics.get('opportunity_pipeline', [])
        }

    def _assess_scale_opportunities(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'personalization_potential': metrics.get('personalization_demand', 0),
            'growth_projections': metrics.get('growth_projections', {})
        }

    def _assess_process_opportunities(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'automation_potential': metrics.get('automation_potential', 0),
            'process_redesign_opportunities': metrics.get('redesign_opportunities', [])
        }

    def _assess_differentiation(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'ai_first_advantage': metrics.get('ai_first_advantage', False),
            'competitive_positioning': metrics.get('competitive_positioning', 'neutral')
        }

    def _assess_regulatory_threats(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'regulatory_pressure': metrics.get('regulatory_pressure', 0),
            'compliance_requirements': metrics.get('compliance_requirements', [])
        }

    def _assess_ethical_threats(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'ethics_risk': metrics.get('ethics_risk', 0),
            'trust_concerns': metrics.get('trust_concerns', [])
        }

    def _assess_competitive_threats(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'commoditization_risk': metrics.get('commoditization_risk', 0),
            'competitor_analysis': metrics.get('competitors', [])
        }

    def _assess_security_threats(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'security_threat_level': metrics.get('security_threat_level', 0),
            'adversarial_risk': metrics.get('adversarial_risk', 0)
        }

    def _generate_benchmark_report(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'accuracy': metrics.get('accuracy', 0),
            'latency_ms': metrics.get('inference_speed_ms', 0),
            'throughput': metrics.get('throughput', 0)
        }

    def _generate_data_moat(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'proprietary_score': 1.0 if metrics.get('proprietary_data') else 0.0,
            'data_quality': metrics.get('data_quality', 0),
            'moat_strength': 'strong' if metrics.get('proprietary_data') else 'weak'
        }

    def _generate_roi_metrics(self, metrics: Dict) -> Dict[str, Any]:
        return {
            'cost_savings': metrics.get('cost_reduction_percent', 0),
            'productivity_gain': metrics.get('productivity_gain', 0)
        }

    def _generate_risk_register(self, metrics: Dict) -> List[Dict[str, Any]]:
        risks = []
        if metrics.get('regulatory_pressure', 0) > 0.5:
            risks.append({'risk': 'Regulatory compliance', 'severity': 'high'})
        if metrics.get('ethics_risk', 0) > 0.3:
            risks.append({'risk': 'Ethical concerns', 'severity': 'medium'})
        return risks


# ============================================================================
# Lifecycle Stage SWOT Analyzer
# ============================================================================

class LifecycleSWOTAnalyzer:
    """SWOT analysis by AI lifecycle stage."""

    LIFECYCLE_STAGES = ['data', 'model', 'deployment', 'monitoring', 'governance']

    def analyze_by_stage(self,
                        stage_assessments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform SWOT analysis for each lifecycle stage."""
        lifecycle_swot = {}

        for stage in self.LIFECYCLE_STAGES:
            assessment = stage_assessments.get(stage, {})
            lifecycle_swot[stage] = self._analyze_stage(stage, assessment)

        # Cross-stage analysis
        cross_stage = self._analyze_cross_stage(lifecycle_swot)

        return {
            'lifecycle_swot': lifecycle_swot,
            'cross_stage_analysis': cross_stage,
            'overall_health': self._calculate_health_score(lifecycle_swot)
        }

    def _analyze_stage(self, stage: str, assessment: Dict) -> Dict[str, Any]:
        stage_swot = {
            'data': {
                'strengths': ['Scale', 'Richness'] if assessment.get('data_quality', 0) > 0.7 else [],
                'weaknesses': ['Bias'] if assessment.get('bias_risk', 0) > 0.3 else [],
                'opportunities': ['New signals'] if assessment.get('new_data_sources', []) else [],
                'threats': ['Privacy laws'] if assessment.get('privacy_risk', 0) > 0.3 else []
            },
            'model': {
                'strengths': ['Accuracy', 'Speed'] if assessment.get('accuracy', 0) > 0.9 else [],
                'weaknesses': ['Overfitting'] if assessment.get('overfit_risk', 0) > 0.3 else [],
                'opportunities': ['Better architectures'] if assessment.get('architecture_opportunities', []) else [],
                'threats': ['Model theft'] if assessment.get('theft_risk', 0) > 0.3 else []
            },
            'deployment': {
                'strengths': ['Automation'] if assessment.get('automation_level', 0) > 0.7 else [],
                'weaknesses': ['Drift'] if assessment.get('drift_detected', False) else [],
                'opportunities': ['New workflows'] if assessment.get('workflow_opportunities', []) else [],
                'threats': ['Operational failure'] if assessment.get('failure_risk', 0) > 0.3 else []
            },
            'monitoring': {
                'strengths': ['Early detection'] if assessment.get('monitoring_coverage', 0) > 0.8 else [],
                'weaknesses': ['Alert fatigue'] if assessment.get('alert_fatigue', False) else [],
                'opportunities': ['Trust building'] if assessment.get('trust_opportunity', False) else [],
                'threats': ['Silent decay'] if assessment.get('decay_risk', 0) > 0.3 else []
            },
            'governance': {
                'strengths': ['Control & trust'] if assessment.get('governance_score', 0) > 0.8 else [],
                'weaknesses': ['Slowness'] if assessment.get('governance_overhead', 0) > 0.5 else [],
                'opportunities': ['Regulator confidence'] if assessment.get('regulatory_opportunity', False) else [],
                'threats': ['Non-compliance'] if assessment.get('compliance_risk', 0) > 0.3 else []
            }
        }

        return stage_swot.get(stage, {
            'strengths': [],
            'weaknesses': [],
            'opportunities': [],
            'threats': []
        })

    def _analyze_cross_stage(self, lifecycle_swot: Dict) -> Dict[str, Any]:
        # Identify patterns across stages
        total_strengths = sum(len(s.get('strengths', [])) for s in lifecycle_swot.values())
        total_weaknesses = sum(len(s.get('weaknesses', [])) for s in lifecycle_swot.values())
        total_opportunities = sum(len(s.get('opportunities', [])) for s in lifecycle_swot.values())
        total_threats = sum(len(s.get('threats', [])) for s in lifecycle_swot.values())

        return {
            'total_strengths': total_strengths,
            'total_weaknesses': total_weaknesses,
            'total_opportunities': total_opportunities,
            'total_threats': total_threats,
            'strongest_stage': max(lifecycle_swot.items(),
                                  key=lambda x: len(x[1].get('strengths', [])))[0] if lifecycle_swot else None,
            'weakest_stage': max(lifecycle_swot.items(),
                                key=lambda x: len(x[1].get('weaknesses', [])))[0] if lifecycle_swot else None
        }

    def _calculate_health_score(self, lifecycle_swot: Dict) -> float:
        total_positive = sum(
            len(s.get('strengths', [])) + len(s.get('opportunities', []))
            for s in lifecycle_swot.values()
        )
        total_negative = sum(
            len(s.get('weaknesses', [])) + len(s.get('threats', []))
            for s in lifecycle_swot.values()
        )

        total = total_positive + total_negative
        return total_positive / total if total > 0 else 0.5


# ============================================================================
# Strategic Matrix Analyzers
# ============================================================================

class StrategicMatrixAnalyzer:
    """Generates strategic matrices from SWOT analysis."""

    def generate_tows_matrix(self,
                            swot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate TOWS strategic options matrix."""
        strengths = swot_analysis.get('strengths', [])
        weaknesses = swot_analysis.get('weaknesses', [])
        opportunities = swot_analysis.get('opportunities', [])
        threats = swot_analysis.get('threats', [])

        # SO Strategies (Strengths-Opportunities)
        so_strategies = []
        for s in strengths[:3]:
            for o in opportunities[:3]:
                so_strategies.append({
                    'strategy': f"Use {s.get('factor', '')} to capture {o.get('factor', '')}",
                    'type': 'SO',
                    'priority': 'high'
                })

        # WO Strategies (Weaknesses-Opportunities)
        wo_strategies = []
        for w in weaknesses[:3]:
            for o in opportunities[:3]:
                wo_strategies.append({
                    'strategy': f"Address {w.get('factor', '')} to enable {o.get('factor', '')}",
                    'type': 'WO',
                    'priority': 'medium'
                })

        # ST Strategies (Strengths-Threats)
        st_strategies = []
        for s in strengths[:3]:
            for t in threats[:3]:
                st_strategies.append({
                    'strategy': f"Use {s.get('factor', '')} to counter {t.get('factor', '')}",
                    'type': 'ST',
                    'priority': 'medium'
                })

        # WT Strategies (Weaknesses-Threats)
        wt_strategies = []
        for w in weaknesses[:3]:
            for t in threats[:3]:
                wt_strategies.append({
                    'strategy': f"Minimize {w.get('factor', '')} to avoid {t.get('factor', '')}",
                    'type': 'WT',
                    'priority': 'high'
                })

        return {
            'tows_matrix': {
                'so_strategies': so_strategies[:5],
                'wo_strategies': wo_strategies[:5],
                'st_strategies': st_strategies[:5],
                'wt_strategies': wt_strategies[:5]
            },
            'strategic_priorities': self._prioritize_strategies(
                so_strategies + wo_strategies + st_strategies + wt_strategies
            )
        }

    def _prioritize_strategies(self, strategies: List[Dict]) -> List[Dict]:
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_strategies = sorted(strategies, key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        return sorted_strategies[:10]

    def generate_ie_matrix(self,
                          internal_score: float,
                          external_score: float) -> Dict[str, Any]:
        """Generate Internal-External (IE) Matrix positioning."""
        # IE Matrix quadrants
        if internal_score >= 2.5 and external_score >= 2.5:
            position = 'Grow and Build'
            recommendation = 'Intensive or integrative strategies'
        elif internal_score >= 2.5 and external_score < 2.5:
            position = 'Hold and Maintain'
            recommendation = 'Market penetration, product development'
        elif internal_score < 2.5 and external_score >= 2.5:
            position = 'Hold and Maintain'
            recommendation = 'Selective investment'
        else:
            position = 'Harvest or Divest'
            recommendation = 'Defensive strategies'

        return {
            'ie_matrix': {
                'internal_score': float(internal_score),
                'external_score': float(external_score),
                'position': position,
                'recommendation': recommendation
            }
        }


# ============================================================================
# Competitive Analysis
# ============================================================================

class CompetitiveAnalyzer:
    """Analyzes competitive positioning using SWOT."""

    def analyze_competitive_position(self,
                                    our_swot: Dict[str, Any],
                                    competitor_swots: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze competitive position relative to competitors."""
        # Calculate our SWOT score
        our_score = self._calculate_swot_score(our_swot)

        competitor_analysis = []
        if competitor_swots:
            for i, comp_swot in enumerate(competitor_swots):
                comp_score = self._calculate_swot_score(comp_swot)
                competitor_analysis.append({
                    'competitor_id': comp_swot.get('competitor_id', f'competitor_{i}'),
                    'swot_score': comp_score,
                    'relative_position': 'ahead' if our_score > comp_score else 'behind'
                })

        # Defensibility analysis
        defensibility = self._analyze_defensibility(our_swot)

        return {
            'competitive_positioning': {
                'our_swot_score': our_score,
                'competitor_analysis': competitor_analysis,
                'market_position': 'leader' if all(c['relative_position'] == 'ahead' for c in competitor_analysis) else 'challenger'
            },
            'defensibility_analysis': defensibility
        }

    def _calculate_swot_score(self, swot: Dict) -> float:
        strengths = len(swot.get('strengths', []))
        weaknesses = len(swot.get('weaknesses', []))
        opportunities = len(swot.get('opportunities', []))
        threats = len(swot.get('threats', []))

        # Weighted score
        return (strengths * 1.0 + opportunities * 0.8 - weaknesses * 0.9 - threats * 0.7)

    def _analyze_defensibility(self, swot: Dict) -> Dict[str, Any]:
        strengths = swot.get('strengths', [])

        moat_factors = []
        for s in strengths:
            factor = s.get('factor', '').lower()
            if 'proprietary' in factor or 'unique' in factor:
                moat_factors.append('Data moat')
            if 'accuracy' in factor or 'performance' in factor:
                moat_factors.append('Performance advantage')

        return {
            'moat_factors': moat_factors,
            'defensibility_score': len(moat_factors) / 5,  # Normalize
            'at_risk': len(moat_factors) < 2
        }


# ============================================================================
# Report Generator
# ============================================================================

class SWOTReportGenerator:
    """Generates comprehensive SWOT analysis reports."""

    def __init__(self):
        self.classic_analyzer = ClassicSWOTAnalyzer()
        self.ai_analyzer = AISWOTAnalyzer()
        self.lifecycle_analyzer = LifecycleSWOTAnalyzer()
        self.strategic_analyzer = StrategicMatrixAnalyzer()
        self.competitive_analyzer = CompetitiveAnalyzer()

    def generate_full_report(self,
                            ai_system_assessment: Dict[str, Any],
                            market_context: Dict[str, Any] = None,
                            stage_assessments: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive SWOT report."""
        report = {
            'report_type': 'swot_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        # Classic SWOT
        classic_swot = self.classic_analyzer.analyze_swot(ai_system_assessment, market_context)
        report['classic_swot'] = classic_swot

        # Generate TOWS matrix
        if 'swot_analysis' in classic_swot:
            report['tows_matrix'] = self.strategic_analyzer.generate_tows_matrix(classic_swot['swot_analysis'])

        # Lifecycle SWOT
        if stage_assessments:
            report['lifecycle_swot'] = self.lifecycle_analyzer.analyze_by_stage(stage_assessments)

        # Competitive analysis
        report['competitive_position'] = self.competitive_analyzer.analyze_competitive_position(
            classic_swot.get('swot_analysis', {})
        )

        # Calculate overall strategic health
        internal_balance = classic_swot.get('internal_balance', 0)
        external_balance = classic_swot.get('external_balance', 0)
        report['strategic_health'] = {
            'internal_balance': internal_balance,
            'external_balance': external_balance,
            'overall_position': 'strong' if internal_balance > 0 and external_balance > 0 else 'challenged'
        }

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
        md = "# SWOT Analysis Report\n\n"
        md += f"Generated: {report.get('timestamp', 'N/A')}\n\n"

        if 'classic_swot' in report:
            swot = report['classic_swot'].get('swot_analysis', {})
            md += "## Strengths\n"
            for s in swot.get('strengths', []):
                md += f"- **{s.get('factor', '')}**: {s.get('evidence', '')}\n"

            md += "\n## Weaknesses\n"
            for w in swot.get('weaknesses', []):
                md += f"- **{w.get('factor', '')}**: {w.get('evidence', '')}\n"

            md += "\n## Opportunities\n"
            for o in swot.get('opportunities', []):
                md += f"- **{o.get('factor', '')}**: {o.get('evidence', '')}\n"

            md += "\n## Threats\n"
            for t in swot.get('threats', []):
                md += f"- **{t.get('factor', '')}**: {t.get('evidence', '')}\n"

        return md
