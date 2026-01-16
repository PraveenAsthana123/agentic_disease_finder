"""
Hallucination Prevention AI Analysis Module
============================================

Comprehensive analysis for AI hallucination prevention and detection.
Implements 20 analysis types for hallucination governance.

Analysis Types:
1. Hallucination Definition & Scope Analysis
2. Use-Case Risk Sensitivity Analysis
3. Knowledge Boundary Analysis
4. Prompt & Instruction Robustness Analysis
5. Retrieval Grounding Analysis (RAG)
6. Source Attribution & Citation Analysis
7. Faithfulness (Answer-Context) Analysis
8. Reasoning Chain Reliability Analysis
9. Uncertainty & Abstention Handling Analysis
10. Over-Generalization & Extrapolation Analysis
11. Fine-Tuning & Alignment Impact Analysis
12. Tool-Use & External Dependency Analysis
13. Consistency & Self-Contradiction Analysis
14. Adversarial & Stress Hallucination Tests
15. Human-in-the-Loop Validation Analysis
16. Monitoring & Hallucination Drift Analysis
17. User Experience & Trust Impact Analysis
18. Incident Response & Correction Analysis
19. Evaluation Metrics & Benchmarks
20. Hallucination Governance & Accountability
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import json
import re


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HallucinationType:
    """Classification of hallucination types."""
    fabricated_facts: bool = False
    unsupported_claims: bool = False
    false_citations: bool = False
    overconfident_uncertainty: bool = False
    temporal_confusion: bool = False
    entity_confusion: bool = False


@dataclass
class HallucinationInstance:
    """Single instance of detected hallucination."""
    instance_id: str
    hallucination_type: str
    content: str
    confidence: float = 0.0
    context: str = ""
    detected_at: Optional[datetime] = None
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class GroundingResult:
    """Result of grounding analysis."""
    query_id: str
    is_grounded: bool = False
    grounding_score: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)


@dataclass
class FaithfulnessScore:
    """Faithfulness analysis result."""
    output_id: str
    faithfulness_score: float = 0.0
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0


# ============================================================================
# Hallucination Definition & Scope (Type 1)
# ============================================================================

class HallucinationScopeAnalyzer:
    """Analyzes hallucination definition and scope."""

    HALLUCINATION_TAXONOMY = {
        'fabricated_facts': 'Information presented as fact with no basis in training or context',
        'unsupported_claims': 'Claims not supported by provided context or evidence',
        'false_citations': 'References to non-existent sources or misattributed quotes',
        'overconfident_uncertainty': 'Expressing certainty about uncertain or unknown information',
        'temporal_confusion': 'Incorrect time references or anachronisms',
        'entity_confusion': 'Mixing up entities, people, or concepts'
    }

    def analyze_scope(self,
                     use_case_context: Dict[str, Any],
                     sample_outputs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Define what counts as hallucination for this context."""
        task_type = use_case_context.get('task_type', 'general')
        domain = use_case_context.get('domain', 'general')

        # Determine relevant hallucination types
        relevant_types = list(self.HALLUCINATION_TAXONOMY.keys())

        # Domain-specific adjustments
        if domain == 'medical':
            severity_weights = {'fabricated_facts': 1.0, 'false_citations': 0.9}
        elif domain == 'legal':
            severity_weights = {'false_citations': 1.0, 'unsupported_claims': 0.9}
        elif domain == 'creative':
            severity_weights = {'fabricated_facts': 0.3, 'entity_confusion': 0.5}
        else:
            severity_weights = {t: 0.7 for t in relevant_types}

        # Analyze sample outputs if provided
        detected_types = defaultdict(int)
        if sample_outputs:
            for output in sample_outputs:
                for h_type in output.get('hallucination_types', []):
                    detected_types[h_type] += 1

        return {
            'hallucination_taxonomy': self.HALLUCINATION_TAXONOMY,
            'scope_definition': {
                'task_type': task_type,
                'domain': domain,
                'relevant_types': relevant_types,
                'severity_weights': severity_weights
            },
            'detected_type_distribution': dict(detected_types),
            'scope_statement': f"Hallucination scope for {domain} {task_type} tasks"
        }


# ============================================================================
# Risk Sensitivity Analysis (Type 2)
# ============================================================================

class RiskSensitivityAnalyzer:
    """Analyzes how dangerous hallucinations are in specific tasks."""

    def analyze_risk_sensitivity(self,
                                use_case: Dict[str, Any],
                                historical_incidents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze risk sensitivity of hallucinations."""
        decision_criticality = use_case.get('decision_criticality', 'medium')
        user_reliance = use_case.get('user_reliance', 'moderate')
        reversibility = use_case.get('harm_reversibility', 'reversible')

        # Calculate base risk score
        criticality_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
        reliance_scores = {'low': 0.2, 'moderate': 0.5, 'high': 0.8, 'complete': 1.0}
        reversibility_scores = {'easily_reversible': 0.2, 'reversible': 0.4,
                               'difficult_reversible': 0.7, 'irreversible': 1.0}

        risk_score = (
            criticality_scores.get(decision_criticality, 0.5) * 0.4 +
            reliance_scores.get(user_reliance, 0.5) * 0.3 +
            reversibility_scores.get(reversibility, 0.5) * 0.3
        )

        # Analyze historical incidents
        incident_severity = 0
        if historical_incidents:
            severity_values = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            incident_severity = np.mean([
                severity_values.get(i.get('severity', 'medium'), 2)
                for i in historical_incidents
            ])

        # Risk classification
        if risk_score >= 0.8 or incident_severity >= 3:
            risk_class = 'critical'
        elif risk_score >= 0.6:
            risk_class = 'high'
        elif risk_score >= 0.4:
            risk_class = 'medium'
        else:
            risk_class = 'low'

        return {
            'risk_classification': risk_class,
            'risk_score': float(risk_score),
            'factors': {
                'decision_criticality': decision_criticality,
                'user_reliance': user_reliance,
                'harm_reversibility': reversibility
            },
            'historical_incident_severity': float(incident_severity),
            'incident_count': len(historical_incidents) if historical_incidents else 0,
            'mitigation_priority': 'immediate' if risk_class in ['critical', 'high'] else 'standard'
        }


# ============================================================================
# Knowledge Boundary Analysis (Type 3)
# ============================================================================

class KnowledgeBoundaryAnalyzer:
    """Analyzes what the model truly knows vs doesn't know."""

    def analyze_knowledge_boundary(self,
                                  knowledge_config: Dict[str, Any],
                                  query_samples: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze knowledge boundaries."""
        training_cutoff = knowledge_config.get('training_cutoff_date')
        domain_coverage = knowledge_config.get('domain_coverage', {})
        known_gaps = knowledge_config.get('known_gaps', [])

        # Analyze query samples for boundary violations
        boundary_violations = []
        if query_samples:
            for query in query_samples:
                query_date = query.get('requires_info_after')
                if query_date and training_cutoff:
                    if query_date > training_cutoff:
                        boundary_violations.append({
                            'query': query.get('text', ''),
                            'violation_type': 'temporal',
                            'required_date': str(query_date)
                        })

                query_domain = query.get('domain')
                if query_domain and query_domain in known_gaps:
                    boundary_violations.append({
                        'query': query.get('text', ''),
                        'violation_type': 'domain_gap',
                        'domain': query_domain
                    })

        return {
            'knowledge_boundary_statement': {
                'training_cutoff': str(training_cutoff) if training_cutoff else None,
                'domain_coverage': domain_coverage,
                'known_gaps': known_gaps
            },
            'boundary_violations': boundary_violations,
            'violation_rate': len(boundary_violations) / len(query_samples) if query_samples else 0,
            'recommendations': self._generate_recommendations(boundary_violations)
        }

    def _generate_recommendations(self, violations: List) -> List[str]:
        recs = []
        temporal_count = sum(1 for v in violations if v.get('violation_type') == 'temporal')
        if temporal_count > 0:
            recs.append(f"Add temporal awareness disclaimers ({temporal_count} violations)")
        domain_count = sum(1 for v in violations if v.get('violation_type') == 'domain_gap')
        if domain_count > 0:
            recs.append(f"Implement domain-specific abstention ({domain_count} gaps)")
        return recs


# ============================================================================
# Prompt Robustness Analysis (Type 4)
# ============================================================================

class PromptRobustnessAnalyzer:
    """Analyzes if prompts encourage hallucination."""

    def analyze_prompt_robustness(self,
                                 prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prompt characteristics for hallucination risk."""
        if not prompts:
            return {'robustness_score': 0, 'risk_prompts': []}

        risk_factors = []
        for prompt in prompts:
            text = prompt.get('text', '')
            risks = []

            # Check for ambiguity
            if '?' not in text and len(text) < 20:
                risks.append('ambiguous')

            # Check for leading questions
            leading_patterns = ['isn\'t it true', 'don\'t you think', 'surely']
            if any(p in text.lower() for p in leading_patterns):
                risks.append('leading')

            # Check for over-specification
            if len(text.split()) > 100:
                risks.append('over_specified')

            # Check for requests beyond knowledge
            beyond_patterns = ['latest', 'current', 'today', 'right now']
            if any(p in text.lower() for p in beyond_patterns):
                risks.append('beyond_knowledge')

            if risks:
                risk_factors.append({
                    'prompt': text[:100] + '...' if len(text) > 100 else text,
                    'risks': risks
                })

        robustness_score = 1 - (len(risk_factors) / len(prompts)) if prompts else 0

        return {
            'robustness_score': float(robustness_score),
            'prompts_analyzed': len(prompts),
            'risk_prompts': risk_factors,
            'risk_prompt_count': len(risk_factors),
            'risk_assessment': {
                'ambiguous_count': sum(1 for r in risk_factors if 'ambiguous' in r['risks']),
                'leading_count': sum(1 for r in risk_factors if 'leading' in r['risks']),
                'over_specified_count': sum(1 for r in risk_factors if 'over_specified' in r['risks']),
                'beyond_knowledge_count': sum(1 for r in risk_factors if 'beyond_knowledge' in r['risks'])
            }
        }


# ============================================================================
# Retrieval Grounding Analysis (Type 5)
# ============================================================================

class RetrievalGroundingAnalyzer:
    """Analyzes if generation is grounded in evidence (RAG)."""

    def analyze_grounding(self,
                         query_context_pairs: List[Dict[str, Any]],
                         outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze retrieval grounding quality."""
        if not query_context_pairs or not outputs:
            return {'grounding_score': 0, 'grounding_quality_report': {}}

        grounding_results = []
        for i, (qc, output) in enumerate(zip(query_context_pairs, outputs)):
            context = qc.get('context', '')
            answer = output.get('answer', '')

            # Simple overlap analysis
            context_words = set(context.lower().split())
            answer_words = set(answer.lower().split())

            if context_words:
                overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0
            else:
                overlap = 0

            # Check for missing evidence
            claims = output.get('claims', [])
            supported_claims = sum(1 for c in claims if c.lower() in context.lower())

            grounding_results.append({
                'query_id': qc.get('query_id', str(i)),
                'overlap_score': float(overlap),
                'total_claims': len(claims),
                'supported_claims': supported_claims,
                'grounding_score': supported_claims / len(claims) if claims else overlap
            })

        avg_grounding = np.mean([r['grounding_score'] for r in grounding_results])

        return {
            'grounding_score': float(avg_grounding),
            'grounding_quality_report': {
                'total_queries': len(grounding_results),
                'well_grounded': sum(1 for r in grounding_results if r['grounding_score'] > 0.7),
                'poorly_grounded': sum(1 for r in grounding_results if r['grounding_score'] < 0.3),
                'avg_overlap': float(np.mean([r['overlap_score'] for r in grounding_results]))
            },
            'detailed_results': grounding_results[:10],  # First 10 for detail
            'retriever_recommendations': self._recommend_retriever_improvements(grounding_results)
        }

    def _recommend_retriever_improvements(self, results: List) -> List[str]:
        recs = []
        poor_count = sum(1 for r in results if r['grounding_score'] < 0.3)
        if poor_count > len(results) * 0.2:
            recs.append("Improve retriever recall - many answers lack supporting context")
        return recs


# ============================================================================
# Source Attribution Analysis (Type 6)
# ============================================================================

class SourceAttributionAnalyzer:
    """Analyzes if sources are real and correct."""

    def analyze_attribution(self,
                           outputs_with_citations: List[Dict[str, Any]],
                           known_sources: Set[str] = None) -> Dict[str, Any]:
        """Analyze source attribution accuracy."""
        if not outputs_with_citations:
            return {'attribution_accuracy': 0, 'source_audit': {}}

        known_sources = known_sources or set()

        verified_citations = 0
        fabricated_citations = 0
        misattributed_quotes = 0
        total_citations = 0

        citation_audit = []
        for output in outputs_with_citations:
            citations = output.get('citations', [])
            for citation in citations:
                total_citations += 1
                source = citation.get('source', '')
                quote = citation.get('quote', '')

                # Check if source is known
                if source in known_sources:
                    verified_citations += 1
                    status = 'verified'
                elif citation.get('verified', False):
                    verified_citations += 1
                    status = 'verified'
                else:
                    fabricated_citations += 1
                    status = 'unverified'

                # Check quote faithfulness
                if citation.get('quote_mismatch', False):
                    misattributed_quotes += 1

                citation_audit.append({
                    'source': source,
                    'status': status,
                    'quote_faithful': not citation.get('quote_mismatch', False)
                })

        attribution_accuracy = verified_citations / total_citations if total_citations > 0 else 1.0

        return {
            'attribution_accuracy': float(attribution_accuracy),
            'source_audit': {
                'total_citations': total_citations,
                'verified_citations': verified_citations,
                'fabricated_citations': fabricated_citations,
                'misattributed_quotes': misattributed_quotes
            },
            'citation_details': citation_audit[:20],  # First 20
            'fabrication_rate': fabricated_citations / total_citations if total_citations > 0 else 0
        }


# ============================================================================
# Faithfulness Analysis (Type 7)
# ============================================================================

class FaithfulnessAnalyzer:
    """Analyzes if answer stays within evidence."""

    def analyze_faithfulness(self,
                            context_answer_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze faithfulness of answers to context."""
        if not context_answer_pairs:
            return {'faithfulness_score': 0, 'faithfulness_report': {}}

        faithfulness_scores = []
        for pair in context_answer_pairs:
            context = pair.get('context', '').lower()
            answer = pair.get('answer', '').lower()

            # Calculate entailment-like score
            answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
            supported_sentences = 0

            for sent in answer_sentences:
                sent_words = set(sent.split())
                context_words = set(context.split())
                overlap = len(sent_words & context_words) / len(sent_words) if sent_words else 0
                if overlap > 0.3:  # Threshold for "supported"
                    supported_sentences += 1

            faithfulness = supported_sentences / len(answer_sentences) if answer_sentences else 1

            faithfulness_scores.append({
                'output_id': pair.get('id', ''),
                'faithfulness_score': float(faithfulness),
                'total_claims': len(answer_sentences),
                'supported_claims': supported_sentences
            })

        avg_faithfulness = np.mean([f['faithfulness_score'] for f in faithfulness_scores])

        return {
            'faithfulness_score': float(avg_faithfulness),
            'faithfulness_report': {
                'total_outputs': len(faithfulness_scores),
                'highly_faithful': sum(1 for f in faithfulness_scores if f['faithfulness_score'] > 0.8),
                'unfaithful': sum(1 for f in faithfulness_scores if f['faithfulness_score'] < 0.5)
            },
            'detailed_scores': faithfulness_scores[:10]
        }


# ============================================================================
# Reasoning Chain Reliability (Type 8)
# ============================================================================

class ReasoningChainAnalyzer:
    """Analyzes reasoning chain reliability."""

    def analyze_reasoning(self,
                         reasoning_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if reasoning is internally consistent."""
        if not reasoning_chains:
            return {'reasoning_integrity': 0, 'chain_analysis': []}

        chain_analyses = []
        for chain in reasoning_chains:
            steps = chain.get('steps', [])
            if len(steps) < 2:
                continue

            # Check logical coherence between steps
            coherent_transitions = 0
            contradictions = 0

            for i in range(1, len(steps)):
                prev_step = steps[i-1].get('content', '').lower()
                curr_step = steps[i].get('content', '').lower()

                # Simple coherence check - shared concepts
                prev_words = set(prev_step.split())
                curr_words = set(curr_step.split())
                overlap = len(prev_words & curr_words)

                if overlap > 0:
                    coherent_transitions += 1

                # Check for contradictions
                negation_words = ['not', 'never', 'no', 'cannot', 'won\'t']
                has_negation_flip = (
                    any(w in curr_step for w in negation_words) !=
                    any(w in prev_step for w in negation_words)
                )
                if has_negation_flip and overlap > 2:
                    contradictions += 1

            total_transitions = len(steps) - 1
            coherence_score = coherent_transitions / total_transitions if total_transitions > 0 else 1

            chain_analyses.append({
                'chain_id': chain.get('id', ''),
                'step_count': len(steps),
                'coherent_transitions': coherent_transitions,
                'contradictions': contradictions,
                'coherence_score': float(coherence_score)
            })

        avg_integrity = np.mean([c['coherence_score'] for c in chain_analyses]) if chain_analyses else 0

        return {
            'reasoning_integrity': float(avg_integrity),
            'chain_analysis': chain_analyses,
            'integrity_assessment': {
                'total_chains': len(chain_analyses),
                'highly_coherent': sum(1 for c in chain_analyses if c['coherence_score'] > 0.8),
                'with_contradictions': sum(1 for c in chain_analyses if c['contradictions'] > 0)
            }
        }


# ============================================================================
# Uncertainty & Abstention Analysis (Type 9)
# ============================================================================

class UncertaintyAbstentionAnalyzer:
    """Analyzes uncertainty handling and abstention behavior."""

    def analyze_abstention(self,
                          outputs: List[Dict[str, Any]],
                          ground_truth: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if model knows when to say 'I don't know'."""
        if not outputs:
            return {'abstention_calibration': 0, 'abstention_policy': {}}

        abstentions = [o for o in outputs if o.get('abstained', False)]
        confident_answers = [o for o in outputs if not o.get('abstained', False)]

        abstention_rate = len(abstentions) / len(outputs)

        # Analyze confidence calibration if ground truth available
        calibration_score = 0
        if ground_truth:
            correct_abstentions = 0
            incorrect_confidences = 0

            for output, truth in zip(outputs, ground_truth):
                should_abstain = truth.get('should_abstain', False)
                did_abstain = output.get('abstained', False)

                if should_abstain and did_abstain:
                    correct_abstentions += 1
                elif not should_abstain and did_abstain:
                    pass  # Over-cautious but not wrong
                elif should_abstain and not did_abstain:
                    incorrect_confidences += 1

            calibration_score = 1 - (incorrect_confidences / len(outputs)) if outputs else 0

        # Analyze confidence distribution
        confidences = [o.get('confidence', 0.5) for o in confident_answers]
        avg_confidence = np.mean(confidences) if confidences else 0.5

        return {
            'abstention_calibration': float(calibration_score),
            'abstention_policy': {
                'total_outputs': len(outputs),
                'abstention_count': len(abstentions),
                'abstention_rate': float(abstention_rate),
                'avg_confidence': float(avg_confidence)
            },
            'calibration_metrics': {
                'overconfident_errors': sum(1 for o in outputs
                                           if o.get('confidence', 0) > 0.8 and o.get('incorrect', False))
            }
        }


# ============================================================================
# Over-Generalization Analysis (Type 10)
# ============================================================================

class OverGeneralizationAnalyzer:
    """Analyzes if model invents beyond data."""

    def analyze_overgeneralization(self,
                                  edge_case_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze over-generalization and extrapolation."""
        if not edge_case_tests:
            return {'overgeneralization_score': 0, 'edge_case_report': {}}

        speculative_answers = 0
        appropriate_hedging = 0

        for test in edge_case_tests:
            output = test.get('output', '')
            is_edge_case = test.get('is_edge_case', True)
            contains_speculation = test.get('contains_speculation', False)
            contains_hedging = any(h in output.lower() for h in
                                  ['might', 'possibly', 'uncertain', 'may', 'could be'])

            if is_edge_case:
                if contains_speculation:
                    speculative_answers += 1
                if contains_hedging:
                    appropriate_hedging += 1

        edge_case_count = sum(1 for t in edge_case_tests if t.get('is_edge_case', True))
        speculation_rate = speculative_answers / edge_case_count if edge_case_count > 0 else 0
        hedging_rate = appropriate_hedging / edge_case_count if edge_case_count > 0 else 0

        # Lower is better for speculation, higher is better for hedging
        overgeneralization_score = 1 - speculation_rate

        return {
            'overgeneralization_score': float(overgeneralization_score),
            'edge_case_report': {
                'total_edge_cases': edge_case_count,
                'speculative_answers': speculative_answers,
                'appropriate_hedging': appropriate_hedging,
                'speculation_rate': float(speculation_rate),
                'hedging_rate': float(hedging_rate)
            }
        }


# ============================================================================
# Fine-Tuning Impact Analysis (Type 11)
# ============================================================================

class FineTuningImpactAnalyzer:
    """Analyzes if fine-tuning increased hallucinations."""

    def analyze_finetuning_impact(self,
                                 pre_ft_metrics: Dict[str, Any],
                                 post_ft_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fine-tuning impact on hallucinations."""
        pre_hallucination_rate = pre_ft_metrics.get('hallucination_rate', 0)
        post_hallucination_rate = post_ft_metrics.get('hallucination_rate', 0)

        delta = post_hallucination_rate - pre_hallucination_rate
        regression_detected = delta > 0.05  # 5% threshold

        # Compare by hallucination type
        type_deltas = {}
        for h_type in ['fabricated_facts', 'unsupported_claims', 'false_citations']:
            pre_rate = pre_ft_metrics.get(f'{h_type}_rate', 0)
            post_rate = post_ft_metrics.get(f'{h_type}_rate', 0)
            type_deltas[h_type] = {
                'pre': float(pre_rate),
                'post': float(post_rate),
                'delta': float(post_rate - pre_rate),
                'regression': post_rate > pre_rate * 1.1
            }

        return {
            'finetuning_hallucination_delta': float(delta),
            'regression_detected': regression_detected,
            'pre_ft_rate': float(pre_hallucination_rate),
            'post_ft_rate': float(post_hallucination_rate),
            'type_breakdown': type_deltas,
            'safety_regression_test': {
                'passed': not regression_detected,
                'threshold': 0.05
            }
        }


# ============================================================================
# Tool-Use Analysis (Type 12)
# ============================================================================

class ToolUseHallucinationAnalyzer:
    """Analyzes if tools reduce or introduce hallucinations."""

    def analyze_tool_impact(self,
                           tool_usage_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze tool usage impact on hallucinations."""
        if not tool_usage_logs:
            return {'tool_grounding_score': 0, 'tool_audit': {}}

        tool_results = defaultdict(lambda: {'uses': 0, 'accurate': 0, 'hallucinations': 0})

        for log in tool_usage_logs:
            tool_name = log.get('tool_name', 'unknown')
            tool_results[tool_name]['uses'] += 1

            if log.get('result_accurate', True):
                tool_results[tool_name]['accurate'] += 1
            if log.get('introduced_hallucination', False):
                tool_results[tool_name]['hallucinations'] += 1

        # Calculate per-tool and overall scores
        tool_audit = {}
        for tool, stats in tool_results.items():
            accuracy = stats['accurate'] / stats['uses'] if stats['uses'] > 0 else 0
            hallucination_rate = stats['hallucinations'] / stats['uses'] if stats['uses'] > 0 else 0
            tool_audit[tool] = {
                'uses': stats['uses'],
                'accuracy': float(accuracy),
                'hallucination_introduction_rate': float(hallucination_rate)
            }

        total_uses = sum(t['uses'] for t in tool_results.values())
        total_accurate = sum(t['accurate'] for t in tool_results.values())
        overall_accuracy = total_accurate / total_uses if total_uses > 0 else 0

        return {
            'tool_grounding_score': float(overall_accuracy),
            'tool_audit': tool_audit,
            'summary': {
                'total_tool_uses': total_uses,
                'tools_analyzed': len(tool_results),
                'overall_accuracy': float(overall_accuracy)
            }
        }


# ============================================================================
# Consistency Analysis (Type 13)
# ============================================================================

class ConsistencyAnalyzer:
    """Analyzes if model contradicts itself."""

    def analyze_consistency(self,
                           multi_prompt_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency across prompts."""
        if not multi_prompt_tests:
            return {'consistency_score': 0, 'consistency_report': {}}

        consistent_responses = 0
        contradictions = []

        for test in multi_prompt_tests:
            responses = test.get('responses', [])
            if len(responses) < 2:
                continue

            # Check if responses are consistent
            first_response = responses[0].get('content', '').lower()
            all_consistent = True

            for resp in responses[1:]:
                other_response = resp.get('content', '').lower()

                # Simple consistency check - key facts should match
                first_facts = set(first_response.split('.')[:3])
                other_facts = set(other_response.split('.')[:3])

                # Check for direct contradictions
                if test.get('has_contradiction', False):
                    all_consistent = False
                    contradictions.append({
                        'query': test.get('query', ''),
                        'responses': [r.get('content', '')[:100] for r in responses[:2]]
                    })

            if all_consistent:
                consistent_responses += 1

        consistency_score = consistent_responses / len(multi_prompt_tests) if multi_prompt_tests else 0

        return {
            'consistency_score': float(consistency_score),
            'consistency_report': {
                'total_tests': len(multi_prompt_tests),
                'consistent': consistent_responses,
                'contradiction_count': len(contradictions)
            },
            'contradictions': contradictions[:5]  # First 5 examples
        }


# ============================================================================
# Adversarial Testing (Type 14)
# ============================================================================

class AdversarialHallucinationTester:
    """Tests adversarial hallucination scenarios."""

    def run_stress_tests(self,
                        test_prompts: List[Dict[str, Any]],
                        model_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run adversarial stress tests."""
        if not test_prompts or not model_outputs:
            return {'stress_test_score': 0, 'test_results': []}

        results = []
        passed = 0

        for prompt, output in zip(test_prompts, model_outputs):
            test_type = prompt.get('test_type', 'general')
            expected_abstention = prompt.get('should_abstain', False)
            did_abstain = output.get('abstained', False)
            hallucinated = output.get('hallucinated', False)

            test_passed = (expected_abstention == did_abstain) and not hallucinated

            results.append({
                'test_type': test_type,
                'prompt': prompt.get('text', '')[:100],
                'passed': test_passed,
                'hallucinated': hallucinated,
                'appropriate_abstention': expected_abstention == did_abstain
            })

            if test_passed:
                passed += 1

        return {
            'stress_test_score': float(passed / len(results)) if results else 0,
            'test_results': results,
            'summary': {
                'total_tests': len(results),
                'passed': passed,
                'failed': len(results) - passed,
                'hallucinations_induced': sum(1 for r in results if r['hallucinated'])
            }
        }


# ============================================================================
# HITL Validation Analysis (Type 15)
# ============================================================================

class HITLValidationAnalyzer:
    """Analyzes human-in-the-loop validation for hallucinations."""

    def analyze_hitl_validation(self,
                               validation_config: Dict[str, Any],
                               validation_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze where humans must verify outputs."""
        review_thresholds = validation_config.get('review_thresholds', {})
        high_risk_categories = validation_config.get('high_risk_categories', [])

        # Analyze validation records
        if validation_records:
            total_reviewed = len(validation_records)
            hallucinations_caught = sum(1 for r in validation_records
                                       if r.get('hallucination_detected', False))
            false_positives = sum(1 for r in validation_records
                                 if r.get('false_positive', False))

            catch_rate = hallucinations_caught / total_reviewed if total_reviewed > 0 else 0
        else:
            total_reviewed = 0
            hallucinations_caught = 0
            false_positives = 0
            catch_rate = 0

        return {
            'hitl_hallucination_control_plan': {
                'review_thresholds': review_thresholds,
                'high_risk_categories': high_risk_categories,
                'mandatory_review_triggers': validation_config.get('mandatory_triggers', [])
            },
            'validation_metrics': {
                'total_reviewed': total_reviewed,
                'hallucinations_caught': hallucinations_caught,
                'false_positives': false_positives,
                'catch_rate': float(catch_rate)
            }
        }


# ============================================================================
# Drift Monitoring (Type 16)
# ============================================================================

class HallucinationDriftMonitor:
    """Monitors hallucination drift over time."""

    def analyze_drift(self,
                     hallucination_history: List[Dict[str, Any]],
                     window_days: int = 30) -> Dict[str, Any]:
        """Analyze if hallucinations are increasing over time."""
        if not hallucination_history:
            return {'drift_detected': False, 'monitoring_dashboard': {}}

        # Sort by date
        sorted_history = sorted(hallucination_history, key=lambda x: x.get('date', datetime.now()))

        # Calculate rates over time
        rates = [h.get('hallucination_rate', 0) for h in sorted_history]

        if len(rates) < 2:
            return {'drift_detected': False, 'reason': 'insufficient_data'}

        # Compare first half to second half
        mid = len(rates) // 2
        first_half_avg = np.mean(rates[:mid])
        second_half_avg = np.mean(rates[mid:])

        drift_score = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
        drift_detected = drift_score > 0.15  # 15% increase threshold

        return {
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'monitoring_dashboard': {
                'baseline_rate': float(first_half_avg),
                'current_rate': float(second_half_avg),
                'trend': 'increasing' if drift_score > 0.05 else 'decreasing' if drift_score < -0.05 else 'stable',
                'alert_level': 'high' if drift_detected else 'normal'
            },
            'samples_analyzed': len(hallucination_history)
        }


# ============================================================================
# User Trust Impact (Type 17)
# ============================================================================

class UserTrustImpactAnalyzer:
    """Analyzes user trust impact from hallucinations."""

    def analyze_trust_impact(self,
                            user_feedback: List[Dict[str, Any]],
                            hallucination_incidents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze how hallucinations affect user trust."""
        if not user_feedback:
            return {'trust_impact_score': 0, 'user_trust_report': {}}

        # Analyze trust signals
        trust_scores = [f.get('trust_score', 5) for f in user_feedback]
        over_trust_signals = sum(1 for f in user_feedback if f.get('over_trusted', False))
        discovered_errors = sum(1 for f in user_feedback if f.get('discovered_error', False))

        avg_trust = np.mean(trust_scores)
        over_trust_rate = over_trust_signals / len(user_feedback)
        error_discovery_rate = discovered_errors / len(user_feedback)

        # Calculate trust impact (lower over-trust and higher error discovery is better)
        trust_calibration = 1 - over_trust_rate + error_discovery_rate * 0.5

        return {
            'trust_impact_score': float(trust_calibration),
            'user_trust_report': {
                'avg_trust_score': float(avg_trust),
                'over_trust_rate': float(over_trust_rate),
                'error_discovery_rate': float(error_discovery_rate),
                'feedback_count': len(user_feedback)
            },
            'recommendations': self._trust_recommendations(over_trust_rate, error_discovery_rate)
        }

    def _trust_recommendations(self, over_trust: float, discovery: float) -> List[str]:
        recs = []
        if over_trust > 0.3:
            recs.append("Users over-trust outputs - add uncertainty indicators")
        if discovery < 0.1:
            recs.append("Low error discovery rate - improve output verification UX")
        return recs


# ============================================================================
# Incident Response (Type 18)
# ============================================================================

class HallucinationIncidentManager:
    """Manages hallucination incident response."""

    def analyze_incidents(self,
                         incident_log: List[Dict[str, Any]],
                         correction_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze incident response and correction."""
        if not incident_log:
            return {'incident_count': 0, 'incident_log': []}

        # Analyze incident severity
        severity_dist = defaultdict(int)
        response_times = []
        corrected = 0

        for incident in incident_log:
            severity_dist[incident.get('severity', 'medium')] += 1

            if incident.get('response_time_hours'):
                response_times.append(incident['response_time_hours'])

            if incident.get('corrected', False):
                corrected += 1

        avg_response_time = np.mean(response_times) if response_times else 0
        correction_rate = corrected / len(incident_log)

        return {
            'incident_count': len(incident_log),
            'severity_distribution': dict(severity_dist),
            'response_metrics': {
                'avg_response_time_hours': float(avg_response_time),
                'correction_rate': float(correction_rate),
                'incidents_corrected': corrected
            },
            'incident_log': incident_log[-10:],  # Last 10 incidents
            'correction_workflow': {
                'user_notification_enabled': True,
                'auto_correction_enabled': False
            }
        }


# ============================================================================
# Evaluation Metrics (Type 19)
# ============================================================================

class HallucinationEvaluator:
    """Evaluates hallucinations with standard metrics."""

    def evaluate(self,
                evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate hallucinations with standard benchmarks."""
        if not evaluation_results:
            return {'evaluation_suite': {}, 'benchmark_scores': {}}

        # Aggregate metrics
        faithfulness_scores = [r.get('faithfulness', 0) for r in evaluation_results]
        factuality_scores = [r.get('factuality', 0) for r in evaluation_results]
        grounding_scores = [r.get('grounding', 0) for r in evaluation_results]

        return {
            'evaluation_suite': {
                'total_samples': len(evaluation_results),
                'metrics_computed': ['faithfulness', 'factuality', 'grounding']
            },
            'benchmark_scores': {
                'faithfulness': {
                    'mean': float(np.mean(faithfulness_scores)) if faithfulness_scores else 0,
                    'std': float(np.std(faithfulness_scores)) if faithfulness_scores else 0
                },
                'factuality': {
                    'mean': float(np.mean(factuality_scores)) if factuality_scores else 0,
                    'std': float(np.std(factuality_scores)) if factuality_scores else 0
                },
                'grounding': {
                    'mean': float(np.mean(grounding_scores)) if grounding_scores else 0,
                    'std': float(np.std(grounding_scores)) if grounding_scores else 0
                }
            }
        }


# ============================================================================
# Governance (Type 20)
# ============================================================================

class HallucinationGovernanceAnalyzer:
    """Analyzes hallucination prevention governance."""

    def analyze_governance(self,
                          governance_config: Dict[str, Any],
                          audit_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze governance for hallucination prevention."""
        # Check governance elements
        has_owner = governance_config.get('hallucination_owner') is not None
        has_raci = governance_config.get('raci_matrix') is not None
        has_release_gates = len(governance_config.get('release_gates', [])) > 0
        has_audit_cadence = governance_config.get('audit_cadence') is not None

        governance_score = sum([has_owner, has_raci, has_release_gates, has_audit_cadence]) / 4

        # Audit trail analysis
        audit_compliance = 0
        if audit_records:
            compliant = sum(1 for a in audit_records if a.get('compliant', False))
            audit_compliance = compliant / len(audit_records)

        return {
            'governance_score': float(governance_score),
            'governance_elements': {
                'owner_assigned': has_owner,
                'raci_defined': has_raci,
                'release_gates_configured': has_release_gates,
                'audit_cadence_defined': has_audit_cadence
            },
            'governance_policy': governance_config.get('policy_document'),
            'ownership': {
                'hallucination_owner': governance_config.get('hallucination_owner'),
                'raci_matrix': governance_config.get('raci_matrix')
            },
            'audit_trail': {
                'total_audits': len(audit_records) if audit_records else 0,
                'compliance_rate': float(audit_compliance)
            }
        }


# ============================================================================
# Report Generator
# ============================================================================

class HallucinationReportGenerator:
    """Generates comprehensive hallucination prevention reports."""

    def __init__(self):
        self.scope_analyzer = HallucinationScopeAnalyzer()
        self.risk_analyzer = RiskSensitivityAnalyzer()
        self.grounding_analyzer = RetrievalGroundingAnalyzer()
        self.faithfulness_analyzer = FaithfulnessAnalyzer()
        self.consistency_analyzer = ConsistencyAnalyzer()
        self.drift_monitor = HallucinationDriftMonitor()
        self.governance_analyzer = HallucinationGovernanceAnalyzer()

    def generate_full_report(self,
                            use_case_context: Dict[str, Any] = None,
                            query_context_pairs: List[Dict[str, Any]] = None,
                            outputs: List[Dict[str, Any]] = None,
                            hallucination_history: List[Dict[str, Any]] = None,
                            governance_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive hallucination prevention report."""
        report = {
            'report_type': 'hallucination_prevention_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if use_case_context:
            report['scope'] = self.scope_analyzer.analyze_scope(use_case_context)
            report['risk_sensitivity'] = self.risk_analyzer.analyze_risk_sensitivity(use_case_context)

        if query_context_pairs and outputs:
            report['grounding'] = self.grounding_analyzer.analyze_grounding(query_context_pairs, outputs)

            # Prepare faithfulness pairs
            faith_pairs = [
                {'context': qc.get('context', ''), 'answer': o.get('answer', '')}
                for qc, o in zip(query_context_pairs, outputs)
            ]
            report['faithfulness'] = self.faithfulness_analyzer.analyze_faithfulness(faith_pairs)

        if hallucination_history:
            report['drift'] = self.drift_monitor.analyze_drift(hallucination_history)

        if governance_config:
            report['governance'] = self.governance_analyzer.analyze_governance(governance_config)

        # Calculate overall score
        scores = []
        if 'grounding' in report:
            scores.append(report['grounding'].get('grounding_score', 0))
        if 'faithfulness' in report:
            scores.append(report['faithfulness'].get('faithfulness_score', 0))
        if 'governance' in report:
            scores.append(report['governance'].get('governance_score', 0))

        report['hallucination_prevention_score'] = float(np.mean(scores)) if scores else 0.0

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
        md = "# Hallucination Prevention Analysis Report\n\n"
        md += f"Generated: {report.get('timestamp', 'N/A')}\n\n"
        md += f"**Overall Hallucination Prevention Score: {report.get('hallucination_prevention_score', 0):.2%}**\n\n"

        for section, data in report.items():
            if isinstance(data, dict) and section not in ['timestamp', 'report_type', 'framework_version', 'hallucination_prevention_score']:
                md += f"## {section.replace('_', ' ').title()}\n\n"
                for key, value in data.items():
                    if not isinstance(value, (dict, list)):
                        md += f"- **{key}**: {value}\n"
                md += "\n"

        return md
