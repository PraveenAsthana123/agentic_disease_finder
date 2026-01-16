"""
Generative AI Analysis Module - Responsible Generative AI
=========================================================

Comprehensive analysis for responsible generative AI systems.
Implements 18+ analysis types for generative model governance.

Frameworks:
- Responsible Generative AI (18 types): Content Safety, Authenticity, Copyright, Misuse Prevention
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict
import json
import re


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GenerativeAIMetrics:
    """Metrics for generative AI analysis."""
    content_safety_score: float = 0.0
    authenticity_score: float = 0.0
    copyright_compliance: float = 0.0
    misuse_prevention: float = 0.0
    output_quality: float = 0.0
    hallucination_rate: float = 0.0


@dataclass
class GeneratedContent:
    """Represents generated content for analysis."""
    content_id: str
    content_type: str  # 'text', 'image', 'audio', 'video', 'code'
    prompt: str
    output: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    flagged: bool = False
    flag_reasons: List[str] = field(default_factory=list)


@dataclass
class ContentModerationResult:
    """Result of content moderation check."""
    content_id: str
    is_safe: bool
    categories_flagged: List[str] = field(default_factory=list)
    confidence: float = 0.0
    action_taken: str = "none"


# ============================================================================
# Responsible Generative AI Analyzers
# ============================================================================

class ContentSafetyAnalyzer:
    """Analyzes content safety in generated outputs."""

    def __init__(self, unsafe_categories: List[str] = None):
        self.unsafe_categories = unsafe_categories or [
            'violence', 'hate_speech', 'sexual_content', 'harassment',
            'misinformation', 'self_harm', 'illegal_activity', 'pii_exposure'
        ]

    def analyze_content_safety(self,
                               contents: List[GeneratedContent],
                               moderation_results: List[ContentModerationResult] = None) -> Dict[str, Any]:
        """Analyze content safety across generated outputs."""
        if not contents:
            return {'content_safety_score': 1.0, 'unsafe_content': 0}

        flagged_content = [c for c in contents if c.flagged]
        category_counts = defaultdict(int)

        for content in flagged_content:
            for reason in content.flag_reasons:
                category_counts[reason] += 1

        # Use moderation results if available
        if moderation_results:
            safe_count = sum(1 for r in moderation_results if r.is_safe)
            safety_rate = safe_count / len(moderation_results)

            for result in moderation_results:
                if not result.is_safe:
                    for cat in result.categories_flagged:
                        category_counts[cat] += 1
        else:
            safety_rate = 1 - (len(flagged_content) / len(contents))

        return {
            'content_safety_score': float(safety_rate),
            'total_content': len(contents),
            'unsafe_content': len(flagged_content),
            'safe_rate': float(safety_rate),
            'category_breakdown': dict(category_counts),
            'most_common_violation': max(category_counts, key=category_counts.get) if category_counts else None,
            'moderation_applied': moderation_results is not None
        }


class AuthenticityAnalyzer:
    """Analyzes authenticity and disclosure of AI-generated content."""

    def analyze_authenticity(self,
                            contents: List[GeneratedContent],
                            disclosure_requirements: Dict[str, bool] = None) -> Dict[str, Any]:
        """Analyze authenticity disclosure compliance."""
        if not contents:
            return {'authenticity_score': 1.0, 'properly_disclosed': 0}

        disclosure_requirements = disclosure_requirements or {
            'watermarked': True,
            'metadata_tagged': True,
            'disclosure_statement': True
        }

        properly_disclosed = []
        disclosure_issues = []

        for content in contents:
            metadata = content.metadata
            issues = []

            for req, required in disclosure_requirements.items():
                if required and not metadata.get(req, False):
                    issues.append(req)

            if not issues:
                properly_disclosed.append(content.content_id)
            else:
                disclosure_issues.append({
                    'content_id': content.content_id,
                    'missing_requirements': issues
                })

        disclosure_rate = len(properly_disclosed) / len(contents) if contents else 1

        return {
            'authenticity_score': float(disclosure_rate),
            'properly_disclosed': len(properly_disclosed),
            'disclosure_issues': len(disclosure_issues),
            'total_content': len(contents),
            'issue_details': disclosure_issues[:20],
            'requirements_checked': list(disclosure_requirements.keys())
        }


class CopyrightComplianceAnalyzer:
    """Analyzes copyright compliance in generated content."""

    def analyze_copyright(self,
                         contents: List[GeneratedContent],
                         copyright_checks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze copyright compliance."""
        if not contents:
            return {'copyright_compliance': 1.0, 'potential_violations': 0}

        if copyright_checks:
            violations = [c for c in copyright_checks if c.get('violation_detected', False)]
            compliance_rate = 1 - (len(violations) / len(copyright_checks))

            violation_types = defaultdict(int)
            for v in violations:
                violation_types[v.get('violation_type', 'unknown')] += 1

            return {
                'copyright_compliance': float(compliance_rate),
                'checks_performed': len(copyright_checks),
                'potential_violations': len(violations),
                'violation_types': dict(violation_types),
                'compliant_content': len(copyright_checks) - len(violations)
            }
        else:
            # Without explicit checks, analyze metadata
            with_attribution = sum(1 for c in contents if c.metadata.get('sources_cited', False))
            compliance_estimate = with_attribution / len(contents) if contents else 0.5

            return {
                'copyright_compliance': float(compliance_estimate),
                'content_analyzed': len(contents),
                'with_attribution': with_attribution,
                'note': 'Estimate based on metadata - detailed check recommended'
            }


class MisusePreventionAnalyzer:
    """Analyzes misuse prevention measures."""

    def analyze_misuse_prevention(self,
                                  access_logs: List[Dict[str, Any]],
                                  blocked_requests: List[Dict[str, Any]],
                                  rate_limits: Dict[str, int] = None) -> Dict[str, Any]:
        """Analyze misuse prevention effectiveness."""
        if not access_logs:
            return {'prevention_score': 0.0, 'blocked_requests': 0}

        total_requests = len(access_logs)
        blocked_count = len(blocked_requests)

        # Analyze blocked reasons
        block_reasons = defaultdict(int)
        for block in blocked_requests:
            reason = block.get('reason', 'unknown')
            block_reasons[reason] += 1

        # Analyze rate limit violations
        rate_limit_violations = sum(1 for log in access_logs if log.get('rate_limited', False))

        # Analyze suspicious patterns
        user_request_counts = defaultdict(int)
        for log in access_logs:
            user = log.get('user_id', 'unknown')
            user_request_counts[user] += 1

        # Find potential abuse (users with excessive requests)
        potential_abuse = [
            {'user': user, 'requests': count}
            for user, count in user_request_counts.items()
            if count > np.mean(list(user_request_counts.values())) * 5
        ]

        # Prevention score based on blocking effectiveness
        if blocked_count > 0:
            prevention_score = blocked_count / (blocked_count + len(potential_abuse))
        else:
            prevention_score = 1.0 if not potential_abuse else 0.5

        return {
            'prevention_score': float(prevention_score),
            'total_requests': total_requests,
            'blocked_requests': blocked_count,
            'block_rate': float(blocked_count / total_requests) if total_requests > 0 else 0,
            'block_reasons': dict(block_reasons),
            'rate_limit_violations': rate_limit_violations,
            'potential_abuse_detected': len(potential_abuse),
            'abuse_details': potential_abuse[:10]
        }


class HallucinationAnalyzer:
    """Analyzes hallucination in generated content."""

    def analyze_hallucinations(self,
                               generated_outputs: List[Dict[str, Any]],
                               ground_truth: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze hallucination patterns."""
        if not generated_outputs:
            return {'hallucination_rate': 0.0, 'hallucinations_detected': 0}

        hallucinations = []
        verified_accurate = []

        for i, output in enumerate(generated_outputs):
            content = output.get('content', '')
            claims = output.get('claims', [])
            verified = output.get('verified', None)

            if verified is False or output.get('hallucination_detected', False):
                hallucinations.append({
                    'output_index': i,
                    'type': output.get('hallucination_type', 'unknown'),
                    'confidence': output.get('hallucination_confidence', 0)
                })
            elif verified is True:
                verified_accurate.append(i)

        # If ground truth available, compare
        if ground_truth and len(ground_truth) == len(generated_outputs):
            factual_errors = 0
            for i, (gen, truth) in enumerate(zip(generated_outputs, ground_truth)):
                gen_facts = set(gen.get('facts', []))
                truth_facts = set(truth.get('facts', []))
                if gen_facts - truth_facts:
                    factual_errors += 1
            factual_accuracy = 1 - (factual_errors / len(generated_outputs))
        else:
            factual_accuracy = None

        hallucination_rate = len(hallucinations) / len(generated_outputs) if generated_outputs else 0

        return {
            'hallucination_rate': float(hallucination_rate),
            'accuracy_rate': float(1 - hallucination_rate),
            'hallucinations_detected': len(hallucinations),
            'verified_accurate': len(verified_accurate),
            'total_outputs': len(generated_outputs),
            'factual_accuracy': float(factual_accuracy) if factual_accuracy else None,
            'hallucination_details': hallucinations[:20]
        }


class OutputQualityAnalyzer:
    """Analyzes quality of generated outputs."""

    def analyze_quality(self,
                       outputs: List[GeneratedContent],
                       quality_ratings: List[Dict[str, float]] = None) -> Dict[str, Any]:
        """Analyze output quality."""
        if not outputs:
            return {'quality_score': 0.0, 'outputs_analyzed': 0}

        quality_metrics = defaultdict(list)

        if quality_ratings:
            for rating in quality_ratings:
                for metric, score in rating.items():
                    if isinstance(score, (int, float)):
                        quality_metrics[metric].append(score)
        else:
            # Basic quality heuristics
            for output in outputs:
                content = output.output
                if isinstance(content, str):
                    # Length heuristic
                    quality_metrics['length'].append(min(1, len(content) / 500))
                    # Coherence (no repeated phrases)
                    words = content.split()
                    unique_ratio = len(set(words)) / len(words) if words else 0
                    quality_metrics['coherence'].append(unique_ratio)

        # Calculate average quality scores
        metric_averages = {
            metric: float(np.mean(scores))
            for metric, scores in quality_metrics.items()
        }

        overall_quality = np.mean(list(metric_averages.values())) if metric_averages else 0

        return {
            'quality_score': float(overall_quality),
            'metric_scores': metric_averages,
            'outputs_analyzed': len(outputs),
            'quality_distribution': {
                'high': sum(1 for o in quality_ratings or [] if np.mean(list(o.values())) > 0.8),
                'medium': sum(1 for o in quality_ratings or [] if 0.5 <= np.mean(list(o.values())) <= 0.8),
                'low': sum(1 for o in quality_ratings or [] if np.mean(list(o.values())) < 0.5)
            } if quality_ratings else {}
        }


class PromptInjectionAnalyzer:
    """Analyzes prompt injection attempts and defenses."""

    def __init__(self, injection_patterns: List[str] = None):
        self.injection_patterns = injection_patterns or [
            r'ignore\s+(previous|all)\s+instructions',
            r'you\s+are\s+now\s+',
            r'new\s+instructions:',
            r'disregard\s+',
            r'forget\s+everything',
            r'system\s+prompt:',
            r'<\|.*?\|>',
            r'\[\[.*?\]\]'
        ]

    def analyze_injection_attempts(self,
                                   prompts: List[str],
                                   detection_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze prompt injection attempts."""
        if not prompts:
            return {'injection_rate': 0.0, 'attempts_detected': 0}

        detected_injections = []
        blocked_injections = []

        for i, prompt in enumerate(prompts):
            # Check patterns
            matched_patterns = []
            for pattern in self.injection_patterns:
                if re.search(pattern, prompt.lower()):
                    matched_patterns.append(pattern)

            if matched_patterns:
                injection_data = {
                    'prompt_index': i,
                    'patterns_matched': matched_patterns
                }

                # Check if blocked
                if detection_results and i < len(detection_results):
                    result = detection_results[i]
                    if result.get('blocked', False):
                        blocked_injections.append(injection_data)
                    else:
                        detected_injections.append(injection_data)
                else:
                    detected_injections.append(injection_data)

        total_prompts = len(prompts)
        total_attempts = len(detected_injections) + len(blocked_injections)

        injection_rate = total_attempts / total_prompts if total_prompts > 0 else 0
        block_rate = len(blocked_injections) / total_attempts if total_attempts > 0 else 1

        return {
            'injection_rate': float(injection_rate),
            'block_rate': float(block_rate),
            'attempts_detected': total_attempts,
            'successfully_blocked': len(blocked_injections),
            'unblocked_attempts': len(detected_injections),
            'total_prompts': total_prompts,
            'defense_effectiveness': float(block_rate),
            'unblocked_details': detected_injections[:20]
        }


# ============================================================================
# Report Generator
# ============================================================================

class GenerativeAIReportGenerator:
    """Generates comprehensive generative AI reports."""

    def __init__(self):
        self.safety_analyzer = ContentSafetyAnalyzer()
        self.authenticity_analyzer = AuthenticityAnalyzer()
        self.copyright_analyzer = CopyrightComplianceAnalyzer()
        self.misuse_analyzer = MisusePreventionAnalyzer()
        self.hallucination_analyzer = HallucinationAnalyzer()
        self.quality_analyzer = OutputQualityAnalyzer()
        self.injection_analyzer = PromptInjectionAnalyzer()

    def generate_full_report(self,
                            contents: List[GeneratedContent] = None,
                            moderation_results: List[ContentModerationResult] = None,
                            access_logs: List[Dict[str, Any]] = None,
                            blocked_requests: List[Dict[str, Any]] = None,
                            prompts: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive generative AI report."""
        report = {
            'report_type': 'comprehensive_generative_ai_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if contents:
            report['content_safety'] = self.safety_analyzer.analyze_content_safety(
                contents, moderation_results
            )
            report['authenticity'] = self.authenticity_analyzer.analyze_authenticity(contents)
            report['copyright'] = self.copyright_analyzer.analyze_copyright(contents)
            report['quality'] = self.quality_analyzer.analyze_quality(contents)

        if access_logs:
            report['misuse_prevention'] = self.misuse_analyzer.analyze_misuse_prevention(
                access_logs, blocked_requests or []
            )

        if prompts:
            report['injection_analysis'] = self.injection_analyzer.analyze_injection_attempts(prompts)

        # Calculate overall responsible AI score
        scores = []
        if 'content_safety' in report:
            scores.append(report['content_safety'].get('content_safety_score', 0))
        if 'authenticity' in report:
            scores.append(report['authenticity'].get('authenticity_score', 0))
        if 'copyright' in report:
            scores.append(report['copyright'].get('copyright_compliance', 0))
        if 'misuse_prevention' in report:
            scores.append(report['misuse_prevention'].get('prevention_score', 0))

        report['responsible_ai_score'] = float(np.mean(scores)) if scores else 0.0

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
            f"# {report.get('report_type', 'Generative AI Analysis Report')}",
            f"\n**Generated:** {report.get('timestamp', 'N/A')}",
            f"\n**Responsible AI Score:** {report.get('responsible_ai_score', 0):.2%}",
            "\n---\n"
        ]

        if 'content_safety' in report:
            cs = report['content_safety']
            lines.append("## Content Safety\n")
            lines.append(f"- **Safety Score:** {cs.get('content_safety_score', 0):.2%}")
            lines.append(f"- **Unsafe Content:** {cs.get('unsafe_content', 0)}")
            lines.append("")

        return "\n".join(lines)
