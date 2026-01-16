"""
Accountability Analysis Module - Accountable AI, Auditable AI, Compliance AI
=============================================================================

Comprehensive analysis for AI accountability, auditability, and compliance.
Implements 54 analysis types across three related frameworks.

Frameworks:
- Accountable AI (18 types): Responsibility, Ownership, RACI, Escalation
- Auditable AI (18 types): Audit Trails, Evidence, Traceability, Records
- Compliance AI (18 types): Regulatory, Standards, Policies, Certifications
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AccountabilityMetrics:
    """Metrics for accountability analysis."""
    responsibility_clarity: float = 0.0
    ownership_coverage: float = 0.0
    raci_completeness: float = 0.0
    escalation_effectiveness: float = 0.0
    decision_traceability: float = 0.0
    stakeholder_notification: float = 0.0
    remediation_rate: float = 0.0


@dataclass
class AuditMetrics:
    """Metrics for audit analysis."""
    audit_trail_completeness: float = 0.0
    evidence_quality: float = 0.0
    traceability_score: float = 0.0
    record_integrity: float = 0.0
    log_coverage: float = 0.0
    retention_compliance: float = 0.0
    audit_readiness: float = 0.0


@dataclass
class ComplianceMetrics:
    """Metrics for compliance analysis."""
    regulatory_compliance: float = 0.0
    standards_adherence: float = 0.0
    policy_compliance: float = 0.0
    certification_status: float = 0.0
    gap_remediation: float = 0.0
    compliance_risk: float = 0.0


@dataclass
class AuditRecord:
    """Represents an audit record."""
    record_id: str
    timestamp: datetime
    action_type: str
    actor: str
    resource: str
    action_details: Dict[str, Any]
    outcome: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceRequirement:
    """Represents a compliance requirement."""
    requirement_id: str
    regulation: str
    description: str
    category: str
    mandatory: bool
    deadline: Optional[datetime] = None
    evidence_required: List[str] = field(default_factory=list)


@dataclass
class RACIEntry:
    """Represents a RACI matrix entry."""
    process_id: str
    process_name: str
    responsible: List[str]
    accountable: str
    consulted: List[str]
    informed: List[str]


# ============================================================================
# Accountable AI Analyzers (18 Analysis Types)
# ============================================================================

class ResponsibilityAnalyzer:
    """Analyzes responsibility assignment and clarity."""

    def analyze_responsibility(self,
                               decisions: List[Dict[str, Any]],
                               responsibility_matrix: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze responsibility assignment for decisions."""
        if not decisions:
            return {'responsibility_clarity': 1.0, 'unassigned_decisions': 0}

        assigned_decisions = []
        unassigned_decisions = []
        ambiguous_decisions = []

        for decision in decisions:
            decision_type = decision.get('type', 'unknown')
            responsible_parties = responsibility_matrix.get(decision_type, [])
            explicit_owner = decision.get('owner')

            if explicit_owner or responsible_parties:
                if len(responsible_parties) == 1 or explicit_owner:
                    assigned_decisions.append({
                        'decision_id': decision.get('id'),
                        'type': decision_type,
                        'owner': explicit_owner or responsible_parties[0]
                    })
                else:
                    # Multiple responsible parties without clear owner
                    ambiguous_decisions.append({
                        'decision_id': decision.get('id'),
                        'type': decision_type,
                        'parties': responsible_parties
                    })
            else:
                unassigned_decisions.append({
                    'decision_id': decision.get('id'),
                    'type': decision_type
                })

        total = len(decisions)
        clarity = len(assigned_decisions) / total if total > 0 else 1.0
        ambiguity_rate = len(ambiguous_decisions) / total if total > 0 else 0

        return {
            'responsibility_clarity': clarity,
            'assigned_decisions': len(assigned_decisions),
            'unassigned_decisions': len(unassigned_decisions),
            'ambiguous_decisions': len(ambiguous_decisions),
            'total_decisions': total,
            'ambiguity_rate': ambiguity_rate,
            'unassigned_details': unassigned_decisions,
            'ambiguous_details': ambiguous_decisions
        }


class OwnershipAnalyzer:
    """Analyzes ownership assignment and coverage."""

    def analyze_ownership(self,
                         resources: List[Dict[str, Any]],
                         owner_registry: Dict[str, str]) -> Dict[str, Any]:
        """Analyze ownership coverage for resources."""
        if not resources:
            return {'ownership_coverage': 1.0, 'unowned_resources': 0}

        owned_resources = []
        unowned_resources = []
        owner_distribution = defaultdict(list)

        for resource in resources:
            resource_id = resource.get('id', 'unknown')
            resource_type = resource.get('type', 'unknown')
            explicit_owner = resource.get('owner')

            owner = explicit_owner or owner_registry.get(resource_id)

            if owner:
                owned_resources.append({
                    'resource_id': resource_id,
                    'type': resource_type,
                    'owner': owner
                })
                owner_distribution[owner].append(resource_id)
            else:
                unowned_resources.append({
                    'resource_id': resource_id,
                    'type': resource_type,
                    'criticality': resource.get('criticality', 'unknown')
                })

        total = len(resources)
        coverage = len(owned_resources) / total if total > 0 else 1.0

        # Analyze owner load balance
        if owner_distribution:
            resources_per_owner = [len(v) for v in owner_distribution.values()]
            load_balance = 1 - (np.std(resources_per_owner) / np.mean(resources_per_owner)) if np.mean(resources_per_owner) > 0 else 1
        else:
            load_balance = 0

        # Critical unowned resources
        critical_unowned = [r for r in unowned_resources if r.get('criticality') in ['high', 'critical']]

        return {
            'ownership_coverage': coverage,
            'owned_resources': len(owned_resources),
            'unowned_resources': len(unowned_resources),
            'total_resources': total,
            'unique_owners': len(owner_distribution),
            'owner_load_balance': float(max(0, load_balance)),
            'critical_unowned': len(critical_unowned),
            'unowned_details': unowned_resources,
            'owner_distribution': {k: len(v) for k, v in owner_distribution.items()}
        }


class RACIAnalyzer:
    """Analyzes RACI matrix completeness and clarity."""

    def analyze_raci(self,
                    raci_entries: List[RACIEntry],
                    required_processes: List[str] = None) -> Dict[str, Any]:
        """Analyze RACI matrix completeness."""
        if not raci_entries:
            return {'raci_completeness': 0.0, 'missing_processes': required_processes or []}

        complete_entries = []
        incomplete_entries = []
        issues = []

        for entry in raci_entries:
            entry_issues = []

            # Check for required fields
            if not entry.responsible:
                entry_issues.append('missing_responsible')
            if not entry.accountable:
                entry_issues.append('missing_accountable')

            # Check for RACI best practices
            if len(entry.responsible) > 3:
                entry_issues.append('too_many_responsible')

            entry_data = {
                'process_id': entry.process_id,
                'process_name': entry.process_name,
                'responsible_count': len(entry.responsible),
                'has_accountable': bool(entry.accountable),
                'consulted_count': len(entry.consulted),
                'informed_count': len(entry.informed)
            }

            if entry_issues:
                incomplete_entries.append(entry_data)
                issues.append({
                    'process_id': entry.process_id,
                    'issues': entry_issues
                })
            else:
                complete_entries.append(entry_data)

        total = len(raci_entries)
        completeness = len(complete_entries) / total if total > 0 else 0

        # Check coverage of required processes
        covered_processes = {e.process_id for e in raci_entries}
        if required_processes:
            required_set = set(required_processes)
            process_coverage = len(covered_processes & required_set) / len(required_set) if required_set else 1
            missing_processes = list(required_set - covered_processes)
        else:
            process_coverage = 1
            missing_processes = []

        return {
            'raci_completeness': completeness,
            'process_coverage': process_coverage,
            'complete_entries': len(complete_entries),
            'incomplete_entries': len(incomplete_entries),
            'total_entries': total,
            'missing_processes': missing_processes,
            'issues': issues,
            'recommendations': self._generate_raci_recommendations(issues)
        }

    def _generate_raci_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate recommendations based on RACI issues."""
        recommendations = []

        missing_accountable = sum(1 for i in issues if 'missing_accountable' in i['issues'])
        if missing_accountable > 0:
            recommendations.append(f"Assign accountable party for {missing_accountable} processes")

        too_many_responsible = sum(1 for i in issues if 'too_many_responsible' in i['issues'])
        if too_many_responsible > 0:
            recommendations.append(f"Reduce responsible parties for {too_many_responsible} processes (recommend max 3)")

        return recommendations


class EscalationAnalyzer:
    """Analyzes escalation paths and effectiveness."""

    def analyze_escalation(self,
                          escalation_events: List[Dict[str, Any]],
                          escalation_policies: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze escalation effectiveness."""
        if not escalation_events:
            return {'escalation_effectiveness': 1.0, 'total_escalations': 0}

        successful_escalations = []
        failed_escalations = []
        escalation_times = []
        escalation_by_type = defaultdict(lambda: {'success': 0, 'fail': 0})

        for event in escalation_events:
            event_type = event.get('type', 'unknown')
            was_resolved = event.get('resolved', False)
            escalation_time = event.get('time_to_escalate_seconds', 0)
            resolution_time = event.get('time_to_resolve_seconds', 0)

            escalation_times.append(escalation_time)

            if was_resolved:
                successful_escalations.append({
                    'event_id': event.get('id'),
                    'type': event_type,
                    'escalation_time': escalation_time,
                    'resolution_time': resolution_time
                })
                escalation_by_type[event_type]['success'] += 1
            else:
                failed_escalations.append({
                    'event_id': event.get('id'),
                    'type': event_type,
                    'reason': event.get('failure_reason', 'unknown')
                })
                escalation_by_type[event_type]['fail'] += 1

        total = len(escalation_events)
        effectiveness = len(successful_escalations) / total if total > 0 else 1.0

        # Analyze policy compliance
        policy_compliance = {}
        for event_type, policy in escalation_policies.items():
            events_of_type = [e for e in escalation_events if e.get('type') == event_type]
            if events_of_type:
                compliant = sum(
                    1 for e in events_of_type
                    if e.get('time_to_escalate_seconds', float('inf')) <= policy.get('max_time_seconds', float('inf'))
                )
                policy_compliance[event_type] = compliant / len(events_of_type)
            else:
                policy_compliance[event_type] = 1.0

        return {
            'escalation_effectiveness': effectiveness,
            'total_escalations': total,
            'successful_escalations': len(successful_escalations),
            'failed_escalations': len(failed_escalations),
            'mean_escalation_time': float(np.mean(escalation_times)) if escalation_times else 0,
            'escalation_by_type': dict(escalation_by_type),
            'policy_compliance': policy_compliance,
            'failed_details': failed_escalations[:20]
        }


class DecisionTraceabilityAnalyzer:
    """Analyzes decision traceability and documentation."""

    def analyze_traceability(self,
                            decisions: List[Dict[str, Any]],
                            traceability_requirements: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Analyze decision traceability."""
        if not decisions:
            return {'traceability_score': 1.0, 'untraceable_decisions': 0}

        required_fields = traceability_requirements or {
            'all': ['timestamp', 'decision_maker', 'rationale', 'inputs', 'outcome']
        }

        traceable_decisions = []
        untraceable_decisions = []

        for decision in decisions:
            decision_type = decision.get('type', 'all')
            required = required_fields.get(decision_type, required_fields.get('all', []))

            missing_fields = []
            for field in required:
                if not decision.get(field):
                    missing_fields.append(field)

            if not missing_fields:
                traceable_decisions.append({
                    'decision_id': decision.get('id'),
                    'type': decision_type
                })
            else:
                untraceable_decisions.append({
                    'decision_id': decision.get('id'),
                    'type': decision_type,
                    'missing_fields': missing_fields
                })

        total = len(decisions)
        traceability = len(traceable_decisions) / total if total > 0 else 1.0

        # Analyze which fields are most commonly missing
        missing_field_counts = defaultdict(int)
        for d in untraceable_decisions:
            for field in d['missing_fields']:
                missing_field_counts[field] += 1

        return {
            'traceability_score': traceability,
            'traceable_decisions': len(traceable_decisions),
            'untraceable_decisions': len(untraceable_decisions),
            'total_decisions': total,
            'missing_field_analysis': dict(missing_field_counts),
            'most_missing_field': max(missing_field_counts, key=missing_field_counts.get) if missing_field_counts else None,
            'untraceable_details': untraceable_decisions[:20]
        }


class StakeholderNotificationAnalyzer:
    """Analyzes stakeholder notification and communication."""

    def analyze_notifications(self,
                             events: List[Dict[str, Any]],
                             notification_requirements: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze stakeholder notification compliance."""
        if not events:
            return {'notification_compliance': 1.0, 'missed_notifications': 0}

        compliant_notifications = []
        missed_notifications = []
        late_notifications = []

        for event in events:
            event_type = event.get('type', 'unknown')
            requirements = notification_requirements.get(event_type, {})

            required_stakeholders = set(requirements.get('stakeholders', []))
            notified_stakeholders = set(event.get('notified_stakeholders', []))
            max_delay = requirements.get('max_delay_seconds', float('inf'))

            # Check if all required stakeholders were notified
            missing_stakeholders = required_stakeholders - notified_stakeholders
            notification_delay = event.get('notification_delay_seconds', 0)

            if not missing_stakeholders and notification_delay <= max_delay:
                compliant_notifications.append({
                    'event_id': event.get('id'),
                    'type': event_type
                })
            elif missing_stakeholders:
                missed_notifications.append({
                    'event_id': event.get('id'),
                    'type': event_type,
                    'missing_stakeholders': list(missing_stakeholders)
                })
            elif notification_delay > max_delay:
                late_notifications.append({
                    'event_id': event.get('id'),
                    'type': event_type,
                    'delay_seconds': notification_delay,
                    'max_allowed': max_delay
                })

        total = len(events)
        compliance = len(compliant_notifications) / total if total > 0 else 1.0

        return {
            'notification_compliance': compliance,
            'compliant_notifications': len(compliant_notifications),
            'missed_notifications': len(missed_notifications),
            'late_notifications': len(late_notifications),
            'total_events': total,
            'missed_details': missed_notifications[:20],
            'late_details': late_notifications[:20]
        }


class RemediationAnalyzer:
    """Analyzes remediation effectiveness and timeliness."""

    def analyze_remediation(self,
                           issues: List[Dict[str, Any]],
                           sla_requirements: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze remediation effectiveness."""
        if not issues:
            return {'remediation_rate': 1.0, 'open_issues': 0}

        sla_requirements = sla_requirements or {
            'critical': 4 * 3600,  # 4 hours
            'high': 24 * 3600,    # 24 hours
            'medium': 72 * 3600,  # 72 hours
            'low': 168 * 3600    # 1 week
        }

        remediated_issues = []
        open_issues = []
        sla_breaches = []

        for issue in issues:
            severity = issue.get('severity', 'medium')
            is_remediated = issue.get('remediated', False)
            time_to_remediate = issue.get('time_to_remediate_seconds')

            if is_remediated:
                remediated_issues.append({
                    'issue_id': issue.get('id'),
                    'severity': severity,
                    'time_to_remediate': time_to_remediate
                })

                # Check SLA compliance
                sla = sla_requirements.get(severity, float('inf'))
                if time_to_remediate and time_to_remediate > sla:
                    sla_breaches.append({
                        'issue_id': issue.get('id'),
                        'severity': severity,
                        'sla': sla,
                        'actual': time_to_remediate
                    })
            else:
                open_issues.append({
                    'issue_id': issue.get('id'),
                    'severity': severity,
                    'age_seconds': issue.get('age_seconds', 0)
                })

        total = len(issues)
        remediation_rate = len(remediated_issues) / total if total > 0 else 1.0
        sla_compliance = 1 - (len(sla_breaches) / len(remediated_issues)) if remediated_issues else 1.0

        # Calculate mean time to remediate by severity
        mttr_by_severity = defaultdict(list)
        for r in remediated_issues:
            if r['time_to_remediate']:
                mttr_by_severity[r['severity']].append(r['time_to_remediate'])

        mttr_analysis = {
            sev: {'mean': float(np.mean(times)), 'max': float(max(times))}
            for sev, times in mttr_by_severity.items()
        }

        return {
            'remediation_rate': remediation_rate,
            'sla_compliance': sla_compliance,
            'remediated_issues': len(remediated_issues),
            'open_issues': len(open_issues),
            'sla_breaches': len(sla_breaches),
            'total_issues': total,
            'mttr_by_severity': mttr_analysis,
            'open_issue_details': open_issues,
            'sla_breach_details': sla_breaches[:20]
        }


# ============================================================================
# Auditable AI Analyzers (18 Analysis Types)
# ============================================================================

class AuditTrailAnalyzer:
    """Analyzes audit trail completeness and quality."""

    def analyze_audit_trail(self,
                           audit_records: List[AuditRecord],
                           required_events: List[str] = None) -> Dict[str, Any]:
        """Analyze audit trail completeness."""
        if not audit_records:
            return {'audit_completeness': 0.0, 'missing_events': required_events or []}

        required_events = required_events or [
            'create', 'read', 'update', 'delete', 'login', 'logout',
            'permission_change', 'config_change', 'error', 'decision'
        ]

        # Analyze event coverage
        recorded_event_types = set(r.action_type for r in audit_records)
        required_set = set(required_events)
        coverage = len(recorded_event_types & required_set) / len(required_set) if required_set else 1

        # Analyze record quality
        quality_scores = []
        incomplete_records = []

        for record in audit_records:
            # Check for required fields
            score = 0
            if record.timestamp:
                score += 0.2
            if record.actor:
                score += 0.2
            if record.resource:
                score += 0.2
            if record.action_details:
                score += 0.2
            if record.outcome:
                score += 0.2

            quality_scores.append(score)

            if score < 1.0:
                incomplete_records.append({
                    'record_id': record.record_id,
                    'quality_score': score,
                    'action_type': record.action_type
                })

        avg_quality = np.mean(quality_scores) if quality_scores else 0
        completeness = coverage * avg_quality

        # Time-based analysis
        if len(audit_records) >= 2:
            sorted_records = sorted(audit_records, key=lambda x: x.timestamp)
            gaps = []
            for i in range(1, len(sorted_records)):
                gap = (sorted_records[i].timestamp - sorted_records[i-1].timestamp).total_seconds()
                if gap > 3600:  # Gap > 1 hour
                    gaps.append({
                        'start': sorted_records[i-1].timestamp.isoformat(),
                        'end': sorted_records[i].timestamp.isoformat(),
                        'gap_seconds': gap
                    })
        else:
            gaps = []

        return {
            'audit_completeness': completeness,
            'event_coverage': coverage,
            'average_record_quality': float(avg_quality),
            'total_records': len(audit_records),
            'event_types_recorded': list(recorded_event_types),
            'missing_event_types': list(required_set - recorded_event_types),
            'incomplete_records': len(incomplete_records),
            'audit_gaps': gaps[:20],
            'incomplete_record_details': incomplete_records[:20]
        }


class EvidenceAnalyzer:
    """Analyzes evidence quality and availability."""

    def analyze_evidence(self,
                        claims: List[Dict[str, Any]],
                        evidence_store: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze evidence quality for claims."""
        if not claims:
            return {'evidence_quality': 1.0, 'unsupported_claims': 0}

        supported_claims = []
        unsupported_claims = []
        partially_supported = []

        for claim in claims:
            claim_id = claim.get('id', 'unknown')
            required_evidence = claim.get('required_evidence', [])
            available_evidence = evidence_store.get(claim_id, [])

            available_evidence_types = {e.get('type') for e in available_evidence}

            if not required_evidence:
                # No specific requirements - check if any evidence exists
                if available_evidence:
                    supported_claims.append({
                        'claim_id': claim_id,
                        'evidence_count': len(available_evidence)
                    })
                else:
                    unsupported_claims.append({
                        'claim_id': claim_id,
                        'reason': 'no_evidence'
                    })
            else:
                required_set = set(required_evidence)
                missing = required_set - available_evidence_types

                if not missing:
                    supported_claims.append({
                        'claim_id': claim_id,
                        'evidence_count': len(available_evidence)
                    })
                elif missing == required_set:
                    unsupported_claims.append({
                        'claim_id': claim_id,
                        'missing_evidence': list(missing)
                    })
                else:
                    partially_supported.append({
                        'claim_id': claim_id,
                        'missing_evidence': list(missing),
                        'available_evidence': list(available_evidence_types)
                    })

        total = len(claims)
        quality = (len(supported_claims) + 0.5 * len(partially_supported)) / total if total > 0 else 1

        return {
            'evidence_quality': quality,
            'supported_claims': len(supported_claims),
            'partially_supported_claims': len(partially_supported),
            'unsupported_claims': len(unsupported_claims),
            'total_claims': total,
            'unsupported_details': unsupported_claims,
            'partial_details': partially_supported
        }


class RecordIntegrityAnalyzer:
    """Analyzes record integrity and tampering detection."""

    def analyze_integrity(self,
                         records: List[Dict[str, Any]],
                         integrity_checks: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze record integrity."""
        if not records:
            return {'integrity_score': 1.0, 'compromised_records': 0}

        valid_records = []
        compromised_records = []
        unverified_records = []

        for record in records:
            record_id = record.get('id', 'unknown')
            has_checksum = 'checksum' in record
            has_signature = 'signature' in record
            verified = record.get('verified', False)

            if verified:
                valid_records.append({
                    'record_id': record_id,
                    'has_checksum': has_checksum,
                    'has_signature': has_signature
                })
            elif record.get('tampering_detected', False):
                compromised_records.append({
                    'record_id': record_id,
                    'issue': record.get('integrity_issue', 'unknown')
                })
            else:
                unverified_records.append({
                    'record_id': record_id,
                    'reason': 'not_verified'
                })

        total = len(records)
        integrity_score = len(valid_records) / total if total > 0 else 1
        compromise_rate = len(compromised_records) / total if total > 0 else 0

        # Analyze integrity check coverage
        checksum_coverage = sum(1 for r in records if 'checksum' in r) / total if total > 0 else 0
        signature_coverage = sum(1 for r in records if 'signature' in r) / total if total > 0 else 0

        return {
            'integrity_score': integrity_score,
            'compromise_rate': compromise_rate,
            'valid_records': len(valid_records),
            'compromised_records': len(compromised_records),
            'unverified_records': len(unverified_records),
            'total_records': total,
            'checksum_coverage': checksum_coverage,
            'signature_coverage': signature_coverage,
            'compromised_details': compromised_records
        }


class LogCoverageAnalyzer:
    """Analyzes logging coverage and completeness."""

    def analyze_log_coverage(self,
                            components: List[Dict[str, Any]],
                            log_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze logging coverage across components."""
        if not components:
            return {'log_coverage': 0.0, 'unlogged_components': []}

        logged_components = set()
        for log in log_records:
            component = log.get('component')
            if component:
                logged_components.add(component)

        component_ids = {c.get('id') for c in components}
        unlogged = component_ids - logged_components

        coverage = len(logged_components & component_ids) / len(component_ids) if component_ids else 1

        # Analyze log volume by component
        log_volume = defaultdict(int)
        for log in log_records:
            log_volume[log.get('component', 'unknown')] += 1

        # Identify components with low log activity
        avg_volume = np.mean(list(log_volume.values())) if log_volume else 0
        low_activity = [c for c, v in log_volume.items() if v < avg_volume * 0.1]

        # Analyze log levels
        log_levels = defaultdict(int)
        for log in log_records:
            log_levels[log.get('level', 'unknown')] += 1

        return {
            'log_coverage': coverage,
            'logged_components': len(logged_components),
            'unlogged_components': list(unlogged),
            'total_components': len(components),
            'total_log_records': len(log_records),
            'log_volume_by_component': dict(log_volume),
            'low_activity_components': low_activity,
            'log_level_distribution': dict(log_levels)
        }


class RetentionAnalyzer:
    """Analyzes data retention compliance."""

    def analyze_retention(self,
                         data_records: List[Dict[str, Any]],
                         retention_policies: Dict[str, int]) -> Dict[str, Any]:
        """Analyze retention policy compliance."""
        if not data_records:
            return {'retention_compliance': 1.0, 'policy_violations': 0}

        compliant_records = []
        expired_records = []
        policy_violations = []
        current_time = datetime.now()

        for record in data_records:
            record_type = record.get('type', 'default')
            created_at = record.get('created_at')
            retention_days = retention_policies.get(record_type, 365)

            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

                age_days = (current_time - created_at.replace(tzinfo=None)).days

                if age_days <= retention_days:
                    compliant_records.append({
                        'record_id': record.get('id'),
                        'age_days': age_days,
                        'retention_days': retention_days
                    })
                else:
                    expired_records.append({
                        'record_id': record.get('id'),
                        'type': record_type,
                        'age_days': age_days,
                        'retention_days': retention_days,
                        'overdue_days': age_days - retention_days
                    })

                    if not record.get('marked_for_deletion', False):
                        policy_violations.append({
                            'record_id': record.get('id'),
                            'violation': 'retention_exceeded_not_deleted'
                        })

        total = len(data_records)
        compliance = len(compliant_records) / total if total > 0 else 1

        return {
            'retention_compliance': compliance,
            'compliant_records': len(compliant_records),
            'expired_records': len(expired_records),
            'policy_violations': len(policy_violations),
            'total_records': total,
            'expired_details': expired_records[:20],
            'violation_details': policy_violations[:20]
        }


class AuditReadinessAnalyzer:
    """Analyzes readiness for external audits."""

    def analyze_readiness(self,
                         documentation: Dict[str, Any],
                         required_artifacts: List[str],
                         control_tests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze audit readiness."""
        # Check documentation completeness
        doc_coverage = len(set(documentation.keys()) & set(required_artifacts)) / len(required_artifacts) if required_artifacts else 1

        # Analyze documentation quality
        doc_quality = {}
        for artifact, content in documentation.items():
            if isinstance(content, str):
                word_count = len(content.split())
                doc_quality[artifact] = {
                    'exists': True,
                    'word_count': word_count,
                    'quality': 'good' if word_count > 100 else 'poor'
                }
            elif isinstance(content, dict):
                doc_quality[artifact] = {
                    'exists': True,
                    'fields': len(content),
                    'quality': 'good' if len(content) > 5 else 'adequate'
                }
            else:
                doc_quality[artifact] = {'exists': True, 'quality': 'adequate'}

        # Check missing artifacts
        missing_artifacts = list(set(required_artifacts) - set(documentation.keys()))

        # Analyze control tests
        control_analysis = {'tested': 0, 'passed': 0, 'failed': 0}
        if control_tests:
            for test in control_tests:
                control_analysis['tested'] += 1
                if test.get('passed', False):
                    control_analysis['passed'] += 1
                else:
                    control_analysis['failed'] += 1

        control_pass_rate = control_analysis['passed'] / control_analysis['tested'] if control_analysis['tested'] > 0 else 1

        # Calculate overall readiness
        readiness_score = (doc_coverage * 0.5) + (control_pass_rate * 0.5)

        return {
            'audit_readiness': readiness_score,
            'documentation_coverage': doc_coverage,
            'control_pass_rate': control_pass_rate,
            'missing_artifacts': missing_artifacts,
            'documentation_quality': doc_quality,
            'control_analysis': control_analysis,
            'readiness_status': 'ready' if readiness_score > 0.9 else ('partial' if readiness_score > 0.7 else 'not_ready')
        }


# ============================================================================
# Compliance AI Analyzers (18 Analysis Types)
# ============================================================================

class RegulatoryComplianceAnalyzer:
    """Analyzes regulatory compliance status."""

    def analyze_regulatory_compliance(self,
                                     requirements: List[ComplianceRequirement],
                                     compliance_status: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze regulatory compliance."""
        if not requirements:
            return {'regulatory_compliance': 1.0, 'non_compliant': 0}

        compliant = []
        non_compliant = []
        pending = []

        for req in requirements:
            status = compliance_status.get(req.requirement_id, {})
            is_compliant = status.get('compliant', False)
            status_type = status.get('status', 'unknown')

            req_data = {
                'requirement_id': req.requirement_id,
                'regulation': req.regulation,
                'mandatory': req.mandatory,
                'deadline': req.deadline.isoformat() if req.deadline else None
            }

            if is_compliant:
                compliant.append(req_data)
            elif status_type == 'in_progress':
                pending.append({**req_data, 'progress': status.get('progress', 0)})
            else:
                non_compliant.append({
                    **req_data,
                    'gaps': status.get('gaps', []),
                    'remediation_plan': status.get('remediation_plan')
                })

        total = len(requirements)
        mandatory_reqs = [r for r in requirements if r.mandatory]
        mandatory_compliant = len([c for c in compliant if c['mandatory']])

        compliance_rate = len(compliant) / total if total > 0 else 1
        mandatory_compliance = mandatory_compliant / len(mandatory_reqs) if mandatory_reqs else 1

        # Check for upcoming deadlines
        upcoming_deadlines = []
        current_time = datetime.now()
        for req in requirements:
            if req.deadline:
                days_until = (req.deadline - current_time).days
                if days_until > 0 and days_until <= 90:
                    status = compliance_status.get(req.requirement_id, {})
                    if not status.get('compliant', False):
                        upcoming_deadlines.append({
                            'requirement_id': req.requirement_id,
                            'regulation': req.regulation,
                            'days_until_deadline': days_until
                        })

        return {
            'regulatory_compliance': compliance_rate,
            'mandatory_compliance': mandatory_compliance,
            'compliant_requirements': len(compliant),
            'non_compliant_requirements': len(non_compliant),
            'pending_requirements': len(pending),
            'total_requirements': total,
            'upcoming_deadlines': upcoming_deadlines,
            'non_compliant_details': non_compliant,
            'pending_details': pending
        }


class StandardsAdherenceAnalyzer:
    """Analyzes adherence to industry standards."""

    def analyze_standards(self,
                         applicable_standards: List[Dict[str, Any]],
                         implementation_status: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze standards adherence."""
        if not applicable_standards:
            return {'standards_adherence': 1.0, 'non_adherent': 0}

        adherent = []
        non_adherent = []
        partial_adherent = []

        for standard in applicable_standards:
            standard_id = standard.get('id', 'unknown')
            standard_name = standard.get('name')
            controls = standard.get('controls', [])

            status = implementation_status.get(standard_id, {})
            implemented_controls = status.get('implemented_controls', [])

            if not controls:
                if status.get('adherent', False):
                    adherent.append({'standard_id': standard_id, 'name': standard_name})
                else:
                    non_adherent.append({'standard_id': standard_id, 'name': standard_name})
            else:
                coverage = len(set(implemented_controls) & set(controls)) / len(controls)

                standard_data = {
                    'standard_id': standard_id,
                    'name': standard_name,
                    'coverage': coverage,
                    'total_controls': len(controls),
                    'implemented': len(implemented_controls)
                }

                if coverage >= 0.95:
                    adherent.append(standard_data)
                elif coverage >= 0.5:
                    partial_adherent.append({
                        **standard_data,
                        'missing_controls': list(set(controls) - set(implemented_controls))
                    })
                else:
                    non_adherent.append(standard_data)

        total = len(applicable_standards)
        adherence = (len(adherent) + 0.5 * len(partial_adherent)) / total if total > 0 else 1

        return {
            'standards_adherence': adherence,
            'fully_adherent': len(adherent),
            'partially_adherent': len(partial_adherent),
            'non_adherent': len(non_adherent),
            'total_standards': total,
            'adherent_details': adherent,
            'partial_details': partial_adherent,
            'non_adherent_details': non_adherent
        }


class PolicyComplianceAnalyzer:
    """Analyzes internal policy compliance."""

    def analyze_policy_compliance(self,
                                  policies: List[Dict[str, Any]],
                                  violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze policy compliance."""
        if not policies:
            return {'policy_compliance': 1.0, 'total_violations': 0}

        policy_violation_map = defaultdict(list)
        for violation in violations:
            policy_id = violation.get('policy_id')
            policy_violation_map[policy_id].append(violation)

        policy_analysis = {}
        compliant_policies = []
        non_compliant_policies = []

        for policy in policies:
            policy_id = policy.get('id')
            policy_name = policy.get('name')
            policy_violations = policy_violation_map.get(policy_id, [])

            violation_count = len(policy_violations)
            severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            weighted_violations = sum(
                severity_weights.get(v.get('severity', 'low'), 1)
                for v in policy_violations
            )

            policy_data = {
                'policy_id': policy_id,
                'name': policy_name,
                'violation_count': violation_count,
                'weighted_violations': weighted_violations
            }

            policy_analysis[policy_id] = policy_data

            if violation_count == 0:
                compliant_policies.append(policy_data)
            else:
                non_compliant_policies.append({
                    **policy_data,
                    'violations': policy_violations[:5]
                })

        total_policies = len(policies)
        compliance_rate = len(compliant_policies) / total_policies if total_policies > 0 else 1

        # Calculate overall compliance score with severity weighting
        total_weighted = sum(p['weighted_violations'] for p in policy_analysis.values())
        max_possible = total_policies * 10  # Assuming max 10 weighted violations per policy is acceptable
        weighted_compliance = max(0, 1 - (total_weighted / max_possible))

        return {
            'policy_compliance': compliance_rate,
            'weighted_compliance': weighted_compliance,
            'compliant_policies': len(compliant_policies),
            'non_compliant_policies': len(non_compliant_policies),
            'total_policies': total_policies,
            'total_violations': len(violations),
            'policy_analysis': policy_analysis,
            'non_compliant_details': non_compliant_policies
        }


class CertificationAnalyzer:
    """Analyzes certification status and validity."""

    def analyze_certifications(self,
                               certifications: List[Dict[str, Any]],
                               required_certifications: List[str] = None) -> Dict[str, Any]:
        """Analyze certification status."""
        if not certifications:
            return {
                'certification_status': 0.0,
                'active_certifications': 0,
                'missing_certifications': required_certifications or []
            }

        current_time = datetime.now()
        active_certs = []
        expired_certs = []
        expiring_soon = []

        for cert in certifications:
            cert_id = cert.get('id')
            cert_name = cert.get('name')
            expiry_date = cert.get('expiry_date')
            is_active = cert.get('active', True)

            if expiry_date:
                if isinstance(expiry_date, str):
                    expiry_date = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
                expiry_date = expiry_date.replace(tzinfo=None)

                days_until_expiry = (expiry_date - current_time).days

                if days_until_expiry < 0:
                    expired_certs.append({
                        'cert_id': cert_id,
                        'name': cert_name,
                        'expired_days_ago': abs(days_until_expiry)
                    })
                elif days_until_expiry <= 90:
                    expiring_soon.append({
                        'cert_id': cert_id,
                        'name': cert_name,
                        'days_until_expiry': days_until_expiry
                    })
                    if is_active:
                        active_certs.append({'cert_id': cert_id, 'name': cert_name})
                elif is_active:
                    active_certs.append({
                        'cert_id': cert_id,
                        'name': cert_name,
                        'days_until_expiry': days_until_expiry
                    })
            elif is_active:
                active_certs.append({'cert_id': cert_id, 'name': cert_name})

        # Check for missing required certifications
        active_cert_names = {c.get('name') for c in active_certs}
        missing = []
        if required_certifications:
            missing = [c for c in required_certifications if c not in active_cert_names]

        total_required = len(required_certifications) if required_certifications else len(certifications)
        certification_status = len(active_certs) / total_required if total_required > 0 else 1

        return {
            'certification_status': certification_status,
            'active_certifications': len(active_certs),
            'expired_certifications': len(expired_certs),
            'expiring_soon': len(expiring_soon),
            'missing_certifications': missing,
            'total_certifications': len(certifications),
            'active_details': active_certs,
            'expired_details': expired_certs,
            'expiring_soon_details': expiring_soon
        }


class GapAnalyzer:
    """Analyzes compliance gaps and remediation status."""

    def analyze_gaps(self,
                    identified_gaps: List[Dict[str, Any]],
                    remediation_status: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze compliance gaps and remediation progress."""
        if not identified_gaps:
            return {'gap_remediation': 1.0, 'open_gaps': 0}

        remediated_gaps = []
        in_progress_gaps = []
        open_gaps = []

        for gap in identified_gaps:
            gap_id = gap.get('id')
            severity = gap.get('severity', 'medium')
            status = remediation_status.get(gap_id, {})

            gap_data = {
                'gap_id': gap_id,
                'severity': severity,
                'category': gap.get('category'),
                'description': gap.get('description')
            }

            remediation_state = status.get('state', 'open')

            if remediation_state == 'closed':
                remediated_gaps.append({
                    **gap_data,
                    'remediation_date': status.get('closed_date')
                })
            elif remediation_state == 'in_progress':
                in_progress_gaps.append({
                    **gap_data,
                    'progress': status.get('progress', 0),
                    'target_date': status.get('target_date')
                })
            else:
                open_gaps.append(gap_data)

        total = len(identified_gaps)
        remediation_rate = len(remediated_gaps) / total if total > 0 else 1

        # Calculate weighted remediation (accounting for in-progress)
        in_progress_credit = sum(g.get('progress', 0) / 100 for g in in_progress_gaps)
        weighted_remediation = (len(remediated_gaps) + in_progress_credit) / total if total > 0 else 1

        # Analyze by severity
        severity_analysis = defaultdict(lambda: {'total': 0, 'remediated': 0, 'open': 0})
        for gap in identified_gaps:
            sev = gap.get('severity', 'medium')
            severity_analysis[sev]['total'] += 1

        for gap in remediated_gaps:
            severity_analysis[gap['severity']]['remediated'] += 1

        for gap in open_gaps:
            severity_analysis[gap['severity']]['open'] += 1

        return {
            'gap_remediation': remediation_rate,
            'weighted_remediation': weighted_remediation,
            'remediated_gaps': len(remediated_gaps),
            'in_progress_gaps': len(in_progress_gaps),
            'open_gaps': len(open_gaps),
            'total_gaps': total,
            'severity_analysis': dict(severity_analysis),
            'open_gap_details': open_gaps,
            'in_progress_details': in_progress_gaps
        }


class ComplianceRiskAnalyzer:
    """Analyzes compliance-related risks."""

    def analyze_compliance_risk(self,
                                requirements: List[ComplianceRequirement],
                                compliance_status: Dict[str, Dict],
                                risk_factors: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze compliance risk exposure."""
        if not requirements:
            return {'compliance_risk': 0.0, 'high_risk_areas': []}

        risk_factors = risk_factors or {
            'regulatory': 1.5,
            'financial': 1.3,
            'operational': 1.0,
            'reputational': 1.2
        }

        risk_analysis = []
        high_risk_areas = []

        for req in requirements:
            status = compliance_status.get(req.requirement_id, {})
            is_compliant = status.get('compliant', False)

            if is_compliant:
                continue

            # Calculate risk score
            base_risk = 0.5  # Base non-compliance risk
            if req.mandatory:
                base_risk += 0.3

            # Apply deadline urgency
            if req.deadline:
                days_until = (req.deadline - datetime.now()).days
                if days_until < 0:
                    base_risk += 0.4  # Overdue
                elif days_until < 30:
                    base_risk += 0.3
                elif days_until < 90:
                    base_risk += 0.1

            # Apply category risk factor
            category_factor = risk_factors.get(req.category, 1.0)
            final_risk = min(1.0, base_risk * category_factor)

            risk_item = {
                'requirement_id': req.requirement_id,
                'regulation': req.regulation,
                'category': req.category,
                'risk_score': final_risk,
                'mandatory': req.mandatory,
                'deadline': req.deadline.isoformat() if req.deadline else None
            }

            risk_analysis.append(risk_item)

            if final_risk > 0.7:
                high_risk_areas.append(risk_item)

        # Calculate overall compliance risk
        if risk_analysis:
            overall_risk = np.mean([r['risk_score'] for r in risk_analysis])
        else:
            overall_risk = 0

        return {
            'compliance_risk': float(overall_risk),
            'high_risk_areas': high_risk_areas,
            'risk_items': len(risk_analysis),
            'critical_risks': len([r for r in risk_analysis if r['risk_score'] > 0.8]),
            'risk_analysis': sorted(risk_analysis, key=lambda x: x['risk_score'], reverse=True),
            'risk_level': 'critical' if overall_risk > 0.8 else ('high' if overall_risk > 0.6 else ('medium' if overall_risk > 0.3 else 'low'))
        }


# ============================================================================
# Report Generator
# ============================================================================

class AccountabilityReportGenerator:
    """Generates comprehensive accountability, audit, and compliance reports."""

    def __init__(self):
        self.responsibility_analyzer = ResponsibilityAnalyzer()
        self.ownership_analyzer = OwnershipAnalyzer()
        self.raci_analyzer = RACIAnalyzer()
        self.escalation_analyzer = EscalationAnalyzer()
        self.traceability_analyzer = DecisionTraceabilityAnalyzer()
        self.notification_analyzer = StakeholderNotificationAnalyzer()
        self.remediation_analyzer = RemediationAnalyzer()

        self.audit_trail_analyzer = AuditTrailAnalyzer()
        self.evidence_analyzer = EvidenceAnalyzer()
        self.integrity_analyzer = RecordIntegrityAnalyzer()
        self.log_coverage_analyzer = LogCoverageAnalyzer()
        self.retention_analyzer = RetentionAnalyzer()
        self.readiness_analyzer = AuditReadinessAnalyzer()

        self.regulatory_analyzer = RegulatoryComplianceAnalyzer()
        self.standards_analyzer = StandardsAdherenceAnalyzer()
        self.policy_analyzer = PolicyComplianceAnalyzer()
        self.certification_analyzer = CertificationAnalyzer()
        self.gap_analyzer = GapAnalyzer()
        self.compliance_risk_analyzer = ComplianceRiskAnalyzer()

    def generate_accountability_report(self,
                                       decisions: List[Dict[str, Any]] = None,
                                       resources: List[Dict[str, Any]] = None,
                                       raci_entries: List[RACIEntry] = None,
                                       responsibility_matrix: Dict[str, List[str]] = None,
                                       owner_registry: Dict[str, str] = None) -> Dict[str, Any]:
        """Generate accountability analysis report."""
        report = {
            'report_type': 'accountability_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if decisions is not None:
            report['responsibility'] = self.responsibility_analyzer.analyze_responsibility(
                decisions, responsibility_matrix or {}
            )
            report['traceability'] = self.traceability_analyzer.analyze_traceability(decisions)

        if resources is not None:
            report['ownership'] = self.ownership_analyzer.analyze_ownership(
                resources, owner_registry or {}
            )

        if raci_entries:
            report['raci'] = self.raci_analyzer.analyze_raci(raci_entries)

        return report

    def generate_audit_report(self,
                             audit_records: List[AuditRecord] = None,
                             components: List[Dict[str, Any]] = None,
                             log_records: List[Dict[str, Any]] = None,
                             documentation: Dict[str, Any] = None,
                             required_artifacts: List[str] = None) -> Dict[str, Any]:
        """Generate audit analysis report."""
        report = {
            'report_type': 'audit_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if audit_records:
            report['audit_trail'] = self.audit_trail_analyzer.analyze_audit_trail(audit_records)

        if components and log_records:
            report['log_coverage'] = self.log_coverage_analyzer.analyze_log_coverage(
                components, log_records
            )

        if documentation and required_artifacts:
            report['readiness'] = self.readiness_analyzer.analyze_readiness(
                documentation, required_artifacts
            )

        return report

    def generate_compliance_report(self,
                                   requirements: List[ComplianceRequirement] = None,
                                   compliance_status: Dict[str, Dict] = None,
                                   policies: List[Dict[str, Any]] = None,
                                   violations: List[Dict[str, Any]] = None,
                                   certifications: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate compliance analysis report."""
        report = {
            'report_type': 'compliance_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if requirements and compliance_status:
            report['regulatory'] = self.regulatory_analyzer.analyze_regulatory_compliance(
                requirements, compliance_status
            )
            report['compliance_risk'] = self.compliance_risk_analyzer.analyze_compliance_risk(
                requirements, compliance_status
            )

        if policies is not None:
            report['policy'] = self.policy_analyzer.analyze_policy_compliance(
                policies, violations or []
            )

        if certifications:
            report['certifications'] = self.certification_analyzer.analyze_certifications(
                certifications
            )

        return report

    def generate_full_report(self,
                            decisions: List[Dict[str, Any]] = None,
                            resources: List[Dict[str, Any]] = None,
                            raci_entries: List[RACIEntry] = None,
                            audit_records: List[AuditRecord] = None,
                            requirements: List[ComplianceRequirement] = None,
                            compliance_status: Dict[str, Dict] = None,
                            policies: List[Dict[str, Any]] = None,
                            violations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive accountability, audit, and compliance report."""
        report = {
            'report_type': 'comprehensive_accountability_compliance_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        # Accountability analysis
        if decisions or resources or raci_entries:
            acc_report = self.generate_accountability_report(
                decisions, resources, raci_entries
            )
            for key, value in acc_report.items():
                if key not in ['report_type', 'timestamp']:
                    report[f'accountability_{key}'] = value

        # Audit analysis
        if audit_records:
            audit_report = self.generate_audit_report(audit_records=audit_records)
            for key, value in audit_report.items():
                if key not in ['report_type', 'timestamp']:
                    report[f'audit_{key}'] = value

        # Compliance analysis
        if requirements or policies:
            comp_report = self.generate_compliance_report(
                requirements, compliance_status, policies, violations
            )
            for key, value in comp_report.items():
                if key not in ['report_type', 'timestamp']:
                    report[f'compliance_{key}'] = value

        # Calculate overall scores
        scores = []
        if 'accountability_responsibility' in report:
            scores.append(report['accountability_responsibility'].get('responsibility_clarity', 0))
        if 'audit_audit_trail' in report:
            scores.append(report['audit_audit_trail'].get('audit_completeness', 0))
        if 'compliance_regulatory' in report:
            scores.append(report['compliance_regulatory'].get('regulatory_compliance', 0))

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
            f"# {report.get('report_type', 'Accountability Analysis Report')}",
            f"\n**Generated:** {report.get('timestamp', 'N/A')}",
            f"\n**Overall Score:** {report.get('overall_score', 0):.2%}",
            "\n---\n"
        ]

        if 'accountability_responsibility' in report:
            lines.append("## Accountability Analysis\n")
            resp = report['accountability_responsibility']
            lines.append(f"- **Responsibility Clarity:** {resp.get('responsibility_clarity', 0):.2%}")
            lines.append(f"- **Unassigned Decisions:** {resp.get('unassigned_decisions', 0)}")
            lines.append("")

        if 'audit_audit_trail' in report:
            lines.append("## Audit Analysis\n")
            audit = report['audit_audit_trail']
            lines.append(f"- **Audit Completeness:** {audit.get('audit_completeness', 0):.2%}")
            lines.append(f"- **Event Coverage:** {audit.get('event_coverage', 0):.2%}")
            lines.append("")

        if 'compliance_regulatory' in report:
            lines.append("## Compliance Analysis\n")
            reg = report['compliance_regulatory']
            lines.append(f"- **Regulatory Compliance:** {reg.get('regulatory_compliance', 0):.2%}")
            lines.append(f"- **Non-Compliant Requirements:** {reg.get('non_compliant_requirements', 0)}")
            lines.append("")

        return "\n".join(lines)
