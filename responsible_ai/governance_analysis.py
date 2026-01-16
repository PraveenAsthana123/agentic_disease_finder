"""
Governance AI Analysis Module
==============================

Comprehensive analysis for AI governance frameworks.
Implements 20 analysis types for AI governance.

Analysis Types:
1. Governance Scope & Authority Definition
2. Ownership & Role Assignment
3. RACI & Decision Rights Mapping
4. Policy Framework Alignment
5. Use-Case Intake & Approval Governance
6. Risk Management Governance
7. Ethics & Values Governance
8. Compliance & Regulatory Governance
9. Data Governance Integration
10. Model Lifecycle Governance
11. Monitoring & Oversight Governance
12. Incident & Escalation Governance
13. Human-in-the-Loop Governance
14. Vendor & Third-Party Governance
15. Change Management & Version Control
16. Documentation & Evidence Governance
17. Transparency & Disclosure Governance
18. Performance & KPI Governance
19. Governance Drift & Effectiveness Review
20. Governance Enforcement & Sanctions
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
class GovernanceRole:
    """Represents a governance role."""
    role_id: str
    role_name: str
    responsibilities: List[str] = field(default_factory=list)
    authority_level: str = "operational"  # strategic, tactical, operational
    assigned_to: str = ""


@dataclass
class RACIEntry:
    """RACI matrix entry."""
    activity: str
    responsible: str = ""
    accountable: str = ""
    consulted: List[str] = field(default_factory=list)
    informed: List[str] = field(default_factory=list)


@dataclass
class PolicyMapping:
    """Policy mapping to AI system."""
    policy_id: str
    policy_name: str
    policy_type: str  # responsible_ai, privacy, security, safety
    applicability: str = "full"  # full, partial, none
    compliance_status: str = "unknown"


# ============================================================================
# Governance Scope Analyzer (Type 1)
# ============================================================================

class GovernanceScopeAnalyzer:
    """Analyzes governance scope and authority."""

    def analyze_scope(self,
                     ai_use_cases: List[Dict[str, Any]],
                     governance_charter: Dict[str, Any] = None) -> Dict[str, Any]:
        """Define what decisions governance controls."""
        if not ai_use_cases:
            return {'scope': {}, 'authority': {}}

        # Categorize use cases
        in_scope = []
        advisory_scope = []
        out_of_scope = []

        for use_case in ai_use_cases:
            risk_level = use_case.get('risk_level', 'medium')
            if risk_level in ['high', 'critical']:
                in_scope.append(use_case)
            elif risk_level == 'medium':
                advisory_scope.append(use_case)
            else:
                out_of_scope.append(use_case)

        # Define authority levels
        authority = {
            'decision_rights': governance_charter.get('decision_rights', []) if governance_charter else [],
            'advisory_roles': governance_charter.get('advisory_roles', []) if governance_charter else [],
            'escalation_authority': governance_charter.get('escalation_authority', '') if governance_charter else ''
        }

        return {
            'governance_scope_charter': {
                'in_scope_use_cases': len(in_scope),
                'advisory_scope_use_cases': len(advisory_scope),
                'out_of_scope_use_cases': len(out_of_scope),
                'total_use_cases': len(ai_use_cases)
            },
            'scope_coverage': len(in_scope) / len(ai_use_cases) if ai_use_cases else 0,
            'authority_definition': authority,
            'scope_statement': f"Governance covers {len(in_scope)} high-risk AI use cases"
        }


# ============================================================================
# Ownership Analyzer (Type 2)
# ============================================================================

class OwnershipAnalyzer:
    """Analyzes ownership and role assignment."""

    REQUIRED_ROLES = [
        'product_owner',
        'data_owner',
        'model_owner',
        'risk_owner',
        'ethics_owner'
    ]

    def analyze_ownership(self,
                         role_assignments: List[GovernanceRole]) -> Dict[str, Any]:
        """Analyze who owns AI decisions end-to-end."""
        if not role_assignments:
            return {'ownership_register': [], 'gaps': self.REQUIRED_ROLES}

        # Check role coverage
        assigned_roles = {r.role_name.lower() for r in role_assignments}
        missing_roles = [r for r in self.REQUIRED_ROLES if r not in assigned_roles]

        # Build ownership register
        ownership_register = []
        for role in role_assignments:
            ownership_register.append({
                'role': role.role_name,
                'assigned_to': role.assigned_to,
                'authority_level': role.authority_level,
                'responsibilities': role.responsibilities,
                'is_named': bool(role.assigned_to)
            })

        # Calculate coverage
        named_owners = sum(1 for r in role_assignments if r.assigned_to)
        coverage = named_owners / len(self.REQUIRED_ROLES)

        return {
            'ownership_register': ownership_register,
            'ownership_gaps': missing_roles,
            'coverage_score': float(coverage),
            'named_accountability': all(r.assigned_to for r in role_assignments),
            'recommendations': self._generate_recommendations(missing_roles)
        }

    def _generate_recommendations(self, gaps: List[str]) -> List[str]:
        recs = []
        for gap in gaps:
            recs.append(f"Assign named owner for {gap.replace('_', ' ')}")
        return recs


# ============================================================================
# RACI Analyzer (Type 3)
# ============================================================================

class RACIAnalyzer:
    """Analyzes RACI and decision rights mapping."""

    def analyze_raci(self,
                    raci_entries: List[RACIEntry],
                    lifecycle_stages: List[str] = None) -> Dict[str, Any]:
        """Analyze RACI mapping per lifecycle stage."""
        if not raci_entries:
            return {'raci_matrix': [], 'completeness': 0}

        lifecycle_stages = lifecycle_stages or [
            'data_collection', 'model_development', 'deployment',
            'monitoring', 'retirement'
        ]

        # Analyze RACI completeness
        complete_entries = 0
        conflict_count = 0
        raci_matrix = []

        for entry in raci_entries:
            is_complete = all([entry.responsible, entry.accountable])
            has_conflict = entry.responsible == entry.accountable and len(entry.consulted) == 0

            if is_complete:
                complete_entries += 1
            if has_conflict:
                conflict_count += 1

            raci_matrix.append({
                'activity': entry.activity,
                'responsible': entry.responsible,
                'accountable': entry.accountable,
                'consulted': entry.consulted,
                'informed': entry.informed,
                'is_complete': is_complete
            })

        completeness = complete_entries / len(raci_entries) if raci_entries else 0

        return {
            'ai_raci_matrix': raci_matrix,
            'completeness_score': float(completeness),
            'conflict_count': conflict_count,
            'lifecycle_stages_covered': len(set(e.activity for e in raci_entries)),
            'conflict_resolution_paths': self._identify_conflicts(raci_entries)
        }

    def _identify_conflicts(self, entries: List[RACIEntry]) -> List[Dict]:
        conflicts = []
        for entry in entries:
            if entry.responsible in entry.consulted:
                conflicts.append({
                    'activity': entry.activity,
                    'conflict': 'Responsible also in Consulted'
                })
        return conflicts


# ============================================================================
# Policy Framework Analyzer (Type 4)
# ============================================================================

class PolicyFrameworkAnalyzer:
    """Analyzes policy framework alignment."""

    def analyze_policy_alignment(self,
                                policies: List[PolicyMapping],
                                ai_system: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which policies apply and how."""
        if not policies:
            return {'policy_mapping': [], 'alignment_score': 0}

        policy_mapping = []
        fully_applicable = 0

        for policy in policies:
            mapping = {
                'policy_id': policy.policy_id,
                'policy_name': policy.policy_name,
                'policy_type': policy.policy_type,
                'applicability': policy.applicability,
                'compliance_status': policy.compliance_status
            }

            if policy.applicability == 'full':
                fully_applicable += 1

            policy_mapping.append(mapping)

        # Group by type
        by_type = defaultdict(list)
        for p in policies:
            by_type[p.policy_type].append(p)

        return {
            'policy_mapping_document': policy_mapping,
            'policy_coverage': {
                'total_policies': len(policies),
                'fully_applicable': fully_applicable,
                'by_type': {k: len(v) for k, v in by_type.items()}
            },
            'alignment_score': fully_applicable / len(policies) if policies else 0
        }


# ============================================================================
# Use-Case Approval Governance (Type 5)
# ============================================================================

class UseCaseApprovalAnalyzer:
    """Analyzes use-case intake and approval governance."""

    def analyze_approval_process(self,
                                use_case_submissions: List[Dict[str, Any]],
                                approval_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze how new AI use cases are approved."""
        if not use_case_submissions:
            return {'approval_records': [], 'process_metrics': {}}

        approved = [u for u in use_case_submissions if u.get('status') == 'approved']
        rejected = [u for u in use_case_submissions if u.get('status') == 'rejected']
        pending = [u for u in use_case_submissions if u.get('status') == 'pending']

        # Risk tiering analysis
        by_risk_tier = defaultdict(list)
        for u in use_case_submissions:
            by_risk_tier[u.get('risk_tier', 'unclassified')].append(u)

        # Approval time analysis
        approval_times = [u.get('approval_time_days', 0) for u in approved if u.get('approval_time_days')]
        avg_approval_time = np.mean(approval_times) if approval_times else 0

        return {
            'use_case_approval_record': {
                'total_submissions': len(use_case_submissions),
                'approved': len(approved),
                'rejected': len(rejected),
                'pending': len(pending),
                'approval_rate': len(approved) / len(use_case_submissions) if use_case_submissions else 0
            },
            'risk_tiering': {k: len(v) for k, v in by_risk_tier.items()},
            'process_metrics': {
                'avg_approval_time_days': float(avg_approval_time),
                'business_justification_required': approval_config.get('requires_justification', True) if approval_config else True
            }
        }


# ============================================================================
# Risk Management Governance (Type 6)
# ============================================================================

class RiskGovernanceAnalyzer:
    """Analyzes risk management governance."""

    def analyze_risk_governance(self,
                               risk_register: List[Dict[str, Any]],
                               risk_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze how AI risks are identified and owned."""
        if not risk_register:
            return {'risk_register': [], 'governance_metrics': {}}

        # Analyze risk ownership
        owned_risks = sum(1 for r in risk_register if r.get('owner'))
        assessed_risks = sum(1 for r in risk_register if r.get('severity') and r.get('likelihood'))

        # Risk distribution
        by_severity = defaultdict(int)
        for r in risk_register:
            by_severity[r.get('severity', 'unknown')] += 1

        return {
            'ai_risk_register': risk_register,
            'governance_metrics': {
                'total_risks': len(risk_register),
                'owned_risks': owned_risks,
                'assessed_risks': assessed_risks,
                'ownership_rate': owned_risks / len(risk_register) if risk_register else 0
            },
            'risk_distribution': dict(by_severity),
            'governance_completeness': assessed_risks / len(risk_register) if risk_register else 0
        }


# ============================================================================
# Ethics Governance (Type 7)
# ============================================================================

class EthicsGovernanceAnalyzer:
    """Analyzes ethics and values governance."""

    def analyze_ethics_governance(self,
                                 ethics_config: Dict[str, Any],
                                 ethics_decisions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze how value conflicts are resolved."""
        has_review_board = ethics_config.get('ethics_review_board', False)
        escalation_criteria = ethics_config.get('escalation_criteria', [])
        ethics_principles = ethics_config.get('principles', [])

        # Analyze ethics decisions
        decisions_log = []
        if ethics_decisions:
            for decision in ethics_decisions:
                decisions_log.append({
                    'decision_id': decision.get('id', ''),
                    'issue': decision.get('issue', ''),
                    'resolution': decision.get('resolution', ''),
                    'escalated': decision.get('escalated', False)
                })

        return {
            'ethics_decision_log': decisions_log,
            'governance_structure': {
                'has_ethics_review_board': has_review_board,
                'escalation_criteria_defined': len(escalation_criteria) > 0,
                'principles_documented': len(ethics_principles)
            },
            'ethics_coverage': {
                'total_decisions': len(ethics_decisions) if ethics_decisions else 0,
                'escalated': sum(1 for d in decisions_log if d.get('escalated', False))
            }
        }


# ============================================================================
# Additional Governance Analyzers (Types 8-20)
# ============================================================================

class ComplianceGovernanceAnalyzer:
    """Analyzes compliance and regulatory governance (Type 8)."""

    def analyze_compliance_governance(self,
                                     jurisdictions: List[str],
                                     compliance_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze how legal compliance is enforced."""
        return {
            'compliance_governance_records': {
                'jurisdictions_covered': jurisdictions,
                'compliance_sign_offs': sum(1 for r in (compliance_records or []) if r.get('signed_off', False)),
                'total_requirements': len(compliance_records) if compliance_records else 0
            }
        }


class DataGovernanceIntegrationAnalyzer:
    """Analyzes data governance integration (Type 9)."""

    def analyze_data_governance(self,
                               data_governance_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how data is governed for AI."""
        return {
            'data_governance_linkage': {
                'data_ownership_defined': data_governance_config.get('data_owners', []) != [],
                'lineage_tracking': data_governance_config.get('lineage_tracking', False),
                'access_control': data_governance_config.get('access_control', 'none')
            }
        }


class LifecycleGovernanceAnalyzer:
    """Analyzes model lifecycle governance (Type 10)."""

    def analyze_lifecycle_governance(self,
                                    lifecycle_config: Dict[str, Any],
                                    gate_records: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze who controls changes across lifecycle."""
        gates = lifecycle_config.get('gates', [])

        return {
            'lifecycle_governance_checkpoints': {
                'total_gates': len(gates),
                'gates_with_approvers': sum(1 for g in gates if g.get('approver')),
                'gate_records': gate_records or []
            }
        }


class MonitoringGovernanceAnalyzer:
    """Analyzes monitoring and oversight governance (Type 11)."""

    def analyze_monitoring_governance(self,
                                     monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze who acts on monitoring signals."""
        return {
            'monitoring_governance_plan': {
                'drift_alert_ownership': monitoring_config.get('drift_alert_owner', ''),
                'response_sla_hours': monitoring_config.get('response_sla_hours', 0),
                'escalation_path': monitoring_config.get('escalation_path', [])
            }
        }


class IncidentGovernanceAnalyzer:
    """Analyzes incident and escalation governance (Type 12)."""

    def analyze_incident_governance(self,
                                   incident_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze who responds when AI fails."""
        return {
            'incident_governance_sop': {
                'severity_levels': incident_config.get('severity_levels', []),
                'escalation_authority': incident_config.get('escalation_authority', ''),
                'response_procedures': incident_config.get('response_procedures', [])
            }
        }


class HITLGovernanceAnalyzer:
    """Analyzes human-in-the-loop governance (Type 13)."""

    def analyze_hitl_governance(self,
                               hitl_policy: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze when humans must intervene by policy."""
        return {
            'hitl_governance_policy': {
                'mandatory_review_thresholds': hitl_policy.get('review_thresholds', {}),
                'override_authority': hitl_policy.get('override_authority', ''),
                'intervention_triggers': hitl_policy.get('intervention_triggers', [])
            }
        }


class VendorGovernanceAnalyzer:
    """Analyzes vendor and third-party governance (Type 14)."""

    def analyze_vendor_governance(self,
                                 vendors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how vendors are governed."""
        return {
            'vendor_governance_assessment': {
                'total_vendors': len(vendors),
                'due_diligence_completed': sum(1 for v in vendors if v.get('due_diligence', False)),
                'contractual_controls': sum(1 for v in vendors if v.get('contractual_controls', False))
            }
        }


class ChangeGovernanceAnalyzer:
    """Analyzes change management governance (Type 15)."""

    def analyze_change_governance(self,
                                 change_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how changes are approved and tracked."""
        return {
            'change_approval_log': change_records,
            'metrics': {
                'total_changes': len(change_records),
                'approved_changes': sum(1 for c in change_records if c.get('approved', False))
            }
        }


class DocumentationGovernanceAnalyzer:
    """Analyzes documentation and evidence governance (Type 16)."""

    def analyze_documentation_governance(self,
                                        documentation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what evidence must be retained."""
        return {
            'evidence_index': {
                'audit_trails_enabled': documentation_config.get('audit_trails', False),
                'retention_rules': documentation_config.get('retention_rules', {}),
                'required_artifacts': documentation_config.get('required_artifacts', [])
            }
        }


class TransparencyGovernanceAnalyzer:
    """Analyzes transparency and disclosure governance (Type 17)."""

    def analyze_transparency_governance(self,
                                       disclosure_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what is disclosed and by whom."""
        return {
            'disclosure_governance_policy': {
                'user_disclosures': disclosure_config.get('user_disclosures', []),
                'public_reporting': disclosure_config.get('public_reporting', False),
                'disclosure_owner': disclosure_config.get('disclosure_owner', '')
            }
        }


class KPIGovernanceAnalyzer:
    """Analyzes performance and KPI governance (Type 18)."""

    def analyze_kpi_governance(self,
                              kpis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what KPIs matter to leadership."""
        return {
            'governance_kpi_dashboard': {
                'risk_kpis': [k for k in kpis if k.get('category') == 'risk'],
                'trust_kpis': [k for k in kpis if k.get('category') == 'trust'],
                'quality_kpis': [k for k in kpis if k.get('category') == 'quality'],
                'total_kpis': len(kpis)
            }
        }


class GovernanceDriftAnalyzer:
    """Analyzes governance drift and effectiveness (Type 19)."""

    def analyze_governance_drift(self,
                                policy_adherence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if governance is weakening over time."""
        if not policy_adherence:
            return {'drift_detected': False, 'effectiveness_report': {}}

        # Calculate adherence trend
        adherence_rates = [p.get('adherence_rate', 0) for p in policy_adherence]

        if len(adherence_rates) >= 2:
            first_half = np.mean(adherence_rates[:len(adherence_rates)//2])
            second_half = np.mean(adherence_rates[len(adherence_rates)//2:])
            drift = first_half - second_half
        else:
            drift = 0

        # Detect bypasses
        bypasses = sum(1 for p in policy_adherence if p.get('bypass_detected', False))

        return {
            'governance_effectiveness_report': {
                'avg_adherence_rate': float(np.mean(adherence_rates)) if adherence_rates else 0,
                'drift_score': float(drift),
                'drift_detected': drift > 0.1,
                'bypass_count': bypasses
            }
        }


class GovernanceEnforcementAnalyzer:
    """Analyzes governance enforcement and sanctions (Type 20)."""

    def analyze_enforcement(self,
                           enforcement_records: List[Dict[str, Any]],
                           enforcement_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze what happens when rules are violated."""
        return {
            'enforcement_sanctions_log': enforcement_records,
            'enforcement_config': {
                'sanctions_defined': enforcement_config.get('sanctions', []) if enforcement_config else [],
                'kill_switch_authority': enforcement_config.get('kill_switch_authority', '') if enforcement_config else ''
            },
            'metrics': {
                'total_violations': len(enforcement_records),
                'sanctions_applied': sum(1 for e in enforcement_records if e.get('sanction_applied', False))
            }
        }


# ============================================================================
# Report Generator
# ============================================================================

class GovernanceReportGenerator:
    """Generates comprehensive governance analysis reports."""

    def __init__(self):
        self.scope_analyzer = GovernanceScopeAnalyzer()
        self.ownership_analyzer = OwnershipAnalyzer()
        self.raci_analyzer = RACIAnalyzer()
        self.policy_analyzer = PolicyFrameworkAnalyzer()
        self.risk_analyzer = RiskGovernanceAnalyzer()
        self.ethics_analyzer = EthicsGovernanceAnalyzer()
        self.drift_analyzer = GovernanceDriftAnalyzer()

    def generate_full_report(self,
                            ai_use_cases: List[Dict[str, Any]] = None,
                            role_assignments: List[GovernanceRole] = None,
                            raci_entries: List[RACIEntry] = None,
                            risk_register: List[Dict[str, Any]] = None,
                            policy_adherence: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive governance report."""
        report = {
            'report_type': 'governance_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if ai_use_cases:
            report['scope'] = self.scope_analyzer.analyze_scope(ai_use_cases)

        if role_assignments:
            report['ownership'] = self.ownership_analyzer.analyze_ownership(role_assignments)

        if raci_entries:
            report['raci'] = self.raci_analyzer.analyze_raci(raci_entries)

        if risk_register:
            report['risk_governance'] = self.risk_analyzer.analyze_risk_governance(risk_register)

        if policy_adherence:
            report['governance_drift'] = self.drift_analyzer.analyze_governance_drift(policy_adherence)

        # Calculate overall governance score
        scores = []
        if 'ownership' in report:
            scores.append(report['ownership'].get('coverage_score', 0))
        if 'raci' in report:
            scores.append(report['raci'].get('completeness_score', 0))
        if 'scope' in report:
            scores.append(report['scope'].get('scope_coverage', 0))

        report['governance_score'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
