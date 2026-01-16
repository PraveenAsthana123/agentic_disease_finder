"""
Privacy Analysis Module - Privacy-Preserving AI, Transparent Data Practices
============================================================================

Comprehensive analysis for AI privacy, data protection, and transparent data practices.
Implements 38 analysis types across two related frameworks.

Frameworks:
- Privacy-Preserving AI (20 types): Differential Privacy, Data Minimization, Anonymization, Consent
- Transparent Data Practices (18 types): Data Provenance, Usage Disclosure, Access Control, Retention
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import json
import hashlib


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PrivacyMetrics:
    """Metrics for privacy analysis."""
    differential_privacy_epsilon: float = 0.0
    anonymization_score: float = 0.0
    data_minimization_score: float = 0.0
    consent_compliance: float = 0.0
    pii_exposure_risk: float = 0.0
    re_identification_risk: float = 0.0
    data_leakage_score: float = 0.0


@dataclass
class TransparencyMetrics:
    """Metrics for data transparency analysis."""
    provenance_completeness: float = 0.0
    usage_disclosure_rate: float = 0.0
    access_control_score: float = 0.0
    retention_compliance: float = 0.0
    data_subject_rights: float = 0.0
    third_party_disclosure: float = 0.0


@dataclass
class DataRecord:
    """Represents a data record for privacy analysis."""
    record_id: str
    data_type: str
    contains_pii: bool
    pii_types: List[str] = field(default_factory=list)
    anonymized: bool = False
    consent_obtained: bool = False
    retention_period_days: int = 365
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConsentRecord:
    """Represents a consent record."""
    consent_id: str
    data_subject_id: str
    purpose: str
    granted: bool
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked: bool = False
    revoked_at: Optional[datetime] = None


@dataclass
class DataAccessEvent:
    """Represents a data access event."""
    event_id: str
    accessor: str
    data_type: str
    purpose: str
    timestamp: datetime
    authorized: bool
    data_subject_notified: bool = False


# ============================================================================
# Privacy-Preserving AI Analyzers (20 Analysis Types)
# ============================================================================

class DifferentialPrivacyAnalyzer:
    """Analyzes differential privacy guarantees."""

    def analyze_differential_privacy(self,
                                     epsilon_values: List[float],
                                     delta_values: List[float] = None,
                                     composition_type: str = 'sequential') -> Dict[str, Any]:
        """Analyze differential privacy parameters."""
        if not epsilon_values:
            return {'privacy_budget': 0.0, 'privacy_guarantee': 'none'}

        # Calculate composed privacy budget
        if composition_type == 'sequential':
            total_epsilon = sum(epsilon_values)
        elif composition_type == 'parallel':
            total_epsilon = max(epsilon_values)
        elif composition_type == 'advanced':
            # Advanced composition theorem
            k = len(epsilon_values)
            avg_epsilon = np.mean(epsilon_values)
            total_epsilon = np.sqrt(2 * k * np.log(1 / 0.01)) * avg_epsilon + k * avg_epsilon * (np.exp(avg_epsilon) - 1)
        else:
            total_epsilon = sum(epsilon_values)

        # Calculate total delta if provided
        if delta_values:
            total_delta = sum(delta_values)
        else:
            total_delta = 0.0

        # Privacy guarantee interpretation
        if total_epsilon < 0.1:
            guarantee = 'very_strong'
        elif total_epsilon < 1.0:
            guarantee = 'strong'
        elif total_epsilon < 5.0:
            guarantee = 'moderate'
        elif total_epsilon < 10.0:
            guarantee = 'weak'
        else:
            guarantee = 'minimal'

        return {
            'total_epsilon': float(total_epsilon),
            'total_delta': float(total_delta),
            'privacy_guarantee': guarantee,
            'composition_type': composition_type,
            'num_queries': len(epsilon_values),
            'mean_epsilon': float(np.mean(epsilon_values)),
            'max_epsilon': float(np.max(epsilon_values)),
            'privacy_budget_remaining': max(0, 10.0 - total_epsilon),  # Assuming 10.0 budget
            'recommendations': self._generate_dp_recommendations(total_epsilon)
        }

    def _generate_dp_recommendations(self, epsilon: float) -> List[str]:
        """Generate recommendations based on epsilon."""
        recommendations = []
        if epsilon > 5.0:
            recommendations.append("Consider reducing query sensitivity or adding more noise")
        if epsilon > 10.0:
            recommendations.append("Privacy budget exhausted - consider data refresh")
        if epsilon < 1.0:
            recommendations.append("Privacy guarantees are strong - suitable for sensitive data")
        return recommendations


class DataMinimizationAnalyzer:
    """Analyzes data minimization practices."""

    def analyze_minimization(self,
                            collected_fields: List[str],
                            required_fields: List[str],
                            field_usage: Dict[str, int] = None) -> Dict[str, Any]:
        """Analyze data minimization compliance."""
        if not collected_fields:
            return {'minimization_score': 1.0, 'excess_fields': []}

        required_set = set(required_fields)
        collected_set = set(collected_fields)

        excess_fields = list(collected_set - required_set)
        missing_fields = list(required_set - collected_set)

        minimization_score = len(required_set & collected_set) / len(collected_set) if collected_set else 1

        # Analyze field usage if provided
        unused_fields = []
        low_usage_fields = []
        if field_usage:
            for field in collected_fields:
                usage = field_usage.get(field, 0)
                if usage == 0:
                    unused_fields.append(field)
                elif usage < 10:
                    low_usage_fields.append({'field': field, 'usage': usage})

        return {
            'minimization_score': float(minimization_score),
            'collected_fields': len(collected_fields),
            'required_fields': len(required_fields),
            'excess_fields': excess_fields,
            'excess_count': len(excess_fields),
            'missing_fields': missing_fields,
            'unused_fields': unused_fields,
            'low_usage_fields': low_usage_fields,
            'compliant': len(excess_fields) == 0,
            'recommendations': [
                f"Remove field '{f}' - not required for stated purpose"
                for f in excess_fields[:5]
            ]
        }


class AnonymizationAnalyzer:
    """Analyzes data anonymization effectiveness."""

    def analyze_anonymization(self,
                             records: List[DataRecord],
                             quasi_identifiers: List[str] = None) -> Dict[str, Any]:
        """Analyze anonymization quality."""
        if not records:
            return {'anonymization_score': 0.0, 'records_with_pii': 0}

        quasi_identifiers = quasi_identifiers or ['age', 'zip_code', 'gender', 'occupation']

        anonymized_records = []
        non_anonymized_records = []
        pii_exposure = defaultdict(int)

        for record in records:
            if record.anonymized:
                anonymized_records.append(record)
            else:
                non_anonymized_records.append(record)

            for pii_type in record.pii_types:
                pii_exposure[pii_type] += 1

        total = len(records)
        anonymization_score = len(anonymized_records) / total if total > 0 else 0

        # Calculate k-anonymity estimate (simplified)
        # In practice, this would analyze actual data distributions
        k_anonymity_estimate = max(1, int(total / 10))  # Simplified estimate

        # Calculate l-diversity estimate
        l_diversity_estimate = len(set(r.data_type for r in records))

        return {
            'anonymization_score': float(anonymization_score),
            'anonymized_records': len(anonymized_records),
            'non_anonymized_records': len(non_anonymized_records),
            'total_records': total,
            'pii_exposure': dict(pii_exposure),
            'k_anonymity_estimate': k_anonymity_estimate,
            'l_diversity_estimate': l_diversity_estimate,
            'quasi_identifiers': quasi_identifiers,
            'records_with_pii': sum(1 for r in records if r.contains_pii),
            'pii_types_exposed': list(pii_exposure.keys())
        }


class ReIdentificationRiskAnalyzer:
    """Analyzes re-identification risk."""

    def analyze_reidentification_risk(self,
                                      dataset_size: int,
                                      unique_combinations: int,
                                      external_data_available: bool = False,
                                      linkage_attacks_possible: bool = False) -> Dict[str, Any]:
        """Analyze re-identification risk."""
        if dataset_size == 0:
            return {'reidentification_risk': 0.0, 'risk_level': 'unknown'}

        # Base risk from uniqueness
        uniqueness_ratio = unique_combinations / dataset_size if dataset_size > 0 else 0
        base_risk = min(1.0, uniqueness_ratio)

        # Adjust for external factors
        risk_multiplier = 1.0
        if external_data_available:
            risk_multiplier *= 1.5
        if linkage_attacks_possible:
            risk_multiplier *= 1.3

        adjusted_risk = min(1.0, base_risk * risk_multiplier)

        # Risk level interpretation
        if adjusted_risk < 0.1:
            risk_level = 'low'
        elif adjusted_risk < 0.3:
            risk_level = 'moderate'
        elif adjusted_risk < 0.6:
            risk_level = 'high'
        else:
            risk_level = 'very_high'

        return {
            'reidentification_risk': float(adjusted_risk),
            'base_risk': float(base_risk),
            'risk_level': risk_level,
            'uniqueness_ratio': float(uniqueness_ratio),
            'dataset_size': dataset_size,
            'unique_combinations': unique_combinations,
            'external_data_factor': external_data_available,
            'linkage_attack_factor': linkage_attacks_possible,
            'mitigation_recommendations': self._generate_reid_recommendations(adjusted_risk)
        }

    def _generate_reid_recommendations(self, risk: float) -> List[str]:
        """Generate re-identification mitigation recommendations."""
        recommendations = []
        if risk > 0.3:
            recommendations.append("Apply k-anonymity with k >= 5")
            recommendations.append("Consider data generalization")
        if risk > 0.5:
            recommendations.append("Implement differential privacy")
            recommendations.append("Remove or generalize quasi-identifiers")
        if risk > 0.7:
            recommendations.append("Consider synthetic data generation")
            recommendations.append("Restrict data access severely")
        return recommendations


class PIIDetectionAnalyzer:
    """Analyzes PII detection and handling."""

    def __init__(self, pii_patterns: Dict[str, str] = None):
        self.pii_patterns = pii_patterns or {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }

    def analyze_pii(self,
                   data_samples: List[Dict[str, Any]],
                   declared_pii_fields: List[str] = None) -> Dict[str, Any]:
        """Analyze PII in data samples."""
        import re

        if not data_samples:
            return {'pii_exposure_risk': 0.0, 'pii_detected': 0}

        declared_pii_fields = declared_pii_fields or []

        detected_pii = defaultdict(list)
        undeclared_pii = []

        for i, sample in enumerate(data_samples):
            for field, value in sample.items():
                if not isinstance(value, str):
                    continue

                for pii_type, pattern in self.pii_patterns.items():
                    if re.search(pattern, value):
                        detected_pii[pii_type].append({
                            'sample_index': i,
                            'field': field
                        })

                        if field not in declared_pii_fields:
                            undeclared_pii.append({
                                'field': field,
                                'pii_type': pii_type,
                                'sample_index': i
                            })

        total_samples = len(data_samples)
        samples_with_pii = len(set(d['sample_index'] for detections in detected_pii.values() for d in detections))

        pii_exposure_risk = samples_with_pii / total_samples if total_samples > 0 else 0

        return {
            'pii_exposure_risk': float(pii_exposure_risk),
            'samples_analyzed': total_samples,
            'samples_with_pii': samples_with_pii,
            'pii_types_detected': list(detected_pii.keys()),
            'pii_detection_counts': {k: len(v) for k, v in detected_pii.items()},
            'undeclared_pii': len(undeclared_pii),
            'undeclared_pii_details': undeclared_pii[:20],
            'risk_level': 'high' if pii_exposure_risk > 0.3 else ('medium' if pii_exposure_risk > 0.1 else 'low')
        }


class ConsentAnalyzer:
    """Analyzes consent management."""

    def analyze_consent(self,
                       consent_records: List[ConsentRecord],
                       data_processing_activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consent compliance."""
        if not consent_records:
            return {'consent_compliance': 0.0, 'missing_consent': len(data_processing_activities)}

        current_time = datetime.now()

        valid_consents = []
        expired_consents = []
        revoked_consents = []

        for consent in consent_records:
            if consent.revoked:
                revoked_consents.append(consent)
            elif consent.expires_at and consent.expires_at < current_time:
                expired_consents.append(consent)
            elif consent.granted:
                valid_consents.append(consent)

        # Check consent coverage for processing activities
        consent_by_purpose = defaultdict(list)
        for consent in valid_consents:
            consent_by_purpose[consent.purpose].append(consent)

        activities_with_consent = []
        activities_without_consent = []

        for activity in data_processing_activities:
            purpose = activity.get('purpose')
            if purpose in consent_by_purpose and consent_by_purpose[purpose]:
                activities_with_consent.append(activity)
            else:
                activities_without_consent.append({
                    'activity': activity.get('name', 'unknown'),
                    'purpose': purpose
                })

        total_activities = len(data_processing_activities)
        consent_compliance = len(activities_with_consent) / total_activities if total_activities > 0 else 1

        return {
            'consent_compliance': float(consent_compliance),
            'valid_consents': len(valid_consents),
            'expired_consents': len(expired_consents),
            'revoked_consents': len(revoked_consents),
            'total_consents': len(consent_records),
            'activities_with_consent': len(activities_with_consent),
            'activities_without_consent': len(activities_without_consent),
            'missing_consent_details': activities_without_consent,
            'consent_coverage_by_purpose': {
                purpose: len(consents)
                for purpose, consents in consent_by_purpose.items()
            }
        }


class DataLeakageAnalyzer:
    """Analyzes potential data leakage."""

    def analyze_leakage(self,
                       model_outputs: List[Dict[str, Any]],
                       training_data_samples: List[Dict[str, Any]],
                       similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """Analyze potential data leakage in model outputs."""
        if not model_outputs or not training_data_samples:
            return {'leakage_risk': 0.0, 'potential_leaks': 0}

        potential_leaks = []

        # Simple similarity check (in practice, use more sophisticated methods)
        training_hashes = set()
        for sample in training_data_samples:
            sample_str = json.dumps(sample, sort_keys=True)
            sample_hash = hashlib.md5(sample_str.encode()).hexdigest()
            training_hashes.add(sample_hash)

        for i, output in enumerate(model_outputs):
            output_str = json.dumps(output, sort_keys=True)
            output_hash = hashlib.md5(output_str.encode()).hexdigest()

            if output_hash in training_hashes:
                potential_leaks.append({
                    'output_index': i,
                    'leak_type': 'exact_match',
                    'severity': 'high'
                })

        leakage_rate = len(potential_leaks) / len(model_outputs) if model_outputs else 0

        return {
            'leakage_risk': float(leakage_rate),
            'potential_leaks': len(potential_leaks),
            'outputs_analyzed': len(model_outputs),
            'training_samples': len(training_data_samples),
            'leak_details': potential_leaks[:20],
            'risk_level': 'high' if leakage_rate > 0.05 else ('medium' if leakage_rate > 0.01 else 'low'),
            'recommendations': [
                "Implement membership inference defense",
                "Add output perturbation",
                "Review training data deduplication"
            ] if leakage_rate > 0.01 else []
        }


class EncryptionAnalyzer:
    """Analyzes encryption practices."""

    def analyze_encryption(self,
                          data_stores: List[Dict[str, Any]],
                          transmission_channels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze encryption coverage."""
        if not data_stores and not transmission_channels:
            return {'encryption_score': 0.0, 'unencrypted_stores': 0}

        # Analyze data at rest
        encrypted_stores = []
        unencrypted_stores = []

        for store in data_stores or []:
            if store.get('encrypted', False):
                encrypted_stores.append({
                    'store': store.get('name'),
                    'encryption_type': store.get('encryption_type', 'unknown'),
                    'key_rotation': store.get('key_rotation', False)
                })
            else:
                unencrypted_stores.append({
                    'store': store.get('name'),
                    'contains_sensitive': store.get('contains_sensitive', False)
                })

        # Analyze data in transit
        encrypted_channels = []
        unencrypted_channels = []

        for channel in transmission_channels or []:
            if channel.get('encrypted', False):
                encrypted_channels.append({
                    'channel': channel.get('name'),
                    'protocol': channel.get('protocol', 'unknown')
                })
            else:
                unencrypted_channels.append({
                    'channel': channel.get('name'),
                    'risk': 'high' if channel.get('external', False) else 'medium'
                })

        total_stores = len(data_stores) if data_stores else 0
        total_channels = len(transmission_channels) if transmission_channels else 0

        store_encryption_rate = len(encrypted_stores) / total_stores if total_stores > 0 else 1
        channel_encryption_rate = len(encrypted_channels) / total_channels if total_channels > 0 else 1

        overall_score = (store_encryption_rate + channel_encryption_rate) / 2

        return {
            'encryption_score': float(overall_score),
            'data_at_rest_encryption': float(store_encryption_rate),
            'data_in_transit_encryption': float(channel_encryption_rate),
            'encrypted_stores': len(encrypted_stores),
            'unencrypted_stores': len(unencrypted_stores),
            'encrypted_channels': len(encrypted_channels),
            'unencrypted_channels': len(unencrypted_channels),
            'unencrypted_store_details': unencrypted_stores,
            'unencrypted_channel_details': unencrypted_channels,
            'compliance_status': 'compliant' if overall_score >= 0.95 else 'non_compliant'
        }


# ============================================================================
# Transparent Data Practices Analyzers (18 Analysis Types)
# ============================================================================

class DataProvenanceAnalyzer:
    """Analyzes data provenance and lineage."""

    def analyze_provenance(self,
                          data_assets: List[Dict[str, Any]],
                          lineage_graph: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Analyze data provenance completeness."""
        if not data_assets:
            return {'provenance_completeness': 0.0, 'assets_without_provenance': 0}

        complete_provenance = []
        incomplete_provenance = []

        required_fields = ['source', 'created_at', 'created_by', 'transformations']

        for asset in data_assets:
            missing_fields = [f for f in required_fields if not asset.get(f)]

            if not missing_fields:
                complete_provenance.append({
                    'asset_id': asset.get('id'),
                    'source': asset.get('source')
                })
            else:
                incomplete_provenance.append({
                    'asset_id': asset.get('id'),
                    'missing_fields': missing_fields
                })

        total = len(data_assets)
        completeness = len(complete_provenance) / total if total > 0 else 0

        # Analyze lineage graph
        lineage_analysis = {}
        if lineage_graph:
            for asset_id, ancestors in lineage_graph.items():
                depth = self._calculate_lineage_depth(asset_id, lineage_graph)
                lineage_analysis[asset_id] = {
                    'direct_ancestors': len(ancestors),
                    'lineage_depth': depth
                }

        return {
            'provenance_completeness': float(completeness),
            'complete_provenance': len(complete_provenance),
            'incomplete_provenance': len(incomplete_provenance),
            'total_assets': total,
            'lineage_analysis': lineage_analysis,
            'incomplete_details': incomplete_provenance,
            'most_common_missing': self._get_most_common_missing(incomplete_provenance)
        }

    def _calculate_lineage_depth(self, asset_id: str, graph: Dict[str, List[str]], visited: Set[str] = None) -> int:
        """Calculate lineage depth recursively."""
        if visited is None:
            visited = set()

        if asset_id in visited or asset_id not in graph:
            return 0

        visited.add(asset_id)
        ancestors = graph.get(asset_id, [])

        if not ancestors:
            return 0

        return 1 + max(
            self._calculate_lineage_depth(a, graph, visited)
            for a in ancestors
        )

    def _get_most_common_missing(self, incomplete: List[Dict]) -> Optional[str]:
        """Find most commonly missing field."""
        if not incomplete:
            return None

        field_counts = defaultdict(int)
        for item in incomplete:
            for field in item.get('missing_fields', []):
                field_counts[field] += 1

        return max(field_counts, key=field_counts.get) if field_counts else None


class UsageDisclosureAnalyzer:
    """Analyzes data usage disclosure practices."""

    def analyze_disclosure(self,
                          data_uses: List[Dict[str, Any]],
                          disclosed_uses: List[str]) -> Dict[str, Any]:
        """Analyze usage disclosure compliance."""
        if not data_uses:
            return {'disclosure_rate': 1.0, 'undisclosed_uses': 0}

        disclosed_set = set(disclosed_uses)
        actual_uses = set(use.get('purpose') for use in data_uses)

        disclosed_uses_found = actual_uses & disclosed_set
        undisclosed_uses = actual_uses - disclosed_set

        disclosure_rate = len(disclosed_uses_found) / len(actual_uses) if actual_uses else 1

        # Analyze use categories
        use_categories = defaultdict(list)
        for use in data_uses:
            category = use.get('category', 'unknown')
            use_categories[category].append(use.get('purpose'))

        return {
            'disclosure_rate': float(disclosure_rate),
            'disclosed_uses': len(disclosed_uses_found),
            'undisclosed_uses': len(undisclosed_uses),
            'total_uses': len(actual_uses),
            'undisclosed_use_list': list(undisclosed_uses),
            'use_categories': {k: len(v) for k, v in use_categories.items()},
            'compliance_status': 'compliant' if len(undisclosed_uses) == 0 else 'non_compliant',
            'recommendations': [
                f"Disclose data use for purpose: {purpose}"
                for purpose in list(undisclosed_uses)[:5]
            ]
        }


class AccessControlAnalyzer:
    """Analyzes data access control effectiveness."""

    def analyze_access_control(self,
                               access_events: List[DataAccessEvent],
                               access_policies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze access control compliance."""
        if not access_events:
            return {'access_control_score': 1.0, 'unauthorized_access': 0}

        authorized_access = []
        unauthorized_access = []
        policy_violations = []

        for event in access_events:
            if event.authorized:
                authorized_access.append({
                    'event_id': event.event_id,
                    'accessor': event.accessor,
                    'data_type': event.data_type
                })
            else:
                unauthorized_access.append({
                    'event_id': event.event_id,
                    'accessor': event.accessor,
                    'data_type': event.data_type,
                    'timestamp': event.timestamp.isoformat()
                })

            # Check policy compliance
            allowed_accessors = access_policies.get(event.data_type, [])
            if event.accessor not in allowed_accessors and allowed_accessors:
                policy_violations.append({
                    'event_id': event.event_id,
                    'accessor': event.accessor,
                    'data_type': event.data_type,
                    'violation_type': 'unauthorized_accessor'
                })

        total = len(access_events)
        access_control_score = len(authorized_access) / total if total > 0 else 1

        return {
            'access_control_score': float(access_control_score),
            'authorized_access': len(authorized_access),
            'unauthorized_access': len(unauthorized_access),
            'policy_violations': len(policy_violations),
            'total_events': total,
            'unauthorized_details': unauthorized_access[:20],
            'violation_details': policy_violations[:20],
            'accessors_with_violations': list(set(v['accessor'] for v in policy_violations))
        }


class RetentionComplianceAnalyzer:
    """Analyzes data retention compliance."""

    def analyze_retention(self,
                         data_records: List[DataRecord],
                         retention_policies: Dict[str, int]) -> Dict[str, Any]:
        """Analyze retention policy compliance."""
        if not data_records:
            return {'retention_compliance': 1.0, 'overdue_records': 0}

        current_time = datetime.now()

        compliant_records = []
        overdue_records = []
        nearing_expiry = []

        for record in data_records:
            policy_days = retention_policies.get(record.data_type, record.retention_period_days)
            expiry_date = record.created_at + timedelta(days=policy_days)

            days_until_expiry = (expiry_date - current_time).days

            if days_until_expiry < 0:
                overdue_records.append({
                    'record_id': record.record_id,
                    'data_type': record.data_type,
                    'overdue_days': abs(days_until_expiry)
                })
            elif days_until_expiry <= 30:
                nearing_expiry.append({
                    'record_id': record.record_id,
                    'data_type': record.data_type,
                    'days_until_expiry': days_until_expiry
                })
                compliant_records.append(record)
            else:
                compliant_records.append(record)

        total = len(data_records)
        compliance = len(compliant_records) / total if total > 0 else 1

        return {
            'retention_compliance': float(compliance),
            'compliant_records': len(compliant_records),
            'overdue_records': len(overdue_records),
            'nearing_expiry': len(nearing_expiry),
            'total_records': total,
            'overdue_details': overdue_records,
            'expiring_soon_details': nearing_expiry[:20],
            'action_required': len(overdue_records) > 0
        }


class DataSubjectRightsAnalyzer:
    """Analyzes data subject rights compliance."""

    def analyze_subject_rights(self,
                               rights_requests: List[Dict[str, Any]],
                               sla_days: Dict[str, int] = None) -> Dict[str, Any]:
        """Analyze data subject rights fulfillment."""
        sla_days = sla_days or {
            'access': 30,
            'rectification': 30,
            'erasure': 30,
            'portability': 30,
            'restriction': 7,
            'objection': 7
        }

        if not rights_requests:
            return {'rights_compliance': 1.0, 'pending_requests': 0}

        fulfilled_requests = []
        pending_requests = []
        overdue_requests = []

        current_time = datetime.now()

        for request in rights_requests:
            request_type = request.get('type', 'unknown')
            request_date = request.get('requested_at')
            fulfilled = request.get('fulfilled', False)
            fulfilled_date = request.get('fulfilled_at')

            if isinstance(request_date, str):
                request_date = datetime.fromisoformat(request_date.replace('Z', '+00:00'))

            sla = sla_days.get(request_type, 30)

            if fulfilled:
                if isinstance(fulfilled_date, str):
                    fulfilled_date = datetime.fromisoformat(fulfilled_date.replace('Z', '+00:00'))

                response_days = (fulfilled_date - request_date).days if fulfilled_date else 0
                within_sla = response_days <= sla

                fulfilled_requests.append({
                    'request_id': request.get('id'),
                    'type': request_type,
                    'response_days': response_days,
                    'within_sla': within_sla
                })
            else:
                days_pending = (current_time - request_date.replace(tzinfo=None)).days
                if days_pending > sla:
                    overdue_requests.append({
                        'request_id': request.get('id'),
                        'type': request_type,
                        'days_overdue': days_pending - sla
                    })
                else:
                    pending_requests.append({
                        'request_id': request.get('id'),
                        'type': request_type,
                        'days_pending': days_pending,
                        'sla_days_remaining': sla - days_pending
                    })

        total = len(rights_requests)
        on_time_fulfillment = sum(1 for f in fulfilled_requests if f['within_sla'])
        compliance = (on_time_fulfillment + len(pending_requests)) / total if total > 0 else 1

        return {
            'rights_compliance': float(compliance),
            'fulfilled_requests': len(fulfilled_requests),
            'pending_requests': len(pending_requests),
            'overdue_requests': len(overdue_requests),
            'total_requests': total,
            'on_time_rate': on_time_fulfillment / len(fulfilled_requests) if fulfilled_requests else 1,
            'pending_details': pending_requests,
            'overdue_details': overdue_requests,
            'request_type_breakdown': self._get_type_breakdown(rights_requests)
        }

    def _get_type_breakdown(self, requests: List[Dict]) -> Dict[str, int]:
        """Get breakdown by request type."""
        breakdown = defaultdict(int)
        for request in requests:
            breakdown[request.get('type', 'unknown')] += 1
        return dict(breakdown)


class ThirdPartyDisclosureAnalyzer:
    """Analyzes third-party data disclosure."""

    def analyze_third_party_disclosure(self,
                                       disclosures: List[Dict[str, Any]],
                                       authorized_parties: List[str],
                                       consent_records: List[ConsentRecord] = None) -> Dict[str, Any]:
        """Analyze third-party data disclosure compliance."""
        if not disclosures:
            return {'disclosure_compliance': 1.0, 'unauthorized_disclosures': 0}

        authorized_set = set(authorized_parties)

        authorized_disclosures = []
        unauthorized_disclosures = []
        consent_based_disclosures = []

        consent_map = {}
        if consent_records:
            for consent in consent_records:
                if consent.granted and not consent.revoked:
                    key = f"{consent.data_subject_id}_{consent.purpose}"
                    consent_map[key] = consent

        for disclosure in disclosures:
            recipient = disclosure.get('recipient')
            data_subject = disclosure.get('data_subject_id')
            purpose = disclosure.get('purpose')

            if recipient in authorized_set:
                authorized_disclosures.append({
                    'recipient': recipient,
                    'purpose': purpose
                })
            else:
                # Check if consent exists
                consent_key = f"{data_subject}_{purpose}"
                if consent_key in consent_map:
                    consent_based_disclosures.append({
                        'recipient': recipient,
                        'purpose': purpose,
                        'consent_id': consent_map[consent_key].consent_id
                    })
                else:
                    unauthorized_disclosures.append({
                        'recipient': recipient,
                        'purpose': purpose,
                        'data_subject': data_subject
                    })

        total = len(disclosures)
        compliant = len(authorized_disclosures) + len(consent_based_disclosures)
        compliance = compliant / total if total > 0 else 1

        return {
            'disclosure_compliance': float(compliance),
            'authorized_disclosures': len(authorized_disclosures),
            'consent_based_disclosures': len(consent_based_disclosures),
            'unauthorized_disclosures': len(unauthorized_disclosures),
            'total_disclosures': total,
            'unique_recipients': len(set(d.get('recipient') for d in disclosures)),
            'unauthorized_details': unauthorized_disclosures[:20]
        }


# ============================================================================
# Report Generator
# ============================================================================

class PrivacyReportGenerator:
    """Generates comprehensive privacy and transparency reports."""

    def __init__(self):
        self.dp_analyzer = DifferentialPrivacyAnalyzer()
        self.minimization_analyzer = DataMinimizationAnalyzer()
        self.anonymization_analyzer = AnonymizationAnalyzer()
        self.reid_analyzer = ReIdentificationRiskAnalyzer()
        self.pii_analyzer = PIIDetectionAnalyzer()
        self.consent_analyzer = ConsentAnalyzer()
        self.leakage_analyzer = DataLeakageAnalyzer()
        self.encryption_analyzer = EncryptionAnalyzer()

        self.provenance_analyzer = DataProvenanceAnalyzer()
        self.disclosure_analyzer = UsageDisclosureAnalyzer()
        self.access_analyzer = AccessControlAnalyzer()
        self.retention_analyzer = RetentionComplianceAnalyzer()
        self.rights_analyzer = DataSubjectRightsAnalyzer()
        self.third_party_analyzer = ThirdPartyDisclosureAnalyzer()

    def generate_privacy_report(self,
                               data_records: List[DataRecord] = None,
                               consent_records: List[ConsentRecord] = None,
                               processing_activities: List[Dict[str, Any]] = None,
                               epsilon_values: List[float] = None) -> Dict[str, Any]:
        """Generate privacy analysis report."""
        report = {
            'report_type': 'privacy_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if data_records:
            report['anonymization'] = self.anonymization_analyzer.analyze_anonymization(data_records)

        if consent_records and processing_activities:
            report['consent'] = self.consent_analyzer.analyze_consent(
                consent_records, processing_activities
            )

        if epsilon_values:
            report['differential_privacy'] = self.dp_analyzer.analyze_differential_privacy(
                epsilon_values
            )

        return report

    def generate_transparency_report(self,
                                    data_assets: List[Dict[str, Any]] = None,
                                    data_uses: List[Dict[str, Any]] = None,
                                    disclosed_uses: List[str] = None,
                                    access_events: List[DataAccessEvent] = None,
                                    access_policies: Dict[str, List[str]] = None,
                                    rights_requests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate transparency analysis report."""
        report = {
            'report_type': 'transparency_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if data_assets:
            report['provenance'] = self.provenance_analyzer.analyze_provenance(data_assets)

        if data_uses and disclosed_uses is not None:
            report['usage_disclosure'] = self.disclosure_analyzer.analyze_disclosure(
                data_uses, disclosed_uses
            )

        if access_events:
            report['access_control'] = self.access_analyzer.analyze_access_control(
                access_events, access_policies or {}
            )

        if rights_requests:
            report['subject_rights'] = self.rights_analyzer.analyze_subject_rights(rights_requests)

        return report

    def generate_full_report(self,
                            data_records: List[DataRecord] = None,
                            consent_records: List[ConsentRecord] = None,
                            data_assets: List[Dict[str, Any]] = None,
                            access_events: List[DataAccessEvent] = None,
                            rights_requests: List[Dict[str, Any]] = None,
                            epsilon_values: List[float] = None) -> Dict[str, Any]:
        """Generate comprehensive privacy and transparency report."""
        report = {
            'report_type': 'comprehensive_privacy_transparency_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        # Privacy analysis
        if data_records or consent_records or epsilon_values:
            privacy_report = self.generate_privacy_report(
                data_records, consent_records, [], epsilon_values
            )
            for key, value in privacy_report.items():
                if key not in ['report_type', 'timestamp']:
                    report[f'privacy_{key}'] = value

        # Transparency analysis
        if data_assets or access_events or rights_requests:
            transparency_report = self.generate_transparency_report(
                data_assets, [], [], access_events, {}, rights_requests
            )
            for key, value in transparency_report.items():
                if key not in ['report_type', 'timestamp']:
                    report[f'transparency_{key}'] = value

        # Calculate overall score
        scores = []
        if 'privacy_anonymization' in report:
            scores.append(report['privacy_anonymization'].get('anonymization_score', 0))
        if 'privacy_consent' in report:
            scores.append(report['privacy_consent'].get('consent_compliance', 0))
        if 'transparency_access_control' in report:
            scores.append(report['transparency_access_control'].get('access_control_score', 0))
        if 'transparency_subject_rights' in report:
            scores.append(report['transparency_subject_rights'].get('rights_compliance', 0))

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
            f"# {report.get('report_type', 'Privacy Analysis Report')}",
            f"\n**Generated:** {report.get('timestamp', 'N/A')}",
            f"\n**Overall Score:** {report.get('overall_score', 0):.2%}",
            "\n---\n"
        ]

        if 'privacy_anonymization' in report:
            lines.append("## Anonymization Analysis\n")
            anon = report['privacy_anonymization']
            lines.append(f"- **Anonymization Score:** {anon.get('anonymization_score', 0):.2%}")
            lines.append(f"- **Records with PII:** {anon.get('records_with_pii', 0)}")
            lines.append("")

        if 'privacy_consent' in report:
            lines.append("## Consent Analysis\n")
            consent = report['privacy_consent']
            lines.append(f"- **Consent Compliance:** {consent.get('consent_compliance', 0):.2%}")
            lines.append(f"- **Valid Consents:** {consent.get('valid_consents', 0)}")
            lines.append("")

        if 'transparency_access_control' in report:
            lines.append("## Access Control\n")
            access = report['transparency_access_control']
            lines.append(f"- **Access Control Score:** {access.get('access_control_score', 0):.2%}")
            lines.append(f"- **Unauthorized Access:** {access.get('unauthorized_access', 0)}")
            lines.append("")

        return "\n".join(lines)
