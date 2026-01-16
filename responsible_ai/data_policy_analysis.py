"""
Data Policy Analysis Module
===========================

Comprehensive analysis framework for data governance policies including:
1. Data Masking - PII redaction, tokenization, pseudonymization
2. Data Reduction - Minimization, aggregation, sampling strategies
3. Data Retention - Lifecycle management, archival, deletion policies
4. Data Classification - Sensitivity levels, access controls
5. Data Quality - Completeness, accuracy, consistency validation

This module provides analyzers for assessing and validating data policy
compliance across AI/ML systems.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Pattern
from enum import Enum
from abc import ABC, abstractmethod
import re
import math
from collections import defaultdict
import hashlib


# =============================================================================
# Enums
# =============================================================================

class MaskingTechnique(Enum):
    """Data masking techniques."""
    REDACTION = "redaction"
    TOKENIZATION = "tokenization"
    PSEUDONYMIZATION = "pseudonymization"
    GENERALIZATION = "generalization"
    SUPPRESSION = "suppression"
    ENCRYPTION = "encryption"
    HASHING = "hashing"
    SHUFFLING = "shuffling"
    NULLING = "nulling"
    SUBSTITUTION = "substitution"


class DataSensitivityLevel(Enum):
    """Data sensitivity classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    HIGHLY_RESTRICTED = "highly_restricted"


class RetentionCategory(Enum):
    """Data retention categories."""
    TRANSIENT = "transient"  # < 24 hours
    SHORT_TERM = "short_term"  # < 30 days
    MEDIUM_TERM = "medium_term"  # < 1 year
    LONG_TERM = "long_term"  # 1-7 years
    PERMANENT = "permanent"  # > 7 years or indefinite


class DataReductionMethod(Enum):
    """Data reduction methods."""
    SAMPLING = "sampling"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    PROJECTION = "projection"
    BINNING = "binning"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    DEDUPLICATION = "deduplication"


class ComplianceStatus(Enum):
    """Policy compliance status."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    ADDRESS = "address"
    DOB = "date_of_birth"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    BIOMETRIC = "biometric"
    CREDENTIAL = "credential"
    IP_ADDRESS = "ip_address"
    DEVICE_ID = "device_id"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MaskingResult:
    """Result of data masking analysis."""
    original_length: int
    masked_length: int
    pii_detected: Dict[PIIType, int]
    pii_masked: Dict[PIIType, int]
    masking_coverage: float
    techniques_applied: List[MaskingTechnique]
    data_utility_preserved: float
    reversibility_risk: float
    compliance_status: ComplianceStatus


@dataclass
class DataReductionResult:
    """Result of data reduction analysis."""
    original_records: int
    reduced_records: int
    reduction_ratio: float
    methods_applied: List[DataReductionMethod]
    information_loss_estimate: float
    statistical_properties_preserved: Dict[str, float]
    minimization_score: float
    necessity_validation: bool


@dataclass
class RetentionAnalysisResult:
    """Result of retention policy analysis."""
    data_category: str
    sensitivity_level: DataSensitivityLevel
    retention_category: RetentionCategory
    retention_days: int
    legal_requirements: List[str]
    deletion_schedule: Optional[str]
    archival_status: str
    compliance_status: ComplianceStatus
    risk_assessment: Dict[str, Any]


@dataclass
class DataClassificationResult:
    """Result of data classification analysis."""
    classification_level: DataSensitivityLevel
    confidence: float
    indicators: List[str]
    recommended_controls: List[str]
    access_requirements: Dict[str, Any]
    handling_instructions: List[str]


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    completeness: float
    accuracy: float
    consistency: float
    timeliness: float
    uniqueness: float
    validity: float
    overall_quality: float
    issues_detected: List[str]


@dataclass
class DataPolicyAssessment:
    """Complete data policy assessment."""
    masking_score: float
    reduction_score: float
    retention_score: float
    classification_score: float
    quality_score: float
    overall_compliance: ComplianceStatus
    findings: List[str]
    recommendations: List[str]
    risk_level: str


# =============================================================================
# Data Masking Analyzer
# =============================================================================

class DataMaskingAnalyzer:
    """Analyzer for data masking policies and effectiveness."""

    def __init__(self):
        self.pii_patterns = {
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            PIIType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            PIIType.IP_ADDRESS: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            PIIType.DOB: r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
            PIIType.CREDENTIAL: r'\b(?:password|pwd|secret|api[_-]?key|token)\s*[:=]\s*\S+',
            PIIType.FINANCIAL: r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }

        self.name_indicators = [
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        ]

        self.masking_patterns = {
            MaskingTechnique.REDACTION: r'\[REDACTED\]|\*{3,}|X{3,}',
            MaskingTechnique.TOKENIZATION: r'\{TOKEN:[A-Z0-9]+\}',
            MaskingTechnique.HASHING: r'[a-f0-9]{32,64}',
            MaskingTechnique.NULLING: r'\bnull\b|\bNULL\b|\bNone\b',
        }

    def analyze(
        self,
        text: str,
        masked_text: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> MaskingResult:
        """Analyze data masking effectiveness."""
        config = config or {}

        # Detect PII in original text
        pii_detected = self._detect_pii(text)

        # If masked text provided, compare
        if masked_text:
            pii_masked = self._detect_masked_pii(text, masked_text, pii_detected)
            techniques_applied = self._detect_masking_techniques(masked_text)
            masked_length = len(masked_text)
        else:
            pii_masked = {pii_type: 0 for pii_type in pii_detected}
            techniques_applied = []
            masked_length = len(text)

        # Calculate coverage
        total_pii = sum(pii_detected.values())
        total_masked = sum(pii_masked.values())
        masking_coverage = total_masked / max(1, total_pii)

        # Estimate data utility preserved
        data_utility_preserved = self._estimate_utility_preservation(
            text, masked_text or text, config
        )

        # Assess reversibility risk
        reversibility_risk = self._assess_reversibility_risk(techniques_applied)

        # Determine compliance
        compliance_status = self._determine_compliance(
            masking_coverage, pii_detected, config
        )

        return MaskingResult(
            original_length=len(text),
            masked_length=masked_length,
            pii_detected=pii_detected,
            pii_masked=pii_masked,
            masking_coverage=masking_coverage,
            techniques_applied=techniques_applied,
            data_utility_preserved=data_utility_preserved,
            reversibility_risk=reversibility_risk,
            compliance_status=compliance_status
        )

    def _detect_pii(self, text: str) -> Dict[PIIType, int]:
        """Detect PII in text."""
        pii_counts = {}

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            pii_counts[pii_type] = len(matches)

        # Detect names
        name_count = 0
        for pattern in self.name_indicators:
            matches = re.findall(pattern, text)
            name_count += len(matches)
        pii_counts[PIIType.NAME] = name_count

        return pii_counts

    def _detect_masked_pii(
        self,
        original: str,
        masked: str,
        pii_detected: Dict[PIIType, int]
    ) -> Dict[PIIType, int]:
        """Detect how many PII items were masked."""
        pii_in_masked = self._detect_pii(masked)

        masked_counts = {}
        for pii_type in pii_detected:
            original_count = pii_detected.get(pii_type, 0)
            remaining_count = pii_in_masked.get(pii_type, 0)
            masked_counts[pii_type] = max(0, original_count - remaining_count)

        return masked_counts

    def _detect_masking_techniques(self, text: str) -> List[MaskingTechnique]:
        """Detect which masking techniques were applied."""
        techniques = []

        for technique, pattern in self.masking_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                techniques.append(technique)

        return techniques

    def _estimate_utility_preservation(
        self,
        original: str,
        masked: str,
        config: Dict
    ) -> float:
        """Estimate how much data utility is preserved after masking."""
        if original == masked:
            return 1.0

        # Length ratio
        length_ratio = len(masked) / max(1, len(original))

        # Word overlap
        original_words = set(original.lower().split())
        masked_words = set(masked.lower().split())
        word_overlap = len(original_words & masked_words) / max(1, len(original_words))

        # Structure preservation
        original_lines = original.count('\n')
        masked_lines = masked.count('\n')
        structure_preserved = 1.0 if original_lines == masked_lines else 0.8

        # Weighted average
        utility = (length_ratio * 0.2 + word_overlap * 0.5 + structure_preserved * 0.3)

        return min(1.0, max(0.0, utility))

    def _assess_reversibility_risk(self, techniques: List[MaskingTechnique]) -> float:
        """Assess risk of masked data being reversed."""
        reversibility_scores = {
            MaskingTechnique.REDACTION: 0.1,  # Very low risk
            MaskingTechnique.SUPPRESSION: 0.1,
            MaskingTechnique.NULLING: 0.1,
            MaskingTechnique.HASHING: 0.3,  # Some risk with rainbow tables
            MaskingTechnique.TOKENIZATION: 0.5,  # Reversible with key
            MaskingTechnique.ENCRYPTION: 0.5,  # Reversible with key
            MaskingTechnique.PSEUDONYMIZATION: 0.6,  # Higher risk
            MaskingTechnique.GENERALIZATION: 0.4,
            MaskingTechnique.SHUFFLING: 0.7,  # Pattern analysis possible
            MaskingTechnique.SUBSTITUTION: 0.6,
        }

        if not techniques:
            return 1.0  # No masking = high risk

        risks = [reversibility_scores.get(t, 0.5) for t in techniques]
        return sum(risks) / len(risks)

    def _determine_compliance(
        self,
        coverage: float,
        pii_detected: Dict[PIIType, int],
        config: Dict
    ) -> ComplianceStatus:
        """Determine masking compliance status."""
        required_coverage = config.get('required_coverage', 0.95)

        # High-risk PII types require complete masking
        high_risk_pii = {PIIType.SSN, PIIType.FINANCIAL, PIIType.MEDICAL, PIIType.CREDENTIAL}
        has_unmasked_high_risk = any(
            pii_detected.get(pii_type, 0) > 0
            for pii_type in high_risk_pii
        )

        if coverage >= required_coverage and not has_unmasked_high_risk:
            return ComplianceStatus.COMPLIANT
        elif coverage >= 0.8:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        return ComplianceStatus.NON_COMPLIANT

    def mask_text(self, text: str, techniques: List[MaskingTechnique] = None) -> str:
        """Apply masking to text."""
        techniques = techniques or [MaskingTechnique.REDACTION]
        masked = text

        for pii_type, pattern in self.pii_patterns.items():
            if MaskingTechnique.REDACTION in techniques:
                masked = re.sub(pattern, '[REDACTED]', masked, flags=re.IGNORECASE)
            elif MaskingTechnique.HASHING in techniques:
                def hash_match(match):
                    return hashlib.sha256(match.group().encode()).hexdigest()[:16]
                masked = re.sub(pattern, hash_match, masked, flags=re.IGNORECASE)

        return masked


# =============================================================================
# Data Reduction Analyzer
# =============================================================================

class DataReductionAnalyzer:
    """Analyzer for data reduction and minimization policies."""

    def __init__(self):
        self.statistical_properties = [
            'mean', 'median', 'std', 'min', 'max', 'distribution'
        ]

    def analyze(
        self,
        original_data: Dict[str, Any],
        reduced_data: Optional[Dict[str, Any]] = None,
        config: Optional[Dict] = None
    ) -> DataReductionResult:
        """Analyze data reduction effectiveness."""
        config = config or {}

        original_records = original_data.get('record_count', 0)
        original_fields = original_data.get('field_count', 0)

        if reduced_data:
            reduced_records = reduced_data.get('record_count', 0)
            reduced_fields = reduced_data.get('field_count', 0)
        else:
            reduced_records = original_records
            reduced_fields = original_fields

        # Calculate reduction ratio
        record_ratio = 1 - (reduced_records / max(1, original_records))
        field_ratio = 1 - (reduced_fields / max(1, original_fields))
        reduction_ratio = (record_ratio + field_ratio) / 2

        # Detect methods applied
        methods_applied = self._detect_reduction_methods(original_data, reduced_data)

        # Estimate information loss
        information_loss = self._estimate_information_loss(
            original_data, reduced_data, methods_applied
        )

        # Check statistical properties preservation
        stats_preserved = self._check_statistical_preservation(
            original_data.get('statistics', {}),
            reduced_data.get('statistics', {}) if reduced_data else {}
        )

        # Calculate minimization score
        minimization_score = self._calculate_minimization_score(
            original_data, reduced_data, config
        )

        # Validate necessity
        necessity_validation = self._validate_necessity(
            original_data.get('fields', []),
            config.get('required_fields', [])
        )

        return DataReductionResult(
            original_records=original_records,
            reduced_records=reduced_records,
            reduction_ratio=reduction_ratio,
            methods_applied=methods_applied,
            information_loss_estimate=information_loss,
            statistical_properties_preserved=stats_preserved,
            minimization_score=minimization_score,
            necessity_validation=necessity_validation
        )

    def _detect_reduction_methods(
        self,
        original: Dict[str, Any],
        reduced: Optional[Dict[str, Any]]
    ) -> List[DataReductionMethod]:
        """Detect which reduction methods were applied."""
        methods = []

        if not reduced:
            return methods

        original_records = original.get('record_count', 0)
        reduced_records = reduced.get('record_count', 0)

        if reduced_records < original_records:
            # Could be sampling or filtering
            if reduced.get('sampling_applied', False):
                methods.append(DataReductionMethod.SAMPLING)
            else:
                methods.append(DataReductionMethod.FILTERING)

        original_fields = original.get('field_count', 0)
        reduced_fields = reduced.get('field_count', 0)

        if reduced_fields < original_fields:
            methods.append(DataReductionMethod.PROJECTION)

        if reduced.get('aggregated', False):
            methods.append(DataReductionMethod.AGGREGATION)

        if reduced.get('deduplicated', False):
            methods.append(DataReductionMethod.DEDUPLICATION)

        if reduced.get('binned', False):
            methods.append(DataReductionMethod.BINNING)

        return methods

    def _estimate_information_loss(
        self,
        original: Dict[str, Any],
        reduced: Optional[Dict[str, Any]],
        methods: List[DataReductionMethod]
    ) -> float:
        """Estimate information loss from reduction."""
        if not reduced:
            return 0.0

        # Base loss from record reduction
        record_loss = 1 - (
            reduced.get('record_count', 0) / max(1, original.get('record_count', 1))
        )

        # Base loss from field reduction
        field_loss = 1 - (
            reduced.get('field_count', 0) / max(1, original.get('field_count', 1))
        )

        # Method-specific adjustments
        method_factors = {
            DataReductionMethod.SAMPLING: 0.3,  # Sampling preserves distribution
            DataReductionMethod.AGGREGATION: 0.5,  # Loses individual details
            DataReductionMethod.FILTERING: 0.4,  # Targeted reduction
            DataReductionMethod.PROJECTION: 0.3,  # Keeps essential fields
            DataReductionMethod.DEDUPLICATION: 0.1,  # Minimal loss
            DataReductionMethod.BINNING: 0.4,  # Loses precision
        }

        method_factor = sum(
            method_factors.get(m, 0.5) for m in methods
        ) / max(1, len(methods))

        # Weighted information loss
        total_loss = (record_loss * 0.5 + field_loss * 0.3) * method_factor

        return min(1.0, max(0.0, total_loss))

    def _check_statistical_preservation(
        self,
        original_stats: Dict[str, Any],
        reduced_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """Check how well statistical properties are preserved."""
        preservation = {}

        for prop in self.statistical_properties:
            original_val = original_stats.get(prop)
            reduced_val = reduced_stats.get(prop)

            if original_val is not None and reduced_val is not None:
                if isinstance(original_val, (int, float)):
                    if original_val != 0:
                        diff_ratio = abs(original_val - reduced_val) / abs(original_val)
                        preservation[prop] = max(0, 1 - diff_ratio)
                    else:
                        preservation[prop] = 1.0 if reduced_val == 0 else 0.0
                else:
                    preservation[prop] = 1.0 if original_val == reduced_val else 0.5
            else:
                preservation[prop] = 0.0 if original_val is not None else 1.0

        return preservation

    def _calculate_minimization_score(
        self,
        original: Dict[str, Any],
        reduced: Optional[Dict[str, Any]],
        config: Dict
    ) -> float:
        """Calculate data minimization compliance score."""
        if not reduced:
            return 0.0

        required_fields = set(config.get('required_fields', []))
        actual_fields = set(reduced.get('fields', []))
        extra_fields = actual_fields - required_fields

        # Penalize for collecting more than necessary
        if required_fields:
            minimization = len(required_fields) / max(1, len(actual_fields))
        else:
            minimization = 1.0 if not extra_fields else 0.5

        # Reward for applying reduction
        if reduced.get('record_count', 0) < original.get('record_count', 0):
            minimization += 0.1

        return min(1.0, minimization)

    def _validate_necessity(
        self,
        actual_fields: List[str],
        required_fields: List[str]
    ) -> bool:
        """Validate that only necessary data is collected."""
        if not required_fields:
            return True

        actual_set = set(actual_fields)
        required_set = set(required_fields)

        # Check if actual fields are subset of required (or equal)
        extra_fields = actual_set - required_set

        return len(extra_fields) == 0


# =============================================================================
# Data Retention Analyzer
# =============================================================================

class DataRetentionAnalyzer:
    """Analyzer for data retention policies."""

    def __init__(self):
        self.legal_requirements = {
            'healthcare': {
                'hipaa': {'min_years': 6, 'description': 'HIPAA medical records'},
                'state_law': {'min_years': 7, 'description': 'State medical records law'},
            },
            'financial': {
                'sox': {'min_years': 7, 'description': 'Sarbanes-Oxley audit records'},
                'sec': {'min_years': 6, 'description': 'SEC trading records'},
                'tax': {'min_years': 7, 'description': 'Tax records'},
            },
            'employment': {
                'eeoc': {'min_years': 1, 'description': 'EEOC employee records'},
                'osha': {'min_years': 5, 'description': 'OSHA safety records'},
            },
            'gdpr': {
                'consent': {'max_years': 3, 'description': 'GDPR consent records'},
                'breach': {'min_years': 5, 'description': 'Data breach records'},
            },
        }

    def analyze(
        self,
        data_category: str,
        current_policy: Dict[str, Any],
        data_metadata: Optional[Dict] = None
    ) -> RetentionAnalysisResult:
        """Analyze retention policy compliance."""
        data_metadata = data_metadata or {}

        # Determine sensitivity level
        sensitivity_level = self._determine_sensitivity(data_category, data_metadata)

        # Get retention requirements
        requirements = self._get_retention_requirements(
            data_category, sensitivity_level
        )

        # Determine retention category
        retention_days = current_policy.get('retention_days', 365)
        retention_category = self._categorize_retention(retention_days)

        # Check legal requirements
        legal_reqs = self._check_legal_requirements(
            data_category, retention_days
        )

        # Build deletion schedule
        deletion_schedule = current_policy.get('deletion_schedule', 'not_defined')

        # Archival status
        archival_status = self._assess_archival_status(current_policy)

        # Compliance check
        compliance_status = self._determine_compliance(
            retention_days, requirements, legal_reqs
        )

        # Risk assessment
        risk_assessment = self._assess_retention_risk(
            sensitivity_level, retention_days, compliance_status
        )

        return RetentionAnalysisResult(
            data_category=data_category,
            sensitivity_level=sensitivity_level,
            retention_category=retention_category,
            retention_days=retention_days,
            legal_requirements=legal_reqs,
            deletion_schedule=deletion_schedule,
            archival_status=archival_status,
            compliance_status=compliance_status,
            risk_assessment=risk_assessment
        )

    def _determine_sensitivity(
        self,
        data_category: str,
        metadata: Dict
    ) -> DataSensitivityLevel:
        """Determine data sensitivity level."""
        high_sensitivity_categories = {
            'healthcare', 'medical', 'financial', 'biometric', 'credential'
        }
        medium_sensitivity_categories = {
            'personal', 'employee', 'customer', 'contact'
        }

        if data_category.lower() in high_sensitivity_categories:
            return DataSensitivityLevel.HIGHLY_RESTRICTED

        if metadata.get('contains_pii', False):
            return DataSensitivityLevel.RESTRICTED

        if data_category.lower() in medium_sensitivity_categories:
            return DataSensitivityLevel.CONFIDENTIAL

        if metadata.get('internal_only', False):
            return DataSensitivityLevel.INTERNAL

        return DataSensitivityLevel.PUBLIC

    def _get_retention_requirements(
        self,
        data_category: str,
        sensitivity: DataSensitivityLevel
    ) -> Dict[str, int]:
        """Get retention requirements for data category."""
        requirements = {}

        # Base requirements by sensitivity
        sensitivity_minimums = {
            DataSensitivityLevel.HIGHLY_RESTRICTED: 365,  # 1 year minimum
            DataSensitivityLevel.RESTRICTED: 180,
            DataSensitivityLevel.CONFIDENTIAL: 90,
            DataSensitivityLevel.INTERNAL: 30,
            DataSensitivityLevel.PUBLIC: 0,
        }

        requirements['minimum_days'] = sensitivity_minimums.get(sensitivity, 0)

        # Maximum based on data minimization
        sensitivity_maximums = {
            DataSensitivityLevel.HIGHLY_RESTRICTED: 2555,  # 7 years
            DataSensitivityLevel.RESTRICTED: 1825,  # 5 years
            DataSensitivityLevel.CONFIDENTIAL: 1095,  # 3 years
            DataSensitivityLevel.INTERNAL: 365,
            DataSensitivityLevel.PUBLIC: 730,
        }

        requirements['maximum_days'] = sensitivity_maximums.get(sensitivity, 365)

        return requirements

    def _categorize_retention(self, days: int) -> RetentionCategory:
        """Categorize retention period."""
        if days < 1:
            return RetentionCategory.TRANSIENT
        elif days < 30:
            return RetentionCategory.SHORT_TERM
        elif days < 365:
            return RetentionCategory.MEDIUM_TERM
        elif days < 2555:
            return RetentionCategory.LONG_TERM
        return RetentionCategory.PERMANENT

    def _check_legal_requirements(
        self,
        data_category: str,
        retention_days: int
    ) -> List[str]:
        """Check applicable legal requirements."""
        applicable_reqs = []

        for domain, regulations in self.legal_requirements.items():
            if domain.lower() in data_category.lower():
                for reg_name, req in regulations.items():
                    if 'min_years' in req:
                        min_days = req['min_years'] * 365
                        if retention_days < min_days:
                            applicable_reqs.append(
                                f"{reg_name.upper()}: Minimum {req['min_years']} years required"
                            )
                        else:
                            applicable_reqs.append(
                                f"{reg_name.upper()}: Compliant ({req['description']})"
                            )
                    if 'max_years' in req:
                        max_days = req['max_years'] * 365
                        if retention_days > max_days:
                            applicable_reqs.append(
                                f"{reg_name.upper()}: Maximum {req['max_years']} years exceeded"
                            )

        return applicable_reqs

    def _assess_archival_status(self, policy: Dict) -> str:
        """Assess archival status."""
        if policy.get('archival_enabled', False):
            if policy.get('archival_encrypted', False):
                return 'active_encrypted'
            return 'active_unencrypted'
        if policy.get('archival_planned', False):
            return 'planned'
        return 'not_configured'

    def _determine_compliance(
        self,
        retention_days: int,
        requirements: Dict[str, int],
        legal_reqs: List[str]
    ) -> ComplianceStatus:
        """Determine retention compliance status."""
        min_days = requirements.get('minimum_days', 0)
        max_days = requirements.get('maximum_days', float('inf'))

        # Check for legal requirement violations
        violations = [r for r in legal_reqs if 'required' in r or 'exceeded' in r]

        if violations:
            return ComplianceStatus.NON_COMPLIANT

        if min_days <= retention_days <= max_days:
            return ComplianceStatus.COMPLIANT

        if retention_days < min_days:
            return ComplianceStatus.NON_COMPLIANT

        # Over maximum but no legal violation
        return ComplianceStatus.PARTIALLY_COMPLIANT

    def _assess_retention_risk(
        self,
        sensitivity: DataSensitivityLevel,
        retention_days: int,
        compliance: ComplianceStatus
    ) -> Dict[str, Any]:
        """Assess retention-related risks."""
        risk_factors = []
        risk_score = 0

        # Sensitivity-based risk
        sensitivity_risk = {
            DataSensitivityLevel.HIGHLY_RESTRICTED: 0.4,
            DataSensitivityLevel.RESTRICTED: 0.3,
            DataSensitivityLevel.CONFIDENTIAL: 0.2,
            DataSensitivityLevel.INTERNAL: 0.1,
            DataSensitivityLevel.PUBLIC: 0.0,
        }
        risk_score += sensitivity_risk.get(sensitivity, 0.1)

        # Duration-based risk
        if retention_days > 1825:  # 5 years
            risk_factors.append('Extended retention period increases breach risk')
            risk_score += 0.2
        elif retention_days > 365:
            risk_score += 0.1

        # Compliance-based risk
        if compliance == ComplianceStatus.NON_COMPLIANT:
            risk_factors.append('Non-compliance with retention requirements')
            risk_score += 0.3

        risk_level = 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.6 else 'high'

        return {
            'risk_level': risk_level,
            'risk_score': min(1.0, risk_score),
            'risk_factors': risk_factors
        }


# =============================================================================
# Data Classification Analyzer
# =============================================================================

class DataClassificationAnalyzer:
    """Analyzer for automated data classification."""

    def __init__(self):
        self.classification_indicators = {
            DataSensitivityLevel.HIGHLY_RESTRICTED: [
                'ssn', 'social security', 'medical record', 'health information',
                'biometric', 'genetic', 'password', 'credential', 'secret key'
            ],
            DataSensitivityLevel.RESTRICTED: [
                'credit card', 'bank account', 'financial', 'salary', 'income',
                'personal identifiable', 'pii', 'date of birth', 'driver license'
            ],
            DataSensitivityLevel.CONFIDENTIAL: [
                'internal only', 'proprietary', 'trade secret', 'employee',
                'customer data', 'contact information', 'address', 'phone'
            ],
            DataSensitivityLevel.INTERNAL: [
                'internal', 'company use', 'business', 'operational'
            ],
            DataSensitivityLevel.PUBLIC: [
                'public', 'published', 'press release', 'marketing'
            ],
        }

        self.control_requirements = {
            DataSensitivityLevel.HIGHLY_RESTRICTED: [
                'encryption_at_rest', 'encryption_in_transit', 'access_logging',
                'mfa_required', 'need_to_know_access', 'audit_trail'
            ],
            DataSensitivityLevel.RESTRICTED: [
                'encryption_at_rest', 'encryption_in_transit', 'access_logging',
                'role_based_access'
            ],
            DataSensitivityLevel.CONFIDENTIAL: [
                'encryption_in_transit', 'access_logging', 'role_based_access'
            ],
            DataSensitivityLevel.INTERNAL: [
                'authentication_required', 'basic_access_control'
            ],
            DataSensitivityLevel.PUBLIC: [],
        }

    def classify(
        self,
        data: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> DataClassificationResult:
        """Classify data sensitivity level."""
        context = context or {}

        # Analyze content for indicators
        classification_scores = self._calculate_classification_scores(data)

        # Determine classification
        classification, confidence = self._determine_classification(classification_scores)

        # Get indicators found
        indicators = self._get_matched_indicators(data, classification)

        # Get recommended controls
        controls = self.control_requirements.get(classification, [])

        # Build access requirements
        access_requirements = self._build_access_requirements(classification, context)

        # Generate handling instructions
        handling_instructions = self._generate_handling_instructions(classification)

        return DataClassificationResult(
            classification_level=classification,
            confidence=confidence,
            indicators=indicators,
            recommended_controls=controls,
            access_requirements=access_requirements,
            handling_instructions=handling_instructions
        )

    def _calculate_classification_scores(
        self,
        data: Dict[str, Any]
    ) -> Dict[DataSensitivityLevel, float]:
        """Calculate classification scores for each level."""
        scores = defaultdict(float)

        # Convert data to searchable text
        text = str(data).lower()

        for level, indicators in self.classification_indicators.items():
            for indicator in indicators:
                if indicator.lower() in text:
                    scores[level] += 1

        # Normalize scores
        total = sum(scores.values()) or 1
        return {level: score / total for level, score in scores.items()}

    def _determine_classification(
        self,
        scores: Dict[DataSensitivityLevel, float]
    ) -> Tuple[DataSensitivityLevel, float]:
        """Determine final classification from scores."""
        if not scores:
            return DataSensitivityLevel.INTERNAL, 0.5

        # Prioritize higher sensitivity levels
        priority_order = [
            DataSensitivityLevel.HIGHLY_RESTRICTED,
            DataSensitivityLevel.RESTRICTED,
            DataSensitivityLevel.CONFIDENTIAL,
            DataSensitivityLevel.INTERNAL,
            DataSensitivityLevel.PUBLIC,
        ]

        for level in priority_order:
            if scores.get(level, 0) > 0.2:  # Threshold
                confidence = min(1.0, scores[level] * 2)
                return level, confidence

        # Default to internal with low confidence
        return DataSensitivityLevel.INTERNAL, 0.3

    def _get_matched_indicators(
        self,
        data: Dict[str, Any],
        classification: DataSensitivityLevel
    ) -> List[str]:
        """Get indicators that matched for the classification."""
        text = str(data).lower()
        matched = []

        indicators = self.classification_indicators.get(classification, [])
        for indicator in indicators:
            if indicator.lower() in text:
                matched.append(indicator)

        return matched

    def _build_access_requirements(
        self,
        classification: DataSensitivityLevel,
        context: Dict
    ) -> Dict[str, Any]:
        """Build access requirements for classification."""
        requirements = {
            'authentication': 'required',
            'authorization': 'role_based',
            'logging': 'enabled',
        }

        if classification in [DataSensitivityLevel.HIGHLY_RESTRICTED, DataSensitivityLevel.RESTRICTED]:
            requirements['mfa'] = 'required'
            requirements['approval_required'] = True
            requirements['time_limited_access'] = True

        if classification == DataSensitivityLevel.HIGHLY_RESTRICTED:
            requirements['background_check'] = 'required'
            requirements['training_required'] = True

        return requirements

    def _generate_handling_instructions(
        self,
        classification: DataSensitivityLevel
    ) -> List[str]:
        """Generate handling instructions for data."""
        instructions = {
            DataSensitivityLevel.HIGHLY_RESTRICTED: [
                'Must be encrypted at all times',
                'Access requires formal approval and documentation',
                'Must not be shared outside authorized personnel',
                'Requires secure deletion when no longer needed',
                'Must be logged for all access and modifications',
            ],
            DataSensitivityLevel.RESTRICTED: [
                'Should be encrypted in transit and at rest',
                'Access limited to job function',
                'Sharing requires management approval',
                'Must follow data retention policies',
            ],
            DataSensitivityLevel.CONFIDENTIAL: [
                'Encrypt when transmitting externally',
                'Share only on need-to-know basis',
                'Dispose of securely when no longer needed',
            ],
            DataSensitivityLevel.INTERNAL: [
                'For internal use only',
                'Do not share externally without approval',
            ],
            DataSensitivityLevel.PUBLIC: [
                'May be shared freely',
                'Ensure accuracy before distribution',
            ],
        }

        return instructions.get(classification, [])


# =============================================================================
# Data Quality Analyzer
# =============================================================================

class DataQualityAnalyzer:
    """Analyzer for data quality assessment."""

    def __init__(self):
        self.quality_dimensions = [
            'completeness', 'accuracy', 'consistency',
            'timeliness', 'uniqueness', 'validity'
        ]

    def analyze(
        self,
        data: Dict[str, Any],
        schema: Optional[Dict] = None,
        config: Optional[Dict] = None
    ) -> DataQualityMetrics:
        """Analyze data quality."""
        config = config or {}
        schema = schema or {}

        completeness = self._assess_completeness(data, schema)
        accuracy = self._assess_accuracy(data, schema)
        consistency = self._assess_consistency(data)
        timeliness = self._assess_timeliness(data, config)
        uniqueness = self._assess_uniqueness(data)
        validity = self._assess_validity(data, schema)

        # Calculate overall quality
        weights = config.get('quality_weights', {
            'completeness': 0.2,
            'accuracy': 0.2,
            'consistency': 0.15,
            'timeliness': 0.15,
            'uniqueness': 0.15,
            'validity': 0.15,
        })

        overall = (
            completeness * weights.get('completeness', 0.2) +
            accuracy * weights.get('accuracy', 0.2) +
            consistency * weights.get('consistency', 0.15) +
            timeliness * weights.get('timeliness', 0.15) +
            uniqueness * weights.get('uniqueness', 0.15) +
            validity * weights.get('validity', 0.15)
        )

        # Collect issues
        issues = self._collect_issues(
            completeness, accuracy, consistency,
            timeliness, uniqueness, validity
        )

        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            uniqueness=uniqueness,
            validity=validity,
            overall_quality=overall,
            issues_detected=issues
        )

    def _assess_completeness(self, data: Dict, schema: Dict) -> float:
        """Assess data completeness."""
        if not schema:
            # Simple null check
            records = data.get('records', [])
            if not records:
                return 1.0

            null_count = 0
            total_fields = 0

            for record in records:
                for value in record.values():
                    total_fields += 1
                    if value is None or value == '':
                        null_count += 1

            return 1 - (null_count / max(1, total_fields))

        # Schema-based completeness
        required_fields = schema.get('required', [])
        if not required_fields:
            return 1.0

        records = data.get('records', [])
        complete_count = 0
        total_required_checks = 0

        for record in records:
            for field in required_fields:
                total_required_checks += 1
                if field in record and record[field] is not None:
                    complete_count += 1

        return complete_count / max(1, total_required_checks)

    def _assess_accuracy(self, data: Dict, schema: Dict) -> float:
        """Assess data accuracy."""
        # Without ground truth, estimate based on format conformance
        records = data.get('records', [])
        if not records:
            return 1.0

        format_rules = schema.get('formats', {})
        if not format_rules:
            return 0.9  # Assume reasonable accuracy without validation

        conforming = 0
        total_checks = 0

        for record in records:
            for field, pattern in format_rules.items():
                if field in record:
                    total_checks += 1
                    if re.match(pattern, str(record[field])):
                        conforming += 1

        return conforming / max(1, total_checks)

    def _assess_consistency(self, data: Dict) -> float:
        """Assess data consistency."""
        records = data.get('records', [])
        if not records:
            return 1.0

        # Check type consistency across records
        field_types = defaultdict(set)

        for record in records:
            for field, value in record.items():
                field_types[field].add(type(value).__name__)

        # Penalize fields with multiple types
        consistent_fields = sum(
            1 for types in field_types.values() if len(types) == 1
        )

        return consistent_fields / max(1, len(field_types))

    def _assess_timeliness(self, data: Dict, config: Dict) -> float:
        """Assess data timeliness."""
        last_updated = data.get('last_updated')
        if not last_updated:
            return 0.5  # Unknown timeliness

        freshness_threshold = config.get('freshness_threshold_days', 7)

        # Simplified: assume data is timely if last_updated is recent
        # In production, would compare to actual datetime
        return 0.9 if last_updated else 0.5

    def _assess_uniqueness(self, data: Dict) -> float:
        """Assess data uniqueness (no duplicates)."""
        records = data.get('records', [])
        if not records:
            return 1.0

        # Convert records to hashable form for duplicate detection
        unique_records = set()
        for record in records:
            record_tuple = tuple(sorted(str(item) for item in record.items()))
            unique_records.add(record_tuple)

        return len(unique_records) / len(records)

    def _assess_validity(self, data: Dict, schema: Dict) -> float:
        """Assess data validity against schema constraints."""
        records = data.get('records', [])
        if not records:
            return 1.0

        constraints = schema.get('constraints', {})
        if not constraints:
            return 0.9

        valid_count = 0
        total_checks = 0

        for record in records:
            for field, constraint in constraints.items():
                if field in record:
                    total_checks += 1
                    value = record[field]

                    # Check min/max constraints
                    is_valid = True
                    if 'min' in constraint and value < constraint['min']:
                        is_valid = False
                    if 'max' in constraint and value > constraint['max']:
                        is_valid = False
                    if 'enum' in constraint and value not in constraint['enum']:
                        is_valid = False

                    if is_valid:
                        valid_count += 1

        return valid_count / max(1, total_checks)

    def _collect_issues(
        self,
        completeness: float,
        accuracy: float,
        consistency: float,
        timeliness: float,
        uniqueness: float,
        validity: float
    ) -> List[str]:
        """Collect quality issues."""
        issues = []

        if completeness < 0.9:
            issues.append(f'Completeness below threshold: {completeness:.2%}')
        if accuracy < 0.9:
            issues.append(f'Accuracy below threshold: {accuracy:.2%}')
        if consistency < 0.95:
            issues.append(f'Consistency issues detected: {consistency:.2%}')
        if timeliness < 0.8:
            issues.append(f'Data may be stale: {timeliness:.2%}')
        if uniqueness < 0.95:
            issues.append(f'Duplicate records detected: {uniqueness:.2%}')
        if validity < 0.9:
            issues.append(f'Validity constraints violated: {validity:.2%}')

        return issues


# =============================================================================
# Comprehensive Data Policy Analyzer
# =============================================================================

class DataPolicyAnalyzer:
    """Comprehensive analyzer for all data policies."""

    def __init__(self):
        self.masking_analyzer = DataMaskingAnalyzer()
        self.reduction_analyzer = DataReductionAnalyzer()
        self.retention_analyzer = DataRetentionAnalyzer()
        self.classification_analyzer = DataClassificationAnalyzer()
        self.quality_analyzer = DataQualityAnalyzer()

    def analyze(
        self,
        data: Dict[str, Any],
        policies: Dict[str, Any],
        config: Optional[Dict] = None
    ) -> DataPolicyAssessment:
        """Perform comprehensive data policy analysis."""
        config = config or {}

        # Masking analysis
        masking_result = None
        if 'text_content' in data:
            masking_result = self.masking_analyzer.analyze(
                data['text_content'],
                data.get('masked_content'),
                config.get('masking', {})
            )
            masking_score = masking_result.masking_coverage
        else:
            masking_score = 1.0

        # Reduction analysis
        reduction_result = self.reduction_analyzer.analyze(
            data.get('original_data', {}),
            data.get('reduced_data'),
            config.get('reduction', {})
        )
        reduction_score = reduction_result.minimization_score

        # Retention analysis
        retention_result = self.retention_analyzer.analyze(
            data.get('data_category', 'general'),
            policies.get('retention', {}),
            data.get('metadata', {})
        )
        retention_score = 1.0 if retention_result.compliance_status == ComplianceStatus.COMPLIANT else 0.5

        # Classification analysis
        classification_result = self.classification_analyzer.classify(
            data,
            config.get('classification_context', {})
        )
        classification_score = classification_result.confidence

        # Quality analysis
        quality_result = self.quality_analyzer.analyze(
            data,
            policies.get('schema', {}),
            config.get('quality', {})
        )
        quality_score = quality_result.overall_quality

        # Overall compliance
        scores = [masking_score, reduction_score, retention_score, classification_score, quality_score]
        avg_score = sum(scores) / len(scores)

        if avg_score >= 0.9:
            overall_compliance = ComplianceStatus.COMPLIANT
        elif avg_score >= 0.7:
            overall_compliance = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_compliance = ComplianceStatus.NON_COMPLIANT

        # Collect findings
        findings = self._collect_findings(
            masking_result, reduction_result, retention_result,
            classification_result, quality_result
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            masking_score, reduction_score, retention_score,
            classification_score, quality_score
        )

        # Assess risk
        risk_level = 'low' if avg_score >= 0.8 else 'medium' if avg_score >= 0.6 else 'high'

        return DataPolicyAssessment(
            masking_score=masking_score,
            reduction_score=reduction_score,
            retention_score=retention_score,
            classification_score=classification_score,
            quality_score=quality_score,
            overall_compliance=overall_compliance,
            findings=findings,
            recommendations=recommendations,
            risk_level=risk_level
        )

    def _collect_findings(
        self,
        masking: Optional[MaskingResult],
        reduction: DataReductionResult,
        retention: RetentionAnalysisResult,
        classification: DataClassificationResult,
        quality: DataQualityMetrics
    ) -> List[str]:
        """Collect findings from all analyses."""
        findings = []

        if masking and masking.masking_coverage < 0.95:
            findings.append(f'Masking coverage: {masking.masking_coverage:.1%}')

        if reduction.information_loss_estimate > 0.3:
            findings.append(f'Information loss from reduction: {reduction.information_loss_estimate:.1%}')

        if retention.compliance_status != ComplianceStatus.COMPLIANT:
            findings.append(f'Retention compliance: {retention.compliance_status.value}')

        if quality.overall_quality < 0.9:
            findings.extend(quality.issues_detected[:3])

        return findings

    def _generate_recommendations(
        self,
        masking: float,
        reduction: float,
        retention: float,
        classification: float,
        quality: float
    ) -> List[str]:
        """Generate recommendations based on scores."""
        recommendations = []

        if masking < 0.95:
            recommendations.append('Improve PII masking coverage')
        if reduction < 0.8:
            recommendations.append('Review data minimization practices')
        if retention < 0.9:
            recommendations.append('Align retention policy with requirements')
        if classification < 0.8:
            recommendations.append('Validate data classification labels')
        if quality < 0.9:
            recommendations.append('Address data quality issues')

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_data_masking(
    text: str,
    masked_text: Optional[str] = None
) -> MaskingResult:
    """Convenience function for data masking analysis."""
    analyzer = DataMaskingAnalyzer()
    return analyzer.analyze(text, masked_text)


def analyze_data_reduction(
    original: Dict[str, Any],
    reduced: Optional[Dict[str, Any]] = None
) -> DataReductionResult:
    """Convenience function for data reduction analysis."""
    analyzer = DataReductionAnalyzer()
    return analyzer.analyze(original, reduced)


def analyze_data_retention(
    category: str,
    policy: Dict[str, Any]
) -> RetentionAnalysisResult:
    """Convenience function for retention analysis."""
    analyzer = DataRetentionAnalyzer()
    return analyzer.analyze(category, policy)


def classify_data(data: Dict[str, Any]) -> DataClassificationResult:
    """Convenience function for data classification."""
    analyzer = DataClassificationAnalyzer()
    return analyzer.classify(data)


def analyze_data_quality(
    data: Dict[str, Any],
    schema: Optional[Dict] = None
) -> DataQualityMetrics:
    """Convenience function for data quality analysis."""
    analyzer = DataQualityAnalyzer()
    return analyzer.analyze(data, schema)


def analyze_data_policies(
    data: Dict[str, Any],
    policies: Dict[str, Any]
) -> DataPolicyAssessment:
    """Convenience function for comprehensive data policy analysis."""
    analyzer = DataPolicyAnalyzer()
    return analyzer.analyze(data, policies)
