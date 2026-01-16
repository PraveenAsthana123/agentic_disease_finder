"""
Data Lifecycle Analysis Module
==============================

Comprehensive data analysis framework covering 18 data analysis categories
for end-to-end AI data lifecycle management.

Categories:
1. Data Inventory Analysis - Catalog and track all data assets
2. PII/PHI Detection Analysis - Identify sensitive data elements
3. Data Minimization Analysis - Ensure minimal data collection
4. Data Quality Analysis - Assess data quality dimensions
5. EDA Analysis - Exploratory data analysis
6. Bias/Fairness Analysis - Data-level bias detection
7. Feature Engineering Analysis - Feature quality and relevance
8. Data Drift Analysis - Monitor data distribution changes
9. Model Input Contract Analysis - Validate input specifications
10. Training Data Analysis - Training dataset evaluation
11. Model Performance Analysis - Performance on data subsets
12. Hallucination/Faithfulness Analysis - Output grounding
13. Robustness/Stress Analysis - Data perturbation testing
14. Explainability Analysis - Data-driven explanations
15. Human-Centered Trust Analysis - User trust in data
16. Security/Access Analysis - Data access control
17. Retention/Deletion Analysis - Data lifecycle management
18. Incident/Post-Mortem Analysis - Data-related incident analysis
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime


# =============================================================================
# ENUMS
# =============================================================================

class DataAssetType(Enum):
    """Types of data assets in inventory."""
    STRUCTURED = auto()
    UNSTRUCTURED = auto()
    SEMI_STRUCTURED = auto()
    STREAMING = auto()
    TIME_SERIES = auto()
    IMAGE = auto()
    TEXT = auto()
    AUDIO = auto()
    VIDEO = auto()
    MULTIMODAL = auto()


class SensitiveDataType(Enum):
    """Types of sensitive data."""
    PII_DIRECT = auto()  # Direct identifiers (name, SSN)
    PII_INDIRECT = auto()  # Quasi-identifiers (DOB, ZIP)
    PHI = auto()  # Protected Health Information
    PCI = auto()  # Payment Card Information
    BIOMETRIC = auto()
    GENETIC = auto()
    LOCATION = auto()
    BEHAVIORAL = auto()
    FINANCIAL = auto()
    CREDENTIALS = auto()


class DataQualityDimension(Enum):
    """Dimensions of data quality."""
    COMPLETENESS = auto()
    ACCURACY = auto()
    CONSISTENCY = auto()
    TIMELINESS = auto()
    VALIDITY = auto()
    UNIQUENESS = auto()
    INTEGRITY = auto()
    RELEVANCE = auto()
    ACCESSIBILITY = auto()
    CONFORMITY = auto()


class DriftSeverity(Enum):
    """Severity levels for data drift."""
    NONE = auto()
    MINOR = auto()
    MODERATE = auto()
    SIGNIFICANT = auto()
    CRITICAL = auto()


class FeatureType(Enum):
    """Types of features for engineering analysis."""
    NUMERICAL = auto()
    CATEGORICAL = auto()
    ORDINAL = auto()
    BINARY = auto()
    TEXT = auto()
    TEMPORAL = auto()
    SPATIAL = auto()
    EMBEDDING = auto()
    DERIVED = auto()
    INTERACTION = auto()


class BiasSource(Enum):
    """Sources of bias in data."""
    HISTORICAL = auto()
    REPRESENTATION = auto()
    MEASUREMENT = auto()
    AGGREGATION = auto()
    SAMPLING = auto()
    LABELING = auto()
    TEMPORAL = auto()
    SURVIVORSHIP = auto()
    SELECTION = auto()
    CONFIRMATION = auto()


class IncidentSeverity(Enum):
    """Severity levels for data incidents."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class RetentionStatus(Enum):
    """Data retention status."""
    ACTIVE = auto()
    ARCHIVED = auto()
    PENDING_DELETION = auto()
    DELETED = auto()
    LEGAL_HOLD = auto()


class AccessLevel(Enum):
    """Data access levels."""
    PUBLIC = auto()
    INTERNAL = auto()
    CONFIDENTIAL = auto()
    RESTRICTED = auto()
    TOP_SECRET = auto()


class ValidationStatus(Enum):
    """Input contract validation status."""
    VALID = auto()
    INVALID = auto()
    WARNING = auto()
    MISSING = auto()
    TYPE_MISMATCH = auto()
    RANGE_VIOLATION = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataAsset:
    """Represents a data asset in the inventory."""
    asset_id: str
    name: str
    asset_type: DataAssetType
    source: str
    owner: str
    description: str = ""
    schema: Dict[str, str] = field(default_factory=dict)
    row_count: int = 0
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    lineage: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    access_level: AccessLevel = AccessLevel.INTERNAL
    retention_policy: str = ""


@dataclass
class SensitiveDataFinding:
    """Finding from sensitive data detection."""
    field_name: str
    data_type: SensitiveDataType
    confidence: float
    sample_count: int
    detection_method: str
    risk_level: str
    recommendations: List[str] = field(default_factory=list)
    patterns_found: List[str] = field(default_factory=list)


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    dimension: DataQualityDimension
    score: float
    issues_found: int
    affected_records: int
    affected_percentage: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DriftDetectionResult:
    """Result from data drift detection."""
    feature_name: str
    drift_score: float
    severity: DriftSeverity
    baseline_stats: Dict[str, float] = field(default_factory=dict)
    current_stats: Dict[str, float] = field(default_factory=dict)
    statistical_test: str = ""
    p_value: float = 0.0
    drift_type: str = ""  # covariate, prior, concept


@dataclass
class FeatureAnalysisResult:
    """Result from feature engineering analysis."""
    feature_name: str
    feature_type: FeatureType
    importance_score: float
    correlation_with_target: float
    missing_rate: float
    cardinality: int = 0
    distribution_type: str = ""
    outlier_rate: float = 0.0
    encoding_recommendation: str = ""
    transformation_applied: List[str] = field(default_factory=list)


@dataclass
class BiasAnalysisResult:
    """Result from bias analysis in data."""
    bias_source: BiasSource
    affected_groups: List[str]
    disparity_ratio: float
    statistical_parity_difference: float
    recommendations: List[str] = field(default_factory=list)
    mitigation_applied: bool = False


@dataclass
class InputContractViolation:
    """Violation of model input contract."""
    field_name: str
    expected_type: str
    actual_type: str
    expected_range: Optional[Tuple[Any, Any]] = None
    actual_value: Any = None
    status: ValidationStatus = ValidationStatus.INVALID
    error_message: str = ""


@dataclass
class TrainingDataMetrics:
    """Metrics for training data analysis."""
    total_samples: int
    class_distribution: Dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 0.0
    feature_coverage: Dict[str, float] = field(default_factory=dict)
    temporal_coverage: Optional[Tuple[datetime, datetime]] = None
    data_version: str = ""
    split_ratios: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceSubsetResult:
    """Performance analysis on data subsets."""
    subset_name: str
    subset_criteria: Dict[str, Any]
    sample_count: int
    metrics: Dict[str, float] = field(default_factory=dict)
    comparison_to_overall: Dict[str, float] = field(default_factory=dict)
    significant_differences: List[str] = field(default_factory=list)


@dataclass
class FaithfulnessResult:
    """Result from faithfulness/hallucination analysis."""
    source_coverage: float
    factual_consistency: float
    attribution_accuracy: float
    hallucination_rate: float
    unsupported_claims: List[str] = field(default_factory=list)
    grounding_evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DataIncident:
    """Data-related incident record."""
    incident_id: str
    severity: IncidentSeverity
    incident_type: str
    affected_assets: List[str]
    root_cause: str
    impact_assessment: str
    resolution: str = ""
    lessons_learned: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


@dataclass
class RetentionRecord:
    """Data retention record."""
    asset_id: str
    status: RetentionStatus
    retention_period_days: int
    created_at: datetime
    expiry_date: datetime
    legal_basis: str = ""
    deletion_date: Optional[datetime] = None
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AccessAuditRecord:
    """Data access audit record."""
    asset_id: str
    user_id: str
    access_type: str
    access_level: AccessLevel
    timestamp: datetime
    purpose: str = ""
    granted: bool = True
    denial_reason: str = ""


@dataclass
class DataLifecycleAssessment:
    """Comprehensive data lifecycle assessment."""
    assessment_id: str
    timestamp: datetime
    inventory_completeness: float
    sensitive_data_coverage: float
    quality_score: float
    bias_risk_score: float
    drift_alert_count: int
    contract_compliance_rate: float
    retention_compliance_rate: float
    security_posture_score: float
    incident_rate: float
    overall_health_score: float
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# ANALYZERS - DATA INVENTORY
# =============================================================================

class DataInventoryAnalyzer:
    """Analyzer for data asset inventory management."""

    def __init__(self):
        self.assets: Dict[str, DataAsset] = {}

    def register_asset(self, asset: DataAsset) -> None:
        """Register a new data asset."""
        self.assets[asset.asset_id] = asset

    def analyze_inventory(self, assets: List[DataAsset]) -> Dict[str, Any]:
        """Analyze data inventory completeness and coverage."""
        for asset in assets:
            self.register_asset(asset)

        type_distribution = {}
        access_distribution = {}
        total_size = 0

        for asset in assets:
            type_name = asset.asset_type.name
            type_distribution[type_name] = type_distribution.get(type_name, 0) + 1

            access_name = asset.access_level.name
            access_distribution[access_name] = access_distribution.get(access_name, 0) + 1

            total_size += asset.size_bytes

        return {
            "total_assets": len(assets),
            "type_distribution": type_distribution,
            "access_distribution": access_distribution,
            "total_size_bytes": total_size,
            "avg_quality_score": sum(a.quality_score for a in assets) / len(assets) if assets else 0,
            "assets_with_lineage": sum(1 for a in assets if a.lineage),
            "assets_with_schema": sum(1 for a in assets if a.schema),
        }

    def find_orphan_assets(self) -> List[DataAsset]:
        """Find assets without lineage or clear ownership."""
        return [a for a in self.assets.values() if not a.lineage and not a.owner]

    def get_lineage_graph(self, asset_id: str) -> Dict[str, List[str]]:
        """Get data lineage graph for an asset."""
        if asset_id not in self.assets:
            return {}
        return {"asset": asset_id, "upstream": self.assets[asset_id].lineage}


class DataCatalogAnalyzer:
    """Analyzer for data catalog management."""

    def analyze_catalog_coverage(self, assets: List[DataAsset]) -> Dict[str, Any]:
        """Analyze how well data assets are cataloged."""
        documented = sum(1 for a in assets if a.description)
        tagged = sum(1 for a in assets if a.tags)
        with_schema = sum(1 for a in assets if a.schema)

        return {
            "total_assets": len(assets),
            "documented_rate": documented / len(assets) if assets else 0,
            "tagged_rate": tagged / len(assets) if assets else 0,
            "schema_documented_rate": with_schema / len(assets) if assets else 0,
            "unique_tags": len(set(tag for a in assets for tag in a.tags)),
        }


# =============================================================================
# ANALYZERS - SENSITIVE DATA
# =============================================================================

class PIIDetectionAnalyzer:
    """Analyzer for PII/PHI detection in data."""

    def __init__(self):
        self.patterns = {
            SensitiveDataType.PII_DIRECT: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z]+\s+[A-Za-z]+\b',  # Name patterns
            ],
            SensitiveDataType.PHI: [
                r'\bMRN\d+\b',  # Medical Record Number
            ],
        }

    def detect_sensitive_data(
        self,
        data: Dict[str, List[Any]],
        columns: Optional[List[str]] = None
    ) -> List[SensitiveDataFinding]:
        """Detect sensitive data in dataset columns."""
        findings = []
        check_columns = columns or list(data.keys())

        for col in check_columns:
            if col not in data:
                continue

            # Simplified detection based on column names
            col_lower = col.lower()

            if any(term in col_lower for term in ['ssn', 'social', 'tax_id']):
                findings.append(SensitiveDataFinding(
                    field_name=col,
                    data_type=SensitiveDataType.PII_DIRECT,
                    confidence=0.95,
                    sample_count=len(data[col]),
                    detection_method="column_name_heuristic",
                    risk_level="HIGH",
                    recommendations=["Mask or encrypt this field", "Review access controls"],
                ))

            elif any(term in col_lower for term in ['name', 'email', 'phone', 'address']):
                findings.append(SensitiveDataFinding(
                    field_name=col,
                    data_type=SensitiveDataType.PII_INDIRECT,
                    confidence=0.85,
                    sample_count=len(data[col]),
                    detection_method="column_name_heuristic",
                    risk_level="MEDIUM",
                    recommendations=["Consider anonymization", "Apply access controls"],
                ))

            elif any(term in col_lower for term in ['diagnosis', 'medical', 'health', 'mrn']):
                findings.append(SensitiveDataFinding(
                    field_name=col,
                    data_type=SensitiveDataType.PHI,
                    confidence=0.90,
                    sample_count=len(data[col]),
                    detection_method="column_name_heuristic",
                    risk_level="CRITICAL",
                    recommendations=["HIPAA compliance required", "Strict access controls"],
                ))

        return findings

    def calculate_exposure_risk(self, findings: List[SensitiveDataFinding]) -> float:
        """Calculate overall sensitive data exposure risk."""
        if not findings:
            return 0.0

        risk_weights = {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}
        total_risk = sum(
            risk_weights.get(f.risk_level, 0.5) * f.confidence
            for f in findings
        )
        return min(total_risk / len(findings), 1.0)


class PHIDetectionAnalyzer:
    """Specialized analyzer for Protected Health Information."""

    def detect_phi(self, data: Dict[str, List[Any]]) -> List[SensitiveDataFinding]:
        """Detect PHI in healthcare data."""
        phi_indicators = [
            'patient', 'diagnosis', 'treatment', 'medication',
            'prescription', 'lab_result', 'vital', 'allergy',
            'procedure', 'condition', 'symptom', 'mrn'
        ]

        findings = []
        for col in data.keys():
            col_lower = col.lower()
            if any(ind in col_lower for ind in phi_indicators):
                findings.append(SensitiveDataFinding(
                    field_name=col,
                    data_type=SensitiveDataType.PHI,
                    confidence=0.90,
                    sample_count=len(data[col]),
                    detection_method="phi_keyword_detection",
                    risk_level="CRITICAL",
                    recommendations=[
                        "Ensure HIPAA compliance",
                        "Implement audit logging",
                        "Apply role-based access control"
                    ],
                ))
        return findings


# =============================================================================
# ANALYZERS - DATA MINIMIZATION
# =============================================================================

class DataMinimizationAnalyzer:
    """Analyzer for data minimization compliance."""

    def analyze_minimization(
        self,
        collected_fields: List[str],
        required_fields: List[str],
        purpose: str
    ) -> Dict[str, Any]:
        """Analyze if data collection follows minimization principles."""
        required_set = set(required_fields)
        collected_set = set(collected_fields)

        excess_fields = collected_set - required_set
        missing_required = required_set - collected_set

        minimization_score = len(required_fields) / len(collected_fields) if collected_fields else 1.0

        return {
            "purpose": purpose,
            "required_fields": len(required_fields),
            "collected_fields": len(collected_fields),
            "excess_fields": list(excess_fields),
            "missing_required": list(missing_required),
            "minimization_score": minimization_score,
            "compliant": len(excess_fields) == 0,
            "recommendations": [
                f"Consider removing field: {f}" for f in list(excess_fields)[:5]
            ],
        }

    def calculate_necessity_score(
        self,
        field_name: str,
        usage_count: int,
        total_operations: int
    ) -> float:
        """Calculate necessity score for a field."""
        if total_operations == 0:
            return 0.0
        return min(usage_count / total_operations, 1.0)


# =============================================================================
# ANALYZERS - DATA QUALITY
# =============================================================================

class DataQualityAnalyzer:
    """Comprehensive data quality analyzer."""

    def analyze_completeness(self, data: Dict[str, List[Any]]) -> DataQualityMetrics:
        """Analyze data completeness."""
        if not data:
            return DataQualityMetrics(
                dimension=DataQualityDimension.COMPLETENESS,
                score=0.0,
                issues_found=0,
                affected_records=0,
                affected_percentage=0.0,
            )

        total_cells = sum(len(col) for col in data.values())
        null_cells = sum(
            sum(1 for v in col if v is None or v == '' or (isinstance(v, float) and v != v))
            for col in data.values()
        )

        score = 1.0 - (null_cells / total_cells) if total_cells > 0 else 0.0

        return DataQualityMetrics(
            dimension=DataQualityDimension.COMPLETENESS,
            score=score,
            issues_found=null_cells,
            affected_records=null_cells,
            affected_percentage=(null_cells / total_cells) * 100 if total_cells > 0 else 0,
            details={"null_cells": null_cells, "total_cells": total_cells},
            recommendations=["Address missing values" if score < 0.95 else "Data completeness acceptable"],
        )

    def analyze_uniqueness(self, data: Dict[str, List[Any]], key_columns: List[str]) -> DataQualityMetrics:
        """Analyze data uniqueness based on key columns."""
        if not key_columns or not data:
            return DataQualityMetrics(
                dimension=DataQualityDimension.UNIQUENESS,
                score=1.0,
                issues_found=0,
                affected_records=0,
                affected_percentage=0.0,
            )

        # Create composite keys
        try:
            key_values = []
            for i in range(len(data[key_columns[0]])):
                key = tuple(data[col][i] for col in key_columns if col in data)
                key_values.append(key)
        except (KeyError, IndexError):
            return DataQualityMetrics(
                dimension=DataQualityDimension.UNIQUENESS,
                score=0.0,
                issues_found=0,
                affected_records=0,
                affected_percentage=0.0,
                details={"error": "Invalid key columns"},
            )

        unique_count = len(set(key_values))
        total_count = len(key_values)
        duplicates = total_count - unique_count

        return DataQualityMetrics(
            dimension=DataQualityDimension.UNIQUENESS,
            score=unique_count / total_count if total_count > 0 else 1.0,
            issues_found=duplicates,
            affected_records=duplicates,
            affected_percentage=(duplicates / total_count) * 100 if total_count > 0 else 0,
            details={"unique_keys": unique_count, "total_keys": total_count},
            recommendations=["Remove duplicate records" if duplicates > 0 else "No duplicates found"],
        )

    def analyze_all_dimensions(self, data: Dict[str, List[Any]]) -> List[DataQualityMetrics]:
        """Analyze all data quality dimensions."""
        results = []
        results.append(self.analyze_completeness(data))
        # Add more dimension analyses as needed
        return results


class DataConsistencyAnalyzer:
    """Analyzer for data consistency across sources."""

    def check_consistency(
        self,
        source_data: Dict[str, List[Any]],
        target_data: Dict[str, List[Any]],
        key_column: str
    ) -> Dict[str, Any]:
        """Check data consistency between two sources."""
        if key_column not in source_data or key_column not in target_data:
            return {"error": "Key column not found in both datasets"}

        source_keys = set(source_data[key_column])
        target_keys = set(target_data[key_column])

        return {
            "source_only": len(source_keys - target_keys),
            "target_only": len(target_keys - source_keys),
            "common": len(source_keys & target_keys),
            "consistency_score": len(source_keys & target_keys) / len(source_keys | target_keys) if source_keys | target_keys else 1.0,
        }


# =============================================================================
# ANALYZERS - EXPLORATORY DATA ANALYSIS
# =============================================================================

class EDAAnalyzer:
    """Exploratory Data Analysis analyzer."""

    def compute_basic_stats(self, data: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
        """Compute basic statistics for each column."""
        stats = {}
        for col, values in data.items():
            numeric_values = [v for v in values if isinstance(v, (int, float)) and v == v]

            if numeric_values:
                stats[col] = {
                    "type": "numeric",
                    "count": len(values),
                    "non_null": len(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                }
            else:
                unique_values = set(v for v in values if v is not None)
                stats[col] = {
                    "type": "categorical",
                    "count": len(values),
                    "non_null": len([v for v in values if v is not None]),
                    "unique": len(unique_values),
                }
        return stats

    def detect_outliers(
        self,
        values: List[float],
        method: str = "iqr"
    ) -> Dict[str, Any]:
        """Detect outliers in numeric data."""
        if not values or len(values) < 4:
            return {"outliers": [], "count": 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [v for v in values if v < lower_bound or v > upper_bound]

        return {
            "method": method,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers": outliers[:10],  # Limit to first 10
            "count": len(outliers),
            "percentage": len(outliers) / len(values) * 100,
        }


# =============================================================================
# ANALYZERS - BIAS AND FAIRNESS
# =============================================================================

class DataBiasAnalyzer:
    """Analyzer for detecting bias in data."""

    def analyze_representation_bias(
        self,
        data: Dict[str, List[Any]],
        protected_attribute: str,
        expected_distribution: Optional[Dict[str, float]] = None
    ) -> BiasAnalysisResult:
        """Analyze representation bias in data."""
        if protected_attribute not in data:
            return BiasAnalysisResult(
                bias_source=BiasSource.REPRESENTATION,
                affected_groups=[],
                disparity_ratio=0.0,
                statistical_parity_difference=0.0,
                recommendations=["Protected attribute not found"],
            )

        values = data[protected_attribute]
        total = len(values)

        # Calculate actual distribution
        actual_dist = {}
        for v in values:
            if v is not None:
                actual_dist[str(v)] = actual_dist.get(str(v), 0) + 1

        # Normalize
        for k in actual_dist:
            actual_dist[k] = actual_dist[k] / total

        # Calculate disparity
        if expected_distribution:
            max_disparity = 0.0
            affected = []
            for group, expected in expected_distribution.items():
                actual = actual_dist.get(group, 0)
                disparity = abs(actual - expected) / expected if expected > 0 else 0
                if disparity > max_disparity:
                    max_disparity = disparity
                if disparity > 0.2:  # 20% threshold
                    affected.append(group)

            return BiasAnalysisResult(
                bias_source=BiasSource.REPRESENTATION,
                affected_groups=affected,
                disparity_ratio=max_disparity,
                statistical_parity_difference=max_disparity,
                recommendations=["Consider resampling or reweighting" if affected else "Distribution acceptable"],
            )

        # Without expected distribution, check for imbalance
        if actual_dist:
            values_list = list(actual_dist.values())
            max_val = max(values_list)
            min_val = min(values_list)
            imbalance = max_val / min_val if min_val > 0 else float('inf')

            return BiasAnalysisResult(
                bias_source=BiasSource.REPRESENTATION,
                affected_groups=[k for k, v in actual_dist.items() if v < 0.1],
                disparity_ratio=imbalance,
                statistical_parity_difference=max_val - min_val,
                recommendations=["Significant class imbalance detected"] if imbalance > 3 else [],
            )

        return BiasAnalysisResult(
            bias_source=BiasSource.REPRESENTATION,
            affected_groups=[],
            disparity_ratio=0.0,
            statistical_parity_difference=0.0,
        )

    def analyze_label_bias(
        self,
        data: Dict[str, List[Any]],
        label_column: str,
        protected_attribute: str
    ) -> BiasAnalysisResult:
        """Analyze labeling bias across protected groups."""
        if label_column not in data or protected_attribute not in data:
            return BiasAnalysisResult(
                bias_source=BiasSource.LABELING,
                affected_groups=[],
                disparity_ratio=0.0,
                statistical_parity_difference=0.0,
                recommendations=["Required columns not found"],
            )

        labels = data[label_column]
        protected = data[protected_attribute]

        # Calculate positive rate per group
        group_positive_rates = {}
        group_counts = {}

        for i, (label, group) in enumerate(zip(labels, protected)):
            if group is None:
                continue
            group_str = str(group)
            group_counts[group_str] = group_counts.get(group_str, 0) + 1
            if label == 1 or label == True or label == "positive":
                group_positive_rates[group_str] = group_positive_rates.get(group_str, 0) + 1

        # Normalize
        for g in group_positive_rates:
            if group_counts[g] > 0:
                group_positive_rates[g] = group_positive_rates[g] / group_counts[g]

        if len(group_positive_rates) < 2:
            return BiasAnalysisResult(
                bias_source=BiasSource.LABELING,
                affected_groups=[],
                disparity_ratio=0.0,
                statistical_parity_difference=0.0,
            )

        rates = list(group_positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        return BiasAnalysisResult(
            bias_source=BiasSource.LABELING,
            affected_groups=[g for g, r in group_positive_rates.items() if r == min_rate],
            disparity_ratio=min_rate / max_rate if max_rate > 0 else 0,
            statistical_parity_difference=max_rate - min_rate,
            recommendations=["Review labeling process" if max_rate - min_rate > 0.1 else "Label distribution acceptable"],
        )


# =============================================================================
# ANALYZERS - FEATURE ENGINEERING
# =============================================================================

class FeatureEngineeringAnalyzer:
    """Analyzer for feature engineering quality."""

    def analyze_feature(
        self,
        values: List[Any],
        feature_name: str,
        target_values: Optional[List[Any]] = None
    ) -> FeatureAnalysisResult:
        """Analyze a single feature."""
        # Determine feature type
        if all(isinstance(v, bool) for v in values if v is not None):
            feature_type = FeatureType.BINARY
        elif all(isinstance(v, (int, float)) for v in values if v is not None):
            feature_type = FeatureType.NUMERICAL
        elif all(isinstance(v, str) for v in values if v is not None):
            unique = len(set(v for v in values if v is not None))
            feature_type = FeatureType.CATEGORICAL if unique < 50 else FeatureType.TEXT
        else:
            feature_type = FeatureType.CATEGORICAL

        # Calculate metrics
        non_null = [v for v in values if v is not None]
        missing_rate = 1 - len(non_null) / len(values) if values else 0

        # Calculate cardinality
        cardinality = len(set(v for v in values if v is not None))

        # Simple importance (correlation with target if provided)
        importance = 0.0
        correlation = 0.0
        if target_values and feature_type == FeatureType.NUMERICAL:
            # Simplified correlation calculation
            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            numeric_targets = [t for t in target_values if isinstance(t, (int, float))]
            if numeric_vals and numeric_targets:
                importance = 0.5  # Placeholder

        return FeatureAnalysisResult(
            feature_name=feature_name,
            feature_type=feature_type,
            importance_score=importance,
            correlation_with_target=correlation,
            missing_rate=missing_rate,
            cardinality=cardinality,
            encoding_recommendation="one-hot" if feature_type == FeatureType.CATEGORICAL and cardinality < 10 else "label",
        )

    def analyze_all_features(
        self,
        data: Dict[str, List[Any]],
        target_column: Optional[str] = None
    ) -> List[FeatureAnalysisResult]:
        """Analyze all features in dataset."""
        results = []
        target_values = data.get(target_column) if target_column else None

        for col, values in data.items():
            if col != target_column:
                results.append(self.analyze_feature(values, col, target_values))

        return results


# =============================================================================
# ANALYZERS - DATA DRIFT
# =============================================================================

class DataDriftAnalyzer:
    """Analyzer for detecting data drift."""

    def detect_drift(
        self,
        baseline: List[float],
        current: List[float],
        feature_name: str,
        threshold: float = 0.1
    ) -> DriftDetectionResult:
        """Detect drift between baseline and current distributions."""
        if not baseline or not current:
            return DriftDetectionResult(
                feature_name=feature_name,
                drift_score=0.0,
                severity=DriftSeverity.NONE,
            )

        # Calculate basic statistics
        baseline_mean = sum(baseline) / len(baseline)
        current_mean = sum(current) / len(current)

        baseline_std = (sum((x - baseline_mean) ** 2 for x in baseline) / len(baseline)) ** 0.5
        current_std = (sum((x - current_mean) ** 2 for x in current) / len(current)) ** 0.5

        # Simple drift score based on mean shift
        if baseline_std > 0:
            drift_score = abs(current_mean - baseline_mean) / baseline_std
        else:
            drift_score = abs(current_mean - baseline_mean)

        # Determine severity
        if drift_score < 0.1:
            severity = DriftSeverity.NONE
        elif drift_score < 0.5:
            severity = DriftSeverity.MINOR
        elif drift_score < 1.0:
            severity = DriftSeverity.MODERATE
        elif drift_score < 2.0:
            severity = DriftSeverity.SIGNIFICANT
        else:
            severity = DriftSeverity.CRITICAL

        return DriftDetectionResult(
            feature_name=feature_name,
            drift_score=drift_score,
            severity=severity,
            baseline_stats={"mean": baseline_mean, "std": baseline_std},
            current_stats={"mean": current_mean, "std": current_std},
            statistical_test="normalized_mean_shift",
            drift_type="covariate",
        )

    def detect_all_drift(
        self,
        baseline_data: Dict[str, List[float]],
        current_data: Dict[str, List[float]]
    ) -> List[DriftDetectionResult]:
        """Detect drift across all features."""
        results = []
        for feature in baseline_data:
            if feature in current_data:
                results.append(self.detect_drift(
                    baseline_data[feature],
                    current_data[feature],
                    feature
                ))
        return results


# =============================================================================
# ANALYZERS - MODEL INPUT CONTRACT
# =============================================================================

class InputContractAnalyzer:
    """Analyzer for model input contract validation."""

    def __init__(self):
        self.contract: Dict[str, Dict[str, Any]] = {}

    def define_contract(self, contract: Dict[str, Dict[str, Any]]) -> None:
        """Define input contract specification."""
        self.contract = contract

    def validate_input(self, input_data: Dict[str, Any]) -> List[InputContractViolation]:
        """Validate input against contract."""
        violations = []

        for field, spec in self.contract.items():
            if field not in input_data:
                violations.append(InputContractViolation(
                    field_name=field,
                    expected_type=spec.get("type", "any"),
                    actual_type="missing",
                    status=ValidationStatus.MISSING,
                    error_message=f"Required field {field} is missing",
                ))
                continue

            value = input_data[field]
            expected_type = spec.get("type")

            # Type check
            if expected_type:
                if expected_type == "int" and not isinstance(value, int):
                    violations.append(InputContractViolation(
                        field_name=field,
                        expected_type=expected_type,
                        actual_type=type(value).__name__,
                        actual_value=value,
                        status=ValidationStatus.TYPE_MISMATCH,
                        error_message=f"Expected {expected_type}, got {type(value).__name__}",
                    ))
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    violations.append(InputContractViolation(
                        field_name=field,
                        expected_type=expected_type,
                        actual_type=type(value).__name__,
                        actual_value=value,
                        status=ValidationStatus.TYPE_MISMATCH,
                        error_message=f"Expected {expected_type}, got {type(value).__name__}",
                    ))

            # Range check
            if "min" in spec and isinstance(value, (int, float)):
                if value < spec["min"]:
                    violations.append(InputContractViolation(
                        field_name=field,
                        expected_type=expected_type or "numeric",
                        actual_type=type(value).__name__,
                        expected_range=(spec["min"], spec.get("max")),
                        actual_value=value,
                        status=ValidationStatus.RANGE_VIOLATION,
                        error_message=f"Value {value} below minimum {spec['min']}",
                    ))

            if "max" in spec and isinstance(value, (int, float)):
                if value > spec["max"]:
                    violations.append(InputContractViolation(
                        field_name=field,
                        expected_type=expected_type or "numeric",
                        actual_type=type(value).__name__,
                        expected_range=(spec.get("min"), spec["max"]),
                        actual_value=value,
                        status=ValidationStatus.RANGE_VIOLATION,
                        error_message=f"Value {value} above maximum {spec['max']}",
                    ))

        return violations

    def get_compliance_rate(self, inputs: List[Dict[str, Any]]) -> float:
        """Calculate contract compliance rate across inputs."""
        if not inputs:
            return 1.0
        compliant = sum(1 for inp in inputs if not self.validate_input(inp))
        return compliant / len(inputs)


# =============================================================================
# ANALYZERS - TRAINING DATA
# =============================================================================

class TrainingDataAnalyzer:
    """Analyzer for training data quality and characteristics."""

    def analyze_training_data(
        self,
        data: Dict[str, List[Any]],
        label_column: str,
        data_version: str = "1.0"
    ) -> TrainingDataMetrics:
        """Analyze training dataset."""
        if label_column not in data:
            return TrainingDataMetrics(total_samples=0, data_version=data_version)

        labels = data[label_column]
        total = len(labels)

        # Class distribution
        class_dist = {}
        for label in labels:
            if label is not None:
                label_str = str(label)
                class_dist[label_str] = class_dist.get(label_str, 0) + 1

        # Imbalance ratio
        if class_dist:
            max_class = max(class_dist.values())
            min_class = min(class_dist.values())
            imbalance = max_class / min_class if min_class > 0 else float('inf')
        else:
            imbalance = 0.0

        # Feature coverage
        feature_coverage = {}
        for col, values in data.items():
            if col != label_column:
                non_null = sum(1 for v in values if v is not None)
                feature_coverage[col] = non_null / len(values) if values else 0.0

        return TrainingDataMetrics(
            total_samples=total,
            class_distribution=class_dist,
            imbalance_ratio=imbalance,
            feature_coverage=feature_coverage,
            data_version=data_version,
        )

    def recommend_sampling_strategy(self, metrics: TrainingDataMetrics) -> str:
        """Recommend sampling strategy based on class imbalance."""
        if metrics.imbalance_ratio < 1.5:
            return "No resampling needed"
        elif metrics.imbalance_ratio < 3:
            return "Consider class weights"
        elif metrics.imbalance_ratio < 10:
            return "Use SMOTE or random oversampling"
        else:
            return "Use aggressive oversampling with augmentation"


# =============================================================================
# ANALYZERS - PERFORMANCE ON SUBSETS
# =============================================================================

class PerformanceSubsetAnalyzer:
    """Analyzer for model performance on data subsets."""

    def analyze_subset_performance(
        self,
        predictions: List[Any],
        actuals: List[Any],
        subset_mask: List[bool],
        subset_name: str,
        overall_metrics: Dict[str, float]
    ) -> PerformanceSubsetResult:
        """Analyze performance on a specific data subset."""
        # Get subset predictions and actuals
        subset_preds = [p for p, m in zip(predictions, subset_mask) if m]
        subset_actual = [a for a, m in zip(actuals, subset_mask) if m]

        if not subset_preds:
            return PerformanceSubsetResult(
                subset_name=subset_name,
                subset_criteria={},
                sample_count=0,
                metrics={},
            )

        # Calculate accuracy for subset
        correct = sum(1 for p, a in zip(subset_preds, subset_actual) if p == a)
        accuracy = correct / len(subset_preds)

        subset_metrics = {"accuracy": accuracy}

        # Compare to overall
        comparison = {}
        significant_diffs = []
        for metric, value in subset_metrics.items():
            if metric in overall_metrics:
                diff = value - overall_metrics[metric]
                comparison[metric] = diff
                if abs(diff) > 0.05:  # 5% threshold
                    significant_diffs.append(f"{metric}: {diff:+.2%}")

        return PerformanceSubsetResult(
            subset_name=subset_name,
            subset_criteria={"mask": "provided"},
            sample_count=len(subset_preds),
            metrics=subset_metrics,
            comparison_to_overall=comparison,
            significant_differences=significant_diffs,
        )


# =============================================================================
# ANALYZERS - FAITHFULNESS/HALLUCINATION
# =============================================================================

class FaithfulnessAnalyzer:
    """Analyzer for output faithfulness to source data."""

    def analyze_faithfulness(
        self,
        output: str,
        source_documents: List[str],
        claims: Optional[List[str]] = None
    ) -> FaithfulnessResult:
        """Analyze faithfulness of output to source documents."""
        if not source_documents:
            return FaithfulnessResult(
                source_coverage=0.0,
                factual_consistency=0.0,
                attribution_accuracy=0.0,
                hallucination_rate=1.0,
            )

        # Simple word overlap check (placeholder for more sophisticated methods)
        output_words = set(output.lower().split())
        source_words = set()
        for doc in source_documents:
            source_words.update(doc.lower().split())

        coverage = len(output_words & source_words) / len(output_words) if output_words else 0

        # Estimate hallucination rate (inverse of coverage for simplicity)
        hallucination_rate = 1 - coverage

        return FaithfulnessResult(
            source_coverage=coverage,
            factual_consistency=coverage,  # Simplified
            attribution_accuracy=coverage,  # Simplified
            hallucination_rate=hallucination_rate,
            unsupported_claims=[],
            grounding_evidence=[],
        )


# =============================================================================
# ANALYZERS - ROBUSTNESS/STRESS
# =============================================================================

class DataRobustnessAnalyzer:
    """Analyzer for data-level robustness testing."""

    def test_missing_value_robustness(
        self,
        model_predict_fn,
        data: Dict[str, List[Any]],
        missing_rates: List[float] = [0.05, 0.1, 0.2]
    ) -> Dict[str, Any]:
        """Test model robustness to missing values."""
        results = {"missing_rates": missing_rates, "performance_degradation": []}

        # Placeholder for actual implementation
        for rate in missing_rates:
            results["performance_degradation"].append({
                "rate": rate,
                "degradation": rate * 0.5,  # Placeholder
            })

        return results

    def test_noise_robustness(
        self,
        model_predict_fn,
        data: Dict[str, List[float]],
        noise_levels: List[float] = [0.01, 0.05, 0.1]
    ) -> Dict[str, Any]:
        """Test model robustness to noise in data."""
        results = {"noise_levels": noise_levels, "performance_degradation": []}

        for level in noise_levels:
            results["performance_degradation"].append({
                "level": level,
                "degradation": level * 0.3,  # Placeholder
            })

        return results


# =============================================================================
# ANALYZERS - DATA EXPLAINABILITY
# =============================================================================

class DataExplainabilityAnalyzer:
    """Analyzer for data-driven explanations."""

    def explain_prediction_by_data(
        self,
        input_data: Dict[str, Any],
        similar_training_samples: List[Dict[str, Any]],
        prediction: Any
    ) -> Dict[str, Any]:
        """Explain prediction using similar training examples."""
        return {
            "prediction": prediction,
            "explanation_type": "example_based",
            "num_similar_samples": len(similar_training_samples),
            "most_similar": similar_training_samples[:3] if similar_training_samples else [],
        }


# =============================================================================
# ANALYZERS - SECURITY/ACCESS
# =============================================================================

class DataAccessAnalyzer:
    """Analyzer for data access control and auditing."""

    def __init__(self):
        self.audit_log: List[AccessAuditRecord] = []

    def log_access(self, record: AccessAuditRecord) -> None:
        """Log a data access event."""
        self.audit_log.append(record)

    def analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze data access patterns."""
        if not self.audit_log:
            return {"total_accesses": 0}

        by_user = {}
        by_asset = {}
        by_level = {}
        denied_count = 0

        for record in self.audit_log:
            by_user[record.user_id] = by_user.get(record.user_id, 0) + 1
            by_asset[record.asset_id] = by_asset.get(record.asset_id, 0) + 1
            level_name = record.access_level.name
            by_level[level_name] = by_level.get(level_name, 0) + 1
            if not record.granted:
                denied_count += 1

        return {
            "total_accesses": len(self.audit_log),
            "unique_users": len(by_user),
            "unique_assets": len(by_asset),
            "access_by_level": by_level,
            "denied_accesses": denied_count,
            "denial_rate": denied_count / len(self.audit_log) if self.audit_log else 0,
        }

    def detect_anomalies(self, threshold_accesses_per_hour: int = 100) -> List[str]:
        """Detect anomalous access patterns."""
        anomalies = []

        # Group by user and hour
        user_hourly = {}
        for record in self.audit_log:
            key = (record.user_id, record.timestamp.strftime("%Y-%m-%d-%H"))
            user_hourly[key] = user_hourly.get(key, 0) + 1

        for (user, hour), count in user_hourly.items():
            if count > threshold_accesses_per_hour:
                anomalies.append(f"User {user} had {count} accesses in hour {hour}")

        return anomalies


# =============================================================================
# ANALYZERS - RETENTION/DELETION
# =============================================================================

class DataRetentionAnalyzer:
    """Analyzer for data retention compliance."""

    def __init__(self):
        self.retention_records: Dict[str, RetentionRecord] = {}

    def register_retention(self, record: RetentionRecord) -> None:
        """Register a data retention record."""
        self.retention_records[record.asset_id] = record

    def check_retention_compliance(self) -> Dict[str, Any]:
        """Check retention policy compliance."""
        now = datetime.now()
        overdue = []
        upcoming = []
        compliant = []

        for asset_id, record in self.retention_records.items():
            if record.status == RetentionStatus.ACTIVE:
                if record.expiry_date < now:
                    overdue.append(asset_id)
                elif (record.expiry_date - now).days < 30:
                    upcoming.append(asset_id)
                else:
                    compliant.append(asset_id)

        return {
            "total_records": len(self.retention_records),
            "overdue_deletion": overdue,
            "upcoming_expiry": upcoming,
            "compliant": len(compliant),
            "compliance_rate": len(compliant) / len(self.retention_records) if self.retention_records else 1.0,
        }

    def get_deletion_schedule(self) -> List[Dict[str, Any]]:
        """Get upcoming deletion schedule."""
        now = datetime.now()
        schedule = []

        for asset_id, record in self.retention_records.items():
            if record.status == RetentionStatus.ACTIVE:
                schedule.append({
                    "asset_id": asset_id,
                    "expiry_date": record.expiry_date.isoformat(),
                    "days_remaining": (record.expiry_date - now).days,
                    "legal_basis": record.legal_basis,
                })

        return sorted(schedule, key=lambda x: x["days_remaining"])


# =============================================================================
# ANALYZERS - INCIDENT/POST-MORTEM
# =============================================================================

class DataIncidentAnalyzer:
    """Analyzer for data-related incidents."""

    def __init__(self):
        self.incidents: List[DataIncident] = []

    def log_incident(self, incident: DataIncident) -> None:
        """Log a data incident."""
        self.incidents.append(incident)

    def analyze_incidents(self) -> Dict[str, Any]:
        """Analyze incident patterns."""
        if not self.incidents:
            return {"total_incidents": 0}

        by_severity = {}
        by_type = {}
        resolved = 0

        for incident in self.incidents:
            severity_name = incident.severity.name
            by_severity[severity_name] = by_severity.get(severity_name, 0) + 1
            by_type[incident.incident_type] = by_type.get(incident.incident_type, 0) + 1
            if incident.resolved_at:
                resolved += 1

        return {
            "total_incidents": len(self.incidents),
            "by_severity": by_severity,
            "by_type": by_type,
            "resolved_count": resolved,
            "resolution_rate": resolved / len(self.incidents) if self.incidents else 0,
        }

    def extract_lessons_learned(self) -> List[str]:
        """Extract lessons learned from all incidents."""
        lessons = []
        for incident in self.incidents:
            lessons.extend(incident.lessons_learned)
        return list(set(lessons))

    def get_prevention_measures(self) -> List[str]:
        """Get prevention measures from all incidents."""
        measures = []
        for incident in self.incidents:
            measures.extend(incident.prevention_measures)
        return list(set(measures))


# =============================================================================
# COMPREHENSIVE ANALYZER
# =============================================================================

class DataLifecycleAnalyzer:
    """Comprehensive data lifecycle analyzer."""

    def __init__(self):
        self.inventory_analyzer = DataInventoryAnalyzer()
        self.pii_analyzer = PIIDetectionAnalyzer()
        self.quality_analyzer = DataQualityAnalyzer()
        self.bias_analyzer = DataBiasAnalyzer()
        self.drift_analyzer = DataDriftAnalyzer()
        self.contract_analyzer = InputContractAnalyzer()
        self.training_analyzer = TrainingDataAnalyzer()
        self.access_analyzer = DataAccessAnalyzer()
        self.retention_analyzer = DataRetentionAnalyzer()
        self.incident_analyzer = DataIncidentAnalyzer()

    def comprehensive_assessment(
        self,
        data: Dict[str, List[Any]],
        assets: List[DataAsset],
        baseline_data: Optional[Dict[str, List[float]]] = None
    ) -> DataLifecycleAssessment:
        """Perform comprehensive data lifecycle assessment."""
        assessment_id = f"DLA-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Inventory analysis
        inventory_result = self.inventory_analyzer.analyze_inventory(assets)
        inventory_completeness = inventory_result.get("assets_with_schema", 0) / len(assets) if assets else 0

        # Sensitive data analysis
        pii_findings = self.pii_analyzer.detect_sensitive_data(data)
        sensitive_coverage = 1.0 - self.pii_analyzer.calculate_exposure_risk(pii_findings)

        # Quality analysis
        quality_metrics = self.quality_analyzer.analyze_all_dimensions(data)
        quality_score = sum(m.score for m in quality_metrics) / len(quality_metrics) if quality_metrics else 0

        # Bias analysis (if protected attribute detected)
        bias_risk = 0.0
        for col in data.keys():
            if any(term in col.lower() for term in ['gender', 'race', 'age', 'ethnicity']):
                bias_result = self.bias_analyzer.analyze_representation_bias(data, col)
                bias_risk = max(bias_risk, bias_result.statistical_parity_difference)

        # Drift analysis
        drift_count = 0
        if baseline_data:
            drift_results = self.drift_analyzer.detect_all_drift(baseline_data, data)
            drift_count = sum(1 for r in drift_results if r.severity in [DriftSeverity.SIGNIFICANT, DriftSeverity.CRITICAL])

        # Contract compliance
        contract_compliance = self.contract_analyzer.get_compliance_rate([])

        # Retention compliance
        retention_result = self.retention_analyzer.check_retention_compliance()
        retention_compliance = retention_result.get("compliance_rate", 1.0)

        # Security posture
        access_result = self.access_analyzer.analyze_access_patterns()
        security_score = 1.0 - access_result.get("denial_rate", 0)

        # Incident rate
        incident_result = self.incident_analyzer.analyze_incidents()
        incident_rate = incident_result.get("total_incidents", 0) / 100  # Per 100 operations

        # Overall health score
        overall = (
            inventory_completeness * 0.1 +
            sensitive_coverage * 0.15 +
            quality_score * 0.2 +
            (1 - bias_risk) * 0.15 +
            (1 - drift_count / 10) * 0.1 +
            contract_compliance * 0.1 +
            retention_compliance * 0.1 +
            security_score * 0.1
        )

        return DataLifecycleAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            inventory_completeness=inventory_completeness,
            sensitive_data_coverage=sensitive_coverage,
            quality_score=quality_score,
            bias_risk_score=bias_risk,
            drift_alert_count=drift_count,
            contract_compliance_rate=contract_compliance,
            retention_compliance_rate=retention_compliance,
            security_posture_score=security_score,
            incident_rate=incident_rate,
            overall_health_score=overall,
            recommendations=[
                "Improve data documentation" if inventory_completeness < 0.8 else None,
                "Review sensitive data handling" if sensitive_coverage < 0.9 else None,
                "Address data quality issues" if quality_score < 0.9 else None,
                "Investigate bias in data" if bias_risk > 0.1 else None,
            ],
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_data_inventory(assets: List[DataAsset]) -> Dict[str, Any]:
    """Analyze data asset inventory."""
    analyzer = DataInventoryAnalyzer()
    return analyzer.analyze_inventory(assets)


def detect_sensitive_data(data: Dict[str, List[Any]]) -> List[SensitiveDataFinding]:
    """Detect sensitive data in dataset."""
    analyzer = PIIDetectionAnalyzer()
    return analyzer.detect_sensitive_data(data)


def analyze_data_quality(data: Dict[str, List[Any]]) -> List[DataQualityMetrics]:
    """Analyze data quality across all dimensions."""
    analyzer = DataQualityAnalyzer()
    return analyzer.analyze_all_dimensions(data)


def detect_data_drift(
    baseline: Dict[str, List[float]],
    current: Dict[str, List[float]]
) -> List[DriftDetectionResult]:
    """Detect data drift between baseline and current."""
    analyzer = DataDriftAnalyzer()
    return analyzer.detect_all_drift(baseline, current)


def analyze_data_bias(
    data: Dict[str, List[Any]],
    protected_attribute: str
) -> BiasAnalysisResult:
    """Analyze bias in data for protected attribute."""
    analyzer = DataBiasAnalyzer()
    return analyzer.analyze_representation_bias(data, protected_attribute)


def validate_input_contract(
    contract: Dict[str, Dict[str, Any]],
    input_data: Dict[str, Any]
) -> List[InputContractViolation]:
    """Validate input data against contract."""
    analyzer = InputContractAnalyzer()
    analyzer.define_contract(contract)
    return analyzer.validate_input(input_data)


def analyze_training_data(
    data: Dict[str, List[Any]],
    label_column: str
) -> TrainingDataMetrics:
    """Analyze training data characteristics."""
    analyzer = TrainingDataAnalyzer()
    return analyzer.analyze_training_data(data, label_column)


def comprehensive_data_assessment(
    data: Dict[str, List[Any]],
    assets: List[DataAsset]
) -> DataLifecycleAssessment:
    """Perform comprehensive data lifecycle assessment."""
    analyzer = DataLifecycleAnalyzer()
    return analyzer.comprehensive_assessment(data, assets)
