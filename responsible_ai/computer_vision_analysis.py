"""
Computer Vision Analysis Module
===============================

Comprehensive computer vision analysis framework covering image quality,
noise analysis, spatial bias, task-specific metrics, and visual model evaluation.

Categories:
1. Image Data Quality Analysis - Resolution, format, color distribution
2. Noise Analysis - Gaussian, salt-pepper, motion blur, compression
3. Spatial Bias Analysis - Position bias, object size distribution
4. Classification Metrics - Accuracy, confusion matrix, per-class performance
5. Object Detection Metrics - mAP, IoU, precision-recall
6. Segmentation Metrics - Dice, IoU, boundary metrics
7. Image Generation Metrics - FID, IS, LPIPS
8. Augmentation Analysis - Augmentation effectiveness
9. Visual Robustness Analysis - Perturbation robustness
10. Attention/Saliency Analysis - Visual attention patterns
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math


# =============================================================================
# ENUMS
# =============================================================================

class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = auto()
    PNG = auto()
    BMP = auto()
    TIFF = auto()
    WEBP = auto()
    GIF = auto()
    UNKNOWN = auto()


class ColorSpace(Enum):
    """Image color spaces."""
    RGB = auto()
    BGR = auto()
    GRAYSCALE = auto()
    HSV = auto()
    LAB = auto()
    RGBA = auto()
    CMYK = auto()


class NoiseType(Enum):
    """Types of image noise."""
    GAUSSIAN = auto()
    SALT_PEPPER = auto()
    POISSON = auto()
    SPECKLE = auto()
    MOTION_BLUR = auto()
    DEFOCUS_BLUR = auto()
    COMPRESSION_ARTIFACT = auto()
    JPEG_ARTIFACT = auto()


class AugmentationType(Enum):
    """Types of image augmentation."""
    FLIP_HORIZONTAL = auto()
    FLIP_VERTICAL = auto()
    ROTATION = auto()
    SCALE = auto()
    TRANSLATE = auto()
    CROP = auto()
    COLOR_JITTER = auto()
    BRIGHTNESS = auto()
    CONTRAST = auto()
    SATURATION = auto()
    HUE = auto()
    GAUSSIAN_BLUR = auto()
    NOISE = auto()
    CUTOUT = auto()
    MIXUP = auto()
    CUTMIX = auto()
    MOSAIC = auto()
    RANDOM_ERASING = auto()


class CVTaskType(Enum):
    """Computer vision task types."""
    CLASSIFICATION = auto()
    OBJECT_DETECTION = auto()
    SEMANTIC_SEGMENTATION = auto()
    INSTANCE_SEGMENTATION = auto()
    PANOPTIC_SEGMENTATION = auto()
    POSE_ESTIMATION = auto()
    DEPTH_ESTIMATION = auto()
    IMAGE_GENERATION = auto()
    SUPER_RESOLUTION = auto()
    DENOISING = auto()
    INPAINTING = auto()


class QualityLevel(Enum):
    """Image quality levels."""
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    POOR = auto()
    UNUSABLE = auto()


class DetectionMetricType(Enum):
    """Object detection metric types."""
    AP = auto()  # Average Precision
    AP50 = auto()  # AP at IoU 0.5
    AP75 = auto()  # AP at IoU 0.75
    MAP = auto()  # Mean Average Precision
    AR = auto()  # Average Recall
    IOU = auto()


class SegmentationMetricType(Enum):
    """Segmentation metric types."""
    DICE = auto()
    IOU = auto()
    PIXEL_ACCURACY = auto()
    MEAN_IOU = auto()
    FREQUENCY_WEIGHTED_IOU = auto()
    BOUNDARY_F1 = auto()


class GenerationMetricType(Enum):
    """Image generation metric types."""
    FID = auto()  # Frechet Inception Distance
    IS = auto()  # Inception Score
    LPIPS = auto()  # Learned Perceptual Image Patch Similarity
    PSNR = auto()  # Peak Signal-to-Noise Ratio
    SSIM = auto()  # Structural Similarity Index
    KID = auto()  # Kernel Inception Distance


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ImageMetadata:
    """Metadata about an image."""
    image_id: str
    width: int
    height: int
    channels: int
    format: ImageFormat
    color_space: ColorSpace
    bit_depth: int = 8
    file_size_bytes: int = 0
    has_alpha: bool = False
    is_corrupted: bool = False


@dataclass
class ImageQualityMetrics:
    """Image quality assessment metrics."""
    sharpness: float
    brightness: float
    contrast: float
    saturation: float
    noise_level: float
    blur_score: float
    quality_level: QualityLevel
    issues_detected: List[str] = field(default_factory=list)


@dataclass
class NoiseAnalysisResult:
    """Result from noise analysis."""
    noise_type: NoiseType
    estimated_noise_level: float
    snr: float  # Signal-to-Noise Ratio
    psnr: float  # Peak Signal-to-Noise Ratio
    affected_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class SpatialBiasMetrics:
    """Metrics about spatial bias in images."""
    position_bias_score: float
    center_bias: float
    edge_bias: float
    size_distribution_uniformity: float
    aspect_ratio_distribution: Dict[str, float] = field(default_factory=dict)
    quadrant_distribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClassificationMetrics:
    """Classification task metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    top_k_accuracy: Dict[int, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class BoundingBox:
    """Bounding box representation."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int
    confidence: float = 1.0
    class_name: str = ""


@dataclass
class DetectionMetrics:
    """Object detection task metrics."""
    metric_type: DetectionMetricType
    value: float
    iou_threshold: float = 0.5
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    precision_recall_curve: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class SegmentationMetrics:
    """Segmentation task metrics."""
    metric_type: SegmentationMetricType
    value: float
    per_class_iou: Dict[str, float] = field(default_factory=dict)
    boundary_precision: float = 0.0
    boundary_recall: float = 0.0


@dataclass
class GenerationMetrics:
    """Image generation quality metrics."""
    metric_type: GenerationMetricType
    value: float
    num_samples: int = 0
    reference_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class AugmentationAnalysisResult:
    """Result from augmentation analysis."""
    augmentation_type: AugmentationType
    effectiveness_score: float
    diversity_improvement: float
    performance_impact: float
    recommended_strength: float = 0.5
    notes: str = ""


@dataclass
class VisualRobustnessResult:
    """Visual model robustness result."""
    perturbation_type: str
    severity_levels: List[float]
    accuracy_at_levels: List[float]
    mean_corruption_error: float
    robustness_score: float


@dataclass
class SaliencyAnalysisResult:
    """Saliency/attention analysis result."""
    image_id: str
    saliency_method: str
    top_regions: List[Tuple[int, int, int, int, float]] = field(default_factory=list)
    alignment_with_objects: float = 0.0
    focus_score: float = 0.0


@dataclass
class ComputerVisionAssessment:
    """Comprehensive computer vision assessment."""
    assessment_id: str
    timestamp: datetime
    task_type: CVTaskType
    data_quality_score: float
    spatial_balance_score: float
    noise_resilience_score: float
    augmentation_diversity_score: float
    model_performance_score: float
    robustness_score: float
    overall_score: float
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# ANALYZERS - IMAGE DATA QUALITY
# =============================================================================

class ImageQualityAnalyzer:
    """Analyzer for image data quality."""

    def analyze_image_quality(
        self,
        image_stats: Dict[str, Any]
    ) -> ImageQualityMetrics:
        """Analyze quality of a single image."""
        # Extract stats
        sharpness = image_stats.get("sharpness", 0.5)
        brightness = image_stats.get("brightness", 0.5)
        contrast = image_stats.get("contrast", 0.5)
        saturation = image_stats.get("saturation", 0.5)
        noise_level = image_stats.get("noise_level", 0.1)
        blur_score = image_stats.get("blur_score", 0.1)

        # Detect issues
        issues = []
        if brightness < 0.2:
            issues.append("Image is too dark")
        elif brightness > 0.8:
            issues.append("Image is overexposed")

        if contrast < 0.2:
            issues.append("Low contrast")

        if sharpness < 0.3:
            issues.append("Image is blurry")

        if noise_level > 0.3:
            issues.append("High noise level")

        # Determine quality level
        quality_score = (sharpness + (1 - noise_level) + contrast) / 3

        if quality_score > 0.8 and not issues:
            quality_level = QualityLevel.EXCELLENT
        elif quality_score > 0.6 and len(issues) <= 1:
            quality_level = QualityLevel.GOOD
        elif quality_score > 0.4:
            quality_level = QualityLevel.ACCEPTABLE
        elif quality_score > 0.2:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.UNUSABLE

        return ImageQualityMetrics(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            noise_level=noise_level,
            blur_score=blur_score,
            quality_level=quality_level,
            issues_detected=issues,
        )

    def analyze_dataset_quality(
        self,
        image_stats_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze quality across a dataset of images."""
        if not image_stats_list:
            return {"num_images": 0}

        metrics = [self.analyze_image_quality(stats) for stats in image_stats_list]

        quality_dist = {}
        for m in metrics:
            level_name = m.quality_level.name
            quality_dist[level_name] = quality_dist.get(level_name, 0) + 1

        all_issues = []
        for m in metrics:
            all_issues.extend(m.issues_detected)

        return {
            "num_images": len(metrics),
            "quality_distribution": quality_dist,
            "avg_sharpness": sum(m.sharpness for m in metrics) / len(metrics),
            "avg_brightness": sum(m.brightness for m in metrics) / len(metrics),
            "avg_contrast": sum(m.contrast for m in metrics) / len(metrics),
            "avg_noise_level": sum(m.noise_level for m in metrics) / len(metrics),
            "most_common_issues": self._get_top_issues(all_issues),
            "usable_rate": (
                sum(1 for m in metrics if m.quality_level != QualityLevel.UNUSABLE) / len(metrics)
            ),
        }

    def _get_top_issues(self, issues: List[str], top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common issues."""
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        return sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]


class ImageMetadataAnalyzer:
    """Analyzer for image metadata."""

    def analyze_metadata(
        self,
        images: List[ImageMetadata]
    ) -> Dict[str, Any]:
        """Analyze metadata across images."""
        if not images:
            return {}

        resolutions = [(img.width, img.height) for img in images]
        formats = [img.format.name for img in images]
        color_spaces = [img.color_space.name for img in images]

        unique_resolutions = list(set(resolutions))
        resolution_counts = {str(r): resolutions.count(r) for r in unique_resolutions[:10]}

        return {
            "num_images": len(images),
            "resolution_distribution": resolution_counts,
            "most_common_resolution": max(set(resolutions), key=resolutions.count),
            "format_distribution": {f: formats.count(f) for f in set(formats)},
            "color_space_distribution": {c: color_spaces.count(c) for c in set(color_spaces)},
            "corrupted_count": sum(1 for img in images if img.is_corrupted),
            "avg_file_size": sum(img.file_size_bytes for img in images) / len(images),
        }


# =============================================================================
# ANALYZERS - NOISE ANALYSIS
# =============================================================================

class NoiseAnalyzer:
    """Analyzer for image noise."""

    def analyze_noise(
        self,
        noise_estimate: float,
        image_stats: Dict[str, Any]
    ) -> NoiseAnalysisResult:
        """Analyze noise in an image."""
        # Estimate SNR
        signal_power = image_stats.get("signal_variance", 1.0)
        noise_power = noise_estimate ** 2 + 1e-10

        snr = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        # Estimate PSNR
        max_pixel = image_stats.get("max_pixel_value", 255)
        mse = noise_power
        psnr = 10 * math.log10((max_pixel ** 2) / mse) if mse > 0 else float('inf')

        # Determine noise type (simplified)
        if image_stats.get("has_salt_pepper", False):
            noise_type = NoiseType.SALT_PEPPER
        elif image_stats.get("has_motion_blur", False):
            noise_type = NoiseType.MOTION_BLUR
        elif image_stats.get("has_jpeg_artifacts", False):
            noise_type = NoiseType.JPEG_ARTIFACT
        else:
            noise_type = NoiseType.GAUSSIAN

        return NoiseAnalysisResult(
            noise_type=noise_type,
            estimated_noise_level=noise_estimate,
            snr=snr,
            psnr=psnr,
        )

    def analyze_noise_distribution(
        self,
        noise_results: List[NoiseAnalysisResult]
    ) -> Dict[str, Any]:
        """Analyze noise distribution across dataset."""
        if not noise_results:
            return {}

        noise_types = [r.noise_type.name for r in noise_results]
        noise_levels = [r.estimated_noise_level for r in noise_results]

        return {
            "num_images": len(noise_results),
            "noise_type_distribution": {t: noise_types.count(t) for t in set(noise_types)},
            "avg_noise_level": sum(noise_levels) / len(noise_levels),
            "max_noise_level": max(noise_levels),
            "min_noise_level": min(noise_levels),
            "avg_snr": sum(r.snr for r in noise_results if r.snr != float('inf')) / len(noise_results),
            "avg_psnr": sum(r.psnr for r in noise_results if r.psnr != float('inf')) / len(noise_results),
        }


# =============================================================================
# ANALYZERS - SPATIAL BIAS
# =============================================================================

class SpatialBiasAnalyzer:
    """Analyzer for spatial bias in image datasets."""

    def analyze_spatial_bias(
        self,
        bounding_boxes: List[BoundingBox],
        image_width: int,
        image_height: int
    ) -> SpatialBiasMetrics:
        """Analyze spatial distribution of objects."""
        if not bounding_boxes:
            return SpatialBiasMetrics(
                position_bias_score=0.0,
                center_bias=0.0,
                edge_bias=0.0,
                size_distribution_uniformity=0.0,
            )

        # Calculate center points
        centers = []
        sizes = []
        quadrants = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}

        for box in bounding_boxes:
            cx = (box.x_min + box.x_max) / 2
            cy = (box.y_min + box.y_max) / 2
            centers.append((cx, cy))

            w = box.x_max - box.x_min
            h = box.y_max - box.y_min
            sizes.append(w * h)

            # Assign to quadrant
            if cx < image_width / 2:
                if cy < image_height / 2:
                    quadrants["TL"] += 1
                else:
                    quadrants["BL"] += 1
            else:
                if cy < image_height / 2:
                    quadrants["TR"] += 1
                else:
                    quadrants["BR"] += 1

        # Calculate center bias
        center_x, center_y = image_width / 2, image_height / 2
        avg_distance_from_center = sum(
            ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
            for cx, cy in centers
        ) / len(centers)
        max_distance = ((image_width / 2) ** 2 + (image_height / 2) ** 2) ** 0.5
        center_bias = 1 - (avg_distance_from_center / max_distance)

        # Calculate quadrant uniformity
        total_objects = len(bounding_boxes)
        expected_per_quadrant = total_objects / 4
        quadrant_deviation = sum(
            abs(count - expected_per_quadrant)
            for count in quadrants.values()
        ) / (4 * expected_per_quadrant) if expected_per_quadrant > 0 else 0

        position_bias_score = quadrant_deviation

        # Size distribution uniformity
        if sizes:
            mean_size = sum(sizes) / len(sizes)
            size_std = (sum((s - mean_size) ** 2 for s in sizes) / len(sizes)) ** 0.5
            size_uniformity = 1 / (1 + size_std / mean_size) if mean_size > 0 else 0
        else:
            size_uniformity = 0

        return SpatialBiasMetrics(
            position_bias_score=position_bias_score,
            center_bias=center_bias,
            edge_bias=1 - center_bias,
            size_distribution_uniformity=size_uniformity,
            quadrant_distribution={k: v / total_objects for k, v in quadrants.items()} if total_objects > 0 else {},
        )


# =============================================================================
# ANALYZERS - CLASSIFICATION METRICS
# =============================================================================

class ClassificationMetricsAnalyzer:
    """Analyzer for image classification metrics."""

    def calculate_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> ClassificationMetrics:
        """Calculate classification metrics."""
        if not predictions or not labels:
            return ClassificationMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
            )

        # Accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(labels)

        # Confusion matrix
        confusion = [[0] * num_classes for _ in range(num_classes)]
        for p, l in zip(predictions, labels):
            if 0 <= p < num_classes and 0 <= l < num_classes:
                confusion[l][p] += 1

        # Per-class metrics
        per_class = {}
        precisions = []
        recalls = []

        for c in range(num_classes):
            tp = confusion[c][c]
            fp = sum(confusion[i][c] for i in range(num_classes)) - tp
            fn = sum(confusion[c]) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            class_name = class_names[c] if class_names and c < len(class_names) else str(c)
            per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1}

            precisions.append(precision)
            recalls.append(recall)

        # Macro averages
        macro_precision = sum(precisions) / len(precisions) if precisions else 0
        macro_recall = sum(recalls) / len(recalls) if recalls else 0
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=macro_precision,
            recall=macro_recall,
            f1_score=macro_f1,
            confusion_matrix=confusion,
            per_class_metrics=per_class,
        )

    def calculate_top_k_accuracy(
        self,
        predictions_probs: List[List[float]],
        labels: List[int],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[int, float]:
        """Calculate top-k accuracy."""
        results = {}

        for k in k_values:
            correct = 0
            for probs, label in zip(predictions_probs, labels):
                top_k_classes = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
                if label in top_k_classes:
                    correct += 1
            results[k] = correct / len(labels) if labels else 0

        return results


# =============================================================================
# ANALYZERS - OBJECT DETECTION METRICS
# =============================================================================

class DetectionMetricsAnalyzer:
    """Analyzer for object detection metrics."""

    def calculate_iou(
        self,
        box1: BoundingBox,
        box2: BoundingBox
    ) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1.x_min, box2.x_min)
        y1 = max(box1.y_min, box2.y_min)
        x2 = min(box1.x_max, box2.x_max)
        y2 = min(box1.y_max, box2.y_max)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
        area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def calculate_ap(
        self,
        predictions: List[BoundingBox],
        ground_truths: List[BoundingBox],
        iou_threshold: float = 0.5,
        class_id: Optional[int] = None
    ) -> DetectionMetrics:
        """Calculate Average Precision."""
        # Filter by class if specified
        if class_id is not None:
            predictions = [p for p in predictions if p.class_id == class_id]
            ground_truths = [g for g in ground_truths if g.class_id == class_id]

        if not ground_truths:
            return DetectionMetrics(
                metric_type=DetectionMetricType.AP,
                value=0.0,
                iou_threshold=iou_threshold,
            )

        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)

        # Match predictions to ground truths
        matched_gt = set()
        tp = []
        fp = []

        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(ground_truths):
                if i in matched_gt:
                    continue
                iou = self.calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched_gt.add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)

        # Calculate precision-recall curve
        tp_cumsum = []
        fp_cumsum = []
        current_tp = 0
        current_fp = 0

        for t, f in zip(tp, fp):
            current_tp += t
            current_fp += f
            tp_cumsum.append(current_tp)
            fp_cumsum.append(current_fp)

        precisions = []
        recalls = []
        pr_curve = []

        for i in range(len(tp_cumsum)):
            precision = tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i])
            recall = tp_cumsum[i] / len(ground_truths)
            precisions.append(precision)
            recalls.append(recall)
            pr_curve.append((recall, precision))

        # Calculate AP (area under PR curve)
        ap = 0.0
        for i in range(len(recalls)):
            if i == 0:
                ap += recalls[i] * precisions[i]
            else:
                ap += (recalls[i] - recalls[i-1]) * precisions[i]

        return DetectionMetrics(
            metric_type=DetectionMetricType.AP,
            value=ap,
            iou_threshold=iou_threshold,
            precision_recall_curve=pr_curve,
        )

    def calculate_map(
        self,
        predictions: List[BoundingBox],
        ground_truths: List[BoundingBox],
        iou_threshold: float = 0.5
    ) -> DetectionMetrics:
        """Calculate Mean Average Precision across all classes."""
        # Get unique classes
        classes = set(g.class_id for g in ground_truths)

        per_class_ap = {}
        total_ap = 0

        for class_id in classes:
            ap_result = self.calculate_ap(predictions, ground_truths, iou_threshold, class_id)
            per_class_ap[str(class_id)] = ap_result.value
            total_ap += ap_result.value

        map_value = total_ap / len(classes) if classes else 0

        return DetectionMetrics(
            metric_type=DetectionMetricType.MAP,
            value=map_value,
            iou_threshold=iou_threshold,
            per_class_ap=per_class_ap,
        )


# =============================================================================
# ANALYZERS - SEGMENTATION METRICS
# =============================================================================

class SegmentationMetricsAnalyzer:
    """Analyzer for segmentation metrics."""

    def calculate_dice(
        self,
        prediction_mask: List[List[int]],
        ground_truth_mask: List[List[int]]
    ) -> float:
        """Calculate Dice coefficient."""
        intersection = 0
        pred_sum = 0
        gt_sum = 0

        for pred_row, gt_row in zip(prediction_mask, ground_truth_mask):
            for p, g in zip(pred_row, gt_row):
                if p == g == 1:
                    intersection += 1
                if p == 1:
                    pred_sum += 1
                if g == 1:
                    gt_sum += 1

        return (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0

    def calculate_iou(
        self,
        prediction_mask: List[List[int]],
        ground_truth_mask: List[List[int]]
    ) -> float:
        """Calculate Intersection over Union for segmentation."""
        intersection = 0
        union = 0

        for pred_row, gt_row in zip(prediction_mask, ground_truth_mask):
            for p, g in zip(pred_row, gt_row):
                if p == 1 or g == 1:
                    union += 1
                if p == g == 1:
                    intersection += 1

        return intersection / union if union > 0 else 0

    def calculate_pixel_accuracy(
        self,
        prediction_mask: List[List[int]],
        ground_truth_mask: List[List[int]]
    ) -> float:
        """Calculate pixel-wise accuracy."""
        correct = 0
        total = 0

        for pred_row, gt_row in zip(prediction_mask, ground_truth_mask):
            for p, g in zip(pred_row, gt_row):
                total += 1
                if p == g:
                    correct += 1

        return correct / total if total > 0 else 0

    def calculate_mean_iou(
        self,
        prediction_masks: List[List[List[int]]],
        ground_truth_masks: List[List[List[int]]],
        num_classes: int
    ) -> SegmentationMetrics:
        """Calculate mean IoU across all classes."""
        per_class_iou = {}
        total_iou = 0
        valid_classes = 0

        for class_id in range(num_classes):
            class_intersection = 0
            class_union = 0

            for pred_mask, gt_mask in zip(prediction_masks, ground_truth_masks):
                for pred_row, gt_row in zip(pred_mask, gt_mask):
                    for p, g in zip(pred_row, gt_row):
                        pred_is_class = (p == class_id)
                        gt_is_class = (g == class_id)

                        if pred_is_class or gt_is_class:
                            class_union += 1
                        if pred_is_class and gt_is_class:
                            class_intersection += 1

            if class_union > 0:
                class_iou = class_intersection / class_union
                per_class_iou[str(class_id)] = class_iou
                total_iou += class_iou
                valid_classes += 1

        mean_iou = total_iou / valid_classes if valid_classes > 0 else 0

        return SegmentationMetrics(
            metric_type=SegmentationMetricType.MEAN_IOU,
            value=mean_iou,
            per_class_iou=per_class_iou,
        )


# =============================================================================
# ANALYZERS - GENERATION METRICS
# =============================================================================

class GenerationMetricsAnalyzer:
    """Analyzer for image generation quality metrics."""

    def calculate_psnr(
        self,
        generated: List[float],
        reference: List[float],
        max_value: float = 255.0
    ) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        if len(generated) != len(reference):
            return 0.0

        mse = sum((g - r) ** 2 for g, r in zip(generated, reference)) / len(generated)

        if mse == 0:
            return float('inf')

        return 10 * math.log10((max_value ** 2) / mse)

    def calculate_ssim(
        self,
        generated_stats: Dict[str, float],
        reference_stats: Dict[str, float]
    ) -> float:
        """Calculate Structural Similarity Index (simplified)."""
        # Simplified SSIM using pre-computed stats
        mu_g = generated_stats.get("mean", 0)
        mu_r = reference_stats.get("mean", 0)
        sigma_g = generated_stats.get("std", 1)
        sigma_r = reference_stats.get("std", 1)
        covar = generated_stats.get("covariance_with_ref", 0)

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        numerator = (2 * mu_g * mu_r + c1) * (2 * covar + c2)
        denominator = (mu_g ** 2 + mu_r ** 2 + c1) * (sigma_g ** 2 + sigma_r ** 2 + c2)

        return numerator / denominator if denominator > 0 else 0

    def analyze_generation_quality(
        self,
        fid_score: float,
        inception_score: float,
        lpips_score: float
    ) -> Dict[str, GenerationMetrics]:
        """Analyze overall generation quality."""
        return {
            "fid": GenerationMetrics(
                metric_type=GenerationMetricType.FID,
                value=fid_score,
            ),
            "inception_score": GenerationMetrics(
                metric_type=GenerationMetricType.IS,
                value=inception_score,
            ),
            "lpips": GenerationMetrics(
                metric_type=GenerationMetricType.LPIPS,
                value=lpips_score,
            ),
        }


# =============================================================================
# ANALYZERS - AUGMENTATION
# =============================================================================

class AugmentationAnalyzer:
    """Analyzer for data augmentation effectiveness."""

    def analyze_augmentation(
        self,
        augmentation_type: AugmentationType,
        baseline_accuracy: float,
        augmented_accuracy: float,
        diversity_metrics: Dict[str, float]
    ) -> AugmentationAnalysisResult:
        """Analyze effectiveness of an augmentation technique."""
        performance_impact = augmented_accuracy - baseline_accuracy
        diversity_improvement = diversity_metrics.get("diversity_gain", 0)

        # Effectiveness score
        effectiveness = 0.5 * max(0, performance_impact / 0.1) + 0.5 * diversity_improvement
        effectiveness = min(effectiveness, 1.0)

        # Generate notes
        notes = []
        if performance_impact > 0.02:
            notes.append("Significant performance improvement")
        elif performance_impact < -0.02:
            notes.append("Performance degradation - reduce strength")

        if diversity_improvement > 0.1:
            notes.append("Good diversity improvement")

        return AugmentationAnalysisResult(
            augmentation_type=augmentation_type,
            effectiveness_score=effectiveness,
            diversity_improvement=diversity_improvement,
            performance_impact=performance_impact,
            notes="; ".join(notes) if notes else "Neutral impact",
        )

    def compare_augmentations(
        self,
        results: Dict[str, AugmentationAnalysisResult]
    ) -> Dict[str, Any]:
        """Compare different augmentation techniques."""
        if not results:
            return {}

        ranked = sorted(
            results.items(),
            key=lambda x: x[1].effectiveness_score,
            reverse=True
        )

        return {
            "ranking": [name for name, _ in ranked],
            "best_augmentation": ranked[0][0] if ranked else None,
            "recommended_combo": [name for name, r in ranked if r.performance_impact > 0][:3],
        }


# =============================================================================
# ANALYZERS - VISUAL ROBUSTNESS
# =============================================================================

class VisualRobustnessAnalyzer:
    """Analyzer for visual model robustness."""

    def analyze_corruption_robustness(
        self,
        corruption_type: str,
        severity_levels: List[int],
        accuracy_at_levels: List[float],
        clean_accuracy: float
    ) -> VisualRobustnessResult:
        """Analyze robustness to image corruptions."""
        # Calculate mean corruption error
        mce = sum(clean_accuracy - acc for acc in accuracy_at_levels) / len(accuracy_at_levels)

        # Robustness score (higher is better)
        robustness_score = sum(accuracy_at_levels) / (len(accuracy_at_levels) * clean_accuracy) if clean_accuracy > 0 else 0

        return VisualRobustnessResult(
            perturbation_type=corruption_type,
            severity_levels=[float(s) for s in severity_levels],
            accuracy_at_levels=accuracy_at_levels,
            mean_corruption_error=mce,
            robustness_score=robustness_score,
        )

    def aggregate_robustness(
        self,
        results: List[VisualRobustnessResult]
    ) -> Dict[str, Any]:
        """Aggregate robustness across all corruption types."""
        if not results:
            return {}

        avg_mce = sum(r.mean_corruption_error for r in results) / len(results)
        avg_robustness = sum(r.robustness_score for r in results) / len(results)

        # Find weakest corruptions
        sorted_by_mce = sorted(results, key=lambda x: x.mean_corruption_error, reverse=True)

        return {
            "num_corruption_types": len(results),
            "average_mce": avg_mce,
            "average_robustness_score": avg_robustness,
            "weakest_corruptions": [r.perturbation_type for r in sorted_by_mce[:3]],
            "strongest_corruptions": [r.perturbation_type for r in sorted_by_mce[-3:]],
        }


# =============================================================================
# ANALYZERS - SALIENCY
# =============================================================================

class SaliencyAnalyzer:
    """Analyzer for visual attention/saliency patterns."""

    def analyze_saliency(
        self,
        image_id: str,
        saliency_map: List[List[float]],
        ground_truth_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        method: str = "gradcam"
    ) -> SaliencyAnalysisResult:
        """Analyze saliency map."""
        if not saliency_map or not saliency_map[0]:
            return SaliencyAnalysisResult(
                image_id=image_id,
                saliency_method=method,
            )

        # Find top salient regions
        height = len(saliency_map)
        width = len(saliency_map[0])

        # Find max saliency points
        max_saliency = 0
        for row in saliency_map:
            for val in row:
                max_saliency = max(max_saliency, val)

        threshold = max_saliency * 0.7
        top_regions = []

        # Simple region detection
        for y in range(height):
            for x in range(width):
                if saliency_map[y][x] >= threshold:
                    # Create small region around high saliency point
                    region = (
                        max(0, x - 10),
                        max(0, y - 10),
                        min(width, x + 10),
                        min(height, y + 10),
                        saliency_map[y][x]
                    )
                    top_regions.append(region)

        # Limit to top 10 regions
        top_regions = sorted(top_regions, key=lambda x: x[4], reverse=True)[:10]

        # Calculate alignment with ground truth
        alignment = 0.0
        if ground_truth_regions and top_regions:
            aligned = 0
            for sr in top_regions:
                for gt in ground_truth_regions:
                    # Check overlap
                    overlap_x = max(0, min(sr[2], gt[2]) - max(sr[0], gt[0]))
                    overlap_y = max(0, min(sr[3], gt[3]) - max(sr[1], gt[1]))
                    if overlap_x > 0 and overlap_y > 0:
                        aligned += 1
                        break
            alignment = aligned / len(top_regions)

        # Focus score (how concentrated the saliency is)
        total_saliency = sum(sum(row) for row in saliency_map)
        high_saliency = sum(1 for row in saliency_map for val in row if val >= threshold)
        focus_score = 1 - (high_saliency / (height * width)) if total_saliency > 0 else 0

        return SaliencyAnalysisResult(
            image_id=image_id,
            saliency_method=method,
            top_regions=top_regions,
            alignment_with_objects=alignment,
            focus_score=focus_score,
        )


# =============================================================================
# COMPREHENSIVE ANALYZER
# =============================================================================

class ComputerVisionAnalyzer:
    """Comprehensive computer vision analyzer."""

    def __init__(self):
        self.quality_analyzer = ImageQualityAnalyzer()
        self.noise_analyzer = NoiseAnalyzer()
        self.spatial_analyzer = SpatialBiasAnalyzer()
        self.classification_analyzer = ClassificationMetricsAnalyzer()
        self.detection_analyzer = DetectionMetricsAnalyzer()
        self.segmentation_analyzer = SegmentationMetricsAnalyzer()
        self.generation_analyzer = GenerationMetricsAnalyzer()
        self.augmentation_analyzer = AugmentationAnalyzer()
        self.robustness_analyzer = VisualRobustnessAnalyzer()
        self.saliency_analyzer = SaliencyAnalyzer()

    def comprehensive_assessment(
        self,
        task_type: CVTaskType,
        image_stats: List[Dict[str, Any]],
        model_metrics: Dict[str, float],
        robustness_results: Optional[List[VisualRobustnessResult]] = None
    ) -> ComputerVisionAssessment:
        """Perform comprehensive computer vision assessment."""
        assessment_id = f"CVA-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Data quality
        quality_result = self.quality_analyzer.analyze_dataset_quality(image_stats)
        data_quality_score = quality_result.get("usable_rate", 0.5)

        # Spatial balance (placeholder)
        spatial_balance_score = 0.7

        # Noise resilience (placeholder)
        noise_resilience_score = 0.8

        # Augmentation diversity (placeholder)
        augmentation_diversity_score = 0.7

        # Model performance
        if task_type == CVTaskType.CLASSIFICATION:
            performance_score = model_metrics.get("accuracy", 0.5)
        elif task_type == CVTaskType.OBJECT_DETECTION:
            performance_score = model_metrics.get("map", 0.5)
        elif task_type in [CVTaskType.SEMANTIC_SEGMENTATION, CVTaskType.INSTANCE_SEGMENTATION]:
            performance_score = model_metrics.get("mean_iou", 0.5)
        else:
            performance_score = sum(model_metrics.values()) / len(model_metrics) if model_metrics else 0.5

        # Robustness
        if robustness_results:
            robustness_agg = self.robustness_analyzer.aggregate_robustness(robustness_results)
            robustness_score = robustness_agg.get("average_robustness_score", 0.5)
        else:
            robustness_score = 0.5

        # Overall score
        overall = (
            data_quality_score * 0.15 +
            spatial_balance_score * 0.1 +
            noise_resilience_score * 0.1 +
            augmentation_diversity_score * 0.1 +
            performance_score * 0.35 +
            robustness_score * 0.2
        )

        recommendations = []
        if data_quality_score < 0.8:
            recommendations.append("Improve image data quality - filter low quality images")
        if performance_score < 0.7:
            recommendations.append("Consider model architecture improvements or hyperparameter tuning")
        if robustness_score < 0.6:
            recommendations.append("Add robustness training with augmentations or adversarial training")

        return ComputerVisionAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            task_type=task_type,
            data_quality_score=data_quality_score,
            spatial_balance_score=spatial_balance_score,
            noise_resilience_score=noise_resilience_score,
            augmentation_diversity_score=augmentation_diversity_score,
            model_performance_score=performance_score,
            robustness_score=robustness_score,
            overall_score=overall,
            recommendations=recommendations,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_image_quality(image_stats: Dict[str, Any]) -> ImageQualityMetrics:
    """Analyze image quality."""
    analyzer = ImageQualityAnalyzer()
    return analyzer.analyze_image_quality(image_stats)


def analyze_noise(noise_estimate: float, image_stats: Dict[str, Any]) -> NoiseAnalysisResult:
    """Analyze image noise."""
    analyzer = NoiseAnalyzer()
    return analyzer.analyze_noise(noise_estimate, image_stats)


def analyze_spatial_bias(
    bounding_boxes: List[BoundingBox],
    image_width: int,
    image_height: int
) -> SpatialBiasMetrics:
    """Analyze spatial bias in object distribution."""
    analyzer = SpatialBiasAnalyzer()
    return analyzer.analyze_spatial_bias(bounding_boxes, image_width, image_height)


def calculate_classification_metrics(
    predictions: List[int],
    labels: List[int],
    num_classes: int
) -> ClassificationMetrics:
    """Calculate classification metrics."""
    analyzer = ClassificationMetricsAnalyzer()
    return analyzer.calculate_metrics(predictions, labels, num_classes)


def calculate_detection_map(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.5
) -> DetectionMetrics:
    """Calculate mean Average Precision for detection."""
    analyzer = DetectionMetricsAnalyzer()
    return analyzer.calculate_map(predictions, ground_truths, iou_threshold)


def calculate_segmentation_metrics(
    prediction_masks: List[List[List[int]]],
    ground_truth_masks: List[List[List[int]]],
    num_classes: int
) -> SegmentationMetrics:
    """Calculate segmentation metrics."""
    analyzer = SegmentationMetricsAnalyzer()
    return analyzer.calculate_mean_iou(prediction_masks, ground_truth_masks, num_classes)


def analyze_augmentation_effectiveness(
    augmentation_type: AugmentationType,
    baseline_accuracy: float,
    augmented_accuracy: float
) -> AugmentationAnalysisResult:
    """Analyze augmentation effectiveness."""
    analyzer = AugmentationAnalyzer()
    return analyzer.analyze_augmentation(
        augmentation_type, baseline_accuracy, augmented_accuracy, {"diversity_gain": 0.1}
    )


def comprehensive_cv_assessment(
    task_type: CVTaskType,
    image_stats: List[Dict[str, Any]],
    model_metrics: Dict[str, float]
) -> ComputerVisionAssessment:
    """Perform comprehensive CV assessment."""
    analyzer = ComputerVisionAnalyzer()
    return analyzer.comprehensive_assessment(task_type, image_stats, model_metrics)
