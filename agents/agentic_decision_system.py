"""
Agentic Decision System for Neurological Disease Detection
===========================================================
Implements the 3-tier architecture from the research paper:
1. Multi-modal Data Preprocessor
2. Agentic Decision System (intelligent model routing)
3. Specialized Analysis Modules

Architecture:
    Input Data -> Preprocessor -> Decision System -> Specialized Model -> Prediction

Author: Research Team
Project: Neurological Disease Detection using Agentic AI
"""

import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of input data"""
    EEG = "eeg"
    MRI = "mri"
    FMRI = "fmri"
    VOICE = "voice"
    GAIT = "gait"
    UNKNOWN = "unknown"


class DiseaseType(Enum):
    """Supported disease types"""
    ALZHEIMER = "alzheimer"
    PARKINSON = "parkinson"
    SCHIZOPHRENIA = "schizophrenia"
    EPILEPSY = "epilepsy"
    AUTISM = "autism"
    STRESS = "stress"
    DEPRESSION = "depression"
    HEALTHY = "healthy"


@dataclass
class DataCharacteristics:
    """Characteristics extracted from input data"""
    data_type: DataType
    num_channels: int
    num_samples: int
    sampling_rate: Optional[float] = None
    signal_quality: float = 0.0  # 0-1 score
    snr: float = 0.0  # Signal-to-noise ratio
    frequency_range: Tuple[float, float] = (0.0, 0.0)
    has_artifacts: bool = False
    artifact_ratio: float = 0.0
    file_format: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    """Result from the agentic decision system"""
    recommended_model: str
    recommended_disease_targets: List[DiseaseType]
    confidence: float
    reasoning: str
    data_characteristics: DataCharacteristics
    preprocessing_applied: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EEGPreprocessor:
    """
    Multi-modal Data Preprocessor for EEG signals

    Handles various EEG data formats and applies preprocessing:
    - Bandpass filtering
    - Artifact removal
    - Normalization
    - Resampling
    """

    def __init__(self, target_sampling_rate: float = 256.0,
                 target_channels: int = 22,
                 target_samples: int = 1000):
        self.target_sampling_rate = target_sampling_rate
        self.target_channels = target_channels
        self.target_samples = target_samples
        self.preprocessing_steps = []

    def preprocess(self, data: np.ndarray,
                   sampling_rate: Optional[float] = None,
                   channel_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess EEG data

        Parameters
        ----------
        data : np.ndarray
            Raw EEG data (samples, channels) or (channels, samples)
        sampling_rate : float, optional
            Original sampling rate
        channel_names : list, optional
            Names of EEG channels

        Returns
        -------
        processed_data : np.ndarray
            Preprocessed EEG data (samples, channels)
        steps : list
            List of preprocessing steps applied
        """
        self.preprocessing_steps = []

        # Ensure correct shape (samples, channels)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.shape[0] < data.shape[1]:
            data = data.T
            self.preprocessing_steps.append("transposed_data")

        # Remove NaN and Inf values
        if np.any(~np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            self.preprocessing_steps.append("removed_nan_inf")

        # Resample if needed
        if sampling_rate and sampling_rate != self.target_sampling_rate:
            data = self._resample(data, sampling_rate, self.target_sampling_rate)
            self.preprocessing_steps.append(f"resampled_{sampling_rate}Hz_to_{self.target_sampling_rate}Hz")

        # Adjust number of samples
        if data.shape[0] != self.target_samples:
            data = self._adjust_samples(data, self.target_samples)
            self.preprocessing_steps.append(f"adjusted_samples_to_{self.target_samples}")

        # Adjust number of channels
        if data.shape[1] != self.target_channels:
            data = self._adjust_channels(data, self.target_channels)
            self.preprocessing_steps.append(f"adjusted_channels_to_{self.target_channels}")

        # Bandpass filter (0.5-50 Hz for EEG)
        data = self._bandpass_filter(data)
        self.preprocessing_steps.append("bandpass_filter_0.5-50Hz")

        # Artifact removal (simple threshold-based)
        data = self._remove_artifacts(data)
        self.preprocessing_steps.append("artifact_removal")

        # Normalize
        data = self._normalize(data)
        self.preprocessing_steps.append("z_score_normalization")

        return data, self.preprocessing_steps

    def _resample(self, data: np.ndarray, orig_rate: float,
                  target_rate: float) -> np.ndarray:
        """Resample data to target rate"""
        orig_samples = data.shape[0]
        target_samples = int(orig_samples * target_rate / orig_rate)

        # Linear interpolation
        x_orig = np.linspace(0, 1, orig_samples)
        x_target = np.linspace(0, 1, target_samples)

        resampled = np.zeros((target_samples, data.shape[1]))
        for ch in range(data.shape[1]):
            resampled[:, ch] = np.interp(x_target, x_orig, data[:, ch])

        return resampled

    def _adjust_samples(self, data: np.ndarray, target_samples: int) -> np.ndarray:
        """Adjust number of samples"""
        current_samples = data.shape[0]

        if current_samples > target_samples:
            # Truncate from center
            start = (current_samples - target_samples) // 2
            return data[start:start + target_samples]
        elif current_samples < target_samples:
            # Pad with zeros
            pad_width = ((0, target_samples - current_samples), (0, 0))
            return np.pad(data, pad_width, mode='constant', constant_values=0)
        return data

    def _adjust_channels(self, data: np.ndarray, target_channels: int) -> np.ndarray:
        """Adjust number of channels"""
        current_channels = data.shape[1]

        if current_channels > target_channels:
            # Select subset of channels (first N channels)
            return data[:, :target_channels]
        elif current_channels < target_channels:
            # Pad with zeros
            pad_width = ((0, 0), (0, target_channels - current_channels))
            return np.pad(data, pad_width, mode='constant', constant_values=0)
        return data

    def _bandpass_filter(self, data: np.ndarray,
                         low_freq: float = 0.5,
                         high_freq: float = 50.0) -> np.ndarray:
        """Apply bandpass filter using FFT"""
        # Simple FFT-based filtering
        n_samples = data.shape[0]
        freqs = np.fft.fftfreq(n_samples, d=1.0/self.target_sampling_rate)

        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            fft_data = np.fft.fft(data[:, ch])
            # Create bandpass mask
            mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
            fft_data[~mask] = 0
            filtered[:, ch] = np.real(np.fft.ifft(fft_data))

        return filtered

    def _remove_artifacts(self, data: np.ndarray,
                          threshold: float = 3.0) -> np.ndarray:
        """Remove artifacts using threshold-based method"""
        # Z-score based artifact detection
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True) + 1e-8
        z_scores = np.abs((data - mean) / std)

        # Replace artifacts with interpolated values
        artifact_mask = z_scores > threshold
        data_cleaned = data.copy()

        for ch in range(data.shape[1]):
            ch_mask = artifact_mask[:, ch]
            if np.any(ch_mask):
                # Linear interpolation for artifact regions
                good_indices = np.where(~ch_mask)[0]
                bad_indices = np.where(ch_mask)[0]
                if len(good_indices) > 1:
                    data_cleaned[bad_indices, ch] = np.interp(
                        bad_indices, good_indices, data[good_indices, ch]
                    )

        return data_cleaned

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization per channel"""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True) + 1e-8
        return (data - mean) / std


class DataAnalyzer:
    """
    Analyzes input data to extract characteristics for decision making
    """

    def analyze(self, data: np.ndarray,
                file_path: Optional[str] = None,
                metadata: Optional[Dict] = None) -> DataCharacteristics:
        """
        Analyze input data and extract characteristics

        Parameters
        ----------
        data : np.ndarray
            Input data array
        file_path : str, optional
            Path to original file
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        characteristics : DataCharacteristics
            Extracted data characteristics
        """
        metadata = metadata or {}

        # Determine data type
        data_type = self._detect_data_type(data, file_path, metadata)

        # Extract dimensions
        if data.ndim == 1:
            num_samples = len(data)
            num_channels = 1
        elif data.ndim == 2:
            # Assume (samples, channels) or determine by shape
            if data.shape[0] > data.shape[1]:
                num_samples, num_channels = data.shape
            else:
                num_channels, num_samples = data.shape
        elif data.ndim == 3:
            # Could be (samples, channels, trials) or 3D image
            if data_type == DataType.MRI:
                num_samples = data.shape[0] * data.shape[1] * data.shape[2]
                num_channels = 1
            else:
                num_samples = data.shape[0]
                num_channels = data.shape[1]
        else:
            num_samples = data.shape[0]
            num_channels = np.prod(data.shape[1:])

        # Calculate signal quality metrics
        signal_quality = self._calculate_signal_quality(data)
        snr = self._calculate_snr(data)

        # Detect artifacts
        has_artifacts, artifact_ratio = self._detect_artifacts(data)

        # Estimate frequency range (for time series data)
        freq_range = self._estimate_frequency_range(data, metadata.get('sampling_rate'))

        # Determine file format
        file_format = self._detect_file_format(file_path) if file_path else "array"

        return DataCharacteristics(
            data_type=data_type,
            num_channels=num_channels,
            num_samples=num_samples,
            sampling_rate=metadata.get('sampling_rate'),
            signal_quality=signal_quality,
            snr=snr,
            frequency_range=freq_range,
            has_artifacts=has_artifacts,
            artifact_ratio=artifact_ratio,
            file_format=file_format,
            metadata=metadata
        )

    def _detect_data_type(self, data: np.ndarray,
                          file_path: Optional[str],
                          metadata: Dict) -> DataType:
        """Detect the type of input data"""
        # Check metadata first
        if 'data_type' in metadata:
            try:
                return DataType(metadata['data_type'].lower())
            except ValueError:
                pass

        # Check file extension
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.edf', '.bdf', '.gdf', '.set']:
                return DataType.EEG
            elif ext in ['.nii', '.nii.gz', '.mgz']:
                return DataType.MRI
            elif ext in ['.wav', '.mp3', '.flac']:
                return DataType.VOICE

        # Heuristics based on data shape
        if data.ndim == 3 and data.shape[0] > 20 and data.shape[1] > 20:
            return DataType.MRI
        elif data.ndim == 2:
            # EEG typically has many samples and fewer channels
            if data.shape[0] > 100 and 1 <= data.shape[1] <= 256:
                return DataType.EEG
            elif data.shape[1] > 100 and 1 <= data.shape[0] <= 256:
                return DataType.EEG

        return DataType.EEG  # Default to EEG for this system

    def _calculate_signal_quality(self, data: np.ndarray) -> float:
        """Calculate overall signal quality score (0-1)"""
        if data.size == 0:
            return 0.0

        # Factors affecting quality
        scores = []

        # 1. Check for NaN/Inf
        nan_ratio = np.sum(~np.isfinite(data)) / data.size
        scores.append(1.0 - nan_ratio)

        # 2. Check variance (flat signals are low quality)
        variance = np.var(data)
        if variance > 0:
            scores.append(min(1.0, np.log10(variance + 1) / 2))
        else:
            scores.append(0.0)

        # 3. Check dynamic range
        data_range = np.ptp(data[np.isfinite(data)]) if np.any(np.isfinite(data)) else 0
        if data_range > 0:
            scores.append(min(1.0, data_range / 1000))
        else:
            scores.append(0.0)

        return np.mean(scores)

    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio estimate"""
        if data.size == 0 or np.std(data) == 0:
            return 0.0

        # Simple SNR estimate using signal variance
        signal_power = np.var(data)

        # Estimate noise as high-frequency component
        if data.ndim >= 2 and data.shape[0] > 10:
            noise = np.diff(data, axis=0)
            noise_power = np.var(noise) / 2
        else:
            noise_power = signal_power * 0.1  # Assume 10% noise

        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            return max(0, min(50, snr_db))  # Clamp to 0-50 dB
        return 0.0

    def _detect_artifacts(self, data: np.ndarray,
                          threshold: float = 4.0) -> Tuple[bool, float]:
        """Detect artifacts in the data"""
        if data.size == 0:
            return False, 0.0

        # Z-score based detection
        finite_data = data[np.isfinite(data)]
        if len(finite_data) == 0:
            return True, 1.0

        mean = np.mean(finite_data)
        std = np.std(finite_data) + 1e-8
        z_scores = np.abs((data - mean) / std)

        artifact_mask = z_scores > threshold
        artifact_ratio = np.sum(artifact_mask) / data.size

        return artifact_ratio > 0.01, artifact_ratio

    def _estimate_frequency_range(self, data: np.ndarray,
                                   sampling_rate: Optional[float] = None) -> Tuple[float, float]:
        """Estimate dominant frequency range"""
        if data.size == 0 or sampling_rate is None:
            return (0.0, 0.0)

        # Flatten if multi-channel
        if data.ndim > 1:
            data = data.flatten()

        # FFT analysis
        n = len(data)
        freqs = np.fft.fftfreq(n, d=1.0/sampling_rate)
        fft_vals = np.abs(np.fft.fft(data))

        # Consider only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]

        # Find dominant frequency range (where most power is)
        if len(fft_vals) > 0:
            cumsum = np.cumsum(fft_vals)
            total = cumsum[-1]
            if total > 0:
                low_idx = np.searchsorted(cumsum, total * 0.1)
                high_idx = np.searchsorted(cumsum, total * 0.9)
                return (freqs[low_idx], freqs[high_idx])

        return (0.0, sampling_rate / 2 if sampling_rate else 0.0)

    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format from path"""
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            '.csv': 'csv',
            '.txt': 'txt',
            '.edf': 'edf',
            '.bdf': 'bdf',
            '.gdf': 'gdf',
            '.set': 'eeglab',
            '.mat': 'matlab',
            '.npy': 'numpy',
            '.npz': 'numpy_compressed',
            '.nii': 'nifti',
            '.nii.gz': 'nifti_compressed',
            '.pkl': 'pickle',
            '.json': 'json'
        }
        return format_map.get(ext, 'unknown')


class AgenticDecisionSystem:
    """
    Agentic Decision System for Intelligent Model Selection

    This is the core component that analyzes input data characteristics
    and routes it to the most appropriate disease detection model.

    Decision Factors:
    1. Data type (EEG, MRI, voice, etc.)
    2. Number of channels
    3. Signal quality
    4. Temporal characteristics
    5. Target disease type (if specified)

    Architecture:
        Input -> Data Analyzer -> Decision Logic -> Model Selection -> Output
    """

    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.preprocessor = EEGPreprocessor()

        # Model registry: maps (data_type, disease) to model configuration
        self.model_registry = self._initialize_model_registry()

        # Decision history for learning
        self.decision_history: List[DecisionResult] = []

        # Performance metrics per model
        self.model_performance: Dict[str, Dict] = {}

    def _initialize_model_registry(self) -> Dict:
        """Initialize the model registry with available models"""
        return {
            # EEG-based models
            (DataType.EEG, DiseaseType.ALZHEIMER): {
                'model_name': 'eeg_cnn_alzheimer',
                'model_class': 'EEGClassifier1D',
                'input_shape': (1000, 22),
                'num_classes': 3,
                'description': 'CNN for Alzheimer detection from EEG'
            },
            (DataType.EEG, DiseaseType.PARKINSON): {
                'model_name': 'eeg_cnn_parkinson',
                'model_class': 'EEGClassifier1D',
                'input_shape': (1000, 22),
                'num_classes': 2,
                'description': 'CNN for Parkinson detection from EEG'
            },
            (DataType.EEG, DiseaseType.SCHIZOPHRENIA): {
                'model_name': 'eeg_cnn_schizophrenia',
                'model_class': 'EEGClassifier1D',
                'input_shape': (1000, 22),
                'num_classes': 2,
                'description': 'CNN for Schizophrenia detection from EEG'
            },
            (DataType.EEG, DiseaseType.EPILEPSY): {
                'model_name': 'eeg_cnn_epilepsy',
                'model_class': 'EEGClassifier1D',
                'input_shape': (1000, 22),
                'num_classes': 2,
                'description': 'CNN for Epilepsy detection from EEG'
            },
            (DataType.EEG, DiseaseType.AUTISM): {
                'model_name': 'eeg_cnn_autism',
                'model_class': 'EEGClassifier1D',
                'input_shape': (1000, 22),
                'num_classes': 2,
                'description': 'CNN for Autism detection from EEG'
            },
            (DataType.EEG, DiseaseType.STRESS): {
                'model_name': 'eeg_cnn_stress',
                'model_class': 'EEGClassifier1D',
                'input_shape': (1000, 22),
                'num_classes': 2,
                'description': 'CNN for Stress detection from EEG'
            },
            (DataType.EEG, DiseaseType.DEPRESSION): {
                'model_name': 'eeg_cnn_depression',
                'model_class': 'EEGClassifier1D',
                'input_shape': (1000, 22),
                'num_classes': 2,
                'description': 'CNN for Depression detection from EEG'
            },
            # MRI-based models
            (DataType.MRI, DiseaseType.ALZHEIMER): {
                'model_name': 'mri_cnn3d_alzheimer',
                'model_class': 'AlzheimerCNN3D',
                'input_shape': (64, 64, 64, 1),
                'num_classes': 3,
                'description': '3D CNN for Alzheimer detection from MRI'
            },
            (DataType.MRI, DiseaseType.PARKINSON): {
                'model_name': 'mri_cnn3d_parkinson',
                'model_class': 'MRIClassifier3D',
                'input_shape': (64, 64, 64, 1),
                'num_classes': 2,
                'description': '3D CNN for Parkinson detection from MRI'
            },
            # Voice-based models
            (DataType.VOICE, DiseaseType.PARKINSON): {
                'model_name': 'voice_lstm_parkinson',
                'model_class': 'ParkinsonVoiceLSTM',
                'input_shape': (100, 26),
                'num_classes': 2,
                'description': 'LSTM for Parkinson detection from voice'
            },
        }

    def decide(self, data: np.ndarray,
               target_disease: Optional[DiseaseType] = None,
               file_path: Optional[str] = None,
               metadata: Optional[Dict] = None) -> DecisionResult:
        """
        Make a decision about which model to use for the input data

        Parameters
        ----------
        data : np.ndarray
            Input data array
        target_disease : DiseaseType, optional
            If specified, restrict to this disease type
        file_path : str, optional
            Path to original data file
        metadata : dict, optional
            Additional metadata about the data

        Returns
        -------
        result : DecisionResult
            Decision result with recommended model and reasoning
        """
        metadata = metadata or {}

        # Step 1: Analyze data characteristics
        characteristics = self.data_analyzer.analyze(data, file_path, metadata)

        # Step 2: Determine candidate disease targets
        candidate_diseases = self._determine_candidate_diseases(
            characteristics, target_disease
        )

        # Step 3: Select best model based on characteristics
        model_name, confidence, reasoning = self._select_model(
            characteristics, candidate_diseases
        )

        # Step 4: Determine preprocessing steps
        preprocessing_steps = self._determine_preprocessing(characteristics)

        # Create decision result
        result = DecisionResult(
            recommended_model=model_name,
            recommended_disease_targets=candidate_diseases,
            confidence=confidence,
            reasoning=reasoning,
            data_characteristics=characteristics,
            preprocessing_applied=preprocessing_steps
        )

        # Store in history
        self.decision_history.append(result)

        logger.info(f"Decision: {model_name} for {[d.value for d in candidate_diseases]} "
                   f"(confidence: {confidence:.2f})")

        return result

    def _determine_candidate_diseases(self,
                                       characteristics: DataCharacteristics,
                                       target_disease: Optional[DiseaseType]) -> List[DiseaseType]:
        """Determine which diseases can be detected from this data"""
        if target_disease:
            return [target_disease]

        # Based on data type, determine possible diseases
        data_type = characteristics.data_type

        disease_map = {
            DataType.EEG: [
                DiseaseType.ALZHEIMER,
                DiseaseType.PARKINSON,
                DiseaseType.SCHIZOPHRENIA,
                DiseaseType.EPILEPSY,
                DiseaseType.AUTISM,
                DiseaseType.STRESS,
                DiseaseType.DEPRESSION
            ],
            DataType.MRI: [
                DiseaseType.ALZHEIMER,
                DiseaseType.PARKINSON,
                DiseaseType.SCHIZOPHRENIA
            ],
            DataType.FMRI: [
                DiseaseType.SCHIZOPHRENIA,
                DiseaseType.DEPRESSION,
                DiseaseType.AUTISM
            ],
            DataType.VOICE: [
                DiseaseType.PARKINSON,
                DiseaseType.DEPRESSION
            ],
            DataType.GAIT: [
                DiseaseType.PARKINSON
            ]
        }

        return disease_map.get(data_type, [DiseaseType.HEALTHY])

    def _select_model(self, characteristics: DataCharacteristics,
                      candidate_diseases: List[DiseaseType]) -> Tuple[str, float, str]:
        """Select the best model based on characteristics"""
        data_type = characteristics.data_type

        # Find matching models
        candidate_models = []
        for disease in candidate_diseases:
            key = (data_type, disease)
            if key in self.model_registry:
                model_info = self.model_registry[key]
                score = self._calculate_model_score(characteristics, model_info)
                candidate_models.append((model_info['model_name'], score, disease))

        if not candidate_models:
            # Default fallback
            return 'eeg_cnn_general', 0.5, 'No specific model found, using general classifier'

        # Sort by score and select best
        candidate_models.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score, best_disease = candidate_models[0]

        # Generate reasoning
        reasoning = self._generate_reasoning(characteristics, best_model, best_score)

        return best_model, best_score, reasoning

    def _calculate_model_score(self, characteristics: DataCharacteristics,
                               model_info: Dict) -> float:
        """Calculate compatibility score between data and model"""
        score = 1.0

        # Check input shape compatibility
        expected_shape = model_info['input_shape']

        # For EEG: (samples, channels)
        if len(expected_shape) == 2:
            expected_samples, expected_channels = expected_shape

            # Channel compatibility (more important)
            channel_ratio = min(characteristics.num_channels, expected_channels) / \
                          max(characteristics.num_channels, expected_channels)
            score *= (0.5 + 0.5 * channel_ratio)

            # Sample compatibility
            sample_ratio = min(characteristics.num_samples, expected_samples) / \
                         max(characteristics.num_samples, expected_samples)
            score *= (0.7 + 0.3 * sample_ratio)

        # Signal quality factor
        score *= (0.5 + 0.5 * characteristics.signal_quality)

        # SNR factor
        snr_factor = min(1.0, characteristics.snr / 20.0)  # Normalize to 20 dB
        score *= (0.8 + 0.2 * snr_factor)

        # Artifact penalty
        if characteristics.has_artifacts:
            score *= (1.0 - 0.3 * characteristics.artifact_ratio)

        # Historical performance boost
        if model_info['model_name'] in self.model_performance:
            perf = self.model_performance[model_info['model_name']]
            if perf.get('accuracy', 0) > 0.7:
                score *= 1.1

        return min(1.0, score)

    def _generate_reasoning(self, characteristics: DataCharacteristics,
                           model_name: str, score: float) -> str:
        """Generate human-readable reasoning for the decision"""
        reasons = []

        # Data type
        reasons.append(f"Data type: {characteristics.data_type.value}")

        # Shape
        reasons.append(f"Shape: {characteristics.num_samples} samples x {characteristics.num_channels} channels")

        # Quality assessment
        if characteristics.signal_quality > 0.8:
            reasons.append("High signal quality")
        elif characteristics.signal_quality > 0.5:
            reasons.append("Moderate signal quality")
        else:
            reasons.append("Low signal quality - results may be less reliable")

        # Artifacts
        if characteristics.has_artifacts:
            reasons.append(f"Artifacts detected ({characteristics.artifact_ratio:.1%} of data)")

        # Model selection
        reasons.append(f"Selected model: {model_name} (score: {score:.2f})")

        return "; ".join(reasons)

    def _determine_preprocessing(self, characteristics: DataCharacteristics) -> List[str]:
        """Determine required preprocessing steps"""
        steps = []

        if characteristics.data_type == DataType.EEG:
            steps.append("bandpass_filter")

            if characteristics.has_artifacts:
                steps.append("artifact_removal")

            if characteristics.num_samples != 1000:
                steps.append("resample_to_1000")

            if characteristics.num_channels != 22:
                steps.append("channel_interpolation")

            steps.append("z_normalization")

        elif characteristics.data_type == DataType.MRI:
            steps.append("skull_stripping")
            steps.append("intensity_normalization")
            steps.append("spatial_normalization")

        return steps

    def preprocess_and_decide(self, data: np.ndarray,
                              target_disease: Optional[DiseaseType] = None,
                              file_path: Optional[str] = None,
                              metadata: Optional[Dict] = None) -> Tuple[np.ndarray, DecisionResult]:
        """
        Preprocess data and make decision in one step

        Returns
        -------
        processed_data : np.ndarray
            Preprocessed data ready for model
        decision : DecisionResult
            Decision result with recommended model
        """
        # Make decision first
        decision = self.decide(data, target_disease, file_path, metadata)

        # Preprocess based on data type
        if decision.data_characteristics.data_type == DataType.EEG:
            processed_data, steps = self.preprocessor.preprocess(
                data,
                metadata.get('sampling_rate') if metadata else None
            )
            decision.preprocessing_applied = steps
        else:
            processed_data = data

        return processed_data, decision

    def update_performance(self, model_name: str, metrics: Dict):
        """Update performance metrics for a model after evaluation"""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {}

        self.model_performance[model_name].update(metrics)
        logger.info(f"Updated performance for {model_name}: {metrics}")

    def get_decision_summary(self) -> Dict:
        """Get summary of decision history"""
        if not self.decision_history:
            return {'total_decisions': 0}

        model_counts = {}
        disease_counts = {}
        avg_confidence = []

        for decision in self.decision_history:
            # Count models
            model_counts[decision.recommended_model] = \
                model_counts.get(decision.recommended_model, 0) + 1

            # Count diseases
            for disease in decision.recommended_disease_targets:
                disease_counts[disease.value] = \
                    disease_counts.get(disease.value, 0) + 1

            avg_confidence.append(decision.confidence)

        return {
            'total_decisions': len(self.decision_history),
            'model_distribution': model_counts,
            'disease_distribution': disease_counts,
            'average_confidence': np.mean(avg_confidence),
            'min_confidence': np.min(avg_confidence),
            'max_confidence': np.max(avg_confidence)
        }


# Agent wrapper for the decision system
from .base_agent import BaseAgent, AgentState, AgentMessage, AgentCapability

class AgenticDecisionAgent(BaseAgent):
    """
    Agent wrapper for the Agentic Decision System

    Integrates the decision system with the agent infrastructure
    for A2A communication and orchestration.
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="Agentic Decision Agent",
            agent_type="decision_system"
        )

        self.decision_system = AgenticDecisionSystem()

    def initialize(self):
        """Initialize the decision agent"""
        logger.info(f"Initializing {self.agent_name}")

        # Register capabilities
        self.register_capability(AgentCapability(
            name="analyze_data",
            description="Analyze input data and determine characteristics",
            input_schema={"data": "ndarray", "metadata": "dict"},
            output_schema={"characteristics": "DataCharacteristics"}
        ))

        self.register_capability(AgentCapability(
            name="decide_model",
            description="Decide which model to use for input data",
            input_schema={"data": "ndarray", "target_disease": "str"},
            output_schema={"decision": "DecisionResult"}
        ))

        self.register_capability(AgentCapability(
            name="preprocess_data",
            description="Preprocess data for model input",
            input_schema={"data": "ndarray", "data_type": "str"},
            output_schema={"processed_data": "ndarray", "steps": "list"}
        ))

        # Register action handlers
        self.register_action('analyze', self._handle_analyze)
        self.register_action('decide', self._handle_decide)
        self.register_action('preprocess', self._handle_preprocess)
        self.register_action('full_pipeline', self._handle_full_pipeline)

        logger.info(f"{self.agent_name} initialized with {len(self.capabilities)} capabilities")

    def cleanup(self):
        """Cleanup resources"""
        self.decision_system.decision_history.clear()
        logger.info(f"{self.agent_name} cleaned up")

    def process_data(self, data: Any) -> Dict[str, Any]:
        """Process data through the decision system"""
        if isinstance(data, dict):
            input_data = data.get('data')
            target_disease = data.get('target_disease')
            metadata = data.get('metadata', {})
        else:
            input_data = data
            target_disease = None
            metadata = {}

        # Convert target disease string to enum
        if target_disease and isinstance(target_disease, str):
            try:
                target_disease = DiseaseType(target_disease.lower())
            except ValueError:
                target_disease = None

        # Run decision system
        processed_data, decision = self.decision_system.preprocess_and_decide(
            input_data, target_disease, metadata=metadata
        )

        return {
            'processed_data': processed_data,
            'decision': {
                'recommended_model': decision.recommended_model,
                'recommended_diseases': [d.value for d in decision.recommended_disease_targets],
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'preprocessing_applied': decision.preprocessing_applied,
                'data_characteristics': {
                    'data_type': decision.data_characteristics.data_type.value,
                    'num_channels': decision.data_characteristics.num_channels,
                    'num_samples': decision.data_characteristics.num_samples,
                    'signal_quality': decision.data_characteristics.signal_quality,
                    'snr': decision.data_characteristics.snr
                }
            }
        }

    def _handle_analyze(self, message: AgentMessage) -> Dict:
        """Handle analyze request"""
        data = message.payload.get('data', np.random.randn(1000, 22))
        metadata = message.payload.get('metadata', {})

        characteristics = self.decision_system.data_analyzer.analyze(data, metadata=metadata)

        return {
            'data_type': characteristics.data_type.value,
            'num_channels': characteristics.num_channels,
            'num_samples': characteristics.num_samples,
            'signal_quality': characteristics.signal_quality,
            'snr': characteristics.snr,
            'has_artifacts': characteristics.has_artifacts
        }

    def _handle_decide(self, message: AgentMessage) -> Dict:
        """Handle decision request"""
        data = message.payload.get('data', np.random.randn(1000, 22))
        target_disease_str = message.payload.get('target_disease')

        target_disease = None
        if target_disease_str:
            try:
                target_disease = DiseaseType(target_disease_str.lower())
            except ValueError:
                pass

        decision = self.decision_system.decide(data, target_disease)

        return {
            'recommended_model': decision.recommended_model,
            'recommended_diseases': [d.value for d in decision.recommended_disease_targets],
            'confidence': decision.confidence,
            'reasoning': decision.reasoning
        }

    def _handle_preprocess(self, message: AgentMessage) -> Dict:
        """Handle preprocessing request"""
        data = message.payload.get('data', np.random.randn(1000, 22))
        sampling_rate = message.payload.get('sampling_rate', 256.0)

        processed, steps = self.decision_system.preprocessor.preprocess(data, sampling_rate)

        return {
            'processed_shape': processed.shape,
            'preprocessing_steps': steps
        }

    def _handle_full_pipeline(self, message: AgentMessage) -> Dict:
        """Handle full pipeline request"""
        return self.process_data(message.payload)


if __name__ == "__main__":
    print("=" * 70)
    print("  AGENTIC DECISION SYSTEM DEMO")
    print("=" * 70)

    # Create decision system
    decision_system = AgenticDecisionSystem()

    # Test with synthetic EEG data
    print("\n1. Testing with synthetic EEG data...")
    eeg_data = np.random.randn(1000, 22)  # 1000 samples, 22 channels

    decision = decision_system.decide(
        eeg_data,
        metadata={'sampling_rate': 256.0, 'data_type': 'eeg'}
    )

    print(f"   Recommended model: {decision.recommended_model}")
    print(f"   Target diseases: {[d.value for d in decision.recommended_disease_targets]}")
    print(f"   Confidence: {decision.confidence:.2f}")
    print(f"   Data type: {decision.data_characteristics.data_type.value}")
    print(f"   Signal quality: {decision.data_characteristics.signal_quality:.2f}")
    print(f"   Reasoning: {decision.reasoning}")

    # Test preprocessing
    print("\n2. Testing preprocessing...")
    processed_data, preprocess_decision = decision_system.preprocess_and_decide(
        eeg_data,
        target_disease=DiseaseType.STRESS,
        metadata={'sampling_rate': 256.0}
    )

    print(f"   Original shape: {eeg_data.shape}")
    print(f"   Processed shape: {processed_data.shape}")
    print(f"   Preprocessing steps: {preprocess_decision.preprocessing_applied}")

    # Test with different data shapes
    print("\n3. Testing with different data shapes...")

    test_shapes = [
        (500, 14),   # Fewer samples and channels
        (2000, 32),  # More samples and channels
        (1000, 64),  # Standard samples, more channels
    ]

    for shape in test_shapes:
        test_data = np.random.randn(*shape)
        decision = decision_system.decide(test_data)
        print(f"   Shape {shape}: model={decision.recommended_model}, "
              f"confidence={decision.confidence:.2f}")

    # Print decision summary
    print("\n4. Decision Summary:")
    summary = decision_system.get_decision_summary()
    print(f"   Total decisions: {summary['total_decisions']}")
    print(f"   Average confidence: {summary['average_confidence']:.2f}")
    print(f"   Model distribution: {summary['model_distribution']}")

    # Test the agent wrapper
    print("\n5. Testing Agent Wrapper...")
    agent = AgenticDecisionAgent()
    agent.initialize()

    result = agent.process_data({
        'data': np.random.randn(1000, 22),
        'target_disease': 'autism',
        'metadata': {'sampling_rate': 256.0}
    })

    print(f"   Agent result:")
    print(f"     Model: {result['decision']['recommended_model']}")
    print(f"     Diseases: {result['decision']['recommended_diseases']}")
    print(f"     Confidence: {result['decision']['confidence']:.2f}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
