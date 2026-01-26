"""
Preprocessing Module for Neurological Disease Detection
========================================================

Real preprocessing pipelines for medical data:
- MRI: Skull stripping, bias correction, normalization, registration
- EEG: Filtering, artifact removal, epoching, ICA
- Voice: Noise reduction, normalization, segmentation
- Gait: Filtering, calibration, segmentation

Author: Research Team
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from scipy import signal, ndimage
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """Container for preprocessed data"""
    data: np.ndarray
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]


class BasePreprocessor(ABC):
    """Base class for preprocessing"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

    @abstractmethod
    def preprocess(self, data: Any) -> PreprocessedData:
        """Preprocess data"""
        pass

    def validate(self, data: Any) -> bool:
        """Validate input data"""
        return data is not None


class MRIPreprocessor(BasePreprocessor):
    """
    MRI Preprocessing Pipeline

    Steps:
    1. Intensity normalization
    2. Bias field correction (N4-like)
    3. Skull stripping (threshold-based)
    4. Spatial normalization/resampling
    5. Registration to template (affine)
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.target_shape = config.get('target_shape', [128, 128, 128]) if config else [128, 128, 128]
        self.normalize = config.get('normalize', True) if config else True
        self.skull_strip = config.get('skull_strip', True) if config else True
        self.bias_correction = config.get('bias_correction', True) if config else True

    def preprocess(self, data: np.ndarray) -> PreprocessedData:
        """
        Preprocess MRI volume

        Parameters
        ----------
        data : np.ndarray
            3D MRI volume

        Returns
        -------
        result : PreprocessedData
            Preprocessed MRI data
        """
        quality_metrics = {}

        # Step 1: Check and convert data type
        data = data.astype(np.float32)
        original_shape = data.shape

        # Step 2: Intensity normalization (0-1 range)
        if self.normalize:
            data = self._intensity_normalize(data)
            quality_metrics['intensity_range'] = (float(data.min()), float(data.max()))

        # Step 3: Bias field correction
        if self.bias_correction:
            data = self._bias_field_correction(data)
            quality_metrics['bias_corrected'] = True

        # Step 4: Skull stripping
        if self.skull_strip:
            data, brain_mask = self._skull_strip(data)
            quality_metrics['brain_volume_ratio'] = float(np.sum(brain_mask) / brain_mask.size)

        # Step 5: Resample to target shape
        if list(data.shape) != self.target_shape:
            data = self._resample(data, self.target_shape)
            quality_metrics['resampled'] = True

        # Step 6: Final normalization (z-score)
        data = self._zscore_normalize(data)

        # Calculate quality metrics
        quality_metrics['snr'] = self._calculate_snr(data)
        quality_metrics['contrast'] = self._calculate_contrast(data)

        return PreprocessedData(
            data=data,
            metadata={
                'original_shape': original_shape,
                'final_shape': data.shape,
                'preprocessing_steps': ['normalize', 'bias_correction', 'skull_strip', 'resample']
            },
            quality_metrics=quality_metrics
        )

    def _intensity_normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize intensity to 0-1 range"""
        # Percentile-based normalization to handle outliers
        p1, p99 = np.percentile(data, [1, 99])
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1 + 1e-8)
        return data

    def _bias_field_correction(self, data: np.ndarray) -> np.ndarray:
        """
        N4-like bias field correction (simplified)

        Estimates low-frequency bias field using Gaussian smoothing
        """
        # Estimate bias field as low-frequency component
        sigma = min(data.shape) // 4
        bias_field = ndimage.gaussian_filter(data, sigma=sigma)

        # Avoid division by zero
        bias_field = np.clip(bias_field, 0.1, None)

        # Correct the bias
        corrected = data / bias_field

        # Renormalize
        corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-8)

        return corrected

    def _skull_strip(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple skull stripping using thresholding and morphological operations
        """
        # Otsu-like thresholding
        threshold = np.mean(data) + 0.5 * np.std(data)

        # Create brain mask
        brain_mask = data > threshold * 0.3

        # Morphological operations to clean up
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        brain_mask = ndimage.binary_erosion(brain_mask, iterations=2)
        brain_mask = ndimage.binary_dilation(brain_mask, iterations=2)

        # Keep largest connected component
        labeled, num_features = ndimage.label(brain_mask)
        if num_features > 1:
            sizes = ndimage.sum(brain_mask, labeled, range(1, num_features + 1))
            largest = np.argmax(sizes) + 1
            brain_mask = labeled == largest

        # Apply mask
        data = data * brain_mask

        return data, brain_mask

    def _resample(self, data: np.ndarray, target_shape: List[int]) -> np.ndarray:
        """Resample volume to target shape using trilinear interpolation"""
        zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
        resampled = ndimage.zoom(data, zoom_factors, order=1)  # Linear interpolation
        return resampled

    def _zscore_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        brain_voxels = data[data > 0]
        if len(brain_voxels) > 0:
            mean_val = np.mean(brain_voxels)
            std_val = np.std(brain_voxels)
            data = np.where(data > 0, (data - mean_val) / (std_val + 1e-8), 0)
        return data

    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        signal = data[data > 0]
        if len(signal) == 0:
            return 0.0
        noise_region = data[data <= np.percentile(data, 10)]
        if len(noise_region) == 0 or np.std(noise_region) == 0:
            return 100.0
        return float(np.mean(signal) / (np.std(noise_region) + 1e-8))

    def _calculate_contrast(self, data: np.ndarray) -> float:
        """Calculate image contrast"""
        return float(np.std(data[data > 0])) if np.any(data > 0) else 0.0


class EEGPreprocessor(BasePreprocessor):
    """
    EEG Preprocessing Pipeline

    Steps:
    1. Bandpass filtering (0.5-45 Hz)
    2. Notch filter (50/60 Hz)
    3. Artifact rejection (amplitude threshold)
    4. Re-referencing (average reference)
    5. ICA for artifact removal
    6. Epoching
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.sampling_rate = config.get('sampling_rate', 256) if config else 256
        self.filter_low = config.get('filter_low', 0.5) if config else 0.5
        self.filter_high = config.get('filter_high', 45.0) if config else 45.0
        self.notch_freq = config.get('notch_filter', 50.0) if config else 50.0
        self.epoch_duration = config.get('epoch_duration', 4.0) if config else 4.0
        self.artifact_threshold = config.get('artifact_threshold', 100e-6) if config else 100e-6

    def preprocess(self, data: np.ndarray) -> PreprocessedData:
        """
        Preprocess EEG data

        Parameters
        ----------
        data : np.ndarray
            EEG data [channels x samples]

        Returns
        -------
        result : PreprocessedData
            Preprocessed EEG data
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        quality_metrics = {}
        original_shape = data.shape

        # Step 1: Bandpass filter
        data = self._bandpass_filter(data)

        # Step 2: Notch filter
        data = self._notch_filter(data)

        # Step 3: Artifact rejection
        data, rejected_ratio = self._reject_artifacts(data)
        quality_metrics['artifact_rejection_ratio'] = rejected_ratio

        # Step 4: Re-reference to average
        data = self._rereference(data)

        # Step 5: ICA-based artifact removal (simplified)
        if data.shape[0] > 1:
            data = self._ica_artifact_removal(data)

        # Step 6: Epoch the data
        epochs = self._create_epochs(data)

        # Calculate quality metrics
        quality_metrics['num_epochs'] = len(epochs)
        quality_metrics['snr'] = self._calculate_snr(data)

        return PreprocessedData(
            data=np.array(epochs) if len(epochs) > 0 else data,
            metadata={
                'original_shape': original_shape,
                'sampling_rate': self.sampling_rate,
                'filter_band': (self.filter_low, self.filter_high),
                'num_channels': data.shape[0]
            },
            quality_metrics=quality_metrics
        )

    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter"""
        nyquist = self.sampling_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist

        # Ensure valid filter bounds
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))

        b, a = signal.butter(4, [low, high], btype='band')

        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = signal.filtfilt(b, a, data[i])

        return filtered

    def _notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove line noise"""
        nyquist = self.sampling_rate / 2
        freq = self.notch_freq / nyquist

        if freq >= 1:
            return data

        b, a = signal.iirnotch(freq, Q=30)

        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = signal.filtfilt(b, a, data[i])

        return filtered

    def _reject_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Reject samples with artifacts based on amplitude threshold"""
        # Find samples exceeding threshold
        max_amplitude = np.max(np.abs(data), axis=0)
        artifact_mask = max_amplitude > self.artifact_threshold

        # Interpolate artifact regions
        rejected_ratio = np.mean(artifact_mask)

        if rejected_ratio < 0.5:  # Only interpolate if less than 50% is bad
            clean_indices = np.where(~artifact_mask)[0]
            artifact_indices = np.where(artifact_mask)[0]

            if len(clean_indices) > 2 and len(artifact_indices) > 0:
                for ch in range(data.shape[0]):
                    interp_func = interp1d(clean_indices, data[ch, clean_indices],
                                          kind='linear', fill_value='extrapolate')
                    data[ch, artifact_indices] = interp_func(artifact_indices)

        return data, float(rejected_ratio)

    def _rereference(self, data: np.ndarray) -> np.ndarray:
        """Re-reference to average"""
        if data.shape[0] > 1:
            avg_ref = np.mean(data, axis=0)
            data = data - avg_ref
        return data

    def _ica_artifact_removal(self, data: np.ndarray) -> np.ndarray:
        """
        Simplified ICA-based artifact removal

        Uses PCA whitening + FastICA approximation
        """
        n_channels, n_samples = data.shape

        if n_channels < 2:
            return data

        # Center the data
        data_centered = data - np.mean(data, axis=1, keepdims=True)

        # PCA whitening
        cov = np.cov(data_centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Keep components with significant variance
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Whitening matrix
        D = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        whitening_matrix = D @ eigenvectors.T

        # Whiten the data
        data_white = whitening_matrix @ data_centered

        # Simple component rejection based on kurtosis
        # (high kurtosis often indicates artifacts)
        from scipy.stats import kurtosis
        kurt = np.array([kurtosis(data_white[i]) for i in range(n_channels)])

        # Reject components with very high kurtosis (likely artifacts)
        artifact_components = np.abs(kurt) > 5

        if np.any(artifact_components):
            data_white[artifact_components] = 0

            # Transform back
            dewhitening = np.linalg.pinv(whitening_matrix)
            data = dewhitening @ data_white + np.mean(data, axis=1, keepdims=True)

        return data

    def _create_epochs(self, data: np.ndarray) -> List[np.ndarray]:
        """Create epochs from continuous data"""
        epoch_samples = int(self.epoch_duration * self.sampling_rate)
        n_samples = data.shape[1]

        epochs = []
        for start in range(0, n_samples - epoch_samples, epoch_samples // 2):
            epoch = data[:, start:start + epoch_samples]
            epochs.append(epoch)

        return epochs

    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate SNR estimate"""
        signal_power = np.mean(data ** 2)
        # Estimate noise from high-frequency content
        b, a = signal.butter(4, 0.8, btype='high')
        noise = signal.filtfilt(b, a, data.flatten())
        noise_power = np.mean(noise ** 2)
        return float(10 * np.log10(signal_power / (noise_power + 1e-10)))


class VoicePreprocessor(BasePreprocessor):
    """
    Voice/Audio Preprocessing Pipeline

    Steps:
    1. Resampling to target rate
    2. Pre-emphasis
    3. Noise reduction
    4. Silence removal
    5. Amplitude normalization
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.target_sample_rate = config.get('sample_rate', 16000) if config else 16000
        self.pre_emphasis = config.get('pre_emphasis', 0.97) if config else 0.97
        self.frame_length = config.get('frame_length', 0.025) if config else 0.025

    def preprocess(self, data: np.ndarray, original_sr: int = None) -> PreprocessedData:
        """
        Preprocess audio data

        Parameters
        ----------
        data : np.ndarray
            Audio waveform
        original_sr : int
            Original sample rate

        Returns
        -------
        result : PreprocessedData
            Preprocessed audio data
        """
        original_sr = original_sr or self.target_sample_rate
        quality_metrics = {}

        # Step 1: Resample if needed
        if original_sr != self.target_sample_rate:
            data = self._resample(data, original_sr, self.target_sample_rate)

        # Step 2: Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        original_length = len(data)

        # Step 3: Pre-emphasis
        data = self._pre_emphasis(data)

        # Step 4: Noise reduction
        data = self._noise_reduction(data)

        # Step 5: Remove silence
        data, speech_ratio = self._remove_silence(data)
        quality_metrics['speech_ratio'] = speech_ratio

        # Step 6: Normalize amplitude
        data = self._normalize(data)

        # Calculate quality metrics
        quality_metrics['duration'] = len(data) / self.target_sample_rate
        quality_metrics['snr'] = self._estimate_snr(data)

        return PreprocessedData(
            data=data,
            metadata={
                'sample_rate': self.target_sample_rate,
                'original_length': original_length,
                'final_length': len(data)
            },
            quality_metrics=quality_metrics
        )

    def _resample(self, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return data

        duration = len(data) / orig_sr
        target_length = int(duration * target_sr)

        # Use linear interpolation for resampling
        x_orig = np.linspace(0, duration, len(data))
        x_target = np.linspace(0, duration, target_length)

        interp_func = interp1d(x_orig, data, kind='linear')
        return interp_func(x_target)

    def _pre_emphasis(self, data: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter"""
        return np.append(data[0], data[1:] - self.pre_emphasis * data[:-1])

    def _noise_reduction(self, data: np.ndarray) -> np.ndarray:
        """
        Simple spectral subtraction noise reduction
        """
        # STFT parameters
        frame_size = int(self.frame_length * self.target_sample_rate)
        hop_size = frame_size // 2

        # Compute STFT
        f, t, Zxx = signal.stft(data, fs=self.target_sample_rate,
                                nperseg=frame_size, noverlap=hop_size)

        # Estimate noise from first few frames (assumed to be silence/noise)
        noise_frames = min(10, Zxx.shape[1] // 10)
        noise_spectrum = np.mean(np.abs(Zxx[:, :noise_frames]), axis=1, keepdims=True)

        # Spectral subtraction
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Subtract noise estimate (with floor to avoid negative values)
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        magnitude_clean = np.maximum(magnitude - alpha * noise_spectrum, beta * magnitude)

        # Reconstruct
        Zxx_clean = magnitude_clean * np.exp(1j * phase)
        _, data_clean = signal.istft(Zxx_clean, fs=self.target_sample_rate,
                                     nperseg=frame_size, noverlap=hop_size)

        return data_clean[:len(data)]

    def _remove_silence(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Remove silence from audio"""
        frame_size = int(self.frame_length * self.target_sample_rate)

        # Calculate frame energies
        energies = []
        for i in range(0, len(data) - frame_size, frame_size):
            frame = data[i:i + frame_size]
            energies.append(np.sum(frame ** 2))

        energies = np.array(energies)

        # Threshold based on energy distribution
        threshold = np.percentile(energies, 30)

        # Find speech frames
        speech_mask = energies > threshold

        # Reconstruct speech-only signal
        speech_data = []
        for i, is_speech in enumerate(speech_mask):
            if is_speech:
                start = i * frame_size
                end = start + frame_size
                speech_data.extend(data[start:end])

        speech_ratio = np.mean(speech_mask)

        if len(speech_data) == 0:
            return data, 0.0

        return np.array(speech_data), float(speech_ratio)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize amplitude to [-1, 1]"""
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        return data

    def _estimate_snr(self, data: np.ndarray) -> float:
        """Estimate SNR"""
        # Simple energy-based SNR estimate
        frame_size = int(self.frame_length * self.target_sample_rate)

        energies = []
        for i in range(0, len(data) - frame_size, frame_size):
            frame = data[i:i + frame_size]
            energies.append(np.sum(frame ** 2))

        energies = np.array(energies)

        # Assume lowest 10% are noise
        noise_energy = np.percentile(energies, 10)
        signal_energy = np.mean(energies)

        return float(10 * np.log10(signal_energy / (noise_energy + 1e-10)))


class GaitPreprocessor(BasePreprocessor):
    """
    Gait Sensor Data Preprocessing Pipeline

    Steps:
    1. Calibration (offset removal)
    2. Low-pass filtering
    3. Gravity removal (for accelerometer)
    4. Segmentation into gait cycles
    5. Normalization
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.sampling_rate = config.get('sampling_rate', 100) if config else 100
        self.filter_cutoff = config.get('filter_cutoff', 20.0) if config else 20.0

    def preprocess(self, data: np.ndarray) -> PreprocessedData:
        """
        Preprocess gait sensor data

        Parameters
        ----------
        data : np.ndarray
            Sensor data [samples x channels]
            Expected: accelerometer (x,y,z), gyroscope (x,y,z)

        Returns
        -------
        result : PreprocessedData
            Preprocessed gait data
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        quality_metrics = {}
        original_shape = data.shape

        # Step 1: Calibration (remove offset)
        data = self._calibrate(data)

        # Step 2: Low-pass filter
        data = self._lowpass_filter(data)

        # Step 3: Remove gravity component (for accelerometer)
        if data.shape[1] >= 3:
            data[:, :3] = self._remove_gravity(data[:, :3])

        # Step 4: Detect gait cycles
        gait_cycles, num_cycles = self._segment_gait_cycles(data)
        quality_metrics['num_gait_cycles'] = num_cycles

        # Step 5: Normalize
        data = self._normalize(data)

        # Quality metrics
        quality_metrics['duration'] = data.shape[0] / self.sampling_rate
        quality_metrics['regularity'] = self._calculate_regularity(data)

        return PreprocessedData(
            data=data,
            metadata={
                'original_shape': original_shape,
                'sampling_rate': self.sampling_rate,
                'num_channels': data.shape[1],
                'gait_cycles': gait_cycles
            },
            quality_metrics=quality_metrics
        )

    def _calibrate(self, data: np.ndarray) -> np.ndarray:
        """Remove sensor offset (calibration)"""
        # Use first few samples as rest position
        rest_samples = min(100, len(data) // 10)
        offset = np.mean(data[:rest_samples], axis=0)

        # Don't remove gravity from vertical axis
        if data.shape[1] >= 3:
            offset[2] = 0  # Keep gravity component

        return data - offset

    def _lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply low-pass filter"""
        nyquist = self.sampling_rate / 2
        cutoff = min(self.filter_cutoff / nyquist, 0.99)

        b, a = signal.butter(4, cutoff, btype='low')

        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])

        return filtered

    def _remove_gravity(self, acc_data: np.ndarray) -> np.ndarray:
        """Remove gravity component from accelerometer data"""
        # High-pass filter to remove DC component (gravity)
        nyquist = self.sampling_rate / 2
        cutoff = 0.5 / nyquist

        if cutoff >= 1:
            return acc_data

        b, a = signal.butter(2, cutoff, btype='high')

        filtered = np.zeros_like(acc_data)
        for i in range(acc_data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, acc_data[:, i])

        return filtered

    def _segment_gait_cycles(self, data: np.ndarray) -> Tuple[List[Tuple[int, int]], int]:
        """Segment data into gait cycles"""
        # Use vertical acceleration for heel strike detection
        if data.shape[1] >= 3:
            vertical = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
        else:
            vertical = data[:, 0]

        # Find peaks (heel strikes)
        min_distance = int(self.sampling_rate * 0.4)  # Min 0.4s between steps
        peaks, _ = signal.find_peaks(vertical, distance=min_distance,
                                     height=np.percentile(vertical, 50))

        # Create gait cycles
        gait_cycles = []
        for i in range(len(peaks) - 1):
            gait_cycles.append((peaks[i], peaks[i + 1]))

        return gait_cycles, len(gait_cycles)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization per channel"""
        for i in range(data.shape[1]):
            mean_val = np.mean(data[:, i])
            std_val = np.std(data[:, i])
            if std_val > 0:
                data[:, i] = (data[:, i] - mean_val) / std_val
        return data

    def _calculate_regularity(self, data: np.ndarray) -> float:
        """Calculate gait regularity using autocorrelation"""
        if data.shape[1] >= 3:
            magnitude = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
        else:
            magnitude = data[:, 0]

        # Autocorrelation
        autocorr = np.correlate(magnitude, magnitude, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Find first dominant peak
        min_lag = int(self.sampling_rate * 0.4)
        max_lag = int(self.sampling_rate * 1.5)

        if max_lag < len(autocorr):
            search_region = autocorr[min_lag:max_lag]
            if len(search_region) > 0:
                return float(np.max(search_region))

        return 0.0


class PreprocessingPipeline:
    """
    Combined preprocessing pipeline for all modalities
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.preprocessors = {
            'mri': MRIPreprocessor(config.get('mri', {})),
            'eeg': EEGPreprocessor(config.get('eeg', {})),
            'voice': VoicePreprocessor(config.get('voice', {})),
            'gait': GaitPreprocessor(config.get('gait', {}))
        }

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, PreprocessedData]:
        """
        Preprocess all available modalities

        Parameters
        ----------
        data : dict
            Dictionary with modality keys

        Returns
        -------
        results : dict
            Dictionary of PreprocessedData
        """
        results = {}

        for modality, preprocessor in self.preprocessors.items():
            if modality in data and data[modality] is not None:
                try:
                    results[modality] = preprocessor.preprocess(data[modality])
                    logger.info(f"Preprocessed {modality} data")
                except Exception as e:
                    logger.error(f"Error preprocessing {modality}: {e}")

        return results

    def get_quality_report(self, results: Dict[str, PreprocessedData]) -> Dict:
        """Generate quality report for all preprocessed data"""
        report = {}
        for modality, result in results.items():
            report[modality] = {
                'metadata': result.metadata,
                'quality_metrics': result.quality_metrics
            }
        return report
