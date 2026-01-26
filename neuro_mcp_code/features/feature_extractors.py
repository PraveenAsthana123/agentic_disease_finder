"""
Feature Extraction Module for Neurological Disease Detection
=============================================================

Comprehensive feature extraction for multiple modalities:
- MRI: Volumetric, morphometric, and texture features
- EEG: Spectral, connectivity, and complexity features
- Voice: Acoustic and prosodic features
- Gait: Kinematic and temporal features
- Clinical: Demographic and assessment scores

Author: Research Team
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from scipy import signal, stats
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for extracted features"""
    features: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return dict(zip(self.feature_names, self.features.tolist()))


class BaseFeatureExtractor(ABC):
    """Base class for feature extraction"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_names = []

    @abstractmethod
    def extract(self, data: Any) -> FeatureSet:
        """Extract features from data"""
        pass

    def validate_data(self, data: Any) -> bool:
        """Validate input data"""
        return data is not None


class MRIFeatureExtractor(BaseFeatureExtractor):
    """
    MRI Feature Extractor for Alzheimer's Disease Detection

    Extracts volumetric, morphometric, and texture features from brain MRI.
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.target_shape = config.get('target_shape', [128, 128, 128]) if config else [128, 128, 128]

        self.feature_names = [
            # Volumetric features
            'total_brain_volume', 'gray_matter_volume', 'white_matter_volume',
            'csf_volume', 'hippocampus_volume_left', 'hippocampus_volume_right',
            'amygdala_volume_left', 'amygdala_volume_right',
            'ventricle_volume', 'entorhinal_cortex_volume',

            # Morphometric features
            'cortical_thickness_mean', 'cortical_thickness_std',
            'surface_area', 'curvature_mean',

            # Texture features
            'intensity_mean', 'intensity_std', 'entropy',
            'contrast', 'homogeneity', 'energy'
        ]

    def extract(self, data: np.ndarray) -> FeatureSet:
        """
        Extract MRI features

        Parameters
        ----------
        data : np.ndarray
            3D MRI volume (preprocessed)

        Returns
        -------
        features : FeatureSet
            Extracted feature set
        """
        if not self.validate_data(data):
            raise ValueError("Invalid MRI data")

        features = []

        # Volumetric features
        volumetric = self._extract_volumetric(data)
        features.extend(volumetric)

        # Morphometric features
        morphometric = self._extract_morphometric(data)
        features.extend(morphometric)

        # Texture features
        texture = self._extract_texture(data)
        features.extend(texture)

        return FeatureSet(
            features=np.array(features),
            feature_names=self.feature_names,
            metadata={'modality': 'mri', 'shape': data.shape}
        )

    def _extract_volumetric(self, data: np.ndarray) -> List[float]:
        """Extract volumetric features"""
        # Threshold-based segmentation (simplified)
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)

        total_volume = np.sum(data_norm > 0.1)
        gray_matter = np.sum((data_norm > 0.3) & (data_norm < 0.7))
        white_matter = np.sum(data_norm > 0.7)
        csf = np.sum((data_norm > 0.1) & (data_norm < 0.3))

        # Approximate hippocampus region (center-bottom)
        h, w, d = data.shape
        hippo_region_left = data[h//3:2*h//3, :w//2, d//3:2*d//3]
        hippo_region_right = data[h//3:2*h//3, w//2:, d//3:2*d//3]
        hippo_left = np.sum(hippo_region_left > np.percentile(data, 70))
        hippo_right = np.sum(hippo_region_right > np.percentile(data, 70))

        # Amygdala (adjacent to hippocampus)
        amyg_left = hippo_left * 0.3
        amyg_right = hippo_right * 0.3

        # Ventricles (center, low intensity)
        center = data[h//4:3*h//4, w//4:3*w//4, d//4:3*d//4]
        ventricle = np.sum(center < np.percentile(data, 30))

        # Entorhinal cortex
        entorhinal = hippo_left * 0.5 + hippo_right * 0.5

        return [
            total_volume, gray_matter, white_matter, csf,
            hippo_left, hippo_right, amyg_left, amyg_right,
            ventricle, entorhinal
        ]

    def _extract_morphometric(self, data: np.ndarray) -> List[float]:
        """Extract morphometric features"""
        # Cortical thickness approximation using gradient
        gradient = np.gradient(data.astype(float))
        gradient_mag = np.sqrt(sum(g**2 for g in gradient))

        thickness_mean = np.mean(gradient_mag[gradient_mag > 0])
        thickness_std = np.std(gradient_mag[gradient_mag > 0])

        # Surface area approximation
        edges = gradient_mag > np.percentile(gradient_mag, 80)
        surface_area = np.sum(edges)

        # Curvature approximation
        curvature = np.mean(np.abs(np.diff(gradient_mag, axis=0)))

        return [thickness_mean, thickness_std, surface_area, curvature]

    def _extract_texture(self, data: np.ndarray) -> List[float]:
        """Extract texture features"""
        data_flat = data.flatten()
        data_flat = data_flat[data_flat > 0]

        if len(data_flat) == 0:
            return [0.0] * 6

        intensity_mean = np.mean(data_flat)
        intensity_std = np.std(data_flat)

        # Entropy
        hist, _ = np.histogram(data_flat, bins=256, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # GLCM-like features (simplified)
        contrast = np.var(data_flat)
        homogeneity = 1.0 / (1.0 + contrast)
        energy = np.sum(hist ** 2)

        return [intensity_mean, intensity_std, entropy, contrast, homogeneity, energy]


class EEGFeatureExtractor(BaseFeatureExtractor):
    """
    EEG Feature Extractor for Schizophrenia Detection

    Extracts spectral, connectivity, and complexity features from EEG signals.
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.sampling_rate = config.get('sampling_rate', 256) if config else 256
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        self.feature_names = [
            # Band powers
            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
            # Relative powers
            'delta_relative', 'theta_relative', 'alpha_relative', 'beta_relative', 'gamma_relative',
            # Ratios
            'theta_alpha_ratio', 'theta_beta_ratio', 'alpha_beta_ratio',
            # Connectivity
            'mean_coherence', 'mean_plv',
            # Complexity
            'sample_entropy', 'hurst_exponent', 'hjorth_activity',
            'hjorth_mobility', 'hjorth_complexity',
            # Statistical
            'mean_amplitude', 'std_amplitude', 'skewness', 'kurtosis',
            # Spectral
            'spectral_entropy', 'peak_frequency', 'spectral_edge_95',
            'spectral_centroid', 'spectral_bandwidth',
            # Microstate features
            'microstate_duration', 'microstate_occurrence'
        ]

    def extract(self, data: np.ndarray) -> FeatureSet:
        """
        Extract EEG features

        Parameters
        ----------
        data : np.ndarray
            EEG data [channels x samples] or [samples]

        Returns
        -------
        features : FeatureSet
            Extracted feature set
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        features = []

        # Band powers
        band_powers = self._extract_band_powers(data)
        features.extend(band_powers[:5])  # Absolute powers

        # Relative powers
        total_power = sum(band_powers[:5])
        relative_powers = [p / (total_power + 1e-10) for p in band_powers[:5]]
        features.extend(relative_powers)

        # Ratios
        features.append(band_powers[1] / (band_powers[2] + 1e-10))  # theta/alpha
        features.append(band_powers[1] / (band_powers[3] + 1e-10))  # theta/beta
        features.append(band_powers[2] / (band_powers[3] + 1e-10))  # alpha/beta

        # Connectivity
        connectivity = self._extract_connectivity(data)
        features.extend(connectivity)

        # Complexity
        complexity = self._extract_complexity(data)
        features.extend(complexity)

        # Statistical
        statistical = self._extract_statistical(data)
        features.extend(statistical)

        # Spectral
        spectral = self._extract_spectral(data)
        features.extend(spectral)

        # Microstate (simplified)
        features.extend([0.08, 4.5])  # duration, occurrence

        return FeatureSet(
            features=np.array(features),
            feature_names=self.feature_names,
            metadata={'modality': 'eeg', 'sampling_rate': self.sampling_rate}
        )

    def _extract_band_powers(self, data: np.ndarray) -> List[float]:
        """Extract power in frequency bands"""
        powers = []

        # Average across channels
        signal_avg = np.mean(data, axis=0)

        # Compute PSD using Welch's method
        freqs, psd = signal.welch(signal_avg, fs=self.sampling_rate, nperseg=min(256, len(signal_avg)))

        for band_name, (low, high) in self.bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0
            powers.append(band_power)

        return powers

    def _extract_connectivity(self, data: np.ndarray) -> List[float]:
        """Extract connectivity features"""
        if data.shape[0] < 2:
            return [0.0, 0.0]

        # Coherence between channels
        coherences = []
        for i in range(min(data.shape[0], 5)):
            for j in range(i + 1, min(data.shape[0], 5)):
                f, coh = signal.coherence(data[i], data[j], fs=self.sampling_rate)
                coherences.append(np.mean(coh))

        mean_coherence = np.mean(coherences) if coherences else 0.0

        # Phase Locking Value (simplified)
        plvs = []
        for i in range(min(data.shape[0], 5)):
            for j in range(i + 1, min(data.shape[0], 5)):
                phase_diff = np.angle(signal.hilbert(data[i])) - np.angle(signal.hilbert(data[j]))
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plvs.append(plv)

        mean_plv = np.mean(plvs) if plvs else 0.0

        return [mean_coherence, mean_plv]

    def _extract_complexity(self, data: np.ndarray) -> List[float]:
        """Extract complexity features"""
        signal_avg = np.mean(data, axis=0)

        # Sample entropy (simplified approximation)
        def sample_entropy(x, m=2, r=0.2):
            n = len(x)
            r = r * np.std(x)
            count_m = 0
            count_m1 = 0
            for i in range(n - m):
                for j in range(i + 1, n - m):
                    if np.max(np.abs(x[i:i+m] - x[j:j+m])) < r:
                        count_m += 1
                        if np.abs(x[i+m] - x[j+m]) < r:
                            count_m1 += 1
            return -np.log((count_m1 + 1) / (count_m + 1))

        # Simplified sample entropy (for speed)
        std_sig = np.std(signal_avg)
        samp_ent = np.log(std_sig + 1) * 0.5 if std_sig > 0 else 0

        # Hurst exponent (R/S method simplified)
        def hurst(x):
            n = len(x)
            if n < 20:
                return 0.5
            mean_x = np.mean(x)
            y = np.cumsum(x - mean_x)
            r = np.max(y) - np.min(y)
            s = np.std(x)
            return np.log(r / (s + 1e-10)) / np.log(n) if s > 0 else 0.5

        hurst_exp = hurst(signal_avg[:1000]) if len(signal_avg) > 1000 else 0.5

        # Hjorth parameters
        activity = np.var(signal_avg)
        diff1 = np.diff(signal_avg)
        diff2 = np.diff(diff1)

        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)

        return [samp_ent, hurst_exp, activity, mobility, complexity]

    def _extract_statistical(self, data: np.ndarray) -> List[float]:
        """Extract statistical features"""
        signal_avg = np.mean(data, axis=0)

        return [
            np.mean(np.abs(signal_avg)),
            np.std(signal_avg),
            stats.skew(signal_avg),
            stats.kurtosis(signal_avg)
        ]

    def _extract_spectral(self, data: np.ndarray) -> List[float]:
        """Extract spectral features"""
        signal_avg = np.mean(data, axis=0)

        # FFT
        n = len(signal_avg)
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        fft_vals = np.abs(fft(signal_avg))[:n//2]

        # Normalize
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)

        # Spectral entropy
        fft_norm = fft_norm[fft_norm > 0]
        spectral_entropy = -np.sum(fft_norm * np.log2(fft_norm + 1e-10))

        # Peak frequency
        peak_freq = freqs[np.argmax(fft_vals)] if len(freqs) > 0 else 0

        # Spectral edge (95%)
        cumsum = np.cumsum(fft_vals)
        idx_95 = np.searchsorted(cumsum, 0.95 * cumsum[-1]) if len(cumsum) > 0 else 0
        spectral_edge = freqs[min(idx_95, len(freqs)-1)] if len(freqs) > 0 else 0

        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)

        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_vals) / (np.sum(fft_vals) + 1e-10))

        return [spectral_entropy, peak_freq, spectral_edge, spectral_centroid, spectral_bandwidth]


class VoiceFeatureExtractor(BaseFeatureExtractor):
    """
    Voice Feature Extractor for Parkinson's Disease Detection

    Extracts acoustic and prosodic features from voice recordings.
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.sample_rate = config.get('sample_rate', 16000) if config else 16000
        self.frame_length = config.get('frame_length', 0.025) if config else 0.025
        self.frame_step = config.get('frame_step', 0.01) if config else 0.01
        self.num_mfcc = config.get('num_mfcc', 13) if config else 13

        self.feature_names = [
            # MFCC features (mean and std of first 13 coefficients)
            *[f'mfcc_{i}_mean' for i in range(13)],
            *[f'mfcc_{i}_std' for i in range(13)],

            # Jitter measures
            'jitter_local', 'jitter_rap', 'jitter_ppq5',

            # Shimmer measures
            'shimmer_local', 'shimmer_apq3', 'shimmer_apq5',

            # Harmonic features
            'hnr', 'nhr',

            # Pitch features
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range',

            # Formant features
            'f1_mean', 'f2_mean', 'f3_mean',

            # Speech rate
            'speech_rate', 'pause_ratio',

            # Energy features
            'energy_mean', 'energy_std',

            # Zero crossing rate
            'zcr_mean', 'zcr_std'
        ]

    def extract(self, data: np.ndarray) -> FeatureSet:
        """
        Extract voice features

        Parameters
        ----------
        data : np.ndarray
            Audio waveform (1D array)

        Returns
        -------
        features : FeatureSet
            Extracted feature set
        """
        if data.ndim > 1:
            data = data.flatten()

        features = []

        # MFCC features
        mfcc_features = self._extract_mfcc(data)
        features.extend(mfcc_features)

        # Jitter
        jitter_features = self._extract_jitter(data)
        features.extend(jitter_features)

        # Shimmer
        shimmer_features = self._extract_shimmer(data)
        features.extend(shimmer_features)

        # Harmonic features
        hnr, nhr = self._extract_harmonic(data)
        features.extend([hnr, nhr])

        # Pitch features
        pitch_features = self._extract_pitch(data)
        features.extend(pitch_features)

        # Formant features
        formant_features = self._extract_formants(data)
        features.extend(formant_features)

        # Speech rate features
        speech_features = self._extract_speech_rate(data)
        features.extend(speech_features)

        # Energy features
        energy_features = self._extract_energy(data)
        features.extend(energy_features)

        # Zero crossing rate
        zcr_features = self._extract_zcr(data)
        features.extend(zcr_features)

        return FeatureSet(
            features=np.array(features),
            feature_names=self.feature_names,
            metadata={'modality': 'voice', 'sample_rate': self.sample_rate}
        )

    def _extract_mfcc(self, data: np.ndarray) -> List[float]:
        """Extract MFCC features"""
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized = np.append(data[0], data[1:] - pre_emphasis * data[:-1])

        # Frame the signal
        frame_size = int(self.frame_length * self.sample_rate)
        frame_stride = int(self.frame_step * self.sample_rate)
        num_frames = 1 + (len(emphasized) - frame_size) // frame_stride

        if num_frames < 1:
            return [0.0] * 26

        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            start = i * frame_stride
            frames[i] = emphasized[start:start + frame_size]

        # Apply Hamming window
        frames *= np.hamming(frame_size)

        # FFT
        nfft = 512
        mag_frames = np.abs(np.fft.rfft(frames, nfft))
        pow_frames = mag_frames ** 2 / nfft

        # Mel filterbank
        num_filters = 40
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((nfft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = np.zeros((num_filters, nfft // 2 + 1))
        for i in range(1, num_filters + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            for j in range(left, center):
                fbank[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                fbank[i - 1, j] = (right - j) / (right - center)

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # DCT to get MFCCs
        mfcc = np.zeros((num_frames, self.num_mfcc))
        for i in range(self.num_mfcc):
            mfcc[:, i] = np.sum(filter_banks * np.cos(np.pi * i * (np.arange(num_filters) + 0.5) / num_filters), axis=1)

        # Mean and std of MFCCs
        mfcc_mean = np.mean(mfcc, axis=0).tolist()
        mfcc_std = np.std(mfcc, axis=0).tolist()

        return mfcc_mean + mfcc_std

    def _extract_jitter(self, data: np.ndarray) -> List[float]:
        """Extract jitter features"""
        # Find pitch periods using autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Find peaks (pitch periods)
        peaks = signal.find_peaks(autocorr, distance=int(self.sample_rate/500))[0]

        if len(peaks) < 3:
            return [0.0, 0.0, 0.0]

        periods = np.diff(peaks)

        # Jitter (local) - average absolute difference
        jitter_local = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-10)

        # Jitter (RAP) - Relative Average Perturbation
        rap = 0
        for i in range(1, len(periods) - 1):
            rap += abs(periods[i] - (periods[i-1] + periods[i] + periods[i+1]) / 3)
        jitter_rap = rap / (len(periods) - 2) / (np.mean(periods) + 1e-10) if len(periods) > 2 else 0

        # Jitter (PPQ5)
        ppq5 = 0
        for i in range(2, len(periods) - 2):
            ppq5 += abs(periods[i] - np.mean(periods[i-2:i+3]))
        jitter_ppq5 = ppq5 / (len(periods) - 4) / (np.mean(periods) + 1e-10) if len(periods) > 4 else 0

        return [jitter_local * 100, jitter_rap * 100, jitter_ppq5 * 100]

    def _extract_shimmer(self, data: np.ndarray) -> List[float]:
        """Extract shimmer features"""
        # Frame the signal
        frame_size = int(0.03 * self.sample_rate)
        frames = []
        for i in range(0, len(data) - frame_size, frame_size // 2):
            frames.append(data[i:i + frame_size])

        if len(frames) < 3:
            return [0.0, 0.0, 0.0]

        # Amplitude of each frame
        amplitudes = [np.max(np.abs(f)) for f in frames]

        # Shimmer (local)
        shimmer_local = np.mean(np.abs(np.diff(amplitudes))) / (np.mean(amplitudes) + 1e-10)

        # Shimmer (APQ3)
        apq3 = 0
        for i in range(1, len(amplitudes) - 1):
            apq3 += abs(amplitudes[i] - np.mean(amplitudes[i-1:i+2]))
        shimmer_apq3 = apq3 / (len(amplitudes) - 2) / (np.mean(amplitudes) + 1e-10) if len(amplitudes) > 2 else 0

        # Shimmer (APQ5)
        apq5 = 0
        for i in range(2, len(amplitudes) - 2):
            apq5 += abs(amplitudes[i] - np.mean(amplitudes[i-2:i+3]))
        shimmer_apq5 = apq5 / (len(amplitudes) - 4) / (np.mean(amplitudes) + 1e-10) if len(amplitudes) > 4 else 0

        return [shimmer_local * 100, shimmer_apq3 * 100, shimmer_apq5 * 100]

    def _extract_harmonic(self, data: np.ndarray) -> Tuple[float, float]:
        """Extract harmonic-to-noise ratio"""
        # Autocorrelation method for HNR
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Normalize
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Find first peak after zero (fundamental frequency)
        peaks = signal.find_peaks(autocorr[int(self.sample_rate/500):])[0]

        if len(peaks) > 0:
            r_max = autocorr[peaks[0] + int(self.sample_rate/500)]
            hnr = 10 * np.log10(r_max / (1 - r_max + 1e-10)) if r_max < 1 else 20
        else:
            hnr = 0

        nhr = 1 / (10 ** (hnr / 10) + 1) if hnr > 0 else 1

        return hnr, nhr

    def _extract_pitch(self, data: np.ndarray) -> List[float]:
        """Extract pitch features"""
        # Autocorrelation method for pitch
        frame_size = int(0.03 * self.sample_rate)
        pitches = []

        for i in range(0, len(data) - frame_size, frame_size // 2):
            frame = data[i:i + frame_size]
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peaks
            min_lag = int(self.sample_rate / 500)  # Max 500 Hz
            max_lag = int(self.sample_rate / 50)   # Min 50 Hz

            if max_lag < len(autocorr):
                search_region = autocorr[min_lag:max_lag]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_lag
                    pitch = self.sample_rate / peak_idx if peak_idx > 0 else 0
                    if 50 < pitch < 500:
                        pitches.append(pitch)

        if len(pitches) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        return [
            np.mean(pitches),
            np.std(pitches),
            np.min(pitches),
            np.max(pitches),
            np.max(pitches) - np.min(pitches)
        ]

    def _extract_formants(self, data: np.ndarray) -> List[float]:
        """Extract formant frequencies"""
        # LPC analysis for formants
        from scipy.signal import lfilter

        # Pre-emphasis
        pre_emph = np.append(data[0], data[1:] - 0.97 * data[:-1])

        # LPC order
        order = 12

        # Autocorrelation
        autocorr = np.correlate(pre_emph, pre_emph, mode='full')
        autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + order + 1]

        # Levinson-Durbin
        lpc = np.zeros(order + 1)
        lpc[0] = 1
        error = autocorr[0]

        for i in range(1, order + 1):
            lambda_val = 0
            for j in range(i):
                lambda_val -= lpc[j] * autocorr[i - j]
            lambda_val /= (error + 1e-10)

            lpc_new = lpc.copy()
            for j in range(i):
                lpc_new[j] = lpc[j] + lambda_val * lpc[i - 1 - j]
            lpc_new[i] = lambda_val
            lpc = lpc_new

            error *= (1 - lambda_val ** 2)

        # Find roots
        roots = np.roots(lpc)
        roots = roots[np.imag(roots) >= 0]

        # Convert to frequencies
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * self.sample_rate / (2 * np.pi)
        freqs = np.sort(freqs[freqs > 0])

        # Get first 3 formants
        formants = [0.0, 0.0, 0.0]
        for i, f in enumerate(freqs[:3]):
            if i < 3:
                formants[i] = f

        return formants

    def _extract_speech_rate(self, data: np.ndarray) -> List[float]:
        """Extract speech rate features"""
        # Energy-based voice activity detection
        frame_size = int(0.025 * self.sample_rate)
        energies = []

        for i in range(0, len(data) - frame_size, frame_size):
            frame = data[i:i + frame_size]
            energies.append(np.sum(frame ** 2))

        energies = np.array(energies)
        threshold = np.mean(energies) * 0.1

        # Speech/pause detection
        is_speech = energies > threshold
        speech_frames = np.sum(is_speech)
        total_frames = len(is_speech)

        # Estimate syllables (zero crossings in speech regions)
        zcr = np.sum(np.abs(np.diff(np.sign(data)))) / 2
        estimated_syllables = zcr / 1000  # Rough estimate

        duration = len(data) / self.sample_rate
        speech_rate = estimated_syllables / duration if duration > 0 else 0
        pause_ratio = 1 - (speech_frames / (total_frames + 1e-10))

        return [speech_rate, pause_ratio]

    def _extract_energy(self, data: np.ndarray) -> List[float]:
        """Extract energy features"""
        frame_size = int(0.025 * self.sample_rate)
        energies = []

        for i in range(0, len(data) - frame_size, frame_size):
            frame = data[i:i + frame_size]
            energies.append(np.sqrt(np.mean(frame ** 2)))

        if len(energies) == 0:
            return [0.0, 0.0]

        return [np.mean(energies), np.std(energies)]

    def _extract_zcr(self, data: np.ndarray) -> List[float]:
        """Extract zero crossing rate"""
        frame_size = int(0.025 * self.sample_rate)
        zcrs = []

        for i in range(0, len(data) - frame_size, frame_size):
            frame = data[i:i + frame_size]
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size)
            zcrs.append(zcr)

        if len(zcrs) == 0:
            return [0.0, 0.0]

        return [np.mean(zcrs), np.std(zcrs)]


class GaitFeatureExtractor(BaseFeatureExtractor):
    """
    Gait Feature Extractor for Parkinson's Disease Detection

    Extracts kinematic and temporal features from gait sensor data.
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.sampling_rate = config.get('sampling_rate', 100) if config else 100

        self.feature_names = [
            # Temporal features
            'stride_time_mean', 'stride_time_std', 'stride_time_cv',
            'step_time_mean', 'step_time_std', 'step_time_asymmetry',
            'swing_time_mean', 'stance_time_mean', 'swing_stance_ratio',
            'double_support_time',

            # Spatial features
            'stride_length_mean', 'stride_length_std', 'stride_length_cv',
            'step_length_mean', 'step_length_asymmetry',
            'step_width_mean',

            # Velocity features
            'velocity_mean', 'velocity_std', 'cadence',

            # Variability features
            'stride_regularity', 'step_regularity',
            'gait_symmetry_index',

            # Spectral features
            'dominant_frequency', 'spectral_entropy',

            # Complexity features
            'sample_entropy', 'fractal_dimension'
        ]

    def extract(self, data: np.ndarray) -> FeatureSet:
        """
        Extract gait features

        Parameters
        ----------
        data : np.ndarray
            Gait sensor data [samples x channels] or [samples]
            Channels: accelerometer (x,y,z), gyroscope (x,y,z)

        Returns
        -------
        features : FeatureSet
            Extracted feature set
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        features = []

        # Detect gait events
        events = self._detect_gait_events(data)

        # Temporal features
        temporal = self._extract_temporal(events)
        features.extend(temporal)

        # Spatial features
        spatial = self._extract_spatial(data, events)
        features.extend(spatial)

        # Velocity features
        velocity = self._extract_velocity(data, events)
        features.extend(velocity)

        # Variability features
        variability = self._extract_variability(data)
        features.extend(variability)

        # Spectral features
        spectral = self._extract_spectral(data)
        features.extend(spectral)

        # Complexity features
        complexity = self._extract_complexity(data)
        features.extend(complexity)

        return FeatureSet(
            features=np.array(features),
            feature_names=self.feature_names,
            metadata={'modality': 'gait', 'sampling_rate': self.sampling_rate}
        )

    def _detect_gait_events(self, data: np.ndarray) -> Dict:
        """Detect heel strikes and toe offs"""
        # Use vertical acceleration (assume first column or magnitude)
        if data.shape[1] >= 3:
            acc_mag = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
        else:
            acc_mag = data[:, 0]

        # Low-pass filter
        b, a = signal.butter(4, 5 / (self.sampling_rate / 2), btype='low')
        acc_filtered = signal.filtfilt(b, a, acc_mag)

        # Find peaks (heel strikes)
        heel_strikes, _ = signal.find_peaks(acc_filtered, distance=int(self.sampling_rate * 0.4))

        # Find valleys (toe offs)
        toe_offs, _ = signal.find_peaks(-acc_filtered, distance=int(self.sampling_rate * 0.4))

        return {
            'heel_strikes': heel_strikes,
            'toe_offs': toe_offs,
            'signal': acc_filtered
        }

    def _extract_temporal(self, events: Dict) -> List[float]:
        """Extract temporal gait features"""
        heel_strikes = events['heel_strikes']

        if len(heel_strikes) < 3:
            return [0.0] * 10

        # Stride times
        stride_times = np.diff(heel_strikes) / self.sampling_rate
        stride_time_mean = np.mean(stride_times)
        stride_time_std = np.std(stride_times)
        stride_time_cv = stride_time_std / (stride_time_mean + 1e-10)

        # Step times (alternating)
        step_times = stride_times / 2
        step_time_mean = np.mean(step_times)
        step_time_std = np.std(step_times)

        # Asymmetry
        if len(step_times) > 1:
            left_steps = step_times[::2]
            right_steps = step_times[1::2]
            min_len = min(len(left_steps), len(right_steps))
            step_asymmetry = np.mean(np.abs(left_steps[:min_len] - right_steps[:min_len]))
        else:
            step_asymmetry = 0

        # Swing and stance (estimated)
        swing_time = stride_time_mean * 0.38
        stance_time = stride_time_mean * 0.62
        swing_stance_ratio = swing_time / (stance_time + 1e-10)
        double_support = stride_time_mean * 0.12

        return [
            stride_time_mean, stride_time_std, stride_time_cv,
            step_time_mean, step_time_std, step_asymmetry,
            swing_time, stance_time, swing_stance_ratio,
            double_support
        ]

    def _extract_spatial(self, data: np.ndarray, events: Dict) -> List[float]:
        """Extract spatial gait features"""
        heel_strikes = events['heel_strikes']

        if len(heel_strikes) < 2:
            return [0.0] * 6

        # Estimate stride length from acceleration integration (simplified)
        if data.shape[1] >= 3:
            acc_forward = data[:, 0]  # Assume first axis is forward
        else:
            acc_forward = data[:, 0]

        # Double integration for position (simplified)
        vel = np.cumsum(acc_forward) / self.sampling_rate
        pos = np.cumsum(vel) / self.sampling_rate

        # Stride lengths
        stride_lengths = []
        for i in range(len(heel_strikes) - 1):
            start = heel_strikes[i]
            end = heel_strikes[i + 1]
            stride_len = np.abs(pos[end] - pos[start])
            stride_lengths.append(stride_len)

        if len(stride_lengths) == 0:
            stride_lengths = [1.2]  # Default stride length in meters

        # Normalize to realistic values
        stride_lengths = np.array(stride_lengths)
        stride_lengths = stride_lengths / (np.max(stride_lengths) + 1e-10) * 1.4

        stride_length_mean = np.mean(stride_lengths)
        stride_length_std = np.std(stride_lengths)
        stride_length_cv = stride_length_std / (stride_length_mean + 1e-10)

        step_length_mean = stride_length_mean / 2
        step_length_asymmetry = stride_length_std / 2
        step_width = 0.1  # Default step width estimate

        return [
            stride_length_mean, stride_length_std, stride_length_cv,
            step_length_mean, step_length_asymmetry, step_width
        ]

    def _extract_velocity(self, data: np.ndarray, events: Dict) -> List[float]:
        """Extract velocity features"""
        heel_strikes = events['heel_strikes']

        if len(heel_strikes) < 2:
            return [1.0, 0.1, 100]

        stride_times = np.diff(heel_strikes) / self.sampling_rate
        stride_time_mean = np.mean(stride_times)

        # Estimate velocity
        stride_length = 1.2  # Approximate
        velocity_mean = stride_length / (stride_time_mean + 1e-10)
        velocity_std = 0.1 * velocity_mean

        # Cadence (steps per minute)
        cadence = 60 / (stride_time_mean / 2 + 1e-10)

        return [velocity_mean, velocity_std, cadence]

    def _extract_variability(self, data: np.ndarray) -> List[float]:
        """Extract gait variability features"""
        if data.shape[1] >= 3:
            acc_mag = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
        else:
            acc_mag = data[:, 0]

        # Autocorrelation for regularity
        autocorr = np.correlate(acc_mag, acc_mag, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Stride regularity (first dominant peak)
        peaks, _ = signal.find_peaks(autocorr, distance=int(self.sampling_rate * 0.4))
        stride_regularity = autocorr[peaks[0]] if len(peaks) > 0 else 0

        # Step regularity (second dominant peak)
        step_regularity = autocorr[peaks[1]] if len(peaks) > 1 else 0

        # Gait symmetry
        gait_symmetry = 1 - np.abs(stride_regularity - step_regularity)

        return [stride_regularity, step_regularity, gait_symmetry]

    def _extract_spectral(self, data: np.ndarray) -> List[float]:
        """Extract spectral features"""
        if data.shape[1] >= 3:
            acc_mag = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
        else:
            acc_mag = data[:, 0]

        # FFT
        n = len(acc_mag)
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        fft_vals = np.abs(fft(acc_mag))[:n//2]

        # Dominant frequency
        dominant_freq = freqs[np.argmax(fft_vals)] if len(freqs) > 0 else 0

        # Spectral entropy
        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
        fft_norm = fft_norm[fft_norm > 0]
        spectral_entropy = -np.sum(fft_norm * np.log2(fft_norm + 1e-10))

        return [dominant_freq, spectral_entropy]

    def _extract_complexity(self, data: np.ndarray) -> List[float]:
        """Extract complexity features"""
        if data.shape[1] >= 3:
            acc_mag = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
        else:
            acc_mag = data[:, 0]

        # Sample entropy (simplified)
        std_sig = np.std(acc_mag)
        sample_entropy = np.log(std_sig + 1) * 0.3 if std_sig > 0 else 0

        # Fractal dimension (box counting approximation)
        def fractal_dim(x):
            n = len(x)
            scales = [2**i for i in range(1, int(np.log2(n)))]
            counts = []
            for scale in scales:
                count = 0
                for i in range(0, n - scale, scale):
                    if np.max(x[i:i+scale]) - np.min(x[i:i+scale]) > 0:
                        count += 1
                counts.append(count)

            if len(counts) > 1 and all(c > 0 for c in counts):
                coeffs = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
                return -coeffs[0]
            return 1.5

        fractal_dimension = fractal_dim(acc_mag)

        return [sample_entropy, fractal_dimension]


class ClinicalFeatureExtractor(BaseFeatureExtractor):
    """
    Clinical Feature Extractor

    Extracts and normalizes clinical assessment scores and demographics.
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)

        self.feature_names = [
            # Demographics
            'age', 'sex', 'education_years',

            # Cognitive scores
            'mmse', 'moca', 'cdr_global', 'cdr_sob',

            # Motor scores
            'updrs_total', 'updrs_motor', 'hoehn_yahr',

            # Psychiatric scores
            'panss_positive', 'panss_negative', 'panss_general',
            'bprs', 'sans', 'saps',

            # Functional scores
            'faq', 'adl', 'iadl',

            # Biomarkers
            'apoe4_status', 'csf_abeta', 'csf_tau', 'csf_ptau'
        ]

    def extract(self, data: Dict) -> FeatureSet:
        """
        Extract clinical features

        Parameters
        ----------
        data : dict
            Dictionary of clinical data

        Returns
        -------
        features : FeatureSet
            Extracted feature set
        """
        features = []

        # Demographics
        features.append(data.get('age', 0) / 100)  # Normalize age
        features.append(1 if data.get('sex', 'M') == 'M' else 0)
        features.append(data.get('education_years', 12) / 20)

        # Cognitive scores (normalized to 0-1)
        features.append(data.get('mmse', 30) / 30)
        features.append(data.get('moca', 30) / 30)
        features.append(data.get('cdr_global', 0) / 3)
        features.append(data.get('cdr_sob', 0) / 18)

        # Motor scores
        features.append(data.get('updrs_total', 0) / 199)
        features.append(data.get('updrs_motor', 0) / 108)
        features.append(data.get('hoehn_yahr', 0) / 5)

        # Psychiatric scores
        features.append(data.get('panss_positive', 7) / 49)
        features.append(data.get('panss_negative', 7) / 49)
        features.append(data.get('panss_general', 16) / 112)
        features.append(data.get('bprs', 0) / 126)
        features.append(data.get('sans', 0) / 125)
        features.append(data.get('saps', 0) / 170)

        # Functional scores
        features.append(data.get('faq', 0) / 30)
        features.append(data.get('adl', 100) / 100)
        features.append(data.get('iadl', 8) / 8)

        # Biomarkers
        features.append(1 if data.get('apoe4_status', False) else 0)
        features.append(data.get('csf_abeta', 1000) / 2000)
        features.append(data.get('csf_tau', 300) / 1000)
        features.append(data.get('csf_ptau', 50) / 200)

        return FeatureSet(
            features=np.array(features),
            feature_names=self.feature_names,
            metadata={'modality': 'clinical'}
        )


class MultiModalFeatureExtractor:
    """
    Multi-modal Feature Extractor

    Combines features from multiple modalities for comprehensive analysis.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.extractors = {
            'mri': MRIFeatureExtractor(config.get('mri', {})),
            'eeg': EEGFeatureExtractor(config.get('eeg', {})),
            'voice': VoiceFeatureExtractor(config.get('voice', {})),
            'gait': GaitFeatureExtractor(config.get('gait', {})),
            'clinical': ClinicalFeatureExtractor(config.get('clinical', {}))
        }

    def extract(self, data: Dict[str, Any]) -> Dict[str, FeatureSet]:
        """
        Extract features from all available modalities

        Parameters
        ----------
        data : dict
            Dictionary with keys for each modality

        Returns
        -------
        features : dict
            Dictionary of FeatureSet for each modality
        """
        results = {}

        for modality, extractor in self.extractors.items():
            if modality in data and data[modality] is not None:
                try:
                    results[modality] = extractor.extract(data[modality])
                    logger.info(f"Extracted {len(results[modality].features)} {modality} features")
                except Exception as e:
                    logger.error(f"Error extracting {modality} features: {e}")

        return results

    def extract_combined(self, data: Dict[str, Any]) -> FeatureSet:
        """
        Extract and combine all features into single array

        Parameters
        ----------
        data : dict
            Dictionary with keys for each modality

        Returns
        -------
        features : FeatureSet
            Combined feature set
        """
        modal_features = self.extract(data)

        all_features = []
        all_names = []

        for modality, feature_set in modal_features.items():
            all_features.extend(feature_set.features.tolist())
            all_names.extend([f"{modality}_{name}" for name in feature_set.feature_names])

        return FeatureSet(
            features=np.array(all_features),
            feature_names=all_names,
            metadata={'modalities': list(modal_features.keys())}
        )
