#!/usr/bin/env python3
"""
Sample Data Generator for EEG Classification
=============================================

Generates synthetic EEG-like data for testing and demonstration purposes.
Produces realistic EEG signals with disease-specific characteristics.

Author: AgenticFinder Research Team
License: MIT
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EEGDataGenerator:
    """
    Generates synthetic EEG data with disease-specific characteristics.

    Each disease has different spectral profiles based on clinical literature:
    - Parkinson's: Increased beta, reduced alpha
    - Epilepsy: Abnormal spike patterns
    - Autism: Altered gamma and alpha asymmetry
    - Schizophrenia: Reduced alpha, increased delta/theta
    - Stress: Increased beta and gamma
    - Alzheimer's: Increased delta/theta, reduced alpha/beta
    - Depression: Alpha asymmetry, increased theta
    """

    # Disease-specific spectral profiles (relative band powers)
    DISEASE_PROFILES = {
        'parkinson': {
            'delta': 0.15, 'theta': 0.15, 'alpha': 0.20,
            'beta': 0.35, 'gamma': 0.15, 'noise': 0.10
        },
        'epilepsy': {
            'delta': 0.25, 'theta': 0.20, 'alpha': 0.15,
            'beta': 0.15, 'gamma': 0.10, 'noise': 0.15,
            'spike_prob': 0.3
        },
        'autism': {
            'delta': 0.15, 'theta': 0.20, 'alpha': 0.25,
            'beta': 0.20, 'gamma': 0.15, 'noise': 0.05
        },
        'schizophrenia': {
            'delta': 0.25, 'theta': 0.25, 'alpha': 0.15,
            'beta': 0.15, 'gamma': 0.10, 'noise': 0.10
        },
        'stress': {
            'delta': 0.10, 'theta': 0.15, 'alpha': 0.20,
            'beta': 0.30, 'gamma': 0.20, 'noise': 0.05
        },
        'alzheimer': {
            'delta': 0.30, 'theta': 0.25, 'alpha': 0.15,
            'beta': 0.10, 'gamma': 0.05, 'noise': 0.15
        },
        'depression': {
            'delta': 0.20, 'theta': 0.25, 'alpha': 0.25,
            'beta': 0.15, 'gamma': 0.05, 'noise': 0.10
        },
        'control': {
            'delta': 0.15, 'theta': 0.15, 'alpha': 0.35,
            'beta': 0.20, 'gamma': 0.10, 'noise': 0.05
        }
    }

    # Frequency bands (Hz)
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    def __init__(self, sampling_rate: int = 256, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            sampling_rate: Sampling rate in Hz
            seed: Random seed for reproducibility
        """
        self.sampling_rate = sampling_rate
        if seed is not None:
            np.random.seed(seed)

    def generate_band_signal(self, duration: float, freq_range: Tuple[float, float],
                            amplitude: float = 1.0) -> np.ndarray:
        """
        Generate signal in a specific frequency band.

        Args:
            duration: Signal duration in seconds
            freq_range: Frequency range (low, high) in Hz
            amplitude: Signal amplitude

        Returns:
            Signal array
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, n_samples)

        # Mix of frequencies in the band
        low, high = freq_range
        n_components = 5
        freqs = np.linspace(low, high, n_components)

        signal = np.zeros(n_samples)
        for freq in freqs:
            phase = np.random.uniform(0, 2 * np.pi)
            signal += np.sin(2 * np.pi * freq * t + phase)

        return amplitude * signal / n_components

    def generate_spike(self, n_samples: int, position: int,
                       amplitude: float = 3.0, width: int = 20) -> np.ndarray:
        """
        Generate epileptic-like spike.

        Args:
            n_samples: Total signal length
            position: Spike position
            amplitude: Spike amplitude
            width: Spike width in samples

        Returns:
            Spike signal
        """
        signal = np.zeros(n_samples)
        half_width = width // 2

        start = max(0, position - half_width)
        end = min(n_samples, position + half_width)

        # Gaussian-like spike
        x = np.arange(start, end) - position
        signal[start:end] = amplitude * np.exp(-0.5 * (x / (width/4))**2)

        return signal

    def generate_eeg_signal(self, duration: float, disease: str = 'control',
                           n_channels: int = 1) -> np.ndarray:
        """
        Generate synthetic EEG signal with disease characteristics.

        Args:
            duration: Signal duration in seconds
            disease: Disease type
            n_channels: Number of channels

        Returns:
            EEG signal array (n_channels x n_samples) or (n_samples,) if single channel
        """
        disease = disease.lower()
        if disease not in self.DISEASE_PROFILES:
            logger.warning(f"Unknown disease '{disease}', using control profile")
            disease = 'control'

        profile = self.DISEASE_PROFILES[disease]
        n_samples = int(duration * self.sampling_rate)

        signals = []
        for ch in range(n_channels):
            signal = np.zeros(n_samples)

            # Add each frequency band
            for band_name, freq_range in self.BANDS.items():
                amplitude = profile.get(band_name, 0.1)
                band_signal = self.generate_band_signal(duration, freq_range, amplitude)
                signal += band_signal

            # Add noise
            noise_level = profile.get('noise', 0.1)
            signal += noise_level * np.random.randn(n_samples)

            # Add spikes for epilepsy
            if disease == 'epilepsy' and 'spike_prob' in profile:
                n_spikes = np.random.poisson(profile['spike_prob'] * duration)
                for _ in range(n_spikes):
                    pos = np.random.randint(0, n_samples)
                    signal += self.generate_spike(n_samples, pos)

            # Normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            signals.append(signal)

        if n_channels == 1:
            return signals[0]
        return np.array(signals)

    def generate_dataset(self, disease: str, n_subjects: int = 20,
                        samples_per_subject: int = 10,
                        duration: float = 2.0,
                        n_channels: int = 1,
                        include_controls: bool = True) -> Dict:
        """
        Generate complete dataset for a disease.

        Args:
            disease: Disease type
            n_subjects: Number of subjects per class
            samples_per_subject: Samples per subject
            duration: Signal duration per sample
            n_channels: Number of EEG channels
            include_controls: Whether to include control group

        Returns:
            Dictionary with signals, labels, and subject IDs
        """
        all_signals = []
        all_labels = []
        subject_ids = []

        subject_id = 0

        # Disease samples
        for subj in range(n_subjects):
            for sample in range(samples_per_subject):
                signal = self.generate_eeg_signal(duration, disease, n_channels)
                all_signals.append(signal)
                all_labels.append(1)  # Disease = 1
                subject_ids.append(subject_id)
            subject_id += 1

        # Control samples
        if include_controls:
            for subj in range(n_subjects):
                for sample in range(samples_per_subject):
                    signal = self.generate_eeg_signal(duration, 'control', n_channels)
                    all_signals.append(signal)
                    all_labels.append(0)  # Control = 0
                    subject_ids.append(subject_id)
                subject_id += 1

        return {
            'signals': np.array(all_signals),
            'labels': np.array(all_labels),
            'subject_ids': np.array(subject_ids),
            'disease': disease,
            'sampling_rate': self.sampling_rate,
            'duration': duration,
            'n_channels': n_channels
        }


def extract_simple_features(signals: np.ndarray, sampling_rate: int = 256) -> np.ndarray:
    """
    Extract simple features from EEG signals.

    Args:
        signals: Array of signals (n_samples x n_timepoints) or (n_samples x n_channels x n_timepoints)
        sampling_rate: Sampling rate in Hz

    Returns:
        Feature matrix (n_samples x n_features)
    """
    n_samples = len(signals)
    features_list = []

    for i, signal in enumerate(signals):
        if signal.ndim == 2:
            signal = signal.flatten()

        features = []

        # Statistical features (15)
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.var(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.median(signal))
        features.append(np.ptp(signal))  # Peak-to-peak

        # Skewness
        n = len(signal)
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            skew = np.sum((signal - mean)**3) / (n * std**3)
        else:
            skew = 0
        features.append(skew)

        # Kurtosis
        if std > 0:
            kurt = np.sum((signal - mean)**4) / (n * std**4) - 3
        else:
            kurt = 0
        features.append(kurt)

        features.append(np.percentile(signal, 25))
        features.append(np.percentile(signal, 75))
        features.append(np.sqrt(np.mean(signal**2)))  # RMS
        features.append(np.mean(np.abs(signal)))  # MAV
        features.append(np.sum(np.abs(np.diff(signal))))  # Line length
        features.append(len(np.where(np.diff(np.sign(signal)))[0]))  # Zero crossings

        # Spectral features (18)
        fft = np.fft.fft(signal)
        psd = np.abs(fft)**2 / len(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)

        # Band powers
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        for (low, high) in bands:
            band_mask = (freqs >= low) & (freqs < high)
            band_power = np.sum(psd[band_mask])
            features.append(band_power)

        # Total power
        total_power = np.sum(psd[:len(psd)//2])
        features.append(total_power)

        # Dominant frequency
        dom_freq = np.argmax(psd[:len(psd)//2]) * sampling_rate / len(signal)
        features.append(dom_freq)

        # Spectral entropy
        psd_norm = psd[:len(psd)//2]
        psd_norm = psd_norm / (np.sum(psd_norm) + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features.append(spec_entropy)

        # More spectral stats
        features.append(np.std(psd[:len(psd)//2]))
        features.append(np.mean(psd[:len(psd)//2]))

        # Additional spectral features to reach 18
        features.extend([
            np.median(psd[:len(psd)//2]),
            np.percentile(psd[:len(psd)//2], 10),
            np.percentile(psd[:len(psd)//2], 90),
            np.max(psd[:len(psd)//2]) / (total_power + 1e-10),
            np.sum(psd[:len(psd)//2] > np.mean(psd[:len(psd)//2])),
            np.argmax(psd[:len(psd)//2]) / len(psd) * 2
        ])

        # Temporal features (9)
        features.append(np.mean(np.abs(np.diff(signal))))
        features.append(np.std(np.diff(signal)))
        features.append(np.max(np.abs(np.diff(signal))))

        # Hjorth mobility
        diff_signal = np.diff(signal)
        var_signal = np.var(signal)
        var_diff = np.var(diff_signal)
        mobility = np.sqrt(var_diff / (var_signal + 1e-10))
        features.append(mobility)

        # Hjorth complexity
        diff2_signal = np.diff(diff_signal)
        var_diff2 = np.var(diff2_signal)
        mobility2 = np.sqrt(var_diff2 / (var_diff + 1e-10))
        complexity = mobility2 / (mobility + 1e-10)
        features.append(complexity)

        # Autocorrelation
        features.append(np.correlate(signal, signal)[0] / len(signal))

        # Slope changes
        features.append(np.sum(np.diff(np.sign(np.diff(signal))) != 0))

        # Trend
        half = len(signal) // 2
        trend = np.mean(signal[:half]) - np.mean(signal[half:])
        features.append(trend)

        # Crest factor
        crest = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10)
        features.append(crest)

        # Nonlinear features (5) - simplified approximations
        # Approximate entropy (simplified)
        features.append(spec_entropy * 0.1)  # Proxy

        # Sample entropy (simplified)
        features.append(spec_entropy * 0.11)  # Proxy

        # Hurst exponent approximation
        n = len(signal)
        rs = (np.max(signal) - np.min(signal)) / (np.std(signal) + 1e-10)
        hurst = np.log(rs) / np.log(n) if n > 1 else 0.5
        hurst = np.clip(hurst, 0, 1)
        features.append(hurst)

        # DFA approximation
        features.append(hurst * 0.9)  # Proxy

        # LZ complexity approximation
        binary = (signal > np.median(signal)).astype(int)
        changes = np.sum(np.diff(binary) != 0)
        lzc = changes / len(signal)
        features.append(lzc)

        # Ensure exactly 47 features
        features = features[:47]
        while len(features) < 47:
            features.append(0.0)

        features_list.append(features)

    return np.array(features_list)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic EEG data for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--disease', '-d',
        type=str,
        default='parkinson',
        choices=['parkinson', 'epilepsy', 'autism', 'schizophrenia',
                'stress', 'alzheimer', 'depression', 'all'],
        help='Disease type to generate'
    )

    parser.add_argument(
        '--subjects', '-n',
        type=int,
        default=20,
        help='Number of subjects per class'
    )

    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=10,
        help='Samples per subject'
    )

    parser.add_argument(
        '--duration', '-t',
        type=float,
        default=2.0,
        help='Signal duration in seconds'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/sample',
        help='Output directory'
    )

    parser.add_argument(
        '--features',
        action='store_true',
        help='Also extract and save features'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = EEGDataGenerator(sampling_rate=256, seed=args.seed)

    # Diseases to generate
    if args.disease == 'all':
        diseases = ['parkinson', 'epilepsy', 'autism', 'schizophrenia',
                   'stress', 'alzheimer', 'depression']
    else:
        diseases = [args.disease]

    for disease in diseases:
        logger.info(f"Generating data for {disease}...")

        # Generate dataset
        data = generator.generate_dataset(
            disease=disease,
            n_subjects=args.subjects,
            samples_per_subject=args.samples,
            duration=args.duration,
            include_controls=True
        )

        # Save raw signals
        signals_file = output_dir / f"{disease}_signals.npz"
        np.savez(
            signals_file,
            signals=data['signals'],
            labels=data['labels'],
            subject_ids=data['subject_ids'],
            sampling_rate=data['sampling_rate'],
            class_names=['Control', disease.capitalize()]
        )
        logger.info(f"Saved signals to {signals_file}")

        # Extract and save features if requested
        if args.features:
            logger.info("Extracting features...")
            features = extract_simple_features(data['signals'], data['sampling_rate'])

            features_file = output_dir / f"{disease}_features.npz"
            np.savez(
                features_file,
                X=features,
                y=data['labels'],
                subject_ids=data['subject_ids'],
                class_names=['Control', disease.capitalize()]
            )
            logger.info(f"Saved features to {features_file}")

        # Print summary
        print(f"\n{disease.upper()} Dataset Summary:")
        print(f"  Total samples: {len(data['labels'])}")
        print(f"  Control samples: {np.sum(data['labels'] == 0)}")
        print(f"  Disease samples: {np.sum(data['labels'] == 1)}")
        print(f"  Unique subjects: {len(np.unique(data['subject_ids']))}")
        print(f"  Signal shape: {data['signals'][0].shape}")

    logger.info(f"\nAll data saved to {output_dir}")


if __name__ == '__main__':
    main()
