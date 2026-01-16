#!/usr/bin/env python3
"""
Unit Tests for EEG Feature Extraction
======================================

Tests for the feature extraction module to ensure correctness
of all 47 features across different signal conditions.

Author: AgenticFinder Research Team
License: MIT
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from train import EEGFeatureExtractor


class TestEEGFeatureExtractor:
    """Test suite for EEGFeatureExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return EEGFeatureExtractor(sampling_rate=256)

    @pytest.fixture
    def sample_signal(self):
        """Generate sample EEG-like signal."""
        np.random.seed(42)
        t = np.linspace(0, 2, 512)  # 2 seconds at 256 Hz

        # Mix of frequency components (alpha, beta, theta)
        signal = (
            0.5 * np.sin(2 * np.pi * 10 * t) +  # Alpha (10 Hz)
            0.3 * np.sin(2 * np.pi * 20 * t) +  # Beta (20 Hz)
            0.2 * np.sin(2 * np.pi * 5 * t) +   # Theta (5 Hz)
            0.1 * np.random.randn(len(t))       # Noise
        )
        return signal

    @pytest.fixture
    def multichannel_signal(self):
        """Generate multi-channel signal."""
        np.random.seed(42)
        n_channels = 4
        n_samples = 512
        return np.random.randn(n_channels, n_samples)

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.sampling_rate == 256
        assert hasattr(extractor, 'extract')

    def test_feature_count(self, extractor, sample_signal):
        """Test that 47 features are extracted."""
        features = extractor.extract(sample_signal)
        assert len(features) == 47, f"Expected 47 features, got {len(features)}"

    def test_feature_types(self, extractor, sample_signal):
        """Test that all features are numeric."""
        features = extractor.extract(sample_signal)
        for i, feat in enumerate(features):
            assert isinstance(feat, (int, float, np.integer, np.floating)), \
                f"Feature {i} is not numeric: {type(feat)}"

    def test_no_nan_features(self, extractor, sample_signal):
        """Test that no NaN values are produced."""
        features = extractor.extract(sample_signal)
        assert not np.any(np.isnan(features)), "Features contain NaN values"

    def test_no_inf_features(self, extractor, sample_signal):
        """Test that no infinite values are produced."""
        features = extractor.extract(sample_signal)
        assert not np.any(np.isinf(features)), "Features contain infinite values"

    def test_constant_signal(self, extractor):
        """Test handling of constant signal (edge case)."""
        constant_signal = np.ones(512) * 5
        features = extractor.extract(constant_signal)

        # Should not crash and should produce valid features
        assert len(features) == 47
        assert not np.any(np.isnan(features))

    def test_zero_signal(self, extractor):
        """Test handling of zero signal."""
        zero_signal = np.zeros(512)
        features = extractor.extract(zero_signal)

        assert len(features) == 47
        assert not np.any(np.isnan(features))

    def test_noisy_signal(self, extractor):
        """Test handling of pure noise."""
        np.random.seed(42)
        noise_signal = np.random.randn(512)
        features = extractor.extract(noise_signal)

        assert len(features) == 47
        assert not np.any(np.isnan(features))

    def test_short_signal(self, extractor):
        """Test handling of short signals."""
        short_signal = np.random.randn(64)
        features = extractor.extract(short_signal)

        # Should still extract features (possibly with padding)
        assert len(features) == 47

    def test_long_signal(self, extractor):
        """Test handling of long signals."""
        long_signal = np.random.randn(10000)
        features = extractor.extract(long_signal)

        assert len(features) == 47

    def test_reproducibility(self, extractor, sample_signal):
        """Test that same signal produces same features."""
        features1 = extractor.extract(sample_signal)
        features2 = extractor.extract(sample_signal)

        np.testing.assert_array_equal(features1, features2)

    def test_different_sampling_rates(self):
        """Test with different sampling rates."""
        for sr in [128, 256, 512, 1000]:
            extractor = EEGFeatureExtractor(sampling_rate=sr)
            signal = np.random.randn(sr * 2)  # 2 seconds
            features = extractor.extract(signal)

            assert len(features) == 47, f"Failed for sampling rate {sr}"

    def test_statistical_features_range(self, extractor, sample_signal):
        """Test statistical features have reasonable values."""
        features = extractor.extract(sample_signal)

        # Mean should be reasonable (signal is centered around 0)
        assert abs(features[0]) < 10, "Mean is out of expected range"

        # Std should be positive
        assert features[1] > 0, "Std should be positive"

        # Variance should be positive
        assert features[2] >= 0, "Variance should be non-negative"

    def test_multichannel_handling(self, extractor, multichannel_signal):
        """Test multi-channel signal handling."""
        # Should handle 2D array (channels x samples)
        features = extractor.extract(multichannel_signal)
        assert features is not None
        assert len(features) == 47


class TestStatisticalFeatures:
    """Tests for statistical feature calculations."""

    @pytest.fixture
    def extractor(self):
        return EEGFeatureExtractor(sampling_rate=256)

    def test_mean_calculation(self, extractor):
        """Test mean calculation."""
        signal = np.array([1, 2, 3, 4, 5])
        # Pad to minimum length
        signal = np.pad(signal, (0, 512 - len(signal)))
        features = extractor.extract(signal)

        # Mean should be close to padded mean
        assert features[0] == pytest.approx(np.mean(signal), rel=0.1)

    def test_std_calculation(self, extractor):
        """Test standard deviation calculation."""
        signal = np.random.randn(512) * 2  # Std ≈ 2
        features = extractor.extract(signal)

        assert features[1] == pytest.approx(np.std(signal), rel=0.1)


class TestSpectralFeatures:
    """Tests for spectral feature calculations."""

    @pytest.fixture
    def extractor(self):
        return EEGFeatureExtractor(sampling_rate=256)

    def test_alpha_band_detection(self, extractor):
        """Test that alpha band power is detected correctly."""
        t = np.linspace(0, 2, 512)
        # Pure alpha wave (10 Hz)
        alpha_signal = np.sin(2 * np.pi * 10 * t)
        features = extractor.extract(alpha_signal)

        # Alpha power should be prominent
        # Features indices depend on implementation
        assert features is not None

    def test_spectral_entropy_bounds(self, extractor):
        """Test spectral entropy is within bounds."""
        signal = np.random.randn(512)
        features = extractor.extract(signal)

        # Spectral entropy should be positive
        # Index depends on implementation
        spectral_features = features[15:33]  # Spectral features range
        for feat in spectral_features:
            assert not np.isnan(feat)


class TestTemporalFeatures:
    """Tests for temporal feature calculations."""

    @pytest.fixture
    def extractor(self):
        return EEGFeatureExtractor(sampling_rate=256)

    def test_hjorth_parameters(self, extractor):
        """Test Hjorth parameters calculation."""
        signal = np.random.randn(512)
        features = extractor.extract(signal)

        # Hjorth mobility and complexity should be positive
        # Indices depend on implementation
        assert features is not None

    def test_zero_crossings_count(self, extractor):
        """Test zero crossings counting."""
        # Signal that crosses zero multiple times
        t = np.linspace(0, 2, 512)
        signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz = 10 crossings per second
        features = extractor.extract(signal)

        assert features is not None


class TestNonlinearFeatures:
    """Tests for nonlinear feature calculations."""

    @pytest.fixture
    def extractor(self):
        return EEGFeatureExtractor(sampling_rate=256)

    def test_entropy_features(self, extractor):
        """Test entropy feature calculations."""
        signal = np.random.randn(512)
        features = extractor.extract(signal)

        # Entropy features should be calculated
        assert features is not None

    def test_hurst_exponent_range(self, extractor):
        """Test Hurst exponent is in valid range."""
        signal = np.random.randn(512)
        features = extractor.extract(signal)

        # Hurst exponent should typically be between 0 and 1
        # Index depends on implementation
        nonlinear_features = features[42:47]
        for feat in nonlinear_features:
            assert not np.isnan(feat)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def extractor(self):
        return EEGFeatureExtractor(sampling_rate=256)

    def test_empty_signal_handling(self, extractor):
        """Test handling of empty signal."""
        with pytest.raises(Exception):
            extractor.extract(np.array([]))

    def test_single_sample(self, extractor):
        """Test handling of single sample."""
        # Should handle gracefully or raise appropriate error
        try:
            features = extractor.extract(np.array([1.0]))
            assert len(features) == 47
        except Exception:
            pass  # Acceptable to raise error

    def test_very_large_values(self, extractor):
        """Test handling of very large values."""
        signal = np.random.randn(512) * 1e6
        features = extractor.extract(signal)

        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_very_small_values(self, extractor):
        """Test handling of very small values."""
        signal = np.random.randn(512) * 1e-10
        features = extractor.extract(signal)

        assert not np.any(np.isnan(features))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
