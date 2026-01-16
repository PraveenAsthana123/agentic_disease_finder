"""
Unit Tests for Feature Extraction Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_extractors import (
    MRIFeatureExtractor,
    EEGFeatureExtractor,
    VoiceFeatureExtractor,
    GaitFeatureExtractor,
    ClinicalFeatureExtractor,
    MultiModalFeatureExtractor,
    FeatureSet
)


class TestMRIFeatureExtractor:
    """Test MRI feature extraction"""

    @pytest.fixture
    def extractor(self):
        return MRIFeatureExtractor()

    @pytest.fixture
    def sample_mri(self):
        """Generate sample MRI data"""
        np.random.seed(42)
        return np.random.rand(64, 64, 64).astype(np.float32)

    def test_initialization(self, extractor):
        """Test extractor initialization"""
        assert extractor is not None
        assert len(extractor.feature_names) == 20

    def test_extract_features(self, extractor, sample_mri):
        """Test feature extraction"""
        result = extractor.extract(sample_mri)

        assert isinstance(result, FeatureSet)
        assert len(result.features) == len(extractor.feature_names)
        assert result.metadata['modality'] == 'mri'

    def test_feature_values_valid(self, extractor, sample_mri):
        """Test that extracted features are valid numbers"""
        result = extractor.extract(sample_mri)

        assert not np.any(np.isnan(result.features))
        assert not np.any(np.isinf(result.features))

    def test_to_dict(self, extractor, sample_mri):
        """Test conversion to dictionary"""
        result = extractor.extract(sample_mri)
        feature_dict = result.to_dict()

        assert isinstance(feature_dict, dict)
        assert len(feature_dict) == len(extractor.feature_names)


class TestEEGFeatureExtractor:
    """Test EEG feature extraction"""

    @pytest.fixture
    def extractor(self):
        return EEGFeatureExtractor({'sampling_rate': 256})

    @pytest.fixture
    def sample_eeg(self):
        """Generate sample EEG data"""
        np.random.seed(42)
        # 8 channels, 10 seconds at 256 Hz
        n_channels = 8
        n_samples = 256 * 10
        return np.random.randn(n_channels, n_samples) * 10e-6  # microvolts

    def test_initialization(self, extractor):
        """Test extractor initialization"""
        assert extractor is not None
        assert extractor.sampling_rate == 256
        assert len(extractor.feature_names) > 20

    def test_extract_features(self, extractor, sample_eeg):
        """Test feature extraction"""
        result = extractor.extract(sample_eeg)

        assert isinstance(result, FeatureSet)
        assert len(result.features) == len(extractor.feature_names)
        assert result.metadata['modality'] == 'eeg'

    def test_single_channel(self, extractor):
        """Test with single channel"""
        single_channel = np.random.randn(2560) * 10e-6
        result = extractor.extract(single_channel)

        assert isinstance(result, FeatureSet)

    def test_band_powers_sum(self, extractor, sample_eeg):
        """Test that band powers are positive"""
        result = extractor.extract(sample_eeg)
        # First 5 features are band powers
        band_powers = result.features[:5]

        assert all(p >= 0 for p in band_powers)


class TestVoiceFeatureExtractor:
    """Test Voice feature extraction"""

    @pytest.fixture
    def extractor(self):
        return VoiceFeatureExtractor({'sample_rate': 16000})

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data"""
        np.random.seed(42)
        # 3 seconds at 16kHz with sine wave + noise
        t = np.linspace(0, 3, 48000)
        audio = np.sin(2 * np.pi * 150 * t) * 0.5  # 150 Hz tone
        audio += np.random.randn(len(audio)) * 0.1  # Noise
        return audio

    def test_initialization(self, extractor):
        """Test extractor initialization"""
        assert extractor is not None
        assert extractor.sample_rate == 16000

    def test_extract_features(self, extractor, sample_audio):
        """Test feature extraction"""
        result = extractor.extract(sample_audio)

        assert isinstance(result, FeatureSet)
        assert result.metadata['modality'] == 'voice'

    def test_mfcc_extraction(self, extractor, sample_audio):
        """Test MFCC extraction specifically"""
        result = extractor.extract(sample_audio)

        # Check MFCC features exist
        mfcc_features = [f for f in result.feature_names if 'mfcc' in f]
        assert len(mfcc_features) == 26  # 13 mean + 13 std


class TestGaitFeatureExtractor:
    """Test Gait feature extraction"""

    @pytest.fixture
    def extractor(self):
        return GaitFeatureExtractor({'sampling_rate': 100})

    @pytest.fixture
    def sample_gait(self):
        """Generate sample gait data"""
        np.random.seed(42)
        # 30 seconds at 100 Hz, 6 channels (acc x,y,z + gyro x,y,z)
        n_samples = 3000
        t = np.linspace(0, 30, n_samples)

        # Simulate walking pattern
        gait_freq = 0.9  # ~0.9 Hz walking
        acc_z = 9.8 + np.sin(2 * np.pi * gait_freq * t) * 2  # Vertical acc
        acc_x = np.sin(2 * np.pi * gait_freq * t) * 0.5
        acc_y = np.cos(2 * np.pi * gait_freq * t) * 0.3

        data = np.column_stack([acc_x, acc_y, acc_z,
                                np.zeros(n_samples),
                                np.zeros(n_samples),
                                np.zeros(n_samples)])
        return data

    def test_initialization(self, extractor):
        """Test extractor initialization"""
        assert extractor is not None
        assert extractor.sampling_rate == 100

    def test_extract_features(self, extractor, sample_gait):
        """Test feature extraction"""
        result = extractor.extract(sample_gait)

        assert isinstance(result, FeatureSet)
        assert result.metadata['modality'] == 'gait'


class TestClinicalFeatureExtractor:
    """Test Clinical feature extraction"""

    @pytest.fixture
    def extractor(self):
        return ClinicalFeatureExtractor()

    @pytest.fixture
    def sample_clinical(self):
        """Generate sample clinical data"""
        return {
            'age': 72,
            'sex': 'M',
            'education_years': 16,
            'mmse': 24,
            'moca': 22,
            'cdr_global': 0.5,
            'updrs_total': 25,
            'apoe4_status': True
        }

    def test_initialization(self, extractor):
        """Test extractor initialization"""
        assert extractor is not None

    def test_extract_features(self, extractor, sample_clinical):
        """Test feature extraction"""
        result = extractor.extract(sample_clinical)

        assert isinstance(result, FeatureSet)
        assert result.metadata['modality'] == 'clinical'

    def test_normalized_values(self, extractor, sample_clinical):
        """Test that values are normalized to 0-1 range"""
        result = extractor.extract(sample_clinical)

        # Most clinical features should be normalized
        assert all(0 <= f <= 2 for f in result.features)


class TestMultiModalFeatureExtractor:
    """Test Multi-modal feature extraction"""

    @pytest.fixture
    def extractor(self):
        return MultiModalFeatureExtractor()

    @pytest.fixture
    def sample_multimodal(self):
        """Generate sample multi-modal data"""
        np.random.seed(42)
        return {
            'mri': np.random.rand(64, 64, 64),
            'eeg': np.random.randn(8, 2560) * 10e-6,
            'clinical': {'age': 72, 'mmse': 24}
        }

    def test_initialization(self, extractor):
        """Test extractor initialization"""
        assert extractor is not None
        assert 'mri' in extractor.extractors
        assert 'eeg' in extractor.extractors

    def test_extract_all(self, extractor, sample_multimodal):
        """Test extracting from all modalities"""
        results = extractor.extract(sample_multimodal)

        assert 'mri' in results
        assert 'eeg' in results
        assert 'clinical' in results

    def test_extract_combined(self, extractor, sample_multimodal):
        """Test combined feature extraction"""
        result = extractor.extract_combined(sample_multimodal)

        assert isinstance(result, FeatureSet)
        assert 'modalities' in result.metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
