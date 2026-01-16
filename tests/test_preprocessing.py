"""
Unit Tests for Preprocessing Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocessors import (
    MRIPreprocessor,
    EEGPreprocessor,
    VoicePreprocessor,
    GaitPreprocessor,
    PreprocessingPipeline,
    PreprocessedData
)


class TestMRIPreprocessor:
    """Test MRI preprocessing"""

    @pytest.fixture
    def preprocessor(self):
        return MRIPreprocessor({
            'target_shape': [64, 64, 64],
            'normalize': True,
            'skull_strip': True
        })

    @pytest.fixture
    def sample_mri(self):
        np.random.seed(42)
        # Create MRI-like data with brain structure
        data = np.zeros((80, 80, 80), dtype=np.float32)
        # Add "brain" in center
        data[20:60, 20:60, 20:60] = np.random.rand(40, 40, 40) * 0.8 + 0.2
        return data

    def test_initialization(self, preprocessor):
        assert preprocessor is not None
        assert preprocessor.target_shape == [64, 64, 64]

    def test_preprocess(self, preprocessor, sample_mri):
        result = preprocessor.preprocess(sample_mri)

        assert isinstance(result, PreprocessedData)
        assert result.data.shape == (64, 64, 64)

    def test_intensity_normalization(self, preprocessor, sample_mri):
        result = preprocessor.preprocess(sample_mri)

        # Check normalized data range
        assert result.data.min() >= -5  # Z-score allows negative
        assert result.data.max() <= 5

    def test_quality_metrics(self, preprocessor, sample_mri):
        result = preprocessor.preprocess(sample_mri)

        assert 'snr' in result.quality_metrics
        assert 'contrast' in result.quality_metrics


class TestEEGPreprocessor:
    """Test EEG preprocessing"""

    @pytest.fixture
    def preprocessor(self):
        return EEGPreprocessor({
            'sampling_rate': 256,
            'filter_low': 0.5,
            'filter_high': 45.0
        })

    @pytest.fixture
    def sample_eeg(self):
        np.random.seed(42)
        # 8 channels, 10 seconds
        n_channels = 8
        n_samples = 2560
        # EEG with alpha rhythm + noise
        t = np.linspace(0, 10, n_samples)
        data = np.zeros((n_channels, n_samples))
        for i in range(n_channels):
            # 10 Hz alpha + noise
            data[i] = np.sin(2 * np.pi * 10 * t) * 10e-6 + np.random.randn(n_samples) * 5e-6
        return data

    def test_initialization(self, preprocessor):
        assert preprocessor is not None
        assert preprocessor.sampling_rate == 256

    def test_preprocess(self, preprocessor, sample_eeg):
        result = preprocessor.preprocess(sample_eeg)

        assert isinstance(result, PreprocessedData)
        assert 'num_epochs' in result.quality_metrics

    def test_bandpass_filter(self, preprocessor, sample_eeg):
        result = preprocessor.preprocess(sample_eeg)

        # Verify filtering was applied
        assert result.metadata['filter_band'] == (0.5, 45.0)

    def test_artifact_rejection(self, preprocessor, sample_eeg):
        # Add artificial artifact
        sample_eeg[0, 1000:1100] = 500e-6  # Large amplitude

        result = preprocessor.preprocess(sample_eeg)

        assert 'artifact_rejection_ratio' in result.quality_metrics


class TestVoicePreprocessor:
    """Test Voice preprocessing"""

    @pytest.fixture
    def preprocessor(self):
        return VoicePreprocessor({'sample_rate': 16000})

    @pytest.fixture
    def sample_audio(self):
        np.random.seed(42)
        # 3 seconds at 16kHz
        t = np.linspace(0, 3, 48000)
        # Speech-like signal with fundamental + harmonics
        audio = (np.sin(2 * np.pi * 150 * t) +
                 0.5 * np.sin(2 * np.pi * 300 * t) +
                 0.3 * np.sin(2 * np.pi * 450 * t))
        audio += np.random.randn(len(audio)) * 0.1
        return audio

    def test_initialization(self, preprocessor):
        assert preprocessor is not None
        assert preprocessor.target_sample_rate == 16000

    def test_preprocess(self, preprocessor, sample_audio):
        result = preprocessor.preprocess(sample_audio)

        assert isinstance(result, PreprocessedData)
        assert result.metadata['sample_rate'] == 16000

    def test_normalize(self, preprocessor, sample_audio):
        result = preprocessor.preprocess(sample_audio)

        # Check normalized to [-1, 1]
        assert result.data.min() >= -1.0
        assert result.data.max() <= 1.0

    def test_quality_metrics(self, preprocessor, sample_audio):
        result = preprocessor.preprocess(sample_audio)

        assert 'duration' in result.quality_metrics
        assert 'snr' in result.quality_metrics


class TestGaitPreprocessor:
    """Test Gait preprocessing"""

    @pytest.fixture
    def preprocessor(self):
        return GaitPreprocessor({'sampling_rate': 100})

    @pytest.fixture
    def sample_gait(self):
        np.random.seed(42)
        # 30 seconds at 100 Hz
        n_samples = 3000
        t = np.linspace(0, 30, n_samples)

        # Walking pattern
        gait_freq = 0.9  # Hz
        acc_z = 9.8 + np.sin(2 * np.pi * gait_freq * t) * 2
        acc_x = np.sin(2 * np.pi * gait_freq * t) * 0.5
        acc_y = np.cos(2 * np.pi * gait_freq * t) * 0.3

        return np.column_stack([acc_x, acc_y, acc_z,
                                np.random.randn(n_samples) * 0.1,
                                np.random.randn(n_samples) * 0.1,
                                np.random.randn(n_samples) * 0.1])

    def test_initialization(self, preprocessor):
        assert preprocessor is not None
        assert preprocessor.sampling_rate == 100

    def test_preprocess(self, preprocessor, sample_gait):
        result = preprocessor.preprocess(sample_gait)

        assert isinstance(result, PreprocessedData)
        assert 'num_gait_cycles' in result.quality_metrics

    def test_gait_cycle_detection(self, preprocessor, sample_gait):
        result = preprocessor.preprocess(sample_gait)

        # Should detect ~27 gait cycles in 30 seconds at 0.9 Hz
        assert result.quality_metrics['num_gait_cycles'] > 20


class TestPreprocessingPipeline:
    """Test combined preprocessing pipeline"""

    @pytest.fixture
    def pipeline(self):
        return PreprocessingPipeline()

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return {
            'mri': np.random.rand(64, 64, 64),
            'eeg': np.random.randn(8, 2560) * 10e-6,
            'voice': np.random.randn(48000) * 0.5
        }

    def test_initialization(self, pipeline):
        assert pipeline is not None
        assert 'mri' in pipeline.preprocessors
        assert 'eeg' in pipeline.preprocessors

    def test_preprocess_all(self, pipeline, sample_data):
        results = pipeline.preprocess(sample_data)

        assert 'mri' in results
        assert 'eeg' in results
        assert 'voice' in results

    def test_quality_report(self, pipeline, sample_data):
        results = pipeline.preprocess(sample_data)
        report = pipeline.get_quality_report(results)

        assert 'mri' in report
        assert 'eeg' in report
        assert 'voice' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
