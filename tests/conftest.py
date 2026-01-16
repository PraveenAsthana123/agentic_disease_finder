#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures
===================================

Shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))


@pytest.fixture(scope='session')
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed):
    """Set random seed before each test."""
    np.random.seed(random_seed)


@pytest.fixture(scope='session')
def sample_eeg_signal():
    """Generate sample EEG-like signal for testing."""
    np.random.seed(42)
    t = np.linspace(0, 2, 512)  # 2 seconds at 256 Hz

    signal = (
        0.5 * np.sin(2 * np.pi * 10 * t) +  # Alpha
        0.3 * np.sin(2 * np.pi * 20 * t) +  # Beta
        0.2 * np.sin(2 * np.pi * 5 * t) +   # Theta
        0.1 * np.random.randn(len(t))       # Noise
    )
    return signal


@pytest.fixture(scope='session')
def sample_features():
    """Generate sample feature matrix."""
    np.random.seed(42)
    return np.random.randn(100, 47)


@pytest.fixture(scope='session')
def sample_labels():
    """Generate sample binary labels."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


@pytest.fixture(scope='session')
def sample_subject_ids():
    """Generate sample subject IDs."""
    np.random.seed(42)
    return np.repeat(np.arange(10), 10)


@pytest.fixture
def multichannel_eeg():
    """Generate multi-channel EEG signal."""
    np.random.seed(42)
    return np.random.randn(4, 512)


# Test markers
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
