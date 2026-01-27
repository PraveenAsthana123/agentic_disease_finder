#!/usr/bin/env python
"""
Template for Loading Real EEG Datasets
Modify this template based on the specific dataset format.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import mne  # For EDF/BDF files
from scipy.signal import welch
from scipy.stats import skew, kurtosis

def extract_features(eeg_data, sfreq=256):
    """Extract features from raw EEG data."""
    features = {}

    # Time domain features
    features['mean'] = np.mean(eeg_data)
    features['std'] = np.std(eeg_data)
    features['var'] = np.var(eeg_data)
    features['min'] = np.min(eeg_data)
    features['max'] = np.max(eeg_data)
    features['median'] = np.median(eeg_data)
    features['ptp'] = np.ptp(eeg_data)
    features['skewness'] = skew(eeg_data)
    features['kurtosis'] = kurtosis(eeg_data)
    features['rms'] = np.sqrt(np.mean(eeg_data**2))

    # Frequency domain features
    freqs, psd = welch(eeg_data, fs=sfreq, nperseg=min(256, len(eeg_data)))

    # Band powers
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs < 30)
    gamma_mask = (freqs >= 30) & (freqs < 100)

    features['delta_power'] = np.sum(psd[delta_mask])
    features['theta_power'] = np.sum(psd[theta_mask])
    features['alpha_power'] = np.sum(psd[alpha_mask])
    features['beta_power'] = np.sum(psd[beta_mask])
    features['gamma_power'] = np.sum(psd[gamma_mask])
    features['total_power'] = np.sum(psd)

    # Spectral features
    features['spectral_entropy'] = -np.sum(psd * np.log2(psd + 1e-10))
    features['dominant_freq'] = freqs[np.argmax(psd)]

    return features


def load_edf_file(edf_path):
    """Load EEG data from EDF file."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    return data, sfreq


def load_csv_eeg(csv_path):
    """Load EEG data from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def process_dataset(data_dir, output_path, label_func):
    """Process a dataset and extract features."""
    data_dir = Path(data_dir)
    all_features = []

    for file_path in data_dir.glob("**/*.edf"):
        try:
            data, sfreq = load_edf_file(file_path)

            # Extract features for each channel and average
            channel_features = []
            for ch in range(data.shape[0]):
                features = extract_features(data[ch], sfreq)
                channel_features.append(features)

            # Average across channels
            avg_features = {}
            for key in channel_features[0].keys():
                avg_features[key] = np.mean([cf[key] for cf in channel_features])

            # Add label
            avg_features['label'] = label_func(file_path)
            avg_features['subject_id'] = file_path.stem

            all_features.append(avg_features)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

    return df


# Example usage:
# process_dataset(
#     data_dir="data/real_eeg/epilepsy/chbmit",
#     output_path="data/epilepsy/real/epilepsy_real.csv",
#     label_func=lambda p: 1 if "seizure" in str(p).lower() else 0
# )
