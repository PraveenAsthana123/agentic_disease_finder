#!/usr/bin/env python3
"""
Create Sample Datasets for GitHub
===================================

Generates 100-row sample datasets for each of the 7 diseases
for demonstration and testing purposes.

Author: AgenticFinder Research Team
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

import numpy as np
import pandas as pd
from generate_sample_data import EEGDataGenerator, extract_simple_features

# Diseases and their configurations
DISEASES = {
    'parkinson': {'subjects': 10, 'samples_per_subject': 5},
    'epilepsy': {'subjects': 10, 'samples_per_subject': 5},
    'autism': {'subjects': 10, 'samples_per_subject': 5},
    'schizophrenia': {'subjects': 10, 'samples_per_subject': 5},
    'stress': {'subjects': 10, 'samples_per_subject': 5},
    'alzheimer': {'subjects': 10, 'samples_per_subject': 5},
    'depression': {'subjects': 10, 'samples_per_subject': 5}
}

# Feature names
FEATURE_NAMES = [
    # Statistical (15)
    'mean', 'std', 'var', 'min', 'max', 'median', 'ptp', 'skewness', 'kurtosis',
    'q25', 'q75', 'rms', 'mav', 'line_length', 'zero_crossings',
    # Spectral (18)
    'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
    'total_power', 'dominant_freq', 'spectral_entropy', 'psd_std', 'psd_mean',
    'psd_median', 'psd_q10', 'psd_q90', 'peak_ratio', 'spectral_flatness', 'spectral_centroid',
    'spectral_bandwidth', 'spectral_rolloff',
    # Temporal (9)
    'mean_abs_diff', 'std_diff', 'max_diff', 'hjorth_mobility', 'hjorth_complexity',
    'autocorr', 'slope_changes', 'trend', 'crest_factor',
    # Nonlinear (5)
    'approx_entropy', 'sample_entropy', 'hurst_exponent', 'dfa_alpha', 'lz_complexity'
]


def create_sample_dataset(disease: str, output_dir: Path, seed: int = 42):
    """Create sample dataset for a disease."""
    print(f"\nGenerating {disease} dataset...")

    config = DISEASES[disease]
    generator = EEGDataGenerator(sampling_rate=256, seed=seed)

    # Generate data
    data = generator.generate_dataset(
        disease=disease,
        n_subjects=config['subjects'],
        samples_per_subject=config['samples_per_subject'],
        duration=2.0,
        include_controls=True
    )

    # Extract features
    X = extract_simple_features(data['signals'], 256)
    y = data['labels']
    subject_ids = data['subject_ids']

    # Limit to 100 samples (50 disease + 50 control)
    n_samples = min(100, len(y))

    # Create balanced subset
    disease_idx = np.where(y == 1)[0][:50]
    control_idx = np.where(y == 0)[0][:50]
    selected_idx = np.concatenate([disease_idx, control_idx])
    np.random.shuffle(selected_idx)

    X_subset = X[selected_idx]
    y_subset = y[selected_idx]
    subject_subset = subject_ids[selected_idx]

    # Create DataFrame
    df = pd.DataFrame(X_subset, columns=FEATURE_NAMES[:47])
    df['label'] = y_subset
    df['subject_id'] = subject_subset
    df['class'] = df['label'].map({0: 'Control', 1: disease.capitalize()})

    # Save to CSV
    csv_path = output_dir / disease / 'sample' / f'{disease}_sample_100.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Samples: {len(df)} (Control: {sum(y_subset==0)}, Disease: {sum(y_subset==1)})")

    # Save NPZ format
    npz_path = output_dir / disease / 'sample' / f'{disease}_sample_100.npz'
    np.savez(
        npz_path,
        X=X_subset,
        y=y_subset,
        subject_ids=subject_subset,
        feature_names=FEATURE_NAMES[:47],
        class_names=['Control', disease.capitalize()]
    )
    print(f"  Saved: {npz_path}")

    # Create README
    readme_path = output_dir / disease / 'sample' / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"""# {disease.capitalize()} Sample Dataset

## Overview
- **Disease**: {disease.capitalize()}
- **Samples**: 100 (50 disease + 50 control)
- **Features**: 47 EEG features
- **Subjects**: {config['subjects']} per class

## Files
- `{disease}_sample_100.csv` - CSV format with headers
- `{disease}_sample_100.npz` - NumPy compressed format

## Feature Categories
1. **Statistical (15)**: mean, std, var, min, max, median, etc.
2. **Spectral (18)**: band powers (δ,θ,α,β,γ), entropy, etc.
3. **Temporal (9)**: Hjorth parameters, zero crossings, etc.
4. **Nonlinear (5)**: entropy measures, Hurst exponent, etc.

## Usage
```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('{disease}_sample_100.csv')

# Load NPZ
data = np.load('{disease}_sample_100.npz')
X, y = data['X'], data['y']
```

## Citation
If you use this data, please cite our paper.
""")
    print(f"  Saved: {readme_path}")

    return df


def main():
    print("=" * 60)
    print(" Creating Sample Datasets for GitHub")
    print("=" * 60)

    output_dir = Path('/media/praveen/Asthana3/rajveer/agenticfinder/data')

    all_stats = []

    for disease in DISEASES.keys():
        df = create_sample_dataset(disease, output_dir)
        all_stats.append({
            'disease': disease,
            'samples': len(df),
            'control': sum(df['label'] == 0),
            'disease_count': sum(df['label'] == 1)
        })

    # Summary
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)

    summary_df = pd.DataFrame(all_stats)
    print(summary_df.to_string(index=False))

    print(f"\nTotal: {summary_df['samples'].sum()} samples across 7 diseases")
    print("\nAll datasets ready for GitHub push!")


if __name__ == '__main__':
    main()
