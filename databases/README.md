# EEG Neurological Disease Databases

This directory contains sample EEG feature datasets for 7 neurological diseases. Each dataset includes 50 samples with 47 extracted EEG features.

## Database Structure

```
databases/
├── parkinson/       # Parkinson's Disease EEG data
├── epilepsy/        # Epilepsy seizure detection data
├── autism/          # Autism Spectrum Disorder data
├── schizophrenia/   # Schizophrenia EEG patterns
├── stress/          # Psychological stress data
├── alzheimer/       # Alzheimer's Disease data
└── depression/      # Depression EEG markers
```

## Feature Description

Each dataset contains 47 EEG features across 5 categories:

### 1. Statistical Features (12)
- `mean`, `std`, `var`, `min`, `max`, `median`
- `ptp` (peak-to-peak), `skewness`, `kurtosis`
- `q25`, `q75`, `rms` (root mean square)

### 2. Time-Domain Features (5)
- `mav` (mean absolute value)
- `line_length` (signal complexity)
- `zero_crossings`

### 3. Spectral Features (15)
- Band powers: `delta_power`, `theta_power`, `alpha_power`, `beta_power`, `gamma_power`
- `total_power`, `dominant_freq`, `spectral_entropy`
- PSD statistics: `psd_std`, `psd_mean`, `psd_median`, `psd_q10`, `psd_q90`
- `peak_ratio`, `spectral_flatness`, `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`

### 4. Temporal Features (7)
- `mean_abs_diff`, `std_diff`, `max_diff`
- Hjorth parameters: `hjorth_mobility`, `hjorth_complexity`
- `autocorr`, `slope_changes`, `trend`, `crest_factor`

### 5. Nonlinear Features (5)
- `approx_entropy` (signal regularity)
- `sample_entropy` (complexity)
- `hurst_exponent` (long-term memory)
- `dfa_alpha` (fractal scaling)
- `lz_complexity` (Lempel-Ziv complexity)

## Labels

- `label`: Binary label (0 = Control, 1 = Disease)
- `subject_id`: Unique subject identifier
- `class`: Class name (Control / Disease name)

## Data Sources

| Disease | Reference Dataset | Subjects | Original Source |
|---------|-------------------|----------|-----------------|
| Parkinson's | PPMI | 31 | ppmi-info.org |
| Epilepsy | CHB-MIT | 24 | physionet.org |
| Autism | ABIDE | 39 | fcon_1000.projects.nitrc.org |
| Schizophrenia | COBRE | 28 | coins.mrn.org |
| Stress | DEAP | 36 | eecs.qmul.ac.uk/mmv/datasets/deap |
| Alzheimer's | ADNI | 88 | adni.loni.usc.edu |
| Depression | OpenNeuro | 64 | openneuro.org |

## Usage

```python
import pandas as pd

# Load a disease dataset
df = pd.read_csv('databases/parkinson/parkinson_sample_50rows.csv')

# Features
X = df.drop(['label', 'subject_id', 'class'], axis=1)
y = df['label']

print(f"Samples: {len(df)}, Features: {X.shape[1]}")
```

## Citation

If you use these datasets, please cite the original data sources and our framework:

```bibtex
@article{neuromcp2024,
  title={NeuroMCP-Agent: Trustworthy Multi-Agent Deep Learning for EEG-Based Neurological Disease Detection},
  author={Research Team},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2024}
}
```

## License

Sample datasets are provided for research and educational purposes. For full datasets, please refer to the original sources and comply with their licensing terms.
