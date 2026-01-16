# Alzheimer Sample Dataset

## Overview
- **Disease**: Alzheimer
- **Samples**: 100 (50 disease + 50 control)
- **Features**: 47 EEG features
- **Subjects**: 10 per class

## Files
- `alzheimer_sample_100.csv` - CSV format with headers
- `alzheimer_sample_100.npz` - NumPy compressed format

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
df = pd.read_csv('alzheimer_sample_100.csv')

# Load NPZ
data = np.load('alzheimer_sample_100.npz')
X, y = data['X'], data['y']
```

## Citation
If you use this data, please cite our paper.
