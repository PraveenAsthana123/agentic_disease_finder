# Data Preprocessing Flow

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA PREPROCESSING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   RAW EEG    │
│   SIGNAL     │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 1: SIGNAL LOADING                                                  │
│  ├── MNE-Python for .edf/.set files                                     │
│  ├── NumPy for .eea/.txt files                                          │
│  └── Pandas for .csv files                                              │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 2: SEGMENTATION                                                    │
│  ├── Window size: 2-4 seconds (512-1024 samples at 256 Hz)             │
│  ├── Overlap: 50%                                                        │
│  └── Max segments: 6-10 per subject                                     │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 3: FEATURE EXTRACTION                                              │
│  ├── Time-domain: mean, std, skew, kurtosis                             │
│  ├── Frequency-domain: Welch PSD → Band powers                          │
│  └── Hjorth parameters: activity, mobility, complexity                  │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 4: NaN HANDLING                                                    │
│  └── np.nan_to_num(X, nan=0)                                            │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 5: TRAIN/TEST SPLIT                                                │
│  └── StratifiedKFold(n_splits=5, shuffle=True)                          │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 6: DATA AUGMENTATION (training set only)                          │
│  └── Gaussian noise injection (1x - 40x)                                │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 7: STANDARDIZATION / NORMALIZATION                                │
│  ├── fit on training data                                               │
│  └── transform both train and test                                      │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 8: MODEL TRAINING                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Standardization vs Normalization

### Comparison

| Aspect | Standardization | Normalization |
|--------|-----------------|---------------|
| **Formula** | z = (x - μ) / σ | x' = (x - min) / (max - min) |
| **Output Range** | Unbounded | [0, 1] |
| **Center** | Mean = 0 | Depends on data |
| **Scale** | Std = 1 | Range = 1 |
| **Outlier Sensitivity** | Moderate | High |
| **Used in Project** | ✓ Primary | ✗ Not used |

### Why We Chose Standardization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STANDARDIZATION (Z-Score)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Formula:  z = (x - μ) / σ                                              │
│                                                                          │
│  Where:                                                                  │
│    x = original value                                                    │
│    μ = mean of the feature                                              │
│    σ = standard deviation                                                │
│    z = standardized value                                                │
│                                                                          │
│  Result:                                                                 │
│    Mean of z = 0                                                         │
│    Std of z = 1                                                          │
│                                                                          │
│  Benefits for EEG:                                                       │
│    ✓ Handles different amplitude scales across channels                 │
│    ✓ Works well with Gaussian-like distributions                        │
│    ✓ Preserves outliers (important for seizure detection)               │
│    ✓ Works well with tree-based models and neural networks              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Scalers Used

### 1. StandardScaler (for Deep Learning)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
X_test_scaled = scaler.transform(X_test)        # Transform test data

# Properties:
# - Removes mean and scales to unit variance
# - z = (x - mean) / std
# - Assumes normal distribution
```

### 2. RobustScaler (for ML Models)

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Properties:
# - Uses median and IQR instead of mean and std
# - z = (x - median) / IQR
# - More robust to outliers (common in EEG)
```

### Comparison: StandardScaler vs RobustScaler

```
Original Data with Outliers:
┌────────────────────────────────────────────────────────────────────────┐
│ Values: [1, 2, 3, 4, 5, 100]  ← outlier                               │
│ Mean: 19.17    Std: 38.5                                               │
└────────────────────────────────────────────────────────────────────────┘

StandardScaler Output:
┌────────────────────────────────────────────────────────────────────────┐
│ z-scores: [-0.47, -0.45, -0.42, -0.39, -0.37, 2.10]                   │
│ Outlier heavily influences scaling                                     │
└────────────────────────────────────────────────────────────────────────┘

RobustScaler Output:
┌────────────────────────────────────────────────────────────────────────┐
│ Values: [-0.67, -0.33, 0, 0.33, 0.67, 32.33]                          │
│ Outlier isolated, main data well-distributed                          │
│ Uses median=3, IQR=3                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Preprocessing Code

```python
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold

# ============================================================
# STEP 1: Load EEG Data
# ============================================================
import mne

def load_eeg(file_path, file_type='edf'):
    if file_type == 'edf':
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    elif file_type == 'set':
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    elif file_type == 'eea':
        data = np.loadtxt(file_path)
        return data, 128  # Default fs for .eea

    data = raw.get_data()
    fs = raw.info['sfreq']
    return data, fs


# ============================================================
# STEP 2: Segmentation
# ============================================================
def segment_signal(data, fs, segment_duration=3, overlap=0.5):
    """
    Segment continuous EEG into fixed-length windows

    Parameters:
    - data: (n_channels, n_samples) array
    - fs: sampling frequency
    - segment_duration: length in seconds
    - overlap: fraction of overlap (0-1)

    Returns:
    - segments: list of (n_channels, segment_length) arrays
    """
    segment_length = int(fs * segment_duration)
    step = int(segment_length * (1 - overlap))

    segments = []
    for start in range(0, data.shape[1] - segment_length, step):
        end = start + segment_length
        segments.append(data[:, start:end])

    return segments


# ============================================================
# STEP 3: Feature Extraction
# ============================================================
def extract_features(segment, fs=256):
    """Extract features from a single segment"""
    features = []
    n_channels = min(segment.shape[0], 8)

    for ch in range(n_channels):
        ch_data = segment[ch]

        # Time-domain features
        features.append(np.mean(ch_data))
        features.append(np.std(ch_data))
        features.append(skew(ch_data))
        features.append(kurtosis(ch_data))

        # Frequency-domain features (Welch PSD)
        nperseg = min(len(ch_data), int(fs))
        f, psd = signal.welch(ch_data, fs, nperseg=nperseg)

        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
        for low, high in bands:
            idx = (f >= low) & (f <= high)
            power = np.trapz(psd[idx], f[idx]) if idx.sum() > 0 else 0
            features.append(power)

    return np.array(features, dtype=np.float32)


# ============================================================
# STEP 4: NaN Handling
# ============================================================
def handle_nan(X):
    """Replace NaN and Inf with 0"""
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)


# ============================================================
# STEP 5: Train/Test Split
# ============================================================
def get_cv_splits(X, y, n_splits=5):
    """Stratified K-Fold cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(skf.split(X, y))


# ============================================================
# STEP 6: Data Augmentation
# ============================================================
def augment_data(X, y, factor=40):
    """
    Gaussian noise augmentation

    Parameters:
    - X: feature matrix
    - y: labels
    - factor: number of augmented samples per original

    Returns:
    - X_aug, y_aug: augmented data
    """
    X = np.array(X, dtype=np.float32)
    std = np.std(X, axis=0, keepdims=True) + 1e-10

    X_aug = [X]
    y_aug = [y]

    for i in range(factor - 1):
        noise_scale = 0.002 + i * 0.0001
        noise = np.random.randn(*X.shape).astype(np.float32) * noise_scale * std
        X_aug.append(X + noise)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)


# ============================================================
# STEP 7: Standardization
# ============================================================
def standardize(X_train, X_test, method='robust'):
    """
    Standardize features

    Parameters:
    - X_train: training features
    - X_test: test features
    - method: 'robust' or 'standard'

    Returns:
    - X_train_scaled, X_test_scaled, scaler
    """
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only!
    X_test_scaled = scaler.transform(X_test)        # Transform test

    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================
def preprocess_pipeline(raw_data, fs, labels, augmentation_factor=40):
    """
    Complete preprocessing from raw EEG to model-ready features

    Returns:
    - X_scaled: standardized features
    - y: labels
    """
    # Step 2: Segment
    all_segments = []
    all_labels = []
    for data, label in zip(raw_data, labels):
        segments = segment_signal(data, fs)
        all_segments.extend(segments)
        all_labels.extend([label] * len(segments))

    # Step 3: Extract features
    X = np.array([extract_features(seg, fs) for seg in all_segments])
    y = np.array(all_labels)

    # Step 4: Handle NaN
    X = handle_nan(X)

    # Steps 5-7 happen in training loop
    return X, y
```

---

## Preprocessing Summary by Disease

| Disease | Scaler | Augmentation | Segment Length | Channels Used |
|---------|--------|--------------|----------------|---------------|
| Schizophrenia | RobustScaler | 1x | 2000 samples | 1 |
| Epilepsy | RobustScaler | 2x | 1024 samples | 23 → 19 |
| Stress | RobustScaler | 2x | Pre-extracted | All |
| Autism | RobustScaler | 3x | Pre-extracted | All |
| Parkinson | RobustScaler | 1x | Pre-extracted | All |
| Depression | StandardScaler | 40x | 768 samples | 8 |

---

## Critical Preprocessing Rules

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING GOLDEN RULES                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. NEVER fit scaler on test data                                       │
│     ✗ scaler.fit_transform(X_test)   # WRONG                           │
│     ✓ scaler.transform(X_test)        # CORRECT                        │
│                                                                          │
│  2. ALWAYS augment AFTER train/test split                               │
│     ✗ augment → split   # Data leakage                                 │
│     ✓ split → augment   # Correct                                      │
│                                                                          │
│  3. ONLY augment training data                                          │
│     ✗ Augment test data                                                 │
│     ✓ Keep test data original                                          │
│                                                                          │
│  4. Handle NaN BEFORE scaling                                           │
│     ✗ Scale first, then handle NaN                                     │
│     ✓ Handle NaN first, then scale                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```
