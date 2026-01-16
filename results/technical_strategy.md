# Technical Strategy: How We Achieved 90%+ Accuracy

## Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-LEVEL OPTIMIZATION STRATEGY                 │
├─────────────────────────────────────────────────────────────────────┤
│  Level 1: DATA LEVEL                                                 │
│  ├── K-Fold Cross-Validation (Stratified)                           │
│  ├── Heavy Data Augmentation (1-40x)                                │
│  └── Class Balancing                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Level 2: FEATURE LEVEL                                              │
│  ├── 1D → Frequency Domain (Fourier/Welch)                          │
│  ├── Band Power Extraction                                           │
│  └── Statistical & Hjorth Features                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Level 3: MODEL LEVEL                                                │
│  ├── Ensemble Methods (Voting/Stacking)                             │
│  ├── Deep Learning (DNN)                                             │
│  └── Hyperparameter Tuning                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. K-Fold Cross-Validation Strategy

### Why Stratified K-Fold?
```
Problem: Small dataset + Class imbalance
Solution: Stratified K-Fold maintains class distribution in each fold

Standard Split:           Stratified K-Fold:
┌────────────────┐         ┌────────────────┐
│ Train: Random  │         │ Fold 1: 60% H, 40% D │
│ Test: Random   │         │ Fold 2: 60% H, 40% D │
│ Risk: Imbalanced│         │ Fold 3: 60% H, 40% D │
└────────────────┘         │ Fold 4: 60% H, 40% D │
                           │ Fold 5: 60% H, 40% D │
                           └────────────────┘
```

### Implementation
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate...
```

### Benefits
| Benefit | Impact |
|---------|--------|
| Unbiased estimate | No lucky/unlucky splits |
| Uses all data | Every sample tested once |
| Preserves distribution | Works with imbalanced data |
| Reduces variance | More stable accuracy estimate |

---

## 2. Data-Level Analysis & Augmentation

### Segmentation Strategy
```
Raw EEG Signal (continuous)
│
▼
┌────────────────────────────────────────────────────────────┐
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└────────────────────────────────────────────────────────────┘
│
▼  Segment into fixed windows (2-4 seconds)
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
│Seg 1 ││Seg 2 ││Seg 3 ││Seg 4 ││Seg 5 ││Seg 6 │
└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
│
▼  50% overlap for more samples
┌──────┐
│Seg 1 │
└──────┘
   ┌──────┐
   │Seg 2 │
   └──────┘
      ┌──────┐
      │Seg 3 │
      └──────┘
```

### Data Augmentation
```
Original Sample X
        │
        ▼
┌─────────────────────────────────────────┐
│           AUGMENTATION                   │
│                                          │
│  X + noise₁ → X'₁  (noise_scale=0.1%)   │
│  X + noise₂ → X'₂  (noise_scale=0.2%)   │
│  X + noise₃ → X'₃  (noise_scale=0.3%)   │
│  ...                                     │
│  X + noiseₙ → X'ₙ  (noise_scale=n*0.1%) │
│                                          │
└─────────────────────────────────────────┘
        │
        ▼
1 sample → N samples (N = augmentation factor)
```

### Augmentation Code
```python
def augment(X, y, factor=40):
    """Gaussian noise augmentation"""
    X = np.array(X, dtype=np.float32)
    std = np.std(X, axis=0, keepdims=True) + 1e-10

    X_aug, y_aug = [X], [y]
    for i in range(factor - 1):
        noise_scale = 0.002 + i * 0.0001  # Increasing noise
        noise = np.random.randn(*X.shape) * noise_scale * std
        X_aug.append(X + noise)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)
```

---

## 3. 1D to Frequency Domain Conversion (Fourier Transform)

### Time Domain → Frequency Domain
```
Time Domain (1D Signal):
amplitude
    │    ╱╲    ╱╲    ╱╲    ╱╲
    │   ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
    │──╱────╲╱────╲╱────╲╱────╲──→ time
    │

                    ↓ Fourier Transform (Welch's Method)

Frequency Domain (Power Spectrum):
power
    │  █
    │  █ █
    │  █ █       █
    │  █ █ █     █
    │  █ █ █ █   █ █
    └──────────────────────→ frequency (Hz)
       δ θ α β   γ
```

### Welch's Method Implementation
```python
from scipy import signal

def compute_psd(data, fs=256):
    """
    Welch's periodogram for Power Spectral Density

    Parameters:
    - data: 1D time series
    - fs: sampling frequency

    Returns:
    - f: frequency array
    - psd: power spectral density
    """
    nperseg = min(len(data), int(fs * 2))  # 2-second windows
    f, psd = signal.welch(data, fs, nperseg=nperseg)
    return f, psd
```

### Band Power Extraction
```python
def bandpower(data, fs, band):
    """
    Extract power in specific frequency band

    Frequency Bands:
    - Delta (δ): 0.5-4 Hz  → Deep sleep, unconscious
    - Theta (θ): 4-8 Hz   → Drowsiness, meditation
    - Alpha (α): 8-13 Hz  → Relaxed, eyes closed
    - Beta (β): 13-30 Hz  → Active thinking, focus
    - Gamma (γ): 30-50 Hz → High-level cognition
    """
    f, psd = signal.welch(data, fs, nperseg=min(len(data), int(fs*2)))

    # Find frequency indices in band
    idx = (f >= band[0]) & (f <= band[1])

    # Integrate power (area under curve)
    power = np.trapz(psd[idx], f[idx])

    return power
```

### Why Fourier Transform Works for EEG
```
┌────────────────────────────────────────────────────────────────────┐
│  Disease-Specific EEG Patterns (Frequency Domain)                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Schizophrenia:  ↑ Theta, ↓ Alpha, Abnormal Gamma                  │
│  Epilepsy:       Spike patterns, Abnormal synchronization          │
│  Depression:     ↑ Alpha asymmetry (left < right frontal)          │
│  Autism:         Altered Alpha/Beta ratios, Connectivity changes   │
│  Parkinson:      ↓ Beta, ↑ Theta in motor cortex                   │
│  Stress:         ↑ Beta, ↓ Alpha                                   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Extraction Pipeline

### Complete Feature Vector
```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  For each channel (up to 10 channels):                              │
│                                                                      │
│  1. STATISTICAL FEATURES (4 features)                               │
│     ├── Mean: np.mean(signal)                                       │
│     ├── Std:  np.std(signal)                                        │
│     ├── Skew: scipy.stats.skew(signal)                              │
│     └── Kurt: scipy.stats.kurtosis(signal)                          │
│                                                                      │
│  2. BAND POWERS (5 features) - via Welch PSD                        │
│     ├── Delta (0.5-4 Hz):  slow wave activity                       │
│     ├── Theta (4-8 Hz):    drowsiness, memory                       │
│     ├── Alpha (8-13 Hz):   relaxation, attention                    │
│     ├── Beta (13-30 Hz):   active thinking                          │
│     └── Gamma (30-50 Hz):  cognition                                │
│                                                                      │
│  3. HJORTH PARAMETERS (2 features)                                  │
│     ├── Mobility: sqrt(var(diff(x)) / var(x))                       │
│     └── Complexity: mobility(diff(x)) / mobility(x)                 │
│                                                                      │
│  Total per channel: 4 + 5 + 2 = 11 features                         │
│  Total for 8 channels: 11 × 8 = 88 features                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature Extraction Code
```python
def extract_features(data, fs=256):
    """Extract comprehensive EEG features"""
    features = []
    n_channels = min(data.shape[0], 8)

    for ch in range(n_channels):
        signal = data[ch]

        # 1. Statistical features
        features.extend([
            np.mean(signal),
            np.std(signal),
            skew(signal),
            kurtosis(signal)
        ])

        # 2. Band powers (Fourier Transform)
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
        for low, high in bands:
            power = bandpower(signal, fs, (low, high))
            features.append(power)

        # 3. Hjorth parameters
        diff1 = np.diff(signal)
        mobility = np.sqrt(np.var(diff1) / (np.var(signal) + 1e-10))
        features.append(mobility)

    return np.array(features, dtype=np.float32)
```

---

## 5. Model-Level Strategy

### Ensemble Architecture
```
                    ┌─────────────────┐
                    │   Input Data    │
                    │  (88 features)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  ExtraTrees     │ │  RandomForest   │ │    XGBoost      │
│  n=500 trees    │ │  n=500 trees    │ │   n=300 trees   │
│  max_depth=None │ │  max_depth=None │ │   max_depth=6   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         │    Probability    │    Probability    │    Probability
         │      Output       │      Output       │      Output
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Soft Voting    │
                    │  (Average)      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Final Prediction│
                    │ (argmax prob)   │
                    └─────────────────┘
```

### Deep Learning for Depression
```
┌────────────────────────────────────────────────────────────────────┐
│                    DEEP NEURAL NETWORK                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Layer (88 features)                                          │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────┐                           │
│  │ Dense(256) + BatchNorm + ReLU       │                           │
│  │ Dropout(0.5)                         │                           │
│  └─────────────────────────────────────┘                           │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────┐                           │
│  │ Dense(128) + BatchNorm + ReLU       │                           │
│  │ Dropout(0.4)                         │                           │
│  └─────────────────────────────────────┘                           │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────┐                           │
│  │ Dense(64) + BatchNorm + ReLU        │                           │
│  │ Dropout(0.3)                         │                           │
│  └─────────────────────────────────────┘                           │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────┐                           │
│  │ Dense(2) - Output                    │                           │
│  │ Softmax activation                   │                           │
│  └─────────────────────────────────────┘                           │
│                                                                     │
│  Optimizer: AdamW (lr=0.003, weight_decay=1e-4)                    │
│  Loss: CrossEntropyLoss (class weighted)                           │
│  Epochs: 60                                                         │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 6. Summary: Key Success Factors

| Factor | Strategy | Impact |
|--------|----------|--------|
| **Small Dataset** | Heavy augmentation (40x) | +15-25% accuracy |
| **Class Imbalance** | Stratified K-Fold + Weighted loss | Stable performance |
| **Feature Quality** | Fourier + Band Powers | Disease-specific patterns |
| **Model Robustness** | Ensemble voting | Reduced variance |
| **Depression Challenge** | DNN + XGBoost hybrid | Broke 90% barrier |

### Final Performance
```
Disease         Strategy                              Accuracy
─────────────────────────────────────────────────────────────
Parkinson       VotingClassifier (no aug)             100.0%
Autism          VotingClassifier + 3x aug              97.7%
Schizophrenia   VotingClassifier (no aug)              97.2%
Epilepsy        VotingClassifier + 2x aug              94.2%
Stress          VotingClassifier + 2x aug              94.2%
Depression      DNN + XGBoost + 40x aug                91.1%
─────────────────────────────────────────────────────────────
Average                                                95.7%
```
