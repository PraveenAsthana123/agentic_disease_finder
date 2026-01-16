# Data Distribution: Healthy vs Disease

## Summary Table

| Disease | Total Subjects | Healthy | Disease | Balance Ratio | Segments |
|---------|----------------|---------|---------|---------------|----------|
| **Schizophrenia** | 84 | 39 (46.4%) | 45 (53.6%) | 0.87:1 | ~8,400 |
| **Epilepsy** | 102 | 51 (50.0%) | 51 (50.0%) | 1:1 | ~5,100 |
| **Stress** | 120 | 60 (50.0%) | 60 (50.0%) | 1:1 | ~6,000 |
| **Autism** | 300 | 150 (50.0%) | 150 (50.0%) | 1:1 | ~3,000 |
| **Parkinson** | 50 | 25 (50.0%) | 25 (50.0%) | 1:1 | ~500 |
| **Depression** | 112 | 74 (66.1%) | 38 (33.9%) | 1.95:1 | ~1,792 |

---

## 1. Schizophrenia

### Dataset Information
```
Source: RepOD Repository (schizophrenia_eeg_real)
Format: .eea files (single-channel EEG text files)
Sampling Rate: 128 Hz
```

### Class Distribution
```
┌─────────────────────────────────────────────────────────────┐
│                     SCHIZOPHRENIA DATA                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HEALTHY CONTROLS        │████████████████████│  39 (46.4%)  │
│  (Label: 0)              │                    │              │
│  Folder: healthy/        │                    │              │
│                                                              │
│  SCHIZOPHRENIA           │██████████████████████│ 45 (53.6%) │
│  (Label: 1)              │                    │              │
│  Folder: schizophrenia/  │                    │              │
│                                                              │
│  TOTAL: 84 subjects                                          │
├─────────────────────────────────────────────────────────────┤
│  Segments per subject: ~100                                  │
│  Total segments: ~8,400                                      │
│  Segment length: 2000 samples (~15.6 seconds)               │
└─────────────────────────────────────────────────────────────┘
```

### Label Criteria
- **Healthy (0):** No psychiatric diagnosis, normal EEG patterns
- **Schizophrenia (1):** Clinically diagnosed schizophrenia patients

---

## 2. Epilepsy

### Dataset Information
```
Source: CHB-MIT Scalp EEG Database (PhysioNet)
Format: .edf files (European Data Format)
Sampling Rate: 256 Hz
Channels: 23 EEG channels
```

### Class Distribution
```
┌─────────────────────────────────────────────────────────────┐
│                       EPILEPSY DATA                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NON-SEIZURE             │████████████████████│  51 (50.0%)  │
│  (Label: 0)              │                    │              │
│  Inter-ictal periods     │                    │              │
│                                                              │
│  SEIZURE/ICTAL           │████████████████████│  51 (50.0%)  │
│  (Label: 1)              │                    │              │
│  Pre-ictal & ictal       │                    │              │
│                                                              │
│  TOTAL: 102 recordings                                       │
├─────────────────────────────────────────────────────────────┤
│  Segments per file: ~25                                      │
│  Total segments: ~5,100                                      │
│  Segment length: 1024 samples (4 seconds)                   │
└─────────────────────────────────────────────────────────────┘
```

### Label Criteria
- **Non-Seizure (0):** Normal brain activity, no seizure markers
- **Seizure (1):** Pre-ictal or ictal (seizure) activity patterns

---

## 3. Stress

### Dataset Information
```
Source: stress_real dataset
Format: CSV with pre-extracted features
Labels: Binary (stressed/relaxed)
```

### Class Distribution
```
┌─────────────────────────────────────────────────────────────┐
│                        STRESS DATA                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  RELAXED/BASELINE        │████████████████████│  60 (50.0%)  │
│  (Label: 0)              │                    │              │
│  Resting state EEG       │                    │              │
│                                                              │
│  STRESSED                │████████████████████│  60 (50.0%)  │
│  (Label: 1)              │                    │              │
│  Task-induced stress     │                    │              │
│                                                              │
│  TOTAL: 120 samples                                          │
├─────────────────────────────────────────────────────────────┤
│  Features per sample: Pre-extracted                          │
│  Total segments: ~6,000 (after augmentation)                │
└─────────────────────────────────────────────────────────────┘
```

### Label Criteria
- **Relaxed (0):** Baseline/resting state recordings
- **Stressed (1):** During cognitive stress tasks (math, memory)

---

## 4. Autism

### Dataset Information
```
Source: autism_real (Kaggle/Research repository)
Format: CSV with pre-extracted EEG features
```

### Class Distribution
```
┌─────────────────────────────────────────────────────────────┐
│                        AUTISM DATA                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NEUROTYPICAL (TD)       │████████████████████│ 150 (50.0%)  │
│  (Label: 0)              │                    │              │
│  Typically developing    │                    │              │
│                                                              │
│  ASD                     │████████████████████│ 150 (50.0%)  │
│  (Label: 1)              │                    │              │
│  Autism Spectrum         │                    │              │
│                                                              │
│  TOTAL: 300 samples                                          │
├─────────────────────────────────────────────────────────────┤
│  Features per sample: Pre-extracted EEG metrics              │
│  Age range: Children (typically 6-18 years)                 │
└─────────────────────────────────────────────────────────────┘
```

### Label Criteria
- **Neurotypical (0):** No developmental disorders
- **ASD (1):** Clinically diagnosed Autism Spectrum Disorder

---

## 5. Parkinson

### Dataset Information
```
Source: parkinson_real dataset
Format: CSV with extracted features
```

### Class Distribution
```
┌─────────────────────────────────────────────────────────────┐
│                      PARKINSON DATA                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HEALTHY CONTROLS        │████████████████████│  25 (50.0%)  │
│  (Label: 0)              │                    │              │
│  Age-matched controls    │                    │              │
│                                                              │
│  PARKINSON'S DISEASE     │████████████████████│  25 (50.0%)  │
│  (Label: 1)              │                    │              │
│  PD patients             │                    │              │
│                                                              │
│  TOTAL: 50 samples                                           │
├─────────────────────────────────────────────────────────────┤
│  Note: Small but highly discriminative dataset               │
│  100% accuracy achieved (clear biomarkers)                  │
└─────────────────────────────────────────────────────────────┘
```

### Label Criteria
- **Healthy (0):** No movement disorders, age-matched
- **Parkinson (1):** Clinically diagnosed Parkinson's disease

---

## 6. Depression

### Dataset Information
```
Source: ds003478 (OpenNeuro)
Format: .set files (EEGLAB format)
Sampling Rate: 256 Hz
Channels: Multi-channel EEG
Labels: Based on BDI (Beck Depression Inventory) score
```

### Class Distribution
```
┌─────────────────────────────────────────────────────────────┐
│                      DEPRESSION DATA                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HEALTHY (BDI ≤ 6)       │██████████████████████████│ 74     │
│  (Label: 0)              │                          │ 66.1%  │
│  Low depression score    │                          │        │
│                                                              │
│  DEPRESSED (BDI ≥ 18)    │█████████████│             38      │
│  (Label: 1)              │             │            33.9%   │
│  Moderate-severe         │             │                    │
│                                                              │
│  TOTAL: 112 subjects (selected from 122)                     │
├─────────────────────────────────────────────────────────────┤
│  Excluded: Ambiguous cases (BDI 7-17)                       │
│  Segments per subject: ~16                                   │
│  Total segments: 1,792                                       │
│  Class imbalance handled with: 40x augmentation + DNN       │
└─────────────────────────────────────────────────────────────┘
```

### Label Criteria
- **Healthy (0):** BDI score ≤ 6 (minimal depression)
- **Depressed (1):** BDI score ≥ 18 (moderate to severe depression)

### BDI Score Interpretation
```
BDI Score    │  Interpretation         │  Used in Study
─────────────┼─────────────────────────┼────────────────
0-6          │  Minimal                │  ✓ Healthy (0)
7-13         │  Mild                   │  ✗ Excluded
14-17        │  Borderline/Mild        │  ✗ Excluded
18-29        │  Moderate               │  ✓ Depressed (1)
30-63        │  Severe                 │  ✓ Depressed (1)
```

---

## Visual Summary

```
                    CLASS DISTRIBUTION OVERVIEW

Disease         Healthy              Disease
─────────────────────────────────────────────────────────────
Schizophrenia   ████████████████░░░░ 46%  ████████████████████ 54%
Epilepsy        ██████████████████░░ 50%  ██████████████████░░ 50%
Stress          ██████████████████░░ 50%  ██████████████████░░ 50%
Autism          ██████████████████░░ 50%  ██████████████████░░ 50%
Parkinson       ██████████████████░░ 50%  ██████████████████░░ 50%
Depression      ████████████████████████ 66%  ████████████░░░░ 34%
─────────────────────────────────────────────────────────────

Note: Depression had class imbalance (66:34), handled with heavy augmentation
```

---

## Data Quality Notes

| Disease | Data Quality | Challenges | Solutions |
|---------|--------------|------------|-----------|
| Schizophrenia | High | Single-channel | Multiple segments |
| Epilepsy | High | Multi-file per subject | File-level labels |
| Stress | Medium | Pre-extracted features | Feature validation |
| Autism | Medium | Pre-extracted features | Feature validation |
| Parkinson | High | Small dataset | Perfect separation |
| Depression | Medium | Imbalanced, BDI-based | Heavy augmentation |
