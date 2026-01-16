# AgenticFinder API Documentation

## Overview

AgenticFinder provides a complete pipeline for EEG-based neurological disease classification with the following components:

1. **Data Generation** - Synthetic EEG data for testing
2. **Feature Extraction** - 47 EEG features
3. **Classification** - Ultra Stacking Ensemble
4. **Evaluation** - Comprehensive metrics
5. **Prediction** - Inference on new data

## Core Classes

### EEGDataGenerator

Generate synthetic EEG signals with disease-specific characteristics.

```python
from scripts.generate_sample_data import EEGDataGenerator

# Initialize
generator = EEGDataGenerator(sampling_rate=256, seed=42)

# Generate dataset
data = generator.generate_dataset(
    disease='parkinson',      # Disease type
    n_subjects=20,            # Subjects per class
    samples_per_subject=10,   # Samples per subject
    duration=2.0,             # Signal duration (seconds)
    n_channels=1,             # Number of EEG channels
    include_controls=True     # Include healthy controls
)

# Access data
signals = data['signals']      # Shape: (n_samples, n_timepoints)
labels = data['labels']        # 0=Control, 1=Disease
subject_ids = data['subject_ids']
```

#### Supported Diseases

| Disease | Key Spectral Profile |
|---------|---------------------|
| `parkinson` | Increased beta, reduced alpha |
| `epilepsy` | Abnormal spikes, increased delta |
| `autism` | Altered gamma and alpha |
| `schizophrenia` | Reduced alpha, increased delta/theta |
| `stress` | Increased beta and gamma |
| `alzheimer` | Increased delta/theta, reduced alpha/beta |
| `depression` | Alpha asymmetry, increased theta |
| `control` | Normal alpha dominance |

### EEGFeatureExtractor

Extract 47 features from EEG signals.

```python
from scripts.train import EEGFeatureExtractor

# Initialize
extractor = EEGFeatureExtractor(sampling_rate=256)

# Extract features
features = extractor.extract(signal)  # Returns (47,) array
```

#### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Statistical | 15 | Mean, std, skewness, kurtosis |
| Spectral | 18 | Band powers (δ,θ,α,β,γ), entropy |
| Temporal | 9 | Hjorth parameters, zero crossings |
| Nonlinear | 5 | Approximate entropy, Hurst exponent |

### UltraStackingEnsemble

15-classifier stacking ensemble with MLP meta-learner.

```python
from scripts.train import UltraStackingEnsemble

# Initialize
ensemble = UltraStackingEnsemble(random_state=42)

# Train
ensemble.fit(X_train, y_train)

# Predict
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)
```

#### Base Classifiers

| # | Classifier | Count |
|---|------------|-------|
| 1-2 | ExtraTrees | 2 |
| 3-4 | Random Forest | 2 |
| 5-6 | Gradient Boosting | 2 |
| 7-8 | AdaBoost | 2 |
| 9-10 | MLP | 2 |
| 11 | SVM | 1 |
| 12-13 | XGBoost | 2 |
| 14-15 | LightGBM | 2 |

#### Methods

```python
# Fit the ensemble
ensemble.fit(X, y)

# Make predictions
y_pred = ensemble.predict(X)

# Get class probabilities
y_prob = ensemble.predict_proba(X)

# Save model
import joblib
joblib.dump({'model': ensemble}, 'model.joblib')

# Load model
checkpoint = joblib.load('model.joblib')
ensemble = checkpoint['model']
```

### ModelEvaluator

Comprehensive evaluation with multiple metrics.

```python
from scripts.evaluate import ModelEvaluator

# Initialize with trained model
evaluator = ModelEvaluator('models/parkinson_model.joblib')

# Generate full report
report = evaluator.generate_report(
    X, y,
    subject_ids=subject_ids,      # For LOSO-CV
    class_names=['Control', 'Disease'],
    output_dir='results/'
)

# Access metrics
accuracy = report['metrics']['accuracy']
f1 = report['metrics']['f1_score']
cm = report['confusion_matrix_analysis']['confusion_matrix']
```

#### Available Metrics

- Accuracy, Balanced Accuracy
- Precision, Recall, F1-Score
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- ROC-AUC, Average Precision
- Per-class Sensitivity, Specificity, PPV, NPV

### EEGPredictor

Make predictions on new data.

```python
from scripts.predict import EEGPredictor

# Initialize
predictor = EEGPredictor('models/parkinson_model.joblib')

# Predict from features
result = predictor.predict(features)
print(result['prediction'])      # 0 or 1
print(result['class_name'])      # 'Control' or 'Disease'
print(result['confidence'])      # 0.0 to 1.0

# Predict from raw signal
result = predictor.predict_from_signal(eeg_signal, sampling_rate=256)

# Batch prediction
results = predictor.predict_batch(feature_matrix)

# With explanations
result = predictor.predict_with_explanation(features)
print(result['explanation']['top_features'])
```

### StreamingPredictor

Real-time streaming prediction.

```python
from scripts.predict import EEGPredictor, StreamingPredictor

# Initialize
predictor = EEGPredictor('models/model.joblib')
streamer = StreamingPredictor(
    predictor,
    window_size=256,    # Samples per window
    step_size=64,       # Slide step
    sampling_rate=256
)

# Process incoming samples
while True:
    new_samples = get_eeg_samples()  # Your data source
    result = streamer.add_samples(new_samples)

    if result:
        print(f"Prediction: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2%}")

# Get smoothed prediction
smoothed = streamer.get_smoothed_prediction(n_recent=5)
```

## Command-Line Interface

### Generate Data

```bash
# All diseases
python scripts/generate_sample_data.py --disease all --subjects 20 --samples 10

# Specific disease with features
python scripts/generate_sample_data.py --disease parkinson --subjects 30 --features
```

### Train Model

```bash
# With synthetic data
python scripts/train.py --disease parkinson --synthetic --output models/

# With custom data
python scripts/train.py --disease parkinson --data data/features.npz
```

### Evaluate Model

```bash
python scripts/evaluate.py --model models/model.joblib --synthetic --output results/
```

### Make Predictions

```bash
python scripts/predict.py --model models/model.joblib --input data/samples.npz
```

### Generate Figures

```bash
python scripts/generate_figures.py
```

## Data Formats

### Input Data (NPZ)

```python
# Required arrays
X = features          # Shape: (n_samples, 47)
y = labels            # Shape: (n_samples,), values 0 or 1

# Optional
subject_ids = ids     # Shape: (n_samples,)
class_names = ['Control', 'Disease']

# Save
np.savez('data.npz', X=X, y=y, subject_ids=subject_ids, class_names=class_names)
```

### Model Checkpoint (joblib)

```python
checkpoint = {
    'model': trained_ensemble,
    'feature_extractor': extractor,
    'class_names': ['Control', 'Disease'],
    'disease_name': 'parkinson',
    'training_info': {
        'n_samples': 1000,
        'accuracy': 0.92,
        'date': '2024-01-15'
    }
}
joblib.dump(checkpoint, 'model.joblib')
```

## Error Handling

```python
try:
    predictor = EEGPredictor('model.joblib')
    result = predictor.predict(features)
except FileNotFoundError:
    print("Model file not found")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Best Practices

1. **Always normalize features** - Use StandardScaler or RobustScaler
2. **Use LOSO-CV** for subject-independent evaluation
3. **Check class balance** before training
4. **Save feature names** with models for reproducibility
5. **Log hyperparameters** for experiment tracking
