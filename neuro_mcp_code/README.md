# NeuroMCP-Agent: EEG-Based Multi-Disease Neurological Detection

A trustworthy multi-agent deep learning framework achieving **92.97% validated accuracy** for EEG-based neurological disease detection across 7 conditions with comprehensive Responsible AI governance.

## Validated Results

### Stratified 5-Fold Cross-Validation (with Bootstrap CI)

| Disease | Accuracy | Sensitivity | Specificity | AUC | 95% CI |
|---------|----------|-------------|-------------|-----|--------|
| **Parkinson's** | **97.94%** | 99.13% | 97.13% | 0.997 | [96.2-99.3] |
| **Epilepsy** | **96.48%** | 92.05% | 97.83% | 0.987 | [93.8-98.6] |
| **Schizophrenia** | **95.52%** | 100.0% | 93.57% | 0.997 | [92.8-97.6] |
| **Depression** | **92.24%** | 91.08% | 92.92% | 0.966 | [88.7-95.2] |
| **Alzheimer's** | **90.06%** | 95.10% | 87.37% | 0.942 | [86.6-93.5] |
| **Autism** | **90.02%** | 96.10% | 86.81% | 0.967 | [86.6-93.1] |
| Stress | 88.50% | 94.72% | 86.35% | 0.947 | [84.5-91.8] |
| **Average** | **92.97%** | 95.45% | 91.71% | **0.972** | -- |

### LOSO-CV Results (Subject-Independent)

| Disease | LOSO Accuracy |
|---------|---------------|
| Autism | 85.94% |
| Parkinson's | 84.11% |
| Alzheimer's | 83.67% |
| Epilepsy | 82.50% |
| Stress | 80.90% |

## Files

### Core Analysis
- `comprehensive_analysis.py` - Complete validation with LOSO-CV, bootstrap CI, sensitivity/specificity
- `train_high_accuracy.py` - Ultra Stacking Ensemble training
- `train_all_diseases.py` - Multi-disease training pipeline

### Data & Features
- `download_eeg_datasets.py` - Dataset download scripts for PhysioNet
- `data_loaders/` - Disease-specific data loaders
- `features/` - 400-feature extraction pipeline
- `preprocessing/` - EEG preprocessing (bandpass, notch filtering)

### Utilities
- `agent_system.py` - Multi-agent coordination system
- `agentic_classifier.py` - Agentic disease classification
- `rag_engine.py` - RAG-based clinical knowledge retrieval
- `utils.py` - Common utilities

### Paper & Results
- `journal_comprehensive_combined.tex` - Full journal paper (LaTeX)
- `journal_comprehensive_combined.pdf` - Compiled paper (19 pages)
- `analysis_results.json` - Complete validation results

## Requirements

```bash
pip install numpy scipy scikit-learn xgboost lightgbm mne
```

## Usage

### Run Comprehensive Analysis
```bash
python comprehensive_analysis.py
```

### Train All Diseases
```bash
python train_high_accuracy.py
```

## Key Features

1. **Ultra Stacking Ensemble**: 15 classifiers (RF, XGBoost, LightGBM, SVM, MLP) + MLP meta-learner
2. **Disease-Specific Biomarkers**: Clinically-validated EEG biomarkers for each condition
3. **400-Feature Extraction**: Statistical, spectral, temporal, nonlinear features
4. **Rigorous Validation**: 5-fold stratified CV, LOSO-CV, 1000-iteration bootstrap CI
5. **RAI Compliance**: 0.91 overall score across 46 modules, 1300+ analysis types

## Citation

If you use this code, please cite:

```
@article{neuromcp2025,
  title={NeuroMCP-Agent: A Trustworthy Multi-Agent Deep Learning Framework with Comprehensive Responsible AI Governance for EEG-Based Multi-Disease Neurological Detection},
  author={Asthana, Praveen and Lalawat, Rajveer Singh and Gond, Sarita Singh},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025}
}
```

## License

MIT License
