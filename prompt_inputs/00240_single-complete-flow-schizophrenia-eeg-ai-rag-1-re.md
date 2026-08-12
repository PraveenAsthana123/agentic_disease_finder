# Request #240

- When: 2026-06-29 17:59:12 MDT
- Status: open

## Input
Single Complete Flow — Schizophrenia EEG → AI → RAG
1. Research Objective
   ↓
Define: screening support / biomarker discovery / severity tracking
   ↓
2. Data Collection
   ↓
EEG + clinical metadata + PANSS + medication + age + gender + diagnosis label
   ↓
3. Data Standardization
   ↓
Convert EDF/BDF/CSV/MAT → BIDS/FIF standard format
   ↓
4. Raw EEG Quality Check
   ↓
Sampling rate check → missing channel check → noise check → bad channel detection
   ↓
5. Preprocessing
   ↓
Bandpass filter 0.5–45 Hz
Notch filter 50/60 Hz
Re-reference EEG
Remove bad channels
ICA artifact removal
   ↓
6. Epoching / Segmentation
   ↓
Split continuous EEG into 2s / 4s / 8s windows
Use subject-level split to avoid leakage
   ↓
7. 1D EEG Signal Preparation
   ↓
Clean channel × time matrix
   ↓
8. Time-Frequency Transformation
   ↓
STFT → Spectrogram
CWT → Scalogram
SPWVD/STWVD → advanced research map
   ↓
9. EEG 1D → 2D Image Conversion
   ↓
Create 2D EEG images:
Spectrogram
Scalogram
Topomap
Connectivity matrix
Power-band heatmap
   ↓
10. Normalization + Standardization
   ↓
Log power normalization
MinMax scaling
Z-score standardization
Subject/session normalization
   ↓
11. Feature Extraction
   ↓
Spectral power
Relative band power
Entropy
Coherence
Phase locking value
Hjorth features
Fractal dimension
Asymmetry
CNN embeddings
   ↓
12. Feature Evaluation
   ↓
ANOVA
Mutual information
Correlation
SHAP ranking
Clinical relevance check
   ↓
13. Feature Selection
   ↓
LASSO
RFE
PCA
Boruta
SelectKBest
   ↓
14. Model Training
   ↓
Classical ML:
SVM / Random Forest / XGBoost

Deep Learning:
EEGNet / CNN / LSTM / Transformer / ViT
   ↓
15. Model Validation
   ↓
Subject-level cross-validation
Train/validation/test split
External dataset validation
   ↓
16. Model Evaluation
   ↓
Accuracy
Precision
Recall
F1-score
AUC
Sensitivity
Specificity
Confusion matrix
   ↓
17. Explainable AI
   ↓
SHAP for features
Saliency map for EEG images
Attention map for Transformer
Channel importance
Frequency-band importance
   ↓
18. RAG Knowledge Layer
   ↓
Index:
Research papers
EEG preprocessing SOPs
Clinical notes
Schizophrenia guidelines
Model cards
Experiment logs
   ↓
19. Retrieval
   ↓
Hybrid search:
Vector search + keyword search + metadata filter
   ↓
20. RAG Report Generation
   ↓
Combine:
Model prediction
Important EEG biomarkers
XAI explanation
Retrieved research evidence
Clinical metadata
   ↓
21. Human Review
   ↓
Psychiatrist / neurophysiologist review
Approve / reject / request more assessment
   ↓
22. Final Output
   ↓
Doctor-facing report:
Risk support score
EEG abnormality summary
Key biomarkers
Evidence citations
Limitations
Recommended next clinical step

Patient-facing report:
Simple explanation
No diagnosis claim
Follow-up recommendation
   ↓
23. Governance + Monitoring
   ↓
Audit logs
PII protection
Bias monitoring
Model drift
Data drift
Performance monitoring
Human override
Model versioning
