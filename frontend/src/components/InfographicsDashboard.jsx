import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, LineChart, Line, AreaChart, Area,
  ComposedChart, Legend, ScatterChart, Scatter
} from 'recharts'

const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#8bc34a', '#ff5722', '#607d8b']

const sections = [
  { id: 'system', label: 'System Architecture' },
  { id: 'datasets', label: 'Dataset Passport' },
  { id: 'data', label: 'Data Architecture' },
  { id: 'model', label: 'Model Architecture' },
  { id: 'genai', label: 'GenAI-RAG' },
  { id: 'stressrag', label: 'Stress-RAG Papers' },
  { id: 'pipeline', label: 'Pipeline Flows' },
  { id: 'validation', label: 'Validation Report' },
  { id: 'analysis', label: 'Analysis Charts' },
  { id: 'statistical', label: 'Statistical Analysis' },
  { id: 'sota', label: 'SOTA Comparison' },
  { id: 'governance', label: 'Governance & RAI' },
  { id: 'techniques', label: 'Techniques & Tools' },
  { id: 'metrics', label: 'Metrics Dashboard' }
]

// ── Data from IEEE papers (eeg_ieee_final.pdf, eeg_complete_master.pdf) ──

const datasetSpecs = [
  { dataset: 'SAM40', n: 40, mf: '22/18', ch: 32, hz: 256, device: 'BioSemi', duration: '60 min/subj', samples: 4800, label: 'STAI + HRV', task: 'Stress' },
  { dataset: 'DASPS', n: 23, mf: '12/11', ch: 14, hz: 128, device: 'Emotiv', duration: '45 min/subj', samples: 2760, label: 'DASS-21', task: 'Stress' },
  { dataset: 'MODA', n: 32, mf: '16/16', ch: 64, hz: 512, device: 'g.tec', duration: '30 min/subj', samples: 3840, label: 'Cortisol + EDA', task: 'Stress' },
  { dataset: 'CHB-MIT', n: 22, mf: '5/17', ch: 23, hz: 256, device: 'Clinical', duration: '844 hr', samples: 45890, label: 'Neurologist', task: 'Epilepsy' },
  { dataset: 'TUH-EEG', n: '100+', mf: '-', ch: 19, hz: 256, device: 'Clinical', duration: '1000+ hr', samples: 40100, label: 'Clinical report', task: 'Epilepsy' },
  { dataset: 'BCI-IV-2a', n: 9, mf: '9/0', ch: 22, hz: 250, device: 'Graz BCI', duration: '4.5 hr', samples: 5760, label: 'Cue paradigm', task: 'Motor Imagery' },
  { dataset: 'PhysioNet', n: 109, mf: '52/57', ch: 64, hz: 160, device: 'BCI2000', duration: '109 hr', samples: 13080, label: 'Cue paradigm', task: 'Motor Imagery' },
  { dataset: 'UNM-PD', n: 27, mf: '15/12', ch: 64, hz: 500, device: 'BioSemi', duration: '27 hr', samples: 1620, label: 'UPDRS', task: "Parkinson's" },
  { dataset: 'Sleep-EDF', n: 78, mf: '37/41', ch: 2, hz: 100, device: 'PSG', duration: '1560 hr', samples: 46800, label: 'R&K staging', task: 'Sleep' },
  { dataset: 'SHHS', n: 5804, mf: '2650/3154', ch: 2, hz: 125, device: 'PSG', duration: '11608 hr', samples: 870600, label: 'AASM staging', task: 'Sleep' }
]

const datasetPerformance = [
  { dataset: 'SAM40', acc: 91.2, prec: 90.8, rec: 89.5, f1: 0.908, auc: 0.956 },
  { dataset: 'DASPS', acc: 87.4, prec: 86.9, rec: 85.2, f1: 0.865, auc: 0.932 },
  { dataset: 'MODA', acc: 93.1, prec: 92.8, rec: 92.0, f1: 0.928, auc: 0.971 },
  { dataset: 'CHB-MIT', acc: 94.5, prec: 78.2, rec: 91.8, f1: 0.891, auc: 0.978 },
  { dataset: 'TUH-EEG', acc: 88.9, prec: 68.5, rec: 86.3, f1: 0.845, auc: 0.945 },
  { dataset: 'BCI-IV', acc: 82.6, prec: 81.5, rec: 80.1, f1: 0.812, auc: 0.905 },
  { dataset: 'PhysioNet', acc: 84.3, prec: 83.2, rec: 82.5, f1: 0.831, auc: 0.918 },
  { dataset: 'UNM-PD', acc: 89.7, prec: 88.9, rec: 88.1, f1: 0.892, auc: 0.952 },
  { dataset: 'Sleep-EDF', acc: 90.2, prec: 89.5, rec: 89.0, f1: 0.895, auc: 0.961 },
  { dataset: 'SHHS', acc: 89.8, prec: 88.2, rec: 88.4, f1: 0.891, auc: 0.958 }
]

const signalStats = [
  { dataset: 'SAM40', mean: 0.12, std: 45.3, skew: 0.08, kurt: 3.21 },
  { dataset: 'DASPS', mean: 0.08, std: 38.7, skew: 0.15, kurt: 3.45 },
  { dataset: 'MODA', mean: 0.05, std: 52.1, skew: -0.02, kurt: 2.98 },
  { dataset: 'CHB-MIT', mean: 0.21, std: 68.4, skew: 0.42, kurt: 4.82 },
  { dataset: 'TUH-EEG', mean: 0.18, std: 72.3, skew: 0.38, kurt: 5.12 },
  { dataset: 'BCI-IV', mean: 0.03, std: 41.2, skew: 0.05, kurt: 3.15 }
]

const featureBreakdown = [
  { name: 'Time Domain', count: 25, pct: 20, color: '#1e88e5' },
  { name: 'Hjorth', count: 3, pct: 2, color: '#00bcd4' },
  { name: 'Entropy', count: 5, pct: 4, color: '#4caf50' },
  { name: 'Frequency', count: 35, pct: 28, color: '#ff9800' },
  { name: 'Connectivity', count: 59, pct: 46, color: '#e91e63' }
]

const topFeatures = [
  { rank: 1, feature: 'Alpha power (Pz)', importance: 0.082 },
  { rank: 2, feature: 'Beta/Alpha ratio (F3)', importance: 0.071 },
  { rank: 3, feature: 'Theta power (Fz)', importance: 0.068 },
  { rank: 4, feature: 'Hjorth Complexity (C3)', importance: 0.065 },
  { rank: 5, feature: 'Sample Entropy (Cz)', importance: 0.058 },
  { rank: 6, feature: 'PLV (F3-F4)', importance: 0.054 },
  { rank: 7, feature: 'Gamma power (P3)', importance: 0.051 },
  { rank: 8, feature: 'Line Length (O1)', importance: 0.048 },
  { rank: 9, feature: 'Spectral Edge 95% (Pz)', importance: 0.045 },
  { rank: 10, feature: 'Delta power (Fp1)', importance: 0.042 }
]

// Validation Report data (validation_report.pdf)
const validationPhases = [
  { phase: 'Phase 1', name: 'Data Input & Loading', score: 94.5, finding: '10/10 datasets loaded, 875K samples' },
  { phase: 'Phase 2', name: 'Signal Preprocessing', score: 92.8, finding: 'SNR improved +9.7dB, 95% artifact removal' },
  { phase: 'Phase 3', name: 'Feature Extraction', score: 95.2, finding: '127 features validated, no NaN/Inf' },
  { phase: 'Phase 4', name: 'Model Training', score: 91.5, finding: '89.3% ensemble accuracy, ECE=0.023' },
  { phase: 'Phase 5', name: 'RAG Explainability', score: 88.5, finding: '500K knowledge base, 89% factual accuracy' },
  { phase: 'Phase 6', name: 'Integration & QC', score: 93.0, finding: 'All quality gates passed' }
]

const preprocessingScores = [
  { step: 'Bandpass Filter', score: 98 },
  { step: 'Notch Filter', score: 99 },
  { step: 'ICA Artifact', score: 88 },
  { step: 'CAR Reference', score: 100 },
  { step: 'Segmentation', score: 96 },
  { step: 'Normalization', score: 97 }
]

// Phase 1: Per-dataset validation scores (validation_report.pdf page 2)
const datasetValidationScores = [
  { dataset: 'SAM40', score: 98.9 },
  { dataset: 'DASPS', score: 98.7 },
  { dataset: 'BCI_IV_2a', score: 100.0 },
  { dataset: 'PhysioNet_MI', score: 98.3 },
  { dataset: 'Sleep_EDF', score: 98.8 },
  { dataset: 'SHHS', score: 98.8 },
  { dataset: 'CHB_MIT', score: 100.0 },
  { dataset: 'TUH_EEG', score: 96.6 },
  { dataset: 'MODA', score: 99.2 },
  { dataset: 'UNM_PD', score: 100.0 }
]

const datasetIssues = [
  { dataset: 'SAM40', issue: '2 recordings with truncated segments' },
  { dataset: 'DASPS', issue: 'Some medium anxiety labels ambiguous' },
  { dataset: 'PhysioNet_MI', issue: '5 subjects with missing channels, 3 corrupted files' },
  { dataset: 'Sleep_EDF', issue: 'Some recordings shorter than 6 hours' },
  { dataset: 'SHHS', issue: '12 recordings with annotation gaps' },
  { dataset: 'TUH_EEG', issue: 'Variable channel configurations, some ambiguous labels' },
  { dataset: 'MODA', issue: 'Inter-rater variability in spindle annotations' }
]

// Phase 2: Quality improvements (validation_report.pdf page 3)
const qualityImprovements = [
  { metric: 'SNR Improvement', value: 9.7, unit: 'dB' },
  { metric: 'Eye Blink Removal', value: 95.2, unit: '%' },
  { metric: 'Muscle Artifact', value: 87.3, unit: '%' },
  { metric: 'Line Noise Removal', value: 99.1, unit: '%' },
  { metric: 'Signal Preserved', value: 98.5, unit: '%' }
]

const snrImprovement = { before: 8.5, after: 18.2, gain: 9.7 }

const filterSpecs = [
  { filter: 'Bandpass', type: '4th order Butterworth', params: '0.5-45 Hz', ripple: '< 0.5 dB', atten: '> 40 dB', phase: 'Zero-phase (filtfilt)' },
  { filter: 'Notch', type: 'IIR Notch', params: '50/60 Hz', qFactor: '30', bw: '2 Hz', atten: '> 30 dB' },
  { filter: 'ICA', type: 'FastICA', params: '20 components', detect: 'Correlation + Kurtosis', fpr: '3.2%' }
]

// Phase 3: Feature category validation (validation_report.pdf page 4)
const featureCategoryValidation = [
  { category: 'Time Domain', count: 25, score: 98 },
  { category: 'Hjorth', count: 3, score: 99 },
  { category: 'Entropy', count: 5, score: 94 },
  { category: 'Frequency', count: 35, score: 98 },
  { category: 'Connectivity', count: 59, score: 91 }
]

const conversionValidation = [
  { method: 'CWT (Morlet)', score: 96.0 },
  { method: 'STFT (Hann)', score: 95.5 }
]

const featureQualityChecks = { nanValues: '0.0%', infValues: '0.0%', constantFeatures: 0, highlyCorrelated: 12 }

// Phase 4: Model accuracy and overfitting (validation_report.pdf page 5)
const modelAccuracyComparison = [
  { model: 'EEGNet', acc: 84.5, params: '2,548', trainAcc: 89.2, gap: 4.7 },
  { model: '2D-CNN', acc: 86.2, params: '148,000', trainAcc: 90.1, gap: 3.9 },
  { model: 'Transformer', acc: 87.8, params: '502,000', trainAcc: 91.2, gap: 3.4 },
  { model: 'Ensemble', acc: 89.3, params: '-', trainAcc: null, gap: null }
]

const ensembleMetrics = [
  { metric: 'Accuracy', value: 89.3 },
  { metric: 'Precision', value: 88.7 },
  { metric: 'Recall', value: 89.8 },
  { metric: 'F1', value: 89.2 },
  { metric: 'AUC-ROC', value: 95.2 }
]

// Phase 5: RAG component validation (validation_report.pdf page 6)
const ragComponentValidation = [
  { component: 'Knowledge Base', score: 95 },
  { component: 'Embedding Model', score: 92 },
  { component: 'Retrieval', score: 88 },
  { component: 'Generation', score: 85 }
]

const ragRetrievalPerf = [
  { metric: 'Precision@5', value: 82 },
  { metric: 'Recall@5', value: 78 },
  { metric: 'MRR', value: 85 }
]

const ragHumanEval = [
  { criterion: 'Readability', score: 4.5 },
  { criterion: 'Completeness', score: 3.8 },
  { criterion: 'Accuracy', score: 4.0 },
  { criterion: 'Helpfulness', score: 4.2 }
]

const ragSystemSpecs = [
  { component: 'Knowledge Base', specs: 'PubMed, 500K abstracts, Neuroscience/EEG/Brain disorders' },
  { component: 'Embedding', specs: 'all-MiniLM-L6-v2, 384 dim, fine-tuned on medical text' },
  { component: 'Retrieval', specs: 'FAISS (IVF-PQ), Top-K=5, Latency <50ms' },
  { component: 'Generation', specs: 'GPT-4 / Claude, 89% factual accuracy, 87% citation accuracy' },
  { component: 'Evaluation', specs: '5 experts, 100 samples, Inter-rater κ=0.72' }
]

// ── eeg-stress-rag papers data (v1 + v2) ──

// v2 Stress Dataset Specs (Tables III-V)
const stressDatasetSpecs = [
  { dataset: 'DEAP', role: 'Benchmark (Arousal Proxy)', subjects: 32, channels: 32, hz: '512→128', trials: 1280, label: 'Arousal ≥5', task: 'Video watching' },
  { dataset: 'SAM-40', role: 'Primary (Cognitive Stress)', subjects: 40, channels: 32, hz: 256, trials: 480, label: 'Task vs Rest', task: 'Stroop/Arithmetic' },
  { dataset: 'EEGMAT', role: 'Supplementary (Workload)', subjects: 25, channels: 14, hz: 128, trials: 500, label: 'Task difficulty', task: 'N-back/Arithmetic' }
]

const stressClassBalance = [
  { dataset: 'DEAP', high: 612, low: 668, ratio: '0.92:1', status: 'Mild' },
  { dataset: 'SAM-40', high: 240, low: 240, ratio: '1.00:1', status: 'Balanced' },
  { dataset: 'EEGMAT', high: 250, low: 250, ratio: '1.00:1', status: 'Balanced' }
]

const epochRejection = [
  { dataset: 'DEAP', total: 15360, rejected: 921, retained: 14439, rate: '6.0%' },
  { dataset: 'SAM-40', total: 5760, rejected: 403, retained: 5357, rate: '7.0%' },
  { dataset: 'EEGMAT', total: 6000, rejected: 480, retained: 5520, rate: '8.0%' }
]

// v2 Model Parameters (Table XIX) - 159,372 total
const v2ModelParams = [
  { component: 'Conv Block', layer: 'Conv1D layers (3) + BatchNorm', params: 30176 },
  { component: 'Bi-LSTM', layer: 'Forward + Backward (hidden=64)', params: 99584 },
  { component: 'Attention', layer: 'Wₐ, wₐ, bias', params: 8321 },
  { component: 'Classifier', layer: 'FC layers (128→256→128→2)', params: 10402 },
  { component: 'Text Encoder', layer: 'Projection (384→128)', params: 49280 }
]

// v1 Model Parameters (Table XIX) - 265,219 total
const v1ModelParams = [
  { component: 'Conv Block 1', layer: 'Conv1D(1→64,k=7)+BN', params: 640, cumulative: 640 },
  { component: 'Conv Block 2', layer: 'Conv1D(64→128,k=5)+BN', params: 41344, cumulative: 41984 },
  { component: 'Conv Block 3', layer: 'Conv1D(128→64,k=3)+BN', params: 24768, cumulative: 66752 },
  { component: 'Bi-LSTM', layer: 'Forward+Backward (64→64)', params: 99584, cumulative: 166336 },
  { component: 'Attention', layer: 'Wₐ(128×64)+wₐ(64×1)', params: 8321, cumulative: 174657 },
  { component: 'Text Encoder', layer: 'Projection (384→128)', params: 49280, cumulative: 223937 },
  { component: 'Classifier', layer: 'FC1+FC2+Output', params: 41282, cumulative: 265219 }
]

// v2 Per-Dataset Metrics with 95% CI (Tables XXIII-XXV)
const v2PerDatasetMetrics = [
  { dataset: 'DEAP', acc: 94.7, ci: '[93.2, 96.2]', prec: 0.945, rec: 0.948, f1: 0.943, kappa: 0.894, auc: 0.978, pr: 0.971, mcc: 0.894 },
  { dataset: 'SAM-40', acc: 93.2, ci: '[91.5, 94.9]', prec: 0.931, rec: 0.933, f1: 0.928, kappa: 0.864, auc: 0.968, pr: 0.958, mcc: 0.864 },
  { dataset: 'EEGMAT', acc: 91.8, ci: '[89.8, 93.8]', prec: 0.915, rec: 0.921, f1: 0.912, kappa: 0.836, auc: 0.956, pr: 0.942, mcc: 0.836 }
]

// v2 Cross-Dataset Transfer (Table XXVII)
const v2CrossTransfer = [
  { train: 'DEAP', test: 'SAM-40', acc: 71.4, f1: 0.70, delta: -22, note: 'Arousal≠Stress' },
  { train: 'SAM-40', test: 'DEAP', acc: 68.2, f1: 0.67, delta: -27, note: 'Stress≠Arousal' },
  { train: 'SAM-40', test: 'EEGMAT', acc: 78.6, f1: 0.77, delta: -13, note: 'Similar paradigm' },
  { train: 'EEGMAT', test: 'SAM-40', acc: 76.8, f1: 0.75, delta: -16, note: 'Moderate' },
  { train: 'DEAP', test: 'EEGMAT', acc: 65.4, f1: 0.64, delta: -26, note: 'Poor' },
  { train: 'EEGMAT', test: 'DEAP', acc: 63.8, f1: 0.62, delta: -28, note: 'Poor' }
]

// v1 Cross-Dataset Transfer (Table XI)
const v1CrossTransfer = [
  { train: 'DEAP', test: 'SAM-40', acc: '68.4±4.2', delta: -13.5, f1: 0.712 },
  { train: 'DEAP', test: 'WESAD', acc: '82.1±3.8', delta: -17.9, f1: 0.834 },
  { train: 'SAM-40', test: 'DEAP', acc: '71.2±5.1', delta: -23.5, f1: 0.726 },
  { train: 'SAM-40', test: 'WESAD', acc: '76.8±4.5', delta: -23.2, f1: 0.781 },
  { train: 'WESAD', test: 'DEAP', acc: '74.6±4.8', delta: -25.4, f1: 0.758 },
  { train: 'WESAD', test: 'SAM-40', acc: '65.2±5.3', delta: -16.7, f1: 0.684 }
]

// v2 Confusion Matrices (Fig 13)
const v2ConfusionData = [
  { dataset: 'DEAP (Arousal Proxy)', tn: 612, fp: 38, fn: 32, tp: 598 },
  { dataset: 'SAM-40 (Cognitive Stress)', tn: 224, fp: 16, fn: 17, tp: 223 },
  { dataset: 'EEGMAT (Workload Proxy)', tn: 229, fp: 21, fn: 20, tp: 230 }
]

// v1 Confusion Matrices (Fig 21)
const v1ConfusionData = [
  { dataset: 'DEAP', tn: 302, fp: 18, fn: 16, tp: 304 },
  { dataset: 'SAM-40', tn: 164, fp: 36, fn: 16, tp: 184 },
  { dataset: 'WESAD', tn: 150, fp: 0, fn: 0, tp: 150 }
]

// v2 Subject-Wise Boxplot Data (Fig 12)
const subjectWiseBoxplot = [
  { dataset: 'DEAP', median: 95.1, iqr: 3.6, q1: 93.2, q3: 96.4, min: 79, max: 100 },
  { dataset: 'SAM-40', median: 93.8, iqr: 4.2, q1: 91.4, q3: 95.1, min: 80, max: 100 },
  { dataset: 'EEGMAT', median: 92.4, iqr: 4.6, q1: 89.6, q3: 94.2, min: 78, max: 100 }
]

// v2 Band Power Analysis per dataset (Tables XXVIII-XXX)
const v2BandPowerDEAP = [
  { band: 'Delta', low: 12.4, high: 14.1, tStat: 2.87, p: '0.004' },
  { band: 'Theta', low: 8.7, high: 11.2, tStat: 5.94, p: '<0.001' },
  { band: 'Alpha', low: 15.3, high: 10.8, tStat: -6.12, p: '<0.001' },
  { band: 'Beta', low: 6.2, high: 9.4, tStat: 8.47, p: '<0.001' },
  { band: 'Gamma', low: 2.1, high: 3.2, tStat: 6.23, p: '<0.001' }
]

const v2BandPowerSAM40 = [
  { band: 'Delta', low: 11.8, high: 13.5, tStat: 2.54, p: '0.012' },
  { band: 'Theta', low: 9.2, high: 12.1, tStat: 5.67, p: '<0.001' },
  { band: 'Alpha', low: 14.8, high: 10.2, tStat: -5.89, p: '<0.001' },
  { band: 'Beta', low: 5.9, high: 8.8, tStat: 7.82, p: '<0.001' },
  { band: 'Gamma', low: 1.9, high: 2.9, tStat: 5.78, p: '<0.001' }
]

const v2BandPowerEEGMAT = [
  { band: 'Delta', low: 13.1, high: 14.8, tStat: 2.21, p: '0.028' },
  { band: 'Theta', low: 8.9, high: 11.5, tStat: 5.21, p: '<0.001' },
  { band: 'Alpha', low: 14.2, high: 9.8, tStat: -5.62, p: '<0.001' },
  { band: 'Beta', low: 6.5, high: 9.1, tStat: 6.94, p: '<0.001' },
  { band: 'Gamma', low: 2.3, high: 3.4, tStat: 5.12, p: '<0.001' }
]

// v2 Alpha Suppression (Table XXXI)
const alphaSuppression = [
  { dataset: 'DEAP', baseline: 15.3, stress: 10.8, suppression: '29%', p: '<.001' },
  { dataset: 'SAM-40', baseline: 14.8, stress: 10.2, suppression: '31%', p: '<.001' },
  { dataset: 'EEGMAT', baseline: 14.2, stress: 9.8, suppression: '31%', p: '<.001' }
]

// v2 Theta/Beta Ratio (Table XXXII)
const thetaBetaRatio = [
  { dataset: 'DEAP', low: 1.40, high: 1.19, delta: '-15%', d: 0.70, p: '<.001' },
  { dataset: 'SAM-40', low: 1.56, high: 1.38, delta: '-12%', d: 0.50, p: '<.001' },
  { dataset: 'EEGMAT', low: 1.37, high: 1.26, delta: '-8%', d: 0.34, p: '.003' }
]

// v2 Feature Importance per dataset (Tables XLII-XLIV)
const featureImpDEAP = [
  { feature: 'Alpha power (Fz)', importance: 0.142 },
  { feature: 'Beta power (F3)', importance: 0.128 },
  { feature: 'Frontal asymmetry', importance: 0.115 },
  { feature: 'Theta/Beta ratio', importance: 0.098 },
  { feature: 'Alpha power (Pz)', importance: 0.087 },
  { feature: 'Beta power (Cz)', importance: 0.076 },
  { feature: 'Theta power (F4)', importance: 0.068 },
  { feature: 'wPLI (F3-F4, α)', importance: 0.054 },
  { feature: 'Gamma power (F3)', importance: 0.048 },
  { feature: 'Alpha power (C3)', importance: 0.042 }
]

const featureImpSAM40 = [
  { feature: 'Alpha power (F4)', importance: 0.138 },
  { feature: 'Beta power (Fz)', importance: 0.125 },
  { feature: 'Theta power (F3)', importance: 0.108 },
  { feature: 'Frontal asymmetry', importance: 0.095 },
  { feature: 'Alpha power (Pz)', importance: 0.082 },
  { feature: 'Theta/Beta ratio', importance: 0.071 },
  { feature: 'Beta power (C3)', importance: 0.062 },
  { feature: 'wPLI (F3-F4, β)', importance: 0.051 },
  { feature: 'Gamma power (Fz)', importance: 0.045 },
  { feature: 'Alpha power (O1)', importance: 0.039 }
]

// v2 Channel × Band Importance Matrix (Table XLV)
const channelBandMatrix = [
  { region: 'Frontal', delta: 0.024, theta: 0.089, alpha: 0.142, beta: 0.128, gamma: 0.048, total: 0.431 },
  { region: 'Central', delta: 0.018, theta: 0.076, alpha: 0.068, beta: 0.052, gamma: 0.032, total: 0.246 },
  { region: 'Parietal', delta: 0.015, theta: 0.045, alpha: 0.062, beta: 0.054, gamma: 0.028, total: 0.204 },
  { region: 'Temporal', delta: 0.012, theta: 0.035, alpha: 0.038, beta: 0.032, gamma: 0.022, total: 0.139 },
  { region: 'Occipital', delta: 0.008, theta: 0.021, alpha: 0.028, beta: 0.024, gamma: 0.015, total: 0.096 }
]

// v2 Comprehensive Benchmark on DEAP (Table L)
const v2Benchmark = [
  { model: 'SVM (RBF)', type: 'ML', params: '-', acc: 82.3, f1: 0.82, auc: 0.89, mcc: 0.65 },
  { model: 'Random Forest', type: 'ML', params: '-', acc: 84.1, f1: 0.84, auc: 0.91, mcc: 0.68 },
  { model: 'XGBoost', type: 'ML', params: '-', acc: 85.6, f1: 0.85, auc: 0.92, mcc: 0.71 },
  { model: 'CNN', type: 'DL', params: '45K', acc: 86.5, f1: 0.86, auc: 0.93, mcc: 0.73 },
  { model: 'LSTM', type: 'DL', params: '82K', acc: 87.2, f1: 0.87, auc: 0.93, mcc: 0.74 },
  { model: 'CNN-LSTM', type: 'DL', params: '125K', acc: 89.8, f1: 0.89, auc: 0.95, mcc: 0.80 },
  { model: 'EEGNet', type: 'DL', params: '2.6K', acc: 90.4, f1: 0.90, auc: 0.95, mcc: 0.81 },
  { model: 'DGCNN', type: 'GNN', params: '180K', acc: 91.2, f1: 0.91, auc: 0.96, mcc: 0.82 },
  { model: 'GenAI-RAG-EEG', type: 'Hybrid', params: '159K', acc: 94.7, f1: 0.94, auc: 0.97, mcc: 0.89 }
]

// v2 Ablation Study (Table LI)
const v2AblationStudy = [
  { config: 'Full Model', acc: 94.7, f1: 0.943, delta: 0, p: '-' },
  { config: '- Text Encoder', acc: 91.2, f1: 0.906, delta: -3.5, p: '0.003' },
  { config: '- Attention', acc: 92.5, f1: 0.919, delta: -2.2, p: '0.012' },
  { config: '- Bi-LSTM', acc: 88.4, f1: 0.877, delta: -6.3, p: '<0.001' },
  { config: '- RAG Module', acc: 94.5, f1: 0.941, delta: -0.2, p: '0.312' },
  { config: 'CNN Baseline', acc: 86.5, f1: 0.858, delta: -8.2, p: '<0.001' }
]

// v2 Component Importance (Fig 21)
const componentImportance = [
  { component: 'Bi-LSTM', contribution: 6.3 },
  { component: 'CNN Blocks', contribution: 3.6 },
  { component: 'Self-Attention', contribution: 2.6 },
  { component: 'Context Encoder', contribution: 0.9 },
  { component: 'RAG Module', contribution: 0.2 }
]

// v2 Cumulative Ablation (Fig 22)
const cumulativeAblation = [
  { step: 'Full Model', acc: 93.2 },
  { step: '-RAG', acc: 93.0 },
  { step: '-Context', acc: 91.3 },
  { step: '-Attention', acc: 88.7 },
  { step: '-Bi-LSTM', acc: 82.4 },
  { step: '-CNN', acc: 65.5 }
]

// v2 Statistical Robustness (Tables LII-LIV)
const statRobustness = [
  { dataset: 'DEAP', mean: 94.7, median: 95.1, q1: 93.2, q3: 96.4, iqr: 3.2, ciLow: 93.2, ciHigh: 96.2, std: 2.3, cv: '2.4%' },
  { dataset: 'SAM-40', mean: 93.2, median: 93.6, q1: 91.4, q3: 95.1, iqr: 3.7, ciLow: 91.5, ciHigh: 94.9, std: 2.6, cv: '2.8%' },
  { dataset: 'EEGMAT', mean: 91.8, median: 92.1, q1: 89.6, q3: 94.2, iqr: 4.6, ciLow: 89.8, ciHigh: 93.8, std: 3.1, cv: '3.4%' }
]

// v2 Statistical Significance (Tables LV-LVIII)
const statSignificance = [
  { dataset: 'DEAP', wilcoxP: '<.001', tP: '<.001', mwP: '<.001', d: 1.47, r: 0.91 },
  { dataset: 'SAM-40', wilcoxP: '<.001', tP: '<.001', mwP: '<.001', d: 1.32, r: 0.88 },
  { dataset: 'EEGMAT', wilcoxP: '<.001', tP: '<.001', mwP: '<.001', d: 1.18, r: 0.85 }
]

// v2 Confound Analysis (Tables XLVI-XLVIII)
const confoundAnalysis = [
  { dataset: 'DEAP', artifactLow: '5.8%±2.1', artifactHigh: '6.2%±2.4', diff: '+0.4%', p: '0.412' },
  { dataset: 'SAM-40', artifactLow: '6.5%±2.5', artifactHigh: '7.4%±2.8', diff: '+0.9%', p: '0.187' },
  { dataset: 'EEGMAT', artifactLow: '7.6%±2.9', artifactHigh: '8.4%±3.2', diff: '+0.8%', p: '0.234' }
]

// v2 Hyperparameter Sensitivity (Figs 8-11)
const hpLearningRate = [
  { lr: '1e-5', deap: 72, sam40: 70, eegmat: 68 },
  { lr: '3e-5', deap: 80, sam40: 78, eegmat: 76 },
  { lr: '1e-4', deap: 94.7, sam40: 93.2, eegmat: 91.8 },
  { lr: '3e-4', deap: 92, sam40: 90, eegmat: 88 },
  { lr: '1e-3', deap: 85, sam40: 83, eegmat: 80 }
]

const hpBatchSize = [
  { size: '16', accuracy: 88, f1: 86 },
  { size: '32', accuracy: 91, f1: 89 },
  { size: '64', accuracy: 94.7, f1: 94 },
  { size: '128', accuracy: 92, f1: 90 }
]

const hpDropout = [
  { rate: '0.1', train: 95.5, val: 89 },
  { rate: '0.2', train: 95, val: 92 },
  { rate: '0.3', train: 94, val: 94.7 },
  { rate: '0.4', train: 92, val: 93 },
  { rate: '0.5', train: 90, val: 90 }
]

// v1 Band Power (Table XIV)
const v1BandPower = [
  { band: 'Delta', stress: 0.771, baseline: 0.947, d: -0.444, p: '<0.001' },
  { band: 'Theta', stress: 6.669, baseline: 8.261, d: -0.486, p: '<0.001' },
  { band: 'Alpha', stress: 3.875, baseline: 4.339, d: -0.295, p: '0.003' },
  { band: 'Beta', stress: 10.685, baseline: 12.685, d: -0.327, p: '<0.001' },
  { band: 'Gamma', stress: 8.782, baseline: 9.387, d: -0.157, p: '0.142' }
]

// v1 Classification Results (Table VI)
const v1ClassResults = [
  { dataset: 'SAM-40', acc: 81.9, prec: 0.851, rec: 0.920, f1: 0.884, auc: 0.780, mcc: 0.485 },
  { dataset: 'DEAP', acc: 94.7, prec: 0.943, rec: 0.951, f1: 0.947, auc: 0.982, mcc: 0.894 },
  { dataset: 'WESAD', acc: 100.0, prec: 1.000, rec: 1.000, f1: 1.000, auc: 1.000, mcc: 1.000 }
]

// v1 Feature Importance (Fig 11)
const v1FeatureImportance = [
  { feature: 'Beta Power', importance: 0.18 },
  { feature: 'Alpha Power', importance: 0.16 },
  { feature: 'Alpha/Beta Ratio', importance: 0.14 },
  { feature: 'Theta Power', importance: 0.12 },
  { feature: 'Beta Entropy', importance: 0.10 },
  { feature: 'Alpha Asymmetry', importance: 0.09 },
  { feature: 'Gamma Power', importance: 0.06 },
  { feature: 'Delta Power', importance: 0.06 },
  { feature: 'Theta/Alpha Ratio', importance: 0.04 },
  { feature: 'Spectral Centroid', importance: 0.03 }
]

// v1 Feature Selection Methods (Fig 24)
const featureSelectionMethods = [
  { method: 'Filter (Corr)', features: 'Beta, Alpha, AlphaAsym', acc: 91.2 },
  { method: 'RFE', features: 'Alpha, Theta, AlphaAsym, Gamma', acc: 94.7 },
  { method: 'LASSO', features: 'Alpha, AlphaAsym, Theta, Gamma', acc: 93.8 },
  { method: 'RF Importance', features: 'Beta, Alpha, AlphaAsym, Theta', acc: 93.2 },
  { method: 'Mutual Info', features: 'Beta, Alpha, AlphaAsym, Theta', acc: 92.5 },
  { method: 'Consensus', features: 'Beta, Alpha, AlphaAsym, Theta', acc: 94.7 }
]

// v1 Computational Complexity (Table X)
const computeComplexity = [
  { model: 'SVM (RBF)', params: '-', memory: '12 MB', inference: '2.1 ms', gpu: 'No' },
  { model: 'Random Forest', params: '-', memory: '45 MB', inference: '3.8 ms', gpu: 'No' },
  { model: 'CNN', params: '45K', memory: '24 MB', inference: '1.2 ms', gpu: 'Yes' },
  { model: 'EEGNet', params: '2.6K', memory: '8 MB', inference: '0.8 ms', gpu: 'Yes' },
  { model: 'CNN-LSTM', params: '125K', memory: '52 MB', inference: '3.4 ms', gpu: 'Yes' },
  { model: 'DGCNN', params: '180K', memory: '68 MB', inference: '4.2 ms', gpu: 'Yes' },
  { model: 'Ours', params: '257K', memory: '86 MB', inference: '4.8 ms', gpu: 'Yes' },
  { model: 'Ours+RAG', params: '257K', memory: '142 MB', inference: '128 ms', gpu: 'Yes' }
]

// v1 Clinical Validation (Table XV) - multi-task
const v1ClinicalVal = [
  { dataset: 'DEAP', task: 'Binary', acc: 94.7, f1: 0.947, kappa: 0.893, auc: 0.982 },
  { dataset: 'DEAP', task: 'Workload (3)', acc: 87.2, f1: 0.868, kappa: 0.808, auc: 0.954 },
  { dataset: 'DEAP', task: 'Cognitive (4)', acc: 82.4, f1: 0.821, kappa: 0.765, auc: 0.921 },
  { dataset: 'SAM-40', task: 'Binary', acc: 81.9, f1: 0.884, kappa: 0.475, auc: 0.780 },
  { dataset: 'SAM-40', task: 'Workload (3)', acc: 74.6, f1: 0.742, kappa: 0.619, auc: 0.842 },
  { dataset: 'WESAD', task: 'Binary', acc: 100.0, f1: 1.000, kappa: 1.000, auc: 1.000 },
  { dataset: 'WESAD', task: 'Workload (3)', acc: 96.8, f1: 0.965, kappa: 0.952, auc: 0.994 }
]

// v1 Subject-Wise LOSO (Table XVII)
const v1SubjectLOSO = [
  { group: 'Subjects 1-8', acc: 92.4, std: 3.8, min: 86.2, max: 97.1 },
  { group: 'Subjects 9-16', acc: 91.8, std: 4.2, min: 84.5, max: 96.8 },
  { group: 'Subjects 17-24', acc: 90.6, std: 5.1, min: 82.1, max: 97.5 },
  { group: 'Subjects 25-32', acc: 89.2, std: 4.6, min: 81.8, max: 95.2 }
]

// v2 Same-Pipeline Baseline (Table XLIX)
const samePipelineBaseline = [
  { data: 'DEAP', method: 'BP+LDA', acc: 78.4, f1: 0.77, auc: 0.84 },
  { data: 'DEAP', method: 'BP+SVM', acc: 82.3, f1: 0.81, auc: 0.88 },
  { data: 'DEAP', method: 'Ours', acc: 94.7, f1: 0.94, auc: 0.98 },
  { data: 'SAM', method: 'BP+LDA', acc: 74.2, f1: 0.73, auc: 0.81 },
  { data: 'SAM', method: 'BP+SVM', acc: 78.6, f1: 0.77, auc: 0.86 },
  { data: 'SAM', method: 'Ours', acc: 93.2, f1: 0.93, auc: 0.97 },
  { data: 'MAT', method: 'BP+LDA', acc: 72.8, f1: 0.71, auc: 0.80 },
  { data: 'MAT', method: 'BP+SVM', acc: 76.4, f1: 0.75, auc: 0.83 },
  { data: 'MAT', method: 'Ours', acc: 91.8, f1: 0.91, auc: 0.96 }
]

// GenAI-RAG data (genai_rag_eeg_v4.pdf)
const ragPerformance = [
  { dataset: 'DEAP', acc: 94.7, prec: 94.5, rec: 94.1, f1: 94.3, auc: 96.7, kappa: 0.894 },
  { dataset: 'SAM-40', acc: 93.2, prec: 93.0, rec: 92.6, f1: 92.8, auc: 95.8, kappa: 0.864 },
  { dataset: 'WESAD', acc: 100.0, prec: 100.0, rec: 100.0, f1: 100.0, auc: 100.0, kappa: 1.000 }
]

const ragExplanationEval = [
  { criterion: 'Scientific Accuracy', agreement: 91.2, rating: 4.3 },
  { criterion: 'Clinical Relevance', agreement: 88.4, rating: 4.1 },
  { criterion: 'Coherence & Readability', agreement: 92.1, rating: 4.4 },
  { criterion: 'Evidence Grounding', agreement: 87.5, rating: 4.0 }
]

const ablationStudy = [
  { config: 'Full Model', acc: 93.2, delta: 0 },
  { config: '- Bi-LSTM', acc: 89.6, delta: -3.6 },
  { config: '- Self-Attention', acc: 91.1, delta: -2.1 },
  { config: '- Context Encoder', acc: 91.5, delta: -1.7 },
  { config: '- RAG Module', acc: 93.0, delta: -0.2 },
  { config: 'CNN Only', acc: 89.6, delta: -3.6 }
]

const bandPowerEffects = [
  { band: 'Delta', deap: 0.38, sam40: 0.42, wesad: 0.35 },
  { band: 'Theta', deap: 0.62, sam40: 0.68, wesad: 0.55 },
  { band: 'Alpha', deap: -0.82, sam40: -0.89, wesad: -0.75 },
  { band: 'Beta', deap: 0.71, sam40: 0.74, wesad: 0.58 },
  { band: 'Gamma', deap: 0.48, sam40: 0.51, wesad: 0.41 }
]

// SOTA comparison
const sotaComparison = [
  { method: 'EEGNet', dataset: 'BCI-IV', acc: 79.8, f1: 0.78, year: 2018 },
  { method: 'DeepConvNet', dataset: 'BCI-IV', acc: 81.2, f1: 0.80, year: 2017 },
  { method: 'ShallowNet', dataset: 'BCI-IV', acc: 78.5, f1: 0.77, year: 2017 },
  { method: 'Riemannian', dataset: 'BCI-IV', acc: 78.5, f1: 0.77, year: 2012 },
  { method: 'DeepSleepNet', dataset: 'Sleep-EDF', acc: 82.0, f1: 0.76, year: 2017 },
  { method: 'TSception', dataset: 'SEED', acc: 85.6, f1: 0.84, year: 2021 },
  { method: 'Conformer', dataset: 'SEED', acc: 87.2, f1: 0.86, year: 2022 },
  { method: 'ATCNet', dataset: 'BCI-IV', acc: 85.4, f1: 0.84, year: 2022 },
  { method: 'BENDR', dataset: 'Multiple', acc: 86.5, f1: 0.85, year: 2021 },
  { method: 'EEG-NeXT', dataset: 'SEED', acc: 88.1, f1: 0.87, year: 2023 },
  { method: 'Ours', dataset: 'Multi-10', acc: 89.2, f1: 0.876, year: 2024 }
]

const modelTrainingLadder = [
  { level: 1, model: 'LR + Bandpower', params: '100', time: '1 min', purpose: 'Fast sanity check' },
  { level: 2, model: 'SVM + Riemannian', params: '500', time: '5 min', purpose: 'Strong EEG baseline' },
  { level: 3, model: 'EEGNet (1D CNN)', params: '2.5K', time: '30 min', purpose: 'Temporal patterns' },
  { level: 4, model: '2D CNN + CWT', params: '150K', time: '2 hr', purpose: 'Time-frequency' },
  { level: 5, model: 'Transformer', params: '500K', time: '4 hr', purpose: 'Attention mechanism' }
]

const productionKPIs = [
  { metric: 'P50 Latency', target: '<50ms', achieved: '32ms', status: true },
  { metric: 'P95 Latency', target: '<100ms', achieved: '78ms', status: true },
  { metric: 'P99 Latency', target: '<200ms', achieved: '145ms', status: true },
  { metric: 'Throughput', target: '>100/s', achieved: '156/s', status: true },
  { metric: 'Availability', target: '>99.9%', achieved: '99.95%', status: true },
  { metric: 'GPU Memory', target: '<4GB', achieved: '2.8GB', status: true },
  { metric: 'Error Rate', target: '<0.1%', achieved: '0.05%', status: true }
]

// 8-Phase Analysis Plan (eeg_analysis_plan.pdf)
const analysisPhases = [
  { phase: 'Phase 1', name: 'Data Analysis', modules: 9, items: 'Distribution, Temporal, Spectral, Correlation, Missing Values, Outliers, Class Balance, DQS, Features' },
  { phase: 'Phase 2', name: 'Model Analysis', modules: 14, items: 'Architecture, Params, FLOPS, Layers, Attention, Gradients, Convergence, LR Schedule, Weights, Activations, Receptive Field, Feature Maps, Pruning, Quantization' },
  { phase: 'Phase 3', name: 'Performance', modules: 14, items: 'Confusion Matrix, ROC-AUC, PR Curves, Threshold Sweep, Per-Class, Error Analysis, Calibration, Prediction Dist, Multi-Label, Ranking, Clinical, Cost-Sensitive, Ensemble, Cross-Dataset' },
  { phase: 'Phase 4', name: 'Subject Analysis', modules: 14, items: 'LOSO, Ranking, Variance, Demographics, Hard Subjects, Clustering, Calibration, Adaptation, Session, Embedding, Transfer, Personalization, Selection, Bias Audit' },
  { phase: 'Phase 5', name: 'Sensitivity', modules: 16, items: 'Noise Injection, Channel Dropout, Temporal Masking, Frequency Perturbation, Artifact Sim, Sampling Rate, Window Size, Overlap, HPO Grid, Ablation, Feature Importance, Input Gradients, Adversarial, Domain Shift, Label Noise, Data Volume' },
  { phase: 'Phase 6', name: 'Statistical', modules: 14, items: 'Paired t-test, Wilcoxon, McNemar, Bootstrap CI, Effect Size, ANOVA/Friedman, Post-hoc, Critical Difference, Power Analysis, Normality, Variance Homogeneity, CV Variance, Bayesian, Reproducibility' },
  { phase: 'Phase 7', name: 'Benchmarking', modules: 14, items: 'Main Results, Baseline, Ablation, SOTA, Leaderboard, Computation Budget, Repro Pack, Model Card, Data Card, Failure Cases, LaTeX Tables, Figure Gen, Supplementary, Thesis' },
  { phase: 'Phase 8', name: 'Monitoring', modules: 16, items: 'Inference Latency, Throughput, Memory, Prediction Dist, Confidence, Feature Drift, Concept Drift, DQ Gate, Alert System, A/B Testing, Shadow Mode, Rollback, Retraining, Audit Log, Dashboard, Health Check' }
]

const statisticalTests = [
  { comparison: 'Ours vs LR', pValue: 0.001, cohenD: 1.25, sig: '***' },
  { comparison: 'Ours vs SVM', pValue: 0.003, cohenD: 0.98, sig: '**' },
  { comparison: 'Ours vs RF', pValue: 0.008, cohenD: 0.82, sig: '**' },
  { comparison: 'Ours vs EEGNet', pValue: 0.042, cohenD: 0.45, sig: '*' }
]

const reliabilityMetrics = [
  { dataset: 'SAM40', icc: 0.92, kappa: 0.88, testRetest: 0.91, alpha: 0.94 },
  { dataset: 'DASPS', icc: 0.88, kappa: 0.82, testRetest: 0.85, alpha: 0.90 },
  { dataset: 'MODA', icc: 0.95, kappa: 0.91, testRetest: 0.93, alpha: 0.96 },
  { dataset: 'CHB-MIT', icc: 0.90, kappa: 0.85, testRetest: 0.88, alpha: 0.92 },
  { dataset: 'BCI-IV', icc: 0.85, kappa: 0.78, testRetest: 0.82, alpha: 0.88 }
]

const techniques = [
  { category: 'Deep Learning Models', items: ['EEGNet (2,548 params)', '2D-CNN (148K params)', 'Transformer (502K params)', 'CNN-LSTM-Attention', 'Bi-LSTM (128 hidden)', 'Multi-Head Attention (8 heads)'] },
  { category: 'Classical ML', items: ['SVM (RBF kernel)', 'Random Forest', 'XGBoost', 'LightGBM', 'AdaBoost', 'Gradient Boosting', 'Logistic Regression', 'Stacking Ensemble'] },
  { category: 'Signal Processing', items: ['Butterworth Bandpass (0.5-45Hz, order=4)', 'IIR Notch Filter (50/60Hz, Q=30)', 'ICA Artifact Removal (20 components)', 'Common Average Reference (CAR)', 'CWT (Morlet wavelet)', 'STFT (Hann window, 256 FFT)'] },
  { category: 'Feature Engineering', items: ['127 Total Features (5 categories)', 'mRMR Selection', 'LASSO (λ=0.01)', 'Random Forest Importance', 'Recursive Feature Elimination', 'Stability Score (0.79-0.88)'] },
  { category: 'RAG Pipeline', items: ['FAISS (IVF-PQ) Index', 'Sentence-BERT (all-MiniLM-L6-v2)', '512-token passages (64 overlap)', 'Top-5 retrieval, <50ms latency', 'GPT-4 / Claude generation', '89.8% expert agreement'] },
  { category: 'Validation', items: ['LOSO Cross-Validation', 'Subject-wise GroupKFold', 'Bootstrap CI (1000 iter)', "McNemar's Test", "DeLong's AUC Test", 'Wilcoxon Signed-Rank', 'Bonferroni Correction'] },
  { category: 'Frameworks & Tools', items: ['PyTorch', 'MNE-Python', 'scikit-learn', 'Optuna (Bayesian HPO)', 'NumPy/SciPy/Pandas', 'FastAPI/Flask', 'React.js/Recharts', 'SHAP/LIME', 'Docker/Kubernetes'] }
]

const raiScores = [
  { pillar: 'Fairness', score: 0.92 },
  { pillar: 'Privacy', score: 0.95 },
  { pillar: 'Safety', score: 0.95 },
  { pillar: 'Transparency', score: 0.88 },
  { pillar: 'Robustness', score: 0.85 }
]
const raiRadar = raiScores.map(r => ({ subject: r.pillar, A: r.score * 100, fullMark: 100 }))

// ── Reusable sub-components ──

function FlowStep({ steps }) {
  return (
    <div style={{ display: 'flex', alignItems: 'stretch', gap: 0, overflowX: 'auto', padding: '8px 0' }}>
      {steps.map((s, i) => (
        <React.Fragment key={i}>
          <div style={{
            background: s.color || 'rgba(30,136,229,0.12)',
            border: `1px solid ${s.borderColor || '#1e88e5'}`,
            borderRadius: 10, padding: '14px 18px', minWidth: 150, flex: '1 1 0',
            display: 'flex', flexDirection: 'column', gap: 6
          }}>
            <div style={{ fontWeight: 700, fontSize: 13, color: s.borderColor || '#1e88e5' }}>{s.title}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{s.desc}</div>
            {s.detail && <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{s.detail}</div>}
          </div>
          {i < steps.length - 1 && (
            <div style={{ display: 'flex', alignItems: 'center', padding: '0 4px', color: '#475569', fontSize: 20, fontWeight: 700 }}>&#8594;</div>
          )}
        </React.Fragment>
      ))}
    </div>
  )
}

function InfoCard({ title, value, subtitle, color }) {
  return (
    <div style={{
      background: `linear-gradient(135deg, ${color}15, ${color}08)`,
      border: `1px solid ${color}30`,
      borderRadius: 12, padding: '16px 20px', textAlign: 'center', flex: '1 1 0', minWidth: 120
    }}>
      <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 24, fontWeight: 800, color }}>{value}</div>
      {subtitle && <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{subtitle}</div>}
    </div>
  )
}

function SectionTitle({ title, subtitle }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <h2 style={{ fontSize: 22, fontWeight: 800, color: '#e2e8f0', margin: 0 }}>{title}</h2>
      {subtitle && <p style={{ fontSize: 13, color: '#94a3b8', margin: '4px 0 0' }}>{subtitle}</p>}
    </div>
  )
}

function ArchBox({ title, items, color }) {
  return (
    <div style={{
      background: `${color}10`, border: `1px solid ${color}40`,
      borderRadius: 12, padding: 16, flex: '1 1 0', minWidth: 160
    }}>
      <div style={{ fontWeight: 700, fontSize: 14, color, marginBottom: 8 }}>{title}</div>
      {items.map((item, i) => (
        <div key={i} style={{ fontSize: 11, color: '#cbd5e1', padding: '3px 0', borderBottom: i < items.length - 1 ? '1px solid #1e293b' : 'none' }}>
          {item}
        </div>
      ))}
    </div>
  )
}

function DataTable({ headers, rows, highlightLast }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
        <thead>
          <tr>{headers.map((h, i) => (
            <th key={i} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #334155', color: '#94a3b8', fontWeight: 600 }}>{h}</th>
          ))}</tr>
        </thead>
        <tbody>{rows.map((row, ri) => (
          <tr key={ri} style={{ background: highlightLast && ri === rows.length - 1 ? '#1e88e515' : ri % 2 === 0 ? '#0f172a' : 'transparent' }}>
            {row.map((cell, ci) => (
              <td key={ci} style={{
                padding: '6px 10px', borderBottom: '1px solid #1e293b', color: '#e2e8f0',
                fontWeight: highlightLast && ri === rows.length - 1 ? 700 : 400
              }}>{cell}</td>
            ))}
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}

// ── Main Component ──

export default function InfographicsDashboard() {
  const [activeSection, setActiveSection] = useState('system')
  const [validationPhaseView, setValidationPhaseView] = useState('overview')
  const [stressRagView, setStressRagView] = useState('overview')

  const renderSystemArch = () => (
    <div>
      <SectionTitle title="System Architecture" subtitle="Agentic AI Disease Finder — C4 Model Architecture (IEEE TBME 2024)" />
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <InfoCard title="Datasets Validated" value="10" subtitle="6,200+ subjects" color="#1e88e5" />
        <InfoCard title="Total Samples" value="875,428" subtitle="Across all datasets" color="#7c4dff" />
        <InfoCard title="Features Documented" value="127" subtitle="5 categories" color="#4caf50" />
        <InfoCard title="Analysis Modules" value="111" subtitle="8-phase plan" color="#ff9800" />
        <InfoCard title="Mean Accuracy" value="89.2%" subtitle="LOSO validated" color="#e91e63" />
        <InfoCard title="Mean AUC" value="0.948" subtitle="All datasets" color="#00bcd4" />
      </div>
      <div className="chart-card" style={{ padding: 20 }}>
        <div className="chart-title">System Layers (C4 Container Level)</div>
        <FlowStep steps={[
          { title: 'Presentation Layer', desc: 'Web Dashboard (React), Mobile (Flutter), CLI (Python), REST API (FastAPI), WebSocket', borderColor: '#1e88e5' },
          { title: 'Application Layer', desc: 'Auth Service, Job Scheduler, Preprocessing Orchestrator, Model Orchestrator, Result Aggregator', borderColor: '#7c4dff' },
          { title: 'Domain Layer (ML/AI)', desc: 'Signal Processor (MNE), Feature Extractor, CNN/Transformer (PyTorch), Riemannian, Ensemble Voter, Explainer (SHAP)', borderColor: '#4caf50' },
          { title: 'Infrastructure Layer', desc: 'PostgreSQL, Redis Cache, MLflow Registry, S3 Storage, Kubernetes, Prometheus/Grafana, Kafka Queue', borderColor: '#ff9800' }
        ]} />
      </div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 16 }}>
        <ArchBox title="API Gateway" color="#1e88e5" items={['FastAPI (3 replicas)', 'REST + WebSocket', 'JWT/OAuth2 Auth']} />
        <ArchBox title="ML Engine" color="#4caf50" items={['PyTorch (4 GPU)', 'EEGNet + 2D-CNN + Transformer', 'Ensemble weighted average']} />
        <ArchBox title="RAG System" color="#7c4dff" items={['FAISS (IVF-PQ)', 'Sentence-BERT embeddings', '500K PubMed abstracts']} />
        <ArchBox title="Monitoring" color="#ff9800" items={['Prometheus metrics', 'Grafana dashboards', 'Drift detection (PSI/KS)']} />
      </div>
    </div>
  )

  const renderDatasets = () => (
    <div>
      <SectionTitle title="Dataset Passport" subtitle="10 EEG Datasets — 6,200+ Subjects — Complete Specifications (IEEE Table II)" />
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <InfoCard title="Total Datasets" value="10" subtitle="5 neurological conditions" color="#1e88e5" />
        <InfoCard title="Total Subjects" value="6,244+" color="#7c4dff" />
        <InfoCard title="Total Samples" value="875,428" color="#4caf50" />
        <InfoCard title="Channel Range" value="2-64" subtitle="EEG channels" color="#ff9800" />
        <InfoCard title="Hz Range" value="100-512" subtitle="Sampling rates" color="#e91e63" />
      </div>
      <div className="chart-card" style={{ padding: 20 }}>
        <div className="chart-title">Complete Dataset Specifications (Table II from IEEE paper)</div>
        <DataTable
          headers={['Dataset', 'N', 'M/F', 'Ch', 'Hz', 'Device', 'Duration', 'Samples', 'Label Source', 'Task']}
          rows={datasetSpecs.map(d => [d.dataset, d.n, d.mf, d.ch, d.hz, d.device, d.duration, d.samples.toLocaleString(), d.label, d.task])}
        />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Per-Dataset Accuracy (%)</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={datasetPerformance} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" stroke="#94a3b8" domain={[75, 100]} />
              <YAxis dataKey="dataset" type="category" stroke="#94a3b8" width={80} fontSize={11} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="acc" name="Accuracy %" radius={[0, 4, 4, 0]}>
                {datasetPerformance.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Per-Dataset Performance Metrics (Table XXII)</div>
          <DataTable
            headers={['Dataset', 'Acc', 'Prec', 'Rec', 'F1', 'AUC']}
            rows={[...datasetPerformance.map(d => [d.dataset, d.acc, d.prec, d.rec, d.f1, d.auc]),
              ['Mean', '89.2', '84.9', '87.3', '0.876', '0.948']]}
            highlightLast
          />
        </div>
      </div>
      <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
        <div className="chart-title">Signal Statistics per Dataset (Table III)</div>
        <DataTable
          headers={['Dataset', 'Mean (μV)', 'Std (μV)', 'Skew', 'Kurtosis']}
          rows={signalStats.map(d => [d.dataset, d.mean, d.std, d.skew, d.kurt])}
        />
      </div>
    </div>
  )

  const renderDataArch = () => (
    <div>
      <SectionTitle title="Data Architecture" subtitle="127 Features — 5 Categories — Complete Extraction Pipeline (Section VI)" />
      <div className="chart-card" style={{ padding: 20 }}>
        <div className="chart-title">Complete Data Processing Pipeline</div>
        <FlowStep steps={[
          { title: 'Raw EEG Input', desc: '10 datasets, variable SR/Ch', detail: '875K total samples', borderColor: '#1e88e5' },
          { title: 'Bandpass Filter', desc: 'Butterworth 4th order, 0.5-45Hz, zero-phase', detail: '-3dB at cutoff', borderColor: '#7c4dff' },
          { title: 'Notch Filter', desc: 'IIR 50/60Hz, Q=30, bandwidth=2Hz', detail: 'Remove powerline + harmonics', borderColor: '#00bcd4' },
          { title: 'Artifact Removal', desc: 'ICA (Extended Infomax, 20 components)', detail: 'Kurtosis>5 eye, hi-freq>20Hz muscle', borderColor: '#4caf50' },
          { title: 'Re-reference', desc: 'Common Average Reference (CAR)', detail: 'x_CAR = x_i - (1/N)Σx_j', borderColor: '#ff9800' },
          { title: 'Segmentation', desc: 'Window: 4s (1024 samples @256Hz)', detail: '50% overlap (512 samples)', borderColor: '#e91e63' },
          { title: 'Normalization', desc: 'Z-score per-channel from training set', detail: 'Robust scaling for CHB/TUH', borderColor: '#1e88e5' }
        ]} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Feature Distribution (127 Total — Table XI)</div>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={featureBreakdown} dataKey="count" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={100}
                label={({ name, count, pct }) => `${name} (${count}, ${pct}%)`}>
                {featureBreakdown.map((entry, i) => <Cell key={i} fill={entry.color} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Top 10 Features by Importance (Table XVI)</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={topFeatures} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" stroke="#94a3b8" domain={[0, 0.1]} />
              <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={140} fontSize={10} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="importance" name="Importance" radius={[0, 4, 4, 0]}>
                {topFeatures.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
        <div className="chart-title">1D → 2D Conversion Methods (Section VII)</div>
        <FlowStep steps={[
          { title: 'CWT Scalogram', desc: 'Complex Morlet (cmor1.5-1.0), fc=1.0Hz, fb=1.5Hz', detail: '64 scales, output: 64×1024 → 64×64', borderColor: '#1e88e5' },
          { title: 'STFT Spectrogram', desc: 'Hann window, 256 samples, 50% overlap', detail: '256 FFT bins → 129×8 → 64×64', borderColor: '#ff9800' },
          { title: 'Channel-to-Image', desc: '32ch→6×6 grid, 64ch→8×8 grid', detail: 'Spherical spline interpolation', borderColor: '#4caf50' },
          { title: 'Multi-View Input', desc: 'CWT(64×64×C) + Topo(8×8×5)', detail: 'Concatenated to 64×64×(C+5)', borderColor: '#e91e63' }
        ]} />
      </div>
    </div>
  )

  const renderModelArch = () => (
    <div>
      <SectionTitle title="Model Architecture" subtitle="3 Deep Learning Models + Ensemble — Bayesian HPO with Optuna (Section X-XI)" />
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <InfoCard title="EEGNet" value="2,548" subtitle="Parameters (F1=8, D=2)" color="#1e88e5" />
        <InfoCard title="2D CNN" value="148K" subtitle="Parameters (3 Conv blocks)" color="#7c4dff" />
        <InfoCard title="Transformer" value="502K" subtitle="Parameters (4 layers, 8 heads)" color="#4caf50" />
        <InfoCard title="Ensemble" value="89.3%" subtitle="Weighted avg [0.4, 0.35, 0.25]" color="#ff9800" />
      </div>
      <div className="chart-card" style={{ padding: 20 }}>
        <div className="chart-title">Model Training Ladder (Table VIII)</div>
        <DataTable
          headers={['Level', 'Model', 'Parameters', 'Training Time', 'Purpose']}
          rows={modelTrainingLadder.map(m => [m.level, m.model, m.params, m.time, m.purpose])}
        />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">EEGNet Layer Specification (Table XVIII)</div>
          <DataTable
            headers={['#', 'Layer', 'Output', 'Params']}
            rows={[
              [1, 'Conv2D (1, 64, 1)', '(F1, C, T)', '64'],
              [2, 'BatchNorm', '(F1, C, T)', '128'],
              [3, 'DepthwiseConv2D', '(F1·D, 1, T)', '256'],
              [4, 'BatchNorm + ELU', '(F1·D, 1, T)', '64'],
              [5, 'AvgPool2D (1, 4)', '(F1·D, 1, T/4)', '0'],
              [6, 'Dropout (0.5)', '(F1·D, 1, T/4)', '0'],
              [7, 'SeparableConv2D', '(F2, 1, T/4)', '512'],
              [8, 'BatchNorm + ELU', '(F2, 1, T/4)', '32'],
              [9, 'AvgPool2D (1, 8)', '(F2, 1, T/32)', '0'],
              [10, 'Flatten + Dense', '(Classes)', '1,024']
            ]}
          />
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">EEG Transformer Specification (Table XX)</div>
          <DataTable
            headers={['Component', 'Configuration']}
            rows={[
              ['Embedding dim', '128'],
              ['Num heads', '8'],
              ['Num layers', '4'],
              ['FFN dim', '512'],
              ['Dropout', '0.1'],
              ['Position encoding', 'Learnable'],
              ['Patch size', '32 samples'],
              ['Num patches', '32'],
              ['Total params', '502,786']
            ]}
          />
        </div>
      </div>
      <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
        <div className="chart-title">Hyperparameter Search Space (Table XXI — Bayesian Optuna, 100 trials/model)</div>
        <DataTable
          headers={['Parameter', 'Range', 'Best Value', 'Selection Method']}
          rows={[
            ['Learning Rate', '[1e-4, 1e-2] log', '1e-3', 'Grid search + cosine decay'],
            ['Batch Size', '[16, 32, 64]', '32', 'Memory-constrained'],
            ['Dropout', '[0.3, 0.5, 0.7]', '0.5', 'Regularization strength'],
            ['Weight Decay', '[1e-5, 1e-3] log', '1e-4', 'L2 penalty'],
            ['Epochs', '100 (patience=10)', '~45', 'Early stopping'],
            ['Optimizer', 'AdamW', 'β1=0.9, β2=0.999', 'Standard for transformers'],
            ['CV Folds', '5', 'Subject-wise GroupKFold', 'Prevent data leakage']
          ]}
        />
      </div>
    </div>
  )

  const renderGenAI = () => (
    <div>
      <SectionTitle title="GenAI-RAG-EEG Architecture" subtitle="CNN-LSTM-Attention + RAG Explainability — 197,635 params (genai_rag_eeg_v4.pdf)" />
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <InfoCard title="DEAP Accuracy" value="94.7%" subtitle="AUC: 0.967" color="#1e88e5" />
        <InfoCard title="SAM-40 Accuracy" value="93.2%" subtitle="AUC: 0.958" color="#7c4dff" />
        <InfoCard title="WESAD Accuracy" value="100%" subtitle="AUC: 1.000" color="#4caf50" />
        <InfoCard title="Average" value="95.97%" subtitle="κ: 0.919" color="#ff9800" />
        <InfoCard title="Expert Agreement" value="89.8%" subtitle="Rating: 4.2/5" color="#e91e63" />
      </div>
      <div className="chart-card" style={{ padding: 20 }}>
        <div className="chart-title">GenAI-RAG-EEG Architecture Pipeline</div>
        <FlowStep steps={[
          { title: 'EEG Encoder', desc: '3 Conv blocks (32→64→64 filters) + BatchNorm + MaxPool', detail: 'Spatial-temporal features', borderColor: '#1e88e5' },
          { title: 'Bi-LSTM', desc: '2 layers, 128 hidden units (64 per direction)', detail: 'Forward + backward temporal', borderColor: '#7c4dff' },
          { title: 'Self-Attention', desc: '8 heads, element-wise relevance scores', detail: 'α_t = exp(e_t)/Σexp(e_k)', borderColor: '#4caf50' },
          { title: 'Context Encoder', desc: 'SBERT (all-MiniLM-L6-v2) → 384d → 128d projection', detail: 'Task metadata encoding', borderColor: '#ff9800' },
          { title: 'Fusion + MLP', desc: 'Concat(128+128)=256 → FC(64→32→2) + softmax', detail: 'ReLU + 30% dropout', borderColor: '#e91e63' },
          { title: 'RAG Explainer', desc: 'FAISS retrieval → Top-5 passages → LLM synthesis', detail: '89% factual, 93.2% citation acc', borderColor: '#00bcd4' }
        ]} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Cross-Dataset Performance (LOSO — Table IV)</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={ragPerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="dataset" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[85, 101]} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Legend />
              <Bar dataKey="acc" name="Accuracy" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              <Bar dataKey="f1" name="F1 %" fill="#4caf50" radius={[4, 4, 0, 0]} />
              <Bar dataKey="auc" name="AUC %" fill="#ff9800" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Band Power Effect Sizes (Cohen's d — Table III)</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={bandPowerEffects}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="band" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[-1, 1]} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Legend />
              <Bar dataKey="deap" name="DEAP" fill="#1e88e5" />
              <Bar dataKey="sam40" name="SAM-40" fill="#4caf50" />
              <Bar dataKey="wesad" name="WESAD" fill="#ff9800" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Ablation Study — Component Contribution (Table VI)</div>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={ablationStudy}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="config" stroke="#94a3b8" fontSize={10} angle={-15} textAnchor="end" height={60} />
              <YAxis stroke="#94a3b8" domain={[85, 95]} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="acc" name="Accuracy %" radius={[4, 4, 0, 0]}>
                {ablationStudy.map((entry, i) => <Cell key={i} fill={entry.delta === 0 ? '#4caf50' : entry.delta < -2 ? '#f44336' : '#ff9800'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">RAG Explanation Expert Evaluation (Table XI)</div>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={ragExplanationEval} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" stroke="#94a3b8" domain={[0, 100]} />
              <YAxis dataKey="criterion" type="category" stroke="#94a3b8" width={130} fontSize={10} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                formatter={(val, name) => [name === 'Rating (1-5)' ? `${val}/5.0` : `${val}%`, name]} />
              <Legend />
              <Bar dataKey="agreement" name="Agreement %" fill="#1e88e5" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )

  const renderStressRag = () => {
    const stressTabs = [
      { id: 'overview', label: 'Overview' },
      { id: 'architecture', label: 'Architecture' },
      { id: 'hyperparams', label: 'Hyperparameters' },
      { id: 'classification', label: 'Dashboard' },
      { id: 'signal', label: 'Signal & Band Power' },
      { id: 'features', label: 'Feature Analysis' },
      { id: 'transfer', label: 'Cross-Dataset Transfer' },
      { id: 'ablation', label: 'Ablation & Components' },
      { id: 'benchmark', label: 'Benchmark' },
      { id: 'stats', label: 'Statistical Analysis' }
    ]
    return (
    <div>
      <SectionTitle title="EEG Stress-RAG Papers — All Graphs" subtitle="GenAI-RAG-EEG v1 (256K params, DEAP/SAM-40/WESAD) + v2 (159K params, DEAP/SAM-40/EEGMAT) — IEEE Sensors Journal" />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 16, padding: '8px 12px', background: '#0f172a', borderRadius: 10, border: '1px solid #1e293b' }}>
        {stressTabs.map(tab => (
          <button key={tab.id} onClick={() => setStressRagView(tab.id)} style={{
            padding: '5px 10px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 10, fontWeight: stressRagView === tab.id ? 700 : 500,
            background: stressRagView === tab.id ? '#1e88e5' : '#1e293b',
            color: stressRagView === tab.id ? '#fff' : '#94a3b8', transition: 'all 0.2s'
          }}>{tab.label}</button>
        ))}
      </div>

      {/* Overview */}
      {stressRagView === 'overview' && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
            <InfoCard title="v2: DEAP" value="94.7%" subtitle="Arousal proxy" color="#1e88e5" />
            <InfoCard title="v2: SAM-40" value="93.2%" subtitle="Cognitive stress" color="#7c4dff" />
            <InfoCard title="v2: EEGMAT" value="91.8%" subtitle="Workload" color="#4caf50" />
            <InfoCard title="v1: WESAD" value="100%" subtitle="Acute stress" color="#ff9800" />
            <InfoCard title="v2 Params" value="159,372" subtitle="Compact model" color="#e91e63" />
            <InfoCard title="v1 Params" value="265,219" subtitle="Full model" color="#00bcd4" />
          </div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v2 Paper: Stress Dataset Specifications (Tables III-V)</div>
            <DataTable headers={['Dataset', 'Role', 'Subjects', 'Channels', 'Hz', 'Trials', 'Label', 'Task']}
              rows={stressDatasetSpecs.map(d => [d.dataset, d.role, d.subjects, d.channels, d.hz, d.trials, d.label, d.task])} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Class Balance Analysis (Table X)</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={stressClassBalance}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="dataset" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Legend />
                  <Bar dataKey="high" name="High Stress" fill="#f44336" radius={[4,4,0,0]} />
                  <Bar dataKey="low" name="Low Stress" fill="#4caf50" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Epoch Rejection Statistics (Table XV)</div>
              <DataTable headers={['Dataset', 'Total', 'Rejected', 'Retained', 'Rate']}
                rows={epochRejection.map(d => [d.dataset, d.total.toLocaleString(), d.rejected, d.retained.toLocaleString(), d.rate])} />
            </div>
          </div>
        </div>
      )}

      {/* Architecture */}
      {stressRagView === 'architecture' && (
        <div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v2 Model Parameters — 159,372 Total (Table XIX, Fig 5)</div>
            <DataTable headers={['Component', 'Layer', 'Parameters']}
              rows={v2ModelParams.map(d => [d.component, d.layer, d.params.toLocaleString()])} highlightLast />
            <div style={{ textAlign: 'center', marginTop: 8, fontSize: 14, fontWeight: 700, color: '#1e88e5' }}>Total Trainable: 159,372</div>
          </div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v1 Model Parameters — 265,219 Total (Table XIX, Fig 14)</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={v1ModelParams}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="component" stroke="#94a3b8" fontSize={10} angle={-15} textAnchor="end" height={60} />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} formatter={(v) => [v.toLocaleString(), 'Parameters']} />
                <Bar dataKey="params" name="Parameters" radius={[4,4,0,0]}>
                  {v1ModelParams.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">Computational Complexity Comparison (v1 Table X)</div>
            <DataTable headers={['Model', 'Params', 'Memory', 'Inference', 'GPU']}
              rows={computeComplexity.map(d => [d.model, d.params, d.memory, d.inference, d.gpu])} highlightLast />
          </div>
        </div>
      )}

      {/* Hyperparameters */}
      {stressRagView === 'hyperparams' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Learning Rate Sensitivity (Fig 8) — Optimal: 10⁻⁴</div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={hpLearningRate}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="lr" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[65, 100]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Legend />
                  <Line type="monotone" dataKey="deap" name="DEAP" stroke="#1e88e5" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="sam40" name="SAM-40" stroke="#4caf50" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="eegmat" name="EEGMAT" stroke="#ff9800" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Batch Size Impact (Fig 9) — Optimal: 64</div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={hpBatchSize}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="size" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[84, 96]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" name="Accuracy %" stroke="#1e88e5" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="f1" name="F1 Score ×100" stroke="#4caf50" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">Dropout Rate Sensitivity (Fig 10) — Optimal: 0.3</div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={hpDropout}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="rate" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" domain={[88, 96]} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                <Legend />
                <Line type="monotone" dataKey="train" name="Training" stroke="#1e88e5" strokeWidth={2} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="val" name="Validation" stroke="#4caf50" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Classification */}
      {stressRagView === 'classification' && (
        <div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v2 Per-Dataset Performance with 95% CI (Tables XXIII-XXV, XXVI)</div>
            <DataTable headers={['Dataset', 'Acc%', '95% CI', 'Prec', 'Rec', 'F1', 'κ', 'AUC-ROC', 'AUC-PR', 'MCC']}
              rows={v2PerDatasetMetrics.map(d => [d.dataset, d.acc, d.ci, d.prec, d.rec, d.f1, d.kappa, d.auc, d.pr, d.mcc])} />
          </div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v2 Confusion Matrices (Fig 13) — Aggregated across LOSO folds</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
              {v2ConfusionData.map((cm, idx) => (
                <div key={idx} style={{ background: '#0f172a', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: COLORS[idx], marginBottom: 8, textAlign: 'center' }}>{cm.dataset}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, textAlign: 'center' }}>
                    <div style={{ background: '#4caf5020', padding: 8, borderRadius: 4 }}><div style={{ fontSize: 18, fontWeight: 700, color: '#4caf50' }}>{cm.tn}</div><div style={{ fontSize: 9, color: '#94a3b8' }}>TN</div></div>
                    <div style={{ background: '#f4433620', padding: 8, borderRadius: 4 }}><div style={{ fontSize: 18, fontWeight: 700, color: '#f44336' }}>{cm.fp}</div><div style={{ fontSize: 9, color: '#94a3b8' }}>FP</div></div>
                    <div style={{ background: '#ff980020', padding: 8, borderRadius: 4 }}><div style={{ fontSize: 18, fontWeight: 700, color: '#ff9800' }}>{cm.fn}</div><div style={{ fontSize: 9, color: '#94a3b8' }}>FN</div></div>
                    <div style={{ background: '#1e88e520', padding: 8, borderRadius: 4 }}><div style={{ fontSize: 18, fontWeight: 700, color: '#1e88e5' }}>{cm.tp}</div><div style={{ fontSize: 9, color: '#94a3b8' }}>TP</div></div>
                  </div>
                  <div style={{ textAlign: 'center', marginTop: 4, fontSize: 10, color: '#94a3b8' }}>Acc: {((cm.tn+cm.tp)/(cm.tn+cm.tp+cm.fn+cm.fp)*100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v1 Confusion Matrices (Fig 21) — DEAP/SAM-40/WESAD</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
              {v1ConfusionData.map((cm, idx) => (
                <div key={idx} style={{ background: '#0f172a', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: COLORS[idx], marginBottom: 8, textAlign: 'center' }}>{cm.dataset}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, textAlign: 'center' }}>
                    <div style={{ background: '#4caf5020', padding: 8, borderRadius: 4 }}><div style={{ fontSize: 18, fontWeight: 700, color: '#4caf50' }}>{cm.tn}</div><div style={{ fontSize: 9, color: '#94a3b8' }}>TN</div></div>
                    <div style={{ background: '#f4433620', padding: 8, borderRadius: 4 }}><div style={{ fontSize: 18, fontWeight: 700, color: '#f44336' }}>{cm.fp}</div><div style={{ fontSize: 9, color: '#94a3b8' }}>FP</div></div>
                    <div style={{ background: '#ff980020', padding: 8, borderRadius: 4 }}><div style={{ fontSize: 18, fontWeight: 700, color: '#ff9800' }}>{cm.fn}</div><div style={{ fontSize: 9, color: '#94a3b8' }}>FN</div></div>
                    <div style={{ background: '#1e88e520', padding: 8, borderRadius: 4 }}><div style={{ fontSize: 18, fontWeight: 700, color: '#1e88e5' }}>{cm.tp}</div><div style={{ fontSize: 9, color: '#94a3b8' }}>TP</div></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Subject-Wise Performance Boxplot Data (Fig 12)</div>
              <DataTable headers={['Dataset', 'Median', 'IQR', 'Q1', 'Q3', 'Min', 'Max']}
                rows={subjectWiseBoxplot.map(d => [d.dataset, `${d.median}%`, d.iqr, d.q1, d.q3, d.min, d.max])} />
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">v1 Classification Results (Table VI)</div>
              <DataTable headers={['Dataset', 'Acc%', 'Prec', 'Rec', 'F1', 'AUC', 'MCC']}
                rows={v1ClassResults.map(d => [d.dataset, d.acc, d.prec, d.rec, d.f1, d.auc, d.mcc])} />
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
            <div className="chart-title">v1 Clinical Validation — Multi-Task (Table XV)</div>
            <DataTable headers={['Dataset', 'Task', 'Acc%', 'F1', 'κ', 'AUC']}
              rows={v1ClinicalVal.map(d => [d.dataset, d.task, d.acc, d.f1, d.kappa, d.auc])} />
          </div>
        </div>
      )}

      {/* Signal & Band Power */}
      {stressRagView === 'signal' && (
        <div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v2 Band Power — DEAP Dataset (Table XXVIII, μV²/Hz)</div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={v2BandPowerDEAP}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="band" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                <Legend />
                <Bar dataKey="low" name="Low Stress" fill="#4caf50" radius={[4,4,0,0]} />
                <Bar dataKey="high" name="High Stress" fill="#f44336" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">v2 Band Power — SAM-40 (Table XXIX)</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={v2BandPowerSAM40}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="band" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Legend />
                  <Bar dataKey="low" name="Low" fill="#4caf50" radius={[4,4,0,0]} />
                  <Bar dataKey="high" name="High" fill="#f44336" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">v2 Band Power — EEGMAT (Table XXX)</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={v2BandPowerEEGMAT}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="band" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Legend />
                  <Bar dataKey="low" name="Low" fill="#4caf50" radius={[4,4,0,0]} />
                  <Bar dataKey="high" name="High" fill="#f44336" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Alpha Suppression Analysis (Table XXXI)</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={alphaSuppression}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="dataset" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[0, 20]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Legend />
                  <Bar dataKey="baseline" name="Baseline α" fill="#1e88e5" radius={[4,4,0,0]} />
                  <Bar dataKey="stress" name="Stress α" fill="#f44336" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', fontSize: 10, color: '#94a3b8', marginTop: 4 }}>29-31% suppression (all p&lt;.001) — validates stress biomarker</div>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Theta/Beta Ratio (Table XXXII)</div>
              <DataTable headers={['Dataset', 'Low Stress', 'High Stress', 'Δ', "Cohen's d", 'p']}
                rows={thetaBetaRatio.map(d => [d.dataset, d.low, d.high, d.delta, d.d, d.p])} />
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">v1 Band Power: Stress vs Baseline (Table XIV, Fig 37)</div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={v1BandPower}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="band" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                <Legend />
                <Bar dataKey="baseline" name="Baseline" fill="#1e88e5" radius={[4,4,0,0]} />
                <Bar dataKey="stress" name="Stress" fill="#f44336" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: 10, color: '#94a3b8', marginTop: 4 }}>Delta/Theta/Alpha/Beta: p&lt;0.001 | Gamma: ns (p=0.142)</div>
          </div>
        </div>
      )}

      {/* Feature Analysis */}
      {stressRagView === 'features' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">v2 Feature Importance — DEAP (Table XLII)</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImpDEAP} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#94a3b8" domain={[0, 0.16]} />
                  <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={120} fontSize={10} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="importance" name="Importance" radius={[0,4,4,0]}>
                    {featureImpDEAP.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">v2 Feature Importance — SAM-40 (Table XLIII)</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImpSAM40} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#94a3b8" domain={[0, 0.16]} />
                  <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={120} fontSize={10} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="importance" name="Importance" radius={[0,4,4,0]}>
                    {featureImpSAM40.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">Channel × Band Importance Matrix (Table XLV — Mean Across Datasets)</div>
            <DataTable headers={['Region', 'δ (1-4)', 'θ (4-8)', 'α (8-13)', 'β (13-30)', 'γ (30-45)', 'Total']}
              rows={[...channelBandMatrix.map(d => [d.region, d.delta, d.theta, d.alpha, d.beta, d.gamma, d.total]),
                ['Band Total', '0.077', '0.242', '0.346', '0.306', '0.145', '1.000']]} highlightLast />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">v1 Feature Importance (Fig 11)</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={v1FeatureImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#94a3b8" domain={[0, 0.20]} />
                  <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={110} fontSize={10} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="importance" name="Importance" radius={[0,4,4,0]}>
                    {v1FeatureImportance.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">v1 Feature Selection Methods (Fig 24)</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureSelectionMethods}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="method" stroke="#94a3b8" fontSize={9} angle={-15} textAnchor="end" height={50} />
                  <YAxis stroke="#94a3b8" domain={[89, 96]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                    formatter={(v, n, p) => [`${v}% — ${p.payload.features}`, 'Accuracy']} />
                  <Bar dataKey="acc" name="Accuracy %" radius={[4,4,0,0]}>
                    {featureSelectionMethods.map((e, i) => <Cell key={i} fill={e.acc >= 94 ? '#4caf50' : '#1e88e5'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
            <div className="chart-title">Confound Analysis — Artifact Rate (Tables XLVI-XLVIII)</div>
            <DataTable headers={['Dataset', 'Artifact Low', 'Artifact High', 'Diff', 'p-value', 'Conclusion']}
              rows={confoundAnalysis.map(d => [d.dataset, d.artifactLow, d.artifactHigh, d.diff, d.p, 'Not significant'])} />
            <div style={{ fontSize: 10, color: '#4caf50', marginTop: 8, textAlign: 'center' }}>All p&gt;0.05 — Classification driven by neural activity, not artifact contamination</div>
          </div>
        </div>
      )}

      {/* Cross-Dataset Transfer */}
      {stressRagView === 'transfer' && (
        <div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v2 Cross-Dataset Transfer Evaluation (Table XXVII)</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={v2CrossTransfer}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="note" stroke="#94a3b8" fontSize={10} />
                <YAxis stroke="#94a3b8" domain={[50, 100]} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                  formatter={(v, n, p) => [`${v}% (${p.payload.train}→${p.payload.test})`, n]} />
                <Bar dataKey="acc" name="Accuracy %" radius={[4,4,0,0]}>
                  {v2CrossTransfer.map((e, i) => <Cell key={i} fill={e.acc > 75 ? '#4caf50' : e.acc > 68 ? '#ff9800' : '#f44336'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <DataTable headers={['Train', 'Test', 'Acc%', 'F1', 'Δ', 'Note']}
              rows={v2CrossTransfer.map(d => [d.train, d.test, d.acc, d.f1, `${d.delta}%`, d.note])} />
          </div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">v1 Cross-Dataset Transfer (Table XI) — Average drop: 20%</div>
            <DataTable headers={['Train → Test', 'Accuracy', 'Δ vs In-Domain', 'F1']}
              rows={[...v1CrossTransfer.map(d => [`${d.train} → ${d.test}`, d.acc, `${d.delta}%`, d.f1]),
                ['Average', '73.1 ± 4.6', '-20.0', '0.749']]} highlightLast />
          </div>
        </div>
      )}

      {/* Ablation & Components */}
      {stressRagView === 'ablation' && (
        <div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">v2 Ablation Study (Table LI) — Component Contribution</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={v2AblationStudy}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="config" stroke="#94a3b8" fontSize={10} angle={-15} textAnchor="end" height={60} />
                <YAxis stroke="#94a3b8" domain={[82, 96]} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                  formatter={(v, n, p) => [`${v}% (Δ: ${p.payload.delta}%, p: ${p.payload.p})`, 'Accuracy']} />
                <Bar dataKey="acc" name="Accuracy %" radius={[4,4,0,0]}>
                  {v2AblationStudy.map((e, i) => <Cell key={i} fill={e.delta === 0 ? '#4caf50' : Math.abs(e.delta) > 5 ? '#f44336' : '#ff9800'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Component Importance Ranking (Fig 21)</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={componentImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#94a3b8" domain={[0, 8]} />
                  <YAxis dataKey="component" type="category" stroke="#94a3b8" width={100} fontSize={10} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                    formatter={(v) => [`+${v}%`, 'Contribution']} />
                  <Bar dataKey="contribution" name="Accuracy Contribution %" radius={[0,4,4,0]}>
                    {componentImportance.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Cumulative Ablation (Fig 22)</div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={cumulativeAblation}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={9} angle={-15} textAnchor="end" height={50} />
                  <YAxis stroke="#94a3b8" domain={[60, 100]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Area type="monotone" dataKey="acc" name="Accuracy %" stroke="#f44336" fill="#f4433620" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', fontSize: 10, color: '#94a3b8', marginTop: 4 }}>Total impact: -28.1% — Temporal modeling (Bi-LSTM) most critical</div>
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">Same-Pipeline Baseline Comparison (Table XLIX)</div>
            <DataTable headers={['Dataset', 'Method', 'Acc%', 'F1', 'AUC']}
              rows={samePipelineBaseline.map(d => [d.data, d.method, d.acc, d.f1, d.auc])} />
          </div>
        </div>
      )}

      {/* Benchmark */}
      {stressRagView === 'benchmark' && (
        <div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">Comprehensive Benchmark on DEAP (v2 Table L)</div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={v2Benchmark}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="model" stroke="#94a3b8" fontSize={9} angle={-20} textAnchor="end" height={60} />
                <YAxis stroke="#94a3b8" domain={[78, 98]} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                <Legend />
                <Bar dataKey="acc" name="Accuracy %" radius={[4,4,0,0]}>
                  {v2Benchmark.map((e, i) => <Cell key={i} fill={e.model === 'GenAI-RAG-EEG' ? '#4caf50' : e.type === 'ML' ? '#1e88e5' : '#ff9800'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">Benchmark Details (v2 Table L)</div>
            <DataTable headers={['Model', 'Type', 'Params', 'Acc%', 'F1', 'AUC', 'MCC']}
              rows={v2Benchmark.map(d => [d.model, d.type, d.params, d.acc, d.f1, d.auc, d.mcc])} highlightLast />
          </div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">v1 Subject-Wise LOSO Performance (Table XVII)</div>
            <DataTable headers={['Subject Group', 'Mean Acc%', 'Std', 'Min', 'Max']}
              rows={[...v1SubjectLOSO.map(d => [d.group, d.acc, d.std, d.min, d.max]),
                ['Overall LOSO', '91.0', '4.4', '81.8', '97.5']]} highlightLast />
          </div>
        </div>
      )}

      {/* Statistical Analysis */}
      {stressRagView === 'stats' && (
        <div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">Statistical Robustness (Tables LII-LIV)</div>
            <DataTable headers={['Dataset', 'Mean', 'Median', 'Q1', 'Q3', 'IQR', '95% CI Low', '95% CI High', 'Std', 'CV']}
              rows={statRobustness.map(d => [d.dataset, d.mean, d.median, d.q1, d.q3, d.iqr, d.ciLow, d.ciHigh, d.std, d.cv])} />
          </div>
          <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
            <div className="chart-title">Statistical Significance — All Tests p&lt;.001 (Tables LV-LVIII)</div>
            <DataTable headers={['Dataset', 'Wilcoxon p', 't-test p', 'Mann-Whitney p', "Cohen's d", 'Effect r']}
              rows={statSignificance.map(d => [d.dataset, d.wilcoxP, d.tP, d.mwP, d.d, d.r])} />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 16 }}>
              {statSignificance.map((d, i) => (
                <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: 12, textAlign: 'center' }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: COLORS[i] }}>{d.dataset}</div>
                  <div style={{ fontSize: 24, fontWeight: 800, color: '#4caf50', marginTop: 4 }}>d = {d.d}</div>
                  <div style={{ fontSize: 10, color: '#94a3b8' }}>Large effect (d &gt; 0.8)</div>
                  <div style={{ fontSize: 10, color: '#94a3b8' }}>r = {d.r} | All p &lt; .001</div>
                </div>
              ))}
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">Key Statistical Findings</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {[
                { title: 'All Tests Significant', desc: 'Wilcoxon, Paired-t, Mann-Whitney all p<.001 across all datasets', color: '#4caf50' },
                { title: 'Large Effect Sizes', desc: "Cohen's d: 1.18-1.47, r: 0.85-0.91 — substantial practical improvement", color: '#1e88e5' },
                { title: 'Low Variability', desc: 'CV: 2.4-3.4%, IQR: 3.2-4.6 — robust across subjects', color: '#ff9800' },
                { title: 'Tight Confidence', desc: '95% CI widths: 3.0-4.0% — reliable performance estimates', color: '#7c4dff' }
              ].map((item, i) => (
                <div key={i} style={{ padding: 12, background: `${item.color}08`, border: `1px solid ${item.color}30`, borderRadius: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: item.color }}>{item.title}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{item.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )}

  const renderPipelineFlows = () => (
    <div>
      <SectionTitle title="Pipeline Flows" subtitle="End-to-end processing pipeline flowcharts from research papers" />
      {[
        { title: '1. Signal Preprocessing Pipeline (Table VII-VIII)', steps: [
          { title: 'Highpass 0.5Hz', desc: 'Butterworth order=4, zero-phase (filtfilt)', detail: '-3dB at 0.5Hz', borderColor: '#1e88e5' },
          { title: 'Lowpass 45Hz', desc: 'Butterworth order=4, zero-phase', detail: 'Stopband atten >40dB', borderColor: '#7c4dff' },
          { title: 'Notch 50/60Hz', desc: 'IIR notch Q=30, bandwidth=2Hz', detail: 'Remove powerline + harmonics', borderColor: '#00bcd4' },
          { title: 'ICA Artifact', desc: 'Extended Infomax, 20 components', detail: 'Frontal kurtosis>5→eye, Temporal hi-freq→muscle', borderColor: '#4caf50' },
          { title: 'CAR Reref', desc: 'x_CAR(t) = x_i(t) - (1/N)Σx_j(t)', detail: 'Common average reference', borderColor: '#ff9800' },
          { title: 'Segment 4s', desc: '1024 samples @256Hz, 50% overlap', detail: 'Samples/subj = (T-W)/S + 1', borderColor: '#e91e63' }
        ]},
        { title: '2. Feature Extraction Pipeline (127 Features — Section VI)', steps: [
          { title: 'Time Domain (27)', desc: 'Mean, Var, Std, Skew, Kurt, Peak-Peak, RMS, ZCR, Line Length', detail: 'Per channel', borderColor: '#1e88e5' },
          { title: 'Hjorth (3)', desc: 'Activity = var(x), Mobility = √(var(x\')/var(x)), Complexity', detail: 'Per channel', borderColor: '#00bcd4' },
          { title: 'Entropy (5)', desc: 'Shannon, Sample (m=2, r=0.2σ), Approx, Spectral, Permutation', detail: 'Signal complexity', borderColor: '#4caf50' },
          { title: 'Frequency (15)', desc: 'δ(0.5-4), θ(4-8), α(8-13), β(13-30), γ(30-45) band powers + ratios', detail: "Welch's PSD", borderColor: '#ff9800' },
          { title: 'Connectivity', desc: 'Coherence, PLV, PLI per channel pair per frequency band', detail: 'N(N-1)/2 pairs', borderColor: '#e91e63' }
        ]},
        { title: '3. Feature Selection Pipeline (Section VIII)', steps: [
          { title: 'mRMR', desc: 'Min Redundancy Maximum Relevance using mutual information', borderColor: '#1e88e5' },
          { title: 'LASSO', desc: 'L1 regularization (λ=0.01) for sparse selection', borderColor: '#7c4dff' },
          { title: 'RF Importance', desc: 'Gini importance from 500 trees', borderColor: '#4caf50' },
          { title: 'RFE', desc: 'Recursive Feature Elimination with SVM backend', borderColor: '#ff9800' },
          { title: 'Stability Test', desc: 'Consistency across 5 CV folds (Jaccard index)', detail: 'Scores: 0.79-0.88', borderColor: '#e91e63' }
        ]},
        { title: '4. Model Training Pipeline (Phases 7-8)', steps: [
          { title: 'Load Dataset', desc: 'Split train/val/test per subject', detail: '70/10/20 split', borderColor: '#1e88e5' },
          { title: 'Extract Features', desc: 'Bandpower, Hjorth, Spectral, Entropy', borderColor: '#7c4dff' },
          { title: 'CWT Scalograms', desc: 'Morlet wavelet → 64×64 time-freq images', borderColor: '#00bcd4' },
          { title: 'Train Models', desc: 'Forward pass → Loss → Backward → Update weights', detail: 'AdamW lr=1e-3, batch=32', borderColor: '#4caf50' },
          { title: 'Validate', desc: 'Compute Acc, F1, AUC per fold', detail: 'Early stop patience=10', borderColor: '#ff9800' },
          { title: 'Save Best', desc: 'Checkpoint best model, final evaluation', borderColor: '#e91e63' }
        ]},
        { title: '5. Active Learning Pipeline (Section XIII)', steps: [
          { title: '10% Labeled', desc: 'Initialize with random labeled subset', borderColor: '#1e88e5' },
          { title: 'Train Model', desc: 'Train on current labeled set', borderColor: '#7c4dff' },
          { title: 'Predict Pool', desc: 'Score unlabeled samples', borderColor: '#4caf50' },
          { title: 'Select Top-K', desc: 'K=5% most uncertain (max entropy)', detail: 'H(y|x) = -Σp(c)log p(c)', borderColor: '#ff9800' },
          { title: 'Expert Label', desc: 'Neurophysiologist annotates selected', borderColor: '#e91e63' },
          { title: 'Result', desc: '89% perf with 50% data → 50% cost reduction', borderColor: '#4caf50' }
        ]}
      ].map((pipeline, idx) => (
        <div key={idx} className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
          <div className="chart-title">{pipeline.title}</div>
          <FlowStep steps={pipeline.steps} />
        </div>
      ))}
    </div>
  )

  const renderValidation = () => (
    <div>
      <SectionTitle title="Validation Report" subtitle="6-Phase Quality Control Assessment — Overall: 92.1% PASSED — Production Ready (validation_report.pdf)" />
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <InfoCard title="Overall Score" value="92.1%" subtitle="PRODUCTION READY" color="#4caf50" />
        <InfoCard title="Datasets" value="10" subtitle="All validated" color="#1e88e5" />
        <InfoCard title="Total Samples" value="875,428" subtitle="Loaded & verified" color="#7c4dff" />
        <InfoCard title="Models Validated" value="3" subtitle="EEGNet, 2D-CNN, Transformer" color="#ff9800" />
        <InfoCard title="Version" value="1.0.0" subtitle="Date: 2025-12-31" color="#e91e63" />
      </div>

      {/* Phase navigation tabs */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 16, padding: '8px 12px',
        background: '#0f172a', borderRadius: 10, border: '1px solid #1e293b'
      }}>
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'phase1', label: 'Phase 1: Data Input (94.5%)' },
          { id: 'phase2', label: 'Phase 2: Preprocessing (92.8%)' },
          { id: 'phase3', label: 'Phase 3: Features (95.2%)' },
          { id: 'phase4', label: 'Phase 4: Model Training (91.5%)' },
          { id: 'phase5', label: 'Phase 5: RAG (88.5%)' },
          { id: 'phase6', label: 'Phase 6: Integration (93.0%)' }
        ].map(tab => (
          <button key={tab.id} onClick={() => setValidationPhaseView(tab.id)} style={{
            padding: '5px 10px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 10, fontWeight: validationPhaseView === tab.id ? 700 : 500,
            background: validationPhaseView === tab.id ? '#4caf50' : '#1e293b',
            color: validationPhaseView === tab.id ? '#fff' : '#94a3b8', transition: 'all 0.2s'
          }}>{tab.label}</button>
        ))}
      </div>

      {/* Overview */}
      {validationPhaseView === 'overview' && (
        <div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">Phase-Level Quality Scores</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={validationPhases} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#94a3b8" domain={[80, 100]} />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" width={160} fontSize={11} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                  formatter={(val, name, props) => [`${val}% — ${props.payload.finding}`, 'Score']} />
                <Bar dataKey="score" name="Score %" radius={[0, 4, 4, 0]}>
                  {validationPhases.map((entry, i) => <Cell key={i} fill={entry.score >= 92 ? '#4caf50' : '#ff9800'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Quality Gate Results</div>
              <DataTable
                headers={['Gate', 'Threshold', 'Actual', 'Status']}
                rows={[
                  ['Data Quality', '90', '94.5', 'PASS'],
                  ['Preprocessing Quality', '85', '92.8', 'PASS'],
                  ['Model Accuracy', '85', '89.3', 'PASS'],
                  ['Model Calibration (ECE)', '0.05', '0.023', 'PASS'],
                  ['Inference Latency (ms)', '100', '45', 'PASS'],
                  ['Explainability Score', '80', '88.5', 'PASS']
                ]}
              />
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Validation Summary (Key Findings)</div>
              <DataTable
                headers={['Phase', 'Score', 'Status', 'Key Finding']}
                rows={validationPhases.map(p => [p.name, `${p.score}%`, 'PASSED', p.finding])}
              />
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
            <div className="chart-title">Certification & Compliance</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
              <div style={{ padding: 16, background: '#0f172a', borderRadius: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#1e88e5', marginBottom: 8 }}>Statistical Validation</div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>
                  Cross-Validation: LOSO<br/>Confidence: 89.3% ± 2.1% (95% CI)<br/>McNemar p=0.0036<br/>Wilcoxon p=0.0021<br/>ECE = 0.023 (well-calibrated)
                </div>
              </div>
              <div style={{ padding: 16, background: '#0f172a', borderRadius: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#4caf50', marginBottom: 8 }}>Certification</div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>
                  IEEE Standards: Compliant<br/>Full Reproducibility (code + data + config)<br/>Complete Documentation<br/>87% Test Coverage<br/>Version: 1.0.0
                </div>
              </div>
              <div style={{ padding: 16, background: '#0f172a', borderRadius: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#ff9800', marginBottom: 8 }}>Recommendations</div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>
                  MONITOR: ICA artifact removal (88%)<br/>IMPROVE: RAG precision@5 (82%)<br/>MAINTAIN: Quarterly retraining<br/>APPROVED: Production deployment
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Phase 1: Data Input & Loading */}
      {validationPhaseView === 'phase1' && (
        <div>
          <div style={{ padding: '12px 16px', background: '#4caf5015', border: '1px solid #4caf5030', borderRadius: 10, marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div><span style={{ fontWeight: 700, color: '#4caf50', fontSize: 16 }}>Phase 1: Data Input & Loading</span><span style={{ color: '#94a3b8', marginLeft: 12, fontSize: 13 }}>Score: 94.5% — PASSED</span></div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#4caf50' }}>94.5%</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Dataset Validation Scores (%)</div>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart data={datasetValidationScores} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#94a3b8" domain={[94, 101]} />
                  <YAxis dataKey="dataset" type="category" stroke="#94a3b8" width={100} fontSize={11} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="score" name="Validation Score" radius={[0, 4, 4, 0]}>
                    {datasetValidationScores.map((entry, i) => <Cell key={i} fill={entry.score >= 99 ? '#4caf50' : entry.score >= 98 ? '#8bc34a' : '#ff9800'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Validation Criteria</div>
              {[
                { title: 'File Integrity', items: ['MD5 checksum verification', 'File corruption detection'] },
                { title: 'Format Compliance', items: ['EDF/GDF/MAT standard adherence', 'Header validation'] },
                { title: 'Channel Verification', items: ['10-20 system naming', 'Channel count verification'] },
                { title: 'Sampling Rate Check', items: ['Matches specification', 'Consistency across files'] },
                { title: 'Duration Validation', items: ['Minimum length requirement', 'Segment completeness'] },
                { title: 'Label Consistency', items: ['Matches documentation', 'No missing labels'] }
              ].map((criteria, i) => (
                <div key={i} style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: '#e2e8f0' }}>{criteria.title}</div>
                  {criteria.items.map((item, j) => (
                    <div key={j} style={{ fontSize: 10, color: '#94a3b8', paddingLeft: 12 }}>- {item}</div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
            <div className="chart-title">Identified Issues</div>
            <DataTable
              headers={['Dataset', 'Issue']}
              rows={datasetIssues.map(d => [d.dataset, d.issue])}
            />
          </div>
        </div>
      )}

      {/* Phase 2: Signal Preprocessing */}
      {validationPhaseView === 'phase2' && (
        <div>
          <div style={{ padding: '12px 16px', background: '#4caf5015', border: '1px solid #4caf5030', borderRadius: 10, marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div><span style={{ fontWeight: 700, color: '#4caf50', fontSize: 16 }}>Phase 2: Signal Preprocessing</span><span style={{ color: '#94a3b8', marginLeft: 12, fontSize: 13 }}>Score: 92.8% — PASSED</span></div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#4caf50' }}>92.8%</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Preprocessing Component Scores</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={preprocessingScores}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={10} angle={-15} textAnchor="end" height={60} />
                  <YAxis stroke="#94a3b8" domain={[80, 101]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="score" name="Score %" radius={[4, 4, 0, 0]}>
                    {preprocessingScores.map((entry, i) => <Cell key={i} fill={entry.score >= 97 ? '#4caf50' : entry.score >= 95 ? '#8bc34a' : entry.score >= 90 ? '#ff9800' : '#f44336'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Quality Improvements</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={qualityImprovements}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="metric" stroke="#94a3b8" fontSize={9} angle={-15} textAnchor="end" height={70} />
                  <YAxis stroke="#94a3b8" domain={[0, 110]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                    formatter={(val, name, props) => [`${val}${props.payload.unit}`, 'Value']} />
                  <Bar dataKey="value" name="Improvement" radius={[4, 4, 0, 0]}>
                    {qualityImprovements.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Signal-to-Noise Ratio Improvement</div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24, padding: 20 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>Before Preprocessing</div>
                  <div style={{ fontSize: 36, fontWeight: 800, color: '#f44336' }}>{snrImprovement.before} dB</div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ fontSize: 18, color: '#4caf50' }}>&#8594;</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#4caf50' }}>+{snrImprovement.gain} dB</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>After Preprocessing</div>
                  <div style={{ fontSize: 36, fontWeight: 800, color: '#4caf50' }}>{snrImprovement.after} dB</div>
                </div>
              </div>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Filter Specifications</div>
              <DataTable
                headers={['Filter', 'Type', 'Parameters', 'Key Spec']}
                rows={filterSpecs.map(f => [f.filter, f.type, f.params, f.atten || f.fpr])}
              />
            </div>
          </div>
        </div>
      )}

      {/* Phase 3: Feature Extraction */}
      {validationPhaseView === 'phase3' && (
        <div>
          <div style={{ padding: '12px 16px', background: '#4caf5015', border: '1px solid #4caf5030', borderRadius: 10, marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div><span style={{ fontWeight: 700, color: '#4caf50', fontSize: 16 }}>Phase 3: Feature Extraction</span><span style={{ color: '#94a3b8', marginLeft: 12, fontSize: 13 }}>Score: 95.2% — PASSED</span></div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#4caf50' }}>95.2%</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Feature Category Validation Scores</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={featureCategoryValidation}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="category" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" domain={[85, 101]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                    formatter={(val, name, props) => [`${val}% (${props.payload.count} features)`, name]} />
                  <Bar dataKey="score" name="Validation Score %" radius={[4, 4, 0, 0]}>
                    {featureCategoryValidation.map((entry, i) => <Cell key={i} fill={entry.score >= 98 ? '#4caf50' : entry.score >= 94 ? '#ff9800' : '#f44336'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">1D to 2D Conversion Validation</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={conversionValidation}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="method" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[90, 100]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="score" name="Score %" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Feature Distribution (127 Total)</div>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={featureBreakdown} dataKey="count" nameKey="name" cx="50%" cy="50%" innerRadius={40} outerRadius={90}
                    label={({ name, count, pct }) => `${name} (${count}, ${pct}%)`}>
                    {featureBreakdown.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Feature Quality Metrics</div>
              <div style={{ display: 'grid', gap: 12, padding: 10 }}>
                <InfoCard title="NaN Values" value={featureQualityChecks.nanValues} subtitle="No missing data" color="#4caf50" />
                <InfoCard title="Inf Values" value={featureQualityChecks.infValues} subtitle="No infinite values" color="#4caf50" />
                <InfoCard title="Constant Features" value={featureQualityChecks.constantFeatures} subtitle="No zero-variance features" color="#4caf50" />
                <InfoCard title="Highly Correlated Pairs" value={featureQualityChecks.highlyCorrelated} subtitle="Monitored" color="#ff9800" />
              </div>
              <div style={{ marginTop: 12, padding: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#4caf50', marginBottom: 4 }}>Validation Tests Passed</div>
                {['Numerical stability verified', 'Range constraints satisfied', 'Reproducibility confirmed (100%)', 'Statistical significance tested'].map((t, i) => (
                  <div key={i} style={{ fontSize: 10, color: '#94a3b8', padding: '2px 0' }}>&#10003; {t}</div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Phase 4: Model Training & Inference */}
      {validationPhaseView === 'phase4' && (
        <div>
          <div style={{ padding: '12px 16px', background: '#4caf5015', border: '1px solid #4caf5030', borderRadius: 10, marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div><span style={{ fontWeight: 700, color: '#4caf50', fontSize: 16 }}>Phase 4: Model Training & Inference</span><span style={{ color: '#94a3b8', marginLeft: 12, fontSize: 13 }}>Score: 91.5% — PASSED</span></div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#4caf50' }}>91.5%</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Model Accuracy Comparison (Threshold: 85%)</div>
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={modelAccuracyComparison}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="model" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[80, 94]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Legend />
                  <Bar dataKey="acc" name="Validation Accuracy %" radius={[4, 4, 0, 0]}>
                    {modelAccuracyComparison.map((entry, i) => <Cell key={i} fill={entry.model === 'Ensemble' ? '#ff9800' : entry.acc >= 87 ? '#4caf50' : '#1e88e5'} />)}
                  </Bar>
                  <Line type="monotone" dataKey="acc" stroke="#e91e63" strokeWidth={0} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Ensemble Model Metrics</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={ensembleMetrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="metric" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[85, 100]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="value" name="%" fill="#ff9800" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Overfitting Analysis (Train vs Validation Gap)</div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={modelAccuracyComparison.filter(m => m.gap !== null)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="model" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[80, 95]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Legend />
                  <Bar dataKey="trainAcc" name="Train Accuracy" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="acc" name="Validation Accuracy" fill="#ff9800" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', fontSize: 10, color: '#94a3b8', marginTop: 4 }}>
                Gaps: EEGNet 4.7% | 2D-CNN 3.9% | Transformer 3.4% — All within acceptable range
              </div>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Model Training Summary</div>
              <DataTable
                headers={['Model', 'Params', 'Architecture', 'Optimizer', 'Cross-Val']}
                rows={[
                  ['EEGNet', '2,548', 'Temporal-Spatial CNN', 'Adam, 100 epochs', 'LOSO, 84.5% ± 3.8%'],
                  ['2D-CNN', '148,000', 'Spectrogram classification', 'Adam, 50 epochs', 'LOSO, 86.2% ± 4.2%'],
                  ['Transformer', '502,000', 'Vision Transformer (ViT)', 'AdamW, 100 epochs', 'LOSO, 87.8% ± 3.5%'],
                  ['Ensemble', 'Weighted', 'Weights: [0.4, 0.35, 0.25]', '-', '89.3%, ECE=0.023']
                ]}
              />
            </div>
          </div>
        </div>
      )}

      {/* Phase 5: RAG Explainability */}
      {validationPhaseView === 'phase5' && (
        <div>
          <div style={{ padding: '12px 16px', background: '#ff980015', border: '1px solid #ff980030', borderRadius: 10, marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div><span style={{ fontWeight: 700, color: '#ff9800', fontSize: 16 }}>Phase 5: RAG Explainability</span><span style={{ color: '#94a3b8', marginLeft: 12, fontSize: 13 }}>Score: 88.5% — PASSED</span></div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#ff9800' }}>88.5%</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">RAG Component Validation (Threshold: 85%)</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={ragComponentValidation}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="component" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" domain={[75, 100]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="score" name="Score %" radius={[4, 4, 0, 0]}>
                    {ragComponentValidation.map((entry, i) => <Cell key={i} fill={entry.score >= 90 ? '#4caf50' : entry.score >= 85 ? '#ff9800' : '#f44336'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Retrieval Performance</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={ragRetrievalPerf}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="metric" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[70, 100]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="value" name="%" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Human Evaluation of Explanations (out of 5)</div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={ragHumanEval} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#94a3b8" domain={[0, 5]} />
                  <YAxis dataKey="criterion" type="category" stroke="#94a3b8" width={100} fontSize={11} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                    formatter={(val) => [`${val}/5.0`, 'Score']} />
                  <Bar dataKey="score" name="Score" fill="#ff9800" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">RAG System Specifications</div>
              <DataTable
                headers={['Component', 'Specifications']}
                rows={ragSystemSpecs.map(r => [r.component, r.specs])}
              />
            </div>
          </div>
        </div>
      )}

      {/* Phase 6: Integration & QC */}
      {validationPhaseView === 'phase6' && (
        <div>
          <div style={{ padding: '12px 16px', background: '#4caf5015', border: '1px solid #4caf5030', borderRadius: 10, marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div><span style={{ fontWeight: 700, color: '#4caf50', fontSize: 16 }}>Phase 6: Integration & Quality Control</span><span style={{ color: '#94a3b8', marginLeft: 12, fontSize: 13 }}>Overall Score: 92.1% — PASSED</span></div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#4caf50' }}>93.0%</div>
          </div>
          <div className="chart-card" style={{ padding: 20 }}>
            <div className="chart-title">Phase-Level Quality Scores</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={validationPhases} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#94a3b8" domain={[80, 100]} />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" width={160} fontSize={11} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                <Bar dataKey="score" name="Score %" radius={[0, 4, 4, 0]}>
                  {validationPhases.map((entry, i) => <Cell key={i} fill={entry.score >= 92 ? '#4caf50' : '#ff9800'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Quality Gate Results</div>
              <DataTable
                headers={['Gate', 'Threshold', 'Actual', 'Status']}
                rows={[
                  ['Data Quality', '90', '94.5', 'PASS'],
                  ['Preprocessing Quality', '85', '92.8', 'PASS'],
                  ['Model Accuracy', '85', '89.3', 'PASS'],
                  ['Model Calibration', '0.05', '0.023', 'PASS'],
                  ['Inference Latency', '100', '45', 'PASS'],
                  ['Explainability Score', '80', '88.5', 'PASS']
                ]}
              />
            </div>
            <div className="chart-card" style={{ padding: 20 }}>
              <div className="chart-title">Compliance & Sign-off</div>
              <div style={{ display: 'grid', gap: 12, padding: 10 }}>
                {[
                  { label: 'IEEE Standards', value: 'Compliant', color: '#4caf50' },
                  { label: 'Reproducibility', value: 'Full (code + data + config versioned)', color: '#4caf50' },
                  { label: 'Documentation', value: 'Complete', color: '#4caf50' },
                  { label: 'Testing Coverage', value: '87%', color: '#4caf50' }
                ].map((item, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: '#0f172a', borderRadius: 6 }}>
                    <span style={{ fontSize: 11, color: '#94a3b8' }}>{item.label}</span>
                    <span style={{ fontSize: 12, fontWeight: 700, color: item.color }}>{item.value}</span>
                  </div>
                ))}
                <div style={{ marginTop: 8, padding: '12px 16px', background: '#4caf5010', border: '1px solid #4caf5030', borderRadius: 8, textAlign: 'center' }}>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#4caf50' }}>92.1%</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#4caf50', marginTop: 4 }}>PRODUCTION READY</div>
                  <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>Validated by: EEG Disease Finder QA Team | Date: 2025-12-31 | v1.0.0</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )

  const renderAnalysisCharts = () => (
    <div>
      <SectionTitle title="8-Phase Analysis Framework" subtitle="111 Specialized Analysis Modules (eeg_analysis_plan.pdf)" />
      <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
        <div className="chart-title">Module Distribution Across 8 Phases</div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={analysisPhases}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} angle={-15} textAnchor="end" height={60} />
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
              formatter={(val, name, props) => [`${val} modules`, props.payload.name]} />
            <Bar dataKey="modules" name="Modules" radius={[4, 4, 0, 0]}>
              {analysisPhases.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {analysisPhases.map((phase, idx) => (
        <div key={idx} className="chart-card" style={{ padding: 16, marginBottom: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <span style={{ fontSize: 14, fontWeight: 700, color: COLORS[idx % COLORS.length] }}>{phase.phase}: {phase.name}</span>
              <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 8 }}>({phase.modules} modules)</span>
            </div>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {phase.items.split(', ').map((item, j) => (
              <span key={j} style={{
                background: `${COLORS[idx % COLORS.length]}15`,
                border: `1px solid ${COLORS[idx % COLORS.length]}30`,
                borderRadius: 6, padding: '3px 8px', fontSize: 10, color: '#e2e8f0'
              }}>{item}</span>
            ))}
          </div>
        </div>
      ))}
    </div>
  )

  const renderStatisticalAnalysis = () => (
    <div>
      <SectionTitle title="Statistical Validation" subtitle="Comprehensive hypothesis testing with confidence intervals (Section X, Tables XVII-XX)" />
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <InfoCard title="Mean F1" value="0.876" subtitle="Across 10 datasets" color="#1e88e5" />
        <InfoCard title="Mean AUC" value="0.948" subtitle="All datasets" color="#7c4dff" />
        <InfoCard title="Sensitivity" value="87.3%" subtitle="Clinical standard" color="#4caf50" />
        <InfoCard title="Specificity" value="90.5%" subtitle="Low false positive" color="#ff9800" />
        <InfoCard title="P50 Latency" value="32ms" subtitle="Real-time capable" color="#e91e63" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Statistical Tests vs Baselines (Table XX)</div>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={statisticalTests}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="comparison" stroke="#94a3b8" fontSize={10} />
              <YAxis stroke="#94a3b8" domain={[0, 1.5]} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Legend />
              <Bar dataKey="cohenD" name="Cohen's d" fill="#1e88e5" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ textAlign: 'center', fontSize: 10, color: '#94a3b8', marginTop: 4 }}>
            * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001 (Bonferroni corrected)
          </div>
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Reliability Metrics per Dataset (Table XVII)</div>
          <DataTable
            headers={['Dataset', 'ICC', 'Kappa', 'Test-Retest', 'Cronbach α']}
            rows={reliabilityMetrics.map(r => [r.dataset, r.icc, r.kappa, r.testRetest, r.alpha])}
          />
        </div>
      </div>
      <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
        <div className="chart-title">Production KPIs (Table XXVIII)</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 10 }}>
          {productionKPIs.map((kpi, i) => (
            <div key={i} style={{
              padding: '12px 16px', background: kpi.status ? '#4caf5010' : '#f4433610',
              border: `1px solid ${kpi.status ? '#4caf50' : '#f44336'}30`, borderRadius: 8
            }}>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>{kpi.metric}</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: kpi.status ? '#4caf50' : '#f44336' }}>{kpi.achieved}</div>
              <div style={{ fontSize: 10, color: '#64748b' }}>Target: {kpi.target}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )

  const renderSOTA = () => (
    <div>
      <SectionTitle title="State-of-the-Art Comparison" subtitle="Benchmarking against published methods (Table XXVII — IEEE citations)" />
      <div className="chart-card" style={{ padding: 20, marginBottom: 16 }}>
        <div className="chart-title">SOTA Accuracy Comparison (Multi-Dataset)</div>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={sotaComparison}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="method" stroke="#94a3b8" fontSize={10} angle={-20} textAnchor="end" height={60} />
            <YAxis stroke="#94a3b8" domain={[70, 95]} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
              formatter={(val, name, props) => [`${val}${name === 'F1' ? '' : '%'} (${props.payload.dataset}, ${props.payload.year})`, name]} />
            <Legend />
            <Bar dataKey="acc" name="Accuracy %" radius={[4, 4, 0, 0]}>
              {sotaComparison.map((entry, i) => <Cell key={i} fill={entry.method === 'Ours' ? '#4caf50' : '#1e88e5'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-card" style={{ padding: 20 }}>
        <div className="chart-title">Detailed SOTA Comparison Table</div>
        <DataTable
          headers={['Method', 'Dataset', 'Accuracy', 'F1', 'Year']}
          rows={sotaComparison.map(s => [s.method, s.dataset, `${s.acc}%`, s.f1, s.year])}
          highlightLast
        />
      </div>
      <div className="chart-card" style={{ padding: 20, marginTop: 16 }}>
        <div className="chart-title">Key Advantages of Our Framework</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          {[
            { title: '11-Phase Strategy', desc: 'Complete ML lifecycle with quality gates from project framing to production deployment', color: '#1e88e5' },
            { title: '111 Analysis Modules', desc: 'Unprecedented depth: 8-phase analysis plan covering data, model, performance, sensitivity, statistical', color: '#7c4dff' },
            { title: '127 Features Documented', desc: 'All features with extraction formulas, not black-box — time, frequency, connectivity, entropy', color: '#4caf50' },
            { title: '1D→2D Conversion', desc: 'CWT/STFT equations with parameters for converting EEG signals to spectrogram images', color: '#ff9800' },
            { title: 'Active Learning', desc: '89% performance with 50% labeled data, reducing annotation cost by 50% ($50-100/hr saved)', color: '#e91e63' },
            { title: 'Multi-Dataset (10)', desc: 'Validated on 10 datasets, 6,200+ subjects covering 5 neurological conditions', color: '#00bcd4' }
          ].map((adv, i) => (
            <div key={i} style={{ padding: 16, background: `${adv.color}08`, border: `1px solid ${adv.color}30`, borderRadius: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: adv.color, marginBottom: 6 }}>{adv.title}</div>
              <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{adv.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )

  const renderGovernance = () => (
    <div>
      <SectionTitle title="Governance & Responsible AI" subtitle="5-Pillar RAI Audit — Clinical Validation Framework (Tables XII-XIII)" />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">RAI Pillar Scores</div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={raiRadar}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="subject" stroke="#94a3b8" fontSize={11} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#94a3b8" />
              <Radar name="Score %" dataKey="A" stroke="#e91e63" fill="#e91e63" fillOpacity={0.3} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Clinical Validation Matrix (12 Domains — Table XII)</div>
          <DataTable
            headers={['#', 'Domain', 'Sub-Analyses', 'Metric']}
            rows={[
              [1, 'Diagnostic Validity', 'Sensitivity, Specificity, PPV, AUC', 'Sensitivity (%)'],
              [2, 'Agreement & Consistency', "Model vs Clinician, Inter-Rater, Cohen's κ", 'κ / ICC'],
              [3, 'Risk & Safety', 'FN Risk, FP Risk, Worst-Case Subject', 'FN/FP Rate'],
              [4, 'Subject-Wise Validation', 'Patient-Wise, LOSO Clinical Evaluation', 'Patient Score'],
              [5, 'Population-Level', 'Age/Gender Subgroups, Comorbidity', 'Δ Accuracy'],
              [6, 'Robustness & Noise', 'Signal Noise, Artifact Resistance', 'Perf Drop (%)'],
              [7, 'Temporal Stability', 'Session-Wise, Drift Sensitivity', 'Δ F1'],
              [8, 'Domain Transferability', 'Lab→Real-World, Device/Sensor Shift', 'AUC Drop'],
              [9, 'Deployment Performance', 'Latency, Throughput, Resource Usage', 'Latency (ms)'],
              [10, 'Clinical Interpretability', 'Feature Attribution, Clinician Trust', 'Expert Score'],
              [11, 'Operational Reliability', 'Stability Under Load, Failure Frequency', 'Failure Rate'],
              [12, 'Statistical Validation', 'CI, Significance Testing', 'Mean ± CI']
            ]}
          />
        </div>
      </div>
    </div>
  )

  const renderTechniques = () => (
    <div>
      <SectionTitle title="Techniques & Frameworks" subtitle="Complete technology inventory across all research papers" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
        {techniques.map((cat, i) => (
          <div key={i} className="chart-card" style={{ padding: 16 }}>
            <div className="chart-title" style={{ color: COLORS[i % COLORS.length] }}>{cat.category}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
              {cat.items.map((item, j) => (
                <span key={j} style={{
                  background: `${COLORS[i % COLORS.length]}15`,
                  border: `1px solid ${COLORS[i % COLORS.length]}30`,
                  borderRadius: 6, padding: '4px 10px', fontSize: 11, color: '#e2e8f0'
                }}>{item}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  const renderMetricsDashboard = () => (
    <div>
      <SectionTitle title="Complete Metrics Dashboard" subtitle="All performance, clinical, deployment, and research metrics at a glance" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12, marginBottom: 16 }}>
        <InfoCard title="Mean Accuracy" value="89.2%" subtitle="10 datasets" color="#1e88e5" />
        <InfoCard title="Mean F1" value="0.876" subtitle="LOSO validated" color="#4caf50" />
        <InfoCard title="Mean AUC" value="0.948" subtitle="All datasets" color="#7c4dff" />
        <InfoCard title="Sensitivity" value="87.3%" subtitle="Clinical grade" color="#ff9800" />
        <InfoCard title="Specificity" value="90.5%" subtitle="Low FP rate" color="#e91e63" />
        <InfoCard title="P50 Latency" value="32ms" subtitle="GPU (RTX 3080)" color="#00bcd4" />
        <InfoCard title="Throughput" value="156/s" subtitle="Production" color="#8bc34a" />
        <InfoCard title="Availability" value="99.95%" subtitle="24/7 uptime" color="#4caf50" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Research Metrics</div>
          <div style={{ display: 'grid', gap: 8 }}>
            <InfoCard title="Datasets" value="10" subtitle="5 conditions" color="#1e88e5" />
            <InfoCard title="Subjects" value="6,200+" color="#7c4dff" />
            <InfoCard title="Features" value="127" subtitle="5 categories" color="#4caf50" />
            <InfoCard title="Analysis Modules" value="111" subtitle="8 phases" color="#ff9800" />
            <InfoCard title="Project Phases" value="11" subtitle="Full lifecycle" color="#e91e63" />
          </div>
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Clinical Metrics</div>
          <div style={{ display: 'grid', gap: 8 }}>
            <InfoCard title="Cohen's κ" value="0.81" subtitle="Substantial agreement" color="#1e88e5" />
            <InfoCard title="Fleiss' κ" value="0.78" subtitle="Inter-rater" color="#7c4dff" />
            <InfoCard title="ECE" value="0.023" subtitle="Well calibrated" color="#4caf50" />
            <InfoCard title="Expert Agreement" value="89.8%" subtitle="3 raters, κ=0.81" color="#ff9800" />
            <InfoCard title="RAG Faithfulness" value="0.87" subtitle="Grounded" color="#e91e63" />
          </div>
        </div>
        <div className="chart-card" style={{ padding: 20 }}>
          <div className="chart-title">Deployment Metrics</div>
          <div style={{ display: 'grid', gap: 8 }}>
            <InfoCard title="GPU Memory" value="2.8GB" subtitle="RTX 3080" color="#1e88e5" />
            <InfoCard title="CPU Inference" value="85ms" subtitle="Intel i7-10700" color="#7c4dff" />
            <InfoCard title="Model Size" value="<200K" subtitle="Parameters" color="#4caf50" />
            <InfoCard title="Error Rate" value="0.05%" subtitle="Production" color="#ff9800" />
            <InfoCard title="Drift Alert" value="2.1%/wk" subtitle="<5% threshold" color="#e91e63" />
          </div>
        </div>
      </div>
    </div>
  )

  const renderSection = () => {
    switch (activeSection) {
      case 'system': return renderSystemArch()
      case 'datasets': return renderDatasets()
      case 'data': return renderDataArch()
      case 'model': return renderModelArch()
      case 'genai': return renderGenAI()
      case 'pipeline': return renderPipelineFlows()
      case 'validation': return renderValidation()
      case 'analysis': return renderAnalysisCharts()
      case 'statistical': return renderStatisticalAnalysis()
      case 'sota': return renderSOTA()
      case 'governance': return renderGovernance()
      case 'techniques': return renderTechniques()
      case 'metrics': return renderMetricsDashboard()
      case 'stressrag': return renderStressRag()
      default: return renderSystemArch()
    }
  }

  return (
    <div>
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20,
        padding: '12px 16px', background: '#0f172a', borderRadius: 12, border: '1px solid #1e293b'
      }}>
        {sections.map(sec => (
          <button
            key={sec.id}
            onClick={() => setActiveSection(sec.id)}
            style={{
              padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 12, fontWeight: activeSection === sec.id ? 700 : 500,
              background: activeSection === sec.id ? '#1e88e5' : '#1e293b',
              color: activeSection === sec.id ? '#fff' : '#94a3b8',
              transition: 'all 0.2s'
            }}
          >
            {sec.label}
          </button>
        ))}
      </div>
      {renderSection()}
    </div>
  )
}
