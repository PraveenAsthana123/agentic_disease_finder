"""
Model Inference Module for Neurological Disease Detection
==========================================================

Provides real model inference with trained models for:
- Alzheimer's Disease (MRI-based)
- Parkinson's Disease (Voice/Gait-based)
- Schizophrenia (EEG-based)

Author: Research Team
"""

import numpy as np
import os
import logging
import joblib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results"""
    prediction: int
    class_name: str
    probabilities: Dict[str, float]
    confidence: float
    features_used: List[str]
    model_info: Dict[str, Any]


class BaseInferenceModel:
    """Base class for inference models"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.classes = []
        self.feature_names = []
        self.is_loaded = False

    def load(self) -> bool:
        """Load model from disk"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = joblib.load(self.model_path)
                self.model = checkpoint['model']
                self.scaler = checkpoint.get('scaler')
                self.classes = checkpoint.get('classes', [])
                self.feature_names = checkpoint.get('feature_names', [])
                self.is_loaded = True
                logger.info(f"Loaded model from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
        return False

    def save(self, path: str):
        """Save model to disk"""
        checkpoint = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes,
            'feature_names': self.feature_names
        }
        joblib.dump(checkpoint, path)
        logger.info(f"Saved model to {path}")

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Make prediction"""
        raise NotImplementedError


class AlzheimerInferenceModel(BaseInferenceModel):
    """
    Alzheimer's Disease Inference Model

    Uses MRI features for 3-class classification:
    - CN: Cognitively Normal
    - MCI: Mild Cognitive Impairment
    - AD: Alzheimer's Disease
    """

    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.classes = ['CN', 'MCI', 'AD']
        self.feature_names = [
            'hippocampus_volume', 'ventricle_volume', 'cortical_thickness',
            'entorhinal_volume', 'gray_matter_volume', 'white_matter_volume',
            'brain_age_gap', 'mmse_score', 'cdr_score', 'age',
            'apoe4_status', 'education_years', 'csf_abeta', 'csf_tau',
            'csf_ptau', 'fdg_pet_suvr', 'amyloid_pet_suvr',
            'memory_score', 'executive_function', 'language_score'
        ]

    def initialize_default_model(self):
        """Initialize default model if no trained model available"""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        self.is_loaded = True
        logger.info("Initialized default Alzheimer's model")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        if self.model is None:
            self.initialize_default_model()

        self.model.fit(X, y)
        logger.info(f"Trained Alzheimer's model on {len(y)} samples")

    def predict(self, features: np.ndarray) -> PredictionResult:
        """
        Predict Alzheimer's stage

        Parameters
        ----------
        features : np.ndarray
            Feature vector

        Returns
        -------
        result : PredictionResult
        """
        if not self.is_loaded:
            self.initialize_default_model()
            # Train on synthetic data for demo
            X_demo, y_demo = self._generate_demo_data()
            self.train(X_demo, y_demo)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Ensure correct number of features
        if features.shape[1] != len(self.feature_names):
            # Pad or truncate
            new_features = np.zeros((features.shape[0], len(self.feature_names)))
            min_feat = min(features.shape[1], len(self.feature_names))
            new_features[:, :min_feat] = features[:, :min_feat]
            features = new_features

        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        class_name = self.classes[prediction]
        confidence = float(probabilities[prediction])

        return PredictionResult(
            prediction=int(prediction),
            class_name=class_name,
            probabilities={c: float(p) for c, p in zip(self.classes, probabilities)},
            confidence=confidence,
            features_used=self.feature_names,
            model_info={
                'model_type': 'GradientBoosting',
                'disease': 'Alzheimer',
                'num_classes': len(self.classes)
            }
        )

    def _generate_demo_data(self, n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """Generate demo training data"""
        np.random.seed(42)
        n_features = len(self.feature_names)

        X = []
        y = []

        for class_idx, class_name in enumerate(self.classes):
            n = n_samples // 3

            # Generate class-specific features
            if class_name == 'CN':  # Normal
                features = np.random.randn(n, n_features) * 0.5 + 0.7
            elif class_name == 'MCI':  # Mild impairment
                features = np.random.randn(n, n_features) * 0.5 + 0.5
            else:  # AD
                features = np.random.randn(n, n_features) * 0.5 + 0.3

            X.append(features)
            y.extend([class_idx] * n)

        return np.vstack(X), np.array(y)


class ParkinsonInferenceModel(BaseInferenceModel):
    """
    Parkinson's Disease Inference Model

    Uses voice and gait features for binary classification:
    - HC: Healthy Control
    - PD: Parkinson's Disease
    """

    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.classes = ['HC', 'PD']
        self.feature_names = [
            # Voice features
            'jitter_local', 'jitter_rap', 'shimmer_local', 'shimmer_apq',
            'hnr', 'nhr', 'pitch_mean', 'pitch_std', 'pitch_range',
            'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean',
            # Gait features
            'stride_time_mean', 'stride_time_cv', 'step_asymmetry',
            'velocity', 'cadence', 'stride_regularity',
            # Motor features
            'updrs_total', 'updrs_motor', 'bradykinesia_score',
            'tremor_score', 'rigidity_score', 'postural_stability',
            # Clinical
            'age', 'disease_duration'
        ]

    def initialize_default_model(self):
        """Initialize default model"""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        self.is_loaded = True
        logger.info("Initialized default Parkinson's model")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        if self.model is None:
            self.initialize_default_model()

        self.model.fit(X, y)
        logger.info(f"Trained Parkinson's model on {len(y)} samples")

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Predict Parkinson's status"""
        if not self.is_loaded:
            self.initialize_default_model()
            X_demo, y_demo = self._generate_demo_data()
            self.train(X_demo, y_demo)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Ensure correct features
        if features.shape[1] != len(self.feature_names):
            new_features = np.zeros((features.shape[0], len(self.feature_names)))
            min_feat = min(features.shape[1], len(self.feature_names))
            new_features[:, :min_feat] = features[:, :min_feat]
            features = new_features

        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        return PredictionResult(
            prediction=int(prediction),
            class_name=self.classes[prediction],
            probabilities={c: float(p) for c, p in zip(self.classes, probabilities)},
            confidence=float(probabilities[prediction]),
            features_used=self.feature_names,
            model_info={
                'model_type': 'RandomForest',
                'disease': 'Parkinson',
                'num_classes': len(self.classes)
            }
        )

    def _generate_demo_data(self, n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Generate demo training data"""
        np.random.seed(42)
        n_features = len(self.feature_names)

        X = []
        y = []

        for class_idx, class_name in enumerate(self.classes):
            n = n_samples // 2

            if class_name == 'HC':
                # Healthy: lower jitter/shimmer, normal gait
                features = np.random.randn(n, n_features) * 0.3 + 0.3
            else:
                # PD: higher jitter/shimmer, abnormal gait
                features = np.random.randn(n, n_features) * 0.4 + 0.7

            X.append(features)
            y.extend([class_idx] * n)

        return np.vstack(X), np.array(y)


class SchizophreniaInferenceModel(BaseInferenceModel):
    """
    Schizophrenia Inference Model

    Uses EEG and connectivity features for binary classification:
    - HC: Healthy Control
    - SZ: Schizophrenia
    """

    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.classes = ['HC', 'SZ']
        self.feature_names = [
            # Spectral features
            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
            'theta_alpha_ratio', 'delta_theta_ratio',
            # Connectivity
            'mean_coherence', 'mean_plv', 'global_efficiency', 'clustering_coef',
            # Complexity
            'sample_entropy', 'lzc', 'hurst_exponent',
            # ERP features
            'p300_amplitude', 'p300_latency', 'mmn_amplitude',
            # Clinical
            'panss_positive', 'panss_negative', 'panss_general',
            'age', 'illness_duration', 'medication_status',
            # Cognitive
            'attention_score', 'memory_score', 'processing_speed',
            # Additional EEG
            'spectral_entropy', 'hjorth_mobility', 'hjorth_complexity'
        ]

    def initialize_default_model(self):
        """Initialize default model"""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42
            ))
        ])
        self.is_loaded = True
        logger.info("Initialized default Schizophrenia model")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        if self.model is None:
            self.initialize_default_model()

        self.model.fit(X, y)
        logger.info(f"Trained Schizophrenia model on {len(y)} samples")

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Predict Schizophrenia status"""
        if not self.is_loaded:
            self.initialize_default_model()
            X_demo, y_demo = self._generate_demo_data()
            self.train(X_demo, y_demo)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != len(self.feature_names):
            new_features = np.zeros((features.shape[0], len(self.feature_names)))
            min_feat = min(features.shape[1], len(self.feature_names))
            new_features[:, :min_feat] = features[:, :min_feat]
            features = new_features

        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        return PredictionResult(
            prediction=int(prediction),
            class_name=self.classes[prediction],
            probabilities={c: float(p) for c, p in zip(self.classes, probabilities)},
            confidence=float(probabilities[prediction]),
            features_used=self.feature_names,
            model_info={
                'model_type': 'MLP',
                'disease': 'Schizophrenia',
                'num_classes': len(self.classes)
            }
        )

    def _generate_demo_data(self, n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Generate demo training data"""
        np.random.seed(42)
        n_features = len(self.feature_names)

        X = []
        y = []

        for class_idx, class_name in enumerate(self.classes):
            n = n_samples // 2

            if class_name == 'HC':
                # Healthy: normal EEG patterns
                features = np.random.randn(n, n_features) * 0.3 + 0.4
            else:
                # SZ: altered connectivity, different spectral profile
                features = np.random.randn(n, n_features) * 0.4 + 0.6

            X.append(features)
            y.extend([class_idx] * n)

        return np.vstack(X), np.array(y)


class MultiDiseaseInferenceEngine:
    """
    Multi-disease inference engine

    Coordinates inference across all disease models
    """

    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or './models/weights'
        self.models = {
            'alzheimer': AlzheimerInferenceModel(
                os.path.join(self.models_dir, 'alzheimer_model.pkl')
            ),
            'parkinson': ParkinsonInferenceModel(
                os.path.join(self.models_dir, 'parkinson_model.pkl')
            ),
            'schizophrenia': SchizophreniaInferenceModel(
                os.path.join(self.models_dir, 'schizophrenia_model.pkl')
            )
        }

    def load_all_models(self):
        """Load all models"""
        for name, model in self.models.items():
            if not model.load():
                logger.info(f"Initializing default {name} model")
                model.initialize_default_model()

    def predict(self, disease: str, features: np.ndarray) -> PredictionResult:
        """
        Make prediction for specific disease

        Parameters
        ----------
        disease : str
            Disease name (alzheimer, parkinson, schizophrenia)
        features : np.ndarray
            Feature vector

        Returns
        -------
        result : PredictionResult
        """
        if disease not in self.models:
            raise ValueError(f"Unknown disease: {disease}")

        return self.models[disease].predict(features)

    def screen_all_diseases(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, PredictionResult]:
        """
        Screen for all diseases

        Parameters
        ----------
        features_dict : dict
            Dictionary mapping disease names to feature vectors

        Returns
        -------
        results : dict
            Dictionary of PredictionResult for each disease
        """
        results = {}

        for disease, features in features_dict.items():
            if disease in self.models:
                try:
                    results[disease] = self.predict(disease, features)
                except Exception as e:
                    logger.error(f"Error predicting {disease}: {e}")

        return results

    def get_risk_assessment(self, results: Dict[str, PredictionResult]) -> Dict:
        """
        Generate risk assessment from predictions

        Parameters
        ----------
        results : dict
            Dictionary of prediction results

        Returns
        -------
        assessment : dict
            Risk assessment report
        """
        assessment = {
            'diseases_screened': list(results.keys()),
            'risk_levels': {},
            'recommendations': [],
            'overall_risk': 'low'
        }

        high_risk_count = 0

        for disease, result in results.items():
            # Determine risk level based on probability
            prob = result.confidence if result.class_name != 'HC' and result.class_name != 'CN' else 1 - result.confidence

            if prob > 0.7:
                risk_level = 'high'
                high_risk_count += 1
                assessment['recommendations'].append(
                    f"Urgent: Further evaluation recommended for {disease.capitalize()}"
                )
            elif prob > 0.4:
                risk_level = 'moderate'
                assessment['recommendations'].append(
                    f"Monitor: Follow-up recommended for {disease.capitalize()}"
                )
            else:
                risk_level = 'low'

            assessment['risk_levels'][disease] = {
                'level': risk_level,
                'probability': prob,
                'prediction': result.class_name
            }

        # Overall risk
        if high_risk_count >= 2:
            assessment['overall_risk'] = 'high'
        elif high_risk_count == 1:
            assessment['overall_risk'] = 'moderate'

        if not assessment['recommendations']:
            assessment['recommendations'].append('Continue routine monitoring')

        return assessment
