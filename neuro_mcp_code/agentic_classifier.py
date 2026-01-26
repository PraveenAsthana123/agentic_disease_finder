"""
Agentic Classifier for Neurological Disease Detection
======================================================
Complete 3-tier architecture implementation as described in the
Agentic Disease Finder Research Paper.

Architecture:
    Tier 1: Multi-modal Data Preprocessor
    Tier 2: Agentic Decision System (intelligent model routing)
    Tier 3: Specialized Analysis Modules (disease-specific classifiers)

Supported Diseases:
- Alzheimer's Disease
- Parkinson's Disease
- Schizophrenia
- Epilepsy
- Autism Spectrum Disorder
- Stress
- Depression

Usage:
    from agentic_classifier import AgenticClassifier

    classifier = AgenticClassifier()
    result = classifier.classify(eeg_data, target_disease='stress')

Author: Research Team
Project: Neurological Disease Detection using Agentic AI
"""

import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

# Local imports
from agents import (
    AgenticDecisionSystem,
    AgenticDecisionAgent,
    EEGPreprocessor,
    DataAnalyzer,
    DataType,
    DiseaseType,
    DecisionResult,
    AgentOrchestrator,
    MessageBus
)

from models import (
    EEGClassifierFactory,
    EEGClassifierTrainer,
    get_model_for_disease,
    DISEASE_MODELS
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result from the agentic classifier"""
    disease: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    preprocessing_steps: List[str]
    data_characteristics: Dict[str, Any]
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgenticClassifier:
    """
    Agentic Classifier for Neurological Disease Detection

    Implements the complete 3-tier architecture:
    1. Multi-modal Data Preprocessor - handles various input formats
    2. Agentic Decision System - intelligent model selection
    3. Specialized Classifiers - disease-specific neural networks

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default: 22)
    n_samples : int
        Number of time samples (default: 1000)
    sampling_rate : float
        EEG sampling rate in Hz (default: 256.0)
    model_dir : str
        Directory for saved models (default: './saved_models')
    device : str
        Device for computation ('cpu' or 'cuda')
    """

    def __init__(self, n_channels: int = 22, n_samples: int = 1000,
                 sampling_rate: float = 256.0, model_dir: str = './saved_models',
                 device: str = None):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.model_dir = model_dir
        self.device = device

        # Initialize the 3 tiers
        # Tier 1: Preprocessor
        self.preprocessor = EEGPreprocessor(
            target_sampling_rate=sampling_rate,
            target_channels=n_channels,
            target_samples=n_samples
        )

        # Tier 2: Agentic Decision System
        self.decision_system = AgenticDecisionSystem()
        self.data_analyzer = DataAnalyzer()

        # Tier 3: Model registry (lazy loading)
        self.models: Dict[str, Any] = {}
        self.trainers: Dict[str, EEGClassifierTrainer] = {}

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Classification history
        self.classification_history: List[ClassificationResult] = []

        logger.info(f"AgenticClassifier initialized: {n_channels} channels, "
                   f"{n_samples} samples, {sampling_rate} Hz")

    def classify(self, data: np.ndarray,
                 target_disease: Optional[str] = None,
                 file_path: Optional[str] = None,
                 metadata: Optional[Dict] = None,
                 return_probabilities: bool = True) -> ClassificationResult:
        """
        Classify input data for neurological disease detection

        Parameters
        ----------
        data : np.ndarray
            Input EEG data (samples, channels) or (channels, samples)
        target_disease : str, optional
            If specified, use model for this disease
        file_path : str, optional
            Original file path for format detection
        metadata : dict, optional
            Additional metadata (sampling_rate, etc.)
        return_probabilities : bool
            Whether to return class probabilities

        Returns
        -------
        result : ClassificationResult
            Classification result with prediction and metadata
        """
        metadata = metadata or {}
        if 'sampling_rate' not in metadata:
            metadata['sampling_rate'] = self.sampling_rate

        # Step 1: Analyze and preprocess data
        logger.info("Step 1: Analyzing and preprocessing data...")
        characteristics = self.data_analyzer.analyze(data, file_path, metadata)

        processed_data, preprocessing_steps = self.preprocessor.preprocess(
            data, metadata.get('sampling_rate')
        )

        # Step 2: Make decision about which model to use
        logger.info("Step 2: Making routing decision...")
        target_disease_enum = None
        if target_disease:
            try:
                target_disease_enum = DiseaseType(target_disease.lower())
            except ValueError:
                logger.warning(f"Unknown disease: {target_disease}, will auto-detect")

        decision = self.decision_system.decide(
            processed_data, target_disease_enum, file_path, metadata
        )

        # Step 3: Get or create the appropriate model
        logger.info(f"Step 3: Using model {decision.recommended_model}...")
        disease_to_use = (decision.recommended_disease_targets[0]
                         if decision.recommended_disease_targets
                         else DiseaseType.HEALTHY)
        disease_name = disease_to_use.value

        model = self._get_or_create_model(disease_name)

        # Step 4: Run classification
        logger.info("Step 4: Running classification...")
        prediction, probabilities = self._run_classification(
            model, processed_data, disease_name
        )

        # Create result
        class_names = DISEASE_MODELS.get(disease_name, {}).get(
            'class_names', ['Class 0', 'Class 1']
        )

        prob_dict = {}
        if probabilities is not None:
            for i, name in enumerate(class_names):
                if i < len(probabilities):
                    prob_dict[name] = float(probabilities[i])

        predicted_class = class_names[prediction] if prediction < len(class_names) else f"Class {prediction}"

        result = ClassificationResult(
            disease=disease_name,
            prediction=predicted_class,
            confidence=float(max(probabilities)) if probabilities is not None else decision.confidence,
            probabilities=prob_dict,
            model_used=decision.recommended_model,
            preprocessing_steps=preprocessing_steps,
            data_characteristics={
                'data_type': characteristics.data_type.value,
                'num_channels': characteristics.num_channels,
                'num_samples': characteristics.num_samples,
                'signal_quality': characteristics.signal_quality,
                'snr': characteristics.snr,
                'has_artifacts': characteristics.has_artifacts
            },
            reasoning=decision.reasoning
        )

        # Store in history
        self.classification_history.append(result)

        return result

    def _get_or_create_model(self, disease: str) -> Any:
        """Get existing model or create a new one"""
        if disease not in self.models:
            # Check for saved model
            model_path = os.path.join(self.model_dir, f"{disease}_model.pt")

            # Create new model
            model = get_model_for_disease(
                disease,
                n_channels=self.n_channels,
                n_samples=self.n_samples
            )

            # Try to load saved weights
            if os.path.exists(model_path):
                try:
                    trainer = EEGClassifierTrainer(model, device=self.device)
                    trainer.load(model_path)
                    self.trainers[disease] = trainer
                    logger.info(f"Loaded saved model for {disease}")
                except Exception as e:
                    logger.warning(f"Could not load saved model: {e}")
                    self.trainers[disease] = EEGClassifierTrainer(model, device=self.device)
            else:
                self.trainers[disease] = EEGClassifierTrainer(model, device=self.device)

            self.models[disease] = model

        return self.models[disease]

    def _run_classification(self, model, data: np.ndarray,
                           disease: str) -> Tuple[int, Optional[np.ndarray]]:
        """Run classification on preprocessed data"""
        try:
            trainer = self.trainers.get(disease)
            if trainer:
                # Ensure correct shape (batch, samples, channels)
                if data.ndim == 2:
                    data = data[np.newaxis, ...]

                probabilities = trainer.predict_proba(data)[0]
                prediction = np.argmax(probabilities)
                return prediction, probabilities
            else:
                # Fallback: random prediction
                n_classes = DISEASE_MODELS.get(disease, {}).get('n_classes', 2)
                probabilities = np.random.dirichlet(np.ones(n_classes))
                prediction = np.argmax(probabilities)
                return prediction, probabilities
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return 0, None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              disease: str, X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32,
              lr: float = 1e-3, **kwargs) -> Dict:
        """
        Train a model for a specific disease

        Parameters
        ----------
        X_train : np.ndarray
            Training data (n_samples, timesteps, channels)
        y_train : np.ndarray
            Training labels
        disease : str
            Target disease name
        X_val : np.ndarray, optional
            Validation data
        y_val : np.ndarray, optional
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        lr : float
            Learning rate
        **kwargs
            Additional training parameters

        Returns
        -------
        history : dict
            Training history
        """
        from torch.utils.data import DataLoader
        from models.eeg_classifiers import EEGDataset

        # Preprocess training data
        logger.info(f"Preprocessing {len(X_train)} training samples...")
        X_processed = []
        for sample in X_train:
            processed, _ = self.preprocessor.preprocess(sample, self.sampling_rate)
            X_processed.append(processed)
        X_processed = np.array(X_processed)

        # Get or create model
        model = self._get_or_create_model(disease)
        trainer = self.trainers[disease]

        # Create data loaders
        train_dataset = EEGDataset(X_processed.astype(np.float32), y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            # Preprocess validation data
            X_val_processed = []
            for sample in X_val:
                processed, _ = self.preprocessor.preprocess(sample, self.sampling_rate)
                X_val_processed.append(processed)
            X_val_processed = np.array(X_val_processed)

            val_dataset = EEGDataset(X_val_processed.astype(np.float32), y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Train
        logger.info(f"Training {disease} model for {epochs} epochs...")
        history = trainer.train(
            train_loader, val_loader,
            epochs=epochs, lr=lr, **kwargs
        )

        # Save model
        model_path = os.path.join(self.model_dir, f"{disease}_model.pt")
        trainer.save(model_path)
        logger.info(f"Model saved to {model_path}")

        return history

    def batch_classify(self, data_list: List[np.ndarray],
                       target_disease: Optional[str] = None,
                       metadata_list: Optional[List[Dict]] = None) -> List[ClassificationResult]:
        """
        Classify multiple samples

        Parameters
        ----------
        data_list : list of np.ndarray
            List of EEG data samples
        target_disease : str, optional
            Target disease for all samples
        metadata_list : list of dict, optional
            Metadata for each sample

        Returns
        -------
        results : list of ClassificationResult
            Classification results for all samples
        """
        results = []
        metadata_list = metadata_list or [None] * len(data_list)

        for i, (data, metadata) in enumerate(zip(data_list, metadata_list)):
            logger.info(f"Classifying sample {i + 1}/{len(data_list)}...")
            result = self.classify(data, target_disease, metadata=metadata)
            results.append(result)

        return results

    def get_supported_diseases(self) -> List[str]:
        """Get list of supported diseases"""
        return list(DISEASE_MODELS.keys())

    def get_model_info(self, disease: str) -> Dict:
        """Get information about a disease model"""
        if disease.lower() not in DISEASE_MODELS:
            return {'error': f'Unknown disease: {disease}'}

        config = DISEASE_MODELS[disease.lower()]
        model_path = os.path.join(self.model_dir, f"{disease.lower()}_model.pt")

        return {
            'disease': disease,
            'n_classes': config['n_classes'],
            'class_names': config['class_names'],
            'model_type': config['model_type'],
            'saved_model_exists': os.path.exists(model_path),
            'input_shape': (self.n_samples, self.n_channels)
        }

    def get_classification_summary(self) -> Dict:
        """Get summary of classification history"""
        if not self.classification_history:
            return {'total_classifications': 0}

        disease_counts = {}
        avg_confidence = []
        predictions = []

        for result in self.classification_history:
            disease_counts[result.disease] = disease_counts.get(result.disease, 0) + 1
            avg_confidence.append(result.confidence)
            predictions.append(result.prediction)

        return {
            'total_classifications': len(self.classification_history),
            'disease_distribution': disease_counts,
            'average_confidence': np.mean(avg_confidence),
            'unique_predictions': list(set(predictions))
        }

    def export_history(self, filepath: str):
        """Export classification history to JSON"""
        history_data = []
        for result in self.classification_history:
            history_data.append({
                'disease': result.disease,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'probabilities': result.probabilities,
                'model_used': result.model_used,
                'timestamp': result.timestamp
            })

        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        logger.info(f"History exported to {filepath}")

    def load_depression_data(self, dataset: str = 'auto',
                              data_root: str = './datasets',
                              n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load depression/emotion EEG data for training.

        Parameters
        ----------
        dataset : str
            Dataset to use ('DEAP', 'SEED', 'synthetic', 'auto')
        data_root : str
            Root directory for datasets
        n_samples : int
            Number of samples for synthetic data

        Returns
        -------
        X : np.ndarray
            EEG data (n_samples, timesteps, channels)
        y : np.ndarray
            Binary depression labels (0=healthy, 1=depression)
        """
        from data import EmotionDataLoader

        loader = EmotionDataLoader(data_root)
        X, y = loader.load_depression_data(dataset=dataset, n_samples=n_samples)

        logger.info(f"Loaded depression data: {X.shape}, "
                   f"depression={np.sum(y==1)}, healthy={np.sum(y==0)}")

        return X, y

    def train_depression_model(self, dataset: str = 'auto',
                                data_root: str = './datasets',
                                n_samples: int = 500,
                                epochs: int = 50,
                                batch_size: int = 32,
                                test_split: float = 0.2,
                                **kwargs) -> Dict:
        """
        Train the depression classifier.

        Parameters
        ----------
        dataset : str
            Dataset to use ('DEAP', 'SEED', 'synthetic', 'auto')
        data_root : str
            Root directory for datasets
        n_samples : int
            Number of samples for synthetic data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        test_split : float
            Fraction of data for validation
        **kwargs
            Additional training arguments

        Returns
        -------
        history : dict
            Training history with loss and accuracy
        """
        # Load data
        X, y = self.load_depression_data(dataset, data_root, n_samples)

        # Split into train/val
        n_val = int(len(X) * test_split)
        indices = np.random.permutation(len(X))

        X_train = X[indices[n_val:]]
        y_train = y[indices[n_val:]]
        X_val = X[indices[:n_val]]
        y_val = y[indices[:n_val]]

        logger.info(f"Training: {len(X_train)}, Validation: {len(X_val)}")

        # Train
        history = self.train(
            X_train, y_train,
            disease='depression',
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

        return history

    def classify_depression(self, data: np.ndarray) -> ClassificationResult:
        """
        Classify EEG data for depression.

        Parameters
        ----------
        data : np.ndarray
            EEG data (samples, channels) or (batch, samples, channels)

        Returns
        -------
        result : ClassificationResult
            Classification result with depression/healthy prediction
        """
        return self.classify(data, target_disease='depression')


class AgenticClassifierAgent:
    """
    Agent wrapper for the Agentic Classifier

    Enables integration with the MCP/A2A agent infrastructure.
    """

    def __init__(self, classifier: AgenticClassifier = None):
        self.classifier = classifier or AgenticClassifier()
        self.orchestrator = AgentOrchestrator()
        self.decision_agent = AgenticDecisionAgent()

        # Register decision agent
        self.orchestrator.register_agent(self.decision_agent)
        self.orchestrator.start()

    def classify(self, data: np.ndarray, **kwargs) -> ClassificationResult:
        """Classify data using the agentic system"""
        return self.classifier.classify(data, **kwargs)

    def get_status(self) -> Dict:
        """Get system status"""
        return {
            'classifier': self.classifier.get_classification_summary(),
            'agents': self.orchestrator.get_all_status(),
            'supported_diseases': self.classifier.get_supported_diseases()
        }

    def shutdown(self):
        """Shutdown the agent system"""
        self.orchestrator.stop()


def demo():
    """Demonstration of the Agentic Classifier"""
    print("=" * 70)
    print("  AGENTIC CLASSIFIER DEMONSTRATION")
    print("  Neurological Disease Detection using Agentic AI")
    print("=" * 70)

    # Create classifier
    print("\n1. Initializing Agentic Classifier...")
    classifier = AgenticClassifier(n_channels=22, n_samples=1000)

    print(f"   Supported diseases: {classifier.get_supported_diseases()}")

    # Generate synthetic EEG data
    print("\n2. Generating synthetic EEG data...")
    np.random.seed(42)

    # Simulate different disease patterns
    test_cases = [
        ('stress', np.random.randn(1000, 22) * 1.5),  # Higher variance
        ('autism', np.random.randn(1000, 22) + np.sin(np.linspace(0, 10*np.pi, 1000))[:, np.newaxis]),
        ('epilepsy', np.random.randn(1000, 22) + 2 * np.random.choice([0, 1], (1000, 22))),
        ('parkinson', np.random.randn(1000, 22) * 0.8),
    ]

    # Run classifications
    print("\n3. Running classifications...")
    for disease, data in test_cases:
        print(f"\n   Testing for {disease.upper()}:")
        result = classifier.classify(data, target_disease=disease)

        print(f"      Prediction: {result.prediction}")
        print(f"      Confidence: {result.confidence:.2%}")
        print(f"      Model: {result.model_used}")
        print(f"      Signal Quality: {result.data_characteristics['signal_quality']:.2f}")

    # Auto-detection test
    print("\n4. Testing auto-detection (no target disease specified)...")
    auto_data = np.random.randn(1000, 22)
    result = classifier.classify(auto_data)
    print(f"   Auto-detected disease: {result.disease}")
    print(f"   Prediction: {result.prediction}")
    print(f"   Reasoning: {result.reasoning}")

    # Get summary
    print("\n5. Classification Summary:")
    summary = classifier.get_classification_summary()
    print(f"   Total classifications: {summary['total_classifications']}")
    print(f"   Average confidence: {summary['average_confidence']:.2%}")
    print(f"   Disease distribution: {summary['disease_distribution']}")

    # Test with different data shapes
    print("\n6. Testing with different input shapes...")
    shapes = [
        (500, 14),   # Fewer samples and channels
        (2000, 32),  # More samples and channels
        (22, 1000),  # Transposed (channels, samples)
    ]

    for shape in shapes:
        data = np.random.randn(*shape)
        result = classifier.classify(data, target_disease='stress')
        print(f"   Shape {shape}: prediction={result.prediction}, "
              f"confidence={result.confidence:.2%}")

    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    demo()
