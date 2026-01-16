"""
Deep Learning Models Module
"""

from .deep_learning_models import (
    ModelFactory,
    ModelTrainer
)

# Inference models
from .inference import (
    AlzheimerInferenceModel,
    ParkinsonInferenceModel,
    SchizophreniaInferenceModel,
    MultiDiseaseInferenceEngine,
    PredictionResult
)

# EEG Classifiers for Agentic Disease Detection
from .eeg_classifiers import (
    EEGClassifierFactory,
    EEGClassifierTrainer,
    EEGDataset,
    DISEASE_MODELS,
    get_model_for_disease
)

# Conditional imports based on available frameworks
try:
    from .deep_learning_models import (
        AlzheimerCNN3D,
        AlzheimerTransformer,
        ParkinsonVoiceLSTM,
        ParkinsonGaitCNN,
        SchizophreniaEEGNet,
        SchizophreniaGraphNet,
        MultiDiseaseEnsemble,
        NeuroDiseaseDataset,
        AttentionBlock
    )
    # Aliases for compatibility
    ParkinsonLSTM = ParkinsonVoiceLSTM
except ImportError:
    pass

try:
    from .eeg_classifiers import (
        EEGClassifier1D,
        EEGClassifierMultiScale,
        EEGClassifierWithAttention
    )
except ImportError:
    pass

__all__ = [
    'ModelFactory',
    'ModelTrainer',
    # Inference
    'AlzheimerInferenceModel',
    'ParkinsonInferenceModel',
    'SchizophreniaInferenceModel',
    'MultiDiseaseInferenceEngine',
    'PredictionResult',
    # Models
    'AlzheimerCNN3D',
    'ParkinsonLSTM',
    'ParkinsonVoiceLSTM',
    'SchizophreniaEEGNet',
    # EEG Classifiers
    'EEGClassifierFactory',
    'EEGClassifierTrainer',
    'EEGDataset',
    'EEGClassifier1D',
    'get_model_for_disease',
    'DISEASE_MODELS'
]
