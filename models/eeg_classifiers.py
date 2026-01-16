"""
EEG Classifiers for Neurological Disease Detection
===================================================
Implements 1D CNN architectures for EEG-based disease classification
as described in the Agentic Disease Finder Research Paper.

Supported Diseases:
- Alzheimer's Disease
- Parkinson's Disease
- Schizophrenia
- Epilepsy
- Autism Spectrum Disorder
- Stress
- Depression

Input Shape: (samples, channels) typically (1000, 22)

Author: Research Team
Project: Neurological Disease Detection using Agentic AI
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, using numpy fallback")


if HAS_TORCH:

    class EEGClassifier1D(nn.Module):
        """
        1D CNN for EEG Classification

        Architecture based on the research paper with:
        - Multiple 1D convolutional blocks
        - Batch normalization
        - Dropout for regularization
        - Global average pooling
        - Fully connected classification head

        Parameters
        ----------
        n_channels : int
            Number of EEG channels (default: 22)
        n_samples : int
            Number of time samples (default: 1000)
        n_classes : int
            Number of output classes (default: 2)
        dropout : float
            Dropout rate (default: 0.5)
        """

        def __init__(self, n_channels: int = 22, n_samples: int = 1000,
                     n_classes: int = 2, dropout: float = 0.5):
            super().__init__()

            self.n_channels = n_channels
            self.n_samples = n_samples
            self.n_classes = n_classes

            # Spatial convolution to learn channel interactions
            self.spatial_conv = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )

            # Temporal convolution blocks
            self.temporal_blocks = nn.Sequential(
                # Block 1: Large receptive field for low-frequency features
                self._make_conv_block(64, 64, kernel_size=25, padding=12),
                nn.MaxPool1d(2),

                # Block 2
                self._make_conv_block(64, 128, kernel_size=15, padding=7),
                nn.MaxPool1d(2),

                # Block 3
                self._make_conv_block(128, 128, kernel_size=11, padding=5),
                nn.MaxPool1d(2),

                # Block 4
                self._make_conv_block(128, 256, kernel_size=7, padding=3),
                nn.MaxPool1d(2),

                # Block 5
                self._make_conv_block(256, 256, kernel_size=5, padding=2),
                nn.AdaptiveAvgPool1d(1)
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes)
            )

        def _make_conv_block(self, in_channels: int, out_channels: int,
                             kernel_size: int, padding: int) -> nn.Sequential:
            """Create a convolutional block"""
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                         padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                         padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch, samples, channels) or (batch, channels, samples)

            Returns
            -------
            output : torch.Tensor
                Class logits of shape (batch, n_classes)
            """
            # Ensure shape is (batch, channels, samples)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            if x.shape[1] == self.n_samples and x.shape[2] == self.n_channels:
                # Input is (batch, samples, channels), transpose
                x = x.transpose(1, 2)

            # Spatial convolution
            x = self.spatial_conv(x)

            # Temporal convolution
            x = self.temporal_blocks(x)

            # Classification
            x = self.classifier(x)

            return x


    class EEGClassifierMultiScale(nn.Module):
        """
        Multi-scale 1D CNN for EEG Classification

        Uses parallel convolutions with different kernel sizes to capture
        features at multiple temporal scales.
        """

        def __init__(self, n_channels: int = 22, n_samples: int = 1000,
                     n_classes: int = 2, dropout: float = 0.5):
            super().__init__()

            self.n_channels = n_channels
            self.n_samples = n_samples

            # Spatial convolution
            self.spatial_conv = nn.Conv1d(n_channels, 64, kernel_size=1)
            self.spatial_bn = nn.BatchNorm1d(64)

            # Multi-scale temporal convolutions
            self.scale1 = self._make_scale_block(64, 32, kernel_size=5)   # Fine detail
            self.scale2 = self._make_scale_block(64, 32, kernel_size=15)  # Medium
            self.scale3 = self._make_scale_block(64, 32, kernel_size=25)  # Coarse
            self.scale4 = self._make_scale_block(64, 32, kernel_size=51)  # Very coarse

            # Merge scales
            self.merge = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(4),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )

            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )

        def _make_scale_block(self, in_channels: int, out_channels: int,
                              kernel_size: int) -> nn.Sequential:
            """Create a scale-specific convolution block"""
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                         padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Handle input shape
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.shape[1] == self.n_samples and x.shape[2] == self.n_channels:
                x = x.transpose(1, 2)

            # Spatial convolution
            x = F.relu(self.spatial_bn(self.spatial_conv(x)))

            # Multi-scale convolutions
            s1 = self.scale1(x)
            s2 = self.scale2(x)
            s3 = self.scale3(x)
            s4 = self.scale4(x)

            # Concatenate scales
            x = torch.cat([s1, s2, s3, s4], dim=1)

            # Merge and classify
            x = self.merge(x)
            x = self.classifier(x)

            return x


    class EEGClassifierWithAttention(nn.Module):
        """
        1D CNN with Self-Attention for EEG Classification

        Combines convolutional feature extraction with attention
        mechanism for focusing on relevant temporal regions.
        """

        def __init__(self, n_channels: int = 22, n_samples: int = 1000,
                     n_classes: int = 2, dropout: float = 0.5,
                     n_heads: int = 4):
            super().__init__()

            self.n_channels = n_channels
            self.n_samples = n_samples

            # Spatial + initial temporal processing
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Conv1d(64, 64, kernel_size=15, padding=7),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, kernel_size=11, padding=5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )

            # Self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=128,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(128)

            # Post-attention processing
            self.decoder = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )

            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Handle input shape
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.shape[1] == self.n_samples and x.shape[2] == self.n_channels:
                x = x.transpose(1, 2)

            # Encode
            x = self.encoder(x)  # (B, 128, T')

            # Self-attention (need sequence format: B, T, C)
            x = x.transpose(1, 2)  # (B, T', 128)
            attn_out, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_out)
            x = x.transpose(1, 2)  # Back to (B, 128, T')

            # Decode and classify
            x = self.decoder(x)
            x = self.classifier(x)

            return x


    class EEGDataset(Dataset):
        """Dataset class for EEG data"""

        def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
            """
            Parameters
            ----------
            X : np.ndarray
                EEG data of shape (n_samples, timesteps, channels)
            y : np.ndarray
                Labels
            transform : callable, optional
                Optional transform to apply
            """
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
            self.transform = transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            x = self.X[idx]
            y = self.y[idx]

            if self.transform:
                x = self.transform(x)

            return x, y


    class EEGClassifierFactory:
        """Factory for creating EEG classifier models"""

        MODELS = {
            'basic': EEGClassifier1D,
            'multiscale': EEGClassifierMultiScale,
            'attention': EEGClassifierWithAttention
        }

        @classmethod
        def create(cls, model_type: str = 'basic',
                   n_channels: int = 22,
                   n_samples: int = 1000,
                   n_classes: int = 2,
                   **kwargs) -> nn.Module:
            """
            Create an EEG classifier model

            Parameters
            ----------
            model_type : str
                Type of model ('basic', 'multiscale', 'attention')
            n_channels : int
                Number of EEG channels
            n_samples : int
                Number of time samples
            n_classes : int
                Number of output classes
            **kwargs
                Additional model parameters

            Returns
            -------
            model : nn.Module
                The created model
            """
            if model_type not in cls.MODELS:
                raise ValueError(f"Unknown model type: {model_type}. "
                               f"Available: {list(cls.MODELS.keys())}")

            model_class = cls.MODELS[model_type]
            return model_class(n_channels=n_channels, n_samples=n_samples,
                             n_classes=n_classes, **kwargs)

        @classmethod
        def create_for_disease(cls, disease: str,
                               n_channels: int = 22,
                               n_samples: int = 1000,
                               model_type: str = 'basic') -> nn.Module:
            """
            Create a model configured for a specific disease

            Parameters
            ----------
            disease : str
                Target disease name
            n_channels : int
                Number of EEG channels
            n_samples : int
                Number of time samples
            model_type : str
                Type of model architecture

            Returns
            -------
            model : nn.Module
                Configured model
            """
            # Disease-specific configurations
            disease_config = {
                'alzheimer': {'n_classes': 3},  # CN, MCI, AD
                'parkinson': {'n_classes': 2},
                'schizophrenia': {'n_classes': 2},
                'epilepsy': {'n_classes': 2},
                'autism': {'n_classes': 2},
                'stress': {'n_classes': 2},
                'depression': {'n_classes': 2}
            }

            disease_lower = disease.lower()
            if disease_lower not in disease_config:
                logger.warning(f"Unknown disease: {disease}, using default config")
                config = {'n_classes': 2}
            else:
                config = disease_config[disease_lower]

            return cls.create(
                model_type=model_type,
                n_channels=n_channels,
                n_samples=n_samples,
                **config
            )

        @classmethod
        def list_models(cls) -> List[str]:
            """List available model types"""
            return list(cls.MODELS.keys())


    class EEGClassifierTrainer:
        """
        Trainer for EEG Classifier models

        Handles training, validation, and evaluation with:
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        """

        def __init__(self, model: nn.Module, device: str = None):
            self.model = model
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            self.history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }

            self.best_val_acc = 0.0
            self.best_model_state = None

        def train(self, train_loader: DataLoader,
                  val_loader: Optional[DataLoader] = None,
                  epochs: int = 100,
                  lr: float = 1e-3,
                  weight_decay: float = 1e-4,
                  patience: int = 15,
                  verbose: bool = True) -> Dict:
            """
            Train the model

            Parameters
            ----------
            train_loader : DataLoader
                Training data loader
            val_loader : DataLoader, optional
                Validation data loader
            epochs : int
                Number of training epochs
            lr : float
                Initial learning rate
            weight_decay : float
                L2 regularization weight
            patience : int
                Early stopping patience
            verbose : bool
                Print progress

            Returns
            -------
            history : dict
                Training history
            """
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )

            criterion = nn.CrossEntropyLoss()

            patience_counter = 0

            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()

                train_loss /= len(train_loader)
                train_acc = train_correct / train_total

                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)

                # Validation phase
                if val_loader:
                    val_loss, val_acc = self.evaluate(val_loader)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)

                    scheduler.step(val_acc)

                    # Early stopping and checkpointing
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break

                    if verbose and (epoch + 1) % 10 == 0:
                        print(f"Epoch {epoch + 1}/{epochs}: "
                              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                else:
                    if verbose and (epoch + 1) % 10 == 0:
                        print(f"Epoch {epoch + 1}/{epochs}: "
                              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

            # Load best model
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state)

            return self.history

        def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
            """Evaluate model on data"""
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for batch_x, batch_y in data_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()

            return total_loss / len(data_loader), correct / total

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions"""
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = outputs.max(1)

            return predicted.cpu().numpy()

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Get prediction probabilities"""
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                proba = F.softmax(outputs, dim=1)

            return proba.cpu().numpy()

        def save(self, path: str):
            """Save model checkpoint"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'history': self.history,
                'best_val_acc': self.best_val_acc
            }, path)
            logger.info(f"Model saved to {path}")

        def load(self, path: str):
            """Load model checkpoint"""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint.get('history', self.history)
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            logger.info(f"Model loaded from {path}")


else:
    # Numpy fallback implementation
    class EEGClassifier1D:
        """Simple EEG classifier using numpy (fallback when PyTorch unavailable)"""

        def __init__(self, n_channels: int = 22, n_samples: int = 1000,
                     n_classes: int = 2, **kwargs):
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.n_classes = n_classes
            self.weights = None
            self.trained = False

        def fit(self, X: np.ndarray, y: np.ndarray):
            """Train using simple feature extraction + logistic regression"""
            # Extract simple features
            features = self._extract_features(X)

            # Simple logistic regression-like weights
            n_features = features.shape[1]
            self.weights = np.random.randn(n_features, self.n_classes) * 0.01
            self.bias = np.zeros(self.n_classes)

            # Simple gradient descent
            lr = 0.01
            for _ in range(100):
                logits = features @ self.weights + self.bias
                probs = self._softmax(logits)

                # One-hot encode y
                y_onehot = np.eye(self.n_classes)[y]

                # Gradient
                grad = features.T @ (probs - y_onehot) / len(y)
                self.weights -= lr * grad
                self.bias -= lr * np.mean(probs - y_onehot, axis=0)

            self.trained = True

        def predict(self, X: np.ndarray) -> np.ndarray:
            features = self._extract_features(X)
            logits = features @ self.weights + self.bias
            return np.argmax(logits, axis=1)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            features = self._extract_features(X)
            logits = features @ self.weights + self.bias
            return self._softmax(logits)

        def _extract_features(self, X: np.ndarray) -> np.ndarray:
            """Extract simple statistical features"""
            # Ensure shape (n_samples, timesteps, channels)
            if X.ndim == 2:
                X = X.reshape(1, *X.shape)

            features = []
            for sample in X:
                f = []
                # Mean per channel
                f.extend(np.mean(sample, axis=0))
                # Std per channel
                f.extend(np.std(sample, axis=0))
                # Max per channel
                f.extend(np.max(sample, axis=0))
                # Min per channel
                f.extend(np.min(sample, axis=0))
                features.append(f)

            return np.array(features)

        def _softmax(self, x: np.ndarray) -> np.ndarray:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    class EEGClassifierFactory:
        """Factory for numpy-based classifiers"""

        @classmethod
        def create(cls, **kwargs):
            return EEGClassifier1D(**kwargs)

        @classmethod
        def create_for_disease(cls, disease: str, **kwargs):
            return EEGClassifier1D(**kwargs)

        @classmethod
        def list_models(cls):
            return ['basic']

    class EEGDataset:
        """Simple dataset wrapper"""
        def __init__(self, X, y, transform=None):
            self.X = X
            self.y = y

    class EEGClassifierTrainer:
        """Simple trainer for numpy classifier"""
        def __init__(self, model, **kwargs):
            self.model = model

        def train(self, X, y, **kwargs):
            self.model.fit(X, y)
            return {}

        def predict(self, X):
            return self.model.predict(X)


# Disease-specific model configurations
DISEASE_MODELS = {
    'alzheimer': {
        'n_classes': 3,
        'class_names': ['Cognitively Normal', 'Mild Cognitive Impairment', 'Alzheimer\'s Disease'],
        'model_type': 'attention'
    },
    'parkinson': {
        'n_classes': 2,
        'class_names': ['Healthy', 'Parkinson\'s Disease'],
        'model_type': 'multiscale'
    },
    'schizophrenia': {
        'n_classes': 2,
        'class_names': ['Healthy', 'Schizophrenia'],
        'model_type': 'attention'
    },
    'epilepsy': {
        'n_classes': 2,
        'class_names': ['Non-Seizure', 'Seizure'],
        'model_type': 'basic'
    },
    'autism': {
        'n_classes': 2,
        'class_names': ['Neurotypical', 'Autism Spectrum Disorder'],
        'model_type': 'multiscale'
    },
    'stress': {
        'n_classes': 2,
        'class_names': ['Relaxed', 'Stressed'],
        'model_type': 'basic'
    },
    'depression': {
        'n_classes': 2,
        'class_names': ['Healthy', 'Depression'],
        'model_type': 'attention'
    }
}


def get_model_for_disease(disease: str,
                          n_channels: int = 22,
                          n_samples: int = 1000) -> Any:
    """
    Get the appropriate model for a specific disease

    Parameters
    ----------
    disease : str
        Disease name
    n_channels : int
        Number of EEG channels
    n_samples : int
        Number of time samples

    Returns
    -------
    model : nn.Module or EEGClassifier1D
        Configured model for the disease
    """
    disease_lower = disease.lower()
    if disease_lower not in DISEASE_MODELS:
        logger.warning(f"Unknown disease: {disease}, using default config")
        config = {'n_classes': 2, 'model_type': 'basic'}
    else:
        config = DISEASE_MODELS[disease_lower]

    return EEGClassifierFactory.create(
        model_type=config.get('model_type', 'basic'),
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=config['n_classes']
    )


if __name__ == "__main__":
    print("=" * 70)
    print("  EEG CLASSIFIER DEMO")
    print("=" * 70)

    print(f"\nPyTorch available: {HAS_TORCH}")
    print(f"Available model types: {EEGClassifierFactory.list_models()}")

    # Test model creation
    print("\n1. Creating models for different diseases...")
    for disease in ['alzheimer', 'parkinson', 'autism', 'stress']:
        model = get_model_for_disease(disease)
        if HAS_TORCH:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"   {disease.capitalize()}: {n_params:,} parameters")
        else:
            print(f"   {disease.capitalize()}: numpy fallback model")

    if HAS_TORCH:
        # Test forward pass
        print("\n2. Testing forward pass...")
        model = EEGClassifierFactory.create('basic', n_channels=22, n_samples=1000, n_classes=2)

        # Test different input shapes
        test_inputs = [
            torch.randn(4, 1000, 22),  # (batch, samples, channels)
            torch.randn(4, 22, 1000),  # (batch, channels, samples)
        ]

        for i, x in enumerate(test_inputs):
            output = model(x)
            print(f"   Input {x.shape} -> Output {output.shape}")

        # Test multi-scale model
        print("\n3. Testing multi-scale model...")
        model_ms = EEGClassifierFactory.create('multiscale')
        x = torch.randn(4, 1000, 22)
        output = model_ms(x)
        print(f"   Input {x.shape} -> Output {output.shape}")

        # Test attention model
        print("\n4. Testing attention model...")
        model_att = EEGClassifierFactory.create('attention')
        x = torch.randn(4, 1000, 22)
        output = model_att(x)
        print(f"   Input {x.shape} -> Output {output.shape}")

        # Test training loop
        print("\n5. Testing training loop...")
        model = EEGClassifierFactory.create('basic', n_classes=2)
        trainer = EEGClassifierTrainer(model)

        # Create dummy data
        X_train = np.random.randn(100, 1000, 22).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 1000, 22).astype(np.float32)
        y_val = np.random.randint(0, 2, 20)

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        history = trainer.train(train_loader, val_loader, epochs=20, verbose=False)
        print(f"   Training complete. Best val acc: {trainer.best_val_acc:.4f}")

        # Test prediction
        print("\n6. Testing prediction...")
        X_test = np.random.randn(5, 1000, 22).astype(np.float32)
        predictions = trainer.predict(X_test)
        probabilities = trainer.predict_proba(X_test)
        print(f"   Predictions: {predictions}")
        print(f"   Probabilities shape: {probabilities.shape}")

    else:
        print("\n2. Testing numpy fallback...")
        model = EEGClassifier1D()
        X = np.random.randn(50, 1000, 22)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        predictions = model.predict(X[:5])
        print(f"   Predictions: {predictions}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
