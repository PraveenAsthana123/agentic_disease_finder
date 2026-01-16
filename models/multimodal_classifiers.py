"""
Multi-Modal Classifiers for Neurological Disease Detection
===========================================================
Supports three classification modes:
1. EEG Only - Sequential brain signal analysis
2. Image Only - MRI/CT scan analysis
3. Hybrid - Combined EEG + Image analysis

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
    logger.warning("PyTorch not available")


if HAS_TORCH:

    # =========================================================================
    # IMAGE CLASSIFIERS (MRI/CT)
    # =========================================================================

    class ImageClassifier2D(nn.Module):
        """
        2D CNN for brain image classification (MRI/CT slices).

        Architecture:
        - VGG-style convolutional blocks
        - Global average pooling
        - Fully connected classifier
        """

        def __init__(self, in_channels: int = 1, image_size: int = 128,
                     n_classes: int = 2, dropout: float = 0.5):
            super().__init__()

            self.in_channels = in_channels
            self.image_size = image_size
            self.n_classes = n_classes

            # Convolutional blocks
            self.conv_blocks = nn.Sequential(
                # Block 1: 128 -> 64
                self._make_conv_block(in_channels, 32, kernel_size=3),
                nn.MaxPool2d(2),

                # Block 2: 64 -> 32
                self._make_conv_block(32, 64, kernel_size=3),
                nn.MaxPool2d(2),

                # Block 3: 32 -> 16
                self._make_conv_block(64, 128, kernel_size=3),
                nn.MaxPool2d(2),

                # Block 4: 16 -> 8
                self._make_conv_block(128, 256, kernel_size=3),
                nn.MaxPool2d(2),

                # Block 5: Global pooling
                self._make_conv_block(256, 512, kernel_size=3),
                nn.AdaptiveAvgPool2d(1)
            )

            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )

        def _make_conv_block(self, in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Handle different input shapes
            if x.dim() == 3:
                x = x.unsqueeze(1)  # Add channel dim
            elif x.dim() == 4 and x.shape[-1] == 1:
                x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

            x = self.conv_blocks(x)
            x = self.classifier(x)
            return x

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            """Extract features before classifier"""
            if x.dim() == 3:
                x = x.unsqueeze(1)
            elif x.dim() == 4 and x.shape[-1] == 1:
                x = x.permute(0, 3, 1, 2)

            x = self.conv_blocks(x)
            return x.flatten(1)


    class ImageClassifier3D(nn.Module):
        """
        3D CNN for volumetric brain image classification.

        For full 3D MRI/CT volumes.
        """

        def __init__(self, in_channels: int = 1, volume_size: int = 64,
                     n_classes: int = 2, dropout: float = 0.5):
            super().__init__()

            self.conv_blocks = nn.Sequential(
                # Block 1
                nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d(2),

                # Block 2
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(2),

                # Block 3
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2),

                # Block 4
                nn.Conv3d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1)
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 4:
                x = x.unsqueeze(1)  # Add channel dim

            x = self.conv_blocks(x)
            x = self.classifier(x)
            return x


    # =========================================================================
    # EEG CLASSIFIER (From eeg_classifiers.py)
    # =========================================================================

    class EEGClassifier(nn.Module):
        """
        1D CNN for EEG classification.
        """

        def __init__(self, n_channels: int = 22, n_samples: int = 1000,
                     n_classes: int = 2, dropout: float = 0.5):
            super().__init__()

            self.n_channels = n_channels
            self.n_samples = n_samples

            # Spatial convolution
            self.spatial_conv = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )

            # Temporal convolutions
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=25, padding=12),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, kernel_size=15, padding=7),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(128, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )

            # Feature dimension
            self.feature_dim = 256

            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Handle input shapes
            if x.dim() == 2:
                x = x.unsqueeze(0)

            # Expect (batch, samples, channels) -> transpose to (batch, channels, samples)
            if x.shape[1] == self.n_samples and x.shape[2] == self.n_channels:
                x = x.transpose(1, 2)

            x = self.spatial_conv(x)
            x = self.temporal_conv(x)
            x = self.classifier(x)
            return x

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            """Extract features before classifier"""
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.shape[1] == self.n_samples and x.shape[2] == self.n_channels:
                x = x.transpose(1, 2)

            x = self.spatial_conv(x)
            x = self.temporal_conv(x)
            return x.flatten(1)


    # =========================================================================
    # HYBRID CLASSIFIER (EEG + Image)
    # =========================================================================

    class HybridClassifier(nn.Module):
        """
        Hybrid classifier combining EEG and Image features.

        Architecture:
        - EEG branch (1D CNN)
        - Image branch (2D CNN)
        - Feature fusion layer
        - Joint classifier
        """

        def __init__(self, n_eeg_channels: int = 22, n_eeg_samples: int = 1000,
                     image_size: int = 128, n_classes: int = 2,
                     dropout: float = 0.5, fusion_method: str = 'concat'):
            super().__init__()

            self.fusion_method = fusion_method

            # EEG branch
            self.eeg_branch = EEGClassifier(
                n_channels=n_eeg_channels,
                n_samples=n_eeg_samples,
                n_classes=n_classes,
                dropout=dropout
            )
            # Remove classifier from EEG branch
            self.eeg_branch.classifier = nn.Identity()
            self.eeg_feature_dim = 256

            # Image branch
            self.image_branch = ImageClassifier2D(
                in_channels=1,
                image_size=image_size,
                n_classes=n_classes,
                dropout=dropout
            )
            # Remove classifier from image branch
            self.image_branch.classifier = nn.Identity()
            self.image_feature_dim = 512

            # Fusion
            if fusion_method == 'concat':
                fusion_dim = self.eeg_feature_dim + self.image_feature_dim
            elif fusion_method == 'attention':
                fusion_dim = self.eeg_feature_dim + self.image_feature_dim
                self.attention = nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 4),
                    nn.ReLU(),
                    nn.Linear(fusion_dim // 4, fusion_dim),
                    nn.Sigmoid()
                )
            else:  # average
                fusion_dim = max(self.eeg_feature_dim, self.image_feature_dim)
                self.eeg_proj = nn.Linear(self.eeg_feature_dim, fusion_dim)
                self.image_proj = nn.Linear(self.image_feature_dim, fusion_dim)

            # Joint classifier
            self.fusion_classifier = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )

        def forward(self, eeg: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
            # Extract features
            eeg_features = self.eeg_branch.extract_features(eeg)
            image_features = self.image_branch.extract_features(image)

            # Fusion
            if self.fusion_method == 'concat':
                fused = torch.cat([eeg_features, image_features], dim=1)
            elif self.fusion_method == 'attention':
                concat = torch.cat([eeg_features, image_features], dim=1)
                attention_weights = self.attention(concat)
                fused = concat * attention_weights
            else:  # average
                eeg_proj = self.eeg_proj(eeg_features)
                image_proj = self.image_proj(image_features)
                fused = (eeg_proj + image_proj) / 2

            # Classify
            output = self.fusion_classifier(fused)
            return output


    # =========================================================================
    # DATASETS
    # =========================================================================

    class EEGDataset(Dataset):
        """Dataset for EEG data"""
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class ImageDataset(Dataset):
        """Dataset for image data"""
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class HybridDataset(Dataset):
        """Dataset for hybrid (EEG + Image) data"""
        def __init__(self, eeg: np.ndarray, images: np.ndarray, y: np.ndarray):
            self.eeg = torch.FloatTensor(eeg)
            self.images = torch.FloatTensor(images)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.eeg[idx], self.images[idx], self.y[idx]


    # =========================================================================
    # UNIFIED MULTI-MODAL CLASSIFIER
    # =========================================================================

    class MultiModalClassifier:
        """
        Unified multi-modal classifier supporting:
        - EEG only mode
        - Image only mode
        - Hybrid mode (EEG + Image)
        """

        def __init__(self, mode: str = 'eeg',
                     n_eeg_channels: int = 22,
                     n_eeg_samples: int = 1000,
                     image_size: int = 128,
                     n_classes: int = 2,
                     device: str = None):
            """
            Initialize multi-modal classifier.

            Args:
                mode: 'eeg', 'image', or 'hybrid'
                n_eeg_channels: Number of EEG channels
                n_eeg_samples: Number of EEG time samples
                image_size: Size of input images
                n_classes: Number of output classes
                device: Device for computation
            """
            self.mode = mode.lower()
            self.n_eeg_channels = n_eeg_channels
            self.n_eeg_samples = n_eeg_samples
            self.image_size = image_size
            self.n_classes = n_classes
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

            # Create appropriate model
            if self.mode == 'eeg':
                self.model = EEGClassifier(
                    n_channels=n_eeg_channels,
                    n_samples=n_eeg_samples,
                    n_classes=n_classes
                )
            elif self.mode == 'image':
                self.model = ImageClassifier2D(
                    in_channels=1,
                    image_size=image_size,
                    n_classes=n_classes
                )
            elif self.mode == 'hybrid':
                self.model = HybridClassifier(
                    n_eeg_channels=n_eeg_channels,
                    n_eeg_samples=n_eeg_samples,
                    image_size=image_size,
                    n_classes=n_classes
                )
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'eeg', 'image', or 'hybrid'")

            self.model.to(self.device)
            self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        def train(self, train_data: Tuple, val_data: Tuple = None,
                  epochs: int = 50, batch_size: int = 32, lr: float = 1e-3) -> Dict:
            """
            Train the model.

            Args:
                train_data: Tuple of (X, y) or (eeg, image, y) for hybrid
                val_data: Validation data (optional)
                epochs: Number of epochs
                batch_size: Batch size
                lr: Learning rate

            Returns:
                history: Training history
            """
            # Create datasets
            if self.mode == 'hybrid':
                train_dataset = HybridDataset(*train_data)
                val_dataset = HybridDataset(*val_data) if val_data else None
            elif self.mode == 'eeg':
                train_dataset = EEGDataset(*train_data)
                val_dataset = EEGDataset(*val_data) if val_data else None
            else:
                train_dataset = ImageDataset(*train_data)
                val_dataset = ImageDataset(*val_data) if val_data else None

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch in train_loader:
                    if self.mode == 'hybrid':
                        eeg, img, labels = [b.to(self.device) for b in batch]
                        outputs = self.model(eeg, img)
                    else:
                        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                        outputs = self.model(inputs)

                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                train_loss /= len(train_loader)
                train_acc = train_correct / train_total

                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)

                # Validation
                if val_loader:
                    val_loss, val_acc = self._evaluate(val_loader, criterion)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)

                    if (epoch + 1) % 10 == 0:
                        print(f"Epoch {epoch + 1}/{epochs}: "
                              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            return self.history

        def _evaluate(self, data_loader: DataLoader,
                      criterion: nn.Module) -> Tuple[float, float]:
            """Evaluate model"""
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in data_loader:
                    if self.mode == 'hybrid':
                        eeg, img, labels = [b.to(self.device) for b in batch]
                        outputs = self.model(eeg, img)
                    else:
                        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                        outputs = self.model(inputs)

                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            return total_loss / len(data_loader), correct / total

        def predict(self, data: Tuple) -> np.ndarray:
            """Make predictions"""
            self.model.eval()

            with torch.no_grad():
                if self.mode == 'hybrid':
                    eeg = torch.FloatTensor(data[0]).to(self.device)
                    img = torch.FloatTensor(data[1]).to(self.device)
                    outputs = self.model(eeg, img)
                else:
                    inputs = torch.FloatTensor(data).to(self.device)
                    outputs = self.model(inputs)

                _, predicted = outputs.max(1)

            return predicted.cpu().numpy()

        def predict_proba(self, data: Tuple) -> np.ndarray:
            """Get prediction probabilities"""
            self.model.eval()

            with torch.no_grad():
                if self.mode == 'hybrid':
                    eeg = torch.FloatTensor(data[0]).to(self.device)
                    img = torch.FloatTensor(data[1]).to(self.device)
                    outputs = self.model(eeg, img)
                else:
                    inputs = torch.FloatTensor(data).to(self.device)
                    outputs = self.model(inputs)

                proba = F.softmax(outputs, dim=1)

            return proba.cpu().numpy()

        def save(self, path: str):
            """Save model"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'mode': self.mode,
                'history': self.history,
                'config': {
                    'n_eeg_channels': self.n_eeg_channels,
                    'n_eeg_samples': self.n_eeg_samples,
                    'image_size': self.image_size,
                    'n_classes': self.n_classes
                }
            }, path)

        def load(self, path: str):
            """Load model"""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint.get('history', self.history)


# Fallback for when PyTorch is not available
else:
    class MultiModalClassifier:
        def __init__(self, **kwargs):
            raise ImportError("PyTorch is required for multi-modal classification")


# =============================================================================
# DISEASE CONFIGURATIONS
# =============================================================================

DISEASE_CONFIG = {
    'alzheimer': {
        'n_classes': 3,
        'class_names': ['Healthy', 'MCI', 'Alzheimer'],
        'modalities': ['eeg', 'image', 'hybrid']
    },
    'parkinson': {
        'n_classes': 2,
        'class_names': ['Healthy', 'Parkinson'],
        'modalities': ['eeg', 'image', 'hybrid']
    },
    'schizophrenia': {
        'n_classes': 2,
        'class_names': ['Healthy', 'Schizophrenia'],
        'modalities': ['eeg', 'image', 'hybrid']
    },
    'epilepsy': {
        'n_classes': 2,
        'class_names': ['Non-Seizure', 'Seizure'],
        'modalities': ['eeg', 'image']
    },
    'autism': {
        'n_classes': 2,
        'class_names': ['Neurotypical', 'ASD'],
        'modalities': ['eeg', 'hybrid']
    },
    'stress': {
        'n_classes': 2,
        'class_names': ['Relaxed', 'Stressed'],
        'modalities': ['eeg']
    },
    'depression': {
        'n_classes': 2,
        'class_names': ['Healthy', 'Depression'],
        'modalities': ['eeg', 'hybrid']
    }
}


def create_classifier(disease: str, mode: str = 'eeg', **kwargs) -> 'MultiModalClassifier':
    """
    Create a classifier for a specific disease and modality.

    Args:
        disease: Disease name
        mode: 'eeg', 'image', or 'hybrid'
        **kwargs: Additional model parameters

    Returns:
        classifier: Configured MultiModalClassifier
    """
    config = DISEASE_CONFIG.get(disease.lower(), {'n_classes': 2})

    return MultiModalClassifier(
        mode=mode,
        n_classes=config['n_classes'],
        **kwargs
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  MULTI-MODAL CLASSIFIER TEST")
    print("=" * 70)

    if not HAS_TORCH:
        print("PyTorch not available!")
        exit(1)

    import sys
    sys.path.insert(0, '..')
    from data.multimodal_loader import MultiModalDataLoader, DiseaseLabel

    # Load test data
    loader = MultiModalDataLoader()

    # Test 1: EEG only mode
    print("\n1. Testing EEG-only classifier...")
    eeg_data, labels = loader.get_eeg_data(
        diseases=[DiseaseLabel.DEPRESSION],
        n_samples=100
    )
    print(f"   EEG data: {eeg_data.shape}, Labels: {np.bincount(labels)}")

    classifier_eeg = MultiModalClassifier(mode='eeg', n_classes=2)
    n_train = int(len(eeg_data) * 0.8)
    history = classifier_eeg.train(
        train_data=(eeg_data[:n_train], labels[:n_train]),
        val_data=(eeg_data[n_train:], labels[n_train:]),
        epochs=10,
        batch_size=16
    )
    print(f"   Final val accuracy: {history['val_acc'][-1]:.2%}")

    # Test 2: Image only mode
    print("\n2. Testing Image-only classifier...")
    img_data, labels = loader.get_image_data(
        diseases=[DiseaseLabel.ALZHEIMER],
        n_samples=100
    )
    print(f"   Image data: {img_data.shape}, Labels: {np.bincount(labels)}")

    classifier_img = MultiModalClassifier(mode='image', n_classes=2)
    history = classifier_img.train(
        train_data=(img_data[:n_train], labels[:n_train]),
        val_data=(img_data[n_train:], labels[n_train:]),
        epochs=10,
        batch_size=16
    )
    print(f"   Final val accuracy: {history['val_acc'][-1]:.2%}")

    # Test 3: Hybrid mode
    print("\n3. Testing Hybrid classifier...")
    eeg, mri, labels = loader.get_hybrid_data(
        diseases=[DiseaseLabel.PARKINSON],
        n_samples=100
    )
    print(f"   EEG: {eeg.shape}, MRI: {mri.shape}, Labels: {np.bincount(labels)}")

    classifier_hybrid = MultiModalClassifier(mode='hybrid', n_classes=2)
    history = classifier_hybrid.train(
        train_data=(eeg[:n_train], mri[:n_train], labels[:n_train]),
        val_data=(eeg[n_train:], mri[n_train:], labels[n_train:]),
        epochs=10,
        batch_size=16
    )
    print(f"   Final val accuracy: {history['val_acc'][-1]:.2%}")

    print("\n" + "=" * 70)
    print("  TEST COMPLETE!")
    print("=" * 70)
