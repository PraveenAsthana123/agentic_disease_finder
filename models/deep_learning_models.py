"""
Deep Learning Models for Neurological Disease Detection
========================================================
Implements disease-specific deep learning architectures for
Alzheimer's, Parkinson's, and Schizophrenia detection.

Author: Research Team
Project: Neurological Disease Detection using Agentic AI
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Try importing deep learning frameworks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not available")


# ============================================================================
# PyTorch Models
# ============================================================================

if HAS_TORCH:

    class AttentionBlock(nn.Module):
        """Multi-head self-attention block"""

        def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            # Self-attention with residual
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # Feed-forward with residual
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            return x


    class AlzheimerCNN3D(nn.Module):
        """
        3D CNN for Alzheimer's Disease Detection from MRI

        Architecture based on VGG-style with 3D convolutions
        for processing volumetric brain MRI data.
        """

        def __init__(self, num_classes: int = 3, in_channels: int = 1,
                     dropout: float = 0.5):
            super().__init__()

            self.features = nn.Sequential(
                # Block 1
                nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                # Block 2
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                # Block 3
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.Conv3d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                # Block 4
                nn.Conv3d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((2, 2, 2))
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 8, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x


    class AlzheimerTransformer(nn.Module):
        """
        Vision Transformer for Alzheimer's Detection

        Processes 3D MRI as sequence of patches using transformer architecture.
        """

        def __init__(self, img_size: int = 64, patch_size: int = 8,
                     in_channels: int = 1, num_classes: int = 3,
                     embed_dim: int = 256, depth: int = 6, num_heads: int = 8,
                     dropout: float = 0.1):
            super().__init__()

            self.patch_size = patch_size
            num_patches = (img_size // patch_size) ** 3

            # Patch embedding
            self.patch_embed = nn.Conv3d(
                in_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size
            )

            # Positional embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            # Transformer blocks
            self.blocks = nn.ModuleList([
                AttentionBlock(embed_dim, num_heads, dropout)
                for _ in range(depth)
            ])

            self.norm = nn.LayerNorm(embed_dim)

            # Classification head
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes)
            )

            # Initialize weights
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def forward(self, x):
            B = x.shape[0]

            # Patch embedding
            x = self.patch_embed(x)  # (B, embed_dim, H', W', D')
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

            # Add cls token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Add positional embedding
            x = x + self.pos_embed[:, :x.size(1)]

            # Transformer blocks
            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            # Classification using cls token
            x = x[:, 0]
            x = self.head(x)

            return x


    class ParkinsonVoiceLSTM(nn.Module):
        """
        LSTM Network for Parkinson's Detection from Voice

        Processes voice features sequences for PD classification.
        """

        def __init__(self, input_size: int = 26, hidden_size: int = 128,
                     num_layers: int = 2, num_classes: int = 2,
                     dropout: float = 0.3, bidirectional: bool = True):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            # LSTM forward
            lstm_out, _ = self.lstm(x)  # (B, seq_len, hidden*directions)

            # Attention mechanism
            attn_weights = self.attention(lstm_out)
            attn_weights = F.softmax(attn_weights, dim=1)

            # Weighted sum
            context = torch.sum(attn_weights * lstm_out, dim=1)

            # Classification
            out = self.classifier(context)

            return out


    class ParkinsonGaitCNN(nn.Module):
        """
        1D CNN for Parkinson's Detection from Gait Sensor Data

        Processes accelerometer/gyroscope time series.
        """

        def __init__(self, in_channels: int = 6, seq_length: int = 1000,
                     num_classes: int = 2, dropout: float = 0.4):
            super().__init__()

            self.conv_blocks = nn.Sequential(
                # Block 1
                nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Block 2
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Block 3
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Block 4
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.conv_blocks(x)
            x = self.classifier(x)
            return x


    class SchizophreniaEEGNet(nn.Module):
        """
        EEGNet-based architecture for Schizophrenia Detection

        Specialized for processing multi-channel EEG data.
        """

        def __init__(self, n_channels: int = 64, n_samples: int = 256,
                     num_classes: int = 2, dropout: float = 0.5,
                     F1: int = 8, D: int = 2, F2: int = 16):
            super().__init__()

            # Temporal convolution
            self.temporal_conv = nn.Sequential(
                nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(F1)
            )

            # Depthwise convolution (spatial)
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout)
            )

            # Separable convolution
            self.separable_conv = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout)
            )

            # Calculate output size
            self._to_linear = None
            self._get_conv_output((1, 1, n_channels, n_samples))

            self.classifier = nn.Linear(self._to_linear, num_classes)

        def _get_conv_output(self, shape):
            x = torch.zeros(shape)
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            self._to_linear = x.numel()

        def forward(self, x):
            # Add channel dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(1)

            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            x = x.flatten(1)
            x = self.classifier(x)

            return x


    class SchizophreniaGraphNet(nn.Module):
        """
        Graph Neural Network for Schizophrenia Detection

        Processes functional connectivity matrices using graph convolutions.
        """

        def __init__(self, n_nodes: int = 116, n_features: int = 1,
                     hidden_dim: int = 64, num_classes: int = 2,
                     dropout: float = 0.3):
            super().__init__()

            self.n_nodes = n_nodes

            # Graph convolution layers (simplified without PyG)
            self.gc1 = nn.Linear(n_features, hidden_dim)
            self.gc2 = nn.Linear(hidden_dim, hidden_dim)
            self.gc3 = nn.Linear(hidden_dim, hidden_dim // 2)

            self.bn1 = nn.BatchNorm1d(n_nodes)
            self.bn2 = nn.BatchNorm1d(n_nodes)

            # Global pooling and classification
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2 * n_nodes, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )

        def forward(self, x, adj=None):
            """
            Parameters
            ----------
            x : tensor
                Node features (B, N, F) or connectivity matrix (B, N, N)
            adj : tensor
                Adjacency matrix (B, N, N), optional if x is connectivity
            """
            B = x.shape[0]

            # If input is connectivity matrix, use it as adj and create node features
            if adj is None and x.shape[-1] == x.shape[-2]:
                adj = x
                x = torch.ones(B, self.n_nodes, 1, device=x.device)

            # Simple graph convolution: H' = A * H * W
            x = torch.bmm(adj, self.gc1(x))
            x = self.bn1(x)
            x = F.relu(x)

            x = torch.bmm(adj, self.gc2(x))
            x = self.bn2(x)
            x = F.relu(x)

            x = torch.bmm(adj, self.gc3(x))
            x = F.relu(x)

            # Flatten and classify
            x = x.flatten(1)
            x = self.classifier(x)

            return x


    class MultiDiseaseEnsemble(nn.Module):
        """
        Ensemble Model for Multi-Disease Detection

        Combines predictions from disease-specific models.
        """

        def __init__(self, alzheimer_model: nn.Module,
                     parkinson_model: nn.Module,
                     schizophrenia_model: nn.Module,
                     fusion_dim: int = 128):
            super().__init__()

            self.alzheimer_model = alzheimer_model
            self.parkinson_model = parkinson_model
            self.schizophrenia_model = schizophrenia_model

            # Freeze base models
            for model in [alzheimer_model, parkinson_model, schizophrenia_model]:
                for param in model.parameters():
                    param.requires_grad = False

            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(3, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fusion_dim, 4)  # 4 classes: AD, PD, SZ, Healthy
            )

        def forward(self, ad_input, pd_input, sz_input):
            # Get predictions from each model
            ad_logits = self.alzheimer_model(ad_input)
            pd_logits = self.parkinson_model(pd_input)
            sz_logits = self.schizophrenia_model(sz_input)

            # Softmax for probabilities
            ad_prob = F.softmax(ad_logits, dim=1)[:, -1:]  # AD probability
            pd_prob = F.softmax(pd_logits, dim=1)[:, -1:]  # PD probability
            sz_prob = F.softmax(sz_logits, dim=1)[:, -1:]  # SZ probability

            # Concatenate and fuse
            combined = torch.cat([ad_prob, pd_prob, sz_prob], dim=1)
            output = self.fusion(combined)

            return output


    class NeuroDiseaseDataset(Dataset):
        """PyTorch Dataset for neurological disease data"""

        def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
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


# ============================================================================
# TensorFlow/Keras Models
# ============================================================================

if HAS_TF:

    def create_alzheimer_cnn3d_keras(input_shape: Tuple = (64, 64, 64, 1),
                                     num_classes: int = 3,
                                     dropout: float = 0.5) -> Model:
        """
        Create 3D CNN for Alzheimer's detection using Keras
        """
        inputs = keras.Input(shape=input_shape)

        # Block 1
        x = layers.Conv3D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(32, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(2)(x)

        # Block 2
        x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(2)(x)

        # Block 3
        x = layers.Conv3D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(2)(x)

        # Block 4
        x = layers.Conv3D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling3D()(x)

        # Classifier
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs, name='AlzheimerCNN3D')
        return model


    def create_parkinson_lstm_keras(input_shape: Tuple = (100, 26),
                                    num_classes: int = 2,
                                    dropout: float = 0.3) -> Model:
        """
        Create LSTM for Parkinson's detection using Keras
        """
        inputs = keras.Input(shape=input_shape)

        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(dropout)(x)

        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)

        # Apply attention
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)

        # Classifier
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs, name='ParkinsonLSTM')
        return model


    def create_schizophrenia_eegnet_keras(n_channels: int = 64,
                                          n_samples: int = 256,
                                          num_classes: int = 2,
                                          dropout: float = 0.5) -> Model:
        """
        Create EEGNet for Schizophrenia detection using Keras
        """
        F1, D, F2 = 8, 2, 16

        inputs = keras.Input(shape=(n_channels, n_samples, 1))

        # Temporal convolution
        x = layers.Conv2D(F1, (1, 64), padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)

        # Depthwise convolution
        x = layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                                   depth_multiplier=D)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AveragePooling2D((1, 4))(x)
        x = layers.Dropout(dropout)(x)

        # Separable convolution
        x = layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AveragePooling2D((1, 8))(x)
        x = layers.Dropout(dropout)(x)

        # Classifier
        x = layers.Flatten()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs, name='SchizophreniaEEGNet')
        return model


    def create_connectivity_gcn_keras(n_nodes: int = 116,
                                      num_classes: int = 2) -> Model:
        """
        Create Graph Convolutional Network for connectivity analysis
        """
        # Input: connectivity matrix
        inputs = keras.Input(shape=(n_nodes, n_nodes))

        # Simple spectral conv approximation
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(32, activation='relu')(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Classifier
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs, name='ConnectivityGCN')
        return model


# ============================================================================
# Model Factory and Training Utilities
# ============================================================================

class ModelFactory:
    """Factory class for creating disease-specific models"""

    @staticmethod
    def create_model(disease: str, model_type: str = 'default',
                     framework: str = 'pytorch', **kwargs) -> Any:
        """
        Create a model for specific disease

        Parameters
        ----------
        disease : str
            Disease type ('alzheimer', 'parkinson', 'schizophrenia')
        model_type : str
            Model architecture type
        framework : str
            Deep learning framework ('pytorch' or 'tensorflow')
        **kwargs : dict
            Additional model parameters

        Returns
        -------
        model : nn.Module or keras.Model
            Instantiated model
        """
        if framework == 'pytorch' and not HAS_TORCH:
            raise ImportError("PyTorch not available")
        if framework == 'tensorflow' and not HAS_TF:
            raise ImportError("TensorFlow not available")

        disease = disease.lower()

        if framework == 'pytorch':
            if disease == 'alzheimer':
                if model_type == 'transformer':
                    return AlzheimerTransformer(**kwargs)
                return AlzheimerCNN3D(**kwargs)

            elif disease == 'parkinson':
                if model_type == 'gait':
                    return ParkinsonGaitCNN(**kwargs)
                return ParkinsonVoiceLSTM(**kwargs)

            elif disease == 'schizophrenia':
                if model_type == 'graph':
                    return SchizophreniaGraphNet(**kwargs)
                return SchizophreniaEEGNet(**kwargs)

        else:  # tensorflow
            if disease == 'alzheimer':
                return create_alzheimer_cnn3d_keras(**kwargs)

            elif disease == 'parkinson':
                return create_parkinson_lstm_keras(**kwargs)

            elif disease == 'schizophrenia':
                if model_type == 'graph':
                    return create_connectivity_gcn_keras(**kwargs)
                return create_schizophrenia_eegnet_keras(**kwargs)

        raise ValueError(f"Unknown disease/model combination: {disease}/{model_type}")

    @staticmethod
    def list_available_models() -> Dict[str, List[str]]:
        """List all available models"""
        return {
            'alzheimer': ['cnn3d', 'transformer'],
            'parkinson': ['voice_lstm', 'gait_cnn'],
            'schizophrenia': ['eegnet', 'graph_net']
        }


class ModelTrainer:
    """Unified model trainer for PyTorch models"""

    def __init__(self, model, device: str = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for training")

        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4,
              patience: int = 10) -> Dict:
        """
        Train the model

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        epochs : int
            Number of epochs
        lr : float
            Learning rate
        weight_decay : float
            L2 regularization
        patience : int
            Early stopping patience

        Returns
        -------
        history : dict
            Training history
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                              f"train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, "
                              f"val_acc={val_acc:.4f}")

        return self.history

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on data"""
        self.model.eval()
        total_loss = 0
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
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)


if __name__ == "__main__":
    print("Deep Learning Models Demo")
    print("=" * 50)

    # List available models
    print("\nAvailable Models:")
    for disease, models in ModelFactory.list_available_models().items():
        print(f"  {disease}: {models}")

    if HAS_TORCH:
        print("\n--- PyTorch Models ---")

        # Create Alzheimer's model
        print("\nAlzheimer's CNN3D:")
        ad_model = ModelFactory.create_model('alzheimer', framework='pytorch')
        print(f"  Parameters: {sum(p.numel() for p in ad_model.parameters()):,}")

        # Test forward pass
        x = torch.randn(2, 1, 64, 64, 64)
        out = ad_model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

        # Create Parkinson's model
        print("\nParkinson's Voice LSTM:")
        pd_model = ModelFactory.create_model('parkinson', framework='pytorch')
        print(f"  Parameters: {sum(p.numel() for p in pd_model.parameters()):,}")

        x = torch.randn(2, 100, 26)
        out = pd_model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

        # Create Schizophrenia model
        print("\nSchizophrenia EEGNet:")
        sz_model = ModelFactory.create_model('schizophrenia', framework='pytorch')
        print(f"  Parameters: {sum(p.numel() for p in sz_model.parameters()):,}")

        x = torch.randn(2, 64, 256)
        out = sz_model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

    if HAS_TF:
        print("\n--- TensorFlow Models ---")

        # Create Alzheimer's model
        print("\nAlzheimer's CNN3D (Keras):")
        ad_model = ModelFactory.create_model('alzheimer', framework='tensorflow')
        ad_model.summary()

    print("\nDemo complete!")
