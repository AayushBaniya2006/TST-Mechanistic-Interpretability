"""
Time Series Transformer training utilities.

This module provides a Transformer-based classifier for multivariate
time series data, along with data loading and training infrastructure.

Architecture:
    - Convolutional embedding (3 layers) for initial feature extraction
    - Learnable positional encoding
    - Standard Transformer encoder
    - Adaptive max pooling + linear classifier

Example:
    >>> train_loader, test_loader = load_dataset("JapaneseVowels")
    >>> model = TimeSeriesTransformer(input_dim=12, num_classes=9, seq_len=29)
    >>> trainer = Trainer(model, train_loader, test_loader)
    >>> trainer.train(epochs=100)
    >>> trainer.evaluate()
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.data import DataLoader, TensorDataset
from aeon.datasets import load_classification


def load_dataset(name: str, batch_size: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess a time series classification dataset.

    Loads data from the UCR/UEA time series archive via aeon,
    preprocesses to (batch, seq_len, channels) format, and
    remaps labels to zero-indexed integers.

    Args:
        name: Dataset name from timeseriesclassification.com
              (e.g., "JapaneseVowels", "PenDigits", "LSST")
        batch_size: Batch size for DataLoaders

    Returns:
        Tuple of (train_loader, test_loader)

    Example:
        >>> train_loader, test_loader = load_dataset("JapaneseVowels", batch_size=32)
        >>> X, y = next(iter(train_loader))
        >>> print(X.shape)  # (32, seq_len, channels)
    """
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    X_train = _preprocess_series(X_train)
    X_test = _preprocess_series(X_test)

    y_train, y_test = _remap_labels(y_train, y_test)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def _preprocess_series(X: np.ndarray) -> torch.Tensor:
    """
    Preprocess time series data for the model.

    Converts from (N, channels, seq_len) to (N, seq_len, channels)
    and casts to float32 tensor.

    Args:
        X: Input array from aeon loader, shape (N, channels, seq_len)

    Returns:
        Tensor of shape (N, seq_len, channels)
    """
    arr = X.astype(np.float32)
    arr = np.swapaxes(arr, 1, 2)
    return torch.tensor(arr)


def _remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert labels to zero-indexed int64 tensors.

    Some datasets have labels starting at 1; this ensures they start at 0.

    Args:
        y_train: Training labels
        y_test: Test labels

    Returns:
        Tuple of (train_labels, test_labels) as int64 tensors
    """
    t_train = torch.tensor(y_train.astype(np.int64))
    t_test = torch.tensor(y_test.astype(np.int64))
    min_val = int(t_train.min())
    if min_val == 0:
        t_train -= min_val
        t_test -= min_val
    return t_train, t_test


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based classifier for multivariate time series.

    Architecture:
        1. Convolutional embedding: 3 Conv1d layers with BatchNorm
           - Conv1(input_dim -> d_model/4, kernel=5)
           - Conv2(d_model/4 -> d_model/2, kernel=3)
           - Conv3(d_model/2 -> d_model, kernel=3)
        2. Learnable positional encoding added to embeddings
        3. Standard TransformerEncoder with num_encoder_layers
        4. Adaptive max pooling over sequence -> (batch, d_model)
        5. Linear classifier -> (batch, num_classes)

    Args:
        input_dim: Number of input channels/features per timestep
        num_classes: Number of output classes
        seq_len: Length of input sequences
        d_model: Transformer hidden dimension (default: 128)
        n_head: Number of attention heads (default: 8)
        num_encoder_layers: Number of transformer layers (default: 3)
        dim_feedforward: FFN hidden dimension (default: 256)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> model = TimeSeriesTransformer(
        ...     input_dim=12,      # 12 features per timestep
        ...     num_classes=9,      # 9-way classification
        ...     seq_len=29,         # sequence length
        ...     d_model=128,
        ...     n_head=8,
        ...     num_encoder_layers=3
        ... )
        >>> x = torch.randn(4, 29, 12)  # (batch, seq_len, input_dim)
        >>> logits = model(x)           # (4, 9)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        n_head: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Convolutional embedding layers
        # kernel_size=5 for first layer captures broader temporal patterns
        # kernel_size=3 for subsequent layers refines features
        self.conv1 = nn.Conv1d(input_dim, d_model // 4, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(d_model // 4)
        self.conv2 = nn.Conv1d(d_model // 4, d_model // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model // 2)
        self.conv3 = nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(d_model)

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Classification head
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # x: (B, seq_len, channels)
        x = x.transpose(1, 2)                          # (B, channels, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))        # conv1 -> bn1 -> relu
        x = torch.relu(self.bn2(self.conv2(x)))        # conv2 -> bn2 -> relu
        x = torch.relu(self.bn3(self.conv3(x)))        # conv3 -> bn3 -> relu
        x = x.transpose(1, 2)                          # (B, seq_len, d_model)
        x = x + self.pos_enc                           # add positional encoding
        x = self.transformer_encoder(x)                # transformer encoder
        x = x.transpose(1, 2)                          # (B, d_model, seq_len)
        x = self.pool(x).squeeze(-1)                   # global pooling -> (B, d_model)
        return self.classifier(x)                      # (B, num_classes)


class Trainer:
    """
    Training and evaluation wrapper for TimeSeriesTransformer.

    Handles device management, optimizer setup, training loop,
    and evaluation.

    Args:
        model: TimeSeriesTransformer model
        train_loader: Training data DataLoader
        test_loader: Test data DataLoader
        lr: Learning rate (default: 1e-3)
        weight_decay: L2 regularization (default: 1e-4)
        device: Torch device (default: auto-detect CUDA)

    Example:
        >>> trainer = Trainer(model, train_loader, test_loader)
        >>> trainer.train(epochs=100)
        >>> accuracy = trainer.evaluate()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.RAdam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, epochs: int = 100) -> None:
        """
        Train the model for specified number of epochs.

        Prints loss and learning rate after each epoch.

        Args:
            epochs: Number of training epochs
        """
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(Xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * Xb.size(0)
            avg_loss = total_loss / len(self.train_loader)
            lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:03d}/{epochs} -- Loss: {avg_loss:.4f} -- LR: {lr:.6f}')

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on test set.

        Returns:
            Test accuracy as a float between 0 and 1
        """
        self.model.eval()
        correct = 0
        total = 0
        for Xb, yb in self.test_loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            preds = self.model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        acc = correct / total
        print(f'Accuracy: {acc * 100:.2f}%')
        return acc


def main():
    """
    Main training script.

    Loads dataset, creates model, trains (or loads from checkpoint),
    and evaluates.
    """
    # Hyperparameters
    dataset_name = "LSST"  # specify name from https://www.timeseriesclassification.com/dataset.php
    batch_size = 32
    num_epochs = 100
    model_path = f"TST_{dataset_name.lower()}.pth"

    # Load data
    train_loader, test_loader = load_dataset(dataset_name, batch_size)

    # Determine num_classes
    train_labels = train_loader.dataset.tensors[1]
    test_labels = test_loader.dataset.tensors[1]
    num_classes = int(torch.cat([train_labels, test_labels]).max().item()) + 1

    # Model instantiation
    sample_seq, _ = next(iter(train_loader))
    seq_len, channels = sample_seq.shape[1], sample_seq.shape[2]
    model = TimeSeriesTransformer(
        input_dim=channels,
        num_classes=num_classes,
        seq_len=seq_len,
    )

    trainer = Trainer(model, train_loader, test_loader)

    # Load or train
    if os.path.exists(model_path):
        trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
        print(f'Model loaded from {model_path}')
    else:
        trainer.train(epochs=num_epochs)
        torch.save(trainer.model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    trainer.evaluate()


if __name__ == '__main__':
    main()
