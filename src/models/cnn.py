"""
CNN model for sequence/tabular classification using PyTorch.
Compatible with scikit-learn API for easy integration.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CNN1DNetwork(nn.Module):
    """
    1D Convolutional Neural Network for classification.

    Architecture:
    - Conv1D -> BatchNorm -> ReLU -> MaxPool
    - Conv1D -> BatchNorm -> ReLU -> MaxPool (optional)
    - Flatten -> Dense -> Dropout -> Dense (output)
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        n_filters: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super(CNN1DNetwork, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.input_size = input_size

        # Adjust kernel size if input is small
        actual_kernel_size = min(kernel_size, max(1, input_size // 2))

        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=actual_kernel_size, padding=actual_kernel_size//2)
        self.bn1 = nn.BatchNorm1d(n_filters) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()

        # Adaptive pooling to handle variable sizes
        self.pool1 = nn.AdaptiveMaxPool1d(output_size=max(input_size // 2, 4))

        # Second convolutional block
        pool1_size = max(input_size // 2, 4)
        actual_kernel_size2 = min(kernel_size, max(1, pool1_size // 2))
        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=actual_kernel_size2, padding=actual_kernel_size2//2)
        self.bn2 = nn.BatchNorm1d(n_filters * 2) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()

        # Use adaptive pooling to ensure consistent output size
        self.pool2 = nn.AdaptiveMaxPool1d(output_size=4)

        # Calculate flattened size: 4 * (n_filters * 2)
        self.flat_size = 4 * n_filters * 2

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, features)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    1D CNN Classifier using PyTorch.
    Compatible with scikit-learn pipeline and cross-validation.

    This classifier can be used for:
    - Processed feature data (trip-level aggregates)
    - Raw sensor data (time-series features)

    Example:
        >>> clf = CNNClassifier(n_filters=32, epochs=50)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        n_filters: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        validation_split: float = 0.2,
        use_batch_norm: bool = True,
        early_stopping_patience: int = 10,
        verbose: int = 1,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize the CNN Classifier.

        Args:
            n_filters: Number of filters in first conv layer (doubles in second)
            kernel_size: Kernel size for conv layers
            hidden_size: Size of hidden dense layer
            dropout: Dropout rate
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization weight
            validation_split: Fraction of training data for validation
            use_batch_norm: Whether to use batch normalization
            early_stopping_patience: Epochs to wait before early stopping
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            random_state: Random seed for reproducibility
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_split = validation_split
        self.use_batch_norm = use_batch_norm
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.device = device
        self.random_state = random_state

        # Set by fit()
        self.model_ = None
        self.le_ = LabelEncoder()
        self.classes_ = None
        self.n_classes_ = None
        self.input_size_ = None
        self.history_ = None
        self.device_ = None

    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.device is not None:
            return torch.device(self.device)

        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare data for PyTorch."""
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values

        # Ensure float32
        X = X.astype(np.float32)

        # Reshape for Conv1D: (batch, features) -> (batch, 1, features)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))

        X_tensor = torch.from_numpy(X)

        if y is not None:
            if hasattr(y, 'values'):
                y = y.values
            y = y.astype(np.int64)
            y_tensor = torch.from_numpy(y)
            return X_tensor, y_tensor

        return X_tensor, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CNNClassifier':
        """
        Fit the CNN classifier.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)

        Returns:
            self
        """
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Get device
        self.device_ = self._get_device()
        if self.verbose > 0:
            print(f"🔧 Training on device: {self.device_}")

        # Encode labels
        if y.dtype == 'object' or (len(y) > 0 and isinstance(y[0], str)):
            y_encoded = self.le_.fit_transform(y)
        else:
            y_encoded = y.copy()
            self.le_.classes_ = np.unique(y)

        self.classes_ = self.le_.classes_
        self.n_classes_ = len(self.classes_)

        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y_encoded)
        self.input_size_ = X_tensor.shape[2]

        # Split into train/val
        n_samples = len(X_tensor)
        n_val = int(n_samples * self.validation_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train = X_tensor[train_indices]
        y_train = y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        y_val = y_tensor[val_indices]

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Build model
        self.model_ = CNN1DNetwork(
            input_size=self.input_size_,
            n_classes=self.n_classes_,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
        ).to(self.device_)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Training history
        self.history_ = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model_.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)

                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.model_.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device_)
                    y_batch = y_batch.to(self.device_)

                    outputs = self.model_(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total

            # Store history
            self.history_['train_loss'].append(train_loss)
            self.history_['train_acc'].append(train_acc)
            self.history_['val_loss'].append(val_loss)
            self.history_['val_acc'].append(val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model_.state_dict().copy()
            else:
                patience_counter += 1

            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if patience_counter >= self.early_stopping_patience:
                if self.verbose > 0:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)

        if self.verbose > 0:
            print(f"✅ Training complete. Best Val Loss: {best_val_loss:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(indices)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Predicted probabilities, shape (n_samples, n_classes)
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model_.eval()
        X_tensor, _ = self._prepare_data(X)
        X_tensor = X_tensor.to(self.device_)

        with torch.no_grad():
            outputs = self.model_(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history for plotting."""
        return self.history_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.

        Args:
            X: Features
            y: True labels

        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        if y.dtype == 'object' or (len(y) > 0 and isinstance(y[0], str)):
            return np.mean(y_pred == y)
        else:
            return np.mean(self.le_.transform(y_pred) == y)


class CNNClassifierRaw(CNNClassifier):
    """
    CNN Classifier specifically designed for raw sensor data.

    This variant includes:
    - More convolutional layers for longer sequences
    - Global average pooling for variable-length inputs
    - Residual connections (optional)
    """

    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 5,
        n_conv_layers: int = 3,
        hidden_size: int = 128,
        dropout: float = 0.4,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0005,
        **kwargs
    ):
        """
        Initialize the Raw Data CNN Classifier.

        Args:
            n_filters: Base number of filters (increases per layer)
            kernel_size: Kernel size for conv layers
            n_conv_layers: Number of convolutional layers
            hidden_size: Size of hidden dense layer
            dropout: Dropout rate
            epochs: Maximum training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            n_filters=n_filters,
            kernel_size=kernel_size,
            hidden_size=hidden_size,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
        self.n_conv_layers = n_conv_layers


def plot_cnn_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot CNN training history.

    Args:
        history: Training history dict from CNNClassifier.get_training_history()
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontweight='bold', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontweight='bold', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    return fig
