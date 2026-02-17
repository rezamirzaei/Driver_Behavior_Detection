"""
ResNet model for tabular/sequence classification using PyTorch.
Compatible with scikit-learn API for easy integration.

ResNet uses skip connections to enable training of deeper networks
and helps prevent vanishing gradients.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with skip connection.

    Architecture:
    - Conv1D -> BatchNorm -> ELU -> Conv1D -> BatchNorm
    - Skip connection (identity or projection)
    - ELU activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super(ResidualBlock1D, self).__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu1 = nn.ELU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        # Skip connection: identity if same dimensions, projection otherwise
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm1d(out_channels)
            )

        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = out + identity  # Skip connection
        out = self.elu2(out)

        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for tabular/sequence classification.

    Architecture:
    - Initial Conv1D -> BatchNorm -> ELU
    - ResidualBlock x n_blocks
    - Global Average Pooling
    - Dense -> Dropout -> Dense (output)

    Uses ELU instead of ReLU for smoother gradients.
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        n_filters: int = 32,
        n_blocks: int = 2,
        kernel_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ):
        super(ResNet1D, self).__init__()

        self.input_size = input_size

        # Adjust kernel size for small inputs
        actual_kernel = min(kernel_size, max(1, input_size // 2))
        if actual_kernel % 2 == 0:
            actual_kernel = max(1, actual_kernel - 1)  # Ensure odd kernel

        # Initial convolution
        self.conv_initial = nn.Conv1d(1, n_filters, actual_kernel, padding=actual_kernel // 2)
        self.bn_initial = nn.BatchNorm1d(n_filters)
        self.elu_initial = nn.ELU(inplace=True)
        self.dropout_initial = nn.Dropout(dropout * 0.5)

        # Residual blocks with increasing filters
        blocks = []
        in_ch = n_filters
        for i in range(n_blocks):
            out_ch = n_filters * (2 ** min(i, 2))  # Cap at 4x initial filters
            blocks.append(ResidualBlock1D(in_ch, out_ch, actual_kernel, stride=1, dropout=dropout))
            in_ch = out_ch
        self.res_blocks = nn.Sequential(*blocks)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers - simpler for small data
        self.fc1 = nn.Linear(in_ch, hidden_size)
        self.bn_fc = nn.BatchNorm1d(hidden_size)
        self.elu_fc = nn.ELU(inplace=True)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_classes)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, features)
        x = self.conv_initial(x)
        x = self.bn_initial(x)
        x = self.elu_initial(x)
        x = self.dropout_initial(x)

        x = self.res_blocks(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.elu_fc(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x  # Raw logits (CrossEntropyLoss applies softmax)


class ResNetClassifier(BaseEstimator, ClassifierMixin):
    """
    1D ResNet Classifier using PyTorch with skip connections.
    Compatible with scikit-learn pipeline and cross-validation.

    Key improvements over CNN:
    - Skip connections prevent vanishing gradients
    - ELU activation for smoother optimization
    - Proper regularization (dropout, weight decay)
    - Careful architecture for small datasets

    Example:
        >>> clf = ResNetClassifier(n_filters=32, n_blocks=2, epochs=100)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        n_filters: int = 32,
        n_blocks: int = 2,
        kernel_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.3,
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-3,
        validation_split: float = 0.2,
        early_stopping_patience: int = 15,
        verbose: int = 1,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize the ResNet Classifier.

        Args:
            n_filters: Base number of filters (increases in deeper blocks)
            n_blocks: Number of residual blocks
            kernel_size: Kernel size for conv layers
            hidden_size: Size of final hidden layer
            dropout: Dropout rate
            epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization (important for small datasets)
            validation_split: Fraction for validation
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level
            device: Device ('cpu', 'cuda', 'mps', or None for auto)
            random_state: Random seed
        """
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_split = validation_split
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
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _prepare_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare data for PyTorch."""
        if hasattr(X, "values"):
            X = X.values

        X = X.astype(np.float32)

        # Reshape for Conv1D: (batch, features) -> (batch, 1, features)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))

        X_tensor = torch.from_numpy(X)

        if y is not None:
            if hasattr(y, "values"):
                y = y.values
            y = y.astype(np.int64)
            y_tensor = torch.from_numpy(y)
            return X_tensor, y_tensor

        return X_tensor, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ResNetClassifier":
        """
        Fit the ResNet classifier.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)

        Returns:
            self
        """
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Get device
        self.device_ = self._get_device()
        if self.verbose > 0:
            print(f"🔧 Training ResNet on device: {self.device_}")

        # Encode labels
        if y.dtype == "object" or (len(y) > 0 and isinstance(y[0], str)):
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
        n_val = max(int(n_samples * self.validation_split), 1)
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
        self.model_ = ResNet1D(
            input_size=self.input_size_,
            n_classes=self.n_classes_,
            n_filters=self.n_filters,
            n_blocks=self.n_blocks,
            kernel_size=self.kernel_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device_)

        # Compute class weights for imbalanced data
        class_counts = np.bincount(y_encoded.astype(int))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device_)

        # Loss and optimizer with weight decay
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=False)

        # Training history
        self.history_ = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")
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

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)

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

            # Update scheduler
            scheduler.step(val_loss)

            # Store history
            self.history_["train_loss"].append(train_loss)
            self.history_["train_acc"].append(train_acc)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_acc"].append(val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1

            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch + 1}/{self.epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                )

            if patience_counter >= self.early_stopping_patience:
                if self.verbose > 0:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
            self.model_ = self.model_.to(self.device_)

        if self.verbose > 0:
            final_train_acc = self.history_["train_acc"][-1] if self.history_["train_acc"] else 0
            final_val_acc = self.history_["val_acc"][-1] if self.history_["val_acc"] else 0
            print(f"✅ Training complete. Best Val Loss: {best_val_loss:.4f}")
            print(f"   Final Train Acc: {final_train_acc:.4f}, Final Val Acc: {final_val_acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(indices)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
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
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        if y.dtype == "object" or (len(y) > 0 and isinstance(y[0], str)):
            return float(np.mean(y_pred == y))
        else:
            return float(np.mean(self.le_.transform(y_pred) == y))


def plot_resnet_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot ResNet training history.

    Args:
        history: Training history dict from ResNetClassifier.get_training_history()
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("ResNet: Training and Validation Loss", fontweight="bold", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-", label="Training Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("ResNet: Training and Validation Accuracy", fontweight="bold", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
    else:
        plt.show()

    return fig
