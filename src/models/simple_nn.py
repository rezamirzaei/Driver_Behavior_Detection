"""
Simple Neural Network Classifier for tabular data using PyTorch.
Properly handles data normalization and is suitable for small datasets.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    """
    Simple Multi-Layer Perceptron for classification.

    Architecture:
    - Input -> BatchNorm -> Dense -> ReLU -> Dropout
    - Dense -> ReLU -> Dropout
    - Dense -> Output (logits)
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        hidden_sizes: List[int] = [64, 32],
        dropout: float = 0.3,
    ):
        super(SimpleNN, self).__init__()

        layers = []
        prev_size = input_size

        # Input batch normalization - CRITICAL for neural networks
        layers.append(nn.BatchNorm1d(input_size))

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, n_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimpleNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple Neural Network Classifier with proper data normalization.
    Compatible with scikit-learn API.

    Key features:
    - StandardScaler normalization (zero mean, unit variance)
    - Batch normalization layers
    - Class weights for imbalanced data
    - Early stopping to prevent overfitting

    Example:
        >>> clf = SimpleNNClassifier(hidden_sizes=[64, 32], epochs=100)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 32],
        dropout: float = 0.3,
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        validation_split: float = 0.2,
        early_stopping_patience: int = 15,
        verbose: int = 1,
        random_state: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.random_state = random_state

        # Set by fit()
        self.model_ = None
        self.scaler_ = StandardScaler()  # IMPORTANT: Normalize inputs
        self.le_ = LabelEncoder()
        self.classes_ = None
        self.n_classes_ = None
        self.history_ = None
        self.device_ = None

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleNNClassifier":
        """Fit the neural network classifier."""
        # Set seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.device_ = self._get_device()
        if self.verbose:
            print(f"🔧 Training SimpleNN on device: {self.device_}")

        # Convert to numpy
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # CRITICAL: Normalize input data
        X_normalized = self.scaler_.fit_transform(X).astype(np.float32)

        # Encode labels
        if y.dtype == "object" or isinstance(y[0], str):
            y_encoded = self.le_.fit_transform(y)
        else:
            y_encoded = y.copy()
            self.le_.classes_ = np.unique(y)

        self.classes_ = self.le_.classes_
        self.n_classes_ = len(self.classes_)

        # Split train/val
        n_samples = len(X_normalized)
        n_val = max(int(n_samples * self.validation_split), 1)
        indices = np.random.permutation(n_samples)

        X_train = torch.FloatTensor(X_normalized[indices[n_val:]])
        y_train = torch.LongTensor(y_encoded[indices[n_val:]])
        X_val = torch.FloatTensor(X_normalized[indices[:n_val]])
        y_val = torch.LongTensor(y_encoded[indices[:n_val]])

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)

        # Build model
        self.model_ = SimpleNN(
            input_size=X_normalized.shape[1],
            n_classes=self.n_classes_,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device_)

        # Class weights for imbalanced data
        class_counts = np.bincount(y_encoded.astype(int))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device_))
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        # Training
        self.history_ = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model_.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)

                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validate
            self.model_.eval()
            val_loss, val_correct, val_total = 0, 0, 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device_)
                    y_batch = y_batch.to(self.device_)

                    outputs = self.model_(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item() * X_batch.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            scheduler.step(val_loss)

            self.history_["train_loss"].append(train_loss)
            self.history_["train_acc"].append(train_acc)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_acc"].append(val_acc)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch + 1}/{self.epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if best_state:
            self.model_.load_state_dict(best_state)
            self.model_ = self.model_.to(self.device_)

        if self.verbose:
            print(f"✅ Training complete. Best Val Loss: {best_val_loss:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(indices)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(X, "values"):
            X = X.values

        # CRITICAL: Apply same normalization as training
        X_normalized = self.scaler_.transform(X).astype(np.float32)

        self.model_.eval()
        X_tensor = torch.FloatTensor(X_normalized).to(self.device_)

        with torch.no_grad():
            outputs = self.model_(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        y_pred = self.predict(X)
        if hasattr(y, "values"):
            y = y.values
        return float(np.mean(y_pred == y))

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history for plotting."""
        return self.history_


def plot_nn_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot neural network training history with consistent styling."""
    import matplotlib.pyplot as plt

    # Set consistent style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontweight="bold", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(1, len(epochs))

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Training Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy", fontweight="bold", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlim(1, len(epochs))

    # Add final accuracy annotation
    final_val = history["val_acc"][-1]
    axes[1].axhline(y=final_val, color="r", linestyle="--", alpha=0.5)
    axes[1].text(len(epochs) * 0.7, final_val + 0.03, f"Final Val: {final_val:.3f}", fontsize=10)

    plt.suptitle("Neural Network Training (with Data Normalization)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

    return fig
