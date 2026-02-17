"""
Advanced Regularized Classifiers.

Includes MCP (Minimax Concave Penalty) and SCAD penalized logistic regression.
These provide better sparse solutions than L1 for feature selection.
"""

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class MCPLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with Minimax Concave Penalty (MCP).

    MCP provides a nearly unbiased sparse solution, unlike L1 which
    shrinks large coefficients too much.

    The MCP penalty is defined as:
        p(|beta|; lambda, gamma) = lambda * |beta| - beta^2/(2*gamma)  if |beta| <= gamma*lambda
                                 = gamma * lambda^2 / 2                 if |beta| > gamma*lambda

    Args:
        alpha: Regularization strength (lambda)
        gamma: MCP concavity parameter (gamma > 1, typically 2-4)
        max_iter: Maximum iterations for optimization
        tol: Convergence tolerance
        random_state: Random seed
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 3.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _mcp_penalty(self, beta: np.ndarray) -> float:
        """Compute MCP penalty value."""
        abs_beta = np.abs(beta)
        penalty = np.where(
            abs_beta <= self.gamma * self.alpha,
            self.alpha * abs_beta - beta**2 / (2 * self.gamma),
            self.gamma * self.alpha**2 / 2
        )
        return penalty.sum()

    def _mcp_gradient(self, beta: np.ndarray) -> np.ndarray:
        """Compute MCP penalty gradient."""
        abs_beta = np.abs(beta)
        grad = np.where(
            abs_beta <= self.gamma * self.alpha,
            self.alpha * np.sign(beta) - beta / self.gamma,
            0
        )
        return grad

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _objective(self, params: np.ndarray, X: np.ndarray, y_onehot: np.ndarray) -> float:
        """Negative log-likelihood + MCP penalty."""
        n_samples, n_features = X.shape
        n_classes = y_onehot.shape[1]

        # Reshape parameters
        W = params[:n_features * n_classes].reshape(n_features, n_classes)
        b = params[n_features * n_classes:]

        # Forward pass
        logits = X @ W + b
        probs = self._softmax(logits)

        # Cross-entropy loss
        eps = 1e-10
        loss = -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))

        # MCP penalty (don't penalize bias)
        penalty = self._mcp_penalty(W.ravel())

        return loss + penalty

    def _gradient(self, params: np.ndarray, X: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
        """Gradient of objective."""
        n_samples, n_features = X.shape
        n_classes = y_onehot.shape[1]

        # Reshape parameters
        W = params[:n_features * n_classes].reshape(n_features, n_classes)
        b = params[n_features * n_classes:]

        # Forward pass
        logits = X @ W + b
        probs = self._softmax(logits)

        # Gradient of cross-entropy
        diff = probs - y_onehot
        grad_W = (X.T @ diff) / n_samples
        grad_b = diff.mean(axis=0)

        # Add MCP gradient
        grad_W += self._mcp_gradient(W)

        return np.concatenate([grad_W.ravel(), grad_b])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)

        # One-hot encode
        y_onehot = np.zeros((len(y), n_classes))
        y_onehot[np.arange(len(y)), y_encoded] = 1

        n_samples, n_features = X.shape

        # Initialize parameters
        if self.random_state is not None:
            np.random.seed(self.random_state)
        W_init = np.random.randn(n_features, n_classes) * 0.01
        b_init = np.zeros(n_classes)
        params_init = np.concatenate([W_init.ravel(), b_init])

        # Optimize
        result = minimize(
            self._objective,
            params_init,
            args=(X, y_onehot),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )

        # Extract parameters
        self.coef_ = result.x[:n_features * n_classes].reshape(n_features, n_classes)
        self.intercept_ = result.x[n_features * n_classes:]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        logits = X @ self.coef_ + self.intercept_
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(indices)

    def get_selected_features(self, threshold: float = 1e-5) -> np.ndarray:
        """Get indices of selected features (non-zero coefficients)."""
        importance = np.abs(self.coef_).max(axis=1)
        return np.where(importance > threshold)[0]


class SCADLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with SCAD (Smoothly Clipped Absolute Deviation) penalty.

    SCAD provides similar benefits to MCP - nearly unbiased sparse solutions.

    Args:
        alpha: Regularization strength (lambda)
        a: SCAD parameter (a > 2, typically 3.7)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
    """

    def __init__(
        self,
        alpha: float = 1.0,
        a: float = 3.7,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ):
        self.alpha = alpha
        self.a = a
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _scad_penalty(self, beta: np.ndarray) -> float:
        """Compute SCAD penalty value."""
        abs_beta = np.abs(beta)
        penalty = np.where(
            abs_beta <= self.alpha,
            self.alpha * abs_beta,
            np.where(
                abs_beta <= self.a * self.alpha,
                (2 * self.a * self.alpha * abs_beta - beta**2 - self.alpha**2) / (2 * (self.a - 1)),
                (self.a + 1) * self.alpha**2 / 2
            )
        )
        return penalty.sum()

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _objective(self, params: np.ndarray, X: np.ndarray, y_onehot: np.ndarray) -> float:
        """Negative log-likelihood + SCAD penalty."""
        n_samples, n_features = X.shape
        n_classes = y_onehot.shape[1]

        W = params[:n_features * n_classes].reshape(n_features, n_classes)
        b = params[n_features * n_classes:]

        logits = X @ W + b
        probs = self._softmax(logits)

        eps = 1e-10
        loss = -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))
        penalty = self._scad_penalty(W.ravel())

        return loss + penalty

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)

        y_onehot = np.zeros((len(y), n_classes))
        y_onehot[np.arange(len(y)), y_encoded] = 1

        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)
        W_init = np.random.randn(n_features, n_classes) * 0.01
        b_init = np.zeros(n_classes)
        params_init = np.concatenate([W_init.ravel(), b_init])

        result = minimize(
            self._objective,
            params_init,
            args=(X, y_onehot),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )

        self.coef_ = result.x[:n_features * n_classes].reshape(n_features, n_classes)
        self.intercept_ = result.x[n_features * n_classes:]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        logits = X @ self.coef_ + self.intercept_
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(indices)

