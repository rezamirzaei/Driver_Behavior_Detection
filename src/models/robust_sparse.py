"""
Robust Sparse Regression with Non-Convex Penalties.

Implements:
- Huber + L1 (Robust Lasso)
- SCAD (Smoothly Clipped Absolute Deviation)
- MCP (Minimax Concave Penalty)

These non-convex penalties reduce bias for large coefficients while
maintaining sparsity, unlike L1 which shrinks all coefficients.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class HuberL1Regressor(BaseEstimator, RegressorMixin):
    """
    Huber Loss + L1 Penalty (Robust Lasso).

    Combines Huber's robustness to outliers with L1 sparsity.
    Uses coordinate descent with soft-thresholding.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        epsilon: float = 1.35,
        max_iter: int = 1000,
        tol: float = 1e-4,
        warm_start: bool = False,
    ):
        """
        Args:
            alpha: L1 regularization strength.
            epsilon: Huber threshold - residuals > epsilon*scale use linear loss.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            warm_start: Reuse previous solution as initialization.
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def _huber_weights(self, residuals: np.ndarray, scale: float) -> np.ndarray:
        """Compute Huber weights for IRLS."""
        abs_res = np.abs(residuals)
        threshold = self.epsilon * scale
        weights = np.ones_like(residuals)
        mask = abs_res > threshold
        weights[mask] = threshold / abs_res[mask]
        return weights

    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding operator for L1."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HuberL1Regressor':
        """Fit using IRLS (Iteratively Reweighted Least Squares) + Coordinate Descent."""
        n_samples, n_features = X.shape

        # Initialize
        if self.warm_start and hasattr(self, 'coef_'):
            coef = self.coef_.copy()
            intercept = self.intercept_
        else:
            coef = np.zeros(n_features)
            intercept = np.median(y)

        # Scale estimate (MAD)
        residuals = y - X @ coef - intercept
        scale = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
        if scale < 1e-10:
            scale = 1.0

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            # Compute Huber weights
            residuals = y - X @ coef - intercept
            weights = self._huber_weights(residuals, scale)
            _sqrt_w = np.sqrt(weights)

            # Weighted coordinate descent
            for j in range(n_features):
                # Compute partial residual
                r_j = y - intercept - X @ coef + X[:, j] * coef[j]

                # Weighted regression coefficient
                numerator = np.sum(weights * X[:, j] * r_j)
                denominator = np.sum(weights * X[:, j] ** 2) + 1e-10

                # Soft threshold for L1
                coef[j] = self._soft_threshold(
                    numerator / denominator,
                    self.alpha / denominator
                )

            # Update intercept (no penalty)
            residuals = y - X @ coef
            intercept = np.sum(weights * residuals) / (np.sum(weights) + 1e-10)

            # Update scale
            residuals = y - X @ coef - intercept
            scale = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
            if scale < 1e-10:
                scale = 1.0

            # Check convergence
            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = iteration + 1
        self.scale_ = scale

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        return X @ self.coef_ + self.intercept_


class SCADRegressor(BaseEstimator, RegressorMixin):
    """
    SCAD (Smoothly Clipped Absolute Deviation) Penalty Regression.

    Non-convex penalty that:
    - Acts like L1 for small coefficients (sparsity)
    - Reduces bias for large coefficients (unlike L1)
    - Satisfies oracle property

    SCAD penalty:
    - |β| ≤ λ: λ|β|
    - λ < |β| ≤ aλ: (-β² + 2aλ|β| - λ²) / (2(a-1))
    - |β| > aλ: λ²(a+1)/2

    Default a=3.7 from Fan & Li (2001).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        a: float = 3.7,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ):
        """
        Args:
            alpha: Regularization strength (λ).
            a: SCAD shape parameter (default 3.7 from theory).
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            fit_intercept: Whether to fit intercept.
        """
        self.alpha = alpha
        self.a = a
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _scad_threshold(self, z: float, lam: float, a: float) -> float:
        """SCAD thresholding operator."""
        abs_z = np.abs(z)
        sign_z = np.sign(z)

        if abs_z <= lam:
            # Soft threshold region
            return sign_z * max(abs_z - lam, 0)
        elif abs_z <= a * lam:
            # Transition region
            return sign_z * ((a - 1) * abs_z - a * lam) / (a - 2)
        else:
            # No shrinkage region
            return z

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SCADRegressor':
        """Fit using Local Linear Approximation (LLA) algorithm."""
        n_samples, n_features = X.shape

        # Standardize X for coordinate descent stability
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_scaled = (X - X_mean) / X_std

        if self.fit_intercept:
            y_mean = y.mean()
            y_centered = y - y_mean
        else:
            y_mean = 0.0
            y_centered = y

        # Initialize with OLS solution
        coef = np.linalg.lstsq(X_scaled, y_centered, rcond=None)[0]

        # LLA iterations
        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            # Coordinate descent with SCAD thresholding
            for j in range(n_features):
                # Partial residual
                r_j = y_centered - X_scaled @ coef + X_scaled[:, j] * coef[j]

                # Least squares update
                z_j = np.dot(X_scaled[:, j], r_j) / n_samples
                norm_j = np.dot(X_scaled[:, j], X_scaled[:, j]) / n_samples

                # SCAD threshold
                coef[j] = self._scad_threshold(z_j / norm_j, self.alpha / norm_j, self.a)

            # Check convergence
            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        # Transform back to original scale
        self.coef_ = coef / X_std
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0

        self.n_iter_ = iteration + 1
        self.n_nonzero_ = np.sum(np.abs(self.coef_) > 1e-10)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        return X @ self.coef_ + self.intercept_


class MCPRegressor(BaseEstimator, RegressorMixin):
    """
    MCP (Minimax Concave Penalty) Regression.

    Non-convex penalty that:
    - Provides nearly unbiased estimates for large coefficients
    - Maintains sparsity for small coefficients
    - More aggressive than SCAD in removing bias

    MCP penalty (γ > 1):
    - |β| ≤ γλ: λ|β| - β²/(2γ)
    - |β| > γλ: γλ²/2

    The penalty derivative:
    - |β| ≤ γλ: λ - |β|/γ
    - |β| > γλ: 0 (no shrinkage!)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 3.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ):
        """
        Args:
            alpha: Regularization strength (λ).
            gamma: MCP shape parameter (γ > 1). Smaller γ = more aggressive.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            fit_intercept: Whether to fit intercept.
        """
        if gamma <= 1:
            raise ValueError("gamma must be > 1")
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _mcp_threshold(self, z: float, lam: float, gamma: float) -> float:
        """MCP thresholding operator."""
        abs_z = np.abs(z)
        sign_z = np.sign(z)

        if abs_z <= lam:
            # Below threshold - set to zero
            return 0.0
        elif abs_z <= gamma * lam:
            # Transition region - partial shrinkage
            return sign_z * (abs_z - lam) / (1 - 1/gamma)
        else:
            # Above threshold - no shrinkage
            return z

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MCPRegressor':
        """Fit using coordinate descent with MCP thresholding."""
        n_samples, n_features = X.shape

        # Standardize
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_scaled = (X - X_mean) / X_std

        if self.fit_intercept:
            y_mean = y.mean()
            y_centered = y - y_mean
        else:
            y_mean = 0.0
            y_centered = y

        # Initialize
        coef = np.zeros(n_features)

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            for j in range(n_features):
                # Partial residual
                r_j = y_centered - X_scaled @ coef + X_scaled[:, j] * coef[j]

                # Least squares direction
                z_j = np.dot(X_scaled[:, j], r_j) / n_samples
                norm_j = np.dot(X_scaled[:, j], X_scaled[:, j]) / n_samples

                # MCP threshold
                coef[j] = self._mcp_threshold(z_j / norm_j, self.alpha / norm_j, self.gamma)

            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        # Transform back
        self.coef_ = coef / X_std
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0

        self.n_iter_ = iteration + 1
        self.n_nonzero_ = np.sum(np.abs(self.coef_) > 1e-10)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        return X @ self.coef_ + self.intercept_


class HuberMCPRegressor(BaseEstimator, RegressorMixin):
    """
    Huber Loss + MCP Penalty (Robust Non-Convex Sparse Regression).

    Combines:
    - Huber's robustness to outliers
    - MCP's nearly unbiased sparse estimation
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 3.0,
        epsilon: float = 1.35,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol

    def _huber_weights(self, residuals: np.ndarray, scale: float) -> np.ndarray:
        """Compute Huber weights."""
        abs_res = np.abs(residuals)
        threshold = self.epsilon * scale
        weights = np.ones_like(residuals)
        mask = abs_res > threshold
        weights[mask] = threshold / abs_res[mask]
        return weights

    def _mcp_threshold(self, z: float, lam: float, gamma: float) -> float:
        """MCP thresholding."""
        abs_z = np.abs(z)
        sign_z = np.sign(z)

        if abs_z <= lam:
            return 0.0
        elif abs_z <= gamma * lam:
            return sign_z * (abs_z - lam) / (1 - 1/gamma)
        else:
            return z

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HuberMCPRegressor':
        """Fit using IRLS + Coordinate Descent with MCP."""
        n_samples, n_features = X.shape

        coef = np.zeros(n_features)
        intercept = np.median(y)

        residuals = y - X @ coef - intercept
        scale = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
        if scale < 1e-10:
            scale = 1.0

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            residuals = y - X @ coef - intercept
            weights = self._huber_weights(residuals, scale)

            for j in range(n_features):
                r_j = y - intercept - X @ coef + X[:, j] * coef[j]

                numerator = np.sum(weights * X[:, j] * r_j)
                denominator = np.sum(weights * X[:, j] ** 2) + 1e-10

                coef[j] = self._mcp_threshold(
                    numerator / denominator,
                    self.alpha / denominator,
                    self.gamma
                )

            residuals = y - X @ coef
            intercept = np.sum(weights * residuals) / (np.sum(weights) + 1e-10)

            residuals = y - X @ coef - intercept
            scale = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
            if scale < 1e-10:
                scale = 1.0

            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = iteration + 1
        self.scale_ = scale
        self.n_nonzero_ = np.sum(np.abs(self.coef_) > 1e-10)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


class HuberSCADRegressor(BaseEstimator, RegressorMixin):
    """
    Huber Loss + SCAD Penalty (Robust Non-Convex Sparse Regression).

    Combines:
    - Huber's robustness to outliers
    - SCAD's nearly unbiased sparse estimation with oracle property
    """

    def __init__(
        self,
        alpha: float = 1.0,
        a: float = 3.7,
        epsilon: float = 1.35,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        self.alpha = alpha
        self.a = a
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol

    def _huber_weights(self, residuals: np.ndarray, scale: float) -> np.ndarray:
        abs_res = np.abs(residuals)
        threshold = self.epsilon * scale
        weights = np.ones_like(residuals)
        mask = abs_res > threshold
        weights[mask] = threshold / abs_res[mask]
        return weights

    def _scad_threshold(self, z: float, lam: float, a: float) -> float:
        abs_z = np.abs(z)
        sign_z = np.sign(z)

        if abs_z <= lam:
            return sign_z * max(abs_z - lam, 0)
        elif abs_z <= a * lam:
            return sign_z * ((a - 1) * abs_z - a * lam) / (a - 2)
        else:
            return z

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HuberSCADRegressor':
        """Fit using IRLS + Coordinate Descent with SCAD."""
        n_samples, n_features = X.shape

        coef = np.zeros(n_features)
        intercept = np.median(y)

        residuals = y - X @ coef - intercept
        scale = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
        if scale < 1e-10:
            scale = 1.0

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            residuals = y - X @ coef - intercept
            weights = self._huber_weights(residuals, scale)

            for j in range(n_features):
                r_j = y - intercept - X @ coef + X[:, j] * coef[j]

                numerator = np.sum(weights * X[:, j] * r_j)
                denominator = np.sum(weights * X[:, j] ** 2) + 1e-10

                coef[j] = self._scad_threshold(
                    numerator / denominator,
                    self.alpha / denominator,
                    self.a
                )

            residuals = y - X @ coef
            intercept = np.sum(weights * residuals) / (np.sum(weights) + 1e-10)

            residuals = y - X @ coef - intercept
            scale = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
            if scale < 1e-10:
                scale = 1.0

            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = iteration + 1
        self.scale_ = scale
        self.n_nonzero_ = np.sum(np.abs(self.coef_) > 1e-10)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


def get_sparse_robust_regressors(
    alpha: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Get dictionary of sparse and robust regressors.

    Returns models with various penalty types:
    - L1 (Lasso)
    - Huber + L1 (Robust Lasso)
    - SCAD (non-convex)
    - MCP (non-convex)
    - Huber + SCAD
    - Huber + MCP
    """
    from sklearn.linear_model import HuberRegressor, Lasso

    return {
        "Lasso (L1)": Lasso(alpha=alpha, max_iter=2000, random_state=random_state),
        "Huber (Robust)": HuberRegressor(epsilon=1.35, max_iter=1000),
        "Huber + L1": HuberL1Regressor(alpha=alpha, epsilon=1.35, max_iter=1000),
        "SCAD (Non-Convex)": SCADRegressor(alpha=alpha, a=3.7, max_iter=1000),
        "MCP (Non-Convex)": MCPRegressor(alpha=alpha, gamma=3.0, max_iter=1000),
        "Huber + SCAD": HuberSCADRegressor(alpha=alpha, a=3.7, epsilon=1.35, max_iter=1000),
        "Huber + MCP": HuberMCPRegressor(alpha=alpha, gamma=3.0, epsilon=1.35, max_iter=1000),
    }

