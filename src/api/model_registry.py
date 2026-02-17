"""Central model registry for API-facing model metadata and constructors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field
from sklearn.base import clone

from src.classification import get_all_classifiers
from src.models.comparison import get_regressors

TaskType = Literal["classification", "regression"]


CLASSIFICATION_MODEL_DETAILS: Dict[str, Dict[str, str]] = {
    "Logistic (L2)": {
        "family": "Linear",
        "description": "L2-regularized logistic regression baseline with balanced class weighting.",
    },
    "Logistic (L1)": {
        "family": "Linear",
        "description": "Sparse L1-regularized logistic regression for embedded feature selection.",
    },
    "Logistic (ElasticNet)": {
        "family": "Linear",
        "description": "ElasticNet logistic regression balancing L1 sparsity and L2 stability.",
    },
    "Logistic (MCP)": {
        "family": "Sparse Non-Convex",
        "description": "Logistic regression with MCP penalty for near-unbiased sparse coefficients.",
    },
    "Logistic (SCAD)": {
        "family": "Sparse Non-Convex",
        "description": "Logistic regression with SCAD penalty to reduce shrinkage bias on strong signals.",
    },
    "SVM (Linear)": {
        "family": "SVM",
        "description": "Linear-kernel support vector machine for max-margin linear decision boundaries.",
    },
    "SVM (RBF)": {
        "family": "SVM",
        "description": "RBF-kernel support vector machine for non-linear class separation.",
    },
    "SVM (Poly)": {
        "family": "SVM",
        "description": "Polynomial-kernel support vector machine capturing higher-order boundaries.",
    },
    "KNN (k=3)": {
        "family": "Instance-Based",
        "description": "K-nearest-neighbors classifier with local neighborhoods (k=3).",
    },
    "KNN (k=5)": {
        "family": "Instance-Based",
        "description": "Distance-weighted KNN classifier with k=5 for smoother local voting.",
    },
    "KNN (k=7)": {
        "family": "Instance-Based",
        "description": "Distance-weighted Manhattan KNN classifier with broader neighborhoods (k=7).",
    },
    "Decision Tree": {
        "family": "Tree",
        "description": "Single interpretable decision tree classifier with depth constraint.",
    },
    "Extra Trees": {
        "family": "Ensemble Tree",
        "description": "Extremely randomized trees ensemble to reduce variance with randomized splits.",
    },
    "Random Forest": {
        "family": "Ensemble Tree",
        "description": "Bagged decision-tree ensemble for robust non-linear classification.",
    },
    "Gradient Boosting": {
        "family": "Boosting",
        "description": "Stage-wise additive boosting ensemble optimized for classification error reduction.",
    },
    "AdaBoost": {
        "family": "Boosting",
        "description": "Adaptive boosting ensemble emphasizing previously misclassified samples.",
    },
    "MLP": {
        "family": "Neural Network",
        "description": "Multi-layer perceptron classifier for non-linear feature interactions.",
    },
    "Naive Bayes": {
        "family": "Probabilistic",
        "description": "Gaussian Naive Bayes classifier assuming conditional feature independence.",
    },
}


REGRESSION_MODEL_DETAILS: Dict[str, Dict[str, str]] = {
    "OLS (Baseline)": {
        "family": "Linear",
        "description": "Ordinary least squares baseline regression without regularization.",
    },
    "Ridge (L2)": {
        "family": "Linear Regularized",
        "description": "L2-regularized linear regression to stabilize correlated coefficients.",
    },
    "Lasso (L1 Sparse)": {
        "family": "Linear Regularized",
        "description": "L1-regularized sparse regression with embedded feature selection.",
    },
    "ElasticNet (L1+L2)": {
        "family": "Linear Regularized",
        "description": "ElasticNet regression combining L1 sparsity and L2 shrinkage.",
    },
    "Huber (Robust)": {
        "family": "Robust",
        "description": "Huber robust regression that down-weights outliers.",
    },
    "SVR (Linear)": {
        "family": "SVM",
        "description": "Linear-kernel support vector regression with margin-based fitting.",
    },
    "SVR (RBF Kernel)": {
        "family": "SVM",
        "description": "RBF-kernel support vector regression for non-linear response surfaces.",
    },
    "KNN (k=3)": {
        "family": "Instance-Based",
        "description": "K-nearest-neighbors regressor with local averaging (k=3).",
    },
    "KNN (k=5, weighted)": {
        "family": "Instance-Based",
        "description": "Distance-weighted KNN regressor with k=5 for smoother local interpolation.",
    },
    "KNN (k=7, manhattan)": {
        "family": "Instance-Based",
        "description": "Distance-weighted Manhattan KNN regressor with broader neighborhoods (k=7).",
    },
    "Random Forest": {
        "family": "Ensemble Tree",
        "description": "Bagged decision-tree regression ensemble for strong non-linear performance.",
    },
    "Gradient Boosting": {
        "family": "Boosting",
        "description": "Stage-wise boosting regressor that iteratively minimizes residual errors.",
    },
}


class ModelDefinition(BaseModel):
    """Validated metadata and constructor hints for one model."""

    name: str = Field(min_length=1)
    task: TaskType
    family: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supports_staged_predictions: bool = False


class ModelRegistry:
    """Task-scoped model templates and metadata used by API services/UI."""

    def __init__(self, random_state: int) -> None:
        self.random_state = random_state
        self._templates: Dict[TaskType, Dict[str, Any]] = {
            "classification": get_all_classifiers(random_state=random_state),
            "regression": get_regressors(random_state=random_state),
        }
        self._definitions: Dict[TaskType, Dict[str, ModelDefinition]] = {
            "classification": self._build_definitions("classification", CLASSIFICATION_MODEL_DETAILS),
            "regression": self._build_definitions("regression", REGRESSION_MODEL_DETAILS),
        }

    def _build_definitions(
        self,
        task: TaskType,
        model_details: Dict[str, Dict[str, str]],
    ) -> Dict[str, ModelDefinition]:
        definitions: Dict[str, ModelDefinition] = {}
        for name, template in self._templates[task].items():
            details = model_details.get(name, {})
            definitions[name] = ModelDefinition(
                name=name,
                task=task,
                family=details.get("family", "Other"),
                description=details.get("description", f"{name} model for {task} task."),
                supports_staged_predictions=hasattr(template, "staged_predict"),
            )
        return definitions

    def list_models(self, task: TaskType) -> List[str]:
        """Return model names for one task in deterministic insertion order."""
        return list(self._templates[task].keys())

    def list_model_payload(self, task: TaskType) -> List[Dict[str, Any]]:
        """Return UI/API payload with model metadata."""
        return [definition.model_dump() for definition in self._definitions[task].values()]

    def create_model(self, task: TaskType, model_name: str) -> Any:
        """Create a fresh model instance from a stored template."""
        template = self._templates[task].get(model_name)
        if template is None:
            raise ValueError(f"Unknown model '{model_name}' for task '{task}'.")

        try:
            return clone(template)
        except Exception:  # pragma: no cover - clone failures depend on estimator implementation
            return deepcopy(template)

    def get_definition(self, task: TaskType, model_name: str) -> ModelDefinition:
        """Return validated metadata for one model."""
        definition = self._definitions[task].get(model_name)
        if definition is None:
            raise ValueError(f"Unknown model '{model_name}' for task '{task}'.")
        return definition
