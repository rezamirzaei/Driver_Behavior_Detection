"""Models module exports."""

from src.models.comparison import (
    compare_classifiers,
    compare_regressors,
    get_classifiers,
    get_knn_classifiers,
    get_knn_regressors,
    get_regressors,
)
from src.models.evaluation import (
    ClassificationEvaluator,
    QuantileRegressor,
    RegressionEvaluator,
    compute_interval_coverage,
    compute_interval_width,
    compute_regression_metrics,
    evaluate_classifier,
    evaluate_regressor,
    train_and_evaluate_classifier,
    train_and_evaluate_regressor,
)
from src.models.robust_sparse import (
    HuberL1Regressor,
    HuberMCPRegressor,
    HuberSCADRegressor,
    MCPRegressor,
    SCADRegressor,
    get_sparse_robust_regressors,
)
from src.models.trainer import ModelTrainer, train_model

# Optional: CNN models (requires PyTorch)
try:
    from src.models.cnn import CNNClassifier, CNNClassifierRaw, plot_cnn_training_history

    _HAS_CNN = True
except ImportError:
    CNNClassifier = None  # type: ignore[assignment,misc]
    CNNClassifierRaw = None  # type: ignore[assignment,misc]
    plot_cnn_training_history = None  # type: ignore[assignment]
    _HAS_CNN = False

# Optional: ResNet models (requires PyTorch)
try:
    from src.models.resnet import ResNetClassifier, plot_resnet_training_history

    _HAS_RESNET = True
except ImportError:
    ResNetClassifier = None  # type: ignore[assignment,misc]
    plot_resnet_training_history = None  # type: ignore[assignment]
    _HAS_RESNET = False

# Optional: Simple Neural Network (requires PyTorch)
try:
    from src.models.simple_nn import SimpleNNClassifier, plot_nn_training_history

    _HAS_SIMPLE_NN = True
except ImportError:
    SimpleNNClassifier = None  # type: ignore[assignment,misc]
    plot_nn_training_history = None  # type: ignore[assignment]
    _HAS_SIMPLE_NN = False

__all__ = [
    "ModelTrainer",
    "train_model",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "evaluate_classifier",
    "evaluate_regressor",
    "compute_regression_metrics",
    "train_and_evaluate_regressor",
    "train_and_evaluate_classifier",
    "QuantileRegressor",
    "compute_interval_coverage",
    "compute_interval_width",
    "get_classifiers",
    "get_regressors",
    "get_knn_classifiers",
    "get_knn_regressors",
    "compare_classifiers",
    "compare_regressors",
    # Robust sparse models
    "HuberL1Regressor",
    "SCADRegressor",
    "MCPRegressor",
    "HuberSCADRegressor",
    "HuberMCPRegressor",
    "get_sparse_robust_regressors",
]

if _HAS_CNN:
    __all__.extend(["CNNClassifier", "CNNClassifierRaw", "plot_cnn_training_history"])

if _HAS_RESNET:
    __all__.extend(["ResNetClassifier", "plot_resnet_training_history"])

if _HAS_SIMPLE_NN:
    __all__.extend(["SimpleNNClassifier", "plot_nn_training_history"])
