"""Features module exports."""

from src.features.analysis import (
    analyze_class_distribution,
    analyze_correlations,
    analyze_driver_distribution,
    analyze_outliers,
    analyze_outliers_dataframe,
    compute_feature_statistics,
    find_high_correlation_pairs,
    generate_data_quality_report,
    get_correlations_with_target,
    get_feature_columns,
    get_score_mapping_info,
    identify_discriminative_features,
    print_outlier_summary,
    print_score_mapping_explanation,
)
from src.features.preprocessing import (
    FeaturePreprocessor,
    TargetEncoder,
    encode_and_scale,
    encode_target,
    engineer_regression_features,
    preprocess_features,
)

__all__ = [
    # Preprocessing
    "FeaturePreprocessor",
    "preprocess_features",
    "TargetEncoder",
    "encode_target",
    "encode_and_scale",
    "engineer_regression_features",
    # Analysis
    "analyze_outliers",
    "analyze_outliers_dataframe",
    "analyze_correlations",
    "analyze_class_distribution",
    "compute_feature_statistics",
    "generate_data_quality_report",
    "get_score_mapping_info",
    "print_score_mapping_explanation",
    "identify_discriminative_features",
    "analyze_driver_distribution",
    "print_outlier_summary",
    "find_high_correlation_pairs",
    "get_feature_columns",
    "get_correlations_with_target",
]
