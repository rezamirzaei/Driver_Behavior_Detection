"""Features module exports."""

from src.features.preprocessing import (
    FeaturePreprocessor,
    preprocess_features,
    TargetEncoder,
    encode_target,
    encode_and_scale,
    engineer_regression_features,
)

from src.features.analysis import (
    analyze_outliers,
    analyze_outliers_dataframe,
    analyze_correlations,
    analyze_class_distribution,
    compute_feature_statistics,
    generate_data_quality_report,
    get_score_mapping_info,
    print_score_mapping_explanation,
    identify_discriminative_features,
    analyze_driver_distribution,
    print_outlier_summary,
    find_high_correlation_pairs,
    get_feature_columns,
    get_correlations_with_target,
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
