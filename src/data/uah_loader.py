"""
Data loader for UAH-DriveSet.

This loader extracts aggregate statistics from trip-level data.
Features are derived from the SEMANTIC_ONLINE.txt file which contains
cumulative driving behavior scores and ratios computed during the trip.

Why these features?
-------------------
1. **Aggregate Statistics**: We use trip-level aggregates (final row of SEMANTIC_ONLINE.txt)
   because they represent the complete driving session without requiring fixed window lengths.

2. **Real-world applicability**: In production, you would compute these metrics over
   complete trips or configurable windows. The features are scale-invariant (ratios, scores)
   so they generalize across different trip durations.

3. **No fixed window assumption**: Unlike time-series approaches that require fixed-length
   sequences, these aggregated features work for trips of any duration.

4. **Domain-relevant**: Features like acceleration patterns, lane discipline, and speed
   compliance are established indicators of driving behavior in telematics research.
"""

from pathlib import Path
from typing import List, Literal, Optional
import warnings

import pandas as pd

from src.core.schemas import Dataset, DatasetInfo
from src.data.sample_models import UAHSemanticSummarySample, UAHTripSummarySample
from src.data.validation import validate_record


class UAHDataLoader:
    """
    Loader for UAH-DriveSet data.

    Extracts trip-level aggregate features for driver behavior classification.
    Uses cumulative statistics from SEMANTIC_ONLINE.txt (last row = trip summary).
    """

    def __init__(self, data_dir: str = "data/UAH-DRIVESET-v1"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"UAH-DriveSet not found at {self.data_dir}. "
                "Download from: http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/"
            )

    def load_trip_summary(self, trip_folder: Path) -> Optional[pd.Series]:
        """Load trip summary from SEMANTIC_ONLINE.txt (last row = final summary)."""
        semantic_file = trip_folder / "SEMANTIC_ONLINE.txt"

        if not semantic_file.exists():
            return None

        try:
            # SEMANTIC_ONLINE has columns, last row is final summary
            online_cols = [
                "timestamp",
                "latitude",
                "longitude",
                "score_total_window",
                "score_acc_window",
                "score_brake_window",
                "score_turn_window",
                "score_weave_window",
                "score_drift_window",
                "score_speed_window",
                "score_follow_window",
                "ratio_normal_window",
                "ratio_drowsy_window",
                "ratio_aggressive_window",
                "ratio_distracted_window",
                "score_total",
                "score_accelerations",
                "score_brakings",
                "score_turnings",
                "score_weaving",
                "score_drifting",
                "score_overspeeding",
                "score_following",
                "ratio_normal",
                "ratio_drowsy",
                "ratio_aggressive",
                "ratio_distracted",
            ]

            raw_df = pd.read_csv(semantic_file, sep=r"\s+", header=None)
            if raw_df.empty:
                return None

            # UAH files are expected to contain 27 columns, but some dataset variants
            # can differ slightly; keep parsing resilient.
            usable_cols = min(len(online_cols), raw_df.shape[1])
            df = raw_df.iloc[:, :usable_cols].copy()
            df.columns = online_cols[:usable_cols]

            if len(df) == 0:
                return None

            # Get last row (final summary)
            last_row = df.iloc[-1]

            def _value(name: str, default: float = 0.0) -> float:
                value = pd.to_numeric(last_row.get(name, default), errors="coerce")
                if pd.isna(value):
                    return float(default)
                return float(value)

            def _ratio(name: str) -> float:
                value = _value(name, 0.0)
                # Some dumps encode ratios as percentages instead of [0, 1].
                if value > 1.0 and value <= 100.0:
                    value = value / 100.0
                return float(max(0.0, min(1.0, value)))

            # Select relevant features
            features = {
                "score_total": _value("score_total"),
                "score_accelerations": _value("score_accelerations"),
                "score_brakings": _value("score_brakings"),
                "score_turnings": _value("score_turnings"),
                "score_weaving": _value("score_weaving"),
                "score_drifting": _value("score_drifting"),
                "score_overspeeding": _value("score_overspeeding"),
                "score_following": _value("score_following"),
                "ratio_normal": _ratio("ratio_normal"),
                "ratio_drowsy": _ratio("ratio_drowsy"),
                "ratio_aggressive": _ratio("ratio_aggressive"),
            }

            validated = validate_record(
                features,
                UAHSemanticSummarySample,
                strict=True,
                context=f"semantic_summary:{trip_folder.name}",
            )
            return pd.Series(validated)

        except Exception as e:
            warnings.warn(f"Failed to load {semantic_file}: {e}")
            return None

    def load(
        self,
        drivers: Optional[List[str]] = None,
        behaviors: Optional[List[str]] = None,
        road_types: Optional[List[str]] = None,
        task: Literal["classification", "regression"] = "classification",
        target_variable: str = "behavior",
        return_driver_info: bool = True,  # Return driver ID for splitting
    ) -> Dataset:
        """
        Load UAH-DriveSet dataset.

        Args:
            drivers: List of drivers to include (e.g., ['D1', 'D2']). None = all.
            behaviors: List of behaviors to include. None = all.
            road_types: List of road types to include. None = all.
            task: 'classification' or 'regression'.
            target_variable: Name of the target column.
            return_driver_info: If True, keeps driver column for driver-based splitting.

        Returns:
            Dataset with features and target (and optionally driver info for splitting).
        """
        if drivers is None:
            drivers = sorted([path.name for path in self.data_dir.glob("D*") if path.is_dir()])

        normalized_behaviors = [behavior.upper() for behavior in behaviors] if behaviors else None
        normalized_road_types = [road_type.upper() for road_type in road_types] if road_types else None

        all_data = []

        for driver in drivers:
            driver_path = self.data_dir / driver
            if not driver_path.exists():
                continue

            for trip_folder in driver_path.iterdir():
                if not trip_folder.is_dir():
                    continue

                # Parse folder name for metadata
                folder_name = trip_folder.name
                parts = folder_name.split("-")

                # Extract behavior and road type
                behavior = None
                road_type = None

                for part in parts:
                    upper_part = part.upper()
                    if upper_part in ["NORMAL", "NORMAL1", "NORMAL2", "AGGRESSIVE", "DROWSY"]:
                        behavior = upper_part.replace("1", "").replace("2", "")
                    elif upper_part in ["MOTORWAY", "SECONDARY"]:
                        road_type = upper_part

                if behavior is None:
                    continue

                # Apply filters
                if normalized_behaviors and behavior not in normalized_behaviors:
                    continue
                if normalized_road_types and road_type not in normalized_road_types:
                    continue

                # Load trip summary
                trip_data = self.load_trip_summary(trip_folder)
                if trip_data is not None:
                    row = validate_record(
                        {
                            **trip_data.to_dict(),
                            "driver": driver,
                            "behavior": behavior,
                            "road_type": road_type,
                        },
                        UAHTripSummarySample,
                        strict=True,
                        context=f"uah_trip_summary:{trip_folder.name}",
                    )
                    all_data.append(row)

        if not all_data:
            raise ValueError("No data found matching the specified criteria.")

        df = pd.DataFrame(all_data)

        # Prepare features and target
        if task == "classification":
            target_col = "behavior"
        else:
            target_col = target_variable

        # Define columns to exclude from features
        metadata_cols = ["driver", "behavior", "road_type"]
        drop_cols = []

        # Always drop the target from features
        if target_col in df.columns:
            drop_cols.append(target_col)

        # If regression, also drop other scores to avoid leakage
        if task == "regression" and "score" in target_col:
            score_cols = [c for c in df.columns if "score" in c and c != target_col]
            drop_cols.extend(score_cols)
            drop_cols.extend(["ratio_normal", "ratio_drowsy", "ratio_aggressive"])  # These are derived from scores

        # Select only numeric features
        feature_cols = [
            c
            for c in df.columns
            if c not in drop_cols and c not in metadata_cols and df[c].dtype in ["float64", "int64"]
        ]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Keep driver info if requested (for proper train/test splitting)
        if return_driver_info:
            X["driver"] = df["driver"].values

        # Handle missing values
        numeric_cols = [c for c in X.columns if c != "driver"]
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        info = DatasetInfo(
            name="UAH-DriveSet",
            n_samples=len(df),
            n_features=len(feature_cols),
            feature_names=feature_cols,
            target_name=target_col,
            task_type=task,
            class_distribution=y.value_counts(normalize=True).to_dict() if task == "classification" else None,
        )

        return Dataset(
            X=X,
            y=y,
            feature_names=feature_cols if not return_driver_info else feature_cols + ["driver"],
            target_name=target_col,
            info=info,
        )


def load_uah_driveset(
    data_dir: str = "data/UAH-DRIVESET-v1",
    drivers: Optional[List[str]] = None,
    behaviors: Optional[List[str]] = None,
    road_types: Optional[List[str]] = None,
    task: Literal["classification", "regression"] = "classification",
    target_variable: str = "behavior",
    return_driver_info: bool = True,
) -> Dataset:
    """
    Convenience function to load UAH-DriveSet.

    Args:
        data_dir: Path to UAH-DRIVESET-v1 directory.
        drivers: List of drivers to include.
        behaviors: List of behaviors to include.
        road_types: List of road types to include.
        task: 'classification' or 'regression'.
        target_variable: Target column name.
        return_driver_info: If True, includes driver column for proper splitting.
    """
    loader = UAHDataLoader(data_dir)
    return loader.load(
        drivers=drivers,
        behaviors=behaviors,
        road_types=road_types,
        task=task,
        target_variable=target_variable,
        return_driver_info=return_driver_info,
    )
