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
from typing import List, Optional
import warnings

import pandas as pd

from src.core.schemas import Dataset, DatasetInfo


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
                'timestamp', 'latitude', 'longitude',
                'score_total_window', 'score_acc_window', 'score_brake_window',
                'score_turn_window', 'score_weave_window', 'score_drift_window',
                'score_speed_window', 'score_follow_window',
                'ratio_normal_window', 'ratio_drowsy_window', 'ratio_aggressive_window',
                'ratio_distracted_window',
                'score_total', 'score_accelerations', 'score_brakings', 'score_turnings',
                'score_weaving', 'score_drifting', 'score_overspeeding', 'score_following',
                'ratio_normal', 'ratio_drowsy', 'ratio_aggressive', 'ratio_distracted'
            ]

            df = pd.read_csv(semantic_file, sep=r'\s+', names=online_cols, header=None)

            if len(df) == 0:
                return None

            # Get last row (final summary)
            last_row = df.iloc[-1]

            # Select relevant features
            features = {
                'score_total': last_row.get('score_total', 0),
                'score_accelerations': last_row.get('score_accelerations', 0),
                'score_brakings': last_row.get('score_brakings', 0),
                'score_turnings': last_row.get('score_turnings', 0),
                'score_weaving': last_row.get('score_weaving', 0),
                'score_drifting': last_row.get('score_drifting', 0),
                'score_overspeeding': last_row.get('score_overspeeding', 0),
                'score_following': last_row.get('score_following', 0),
                'ratio_normal': last_row.get('ratio_normal', 0),
                'ratio_drowsy': last_row.get('ratio_drowsy', 0),
                'ratio_aggressive': last_row.get('ratio_aggressive', 0),
            }

            return pd.Series(features)

        except Exception as e:
            warnings.warn(f"Failed to load {semantic_file}: {e}")
            return None

    def load(
        self,
        drivers: Optional[List[str]] = None,
        behaviors: Optional[List[str]] = None,
        road_types: Optional[List[str]] = None,
        task: str = "classification",
        target_variable: str = "behavior",
        return_driver_info: bool = True  # Return driver ID for splitting
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
            drivers = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

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
                parts = folder_name.split('-')

                # Extract behavior and road type
                behavior = None
                road_type = None

                for part in parts:
                    if part in ['NORMAL', 'NORMAL1', 'NORMAL2', 'AGGRESSIVE', 'DROWSY']:
                        behavior = part.replace('1', '').replace('2', '')
                    elif part in ['MOTORWAY', 'SECONDARY']:
                        road_type = part

                if behavior is None:
                    continue

                # Apply filters
                if behaviors and behavior not in behaviors:
                    continue
                if road_types and road_type not in road_types:
                    continue

                # Load trip summary
                trip_data = self.load_trip_summary(trip_folder)
                if trip_data is not None:
                    trip_data['driver'] = driver
                    trip_data['behavior'] = behavior
                    trip_data['road_type'] = road_type
                    all_data.append(trip_data)

        if not all_data:
            raise ValueError("No data found matching the specified criteria.")

        df = pd.DataFrame(all_data)

        # Prepare features and target
        if task == "classification":
             target_col = "behavior"
        else:
             target_col = target_variable

        # Define columns to exclude from features
        metadata_cols = ['driver', 'behavior', 'road_type']
        drop_cols = []

        # Always drop the target from features
        if target_col in df.columns:
            drop_cols.append(target_col)

        # If regression, also drop other scores to avoid leakage
        if task == "regression" and "score" in target_col:
             score_cols = [c for c in df.columns if "score" in c and c != target_col]
             drop_cols.extend(score_cols)
             drop_cols.extend(['ratio_normal', 'ratio_drowsy', 'ratio_aggressive'])  # These are derived from scores

        # Select only numeric features
        feature_cols = [c for c in df.columns
                       if c not in drop_cols
                       and c not in metadata_cols
                       and df[c].dtype in ['float64', 'int64']]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Keep driver info if requested (for proper train/test splitting)
        if return_driver_info:
            X['driver'] = df['driver'].values

        # Handle missing values
        numeric_cols = [c for c in X.columns if c != 'driver']
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        info = DatasetInfo(
            name="UAH-DriveSet",
            n_samples=len(df),
            n_features=len(feature_cols),
            feature_names=feature_cols,
            target_name=target_col,
            task_type=task,
            class_distribution=y.value_counts(normalize=True).to_dict() if task == "classification" else None
        )

        return Dataset(
            X=X,
            y=y,
            feature_names=feature_cols if not return_driver_info else feature_cols + ['driver'],
            target_name=target_col,
            info=info
        )


def load_uah_driveset(
    data_dir: str = "data/UAH-DRIVESET-v1",
    drivers: Optional[List[str]] = None,
    behaviors: Optional[List[str]] = None,
    road_types: Optional[List[str]] = None,
    task: str = "classification",
    target_variable: str = "behavior",
    return_driver_info: bool = True
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
        return_driver_info=return_driver_info
    )
