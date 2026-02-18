"""
Data loader for EPA Fuel Economy dataset (online).
"""

from typing import Optional

import pandas as pd

from src.core.schemas import Dataset, DatasetInfo
from src.data.sample_models import EPAVehicleSample
from src.data.validation import validate_dataframe_records


class EPADataLoader:
    """Loader for EPA Fuel Economy data."""

    URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"

    # Features to use - all useful vehicle characteristics
    # EXCLUDED (data leakage - derived from target comb08):
    #   - city08, highway08, cityA08, highwayA08 (component MPG)
    #   - co2TailpipeGpm, co2 (derived from MPG)
    #   - fuelCost08, fuelCostA08, youSaveSpend (derived from MPG)
    #   - ghgScore, feScore (derived scores)
    # EXCLUDED (not useful):
    #   - id, createdOn, modifiedOn (metadata)
    KEEP_COLS = [
        # Target
        "comb08",
        # Vehicle identification
        "year",
        "make",
        "model",
        # Vehicle class & body
        "VClass",
        "sCharger",
        "tCharger",
        "atvType",
        # Drivetrain
        "drive",
        "trany",
        "trans_dscr",
        # Engine
        "cylinders",
        "displ",
        "eng_dscr",
        "engId",
        # Fuel
        "fuelType",
        "fuelType1",
        "fuelType2",
        # Electric/Hybrid
        "evMotor",
        "mfrCode",
        "c240Dscr",
        "charge240b",
        "c240bDscr",
        # Range
        "range",
        "rangeCity",
        "rangeHwy",
        "rangeA",
        # Physical
        "hlv",
        "hpv",
        "lv2",
        "lv4",
        "pv2",
        "pv4",  # cargo volumes
        "startStop",
        "phevBlended",
        # Certifications
        "guzzler",
        "phevCity",
        "phevHwy",
        "phevComb",
    ]

    def load(
        self,
        year_min: Optional[int] = 2015,
        year_max: Optional[int] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> Dataset:
        """
        Load EPA Fuel Economy dataset.

        Args:
            year_min: Minimum model year to include.
            year_max: Maximum model year to include.
            sample_size: If specified, randomly sample this many rows.
            random_state: Random seed for sampling.

        Returns:
            Dataset with features and target.
        """
        print(f"Downloading EPA Fuel Economy data from {self.URL}...")

        try:
            df = pd.read_csv(self.URL, low_memory=False)
        except Exception as e:
            raise ConnectionError(f"Failed to download EPA data: {e}")

        # Select relevant columns (only those that exist in the dataset)
        keep_cols = [c for c in self.KEEP_COLS if c in df.columns]
        df = df[keep_cols].copy()
        print(f"   Selected {len(keep_cols)} useful columns")

        # Filter by year
        if year_min:
            df = df[df["year"] >= year_min]
        if year_max:
            df = df[df["year"] <= year_max]

        # Drop rows with missing target
        target_col = "comb08"
        df = df.dropna(subset=[target_col])

        # Drop unrealistic values
        df = df[(df[target_col] > 0) & (df[target_col] < 200)]

        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=random_state)

        df = df.reset_index(drop=True)
        df = validate_dataframe_records(df, EPAVehicleSample, strict=False, context="epa_dataset")

        # Separate features and target
        feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        info = DatasetInfo(
            name="EPA Fuel Economy",
            n_samples=len(df),
            n_features=len(feature_cols),
            feature_names=feature_cols,
            target_name=target_col,
            task_type="regression",
        )

        print(f"Loaded {len(df):,} vehicles.")

        return Dataset(X=X, y=y, feature_names=feature_cols, target_name=target_col, info=info)


def load_epa_fuel_economy(
    year_min: Optional[int] = 2015,
    year_max: Optional[int] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dataset:
    """
    Convenience function to load EPA Fuel Economy dataset.

    Args:
        year_min: Minimum model year.
        year_max: Maximum model year.
        sample_size: Number of samples to return.
        random_state: Random seed.

    Returns:
        Dataset with features and target.
    """
    loader = EPADataLoader()
    return loader.load(year_min=year_min, year_max=year_max, sample_size=sample_size, random_state=random_state)
