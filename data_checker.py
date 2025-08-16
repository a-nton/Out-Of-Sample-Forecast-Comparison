"""Utility module to inspect raw data files for column names, data types, and basic statistics.
Run this script directly to view information about the CRSP and Fama-French datasets.
"""

import os
from pathlib import Path
import pandas as pd

from config import DATA_PATHS
from data_loader import load_ff_factors

# Ensure working directory is repository root
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)


def _inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    """Print columns, dtypes, and mean of numeric columns for a DataFrame."""
    print(f"\n{name} dataset overview")
    print("Columns and data types:")
    print(df.dtypes)
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        print("\nAverage values:")
        print(numeric_cols.mean())


def main() -> None:
    """Load each data file and display column information."""
    try:
        crsp_df = pd.read_parquet(DATA_PATHS['crsp_file'], engine='fastparquet')
        _inspect_dataframe(crsp_df, "CRSP")
    except FileNotFoundError:
        print(f"CRSP data file not found: {DATA_PATHS['crsp_file']}")

    try:
        ff_df = load_ff_factors(DATA_PATHS['ff_factors'])
        _inspect_dataframe(ff_df, "Fama-French factors")
    except FileNotFoundError:
        print(f"Fama-French factors file not found: {DATA_PATHS['ff_factors']}")


if __name__ == "__main__":
    main()
