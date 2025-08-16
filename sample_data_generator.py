"""Utility to create a small CRSP dataset for testing.

This module allows extracting a subset of the full CRSP dataset that
contains a limited number of valid samples.  The resulting Parquet file
is small enough to commit to version control (\u003c25 MB) yet preserves the
structure required by the sampling and modelling pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import SAMPLING_CONFIG, DATA_PATHS
from data_loader import load_crsp_data, load_ff_factors, prepare_analysis_data
from sampling import sample_events_value_weighted


def build_subset_from_samples(samples: Dict[int, List[dict]]) -> pd.DataFrame:
    """Compile unique PERMNO-date rows from sampled events.

    Parameters
    ----------
    samples : dict
        Output of ``sample_events_value_weighted`` where each horizon
        maps to a list of sample dictionaries containing estimation and
        forecast data.

    Returns
    -------
    pandas.DataFrame
        Deduplicated DataFrame containing all rows required for the
        sampled estimation and forecast windows.
    """
    frames = []
    for horizon_samples in samples.values():
        for sample in horizon_samples:
            frames.append(sample["estimation_data"])
            # ``forecast_data`` is a Series; convert to DataFrame
            frames.append(sample["forecast_data"].to_frame().T)

    subset = pd.concat(frames, ignore_index=True)
    subset = subset.drop_duplicates(subset=["PERMNO", "date"])
    subset = subset.sort_values(["PERMNO", "date"]).reset_index(drop=True)
    return subset


def extract_sample_dataset(
    full_crsp_path: str,
    output_path: str = "data/crsp_sample.parquet",
    n_samples: int = 105,
    estimation_window: int = SAMPLING_CONFIG["estimation_window"],
    forecast_horizons: List[int] | None = None,
    random_seed: int | None = SAMPLING_CONFIG.get("random_seed"),
    verbose: bool = True,
) -> pd.DataFrame:
    """Create and save a reduced CRSP dataset with sampled windows.

    This function loads the full CRSP Parquet file and Fama-French
    factor data, draws random samples using the project's standard
    sampling routine, and saves a deduplicated subset of the data to a
    new Parquet file.

    Parameters
    ----------
    full_crsp_path : str
        Path to the full CRSP Parquet file on the user's machine.
    output_path : str, default ``"data/crsp_sample.parquet"``
        Destination for the reduced dataset.
    n_samples : int, default ``105``
        Number of sample windows to draw (includes a buffer above the
        100 used in analysis).
    estimation_window : int
        Length of estimation window in trading days.
    forecast_horizons : list[int], optional
        Forecast horizons to accommodate.  Defaults to the configuration
        value.
    random_seed : int, optional
        Seed for reproducible sampling.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    pandas.DataFrame
        The subset DataFrame that was written to ``output_path``.
    """
    if forecast_horizons is None:
        forecast_horizons = SAMPLING_CONFIG["forecast_horizons"]

    # Load full data
    crsp_df = load_crsp_data(full_crsp_path)
    ff_df = load_ff_factors(DATA_PATHS["ff_factors"])
    merged_df = prepare_analysis_data(crsp_df, ff_df)

    # Draw samples
    samples = sample_events_value_weighted(
        merged_df,
        n_samples=n_samples,
        estimation_window=estimation_window,
        forecast_horizons=forecast_horizons,
        config=SAMPLING_CONFIG,
        random_seed=random_seed,
        verbose=verbose,
    )

    # Compile subset and save
    subset = build_subset_from_samples(samples)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if verbose:
        print(
            f"Saved {len(subset):,} rows "
            f"({subset['PERMNO'].nunique()} stocks) to {output_path} "
            f"[{size_mb:.2f} MB]"
        )

    return subset


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Create a small CRSP sample dataset.")
    parser.add_argument("full_crsp_path", help="Path to the full CRSP Parquet file")
    parser.add_argument("--output", default="data/crsp_sample.parquet", help="Output path")
    parser.add_argument("--n-samples", type=int, default=105, help="Number of sample windows to draw")
    args = parser.parse_args()

    extract_sample_dataset(
        full_crsp_path=args.full_crsp_path,
        output_path=args.output,
        n_samples=args.n_samples,
    )
