# Out-Of-Sample-Forecast-Comparison
Seminar Thesis

## Generating a sample dataset

The full CRSP dataset used in the project is too large to include in the
repository.  To facilitate testing, a helper script
`sample_data_generator.py` can create a much smaller Parquet file with a
limited number of pre-sampled windows:

```bash
python sample_data_generator.py [path/to/full/CRSP.parquet]
```

If no path is provided, the script will look for the `CRSP_FULL_PATH`
environment variable and finally fall back to
`data/CRSP 1970-2024.parquet`.

The resulting file is written to `data/crsp_sample.parquet` and is small
enough to commit to the repo.
