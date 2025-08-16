from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


def main() -> None:
    data_path = Path("data") / "CRSP 1970-2024.parquet"
    output_path = Path("sample_crsp.parquet")

    dataset = ds.dataset(data_path, format="parquet")
    permnos = dataset.to_table(columns=["PERMNO"])["PERMNO"].unique().to_pylist()

    tables = []
    total_bytes = 0
    for permno in permnos:
        table = dataset.to_table(filter=ds.field("PERMNO") == permno)
        if total_bytes + table.nbytes > MAX_FILE_SIZE:
            break
        tables.append(table)
        total_bytes += table.nbytes

    if not tables:
        print("No data written: dataset is empty or size limit too small.")
        return

    combined = pa.concat_tables(tables)
    pq.write_table(combined, output_path)

    while output_path.stat().st_size > MAX_FILE_SIZE and len(tables) > 1:
        tables.pop()
        combined = pa.concat_tables(tables)
        pq.write_table(combined, output_path)

    final_size = output_path.stat().st_size / (1024 ** 2)
    print(
        f"Saved {len(tables)} PERMNOs to {output_path} "
        f"({final_size:.2f} MB)"
    )


if __name__ == "__main__":
    main()
