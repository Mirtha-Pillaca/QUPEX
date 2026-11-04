
from pathlib import Path
import json
import os

import h5py
import numpy as np
import pandas as pd
import qm9pack


OUTPUT_PATH = Path("data/qm9.hdf5")


def serialize_value(value):
    """Convert a value into a UTF-8-compatible string."""
    if value is None:
        return ""

    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(
            value.tolist() if isinstance(value, np.ndarray) else value
        )

    if pd.isna(value):
        return ""

    return str(value)


def main(output_path=OUTPUT_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = qm9pack.get_data("qm9")

    if df is None or df.empty:
        raise ValueError("qm9pack returned no QM9 data.")

    with h5py.File(output_path, "w") as hdf:
        hdf.attrs["dataset_name"] = "QM9"
        hdf.attrs["source"] = "qm9pack"
        hdf.attrs["row_count"] = len(df)
        hdf.attrs["format_version"] = 1

        for column in df.columns:
            series = df[column]

            if pd.api.types.is_numeric_dtype(series.dtype):
                values = series.to_numpy()

                hdf.create_dataset(
                    column,
                    data=values,
                    compression="gzip",
                    compression_opts=4,
                )
            else:
                values = [
                    serialize_value(value)
                    for value in series
                ]

                hdf.create_dataset(
                    column,
                    data=np.asarray(values, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression="gzip",
                    compression_opts=4,
                )

    print(f"Saved {len(df):,} molecules to {output_path}")
    print(f"Columns: {len(df.columns)}")


def ensure_default_qm9_hdf5(output_path=OUTPUT_PATH):
    """
    Return the path to the default QM9 HDF5, building it with qm9pack
    only if it does not exist yet. An existing file is never
    overwritten; the build goes to a temporary ".part" file that is
    renamed into place only after it completes.
    """
    output_path = Path(output_path)

    if output_path.exists():
        return str(output_path)

    partial_path = output_path.with_name(output_path.name + ".part")

    try:
        main(output_path=partial_path)

        if output_path.exists():
            raise FileExistsError(f"{output_path} appeared during the build.")

        os.replace(partial_path, output_path)

    finally:
        partial_path.unlink(missing_ok=True)

    return str(output_path)


if __name__ == "__main__":
    main()
