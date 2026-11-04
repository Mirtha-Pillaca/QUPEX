
from pathlib import Path
from collections import Counter
import ast
import json

import h5py
import numpy as np
import pandas as pd

from modules.datasets import QM9_SCALAR_PROPERTIES


# Version of the QM9 index layout. Bump it whenever the QM9 index
# columns change so cached QM9 indexes are rebuilt (QM7-X caches are
# not affected; see prepare_indexes.index_is_compatible).
QM9_INDEX_VERSION = 2

# Text/identifier columns copied as-is into the optimized index.
QM9_TEXT_COLUMNS = (
    "SMILES",
    "InChi",
    "N_atoms",
)

# All columns copied as-is into the optimized index; the scalar
# properties come from the dataset registry (modules.datasets).
QM9_PROPERTY_COLUMNS = ( *QM9_TEXT_COLUMNS, *QM9_SCALAR_PROPERTIES, )

# QM9 contains only these elements (formula in this order).
QM9_ELEMENTS = ("C", "H", "N", "O", "F")


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _parse_list(value):
    """
    Parse a list-like value stored as a string.

    `Elements` is written as a Python literal (e.g. "['C','H']"), which
    is not valid JSON, and a few numeric lists use non-JSON number
    formats, so fall back to `ast.literal_eval` when JSON fails.
    """
    value = _decode(value)

    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return ast.literal_eval(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return list(value)

    return value


def _formula(element_counts):
    formula = ""

    for symbol in QM9_ELEMENTS:
        count = element_counts.get(symbol, 0)

        if count:
            formula += symbol + (str(count) if count > 1 else "")

    return formula


def _read_column(hdf, name, total):
    """Read the first `total` values of a column in one HDF5 call."""
    return hdf[name][:total]


def build_qm9_indexes(
    hdf,
    opt_index_path="data/qm9_opt.parquet",
    geometry_index_path="data/qm9_geometry.parquet",
    max_molecules=None,
):
    """
    Build lightweight QM9 analysis indexes from the columnar HDF5.
    """

    if "row_count" in hdf.attrs:
        total = int(hdf.attrs["row_count"])
    else:
        total = len(hdf["XYZ_file"])

    if max_molecules is not None:
        total = min(total, int(max_molecules))

    # Read whole columns at once: row-by-row access on gzip-compressed
    # datasets re-decompresses a chunk per value and is very slow.
    xyz_files = _read_column(hdf, "XYZ_file", total)
    elements_col = _read_column(hdf, "Elements", total)
    coordinates_col = _read_column(hdf, "XYZ_Ang", total)

    property_cols = {
        name: _read_column(hdf, name, total)
        for name in QM9_PROPERTY_COLUMNS
        if name in hdf
    }

    opt_rows = []
    geometry_rows = []

    for i in range(total):
        elements = _parse_list(elements_col[i])
        coordinates = _parse_list(coordinates_col[i])
        molecule_id = str(_decode(xyz_files[i]))

        if len(elements) != len(coordinates):
            raise ValueError(
                f"Atom/coordinate count mismatch for {molecule_id}"
            )

        properties = {
            name: _decode(values[i])
            for name, values in property_cols.items()
        }

        element_counts = Counter(elements)
        geometry_id = f"{molecule_id}-opt"

        row = {
            "molecule_id": molecule_id,
            "geometry_id": geometry_id,
            "is_opt": True,
            "dataset": "QM9",
            "n_atoms": len(elements),
            "formula": _formula(element_counts),
            **{
                f"n_{symbol}": int(element_counts.get(symbol, 0))
                for symbol in QM9_ELEMENTS
            },
            "elements": elements,
            "coordinates": coordinates,
            **properties,
        }

        opt_rows.append(row)

        # QM9 has exactly one (optimized) geometry per molecule, so the
        # geometry index carries the same scalar properties.
        geometry_rows.append({
            "molecule_id": molecule_id,
            "dataset": "QM9",
            "geometry_id": geometry_id,
            "is_opt": True,
            "elements": elements,
            "coordinates": coordinates,
            "n_atoms": len(elements),
            **{
                name: properties[name]
                for name in QM9_SCALAR_PROPERTIES
                if name in properties
            },
        })

    opt_df = pd.DataFrame(opt_rows)
    geometry_df = pd.DataFrame(geometry_rows)

    Path(opt_index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(geometry_index_path).parent.mkdir(parents=True, exist_ok=True)

    opt_df.to_parquet(opt_index_path, index=False, compression="zstd")
    geometry_df.to_parquet(
        geometry_index_path, index=False, compression="zstd"
    )

    return {"opt_df": opt_df, "geometry_df": geometry_df}
