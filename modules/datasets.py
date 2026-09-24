# modules/datasets.py

import pandas as pd

from modules.loader import ( QM7X_SCALAR_PROPERTIES, qm7x_property_table, )

def _build_property_info():
    """
    Build a {property_code: {"name": ..., "unit": ...}} lookup
    from the QM7-X property reference table, restricted to the
    scalar properties the app actually indexes.
    """

    table = qm7x_property_table()

    info = {}

    for _, row in table.iterrows():

        code = row["Property"]

        info[code] = { "name": row["Meaning"], "unit": row["Units"], }

    return info


# Public lookup: PROPERTY_INFO["ePBE0+MBD"] -> {"name": ..., "unit": ...}
PROPERTY_INFO = _build_property_info()


def get_available_properties(df):
    """
    Return the QM7-X scalar descriptor columns that are actually
    present (and numeric) in a given index/analysis DataFrame,
    in a stable, sensible order.
    """

    if df is None or len(df) == 0:
        return []

    available = []

    for prop in QM7X_SCALAR_PROPERTIES:

        if prop not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[prop]):
            continue

        available.append(prop)

    return available


def get_property_label(prop, with_unit=True):
    """Human-readable label for a property code, e.g. 'PBE0 + MBD energy (eV)'."""

    entry = PROPERTY_INFO.get(prop, {})

    name = entry.get("name", prop)
    unit = entry.get("unit")

    if with_unit and unit:
        return f"{name} ({unit})"

    return name
