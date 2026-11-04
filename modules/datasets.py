# modules/datasets.py

import pandas as pd

# NOTE: modules.loader must never import this module (it would create a
# circular import); the dependency is one-way: datasets -> loader.
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


# ============================================================
# QM9 PROPERTY DEFINITIONS
# ============================================================
#
# Keys are the exact column names of the QM9 HDF5 / index (as written by
# qm9pack). Note the source spellings: "ZPVE_au" (qm9pack's help text
# says "ZOVE_au", but the data column is "ZPVE_au") and
# "Enthalphy_298K_au" (misspelled in the source data itself).
#
# "clustering": whether the property is eligible for clustering.
# Rotational constants have extreme outliers (RotA up to ~6e5 GHz) and
# polarizability tensor components depend on molecular orientation,
# so both are available for analysis but not for clustering.

QM9_PROPERTIES = {

    # --------------------------------------------------------
    # Electronic
    # --------------------------------------------------------
    "HOMO_au": { "name": "HOMO Energy", "unit": "Ha", "category": "Electronic", "description": "Highest occupied molecular orbital energy.", "clustering": True, },

    "LUMO_au": { "name": "LUMO Energy", "unit": "Ha", "category": "Electronic", "description": "Lowest unoccupied molecular orbital energy.", "clustering": True, },

    "HOMO_LUMO_gap_au": { "name": "HOMO–LUMO Gap", "unit": "Ha", "category": "Electronic", "description": "Energy difference between the LUMO and HOMO.", "clustering": True, },

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------
    "Dipole_debye": { "name": "Dipole Moment", "unit": "D", "category": "Response", "description": "Total molecular dipole moment.", "clustering": True, },

    "Polarizability_bohr3": { "name": "Isotropic Polarizability", "unit": "a₀³", "category": "Response", "description": "Isotropic dipole polarizability.", "clustering": True, },

    "R2_bohr2": { "name": "Electronic Spatial Extent", "unit": "a₀²", "category": "Response", "description": "Expectation value of R² (spread of the electron density).", "clustering": True, },

    # --------------------------------------------------------
    # Thermochemistry
    # --------------------------------------------------------
    "ZPVE_au": { "name": "Zero-Point Vibrational Energy", "unit": "Ha", "category": "Thermochemistry", "description": "Harmonic zero-point vibrational energy.", "clustering": True, },

    "InternalEnergy_0K_au": { "name": "Internal Energy (0 K)", "unit": "Ha", "category": "Thermochemistry", "description": "Internal energy at 0 K.", "clustering": True, },

    "InternalEnergy_298K_au": { "name": "Internal Energy (298.15 K)", "unit": "Ha", "category": "Thermochemistry", "description": "Internal energy at 298.15 K.", "clustering": True, },

    "Enthalphy_298K_au": { "name": "Enthalpy (298.15 K)", "unit": "Ha", "category": "Thermochemistry", "description": "Enthalpy at 298.15 K.", "clustering": True, },

    "GibbsFreeEnergy_298K_au": { "name": "Gibbs Free Energy (298.15 K)", "unit": "Ha", "category": "Thermochemistry", "description": "Gibbs free energy at 298.15 K.", "clustering": True, },

    "Heatcapacity_Cv_cal_mol_K": { "name": "Heat Capacity (Cv, 298.15 K)", "unit": "cal/(mol·K)", "category": "Thermochemistry", "description": "Heat capacity at constant volume.", "clustering": True, },

    # --------------------------------------------------------
    # Rotational constants
    # --------------------------------------------------------
    "RotA_GHz": { "name": "Rotational Constant A", "unit": "GHz", "category": "Rotational", "description": "Rotational constant A.", "clustering": False, },

    "RotB_GHz": { "name": "Rotational Constant B", "unit": "GHz", "category": "Rotational", "description": "Rotational constant B.", "clustering": False, },

    "RotC_GHz": { "name": "Rotational Constant C", "unit": "GHz", "category": "Rotational", "description": "Rotational constant C.", "clustering": False, },

    # --------------------------------------------------------
    # Polarizability tensor
    # --------------------------------------------------------
    "A_xx": { "name": "Polarizability Tensor αxx", "unit": "a₀³", "category": "Polarizability tensor", "description": "xx component of the dipole polarizability tensor.", "clustering": False, },

    "A_yy": { "name": "Polarizability Tensor αyy", "unit": "a₀³", "category": "Polarizability tensor", "description": "yy component of the dipole polarizability tensor.", "clustering": False, },

    "A_zz": { "name": "Polarizability Tensor αzz", "unit": "a₀³", "category": "Polarizability tensor", "description": "zz component of the dipole polarizability tensor.", "clustering": False, },

    "A_xy": { "name": "Polarizability Tensor αxy", "unit": "a₀³", "category": "Polarizability tensor", "description": "xy component of the dipole polarizability tensor.", "clustering": False, },

    "A_xz": { "name": "Polarizability Tensor αxz", "unit": "a₀³", "category": "Polarizability tensor", "description": "xz component of the dipole polarizability tensor.", "clustering": False, },

    "A_yz": { "name": "Polarizability Tensor αyz", "unit": "a₀³", "category": "Polarizability tensor", "description": "yz component of the dipole polarizability tensor.", "clustering": False, },
}

# Scalar descriptor columns, in display order.
QM9_SCALAR_PROPERTIES = list(QM9_PROPERTIES)

# Non-descriptor columns that only get a label (not offered as properties).
QM9_LABEL_ONLY_PROPERTIES = {
    "n_atoms": { "name": "Number of Atoms", "unit": "atoms", "category": "Structure", "description": "Number of atoms in the molecule.", "clustering": False, },
}


# ============================================================
# DATASET REGISTRY
# ============================================================
#
# One entry per dataset identity returned by
# `loader.identify_dataset`. Any other identity (None, "Custom HDF5
# dataset", ...) falls back to DEFAULT_DATASET, which reproduces the
# behavior the app had before the registry existed.

DEFAULT_DATASET = "QM7-X"

DATASET_REGISTRY = {

    "QM7-X": {
        "properties": PROPERTY_INFO,
        "scalar_properties": QM7X_SCALAR_PROPERTIES,
        # Every QM7-X scalar property stays selectable for clustering.
        "clustering_properties": QM7X_SCALAR_PROPERTIES,
        "clustering_defaults": [ "mPOL", "eMBD", "eAT", "HLgap", ],
    },

    "QM9": {
        "properties": { **QM9_PROPERTIES, **QM9_LABEL_ONLY_PROPERTIES },
        "scalar_properties": QM9_SCALAR_PROPERTIES,
        "clustering_properties": [ prop for prop, entry in QM9_PROPERTIES.items() if entry["clustering"] ],
        "clustering_defaults": [ "HOMO_LUMO_gap_au", "Polarizability_bohr3", "Dipole_debye", "ZPVE_au", ],
    },
}


def get_dataset_config(dataset=None):
    """Registry entry for a dataset identity (QM7-X for unknown/None)."""

    return DATASET_REGISTRY.get(dataset, DATASET_REGISTRY[DEFAULT_DATASET])


def get_property_info(dataset=None):
    """{property_code: {"name": ..., "unit": ..., ...}} for a dataset."""

    return get_dataset_config(dataset)["properties"]


def get_clustering_defaults(dataset=None):
    """Default clustering properties for a dataset."""

    return list(get_dataset_config(dataset)["clustering_defaults"])


def get_available_properties(df, dataset=None, clustering_only=False):
    """
    Return the scalar descriptor columns of the given dataset
    (QM7-X by default) that are actually present (and numeric) in a
    given index/analysis DataFrame, in a stable, sensible order.

    clustering_only=True restricts the result to properties eligible
    for clustering.
    """

    if df is None or len(df) == 0:
        return []

    config = get_dataset_config(dataset)

    candidates = ( config["clustering_properties"] if clustering_only else config["scalar_properties"] )

    available = []

    for prop in candidates:

        if prop not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[prop]):
            continue

        available.append(prop)

    return available


def get_property_label(prop, with_unit=True, dataset=None):
    """Human-readable label for a property code, e.g. 'PBE0 + MBD energy (eV)'."""

    entry = get_property_info(dataset).get(prop, {})

    name = entry.get("name", prop)
    unit = entry.get("unit")

    if with_unit and unit:
        return f"{name} ({unit})"

    return name
