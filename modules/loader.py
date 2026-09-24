import inspect
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import streamlit as st
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import lzma
import shutil
from urllib.parse import urlparse


# Bytes of raw chunk cache per open dataset/file (default is 1 MiB).
HDF5_CHUNK_CACHE_BYTES = 512 * 1024 * 1024  # 512 MB

# Number of chunk slots in the cache 
HDF5_CHUNK_CACHE_SLOTS = 1_000_003

# Default number of worker processes for building the full geometry index. 
DEFAULT_INDEX_WORKERS = min(8, max(1, (os.cpu_count() or 4) - 1))

# ============================================================
# HDF5 OPENING
# ============================================================

@st.cache_resource(show_spinner=False)
def open_hdf5(path: str):
    """
    Open a local HDF5 file without loading it into RAM.
    """

    if not os.path.exists(path):
        raise FileNotFoundError( f"File not found: {path}" )
    try:
        return h5py.File( path, "r", rdcc_nbytes=HDF5_CHUNK_CACHE_BYTES, rdcc_nslots=HDF5_CHUNK_CACHE_SLOTS, )

    except TypeError:

        return h5py.File(path, "r")


def _open_hdf5_worker(path: str):
    """
    Open a fresh, independent read-only handle to the same file.
    """

    try:

        return h5py.File( path, "r", rdcc_nbytes=HDF5_CHUNK_CACHE_BYTES, rdcc_nslots=HDF5_CHUNK_CACHE_SLOTS, )

    except TypeError:

        return h5py.File(path, "r")


def _normalize_zenodo_url(url: str) -> str:
    """
    Normalize a user-supplied Zenodo URL before it is ever used to
    build a request or a cache key.
    """

    return str(url).strip()


def download_zenodo_file( url: str, output_path: str, ):
    """
    Download a Zenodo file directly to disk.
    """

    url = _normalize_zenodo_url(url)

    output_path = Path(output_path)

    output_path.parent.mkdir( parents=True, exist_ok=True, )

    partial_path = output_path.with_name( output_path.name + ".part" )

    try:

        with requests.get(
            url,
            stream=True,
            # (connect timeout, read timeout). 
            timeout=(15, 120),
        ) as response:

            response.raise_for_status()

            with open(partial_path, "wb") as f:

                for chunk in response.iter_content( chunk_size=1024 * 1024 ):

                    if chunk:
                        f.write(chunk)

        os.replace(partial_path, output_path)

    except BaseException:

        # Clean up a truncated partial download so a retry doesn't mistake it for a completed one.
        try:
            partial_path.unlink(missing_ok=True)
        except Exception:
            pass

        raise

    return str(output_path)


def ensure_zenodo_file_cached( url: str, cache_dir="data", filename=None, ):
    """
    Ensure a file referenced by a direct-download Zenodo URL exists
    in a local cache directory, downloading it once if it is missing.
    """

    url = _normalize_zenodo_url(url)

    cache_dir = Path(cache_dir)

    cache_dir.mkdir( parents=True, exist_ok=True, )

    if filename is None:

        url_path = urlparse(url).path

        filename = Path(url_path).name

        if not filename:
            raise ValueError( "Could not determine the filename from the Zenodo URL." )

    local_path = cache_dir / filename

    if not local_path.exists():

        with st.spinner(
            f"Downloading {filename} from Zenodo (first run only; "
            f"cached locally afterwards)..."
        ):

            download_zenodo_file( url, str(local_path), )

    return str(local_path)


@st.cache_resource(show_spinner=False)
def open_zenodo_hdf5( url: str, cache_dir="data", ):
    """Download a Zenodo HDF5 or .xz-compressed HDF5 and open it."""

    url = _normalize_zenodo_url(url)

    cache_dir = Path(cache_dir)

    cache_dir.mkdir( parents=True, exist_ok=True, )

    # ========================================================
    # GET FILENAME FROM URL
    # ========================================================

    url_path = urlparse(url).path

    filename = Path(url_path).name

    filename_lower = filename.lower()

    if not filename:
        raise ValueError( "Could not determine the filename from the Zenodo URL." )

    # ========================================================
    # DIRECT HDF5
    # ========================================================

    if filename_lower.endswith( (".h5", ".hdf5") ):

        local_path = cache_dir / filename

        if not local_path.exists():

            with st.spinner( "Downloading HDF5 from Zenodo..." ):

                download_zenodo_file( url, str(local_path), )

        return h5py.File( local_path, "r", )

    # ========================================================
    # COMPRESSED .XZ
    # ========================================================

    if filename_lower.endswith(".xz"):

        compressed_path = cache_dir / filename

        # Remove ".xz"
        hdf5_filename = filename[:-3]

        hdf5_path = cache_dir / hdf5_filename

        # ----------------------------------------------------
        # Download .xz
        # ----------------------------------------------------

        if not compressed_path.exists():

            with st.spinner( "Downloading dataset from Zenodo..." ):

                download_zenodo_file( url, str(compressed_path), )

        # ----------------------------------------------------
        # Decompress .xz
        # ----------------------------------------------------

        if not hdf5_path.exists():

            with st.spinner(
                "Decompressing dataset..." ):

                with lzma.open( compressed_path, "rb", ) as source:

                    with open( hdf5_path, "wb", ) as destination:

                        shutil.copyfileobj( source, destination, length=1024 * 1024, )

        # ----------------------------------------------------
        # Open decompressed HDF5
        # ----------------------------------------------------

        return h5py.File( hdf5_path, "r", )

    # ========================================================
    # UNSUPPORTED FORMAT
    # ========================================================

    raise ValueError(
        "Unsupported Zenodo file format. "
        "Please provide an .h5, .hdf5, or .xz file."
    )

# ============================================================
# HDF5 INFORMATION
# ============================================================

def _sorted_molecule_keys(hdf):
    """Return molecule IDs sorted numerically when possible."""

    keys = list(hdf.keys())

    try:

        return sorted( keys, key=lambda x: int(x), )

    except (ValueError, TypeError):

        return sorted(keys)


# ============================================================
# DATASET IDENTIFICATION
# ============================================================

_QM7X_GEOMETRY_ID_PATTERN = re.compile( r"^geom-m\d+-i\d+-c\d+(-opt)?$" )

_QM7X_SIGNATURE_PROPERTIES = { "eMBD", "ePBE0", "ePBE0+MBD", "atPOL", "mC6", "mPOL", "HLgap", "DIP", }

# Minimum number of the properties above that must be present on a
# sampled geometry before it counts as a QM7-X-shaped match.
_QM7X_MIN_PROPERTY_HITS = 3


def identify_dataset(hdf, sample_molecules=5):
    """
    Best-effort scientific identification of an opened HDF5 dataset.
    """

    try:
        molecule_keys = _sorted_molecule_keys(hdf)
    except Exception:
        return None

    if not molecule_keys:
        return None

    inspected = 0
    geometry_id_hits = 0
    property_hits = 0

    for mol_key in molecule_keys[:sample_molecules]:

        try:
            mol_group = hdf[mol_key]
            geometry_keys = list(mol_group.keys())
        except Exception:
            continue

        if not geometry_keys:
            continue

        inspected += 1

        if any( _QM7X_GEOMETRY_ID_PATTERN.match(str(gid).lower()) for gid in geometry_keys ):
            geometry_id_hits += 1

        try:

            geom_keys = set( mol_group[geometry_keys[0]].keys() )

        except Exception:

            geom_keys = set()

        if len(
            geom_keys & _QM7X_SIGNATURE_PROPERTIES ) >= _QM7X_MIN_PROPERTY_HITS:

            property_hits += 1

    if inspected == 0:
        return None

    if geometry_id_hits == inspected and property_hits == inspected:
        return "QM7-X"

    return None


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def get_geometry_keys(mol_group):
    """
    Return all geometry-group keys for one molecule.
    """

    return [key for key in mol_group.keys() if isinstance(mol_group[key], h5py.Group)]


def get_opt_geometry(mol_group):
    """
    Return the optimized (-opt) geometry.

    Returns:
        (geometry_id, geometry_group)

    Falls back to the first available geometry if no
    '-opt' geometry exists.
    """

    geometry_keys = get_geometry_keys( mol_group )

    if not geometry_keys:

        return None, None

    opt_keys = [key for key in geometry_keys if str(key).endswith("-opt")]

    key = ( opt_keys[0] if opt_keys else geometry_keys[0] )
    return key, mol_group[key]


def is_optimized_geometry(geometry_id):
    """Return True when a geometry ID represents an optimized geometry."""

    return str( geometry_id ).lower().endswith("-opt")


def geometry_type_from_id(geometry_id):
    """
    Return a human-readable geometry category based on
    the geometry identifier.

    This does NOT infer chemical geometry from coordinates.
    It only interprets the HDF5 geometry ID.
    """

    name = str( geometry_id ).lower()

    if name.endswith("-opt"):
        return "Optimized"

    return "Non-equilibrium"


# ============================================================
# GEOMETRY ID CLASSIFICATION
# ============================================================

def classify_geometry_id(geometry_id):
    """
    Parse a QM7-X geometry identifier.

    Example
    -------
    Geom-m2682-i4-c3-opt

    Returns
    -------
    dict
        Parsed geometry metadata.
    """

    geometry_id = str(geometry_id)

    parts = geometry_id.split("-")

    result = {
        "geometry_id": geometry_id,
        "is_opt": geometry_id.endswith("-opt"),
        "molecule_token": None,
        "isomer_token": None,
        "configuration_token": None,
    }

    for part in parts:

        if part.startswith("m") and part[1:].isdigit():

            result["molecule_token"] = part

        elif part.startswith("i") and part[1:].isdigit():

            result["isomer_token"] = part

        elif part.startswith("c") and part[1:].isdigit():

            result["configuration_token"] = part

    return result


# ============================================================
# GEOMETRY STATISTICS
# ============================================================


def get_geometry_statistics_from_index(
    geometry_df,
    opt_df=None,
    n_molecules_total=None,
):
    """
    Vectorized equivalent of `get_geometry_statistics`, computed
    entirely from the already-built Parquet indexes.
    """

    has_tokens = (
        "molecule_token" in geometry_df.columns
        and "isomer_token" in geometry_df.columns
        and "configuration_token" in geometry_df.columns
    )

    if not has_tokens:

        parsed = geometry_df["geometry_id"].apply(classify_geometry_id)

        geometry_df = geometry_df.assign(
            molecule_token=parsed.apply(lambda p: p["molecule_token"]),
            isomer_token=parsed.apply(lambda p: p["isomer_token"]),
            configuration_token=parsed.apply(
                lambda p: p["configuration_token"]
            ),
        )

    total_geometries = int(len(geometry_df))

    is_opt = geometry_df["is_opt"].astype(bool)

    optimized_geometries = int(is_opt.sum())
    nonoptimized_geometries = total_geometries - optimized_geometries

    per_molecule_counts = geometry_df.groupby("molecule_id").size()

    molecules_with_geometry = int(per_molecule_counts.shape[0])

    molecules_without_geometry = (
        max(int(n_molecules_total) - molecules_with_geometry, 0)
        if n_molecules_total is not None
        else 0
    )

    molecules_with_multiple_geometries = int((per_molecule_counts > 1).sum())

    opt_per_molecule_counts = ( geometry_df.loc[is_opt] .groupby("molecule_id") .size() )

    molecules_with_opt = int((opt_per_molecule_counts > 0).sum())
    molecules_without_opt = molecules_with_geometry - molecules_with_opt

    molecules_with_multiple_optimized_geometries = int( (opt_per_molecule_counts > 1).sum() )

    geometry_counts = per_molecule_counts.tolist()

    if geometry_counts:
        average_geometries = total_geometries / len(geometry_counts)
        min_geometries = int(min(geometry_counts))
        max_geometries = int(max(geometry_counts))
    else:
        average_geometries = 0.0
        min_geometries = 0
        max_geometries = 0

    geometry_id_counts = ( geometry_df["geometry_id"].value_counts().to_dict() )

    molecule_token_counts = ( geometry_df["molecule_token"].dropna().value_counts().to_dict() )

    isomer_token_counts = ( geometry_df["isomer_token"].dropna().value_counts().to_dict() )

    configuration_token_counts = ( geometry_df["configuration_token"].dropna().value_counts().to_dict() )

    optimized_geometry_ids = geometry_df.loc[is_opt, "geometry_id"].tolist()

    nonoptimized_geometry_ids = geometry_df.loc[ ~is_opt, "geometry_id" ].tolist()

    return {
        "molecules_with_geometry": molecules_with_geometry,
        "molecules_without_geometry": molecules_without_geometry,
        "molecules_with_opt": molecules_with_opt,
        "molecules_without_opt": molecules_without_opt,
        "molecules_with_multiple_geometries": molecules_with_multiple_geometries,
        "molecules_with_multiple_optimized_geometries":
            molecules_with_multiple_optimized_geometries,
        "total_geometries": total_geometries,
        "optimized_geometries": optimized_geometries,
        "nonoptimized_geometries": nonoptimized_geometries,
        # Compatibility key -- see get_geometry_statistics().
        "nonequilibrium_geometries": nonoptimized_geometries,
        "average_geometries": average_geometries,
        "min_geometries": min_geometries,
        "max_geometries": max_geometries,
        "geometry_counts": geometry_counts,
        "geometry_id_counts": geometry_id_counts,
        "molecule_token_counts": molecule_token_counts,
        "isomer_token_counts": isomer_token_counts,
        "configuration_token_counts": configuration_token_counts,
        "optimized_geometry_ids": optimized_geometry_ids,
        "nonoptimized_geometry_ids": nonoptimized_geometry_ids,
    }


# ============================================================
# BUILD QM7-X INDEX
# ============================================================

QM7X_SCALAR_PROPERTIES = [
    "eMBD",
    "ePBE0",
    "ePBE0+MBD",
    "eAT",
    "eTS",
    "eNN",
    "eKIN",
    "eNE",
    "eEE",
    "eXC",
    "eX",
    "eC",
    "eXX",
    "eH",
    "eL",
    "HLgap",
    "DIP",
    "mC6",
    "mPOL",
]


def _geometries_for_mode(mol_group, geometry_mode):
    """Select the (geometry_id, geometry_group) pairs for one molecule."""

    if geometry_mode == "opt":

        geometry_id, geom = get_opt_geometry(mol_group)

        return ( [(geometry_id, geom)] if geom is not None else [] )

    geometry_keys = get_geometry_keys(mol_group)

    if geometry_mode == "nonequilibrium":

        geometry_keys = [key for key in geometry_keys if not is_optimized_geometry(key)]

    # mol_group.get() avoids raising on a race/edge case 
    return [(key, mol_group[key]) for key in geometry_keys]


# ============================================================
# ATOM-COUNT FALLBACK (non-QM7-X schemas)
# ============================================================

_ATOM_COUNT_FALLBACK_KEYS = (
    "atomic_numbers", "numbers", "Z", "species", "atom_types",
    "TBchg", "charges", "partial_charges", "mulliken_charges",
)

_ATOM_COORDINATE_FALLBACK_KEYS = ( "atXYZ", "coordinates", "positions", "xyz", "coords", )

def _infer_atom_count(geom, geom_keys):
    """
    Best-effort atom count for one geometry when `atNUM` isn't
    present. Tries a short list of other conventional per-atom
    dataset names (see module-level fallback key lists above), then
    a coordinate array's atom axis. Returns None -- never a guessed
    number -- when nothing usable is found.
    """

    for key in _ATOM_COUNT_FALLBACK_KEYS:

        if key not in geom_keys:
            continue

        try:

            arr = np.asarray(geom[key][:])

            if arr.ndim >= 1 and arr.shape[0] > 0:
                return int(arr.shape[0])

        except Exception:
            continue

    for key in _ATOM_COORDINATE_FALLBACK_KEYS:

        if key not in geom_keys:
            continue

        try:

            arr = np.asarray(geom[key][:])

            if arr.ndim == 2 and 3 in arr.shape:

                # (n_atoms, 3) or (3, n_atoms) -- the atom axis is
                # whichever one isn't the fixed x/y/z axis of length 3.
                atom_axis = 0 if arr.shape[1] == 3 else 1

                return int(arr.shape[atom_axis])

        except Exception:
            continue

    return None


def _extract_geometry_row(mol_key, geometry_id, geom, include_composition=False):
    """
    Build one output row for a single geometry.
    """

    geometry_id = str(geometry_id)

    parsed_id = classify_geometry_id(geometry_id)

    row = {
        "molecule_id": str(mol_key),
        "geometry_id": geometry_id,
        "geometry_type": geometry_type_from_id(geometry_id),
        "is_opt": is_optimized_geometry(geometry_id),
        "molecule_token": parsed_id["molecule_token"],
        "isomer_token": parsed_id["isomer_token"],
        "configuration_token": parsed_id["configuration_token"],
    }


    geom_keys = set(geom.keys())

    atnums = None

    if "atNUM" in geom_keys:

        if include_composition:

            try:
                atnums = np.asarray( geom["atNUM"][:] ).reshape(-1)
                row["n_atoms"] = int(atnums.size)

            except Exception:

                atnums = None

        else:

            try:
                atnums = np.asarray( geom["atNUM"][:] ).reshape(-1)

                row["n_atoms"] = int(atnums.size)

                # Number of hydrogen atoms.
                # This is enough to calculate heavy atoms later:
                # heavy atoms = n_atoms - n_H
                row["n_H"] = int( np.count_nonzero(atnums == 1) )

            except Exception:

                pass


    if "n_atoms" not in row:

        inferred_n_atoms = _infer_atom_count(geom, geom_keys)

        if inferred_n_atoms is not None:
            row["n_atoms"] = inferred_n_atoms


    if include_composition and atnums is not None:

        try:
            row["formula"] = formula_from_atnums(atnums)
        except Exception:
            row["formula"] = None

        element_counts = Counter( ATOMIC_SYMBOLS.get(int(n)) for n in atnums )

        for symbol in ("H", "C", "N", "O", "S", "Cl"):
            row[f"n_{symbol}"] = int(element_counts.get(symbol, 0))

    for prop in QM7X_SCALAR_PROPERTIES:

        if prop not in geom_keys:
            continue

        try:

            value = geom[prop][()]

            row[prop] = float(np.asarray(value).reshape(-1)[0])

        except Exception:
            continue

    return row


def _extract_molecule_rows(hdf, mol_key, geometry_mode):
    """Extract all output rows for one molecule (any geometry_mode)."""

    try:
        mol_group = hdf[mol_key]
    except Exception:
        return []

    # Composition (formula / element counts) is a per-molecule property.
    include_composition = geometry_mode == "opt"

    rows = []

    for geometry_id, geom in _geometries_for_mode(mol_group, geometry_mode):

        rows.append(
            _extract_geometry_row(
                mol_key,
                geometry_id,
                geom,
                include_composition=include_composition,
            )
        )

    return rows


def _build_index_worker_chunk(hdf_path, mol_keys_chunk, geometry_mode):
    """
    Worker entry point for parallel index building.
    """

    local_hdf = _open_hdf5_worker(hdf_path)

    rows = []

    try:

        for mol_key in mol_keys_chunk:

            rows.extend(
                _extract_molecule_rows(
                    local_hdf,
                    mol_key,
                    geometry_mode,
                )
            )

    finally:

        try:
            local_hdf.close()
        except Exception:
            pass

    return rows


def _chunk_list(items, n_chunks):
    """Split a list into n_chunks roughly-equal contiguous pieces."""

    n_chunks = max(1, min(n_chunks, len(items)))

    chunk_size = -(-len(items) // n_chunks)  # ceil division

    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_qm7x_index(
    hdf,
    output_path="data/qm7x_index.parquet",
    max_molecules=None,
    geometry_mode="opt",
    n_workers=DEFAULT_INDEX_WORKERS,
    use_multiprocessing=True,
    min_molecules_for_parallel=200,
):
    """
    Build a lightweight QM7-X Parquet analysis index.

    geometry_mode:
        opt             -> one (optimized) geometry per molecule
        nonequilibrium  -> all non-optimized geometries
        all             -> every geometry

    Performance
    -----------
    For small workloads (e.g. "opt" mode, or a small dataset) this
    runs as a single fast sequential pass -- process start-up cost
    would outweigh any benefit.

    For large workloads (e.g. "all"/"nonequilibrium" geometry mode
    over a big HDF5 file) the molecule keys are split across
    worker *processes*, each holding an independent read-only file
    handle, so HDF5 reads genuinely run in parallel across CPU
    cores instead of being serialized behind h5py's global lock.
    If multiprocessing isn't available in the current environment
    (e.g. some sandboxed/managed deployments), this transparently
    falls back to the sequential path.
    """

    if geometry_mode not in {"opt", "nonequilibrium", "all"}:

        raise ValueError(
            "geometry_mode must be "
            "'opt', 'nonequilibrium', or 'all'."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mol_keys = _sorted_molecule_keys(hdf)

    if max_molecules is not None:
        mol_keys = mol_keys[:int(max_molecules)]

    total = len(mol_keys)

    hdf_path = getattr(hdf, "filename", None)

    run_parallel = (
        use_multiprocessing
        and hdf_path is not None
        and os.path.exists(str(hdf_path))
        and n_workers
        and n_workers > 1
        and total >= min_molecules_for_parallel
    )

    rows = []
    progress = st.progress(0)
    status = st.empty()

    # --------------------------------------------------------
    # PARALLEL PATH (process pool, one HDF5 handle per worker)
    # --------------------------------------------------------

    if run_parallel:

        try:

            chunks = _chunk_list(mol_keys, n_workers)

            start = time.time()
            completed_molecules = 0

            with ProcessPoolExecutor(max_workers=len(chunks)) as executor:

                futures = {
                    executor.submit(
                        _build_index_worker_chunk,
                        str(hdf_path),
                        chunk,
                        geometry_mode,
                    ): len(chunk)
                    for chunk in chunks
                }

                for future in as_completed(futures):

                    rows.extend(future.result())

                    completed_molecules += futures[future]

                    progress.progress( min(completed_molecules / total, 1.0) )

                    status.caption(
                        f"Indexed {completed_molecules:,} / {total:,} "
                        f"molecules "
                        f"({time.time() - start:.1f}s elapsed, "
                        f"{len(chunks)} workers)"
                    )

        except Exception as exc:

            # Fall back to sequential processing rather than fail outright 
            status.caption(
                f"Parallel build unavailable ({exc}); "
                "falling back to a single process."
            )

            rows = []
            run_parallel = False

    # --------------------------------------------------------
    # SEQUENTIAL PATH
    # --------------------------------------------------------

    if not run_parallel:

        for i, mol_key in enumerate(mol_keys):

            rows.extend( _extract_molecule_rows(hdf, mol_key, geometry_mode) )

            if total and (i % 100 == 0 or i == total - 1):

                progress.progress((i + 1) / total)

    progress.empty()
    status.empty()

    result = pd.DataFrame(rows)

    if result.empty:

        raise ValueError( "No valid geometries were found." )

    # zstd gives a good speed/size trade-off for repeated re-reads
    # of the cached index on subsequent app launches.
    try:

        result.to_parquet( output_path, index=False, compression="zstd", )

    except Exception:

        result.to_parquet( output_path, index=False, )

    return result


@st.cache_data(show_spinner=False)
def _load_parquet_index_cached(path, mtime):
    """
    Cached Parquet read, keyed on (path, mtime).
    """
    return pd.read_parquet(path)


def load_qm7x_index( path="data/qm7x_index.parquet", ):
    """
    Load the lightweight Parquet QM7-X index.
    """

    if not os.path.exists( path ):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None

    return _load_parquet_index_cached(path, mtime)


# ============================================================
# MOLECULE LOADING
# ============================================================

def load_molecule( hdf, molecule_id, geometry_id=None, ):
    """
    Load one molecule and one geometry from HDF5.
    """

    molecule_id = str( molecule_id )

    if molecule_id not in hdf:

        raise KeyError( f"Molecule {molecule_id} not found." )

    mol = hdf[ molecule_id ]

    if geometry_id is None:

        geometry_id, geom = ( get_opt_geometry( mol ) )
    else:

        if geometry_id not in mol:

            raise KeyError(
                f"Geometry {geometry_id} "
                f"not found for molecule "
                f"{molecule_id}."
            )

        geom = mol[
            geometry_id
        ]

    if geom is None:

        raise ValueError(
            f"No geometry found for "
            f"molecule {molecule_id}."
        )

    result = {"molecule_id": molecule_id, "geometry_id": geometry_id}

    # Coordinates
    if "atNUM" in geom:

        result[ "atNUM" ] = geom[ "atNUM" ][:]

    if "atXYZ" in geom:

        result[ "atXYZ" ] = geom[ "atXYZ" ][:]


    # ========================================================
    # ATOMIC PROPERTIES
    # ========================================================

    for prop in ("atPOL", "hCHG", "vdwR"):

        if prop in geom:

            try:

                result[prop] = np.asarray(
                    geom[prop][:]
                )

            except Exception:

                pass
    # Scalar properties
    for prop in ( QM7X_SCALAR_PROPERTIES ):

        if prop not in geom:
            continue

        try:

            result[prop] = np.asarray( geom[prop][()] ).squeeze()

        except Exception:
            pass

    # Forces
    for prop in ( "totFOR", "pbe0FOR", "vdwFOR", ):

        if prop in geom:

            result[prop] = ( geom[prop][:] )

    return result


# ============================================================
# RDKit
# ============================================================

def qm7x_to_rdkit(molecule):
    """Convert a QM7-X molecule into an RDKit molecule."""

    atomic_numbers = np.asarray(
        molecule["atNUM"]
    ).reshape(-1)

    coordinates = np.asarray(
        molecule["atXYZ"],
        dtype=float,
    )

    if len(
        atomic_numbers
    ) != len(
        coordinates
    ):

        raise ValueError(
            "Number of atoms and coordinates "
            "do not match."
        )

    mol = Chem.RWMol()

    for atomic_number in atomic_numbers:

        mol.AddAtom(
            Chem.Atom(
                int(
                    atomic_number
                )
            )
        )

    mol = mol.GetMol()

    conformer = Chem.Conformer(
        len(
            atomic_numbers
        )
    )

    for i, xyz in enumerate(
        coordinates
    ):

        conformer.SetAtomPosition(
            i,
            tuple(float(x) for x in xyz),
        )

    mol.AddConformer(
        conformer,
        assignId=True,
    )

    try:

        rdDetermineBonds.DetermineBonds(
            mol,
            charge=0,
        )

    except Exception as e:

        print(
            f"Bond determination warning: {e}"
        )

    return mol


# ============================================================
# 2D MOLECULAR STRUCTURE
# ============================================================

def molecule_2d_image(
    molecule,
    size=(500, 400),
    background="white",
    show_hydrogens=True,
    show_atom_labels=True,
):
    """
    Create a 2D molecular image using RDKit.

    Parameters
    ----------
    molecule : dict
        QM7-X molecule.

    size : tuple
        Image size as (width, height).

    background : str
        "white", "black", or "transparent".

    show_hydrogens : bool
        Show hydrogen atoms.

    show_atom_labels : bool
        Show atom labels.
    """

    # ========================================================
    # QM7-X -> RDKit
    # ========================================================

    mol = qm7x_to_rdkit(
        molecule
    )

    mol = Chem.Mol(
        mol
    )

    # ========================================================
    # HYDROGENS
    # ========================================================

    if show_hydrogens:

        mol = Chem.AddHs(
            mol
        )

    else:

        mol = Chem.RemoveHs(
            mol
        )

    # ========================================================
    # 2D COORDINATES
    # ========================================================

    Chem.rdDepictor.Compute2DCoords(
        mol
    )

    # ========================================================
    # DRAWER
    # ========================================================

    width = int(size[0])
    height = int(size[1])

    drawer = rdMolDraw2D.MolDraw2DCairo(
        width,
        height,
    )

    options = drawer.drawOptions()

    # ========================================================
    # BACKGROUND
    # ========================================================

    if background == "black":

        options.setBackgroundColour(
            (0.0, 0.0, 0.0)
        )

    elif background == "transparent":

        options.setBackgroundColour(
            (0.0, 0.0, 0.0, 0.0)
        )

    else:

        options.setBackgroundColour( (1.0, 1.0, 1.0) )

    # ========================================================
    # ATOM LABELS
    # ========================================================

    if not show_atom_labels:

        for atom in mol.GetAtoms():

            atom.SetProp( "_displayLabel", "", )

    # ========================================================
    # DRAW
    # ========================================================

    drawer.DrawMolecule( mol )

    drawer.FinishDrawing()

    # ========================================================
    # PNG -> PIL
    # ========================================================

    from PIL import Image
    from io import BytesIO

    image = Image.open( BytesIO( drawer.GetDrawingText() ) )

    return image

# ============================================================
# XYZ / 3D
# ============================================================

ATOM_INFO = {
    1: ("H", "Hydrogen"),
    6: ("C", "Carbon"),
    7: ("N", "Nitrogen"),
    8: ("O", "Oxygen"),
    16: ("S", "Sulfur"),
    17: ("Cl", "Chlorine"),
}


ATOMIC_SYMBOLS = { atomic_number: info[0] for atomic_number, info in ATOM_INFO.items() }


def molecule_to_xyz(molecule):
    """Convert molecular coordinates to XYZ format."""

    atomic_numbers = np.asarray(
        molecule["atNUM"]
    ).reshape(-1)

    coordinates = np.asarray( molecule["atXYZ"], dtype=float, )

    lines = [ str(len(atomic_numbers)), "Molecular structure", ]

    for atomic_number, xyz in zip( atomic_numbers, coordinates, ):

        symbol = ATOMIC_SYMBOLS.get( int(atomic_number), "X", )

        x, y, z = xyz

        lines.append( f"{symbol} " f"{x:.6f} " f"{y:.6f} " f"{z:.6f}" )

    return "\n".join(lines)


def molecule_to_position_dataframe( molecule, molecule_id, ):
    """
    Convert atomic numbers and Cartesian coordinates
    into a tabular representation.
    """

    atomic_numbers = np.asarray( molecule["atNUM"] ).reshape(-1)

    coordinates = np.asarray( molecule["atXYZ"], dtype=float, )
    rows = []

    for atom_index, ( atomic_number, xyz, ) in enumerate( zip( atomic_numbers, coordinates, ), start=1, ):

        atomic_number = int( atomic_number )

        symbol = ATOMIC_SYMBOLS.get( atomic_number, "X", )

        x, y, z = xyz

        rows.append(
            {
                "Molecule ID": str( molecule_id ),
                "Atom": atom_index,
                "Element": symbol,
                "Atomic number": atomic_number,
                "X (Å)": float(x),
                "Y (Å)": float(y),
                "Z (Å)": float(z),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# HDF5 TREE
# ============================================================

def print_hdf5_tree_limited( hdf, max_molecules=2, ):
    """Display a limited preview of HDF5 structure."""

    mol_keys = _sorted_molecule_keys(hdf)[:max_molecules]

    st.markdown("### 📁 Root `/`")

    st.caption( "Limited preview of the dataset structure." )

    for mol in mol_keys:

        st.markdown( f"#### 📁 Molecule `{mol}`" )

        mol_group = hdf[mol]

        for geom in get_geometry_keys(mol_group):

            st.markdown( f"**📐 Conformer `{geom}`**" )

            geom_group = mol_group[geom]

            rows = []

            for prop in sorted( geom_group.keys(), key=str.lower, ):

                obj = geom_group[prop]

                shape = ( obj.shape if hasattr(obj, "shape") else "scalar" )

                dtype = ( str(obj.dtype) if hasattr(obj, "dtype") else "-" )

                rows.append(
                    f""" <tr> <td><code>{prop}</code></td> <td>{shape}</td> <td>{dtype}</td> </tr> """ )

            table_html = f"""
            <div style="width: fit-content; max-width: 100%;">
                <div class="qm7x-table-wrapper">
                <table class="qm7x-table">
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Shape</th>
                            <th>Dtype</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
                </div>
            </div>
            """
            st.markdown( table_html, unsafe_allow_html=True, )

    st.info( f"Only the first molecule {len(mol_keys)} is shown." )

# ============================================================
# MOLECULAR COMPOSITION
# ============================================================

def formula_from_atnums(atnums):
    """Generate a molecular formula from atomic numbers."""

    counts = Counter( int(x) for x in np.asarray( atnums ).reshape(-1) )

    elements = [(6, "C"), (1, "H"), (7, "N"), (8, "O"), (16, "S"), (17, "Cl")]

    formula = ""

    for atomic_number, symbol in elements:

        count = counts.get( atomic_number, 0, )

        if count:

            formula += symbol

            if count > 1:

                formula += str( count )

    return formula


# ============================================================
# PROPERTY DEFINITIONS
# ============================================================

PROPERTY_INFO = {

    # --------------------------------------------------------
    # Energetics
    # --------------------------------------------------------
    "ePBE0+MBD": { "name": "PBE0 + MBD Energy", "description": ( "Total energy including PBE0 and many-body " "dispersion correction." ), "unit": "eV", "category": "Energetics", },

    "ePBE0": { "name": "PBE0 Energy", "description": "Total PBE0 energy.", "unit": "eV", "category": "Energetics", },

    "eMBD": { "name": "MBD Energy", "description": "Many-body dispersion energy contribution.", "unit": "eV", "category": "Energetics", },

    "eAT": { "name": "Atomization Energy", "description": "PBE0 atomization energy.", "unit": "eV", "category": "Energetics", },

    "eTS": { "name": "TS Dispersion Energy", "description": ( "Tkatchenko–Scheffler dispersion energy." ), "unit": "eV", "category": "Energetics", },

    "eNN": { "name": "Nuclear–Nuclear Energy", "description": "Nuclear–nuclear repulsion energy.", "unit": "eV", "category": "Energetics", },

    "eNE": { "name": "Nuclear–Electron Energy", "description": "Nuclear–electron attraction energy.", "unit": "eV", "category": "Energetics", },

    "eEE": { "name": "Electron–Electron Energy", "description": "Classical electron–electron Coulomb energy.", "unit": "eV", "category": "Energetics", },
    # --------------------------------------------------------
    # Electronic
    # --------------------------------------------------------

    "eH": { "name": "HOMO Energy", "description": ( "Energy of the highest occupied molecular orbital." ), "unit": "eV", "category": "Electronic", },
    "eL": { "name": "LUMO Energy", "description": ( "Energy of the lowest unoccupied molecular orbital." ), "unit": "eV", "category": "Electronic", },
    "HLgap": { "name": "HOMO–LUMO Gap", "description": ( "Energy difference between the HOMO and LUMO." ), "unit": "eV", "category": "Electronic", },

    # --------------------------------------------------------
    # Molecular response
    # --------------------------------------------------------
    "DIP": { "name": "Dipole Moment", "description": "Total molecular dipole moment.", "unit": "e·Å", "category": "Response", },

    "mPOL": { "name": "Molecular Polarizability", "description": ( "Molecular polarizability calculated using SCS." ), "unit": "a₀³", "category": "Response", },

    "mC6": { "name": "Molecular C₆ Coefficient", "description": ( "Molecular C₆ dispersion coefficient." ), "unit": "hartree·bohr⁶", "category": "Response", },

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    "n_atoms": { "name": "Number of Atoms", "description": ( "Number of atoms in the molecular structure." ), "unit": "atoms", "category": "Structure", },
}

# ============================================================
# PROPERTY TABLE
# ============================================================


def qm7x_property_table():
    """Return the QM7-X property reference table."""

    data = [
        ( "DIP", "Dipole moment", "e·Å", ),
        ( "HLgap", "HOMO–LUMO gap", "eV", ),
        ( "atNUM", "Atomic numbers", "unitless", ),
        ( "atPOL", "Atomic polarizabilities", "a₀³", ),
        ( "atXYZ", "Cartesian coordinates", "Å", ),
        ( "eAT", "Atomic energy", "eV", ),
        ( "eC", "Coulomb energy", "eV", ),
        ( "eEE", "Electron–electron energy", "eV", ),
        ( "eH", "Hartree energy", "eV", ),
        ( "eKIN", "Kinetic energy", "eV", ),
        ( "eL", "Local energy", "eV", ),
        ( "eMBD", "Many-body dispersion energy", "eV", ),
        ( "eNE", "Nuclear–electron energy", "eV", ),
        ( "eNN", "Nuclear–nuclear energy", "eV", ),
        ( "ePBE0", "PBE0 total energy", "eV", ),
        ( "ePBE0+MBD", "PBE0 + MBD energy", "eV", ),
        ( "eTS", "TS dispersion energy", "eV", ),
        ( "eX", "Exchange energy", "eV", ),
        ( "eXC", "Exchange–correlation energy", "eV", ),
        ( "eXX", "Exact exchange energy", "eV", ),
        ( "hCHG", "Hirshfeld atomic charges", "e", ),
        ( "mC6", "Molecular C6 coefficient", "a.u.", ),
        ( "mPOL", "Molecular polarizability", "a₀³", ),
        ( "vdwR", "van der Waals radii", "bohr", ),
    ]

    return pd.DataFrame( data, columns=[ "Property", "Meaning", "Units", ], )


# ============================================================
# AVAILABLE DATASET PROPERTIES
# ============================================================

def get_available_properties(hdf):
    """
    Return the molecular properties available in the loaded
    HDF5 dataset. Properties are detected from the first available molecular geometry.
    """

    if hdf is None:
        return []

    mol_keys = _sorted_molecule_keys(hdf)

    if not mol_keys:
        return []

    mol_group = hdf[mol_keys[0]]

    geometry_keys = get_geometry_keys(mol_group)

    if not geometry_keys:
        return []

    geom_group = mol_group[geometry_keys[0]]

    return [ key for key in geom_group.keys() if isinstance(geom_group[key], h5py.Dataset) ]

# ============================================================
# PROPERTY TYPE DETECTION
# ============================================================

def get_composition_statistics( hdf, max_molecules=None, ):
    """
    Analyze molecular composition from optimized geometries.
    """

    element_counts = Counter()
    atom_counts = []
    formulas = []

    molecule_keys = _sorted_molecule_keys(hdf)

    if max_molecules is not None:
        molecule_keys = molecule_keys[:max_molecules]

    for molecule_id in molecule_keys:

        try:
            molecule = hdf[molecule_id]

            _, geometry = get_opt_geometry( molecule )

        except Exception:
            continue

        if geometry is None:
            continue

        if "atNUM" not in geometry:
            continue

        try:
            atnums = np.asarray( geometry["atNUM"][:] ).reshape(-1)

        except Exception:
            continue

        # -----------------------------------------------
        # Atom count
        # -----------------------------------------------

        n_atoms = len(atnums)

        atom_counts.append( n_atoms )

        # -----------------------------------------------
        # Element counts
        # -----------------------------------------------

        for atomic_number in atnums:

            symbol = ATOMIC_SYMBOLS.get( int(atomic_number) )

            if symbol is not None:
                element_counts[symbol] += 1

        # -----------------------------------------------
        # Molecular formula
        # -----------------------------------------------

        formula = formula_from_atnums( atnums )

        if formula:
            formulas.append(formula)

    # -----------------------------------------------
    # Summary
    # -----------------------------------------------

    total_atoms = sum( element_counts.values() )

    n_molecules = len(atom_counts)

    average_atoms = ( total_atoms / n_molecules if n_molecules else 0.0 )

    formula_counts = Counter( formulas )

    return {
        "element_counts":
            dict(element_counts),

        "atom_counts":
            atom_counts,

        "formulas":
            formulas,

        "formula_counts":
            dict(formula_counts),

        "total_atoms":
            total_atoms,

        "molecules_analyzed":
            n_molecules,

        "average_atoms":
            average_atoms,

        "min_atoms":
            min(atom_counts)
            if atom_counts
            else 0,

        "max_atoms":
            max(atom_counts)
            if atom_counts
            else 0,
    }


_COMPOSITION_ELEMENT_COLUMNS = ("H", "C", "N", "O", "S", "Cl")


def get_composition_statistics_from_index(opt_df, max_molecules=None):
    """
    Vectorized equivalent of `get_composition_statistics`, computed
    entirely from the optimized-molecule Parquet index.
    """

    element_columns = [ f"n_{symbol}" for symbol in _COMPOSITION_ELEMENT_COLUMNS ]

    required = {"formula", "n_atoms", *element_columns}

    if not required.issubset(opt_df.columns):

        raise ValueError(
            "The optimized index does not contain composition "
            "columns (formula / n_H / n_C / ...). Rebuild the "
            "dataset index to enable index-based composition "
            "statistics."
        )

    df = opt_df

    if max_molecules is not None:
        df = df.head(int(max_molecules))

    df = df.dropna(subset=["n_atoms"])

    atom_counts = df["n_atoms"].astype(int).tolist()

    formulas = df["formula"].dropna().tolist()

    element_counts = { symbol: int(df[f"n_{symbol}"].sum()) for symbol in _COMPOSITION_ELEMENT_COLUMNS if int(df[f"n_{symbol}"].sum()) > 0 }

    total_atoms = sum(element_counts.values())

    n_molecules = len(atom_counts)

    average_atoms = total_atoms / n_molecules if n_molecules else 0.0

    formula_counts = Counter(formulas)

    return {
        "element_counts": element_counts,
        "atom_counts": atom_counts,
        "formulas": formulas,
        "formula_counts": dict(formula_counts),
        "total_atoms": total_atoms,
        "molecules_analyzed": n_molecules,
        "average_atoms": average_atoms,
        "min_atoms": min(atom_counts) if atom_counts else 0,
        "max_atoms": max(atom_counts) if atom_counts else 0,
    }

