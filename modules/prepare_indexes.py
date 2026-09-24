# modules/prepare_indexes.py

from pathlib import Path
import hashlib
import json

import pandas as pd

from modules.loader import ( build_qm7x_index, load_qm7x_index, )


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_OPT_INDEX = Path( "data/qm7x_index.parquet" )

DEFAULT_GEOMETRY_INDEX = Path( "data/qm7x_geometries.parquet" )

DEFAULT_METADATA = Path( "data/qm7x_metadata.json" )

# Root directory under which each *non-default* dataset gets its own
# isolated index/metadata subdirectory -- see `resolve_index_paths`.
DEFAULT_INDEX_CACHE_DIR = Path( "data/index_cache" )

# ============================================================
# INDEX SCHEMA VERSION
# ============================================================

INDEX_SCHEMA_VERSION = 4

# ============================================================
# FILE SIGNATURE
# ============================================================

def get_file_signature(file_path):
    """
    Create a lightweight signature for a local dataset.

    The signature is based on:
        - file name
        - file size
        - modification time
    """

    path = Path(file_path)

    if not path.exists():
        return None

    stat = path.stat()

    return {"name": path.name, "size": stat.st_size, "modified": stat.st_mtime}


# ============================================================
# DATASET IDENTITY
# ============================================================

def compute_dataset_id(
    source,
    source_path=None,
    dataset_name=None,
    dataset_size_mb=None,
):
    """
    Build a short, stable, filesystem-safe identifier for "the
    dataset currently active in the app".

    This is the single identifier used both to isolate per-dataset
    Parquet indexes on disk (`resolve_index_paths`) and to key the
    per-tab analysis caches in `pages/1_Dataset_Overview.py` -- so a
    dataset switch (Local / Upload / Zenodo, in any direction) can
    never keep reading another dataset's cached index or stats.

    Folds in more than just `dataset_name` on purpose: two different
    uploads (or two different Zenodo files) can legitimately share a
    filename, and must still resolve to different ids.

        Local   -> name + size + mtime of the file on disk
        Upload  -> name + size (no stable filesystem signature exists
                   for an in-memory upload)
        Zenodo  -> the (stripped) source URL itself
    """

    if source == "Local" and source_path:

        signature = get_file_signature(source_path)

        if signature is not None:

            basis = (
                f"local:{signature['name']}:"
                f"{signature['size']}:{signature['modified']}"
            )

        else:

            basis = f"local:{source_path}"

    elif source == "Zenodo" and source_path:

        basis = f"zenodo:{str(source_path).strip()}"

    elif source == "Upload":

        basis = f"upload:{dataset_name}:{dataset_size_mb}"

    else:

        basis = f"{source}:{dataset_name}:{dataset_size_mb}"

    return hashlib.sha1(
        basis.encode("utf-8")
    ).hexdigest()[:16]

# ============================================================
# METADATA
# ============================================================

def save_metadata(
    metadata,
    output_path=DEFAULT_METADATA,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


def load_metadata(
    metadata_path=DEFAULT_METADATA,
):

    metadata_path = Path(
        metadata_path
    )

    if not metadata_path.exists():
        return None

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================
# INDEX AVAILABILITY
# ============================================================

def indexes_exist(
    opt_index_path=DEFAULT_OPT_INDEX,
    geometry_index_path=DEFAULT_GEOMETRY_INDEX,
):

    return (
        Path(opt_index_path).exists()
        and
        Path(geometry_index_path).exists()
    )


# ============================================================
# INDEX VALIDITY (VERSIONING)
# ============================================================

def index_is_compatible(
    source_path=None,
    metadata_path=DEFAULT_METADATA,
):
    """
    Decide whether an on-disk index can be safely reused.

    Two independent checks, both must pass:

      1. Schema version -- the metadata must have been written by
         the same INDEX_SCHEMA_VERSION as the running code. This
         catches an index built before a column change (e.g. the
         composition / geometry-ID-token columns).

      2. Source signature -- when `source_path` is known, its
         current (name, size, mtime) signature must match the one
         recorded when the index was built. This catches pointing
         the app at a different HDF5 file while reusing indexes
         left over from a previous one.

    If `source_path` is not available (e.g. an uploaded file with
    no stable local path, or a dataset loaded before this metadata
    field existed), only the schema-version check applies -- this
    intentionally keeps the check lightweight rather than forcing
    a rebuild whenever the source can't be identified.
    """

    metadata = load_metadata(metadata_path)

    if not metadata:
        return False

    if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
        return False

    if source_path:

        current_signature = get_file_signature(source_path)

        recorded_signature = metadata.get("source")

        if current_signature is not None and recorded_signature is not None:

            if current_signature != recorded_signature:
                return False

    return True


def _default_index_matches_source(
    source_path,
    metadata_path=DEFAULT_METADATA,
):
    """
    True only when we can *positively confirm* the existing default
    index/metadata (the historical single shared location) was built
    from this exact local file -- same name, size, and mtime.

    This is deliberately stricter than `index_is_compatible`: that
    function treats an unresolvable signature (no local `source_path`
    at all, e.g. an upload or a Zenodo URL) as "compatible", which is
    the right call for "does a rebuild have to run", but the wrong
    call for "is it safe to hand this dataset another dataset's
    cached index". Here, no signature ever means no match.

    Deliberately does NOT check schema_version, unlike
    `index_is_compatible` -- this function only answers "does this
    source file *identify* as the one behind the default location",
    a question of file identity, not of index freshness. Folding the
    schema-version check in here would mean a schema bump (e.g. this
    module's own `INDEX_SCHEMA_VERSION` bump) makes the *same* QM7-X
    file stop matching its own historical location and get rerouted
    into a fresh isolated subdirectory instead of rebuilt in place.
    Staleness is `index_is_compatible`'s job -- it still runs
    (inside `prepare_dataset_indexes`) against whatever path this
    function resolves to, and triggers a rebuild there when the
    schema version doesn't match.
    """

    metadata = load_metadata(metadata_path)

    if not metadata:
        return False

    if not source_path:
        return False

    current_signature = get_file_signature(source_path)
    recorded_signature = metadata.get("source")

    return (
        current_signature is not None
        and recorded_signature is not None
        and current_signature == recorded_signature
    )


# ============================================================
# INDEX PATH RESOLUTION
# ============================================================

def resolve_index_paths(
    dataset_id=None,
    source_path=None,
    opt_index_path=None,
    geometry_index_path=None,
    metadata_path=None,
    base_dir=DEFAULT_INDEX_CACHE_DIR,
):
    """
    Decide which on-disk Parquet/metadata paths a dataset should use.

    Explicit paths (all three given) always win -- a caller that
    already knows exactly where it wants its index can bypass
    dataset-id resolution entirely.

    Otherwise:

      - If the file at `source_path` is a positive match for the
        file that built the *existing* default index (see
        `_default_index_matches_source`), that default, historical
        location is reused as-is. This keeps the already-built
        QM7-X cache fast/zero-rebuild for the common "same local
        dataset every run" case.

      - Otherwise, `dataset_id` gets its own isolated subdirectory
        under `base_dir`, so a different local file, an upload, or a
        Zenodo dataset can never read back -- or silently overwrite
        -- another dataset's cached index.
    """

    if opt_index_path and geometry_index_path and metadata_path:

        return (Path(opt_index_path), Path(geometry_index_path), Path(metadata_path))

    if _default_index_matches_source(source_path, DEFAULT_METADATA):

        return (DEFAULT_OPT_INDEX, DEFAULT_GEOMETRY_INDEX, DEFAULT_METADATA)

    safe_id = dataset_id or "unknown_dataset"

    dataset_dir = Path(base_dir) / safe_id

    return (
        dataset_dir / "qm7x_index.parquet",
        dataset_dir / "qm7x_geometries.parquet",
        dataset_dir / "qm7x_metadata.json",
    )


# ============================================================
# LOAD INDEXES
# ============================================================

def load_analysis_indexes(
    opt_index_path=DEFAULT_OPT_INDEX,
    geometry_index_path=DEFAULT_GEOMETRY_INDEX,
):
    """
    Load both Parquet indexes.

    Delegates to `loader.load_qm7x_index`, which is
    `st.cache_data`-wrapped, so repeated calls (e.g. across
    Streamlit reruns) reuse the cached DataFrame instead of
    re-reading Parquet from disk every time.
    """

    opt_df = load_qm7x_index(str(opt_index_path))

    geometry_df = load_qm7x_index(str(geometry_index_path))

    return (opt_df, geometry_df)


# ============================================================
# BUILD ANALYSIS INDEXES
# ============================================================

def build_analysis_indexes(
    hdf,
    opt_index_path=DEFAULT_OPT_INDEX,
    geometry_index_path=DEFAULT_GEOMETRY_INDEX,
    max_molecules=None,
):
    """
    Build the lightweight Parquet indexes used by
    the Streamlit application.

    The original HDF5 file is NOT modified.

    Returns
    -------
    dict
        {
            "opt_df": DataFrame,
            "geometry_df": DataFrame
        }
    """

    opt_index_path = Path(
        opt_index_path
    )

    geometry_index_path = Path(
        geometry_index_path
    )

    opt_index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    geometry_index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # OPTIMIZED INDEX
    # ========================================================

    opt_df = build_qm7x_index(
        hdf,
        output_path=str(
            opt_index_path
        ),
        max_molecules=max_molecules,
        geometry_mode="opt",
    )

    if not isinstance(
        opt_df,
        pd.DataFrame,
    ):

        if isinstance(
            opt_df,
            dict,
        ):

            opt_df = opt_df.get(
                "df",
                opt_df.get("data"),
            )

    if opt_df is None:

        raise ValueError(
            "Could not create optimized index."
        )

    # ========================================================
    # GEOMETRY INDEX
    # ========================================================

    geometry_df = build_qm7x_index(
        hdf,
        output_path=str(
            geometry_index_path
        ),
        max_molecules=None,
        geometry_mode="all",
    )

    if not isinstance(
        geometry_df,
        pd.DataFrame,
    ):

        if isinstance(
            geometry_df,
            dict,
        ):

            geometry_df = geometry_df.get(
                "df",
                geometry_df.get("data"),
            )

    if geometry_df is None:

        raise ValueError(
            "Could not create geometry index."
        )

    return {"opt_df": opt_df, "geometry_df": geometry_df}


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset_indexes(
    hdf,
    source_path=None,
    dataset_id=None,
    force=False,
    opt_index_path=None,
    geometry_index_path=None,
    metadata_path=None,
    max_molecules=None,
):
    """
    Prepare all lightweight indexes for a dataset.

    If the indexes already exist, they are reused.

    Set force=True to rebuild them.

    `dataset_id` (see `compute_dataset_id`) is what keeps two
    different datasets from ever sharing -- or silently overwriting
    -- each other's on-disk index: when `opt_index_path` /
    `geometry_index_path` / `metadata_path` aren't all given
    explicitly, they're resolved per-dataset via
    `resolve_index_paths` instead of always falling back to the
    single historical `data/qm7x_*` location.
    """

    opt_index_path, geometry_index_path, metadata_path = (
        resolve_index_paths(
            dataset_id=dataset_id,
            source_path=source_path,
            opt_index_path=opt_index_path,
            geometry_index_path=geometry_index_path,
            metadata_path=metadata_path,
        )
    )

    # ========================================================
    # CHECK EXISTING INDEX
    # ========================================================

    if not force:

        if indexes_exist(
            opt_index_path,
            geometry_index_path,
        ) and index_is_compatible(
            source_path,
            metadata_path,
        ):

            opt_df, geometry_df = (
                load_analysis_indexes(
                    opt_index_path,
                    geometry_index_path,
                )
            )

            return {"opt_df": opt_df, "geometry_df": geometry_df, "created": False}

    # An incompatible index on disk (stale source, or older
    # schema version) is rebuilt below rather than silently reused.

    # ========================================================
    # BUILD
    # ========================================================

    result = build_analysis_indexes(
        hdf,
        opt_index_path=opt_index_path,
        geometry_index_path=geometry_index_path,
        max_molecules=max_molecules,
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "schema_version": INDEX_SCHEMA_VERSION,

        "source": get_file_signature(
            source_path
        )
        if source_path
        else None,

        "optimized_index": str(
            opt_index_path
        ),

        "geometry_index": str(
            geometry_index_path
        ),

        "optimized_molecules": len(
            result["opt_df"]
        ),

        "geometry_records": len(
            result["geometry_df"]
        ),
    }

    save_metadata(
        metadata,
        metadata_path,
    )

    return {"opt_df": result["opt_df"], "geometry_df": result["geometry_df"], "created": True}