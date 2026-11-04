# ============================================================
# PRECOMPUTED ML VALIDATION RESULTS (Page 4)
# ============================================================
#
# Generate locally, from the repository root, whenever the model or the
# validation datasets change:
#
#     python -m modules.build_ml_validation_artifacts
#
# This downloads (or reuses) the Zenodo model and train/test HDF5 files
# in models/ through the existing cache helper, runs the exact same
# parsing and prediction code as Page 4 (ml_tools.get_predictions), and
# writes two small files that can be tracked in Git:
#
#     validation_ml_data/ml_validation_results.npz   (arrays)
#     validation_ml_data/ml_validation_results.json  (metadata / compatibility)
#
# The model and the HDF5 datasets themselves stay on Zenodo.
#
# Page 4 calls load_validation_artifact(); when it returns None the page
# falls back to the full HDF5 + prediction pipeline.
#
# Compatibility checks (cheap enough for every page load):
#   - both files present, JSON readable, matching ARTIFACT_VERSION
#   - SHA-256 of the NPZ matches the value recorded in the JSON
#     (detects a corrupt or mismatched pair)
#   - SHA-256 of the local model file matches the recorded model hash
#     (hashed once per model file signature, then memoized)
#   - Zenodo URLs and FEATURE_NAMES/order match the current code
#   - array names, lengths, shapes and finiteness are consistent, and the
#     metrics recomputed from the arrays match the recorded ones
#   - if a train/test HDF5 file is cached locally, its size matches
#
# Limitations: the HDF5 contents are not re-hashed on page load (only
# their size is compared, and only when they are present locally; mtime
# is recorded for information but differs on every fresh download). A
# dataset replaced on Zenodo under the same URL with the same byte size
# would not be detected -- regenerate the artifact after any data update.
# Library versions are recorded and reported but do not invalidate the
# artifact, since the saved predictions were produced at generation time.

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "validation_ml_data"

ARTIFACT_NAME = "ml_validation_results"
ARTIFACT_VERSION = 1

# Number of raw independent-test feature rows kept for SHAP (the SHAP
# slider maximum on Page 4).
SHAP_MAX_ROWS = 200

MODEL_URL = "https://zenodo.org/records/22893015/files/qued-model_dip.pkl?download=1"
TEST_URL = "https://zenodo.org/records/22893015/files/QMdesc_qm7x-eq_test.h5?download=1"
TRAIN_URL = "https://zenodo.org/records/22893015/files/QMdesc_qm7x-eq_train.h5?download=1"

ARRAY_NAMES = ( "y_train", "y_train_pred", "y_test", "y_pred", "X_test_shap", )

METRICS_RTOL = 1e-9


def artifact_paths(artifacts_dir=ARTIFACTS_DIR):

    artifacts_dir = Path(artifacts_dir)

    return ( artifacts_dir / f"{ARTIFACT_NAME}.npz", artifacts_dir / f"{ARTIFACT_NAME}.json", )


def sha256_file(path, chunk_size=1 << 20):

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)

    return digest.hexdigest()


@lru_cache(maxsize=8)
def _sha256_for_signature(path, mtime, size):
    """Hash a file once per (path, mtime, size) signature."""

    return sha256_file(path)


def cached_sha256(path):

    stat = Path(path).stat()

    return _sha256_for_signature(str(path), stat.st_mtime, stat.st_size)


def _file_signature(path):

    stat = Path(path).stat()

    return {
        "filename": Path(path).name,
        "size": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _library_versions():

    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def _metrics_dict(metrics):

    return { "r2": float(metrics["r2"]), "mae": float(metrics["mae"]), "rmse": float(metrics["rmse"]), "n": int(metrics["n"]), }


@contextmanager
def _progress_stage(label):
    """Print when a generation stage starts and finishes, with its duration."""

    print(f"[start] {label}...", flush=True)
    start = time.perf_counter()

    try:
        yield
    except BaseException:
        print(f"[failed] {label} after {time.perf_counter() - start:.1f}s", flush=True)
        raise

    print(f"[done] {label} in {time.perf_counter() - start:.1f}s", flush=True)


# ============================================================
# WRITING
# ============================================================

def write_validation_artifact( arrays, model_path, train_path, test_path, artifacts_dir=ARTIFACTS_DIR, ):
    """
    Write the NPZ + JSON pair from already-computed arrays. Both files are
    written to temporary names first and then moved into place.
    """

    from modules import ml_tools

    missing = [name for name in ARRAY_NAMES if name not in arrays]

    if missing:
        raise ValueError(f"Missing arrays: {missing}")

    arrays = {name: np.asarray(arrays[name]) for name in ARRAY_NAMES}

    npz_path, json_path = artifact_paths(artifacts_dir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    npz_tmp = npz_path.with_name(npz_path.name + ".part.npz")
    json_tmp = json_path.with_name(json_path.name + ".part")

    try:

        np.savez_compressed(npz_tmp, **arrays)

        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "npz_sha256": sha256_file(npz_tmp),
            "model": { "url": MODEL_URL, "sha256": sha256_file(model_path), **_file_signature(model_path), },
            "datasets": {
                "train": {"url": TRAIN_URL, **_file_signature(train_path)},
                "test": {"url": TEST_URL, **_file_signature(test_path)},
            },
            "feature_names": list(ml_tools.FEATURE_NAMES),
            "n_features": len(ml_tools.FEATURE_NAMES),
            "x_test_shap_rows": int(arrays["X_test_shap"].shape[0]),
            "x_test_shap_note": (
                "First rows of the independent test set in file order, raw "
                "(unscaled) features from ml_tools._extract_feature_row."
            ),
            "metrics": {
                "train": _metrics_dict(ml_tools.compute_metrics(arrays["y_train"], arrays["y_train_pred"])),
                "test": _metrics_dict(ml_tools.compute_metrics(arrays["y_test"], arrays["y_pred"])),
            },
            "versions": _library_versions(),
        }

        json_tmp.write_text(json.dumps(metadata, indent=2))

        os.replace(npz_tmp, npz_path)
        os.replace(json_tmp, json_path)

    finally:

        npz_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)

    return npz_path, json_path


def build_validation_artifact(models_dir=MODELS_DIR, artifacts_dir=ARTIFACTS_DIR):
    """
    Full (expensive) generation: Zenodo download/cache, HDF5 parsing and
    predictions, exactly as Page 4's fallback pipeline does them.
    """

    from modules import ml_tools
    from modules.loader import ensure_zenodo_file_cached

    with _progress_stage("Locating model and train/test HDF5 files"):
        model_path = ensure_zenodo_file_cached(MODEL_URL, cache_dir=str(models_dir))
        train_path = ensure_zenodo_file_cached(TRAIN_URL, cache_dir=str(models_dir))
        test_path = ensure_zenodo_file_cached(TEST_URL, cache_dir=str(models_dir))

    with _progress_stage(f"Parsing training data and predicting ({Path(train_path).name})"):
        _, y_train, y_train_pred = ml_tools.get_predictions( *ml_tools.get_file_signature(train_path), str(model_path) )

    print(f"        training samples: {len(y_train)}", flush=True)

    with _progress_stage(f"Parsing independent test data and predicting ({Path(test_path).name})"):
        X_test, y_test, y_pred = ml_tools.get_predictions( *ml_tools.get_file_signature(test_path), str(model_path) )

    print(f"        test samples: {len(y_test)}", flush=True)

    arrays = {
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test_shap": X_test[:SHAP_MAX_ROWS],
    }

    with _progress_stage(f"Writing output files to {artifacts_dir}"):
        return write_validation_artifact( arrays, model_path, train_path, test_path, artifacts_dir=artifacts_dir, )


# ============================================================
# LOADING / VALIDATION (used by Page 4)
# ============================================================

def load_validation_artifact( model_path, models_dir=MODELS_DIR, artifacts_dir=ARTIFACTS_DIR, ):
    """
    Return (arrays, metadata, None) for a valid artifact, or
    (None, None, reason) when it is missing, corrupt or incompatible.
    """

    npz_path, json_path = artifact_paths(artifacts_dir)

    if not npz_path.exists() or not json_path.exists():
        return None, None, "no precomputed results found"

    try:

        from modules import ml_tools

        metadata = json.loads(json_path.read_text())

        if metadata.get("artifact_version") != ARTIFACT_VERSION:
            return None, None, "precomputed results use an older format"

        if cached_sha256(npz_path) != metadata.get("npz_sha256"):
            return None, None, "precomputed arrays do not match their metadata (corrupt or mismatched files)"

        if metadata["model"].get("url") != MODEL_URL or metadata["datasets"]["train"].get("url") != TRAIN_URL or metadata["datasets"]["test"].get("url") != TEST_URL:
            return None, None, "precomputed results refer to different Zenodo files"

        if cached_sha256(model_path) != metadata["model"].get("sha256"):
            return None, None, "precomputed results were generated with a different model file"

        if metadata.get("feature_names") != list(ml_tools.FEATURE_NAMES):
            return None, None, "precomputed results use a different feature set"

        for split in ("train", "test"):

            local = Path(models_dir) / metadata["datasets"][split]["filename"]

            if local.exists() and local.stat().st_size != metadata["datasets"][split]["size"]:
                return None, None, f"the local {split} dataset differs from the one used for the precomputed results"

        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in ARRAY_NAMES}

        n_train = metadata["metrics"]["train"]["n"]
        n_test = metadata["metrics"]["test"]["n"]
        n_shap = min(SHAP_MAX_ROWS, n_test)

        shapes_ok = (
            arrays["y_train"].shape == (n_train,)
            and arrays["y_train_pred"].shape == (n_train,)
            and arrays["y_test"].shape == (n_test,)
            and arrays["y_pred"].shape == (n_test,)
            and arrays["X_test_shap"].shape == (n_shap, len(ml_tools.FEATURE_NAMES))
            and metadata.get("x_test_shap_rows") == n_shap
        )

        if not shapes_ok:
            return None, None, "precomputed arrays have unexpected shapes"

        if not all(np.isfinite(arrays[name]).all() for name in ARRAY_NAMES):
            return None, None, "precomputed arrays contain non-finite values"

        for split, (y_true, y_hat) in { "train": (arrays["y_train"], arrays["y_train_pred"]), "test": (arrays["y_test"], arrays["y_pred"]), }.items():

            recomputed = _metrics_dict(ml_tools.compute_metrics(y_true, y_hat))
            stored = metadata["metrics"][split]

            if not all(np.isclose(recomputed[k], stored[k], rtol=METRICS_RTOL, atol=0) for k in ("r2", "mae", "rmse", "n")):
                return None, None, f"recorded {split} metrics do not match the precomputed arrays"

    except Exception as exc:

        return None, None, f"precomputed results could not be read ({type(exc).__name__}: {exc})"

    return arrays, metadata, None


def version_differences(metadata):
    """Library versions that differ from the ones used at generation time."""

    try:
        current = _library_versions()
    except Exception:
        return {}

    recorded = metadata.get("versions", {})

    return { name: (recorded.get(name), version) for name, version in current.items() if recorded.get(name) != version }


# ============================================================
# CLI
# ============================================================

def main(argv=None):

    parser = argparse.ArgumentParser( description="Precompute Page 4 ML validation results (expensive: parses the full train/test HDF5 files)." )
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifact files.")

    args = parser.parse_args(argv)

    npz_path, json_path = artifact_paths(args.artifacts_dir)

    if (npz_path.exists() or json_path.exists()) and not args.force:
        print(f"Artifact already exists in {args.artifacts_dir}; pass --force to regenerate.", file=sys.stderr)
        return 1

    npz_path, json_path = build_validation_artifact(args.models_dir, args.artifacts_dir)

    metadata = json.loads(json_path.read_text())

    print(f"Wrote {npz_path} ({npz_path.stat().st_size / 1e6:.2f} MB) and {json_path}")
    print("Train metrics:", metadata["metrics"]["train"])
    print("Test metrics: ", metadata["metrics"]["test"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
