# QUPEX — Quantum Property EXplorer

An interactive [Streamlit](https://streamlit.io) application for exploring the **QM7-X** quantum-chemistry dataset (Hoja et al., *Scientific Data* 8, 43, 2021) and running a machine-learning surrogate model over it.

QUPEX turns a QM7-X-format HDF5 file into lightweight, cached Parquet indexes and exposes four analysis pages on top of them: dataset composition and geometry statistics, 2D/3D molecular structure visualization with property correlations, PCA/t-SNE chemical-space analysis with Gaussian-Mixture clustering, and a pretrained XGBoost model for dipole-moment prediction with SHAP-based interpretability.

## Contents

- [Features](#features)
- [Screenshot](#screenshot)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Data sources](#data-sources)
- [Project structure](#project-structure)
- [Reproducibility notes](#reproducibility-notes)
- [QM7-X attribution](#qm7-x-attribution)
- [License](#license)
- [Citation](#citation)

## Features

- **Dataset Overview** — dataset-wide summary statistics, elemental/molecular composition, geometry counts, the geometry-identifier hierarchy (Molecule → Isomer → Configuration → Geometry), and a curated QM7-X property reference table.
- **Molecular Properties** — property-distribution and property-correlation views over the indexed scalar properties, a per-molecule 2D (RDKit) + 3D ([py3Dmol](https://pypi.org/project/py3Dmol/)) structure viewer, atomic-level analysis, and side-by-side molecule comparison.
- **Chemical Space Analysis** — PCA and t-SNE dimensionality reduction over a user-selected subset of QM7-X scalar properties, with Gaussian Mixture Model clustering, silhouette-based cluster-quality evaluation, and a PCA-vs-t-SNE cluster-agreement comparison.
- **Machine Learning** — loads a pretrained scikit-learn `Pipeline` (`StandardScaler` + `XGBRegressor`) that predicts dipole moment from a 40-feature tight-binding/DFTB-style descriptor set, reports held-out validation metrics, provides SHAP `TreeExplainer` interpretability, and accepts new HDF5/CSV data for inference.
- **One-click default dataset** — a default QM7-X dataset can be loaded from Zenodo with a single checkbox on the Home page, with no URL or file to provide; it downloads once and is cached locally for every run after that.
- **Flexible data loading** — an HDF5 file (QM7-X-format or QM7-X-like) can also be loaded from a local path, a browser upload, or a different Zenodo URL; the app inspects the file's structure to identify it and caches its own indexes under a stable per-dataset ID.
- **Documentation** — a dedicated in-app page explaining how QUPEX works, where its data comes from, and how to interpret its results, with references and data-source links.

## Screenshot

*(Add a screenshot of the running app here, e.g. `assets/screenshot.png`.)*

## Installation

Requires Python 3.10+ (developed against Python 3.10).

```bash
git clone <this-repository-url>
cd QUPEX

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### macOS note (native crash mitigation)

On macOS, three separate builds of `libomp.dylib` can end up loaded at once (Conda's own copy, one bundled in the `scikit-learn` wheel, and one linked into `xgboost`'s native library), which can cause a native crash when the model is unpickled or used for prediction. `pages/4_Machine_Learning.py` mitigates this by setting `OMP_NUM_THREADS=1` before any `numpy`/`scikit-learn`/`xgboost`/`shap` import. No action is required to benefit from this fix, but it is documented here in case you see native crashes with a different environment setup.

## Running the app

```bash
streamlit run app.py
```

The **Home** page's "Get Started" section is where a dataset is loaded:

- **Use default QM7-X dataset from Zenodo** (checkbox, unchecked by default) — check it to load the QM7-X equilibrium dataset directly from a fixed Zenodo URL, with no need to enter anything. The file downloads once on first use and is cached locally afterwards; every later run (and every app restart) reuses the cached copy instead of re-downloading it.
- Leave it unchecked to pick a data source instead:
  - **Local** — point at an HDF5 file already on disk.
  - **Upload** — upload an HDF5 file through the browser.
  - **Zenodo** — paste a different Zenodo file URL; the app downloads and caches it locally the same way.

The first load of a given file builds its Parquet indexes (cached under `data/index_cache/`, keyed by a hash of the source file's identity); subsequent loads of the same file reuse the cache. Building the index for the full QM7-X file can take a while the first time — this is expected and only happens once per source file.

The **Machine Learning** page fetches its own pretrained model and validation sets automatically from a separate Zenodo record the first time each is needed, and caches them under `models/` for every run after that — see [Data sources](#data-sources). If a download fails (e.g. no network access), the affected page shows a clear error and stops rather than failing unexpectedly.

## Data sources

**The HDF5 datasets and pretrained model artifacts are not included in this repository** (see `.gitignore`) — they are large binary files unsuited to version control and, in every case below, already permanently archived on Zenodo.

- **Default QM7-X dataset** (loaded via the Home page's default-dataset checkbox): [`qm7x-eq.hdf5`](https://zenodo.org/records/22893015/files/qm7x-eq.hdf5?download=1), from Zenodo record [10.5281/zenodo.22893015](https://zenodo.org/records/22893015). Downloaded once into `data/` the first time it is requested, then reused from that local cache on every subsequent run.
- **QM7-X, canonical archive** (if you'd rather load it manually, or load a different QM7-X file): [10.5281/zenodo.4288677](https://zenodo.org/records/4288677) — use the app's Local, Upload, or Zenodo option (uncheck the default-dataset box first).
- **Custom / QM7-X-like datasets**: any HDF5 file that follows the same geometry-identifier convention and property-name signature will be identified as QM7-X-compatible; anything else is treated as a generic "Custom HDF5 dataset."
- **Pretrained ML model and validation sets** (`models/`): downloaded automatically from a dedicated Zenodo record ([10.5281/zenodo.22893015](https://zenodo.org/records/22893015)) the first time the Machine Learning page needs each file:
  - the model (`qued-model_dip.pkl`) is fetched as soon as that page loads;
  - the training and test sets (`QMdesc_qm7x-eq_train.h5`, `QMdesc_qm7x-eq_test.h5`) are fetched only when you click **Run validation**, since they are not needed for new-data prediction alone.

Every file above downloads once and is then cached on disk (`data/` for the main dataset, `models/` for the ML artifacts); subsequent runs and app restarts reuse the cached copy and never re-download it. To force a re-download of any of them, delete the corresponding file from `data/` or `models/`.

No manual placement is required for the default setup — both `data/` and `models/` populate themselves automatically on first use:

```
data/    <- default QM7-X dataset downloads here automatically; or place your own QM7-X/QM7-X-like HDF5 file(s) here for Local loading
models/  <- populated automatically on first use of the Machine Learning page
```

## Project structure

```
app.py                     Navigation shell (st.navigation) + sidebar "Data source" info
home.py                    Landing page; default-dataset checkbox + Local/Upload/Zenodo data-loading controls
pages/
  1_Dataset_Overview.py     Composition, geometries, elements, dataset statistics
  2_Molecular_Properties.py Per-molecule 2D/3D viewer, property distributions & correlations
  3_Chemical_Space_Analysis.py  PCA / t-SNE / GMM over QM7-X scalar properties
  4_Machine_Learning.py     XGBoost dipole-moment model: validation, SHAP, new-data prediction
  5_Documentation.py        In-app documentation: how the app works, data sources, references
modules/
  loader.py                 HDF5 access, RDKit conversion, index building
  prepare_indexes.py        Orchestrates + caches the HDF5 -> Parquet index pipeline
  ml_tools.py                Model loading, validation metrics, SHAP helpers
  molecular_visualization.py 2D/3D molecular rendering helpers
  plot_distributions.py       Molecule size-distribution plot builder
  datasets.py                 Property metadata lookup
  theme.py                    App color palette + apply_global_theme()
  plot_style.py                Shared Plotly layout/styling helpers
data/                       Default + user-supplied HDF5 input, downloaded/cached automatically where applicable, + cached Parquet indexes (not tracked in git)
models/                     Pretrained model + validation HDF5 files, downloaded from Zenodo on first use (not tracked in git)
assets/                     Static images used by the Home page (logo, background)
```

## Reproducibility notes

- All page-level analysis reads from the cached Parquet indexes (`modules.loader.load_qm7x_index`) rather than re-scanning the source HDF5 file; the raw HDF5 file is read directly only when a page needs full atomic detail for one selected molecule/geometry.
- Chemical-space results (Page 3) and SHAP sample selections (Page 4) are computed live from the currently loaded dataset and the properties/parameters the user selects at run time — they are exploratory, parameter-dependent outputs of the tool, not fixed precomputed results shipped with the repository.
- The Machine Learning page's held-out validation metrics are computed by re-running the model against the test set each time validation is run, so results are always reproducible from the exact model and data files archived on Zenodo (see [Data sources](#data-sources)) — the same files are used regardless of who runs the app or when, since they are fetched from the same fixed Zenodo record rather than supplied ad hoc.
- With the default-dataset checkbox checked, every user and every run loads the same fixed QM7-X file from the same Zenodo URL, so the statistics on Dataset Overview and the other pages are directly comparable across machines and over time; using Local/Upload/a different Zenodo URL instead trades that comparability for flexibility.
- Index caches are keyed by the source file's identity (name, size, modification time); replacing a dataset file with a different one and reloading will rebuild the cache under a new key rather than silently reusing a stale index.

## QM7-X attribution

This application is built around the **QM7-X** dataset and follows its HDF5 data organization and property conventions. QM7-X is not redistributed with this repository.

> J. Hoja, L. Medrano Sandonas, B. G. Ernst, A. Vazquez-Mayagoitia, R. A. DiStasio Jr., and A. Tkatchenko, "QM7-X, a comprehensive dataset of quantum-mechanical properties spanning the chemical space of small organic molecules," *Scientific Data*, vol. 8, no. 43, 2021. DOI: [10.1038/s41597-021-00812-2](https://doi.org/10.1038/s41597-021-00812-2)
>
> Dataset archive: [10.5281/zenodo.4288677](https://zenodo.org/records/4288677)

Please cite the original QM7-X publication when using this dataset in derived work, in addition to any citation requested for this application (see [Citation](#citation)).

## License

*(Add a license file and state it here — e.g. MIT, Apache-2.0 — before making the repository public. No license is currently declared.)*

## Citation

If you use this application in your research, please cite it accordingly; citation details will be added here once available.
