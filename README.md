# QUPEX — QUantum Property EXplorer

An interactive [Streamlit](https://streamlit.io) application for exploring the **QM7-X** quantum-chemistry dataset (Hoja et al., *Scientific Data* 8, 43, 2021) and running a machine-learning surrogate model over it.

QUPEX turns a QM7-X-format HDF5 file into lightweight, cached Parquet indexes and exposes four analysis pages on top of them: dataset composition and geometry statistics, 2D/3D molecular structure visualization with property correlations, PCA/t-SNE chemical-space analysis with Gaussian-Mixture clustering, and a pretrained XGBoost model for dipole-moment prediction with SHAP-based interpretability.

## Contents

- [Features](#features)
- [Screenshot](#screenshot)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Testing the machine-learning prediction functionality](#testing-the-machine-learning-prediction-functionality)
- [Data sources](#data-sources)
- [Project structure](#project-structure)
- [Reproducibility notes](#reproducibility-notes)
- [QM7-X attribution](#qm7-x-attribution)
- [License](#license)
- [Citation](#citation)

## Features

- **Dataset Overview** — dataset-wide summary statistics, elemental/molecular composition, geometry counts, the geometry-identifier hierarchy (Molecule → Isomer → Configuration → Geometry), and a curated QM7-X property reference table.
- **Property Space Analysis** — property-distribution and property-correlation views over the indexed scalar properties (with a Molecules/Conformers scope toggle), a correlation-matrix heatmap, and pairwise property relationships paired with a per-molecule 2D (RDKit) + 3D ([py3Dmol](https://pypi.org/project/py3Dmol/)) structure viewer and atomic-level analysis.
- **Molecular Clustering** — PCA and t-SNE dimensionality reduction over a user-selected subset of QM7-X scalar properties (Molecules or Conformers scope), with Gaussian Mixture Model clustering, silhouette-based cluster-quality evaluation, and a PCA-vs-t-SNE cluster-agreement comparison.
- **Machine Learning** — loads a pretrained scikit-learn `Pipeline` (`StandardScaler` + `XGBRegressor`) that predicts dipole moment from a 40-feature tight-binding/DFTB-style descriptor set, reports held-out validation metrics, provides SHAP `TreeExplainer` interpretability, and accepts new HDF5/CSV data for inference.
- **One-click default dataset** — a default QM7-X dataset can be loaded from Zenodo with a single checkbox on the Home page, with no URL or file to provide; it downloads once and is cached locally for every run after that.
- **Flexible data loading** — an HDF5 file (QM7-X-format or QM7-X-like) can also be loaded from a local path, a browser upload, or a different Zenodo URL; the app inspects the file's structure to identify it and caches its own indexes under a stable per-dataset ID.
- **Documentation** — a dedicated in-app page explaining how QUPEX works, where its data comes from, and how to interpret its results, with references and data-source links.

## Screenshot

<p align="center">
  <img src="assets/screenshot.png" alt="Landing page of the application" width="500">
  <br>
  <em>Figure 1. Landing page of the application, featuring the default‑dataset option and the three supported data‑loading modes: Local file, Upload, and Zenodo.</em>
</p>


## Installation

Requires Python 3.10+ (developed against Python 3.10). The steps below assume [Conda](https://docs.conda.io/) (Miniconda or Anaconda) is installed; a plain virtual environment works too — see the note at the end of this section.

1. **Clone the repository:**

   ```bash
   git clone <this-repository-url>
   cd QUPEX
   ```

2. **Create a dedicated Conda environment.** `<environment_name>` can be replaced with any name you like, e.g. `qupex`:

   ```bash
   conda create --name <environment_name> python=3.10
   ```

3. **Activate the environment:**

   ```bash
   conda activate <environment_name>
   ```

4. **Install the Python dependencies:**

   ```bash
   python3 -m pip install -r requirements.txt
   ```

5. **Streamlit** is already listed in `requirements.txt`, so step 4 installs it automatically. If it's ever missing (e.g. after a partial or manual install), install it explicitly:

   ```bash
   python3 -m pip install streamlit
   ```

*Not using Conda?* A plain virtual environment works the same way — replace steps 2–3 with:
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```
then continue from step 4.

### macOS requirement: `libomp`

6. `scikit-learn`, `xgboost`, and `shap` (used on the **Machine Learning** page) depend on OpenMP, which macOS does not ship by default. Without it, `xgboost` can fail to import entirely. Install it once via [Homebrew](https://brew.sh) (Homebrew itself must already be installed):

   ```bash
   brew install libomp
   ```

   No further configuration is needed afterwards.

   Separately: on macOS, three different builds of `libomp.dylib` can end up loaded at once (Conda's own copy, one bundled in the `scikit-learn` wheel, and one linked into `xgboost`'s native library), which can cause a native crash when the model is unpickled or used for prediction. `pages/4_Machine_Learning.py` mitigates this automatically by setting `OMP_NUM_THREADS=1` before any `numpy`/`scikit-learn`/`xgboost`/`shap` import — no action is required for this second issue.

## Running the app

7. Start the application from the repository root, with the environment activated:

   ```bash
   streamlit run app.py
   ```

8. Streamlit prints a local URL (typically `http://localhost:8501`) and normally opens it in your default browser automatically; if it doesn't, open that URL manually. The app starts on the **Home** page.

The **Home** page's "Get Started" section is where a dataset is loaded:

- **Use default QM7-X dataset from Zenodo** (checkbox, unchecked by default) — check it to load the QM7-X equilibrium dataset directly from a fixed Zenodo URL, with no need to enter anything. The file downloads once on first use and is cached locally afterwards; every later run (and every app restart) reuses the cached copy instead of re-downloading it.
- Leave it unchecked to pick a data source instead:
  - **Local** — point at an HDF5 file already on disk.
  - **Upload** — upload an HDF5 file through the browser.
  - **Zenodo** — paste a different Zenodo file URL; the app downloads and caches it locally the same way.

The first load of a given file builds its Parquet indexes (cached under `data/index_cache/`, keyed by a hash of the source file's identity); subsequent loads of the same file reuse the cache. Building the index for the full QM7-X file can take a while the first time — this is expected and only happens once per source file.

The **Machine Learning** page fetches its own pretrained model and validation sets automatically from a separate Zenodo record the first time each is needed, and caches them under `models/` for every run after that — see [Data sources](#data-sources). If a download fails (e.g. no network access), the affected page shows a clear error and stops rather than failing unexpectedly. Unlike the other pages, **Machine Learning** does not require a QM7-X dataset to be loaded first — it only needs its own model file.

### Troubleshooting

9. Common installation and startup issues:

   - **`ModuleNotFoundError` for a package listed in `requirements.txt`** — the environment wasn't activated before installing, or the install step was skipped; re-run steps 3–4 above.
   - **`xgboost` fails to import on macOS** (e.g. an error mentioning `libomp.dylib` not being loaded) — install `libomp` via Homebrew as described above, then restart the app.
   - **The app crashes when unpickling the model or running a prediction on macOS** — see the `libomp.dylib` conflict note above; it is mitigated automatically as long as the app is started with `streamlit run app.py` (not by importing a page module directly).
   - **A "PNG export requires Kaleido" message appears** instead of a downloadable image — `kaleido` is listed in `requirements.txt`; re-run step 4 to make sure it installed correctly.
   - **A page shows "No dataset is currently loaded"** — this is expected behavior, not an error: return to the **Home** page and load a dataset first (default Zenodo dataset, Local, Upload, or a Zenodo URL). The **Machine Learning** page is the only exception and does not require this.
   - **Port already in use** — run `streamlit run app.py --server.port <another_port>` and open the printed URL instead.

## Testing the machine-learning prediction functionality

A ready-to-use sample file, [`test_dftb_dipole_dataset.csv`](test_dftb_dipole_dataset.csv), is included in the repository root so the dipole-moment prediction workflow can be tried out without preparing your own descriptor data.

To use it:

1. Open the app and go directly to the **Machine Learning** page (no QM7-X dataset needs to be loaded first for this page).
2. Switch to the **🔮 New-data prediction** tab.
3. Under "Upload a dataset", upload `test_dftb_dipole_dataset.csv` from the repository root.
4. The app validates that the file contains the 40 required descriptor columns (listed under "Required input format" on that tab, e.g. `FermiEne`, `BandEne`, `TBeig_1`…`TBeig_8`, `TBchg_1`…`TBchg_23`) and shows a preview of the loaded rows.
5. Click **Run prediction** to generate dipole-moment predictions, view the summary metrics and distribution plot, and optionally download the results as a CSV.

This file is a convenience test dataset for exercising the prediction workflow only; it is not part of the QM7-X dataset and is not used anywhere else in the application.

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
app.py                     Navigation shell (st.navigation) + sidebar logo + global footer
home.py                    Landing page; default-dataset checkbox + Local/Upload/Zenodo data-loading controls
pages/
  1_Dataset_Overview.py         Composition, geometries, elements, dataset statistics
  2_Property_Space_Analysis.py  Property distributions, correlation matrix, pairwise plots + 2D/3D viewer
  3_Molecular_Clustering.py     PCA / t-SNE / GMM clustering over QM7-X scalar properties
  4_Machine_Learning.py         XGBoost dipole-moment model: validation, SHAP, new-data prediction
  5_Documentation.py            In-app documentation: how the app works, data sources, references
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
test_dftb_dipole_dataset.csv  Sample CSV for testing the Machine Learning "New-data prediction" tab
```

## Reproducibility notes

- All page-level analysis reads from the cached Parquet indexes (`modules.loader.load_qm7x_index`) rather than re-scanning the source HDF5 file; the raw HDF5 file is read directly only when a page needs full atomic detail for one selected molecule/geometry.
- Molecular Clustering results (Page 3) and SHAP sample selections (Page 4) are computed live from the currently loaded dataset and the properties/parameters the user selects at run time — they are exploratory, parameter-dependent outputs of the tool, not fixed precomputed results shipped with the repository.
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
