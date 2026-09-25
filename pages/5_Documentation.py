import streamlit as st

from modules.theme import apply_global_theme, section_header


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config( page_title="Documentation", page_icon="📚", layout="wide", )

apply_global_theme()

# ============================================================
# TITLE
# ============================================================

st.title("Documentation")
st.write("How QUPEX works, where its data comes from, and how to interpret its results.")

# ============================================================
# LAYOUT
# ============================================================

_, col_main, _ = st.columns([1, 8, 1])

with col_main:

    # ========================================================
    # 1. ABOUT THE APPLICATION
    # ========================================================

    section_header("1. About the Application")

    st.markdown(
        """
        QUPEX (**Qu**antum **P**roperty **EX**plorer) is a web-based scientific application for the
        interactive exploration, visualization, statistical analysis, and machine-learning-based
        prediction of molecular properties, built around the **QM7-X** quantum-chemistry dataset.

        It is important to distinguish between two separate things:

        - **QM7-X** is a quantum-chemistry dataset created and published by its original authors
          (Hoja *et al.*, 2021). QUPEX does not create, modify, or claim authorship of this dataset.
        - **QUPEX** is the software application described on this page. It was developed to load,
          index, explore, and analyze QM7-X (or QM7-X-like) data, and to run a machine-learning
          workflow on top of it.
        """
    )

    # ========================================================
    # 2. WHAT THE APPLICATION PROVIDES
    # ========================================================

    section_header("2. What the Application Provides")

    st.markdown(
        """
        - **Dataset overview** — dataset-wide composition, elemental counts, and geometry statistics.
        - **Molecular property exploration** — statistical distributions, pairwise relationships, and
          correlation analysis over the indexed scalar properties.
        - **Molecular visualization** — per-molecule 2D (RDKit) and 3D (py3Dmol) structure viewers with
          atom-level information.
        - **Chemical-space exploration** — PCA and t-SNE dimensionality reduction with Gaussian Mixture
          Model clustering and cluster-quality evaluation.
        - **Machine-learning workflows** — a pretrained model for dipole-moment prediction, held-out
          validation metrics, and SHAP-based interpretability.
        - **Interactive scientific plots** throughout the application, most of which can be exported as
          images.
        """
    )

    # ========================================================
    # 3. DATA SOURCE
    # ========================================================

    section_header("3. Data Source")

    st.markdown(
        """
        QUPEX does not ship any molecular data. On the **Home** page, a dataset can be obtained in one
        of four ways:

        - **Default QM7-X dataset** — a single checkbox loads the QM7-X equilibrium dataset directly
          from a fixed Zenodo URL, with nothing to configure:
          [`qm7x-eq.hdf5`](https://zenodo.org/records/22893015/files/qm7x-eq.hdf5?download=1)
        - **Zenodo** — a different Zenodo file URL can be entered manually.
        - **Local** — an HDF5 file already available on disk can be opened by path.
        - **Upload** — an HDF5 file can be uploaded directly through the browser.

        Any file downloaded from Zenodo is cached locally and reused on later runs, so the application
        does not download the same file again unless it is removed from the local cache or a different
        source is selected.
        """
    )

    # ========================================================
    # 4. QM7-X DATASET
    # ========================================================

    section_header("4. QM7-X Dataset")

    st.markdown(
        """
        QM7-X is a dataset of quantum-mechanical properties computed for small organic molecules and
        their conformers, distributed as an HDF5 file organized by molecule and geometry.

        QUPEX reads two kinds of information from that HDF5 structure:

        - **Scalar molecular properties** (e.g. energies, HOMO–LUMO gap, dipole moment, polarizability)
          — used for the statistical analysis, distribution, correlation, and chemical-space pages.
        - **Molecular and atomic geometry data** (atomic numbers and coordinates) — used for molecular
          visualization and geometry-related analyses.

        **Reference**

        > J. Hoja, L. Medrano Sandonas, B. G. Ernst, A. Vazquez-Mayagoitia, R. A. DiStasio Jr., and
        > A. Tkatchenko, "QM7-X, a comprehensive dataset of quantum-mechanical properties spanning the
        > chemical space of small organic molecules," *Scientific Data*, vol. 8, no. 43, 2021.
        > DOI: [10.1038/s41597-021-00812-2](https://doi.org/10.1038/s41597-021-00812-2)

        The canonical QM7-X archive is available at
        [10.5281/zenodo.4288677](https://zenodo.org/records/4288677).
        """
    )

    # ========================================================
    # 5. MACHINE-LEARNING DATA
    # ========================================================

    section_header("5. Machine-Learning Data")

    st.markdown(
        """
        The **Machine Learning** page predicts dipole moment using a pretrained scikit-learn
        `Pipeline` (`StandardScaler` + `XGBRegressor`) over a 40-feature descriptor set, and provides
        SHAP `TreeExplainer` interpretability.

        The model and its validation data are fetched from a dedicated Zenodo record:

        - **Model** — [`qued-model_dip.pkl`](https://zenodo.org/records/22893015/files/qued-model_dip.pkl?download=1)
        - **Training data** — [`QMdesc_qm7x-eq_train.h5`](https://zenodo.org/records/22893015/files/QMdesc_qm7x-eq_train.h5?download=1)
        - **Test data** — [`QMdesc_qm7x-eq_test.h5`](https://zenodo.org/records/22893015/files/QMdesc_qm7x-eq_test.h5?download=1)

        These files are not stored in the GitHub repository. The model is downloaded the first time the
        Machine Learning page is opened; the training and test sets are downloaded only when validation
        is run. Every file is cached locally afterwards and is not downloaded again on subsequent runs.

        A separate sample file, `test_dftb_dipole_dataset.csv`, ships in the repository root. It
        contains the 40 required descriptor columns and can be uploaded directly on the "New-data
        prediction" tab to try the prediction workflow without preparing your own data.
        """
    )

    # ========================================================
    # 6. HOW TO USE THE APPLICATION
    # ========================================================

    section_header("6. How to Use the Application")

    st.markdown(
        """
        1. Open the application; it starts on the **Home** page.
        2. Load a dataset — use the default QM7-X dataset, or select Local, Upload, or Zenodo.
        3. Explore dataset composition and geometry statistics on **Dataset Overview**.
        4. Inspect property distributions, correlations, and pairwise relationships on
          **Property Space Analysis**.
        5. Visualize individual molecules in 2D/3D from the same page.
        6. Explore chemical space (PCA / t-SNE / GMM clustering) on **Molecular Clustering**.
        7. Use **Machine Learning** for dipole-moment prediction and model evaluation — a
          sample CSV (`test_dftb_dipole_dataset.csv`, in the repository root) is available
          to try the "New-data prediction" tab without preparing your own descriptor data.
        8. Return to this **Documentation** page for references and data information at any time.
        """
    )

    # ========================================================
    # 7. SCIENTIFIC METHODOLOGY
    # ========================================================

    section_header("7. Scientific Methodology")

    st.markdown(
        """
        QUPEX is intended as an interactive exploratory and analytical interface for QM7-X-format
        data, not as a replacement for the underlying QM7-X scientific publication.

        Every statistical summary, plot, and machine-learning result shown in the application depends
        on the dataset that is currently loaded, the properties selected, and the parameters chosen by
        the user (e.g. sample size, number of clusters, or selected features). Results should be
        interpreted accordingly, alongside the original QM7-X publication.
        """
    )

    # ========================================================
    # 8. REFERENCES
    # ========================================================

    section_header("8. References")

    st.markdown(
        """
        A citation for the QUPEX application itself is in preparation (see the "Reference" link in
        the page footer) and will be added here once finalized.
        """
    )

    # ========================================================
    # 9. SOURCE CODE AND PROJECT INFORMATION
    # ========================================================

    section_header("9. Source Code and Project Information")

    st.markdown(
        """
        - **Web application:** *https://qupex-app.streamlit.app*
        - **Source code (GitHub):** *https://github.com/Mirtha-Pillaca/QUPEX*
        """
    )
