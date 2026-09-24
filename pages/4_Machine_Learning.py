
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st

from modules import ml_tools
from modules.loader import ensure_zenodo_file_cached
from modules.theme import apply_global_theme, section_header, section_header_title

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Machine Learning", page_icon="🧠", layout="wide")

apply_global_theme()

# ============================================================
# PATHS / REMOTE SOURCES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "models"

MODEL_URL = "https://zenodo.org/records/22893015/files/qued-model_dip.pkl?download=1"
TEST_URL = "https://zenodo.org/records/22893015/files/QMdesc_qm7x-eq_test.h5?download=1"
TRAIN_URL = "https://zenodo.org/records/22893015/files/QMdesc_qm7x-eq_train.h5?download=1"


# ============================================================
# COLORS
# ============================================================

CYAN = "#0891B2"
CYAN_BG = "#ECFEFF"

VIOLET = "#7C3AED"
VIOLET_BG = "#F5F3FF"

BLUE = "#2563EB"
BLUE_BG = "#EFF6FF"

ORANGE = "#F97316"
ORANGE_BG = "#FFF7ED"


# ============================================================
# SECTION HEADER
# ============================================================

def _panel_download_row(fig, base_filename, key_prefix):
    """
    PNG + SVG download buttons for a Plotly figure, mirroring the
    try/except pattern already used on Page 1 (Dataset Overview) for
    its size-distribution PNG exports -- a missing/broken Kaleido
    install degrades gracefully to whichever formats actually work,
    instead of crashing the page.
    """

    dl1, dl2 = st.columns(2)

    with dl1:

        try:
            png_bytes = fig.to_image(format="png", scale=2)

            st.download_button(
                "↓ PNG",
                data=png_bytes,
                file_name=f"{base_filename}.png",
                mime="image/png",
                width="stretch",
                key=f"{key_prefix}_png",
            )

        except Exception:
            st.caption("PNG export requires Kaleido.")

    with dl2:

        try:
            svg_bytes = fig.to_image(format="svg")

            st.download_button(
                "↓ SVG",
                data=svg_bytes,
                file_name=f"{base_filename}.svg",
                mime="image/svg+xml",
                width="stretch",
                key=f"{key_prefix}_svg",
            )

        except Exception:
            st.caption("SVG export requires Kaleido.")


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Machine Learning")

st.write( "Evaluates the trained QUED‑XGBoost model and deploys it for dipole‑moment prediction on new molecular inputs." )

# ============================================================
# LOAD MODEL 
# ============================================================

try:
    MODEL_PATH = Path( ensure_zenodo_file_cached(MODEL_URL, cache_dir=str(ASSETS_DIR)) )
except Exception as exc:
    st.error(f"Could not obtain the model file from Zenodo: {exc}")
    st.stop()

parameters = ml_tools.load_model(str(MODEL_PATH))
model = parameters["estimator"]


# ============================================================
# TABS
# ============================================================

tab_validation, tab_prediction = st.tabs(["📊 Model validation", "🔮 New-data prediction"])

# ============================================================
# TAB 1 — VALIDATION
# ============================================================

with tab_validation:

    section_header_title(
        "Model validation",
        "Performance of the trained XGBoost model on the training set "
        "and on the independent QM7-X test set.",
        color=CYAN,
        background=CYAN_BG,
    )

    # --------------------------------------------------------
    # Model information 
    # --------------------------------------------------------

    with st.expander("Model configuration"):

        c1, c2 = st.columns(2)

        with c1:
            st.write("**Architecture:**", parameters["architecture"])
            st.write("**Target property:**", parameters["target_property"])
            st.write("**Representation:**", parameters["representation"])
            st.write("**Descriptor size:**", parameters["descriptor_size"])

        with c2:
            st.write("**Maximum atoms:**", parameters["max_atoms"])
            st.write("**QM features:**", "9 global + 8 orbital + 23 charge")
            st.write("**Total input features:**", len(ml_tools.FEATURE_NAMES))
            st.write("**Conformer selection:**", parameters["conf_selector"])


    if "ml_validation_ran" not in st.session_state:
        st.session_state["ml_validation_ran"] = False

    run_col, info_col = st.columns([1, 3])

    with run_col:
        run_clicked = st.button( "▶ Run validation", type="primary", width="stretch", key="ml_run_validation_button", )
    if run_clicked:
        st.session_state["ml_validation_ran"] = True

    if not st.session_state["ml_validation_ran"]:

        with info_col:
            st.info(
                "Nothing has been loaded yet. Click **Run validation** "
                "to load the training and independent test datasets, "
                "run predictions, and view the diagnostics below."
            )

    else:

        try:
            TRAIN_PATH = Path( ensure_zenodo_file_cached(TRAIN_URL, cache_dir=str(ASSETS_DIR)) )

            TEST_PATH = Path( ensure_zenodo_file_cached(TEST_URL, cache_dir=str(ASSETS_DIR)) )
        except Exception as exc:
            st.error(f"Could not obtain the validation datasets from Zenodo: {exc}")
            st.stop()

        with st.spinner("Loading training and test datasets and running predictions..."):

            train_path_str, train_mtime, train_size = ml_tools.get_file_signature(TRAIN_PATH)
            test_path_str, test_mtime, test_size = ml_tools.get_file_signature(TEST_PATH)

            X_train, y_train, y_train_pred = ml_tools.get_predictions( train_path_str, train_mtime, train_size, str(MODEL_PATH) )

            X_test, y_test, y_pred = ml_tools.get_predictions( test_path_str, test_mtime, test_size, str(MODEL_PATH) )

        # ----------------------------------------------------
        # Metrics -- computed independently for train and test
        # ----------------------------------------------------

        metrics_train = ml_tools.compute_metrics(y_train, y_train_pred)
        metrics_test = ml_tools.compute_metrics(y_test, y_pred)

        st.info(
            "The **independent test set** remains the primary estimate "
            "of generalization performance -- it was never seen during "
            "training. Training performance is shown alongside it for "
            "comparison, not as a claim of model quality on its own."
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Test R²", f"{metrics_test['r2']:.4f}")
        with c2:
            st.metric("Test MAE", f"{metrics_test['mae']:.4f}")
        with c3:
            st.metric("Test RMSE", f"{metrics_test['rmse']:.4f}")
        with c4:
            st.metric("Test samples", f"{metrics_test['n']:,}")

        # ----------------------------------------------------
        # Training vs test comparison table + generalization gap
        # ----------------------------------------------------

        comparison_df = pd.DataFrame(
            [
                {
                    "Dataset": "Training",
                    "N": metrics_train["n"],
                    "R²": metrics_train["r2"],
                    "MAE": metrics_train["mae"],
                    "RMSE": metrics_train["rmse"],
                },
                {
                    "Dataset": "Independent test",
                    "N": metrics_test["n"],
                    "R²": metrics_test["r2"],
                    "MAE": metrics_test["mae"],
                    "RMSE": metrics_test["rmse"],
                },
            ]
        )
        
        with st.expander("Model performance: training vs. independent test"):

            st.dataframe(
                comparison_df.style.format( {"N": "{:,}", "R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"} ),
                width="stretch",
                hide_index=True,
            )

            # Generalization gap
            r2_gap = metrics_train["r2"] - metrics_test["r2"]
            mae_gap = metrics_test["mae"] - metrics_train["mae"]
            rmse_gap = metrics_test["rmse"] - metrics_train["rmse"]

            g1, g2, g3 = st.columns(3)

            with g1:
                st.metric("Generalization gap (R², train − test)", f"{r2_gap:+.4f}")
            with g2:
                st.metric("Generalization gap (MAE, test − train)", f"{mae_gap:+.4f}")
            with g3:
                st.metric("Generalization gap (RMSE, test − train)", f"{rmse_gap:+.4f}")

        # ----------------------------------------------------
        # Performance plots
        # ----------------------------------------------------

        section_header_title(
            "Regression diagnostics",
            "Standard diagnostics for evaluating a regression model, "
            "shown for training and independent test data.",
            color=BLUE,
            background=BLUE_BG,
        )

        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_pred

        display_mode = st.radio(
            "Display mode",
            ["Separate panels", "Overlay training and independent test"],
            index=0,
            horizontal=True,
            key="ml_display_mode",
        )

        overlay = (display_mode == "Overlay training and independent test")

        # ====================================================
        # PREDICTED VS ACTUAL
        # ====================================================

        section_header("Predicted vs actual -- training vs. independent test")

        fig_pva = ml_tools.build_predicted_vs_actual_figure(
            y_train, y_train_pred, y_test, y_pred,
            metrics_train, metrics_test,
            overlay=overlay,
        )

        st.plotly_chart(fig_pva, width="stretch")

        _panel_download_row(fig_pva, "dipole_predicted_vs_actual_train_test", "download_pva")

        # ====================================================
        # RESIDUALS
        # ====================================================


        section_header("Residuals vs predicted -- training vs. independent test")

        fig_resid = ml_tools.build_residuals_figure(
            y_train_pred, residuals_train, y_pred, residuals_test,
            overlay=overlay,
        )

        st.plotly_chart(fig_resid, width="stretch")

        _panel_download_row(fig_resid, "dipole_residuals_train_test", "download_residuals")

        # ----------------------------------------------------
        # DISTRIBUTIONS
        # ----------------------------------------------------

        section_header("Actual vs predicted distributions -- training vs. independent test")


        fig_dist = ml_tools.build_distribution_figure( y_train, y_train_pred, y_test, y_pred, overlay=overlay, )

        st.plotly_chart(fig_dist, width="stretch")

        _panel_download_row(fig_dist, "dipole_distributions_train_test", "download_distributions")

        # ----------------------------------------------------
        # RESIDUAL DISTRIBUTION
        # ----------------------------------------------------

        section_header("Residual distribution -- training vs. independent test")

        fig_resid_dist = ml_tools.build_residual_distribution_figure( residuals_train, residuals_test, overlay=overlay, )

        st.plotly_chart(fig_resid_dist, width="stretch")

        _panel_download_row(fig_resid_dist, "dipole_residual_distribution_train_test", "download_residual_distribution")

        # ====================================================
        # SHAP 
        # ====================================================

        section_header_title(
            "Model interpretability",
            "SHAP analysis showing which QUED features contribute most to the prediction.",
            color=VIOLET,
            background=VIOLET_BG,
        )

        st.info(
            "SHAP values quantify the contribution of each descriptor "
            "to the model prediction. The analysis below uses a "
            "representative subset of the validation set. This is "
            "opt-in: nothing SHAP-related runs until you click "
            "**Run SHAP analysis** below."
        )

        shap_sample_size = st.slider(
            "Number of molecules used for SHAP",
            min_value=25,
            max_value=min(200, len(X_test)),
            value=min(50, len(X_test)),
            step=25,
            key="ml_shap_sample_size",
        )

        run_shap_clicked = st.button(
            "Run SHAP analysis",
            type="primary",
            width="stretch",
            key="ml_run_shap_button",
        )

        if run_shap_clicked:

            with st.spinner("Computing SHAP explanations..."):

                X_shap = X_test[:shap_sample_size]

                shap_values = ml_tools.compute_shap_values(model, X_shap)

                # Stored directly in session_state (not st.cache_data)
                # -- a shap.Explanation built from an in-memory model
                # object isn't something we want st.cache_data hashing
                # the model to key on; session_state is the simpler,
                # correct tool here so the result survives an
                # unrelated rerun without re-running SHAP.
                st.session_state["ml_shap_values"] = shap_values
                st.session_state["ml_shap_sample_size_used"] = shap_sample_size

        if "ml_shap_values" in st.session_state:

            shap_values = st.session_state["ml_shap_values"]

            fig = ml_tools.build_shap_beeswarm_figure(shap_values, max_display=12)

            st.pyplot(fig, width="content")

            try:
                png_buf = io.BytesIO()
                fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")

                st.download_button(
                    "↓ Download SHAP figure (PNG)",
                    data=png_buf.getvalue(),
                    file_name="shap_feature_importance.png",
                    mime="image/png",
                    width="stretch",
                    key="download_shap_png",
                )

            except Exception:
                st.caption("PNG export of the SHAP figure is not available in this environment.")

            import matplotlib.pyplot as plt
            plt.close(fig)


# ============================================================
# TAB 2 — NEW DATA PREDICTION
# ============================================================

with tab_prediction:

    section_header_title(
        "Predict on a new dataset",
        "Upload molecular descriptors compatible with the trained QUED-XGBoost model.",
        color=ORANGE,
        background=ORANGE_BG,
    )


    st.markdown(
        """
        ### What does the model require?

        This model predicts **molecular dipole moment** from a
        40-dimensional quantum-mechanical descriptor.

        Your new dataset must contain the same descriptors used
        during training.
        """
    )

    # --------------------------------------------------------
    # Required input format
    # --------------------------------------------------------

    with st.expander("Required input format", expanded=False):

        st.markdown(
            """
            **Recommended format: HDF5 (`.h5`)**

            Each conformer should contain:

            - `FermiEne`
            - `BandEne`
            - `NumElec`
            - `h0Ene`
            - `sccEne`
            - `3rdEne`
            - `repEne`
            - `mbdEne`
            - `TBdip` — 3-dimensional vector
            - `TBeig` — 8 values
            - `TBchg` — up to 23 atomic charges

            For a CSV file, these are converted to 40 columns:

            `9 global/electronic + 8 orbital + 23 atomic-charge features`.
            """
        )

        st.code("\n".join(ml_tools.FEATURE_NAMES), language="text")

        st.caption("The model does not require the target property for inference.")

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["h5", "hdf5", "csv"],
        help=(
            "Upload an HDF5 dataset containing QUED properties "
            "or a CSV containing the 40 required descriptor columns."
        ),
        width="stretch",
    )

    if uploaded_file is not None:

        st.write(f"**File:** `{uploaded_file.name}`")

        file_extension = Path(uploaded_file.name).suffix.lower()

        try:

            hdf5_identifiers = None

            if file_extension in [".h5", ".hdf5"]:

                X_new, hdf5_identifiers = ml_tools.load_hdf5_upload(uploaded_file)
                input_df = pd.DataFrame(X_new, columns=ml_tools.FEATURE_NAMES)

            elif file_extension == ".csv":

                input_df, X_new = ml_tools.load_csv_upload(uploaded_file)
                X_new = X_new.to_numpy(dtype=float)

            else:

                st.error("Unsupported file format.")
                st.stop()

            # ------------------------------------------------
            # Input validation
            # ------------------------------------------------

            if not np.isfinite(X_new).all():
                st.error("The dataset contains NaN or infinite values.")
                st.stop()

            st.success(
                f"Dataset successfully loaded: "
                f"{len(X_new):,} samples × "
                f"{X_new.shape[1]} features."
            )

            # ------------------------------------------------
            # Molecule identifier 
            # ------------------------------------------------

            identifiers = ml_tools.resolve_upload_identifiers(
                len(X_new),
                hdf5_identifiers=hdf5_identifiers,
                source_df=input_df if file_extension == ".csv" else None,
            )

            preview_df = input_df.copy()
            preview_df.insert(0, "Molecule ID", identifiers)

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            with st.expander("Preview input descriptors"):
                st.dataframe(preview_df.head(10), width="stretch")

            # ------------------------------------------------
            # Prediction 
            # ------------------------------------------------

            if st.button("Run prediction", type="primary", width="stretch"):

                with st.spinner("Running molecular property predictions..."):
                    predictions = model.predict(X_new).flatten()

                # --------------------------------------------
                # Summary
                # --------------------------------------------

                st.markdown("### Prediction summary")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric("Molecules / conformers", f"{len(predictions):,}")
                with c2:
                    st.metric("Mean prediction", f"{predictions.mean():.4f}")
                with c3:
                    st.metric("Minimum", f"{predictions.min():.4f}")
                with c4:
                    st.metric("Maximum", f"{predictions.max():.4f}")

                # --------------------------------------------
                # Distribution
                # --------------------------------------------

                section_header("Prediction distribution")

                fig = ml_tools.build_new_data_distribution_figure(predictions)

                st.plotly_chart(fig, width="stretch")

                # --------------------------------------------
                # Results table 
                # --------------------------------------------

                results = pd.DataFrame( { "Molecule ID": identifiers, "predicted_dipole_moment": predictions, } )

                with st.expander("Predicted molecular dipole moments"):

                    st.dataframe(results, width="stretch")

                # --------------------------------------------
                # Download
                # --------------------------------------------

                csv_output = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download predictions as CSV",
                    data=csv_output,
                    file_name="dipole_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )

        except Exception as exc:

            st.error("The dataset could not be processed.")
            st.exception(exc)
