
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from functools import partial
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

# Precomputed validation results, generated separately by
# modules/build_ml_validation_artifacts.py.
ARTIFACTS_DIR = BASE_DIR / "validation_ml_data"
ARTIFACT_NPZ_PATH = ARTIFACTS_DIR / "ml_validation_results.npz"
ARTIFACT_JSON_PATH = ARTIFACTS_DIR / "ml_validation_results.json"

SHAP_MAX_ROWS = ml_tools.SHAP_MAX_ROWS
SHAP_DEFAULT_MAX_DISPLAY = 12
SHAP_SOURCE_PRECOMPUTED = "Precomputed Independent Test sample"
SHAP_SOURCE_ZENODO = "Independent Test sample from Zenodo"

PANEL_TITLES = ("Training", "Independent Test")
PREDICTED_VS_ACTUAL = "Predicted vs Actual"


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
# FIGURE DOWNLOADS AND LAYOUT
# ============================================================

_DEFERRED_DOWNLOADS = "deferred" in (st.download_button.__doc__ or "")


def _panel_download_row(fig, base_filename, key_prefix):

    formats = ( ("PNG", "png", "image/png", {"scale": 2}), ("SVG", "svg", "image/svg+xml", {}), )

    for column, (label, image_format, mime, options) in zip(st.columns(2), formats):

        with column:

            if _DEFERRED_DOWNLOADS:

                # The callable captures this rerun's figure
                data = partial(fig.to_image, format=image_format, **options)

            else:

                try:
                    data = fig.to_image(format=image_format, **options)
                except Exception:
                    st.caption(f"{label} export requires Kaleido.")
                    continue

            st.download_button(
                f"↓ {label}",
                data=data,
                file_name=f"{base_filename}.{image_format}",
                mime=mime,
                width="stretch",
                key=f"{key_prefix}_{image_format}",
            )


def _keep_single_panel(fig, panel):
    """
    Reduce a 1x2 "Training | Independent test" figure from ml_tools
    to a single full-width panel (1 = training, 2 = independent test),
    keeping its traces, axis settings, titles, metric boxes and
    reference lines unchanged.
    """

    def _panel_of(ref):

        if not ref or ref == "paper":
            return None

        return 2 if ref.split()[0].endswith("2") else 1

    def _remap(ref):

        return ref.replace("x2", "x").replace("y2", "y") if ref else ref

    traces = []

    for trace in fig.data:

        if _panel_of(trace.xaxis or "x") != panel:
            continue

        trace = go.Figure(trace).data[0]
        trace.update(xaxis="x", yaxis="y")

        # The test panel's legend entries are hidden in the 1x2 layout, but we want them visible when the test panel is the only one shown.
        if panel == 2:
            trace.showlegend = True

        traces.append(trace)

    layout = fig.layout.to_plotly_json()

    xaxis2 = layout.pop("xaxis2", {})
    yaxis2 = layout.pop("yaxis2", {})

    if panel == 2:
        layout["xaxis"] = xaxis2
        layout["yaxis"] = yaxis2

    layout.setdefault("xaxis", {}).update(domain=[0, 1], anchor="y")
    layout.setdefault("yaxis", {}).update(anchor="x")

    for item_type in ("annotations", "shapes"):

        kept = []

        for item in layout.get(item_type, []):

            item_panel = _panel_of(item.get("xref")) or _panel_of(item.get("yref"))

            if item_panel is None:

                if item.get("text") in PANEL_TITLES:

                    if item["text"] != PANEL_TITLES[panel - 1]:
                        continue

                    item["x"] = 0.5

                kept.append(item)

            elif item_panel == panel:

                item["xref"] = _remap(item.get("xref"))
                item["yref"] = _remap(item.get("yref"))
                kept.append(item)

        layout[item_type] = kept

    return go.Figure(data=traces, layout=layout)


def _apply_validation_figure_layout(fig, diagnostic_type, overlay, two_panels):
    """
    Presentation only: frameless legends inside the plot area and a
    wider gap between the two panels, so the second panel's Y-axis
    clears the first panel. Traces and axis settings are unchanged.
    """

    # Predicted vs Actual 
    if diagnostic_type == PREDICTED_VS_ACTUAL:
        legend_y, legend_yanchor = 0.03, "bottom"
    else:
        legend_y, legend_yanchor = 0.97, "top"

    def _inside_legend(x_start, x_end):

        return dict(
            orientation="v",
            x=x_start + 0.97 * (x_end - x_start),
            xanchor="right",
            y=legend_y,
            yanchor=legend_yanchor,
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0,
            font=fig.layout.legend.font,
        )

    # The colored overlay boxes keep their border: it identifies the dataset.
    if not overlay:
        fig.update_annotations(borderwidth=0)

    if not two_panels:

        fig.update_layout(legend=_inside_legend(0, 1))

        return fig

    first_panel, second_panel = (0, 0.43), (0.57, 1)

    fig.update_layout(
        xaxis_domain=first_panel,
        xaxis2_domain=second_panel,
        legend=_inside_legend(*first_panel),
        legend2=_inside_legend(*second_panel),
    )

    # Subplot titles
    title_centers = dict(zip(PANEL_TITLES, (sum(first_panel) / 2, sum(second_panel) / 2)))

    for annotation in fig.layout.annotations:

        if annotation.xref == "paper" and annotation.text in title_centers:
            annotation.x = title_centers[annotation.text]

    # Each panel gets its own legend with its own entries.
    for trace in fig.data:

        if trace.xaxis == "x2":
            trace.update(legend="legend2", showlegend=True)

    return fig

# ============================================================
# MODEL
# ============================================================

def _get_model():
    """Download (first use only) and load the model: (parameters, model path)."""

    model_path = ensure_zenodo_file_cached(MODEL_URL, cache_dir=str(ASSETS_DIR))

    return ml_tools.load_model(str(model_path)), model_path

# ============================================================
# SHAP SECTION
# ============================================================

def _render_shap_figure(build_figure, file_name, key):

    import matplotlib.pyplot as plt

    fig = build_figure()

    st.pyplot(fig, width="content")

    try:
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")

        st.download_button(
            "↓ Download SHAP figure (PNG)",
            data=png_buf.getvalue(),
            file_name=file_name,
            mime="image/png",
            width="stretch",
            key=key,
        )

    except Exception:
        st.caption("PNG export of the SHAP figure is not available in this environment.")

    plt.close(fig)


def _render_shap_section(artifact_arrays, artifact_metadata):

    section_header_title(
        "Model interpretability",
        "SHAP analysis showing which QUED features contribute most to the prediction.",
        color=VIOLET,
        background=VIOLET_BG,
    )

    st.info(
        "SHAP values quantify the contribution of each descriptor "
        "to the model prediction. The analysis below uses up to the "
        f"first {SHAP_MAX_ROWS} molecules of the Independent-Test set. This is "
        "opt-in: nothing SHAP-related runs until you click "
        "**Run SHAP analysis** below."
    )

    shap_source = st.radio(
        "SHAP data source",
        [SHAP_SOURCE_PRECOMPUTED, SHAP_SOURCE_ZENODO],
        index=0,
        key="ml_shap_source",
        horizontal=True,
        help="Both options use the same first Independent-Test molecules, with the same 40 features in the same order.",
    )

    precomputed_selected = (shap_source == SHAP_SOURCE_PRECOMPUTED)
    precomputed_missing = precomputed_selected and artifact_arrays is None

    if precomputed_missing:
        st.warning("The precomputed sample is unavailable because the validation artifacts are missing or invalid. Select the Zenodo option, or regenerate them.")

    # 25-200 range 
    shap_max_rows = SHAP_MAX_ROWS

    if precomputed_selected and artifact_arrays is not None:
        shap_max_rows = min(SHAP_MAX_ROWS, len(artifact_arrays["X_test_shap"]))

    shap_min_rows = min(25, shap_max_rows)

    sample_col, display_col = st.columns(2)

    with sample_col:

        if shap_max_rows > shap_min_rows:

            shap_sample_size = st.slider(
                "Number of molecules used for SHAP",
                min_value=shap_min_rows,
                max_value=shap_max_rows,
                value=min(50, shap_max_rows),
                step=25 if shap_min_rows == 25 else 1,
                key=f"ml_shap_sample_size_{shap_max_rows}",
            )

        else:

            shap_sample_size = shap_max_rows

            st.caption(f"SHAP uses all {shap_max_rows} available Independent-Test rows.")

    # Display-only settings
    with display_col:

        shap_max_display = st.slider(
            "Maximum features displayed",
            min_value=1,
            max_value=len(ml_tools.FEATURE_NAMES),
            value=SHAP_DEFAULT_MAX_DISPLAY,
            key="ml_shap_max_display",
        )

    shap_plot_type = st.radio(
        "SHAP plot type",
        ["Beeswarm", "Bar", "Both"],
        index=0,
        key="ml_shap_plot_type",
        horizontal=True,
        help=(
        "**Beeswarm:** Shows the distribution and direction of feature contributions "
        "across the selected molecules.\n\n"
        "**Bar:** Shows global feature importance based on mean absolute SHAP values, "
        "without indicating contribution direction.\n\n"
        "**Both:** Displays both visualizations."
         ),
    )

    run_shap_clicked = st.button(
        "Run SHAP analysis",
        type="primary",
        width="stretch",
        key="ml_run_shap_button",
        disabled=shap_sample_size < 1 or precomputed_missing,
    )

    if run_shap_clicked:

        try:

            with st.spinner("Computing SHAP explanations..."):

                rows_match_artifact = None

                if precomputed_selected:

                    test_rows = artifact_arrays["X_test_shap"]

                else:

                    test_path = ensure_zenodo_file_cached(TEST_URL, cache_dir=str(ASSETS_DIR))
                    test_rows = ml_tools.load_first_test_rows(*ml_tools.get_file_signature(test_path))

                    if artifact_arrays is not None:
                        rows_match_artifact = np.array_equal(test_rows, artifact_arrays["X_test_shap"])

                X_shap = test_rows[:shap_sample_size]

                if len(X_shap) == 0:
                    raise ValueError("no valid rows were found in the Independent-Test dataset")

                parameters, model_path = _get_model()

                model_matches_artifact = None

                if artifact_metadata is not None:
                    model_matches_artifact = ml_tools.cached_sha256(*ml_tools.get_file_signature(model_path)) == artifact_metadata["model"].get("sha256")

                shap_values = ml_tools.compute_shap_values(parameters["estimator"], X_shap)

                st.session_state["ml_shap_values"] = shap_values
                st.session_state["ml_shap_sample_size_used"] = len(X_shap)
                st.session_state["ml_shap_source_used"] = shap_source
                st.session_state["ml_shap_rows_match_artifact"] = rows_match_artifact
                st.session_state["ml_shap_model_matches_artifact"] = model_matches_artifact

        except Exception as exc:

            st.error(f"SHAP analysis could not be completed: {exc}")

    if "ml_shap_source_used" in st.session_state:

        shap_values = st.session_state["ml_shap_values"]

        st.caption(f"Showing SHAP values for {st.session_state['ml_shap_sample_size_used']} molecules: {st.session_state['ml_shap_source_used']}.")

        if st.session_state["ml_shap_rows_match_artifact"] is False:
            st.warning("The rows read from the Zenodo Independent-Test dataset differ from the precomputed sample.")

        if st.session_state["ml_shap_model_matches_artifact"] is False:
            st.warning("The model used for SHAP differs from the model recorded in the validation artifacts.")

        plots = {
            "Beeswarm": ( partial(ml_tools.build_shap_beeswarm_figure, shap_values, max_display=shap_max_display), "shap_feature_importance.png", "download_shap_png", ),
            "Bar": ( partial(ml_tools.build_shap_bar_figure, shap_values, max_display=shap_max_display), "shap_feature_importance_bar.png", "download_shap_bar_png", ),
        }

        selected_plots = list(plots) if shap_plot_type == "Both" else [shap_plot_type]

        for column, plot in zip(st.columns(len(selected_plots)), selected_plots):

            with column:
                _render_shap_figure(*plots[plot])


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Machine Learning")

st.write( "Evaluates the trained QUED‑XGBoost model and deploys it for dipole‑moment prediction on new molecular inputs." )

# ============================================================
# LOAD MODEL (for the Model configuration panel)
# ============================================================

# A failed download must not block the precomputed validation results.
model_parameters = None

try:
    model_parameters, _ = _get_model()
except Exception as exc:
    st.warning(f"Could not obtain the model file from Zenodo: {exc}")

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
        "and on the Independent QM7-X Test set.",
        color=CYAN,
        background=CYAN_BG,
    )

    # --------------------------------------------------------
    # Model information 
    # --------------------------------------------------------

    with st.expander("Model configuration"):

        parameters = model_parameters

        c1, c2 = st.columns(2)

        with c1:
            if parameters is not None:
                st.write("**Architecture:**", parameters["architecture"])
                st.write("**Target property:**", parameters["target_property"])
                st.write("**Representation:**", parameters["representation"])
                st.write("**Descriptor size:**", parameters["descriptor_size"])

        with c2:
            if parameters is not None:
                st.write("**Maximum atoms:**", parameters["max_atoms"])
            st.write("**QM features:**", "9 global + 8 orbital + 23 charge")
            st.write("**Total input features:**", len(ml_tools.FEATURE_NAMES))
            if parameters is not None:
                st.write("**Conformer selection:**", parameters["conf_selector"])

        if parameters is None:
            st.caption("The remaining settings are stored in the model file and appear here once the model has been loaded by SHAP analysis or new-data prediction.")


    run_col, info_col = st.columns([1, 3])

    with run_col:
        run_clicked = st.button( "▶ Run validation", type="primary", width="stretch", key="ml_run_validation_button", )
    if run_clicked:
        st.session_state["ml_validation_ran"] = True

    validation_ran = st.session_state.get("ml_validation_ran", False)

    artifact_arrays, artifact_metadata, artifact_issue = None, None, None

    if validation_ran:

        # Precomputed results only (generated separately by
        # modules/build_ml_validation_artifacts.py).
        artifact_arrays, artifact_metadata, artifact_issue = ml_tools.load_validation_artifacts( ARTIFACT_NPZ_PATH, ARTIFACT_JSON_PATH, MODEL_URL, TRAIN_URL, TEST_URL, )

    if not validation_ran:

        with info_col:
            st.info(
                "Nothing has been loaded yet. Click **Run validation** "
                "to display the precomputed validation results."
            )

    elif artifact_arrays is None:

        st.warning(
            f"The precomputed validation results are unavailable: {artifact_issue}. "
            "Generate `validation_ml_data/ml_validation_results.npz` and "
            "`validation_ml_data/ml_validation_results.json` separately with "
            "`modules/build_ml_validation_artifacts.py`."
        )

    else:

        y_train = artifact_arrays["y_train"]
        y_train_pred = artifact_arrays["y_train_pred"]
        y_test = artifact_arrays["y_test"]
        y_pred = artifact_arrays["y_pred"]

        st.caption(
            "Showing precomputed validation results generated on "
            f"{artifact_metadata['created_utc'][:10]}."
        )

        version_changes = ml_tools.library_version_differences(artifact_metadata)

        if version_changes:
            st.caption(
                "Library versions differ from those used to precompute the results: "
                + ", ".join( f"{name} {old} → {new}" for name, (old, new) in version_changes.items() )
            )

        # ----------------------------------------------------
        # Metrics -- computed independently for train and test
        # ----------------------------------------------------

        metrics_train = ml_tools.compute_metrics(y_train, y_train_pred)
        metrics_test = ml_tools.compute_metrics(y_test, y_pred)

        st.info(
            "The **Independent Test Set** remains the primary estimate "
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
                { "Dataset": dataset, "N": metrics["n"], "R²": metrics["r2"], "MAE": metrics["mae"], "RMSE": metrics["rmse"], }
                for dataset, metrics in (("Training", metrics_train), ("Independent Test", metrics_test))
            ]
        )

        with st.expander("Model performance: Training vs. Independent Test"):

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
            "shown for Training and Independent Test data.",
            color=BLUE,
            background=BLUE_BG,
        )

        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_pred

        # Only the selected diagnostic is built
        diagnostics = {
            PREDICTED_VS_ACTUAL: (
                "dipole_predicted_vs_actual_train_test",
                "download_pva",
                lambda overlay: ml_tools.build_predicted_vs_actual_figure(
                    y_train, y_train_pred, y_test, y_pred,
                    metrics_train, metrics_test,
                    overlay=overlay,
                ),
            ),
            "Residuals vs Predicted": (
                "dipole_residuals_train_test",
                "download_residuals",
                lambda overlay: ml_tools.build_residuals_figure(
                    y_train_pred, residuals_train, y_pred, residuals_test,
                    overlay=overlay,
                ),
            ),
            "Actual vs Predicted Distributions": (
                "dipole_distributions_train_test",
                "download_distributions",
                lambda overlay: ml_tools.build_distribution_figure( y_train, y_train_pred, y_test, y_pred, overlay=overlay, ),
            ),
            "Residual Distribution": (
                "dipole_residual_distribution_train_test",
                "download_residual_distribution",
                lambda overlay: ml_tools.build_residual_distribution_figure( residuals_train, residuals_test, overlay=overlay, ),
            ),
        }


        fig_col, controls_col = st.columns( [3, 1], gap="large" )

        with controls_col:

            # ============================================================
            # SELECTION PANEL
            # ============================================================

            st.markdown("<div style='height: 81px'></div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div style="
                    padding: 0.2rem 0 0.8rem 0;
                ">
                <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #0891B2; margin-bottom: 0.3rem; ">
                    Diagnostic display settings
                </div>

                <div style=" font-size: 0.88rem; line-height: 1.5; color: #64748B; ">
                    Choose the diagnostic plot, how the training and independent test results are displayed, and which datasets are included.
                </div>
                </div>
                """,
                unsafe_allow_html=True,
                )

            diagnostic_type = st.radio(
                "Diagnostic type",
                list(diagnostics),
                index=0,
                key="ml_diagnostic_type",
            )

            display_mode = st.radio(
                "Dataset view",
                ["Separate panels", "Overlay Training and Independent Test"],
                index=0,
                key="ml_display_mode",
            )

            overlay = (display_mode == "Overlay Training and Independent Test")

            dataset_selection = st.radio(
                "Datasets shown",
                ["Both", "Training only", "Independent Test only"],
                index=0,
                key="ml_diagnostic_datasets",
                disabled=overlay,
                help="Applies to separate panels; the combined view always shows both.",
            )

        base_filename, key_prefix, build_figure = diagnostics[diagnostic_type]

        fig = build_figure(overlay)

        if not overlay and dataset_selection == "Training only":
            fig = _keep_single_panel(fig, 1)

        elif not overlay and dataset_selection == "Independent Test only":
            fig = _keep_single_panel(fig, 2)

        fig = _apply_validation_figure_layout( fig, diagnostic_type, overlay, two_panels=(not overlay and dataset_selection == "Both"), )

        with fig_col:

            section_header(diagnostic_type)

            st.plotly_chart(fig, width="stretch")

            _panel_download_row(fig, base_filename, key_prefix)

    if validation_ran:

        _render_shap_section(artifact_arrays, artifact_metadata)

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

                    try:
                        model = _get_model()[0]["estimator"]
                    except Exception as exc:
                        st.error(f"Could not obtain the model file from Zenodo: {exc}")
                        st.stop()

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
