# ============================================================
# ML TOOLS -- QUED-XGBoost dipole-moment model
# ============================================================

import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import ( mean_absolute_error, mean_squared_error, r2_score, )

from modules.plot_style import scientific_layout


# ============================================================
# COLORS
# ============================================================

CYAN = "#0891B2"
VIOLET = "#7C3AED"
ORANGE = "#F97316"
MUTED = "#64748B"
TEXT = "#0F172A"


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "FermiEne",
    "BandEne",
    "NumElec",
    "h0Ene",
    "sccEne",
    "3rdEne",
    "repEne",
    "mbdEne",
    "TBdip_norm",
]

FEATURE_NAMES += [f"TBeig_{i}" for i in range(1, 9)]

FEATURE_NAMES += [f"TBchg_{i}" for i in range(1, 24)]

assert len(FEATURE_NAMES) == 40, ( f"FEATURE_NAMES must have exactly 40 entries, got {len(FEATURE_NAMES)}." )

# ============================================================
# FILE SIGNATURE (explicit cache-key helper)
# ============================================================

def get_file_signature(path):
    """
    Lightweight (path, mtime, size) signature for a local file.
    """

    resolved = Path(path)
    stat = resolved.stat()

    return (str(resolved), stat.st_mtime, stat.st_size)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """
    Cached, lazy model load.
    """

    with open(model_path, "rb") as f:
        parameters = pickle.load(f)

    return parameters

# ============================================================
# 40-FEATURE QUED EXTRACTION (shared by validation + new-data paths)
# ============================================================

def _extract_feature_row(properties):
    """
    Build one 40-feature QUED row from an HDF5 conformer group.

    Single shared implementation of the exact feature order used by
    BOTH load_validation_data() (train/test files) and
    load_hdf5_upload() (new-data prediction uploads) -- these used
    to be two independently copy-pasted implementations; this is the
    one place that order is allowed to be defined. Order:

        1-9   : FermiEne, BandEne, NumElec, h0Ene, sccEne, 3rdEne,
                repEne, mbdEne, ||TBdip||
        10-17 : TBeig (8 orbital eigenvalues, unpadded --
                raises if not exactly 8)
        18-40 : TBchg (up to 23 atomic charges, zero-padded on the
                right up to 23)

    Raises ValueError if TBeig isn't exactly 8 values or TBchg has
    more than 23.
    """

    row = [
        float(properties["FermiEne"][()]),
        float(properties["BandEne"][()]),
        float(properties["NumElec"][()]),
        float(properties["h0Ene"][()]),
        float(properties["sccEne"][()]),
        float(properties["3rdEne"][()]),
        float(properties["repEne"][()]),
        float(properties["mbdEne"][()]),
        float(np.linalg.norm(properties["TBdip"][()])),
    ]

    eig = np.asarray(properties["TBeig"][()], dtype=float)

    if len(eig) != 8:
        raise ValueError("TBeig must contain exactly 8 values.")

    row.extend(eig.tolist())

    charges = np.asarray(properties["TBchg"][()], dtype=float)

    if len(charges) > 23:
        raise ValueError( "The model supports a maximum of 23 atomic charges." )

    padded_charges = np.zeros(23)
    padded_charges[: len(charges)] = charges

    row.extend(padded_charges.tolist())

    return row


REQUIRED_HDF5_KEYS = [
    "FermiEne",
    "BandEne",
    "NumElec",
    "h0Ene",
    "sccEne",
    "3rdEne",
    "repEne",
    "mbdEne",
    "TBdip",
    "TBeig",
    "TBchg",
]

# ============================================================
# VALIDATION-FILE LOADING (train/test HDF5 -> X, y)
# ============================================================

@st.cache_data(show_spinner=False)
def load_validation_data(path, mtime, size):
    """
    Parse one QUED validation HDF5 file into (X, y).
    """

    X = []
    y = []

    with h5py.File(path, "r") as dataset:

        for molid in dataset.keys():

            conformers = dataset[molid]

            for confid in conformers.keys():

                properties = conformers[confid]

                X.append(_extract_feature_row(properties))

                y.append(float(properties["dipole_moment"][()]))

    return np.asarray(X), np.asarray(y)


# ============================================================
# CACHED PREDICTIONS FOR A VALIDATION FILE
# ============================================================

@st.cache_data(show_spinner=False)
def get_predictions(path, mtime, size, model_path):
    """
    Cached (X, y, y_pred) for one HDF5 validation file
    """

    X, y = load_validation_data(path, mtime, size)

    parameters = load_model(model_path)
    model = parameters["estimator"]

    y_pred = model.predict(X).flatten()

    return X, y, y_pred


# ============================================================
# NEW-DATA UPLOAD LOADERS (not cached -- single-use per upload)
# ============================================================

def load_csv_upload(uploaded_file, feature_names=FEATURE_NAMES):
    """Load a CSV upload containing the 40 required descriptor columns."""

    df = pd.read_csv(uploaded_file)

    missing = [feature for feature in feature_names if feature not in df.columns]

    if missing:
        raise ValueError( "Missing required columns: " + ", ".join(missing) )

    X = df[feature_names].copy()

    return df, X


# Column names checked (case-insensitively), in priority order, when
# looking for a genuine identifier already present in an uploaded
# CSV. First match wins.
CSV_IDENTIFIER_COLUMNS = ["molecule_id", "id", "name"]


def resolve_upload_identifiers(n_rows, hdf5_identifiers=None, source_df=None):
    """
    Decide the identifier column used for BOTH the "Preview input
    descriptors" table and the final results table/CSV in New-data
    Prediction
    """

    if hdf5_identifiers is not None:
        return [str(v) for v in hdf5_identifiers]

    if source_df is not None:

        lower_to_actual = {col.lower(): col for col in source_df.columns}

        for candidate in CSV_IDENTIFIER_COLUMNS:

            if candidate in lower_to_actual:
                return source_df[lower_to_actual[candidate]].astype(str).tolist()

    return [str(i) for i in range(1, n_rows + 1)]


def load_hdf5_upload(uploaded_file):
    """
    Load an uploaded HDF5 file of new molecules into a feature matrix.
    """

    X = []
    identifiers = []

    with h5py.File(uploaded_file, "r") as dataset:

        for molid in dataset.keys():

            conformers = dataset[molid]

            for confid in conformers.keys():

                properties = conformers[confid]

                missing = [key for key in REQUIRED_HDF5_KEYS if key not in properties]

                if missing:
                    raise ValueError( f"Missing properties in molecule " f"{molid}/{confid}: " + ", ".join(missing) )

                X.append(_extract_feature_row(properties))
                identifiers.append(f"{molid}/{confid}")

    return np.asarray(X), identifiers


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    """R²/MAE/RMSE/N for one (y_true, y_pred) pair -- same formulas as before."""

    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "n": len(y_true),
    }


# ============================================================
# SHARED RANGE HELPER
# ============================================================

def padded_range(*arrays, pad_frac=0.04):
    """
    Union min/max across all given arrays
    """

    values = np.concatenate([np.asarray(a).ravel() for a in arrays])

    span = values.max() - values.min()
    pad = pad_frac * span

    return values.min() - pad, values.max() + pad


def _hex_to_rgba(hex_color, alpha):
    """`#RRGGBB` -> `rgba(r,g,b,alpha)` -- for a translucent tint of an
    already-used color, not a new color."""

    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    return f"rgba({r},{g},{b},{alpha})"


def _panel_annotation( xref, yref, r2_value, mae_value, rmse_value, x=0.03, y=0.97, xanchor="left", label=None, color=None, ):
    """
    Compact R²/MAE/RMSE annotation box, anchored inside one
    subplot/panel.
    """

    header = f"<b>{label}</b><br>" if label is not None else ""

    if color is not None:
        bordercolor = color
        bgcolor = _hex_to_rgba(color, 0.12)
        borderwidth = 1.5
    else:
        bordercolor = MUTED
        bgcolor = "rgba(255,255,255,0.75)"
        borderwidth = 1

    return dict(
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        xanchor=xanchor,
        yanchor="top",
        showarrow=False,
        align="left",
        bgcolor=bgcolor,
        bordercolor=bordercolor,
        borderwidth=borderwidth,
        borderpad=4,
        text=(
            f"{header}"
            f"R² = {r2_value:.4f}<br>"
            f"MAE = {mae_value:.4f}<br>"
            f"RMSE = {rmse_value:.4f}"
        ),
        font=dict(size=11, color=TEXT),
    )


def _lift_subplot_titles(fig, n_titles, delta=0.05):
    """
    Nudge make_subplots()-generated subplot titles further above their plot area.
    """

    for annotation in fig.layout.annotations[:n_titles]:
        annotation.update(y=annotation.y + delta)

    return fig


# ============================================================
# FIGURE BUILDERS
# ============================================================

def build_predicted_vs_actual_figure(
    y_train,
    y_train_pred,
    y_test,
    y_pred,
    metrics_train,
    metrics_test,
    overlay=False,
    height=440,
):
    """Predicted vs actual, training vs. independent test."""

    axis_min, axis_max = padded_range(y_train, y_train_pred, y_test, y_pred)

    if not overlay:

        fig = make_subplots( rows=1, cols=2, subplot_titles=("Training", "Independent test"), )

        fig.add_trace(
            go.Scattergl(
                x=y_train, y=y_train_pred, mode="markers",
                marker=dict(size=4, opacity=0.35, color=CYAN),
                name="Training",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scattergl(
                x=y_test, y=y_pred, mode="markers",
                marker=dict(size=4, opacity=0.35, color=ORANGE),
                name="Test",
            ),
            row=1, col=2,
        )

        for col_index in (1, 2):

            fig.add_trace(
                go.Scatter(
                    x=[axis_min, axis_max], y=[axis_min, axis_max],
                    mode="lines",
                    line=dict(dash="dash", width=2, color=MUTED),
                    # "Ideal (y = x)"
                    
                    name="Ideal (y = x)",
                    showlegend=(col_index == 1),
                ),
                row=1, col=col_index,
            )

        scientific_layout(fig, height=height, show_legend=True, dark_mode=False)

        fig.update_xaxes(title_text="Actual dipole moment (e·Å)", range=[axis_min, axis_max])
        fig.update_yaxes(title_text="Predicted dipole moment (e·Å)", range=[axis_min, axis_max])

        _lift_subplot_titles(fig, n_titles=2)

        fig.update_layout(
            margin=dict(l=20, r=20, t=55, b=20),
            annotations=list(fig.layout.annotations) + [
                _panel_annotation("x domain", "y domain", metrics_train["r2"], metrics_train["mae"], metrics_train["rmse"]),
                _panel_annotation("x2 domain", "y2 domain", metrics_test["r2"], metrics_test["mae"], metrics_test["rmse"]),
            ],
        )

    else:

        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=y_train, y=y_train_pred, mode="markers",
                marker=dict(size=4, opacity=0.30, color=CYAN, symbol="circle"),
                name="Training — actual vs. predicted",
            )
        )

        fig.add_trace(
            go.Scattergl(
                x=y_test, y=y_pred, mode="markers",
                marker=dict(size=5, opacity=0.45, color=ORANGE, symbol="diamond"),
                name="Independent test — actual vs. predicted",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[axis_min, axis_max], y=[axis_min, axis_max],
                mode="lines",
                line=dict(dash="dash", width=2, color=MUTED),
                name="Ideal (y = x)",
            )
        )

        scientific_layout(fig, height=height, show_legend=True, dark_mode=False)

        fig.update_xaxes(title_text="Actual dipole moment (e·Å)", range=[axis_min, axis_max])
        fig.update_yaxes(title_text="Predicted dipole moment (e·Å)", range=[axis_min, axis_max])

        fig.update_layout(
            margin=dict(l=20, r=20, t=70, b=20),
            # label=/color= make each box unambiguous 
            annotations=[
                _panel_annotation(
                    "x domain", "y domain",
                    metrics_train["r2"], metrics_train["mae"], metrics_train["rmse"],
                    y=0.97, label="Training", color=CYAN,
                ),
                _panel_annotation(
                    "x domain", "y domain",
                    metrics_test["r2"], metrics_test["mae"], metrics_test["rmse"],
                    y=0.65, label="Independent test", color=ORANGE,
                ),
            ],
            # Horizontal legend ABOVE the plot 
            legend=dict( orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, ),
        )

    return fig


def build_residuals_figure(
    y_train_pred,
    residuals_train,
    y_pred,
    residuals_test,
    overlay=False,
    height=440,
):
    """Residuals vs predicted, training vs. independent test."""

    x_min, x_max = padded_range(y_train_pred, y_pred)
    y_min, y_max = padded_range(residuals_train, residuals_test)

    if not overlay:

        fig = make_subplots( rows=1, cols=2, subplot_titles=("Training", "Independent test"), )

        fig.add_trace(
            go.Scattergl(
                x=y_train_pred, y=residuals_train, mode="markers",
                marker=dict(size=4, opacity=0.35, color=CYAN),
                name="Training",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scattergl(
                x=y_pred, y=residuals_test, mode="markers",
                marker=dict(size=4, opacity=0.35, color=ORANGE),
                name="Test",
            ),
            row=1, col=2,
        )

        scientific_layout(fig, height=height, show_legend=False, dark_mode=False)

        fig.update_xaxes(title_text="Predicted dipole moment (e·Å)", range=[x_min, x_max])
        fig.update_yaxes(title_text="Residual (e·Å)", range=[y_min, y_max])

        fig.add_hline(y=0, line_dash="dash", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", row=1, col=2)

        # Subplot titles sat flush against the plot area 
        _lift_subplot_titles(fig, n_titles=2)
        fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))

    else:

        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=y_train_pred, y=residuals_train, mode="markers",
                marker=dict(size=4, opacity=0.30, color=CYAN),
                name="Training",
            )
        )

        fig.add_trace(
            go.Scattergl(
                x=y_pred, y=residuals_test, mode="markers",
                marker=dict(size=4, opacity=0.45, color=ORANGE),
                name="Test",
            )
        )

        # show_legend=True here (vs False in separate mode)
        scientific_layout(fig, height=height, show_legend=True, dark_mode=False)

        fig.update_xaxes(title_text="Predicted dipole moment (e·Å)", range=[x_min, x_max])
        fig.update_yaxes(title_text="Residual (e·Å)", range=[y_min, y_max])

        fig.add_hline(y=0, line_dash="dash")

        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    return fig


def build_distribution_figure(
    y_train,
    y_train_pred,
    y_test,
    y_pred,
    overlay=False,
    height=420,
    n_bins=50,
):
    """Actual vs predicted value distributions, training vs. independent test."""

    axis_min, axis_max = padded_range(y_train, y_train_pred, y_test, y_pred)
    bin_size = (axis_max - axis_min) / n_bins
    xbins = dict(start=axis_min, end=axis_max, size=bin_size)

    if not overlay:

        fig = make_subplots( rows=1, cols=2, subplot_titles=("Training", "Independent test"), )

        fig.add_trace(
            go.Histogram(
                x=y_train, xbins=xbins, histnorm="probability density",
                opacity=0.55, marker_color=CYAN, name="Actual", legendgroup="actual",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=y_train_pred, xbins=xbins, histnorm="probability density",
                opacity=0.55, marker_color=VIOLET, name="Predicted", legendgroup="predicted",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=y_test, xbins=xbins, histnorm="probability density",
                opacity=0.55, marker_color=CYAN, name="Actual", legendgroup="actual",
                showlegend=False,
            ),
            row=1, col=2,
        )

        fig.add_trace(
            go.Histogram(
                x=y_pred, xbins=xbins, histnorm="probability density",
                opacity=0.55, marker_color=VIOLET, name="Predicted", legendgroup="predicted",
                showlegend=False,
            ),
            row=1, col=2,
        )

        scientific_layout(fig, height=height, show_legend=True, dark_mode=False)

        fig.update_xaxes(title_text="Dipole moment (e·Å)", range=[axis_min, axis_max])
        fig.update_yaxes(title_text="Density")

        _lift_subplot_titles(fig, n_titles=2)
        fig.update_layout(barmode="overlay", margin=dict(l=20, r=20, t=55, b=20))

    else:

        # 4 series in one panel (train-actual, train-predicted,
        # test-actual, test-predicted). 
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=y_train, xbins=xbins, histnorm="probability density",
                opacity=0.60, marker_color=CYAN, name="Training · Actual",
            )
        )

        fig.add_trace(
            go.Histogram(
                x=y_train_pred, xbins=xbins, histnorm="probability density",
                opacity=0.60, marker_color=VIOLET, name="Training · Predicted",
            )
        )

        fig.add_trace(
            go.Histogram(
                x=y_test, xbins=xbins, histnorm="probability density",
                opacity=0.35, marker_color=CYAN, name="Test · Actual",
            )
        )

        fig.add_trace(
            go.Histogram(
                x=y_pred, xbins=xbins, histnorm="probability density",
                opacity=0.35, marker_color=VIOLET, name="Test · Predicted",
            )
        )

        scientific_layout(fig, height=height, show_legend=True, dark_mode=False)

        fig.update_xaxes(title_text="Dipole moment (e·Å)", range=[axis_min, axis_max])
        fig.update_yaxes(title_text="Density")

        fig.update_layout(barmode="overlay", margin=dict(l=20, r=20, t=40, b=20))

    return fig


def build_residual_distribution_figure(
    residuals_train,
    residuals_test,
    overlay=False,
    height=400,
    n_bins=60,
):
    """Residual distributions, training vs. independent test."""

    r_min, r_max = padded_range(residuals_train, residuals_test)
    r_bin_size = (r_max - r_min) / n_bins
    r_xbins = dict(start=r_min, end=r_max, size=r_bin_size)

    if not overlay:

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Training", "Independent test"),
        )

        fig.add_trace(
            go.Histogram(
                x=residuals_train, xbins=r_xbins, histnorm="probability density",
                marker_color=CYAN, name="Training",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=residuals_test, xbins=r_xbins, histnorm="probability density",
                marker_color=ORANGE, name="Test",
            ),
            row=1, col=2,
        )

        scientific_layout(fig, height=height, show_legend=False,  dark_mode=False)

        fig.update_xaxes(title_text="Residual (e·Å)", range=[r_min, r_max])
        fig.update_yaxes(title_text="Density")

        fig.add_vline(x=0, line_dash="dash", row=1, col=1)
        fig.add_vline(x=0, line_dash="dash", row=1, col=2)

        _lift_subplot_titles(fig, n_titles=2)
        fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))

    else:

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=residuals_train, xbins=r_xbins, histnorm="probability density",
                opacity=0.55, marker_color=CYAN, name="Training",
            )
        )

        fig.add_trace(
            go.Histogram(
                x=residuals_test, xbins=r_xbins, histnorm="probability density",
                opacity=0.55, marker_color=ORANGE, name="Test",
            )
        )

        scientific_layout(fig, height=height, show_legend=True, dark_mode=False)

        fig.update_xaxes(title_text="Residual (e·Å)", range=[r_min, r_max])
        fig.update_yaxes(title_text="Density")

        fig.add_vline(x=0, line_dash="dash")

        fig.update_layout(barmode="overlay", margin=dict(l=20, r=20, t=40, b=20))

    return fig


def build_new_data_distribution_figure(predictions, height=420, n_bins=50):
    """
    Distribution of predictions on a newly uploaded dataset (the New-data Prediction tab). 
    """

    fig = go.Figure()

    fig.add_trace( go.Histogram(x=predictions, nbinsx=n_bins, name="Predictions") )

    scientific_layout(fig, height=height, show_legend=False, dark_mode=False)

    fig.update_xaxes(title_text="Predicted dipole moment (e·Å)")
    fig.update_yaxes(title_text="Number of samples")

    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    return fig


# ============================================================
# SHAP HELPERS
# ============================================================

def compute_shap_values(model, X_shap, feature_names=FEATURE_NAMES):
    """
    Run SHAP TreeExplainer on a sample of already-scaled features.
    """

    import shap

    scaler = model.named_steps["scaler"]
    xgb_model = model.named_steps["model"]

    X_shap_scaled = scaler.transform(X_shap)

    explainer = shap.TreeExplainer(xgb_model, feature_names=feature_names)

    return explainer(X_shap_scaled)


SHAP_FEATURE_LABEL_FONTSIZE = 5
SHAP_AXIS_LABEL_FONTSIZE = 5
SHAP_TICK_LABEL_FONTSIZE = 5


def build_shap_beeswarm_figure(shap_values, max_display=10):
    """
    Build a compact SHAP beeswarm figure with controlled font sizes.
    """

    import shap
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    beeswarm_ax = shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False,
        plot_size=(5.5, 3.5),
    )

    for label in beeswarm_ax.get_yticklabels():
        label.set_fontsize(SHAP_FEATURE_LABEL_FONTSIZE)

    for label in beeswarm_ax.get_xticklabels():
        label.set_fontsize(SHAP_TICK_LABEL_FONTSIZE)

    beeswarm_ax.xaxis.label.set_fontsize(SHAP_AXIS_LABEL_FONTSIZE)

    fig = beeswarm_ax.get_figure()

    # The colorbar SHAP 
    for other_ax in fig.axes:
        if other_ax is not beeswarm_ax:
            other_ax.tick_params(labelsize=SHAP_TICK_LABEL_FONTSIZE)

    plt.tight_layout()

    return fig
