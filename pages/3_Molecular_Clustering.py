
import os

# Must be set before scikit-learn (and its OpenMP-using dependencies

os.environ.setdefault( "KMP_DUPLICATE_LIB_OK", "TRUE", )

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from modules.datasets import (PROPERTY_INFO, get_available_properties, get_property_label, )
from modules.plot_style import (
    scientific_layout,
    SEQUENTIAL_COLORSCALE_STOPS,
    use_webgl_for_large_scatter_total,
    get_qualitative_color,
)
from modules.theme import apply_global_theme, COLORS, section_header
import plotly.graph_objects as go
from scipy.stats import chi2

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config( page_title="Chemical Space", page_icon="🧬", layout="wide", )

apply_global_theme()

# ============================================================
# TITLE
# ============================================================

st.title("Molecular Clustering")

st.write("Explores chemical-space structure through PCA and t-SNE dimensionality reduction. A Gaussian Mixture Model (GMM) is fitted independently to each resulting 2D projection to identify molecular clusters.")

# ============================================================
# CHECK DATASET
# ============================================================

if "hdf" not in st.session_state:

    st.warning("⚠️ **No dataset is currently loaded.** ")
    st.info("ℹ️ Please return to the Home page and load a dataset from a local file, an uploaded file, or Zenodo."
        )
    st.stop()

if not st.session_state.get( "analysis_indexes_ready", False ):

    st.warning( "⚠️ **Dataset analysis indexes are not ready.**" )
    st.info( "ℹ️ Please return to the Home page and load the dataset again." )
    st.stop()

dataset_id = (
    st.session_state.get("dataset_id")
    or st.session_state.get("dataset_source_path")
    or st.session_state.get("dataset_name")
    or "current_dataset"
)


# ============================================================
# ANALYSIS DATASET
# ============================================================

clustering_analysis_scope = st.session_state.get( "clustering_analysis_scope", "Molecules", )

if clustering_analysis_scope is None:
    clustering_analysis_scope = "Molecules"

if clustering_analysis_scope == "Conformers":

    df = st.session_state.get("geometry_index")
    scope_unit_label = "conformers"

else:

    df = st.session_state.get("opt_index")
    scope_unit_label = "molecules"

if df is None or df.empty:

    st.warning( f"The {scope_unit_label} analysis index is empty or unavailable for the active dataset." )

    st.stop()

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_PROPERTIES = [ "mPOL", "eMBD", "eAT", "HLgap", ]

GMM_COMPONENT_MIN = 2
GMM_COMPONENT_MAX = 10
DEFAULT_GMM_COMPONENTS = 4
DEFAULT_RANDOM_SEED = 42


GMM_COVARIANCE_TYPE = "full"
GMM_N_INIT = 5

GMM_REG_COVAR = 1e-4

ELLIPSE_CONFIDENCE_OPTIONS = [68, 95]
DEFAULT_ELLIPSE_CONFIDENCE_PCT = 95

DEFAULT_POINT_SIZE = 6

TSNE_MAX_ITER = 1000


available_properties = get_available_properties(df)

if len(available_properties) < 2:

    st.error( "At least two numeric properties are required for chemical-space analysis." )

    st.stop()


properties_key = f"chemical_space_properties_{dataset_id}"

pca_gmm_key = "pca_gmm_components"
tsne_gmm_key = "tsne_gmm_components"
tsne_perplexity_key = "tsne_perplexity"
tsne_seed_key = "tsne_seed"

default_properties = [ prop for prop in DEFAULT_PROPERTIES if prop in available_properties ]

if len(default_properties) < 2:
    default_properties = available_properties[:5]

selected_properties = st.session_state.get( properties_key, default_properties, )

selected_properties = [ prop for prop in selected_properties if prop in available_properties ] or default_properties

pca_gmm_components = st.session_state.get( pca_gmm_key, DEFAULT_GMM_COMPONENTS, )

tsne_gmm_components = st.session_state.get( tsne_gmm_key, DEFAULT_GMM_COMPONENTS, )

tsne_perplexity = st.session_state.get( tsne_perplexity_key, 30, )

tsne_seed = st.session_state.get( tsne_seed_key, DEFAULT_RANDOM_SEED, )

# ============================================================
# VALIDATE PROPERTY SELECTION
# ============================================================

if len(selected_properties) < 2:

    st.warning(
        "Select at least two properties/features below -- "
        "at least two are required for PCA and t-SNE."
    )
    st.stop()


# ============================================================
# ANALYSIS PIPELINE -- CACHED, INDEPENDENT PER STAGE
# ============================================================
#
# Every function below is cached on its own actual inputs, never on a visualization setting. 

@st.cache_data(show_spinner=False)
def prepare_data(df, properties, color_by_properties):
    """
    Extract the selected numeric descriptors for the *complete*
    active dataset (no sampling of any kind).
    """

    properties = list(properties)
    color_by_properties = list(color_by_properties)

    required_columns = [ prop for prop in properties if prop in df.columns ]

    if len(required_columns) < 2:

        raise ValueError( "Fewer than two selected properties are present in the dataset." )

    extra_columns = [ prop for prop in color_by_properties if prop in df.columns and prop not in required_columns ]

    columns = required_columns + extra_columns

    if "molecule_id" in df.columns:

        columns.insert( 0, "molecule_id", )

    if ( "n_atoms" in df.columns and "n_atoms" not in columns ):

        columns.append("n_atoms")

    data = df.loc[:, columns].copy()

    for prop in required_columns + extra_columns:

        data[prop] = pd.to_numeric( data[prop], errors="coerce", )

    if "n_atoms" in data.columns:

        data["n_atoms"] = pd.to_numeric( data["n_atoms"], errors="coerce", )

    data = data.replace( [np.inf, -np.inf], np.nan, )

    # Only the properties actually feeding PCA/t-SNE can drop a molecule from the analysis
    data = data.dropna( subset=required_columns, )

    data = data.reset_index( drop=True, )

    if "molecule_id" not in data.columns:

        data["molecule_id"] = ( data.index.astype(str) )

    molecule_ids = ( data["molecule_id"] .astype(str) .to_numpy() )
    X = data[ required_columns ].to_numpy( dtype=float, )

    return ( data, X, molecule_ids, required_columns, )

try:

    ( analysis_data, X, molecule_ids, selected_properties, ) = prepare_data( df, tuple(selected_properties), tuple(available_properties), )

except Exception as exc:

    st.error( f"Could not prepare analysis data: {exc}" )
    st.stop()

if len(X) < 10:

    st.error( f"Not enough valid {scope_unit_label} remain after cleaning." )

    st.stop()


@st.cache_data(show_spinner=False)
def standardize_data(X):
    """Standardize molecular descriptors."""

    scaler = StandardScaler()

    return scaler.fit_transform(X)


X_scaled = standardize_data(X)

# ------------------------------------------------------------
# PCA (always 2 components).
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def calculate_pca(X_scaled):

    pca = PCA( n_components=2, )

    coordinates = pca.fit_transform( X_scaled, )

    return ( coordinates, pca.explained_variance_ratio_, pca.components_, pca, )

( X_pca, pca_explained_variance, pca_components, pca_model, ) = calculate_pca( X_scaled, )

# ------------------------------------------------------------
# t-SNE -- ALWAYS on the complete standardized descriptor matrix.
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def calculate_tsne_full( X_scaled, perplexity, random_state, ):
    """
    Two-dimensional t-SNE projection of the *complete* dataset.
    """

    n_samples = len( X_scaled, )

    if n_samples < 10:

        raise ValueError( "t-SNE requires at least 10 molecules." )

    max_perplexity = max( 2, (n_samples - 1) // 3, )

    actual_perplexity = min( perplexity, max_perplexity, )

    t0 = time.time()

    try:

        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            learning_rate="auto",
            max_iter=TSNE_MAX_ITER,
            random_state=random_state,
            init="pca",
        )

    except TypeError:

        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            learning_rate="auto",
            n_iter=TSNE_MAX_ITER,
            random_state=random_state,
            init="pca",
        )

    coordinates = tsne.fit_transform( X_scaled, )

    runtime_seconds = time.time() - t0

    return coordinates, runtime_seconds


with st.spinner(
    f"Running t-SNE on all {len(X_scaled):,} {scope_unit_label} "
    "(this is a one-time cost per configuration -- cached "
    "afterward; no sampling is used)..."
):

    X_tsne, tsne_runtime_seconds = calculate_tsne_full( X_scaled, tsne_perplexity, tsne_seed, )

# ------------------------------------------------------------
# GMM -- fit directly on each projection's OWN 2D coordinates.
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def calculate_gmm_2d( X_2d, n_components, covariance_type, n_init, random_state, reg_covar=GMM_REG_COVAR, ):
    """
    Fit a Gaussian Mixture Model directly on 2D projection
    coordinates (PCA or t-SNE -- this function has no idea which,
    by design, since the two must never share state).
    """

    model = GaussianMixture( n_components=n_components, covariance_type=covariance_type, n_init=n_init, random_state=random_state, reg_covar=reg_covar, )

    model.fit(X_2d)

    labels = model.predict(X_2d)

    probabilities = model.predict_proba(X_2d)

    max_probability = probabilities.max(axis=1)

    return ( model, labels, probabilities, max_probability, )


@st.cache_data(show_spinner=False)
def calculate_silhouette(X_2d, labels):

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) >= len(X_2d):
        return np.nan

    return silhouette_score(X_2d, labels)


@st.cache_data(show_spinner=False)
def calculate_gmm_model_selection( X_2d, component_range, covariance_type, n_init, random_state, reg_covar=GMM_REG_COVAR, ):
    """AIC/BIC/silhouette for several component counts, on 2D data."""

    records = []

    for n_components in component_range:

        try:

            model = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                n_init=n_init,
                random_state=random_state,
                reg_covar=reg_covar,
            )

            model.fit(X_2d)

            labels = model.predict(X_2d)

            silhouette = calculate_silhouette(X_2d, labels)

            records.append( { "Components": n_components, "AIC": model.aic(X_2d), "BIC": model.bic(X_2d), "Silhouette": silhouette, } )

        except Exception:

            records.append( { "Components": n_components, "AIC": np.nan, "BIC": np.nan, "Silhouette": np.nan, } )

    return pd.DataFrame(records)

# --- PCA's own, independent GMM -----------------------------

try:

    ( pca_gmm_model, pca_gmm_labels, pca_gmm_probabilities, pca_gmm_max_probability, ) = calculate_gmm_2d( X_pca, pca_gmm_components, GMM_COVARIANCE_TYPE, GMM_N_INIT, DEFAULT_RANDOM_SEED, )

except Exception as exc:

    st.error(
        f"Could not fit the PCA GMM with {pca_gmm_components} "
        f"components: {exc}. Try a lower \"Number of GMM "
        "components\" value for PCA."
    )

    st.stop()

pca_gmm_silhouette = calculate_silhouette( X_pca, pca_gmm_labels, )

pca_gmm_aic = pca_gmm_model.aic(X_pca)
pca_gmm_bic = pca_gmm_model.bic(X_pca)


# --- t-SNE's own, independent GMM ---------------------------

try:

    ( tsne_gmm_model, tsne_gmm_labels, tsne_gmm_probabilities, tsne_gmm_max_probability, ) = calculate_gmm_2d( X_tsne, tsne_gmm_components, GMM_COVARIANCE_TYPE, GMM_N_INIT, DEFAULT_RANDOM_SEED, )

except Exception as exc:

    st.error(
        f"Could not fit the t-SNE GMM with {tsne_gmm_components} "
        f"components: {exc}. Try a lower \"Number of GMM "
        "components\" value for t-SNE."
    )

    st.stop()

tsne_gmm_silhouette = calculate_silhouette( X_tsne, tsne_gmm_labels, )

tsne_gmm_aic = tsne_gmm_model.aic(X_tsne)
tsne_gmm_bic = tsne_gmm_model.bic(X_tsne)


# ============================================================
# VISUALIZATION HELPERS 
# ============================================================

def property_name(prop):
    """Readable property name."""

    return PROPERTY_INFO.get( prop, {}, ).get( "name", prop, )


def color_label(value):
    """Readable label for a 'Color by' selection."""

    if value == "Cluster":
        return "GMM Cluster"

    return property_name(value)


def _hex_to_rgba(hex_color, alpha):
    """
    '#RRGGBB' -> 'rgba(r,g,b,alpha)'.

    Needed so an ellipse's FILL can fade independently of its
    OUTLINE: Plotly's trace-level `opacity` dims a Scatter trace's
    line and fill together, but `fillcolor` accepts its own alpha
    channel -- baking the requested opacity into `fillcolor` (and
    leaving the trace/line at full strength) is what keeps the
    ellipse outline readable even at low fill opacity.
    """

    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f"rgba({r},{g},{b},{alpha})"


def gmm_ellipse_xy(mean, covariance, confidence=0.95, n_points=200):
    """
    (x, y) points tracing a GMM component's confidence ellipse,
    directly in whatever 2D space `mean`/`covariance` already live
    in (PCA or t-SNE coordinates) -- no projection needed, since
    both GMMs here are fitted directly on 2D projection output.
    """

    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0

    mean = np.asarray(mean, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    eigenvalues = np.clip(eigenvalues, 0.0, None)

    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    radius = np.sqrt(chi2.ppf(confidence, df=2))

    theta = np.linspace(0, 2 * np.pi, n_points)

    circle = np.vstack( [np.cos(theta), np.sin(theta)] )

    scaled = ( eigenvectors @ np.diag(np.sqrt(eigenvalues) * radius) @ circle )

    x = scaled[0] + mean[0]
    y = scaled[1] + mean[1]

    return x, y


def build_projection_figure( plot_df, x_col, y_col, x_title, y_title, gmm_model, gmm_labels, gmm_max_probability, color_by, color_options, show_ellipses, ellipse_confidence, ellipse_opacity, show_centers, point_size, cluster_order, figure_key, ):
    """
    Build one projection's scatter plot (PCA or t-SNE) -- identical
    logic for both, parameterized by which 2D columns/GMM/labels to
    use. Visualization-only arguments (color_by, show_ellipses,
    ellipse_confidence, ellipse_opacity, show_centers, point_size)
    only ever affect this uncached function -- never the cached
    analysis pipeline above.
    """

    plot_df = plot_df.copy()

    plot_df["Cluster"] = gmm_labels.astype(str)
    plot_df["Membership probability"] = gmm_max_probability

    hover_data = {
        "molecule_id": True,
        x_col: ":.4f",
        y_col: ":.4f",
        # Always show GMM assignment confidence in the hover, since
        # it was already computed and otherwise silently dropped.
        "Membership probability": ":.1%",
    }

    cluster_color_sequence = [ get_qualitative_color(i) for i in range(len(cluster_order)) ]

    if color_by == "Cluster":

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="Cluster",
            category_orders={"Cluster": cluster_order},
            color_discrete_sequence=cluster_color_sequence,
            hover_data=hover_data,
        )

        fig.update_layout(legend_title_text="GMM Cluster")

    else:

        hover_data = {**hover_data, "Cluster": True}

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_by,
            color_continuous_scale=SEQUENTIAL_COLORSCALE_STOPS,
            labels={color_by: get_property_label(color_by)},
            hover_data=hover_data,
        )

    fig.update_traces( marker=dict( size=point_size, opacity=0.72, ), selector=dict(mode="markers"), )

    if show_ellipses:

        for component in range(gmm_model.n_components):

            # Same deterministic sequence used for the point colors
            # above, so an ellipse always matches its own cluster's
            # point color -- including past 8 components, where the
            # plain QUALITATIVE_COLORS list alone would repeat.
            component_color = get_qualitative_color(component)

            ellipse_x, ellipse_y = gmm_ellipse_xy(
                gmm_model.means_[component],
                gmm_model.covariances_[component],
                confidence=ellipse_confidence,
            )

            fig.add_trace(
                go.Scatter(
                    x=ellipse_x,
                    y=ellipse_y,
                    mode="lines",
                    line=dict(
                        color=component_color,
                        width=2.5,
                    ),
                    fill="toself",
                    fillcolor=_hex_to_rgba(
                        component_color,
                        ellipse_opacity,
                    ),
                    name=f"GMM {component} ({int(ellipse_confidence * 100)}%)",
                    hoverinfo="skip",
                    legendgroup=f"gmm_{component}",
                    # Avoid a redundant legend entry when points are
                    # already colored by the same GMM cluster.
                    showlegend=(color_by != "Cluster"),
                )
            )

            if show_centers:

                fig.add_trace(
                    go.Scatter(
                        x=[gmm_model.means_[component][0]],
                        y=[gmm_model.means_[component][1]],
                        mode="markers",
                        marker=dict(
                            size=11,
                            symbol="x",
                            color=component_color,
                            line=dict(width=2),
                        ),
                        name=f"GMM {component} center",
                        hovertemplate=(
                            f"GMM component {component}"
                            f"<br>{x_title}=%{{x:.3f}}"
                            f"<br>{y_title}=%{{y:.3f}}"
                            "<extra></extra>"
                        ),
                        legendgroup=f"gmm_{component}",
                        showlegend=False,
                    )
                )

    fig.update_layout( height=560, xaxis_title=x_title, yaxis_title=y_title, )

    fig = scientific_layout(fig)

    fig.update_layout( legend=dict( bgcolor="rgba(0,0,0,0)", borderwidth=0, ), )

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")

    fig = use_webgl_for_large_scatter_total(fig)

    st.plotly_chart( fig, width='stretch', key=figure_key, )


# ============================================================
#TABS
# ============================================================

tab_chemical_analysis = st.tabs( ["🧩 PCA vs t-SNE (GMM)"] )[0]



with tab_chemical_analysis:

    col_scope_info, col_scope_control = st.columns( [3, 1], gap="medium", )

    with col_scope_info:

        st.caption(
            "Choose whether the chemical-space analysis below "
            "describes unique molecules or every conformer record."
        )

    with col_scope_control:

        st.segmented_control(
            "Analysis scope",
            options=[ "Molecules", "Conformers", ],
            default="Molecules",
            key="clustering_analysis_scope",
            label_visibility="collapsed",
        )

    # ============================================================
    # TOP-LEVEL SUMMARY METRICS
    # ============================================================

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = ( st.columns(5) )

    with metric_col1:

        st.metric( f"{scope_unit_label.capitalize()} analyzed", f"{len(analysis_data):,}", help=( f"Every valid {scope_unit_label[:-1]} in the active dataset for the " "properties selected below -- never a sample or subset." ), )

    with metric_col2:

        st.metric( "Properties / features", f"{len(selected_properties)}", help="Number of input variables PCA and t-SNE operate on.", )

    with metric_col3:

        st.metric( "PCA variance (2D)", f"{pca_explained_variance[:2].sum() * 100:.1f}%", help="Cumulative explained variance of the 2D PCA projection shown below.", )
    with metric_col4:

        st.metric( "PCA GMM silhouette", ( "—" if np.isnan(pca_gmm_silhouette) else f"{pca_gmm_silhouette:.3f}" ), help="From the GMM fitted directly on the PCA coordinates.", )

    with metric_col5:

        st.metric( "t-SNE GMM silhouette", ( "—" if np.isnan(tsne_gmm_silhouette) else f"{tsne_gmm_silhouette:.3f}" ), help="From a SEPARATE GMM fitted directly on the t-SNE coordinates.", )

    # ============================================================
    # SHARED ANALYSIS CONTROLS
    # ============================================================

    selected_properties_widget = st.multiselect(
        "Properties / features",
        options=available_properties,
        default=default_properties,
        format_func=lambda prop: (
            f"{PROPERTY_INFO.get(prop, {}).get('name', prop)} "
            f"({prop})"
        ),
        help=(
            "At least two are required. PCA reduces these to 2D for "
            "plotting; t-SNE embeds them directly in 2D; both "
            "independent GMMs cluster on their own 2D projection, not "
            "on these raw properties."
        ),
        key=properties_key,
    )

    st.caption(
        f"**{len(selected_properties)}** properties/features selected "
        f"for this analysis."
    )

    # ============================================================
    # PCA / t-SNE SIDE BY SIDE
    # ============================================================

    col_pca, col_tsne = st.columns(2, gap="large")

    # ------------------------------------------------------------
    # PCA COLUMN
    # ------------------------------------------------------------

    with col_pca:

        st.markdown(
            f"""
            <div style=" margin-top: 1.8rem; margin-bottom: 1rem; padding: 0.65rem 1rem; background: none; border-left: 4px ; border-radius: 6px; ">
            <div style=" font-size: 1.25rem; font-weight: 700; color: #0F172A; ">
                    PCA Analysis
                </div>
            <div style=" font-size: 0.85rem; color: #64748B; margin-top: 0.15rem; ">
                Principal Component Analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

        # -------------------------------------------------------
        # PCA analysis controls
        # -------------------------------------------------------

        with st.expander("PCA controls", expanded=False):

            st.caption("Analysis controls")

            pca_gmm_components_widget = st.slider(
                "GMM components",
                min_value=GMM_COMPONENT_MIN,
                max_value=GMM_COMPONENT_MAX,
                value=DEFAULT_GMM_COMPONENTS,
                step=1,
                key=pca_gmm_key,
                help=(
                    "Clusters a SEPARATE Gaussian Mixture Model fits "
                    "directly on the 2D PCA coordinates -- independent of "
                    "the t-SNE GMM, and independent of the number of "
                    "properties/features selected above."
                ),
            )

            # -------------------------------------------------------
            # PCA visualization controls
            # -------------------------------------------------------

            st.caption("Visualization controls")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:

                pca_color_options = ["Cluster"] + available_properties

                pca_color_by = st.selectbox(
                    "Color by",
                    options=pca_color_options,
                    format_func=color_label,
                    key="pca_color_by_" + dataset_id,
                )

                pca_show_ellipses = st.checkbox( "Show Gaussian ellipses", value=False, key="pca_show_ellipses", )

                pca_show_centers = st.checkbox( "Show cluster centers", value=False, key="pca_show_centers", )

            with viz_col2:

                pca_ellipse_confidence_pct = st.selectbox(
                    "Ellipse confidence",
                    options=ELLIPSE_CONFIDENCE_OPTIONS,
                    format_func=lambda x: f"{x}%",
                    index=ELLIPSE_CONFIDENCE_OPTIONS.index( DEFAULT_ELLIPSE_CONFIDENCE_PCT ),
                    key="pca_ellipse_confidence",
                )

                pca_ellipse_opacity = st.slider( "Ellipse fill opacity", min_value=0.0, max_value=0.6, value=0.16, step=0.02, key="pca_ellipse_opacity", )

                pca_point_size = st.slider( "Point size", min_value=3, max_value=12, value=DEFAULT_POINT_SIZE, step=1, key="pca_point_size", )

        # -------------------------------------------------------
        # PCA plot
        # -------------------------------------------------------

        pca_cluster_order = [ str(c) for c in range(pca_gmm_components) ]

        pca_plot_df = analysis_data.copy()
        pca_plot_df["PC1"] = X_pca[:, 0]
        pca_plot_df["PC2"] = X_pca[:, 1]

        build_projection_figure(
            plot_df=pca_plot_df,
            x_col="PC1",
            y_col="PC2",
            x_title=f"PC1 ({pca_explained_variance[0] * 100:.1f}%)",
            y_title=f"PC2 ({pca_explained_variance[1] * 100:.1f}%)",
            gmm_model=pca_gmm_model,
            gmm_labels=pca_gmm_labels,
            gmm_max_probability=pca_gmm_max_probability,
            color_by=pca_color_by,
            color_options=pca_color_options,
            show_ellipses=pca_show_ellipses,
            ellipse_confidence=pca_ellipse_confidence_pct / 100,
            ellipse_opacity=pca_ellipse_opacity,
            show_centers=pca_show_centers,
            point_size=pca_point_size,
            cluster_order=pca_cluster_order,
            figure_key="pca_main_plot",
        )

        # -------------------------------------------------------
        # PCA cluster summary
        # -------------------------------------------------------

        st.caption("PCA GMM clusters")

        pca_cluster_summary = ( pd.Series(pca_gmm_labels, name="Cluster") .value_counts() .sort_index() .rename("Molecules") .reset_index() )

        pca_cluster_summary["Fraction"] = ( pca_cluster_summary["Molecules"] / len(pca_gmm_labels) ).map(lambda x: f"{x:.1%}")

        pca_cluster_summary["Cluster"] = ( "PCA " + pca_cluster_summary["Cluster"].astype(str) )

        st.dataframe( pca_cluster_summary, width='stretch', hide_index=True, )

        # -------------------------------------------------------
        # PCA diagnostics
        # -------------------------------------------------------

        with st.expander("▼ PCA diagnostics"):

            st.markdown("**Selected properties**")

            st.write( ", ".join( property_name(p) for p in selected_properties ) )

            st.markdown("**Preprocessing**")

            st.caption( "Standardized (zero mean, unit variance) via scikit-learn's StandardScaler before PCA." )

            st.markdown("**Explained variance**")

            pca_variance_df = pd.DataFrame(
                {
                    "Component": ["PC1", "PC2"],
                    "Explained variance (%)": ( pca_explained_variance[:2] * 100 ),
                    "Cumulative variance (%)": np.cumsum( pca_explained_variance[:2] * 100 ),
                }
            )

            st.dataframe( pca_variance_df, width='stretch', hide_index=True, )

            st.caption(
                f"{len(analysis_data):,} {scope_unit_label} analyzed · "
                f"{len(selected_properties)} properties reduced to "
                "2 PCA dimensions for this plot."
            )

        # -------------------------------------------------------
        # PCA GMM diagnostics
        # -------------------------------------------------------

        with st.expander("▼ GMM diagnostics (PCA space)"):

            gmm_diag_col1, gmm_diag_col2 = ( st.columns(2) )

            with gmm_diag_col1:
                st.metric("Components", pca_gmm_components)

                st.metric("AIC", f"{pca_gmm_aic:,.0f}")

            with gmm_diag_col2:
                st.metric("BIC", f"{pca_gmm_bic:,.0f}")

                st.metric( "Silhouette", ( "—" if np.isnan(pca_gmm_silhouette) else f"{pca_gmm_silhouette:.3f}" ), )

            st.caption(
                "Converged: "
                f"{'yes' if pca_gmm_model.converged_ else 'no'} "
                f"in {pca_gmm_model.n_iter_} iterations."
            )

            st.markdown("**Cluster weights**")

            pca_weights_df = pd.DataFrame( { "Cluster": [ f"PCA {i}" for i in range(pca_gmm_components) ], "Weight": pca_gmm_model.weights_, } )

            st.dataframe( pca_weights_df, width='stretch', hide_index=True, )

            st.markdown("**Model selection (2-10 components)**")

            pca_component_range = tuple( range(GMM_COMPONENT_MIN, GMM_COMPONENT_MAX + 1) )

            pca_selection_df = calculate_gmm_model_selection(
                X_pca,
                pca_component_range,
                GMM_COVARIANCE_TYPE,
                GMM_N_INIT,
                DEFAULT_RANDOM_SEED,
            )

            def _highlight_selected(row, selected):

                if row["Components"] == selected:

                    return [ f"background-color: {COLORS.get('accent_bg', '#FFF3CD')}" ] * len(row)

                return [""] * len(row)

            st.dataframe( pca_selection_df.style.apply( _highlight_selected, selected=pca_gmm_components, axis=1, ), width='stretch', hide_index=True, )

            st.caption(
                "The highlighted row is the currently selected "
                "\"Number of GMM components\" value above."
            )

    # ------------------------------------------------------------
    # t-SNE COLUMN
    # ------------------------------------------------------------

    with col_tsne:

        st.markdown(
                    f"""
                    <div style=" margin-top: 1.8rem; margin-bottom: 1rem; padding: 0.65rem 1rem; background: none; border-left: 4px ; border-radius: 6px; ">
                    <div style=" font-size: 1.25rem; font-weight: 700; color: #0F172A; ">
                            t-SNE Analysis
                        </div>
                    <div style=" font-size: 0.85rem; color: #64748B; margin-top: 0.15rem; ">
                        t-distributed Stochastic Neighbor Embedding.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        with st.expander("t-SNE controls", expanded=False):

            # -------------------------------------------------------
            # t-SNE analysis controls
            # -------------------------------------------------------

            st.caption("Analysis controls")

            tsne_analysis_col1, tsne_analysis_col2, tsne_analysis_col3 = st.columns(3)

            with tsne_analysis_col1:

                tsne_gmm_components_widget = st.slider(
                    "GMM components",
                    min_value=GMM_COMPONENT_MIN,
                    max_value=GMM_COMPONENT_MAX,
                    value=DEFAULT_GMM_COMPONENTS,
                    step=1,
                    key=tsne_gmm_key,
                    help=(
                        "A SEPARATE Gaussian Mixture Model fitted directly "
                        "on the 2D t-SNE coordinates -- independent of the "
                        "PCA GMM."
                    ),
                )

            with tsne_analysis_col2:

                tsne_perplexity_widget = st.slider(
                    "Perplexity",
                    min_value=5,
                    max_value=50,
                    value=30,
                    step=5,
                    key=tsne_perplexity_key,
                    help=(
                        "Roughly, the number of effective nearest "
                        "neighbors t-SNE considers around each point."
                    ),
                )

            with tsne_analysis_col3:

                tsne_seed_widget = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=9999,
                    value=DEFAULT_RANDOM_SEED,
                    step=1,
                    key=tsne_seed_key,
                )

            # -------------------------------------------------------
            # t-SNE visualization controls
            # -------------------------------------------------------

            st.caption("Visualization controls")

            tsne_viz_col1, tsne_viz_col2 = st.columns(2)

            with tsne_viz_col1:

                tsne_color_options = ["Cluster"] + available_properties

                tsne_color_by = st.selectbox( "Color by", options=tsne_color_options, format_func=color_label, key="tsne_color_by_" + dataset_id, )

                tsne_show_ellipses = st.checkbox( "Show Gaussian ellipses", value=False, key="tsne_show_ellipses", )

                tsne_show_centers = st.checkbox( "Show cluster centers", value=False, key="tsne_show_centers", )

            with tsne_viz_col2:

                tsne_ellipse_confidence_pct = st.selectbox(
                    "Ellipse confidence",
                    options=ELLIPSE_CONFIDENCE_OPTIONS,
                    format_func=lambda x: f"{x}%",
                    index=ELLIPSE_CONFIDENCE_OPTIONS.index( DEFAULT_ELLIPSE_CONFIDENCE_PCT ),
                    key="tsne_ellipse_confidence",
                )

                tsne_ellipse_opacity = st.slider(
                    "Ellipse fill opacity",
                    min_value=0.0,
                    max_value=0.6,
                    value=0.16,
                    step=0.02,
                    key="tsne_ellipse_opacity",
                )

                tsne_point_size = st.slider(
                    "Point size",
                    min_value=3,
                    max_value=12,
                    value=DEFAULT_POINT_SIZE,
                    step=1,
                    key="tsne_point_size",
                )

        # -------------------------------------------------------
        # t-SNE plot
        # -------------------------------------------------------

        tsne_cluster_order = [ str(c) for c in range(tsne_gmm_components) ]

        tsne_plot_df = analysis_data.copy()
        tsne_plot_df["TSNE1"] = X_tsne[:, 0]
        tsne_plot_df["TSNE2"] = X_tsne[:, 1]

        build_projection_figure(
            plot_df=tsne_plot_df,
            x_col="TSNE1",
            y_col="TSNE2",
            x_title="t-SNE 1",
            y_title="t-SNE 2",
            gmm_model=tsne_gmm_model,
            gmm_labels=tsne_gmm_labels,
            gmm_max_probability=tsne_gmm_max_probability,
            color_by=tsne_color_by,
            color_options=tsne_color_options,
            show_ellipses=tsne_show_ellipses,
            ellipse_confidence=tsne_ellipse_confidence_pct / 100,
            ellipse_opacity=tsne_ellipse_opacity,
            show_centers=tsne_show_centers,
            point_size=tsne_point_size,
            cluster_order=tsne_cluster_order,
            figure_key="tsne_main_plot",
        )

        # -------------------------------------------------------
        # t-SNE cluster summary
        # -------------------------------------------------------

        st.caption("t-SNE GMM clusters")

        tsne_cluster_summary = (
            pd.Series(tsne_gmm_labels, name="Cluster")
            .value_counts()
            .sort_index()
            .rename("Molecules")
            .reset_index()
        )

        tsne_cluster_summary["Fraction"] = ( tsne_cluster_summary["Molecules"] / len(tsne_gmm_labels) ).map(lambda x: f"{x:.1%}")

        tsne_cluster_summary["Cluster"] = ( "t-SNE " + tsne_cluster_summary["Cluster"].astype(str) )

        st.dataframe( tsne_cluster_summary, width='stretch', hide_index=True, )

        # -------------------------------------------------------
        # t-SNE diagnostics
        # -------------------------------------------------------

        with st.expander("▼ t-SNE diagnostics"):

            st.markdown("**Selected properties**")

            st.write( ", ".join( property_name(p) for p in selected_properties ) )

            st.markdown("**t-SNE parameters**")

            tsne_params_df = pd.DataFrame(
                {
                    "Parameter": [ "Perplexity", "Learning rate", "Iterations (max)", "Random seed", f"{scope_unit_label.capitalize()} analyzed", "Runtime", ],
                    "Value": [ f"{tsne_perplexity}", "auto", f"{TSNE_MAX_ITER}", f"{tsne_seed}", f"{len(analysis_data):,}", f"{tsne_runtime_seconds:.1f}s", ],
                }
            )

            st.dataframe( tsne_params_df, width='stretch', hide_index=True, )

            st.caption(
                "Preprocessing: standardized (zero mean, unit "
                "variance) via StandardScaler, same as PCA, before "
                "t-SNE. No PCA-specific diagnostics apply here."
            )

        # -------------------------------------------------------
        # t-SNE GMM diagnostics
        # -------------------------------------------------------

        with st.expander("▼ GMM diagnostics (t-SNE space)"):

            tsne_gmm_diag_col1, tsne_gmm_diag_col2 = ( st.columns(2) )

            with tsne_gmm_diag_col1:
                st.metric("Components", tsne_gmm_components)

                st.metric("AIC", f"{tsne_gmm_aic:,.0f}")

            with tsne_gmm_diag_col2:
                st.metric("BIC", f"{tsne_gmm_bic:,.0f}")

                st.metric( "Silhouette", ( "—" if np.isnan(tsne_gmm_silhouette) else f"{tsne_gmm_silhouette:.3f}" ), )

            st.caption(
                "Converged: "
                f"{'yes' if tsne_gmm_model.converged_ else 'no'} "
                f"in {tsne_gmm_model.n_iter_} iterations."
            )

            st.markdown("**Cluster weights**")

            tsne_weights_df = pd.DataFrame( { "Cluster": [ f"t-SNE {i}" for i in range(tsne_gmm_components) ], "Weight": tsne_gmm_model.weights_, } )

            st.dataframe( tsne_weights_df, width='stretch', hide_index=True, )

            st.markdown("**Model selection (2-10 components)**")

            tsne_component_range = tuple( range(GMM_COMPONENT_MIN, GMM_COMPONENT_MAX + 1) )

            tsne_selection_df = calculate_gmm_model_selection( X_tsne, tsne_component_range, GMM_COVARIANCE_TYPE, GMM_N_INIT, DEFAULT_RANDOM_SEED, )

            st.dataframe( tsne_selection_df.style.apply( _highlight_selected, selected=tsne_gmm_components, axis=1, ), width='stretch', hide_index=True, )

            st.caption( "The highlighted row is the currently selected \"Number of GMM components\" value above." )

    # ============================================================
    # PCA VS t-SNE COMPARISON
    # ============================================================

    section_header("PCA vs t-SNE comparison")

    comparison_table = pd.DataFrame(
        {
            "Metric": [ f"{scope_unit_label.capitalize()} analyzed", "Selected properties", "Projection dimensions", "GMM components", "Silhouette", "AIC", "BIC", ],
            "PCA": [ f"{len(analysis_data):,}", f"{len(selected_properties)}", "2", f"{pca_gmm_components}", ( "—" if np.isnan(pca_gmm_silhouette) else f"{pca_gmm_silhouette:.3f}" ), f"{pca_gmm_aic:,.0f}", f"{pca_gmm_bic:,.0f}", ],
            "t-SNE": [ f"{len(analysis_data):,}", f"{len(selected_properties)}", "2", f"{tsne_gmm_components}", ( "—" if np.isnan(tsne_gmm_silhouette) else f"{tsne_gmm_silhouette:.3f}" ), f"{tsne_gmm_aic:,.0f}", f"{tsne_gmm_bic:,.0f}", ],
        }
    )

    st.dataframe( comparison_table, width='stretch', hide_index=True, )

    st.info(
        "The PCA and t-SNE GMMs are fitted **independently**, each "
        "directly on its own 2D projection -- \"PCA Cluster 0\" and "
        "\"t-SNE Cluster 0\" are not the same grouping and should not "
        "be compared as if they were the same cluster. Differences in "
        "silhouette/AIC/BIC above reflect how the two projections "
        "themselves separate the data, not a shared clustering."
    )
