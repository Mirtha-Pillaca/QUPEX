
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from modules.datasets import (PROPERTY_INFO, get_available_properties, get_property_label, get_property_info, get_clustering_defaults, )
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

st.write("Explores chemical-space structure through PCA and t-SNE dimensionality reduction. A Gaussian Mixture Model (GMM) or K-Means is fitted independently to each resulting 2D projection to identify molecular clusters.")

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

CLUSTERING_ALGORITHMS = ["GMM", "K-Means"]
DEFAULT_CLUSTERING_ALGORITHM = "GMM"

KMEANS_N_INIT = 10


dataset_identity = st.session_state.get("dataset_identity")

# QM9 uses its own property definitions, clustering-eligible properties
# and defaults from the dataset registry; every other dataset keeps the
# QM7-X behavior.
if dataset_identity == "QM9":

    PROPERTY_INFO = get_property_info("QM9")

    DEFAULT_PROPERTIES = get_clustering_defaults("QM9")

    available_properties = get_available_properties(df, dataset="QM9", clustering_only=True)

    # Per-atom list columns are not used here and make Streamlit's cache
    # fall back to pickling the whole table on every rerun.
    df = df.drop(columns=["elements", "coordinates"], errors="ignore")

else:

    available_properties = get_available_properties(df)

if len(available_properties) < 2:

    st.error( "At least two numeric properties are required for chemical-space analysis." )

    st.stop()


properties_key = f"chemical_space_properties_{dataset_id}"

clustering_algorithm_key = "clustering_algorithm"
n_clusters_key = "clustering_n_clusters"
tsne_perplexity_key = "tsne_perplexity"
tsne_seed_key = "tsne_seed"

default_properties = [ prop for prop in DEFAULT_PROPERTIES if prop in available_properties ]

if len(default_properties) < 2:
    default_properties = available_properties[:5]

selected_properties = st.session_state.get( properties_key, default_properties, )

selected_properties = [ prop for prop in selected_properties if prop in available_properties ] or default_properties

clustering_algorithm = st.session_state.get( clustering_algorithm_key, DEFAULT_CLUSTERING_ALGORITHM, )

if clustering_algorithm not in CLUSTERING_ALGORITHMS:
    clustering_algorithm = DEFAULT_CLUSTERING_ALGORITHM

is_gmm = clustering_algorithm == "GMM"

# One shared cluster count; each projection still gets its own, independent fit.
n_clusters = st.session_state.get( n_clusters_key, DEFAULT_GMM_COMPONENTS, )

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

# ------------------------------------------------------------
# K-Means -- also fit directly on each projection's OWN 2D coordinates.
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def calculate_kmeans_2d( X_2d, n_clusters, n_init, random_state, ):
    """
    Fit K-Means directly on 2D projection coordinates (PCA or t-SNE),
    mirroring calculate_gmm_2d.
    """

    model = KMeans( n_clusters=n_clusters, n_init=n_init, random_state=random_state, )

    labels = model.fit_predict(X_2d)

    distance_to_center = np.linalg.norm( X_2d - model.cluster_centers_[labels], axis=1, )

    return ( model, labels, distance_to_center, )


@st.cache_data(show_spinner=False)
def calculate_cluster_scores(X_2d, labels):
    """Calinski-Harabasz and Davies-Bouldin indexes (both linear in n, unlike silhouette)."""

    if len(np.unique(labels)) < 2:
        return np.nan, np.nan

    return ( calinski_harabasz_score(X_2d, labels), davies_bouldin_score(X_2d, labels), )


@st.cache_data(show_spinner=False)
def calculate_kmeans_model_selection( X_2d, cluster_range, n_init, random_state, ):
    """Inertia/silhouette/Calinski-Harabasz/Davies-Bouldin for several cluster counts, on 2D data."""

    records = []

    for n_clusters in cluster_range:

        try:

            model = KMeans( n_clusters=n_clusters, n_init=n_init, random_state=random_state, )

            labels = model.fit_predict(X_2d)

            calinski_harabasz, davies_bouldin = calculate_cluster_scores(X_2d, labels)

            records.append( { "Clusters": n_clusters, "Inertia": model.inertia_, "Silhouette": calculate_silhouette(X_2d, labels), "Calinski-Harabasz": calinski_harabasz, "Davies-Bouldin": davies_bouldin, } )

        except Exception:

            records.append( { "Clusters": n_clusters, "Inertia": np.nan, "Silhouette": np.nan, "Calinski-Harabasz": np.nan, "Davies-Bouldin": np.nan, } )

    return pd.DataFrame(records)


def fit_projection_clustering(X_2d, projection_label):
    """
    Fit the selected algorithm on one projection's own 2D coordinates.
    PCA and t-SNE each call this separately -- they never share a model.
    """

    try:

        if is_gmm:

            ( model, labels, _, max_probability, ) = calculate_gmm_2d( X_2d, n_clusters, GMM_COVARIANCE_TYPE, GMM_N_INIT, DEFAULT_RANDOM_SEED, )

            distance_to_center = None

        else:

            ( model, labels, distance_to_center, ) = calculate_kmeans_2d( X_2d, n_clusters, KMEANS_N_INIT, DEFAULT_RANDOM_SEED, )

            max_probability = None

    except Exception as exc:

        st.error(
            f"Could not fit the {projection_label} {clustering_algorithm} with "
            f"{n_clusters} clusters: {exc}. Try a lower \"Number of "
            "clusters\" value."
        )

        st.stop()

    result = {
        "model": model,
        "labels": labels,
        "max_probability": max_probability,
        "distance_to_center": distance_to_center,
        "silhouette": calculate_silhouette( X_2d, labels, ),
    }

    if is_gmm:

        result["aic"] = model.aic(X_2d)
        result["bic"] = model.bic(X_2d)

    else:

        result["inertia"] = model.inertia_
        result["calinski_harabasz"], result["davies_bouldin"] = calculate_cluster_scores( X_2d, labels, )

    return result


pca_clustering = fit_projection_clustering( X_pca, "PCA", )

tsne_clustering = fit_projection_clustering( X_tsne, "t-SNE", )


# ============================================================
# VISUALIZATION HELPERS 
# ============================================================

def property_name(prop):
    """Readable property name."""

    return PROPERTY_INFO.get( prop, {}, ).get( "name", prop, )


def color_label(value):
    """Readable label for a 'Color by' selection."""

    if value == "Cluster":
        return f"{clustering_algorithm} Cluster"

    return property_name(value)


def format_metric(value, spec=".3f"):
    """Format a score, showing an em dash when it is undefined."""

    return "—" if value is None or np.isnan(value) else f"{value:{spec}}"


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


def build_projection_figure( plot_df, x_col, y_col, x_title, y_title, clustering, color_by, color_options, show_ellipses, ellipse_confidence, ellipse_opacity, show_centers, point_size, cluster_order, figure_key, ):
    """
    Build one projection's scatter plot (PCA or t-SNE) -- identical
    logic for both, parameterized by which 2D columns and clustering
    result (GMM or K-Means) to use. Visualization-only arguments
    (color_by, show_ellipses, ellipse_confidence, ellipse_opacity,
    show_centers, point_size) only ever affect this uncached function
    -- never the cached analysis pipeline above.
    """

    model = clustering["model"]

    plot_df = plot_df.copy()

    plot_df["Cluster"] = clustering["labels"].astype(str)

    hover_data = {
        "molecule_id": True,
        x_col: ":.4f",
        y_col: ":.4f",
    }

    # Always show the per-point assignment quality in the hover: GMM
    # membership probability, or K-Means distance to the cluster center.
    if clustering["max_probability"] is not None:

        plot_df["Membership probability"] = clustering["max_probability"]
        hover_data["Membership probability"] = ":.1%"

    if clustering["distance_to_center"] is not None:

        plot_df["Distance to center"] = clustering["distance_to_center"]
        hover_data["Distance to center"] = ":.3f"

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

        fig.update_layout(legend_title_text=f"{clustering_algorithm} Cluster")

    else:

        hover_data = {**hover_data, "Cluster": True}

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_by,
            color_continuous_scale=SEQUENTIAL_COLORSCALE_STOPS,
            labels={color_by: get_property_label(color_by, dataset=dataset_identity)},
            hover_data=hover_data,
        )

    fig.update_traces( marker=dict( size=point_size, opacity=0.72, ), selector=dict(mode="markers"), )

    centers = model.means_ if is_gmm else model.cluster_centers_

    center_word = "component" if is_gmm else "cluster"

    for component in range(len(centers)):

        # Same deterministic sequence used for the point colors
        # above, so an ellipse/center always matches its own cluster's
        # point color -- including past 8 components, where the
        # plain QUALITATIVE_COLORS list alone would repeat.
        component_color = get_qualitative_color(component)

        # Gaussian ellipses only exist for GMM.
        if show_ellipses and is_gmm:

            ellipse_x, ellipse_y = gmm_ellipse_xy(
                model.means_[component],
                model.covariances_[component],
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
                    legendgroup=f"cluster_{component}",
                    # Avoid a redundant legend entry when points are
                    # already colored by the same GMM cluster.
                    showlegend=(color_by != "Cluster"),
                )
            )

        if show_centers:

            fig.add_trace(
                go.Scatter(
                    x=[centers[component][0]],
                    y=[centers[component][1]],
                    mode="markers",
                    marker=dict(
                        size=11,
                        symbol="x",
                        color=component_color,
                        line=dict(width=2),
                    ),
                    name=f"{clustering_algorithm} {component} center",
                    hovertemplate=(
                        f"{clustering_algorithm} {center_word} {component}"
                        f"<br>{x_title}=%{{x:.3f}}"
                        f"<br>{y_title}=%{{y:.3f}}"
                        "<extra></extra>"
                    ),
                    legendgroup=f"cluster_{component}",
                    showlegend=False,
                )
            )

    fig.update_layout( height=560, xaxis_title=x_title, yaxis_title=y_title, )

    fig = scientific_layout(fig)

    fig.update_layout( legend=dict( bgcolor="rgba(0,0,0,0)", borderwidth=0, ), )

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")

    if color_by != "Cluster":

        # Vertical colorbar on the right, its title rotated to run along
        # the bar, so long property names with units are not clipped.
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=dict( text=get_property_label(color_by, dataset=dataset_identity), side="right", ),
            ),
        )

    fig = use_webgl_for_large_scatter_total(fig)

    st.plotly_chart( fig, width='stretch', key=figure_key, )


def render_cluster_summary(clustering, projection_label):
    """Cluster sizes table for one projection."""

    labels = clustering["labels"]

    st.caption(f"{projection_label} {clustering_algorithm} clusters")

    cluster_summary = ( pd.Series(labels, name="Cluster") .value_counts() .sort_index() .rename("Molecules") .reset_index() )

    cluster_summary["Fraction"] = ( cluster_summary["Molecules"] / len(labels) ).map(lambda x: f"{x:.1%}")

    cluster_summary["Cluster"] = ( f"{projection_label} " + cluster_summary["Cluster"].astype(str) )

    st.dataframe( cluster_summary, width='stretch', hide_index=True, )


def reset_color_by_to_cluster():
    """On an algorithm switch, color both projections by the new clusters again."""

    for prefix in ("pca", "tsne"):
        st.session_state[f"{prefix}_color_by_{dataset_id}"] = "Cluster"


def _highlight_selected(row, selected, column="Components"):

    if row[column] == selected:

        return [ f"background-color: {COLORS.get('accent_bg', '#FFF3CD')}" ] * len(row)

    return [""] * len(row)


def render_clustering_diagnostics(X_2d, clustering, projection_label, axis_names):
    """
    Algorithm-specific diagnostics for one projection: AIC/BIC and
    weights for GMM; inertia, Calinski-Harabasz, Davies-Bouldin and
    centers for K-Means. Both include 2-10 cluster model selection.
    """

    model = clustering["model"]

    cluster_range = tuple( range(GMM_COMPONENT_MIN, GMM_COMPONENT_MAX + 1) )

    if is_gmm:

        with st.expander(f"▼ GMM diagnostics ({projection_label} space)"):

            gmm_diag_col1, gmm_diag_col2 = ( st.columns(2) )

            with gmm_diag_col1:
                st.metric("Components", n_clusters)

                st.metric("AIC", f"{clustering['aic']:,.0f}")

            with gmm_diag_col2:
                st.metric("BIC", f"{clustering['bic']:,.0f}")

                st.metric( "Silhouette", format_metric(clustering["silhouette"]), )

            st.caption(
                "Converged: "
                f"{'yes' if model.converged_ else 'no'} "
                f"in {model.n_iter_} iterations."
            )

            st.markdown("**Cluster weights**")

            weights_df = pd.DataFrame( { "Cluster": [ f"{projection_label} {i}" for i in range(n_clusters) ], "Weight": model.weights_, } )

            st.dataframe( weights_df, width='stretch', hide_index=True, )

            st.markdown("**Model selection (2-10 components)**")

            selection_df = calculate_gmm_model_selection( X_2d, cluster_range, GMM_COVARIANCE_TYPE, GMM_N_INIT, DEFAULT_RANDOM_SEED, )

            st.dataframe( selection_df.style.apply( _highlight_selected, selected=n_clusters, axis=1, ), width='stretch', hide_index=True, )

            st.caption( "The highlighted row is the currently selected \"Number of clusters\" value above." )

    else:

        with st.expander(f"▼ K-Means diagnostics ({projection_label} space)"):

            kmeans_diag_col1, kmeans_diag_col2 = ( st.columns(2) )

            with kmeans_diag_col1:
                st.metric("Clusters", n_clusters)

                st.metric( "Inertia (WCSS)", f"{clustering['inertia']:,.1f}", help="Within-cluster sum of squared distances to the cluster centers (lower is tighter).", )

                st.metric( "Calinski-Harabasz", format_metric(clustering["calinski_harabasz"], ",.1f"), help="Ratio of between- to within-cluster dispersion (higher is better).", )

            with kmeans_diag_col2:
                st.metric( "Silhouette", format_metric(clustering["silhouette"]), )

                st.metric( "Davies-Bouldin", format_metric(clustering["davies_bouldin"]), help="Average similarity of each cluster to its most similar cluster (lower is better).", )

            st.caption( f"Best of {KMEANS_N_INIT} k-means++ initializations; " f"{model.n_iter_} iterations." )

            st.markdown("**Cluster centers**")

            centers_df = pd.DataFrame(
                {
                    "Cluster": [ f"{projection_label} {i}" for i in range(n_clusters) ],
                    f"{axis_names[0]} center": model.cluster_centers_[:, 0],
                    f"{axis_names[1]} center": model.cluster_centers_[:, 1],
                    "Size": np.bincount( clustering["labels"], minlength=n_clusters, ),
                }
            )

            st.dataframe( centers_df, width='stretch', hide_index=True, )

            st.markdown("**Model selection (2-10 clusters)**")

            selection_df = calculate_kmeans_model_selection( X_2d, cluster_range, KMEANS_N_INIT, DEFAULT_RANDOM_SEED, )

            st.dataframe( selection_df.style.apply( _highlight_selected, selected=n_clusters, column="Clusters", axis=1, ), width='stretch', hide_index=True, )

            st.caption(
                "The highlighted row is the currently selected \"Number of "
                "clusters\" value above. Look for an elbow in inertia; higher "
                "silhouette/Calinski-Harabasz and lower Davies-Bouldin are better."
            )


# ============================================================
#TABS
# ============================================================

tab_chemical_analysis = st.tabs( ["🧩 PCA vs t-SNE (GMM & K-Means)"] )[0]



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

        st.metric( f"PCA {clustering_algorithm} silhouette", format_metric(pca_clustering["silhouette"]), help=f"From the {clustering_algorithm} fitted directly on the PCA coordinates.", )

    with metric_col5:

        st.metric( f"t-SNE {clustering_algorithm} silhouette", format_metric(tsne_clustering["silhouette"]), help=f"From a SEPARATE {clustering_algorithm} fitted directly on the t-SNE coordinates.", )

    # ============================================================
    # SHARED ANALYSIS CONTROLS
    # ============================================================

    with st.container(border=True):

        st.caption("Shared settings -- applied to both the PCA and the t-SNE projection")

        shared_col1, shared_col2 = st.columns(2, gap="large")

        with shared_col1:

            st.radio(
                "Clustering algorithm",
                options=CLUSTERING_ALGORITHMS,
                index=CLUSTERING_ALGORITHMS.index(DEFAULT_CLUSTERING_ALGORITHM),
                horizontal=True,
                key=clustering_algorithm_key,
                on_change=reset_color_by_to_cluster,
                help=(
                    "Fitted separately on the 2D PCA coordinates and on the "
                    "2D t-SNE coordinates -- never on the raw properties. "
                    "GMM: soft, elliptical clusters with membership "
                    "probabilities and AIC/BIC. K-Means: hard assignments "
                    "to the nearest of k centers."
                ),
            )

        with shared_col2:

            st.slider(
                "Number of clusters",
                min_value=GMM_COMPONENT_MIN,
                max_value=GMM_COMPONENT_MAX,
                value=DEFAULT_GMM_COMPONENTS,
                step=1,
                key=n_clusters_key,
                help=(
                    "Used for both projections; each projection still gets "
                    "its own, independent fit."
                ),
            )

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
                "plotting; t-SNE embeds them directly in 2D; the selected "
                "clustering algorithm is fitted independently on each 2D "
                "projection, not on these raw properties."
            ),
            key=properties_key,
        )

        st.caption(
            f"**{len(selected_properties)}** properties/features selected "
            f"for this analysis."
        )

    ellipse_help = None if is_gmm else "Gaussian ellipses are only available for GMM."

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
        # PCA visualization controls
        # -------------------------------------------------------

        with st.expander("PCA controls", expanded=False):

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

                pca_show_ellipses = st.checkbox( "Show Gaussian ellipses", value=False, key="pca_show_ellipses", disabled=not is_gmm, help=ellipse_help, )

                pca_show_centers = st.checkbox( "Show cluster centers", value=False, key="pca_show_centers", )

            with viz_col2:

                pca_ellipse_confidence_pct = st.selectbox(
                    "Ellipse confidence",
                    options=ELLIPSE_CONFIDENCE_OPTIONS,
                    format_func=lambda x: f"{x}%",
                    index=ELLIPSE_CONFIDENCE_OPTIONS.index( DEFAULT_ELLIPSE_CONFIDENCE_PCT ),
                    key="pca_ellipse_confidence",
                    disabled=not is_gmm,
                )

                pca_ellipse_opacity = st.slider( "Ellipse fill opacity", min_value=0.0, max_value=0.6, value=0.16, step=0.02, key="pca_ellipse_opacity", disabled=not is_gmm, )

                pca_point_size = st.slider( "Point size", min_value=3, max_value=12, value=DEFAULT_POINT_SIZE, step=1, key="pca_point_size", )

        # -------------------------------------------------------
        # PCA plot
        # -------------------------------------------------------

        pca_cluster_order = [ str(c) for c in range(n_clusters) ]

        pca_plot_df = analysis_data.copy()
        pca_plot_df["PC1"] = X_pca[:, 0]
        pca_plot_df["PC2"] = X_pca[:, 1]

        build_projection_figure(
            plot_df=pca_plot_df,
            x_col="PC1",
            y_col="PC2",
            x_title=f"PC1 ({pca_explained_variance[0] * 100:.1f}%)",
            y_title=f"PC2 ({pca_explained_variance[1] * 100:.1f}%)",
            clustering=pca_clustering,
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

        render_cluster_summary( pca_clustering, "PCA", )

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
        # PCA clustering diagnostics (GMM or K-Means)
        # -------------------------------------------------------

        render_clustering_diagnostics( X_pca, pca_clustering, "PCA", ("PC1", "PC2"), )

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

            tsne_analysis_col1, tsne_analysis_col2 = st.columns(2)

            with tsne_analysis_col1:

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

            with tsne_analysis_col2:

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

                tsne_show_ellipses = st.checkbox( "Show Gaussian ellipses", value=False, key="tsne_show_ellipses", disabled=not is_gmm, help=ellipse_help, )

                tsne_show_centers = st.checkbox( "Show cluster centers", value=False, key="tsne_show_centers", )

            with tsne_viz_col2:

                tsne_ellipse_confidence_pct = st.selectbox(
                    "Ellipse confidence",
                    options=ELLIPSE_CONFIDENCE_OPTIONS,
                    format_func=lambda x: f"{x}%",
                    index=ELLIPSE_CONFIDENCE_OPTIONS.index( DEFAULT_ELLIPSE_CONFIDENCE_PCT ),
                    key="tsne_ellipse_confidence",
                    disabled=not is_gmm,
                )

                tsne_ellipse_opacity = st.slider(
                    "Ellipse fill opacity",
                    min_value=0.0,
                    max_value=0.6,
                    value=0.16,
                    step=0.02,
                    key="tsne_ellipse_opacity",
                    disabled=not is_gmm,
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

        tsne_cluster_order = [ str(c) for c in range(n_clusters) ]

        tsne_plot_df = analysis_data.copy()
        tsne_plot_df["TSNE1"] = X_tsne[:, 0]
        tsne_plot_df["TSNE2"] = X_tsne[:, 1]

        build_projection_figure(
            plot_df=tsne_plot_df,
            x_col="TSNE1",
            y_col="TSNE2",
            x_title="t-SNE 1",
            y_title="t-SNE 2",
            clustering=tsne_clustering,
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

        render_cluster_summary( tsne_clustering, "t-SNE", )

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
        # t-SNE clustering diagnostics (GMM or K-Means)
        # -------------------------------------------------------

        render_clustering_diagnostics( X_tsne, tsne_clustering, "t-SNE", ("t-SNE 1", "t-SNE 2"), )

    # ============================================================
    # PCA VS t-SNE COMPARISON
    # ============================================================

    section_header("PCA vs t-SNE comparison")

    def _comparison_values(clustering):

        values = [ f"{len(analysis_data):,}", f"{len(selected_properties)}", "2", f"{n_clusters}", format_metric(clustering["silhouette"]), ]

        if is_gmm:
            return values + [ f"{clustering['aic']:,.0f}", f"{clustering['bic']:,.0f}", ]

        return values + [ f"{clustering['inertia']:,.1f}", format_metric(clustering["calinski_harabasz"], ",.1f"), format_metric(clustering["davies_bouldin"]), ]

    comparison_metrics = [ f"{scope_unit_label.capitalize()} analyzed", "Selected properties", "Projection dimensions", "GMM components" if is_gmm else "K-Means clusters", "Silhouette", ]

    comparison_metrics += ( [ "AIC", "BIC", ] if is_gmm else [ "Inertia (WCSS)", "Calinski-Harabasz", "Davies-Bouldin", ] )

    comparison_table = pd.DataFrame(
        {
            "Metric": comparison_metrics,
            "PCA": _comparison_values(pca_clustering),
            "t-SNE": _comparison_values(tsne_clustering),
        }
    )

    st.dataframe( comparison_table, width='stretch', hide_index=True, )

    if is_gmm:

        st.info(
            "The PCA and t-SNE GMMs are fitted **independently**, each "
            "directly on its own 2D projection -- \"PCA Cluster 0\" and "
            "\"t-SNE Cluster 0\" are not the same grouping and should not "
            "be compared as if they were the same cluster. Differences in "
            "silhouette/AIC/BIC above reflect how the two projections "
            "themselves separate the data, not a shared clustering."
        )

    else:

        st.info(
            "The PCA and t-SNE K-Means models are fitted **independently**, "
            "each directly on its own 2D projection -- \"PCA Cluster 0\" and "
            "\"t-SNE Cluster 0\" are not the same grouping and should not "
            "be compared as if they were the same cluster. Inertia and "
            "Calinski-Harabasz depend on each projection's coordinate scale, "
            "so compare them across cluster counts within one projection "
            "rather than between PCA and t-SNE."
        )
