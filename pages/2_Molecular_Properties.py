import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from modules.loader import (
    load_molecule,
    molecule_2d_image,
    molecule_to_xyz,
    molecule_to_position_dataframe,
    formula_from_atnums,
    ATOM_INFO, PROPERTY_INFO
)

from modules.plot_style import scientific_layout

from modules.theme import apply_global_theme, COLORS, section_header

from modules.molecular_visualization import display_3d_molecule, display_comparative_3d_viewers


st.set_page_config(page_title="Molecular Properties",page_icon="⚛️",layout="wide",)
apply_global_theme()

# ============================================================
# HEADER
# ============================================================

st.title("Property Correlation & Molecular Visualization")
   
st.write("Provides correlation analysis of quantum‑chemical properties alonside interactive 2D/3D molecular visualizations with atom‑specific information.")
# ============================================================
# CHECK DATASET
# ============================================================

if "hdf" not in st.session_state:

    st.warning("⚠️ **No dataset is currently loaded.** ")
    st.info("ℹ️ Please return to the Home page and load a dataset from a local file, an uploaded file, or Zenodo.")
    st.stop()


hdf = st.session_state["hdf"]

# ============================================================
# LOAD INDEX
# ============================================================

if not st.session_state.get( "analysis_indexes_ready", False ):

    st.warning( "⚠️ **Dataset analysis indexes are not ready.**" )
    st.info( "ℹ️ Please return to the Home page and load the dataset again." )
    st.stop()

df = st.session_state["opt_index"]

if df is None or df.empty:

    st.warning( "The dataset analysis index is empty." )
    st.stop()

# ============================================================
# AVAILABLE PROPERTIES
# ============================================================

available_properties = [ p for p in PROPERTY_INFO if p in df.columns ]

if not available_properties:

    st.error( "No recognized molecular properties were found in the analysis index." )
    st.stop()


# ============================================================
# 🔎 PROPERTY DISTRIBUTION ANALYSIS
# ============================================================

tab_distribution, tab_matrix_correlation, tab_pairwise = st.tabs([
    "📊 Property Distributions",
    "🔗 Correlation Matrix",
    "🧬 Pairplots & Molecular Structures",
])

with tab_distribution:

    col_scope_info, col_scope_control = st.columns( [3, 1], gap="medium", )

    with col_scope_info:

        st.caption(
            "Choose whether the property distribution below "
            "describes unique molecules or every geometry record "
            "(including conformers)."
        )

    with col_scope_control:

        distribution_analysis_scope = st.segmented_control(
            "Analysis scope",
            options=[ "Molecules", "Geometries", ],
            default="Molecules",
            key="property_distribution_scope",
            label_visibility="collapsed",
        )

    if distribution_analysis_scope is None:
        distribution_analysis_scope = "Molecules"


    if distribution_analysis_scope == "Geometries":

        # `CHECK REQUIRED DATA` 
        scope_df = st.session_state.get("geometry_index")

        scope_unit_label = "geometries"
        scope_unit_label_singular = "geometry"
        scope_id_column = "geometry_id"
        scope_id_header = "Geometry ID"

    else:

        scope_df = df

        scope_unit_label = "molecules"
        scope_unit_label_singular = "molecule"
        scope_id_column = "molecule_id"
        scope_id_header = "QM7-X Molecule ID"

    # ============================================================
    # CHECK REQUIRED DATA
    # ============================================================

    if scope_df is None or scope_df.empty:

        st.error( "The analysis DataFrame is not available for the " f"{distribution_analysis_scope} scope." )

    elif "PROPERTY_INFO" not in globals():

        st.error("PROPERTY_INFO is not defined.")

    else:

        available_properties = [ property_key for property_key in PROPERTY_INFO if property_key in scope_df.columns ]

        if not available_properties:

            st.warning( "No properties from PROPERTY_INFO were found " f"for {scope_unit_label}." )

        else:

            # ====================================================
            # PROPERTY SELECTION
            # ====================================================

            selected_property = st.selectbox(
                "**Property to explore:**",
                options=available_properties,
                format_func=lambda x: ( f"{PROPERTY_INFO[x].get('name', x)} ({x})" ),
                key=( "property_distribution_selection_" + distribution_analysis_scope ),
            )

            info = PROPERTY_INFO[selected_property]

            property_name = info.get( "name", selected_property, )
            property_description = info.get( "description", "No description available.", )

            property_unit = info.get( "unit", "", )

            property_category = info.get( "category", "", )

            # ====================================================
            # PROPERTY HEADER
            # ====================================================

            st.caption(
                f"**Description:** {property_description or '—'}  ·  "
                f"**Field:** `{selected_property}`  ·  "
                f"**Unit:** {property_unit or '—'}  ·  "
                f"**Category:** {property_category or '—'}"
            )

            # ====================================================
            # PREPARE DATA
            # ====================================================

            distribution_columns = [selected_property]

            if scope_id_column in scope_df.columns:

                distribution_columns.insert( 0, scope_id_column, )

            distribution_df = scope_df[ distribution_columns ].copy()

            distribution_df[selected_property] = pd.to_numeric( distribution_df[selected_property], errors="coerce", )

            distribution_df = ( distribution_df .dropna(subset=[selected_property]) .copy() )

            distribution_df = distribution_df[ np.isfinite( distribution_df[selected_property] ) ].copy()

            # ====================================================
            # CHECK DATA
            # ====================================================

            if distribution_df.empty:

                st.warning( f"No valid numerical values are available " f"for {property_name}." )

            else:

                values = distribution_df[ selected_property ]

                # ==================================================
                # DISTRIBUTION
                # ==================================================

                distribution_col1, distribution_col2 = st.columns( [4, 1.2], gap="large" )

                # ==================================================
                # CONTROLS
                # ==================================================

                with distribution_col2:

                    st.markdown("**Controls**")

                    n_bins = st.slider( "Histogram bins", min_value=20, max_value=100, value=50, step=5, key=("distribution_bins_" + selected_property), )

                    scale_mode = st.radio( "Y-axis scale", ["Linear", "Logarithmic"], horizontal=False, key=("distribution_y_scale_" + selected_property), )

                    st.caption( f"Distribution of {property_name} across " f"{len(values):,} {scope_unit_label}." )

                # ==================================================
                # HISTOGRAM
                # ==================================================

                with distribution_col1:

                    histogram_fig = px.histogram( distribution_df, x=selected_property, nbins=n_bins, )

                    histogram_fig.update_traces(
                        marker_line_width=0.7,
                        hovertemplate=(
                            f"<b>{property_name}</b>: "
                            "%{x:.5g}"
                            "<br>"
                            f"<b>{scope_unit_label.capitalize()}:</b> "
                            "%{y:,}"
                            "<extra></extra>"
                        ),
                    )

                    # Global scientific style
                    histogram_fig = scientific_layout( histogram_fig, height=450, show_legend=False, dark_mode=False, )

                    # Property-specific axis labels only
                    histogram_fig.update_xaxes( title_text=( f"{property_name}" + ( f" ({property_unit})" if property_unit else "" ) ), )

                    histogram_fig.update_yaxes( title_text=f"Number of {scope_unit_label}", type=( "log" if scale_mode == "Logarithmic" else "linear" ), )

                    st.plotly_chart( histogram_fig, width="stretch", key=f"distribution_histogram_{selected_property}", )

                # ==================================================
                # STATISTICAL SUMMARY
                # ==================================================

                with st.expander( "📊 Statistical summary" ):

                    summary_table = pd.DataFrame(
                        {
                            "Statistic": [
                                f"Number of {scope_unit_label}",
                                "Mean",
                                "Median",
                                "Standard deviation",
                                "Minimum",
                                "Maximum",
                                "Range",
                            ],
                            "Value": [
                                f"{len(values):,}",
                                f"{values.mean():.6g}",
                                f"{values.median():.6g}",
                                f"{values.std():.6g}",
                                f"{values.min():.6g}",
                                f"{values.max():.6g}",
                                f"{values.min():.6g} → "
                                f"{values.max():.6g}",
                            ],
                        }
                    )

                    st.dataframe( summary_table, width='stretch', hide_index=True, )

                # ==================================================
                # MOLECULE / GEOMETRY PROPERTY VALUES
                # ==================================================

                with st.expander( "📋 " + scope_unit_label_singular.capitalize() + " property values" ):

                    property_table = (
                        distribution_df[[scope_id_column, selected_property]]
                        .copy()
                        .rename(
                            columns={
                                scope_id_column:
                                    scope_id_header,
                                selected_property:
                                    (
                                        f"{property_name}"
                                        + (
                                            f" ({property_unit})"
                                            if property_unit
                                            else ""
                                        )
                                    ),
                            }
                        )
                    )

                    st.dataframe( property_table, width='stretch', hide_index=True, )

with tab_matrix_correlation:

    # ============================================================
    # CORRELATION MATRIX
    # ============================================================

    correlation_properties = st.multiselect(
        "Properties to include in correlation matrix:",
        options=available_properties,
        default=available_properties,
        format_func=lambda x: (
            f"{PROPERTY_INFO[x].get('name', x)} ({x})"
        ),
        key="distribution_correlation_properties",
    )

    correlation_method = st.selectbox(
        "Correlation method:",
        options=["pearson", "spearman"],
        format_func=lambda x: x.capitalize(),
        key="distribution_correlation_method",
    )

    if len(correlation_properties) < 2:

        st.info( "Select at least two properties to display the correlation matrix." )

    else:

        # --------------------------------------------------------
        # Prepare correlation data
        # --------------------------------------------------------

        correlation_df = df[ correlation_properties ].copy()

        for column in correlation_properties:
            correlation_df[column] = pd.to_numeric( correlation_df[column], errors="coerce", )

        correlation_df = correlation_df.replace( [np.inf, -np.inf], np.nan, )

        # --------------------------------------------------------
        # Check number of valid observations
        # --------------------------------------------------------

        valid_counts = correlation_df.notna().sum()

        insufficient_data = valid_counts[ valid_counts < 3 ]

        if not insufficient_data.empty:

            st.warning(
                "Some selected properties contain fewer than "
                "3 valid observations. Correlations involving "
                "these properties may be unreliable."
            )

        # --------------------------------------------------------
        # Correlation matrix
        # --------------------------------------------------------

        correlation_matrix = correlation_df.corr( method=correlation_method )

        # --------------------------------------------------------
        # Display names
        # --------------------------------------------------------

        property_labels = { property_key: PROPERTY_INFO[property_key].get( "name", property_key, ) for property_key in correlation_properties }

        correlation_matrix = correlation_matrix.rename( index=property_labels, columns=property_labels, )

        # --------------------------------------------------------
        # Top 5 strongest correlations
        # --------------------------------------------------------

        correlation_pairs = []

        for i in range(len(correlation_matrix.columns)):

            for j in range(i):

                r = correlation_matrix.iloc[i, j]

                if pd.notna(r):

                    correlation_pairs.append(
                        {
                            "Property 1": correlation_matrix.columns[i],
                            "Property 2": correlation_matrix.columns[j],
                            "Correlation": r,
                            "Absolute correlation": abs(r),
                        }
                    )

        top_5_correlations = ( pd.DataFrame(correlation_pairs) .sort_values( "Absolute correlation", ascending=False, ) .head(5) .drop(columns="Absolute correlation") .reset_index(drop=True) )

        top_5_correlations["Correlation"] = ( top_5_correlations["Correlation"].round(3) )

        # --------------------------------------------------------
        # Lower-triangle mask
        # --------------------------------------------------------

        mask = np.triu( np.ones( correlation_matrix.shape, dtype=bool, ) )

        masked_correlation = correlation_matrix.mask(mask)

        # --------------------------------------------------------
        # Correlation heatmap
        # --------------------------------------------------------

        correlation_fig = px.imshow(
            masked_correlation,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
        )

        correlation_fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b> × <b>%{x}</b>"
                "<br>"
                f"{correlation_method.capitalize()} r: "
                "%{z:.3f}"
                "<extra></extra>"
            )
        )

        correlation_fig.update_xaxes( side="bottom", tickangle=-45, )

        correlation_fig.update_yaxes( autorange="reversed", )

        correlation_fig.update_coloraxes( colorbar_title=( f"{correlation_method.capitalize()} r" ), cmin=-1, cmax=1, )

        correlation_fig = scientific_layout( correlation_fig, height=max( 300, 50 * len(correlation_properties), ), show_legend=False, dark_mode=False, )

        st.plotly_chart( correlation_fig, width='stretch', key="distribution_correlation_matrix", )

        # --------------------------------------------------------
        # Top 5 table
        # --------------------------------------------------------
        section_header("Strongest property correlations")  

        st.dataframe( top_5_correlations, width='stretch', hide_index=True, )

        # --------------------------------------------------------
        # Interpretation
        # --------------------------------------------------------

        st.caption(
            f"{correlation_method.capitalize()} correlation coefficient "
            "ranges from −1 (perfect negative correlation) "
            "to +1 (perfect positive correlation). "
            "Values near 0 indicate little association."
        )

        # --------------------------------------------------------
        # Download correlation matrix
        # --------------------------------------------------------

        correlation_csv = ( correlation_matrix .to_csv() .encode("utf-8") )

        st.download_button(
            label="Download correlation matrix",
            data=correlation_csv,
            file_name="correlation_matrix.csv",
            mime="text/csv",
            key="download_correlation_matrix",
        )

with tab_pairwise:

    # st.subheader("Pairwise correlation & Molecular Structure")
    st.caption("Explore relationships and correlations between molecular properties and their structural characteristics.")

    # ============================================================
    # PROPERTY DEFINITIONS
    # ============================================================

    relationship_properties = {

        "eMBD": "MBD energy (eV)",
        "ePBE0": "PBE0 energy (eV)",
        "ePBE0+MBD": "PBE0 + MBD energy (eV)",
        "eAT": "Atomization energy (eV)",
        "eTS": "TS dispersion energy (eV)",
        "eNN": "Nuclear–nuclear energy (eV)",
        "eKIN": "Kinetic energy (eV)",
        "eNE": "Nuclear–electron energy (eV)",
        "eEE": "Electron–electron energy (eV)",
        "eXC": "Exchange–correlation energy (eV)",
        "eX": "Exchange energy (eV)",
        "eC": "Correlation energy (eV)",
        "eXX": "Exact exchange energy (eV)",
        "eH": "HOMO energy (eV)",
        "eL": "LUMO energy (eV)",
        "HLgap": "HOMO–LUMO gap (eV)",
        "DIP": "Dipole moment (e·Å)",
        "mC6": "Molecular C₆ coefficient",
        "mPOL": "Molecular polarizability",
        "natoms": "Number of atoms",
    }

    relationship_options = [ prop for prop in relationship_properties if prop in df.columns ]

    # ============================================================
    # VALIDATION
    # ============================================================

    if len(relationship_options) < 2:

        st.warning( "At least two compatible properties are required." )

    else:

        # ========================================================
        # PROPERTY SELECTION
        # ========================================================

        col_x, col_y, col_z = st.columns(3)

        with col_x:

            default_x = ( "HLgap" if "HLgap" in relationship_options else relationship_options[0] )

            x_property = st.selectbox(
                "X-axis property",
                relationship_options,
                index=relationship_options.index(default_x),
                format_func=lambda x:
                    relationship_properties[x],
                key="relationship_x_property",
            )

        with col_y:

            possible_y = [ p for p in relationship_options if p != x_property ]

            default_y = ( "mPOL" if "mPOL" in possible_y else possible_y[0] )
            y_property = st.selectbox(
                "Y-axis property",
                possible_y,
                index=possible_y.index(default_y),
                format_func=lambda x:
                    relationship_properties[x],
                key="relationship_y_property",
            )

        with col_z:
            # ========================================================
            # COLOR PROPERTY
            # ========================================================

            color_options = [ p for p in relationship_options if p not in [x_property, y_property] ]

            color_col = st.selectbox(
                "Color scale",
                options=["None"] + color_options,
                format_func=lambda x:
                    "No color scale"
                    if x == "None"
                    else relationship_properties[x],
                key="relationship_color_property",
            )

        # ========================================================
        # DATA
        # ========================================================

        required_columns = [ "molecule_id", x_property, y_property, ]

        if color_col != "None":

            required_columns.append( color_col )

        relationship_df = df[ required_columns ].copy()
        for column in required_columns[1:]:

            relationship_df[column] = pd.to_numeric( relationship_df[column], errors="coerce", )

        plot_df = relationship_df.dropna( subset=required_columns[1:] ).copy()

        # ========================================================
        # EMPTY DATA
        # ========================================================

        if plot_df.empty:

            st.warning( "No valid data are available for this relationship." )

        else:

            # ====================================================
            # SCATTER
            # ====================================================

            scatter_kwargs = dict(

                data_frame=plot_df,
                x=x_property,
                y=y_property,
                custom_data=[ "molecule_id" ],
                opacity=0.70,
                labels={ x_property: relationship_properties[ x_property ], y_property: relationship_properties[ y_property ], },
            )

            # ====================================================
            # COLOR SCALE
            # ====================================================

            if color_col != "None":

                scatter_kwargs["color"] = color_col

                scatter_kwargs[ "color_continuous_scale" ] = COLORS["sequential"]

            # ====================================================
            # CREATE FIGURE
            # ====================================================

            fig_relationship = px.scatter( **scatter_kwargs )

            # ====================================================
            # TRACE STYLE
            # ====================================================

            fig_relationship.update_traces(

                marker=dict( size=9, line=dict( width=0.6, color="rgba(255,255,255,0.45)", ), ),

                hovertemplate=(

                    "<b>QM7-X ID:</b> "
                    "%{customdata[0]}"

                    "<br>"

                    f"<b>"
                    f"{relationship_properties[x_property]}"
                    f"</b>: "
                    "%{x:.6g}"

                    "<br>"

                    f"<b>"
                    f"{relationship_properties[y_property]}"
                    f"</b>: "
                    "%{y:.6g}"

                    +

                    (
                        f"<br><b>"
                        f"{relationship_properties[color_col]}"
                        f"</b>: "
                        "%{marker.color:.6g}"
                        if color_col != "None"
                        else ""
                    )

                    +

                    "<extra></extra>"
                ),
            )

            # ====================================================
            # SCIENTIFIC LAYOUT
            # ====================================================

            fig_relationship = scientific_layout(
                fig_relationship, height=540, show_legend=False, dark_mode=False, )

            # ====================================================
            # COLORBAR
            # ====================================================

            if color_col != "None":

                fig_relationship.update_coloraxes(
                    colorbar=dict(
                        title=dict(
                            text=relationship_properties[color_col],
                            font=dict( family="Arial", size=13, color=COLORS["text_secondary"], ),
                        ),
                        tickfont=dict( family="Arial", size=11, color=COLORS["text_secondary"], ),
                        thickness=14,
                        len=0.78,
                        x=1.02,
                        xanchor="left",
                        outlinewidth=0,
                    )
                )

            # ====================================================
            # CORRELATION
            # ====================================================

            correlation = plot_df[ [ x_property, y_property, ] ].corr().iloc[0, 1]

            # ====================================================
            # INTELLIGENT CORRELATION POSITION
            # ====================================================

            if np.isfinite(correlation):

                x_values = plot_df[ x_property ].to_numpy( dtype=float )
                y_values = plot_df[ y_property ].to_numpy( dtype=float )
                x_min = np.nanmin( x_values )
                x_max = np.nanmax( x_values )
                y_min = np.nanmin( y_values )
                y_max = np.nanmax( y_values )

                x_range = x_max - x_min
                y_range = y_max - y_min

                if x_range == 0:
                    x_range = 1.0

                if y_range == 0:
                    y_range = 1.0

                x_norm = ( (x_values - x_min) / x_range )

                y_norm = ( (y_values - y_min) / y_range )

                # ------------------------------------------------
                # Candidate positions
                # ------------------------------------------------

                corners = {

                    "top_left": { "x": 0.03, "y": 0.97, "xanchor": "left", "yanchor": "top", "x_low": 0.00, "x_high": 0.25, "y_low": 0.75, "y_high": 1.00, },

                    "top_right": { "x": 0.97, "y": 0.97, "xanchor": "right", "yanchor": "top", "x_low": 0.75, "x_high": 1.00, "y_low": 0.75, "y_high": 1.00, },

                    "bottom_left": { "x": 0.03, "y": 0.03, "xanchor": "left", "yanchor": "bottom", "x_low": 0.00, "x_high": 0.25, "y_low": 0.00, "y_high": 0.25, },

                    "bottom_right": { "x": 0.97, "y": 0.03, "xanchor": "right", "yanchor": "bottom", "x_low": 0.75, "x_high": 1.00, "y_low": 0.00, "y_high": 0.25, },
                }

                # ------------------------------------------------
                # Score each corner by number of points
                # ------------------------------------------------

                corner_scores = {}

                for name, corner in corners.items():

                    mask = ( (x_norm >= corner["x_low"]) & (x_norm <= corner["x_high"]) & (y_norm >= corner["y_low"]) & (y_norm <= corner["y_high"]) )

                    corner_scores[name] = int( np.sum(mask) )

                # ------------------------------------------------
                # Choose least populated corner
                # ------------------------------------------------

                best_corner = min( corner_scores, key=corner_scores.get, )

                selected_corner = corners[ best_corner ]

                # ------------------------------------------------
                # Correlation annotation
                # ------------------------------------------------

                correlation_text = ( "Pearson correlation: " f"<b>r = {correlation:.3f}</b>" )

                fig_relationship.add_annotation(

                    x=selected_corner["x"],

                    y=selected_corner["y"],

                    xref="paper",

                    yref="paper",

                    text=correlation_text,

                    showarrow=False,

                    xanchor=selected_corner[ "xanchor" ],

                    yanchor=selected_corner[ "yanchor" ],

                    font=dict( size=12, color=COLORS[ "text_primary" ], ),

                    opacity=0.95,
                )

            # ====================================================
            # HOVER MODE
            # ====================================================

            fig_relationship.update_layout( hovermode="closest" )

            # ============================================================
            # INITIALIZE SELECTION STATE
            # ============================================================

            if "relationship_selected_molecules" not in st.session_state:

                st.session_state[ "relationship_selected_molecules" ] = []


            if "relationship_plot_version" not in st.session_state:

                st.session_state[ "relationship_plot_version" ] = 0

            # ============================================================
            # HOVER MODE
            # ============================================================

            fig_relationship.update_layout( hovermode="closest" )

            # ============================================================
            # DISPLAY SCATTER
            # ============================================================

            col1, col2 = st.columns([3, 1])

            with col1:

                plot_key = ( "property_relationship_scatter_" f"{st.session_state['relationship_plot_version']}" )

                st.markdown(
                    """
                    <style>
                    div[data-testid="stPlotlyChart"] .js-plotly-plot,
                    div[data-testid="stPlotlyChart"] .js-plotly-plot *,
                    div[data-testid="stPlotlyChart"] .plotly,
                    div[data-testid="stPlotlyChart"] .plotly * {
                        cursor: pointer !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                event = st.plotly_chart( fig_relationship, width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"), key=plot_key, )

                st.caption( f"{len(plot_df):,} molecules available for this property relationship." )


            with col2:

                # ============================================================
                # SELECTION PANEL
                # ============================================================

                st.markdown(
                    """
                    <div style="
                        padding: 0.2rem 0 0.8rem 0;
                    ">
                    <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #0891B2; margin-bottom: 0.3rem; ">
                        Molecular selection
                    </div>

                    <div style=" font-size: 1.25rem; font-weight: 700; color: #0F172A; margin-bottom: 0.35rem; ">
                        Select molecules
                    </div>

                    <div style=" font-size: 0.88rem; line-height: 1.5; color: #64748B; ">
                        Select individual molecules by clicking a point,
                        or use Box Select / Lasso Select for multiple molecules.
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ============================================================
                # ACTIONS
                # ============================================================

                st.markdown(
                    """
                    <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #64748B; margin: 0.5rem 0 0.45rem 0; ">
                        Selection controls
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                start_selection = st.button(
                    "▶  Start selection",
                    key="relationship_start_selection",
                    width="stretch",
                    help="Clear the current selection and start selecting molecules again.",
                )

                reset_selection = st.button(
                    "↻  Reset selection",
                    key="relationship_reset_selection",
                    width="stretch",
                    help="Clear all selected molecules.",
                )

                # ============================================================
                # HANDLE START / RESET
                # ============================================================

                if start_selection or reset_selection:

                    st.session_state["relationship_selected_molecules"] = []

                    st.session_state["relationship_plot_version"] += 1

                    st.rerun()

                # ============================================================
                # READ PLOTLY SELECTION
                # ============================================================

                plot_selected_ids = []

                if event is not None:

                    try:
                        selected_points = event.selection.get("points", [])
                    except Exception:
                        selected_points = []

                    for point in selected_points:

                        customdata = point.get("customdata")

                        if customdata is None:
                            continue

                        # Plotly normally returns customdata as a sequence.
                        if isinstance(customdata, (list, tuple)):

                            if not customdata:
                                continue

                            molecule_id = customdata[0]

                        else:
                            molecule_id = customdata

                        molecule_id = str(molecule_id)

                        if molecule_id not in plot_selected_ids:
                            plot_selected_ids.append(molecule_id)

                # ============================================================
                # SYNCHRONIZE PLOTLY SELECTION WITH SESSION STATE
                # ============================================================

                if event is not None:

                    st.session_state["relationship_selected_molecules"] = ( plot_selected_ids )

                # ============================================================
                # CURRENT SELECTION
                # ============================================================

                current_selection = st.session_state.get( "relationship_selected_molecules", [], )

                current_selection = [ str(molecule_id) for molecule_id in current_selection ]

                # ============================================================
                # SELECTION STATUS
                # ============================================================

                st.markdown(
                    """
                    <div style=" margin-top: 1.2rem; padding-top: 0.9rem; border-top: 1px solid #E2E8F0; ">
                    <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #64748B; margin-bottom: 0.55rem; ">
                        Current selection
                    </div> </div> """, unsafe_allow_html=True, )

                # ============================================================
                # DYNAMIC SELECTION BADGE
                # ============================================================

                if not current_selection:

                    st.markdown(
                        """
                        <div style=" display: flex; align-items: center; gap: 0.6rem; padding: 0.7rem 0.8rem; border-radius: 0.55rem; background: #F8FAFC; border: 1px solid #E2E8F0; ">
                        <div style=" font-size: 1rem; ">
                            ○
                        </div>
                        <div>
                        <div style=" font-size: 0.86rem; font-weight: 600; color: #334155; ">
                            No molecules selected
                        </div>

                        <div style=" font-size: 0.75rem; color: #64748B; ">
                            Click a point or use Box / Lasso Select
                        </div>
                            </div>
                        </div> """, unsafe_allow_html=True, )

                elif len(current_selection) == 1:

                    st.markdown(
                        """
                        <div style=" padding: 0.75rem 0.85rem; border-radius: 0.55rem; background: #ECFEFF; border: 1px solid #A5F3FC; ">
                        <div style=" font-size: 0.75rem; font-weight: 700; color: #0891B2; text-transform: uppercase; letter-spacing: 0.05em; ">
                            Single molecule
                        </div>

                        <div style=" margin-top: 0.15rem; font-size: 1.05rem; font-weight: 700; color: #0F172A; ">
                            1 molecule selected
                        </div>

                        <div style=" margin-top: 0.2rem; font-size: 0.76rem; color: #475569; ">
                            Ready for molecular analysis
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.markdown(
                        f"""
                        <div style=" padding: 0.75rem 0.85rem; border-radius: 0.55rem; background: #EFF6FF; border: 1px solid #BFDBFE; ">
                        <div style=" font-size: 0.75rem; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 0.05em; ">
                            Multiple molecules
                        </div>

                        <div style=" margin-top: 0.15rem; font-size: 1.05rem; font-weight: 700; color: #0F172A; ">
                            {len(current_selection):,} molecules selected
                        </div>

                        <div style=" margin-top: 0.2rem; font-size: 0.76rem; color: #475569; ">
                            Ready for comparative analysis
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ============================================================
                # GUIDANCE
                # ============================================================

                st.markdown(
                    """
                    <div style=" margin-top: 0.9rem; padding: 0.8rem 0.85rem; border-left: 3px solid #0891B2; background: #F8FAFC; border-radius: 0 0.45rem 0.45rem 0; ">
                    <div style=" font-size: 0.78rem; font-weight: 700; color: #334155; margin-bottom: 0.25rem; ">
                        How to select
                    </div>

                    <div style=" font-size: 0.75rem; line-height: 1.55; color: #64748B; ">
                        <strong>Click</strong> a point for single-molecule analysis.
                        <br>
                        <strong>Box Select</strong> or <strong>Lasso Select</strong>
                        for comparative analysis.
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ========================================================
    # SELECTED MOLECULES / COMPARATIVE ANALYSIS
    # ========================================================

    selected_ids = st.session_state.get( "relationship_selected_molecules", [], )

    # Remove duplicates while preserving order
    selected_ids = list( dict.fromkeys( str(molecule_id) for molecule_id in selected_ids ) )

    # ========================================================
    # EMPTY STATE
    # ========================================================

    if not selected_ids:

        st.markdown(
            """
            <div style=" margin-top: 1.5rem; padding: 1rem 1.1rem; border: 1px solid #E2E8F0; border-radius: 0.65rem; background: #F8FAFC; ">
            <div style=" font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 0.3rem; ">
                Molecular analysis
            </div>

            <div style=" font-size: 0.9rem; color: #475569; ">
                Select a molecule from the scatter plot to inspect
                its 2D structure and interactive 3D geometry.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # SINGLE MOLECULE
    # ========================================================

    elif len(selected_ids) == 1:

        molecule_id = selected_ids[0]

        try:

            molecule = load_molecule( hdf, molecule_id, )

            # =================================================
            # BASIC MOLECULAR INFORMATION
            # =================================================

            atomic_numbers = np.asarray( molecule["atNUM"] ).reshape(-1)

            formula = formula_from_atnums( atomic_numbers )

            n_atoms = len( atomic_numbers )

            unique_elements = len( set( int(x) for x in atomic_numbers ) )

        except Exception as e:

            st.error( f"Could not load molecule {molecule_id}: {e}" )

        else:

            # =================================================
            # MOLECULE HEADER
            # =================================================

            st.markdown(
                f"""
                <div style=" margin-top: 1.5rem; margin-bottom: 1rem; padding: 1rem 1.1rem; border: 1px solid #E2E8F0; border-radius: 0.7rem; background: #FFFFFF; ">

                <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #0891B2; margin-bottom: 0.25rem; ">
                    Selected molecule
                </div>

                <div style=" font-size: 1.5rem; font-weight: 700; color: #0F172A; margin-bottom: 0.55rem; ">
                    🧬 {formula}
                </div>

                <div style=" display: flex; flex-wrap: wrap; gap: 1.4rem; color: #64748B; font-size: 0.78rem; ">

                <span>
                    <strong style="color:#334155;">
                        Molecule ID
                    </strong>
                    &nbsp; {molecule_id}
                </span>

                <span>
                    <strong style="color:#334155;">
                        Atoms
                    </strong>
                    &nbsp; {n_atoms}
                </span>

                <span>
                    <strong style="color:#334155;">
                        Elements
                    </strong>
                    &nbsp; {unique_elements}
                </span>

                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption( "Explore the selected molecule in two and three dimensions." )

            view_mode = st.segmented_control(
                "View",
                options=["2D", "3D", "2D + 3D"],
                default="2D + 3D",
                key="molecular_analysis_view_mode", width="content",
            )

            # view_mode represents the user's general viewing preference (2D vs 3D vs both)

            show_2d = view_mode in ("2D", "2D + 3D")
            show_3d = view_mode in ("3D", "2D + 3D")

            def _render_2d_controls():

                with st.expander("2D Visualization Controls"):

                    col_a, col_b = st.columns(2)

                    with col_a:

                        background_2d = st.selectbox( "Background", [ "White", "Black", "Transparent", ], index=0, key="molecular_analysis_2d_background", )

                    with col_b:

                        size_2d = st.selectbox( "Image size", [ "Small", "Medium", "Large", ], index=1, key="molecular_analysis_2d_size", )

                    col_c, col_d = st.columns(2)

                    with col_c:

                        labels_2d = st.checkbox( "Atom labels", value=True, key="molecular_analysis_2d_labels", )

                    with col_d:

                        hydrogens_2d = st.checkbox( "Show H", value=True, key="molecular_analysis_2d_hydrogens", )

                return background_2d, size_2d, labels_2d, hydrogens_2d


            def _render_2d_figure(background_2d, size_2d, labels_2d, hydrogens_2d):

                st.markdown(
                    """
                    <div style="
                        margin: 1.2rem 0 0.7rem 0;
                        padding-bottom: 0.45rem;
                        border-bottom: 1px solid var(--clr-border-soft);
                    ">
                        <div style="
                            font-size: 0.80rem;
                            font-weight: 700;
                            letter-spacing: 0.12em;
                            text-transform: uppercase;
                            color: var(--clr-accent-strong);
                            text-align: center;
                        ">
                            2D Molecular Structure
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                size_map_2d = { "Small": (400, 300), "Medium": (500, 400), "Large": (650, 500), }

                background_map_2d = { "White": "white", "Black": "black", "Transparent": "transparent", }

                try:

                    image_2d = molecule_2d_image(
                        molecule,
                        size=size_map_2d[size_2d],
                        background=background_map_2d[ background_2d ],
                        show_hydrogens=hydrogens_2d,
                        show_atom_labels=labels_2d,
                    )

                    image_col_left, image_col_center, image_col_right = st.columns( [1, 6, 1] )

                    with image_col_center:

                        st.image( image_2d, width="stretch", )

                        png_buffer = io.BytesIO()

                        image_2d.save( png_buffer, format="PNG", )

                        st.download_button(
                            "↓ Export PNG",
                            data=png_buffer.getvalue(),
                            file_name=( f"molecule_{molecule_id}_2d.png" ),
                            mime="image/png",
                            width='stretch',
                            key=( f"molecular_analysis_2d_download_" f"{molecule_id}" ),
                        )

                except Exception as e:

                    st.warning( f"Could not create 2D structure: {e}" )


            def _render_3d_controls():

                with st.expander("3D Visualization Controls"):

                    control_col1, control_col2, control_col3 = ( st.columns(3) )

                    with control_col1:

                        viewer_projection = st.selectbox( "Projection", [ "Free 3D", "X", "Y", "Z", ], key="molecular_analysis_projection", )

                    with control_col2:

                        viewer_background = st.selectbox( "Background", [ "black", "white", "transparent", ], key="molecular_analysis_background", )

                    with control_col3:

                        viewer_representation = st.selectbox( "Representation", [ "Stick + Sphere", "Stick", "Sphere", "Line", ], key="molecular_analysis_representation", )

                    control_col4, control_col5 = ( st.columns(2) )

                    with control_col4:

                        viewer_atom_scale = st.slider( "Atom size", min_value=0.15, max_value=0.60, value=0.30, step=0.01, key="molecular_analysis_atom_scale", )

                    with control_col5:

                        viewer_bond_radius = st.slider( "Bond radius", min_value=0.03, max_value=0.30, value=0.12, step=0.01, key="molecular_analysis_bond_radius", )

                    control_col6, control_col7, control_col8 = ( st.columns(3) )

                    with control_col6:

                        viewer_show_labels = st.checkbox( "Atom labels", value=True, key="molecular_analysis_labels", )

                    with control_col7:

                        viewer_orthographic = st.checkbox( "Orthographic", value=False, key="molecular_analysis_orthographic", )

                    with control_col8:

                        viewer_spin = st.checkbox( "Auto rotate", value=False, key="molecular_analysis_spin", )

                return (
                    viewer_projection,
                    viewer_background,
                    viewer_representation,
                    viewer_atom_scale,
                    viewer_bond_radius,
                    viewer_show_labels,
                    viewer_orthographic,
                    viewer_spin,
                )


            def _render_3d_figure(
                viewer_projection,
                viewer_background,
                viewer_representation,
                viewer_atom_scale,
                viewer_bond_radius,
                viewer_show_labels,
                viewer_orthographic,
                viewer_spin,
            ):

                st.markdown(
                    """
                    <div style="
                        margin: 1.2rem 0 0.7rem 0;
                        padding-bottom: 0.45rem;
                        border-bottom: 1px solid var(--clr-border-soft);
                    ">
                        <div style="
                            font-size: 0.80rem;
                            font-weight: 700;
                            letter-spacing: 0.12em;
                            text-transform: uppercase;
                            color: var(--clr-accent-strong);
                            text-align: center;
                        ">
                            3D Molecular Structure
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                try:

                    xyz = molecule_to_xyz( molecule )

                    projection_map = { "Free 3D": "free", "X": "x", "Y": "y", "Z": "z", }

                    projection = projection_map[ viewer_projection ]

                    # =============================================
                    # ATOM CLICK BRIDGE (hidden widget)
                    # =============================================

                    atom_click_bridge_label = ( "__qm7x_atom_click_bridge_" f"{molecule_id}__" )

                    st.markdown(
                        f"""
                        <style>
                        div[data-testid="stNumberInput"]:has(input[aria-label="{atom_click_bridge_label}"]) {{
                            opacity: 0;
                            position: fixed;
                            top: -9999px;
                            left: -9999px;
                            pointer-events: none;
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    clicked_atom_raw = st.number_input(
                        atom_click_bridge_label,
                        min_value=-1,
                        max_value=max(n_atoms - 1, 0),
                        value=-1,
                        step=1,
                        key=( "pairwise_atom_click_bridge_" f"{molecule_id}" ),
                        label_visibility="collapsed",
                    )

                    clicked_atom_index = ( int(clicked_atom_raw) if 0 <= int(clicked_atom_raw) < n_atoms else None )

                    display_3d_molecule(
                        xyz=xyz,
                        height=375,
                        representation=viewer_representation,
                        show_labels=viewer_show_labels,
                        atom_scale=viewer_atom_scale,
                        bond_radius=viewer_bond_radius,
                        background=viewer_background,
                        spin=viewer_spin,
                        projection=projection,
                        orthographic=viewer_orthographic,
                        show_controls=False,
                        enable_atom_selection=True,
                        selected_atom_index=clicked_atom_index,
                        atom_click_bridge_label=atom_click_bridge_label,
                        download_filename=( f"molecule_{molecule_id}_3d.png" ),
                        download_filename_xyz=( f"molecule_{molecule_id}.xyz" ),
                    )

                except Exception as e:

                    st.warning( f"Could not create 3D structure: {e}" )

                    clicked_atom_index = None

                # =================================================
                # 🔬 ATOMIC INSPECTION (clicked atom)
                # =================================================

                st.markdown( "#### 🔬 Atomic Inspection" )

                if clicked_atom_index is None:

                    st.info( "No atom selected yet. Click an atom in the " "3D structure above." )

                else:

                    atXYZ_inspect = np.asarray( molecule.get("atXYZ", []) )

                    atPOL_inspect = np.asarray( molecule.get("atPOL", []) ).reshape(-1)

                    hCHG_inspect = np.asarray( molecule.get("hCHG", []) ).reshape(-1)

                    vdwR_inspect = np.asarray( molecule.get("vdwR", []) ).reshape(-1)

                    element_symbol_inspect = ATOM_INFO.get( int(atomic_numbers[clicked_atom_index]), ("?", "Unknown"), )[0]

                    insp_col1, insp_col2, insp_col3 = st.columns(3)

                    with insp_col1:

                        st.metric( "Element", f"{element_symbol_inspect} " f"(atom {clicked_atom_index + 1})", )

                    with insp_col2:

                        st.metric( "Atomic number", int(atomic_numbers[clicked_atom_index]), )

                    with insp_col3:

                        if len(hCHG_inspect) == n_atoms:

                            st.metric( "Hirshfeld charge", f"{hCHG_inspect[clicked_atom_index]:.4f} e", )

                        else:

                            st.metric( "Hirshfeld charge", "N/A", )

                    insp_col4, insp_col5, insp_col6 = st.columns(3)

                    with insp_col4:

                        if len(atPOL_inspect) == n_atoms:

                            st.metric( "Polarizability", f"{atPOL_inspect[clicked_atom_index]:.4f} a₀³", )

                        else:

                            st.metric( "Polarizability", "N/A", )

                    with insp_col5:

                        if len(vdwR_inspect) == n_atoms:

                            st.metric( "vdW radius", f"{vdwR_inspect[clicked_atom_index]:.4f} bohr", )

                        else:

                            st.metric( "vdW radius", "N/A", )

                    with insp_col6:

                        if ( atXYZ_inspect.ndim == 2 and atXYZ_inspect.shape[1] == 3 and len(atXYZ_inspect) == n_atoms ):

                            xyz_atom = atXYZ_inspect[clicked_atom_index]

                            st.metric( "X, Y, Z (Å)", f"{xyz_atom[0]:.3f}, " f"{xyz_atom[1]:.3f}, " f"{xyz_atom[2]:.3f}", )

                        else:

                            st.metric( "X, Y, Z (Å)", "N/A", )

            # ====================================================
            # LAYOUT PER VIEW MODE
            # ====================================================

            if view_mode == "2D + 3D":

                # All controls above; both figures side by side below with equal width.

                controls_col_2d, controls_col_3d = st.columns( 2, gap="large" )

                with controls_col_2d:

                    settings_2d = _render_2d_controls()

                with controls_col_3d:

                    settings_3d = _render_3d_controls()

                fig_col_2d, fig_col_3d = st.columns( 2, gap="large" )

                with fig_col_2d:

                    _render_2d_figure( *settings_2d )

                with fig_col_3d:

                    _render_3d_figure( *settings_3d )

            elif view_mode == "2D":

                # Figure centered on the left; controls in a sidebar on the right.

                fig_col, controls_col = st.columns( [3, 1], gap="large" )

                with controls_col:

                    settings_2d = _render_2d_controls()

                with fig_col:

                    _render_2d_figure( *settings_2d )

            else:

                # Figure centered and scaled to the available width; controls in a sidebar on the right.

                fig_col, controls_col = st.columns( [3, 1], gap="large" )

                with controls_col:

                    settings_3d = _render_3d_controls()

                with fig_col:

                    _render_3d_figure( *settings_3d )

        # =================================================
        # ATOMIC PROPERTIES
        # =================================================

        with st.expander( "⚛ Atomic properties", expanded=False, ):

            st.caption( "Per-atom properties for the selected molecular geometry." )

            # =================================================
            # ATOMIC DATA
            # =================================================

            atomic_numbers = np.asarray( molecule.get("atNUM", []) ).reshape(-1)

            n_atoms = len(atomic_numbers)

            if n_atoms == 0:

                st.info( "No atomic information available." )

            else:

                # =================================================
                # SAFE ARRAY CONVERSION
                # =================================================

                def atomic_array( property_name, n_atoms, ):

                    if property_name not in molecule:

                        return np.full( n_atoms, np.nan, )

                    try:

                        values = np.asarray( molecule[property_name] ).reshape(-1)

                        if len(values) != n_atoms:

                            return np.full( n_atoms, np.nan, )

                        return values.astype(float)

                    except Exception:

                        return np.full( n_atoms, np.nan, )

                # =================================================
                # ATOMIC PROPERTIES
                # =================================================

                hirshfeld_charge = atomic_array( "hCHG", n_atoms, )

                polarizability = atomic_array( "atPOL", n_atoms, )

                vdw_radius = atomic_array( "vdwR", n_atoms, )

                # =================================================
                # BUILD TABLE
                # =================================================

                atomic_rows = []

                for i, atomic_number in enumerate( atomic_numbers ):

                    atomic_number = int( atomic_number )

                    symbol, name = ATOM_INFO.get( atomic_number, ( "X", "Unknown", ), )

                    atomic_rows.append(
                        {
                            "Atom": i + 1,

                            "Element": symbol,

                            "Atomic number": atomic_number,

                            "Hirshfeld charge (e)": hirshfeld_charge[i],

                            "Polarizability (a₀³)": polarizability[i],

                            "vdW radius (bohr)": vdw_radius[i],
                        }
                    )

                atomic_df = pd.DataFrame( atomic_rows )

                # =================================================
                # TABLE
                # =================================================

                st.dataframe(
                    atomic_df,
                    width='stretch',
                    hide_index=True,
                    column_config={

                        "Atom": st.column_config.NumberColumn( "Atom", format="%d", ),

                        "Element": st.column_config.TextColumn( "Element", ),

                        "Atomic number": st.column_config.NumberColumn( "Z", format="%d", ),

                        "Hirshfeld charge (e)": st.column_config.NumberColumn( "Hirshfeld charge", format="%.6f", ),

                        "Polarizability (a₀³)": st.column_config.NumberColumn( "Polarizability", format="%.6f", ),

                        "vdW radius (bohr)": st.column_config.NumberColumn( "vdW radius", format="%.6f", ),
                    },
                )

                # =================================================
                # SUMMARY METRICS
                # =================================================

                valid_charge = hirshfeld_charge[ np.isfinite(hirshfeld_charge) ]

                valid_polarizability = polarizability[ np.isfinite(polarizability) ]

                valid_vdw = vdw_radius[ np.isfinite(vdw_radius) ]

                metric_col1, metric_col2, metric_col3, metric_col4 = ( st.columns(4) )

                with metric_col1:

                    st.metric( "Atoms", f"{n_atoms:,}", )

                with metric_col2:

                    if len(valid_charge):

                        total_charge = float( np.sum(valid_charge) )

                        # Remove numerical noise
                        if abs(total_charge) < 1e-10:
                            total_charge = 0.0

                        st.metric( "Total charge", f"{total_charge:.4f} e", )

                    else:

                        st.metric( "Total charge", "N/A", )

                with metric_col3:

                    if len(valid_polarizability):

                        st.metric( "Mean polarizability", f"{np.mean(valid_polarizability):.4f} a₀³", )

                    else:

                        st.metric( "Mean polarizability", "N/A", )

                with metric_col4:

                    if len(valid_vdw):

                        st.metric( "Mean vdW radius", f"{np.mean(valid_vdw):.4f} bohr", )

                    else:

                        st.metric( "Mean vdW radius", "N/A", )

                # =================================================
                # CHARGE INTERPRETATION
                # =================================================

                if len(valid_charge):

                    positive_atoms = int( np.sum( valid_charge > 1e-6 ) )

                    negative_atoms = int( np.sum( valid_charge < -1e-6 ) )

                    neutral_atoms = ( n_atoms - positive_atoms - negative_atoms )

                    st.caption(
                        f"Charge distribution: "
                        f"{positive_atoms} positive, "
                        f"{negative_atoms} negative, "
                        f"{neutral_atoms} approximately neutral."
                    )

                # =================================================
                # MISSING DATA
                # =================================================

                missing_properties = []

                if not np.isfinite( hirshfeld_charge ).any():

                    missing_properties.append( "Hirshfeld charge" )

                if not np.isfinite( polarizability ).any():

                    missing_properties.append( "Polarizability" )

                if not np.isfinite( vdw_radius ).any():

                    missing_properties.append( "vdW radius" )

                if missing_properties:

                    st.caption( "Unavailable atomic properties: " + ", ".join( missing_properties ))


        # =================================================
        # ATOMIC POSITIONS
        # =================================================

        with st.expander( "📍 Atomic positions", expanded=False, ):

            try:

                positions_df = molecule_to_position_dataframe( molecule, molecule_id, )

                st.dataframe(
                    positions_df,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "Molecule ID": st.column_config.TextColumn( "Molecule ID", ),

                        "Atom": st.column_config.NumberColumn( "Atom", format="%d", ),

                        "Element": st.column_config.TextColumn( "Element", ),

                        "Atomic number": st.column_config.NumberColumn( "Atomic number", format="%d", ),

                        "X (Å)": st.column_config.NumberColumn( "X (Å)", format="%.6f", ),

                        "Y (Å)": st.column_config.NumberColumn( "Y (Å)", format="%.6f", ),

                        "Z (Å)": st.column_config.NumberColumn( "Z (Å)", format="%.6f", ),
                    },
                )

            except Exception as e:

                st.warning( f"Could not generate atomic positions: {e}" )
        

    # ============================================================
    # MULTIPLE MOLECULES SELECTED
    # ============================================================

    else:

        # --------------------------------------------------------
        # SELECTION SUMMARY
        # --------------------------------------------------------

        n_selected = len(selected_ids)

        st.markdown(
            f"""
            <div style=" margin: 1.4rem 0 1rem 0; padding: 1rem 1.15rem; border: 1px solid #BFDBFE; border-radius: 12px; background: #EFF6FF; ">
            <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #2563EB; margin-bottom: 0.3rem; ">
                Multiple molecules selected
            </div>

            <div style=" font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem; ">
                {n_selected} molecules ready for comparison
            </div>

            <div style=" font-size: 0.86rem; line-height: 1.5; color: #475569; ">
                Compare their molecular structures and quantum-chemical
                properties side by side.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # OPEN COMPARATIVE ANALYSIS
        # --------------------------------------------------------

        open_comparative = st.button(
            f"🧬  Analyze {n_selected} selected molecules",
            key="open_comparative_analysis_button",
            width='content',
            help="Open the comparative analysis for the selected molecules.",
        )

        if open_comparative:
            st.session_state["show_comparative_analysis"] = True

        # --------------------------------------------------------
        # COMPARATIVE ANALYSIS
        # --------------------------------------------------------

        if st.session_state.get( "show_comparative_analysis", False, ):

            st.markdown(
                """
                <div style=" margin: 1.8rem 0 1rem 0; padding-left: 1rem; border-left: 3px solid #2563EB; ">
                <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #2563EB; margin-bottom: 0.3rem; ">
                    Comparative analysis
                </div>

                <div style=" font-size: 1.45rem; font-weight: 700; color: #0F172A; margin-bottom: 0.3rem; ">
                    Molecular comparison
                </div>

                <div style=" font-size: 0.9rem; color: #64748B; line-height: 1.5; max-width: 850px; ">
                    Compare molecular structures and selected
                    quantum-chemical properties across the molecules
                    selected in the scatter plot.
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ----------------------------------------------------
            # CLOSE ANALYSIS
            # ----------------------------------------------------

            if st.button( "×  Close comparative analysis", key="close_comparative_analysis", ):
                st.session_state["show_comparative_analysis"] = False
                st.rerun()

            # ----------------------------------------------------
            # LOAD SELECTED MOLECULES
            # ----------------------------------------------------

            comparison_molecules = []

            for molecule_id in selected_ids:

                try:

                    molecule = load_molecule( hdf, molecule_id, )

                    atomic_numbers = np.asarray( molecule["atNUM"] ).reshape(-1)

                    formula = formula_from_atnums( atomic_numbers )

                    comparison_molecules.append( { "id": str(molecule_id), "molecule": molecule, "formula": formula, "n_atoms": len(atomic_numbers), } )

                except Exception as e:

                    st.warning( f"Could not load molecule {molecule_id}: {e}" )

            if not comparison_molecules:

                st.error( "No valid molecules could be loaded for comparison." )

            else:

                # =================================================
                # SUMMARY METRICS
                # =================================================

                n_molecules = len( comparison_molecules )
                atom_counts = [ item["n_atoms"] for item in comparison_molecules ]
                formulas = [ item["formula"] for item in comparison_molecules ]

                unique_formulas = len( set(formulas) )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric( "Selected molecules", n_molecules, )

                with col2:
                    st.metric( "Minimum atoms", min(atom_counts), )

                with col3:
                    st.metric( "Maximum atoms", max(atom_counts), )

                with col4:
                    st.metric( "Unique formulas", unique_formulas, )

                st.caption( "Molecules selected from the property Relationships scatter plot." )

                # =================================================
                # MOLECULAR STRUCTURES
                # =================================================

                st.markdown( "### 🧬 Molecular structures" )

                st.caption( "Interactive 3D representations of the selected molecules." )

                display_comparative_3d_viewers( comparison_molecules )

                # =================================================
                # PROPERTY COMPARISON
                # =================================================

                section_header("Property comparison")

                st.caption( "Compare one quantum-chemical property across the selected molecules." )

                comparison_properties = {

                    "ePBE0+MBD": "PBE0 + MBD energy (eV)",
                    "ePBE0": "PBE0 energy (eV)",

                    "eMBD": "MBD energy (eV)",

                    "eAT": "Atomization energy (eV)",

                    "eTS": "TS dispersion energy (eV)",

                    "HLgap": "HOMO–LUMO gap (eV)",

                    "eH": "HOMO energy (eV)",

                    "eL":
                        "LUMO energy (eV)",

                    "DIP":
                        "Dipole moment (e·Å)",

                    "mPOL":
                        "Molecular polarizability",

                    "mC6":
                        "Molecular C₆ coefficient",

                    "eNN":
                        "Nuclear–nuclear energy (eV)",

                    "eNE":
                        "Nuclear–electron energy (eV)",

                    "eEE":
                        "Electron–electron energy (eV)",

                    "eXC":
                        "Exchange–correlation energy (eV)",

                    "eX":
                        "Exchange energy (eV)",

                    "eC":
                        "Correlation energy (eV)",

                    "eXX":
                        "Exact exchange energy (eV)",
                }

                available_comparison_properties = [

                    prop

                    for prop in comparison_properties

                    if any( prop in item["molecule"] for item in comparison_molecules )
                ]

                if available_comparison_properties:

                    default_property = (
                        "HLgap"
                        if "HLgap"
                        in available_comparison_properties
                        else available_comparison_properties[0]
                    )

                    selected_plot_property = st.selectbox(

                        "Property to compare",

                        options=available_comparison_properties,

                        index=available_comparison_properties.index( default_property ),

                        format_func=lambda x: comparison_properties[x],

                        key="comparison_plot_property",
                    )

                    comparison_plot_rows = []

                    for item in comparison_molecules:

                        molecule = item["molecule"]

                        if selected_plot_property not in molecule:
                            continue

                        try:

                            value = float( np.asarray( molecule[ selected_plot_property ] ) .reshape(-1)[0] )

                        except Exception:

                            continue

                        comparison_plot_rows.append( { "Molecule ID": item["id"], "Formula": item["formula"], "Atoms": item["n_atoms"], "Value": value, } )

                    comparison_plot_df = pd.DataFrame( comparison_plot_rows )

                    if not comparison_plot_df.empty:

                        fig_comparison = px.bar(

                            comparison_plot_df,

                            x="Molecule ID",

                            y="Value",

                            custom_data=[ "Molecule ID", "Formula", "Atoms", ],

                            labels={ "Value": comparison_properties[ selected_plot_property ], "Molecule ID": "Molecule", }, )

                        fig_comparison.update_traces(

                            marker=dict( line=dict( width=0.7, color="rgba(255,255,255,0.35)", ) ),

                            hovertemplate=(

                                "<b>Molecule ID:</b> %{customdata[0]}"

                                "<br><b>Formula:</b> "
                                "%{customdata[1]}"

                                "<br><b>Atoms:</b> "
                                "%{customdata[2]}"

                                "<br>"

                                f"<b>"
                                f"{comparison_properties[selected_plot_property]}"
                                f"</b>: "

                                "%{y:.6e}"

                                "<extra></extra>"
                            ),
                        )

                        fig_comparison = scientific_layout( fig_comparison, height=430, show_legend=False, dark_mode=False, )

                        fig_comparison.update_layout( xaxis_title="Molecule", yaxis_title=comparison_properties[selected_plot_property], hovermode="closest", )

                        fig_comparison.update_yaxes( exponentformat="e", showexponent="all", )

                        st.plotly_chart( fig_comparison, width="stretch", key="comparison_property_plot", )

                    else:

                        st.info( "No valid numerical values are available for the selected property." )

                else:

                    st.info( "No comparable quantum-chemical properties are available for the selected molecules." )

                # =================================================
                # PROPERTY TABLE
                # =================================================

                st.divider()

                with st.expander( "### 🔬 Molecular properties" ):

                    st.caption( "Numerical properties of the selected molecules." )

                    property_rows = []

                    for item in comparison_molecules:

                        molecule = item["molecule"]

                        row = { "Molecule ID": item["id"], "Formula": item["formula"], "Atoms": item["n_atoms"], }

                        for prop, label in comparison_properties.items():

                            if prop not in molecule:
                                continue

                            try:

                                value = float( np.asarray( molecule[prop] ) .reshape(-1)[0] )

                                row[label] = value

                            except Exception:

                                continue

                        property_rows.append(row)

                    comparison_df = pd.DataFrame( property_rows )

                    if not comparison_df.empty:

                        numeric_columns = [

                            column

                            for column
                            in comparison_df.columns

                            if column
                            not in [ "Molecule ID", "Formula", "Atoms", ]

                            and pd.api.types.is_numeric_dtype( comparison_df[column] )
                        ]

                        formatted_comparison_df = (
                            comparison_df.copy()
                        )

                        for column in numeric_columns:

                            formatted_comparison_df[column] = (
                                formatted_comparison_df[column]
                                .map(
                                    lambda value:
                                        f"{value:.6e}"
                                        if pd.notna(value)
                                        else ""
                                )
                            )

                        st.dataframe( formatted_comparison_df, width='stretch', hide_index=True, )

                    else:

                        numeric_columns = []

                        st.info( "No molecular property data are available." )

                # =================================================
                # ADVANCED PROPERTY PROFILE
                # =================================================

                if ( not comparison_df.empty and len(numeric_columns) > 0 and len(comparison_df) > 1 ):

                    with st.expander( "📊 Advanced property profile", expanded=False, ):

                        st.caption(
                            "Properties are standardized independently "
                            "using z-scores to visualize relative "
                            "differences across different physical scales."
                        )

                        heatmap_df = ( comparison_df[ numeric_columns ].copy() )

                        heatmap_df.index = ( comparison_df[ "Molecule ID" ] )
                        normalized_df = ( heatmap_df - heatmap_df.mean() )

                        std = heatmap_df.std()

                        std = std.replace( 0, np.nan, )

                        normalized_df = ( normalized_df / std ).fillna(0)

                        fig_heatmap = go.Figure(

                            data=go.Heatmap(

                                z=normalized_df.T.values,

                                x=normalized_df.index,

                                y=normalized_df.columns,

                                colorscale=COLORS["diverging"],

                                colorbar=dict( title=dict( text="Z-score" ), thickness=14, ),

                                hovertemplate=( "<b>QM7-X ID:</b> %{x}" "<br><b>Property:</b> %{y}" "<br><b>Z-score:</b> %{z:.2f}" "<extra></extra>" ),
                            )
                        )

                        fig_heatmap = scientific_layout( fig_heatmap, height=520, show_legend=False, dark_mode=True, )
                        fig_heatmap.update_layout( xaxis_title="Molecule ID", yaxis_title="Property", )

                        st.plotly_chart( fig_heatmap, width='stretch', key="comparison_property_heatmap", )
