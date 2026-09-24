import streamlit as st
import plotly.express as px
import pandas as pd

from modules.loader import (
    print_hdf5_tree_limited,
    qm7x_property_table,
    get_geometry_statistics_from_index,
    classify_geometry_id,
    get_composition_statistics,
    get_composition_statistics_from_index,
    get_available_properties,   
)

from modules.plot_style import scientific_layout
from modules.plot_distributions import plot_size_distribution

from modules.theme import apply_global_theme, property_table_html, COLORS, section_header


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config( page_title="Overview", page_icon="🧬", layout="wide", )

apply_global_theme()

# ============================================================
# TITLE
# ============================================================

st.title("Dataset Exploration")
st.write("Composition, conformers, elemental distribution, and core dataset statistics.")

# ============================================================
# CHECK DATASET
# ============================================================

if "hdf" not in st.session_state:
    st.warning("⚠️ **No dataset is currently loaded.** ")
    st.info(" ℹ️ Please return to the Home page and load a dataset from a local file, an uploaded file, or Zenodo.")
    st.stop()

hdf = st.session_state["hdf"]

# ============================================================
# CHECK ANALYSIS INDEXES
# ============================================================

if not st.session_state.get("analysis_indexes_ready", False):
    st.warning("Dataset analysis indexes are not ready.")
    st.info("Please return to the Home page and load the dataset again.")
    st.stop()

# ============================================================
# GET PREPARED INDEXES
# ============================================================

geometry_df = st.session_state["geometry_index"]
opt_df = st.session_state["opt_index"]

# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

try:
    n_molecules = len(hdf)

except Exception:
    n_molecules = len(list(hdf.keys()))

# ============================================================
# DATASET CACHE ID
# ============================================================

dataset_id = (
    st.session_state.get("dataset_id")
    or st.session_state.get("dataset_source_path")
    or st.session_state.get("dataset_name")
    or "current_dataset"
)

# ============================================================
# GEOMETRY STATISTICS
# ============================================================

if "geometry_stats_cache" not in st.session_state:
    st.session_state["geometry_stats_cache"] = {}

geometry_cache = st.session_state["geometry_stats_cache"]


if dataset_id not in geometry_cache:

    try:

        geometry_stats = get_geometry_statistics_from_index(
            geometry_df,
            opt_df,
            n_molecules_total=n_molecules,
        )

        geometry_cache[dataset_id] = geometry_stats

    except Exception as e:

        st.error( "Could not analyze geometry index structure: " f"{e}" )
        st.stop()

else:

    geometry_stats = geometry_cache[dataset_id]

# ============================================================
# PROPERTY INFORMATION
# ============================================================

try:

    available_properties = get_available_properties(hdf)

    property_table = qm7x_property_table()

    property_table = property_table[ property_table["Property"].isin( available_properties ) ].copy()

    n_properties = len(property_table)

except Exception:

    property_table = pd.DataFrame()
    n_properties = 0

# ============================================================
# COMPOSITION STATISTICS
# ============================================================

if "composition_stats_cache" not in st.session_state:
    st.session_state["composition_stats_cache"] = {}

composition_cache = st.session_state["composition_stats_cache"]

composition_cache_key = f"{dataset_id}_composition"


if composition_cache_key not in composition_cache:

    with st.spinner("Analyzing molecular composition..."):

        try:

            composition_cache[composition_cache_key] = ( get_composition_statistics_from_index(opt_df) )

        except ValueError:

            try:

                composition_cache[composition_cache_key] = ( get_composition_statistics( hdf, max_molecules=None, ) )

            except Exception as e:

                st.error( f"Could not analyze molecular composition: {e}" )

                st.stop()

        except Exception as e:

            st.error( f"Could not analyze molecular composition: {e}" )

            st.stop()

composition_stats = composition_cache[composition_cache_key]

# ============================================================
# TABS
# ============================================================

tab_overview, tab_molecular, tab_properties = st.tabs([ "📊 Overview", "⚛️ Molecular Analysis", "🧪 Physicochemical Properties", ])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab_overview:

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    # ============================================================
    # Helpers
    # ============================================================

    def _prepare_size_distribution(df, without_h=False):

        if df is None or df.empty:
            return None
        
        if "n_atoms" not in df.columns:
            return None

        n_atoms = pd.to_numeric( df["n_atoms"], errors="coerce", )

        if without_h:

            if "n_H" not in df.columns:
                return None

            n_H = pd.to_numeric( df["n_H"], errors="coerce", ).fillna(0)

            values = n_atoms - n_H

        else:

            values = n_atoms

        values = values.dropna().astype(int)

        if values.empty:
            return None

        return ( values .value_counts() .sort_index() .rename_axis("Atoms") .reset_index(name="Count") )


    def _get_size_means(df):

        if df is None or df.empty:
            return {}

        if "n_atoms" not in df.columns:
            return {}

        n_atoms = pd.to_numeric( df["n_atoms"], errors="coerce", )

        result = { "all": float(n_atoms.mean()) }

        if "n_H" in df.columns:

            n_H = pd.to_numeric( df["n_H"], errors="coerce", )

            result["without_H"] = float( (n_atoms - n_H).mean() )

        return result


    def _get_elements(df):

        """
        Return the elements present in the molecular dataset.
        """

        if df is None or df.empty:
            return []

        element_columns = {"H": "n_H", "C": "n_C", "N": "n_N", "O": "n_O", "S": "n_S", "Cl": "n_Cl"}

        elements = []

        for symbol, column in element_columns.items():

            if column not in df.columns:
                continue

            values = pd.to_numeric( df[column], errors="coerce", )

            if values.fillna(0).max() > 0:
                elements.append(symbol)

        return elements

    # ============================================================
    # Prepare data
    # ============================================================

    molecule_all = _prepare_size_distribution( opt_df, without_h=False, )

    molecule_without_H = _prepare_size_distribution( opt_df, without_h=True, )

    structure_all = _prepare_size_distribution( geometry_df, without_h=False, )

    structure_without_H = _prepare_size_distribution( geometry_df, without_h=True, )

    molecule_means = _get_size_means(opt_df)
    structure_means = _get_size_means(geometry_df)

    elements = _get_elements(opt_df)
    
    with col1:
        st.metric( "Molecules", f"{n_molecules:,}", )

    with col2:
        st.metric( "Total conformers", f"{geometry_stats['total_geometries']:,}",  )

    with col3:
        st.metric( "Elements", ", ".join(elements), )

    with col4:
        st.metric( "Properties / descriptors", f"{n_properties:,}", )

    # ========================================================
    # ANALYSIS SECTION
    # ========================================================

    section_header("Molecular Size Distribution")


    dark_mode = False

    size_distribution_available = any( data is not None for data in ( molecule_all, molecule_without_H, structure_all, structure_without_H, ) )
    
    # ============================================================
    # Plot helper
    # ============================================================

    if size_distribution_available:

        # ============================================================
        # Controls + figures
        # ============================================================

        col1, col2, col3 = st.columns( [1, 1, 0.48], gap="medium", )

        with col3:

            st.markdown("**Controls**")

            size_mode = st.radio( "Size", ["All atoms", "Without H"], index=0, key="molecular_size_mode", )

            show_mean = st.checkbox( "Show mean value", value=True, key="molecular_size_show_mean", )

            show_bar_counts = st.checkbox( "Show bar counts", value=True, key="molecular_size_show_bar_counts", )

            st.divider()

            st.caption("Download figures")

            # ========================================================
            # Molecule figure for download
            # ========================================================

            molecule_fig = plot_size_distribution(
                molecule_all,
                molecule_without_H,
                molecule_means,
                "Molecules · "
                f"{n_molecules:,}",
                "molecule",
                size_mode,
                dark_mode=False,
                bar_color=COLORS["accent"],
                show_mean=show_mean,
                show_bar_counts=show_bar_counts,
            )

            try:

                molecule_png = molecule_fig.to_image( format="png", scale=2, )
                st.download_button(
                    label="↓ Molecules PNG",
                    data=molecule_png,
                    file_name="molecular_size_distribution.png",
                    mime="image/png",
                    width='stretch',
                )

            except Exception:

                st.caption( "PNG export requires Kaleido." )

            # ========================================================
            # Conformer figure for download
            # ========================================================

            structure_fig = plot_size_distribution(
                structure_all,
                structure_without_H,
                structure_means,
                "Conformers · "
                f"{geometry_stats['total_geometries']:,}",
                "structure",
                size_mode,
                dark_mode=False,
                bar_color="#C79124",
                show_mean=show_mean,
                show_bar_counts=show_bar_counts,
            )

            try:

                structure_png = structure_fig.to_image( format="png", scale=2, )

                st.download_button(
                    label="↓ Conformers PNG",
                    data=structure_png,
                    file_name="geometry_size_distribution.png",
                    mime="image/png",
                    width='stretch',
                )

            except Exception:

                st.caption( "PNG export requires Kaleido." )


        # ============================================================
        # Molecules
        # ============================================================

        with col1:

            fig = plot_size_distribution(
                molecule_all,
                molecule_without_H,
                molecule_means,
                "Molecules · "
                f"{n_molecules:,}",
                "molecule",
                size_mode,
                dark_mode=False,
                bar_color=COLORS["accent"],
                show_mean=show_mean,
                show_bar_counts=show_bar_counts,
            )

            st.plotly_chart( fig, width='stretch', )

            # --------------------------------------------------------
            # Elements found
            # --------------------------------------------------------

            if elements:

                st.caption( "Elements found: " + ", ".join(elements) )

        # ============================================================
        # Structures
        # ============================================================

        with col2:

            fig = plot_size_distribution(
                structure_all,
                structure_without_H,
                structure_means,
                "Conformers · "
                f"{geometry_stats['total_geometries']:,}",
                "structure",
                size_mode,
                dark_mode=False,
                bar_color="#C79124",
                show_mean=show_mean,
                show_bar_counts=show_bar_counts,
            )

            st.plotly_chart( fig, width='stretch', )

            # --------------------------------------------------------
            # Elements found
            # --------------------------------------------------------

            if elements:

                st.caption( "Elements found: " + ", ".join(elements) )
    else:

        st.info(
            "This dataset's HDF5 schema does not expose atom-count "
            "information (no atomic-number array, coordinate array, "
            "or other recognized per-atom dataset was found), so "
            "molecular and geometry size distributions are not "
            "available for this dataset."
        )

    # ========================================================
    # HDF5 STRUCTURE
    # ========================================================

    with st.expander("🌳 HDF5 dataset structure", expanded=False):
        try:
            print_hdf5_tree_limited( hdf, max_molecules=1, )
        except Exception as e:
            st.error(f"Could not inspect HDF5 structure: {e}")

# ============================================================
# TAB 2 — MOLECULAR COMPOSITION & STRUCTURE
# ============================================================        

with tab_molecular:

    # ========================================================
    # ANALYSIS SCOPE CONTROL
    # ========================================================

    col_scope_info, col_scope_control = st.columns( [3, 1], gap="medium", )

    with col_scope_info:

        st.caption(
            "Choose whether the summary metrics and charts below "
            "describe unique molecules or every conformer record. "
        )

    with col_scope_control:

        analysis_scope = st.segmented_control(
            "Analysis scope",
            options=[ "Molecules", "Conformers", ],
            default="Molecules",
            key="composition_analysis_scope",
            label_visibility="collapsed",
        )

    if analysis_scope is None:

        analysis_scope = "Molecules"

    element_symbols = ("H", "C", "N", "O", "S", "Cl")

    required_opt_columns = {"molecule_id", "formula"}

    opt_has_composition_columns = required_opt_columns.issubset( set(opt_df.columns) )

    if analysis_scope == "Conformers" and opt_has_composition_columns:

        composition_cache_key_geometry = ( f"{dataset_id}_composition_geometry" )

        if composition_cache_key_geometry not in composition_cache:

            with st.spinner( "Analyzing molecular composition across " "all geometry records..." ):

                try:

                    opt_element_columns = [ f"n_{symbol}" for symbol in element_symbols if f"n_{symbol}" in opt_df.columns ]

                    molecule_composition = opt_df[ [ "molecule_id", "formula", *opt_element_columns, ] ]

                    geometry_composition = ( geometry_df[["molecule_id", "n_atoms"]] .merge( molecule_composition, on="molecule_id", how="left", ) )
                    formula_counts_geometry = ( geometry_composition["formula"] .dropna() .value_counts() .to_dict() )

                    element_counts_geometry = {
                        symbol: int( geometry_composition[f"n_{symbol}"].sum() )
                        for symbol in element_symbols
                        if f"n_{symbol}" in geometry_composition.columns
                        and int( geometry_composition[f"n_{symbol}"].sum() ) > 0
                    }

                    total_atoms_geometry = int( geometry_composition["n_atoms"].sum() )

                    composition_cache[composition_cache_key_geometry] = {
                        "formula_counts": formula_counts_geometry,
                        "element_counts": element_counts_geometry,
                        "total_atoms": total_atoms_geometry,
                    }

                except Exception as e:

                    st.warning(
                        "Could not analyze molecular composition "
                        f"across geometries ({e}). Showing "
                        "molecule-level composition instead."
                    )

                    composition_cache[composition_cache_key_geometry] = ( composition_stats )

        active_composition_stats = ( composition_cache[composition_cache_key_geometry] )

    else:

        if analysis_scope == "Conformers":

            st.info(
                "Conformer-level elemental/formula composition "
                "requires composition columns in the optimized "
                "index. Rebuild the dataset index to enable it. "
                "Showing molecule-level composition instead."
            )

        active_composition_stats = composition_stats

    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    formula_counts = active_composition_stats["formula_counts"]
    molecules_with_geometry = ( geometry_stats["molecules_with_geometry"] )
    geometry_counts = ( geometry_stats["geometry_counts"] )
    # --------------------------------------------------------
    # Molecules with multiple geometries
    # --------------------------------------------------------

    if molecules_with_geometry > 0:

        molecules_with_multiple_geometries = sum( count > 1 for count in geometry_counts )

        multi_geometry_fraction = ( molecules_with_multiple_geometries / molecules_with_geometry * 100 )

    else:

        multi_geometry_fraction = 0.0

    # --------------------------------------------------------
    # Four summary metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total atoms",
            f"{active_composition_stats['total_atoms']:,}",
            help=(
                "Summed across every geometry record "
                "(including conformers)."
                if analysis_scope == "Conformers"
                else "Summed across unique molecules."
            ),
        )

    with col2:

        st.metric( "Unique formulas", f"{len(formula_counts):,}", )

    with col3:

        most_frequent_formula = ( max( formula_counts, key=formula_counts.get, ) if formula_counts else "—" )

        st.metric( "Most frequent formula", most_frequent_formula, )

    with col4:

        if analysis_scope == "Conformers":

            # "Molecules with multiple geometries" 
            st.metric(
                "Avg. conformers / molecule",
                f"{geometry_stats['average_geometries']:.1f}",
                help=(
                    "\"Molecules with multiple geometries\" is a "
                    "molecule-level concept and has no conformer-"
                    "level equivalent -- showing the average number "
                    "of conformer records per molecule instead."
                ),
            )

        else:

            st.metric( "Molecules with multiple geometries", f"{multi_geometry_fraction:.1f}%", )


    # ========================================================
    # TWO-COLUMN LAYOUT
    # ========================================================

    col1, col2 = st.columns(2)

    # ========================================================
    # ELEMENTAL COMPOSITION
    # ========================================================

    with col1:

        section_header( f"Elemental composition &middot; {analysis_scope}" )
        

        element_counts = active_composition_stats["element_counts"]

        if element_counts:

            # ------------------------------------------------
            # DATAFRAME
            # ------------------------------------------------

            composition_df = (
                pd.DataFrame( { "Element": list( element_counts.keys() ), "Count": list( element_counts.values() ), } )
                .query("Count > 0")
                .sort_values( "Count", ascending=False, ) .reset_index(drop=True) )

            total_element_atoms = ( composition_df["Count"].sum() )

            composition_df["Percentage"] = ( composition_df["Count"] / total_element_atoms * 100 )

            # ------------------------------------------------
            # BAR PLOT
            # ------------------------------------------------

            fig_elements = px.bar( composition_df, x="Element", y="Count", text="Count", labels={ "Element": "Element", "Count": "Number of atoms", }, )

            fig_elements.update_traces(
                marker=dict( color=composition_df["Count"], colorscale=COLORS["sequential_stops"], showscale=False, ),
                texttemplate="%{text:,}", textposition="outside", textfont=dict( color=COLORS["text_primary"], ),
                hovertemplate=( "<b>%{x}</b><br>" "Atoms: %{y:,}" "<extra></extra>" ),
            )


            # ------------------------------------------------
            # LAYOUT
            # ------------------------------------------------

            fig_elements.update_layout(
                plot_bgcolor=COLORS["bg"],
                paper_bgcolor=COLORS["bg"],
                font=dict( color=COLORS["text_primary"], ),

                margin=dict( l=20, r=20, t=25, b=45, ),

                xaxis=dict(
                    title="Element",
                    showgrid=False,
                    linecolor=COLORS["border"],
                    tickcolor=COLORS["border"],
                    tickfont=dict( color=COLORS["text_secondary"], ),
                    title_font=dict( color=COLORS["text_secondary"], ),
                ),

                yaxis=dict(
                    title="Number of atoms",
                    showgrid=True,
                    gridcolor=COLORS["border_soft"],
                    gridwidth=1,
                    zeroline=False,
                    linecolor=COLORS["border"],
                    tickcolor=COLORS["border"],
                    tickfont=dict( color=COLORS["text_secondary"], ),
                    title_font=dict( color=COLORS["text_secondary"], ),
                ),

                hoverlabel=dict( bgcolor=COLORS["bg_elevated"], bordercolor=COLORS["border"], font_color=COLORS["text_primary"], ),
            )

            # ------------------------------------------------
            # HEIGHT / GENERAL SCIENTIFIC STYLE
            # ------------------------------------------------

            fig_elements = scientific_layout( fig_elements, height=450, show_legend=False, dark_mode=False, )

            st.plotly_chart( fig_elements, width='stretch', key="composition_element_counts", )

            st.caption( f"{total_element_atoms:,} atoms across " f"{len(composition_df)} elements." )
            
        else:

            st.info( "No elemental information was found." )

    # ========================================================
    # MOST FREQUENT MOLECULAR FORMULAS
    # ========================================================

    with col2:

        section_header(f"Most frequent molecular formulas &middot; {analysis_scope}" )

        if formula_counts:

            # ------------------------------------------------
            # DATAFRAME
            # ------------------------------------------------

            formula_df = ( pd.DataFrame( { "Molecular formula": list( formula_counts.keys() ), "Molecules": list( formula_counts.values() ), } ) .sort_values( "Molecules", ascending=False, ) .reset_index(drop=True) )

            # ------------------------------------------------
            # TOP 10 FOR PLOT
            # ------------------------------------------------

            top_formula_df = ( formula_df .head(10) .sort_values( "Molecules", ascending=True, ) )

            # ------------------------------------------------
            # BAR PLOT
            # ------------------------------------------------

            fig_formulas = px.bar(
                top_formula_df,
                x="Molecules",
                y="Molecular formula",
                orientation="h",
                text="Molecules",
                labels={ "Molecules": "Number of molecules", "Molecular formula": ( "Molecular formula" ), }, )

            fig_formulas.update_traces(
                marker=dict( color=top_formula_df["Molecules"], colorscale=COLORS["sequential_orange_stops"], showscale=False, ),
                texttemplate="%{text:,}",
                textposition="outside",
                textfont=dict( color=COLORS["text_primary"], ),
                hovertemplate=( "<b>%{y}</b><br>" "Molecules: %{x:,}" "<extra></extra>" ),
            )


            # ------------------------------------------------
            # LAYOUT
            # ------------------------------------------------

            fig_formulas.update_layout(
                plot_bgcolor=COLORS["bg"],
                paper_bgcolor=COLORS["bg"],
                font=dict( color=COLORS["text_primary"], ),

                margin=dict( l=20, r=20, t=25, b=45, ),

                xaxis=dict(
                    title="Number of molecules",
                    showgrid=True,
                    gridcolor=COLORS["border_soft"],
                    gridwidth=1,
                    zeroline=False,
                    linecolor=COLORS["border"],
                    tickcolor=COLORS["border"],
                    tickfont=dict( color=COLORS["text_secondary"], ),
                    title_font=dict( color=COLORS["text_secondary"], ), ),

                yaxis=dict(
                    title="Molecular formula",
                    showgrid=False,
                    linecolor=COLORS["border"],
                    tickcolor=COLORS["border"],
                    tickfont=dict( color=COLORS["text_secondary"], ),
                    title_font=dict( color=COLORS["text_secondary"], ), ),

                hoverlabel=dict(
                    bgcolor=COLORS["bg_elevated"],
                    bordercolor=COLORS["border"],
                    font_color=COLORS["text_primary"], ),
            )

            # ------------------------------------------------
            # HEIGHT / GENERAL SCIENTIFIC STYLE
            # ------------------------------------------------

            fig_formulas = scientific_layout( fig_formulas, height=450, show_legend=False, dark_mode=False, )

            st.plotly_chart( fig_formulas, width='stretch', key="composition_formula_distribution", )

            st.caption( f"Top 10 of {len(formula_df):,} " "unique molecular formulas." )

        else:
            st.info( "No molecular formulas could be determined." )


    section_header(f"Conformational Analysis &middot; {analysis_scope}")

    # ========================================================
    # BASIC VALUES
    # ========================================================

    min_geometries = ( geometry_stats[ "min_geometries" ] )

    max_geometries = ( geometry_stats[ "max_geometries" ] )

    geometry_counts = ( geometry_stats[ "geometry_counts" ] )

    # ============================================================
    # GEOMETRIES PER MOLECULE
    # ============================================================

    if geometry_counts:

        # --------------------------------------------------------
        # Prepare distribution
        # --------------------------------------------------------

        geometry_count_df = ( pd.Series(geometry_counts) .value_counts() .sort_index() .rename_axis("Conformers per molecule") .reset_index(name="Molecules") )

        geometry_count_df["Conformer records"] = ( geometry_count_df["Conformers per molecule"] * geometry_count_df["Molecules"] )

        if analysis_scope == "Conformers":

            distribution_y_column = "Conformer records"
            distribution_y_title = "Number of conformer records"
            distribution_hover_label = "Conformer records"

        else:

            distribution_y_column = "Molecules"
            distribution_y_title = "Number of molecules"
            distribution_hover_label = "Molecules"

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------

        fig_distribution = px.bar(
            geometry_count_df,
            x="Conformers per molecule",
            y=distribution_y_column,
            labels={ "Conformers per molecule": "Conformers per molecule", distribution_y_column: distribution_y_title, },
        )

        # --------------------------------------------------------
        # Trace styling
        # --------------------------------------------------------

        fig_distribution.update_traces(
            hovertemplate=( "<b>%{x} geometries per molecule</b><br>" f"{distribution_hover_label}: " "%{y:,}" "<extra></extra>" ),
            cliponaxis=False,
        )

        # --------------------------------------------------------
        # Scientific layout
        # --------------------------------------------------------

        fig_distribution = scientific_layout( fig_distribution, height=500, show_legend=False, dark_mode=False, )

        # --------------------------------------------------------
        # X-axis
        # --------------------------------------------------------

        max_geometries = geometry_count_df["Conformers per molecule"].max()

        fig_distribution.update_xaxes( tickmode="linear", dtick=5, range=[0.5, max_geometries + 0.5], )

        # --------------------------------------------------------
        # Y-axis
        # --------------------------------------------------------

        fig_distribution.update_yaxes( title_text=distribution_y_title, )

        # --------------------------------------------------------
        # Display
        # --------------------------------------------------------

        st.plotly_chart( fig_distribution, width='stretch', key="geometry_count_distribution", )

        if analysis_scope == "Conformers":

            st.caption(
                "Distribution of conformer records grouped by how "
                "many total conformers their parent molecule has. "
                "Exact values are available by hovering over each bar."
            )

        else:

            st.caption( "Distribution of molecules by the number of associated conformer records. Exact values are available by hovering over each bar." )
    else:

        st.info( "No conformer-count information is available." )

# ========================================================
# 4. GEOMETRY IDENTIFIER STRUCTURE
# ========================================================

    geometry_id_counts = ( geometry_stats.get( "geometry_id_counts", {}, ) )

    if geometry_id_counts:

        if { "molecule_token", "isomer_token", "configuration_token", }.issubset(geometry_df.columns): parsed_geometry_df = geometry_df[ [ "geometry_id", "is_opt", "molecule_token", "isomer_token", "configuration_token", ] ].rename( columns={ "geometry_id": "Conformer ID", "is_opt": "Optimized", "molecule_token": "Molecule", "isomer_token": "Isomer", "configuration_token": "Configuration", } )
        else:

            parsed = geometry_df["geometry_id"].apply(classify_geometry_id)

            parsed_geometry_df = pd.DataFrame( { "Conformer ID": geometry_df["geometry_id"], "Optimized": geometry_df["is_opt"], "Molecule": parsed.apply(lambda p: p["molecule_token"]), "Isomer": parsed.apply(lambda p: p["isomer_token"]), "Configuration": parsed.apply( lambda p: p["configuration_token"] ), } )

        parsed_geometry_df["Occurrences"] = ( parsed_geometry_df["Conformer ID"].map( parsed_geometry_df["Conformer ID"].value_counts() ) )

        if not parsed_geometry_df.empty:

            configuration_df = ( parsed_geometry_df["Configuration"] .dropna() .value_counts() .rename_axis("Configuration") .reset_index(name="Conformer IDs") )

        # ------------------------------------------------------------
        # ISOMER
        # ------------------------------------------------------------

            isomer_df = ( parsed_geometry_df["Isomer"] .dropna() .value_counts() .rename_axis("Isomer") .reset_index(name="Conformer IDs") )

    # ============================================================
    # IDENTIFIER PREVIEW
    # ============================================================

    with st.expander( "📋 Elemental composition table" ):

        if element_counts:

            display_elements = ( composition_df.copy() )

            display_elements["Count"] = ( display_elements["Count"] .map(lambda x: f"{x:,}") )

            display_elements["Percentage"] = ( display_elements["Percentage"] .map(lambda x: f"{x:.2f}%") )

            st.dataframe( display_elements, width='stretch', hide_index=True, )
        else:

            st.info( "No elemental composition data is available for this dataset." )


    with st.expander( "📋 Complete molecular formula table" ):

        if formula_counts:

            st.dataframe( formula_df, width='stretch', hide_index=True, )
    
        else:

            st.info( "No molecular formula data is available for this dataset." )


    with st.expander("ℹ️ About this analysis", expanded=False):

        st.markdown( """ This analysis describes the structure of the geometry identifiers used in the QM7-X dataset. """ )

        # ========================================================
        # GEOMETRY IDENTIFIER STRUCTURE
        # ========================================================

        st.markdown("#### Conformer identifier structure based in QM7X dataset")

        st.code( "Geom-m1-i1-c1-opt", language="text", )

        st.markdown(
            """
            Each conformer identifier is composed of several tokens:

            - **`m1`** → molecule identifier
            - **`i1`** → isomer identifier
            - **`c1`** → configuration identifier
            - **`opt`** → optimized conformer
            """
        )

        st.caption(
            "`-opt` indicates that the geometry is classified as optimized. "
            "The absence of this suffix indicates a non-optimized conformer."
        )

        # ========================================================
        # IDENTIFIER STATISTICS
        # ========================================================

        st.markdown("#### Identifier statistics")

        unique_molecules = ( parsed_geometry_df["Molecule"] .dropna() .nunique() )

        unique_isomers = ( parsed_geometry_df["Isomer"] .dropna() .nunique() )

        unique_configurations = ( parsed_geometry_df["Configuration"] .dropna() .nunique() )

        unique_geometry_ids = ( parsed_geometry_df["Conformer ID"] .dropna() .nunique() )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric( "Molecule identifiers", f"{unique_molecules:,}", )

        with col2:
            st.metric( "Isomer identifiers", f"{unique_isomers:,}", )

        with col3:
            st.metric( "Configuration identifiers", f"{unique_configurations:,}", )

        with col4:
            st.metric( "Unique geometry IDs", f"{unique_geometry_ids:,}", )

        # ========================================================
        # INTERPRETATION
        # ========================================================

        st.markdown("#### Interpretation")

        st.markdown(
            """
            The identifier hierarchy can therefore be interpreted as:

            **Molecule → Isomer → Configuration → Conformer**

            A molecule may contain multiple isomers, and an isomer may
            contain multiple configurations. Each conformer record
            corresponds to a specific combination of these identifiers.

            The identifier-based analysis describes how records are
            organized in the dataset. It does **not** by itself classify
            molecular shape or physical state.
            """
        )

        st.info(
            """
            Molecular shape classifications such as linear, planar,
            bent, or tetrahedral require the actual atomic coordinates
            and are therefore outside the scope of this identifier-based
            analysis.
            """
        )






# ============================================================
# TAB — PROPERTIES
# ============================================================

with tab_properties:

    # --------------------------------------------------------
    # PROPERTIES AVAILABLE IN LOADED HDF5
    # --------------------------------------------------------

    available_properties = get_available_properties(hdf)

    # --------------------------------------------------------
    # COMPLETE QM7-X PROPERTY REFERENCE
    # --------------------------------------------------------

    reference_df = qm7x_property_table()

    # Keep only properties that are actually present
    # in the currently loaded HDF5 dataset.
    properties_df = reference_df[ reference_df["Property"].isin( available_properties ) ].copy()

    properties_df.reset_index( drop=True, inplace=True, )

    # --------------------------------------------------------
    # CATEGORY DEFINITIONS
    # --------------------------------------------------------

    energetic_codes = { "eAT", "eC", "eEE", "eKIN", "eMBD", "eNE", "eNN", "ePBE0", "ePBE0+MBD", "eTS", "eX", "eXC", "eXX", }

    electronic_codes = { "eH", "eL", "HLgap", }

    response_codes = { "DIP", "mC6", "mPOL", }

    structural_codes = { "atNUM", "atPOL", "atXYZ", "hCHG", "vdwR", }

    # --------------------------------------------------------
    # CATEGORY TABLES
    # --------------------------------------------------------

    energetic_df = ( properties_df[ properties_df["Property"].isin( energetic_codes ) ] .reset_index(drop=True) )

    molecular_df = ( properties_df[ properties_df["Property"].isin( electronic_codes | response_codes ) ] .reset_index(drop=True) )

    atomic_df = ( properties_df[ properties_df["Property"].isin( structural_codes ) ] .reset_index(drop=True) )

    # --------------------------------------------------------
    # DYNAMIC COUNTS
    # --------------------------------------------------------

    total_properties = len(properties_df)
    energetic_count = len(energetic_df)
    molecular_count = len(molecular_df)
    structural_count = len(atomic_df)

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    st.caption(
        f"Overview of the {total_properties} molecular, "
        "energetic, electronic, atomic and structural "
        "properties available in the loaded dataset."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns( 4, gap="medium", )

    with col1:
        st.metric( "Total properties", total_properties, )

    with col2:
        st.metric( "Energetic", energetic_count, )

    with col3:
        st.metric( "Molecular / electronic", molecular_count, )

    with col4:
        st.metric( "Atomic / structural", structural_count, )

    # --------------------------------------------------------
    # ENERGETIC PROPERTIES
    # --------------------------------------------------------

    if not energetic_df.empty:

        section_header("Energetic properties")

        st.markdown( property_table_html( energetic_df ), unsafe_allow_html=True, )

    # --------------------------------------------------------
    # MOLECULAR / ELECTRONIC PROPERTIES
    # --------------------------------------------------------

    if not molecular_df.empty:

        section_header("Molecular and electronic properties")

        st.markdown( property_table_html( molecular_df ), unsafe_allow_html=True, )

    # --------------------------------------------------------
    # ATOMIC / STRUCTURAL PROPERTIES
    # --------------------------------------------------------

    if not atomic_df.empty:

        section_header("Atomic and structural properties")

        st.markdown( property_table_html( atomic_df ), unsafe_allow_html=True, )