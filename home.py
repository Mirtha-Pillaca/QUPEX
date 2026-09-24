
import streamlit as st
import h5py

from pathlib import Path
from PIL import Image

from modules.loader import ( open_hdf5, open_zenodo_hdf5, identify_dataset, )

from modules.prepare_indexes import ( prepare_dataset_indexes, compute_dataset_id, )

from modules.theme import apply_global_theme

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config( page_title="Home", page_icon="🧬", layout="wide", )

apply_global_theme()

st.markdown(
    """
    <div style="
        margin: 1.8rem 0 1.4rem 0;
    ">
    <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #0891B2; margin-bottom: 0.35rem; ">
        Welcome
    </div>

    <div style=" font-size: 2rem; font-weight: 750; color: #111827; line-height: 1.15; margin-bottom: 0.4rem; ">
        Quantum Property EXplorer
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================
logo = Image.open( "assets/logo_white.png" ).convert("RGBA")

col_logo, col_text = st.columns( [1, 3], vertical_alignment="center", )

# ============================================================
# LOGO
# ============================================================
with col_logo:
    st.image(logo)

# ============================================================
# HERO TEXT
# ============================================================
with col_text:

    st.markdown("""

        QUPEX is a lightweight app to explore **molecular structures**, **geometries**, and **chemical space** properties.

        It converts **HDF5** data into fast **Parquet indexes** and provides:
        - Molecular & conformer statistics  
        - 2D/3D molecular visualization 
        - Property-space exploration 
        - Molecular clustering 
        
        Additionally, the app includes an optional **machine‑learning** module for dipole‑moment prediction.

        *Note: The input dataset must follow a data structure similar to **QM7‑X**, as described in Hoja et al. (2021), available at [Zenodo](https://zenodo.org/records/4288677).*       
    """)

# ============================================================
# HERO DIVIDER
# ============================================================

st.markdown( """ <div style=" width: 100%; height: 2px; background: var(--clr-accent); border-radius: 2px; margin: 1.25rem 0 1.75rem 0; "></div> """, unsafe_allow_html=True, )
# ============================================================
# GET STARTED
# ============================================================

st.markdown(
    """
    <div style=" margin: 1.8rem 0 1.4rem 0; ">
    <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #0891B2; margin-bottom: 0.35rem; ">
       Get Started
    </div>
    <div style=" font-size: 2rem; font-weight: 750; color: #111827; line-height: 1.15; margin-bottom: 0.4rem; ">
        Load a molecular dataset 
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA SOURCE
# ============================================================

DEFAULT_QM7X_ZENODO_URL = "https://zenodo.org/records/22893015/files/qm7x-eq.hdf5?download=1"

use_default_dataset = st.checkbox( "Use default QM7-X dataset from Zenodo", value=False, key="app_use_default_dataset", )

if use_default_dataset:

    st.caption( "Loads the QM7-X equilibrium dataset directly from Zenodo -- no URL needed. The file is downloaded once and cached locally; later runs reuse the cached copy." )

    source = "Zenodo"

    zenodo_url = DEFAULT_QM7X_ZENODO_URL

    # Auto-load once per session 
    load_button = ( not st.session_state.get( "default_dataset_attempted", False, ) )

    if load_button:
        st.session_state[ "default_dataset_attempted" ] = True

else:

    st.session_state[ "default_dataset_attempted" ] = False

    source = st.segmented_control( "Data source", options=[ "Local", "Upload", "Zenodo", ], default="Local", key="app_dataset_source",  )

    # ========================================================
    # LOCAL
    # ========================================================

    if source == "Local":

        st.caption( "Open an HDF5 file already stored on your computer." )

        file_path = st.text_input( "HDF5 file path", value="", key="app_local_hdf5_path", )

        load_button = st.button( "Load local dataset", type="primary", use_container_width=True, key="load_local_dataset", )

    # ========================================================
    # UPLOAD
    # ========================================================

    elif source == "Upload":

        st.caption( "Upload an HDF5 file through the browser." )

        uploaded_file = st.file_uploader( "HDF5 file", type=[ "h5", "hdf5", ], key="app_uploaded_hdf5", )

        load_button = ( uploaded_file is not None )

    # ========================================================
    # ZENODO
    # ========================================================

    elif source == "Zenodo":

        st.caption( "Open an HDF5 dataset directly from Zenodo." )

        zenodo_url = st.text_input( "Zenodo file URL", key="app_zenodo_url", )

        load_button = st.button( "Load Zenodo dataset", type="primary", use_container_width=True, key="load_zenodo_dataset", )

# ============================================================
# LOAD DATASET
# ============================================================

if load_button:

    try:
        # ====================================================
        # LOCAL
        # ====================================================

        if source == "Local":

            if not file_path.strip():

                st.warning( "Please provide an HDF5 file path." ) 
                st.stop()

            file_path_obj = Path( file_path )

            if not file_path_obj.exists():

                st.error( f"File not found: {file_path}" )

                st.stop()

            hdf = open_hdf5( file_path )

            dataset_name = ( file_path_obj.name )

            source_path = file_path

            dataset_size_mb = ( file_path_obj.stat().st_size / (1024 ** 2) )
        # ====================================================
        # UPLOAD
        # ====================================================

        elif source == "Upload":

            hdf = h5py.File( uploaded_file, "r", )

            dataset_name = ( uploaded_file.name )

            source_path = None

            dataset_size_mb = ( uploaded_file.size / (1024 ** 2) )

        # ====================================================
        # ZENODO
        # ====================================================

        elif source == "Zenodo":

            zenodo_url = zenodo_url.strip()

            if not zenodo_url:
                st.warning( "Please enter a Zenodo URL." )
                st.stop()
            hdf = open_zenodo_hdf5( zenodo_url )

            # `hdf.filename` is the actual local path 
            dataset_name = ( Path(hdf.filename).name or "Zenodo dataset" )
            source_path = zenodo_url

            try:
                dataset_size_mb = ( Path(hdf.filename).stat().st_size / (1024 ** 2) )
            except OSError:
                dataset_size_mb = None

        dataset_identity = ( identify_dataset(hdf) or "Custom HDF5 dataset" )

        # ====================================================
        # DATASET IDENTIFIER (for index/cache isolation)
        # ====================================================

        dataset_id = compute_dataset_id( source, source_path=source_path, dataset_name=dataset_name, dataset_size_mb=dataset_size_mb, )

        # ====================================================
        # STORE DATASET INFORMATION
        # ====================================================

        st.session_state[ "hdf" ] = hdf

        st.session_state[ "dataset_name" ] = dataset_name

        st.session_state[ "dataset_identity" ] = dataset_identity

        st.session_state[ "dataset_id" ] = dataset_id

        st.session_state[ "dataset_size_mb" ] = dataset_size_mb

        st.session_state[ "dataset_source" ] = source

        st.session_state[ "dataset_source_path" ] = source_path

        # ====================================================
        # RESET INDEXES
        # ====================================================

        st.session_state.pop( "opt_index", None, )

        st.session_state.pop( "geometry_index", None, )

        st.session_state[ "analysis_indexes_ready" ] = False

        # ====================================================
        # PREPARE INDEXES
        # ====================================================

        with st.spinner( "Preparing dataset for analysis..." ):

            index_result = ( prepare_dataset_indexes( hdf, source_path=source_path, dataset_id=dataset_id, ) )

            st.session_state[ "opt_index" ] = index_result["opt_df"]

            st.session_state[ "geometry_index" ] = index_result["geometry_df"]

            st.session_state[ "analysis_indexes_ready" ] = True

        # ====================================================
        # SUCCESS
        # ====================================================

        st.success( f"✓ Dataset successfully loaded: " f"**{dataset_name}**" )

    except Exception as e:

        st.error( f"Could not load the dataset: {e}" )

# ============================================================
# DATASET INFORMATION
# ============================================================

if st.session_state.get( "analysis_indexes_ready", False, ):

    # FILE -- the filename the user actually supplied (Local path /
    # uploaded file / Zenodo download).
    dataset_name = st.session_state.get( "dataset_name", "Current dataset", )

    # DATASET -- the scientific dataset the file was identified as,
    # never assumed equal to the filename above.
    dataset_identity = st.session_state.get( "dataset_identity", "Custom HDF5 dataset", )

    dataset_size_mb = st.session_state.get( "dataset_size_mb", None, )

    dataset_source = st.session_state.get( "dataset_source", "Local", )

    source_label = { "Local": "Local file", "Upload": "Uploaded file", "Zenodo": "Zenodo", }.get( dataset_source, dataset_source, )

    c1, c2, c3, c4 = st.columns(4)

    # ========================================================
    # FILE
    # ========================================================

    with c1:
        st.caption( "FILE" )
        st.write( f"**{dataset_name}**" )

    # ========================================================
    # DATASET
    # ========================================================

    with c2:
        st.caption( "DATASET" )
        st.write( f"**{dataset_identity}**" )

    # ========================================================
    # SIZE
    # ========================================================

    with c3:
        st.caption( "SIZE" )
        if dataset_size_mb is not None:
            st.write( f"**{dataset_size_mb:,.1f} MB**" )
        else:
            st.write( "**—**" )

    # ========================================================
    # SOURCE
    # ========================================================

    with c4:
        st.caption( "SOURCE" )
        st.write( f"**{source_label}**" )

# ============================================================
# DATASET ALREADY LOADED
# ============================================================

if st.session_state.get( "analysis_indexes_ready", False, ):

    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    if not load_button:

        st.success( f"✓ Dataset loaded: " f"**{st.session_state.get('dataset_name', 'Current dataset')}**" )

    # ========================================================
    # EXPLORE DIVIDER
    # ========================================================

    st.markdown( """ <div style=" width: 100%; height: 2px; background: var(--clr-accent); border-radius: 2px; margin: 1.25rem 0 1.75rem 0; "></div> """, unsafe_allow_html=True, )
# ============================================================
# EXPLORATION DASHBOARD
# ============================================================

st.markdown(
    """
    <div style=" margin: 1.8rem 0 1.4rem 0; ">
    <div style=" font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #0891B2; margin-bottom: 0.35rem; ">
        Explore
    </div>

    <div style=" font-size: 2rem; font-weight: 750; color: #111827; line-height: 1.15; margin-bottom: 0.4rem; ">
        What would you like to explore?
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Explore your dataset, molecular properties, chemical space and machine-learning models.

c1, c2, c3, c4 = st.columns(4, gap="medium")

# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

with c1:
    with st.container(border=True, height=215):

        st.markdown(
            """
            <div style=" display:flex; align-items:center; gap:0.75rem; margin-bottom:0.85rem; ">

            <div style=" width:44px; height:44px; display:flex; align-items:center; justify-content:center; background:#ECFEFF; border-radius:10px; font-size:1.35rem; flex-shrink:0; ">
                📂
            </div>

            <div>
            <div style=" font-size:1.02rem; font-weight:700; color:#111827; line-height:1.2; ">
                    Dataset Overview
                </div>

            <div style=" font-size:0.72rem; color:#0891B2; font-weight:600; margin-top:0.2rem; ">
                    DATA EXPLORATION
                </div>
            </div>

            </div>

            <div style=" height:1px; background:#E5E7EB; margin:0.2rem 0 0.85rem 0; "></div>

            <div style=" font-size:0.86rem; color:#6B7280; line-height:1.55; min-height:87px; ">
                Compositional profiles, conformer and molecular distribution patterns, and core statistical characteristics of the dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.page_link( "pages/1_Dataset_Overview.py", label="Explore dataset →", width="stretch", )

# ============================================================
# 2. PROPERTY SPACE ANALYSIS
# ============================================================

with c2:
    with st.container(border=True, height=215):

        st.markdown(
            """
            <div style=" display:flex; align-items:center; gap:0.75rem; margin-bottom:0.85rem; ">

            <div style=" width:44px; height:44px; display:flex; align-items:center; justify-content:center; background:#F5F3FF; border-radius:10px; font-size:1.35rem; flex-shrink:0; ">
                    ⚛️
            </div>

            <div>
            <div style=" font-size:1.02rem; font-weight:700; color:#111827; line-height:1.2; ">
                    Property Space Analysis
                    </div>

            <div style=" font-size:0.72rem; color:#7C3AED; font-weight:600; margin-top:0.2rem; ">
                    MOLECULAR PROPERTIES
                </div>
            </div>

            </div>

            <div style=" height:1px; background:#E5E7EB; margin:0.2rem 0 0.85rem 0; "></div>

            <div style=" font-size:0.86rem; color:#6B7280; line-height:1.55; min-height:87px; ">
                Combines quantum‑property correlation analysis with interactive 2D/3D molecular visualization and atom‑level details.
            
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.page_link( "pages/2_Property_Space_Analysis.py", label="Explore properties →", width="stretch", )

# ============================================================
# 3. MOLECULAR CLUSTERING
# ============================================================

with c3:
    with st.container(border=True, height=215):

        st.markdown(
            """
            <div style=" display:flex; align-items:center; gap:0.75rem; margin-bottom:0.85rem; ">

            <div style=" width:44px; height:44px; display:flex; align-items:center; justify-content:center; background:#EFF6FF; border-radius:10px; font-size:1.35rem; flex-shrink:0; ">
                🧭
            </div>

            <div>
            <div style=" font-size:1.02rem; font-weight:700; color:#111827; line-height:1.2; ">
                Molecular Clustering
                </div>

            <div style=" font-size:0.72rem; color:#2563EB; font-weight:600; margin-top:0.2rem; ">
                    CLUSTER ANALYSIS
                </div>
            </div>

            </div>

            <div style=" height:1px; background:#E5E7EB; margin:0.2rem 0 0.85rem 0; "></div>

            <div style=" font-size:0.86rem; color:#6B7280; line-height:1.55; min-height:87px; ">
                Maps chemical space with PCA and t‑SNE, fitting independent GMMs on each 2D projection to identify molecular clusters.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.page_link( "pages/3_Molecular_Clustering.py", label="Explore clusters →", width="stretch", )

# ============================================================
# 4. MACHINE LEARNING
# ============================================================

with c4:
    with st.container(border=True, height=215):

        st.markdown( """ 
            <div style=" display:flex; align-items:center; gap:0.75rem; margin-bottom:0.85rem; ">

            <div style=" width:44px; height:44px; display:flex; align-items:center; justify-content:center; background:#FFF7ED; border-radius:10px; font-size:1.35rem; flex-shrink:0;
            ">
            🤖
            </div>

            <div>
            <div style=" font-size:1.02rem; font-weight:700; color:#111827; line-height:1.2; ">
                Machine Learning
                </div>

            <div style=" font-size:0.72rem; color:#EA580C; font-weight:600; margin-top:0.2rem; ">
                PREDICTIVE MODELING
                </div>
                </div>

            </div>
            <div style=" height:1px; background:#E5E7EB; margin:0.2rem 0 0.85rem 0; "></div>

            <div style=" font-size:0.86rem; color:#6B7280; line-height:1.55; min-height:87px; ">
                Evaluates the trained QUED‑XGBoost model and deploys it for dipole‑moment prediction on new molecular inputs.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.page_link( "pages/4_Machine_Learning.py", label="Explore machine learning →", width="stretch", )
