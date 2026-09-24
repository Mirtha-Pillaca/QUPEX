import streamlit as st
from modules.theme import render_footer

st.set_page_config( page_title="Quantum Property Space Explorer", page_icon="🧬", layout="wide", )

# =========================
# SIDEBAR
# =========================

st.markdown(
    """
    <style>
    [data-testid="stSidebarHeader"] {
        height: auto;
        flex-direction: column;
        align-items: center;
        padding: 1.25rem 0 0.5rem 0;
    }
    [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
        align-self: flex-end;
    }
    [data-testid="stSidebarLogo"] {
        display: block;
        width: 150px;
        max-width: 90%;
        height: auto;
        margin: 0 auto;
    }
    ul[data-testid="stSidebarNavItems"] > li:last-child {
        margin-top: 24.0rem;
        padding-top: 0.0rem;
        border-top: 1px solid rgba(203, 213, 225, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.logo("assets/logo_wo_background.png", size="large")

pages = [
    st.Page("home.py", title="Home", icon="🏠"),
    st.Page( "pages/1_Dataset_Overview.py", title="Dataset Overview", icon="📊", ),
    st.Page( "pages/2_Property_Space_Analysis.py", title="Property Space Analysis", icon="🧬", ),
    st.Page( "pages/3_Molecular_Clustering.py", title="Molecular Clustering", icon="🔬", ),
    st.Page( "pages/4_Machine_Learning.py", title="Machine Learning", icon="🤖", ),
    st.Page( "pages/5_Documentation.py", title="Documentation", icon="📚", ),
]
pg = st.navigation(pages)


pg.run()

render_footer()