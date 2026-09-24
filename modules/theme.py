
import base64
from pathlib import Path
import streamlit as st


# ============================================================
# COLOR TOKENS
# ============================================================
#
# Global light scientific theme.
#
# The palette is intentionally kept in one place so that
# plot_style.py and other modules can reuse the same colors.
# ============================================================

COLORS = {

    # --------------------------------------------------------
    # SURFACES
    # --------------------------------------------------------

    "bg": "#FFFFFF",
    "bg_elevated": "#F8FAFC",
    "sidebar": "#F1F5F9",

    "border": "#CBD5E1",
    "border_soft": "#E2E8F0",

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    "text_primary": "#0F172A",
    "text_secondary": "#334155",
    "text_plot": "#0F0F0F",
    "text_muted": "#64748B",
    "text_primary_selected_button": "#F8FAFC",

    # --------------------------------------------------------
    # PRIMARY ACCENT
    # --------------------------------------------------------

    "accent": "#0891B2",
    "accent_strong": "#0E7490",
    "accent_hover": "#155E75",

    # --------------------------------------------------------
    # SECONDARY ACCENT
    # --------------------------------------------------------

    "accent2": "#3BBCD0",
    "accent2_strong": "#24BEE1",
    "accent2_hover": "#3CE0E9",
    "accent_bg": "#ECFEFF",
    "accent2_bg": "#EBFFFE",
    "accent3": "#7C3AED",
    "accent3_strong": "#6D28D9",
    "accent3_hover": "#5B21B6",
    "accent3_bg": "#F5F3FF",

    # --------------------------------------------------------
    # SEMANTIC
    # --------------------------------------------------------

    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
    "info": "#0284C7",

    # --------------------------------------------------------
    # QUALITATIVE CHART PALETTE
    # --------------------------------------------------------

    "qualitative": ["#0891B2", "#F59E0B", "#7C3AED", "#E11D48", "#16A34A", "#2563EB", "#CA8A04", "#DB2777"],

    # --------------------------------------------------------
    # CONTINUOUS / DIVERGING
    # --------------------------------------------------------

    "sequential": ["#F8FAFC", "#67E8F9", "#0891B2", "#164E63"],

    "sequential2": ["#443A83", "#26828E", "#35B779", "#8FD744", "#FDE725"],

    "diverging": ["#155E75", "#67E8F9", "#FFFFFF", "#FBBF24", "#B45309"],

    # --------------------------------------------------------
    # PLOTLY CONTINUOUS COLORSCALE
    # --------------------------------------------------------

    "sequential_stops": [[0.0, "#67E8F9"], [0.35, "#22D3EE"], [0.7, "#0891B2"], [1.0, "#164E63"]],

    "sequential_orange_stops": [[0.0, "#FFFBEB"], [0.35, "#FCD34D"], [0.7, "#F59E0B"], [1.0, "#B45309"]],
}

def page_title(title: str):
    st.markdown(
        f"""
        <div class="app-page-title">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )

def apply_global_theme():

    c = COLORS

    background_image = get_background_image()

    if background_image:

        background_css = f"""
        background-image:
            linear-gradient(
                rgba(255, 255, 255, 0.75),
                rgba(255, 255, 255, 0.75)
            ),
            url("data:image/png;base64,{background_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        """

    else:

        background_css = """
        background-image: none;
        """


    st.markdown(
        f"""
        <style>

        /* =====================================================
           ROOT VARIABLES
           ===================================================== */

        :root {{
            --clr-bg: {c["bg"]};
            --clr-bg-elevated: {c["bg_elevated"]};
            --clr-sidebar: {c["sidebar"]};

            --clr-border: {c["border"]};
            --clr-border-soft: {c["border_soft"]};

            --clr-text-primary: {c["text_primary"]};
            --clr-text-secondary: {c["text_secondary"]};
            --clr-text-muted: {c["text_muted"]};

            --clr-accent: {c["accent"]};
            --clr-accent-strong: {c["accent_strong"]};
            --clr-accent-hover: {c["accent_hover"]};
            --clr-accent-bg: {c["accent_bg"]};

            --clr-accent2: {c["accent2"]};
            --clr-accent2-strong: {c["accent2_strong"]};
            --clr-accent2-hover: {c["accent2_hover"]};

            --clr-success: {c["success"]};
            --clr-warning: {c["warning"]};
            --clr-danger: {c["danger"]};
            --clr-info: {c["info"]};
        }}


        /* =====================================================
           APPLICATION BACKGROUND
           ===================================================== */

        .stApp {{
            background-color: #FFFFFF;

            {background_css}

            color: var(--clr-text-primary);
        }}

        .stAppViewContainer {{
            background: transparent !important;
        }}

        .main {{
            background: transparent !important;
        }}


        /* =====================================================
           MAIN CONTENT
           ===================================================== */

        .stAppViewContainer .main .block-container,
        [data-testid="stMainBlockContainer"] {{
            padding-top: 0.00rem !important;
            padding-bottom: 2rem !important;
            background: transparent !important;
        }}

        .stAppViewContainer .main .block-container > div,
        [data-testid="stMainBlockContainer"] > div {{
            margin-top: 0 !important;
            padding-top: 0 !important;
        }}

        /* =====================================================
        COMPACT STREAMLIT HEADER
        ===================================================== */

        [data-testid="stHeader"] {{
            background: transparent !important;
            height: 2rem !important;
        }}

        [data-testid="stHeader"] > div {{
            background: transparent !important;
       }}

        /* =====================================================
           SIDEBAR
           ===================================================== */

        section[data-testid="stSidebar"] {{
            background: rgba(241, 245, 249, 0.97) !important;
            border-right: 1px solid var(--clr-border);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--clr-text-secondary);
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: var(--clr-text-primary) !important;
        }}


        /* =====================================================
           SIDEBAR NAVIGATION
           ===================================================== */

        div[data-testid="stSidebarNav"] a {{
            color: var(--clr-text-secondary) !important;
            border-radius: 8px;
        }}

        div[data-testid="stSidebarNav"] a:hover {{
            background: #FFFFFF !important;
            color: var(--clr-text-primary) !important;
        }}

        div[data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(8, 145, 178, 0.10) !important;
            color: var(--clr-accent-strong) !important;
            font-weight: 900;
        }}


        /* =====================================================
           HEADINGS
           ===================================================== */

        .app-page-title {{
            margin: 0 !important;
            padding: 0 !important;

            font-size: 2.0rem !important;
            line-height: 1.3 !important;
            font-weight: 650 !important;
            letter-spacing: -0.015em;
        }}
        h1 {{
            color: var(text-primary) !important;
            font-size: 2.4rem !important;
            line-height: 1.15 !important;
            font-weight: 650 !important;
            letter-spacing: -0.015em;
            margin-top: 0 !important;
            margin-bottom: 0.1rem !important;
        }}

        h2 {{
            color: var(--clr-text-primary) !important;
            font-weight: 500;
            
        }}

        h3 {{
            color: var(--clr-text-secondary) !important;
            font-weight: 500;
        }}


        /* =====================================================
           BODY TEXT
           ===================================================== */

        p,
        label,
        span {{
            color: var(--clr-text-secondary);
        }}

        .stCaption,
        [data-testid="stCaptionContainer"] {{
            color: var(--clr-text-muted) !important;
        }}


        /* =====================================================
           TABS
           ===================================================== */

        div[data-testid="stTabs"] {{
            width: 100%;
        }}

        div[data-testid="stTabs"] [role="tablist"] {{
            gap: 8px !important;
            border-bottom: 1px solid var(--clr-border);
        }}

        div[data-testid="stTabs"] [role="tab"] {{
            min-height: 56px !important;
            padding: 14px 26px !important;

            font-size: 18px !important;
            font-weight: 650 !important;

            color: var(--clr-text-muted) !important;
        }}

        div[data-testid="stTabs"] [role="tab"] span {{
            font-size: 18px !important;
            font-weight: 650 !important;
        }}

        div[data-testid="stTabs"] [role="tab"] p {{
            font-size: 18px !important;
            font-weight: 650 !important;
            margin: 0 !important;
        }}

        div[data-testid="stTabs"]
        [role="tab"][aria-selected="true"] {{
            background-color: var(--clr-accent-bg) !important;
            color: var(--clr-accent-strong) !important;
            border-radius: 8px 8px 0 0;
        }}

        div[data-testid="stTabs"]
        [data-baseweb="tab-highlight"] {{
            background-color: var(--clr-accent) !important;
            height: 5px !important;
        }}

        div[data-testid="stTabs"]
        [data-baseweb="tab-border"] {{
            background-color: var(--clr-border) !important;
        }}


        /* =====================================================
        METRICS
        ===================================================== */

        div[data-testid="stMetric"] {{
            background: var(--clr-bg-elevated) !important;

            border: 1px solid var(--clr-border-soft) !important;
            border-radius: 10px !important;

            padding: 8px 14px !important;
            min-height: 70px !important;

            box-shadow:
                0 1px 3px rgba(15, 23, 42, 0.04),
                0 4px 10px rgba(15, 23, 42, 0.035);

            transition:
                transform 0.16s ease,
                box-shadow 0.16s ease,
                border-color 0.16s ease,
                background-color 0.16s ease;
        }}


        /* =====================================================
        HOVER
        ===================================================== */

        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);

            border-color: var(--clr-accent) !important;

            box-shadow:
                0 3px 7px rgba(15, 23, 42, 0.06),
                0 8px 18px rgba(8, 145, 178, 0.10);
        }}


        /* =====================================================
        LABEL
        ===================================================== */

        div[data-testid="stMetric"] label {{
            color: var(--clr-text-muted) !important;

            font-size: 0.76rem !important;
            font-weight: 600 !important;

            line-height: 1.25 !important;
            letter-spacing: 0.01em;
        }}


        /* =====================================================
        VALUE
        ===================================================== */

        div[data-testid="stMetric"]
        [data-testid="stMetricValue"] {{
            color: var(--clr-text-primary) !important;

            font-size: 1.50rem !important;
            font-weight: 700 !important;

            line-height: 1.15 !important;
            letter-spacing: -0.035em;

            margin-top: 3px;
        }}


        /* =====================================================
        DELTA
        ===================================================== */

        div[data-testid="stMetric"]
        [data-testid="stMetricDelta"] {{
            font-size: 0.72rem !important;
            font-weight: 600 !important;

            margin-top: 3px;
        }}


        /* =====================================================
           BUTTONS
           ===================================================== */

        .stButton > button {{
            background-color: var(--clr-accent-strong) !important;

            color: white !important;

            border: 1px solid var(--clr-accent-strong) !important;

            border-radius: 8px;

            font-weight: 600;
        }}

        .stButton > button:hover {{
            background-color: var(--clr-accent-hover) !important;
            border-color: var(--clr-accent-hover) !important;
        }}

        .stButton > button[kind="primary"] {{
            background-color: var(--clr-accent2-strong) !important;

            border-color: var(--clr-accent2-strong) !important;

            color: white !important;

            font-weight: 650;
        }}

        .stButton > button[kind="primary"]:hover {{
            background-color: var(--clr-accent2-hover) !important;
            border-color: var(--clr-accent2-hover) !important;
        }}

        /* =====================================================
        SELECT / MULTISELECT
        ===================================================== */

        div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;

            border-color: var(--clr-border) !important;

            color: var(--clr-text-primary) !important;
        }}

        div[data-baseweb="select"] input {{
            color: var(--clr-text-primary) !important;
        }}

        div[data-baseweb="tag"] {{
            background-color: #ECFEFF !important;
            color: #0F172A !important;
        }}

        div[data-baseweb="tag"] span {{
            color: #0F172A !important;
        }}

        div[data-baseweb="tag"] svg {{
            color: #0F172A !important;
            fill: #0F172A !important;
        }}

        /* =====================================================
           DROPDOWN
           ===================================================== */

        div[data-baseweb="popover"] {{
            background: #FFFFFF !important;
        }}

        div[data-baseweb="menu"] {{
            background: #FFFFFF !important;
        }}

        li[role="option"] {{
            color: var(--clr-text-primary) !important;
        }}

        li[role="option"]:hover {{
            background: var(--clr-bg-elevated) !important;
        }}


        /* =====================================================
           INPUTS
           ===================================================== */

        input,
        textarea {{
            background-color: #FFFFFF !important;
            color: var(--clr-text-primary) !important;
            border-color: var(--clr-border) !important;
        }}

        input::placeholder,
        textarea::placeholder {{
            color: var(--clr-text-muted) !important;
        }}


        /* =====================================================
           SLIDER
           ===================================================== */

        div[data-testid="stSlider"] div[role="slider"] {{
            background-color: var(--clr-accent) !important;
        }}

        div[data-baseweb="slider"] > div > div {{
            background: var(--clr-accent-strong) !important;
        }}


        /* =====================================================
           CHECKBOX / RADIO
           ===================================================== */

        input[type="checkbox"]:checked,
        input[type="radio"]:checked {{
            accent-color: var(--clr-accent);
        }}


        /* =====================================================
           ALERTS
           ===================================================== */

        div[data-testid="stAlertContentSuccess"],
        div[data-baseweb="notification"][kind="success"] {{
            background-color: none !important;
            color: none !important;
            box-shadow: none !important;
        }}

        div[data-testid="stAlertContentInfo"],
        div[data-baseweb="notification"][kind="info"] {{
            background-color: none !important;
            box-shadow: none !important;
            
        }}

        div[data-testid="stAlertContentWarning"],
        div[data-baseweb="notification"][kind="warning"] {{
            background-color: none !important;
            color: none !important;
            box-shadow: none !important;
        }}

        div[data-testid="stAlertContentError"],
        div[data-baseweb="notification"][kind="negative"] {{
            background-color: rgba(220, 38, 38, 0.08) !important;
            color: none !important;
            box-shadow: none !important;
        }}


        /* =====================================================
           EXPANDER
           ===================================================== */

        details {{
            background: rgba(248, 250, 252, 0.94);

            border: 1px solid var(--clr-border);

            border-radius: 9px;
        }}

        summary {{
            color: var(--clr-text-primary) !important;
            font-weight: 600;
        }}


        /* =====================================================
           DATAFRAME
           ===================================================== */

        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--clr-border);
            border-radius: 8px;
            overflow: hidden;
            background: #FFFFFF;
        }}


        /* =====================================================
           FILE UPLOADER
           ===================================================== */

        section[data-testid="stFileUploaderDropzone"] {{
            background: rgba(248, 250, 252, 0.94) !important;

            border: 1px dashed var(--clr-border) !important;

            border-radius: 8px;
        }}


        /* =====================================================
           LINKS
           ===================================================== */

        a {{
            color: var(--clr-accent-strong) !important;
        }}

        a:hover {{
            color: var(--clr-accent-hover) !important;
        }}


        /* =====================================================
           DIVIDERS
           ===================================================== */

        hr {{
            border-color: var(--clr-border-soft) !important;
        }}


        /* =====================================================
           SCROLLBAR
           ===================================================== */

        ::-webkit-scrollbar {{
            width: 9px;
            height: 9px;
        }}

        ::-webkit-scrollbar-track {{
            background: #F8FAFC;
        }}

        ::-webkit-scrollbar-thumb {{
            background: #CBD5E1;
            border-radius: 6px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: #94A3B8;
        }}


        /* =====================================================
           PROPERTY TABLE
           ===================================================== */

        .table-wrapper {{
            width: 100%;

            border: 1px solid var(--clr-border);

            border-radius: 10px;

            overflow: hidden;

            margin: 12px 0 20px 0;
        }}

        .table {{
            width: 100%;

            border-collapse: collapse;

            background: rgba(255, 255, 255, 0.97);

            font-size: 0.92rem;
        }}

        .table th {{
            background: var(--clr-sidebar);

            color: var(--clr-text-primary);

            text-align: left;

            font-weight: 650;

            padding: 11px 14px;

            border-bottom: 1px solid var(--clr-border);
        }}

        .table td {{
            color: var(--clr-text-secondary);

            padding: 9px 14px;

            border-bottom: 1px solid var(--clr-border-soft);
        }}

        .table tr:last-child td {{
            border-bottom: none;
        }}

        .table tr:hover td {{
            background: rgba(8, 145, 178, 0.04);
        }}

        .table td:first-child {{
            width: 18%;
        }}

        .table td:last-child {{
            width: 15%;
            color: var(--clr-text-muted);
        }}

        .table td code {{
            color: var(--clr-accent-strong);

            background: rgba(8, 145, 178, 0.08);

            padding: 2px 6px;

            border-radius: 4px;

            font-family: monospace;
        }}

        /* =====================================================
            FOOTER
            ===================================================== */
        
        .app-footer {{
            margin-top: 3rem;
            padding: 1.5rem 0 1rem 0;
            border-top: 1px solid var(--clr-border-soft);
            text-align: center;
            color: var(--clr-text-muted);
            font-size: 0.78rem;
            line-height: 1.6;
        }}

        .app-footer-title {{
            color: var(--clr-text-secondary);
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
        }}

        .app-footer-meta {{
            color: var(--clr-text-muted);
            margin-bottom: 0.15rem;
        }}

        .app-footer-links {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.45rem;
            flex-wrap: wrap;
        }}

        .app-footer-links a {{
            color: var(--clr-text-muted) !important;
            text-decoration: none !important;
        }}

        .app-footer-links a:hover {{
            color: var(--clr-accent) !important;
            text-decoration: underline !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PROPERTY TABLE
# ============================================================

def property_table_html(df):

    rows = "".join( f""" <tr> <td><code>{row["Property"]}</code></td> <td>{row["Meaning"]}</td> <td>{row["Units"]}</td> </tr> """ for _, row in df.iterrows() )

    return f""" <div class="table-wrapper"> <table class="table"> <thead> <tr> <th>Property</th> <th>Meaning</th> <th>Units</th> </tr> </thead> <tbody> {rows} </tbody> </table> </div> """


def get_background_image():

    image_path = ( Path(__file__).resolve().parent.parent / "assets" / "background.png" )

    if not image_path.exists():
        return None

    with open(image_path, "rb") as image_file: encoded = base64.b64encode( image_file.read() ).decode()

    return encoded



# ============================================================
# FIGURE SECTION HEADERS
# ============================================================

FIGURE_HEADER_STYLE = {
    "margin_top": "1.8rem",
    "margin_bottom": "1rem",
    "padding": "0.65rem 0",
    "font_size": "1.25rem",
    "font_weight": "700",
    "color": COLORS["text_primary"],
}


def section_header(title):
    """
    Render a consistent section header for figure groups.
    """

    st.markdown(
        f"""
        <div style="
            margin-top: {FIGURE_HEADER_STYLE["margin_top"]};
            margin-bottom: {FIGURE_HEADER_STYLE["margin_bottom"]};
            padding: {FIGURE_HEADER_STYLE["padding"]};
            background: transparent;
        ">
            <div style="
                font-size: {FIGURE_HEADER_STYLE["font_size"]};
                font-weight: {FIGURE_HEADER_STYLE["font_weight"]};
                color: {FIGURE_HEADER_STYLE["color"]};
                line-height: 1.25;
            ">
                {title}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header_title(
    title,
    subtitle,
    color=None,
    background=None,
):

    color = color or COLORS["accent"]
    background = background or COLORS["accent_bg"]

    st.markdown(
        f"""
        <div style="
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding: 0.7rem 1rem;
            background: {background};
            border-left: 4px solid {color};
            border-radius: 6px;
        ">
        <div style="
            font-size: 1.05rem;
            font-weight: 700;
            color: {COLORS["text_primary"]};
        ">
            {title}
        </div>

        <div style="
            font-size: 0.85rem;
            color: {COLORS["text_muted"]};
            margin-top: 0.15rem;
        ">
            {subtitle}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        f"""
        <div class="app-footer">
        <div class="app-footer-title">
            QUPEX: Quantum Property EXplorer
        </div>

        <div class="app-footer-meta">
            © 2026 · Scientific visualization and machine-learning application
        </div>

        <div class="app-footer-links">
            <a href="" target="_blank">
                Web application
            </a>
            <span>·</span>
            <a href="" target="_blank">
                Reference
            </a>
            <span>·</span>
            <a href="https://github.com/Mirtha-Pillaca/QUPEX" target="_blank">
                Source code
            </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )