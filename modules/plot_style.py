from modules.theme import COLORS


# ============================================================
# SHARED COLOR SEQUENCES
# ============================================================
#
# Reuses the exact hexes from modules.theme so charts and chrome
# never drift out of sync. Use these instead of Plotly's default
# qualitative palette (px.colors.qualitative.Plotly) wherever a
# page picks colors for clusters, categories, or elements.
# ============================================================

QUALITATIVE_COLORS = COLORS["qualitative"]

SEQUENTIAL_COLORSCALE = COLORS["sequential"]

DIVERGING_COLORSCALE = COLORS["diverging"]

# Plotly-friendly [position, color] stops for continuous data
# (e.g. property heatmaps, density plots).
SEQUENTIAL_COLORSCALE_STOPS = COLORS["sequential_stops"]


def get_qualitative_color(index):
    """
    Deterministic color for the i-th category in a categorical
    series, extending QUALITATIVE_COLORS (8 entries) past its
    length instead of silently repeating an identical hex once a
    category count exceeds the base palette's size.

    Needed wherever a categorical count on a page isn't bounded by
    len(QUALITATIVE_COLORS) -- e.g. Chemical Space Analysis allows
    up to 10 GMM components, which would otherwise make components
    0 and 8 (and 1 and 9) visually indistinguishable. Cycle 0 is
    the base palette unchanged; each further cycle reuses the same
    hues, darkened, so colors stay recognizably related to the base
    palette (no new hues introduced) while remaining distinguishable
    from the first pass.
    """

    n_colors = len(QUALITATIVE_COLORS)

    cycle = index // n_colors

    base_hex = QUALITATIVE_COLORS[index % n_colors]

    if cycle == 0:
        return base_hex

    factor = max(0.35, 1 - 0.25 * cycle)

    base_hex = base_hex.lstrip("#")

    r, g, b = ( int(base_hex[0:2], 16), int(base_hex[2:4], 16), int(base_hex[4:6], 16), )

    r, g, b = ( max(0, min(255, int(channel * factor))) for channel in (r, g, b) )

    return f"#{r:02X}{g:02X}{b:02X}"


def scientific_layout( fig, height=450, show_legend=True, dark_mode=False, ):
    """
    Apply a consistent scientific-paper style to Plotly figures.

    dark_mode=False (default, unchanged from before) produces the
    classic white publication look -- best for exported/printed
    figures.

    dark_mode=True matches the app's dark theme, for figures meant
    to be viewed on-screen inside the dashboard (avoids a jarring
    white rectangle in the middle of a dark page).
    """

    if dark_mode:

        paper_bg = COLORS["bg_elevated"]
        plot_bg = COLORS["bg_elevated"]
        line_color = COLORS["border"]
        text_color = COLORS["text_secondary"]
        title_color = COLORS["text_primary"]
        legend_bg = COLORS["bg_elevated"]

    else:

        paper_bg = "white"
        plot_bg = "white"
        line_color = "black"
        text_color = "black"
        title_color = "black"
        legend_bg = "white"

    fig.update_layout(
        # ----------------------------------------------------
        # General
        # ----------------------------------------------------
        template="simple_white" if not dark_mode else "plotly_dark",
        height=height,

        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,

        font=dict( family="Arial", size=13, color=text_color, ),

        colorway=QUALITATIVE_COLORS,

        # ----------------------------------------------------
        # Margins
        # ----------------------------------------------------
        margin=dict( l=70, r=30, t=30, b=65, ),
        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------
        showlegend=show_legend,

        legend=dict( font=dict( family="Arial", size=12, color=text_color, ),
            bgcolor=legend_bg,
            bordercolor=line_color,
            borderwidth=1,
        ),
    )

    # ========================================================
    # X AXIS
    # ========================================================

    fig.update_xaxes(

        showgrid=False,
        mirror=True,

        showline=True,
        linecolor=line_color,
        linewidth=1.2,

        ticks="outside",
        tickcolor=line_color,
        tickwidth=1,
        ticklen=5,

        tickfont=dict( family="Arial", size=12, color=text_color, ),

        title_font=dict( family="Arial", size=14, color=title_color, ),

        zeroline=False,
    )

    # ========================================================
    # Y AXIS
    # ========================================================

    fig.update_yaxes(
        showgrid=False,
        showline=True,
        mirror=True,
        linecolor=line_color,
        linewidth=1.2,

        ticks="outside",
        tickcolor=line_color,
        tickwidth=1,
        ticklen=5,

        tickfont=dict( family="Arial", size=12, color=text_color, ),

        title_font=dict( family="Arial", size=14, color=title_color, ),

        zeroline=False,
    )

    # ============================================================
    # TRACE-SPECIFIC MARKER STYLING
    # ============================================================

    for trace in fig.data:

        if trace.type in { "scatter", "scattergl", "bar", "histogram", "box", "violin", }:

            try:

                trace.update( marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.35)", )

            except Exception:
                pass

    return fig


def use_webgl_for_large_scatter_total(fig, threshold=5000):
    """
    Swap Scatter traces to Scattergl when a figure's traces
    TOGETHER carry many points, even if no single trace does.

    use_webgl_for_large_scatter() checks each trace's own point
    count against `threshold`, which is the right test for one big
    cloud of points in a single trace. It misses a common case on
    categorically-colored scatters: a plot split into several
    per-category traces (one per cluster, say) where each
    individual trace stays under `threshold` but the plot as a
    whole renders just as many points -- and is just as slow in
    SVG. This sums points across all plain Scatter traces instead,
    and swaps all of them together once that combined total
    reaches `threshold`, so a plot renders consistently regardless
    of how many color categories split it into traces. Safe to call
    on any figure -- a small multi-trace figure, or one already
    below threshold, is left untouched.
    """

    scatter_traces = [ trace for trace in fig.data if trace.type == "scatter" ]

    total_points = sum( len(trace.x) if trace.x is not None else 0 for trace in scatter_traces )

    if total_points >= threshold:

        for trace in scatter_traces:
            trace.type = "scattergl"

    return fig
