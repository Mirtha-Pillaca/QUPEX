import plotly.graph_objects as go

from modules.plot_style import scientific_layout


def _format_count(value):
    """Format large counts using K/M/B suffixes."""

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{int(value):,}"


def plot_size_distribution(
    all_data,
    without_H_data,
    means,
    title,
    record_label,
    mode,
    dark_mode=False,
    bar_color=None,
    show_mean=True,
    show_bar_counts=True,
):
    """
    Plot the molecular size distribution.

    Parameters
    ----------
    all_data : pandas.DataFrame or None
        Distribution including hydrogen atoms.

    without_H_data : pandas.DataFrame or None
        Distribution excluding hydrogen atoms.

    means : dict
        Mean atom counts. Expected keys:
        - "all"
        - "without_H"

    title : str
        Figure title.

    record_label : str
        Label used in the mean annotation.

    mode : str
        Either "All atoms" or "Without H".

    dark_mode : bool, default=False
        Use the application dark plotting style when True.

    bar_color : str or None, default=None
        Bar and mean-line color.

    show_mean : bool, default=True
        Show the mean line and legend entry.

    show_bar_counts : bool, default=True
        Show counts above the bars.

    Returns
    -------
    plotly.graph_objects.Figure
        Configured Plotly figure.
    """

    fig = go.Figure()

    # ========================================================
    # Select data according to mode
    # ========================================================

    if mode == "All atoms":

        data = all_data
        mean_key = "all"
        atom_label = "atoms"
        bar_name = "All atoms"

    elif mode == "Without H":

        data = without_H_data
        mean_key = "without_H"
        atom_label = "heavy atoms"
        bar_name = "Without H"

    else:

        raise ValueError( f"Unsupported size-distribution mode: {mode!r}. Expected 'All atoms' or 'Without H'." )

    # ========================================================
    # Distribution
    # ========================================================

    if data is not None and not data.empty:

        fig.add_trace(
            go.Bar(
                x=data["Atoms"],
                y=data["Count"],

                name=bar_name,
                showlegend=False,

                opacity=0.78,

                marker=dict( color=bar_color, ),

                # Keep count data in the trace.
                # Only control whether the labels are visible.
                text=data["Count"].map(_format_count),
                texttemplate="%{text}",
                
                textposition=( "outside" if show_bar_counts else "none" ),

                textfont=dict( family="Arial", size=12, color="#000000", ),

                cliponaxis=False,

                hovertemplate=( f"<b>%{{x}} {atom_label}</b><br>" f"Counts: %{{y:,}}" "<extra></extra>" ),
            )
        )

        # ====================================================
        # Mean line
        # ====================================================

        if show_mean and mean_key in means:

            mean_value = means[mean_key]

            max_count = int(data["Count"].max())

            fig.add_trace(
                go.Scatter(
                    x=[ mean_value, mean_value, ],

                    y=[ 0, max_count * 1.20, ],

                    mode="lines",

                    name=( f"Mean = {mean_value:.2f} " f"{atom_label}/{record_label}" ),

                    line=dict( dash="dot", width=2, color=bar_color, ),

                    hovertemplate=( f"Mean: {mean_value:.2f} " f"{atom_label}/{record_label}" "<extra></extra>" ),

                    showlegend=True,
                )
            )

    # ========================================================
    # Scientific layout
    # ========================================================

    scientific_layout(
        fig,
        height=460,
        show_legend=( show_mean and data is not None and not data.empty and mean_key in means ),
        dark_mode=dark_mode,
    )

    # ========================================================
    # Figure-specific layout
    # ========================================================

    fig.update_layout(

        title=dict( text=title, x=0.02, xanchor="left", ),

        xaxis_title="Total Number of Atoms",

        yaxis_title="Counts",

        barmode="overlay",

        bargap=0.15,

        hovermode="x unified",

        legend=dict( orientation="h", x=0.02, xanchor="left", y=0.98, yanchor="top", bgcolor="rgba(0,0,0,0)", borderwidth=0, ),

        margin=dict( l=70, r=30, t=80, b=65, ),
    )

    # ========================================================
    # Independent Y-axis scaling
    # ========================================================

    if data is not None and not data.empty:

        max_count = int(data["Count"].max())

        if max_count > 0:

            fig.update_yaxes( range=[ 0, max_count * 1.20, ] )

    # ========================================================
    # Scientific axis formatting
    # ========================================================

    # ========================================================
    # Force publication-style black axis text
    # ========================================================

    fig.update_xaxes(
        title_font=dict( family="Arial", size=14, color="#000000", ),
        tickfont=dict( family="Arial", size=12, color="#000000", ),
        
    )

    fig.update_yaxes(
        title_font=dict( family="Arial", size=14, color="#000000", ),
        tickfont=dict( family="Arial", size=12, color="#000000", ),
    )

    return fig