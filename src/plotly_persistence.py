import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import warnings

# Configure Plotly for Jupyter notebooks
try:
    # Try to initialize Plotly for Jupyter
    pyo.init_notebook_mode(connected=True)
except:
    pass


def _format_handler(a):
    """
    Handle different input formats for persistence data.
    Returns: (formatted_persistence, input_type)
    """
    # Array case
    try:
        first_death_value = a[0][1]
        if isinstance(first_death_value, (np.floating, float, np.integer, int)):
            return [[0, x] for x in a], 1
    except (IndexError, TypeError):
        pass
    
    # Iterable of arrays
    try:
        pers = []
        fake_dim = 0
        for elt in a:
            first_death_value = elt[0][1]
            if not isinstance(first_death_value, (np.floating, float, np.integer, int)):
                raise TypeError("Should be a list of (birth,death)")
            pers.extend([fake_dim, x] for x in elt)
            fake_dim = fake_dim + 1
        return pers, 2
    except (TypeError, IndexError):
        pass
    
    # Nothing to be done otherwise
    return a, 0


def _limit_to_max_intervals(persistence, max_intervals, key):
    """Limit the number of intervals to display."""
    if max_intervals > 0 and max_intervals < len(persistence):
        warnings.warn(
            f"There are {len(persistence)} intervals given as input, "
            f"whereas max_intervals is set to {max_intervals}."
        )
        return sorted(persistence, key=key, reverse=True)[:max_intervals]
    else:
        return persistence


def _min_birth_max_death(persistence, band=0.0):
    """Find min birth and max death values from persistence data."""
    if not persistence:
        return 0.0, 1.0
    
    max_death = 0
    min_birth = persistence[0][1][0]
    
    for interval in persistence:
        if float(interval[1][1]) != float("inf"):
            if float(interval[1][1]) > max_death:
                max_death = float(interval[1][1])
        if float(interval[1][0]) > max_death:
            max_death = float(interval[1][0])
        if float(interval[1][0]) < min_birth:
            min_birth = float(interval[1][0])
    
    if band > 0.0:
        max_death += band
    
    if min_birth == max_death:
        max_death = max_death + 1.0
    
    return min_birth, max_death


def plot_persistence_diagram(
    persistence,
    alpha=0.6,
    band=0.0,
    max_intervals=1000000,
    inf_delta=0.1,
    legend=True,
    colormap=None,
    title="Persistence Diagram",
    width=800,
    height=800,
    greyblock=True,
    show_diagonal=True,
    show=True,
):
    """
    Plot interactive persistence diagram using Plotly.
    
    :param persistence: Persistence intervals. Can be:
        - List of (dimension, (birth, death)) tuples
        - Numpy array of shape (N, 2) for single dimension
        - List of numpy arrays for multiple dimensions
    :param alpha: Point opacity (0.0 to 1.0, default 0.6)
    :param band: Confidence band width (default 0.0)
    :param max_intervals: Maximum intervals to display (default 1000000)
    :param inf_delta: Infinity placement factor (default 0.1)
    :param legend: Show dimension legend (default True)
    :param colormap: Color sequence for dimensions (default None uses Plotly colors)
    :param title: Plot title (default "Persistence Diagram")
    :param width: Plot width in pixels (default 800)
    :param height: Plot height in pixels (default 800)
    :param greyblock: Show grey lower triangle (default True)
    :param show_diagonal: Show diagonal line (default True)
    :param show: Whether to display the plot immediately (default True)
    :returns: plotly.graph_objects.Figure
    """
    
    # Handle different input formats
    formatted_persistence = []
    input_has_dimensions = False
    
    if isinstance(persistence, np.ndarray):
        # Single dimension case: numpy array of shape (N, 2)
        if persistence.ndim == 2 and persistence.shape[1] == 2:
            formatted_persistence = [(0, (birth, death)) for birth, death in persistence]
        else:
            raise ValueError("Numpy array must have shape (N, 2)")
    elif isinstance(persistence, list):
        if len(persistence) == 0:
            formatted_persistence = []
        elif isinstance(persistence[0], tuple) and len(persistence[0]) == 2:
            # Check if it's (dimension, (birth, death)) format
            if isinstance(persistence[0][1], tuple):
                formatted_persistence = persistence
                input_has_dimensions = True
            else:
                # It's (birth, death) format
                formatted_persistence = [(0, interval) for interval in persistence]
        elif isinstance(persistence[0], (list, np.ndarray)):
            # Multiple dimensions as list of arrays
            for dim, intervals in enumerate(persistence):
                formatted_persistence.extend([(dim, (birth, death)) for birth, death in intervals])
            input_has_dimensions = True
        else:
            raise ValueError("Invalid persistence format")
    else:
        raise ValueError("Persistence must be numpy array or list")
    
    # Use GUDHI's format handler for additional compatibility
    try:
        formatted_persistence, input_type = _format_handler(formatted_persistence)
        if input_type > 0:
            input_has_dimensions = True
    except:
        pass
    
    # Limit number of intervals if specified
    if max_intervals > 0 and len(formatted_persistence) > max_intervals:
        formatted_persistence = _limit_to_max_intervals(
            formatted_persistence, 
            max_intervals, 
            key=lambda x: x[1][1] - x[1][0] if x[1][1] != float('inf') else float('inf')
        )
    
    # Find axis bounds
    if len(formatted_persistence) == 0:
        min_birth, max_death = 0.0, 1.0
    else:
        min_birth, max_death = _min_birth_max_death(formatted_persistence, band)
    
    # Handle infinity values
    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta
    axis_end = max_death + delta / 2
    axis_start = min_birth - delta
    
    # Create figure
    fig = go.Figure()
    
    # Set up colormap
    if colormap is None:
        # Use Plotly's default color sequence
        colormap = px.colors.qualitative.Set1
    
    # Add confidence band
    if band > 0.0:
        x_band = np.linspace(axis_start, infinity, 100)
        fig.add_trace(go.Scatter(
            x=x_band,
            y=x_band + band,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=x_band,
            y=x_band,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            fillcolor=f'rgba(255,0,0,{alpha})',
            name='Confidence Band',
            showlegend=True,
            hoverinfo='skip'
        ))
    
    # Add grey lower triangle
    if greyblock:
        fig.add_shape(
            type="path",
            path=f"M {axis_start},{axis_start} L {axis_end},{axis_start} L {axis_end},{axis_end} Z",
            fillcolor="lightgrey",
            #opacity=0.8,
            line_width=0,
            layer="below"
        )
    
    # Add diagonal line (birth = death)
    if show_diagonal:
        fig.add_trace(go.Scatter(
            x=[axis_start, axis_end],
            y=[axis_start, axis_end],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='y = x',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot points by dimension
    if formatted_persistence:
        dimensions = sorted(set(interval[0] for interval in formatted_persistence))
        
        for dim in dimensions:
            dim_intervals = [interval for interval in formatted_persistence if interval[0] == dim]
            
            x_vals = [interval[1][0] for interval in dim_intervals]  # births
            y_vals = [interval[1][1] if interval[1][1] != float('inf') else infinity 
                     for interval in dim_intervals]  # deaths
            
            # Create hover text
            hover_text = []
            for interval in dim_intervals:
                birth, death = interval[1]
                if death == float('inf'):
                    hover_text.append(f"Dim: {dim}<br>Birth: {birth:.3f}<br>Death: ∞<br>Persistence: ∞")
                else:
                    persistence_val = death - birth
                    hover_text.append(f"Dim: {dim}<br>Birth: {birth:.3f}<br>Death: {death:.3f}<br>Persistence: {persistence_val:.3f}")
            
            color = colormap[dim % len(colormap)]
                    # Choose marker symbol based on dimension
            if dim == 0:
                symbol = 'square'
            elif dim == 1:
                symbol = 'circle'
            elif dim == 2:
                symbol = 'triangle-up'
            else:
                symbol = 'star'
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=alpha,
                    symbol=symbol
                    #line=dict(width=1, color='black')
                ),
                name=f'Dimension {dim}',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=legend and (input_has_dimensions or len(dimensions) > 1)
            ))
        
        # Add infinity line if needed
        if any(interval[1][1] == float('inf') for interval in formatted_persistence):
            fig.add_trace(go.Scatter(
                x=[axis_start, axis_end],
                y=[infinity, infinity],
                mode='lines',
                line=dict(color='black', width=1),
                name='Infinity',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            title='Birth',
            range=[axis_start, axis_end],
            constrain='domain'
        ),
        yaxis=dict(
            title='Death',
            range=[axis_start, infinity + delta / 2],
            scaleanchor="x",
            scaleratio=1,
            constrain='domain'
        ),
        width=width,
        height=height,
        hovermode='closest',
        showlegend=legend
    )
    
    # Custom y-axis ticks for infinity
    if formatted_persistence and any(interval[1][1] == float('inf') for interval in formatted_persistence):
        # Get current tick values
        y_ticks = list(np.linspace(axis_start, max_death, 6))
        y_tick_labels = [f"{val:.3f}" for val in y_ticks]
        
        # # Add infinity tick
        # y_ticks.append(infinity)
        # y_tick_labels.append('∞')
        
        fig.update_yaxes(
            tickmode='array',
            tickvals=y_ticks,
            ticktext=y_tick_labels
        )

        # Add custom annotation for infinity symbol
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=-0.0,  # 2% to the left of the plot area
            y=infinity,
            text="∞",
            showarrow=False,
            font=dict(size=24, color='black'),
            xanchor="right",
            yanchor="middle"
        )
    
    # Show the plot if requested
    if show:
        fig.show()
    
    return fig