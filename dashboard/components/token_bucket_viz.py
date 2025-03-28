import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import calendar

def create_token_bucket_gauge(
    current_tokens: float,
    max_tokens: Optional[float] = None,
    capacity: Optional[float] = None,
    threshold_red: float = 0.25,
    threshold_yellow: float = 0.5,
    title: str = 'Token Bucket Level',
    user_type: Optional[str] = None
) -> go.Figure:
    """
    Create a gauge chart showing the current token bucket level.
    
    Args:
        current_tokens: Current number of tokens in the bucket
        max_tokens: Maximum capacity of the token bucket (will be used if capacity is None)
        capacity: Alias for max_tokens (for backward compatibility)
        threshold_red: Threshold for red zone (percentage of max)
        threshold_yellow: Threshold for yellow zone (percentage of max)
        title: Chart title
        user_type: User type (e.g. "STD" or "PRM") to customize the appearance
        
    Returns:
        Plotly figure object
    """
    # Handle capacity/max_tokens
    if max_tokens is None:
        max_tokens = capacity if capacity is not None else 100
    
    # Calculate the percentage fill
    if max_tokens == 0:  # Avoid division by zero
        percentage = 0
    else:
        percentage = min(100, max(0, (current_tokens / max_tokens) * 100))
    
    # Determine colors based on user type
    if user_type:
        if user_type == "STD":
            bar_color = "royalblue"
        elif user_type == "PRM":
            bar_color = "orange"
        else:
            bar_color = "royalblue"
    else:
        bar_color = "royalblue"
    
    # Set title if user_type is provided
    if user_type and title == 'Token Bucket Level':
        title = f"{user_type} {title}"
    
    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        title={'text': title},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, threshold_red * 100], 'color': "tomato"},
                {'range': [threshold_red * 100, threshold_yellow * 100], 'color': "gold"},
                {'range': [threshold_yellow * 100, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': percentage
            }
        }
    ))
    
    # Add annotation showing actual token count
    fig.add_annotation(
        text=f"Tokens: {current_tokens:.1f} / {max_tokens:.1f}",
        x=0.5,
        y=0.25,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14)
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_token_consumption_graph(
    token_data: pd.DataFrame,
    time_column: str = 'timestamp',
    consumption_column: str = 'tokens_consumed',
    window_minutes: int = 60,
    title: str = 'Token Consumption Rate'
) -> go.Figure:
    """
    Create a graph showing token consumption over time.
    
    Args:
        token_data: DataFrame with token consumption data
        time_column: Column containing timestamps
        consumption_column: Column containing token consumption values
        window_minutes: Time window to display (in minutes)
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if token_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No token consumption data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Tokens Consumed",
            height=350
        )
        return fig
    
    # Make a copy to avoid modifying original
    data = token_data.copy()
    
    # Convert timestamp to datetime if needed
    if pd.api.types.is_numeric_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column], unit='s')
    
    # Filter to recent window
    if not data.empty:
        latest_time = data[time_column].max()
        start_time = latest_time - timedelta(minutes=window_minutes)
        data = data[data[time_column] >= start_time]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar for token consumption
    fig.add_trace(go.Bar(
        x=data[time_column],
        y=data[consumption_column],
        name='Tokens Consumed',
        marker_color='royalblue'
    ))
    
    # Add moving average if enough data points
    if len(data) >= 5:
        window_size = min(5, len(data) // 5)
        data['ma'] = data[consumption_column].rolling(window=window_size).mean()
        
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data['ma'],
            mode='lines',
            name=f'{window_size}-Point Moving Avg',
            line=dict(color='red', width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Tokens Consumed",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_token_refill_graph(
    token_data: pd.DataFrame,
    time_column: str = 'timestamp',
    refill_column: str = 'tokens_refilled',
    window_minutes: int = 60,
    title: str = 'Token Refill Rate'
) -> go.Figure:
    """
    Create a graph showing token refill over time.
    
    Args:
        token_data: DataFrame with token refill data
        time_column: Column containing timestamps
        refill_column: Column containing token refill values
        window_minutes: Time window to display (in minutes)
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if token_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No token refill data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Tokens Refilled",
            height=350
        )
        return fig
    
    # Make a copy to avoid modifying original
    data = token_data.copy()
    
    # Convert timestamp to datetime if needed
    if pd.api.types.is_numeric_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column], unit='s')
    
    # Filter to recent window
    if not data.empty:
        latest_time = data[time_column].max()
        start_time = latest_time - timedelta(minutes=window_minutes)
        data = data[data[time_column] >= start_time]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar for token refill
    fig.add_trace(go.Bar(
        x=data[time_column],
        y=data[refill_column],
        name='Tokens Refilled',
        marker_color='green'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Tokens Refilled",
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_token_level_history(
    token_data: Optional[pd.DataFrame] = None,
    bucket_history: Optional[pd.DataFrame] = None,
    time_column: str = 'timestamp',
    level_column: str = 'token_level',
    capacity_column: Optional[str] = 'token_capacity',
    window_minutes: int = 60,
    title: str = 'Token Bucket History',
    user_type: Optional[str] = None
) -> go.Figure:
    """
    Create a graph showing token bucket level over time.
    
    Args:
        token_data: DataFrame with token level data (deprecated, use bucket_history instead)
        bucket_history: DataFrame with token bucket history data
        time_column: Column containing timestamps
        level_column: Column containing token level values (defaults to 'token_level' but will use 'tokens' if available)
        capacity_column: Column containing bucket capacity (if None, won't show)
        window_minutes: Time window to display (in minutes)
        title: Chart title
        user_type: User type (e.g. "STD" or "PRM") to customize the appearance
        
    Returns:
        Plotly figure object
    """
    # Use bucket_history if provided, otherwise use token_data
    data = bucket_history if bucket_history is not None else token_data
    
    # Check for empty data
    if data is None or data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No token level data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Token Level",
            height=350
        )
        return fig
    
    # Make a copy to avoid modifying original
    data = data.copy()
    
    # Check for 'tokens' column if level_column isn't found
    if level_column not in data.columns and 'tokens' in data.columns:
        level_column = 'tokens'
    
    # Check for 'capacity' column if capacity_column isn't found
    if capacity_column not in data.columns and 'capacity' in data.columns:
        capacity_column = 'capacity'
    
    # Convert timestamp to datetime if needed
    if pd.api.types.is_numeric_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column], unit='s')
    
    # Filter to recent window
    if not data.empty:
        latest_time = data[time_column].max()
        start_time = latest_time - timedelta(minutes=window_minutes)
        data = data[data[time_column] >= start_time]
    
    # Set color based on user type
    if user_type:
        if user_type == "STD":
            line_color = "royalblue"
        elif user_type == "PRM":
            line_color = "orange"
        else:
            line_color = "royalblue"
    else:
        line_color = "royalblue"
    
    # Set title if user_type is provided
    if user_type and title == 'Token Bucket History':
        title = f"{user_type} {title}"
    
    # Create figure
    fig = go.Figure()
    
    # Add area for token level
    fig.add_trace(go.Scatter(
        x=data[time_column],
        y=data[level_column],
        fill='tozeroy',
        mode='lines',
        name='Token Level',
        line=dict(color=line_color, width=2)
    ))
    
    # Add capacity reference line if available
    if capacity_column and capacity_column in data.columns:
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data[capacity_column],
            mode='lines',
            name='Capacity',
            line=dict(color='red', width=1, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Token Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_token_comparison(
    std_token_level: Optional[float] = None,
    std_capacity: Optional[float] = None,
    prm_token_level: Optional[float] = None,
    prm_capacity: Optional[float] = None,
    std_data: Optional[Dict[str, Any]] = None,
    prm_data: Optional[Dict[str, Any]] = None,
    title: str = 'Token Bucket Comparison'
) -> go.Figure:
    """
    Create a bar chart comparing standard and premium token bucket levels.
    
    Args:
        std_token_level: Current token level for standard users
        std_capacity: Maximum capacity for standard users
        prm_token_level: Current token level for premium users
        prm_capacity: Maximum capacity for premium users
        std_data: Dictionary with standard user data (contains 'tokens' and 'capacity' keys)
        prm_data: Dictionary with premium user data (contains 'tokens' and 'capacity' keys)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Extract data from dictionaries if provided
    if std_data:
        std_token_level = std_data.get('tokens', 0)
        std_capacity = std_data.get('capacity', 1)
    
    if prm_data:
        prm_token_level = prm_data.get('tokens', 0)
        prm_capacity = prm_data.get('capacity', 1)
    
    # Ensure we have valid values
    std_token_level = float(std_token_level or 0)
    std_capacity = float(std_capacity or 1)
    prm_token_level = float(prm_token_level or 0)
    prm_capacity = float(prm_capacity or 1)
    
    # Calculate percentages
    std_pct = (std_token_level / std_capacity * 100) if std_capacity > 0 else 0
    prm_pct = (prm_token_level / prm_capacity * 100) if prm_capacity > 0 else 0
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for both user types
    fig.add_trace(go.Bar(
        x=['Standard', 'Premium'],
        y=[std_token_level, prm_token_level],
        name='Current Tokens',
        marker_color=['royalblue', 'orange']
    ))
    
    fig.add_trace(go.Bar(
        x=['Standard', 'Premium'],
        y=[std_capacity - std_token_level, prm_capacity - prm_token_level],
        name='Available Space',
        marker_color='lightgray'
    ))
    
    # Add annotations for percentages
    fig.add_annotation(
        x='Standard',
        y=std_capacity / 2,
        text=f"{std_pct:.1f}%",
        showarrow=False,
        font=dict(color='white', size=14)
    )
    
    fig.add_annotation(
        x='Premium',
        y=prm_capacity / 2,
        text=f"{prm_pct:.1f}%",
        showarrow=False,
        font=dict(color='white', size=14)
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title="Tokens",
        barmode='stack',
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_traffic_overview(
    traffic_data: pd.DataFrame,
    time_column: str = 'timestamp',
    value_column: str = 'request_count',
    window_minutes: int = 60,
    title: str = 'API Traffic Overview'
) -> go.Figure:
    """
    Create a graph showing overall API traffic over time.
    
    Args:
        traffic_data: DataFrame with traffic data
        time_column: Column containing timestamps
        value_column: Column containing traffic values
        window_minutes: Time window to display (in minutes)
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if traffic_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No traffic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Requests",
            height=350
        )
        return fig
    
    # Make a copy to avoid modifying original
    data = traffic_data.copy()
    
    # Check if columns exist
    if time_column not in data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Time column '{time_column}' not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Requests",
            height=350
        )
        return fig
    
    if value_column not in data.columns:
        # Try to use a different column if available
        possible_columns = ['request_count', 'requests', 'count', 'response_time_count']
        found = False
        for col in possible_columns:
            if col in data.columns:
                value_column = col
                found = True
                break
        
        if not found:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Value column not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Requests",
                height=350
            )
            return fig
    
    # Convert timestamp to datetime if needed
    if pd.api.types.is_numeric_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column], unit='s')
    
    # Filter to recent window
    if not data.empty:
        latest_time = data[time_column].max()
        start_time = latest_time - timedelta(minutes=window_minutes)
        data = data[data[time_column] >= start_time]
    
    # Create figure
    fig = go.Figure()
    
    # Add line for request count
    fig.add_trace(go.Scatter(
        x=data[time_column],
        y=data[value_column],
        mode='lines+markers',
        name='Request Count',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add moving average if enough data points
    if len(data) >= 5:
        window_size = min(5, len(data) // 5)
        data['ma'] = data[value_column].rolling(window=window_size).mean()
        
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data['ma'],
            mode='lines',
            name=f'{window_size}-Point Moving Avg',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Add error rate if available
    if 'error_count' in data.columns and value_column in data.columns:
        # Calculate error rate
        data['error_rate'] = (data['error_count'] / data[value_column] * 100).fillna(0)
        
        # Add error rate on secondary y-axis
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data['error_rate'],
            mode='lines',
            name='Error Rate (%)',
            line=dict(color='tomato', width=2),
            yaxis='y2'
        ))
    
    # Add rate limited count if available
    if 'rate_limited_count' in data.columns:
        fig.add_trace(go.Bar(
            x=data[time_column],
            y=data['rate_limited_count'],
            name='Rate Limited',
            marker_color='orange',
            opacity=0.7
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Requests",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    # Add secondary y-axis for error rate if used
    if 'error_rate' in data.columns:
        fig.update_layout(
            yaxis2=dict(
                title="Error Rate (%)",
                overlaying="y",
                side="right",
                range=[0, max(data['error_rate']) * 1.2 if not data['error_rate'].empty else 10]
            )
        )
    
    return fig

def create_endpoint_distribution(
    traffic_data: pd.DataFrame,
    endpoint_column: str = 'endpoint',
    count_column: Optional[str] = None,
    max_endpoints: int = 10,
    title: str = 'Endpoint Distribution'
) -> go.Figure:
    """
    Create a pie or bar chart showing the distribution of traffic across endpoints.
    
    Args:
        traffic_data: DataFrame with traffic data
        endpoint_column: Column containing endpoint information
        count_column: Column containing count values (if None, will count rows)
        max_endpoints: Maximum number of endpoints to show individually (rest grouped as 'Other')
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if traffic_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No traffic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            height=350
        )
        return fig
    
    # Check if endpoint column exists
    if endpoint_column not in traffic_data.columns:
        # Try to find a column that might contain endpoint information
        possible_columns = ['endpoint', 'route', 'path', 'url']
        found = False
        for col in possible_columns:
            if col in traffic_data.columns:
                endpoint_column = col
                found = True
                break
        
        if not found:
            fig = go.Figure()
            fig.add_annotation(
                text="Endpoint information not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=title,
                height=350
            )
            return fig
    
    # Group by endpoint
    if count_column and count_column in traffic_data.columns:
        # Use provided count column
        endpoint_counts = traffic_data.groupby(endpoint_column)[count_column].sum()
    else:
        # Count rows
        endpoint_counts = traffic_data.groupby(endpoint_column).size()
    
    # Sort and limit to max_endpoints
    endpoint_counts = endpoint_counts.sort_values(ascending=False)
    
    if len(endpoint_counts) > max_endpoints:
        # Group small endpoints as 'Other'
        top_endpoints = endpoint_counts.iloc[:max_endpoints]
        other_count = endpoint_counts.iloc[max_endpoints:].sum()
        
        # Create new series with 'Other' category
        endpoint_counts = pd.concat([top_endpoints, pd.Series({'Other': other_count})])
    
    # Clean endpoint names (shorter display)
    cleaned_endpoints = []
    for endpoint in endpoint_counts.index:
        if endpoint == 'Other':
            cleaned_endpoints.append('Other')
        else:
            # Remove common prefixes and shorten
            clean = str(endpoint).replace('/api/', '').replace('/resources/', '')
            if len(clean) > 25:
                clean = clean[:22] + '...'
            cleaned_endpoints.append(clean)
    
    # Create pie chart
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=cleaned_endpoints,
        values=endpoint_counts.values,
        hole=0.4,
        textinfo='label+percent',
        marker=dict(
            colors=px.colors.qualitative.Plotly[:len(endpoint_counts)]
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_user_type_distribution(
    traffic_data: pd.DataFrame,
    user_type_column: str = 'user_type',
    count_column: Optional[str] = None,
    title: str = 'User Type Distribution'
) -> go.Figure:
    """
    Create a pie chart showing the distribution of traffic across user types.
    
    Args:
        traffic_data: DataFrame with traffic data
        user_type_column: Column containing user type information
        count_column: Column containing count values (if None, will count rows)
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if traffic_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No traffic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            height=300
        )
        return fig
    
    # Check if user type column exists
    if user_type_column not in traffic_data.columns:
        # Try to find a column that might contain user type information
        possible_columns = ['user_type', 'usertype', 'type', 'account_type']
        found = False
        for col in possible_columns:
            if col in traffic_data.columns:
                user_type_column = col
                found = True
                break
        
        if not found:
            fig = go.Figure()
            fig.add_annotation(
                text="User type information not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=title,
                height=300
            )
            return fig
    
    # Group by user type
    if count_column and count_column in traffic_data.columns:
        # Use provided count column
        user_type_counts = traffic_data.groupby(user_type_column)[count_column].sum()
    else:
        # Count rows
        user_type_counts = traffic_data.groupby(user_type_column).size()
    
    # Create colors dictionary
    colors_dict = {
        'STD': 'royalblue',
        'PRM': 'orange',
    }
    
    # Map colors to user types, defaulting to Plotly colors for unknown types
    colors = [colors_dict.get(ut, px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]) 
              for i, ut in enumerate(user_type_counts.index)]
    
    # Create pie chart
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=user_type_counts.index,
        values=user_type_counts.values,
        hole=0.4,
        textinfo='label+percent',
        marker=dict(colors=colors)
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_traffic_heatmap(
    traffic_data: pd.DataFrame,
    time_column: str = 'timestamp',
    value_column: str = 'request_count',
    days_to_show: int = 7,
    title: str = 'Traffic Heatmap by Hour and Day'
) -> go.Figure:
    """
    Create a heatmap showing traffic patterns by hour and day.
    
    Args:
        traffic_data: DataFrame with traffic data
        time_column: Column containing timestamps
        value_column: Column containing traffic values
        days_to_show: Number of past days to include
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if traffic_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No traffic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            height=400
        )
        return fig
    
    # Check if required columns exist
    if time_column not in traffic_data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Time column '{time_column}' not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            height=400
        )
        return fig
    
    if value_column not in traffic_data.columns:
        # Try to use a different column if available
        possible_columns = ['request_count', 'requests', 'count', 'response_time_count']
        found = False
        for col in possible_columns:
            if col in traffic_data.columns:
                value_column = col
                found = True
                break
        
        if not found:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Value column not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=title,
                height=400
            )
            return fig
    
    # Make a copy to avoid modifying original
    data = traffic_data.copy()
    
    # Convert timestamp to datetime if needed
    if pd.api.types.is_numeric_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column], unit='s')
    
    # Filter to recent days
    if not data.empty:
        latest_time = data[time_column].max()
        start_time = latest_time - timedelta(days=days_to_show)
        data = data[data[time_column] >= start_time]
    
    # Extract hour and day of week
    data['hour'] = data[time_column].dt.hour
    data['day_of_week'] = data[time_column].dt.dayofweek
    
    # Group by hour and day, summing the value column
    heatmap_data = data.groupby(['day_of_week', 'hour'])[value_column].sum().unstack(fill_value=0)
    
    # Ensure all hours are represented
    for hour in range(24):
        if hour not in heatmap_data.columns:
            heatmap_data[hour] = 0
    
    # Ensure all days are represented
    for day in range(7):
        if day not in heatmap_data.index:
            heatmap_data.loc[day] = [0] * 24
    
    # Sort columns and rows
    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)
    
    # Get day names
    day_names = [calendar.day_name[day] for day in heatmap_data.index]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=day_names,
        colorscale='Viridis',
        colorbar=dict(title='Request Count')
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_anomaly_timeline(
    anomaly_data: pd.DataFrame,
    time_column: str = 'timestamp',
    feature_column: str = 'request_rate',
    anomaly_column: str = 'is_anomaly',
    window_minutes: int = 120,
    title: str = 'Traffic Anomaly Timeline'
) -> go.Figure:
    """
    Create a timeline showing traffic with highlighted anomalies.
    
    Args:
        anomaly_data: DataFrame with anomaly detection results
        time_column: Column containing timestamps
        feature_column: Column containing the feature to plot
        anomaly_column: Column indicating anomalies (True/False)
        window_minutes: Time window to display (in minutes)
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if anomaly_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No anomaly data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=feature_column.replace('_', ' ').title(),
            height=350
        )
        return fig
    
    # Check if required columns exist
    for col in [time_column, feature_column, anomaly_column]:
        if col not in anomaly_data.columns:
            # For anomaly column, also check 'is_score_anomaly' as an alternative
            if col == anomaly_column and 'is_score_anomaly' in anomaly_data.columns:
                anomaly_column = 'is_score_anomaly'
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Required column '{col}' not found in data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(
                    title=title,
                    xaxis_title="Time",
                    yaxis_title=feature_column.replace('_', ' ').title(),
                    height=350
                )
                return fig
    
    # Make a copy to avoid modifying original
    data = anomaly_data.copy()
    
    # Convert timestamp to datetime if needed
    if pd.api.types.is_numeric_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column], unit='s')
    
    # Filter to recent window
    if not data.empty:
        latest_time = data[time_column].max()
        start_time = latest_time - timedelta(minutes=window_minutes)
        data = data[data[time_column] >= start_time]
    
    # Split data into normal and anomalous points
    normal_data = data[~data[anomaly_column]]
    anomalous_data = data[data[anomaly_column]]
    
    # Create figure
    fig = go.Figure()
    
    # Add line for normal data
    fig.add_trace(go.Scatter(
        x=data[time_column],
        y=data[feature_column],
        mode='lines',
        name='Traffic',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add markers for anomalies
    if not anomalous_data.empty:
        fig.add_trace(go.Scatter(
            x=anomalous_data[time_column],
            y=anomalous_data[feature_column],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    # Add anomaly score if available
    if 'anomaly_score' in data.columns:
        # Normalize anomaly scores to a reasonable range
        min_score = data['anomaly_score'].min()
        max_score = data['anomaly_score'].max()
        
        if min_score != max_score:
            norm_scores = (data['anomaly_score'] - min_score) / (max_score - min_score) * 100
            
            fig.add_trace(go.Scatter(
                x=data[time_column],
                y=norm_scores,
                mode='lines',
                name='Anomaly Score',
                line=dict(color='orange', width=1, dash='dot'),
                yaxis='y2'
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=feature_column.replace('_', ' ').title(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    # Add secondary y-axis for anomaly score if used
    if 'anomaly_score' in data.columns:
        fig.update_layout(
            yaxis2=dict(
                title="Anomaly Score",
                overlaying="y",
                side="right",
                range=[0, 100]
            )
        )
    
    return fig

def create_response_time_graph(
    traffic_data: pd.DataFrame,
    time_column: str = 'timestamp',
    response_time_column: str = 'avg_response_time',
    window_minutes: int = 60,
    title: str = 'API Response Time'
) -> go.Figure:
    """
    Create a graph showing API response times over time.
    
    Args:
        traffic_data: DataFrame with traffic data
        time_column: Column containing timestamps
        response_time_column: Column containing response time values
        window_minutes: Time window to display (in minutes)
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if traffic_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No traffic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Response Time (s)",
            height=300
        )
        return fig
    
    # Check if required columns exist
    if time_column not in traffic_data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Time column '{time_column}' not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Response Time (s)",
            height=300
        )
        return fig
    
    # Check for response time column
    rt_column = response_time_column
    if rt_column not in traffic_data.columns:
        # Try alternative column names
        alternatives = ['response_time_mean', 'mean_response_time', 'response_time']
        for alt in alternatives:
            if alt in traffic_data.columns:
                rt_column = alt
                break
        
        if rt_column not in traffic_data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Response time data not found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Response Time (s)",
                height=300
            )
            return fig
    
    # Make a copy to avoid modifying original
    data = traffic_data.copy()
    
    # Convert timestamp to datetime if needed
    if pd.api.types.is_numeric_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column], unit='s')
    
    # Filter to recent window
    if not data.empty:
        latest_time = data[time_column].max()
        start_time = latest_time - timedelta(minutes=window_minutes)
        data = data[data[time_column] >= start_time]
    
    # Create figure
    fig = go.Figure()
    
    # Add response time line
    fig.add_trace(go.Scatter(
        x=data[time_column],
        y=data[rt_column],
        mode='lines+markers',
        name='Avg Response Time',
        line=dict(color='green', width=2)
    ))
    
    # Add min and max lines if available
    if 'response_time_min' in data.columns and 'response_time_max' in data.columns:
        # Add range
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data['response_time_min'],
            mode='lines',
            name='Min',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data['response_time_max'],
            mode='lines',
            name='Max',
            fill='tonexty',
            fillcolor='rgba(0, 128, 0, 0.1)',
            line=dict(width=0),
            showlegend=False
        ))
    
    # Add moving average
    if len(data) >= 5:
        window_size = min(5, len(data) // 5)
        data['ma'] = data[rt_column].rolling(window=window_size).mean()
        
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data['ma'],
            mode='lines',
            name=f'{window_size}-Point Moving Avg',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Response Time (s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig
