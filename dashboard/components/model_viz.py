import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import time
from datetime import datetime, timedelta

def create_model_performance_graph(
    training_history: Dict[str, List[Dict[str, Any]]], 
    model_type: str = 'traffic_predictor'
) -> go.Figure:
    """
    Create a graph showing model performance over training iterations.
    
    Args:
        training_history: Dictionary with training history data
        model_type: Type of model ('traffic_predictor' or 'anomaly_detector')
        
    Returns:
        Plotly figure object
    """
    if not training_history or model_type not in training_history:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No training history available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=f"{model_type.replace('_', ' ').title()} Training History",
            xaxis_title="Training Iteration",
            yaxis_title="Metric Value",
            height=400
        )
        return fig
    
    # Extract history for the specified model
    history = training_history[model_type]
    
    if not history:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No training history available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=f"{model_type.replace('_', ' ').title()} Training History",
            xaxis_title="Training Iteration",
            yaxis_title="Metric Value",
            height=400
        )
        return fig
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(history)
    
    if model_type == 'traffic_predictor':
        # Plot loss metrics for traffic predictor
        fig = go.Figure()
        
        if 'train_loss' in df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['train_loss'],
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='blue')
            ))
        
        if 'val_loss' in df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['val_loss'],
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='orange')
            ))
        
        if 'test_loss' in df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['test_loss'],
                mode='lines+markers',
                name='Test Loss',
                line=dict(color='green')
            ))
        
        fig.update_layout(
            title='Traffic Predictor Training History',
            xaxis_title='Training Iteration',
            yaxis_title='Loss (MAE)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
    
    elif model_type == 'anomaly_detector':
        # For anomaly detector, plot sample count and contamination
        fig = go.Figure()
        
        if 'num_samples' in df.columns:
            fig.add_trace(go.Bar(
                x=list(range(1, len(df) + 1)),
                y=df['num_samples'],
                name='Number of Samples',
                marker_color='blue'
            ))
        
        # Create a secondary y-axis for contamination
        if 'contamination' in df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['contamination'],
                mode='lines+markers',
                name='Contamination',
                line=dict(color='red'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title='Anomaly Detector Training History',
            xaxis_title='Training Iteration',
            yaxis_title='Number of Samples',
            yaxis2=dict(
                title='Contamination',
                overlaying='y',
                side='right',
                range=[0, max(df['contamination']) * 1.2] if 'contamination' in df.columns else [0, 0.1]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
    
    return fig

def create_prediction_graph(
    historical_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
    feature: str = 'request_rate',
    title: str = 'Traffic Prediction'
) -> go.Figure:
    """
    Create a graph showing traffic prediction against historical data.
    
    Args:
        historical_data: DataFrame with historical traffic data
        prediction_data: DataFrame with predicted traffic data
        feature: Feature to plot
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if historical_data.empty or prediction_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for prediction visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=feature.replace('_', ' ').title(),
            height=400
        )
        return fig
    
    # Check if feature exists in both dataframes
    if feature not in historical_data.columns or feature not in prediction_data.columns:
        # Try with _norm suffix if the base feature isn't found
        norm_feature = f"{feature}_norm"
        if norm_feature in historical_data.columns and norm_feature in prediction_data.columns:
            feature = norm_feature
        else:
            # Find the first common feature that contains the feature name
            common_features = [col for col in historical_data.columns if feature in col and col in prediction_data.columns]
            if common_features:
                feature = common_features[0]
            else:
                # Create empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Feature '{feature}' not found in data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(
                    title=title,
                    xaxis_title="Time",
                    yaxis_title=feature.replace('_', ' ').title(),
                    height=400
                )
                return fig
    
    # Determine x-axis values (timestamps)
    historical_x = historical_data['timestamp'] if 'timestamp' in historical_data.columns else historical_data.index
    prediction_x = prediction_data['timestamp'] if 'timestamp' in prediction_data.columns else prediction_data.index
    
    # Convert timestamps to datetime if they're numeric
    if pd.api.types.is_numeric_dtype(historical_x):
        historical_x = pd.to_datetime(historical_x, unit='s')
    if pd.api.types.is_numeric_dtype(prediction_x):
        prediction_x = pd.to_datetime(prediction_x, unit='s')
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_x,
        y=historical_data[feature],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add prediction
    fig.add_trace(go.Scatter(
        x=prediction_x,
        y=prediction_data[feature],
        mode='lines',
        name='Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    # Add shaded area for prediction range
    if len(historical_x) > 0 and len(prediction_x) > 0:
        boundary_x = historical_x.iloc[-1] if hasattr(historical_x, 'iloc') else historical_x[-1]
        
        fig.add_vrect(
            x0=boundary_x,
            x1=prediction_x.iloc[-1] if hasattr(prediction_x, 'iloc') else prediction_x[-1],
            fillcolor="lightgray",
            opacity=0.3,
            layer="below",
            line_width=0,
            annotation_text="Prediction Range",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=feature.replace('_', ' ').title(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    return fig

def create_anomaly_detection_graph(
    data: pd.DataFrame,
    x_feature: str = 'timestamp',
    y_feature: str = 'request_rate',
    title: str = 'Anomaly Detection'
) -> go.Figure:
    """
    Create a graph highlighting anomalies in traffic data.
    
    Args:
        data: DataFrame with anomaly detection results
        x_feature: Feature for x-axis
        y_feature: Feature for y-axis
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for anomaly visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_feature.replace('_', ' ').title(),
            yaxis_title=y_feature.replace('_', ' ').title(),
            height=400
        )
        return fig
    
    # Check if features exist
    if x_feature not in data.columns or y_feature not in data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Features not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_feature.replace('_', ' ').title(),
            yaxis_title=y_feature.replace('_', ' ').title(),
            height=400
        )
        return fig
    
    # Check if anomaly column exists
    if 'is_anomaly' not in data.columns:
        # Just create a regular scatter plot
        fig = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            title=title
        )
        fig.update_layout(height=400)
        return fig
    
    # Convert timestamps to datetime if they're numeric
    x_values = data[x_feature]
    if x_feature == 'timestamp' and pd.api.types.is_numeric_dtype(x_values):
        x_values = pd.to_datetime(x_values, unit='s')
    
    # Split data into normal and anomalous
    normal_data = data[~data['is_anomaly']]
    anomaly_data = data[data['is_anomaly']]
    
    normal_x = normal_data[x_feature]
    anomaly_x = anomaly_data[x_feature]
    
    if x_feature == 'timestamp' and pd.api.types.is_numeric_dtype(normal_x):
        normal_x = pd.to_datetime(normal_x, unit='s')
    if x_feature == 'timestamp' and pd.api.types.is_numeric_dtype(anomaly_x):
        anomaly_x = pd.to_datetime(anomaly_x, unit='s')
    
    # Create figure
    fig = go.Figure()
    
    # Add normal data
    fig.add_trace(go.Scatter(
        x=normal_x,
        y=normal_data[y_feature],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    # Add anomalies
    fig.add_trace(go.Scatter(
        x=anomaly_x,
        y=anomaly_data[y_feature],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=12, symbol='x')
    ))
    
    # If anomaly_score is available, add size variation
    if 'anomaly_score' in data.columns and not anomaly_data.empty:
        anomaly_scores = anomaly_data['anomaly_score']
        
        # Normalize scores to a reasonable size range (8-20)
        if not anomaly_scores.empty and anomaly_scores.max() != anomaly_scores.min():
            min_score = anomaly_scores.min()
            max_score = anomaly_scores.max()
            normalized_sizes = ((anomaly_scores - min_score) / (max_score - min_score) * 12 + 8).tolist()
        else:
            normalized_sizes = [12] * len(anomaly_scores)
        
        fig.data[1].marker.size = normalized_sizes
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_feature.replace('_', ' ').title(),
        yaxis_title=y_feature.replace('_', ' ').title(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    return fig

def create_feature_importance_graph(
    feature_importances: Dict[str, float],
    title: str = 'Feature Importance',
    max_features: int = 10
) -> go.Figure:
    """
    Create a graph showing the importance of different features in the ML model.
    
    Args:
        feature_importances: Dictionary mapping feature names to importance scores
        title: Graph title
        max_features: Maximum number of features to display
        
    Returns:
        Plotly figure object
    """
    # Check for empty data
    if not feature_importances:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400
        )
        return fig
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Limit to max_features
    sorted_features = sorted_features[:max_features]
    
    # Extract names and values
    feature_names = [item[0].replace('_', ' ').title() for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=feature_names,
        x=importance_values,
        orientation='h',
        marker_color='royalblue'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, 50 * len(feature_names) + 150)  # Dynamic height based on feature count
    )
    
    return fig
