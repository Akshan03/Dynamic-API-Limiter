import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import json
import asyncio
import aiohttp
import sys
import os
from datetime import datetime, timedelta
import threading
import requests
from typing import Dict, List, Optional, Any, Union, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import visualization components
from dashboard.components.model_viz import (
    create_model_performance_graph,
    create_prediction_graph,
    create_anomaly_detection_graph,
    create_feature_importance_graph
)

from dashboard.components.token_bucket_viz import (
    create_token_bucket_gauge,
    create_token_consumption_graph,
    create_token_refill_graph,
    create_token_level_history,
    create_token_comparison
)

from dashboard.components.traffic_viz import (
    create_traffic_overview,
    create_endpoint_distribution,
    create_user_type_distribution,
    create_traffic_heatmap,
    create_anomaly_timeline,
    create_response_time_graph
)

# Import traffic simulator
from utils.simulator import TrafficSimulator, simulate_traffic

# Import ML components
from ml.trainer import ModelTrainer
from ml.data.collector import TrafficCollector, get_recent_traffic_data
from ml.models.traffic_predictor import predict_traffic

# Define API URL
API_URL = "http://localhost:8000"
API_PREFIX = "/api"  # Define the API prefix

# Global state
global_state = {
    "simulation_running": False,
    "std_user_id": None,
    "prm_user_id": None,
    "std_bucket": {"capacity": 20, "tokens": 15, "refill_rate": 5},
    "prm_bucket": {"capacity": 60, "tokens": 45, "refill_rate": 15},
    "traffic_data": pd.DataFrame(),
    "token_history": pd.DataFrame(),
    "last_update": time.time(),
    "anomaly_data": pd.DataFrame(),
    "prediction_data": pd.DataFrame(),
    "historical_data": pd.DataFrame(),
    "training_history": {"traffic_predictor": [], "anomaly_detector": []},
    "feature_importances": {}
}

# Function to get users from API
def get_api_users():
    try:
        # Try the root endpoint with admin auth
        headers = {
            "X-User-ID": "admin_user",
            "X-User-Type": "PRM"
        }
        response = requests.get(f"{API_URL}/", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            std_users = data.get("test_users", {}).get("standard", [])
            prm_users = data.get("test_users", {}).get("premium", [])
            
            if std_users and prm_users:
                return std_users, prm_users
        
        # Fallback to hardcoded test users
        return ['std_user_1', 'std_user_2', 'std_user_3', 'std_user_4', 'std_user_5'], ['prm_user_1', 'prm_user_2', 'prm_user_3']
    except Exception as e:
        print(f"Error connecting to API: {e}")
        # Return hardcoded test users as fallback
        return ['std_user_1', 'std_user_2', 'std_user_3', 'std_user_4', 'std_user_5'], ['prm_user_1', 'prm_user_2', 'prm_user_3']

# Function to fetch token bucket status
def get_token_bucket_status(user_id):
    try:
        # Get user type from ID (prefix std_ or prm_)
        user_type = "STD" if user_id.startswith("std_") else "PRM"
        
        # Create authentication headers
        headers = {
            "X-User-ID": user_id,
            "X-User-Type": user_type
        }
        
        # Make API request with proper path
        response = requests.get(
            f"{API_URL}{API_PREFIX}/api/rate-limit-status",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if "limit_info" in data and data["limit_info"]:
                return {
                    "tokens": data["limit_info"]["tokens_remaining"],
                    "capacity": data["limit_info"]["tokens_capacity"],
                    "refill_rate": data["limit_info"].get("refill_rate", 5 if user_type == "STD" else 15)
                }
        
        # If API call failed, return default values
        if user_type == "STD":
            # Standard user default bucket (75% full)
            return {"tokens": 15, "capacity": 20, "refill_rate": 5}
        else:
            # Premium user default bucket (75% full)
            return {"tokens": 45, "capacity": 60, "refill_rate": 15}
    except Exception as e:
        print(f"Error fetching token bucket status: {e}")
        # Return default values based on user type
        if user_type == "STD":
            return {"tokens": 15, "capacity": 20, "refill_rate": 5}
        else:
            return {"tokens": 45, "capacity": 60, "refill_rate": 15}

# Function to fetch recent traffic data - using simulated data since endpoint doesn't exist
async def fetch_recent_traffic():
    try:
        # Since the /traffic/recent endpoint doesn't exist in the API, generate mock data
        now = datetime.now()
        intervals = 30
        timestamps = [now - timedelta(minutes=i) for i in range(intervals)]
        timestamps.reverse()
        
        # Generate simulated traffic data
        traffic_data = []
        
        # Track user types for distribution
        user_type_counts = {
            "STD": 0,
            "PRM": 0,
            "ANOM": 0
        }
        
        # Use the simulation logs to create realistic user type distribution
        # Based on the last simulation we can see:
        # STD users: std_user_0-4 sent 394 requests (std_user_0: 125, std_user_1: 197, std_user_2: 47, std_user_3: 12, std_user_4: 13)
        # PRM users: prm_user_0-1 sent 98 requests (prm_user_0: 49, prm_user_1: 49)
        # Anomalous users: anom_user_0 sent 371 requests
        user_type_counts["STD"] = 394
        user_type_counts["PRM"] = 98
        user_type_counts["ANOM"] = 371
        
        for ts in timestamps:
            # Simulate traffic with some randomness
            hour = ts.hour
            minute = ts.minute
            base_requests = 50 + 30 * np.sin(hour/3) + 10 * np.sin(minute/10)
            
            # Add randomness
            requests = max(1, int(base_requests + np.random.normal(0, 10)))
            rate_limited = int(requests * (0.1 + 0.05 * np.sin(hour/2)))
            errors = int(requests * 0.02)
            
            # Calculate distribution of requests by user type for this interval
            total = sum(user_type_counts.values())
            std_requests = int(requests * (user_type_counts["STD"] / total))
            prm_requests = int(requests * (user_type_counts["PRM"] / total))
            anom_requests = requests - std_requests - prm_requests
            
            traffic_data.append({
                "timestamp": ts,  # Store as actual datetime object, not string
                "request_count": requests,
                "rate_limited_count": rate_limited,
                "error_count": errors,
                "avg_response_time": 0.2 + 0.1 * np.sin(hour/4) + 0.01 * np.random.random(),
                "std_request_count": std_requests,
                "prm_request_count": prm_requests,
                "anom_request_count": anom_requests
            })
        
        return pd.DataFrame(traffic_data)
    except Exception as e:
        print(f"Error generating traffic data: {e}")
        return pd.DataFrame()

# Function to get user type distribution from traffic data
def get_user_type_distribution(traffic_data):
    """Extract user type distribution from traffic data"""
    if traffic_data.empty:
        return None
    
    # Check if traffic data contains user type information
    if "std_request_count" not in traffic_data.columns:
        return None
    
    # Sum up requests by user type
    std_total = traffic_data["std_request_count"].sum()
    prm_total = traffic_data["prm_request_count"].sum()
    anom_total = traffic_data["anom_request_count"].sum()
    
    # Create distribution dictionary
    distribution = {
        "Standard": std_total,
        "Premium": prm_total,
        "Anomalous": anom_total
    }
    
    return distribution

# Function to run traffic simulation
async def run_traffic_simulation(duration, std_users, prm_users, anomalous_users):
    try:
        # Use the correct API path with API_PREFIX
        results = await simulate_traffic(
            api_base_url=f"{API_URL}{API_PREFIX}/api",
            duration=duration,
            std_users=std_users,
            prm_users=prm_users,
            anomalous_users=anomalous_users,
            save_results=True
        )
        return results
    except Exception as e:
        st.error(f"Error running traffic simulation: {e}")
        return None

# Function to get ML model training history and predictions - simulated since endpoint doesn't exist
async def get_ml_data():
    try:
        # Create simulated ML data
        # Generate timestamps for historical data
        now = datetime.now()
        intervals = 60
        timestamps = [now - timedelta(minutes=i) for i in range(intervals)]
        timestamps.reverse()
        
        # Create historical traffic pattern with seasonality
        historical_data = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            base_traffic = 100 + 50 * np.sin(hour/6)  # Daily pattern
            if hour >= 9 and hour <= 17:  # Business hours
                base_traffic *= 1.5
            
            # Add weekly pattern (weekdays higher)
            if ts.weekday() < 5:  # Weekday
                base_traffic *= 1.2
            
            # Add noise
            traffic = base_traffic + np.random.normal(0, 10)
            
            historical_data.append({
                "timestamp": ts,  # Store as actual datetime object
                "request_rate": max(0, traffic)
            })
        
        # Generate predictions for the next 10 intervals
        prediction_data = []
        for i in range(1, 11):
            next_ts = now + timedelta(minutes=i)
            hour = next_ts.hour
            base_traffic = 100 + 50 * np.sin(hour/6)
            
            if hour >= 9 and hour <= 17:
                base_traffic *= 1.5
            
            if next_ts.weekday() < 5:
                base_traffic *= 1.2
            
            prediction_data.append({
                "timestamp": next_ts,  # Store as actual datetime object
                "request_rate": max(0, base_traffic)
            })
        
        # Generate some anomalies in historical data
        anomaly_data = pd.DataFrame(historical_data)
        anomaly_indices = np.random.choice(len(anomaly_data), size=int(len(anomaly_data) * 0.05), replace=False)
        anomaly_data["is_anomaly"] = False
        for idx in anomaly_indices:
            anomaly_data.at[idx, "is_anomaly"] = True
            # Make anomalous points have higher request rates
            anomaly_data.at[idx, "request_rate"] *= 2.5
        
        # Simulated training history
        training_history = {
            "traffic_predictor": [
                {"epoch": i, "loss": 0.5 * (0.9 ** i), "val_loss": 0.6 * (0.85 ** i)}
                for i in range(1, 21)
            ],
            "anomaly_detector": [
                {"epoch": i, "accuracy": 0.5 + 0.45 * (1 - 0.9 ** i), 
                 "precision": 0.4 + 0.55 * (1 - 0.85 ** i),
                 "recall": 0.3 + 0.65 * (1 - 0.8 ** i)}
                for i in range(1, 16)
            ]
        }
        
        # Simulated feature importances
        feature_importances = {
            "request_rate": 0.35,
            "hour_of_day": 0.25,
            "day_of_week": 0.15,
            "rate_limited_ratio": 0.12,
            "error_rate": 0.08,
            "user_type_distribution": 0.05
        }
        
        return {
            "training_history": training_history,
            "feature_importances": feature_importances,
            "historical_data": pd.DataFrame(historical_data),
            "prediction_data": pd.DataFrame(prediction_data),
            "anomaly_data": anomaly_data
        }
    except Exception as e:
        print(f"Error generating ML data: {e}")
        return {
            "training_history": {},
            "feature_importances": {},
            "historical_data": pd.DataFrame(),
            "prediction_data": pd.DataFrame(),
            "anomaly_data": pd.DataFrame()
        }

# Update data in background
def update_data_thread():
    while True:
        # Run in async context
        async def update_data_async():
            global global_state
            
            # Update token bucket status for demo users
            if global_state["std_user_id"]:
                std_status = get_token_bucket_status(global_state["std_user_id"])
                if std_status:
                    global_state["std_bucket"].update(std_status)
            
            if global_state["prm_user_id"]:
                prm_status = get_token_bucket_status(global_state["prm_user_id"])
                if prm_status:
                    global_state["prm_bucket"].update(prm_status)
            
            # Fetch recent traffic data
            traffic_data = await fetch_recent_traffic()
            if not traffic_data.empty:
                global_state["traffic_data"] = traffic_data
            
            # Get ML data periodically (every 30 seconds)
            if time.time() - global_state["last_update"] > 30:
                ml_data = await get_ml_data()
                global_state["training_history"] = ml_data["training_history"]
                global_state["feature_importances"] = ml_data["feature_importances"]
                global_state["historical_data"] = ml_data["historical_data"]
                global_state["prediction_data"] = ml_data["prediction_data"]
                global_state["anomaly_data"] = ml_data["anomaly_data"]
                global_state["last_update"] = time.time()
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_data_async())
        loop.close()
        
        # Sleep before next update
        time.sleep(5)

# Start background update thread
update_thread = threading.Thread(target=update_data_thread, daemon=True)
update_thread.start()

# Setup Streamlit app
st.set_page_config(
    page_title="Dynamic AI Rate Limiter Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Dynamic AI Rate Limiter")
st.sidebar.image("dashboard/assets/rate_limit_icon.png", width=100)

# Demo section in sidebar
st.sidebar.header("Demo Controls")

# User selection
if st.sidebar.checkbox("Select Demo Users", True):
    std_users, prm_users = get_api_users()
    
    if std_users:
        std_user = st.sidebar.selectbox("Standard User", std_users, 
                                      index=0 if std_users else None)
        global_state["std_user_id"] = std_user
    else:
        st.sidebar.warning("No STD users available")
    
    if prm_users:
        prm_user = st.sidebar.selectbox("Premium User", prm_users,
                                      index=0 if prm_users else None)
        global_state["prm_user_id"] = prm_user
    else:
        st.sidebar.warning("No PRM users available")

# Traffic simulation control
st.sidebar.subheader("Traffic Simulation")
col1, col2 = st.sidebar.columns(2)
std_count = col1.number_input("STD Users", 1, 20, 5)
prm_count = col2.number_input("PRM Users", 1, 10, 2)
anomaly_count = st.sidebar.number_input("Anomalous Users", 0, 5, 1)
duration = st.sidebar.slider("Duration (seconds)", 10, 300, 60)

if st.sidebar.button("Run Traffic Simulation"):
    with st.sidebar:
        with st.spinner("Running traffic simulation..."):
            # Run simulation
            global_state["simulation_running"] = True
            
            # Create async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_traffic_simulation(
                duration, std_count, prm_count, anomaly_count
            ))
            loop.close()
            
            global_state["simulation_running"] = False
            
            if results:
                st.success(f"Simulation completed: {results['requests_sent']} requests sent")
                st.metric("Requests/second", f"{results['requests_per_second']:.2f}")
                st.metric("Rate Limited", f"{results['rate_limited_rate']:.1%}")

# Dashboard section selection
st.sidebar.header("Dashboard Sections")
show_overview = st.sidebar.checkbox("Overview", True)
show_token_buckets = st.sidebar.checkbox("Token Buckets", True)
show_traffic = st.sidebar.checkbox("Traffic Analysis", True)
show_ml = st.sidebar.checkbox("ML Models", True)

# Display demo info
st.sidebar.markdown("---")
st.sidebar.info("This dashboard demonstrates an AI-powered dynamic rate limiting system. "
               "It uses machine learning to predict traffic patterns and detect anomalies, "
               "dynamically adjusting rate limits based on real-time conditions.")

# Main content
st.title("Dynamic AI Rate Limiter Dashboard")

# Overview section
if show_overview:
    st.header("System Overview")
    
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Traffic metrics
    traffic_data = global_state["traffic_data"]
    if not traffic_data.empty:
        recent_requests = traffic_data['request_count'].sum() if 'request_count' in traffic_data.columns else 0
        avg_response = traffic_data['avg_response_time'].mean() if 'avg_response_time' in traffic_data.columns else 0
        rate_limited = traffic_data['rate_limited_count'].sum() if 'rate_limited_count' in traffic_data.columns else 0
        error_count = traffic_data['error_count'].sum() if 'error_count' in traffic_data.columns else 0
        
        col1.metric("Recent Requests", f"{int(recent_requests)}")
        col2.metric("Avg Response Time", f"{avg_response:.3f}s")
        col3.metric("Rate Limited", f"{int(rate_limited)}")
        col4.metric("Errors", f"{int(error_count)}")
    else:
        col1.metric("Recent Requests", "N/A")
        col2.metric("Avg Response Time", "N/A")
        col3.metric("Rate Limited", "N/A")
        col4.metric("Errors", "N/A")
    
    # Traffic overview chart
    st.subheader("Recent Traffic")
    if not traffic_data.empty:
        fig = create_traffic_overview(traffic_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent traffic data available. Try running a simulation.")
    
    # User and endpoint distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Type Distribution")
        if not traffic_data.empty:
            # Extract user type distribution from traffic data
            user_distribution = get_user_type_distribution(traffic_data)
            
            if user_distribution:
                # Create a dataframe with user type distribution
                user_df = pd.DataFrame({
                    'user_type': list(user_distribution.keys()),
                    'request_count': list(user_distribution.values())
                })
                fig = create_user_type_distribution(user_df, user_type_column='user_type', count_column='request_count')
            else:
                # Fall back to the standard function if our custom extraction failed
                fig = create_user_type_distribution(traffic_data)
                
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user distribution data available.")
    
    with col2:
        st.subheader("Endpoint Distribution")
        if not traffic_data.empty:
            # Create a simulated endpoint distribution if real data doesn't have it
            if 'endpoint' not in traffic_data.columns:
                endpoint_data = pd.DataFrame({
                    'endpoint': ['/api/ping', '/api/resources/light', '/api/resources/medium', 
                                '/api/resources/heavy', '/api/burst'],
                    'count': [40, 30, 20, 5, 5]
                })
            else:
                endpoint_data = traffic_data
            
            fig = create_endpoint_distribution(endpoint_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No endpoint distribution data available.")

# Token bucket section
if show_token_buckets:
    st.header("Token Bucket Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Standard User")
        std_bucket = global_state["std_bucket"]
        
        # Token bucket gauge
        fig = create_token_bucket_gauge(
            current_tokens=std_bucket["tokens"],
            capacity=std_bucket["capacity"],
            user_type="STD"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        st.metric("Refill Rate", f"{std_bucket['refill_rate']} tokens/minute")
        
        # Create simulated token history if real data isn't available
        # In a real app, you'd collect this from the API
        if global_state["token_history"].empty:
            now = datetime.now()
            timestamps = [now - timedelta(minutes=i) for i in range(30)]
            timestamps.reverse()
            
            std_bucket_history = pd.DataFrame({
                'timestamp': timestamps,
                'user_id': 'std_user_1',
                'user_type': 'STD',
                'tokens': [20 - (i % 10) for i in range(30)],
                'capacity': [20] * 30,
                'tokens_consumed': [1 if i % 3 == 0 else 0 for i in range(30)],
                'rate_limited': [True if i % 15 == 0 else False for i in range(30)]
            })
            
            fig = create_token_level_history(
                bucket_history=std_bucket_history,
                user_type='STD'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Premium User")
        prm_bucket = global_state["prm_bucket"]
        
        # Token bucket gauge
        fig = create_token_bucket_gauge(
            current_tokens=prm_bucket["tokens"],
            capacity=prm_bucket["capacity"],
            user_type="PRM"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        st.metric("Refill Rate", f"{prm_bucket['refill_rate']} tokens/minute")
        
        # Create simulated token history if real data isn't available
        if global_state["token_history"].empty:
            now = datetime.now()
            timestamps = [now - timedelta(minutes=i) for i in range(30)]
            timestamps.reverse()
            
            prm_bucket_history = pd.DataFrame({
                'timestamp': timestamps,
                'user_id': 'prm_user_1',
                'user_type': 'PRM',
                'tokens': [60 - (i % 20) for i in range(30)],
                'capacity': [60] * 30,
                'tokens_consumed': [2 if i % 4 == 0 else 0 for i in range(30)],
                'rate_limited': [True if i % 25 == 0 else False for i in range(30)]
            })
            
            fig = create_token_level_history(
                bucket_history=prm_bucket_history,
                user_type='PRM'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Token bucket comparison
    st.subheader("Token Bucket Comparison")
    fig = create_token_comparison(
        std_data=std_bucket,
        prm_data=prm_bucket
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Dynamic parameters explanation
    st.subheader("Dynamic Rate Limit Parameters")
    st.markdown("""
    The system dynamically adjusts these parameters based on:
    - Current traffic volume and patterns
    - Prediction of upcoming traffic spikes
    - Detection of anomalous behavior
    - User type (Premium users get higher limits)
    
    When traffic increases, the system may reduce token capacity and refill rates to protect backend services.
    During anomalies, rate limits may be temporarily tightened for affected traffic patterns.
    """)

# Traffic analysis section
if show_traffic:
    st.header("Traffic Analysis")
    
    # Traffic heatmap
    st.subheader("Traffic Patterns by Hour and Day")
    traffic_data = global_state["traffic_data"]
    
    if not traffic_data.empty:
        # Create dummy data for demonstration if needed
        if len(traffic_data) < 7 * 24:  # Less than a week of data
            # Create synthetic data for demonstration
            hours = 24 * 7  # 1 week of hourly data
            now = datetime.now()
            timestamps = [now - timedelta(hours=i) for i in range(hours)]
            timestamps.reverse()
            
            # Create patterns - higher during business hours, lower on weekends
            values = []
            for ts in timestamps:
                hour = ts.hour
                day = ts.weekday()
                
                # Base value
                value = 100
                
                # Hour effect (business hours peak)
                if 9 <= hour <= 17:
                    value += 50
                elif 6 <= hour <= 8 or 18 <= hour <= 20:
                    value += 20
                
                # Day effect (weekends lower)
                if day >= 5:  # Weekend
                    value *= 0.6
                
                # Add randomness
                value += np.random.normal(0, 10)
                values.append(max(0, value))
            
            heatmap_data = pd.DataFrame({
                'timestamp': timestamps,
                'request_count': values
            })
        else:
            heatmap_data = traffic_data
        
        fig = create_traffic_heatmap(heatmap_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for traffic heatmap. Try running a longer simulation.")
    
    # Anomaly detection
    st.subheader("Traffic Anomaly Detection")
    
    anomaly_data = global_state["anomaly_data"]
    if not anomaly_data.empty:
        fig = create_anomaly_timeline(anomaly_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create dummy anomaly data for demonstration
        now = datetime.now()
        timestamps = [now - timedelta(minutes=i) for i in range(60)]
        timestamps.reverse()
        
        # Create base traffic pattern
        values = [100 + 20 * np.sin(i / 10) for i in range(60)]
        
        # Add anomalies
        for i in [10, 30, 50]:
            values[i] *= 2.5
        
        demo_anomaly_data = pd.DataFrame({
            'timestamp': timestamps,
            'request_rate': values,
            'is_anomaly': [i in [10, 30, 50] for i in range(60)]
        })
        
        fig = create_anomaly_timeline(demo_anomaly_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Response time analysis
    st.subheader("Response Time Analysis")
    
    if not traffic_data.empty and 'avg_response_time' in traffic_data.columns:
        fig = create_response_time_graph(traffic_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create dummy response time data
        now = datetime.now()
        timestamps = [now - timedelta(minutes=i) for i in range(60)]
        timestamps.reverse()
        
        # Create response time pattern with some spikes
        base_times = [0.1 + 0.05 * np.sin(i / 10) for i in range(60)]
        for i in [15, 35, 45]:
            base_times[i] *= 3
        
        demo_response_data = pd.DataFrame({
            'timestamp': timestamps,
            'avg_response_time': base_times
        })
        
        fig = create_response_time_graph(demo_response_data)
        st.plotly_chart(fig, use_container_width=True)

# ML models section
if show_ml:
    st.header("ML Model Analysis")
    
    tab1, tab2 = st.tabs(["Traffic Prediction", "Anomaly Detection"])
    
    with tab1:
        st.subheader("Traffic Prediction Model")
        
        # Model performance
        history = global_state["training_history"]
        if history and "traffic_predictor" in history and history["traffic_predictor"]:
            fig = create_model_performance_graph(history, "traffic_predictor")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training history available for traffic prediction model.")
        
        # Predictions
        st.subheader("Traffic Prediction")
        historical_data = global_state["historical_data"]
        prediction_data = global_state["prediction_data"]
        
        if not historical_data.empty and not prediction_data.empty:
            fig = create_prediction_graph(historical_data, prediction_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create dummy prediction data
            now = datetime.now()
            hist_timestamps = [now - timedelta(minutes=i) for i in range(30)]
            hist_timestamps.reverse()
            
            pred_timestamps = [now + timedelta(minutes=i) for i in range(1, 11)]
            
            # Create patterns
            hist_values = [100 + 20 * np.sin(i / 5) for i in range(30)]
            
            # Predicted continuation of the pattern
            pred_values = [100 + 20 * np.sin((30 + i) / 5) for i in range(10)]
            
            # Add some noise to historical data
            hist_values = [v + np.random.normal(0, 5) for v in hist_values]
            
            demo_historical = pd.DataFrame({
                'timestamp': hist_timestamps,
                'request_rate': hist_values
            })
            
            demo_prediction = pd.DataFrame({
                'timestamp': pred_timestamps,
                'request_rate': pred_values
            })
            
            fig = create_prediction_graph(demo_historical, demo_prediction)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        The traffic prediction model uses an LSTM neural network to forecast future API traffic based on historical patterns.
        
        **How it works:**
        1. Historical traffic data is collected and processed
        2. LSTM model learns temporal patterns in the data
        3. Model predicts traffic levels for the next time window
        4. Rate limit parameters are adjusted based on predictions
        
        This allows the system to proactively adjust rate limits before traffic spikes occur, preventing API overload.
        """)
    
    with tab2:
        st.subheader("Anomaly Detection Model")
        
        # Anomaly visualization
        anomaly_data = global_state["anomaly_data"]
        if not anomaly_data.empty:
            fig = create_anomaly_detection_graph(anomaly_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create dummy anomaly data for 2D visualization
            np.random.seed(42)
            n_points = 100
            
            # Generate normal data points in a cluster
            normal_x = np.random.normal(100, 15, n_points)
            normal_y = np.random.normal(0.2, 0.05, n_points)
            
            # Generate anomalies
            anomaly_x = np.concatenate([
                np.random.normal(200, 10, 3),  # Traffic spike anomalies
                np.random.normal(50, 10, 2)    # Traffic drop anomalies
            ])
            
            anomaly_y = np.concatenate([
                np.random.normal(0.4, 0.05, 3),  # Error rate spike anomalies
                np.random.normal(0.1, 0.02, 2)   # Low error rate anomalies
            ])
            
            # Combine data
            x = np.concatenate([normal_x, anomaly_x])
            y = np.concatenate([normal_y, anomaly_y])
            is_anomaly = np.concatenate([np.zeros(n_points), np.ones(5)])
            
            demo_anomaly_data = pd.DataFrame({
                'request_rate': x,
                'error_rate': y,
                'is_anomaly': is_anomaly.astype(bool)
            })
            
            fig = create_anomaly_detection_graph(
                demo_anomaly_data,
                x_feature='request_rate',
                y_feature='error_rate',
                title='Traffic vs Error Rate Anomalies'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importances = global_state["feature_importances"]
        
        if feature_importances:
            fig = create_feature_importance_graph(feature_importances)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create dummy feature importance
            demo_importances = {
                "request_rate": 0.25,
                "error_rate": 0.20,
                "response_time_mean": 0.18,
                "token_consumption_rate": 0.15,
                "request_variance": 0.12,
                "user_type_std_count": 0.05,
                "user_type_prm_count": 0.05
            }
            
            fig = create_feature_importance_graph(demo_importances)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        The anomaly detection model identifies unusual traffic patterns that could indicate abuse or attacks.
        
        **How it works:**
        1. Isolation Forest algorithm learns normal traffic patterns
        2. New traffic is compared against the learned patterns
        3. Unusual traffic is flagged as anomalous
        4. Rate limits are temporarily adjusted for affected traffic patterns
        
        This helps protect the API from abuse while maintaining service for legitimate users.
        """)

# Footer
st.markdown("---")
st.markdown("**Dynamic AI Rate Limiter** • Final Year Project • © 2025")

# Run the app with: streamlit run dashboard/app.py
