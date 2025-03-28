import asyncio
import time
import random
from typing import Dict, Optional, Tuple, List
import math

# In a real implementation, this would import the ML models
# from ml.models.traffic_predictor import predict_traffic
# from ml.models.anomaly_detector import detect_anomalies

# Global cache for parameters to avoid recalculating too frequently
# In production, use Redis or a similar distributed cache
_parameter_cache = {
    "STD": {
        "capacity": 20,
        "refill_rate": 5,
        "last_updated": 0,
        "traffic_level": "normal"
    },
    "PRM": {
        "capacity": 60,
        "refill_rate": 15,
        "last_updated": 0,
        "traffic_level": "normal"
    }
}

# How often to recalculate parameters (in seconds)
PARAMETER_CACHE_TTL = 60

# Traffic levels and their multipliers
TRAFFIC_LEVELS = {
    "very_low": 1.5,    # Very low traffic, increase limits
    "low": 1.2,         # Low traffic, slightly increase limits
    "normal": 1.0,      # Normal traffic, use baseline limits
    "high": 0.8,        # High traffic, slightly reduce limits
    "very_high": 0.6,   # Very high traffic, reduce limits significantly
    "extreme": 0.4      # Extreme traffic, reduce limits drastically
}

# Baseline parameters for different user types
BASELINE_PARAMETERS = {
    "STD": {
        "capacity": 20,
        "refill_rate": 5
    },
    "PRM": {
        "capacity": 60,
        "refill_rate": 15
    }
}

async def get_current_parameters(user_type: str) -> Dict[str, int]:
    """
    Get the current adjusted parameters for a user type.
    
    This function checks if the cached parameters are still valid,
    and if not, recalculates them based on the current traffic conditions.
    
    Args:
        user_type: User type (STD or PRM)
        
    Returns:
        Dictionary with capacity and refill_rate keys
    """
    global _parameter_cache
    
    # Normalize user type
    user_type = user_type.upper()
    if user_type not in ["STD", "PRM"]:
        user_type = "STD"  # Default to STD for unknown types
    
    # Check if we need to update the parameters
    now = time.time()
    last_updated = _parameter_cache[user_type]["last_updated"]
    
    if now - last_updated > PARAMETER_CACHE_TTL:
        # Time to update parameters
        # In a real implementation, this would use the ML model predictions
        await _update_parameters(user_type)
    
    return {
        "capacity": _parameter_cache[user_type]["capacity"],
        "refill_rate": _parameter_cache[user_type]["refill_rate"]
    }

async def _update_parameters(user_type: str) -> None:
    """
    Update the parameters for a user type based on current traffic.
    
    This is where the ML model predictions would be used.
    For now, we'll use a simulated approach that varies parameters
    based on the current time.
    
    Args:
        user_type: User type (STD or PRM)
    """
    global _parameter_cache
    
    # In a real implementation, get predictions from ML models
    # predicted_traffic = await predict_traffic()
    # anomalies = await detect_anomalies()
    
    # Simulate traffic levels based on current time
    # This creates a cyclical pattern for demonstration purposes
    current_hour = time.localtime().tm_hour
    traffic_level = await _simulate_traffic_level(current_hour)
    
    # Get traffic multiplier
    multiplier = TRAFFIC_LEVELS[traffic_level]
    
    # Get baseline parameters
    baseline = BASELINE_PARAMETERS[user_type]
    
    # Calculate new parameters
    new_params = {
        "capacity": max(5, int(baseline["capacity"] * multiplier)),
        "refill_rate": max(1, int(baseline["refill_rate"] * multiplier)),
        "last_updated": time.time(),
        "traffic_level": traffic_level
    }
    
    # Update cache
    _parameter_cache[user_type] = new_params
    
    # Log the update
    print(f"Updated parameters for {user_type}: {new_params}")

async def _simulate_traffic_level(hour: int) -> str:
    """
    Simulate traffic level based on time of day.
    
    This function creates a realistic traffic pattern:
    - Low traffic in early morning and late night
    - High traffic during business hours
    - Peak traffic during lunch and end of workday
    
    Args:
        hour: Current hour (0-23)
        
    Returns:
        Traffic level as a string
    """
    # Early morning (0-6): very low to low
    if 0 <= hour < 6:
        return "very_low"
    
    # Morning ramp-up (6-9): low to normal
    elif 6 <= hour < 9:
        return "low"
    
    # Morning work hours (9-12): normal to high
    elif 9 <= hour < 12:
        return "normal"
    
    # Lunch hour spike (12-14): high to very high
    elif 12 <= hour < 14:
        return "high"
    
    # Afternoon work hours (14-17): normal to high
    elif 14 <= hour < 17:
        return "normal"
    
    # End-of-day spike (17-19): high to very high
    elif 17 <= hour < 19:
        return "high"
    
    # Evening wind-down (19-22): normal to low
    elif 19 <= hour < 22:
        return "low"
    
    # Late night (22-24): very low
    else:
        return "very_low"

async def adjust_parameters_for_user(user_id: str, adjustment_factor: float) -> None:
    """
    Adjust parameters for a specific user based on their behavior.
    
    This would be called when the anomaly detection system identifies
    unusual behavior from a specific user.
    
    Args:
        user_id: User ID
        adjustment_factor: Factor to adjust parameters by (0.0-1.0)
                           where lower means stricter limits
    
    In a real implementation, this would update a user-specific
    entry in Redis or another distributed cache.
    """
    # This is a placeholder for a real implementation
    print(f"Adjusting parameters for user {user_id} with factor {adjustment_factor}")
    
    # Here you would adjust the specific user's token bucket
    # For example:
    # user_bucket = get_user_bucket(user_id)
    # user_bucket.update_parameters(
    #     capacity=int(user_bucket.capacity * adjustment_factor),
    #     refill_rate=int(user_bucket.refill_rate * adjustment_factor)
    # )

async def adjust_parameters_for_traffic(traffic_level: str) -> None:
    """
    Adjust parameters globally based on traffic level.
    
    This would be called when the system detects a change in
    overall traffic patterns.
    
    Args:
        traffic_level: Traffic level (very_low, low, normal, high, very_high, extreme)
    """
    # This is a placeholder for a real implementation
    print(f"Adjusting parameters for traffic level: {traffic_level}")
    
    # Here you would update the global parameter cache
    # For example:
    for user_type in ["STD", "PRM"]:
        await _update_parameters(user_type)
