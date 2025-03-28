from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any


class Settings(BaseSettings):
    """Application settings"""
    # API settings
    API_TITLE: str = "Dynamic AI Rate Limiter"
    API_DESCRIPTION: str = "AI-based dynamic rate limiting API"
    API_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    
    # Rate limiting default settings
    RATE_LIMIT_STD_CAPACITY: int = 20
    RATE_LIMIT_STD_REFILL_RATE: int = 5
    RATE_LIMIT_PRM_CAPACITY: int = 60
    RATE_LIMIT_PRM_REFILL_RATE: int = 15
    RATE_LIMIT_REFILL_DURATION: int = 60  # seconds
    
    # ML model settings
    ML_MODEL_UPDATE_INTERVAL: int = 300  # seconds (5 minutes)
    ML_TRAFFIC_PREDICTION_WINDOW: int = 60  # minutes
    ML_ANOMALY_DETECTION_SENSITIVITY: float = 0.05  # 5% anomaly threshold
    
    # Data collection settings
    DATA_COLLECTION_ENABLED: bool = True
    DATA_STORAGE_PATH: str = "./data"
    
    # Dashboard settings
    DASHBOARD_ENABLED: bool = True
    DASHBOARD_PORT: int = 8501  # Default Streamlit port
    
    # Feature flags
    FEATURES_DYNAMIC_ADJUSTMENT: bool = True
    FEATURES_ANOMALY_DETECTION: bool = True
    FEATURES_USER_SPECIFIC_LIMITS: bool = True
    
    model_config = {
        # Environment variables prefix
        "env_prefix": "RATELIMIT_",
        
        # Allow environment variables to override
        "env_file": ".env"
    }


# Create settings instance
settings = Settings()

# Export settings
__all__ = ["settings"]
