import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
import logging
from typing import Union, List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anomaly_detector")

class AnomalyDetector:
    """
    Anomaly detection model for identifying unusual API traffic patterns.
    Uses Isolation Forest algorithm to detect outliers.
    """
    
    def __init__(
        self,
        contamination: float = 0.05,  # Expected proportion of anomalies
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.preprocessor = None
        self.is_trained = False
        
        # Performance metrics
        self.feature_importances = None
        
        # Thresholds
        self.anomaly_threshold = None
        self.score_threshold = None
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline for features.
        
        Returns:
            ColumnTransformer for preprocessing features
        """
        # Create transformers for different feature types
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features),
                ('num', numerical_transformer, self.numerical_features)
            ]
        )
        
        return preprocessor
    
    def _build_model(self) -> None:
        """Build the anomaly detection model."""
        # Create preprocessor
        self.preprocessor = self._create_preprocessor()
        
        # Create Isolation Forest model
        isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto'
        )
        
        # Create full pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('anomaly_detector', isolation_forest)
        ])
    
    def train(
        self,
        data: pd.DataFrame,
        calculate_thresholds: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the anomaly detection model.
        
        Args:
            data: DataFrame with traffic data
            calculate_thresholds: Whether to calculate anomaly score thresholds
            verbose: Whether to print training information
            
        Returns:
            Dictionary with training metrics
        """
        # Automatically determine features if not provided
        if not self.numerical_features and not self.categorical_features:
            # Exclude timestamp column
            all_features = [col for col in data.columns if col != 'timestamp']
            
            # Simple heuristic: if column has < 10 unique values, treat as categorical
            self.categorical_features = [
                col for col in all_features 
                if data[col].nunique() < 10 and data[col].dtype != float
            ]
            self.numerical_features = [
                col for col in all_features 
                if col not in self.categorical_features
            ]
            
            if verbose:
                logger.info(f"Auto-detected categorical features: {self.categorical_features}")
                logger.info(f"Auto-detected numerical features: {self.numerical_features}")
        
        # Build model if not already built
        if self.model is None:
            self._build_model()
        
        # Fit the model
        if verbose:
            logger.info(f"Training anomaly detector on {len(data)} samples")
        
        # Select only relevant features
        features = data[self.categorical_features + self.numerical_features]
        
        # Train the model
        self.model.fit(features)
        
        # Extract feature importances if possible
        if hasattr(self.model['anomaly_detector'], 'feature_importances_'):
            self.feature_importances = dict(zip(
                self.categorical_features + self.numerical_features,
                self.model['anomaly_detector'].feature_importances_
            ))
        
        # Calculate anomaly thresholds if requested
        if calculate_thresholds:
            scores = self.score_samples(data)
            self._calculate_thresholds(scores)
        
        self.is_trained = True
        
        return {
            "feature_importances": self.feature_importances,
            "anomaly_threshold": self.anomaly_threshold,
            "score_threshold": self.score_threshold
        }
    
    def _calculate_thresholds(self, scores: np.ndarray) -> None:
        """
        Calculate thresholds for anomaly detection.
        
        Args:
            scores: Anomaly scores for a dataset
        """
        # Lower scores indicate more anomalous
        self.score_threshold = np.percentile(scores, self.contamination * 100)
        self.anomaly_threshold = -0.5  # Default threshold for Isolation Forest
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict whether samples are anomalies.
        
        Args:
            data: DataFrame with traffic data
            
        Returns:
            Array with 1 for normal samples and -1 for anomalies
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Select only relevant features
        features = data[self.categorical_features + self.numerical_features]
        
        # Make predictions
        predictions = self.model.predict(features)
        
        return predictions
    
    def score_samples(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate anomaly scores for samples.
        
        Lower scores indicate more anomalous behavior.
        
        Args:
            data: DataFrame with traffic data
            
        Returns:
            Array with anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring samples")
        
        # Select only relevant features
        features = data[self.categorical_features + self.numerical_features]
        
        # Calculate anomaly scores
        scores = self.model['anomaly_detector'].score_samples(
            self.model['preprocessor'].transform(features)
        )
        
        return scores
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        return_scores: bool = True,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Args:
            data: DataFrame with traffic data
            return_scores: Whether to include anomaly scores in the result
            threshold: Custom score threshold (overrides the calculated one)
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        # Make copy of data to avoid modifying the original
        result = data.copy()
        
        # Get predictions (1 for normal, -1 for anomalies)
        result['is_anomaly'] = self.predict(data) == -1
        
        # Calculate scores if requested
        if return_scores:
            result['anomaly_score'] = self.score_samples(data)
            
            # Add threshold-based detection
            threshold = threshold or self.score_threshold
            if threshold is not None:
                result['is_score_anomaly'] = result['anomaly_score'] < threshold
        
        return result
    
    def save(self, path: str) -> None:
        """
        Save the model and preprocessing objects.
        
        Args:
            path: Path to save the model (without extension)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        model_path = f"{path}.joblib"
        joblib.dump(self.model, model_path)
        
        # Save configuration
        config = {
            'contamination': self.contamination,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'random_state': self.random_state,
            'is_trained': self.is_trained,
            'feature_importances': self.feature_importances,
            'anomaly_threshold': self.anomaly_threshold,
            'score_threshold': self.score_threshold
        }
        
        config_path = f"{path}_config.joblib"
        joblib.dump(config, config_path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model (without extension)
            
        Returns:
            Loaded AnomalyDetector instance
        """
        # Load configuration
        config_path = f"{path}_config.joblib"
        config = joblib.load(config_path)
        
        # Create instance with saved configuration
        instance = cls(
            contamination=config['contamination'],
            categorical_features=config['categorical_features'],
            numerical_features=config['numerical_features'],
            random_state=config['random_state']
        )
        
        # Restore configuration
        instance.is_trained = config['is_trained']
        instance.feature_importances = config['feature_importances']
        instance.anomaly_threshold = config['anomaly_threshold']
        instance.score_threshold = config['score_threshold']
        
        # Load the model
        model_path = f"{path}.joblib"
        instance.model = joblib.load(model_path)
        
        logger.info(f"Model loaded from {path}")
        return instance


# Helper functions for more convenient usage

def train_anomaly_detector(
    data: pd.DataFrame,
    contamination: float = 0.05,
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> AnomalyDetector:
    """
    Train an anomaly detector model.
    
    Args:
        data: DataFrame with traffic data
        contamination: Expected proportion of anomalies
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        save_path: Path to save the trained model
        
    Returns:
        Trained AnomalyDetector instance
    """
    detector = AnomalyDetector(
        contamination=contamination,
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )
    
    detector.train(data=data)
    
    if save_path:
        detector.save(save_path)
    
    return detector


def load_anomaly_detector(path: str) -> AnomalyDetector:
    """
    Load a trained anomaly detector model.
    
    Args:
        path: Path to the saved model (without extension)
        
    Returns:
        Loaded AnomalyDetector instance
    """
    return AnomalyDetector.load(path)


async def detect_anomalies(
    data: pd.DataFrame,
    model_path: Optional[str] = None,
    detector: Optional[AnomalyDetector] = None
) -> pd.DataFrame:
    """
    Detect anomalies in traffic data.
    
    Args:
        data: DataFrame with traffic data
        model_path: Path to the saved model (without extension)
        detector: Optional AnomalyDetector instance (if already loaded)
        
    Returns:
        DataFrame with anomaly detection results
    """
    # Load model if not provided
    if detector is None:
        if model_path is None:
            raise ValueError("Either model_path or detector must be provided")
        detector = load_anomaly_detector(model_path)
    
    # Detect anomalies
    result = detector.detect_anomalies(data)
    
    return result


def visualize_anomalies(
    data: pd.DataFrame,
    feature_x: str,
    feature_y: Optional[str] = None,
    time_feature: Optional[str] = 'timestamp',
    title: str = "Anomaly Detection",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize anomalies in the data.
    
    Args:
        data: DataFrame with anomaly detection results (must contain 'is_anomaly' column)
        feature_x: Feature for x-axis (or time feature for time series plot)
        feature_y: Feature for y-axis (if None, will use 'anomaly_score' if available)
        time_feature: If provided, will create a time series plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if 'is_anomaly' not in data.columns:
        raise ValueError("Data must contain 'is_anomaly' column from anomaly detection")
    
    # If feature_y not specified, use anomaly_score if available
    if feature_y is None and 'anomaly_score' in data.columns:
        feature_y = 'anomaly_score'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine if we're making a time series or scatter plot
    if time_feature and time_feature in data.columns:
        # Time series plot
        normal_data = data[~data['is_anomaly']]
        anomaly_data = data[data['is_anomaly']]
        
        # Plot normal data
        ax.plot(
            normal_data[time_feature], 
            normal_data[feature_x],
            'o-', 
            color='blue', 
            alpha=0.5,
            label='Normal'
        )
        
        # Plot anomalies
        ax.scatter(
            anomaly_data[time_feature], 
            anomaly_data[feature_x],
            color='red', 
            s=50, 
            label='Anomaly'
        )
        
        ax.set_xlabel('Time')
        ax.set_ylabel(feature_x)
        
    elif feature_y:
        # Scatter plot (2D)
        normal_data = data[~data['is_anomaly']]
        anomaly_data = data[data['is_anomaly']]
        
        # Plot normal data
        ax.scatter(
            normal_data[feature_x], 
            normal_data[feature_y],
            color='blue', 
            alpha=0.5,
            label='Normal'
        )
        
        # Plot anomalies
        ax.scatter(
            anomaly_data[feature_x], 
            anomaly_data[feature_y],
            color='red', 
            alpha=0.8,
            label='Anomaly'
        )
        
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        
    else:
        # 1D plot (histogram)
        sns.histplot(
            data=data, 
            x=feature_x, 
            hue='is_anomaly',
            palette={True: 'red', False: 'blue'},
            ax=ax,
            bins=30
        )
        
        ax.set_xlabel(feature_x)
        ax.set_ylabel('Count')
    
    # Add threshold line if we're plotting anomaly scores
    if feature_y == 'anomaly_score' and 'score_threshold' in data.columns:
        ax.axhline(
            y=data['score_threshold'].iloc[0],
            color='green',
            linestyle='--',
            label='Threshold'
        )
    
    # Format the plot
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis as times if using timestamps
    if time_feature and time_feature in data.columns and data[time_feature].max() > 1e9:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    return fig


# Example usage if run directly
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    
    # Create time series with daily and weekly patterns + anomalies
    hours = 24 * 7  # 1 week of hourly data
    timestamps = np.array([time.time() + i * 3600 for i in range(hours)])
    
    # Base pattern: daily cycle with weekly pattern
    hour_of_day = np.array([datetime.fromtimestamp(t).hour for t in timestamps])
    day_of_week = np.array([datetime.fromtimestamp(t).weekday() for t in timestamps])
    
    # Create traffic pattern
    traffic = 100 + 50 * np.sin(hour_of_day * 2 * np.pi / 24) + 30 * np.sin(day_of_week * 2 * np.pi / 7)
    traffic = traffic + np.random.normal(0, 10, size=hours)  # Add noise
    
    # Create additional features
    error_rate = 0.05 + 0.03 * np.sin(hour_of_day * 2 * np.pi / 24 + 2)
    error_rate = np.maximum(0, error_rate + np.random.normal(0, 0.01, size=hours))
    
    avg_response_time = 0.2 + 0.1 * np.sin(hour_of_day * 2 * np.pi / 24 + 1)
    avg_response_time = np.maximum(0.1, avg_response_time + np.random.normal(0, 0.03, size=hours))
    
    # Add user type as categorical feature
    user_type = np.random.choice(['STD', 'PRM'], size=hours, p=[0.7, 0.3])
    
    # Add endpoint type as categorical feature
    endpoint_type = np.random.choice(['light', 'medium', 'heavy'], size=hours, p=[0.5, 0.3, 0.2])
    
    # Injected anomalies
    # Anomaly 1: Traffic spike
    anomaly_idx1 = np.random.randint(24, hours - 24)
    traffic[anomaly_idx1:anomaly_idx1+3] *= 3.0
    
    # Anomaly 2: Error rate spike
    anomaly_idx2 = np.random.randint(24, hours - 24)
    error_rate[anomaly_idx2:anomaly_idx2+2] = 0.5
    
    # Anomaly 3: Slow response time
    anomaly_idx3 = np.random.randint(24, hours - 24)
    avg_response_time[anomaly_idx3:anomaly_idx3+4] = 1.5
    
    # Create dataframe
    data = pd.DataFrame({
        'timestamp': timestamps,
        'requests_per_minute': traffic,
        'error_rate': error_rate,
        'avg_response_time': avg_response_time,
        'user_type': user_type,
        'endpoint_type': endpoint_type
    })
    
    # Train anomaly detector
    detector = train_anomaly_detector(
        data=data,
        contamination=0.05,
        categorical_features=['user_type', 'endpoint_type'],
        numerical_features=['requests_per_minute', 'error_rate', 'avg_response_time'],
        save_path="./data/models/anomaly_detector"
    )
    
    # Detect anomalies
    result = detector.detect_anomalies(data)
    
    # Visualize anomalies
    fig1 = visualize_anomalies(
        data=result,
        feature_x='requests_per_minute',
        time_feature='timestamp',
        title='API Traffic Anomalies'
    )
    
    fig1.savefig('./data/models/traffic_anomalies.png')
    plt.close(fig1)
    
    # Create 2D visualization
    fig2 = visualize_anomalies(
        data=result,
        feature_x='requests_per_minute',
        feature_y='error_rate',
        time_feature=None,  # No time feature for scatter plot
        title='API Traffic vs Error Rate Anomalies'
    )
    
    fig2.savefig('./data/models/feature_anomalies.png')
    plt.close(fig2)
    
    print("Anomaly detection model demo complete")
