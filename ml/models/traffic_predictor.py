import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
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
logger = logging.getLogger("traffic_predictor")

class TrafficPredictor:
    """
    LSTM-based model for predicting API traffic patterns.
    Used to dynamically adjust rate limiting parameters based on predicted traffic.
    """
    
    def __init__(
        self,
        lookback_window: int = 60,  # 60 minutes of historical data
        prediction_horizon: int = 10,  # Predict 10 minutes ahead
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the traffic predictor.
        
        Args:
            lookback_window: Number of time steps to use as input
            prediction_horizon: Number of time steps to predict
            feature_columns: List of column names to use as features (default: all except timestamp)
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns
        
        # Model components
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.is_trained = False
        
        # Configuration
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.patience = 10
        
        # Performance metrics
        self.train_history = None
        self.validation_loss = None
        self.test_loss = None
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for LSTM model.
        
        Args:
            data: DataFrame with traffic data (should have a 'timestamp' column)
            
        Returns:
            Tuple of (X, y) where X is the input data and y is the target data
        """
        # Ensure data is sorted by timestamp
        data = data.sort_values('timestamp')
        
        # Extract features
        if self.feature_columns is None:
            feature_cols = [col for col in data.columns if col != 'timestamp']
        else:
            feature_cols = self.feature_columns
        
        # Scale each feature individually
        scaled_data = pd.DataFrame()
        for col in feature_cols:
            if col not in self.feature_scalers:
                self.feature_scalers[col] = MinMaxScaler()
                scaled_data[col] = self.feature_scalers[col].fit_transform(data[col].values.reshape(-1, 1)).flatten()
            else:
                scaled_data[col] = self.feature_scalers[col].transform(data[col].values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback_window - self.prediction_horizon + 1):
            # Input sequence
            X.append(scaled_data.iloc[i:(i + self.lookback_window)].values)
            
            # Target sequence - multi-step prediction
            y.append(scaled_data.iloc[(i + self.lookback_window):(i + self.lookback_window + self.prediction_horizon)].values)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (lookback_window, feature_count)
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            
            # Output layer - shape is (prediction_horizon, feature_count)
            Dense(self.prediction_horizon * input_shape[1])
        ])
        
        # Reshape the output to match the prediction horizon and feature count
        model.add(tf.keras.layers.Reshape((self.prediction_horizon, input_shape[1])))
        
        # Compile with MAE loss for better interpretability
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mae',
            metrics=['mse']
        )
        
        self.model = model
        logger.info(f"Model built with input shape {input_shape}")
    
    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 1,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the LSTM model on traffic data.
        
        Args:
            train_data: DataFrame with training data
            validation_data: Optional DataFrame with validation data
            test_data: Optional DataFrame with test data
            epochs: Number of training epochs (default: self.epochs)
            batch_size: Batch size for training (default: self.batch_size)
            verbose: Verbosity level for training
            save_path: Path to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        # Set training parameters
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        
        # Preprocess training data
        X_train, y_train = self._preprocess_data(train_data)
        
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # Prepare validation data if provided
        if validation_data is not None:
            X_val, y_val = self._preprocess_data(validation_data)
            validation_data = (X_val, y_val)
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=self.patience,
                restore_best_weights=True
            )
        ]
        
        # Add model checkpoint if save path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_loss' if validation_data is not None else 'loss',
                    save_best_only=True
                )
            )
        
        # Train the model
        logger.info(f"Training model with {len(X_train)} sequences, {self.epochs} epochs")
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.train_history = history.history
        self.is_trained = True
        
        # Evaluate on test data if provided
        if test_data is not None:
            X_test, y_test = self._preprocess_data(test_data)
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            self.test_loss = test_loss
            logger.info(f"Test loss: {test_loss}")
        
        # Save the model if path provided
        if save_path:
            self.save(save_path)
        
        return {
            "history": self.train_history,
            "validation_loss": self.train_history.get('val_loss', [])[-1] if validation_data is not None else None,
            "test_loss": self.test_loss
        }
    
    def predict(
        self,
        input_data: pd.DataFrame,
        return_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict traffic for the next prediction_horizon time steps.
        
        Args:
            input_data: DataFrame with historical traffic data 
                        (should have at least lookback_window entries)
            return_dataframe: Whether to return results as a DataFrame with timestamps
            
        Returns:
            Numpy array or DataFrame with predicted traffic
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure input data has enough history
        if len(input_data) < self.lookback_window:
            raise ValueError(f"Input data must have at least {self.lookback_window} entries")
        
        # Preprocess the input data
        input_features = input_data.copy()
        
        # Extract features
        if self.feature_columns is None:
            feature_cols = [col for col in input_features.columns if col != 'timestamp']
        else:
            feature_cols = self.feature_columns
        
        # Scale the features
        scaled_input = pd.DataFrame()
        for col in feature_cols:
            if col in self.feature_scalers:
                scaled_input[col] = self.feature_scalers[col].transform(
                    input_features[col].values.reshape(-1, 1)
                ).flatten()
            else:
                logger.warning(f"Feature {col} not found in trained scalers, using default scaling")
                scaler = MinMaxScaler()
                scaled_input[col] = scaler.fit_transform(
                    input_features[col].values.reshape(-1, 1)
                ).flatten()
        
        # Take the most recent lookback_window entries
        recent_data = scaled_input.iloc[-self.lookback_window:].values
        
        # Reshape for LSTM input [samples, time steps, features]
        X_pred = recent_data.reshape(1, self.lookback_window, len(feature_cols))
        
        # Make prediction
        y_pred = self.model.predict(X_pred, verbose=0)
        
        # Inverse transform the predictions
        predictions = np.squeeze(y_pred, axis=0)  # Remove batch dimension
        
        # Inverse scale each feature
        unscaled_predictions = np.zeros_like(predictions)
        for i, col in enumerate(feature_cols):
            if col in self.feature_scalers:
                unscaled_predictions[:, i] = self.feature_scalers[col].inverse_transform(
                    predictions[:, i].reshape(-1, 1)
                ).flatten()
        
        # Create DataFrame with timestamps if requested
        if return_dataframe:
            # Get the last timestamp from input data
            last_timestamp = input_data['timestamp'].iloc[-1]
            
            # Generate future timestamps
            # Assuming timestamps are in seconds and spaced evenly
            if len(input_data) >= 2:
                time_step = input_data['timestamp'].iloc[-1] - input_data['timestamp'].iloc[-2]
            else:
                time_step = 60  # Default to 1 minute
            
            future_timestamps = [last_timestamp + (i + 1) * time_step for i in range(self.prediction_horizon)]
            
            # Create DataFrame
            result_df = pd.DataFrame()
            result_df['timestamp'] = future_timestamps
            
            for i, col in enumerate(feature_cols):
                result_df[col] = unscaled_predictions[:, i]
            
            return result_df
        
        return unscaled_predictions
    
    def save(self, path: str) -> None:
        """
        Save the model and preprocessing objects.
        
        Args:
            path: Path to save the model (without extension)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save Keras model
        model_path = f"{path}.h5"
        self.model.save(model_path)
        
        # Save scalers and configuration
        config = {
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'feature_columns': self.feature_columns,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'is_trained': self.is_trained,
            'train_history': self.train_history,
            'validation_loss': self.validation_loss,
            'test_loss': self.test_loss,
            'feature_scalers': self.feature_scalers
        }
        
        config_path = f"{path}_config.joblib"
        joblib.dump(config, config_path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrafficPredictor':
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model (without extension)
            
        Returns:
            Loaded TrafficPredictor instance
        """
        # Load configuration
        config_path = f"{path}_config.joblib"
        config = joblib.load(config_path)
        
        # Create instance with saved configuration
        instance = cls(
            lookback_window=config['lookback_window'],
            prediction_horizon=config['prediction_horizon'],
            feature_columns=config['feature_columns']
        )
        
        # Restore configuration
        instance.batch_size = config['batch_size']
        instance.epochs = config['epochs']
        instance.learning_rate = config['learning_rate']
        instance.patience = config['patience']
        instance.is_trained = config['is_trained']
        instance.train_history = config['train_history']
        instance.validation_loss = config['validation_loss']
        instance.test_loss = config['test_loss']
        instance.feature_scalers = config['feature_scalers']
        
        # Load Keras model
        model_path = f"{path}.h5"
        instance.model = load_model(model_path)
        
        logger.info(f"Model loaded from {path}")
        return instance


# Helper functions for more convenient usage

def train_traffic_predictor(
    train_data: pd.DataFrame,
    validation_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    lookback_window: int = 60,
    prediction_horizon: int = 10,
    feature_columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> TrafficPredictor:
    """
    Train a traffic predictor model.
    
    Args:
        train_data: DataFrame with training data
        validation_data: Optional DataFrame with validation data
        test_data: Optional DataFrame with test data
        lookback_window: Number of time steps to use as input
        prediction_horizon: Number of time steps to predict
        feature_columns: List of column names to use as features
        save_path: Path to save the trained model
        
    Returns:
        Trained TrafficPredictor instance
    """
    predictor = TrafficPredictor(
        lookback_window=lookback_window,
        prediction_horizon=prediction_horizon,
        feature_columns=feature_columns
    )
    
    predictor.train(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        save_path=save_path
    )
    
    return predictor


def load_traffic_predictor(path: str) -> TrafficPredictor:
    """
    Load a trained traffic predictor model.
    
    Args:
        path: Path to the saved model (without extension)
        
    Returns:
        Loaded TrafficPredictor instance
    """
    return TrafficPredictor.load(path)


async def predict_traffic(
    recent_data: pd.DataFrame,
    model_path: Optional[str] = None,
    predictor: Optional[TrafficPredictor] = None
) -> pd.DataFrame:
    """
    Predict traffic for the next time steps.
    
    Args:
        recent_data: DataFrame with recent traffic data
        model_path: Path to the saved model (without extension)
        predictor: Optional TrafficPredictor instance (if already loaded)
        
    Returns:
        DataFrame with predicted traffic
    """
    # Load model if not provided
    if predictor is None:
        if model_path is None:
            raise ValueError("Either model_path or predictor must be provided")
        predictor = load_traffic_predictor(model_path)
    
    # Make prediction
    predictions = predictor.predict(recent_data, return_dataframe=True)
    
    return predictions


def visualize_traffic_prediction(
    historical_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    feature_to_plot: Optional[str] = None,
    title: str = "Traffic Prediction",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize traffic prediction against historical data.
    
    Args:
        historical_data: DataFrame with historical traffic data
        predicted_data: DataFrame with predicted traffic data
        feature_to_plot: Feature to plot (if None, will use the first non-timestamp column)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Determine which feature to plot
    if feature_to_plot is None:
        feature_to_plot = [col for col in historical_data.columns if col != 'timestamp'][0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical data
    ax.plot(
        historical_data['timestamp'], 
        historical_data[feature_to_plot],
        label='Historical', 
        color='blue'
    )
    
    # Plot predicted data
    ax.plot(
        predicted_data['timestamp'], 
        predicted_data[feature_to_plot],
        label='Predicted', 
        color='red', 
        linestyle='--'
    )
    
    # Add prediction range shading
    ax.axvspan(
        historical_data['timestamp'].iloc[-1],
        predicted_data['timestamp'].iloc[-1],
        alpha=0.2,
        color='gray',
        label='Prediction Range'
    )
    
    # Format the plot
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(feature_to_plot)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis as times if timestamps are in seconds
    if historical_data['timestamp'].max() > 1e9:  # Unix timestamp check
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    return fig


# Example usage if run directly
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    
    # Create time series with daily and weekly patterns
    hours = 24 * 7 * 2  # 2 weeks of hourly data
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
    
    # Create dataframe
    data = pd.DataFrame({
        'timestamp': timestamps,
        'requests_per_minute': traffic,
        'error_rate': error_rate,
        'avg_response_time': avg_response_time
    })
    
    # Split into train/validation/test
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    # Train model
    predictor = train_traffic_predictor(
        train_data=train_data,
        validation_data=val_data,
        test_data=test_data,
        lookback_window=24,  # 24 hours lookback
        prediction_horizon=6,  # Predict 6 hours ahead
        save_path="./data/models/traffic_predictor"
    )
    
    # Make a prediction
    prediction = predictor.predict(test_data.head(30), return_dataframe=True)
    
    # Visualize
    fig = visualize_traffic_prediction(
        historical_data=test_data.head(30),
        predicted_data=prediction,
        feature_to_plot='requests_per_minute',
        title='API Traffic Prediction'
    )
    
    # Save figure
    fig.savefig('./data/models/traffic_prediction.png')
    plt.close(fig)
    
    print("Traffic prediction model demo complete")
