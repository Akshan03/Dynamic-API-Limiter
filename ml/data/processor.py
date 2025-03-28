import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("traffic_processor")

class TrafficProcessor:
    """
    Process raw traffic data for ML model training and prediction.
    Handles feature engineering, cleaning, and transformation.
    """
    
    def __init__(
        self,
        output_dir: str = "./data/processed",
        time_features: bool = True,
        endpoint_features: bool = True,
        user_features: bool = True,
        normalize: bool = True
    ):
        """
        Initialize the traffic processor.
        
        Args:
            output_dir: Directory to store processed data
            time_features: Whether to extract time-based features
            endpoint_features: Whether to create endpoint-specific features
            user_features: Whether to create user-specific features
            normalize: Whether to normalize numerical features
        """
        self.output_dir = Path(output_dir)
        self.time_features = time_features
        self.endpoint_features = endpoint_features
        self.user_features = user_features
        self.normalize = normalize
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scalers for numerical features
        self.scalers = {}
        
        logger.info(f"Traffic processor initialized with output directory: {self.output_dir}")
    
    def process(
        self,
        data: pd.DataFrame,
        aggregate_window: str = '1min',
        min_samples: int = 10,
        save: bool = False,
        filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process traffic data for ML models.
        
        Args:
            data: Raw traffic data
            aggregate_window: Time window for aggregation (pandas offset string)
            min_samples: Minimum number of samples per time window
            save: Whether to save processed data
            filename: Filename for saving (without extension)
            
        Returns:
            Processed DataFrame ready for ML models
        """
        if data.empty:
            logger.warning("Empty dataframe provided to processor")
            return pd.DataFrame()
        
        # Ensure timestamp is datetime
        if 'timestamp' in data.columns:
            if pd.api.types.is_numeric_dtype(data['timestamp']):
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
            else:
                data['datetime'] = pd.to_datetime(data['timestamp'])
        else:
            logger.warning("No timestamp column found in data")
            return pd.DataFrame()
        
        # Aggregate data by time window
        result = self._aggregate_by_time(data, aggregate_window, min_samples)
        
        # Extract time features
        if self.time_features:
            result = self._extract_time_features(result)
        
        # Create endpoint features
        if self.endpoint_features and 'endpoint' in data.columns:
            result = self._create_endpoint_features(data, result, aggregate_window)
        
        # Create user features
        if self.user_features and 'user_id' in data.columns and 'user_type' in data.columns:
            result = self._create_user_features(data, result, aggregate_window)
        
        # Normalize numerical features
        if self.normalize:
            result = self._normalize_features(result)
        
        # Save processed data if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_{timestamp}"
            
            output_path = self.output_dir / f"{filename}.csv"
            result.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        
        return result
    
    def _aggregate_by_time(
        self,
        data: pd.DataFrame,
        aggregate_window: str,
        min_samples: int
    ) -> pd.DataFrame:
        """
        Aggregate data by time window.
        
        Args:
            data: Raw traffic data
            aggregate_window: Time window for aggregation
            min_samples: Minimum number of samples per window
            
        Returns:
            Aggregated DataFrame
        """
        # Set datetime as index for resampling
        data_indexed = data.set_index('datetime')
        
        # Define aggregation functions
        agg_funcs = {
            'timestamp': ['min', 'max'],  # Keep original timestamp range
            'response_time': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'tokens_consumed': ['sum', 'mean'],
            'tokens_remaining': ['mean', 'min'],
            'error': ['sum', 'mean'],
            'rate_limited': ['sum', 'mean'],
            'status_code': lambda x: (x == 200).mean()  # Percentage of 200 responses
        }
        
        # Only include columns that exist in the data
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in data.columns}
        
        # Resample and aggregate
        aggregated = data_indexed.resample(aggregate_window).agg(agg_funcs)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        
        # Reset index to get datetime as a column
        aggregated = aggregated.reset_index()
        
        # Filter out windows with too few samples
        if 'response_time_count' in aggregated.columns:
            aggregated = aggregated[aggregated['response_time_count'] >= min_samples]
        
        # Calculate request rate (requests per second)
        if 'timestamp_max' in aggregated.columns and 'timestamp_min' in aggregated.columns:
            # Calculate window duration in seconds
            aggregated['window_duration'] = aggregated['timestamp_max'] - aggregated['timestamp_min']
            
            # Avoid division by zero
            aggregated['window_duration'] = aggregated['window_duration'].replace(0, 1)
            
            # Calculate request rate
            if 'response_time_count' in aggregated.columns:
                aggregated['request_rate'] = aggregated['response_time_count'] / aggregated['window_duration']
        
        return aggregated
    
    def _extract_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from datetime.
        
        Args:
            data: Aggregated data with datetime column
            
        Returns:
            DataFrame with time features added
        """
        result = data.copy()
        
        # Extract basic time components
        result['hour'] = result['datetime'].dt.hour
        result['day_of_week'] = result['datetime'].dt.dayofweek
        result['day_of_month'] = result['datetime'].dt.day
        result['month'] = result['datetime'].dt.month
        result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
        
        # Create cyclical features for time
        # This helps ML models understand the cyclical nature of time
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        return result
    
    def _create_endpoint_features(
        self,
        raw_data: pd.DataFrame,
        aggregated_data: pd.DataFrame,
        aggregate_window: str
    ) -> pd.DataFrame:
        """
        Create features related to API endpoints.
        
        Args:
            raw_data: Raw traffic data with endpoint information
            aggregated_data: Time-aggregated data
            aggregate_window: Time window used for aggregation
            
        Returns:
            DataFrame with endpoint features added
        """
        result = aggregated_data.copy()
        
        # Create a copy of raw data with datetime
        raw_with_dt = raw_data.copy()
        
        # Group by datetime window and endpoint
        raw_with_dt['time_bucket'] = raw_with_dt['datetime'].dt.floor(aggregate_window)
        
        # Count requests by endpoint type
        endpoint_counts = raw_with_dt.groupby(['time_bucket', 'endpoint']).size().unstack(fill_value=0)
        
        # Rename columns to indicate they're endpoint counts
        endpoint_counts.columns = [f'endpoint_{col}_count' for col in endpoint_counts.columns]
        
        # Reset index to get time_bucket as a column
        endpoint_counts = endpoint_counts.reset_index()
        
        # Merge with result
        result = pd.merge(
            result, 
            endpoint_counts, 
            left_on='datetime', 
            right_on='time_bucket', 
            how='left'
        )
        
        # Drop the time_bucket column
        if 'time_bucket' in result.columns:
            result = result.drop('time_bucket', axis=1)
        
        # Fill NaN values in endpoint count columns
        for col in endpoint_counts.columns:
            if col != 'time_bucket' and col in result.columns:
                result[col] = result[col].fillna(0)
        
        # Calculate percentage of requests by endpoint
        if 'response_time_count' in result.columns:
            for col in endpoint_counts.columns:
                if col != 'time_bucket' and col in result.columns:
                    result[f'{col}_pct'] = result[col] / result['response_time_count']
                    result[f'{col}_pct'] = result[f'{col}_pct'].fillna(0)
        
        return result
    
    def _create_user_features(
        self,
        raw_data: pd.DataFrame,
        aggregated_data: pd.DataFrame,
        aggregate_window: str
    ) -> pd.DataFrame:
        """
        Create features related to users.
        
        Args:
            raw_data: Raw traffic data with user information
            aggregated_data: Time-aggregated data
            aggregate_window: Time window used for aggregation
            
        Returns:
            DataFrame with user features added
        """
        result = aggregated_data.copy()
        
        # Create a copy of raw data with datetime
        raw_with_dt = raw_data.copy()
        
        # Group by datetime window and user type
        raw_with_dt['time_bucket'] = raw_with_dt['datetime'].dt.floor(aggregate_window)
        
        # Count users by type
        user_type_counts = raw_with_dt.groupby(['time_bucket', 'user_type']).size().unstack(fill_value=0)
        
        # Rename columns to indicate they're user type counts
        user_type_counts.columns = [f'user_type_{col}_count' for col in user_type_counts.columns]
        
        # Reset index to get time_bucket as a column
        user_type_counts = user_type_counts.reset_index()
        
        # Merge with result
        result = pd.merge(
            result, 
            user_type_counts, 
            left_on='datetime', 
            right_on='time_bucket', 
            how='left'
        )
        
        # Drop the time_bucket column
        if 'time_bucket' in result.columns:
            result = result.drop('time_bucket', axis=1)
        
        # Fill NaN values in user type count columns
        for col in user_type_counts.columns:
            if col != 'time_bucket' and col in result.columns:
                result[col] = result[col].fillna(0)
        
        # Calculate percentage of requests by user type
        if 'response_time_count' in result.columns:
            for col in user_type_counts.columns:
                if col != 'time_bucket' and col in result.columns:
                    result[f'{col}_pct'] = result[col] / result['response_time_count']
                    result[f'{col}_pct'] = result[f'{col}_pct'].fillna(0)
        
        # Count unique users
        unique_users = raw_with_dt.groupby('time_bucket')['user_id'].nunique().reset_index()
        unique_users.columns = ['time_bucket', 'unique_users']
        
        # Merge with result
        result = pd.merge(
            result, 
            unique_users, 
            left_on='datetime', 
            right_on='time_bucket', 
            how='left'
        )
        
        # Drop the time_bucket column
        if 'time_bucket' in result.columns:
            result = result.drop('time_bucket', axis=1)
        
        # Fill NaN values
        if 'unique_users' in result.columns:
            result['unique_users'] = result['unique_users'].fillna(0)
        
        return result
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features using Min-Max scaling.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        result = data.copy()
        
        # Identify numerical columns (excluding datetime and categorical columns)
        excluded_cols = ['datetime', 'timestamp_min', 'timestamp_max']
        categorical_cols = []
        
        numerical_cols = [
            col for col in result.columns 
            if col not in excluded_cols 
            and col not in categorical_cols
            and pd.api.types.is_numeric_dtype(result[col])
        ]
        
        # Normalize each numerical column
        for col in numerical_cols:
            if col not in self.scalers:
                self.scalers[col] = MinMaxScaler()
                result[f'{col}_norm'] = self.scalers[col].fit_transform(
                    result[col].values.reshape(-1, 1)
                ).flatten()
            else:
                result[f'{col}_norm'] = self.scalers[col].transform(
                    result[col].values.reshape(-1, 1)
                ).flatten()
        
        return result
    
    def save_scalers(self, path: str) -> None:
        """
        Save scalers for later use.
        
        Args:
            path: Path to save scalers
        """
        import joblib
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save scalers
        joblib.dump(self.scalers, path)
        logger.info(f"Saved scalers to {path}")
    
    def load_scalers(self, path: str) -> None:
        """
        Load scalers from disk.
        
        Args:
            path: Path to load scalers from
        """
        import joblib
        
        # Load scalers
        self.scalers = joblib.load(path)
        logger.info(f"Loaded scalers from {path}")


def process_traffic_data(
    data: pd.DataFrame,
    aggregate_window: str = '1min',
    output_dir: Optional[str] = None,
    save: bool = False,
    filename: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Process traffic data for ML models.
    
    Args:
        data: Raw traffic data
        aggregate_window: Time window for aggregation
        output_dir: Directory to save processed data
        save: Whether to save processed data
        filename: Filename for saving
        **kwargs: Additional arguments to pass to TrafficProcessor
        
    Returns:
        Processed DataFrame
    """
    processor = TrafficProcessor(output_dir=output_dir or "./data/processed", **kwargs)
    
    return processor.process(
        data=data,
        aggregate_window=aggregate_window,
        save=save,
        filename=filename
    )


def aggregate_traffic_data(
    data: pd.DataFrame,
    aggregate_window: str = '1min',
    min_samples: int = 10
) -> pd.DataFrame:
    """
    Aggregate traffic data by time window.
    
    Args:
        data: Raw traffic data
        aggregate_window: Time window for aggregation
        min_samples: Minimum number of samples per window
        
    Returns:
        Aggregated DataFrame
    """
    processor = TrafficProcessor(time_features=False, endpoint_features=False, 
                                user_features=False, normalize=False)
    
    return processor._aggregate_by_time(
        data=data,
        aggregate_window=aggregate_window,
        min_samples=min_samples
    )


def split_train_val_test(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data: Processed traffic data
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        time_based: Whether to split based on time (sequential) or randomly
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Sort data by datetime if time-based split
    if time_based and 'datetime' in data.columns:
        data = data.sort_values('datetime')
    
    # Calculate split indices
    n = len(data)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)
    
    if time_based:
        # Sequential split
        train_data = data.iloc[:train_idx]
        val_data = data.iloc[train_idx:val_idx]
        test_data = data.iloc[val_idx:]
    else:
        # Random split
        train_data, temp_data = train_test_split(data, test_size=(val_ratio + test_ratio), random_state=42)
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42
        )
    
    return train_data, val_data, test_data


# Example usage if run directly
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create some sample traffic data
    n_samples = 1000
    timestamps = np.array([time.time() - i * 60 for i in range(n_samples)])
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'user_id': [f"user_{i % 10}" for i in range(n_samples)],
        'user_type': ["STD" if i % 3 != 0 else "PRM" for i in range(n_samples)],
        'endpoint': [f"/api/resource/{i % 5}" for i in range(n_samples)],
        'response_time': np.random.gamma(shape=2, scale=0.05, size=n_samples),
        'status_code': [200 if i % 10 != 0 else (429 if i % 20 == 0 else 500) for i in range(n_samples)],
        'tokens_consumed': [1 if i % 10 != 0 else 0 for i in range(n_samples)],
        'tokens_remaining': [np.random.randint(1, 20) if i % 10 != 0 else 0 for i in range(n_samples)],
        'error': [i % 10 == 0 and i % 20 != 0 for i in range(n_samples)],
        'rate_limited': [i % 20 == 0 for i in range(n_samples)]
    })
    
    # Process data
    processor = TrafficProcessor()
    processed_data = processor.process(data, aggregate_window='5min', save=True)
    
    print(f"Raw data shape: {data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    
    # Print processed data columns
    print("\nProcessed data columns:")
    for col in processed_data.columns:
        print(f"- {col}")
    
    # Split data
    train_data, val_data, test_data = split_train_val_test(processed_data)
    
    print(f"\nData split:")
    print(f"- Train: {train_data.shape}")
    print(f"- Validation: {val_data.shape}")
    print(f"- Test: {test_data.shape}")
    
    print("\nTraffic processor demo complete")
