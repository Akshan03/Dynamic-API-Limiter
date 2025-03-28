import os
import time
import logging
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
from functools import partial

# Import components from other modules
from ml.data.collector import TrafficCollector, get_recent_traffic_data
from ml.data.processor import TrafficProcessor, process_traffic_data, split_train_val_test
from ml.models.traffic_predictor import (
    TrafficPredictor, 
    train_traffic_predictor,
    load_traffic_predictor, 
    predict_traffic,
    visualize_traffic_prediction
)
from ml.models.anomaly_detector import (
    AnomalyDetector,
    train_anomaly_detector, 
    load_anomaly_detector, 
    detect_anomalies,
    visualize_anomalies
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml_trainer")

class ModelTrainer:
    """
    Coordinates the training and evaluation of ML models for rate limiting.
    Handles data preparation, model training, evaluation, and scheduling.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        models_dir: str = "./data/models",
        training_interval_hours: int = 24,  # Train every 24 hours
        min_training_samples: int = 1000,   # Minimum samples needed for training
        traffic_collector: Optional[TrafficCollector] = None,
        traffic_processor: Optional[TrafficProcessor] = None
    ):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Base directory for data (raw and processed)
            models_dir: Directory to store trained models
            training_interval_hours: How often to retrain models
            min_training_samples: Minimum samples needed for training
            traffic_collector: Optional existing collector instance
            traffic_processor: Optional existing processor instance
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.models_dir = Path(models_dir)
        self.training_interval_hours = training_interval_hours
        self.min_training_samples = min_training_samples
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data components
        self.traffic_collector = traffic_collector or TrafficCollector(
            data_dir=str(self.raw_data_dir)
        )
        
        self.traffic_processor = traffic_processor or TrafficProcessor(
            output_dir=str(self.processed_data_dir)
        )
        
        # Initialize models
        self.traffic_predictor = None
        self.anomaly_detector = None
        
        # Training state
        self.last_training_time = None
        self.is_training = False
        self.training_thread = None
        self.training_scheduled = False
        
        # Training metrics
        self.training_history = {
            "traffic_predictor": [],
            "anomaly_detector": []
        }
        
        logger.info(f"Model trainer initialized with data directory: {self.data_dir}")
        
        # Try to load existing models
        self._load_existing_models()
    
    def _load_existing_models(self) -> None:
        """Load existing models if available."""
        # Check for traffic predictor
        traffic_predictor_path = self.models_dir / "traffic_predictor"
        if (traffic_predictor_path.with_suffix(".h5").exists() and 
            traffic_predictor_path.with_suffix("_config.joblib").exists()):
            try:
                self.traffic_predictor = load_traffic_predictor(str(traffic_predictor_path))
                logger.info("Loaded existing traffic predictor model")
            except Exception as e:
                logger.error(f"Error loading traffic predictor: {e}")
        
        # Check for anomaly detector
        anomaly_detector_path = self.models_dir / "anomaly_detector"
        if (anomaly_detector_path.with_suffix(".joblib").exists() and 
            anomaly_detector_path.with_suffix("_config.joblib").exists()):
            try:
                self.anomaly_detector = load_anomaly_detector(str(anomaly_detector_path))
                logger.info("Loaded existing anomaly detector model")
            except Exception as e:
                logger.error(f"Error loading anomaly detector: {e}")
    
    async def prepare_training_data(
        self,
        days_to_collect: int = 7,
        aggregate_window: str = '5min',
        min_samples_per_window: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training.
        
        Args:
            days_to_collect: How many days of data to collect
            aggregate_window: Time window for aggregation
            min_samples_per_window: Minimum samples per time window
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Calculate time range
        end_time = time.time()
        start_time = end_time - (days_to_collect * 24 * 60 * 60)
        
        # Get raw data
        raw_data = self.traffic_collector.get_data_since(start_time, end_time)
        
        if len(raw_data) < self.min_training_samples:
            logger.warning(f"Insufficient data for training: {len(raw_data)} samples < {self.min_training_samples} required")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Collected {len(raw_data)} samples for training")
        
        # Process data
        processed_data = self.traffic_processor.process(
            data=raw_data,
            aggregate_window=aggregate_window,
            min_samples=min_samples_per_window,
            save=True,
            filename=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Split data
        train_data, val_data, test_data = split_train_val_test(
            processed_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            time_based=True
        )
        
        logger.info(f"Prepared data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    async def train_traffic_predictor(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        lookback_window: int = 12,  # 12 time steps lookback
        prediction_horizon: int = 6,  # Predict 6 time steps ahead
        feature_columns: Optional[List[str]] = None,
        save: bool = True
    ) -> TrafficPredictor:
        """
        Train the traffic predictor model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            lookback_window: Number of time steps to use as input
            prediction_horizon: Number of time steps to predict
            feature_columns: List of columns to use as features
            save: Whether to save the trained model
            
        Returns:
            Trained TrafficPredictor instance
        """
        if train_data.empty or val_data.empty or test_data.empty:
            logger.error("Cannot train traffic predictor: empty dataset")
            return None
        
        # Determine feature columns if not provided
        if feature_columns is None:
            # Use columns with _norm suffix (normalized features)
            norm_columns = [col for col in train_data.columns if col.endswith('_norm')]
            
            # If no normalized features, use numerical columns excluding timestamp and datetime
            if not norm_columns:
                feature_columns = [
                    col for col in train_data.columns 
                    if col not in ['timestamp', 'datetime', 'timestamp_min', 'timestamp_max'] 
                    and pd.api.types.is_numeric_dtype(train_data[col])
                ]
            else:
                feature_columns = norm_columns
        
        # Add timestamp column for sequence ordering
        if 'timestamp' not in feature_columns and 'timestamp' in train_data.columns:
            feature_columns.append('timestamp')
        
        logger.info(f"Training traffic predictor with {len(feature_columns)} features")
        
        # Train the model
        model_path = str(self.models_dir / "traffic_predictor") if save else None
        
        predictor = train_traffic_predictor(
            train_data=train_data,
            validation_data=val_data,
            test_data=test_data,
            lookback_window=lookback_window,
            prediction_horizon=prediction_horizon,
            feature_columns=feature_columns,
            save_path=model_path
        )
        
        # Create visualization
        if save:
            try:
                # Generate visualization using last part of test data
                recent_data = test_data.iloc[-lookback_window*2:]
                predictions = predictor.predict(recent_data, return_dataframe=True)
                
                # Only create visualization if there are multiple features
                for feature in feature_columns:
                    if feature != 'timestamp' and feature in predictions.columns:
                        fig = visualize_traffic_prediction(
                            historical_data=recent_data,
                            predicted_data=predictions,
                            feature_to_plot=feature,
                            title=f'Traffic Prediction: {feature}'
                        )
                        
                        # Save figure
                        fig_path = self.models_dir / f"traffic_prediction_{feature.split('_')[0]}.png"
                        fig.savefig(str(fig_path))
                        plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating traffic prediction visualization: {e}")
        
        # Update instance
        self.traffic_predictor = predictor
        
        # Record training metrics
        training_metrics = {
            "timestamp": time.time(),
            "num_samples": len(train_data) + len(val_data) + len(test_data),
            "features": feature_columns,
            "lookback_window": lookback_window,
            "prediction_horizon": prediction_horizon
        }
        
        if predictor.train_history is not None:
            # Add loss metrics if available
            training_metrics["train_loss"] = predictor.train_history.get('loss', [])[-1] if predictor.train_history.get('loss') else None
            training_metrics["val_loss"] = predictor.train_history.get('val_loss', [])[-1] if predictor.train_history.get('val_loss') else None
            training_metrics["test_loss"] = predictor.test_loss
        
        self.training_history["traffic_predictor"].append(training_metrics)
        
        logger.info("Traffic predictor training complete")
        return predictor
    
    async def train_anomaly_detector(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        contamination: float = 0.05,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        save: bool = True
    ) -> AnomalyDetector:
        """
        Train the anomaly detector model.
        
        Args:
            train_data: Training data
            test_data: Test data
            contamination: Expected proportion of anomalies
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            save: Whether to save the trained model
            
        Returns:
            Trained AnomalyDetector instance
        """
        if train_data.empty or test_data.empty:
            logger.error("Cannot train anomaly detector: empty dataset")
            return None
        
        # Determine categorical features if not provided
        if categorical_features is None:
            categorical_features = [
                col for col in train_data.columns 
                if col not in ['timestamp', 'datetime', 'timestamp_min', 'timestamp_max']
                and not pd.api.types.is_numeric_dtype(train_data[col]) 
                or (pd.api.types.is_numeric_dtype(train_data[col]) and train_data[col].nunique() < 10)
            ]
        
        # Determine numerical features if not provided
        if numerical_features is None:
            # Use columns with _norm suffix (normalized features)
            norm_columns = [col for col in train_data.columns if col.endswith('_norm')]
            
            # If no normalized features, use numerical columns excluding timestamp and datetime
            if not norm_columns:
                numerical_features = [
                    col for col in train_data.columns 
                    if col not in ['timestamp', 'datetime', 'timestamp_min', 'timestamp_max'] 
                    and col not in categorical_features
                    and pd.api.types.is_numeric_dtype(train_data[col])
                ]
            else:
                numerical_features = norm_columns
        
        logger.info(f"Training anomaly detector with {len(categorical_features)} categorical features and {len(numerical_features)} numerical features")
        
        # Train the model
        model_path = str(self.models_dir / "anomaly_detector") if save else None
        
        detector = train_anomaly_detector(
            data=train_data,
            contamination=contamination,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            save_path=model_path
        )
        
        # Create visualizations
        if save:
            try:
                # Detect anomalies in test data
                test_results = detector.detect_anomalies(test_data)
                
                # Create visualization for each numeric feature vs time
                time_col = 'datetime' if 'datetime' in test_data.columns else 'timestamp'
                
                for feature in numerical_features[:5]:  # Limit to first 5 features to avoid too many plots
                    fig = visualize_anomalies(
                        data=test_results,
                        feature_x=feature,
                        time_feature=time_col,
                        title=f'Anomaly Detection: {feature}'
                    )
                    
                    # Save figure
                    fig_path = self.models_dir / f"anomaly_detection_{feature.split('_')[0]}.png"
                    fig.savefig(str(fig_path))
                    plt.close(fig)
                
                # Create 2D visualization with the two most important features
                if len(numerical_features) >= 2:
                    fig = visualize_anomalies(
                        data=test_results,
                        feature_x=numerical_features[0],
                        feature_y=numerical_features[1],
                        time_feature=None,  # No time for 2D visualization
                        title=f'Anomaly Detection: {numerical_features[0]} vs {numerical_features[1]}'
                    )
                    
                    # Save figure
                    fig_path = self.models_dir / "anomaly_detection_2d.png"
                    fig.savefig(str(fig_path))
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating anomaly detection visualization: {e}")
        
        # Update instance
        self.anomaly_detector = detector
        
        # Record training metrics
        feature_importances = detector.feature_importances or {}
        top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        
        training_metrics = {
            "timestamp": time.time(),
            "num_samples": len(train_data) + len(test_data),
            "contamination": contamination,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "top_features": dict(top_features) if top_features else None
        }
        
        self.training_history["anomaly_detector"].append(training_metrics)
        
        logger.info("Anomaly detector training complete")
        return detector
    
    async def train_models(
        self,
        days_to_collect: int = 7,
        aggregate_window: str = '5min',
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train all ML models.
        
        Args:
            days_to_collect: How many days of data to collect
            aggregate_window: Time window for aggregation
            force_retrain: Whether to force retraining even if recently trained
            
        Returns:
            Dictionary with training results
        """
        # Check if training is already in progress
        if self.is_training:
            logger.warning("Training already in progress, skipping")
            return {"status": "skipped", "reason": "training_in_progress"}
        
        # Check if recently trained and not forced
        if not force_retrain and self.last_training_time:
            hours_since_last_training = (time.time() - self.last_training_time) / 3600
            if hours_since_last_training < self.training_interval_hours:
                logger.info(f"Models were trained {hours_since_last_training:.1f} hours ago, skipping")
                return {"status": "skipped", "reason": "recently_trained"}
        
        # Set training flag
        self.is_training = True
        
        try:
            # Prepare data
            train_data, val_data, test_data = await self.prepare_training_data(
                days_to_collect=days_to_collect,
                aggregate_window=aggregate_window
            )
            
            if train_data.empty:
                logger.warning("Insufficient data for training")
                self.is_training = False
                return {"status": "failed", "reason": "insufficient_data"}
            
            # Train traffic predictor
            traffic_predictor = await self.train_traffic_predictor(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )
            
            # Train anomaly detector
            anomaly_detector = await self.train_anomaly_detector(
                train_data=train_data,
                test_data=test_data
            )
            
            # Update last training time
            self.last_training_time = time.time()
            
            # Save training history
            self._save_training_history()
            
            # Training successful
            logger.info("Model training completed successfully")
            return {
                "status": "success",
                "timestamp": self.last_training_time,
                "models_trained": {
                    "traffic_predictor": traffic_predictor is not None,
                    "anomaly_detector": anomaly_detector is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}
            
        finally:
            # Reset training flag
            self.is_training = False
    
    def _save_training_history(self) -> None:
        """Save training history to disk."""
        try:
            history_path = self.models_dir / "training_history.json"
            
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
                
            logger.info(f"Saved training history to {history_path}")
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
    
    async def evaluate_models(
        self,
        recent_hours: int = 24,
        aggregate_window: str = '5min'
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on recent data.
        
        Args:
            recent_hours: Hours of recent data to evaluate
            aggregate_window: Time window for aggregation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if models are trained
        if self.traffic_predictor is None or self.anomaly_detector is None:
            logger.warning("Models not trained, cannot evaluate")
            return {"status": "failed", "reason": "models_not_trained"}
        
        try:
            # Get recent data
            end_time = time.time()
            start_time = end_time - (recent_hours * 60 * 60)
            
            raw_data = self.traffic_collector.get_data_since(start_time, end_time)
            
            if len(raw_data) < 100:  # Need at least some data to evaluate
                logger.warning(f"Insufficient data for evaluation: {len(raw_data)} samples")
                return {"status": "failed", "reason": "insufficient_data"}
            
            # Process data
            processed_data = self.traffic_processor.process(
                data=raw_data,
                aggregate_window=aggregate_window
            )
            
            # Evaluate traffic predictor
            traffic_metrics = await self._evaluate_traffic_predictor(processed_data)
            
            # Evaluate anomaly detector
            anomaly_metrics = await self._evaluate_anomaly_detector(processed_data)
            
            return {
                "status": "success",
                "timestamp": time.time(),
                "data_points": len(processed_data),
                "metrics": {
                    "traffic_predictor": traffic_metrics,
                    "anomaly_detector": anomaly_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}
    
    async def _evaluate_traffic_predictor(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate traffic predictor performance.
        
        Args:
            data: Recent processed data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.traffic_predictor is None:
            return {"error": "model_not_trained"}
        
        try:
            # Get feature columns from model
            feature_columns = self.traffic_predictor.feature_columns
            
            # Check if all required features exist in data
            missing_features = [col for col in feature_columns if col not in data.columns]
            if missing_features:
                return {"error": f"missing_features: {missing_features}"}
            
            # Prepare validation data - use last part of data for testing
            lookback = self.traffic_predictor.lookback_window
            horizon = self.traffic_predictor.prediction_horizon
            
            if len(data) < lookback + horizon:
                return {"error": "insufficient_data_points"}
            
            # Split into validation sets
            num_validation_sets = min(5, len(data) // (lookback + horizon))
            
            metrics = []
            
            for i in range(num_validation_sets):
                # Use different segments of data for validation
                start_idx = i * (lookback + horizon)
                if start_idx + lookback + horizon > len(data):
                    break
                    
                input_data = data.iloc[start_idx:start_idx + lookback]
                actual_data = data.iloc[start_idx + lookback:start_idx + lookback + horizon]
                
                # Make prediction
                predicted_data = self.traffic_predictor.predict(input_data, return_dataframe=True)
                
                # Calculate error for each feature
                errors = {}
                for feature in [col for col in feature_columns if col != 'timestamp']:
                    if feature in predicted_data.columns and feature in actual_data.columns:
                        # Mean Absolute Error
                        mae = np.mean(np.abs(predicted_data[feature].values - actual_data[feature].values))
                        # Mean Absolute Percentage Error
                        mape = np.mean(np.abs((predicted_data[feature].values - actual_data[feature].values) / 
                                            actual_data[feature].values)) * 100 if not np.isclose(actual_data[feature].values, 0).any() else np.nan
                        
                        errors[feature] = {
                            "mae": float(mae),
                            "mape": float(mape) if not np.isnan(mape) else None
                        }
                
                metrics.append({
                    "validation_set": i,
                    "errors": errors
                })
            
            # Aggregate metrics across validation sets
            aggregated_metrics = {"features": {}}
            
            if metrics:
                for feature in metrics[0]["errors"].keys():
                    feature_maes = [m["errors"][feature]["mae"] for m in metrics]
                    feature_mapes = [m["errors"][feature]["mape"] for m in metrics if m["errors"][feature]["mape"] is not None]
                    
                    aggregated_metrics["features"][feature] = {
                        "mae": {
                            "mean": float(np.mean(feature_maes)),
                            "min": float(np.min(feature_maes)),
                            "max": float(np.max(feature_maes))
                        }
                    }
                    
                    if feature_mapes:
                        aggregated_metrics["features"][feature]["mape"] = {
                            "mean": float(np.mean(feature_mapes)),
                            "min": float(np.min(feature_mapes)),
                            "max": float(np.max(feature_mapes))
                        }
            
            return {
                "status": "success",
                "aggregated_metrics": aggregated_metrics,
                "validation_sets": len(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating traffic predictor: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def _evaluate_anomaly_detector(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly detector performance.
        
        Args:
            data: Recent processed data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.anomaly_detector is None:
            return {"error": "model_not_trained"}
        
        try:
            # Get required features
            categorical_features = self.anomaly_detector.categorical_features
            numerical_features = self.anomaly_detector.numerical_features
            
            # Check if all required features exist in data
            missing_categorical = [col for col in categorical_features if col not in data.columns]
            missing_numerical = [col for col in numerical_features if col not in data.columns]
            
            if missing_categorical or missing_numerical:
                return {
                    "error": "missing_features",
                    "missing_categorical": missing_categorical,
                    "missing_numerical": missing_numerical
                }
            
            # Detect anomalies
            results = self.anomaly_detector.detect_anomalies(data)
            
            # Calculate metrics
            anomaly_count = results['is_anomaly'].sum()
            anomaly_rate = float(anomaly_count / len(results)) if len(results) > 0 else 0
            
            # Get anomaly scores distribution
            if 'anomaly_score' in results.columns:
                scores = results['anomaly_score'].dropna()
                score_metrics = {
                    "mean": float(scores.mean()) if not scores.empty else None,
                    "min": float(scores.min()) if not scores.empty else None,
                    "max": float(scores.max()) if not scores.empty else None,
                    "percentiles": {
                        "1": float(np.percentile(scores, 1)) if not scores.empty else None,
                        "5": float(np.percentile(scores, 5)) if not scores.empty else None,
                        "25": float(np.percentile(scores, 25)) if not scores.empty else None,
                        "50": float(np.percentile(scores, 50)) if not scores.empty else None,
                        "75": float(np.percentile(scores, 75)) if not scores.empty else None,
                        "95": float(np.percentile(scores, 95)) if not scores.empty else None,
                        "99": float(np.percentile(scores, 99)) if not scores.empty else None
                    }
                }
            else:
                score_metrics = None
            
            return {
                "status": "success",
                "data_points": len(results),
                "anomaly_count": int(anomaly_count),
                "anomaly_rate": anomaly_rate,
                "score_metrics": score_metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating anomaly detector: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def predict_rate_limit_parameters(
        self,
        user_type: str,
        recent_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Predict optimal rate limit parameters based on traffic prediction.
        
        Args:
            user_type: User type (STD or PRM)
            recent_minutes: Minutes of recent data to use for prediction
            
        Returns:
            Dictionary with rate limit parameters
        """
        if self.traffic_predictor is None:
            logger.warning("Traffic predictor not trained, using default parameters")
            return {
                "status": "default",
                "parameters": _get_default_parameters(user_type)
            }
        
        try:
            # Get recent data
            recent_data = await get_recent_traffic_data(
                minutes=recent_minutes,
                aggregate=True,
                interval_seconds=60,
                collector=self.traffic_collector
            )
            
            if recent_data.empty:
                logger.warning("No recent data available for prediction")
                return {
                    "status": "default",
                    "parameters": _get_default_parameters(user_type)
                }
            
            # Process data
            processed_data = self.traffic_processor.process(
                data=recent_data,
                aggregate_window='1min'
            )
            
            # Make prediction
            prediction = await predict_traffic(
                recent_data=processed_data,
                predictor=self.traffic_predictor
            )
            
            # Calculate parameters based on prediction
            parameters = self._calculate_rate_limit_parameters(
                prediction=prediction,
                user_type=user_type
            )
            
            return {
                "status": "success",
                "parameters": parameters,
                "prediction_horizon_minutes": self.traffic_predictor.prediction_horizon
            }
            
        except Exception as e:
            logger.error(f"Error predicting rate limit parameters: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "parameters": _get_default_parameters(user_type)
            }
    
    def _calculate_rate_limit_parameters(
        self,
        prediction: pd.DataFrame,
        user_type: str
    ) -> Dict[str, int]:
        """
        Calculate rate limit parameters based on traffic prediction.
        
        Args:
            prediction: DataFrame with traffic predictions
            user_type: User type (STD or PRM)
            
        Returns:
            Dictionary with rate limit parameters (capacity, refill_rate)
        """
        # Default multipliers for different user types
        std_multiplier = 1.0
        prm_multiplier = 3.0
        
        # Extract predicted request rates if available
        if 'request_rate' in prediction.columns:
            predicted_rates = prediction['request_rate'].values
        elif 'request_rate_norm' in prediction.columns and 'request_rate' in self.traffic_predictor.feature_scalers:
            # Inverse transform normalized values
            scaler = self.traffic_predictor.feature_scalers['request_rate']
            predicted_rates = scaler.inverse_transform(
                prediction['request_rate_norm'].values.reshape(-1, 1)
            ).flatten()
        elif 'response_time_count' in prediction.columns and 'window_duration' in prediction.columns:
            # Calculate from count and duration
            predicted_rates = prediction['response_time_count'] / prediction['window_duration']
        else:
            # No suitable rate column found
            logger.warning("No suitable rate column found in prediction")
            return _get_default_parameters(user_type)
        
        # Calculate statistics
        avg_rate = np.mean(predicted_rates)
        peak_rate = np.max(predicted_rates)
        
        # Calculate parameters based on user type
        if user_type.upper() == 'STD':
            multiplier = std_multiplier
        else:  # PRM
            multiplier = prm_multiplier
        
        # Adjust based on predicted traffic
        capacity = int(max(10, min(200, multiplier * (avg_rate + 0.5 * (peak_rate - avg_rate)))))
        refill_rate = int(max(1, min(30, multiplier * 0.8 * avg_rate)))
        
        return {
            "capacity": capacity,
            "refill_rate": refill_rate
        }
    
    async def detect_traffic_anomalies(
        self,
        recent_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Detect anomalies in recent traffic.
        
        Args:
            recent_minutes: Minutes of recent data to check
            
        Returns:
            Dictionary with anomaly detection results
        """
        if self.anomaly_detector is None:
            logger.warning("Anomaly detector not trained, cannot detect anomalies")
            return {"status": "error", "reason": "model_not_trained"}
        
        try:
            # Get recent data
            recent_data = await get_recent_traffic_data(
                minutes=recent_minutes,
                aggregate=True,
                interval_seconds=60,
                collector=self.traffic_collector
            )
            
            if recent_data.empty:
                logger.warning("No recent data available for anomaly detection")
                return {"status": "error", "reason": "no_recent_data"}
            
            # Process data
            processed_data = self.traffic_processor.process(
                data=recent_data,
                aggregate_window='1min'
            )
            
            # Detect anomalies
            results = await detect_anomalies(
                data=processed_data,
                detector=self.anomaly_detector
            )
            
            # Extract anomalies
            anomalies = results[results['is_anomaly']]
            
            if anomalies.empty:
                return {
                    "status": "success",
                    "anomaly_detected": False,
                    "data_points": len(results)
                }
            
            # Prepare anomaly information
            anomaly_info = {
                "status": "success",
                "anomaly_detected": True,
                "data_points": len(results),
                "anomaly_count": len(anomalies),
                "anomaly_percentage": len(anomalies) / len(results) * 100,
                "recent_anomalies": []
            }
            
            # Add information about recent anomalies
            for _, row in anomalies.tail(5).iterrows():
                anomaly_data = {"timestamp": row.get('timestamp')}
                
                # Add key metrics
                for col in ['request_rate', 'error_rate', 'response_time_mean', 'rate_limited_sum']:
                    if col in row:
                        anomaly_data[col] = row[col]
                
                # Add anomaly score if available
                if 'anomaly_score' in row:
                    anomaly_data['anomaly_score'] = row['anomaly_score']
                
                anomaly_info["recent_anomalies"].append(anomaly_data)
            
            return anomaly_info
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def start_periodic_training(self) -> None:
        """Start periodic model training in background."""
        if self.training_scheduled:
            logger.warning("Periodic training already scheduled")
            return
        
        self.training_scheduled = True
        
        # Start training thread
        def training_loop():
            while self.training_scheduled:
                try:
                    # Run training (need to use asyncio.run for async function)
                    asyncio.run(self.train_models())
                    
                    # Sleep until next training interval
                    sleep_time = self.training_interval_hours * 3600
                    logger.info(f"Next training scheduled in {self.training_interval_hours} hours")
                    
                    # Sleep in smaller chunks to allow for clean shutdown
                    for _ in range(int(sleep_time / 60)):
                        if not self.training_scheduled:
                            break
                        time.sleep(60)
                        
                except Exception as e:
                    logger.error(f"Error in training loop: {e}", exc_info=True)
                    time.sleep(600)  # Sleep for 10 minutes on error
        
        # Start thread
        self.training_thread = threading.Thread(
            target=training_loop,
            daemon=True
        )
        self.training_thread.start()
        
        logger.info("Periodic training started")
    
    def stop_periodic_training(self) -> None:
        """Stop periodic model training."""
        self.training_scheduled = False
        
        if self.training_thread:
            self.training_thread.join(timeout=10.0)
            self.training_thread = None
            
        logger.info("Periodic training stopped")


# Helper functions for simpler usage

def _get_default_parameters(user_type: str) -> Dict[str, int]:
    """Get default rate limit parameters for a user type."""
    if user_type.upper() == 'STD':
        return {
            "capacity": 20,
            "refill_rate": 5
        }
    else:  # PRM
        return {
            "capacity": 60,
            "refill_rate": 15
        }


async def train_models(
    days_to_collect: int = 7,
    aggregate_window: str = '5min',
    data_dir: str = "./data",
    models_dir: str = "./data/models",
    force_retrain: bool = False,
    trainer: Optional[ModelTrainer] = None
) -> Dict[str, Any]:
    """
    Train all ML models.
    
    Args:
        days_to_collect: How many days of data to collect
        aggregate_window: Time window for aggregation
        data_dir: Base directory for data
        models_dir: Directory to store trained models
        force_retrain: Whether to force retraining
        trainer: Optional existing trainer instance
        
    Returns:
        Dictionary with training results
    """
    # Use existing trainer or create a new one
    if trainer is None:
        trainer = ModelTrainer(
            data_dir=data_dir,
            models_dir=models_dir
        )
    
    # Train models
    return await trainer.train_models(
        days_to_collect=days_to_collect,
        aggregate_window=aggregate_window,
        force_retrain=force_retrain
    )


async def evaluate_models(
    recent_hours: int = 24,
    aggregate_window: str = '5min',
    trainer: Optional[ModelTrainer] = None,
    data_dir: str = "./data",
    models_dir: str = "./data/models"
) -> Dict[str, Any]:
    """
    Evaluate model performance on recent data.
    
    Args:
        recent_hours: Hours of recent data to evaluate
        aggregate_window: Time window for aggregation
        trainer: Optional existing trainer instance
        data_dir: Base directory for data
        models_dir: Directory to store trained models
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Use existing trainer or create a new one
    if trainer is None:
        trainer = ModelTrainer(
            data_dir=data_dir,
            models_dir=models_dir
        )
    
    # Evaluate models
    return await trainer.evaluate_models(
        recent_hours=recent_hours,
        aggregate_window=aggregate_window
    )


def setup_periodic_training(
    training_interval_hours: int = 24,
    data_dir: str = "./data",
    models_dir: str = "./data/models"
) -> ModelTrainer:
    """
    Set up periodic model training.
    
    Args:
        training_interval_hours: How often to retrain models (in hours)
        data_dir: Base directory for data
        models_dir: Directory to store trained models
        
    Returns:
        ModelTrainer instance with periodic training started
    """
    # Create trainer
    trainer = ModelTrainer(
        data_dir=data_dir,
        models_dir=models_dir,
        training_interval_hours=training_interval_hours
    )
    
    # Start periodic training
    trainer.start_periodic_training()
    
    return trainer


# Example usage if run directly
if __name__ == "__main__":
    async def main():
        # Create trainer
        trainer = ModelTrainer()
        
        # Generate some sample data - in a real application, this would come from actual API traffic
        collector = trainer.traffic_collector
        
        # Generate sample data if no data exists
        if len(collector.get_data_since(time.time() - 86400)) < 100:
            logger.info("Generating sample data for demonstration")
            
            # Generate data over the last 3 days
            end_time = time.time()
            start_time = end_time - (3 * 24 * 60 * 60)
            
            # Create timestamps every minute
            timestamps = np.arange(start_time, end_time, 60)
            
            # Generate cyclical traffic pattern
            hours = [(datetime.fromtimestamp(ts).hour + datetime.fromtimestamp(ts).minute / 60) for ts in timestamps]
            days = [datetime.fromtimestamp(ts).weekday() for ts in timestamps]
            
            # Daily and weekly patterns
            traffic_pattern = (
                100 + 50 * np.sin(np.array(hours) * 2 * np.pi / 24) +  # Daily cycle
                20 * np.sin(np.array(days) * 2 * np.pi / 7)            # Weekly cycle
            )
            
            # Add noise and ensure positive
            traffic = np.maximum(1, traffic_pattern + np.random.normal(0, 10, size=len(timestamps)))
            
            # Generate some data
            for i, ts in enumerate(timestamps):
                # Determine how many requests in this minute
                num_requests = int(traffic[i] / 10)  # Scale down to reasonable number
                
                # Generate requests
                for j in range(num_requests):
                    # Random user type
                    user_type = "STD" if np.random.random() < 0.7 else "PRM"
                    
                    # Random endpoint
                    endpoint_type = np.random.choice(["light", "medium", "heavy"], p=[0.6, 0.3, 0.1])
                    endpoint = f"/api/resources/{endpoint_type}/item_{np.random.randint(1, 10)}"
                    
                    # Response time based on endpoint type
                    if endpoint_type == "light":
                        response_time = np.random.gamma(shape=2, scale=0.02)
                    elif endpoint_type == "medium":
                        response_time = np.random.gamma(shape=2, scale=0.05)
                    else:  # heavy
                        response_time = np.random.gamma(shape=2, scale=0.1)
                    
                    # Record request
                    user_id = f"user_{np.random.randint(1, 100)}"
                    
                    # Sometimes generate rate limited or error responses
                    rate_limited = np.random.random() < 0.02  # 2% rate limited
                    error = not rate_limited and np.random.random() < 0.01  # 1% errors (if not rate limited)
                    
                    status_code = 429 if rate_limited else (500 if error else 200)
                    
                    # Record request
                    collector.record_request(
                        user_id=user_id,
                        user_type=user_type,
                        endpoint=endpoint,
                        response_time=response_time,
                        status_code=status_code,
                        tokens_consumed=1 if not rate_limited else 0,
                        tokens_remaining=np.random.randint(0, 20),
                        error=error,
                        rate_limited=rate_limited
                    )
            
            # Make sure data is written
            collector._write_batch()
            logger.info(f"Generated {len(timestamps)} minutes of sample traffic data")
        
        # Train models
        logger.info("Training models...")
        training_result = await trainer.train_models(force_retrain=True)
        logger.info(f"Training result: {training_result}")
        
        # Evaluate models
        logger.info("Evaluating models...")
        evaluation_result = await trainer.evaluate_models()
        logger.info(f"Evaluation result: {evaluation_result}")
        
        # Predict rate limit parameters
        logger.info("Predicting rate limit parameters...")
        std_params = await trainer.predict_rate_limit_parameters("STD")
        prm_params = await trainer.predict_rate_limit_parameters("PRM")
        
        logger.info(f"STD parameters: {std_params}")
        logger.info(f"PRM parameters: {prm_params}")
        
        # Detect anomalies
        logger.info("Detecting anomalies...")
        anomaly_result = await trainer.detect_traffic_anomalies()
        logger.info(f"Anomaly detection result: {anomaly_result}")
        
        logger.info("Model trainer demo complete")
    
    # Run the async main function
    asyncio.run(main())
