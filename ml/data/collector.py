import os
import time
import pandas as pd
import numpy as np
import asyncio
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("traffic_collector")

class TrafficCollector:
    """
    Collects and stores API traffic data for ML model training.
    Operates in real-time to gather traffic metrics from the API.
    """
    
    def __init__(
        self,
        data_dir: str = "./data/raw",
        collection_interval: int = 60,  # seconds
        batch_size: int = 100,  # Number of records before writing to disk
        retention_days: int = 30,  # How long to keep data
        enabled: bool = True
    ):
        """
        Initialize the traffic collector.
        
        Args:
            data_dir: Directory to store collected data
            collection_interval: How often to aggregate and save data (seconds)
            batch_size: Number of records to collect before writing to disk
            retention_days: How long to keep raw data (days)
            enabled: Whether data collection is enabled
        """
        self.data_dir = Path(data_dir)
        self.collection_interval = collection_interval
        self.batch_size = batch_size
        self.retention_days = retention_days
        self.enabled = enabled
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Queue for collecting data asynchronously
        self.queue = queue.Queue()
        
        # Current batch of records
        self.current_batch = []
        
        # Tracking variables
        self.last_write_time = time.time()
        self.running = False
        self.collection_thread = None
        
        # Initialize collector
        logger.info(f"Traffic collector initialized with data directory: {self.data_dir}")
        
        # Start collection thread if enabled
        if self.enabled:
            self.start()
    
    def start(self) -> None:
        """Start the data collection thread."""
        if self.running:
            logger.warning("Collection thread already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True  # Allow Python to exit even if thread is running
        )
        self.collection_thread.start()
        logger.info("Traffic collection thread started")
    
    def stop(self) -> None:
        """Stop the data collection thread."""
        if not self.running:
            logger.warning("Collection thread not running")
            return
        
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
            
        # Write any remaining records
        self._write_batch()
        logger.info("Traffic collection thread stopped")
    
    def record_request(
        self,
        user_id: str,
        user_type: str,
        endpoint: str,
        response_time: float,
        status_code: int,
        tokens_consumed: int = 1,
        tokens_remaining: Optional[int] = None,
        error: bool = False,
        rate_limited: bool = False,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an API request for data collection.
        
        Args:
            user_id: User ID
            user_type: User type (STD or PRM)
            endpoint: API endpoint
            response_time: Response time in seconds
            status_code: HTTP status code
            tokens_consumed: Number of tokens consumed
            tokens_remaining: Number of tokens remaining after request
            error: Whether request resulted in error
            rate_limited: Whether request was rate limited
            additional_data: Additional data to record
        """
        if not self.enabled:
            return
        
        # Create record
        record = {
            'timestamp': time.time(),
            'user_id': user_id,
            'user_type': user_type,
            'endpoint': endpoint,
            'response_time': response_time,
            'status_code': status_code,
            'tokens_consumed': tokens_consumed,
            'tokens_remaining': tokens_remaining,
            'error': error,
            'rate_limited': rate_limited
        }
        
        # Add additional data if provided
        if additional_data:
            record.update(additional_data)
        
        # Add to queue
        self.queue.put(record)
    
    def _collection_loop(self) -> None:
        """Main data collection loop that runs in a separate thread."""
        while self.running:
            try:
                # Try to get a record from the queue
                try:
                    record = self.queue.get(timeout=1.0)
                    self.current_batch.append(record)
                    self.queue.task_done()
                except queue.Empty:
                    # No records in queue, continue loop
                    pass
                
                # Check if it's time to write batch
                current_time = time.time()
                time_to_write = current_time - self.last_write_time >= self.collection_interval
                batch_full = len(self.current_batch) >= self.batch_size
                
                if (batch_full or time_to_write) and self.current_batch:
                    self._write_batch()
                    self.last_write_time = current_time
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
    
    def _write_batch(self) -> None:
        """Write the current batch of records to disk."""
        if not self.current_batch:
            return
        
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = self.data_dir / f"traffic_{timestamp}.json"
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(self.current_batch, f)
            
            logger.info(f"Wrote {len(self.current_batch)} records to {filename}")
            
            # Clear current batch
            self.current_batch = []
            
            # Clean up old files
            self._cleanup_old_files()
            
        except Exception as e:
            logger.error(f"Error writing batch: {e}")
    
    def _cleanup_old_files(self) -> None:
        """Remove files older than retention_days."""
        try:
            # Calculate cutoff time
            cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
            
            # Find and remove old files
            removed_count = 0
            for file_path in self.data_dir.glob("traffic_*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old data files")
                
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")
    
    def get_data_since(
        self,
        start_time: float,
        end_time: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get collected data for a time range.
        
        Args:
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time, default: now)
            
        Returns:
            DataFrame with collected data
        """
        end_time = end_time or time.time()
        
        # Find relevant files
        all_files = sorted(self.data_dir.glob("traffic_*.json"))
        
        # Read and filter data
        data_frames = []
        
        for file_path in all_files:
            # Check file modification time as a quick filter
            # If file was modified before start_time, skip it
            if file_path.stat().st_mtime < start_time:
                continue
            
            try:
                # Read file
                with open(file_path, 'r') as f:
                    records = json.load(f)
                
                # Filter records by timestamp
                filtered_records = [
                    r for r in records 
                    if r.get('timestamp', 0) >= start_time and r.get('timestamp', 0) <= end_time
                ]
                
                if filtered_records:
                    data_frames.append(pd.DataFrame(filtered_records))
                    
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        # Combine all dataframes
        if data_frames:
            result = pd.concat(data_frames, ignore_index=True)
            return result
        else:
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=[
                'timestamp', 'user_id', 'user_type', 'endpoint', 'response_time',
                'status_code', 'tokens_consumed', 'tokens_remaining', 'error', 'rate_limited'
            ])
    
    async def get_recent_metrics(
        self,
        minutes: int = 60,
        aggregate: bool = True,
        interval_seconds: int = 60
    ) -> pd.DataFrame:
        """
        Get recent traffic metrics, optionally aggregated by time interval.
        
        Args:
            minutes: Number of minutes of data to retrieve
            aggregate: Whether to aggregate by time interval
            interval_seconds: Time interval for aggregation (in seconds)
            
        Returns:
            DataFrame with traffic metrics
        """
        # Calculate start time
        start_time = time.time() - (minutes * 60)
        
        # Get raw data
        data = self.get_data_since(start_time)
        
        if data.empty:
            if aggregate:
                # Return empty aggregated dataframe
                return pd.DataFrame(columns=['timestamp', 'request_count', 'error_count', 
                                            'rate_limited_count', 'avg_response_time'])
            else:
                return data
        
        # Process data based on aggregation preference
        if aggregate:
            # Create time buckets
            data['time_bucket'] = (data['timestamp'] // interval_seconds) * interval_seconds
            
            # Aggregate by time bucket
            aggregated = data.groupby('time_bucket').agg({
                'timestamp': 'count',  # Count of requests
                'error': 'sum',        # Count of errors
                'rate_limited': 'sum', # Count of rate limited requests
                'response_time': 'mean' # Average response time
            }).reset_index()
            
            # Rename columns
            aggregated.columns = ['timestamp', 'request_count', 'error_count', 
                                 'rate_limited_count', 'avg_response_time']
            
            return aggregated
            
        return data


# Helper functions for more convenient usage

async def collect_traffic_data(
    collector: Optional[TrafficCollector] = None,
    **kwargs
) -> None:
    """
    Initialize and start traffic data collection.
    
    Args:
        collector: Optional existing collector instance
        **kwargs: Arguments to pass to TrafficCollector constructor
    """
    # Use existing collector or create a new one
    if collector is None:
        collector = TrafficCollector(**kwargs)
    
    # Ensure collector is running
    if not collector.running:
        collector.start()
    
    return collector


async def get_recent_traffic_data(
    minutes: int = 60,
    aggregate: bool = True,
    interval_seconds: int = 60,
    collector: Optional[TrafficCollector] = None,
    data_dir: str = "./data/raw"
) -> pd.DataFrame:
    """
    Get recent traffic data for ML model input.
    
    Args:
        minutes: Number of minutes of data to retrieve
        aggregate: Whether to aggregate by time interval
        interval_seconds: Time interval for aggregation (in seconds)
        collector: Optional existing collector instance
        data_dir: Directory where traffic data is stored
        
    Returns:
        DataFrame with traffic data
    """
    # Use existing collector or create a new one
    if collector is None:
        collector = TrafficCollector(data_dir=data_dir, enabled=False)
    
    # Get data
    return await collector.get_recent_metrics(
        minutes=minutes,
        aggregate=aggregate,
        interval_seconds=interval_seconds
    )


# Example usage if run directly
if __name__ == "__main__":
    # Create collector
    collector = TrafficCollector()
    
    # Generate some sample data
    for i in range(100):
        user_type = "STD" if i % 3 != 0 else "PRM"
        endpoint = f"/api/resource/{i % 5}"
        
        # Simulate normal request
        collector.record_request(
            user_id=f"user_{i % 10}",
            user_type=user_type,
            endpoint=endpoint,
            response_time=0.1 + (i % 5) * 0.05,
            status_code=200,
            tokens_consumed=1,
            tokens_remaining=10 - (i % 10),
            error=False,
            rate_limited=False
        )
        
        # Add some rate limited requests
        if i % 15 == 0:
            collector.record_request(
                user_id=f"user_{i % 5}",
                user_type="STD",
                endpoint=endpoint,
                response_time=0.02,
                status_code=429,
                tokens_consumed=0,
                tokens_remaining=0,
                error=False,
                rate_limited=True
            )
        
        # Add some errors
        if i % 20 == 0:
            collector.record_request(
                user_id=f"user_{i % 7}",
                user_type=user_type,
                endpoint=endpoint,
                response_time=0.5,
                status_code=500,
                tokens_consumed=1,
                tokens_remaining=5,
                error=True,
                rate_limited=False
            )
        
        # Sleep a bit to simulate time passing
        time.sleep(0.01)
    
    # Wait for collector to process queue
    time.sleep(2)
    
    # Stop collector
    collector.stop()
    
    # Get recent data
    async def test_get_data():
        data = await collector.get_recent_metrics(minutes=60, aggregate=True)
        print(f"Retrieved {len(data)} aggregated data points")
        print(data.head())
        
        # Get raw data
        raw_data = collector.get_data_since(time.time() - 3600)
        print(f"Retrieved {len(raw_data)} raw data points")
        print(raw_data.head())
    
    # Run async function
    asyncio.run(test_get_data())
    
    print("Traffic collector test complete")
