import time
import random
import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("traffic_simulator")

class UserBehaviorProfile:
    """
    Defines a user behavior profile for simulating API traffic.
    Different profiles can represent different types of users or anomalous behavior.
    """
    def __init__(
        self, 
        name: str,
        request_rate: float,  # Requests per second
        endpoint_weights: Dict[str, float],  # Probability distribution of endpoints
        burst_probability: float = 0.0,  # Probability of burst in each interval
        burst_size_range: Tuple[int, int] = (5, 15),  # Min/max size of bursts
        anomalous: bool = False  # Whether this profile represents anomalous behavior
    ):
        self.name = name
        self.request_rate = request_rate
        self.endpoint_weights = endpoint_weights
        self.burst_probability = burst_probability
        self.burst_size_range = burst_size_range
        self.anomalous = anomalous
        
    def __str__(self) -> str:
        return f"UserBehaviorProfile({self.name}, rate={self.request_rate}, anomalous={self.anomalous})"


class TrafficSimulator:
    """
    Simulates API traffic based on user behavior profiles.
    Can generate normal traffic patterns and anomalies.
    """
    
    def __init__(
        self, 
        api_base_url: str = "http://localhost:8000/api",
        data_collection_enabled: bool = True,
        data_storage_path: str = "./data/simulator"
    ):
        """
        Initialize the traffic simulator.
        
        Args:
            api_base_url: Base URL for the API
            data_collection_enabled: Whether to collect simulation data
            data_storage_path: Where to store simulation data
        """
        self.api_base_url = api_base_url
        self.data_collection_enabled = data_collection_enabled
        self.data_storage_path = Path(data_storage_path)
        
        # Ensure data directory exists
        if data_collection_enabled:
            self.data_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Available endpoints to simulate
        self.endpoints = {
            "ping": "/ping",
            "light_list": "/resources/light",
            "light_item": "/resources/light/item_1",
            "medium_list": "/resources/medium",
            "medium_item": "/resources/medium/item_1",
            "heavy_list": "/resources/heavy",
            "heavy_item": "/resources/heavy/item_1",
            "burst": "/burst/10",
            "status": "/rate-limit-status"
        }
        
        # Track simulation results
        self.results = {
            "requests_sent": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "requests_rate_limited": 0,
            "response_times": [],
            "start_time": None,
            "end_time": None,
            "users": {}
        }
    
    def create_default_profiles(self) -> Dict[str, UserBehaviorProfile]:
        """
        Create default user behavior profiles.
        
        Returns:
            Dictionary of profile name to profile
        """
        return {
            "standard_light": UserBehaviorProfile(
                name="standard_light",
                request_rate=0.2,  # 1 request per 5 seconds on average
                endpoint_weights={
                    "ping": 0.3,
                    "light_list": 0.3,
                    "light_item": 0.3,
                    "status": 0.1
                },
                burst_probability=0.05,
                burst_size_range=(3, 8)
            ),
            "standard_medium": UserBehaviorProfile(
                name="standard_medium",
                request_rate=0.5,  # 1 request per 2 seconds on average
                endpoint_weights={
                    "ping": 0.2,
                    "light_list": 0.2,
                    "light_item": 0.2,
                    "medium_list": 0.2,
                    "medium_item": 0.15,
                    "status": 0.05
                },
                burst_probability=0.1,
                burst_size_range=(5, 12)
            ),
            "standard_heavy": UserBehaviorProfile(
                name="standard_heavy",
                request_rate=1.0,  # 1 request per second on average
                endpoint_weights={
                    "ping": 0.1,
                    "light_list": 0.15,
                    "light_item": 0.15,
                    "medium_list": 0.2,
                    "medium_item": 0.2,
                    "heavy_list": 0.1,
                    "heavy_item": 0.05,
                    "status": 0.05
                },
                burst_probability=0.15,
                burst_size_range=(8, 15)
            ),
            "premium_light": UserBehaviorProfile(
                name="premium_light",
                request_rate=0.5,  # 1 request per 2 seconds on average
                endpoint_weights={
                    "ping": 0.2,
                    "light_list": 0.3,
                    "light_item": 0.3,
                    "medium_list": 0.1,
                    "medium_item": 0.05,
                    "status": 0.05
                },
                burst_probability=0.1,
                burst_size_range=(5, 10)
            ),
            "premium_heavy": UserBehaviorProfile(
                name="premium_heavy",
                request_rate=2.0,  # 2 requests per second on average
                endpoint_weights={
                    "ping": 0.1,
                    "light_list": 0.1,
                    "light_item": 0.1,
                    "medium_list": 0.2,
                    "medium_item": 0.2,
                    "heavy_list": 0.15,
                    "heavy_item": 0.1,
                    "status": 0.05
                },
                burst_probability=0.2,
                burst_size_range=(10, 20)
            ),
            "anomalous_scanner": UserBehaviorProfile(
                name="anomalous_scanner",
                request_rate=5.0,  # 5 requests per second on average
                endpoint_weights={
                    "ping": 0.05,
                    "light_list": 0.2,
                    "medium_list": 0.2,
                    "heavy_list": 0.2,
                    "light_item": 0.1,
                    "medium_item": 0.1,
                    "heavy_item": 0.1,
                    "status": 0.05
                },
                burst_probability=0.3,
                burst_size_range=(15, 30),
                anomalous=True
            ),
            "anomalous_burst": UserBehaviorProfile(
                name="anomalous_burst",
                request_rate=1.0,  # 1 request per second normally
                endpoint_weights={
                    "ping": 0.05,
                    "burst": 0.8,  # Mainly uses the burst endpoint
                    "status": 0.15
                },
                burst_probability=0.5,  # 50% chance of burst in each interval
                burst_size_range=(20, 50),  # Large bursts
                anomalous=True
            ),
            "anomalous_heavy": UserBehaviorProfile(
                name="anomalous_heavy",
                request_rate=3.0,  # 3 requests per second
                endpoint_weights={
                    "heavy_list": 0.4,
                    "heavy_item": 0.4,
                    "medium_list": 0.1,
                    "medium_item": 0.1
                },
                burst_probability=0.2,
                burst_size_range=(10, 20),
                anomalous=True
            )
        }
    
    async def simulate_user(
        self,
        user_id: str,
        user_type: str, 
        profile: UserBehaviorProfile, 
        duration: int,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """
        Simulate a single user's API activity.
        
        Args:
            user_id: User ID to use in requests
            user_type: User type (STD or PRM)
            profile: User behavior profile to simulate
            duration: Duration of simulation in seconds
            session: aiohttp client session for making requests
            
        Returns:
            Dictionary with simulation results for this user
        """
        logger.info(f"Starting simulation for user {user_id} ({user_type}) with profile {profile.name}")
        
        # Track user-specific results
        user_results = {
            "user_id": user_id,
            "user_type": user_type,
            "profile": profile.name,
            "requests_sent": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "requests_rate_limited": 0,
            "response_times": [],
            "endpoints": {},
            "anomalous": profile.anomalous
        }
        
        # Calculate expected number of requests based on rate and duration
        expected_requests = int(profile.request_rate * duration)
        
        # Add some randomness to the total
        actual_requests = max(1, int(random.normalvariate(expected_requests, expected_requests * 0.1)))
        
        # Spread requests over the duration
        start_time = time.time()
        end_time = start_time + duration
        
        # Calculate average delay between requests
        avg_delay = duration / actual_requests if actual_requests > 0 else 1.0
        
        current_time = start_time
        
        # Process requests
        while current_time < end_time:
            # Determine if this is a burst
            is_burst = random.random() < profile.burst_probability
            
            if is_burst:
                # Generate a burst of requests
                burst_size = random.randint(*profile.burst_size_range)
                logger.info(f"User {user_id} generating burst of {burst_size} requests")
                
                # Send burst requests concurrently
                burst_tasks = []
                for _ in range(burst_size):
                    endpoint_key = random.choices(
                        list(profile.endpoint_weights.keys()),
                        weights=list(profile.endpoint_weights.values())
                    )[0]
                    endpoint = self.endpoints[endpoint_key]
                    
                    task = asyncio.create_task(
                        self._send_request(
                            session=session,
                            user_id=user_id,
                            user_type=user_type,
                            endpoint=endpoint,
                            user_results=user_results
                        )
                    )
                    burst_tasks.append(task)
                
                # Wait for all burst requests to complete
                await asyncio.gather(*burst_tasks)
                
                # Update time - bursts take time
                current_time += burst_size * 0.1  # Assume 100ms per request in burst
            else:
                # Regular single request
                endpoint_key = random.choices(
                    list(profile.endpoint_weights.keys()),
                    weights=list(profile.endpoint_weights.values())
                )[0]
                endpoint = self.endpoints[endpoint_key]
                
                await self._send_request(
                    session=session,
                    user_id=user_id,
                    user_type=user_type,
                    endpoint=endpoint,
                    user_results=user_results
                )
            
            # Calculate next request time
            # Use exponential distribution for more realistic intervals
            next_delay = random.expovariate(1.0 / avg_delay)
            current_time += next_delay
            
            # Sleep until next request time if in the future
            sleep_time = current_time - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Update global results
        self.results["requests_sent"] += user_results["requests_sent"]
        self.results["requests_succeeded"] += user_results["requests_succeeded"]
        self.results["requests_failed"] += user_results["requests_failed"]
        self.results["requests_rate_limited"] += user_results["requests_rate_limited"]
        self.results["response_times"].extend(user_results["response_times"])
        self.results["users"][user_id] = user_results
        
        logger.info(f"Completed simulation for user {user_id}: "
                   f"sent {user_results['requests_sent']}, "
                   f"succeeded {user_results['requests_succeeded']}, "
                   f"rate limited {user_results['requests_rate_limited']}")
        
        return user_results
    
    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        user_id: str,
        user_type: str,
        endpoint: str,
        user_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a single API request.
        
        Args:
            session: aiohttp client session
            user_id: User ID to use in headers
            user_type: User type (STD or PRM)
            endpoint: API endpoint to call
            user_results: User results dictionary to update
            
        Returns:
            Dictionary with request results
        """
        url = f"{self.api_base_url}{endpoint}"
        headers = {
            "X-User-ID": user_id,
            "X-User-Type": user_type
        }
        
        result = {
            "url": url,
            "user_id": user_id,
            "user_type": user_type,
            "timestamp": time.time(),
            "success": False,
            "rate_limited": False,
            "status_code": None,
            "response_time": None,
            "error": None
        }
        
        # Track in user results
        user_results["requests_sent"] += 1
        endpoint_key = endpoint.split("/")[1] if len(endpoint.split("/")) > 1 else endpoint
        user_results["endpoints"][endpoint_key] = user_results["endpoints"].get(endpoint_key, 0) + 1
        
        start_time = time.time()
        try:
            async with session.get(url, headers=headers) as response:
                result["status_code"] = response.status
                result["response_time"] = time.time() - start_time
                user_results["response_times"].append(result["response_time"])
                
                # Check if rate limited
                if response.status == 429:
                    result["rate_limited"] = True
                    user_results["requests_rate_limited"] += 1
                    
                    # Extract rate limit headers if present
                    result["rate_limit_limit"] = response.headers.get("X-RateLimit-Limit")
                    result["rate_limit_remaining"] = response.headers.get("X-RateLimit-Remaining")
                    result["rate_limit_reset"] = response.headers.get("X-RateLimit-Reset")
                    
                    # Extract response data
                    try:
                        response_json = await response.json()
                        result["response_data"] = response_json
                    except:
                        pass
                
                # Request succeeded (even if rate limited)
                if 200 <= response.status < 500:
                    result["success"] = True
                    if response.status != 429:
                        user_results["requests_succeeded"] += 1
                else:
                    user_results["requests_failed"] += 1
                
                # Try to parse response data
                if response.status != 429:
                    try:
                        response_json = await response.json()
                        result["response_data"] = response_json
                    except:
                        pass
                
        except Exception as e:
            end_time = time.time()
            result["error"] = str(e)
            result["response_time"] = end_time - start_time
            user_results["requests_failed"] += 1
            logger.error(f"Error sending request to {url}: {e}")
        
        return result
    
    async def run_simulation(
        self,
        users_config: List[Dict[str, Any]],
        duration: int = 60,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run a full traffic simulation with multiple users.
        
        Args:
            users_config: List of user configurations
                Each configuration should have:
                - user_id: User ID
                - user_type: User type (STD or PRM)
                - profile_name: Name of behavior profile to use
            duration: Duration of simulation in seconds
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with simulation results
        """
        # Reset results
        self.results = {
            "requests_sent": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "requests_rate_limited": 0,
            "response_times": [],
            "start_time": time.time(),
            "end_time": None,
            "users": {},
            "duration": duration,
            "user_count": len(users_config)
        }
        
        # Get profiles
        profiles = self.create_default_profiles()
        
        # Create session for API requests
        async with aiohttp.ClientSession() as session:
            # Create tasks for each user
            tasks = []
            for user_config in users_config:
                user_id = user_config["user_id"]
                user_type = user_config["user_type"]
                profile_name = user_config["profile_name"]
                
                if profile_name not in profiles:
                    logger.warning(f"Profile {profile_name} not found, using standard_medium")
                    profile_name = "standard_medium"
                
                profile = profiles[profile_name]
                
                task = asyncio.create_task(
                    self.simulate_user(
                        user_id=user_id,
                        user_type=user_type,
                        profile=profile,
                        duration=duration,
                        session=session
                    )
                )
                tasks.append(task)
            
            # Run all user simulations concurrently
            await asyncio.gather(*tasks)
        
        # Record end time
        self.results["end_time"] = time.time()
        self.results["actual_duration"] = self.results["end_time"] - self.results["start_time"]
        
        # Calculate statistics
        if self.results["requests_sent"] > 0:
            self.results["success_rate"] = self.results["requests_succeeded"] / self.results["requests_sent"]
            self.results["failure_rate"] = self.results["requests_failed"] / self.results["requests_sent"]
            self.results["rate_limited_rate"] = self.results["requests_rate_limited"] / self.results["requests_sent"]
        else:
            self.results["success_rate"] = 0.0
            self.results["failure_rate"] = 0.0
            self.results["rate_limited_rate"] = 0.0
        
        # Calculate average response time
        if self.results["response_times"]:
            self.results["avg_response_time"] = sum(self.results["response_times"]) / len(self.results["response_times"])
            self.results["min_response_time"] = min(self.results["response_times"])
            self.results["max_response_time"] = max(self.results["response_times"])
        else:
            self.results["avg_response_time"] = 0.0
            self.results["min_response_time"] = 0.0
            self.results["max_response_time"] = 0.0
        
        # Calculate requests per second
        self.results["requests_per_second"] = (
            self.results["requests_sent"] / self.results["actual_duration"]
            if self.results["actual_duration"] > 0 else 0.0
        )
        
        # Save results if requested
        if save_results and self.data_collection_enabled:
            self._save_results()
        
        logger.info(f"Simulation completed: "
                   f"{self.results['requests_sent']} requests sent, "
                   f"{self.results['success_rate']:.2%} success rate, "
                   f"{self.results['rate_limited_rate']:.2%} rate limited")
        
        return self.results
    
    def _save_results(self) -> None:
        """Save simulation results to disk"""
        if not self.data_collection_enabled:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if it doesn't exist
        results_dir = self.data_storage_path / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save overall results
        overall_results = {k: v for k, v in self.results.items() if k != "users"}
        with open(results_dir / f"simulation_{timestamp}_overall.json", "w") as f:
            json.dump(overall_results, f, indent=2)
        
        # Save per-user results
        for user_id, user_results in self.results["users"].items():
            with open(results_dir / f"simulation_{timestamp}_user_{user_id}.json", "w") as f:
                json.dump(user_results, f, indent=2)
        
        # Create a summary CSV
        user_summary = []
        for user_id, user_results in self.results["users"].items():
            user_summary.append({
                "user_id": user_id,
                "user_type": user_results["user_type"],
                "profile": user_results["profile"],
                "requests_sent": user_results["requests_sent"],
                "requests_succeeded": user_results["requests_succeeded"],
                "requests_failed": user_results["requests_failed"],
                "requests_rate_limited": user_results["requests_rate_limited"],
                "success_rate": user_results["requests_succeeded"] / user_results["requests_sent"] 
                    if user_results["requests_sent"] > 0 else 0,
                "rate_limited_rate": user_results["requests_rate_limited"] / user_results["requests_sent"]
                    if user_results["requests_sent"] > 0 else 0,
                "avg_response_time": sum(user_results["response_times"]) / len(user_results["response_times"])
                    if user_results["response_times"] else 0,
                "anomalous": user_results["anomalous"]
            })
        
        # Save summary to CSV
        try:
            import pandas as pd
            df = pd.DataFrame(user_summary)
            df.to_csv(results_dir / f"simulation_{timestamp}_summary.csv", index=False)
        except ImportError:
            logger.warning("Pandas not available, skipping CSV summary creation")


async def simulate_traffic(
    api_base_url: str = "http://localhost:8000/api",
    duration: int = 60,
    std_users: int = 5,
    prm_users: int = 2,
    anomalous_users: int = 1,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Utility function to run a traffic simulation with a mix of users.
    
    Args:
        api_base_url: Base URL for the API
        duration: Duration of simulation in seconds
        std_users: Number of standard users to simulate
        prm_users: Number of premium users to simulate
        anomalous_users: Number of anomalous users to simulate
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary with simulation results
    """
    # Create simulator
    simulator = TrafficSimulator(api_base_url=api_base_url)
    
    # Get behavior profiles
    profiles = simulator.create_default_profiles()
    
    # Configure users
    users_config = []
    
    # Standard users
    std_profiles = ["standard_light", "standard_medium", "standard_heavy"]
    for i in range(std_users):
        profile_name = random.choice(std_profiles)
        users_config.append({
            "user_id": f"std_user_{i}",
            "user_type": "STD",
            "profile_name": profile_name
        })
    
    # Premium users
    prm_profiles = ["premium_light", "premium_heavy"]
    for i in range(prm_users):
        profile_name = random.choice(prm_profiles)
        users_config.append({
            "user_id": f"prm_user_{i}",
            "user_type": "PRM",
            "profile_name": profile_name
        })
    
    # Anomalous users
    if anomalous_users > 0:
        anom_profiles = ["anomalous_scanner", "anomalous_burst", "anomalous_heavy"]
        for i in range(anomalous_users):
            profile_name = random.choice(anom_profiles)
            # Anomalous users can be either STD or PRM
            user_type = random.choice(["STD", "PRM"])
            users_config.append({
                "user_id": f"anom_user_{i}",
                "user_type": user_type,
                "profile_name": profile_name
            })
    
    # Run simulation
    results = await simulator.run_simulation(
        users_config=users_config,
        duration=duration,
        save_results=save_results
    )
    
    return results


# Run the simulation if executed directly
if __name__ == "__main__":
    async def main():
        # Default simulation: 60 seconds with 5 STD users, 2 PRM users, and 1 anomalous user
        results = await simulate_traffic(duration=60)
        print(f"Simulation complete: {results['requests_sent']} requests sent")

    asyncio.run(main())
