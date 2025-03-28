from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import asyncio
from typing import Dict, Optional, Tuple

from app.models.user import get_user_by_id, USERS_DB
from app.rate_limiting.token_bucket import TokenBucket
from app.rate_limiting.dynamic_adjuster import get_current_parameters

# In-memory storage for token buckets (in production, use Redis or similar)
user_buckets: Dict[str, TokenBucket] = {}

# Resource costs in tokens
ENDPOINT_COSTS = {
    # Default cost is 1 token
    "default": 1,
    
    # GET requests
    "GET:/api/resources/medium": 3,
    "GET:/api/resources/medium/{item_id}": 3,
    "GET:/api/resources/heavy": 5,
    "GET:/api/resources/heavy/{item_id}": 5,
    
    # POST requests
    "POST:/api/resources/light": 2,
    "POST:/api/resources/medium": 5,
    "POST:/api/resources/heavy": 10,
    
    # Burst test
    "GET:/api/burst/{count}": 1,
    
    # No cost (for monitoring)
    "GET:/api/rate-limit-status": 0,
}

# Endpoints excluded from rate limiting
EXCLUDED_ENDPOINTS = [
    "/docs",
    "/redoc",
    "/openapi.json",
]

class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app,
        default_std_capacity: int = 20,
        default_std_refill_rate: int = 5,
        default_prm_capacity: int = 60,
        default_prm_refill_rate: int = 15,
        refill_duration: int = 60  # seconds
    ):
        super().__init__(app)
        self.default_std_capacity = default_std_capacity
        self.default_std_refill_rate = default_std_refill_rate
        self.default_prm_capacity = default_prm_capacity
        self.default_prm_refill_rate = default_prm_refill_rate
        self.refill_duration = refill_duration
    
    async def dispatch(self, request: Request, call_next):
        # Print USERS_DB contents
        print("\nUSERS_DB contents in middleware:")
        for user_id, user_data in USERS_DB.items():
            print(f"  {user_id}: {user_data}")
        
        # Skip rate limiting for excluded endpoints
        if any(request.url.path.startswith(path) for path in EXCLUDED_ENDPOINTS):
            return await call_next(request)

        # Extract user information from headers
        user_id = request.headers.get("X-User-ID")
        print(f"Request user_id from header: '{user_id}'")
        
        if not user_id:
            return JSONResponse(
                status_code=401,
                content={"detail": "X-User-ID header is required"}
            )
        
        # Get user information - would typically come from auth service or database
        user = get_user_by_id(user_id)
        print(f"Result of get_user_by_id('{user_id}'): {user}")
        
        if not user:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid user ID"}
            )
        
        user_type = user.get("type", "STD")  # Default to STD if not specified
        
        # Get or create token bucket for this user
        bucket = await self.get_token_bucket(user_id, user_type)
        
        # Determine how many tokens this endpoint requires
        tokens_required = self.get_endpoint_cost(request)
        
        # Skip token consumption for zero-cost endpoints
        if tokens_required == 0:
            response = await call_next(request)
            # Add rate limit info to response headers
            self.add_ratelimit_headers(response, bucket)
            return response
        
        # Check if user has enough tokens
        if bucket.consume(tokens_required):
            # User has enough tokens, proceed with the request
            start_time = time.time()
            
            # Store rate limit info in request state for potential use by endpoints
            request.state.rate_limit_info = {
                "user_id": user_id,
                "user_type": user_type,
                "tokens_remaining": bucket.tokens,
                "tokens_capacity": bucket.capacity
            }
            
            response = await call_next(request)
            
            # Log the request (in a real application, send to the ML data collector)
            request_duration = time.time() - start_time
            self.log_request(user_id, user_type, request, tokens_required, request_duration)
            
            # Add rate limit info to response headers
            self.add_ratelimit_headers(response, bucket)
            
            return response
        else:
            # User has exceeded their rate limit
            reset_time = bucket.last_refill + bucket.refill_duration
            retry_after = max(1, int(reset_time - time.time()))
            
            # Log the rate limit event
            self.log_rate_limit_event(user_id, user_type, request, tokens_required)
            
            return JSONResponse(
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(bucket.capacity),
                    "X-RateLimit-Remaining": str(bucket.tokens),
                    "X-RateLimit-Reset": str(int(reset_time)),
                    "Retry-After": str(retry_after)
                },
                content={
                    "detail": "Rate limit exceeded",
                    "tokens_required": tokens_required,
                    "tokens_available": bucket.tokens,
                    "retry_after_seconds": retry_after
                }
            )
    
    async def get_token_bucket(self, user_id: str, user_type: str) -> TokenBucket:
        """Get or create a token bucket for the specified user"""
        # Check if bucket already exists
        if user_id in user_buckets:
            # Consider updating the bucket parameters if they've changed
            bucket = user_buckets[user_id]
            
            # Optionally update bucket parameters if they should change 
            # (e.g., user type changed or dynamic parameters changed)
            # await self.maybe_update_bucket_parameters(bucket, user_type)
            
            return bucket
        
        # Get parameters based on user type and current system state
        parameters = await self.get_bucket_parameters(user_type)
        
        # Create new bucket
        bucket = TokenBucket(
            capacity=parameters["capacity"],
            refill_rate=parameters["refill_rate"],
            refill_duration=self.refill_duration
        )
        
        # Store for future use
        user_buckets[user_id] = bucket
        return bucket
    
    async def get_bucket_parameters(self, user_type: str) -> Dict[str, int]:
        """Get the appropriate bucket parameters based on user type and system state"""
        # In a production system, this would query the ML model for dynamic parameters
        # For simplicity, we use static defaults with a call to the dynamic adjuster
        try:
            # Try to get dynamically adjusted parameters
            dynamic_params = await get_current_parameters(user_type)
            
            # Use the dynamic parameters if available
            if dynamic_params:
                return dynamic_params
            
        except Exception as e:
            # Log the error but continue with default parameters
            print(f"Error getting dynamic parameters: {e}")
        
        # Fall back to default parameters based on user type
        if user_type == "PRM":
            return {
                "capacity": self.default_prm_capacity,
                "refill_rate": self.default_prm_refill_rate
            }
        else:  # STD or any other type
            return {
                "capacity": self.default_std_capacity,
                "refill_rate": self.default_std_refill_rate
            }
    
    def get_endpoint_cost(self, request: Request) -> int:
        """Determine how many tokens the current endpoint requires"""
        # Create a key for looking up the cost
        method = request.method
        path = request.url.path
        
        # Replace path parameters with placeholders
        # This is a simplified version - in production, use router matching
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part and part[0].isdigit():
                parts[i] = "{item_id}"  # Simple placeholder replacement
        
        path_template = "/".join(parts)
        
        # Generate possible keys to check
        keys_to_check = [
            f"{method}:{path_template}",
            f"{method}:{path}",  # Exact match
        ]
        
        # Check if endpoint has a specific cost
        for key in keys_to_check:
            if key in ENDPOINT_COSTS:
                return ENDPOINT_COSTS[key]
        
        # Return default cost
        return ENDPOINT_COSTS["default"]
    
    def add_ratelimit_headers(self, response, bucket: TokenBucket):
        """Add rate limit information to response headers"""
        response.headers["X-RateLimit-Limit"] = str(bucket.capacity)
        response.headers["X-RateLimit-Remaining"] = str(bucket.tokens)
        response.headers["X-RateLimit-Reset"] = str(int(bucket.last_refill + bucket.refill_duration))
    
    def log_request(self, user_id, user_type, request, tokens_required, duration):
        """Log request information for ML training"""
        # In a production system, this would send data to the ML data collector
        # For simplicity, we just print the information
        print(f"REQUEST: User: {user_id} ({user_type}), Endpoint: {request.url.path}, " 
              f"Tokens: {tokens_required}, Duration: {duration:.4f}s")
    
    def log_rate_limit_event(self, user_id, user_type, request, tokens_required):
        """Log when a request is rate limited"""
        # In a production system, this would send data to the ML data collector
        # For simplicity, we just print the information
        print(f"RATE LIMITED: User: {user_id} ({user_type}), Endpoint: {request.url.path}, " 
              f"Tokens Required: {tokens_required}")
