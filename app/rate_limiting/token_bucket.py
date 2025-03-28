import time
from typing import Optional

class TokenBucket:
    """
    Token Bucket algorithm implementation for rate limiting.
    
    This implementation uses time-based token refilling.
    Each user gets their own token bucket with a capacity and refill rate 
    based on their user type and system conditions.
    """
    
    def __init__(
        self, 
        capacity: int, 
        refill_rate: int, 
        refill_duration: int = 60,
        initial_tokens: Optional[int] = None
    ):
        """
        Initialize a token bucket.
        
        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Number of tokens to add per refill_duration
            refill_duration: Duration in seconds for refill cycle
            initial_tokens: Initial number of tokens (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_duration = refill_duration
        self.tokens = capacity if initial_tokens is None else initial_tokens
        self.last_refill = time.time()
        
        # For statistics and monitoring
        self.total_tokens_consumed = 0
        self.total_requests_allowed = 0
        self.total_requests_limited = 0
        self.last_consumed_at = 0
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        # First refill the bucket based on elapsed time
        self._refill()
        
        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            self.total_tokens_consumed += tokens
            self.total_requests_allowed += 1
            self.last_consumed_at = time.time()
            return True
        
        # Not enough tokens
        self.total_requests_limited += 1
        return False
    
    def _refill(self) -> None:
        """
        Refill tokens based on time elapsed since last refill.
        This is called automatically when consume() is called.
        """
        now = time.time()
        time_passed = now - self.last_refill
        
        # Calculate how many tokens to add
        tokens_to_add = (time_passed / self.refill_duration) * self.refill_rate
        
        if tokens_to_add > 0:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def get_status(self) -> dict:
        """
        Get the current status of the token bucket.
        
        Returns:
            Dictionary with bucket stats
        """
        self._refill()  # Ensure tokens are up to date
        
        return {
            "capacity": self.capacity,
            "available_tokens": self.tokens,
            "refill_rate": self.refill_rate,
            "refill_duration": self.refill_duration,
            "tokens_per_second": self.refill_rate / self.refill_duration,
            "total_consumed": self.total_tokens_consumed,
            "requests_allowed": self.total_requests_allowed,
            "requests_limited": self.total_requests_limited,
            "utilization_percent": (self.capacity - self.tokens) / self.capacity * 100 if self.capacity > 0 else 0,
            "next_token_available_in": 0 if self.tokens > 0 else (
                (1 - (self.tokens % 1)) * self.refill_duration / self.refill_rate
            ),
        }
    
    def update_parameters(self, capacity: Optional[int] = None, refill_rate: Optional[int] = None) -> None:
        """
        Update the bucket parameters.
        
        This allows for dynamic adjustment of rate limits based on traffic patterns.
        
        Args:
            capacity: New maximum capacity (if None, keeps current value)
            refill_rate: New refill rate (if None, keeps current value)
        """
        # Refill before updating to ensure we have the latest token count
        self._refill()
        
        if capacity is not None:
            # If increasing capacity, also increase available tokens proportionally
            if capacity > self.capacity and self.capacity > 0:
                # Calculate the current fill percentage
                fill_percent = self.tokens / self.capacity
                
                # Set new capacity
                self.capacity = capacity
                
                # Adjust tokens to maintain the same fill percentage
                self.tokens = min(capacity, fill_percent * capacity)
            else:
                # If decreasing capacity, make sure tokens doesn't exceed new capacity
                self.capacity = capacity
                self.tokens = min(self.tokens, capacity)
        
        if refill_rate is not None:
            self.refill_rate = refill_rate
