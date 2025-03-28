from fastapi import APIRouter, Depends, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List, Any
import time
import random
from pydantic import BaseModel

# Import user authentication (simplified for demo)
from app.models.user import get_user_by_id

# Create router
router = APIRouter(tags=["api"])

# Models for request/response
class ResourceRequest(BaseModel):
    data: Dict[str, Any] = {}

class ResourceResponse(BaseModel):
    id: str
    type: str
    data: Dict[str, Any]
    processed_at: float

# Simulated database of resources
resources = {
    "light": {
        f"item_{i}": {"name": f"Light Resource {i}", "cost": "low"} 
        for i in range(1, 11)
    },
    "medium": {
        f"item_{i}": {"name": f"Medium Resource {i}", "cost": "medium"} 
        for i in range(1, 11)
    },
    "heavy": {
        f"item_{i}": {"name": f"Heavy Resource {i}", "cost": "high", "details": "Complex data"} 
        for i in range(1, 11)
    }
}

# Simple endpoint - 1 token
@router.get("/ping")
async def ping():
    """Simple health check that consumes 1 token"""
    return {"status": "ok", "timestamp": time.time()}

# Light resource endpoints - 1 token
@router.get("/resources/light", response_model=List[str])
async def get_light_resources():
    """List light resources - consumes 1 token"""
    return list(resources["light"].keys())

@router.get("/resources/light/{item_id}")
async def get_light_resource(item_id: str):
    """Get a specific light resource - consumes 1 token"""
    if item_id not in resources["light"]:
        raise HTTPException(status_code=404, detail="Resource not found")
    return {
        "id": item_id,
        "type": "light",
        "data": resources["light"][item_id],
        "processed_at": time.time()
    }

# Medium resource endpoints - 3 tokens
@router.get("/resources/medium", response_model=List[str])
async def get_medium_resources():
    """List medium resources - consumes 3 tokens"""
    # Simulate some processing time
    time.sleep(0.1)
    return list(resources["medium"].keys())

@router.get("/resources/medium/{item_id}")
async def get_medium_resource(item_id: str):
    """Get a specific medium resource - consumes 3 tokens"""
    # Simulate some processing time
    time.sleep(0.1)
    
    if item_id not in resources["medium"]:
        raise HTTPException(status_code=404, detail="Resource not found")
    return {
        "id": item_id,
        "type": "medium",
        "data": resources["medium"][item_id],
        "processed_at": time.time()
    }

# Heavy resource endpoints - 5 tokens
@router.get("/resources/heavy", response_model=List[str])
async def get_heavy_resources():
    """List heavy resources - consumes 5 tokens"""
    # Simulate more processing time
    time.sleep(0.2)
    return list(resources["heavy"].keys())

@router.get("/resources/heavy/{item_id}")
async def get_heavy_resource(item_id: str):
    """Get a specific heavy resource - consumes 5 tokens"""
    # Simulate more processing time
    time.sleep(0.2)
    
    if item_id not in resources["heavy"]:
        raise HTTPException(status_code=404, detail="Resource not found")
    return {
        "id": item_id,
        "type": "heavy",
        "data": resources["heavy"][item_id],
        "processed_at": time.time()
    }

# Create resource endpoint - varies by resource type
@router.post("/resources/{resource_type}", response_model=ResourceResponse)
async def create_resource(
    resource_type: str,
    resource_data: ResourceRequest,
):
    """Create a new resource - tokens vary by type (light:2, medium:5, heavy:10)"""
    if resource_type not in ["light", "medium", "heavy"]:
        raise HTTPException(status_code=400, detail="Invalid resource type")
    
    # Simulate processing time based on resource type
    if resource_type == "light":
        time.sleep(0.1)
    elif resource_type == "medium":
        time.sleep(0.3)
    else:  # heavy
        time.sleep(0.5)
    
    # Create new resource
    new_id = f"item_{len(resources[resource_type]) + 1}"
    resources[resource_type][new_id] = resource_data.data
    
    return ResourceResponse(
        id=new_id,
        type=resource_type,
        data=resource_data.data,
        processed_at=time.time()
    )

# Endpoint to simulate a burst of traffic (for testing rate limiting)
@router.get("/burst/{count}")
async def burst_requests(count: int = 10):
    """
    Simulate a burst of requests (for testing rate limiting)
    Each request in the burst consumes 1 token
    """
    if count > 100:
        count = 100  # Limit for safety
        
    results = []
    for i in range(count):
        results.append({
            "request_id": i,
            "processed": True,
            "timestamp": time.time()
        })
    
    return {"burst_size": count, "results": results}

# Endpoint to check rate limit status (doesn't consume tokens)
@router.get("/rate-limit-status")
async def rate_limit_status(request: Request):
    """
    Get current rate limit status - doesn't consume tokens
    Used for demonstration purposes
    """
    # This information is added by the middleware
    return {
        "has_limit_info": hasattr(request.state, "rate_limit_info"),
        "limit_info": getattr(request.state, "rate_limit_info", None),
        "current_time": time.time()
    }

# Admin endpoint to get comprehensive stats
@router.get("/admin/stats")
async def admin_stats(request: Request, x_user_id: str = Header(...)):
    """Admin endpoint to get system stats"""
    # Check if user is admin (simplified)
    user = get_user_by_id(x_user_id)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get stats from the app state (would be implemented in a real app)
    return {
        "total_requests": random.randint(1000, 10000),
        "rate_limited_requests": random.randint(10, 1000),
        "user_counts": {
            "STD": random.randint(50, 500),
            "PRM": random.randint(10, 100)
        },
        "current_time": time.time()
    }
