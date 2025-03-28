import time
import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from app.config import settings
from app.api import router as api_router
from app.api.middleware import RateLimitingMiddleware
from app.models.user import create_test_users, USERS_DB

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    debug=settings.DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitingMiddleware,
    default_std_capacity=settings.RATE_LIMIT_STD_CAPACITY,
    default_std_refill_rate=settings.RATE_LIMIT_STD_REFILL_RATE,
    default_prm_capacity=settings.RATE_LIMIT_PRM_CAPACITY,
    default_prm_refill_rate=settings.RATE_LIMIT_PRM_REFILL_RATE,
    refill_duration=settings.RATE_LIMIT_REFILL_DURATION,
)

# Custom OpenAPI to include rate limiting info
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
        routes=app.routes,
    )
    
    # Add rate limiting information to OpenAPI schema
    openapi_schema["components"]["securitySchemes"] = {
        "RateLimit": {
            "type": "apiKey",
            "in": "header",
            "name": "X-User-ID",
            "description": "API key for rate limiting. Standard users have lower limits than Premium users."
        },
        "UserType": {
            "type": "apiKey",
            "in": "header",
            "name": "X-User-Type",
            "description": "User type (STD or PRM) determining rate limit thresholds."
        }
    }
    
    # Add security requirement to all routes
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"RateLimit": [], "UserType": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Include API routes
app.include_router(api_router, prefix=settings.API_PREFIX)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Root endpoint for API health check
@app.get("/")
async def root():
    """Health check endpoint"""
    # Get existing users instead of creating new ones
    
    # Return basic API information
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "rate_limiting": "enabled",
        "test_users": {
            "standard": [user_id for user_id, user in USERS_DB.items() if user["type"] == "STD"],
            "premium": [user_id for user_id, user in USERS_DB.items() if user["type"] == "PRM" and user["role"] != "admin"],
            "admin": [user_id for user_id, user in USERS_DB.items() if user["role"] == "admin"],
        },
        "documentation": "/docs",
    }


# Startup event to initialize the application
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    print(f"Rate limiting settings:")
    print(f"  Standard users: {settings.RATE_LIMIT_STD_CAPACITY} capacity, {settings.RATE_LIMIT_STD_REFILL_RATE} refill rate")
    print(f"  Premium users: {settings.RATE_LIMIT_PRM_CAPACITY} capacity, {settings.RATE_LIMIT_PRM_REFILL_RATE} refill rate")
    
    # Create test users on startup
    users = create_test_users()
    user_counts = {
        "STD": len([u for u, t in users.items() if t == "STD"]),
        "PRM": len([u for u, t in users.items() if t == "PRM"]),
        "ADMIN": len([u for u, t in users.items() if "ADMIN" in t]),
    }
    print(f"Created test users: {user_counts}")
    print("User IDs:")
    for user_id, user_type in users.items():
        print(f"  {user_type}: {user_id}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    print(f"Shutting down {settings.API_TITLE}")


# Run the application if executed directly
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=settings.DEBUG
    )
