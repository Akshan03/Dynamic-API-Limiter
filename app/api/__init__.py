from fastapi import APIRouter
from app.api.endpoints import router as api_router

# Main API router
router = APIRouter()

# Include the API endpoints router with a prefix
router.include_router(api_router, prefix="/api")

# Export the router for use in the main application
__all__ = ["router"]
