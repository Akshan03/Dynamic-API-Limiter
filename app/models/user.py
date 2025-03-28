from pydantic import BaseModel
from typing import Dict, Optional, List
import uuid

# A simple in-memory database of users
# In a real application, this would be a database
USERS_DB = {}

class User(BaseModel):
    """User model representing API consumers"""
    id: str
    username: str
    type: str  # "STD" or "PRM"
    role: str = "user"  # "user" or "admin"
    email: Optional[str] = None
    created_at: float = 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "id": "user_123",
                "username": "johndoe",
                "type": "STD",
                "role": "user",
                "email": "john@example.com"
            }
        }

def get_user_by_id(user_id: str) -> Optional[Dict]:
    """
    Retrieve a user by ID
    
    In a real application, this would query a database
    """
    return USERS_DB.get(user_id)

def create_user(username: str, user_type: str = "STD", role: str = "user", email: Optional[str] = None) -> User:
    """
    Create a new user
    
    Args:
        username: Username
        user_type: User type (STD or PRM)
        role: User role (user or admin)
        email: Optional email address
    
    Returns:
        The created user
    """
    import time
    
    # Generate a unique ID
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    
    # Create user
    user = User(
        id=user_id,
        username=username,
        type=user_type,
        role=role,
        email=email,
        created_at=time.time()
    )
    
    # Save to "database"
    USERS_DB[user_id] = user.dict()
    
    return user

def create_test_users():
    """
    Create test users for demonstration
    
    Returns:
        Dictionary with user IDs mapped to user types
    """
    # Clear existing users to prevent duplicates
    USERS_DB.clear()
    
    users = {}
    
    # Create standard users with fixed IDs
    std_user_ids = ["std_user_1", "std_user_2", "std_user_3", "std_user_4", "std_user_5"]
    for i, user_id in enumerate(std_user_ids):
        import time
        
        # Create user
        user = User(
            id=user_id,
            username=f"std_user_{i}",
            type="STD",
            role="user",
            email=None,
            created_at=time.time()
        )
        
        # Save to "database"
        USERS_DB[user_id] = user.dict()
        users[user_id] = "STD"
    
    # Create premium users with fixed IDs
    prm_user_ids = ["prm_user_1", "prm_user_2", "prm_user_3"]
    for i, user_id in enumerate(prm_user_ids):
        import time
        
        # Create user
        user = User(
            id=user_id,
            username=f"prm_user_{i}",
            type="PRM",
            role="user",
            email=None,
            created_at=time.time()
        )
        
        # Save to "database"
        USERS_DB[user_id] = user.dict()
        users[user_id] = "PRM"
    
    # Create admin user with fixed ID
    admin_id = "admin_user"
    import time
    
    # Create admin user
    admin = User(
        id=admin_id,
        username="admin",
        type="PRM",
        role="admin",
        email="admin@example.com",
        created_at=time.time()
    )
    
    # Save to "database"
    USERS_DB[admin_id] = admin.dict()
    users[admin_id] = "PRM-ADMIN"
    
    print("USERS_DB contents:")
    for user_id, user_data in USERS_DB.items():
        print(f"  {user_id}: {user_data}")
    
    return users

# Initialize test users when module is imported
# In a real application, this would be replaced with database initialization
if not USERS_DB:
    create_test_users()
