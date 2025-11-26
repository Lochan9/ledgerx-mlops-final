"""
LedgerX - Authentication Module (SIMPLE VERSION - NO BCRYPT)
=============================================================
For production, use proper password hashing!
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ledgerx-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -------------------------------------------------------------------
# DATA MODELS
# -------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str = "user"
    disabled: bool = False

class UserInDB(User):
    password: str  # Plain text for development only!

# -------------------------------------------------------------------
# USER DATABASE (PLAIN PASSWORDS - DEVELOPMENT ONLY!)
# -------------------------------------------------------------------
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@ledgerx.com",
        "password": "admin123",
        "role": "admin",
        "disabled": False,
    },
    "john_doe": {
        "username": "john_doe",
        "full_name": "John Doe",
        "email": "john@example.com",
        "password": "password123",
        "role": "user",
        "disabled": False,
    },
    "jane_viewer": {
        "username": "jane_viewer",
        "full_name": "Jane Viewer",
        "email": "jane@example.com",
        "password": "viewer123",
        "role": "readonly",
        "disabled": False,
    },
}

print("[AUTH] ⚠️  WARNING: Using plain text passwords (development only)")
print("[AUTH] Test credentials loaded:")
print("[AUTH]   - admin / admin123 (Admin)")
print("[AUTH]   - john_doe / password123 (User)")
print("[AUTH]   - jane_viewer / viewer123 (Readonly)")

# -------------------------------------------------------------------
# USER UTILITIES
# -------------------------------------------------------------------
def get_user(username: str) -> Optional[UserInDB]:
    """Retrieve user from database"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user credentials (simple comparison)"""
    user = get_user(username)
    if not user:
        return None
    if password != user.password:
        return None
    return user

# -------------------------------------------------------------------
# JWT TOKEN UTILITIES
# -------------------------------------------------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None:
            return None
            
        return TokenData(username=username, role=role)
    except JWTError:
        return None

# -------------------------------------------------------------------
# DEPENDENCY FUNCTIONS
# -------------------------------------------------------------------
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Dependency to get current authenticated user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = decode_access_token(token)
    if token_data is None or token_data.username is None:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency to ensure user is active"""
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

# -------------------------------------------------------------------
# ROLE-BASED ACCESS CONTROL
# -------------------------------------------------------------------
class RoleChecker:
    """Dependency class to check user roles"""
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: User = Depends(get_current_active_user)) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(self.allowed_roles)}"
            )
        return user

# Convenience role checker instances
require_admin = RoleChecker(["admin"])
require_user = RoleChecker(["admin", "user"])
require_any_authenticated = RoleChecker(["admin", "user", "readonly"])