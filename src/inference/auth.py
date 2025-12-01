"""
LedgerX - Authentication Module (Production Version with Secret Manager)
=========================================================================

Features:
- JWT token-based authentication
- Bcrypt password hashing
- Role-based access control (RBAC)
- Secret Manager integration for secure credential storage
- Environment variable fallback for local development
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from passlib.context import CryptContext

# Import Secret Manager
try:
    from ..utils.secret_manager import get_jwt_secret
    USE_SECRET_MANAGER = True
except ImportError:
    USE_SECRET_MANAGER = False
    print("[AUTH] ?? Secret Manager not available, using environment variables")

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# JWT secret from Secret Manager or environment variable
if USE_SECRET_MANAGER:
    try:
        SECRET_KEY = get_jwt_secret()
        print("[AUTH] ? JWT secret loaded from Secret Manager")
    except Exception as e:
        print(f"[AUTH] ?? Failed to load from Secret Manager: {e}")
        SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ledgerx-development-key-change-in-production")
        print("[AUTH] ?? Using JWT secret from environment variable")
else:
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ledgerx-development-key-change-in-production")
    print("[AUTH] ?? Using JWT secret from environment variable")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
    id: Optional[int] = None
    hashed_password: str

# -------------------------------------------------------------------
# PASSWORD UTILITIES
# -------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def validate_password(password: str):
    """
    Enforce strong password policy
    
    Requirements:
    - At least 12 characters
    - Contains uppercase letter
    - Contains lowercase letter
    - Contains number
    - Contains special character
    
    Raises:
        ValueError: If password doesn't meet requirements
    """
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters")
    if not any(c.isupper() for c in password):
        raise ValueError("Password must contain uppercase letter")
    if not any(c.islower() for c in password):
        raise ValueError("Password must contain lowercase letter")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must contain number")
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        raise ValueError("Password must contain special character")

# -------------------------------------------------------------------
# CLOUD SQL DATABASE INTEGRATION
# -------------------------------------------------------------------
try:
    from ..utils.database import get_user_by_username as db_get_user, verify_password as db_verify_password
    USE_CLOUD_SQL = True
    print("[AUTH] ? Cloud SQL enabled for user authentication")
except ImportError:
    USE_CLOUD_SQL = False
    print("[AUTH] ?? Cloud SQL not available, using fallback users")
    
    # Fallback fake users (if Cloud SQL unavailable)
    fake_users_db = {
        "admin": {
            "username": "admin",
            "full_name": "Admin User",
            "email": "admin@ledgerx.com",
            "hashed_password": hash_password("admin123"),
            "role": "admin",
            "disabled": False,
        },
        "john_doe": {
            "username": "john_doe",
            "full_name": "John Doe",
            "email": "john@example.com",
            "hashed_password": hash_password("password123"),
            "role": "user",
            "disabled": False,
        },
    }

print("[AUTH] ? Using bcrypt password hashing (production-ready)")
print("[AUTH] Test credentials: admin/admin123, john_doe/password123")

# -------------------------------------------------------------------
# USER UTILITIES
# -------------------------------------------------------------------
def get_user(username: str) -> Optional[UserInDB]:
    """Retrieve user from Cloud SQL or fallback"""
    if USE_CLOUD_SQL:
        user_dict = db_get_user(username)
        if user_dict:
            return UserInDB(**user_dict)
        return None
    else:
        # Fallback to fake users
        if username in fake_users_db:
            user_dict = fake_users_db[username]
            return UserInDB(**user_dict)
        return None

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user credentials"""
    user = get_user(username)
    if not user:
        return None
    
    # Verify password
    if USE_CLOUD_SQL:
        if not db_verify_password(password, user.hashed_password):
            return None
    else:
        if not verify_password(password, user.hashed_password):
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
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to get current active user"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# -------------------------------------------------------------------
# ROLE-BASED ACCESS CONTROL
# -------------------------------------------------------------------
def require_role(required_role: str):
    """Dependency factory for role-based access"""
    async def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {required_role}"
            )
        return current_user
    return role_checker

def require_any_role(*allowed_roles: str):
    """Dependency factory for multiple role check"""
    async def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
            )
        return current_user
    return role_checker

# Predefined dependencies
require_admin = require_role("admin")
require_user = require_any_role("user", "admin")
require_any_authenticated = get_current_active_user

