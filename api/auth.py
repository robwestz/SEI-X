"""
Authentication and Security Module for SIE-X API.

Handles:
- API Key validation
- JWT Token generation and verification
- User authentication dependencies
- Password hashing
"""

from typing import Optional, List
from datetime import datetime, timedelta
from enum import Enum
import os

from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from jose import JWTError, jwt
from passlib.context import CryptContext

from sie_x.config import get_config

# Configuration from unified config
_auth_cfg = get_config().auth
SECRET_KEY = _auth_cfg.secret_key.get_secret_value()
ALGORITHM = _auth_cfg.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = _auth_cfg.access_token_expire_minutes
API_KEY_NAME = "X-API-Key"

# Security Schemes
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Models ---

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    READ_ONLY = "read_only"

class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    disabled: bool = False
    role: UserRole = UserRole.USER

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    hashed_password: str

class User(UserBase):
    """Public User model"""
    pass

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# --- Mock Database (Replace with real DB in Phase 4) ---
# Format: username -> UserInDB
FAKE_USERS_DB = {
    "admin": UserInDB(
        username="admin",
        email="admin@example.com",
        disabled=False,
        role=UserRole.ADMIN,
        hashed_password=pwd_context.hash("admin")  # password: admin
    ),
    "user": UserInDB(
        username="user",
        email="user@example.com",
        disabled=False,
        role=UserRole.USER,
        hashed_password=pwd_context.hash("user")  # password: user
    )
}

# Mock API Keys (Replace with DB/Redis)
VALID_API_KEYS = {
    "siex-dev-key-123": {"user": "admin", "role": "admin"},
    "siex-client-key-456": {"user": "client_app", "role": "user"}
}

# --- Utilities ---

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Dependencies ---

async def get_api_key(
    api_key_header: str = Security(api_key_header),
) -> Optional[dict]:
    """
    Validate API Key from header.
    Returns user info dict if valid, else None.
    """
    if not api_key_header:
        return None

    if api_key_header in VALID_API_KEYS:
        return VALID_API_KEYS[api_key_header]
    
    # If key provided but invalid, raise 403
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate API Key"
    )

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    api_key_info: dict = Depends(get_api_key)
) -> User:
    """
    Get current user from either JWT token or API Key.
    API Key takes precedence if both provided (or simplified logic).
    """
    
    # 1. Try API Key first
    if api_key_info:
        # Create a transient user object from API key info
        return User(
            username=api_key_info["user"],
            role=UserRole(api_key_info["role"]),
            disabled=False
        )

    # 2. Try JWT
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        if username is None:
            raise JWTError()
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_in_db = FAKE_USERS_DB.get(token_data.username)
    if user_in_db is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
        
    return User(**user_in_db.model_dump())

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user
