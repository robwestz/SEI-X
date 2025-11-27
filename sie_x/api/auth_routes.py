"""
Authentication routes for SIE-X.

Endpoints:
- POST /auth/register-tenant - Create new tenant with admin user
- POST /auth/register - Create new user (requires existing tenant)
- POST /auth/login - Authenticate and get JWT token
- POST /auth/logout - Blacklist JWT token
- GET /auth/me - Get current user info
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional

from sie_x.models.tenant import (
    CreateTenantRequest, CreateUserRequest, LoginRequest, LoginResponse,
    User, Tenant, TenantContext
)
from sie_x.services.auth_service import AuthService
from sie_x.services.user_service import UserService
from sie_x.api.decorators import get_current_tenant
from sie_x.cache.redis_cache import RedisCache
import os

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Initialize auth service (will be set by server on startup)
auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Dependency to get auth service."""
    if auth_service is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication service not initialized"
        )
    return auth_service


@router.post("/register-tenant", response_model=dict)
async def register_tenant(
    request: CreateTenantRequest,
    service: UserService = Depends(lambda: UserService())
):
    """
    Create a new tenant with an admin user.

    This endpoint is public and used for initial tenant registration.
    After registration, the admin user can login and create additional users.

    Request Body:
    ```json
    {
        "name": "Acme Corporation",
        "slug": "acme-corp",
        "plan": "free",
        "admin_email": "admin@acme.com",
        "admin_first_name": "John",
        "admin_last_name": "Doe",
        "admin_password": "SecurePassword123!"
    }
    ```

    Returns:
    ```json
    {
        "tenant": {...},
        "admin_user": {...},
        "message": "Tenant created successfully. Please login with your credentials."
    }
    ```

    **Security Note:** In production, you might want to:
    - Require email verification
    - Add CAPTCHA to prevent abuse
    - Rate limit this endpoint
    - Require payment information for non-free plans
    """
    tenant, admin_user, _ = await service.create_tenant_with_admin(request)

    return {
        "tenant": {
            "id": tenant.id,
            "name": tenant.name,
            "slug": tenant.slug,
            "plan": tenant.plan.value
        },
        "admin_user": {
            "id": admin_user.id,
            "email": admin_user.email,
            "role": admin_user.role.value
        },
        "message": "Tenant created successfully. Please login with your credentials."
    }


@router.post("/register", response_model=User)
async def register_user(
    request: CreateUserRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: UserService = Depends(lambda: UserService())
):
    """
    Create a new user within the current tenant.

    This endpoint requires authentication (JWT token).
    Only tenant admins can create new users.

    Request Body:
    ```json
    {
        "email": "user@acme.com",
        "first_name": "Jane",
        "last_name": "Smith",
        "password": "SecurePassword123!",
        "role": "member"
    }
    ```

    Returns:
        User object (without password)
    """
    # Only admins can create users
    if not tenant.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only tenant admins can create users"
        )

    user = await service.create_user(tenant.id, request)
    return user


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate user and get JWT token.

    Request Body:
    ```json
    {
        "email": "admin@acme.com",
        "password": "SecurePassword123!",
        "tenant_slug": "acme-corp"
    }
    ```

    Returns:
    ```json
    {
        "access_token": "eyJhbG...",
        "token_type": "bearer",
        "expires_in": 3600,
        "user": {...},
        "tenant": {...}
    }
    ```

    Use the access_token in subsequent requests:
    ```
    Authorization: Bearer <access_token>
    ```
    """
    return await service.login(request)


@router.post("/logout")
async def logout(
    authorization: str = Header(...),
    service: AuthService = Depends(get_auth_service)
):
    """
    Logout user by blacklisting the JWT token.

    Headers:
        Authorization: Bearer <token>

    Returns:
    ```json
    {
        "message": "Logged out successfully"
    }
    ```

    After logout, the token will be blacklisted and cannot be used
    for authentication until it expires.
    """
    # Extract token from header
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]

    # Blacklist token
    await service.logout(token)

    return {"message": "Logged out successfully"}


@router.get("/me", response_model=dict)
async def get_current_user_info(
    tenant: TenantContext = Depends(get_current_tenant),
    service: UserService = Depends(lambda: UserService())
):
    """
    Get current authenticated user's information.

    Requires:
        Authorization: Bearer <token>

    Returns:
    ```json
    {
        "user": {
            "id": "...",
            "email": "user@acme.com",
            "first_name": "Jane",
            "last_name": "Smith",
            "role": "member"
        },
        "tenant": {
            "id": "...",
            "name": "Acme Corporation",
            "slug": "acme-corp",
            "plan": "professional",
            "status": "active"
        },
        "permissions": {
            "is_admin": false,
            "is_super_admin": false,
            "can_manage_users": false
        }
    }
    ```
    """
    # Get full user details
    user = await service.get_user(tenant.user_id, tenant.id)

    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role.value,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None
        },
        "tenant": {
            "id": tenant.id,
            "slug": tenant.slug,
            "plan": tenant.plan.value,
            "status": tenant.status.value
        },
        "permissions": {
            "is_admin": tenant.is_admin,
            "is_super_admin": tenant.is_super_admin,
            "can_manage_users": tenant.can_manage_users()
        }
    }
