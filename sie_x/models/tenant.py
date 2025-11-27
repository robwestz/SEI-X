"""
Tenant and user models for multi-tenant architecture.

This module defines the core models for tenant isolation, RBAC, and user management.
"""

from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional
from datetime import datetime
from enum import Enum


class Role(str, Enum):
    """User roles for Role-Based Access Control (RBAC)."""
    SUPER_ADMIN = "super_admin"      # Platform administrator (all tenants)
    TENANT_ADMIN = "tenant_admin"    # Tenant administrator (all tenant resources)
    MANAGER = "manager"              # Can create/edit/delete resources
    MEMBER = "member"                # Can create/edit own resources
    VIEWER = "viewer"                # Read-only access


class TenantPlan(str, Enum):
    """Subscription plans with different quotas and features."""
    FREE = "free"                    # 100 requests/hour
    STARTER = "starter"              # 1,000 requests/hour
    PROFESSIONAL = "professional"    # 10,000 requests/hour
    ENTERPRISE = "enterprise"        # Unlimited + SLA


class TenantStatus(str, Enum):
    """Tenant account status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CANCELLED = "cancelled"


class Tenant(BaseModel):
    """
    Represents a tenant (organization/customer).

    Each tenant has isolated data and its own set of users.
    """

    id: str = Field(..., description="Unique tenant ID (UUID)")
    name: str = Field(..., min_length=1, max_length=255, description="Organization name")
    slug: str = Field(
        ...,
        min_length=3,
        max_length=100,
        pattern="^[a-z0-9-]+$",
        description="URL-friendly identifier"
    )
    status: TenantStatus = Field(default=TenantStatus.ACTIVE, description="Account status")
    plan: TenantPlan = Field(default=TenantPlan.FREE, description="Subscription plan")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Quotas
    monthly_quota: int = Field(default=10000, description="Monthly extraction quota")
    monthly_usage: int = Field(default=0, description="Current month usage")

    # Metadata
    metadata: dict = Field(default_factory=dict, description="Additional tenant metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Acme Corporation",
                "slug": "acme-corp",
                "status": "active",
                "plan": "professional",
                "monthly_quota": 100000,
                "monthly_usage": 15230
            }
        }


class User(BaseModel):
    """
    Represents a user within a tenant.

    Users belong to exactly one tenant and have a role within that tenant.
    """

    id: str = Field(..., description="Unique user ID (UUID)")
    tenant_id: str = Field(..., description="Tenant this user belongs to")
    email: EmailStr = Field(..., description="User email (unique per tenant)")
    role: Role = Field(default=Role.MEMBER, description="User role")
    is_active: bool = Field(default=True, description="Whether user can authenticate")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = Field(default=None)

    # Profile
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)

    # Metadata
    metadata: dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "650e8400-e29b-41d4-a716-446655440001",
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "john.doe@acme.com",
                "role": "manager",
                "is_active": True,
                "first_name": "John",
                "last_name": "Doe"
            }
        }


class TenantContext(BaseModel):
    """
    Tenant context extracted from JWT and passed to endpoints.

    This is the security context for a request, containing both tenant
    and user information extracted from the authenticated JWT token.

    Usage:
        @app.get("/resources")
        async def get_resources(tenant: TenantContext = Depends(get_current_tenant)):
            # tenant.id is the tenant ID to use for filtering
            # tenant.user_role determines permissions
            return await db.query(Resource).filter_by(tenant_id=tenant.id).all()
    """

    # Tenant info
    id: str = Field(..., description="Tenant ID from JWT")
    slug: str = Field(..., description="Tenant slug")
    plan: TenantPlan = Field(..., description="Subscription plan")
    status: TenantStatus = Field(default=TenantStatus.ACTIVE)

    # User info
    user_id: str = Field(..., description="User ID from JWT")
    user_email: EmailStr = Field(..., description="User email from JWT")
    user_role: Role = Field(..., description="User role for RBAC")

    # Permissions (computed)
    is_admin: bool = Field(default=False, description="Whether user is tenant admin")
    is_super_admin: bool = Field(default=False, description="Whether user is platform admin")

    def __init__(self, **data):
        super().__init__(**data)
        # Compute permission flags
        self.is_admin = self.user_role in [Role.TENANT_ADMIN, Role.SUPER_ADMIN]
        self.is_super_admin = self.user_role == Role.SUPER_ADMIN

    def has_role(self, *roles: Role) -> bool:
        """Check if user has one of the specified roles."""
        return self.user_role in roles

    def can_manage_users(self) -> bool:
        """Check if user can manage other users."""
        return self.user_role in [Role.TENANT_ADMIN, Role.SUPER_ADMIN]

    def can_modify_resource(self, resource_owner_id: str) -> bool:
        """
        Check if user can modify a resource.

        Rules:
        - Admins can modify anything in their tenant
        - Managers can modify their own resources
        - Members can modify their own resources
        - Viewers cannot modify anything
        """
        if self.is_admin:
            return True
        if self.user_role == Role.VIEWER:
            return False
        return resource_owner_id == self.user_id

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "slug": "acme-corp",
                "plan": "professional",
                "status": "active",
                "user_id": "650e8400-e29b-41d4-a716-446655440001",
                "user_email": "john.doe@acme.com",
                "user_role": "manager",
                "is_admin": False,
                "is_super_admin": False
            }
        }


class CreateTenantRequest(BaseModel):
    """Request to create a new tenant."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=3, max_length=100, pattern="^[a-z0-9-]+$")
    plan: TenantPlan = Field(default=TenantPlan.FREE)

    # Admin user
    admin_email: EmailStr
    admin_first_name: str = Field(..., min_length=1, max_length=100)
    admin_last_name: str = Field(..., min_length=1, max_length=100)
    admin_password: str = Field(..., min_length=8, max_length=100)

    @field_validator('admin_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Ensure password meets security requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class CreateUserRequest(BaseModel):
    """Request to create a new user within a tenant."""

    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    role: Role = Field(default=Role.MEMBER)
    password: str = Field(..., min_length=8, max_length=100)

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Ensure password meets security requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    """Request to authenticate and get JWT token."""

    email: EmailStr
    password: str = Field(..., min_length=1)
    tenant_slug: Optional[str] = Field(
        None,
        description="Tenant slug (required if user belongs to multiple tenants)"
    )


class LoginResponse(BaseModel):
    """Response containing JWT token and user info."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(..., description="Token expiry in seconds")

    # User info
    user: User
    tenant: Tenant


class APIKey(BaseModel):
    """API key for programmatic access."""

    id: str
    tenant_id: str
    name: str = Field(..., min_length=1, max_length=255)
    key_prefix: str = Field(..., description="First 8 chars of key (for display)")
    key_hash: str = Field(..., description="SHA-256 hash of full key")

    created_by: str = Field(..., description="User ID who created this key")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None)
    last_used_at: Optional[datetime] = Field(None)

    is_active: bool = Field(default=True)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "750e8400-e29b-41d4-a716-446655440002",
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Production API Key",
                "key_prefix": "sk_live_",
                "created_by": "650e8400-e29b-41d4-a716-446655440001",
                "is_active": True
            }
        }
