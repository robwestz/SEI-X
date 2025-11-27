"""User management service."""

from typing import List
from fastapi import HTTPException

from sie_x.models.tenant import User, CreateUserRequest, Role, CreateTenantRequest, Tenant
from sie_x.db.repositories import UserRepository, TenantRepository
from sie_x.services.auth_service import AuthService


class UserService:
    """Service for user management operations."""

    @staticmethod
    async def create_user(
        tenant_id: str,
        request: CreateUserRequest
    ) -> User:
        """
        Create a new user within a tenant.

        Args:
            tenant_id: Tenant ID
            request: User creation request

        Returns:
            Created User

        Raises:
            HTTPException: If user creation fails
        """
        # Hash password
        password_hash = AuthService.hash_password(request.password)

        try:
            user = await UserRepository.create(
                tenant_id=tenant_id,
                email=request.email,
                password_hash=password_hash,
                first_name=request.first_name,
                last_name=request.last_name,
                role=request.role
            )
            return user

        except Exception as e:
            # Check for unique constraint violation (duplicate email)
            if "unique" in str(e).lower():
                raise HTTPException(
                    status_code=400,
                    detail="Email already exists in this tenant"
                )
            raise HTTPException(status_code=500, detail="Failed to create user")

    @staticmethod
    async def create_tenant_with_admin(request: CreateTenantRequest) -> tuple[Tenant, User, str]:
        """
        Create a new tenant with an admin user.

        Args:
            request: Tenant creation request

        Returns:
            Tuple of (Tenant, User, password) - password returned for first-time setup

        Raises:
            HTTPException: If creation fails
        """
        try:
            # Create tenant
            tenant = await TenantRepository.create(
                name=request.name,
                slug=request.slug,
                plan=request.plan
            )

            # Hash admin password
            password_hash = AuthService.hash_password(request.admin_password)

            # Create admin user
            admin_user = await UserRepository.create(
                tenant_id=tenant.id,
                email=request.admin_email,
                password_hash=password_hash,
                first_name=request.admin_first_name,
                last_name=request.admin_last_name,
                role=Role.TENANT_ADMIN
            )

            return (tenant, admin_user, request.admin_password)

        except Exception as e:
            if "unique" in str(e).lower():
                raise HTTPException(
                    status_code=400,
                    detail="Tenant slug or admin email already exists"
                )
            raise HTTPException(status_code=500, detail="Failed to create tenant")

    @staticmethod
    async def list_users(tenant_id: str) -> List[User]:
        """
        List all users in a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            List of users
        """
        return await UserRepository.list_by_tenant(tenant_id)

    @staticmethod
    async def get_user(user_id: str, tenant_id: str) -> User:
        """
        Get a user by ID.

        Args:
            user_id: User ID
            tenant_id: Tenant ID

        Returns:
            User

        Raises:
            HTTPException: If user not found
        """
        user = await UserRepository.get_by_id(user_id, tenant_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
