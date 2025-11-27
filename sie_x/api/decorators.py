"""
FastAPI dependencies and decorators for tenant isolation and RBAC.

This module provides reusable dependencies for securing endpoints with
tenant context and role-based access control.
"""

from fastapi import Depends, HTTPException, Request
from sie_x.models.tenant import TenantContext, Role
from typing import List, Optional


async def get_current_tenant(request: Request) -> TenantContext:
    """
    Dependency to get current tenant context from request.

    This should be used on ALL protected endpoints to ensure tenant isolation.

    Usage:
        @app.get("/resources")
        async def get_resources(tenant: TenantContext = Depends(get_current_tenant)):
            # Use tenant.id to filter queries
            return await db.query(Resource).filter_by(tenant_id=tenant.id).all()

    Args:
        request: FastAPI request object

    Returns:
        TenantContext with tenant and user information

    Raises:
        HTTPException: 401 if no tenant context (should never happen if middleware is configured)
    """
    if not hasattr(request.state, "tenant"):
        raise HTTPException(
            status_code=401,
            detail="No tenant context. This is a configuration error."
        )

    return request.state.tenant


def require_roles(*allowed_roles: Role):
    """
    Dependency factory to require specific roles.

    Usage:
        # Single role
        @app.delete("/admin/users/{user_id}")
        async def delete_user(
            user_id: str,
            tenant: TenantContext = Depends(require_roles(Role.TENANT_ADMIN))
        ):
            # Only tenant admins can access
            pass

        # Multiple roles
        @app.post("/projects")
        async def create_project(
            tenant: TenantContext = Depends(require_roles(Role.MANAGER, Role.TENANT_ADMIN))
        ):
            # Managers and admins can access
            pass

    Args:
        *allowed_roles: One or more Role enum values

    Returns:
        Dependency function that validates role and returns TenantContext

    Raises:
        HTTPException: 403 if user doesn't have required role
    """

    async def role_checker(tenant: TenantContext = Depends(get_current_tenant)) -> TenantContext:
        if tenant.user_role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required one of: {[r.value for r in allowed_roles]}"
            )
        return tenant

    return role_checker


def require_admin(tenant: TenantContext = Depends(get_current_tenant)) -> TenantContext:
    """
    Dependency to require tenant admin or super admin role.

    Shorthand for require_roles(Role.TENANT_ADMIN, Role.SUPER_ADMIN).

    Usage:
        @app.get("/admin/settings")
        async def get_settings(tenant: TenantContext = Depends(require_admin)):
            # Only admins can access
            pass

    Args:
        tenant: Tenant context (injected by dependency)

    Returns:
        TenantContext if user is admin

    Raises:
        HTTPException: 403 if user is not admin
    """
    if not tenant.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Administrator permissions required"
        )
    return tenant


def require_super_admin(tenant: TenantContext = Depends(get_current_tenant)) -> TenantContext:
    """
    Dependency to require super admin role (platform administrator).

    Use this for platform-wide operations that affect all tenants.

    Usage:
        @app.get("/platform/tenants")
        async def list_all_tenants(tenant: TenantContext = Depends(require_super_admin)):
            # Only platform admins can access
            pass

    Args:
        tenant: Tenant context (injected by dependency)

    Returns:
        TenantContext if user is super admin

    Raises:
        HTTPException: 403 if user is not super admin
    """
    if not tenant.is_super_admin:
        raise HTTPException(
            status_code=403,
            detail="Platform administrator permissions required"
        )
    return tenant


def require_active_plan(*required_plans: str):
    """
    Dependency factory to require specific subscription plans.

    Usage:
        @app.post("/advanced/analysis")
        async def advanced_analysis(
            tenant: TenantContext = Depends(require_active_plan("professional", "enterprise"))
        ):
            # Only pro and enterprise tenants can access
            pass

    Args:
        *required_plans: One or more plan names (free, starter, professional, enterprise)

    Returns:
        Dependency function that validates plan and returns TenantContext

    Raises:
        HTTPException: 402 Payment Required if tenant doesn't have required plan
    """

    async def plan_checker(tenant: TenantContext = Depends(get_current_tenant)) -> TenantContext:
        if tenant.plan.value not in required_plans:
            raise HTTPException(
                status_code=402,
                detail=f"This feature requires one of these plans: {required_plans}. "
                       f"Your current plan: {tenant.plan.value}"
            )
        return tenant

    return plan_checker


class ResourceOwnershipValidator:
    """
    Validator to check if user can access/modify a resource.

    This handles the common pattern of checking if:
    - User is admin (can access any resource in tenant)
    - User is owner (can access own resource)
    - User is viewer (cannot modify)

    Usage:
        ownership_validator = ResourceOwnershipValidator()

        @app.put("/projects/{project_id}")
        async def update_project(
            project_id: str,
            data: UpdateProjectDto,
            tenant: TenantContext = Depends(get_current_tenant)
        ):
            # Get project from database
            project = await db.get_project(project_id, tenant.id)
            if not project:
                raise HTTPException(404, "Project not found")

            # Check if user can modify
            ownership_validator.can_modify(tenant, project.owner_id)

            # Update project
            await db.update_project(project_id, data)
    """

    def can_view(
        self,
        tenant: TenantContext,
        resource_owner_id: str,
        allow_viewer: bool = True
    ) -> bool:
        """
        Check if user can view a resource.

        Args:
            tenant: Tenant context
            resource_owner_id: User ID who owns the resource
            allow_viewer: Whether viewers can access (default: True)

        Returns:
            True if user can view, False otherwise
        """
        # Admins can view anything
        if tenant.is_admin:
            return True

        # Owner can view their own resources
        if resource_owner_id == tenant.user_id:
            return True

        # Viewers can view if allowed
        if allow_viewer and tenant.user_role == Role.VIEWER:
            return True

        return False

    def can_modify(self, tenant: TenantContext, resource_owner_id: str) -> bool:
        """
        Check if user can modify a resource (update/delete).

        Args:
            tenant: Tenant context
            resource_owner_id: User ID who owns the resource

        Returns:
            True if user can modify, False otherwise

        Raises:
            HTTPException: 403 if user cannot modify
        """
        if not tenant.can_modify_resource(resource_owner_id):
            if tenant.user_role == Role.VIEWER:
                raise HTTPException(
                    status_code=403,
                    detail="Viewers cannot modify resources"
                )
            else:
                raise HTTPException(
                    status_code=403,
                    detail="You can only modify your own resources"
                )
        return True

    def require_ownership(
        self,
        tenant: TenantContext,
        resource_owner_id: str
    ) -> bool:
        """
        Require user to be the owner (admins can bypass).

        Args:
            tenant: Tenant context
            resource_owner_id: User ID who owns the resource

        Returns:
            True if user is owner or admin

        Raises:
            HTTPException: 403 if user is not owner and not admin
        """
        if not tenant.is_admin and resource_owner_id != tenant.user_id:
            raise HTTPException(
                status_code=403,
                detail="You must be the resource owner to perform this action"
            )
        return True


# Global instance
resource_validator = ResourceOwnershipValidator()


# Convenience functions
def require_ownership(tenant: TenantContext, resource_owner_id: str):
    """
    Shorthand for resource_validator.require_ownership().

    Usage:
        require_ownership(tenant, project.owner_id)
    """
    return resource_validator.require_ownership(tenant, resource_owner_id)


def can_modify(tenant: TenantContext, resource_owner_id: str):
    """
    Shorthand for resource_validator.can_modify().

    Usage:
        can_modify(tenant, project.owner_id)
    """
    return resource_validator.can_modify(tenant, resource_owner_id)


# Public endpoint decorator
class Public:
    """
    Marker to indicate an endpoint is public (no authentication required).

    This is used in documentation and can be checked by middleware.

    Usage:
        @app.get("/health")
        @Public()
        async def health():
            return {"status": "healthy"}
    """

    def __call__(self, func):
        func.__public__ = True
        return func
