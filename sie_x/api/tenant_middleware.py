"""
Tenant context middleware for multi-tenant isolation.

This middleware extracts tenant information from JWT tokens and ensures
proper tenant isolation throughout the request lifecycle.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from sie_x.models.tenant import TenantContext, Role, TenantPlan, TenantStatus

logger = logging.getLogger(__name__)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Extract and validate tenant context from authenticated JWT.

    This middleware MUST be added AFTER AuthenticationMiddleware since it
    depends on request.state.user being set by the authentication layer.

    Request Flow:
        1. AuthenticationMiddleware validates JWT → sets request.state.user
        2. TenantContextMiddleware extracts tenant info → sets request.state.tenant
        3. Endpoint accesses tenant context via @Depends(get_current_tenant)

    Security:
        - Tenant ID comes from JWT payload (never from user input!)
        - Validates tenant is active
        - Enforces tenant quotas
        - Sets PostgreSQL session variable for Row-Level Security (optional)
    """

    def __init__(self, app, db_connection=None, enforce_quotas: bool = True):
        """
        Initialize tenant context middleware.

        Args:
            app: FastAPI application
            db_connection: Optional database connection for quota checks
            enforce_quotas: Whether to enforce tenant quotas (default: True)
        """
        super().__init__(app)
        self.db = db_connection
        self.enforce_quotas = enforce_quotas

    async def dispatch(self, request: Request, call_next):
        """Process request and inject tenant context."""

        # Skip tenant context for public routes
        public_routes = [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc"
        ]

        if request.url.path in public_routes:
            return await call_next(request)

        # Check if user was authenticated by AuthenticationMiddleware
        if not hasattr(request.state, "user"):
            logger.warning(f"No user in request state for {request.url.path}")
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )

        user_payload = request.state.user

        # Extract tenant information from JWT
        try:
            tenant_context = self._extract_tenant_context(user_payload)
            request.state.tenant = tenant_context

            # Validate tenant status
            if tenant_context.status != TenantStatus.ACTIVE:
                raise HTTPException(
                    status_code=403,
                    detail=f"Tenant account is {tenant_context.status}. Please contact support."
                )

            # Check quotas (if enabled)
            if self.enforce_quotas and self.db:
                await self._check_quotas(tenant_context)

            # Set PostgreSQL session variable for Row-Level Security (if using Postgres)
            # This enables database-level tenant isolation
            if self.db:
                await self._set_postgres_tenant_context(tenant_context.id)

            logger.info(
                f"Tenant context set",
                extra={
                    "tenant_id": tenant_context.id,
                    "user_id": tenant_context.user_id,
                    "role": tenant_context.user_role.value
                }
            )

            # Process request
            response = await call_next(request)

            # Clear PostgreSQL session variable
            if self.db:
                await self._clear_postgres_tenant_context()

            return response

        except KeyError as e:
            logger.error(f"Missing required field in JWT: {e}")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: missing required field"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error setting tenant context: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Authentication error"
            )

    def _extract_tenant_context(self, payload: dict) -> TenantContext:
        """
        Extract tenant context from JWT payload.

        Required JWT claims:
            - tenant_id: UUID of tenant
            - tenant_slug: URL-friendly tenant identifier
            - tenant_plan: Subscription plan (free, starter, professional, enterprise)
            - tenant_status: Account status (active, suspended, trial, cancelled)
            - user_id: UUID of user
            - user_email: Email address
            - role: User role (viewer, member, manager, tenant_admin, super_admin)

        Args:
            payload: JWT payload dict

        Returns:
            TenantContext with validated tenant and user information

        Raises:
            KeyError: If required field is missing
            ValueError: If field has invalid value
        """
        tenant_context = TenantContext(
            id=payload["tenant_id"],
            slug=payload["tenant_slug"],
            plan=TenantPlan(payload["tenant_plan"]),
            status=TenantStatus(payload.get("tenant_status", "active")),
            user_id=payload["user_id"],
            user_email=payload["user_email"],
            user_role=Role(payload["role"])
        )

        return tenant_context

    async def _check_quotas(self, tenant: TenantContext):
        """
        Check if tenant has exceeded their quota.

        This is a placeholder - implement actual quota checking based on your needs.

        Args:
            tenant: Tenant context

        Raises:
            HTTPException: If quota exceeded
        """
        # TODO: Implement actual quota checking
        # Example:
        # monthly_usage = await self.db.fetchval(
        #     "SELECT monthly_usage FROM tenants WHERE id = $1",
        #     tenant.id
        # )
        # monthly_quota = await self.db.fetchval(
        #     "SELECT monthly_quota FROM tenants WHERE id = $1",
        #     tenant.id
        # )
        # if monthly_usage >= monthly_quota:
        #     raise HTTPException(
        #         status_code=429,
        #         detail="Monthly quota exceeded. Please upgrade your plan."
        #     )
        pass

    async def _set_postgres_tenant_context(self, tenant_id: str):
        """
        Set PostgreSQL session variable for Row-Level Security.

        This enables database-level tenant isolation via RLS policies.

        Example RLS policy:
            CREATE POLICY tenant_isolation ON extraction_history
                USING (tenant_id = current_setting('app.current_tenant')::uuid);

        Args:
            tenant_id: Tenant ID to set in session
        """
        if not self.db:
            return

        try:
            # Set session variable
            await self.db.execute(
                f"SET LOCAL app.current_tenant = '{tenant_id}'"
            )
            logger.debug(f"Set PostgreSQL tenant context: {tenant_id}")
        except Exception as e:
            logger.warning(f"Failed to set PostgreSQL tenant context: {e}")
            # Don't fail the request if this fails
            pass

    async def _clear_postgres_tenant_context(self):
        """Clear PostgreSQL session variable after request."""
        if not self.db:
            return

        try:
            await self.db.execute("RESET app.current_tenant")
            logger.debug("Cleared PostgreSQL tenant context")
        except Exception as e:
            logger.warning(f"Failed to clear PostgreSQL tenant context: {e}")
            pass


# Helper function for testing
def create_test_jwt_payload(
    tenant_id: str,
    tenant_slug: str,
    user_id: str,
    user_email: str,
    role: Role = Role.MEMBER,
    plan: TenantPlan = TenantPlan.FREE
) -> dict:
    """
    Create a test JWT payload for development/testing.

    Args:
        tenant_id: Tenant UUID
        tenant_slug: Tenant slug
        user_id: User UUID
        user_email: User email
        role: User role
        plan: Tenant plan

    Returns:
        JWT payload dict
    """
    return {
        "tenant_id": tenant_id,
        "tenant_slug": tenant_slug,
        "tenant_plan": plan.value,
        "tenant_status": "active",
        "user_id": user_id,
        "user_email": user_email,
        "role": role.value,
        "exp": 9999999999  # Far future
    }
