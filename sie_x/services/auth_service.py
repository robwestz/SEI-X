"""
Authentication service for SIE-X.

Handles:
- Password hashing and verification (bcrypt)
- JWT token generation and validation
- Token blacklisting (revocation)
- API key authentication
"""

import os
import jwt
import bcrypt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from fastapi import HTTPException

from sie_x.models.tenant import (
    Tenant, User, TenantContext, LoginRequest, LoginResponse,
    Role, TenantPlan, TenantStatus
)
from sie_x.db.repositories import UserRepository, TenantRepository, APIKeyRepository
from sie_x.cache.redis_cache import FallbackCache


class AuthService:
    """
    Authentication service handling login, JWT, and API keys.
    """

    def __init__(self, cache: Optional[FallbackCache] = None):
        """
        Initialize auth service.

        Args:
            cache: Optional cache for token blacklist and API key lookup
        """
        self.cache = cache
        self.jwt_secret = os.getenv("SIE_X_JWT_SECRET", "change-me-in-production")
        self.jwt_algorithm = os.getenv("SIE_X_JWT_ALGORITHM", "HS256")
        self.token_expiry = int(os.getenv("SIE_X_TOKEN_EXPIRY", "3600"))  # 1 hour

        # Validate JWT secret in production
        if os.getenv("SIE_X_ENV") == "production" and len(self.jwt_secret) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters in production")

    # =============================================================================
    # PASSWORD HASHING
    # =============================================================================

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Bcrypt hash as string
        """
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=12)  # 12 rounds = good balance
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Plain text password
            password_hash: Bcrypt hash

        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                password_hash.encode('utf-8')
            )
        except Exception:
            return False

    # =============================================================================
    # JWT TOKENS
    # =============================================================================

    def create_token(
        self,
        tenant: Tenant,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT token for a user.

        Args:
            tenant: Tenant object
            user: User object
            expires_delta: Optional custom expiration time

        Returns:
            JWT token as string
        """
        if expires_delta is None:
            expires_delta = timedelta(seconds=self.token_expiry)

        # Calculate expiration
        expire = datetime.utcnow() + expires_delta

        # JWT payload with all required fields for TenantContext
        payload = {
            # Tenant information
            "tenant_id": tenant.id,
            "tenant_slug": tenant.slug,
            "tenant_plan": tenant.plan.value,
            "tenant_status": tenant.status.value,

            # User information
            "user_id": user.id,
            "user_email": user.email,
            "role": user.role.value,

            # Token metadata
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }

        # Encode token
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token

    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload dict

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )

            # Check if token is blacklisted
            if self.cache and self.cache.is_available:
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                blacklisted = self.cache.get(f"blacklist:{token_hash}")
                if blacklisted:
                    raise HTTPException(status_code=401, detail="Token has been revoked")

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def blacklist_token(self, token: str) -> None:
        """
        Blacklist a token (logout/revocation).

        Args:
            token: JWT token to blacklist
        """
        if not self.cache or not self.cache.is_available:
            raise HTTPException(
                status_code=503,
                detail="Token blacklist unavailable (cache down)"
            )

        try:
            # Decode to get expiry
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={"verify_exp": False}  # Don't verify exp, we need it
            )

            # Calculate TTL (time until token expires)
            exp = datetime.fromtimestamp(payload.get("exp", 0))
            ttl = int((exp - datetime.utcnow()).total_seconds())

            if ttl > 0:
                # Store token hash in blacklist with TTL
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                await self.cache.set(
                    f"blacklist:{token_hash}",
                    "1",
                    ttl=ttl
                )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Failed to blacklist token"
            )

    # =============================================================================
    # AUTHENTICATION
    # =============================================================================

    async def authenticate_user(
        self,
        email: str,
        password: str,
        tenant_slug: str
    ) -> Tuple[Tenant, User]:
        """
        Authenticate a user with email + password.

        Args:
            email: User email
            password: Plain text password
            tenant_slug: Tenant slug

        Returns:
            Tuple of (Tenant, User)

        Raises:
            HTTPException: If authentication fails
        """
        # Get tenant by slug
        tenant = await TenantRepository.get_by_slug(tenant_slug)
        if not tenant:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check tenant is active
        if tenant.status != TenantStatus.ACTIVE:
            raise HTTPException(
                status_code=403,
                detail=f"Tenant account is {tenant.status.value}"
            )

        # Get user by email
        result = await UserRepository.get_by_email(email, tenant.id)
        if not result:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user, password_hash = result

        # Check user is active
        if not user.is_active:
            raise HTTPException(status_code=403, detail="User account is disabled")

        # Verify password
        if not self.verify_password(password, password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Update last login
        await UserRepository.update_last_login(user.id, tenant.id)

        return (tenant, user)

    async def login(self, request: LoginRequest) -> LoginResponse:
        """
        Perform login and return JWT token.

        Args:
            request: LoginRequest with email, password, tenant_slug

        Returns:
            LoginResponse with access token and user info
        """
        # Authenticate user
        tenant, user = await self.authenticate_user(
            request.email,
            request.password,
            request.tenant_slug
        )

        # Create JWT token
        access_token = self.create_token(tenant, user)

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.token_expiry,
            user=user,
            tenant=tenant
        )

    async def logout(self, token: str) -> None:
        """
        Logout user by blacklisting token.

        Args:
            token: JWT token to blacklist
        """
        await self.blacklist_token(token)

    # =============================================================================
    # API KEY AUTHENTICATION
    # =============================================================================

    @staticmethod
    def generate_api_key() -> Tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (full_key, key_prefix, key_hash)

        Example:
            full_key = "sk_live_abc123def456..."  # Give to user (ONCE)
            key_prefix = "sk_live_"              # Store for display
            key_hash = "sha256hash..."            # Store in database
        """
        # Generate random key
        random_part = secrets.token_urlsafe(32)  # 32 bytes = 43 chars base64
        full_key = f"sk_live_{random_part}"

        # Extract prefix (first 8 chars)
        key_prefix = full_key[:8]

        # Hash for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()

        return (full_key, key_prefix, key_hash)

    async def authenticate_api_key(self, api_key: str) -> Tuple[Tenant, User]:
        """
        Authenticate using API key.

        Args:
            api_key: Full API key (sk_live_...)

        Returns:
            Tuple of (Tenant, User)

        Raises:
            HTTPException: If API key is invalid
        """
        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Look up API key
        api_key_obj = await APIKeyRepository.get_by_hash(key_hash)
        if not api_key_obj:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Check if expired
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="API key has expired")

        # Get tenant
        tenant = await TenantRepository.get_by_id(api_key_obj.tenant_id)
        if not tenant or tenant.status != TenantStatus.ACTIVE:
            raise HTTPException(status_code=403, detail="Tenant account is not active")

        # Get user who created the key
        user = await UserRepository.get_by_id(api_key_obj.created_by, api_key_obj.tenant_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=403, detail="User account is not active")

        # Update last used
        await APIKeyRepository.update_last_used(key_hash)

        return (tenant, user)

    def create_tenant_context_from_token(self, token_payload: Dict[str, Any]) -> TenantContext:
        """
        Create TenantContext from decoded JWT payload.

        Args:
            token_payload: Decoded JWT payload

        Returns:
            TenantContext object
        """
        return TenantContext(
            id=token_payload["tenant_id"],
            slug=token_payload["tenant_slug"],
            plan=TenantPlan(token_payload["tenant_plan"]),
            status=TenantStatus(token_payload["tenant_status"]),
            user_id=token_payload["user_id"],
            user_email=token_payload["user_email"],
            user_role=Role(token_payload["role"])
        )
