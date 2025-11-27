"""
Repository pattern for tenant-isolated database operations.

This module provides repositories for all main entities with automatic
tenant isolation enforced at the database level using RLS.
"""

import hashlib
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from sie_x.db.connection import fetch_one, fetch_all, execute_query, fetch_val, get_transaction
from sie_x.models.tenant import Tenant, User, APIKey, TenantPlan, TenantStatus, Role
from sie_x.core.models import Keyword


class TenantRepository:
    """
    Repository for tenant operations.

    Note: Most operations require super_admin role.
    """

    @staticmethod
    async def create(name: str, slug: str, plan: TenantPlan = TenantPlan.FREE) -> Tenant:
        """
        Create a new tenant.

        Args:
            name: Organization name
            slug: URL-friendly identifier
            plan: Subscription plan

        Returns:
            Created Tenant

        Raises:
            asyncpg.UniqueViolationError: If slug already exists
        """
        query = """
            INSERT INTO tenants (id, name, slug, plan, status, monthly_quota, monthly_usage)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, name, slug, status, plan, monthly_quota, monthly_usage, created_at, updated_at
        """

        # Default quotas by plan
        quotas = {
            TenantPlan.FREE: 10000,
            TenantPlan.STARTER: 100000,
            TenantPlan.PROFESSIONAL: 1000000,
            TenantPlan.ENTERPRISE: 10000000,
        }

        tenant_id = str(uuid.uuid4())
        row = await fetch_one(
            query,
            tenant_id,
            name,
            slug,
            plan.value,
            TenantStatus.ACTIVE.value,
            quotas[plan],
            0
        )

        return Tenant(
            id=row['id'],
            name=row['name'],
            slug=row['slug'],
            status=TenantStatus(row['status']),
            plan=TenantPlan(row['plan']),
            monthly_quota=row['monthly_quota'],
            monthly_usage=row['monthly_usage'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    @staticmethod
    async def get_by_id(tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        query = """
            SELECT id, name, slug, status, plan, monthly_quota, monthly_usage,
                   created_at, updated_at, metadata
            FROM tenants
            WHERE id = $1
        """
        row = await fetch_one(query, tenant_id)
        if not row:
            return None

        return Tenant(
            id=row['id'],
            name=row['name'],
            slug=row['slug'],
            status=TenantStatus(row['status']),
            plan=TenantPlan(row['plan']),
            monthly_quota=row['monthly_quota'],
            monthly_usage=row['monthly_usage'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            metadata=row['metadata'] or {}
        )

    @staticmethod
    async def get_by_slug(slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        query = """
            SELECT id, name, slug, status, plan, monthly_quota, monthly_usage,
                   created_at, updated_at, metadata
            FROM tenants
            WHERE slug = $1
        """
        row = await fetch_one(query, slug)
        if not row:
            return None

        return Tenant(
            id=row['id'],
            name=row['name'],
            slug=row['slug'],
            status=TenantStatus(row['status']),
            plan=TenantPlan(row['plan']),
            monthly_quota=row['monthly_quota'],
            monthly_usage=row['monthly_usage'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            metadata=row['metadata'] or {}
        )

    @staticmethod
    async def increment_usage(tenant_id: str, amount: int = 1) -> None:
        """Increment monthly usage for tenant."""
        query = """
            UPDATE tenants
            SET monthly_usage = monthly_usage + $1
            WHERE id = $2
        """
        await execute_query(query, amount, tenant_id)

    @staticmethod
    async def check_quota(tenant_id: str) -> bool:
        """
        Check if tenant has quota remaining.

        Returns:
            True if quota available, False if exceeded
        """
        query = """
            SELECT monthly_usage < monthly_quota AS has_quota
            FROM tenants
            WHERE id = $1
        """
        result = await fetch_val(query, tenant_id)
        return result if result is not None else False


class UserRepository:
    """
    Repository for user operations (tenant-isolated).
    """

    @staticmethod
    async def create(
        tenant_id: str,
        email: str,
        password_hash: str,
        first_name: str,
        last_name: str,
        role: Role = Role.MEMBER
    ) -> User:
        """
        Create a new user within a tenant.

        Args:
            tenant_id: Tenant ID
            email: User email (unique within tenant)
            password_hash: Hashed password (bcrypt)
            first_name: First name
            last_name: Last name
            role: User role

        Returns:
            Created User

        Raises:
            asyncpg.UniqueViolationError: If email exists in tenant
        """
        query = """
            INSERT INTO users (id, tenant_id, email, password_hash, first_name, last_name, role, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, tenant_id, email, first_name, last_name, role, is_active, created_at, last_login_at
        """

        user_id = str(uuid.uuid4())
        row = await fetch_one(
            query,
            user_id,
            tenant_id,
            email,
            password_hash,
            first_name,
            last_name,
            role.value,
            True,
            tenant_id=tenant_id
        )

        return User(
            id=row['id'],
            tenant_id=row['tenant_id'],
            email=row['email'],
            first_name=row['first_name'],
            last_name=row['last_name'],
            role=Role(row['role']),
            is_active=row['is_active'],
            created_at=row['created_at'],
            last_login_at=row['last_login_at']
        )

    @staticmethod
    async def get_by_id(user_id: str, tenant_id: str) -> Optional[User]:
        """
        Get user by ID (tenant-isolated).

        Args:
            user_id: User ID
            tenant_id: Tenant ID (for RLS)

        Returns:
            User or None
        """
        query = """
            SELECT id, tenant_id, email, first_name, last_name, role, is_active,
                   created_at, last_login_at, metadata
            FROM users
            WHERE id = $1 AND tenant_id = $2
        """
        row = await fetch_one(query, user_id, tenant_id, tenant_id=tenant_id)
        if not row:
            return None

        return User(
            id=row['id'],
            tenant_id=row['tenant_id'],
            email=row['email'],
            first_name=row['first_name'],
            last_name=row['last_name'],
            role=Role(row['role']),
            is_active=row['is_active'],
            created_at=row['created_at'],
            last_login_at=row['last_login_at'],
            metadata=row['metadata'] or {}
        )

    @staticmethod
    async def get_by_email(email: str, tenant_id: str) -> Optional[tuple[User, str]]:
        """
        Get user by email (tenant-isolated) including password hash.

        Args:
            email: User email
            tenant_id: Tenant ID (for RLS)

        Returns:
            Tuple of (User, password_hash) or None
        """
        query = """
            SELECT id, tenant_id, email, password_hash, first_name, last_name, role,
                   is_active, created_at, last_login_at, metadata
            FROM users
            WHERE email = $1 AND tenant_id = $2
        """
        row = await fetch_one(query, email, tenant_id, tenant_id=tenant_id)
        if not row:
            return None

        user = User(
            id=row['id'],
            tenant_id=row['tenant_id'],
            email=row['email'],
            first_name=row['first_name'],
            last_name=row['last_name'],
            role=Role(row['role']),
            is_active=row['is_active'],
            created_at=row['created_at'],
            last_login_at=row['last_login_at'],
            metadata=row['metadata'] or {}
        )

        return (user, row['password_hash'])

    @staticmethod
    async def update_last_login(user_id: str, tenant_id: str) -> None:
        """Update user's last login timestamp."""
        query = """
            UPDATE users
            SET last_login_at = NOW()
            WHERE id = $1 AND tenant_id = $2
        """
        await execute_query(query, user_id, tenant_id, tenant_id=tenant_id)

    @staticmethod
    async def list_by_tenant(tenant_id: str) -> List[User]:
        """List all users in a tenant."""
        query = """
            SELECT id, tenant_id, email, first_name, last_name, role, is_active,
                   created_at, last_login_at, metadata
            FROM users
            WHERE tenant_id = $1
            ORDER BY created_at DESC
        """
        rows = await fetch_all(query, tenant_id, tenant_id=tenant_id)

        return [
            User(
                id=row['id'],
                tenant_id=row['tenant_id'],
                email=row['email'],
                first_name=row['first_name'],
                last_name=row['last_name'],
                role=Role(row['role']),
                is_active=row['is_active'],
                created_at=row['created_at'],
                last_login_at=row['last_login_at'],
                metadata=row['metadata'] or {}
            )
            for row in rows
        ]


class APIKeyRepository:
    """Repository for API key operations (tenant-isolated)."""

    @staticmethod
    async def create(
        tenant_id: str,
        created_by: str,
        name: str,
        key_prefix: str,
        key_hash: str
    ) -> APIKey:
        """
        Create a new API key.

        Args:
            tenant_id: Tenant ID
            created_by: User ID who created the key
            name: Friendly name for the key
            key_prefix: First 8 chars of key (for display)
            key_hash: SHA-256 hash of full key

        Returns:
            Created APIKey
        """
        query = """
            INSERT INTO api_keys (id, tenant_id, created_by, name, key_prefix, key_hash, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, tenant_id, created_by, name, key_prefix, key_hash, is_active,
                      created_at, expires_at, last_used_at
        """

        key_id = str(uuid.uuid4())
        row = await fetch_one(
            query,
            key_id,
            tenant_id,
            created_by,
            name,
            key_prefix,
            key_hash,
            True,
            tenant_id=tenant_id
        )

        return APIKey(
            id=row['id'],
            tenant_id=row['tenant_id'],
            created_by=row['created_by'],
            name=row['name'],
            key_prefix=row['key_prefix'],
            key_hash=row['key_hash'],
            is_active=row['is_active'],
            created_at=row['created_at'],
            expires_at=row['expires_at'],
            last_used_at=row['last_used_at']
        )

    @staticmethod
    async def get_by_hash(key_hash: str) -> Optional[APIKey]:
        """
        Get API key by hash.

        Note: This does NOT use tenant_id filter since we need to find the tenant
        from the API key itself. The key_hash is unique across all tenants.
        """
        query = """
            SELECT id, tenant_id, created_by, name, key_prefix, key_hash, is_active,
                   created_at, expires_at, last_used_at, usage_count
            FROM api_keys
            WHERE key_hash = $1 AND is_active = TRUE
        """
        row = await fetch_one(query, key_hash)
        if not row:
            return None

        return APIKey(
            id=row['id'],
            tenant_id=row['tenant_id'],
            created_by=row['created_by'],
            name=row['name'],
            key_prefix=row['key_prefix'],
            key_hash=row['key_hash'],
            is_active=row['is_active'],
            created_at=row['created_at'],
            expires_at=row['expires_at'],
            last_used_at=row['last_used_at']
        )

    @staticmethod
    async def update_last_used(key_hash: str) -> None:
        """Update API key's last used timestamp and usage count."""
        query = """
            UPDATE api_keys
            SET last_used_at = NOW(), usage_count = usage_count + 1
            WHERE key_hash = $1
        """
        await execute_query(query, key_hash)

    @staticmethod
    async def list_by_tenant(tenant_id: str) -> List[APIKey]:
        """List all API keys for a tenant."""
        query = """
            SELECT id, tenant_id, created_by, name, key_prefix, key_hash, is_active,
                   created_at, expires_at, last_used_at
            FROM api_keys
            WHERE tenant_id = $1
            ORDER BY created_at DESC
        """
        rows = await fetch_all(query, tenant_id, tenant_id=tenant_id)

        return [
            APIKey(
                id=row['id'],
                tenant_id=row['tenant_id'],
                created_by=row['created_by'],
                name=row['name'],
                key_prefix=row['key_prefix'],
                key_hash=row['key_hash'],
                is_active=row['is_active'],
                created_at=row['created_at'],
                expires_at=row['expires_at'],
                last_used_at=row['last_used_at']
            )
            for row in rows
        ]

    @staticmethod
    async def revoke(key_id: str, tenant_id: str) -> None:
        """Revoke an API key."""
        query = """
            UPDATE api_keys
            SET is_active = FALSE
            WHERE id = $1 AND tenant_id = $2
        """
        await execute_query(query, key_id, tenant_id, tenant_id=tenant_id)


class ExtractionHistoryRepository:
    """Repository for extraction history (tenant-isolated)."""

    @staticmethod
    async def create(
        tenant_id: str,
        user_id: str,
        text: str,
        keywords: List[Keyword],
        processing_time: float,
        source: str = "api",
        language: str = "en",
        options: Dict[str, Any] = None
    ) -> str:
        """
        Save extraction to history.

        Args:
            tenant_id: Tenant ID
            user_id: User ID who performed extraction
            text: Input text
            keywords: Extracted keywords
            processing_time: Processing time in seconds
            source: Source of extraction (api, sdk, web)
            language: Language code
            options: Extraction options

        Returns:
            Extraction ID
        """
        # Hash text for deduplication
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        query = """
            INSERT INTO extraction_history
            (id, tenant_id, user_id, text_hash, text_length, keywords, processing_time, source, language, options)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """

        extraction_id = str(uuid.uuid4())
        keywords_json = json.dumps([k.model_dump() for k in keywords])

        row = await fetch_one(
            query,
            extraction_id,
            tenant_id,
            user_id,
            text_hash,
            len(text),
            keywords_json,
            processing_time,
            source,
            language,
            json.dumps(options or {}),
            tenant_id=tenant_id
        )

        return row['id']

    @staticmethod
    async def list_by_user(
        tenant_id: str,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get extraction history for a user.

        Args:
            tenant_id: Tenant ID
            user_id: User ID
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of extraction records
        """
        query = """
            SELECT id, text_hash, text_length, keywords, processing_time, source, language, created_at
            FROM extraction_history
            WHERE tenant_id = $1 AND user_id = $2
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
        """
        rows = await fetch_all(query, tenant_id, user_id, limit, offset, tenant_id=tenant_id)

        return [
            {
                "id": row['id'],
                "text_hash": row['text_hash'],
                "text_length": row['text_length'],
                "keywords": json.loads(row['keywords']),
                "processing_time": row['processing_time'],
                "source": row['source'],
                "language": row['language'],
                "created_at": row['created_at'].isoformat()
            }
            for row in rows
        ]

    @staticmethod
    async def get_stats(tenant_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for a tenant.

        Returns:
            Dict with total_extractions, total_keywords, avg_processing_time
        """
        query = """
            SELECT
                COUNT(*) as total_extractions,
                SUM(jsonb_array_length(keywords)) as total_keywords,
                AVG(processing_time) as avg_processing_time,
                SUM(text_length) as total_text_length
            FROM extraction_history
            WHERE tenant_id = $1
        """
        row = await fetch_one(query, tenant_id, tenant_id=tenant_id)

        return {
            "total_extractions": row['total_extractions'] or 0,
            "total_keywords": row['total_keywords'] or 0,
            "avg_processing_time": float(row['avg_processing_time']) if row['avg_processing_time'] else 0.0,
            "total_text_length": row['total_text_length'] or 0
        }
