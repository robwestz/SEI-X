"""
Database connection pool management using asyncpg.

This module provides async PostgreSQL connection pooling with:
- Automatic connection management
- Query execution helpers
- Transaction support
- Row-Level Security session variable setting
"""

import asyncpg
import os
import logging
from typing import Optional, Any, List, Dict
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


async def get_db_pool() -> asyncpg.Pool:
    """
    Get or create the database connection pool.

    Returns:
        asyncpg.Pool: Database connection pool

    Raises:
        RuntimeError: If database URL is not configured
        asyncpg.PostgresError: If connection fails
    """
    global _pool

    if _pool is None:
        db_url = os.getenv("SIE_X_DB_URL")
        if not db_url:
            raise RuntimeError(
                "SIE_X_DB_URL environment variable not set. "
                "Example: postgresql://user:pass@localhost:5432/siex"
            )

        pool_size = int(os.getenv("SIE_X_DB_POOL_SIZE", "10"))
        max_overflow = int(os.getenv("SIE_X_DB_MAX_OVERFLOW", "20"))

        logger.info(f"Creating database pool (size: {pool_size}, max: {max_overflow})")

        try:
            _pool = await asyncpg.create_pool(
                db_url,
                min_size=pool_size,
                max_size=pool_size + max_overflow,
                command_timeout=60,
                # SSL mode (required for production)
                ssl=os.getenv("SIE_X_DB_SSL_MODE") == "require",
            )

            logger.info("✅ Database pool created successfully")

            # Test connection
            async with _pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                logger.info("✅ Database connection test passed")

        except Exception as e:
            logger.error(f"❌ Failed to create database pool: {e}")
            raise

    return _pool


async def close_db_pool():
    """Close the database connection pool."""
    global _pool

    if _pool:
        await _pool.close()
        logger.info("Database pool closed")
        _pool = None


@asynccontextmanager
async def get_connection():
    """
    Context manager for database connections.

    Usage:
        async with get_connection() as conn:
            result = await conn.fetchval("SELECT 1")
    """
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        yield connection


@asynccontextmanager
async def get_transaction():
    """
    Context manager for database transactions.

    Usage:
        async with get_transaction() as conn:
            await conn.execute("INSERT INTO users ...")
            await conn.execute("INSERT INTO tenants ...")
            # Auto-commit if no exception, auto-rollback on exception
    """
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        async with connection.transaction():
            yield connection


async def execute_query(
    query: str,
    *args,
    tenant_id: Optional[str] = None,
    timeout: Optional[float] = None
) -> str:
    """
    Execute a query (INSERT, UPDATE, DELETE) and return status.

    Args:
        query: SQL query
        *args: Query parameters
        tenant_id: Optional tenant ID to set for RLS
        timeout: Optional query timeout in seconds

    Returns:
        Status message from PostgreSQL

    Example:
        await execute_query(
            "INSERT INTO users (id, email) VALUES ($1, $2)",
            user_id, email,
            tenant_id=tenant.id
        )
    """
    async with get_connection() as conn:
        # Set RLS tenant context if provided
        if tenant_id:
            await conn.execute(f"SET LOCAL app.current_tenant = '{tenant_id}'")

        result = await conn.execute(query, *args, timeout=timeout)
        return result


async def fetch_one(
    query: str,
    *args,
    tenant_id: Optional[str] = None,
    timeout: Optional[float] = None
) -> Optional[asyncpg.Record]:
    """
    Fetch a single row.

    Args:
        query: SQL query
        *args: Query parameters
        tenant_id: Optional tenant ID to set for RLS
        timeout: Optional query timeout in seconds

    Returns:
        Single row as asyncpg.Record or None

    Example:
        user = await fetch_one(
            "SELECT * FROM users WHERE id = $1",
            user_id,
            tenant_id=tenant.id
        )
        if user:
            print(user['email'])
    """
    async with get_connection() as conn:
        # Set RLS tenant context if provided
        if tenant_id:
            await conn.execute(f"SET LOCAL app.current_tenant = '{tenant_id}'")

        result = await conn.fetchrow(query, *args, timeout=timeout)
        return result


async def fetch_all(
    query: str,
    *args,
    tenant_id: Optional[str] = None,
    timeout: Optional[float] = None
) -> List[asyncpg.Record]:
    """
    Fetch all rows.

    Args:
        query: SQL query
        *args: Query parameters
        tenant_id: Optional tenant ID to set for RLS
        timeout: Optional query timeout in seconds

    Returns:
        List of rows as asyncpg.Record

    Example:
        users = await fetch_all(
            "SELECT * FROM users WHERE tenant_id = $1",
            tenant_id
        )
        for user in users:
            print(user['email'])
    """
    async with get_connection() as conn:
        # Set RLS tenant context if provided
        if tenant_id:
            await conn.execute(f"SET LOCAL app.current_tenant = '{tenant_id}'")

        result = await conn.fetch(query, *args, timeout=timeout)
        return result


async def fetch_val(
    query: str,
    *args,
    tenant_id: Optional[str] = None,
    timeout: Optional[float] = None
) -> Any:
    """
    Fetch a single value (first column of first row).

    Args:
        query: SQL query
        *args: Query parameters
        tenant_id: Optional tenant ID to set for RLS
        timeout: Optional query timeout in seconds

    Returns:
        Single value

    Example:
        count = await fetch_val(
            "SELECT COUNT(*) FROM users WHERE tenant_id = $1",
            tenant_id
        )
    """
    async with get_connection() as conn:
        # Set RLS tenant context if provided
        if tenant_id:
            await conn.execute(f"SET LOCAL app.current_tenant = '{tenant_id}'")

        result = await conn.fetchval(query, *args, timeout=timeout)
        return result


# Convenience function for setting RLS context
async def set_tenant_context(conn: asyncpg.Connection, tenant_id: str):
    """
    Set PostgreSQL session variable for Row-Level Security.

    Args:
        conn: Database connection
        tenant_id: Tenant ID to set
    """
    await conn.execute(f"SET LOCAL app.current_tenant = '{tenant_id}'")


async def clear_tenant_context(conn: asyncpg.Connection):
    """
    Clear PostgreSQL session variable for Row-Level Security.

    Args:
        conn: Database connection
    """
    await conn.execute("RESET app.current_tenant")
