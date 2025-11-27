"""API key management service."""

from typing import List
from fastapi import HTTPException

from sie_x.models.tenant import APIKey
from sie_x.db.repositories import APIKeyRepository
from sie_x.services.auth_service import AuthService


class APIKeyService:
    """Service for API key management."""

    @staticmethod
    async def create_api_key(
        tenant_id: str,
        user_id: str,
        name: str
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Args:
            tenant_id: Tenant ID
            user_id: User ID creating the key
            name: Friendly name for the key

        Returns:
            Tuple of (full_key, APIKey object)
            WARNING: full_key is only returned ONCE and should be shown to user immediately!

        Raises:
            HTTPException: If creation fails
        """
        # Generate API key
        full_key, key_prefix, key_hash = AuthService.generate_api_key()

        try:
            # Save to database
            api_key = await APIKeyRepository.create(
                tenant_id=tenant_id,
                created_by=user_id,
                name=name,
                key_prefix=key_prefix,
                key_hash=key_hash
            )

            return (full_key, api_key)

        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to create API key")

    @staticmethod
    async def list_api_keys(tenant_id: str) -> List[APIKey]:
        """
        List all API keys for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            List of API keys
        """
        return await APIKeyRepository.list_by_tenant(tenant_id)

    @staticmethod
    async def revoke_api_key(key_id: str, tenant_id: str) -> None:
        """
        Revoke an API key.

        Args:
            key_id: API key ID
            tenant_id: Tenant ID

        Raises:
            HTTPException: If revocation fails
        """
        try:
            await APIKeyRepository.revoke(key_id, tenant_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to revoke API key")
