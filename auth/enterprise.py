"""
Enterprise authentication with OIDC and SAML support.
"""

from typing import Dict, Any, Optional, List
import json
import jwt
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from authlib.integrations.starlette_client import OAuth
from python3_saml import OneLogin_Saml2_Auth, OneLogin_Saml2_Utils
import ldap3
from dataclasses import dataclass
import redis.asyncio as redis
from datetime import datetime, timedelta


@dataclass
class UserInfo:
    """Authenticated user information."""
    id: str
    email: str
    name: str
    roles: List[str]
    groups: List[str]
    metadata: Dict[str, Any]
    auth_method: str  # 'jwt', 'oidc', 'saml', 'ldap'


class EnterpriseAuthManager:
    """Unified authentication manager for enterprise SSO."""

    def __init__(
            self,
            oidc_providers: Dict[str, Dict[str, str]],
            saml_config: Dict[str, Any],
            ldap_config: Optional[Dict[str, Any]] = None,
            redis_client: Optional[redis.Redis] = None
    ):
        self.oidc_providers = oidc_providers
        self.saml_config = saml_config
        self.ldap_config = ldap_config
        self.redis = redis_client

        # Initialize OAuth clients for OIDC
        self.oauth = OAuth()
        self._init_oidc_providers()

    def _init_oidc_providers(self):
        """Initialize OIDC providers."""
        for name, config in self.oidc_providers.items():
            self.oauth.register(
                name=name,
                client_id=config['client_id'],
                client_secret=config['client_secret'],
                server_metadata_url=config['discovery_url'],
                client_kwargs={
                    'scope': config.get('scope', 'openid email profile')
                }
            )

    async def authenticate_oidc(
            self,
            provider: str,
            authorization_code: str,
            redirect_uri: str
    ) -> UserInfo:
        """Authenticate using OIDC provider."""
        if provider not in self.oidc_providers:
            raise HTTPException(400, f"Unknown OIDC provider: {provider}")

        client = self.oauth.create_client(provider)

        # Exchange code for token
        token = await client.authorize_access_token(
            code=authorization_code,
            redirect_uri=redirect_uri
        )

        # Get user info
        user_info = token.get('userinfo')
        if not user_info:
            resp = await client.get('userinfo', token=token)
            user_info = resp.json()

        # Map OIDC claims to UserInfo
        return self._map_oidc_claims(user_info, provider)

    def authenticate_saml(self, request: Request) -> UserInfo:
        """Authenticate using SAML."""
        saml_auth = self._prepare_saml_auth(request)
        saml_auth.process_response()

        if not saml_auth.is_authenticated():
            errors = saml_auth.get_errors()
            raise HTTPException(401, f"SAML authentication failed: {errors}")

        attributes = saml_auth.get_attributes()
        nameid = saml_auth.get_nameid()

        # Map SAML attributes to UserInfo
        return self._map_saml_attributes(nameid, attributes)

    async def authenticate_ldap(
            self,
            username: str,
            password: str
    ) -> UserInfo:
        """Authenticate using LDAP."""
        if not self.ldap_config:
            raise HTTPException(500, "LDAP not configured")

        server = ldap3.Server(
            self.ldap_config['server'],
            use_ssl=self.ldap_config.get('use_ssl', True)
        )

        # Bind with user credentials
        user_dn = self.ldap_config['user_dn_template'].format(username=username)
        conn = ldap3.Connection(
            server,
            user=user_dn,
            password=password,
            auto_bind=True
        )

        # Search for user attributes
        conn.search(
            user_dn,
            '(objectclass=person)',
            attributes=['mail', 'cn', 'memberOf']
        )

        if not conn.entries:
            raise HTTPException(401, "LDAP authentication failed")

        entry = conn.entries[0]

        # Extract groups
        groups = []
        if 'memberOf' in entry:
            groups = [self._extract_cn(dn) for dn in entry.memberOf]

        return UserInfo(
            id=username,
            email=str(entry.mail) if 'mail' in entry else f"{username}@ldap",
            name=str(entry.cn) if 'cn' in entry else username,
            roles=self._map_ldap_groups_to_roles(groups),
            groups=groups,
            metadata={'ldap_dn': user_dn},
            auth_method='ldap'
        )

    def _prepare_saml_auth(self, request: Request) -> OneLogin_Saml2_Auth:
        """Prepare SAML authentication object."""
        req = {
            'https': 'on' if request.url.scheme == 'https' else 'off',
            'http_host': request.headers.get('host'),
            'script_name': request.url.path,
            'get_data': dict(request.query_params),
            'post_data': {}  # Would be populated from request body
        }

        return OneLogin_Saml2_Auth(req, self.saml_config)

    def _map_oidc_claims(self, claims: Dict[str, Any], provider: str) -> UserInfo:
        """Map OIDC claims to UserInfo."""
        return UserInfo(
            id=claims.get('sub', ''),
            email=claims.get('email', ''),
            name=claims.get('name', claims.get('given_name', '') + ' ' + claims.get('family_name', '')),
            roles=self._extract_oidc_roles(claims, provider),
            groups=claims.get('groups', []),
            metadata={
                'provider': provider,
                'claims': claims
            },
            auth_method='oidc'
        )

    def _map_saml_attributes(self, nameid: str, attributes: Dict[str, List[str]]) -> UserInfo:
        """Map SAML attributes to UserInfo."""
        return UserInfo(
            id=nameid,
            email=attributes.get('email', [nameid])[0],
            name=attributes.get('displayName', [nameid])[0],
            roles=self._extract_saml_roles(attributes),
            groups=attributes.get('groups', []),
            metadata={
                'attributes': attributes
            },
            auth_method='saml'
        )

    def _extract_oidc_roles(self, claims: Dict[str, Any], provider: str) -> List[str]:
        """Extract roles from OIDC claims."""
        # Provider-specific role mapping
        if provider == 'azure':
            return claims.get('roles', [])
        elif provider == 'okta':
            return claims.get('groups', [])
        elif provider == 'auth0':
            return claims.get('permissions', [])
        else:
            return claims.get('roles', claims.get('groups', []))

    def _extract_saml_roles(self, attributes: Dict[str, List[str]]) -> List[str]:
        """Extract roles from SAML attributes."""
        roles = []

        # Common SAML role attribute names
        role_attrs = ['roles', 'groups', 'memberOf', 'Role']

        for attr in role_attrs:
            if attr in attributes:
                roles.extend(attributes[attr])

        return roles

    def _map_ldap_groups_to_roles(self, groups: List[str]) -> List[str]:
        """Map LDAP groups to application roles."""
        role_mapping = self.ldap_config.get('group_role_mapping', {})
        roles = []

        for group in groups:
            if group in role_mapping:
                roles.append(role_mapping[group])
            else:
                # Use group name as role
                roles.append(group.lower().replace(' ', '_'))

        return roles

    def _extract_cn(self, dn: str) -> str:
        """Extract CN from LDAP DN."""
        parts = dn.split(',')
        for part in parts:
            if part.strip().startswith('CN='):
                return part.strip()[3:]
        return dn


class TokenManager:
    """Manage authentication tokens with Redis backend."""

    def __init__(self, redis_client: redis.Redis, secret: str):
        self.redis = redis_client
        self.secret = secret

    async def create_token(self, user_info: UserInfo, ttl: int = 3600) -> str:
        """Create JWT token for user."""
        payload = {
            'sub': user_info.id,
            'email': user_info.email,
            'name': user_info.name,
            'roles': user_info.roles,
            'auth_method': user_info.auth_method,
            'exp': datetime.utcnow() + timedelta(seconds=ttl),
            'iat': datetime.utcnow()
        }

        token = jwt.encode(payload, self.secret, algorithm='HS256')

        # Store in Redis for validation
        await self.redis.setex(
            f"token:{user_info.id}:{token[-8:]}",
            ttl,
            json.dumps({
                'user_id': user_info.id,
                'roles': user_info.roles,
                'created_at': datetime.utcnow().isoformat()
            })
        )

        return token

    async def validate_token(self, token: str) -> Optional[UserInfo]:
        """Validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])

            # Check if token exists in Redis
            token_key = f"token:{payload['sub']}:{token[-8:]}"
            if not await self.redis.exists(token_key):
                return None

            return UserInfo(
                id=payload['sub'],
                email=payload['email'],
                name=payload['name'],
                roles=payload['roles'],
                groups=[],  # Would need to fetch from source
                metadata={'token_iat': payload['iat']},
                auth_method=payload['auth_method']
            )

        except jwt.PyJWTError:
            return None

    async def revoke_token(self, token: str):
        """Revoke a token."""
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            token_key = f"token:{payload['sub']}:{token[-8:]}"
            await self.redis.delete(token_key)

            # Add to blacklist
            ttl = payload['exp'] - datetime.utcnow().timestamp()
            if ttl > 0:
                await self.redis.setex(f"blacklist:{token}", int(ttl), "1")

        except jwt.PyJWTError:
            pass


# FastAPI Dependencies
async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
        token_manager: TokenManager = Depends(get_token_manager)
) -> UserInfo:
    """FastAPI dependency to get current user."""
    user = await token_manager.validate_token(credentials.credentials)

    if not user:
        raise HTTPException(401, "Invalid authentication")

    return user


def require_roles(*roles: str):
    """Decorator to require specific roles."""

    async def role_checker(user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if not any(role in user.roles for role in roles):
            raise HTTPException(403, f"Requires one of roles: {roles}")
        return user

    return role_checker


def require_groups(*groups: str):
    """Decorator to require specific groups."""

    async def group_checker(user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if not any(group in user.groups for group in groups):
            raise HTTPException(403, f"Requires one of groups: {groups}")
        return user

    return group_checker