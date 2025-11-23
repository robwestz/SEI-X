# Platform Foundation + API Pipelines - Implementation Plan

**Status:** In Progress
**Timeline:** 8-10 weeks
**Investment:** $224K (2 developers × 10 weeks)
**Expected ROI:** $1.79M ARR in Year 1

---

## Executive Summary

This implementation plan combines **Platform Foundation** (Phase 1) with **API Pipelines** (Phase 2) - our killer USP feature that makes SIE-X the "Zapier for SEO with AI".

**What We're Building:**
1. **Platform Foundation** - Production-grade infrastructure (users, auth, billing, storage)
2. **API Pipeline Engine** - Code-first + No-code workflow automation for SEO
3. **Visual Pipeline Builder** - Drag-and-drop UI like Zapier
4. **20 Pre-built Templates** - Ready-to-use SEO automation workflows
5. **Integration Hub** - Connect with 20+ external services

**Why This Is Our Moat:**
- ✅ No competitor has API Pipelines for SEO
- ✅ Zapier doesn't have semantic intelligence
- ✅ Ahrefs/SEMrush don't have workflow automation
- ✅ This makes SIE-X 10x more valuable than just keyword extraction

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Frontend (React + TypeScript)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Pipeline     │  │ Admin        │  │ User         │           │
│  │ Builder      │  │ Dashboard    │  │ Dashboard    │           │
│  │ (React Flow) │  │              │  │              │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼──────────────────┼──────────────────┼───────────────────┘
          │                  │                  │
          │ REST + WS        │ REST             │ REST
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼───────────────────┐
│         ▼                  ▼                  ▼                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              FastAPI Gateway (Port 8000)                 │   │
│  │  - Authentication (JWT, OAuth2, API Keys)                │   │
│  │  - Rate Limiting (Redis)                                 │   │
│  │  - Request Routing                                       │   │
│  │  - WebSocket Support (real-time pipeline updates)       │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│         ┌───────────────┴───────────────┐                        │
│         │                               │                        │
│         ▼                               ▼                        │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │ Platform API     │          │ Pipeline API     │             │
│  │ (Port 8001)      │          │ (Port 8002)      │             │
│  │ - Users          │          │ - Pipeline CRUD  │             │
│  │ - Auth           │          │ - Execution      │             │
│  │ - Billing        │          │ - Templates      │             │
│  │ - Admin          │          │ - Integrations   │             │
│  └────────┬─────────┘          └────────┬─────────┘             │
│           │                             │                        │
│           └──────────┬──────────────────┘                        │
│                      │                                           │
│  ┌───────────────────┴────────────────────────────────────────┐ │
│  │              Celery Workers (Distributed)                  │ │
│  │  - Pipeline Execution (async)                              │ │
│  │  - Trigger Processing (RSS, Schedule, Webhook)             │ │
│  │  - Action Execution (Extract, Transform, Send)             │ │
│  │  - Retry Logic & Error Handling                            │ │
│  └───────────────────┬────────────────────────────────────────┘ │
│                      │                                           │
│  ┌───────────────────┴────────────────────────────────────────┐ │
│  │                   SIE-X Core Engine                        │ │
│  │  - Semantic Extraction (SimpleEngine, EnterpriseEngine)    │ │
│  │  - Transformers (BACOWR, SEO, Legal, Medical, etc.)       │ │
│  │  - Multi-language Support (11 languages)                  │ │
│  │  - Streaming & Batch Processing                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Data Layer (Persistent Storage)             │   │
│  │                                                          │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐       │   │
│  │  │ PostgreSQL          │  │ Redis                │       │   │
│  │  │ - Users & Auth      │  │ - Cache              │       │   │
│  │  │ - Subscriptions     │  │ - Rate Limiting      │       │   │
│  │  │ - Pipelines         │  │ - Celery Queue       │       │   │
│  │  │ - Executions        │  │ - Pipeline State     │       │   │
│  │  │ - Usage Logs        │  │ - WebSocket Sessions │       │   │
│  │  └─────────────────────┘  └─────────────────────┘       │   │
│  │                                                          │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐       │   │
│  │  │ S3 / MinIO          │  │ RabbitMQ             │       │   │
│  │  │ - File Uploads      │  │ - Message Broker     │       │   │
│  │  │ - Export Results    │  │ - Event Bus          │       │   │
│  │  │ - Logs              │  │ - Webhook Delivery   │       │   │
│  │  └─────────────────────┘  └─────────────────────┘       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              External Integrations (20+)                 │   │
│  │  - Slack, Discord, Telegram (notifications)              │   │
│  │  - Airtable, Google Sheets (data storage)                │   │
│  │  - WordPress, HubSpot (publishing)                        │   │
│  │  - SendGrid, Mailgun (email)                              │   │
│  │  - Webhooks (custom integrations)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Platform Foundation (Week 1-4)

### Week 1: Database & Core Infrastructure

#### Day 1-2: PostgreSQL Schema Design & Implementation

**File:** `sie_x/platform/database/schema.sql`

```sql
-- ============================================================================
-- USERS & AUTHENTICATION
-- ============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),  -- NULL for OAuth users
    full_name VARCHAR(255),
    company VARCHAR(255),

    -- Plan & Billing
    plan VARCHAR(50) DEFAULT 'free',  -- free, creator, professional, agency, enterprise
    plan_started_at TIMESTAMP,
    plan_expires_at TIMESTAMP,
    stripe_customer_id VARCHAR(255),

    -- API Keys
    api_key_hash VARCHAR(255),
    api_key_prefix VARCHAR(20),  -- For display (pk_live_abc...)

    -- Usage Limits
    monthly_quota JSONB DEFAULT '{"extractions": 1000, "pipelines": 10, "executions": 100}',
    usage_this_month JSONB DEFAULT '{"extractions": 0, "pipelines": 0, "executions": 0}',

    -- Account Status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    is_admin BOOLEAN DEFAULT false,

    -- OAuth
    oauth_provider VARCHAR(50),  -- google, github, microsoft
    oauth_id VARCHAR(255),

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login_at TIMESTAMP,
    deleted_at TIMESTAMP  -- Soft delete
);

CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_api_key_hash ON users(api_key_hash);
CREATE INDEX idx_users_oauth ON users(oauth_provider, oauth_id);

-- ============================================================================
-- SUBSCRIPTIONS & BILLING
-- ============================================================================

CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Stripe Integration
    stripe_subscription_id VARCHAR(255) UNIQUE,
    stripe_price_id VARCHAR(255),

    -- Plan Details
    plan VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',  -- active, canceled, past_due, trialing

    -- Billing Cycle
    current_period_start TIMESTAMP NOT NULL,
    current_period_end TIMESTAMP NOT NULL,
    trial_end TIMESTAMP,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    canceled_at TIMESTAMP
);

CREATE INDEX idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_stripe ON subscriptions(stripe_subscription_id);

CREATE TABLE invoices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    subscription_id UUID REFERENCES subscriptions(id),

    -- Stripe
    stripe_invoice_id VARCHAR(255) UNIQUE,

    -- Invoice Details
    amount_due INTEGER NOT NULL,  -- in cents
    amount_paid INTEGER DEFAULT 0,
    currency VARCHAR(3) DEFAULT 'usd',
    status VARCHAR(50) DEFAULT 'draft',  -- draft, open, paid, void, uncollectible

    -- Dates
    invoice_date TIMESTAMP NOT NULL,
    due_date TIMESTAMP,
    paid_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- USAGE TRACKING
-- ============================================================================

CREATE TABLE usage_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Resource Used
    resource_type VARCHAR(50) NOT NULL,  -- extraction, pipeline_execution, api_call
    resource_id UUID,  -- Reference to pipeline, extraction, etc.

    -- Usage Details
    credits_used INTEGER DEFAULT 1,
    metadata JSONB,  -- Additional context

    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_logs_user_date ON usage_logs(user_id, created_at DESC);
CREATE INDEX idx_usage_logs_resource ON usage_logs(resource_type, resource_id);

-- Partitioning by month for performance
CREATE TABLE usage_logs_2025_01 PARTITION OF usage_logs
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
-- Add more partitions as needed

-- ============================================================================
-- API KEYS & TOKENS
-- ============================================================================

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Key Details
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(20) NOT NULL,  -- For display
    name VARCHAR(255),  -- User-friendly name

    -- Permissions
    scopes TEXT[] DEFAULT ARRAY['read', 'write'],  -- Granular permissions

    -- Rate Limiting
    rate_limit_per_minute INTEGER DEFAULT 60,
    rate_limit_per_hour INTEGER DEFAULT 1000,

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW(),
    revoked_at TIMESTAMP
);

CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

CREATE TABLE jwt_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Token
    jti VARCHAR(255) UNIQUE NOT NULL,  -- JWT ID
    token_type VARCHAR(50) DEFAULT 'access',  -- access, refresh

    -- Expiry
    expires_at TIMESTAMP NOT NULL,

    -- Revocation
    revoked_at TIMESTAMP,
    revoked_reason VARCHAR(255),

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_jwt_tokens_jti ON jwt_tokens(jti);
CREATE INDEX idx_jwt_tokens_user ON jwt_tokens(user_id);

-- ============================================================================
-- TEAMS & COLLABORATION (Agency+)
-- ============================================================================

CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,

    -- Plan
    plan VARCHAR(50) DEFAULT 'agency',
    max_members INTEGER DEFAULT 5,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE team_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    role VARCHAR(50) DEFAULT 'member',  -- owner, admin, member, viewer

    -- Permissions
    can_create_pipelines BOOLEAN DEFAULT true,
    can_edit_pipelines BOOLEAN DEFAULT true,
    can_delete_pipelines BOOLEAN DEFAULT false,
    can_manage_members BOOLEAN DEFAULT false,

    joined_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(team_id, user_id)
);

-- ============================================================================
-- AUDIT LOGS (Compliance & Security)
-- ============================================================================

CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    -- Event
    event_type VARCHAR(100) NOT NULL,  -- user.login, pipeline.created, etc.
    resource_type VARCHAR(50),
    resource_id UUID,

    -- Details
    ip_address INET,
    user_agent TEXT,
    metadata JSONB,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_user ON audit_logs(user_id, created_at DESC);
CREATE INDEX idx_audit_logs_event ON audit_logs(event_type, created_at DESC);
```

**File:** `sie_x/platform/database/models.py`

```python
# SQLAlchemy ORM Models
from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, ForeignKey, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255))
    full_name = Column(String(255))
    company = Column(String(255))

    # Plan & Billing
    plan = Column(String(50), default='free')
    plan_started_at = Column(DateTime)
    plan_expires_at = Column(DateTime)
    stripe_customer_id = Column(String(255))

    # API Keys
    api_key_hash = Column(String(255), index=True)
    api_key_prefix = Column(String(20))

    # Usage
    monthly_quota = Column(JSON, default={"extractions": 1000, "pipelines": 10, "executions": 100})
    usage_this_month = Column(JSON, default={"extractions": 0, "pipelines": 0, "executions": 0})

    # Account Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)

    # OAuth
    oauth_provider = Column(String(50))
    oauth_id = Column(String(255))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)
    deleted_at = Column(DateTime)

    # Relationships
    subscriptions = relationship("Subscription", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")
    pipelines = relationship("Pipeline", back_populates="user")

    def to_dict(self):
        return {
            'id': str(self.id),
            'email': self.email,
            'full_name': self.full_name,
            'company': self.company,
            'plan': self.plan,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Subscription(Base):
    __tablename__ = 'subscriptions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))

    stripe_subscription_id = Column(String(255), unique=True)
    stripe_price_id = Column(String(255))

    plan = Column(String(50), nullable=False)
    status = Column(String(50), default='active')

    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    trial_end = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    canceled_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="subscriptions")

class APIKey(Base):
    __tablename__ = 'api_keys'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))

    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(20), nullable=False)
    name = Column(String(255))

    scopes = Column(ARRAY(Text), default=['read', 'write'])

    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=1000)

    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)
    expires_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    revoked_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="api_keys")

class UsageLog(Base):
    __tablename__ = 'usage_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))

    resource_type = Column(String(50), nullable=False)
    resource_id = Column(UUID(as_uuid=True))

    credits_used = Column(Integer, default=1)
    metadata = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class AuditLog(Base):
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id', ondelete='SET NULL'))

    event_type = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))

    ip_address = Column(INET)
    user_agent = Column(Text)
    metadata = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
```

#### Day 3-4: Authentication System

**File:** `sie_x/platform/auth/service.py`

```python
from typing import Optional, Dict
from datetime import datetime, timedelta
import jwt
import bcrypt
import secrets
from sqlalchemy.orm import Session
from ..database.models import User, APIKey, JWTToken

class AuthService:
    """Authentication & Authorization Service"""

    def __init__(self, db: Session, jwt_secret: str):
        self.db = db
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = 60
        self.refresh_token_expire_days = 30

    # ============================================================================
    # USER REGISTRATION & LOGIN
    # ============================================================================

    def register_user(
        self,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        company: Optional[str] = None
    ) -> User:
        """Register new user with email/password"""

        # Check if email already exists
        existing = self.db.query(User).filter(User.email == email).first()
        if existing:
            raise ValueError("Email already registered")

        # Hash password
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        # Create user
        user = User(
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            company=company,
            plan='free',
            is_verified=False
        )

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        return user

    def login(self, email: str, password: str) -> Dict:
        """Login with email/password, returns access & refresh tokens"""

        user = self.db.query(User).filter(User.email == email).first()

        if not user or not user.password_hash:
            raise ValueError("Invalid credentials")

        # Verify password
        if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            raise ValueError("Invalid credentials")

        if not user.is_active:
            raise ValueError("Account is inactive")

        # Update last login
        user.last_login_at = datetime.utcnow()
        self.db.commit()

        # Generate tokens
        access_token = self._create_access_token(user)
        refresh_token = self._create_refresh_token(user)

        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'bearer',
            'expires_in': self.access_token_expire_minutes * 60,
            'user': user.to_dict()
        }

    # ============================================================================
    # JWT TOKEN MANAGEMENT
    # ============================================================================

    def _create_access_token(self, user: User) -> str:
        """Create JWT access token"""

        jti = secrets.token_urlsafe(32)
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            'sub': str(user.id),
            'email': user.email,
            'plan': user.plan,
            'is_admin': user.is_admin,
            'jti': jti,
            'exp': expire,
            'iat': datetime.utcnow(),
            'type': 'access'
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        # Store JTI for revocation
        jwt_token = JWTToken(
            user_id=user.id,
            jti=jti,
            token_type='access',
            expires_at=expire
        )
        self.db.add(jwt_token)
        self.db.commit()

        return token

    def _create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token"""

        jti = secrets.token_urlsafe(32)
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        payload = {
            'sub': str(user.id),
            'jti': jti,
            'exp': expire,
            'iat': datetime.utcnow(),
            'type': 'refresh'
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        jwt_token = JWTToken(
            user_id=user.id,
            jti=jti,
            token_type='refresh',
            expires_at=expire
        )
        self.db.add(jwt_token)
        self.db.commit()

        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload"""

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Check if token is revoked
            jwt_token = self.db.query(JWTToken).filter(
                JWTToken.jti == payload['jti']
            ).first()

            if not jwt_token or jwt_token.revoked_at:
                return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    # ============================================================================
    # API KEY MANAGEMENT
    # ============================================================================

    def create_api_key(
        self,
        user: User,
        name: Optional[str] = None,
        scopes: list = ['read', 'write']
    ) -> tuple[str, APIKey]:
        """
        Create new API key for user.
        Returns (plain_key, api_key_model)
        """

        # Generate random API key
        plain_key = f"sk_{'live' if user.plan != 'free' else 'test'}_{secrets.token_urlsafe(32)}"

        # Hash key
        key_hash = bcrypt.hashpw(plain_key.encode(), bcrypt.gensalt()).decode()
        key_prefix = plain_key[:20] + "..."

        api_key = APIKey(
            user_id=user.id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name or f"API Key {datetime.utcnow().strftime('%Y-%m-%d')}",
            scopes=scopes,
            is_active=True
        )

        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)

        return (plain_key, api_key)

    def verify_api_key(self, plain_key: str) -> Optional[User]:
        """Verify API key and return associated user"""

        # Hash and find
        for api_key in self.db.query(APIKey).filter(APIKey.is_active == True).all():
            if bcrypt.checkpw(plain_key.encode(), api_key.key_hash.encode()):
                # Update last used
                api_key.last_used_at = datetime.utcnow()
                self.db.commit()

                return api_key.user

        return None
```

#### Day 5-7: Rate Limiting & Middleware

**File:** `sie_x/platform/middleware/rate_limit.py`

```python
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import redis.asyncio as aioredis
from datetime import datetime
import hashlib

class RateLimiter:
    """Redis-based rate limiting"""

    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)

    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        limit_per_minute: int = 60,
        limit_per_hour: int = 1000
    ) -> tuple[bool, dict]:
        """
        Check if user has exceeded rate limits.
        Returns (is_allowed, rate_limit_info)
        """

        now = datetime.utcnow()
        minute_key = f"ratelimit:{user_id}:{endpoint}:minute:{now.strftime('%Y%m%d%H%M')}"
        hour_key = f"ratelimit:{user_id}:{endpoint}:hour:{now.strftime('%Y%m%d%H')}"

        # Increment counters
        pipe = self.redis.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)
        results = await pipe.execute()

        minute_count = results[0]
        hour_count = results[2]

        # Check limits
        if minute_count > limit_per_minute:
            return False, {
                'limit': limit_per_minute,
                'remaining': 0,
                'reset_at': (now.replace(second=0, microsecond=0) + timedelta(minutes=1)).isoformat()
            }

        if hour_count > limit_per_hour:
            return False, {
                'limit': limit_per_hour,
                'remaining': 0,
                'reset_at': (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).isoformat()
            }

        return True, {
            'limit_minute': limit_per_minute,
            'remaining_minute': limit_per_minute - minute_count,
            'limit_hour': limit_per_hour,
            'remaining_hour': limit_per_hour - hour_count
        }
```

---

## Phase 2: API Pipeline Engine (Week 3-6)

### Week 3: Pipeline Database Schema & Models

**File:** `sie_x/platform/database/pipeline_schema.sql`

```sql
-- ============================================================================
-- PIPELINES
-- ============================================================================

CREATE TABLE pipelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,

    -- Pipeline Info
    name VARCHAR(500) NOT NULL,
    description TEXT,
    slug VARCHAR(500),  -- URL-friendly name

    -- Pipeline Definition (JSON)
    definition JSONB NOT NULL,  -- Full pipeline config
    /*
    Example definition:
    {
        "trigger": {
            "type": "schedule",
            "config": {"cron": "0 9 * * *"}
        },
        "nodes": [
            {
                "id": "node1",
                "type": "extract",
                "config": {"top_k": 10},
                "position": {"x": 100, "y": 100}
            },
            {
                "id": "node2",
                "type": "filter",
                "config": {"min_score": 0.8},
                "position": {"x": 300, "y": 100}
            }
        ],
        "edges": [
            {"from": "node1", "to": "node2"}
        ]
    }
    */

    -- Status
    status VARCHAR(50) DEFAULT 'draft',  -- draft, active, paused, archived
    is_template BOOLEAN DEFAULT false,

    -- Execution Stats
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    failed_executions INTEGER DEFAULT 0,
    last_execution_at TIMESTAMP,
    last_execution_status VARCHAR(50),

    -- Metadata
    tags TEXT[],
    category VARCHAR(100),  -- seo, content, automation
    version INTEGER DEFAULT 1,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP
);

CREATE INDEX idx_pipelines_user ON pipelines(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_pipelines_team ON pipelines(team_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_pipelines_status ON pipelines(status);
CREATE INDEX idx_pipelines_template ON pipelines(is_template) WHERE is_template = true;

-- ============================================================================
-- PIPELINE EXECUTIONS
-- ============================================================================

CREATE TABLE pipeline_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_id UUID REFERENCES pipelines(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Trigger
    triggered_by VARCHAR(50),  -- manual, schedule, webhook, api
    trigger_data JSONB,  -- Data that triggered execution

    -- Status
    status VARCHAR(50) DEFAULT 'pending',  -- pending, running, completed, failed, canceled
    error_message TEXT,

    -- Execution Details
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds FLOAT,

    -- Node Results
    node_results JSONB,  -- Results from each node
    /*
    {
        "node1": {
            "status": "completed",
            "output": {...},
            "duration_ms": 120
        },
        "node2": {
            "status": "failed",
            "error": "Rate limit exceeded",
            "duration_ms": 50
        }
    }
    */

    -- Resource Usage
    credits_used INTEGER DEFAULT 0,
    api_calls_made INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_executions_pipeline ON pipeline_executions(pipeline_id, created_at DESC);
CREATE INDEX idx_executions_user ON pipeline_executions(user_id, created_at DESC);
CREATE INDEX idx_executions_status ON pipeline_executions(status);

-- Partitioning by month
CREATE TABLE pipeline_executions_2025_01 PARTITION OF pipeline_executions
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- ============================================================================
-- PIPELINE TRIGGERS
-- ============================================================================

CREATE TABLE pipeline_triggers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_id UUID REFERENCES pipelines(id) ON DELETE CASCADE,

    -- Trigger Type
    trigger_type VARCHAR(50) NOT NULL,  -- schedule, webhook, rss, manual

    -- Trigger Config
    config JSONB NOT NULL,
    /*
    Schedule: {"cron": "0 9 * * *", "timezone": "UTC"}
    Webhook: {"secret": "...", "allowed_ips": ["1.2.3.4"]}
    RSS: {"url": "https://...", "check_interval_minutes": 60}
    */

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_triggered_at TIMESTAMP,
    next_trigger_at TIMESTAMP,  -- For scheduled triggers

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_triggers_pipeline ON pipeline_triggers(pipeline_id);
CREATE INDEX idx_triggers_next ON pipeline_triggers(next_trigger_at) WHERE is_active = true;

-- ============================================================================
-- INTEGRATIONS
-- ============================================================================

CREATE TABLE integrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Integration Type
    integration_type VARCHAR(50) NOT NULL,  -- slack, discord, wordpress, etc.
    name VARCHAR(255),

    -- Configuration
    config JSONB NOT NULL,  -- Credentials, endpoints, etc. (encrypted)

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP,

    -- Testing
    test_successful BOOLEAN,
    test_message TEXT,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_integrations_user ON integrations(user_id);
CREATE INDEX idx_integrations_type ON integrations(integration_type);

-- ============================================================================
-- WEBHOOKS
-- ============================================================================

CREATE TABLE webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    pipeline_id UUID REFERENCES pipelines(id) ON DELETE CASCADE,

    -- Webhook Details
    url VARCHAR(2000) NOT NULL,
    secret VARCHAR(255),  -- For signature verification

    -- Configuration
    method VARCHAR(10) DEFAULT 'POST',  -- POST, PUT, PATCH
    headers JSONB,  -- Custom headers

    -- Events
    events TEXT[],  -- Which events trigger this webhook

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_triggered_at TIMESTAMP,
    failed_attempts INTEGER DEFAULT 0,

    -- Retry Config
    max_retries INTEGER DEFAULT 3,
    retry_delay_seconds INTEGER DEFAULT 60,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_webhooks_pipeline ON webhooks(pipeline_id);
```

### Week 4: Pipeline Engine Core

**File:** `sie_x/platform/pipelines/engine.py`

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import uuid
from datetime import datetime

@dataclass
class PipelineNode:
    """A node in the pipeline"""
    id: str
    type: str  # extract, filter, transform, webhook, etc.
    config: Dict[str, Any]
    position: Dict[str, int]  # For visual builder

@dataclass
class PipelineEdge:
    """An edge connecting two nodes"""
    from_node: str
    to_node: str
    condition: Optional[str] = None  # Optional condition for edge

@dataclass
class PipelineTrigger:
    """Pipeline trigger configuration"""
    type: str  # schedule, webhook, rss, manual
    config: Dict[str, Any]

@dataclass
class PipelineDefinition:
    """Complete pipeline definition"""
    trigger: PipelineTrigger
    nodes: List[PipelineNode]
    edges: List[PipelineEdge]

class PipelineEngine:
    """
    Core pipeline execution engine.

    Executes pipelines by:
    1. Resolving execution graph (DAG)
    2. Executing nodes in topological order
    3. Passing data between nodes
    4. Handling errors and retries
    """

    def __init__(self, db: Session, sie_x_engine, integrations_manager):
        self.db = db
        self.sie_x = sie_x_engine
        self.integrations = integrations_manager

    async def execute(
        self,
        pipeline: Pipeline,
        trigger_data: Optional[Dict] = None
    ) -> PipelineExecution:
        """Execute a pipeline"""

        # Create execution record
        execution = PipelineExecution(
            id=uuid.uuid4(),
            pipeline_id=pipeline.id,
            user_id=pipeline.user_id,
            triggered_by='manual',
            trigger_data=trigger_data or {},
            status='pending',
            created_at=datetime.utcnow()
        )
        self.db.add(execution)
        self.db.commit()

        try:
            # Update status
            execution.status = 'running'
            execution.started_at = datetime.utcnow()
            self.db.commit()

            # Parse pipeline definition
            definition = PipelineDefinition(**pipeline.definition)

            # Build execution graph
            graph = self._build_execution_graph(definition)

            # Execute nodes in topological order
            node_results = {}
            for node_id in self._topological_sort(graph):
                node = self._get_node_by_id(definition.nodes, node_id)

                # Get input data from previous nodes
                input_data = self._get_node_input(node_id, graph, node_results, trigger_data)

                # Execute node
                result = await self._execute_node(node, input_data, pipeline.user_id)
                node_results[node_id] = result

                # Check if node failed
                if result['status'] == 'failed':
                    raise Exception(f"Node {node_id} failed: {result.get('error')}")

            # Success!
            execution.status = 'completed'
            execution.node_results = node_results
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()

            # Update pipeline stats
            pipeline.total_executions += 1
            pipeline.successful_executions += 1
            pipeline.last_execution_at = execution.completed_at
            pipeline.last_execution_status = 'completed'

        except Exception as e:
            # Failure
            execution.status = 'failed'
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()

            # Update pipeline stats
            pipeline.total_executions += 1
            pipeline.failed_executions += 1
            pipeline.last_execution_at = execution.completed_at
            pipeline.last_execution_status = 'failed'

        finally:
            self.db.commit()
            self.db.refresh(execution)

        return execution

    def _build_execution_graph(self, definition: PipelineDefinition) -> Dict:
        """Build directed acyclic graph (DAG) from pipeline definition"""

        graph = {node.id: [] for node in definition.nodes}

        for edge in definition.edges:
            graph[edge.from_node].append({
                'to': edge.to_node,
                'condition': edge.condition
            })

        return graph

    def _topological_sort(self, graph: Dict) -> List[str]:
        """
        Topological sort to determine execution order.
        Uses Kahn's algorithm.
        """

        # Calculate in-degree for each node
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for edge in graph[node]:
                in_degree[edge['to']] += 1

        # Queue of nodes with no dependencies
        queue = [node for node in in_degree if in_degree[node] == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)

            # Remove edges from this node
            for edge in graph[node]:
                in_degree[edge['to']] -= 1
                if in_degree[edge['to']] == 0:
                    queue.append(edge['to'])

        # Check for cycles
        if len(sorted_nodes) != len(graph):
            raise ValueError("Pipeline contains a cycle!")

        return sorted_nodes

    def _get_node_by_id(self, nodes: List[PipelineNode], node_id: str) -> PipelineNode:
        """Get node by ID"""
        for node in nodes:
            if node.id == node_id:
                return node
        raise ValueError(f"Node {node_id} not found")

    def _get_node_input(
        self,
        node_id: str,
        graph: Dict,
        node_results: Dict,
        trigger_data: Optional[Dict]
    ) -> Dict:
        """Get input data for a node from previous nodes"""

        # Find nodes that feed into this one
        input_data = {}

        for source_node_id, edges in graph.items():
            for edge in edges:
                if edge['to'] == node_id:
                    # This node feeds into current node
                    if source_node_id in node_results:
                        input_data[source_node_id] = node_results[source_node_id]['output']

        # If no input nodes, use trigger data
        if not input_data:
            input_data = trigger_data or {}

        return input_data

    async def _execute_node(
        self,
        node: PipelineNode,
        input_data: Dict,
        user_id: uuid.UUID
    ) -> Dict:
        """Execute a single node"""

        start_time = datetime.utcnow()

        try:
            # Route to appropriate handler based on node type
            if node.type == 'extract':
                output = await self._execute_extract_node(node, input_data)
            elif node.type == 'filter':
                output = await self._execute_filter_node(node, input_data)
            elif node.type == 'transform':
                output = await self._execute_transform_node(node, input_data)
            elif node.type == 'webhook':
                output = await self._execute_webhook_node(node, input_data)
            elif node.type == 'conditional':
                output = await self._execute_conditional_node(node, input_data)
            else:
                raise ValueError(f"Unknown node type: {node.type}")

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                'status': 'completed',
                'output': output,
                'duration_ms': duration_ms
            }

        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                'status': 'failed',
                'error': str(e),
                'duration_ms': duration_ms
            }

    async def _execute_extract_node(self, node: PipelineNode, input_data: Dict) -> Dict:
        """Execute keyword extraction node"""

        text = input_data.get('text', '')
        if not text:
            raise ValueError("No text provided for extraction")

        config = node.config
        keywords = self.sie_x.extract(
            text,
            top_k=config.get('top_k', 10),
            language=config.get('language', 'en')
        )

        return {
            'keywords': [
                {'text': kw.text, 'score': kw.score, 'type': kw.type}
                for kw in keywords
            ]
        }

    async def _execute_filter_node(self, node: PipelineNode, input_data: Dict) -> Dict:
        """Execute filter node (filter keywords by condition)"""

        keywords = input_data.get('keywords', [])
        if not keywords:
            return {'keywords': []}

        config = node.config
        min_score = config.get('min_score', 0.0)

        filtered = [kw for kw in keywords if kw.get('score', 0) >= min_score]

        return {'keywords': filtered}

    async def _execute_transform_node(self, node: PipelineNode, input_data: Dict) -> Dict:
        """Execute transform node (apply transformer like BACOWR)"""

        config = node.config
        transformer_type = config.get('transformer_type', 'bacowr')

        # TODO: Apply transformer
        # For now, pass through
        return input_data

    async def _execute_webhook_node(self, node: PipelineNode, input_data: Dict) -> Dict:
        """Execute webhook node (send data to external URL)"""

        import httpx

        config = node.config
        url = config.get('url')
        method = config.get('method', 'POST')
        headers = config.get('headers', {})

        async with httpx.AsyncClient() as client:
            if method == 'POST':
                response = await client.post(url, json=input_data, headers=headers, timeout=30)
            elif method == 'PUT':
                response = await client.put(url, json=input_data, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            return {
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type') == 'application/json' else response.text
            }

    async def _execute_conditional_node(self, node: PipelineNode, input_data: Dict) -> Dict:
        """Execute conditional node (if-then-else logic)"""

        config = node.config
        condition = config.get('condition')  # Python expression

        # Evaluate condition
        # SECURITY: Use safe eval (ast.literal_eval or restricted eval)
        # For now, simple comparison

        return input_data
```

**File:** `sie_x/platform/pipelines/dsl.py`

```python
"""
Pipeline DSL (Domain Specific Language)

Allows code-first pipeline creation like:

```python
from sie_x.platform.pipelines.dsl import Pipeline, Extract, Filter, Webhook

pipeline = Pipeline("RSS Monitor") \
    .trigger(RSS(url="https://techcrunch.com/feed")) \
    .add(Extract(top_k=10)) \
    .add(Filter(lambda kw: kw.score > 0.8)) \
    .add(Webhook("https://myapp.com/keywords")) \
    .build()
```
"""

from typing import Callable, List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Node:
    """Base node class"""
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")

class Extract(Node):
    """Keyword extraction node"""
    def __init__(self, top_k: int = 10, language: str = "en"):
        super().__init__(
            type="extract",
            config={"top_k": top_k, "language": language}
        )

class Filter(Node):
    """Filter node"""
    def __init__(self, condition: Callable):
        super().__init__(
            type="filter",
            config={"condition": condition.__name__, "lambda": str(condition)}
        )

class Transform(Node):
    """Transform node"""
    def __init__(self, transformer_type: str, **kwargs):
        super().__init__(
            type="transform",
            config={"transformer_type": transformer_type, **kwargs}
        )

class Webhook(Node):
    """Webhook node"""
    def __init__(self, url: str, method: str = "POST", headers: Dict = None):
        super().__init__(
            type="webhook",
            config={"url": url, "method": method, "headers": headers or {}}
        )

class Pipeline:
    """Pipeline builder"""

    def __init__(self, name: str):
        self.name = name
        self.nodes: List[Node] = []
        self.trigger_config = None

    def trigger(self, trigger_type: str, **config):
        """Set pipeline trigger"""
        self.trigger_config = {"type": trigger_type, "config": config}
        return self

    def add(self, node: Node):
        """Add node to pipeline"""
        self.nodes.append(node)
        return self

    def build(self) -> Dict:
        """Build pipeline definition"""

        # Create edges (sequential by default)
        edges = []
        for i in range(len(self.nodes) - 1):
            edges.append({
                "from": self.nodes[i].id,
                "to": self.nodes[i + 1].id
            })

        return {
            "name": self.name,
            "trigger": self.trigger_config,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "config": node.config,
                    "position": {"x": i * 200, "y": 100}
                }
                for i, node in enumerate(self.nodes)
            ],
            "edges": edges
        }
```

### Week 5-6: Visual Pipeline Builder (React Frontend)

**File:** `frontend/src/components/PipelineBuilder/index.tsx`

```typescript
import React, { useState, useCallback } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Custom node types
import ExtractNode from './nodes/ExtractNode';
import FilterNode from './nodes/FilterNode';
import TransformNode from './nodes/TransformNode';
import WebhookNode from './nodes/WebhookNode';

const nodeTypes = {
  extract: ExtractNode,
  filter: FilterNode,
  transform: TransformNode,
  webhook: WebhookNode,
};

export default function PipelineBuilder() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [pipelineName, setPipelineName] = useState('');

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const addNode = (type: string) => {
    const newNode: Node = {
      id: `node_${Date.now()}`,
      type,
      position: { x: Math.random() * 400, y: Math.random() * 400 },
      data: { label: `${type} node`, config: {} },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const savePipeline = async () => {
    const pipelineDefinition = {
      name: pipelineName,
      trigger: { type: 'manual', config: {} },
      nodes: nodes.map(node => ({
        id: node.id,
        type: node.type!,
        config: node.data.config || {},
        position: node.position,
      })),
      edges: edges.map(edge => ({
        from_node: edge.source,
        to_node: edge.target,
      })),
    };

    const response = await fetch('/api/pipelines', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pipelineDefinition),
    });

    if (response.ok) {
      alert('Pipeline saved!');
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <div style={{ padding: '10px', background: '#fff', borderBottom: '1px solid #ddd' }}>
        <input
          type="text"
          placeholder="Pipeline name"
          value={pipelineName}
          onChange={(e) => setPipelineName(e.target.value)}
          style={{ marginRight: '10px', padding: '5px' }}
        />
        <button onClick={() => addNode('extract')}>Add Extract</button>
        <button onClick={() => addNode('filter')}>Add Filter</button>
        <button onClick={() => addNode('transform')}>Add Transform</button>
        <button onClick={() => addNode('webhook')}>Add Webhook</button>
        <button onClick={savePipeline} style={{ marginLeft: '20px', background: '#0066cc', color: '#fff' }}>
          Save Pipeline
        </button>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
      >
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  );
}
```

### Week 7: Pre-built Pipeline Templates

**File:** `sie_x/platform/pipelines/templates.py`

```python
"""
20 Pre-built Pipeline Templates

Users can start from these templates and customize.
"""

TEMPLATES = [
    {
        "id": "rss-keyword-monitoring",
        "name": "RSS Feed Keyword Monitor",
        "description": "Monitor RSS feeds for relevant keywords, send to Slack",
        "category": "monitoring",
        "definition": {
            "trigger": {
                "type": "rss",
                "config": {
                    "url": "https://techcrunch.com/feed",
                    "check_interval_minutes": 60
                }
            },
            "nodes": [
                {
                    "id": "extract",
                    "type": "extract",
                    "config": {"top_k": 10, "language": "en"},
                    "position": {"x": 100, "y": 100}
                },
                {
                    "id": "filter",
                    "type": "filter",
                    "config": {"min_score": 0.7},
                    "position": {"x": 300, "y": 100}
                },
                {
                    "id": "slack",
                    "type": "webhook",
                    "config": {
                        "url": "https://hooks.slack.com/...",
                        "method": "POST"
                    },
                    "position": {"x": 500, "y": 100}
                }
            ],
            "edges": [
                {"from_node": "extract", "to_node": "filter"},
                {"from_node": "filter", "to_node": "slack"}
            ]
        }
    },

    {
        "id": "content-seo-optimizer",
        "name": "Automated Content SEO Optimizer",
        "description": "Analyze new blog posts, extract keywords, generate SEO recommendations",
        "category": "seo",
        "definition": {
            "trigger": {
                "type": "webhook",
                "config": {"secret": "..."}
            },
            "nodes": [
                {
                    "id": "extract",
                    "type": "extract",
                    "config": {"top_k": 20},
                    "position": {"x": 100, "y": 100}
                },
                {
                    "id": "bacowr",
                    "type": "transform",
                    "config": {"transformer_type": "bacowr"},
                    "position": {"x": 300, "y": 100}
                },
                {
                    "id": "email",
                    "type": "webhook",
                    "config": {
                        "url": "https://api.sendgrid.com/v3/mail/send",
                        "method": "POST"
                    },
                    "position": {"x": 500, "y": 100}
                }
            ],
            "edges": [
                {"from_node": "extract", "to_node": "bacowr"},
                {"from_node": "bacowr", "to_node": "email"}
            ]
        }
    },

    # ... 18 more templates
]
```

### Week 8: Testing, Deployment & Documentation

**File:** `tests/test_pipeline_engine.py`

```python
import pytest
from sie_x.platform.pipelines.engine import PipelineEngine
from sie_x.platform.database.models import Pipeline, User

@pytest.mark.asyncio
async def test_simple_pipeline_execution(db, sie_x_engine):
    """Test execution of simple Extract → Filter pipeline"""

    # Create user
    user = User(email="test@example.com", plan="professional")
    db.add(user)
    db.commit()

    # Create pipeline
    pipeline = Pipeline(
        user_id=user.id,
        name="Test Pipeline",
        definition={
            "trigger": {"type": "manual", "config": {}},
            "nodes": [
                {
                    "id": "extract",
                    "type": "extract",
                    "config": {"top_k": 5},
                    "position": {"x": 0, "y": 0}
                },
                {
                    "id": "filter",
                    "type": "filter",
                    "config": {"min_score": 0.8},
                    "position": {"x": 200, "y": 0}
                }
            ],
            "edges": [
                {"from_node": "extract", "to_node": "filter"}
            ]
        }
    )
    db.add(pipeline)
    db.commit()

    # Execute pipeline
    engine = PipelineEngine(db, sie_x_engine, None)
    execution = await engine.execute(
        pipeline,
        trigger_data={"text": "Machine learning and artificial intelligence are transforming SEO"}
    )

    # Verify execution
    assert execution.status == "completed"
    assert "extract" in execution.node_results
    assert "filter" in execution.node_results
    assert len(execution.node_results["filter"]["output"]["keywords"]) <= 5
```

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: siex_platform
      POSTGRES_USER: siex
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sie_x/platform/database/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
      - ./sie_x/platform/database/pipeline_schema.sql:/docker-entrypoint-initdb.d/02-pipelines.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U siex"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis (Cache + Queue)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # RabbitMQ (Message Broker)
  rabbitmq:
    image: rabbitmq:3-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: siex
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
    ports:
      - "5672:5672"
      - "15672:15672"  # Management UI
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  # FastAPI Gateway
  api-gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway
    environment:
      DATABASE_URL: postgresql://siex:${DB_PASSWORD}@postgres:5432/siex_platform
      REDIS_URL: redis://redis:6379
      JWT_SECRET: ${JWT_SECRET}
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./sie_x:/app/sie_x

  # Celery Workers
  celery-worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      DATABASE_URL: postgresql://siex:${DB_PASSWORD}@postgres:5432/siex_platform
      REDIS_URL: redis://redis:6379
      CELERY_BROKER_URL: amqp://siex:${RABBITMQ_PASSWORD}@rabbitmq:5672
    depends_on:
      - postgres
      - redis
      - rabbitmq
    volumes:
      - ./sie_x:/app/sie_x
    command: celery -A sie_x.platform.tasks worker --loglevel=info

  # React Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      REACT_APP_API_URL: http://localhost:8000
    volumes:
      - ./frontend/src:/app/src

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
```

---

## Implementation Timeline

### Summary Table

| Week | Phase | Deliverables | Status |
|------|-------|--------------|--------|
| 1 | Database & Auth | PostgreSQL schema, User models, Auth service | ⏳ Pending |
| 2 | Middleware & Admin | Rate limiting, Admin API, Billing integration | ⏳ Pending |
| 3 | Pipeline Schema | Pipeline DB models, Execution tracking | ⏳ Pending |
| 4 | Pipeline Engine | Core execution engine, DSL, Node types | ⏳ Pending |
| 5 | Visual Builder | React Flow UI, Node editor | ⏳ Pending |
| 6 | Integrations | Slack, Discord, WordPress, Webhooks | ⏳ Pending |
| 7 | Templates & Testing | 20 templates, Unit/Integration tests | ⏳ Pending |
| 8 | Deployment | Docker, K8s, CI/CD, Documentation | ⏳ Pending |

### Detailed Schedule

**Week 1: Foundation**
- Day 1-2: PostgreSQL setup, schema creation
- Day 3-4: SQLAlchemy models, migrations
- Day 5-7: Auth service (JWT, API keys, OAuth)

**Week 2: Infrastructure**
- Day 1-2: Rate limiting middleware
- Day 3-4: Admin API (user management)
- Day 5: Stripe billing integration

**Week 3: Pipelines (Database)**
- Day 1-2: Pipeline schema
- Day 3-4: Execution tracking
- Day 5: Trigger management

**Week 4: Pipelines (Engine)**
- Day 1-3: Core execution engine
- Day 4-5: Node types (Extract, Filter, Transform, Webhook)
- Day 6-7: DSL for code-first pipelines

**Week 5: Frontend (Builder)**
- Day 1-3: React Flow setup, Canvas
- Day 4-5: Custom node components
- Day 6-7: Save/Load pipelines

**Week 6: Integrations**
- Day 1: Slack, Discord
- Day 2: WordPress, HubSpot
- Day 3: SendGrid, Mailgun
- Day 4: Airtable, Google Sheets
- Day 5: Webhook framework

**Week 7: Polish**
- Day 1-3: 20 pre-built templates
- Day 4-5: Unit/Integration tests
- Day 6-7: Load testing

**Week 8: Launch**
- Day 1-2: Docker/K8s deployment
- Day 3-4: CI/CD pipeline
- Day 5: Documentation
- Day 6-7: Beta testing

---

## Success Metrics

### Technical KPIs

**Performance:**
- API response time: < 200ms (p95)
- Pipeline execution: < 5s for simple pipelines
- Database queries: < 50ms (p95)
- Cache hit rate: > 80%

**Reliability:**
- Uptime: 99.9%
- Error rate: < 0.1%
- Pipeline success rate: > 95%

**Scalability:**
- Support 10,000 concurrent users
- Handle 1M pipeline executions/day
- Store 100M execution records

### Business KPIs

**Adoption:**
- Month 1: 100 beta users
- Month 3: 1,000 active users
- Month 6: 5,000 active users
- Year 1: 20,000 active users

**Revenue:**
- Month 3: $10K MRR
- Month 6: $50K MRR
- Year 1: $149K MRR (pipelines alone)
- Year 2: $990K MRR (with all use cases)

**Engagement:**
- Daily Active Users: 40%
- Pipelines per user: 3 avg
- Executions per pipeline: 50/month avg

---

## Risk Mitigation

### Technical Risks

**Risk: Database Performance Degradation**
- Mitigation: Partitioning (executions by month), Indexes, Connection pooling
- Monitoring: Query time alerts, Slow query log

**Risk: Celery Worker Overload**
- Mitigation: Auto-scaling (Kubernetes HPA), Queue prioritization
- Monitoring: Queue length, Worker CPU/memory

**Risk: Pipeline Security (Code Injection)**
- Mitigation: Sandboxed execution, Input validation, Rate limiting
- Monitoring: Anomaly detection, Audit logs

### Business Risks

**Risk: Low Adoption**
- Mitigation: 20 pre-built templates, Video tutorials, Free tier
- Validation: Beta testing with 100 users before launch

**Risk: High Infrastructure Costs**
- Mitigation: Aggressive caching, Auto-scaling, Usage-based pricing
- Monitoring: Cost per user, Infrastructure utilization

---

## Next Steps

1. ✅ **Complete this plan** - Document all code and architecture
2. ⏳ **Setup development environment** - PostgreSQL, Redis, RabbitMQ locally
3. ⏳ **Implement Week 1** - Database schema and auth system
4. ⏳ **Implement Week 2-4** - Pipeline engine core
5. ⏳ **Implement Week 5-6** - Visual builder and integrations
6. ⏳ **Test and deploy** - Week 7-8

---

**This is the complete implementation plan. Ready to start building! 🚀**