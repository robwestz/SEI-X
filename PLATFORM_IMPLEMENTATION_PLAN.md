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

**To be continued in next response... (4,600+ lines total)**

Vill du att jag fortsätter med resten av planen? Den omfattar:
- Week 1-2: Auth middleware, rate limiting, admin API
- Week 3-4: Pipeline Engine core, DSL, execution
- Week 5-6: Visual Builder, integrations, templates
- Week 7-8: Testing, deployment, documentation

Låter det bra att jag kör på?