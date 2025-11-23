-- ============================================================================
-- SIE-X Platform Database Schema
-- PostgreSQL 15+
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

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
-- CREATE TABLE usage_logs_2025_01 PARTITION OF usage_logs
--     FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

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
