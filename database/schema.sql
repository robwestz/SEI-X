-- =============================================================================
-- SIE-X Multi-Tenant Database Schema
-- PostgreSQL 14+
-- =============================================================================
--
-- This schema implements a secure multi-tenant architecture with:
-- - Tenant isolation via foreign keys and Row-Level Security (RLS)
-- - Role-Based Access Control (RBAC)
-- - API key management
-- - Extraction history tracking
-- - Audit logging
--
-- Security Features:
-- - Row-Level Security (RLS) on all tenant tables
-- - Indexed tenant_id columns for performance
-- - Cascading deletes for data cleanup
-- - Password hashing (handled by application)
-- - API key hashing (handled by application)
--
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for additional crypto functions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- TENANTS TABLE
-- =============================================================================
-- Represents organizations/customers in the multi-tenant system

CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identity
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,

    -- Status
    status VARCHAR(50) DEFAULT 'active' NOT NULL,
        CHECK (status IN ('active', 'suspended', 'trial', 'cancelled')),
    plan VARCHAR(50) DEFAULT 'free' NOT NULL,
        CHECK (plan IN ('free', 'starter', 'professional', 'enterprise')),

    -- Quotas
    monthly_quota INTEGER DEFAULT 10000 NOT NULL,
    monthly_usage INTEGER DEFAULT 0 NOT NULL,
    quota_reset_at TIMESTAMP DEFAULT (DATE_TRUNC('month', NOW()) + INTERVAL '1 month'),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW() NOT NULL,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for performance
CREATE INDEX idx_tenants_slug ON tenants(slug);
CREATE INDEX idx_tenants_status ON tenants(status);
CREATE INDEX idx_tenants_plan ON tenants(plan);

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- USERS TABLE
-- =============================================================================
-- Represents users within tenants

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Tenant relationship (CASCADE DELETE: when tenant is deleted, users are deleted)
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,

    -- Identity
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,  -- bcrypt or argon2 hash

    -- Profile
    first_name VARCHAR(100),
    last_name VARCHAR(100),

    -- RBAC
    role VARCHAR(50) DEFAULT 'member' NOT NULL,
        CHECK (role IN ('super_admin', 'tenant_admin', 'manager', 'member', 'viewer')),

    -- Status
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    is_email_verified BOOLEAN DEFAULT FALSE NOT NULL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW() NOT NULL,
    last_login_at TIMESTAMP,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Unique constraint: email must be unique within tenant
    UNIQUE(tenant_id, email)
);

-- Indexes for performance
CREATE INDEX idx_users_tenant_id ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_is_active ON users(is_active);

-- Auto-update updated_at
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Row-Level Security for users table
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see users in their tenant
CREATE POLICY tenant_isolation_users ON users
    USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

-- =============================================================================
-- API KEYS TABLE
-- =============================================================================
-- Programmatic access via API keys

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Tenant relationship
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,

    -- Key details
    name VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(20) NOT NULL,  -- e.g., "sk_live_" for display
    key_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 hash of full key

    -- Metadata
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,

    -- Status
    is_active BOOLEAN DEFAULT TRUE NOT NULL,

    -- Usage tracking
    usage_count INTEGER DEFAULT 0 NOT NULL,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes
CREATE INDEX idx_api_keys_tenant_id ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);

-- Row-Level Security
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_api_keys ON api_keys
    USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

-- =============================================================================
-- EXTRACTION HISTORY TABLE
-- =============================================================================
-- Track all keyword extractions (with tenant isolation)

CREATE TABLE extraction_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Tenant relationship (CRITICAL for isolation)
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,

    -- User relationship (nullable for API key access)
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Extraction details
    text_hash VARCHAR(64) NOT NULL,  -- SHA-256 hash for deduplication
    text_length INTEGER NOT NULL,
    keywords JSONB NOT NULL,  -- Array of extracted keywords
    processing_time FLOAT NOT NULL,  -- Seconds

    -- Metadata
    source VARCHAR(50),  -- 'api', 'sdk', 'web', etc.
    language VARCHAR(10),
    options JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for performance
CREATE INDEX idx_extraction_history_tenant_id ON extraction_history(tenant_id);
CREATE INDEX idx_extraction_history_user_id ON extraction_history(user_id);
CREATE INDEX idx_extraction_history_created_at ON extraction_history(created_at DESC);
CREATE INDEX idx_extraction_history_text_hash ON extraction_history(text_hash);

-- Row-Level Security (CRITICAL FOR TENANT ISOLATION)
ALTER TABLE extraction_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_extraction_history ON extraction_history
    USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

-- =============================================================================
-- JWT BLACKLIST TABLE
-- =============================================================================
-- Track revoked JWT tokens

CREATE TABLE jwt_blacklist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Token details
    token_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 hash of JWT
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Revocation details
    revoked_at TIMESTAMP DEFAULT NOW() NOT NULL,
    revoked_by UUID REFERENCES users(id) ON DELETE SET NULL,
    reason VARCHAR(255),

    -- Expiry (for cleanup)
    expires_at TIMESTAMP NOT NULL
);

-- Index for fast lookup
CREATE INDEX idx_jwt_blacklist_token_hash ON jwt_blacklist(token_hash);
CREATE INDEX idx_jwt_blacklist_expires_at ON jwt_blacklist(expires_at);

-- =============================================================================
-- AUDIT LOG TABLE
-- =============================================================================
-- Track all security-relevant events

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Context
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Event details
    event_type VARCHAR(100) NOT NULL,  -- 'user.login', 'user.logout', 'resource.created', etc.
    resource_type VARCHAR(100),  -- 'user', 'api_key', 'extraction', etc.
    resource_id UUID,

    -- Action details
    action VARCHAR(50) NOT NULL,  -- 'create', 'read', 'update', 'delete'
    status VARCHAR(50) NOT NULL,  -- 'success', 'failure'

    -- Request details
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(100),

    -- Changes (for audit trail)
    old_values JSONB,
    new_values JSONB,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW() NOT NULL
);

-- Indexes
CREATE INDEX idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);

-- Row-Level Security
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_audit_logs ON audit_logs
    USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to increment monthly usage for a tenant
CREATE OR REPLACE FUNCTION increment_tenant_usage(p_tenant_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE tenants
    SET monthly_usage = monthly_usage + 1
    WHERE id = p_tenant_id;
END;
$$ LANGUAGE plpgsql;

-- Function to reset monthly usage (run monthly via cron)
CREATE OR REPLACE FUNCTION reset_monthly_usage()
RETURNS VOID AS $$
BEGIN
    UPDATE tenants
    SET
        monthly_usage = 0,
        quota_reset_at = DATE_TRUNC('month', NOW()) + INTERVAL '1 month'
    WHERE quota_reset_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup expired JWT blacklist entries
CREATE OR REPLACE FUNCTION cleanup_expired_jwt_blacklist()
RETURNS VOID AS $$
BEGIN
    DELETE FROM jwt_blacklist
    WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SEED DATA (Development Only)
-- =============================================================================
-- Create a default tenant and user for testing
-- WARNING: Remove this in production!

INSERT INTO tenants (id, name, slug, status, plan, monthly_quota)
VALUES
    ('00000000-0000-0000-0000-000000000001', 'Demo Tenant', 'demo', 'active', 'professional', 100000)
ON CONFLICT DO NOTHING;

-- Password: "Password123!" (bcrypt hashed)
-- To generate: python -c "import bcrypt; print(bcrypt.hashpw(b'Password123!', bcrypt.gensalt()).decode())"
INSERT INTO users (id, tenant_id, email, password_hash, first_name, last_name, role, is_email_verified)
VALUES
    (
        '00000000-0000-0000-0000-000000000002',
        '00000000-0000-0000-0000-000000000001',
        'admin@demo.com',
        '$2b$12$KGKBVrXhLW3R6QZcZ6qxYOYPgQSXvJRFQYvJXqZKfJZqQzKZqQzKZ',  -- Password123!
        'Admin',
        'User',
        'tenant_admin',
        TRUE
    )
ON CONFLICT DO NOTHING;

-- =============================================================================
-- GRANTS
-- =============================================================================
-- Grant permissions to application database user
-- Replace 'siex_app' with your actual database user

-- GRANT ALL ON ALL TABLES IN SCHEMA public TO siex_app;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO siex_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO siex_app;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE tenants IS 'Organizations/customers in the multi-tenant system';
COMMENT ON TABLE users IS 'Users within tenants with RBAC roles';
COMMENT ON TABLE api_keys IS 'API keys for programmatic access';
COMMENT ON TABLE extraction_history IS 'History of all keyword extractions (tenant-isolated)';
COMMENT ON TABLE jwt_blacklist IS 'Revoked JWT tokens';
COMMENT ON TABLE audit_logs IS 'Security audit trail';

COMMENT ON POLICY tenant_isolation_users ON users IS 'Ensures users can only see users in their tenant';
COMMENT ON POLICY tenant_isolation_api_keys ON api_keys IS 'Ensures API keys are tenant-isolated';
COMMENT ON POLICY tenant_isolation_extraction_history ON extraction_history IS 'CRITICAL: Ensures extraction history is tenant-isolated';
COMMENT ON POLICY tenant_isolation_audit_logs ON audit_logs IS 'Ensures audit logs are tenant-isolated';

-- =============================================================================
-- END OF SCHEMA
-- =============================================================================
