-- ============================================================================
-- SIE-X Platform - Pipeline Database Schema
-- PostgreSQL 15+
-- ============================================================================

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

    -- Resource Usage
    credits_used INTEGER DEFAULT 0,
    api_calls_made INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_executions_pipeline ON pipeline_executions(pipeline_id, created_at DESC);
CREATE INDEX idx_executions_user ON pipeline_executions(user_id, created_at DESC);
CREATE INDEX idx_executions_status ON pipeline_executions(status);

-- Partitioning by month
-- CREATE TABLE pipeline_executions_2025_01 PARTITION OF pipeline_executions
--     FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

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
