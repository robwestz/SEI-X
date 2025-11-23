"""
SQLAlchemy ORM Models for SIE-X Platform

All database models using declarative base.
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, ForeignKey, Text, ARRAY, Float, BigInteger
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


# ============================================================================
# USERS & AUTHENTICATION
# ============================================================================

class User(Base):
    """User model with authentication and billing"""
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
    monthly_quota = Column(JSONB, default={"extractions": 1000, "pipelines": 10, "executions": 100})
    usage_this_month = Column(JSONB, default={"extractions": 0, "pipelines": 0, "executions": 0})

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
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    pipelines = relationship("Pipeline", back_populates="user", cascade="all, delete-orphan")
    jwt_tokens = relationship("JWTToken", back_populates="user", cascade="all, delete-orphan")
    usage_logs = relationship("UsageLog", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert user to dict for API responses"""
        return {
            'id': str(self.id),
            'email': self.email,
            'full_name': self.full_name,
            'company': self.company,
            'plan': self.plan,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Subscription(Base):
    """Stripe subscription model"""
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
    """API key model for authentication"""
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


class JWTToken(Base):
    """JWT token tracking for revocation"""
    __tablename__ = 'jwt_tokens'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))

    jti = Column(String(255), unique=True, nullable=False, index=True)
    token_type = Column(String(50), default='access')

    expires_at = Column(DateTime, nullable=False)

    revoked_at = Column(DateTime)
    revoked_reason = Column(String(255))

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="jwt_tokens")


class UsageLog(Base):
    """Usage tracking for billing and quotas"""
    __tablename__ = 'usage_logs'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))

    resource_type = Column(String(50), nullable=False)
    resource_id = Column(UUID(as_uuid=True))

    credits_used = Column(Integer, default=1)
    metadata = Column(JSONB)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="usage_logs")


class AuditLog(Base):
    """Audit log for compliance"""
    __tablename__ = 'audit_logs'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id', ondelete='SET NULL'))

    event_type = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))

    ip_address = Column(INET)
    user_agent = Column(Text)
    metadata = Column(JSONB)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# ============================================================================
# TEAMS & COLLABORATION
# ============================================================================

class Team(Base):
    """Team model for agencies"""
    __tablename__ = 'teams'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))

    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, nullable=False)

    plan = Column(String(50), default='agency')
    max_members = Column(Integer, default=5)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    members = relationship("TeamMember", back_populates="team", cascade="all, delete-orphan")
    pipelines = relationship("Pipeline", back_populates="team")


class TeamMember(Base):
    """Team member relationship"""
    __tablename__ = 'team_members'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id', ondelete='CASCADE'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))

    role = Column(String(50), default='member')

    can_create_pipelines = Column(Boolean, default=True)
    can_edit_pipelines = Column(Boolean, default=True)
    can_delete_pipelines = Column(Boolean, default=False)
    can_manage_members = Column(Boolean, default=False)

    joined_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    team = relationship("Team", back_populates="members")


# ============================================================================
# PIPELINES
# ============================================================================

class Pipeline(Base):
    """Pipeline model"""
    __tablename__ = 'pipelines'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id', ondelete='CASCADE'))

    name = Column(String(500), nullable=False)
    description = Column(Text)
    slug = Column(String(500))

    definition = Column(JSONB, nullable=False)

    status = Column(String(50), default='draft')
    is_template = Column(Boolean, default=False)

    total_executions = Column(Integer, default=0)
    successful_executions = Column(Integer, default=0)
    failed_executions = Column(Integer, default=0)
    last_execution_at = Column(DateTime)
    last_execution_status = Column(String(50))

    tags = Column(ARRAY(Text))
    category = Column(String(100))
    version = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="pipelines")
    team = relationship("Team", back_populates="pipelines")
    executions = relationship("PipelineExecution", back_populates="pipeline", cascade="all, delete-orphan")
    triggers = relationship("PipelineTrigger", back_populates="pipeline", cascade="all, delete-orphan")


class PipelineExecution(Base):
    """Pipeline execution record"""
    __tablename__ = 'pipeline_executions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_id = Column(UUID(as_uuid=True), ForeignKey('pipelines.id', ondelete='CASCADE'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))

    triggered_by = Column(String(50))
    trigger_data = Column(JSONB)

    status = Column(String(50), default='pending')
    error_message = Column(Text)

    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)

    node_results = Column(JSONB)

    credits_used = Column(Integer, default=0)
    api_calls_made = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pipeline = relationship("Pipeline", back_populates="executions")


class PipelineTrigger(Base):
    """Pipeline trigger configuration"""
    __tablename__ = 'pipeline_triggers'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_id = Column(UUID(as_uuid=True), ForeignKey('pipelines.id', ondelete='CASCADE'))

    trigger_type = Column(String(50), nullable=False)
    config = Column(JSONB, nullable=False)

    is_active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime)
    next_trigger_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    pipeline = relationship("Pipeline", back_populates="triggers")


class Integration(Base):
    """External service integration"""
    __tablename__ = 'integrations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))

    integration_type = Column(String(50), nullable=False)
    name = Column(String(255))

    config = Column(JSONB, nullable=False)

    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)

    test_successful = Column(Boolean)
    test_message = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Webhook(Base):
    """Webhook configuration"""
    __tablename__ = 'webhooks'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    pipeline_id = Column(UUID(as_uuid=True), ForeignKey('pipelines.id', ondelete='CASCADE'))

    url = Column(String(2000), nullable=False)
    secret = Column(String(255))

    method = Column(String(10), default='POST')
    headers = Column(JSONB)

    events = Column(ARRAY(Text))

    is_active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime)
    failed_attempts = Column(Integer, default=0)

    max_retries = Column(Integer, default=3)
    retry_delay_seconds = Column(Integer, default=60)

    created_at = Column(DateTime, default=datetime.utcnow)
