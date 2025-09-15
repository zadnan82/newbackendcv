# app/models.py
"""
Minimal database models for privacy-first CV platform.
NO personal data stored - only anonymous sessions and public job data.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    JSON,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class AnonymousSession(Base):
    """Anonymous user sessions with encrypted cloud provider tokens"""

    __tablename__ = "anonymous_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)

    # Encrypted cloud provider tokens
    cloud_tokens = Column(Text, nullable=True)  # JSON encrypted with session key

    # Session metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

    # Analytics (anonymous)
    ip_hash = Column(String(64), nullable=True)  # Hashed IP for rate limiting
    user_agent_hash = Column(String(64), nullable=True)  # Hashed user agent

    __table_args__ = (
        Index("idx_session_id", "session_id"),
        Index("idx_session_expires", "expires_at"),
        Index("idx_session_activity", "last_activity"),
    )


class JobPosting(Base):
    """Public job postings data for CV matching"""

    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, index=True)

    # Job information (public data)
    title = Column(String(200), nullable=False, index=True)
    company = Column(String(100), nullable=False, index=True)
    location = Column(String(100), nullable=True)
    country = Column(String(50), nullable=True, index=True)

    # Job details
    description = Column(Text, nullable=True)
    requirements = Column(Text, nullable=True)

    # Salary information
    salary_min = Column(Integer, nullable=True)
    salary_max = Column(Integer, nullable=True)
    salary_currency = Column(String(3), default="USD")

    # Job metadata
    job_type = Column(String(50), nullable=True)  # full-time, part-time, contract
    experience_level = Column(String(50), nullable=True)  # entry, mid, senior
    industry = Column(String(100), nullable=True, index=True)

    # Source information
    source = Column(String(50), nullable=False)  # API source
    external_id = Column(String(100), nullable=True)
    url = Column(String(500), nullable=True)

    # System metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, index=True)

    __table_args__ = (
        Index("idx_job_location_active", "country", "location", "is_active"),
        Index("idx_job_industry_active", "industry", "is_active"),
        Index("idx_job_source_external", "source", "external_id"),
        Index("idx_job_created", "created_at"),
    )


class TemporaryShare(Base):
    """Temporary encrypted CV shares for QR codes/public links"""

    __tablename__ = "temporary_shares"

    id = Column(Integer, primary_key=True, index=True)
    share_id = Column(String(255), unique=True, index=True, nullable=False)

    # Encrypted CV data (temporary only)
    encrypted_data = Column(Text, nullable=False)

    # Share settings
    max_views = Column(Integer, default=50)
    view_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

    # Optional password protection
    password_hash = Column(String(255), nullable=True)

    __table_args__ = (
        Index("idx_share_id", "share_id"),
        Index("idx_share_expires", "expires_at"),
    )


class UsageAnalytics(Base):
    """Anonymous usage analytics for service improvement"""

    __tablename__ = "usage_analytics"

    id = Column(Integer, primary_key=True, index=True)

    # Anonymous identifiers
    session_hash = Column(String(64), nullable=True)  # Hashed session ID

    # Feature usage
    feature_used = Column(String(100), nullable=False)  # ai-enhance, cover-letter, etc.
    processing_time_ms = Column(Integer, nullable=True)
    success = Column(Boolean, nullable=False)

    # Anonymous metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    cloud_provider = Column(String(50), nullable=True)  # google, onedrive, etc.

    # Error tracking (no personal data)
    error_type = Column(String(100), nullable=True)

    __table_args__ = (
        Index("idx_analytics_feature", "feature_used"),
        Index("idx_analytics_timestamp", "timestamp"),
        Index("idx_analytics_provider", "cloud_provider"),
    )


class AIUsageTracking(Base):
    """Track AI usage for rate limiting (anonymous)"""

    __tablename__ = "ai_usage_tracking"

    id = Column(Integer, primary_key=True, index=True)

    # Anonymous session tracking
    session_hash = Column(String(64), nullable=False, index=True)

    # Usage details
    service_type = Column(String(50), nullable=False)  # cv-enhance, cover-letter, etc.
    tokens_used = Column(Integer, nullable=True)
    cost_estimate = Column(Float, nullable=True)

    # Timestamps for rate limiting
    used_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_ai_usage_session_date", "session_hash", "used_at"),
        Index("idx_ai_usage_service", "service_type"),
    )


class PricingTier(Base):
    """Pricing tiers and feature limits"""

    __tablename__ = "pricing_tiers"

    id = Column(Integer, primary_key=True, index=True)

    # Tier information
    tier_name = Column(String(50), unique=True, nullable=False)  # free, pro, business
    price_monthly = Column(Float, nullable=False)

    # Feature limits
    ai_operations_daily = Column(Integer, nullable=False)
    cloud_providers_limit = Column(Integer, nullable=True)  # NULL = unlimited
    storage_limit_mb = Column(Integer, nullable=True)  # NULL = unlimited

    # Features (JSON)
    features = Column(
        JSON, nullable=True
    )  # {"advanced_templates": true, "priority_support": true}

    # System
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_tier_name", "tier_name"),
        Index("idx_tier_active", "is_active"),
    )


class OAuthState(Base):
    """OAuth state storage for database fallback when Redis is not available"""

    __tablename__ = "oauth_states"

    state = Column(String(255), primary_key=True, index=True)
    provider = Column(String(50), nullable=False)
    session_id = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    data = Column(Text, nullable=True)  # JSON data for additional OAuth parameters

    __table_args__ = (
        Index("idx_oauth_state_expires", "expires_at"),
        Index("idx_oauth_state_provider", "provider"),
        Index("idx_oauth_state_session", "session_id"),
    )
