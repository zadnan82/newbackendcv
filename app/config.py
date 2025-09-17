# app/config.py
"""
Configuration settings for privacy-first CV platform
"""

import os
import base64
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App Settings
    app_name: str = "CV Privacy Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # Security
    secret_key: str = os.getenv(
        "SECRET_KEY", "change-this-in-production-minimum-32-chars"
    )
    jwt_algorithm: str = "HS256"
    session_expire_hours: int = 24

    # Database (minimal usage)
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./cv_privacy.db")

    # Redis for caching and session state (recommended for production)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # CORS Settings
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://your-frontend-domain.com",
    ]

    # Google Drive OAuth - SECURE: All from environment variables
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    google_redirect_uri: str = os.getenv(
        "GOOGLE_REDIRECT_URI", "http://localhost:5173/cloud/callback/google_drive"
    )

    # Microsoft OneDrive OAuth
    microsoft_client_id: str = os.getenv("MICROSOFT_CLIENT_ID", "")
    microsoft_client_secret: str = os.getenv("MICROSOFT_CLIENT_SECRET", "")
    microsoft_redirect_uri: str = os.getenv(
        "MICROSOFT_REDIRECT_URI", "http://localhost:8000/auth/microsoft/callback"
    )

    # Dropbox OAuth
    dropbox_app_key: str = os.getenv("DROPBOX_APP_KEY", "")
    dropbox_app_secret: str = os.getenv("DROPBOX_APP_SECRET", "")
    dropbox_redirect_uri: str = os.getenv(
        "DROPBOX_REDIRECT_URI", "http://localhost:8000/auth/dropbox/callback"
    )

    # Box OAuth
    box_client_id: str = os.getenv("BOX_CLIENT_ID", "")
    box_client_secret: str = os.getenv("BOX_CLIENT_SECRET", "")
    box_redirect_uri: str = os.getenv(
        "BOX_REDIRECT_URI", "http://localhost:8000/auth/box/callback"
    )

    # AI Services
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # AI Service Configuration
    ai_model_primary: str = "gpt-3.5-turbo"  # Cost-optimized primary model
    ai_model_premium: str = "gpt-4o-mini"  # Premium model for quality control
    ai_max_tokens: int = 200  # Token limit for cost control
    ai_temperature: float = 0.7
    ai_timeout_seconds: int = 30

    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_ai_operations_per_day: int = 6  # Free tier

    # Pricing Tiers
    free_tier_ai_operations: int = 6
    pro_tier_ai_operations: int = 50
    business_tier_ai_operations: int = 200

    # File Storage Limits
    max_file_size_mb: int = 10
    max_files_per_session: int = 20

    # Encryption - FIXED: Proper persistent encryption key
    encryption_key: str = os.getenv(
        "ENCRYPTION_KEY", "change-this-to-32-char-key-for-prod"
    )

    # Email (for notifications if needed)
    smtp_server: str = os.getenv("SMTP_SERVER", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_use_tls: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    # Analytics (anonymous only)
    enable_analytics: bool = True
    analytics_retention_days: int = 90

    # Job Search Integration
    enable_job_search: bool = True
    job_search_sources: List[str] = ["github", "stackoverflow", "indeed"]

    # Cloud Storage Settings
    cloud_timeout_seconds: int = 30
    cloud_retry_attempts: int = 3
    cloud_max_file_size_mb: int = 100

    # Session Settings
    session_cleanup_interval_hours: int = 6
    max_sessions_per_ip: int = 10

    # OAuth State Storage Settings
    oauth_state_expiry_minutes: int = 10
    use_redis_for_oauth_state: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

    def validate_oauth_config(self) -> dict:
        """Validate OAuth configuration and return available providers"""
        providers = {}

        if self.google_client_id and self.google_client_secret:
            providers["google_drive"] = True
        else:
            providers["google_drive"] = False

        if self.microsoft_client_id and self.microsoft_client_secret:
            providers["onedrive"] = True
        else:
            providers["onedrive"] = False

        if self.dropbox_app_key and self.dropbox_app_secret:
            providers["dropbox"] = True
        else:
            providers["dropbox"] = False

        if self.box_client_id and self.box_client_secret:
            providers["box"] = True
        else:
            providers["box"] = False

        return providers

    def get_encryption_key_bytes(self) -> bytes:
        """Get properly formatted encryption key for Fernet"""
        # Ensure key is exactly 32 bytes, pad or truncate as needed
        key_bytes = self.encryption_key.encode()[:32].ljust(32, b"0")
        return base64.urlsafe_b64encode(key_bytes)


# Pricing Configuration
class PricingConfig:
    TIERS = {
        "free": {
            "name": "Free",
            "price": 0,
            "ai_operations_daily": 6,
            "cloud_providers": 1,
            "advanced_templates": False,
            "priority_support": False,
            "api_access": False,
            "bulk_operations": False,
            "features": [
                "Basic CV creation",
                "Single cloud provider",
                "Basic templates",
                "AI enhancement (limited)",
            ],
        },
        "pro": {
            "name": "Pro",
            "price": 9.99,
            "ai_operations_daily": 50,
            "cloud_providers": 4,  # All providers
            "advanced_templates": True,
            "priority_support": True,
            "api_access": False,
            "bulk_operations": False,
            "features": [
                "All cloud providers",
                "Advanced templates",
                "Priority AI processing",
                "Advanced cover letter generation",
                "Job matching analysis",
                "Priority support",
            ],
        },
        "business": {
            "name": "Business",
            "price": 29.99,
            "ai_operations_daily": 200,
            "cloud_providers": 4,
            "advanced_templates": True,
            "priority_support": True,
            "api_access": True,
            "bulk_operations": True,
            "features": [
                "Everything in Pro",
                "API access",
                "Bulk CV processing",
                "White-label options",
                "Custom integrations",
                "SLA guarantees",
                "Advanced analytics",
            ],
        },
    }


# Cloud Provider Configuration
class CloudConfig:
    PROVIDERS = {
        "google_drive": {
            "name": "Google Drive",
            "scopes": ["https://www.googleapis.com/auth/drive.file"],
            "api_base": "https://www.googleapis.com/drive/v3",
            "auth_url": "https://accounts.google.com/o/oauth2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "revoke_url": "https://oauth2.googleapis.com/revoke",
        },
        "onedrive": {
            "name": "Microsoft OneDrive",
            "scopes": ["Files.ReadWrite"],
            "api_base": "https://graph.microsoft.com/v1.0",
            "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        },
        "dropbox": {
            "name": "Dropbox",
            "scopes": ["files.content.write", "files.content.read"],
            "api_base": "https://api.dropboxapi.com/2",
            "auth_url": "https://www.dropbox.com/oauth2/authorize",
            "token_url": "https://api.dropboxapi.com/oauth2/token",
            "revoke_url": "https://api.dropboxapi.com/2/auth/token/revoke",
        },
        "box": {
            "name": "Box",
            "scopes": ["root_readwrite"],
            "api_base": "https://api.box.com/2.0",
            "auth_url": "https://account.box.com/api/oauth2/authorize",
            "token_url": "https://api.box.com/oauth2/token",
            "revoke_url": "https://api.box.com/oauth2/revoke",
        },
    }


# AI Model Configuration
class AIConfig:
    MODELS = {
        "cost_optimized": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 200,
            "temperature": 0.7,
            "cost_per_1k_tokens": 0.0005,
        },
        "quality_focused": {
            "model": "gpt-4o-mini",
            "max_tokens": 250,
            "temperature": 0.7,
            "cost_per_1k_tokens": 0.000150,
        },
        "premium": {
            "model": "gpt-4",
            "max_tokens": 300,
            "temperature": 0.7,
            "cost_per_1k_tokens": 0.03,
        },
    }

    # Feature-specific model mapping
    FEATURE_MODELS = {
        "cv_enhancement": "cost_optimized",
        "cover_letter": "cost_optimized",
        "job_analysis": "quality_focused",
        "premium_features": "premium",
    }


# Security Configuration
class SecurityConfig:
    # Session security
    SESSION_COOKIE_NAME = "cv_session"
    SESSION_COOKIE_SECURE = True  # HTTPS only in production
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "lax"

    # Rate limiting by feature
    RATE_LIMITS = {
        "auth": "10/minute",
        "file_upload": "20/hour",
        "ai_operations": "6/day",  # Free tier
        "cloud_operations": "100/hour",
        "public_access": "50/hour",
    }

    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        # "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; img-src 'self' data: fastapi.tiangolo.com;",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }


def get_settings():
    return Settings()


# Add this line:
settings = get_settings()
