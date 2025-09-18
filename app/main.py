# app/main.py - FIXED: Correct imports and session endpoint
"""
Privacy-First CV Platform - Main FastAPI Application
"""

import logging
from datetime import datetime
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.config import SecurityConfig
from app.config import get_settings

from app.database import (
    create_tables,
    check_database_connection,
    initialize_pricing_tiers,
)
from app.api import google_drive_api, resume, cloud, ai_enhance, cover_letter

# FIXED: Correct import of session functions - at module level, not in function
from app.auth.sessions import (
    get_current_session,
    create_anonymous_session,
    session_manager,  # Add this import
)
from app.schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="CV Privacy Platform",
    description="""
    ## Privacy-First CV Platform
    
    A revolutionary CV/resume platform that stores NO personal data on our servers.
    Your CV data lives in YOUR cloud storage (Google Drive, OneDrive, Dropbox, Box).
    
    ### Key Features
    
    🔒 **Zero Data Liability**: We never store your personal information
    
    ☁️ **Multi-Cloud Support**: Works with Google Drive, OneDrive, Dropbox, Box
    
    🤖 **AI-Powered**: Advanced CV enhancement and cover letter generation
    
    🎨 **Professional Templates**: Beautiful, ATS-friendly resume designs
    
    📱 **QR Code Sharing**: Share your CV instantly with encrypted QR codes
    
    🔄 **Real-time Sync**: Keep your CV updated across all your devices
    
    ### Privacy & Security
    
    - **No user accounts** - Anonymous sessions only
    - **No personal data storage** - Everything stays in your cloud
    - **End-to-end encryption** - Your data is always protected
    - **GDPR compliant** - By design, not by policy
    
    ### Pricing Tiers
    
    - **Free**: 6 AI operations/day, 1 cloud provider
    - **Pro**: 50 AI operations/day, all cloud providers, advanced features
    - **Business**: 200 AI operations/day, API access, bulk operations
    
    ## Getting Started
    
    1. Connect your preferred cloud storage
    2. Create your first CV
    3. Use AI to enhance your content
    4. Generate targeted cover letters
    5. Share with QR codes or public links
    
    Your data never leaves your control.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Cloud Storage",
            "description": "Connect and manage cloud storage providers",
        },
        {
            "name": "Resume Management",
            "description": "Create, edit, and manage your CVs",
        },
        {
            "name": "AI Enhancement",
            "description": "AI-powered CV improvement and job matching",
        },
        {"name": "Cover Letters", "description": "Generate targeted cover letters"},
        {
            "name": "Health & Analytics",
            "description": "System health and anonymous usage analytics",
        },
    ],
)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # Directory might not exist

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware - FIXED: Allow all for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=600,
)

# Trusted hosts middleware (production security)
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "*.cvati.com"]
    )


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)

    for header, value in SecurityConfig.SECURITY_HEADERS.items():
        response.headers[header] = value

    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = datetime.utcnow()

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = (datetime.utcnow() - start_time).total_seconds()

    # Log request details (no personal data)
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {process_time:.3f}s "
        f"- {request.client.host if request.client else 'unknown'}"
    )

    return response


# =================== FIXED SESSION ENDPOINT ===================
@app.post("/api/session", tags=["Health & Analytics"])
@limiter.limit("20/minute")
async def create_session_endpoint(request: Request):
    """
    FIXED: Create anonymous session with proper error handling
    """
    try:
        logger.info("🔄 Creating new anonymous session...")

        # Check for existing session in headers
        auth_header = request.headers.get("Authorization")

        if auth_header and auth_header.startswith("Bearer "):
            # Try to restore existing session
            try:
                token = auth_header.replace("Bearer ", "")
                token_data = session_manager.decode_session_token(token)
                session_id = token_data.get("session_id")

                if session_id:
                    # Check if session exists in database
                    existing_session = await session_manager.get_session(session_id)
                    if existing_session:
                        logger.info(f"✅ Existing session restored: {session_id}")
                        return {
                            "session_id": existing_session["session_id"],
                            "token": token,  # Return the same token
                            "expires_at": existing_session["expires_at"],
                            "restored": True,
                            "message": "Session restored successfully",
                        }
            except Exception as e:
                logger.warning(f"Failed to restore existing session: {e}")
                # Continue to create new session

        # Create new session
        session_data = await create_anonymous_session(request)

        logger.info(f"✅ New session created: {session_data['session_id']}")

        return {
            "session_id": session_data["session_id"],
            "token": session_data["token"],
            "expires_at": session_data["expires_at"],
            "restored": False,
            "message": "Anonymous session created successfully",
        }

    except Exception as e:
        logger.error(f"❌ Session creation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )


# Include routers
app.include_router(cloud.router, prefix="/api/cloud", tags=["Cloud Storage"])
app.include_router(resume.router, prefix="/api/resume", tags=["Resume Management"])
app.include_router(ai_enhance.router, prefix="/api/ai", tags=["AI Enhancement"])
app.include_router(
    cover_letter.router, prefix="/api/cover-letter", tags=["Cover Letters"]
)
app.include_router(
    google_drive_api.router, prefix="/api/google-drive", tags=["google-drive"]
)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting CV Privacy Platform...")

    # Check database connection
    if not check_database_connection():
        logger.error("Database connection failed")
        raise RuntimeError("Cannot connect to database")

    # Create tables if they don't exist
    if not create_tables():
        logger.error("Failed to create database tables")
        raise RuntimeError("Database initialization failed")

    # Initialize pricing tiers
    if not initialize_pricing_tiers():
        logger.warning("Failed to initialize pricing tiers")

    logger.info("✅ CV Privacy Platform started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 CV Privacy Platform shutting down...")


# Health check endpoints
@app.get("/health", tags=["Health & Analytics"])
async def health_check():
    """Basic health check endpoint"""
    from .database import health_check as db_health_check

    db_health = db_health_check()

    return {
        "status": "healthy" if db_health["database_connected"] else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "database": db_health,
        "features": {
            "cloud_providers": ["google_drive", "onedrive", "dropbox", "box"],
            "ai_enhancement": True,
            "cover_letter_generation": True,
            "anonymous_sessions": True,
            "privacy_first": True,
        },
    }


@app.get("/health/detailed", tags=["Health & Analytics"])
async def detailed_health_check():
    """Detailed health check with system metrics"""
    from .database import health_check as db_health_check, get_database_stats

    db_health = db_health_check()
    db_stats = get_database_stats()

    return {
        "status": "healthy" if db_health["database_connected"] else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": settings.environment,
        "database": {**db_health, "statistics": db_stats},
        "configuration": {
            "cors_enabled": len(settings.cors_origins) > 0,
            "rate_limiting": True,
            "encryption": bool(settings.encryption_key),
            "ai_services": bool(settings.openai_api_key),
        },
        "limits": {
            "free_tier_ai_operations": settings.free_tier_ai_operations,
            "max_file_size_mb": settings.max_file_size_mb,
            "session_expire_hours": settings.session_expire_hours,
        },
    }


# API Information endpoints
@app.get("/api/info", tags=["Health & Analytics"])
async def api_info():
    """Get API information and capabilities"""
    return {
        "name": "CV Privacy Platform API",
        "version": "1.0.0",
        "description": "Privacy-first CV platform with cloud storage integration",
        "features": {
            "privacy_first": {
                "description": "No personal data stored on our servers",
                "implementation": "Cloud storage integration",
            },
            "multi_cloud": {
                "description": "Support for multiple cloud providers",
                "providers": ["Google Drive", "OneDrive", "Dropbox", "Box"],
            },
            "ai_powered": {
                "description": "AI-enhanced CV improvement and cover letter generation",
                "models": ["GPT-3.5-turbo", "GPT-4o-mini"],
            },
            "anonymous_sessions": {
                "description": "No user accounts required",
                "session_duration": f"{settings.session_expire_hours} hours",
            },
        },
        "pricing": {
            "free": {"ai_operations_daily": 6, "cloud_providers": 1, "price": "$0"},
            "pro": {
                "ai_operations_daily": 50,
                "cloud_providers": 4,
                "price": "$9.99/month",
            },
            "business": {
                "ai_operations_daily": 200,
                "api_access": True,
                "price": "$29.99/month",
            },
        },
        "endpoints": {
            "cloud_management": "/api/cloud/*",
            "resume_operations": "/api/resume/*",
            "ai_enhancement": "/api/ai/*",
            "cover_letters": "/api/cover-letter/*",
        },
    }


@app.get("/api/pricing", tags=["Health & Analytics"])
async def get_pricing_info():
    """Get current pricing information"""
    from .config import PricingConfig

    return {
        "tiers": PricingConfig.TIERS,
        "currency": "USD",
        "billing_period": "monthly",
        "free_trial": {
            "duration": "unlimited",
            "limitations": "6 AI operations per day, 1 cloud provider",
        },
        "enterprise": {
            "available": True,
            "contact": "enterprise@cvprivacy.com",
            "features": [
                "Custom pricing",
                "SLA",
                "Dedicated support",
                "White-label options",
            ],
        },
    }


# Analytics endpoints (anonymous only)
@app.get("/api/analytics/usage", tags=["Health & Analytics"])
@limiter.limit("10/minute")
async def get_anonymous_usage_stats(request: Request):
    """Get anonymous usage statistics"""
    from .database import get_db
    from .models import UsageAnalytics
    from sqlalchemy import func
    from datetime import datetime, timedelta

    db = next(get_db())

    try:
        # Get stats for last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)

        # Most popular features
        popular_features = (
            db.query(
                UsageAnalytics.feature_used,
                func.count(UsageAnalytics.id).label("usage_count"),
            )
            .filter(UsageAnalytics.timestamp >= thirty_days_ago)
            .group_by(UsageAnalytics.feature_used)
            .all()
        )

        # Success rates by feature
        success_rates = (
            db.query(
                UsageAnalytics.feature_used,
                func.avg(func.cast(UsageAnalytics.success, func.Float)).label(
                    "success_rate"
                ),
            )
            .filter(UsageAnalytics.timestamp >= thirty_days_ago)
            .group_by(UsageAnalytics.feature_used)
            .all()
        )

        # Cloud provider usage
        provider_usage = (
            db.query(
                UsageAnalytics.cloud_provider,
                func.count(UsageAnalytics.id).label("usage_count"),
            )
            .filter(
                UsageAnalytics.timestamp >= thirty_days_ago,
                UsageAnalytics.cloud_provider.isnot(None),
            )
            .group_by(UsageAnalytics.cloud_provider)
            .all()
        )

        return {
            "period": "last_30_days",
            "popular_features": [
                {"feature": row.feature_used, "usage_count": row.usage_count}
                for row in popular_features
            ],
            "success_rates": [
                {
                    "feature": row.feature_used,
                    "success_rate": round(row.success_rate * 100, 2),
                }
                for row in success_rates
            ],
            "cloud_provider_usage": [
                {"provider": row.cloud_provider, "usage_count": row.usage_count}
                for row in provider_usage
                if row.cloud_provider
            ],
            "note": "All data is anonymous and aggregated",
        }

    finally:
        db.close()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code, content=ErrorResponse(detail=exc.detail).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    if settings.debug:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail=f"Internal server error: {str(exc)}").dict(),
        )
    else:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="Internal server error").dict(),
        )


# Development endpoints (only in debug mode)
if settings.debug:

    @app.get("/api/dev/test-cloud/{provider}")
    async def test_cloud_provider(
        provider: str, session: dict = Depends(get_current_session)
    ):
        """Test cloud provider connectivity (development only)"""
        from .cloud.service import cloud_service
        from .schemas import CloudProvider

        try:
            cloud_provider = CloudProvider(provider)

            if not session.get("cloud_tokens"):
                raise HTTPException(
                    status_code=401, detail=f"No {provider} connection found in session"
                )

            status = await cloud_service.get_connection_status(
                session["cloud_tokens"], cloud_provider
            )

            return status

        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")

    @app.get("/api/dev/cleanup")
    async def cleanup_expired_data():
        """Cleanup expired sessions and shares (development only)"""
        from .database import cleanup_expired_sessions

        success = cleanup_expired_sessions()

        return {
            "success": success,
            "message": "Cleanup completed" if success else "Cleanup failed",
        }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CV Privacy Platform API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "privacy_first": True,
        "tagline": "Your CV data never leaves your control",
    }


# Add this endpoint to main.py
@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CV Privacy Platform API</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
        <style>
            body { margin: 0; padding: 0; }
        </style>
    </head>
    <body>
        <redoc spec-url="/openapi.json"></redoc>
        <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
    </body>
    </body>
    </html>
    """)
