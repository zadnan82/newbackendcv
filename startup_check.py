# startup_check.py
"""
Security and configuration validation script for CV Privacy Platform
Run this before starting your application to ensure everything is secure
"""

import os
import secrets
import base64
from cryptography.fernet import Fernet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_secure_key(length=32):
    """Generate a cryptographically secure random key"""
    return secrets.token_urlsafe(length)


def generate_fernet_key():
    """Generate a Fernet-compatible encryption key"""
    return Fernet.generate_key().decode()


def check_environment_variables():
    """Check and validate environment variables"""
    logger.info("🔍 Checking environment variables...")

    critical_vars = {
        "SECRET_KEY": "Application secret key for JWT tokens",
        "ENCRYPTION_KEY": "Encryption key for cloud tokens",
    }

    optional_vars = {
        "GOOGLE_CLIENT_ID": "Google Drive OAuth",
        "GOOGLE_CLIENT_SECRET": "Google Drive OAuth",
        "MICROSOFT_CLIENT_ID": "OneDrive OAuth",
        "MICROSOFT_CLIENT_SECRET": "OneDrive OAuth",
        "DROPBOX_APP_KEY": "Dropbox OAuth",
        "DROPBOX_APP_SECRET": "Dropbox OAuth",
        "BOX_CLIENT_ID": "Box OAuth",
        "BOX_CLIENT_SECRET": "Box OAuth",
        "OPENAI_API_KEY": "OpenAI API access",
        "ANTHROPIC_API_KEY": "Anthropic API access",
    }

    issues = []

    # Check critical variables
    for var, description in critical_vars.items():
        value = os.getenv(var)
        if not value:
            issues.append(f"❌ CRITICAL: {var} is not set ({description})")
        elif len(value) < 32:
            issues.append(f"⚠️  WARNING: {var} should be at least 32 characters long")
        else:
            logger.info(f"✅ {var} is properly configured")

    # Check optional variables
    configured_providers = []
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if "CLIENT_ID" in var or "APP_KEY" in var:
                provider = var.split("_")[0].lower()
                configured_providers.append(provider)
            logger.info(f"✅ {var} is configured for {description}")
        else:
            logger.info(f"ℹ️  {var} not set ({description} disabled)")

    if configured_providers:
        logger.info(
            f"🔗 Cloud providers configured: {', '.join(set(configured_providers))}"
        )
    else:
        issues.append(
            "⚠️  WARNING: No cloud providers configured - users won't be able to store CVs"
        )

    # Check AI services
    ai_services = []
    if os.getenv("OPENAI_API_KEY"):
        ai_services.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        ai_services.append("Anthropic")

    if ai_services:
        logger.info(f"🤖 AI services configured: {', '.join(ai_services)}")
    else:
        issues.append(
            "⚠️  WARNING: No AI services configured - AI features will be disabled"
        )

    return issues


def check_database_config():
    """Check database configuration"""
    logger.info("🗄️  Checking database configuration...")

    db_url = os.getenv("DATABASE_URL", "sqlite:///./cv_privacy.db")

    if db_url.startswith("sqlite"):
        logger.info("📁 Using SQLite database (development mode)")
        return []
    elif "postgresql" in db_url:
        logger.info("🐘 Using PostgreSQL database (production mode)")
        return []
    else:
        return ["⚠️  WARNING: Unknown database type in DATABASE_URL"]


def check_redis_config():
    """Check Redis configuration for production"""
    logger.info("🔴 Checking Redis configuration...")

    redis_url = os.getenv("REDIS_URL")
    use_redis = os.getenv("USE_REDIS_FOR_OAUTH_STATE", "true").lower() == "true"

    if use_redis and not redis_url:
        return ["⚠️  WARNING: USE_REDIS_FOR_OAUTH_STATE is true but REDIS_URL not set"]
    elif redis_url:
        logger.info("✅ Redis configured for OAuth state storage")
        return []
    else:
        logger.info("ℹ️  Redis not configured, using database for OAuth state")
        return []


def check_security_settings():
    """Check security-related settings"""
    logger.info("🔒 Checking security settings...")

    issues = []

    # Check environment
    env = os.getenv("ENVIRONMENT", "development")
    debug = os.getenv("DEBUG", "false").lower() == "true"

    if env == "production" and debug:
        issues.append("❌ CRITICAL: DEBUG should be false in production")

    # Check CORS origins
    cors_origins = os.getenv("CORS_ORIGINS", "")
    if not cors_origins:
        issues.append(
            "⚠️  WARNING: CORS_ORIGINS not set - may cause frontend connection issues"
        )

    # Check session settings
    session_expire = int(os.getenv("SESSION_EXPIRE_HOURS", "24"))
    if session_expire > 168:  # 1 week
        issues.append("⚠️  WARNING: SESSION_EXPIRE_HOURS is very high (>1 week)")

    logger.info(f"🌍 Environment: {env}")
    logger.info(f"🐛 Debug mode: {debug}")
    logger.info(f"⏰ Session expiry: {session_expire} hours")

    return issues


def validate_encryption_key():
    """Validate that encryption key works with Fernet"""
    logger.info("🔐 Validating encryption key...")

    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        return ["❌ CRITICAL: ENCRYPTION_KEY not set"]

    try:
        # Test if key works with Fernet
        key_bytes = encryption_key.encode()[:32].ljust(32, b"0")
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        cipher = Fernet(fernet_key)

        # Test encryption/decryption
        test_data = b"test_encryption"
        encrypted = cipher.encrypt(test_data)
        decrypted = cipher.decrypt(encrypted)

        if decrypted == test_data:
            logger.info("✅ Encryption key is valid")
            return []
        else:
            return ["❌ CRITICAL: Encryption key validation failed"]

    except Exception as e:
        return [f"❌ CRITICAL: Encryption key error: {str(e)}"]


def generate_secure_config():
    """Generate secure configuration values"""
    logger.info("🔧 Generating secure configuration values...")

    print("\n" + "=" * 60)
    print("SECURE CONFIGURATION GENERATOR")
    print("=" * 60)

    print(f"\n# Add these to your .env file:")
    print(f"SECRET_KEY={generate_secure_key(64)}")
    print(f"ENCRYPTION_KEY={generate_secure_key(32)}")

    print(f"\n# Alternative Fernet-compatible encryption key:")
    print(f"ENCRYPTION_KEY={generate_fernet_key()}")

    print("\n" + "=" * 60)


def main():
    """Main security check function"""
    print("🚀 CV Privacy Platform Security Check")
    print("=" * 50)

    all_issues = []

    # Run all checks
    all_issues.extend(check_environment_variables())
    all_issues.extend(check_database_config())
    all_issues.extend(check_redis_config())
    all_issues.extend(check_security_settings())
    all_issues.extend(validate_encryption_key())

    print("\n" + "=" * 50)
    print("SECURITY CHECK SUMMARY")
    print("=" * 50)

    if not all_issues:
        print("🎉 All security checks passed! Your application is ready to run.")
        return True
    else:
        print("⚠️  Issues found:")
        for issue in all_issues:
            print(f"   {issue}")

        critical_issues = [i for i in all_issues if "CRITICAL" in i]
        if critical_issues:
            print(
                f"\n❌ {len(critical_issues)} CRITICAL issues must be fixed before running in production!"
            )
            return False
        else:
            print(
                f"\n⚠️  {len(all_issues)} warnings found - review before production deployment"
            )
            return True


def create_env_file():
    """Create a template .env file if it doesn't exist"""
    if not os.path.exists(".env"):
        logger.info("📝 Creating .env template file...")

        env_template = f"""# CV Privacy Platform Environment Variables
# Generated on {os.popen("date").read().strip()}

# SECURITY SETTINGS (CRITICAL)
SECRET_KEY={generate_secure_key(64)}
ENCRYPTION_KEY={generate_secure_key(32)}

# DATABASE
DATABASE_URL=sqlite:///./cv_privacy.db

# REDIS (Optional)
REDIS_URL=redis://localhost:6379/0
USE_REDIS_FOR_OAUTH_STATE=true

# GOOGLE DRIVE OAUTH (Replace with your credentials)
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/cloud/callback/google_drive

# AI SERVICES
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key

# DEVELOPMENT SETTINGS
ENVIRONMENT=development
DEBUG=false
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# SESSION SETTINGS
SESSION_EXPIRE_HOURS=24
OAUTH_STATE_EXPIRY_MINUTES=10
MAX_SESSIONS_PER_IP=10
"""

        with open(".env", "w") as f:
            f.write(env_template)

        print("✅ Created .env file with secure defaults")
        print("🔧 Please update the OAuth credentials and API keys in .env")
    else:
        print("ℹ️  .env file already exists")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_secure_config()
    elif len(sys.argv) > 1 and sys.argv[1] == "init":
        create_env_file()
    else:
        # Load environment variables from .env if it exists
        if os.path.exists(".env"):
            from dotenv import load_dotenv

            load_dotenv()

        success = main()

        if not success:
            print("\n🔧 Run 'python startup_check.py generate' to create secure keys")
            print("🔧 Run 'python startup_check.py init' to create .env template")
            sys.exit(1)
        else:
            print("\n🚀 Ready to start your application!")
            sys.exit(0)
