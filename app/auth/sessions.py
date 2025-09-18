# app/auth/sessions.py
"""
Anonymous session management for privacy-first CV platform
FIXED: Minor improvements for better error handling
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import logging

from ..config import get_settings
from ..database import get_db
from ..models import AnonymousSession
from ..schemas import CloudProvider

logger = logging.getLogger(__name__)
settings = get_settings()  # FIXED: Get settings instance

security = HTTPBearer(auto_error=False)


class SessionManager:
    """Manage anonymous sessions without user accounts"""

    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.jwt_algorithm
        self.expire_hours = settings.session_expire_hours

    def _hash_identifier(self, identifier: str) -> str:
        """Create hash for anonymous identification"""
        return hashlib.sha256(f"{identifier}_{self.secret_key}".encode()).hexdigest()[
            :64
        ]

    def create_session_token(self, session_data: Dict[str, Any]) -> str:
        """Create JWT session token"""
        expiration = datetime.utcnow() + timedelta(hours=self.expire_hours)

        payload = {
            **session_data,
            "exp": expiration,
            "iat": datetime.utcnow(),
            "type": "anonymous_session",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode_session_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT session token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") != "anonymous_session":
                raise jwt.InvalidTokenError("Invalid token type")

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Session expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid session token")

    async def create_anonymous_session(
        self, request: Request, cloud_tokens: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create new anonymous session"""
        db = next(get_db())

        try:
            # Generate session ID
            session_id = secrets.token_urlsafe(32)

            # Create session expiration
            expires_at = datetime.utcnow() + timedelta(hours=self.expire_hours)

            # Hash IP and user agent for analytics (privacy-preserving)
            ip_hash = self._hash_identifier(
                request.client.host if request.client else "unknown"
            )
            user_agent = request.headers.get("user-agent", "")
            user_agent_hash = self._hash_identifier(user_agent)

            # Encrypt cloud tokens if provided
            encrypted_tokens = None
            if cloud_tokens:
                try:
                    from ..cloud.service import cloud_service

                    encrypted_tokens = cloud_service._encrypt_tokens(cloud_tokens)
                except Exception as e:
                    logger.warning(f"Failed to encrypt cloud tokens: {e}")
                    # Continue without encrypted tokens

            # Create session record
            session = AnonymousSession(
                session_id=session_id,
                cloud_tokens=encrypted_tokens,
                expires_at=expires_at,
                ip_hash=ip_hash,
                user_agent_hash=user_agent_hash,
            )

            db.add(session)
            db.commit()

            # Create session data for JWT
            session_data = {
                "session_id": session_id,
                "expires_at": expires_at.isoformat(),
                "cloud_providers": list(cloud_tokens.keys()) if cloud_tokens else [],
            }

            # Create JWT token
            token = self.create_session_token(session_data)

            logger.info(f"Created anonymous session: {session_id}")

            return {
                "session_id": session_id,
                "token": token,
                "expires_at": expires_at.isoformat(),
                "cloud_providers": list(cloud_tokens.keys()) if cloud_tokens else [],
            }

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            db.rollback()  # FIXED: Add rollback on error
            raise
        finally:
            db.close()

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID"""
        db = next(get_db())

        try:
            session = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: db.query(AnonymousSession)
                    .filter(
                        AnonymousSession.session_id == session_id,
                        AnonymousSession.expires_at > datetime.utcnow(),
                    )
                    .first()
                ),
                timeout=5.0,  # 5 second timeout
            )

            if not session:
                return None

            # Update last activity
            session.last_activity = datetime.utcnow()
            db.commit()

            # Decrypt cloud tokens
            cloud_tokens = {}
            if session.cloud_tokens:
                try:
                    from ..cloud.service import cloud_service

                    cloud_tokens = cloud_service._decrypt_tokens(session.cloud_tokens)
                except Exception as e:
                    logger.warning(f"Failed to decrypt cloud tokens: {e}")
                    # Continue with empty tokens

            return {
                "session_id": session.session_id,
                "cloud_tokens": cloud_tokens,
                "expires_at": session.expires_at.isoformat(),
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            db.rollback()  # FIXED: Add rollback on error
            return None
        finally:
            db.close()

    async def update_session_cloud_tokens(
        self, session_id: str, cloud_tokens: Dict[str, Any]
    ) -> bool:
        """Update cloud tokens for existing session"""
        db = next(get_db())

        try:
            session = (
                db.query(AnonymousSession)
                .filter(
                    AnonymousSession.session_id == session_id,
                    AnonymousSession.expires_at > datetime.utcnow(),
                )
                .first()
            )

            if not session:
                logger.warning(f"Session not found for token update: {session_id}")
                return False

            # Encrypt and update tokens
            try:
                from ..cloud.service import cloud_service

                session.cloud_tokens = cloud_service._encrypt_tokens(cloud_tokens)
            except Exception as e:
                logger.error(f"Failed to encrypt tokens: {e}")
                return False

            session.last_activity = datetime.utcnow()
            db.commit()

            logger.info(f"Updated cloud tokens for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update session tokens: {e}")
            db.rollback()  # FIXED: Add rollback on error
            return False
        finally:
            db.close()

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke (delete) session"""
        db = next(get_db())

        try:
            session = (
                db.query(AnonymousSession)
                .filter(AnonymousSession.session_id == session_id)
                .first()
            )

            if session:
                db.delete(session)
                db.commit()
                logger.info(f"Revoked session: {session_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to revoke session: {e}")
            db.rollback()  # FIXED: Add rollback on error
            return False
        finally:
            db.close()

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        db = next(get_db())

        try:
            expired_count = (
                db.query(AnonymousSession)
                .filter(AnonymousSession.expires_at <= datetime.utcnow())
                .count()
            )

            db.query(AnonymousSession).filter(
                AnonymousSession.expires_at <= datetime.utcnow()
            ).delete()

            db.commit()

            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired sessions")

            return expired_count

        except Exception as e:
            logger.error(f"Failed to cleanup sessions: {e}")
            db.rollback()  # FIXED: Add rollback on error
            return 0
        finally:
            db.close()


# Global session manager instance
session_manager = SessionManager()


# Dependency functions
async def get_current_session(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """Get current session from JWT token"""

    if not credentials:
        raise HTTPException(status_code=401, detail="Session token required")

    # Decode JWT token
    token_data = session_manager.decode_session_token(credentials.credentials)
    session_id = token_data.get("session_id")

    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session token")

    # Get session from database
    session_data = await session_manager.get_session(session_id)

    if not session_data:
        raise HTTPException(status_code=401, detail="Session not found or expired")

    return session_data


async def get_optional_session(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Get current session, but don't raise error if not present"""

    if not credentials:
        return None

    try:
        return await get_current_session(request, credentials)
    except HTTPException:
        return None


async def require_cloud_connection(
    provider: CloudProvider, session: Dict[str, Any] = Depends(get_current_session)
) -> Dict[str, Any]:
    """Require specific cloud provider connection"""

    cloud_tokens = session.get("cloud_tokens", {})

    if provider.value not in cloud_tokens:
        raise HTTPException(
            status_code=403,
            detail=f"No {provider.value} connection found. Please connect your {provider.value} account first.",
        )

    return cloud_tokens


async def create_anonymous_session(
    request: Request = None, cloud_tokens: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create new anonymous session (standalone function)"""

    if not request:
        # Create minimal request object for testing
        class MockRequest:
            def __init__(self):
                self.client = type("obj", (object,), {"host": "127.0.0.1"})
                self.headers = {}

        request = MockRequest()

    return await session_manager.create_anonymous_session(request, cloud_tokens)


# Rate limiting helpers
async def get_session_for_rate_limiting(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """Get identifier for rate limiting (session ID or IP)"""

    if credentials:
        try:
            token_data = session_manager.decode_session_token(credentials.credentials)
            return f"session:{token_data.get('session_id', 'unknown')}"
        except:
            pass

    # Fall back to IP-based rate limiting
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


# Session validation helpers
def validate_session_permissions(
    session: Dict[str, Any],
    required_providers: Optional[List[CloudProvider]] = None,
    min_session_age_minutes: int = 0,
) -> bool:
    """Validate session has required permissions"""

    # Check session age
    if min_session_age_minutes > 0:
        try:
            created_at = datetime.fromisoformat(session["created_at"])
            age_minutes = (datetime.utcnow() - created_at).total_seconds() / 60

            if age_minutes < min_session_age_minutes:
                return False
        except Exception as e:
            logger.warning(f"Failed to validate session age: {e}")
            return False

    # Check cloud provider connections
    if required_providers:
        cloud_tokens = session.get("cloud_tokens", {})

        for provider in required_providers:
            if provider.value not in cloud_tokens:
                return False

    return True


# Analytics helpers
async def record_session_activity(
    session_id: str, activity_type: str, metadata: Optional[Dict[str, Any]] = None
):
    """Record anonymous session activity for analytics"""

    db = next(get_db())

    try:
        from ..models import UsageAnalytics

        # Hash session ID for privacy
        session_hash = hashlib.sha256(
            f"{session_id}_{settings.secret_key}".encode()
        ).hexdigest()[:64]

        analytics = UsageAnalytics(
            session_hash=session_hash,
            feature_used=activity_type,
            success=True,
            cloud_provider=metadata.get("provider") if metadata else None,
        )

        db.add(analytics)
        db.commit()

    except Exception as e:
        logger.warning(f"Failed to record analytics: {e}")
        try:
            db.rollback()
        except:
            pass

    finally:
        db.close()
