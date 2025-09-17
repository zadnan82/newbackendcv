# app/auth/oauth.py
"""
OAuth2 manager for cloud provider authentication
FIXED: Persistent state storage and proper token lifecycle management
"""

import aiohttp
import secrets
import logging
import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlencode
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning(
        "Redis not available, falling back to database storage for OAuth state"
    )

from ..config import get_settings, CloudConfig
from ..schemas import CloudProvider
from ..database import get_db
from ..models import OAuthState
from sqlalchemy import Column, String, DateTime, Text

settings = get_settings()
logger = logging.getLogger(__name__)


class OAuthStateStorage:
    """Abstract OAuth state storage with Redis and database fallback"""

    def __init__(self):
        self.redis_client = None
        # Check if we can use Redis for OAuth state storage
        self.use_redis = (
            REDIS_AVAILABLE
            and hasattr(settings, "use_redis_for_oauth_state")
            and settings.use_redis_for_oauth_state
        )

        if self.use_redis:
            try:
                self.redis_client = redis.from_url(settings.redis_url)
                logger.info("Using Redis for OAuth state storage")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis: {e}, falling back to database"
                )
                self.use_redis = False

        if not self.use_redis:
            logger.info("Using database for OAuth state storage")

    async def store_state(self, state: str, data: dict, expires_minutes: int = 10):
        """Store OAuth state with expiration"""
        expires_at = datetime.utcnow() + timedelta(minutes=expires_minutes)

        if self.use_redis and self.redis_client:
            try:
                await self.redis_client.setex(
                    f"oauth_state:{state}",
                    expires_minutes * 60,
                    json.dumps(data, default=str),
                )
                return
            except Exception as e:
                logger.warning(f"Redis storage failed: {e}, falling back to database")

        # Database fallback
        db = next(get_db())
        try:
            oauth_state = OAuthState(
                state=state,
                provider=data.get("provider", "unknown"),
                session_id=data.get("session_id", "unknown"),
                expires_at=expires_at,
                data=json.dumps(data, default=str),
            )
            db.add(oauth_state)
            db.commit()
        finally:
            db.close()

    async def get_state(self, state: str) -> Optional[dict]:
        """Retrieve and validate OAuth state"""
        if self.use_redis and self.redis_client:
            try:
                data = await self.redis_client.get(f"oauth_state:{state}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis retrieval failed: {e}, trying database")

        # Database fallback
        db = next(get_db())
        try:
            oauth_state = (
                db.query(OAuthState)
                .filter(
                    OAuthState.state == state, OAuthState.expires_at > datetime.utcnow()
                )
                .first()
            )

            if oauth_state:
                return json.loads(oauth_state.data) if oauth_state.data else {}
            return None
        finally:
            db.close()

    async def delete_state(self, state: str):
        """Delete OAuth state after use"""
        if self.use_redis and self.redis_client:
            try:
                await self.redis_client.delete(f"oauth_state:{state}")
                return
            except Exception as e:
                logger.warning(f"Redis deletion failed: {e}, trying database")

        # Database fallback
        db = next(get_db())
        try:
            db.query(OAuthState).filter(OAuthState.state == state).delete()
            db.commit()
        finally:
            db.close()

    async def cleanup_expired_states(self):
        """Clean up expired OAuth states"""
        if self.use_redis:
            # Redis handles expiration automatically
            return

        # Database cleanup
        db = next(get_db())
        try:
            expired_count = (
                db.query(OAuthState)
                .filter(OAuthState.expires_at <= datetime.utcnow())
                .count()
            )

            db.query(OAuthState).filter(
                OAuthState.expires_at <= datetime.utcnow()
            ).delete()

            db.commit()

            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired OAuth states")
        finally:
            db.close()


class OAuthManager:
    """Manage OAuth2 flows for cloud providers"""

    def __init__(self):
        self.state_storage = OAuthStateStorage()

    def _get_provider_config(self, provider: CloudProvider) -> Dict[str, Any]:
        """Get OAuth configuration for provider"""
        configs = {
            CloudProvider.GOOGLE_DRIVE: {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "auth_url": CloudConfig.PROVIDERS["google_drive"]["auth_url"],
                "token_url": CloudConfig.PROVIDERS["google_drive"]["token_url"],
                "revoke_url": CloudConfig.PROVIDERS["google_drive"].get("revoke_url"),
                "scopes": CloudConfig.PROVIDERS["google_drive"]["scopes"],
            },
            CloudProvider.ONEDRIVE: {
                "client_id": getattr(settings, "microsoft_client_id", None),
                "client_secret": getattr(settings, "microsoft_client_secret", None),
                "redirect_uri": getattr(settings, "microsoft_redirect_uri", None),
                "auth_url": CloudConfig.PROVIDERS["onedrive"]["auth_url"],
                "token_url": CloudConfig.PROVIDERS["onedrive"]["token_url"],
                "scopes": CloudConfig.PROVIDERS["onedrive"]["scopes"],
            },
            CloudProvider.DROPBOX: {
                "client_id": getattr(settings, "dropbox_app_key", None),
                "client_secret": getattr(settings, "dropbox_app_secret", None),
                "redirect_uri": getattr(settings, "dropbox_redirect_uri", None),
                "auth_url": CloudConfig.PROVIDERS["dropbox"]["auth_url"],
                "token_url": CloudConfig.PROVIDERS["dropbox"]["token_url"],
                "revoke_url": CloudConfig.PROVIDERS["dropbox"].get("revoke_url"),
                "scopes": CloudConfig.PROVIDERS["dropbox"]["scopes"],
            },
            CloudProvider.BOX: {
                "client_id": getattr(settings, "box_client_id", None),
                "client_secret": getattr(settings, "box_client_secret", None),
                "redirect_uri": getattr(settings, "box_redirect_uri", None),
                "auth_url": CloudConfig.PROVIDERS["box"]["auth_url"],
                "token_url": CloudConfig.PROVIDERS["box"]["token_url"],
                "revoke_url": CloudConfig.PROVIDERS["box"].get("revoke_url"),
                "scopes": CloudConfig.PROVIDERS["box"]["scopes"],
            },
        }

        return configs.get(provider)

    async def get_authorization_url(
        self, provider: CloudProvider, session_id: str, state: str
    ) -> Tuple[str, str]:
        """Generate OAuth authorization URL"""

        config = self._get_provider_config(provider)
        if not config:
            raise ValueError(f"Unsupported provider: {provider}")

        # Check if provider is properly configured
        if not config.get("client_id") or not config.get("client_secret"):
            raise ValueError(
                f"Provider {provider.value} is not properly configured - missing client credentials"
            )

        logger.info(f"🔍 OAuth Config - Redirect URI: {config['redirect_uri']}")
        logger.info(f"🔍 OAuth Config - Client ID: {config['client_id']}")

        # Generate OAuth state
        oauth_state = f"{provider.value}_{state}_{session_id}"

        # Store state with metadata
        state_data = {
            "provider": provider.value,
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Get expiry minutes from settings or use default
        expiry_minutes = getattr(settings, "oauth_state_expiry_minutes", 10)
        await self.state_storage.store_state(
            oauth_state, state_data, expires_minutes=expiry_minutes
        )

        # Build authorization URL based on provider
        if provider == CloudProvider.GOOGLE_DRIVE:
            params = {
                "client_id": config["client_id"],
                "redirect_uri": config["redirect_uri"],
                "scope": " ".join(config["scopes"]),
                "response_type": "code",
                "state": oauth_state,
                "access_type": "offline",
                "prompt": "consent",
            }
        elif provider == CloudProvider.ONEDRIVE:
            params = {
                "client_id": config["client_id"],
                "redirect_uri": config["redirect_uri"],
                "scope": " ".join(config["scopes"]),
                "response_type": "code",
                "state": oauth_state,
            }
        elif provider == CloudProvider.DROPBOX:
            params = {
                "client_id": config["client_id"],
                "redirect_uri": config["redirect_uri"],
                "response_type": "code",
                "state": oauth_state,
                "token_access_type": "offline",
            }
        elif provider == CloudProvider.BOX:
            params = {
                "client_id": config["client_id"],
                "redirect_uri": config["redirect_uri"],
                "response_type": "code",
                "state": oauth_state,
            }

        auth_url = f"{config['auth_url']}?{urlencode(params)}"

        logger.info(f"Generated OAuth URL for {provider.value}")
        return auth_url, oauth_state

    async def exchange_code_for_tokens(
        self, provider: CloudProvider, code: str, state: str, session_id: str = None
    ) -> Dict[str, Any]:
        """Exchange authorization code for access tokens"""

        # Validate and retrieve state
        state_data = await self.state_storage.get_state(state)
        if not state_data:
            raise ValueError("Invalid or expired OAuth state")

        if state_data["provider"] != provider.value:
            raise ValueError("Provider mismatch")

        # Clean up state
        await self.state_storage.delete_state(state)

        config = self._get_provider_config(provider)
        if not config:
            raise ValueError(f"Unsupported provider: {provider}")

        # Prepare token request data
        token_data = {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "redirect_uri": config["redirect_uri"],
            "code": code,
            "grant_type": "authorization_code",
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # Special handling for different providers
        if provider == CloudProvider.DROPBOX:
            headers["Content-Type"] = "application/json"

        async with aiohttp.ClientSession() as session:
            try:
                if provider == CloudProvider.DROPBOX:
                    async with session.post(
                        config["token_url"],
                        json=token_data,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        response_data = await self._handle_token_response(
                            response, provider
                        )
                else:
                    async with session.post(
                        config["token_url"],
                        data=token_data,
                        headers=headers,
                    ) as response:
                        response_data = await self._handle_token_response(
                            response, provider
                        )

                # Standardize token format with additional metadata
                tokens = {
                    "access_token": response_data["access_token"],
                    "refresh_token": response_data.get("refresh_token"),
                    "expires_in": response_data.get("expires_in", 3600),
                    "token_type": response_data.get("token_type", "Bearer"),
                    "scope": response_data.get("scope", ""),
                    "obtained_at": datetime.utcnow().isoformat(),
                    "expires_at": (
                        datetime.utcnow()
                        + timedelta(seconds=response_data.get("expires_in", 3600))
                    ).isoformat(),
                    "provider": provider.value,
                }

                logger.info(f"Successfully exchanged tokens for {provider.value}")
                return tokens

            except Exception as e:
                logger.error(f"Token exchange failed for {provider.value}: {str(e)}")
                raise ValueError(f"Token exchange failed: {str(e)}")

    async def _handle_token_response(
        self, response: aiohttp.ClientResponse, provider: CloudProvider
    ) -> Dict[str, Any]:
        """Handle token response from OAuth provider"""
        if response.status != 200:
            error_text = await response.text()
            logger.error(f"Token exchange error for {provider.value}: {error_text}")
            raise ValueError(f"Token exchange failed: HTTP {response.status}")

        try:
            return await response.json()
        except Exception as e:
            error_text = await response.text()
            logger.error(
                f"Failed to parse token response for {provider.value}: {error_text}"
            )
            raise ValueError(f"Invalid token response format")

    async def refresh_access_token(
        self, provider: CloudProvider, refresh_token: str
    ) -> Dict[str, Any]:
        """Refresh access token using refresh token"""

        if not refresh_token:
            raise ValueError("Refresh token is required")

        config = self._get_provider_config(provider)
        if not config:
            raise ValueError(f"Unsupported provider: {provider}")

        # Prepare refresh request
        refresh_data = {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        # Provider-specific refresh handling
        refresh_url = config["token_url"]

        async with aiohttp.ClientSession() as session:
            try:
                headers = {"Content-Type": "application/x-www-form-urlencoded"}

                if provider == CloudProvider.DROPBOX:
                    headers["Content-Type"] = "application/json"
                    async with session.post(
                        refresh_url,
                        json=refresh_data,
                        headers=headers,
                    ) as response:
                        response_data = await self._handle_token_response(
                            response, provider
                        )
                else:
                    async with session.post(
                        refresh_url,
                        data=refresh_data,
                        headers=headers,
                    ) as response:
                        response_data = await self._handle_token_response(
                            response, provider
                        )

                # Return refreshed tokens with metadata
                tokens = {
                    "access_token": response_data["access_token"],
                    "refresh_token": response_data.get(
                        "refresh_token", refresh_token
                    ),  # Some providers don't return new refresh token
                    "expires_in": response_data.get("expires_in", 3600),
                    "token_type": response_data.get("token_type", "Bearer"),
                    "scope": response_data.get("scope", ""),
                    "obtained_at": datetime.utcnow().isoformat(),
                    "expires_at": (
                        datetime.utcnow()
                        + timedelta(seconds=response_data.get("expires_in", 3600))
                    ).isoformat(),
                    "provider": provider.value,
                }

                logger.info(f"Successfully refreshed tokens for {provider.value}")
                return tokens

            except Exception as e:
                logger.error(f"Token refresh failed for {provider.value}: {str(e)}")
                raise ValueError(f"Token refresh failed: {str(e)}")

    async def revoke_token(self, provider: CloudProvider, access_token: str) -> bool:
        """Revoke access token (if supported by provider)"""

        config = self._get_provider_config(provider)
        revoke_url = config.get("revoke_url") if config else None

        if not revoke_url:
            logger.warning(f"Token revocation not supported for {provider.value}")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                if provider == CloudProvider.GOOGLE_DRIVE:
                    # Google uses POST with token parameter
                    async with session.post(
                        revoke_url,
                        data={"token": access_token},
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                    ) as response:
                        success = response.status == 200

                elif provider == CloudProvider.DROPBOX:
                    # Dropbox uses POST with Authorization header
                    async with session.post(
                        revoke_url,
                        headers={"Authorization": f"Bearer {access_token}"},
                    ) as response:
                        success = response.status == 200

                elif provider == CloudProvider.BOX:
                    # Box uses POST with client credentials and token
                    async with session.post(
                        revoke_url,
                        data={
                            "client_id": config["client_id"],
                            "client_secret": config["client_secret"],
                            "token": access_token,
                        },
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                    ) as response:
                        success = response.status == 200
                else:
                    success = False

                if success:
                    logger.info(f"Successfully revoked token for {provider.value}")
                else:
                    logger.warning(f"Failed to revoke token for {provider.value}")

                return success

        except Exception as e:
            logger.error(f"Error revoking token for {provider.value}: {str(e)}")
            return False

    def validate_provider_config(self, provider: CloudProvider) -> bool:
        """Validate that provider configuration is complete"""
        config = self._get_provider_config(provider)

        if not config:
            return False

        required_fields = [
            "client_id",
            "client_secret",
            "redirect_uri",
            "auth_url",
            "token_url",
        ]

        for field in required_fields:
            if not config.get(field):
                logger.error(f"Missing {field} for {provider.value}")
                return False

        return True

    def get_configured_providers(self) -> List[CloudProvider]:
        """Get list of properly configured providers"""
        configured = []

        for provider in CloudProvider:
            if self.validate_provider_config(provider):
                configured.append(provider)

        return configured

    async def is_token_expired(self, token_data: Dict[str, Any]) -> bool:
        """Check if access token is expired"""
        if "expires_at" not in token_data:
            return False  # Assume valid if no expiration info

        try:
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            return datetime.utcnow() >= expires_at
        except:
            return False

    async def ensure_valid_token(
        self, provider: CloudProvider, token_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure token is valid, refresh if necessary"""
        if not await self.is_token_expired(token_data):
            return token_data

        if not token_data.get("refresh_token"):
            raise ValueError("Token expired and no refresh token available")

        logger.info(f"Refreshing expired token for {provider.value}")
        return await self.refresh_access_token(provider, token_data["refresh_token"])

    async def cleanup_expired_states(self):
        """Clean up expired OAuth states"""
        await self.state_storage.cleanup_expired_states()


# Global OAuth manager instance
oauth_manager = OAuthManager()
