# app/api/cloud.py
"""
Cloud provider connection and management API
"""

from datetime import datetime, timedelta
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
import secrets
import asyncio

from pydantic import BaseModel

from app.config import get_settings

from ..schemas import (
    CloudProvider,
    CloudAuthRequest,
    CloudAuthResponse,
    CloudConnectionStatus,
    CloudSession,
)
from ..auth.sessions import (
    get_current_session,
    get_optional_session,
    session_manager,
    record_session_activity,
)
from ..auth.oauth import oauth_manager
from ..cloud.service import cloud_service, CloudProviderError
import aiohttp

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/providers")
async def list_available_providers():
    """Get list of available cloud providers"""
    return {
        "providers": [
            {
                "id": "google_drive",
                "name": "Google Drive",
                "description": "Store your CVs in Google Drive",
                "logo_url": "/static/logos/google-drive.png",
                "supported_features": ["read", "write", "delete", "folders"],
            },
            {
                "id": "onedrive",
                "name": "Microsoft OneDrive",
                "description": "Store your CVs in OneDrive",
                "logo_url": "/static/logos/onedrive.png",
                "supported_features": ["read", "write", "delete", "folders"],
            },
            {
                "id": "dropbox",
                "name": "Dropbox",
                "description": "Store your CVs in Dropbox",
                "logo_url": "/static/logos/dropbox.png",
                "supported_features": ["read", "write", "delete", "folders"],
            },
            {
                "id": "box",
                "name": "Box",
                "description": "Store your CVs in Box",
                "logo_url": "/static/logos/box.png",
                "supported_features": ["read", "write", "delete", "folders"],
            },
        ]
    }


@router.post("/connect/{provider}", response_model=CloudAuthResponse)
async def initiate_cloud_connection(
    provider: CloudProvider,
    request: Request,
    session: dict = Depends(get_optional_session),
):
    """Initiate OAuth connection to cloud provider"""

    try:
        # Create session if doesn't exist
        if not session:
            from ..auth.sessions import create_anonymous_session

            session_data = await create_anonymous_session(request)
            session_id = session_data["session_id"]
        else:
            session_id = session["session_id"]

        # Generate OAuth state parameter
        state = secrets.token_urlsafe(32)

        # Get OAuth URL
        auth_url, oauth_state = await oauth_manager.get_authorization_url(
            provider, session_id, state
        )

        return CloudAuthResponse(auth_url=auth_url, state=oauth_state)

    except Exception as e:
        logger.error(f"Cloud connection initiation error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to initiate cloud connection"
        )


@router.get("/status", response_model=List[CloudConnectionStatus])
async def get_connection_status(session: dict = Depends(get_current_session)):
    """Get connection status for all cloud providers"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        # DEBUG: Log what tokens we have
        logger.info(f"🔍 STATUS: Session ID: {session.get('session_id')}")
        logger.info(f"🔍 STATUS: Cloud tokens found: {list(cloud_tokens.keys())}")
        logger.info(f"🔍 STATUS: Full cloud_tokens: {cloud_tokens}")

        # Get status for all providers
        statuses = await cloud_service.get_all_connection_statuses(cloud_tokens)

        logger.info(f"🔍 STATUS: Returning statuses: {[s.dict() for s in statuses]}")

        return statuses

    except Exception as e:
        logger.error(f"Connection status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get connection status")


@router.get("/status/{provider}", response_model=CloudConnectionStatus)
async def get_provider_status(
    provider: CloudProvider, session: dict = Depends(get_current_session)
):
    """Get connection status for specific cloud provider"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        status = await cloud_service.get_connection_status(cloud_tokens, provider)

        return status

    except Exception as e:
        logger.error(f"Provider status error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get {provider.value} status"
        )


@router.delete("/disconnect/{provider}")
async def disconnect_provider(
    provider: CloudProvider, session: dict = Depends(get_current_session)
):
    """Disconnect from cloud provider"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=404, detail=f"No {provider.value} connection found"
            )

        # Remove provider tokens
        del cloud_tokens[provider.value]

        # Update session
        await session_manager.update_session_cloud_tokens(
            session["session_id"], cloud_tokens
        )

        # Record activity
        await record_session_activity(
            session["session_id"], "cloud_disconnected", {"provider": provider.value}
        )

        return {
            "message": f"Disconnected from {provider.value}",
            "provider": provider.value,
        }

    except Exception as e:
        logger.error(f"Provider disconnection error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to disconnect from {provider.value}"
        )


@router.post("/test/{provider}")
async def test_provider_connection(
    provider: CloudProvider, session: dict = Depends(get_current_session)
):
    """Test connection to cloud provider"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Test connection
        access_token = cloud_tokens[provider.value]["access_token"]
        health_info = await cloud_service.get_provider_health(provider, access_token)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cloud_test",
            {"provider": provider.value, "status": health_info["status"]},
        )

        return health_info

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud provider error: {str(e)}")
    except Exception as e:
        logger.error(f"Provider test error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to test {provider.value} connection"
        )


@router.get("/health")
async def check_all_providers_health(session: dict = Depends(get_current_session)):
    """Check health of all connected providers"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if not cloud_tokens:
            return {"message": "No cloud providers connected", "providers": []}

        # Test all connected providers concurrently
        health_tasks = []
        for provider_name, token_data in cloud_tokens.items():
            try:
                provider = CloudProvider(provider_name)
                access_token = token_data["access_token"]

                health_task = cloud_service.get_provider_health(provider, access_token)
                health_tasks.append(health_task)
            except:
                continue

        if health_tasks:
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        else:
            health_results = []

        # Process results
        provider_health = []
        for result in health_results:
            if isinstance(result, Exception):
                provider_health.append(
                    {"provider": "unknown", "status": "error", "error": str(result)}
                )
            else:
                provider_health.append(result)

        # Calculate overall health
        healthy_count = sum(1 for p in provider_health if p.get("status") == "healthy")
        total_count = len(provider_health)

        overall_status = (
            "healthy"
            if healthy_count == total_count
            else "degraded"
            if healthy_count > 0
            else "unhealthy"
        )

        return {
            "overall_status": overall_status,
            "healthy_providers": healthy_count,
            "total_providers": total_count,
            "providers": provider_health,
        }

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check provider health")


@router.post("/refresh-tokens")
async def refresh_all_tokens(session: dict = Depends(get_current_session)):
    """Refresh access tokens for all connected providers"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if not cloud_tokens:
            raise HTTPException(status_code=404, detail="No cloud providers connected")

        refresh_results = {}
        updated_tokens = {}

        for provider_name, token_data in cloud_tokens.items():
            try:
                provider = CloudProvider(provider_name)

                # Attempt to refresh tokens
                new_tokens = await oauth_manager.refresh_access_token(
                    provider, token_data.get("refresh_token")
                )

                updated_tokens[provider_name] = new_tokens
                refresh_results[provider_name] = {"success": True}

            except Exception as e:
                logger.warning(f"Failed to refresh {provider_name} tokens: {e}")
                # Keep old tokens if refresh fails
                updated_tokens[provider_name] = token_data
                refresh_results[provider_name] = {"success": False, "error": str(e)}

        # Update session with refreshed tokens
        await session_manager.update_session_cloud_tokens(
            session["session_id"], updated_tokens
        )

        # Record activity
        await record_session_activity(
            session["session_id"], "tokens_refreshed", {"results": refresh_results}
        )

        return {"message": "Token refresh completed", "results": refresh_results}

    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to refresh tokens")


class OAuthCallbackRequest(BaseModel):
    code: str
    state: str
    redirect_uri: str


@router.post("/callback/{provider}")
async def handle_oauth_callback_post(
    provider: CloudProvider,
    request_data: OAuthCallbackRequest,
    session: dict = Depends(get_current_session),
):
    logger.info(f"🔗 CALLBACK: Processing OAuth callback for {provider.value}")
    logger.info(f"🔗 CALLBACK: Code: {request_data.code[:20]}...")
    logger.info(f"🔗 CALLBACK: State: {request_data.state}")

    try:
        # ACTUALLY exchange the code for real tokens
        token_data = await exchange_code_for_tokens(
            provider, request_data.code, request_data.redirect_uri
        )

        # Get user info from the real tokens
        user_info = await get_user_info_from_tokens(
            provider, token_data["access_token"]
        )

        # Store the REAL tokens in session
        session_tokens = session.get("cloud_tokens", {})
        session_tokens[provider.value] = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_at": token_data.get("expires_at"),
            "email": user_info.get("email"),
            "name": user_info.get("name"),
        }

        # Update session with REAL tokens
        await session_manager.update_session_cloud_tokens(
            session["session_id"], session_tokens
        )

        logger.info(f"✅ CALLBACK: Successfully connected {provider.value}")
        logger.info(f"✅ User email: {user_info.get('email')}")

        return {
            "success": True,
            "provider": provider.value,
            "message": "Connected successfully",
            "email": user_info.get("email"),
        }

    except Exception as e:
        logger.error(f"❌ CALLBACK: Error processing callback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def exchange_google_code_for_tokens(code: str, redirect_uri: str) -> dict:
    """Exchange Google authorization code for tokens"""

    import aiohttp
    from ..config import get_settings

    settings = get_settings()

    token_url = "https://oauth2.googleapis.com/token"

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": redirect_uri,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(token_url, data=data) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Google token exchange failed: {error_text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Google token exchange failed: {response.status}",
                )

            token_data = await response.json()

            # Calculate expiry time
            expires_in = token_data.get("expires_in", 3600)
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            return {
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token"),
                "expires_at": expires_at.isoformat(),
            }


async def get_google_user_info(access_token: str) -> dict:
    """Get Google user information"""

    user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with aiohttp.ClientSession() as session:
        async with session.get(user_info_url, headers=headers) as response:
            if response.status == 200:
                user_data = await response.json()
                return {"email": user_data.get("email"), "name": user_data.get("name")}
            else:
                logger.warning(f"Failed to get Google user info: {response.status}")
                return {"email": None, "name": None}


async def exchange_microsoft_code_for_tokens(code: str, redirect_uri: str) -> dict:
    """Exchange Microsoft authorization code for tokens"""

    token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": settings.microsoft_client_id,
        "client_secret": settings.microsoft_client_secret,
        "redirect_uri": redirect_uri,
        "scope": "Files.ReadWrite offline_access",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(token_url, data=data) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Microsoft token exchange failed: {error_text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Microsoft token exchange failed: {response.status}",
                )

            token_data = await response.json()

            # Calculate expiry time
            expires_in = token_data.get("expires_in", 3600)
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            return {
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token"),
                "expires_at": expires_at.isoformat(),
            }


async def get_microsoft_user_info(access_token: str) -> dict:
    """Get Microsoft user information"""

    import aiohttp

    user_info_url = "https://graph.microsoft.com/v1.0/me"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with aiohttp.ClientSession() as session:
        async with session.get(user_info_url, headers=headers) as response:
            if response.status == 200:
                user_data = await response.json()
                return {
                    "email": user_data.get("mail")
                    or user_data.get("userPrincipalName"),
                    "name": user_data.get("displayName"),
                }
            else:
                logger.warning(f"Failed to get Microsoft user info: {response.status}")
                return {"email": None, "name": None}


# UPDATED exchange_code_for_tokens function to handle missing providers gracefully
async def exchange_code_for_tokens(
    provider: CloudProvider, code: str, redirect_uri: str
) -> dict:
    """Exchange authorization code for access tokens"""

    if provider == CloudProvider.GOOGLE_DRIVE:
        return await exchange_google_code_for_tokens(code, redirect_uri)
    elif provider == CloudProvider.ONEDRIVE:
        # Check if Microsoft credentials are configured
        settings = get_settings()
        if not settings.microsoft_client_id or not settings.microsoft_client_secret:
            raise HTTPException(
                status_code=501,
                detail="Microsoft OAuth not configured - missing client credentials",
            )
        return await exchange_microsoft_code_for_tokens(code, redirect_uri)
    else:
        # For now, only Google Drive is fully supported
        raise HTTPException(
            status_code=501, detail=f"OAuth not yet implemented for {provider.value}"
        )


# UPDATED get_user_info_from_tokens function
async def get_user_info_from_tokens(provider: CloudProvider, access_token: str) -> dict:
    """Get user information from access token"""

    if provider == CloudProvider.GOOGLE_DRIVE:
        return await get_google_user_info(access_token)
    elif provider == CloudProvider.ONEDRIVE:
        settings = get_settings()
        if not settings.microsoft_client_id or not settings.microsoft_client_secret:
            return {"email": "not_configured@onedrive.com", "name": "OneDrive User"}
        return await get_microsoft_user_info(access_token)
    else:
        # Return placeholder info for unsupported providers
        return {
            "email": f"user@{provider.value}.com",
            "name": f"{provider.value.title()} User",
        }


@router.get("/debug/oauth-config")
async def debug_oauth_config():
    """Debug endpoint to see actual OAuth configuration"""
    configs = {}
    for provider in CloudProvider:
        config = oauth_manager._get_provider_config(provider)
        if config:
            configs[provider.value] = {
                "client_id": config["client_id"],
                "redirect_uri": config["redirect_uri"],
                "has_client_secret": bool(config["client_secret"]),
            }
    return configs
