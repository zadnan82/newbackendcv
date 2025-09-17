# app/api/cloud.py
"""
Cloud provider connection and management API
"""

from datetime import datetime, timedelta
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import RedirectResponse
import secrets
import asyncio

from pydantic import BaseModel

from app.api.resume import create_resume
from app.config import get_settings

from ..schemas import (
    CloudProvider,
    CloudAuthRequest,
    CloudAuthResponse,
    CloudConnectionStatus,
    CloudSession,
    CompleteCV,
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

from app.auth import sessions

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

        logger.info(f"🔗 CONNECT: Generated auth URL for {provider.value}")
        logger.info(f"🔗 CONNECT: Auth URL: {auth_url}")

        return CloudAuthResponse(auth_url=auth_url, state=oauth_state)

    except Exception as e:
        logger.error(f"Cloud connection initiation error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to initiate cloud connection"
        )


# ===== FIXED: ADD GET ENDPOINT FOR OAUTH CALLBACK =====
@router.get("/callback/{provider}")
async def handle_oauth_callback_get(
    request: Request,  # Add this to access the request
    provider: CloudProvider,
    code: str = Query(..., description="Authorization code from OAuth provider"),
    state: str = Query(..., description="State parameter from OAuth provider"),
    scope: Optional[str] = Query(None, description="Granted scopes"),
    error: Optional[str] = Query(None, description="Error from OAuth provider"),
    error_description: Optional[str] = Query(None, description="Error description"),
):
    """
    Handle OAuth callback from cloud providers (GET method)
    This is the endpoint that Google/Microsoft/etc will redirect to
    """

    logger.info(f"🔗 GET CALLBACK: Processing OAuth callback for {provider.value}")
    logger.info(f"🔗 GET CALLBACK: Code: {code[:20] if code else 'None'}...")
    logger.info(f"🔗 GET CALLBACK: State: {state}")
    logger.info(f"🔗 GET CALLBACK: Scope: {scope}")

    # Check for OAuth errors first
    if error:
        logger.error(f"❌ GET CALLBACK: OAuth error: {error}")
        error_msg = f"OAuth authorization failed: {error}"
        if error_description:
            error_msg += f" - {error_description}"

        # Redirect to frontend with error
        frontend_url = (
            f"http://localhost:5173/cloud/callback/{provider.value}?error={error}"
        )
        if error_description:
            frontend_url += f"&error_description={error_description}"

        return RedirectResponse(url=frontend_url)

    if not code:
        logger.error("❌ GET CALLBACK: No authorization code received")
        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/{provider.value}?error=no_code"
        )

    if not state:
        logger.error("❌ GET CALLBACK: No state parameter received")
        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/{provider.value}?error=no_state"
        )

    try:
        # Process the OAuth callback
        logger.info("🔄 GET CALLBACK: Exchanging authorization code for tokens...")

        # Exchange authorization code for tokens using the configured redirect URI
        redirect_uri = (
            settings.google_redirect_uri
            if provider == CloudProvider.GOOGLE_DRIVE
            else None
        )
        if not redirect_uri:
            raise ValueError(f"No redirect URI configured for {provider.value}")

        tokens = await exchange_code_for_tokens(provider, code, redirect_uri)
        logger.info("✅ GET CALLBACK: Successfully exchanged code for tokens")

        # Get user info
        user_info = await get_user_info_from_tokens(provider, tokens["access_token"])
        logger.info(
            f"✅ GET CALLBACK: Got user info: {user_info.get('email', 'no email')}"
        )

        # FIX: Extract session ID from state parameter
        # The state parameter is typically formatted as "session_id:csrf_token"
        try:
            # Split the state to get session_id (first part before colon)
            session_id = state.split(":")[0] if ":" in state else state
            logger.info(f"🔄 GET CALLBACK: Extracted session ID: {session_id}")

            # Verify session exists
            session_data = await session_manager.get_session(session_id)
            if not session_data:
                logger.error(f"❌ GET CALLBACK: Session not found: {session_id}")
                raise HTTPException(status_code=400, detail="Invalid session")

            logger.info(f"🔄 GET CALLBACK: Storing tokens for session: {session_id}")

            # Get current session tokens
            cloud_tokens = session_data.get("cloud_tokens", {})

            # Store the new tokens
            cloud_tokens[provider.value] = {
                "access_token": tokens["access_token"],
                "refresh_token": tokens.get("refresh_token"),
                "expires_at": tokens.get("expires_at"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
            }

            # Update session
            await session_manager.update_session_cloud_tokens(session_id, cloud_tokens)
            logger.info(
                f"✅ GET CALLBACK: Successfully stored tokens for {provider.value}"
            )

        except Exception as storage_error:
            logger.error(f"❌ GET CALLBACK: Failed to store tokens: {storage_error}")
            # Continue with redirect even if storage fails

        # Redirect to frontend callback handler with success
        frontend_url = f"http://localhost:5173/cloud/callback/{provider.value}?success=true&provider={provider.value}"

        logger.info(f"✅ GET CALLBACK: Redirecting to frontend: {frontend_url}")

        return RedirectResponse(url=frontend_url)

    except Exception as e:
        logger.error(f"❌ GET CALLBACK: Error processing callback: {str(e)}")
        # Redirect to frontend with error
        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/{provider.value}?error=processing_failed&error_description={str(e)}"
        )


# Add this helper function to your app/api/cloud.py file:


async def _ensure_valid_google_token(token_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure Google token is valid, refresh if expired
    CRITICAL FIX: Auto-refresh expired Google tokens
    """
    if not token_data.get("expires_at"):
        return token_data

    try:
        from datetime import datetime

        expires_at = datetime.fromisoformat(token_data["expires_at"])
        now = datetime.utcnow()

        # Check if token expires in the next 5 minutes
        if (expires_at - now).total_seconds() < 300:
            logger.info(f"🔄 Google Drive token expired/expiring, refreshing...")

            if not token_data.get("refresh_token"):
                logger.error(f"❌ No refresh token available for Google Drive")
                return token_data

            try:
                # Refresh using your existing function
                refreshed_tokens = await exchange_google_code_for_tokens(
                    code=None,  # Not needed for refresh
                    redirect_uri=settings.google_redirect_uri,
                    refresh_token=token_data[
                        "refresh_token"
                    ],  # We'll modify the function
                )

                # Merge old data with new tokens
                updated_token_data = {
                    **token_data,
                    "access_token": refreshed_tokens["access_token"],
                    "expires_at": refreshed_tokens["expires_at"],
                    # Keep refresh token from new response or old one
                    "refresh_token": refreshed_tokens.get(
                        "refresh_token", token_data.get("refresh_token")
                    ),
                }

                logger.info(f"✅ Successfully refreshed Google Drive token")
                return updated_token_data

            except Exception as refresh_error:
                logger.error(f"❌ Token refresh failed: {refresh_error}")
                return token_data

    except Exception as e:
        logger.warning(f"Token validation error: {e}")
        return token_data


# Modify your existing exchange_google_code_for_tokens function to handle refresh:


async def exchange_google_code_for_tokens(
    code: str = None, redirect_uri: str = None, refresh_token: str = None
) -> dict:
    """Exchange Google authorization code for tokens OR refresh existing token"""

    token_url = "https://oauth2.googleapis.com/token"

    if refresh_token:
        # Refresh token flow
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
        }
    else:
        # Authorization code flow (existing)
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
                logger.error(
                    f"Google token {'refresh' if refresh_token else 'exchange'} failed: {error_text}"
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"Google token {'refresh' if refresh_token else 'exchange'} failed: {response.status}",
                )

            token_data = await response.json()

            # Calculate expiry time
            expires_in = token_data.get("expires_in", 3600)
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            return {
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get(
                    "refresh_token"
                ),  # May be None for refresh calls
                "expires_at": expires_at.isoformat(),
            }


# In app/api/cloud.py - Fix the get_connection_status function


@router.get("/status", response_model=List[CloudConnectionStatus])
async def get_connection_status(session: dict = Depends(get_current_session)):
    """Get connection status for all cloud providers with simplified token handling"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        logger.info(f"🔍 STATUS: Session ID: {session.get('session_id')}")
        logger.info(f"🔍 STATUS: Cloud tokens found: {list(cloud_tokens.keys())}")

        if not cloud_tokens:
            # Return all providers as disconnected
            return [
                CloudConnectionStatus(
                    provider=CloudProvider.GOOGLE_DRIVE,
                    connected=False,
                    email=None,
                    storage_quota=None,
                ),
                CloudConnectionStatus(
                    provider=CloudProvider.ONEDRIVE,
                    connected=False,
                    email=None,
                    storage_quota=None,
                ),
                CloudConnectionStatus(
                    provider=CloudProvider.DROPBOX,
                    connected=False,
                    email=None,
                    storage_quota=None,
                ),
                CloudConnectionStatus(
                    provider=CloudProvider.BOX,
                    connected=False,
                    email=None,
                    storage_quota=None,
                ),
            ]

        statuses = []

        # Check Google Drive - SIMPLIFIED without auto-refresh for now
        if "google_drive" in cloud_tokens:
            try:
                token_data = cloud_tokens["google_drive"]
                logger.info(
                    f"🔍 STATUS: Google Drive token data keys: {list(token_data.keys())}"
                )

                access_token = token_data.get("access_token")

                if access_token:
                    # Simple connection test without token refresh
                    logger.info("🔍 STATUS: Testing Google Drive connection...")

                    import aiohttp

                    async with aiohttp.ClientSession() as client_session:
                        try:
                            async with client_session.get(
                                "https://www.googleapis.com/oauth2/v1/userinfo",
                                headers={"Authorization": f"Bearer {access_token}"},
                                timeout=aiohttp.ClientTimeout(total=10),
                            ) as response:
                                if response.status == 200:
                                    user_info = await response.json()
                                    logger.info(
                                        f"✅ Google Drive connection verified for {user_info.get('email')}"
                                    )

                                    statuses.append(
                                        CloudConnectionStatus(
                                            provider=CloudProvider.GOOGLE_DRIVE,
                                            connected=True,
                                            email=user_info.get("email"),
                                            storage_quota=None,
                                        )
                                    )
                                else:
                                    logger.warning(
                                        f"❌ Google Drive connection test failed: {response.status}"
                                    )
                                    response_text = await response.text()
                                    logger.warning(f"❌ Response: {response_text}")

                                    statuses.append(
                                        CloudConnectionStatus(
                                            provider=CloudProvider.GOOGLE_DRIVE,
                                            connected=False,
                                            email=None,
                                            storage_quota=None,
                                        )
                                    )
                        except Exception as request_error:
                            logger.error(
                                f"❌ Google Drive request failed: {request_error}"
                            )
                            statuses.append(
                                CloudConnectionStatus(
                                    provider=CloudProvider.GOOGLE_DRIVE,
                                    connected=False,
                                    email=None,
                                    storage_quota=None,
                                )
                            )
                else:
                    logger.warning("❌ No access token for Google Drive")
                    statuses.append(
                        CloudConnectionStatus(
                            provider=CloudProvider.GOOGLE_DRIVE,
                            connected=False,
                            email=None,
                            storage_quota=None,
                        )
                    )

            except Exception as e:
                logger.error(f"❌ Google Drive status check error: {e}")
                statuses.append(
                    CloudConnectionStatus(
                        provider=CloudProvider.GOOGLE_DRIVE,
                        connected=False,
                        email=None,
                        storage_quota=None,
                    )
                )
        else:
            logger.info("❌ No Google Drive tokens found")
            statuses.append(
                CloudConnectionStatus(
                    provider=CloudProvider.GOOGLE_DRIVE,
                    connected=False,
                    email=None,
                    storage_quota=None,
                )
            )

        # Add other providers as disconnected for now
        for provider in [
            CloudProvider.ONEDRIVE,
            CloudProvider.DROPBOX,
            CloudProvider.BOX,
        ]:
            statuses.append(
                CloudConnectionStatus(
                    provider=provider,
                    connected=False,
                    email=None,
                    storage_quota=None,
                )
            )

        logger.info(
            f"🔍 STATUS: Returning {len(statuses)} statuses with {sum(1 for s in statuses if s.connected)} connected"
        )

        return statuses

    except Exception as e:
        logger.error(f"❌ Connection status error: {str(e)}")
        # Return default disconnected state on error
        return [
            CloudConnectionStatus(
                provider=provider, connected=False, email=None, storage_quota=None
            )
            for provider in [
                CloudProvider.GOOGLE_DRIVE,
                CloudProvider.ONEDRIVE,
                CloudProvider.DROPBOX,
                CloudProvider.BOX,
            ]
        ]


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


# KEEP POST ENDPOINT FOR FRONTEND AJAX CALLS
@router.post("/callback/{provider}")
async def handle_oauth_callback_post(
    provider: CloudProvider,
    request_data: OAuthCallbackRequest,
    session: dict = Depends(get_current_session),
):
    """Handle OAuth callback via POST (for frontend AJAX calls)"""
    logger.info(f"🔗 POST CALLBACK: Processing OAuth callback for {provider.value}")
    logger.info(f"🔗 POST CALLBACK: Code: {request_data.code[:20]}...")
    logger.info(f"🔗 POST CALLBACK: State: {request_data.state}")

    try:
        # Exchange the code for tokens
        token_data = await exchange_code_for_tokens(
            provider, request_data.code, request_data.redirect_uri
        )

        # Get user info from tokens
        user_info = await get_user_info_from_tokens(
            provider, token_data["access_token"]
        )

        # Store tokens in session
        session_tokens = session.get("cloud_tokens", {})
        session_tokens[provider.value] = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_at": token_data.get("expires_at"),
            "email": user_info.get("email"),
            "name": user_info.get("name"),
        }

        # Update session with tokens
        await session_manager.update_session_cloud_tokens(
            session["session_id"], session_tokens
        )

        logger.info(f"✅ POST CALLBACK: Successfully connected {provider.value}")
        logger.info(f"✅ User email: {user_info.get('email')}")

        return {
            "success": True,
            "provider": provider.value,
            "message": "Connected successfully",
            "email": user_info.get("email"),
        }

    except Exception as e:
        logger.error(f"❌ POST CALLBACK: Error processing callback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


async def exchange_code_for_tokens(
    provider: CloudProvider, code: str, redirect_uri: str
) -> dict:
    """Exchange authorization code for access tokens"""

    if provider == CloudProvider.GOOGLE_DRIVE:
        return await exchange_google_code_for_tokens(code, redirect_uri)
    elif provider == CloudProvider.ONEDRIVE:
        # Check if Microsoft credentials are configured
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


async def get_user_info_from_tokens(provider: CloudProvider, access_token: str) -> dict:
    """Get user information from access token"""

    if provider == CloudProvider.GOOGLE_DRIVE:
        return await get_google_user_info(access_token)
    elif provider == CloudProvider.ONEDRIVE:
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


@router.get("/debug/tokens/{session_id}")
async def debug_session_tokens(session_id: str):
    """Debug endpoint to check stored tokens for a session"""
    try:
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            return {"error": "Session not found"}

        cloud_tokens = session_data.get("cloud_tokens", {})
        return {
            "session_id": session_id,
            "cloud_tokens": cloud_tokens,
            "providers": list(cloud_tokens.keys()),
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/save/{provider}")
async def save_cv_to_cloud(
    provider: CloudProvider,
    cv: dict,
    session: dict = Depends(get_current_session),
):
    """Save a CV JSON into the user's cloud provider"""
    if provider.value not in session.get("cloud_tokens", {}):
        raise HTTPException(403, f"Not connected to {provider.value}")

    access_token = session["cloud_tokens"][provider.value]["access_token"]

    if provider == CloudProvider.GOOGLE_DRIVE:
        metadata = {
            "name": cv.get("title", "resume") + ".json",
            "mimeType": "application/json",
        }
        files = {
            "metadata": ("metadata", json.dumps(metadata), "application/json"),
            "file": ("file", json.dumps(cv), "application/json"),
        }

        upload_url = (
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
        )

        async with aiohttp.ClientSession() as client:
            async with client.post(
                upload_url,
                headers={"Authorization": f"Bearer {access_token}"},
                data=files,
            ) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    logger.error(f"❌ Google Drive upload failed: {text}")
                    raise HTTPException(500, "Failed to upload file to Google Drive")
                return await resp.json()

    raise HTTPException(501, f"Saving not implemented for {provider.value}")


# Add this debug endpoint to your app/api/cloud.py to see what's happening


@router.get("/debug/status")
async def debug_connection_status(session: dict = Depends(get_current_session)):
    """Debug endpoint to see raw connection status"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        logger.info(f"🔍 DEBUG: Session ID: {session.get('session_id')}")
        logger.info(f"🔍 DEBUG: Cloud tokens found: {list(cloud_tokens.keys())}")

        # Log each token's details (safely)
        for provider_name, token_data in cloud_tokens.items():
            logger.info(
                f"🔍 DEBUG: {provider_name} token data keys: {list(token_data.keys())}"
            )
            logger.info(
                f"🔍 DEBUG: {provider_name} has access_token: {'access_token' in token_data}"
            )
            logger.info(
                f"🔍 DEBUG: {provider_name} email: {token_data.get('email', 'not set')}"
            )

        # Get status for all providers (same as normal endpoint but with debug)
        if not cloud_tokens:
            return {
                "debug": "No cloud tokens found in session",
                "session_keys": list(session.keys()),
                "cloud_tokens": {},
            }

        statuses = await cloud_service.get_all_connection_statuses(cloud_tokens)

        # Convert to debug format
        debug_statuses = []
        for status in statuses:
            debug_statuses.append(
                {
                    "provider": status.provider,
                    "connected": status.connected,
                    "email": status.email if hasattr(status, "email") else None,
                    "error": status.error if hasattr(status, "error") else None,
                    "raw_status": status.dict(),
                }
            )

        return {
            "debug": "Cloud status debug info",
            "token_count": len(cloud_tokens),
            "statuses": debug_statuses,
            "connected_count": sum(1 for s in statuses if s.connected),
        }

    except Exception as e:
        logger.error(f"Debug status error: {str(e)}")
        return {
            "debug": "Error occurred",
            "error": str(e),
            "session_id": session.get("session_id", "unknown"),
        }
