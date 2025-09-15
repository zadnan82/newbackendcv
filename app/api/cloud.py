# app/api/cloud.py
"""
Cloud provider connection and management API
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
import secrets
import asyncio

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

logger = logging.getLogger(__name__)
router = APIRouter()


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


@router.get("/callback/{provider}")
async def handle_oauth_callback(
    provider: CloudProvider,
    code: str,
    state: str,
    request: Request,
    session_id: str = None,
):
    """Handle OAuth callback from cloud provider"""

    try:
        # Exchange code for tokens
        tokens = await oauth_manager.exchange_code_for_tokens(
            provider, code, state, session_id
        )

        # Get or create session
        if session_id:
            session_data = await session_manager.get_session(session_id)
            if session_data:
                # Update existing session with new tokens
                existing_tokens = session_data.get("cloud_tokens", {})
                existing_tokens[provider.value] = tokens

                await session_manager.update_session_cloud_tokens(
                    session_id, existing_tokens
                )
            else:
                # Create new session
                session_data = await session_manager.create_anonymous_session(
                    request, {provider.value: tokens}
                )
        else:
            # Create new session
            session_data = await session_manager.create_anonymous_session(
                request, {provider.value: tokens}
            )

        # Record activity
        await record_session_activity(
            session_data["session_id"], "cloud_connected", {"provider": provider.value}
        )

        # Redirect to frontend with success
        frontend_url = request.headers.get("referer", "http://localhost:3000")
        return RedirectResponse(
            url=f"{frontend_url}/cloud/connected?provider={provider.value}&session={session_data['token']}"
        )

    except Exception as e:
        logger.error(f"OAuth callback error: {str(e)}")
        # Redirect to frontend with error
        frontend_url = request.headers.get("referer", "http://localhost:3000")
        return RedirectResponse(
            url=f"{frontend_url}/cloud/error?message=connection_failed"
        )


@router.get("/status", response_model=List[CloudConnectionStatus])
async def get_connection_status(session: dict = Depends(get_current_session)):
    """Get connection status for all cloud providers"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        # Get status for all providers
        statuses = await cloud_service.get_all_connection_statuses(cloud_tokens)

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
