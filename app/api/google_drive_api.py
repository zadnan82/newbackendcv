# app/api/google_drive_api.py
"""
Simplified Google Drive API - Focus on getting Google Drive working first
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from datetime import datetime
import secrets

from ..auth.sessions import (
    get_current_session,
    get_optional_session,
    session_manager,
    record_session_activity,
)
from ..cloud.google_drive_service import google_drive_service, GoogleDriveError
from ..schemas import CompleteCV, CloudConnectionStatus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/providers")
async def list_google_drive_info():
    """Get Google Drive provider information"""
    return {
        "providers": [
            {
                "id": "google_drive",
                "name": "Google Drive",
                "description": "Store your CVs in Google Drive",
                "logo_url": "/static/logos/google-drive.png",
                "supported_features": ["read", "write", "delete", "folders"],
                "status": "available",
            }
        ]
    }


@router.post("/connect")
async def initiate_google_drive_connection(
    request: Request, session: dict = Depends(get_optional_session)
):
    """Initiate Google Drive OAuth connection"""
    try:
        # Create session if doesn't exist
        if not session:
            from ..auth.sessions import create_anonymous_session

            session_data = await create_anonymous_session(request)
            session_id = session_data["session_id"]
        else:
            session_id = session["session_id"]

        # Generate OAuth state parameter
        state = f"{session_id}:{secrets.token_urlsafe(16)}"

        # Get OAuth URL from Google Drive service
        auth_url = google_drive_service.get_oauth_url(state)

        logger.info(f"🔗 Generated Google Drive OAuth URL for session: {session_id}")

        return {"auth_url": auth_url, "state": state, "provider": "google_drive"}

    except Exception as e:
        logger.error(f"❌ Google Drive connection initiation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate Google Drive connection: {str(e)}",
        )


@router.get("/callback")
async def handle_google_drive_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="State parameter from Google"),
    scope: Optional[str] = Query(None, description="Granted scopes"),
    error: Optional[str] = Query(None, description="Error from Google"),
    error_description: Optional[str] = Query(None, description="Error description"),
):
    """Handle Google Drive OAuth callback"""

    logger.info(f"🔗 Google Drive OAuth callback received")
    logger.info(f"🔗 Code: {code[:20] if code else 'None'}...")
    logger.info(f"🔗 State: {state}")
    logger.info(f"🔗 Error: {error}")

    # Check for OAuth errors
    if error:
        logger.error(f"❌ Google OAuth error: {error}")
        error_msg = f"Google authorization failed: {error}"
        if error_description:
            error_msg += f" - {error_description}"

        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/google_drive?error={error}&error_description={error_description or ''}"
        )

    if not code:
        logger.error("❌ No authorization code received")
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/google_drive?error=no_code"
        )

    if not state:
        logger.error("❌ No state parameter received")
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/google_drive?error=no_state"
        )

    try:
        # Extract session ID from state
        session_id = state.split(":")[0] if ":" in state else state
        logger.info(f"🔄 Processing callback for session: {session_id}")

        # Verify session exists
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            logger.error(f"❌ Session not found: {session_id}")
            return RedirectResponse(
                url="http://localhost:5173/cloud/callback/google_drive?error=invalid_session"
            )

        # Exchange code for tokens
        logger.info("🔄 Exchanging authorization code for tokens...")
        tokens = await google_drive_service.exchange_code_for_tokens(code)

        logger.info("✅ Successfully exchanged code for tokens")

        # Test the connection to get user info
        connection_test = await google_drive_service.test_connection(tokens)
        if not connection_test["success"]:
            raise GoogleDriveError(
                f"Connection test failed: {connection_test.get('error')}"
            )

        # Store tokens in session
        cloud_tokens = session_data.get("cloud_tokens", {})
        cloud_tokens["google_drive"] = {
            **tokens,
            "email": connection_test["user"]["email"],
            "name": connection_test["user"]["name"],
        }

        # Update session with tokens
        await session_manager.update_session_cloud_tokens(session_id, cloud_tokens)
        logger.info("✅ Successfully stored Google Drive tokens")

        # Record activity
        await record_session_activity(
            session_id,
            "google_drive_connected",
            {"email": connection_test["user"]["email"]},
        )

        # Redirect to frontend with success
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/google_drive?success=true&provider=google_drive"
        )

    except Exception as e:
        logger.error(f"❌ Google Drive callback processing failed: {str(e)}")
        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/google_drive?error=processing_failed&error_description={str(e)}"
        )


@router.get("/status")
async def get_google_drive_status(session: dict = Depends(get_current_session)):
    """Get Google Drive connection status"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            return {
                "provider": "google_drive",
                "connected": False,
                "email": None,
                "storage_quota": None,
            }

        # Check connection status using the service
        status = await google_drive_service.get_connection_status(google_drive_tokens)
        return status.dict()

    except Exception as e:
        logger.error(f"❌ Google Drive status check failed: {str(e)}")
        return {
            "provider": "google_drive",
            "connected": False,
            "email": None,
            "storage_quota": None,
            "error": str(e),
        }


@router.post("/test-save")
async def test_google_drive_save(session: dict = Depends(get_current_session)):
    """Test Google Drive save functionality with simple data"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            return {
                "success": False,
                "error": "No Google Drive connection found",
                "provider": "google_drive",
            }

        # Create simple test data
        test_data = CompleteCV(
            title="Test CV",
            personal_info={"full_name": "Test User", "email": "test@example.com"},
            sections={},
        )

        # Save test data
        file_id = await google_drive_service.save_cv(google_drive_tokens, test_data)

        return {"success": True, "file_id": file_id, "message": "Test save successful"}

    except Exception as e:
        logger.error(f"❌ Test save failed: {str(e)}")
        return {"success": False, "error": str(e), "provider": "google_drive"}


@router.post("/save")
async def save_cv_to_google_drive(
    cv_data: CompleteCV,
    session: dict = Depends(get_current_session),
):
    """Save a CV to Google Drive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            raise HTTPException(
                status_code=403,
                detail="No Google Drive connection found. Please connect your Google Drive account first.",
            )

        logger.info(f"💾 Saving CV to Google Drive: {cv_data.title}")

        # Ensure token is valid (refresh if needed)
        valid_tokens = await google_drive_service.ensure_valid_token(
            google_drive_tokens
        )

        # Update session if tokens were refreshed
        if valid_tokens != google_drive_tokens:
            cloud_tokens["google_drive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Save CV to Google Drive
        file_id = await google_drive_service.save_cv(valid_tokens, cv_data)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_saved",
            {"provider": "google_drive", "file_id": file_id},
        )

        # IMPORTANT: Return the exact format that frontend expects
        return {
            "success": True,
            "provider": "google_drive",
            "file_id": file_id,  # This must match frontend expectation
            "message": f"CV '{cv_data.title}' saved to Google Drive successfully",
        }

    except GoogleDriveError as e:
        logger.error(f"❌ Google Drive save failed: {str(e)}")
        # Return proper error format that frontend expects
        return {
            "success": False,
            "error": f"Google Drive error: {str(e)}",
            "provider": "google_drive",
        }
    except Exception as e:
        logger.error(f"❌ CV save failed: {str(e)}")
        # Return proper error format that frontend expects
        return {
            "success": False,
            "error": f"Failed to save CV: {str(e)}",
            "provider": "google_drive",
        }


@router.get("/list")
async def list_google_drive_cvs(session: dict = Depends(get_current_session)):
    """List all CVs from Google Drive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            raise HTTPException(
                status_code=403, detail="No Google Drive connection found"
            )

        # Ensure token is valid
        valid_tokens = await google_drive_service.ensure_valid_token(
            google_drive_tokens
        )

        # Update session if tokens were refreshed
        if valid_tokens != google_drive_tokens:
            cloud_tokens["google_drive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # List CVs
        files = await google_drive_service.list_cvs(valid_tokens)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_list",
            {"provider": "google_drive", "count": len(files)},
        )

        return {
            "provider": "google_drive",
            "files": [file.dict() for file in files],
            "count": len(files),
        }

    except GoogleDriveError as e:
        logger.error(f"❌ Google Drive list failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Google Drive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV list failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list CVs: {str(e)}")


@router.get("/load/{file_id}")
async def load_cv_from_google_drive(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Load a specific CV from Google Drive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            raise HTTPException(
                status_code=403, detail="No Google Drive connection found"
            )

        # Ensure token is valid
        valid_tokens = await google_drive_service.ensure_valid_token(
            google_drive_tokens
        )

        # Update session if tokens were refreshed
        if valid_tokens != google_drive_tokens:
            cloud_tokens["google_drive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Load CV from Google Drive
        cv_data = await google_drive_service.load_cv(valid_tokens, file_id)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_loaded",
            {"provider": "google_drive", "file_id": file_id},
        )

        # Convert to response format
        response_data = cv_data.dict()
        response_data["id"] = file_id

        return {"success": True, "provider": "google_drive", "cv_data": response_data}

    except GoogleDriveError as e:
        logger.error(f"❌ Google Drive load failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Google Drive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV load failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load CV: {str(e)}")


@router.delete("/delete/{file_id}")
async def delete_cv_from_google_drive(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Delete a CV from Google Drive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            raise HTTPException(
                status_code=403, detail="No Google Drive connection found"
            )

        # Ensure token is valid
        valid_tokens = await google_drive_service.ensure_valid_token(
            google_drive_tokens
        )

        # Update session if tokens were refreshed
        if valid_tokens != google_drive_tokens:
            cloud_tokens["google_drive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Delete CV from Google Drive
        success = await google_drive_service.delete_cv(valid_tokens, file_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="CV not found or could not be deleted"
            )

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_deleted",
            {"provider": "google_drive", "file_id": file_id},
        )

        return {
            "success": True,
            "message": "CV deleted successfully",
            "provider": "google_drive",
            "file_id": file_id,
        }

    except GoogleDriveError as e:
        logger.error(f"❌ Google Drive delete failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Google Drive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete CV: {str(e)}")


@router.post("/disconnect")
async def disconnect_google_drive(session: dict = Depends(get_current_session)):
    """Disconnect from Google Drive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if "google_drive" not in cloud_tokens:
            raise HTTPException(
                status_code=404, detail="No Google Drive connection found"
            )

        # Remove Google Drive tokens
        del cloud_tokens["google_drive"]

        # Update session
        await session_manager.update_session_cloud_tokens(
            session["session_id"], cloud_tokens
        )

        # Record activity
        await record_session_activity(
            session["session_id"], "google_drive_disconnected", {}
        )

        return {
            "success": True,
            "message": "Disconnected from Google Drive",
            "provider": "google_drive",
        }

    except Exception as e:
        logger.error(f"❌ Google Drive disconnection failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to disconnect from Google Drive: {str(e)}"
        )


@router.get("/debug")
async def debug_google_drive_session(session: dict = Depends(get_current_session)):
    """Debug endpoint for Google Drive session info"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive", {})

        return {
            "session_id": session.get("session_id"),
            "has_google_drive_tokens": "google_drive" in cloud_tokens,
            "token_keys": list(google_drive_tokens.keys())
            if google_drive_tokens
            else [],
            "has_access_token": bool(google_drive_tokens.get("access_token")),
            "has_refresh_token": bool(google_drive_tokens.get("refresh_token")),
            "expires_at": google_drive_tokens.get("expires_at"),
            "email": google_drive_tokens.get("email"),
            "provider_count": len(cloud_tokens),
        }

    except Exception as e:
        logger.error(f"❌ Debug info failed: {str(e)}")
        return {"error": str(e), "session_id": session.get("session_id", "unknown")}
