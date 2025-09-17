# Create a new file: app/api/session.py
"""
Session management API endpoints - MISSING FROM BACKEND
This file was missing and causing the 404 errors in the frontend
"""

import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Depends
from datetime import datetime

from ..auth.sessions import (
    get_current_session,
    get_optional_session,
    session_manager,
    create_anonymous_session,
    record_session_activity,
)
from ..schemas import CloudProvider

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/session")
async def create_or_get_session(request: Request):
    """
    Create a new anonymous session or return existing session info
    This endpoint was missing and causing frontend initialization errors
    """
    try:
        # Try to get existing session from headers
        auth_header = request.headers.get("Authorization")

        if auth_header and auth_header.startswith("Bearer "):
            # Try to validate existing session
            try:
                token = auth_header.replace("Bearer ", "")
                session_data = session_manager.decode_session_token(token)
                session_id = session_data.get("session_id")

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
                            "cloud_providers": list(
                                existing_session.get("cloud_tokens", {}).keys()
                            ),
                        }
            except Exception as e:
                logger.warning(f"Failed to restore existing session: {e}")
                # Continue to create new session

        # Create new session
        logger.info("🔄 Creating new anonymous session...")
        session_data = await create_anonymous_session(request)

        logger.info(f"✅ New session created: {session_data['session_id']}")

        return {
            "session_id": session_data["session_id"],
            "token": session_data["token"],
            "expires_at": session_data["expires_at"],
            "restored": False,
            "cloud_providers": session_data.get("cloud_providers", []),
        }

    except Exception as e:
        logger.error(f"❌ Session creation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )


@router.get("/session/status")
async def get_session_status(session: dict = Depends(get_current_session)):
    """
    Get current session status and information
    """
    try:
        cloud_tokens = session.get("cloud_tokens", {})

        return {
            "session_id": session.get("session_id"),
            "active": True,
            "expires_at": session.get("expires_at"),
            "created_at": session.get("created_at"),
            "last_activity": session.get("last_activity"),
            "cloud_providers": list(cloud_tokens.keys()),
            "cloud_provider_count": len(cloud_tokens),
        }

    except Exception as e:
        logger.error(f"Session status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session status")


@router.post("/session/activity")
async def record_activity(
    activity_data: dict, session: dict = Depends(get_current_session)
):
    """
    Record session activity for analytics
    """
    try:
        activity_type = activity_data.get("type", "unknown")
        metadata = activity_data.get("metadata", {})

        await record_session_activity(session["session_id"], activity_type, metadata)

        return {"message": "Activity recorded", "type": activity_type}

    except Exception as e:
        logger.error(f"Activity recording error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record activity")


@router.delete("/session")
async def delete_session(session: dict = Depends(get_current_session)):
    """
    Delete/revoke current session
    """
    try:
        session_id = session["session_id"]

        # Revoke the session
        success = await session_manager.revoke_session(session_id)

        if success:
            # Record the session deletion
            await record_session_activity(
                session_id, "session_deleted", {"manual": True}
            )

            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except Exception as e:
        logger.error(f"Session deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.post("/session/extend")
async def extend_session(session: dict = Depends(get_current_session)):
    """
    Extend current session expiration
    """
    try:
        session_id = session["session_id"]

        # Create a new token with extended expiration
        from datetime import timedelta

        extended_session_data = {
            "session_id": session_id,
            "expires_at": (
                datetime.utcnow() + timedelta(hours=session_manager.expire_hours)
            ).isoformat(),
            "cloud_providers": list(session.get("cloud_tokens", {}).keys()),
        }

        new_token = session_manager.create_session_token(extended_session_data)

        await record_session_activity(
            session_id,
            "session_extended",
            {"new_expiry": extended_session_data["expires_at"]},
        )

        return {
            "message": "Session extended successfully",
            "new_token": new_token,
            "expires_at": extended_session_data["expires_at"],
        }

    except Exception as e:
        logger.error(f"Session extension error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to extend session")


@router.get("/session/debug")
async def debug_session_info(
    request: Request, session: dict = Depends(get_optional_session)
):
    """
    Debug endpoint to inspect session information (development only)
    """
    try:
        # Get request info
        headers = dict(request.headers)
        auth_header = headers.get("authorization", "Not provided")

        # Session info
        session_info = {
            "has_session": session is not None,
            "session_id": session.get("session_id") if session else None,
            "cloud_providers": list(session.get("cloud_tokens", {}).keys())
            if session
            else [],
            "expires_at": session.get("expires_at") if session else None,
        }

        # Request info
        request_info = {
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host if request.client else "unknown",
            "user_agent": headers.get("user-agent", "unknown"),
            "has_auth_header": "authorization" in headers,
            "auth_header_preview": auth_header[:50] + "..."
            if len(auth_header) > 50
            else auth_header,
        }

        return {
            "session": session_info,
            "request": request_info,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Debug info error: {str(e)}")
        return {
            "error": str(e),
            "session": None,
            "request": None,
            "timestamp": datetime.utcnow().isoformat(),
        }
