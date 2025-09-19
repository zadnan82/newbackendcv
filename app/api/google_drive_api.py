# app/api/google_drive_api.py - FIXED to handle frontend schema properly
"""
Simplified Google Drive API - Fixed schema handling and error responses
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

# Replace this import:
from datetime import datetime, time

# With this:
import time
from datetime import datetime
import secrets
import json
from pydantic import BaseModel, ValidationError

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

        # Create simple test data that matches the expected schema
        test_data = {
            "title": "Test CV",
            "is_public": False,
            "customization": {
                "template": "stockholm",
                "accent_color": "#1a5276",
                "font_family": "Helvetica, Arial, sans-serif",
                "line_spacing": 1.5,
                "headings_uppercase": False,
                "hide_skill_level": False,
                "language": "en",
            },
            "personal_info": {
                "full_name": "Test User",
                "email": "test@example.com",
                "mobile": "+1234567890",
                "city": "",
                "address": "",
                "postal_code": "",
                "driving_license": "",
                "nationality": "",
                "place_of_birth": "",
                "date_of_birth": "",
                "linkedin": "",
                "website": "",
                "summary": "",
                "title": "",
            },
            "educations": [],
            "experiences": [],
            "skills": [],
            "languages": [],
            "referrals": [],
            "custom_sections": [],
            "extracurriculars": [],
            "hobbies": [],
            "courses": [],
            "internships": [],
            "photo": {"photolink": None},
        }

        # Save test data
        file_id = await google_drive_service.save_cv(google_drive_tokens, test_data)

        return {"success": True, "file_id": file_id, "message": "Test save successful"}

    except Exception as e:
        logger.error(f"❌ Test save failed: {str(e)}")
        return {"success": False, "error": str(e), "provider": "google_drive"}


class CVData(BaseModel):
    title: str
    is_public: bool = False
    customization: Dict[str, Any] = {}
    personal_info: Dict[str, Any] = {}
    educations: list = []
    experiences: list = []
    skills: list = []
    languages: list = []
    referrals: list = []
    custom_sections: list = []
    extracurriculars: list = []
    hobbies: list = []
    courses: list = []
    internships: list = []
    photo: Dict[str, Any] = {}

    class Config:
        extra = "allow"  # Allow additional fields


@router.post("/save")
async def save_cv_to_google_drive(
    cv_data: dict,  # Simple dict - FastAPI will parse JSON automatically
    session: dict = Depends(get_current_session),
):
    """Save a CV to Google Drive - QUICK FIX VERSION"""
    import time

    start_time = time.time()

    try:
        logger.info("🐛 STEP 1: Starting save_cv_to_google_drive function")

        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        logger.info(
            f"🐛 STEP 2: Session data retrieved, has google_drive_tokens: {bool(google_drive_tokens)}"
        )

        if not google_drive_tokens:
            logger.error("❌ No Google Drive connection found")
            return {
                "success": False,
                "error": "No Google Drive connection found. Please connect your Google Drive account first.",
                "provider": "google_drive",
            }

        logger.info("🐛 STEP 3: CV data received via FastAPI automatic parsing")
        logger.info(f"🐛 STEP 4: CV data parsed successfully")
        logger.info(f"   - Title: {cv_data.get('title', 'No title')}")
        logger.info(f"   - Has personal_info: {bool(cv_data.get('personal_info'))}")
        logger.info(
            f"   - Photo field: {cv_data.get('photo', {}).get('photolink') is not None}"
        )
        logger.info(f"   - Data keys: {list(cv_data.keys())}")

        logger.info("🐛 STEP 5: About to validate CV data")

        # Validate that we have minimum required data
        if not cv_data.get("title"):
            logger.error("❌ STEP 5 FAILED: No title provided in CV data")
            return {
                "success": False,
                "error": "CV title is required",
                "provider": "google_drive",
            }

        logger.info("🐛 STEP 6: About to ensure valid token")

        # Ensure token is valid (refresh if needed)
        try:
            valid_tokens = await google_drive_service.ensure_valid_token(
                google_drive_tokens
            )
            logger.info("🐛 STEP 7: Token validation successful")
        except Exception as token_error:
            logger.error(f"❌ STEP 6 FAILED: Token validation failed: {token_error}")
            return {
                "success": False,
                "error": "Google Drive connection expired. Please reconnect.",
                "provider": "google_drive",
            }

        logger.info("🐛 STEP 8: About to update session if needed")

        # Update session if tokens were refreshed
        if valid_tokens != google_drive_tokens:
            cloud_tokens["google_drive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )
            logger.info("🐛 STEP 9: Session updated with new tokens")
        else:
            logger.info("🐛 STEP 9: No session update needed")

        logger.info("🐛 STEP 10: About to call google_drive_service.save_cv")

        # Save CV to Google Drive
        try:
            logger.info(f"💾 Starting Google Drive save for: {cv_data.get('title')}")

            file_id = await google_drive_service.save_cv(valid_tokens, cv_data)

            processing_time = time.time() - start_time
            logger.info(
                f"✅ STEP 11: Google Drive save completed successfully: {file_id} (took {processing_time:.2f}s)"
            )

            # Record activity in background
            try:
                await record_session_activity(
                    session["session_id"],
                    "cv_saved",
                    {"provider": "google_drive", "file_id": file_id},
                )
                logger.info("🐛 STEP 12: Activity recorded successfully")
            except Exception as activity_error:
                logger.warning(
                    f"⚠️ Failed to record activity (non-critical): {activity_error}"
                )

            # Return success response
            response_data = {
                "success": True,
                "provider": "google_drive",
                "file_id": file_id,
                "message": f"CV '{cv_data.get('title')}' saved to Google Drive successfully",
            }
            logger.info("🐛 STEP 13: Returning success response")
            return response_data

        except ValidationError as ve:
            logger.error(f"❌ STEP 10 FAILED - CV validation error during save: {ve}")
            validation_details = []
            for error in ve.errors():
                validation_details.append(
                    f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                )

            return {
                "success": False,
                "error": f"CV data validation failed: {'; '.join(validation_details[:3])}{'...' if len(validation_details) > 3 else ''}",
                "provider": "google_drive",
            }

        except GoogleDriveError as gd_error:
            logger.error(f"❌ STEP 10 FAILED - Google Drive service error: {gd_error}")
            return {
                "success": False,
                "error": f"Google Drive error: {str(gd_error)}",
                "provider": "google_drive",
            }
        except Exception as save_error:
            logger.error(f"❌ STEP 10 FAILED - Unexpected save error: {save_error}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Unexpected error during save: {str(save_error)}",
                "provider": "google_drive",
            }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ OVERALL FAILURE after {processing_time:.2f}s: {str(e)}")
        import traceback

        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Failed to save CV: {str(e)}",
            "provider": "google_drive",
        }


# Add this test endpoint to your google_drive_api.py to verify basic functionality


@router.post("/test-save")
async def test_save_endpoint(
    request: Request,
    session: dict = Depends(get_current_session),
):
    """Test endpoint to verify request handling is working"""
    try:
        logger.info("🧪 TEST: Starting test endpoint")

        # Read body
        body = await request.body()
        logger.info(f"🧪 TEST: Body size: {len(body)} bytes")

        # Parse JSON
        cv_data = json.loads(body.decode("utf-8"))
        logger.info(f"🧪 TEST: JSON parsed, title: {cv_data.get('title')}")

        # Check session
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")
        logger.info(f"🧪 TEST: Has Google Drive tokens: {bool(google_drive_tokens)}")

        return {
            "success": True,
            "message": "Test endpoint working",
            "data_received": {
                "title": cv_data.get("title"),
                "has_personal_info": bool(cv_data.get("personal_info")),
                "sections_count": len(
                    [k for k in cv_data.keys() if isinstance(cv_data.get(k), list)]
                ),
                "body_size": len(body),
            },
        }

    except Exception as e:
        logger.error(f"🧪 TEST FAILED: {str(e)}")
        import traceback

        logger.error(f"🧪 TEST TRACEBACK: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}


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

        # Convert to response format - ensure it matches frontend expectations
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


# Add this to your app/api/google_drive_api.py


@router.get("/debug-download/{file_id}")
async def debug_download_file(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Debug endpoint to see what Google Drive actually returns"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            return {"error": "No Google Drive connection found"}

        access_token = google_drive_tokens["access_token"]

        # Test different download methods
        import aiohttp

        results = {}

        # Method 1: Direct file download with alt=media
        try:
            async with aiohttp.ClientSession() as client:
                url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
                headers = {"Authorization": f"Bearer {access_token}"}

                # Try alt=media
                async with client.get(
                    url, params={"alt": "media"}, headers=headers
                ) as resp:
                    content = await resp.text()
                    results["alt_media"] = {
                        "status": resp.status,
                        "content_type": resp.headers.get("content-type"),
                        "content_length": len(content),
                        "content_preview": content[:200] + "..."
                        if len(content) > 200
                        else content,
                        "is_json": content.strip().startswith("{"),
                    }
        except Exception as e:
            results["alt_media"] = {"error": str(e)}

        # Method 2: Get file metadata
        try:
            async with aiohttp.ClientSession() as client:
                url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
                headers = {"Authorization": f"Bearer {access_token}"}

                async with client.get(url, headers=headers) as resp:
                    metadata = await resp.json()
                    results["metadata"] = {
                        "status": resp.status,
                        "name": metadata.get("name"),
                        "mimeType": metadata.get("mimeType"),
                        "size": metadata.get("size"),
                        "parents": metadata.get("parents"),
                    }
        except Exception as e:
            results["metadata"] = {"error": str(e)}

        # Method 3: Try export instead (for Google Docs format)
        try:
            async with aiohttp.ClientSession() as client:
                url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
                headers = {"Authorization": f"Bearer {access_token}"}

                async with client.get(
                    url, params={"mimeType": "text/plain"}, headers=headers
                ) as resp:
                    content = await resp.text()
                    results["export_text"] = {
                        "status": resp.status,
                        "content_type": resp.headers.get("content-type"),
                        "content_length": len(content),
                        "content_preview": content[:200] + "..."
                        if len(content) > 200
                        else content,
                    }
        except Exception as e:
            results["export_text"] = {"error": str(e)}

        return {
            "file_id": file_id,
            "debug_results": results,
            "recommendation": "Check which method returns valid JSON content",
        }

    except Exception as e:
        logger.error(f"Debug download failed: {str(e)}")
        return {"error": str(e)}


@router.put("/update-file/{file_id}")
async def update_cv_in_google_drive(
    file_id: str,
    cv_data: dict,
    session: dict = Depends(get_current_session),
):
    """Update an existing CV in Google Drive"""
    try:
        logger.info(f"🔄 Updating CV in Google Drive: {file_id}")

        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            return {
                "success": False,
                "error": "No Google Drive connection found",
                "provider": "google_drive",
            }

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

        # Update CV in Google Drive
        success = await google_drive_service.update_cv(valid_tokens, file_id, cv_data)

        if success:
            # Record activity
            await record_session_activity(
                session["session_id"],
                "cv_updated",
                {"provider": "google_drive", "file_id": file_id},
            )

            return {
                "success": True,
                "file_id": file_id,
                "message": f"CV updated successfully",
                "provider": "google_drive",
            }
        else:
            return {
                "success": False,
                "error": "Failed to update CV",
                "provider": "google_drive",
            }

    except Exception as e:
        logger.error(f"❌ Update CV failed: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to update CV: {str(e)}",
            "provider": "google_drive",
        }
