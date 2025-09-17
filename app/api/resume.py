# app/api/resume.py
"""
Resume management API endpoints for privacy-first CV platform
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime

from ..schemas import (
    CompleteCV,
    ResumeResponse,
    CloudProvider,
    CloudFileMetadata,
    ErrorResponse,
)
from ..auth.sessions import get_current_session, record_session_activity
from ..cloud.service import cloud_service, CloudProviderError

logger = logging.getLogger(__name__)
router = APIRouter()


# In resume.py - Add debug logging
@router.post("/", response_model=ResumeResponse)
async def create_resume(
    cv_data: CompleteCV,
    provider: CloudProvider = Query(..., description="Cloud provider to save to"),
    session: dict = Depends(get_current_session),
):
    """Create a new resume and save to cloud storage"""

    logger.info(f"📝 Creating resume for provider: {provider.value}")
    logger.info(f"📝 Session ID: {session.get('session_id')}")
    logger.info(f"📝 CV Title: {cv_data.title}")

    try:
        # Check if user has connected the specified provider
        cloud_tokens = session.get("cloud_tokens", {})
        logger.info(f"📝 Available cloud tokens: {list(cloud_tokens.keys())}")

        if provider.value not in cloud_tokens:
            logger.error(f"❌ Provider {provider.value} not found in session tokens")
            raise HTTPException(
                status_code=403,
                detail=f"No {provider.value} connection found.",
            )

        # Save CV to cloud
        logger.info(f"💾 Saving CV to {provider.value}...")
        file_id = await cloud_service.save_cv(cloud_tokens, provider, cv_data)
        logger.info(f"✅ CV saved with file ID: {file_id}")

        # Record activity
        await record_session_activity(
            session["session_id"], "cv_created", {"provider": provider.value}
        )

        # Convert to response format with generated IDs
        response_data = cv_data.dict()
        response_data["id"] = file_id

        # Add IDs to all sub-components for frontend compatibility
        if response_data.get("personal_info"):
            response_data["personal_info"]["id"] = "personal_info"

        if response_data.get("customization"):
            response_data["customization"]["id"] = "customization"

        if response_data.get("photo"):
            response_data["photos"] = {**response_data["photo"], "id": "photo"}
            del response_data["photo"]

        # Add IDs to list items
        for section_name in [
            "educations",
            "experiences",
            "skills",
            "languages",
            "referrals",
            "custom_sections",
            "extracurriculars",
            "hobbies",
            "courses",
            "internships",
        ]:
            section_data = response_data.get(section_name, [])
            for i, item in enumerate(section_data):
                item["id"] = str(i)

        return ResumeResponse.parse_obj(response_data)

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resume")


@router.put("/{file_id}", response_model=ResumeResponse)
async def update_resume(
    file_id: str,
    cv_data: CompleteCV,
    provider: CloudProvider = Query(..., description="Cloud provider to update"),
    session: dict = Depends(get_current_session),
):
    """Update an existing resume in cloud storage"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Save updated CV to cloud (this will overwrite the existing file)
        updated_file_id = await cloud_service.save_cv(
            cloud_tokens,
            provider,
            cv_data,
            file_name=None,  # Will generate new filename with timestamp
        )

        # Record activity
        await record_session_activity(
            session["session_id"], "cv_updated", {"provider": provider.value}
        )

        # Convert to response format
        response_data = cv_data.dict()
        response_data["id"] = updated_file_id

        # Add IDs for frontend compatibility
        if response_data.get("personal_info"):
            response_data["personal_info"]["id"] = "personal_info"

        if response_data.get("customization"):
            response_data["customization"]["id"] = "customization"

        if response_data.get("photo"):
            response_data["photos"] = {**response_data["photo"], "id": "photo"}
            del response_data["photo"]

        # Add IDs to list items
        for section_name in [
            "educations",
            "experiences",
            "skills",
            "languages",
            "referrals",
            "custom_sections",
            "extracurriculars",
            "hobbies",
            "courses",
            "internships",
        ]:
            section_data = response_data.get(section_name, [])
            for i, item in enumerate(section_data):
                item["id"] = str(i)

        return ResumeResponse.parse_obj(response_data)

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update resume")


@router.delete("/{file_id}")
async def delete_resume(
    file_id: str,
    provider: CloudProvider = Query(..., description="Cloud provider to delete from"),
    session: dict = Depends(get_current_session),
):
    """Delete a resume from cloud storage"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Delete from cloud
        success = await cloud_service.delete_cv(cloud_tokens, provider, file_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="Resume not found or could not be deleted"
            )

        # Record activity
        await record_session_activity(
            session["session_id"], "cv_deleted", {"provider": provider.value}
        )

        return {"message": "Resume deleted successfully", "file_id": file_id}

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete resume")


@router.post("/{file_id}/backup")
async def backup_resume(
    file_id: str,
    source_provider: CloudProvider = Query(..., description="Source cloud provider"),
    backup_providers: List[CloudProvider] = Query(
        ..., description="Target backup providers"
    ),
    session: dict = Depends(get_current_session),
):
    """Backup resume to multiple cloud providers"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        # Check source provider connection
        if source_provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {source_provider.value} connection found"
            )

        # Check backup provider connections
        for provider in backup_providers:
            if provider.value not in cloud_tokens:
                raise HTTPException(
                    status_code=403,
                    detail=f"No {provider.value} connection found for backup",
                )

        # Perform backup
        backup_results = await cloud_service.backup_cv(
            cloud_tokens, source_provider, file_id, backup_providers
        )

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_backup",
            {
                "source_provider": source_provider.value,
                "backup_providers": [p.value for p in backup_providers],
            },
        )

        # Format results
        results = {}
        for provider, result_file_id in backup_results.items():
            results[provider.value] = {
                "success": result_file_id is not None,
                "file_id": result_file_id,
            }

        return {
            "message": "Backup completed",
            "source_file_id": file_id,
            "backup_results": results,
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume backup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to backup resume")


@router.get("/search/")
async def search_resumes(
    q: str = Query(..., description="Search term", min_length=1),
    providers: Optional[List[CloudProvider]] = Query(
        None, description="Providers to search"
    ),
    session: dict = Depends(get_current_session),
):
    """Search for resumes across cloud providers"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if not cloud_tokens:
            raise HTTPException(status_code=403, detail="No cloud providers connected")

        # Search across providers
        search_results = await cloud_service.search_cvs(cloud_tokens, q, providers)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_search",
            {"search_term": q[:50]},  # Limit for privacy
        )

        return {
            "search_term": q,
            "results_count": len(search_results),
            "results": search_results,
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search resumes")


@router.post("/{file_id}/sync")
async def sync_resume(
    file_id: str,
    source_provider: CloudProvider = Query(..., description="Source provider"),
    target_providers: List[CloudProvider] = Query(
        ..., description="Target providers to sync to"
    ),
    session: dict = Depends(get_current_session),
):
    """Sync resume across multiple cloud providers"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        # Check all provider connections
        all_providers = [source_provider] + target_providers
        for provider in all_providers:
            if provider.value not in cloud_tokens:
                raise HTTPException(
                    status_code=403, detail=f"No {provider.value} connection found"
                )

        # Load CV from source
        cv_data = await cloud_service.load_cv(cloud_tokens, source_provider, file_id)

        # Sync to target providers
        sync_results = await cloud_service.sync_cv_across_providers(
            cloud_tokens, cv_data, target_providers
        )

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_sync",
            {
                "source_provider": source_provider.value,
                "target_providers": [p.value for p in target_providers],
            },
        )

        # Format results
        results = {}
        for provider, result_file_id in sync_results.items():
            results[provider.value] = {
                "success": result_file_id is not None,
                "file_id": result_file_id,
            }

        return {
            "message": "Sync completed",
            "source_file_id": file_id,
            "sync_results": results,
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume sync error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to sync resume")


@router.get("/{file_id}/public")
async def get_public_resume(
    file_id: str,
    provider: CloudProvider = Query(..., description="Cloud provider"),
    session: dict = Depends(get_current_session),
):
    """Get public version of resume (for sharing)"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Load CV
        cv_data = await cloud_service.load_cv(cloud_tokens, provider, file_id)

        # Check if CV is marked as public
        if not cv_data.is_public:
            raise HTTPException(
                status_code=403, detail="Resume is not marked as public"
            )

        # Record activity
        await record_session_activity(
            session["session_id"], "cv_public_access", {"provider": provider.value}
        )

        # Return CV data without sensitive information
        return cv_data

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Public resume access error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to access public resume")


@router.patch("/{file_id}")
async def patch_resume(
    file_id: str,
    updates: dict,
    provider: CloudProvider = Query(..., description="Cloud provider"),
    session: dict = Depends(get_current_session),
):
    """Partially update a resume (for frontend compatibility)"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Load existing CV
        cv_data = await cloud_service.load_cv(cloud_tokens, provider, file_id)

        # Apply updates
        cv_dict = cv_data.dict()
        for key, value in updates.items():
            if key in cv_dict:
                cv_dict[key] = value

        # Validate and save updated CV
        updated_cv = CompleteCV.parse_obj(cv_dict)
        new_file_id = await cloud_service.save_cv(cloud_tokens, provider, updated_cv)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_patched",
            {"provider": provider.value, "fields_updated": list(updates.keys())},
        )

        return {
            "message": "Resume updated successfully",
            "file_id": new_file_id,
            "updated_fields": list(updates.keys()),
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume patch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to patch resume")


@router.get("/{file_id}/versions")
async def get_resume_versions(
    file_id: str,
    provider: CloudProvider = Query(..., description="Cloud provider"),
    session: dict = Depends(get_current_session),
):
    """Get version history of a resume (if supported by provider)"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # This is a placeholder - actual implementation would depend on
        # cloud provider's version history capabilities
        return {
            "message": "Version history not yet implemented",
            "file_id": file_id,
            "provider": provider.value,
            "versions": [],
        }

    except Exception as e:
        logger.error(f"Version history error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get version history")


@router.post("/{file_id}/duplicate")
async def duplicate_resume(
    file_id: str,
    new_title: str,
    source_provider: CloudProvider = Query(..., description="Source provider"),
    target_provider: CloudProvider = Query(
        None, description="Target provider (defaults to source)"
    ),
    session: dict = Depends(get_current_session),
):
    """Create a duplicate of an existing resume"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        target_provider = target_provider or source_provider

        # Check provider connections
        for provider in [source_provider, target_provider]:
            if provider.value not in cloud_tokens:
                raise HTTPException(
                    status_code=403, detail=f"No {provider.value} connection found"
                )

        # Load source CV
        cv_data = await cloud_service.load_cv(cloud_tokens, source_provider, file_id)

        # Update title for duplicate
        cv_data.title = new_title

        # Save duplicate
        duplicate_file_id = await cloud_service.save_cv(
            cloud_tokens, target_provider, cv_data
        )

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_duplicated",
            {
                "source_provider": source_provider.value,
                "target_provider": target_provider.value,
            },
        )

        return {
            "message": "Resume duplicated successfully",
            "original_file_id": file_id,
            "duplicate_file_id": duplicate_file_id,
            "new_title": new_title,
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume duplication error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to duplicate resume")


@router.get("/", response_model=List[CloudFileMetadata])
async def list_resumes(
    provider: Optional[CloudProvider] = Query(
        None, description="Specific cloud provider to list from"
    ),
    session: dict = Depends(get_current_session),
):
    """List all resumes from cloud storage"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if provider:
            # List from specific provider
            if provider.value not in cloud_tokens:
                raise HTTPException(
                    status_code=403, detail=f"No {provider.value} connection found"
                )

            files = await cloud_service.list_cvs(cloud_tokens, provider)
        else:
            # List from all connected providers
            all_files = []
            for provider_name in cloud_tokens.keys():
                try:
                    cloud_provider = CloudProvider(provider_name)
                    files = await cloud_service.list_cvs(cloud_tokens, cloud_provider)
                    all_files.extend(files)
                except Exception as e:
                    logger.warning(f"Failed to list files from {provider_name}: {e}")
                    continue

            # Sort by last modified
            files = sorted(all_files, key=lambda x: x.last_modified, reverse=True)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_listed",
            {"provider": provider.value if provider else "all"},
        )

        return files

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume listing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list resumes")


@router.get("/{file_id}", response_model=ResumeResponse)
async def get_resume(
    file_id: str,
    provider: CloudProvider = Query(..., description="Cloud provider to load from"),
    session: dict = Depends(get_current_session),
):
    """Get a specific resume from cloud storage"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Load CV from cloud
        cv_data = await cloud_service.load_cv(cloud_tokens, provider, file_id)

        # Record activity
        await record_session_activity(
            session["session_id"], "cv_loaded", {"provider": provider.value}
        )

        # Convert to response format
        response_data = cv_data.dict()
        response_data["id"] = file_id

        # Add IDs for frontend compatibility (same as create_resume)
        if response_data.get("personal_info"):
            response_data["personal_info"]["id"] = "personal_info"

        if response_data.get("customization"):
            response_data["customization"]["id"] = "customization"

        if response_data.get("photo"):
            response_data["photos"] = {**response_data["photo"], "id": "photo"}
            del response_data["photo"]

        # Add IDs to list items
        for section_name in [
            "educations",
            "experiences",
            "skills",
            "languages",
            "referrals",
            "custom_sections",
            "extracurriculars",
            "hobbies",
            "courses",
            "internships",
        ]:
            section_data = response_data.get(section_name, [])
            for i, item in enumerate(section_data):
                item["id"] = str(i)

        return ResumeResponse.parse_obj(response_data)

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Resume retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resume")
