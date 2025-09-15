# app/api/cover_letter.py
"""
AI-powered cover letter generation API endpoints
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime

import openai
from anthropic import Anthropic

from ..config import settings, AIConfig
from ..schemas import (
    CompleteCV,
    CoverLetterBase,
    CoverLetterGenerationRequest,
    CoverLetterResponse,
    CloudProvider,
)
from ..auth.sessions import get_current_session, record_session_activity
from ..cloud.service import cloud_service, CloudProviderError
from .ai_enhance import ai_service

logger = logging.getLogger(__name__)
router = APIRouter()


class CoverLetterService:
    """AI service for cover letter generation"""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None

        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.openai_client = openai

        if settings.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)

    async def generate_cover_letter(
        self,
        cv_data: CompleteCV,
        job_description: str,
        job_title: str,
        company_name: str = "",
        recipient_name: str = "",
        recipient_title: str = "",
    ) -> str:
        """Generate a cover letter based on CV and job description"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        # Extract relevant information from CV
        personal_info = cv_data.personal_info
        experiences = cv_data.experiences[:3]  # Limit to recent experiences
        skills = [skill.name for skill in cv_data.skills[:10]]  # Limit skills
        educations = cv_data.educations[:2]  # Limit education entries

        # Build CV summary for context
        cv_context = {
            "name": personal_info.full_name if personal_info else "Applicant",
            "title": personal_info.title if personal_info else "",
            "summary": personal_info.summary if personal_info else "",
            "experiences": [
                {
                    "position": exp.position,
                    "company": exp.company,
                    "description": exp.description[:200] if exp.description else "",
                }
                for exp in experiences
            ],
            "skills": skills,
            "education": [
                {
                    "degree": edu.degree,
                    "field": edu.field_of_study,
                    "institution": edu.institution,
                }
                for edu in educations
            ],
        }

        # Format recipient information
        recipient_info = ""
        if recipient_name and recipient_title:
            recipient_info = f"Dear {recipient_name}, {recipient_title},"
        elif recipient_name:
            recipient_info = f"Dear {recipient_name},"
        else:
            recipient_info = "Dear Hiring Manager,"

        # Create the prompt
        prompt = f"""
        Write a professional cover letter for this job application:

        JOB DETAILS:
        Position: {job_title}
        Company: {company_name}
        Job Description: {job_description[:1000]}

        APPLICANT PROFILE:
        Name: {cv_context["name"]}
        Current Title: {cv_context["title"]}
        Professional Summary: {cv_context["summary"]}
        
        Recent Experience:
        {self._format_experiences(cv_context["experiences"])}
        
        Key Skills: {", ".join(cv_context["skills"])}
        
        Education: {self._format_education(cv_context["education"])}

        REQUIREMENTS:
        - Start with: "{recipient_info}"
        - Write a compelling 3-4 paragraph cover letter
        - Highlight relevant experience and skills from the CV
        - Show enthusiasm for the specific role and company
        - Demonstrate how the applicant's background aligns with job requirements
        - Use professional tone but show personality
        - End with appropriate closing
        - Keep it concise but impactful
        - Make it ATS-friendly

        Cover Letter:
        """

        try:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=settings.ai_model_premium,  # Use better model for cover letters
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional career coach and expert cover letter writer. Create compelling, personalized cover letters that help candidates stand out while being ATS-friendly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,  # Cover letters can be longer
                temperature=0.7,  # Balanced creativity and consistency
            )

            cover_letter = response.choices[0].message.content.strip()

            return cover_letter

        except Exception as e:
            logger.error(f"Cover letter generation failed: {e}")
            raise HTTPException(
                status_code=500, detail="Cover letter generation failed"
            )

    def _format_experiences(self, experiences: List[Dict]) -> str:
        """Format experience list for prompt"""
        formatted = []
        for exp in experiences:
            desc = (
                exp.get("description", "")[:150] + "..."
                if len(exp.get("description", "")) > 150
                else exp.get("description", "")
            )
            formatted.append(f"- {exp['position']} at {exp['company']}: {desc}")
        return "\n".join(formatted)

    def _format_education(self, educations: List[Dict]) -> str:
        """Format education list for prompt"""
        formatted = []
        for edu in educations:
            formatted.append(
                f"{edu['degree']} in {edu['field']} from {edu['institution']}"
            )
        return ", ".join(formatted)

    async def customize_cover_letter(
        self, original_letter: str, job_description: str, customization_request: str
    ) -> str:
        """Customize an existing cover letter based on specific requirements"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        prompt = f"""
        Customize this cover letter based on the specific requirements:

        ORIGINAL COVER LETTER:
        {original_letter}

        JOB DESCRIPTION:
        {job_description[:800]}

        CUSTOMIZATION REQUEST:
        {customization_request}

        REQUIREMENTS:
        - Maintain the professional tone and structure
        - Incorporate the requested changes while keeping the letter coherent
        - Ensure all claims remain truthful and backed by the original content
        - Keep the same approximate length
        - Maintain ATS-friendliness

        Customized Cover Letter:
        """

        try:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=settings.ai_model_primary,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional editor specializing in cover letter customization.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.5,  # Lower temperature for customization
            )

            customized_letter = response.choices[0].message.content.strip()

            return customized_letter

        except Exception as e:
            logger.error(f"Cover letter customization failed: {e}")
            raise HTTPException(
                status_code=500, detail="Cover letter customization failed"
            )

    def _prepare_cover_letter_for_storage(
        self, cover_letter_data: CoverLetterBase
    ) -> str:
        """Prepare cover letter data for cloud storage"""
        from ..schemas import CVFileMetadata

        storage_data = {
            "metadata": CVFileMetadata(
                version="1.0",
                created_at=datetime.utcnow(),
                last_modified=datetime.utcnow(),
                created_with="cv-privacy-platform",
            ).dict(),
            "cover_letter_data": cover_letter_data.dict(),
        }
        return json.dumps(storage_data, indent=2, default=str)

    def _parse_cover_letter_from_storage(self, content: str) -> CoverLetterBase:
        """Parse cover letter data from cloud storage"""
        import json

        try:
            data = json.loads(content)
            letter_data = data.get("cover_letter_data", data)
            return CoverLetterBase.parse_obj(letter_data)
        except Exception as e:
            logger.error(f"Cover letter parsing failed: {e}")
            raise ValueError(f"Invalid cover letter file format: {e}")


# Global cover letter service instance
cover_letter_service = CoverLetterService()


@router.post("/generate", response_model=CoverLetterResponse)
async def generate_cover_letter(
    request: CoverLetterGenerationRequest,
    session: dict = Depends(get_current_session),
):
    """Generate a new cover letter based on CV and job description"""

    try:
        # Check AI usage limits
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI usage limit reached. Used {usage_stats['used_today']}/{usage_stats['limit']} operations.",
            )

        # Load CV from cloud storage
        cloud_tokens = session.get("cloud_tokens", {})
        if not cloud_tokens:
            raise HTTPException(status_code=403, detail="No cloud providers connected")

        # Try to load CV from any connected provider
        cv_data = None
        used_provider = None
        for provider_name, tokens in cloud_tokens.items():
            try:
                provider = CloudProvider(provider_name)
                cv_data = await cloud_service.load_cv(
                    cloud_tokens, provider, request.cv_file_id
                )
                used_provider = provider
                break
            except:
                continue

        if not cv_data:
            raise HTTPException(
                status_code=404,
                detail="CV file not found in any connected cloud storage",
            )

        # Generate cover letter
        cover_letter_content = await cover_letter_service.generate_cover_letter(
            cv_data=cv_data,
            job_description=request.job_description,
            job_title=request.job_title or "Position",
            company_name=request.company_name or "",
            recipient_name=request.recipient_name or "",
            recipient_title=request.recipient_title or "",
        )

        # Create cover letter object
        cover_letter = CoverLetterBase(
            title=f"Cover Letter - {request.job_title or 'Position'} at {request.company_name or 'Company'}",
            company_name=request.company_name,
            job_title=request.job_title,
            job_description=request.job_description[:500] + "..."
            if len(request.job_description) > 500
            else request.job_description,
            recipient_name=request.recipient_name,
            recipient_title=request.recipient_title,
            cover_letter_content=cover_letter_content,
            is_favorite=False,
        )

        # Save to cloud storage
        import json

        filename = f"cover_letter_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        file_content = cover_letter_service._prepare_cover_letter_for_storage(
            cover_letter
        )

        file_id = await cloud_service.save_cv(
            cloud_tokens,
            used_provider,
            CompleteCV(
                title=cover_letter.title,
                customization=cv_data.customization,
                personal_info=cv_data.personal_info,
            ),
            filename,
        )

        # Track usage
        await ai_service._track_ai_usage(
            session["session_id"], "cover_letter_generation"
        )
        await record_session_activity(
            session["session_id"],
            "cover_letter_generated",
            {"provider": used_provider.value, "job_title": request.job_title},
        )

        # Return response
        response_data = cover_letter.dict()
        response_data.update(
            {
                "id": file_id,
                "cv_file_id": request.cv_file_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )

        return CoverLetterResponse(**response_data)

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Cover letter generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Cover letter generation failed")


@router.post("/customize")
async def customize_existing_cover_letter(
    cover_letter_id: str = Query(..., description="Cover letter file ID"),
    customization_request: str = Query(
        ..., description="Specific customization requirements"
    ),
    job_description: str = Query(..., description="Updated job description"),
    provider: CloudProvider = Query(..., description="Cloud provider"),
    session: dict = Depends(get_current_session),
):
    """Customize an existing cover letter"""

    try:
        # Check AI usage limits
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI usage limit reached. Used {usage_stats['used_today']}/{usage_stats['limit']} operations.",
            )

        # Load cover letter from cloud storage
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Load the cover letter file
        cover_letter_content = await cloud_service.download_file(
            cloud_tokens, provider, cover_letter_id
        )
        original_letter = cover_letter_service._parse_cover_letter_from_storage(
            cover_letter_content
        )

        # Customize the cover letter
        customized_content = await cover_letter_service.customize_cover_letter(
            original_letter.cover_letter_content, job_description, customization_request
        )

        # Update cover letter object
        original_letter.cover_letter_content = customized_content
        original_letter.job_description = (
            job_description[:500] + "..."
            if len(job_description) > 500
            else job_description
        )

        # Save updated version
        updated_content = cover_letter_service._prepare_cover_letter_for_storage(
            original_letter
        )
        new_filename = f"cover_letter_customized_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        new_file_id = await cloud_service.upload_file(
            cloud_tokens, provider, new_filename, updated_content
        )

        # Track usage
        await ai_service._track_ai_usage(
            session["session_id"], "cover_letter_customization"
        )
        await record_session_activity(
            session["session_id"],
            "cover_letter_customized",
            {"provider": provider.value},
        )

        return {
            "original_cover_letter_id": cover_letter_id,
            "customized_cover_letter_id": new_file_id,
            "customization_applied": customization_request,
            "usage": await ai_service._check_daily_usage(session["session_id"]),
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Cover letter customization error: {str(e)}")
        raise HTTPException(status_code=500, detail="Cover letter customization failed")


@router.get("/")
async def list_cover_letters(
    provider: Optional[CloudProvider] = Query(
        None, description="Specific cloud provider"
    ),
    session: dict = Depends(get_current_session),
):
    """List all cover letters from cloud storage"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if provider:
            # List from specific provider
            if provider.value not in cloud_tokens:
                raise HTTPException(
                    status_code=403, detail=f"No {provider.value} connection found"
                )

            files = await cloud_service.list_files(
                cloud_tokens, provider, folder_name="Cover Letters"
            )
        else:
            # List from all connected providers
            all_files = []
            for provider_name in cloud_tokens.keys():
                try:
                    cloud_provider = CloudProvider(provider_name)
                    files = await cloud_service.list_files(
                        cloud_tokens, cloud_provider, folder_name="Cover Letters"
                    )
                    all_files.extend(files)
                except Exception as e:
                    logger.warning(
                        f"Failed to list cover letters from {provider_name}: {e}"
                    )
                    continue

            files = sorted(all_files, key=lambda x: x.last_modified, reverse=True)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cover_letters_listed",
            {"provider": provider.value if provider else "all"},
        )

        return {"cover_letters": files, "total_count": len(files)}

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Cover letter listing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list cover letters")


@router.get("/{cover_letter_id}")
async def get_cover_letter(
    cover_letter_id: str,
    provider: CloudProvider = Query(..., description="Cloud provider"),
    session: dict = Depends(get_current_session),
):
    """Get a specific cover letter from cloud storage"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Load cover letter content
        content = await cloud_service.download_file(
            cloud_tokens, provider, cover_letter_id
        )
        cover_letter = cover_letter_service._parse_cover_letter_from_storage(content)

        # Record activity
        await record_session_activity(
            session["session_id"], "cover_letter_viewed", {"provider": provider.value}
        )

        # Return with metadata
        response_data = cover_letter.dict()
        response_data.update({"id": cover_letter_id, "provider": provider.value})

        return response_data

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Cover letter retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cover letter")


@router.delete("/{cover_letter_id}")
async def delete_cover_letter(
    cover_letter_id: str,
    provider: CloudProvider = Query(..., description="Cloud provider"),
    session: dict = Depends(get_current_session),
):
    """Delete a cover letter from cloud storage"""

    try:
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        # Delete from cloud
        success = await cloud_service.delete_file(
            cloud_tokens, provider, cover_letter_id
        )

        if not success:
            raise HTTPException(
                status_code=404, detail="Cover letter not found or could not be deleted"
            )

        # Record activity
        await record_session_activity(
            session["session_id"], "cover_letter_deleted", {"provider": provider.value}
        )

        return {
            "message": "Cover letter deleted successfully",
            "cover_letter_id": cover_letter_id,
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Cover letter deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete cover letter")
