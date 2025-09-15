# app/api/ai_enhance.py
"""
AI-powered CV enhancement API endpoints
FIXED: Modern OpenAI API usage and proper error handling
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime

# FIXED: Use modern OpenAI client
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from ..config import get_settings, AIConfig
from ..schemas import (
    CompleteCV,
    PersonalInfoSummaryRequest,
    ExperienceDescriptionRequest,
    SkillsUpdateRequest,
    JobAnalysisRequest,
    JobAnalysisResponse,
    AIUsageResponse,
    CloudProvider,
)
from ..auth.sessions import get_current_session, record_session_activity
from ..cloud.service import cloud_service, CloudProviderError
from ..database import get_db
from ..models import AIUsageTracking

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter()


class AIService:
    """AI service for CV enhancement and analysis"""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None

        # FIXED: Initialize modern OpenAI client
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        if settings.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

        logger.info(
            f"AI Service initialized - OpenAI: {bool(self.openai_client)}, Anthropic: {bool(self.anthropic_client)}"
        )

    async def _track_ai_usage(
        self, session_id: str, service_type: str, tokens_used: int = None
    ):
        """Track AI usage for rate limiting"""
        db = next(get_db())

        try:
            # Hash session ID for privacy
            import hashlib

            session_hash = hashlib.sha256(
                f"{session_id}_{settings.secret_key}".encode()
            ).hexdigest()[:64]

            usage = AIUsageTracking(
                session_hash=session_hash,
                service_type=service_type,
                tokens_used=tokens_used,
                cost_estimate=self._estimate_cost(tokens_used) if tokens_used else None,
            )

            db.add(usage)
            db.commit()

        except Exception as e:
            logger.error(f"Failed to track AI usage: {e}")
        finally:
            db.close()

    def _estimate_cost(self, tokens: int, model: str = "gpt-3.5-turbo") -> float:
        """Estimate cost based on tokens and model"""
        model_costs = AIConfig.MODELS

        for model_config in model_costs.values():
            if model_config["model"] == model:
                return (tokens / 1000) * model_config["cost_per_1k_tokens"]

        return 0.0

    async def _check_daily_usage(self, session_id: str) -> Dict[str, int]:
        """Check daily AI usage for rate limiting"""
        db = next(get_db())

        try:
            import hashlib
            from datetime import datetime, timedelta

            session_hash = hashlib.sha256(
                f"{session_id}_{settings.secret_key}".encode()
            ).hexdigest()[:64]
            today_start = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            # Count today's usage
            usage_count = (
                db.query(AIUsageTracking)
                .filter(
                    AIUsageTracking.session_hash == session_hash,
                    AIUsageTracking.used_at >= today_start,
                )
                .count()
            )

            return {
                "used_today": usage_count,
                "limit": settings.free_tier_ai_operations,
                "remaining": max(0, settings.free_tier_ai_operations - usage_count),
            }

        except Exception as e:
            logger.error(f"Failed to check daily usage: {e}")
            return {
                "used_today": 0,
                "limit": settings.free_tier_ai_operations,
                "remaining": settings.free_tier_ai_operations,
            }
        finally:
            db.close()

    async def enhance_summary(
        self, summary: str, cv_context: Optional[Dict] = None
    ) -> str:
        """Enhance personal summary using AI"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        prompt = f"""
        Improve this professional summary to be more compelling and ATS-friendly:
        
        Original: {summary}
        
        Requirements:
        - Keep it professional and concise (2-3 sentences)
        - Include relevant keywords for the person's field
        - Highlight key strengths and value proposition
        - Make it ATS-friendly
        - Maintain the person's authentic voice
        
        Enhanced summary:
        """

        try:
            # FIXED: Use modern OpenAI API
            response = await self.openai_client.chat.completions.create(
                model=settings.ai_model_primary,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional CV writer helping to enhance personal summaries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=settings.ai_max_tokens,
                temperature=settings.ai_temperature,
            )

            enhanced_summary = response.choices[0].message.content.strip()

            # Remove any quotes or formatting that might have been added
            enhanced_summary = enhanced_summary.strip('"').strip("'")

            # Track token usage
            tokens_used = response.usage.total_tokens if response.usage else None
            logger.info(f"Summary enhancement used {tokens_used} tokens")

            return enhanced_summary

        except Exception as e:
            logger.error(f"AI summary enhancement failed: {e}")
            raise HTTPException(status_code=500, detail="AI enhancement failed")

    async def enhance_experience_description(
        self, description: str, company: str, position: str
    ) -> str:
        """Enhance work experience description using AI"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        prompt = f"""
        Improve this work experience description to be more impactful and ATS-friendly:
        
        Position: {position} at {company}
        Original description: {description}
        
        Requirements:
        - Use action verbs and quantifiable achievements where possible
        - Make it ATS-friendly with relevant keywords
        - Keep the same length or slightly longer
        - Focus on impact and results
        - Use professional tone
        
        Enhanced description:
        """

        try:
            # FIXED: Use modern OpenAI API
            response = await self.openai_client.chat.completions.create(
                model=settings.ai_model_primary,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional CV writer helping to enhance work experience descriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=settings.ai_max_tokens,
                temperature=settings.ai_temperature,
            )

            enhanced_description = response.choices[0].message.content.strip()

            # Track token usage
            tokens_used = response.usage.total_tokens if response.usage else None
            logger.info(f"Experience enhancement used {tokens_used} tokens")

            return enhanced_description

        except Exception as e:
            logger.error(f"AI experience enhancement failed: {e}")
            raise HTTPException(status_code=500, detail="AI enhancement failed")

    async def suggest_skills(self, cv_data: CompleteCV) -> List[str]:
        """Suggest additional skills based on CV content"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        # Extract context from CV
        experiences = "\n".join(
            [
                f"- {exp.position} at {exp.company}: {exp.description[:200] if exp.description else ''}"
                for exp in cv_data.experiences[:3]  # Limit to prevent token overflow
            ]
        )

        current_skills = ", ".join([skill.name for skill in cv_data.skills])

        prompt = f"""
        Based on this professional profile, suggest 5-8 additional relevant skills:
        
        Current Skills: {current_skills}
        
        Recent Experience:
        {experiences}
        
        Requirements:
        - Suggest skills that are relevant but not already listed
        - Focus on in-demand skills for their field
        - Include both technical and soft skills
        - Make suggestions realistic based on their experience
        
        Return only the skill names, comma-separated:
        """

        try:
            # FIXED: Use modern OpenAI API
            response = await self.openai_client.chat.completions.create(
                model=settings.ai_model_primary,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a career advisor suggesting relevant skills for professionals.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.7,
            )

            suggestions = response.choices[0].message.content.strip()

            # Parse suggestions
            skill_list = [skill.strip() for skill in suggestions.split(",")]

            # Track token usage
            tokens_used = response.usage.total_tokens if response.usage else None
            logger.info(f"Skill suggestions used {tokens_used} tokens")

            return skill_list[:8]  # Limit to 8 suggestions

        except Exception as e:
            logger.error(f"AI skill suggestions failed: {e}")
            raise HTTPException(status_code=500, detail="Skill suggestion failed")

    async def analyze_cv_for_job(
        self, cv_data: CompleteCV, job_description: str, job_title: str
    ) -> JobAnalysisResponse:
        """Analyze CV compatibility with job posting"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        # Prepare CV summary for analysis
        cv_summary = {
            "title": cv_data.personal_info.title
            if cv_data.personal_info
            else "Professional",
            "summary": cv_data.personal_info.summary if cv_data.personal_info else "",
            "skills": [skill.name for skill in cv_data.skills],
            "experiences": [
                {
                    "position": exp.position,
                    "company": exp.company,
                    "description": exp.description[:300] if exp.description else "",
                }
                for exp in cv_data.experiences[:3]
            ],
            "education": [
                {
                    "degree": edu.degree,
                    "field": edu.field_of_study,
                    "institution": edu.institution,
                }
                for edu in cv_data.educations[:2]
            ],
        }

        prompt = f"""
        Analyze this CV against the job posting and provide a detailed compatibility assessment:
        
        JOB POSTING:
        Title: {job_title}
        Description: {job_description[:1500]}
        
        CV SUMMARY:
        Title: {cv_summary["title"]}
        Summary: {cv_summary["summary"]}
        Skills: {", ".join(cv_summary["skills"])}
        Recent Experience: {cv_summary["experiences"]}
        Education: {cv_summary["education"]}
        
        Please provide your analysis in this exact JSON format:
        {{
            "match_score": 75,
            "ats_compatibility_score": 80,
            "keywords_present": ["keyword1", "keyword2"],
            "keywords_missing": ["missing1", "missing2"],
            "recommendations": ["improvement1", "improvement2"],
            "strengths": ["strength1", "strength2"],
            "improvement_areas": ["area1", "area2"],
            "should_apply": true,
            "application_tips": ["tip1", "tip2"]
        }}
        """

        try:
            # FIXED: Use modern OpenAI API
            response = await self.openai_client.chat.completions.create(
                model=settings.ai_model_premium,  # Use better model for analysis
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert recruiter and CV analyst. Provide detailed, actionable feedback in valid JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more consistent output
            )

            analysis_text = response.choices[0].message.content.strip()

            # Track token usage
            tokens_used = response.usage.total_tokens if response.usage else None
            logger.info(f"Job analysis used {tokens_used} tokens")

            # Try to parse JSON response
            import json

            try:
                analysis_data = json.loads(analysis_text)
                return JobAnalysisResponse(**analysis_data)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse AI response as JSON, using fallback")
                return JobAnalysisResponse(
                    match_score=70.0,
                    ats_compatibility_score=75.0,
                    keywords_present=["relevant skills found"],
                    keywords_missing=["additional keywords needed"],
                    recommendations=[
                        "AI analysis completed but response format needs adjustment"
                    ],
                    strengths=["Experience relevant to role"],
                    improvement_areas=["Consider adding more specific keywords"],
                    should_apply=True,
                    application_tips=[
                        "Tailor your CV to include job-specific keywords"
                    ],
                )

        except Exception as e:
            logger.error(f"AI job analysis failed: {e}")
            raise HTTPException(status_code=500, detail="Job analysis failed")


# Global AI service instance
ai_service = AIService()


@router.get("/usage", response_model=AIUsageResponse)
async def get_ai_usage(session: dict = Depends(get_current_session)):
    """Get current AI usage statistics"""

    usage_stats = await ai_service._check_daily_usage(session["session_id"])
    return AIUsageResponse(**usage_stats)


@router.post("/enhance-summary")
async def enhance_personal_summary(
    request: PersonalInfoSummaryRequest,
    cv_file_id: str = Query(..., description="CV file ID for context"),
    provider: CloudProvider = Query(..., description="Cloud provider"),
    session: dict = Depends(get_current_session),
):
    """Enhance personal summary using AI"""

    try:
        # Check usage limits
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI usage limit reached. Used {usage_stats['used_today']}/{usage_stats['limit']} operations.",
            )

        # Load CV for context
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        cv_data = await cloud_service.load_cv(cloud_tokens, provider, cv_file_id)

        # Enhance summary
        enhanced_summary = await ai_service.enhance_summary(
            request.summary, cv_context=cv_data.dict()
        )

        # Track usage
        await ai_service._track_ai_usage(session["session_id"], "summary_enhancement")
        await record_session_activity(
            session["session_id"], "ai_enhance_summary", {"provider": provider.value}
        )

        return {
            "original_summary": request.summary,
            "enhanced_summary": enhanced_summary,
            "usage": await ai_service._check_daily_usage(session["session_id"]),
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Summary enhancement error: {str(e)}")
        raise HTTPException(status_code=500, detail="Summary enhancement failed")


@router.post("/enhance-experience")
async def enhance_experience_description(
    request: ExperienceDescriptionRequest,
    cv_file_id: str = Query(..., description="CV file ID for context"),
    provider: CloudProvider = Query(..., description="Cloud provider"),
    company: str = Query(..., description="Company name"),
    position: str = Query(..., description="Position title"),
    session: dict = Depends(get_current_session),
):
    """Enhance work experience description using AI"""

    try:
        # Check usage limits
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI usage limit reached. Used {usage_stats['used_today']}/{usage_stats['limit']} operations.",
            )

        # Enhance experience description
        enhanced_description = await ai_service.enhance_experience_description(
            request.description, company, position
        )

        # Track usage
        await ai_service._track_ai_usage(
            session["session_id"], "experience_enhancement"
        )
        await record_session_activity(
            session["session_id"], "ai_enhance_experience", {"provider": provider.value}
        )

        return {
            "original_description": request.description,
            "enhanced_description": enhanced_description,
            "usage": await ai_service._check_daily_usage(session["session_id"]),
        }

    except Exception as e:
        logger.error(f"Experience enhancement error: {str(e)}")
        raise HTTPException(status_code=500, detail="Experience enhancement failed")


@router.post("/suggest-skills")
async def suggest_additional_skills(
    cv_file_id: str = Query(..., description="CV file ID"),
    provider: CloudProvider = Query(..., description="Cloud provider"),
    session: dict = Depends(get_current_session),
):
    """Suggest additional skills based on CV content"""

    try:
        # Check usage limits
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI usage limit reached. Used {usage_stats['used_today']}/{usage_stats['limit']} operations.",
            )

        # Load CV
        cloud_tokens = session.get("cloud_tokens", {})
        if provider.value not in cloud_tokens:
            raise HTTPException(
                status_code=403, detail=f"No {provider.value} connection found"
            )

        cv_data = await cloud_service.load_cv(cloud_tokens, provider, cv_file_id)

        # Get skill suggestions
        suggested_skills = await ai_service.suggest_skills(cv_data)

        # Track usage
        await ai_service._track_ai_usage(session["session_id"], "skill_suggestions")
        await record_session_activity(
            session["session_id"], "ai_suggest_skills", {"provider": provider.value}
        )

        return {
            "current_skills": [skill.name for skill in cv_data.skills],
            "suggested_skills": suggested_skills,
            "usage": await ai_service._check_daily_usage(session["session_id"]),
        }

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Skill suggestion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Skill suggestion failed")


@router.post("/analyze-job-match", response_model=JobAnalysisResponse)
async def analyze_job_compatibility(
    request: JobAnalysisRequest,
    session: dict = Depends(get_current_session),
):
    """Analyze CV compatibility with job posting"""

    try:
        # Check usage limits
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI usage limit reached. Used {usage_stats['used_today']}/{usage_stats['limit']} operations.",
            )

        # Extract provider from file_id or try all connected providers
        cloud_tokens = session.get("cloud_tokens", {})
        if not cloud_tokens:
            raise HTTPException(status_code=403, detail="No cloud providers connected")

        cv_data = None
        for provider_name, tokens in cloud_tokens.items():
            try:
                provider = CloudProvider(provider_name)
                cv_data = await cloud_service.load_cv(
                    cloud_tokens, provider, request.cv_file_id
                )
                break
            except:
                continue

        if not cv_data:
            raise HTTPException(
                status_code=404,
                detail="CV file not found in any connected cloud storage",
            )

        # Analyze job compatibility
        analysis = await ai_service.analyze_cv_for_job(
            cv_data, request.job_description, request.job_title
        )

        # Track usage
        await ai_service._track_ai_usage(session["session_id"], "job_analysis")
        await record_session_activity(
            session["session_id"], "ai_analyze_job", {"job_title": request.job_title}
        )

        return analysis

    except CloudProviderError as e:
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Job analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Job analysis failed")


@router.get("/models")
async def get_available_ai_models():
    """Get information about available AI models and their capabilities"""

    return {
        "available_models": AIConfig.MODELS,
        "feature_models": AIConfig.FEATURE_MODELS,
        "current_config": {
            "primary_model": settings.ai_model_primary,
            "premium_model": settings.ai_model_premium,
            "max_tokens": settings.ai_max_tokens,
            "temperature": settings.ai_temperature,
        },
        "rate_limits": {
            "free_tier": settings.free_tier_ai_operations,
            "pro_tier": settings.pro_tier_ai_operations,
            "business_tier": settings.business_tier_ai_operations,
        },
    }
