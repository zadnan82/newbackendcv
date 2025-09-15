# app/schemas.py
"""
Pydantic schemas for privacy-first CV platform.
Preserves existing CV structure for frontend compatibility.
"""

import re
from pydantic import BaseModel, EmailStr, field_validator, Field
from typing import Any, Dict, Literal, Optional, List, Union
from datetime import date, datetime
from enum import Enum


# ================== PRESERVED CV STRUCTURE (FROM ORIGINAL) ==================


# Core CV Component Schemas (PRESERVED EXACTLY)
class PersonalInfoBase(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = Field(None, max_length=100)
    email: EmailStr
    mobile: str = Field(..., min_length=5, max_length=20)
    city: Optional[str] = Field(None, max_length=100)
    address: Optional[str] = Field(None, max_length=200)
    postal_code: Optional[str] = Field(None, max_length=20)
    driving_license: Optional[str] = Field(None, max_length=50)
    nationality: Optional[str] = Field(None, max_length=50)
    place_of_birth: Optional[str] = Field(None, max_length=100)
    date_of_birth: Optional[date] = None
    linkedin: Optional[str] = Field(None, max_length=200)
    website: Optional[str] = Field(None, max_length=200)
    summary: Optional[str] = Field(None, max_length=2000)


class EducationBase(BaseModel):
    institution: str = Field(..., min_length=1, max_length=100)
    degree: str = Field(..., min_length=1, max_length=100)
    field_of_study: str = Field(..., min_length=1, max_length=100)
    location: Optional[str] = Field(None, max_length=100)
    start_date: date
    end_date: Optional[date] = None
    current: Optional[bool] = False
    gpa: Optional[str] = Field(None, max_length=10)


class ExperienceBase(BaseModel):
    company: str = Field(..., min_length=1, max_length=100)
    position: str = Field(..., min_length=1, max_length=100)
    location: Optional[str] = Field(None, max_length=100)
    start_date: date
    end_date: Optional[date] = None
    current: Optional[bool] = False
    description: Optional[str] = Field(None, max_length=2000)


class SkillBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    level: Optional[str] = Field(None, max_length=20)


class LanguageBase(BaseModel):
    language: str = Field(..., min_length=1, max_length=50)
    proficiency: str = Field(..., min_length=1, max_length=20)


class ReferralBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    relation: str = Field(..., min_length=1, max_length=50)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None


class CustomSectionBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    content: Optional[str] = Field(None, max_length=2000)


class ExtracurricularActivityBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)


class HobbyBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)


class CourseBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    institution: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)


class InternshipBase(BaseModel):
    company: str = Field(..., min_length=1, max_length=100)
    position: str = Field(..., min_length=1, max_length=100)
    location: Optional[str] = Field(None, max_length=100)
    start_date: date
    end_date: Optional[date] = None
    current: Optional[bool] = False
    description: Optional[str] = Field(None, max_length=2000)


class PhotoBase(BaseModel):
    photolink: Optional[str] = Field(None)


class CustomizationBase(BaseModel):
    template: str = Field("stockholm", min_length=1, max_length=50)
    accent_color: str = Field("#1a5276", max_length=20)
    font_family: str = Field("Helvetica, Arial, sans-serif", max_length=100)
    line_spacing: float = Field(1.5, ge=1.0, le=2.0)
    headings_uppercase: bool = False
    hide_skill_level: bool = False
    language: str = Field("en", min_length=2, max_length=10)

    class Config:
        from_attributes = True


# Complete CV Structure (PRESERVED)
class CompleteCV(BaseModel):
    """Complete CV structure - exactly as frontend expects"""

    title: str = Field(..., min_length=1, max_length=100)
    is_public: bool = False
    customization: CustomizationBase
    personal_info: Optional[PersonalInfoBase] = None

    # All sections optional as lists
    educations: List[EducationBase] = []
    experiences: List[ExperienceBase] = []
    skills: List[SkillBase] = []
    languages: List[LanguageBase] = []
    referrals: List[ReferralBase] = []
    custom_sections: List[CustomSectionBase] = []
    extracurriculars: List[ExtracurricularActivityBase] = []
    hobbies: List[HobbyBase] = []
    courses: List[CourseBase] = []
    internships: List[InternshipBase] = []

    # Photo optional
    photo: Optional[PhotoBase] = None


# Response Schemas with IDs (for API responses)
class PersonalInfoResponse(PersonalInfoBase):
    id: str  # Cloud file reference

    class Config:
        from_attributes = True


class EducationResponse(EducationBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class ExperienceResponse(ExperienceBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class SkillResponse(SkillBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class LanguageResponse(LanguageBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class ReferralResponse(ReferralBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class CustomSectionResponse(CustomSectionBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class ExtracurricularActivityResponse(ExtracurricularActivityBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class HobbyResponse(HobbyBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class CourseResponse(CourseBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class InternshipResponse(InternshipBase):
    id: str  # Index in array

    class Config:
        from_attributes = True


class PhotoResponse(PhotoBase):
    id: str  # Cloud reference

    class Config:
        from_attributes = True


class CustomizationResponse(CustomizationBase):
    id: str  # Part of CV file

    class Config:
        from_attributes = True


# Complete Resume Response (what frontend expects)
class ResumeResponse(BaseModel):
    """Frontend-compatible resume response"""

    id: str  # Cloud file ID
    title: str
    is_public: bool = False
    customization: Optional[CustomizationResponse] = None
    personal_info: Optional[PersonalInfoResponse] = None
    educations: List[EducationResponse] = []
    experiences: List[ExperienceResponse] = []
    skills: List[SkillResponse] = []
    languages: List[LanguageResponse] = []
    referrals: List[ReferralResponse] = []
    custom_sections: List[CustomSectionResponse] = []
    extracurriculars: List[ExtracurricularActivityResponse] = []
    hobbies: List[HobbyResponse] = []
    courses: List[CourseResponse] = []
    internships: List[InternshipResponse] = []
    photos: Optional[PhotoResponse] = None

    class Config:
        from_attributes = True


# ================== CLOUD & SESSION SCHEMAS ==================


class CloudProvider(str, Enum):
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    DROPBOX = "dropbox"
    BOX = "box"


class CloudFileMetadata(BaseModel):
    """Cloud file metadata"""

    file_id: str
    name: str
    provider: CloudProvider
    last_modified: datetime
    size_bytes: Optional[int] = None
    created_at: datetime


class CloudSession(BaseModel):
    """Anonymous cloud session"""

    session_id: str
    connected_providers: List[CloudProvider]
    expires_at: datetime
    created_at: datetime


class CloudAuthRequest(BaseModel):
    """Cloud provider authentication request"""

    provider: CloudProvider
    redirect_uri: str


class CloudAuthResponse(BaseModel):
    """Cloud provider authentication response"""

    auth_url: str
    state: str


class CloudConnectionStatus(BaseModel):
    """Cloud connection status"""

    provider: CloudProvider
    connected: bool
    email: Optional[str] = None  # User's email from provider
    storage_quota: Optional[Dict[str, Any]] = None


# ================== AI ENHANCEMENT SCHEMAS (PRESERVED) ==================


class PersonalInfoSummaryRequest(BaseModel):
    summary: str


class ExperienceDescriptionRequest(BaseModel):
    description: str


class SkillRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    level: Optional[str] = Field(None, max_length=20)

    @field_validator("name")
    def process_skill_name(cls, v):
        v = " ".join(v.split())
        return v


class SkillsUpdateRequest(BaseModel):
    skills: Union[List[SkillRequest], SkillRequest]

    @field_validator("skills")
    def ensure_list(cls, v):
        if isinstance(v, SkillRequest):
            return [v]
        return v


class AIUsageResponse(BaseModel):
    used_today: int
    limit: int
    remaining: int

    class Config:
        from_attributes = True


# ================== COVER LETTER SCHEMAS (PRESERVED) ==================


class CoverLetterBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    company_name: Optional[str] = Field(None, max_length=100)
    job_title: Optional[str] = Field(None, max_length=100)
    job_description: Optional[str] = Field(None)
    recipient_name: Optional[str] = Field(None, max_length=100)
    recipient_title: Optional[str] = Field(None, max_length=100)
    cover_letter_content: str
    is_favorite: Optional[bool] = False


class CoverLetterGenerationRequest(BaseModel):
    job_description: str = Field(
        ..., description="Job description for cover letter generation"
    )
    job_title: Optional[str] = Field(None, max_length=100)
    company_name: Optional[str] = Field(None, max_length=100)
    recipient_name: Optional[str] = Field(None, max_length=100)
    recipient_title: Optional[str] = Field(None, max_length=100)
    cv_file_id: str = Field(..., description="Cloud file ID of the CV to use")


class CoverLetterResponse(CoverLetterBase):
    id: str  # Cloud file ID
    cv_file_id: str  # Reference to CV used
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ================== JOB MATCHING SCHEMAS ==================


class JobAnalysisRequest(BaseModel):
    cv_file_id: str = Field(..., description="Cloud file ID of CV to analyze")
    job_description: str = Field(..., min_length=20, max_length=20000)
    job_title: str = Field(..., min_length=2, max_length=200)
    company_name: str = Field("", max_length=200)


class JobAnalysisResponse(BaseModel):
    match_score: float = Field(..., ge=0, le=100)
    ats_compatibility_score: float = Field(..., ge=0, le=100)
    keywords_present: List[str] = []
    keywords_missing: List[str] = []
    recommendations: List[str] = []
    strengths: List[str] = []
    improvement_areas: List[str] = []
    should_apply: bool
    application_tips: List[str] = []


# ================== SUBSCRIPTION & PRICING ==================


class PricingTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    BUSINESS = "business"


class SubscriptionFeatures(BaseModel):
    ai_operations_daily: int
    cloud_providers: List[CloudProvider]
    advanced_templates: bool
    priority_support: bool
    api_access: bool
    bulk_operations: bool


class SubscriptionStatus(BaseModel):
    tier: PricingTier
    features: SubscriptionFeatures
    usage_today: Dict[str, int]  # {"ai_enhance": 3, "cover_letter": 1}
    expires_at: Optional[datetime] = None


# ================== TEMPORARY SHARING ==================


class TemporaryShareRequest(BaseModel):
    cv_file_id: str
    max_views: int = Field(50, ge=1, le=1000)
    expires_hours: int = Field(24, ge=1, le=168)  # Max 1 week
    password: Optional[str] = Field(None, min_length=4, max_length=50)


class TemporaryShareResponse(BaseModel):
    share_id: str
    share_url: str
    qr_code_url: str
    expires_at: datetime
    max_views: int
    view_count: int


# ================== ERROR RESPONSES ==================


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    detail: str
    field_errors: Dict[str, List[str]]


# ================== CLOUD STORAGE METADATA ==================


class CVFileMetadata(BaseModel):
    """Metadata stored in cloud CV files"""

    version: str = "1.0"
    created_at: datetime
    last_modified: datetime
    created_with: str = "cv-privacy-platform"

    # Optional analytics consent
    analytics_consent: bool = False
