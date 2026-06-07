from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class UserProfile(BaseModel):
    name: str
    major: str
    academicLevel: str
    skills: List[str]
    location: str
    gpa: str
    interests: List[str]

class Opportunity(BaseModel):
    id: str
    title: str
    organization: str
    requirements: str
    type: Optional[str] = None

class AnalysisRequest(BaseModel):
    profile: UserProfile
    opportunity: Opportunity

class ContactResult(BaseModel):
    name: str
    profile_url: str
    snippet: str
    suggested_message: str

class AnalysisResponse(BaseModel):
    opp_id: str
    match_score: int = Field(..., ge=0, le=100)
    success_probability: str  # Low, Medium, High
    eligibility_reasoning: str
    application_tips: List[str]
    suggested_contacts: List[ContactResult]
    reasoning_details: Optional[Dict[str, Any]] = None

class ChatHistoryItem(BaseModel):
    role: str  # user, assistant, system
    content: Optional[str] = None
    reasoning_details: Optional[Any] = None

class ChatRequest(BaseModel):
    messages: List[ChatHistoryItem]
    profile: Optional[UserProfile] = None
    opportunity: Optional[Opportunity] = None

class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    reasoning_details: Optional[Any] = None

class DiscoveryRequest(BaseModel):
    category: str
    user_interest: str
    academic_level: str
    location: str
    exclude_urls: Optional[List[str]] = []

class DiscoveryOpportunity(BaseModel):
    title: str
    url: str
    organization: str
    description: str
    availability: str

class DiscoveryResponse(BaseModel):
    opportunities: List[DiscoveryOpportunity]
