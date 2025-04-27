from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatSessionCreate(BaseModel):
    """Schema for creating a new chat session"""
    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Schema for chat session response"""
    session_id: str
    user_email: Optional[str] = None
    title: Optional[str] = None
    session_start: datetime


class ChatHistoryCreate(BaseModel):
    """Schema for creating a new chat history entry"""
    query_text: str
    session_id: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Schema for chat history response"""
    response_id: str
    query_text: str
    response_text: str
    created_at: datetime


class SessionWithHistory(BaseModel):
    """Schema for session with history response"""
    session_id: str
    data: List[Dict]  # Using Dict instead of ChatHistoryResponse for flexibility


class SessionGrouped(BaseModel):
    """Schema for grouped sessions response"""
    today: Optional[List[Dict]] = None
    yesterday: Optional[List[Dict]] = None
    last_week: Optional[List[Dict]] = None


class SymptomInput(BaseModel):
    """Schema for MS symptom analysis input"""
    clinical_text: str = Field(..., description="Description of MS symptoms or concerns")


class SymptomAnalysisResponse(BaseModel):
    """Schema for MS symptom analysis response"""
    analysis: str = Field(..., description="Analysis of described MS symptoms")
    used_knowledge_base: bool = Field(..., description="Whether the knowledge base was used")


class DocumentUploadResponse(BaseModel):
    """Schema for document training response"""
    status: str
    message: str
    chunks_created: int


class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: str
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserResponse(UserBase):
    """Schema for user response"""
    created_at: datetime