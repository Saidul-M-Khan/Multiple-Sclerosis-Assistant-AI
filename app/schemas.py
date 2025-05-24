# from pydantic import BaseModel, Field, EmailStr, validator
# from typing import List, Optional, Dict, Any
# from datetime import datetime


# class ChatSessionCreate(BaseModel):
#     """Schema for creating a new chat session"""
#     title: Optional[str] = None


# class ChatSessionResponse(BaseModel):
#     """Schema for chat session response"""
#     session_id: str
#     user_email: Optional[str] = None
#     title: Optional[str] = None
#     session_start: datetime


# class ChatHistoryCreate(BaseModel):
#     """Schema for creating a new chat history entry"""
#     query_text: str
#     session_id: Optional[str] = None


# class ChatHistoryResponse(BaseModel):
#     """Schema for chat history response"""
#     response_id: str
#     query_text: str
#     response_text: str
#     created_at: datetime


# class SessionWithHistory(BaseModel):
#     """Schema for session with history response"""
#     session_id: str
#     data: List[Dict]  # Using Dict instead of ChatHistoryResponse for flexibility


# class SessionGrouped(BaseModel):
#     """Schema for grouped sessions response"""
#     today: Optional[List[Dict]] = None
#     yesterday: Optional[List[Dict]] = None
#     last_week: Optional[List[Dict]] = None


# class SymptomInput(BaseModel):
#     """Schema for MS symptom analysis input"""
#     clinical_text: str = Field(..., description="Description of MS symptoms or concerns")


# class SymptomAnalysisResponse(BaseModel):
#     """Schema for MS symptom analysis response"""
#     analysis: str = Field(..., description="Analysis of described MS symptoms")
#     used_knowledge_base: bool = Field(..., description="Whether the knowledge base was used")


# class DocumentUploadResponse(BaseModel):
#     """Schema for document training response"""
#     status: str
#     message: str
#     chunks_created: int


# class UserBase(BaseModel):
#     """Base user schema"""
#     email: EmailStr


# class UserCreate(UserBase):
#     """Schema for creating a new user"""
#     password: str
    
#     @validator('password')
#     def password_strength(cls, v):
#         if len(v) < 8:
#             raise ValueError('Password must be at least 8 characters long')
#         return v


# class UserResponse(UserBase):
#     """Schema for user response"""
#     created_at: datetime

from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class EmailRequest(BaseModel):
    """Schema for email request"""
    email: EmailStr


class SessionResponse(BaseModel):
    """Schema for session response"""
    session_id: str
    created_at: datetime
    email: str
    stage: str
    analysis_complete: bool
    title: str

    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj):
        # Convert UUID to string
        if hasattr(obj, 'id'):
            obj.id = str(obj.id)
        return super().from_orm(obj)


class ChatMessageRequest(BaseModel):
    """Schema for chat message request"""
    session_id: Optional[str] = None
    message: str
    email: EmailStr
    stage: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Schema for chat message response"""
    response: str
    session_id: str
    analysis_complete: bool
    message: str
    timestamp: datetime

    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj):
        # Convert UUID to string
        if hasattr(obj, 'session_id'):
            obj.session_id = str(obj.session_id)
        return super().from_orm(obj)


class SessionTitleUpdate(BaseModel):
    """Schema for updating session title"""
    title: str


class ChatSessionCreate(BaseModel):
    """Schema for creating a new chat session"""
    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Schema for chat session response"""
    session_id: str
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
    title: Optional[str] = None
    session_start: Optional[datetime] = None
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