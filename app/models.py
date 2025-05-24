from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, JSON, UniqueConstraint, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("Session", back_populates="user")

    __table_args__ = (
        UniqueConstraint('email', name='uq_user_email'),
    )

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, ForeignKey("users.email"))
    stage = Column(String, default="initial")
    analysis_complete = Column(Boolean, default=False)
    ai_state = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    title = Column(String, default="New MS Consultation")
    
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    message = Column(Text)
    response = Column(Text)
    stage = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("Session", back_populates="messages")