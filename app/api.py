import os
from fastapi import FastAPI, Depends, HTTPException, Header, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Union, Any
import uuid
from datetime import timedelta, datetime
from pydantic import BaseModel, ValidationError, EmailStr
import logging
from dotenv import load_dotenv
from .ms_health_ai import MSHealthAI, MSHealthAIError, InvalidStateError, ParsingError
from sqlalchemy.orm import Session
from fastapi.openapi.utils import get_openapi

from .database import get_db, engine, init_db
from .models import Base, User, Session as DBSession, ChatMessage
from .schemas import (
    EmailRequest,
    SessionResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    SessionTitleUpdate
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize database
init_db()

# Initialize MSHealthAI with database session
def get_ms_health_ai(db: Session = Depends(get_db)) -> MSHealthAI:
    return MSHealthAI(db)

app = FastAPI(
    title="Multiple Sclerosis Health Assistant",
    description="FastAPI backend for a Multiple Sclerosis Health Assistant",
    version=API_VERSION,
    debug=DEBUG_MODE
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handlers
@app.exception_handler(MSHealthAIError)
async def ms_health_ai_exception_handler(request: Request, exc: MSHealthAIError):
    logger.error(f"MSHealthAI Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"AI System Error: {str(exc)}"}
    )

@app.exception_handler(InvalidStateError)
async def invalid_state_exception_handler(request: Request, exc: InvalidStateError):
    logger.error(f"Invalid State Error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid State: {str(exc)}"}
    )

@app.exception_handler(ParsingError)
async def parsing_exception_handler(request: Request, exc: ParsingError):
    logger.error(f"Parsing Error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Failed to parse input: {str(exc)}"}
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation Error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

@app.post("/session/create", response_model=SessionResponse)
def create_session(request: EmailRequest, db: Session = Depends(get_db)):
    # Check if user exists, if not create
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        user = User(email=request.email)
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Create new session
    session = DBSession(
        email=request.email,
        stage="initial",
        analysis_complete=False,
        ai_state={}
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Convert session to response model
    return SessionResponse(
        session_id=str(session.id),
        created_at=session.created_at,
        email=session.email,
        stage=session.stage,
        analysis_complete=session.analysis_complete,
        title=session.title
    )

@app.post("/chat", response_model=ChatMessageResponse)
def chat(request: ChatMessageRequest, db: Session = Depends(get_db)):
    """
    Handle chat messages. Can be used in two ways:
    1. Start a new chat (no session_id provided) - creates new session
    2. Continue existing chat (session_id provided) - uses existing session
    """
    try:
        # Check if user exists, if not create
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            user = User(email=request.email)
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # Get or create session
        if request.session_id:
            # Use existing session if provided
            session = db.query(DBSession).filter(DBSession.id == request.session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            if session.email != request.email:
                raise HTTPException(status_code=403, detail="Email mismatch for session")
        else:
            # Create new session if no session_id provided
            session = DBSession(
                email=request.email,
                stage="initial",
                analysis_complete=False,
                ai_state={},
                title=generate_session_title(request.message)
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            request.session_id = str(session.id)

        # Process message with AI
        ms_health_ai = get_ms_health_ai(db)
        response = ms_health_ai.process_message(
            session_id=str(session.id),
            message=request.message,
            email=request.email
        )
        
        # Get current state
        state = ms_health_ai.get_session_state(str(session.id))
        if not state:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Store message and response
        chat_message = ChatMessage(
            session_id=session.id,
            message=request.message,
            response=response,
            stage=state.get("stage", "initial")
        )
        db.add(chat_message)
        
        # Update session state
        session.stage = state.get("stage", "initial")
        session.analysis_complete = state.get("analysis_complete", False)
        session.ai_state = state
        session.last_updated = datetime.utcnow()
        db.commit()
        
        return ChatMessageResponse(
            response=response,
            session_id=str(session.id),
            analysis_complete=session.analysis_complete,
            message=request.message,
            timestamp=chat_message.timestamp
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

def generate_session_title(message: str) -> str:
    """
    Generate a meaningful title for the session based on the first message.
    """
    # Clean the message
    message = message.strip()
    
    # If message is too short, use a default title
    if len(message) < 10:
        return "New Chat Session"
    
    # Take first 50 characters or up to the first sentence
    title = message[:50]
    
    # If there's a period in the first 50 chars, cut at the period
    if '.' in title:
        title = title.split('.')[0]
    
    # If there's a question mark, cut at the question mark
    if '?' in title:
        title = title.split('?')[0]
    
    # Clean up the title
    title = title.strip()
    
    # If title is too short after cleaning, use a default
    if len(title) < 10:
        return "New Chat Session"
    
    # Add ellipsis if we cut the message
    if len(message) > len(title):
        title += "..."
    
    return title

@app.get("/session/{session_id}/chats", response_model=List[ChatMessageResponse])
def get_session_chats(session_id: str, db: Session = Depends(get_db)):
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
    return [
        ChatMessageResponse(
            response=msg.response,
            session_id=str(session_id),
            analysis_complete=session.analysis_complete,
            message=msg.message,
            timestamp=msg.timestamp
        )
        for msg in messages
    ]

@app.post("/generate_report/{session_id}")
def generate_report(session_id: str, db: Session = Depends(get_db)):
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.analysis_complete:
        raise HTTPException(status_code=400, detail="Analysis not complete")
    
    return {
        "session_id": session_id,
        "analysis": get_ms_health_ai(db).get_session_state(session_id)["analysis"],
        "recommendations": get_ms_health_ai(db).get_session_state(session_id)["recommendations"]
    }

@app.delete("/session/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """
    Delete a session and all its associated chat messages.
    This will permanently remove the session and all its data from the database.
    """
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        ms_health_ai = get_ms_health_ai(db)
        ms_health_ai.clear_session(session_id)
        return {"message": "Session and all associated messages deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@app.patch("/session/{session_id}/title", response_model=SessionResponse)
def update_session_title(session_id: str, title_update: SessionTitleUpdate, db: Session = Depends(get_db)):
    """
    Update the title of a session.
    """
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session.title = title_update.title
        session.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(session)
        
        return SessionResponse(
            session_id=session.id,
            created_at=session.created_at,
            email=session.email,
            stage=session.stage,
            analysis_complete=session.analysis_complete,
            title=session.title
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update session title: {str(e)}")

@app.get("/user/{email}/sessions", response_model=List[SessionResponse])
def get_user_sessions(email: str, db: Session = Depends(get_db)):
    """
    Get all sessions for a specific user by their email.
    Returns a list of sessions ordered by last_updated timestamp (most recent first).
    """
    # First check if user exists
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get all sessions for the user
    sessions = db.query(DBSession).filter(DBSession.email == email).order_by(DBSession.last_updated.desc()).all()
    
    return [
        SessionResponse(
            session_id=str(session.id),
            created_at=session.created_at,
            email=session.email,
            stage=session.stage,
            analysis_complete=session.analysis_complete,
            title=session.title
        )
        for session in sessions
    ]

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="MS Health Assistant API",
        version="1.0.0",
        description="API for MS Health Assistant",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi 