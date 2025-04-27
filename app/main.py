import os
from fastapi import FastAPI, Depends, HTTPException, Header, Form, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from typing import List, Dict, Optional, Union, Any
import uuid
from datetime import date, timedelta, datetime
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse

from .database import get_db, chat_sessions, chat_history
from .models import create_chat_session, create_chat_history
from . import schemas
from .ms_integration import MSHealthAI

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from .auth import Token, UserCreate, authenticate_user, create_access_token, get_current_user, register_user, get_user_email_from_token

app = FastAPI(
    title="MS Assistant API",
    description="FastAPI backend for a MS health chatbot",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication and Registration Models
class LoginForm(BaseModel):
    email: EmailStr
    password: str

class RegisterForm(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    email: str
    message: str

# Helper functions for MongoDB date filtering
def get_today_date_range():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    return today, tomorrow

def get_yesterday_date_range():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    return yesterday, today

def get_last_week_date_range():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    last_week = yesterday - timedelta(days=6)  # Last 7 days excluding today and yesterday
    return last_week, yesterday

# Registration
@app.post("/auth/register", response_model=AuthResponse)
async def user_registration(form_data: RegisterForm, db=Depends(get_db)):
    """
    Register a new user and return login token
    """
    # Validate password match
    if form_data.password != form_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # Create user
    user = UserCreate(email=form_data.email, password=form_data.password)
    result = register_user(db, user)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Auto login after registration - create and return token
    access_token = create_access_token(data={"sub": form_data.email})
    
    return {
        # "access_token": access_token,
        # "token_type": "bearer",
        "email": form_data.email, 
        "message": "Registration successful"
    }

# Login
@app.post("/auth/login", response_model=AuthResponse)
async def user_login(form_data: LoginForm, db=Depends(get_db)):
    """
    Authenticate user and return login token
    """
    user = authenticate_user(db, form_data.email, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user["email"]})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user["email"],
        "message": "Login successful"
    }


@app.get("/auth/me")
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user info"""
    return {
        "email": current_user["email"],
        "created_at": current_user["created_at"]
    }


@app.post("/upload_training_document/", response_model=Dict[str, Any])
async def upload_training_document(
    file: UploadFile = File(...),
    document_title: str = Form(None),
    document_description: str = Form(None),
    current_user=Depends(get_current_user)
):
    """
    Upload and process a document to train the MS knowledge base
    """
    # Check if file is empty
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = ['.pdf', '.txt', '.csv', '.json']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save the uploaded file
        from .rag_system import save_uploaded_file
        user_email = current_user["email"]
        file_path, file_extension = await save_uploaded_file(file, user_email)
        
        # Process the file using the MSHealthAI helper
        result = await MSHealthAI.upload_training_document(
            file_path=file_path,
            user_email=user_email,
            document_title=document_title or file.filename,
            document_description=document_description
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Chat session endpoints
@app.get("/session_chats/{session_id}", response_model=schemas.SessionWithHistory)
async def get_session_chats(
    session_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Get all chat messages for a specific session
    """
    user_email = current_user["email"]
    
    # Get session and verify it belongs to the user
    session = chat_sessions.find_one({"session_id": session_id})
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if the session belongs to the current user
    if session.get("user_email") and session["user_email"] != user_email:
        raise HTTPException(status_code=403, detail="Access denied to this session")
    
    # Get all chat messages for this session, sorted by creation time (oldest first)
    chats = list(chat_history
                .find({"session_id": session_id})
                .sort("created_at", 1))  # 1 for ascending order (oldest first)
    
    # Convert ObjectId to string for each chat message
    for item in chats:
        if '_id' in item:
            item['_id'] = str(item['_id'])
    
    # Return session info and all chats
    return {
        "session_id": session_id,
        "title": session.get("title", "Untitled Session"),
        "session_start": session.get("session_start"),
        "data": chats
    }

# Chatting with AI
@app.post("/chat/", response_model=schemas.SessionWithHistory)
async def process_chat(
    chat_input: schemas.ChatHistoryCreate,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    user_email = current_user["email"]
    session_id = chat_input.session_id
    
    # If no session_id is provided, check for unused sessions or create a new one
    if not session_id:
        # Check for unused sessions (sessions without chat history)
        unused_sessions = list(chat_sessions.find({
            "user_email": user_email,
            "session_id": {"$nin": list(chat_history.distinct("session_id"))}
        }))
        
        if unused_sessions:
            session_id = unused_sessions[0]["session_id"]
        else:
            # Create new session for the user
            new_session = create_chat_session(user_email=user_email)
            chat_sessions.insert_one(new_session)
            session_id = new_session["session_id"]
    else:
        # Validate that the session exists and belongs to the user
        session = chat_sessions.find_one({"session_id": session_id})
        
        if not session:
            # Session not found - create new one
            new_session = create_chat_session(user_email=user_email)
            chat_sessions.insert_one(new_session)
            session_id = new_session["session_id"]
        elif session.get("user_email") and session["user_email"] != user_email:
            # Session belongs to another user - don't allow access
            raise HTTPException(status_code=403, detail="Access denied to this session")
    
    # Check if this is the first message for title generation
    history_count = chat_history.count_documents({"session_id": session_id})
    
    # Generate title for the session if this is the first message
    if history_count == 0:
        title = MSHealthAI.generate_title(chat_input.query_text)
        chat_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"title": title}}
        )
    
    # Get ALL conversation history for this session, sorted chronologically
    history = list(chat_history
                  .find({"session_id": session_id})
                  .sort("created_at", 1))  # 1 for ascending order (oldest first)
    
    history_for_ai = [
        {"query_text": item["query_text"], "response_text": item["response_text"]}
        for item in history
    ]
    
    # Process query through AI with complete history context and user email for RAG
    conversation_chain = MSHealthAI.get_conversation_chain(user_email=user_email)
    response = conversation_chain["invoke"](chat_input.query_text, history_for_ai)
    
    # Save response to database
    new_chat = create_chat_history(
        session_id=session_id,
        query_text=chat_input.query_text,
        response_text=response["answer"]
    )
    
    chat_history.insert_one(new_chat)
    
    # Return updated history
    updated_history = list(chat_history
                          .find({"session_id": session_id})
                          .sort("created_at", -1))  # Most recent first for the response
    
    # Convert ObjectId to string
    for item in updated_history:
        if '_id' in item:
            item['_id'] = str(item['_id'])
    
    return {
        "session_id": session_id,
        "data": updated_history
    }

# Get All Session List
@app.get("/all_sessions/", response_model=schemas.SessionGrouped)
async def get_all_sessions(current_user=Depends(get_current_user), db=Depends(get_db)):
    user_email = current_user["email"]
    
    today_start, tomorrow = get_today_date_range()
    yesterday_start, today_start_again = get_yesterday_date_range()
    last_week_start, yesterday_start_again = get_last_week_date_range()
    
    # Get sessions for today for this user
    today_sessions = list(chat_sessions.find({
        "user_email": user_email,
        "session_start": {"$gte": today_start, "$lt": tomorrow}
    }).sort("session_start", -1))
    
    # Get sessions for yesterday for this user
    yesterday_sessions = list(chat_sessions.find({
        "user_email": user_email,
        "session_start": {"$gte": yesterday_start, "$lt": today_start}
    }).sort("session_start", -1))
    
    # Get sessions for last week for this user
    last_week_sessions = list(chat_sessions.find({
        "user_email": user_email,
        "session_start": {"$gte": last_week_start, "$lt": yesterday_start}
    }).sort("session_start", -1))
    
    # Convert ObjectId to string
    for sessions in [today_sessions, yesterday_sessions, last_week_sessions]:
        for item in sessions:
            if '_id' in item:
                item['_id'] = str(item['_id'])
    
    response_data = {}
    
    if today_sessions:
        response_data["today"] = today_sessions
    
    if yesterday_sessions:
        response_data["yesterday"] = yesterday_sessions
    
    if last_week_sessions:
        response_data["last_week"] = last_week_sessions
    
    return response_data