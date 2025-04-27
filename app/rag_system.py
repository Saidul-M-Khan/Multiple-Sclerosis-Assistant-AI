# rag_system.py
import os
import uuid
import json
from typing import List, Dict, Optional, Union
import shutil
import asyncio
from datetime import datetime
from fastapi import UploadFile, HTTPException
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Get MongoDB connection details from environment variables
MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017/")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "ms_assistant")

# Initialize MongoDB collections for document storage
db_client = MongoClient(MONGODB_URL)
db = db_client[DATABASE_NAME]
knowledge_base = db["knowledge_base"]
vector_indices = db["vector_indices"]
document_metadata = db["document_metadata"]

# Set up file storage paths
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
VECTOR_DB_DIR = os.environ.get("VECTOR_STORE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db"))

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Configure embeddings
embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

# File processors for different file types
file_processors = {
    "pdf": PyPDFLoader,
    "txt": TextLoader,
    "csv": CSVLoader,
    "json": JSONLoader
}

async def process_file(file_path: str, file_type: str, user_email: str, 
                       collection_name: str, document_title: str = None, document_description: str = None):
    """Process a file and store its embeddings in the FAISS vector database"""
    try:
        # Load documents based on file type
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_type == "txt":
            loader = TextLoader(file_path)
            documents = loader.load()
        elif file_type == "csv":
            loader = CSVLoader(file_path)
            documents = loader.load()
        elif file_type == "json":
            # For JSON, we need to specify the jq schema to extract text
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.[]',
                text_content=False
            )
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        
        # Create directory for this collection
        persist_directory = os.path.join(VECTOR_DB_DIR, collection_name)
        os.makedirs(persist_directory, exist_ok=True)
        
        # Generate embeddings and create FAISS index
        vectordb = FAISS.from_documents(docs, embeddings)
        
        # Save the FAISS index
        faiss_index_path = os.path.join(persist_directory, "faiss_index")
        vectordb.save_local(faiss_index_path)
        
        # Save document metadata
        doc_metadata = {
            "user_email": user_email,
            "collection_name": collection_name,
            "file_name": os.path.basename(file_path),
            "document_title": document_title or os.path.basename(file_path),
            "document_description": document_description,
            "upload_date": datetime.now(),
            "document_count": len(docs),
            "vector_db_path": faiss_index_path
        }
        
        document_metadata.insert_one(doc_metadata)
        
        return {
            "status": "success",
            "collection_name": collection_name,
            "document_count": len(docs),
            "message": f"Successfully processed {os.path.basename(file_path)}"
        }
        
    except Exception as e:
        # Clean up in case of failure
        persist_directory = os.path.join(VECTOR_DB_DIR, collection_name)
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

async def save_uploaded_file(file: UploadFile, user_email: str) -> tuple:
    """Save an uploaded file to disk and return the file path and extension"""
    # Create user directory if it doesn't exist
    user_dir = os.path.join(UPLOAD_DIR, user_email)
    os.makedirs(user_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = file.filename
    file_extension = original_filename.split(".")[-1].lower()
    unique_filename = f"{timestamp}_{uuid.uuid4().hex}.{file_extension}"
    file_path = os.path.join(user_dir, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return file_path, file_extension

def get_user_collections(user_email: str) -> List[Dict]:
    """Get all collections for a specific user"""
    collections = list(document_metadata.find({"user_email": user_email}))
    # Convert ObjectId to string
    for collection in collections:
        if '_id' in collection:
            collection['_id'] = str(collection['_id'])
    
    return collections

def query_knowledge_base(user_email: str, query: str, collection_names: List[str] = None, top_k: int = 5):
    """Query the knowledge base with a user query using FAISS"""
    results = []
    
    # If no specific collections are provided, query all collections for the user
    if not collection_names:
        user_collections = get_user_collections(user_email)
        collection_names = [collection["collection_name"] for collection in user_collections]
    
    # Query each collection
    for collection_name in collection_names:
        # Check if collection exists
        collection_info = document_metadata.find_one({"collection_name": collection_name})
        if not collection_info:
            continue
        
        try:
            # Load the FAISS index
            faiss_index_path = collection_info["vector_db_path"]
            vectordb = FAISS.load_local(faiss_index_path, embeddings)
            
            # Perform similarity search
            docs_with_scores = vectordb.similarity_search_with_score(query, k=top_k)
            
            # Add results from this collection
            for doc, score in docs_with_scores:
                # FAISS returns euclidean distance, convert to similarity score (closer to 1 is better)
                # Normalize score to be between 0 and 1 (1 being the best match)
                normalized_score = 1.0 / (1.0 + score)  # Convert distance to similarity
                
                results.append({
                    "collection_name": collection_name,
                    "document_title": collection_info["document_title"],
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": normalized_score
                })
        except Exception as e:
            print(f"Error querying collection {collection_name}: {str(e)}")
    
    # Sort results by relevance (higher score is better)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results[:top_k]