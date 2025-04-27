# Multiple Sclerosis (MS) Assistant API

## Overview

This API provides a specialized Multiple Sclerosis (MS) assistant that helps users understand, identify, and manage MS symptoms and concerns. The assistant uses advanced AI techniques including Retrieval Augmented Generation (RAG) to provide accurate and helpful information about MS.

## Key Features

1. **MS-Focused Conversations**: Chat with an AI assistant specialized in MS knowledge and support
2. **Symptom Analysis**: Get analysis of potential MS symptoms and management suggestions
3. **Knowledge Base Training**: Upload and process MS-related research papers and documents to enhance the AI's knowledge
4. **Conversation History**: Track and revisit past conversations about MS symptoms and concerns
5. **User Authentication**: Secure access to personal conversation history and data

## API Endpoints

### Authentication Endpoints

- `POST /auth/register`: Register a new user
- `POST /auth/login`: Login and receive an access token
- `GET /auth/me`: Get current user information

### Chat Endpoints

- `POST /chat_session/`: Create a new chat session
- `GET /session_chats/{session_id}`: Get all messages in a specific chat session
- `POST /chat/`: Send a message and get AI response about MS
- `GET /chat/`: Get chat history for a specific session
- `GET /all_sessions/`: Get all chat sessions grouped by time period

### MS-Specific Endpoints

- `POST /analyze_ms_symptoms/`: Analyze provided MS symptoms and get recommendations
- `POST /upload_training_document/`: Upload MS research papers or documents to train the AI

## MS Symptom Analysis

The symptom analysis feature uses advanced AI techniques to:

1. Identify which symptoms might be consistent with MS
2. Explain how these symptoms relate to MS pathophysiology
3. Mention other possible causes for these symptoms
4. Provide practical advice for symptom management
5. Emphasize the importance of proper medical diagnosis

**Example Request:**
```json
POST /analyze_ms_symptoms/
{
  "clinical_text": "I've been experiencing numbness in my left arm and leg for the past two weeks. I'm also having trouble focusing my eyes sometimes, and I feel unusually tired even after a full night's sleep."
}
```

## Knowledge Base Training

The API allows you to train the MS assistant with:

- Medical research papers on MS
- Clinical guidelines
- Educational materials about MS
- Treatment information
- Patient resources

**Supported File Formats:**
- PDF (`.pdf`)
- Text (`.txt`)
- CSV (`.csv`)
- Word documents (`.docx`, `.doc`)
- Rich Text Format (`.rtf`)
- Markdown (`.md`)

## Implementation Details

The MS Assistant API uses:

1. **OpenAI GPT-4o**: For natural language understanding and response generation
2. **LangChain**: For document processing and retrieval
3. **Chroma Vector Database**: For storing and retrieving relevant information
4. **RAG (Retrieval Augmented Generation)**: To combine the power of retrieval and generation for more accurate responses

## Important Notes

- The MS Assistant is designed to provide information and support but is not a replacement for professional medical advice.
- All communications emphasize the importance of consulting healthcare providers.
- The system continuously improves as more MS-specific documents are added to the knowledge base.