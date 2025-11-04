import json
import logging
import re
import time
from typing import Dict, List, Literal, Optional, Any, Union

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from datetime import date, time as dt_time, datetime

# ----------------------------------------------------------------------
# 1. Configuration and Mocks for External Services
# ----------------------------------------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mock Configuration ---
class AppConfig:
    """Mock configuration for external services."""
    VECTOR_DIMENSION = 128 # Mock dimension
    LLM_MODEL = "gemini-2.5-flash-preview-09-2025"
    DB_MOCK_DELAY = 0.05
    # Mock LLM API Key would go here
    # Mock Pinecone/Redis/PostgreSQL connection strings would go here

# --- Mock Implementations for Persistence and AI Services ---

class MockRedisCache:
    """Mock for Redis-based chat memory storage."""
    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        logger.info("MockRedisCache initialized.")

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves chat history for a session."""
        time.sleep(AppConfig.DB_MOCK_DELAY)
        return self._store.get(session_id, [])

    def save_message(self, session_id: str, role: str, content: str):
        """Adds a new message to the history."""
        history = self._store.get(session_id, [])
        history.append({"role": role, "content": content})
        self._store[session_id] = history[-10:]  # Keep last 10 messages for memory
        time.sleep(AppConfig.DB_MOCK_DELAY)
        logger.info(f"Saved message to session {session_id}. History length: {len(history)}")

class MockPostgreSQLDB:
    """Mock for SQL/NoSQL DB storing metadata and bookings."""
    def __init__(self):
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.booking_store: List[Dict[str, Any]] = []
        logger.info("MockPostgreSQLDB initialized.")

    def save_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """Saves document metadata."""
        time.sleep(AppConfig.DB_MOCK_DELAY)
        self.metadata_store[doc_id] = metadata
        logger.info(f"Metadata saved for document ID: {doc_id}")

    def save_booking(self, booking_data: Dict[str, Any]):
        """Saves a new interview booking."""
        time.sleep(AppConfig.DB_MOCK_DELAY)
        booking_data['booking_id'] = f"BK-{len(self.booking_store) + 1}"
        self.booking_store.append(booking_data)
        logger.info(f"Booking saved: {booking_data['booking_id']}")
        return booking_data['booking_id']

    def get_metadata_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Mock retrieval of metadata for RAG context."""
        time.sleep(AppConfig.DB_MOCK_DELAY)
        # Simplified: Check all stored metadata for a matching chunk ID prefix
        for doc_id, meta in self.metadata_store.items():
            if chunk_id.startswith(doc_id):
                 return meta
        return None


class MockPineconeVectorDB:
    """Mock for Pinecone Vector Database operations."""
    def __init__(self):
        self.index: Dict[str, Dict[str, Any]] = {}  # {id: {vector: [], text: "", metadata: {}}}
        logger.info("MockPineconeVectorDB initialized.")

    def upsert(self, vectors: List[Dict[str, Any]]):
        """Mocks upserting vectors."""
        time.sleep(AppConfig.DB_MOCK_DELAY)
        for vector in vectors:
            self.index[vector['id']] = vector
        logger.info(f"Upserted {len(vectors)} vectors into mock index.")

    def query(self, embedding: List[float], top_k: int = 3) -> List[str]:
        """Mocks vector similarity search. Returns the source texts."""
        time.sleep(AppConfig.DB_MOCK_DELAY)
        if not self.index:
            return []

        # Simplified: Just return the first few documents for mock purposes
        results = list(self.index.values())[:top_k]
        logger.info(f"Mock Query performed. Retrieved {len(results)} context chunks.")
        return [res.get('text', '') for res in results]

    def get_document_source(self, chunk_id: str) -> str:
        """Retrieves the source text for a chunk ID."""
        return self.index.get(chunk_id, {}).get('text', 'Source text not found.')

# Initialize Mocks (Dependency Injection)
mock_redis = MockRedisCache()
mock_db = MockPostgreSQLDB()
mock_pinecone = MockPineconeVectorDB()

# ----------------------------------------------------------------------
# 2. Pydantic Schemas and Models
# ----------------------------------------------------------------------

ChunkingStrategy = Literal["recursive_char", "sentence_split"]

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    upload_timestamp: datetime
    chunking_strategy: ChunkingStrategy
    chunk_count: int

class ChunkingRequest(BaseModel):
    strategy: ChunkingStrategy

class BookingData(BaseModel):
    name: str = Field(description="The full name of the person booking the interview.")
    email: EmailStr = Field(description="The email address for contact and confirmation.")
    dte: date = Field(description="The date of the interview (YYYY-MM-DD format).")
    time: dt_time = Field(description="The time of the interview (HH:MM:SS format).")

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the chat session.")
    query: str = Field(..., description="The user's current query.")

class ChatResponse(BaseModel):
    response: str
    context_sources: List[str]
    booking_confirmation: Optional[str] = None
    booking_details: Optional[BookingData] = None

# Define the JSON schema for the booking tool for the LLM
BOOKING_TOOL_SCHEMA = {
    "name": "interview_booking",
    "description": "Book an interview for a candidate. This must be called when the user explicitly requests to schedule an interview and provides all necessary information (name, email, date, time).",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The full name of the person booking."},
            "email": {"type": "string", "format": "email", "description": "The email address."},
            "date": {"type": "string", "format": "date", "description": "The interview date in YYYY-MM-DD format."},
            "time": {"type": "string", "format": "time", "description": "The interview time in HH:MM:SS format."},
        },
        "required": ["name", "email", "date", "time"]
    }
}


# ----------------------------------------------------------------------
# 3. Services (Core Logic)
# ----------------------------------------------------------------------

class LLMMockService:
    """Mock service for LLM interaction with Tool Calling logic."""

    @staticmethod
    def mock_embed(text: str) -> List[float]:
        """Generates a mock embedding vector (fixed size 128)."""
        # In a real scenario, this would call an API like OpenAI, Cohere, or Google Embeddings
        # To simulate variation, we use the length of the text
        base_val = len(text) % 100 / 100.0
        return [(base_val + i * 0.001) % 1.0 for i in range(AppConfig.VECTOR_DIMENSION)]

    @staticmethod
    def _create_llm_prompt(history: List[Dict[str, str]], context: str, user_query: str) -> str:
        """Constructs the full prompt including history and context."""
        system_instruction = (
            "You are a helpful and professional conversational RAG assistant. "
            "Your knowledge comes from the provided context and the conversation history. "
            "If the user asks about scheduling an interview, you MUST use the `interview_booking` tool. "
            "If the information is incomplete for booking, ask follow-up questions. "
            "If the query is a simple question, answer concisely based on the context. "
            "Cite your source documents by referring to the information itself (e.g., 'According to the document...')."
        )
        history_str = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in history])

        full_prompt = (
            f"--- System Instruction ---\n{system_instruction}\n\n"
            f"--- Conversation History ---\n{history_str}\n\n"
            f"--- Retrieved Context ---\n{context}\n\n"
            f"--- User Query ---\n{user_query}"
        )
        return full_prompt

    @staticmethod
    def generate_response(
        history: List[Dict[str, str]],
        context: List[str],
        user_query: str
    ) -> Union[str, Dict[str, Any]]:
        """Mocks LLM generation with tool-calling simulation."""
        full_context = "\n---\n".join(context)
        prompt = LLMMockService._create_llm_prompt(history, full_context, user_query)
        logger.debug(f"Prompt sent to LLM: {prompt}")

        # --- MOCK TOOL CALLING LOGIC ---
        # Simulate LLM deciding to call the tool based on keywords
        if "book" in user_query.lower() or "schedule" in user_query.lower() or "interview" in user_query.lower():
            # Attempt to extract info using regex (mocking NLU/Tool-call parsing)
            name_match = re.search(r'name\s*[:=]\s*([\w\s]+)', user_query, re.IGNORECASE)
            email_match = re.search(r'[\w\.-]+@[\w\.-]+', user_query)
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', user_query)
            time_match = re.search(r'(\d{1,2}:\d{2}(:\d{2})?)', user_query)

            if name_match and email_match and date_match and time_match:
                try:
                    # Return a structure that signals a tool call
                    return {
                        "tool_call": True,
                        "tool_name": "interview_booking",
                        "args": {
                            "name": name_match.group(1).strip(),
                            "email": email_match.group(0),
                            "date": date_match.group(0),
                            "time": time_match.group(1),
                        }
                    }
                except Exception as e:
                    logger.error(f"Error parsing booking data: {e}")
                    pass # Fall through to text response

            # If booking intent is clear but data is missing
            if not (name_match and email_match and date_match and time_match):
                return "I can certainly help you book an interview! Could you please provide your full name, email address, the date (YYYY-MM-DD), and the time (HH:MM) you prefer?"

        # --- MOCK RAG/Standard Text Response ---
        # Simulate LLM response based on context/query
        if full_context:
            return f"Based on the provided documents, the core information related to '{user_query}' is: '{full_context[:100]}...'. I'm happy to elaborate or check for other details."
        else:
            return f"I can't find direct context for '{user_query}' in the current documents. Please upload relevant files first."


class ChunkingService:
    """Handles different document chunking strategies."""

    def __init__(self, doc_id: str, content: str):
        self.doc_id = doc_id
        self.content = content

    def _recursive_char_split(self, chunk_size=1000, chunk_overlap=200) -> List[Dict[str, str]]:
        """Strategy 1: Recursive Character Text Splitter Mock. Good for code/structured data."""
        text = self.content
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i + chunk_size]
            chunks.append({
                "id": f"{self.doc_id}-rc-{len(chunks)}",
                "text": chunk
            })
            i += chunk_size - chunk_overlap
        return chunks

    def _sentence_split(self) -> List[Dict[str, str]]:
        """Strategy 2: Sentence Splitter (split aggressively on punctuation). Good for prose."""
        # Split by periods, question marks, exclamation marks, followed by a space/newline
        sentences = re.split(r'(?<=[.?!])\s+|\n', self.content)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i, sentence in enumerate(sentences):
            chunks.append({
                "id": f"{self.doc_id}-ss-{i}",
                "text": sentence
            })
        return chunks

    def chunk(self, strategy: ChunkingStrategy) -> List[Dict[str, str]]:
        """Applies the selected chunking strategy."""
        if strategy == "recursive_char":
            return self._recursive_char_split()
        elif strategy == "sentence_split":
            return self._sentence_split()
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

# ----------------------------------------------------------------------
# 4. API Routers and Endpoints
# ----------------------------------------------------------------------

app = FastAPI(
    title="Custom RAG System Backend",
    description="Backend services for Document Ingestion and Conversational RAG with Tool Calling."
)

# Dependency to access mock services easily
def get_services() -> Dict[str, Any]:
    """Returns the initialized service mocks."""
    return {
        "db": mock_db,
        "vector_db": mock_pinecone,
        "cache": mock_redis,
        "llm": LLMMockService
    }


@app.post("/ingest/document", status_code=202, tags=["Document Ingestion API"])
async def ingest_document(
    strategy: ChunkingStrategy,
    file: UploadFile = File(...),
    services: Dict[str, Any] = Depends(get_services)
):
    """
    Document Ingestion API: Uploads a file, chunks it using a selected strategy,
    generates embeddings, and stores data in mock Vector DB and Metadata DB.
    """
    doc_id = str(hash(file.filename + str(datetime.now())))[:10]
    filename = file.filename or "unknown_file"
    file_type = filename.split('.')[-1].lower()

    if file_type not in ["pdf", "txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf and .txt are allowed.")

    try:
        # 1. Extract Text (Mock PDF/TXT parsing)
        content = (await file.read()).decode("utf-8")
        if file_type == 'pdf':
             # MOCK: In real life, use pypdf or similar to extract text from PDF
             content = f"PDF content placeholder for {filename}. Extracted text: {content[:200]}"

        # 2. Apply Chunking Strategy
        chunker = ChunkingService(doc_id, content)
        chunks = chunker.chunk(strategy)

        # 3. Generate Embeddings & Prepare for Upsert
        vectors_to_upsert = []
        for chunk in chunks:
            embedding = services['llm'].mock_embed(chunk['text'])
            vectors_to_upsert.append({
                "id": chunk['id'],
                "vector": embedding,
                "text": chunk['text'],
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "strategy": strategy,
                    "chunk_id": chunk['id']
                }
            })

        # 4. Store in Mock Vector DB
        services['vector_db'].upsert(vectors_to_upsert)

        # 5. Save Metadata in Mock PostgreSQL/NoSQL DB
        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            upload_timestamp=datetime.now(),
            chunking_strategy=strategy,
            chunk_count=len(chunks)
        )
        services['db'].save_metadata(doc_id, metadata.model_dump())

        return {
            "message": f"Document '{filename}' ingested successfully.",
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "strategy": strategy
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during ingestion: {e}")


@app.post("/chat/query", response_model=ChatResponse, tags=["Conversational RAG API"])
async def conversational_rag_query(
    request: ChatRequest,
    services: Dict[str, Any] = Depends(get_services)
):
    """
    Conversational RAG API: Handles multi-turn queries, uses Redis for memory,
    implements custom RAG, and supports interview booking (tool calling).
    """
    session_id = request.session_id
    user_query = request.query
    db = services['db']
    vector_db = services['vector_db']
    cache = services['cache']
    llm_service = services['llm']

    try:
        # 1. Retrieve Chat History (Redis for Chat Memory)
        history = cache.get_history(session_id)

        # 2. Contextualize Query (Mocked for simplicity, in real life: LLM prompt)
        # If history is long, LLM would rephrase the user_query to be context-independent.
        contextualized_query = user_query # Simple mock

        # 3. Retrieval (Custom RAG - No RetrievalQAChain)
        query_embedding = llm_service.mock_embed(contextualized_query)
        context_chunks = vector_db.query(query_embedding, top_k=3)

        # 4. LLM Generation and Tool Call Check
        llm_output = llm_service.generate_response(
            history=history,
            context=context_chunks,
            user_query=user_query
        )

        response_text = ""
        booking_conf = None
        booking_data = None

        # 5. Tool Calling Execution
        if isinstance(llm_output, dict) and llm_output.get("tool_call") and llm_output["tool_name"] == "interview_booking":
            try:
                args = llm_output['args']
                # Validate the arguments with Pydantic model
                validated_booking = BookingData(**args)
                booking_details = validated_booking.model_dump()

                # Store booking info in SQL/NoSQL DB
                booking_id = db.save_booking(booking_details)

                response_text = f"Confirmed! Your interview is booked under the name {validated_booking.name} for {validated_booking.date} at {validated_booking.time}. A confirmation email will be sent to {validated_booking.email}. Your reference ID is {booking_id}."
                booking_conf = booking_id
                booking_data = validated_booking

            except Exception as e:
                response_text = f"I tried to book the interview, but I encountered an error with the data provided: {e}. Could you please re-check your details?"
                logger.error(f"Booking validation/storage error: {e}")
        else:
            # Standard RAG response
            response_text = str(llm_output)

        # 6. Save Conversation History (for multi-turn)
        cache.save_message(session_id, "user", user_query)
        cache.save_message(session_id, "assistant", response_text)

        return ChatResponse(
            response=response_text,
            context_sources=[f"Source Chunk {i+1}" for i in range(len(context_chunks))], # Mock source citation
            booking_confirmation=booking_conf,
            booking_details=booking_data
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during RAG process.")

# ----------------------------------------------------------------------
# 5. Status Endpoint (For Monitoring)
# ----------------------------------------------------------------------

@app.get("/status", tags=["System"])
async def get_system_status(services: Dict[str, Any] = Depends(get_services)):
    """Provides a health check and mock service status."""
    return {
        "status": "Running",
        "timestamp": datetime.now(),
        "llm_model": AppConfig.LLM_MODEL,
        "mock_db_metadata_count": len(services['db'].metadata_store),
        "mock_db_booking_count": len(services['db'].booking_store),
        "mock_redis_sessions": len(services['cache']._store),
        "mock_pinecone_vector_count": len(services['vector_db'].index)
    }

# ----------------------------------------------------------------------
# 6. Example Usage and Documentation (Optional but helpful)
# ----------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Custom RAG Backend. Check /docs for API documentation."}

# Example of how to run this application:
# uvicorn app:app --reload
