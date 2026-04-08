from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import requests

from src.config.settings import settings
from src.conversation.memory import ConversationMemory
from src.retrieval.rag_engine import RAGEngine
from src.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()

rag = RAGEngine(settings.raw_docs_dir)
memory = ConversationMemory()


@router.get("/")
async def health():
    return {"message": "Pakistan Legal RAG Chatbot is running"}


@router.get("/health/ollama")
async def ollama_health():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama not reachable: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or "default"
        history = memory.get_history(session_id)

        result = rag.generate(request.query, history=history)
        memory.add_message(session_id, request.query, result["answer"])

        return ChatResponse(
            answer=result["answer"],
            relevant_context=result["relevant_context"],
            metadata=result["metadata"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/context")
async def chat_context(request: ChatRequest):
    try:
        result = rag.get_context(request.query, top_k=3)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        session_id = request.session_id or "default"
        history = memory.get_history(session_id)

        def event_generator():
            full_answer = []

            for token in rag.stream_generate(request.query, history=history):
                full_answer.append(token)
                yield token

            memory.add_message(session_id, request.query, "".join(full_answer))

        return StreamingResponse(event_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/clear")
async def clear_chat(request: ChatRequest):
    session_id = request.session_id or "default"
    memory.clear_history(session_id)
    return {"message": "Conversation history cleared"}