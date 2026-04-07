import json

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

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


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        session_id = request.session_id or "default"
        history = memory.get_history(session_id)

        result = rag.generate(request.query, history=history)
        relevant_context = result["relevant_context"]
        answer = result["answer"]

        def event_generator():
            yield f"event: context\ndata: {json.dumps(relevant_context)}\n\n"
            yield f"event: meta\ndata: {json.dumps(result['metadata'])}\n\n"

            for char in answer:
                yield f"event: token\ndata: {json.dumps(char)}\n\n"

            memory.add_message(session_id, request.query, answer)
            yield 'event: done\ndata: "done"\n\n'

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/clear")
async def clear_chat(request: ChatRequest):
    session_id = request.session_id or "default"
    memory.clear_history(session_id)
    return {"message": "Conversation history cleared"}