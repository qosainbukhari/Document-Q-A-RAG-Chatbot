import uuid

from fastapi import APIRouter, HTTPException, Request

from app.schemas import ChatRequest
from app.services.rag_service import RAGService

from src.utils.logger import logger


router = APIRouter()


# Plain `def` (not async): retrieval and the LLM call are blocking,
# so FastAPI runs this in a thread pool instead of freezing the event loop.
@router.post("/chat")
def chat(
    request: Request,
    chat_request: ChatRequest
):

    session_id = chat_request.session_id or str(uuid.uuid4())

    try:

        # Access globally loaded resources
        components = request.app.state.components

        memory = request.app.state.memory_store.get(session_id)

        # Create service
        rag_service = RAGService(
            components,
            memory
        )

        # Generate response
        response = rag_service.ask_question(
            chat_request.question,
            session_id=session_id
        )

    except Exception as e:

        logger.error(f"/chat failed: {e}")

        # Groq's free tier allows only so many tokens per minute
        if getattr(e, "status_code", None) == 429:

            raise HTTPException(
                status_code=429,
                detail="The LLM rate limit was reached. Please wait a few seconds and try again."
            )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer"
        )

    response["session_id"] = session_id

    return response
