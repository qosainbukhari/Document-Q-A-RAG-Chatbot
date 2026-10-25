from fastapi import APIRouter, HTTPException, Request

from app.schemas import ChatRequest
from app.services import actions


router = APIRouter()


# Plain `def` (not async): retrieval and the LLM call are blocking,
# so FastAPI runs this in a thread pool instead of freezing the event loop.
@router.post("/chat")
def chat(
    request: Request,
    chat_request: ChatRequest
):

    try:
        return actions.ask(
            request.app.state.components,
            request.app.state.memory_store,
            chat_request.question,
            session_id=chat_request.session_id
        )

    except actions.ActionError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
