from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    question: str = Field(..., min_length=1)

    # Omit on the first message; reuse the session_id returned by /chat
    # to keep your own conversation history.
    session_id: Optional[str] = None
