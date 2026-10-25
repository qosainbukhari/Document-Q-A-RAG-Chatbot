from dotenv import load_dotenv
from fastapi import FastAPI

# Read GROQ_API_KEY, CHROMA_DB_DIR, ... from .env before anything else
load_dotenv()

from app.core.startup import load_models
from src.observability.tracing import get_tracer

from app.routes.upload import router as upload_router
from app.routes.chat import router as chat_router


app = FastAPI(
    title="RAG Chatbot API"
)


@app.on_event("startup")
async def startup_event():

    load_models(app)


@app.on_event("shutdown")
async def shutdown_event():

    # Send any traces still waiting in the Langfuse buffer
    get_tracer().flush()


@app.get("/health")
async def health_check():

    return {
        "status": "ok"
    }


# Register routes
app.include_router(upload_router)

app.include_router(chat_router)