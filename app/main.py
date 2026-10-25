from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

# Read GROQ_API_KEY, CHROMA_DB_DIR, ... from .env before anything else
load_dotenv()

from app.core.startup import load_models
from src.observability.tracing import get_tracer

from app.routes.upload import router as upload_router
from app.routes.chat import router as chat_router
from app.routes.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup: load models and build the RAG pipeline once
    load_models(app)

    yield

    # Shutdown: send any traces still waiting in the Langfuse buffer
    get_tracer().flush()


app = FastAPI(
    title="Document Q&A RAG Chatbot API",
    description="Upload PDFs and ask questions with advanced RAG and page citations.",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():

    return {
        "status": "ok"
    }


# Register routes
app.include_router(upload_router)

app.include_router(chat_router)

app.include_router(documents_router)
