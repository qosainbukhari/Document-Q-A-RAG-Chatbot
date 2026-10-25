import shutil
import uuid
from pathlib import Path

from pypdf.errors import PdfReadError

from app.services.rag_service import RAGService
from src.ingestion.pdf_loader import PDFLoader

from src.utils.logger import logger


class ActionError(Exception):

    # An error meant for the user: an HTTP-style status code + a message.
    # FastAPI turns it into an HTTP response, Streamlit shows the message.
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def upload_pdf(components, filename, file, upload_dir="data/raw"):

    # Keep only the base name so "../../x.pdf" can't escape upload_dir
    filename = Path(filename or "").name

    # Validate extension
    if not filename.lower().endswith(".pdf"):
        raise ActionError(400, "Only PDF files allowed")

    try:

        Path(upload_dir).mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = Path(upload_dir) / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)

        documents = PDFLoader(str(file_path)).load_pdf()

    except (ValueError, PdfReadError) as e:

        # User's fault: corrupt PDF, or scanned PDF with no extractable text
        raise ActionError(422, f"Could not read PDF: {e}")

    except Exception as e:

        logger.error(f"Upload failed while reading PDF: {e}")
        raise ActionError(500, "Failed to read PDF")

    try:

        # Chunk + index (replaces old chunks of this file)
        stats = components.ingestion.ingest_documents(documents)

    except Exception as e:

        logger.error(f"Upload failed while indexing: {e}")
        raise ActionError(500, "Failed to index PDF")

    return {
        "message": "PDF uploaded successfully",
        "filename": filename,
        **stats
    }


def ask(components, memory_store, question, session_id=None):

    session_id = session_id or str(uuid.uuid4())

    try:

        memory = memory_store.get(session_id)

        response = RAGService(components, memory).ask_question(
            question,
            session_id=session_id
        )

    except Exception as e:

        logger.error(f"Chat failed: {e}")

        # Groq's free tier allows only so many tokens per minute / day
        if getattr(e, "status_code", None) == 429:
            raise ActionError(
                429,
                "The LLM rate limit was reached. Please wait a few seconds and try again."
            )

        raise ActionError(500, "Failed to generate an answer")

    response["session_id"] = session_id

    return response
