import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pypdf.errors import PdfReadError

from src.ingestion.pdf_loader import PDFLoader

from src.utils.logger import logger


router = APIRouter()

UPLOAD_DIR = "data/raw"


@router.post("/upload")
def upload_pdf(
    request: Request,
    file: UploadFile = File(...)
):

    # Keep only the base name so "../../x.pdf" can't escape UPLOAD_DIR
    filename = Path(file.filename or "").name

    # Validate extension
    if not filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed"
        )

    try:

        Path(UPLOAD_DIR).mkdir(
            parents=True,
            exist_ok=True
        )

        # Save uploaded file
        file_path = f"{UPLOAD_DIR}/{filename}"

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # Load PDF
        loader = PDFLoader(file_path)

        documents = loader.load_pdf()

    except (ValueError, PdfReadError) as e:

        # Client's fault: corrupt PDF, or scanned PDF with no extractable text
        raise HTTPException(
            status_code=422,
            detail=f"Could not read PDF: {e}"
        )

    except Exception as e:

        logger.error(f"/upload failed while reading PDF: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to read PDF"
        )

    try:

        # Chunk + index (replaces old chunks of this file)
        stats = request.app.state.components.ingestion.ingest_documents(
            documents
        )

    except Exception as e:

        logger.error(f"/upload failed while indexing: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to index PDF"
        )

    return {
        "message": "PDF uploaded successfully",
        "filename": filename,
        **stats
    }
