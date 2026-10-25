from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import TextChunker


def test_chunker():

    loader = PDFLoader(
        "samples/Ml_book.pdf"
    )

    documents = loader.load_pdf()

    chunker = TextChunker()

    chunks = chunker.create_chunks(
        documents
    )

    assert chunks is not None

    assert len(chunks) > 0