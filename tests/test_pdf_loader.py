
from src.ingestion.pdf_loader import PDFLoader


def test_pdf_loader():

    loader = PDFLoader("samples/Ml_book.pdf")

    documents = loader.load_pdf()

    assert documents is not None

    assert len(documents) > 0