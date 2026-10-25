from fastapi.testclient import TestClient

from app.main import app
from src.llm.conversation_memory import SessionMemoryStore

from tests.conftest import PDF_PATH


def test_documents_lists_uploaded_pdfs(make_components):
    app.state.components = make_components()
    app.state.memory_store = SessionMemoryStore()
    client = TestClient(app)

    assert client.get("/documents").json() == []

    app.state.components.ingestion.ingest_file(PDF_PATH)
    app.state.components.ingestion.ingest_file(PDF_PATH)  # re-upload: no duplicates

    [document] = client.get("/documents").json()

    assert document["source"] == "Ml_book.pdf"
    assert document["pages"] == 15
    assert document["chunks"] == len(app.state.components.docstore.children)
