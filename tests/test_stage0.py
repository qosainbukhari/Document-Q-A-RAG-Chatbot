import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.main import app
from app.routes import upload as upload_route
from src.ingestion.chunker import TextChunker
from src.ingestion.pdf_loader import PDFLoader
from src.llm.conversation_memory import SessionMemoryStore
from src.vectordb.chroma_store import ChromaVectorStore


PDF_PATH = "test/Ml_book.pdf"


class FakeResponse:

    def __init__(self, content):
        self.content = content


class RecordingLLM:

    # Stands in for Groq: no network, and remembers every prompt it saw
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeResponse(f"fake answer {len(self.prompts)}")


@pytest.fixture(scope="module")
def chunks():
    documents = PDFLoader(PDF_PATH).load_pdf()
    return TextChunker().create_chunks(documents)


@pytest.fixture
def vector_store(tmp_path):
    return ChromaVectorStore(
        DeterministicFakeEmbedding(size=32),
        persist_directory=str(tmp_path / "chroma")
    )


def count(vectordb):
    return len(vectordb.get()["ids"])


# ---------- 0a. stable chunk ids ----------

def test_chunk_ids_are_stable_and_unique(chunks):
    documents = PDFLoader(PDF_PATH).load_pdf()
    again = TextChunker().create_chunks(documents)

    ids = [c.metadata["chunk_id"] for c in chunks]

    assert ids == [c.metadata["chunk_id"] for c in again]
    assert len(ids) == len(set(ids))


def test_reupload_does_not_duplicate(vector_store, chunks):
    vectordb = vector_store.create_vector_store(chunks)
    first = count(vectordb)

    vector_store.add_chunks(vectordb, chunks)

    assert first > 0
    assert count(vectordb) == first


def test_reupload_removes_stale_chunks(vector_store, chunks):
    vectordb = vector_store.create_vector_store(chunks)

    # New version of the same file that only has its first two pages
    smaller = [c for c in chunks if c.metadata["page"] <= 2]
    stored = vector_store.add_chunks(vectordb, smaller)

    assert count(vectordb) == stored
    assert stored < len(chunks)


# ---------- 0c. memory per session ----------

def test_sessions_have_separate_memory():
    store = SessionMemoryStore(window_size=5)

    store.get("alice").save_context("What is SVM?", "A classifier.")

    assert "SVM" in store.get("alice").get_history()
    assert store.get("bob").get_history() == ""


def test_least_recently_used_session_is_evicted():
    store = SessionMemoryStore(window_size=5, max_sessions=2)

    store.get("a").save_context("q", "a")
    store.get("b")
    store.get("a")  # touch "a" so "b" becomes the oldest
    store.get("c")

    assert list(store.sessions) == ["a", "c"]


# ---------- API: status codes + sessions ----------

@pytest.fixture
def client(vector_store, monkeypatch, tmp_path):
    # TestClient without `with` skips startup, so no real models are loaded
    monkeypatch.setattr(upload_route, "UPLOAD_DIR", str(tmp_path / "raw"))

    app.state.vector_store = vector_store
    app.state.vectordb = vector_store.load_vector_store()
    app.state.llm = RecordingLLM()
    app.state.memory_store = SessionMemoryStore(window_size=5)

    return TestClient(app)


def upload(client, name="Ml_book.pdf", data=None):
    if data is None:
        with open(PDF_PATH, "rb") as f:
            data = f.read()

    return client.post(
        "/upload",
        files={"file": (name, data, "application/pdf")}
    )


def test_upload_rejects_non_pdf(client):
    response = upload(client, name="notes.txt", data=b"hello")

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_corrupt_pdf(client):
    response = upload(client, name="broken.pdf", data=b"not a pdf")

    assert response.status_code == 422


def test_upload_twice_keeps_same_chunk_count(client):
    first = upload(client)
    second = upload(client)

    assert first.status_code == 200
    assert second.status_code == 200
    assert count(app.state.vectordb) == first.json()["total_chunks"]


def test_chat_rejects_empty_question(client):
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422


def test_chat_returns_500_when_llm_fails(client):
    upload(client)

    class BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("groq is down")

    app.state.llm = BrokenLLM()

    response = client.post("/chat", json={"question": "What is ML?"})

    assert response.status_code == 500
    assert "groq" not in response.json()["detail"]


def test_chat_history_is_per_session(client):
    upload(client)
    llm = app.state.llm

    alice = client.post("/chat", json={"question": "alice question one"})
    session_id = alice.json()["session_id"]

    client.post(
        "/chat",
        json={"question": "alice question two", "session_id": session_id}
    )

    # No session_id -> brand new session with empty history
    bob = client.post("/chat", json={"question": "bob question"})

    assert alice.status_code == 200
    assert bob.json()["session_id"] != session_id

    # Alice's second prompt saw her first question; Bob's prompt did not
    assert "alice question one" in llm.prompts[1]
    assert "alice question" not in llm.prompts[2]
