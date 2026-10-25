import os

import pytest

from src.embeddings.embedding_model import EmbeddingModel
from src.ingestion.chunker import TextChunker
from src.ingestion.pdf_loader import PDFLoader
from src.llm.groq_client import GroqClient
from src.llm.rag_chain import RAGChain
from src.retriever.retriever import Retriever
from src.vectordb.chroma_store import ChromaVectorStore

from tests.conftest import PDF_PATH


# These tests use the REAL embedding model (and Groq when a key is set).
# All other test files use fakes so they run fast and offline.

needs_groq = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")


@pytest.fixture(scope="module")
def chunks():
    return TextChunker().create_chunks(PDFLoader(PDF_PATH).load_pdf())


@pytest.fixture(scope="module")
def retriever(chunks, tmp_path_factory):
    vectordb = ChromaVectorStore(
        EmbeddingModel().get_model(),
        persist_directory=str(tmp_path_factory.mktemp("chroma"))
    ).create_vector_store(chunks)

    return Retriever(vectordb)


def test_pdf_pages_keep_source_and_page_number():
    pages = PDFLoader(PDF_PATH).load_pdf()

    assert pages and all(p.page_content.strip() for p in pages)
    assert pages[0].metadata == {"source": "Ml_book.pdf", "page": 1}


def test_pdf_loader_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        PDFLoader("samples/missing.pdf").load_pdf()


def test_embeddings_are_384_numbers():
    vector = EmbeddingModel().get_model().embed_query("What is machine learning?")

    assert len(vector) == 384


def test_semantic_search_finds_the_right_page(retriever):
    pages = [doc.metadata["page"] for doc in retriever.retrieve("What is overfitting?")]

    assert len(pages) == 4
    assert {19, 22} & set(pages)


@needs_groq
def test_groq_llm_responds():
    response = GroqClient().get_llm().invoke("What is machine learning? Answer in one sentence.")

    assert response.content.strip()


@needs_groq
def test_full_rag_answer_has_sources(retriever):
    result = RAGChain(retriever, GroqClient().get_llm()).generate_answer("What is overfitting?")

    assert result["answer"].strip()
    assert all(source["source"] == "Ml_book.pdf" for source in result["sources"])
