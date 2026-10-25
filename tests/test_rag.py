import os

import pytest

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import TextChunker

from src.embeddings.embedding_model import EmbeddingModel
from src.vectordb.chroma_store import ChromaVectorStore

from src.retriever.retriever import Retriever

from src.llm.groq_client import GroqClient
from src.llm.rag_chain import RAGChain


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_rag(tmp_path):

    loader = PDFLoader(
        "samples/Ml_book.pdf"
    )

    documents = loader.load_pdf()

    chunker = TextChunker()

    chunks = chunker.create_chunks(
        documents
    )

    embedding_model = (
        EmbeddingModel().get_model()
    )

    # tmp_path: never write test data into the real CHROMA_DB_DIR
    vectordb = ChromaVectorStore(
        embedding_model,
        persist_directory=str(tmp_path)
    ).create_vector_store(chunks)

    retriever = Retriever(vectordb)

    llm = GroqClient().get_llm()

    rag = RAGChain(
        retriever,
        llm
    )

    response = rag.generate_answer(
        "What is this document about?"
    )

    assert response is not None

    assert "answer" in response