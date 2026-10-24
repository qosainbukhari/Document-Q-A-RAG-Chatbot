from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import TextChunker

from src.embeddings.embedding_model import EmbeddingModel
from src.vectordb.chroma_store import ChromaVectorStore

from src.retriever.retriever import Retriever

from src.llm.groq_client import GroqClient
from src.llm.rag_chain import RAGChain


def test_rag():

    loader = PDFLoader(
        "test/Ml_book.pdf"
    )

    documents = loader.load_pdf()

    chunker = TextChunker()

    chunks = chunker.create_chunks(
        documents
    )

    embedding_model = (
        EmbeddingModel().get_model()
    )

    vectordb = ChromaVectorStore(
        embedding_model
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