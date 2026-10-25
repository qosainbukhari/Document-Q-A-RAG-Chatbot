from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import TextChunker

from src.embeddings.embedding_model import EmbeddingModel
from src.vectordb.chroma_store import ChromaVectorStore


def test_vectordb(tmp_path):

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

    assert vectordb is not None