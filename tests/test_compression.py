from langchain_core.documents import Document

from src.retriever.compressor import EmbeddingCompressor, split_sentences

from tests.conftest import PDF_PATH


class KeywordEmbeddings:

    # Fake embedding with 2 dimensions: [mentions "cat", mentions "dog"].
    # Makes similarity predictable: "cat" sentences match a "cat" query.
    def embed_query(self, text):
        return self._vector(text)

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        text = text.lower()
        return [1.0 if "cat" in text else 0.0, 1.0 if "dog" in text else 0.01]


PAGE = Document(
    page_content="Cats purr. Dogs bark loudly. A cat sleeps a lot. "
                 "Dogs love walks. Dogs dig holes.",
    metadata={"source": "pets.pdf", "page": 7, "parent_id": "p1"},
)


def test_split_sentences_handles_bullets_and_broken_lines():
    text = "Overfitting\n• Model memorizes\ndata\n• Poor performance. On new data"

    assert split_sentences(text) == [
        "Overfitting", "Model memorizes data", "Poor performance.", "On new data"
    ]


def test_keeps_only_relevant_sentences_and_page_metadata():
    compressor = EmbeddingCompressor(
        KeywordEmbeddings(), similarity_threshold=0.5, min_sentences=1
    )

    [result] = compressor.compress("tell me about the cat", [PAGE])

    assert result.page_content == "Cats purr. A cat sleeps a lot."
    assert result.metadata["page"] == 7
    assert result.metadata["source"] == "pets.pdf"
    assert result.metadata["compression_ratio"] < 1


def test_never_returns_an_empty_chunk():
    compressor = EmbeddingCompressor(
        KeywordEmbeddings(), similarity_threshold=1.5, min_sentences=2
    )

    # cosine similarity is at most 1, so no sentence passes 1.5
    [result] = compressor.compress("parrots", [PAGE])

    assert len(split_sentences(result.page_content)) == 2


def test_compression_in_full_pipeline_keeps_citations(make_components):
    components = make_components({"retriever": {"compression": {"enabled": True}}})
    components.ingestion.ingest_file(PDF_PATH)

    results = components.retriever.retrieve("What is overfitting?")

    assert len(results) == 4
    assert all(isinstance(doc.metadata["page"], int) for doc in results)
    assert all(doc.page_content for doc in results)
