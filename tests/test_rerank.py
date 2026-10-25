import pytest
from langchain_core.documents import Document

from src.retriever.reranker import CrossEncoderReranker

from tests.conftest import PDF_PATH


class LengthReranker:

    # Fake cross-encoder: "most relevant" = longest text, easy to predict
    def rerank(self, query, documents, top_k):
        ranked = sorted(documents, key=lambda d: len(d.page_content), reverse=True)
        return ranked[:top_k]


@pytest.fixture(scope="module")
def cross_encoder():
    return CrossEncoderReranker()


def test_retriever_uses_reranker_order(make_components):
    components = make_components(
        {"retriever": {"rerank": {"enabled": True}}},
        reranker=LengthReranker()
    )
    components.ingestion.ingest_file(PDF_PATH)

    results = components.retriever.retrieve("machine learning")
    lengths = [len(d.page_content) for d in results]

    assert len(results) == 4
    assert lengths == sorted(lengths, reverse=True)


def test_cross_encoder_puts_relevant_passage_first(cross_encoder):
    passages = [
        Document(page_content="The capital of France is Paris.", metadata={"page": 1}),
        Document(page_content="Overfitting happens when a model memorises the "
                              "training data and fails to generalise.", metadata={"page": 2}),
        Document(page_content="Bananas are rich in potassium.", metadata={"page": 3}),
    ]

    results = cross_encoder.rerank("What is overfitting?", passages, top_k=2)

    assert len(results) == 2
    assert results[0].metadata["page"] == 2
    assert results[0].metadata["rerank_score"] > results[1].metadata["rerank_score"]


def test_full_retrieval_with_real_reranker(make_components, cross_encoder):
    components = make_components(
        {"retriever": {"rerank": {"enabled": True}}},
        reranker=cross_encoder
    )
    components.ingestion.ingest_file(PDF_PATH)

    results = components.retriever.retrieve("What is overfitting?")
    scores = [d.metadata["rerank_score"] for d in results]

    assert len(results) == 4
    assert scores == sorted(scores, reverse=True)
