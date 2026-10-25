import pytest
from langchain_core.documents import Document

from src.retriever.bm25_index import BM25Index, tokenize
from src.retriever.fusion import reciprocal_rank_fusion

from tests.conftest import PDF_PATH


def doc(chunk_id, text=""):
    return Document(page_content=text or chunk_id, metadata={"chunk_id": chunk_id})


def test_rrf_matches_hand_calculation():
    dense = [doc("a"), doc("b"), doc("c")]
    sparse = [doc("c"), doc("a")]

    fused = dict(
        (d.metadata["chunk_id"], score)
        for d, score in reciprocal_rank_fusion([dense, sparse], k=60)
    )

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_rewards_agreement_between_retrievers():
    dense = [doc("x"), doc("both")]
    sparse = [doc("y"), doc("both")]

    ranked = [d.metadata["chunk_id"] for d, _ in reciprocal_rank_fusion([dense, sparse])]

    # "both" is 2nd in each list but beats the two single-list winners
    assert ranked[0] == "both"


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Gradient-Descent step?") == ["gradient", "descent", "step"]


def test_bm25_finds_exact_keyword():
    index = BM25Index([
        doc("1", "neural networks learn weights"),
        doc("2", "the kernel trick in support vector machines"),
        doc("3", "decision trees split on features"),
    ])

    results = index.search("kernel trick", k=3)

    assert [d.metadata["chunk_id"] for d in results] == ["2"]


def test_empty_bm25_index_returns_nothing():
    assert BM25Index([]).search("anything", k=5) == []


def test_bm25_is_rebuilt_after_upload(make_components):
    components = make_components()

    assert components.bm25.documents == []

    components.ingestion.ingest_file(PDF_PATH)

    assert len(components.bm25.documents) == len(components.docstore.children)


def test_bm25_is_restored_from_docstore_on_restart(make_components):
    make_components().ingestion.ingest_file(PDF_PATH)

    restarted = make_components()  # same persist directory

    assert len(restarted.bm25.documents) > 0


def test_hybrid_rescues_keyword_query_when_embeddings_are_useless(make_components):
    # Fake embeddings are random, so dense search alone is pure luck;
    # BM25 still finds the page that literally contains the word.
    keyword = "overfitting"

    hybrid = make_components()
    hybrid.ingestion.ingest_file(PDF_PATH)

    results = hybrid.retriever.retrieve(keyword)

    assert keyword in results[0].page_content.lower()
