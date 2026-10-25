from src.config import load_config
from src.ingestion.chunker import ParentChildChunker
from src.ingestion.pdf_loader import PDFLoader
from src.vectordb.docstore import JSONDocStore

from tests.conftest import PDF_PATH


def load_pages():
    return PDFLoader(PDF_PATH).load_pdf()


def test_children_are_small_and_point_to_parents():
    config = load_config()
    parents, children = ParentChildChunker(config).create_chunks(load_pages())

    parent_ids = {p.metadata["parent_id"] for p in parents}
    settings = config["chunking"]

    assert len(children) > len(parents) > 0
    assert all(len(c.page_content) <= settings["child_chunk_size"] for c in children)
    assert all(len(p.page_content) <= settings["parent_chunk_size"] for p in parents)
    assert all(c.metadata["parent_id"] in parent_ids for c in children)


def test_child_keeps_the_page_of_its_parent():
    parents, children = ParentChildChunker(load_config()).create_chunks(load_pages())

    page_of = {p.metadata["parent_id"]: p.metadata["page"] for p in parents}

    assert all(c.metadata["page"] == page_of[c.metadata["parent_id"]] for c in children)


def test_docstore_persists_and_replaces_source(tmp_path):
    parents, children = ParentChildChunker(load_config()).create_chunks(load_pages())
    path = tmp_path / "docstore.json"

    store = JSONDocStore(path)
    store.replace_source("Ml_book.pdf", parents, children)
    store.replace_source("Ml_book.pdf", parents[:1], children[:2])

    reloaded = JSONDocStore(path)
    parent_id = parents[0].metadata["parent_id"]

    assert len(reloaded.parents) == 1
    assert len(reloaded.all_children()) == 2
    assert reloaded.get_parent(parent_id).page_content == parents[0].page_content


def test_retrieval_returns_unique_parents_with_pages(make_components):
    components = make_components()
    components.ingestion.ingest_file(PDF_PATH)

    results = components.retriever.retrieve("What is machine learning?")
    parent_ids = [doc.metadata["parent_id"] for doc in results]

    assert len(results) == components.config["retriever"]["top_k"]
    assert len(parent_ids) == len(set(parent_ids))
    assert all(isinstance(doc.metadata["page"], int) for doc in results)
    # Parents are bigger than the children we searched over
    assert max(len(doc.page_content) for doc in results) > 300


def test_naive_strategy_returns_chunks_directly(make_components):
    components = make_components({"chunking": {"strategy": "naive"}})
    components.ingestion.ingest_file(PDF_PATH)

    results = components.retriever.retrieve("What is machine learning?")

    assert len(results) == 4
    assert all("parent_id" not in doc.metadata for doc in results)
