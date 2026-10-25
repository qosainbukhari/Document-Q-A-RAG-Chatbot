from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.ingestion.pipeline import IngestionPipeline
from src.llm.query_transformer import QueryTransformer
from src.retriever.bm25_index import BM25Index
from src.retriever.compressor import EmbeddingCompressor
from src.retriever.reranker import CrossEncoderReranker
from src.retriever.retriever import Retriever
from src.vectordb.chroma_store import ChromaVectorStore
from src.vectordb.docstore import JSONDocStore


@dataclass
class RAGComponents:

    config: dict
    embedding_model: Any
    llm: Any
    vector_store: ChromaVectorStore
    vectordb: Any
    docstore: JSONDocStore
    bm25: Optional[BM25Index]
    reranker: Optional[CrossEncoderReranker]
    compressor: Optional[EmbeddingCompressor]
    retriever: Retriever
    query_transformer: QueryTransformer
    ingestion: IngestionPipeline


def build_components(config, embedding_model, llm, persist_directory=None,
                     reranker=None):

    # Builds every piece of the pipeline from the config, once.
    # Used by the FastAPI startup, the evaluation script and the tests.

    strategy = config["chunking"]["strategy"]

    vector_store = ChromaVectorStore(
        embedding_model,
        persist_directory=persist_directory,
        collection_name=f"rag_{strategy}"
    )

    persist_directory = vector_store.persist_directory

    vectordb = vector_store.load_vector_store()

    # Keep the docstore next to Chroma; if Chroma is in-memory, so is the docstore
    docstore = JSONDocStore(
        Path(persist_directory) / f"docstore_{strategy}.json"
        if persist_directory else None
    )

    retriever_config = config["retriever"]

    # BM25 lives in memory: rebuild it from the docstore's chunks now,
    # and again after every upload
    bm25 = None
    on_indexed = []

    if retriever_config.get("hybrid", {}).get("enabled", False):

        bm25 = BM25Index(docstore.all_children())

        on_indexed.append(lambda: bm25.rebuild(docstore.all_children()))

    # The cross-encoder is ~90 MB: load it once. Callers (e.g. the eval
    # script running several configs) may pass an already loaded one.
    rerank_config = retriever_config.get("rerank", {})

    if not rerank_config.get("enabled", False):
        reranker = None
    elif reranker is None:
        reranker = CrossEncoderReranker(rerank_config["model"])

    compression_config = retriever_config.get("compression", {})

    compressor = None

    if compression_config.get("enabled", False):

        compressor = EmbeddingCompressor(
            embedding_model,
            similarity_threshold=compression_config.get("similarity_threshold", 0.3),
            min_sentences=compression_config.get("min_sentences", 2)
        )

    retriever = Retriever(
        vectordb,
        config=config,
        docstore=docstore if strategy == "parent_child" else None,
        bm25=bm25,
        reranker=reranker,
        compressor=compressor
    )

    query_transformer = QueryTransformer(llm, config)

    ingestion = IngestionPipeline(
        config, vector_store, vectordb, docstore, on_indexed=on_indexed,
        llm=llm
    )

    return RAGComponents(
        config=config,
        embedding_model=embedding_model,
        llm=llm,
        vector_store=vector_store,
        vectordb=vectordb,
        docstore=docstore,
        bm25=bm25,
        reranker=reranker,
        compressor=compressor,
        retriever=retriever,
        query_transformer=query_transformer,
        ingestion=ingestion,
    )
