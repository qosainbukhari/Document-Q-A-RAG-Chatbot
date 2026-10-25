from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ingestion.pipeline import IngestionPipeline
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
    retriever: Retriever
    ingestion: IngestionPipeline


def build_components(config, embedding_model, llm, persist_directory=None):

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

    retriever = Retriever(
        vectordb,
        config=config,
        docstore=docstore if strategy == "parent_child" else None
    )

    ingestion = IngestionPipeline(config, vector_store, vectordb, docstore)

    return RAGComponents(
        config=config,
        embedding_model=embedding_model,
        llm=llm,
        vector_store=vector_store,
        vectordb=vectordb,
        docstore=docstore,
        retriever=retriever,
        ingestion=ingestion,
    )
