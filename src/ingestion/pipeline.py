from src.ingestion.chunker import ParentChildChunker, TextChunker
from src.ingestion.pdf_loader import PDFLoader

from src.utils.logger import logger


class IngestionPipeline:

    # PDF -> pages -> chunks -> (Chroma + docstore).
    # The API, the evaluation script and the tests all ingest through here.

    def __init__(self, config, vector_store, vectordb, docstore,
                 on_indexed=None):

        self.config = config
        self.vector_store = vector_store
        self.vectordb = vectordb
        self.docstore = docstore

        # Called after every ingest, e.g. to rebuild the BM25 index
        self.on_indexed = on_indexed or []

        self.strategy = config["chunking"]["strategy"]

    def chunk(self, documents):

        if self.strategy == "parent_child":
            return ParentChildChunker(self.config).create_chunks(documents)

        # Naive: there are no parents, each chunk is sent to the LLM as is
        return [], TextChunker(config=self.config).create_chunks(documents)

    def ingest_documents(self, documents):

        parents, children = self.chunk(documents)

        stored = self.vector_store.add_chunks(self.vectordb, children)

        sources = {doc.metadata.get("source") for doc in documents}

        for source in sources:

            self.docstore.replace_source(
                source,
                [p for p in parents if p.metadata.get("source") == source],
                [c for c in children if c.metadata.get("source") == source]
            )

        for callback in self.on_indexed:
            callback()

        logger.info(
            f"Ingested {len(documents)} pages: "
            f"{len(parents)} parents, {stored} indexed chunks"
        )

        return {
            "pages": len(documents),
            "parent_chunks": len(parents),
            "total_chunks": stored,
        }

    def ingest_file(self, file_path):

        return self.ingest_documents(PDFLoader(file_path).load_pdf())
