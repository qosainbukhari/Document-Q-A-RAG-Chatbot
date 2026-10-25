import hashlib

import yaml

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import logger


def make_chunk_id(source, page, chunk_index):

    # Same (source, page, index) -> same id, so re-uploading a PDF
    # overwrites its chunks instead of duplicating them.
    key = f"{source}:{page}:{chunk_index}"

    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class TextChunker:

    def __init__(self,
                 config_path="configs/config.yaml",
                 config=None):

        if config is None:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def create_chunks(self, documents):

        try:
            chunks = []

            # Split page by page so the chunk index restarts on every page
            for document in documents:

                page_chunks = self.text_splitter.split_documents(
                    [document]
                )

                for chunk_index, chunk in enumerate(page_chunks):

                    chunk.metadata["chunk_id"] = make_chunk_id(
                        chunk.metadata.get("source"),
                        chunk.metadata.get("page"),
                        chunk_index
                    )

                    chunks.append(chunk)

            logger.info(
                f"Created {len(chunks)} chunks"
            )

            return chunks

        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise


class ParentChildChunker:

    # Two levels of chunks from every page:
    #   parent (~1500 chars) -> enough context for the LLM to answer
    #   child  (~300 chars)  -> small and focused, so its embedding is precise
    # We search over children, then hand their parents to the LLM.

    def __init__(self, config):

        settings = config["chunking"]

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings["parent_chunk_size"],
            chunk_overlap=settings["parent_chunk_overlap"]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings["child_chunk_size"],
            chunk_overlap=settings["child_chunk_overlap"]
        )

    def create_chunks(self, documents):

        parents = []
        children = []

        # Splitting page by page means every parent and child keeps its page
        for document in documents:

            source = document.metadata.get("source")
            page = document.metadata.get("page")

            page_parents = self.parent_splitter.split_documents([document])

            for parent_index, parent in enumerate(page_parents):

                parent_id = make_chunk_id(source, page, f"p{parent_index}")

                parent.metadata["parent_id"] = parent_id

                parents.append(parent)

                for child_index, child in enumerate(
                    self.child_splitter.split_documents([parent])
                ):

                    # child inherits source, page and parent_id from the parent
                    child.metadata["chunk_id"] = make_chunk_id(
                        source, page, f"p{parent_index}c{child_index}"
                    )

                    children.append(child)

        logger.info(
            f"Created {len(parents)} parent and {len(children)} child chunks"
        )

        return parents, children
