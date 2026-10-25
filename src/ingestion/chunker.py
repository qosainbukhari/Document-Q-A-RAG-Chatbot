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
                 config_path="configs/config.yaml"):

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
