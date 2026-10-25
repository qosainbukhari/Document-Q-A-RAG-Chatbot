import json
from pathlib import Path

from langchain_core.documents import Document

from src.utils.logger import logger


class JSONDocStore:

    # A tiny key-value store kept in one JSON file:
    #   parents  -> the big chunks we send to the LLM
    #   children -> the small chunks we search over (also the BM25 corpus)
    # Chroma only holds child vectors; the text of parents lives here.

    def __init__(self, path):

        self.path = Path(path) if path else None

        self.parents = {}
        self.children = {}

        if self.path and self.path.exists():
            self._load()

    def _load(self):

        with open(self.path, "r") as file:
            data = json.load(file)

        self.parents = data.get("parents", {})
        self.children = data.get("children", {})

        logger.info(
            f"Docstore loaded: {len(self.parents)} parents, "
            f"{len(self.children)} children"
        )

    def save(self):

        if not self.path:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file then rename, so a crash never leaves half a JSON
        tmp_path = self.path.with_suffix(".tmp")

        with open(tmp_path, "w") as file:
            json.dump(
                {"parents": self.parents, "children": self.children},
                file
            )

        tmp_path.replace(self.path)

    @staticmethod
    def _to_record(doc):
        return {"text": doc.page_content, "metadata": doc.metadata}

    @staticmethod
    def _to_document(record):
        return Document(
            page_content=record["text"],
            metadata=dict(record["metadata"])
        )

    def replace_source(self, source, parents, children):

        # Same "re-upload replaces" rule as the vector store
        self.parents = {
            key: value for key, value in self.parents.items()
            if value["metadata"].get("source") != source
        }
        self.children = {
            key: value for key, value in self.children.items()
            if value["metadata"].get("source") != source
        }

        for parent in parents:
            self.parents[parent.metadata["parent_id"]] = self._to_record(parent)

        for child in children:
            self.children[child.metadata["chunk_id"]] = self._to_record(child)

        self.save()

    def get_parent(self, parent_id):

        record = self.parents.get(parent_id)

        return self._to_document(record) if record else None

    def all_children(self):

        return [
            self._to_document(record)
            for record in self.children.values()
        ]
