from src.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.logger import logger


class Retriever:

    def __init__(self, vectordb, config_path=DEFAULT_CONFIG_PATH,
                 config=None, docstore=None):

        self.vectordb = vectordb

        self.config = config or load_config(config_path)

        settings = self.config["retriever"]

        self.top_k = settings["top_k"]
        self.candidates = settings.get("candidates", self.top_k)

        # Only needed for parent-child chunking
        self.docstore = docstore

    def search(self, query):

        # Ranked list of indexed (child) chunks for one query
        return self.vectordb.similarity_search(query, k=self.candidates)

    def to_parents(self, chunks):

        # Many children can come from the same parent: keep each parent
        # once, at the rank of its best child. Without a docstore (naive
        # chunking) the chunk itself is what we send to the LLM.
        if self.docstore is None:
            return chunks

        parents = []
        seen = set()

        for chunk in chunks:

            parent_id = chunk.metadata.get("parent_id")

            if parent_id is None:
                parents.append(chunk)
                continue

            if parent_id in seen:
                continue

            parent = self.docstore.get_parent(parent_id)

            if parent is None:
                logger.warning(f"Parent {parent_id} missing from docstore")
                parents.append(chunk)
                continue

            seen.add(parent_id)
            parents.append(parent)

        return parents

    def retrieve(self, query):

        try:
            chunks = self.search(query)

            results = self.to_parents(chunks)[:self.top_k]

            logger.info(
                f"Retrieved {len(results)} relevant chunks "
                f"(from {len(chunks)} candidates)"
            )

            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
