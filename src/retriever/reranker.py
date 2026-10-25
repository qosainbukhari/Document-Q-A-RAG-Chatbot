import time

from sentence_transformers import CrossEncoder

from src.utils.logger import logger


class CrossEncoderReranker:

    # A cross-encoder reads (query, passage) TOGETHER and outputs one
    # relevance score. Far more accurate than comparing two separate
    # embeddings, but it must run once per candidate, so we only use it
    # on a short list (~20) that a fast retriever already found.

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):

        self.model = CrossEncoder(model_name, device="cpu")

        logger.info(f"Reranker loaded: {model_name}")

    def rerank(self, query, documents, top_k):

        if not documents:
            return []

        start = time.perf_counter()

        scores = self.model.predict(
            [(query, doc.page_content) for doc in documents]
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda pair: pair[1],
            reverse=True
        )

        logger.info(
            f"Reranked {len(documents)} candidates in "
            f"{(time.perf_counter() - start) * 1000:.0f} ms"
        )

        results = []

        for doc, score in ranked[:top_k]:

            doc.metadata["rerank_score"] = float(score)

            results.append(doc)

        return results
