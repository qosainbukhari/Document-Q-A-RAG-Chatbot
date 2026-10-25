from src.config import DEFAULT_CONFIG_PATH, load_config
from src.observability.tracing import get_tracer
from src.retriever.fusion import reciprocal_rank_fusion
from src.utils.logger import logger


class Retriever:

    def __init__(self, vectordb, config_path=DEFAULT_CONFIG_PATH,
                 config=None, docstore=None, bm25=None, reranker=None,
                 compressor=None):

        self.vectordb = vectordb

        self.config = config or load_config(config_path)

        settings = self.config["retriever"]

        self.top_k = settings["top_k"]
        self.candidates = settings.get("candidates", self.top_k)

        hybrid = settings.get("hybrid", {})

        self.hybrid = hybrid.get("enabled", False) and bm25 is not None
        self.rrf_k = hybrid.get("rrf_k", 60)

        # Only needed for parent-child chunking
        self.docstore = docstore

        # Only needed for hybrid search
        self.bm25 = bm25

        # Optional: re-scores candidates; without it we trust search order
        self.reranker = reranker

        # Optional: drops sentences unrelated to the query
        self.compressor = compressor

        self.tracer = get_tracer()

    def search(self, query):

        # Ranked list of indexed (child) chunks for one query
        dense = self.vectordb.similarity_search(query, k=self.candidates)

        if not self.hybrid:
            return dense

        sparse = self.bm25.search(query, k=self.candidates)

        fused = reciprocal_rank_fusion([dense, sparse], k=self.rrf_k)

        return [doc for doc, _ in fused[:self.candidates]]

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

    def retrieve(self, query, search_queries=None):

        # `query` is the question used for reranking and compression.
        # `search_queries` (multi-query, HyDE, sub-questions) are each
        # searched separately and their rankings fused with RRF.
        try:
            queries = search_queries or [query]

            with self.tracer.span("search", as_type="retriever",
                                  input={"queries": queries,
                                         "hybrid": self.hybrid}) as span:

                ranked_lists = [self.search(q) for q in queries]

                if len(ranked_lists) == 1:
                    chunks = ranked_lists[0]
                else:
                    fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
                    chunks = [doc for doc, _ in fused[:self.candidates]]

                span.update(output=describe(chunks))

            with self.tracer.span("parent-mapping") as span:

                candidates = self.to_parents(chunks)

                span.update(output=describe(candidates))

            if self.reranker:

                with self.tracer.span("rerank", input={"query": query}) as span:

                    results = self.reranker.rerank(query, candidates, self.top_k)

                    span.update(output=describe(results, "rerank_score"))

            else:
                results = candidates[:self.top_k]

            if self.compressor:

                with self.tracer.span("compression") as span:

                    results = self.compressor.compress(query, results)

                    span.update(output=describe(results, "compression_ratio"))

            logger.info(
                f"Retrieved {len(results)} relevant chunks "
                f"(from {len(chunks)} candidates)"
            )

            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise


def describe(documents, score_key=None):

    # Compact view of chunks for the trace: page, score and a text preview
    return [
        {
            "page": doc.metadata.get("page"),
            **({score_key: doc.metadata.get(score_key)} if score_key else {}),
            "text": doc.page_content[:150],
        }
        for doc in documents
    ]
