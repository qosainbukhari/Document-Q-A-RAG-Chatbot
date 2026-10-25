# Retrieval metrics in plain Python. Input is the list of page numbers of
# the retrieved chunks, in rank order, and the set of relevant pages.


def recall_at_k(retrieved_pages, relevant_pages, k):

    # 1.0 if ANY relevant page is in the top k results, else 0.0
    # (often called "hit rate"; averaged over questions it is Recall@K)
    top_k = retrieved_pages[:k]

    return 1.0 if any(page in relevant_pages for page in top_k) else 0.0


def reciprocal_rank(retrieved_pages, relevant_pages):

    # 1 / position of the first relevant result; 0 if none is relevant.
    # Averaged over questions this is MRR (Mean Reciprocal Rank).
    for rank, page in enumerate(retrieved_pages, start=1):

        if page in relevant_pages:
            return 1.0 / rank

    return 0.0


def mean(values):

    values = [value for value in values if value is not None]

    return sum(values) / len(values) if values else None
