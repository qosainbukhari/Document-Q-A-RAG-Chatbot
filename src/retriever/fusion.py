def doc_key(doc):

    # Child chunks have a chunk_id; fall back to the text for anything else
    return doc.metadata.get("chunk_id") or doc.page_content


def reciprocal_rank_fusion(ranked_lists, k=60, key=doc_key):

    # score(doc) = sum over lists of 1 / (k + rank), rank starting at 1.
    # Only positions are used, never the raw scores, so a cosine similarity
    # (0..1) and a BM25 score (0..30+) can be combined fairly.

    scores = {}
    documents = {}

    for ranked in ranked_lists:

        for rank, doc in enumerate(ranked, start=1):

            doc_id = key(doc)

            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

            documents.setdefault(doc_id, doc)

    fused = sorted(scores, key=scores.get, reverse=True)

    return [(documents[doc_id], scores[doc_id]) for doc_id in fused]
