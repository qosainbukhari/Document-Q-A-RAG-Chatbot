import re

import numpy as np
from langchain_core.documents import Document


def split_sentences(text):

    # Slides are mostly bullet points, so every bullet is its own unit.
    # Inside a bullet, PDF text breaks lines mid-sentence: join the lines,
    # then end a sentence at . ! or ? followed by whitespace.
    sentences = []

    for part in re.split(r"[•▪●]|\n\s*[-*]\s+", text):

        part = re.sub(r"\s*\n\s*", " ", part).strip()

        sentences += [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", part)
            if sentence.strip()
        ]

    return sentences


def cosine_similarity(query_vector, sentence_vectors):

    query_vector = np.asarray(query_vector)
    sentence_vectors = np.asarray(sentence_vectors)

    norms = (
        np.linalg.norm(sentence_vectors, axis=1)
        * np.linalg.norm(query_vector)
    )

    return sentence_vectors @ query_vector / np.maximum(norms, 1e-10)


class EmbeddingCompressor:

    # Keeps only the sentences of each chunk that are similar enough to
    # the query. No LLM call: one batch embedding call per chunk.

    def __init__(self, embedding_model, similarity_threshold=0.3,
                 min_sentences=2):

        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences

    def compress_document(self, query_vector, doc):

        sentences = split_sentences(doc.page_content)

        if len(sentences) <= self.min_sentences:
            return doc

        similarities = cosine_similarity(
            query_vector,
            self.embedding_model.embed_documents(sentences)
        )

        keep = {
            index for index, similarity in enumerate(similarities)
            if similarity >= self.similarity_threshold
        }

        # Never return an empty chunk: keep at least the best few sentences
        if len(keep) < self.min_sentences:
            best = np.argsort(similarities)[::-1][:self.min_sentences]
            keep.update(int(index) for index in best)

        # Original order keeps the text readable
        kept_text = " ".join(
            sentences[index] for index in sorted(keep)
        )

        # Metadata (source, page, ...) is copied unchanged -> citations survive
        metadata = dict(doc.metadata)
        metadata["compression_ratio"] = round(
            len(kept_text) / max(len(doc.page_content), 1), 2
        )

        return Document(page_content=kept_text, metadata=metadata)

    def compress(self, query, documents):

        if not documents:
            return []

        query_vector = self.embedding_model.embed_query(query)

        return [
            self.compress_document(query_vector, doc)
            for doc in documents
        ]
