import re

from rank_bm25 import BM25Okapi

from src.utils.logger import logger


# Very common words carry no meaning for keyword search
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to",
    "in", "on", "for", "and", "or", "with", "as", "by", "at", "it", "its",
    "this", "that", "these", "those", "what", "which", "how", "why", "do",
    "does", "can", "from", "into", "about", "than", "then", "there", "their",
}


def tokenize(text):

    # "Gradient-Descent's step" -> ["gradient", "descent", "step"]
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    return [token for token in tokens if token not in STOPWORDS]


class BM25Index:

    # Keyword (sparse) search over the child chunks.
    # rank_bm25 keeps everything in memory, so we simply rebuild the
    # whole index from the docstore at startup and after every upload.

    def __init__(self, documents=None):

        self.documents = []
        self.bm25 = None

        self.rebuild(documents or [])

    def rebuild(self, documents):

        self.documents = list(documents)

        if self.documents:

            corpus = [tokenize(doc.page_content) for doc in self.documents]

            self.bm25 = BM25Okapi(corpus)

        else:
            self.bm25 = None

        logger.info(f"BM25 index built over {len(self.documents)} chunks")

    def search(self, query, k):

        if self.bm25 is None:
            return []

        tokens = tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        # A score of 0 means no query word appears in the chunk at all
        return [
            self.documents[index]
            for index in ranked[:k]
            if scores[index] > 0
        ]
