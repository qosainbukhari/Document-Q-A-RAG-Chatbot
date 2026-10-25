import re

from src.llm.prompts import (
    DECOMPOSITION_PROMPT,
    HYDE_PROMPT,
    MULTI_QUERY_PROMPT,
    QUERY_REWRITE_PROMPT,
)
from src.utils.logger import logger


def llm_text(llm, prompt):

    response = llm.invoke(prompt)

    text = response.content if hasattr(response, "content") else str(response)

    return text.strip()


def parse_lines(text, limit):

    # "1. foo\n- bar\n\nbaz" -> ["foo", "bar", "baz"]
    lines = []

    for line in text.splitlines():

        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()

        if line:
            lines.append(line)

    return lines[:limit]


class QueryTransformer:

    # Turns the user's question into (1) a standalone question for the
    # answer prompt and reranker, and (2) a list of search queries.
    # Every enabled technique costs one extra LLM call.

    def __init__(self, llm, config):

        self.llm = llm

        settings = config.get("query_transform", {})

        self.rewrite = settings.get("rewrite", True)
        self.multi_query = settings.get("multi_query", False)
        self.hyde = settings.get("hyde", False)
        self.decomposition = settings.get("decomposition", False)
        self.num_queries = settings.get("num_queries", 3)

    def rewrite_question(self, question, history):

        # No history -> nothing to resolve, so skip the LLM call
        if not self.rewrite or not history.strip():
            return question

        rewritten = llm_text(
            self.llm,
            QUERY_REWRITE_PROMPT.format(history=history, question=question)
        )

        return rewritten.splitlines()[0].strip() if rewritten else question

    def transform(self, question, history=""):

        standalone = self.rewrite_question(question, history)

        search_queries = [standalone]

        try:

            if self.multi_query:
                search_queries += parse_lines(
                    llm_text(self.llm, MULTI_QUERY_PROMPT.format(
                        n=self.num_queries, question=standalone
                    )),
                    self.num_queries
                )

            if self.hyde:
                search_queries.append(
                    llm_text(self.llm, HYDE_PROMPT.format(question=standalone))
                )

            if self.decomposition:
                search_queries += parse_lines(
                    llm_text(self.llm, DECOMPOSITION_PROMPT.format(
                        n=self.num_queries, question=standalone
                    )),
                    self.num_queries
                )

        except Exception as e:

            # Extra queries are a bonus: if the LLM fails, search with what we have
            logger.warning(f"Query expansion failed, continuing: {e}")

        # Drop duplicates but keep order
        search_queries = list(dict.fromkeys(search_queries))

        logger.info(f"Standalone question: {standalone!r}; "
                    f"{len(search_queries)} search queries")

        return {
            "standalone_question": standalone,
            "search_queries": search_queries,
        }
