from src.llm.prompts import RAG_ANSWER_PROMPT
from src.utils.logger import logger


NO_ANSWER = "I don't know based on the document."


def format_context(documents):

    return "\n\n".join([
        f"""
Source: {doc.metadata.get("source", "Unknown")}
Page: {doc.metadata.get("page", "N/A")}

Content:
{doc.page_content}
"""
        for doc in documents
    ])


def unique_sources(documents):

    sources = []

    for doc in documents:

        source_info = {
            "source": doc.metadata.get(
                "source",
                "Unknown"
            ),
            "page": doc.metadata.get(
                "page",
                "N/A"
            )
        }

        if source_info not in sources:

            sources.append(source_info)

    return sources


class RAGChain:

    def __init__(
        self,
        retriever,
        llm,
        memory=None,
        query_transformer=None
    ):

        self.retriever = retriever
        self.llm = llm
        self.memory = memory

        # Optional: rewrites follow-ups and adds extra search queries
        self.query_transformer = query_transformer

    def retrieve(self, query, history_text=""):

        # "What are its disadvantages?" -> "What are the disadvantages of SVMs?"
        if self.query_transformer:

            transformed = self.query_transformer.transform(
                query,
                history_text
            )

        else:

            transformed = {
                "standalone_question": query,
                "search_queries": [query]
            }

        documents = self.retriever.retrieve(
            transformed["standalone_question"],
            search_queries=transformed["search_queries"]
        )

        return transformed["standalone_question"], documents

    def answer(self, query, documents, history_text=""):

        # Handle empty retrieval
        if not documents:

            logger.warning(
                "No relevant documents retrieved"
            )

            return NO_ANSWER

        prompt = RAG_ANSWER_PROMPT.format(
            history=history_text,
            context=format_context(documents),
            question=query
        )

        # Generate response
        response = self.llm.invoke(prompt)

        # Handle response format
        return (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

    def generate_answer(self, query: str, return_contexts=False):

        try:

            # Load conversation history
            history_text = ""

            if self.memory:

                history_text = (
                    self.memory.get_history()
                )

            standalone, retrieved_docs = self.retrieve(query, history_text)

            answer = self.answer(query, retrieved_docs, history_text)

            # Save conversation (but not "I don't know" from an empty search)
            if self.memory and retrieved_docs:

                self.memory.save_context(
                    query,
                    answer
                )

            logger.info(
                "Generated RAG response successfully"
            )

            result = {
                "question": query,
                "standalone_question": standalone,
                "answer": answer,
                "sources": unique_sources(retrieved_docs)
            }

            # The evaluation judges need the exact text the LLM saw
            if return_contexts:

                result["contexts"] = [
                    doc.page_content for doc in retrieved_docs
                ]
                result["pages"] = [
                    doc.metadata.get("page") for doc in retrieved_docs
                ]

            return result

        except Exception as e:

            logger.error(
                f"RAG chain failed: {e}"
            )

            raise
