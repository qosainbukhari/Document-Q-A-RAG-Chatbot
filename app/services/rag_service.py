from src.llm.rag_chain import RAGChain


class RAGService:

    def __init__(
        self,
        components,
        memory=None
    ):

        self.rag_chain = RAGChain(
            components.retriever,
            components.llm,
            memory,
            query_transformer=components.query_transformer
        )

    def ask_question(
        self,
        question: str
    ):

        response = (
            self.rag_chain.generate_answer(
                question
            )
        )

        return response
