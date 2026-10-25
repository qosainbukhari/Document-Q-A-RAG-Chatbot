from src.llm.query_transformer import QueryTransformer, parse_lines
from src.llm.rag_chain import RAGChain
from src.llm.conversation_memory import ConversationMemory

from tests.conftest import PDF_PATH, RecordingLLM


def config(**flags):
    return {"query_transform": {"rewrite": False, **flags}}


def test_parse_lines_strips_numbering_and_bullets():
    text = "1. What is ML?\n- Define machine learning\n\n* ML meaning\n4) extra"

    assert parse_lines(text, 3) == ["What is ML?", "Define machine learning", "ML meaning"]


def test_rewrite_is_skipped_without_history():
    llm = RecordingLLM()
    transformer = QueryTransformer(llm, config(rewrite=True))

    result = transformer.transform("What is overfitting?", history="")

    assert result["standalone_question"] == "What is overfitting?"
    assert llm.prompts == []  # no Groq call wasted


def test_rewrite_uses_history_to_make_standalone_question():
    llm = RecordingLLM(replies=["What are the disadvantages of decision trees?"])
    transformer = QueryTransformer(llm, config(rewrite=True))

    result = transformer.transform(
        "What are its disadvantages?",
        history="User: What is a decision tree?\nAssistant: A tree of rules."
    )

    assert result["standalone_question"] == "What are the disadvantages of decision trees?"
    assert "What is a decision tree?" in llm.prompts[0]


def test_multi_query_hyde_and_decomposition_add_search_queries():
    llm = RecordingLLM(replies=[
        "1. q variant one\n2. q variant two\n3. q variant three",
        "A hypothetical passage about the question.",
        "sub question a\nsub question b",
    ])
    transformer = QueryTransformer(
        llm, config(multi_query=True, hyde=True, decomposition=True, num_queries=3)
    )

    result = transformer.transform("original question")

    assert result["search_queries"] == [
        "original question",
        "q variant one", "q variant two", "q variant three",
        "A hypothetical passage about the question.",
        "sub question a", "sub question b",
    ]
    assert len(llm.prompts) == 3  # one call per enabled technique


def test_failed_expansion_falls_back_to_original_question():
    class BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("rate limited")

    transformer = QueryTransformer(BrokenLLM(), config(multi_query=True))

    assert transformer.transform("q")["search_queries"] == ["q"]


def test_retriever_fuses_several_search_queries(make_components):
    components = make_components()
    components.ingestion.ingest_file(PDF_PATH)

    results = components.retriever.retrieve(
        "overfitting", search_queries=["overfitting", "validation set", "applications"]
    )

    assert 0 < len(results) <= 4


def test_follow_up_question_is_rewritten_before_retrieval(make_components):
    llm = RecordingLLM(replies=["What are the problems caused by overfitting?", "the answer"])
    components = make_components({"query_transform": {"rewrite": True}}, llm=llm)
    components.ingestion.ingest_file(PDF_PATH)

    memory = ConversationMemory()
    memory.save_context("What is overfitting?", "When a model memorizes data.")

    chain = RAGChain(
        components.retriever, llm, memory,
        query_transformer=components.query_transformer
    )
    result = chain.generate_answer("What problems does it cause?")

    assert result["standalone_question"] == "What are the problems caused by overfitting?"
    assert result["answer"] == "the answer"
    # The answer prompt keeps the user's own words plus the history
    assert "What problems does it cause?" in llm.prompts[1]
