import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.components import build_components
from src.config import deep_merge, load_config


PDF_PATH = "samples/Ml_book.pdf"


class FakeResponse:

    def __init__(self, content):
        self.content = content


class RecordingLLM:

    # Stands in for Groq: no network, and remembers every prompt it saw.
    # Give it a list of replies, or it answers "fake answer N".
    def __init__(self, replies=None):
        self.prompts = []
        self.replies = list(replies or [])

    def invoke(self, prompt):
        self.prompts.append(prompt)

        if self.replies:
            return FakeResponse(self.replies.pop(0))

        return FakeResponse(f"fake answer {len(self.prompts)}")


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding(size=32)


# Heavy or LLM-calling features are OFF in tests unless a test turns them on
TEST_DEFAULTS = {
    "retriever": {
        "rerank": {"enabled": False},
        "compression": {"enabled": False},
    },
    "query_transform": {"rewrite": False},
}


@pytest.fixture
def make_components(fake_embeddings, tmp_path):

    # make_components({"retriever": {"top_k": 2}}) -> components on a temp dir
    def factory(overrides=None, llm=None, embeddings=None, reranker=None):
        config = deep_merge(load_config(), TEST_DEFAULTS)
        config = deep_merge(config, overrides or {})

        return build_components(
            config,
            embeddings or fake_embeddings,
            llm or RecordingLLM(),
            persist_directory=str(tmp_path / "store"),
            reranker=reranker
        )

    return factory
