from contextlib import contextmanager

from src.llm.rag_chain import RAGChain
from src.observability.tracing import Tracer

from tests.conftest import PDF_PATH, RecordingLLM


class FakeObservation:

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def update(self, **fields):
        self.log.append((self.name, "update", sorted(fields)))


class FakeLangfuse:

    # Records what our Tracer asks Langfuse to do
    def __init__(self, fail=False):
        self.log = []
        self.fail = fail

    @contextmanager
    def start_as_current_observation(self, name, as_type, input=None):
        if self.fail:
            raise RuntimeError("langfuse is down")
        self.log.append((name, "start", as_type))
        yield FakeObservation(name, self.log)
        self.log.append((name, "end", as_type))

    def flush(self):
        self.log.append(("flush",))


def make_tracer(client):
    tracer = Tracer()
    tracer.client = client
    tracer.propagate_attributes = lambda **kwargs: contextmanager(lambda: (yield))()
    return tracer


def test_tracing_is_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    tracer = Tracer()

    assert not tracer.enabled
    with tracer.span("step") as span:
        span.update(output="ignored")  # must not raise


def test_spans_record_latency_and_nest():
    client = FakeLangfuse()
    tracer = make_tracer(client)

    with tracer.span("outer"):
        with tracer.span("inner", as_type="retriever"):
            pass

    assert [entry[:2] for entry in client.log if entry[1] in ("start", "end")] == [
        ("outer", "start"), ("inner", "start"), ("inner", "end"), ("outer", "end")
    ]
    assert ("inner", "update", ["metadata"]) in client.log


def test_broken_langfuse_never_breaks_the_app():
    tracer = make_tracer(FakeLangfuse(fail=True))

    with tracer.span("step") as span:
        span.update(output="x")
        result = 42

    assert result == 42


def test_full_question_is_traced_step_by_step(make_components):
    components = make_components({
        "retriever": {"compression": {"enabled": True}},
        "query_transform": {"rewrite": True},
    })
    components.ingestion.ingest_file(PDF_PATH)

    client = FakeLangfuse()
    tracer = make_tracer(client)

    chain = RAGChain(components.retriever, RecordingLLM(), None,
                     query_transformer=components.query_transformer)
    chain.tracer = tracer
    components.retriever.tracer = tracer

    chain.generate_answer("What is overfitting?", session_id="abc")

    started = [entry[0] for entry in client.log if entry[1] == "start"]

    assert started == [
        "rag-query", "query-transform", "search", "parent-mapping",
        "compression", "generation",
    ]
