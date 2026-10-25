from tests.conftest import PDF_PATH, RecordingLLM


def test_summaries_are_indexed_and_point_to_parents(make_components):
    llm = RecordingLLM()
    components = make_components(
        {"multi_vector": {"enabled": True, "sleep_seconds": 0}}, llm=llm
    )

    stats = components.ingestion.ingest_file(PDF_PATH)

    summaries = [
        c for c in components.docstore.children.values()
        if c["metadata"].get("is_summary")
    ]

    # one LLM call and one summary per parent
    assert len(llm.prompts) == stats["parent_chunks"] == len(summaries)
    assert all(s["metadata"]["parent_id"] in components.docstore.parents for s in summaries)


def test_multi_vector_is_off_by_default(make_components):
    llm = RecordingLLM()
    components = make_components(llm=llm)

    components.ingestion.ingest_file(PDF_PATH)

    assert llm.prompts == []


def test_summary_failure_does_not_break_upload(make_components):
    class BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("rate limited")

    components = make_components(
        {"multi_vector": {"enabled": True, "sleep_seconds": 0}}, llm=BrokenLLM()
    )

    # retries are slow; make them instant for the test
    import src.llm.llm_utils as llm_utils
    original = llm_utils.time.sleep
    llm_utils.time.sleep = lambda seconds: None
    try:
        stats = components.ingestion.ingest_file(PDF_PATH)
    finally:
        llm_utils.time.sleep = original

    assert stats["total_chunks"] > 0
