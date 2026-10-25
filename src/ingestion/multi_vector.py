import time

from langchain_core.documents import Document

from src.ingestion.chunker import make_chunk_id
from src.llm.llm_utils import call_with_retry
from src.llm.prompts import CHUNK_SUMMARY_PROMPT
from src.utils.logger import logger


class SummaryIndexer:

    # Multi-vector retrieval: a parent can be found through several
    # vectors -- its children AND an LLM-written summary of it.
    # Costs one LLM call per parent at upload time.

    def __init__(self, llm, sleep_seconds=2.0):

        self.llm = llm
        self.sleep_seconds = sleep_seconds

    def summarise(self, parents):

        summaries = []

        for index, parent in enumerate(parents, start=1):

            try:
                summary = call_with_retry(
                    self.llm,
                    CHUNK_SUMMARY_PROMPT.format(text=parent.page_content)
                )

            except Exception as e:
                # A missing summary only loses one extra vector; keep going
                logger.warning(f"Summary failed for parent {index}: {e}")
                continue

            metadata = dict(parent.metadata)
            metadata["chunk_id"] = make_chunk_id(
                metadata.get("source"), metadata.get("page"),
                f"{metadata['parent_id']}-summary"
            )
            metadata["is_summary"] = True

            summaries.append(Document(page_content=summary, metadata=metadata))

            logger.info(f"Summarised parent {index}/{len(parents)}")

            # Stay under Groq's requests-per-minute limit
            time.sleep(self.sleep_seconds)

        return summaries
