import json
import re
import time

from src.llm.llm_utils import call_with_retry
from src.llm.prompts import (
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
    FAITHFULNESS_PROMPT,
)


NO_ANSWER = "I don't know based on the document."


def parse_json(text):

    # Models often wrap JSON in prose or ``` fences: grab the outer {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON in judge output: {text[:200]!r}")

    return json.loads(match.group(0))


def format_contexts(contexts):

    return "\n\n".join(
        f"[{index}] {context}"
        for index, context in enumerate(contexts, start=1)
    )


def is_true(value):

    # bool("false") is True in Python, so check strings explicitly
    return value is True or str(value).strip().lower() in ("true", "yes", "1")


def fraction(verdicts):

    return sum(verdicts) / len(verdicts) if verdicts else None


class LLMJudge:

    # Three LLM-as-judge metrics. The LLM only gives yes/no verdicts;
    # the scores are computed here in Python, so they are easy to check.

    def __init__(self, llm, sleep_seconds=2.0):

        self.llm = llm
        self.sleep_seconds = sleep_seconds

    def _ask(self, prompt):

        result = parse_json(call_with_retry(self.llm, prompt))

        # Space calls out so we stay under the requests-per-minute limit
        time.sleep(self.sleep_seconds)

        return result

    def faithfulness(self, answer, contexts):

        # supported claims / all claims in the answer
        if answer.strip() == NO_ANSWER:
            return None  # an abstention makes no claims to check

        result = self._ask(FAITHFULNESS_PROMPT.format(
            contexts=format_contexts(contexts), answer=answer
        ))

        return fraction([
            is_true(claim.get("supported")) for claim in result.get("claims", [])
        ])

    def context_precision(self, question, contexts):

        # relevant retrieved chunks / all retrieved chunks
        result = self._ask(CONTEXT_PRECISION_PROMPT.format(
            n=len(contexts), question=question,
            contexts=format_contexts(contexts)
        ))

        # Trust at most one verdict per chunk, even if the LLM adds extras
        verdicts = [is_true(v) for v in result.get("relevant", [])][:len(contexts)]

        return fraction(verdicts)

    def context_recall(self, reference_answer, contexts):

        # reference statements found in the context / all reference statements
        result = self._ask(CONTEXT_RECALL_PROMPT.format(
            contexts=format_contexts(contexts), reference=reference_answer
        ))

        return fraction([
            is_true(s.get("in_context")) for s in result.get("statements", [])
        ])
