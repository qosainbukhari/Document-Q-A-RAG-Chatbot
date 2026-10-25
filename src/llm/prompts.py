# Every prompt the project sends to the LLM lives in this file,
# so they are easy to read, compare and tweak in one place.
# Placeholders like {question} are filled with str.format().


# ---------- Answer generation ----------

RAG_ANSWER_PROMPT = """
You are an intelligent Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question using ONLY the provided context.

Instructions:
- Do NOT use outside knowledge.
- If the answer is not available in the context, respond exactly with:
  "I don't know based on the document."
- Provide clear, accurate, and well-structured answers.
- Explain technical concepts in simple language.
- Include important definitions, examples, formulas, or key points when available.
- Use chat history only for conversational continuity.
- Avoid repeating information.
- Keep the response professional and concise.
- Mention source page numbers naturally when relevant.

Chat History:
---------------------
{history}
---------------------

Context:
---------------------
{context}
---------------------

Question:
{question}

Answer:
"""


# ---------- Query transformation ----------

QUERY_REWRITE_PROMPT = """
Given the chat history and a follow-up question, rewrite the follow-up
into a standalone question that can be understood without the history.
Replace pronouns like "it", "they", "this" with what they refer to.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Chat History:
{history}

Follow-up question: {question}

Standalone question:"""


MULTI_QUERY_PROMPT = """
Write {n} different versions of the question below, to search a document
with. Use different wording and synonyms, but keep the same meaning.
Return ONLY the questions, one per line, no numbering.

Question: {question}"""


HYDE_PROMPT = """
Write a short passage (3-4 sentences) from a machine learning textbook
that answers the question below. It is fine to guess; the passage is only
used to search for similar text.

Question: {question}

Passage:"""


DECOMPOSITION_PROMPT = """
Break the question below into at most {n} simpler sub-questions that
together answer it. If it is already simple, return just the question.
Return ONLY the sub-questions, one per line, no numbering.

Question: {question}"""


# ---------- Multi-vector indexing (one call per parent chunk at upload) ----------

CHUNK_SUMMARY_PROMPT = """
Summarise the following passage in 2-3 sentences. Mention the main
concepts and terms it covers, so the summary can be used to find it.
Return ONLY the summary.

Passage:
{text}

Summary:"""


# ---------- Evaluation: LLM-as-judge (used by evaluation/judges.py) ----------
# Each judge returns a JSON list of yes/no verdicts; Python computes the score.
# Literal braces are doubled ({{ }}) because of str.format().

FAITHFULNESS_PROMPT = """
You check an ANSWER for hallucinations against a CONTEXT.
1. Split the ANSWER into short, separate factual claims.
2. For each claim, decide if the CONTEXT directly supports it.
Ignore page numbers and phrases like "according to the document".

CONTEXT:
{contexts}

ANSWER:
{answer}

Return ONLY JSON in this format:
{{"claims": [{{"claim": "...", "supported": true}}]}}"""


CONTEXT_PRECISION_PROMPT = """
Below is a QUESTION and {n} numbered CONTEXT chunks.
For each chunk, in order, decide if it contains information that helps
answer the QUESTION.

QUESTION:
{question}

CONTEXT:
{contexts}

Return ONLY JSON with exactly {n} booleans, one per chunk, in order:
{{"relevant": [true, false]}}"""


CONTEXT_RECALL_PROMPT = """
You check whether a CONTEXT contains everything a REFERENCE ANSWER says.
1. Split the REFERENCE ANSWER into short, separate statements.
2. For each statement, decide if it can be found in (or directly
   inferred from) the CONTEXT.

CONTEXT:
{contexts}

REFERENCE ANSWER:
{reference}

Return ONLY JSON in this format:
{{"statements": [{{"statement": "...", "in_context": true}}]}}"""
