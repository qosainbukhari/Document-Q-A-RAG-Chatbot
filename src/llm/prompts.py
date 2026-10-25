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
