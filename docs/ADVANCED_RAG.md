# Advanced RAG — every technique in this project, explained simply

This is a study guide. Each section answers four questions:
**What is it? Why does it help? What does it cost? Where is it in the code?**
Every technique can be switched ON/OFF in [`configs/config.yaml`](../configs/config.yaml).

---

## 0. The naive RAG baseline (where we started)

```
PDF → pages → 1000-char chunks → embeddings → Chroma
question → embedding → 4 most similar chunks → prompt → LLM → answer
```

An **embedding** turns text into a vector (384 numbers with `all-MiniLM-L6-v2`).
Texts with similar meaning get vectors that point in similar directions, measured
with **cosine similarity** (1 = same direction, 0 = unrelated).

Naive RAG works, but it has known weak spots — each technique below fixes one:

| Problem | Fix |
|---|---|
| A chunk is either too small to answer from, or too big to search precisely | Parent-child chunking |
| Embeddings miss exact keywords, names, codes | Hybrid search (BM25 + dense) |
| The "most similar" vector is not always the best answer | Cross-encoder reranking |
| Retrieved chunks contain many irrelevant sentences | Contextual compression |
| Follow-ups like "what are its disadvantages?" retrieve nothing useful | Query rewriting |
| One phrasing of a question misses relevant text | Multi-query, HyDE, decomposition |

---

## Stage 0 — Engineering fixes

**Stable chunk IDs.** Every row in a vector DB has an ID. Chroma invents random
UUIDs if you don't give one, so uploading the same PDF twice stored every chunk
twice — and retrieval then returned the same text twice, wasting context slots.
Now `id = sha256("file.pdf:page:index")`: the same chunk always gets the same ID,
so writing it again **replaces** it. This is called *idempotent* ingestion.
Stable IDs alone don't remove chunks that disappeared in a new version of the file,
so before indexing we also **delete all chunks whose `source` is this file**.
→ `make_chunk_id()` in `src/ingestion/chunker.py`, `add_chunks()` in `src/vectordb/chroma_store.py`

**HTTP status codes.** `200 {"error": ...}` tells every client and monitoring tool
"success". We now raise `HTTPException`: **400** wrong file type, **422** unreadable
PDF / empty question, **500** server error (details go to the log, not the user).

**Memory per session.** One shared memory meant user B's "it" was resolved with
user A's history. Now each `session_id` has its own window memory. The client
sends no `session_id` the first time; the server creates one (UUID) and returns it.
An `OrderedDict` keeps at most `max_sessions`, dropping the least recently used (LRU).
→ `SessionMemoryStore` in `src/llm/conversation_memory.py`

---

## Stage 1 — Parent-child chunking

**The dilemma.** Small chunks (~300 chars) give *precise* embeddings: one idea per
vector, so search finds exactly the right spot. But 300 chars is often too little
for the LLM to answer from. Big chunks (~1500 chars) give the LLM enough context,
but their embedding is an "average" of many ideas, so search is fuzzier.

**The fix: use both.**
```
page ──split──► parents (1500 chars)  → stored as text in a JSON docstore
                  └─split─► children (300 chars) → embedded in Chroma, with parent_id
query → search children → take their parent_ids → send the unique PARENTS to the LLM
```

*Example:* "What is a validation set used for?" matches the 300-char child
"The validation set is used to tune hyper parameters" very precisely; the LLM then
receives the whole 1500-char parent with the surrounding explanation.

**Details that matter**
- We split **page by page**, so every parent and child has exactly one page → citations stay correct.
- Several children can point to the same parent: we keep each parent **once**, at the rank of its best child.
- `naive` and `parent_child` use **separate Chroma collections** so they never mix.

**Cost:** more vectors to store (≈ 2–5× more chunks); a second store (JSON) to keep in sync.
→ `ParentChildChunker` in `src/ingestion/chunker.py`, `JSONDocStore` in `src/vectordb/docstore.py`,
`to_parents()` in `src/retriever/retriever.py` · flag: `chunking.strategy`

---

## Stage 2 — Hybrid search (BM25 + dense) with Reciprocal Rank Fusion

**Dense search** (embeddings) understands meaning: "car" ≈ "automobile".
But it is weak on **exact tokens**: error codes, names, rare terms, numbers.

**Sparse search (BM25)** is classic keyword search. A chunk scores high if it
contains the query words, especially *rare* words (IDF), with diminishing returns
for repeating a word, normalised by chunk length. It knows nothing about meaning.

**New library: `rank_bm25`.** A tiny pure-Python BM25 implementation:
`BM25Okapi(list_of_token_lists).get_scores(query_tokens)`. It lives in memory,
so we rebuild it from the docstore's child chunks at startup and after each upload.

**Combining them — why ranks, not scores?** A cosine similarity is ~0–1; a BM25
score can be 0–30 and depends on the query length and corpus. Adding them would
let BM25 dominate, and normalising them is fragile. **Reciprocal Rank Fusion** only
uses each document's *position*:

```
score(doc) = Σ over lists  1 / (k + rank)        k = 60, rank starts at 1
```

*Example:* doc A is 1st in dense and 2nd in BM25 → 1/61 + 1/62 = 0.0325.
Doc B is 1st in BM25 only → 1/61 = 0.0164. A wins: **agreement between retrievers
is rewarded**. `k = 60` (from the original RRF paper) flattens the curve so rank 1
is not overwhelmingly more important than rank 5.

**Cost:** tiny (~5 ms). **Risk:** when every chunk contains the query words
(e.g. "machine learning" in an ML lecture), BM25 ranking is noisy and can push
a good dense result down — exactly what our evaluation measured.
→ `src/retriever/bm25_index.py`, `src/retriever/fusion.py` · flag: `retriever.hybrid.enabled`

---

## Stage 3 — Reranking with a cross-encoder

**Bi-encoder** (our embedding model): question and chunk are embedded
**separately**, then compared with cosine similarity. Chunk vectors are computed
once at upload, so search over millions of chunks is fast. But the model never
sees the question and chunk *together*.

**Cross-encoder:** the model reads `[question] [SEP] [chunk]` **together** and
outputs one relevance score. Every word of the question can attend to every word
of the chunk → much more accurate. But nothing can be precomputed: one full model
run **per (question, chunk) pair**.

**So we combine them:** fast retrieval finds ~20 candidates → the cross-encoder
re-scores just those 20 → keep the top 4.

**New library: `sentence-transformers`' `CrossEncoder`** with
`cross-encoder/ms-marco-MiniLM-L-6-v2` (≈ 90 MB, CPU). It was trained on
MS MARCO: real Bing search queries and web passages.

**Cost (measured on a laptop CPU):** ~550–650 ms to rerank 15–20 parent chunks —
the slowest retrieval step. **Risk:** it is only as good as its training domain;
on bullet-point lecture slides it sometimes preferred a general "What is ML" page.
→ `src/retriever/reranker.py` · flag: `retriever.rerank.enabled`

---

## Stage 4 — Contextual compression (no LLM)

Even the right chunk contains sentences that don't help. They cost tokens and can
distract the LLM ("lost in the middle").

**How ours works:** split each chunk into sentences (also on bullets `•`) → embed
the sentences and the question → keep the sentences with cosine similarity
≥ `similarity_threshold` (0.3), in their original order → always keep at least
`min_sentences` so a chunk is never empty. Metadata is copied, so the page citation
survives. One batch embedding call per chunk — **no LLM call**.

**Trade-off / failure mode:** the threshold is a guess. A sentence can be
important without sharing words with the question. Real example from our PDF, for
"What is overfitting?": the heading "Overfitting" scored 0.86, but the line that
explains it, "Model memorizes training data, fails on new data", scored only 0.21
and was dropped. Compression saves tokens but can remove the answer.
The alternative, LLM-based extraction, is smarter but costs one LLM call per chunk.
→ `src/retriever/compressor.py` · flags: `retriever.compression.*`

---

## Stage 5 — Query transformation

All prompts live in [`src/llm/prompts.py`](../src/llm/prompts.py).

**Query rewriting (ON).** Chat history + "What are its disadvantages?" → the LLM
writes a *standalone* question: "What are the disadvantages of decision trees?".
Retrieval only sees the question, so without this, "its" would match nothing.
We **skip the LLM call when there is no history** (the first question is
already standalone), so it is free for single questions.
*Measured live:* "How is it different from underfitting?" → "How is overfitting different from underfitting?"

**Multi-query (OFF).** The LLM writes 3 rephrasings; we search with all 4 and fuse
the rankings with RRF. Helps when the user's wording differs from the document's.

**HyDE — Hypothetical Document Embeddings (OFF).** The LLM writes a *fake answer*
passage; we search with it. A question ("what is X?") and an answer ("X is ...")
look different; a fake answer looks like the real answer in the document. Risk:
if the guess is wrong, it searches for the wrong thing.

**Decomposition (OFF).** "Compare overfitting and underfitting" → sub-questions
"What is overfitting?" + "What is underfitting?", each searched, results fused.

**Cost:** one Groq call each, per question. Our free tier has ~8,000 tokens/minute,
so they are OFF by default. If the extra call fails, we continue with the original
question.
→ `src/llm/query_transformer.py` · flags: `query_transform.*`

---

## Stage 6 — Multi-vector retrieval (optional, OFF)

A chunk can be found through **more than one vector**. We already do this with
parent-child (many children → one parent). Multi-vector adds an **LLM-written
summary** of each parent as one more vector pointing to the same parent. A question
phrased at a high level ("what does this section say about evaluation?") can match
the summary even if it matches no single child.

**Why it is expensive on the free tier:** it needs **one LLM call per parent chunk
at upload time**. A 100-page PDF ≈ 200 parents ≈ 200 calls ≈ 200k+ tokens — with
an 8k tokens/minute limit that is 25+ minutes per upload, and it eats the daily quota.
We pause between calls (`sleep_seconds`) and retry on rate limits; a failed summary is
skipped, never failing the upload.
→ `src/ingestion/multi_vector.py` · flag: `multi_vector.enabled`

---

## Stage 7 — Evaluation

You cannot claim "advanced is better" without measuring. We measure **retrieval**
and **generation** separately, because they fail differently.

**Dataset:** `evaluation/eval_dataset.json` — 25 questions on `samples/Ml_book.pdf`,
each with a reference answer and the page(s) that contain the answer.

**Retrieval metrics (pure Python, no LLM)** — `evaluation/metrics.py`
- **Recall@K** (here: hit rate): 1 if any relevant page is in the top K, else 0; averaged.
  *Example:* relevant page 17, retrieved [18, 19, 17, 4] → Recall@1 = 0, Recall@2 = 0, Recall@4 = 1.
- **MRR** (Mean Reciprocal Rank): 1 / position of the first relevant result.
  Same example → 1/3 = 0.33. Rewards putting the right page *first*.

**Generation metrics (LLM-as-judge)** — `evaluation/judges.py`. The LLM gives only
yes/no verdicts as JSON; Python computes the score, so it is easy to inspect.
- **Faithfulness:** split the answer into claims; fraction supported by the context.
  Low = hallucination.
- **Context precision:** fraction of the retrieved chunks that are relevant.
  Low = noisy context. (With top-4 and one relevant page, the maximum is often 0.25–0.5.)
- **Context recall:** split the *reference* answer into statements; fraction found
  in the context. Low = retrieval missed information.

**RAGAS** is a popular library that implements these metrics (and more). We wrote
our own because: (1) we understand and can debug every line, (2) RAGAS makes many
LLM calls per sample with long prompts, which is hard on Groq's free tier,
(3) its default judges and versions change, which makes numbers hard to compare.
RAGAS is a good next step once the concepts are clear.

**What our numbers say** (see the README table): on this small, repetitive slide
deck the naive baseline already finds the right page almost always, and hybrid
search and reranking *lowered* Recall@1. That is a real result, not a bug:
these techniques are designed for large, diverse corpora; on 15 short pages that
all say "machine learning", BM25 and a web-trained reranker add noise. Always
evaluate on *your* documents before switching techniques on.

---

## Stage 8 — Observability (tracing) with Langfuse

A RAG answer is the end of a chain: rewrite → search → fuse → map → rerank →
compress → prompt → LLM. When an answer is wrong, you need to see **which step**
went wrong: was the right page never retrieved? Retrieved but reranked away?
Compressed away? Or retrieved fine, and the LLM ignored it?

**Tracing** records each step as a **span** (name, input, output, latency), nested
inside one **trace** per question. Langfuse shows it as a timeline.

Our `Tracer` (`src/observability/tracing.py`) creates spans: `rag-query` →
`query-transform`, `search`, `parent-mapping`, `rerank` (with scores),
`compression` (with ratios), `generation` (prompt, answer, model); plus `ingest`
for uploads. Traces are grouped by `session_id`. **No keys in `.env` → tracing is
off and nothing else changes**; any Langfuse error is caught and only logged.
Latency per step is always written to the log:

```
[trace] query-transform: 622 ms   [trace] search: 29 ms   [trace] parent-mapping: 1 ms
[trace] rerank: 556 ms            [trace] compression: 369 ms   [trace] generation: 714 ms
```

**Langfuse vs LangSmith:** both trace LLM apps and run evaluations. **Langfuse** is
open-source (MIT), can be self-hosted for free with Docker, has a free cloud tier,
and works with any framework through OpenTelemetry. **LangSmith** is LangChain's
hosted, closed-source product: very smooth if you use LangChain/LangGraph heavily,
with a free developer tier, but self-hosting is an enterprise feature. We chose
Langfuse because it is open-source and free to self-host.

---

## How a question flows through the advanced pipeline

```
question + session history
   │
   ├─ query rewrite (LLM, only if history)        → standalone question
   ├─ [multi-query / HyDE / decomposition]        → extra search queries (optional)
   │
   ├─ for each query: dense search (children) ─┐
   │                  BM25 search (children) ──┴─ RRF fuse ─┐
   ├─ RRF fuse across queries ◄──────────────────────────────┘
   ├─ map children → unique parents (docstore)
   ├─ cross-encoder rerank → top 4 parents
   ├─ compression → only query-relevant sentences (page metadata kept)
   │
   └─ prompt (history + context + question) → Groq LLM → answer + page citations
```
