# Interview questions about this project — with short answers

Practise saying these out loud. Details are in [ADVANCED_RAG.md](ADVANCED_RAG.md).

## The project

**1. Describe your project in 30 seconds.**
A document Q&A chatbot: users upload PDFs and ask questions; answers are grounded
in the PDF with page citations. FastAPI backend, Streamlit UI, ChromaDB, MiniLM
embeddings, Llama/GPT-OSS via Groq. I upgraded it from naive RAG to advanced RAG —
parent-child chunking, hybrid BM25 + dense search with RRF, cross-encoder
reranking, compression, query rewriting — every technique switchable in config,
and I built an evaluation to measure whether each one actually helps.

**2. What is RAG and why use it instead of fine-tuning?**
Retrieve relevant text, put it in the prompt, let the LLM answer from it. Fine-tuning
teaches style/behaviour but is a poor way to add facts: expensive, hard to update,
no citations. RAG updates instantly when you upload a new document and can cite sources.

**3. What was wrong with your first (naive) version?**
Duplicate chunks on re-upload, HTTP 200 for errors, one chat memory shared by all
users, fixed-size chunks, embedding-only search, and no measurements.

## Chunking & retrieval

**4. How do you choose chunk size?**
Trade-off: small chunks = precise search but little context; big chunks = more
context but fuzzy embeddings. Parent-child avoids choosing: search small children,
return big parents.

**5. How do you keep citations correct with parent-child chunking?**
I split page by page, so each parent and child belongs to exactly one page and
inherits its `page` metadata.

**6. What is BM25? When does it beat embeddings?**
Keyword ranking using term frequency, inverse document frequency (rare words count
more) and length normalisation. It beats embeddings on exact tokens: names, codes,
acronyms, rare technical terms.

**7. Why Reciprocal Rank Fusion instead of adding scores?**
Cosine and BM25 scores are on different, query-dependent scales. RRF uses only ranks:
Σ 1/(k + rank), k = 60, so it needs no normalisation and rewards documents that both
retrievers agree on.

**8. Bi-encoder vs cross-encoder?**
Bi-encoder embeds query and document separately → precomputable, fast, less accurate.
Cross-encoder reads them together → accurate, but one model run per pair, so we only
use it to rerank ~20 candidates. Measured ~600 ms on CPU for that.

**9. Why not rerank the whole collection?**
Cost grows linearly with the number of chunks; 100k chunks × ~30 ms = minutes per query.

**10. What does contextual compression do and what can go wrong?**
Keeps only sentences similar to the query to save tokens and reduce noise. It can
drop an important sentence that doesn't share words with the query — I saw that
with a slide heading matching but its explanation not.

**11. Why rewrite the query?**
Follow-ups like "what about its disadvantages?" have no meaning on their own. The LLM
turns them into standalone questions using chat history. I skip the call when there
is no history, to save free-tier quota.

**12. Explain HyDE.**
Generate a hypothetical answer and search with it, because an answer looks more
like the document text than a question does. Risk: a wrong guess searches for the
wrong thing.

**13. What is multi-vector retrieval?**
Index several vectors per document (children, an LLM summary, hypothetical
questions) that all point to the same parent. I made it optional because it costs
one LLM call per chunk at upload.

## Evaluation

**14. How did you evaluate the system?**
25 questions with reference answers and relevant pages. Retrieval: Recall@K and MRR
in plain Python. Generation: LLM-as-judge for faithfulness, context precision and
context recall. I ran it for naive, +hybrid, +rerank and the full pipeline.

**15. Recall@K vs MRR?**
Recall@K: is a relevant result anywhere in the top K? MRR: how high is the first
relevant result (1/rank)? MRR rewards getting it first.

**16. Faithfulness vs answer correctness?**
Faithfulness = is the answer supported by the retrieved context (no hallucination)?
An answer can be faithful to the wrong context, so you also need context recall
or correctness against a reference answer.

**17. What did your evaluation show?**
On the small 15-page test lecture, naive dense retrieval was already near-perfect
(Recall@1 0.96) and hybrid and reranking lowered top-1 accuracy, because every page
contains the same keywords and the reranker is trained on web search data. Lesson:
measure on your own data; advanced techniques pay off on larger, diverse corpora.
(Quote the generation numbers from the README table.)

**18. Limitations of LLM-as-judge?**
Judges are biased (prefer long answers, their own style), non-deterministic, and
cost tokens. Mitigations: yes/no verdicts per claim instead of a 1–10 score, scores
computed in code, a fixed judge model, and spot-checking verdicts by hand.

**19. Why not RAGAS?**
I wanted to understand every metric, and RAGAS makes many long LLM calls that hit
Groq's free-tier limits. RAGAS is a reasonable next step.

## Engineering

**20. How do you avoid duplicate chunks?**
Deterministic IDs: hash(source, page, chunk index), plus deleting the old chunks of
a file before re-indexing it. Re-uploading is idempotent.

**21. How do you handle multiple users' chat history?**
A memory per `session_id`, kept in an LRU-capped dictionary. In production I'd move
it to Redis so it survives restarts and works with several server processes.

**22. How do you handle Groq rate limits?**
Exponential backoff on errors (5 s, 10 s, 20 s, …), sleeps between evaluation calls,
expensive techniques OFF by default, and no rewrite call when there is no history.

**23. Why tracing? What do you trace?**
To find which step caused a bad answer. One trace per question with spans for
rewrite, search, parent mapping, rerank (scores), compression, generation (prompt,
answer), each with latency. It is optional: no keys → no-op.

**24. Langfuse vs LangSmith?**
Both trace and evaluate LLM apps. Langfuse is open-source and free to self-host;
LangSmith is LangChain's hosted product, with self-hosting only in the enterprise tier.

**25. Why are your FastAPI routes `def` and not `async def`?**
They do blocking work (embedding, reranking, HTTP to Groq). In an `async def` that
would block the event loop for all users; FastAPI runs plain `def` routes in a
thread pool.

**26. What would you do next?**
Evaluate on a larger, real document set; tune top_k / thresholds on a dev set;
store sessions in Redis; stream answers; add a docstore database instead of JSON;
try a stronger multilingual embedding model; cache embeddings of repeated queries.
