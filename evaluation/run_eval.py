"""Evaluate one pipeline configuration on evaluation/eval_dataset.json.

    python -m evaluation.run_eval --config naive
    python -m evaluation.run_eval --config advanced --retrieval-only

--config NAME loads configs/config.yaml, then the overrides in
configs/eval/NAME.yaml ("default" = config.yaml as is). The PDF is indexed
into a temporary directory, so your real vector DB is never touched.
"""

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from evaluation.judges import LLMJudge
from evaluation.metrics import mean, recall_at_k, reciprocal_rank
from src.components import build_components
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.embeddings.embedding_model import EmbeddingModel
from src.llm.groq_client import GroqClient
from src.llm.llm_utils import RetryingLLM
from src.llm.rag_chain import RAGChain
from src.utils.logger import logger


RESULTS_DIR = Path("evaluation/results")

# Order of rows in the summary table
CONFIG_ORDER = ["naive", "hybrid", "rerank", "advanced", "advanced_multiquery"]

K_VALUES = [1, 2, 4]

SUMMARY_COLUMNS = (
    ["config", "questions"]
    + [f"recall@{k}" for k in K_VALUES]
    + ["mrr", "faithfulness", "context_precision", "context_recall",
       "retrieval_ms"]
)


def parse_args():

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="advanced",
                        help="name of a file in configs/eval/ (or 'default')")
    parser.add_argument("--dataset", default="evaluation/eval_dataset.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N questions (quick test)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip answer generation and LLM judges (no Groq calls)")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="seconds to wait after each judge call (rate limits)")

    return parser.parse_args()


def load_pipeline_config(name):

    if name == "default":
        return load_config(DEFAULT_CONFIG_PATH)

    return load_config(DEFAULT_CONFIG_PATH, f"configs/eval/{name}.yaml")


def safe(metric, *args):

    # One bad judge reply must not kill a 25-question run
    try:
        return metric(*args)
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return None


def evaluate(args):

    config = load_pipeline_config(args.config)

    with open(args.dataset) as file:
        dataset = json.load(file)

    questions = dataset["questions"][:args.limit]

    embedding_model = EmbeddingModel().get_model()

    llm = RetryingLLM(GroqClient().get_llm())

    judge = LLMJudge(llm, sleep_seconds=args.sleep)

    rows = []

    with tempfile.TemporaryDirectory() as tmp_dir:

        components = build_components(
            config, embedding_model, llm, persist_directory=tmp_dir
        )

        components.ingestion.ingest_file(dataset["document"])

        # No memory: every eval question is independent
        chain = RAGChain(
            components.retriever, llm, memory=None,
            query_transformer=components.query_transformer
        )

        for number, item in enumerate(questions, start=1):

            question = item["question"]
            relevant = set(item["relevant_pages"])

            start = time.perf_counter()
            standalone, documents = chain.retrieve(question)
            retrieval_ms = (time.perf_counter() - start) * 1000

            pages = [doc.metadata.get("page") for doc in documents]
            contexts = [doc.page_content for doc in documents]

            row = {
                "question": question,
                "relevant_pages": item["relevant_pages"],
                "retrieved_pages": pages,
                **{f"recall@{k}": recall_at_k(pages, relevant, k) for k in K_VALUES},
                "mrr": reciprocal_rank(pages, relevant),
                "retrieval_ms": round(retrieval_ms),
            }

            if not args.retrieval_only:

                answer = chain.answer(question, documents)

                row["answer"] = answer
                row["faithfulness"] = safe(judge.faithfulness, answer, contexts)
                row["context_precision"] = safe(judge.context_precision, question, contexts)
                row["context_recall"] = safe(
                    judge.context_recall, item["reference_answer"], contexts
                )

            rows.append(row)

            logger.info(
                f"[{args.config}] {number}/{len(questions)} "
                f"pages={pages} relevant={item['relevant_pages']} "
                f"rr={row['mrr']:.2f}"
            )

    return rows


def summarise(name, rows):

    summary = {"config": name, "questions": len(rows)}

    for column in SUMMARY_COLUMNS[2:]:

        value = mean([row.get(column) for row in rows])

        summary[column] = (
            "" if value is None
            else round(value) if column == "retrieval_ms"
            else round(value, 3)
        )

    return summary


def write_results(name, rows, summary):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-question details, to see WHICH questions fail
    per_question = RESULTS_DIR / f"{name}_per_question.csv"

    with open(per_question, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # summary.csv keeps one row per config; re-running a config replaces its row
    summary_csv = RESULTS_DIR / "summary.csv"
    summaries = {}

    if summary_csv.exists():
        with open(summary_csv) as file:
            summaries = {row["config"]: row for row in csv.DictReader(file)}

    summaries[name] = summary

    ordered = sorted(
        summaries.values(),
        key=lambda row: (
            CONFIG_ORDER.index(row["config"]) if row["config"] in CONFIG_ORDER
            else len(CONFIG_ORDER)
        )
    )

    with open(summary_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)

    # Same table in markdown, ready to paste into the README
    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "|" + "---|" * len(SUMMARY_COLUMNS),
    ]
    lines += [
        "| " + " | ".join(str(row.get(c, "")) for c in SUMMARY_COLUMNS) + " |"
        for row in ordered
    ]

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")

    return per_question


def main():

    load_dotenv()

    args = parse_args()

    rows = evaluate(args)

    summary = summarise(args.config, rows)

    per_question = write_results(args.config, rows, summary)

    print(json.dumps(summary, indent=2))
    print(f"Per-question results: {per_question}")
    print(f"Summary table: {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
