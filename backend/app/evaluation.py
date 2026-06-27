from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import types
from pathlib import Path
from typing import Callable

from datasets import Dataset

from .documents import read_text
from .settings import get_settings


def parse_question_set(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [
                {key: str(value or "").strip() for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    rows: list[dict[str, str]] = []
    for line in read_text(path).splitlines():
        if not line.startswith("| TQ-"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "id": parts[0],
                "document": parts[1],
                "keyword": parts[2],
                "question": parts[3],
                "expected": parts[4],
                "expected_chunk_id": parts[5] if len(parts) > 5 else "",
            }
        )
    return rows


def select_questions(questions: list[dict[str, str]], limit: int, seed: int | None) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(questions):
        return questions
    if seed is None:
        return questions[:limit]
    rng = random.Random(seed)
    return sorted(rng.sample(questions, limit), key=lambda item: item["id"])


def run_ragas(records: list[dict[str, object]], model: str | None = None):
    _install_ragas_langchain_shims()
    try:
        from ragas import evaluate
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall
        from ragas.run_config import RunConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ragas is installed, but its optional LangChain compatibility imports are unavailable. "
            "Use the generated JSONL records directly, or pin compatible ragas/langchain versions before --run-ragas."
        ) from exc
    settings = get_settings()
    from .providers import get_chat_model, get_embeddings

    ragas_llm = get_chat_model(settings, model or settings.ragas_model, temperature=0.0)
    ragas_embeddings = get_embeddings(settings)
    dataset = Dataset.from_list(records)
    return evaluate(
        dataset,
        metrics=[
            AnswerRelevancy(strictness=1),
            ContextPrecision(max_retries=3),
            ContextRecall(max_retries=3),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            timeout=300,
            max_retries=10,
            max_wait=90,
            max_workers=1,
            seed=42,
        ),
        raise_exceptions=False,
        batch_size=1,
    )


def serialize_ragas_result(result: object) -> dict[str, object]:
    metrics: dict[str, object] = {}
    valid_samples: dict[str, int] = {}
    scores = getattr(result, "scores", None)
    if isinstance(scores, list):
        metric_names = sorted({key for row in scores if isinstance(row, dict) for key in row})
        for metric_name in metric_names:
            values: list[float] = []
            for row in scores:
                if not isinstance(row, dict):
                    continue
                try:
                    value = float(row.get(metric_name))
                except (TypeError, ValueError):
                    continue
                if not math.isnan(value):
                    values.append(value)
            metrics[metric_name] = round(sum(values) / len(values), 4) if values else None
            valid_samples[metric_name] = len(values)
    else:
        try:
            raw_items = dict(result).items()  # type: ignore[arg-type]
        except Exception:
            raw_items = []
        for key, value in raw_items:
            try:
                numeric_value = float(value)
                metrics[str(key)] = None if math.isnan(numeric_value) else round(numeric_value, 4)
            except (TypeError, ValueError):
                metrics[str(key)] = value
    aliases = {
        "context_precision": "上下文精确率",
        "context_recall": "上下文召回率",
        "answer_relevancy": "答案相关性",
        "faithfulness": "忠实度",
    }
    sample_count = len(scores) if isinstance(scores, list) else None
    coverage = {
        metric_name: round(count / sample_count, 4) if sample_count else 0.0
        for metric_name, count in valid_samples.items()
    }
    return {
        "sample_count": sample_count,
        "metrics": metrics,
        "valid_samples": valid_samples,
        "metric_coverage": coverage,
        "complete": bool(sample_count) and all(count == sample_count for count in valid_samples.values()),
        "aliases": aliases,
    }


def _install_ragas_langchain_shims() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - compatibility type for ragas imports only.
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def build_answer_records(
    questions: list[dict[str, str]],
    scenario: str | None = None,
    top_k: int = 8,
    existing_records: dict[str, dict[str, object]] | None = None,
    on_record: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    from .main import rag_service

    records: list[dict[str, object]] = []
    for item in questions:
        existing = (existing_records or {}).get(item["id"])
        if existing is not None:
            records.append(existing)
            continue
        response = rag_service.chat(message=item["question"], scenario=scenario, top_k=top_k)
        record = {
            "id": item["id"],
            "user_input": item["question"],
            "response": response.answer,
            "retrieved_contexts": [source.text for source in response.sources],
            "reference": item["expected"],
            "expected_chunk_id": item.get("expected_chunk_id", ""),
            "source_documents": [source.source_name for source in response.sources],
            "source_chunk_ids": [source.chunk_id for source in response.sources],
        }
        records.append(record)
        if on_record is not None:
            on_record(record)
    return records


def load_answer_records(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id")
            if isinstance(record_id, str):
                records[record_id] = record
    return records


def build_retrieval_report(
    questions: list[dict[str, str]],
    *,
    scenario: str | None = None,
    retrieval_top_k: int = 20,
    hit_k: int = 5,
) -> dict[str, object]:
    from .main import hybrid_index, settings

    records: list[dict[str, object]] = []
    for item in questions:
        start = time.perf_counter()
        sources, debug = hybrid_index.search(
            item["question"],
            scenario=scenario,
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            final_top_k=retrieval_top_k,
            rrf_k=settings.rrf_k,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        expected_chunk_ids = parse_expected_chunk_ids(item)
        if not expected_chunk_ids:
            raise ValueError(f"Question {item['id']} is missing expected_chunk_id or expected_chunk_ids.")
        source_chunk_ids = [source.chunk_id for source in sources]
        first_rank = first_matching_chunk_rank(expected_chunk_ids, source_chunk_ids)
        relevant_at_k = sum(
            1 for chunk_id in source_chunk_ids[:hit_k] if chunk_id in expected_chunk_ids
        )
        document_rank = first_relevant_rank(item["document"], [source.source_name for source in sources])
        records.append(
            {
                "id": item["id"],
                "document": item["document"],
                "keyword": item["keyword"],
                "question": item["question"],
                "expected": item["expected"],
                "expected_chunk_id": item.get("expected_chunk_id", "").strip(),
                "expected_chunk_ids": sorted(expected_chunk_ids),
                "first_relevant_rank": first_rank,
                "first_document_rank": document_rank,
                f"hit@{hit_k}": first_rank is not None and first_rank <= hit_k,
                f"precision@{hit_k}": relevant_at_k / hit_k,
                f"recall@{hit_k}": relevant_at_k / len(expected_chunk_ids),
                "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
                "latency_ms": round(latency_ms, 2),
                "source_chunk_ids": source_chunk_ids,
                "source_documents": [source.source_name for source in sources],
                "sources": [
                    {
                        "chunk_id": source.chunk_id,
                        "source_name": source.source_name,
                        "section": source.section,
                        "page": source.page,
                        "dense_rank": source.dense_rank,
                        "bm25_rank": source.bm25_rank,
                        "rrf_score": source.rrf_score,
                        "rerank_rank": source.rerank_rank,
                        "rerank_score": source.rerank_score,
                    }
                    for source in sources
                ],
                "retrieval_debug": debug,
            }
        )
    return {"summary": summarize_retrieval_records(records, retrieval_top_k, hit_k), "records": records}


def summarize_retrieval_records(
    records: list[dict[str, object]],
    retrieval_top_k: int,
    hit_k: int,
) -> dict[str, object]:
    total = len(records)
    if total == 0:
        return {
            "question_count": 0,
            "retrieval_top_k": retrieval_top_k,
            "hit_k": hit_k,
            f"hit@{hit_k}": 0.0,
            f"precision@{hit_k}": 0.0,
            f"recall@{hit_k}": 0.0,
            "MRR": 0.0,
            "miss_count": 0,
            "relevance_unit": "exact_chunk_id",
            "mean_latency_ms": 0.0,
            "P99_ms": 0.0,
        }
    hit_count = sum(1 for record in records if record[f"hit@{hit_k}"])
    reciprocal_ranks = [float(record["reciprocal_rank"]) for record in records]
    precisions = [float(record[f"precision@{hit_k}"]) for record in records]
    recalls = [float(record[f"recall@{hit_k}"]) for record in records]
    latencies = [float(record["latency_ms"]) for record in records]
    return {
        "question_count": total,
        "retrieval_top_k": retrieval_top_k,
        "hit_k": hit_k,
        f"hit@{hit_k}": round(hit_count / total, 4),
        f"precision@{hit_k}": round(sum(precisions) / total, 4),
        f"recall@{hit_k}": round(sum(recalls) / total, 4),
        "MRR": round(sum(reciprocal_ranks) / total, 4),
        "miss_count": sum(1 for record in records if record["first_relevant_rank"] is None),
        "relevance_unit": "exact_chunk_id",
        "mean_latency_ms": round(sum(latencies) / total, 2),
        "P99_ms": round(percentile_nearest_rank(latencies, 99), 2),
        "P99_definition": "99% of retrieval requests have latency less than or equal to this value.",
    }


def first_relevant_rank(expected_document: str, source_documents: list[str]) -> int | None:
    for index, source_document in enumerate(source_documents, start=1):
        if document_matches(expected_document, source_document):
            return index
    return None


def first_chunk_rank(expected_chunk_id: str, source_chunk_ids: list[str]) -> int | None:
    return first_matching_chunk_rank({expected_chunk_id}, source_chunk_ids)


def first_matching_chunk_rank(expected_chunk_ids: set[str], source_chunk_ids: list[str]) -> int | None:
    for index, chunk_id in enumerate(source_chunk_ids, start=1):
        if chunk_id in expected_chunk_ids:
            return index
    return None


def parse_expected_chunk_ids(item: dict[str, str]) -> set[str]:
    values = item.get("expected_chunk_ids", "") or item.get("expected_chunk_id", "")
    return {
        chunk_id.strip()
        for chunk_id in values.replace(",", ";").split(";")
        if chunk_id.strip()
    }


def document_matches(expected_document: str, source_document: str) -> bool:
    expected = normalize_document_name(expected_document)
    source = normalize_document_name(Path(source_document).stem)
    return bool(expected and source and (expected in source or source in expected))


def normalize_document_name(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def percentile_nearest_rank(values: list[float] | list[int], percentile: int) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG retrieval and optional Ragas evaluation from a markdown question set.")
    parser.add_argument("question_set", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("ragas_questions.csv"))
    parser.add_argument("--answers-output", type=Path, default=Path("ragas_records.jsonl"))
    parser.add_argument("--ragas-output", type=Path, default=Path("ragas_summary.json"))
    parser.add_argument("--retrieval-output", type=Path, default=Path("retrieval_report.json"))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--hit-k", type=int, default=5)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--run-retrieval", action="store_true")
    parser.add_argument("--run-ragas", action="store_true")
    parser.add_argument("--ragas-model", default=None)
    args = parser.parse_args()
    questions = select_questions(parse_question_set(args.question_set), args.limit, args.sample_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "document",
                "keyword",
                "question",
                "expected",
                "expected_chunk_id",
                "expected_chunk_ids",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(questions)
    print(f"Wrote {len(questions)} sampled questions to {args.output}")
    if args.run_retrieval:
        report = build_retrieval_report(
            questions,
            scenario=args.scenario,
            retrieval_top_k=args.retrieval_top_k,
            hit_k=args.hit_k,
        )
        args.retrieval_output.parent.mkdir(parents=True, exist_ok=True)
        args.retrieval_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"Wrote retrieval report to {args.retrieval_output}")
    if not args.generate_answers and not args.run_ragas:
        return
    args.answers_output.parent.mkdir(parents=True, exist_ok=True)
    existing_records = load_answer_records(args.answers_output)
    selected_ids = {item["id"] for item in questions}
    existing_records = {key: value for key, value in existing_records.items() if key in selected_ids}
    file_mode = "a" if existing_records else "w"
    with args.answers_output.open(file_mode, encoding="utf-8") as stream:
        def write_record(record: dict[str, object]) -> None:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

        records = build_answer_records(
            questions,
            scenario=args.scenario,
            top_k=args.retrieval_top_k,
            existing_records=existing_records,
            on_record=write_record,
        )
    print(f"Wrote {len(records)} answer records to {args.answers_output}")
    if args.run_ragas:
        result = run_ragas(records, model=args.ragas_model)
        summary = serialize_ragas_result(result)
        summary["evaluation_model"] = args.ragas_model or get_settings().ragas_model
        args.ragas_output.parent.mkdir(parents=True, exist_ok=True)
        args.ragas_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Wrote Ragas summary to {args.ragas_output}")


if __name__ == "__main__":
    main()
