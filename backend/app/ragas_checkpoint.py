from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import Dataset

from .evaluation import _install_ragas_langchain_shims
from .providers import get_chat_model, get_embeddings
from .settings import get_settings


METRICS = ("answer_relevancy", "context_recall", "context_precision")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_checkpoint(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "evaluation_model": "",
        "context_top_k": 0,
        "records": {
            record["id"]: {
                "id": record["id"],
                "metrics": {},
                "errors": {},
                "elapsed_seconds": {},
            }
            for record in records
        },
    }


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["summary"] = summarize(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize(checkpoint: dict[str, Any]) -> dict[str, Any]:
    records = list(checkpoint["records"].values())
    metrics: dict[str, float | None] = {}
    valid_samples: dict[str, int] = {}
    for metric_name in METRICS:
        values = [
            float(record["metrics"][metric_name])
            for record in records
            if record["metrics"].get(metric_name) is not None
        ]
        metrics[metric_name] = round(sum(values) / len(values), 4) if values else None
        valid_samples[metric_name] = len(values)
    sample_count = len(records)
    return {
        "sample_count": sample_count,
        "metrics": metrics,
        "valid_samples": valid_samples,
        "metric_coverage": {
            metric_name: round(count / sample_count, 4) if sample_count else 0.0
            for metric_name, count in valid_samples.items()
        },
        "complete": all(count == sample_count for count in valid_samples.values()),
    }


def evaluate_one(
    record: dict[str, Any],
    metric_name: str,
    *,
    model: str,
    context_top_k: int,
    timeout: float,
    attempts: int,
) -> tuple[str, str, float | None, str | None, float]:
    _install_ragas_langchain_shims()
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.run_config import RunConfig

    metric_factories = {
        "answer_relevancy": lambda: AnswerRelevancy(strictness=1),
        "context_recall": lambda: ContextRecall(max_retries=1),
        "context_precision": lambda: ContextPrecision(max_retries=1),
    }
    payload = dict(record)
    if metric_name in {"context_recall", "context_precision"}:
        payload["retrieved_contexts"] = payload["retrieved_contexts"][:context_top_k]
        payload["source_chunk_ids"] = payload.get("source_chunk_ids", [])[:context_top_k]
        payload["source_documents"] = payload.get("source_documents", [])[:context_top_k]
    settings = get_settings()
    started = time.perf_counter()
    value = None
    error = None
    for attempt in range(1, attempts + 1):
        try:
            result = evaluate(
                Dataset.from_list([payload]),
                metrics=[metric_factories[metric_name]()],
                llm=get_chat_model(
                    settings,
                    model,
                    temperature=0.0,
                    timeout=timeout,
                    max_retries=1,
                ),
                embeddings=get_embeddings(settings),
                run_config=RunConfig(
                    timeout=timeout,
                    max_retries=1,
                    max_wait=10,
                    max_workers=1,
                    seed=42,
                ),
                raise_exceptions=True,
                batch_size=1,
                show_progress=False,
            )
            raw_value = result.scores[0].get(metric_name)
            value = float(raw_value)
            if math.isnan(value):
                raise RuntimeError("Ragas returned NaN.")
            error = None
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(30 * attempt)
    return record["id"], metric_name, value, error, time.perf_counter() - started


def run(
    records_path: Path,
    output_path: Path,
    *,
    model: str,
    context_top_k: int,
    workers: int,
    timeout: float,
    attempts: int,
) -> dict[str, Any]:
    records = load_jsonl(records_path)
    checkpoint = load_checkpoint(output_path, records)
    checkpoint["evaluation_model"] = model
    checkpoint["context_top_k"] = context_top_k
    save_checkpoint(output_path, checkpoint)

    for metric_name in METRICS:
        pending = [
            record
            for record in records
            if checkpoint["records"][record["id"]]["metrics"].get(metric_name) is None
        ]
        if not pending:
            continue
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    evaluate_one,
                    record,
                    metric_name,
                    model=model,
                    context_top_k=context_top_k,
                    timeout=timeout,
                    attempts=attempts,
                )
                for record in pending
            ]
            for future in as_completed(futures):
                record_id, completed_metric, value, error, elapsed = future.result()
                target = checkpoint["records"][record_id]
                target["metrics"][completed_metric] = value
                target["elapsed_seconds"][completed_metric] = round(elapsed, 2)
                if error:
                    target["errors"][completed_metric] = error
                else:
                    target["errors"].pop(completed_metric, None)
                save_checkpoint(output_path, checkpoint)
                print(
                    f"{completed_metric} {record_id}: "
                    f"{value if value is not None else error} ({elapsed:.1f}s)",
                    flush=True,
                )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable per-sample Ragas evaluation.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="glm-4.7")
    parser.add_argument("--context-top-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args()
    result = run(
        args.records,
        args.output,
        model=args.model,
        context_top_k=args.context_top_k,
        workers=args.workers,
        timeout=args.timeout,
        attempts=args.attempts,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
