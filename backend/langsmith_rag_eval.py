# Run normal: cd "D:\Codex Projects\RAG_project\backend"; ..\.venv\Scripts\python .\langsmith_rag_eval.py --dataset rag_evaluation --metric-k 5 --max-concurrency 1
# Run agentic: cd "D:\Codex Projects\RAG_project\backend"; ..\.venv\Scripts\python .\langsmith_rag_eval.py --dataset rag_evaluation --metric-k 5 --max-concurrency 1 --agentic
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv("../.env", override=True)
import argparse
import os
import re
from typing import Any

import httpx
from langsmith import Client, evaluate


DEFAULT_DATASET_NAME = "rag_evaluation"
DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_TOP_K = 8
DEFAULT_METRIC_K = 5
PRECISION_K = 1


def _first_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_text_from_example(
    inputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None,
    keys: tuple[str, ...],
) -> str | None:
    """Read expected fields from outputs first, then inputs.

    LangSmith CSV uploads often put every column under inputs unless output
    columns are selected explicitly in the UI.
    """
    return _first_text(reference_outputs or {}, keys) or _first_text(inputs, keys)


def _question(inputs: dict[str, Any]) -> str:
    value = _first_text(inputs, ("question", "query", "input"))
    if value is None:
        raise ValueError("Dataset example input must contain one of: question, query, input.")
    return value


def _bool_input(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _top_k(inputs: dict[str, Any]) -> int:
    raw_value = inputs.get("top_k") or os.getenv("LANGSMITH_RAG_TOP_K") or DEFAULT_TOP_K
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_TOP_K


def build_rag_target(api_base: str, *, agentic: bool | None = None):
    def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "message": _question(inputs),
            "top_k": _top_k(inputs),
        }
        input_agentic = _bool_input(inputs.get("agentic"))
        if agentic is not None:
            payload["agentic"] = agentic
        elif input_agentic is not None:
            payload["agentic"] = input_agentic
        for field in ("session_id", "scenario", "model"):
            value = inputs.get(field)
            if isinstance(value, str) and value.strip():
                payload[field] = value.strip()

        with httpx.Client(timeout=300.0) as client:
            response = client.post(f"{api_base.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
        result = response.json()
        sources = result.get("sources") or []
        retrieval_debug = result.get("retrieval_debug", {})

        return {
            "answer": result.get("answer", ""),
            "session_id": result.get("session_id"),
            "agentic_requested": bool(payload.get("agentic")),
            "top_k": payload["top_k"],
            "source_chunk_ids": [
                source.get("chunk_id") for source in sources if source.get("chunk_id")
            ],
            "source_documents": [
                source.get("source_name") for source in sources if source.get("source_name")
            ],
            "retrieval_backend": retrieval_debug.get("retrieval_backend") if isinstance(retrieval_debug, dict) else None,
            "agentic_debug": retrieval_debug.get("agentic") if isinstance(retrieval_debug, dict) else None,
            "retrieval_debug": retrieval_debug,
        }

    return rag_target


def hit(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = _expected_chunk_ids(inputs, reference_outputs)
    k = _metric_k(inputs)
    retrieved = _retrieved_chunk_ids(outputs)[:k]
    if not expected:
        return _skipped(f"hit@{k}", "dataset has no expected_chunk_id/expected_chunk_ids/chunk_id field")
    score = 1 if any(chunk_id in retrieved for chunk_id in expected) else 0
    return {
        "key": f"hit@{k}",
        "score": score,
        "comment": _comment(expected, retrieved, outputs, k=k),
    }


def recall(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = _expected_chunk_ids(inputs, reference_outputs)
    k = _metric_k(inputs)
    retrieved = _retrieved_chunk_ids(outputs)[:k]
    if not expected:
        return _skipped(f"recall@{k}", "dataset has no expected chunk field")
    matched = [chunk_id for chunk_id in expected if chunk_id in retrieved]
    return {
        "key": f"recall@{k}",
        "score": round(len(matched) / len(expected), 4),
        "comment": _comment(expected, retrieved, outputs, matched=matched, k=k),
    }


def precision(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = _expected_chunk_ids(inputs, reference_outputs)
    retrieved = _retrieved_chunk_ids(outputs)
    if not expected:
        return _skipped(f"precision@{PRECISION_K}", "dataset has no expected chunk field")
    k = PRECISION_K
    retrieved_at_k = retrieved[:k]
    matched_count = sum(1 for chunk_id in retrieved_at_k if chunk_id in expected)
    return {
        "key": f"precision@{k}",
        "score": round(matched_count / k, 4),
        "comment": _comment(expected, retrieved_at_k, outputs, matched_count=matched_count, k=k),
    }


def mrr(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = _expected_chunk_ids(inputs, reference_outputs)
    k = _metric_k(inputs)
    retrieved = _retrieved_chunk_ids(outputs)[:k]
    if not expected:
        return _skipped(f"mrr@{k}", "dataset has no expected chunk field")
    first_rank = next(
        (rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in expected),
        None,
    )
    return {
        "key": f"mrr@{k}",
        "score": 0 if first_rank is None else round(1 / first_rank, 4),
        "comment": _comment(expected, retrieved, outputs, first_rank=first_rank, k=k),
    }


def _expected_chunk_ids(
    inputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None,
) -> list[str]:
    raw = _first_text_from_example(
        inputs,
        reference_outputs,
        ("expected_chunk_ids", "expected_chunk_id", "chunk_id"),
    )
    if not raw:
        return []
    return [
        item.strip()
        for item in re.split(r"[;,，；|]", raw)
        if item.strip()
    ]


def _retrieved_chunk_ids(outputs: dict[str, Any]) -> list[str]:
    values = outputs.get("source_chunk_ids") or []
    return [str(value) for value in values if value]


def _metric_k(inputs: dict[str, Any]) -> int:
    raw_value = inputs.get("metric_k") or os.getenv("LANGSMITH_METRIC_K") or DEFAULT_METRIC_K
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_METRIC_K


def _skipped(key: str, reason: str) -> dict[str, Any]:
    return {"key": key, "score": None, "comment": f"Skipped: {reason}."}


def _comment(
    expected: list[str],
    retrieved: list[str],
    outputs: dict[str, Any],
    **extra: Any,
) -> str:
    parts = [
        f"expected={expected}",
        f"retrieved_top5={retrieved[:5]}",
        f"agentic_requested={outputs.get('agentic_requested')}",
        f"retrieval_backend={outputs.get('retrieval_backend')}",
    ]
    for key, value in extra.items():
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval metrics against a LangSmith dataset.")
    parser.add_argument("--dataset", default=os.getenv("LANGSMITH_DATASET", DEFAULT_DATASET_NAME))
    parser.add_argument("--api-base", default=os.getenv("RAG_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--experiment-prefix", default="rag retrieval metrics")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--metric-k", type=int, default=DEFAULT_METRIC_K)
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Send agentic=true to /api/chat so the backend runs the Agentic RAG graph branch.",
    )
    args = parser.parse_args()
    os.environ["LANGSMITH_METRIC_K"] = str(max(1, args.metric_k))

    client = Client()
    if not client.has_dataset(dataset_name=args.dataset):
        raise RuntimeError(f"LangSmith dataset does not exist: {args.dataset}")

    evaluate(
        build_rag_target(args.api_base, agentic=True if args.agentic else None),
        data=args.dataset,
        evaluators=[hit, recall, mrr, precision],
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
        client=client,
    )


if __name__ == "__main__":
    main()
