from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from typing import Any

import pymilvus
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)


DENSE_FIELD = "dense"
SPARSE_FIELD = "sparse"
TEXT_FIELD = "text"


DOCS = [
    {
        "chunk_id": "chunk-001",
        "kb_id": "kb-poc",
        "doc_id": "doc-policy",
        "index_version": "poc-v1",
        "text": "Refund requests over 5000 require finance manager approval.",
    },
    {
        "chunk_id": "chunk-002",
        "kb_id": "kb-poc",
        "doc_id": "doc-policy",
        "index_version": "poc-v1",
        "text": "Employee travel reimbursement must include invoices and itinerary.",
    },
    {
        "chunk_id": "chunk-003",
        "kb_id": "kb-poc",
        "doc_id": "doc-runbook",
        "index_version": "poc-v1",
        "text": "For API 500 incidents, check traceId, error logs, and recent deployments.",
    },
    {
        "chunk_id": "chunk-004",
        "kb_id": "kb-poc",
        "doc_id": "doc-runbook",
        "index_version": "poc-v1",
        "text": "Database latency alerts should be correlated with slow query logs.",
    },
    {
        "chunk_id": "chunk-005",
        "kb_id": "kb-poc",
        "doc_id": "doc-hr",
        "index_version": "poc-v1",
        "text": "Annual leave approval follows the team lead and HR workflow.",
    },
    {
        "chunk_id": "chunk-006",
        "kb_id": "kb-poc",
        "doc_id": "doc-security",
        "index_version": "poc-v1",
        "text": "Password reset requests require identity verification before account recovery.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="POC for Milvus BM25 sparse and hybrid search.")
    parser.add_argument("--uri", default=os.environ.get("MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--token", default=os.environ.get("MILVUS_TOKEN", ""))
    parser.add_argument("--collection", default=f"rag_bm25_poc_{int(time.time())}")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary collection after the POC.")
    args = parser.parse_args()

    capability = check_capabilities()
    print_json("capabilities", capability)
    if not capability["supported"]:
        print("STOP: current pymilvus does not expose the required BM25/hybrid APIs.")
        print("Upgrade suggestion: use Milvus 2.6.x+ and pymilvus 2.6.x+ with FunctionType.BM25 and MilvusClient.hybrid_search.")
        return

    client_kwargs: dict[str, str] = {"uri": args.uri}
    if args.token:
        client_kwargs["token"] = args.token
    client = MilvusClient(**client_kwargs)

    if client.has_collection(args.collection):
        client.drop_collection(args.collection)

    try:
        create_collection(client, args.collection)
        print_json("collection_schema", describe_collection(client, args.collection))

        rows = [
            {
                **doc,
                DENSE_FIELD: dense_embedding(doc["text"]),
            }
            for doc in DOCS
        ]
        client.insert(collection_name=args.collection, data=rows)
        client.flush(collection_name=args.collection)
        client.load_collection(collection_name=args.collection)

        query = "API 500 traceId error logs"
        query_dense = dense_embedding(query)
        filter_expr = 'kb_id == "kb-poc" and index_version == "poc-v1"'
        output_fields = ["chunk_id", "kb_id", "doc_id", "index_version", "text"]

        dense_results = client.search(
            collection_name=args.collection,
            data=[query_dense],
            anns_field=DENSE_FIELD,
            limit=3,
            filter=filter_expr,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE"},
        )
        print_json("dense_search", simplify_results(dense_results))

        bm25_results = client.search(
            collection_name=args.collection,
            data=[query],
            anns_field=SPARSE_FIELD,
            limit=3,
            filter=filter_expr,
            output_fields=output_fields,
            search_params={"metric_type": "BM25"},
        )
        print_json("bm25_sparse_search", simplify_results(bm25_results))

        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field=DENSE_FIELD,
            param={"metric_type": "COSINE"},
            limit=5,
            filter=filter_expr,
        )
        bm25_req = AnnSearchRequest(
            data=[query],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "BM25"},
            limit=5,
            filter=filter_expr,
        )

        hybrid_rrf = client.hybrid_search(
            collection_name=args.collection,
            reqs=[dense_req, bm25_req],
            ranker=RRFRanker(k=60),
            limit=5,
            output_fields=output_fields,
        )
        print_json("hybrid_search_rrf", simplify_results(hybrid_rrf))

        hybrid_weighted = client.hybrid_search(
            collection_name=args.collection,
            reqs=[dense_req, bm25_req],
            ranker=WeightedRanker(0.4, 0.6),
            limit=5,
            output_fields=output_fields,
        )
        print_json("hybrid_search_weighted", simplify_results(hybrid_weighted))

        print_json(
            "summary",
            {
                "pymilvus_version": pymilvus.__version__,
                "collection": args.collection,
                "query": query,
                "dense_top_chunk": top_chunk_id(dense_results),
                "bm25_top_chunk": top_chunk_id(bm25_results),
                "hybrid_rrf_top_chunk": top_chunk_id(hybrid_rrf),
                "hybrid_weighted_top_chunk": top_chunk_id(hybrid_weighted),
            },
        )
    finally:
        if not args.keep and client.has_collection(args.collection):
            client.drop_collection(args.collection)


def check_capabilities() -> dict[str, Any]:
    bm25_value = getattr(FunctionType, "BM25", None)
    return {
        "pymilvus_version": pymilvus.__version__,
        "has_function_type_bm25": bm25_value is not None,
        "has_hybrid_search": hasattr(MilvusClient, "hybrid_search"),
        "has_sparse_float_vector": getattr(DataType, "SPARSE_FLOAT_VECTOR", None) is not None,
        "has_rankers": RRFRanker is not None and WeightedRanker is not None,
        "supported": (
            bm25_value is not None
            and hasattr(MilvusClient, "hybrid_search")
            and getattr(DataType, "SPARSE_FLOAT_VECTOR", None) is not None
        ),
    }


def create_collection(client: MilvusClient, collection_name: str) -> None:
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(field_name="kb_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="index_version", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name=TEXT_FIELD, datatype=DataType.VARCHAR, max_length=4096, enable_analyzer=True)
    schema.add_field(field_name=DENSE_FIELD, datatype=DataType.FLOAT_VECTOR, dim=16)
    schema.add_field(field_name=SPARSE_FIELD, datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=[TEXT_FIELD],
            output_field_names=[SPARSE_FIELD],
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(field_name=DENSE_FIELD, index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name=SPARSE_FIELD, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def dense_embedding(text: str, dimensions: int = 16) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().replace(".", " ").replace(",", " ").split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def describe_collection(client: MilvusClient, collection_name: str) -> dict[str, Any]:
    raw = client.describe_collection(collection_name=collection_name)
    return json.loads(json.dumps(raw, default=str))


def simplify_results(results: Any) -> list[list[dict[str, Any]]]:
    simplified: list[list[dict[str, Any]]] = []
    for hits in results:
        group: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {})
            chunk_id = None
            if isinstance(hit, dict):
                chunk_id = hit.get("id", hit.get("chunk_id"))
            else:
                chunk_id = getattr(hit, "id", getattr(hit, "chunk_id", None))
            if chunk_id is None and isinstance(entity, dict):
                chunk_id = entity.get("chunk_id")
            group.append(
                {
                    "chunk_id": chunk_id,
                    "score": hit.get("distance", hit.get("score")) if isinstance(hit, dict) else getattr(hit, "distance", None),
                    "text": entity.get("text") if isinstance(entity, dict) else None,
                    "doc_id": entity.get("doc_id") if isinstance(entity, dict) else None,
                }
            )
        simplified.append(group)
    return simplified


def top_chunk_id(results: Any) -> str | None:
    simplified = simplify_results(results)
    if not simplified or not simplified[0]:
        return None
    value = simplified[0][0]["chunk_id"]
    return str(value) if value is not None else None


def print_json(label: str, payload: Any) -> None:
    print(f"\n## {label}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
