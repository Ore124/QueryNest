from __future__ import annotations

from typing import Any

import pytest


class FakeMilvusClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def drop_collection(self, *, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, *, collection_name: str, dimension: int | None = None, schema: Any | None = None, **_: Any) -> None:
        self.collections[collection_name] = {"dimension": dimension, "schema": schema, "rows": {}}

    def insert(self, *, collection_name: str, data: list[dict[str, Any]]) -> None:
        rows = self.collections[collection_name]["rows"]
        for row in data:
            rows[row["chunk_id"]] = row

    def flush(self, *, collection_name: str) -> None:
        _ = collection_name

    def load_collection(self, *, collection_name: str) -> None:
        _ = collection_name

    def search(
        self,
        *,
        collection_name: str,
        data: list[Any],
        anns_field: str,
        limit: int,
        output_fields: list[str],
        search_params: dict[str, Any] | None = None,
        filter: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        _ = search_params
        query = data[0]
        rows = [
            row
            for row in self.collections[collection_name]["rows"].values()
            if _matches_filter(row, filter)
        ]
        scored = []
        for row in rows:
            if anns_field == "sparse":
                distance = _text_score(str(query), str(row.get("text", "")))
            else:
                distance = _cosine(query, row[anns_field])
            scored.append(
                {
                    "id": row["chunk_id"],
                    "distance": distance,
                    "entity": {field: row.get(field) for field in output_fields},
                }
            )
        return [sorted(scored, key=lambda item: (item["distance"], item["id"]), reverse=True)[:limit]]

    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[Any],
        ranker: Any,
        limit: int,
        output_fields: list[str],
        **_: Any,
    ) -> list[list[dict[str, Any]]]:
        _ = ranker
        scores: dict[str, float] = {}
        rows_by_id: dict[str, dict[str, Any]] = self.collections[collection_name]["rows"]
        for request in reqs:
            hits = self.search(
                collection_name=collection_name,
                data=request.data,
                anns_field=request.anns_field,
                limit=request.limit,
                output_fields=output_fields,
                search_params=request.param,
                filter=request.filter,
            )[0]
            for rank, hit in enumerate(hits, start=1):
                scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (60 + rank)
        results = []
        for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]:
            row = rows_by_id[chunk_id]
            results.append(
                {
                    "id": chunk_id,
                    "distance": score,
                    "entity": {field: row.get(field) for field in output_fields},
                }
            )
        return [results]


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


def _matches_filter(row: dict[str, Any], expr: str | None) -> bool:
    if not expr:
        return True
    for part in expr.split(" and "):
        if "==" not in part:
            continue
        field, raw_value = part.split("==", 1)
        field = field.strip()
        value = raw_value.strip().strip('"')
        if str(row.get(field, "")) != value:
            return False
    return True


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5 or 1.0
    right_norm = sum(b * b for b in right) ** 0.5 or 1.0
    return dot / (left_norm * right_norm)


def _text_score(query: str, text: str) -> float:
    query_terms = {term.lower() for term in query.split()}
    text_terms = {term.lower().strip(".,;:") for term in text.split()}
    return float(len(query_terms & text_terms))


@pytest.fixture
def fake_milvus_client() -> FakeMilvusClient:
    return FakeMilvusClient()
