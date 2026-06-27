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

    def create_collection(self, *, collection_name: str, dimension: int, **_: Any) -> None:
        self.collections[collection_name] = {"dimension": dimension, "rows": {}}

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
        data: list[list[float]],
        anns_field: str,
        limit: int,
        output_fields: list[str],
        search_params: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        _ = output_fields, search_params
        query = data[0]
        rows = self.collections[collection_name]["rows"].values()
        scored = []
        for row in rows:
            distance = sum((left - right) ** 2 for left, right in zip(query, row[anns_field], strict=True))
            scored.append(
                {
                    "id": row["chunk_id"],
                    "distance": distance,
                    "entity": {"chunk_id": row["chunk_id"]},
                }
            )
        return [sorted(scored, key=lambda item: (item["distance"], item["id"]))[:limit]]


@pytest.fixture
def fake_milvus_client() -> FakeMilvusClient:
    return FakeMilvusClient()
