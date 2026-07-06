from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from .settings import Settings


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]: ...


class DashScopeReranker:
    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        model: str,
        timeout_seconds: float,
        instruct: str,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.instruct = instruct

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        response = httpx.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {
                    "query": {"text": query},
                    "documents": [{"text": document} for document in documents],
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": min(top_n, len(documents)),
                    "instruct": self.instruct,
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("output", {}).get("results", [])
        results = [
            RerankResult(index=int(item["index"]), score=float(item["relevance_score"]))
            for item in raw_results
            if 0 <= int(item["index"]) < len(documents)
        ]
        if not results:
            raise RuntimeError("Rerank API returned no valid results.")
        return results


class ZhipuReranker:
    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        response = httpx.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = (
            payload.get("results")
            or payload.get("data")
            or payload.get("output", {}).get("results")
            or []
        )
        results = []
        for item in raw_results:
            index = item.get("index", item.get("document_index"))
            score = item.get("relevance_score", item.get("score"))
            if index is None or score is None:
                continue
            index = int(index)
            if 0 <= index < len(documents):
                results.append(RerankResult(index=index, score=float(score)))
        if not results:
            raise RuntimeError("Rerank API returned no valid results.")
        return results


def create_reranker(settings: Settings) -> Reranker | None:
    provider = settings.resolved_rerank_provider
    if provider == "none":
        return None
    api_key = settings.resolved_rerank_api_key
    if not api_key:
        return None
    if provider == "bailian":
        return DashScopeReranker(
            api_key=api_key,
            api_url=settings.resolved_rerank_base_url,
            model=settings.rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
            instruct=settings.rerank_instruct,
        )
    if provider == "zhipu":
        return ZhipuReranker(
            api_key=api_key,
            api_url=settings.resolved_rerank_base_url,
            model=settings.rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
        )
    raise ValueError(f"Unsupported rerank provider: {provider}")
