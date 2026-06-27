from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


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
