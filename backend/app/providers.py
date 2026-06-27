from __future__ import annotations

import hashlib
from typing import Iterable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .cache import JsonCache
from .settings import Settings


class HashEmbeddings(Embeddings):
    """Deterministic local embeddings for tests and offline smoke checks."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


class CachedEmbeddings(Embeddings):
    """Exact-match Redis cache for query embeddings."""

    def __init__(
        self,
        inner: Embeddings,
        *,
        cache: JsonCache,
        model: str,
        dimensions: int,
        ttl_seconds: int,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.model = model
        self.dimensions = dimensions
        self.ttl_seconds = ttl_seconds
        self.last_query_cache_hit: bool | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        payload = {"model": self.model, "dimensions": self.dimensions, "text": text}
        cached = self.cache.get_json("embedding-query", payload)
        if isinstance(cached, dict) and isinstance(cached.get("embedding"), list):
            self.last_query_cache_hit = True
            return [float(value) for value in cached["embedding"]]
        self.last_query_cache_hit = False
        embedding = self.inner.embed_query(text)
        self.cache.set_json(
            "embedding-query",
            payload,
            {"embedding": embedding},
            ttl_seconds=self.ttl_seconds,
        )
        return embedding


def get_embeddings(settings: Settings, *, offline: bool = False) -> Embeddings:
    if offline or not settings.zai_api_key:
        return HashEmbeddings(dimensions=min(settings.embedding_dimensions, 256))
    return OpenAIEmbeddings(
        model=settings.default_embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.zai_api_key,
        base_url=settings.zai_api_base,
        chunk_size=64,
        check_embedding_ctx_length=False,
    )


def get_chat_model(
    settings: Settings,
    model: str | None = None,
    *,
    temperature: float = 0.2,
    timeout: float = 300,
    max_retries: int = 5,
    thinking: bool | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    if not settings.zai_api_key:
        raise RuntimeError("ZAI_API_KEY is required for model calls.")
    options: dict[str, object] = {}
    if thinking is not None:
        options["extra_body"] = {
            "thinking": {"type": "enabled" if thinking else "disabled"}
        }
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    return ChatOpenAI(
        model=model or settings.default_chat_model,
        api_key=settings.zai_api_key,
        base_url=settings.zai_api_base,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        **options,
    )


def batch_iter(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
