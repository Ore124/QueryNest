from __future__ import annotations

import logging
import time
import hashlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .mineru_api_client import MineruApiClient, MineruApiError
from .pymupdf_fast_parser import analyze_pdf_pages, extract_text_with_pymupdf, score_text_quality
from .settings import Settings


PdfParseStrategy = Literal["hybrid", "mineru_api", "pymupdf"]
PdfParseResult = dict[str, object]

logger = logging.getLogger(__name__)

PARSER_VERSIONS = {
    "mineru_api": "mineru-api-v1",
    "pymupdf": "pymupdf-fast-v1",
}


class PdfParseCache:
    def __init__(self) -> None:
        self._request_to_result_key: dict[str, str] = {}
        self._results: dict[str, PdfParseResult] = {}

    def get(self, request_key: str) -> PdfParseResult | None:
        result_key = self._request_to_result_key.get(request_key)
        if result_key is None:
            return None
        result = self._results.get(result_key)
        return copy.deepcopy(result) if result is not None else None

    def set(self, request_key: str, result_key: str, result: PdfParseResult) -> None:
        self._request_to_result_key[request_key] = result_key
        self._results[result_key] = copy.deepcopy(result)


@dataclass(frozen=True)
class PdfParseRouterConfig:
    strategy: str = "hybrid"
    complex_page_ratio_threshold: float = 0.35
    pymupdf_min_quality_score: float = 0.6
    enable_mineru_fallback: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> "PdfParseRouterConfig":
        return cls(
            strategy=settings.pdf_parse_strategy,
            complex_page_ratio_threshold=settings.pdf_complex_page_ratio_threshold,
            pymupdf_min_quality_score=settings.pdf_pymupdf_min_quality_score,
            enable_mineru_fallback=settings.pdf_enable_mineru_fallback,
        )


class PdfParseRouter:
    def __init__(
        self,
        config: PdfParseRouterConfig,
        mineru_api_client: MineruApiClient,
        *,
        cache: PdfParseCache | None = None,
    ) -> None:
        self.config = config
        self.mineru_api_client = mineru_api_client
        self.cache = cache or PdfParseCache()

    def parse_pdf(self, pdf_path: Path) -> PdfParseResult:
        started = time.perf_counter()
        strategy = self._strategy()
        file_hash = _file_hash(pdf_path)
        page_count = _page_count(pdf_path)
        request_key = self._request_cache_key(file_hash, strategy)
        cached = self.cache.get(request_key)
        if cached is not None:
            logger.info("PDF parse cache hit path=%s selected_strategy=%s file_hash=%s", pdf_path, strategy, file_hash)
            self._log_metrics(
                pdf_path=pdf_path,
                file_hash=file_hash,
                page_count=page_count,
                strategy=strategy,
                parser=str(cached.get("parser", "")),
                mineru_required_page_ratio=_mineru_required_page_ratio(cached.get("page_profiles", [])) if isinstance(cached.get("page_profiles"), list) else None,
                quality_score=cached.get("quality_score"),
                fallback_used=bool(cached.get("fallback_used", False)),
                duration_seconds=time.perf_counter() - started,
                mineru_api_duration_seconds=0.0,
                pymupdf_duration_seconds=0.0,
                error_message=None,
            )
            return cached
        logger.info("PDF parse router selected strategy=%s path=%s", strategy, pdf_path)
        metrics = {
            "mineru_api_duration_seconds": 0.0,
            "pymupdf_duration_seconds": 0.0,
            "error_message": None,
        }
        page_profiles: list[dict[str, Any]] = []
        try:
            if strategy == "mineru_api":
                result = self._parse_with_mineru_api(
                    pdf_path,
                    strategy=strategy,
                    page_profiles=[],
                    fallback_used=False,
                    metrics=metrics,
                )
            elif strategy == "pymupdf":
                result = self._parse_with_pymupdf(
                    pdf_path,
                    strategy=strategy,
                    page_profiles=[],
                    fallback_used=False,
                    metrics=metrics,
                )
            else:
                page_profiles = analyze_pdf_pages(pdf_path)
                mineru_required_page_ratio = _mineru_required_page_ratio(page_profiles)
                logger.info(
                    "PDF hybrid preflight path=%s pages=%d mineru_required_page_ratio=%.4f threshold=%.4f",
                    pdf_path,
                    len(page_profiles),
                    mineru_required_page_ratio,
                    self.config.complex_page_ratio_threshold,
                )
                if mineru_required_page_ratio > self.config.complex_page_ratio_threshold:
                    logger.info("PDF hybrid routed to MinerU API due to mineru_required_page_ratio path=%s", pdf_path)
                    try:
                        result = self._parse_with_mineru_api(
                            pdf_path,
                            strategy=strategy,
                            page_profiles=page_profiles,
                            fallback_used=False,
                            metadata={"mineru_required_page_ratio": mineru_required_page_ratio},
                            metrics=metrics,
                        )
                    except MineruApiError as exc:
                        logger.warning(
                            "PDF hybrid MinerU API failed; falling back to PyMuPDF path=%s error=%s",
                            pdf_path,
                            exc,
                        )
                        result = self._parse_with_pymupdf(
                            pdf_path,
                            strategy=strategy,
                            page_profiles=page_profiles,
                            fallback_used=True,
                            metadata={
                                "mineru_required_page_ratio": mineru_required_page_ratio,
                                "mineru_api_error": str(exc),
                            },
                            metrics=metrics,
                        )
                else:
                    result = self._parse_with_pymupdf(
                        pdf_path,
                        strategy=strategy,
                        page_profiles=page_profiles,
                        fallback_used=False,
                        metrics=metrics,
                    )
                    quality_score = float(result["quality_score"])
                    if quality_score < self.config.pymupdf_min_quality_score:
                        if self.config.enable_mineru_fallback:
                            logger.warning(
                                "PDF hybrid PyMuPDF quality below threshold; falling back to MinerU API path=%s quality=%.4f threshold=%.4f",
                                pdf_path,
                                quality_score,
                                self.config.pymupdf_min_quality_score,
                            )
                            try:
                                result = self._parse_with_mineru_api(
                                    pdf_path,
                                    strategy=strategy,
                                    page_profiles=page_profiles,
                                    fallback_used=True,
                                    metadata={
                                        "pymupdf_quality_score": quality_score,
                                        "mineru_required_page_ratio": mineru_required_page_ratio,
                                    },
                                    metrics=metrics,
                                )
                            except MineruApiError as exc:
                                logger.warning(
                                    "PDF hybrid MinerU API fallback failed; keeping PyMuPDF result path=%s error=%s",
                                    pdf_path,
                                    exc,
                                )
                                result = _with_mineru_api_error(
                                    result,
                                    exc,
                                    mineru_required_page_ratio=mineru_required_page_ratio,
                                )
                        else:
                            logger.warning(
                                "PDF hybrid PyMuPDF quality below threshold but fallback disabled path=%s quality=%.4f threshold=%.4f",
                                pdf_path,
                                quality_score,
                                self.config.pymupdf_min_quality_score,
                            )
            self._store_cache(request_key, file_hash, strategy, result)
            self._log_metrics(
                pdf_path=pdf_path,
                file_hash=file_hash,
                page_count=page_count,
                strategy=strategy,
                parser=str(result.get("parser", "")),
                mineru_required_page_ratio=_mineru_required_page_ratio(result.get("page_profiles", [])) if isinstance(result.get("page_profiles"), list) else None,
                quality_score=result.get("quality_score"),
                fallback_used=bool(result.get("fallback_used", False)),
                duration_seconds=time.perf_counter() - started,
                mineru_api_duration_seconds=float(metrics["mineru_api_duration_seconds"]),
                pymupdf_duration_seconds=float(metrics["pymupdf_duration_seconds"]),
                error_message=None,
            )
            return result
        except Exception as exc:
            metrics["error_message"] = str(exc)
            self._log_metrics(
                pdf_path=pdf_path,
                file_hash=file_hash,
                page_count=page_count,
                strategy=strategy,
                parser="",
                mineru_required_page_ratio=_mineru_required_page_ratio(page_profiles) if page_profiles else None,
                quality_score=None,
                fallback_used=False,
                duration_seconds=time.perf_counter() - started,
                mineru_api_duration_seconds=float(metrics["mineru_api_duration_seconds"]),
                pymupdf_duration_seconds=float(metrics["pymupdf_duration_seconds"]),
                error_message=str(exc),
            )
            raise

    def _parse_with_pymupdf(
        self,
        pdf_path: Path,
        *,
        strategy: str,
        page_profiles: list[dict[str, Any]],
        fallback_used: bool,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, object] | None = None,
    ) -> PdfParseResult:
        started = time.perf_counter()
        try:
            content = extract_text_with_pymupdf(pdf_path)
        finally:
            if metrics is not None:
                metrics["pymupdf_duration_seconds"] = float(metrics.get("pymupdf_duration_seconds", 0.0)) + (
                    time.perf_counter() - started
                )
        quality = score_text_quality(content)
        logger.info(
            "PDF parse router used PyMuPDF path=%s strategy=%s quality=%.4f text_len=%s",
            pdf_path,
            strategy,
            float(quality["quality_score"]),
            quality["text_len"],
        )
        _log_selection(
            path=pdf_path,
            selected_strategy=strategy,
            selected_parser="pymupdf",
            fallback_used=fallback_used,
            mineru_api_called=False,
            paddleocr_called=False,
        )
        merged_metadata = {"text_quality": quality}
        if metadata:
            merged_metadata.update(metadata)
        return {
            "content": content,
            "parser": "pymupdf",
            "strategy": strategy,
            "quality_score": quality["quality_score"],
            "page_profiles": page_profiles,
            "fallback_used": fallback_used,
            "metadata": merged_metadata,
        }

    def _parse_with_mineru_api(
        self,
        pdf_path: Path,
        *,
        strategy: str,
        page_profiles: list[dict[str, Any]],
        fallback_used: bool,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, object] | None = None,
    ) -> PdfParseResult:
        started = time.perf_counter()
        try:
            result = self.mineru_api_client.parse_pdf(pdf_path)
        finally:
            if metrics is not None:
                metrics["mineru_api_duration_seconds"] = float(metrics.get("mineru_api_duration_seconds", 0.0)) + (
                    time.perf_counter() - started
                )
        logger.info(
            "PDF parse router used MinerU API path=%s strategy=%s fallback_used=%s",
            pdf_path,
            strategy,
            fallback_used,
        )
        _log_selection(
            path=pdf_path,
            selected_strategy=strategy,
            selected_parser="mineru_api",
            fallback_used=fallback_used,
            mineru_api_called=True,
            paddleocr_called=False,
        )
        merged_metadata = dict(metadata or {})
        api_metadata = result.get("metadata")
        if isinstance(api_metadata, dict):
            merged_metadata.update(api_metadata)
        return {
            "content": str(result.get("markdown", "")),
            "parser": "mineru_api",
            "strategy": strategy,
            "quality_score": None,
            "page_profiles": page_profiles,
            "fallback_used": fallback_used,
            "metadata": merged_metadata,
        }

    def _strategy(self) -> PdfParseStrategy:
        strategy = self.config.strategy.strip().lower()
        if strategy not in {"hybrid", "mineru_api", "pymupdf"}:
            raise ValueError(
                "Unsupported PDF_PARSE_STRATEGY "
                f"'{self.config.strategy}'. Expected one of: hybrid, mineru_api, pymupdf."
            )
        return strategy  # type: ignore[return-value]

    def _request_cache_key(self, file_hash: str, strategy: str) -> str:
        return _stable_cache_key(
            {
                "file_hash": file_hash,
                "strategy": strategy,
                "key_params": self._key_params(),
            }
        )

    def _store_cache(self, request_key: str, file_hash: str, strategy: str, result: PdfParseResult) -> None:
        parser = str(result.get("parser", ""))
        result_key = _stable_cache_key(
            {
                "file_hash": file_hash,
                "parser": parser,
                "parser_version": PARSER_VERSIONS.get(parser, "unknown"),
                "strategy": strategy,
                "key_params": self._key_params(),
            }
        )
        cache_result = {
            "content": result.get("content", ""),
            "parser": result.get("parser", ""),
            "strategy": result.get("strategy", strategy),
            "quality_score": result.get("quality_score"),
            "page_profiles": result.get("page_profiles", []),
            "fallback_used": result.get("fallback_used", False),
            "metadata": result.get("metadata", {}),
        }
        self.cache.set(request_key, result_key, cache_result)

    def _key_params(self) -> dict[str, object]:
        return {
            "complex_page_ratio_threshold": self.config.complex_page_ratio_threshold,
            "pymupdf_min_quality_score": self.config.pymupdf_min_quality_score,
            "enable_mineru_fallback": self.config.enable_mineru_fallback,
            "mineru_api_enable_formula": getattr(getattr(self.mineru_api_client, "config", None), "enable_formula", None),
            "mineru_api_enable_table": getattr(getattr(self.mineru_api_client, "config", None), "enable_table", None),
            "mineru_api_is_ocr": getattr(getattr(self.mineru_api_client, "config", None), "is_ocr", None),
        }

    @staticmethod
    def _log_metrics(
        *,
        pdf_path: Path,
        file_hash: str,
        page_count: int | None,
        strategy: str,
        parser: str,
        mineru_required_page_ratio: float | None,
        quality_score: object,
        fallback_used: bool,
        duration_seconds: float,
        mineru_api_duration_seconds: float,
        pymupdf_duration_seconds: float,
        error_message: str | None,
    ) -> None:
        logger.info(
            "PDF parse metrics file_name=%s file_hash=%s page_count=%s strategy=%s parser=%s "
            "mineru_required_page_ratio=%s quality_score=%s fallback_used=%s duration_seconds=%.4f "
            "mineru_api_duration_seconds=%.4f pymupdf_duration_seconds=%.4f error_message=%s",
            pdf_path.name,
            file_hash,
            page_count,
            strategy,
            parser,
            None if mineru_required_page_ratio is None else round(mineru_required_page_ratio, 4),
            quality_score,
            fallback_used,
            duration_seconds,
            mineru_api_duration_seconds,
            pymupdf_duration_seconds,
            error_message,
        )


def _mineru_required_page_ratio(page_profiles: list[dict[str, Any]]) -> float:
    if not page_profiles:
        return 0.0
    mineru_required_pages = sum(1 for profile in page_profiles if profile.get("route") != "simple_text")
    return mineru_required_pages / len(page_profiles)


def _with_mineru_api_error(
    result: PdfParseResult,
    error: MineruApiError,
    *,
    mineru_required_page_ratio: float | None,
) -> PdfParseResult:
    updated = dict(result)
    metadata = dict(updated.get("metadata") or {})
    metadata["mineru_api_error"] = str(error)
    metadata["mineru_api_fallback_failed"] = True
    if mineru_required_page_ratio is not None:
        metadata["mineru_required_page_ratio"] = mineru_required_page_ratio
    updated["metadata"] = metadata
    updated["fallback_used"] = True
    return updated


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _page_count(path: Path) -> int:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to count PDF pages.") from exc
    with fitz.open(path) as document:
        return len(document)


def _stable_cache_key(payload: dict[str, object]) -> str:
    import json

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _log_selection(
    *,
    path: Path,
    selected_strategy: str,
    selected_parser: str,
    fallback_used: bool,
    mineru_api_called: bool,
    paddleocr_called: bool | str,
) -> None:
    logger.info(
        "PDF parse decision path=%s selected_strategy=%s selected_parser=%s fallback_used=%s "
        "mineru_api_called=%s paddleocr_called=%s",
        path,
        selected_strategy,
        selected_parser,
        fallback_used,
        mineru_api_called,
        paddleocr_called,
    )
