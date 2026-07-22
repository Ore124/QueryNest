from __future__ import annotations

from pathlib import Path
from typing import Any

from .document_parsers import ParsedBlock


IMAGE_OCR_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class PaddleOcrParser:
    def __init__(self, *, language: str = "ch", device: str = "cpu") -> None:
        self.language = language
        self.device = device
        self._engine: Any | None = None

    def parse(self, image_path: Path) -> ParsedBlock:
        suffix = image_path.suffix.lower()
        if suffix not in IMAGE_OCR_EXTENSIONS:
            raise ValueError(f"Unsupported image OCR file type: {suffix}")
        text = self.extract_text(image_path)
        return ParsedBlock(
            text=f"# 图片资产: {image_path.stem}\n\n{text}",
            content_type="image",
            parser="paddleocr",
            section=image_path.stem,
        )

    def extract_text(self, image_path: Path) -> str:
        engine = self._get_engine()
        results = engine.predict(input=str(image_path))
        text = paddle_result_to_text(list(results)).strip()
        return text or f"图片 {image_path.name} 未识别到文字。"

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is required for image parsing. "
                    'Install "paddleocr==3.7.0" and "paddlepaddle==3.3.1".'
                ) from exc
            self._engine = PaddleOCR(
                lang=self.language,
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        return self._engine


def paddle_result_to_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, (list, tuple)):
        return "\n".join(filter(None, (paddle_result_to_text(item) for item in result)))
    if not isinstance(result, dict):
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if payload is not None:
            return paddle_result_to_text(payload)
        return ""
    if isinstance(result.get("res"), dict):
        return paddle_result_to_text(result["res"])
    texts = result.get("rec_texts")
    if isinstance(texts, list):
        return "\n".join(str(text).strip() for text in texts if str(text).strip())
    text = result.get("text")
    return str(text).strip() if text else ""
