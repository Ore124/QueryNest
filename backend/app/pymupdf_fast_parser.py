from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any


MIN_SIMPLE_TEXT_LEN = 200
MAX_SIMPLE_GARBLED_SCORE = 0.05
MAX_SIMPLE_IMAGE_AREA_RATIO = 0.20
HIGH_IMAGE_AREA_RATIO = 0.45
HIGH_DRAWING_COUNT = 40
HIGH_MATH_SCORE = 0.02

MOJIBAKE_MARKERS = ("�", "锟斤拷", "ï¿½", "□□", "\x00")
MATH_CHARS = set("∑∫√∞≈≠≤≥±×÷∂πθλμσΔΣΩαβγ^=<>")
MATH_PATTERNS = [
    re.compile(r"\b[a-zA-Z]\s*[=<>]\s*[-+*/()a-zA-Z0-9]+"),
    re.compile(r"\\(?:frac|sum|int|sqrt|alpha|beta|gamma|theta|lambda)\b"),
]


def extract_text_with_pymupdf(pdf_path: Path) -> str:
    """Extract selectable PDF text quickly with PyMuPDF."""
    fitz = _fitz()
    page_texts: list[str] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            text = _normalize_text(page.get_text("text"))
            if not text:
                continue
            if len(document) == 1:
                page_texts.append(text)
            else:
                page_texts.append(f"## Page {page_index}\n\n{text}")
    return "\n\n".join(page_texts).strip()


def analyze_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Profile PDF pages for fast text extraction vs. MinerU routing."""
    fitz = _fitz()
    profiles: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = _normalize_text(page.get_text("text"))
            quality = score_text_quality(text)
            image_area_ratio = _image_area_ratio(page)
            drawing_count = len(page.get_drawings())
            math_score = _math_score(text)
            route, reason = _route_page(
                text_len=int(quality["text_len"]),
                garbled_score=float(quality["garbled_score"]),
                image_area_ratio=image_area_ratio,
                drawing_count=drawing_count,
                math_score=math_score,
            )
            profiles.append(
                {
                    "page_index": page_index,
                    "text_len": quality["text_len"],
                    "garbled_score": quality["garbled_score"],
                    "image_area_ratio": image_area_ratio,
                    "drawing_count": drawing_count,
                    "math_score": math_score,
                    "route": route,
                    "reason": reason,
                }
            )
    return profiles


def score_text_quality(text: str) -> dict[str, object]:
    clean = _normalize_text(text)
    text_len = len(clean)
    if text_len == 0:
        return {
            "text_len": 0,
            "garbled_score": 1.0,
            "empty": True,
            "repeat_score": 0.0,
            "quality_score": 0.0,
        }
    garbled_score = _garbled_score(clean)
    repeat_score = _repeat_score(clean)
    length_score = min(text_len / MIN_SIMPLE_TEXT_LEN, 1.0)
    quality_score = max(0.0, min(1.0, length_score * (1.0 - garbled_score) * (1.0 - repeat_score)))
    return {
        "text_len": text_len,
        "garbled_score": round(garbled_score, 4),
        "empty": False,
        "repeat_score": round(repeat_score, 4),
        "quality_score": round(quality_score, 4),
    }


def _route_page(
    *,
    text_len: int,
    garbled_score: float,
    image_area_ratio: float,
    drawing_count: int,
    math_score: float,
) -> tuple[str, str]:
    if text_len < MIN_SIMPLE_TEXT_LEN and image_area_ratio >= HIGH_IMAGE_AREA_RATIO:
        return "mineru_ocr", "text layer is sparse and image area is high"
    if text_len > 0 and garbled_score > MAX_SIMPLE_GARBLED_SCORE:
        return "mineru_ocr", "text layer appears garbled"
    if drawing_count >= HIGH_DRAWING_COUNT:
        return "mineru_complex", "many drawings or line elements suggest tables or layout complexity"
    if math_score >= HIGH_MATH_SCORE:
        return "mineru_complex", "math or formula features are present"
    if image_area_ratio >= HIGH_IMAGE_AREA_RATIO:
        return "mineru_complex", "large image area suggests complex page layout"
    if text_len >= MIN_SIMPLE_TEXT_LEN and image_area_ratio <= MAX_SIMPLE_IMAGE_AREA_RATIO:
        return "simple_text", "selectable text is sufficient and page image area is low"
    return "mineru_complex", "page does not meet simple text extraction thresholds"


def _garbled_score(text: str) -> float:
    if not text:
        return 1.0
    marker_chars = sum(text.count(marker) * len(marker) for marker in MOJIBAKE_MARKERS)
    suspect_chars = 0
    for char in text:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Co", "Cs"}:
            suspect_chars += 1
        elif char == "\ufffd":
            suspect_chars += 1
    return min(1.0, (marker_chars + suspect_chars) / max(len(text), 1))


def _repeat_score(text: str) -> float:
    if not text:
        return 0.0
    longest_run = 1
    current_run = 1
    previous = ""
    for char in text:
        if char == previous and not char.isspace():
            current_run += 1
        else:
            longest_run = max(longest_run, current_run)
            current_run = 1
            previous = char
    longest_run = max(longest_run, current_run)
    return min(1.0, longest_run / max(len(text), 1))


def _math_score(text: str) -> float:
    clean = _normalize_text(text)
    if not clean:
        return 0.0
    symbol_hits = sum(1 for char in clean if char in MATH_CHARS)
    pattern_hits = sum(len(pattern.findall(clean)) for pattern in MATH_PATTERNS)
    score = (symbol_hits + pattern_hits * 8) / len(clean)
    return round(min(1.0, score), 4)


def _image_area_ratio(page: Any) -> float:
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    image_area = 0.0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox]
        width = max(0.0, min(x1, page.rect.x1) - max(x0, page.rect.x0))
        height = max(0.0, min(y1, page.rect.y1) - max(y0, page.rect.y0))
        image_area += width * height
    return round(min(1.0, image_area / page_area), 4)


def _normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", text or "").strip()


def _fitz():
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for fast PDF parsing.") from exc
    return fitz
