"""Pure helpers for removing common extraction noise before ingestion."""

import re
import unicodedata
from collections import Counter


_PAGE_NUMBER_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|第\s*\d+\s*页)$",
    re.IGNORECASE,
)
_TOC_LINE_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[^\n]{1,100}?\s*(?:\.{2,}|…+)\s*\d+\s*$")
_TOC_HEADING_RE = re.compile(r"(?:(?:table\s+of\s+)?contents|目\s*录)", re.IGNORECASE)


def clean_extracted_text(text: str, *, file_type: str) -> str:
    """Normalize non-table parser output without changing its semantic content."""
    normalized = _normalize_whitespace(text)
    if file_type != "pdf":
        return normalized

    lines = [
        line
        for line in normalized.splitlines()
        if not _looks_like_toc_line(line) and not _is_toc_heading(line)
    ]
    lines = _remove_recurring_margin_lines(lines)
    return _join_wrapped_lines(lines)


def _normalize_whitespace(text: str) -> str:
    """Trim line-level whitespace while retaining paragraph boundaries."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False
    return "\n".join(normalized_lines).strip()


def _looks_like_toc_line(line: str) -> bool:
    stripped = line.strip()
    return bool(_PAGE_NUMBER_RE.fullmatch(stripped) or _TOC_LINE_RE.fullmatch(stripped))


def _is_toc_heading(line: str) -> bool:
    return bool(_TOC_HEADING_RE.fullmatch(line.strip()))


def _remove_recurring_margin_lines(lines: list[str]) -> list[str]:
    candidates = [line.strip() for line in lines if _is_margin_candidate(line)]
    repeated = {line for line, count in Counter(candidates).items() if count >= 2}
    return [line for line in lines if line.strip() not in repeated]


def _is_margin_candidate(line: str) -> bool:
    stripped = line.strip()
    return (
        bool(stripped)
        and len(stripped) <= 80
        and not stripped.endswith((".", "!", "?", ":", ";"))
        and not stripped.startswith("#")
        and "|" not in stripped
    )


def _join_wrapped_lines(lines: list[str]) -> str:
    """Join extraction line wraps while preserving blank-line paragraph breaks."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line.strip())
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def chunk_rejection_reason(text: str, *, content_type: str) -> str | None:
    """Return the reason a non-tabular chunk should not reach the index."""
    if content_type in {"table", "table_row"}:
        return None
    if _is_title_only(text):
        return "title_only"
    if _looks_like_toc_block(text):
        return "table_of_contents"
    if _garbled_ratio(text) > 0.20:
        return "garbled"
    return None


def is_duplicate_or_overlap(text: str, seen: set[str], seen_normalized: list[str]) -> bool:
    """Track normalized chunks and identify exact or containment overlap."""
    normalized = normalize_for_deduplication(text)
    if normalized in seen:
        return True
    if any(normalized in prior or prior in normalized for prior in seen_normalized):
        return True
    seen.add(normalized)
    seen_normalized.append(normalized)
    return False


def normalize_for_deduplication(text: str) -> str:
    """Normalize only presentation differences before source-local deduplication."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _is_title_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    line = lines[0]
    return bool(re.fullmatch(r"#{1,6}\s+.+", line))


def _looks_like_toc_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    toc_lines = sum(_looks_like_toc_line(line) for line in lines)
    has_contents_heading = any(_is_toc_heading(line) for line in lines)
    return toc_lines == len(lines) or (has_contents_heading and toc_lines > 0)


def _garbled_ratio(text: str) -> float:
    characters = [char for char in text if not char.isspace()]
    if not characters:
        return 0.0
    garbled = sum(
        char == "\ufffd" or unicodedata.category(char).startswith("C")
        for char in characters
    )
    return garbled / len(characters)
