from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    content_type: str
    parser: str
    page: int | None = None
    section: str | None = None
    table_id: str | None = None
    table_markdown: str | None = None
    table_json: str | None = None
    table_html: str | None = None


class DocumentParserRouter:
    def __init__(self, mineru: "MinerUParser", paddleocr: "PaddleOcrParser") -> None:
        self.mineru = mineru
        self.paddleocr = paddleocr

    def parse(self, path: Path, scenario: str) -> list[ParsedBlock]:
        del scenario
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            return [self.paddleocr.parse(path)]
        return self.mineru.parse(path)


class MinerUParser:
    def __init__(
        self,
        executable: Path,
        output_root: Path,
        image_parser: "PaddleOcrParser",
        *,
        backend: str = "pipeline",
        method: str = "auto",
        language: str = "ch",
    ) -> None:
        self.executable = executable
        self.output_root = output_root
        self.image_parser = image_parser
        self.backend = backend
        self.method = method
        self.language = language

    def parse(self, path: Path) -> list[ParsedBlock]:
        path = path.resolve()
        output_dir = self.output_root / _source_cache_key(path)
        content_list_path = _find_content_list(output_dir, path.stem)
        if content_list_path is None:
            self._run(path, output_dir)
            content_list_path = _find_content_list(output_dir, path.stem)
        if content_list_path is None:
            raise RuntimeError(f"MinerU did not produce a content list for {path.name}.")
        content = json.loads(content_list_path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            raise RuntimeError(f"Unexpected MinerU content list format: {content_list_path}")
        return parse_mineru_content_list(
            content,
            source_stem=path.stem,
            asset_root=content_list_path.parent,
            image_parser=self.image_parser,
        )

    def _run(self, path: Path, output_dir: Path) -> None:
        if not self.executable.exists():
            raise RuntimeError(
                f"MinerU executable was not found at {self.executable}. "
                "Create .mineru-venv and install mineru[pipeline]."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.executable),
            "-p",
            str(path),
            "-o",
            str(output_dir),
            "-b",
            self.backend,
            "-m",
            self.method,
            "-l",
            self.language,
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"MinerU failed for {path.name}: {detail}")


class PaddleOcrParser:
    def __init__(self, *, language: str = "ch", device: str = "cpu") -> None:
        self.language = language
        self.device = device
        self._engine: Any | None = None

    def parse(self, image_path: Path) -> ParsedBlock:
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


def parse_mineru_content_list(
    content: list[dict[str, Any]],
    *,
    source_stem: str,
    asset_root: Path | None = None,
    image_parser: PaddleOcrParser | None = None,
) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    current_section: str | None = None
    for index, item in enumerate(content):
        block_type = str(item.get("type", "text")).lower()
        page = _page_number(item.get("page_idx"))
        if block_type == "table":
            table_html = str(item.get("table_body") or item.get("table_html") or "").strip()
            title = _caption_text(item.get("table_caption")) or current_section or f"{source_stem} 表格 {index + 1}"
            structure = html_table_to_structure(table_html, title=title)
            markdown = table_structure_to_markdown(structure)
            table_id = hashlib.sha1(f"{source_stem}|{index}|{table_html}".encode("utf-8")).hexdigest()[:20]
            blocks.append(
                ParsedBlock(
                    text=markdown,
                    content_type="table",
                    parser="mineru",
                    page=page,
                    section=title,
                    table_id=table_id,
                    table_markdown=markdown,
                    table_json=json.dumps(structure, ensure_ascii=False),
                    table_html=table_html,
                )
            )
            continue
        if block_type == "image":
            image_path = _resolve_image_path(item, asset_root)
            if image_path is None or image_parser is None:
                continue
            ocr_text = image_parser.extract_text(image_path)
            title = _caption_text(item.get("image_caption")) or image_path.stem
            blocks.append(
                ParsedBlock(
                    text=f"### {title}\n\n{ocr_text}",
                    content_type="image",
                    parser="mineru+paddleocr",
                    page=page,
                    section=title,
                )
            )
            continue
        text = _content_text(item)
        if not text:
            continue
        level = item.get("text_level")
        if isinstance(level, int) and 1 <= level <= 6:
            current_section = text.strip()
            text = f"{'#' * level} {current_section}"
        blocks.append(
            ParsedBlock(
                text=text,
                content_type="text",
                parser="mineru",
                page=page,
                section=current_section,
            )
        )
    return blocks


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


def html_table_to_structure(table_html: str, *, title: str) -> dict[str, Any]:
    parser = _TableHtmlParser()
    parser.feed(table_html)
    cells = parser.rows
    text_rows = [[str(cell["text"]) for cell in row] for row in cells]
    header_index = 0
    if (
        len(cells) > 1
        and len(cells[0]) == 1
        and int(cells[0][0].get("colspan", 1)) > 1
        and len(cells[1]) > 1
    ):
        title = str(cells[0][0]["text"]) or title
        header_index = 1
    headers = text_rows[header_index] if text_rows else []
    rows = text_rows[header_index + 1 :] if len(text_rows) > header_index + 1 else []
    return {
        "title": title,
        "headers": headers,
        "rows": rows,
        "header_row_index": header_index,
        "cells": cells,
        "html": table_html,
    }


def table_structure_to_markdown(structure: dict[str, Any]) -> str:
    title = str(structure.get("title") or "表格")
    headers = [str(value) for value in structure.get("headers") or []]
    rows = [[str(value) for value in row] for row in structure.get("rows") or []]
    if not headers:
        return f"### {title}"
    lines = [
        f"### {title}",
        "",
        _markdown_row(headers),
        _markdown_row(["---"] * len(headers)),
    ]
    for row in rows:
        normalized = (row + [""] * len(headers))[: len(headers)]
        lines.append(_markdown_row(normalized))
    return "\n".join(lines)


def delimited_table_to_block(path: Path, delimiter: str) -> ParsedBlock:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [[cell.strip() for cell in row] for row in csv.reader(stream, delimiter=delimiter)]
    rows = [row for row in rows if any(row)]
    headers = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    structure = {
        "title": path.stem,
        "headers": headers,
        "rows": body,
        "cells": [],
        "html": "",
    }
    markdown = table_structure_to_markdown(structure)
    return ParsedBlock(
        text=markdown,
        content_type="table",
        parser="delimited",
        section=path.stem,
        table_id=hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:20],
        table_markdown=markdown,
        table_json=json.dumps(structure, ensure_ascii=False),
    )


class _TableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            attributes = dict(attrs)
            self._cell = {
                "text": "",
                "tag": tag,
                "rowspan": _positive_int(attributes.get("rowspan")),
                "colspan": _positive_int(attributes.get("colspan")),
            }
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._cell["text"] = re.sub(r"\s+", " ", "".join(self._parts)).strip()
            self._row.append(self._cell)
            self._cell = None
            self._parts = []
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _content_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "equation"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _caption_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        text = " ".join(str(item).strip() for item in value if str(item).strip())
        return text or None
    return None


def _page_number(value: Any) -> int | None:
    return value + 1 if isinstance(value, int) else None


def _positive_int(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def _markdown_row(values: list[str]) -> str:
    escaped = [value.replace("|", "\\|").replace("\n", "<br>") for value in values]
    return f"| {' | '.join(escaped)} |"


def _resolve_image_path(item: dict[str, Any], asset_root: Path | None) -> Path | None:
    raw_path = item.get("img_path") or item.get("image_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if not path.is_absolute() and asset_root is not None:
        path = asset_root / path
    return path if path.exists() else None


def _source_cache_key(path: Path) -> str:
    stat = path.stat()
    value = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def _find_content_list(output_dir: Path, source_stem: str) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.rglob("*_content_list.json"))
    matching = [path for path in candidates if source_stem.lower() in path.name.lower()]
    return (matching or candidates)[0] if candidates else None
