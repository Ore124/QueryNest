from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .ingestion_cleaning import clean_extracted_text


logger = logging.getLogger(__name__)

PARSER_VERSIONS = {
    "markdown": "markdown-v1",
    "text": "text-v1",
    "xlsx": "xlsx-stdlib-v1",
    "paddleocr": "paddleocr-v1",
    "pymupdf": "pymupdf-fast-v1",
    "mineru_api": "mineru-api-v1",
}
MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/plain": "txt",
    "image/png": "png",
    "image/jpeg": "jpg",
}
MARKDOWN_HEADING_PATTERN = r"^(#{1,6})\s+(.+?)\s*#*\s*$"


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    content_type: str
    parser: str
    page: int | None = None
    section: str | None = None
    parse_notice: str | None = None
    table_id: str | None = None
    table_markdown: str | None = None
    table_json: str | None = None
    table_html: str | None = None
    cache_key: str | None = None
    parser_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class DocumentParseResult:
    file_type: str
    parser: str
    blocks: list[ParsedBlock]
    cache_key: str
    metadata: dict[str, Any]


class DocumentParserRouter:
    def __init__(self, image_ocr: Any, pdf_router: Any) -> None:
        self.image_ocr = image_ocr
        self.pdf_router = pdf_router

    def parse(
        self,
        path: Path,
        scenario: str,
        include_images: bool = True,
        mime_type: str | None = None,
    ) -> DocumentParseResult:
        return self.parse_result(path, scenario, include_images=include_images, mime_type=mime_type)

    def parse_result(
        self,
        path: Path,
        scenario: str,
        include_images: bool = True,
        mime_type: str | None = None,
    ) -> DocumentParseResult:
        del scenario
        file_type = resolve_file_type(path, mime_type)
        if file_type in {"png", "jpg", "jpeg"}:
            if not include_images:
                return _empty_parse_result(path, file_type, "paddleocr")
            block = self.image_ocr.parse(path)
            return _result_from_blocks(path, file_type, block.parser, [block], {"mime_type": mime_type})
        if file_type == "pdf":
            result = self.pdf_router.parse_pdf(path)
            content = clean_extracted_text(str(result.get("content", "")), file_type="pdf")
            notice = _pdf_parse_notice(path, result)
            parser = str(result.get("parser", "pymupdf"))
            metadata = result.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            cache_key = parser_cache_key(path, parser)
            blocks = [
                ParsedBlock(
                    text=content,
                    content_type="text",
                    parser=parser,
                    section=path.stem,
                    parse_notice=notice,
                    cache_key=cache_key,
                    parser_metadata=metadata,
                )
            ] if content else []
            return DocumentParseResult(
                file_type=file_type,
                parser=parser,
                blocks=blocks,
                cache_key=cache_key,
                metadata={"mime_type": mime_type, **metadata},
            )
        if file_type == "xlsx":
            return xlsx_to_parse_result(path, mime_type=mime_type)
        if file_type in {"md", "txt"}:
            return text_file_to_parse_result(path, file_type=file_type, mime_type=mime_type)
        return _empty_parse_result(path, file_type, "unsupported")


def resolve_file_type(path: Path, mime_type: str | None = None) -> str:
    if mime_type:
        normalized = mime_type.split(";", 1)[0].strip().lower()
        if normalized in MIME_TYPES:
            return MIME_TYPES[normalized]
    return path.suffix.lower().lstrip(".")


def text_file_to_parse_result(path: Path, *, file_type: str | None = None, mime_type: str | None = None) -> DocumentParseResult:
    file_type = file_type or resolve_file_type(path, mime_type)
    parser = "markdown" if file_type == "md" else "text"
    text = read_text(path)
    cache_key = parser_cache_key(path, parser)
    blocks: list[ParsedBlock] = []
    for block_text, section in split_markdown_heading_blocks(text):
        clean_text = clean_extracted_text(block_text, file_type=file_type)
        if not clean_text:
            continue
        blocks.append(
            ParsedBlock(
                text=clean_text,
                content_type="text",
                parser=parser,
                section=section or extract_first_heading(clean_text),
                cache_key=cache_key,
                parser_metadata={"mime_type": mime_type, "encoding": "auto"},
            )
        )
    logger.info(
        "Document parse file_name=%s file_type=%s parser=%s blocks=%d cache_key=%s",
        path.name,
        file_type,
        parser,
        len(blocks),
        cache_key,
    )
    return DocumentParseResult(
        file_type=file_type,
        parser=parser,
        blocks=blocks,
        cache_key=cache_key,
        metadata={"mime_type": mime_type, "encoding": "auto", "block_count": len(blocks)},
    )


def xlsx_to_parse_result(path: Path, *, mime_type: str | None = None) -> DocumentParseResult:
    parser = "xlsx"
    cache_key = parser_cache_key(path, parser)
    try:
        blocks = xlsx_to_blocks(path, cache_key=cache_key)
    except (KeyError, ET.ParseError, zipfile.BadZipFile) as exc:
        logger.exception("XLSX parse failed path=%s error=%s", path, exc)
        raise ValueError(f"Failed to parse XLSX file {path.name}: {exc}") from exc
    logger.info(
        "Document parse file_name=%s file_type=xlsx parser=xlsx blocks=%d cache_key=%s",
        path.name,
        len(blocks),
        cache_key,
    )
    return DocumentParseResult(
        file_type="xlsx",
        parser=parser,
        blocks=blocks,
        cache_key=cache_key,
        metadata={"mime_type": mime_type, "sheet_count": len(blocks)},
    )


def _pdf_parse_notice(path: Path, result: dict[str, object]) -> str | None:
    parser = str(result.get("parser", ""))
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if parser == "mineru_api":
        return f"PDF 解析提示：{path.name} 已使用 MinerU API 成功解析。"
    if bool(result.get("fallback_used", False)) and parser == "pymupdf" and metadata.get("mineru_api_error"):
        return (
            f"PDF 解析提示：{path.name} 调用 MinerU API 失败，已降级使用 PyMuPDF；"
            f"错误：{metadata['mineru_api_error']}"
        )
    return None


def _empty_parse_result(path: Path, file_type: str, parser: str) -> DocumentParseResult:
    cache_key = parser_cache_key(path, parser)
    return DocumentParseResult(file_type=file_type, parser=parser, blocks=[], cache_key=cache_key, metadata={})


def _result_from_blocks(
    path: Path,
    file_type: str,
    parser: str,
    blocks: list[ParsedBlock],
    metadata: dict[str, Any] | None = None,
) -> DocumentParseResult:
    cache_key = parser_cache_key(path, parser)
    updated_blocks = [
        ParsedBlock(
            text=block.text,
            content_type=block.content_type,
            parser=block.parser,
            page=block.page,
            section=block.section,
            parse_notice=block.parse_notice,
            table_id=block.table_id,
            table_markdown=block.table_markdown,
            table_json=block.table_json,
            table_html=block.table_html,
            cache_key=block.cache_key or cache_key,
            parser_metadata=block.parser_metadata or metadata,
        )
        for block in blocks
    ]
    return DocumentParseResult(
        file_type=file_type,
        parser=parser,
        blocks=updated_blocks,
        cache_key=cache_key,
        metadata=metadata or {},
    )


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def split_markdown_heading_blocks(text: str) -> list[tuple[str, str | None]]:
    heading_re = re.compile(MARKDOWN_HEADING_PATTERN)
    blocks: list[tuple[str, str | None]] = []
    heading_stack: list[tuple[int, str, str]] = []
    current_lines: list[str] = []
    current_prefix: list[str] = []
    current_section: str | None = None
    current_has_body = False

    def flush() -> None:
        nonlocal current_lines, current_prefix, current_section, current_has_body
        if not current_lines:
            return
        block_text = "\n".join(current_prefix + current_lines).strip()
        if block_text and (current_section is None or current_has_body):
            blocks.append((block_text, current_section))
        current_lines = []
        current_prefix = []
        current_section = None
        current_has_body = False

    for line in text.splitlines():
        match = heading_re.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append((level, title, line.strip()))
            current_prefix = markdown_heading_prefix(heading_stack[:-1])
            current_lines = [line.strip()]
            current_section = " / ".join(item[1] for item in heading_stack)
            current_has_body = False
            continue
        current_lines.append(line)
        if line.strip():
            current_has_body = True

    flush()
    if blocks:
        return blocks
    clean_text = text.strip()
    return [(clean_text, extract_first_heading(clean_text))] if clean_text else []


def markdown_heading_prefix(headings: list[tuple[int, str, str]]) -> list[str]:
    lines: list[str] = []
    for _level, _title, raw in headings:
        if lines:
            lines.append("")
        lines.append(raw)
    if lines:
        lines.append("")
    return lines


def extract_first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def xlsx_to_blocks(path: Path, *, cache_key: str | None = None) -> list[ParsedBlock]:
    cache_key = cache_key or parser_cache_key(path, "xlsx")
    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        sheets = _xlsx_sheet_paths(archive)
        blocks: list[ParsedBlock] = []
        for sheet_name, sheet_path in sheets:
            rows = _xlsx_rows(archive, sheet_path, shared_strings)
            rows = [row for row in rows if any(cell.strip() for cell in row)]
            if not rows:
                continue
            width = max(len(row) for row in rows)
            normalized_rows = [(row + [""] * width)[:width] for row in rows]
            headers = normalized_rows[0]
            body = normalized_rows[1:]
            title = sheet_name or path.stem
            structure = {
                "title": title,
                "headers": headers,
                "rows": body,
                "cells": [],
                "html": "",
            }
            markdown = table_structure_to_markdown(structure)
            table_id = hashlib.sha1(f"{path.resolve()}|{sheet_name}".encode("utf-8")).hexdigest()[:20]
            blocks.append(
                ParsedBlock(
                    text=markdown,
                    content_type="table",
                    parser="xlsx",
                    section=title,
                    table_id=table_id,
                    table_markdown=markdown,
                    table_json=json.dumps(structure, ensure_ascii=False),
                    cache_key=cache_key,
                    parser_metadata={
                        "sheet_name": sheet_name,
                        "rows": len(body),
                        "columns": width,
                        "cache_key": cache_key,
                    },
                )
            )
        return blocks


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.itertext()) for node in root.findall(".//{*}si")]


def _xlsx_sheet_paths(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship_paths = {
        rel.attrib["Id"]: _xlsx_target_path(rel.attrib["Target"])
        for rel in rels.findall("{*}Relationship")
        if "Id" in rel.attrib and "Target" in rel.attrib
    }
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall(".//{*}sheet"):
        relationship_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if relationship_id and relationship_id in relationship_paths:
            sheets.append((sheet.attrib.get("name", ""), relationship_paths[relationship_id]))
    return sheets


def _xlsx_target_path(target: str) -> str:
    normalized = target.lstrip("/")
    if normalized.startswith("xl/"):
        return normalized
    return f"xl/{normalized}"


def _xlsx_rows(archive: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[list[str]] = []
    for row in root.findall(".//{*}sheetData/{*}row"):
        values: list[str] = []
        for cell in row.findall("{*}c"):
            column_index = _xlsx_column_index(cell.attrib.get("r", ""))
            while len(values) < column_index:
                values.append("")
            values.append(_xlsx_cell_text(cell, shared_strings))
        rows.append(values)
    return rows


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("{*}is")
        return "".join(inline.itertext()).strip() if inline is not None else ""
    value = cell.find("{*}v")
    if value is None or value.text is None:
        return ""
    raw = value.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw)].strip()
        except (ValueError, IndexError):
            return raw
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def _xlsx_column_index(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha()).upper()
    if not letters:
        return 0
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


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
    title = str(structure.get("title") or "琛ㄦ牸")
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

    text = read_text(path)
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text), delimiter=delimiter)]
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
        cache_key=parser_cache_key(path, "delimited"),
        parser_metadata={"delimiter": delimiter},
    )


def parser_cache_key(path: Path, parser: str) -> str:
    payload = {
        "file_hash": _file_hash(path),
        "parser": parser,
        "parser_version": PARSER_VERSIONS.get(parser, "unknown"),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _positive_int(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def _markdown_row(values: list[str]) -> str:
    escaped = [value.replace("|", "\\|").replace("\n", "<br>") for value in values]
    return f"| {' | '.join(escaped)} |"
