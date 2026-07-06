from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_parsers import DocumentParseResult, ParsedBlock, delimited_table_to_block, read_text


SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
    ".csv",
    ".tsv",
    ".xlsx",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
}
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/markdown",
    "text/x-markdown",
    "text/plain",
    "image/png",
    "image/jpeg",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_EXCLUDES = ["**/tmp/**", "**/*测试问题集.md", "**/RAG测试问题集.md"]


@dataclass(frozen=True)
class RawDocument:
    text: str
    source_path: str
    source_name: str
    file_type: str
    scenario: str
    page: int | None = None
    section: str | None = None
    content_type: str = "text"
    parser: str = "text"
    parse_notice: str | None = None
    table_id: str | None = None
    table_markdown: str | None = None
    table_json: str | None = None
    table_html: str | None = None
    cache_key: str | None = None
    parser_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, object]


def discover_files(
    root: Path,
    include_images: bool = True,
    excludes: list[str] | None = None,
    mime_types: dict[str, str] | None = None,
) -> list[Path]:
    root = root.resolve()
    patterns = excludes or DEFAULT_EXCLUDES
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        normalized = path.as_posix()
        if any(fnmatch.fnmatch(normalized, pattern.replace("\\", "/")) for pattern in patterns):
            continue
        suffix = path.suffix.lower()
        mime_type = _mime_type_for_path(path, mime_types)
        if suffix not in SUPPORTED_EXTENSIONS and not _is_supported_mime_type(mime_type):
            continue
        if (suffix in IMAGE_EXTENSIONS or _is_image_mime_type(mime_type)) and not include_images:
            continue
        files.append(path)
    return sorted(files, key=lambda item: str(item).lower())


def load_documents(
    root: Path,
    parsers: Any,
    include_images: bool = True,
    mime_types: dict[str, str] | None = None,
) -> list[RawDocument]:
    root = root.resolve()
    documents: list[RawDocument] = []
    for path in discover_files(root, include_images=include_images, mime_types=mime_types):
        documents.extend(
            load_one(
                path,
                root,
                parsers,
                include_images=include_images,
                mime_type=_mime_type_for_path(path, mime_types),
            )
        )
    return documents


def load_one(
    path: Path,
    root: Path,
    parsers: Any,
    include_images: bool = True,
    mime_type: str | None = None,
) -> list[RawDocument]:
    suffix = path.suffix.lower()
    scenario = derive_scenario(path, root)
    if suffix in {".csv", ".tsv"}:
        block = delimited_table_to_block(path, "," if suffix == ".csv" else "\t")
        return [_to_raw_document(block, path, scenario)]
    if suffix in {".md", ".txt", ".xlsx", ".pdf"} | IMAGE_EXTENSIONS or _is_supported_mime_type(mime_type):
        parsed = parsers.parse(path, scenario, include_images=include_images, mime_type=mime_type)
        if isinstance(parsed, DocumentParseResult):
            return [_to_raw_document(block, path, scenario, file_type=parsed.file_type) for block in parsed.blocks]
        return [_to_raw_document(block, path, scenario) for block in parsed]
    return []


def _mime_type_for_path(path: Path, mime_types: dict[str, str] | None) -> str | None:
    if not mime_types:
        return None
    resolved = str(path.resolve())
    return mime_types.get(resolved) or mime_types.get(str(path)) or mime_types.get(path.name)


def _is_supported_mime_type(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    return mime_type.split(";", 1)[0].strip().lower() in SUPPORTED_MIME_TYPES


def _is_image_mime_type(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    return mime_type.split(";", 1)[0].strip().lower() in {"image/png", "image/jpeg"}


def split_documents(documents: list[RawDocument], chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks: list[DocumentChunk] = []
    for document_index, document in enumerate(documents):
        if document.content_type == "table":
            parts = [(part, "table") for part in split_table_markdown(document.text, chunk_size)]
            parts.extend((part, "table_row") for part in split_table_rows(document.text))
        else:
            parts = [(part, document.content_type) for part in splitter.split_text(document.text)]
        for index, (part, content_type) in enumerate(parts):
            clean_text = part.strip()
            if not clean_text:
                continue
            chunk_id = make_chunk_id(document.source_path, document.page, document_index, index, clean_text)
            metadata = {
                "chunk_id": chunk_id,
                "source_path": document.source_path,
                "source_name": document.source_name,
                "file_type": document.file_type,
                "scenario": document.scenario,
                "section": document.section or infer_section(clean_text),
                "page": document.page,
                "document_index": document_index,
                "chunk_index": index,
                "content_type": content_type,
                "parser": document.parser,
                "table_id": document.table_id,
                "table_markdown": document.table_markdown,
                "table_json": document.table_json,
                "table_html": document.table_html,
                "parse_notice": document.parse_notice,
                "cache_key": document.cache_key,
                "parser_metadata": document.parser_metadata,
            }
            chunks.append(DocumentChunk(chunk_id=chunk_id, text=clean_text, metadata=metadata))
    return chunks


def split_table_markdown(markdown: str, chunk_size: int) -> list[str]:
    lines = [line.rstrip() for line in markdown.strip().splitlines()]
    header_index = next((index for index, line in enumerate(lines) if line.strip().startswith("|")), None)
    if header_index is None or header_index + 1 >= len(lines):
        return [markdown]
    prefix = lines[:header_index]
    header = lines[header_index : header_index + 2]
    body = [line for line in lines[header_index + 2 :] if line.strip()]
    if not body:
        return [markdown]
    base = prefix + header
    chunks: list[str] = []
    current_rows: list[str] = []
    for row in body:
        candidate = "\n".join(base + current_rows + [row])
        if current_rows and len(candidate) > chunk_size:
            chunks.append("\n".join(base + current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)
    if current_rows:
        chunks.append("\n".join(base + current_rows))
    return chunks


def split_table_rows(markdown: str) -> list[str]:
    lines = [line.rstrip() for line in markdown.strip().splitlines()]
    header_index = next((index for index, line in enumerate(lines) if line.strip().startswith("|")), None)
    if header_index is None or header_index + 1 >= len(lines):
        return []
    prefix = lines[:header_index]
    header = lines[header_index : header_index + 2]
    body = [line for line in lines[header_index + 2 :] if line.strip()]
    base = prefix + header
    return ["\n".join(base + [row]) for row in body]


def derive_scenario(path: Path, root: Path) -> str:
    try:
        first = path.resolve().relative_to(root.resolve()).parts[0]
    except ValueError:
        first = path.parent.name
    if "_" in first and first[:2].isdigit():
        first = first.split("_", 1)[1]
    return first or "默认知识库"


def extract_first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def infer_section(text: str) -> str | None:
    return extract_first_heading(text)


def make_chunk_id(source_path: str, page: int | None, document_index: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(
        f"{source_path}|{page}|{document_index}|{chunk_index}|{text[:160]}".encode("utf-8")
    ).hexdigest()
    return digest[:20]


def _to_raw_document(
    block: ParsedBlock | RawDocument,
    path: Path,
    scenario: str,
    file_type: str | None = None,
) -> RawDocument:
    if isinstance(block, RawDocument):
        return block
    return RawDocument(
        text=block.text,
        source_path=str(path),
        source_name=path.name,
        file_type=file_type or path.suffix.lower().lstrip("."),
        scenario=scenario,
        page=block.page,
        section=block.section,
        content_type=block.content_type,
        parser=block.parser,
        parse_notice=block.parse_notice,
        table_id=block.table_id,
        table_markdown=block.table_markdown,
        table_json=block.table_json,
        table_html=block.table_html,
        cache_key=block.cache_key,
        parser_metadata=block.parser_metadata,
    )
