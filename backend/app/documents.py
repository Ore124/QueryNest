from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_parsers import ParsedBlock, delimited_table_to_block


SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
    ".csv",
    ".tsv",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MINERU_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
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
    table_id: str | None = None
    table_markdown: str | None = None
    table_json: str | None = None
    table_html: str | None = None


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, object]


def discover_files(root: Path, include_images: bool = True, excludes: list[str] | None = None) -> list[Path]:
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
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        if suffix in IMAGE_EXTENSIONS and not include_images:
            continue
        files.append(path)
    return sorted(files, key=lambda item: str(item).lower())


def load_documents(root: Path, parsers: Any, include_images: bool = True) -> list[RawDocument]:
    root = root.resolve()
    documents: list[RawDocument] = []
    for path in discover_files(root, include_images=include_images):
        documents.extend(load_one(path, root, parsers))
    return documents


def load_one(path: Path, root: Path, parsers: Any) -> list[RawDocument]:
    suffix = path.suffix.lower()
    scenario = derive_scenario(path, root)
    if suffix in {".md", ".txt"}:
        text = read_text(path)
        return [
            RawDocument(
                text=text,
                source_path=str(path),
                source_name=path.name,
                file_type=suffix.lstrip("."),
                scenario=scenario,
                section=extract_first_heading(text),
            )
        ]
    if suffix in {".csv", ".tsv"}:
        block = delimited_table_to_block(path, "," if suffix == ".csv" else "\t")
        return [_to_raw_document(block, path, scenario)]
    if suffix in MINERU_EXTENSIONS | IMAGE_EXTENSIONS:
        return [_to_raw_document(block, path, scenario) for block in parsers.parse(path, scenario)]
    return []


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def split_documents(documents: list[RawDocument], chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks: list[DocumentChunk] = []
    for document_index, document in enumerate(documents):
        parts = (
            split_table_markdown(document.text, chunk_size)
            if document.content_type == "table"
            else splitter.split_text(document.text)
        )
        for index, part in enumerate(parts):
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
                "content_type": document.content_type,
                "parser": document.parser,
                "table_id": document.table_id,
                "table_markdown": document.table_markdown,
                "table_json": document.table_json,
                "table_html": document.table_html,
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


def _to_raw_document(block: ParsedBlock | RawDocument, path: Path, scenario: str) -> RawDocument:
    if isinstance(block, RawDocument):
        return block
    return RawDocument(
        text=block.text,
        source_path=str(path),
        source_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        scenario=scenario,
        page=block.page,
        section=block.section,
        content_type=block.content_type,
        parser=block.parser,
        table_id=block.table_id,
        table_markdown=block.table_markdown,
        table_json=block.table_json,
        table_html=block.table_html,
    )
