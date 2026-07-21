from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import Any

from app.kb_metadata import PostgresMetadataStore
from app.settings import get_settings


DEFAULT_INPUT = Path(r"D:\Codex Projects\QA-extractor-main\output\qa_pairs_20260703_102308.csv")


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"\s+", "", text)


def basename_key(value: object) -> str:
    text = str(value or "").replace("\\", "/")
    return normalize(Path(text).name)


def leading_int(value: object) -> int | None:
    match = re.match(r"\s*(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def page_from_chunk(chunk: dict[str, Any]) -> int | None:
    metadata = chunk.get("metadata", {})
    page = metadata.get("page")
    if page is not None:
        try:
            return int(page)
        except (TypeError, ValueError):
            pass
    match = re.search(r"##\s*Page\s+(\d+)", str(chunk.get("text") or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def text_ngrams(text: str, size: int = 3) -> set[str]:
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def recall_score(expected: str, actual: str) -> float:
    expected = normalize(expected)
    actual = normalize(actual)
    if not expected:
        return 0.0
    if expected in actual:
        return 1.0
    grams = text_ngrams(expected)
    if not grams:
        return 0.0
    actual_grams = text_ngrams(actual)
    return len(grams & actual_grams) / len(grams)


def identifier_hit(row: dict[str, str], chunk_text: str) -> bool:
    source = " ".join(str(row.get(key) or "") for key in ("question", "answer", "evidence"))
    identifiers = set(
        re.findall(
            r"\b[A-Z]-\d{3,}\b|\b[A-Z]{2,}(?:/[A-Z]{2,})*\b|[\w.+-]+@[\w.-]+\.\w+",
            source,
            flags=re.IGNORECASE,
        )
    )
    if not identifiers:
        return False
    normalized_chunk = normalize(chunk_text)
    return any(normalize(identifier) in normalized_chunk for identifier in identifiers)


def score_chunk(row: dict[str, str], chunk: dict[str, Any]) -> tuple[float, dict[str, float]]:
    chunk_text = str(chunk.get("text") or "")
    expected_page = leading_int(row.get("source_chunk"))
    actual_page = page_from_chunk(chunk)
    page_bonus = 1.0 if expected_page is not None and actual_page == expected_page else 0.0
    evidence = recall_score(row.get("evidence", ""), chunk_text)
    answer = recall_score(row.get("answer", ""), chunk_text)
    question = recall_score(row.get("question", ""), chunk_text)
    identifier = 1.0 if identifier_hit(row, chunk_text) else 0.0
    total = 0.62 * evidence + 0.16 * answer + 0.04 * question + 0.10 * page_bonus + 0.08 * identifier
    return total, {
        "evidence": evidence,
        "answer": answer,
        "question": question,
        "page": page_bonus,
        "identifier": identifier,
    }


def load_active_chunks() -> list[dict[str, Any]]:
    settings = get_settings()
    snapshot = PostgresMetadataStore(settings.postgres_dsn).load_active_snapshot(settings.kb_id)
    if snapshot is None or not snapshot.chunks:
        raise RuntimeError(f"No active chunks found for kb_id={settings.kb_id!r}.")
    return snapshot.chunks


def match_row(row: dict[str, str], chunks: list[dict[str, Any]]) -> dict[str, str]:
    expected_source = basename_key(row.get("source_document") or row.get("source_path"))
    candidates = [
        chunk for chunk in chunks
        if basename_key(chunk.get("metadata", {}).get("source_name")) == expected_source
        or basename_key(chunk.get("metadata", {}).get("source_path")) == expected_source
    ]
    if not candidates:
        return {
            "expected_chunk_id": "",
            "expected_chunk_match_score": "0.0000",
            "expected_chunk_match_status": "source_not_found",
            "expected_chunk_source_name": "",
            "expected_chunk_page": "",
            "expected_chunk_index": "",
            "expected_chunk_preview": "",
        }

    scored = []
    for chunk in candidates:
        score, parts = score_chunk(row, chunk)
        scored.append((score, parts, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    score, parts, best = scored[0]
    metadata = best.get("metadata", {})
    status = "matched" if score >= 0.35 or parts["identifier"] else "low_confidence"
    preview = re.sub(r"\s+", " ", str(best.get("text") or "")).strip()[:180]
    return {
        "expected_chunk_id": str(best.get("chunk_id") or ""),
        "expected_chunk_match_score": f"{score:.4f}",
        "expected_chunk_match_status": status,
        "expected_chunk_source_name": str(metadata.get("source_name") or ""),
        "expected_chunk_page": str(page_from_chunk(best) or ""),
        "expected_chunk_index": str(metadata.get("chunk_index") if metadata.get("chunk_index") is not None else ""),
        "expected_chunk_preview": preview,
        "expected_chunk_score_detail": (
            f"evidence={parts['evidence']:.3f};"
            f"answer={parts['answer']:.3f};"
            f"question={parts['question']:.3f};"
            f"page={parts['page']:.0f};"
            f"identifier={parts['identifier']:.0f}"
        ),
    }


def fill_csv(input_path: Path, output_path: Path) -> None:
    chunks = load_active_chunks()
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0].keys()) if rows else []

    added = [
        "expected_chunk_id",
        "expected_chunk_match_score",
        "expected_chunk_match_status",
        "expected_chunk_source_name",
        "expected_chunk_page",
        "expected_chunk_index",
        "expected_chunk_score_detail",
        "expected_chunk_preview",
    ]
    for row in rows:
        row.update(match_row(row, chunks))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames + [name for name in added if name not in fieldnames])
        writer.writeheader()
        writer.writerows(rows)

    statuses: dict[str, int] = {}
    for row in rows:
        status = row["expected_chunk_match_status"]
        statuses[status] = statuses.get(status, 0) + 1
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(statuses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill expected_chunk_id in a QA CSV from the active RAG chunk snapshot.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_with_chunk_ids{args.input.suffix}")
    fill_csv(args.input, output)


if __name__ == "__main__":
    main()
