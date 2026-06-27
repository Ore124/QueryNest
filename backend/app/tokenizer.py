from __future__ import annotations

import re

import jieba

TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+|[\u4e00-\u9fff]+", re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.findall(text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            tokens.extend(part for part in jieba.lcut(match) if part.strip())
        else:
            tokens.append(match.lower())
    return tokens
