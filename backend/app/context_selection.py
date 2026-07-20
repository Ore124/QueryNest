from __future__ import annotations

from typing import Any

from .schemas import Source


def select_context_sources(
    sources: list[Source],
    *,
    max_sources: int,
    min_sources: int,
    score_ratio: float,
    max_chars: int,
    score_filter_enabled: bool,
) -> tuple[list[Source], dict[str, Any]]:
    candidate_count = len(sources)
    capped_sources = sources[: max(max_sources, 0)]
    required_count = min(max(min_sources, 0), len(capped_sources))
    score_filter_applied = score_filter_enabled and bool(capped_sources) and all(
        source.rerank_score is not None for source in capped_sources
    )
    score_cutoff = (
        float(capped_sources[0].rerank_score) * score_ratio
        if score_filter_applied
        else None
    )

    selected: list[Source] = []
    used_chars = 0
    for source in capped_sources:
        if len(selected) >= required_count:
            if score_cutoff is not None and float(source.rerank_score) < score_cutoff:
                break
            if used_chars + len(source.text) > max_chars:
                break
        selected.append(source)
        used_chars += len(source.text)

    strategy = "rerank_ratio_and_char_budget" if score_filter_applied else "char_budget"
    telemetry = {
        "candidate_count": candidate_count,
        "selected_count": len(selected),
        "dropped_count": candidate_count - len(selected),
        "strategy": strategy,
        "score_cutoff": score_cutoff,
        "score_filter_enabled": score_filter_enabled,
        "score_filter_applied": score_filter_applied,
        "used_chars": used_chars,
        "configured_max_sources": max_sources,
        "configured_min_sources": min_sources,
        "configured_score_ratio": score_ratio,
        "configured_max_chars": max_chars,
    }
    return selected, telemetry
