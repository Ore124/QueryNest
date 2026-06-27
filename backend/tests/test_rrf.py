from app.index import SearchHit, reciprocal_rank_fusion


def hit(chunk_id: str) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, text=chunk_id, metadata={"chunk_id": chunk_id})


def test_rrf_fuses_deduplicates_and_keeps_rank_details():
    fused = reciprocal_rank_fusion(
        dense_hits=[hit("a"), hit("b"), hit("c")],
        bm25_hits=[hit("b"), hit("d"), hit("a")],
        rrf_k=60,
    )

    assert [item.chunk_id for item in fused] == ["b", "a", "d", "c"]
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert round(fused[0].rrf_score, 6) == round(1 / 62 + 1 / 61, 6)
