from app.tokenizer import tokenize_for_bm25


def test_bm25_tokenizer_keeps_chinese_numbers_and_api_terms():
    tokens = tokenize_for_bm25("接口 500、FAQ Q12 和 traceId abc-123 应该如何排查？")

    assert "500" in tokens
    assert "faq" in tokens
    assert "q12" in tokens
    assert "traceid" in tokens
    assert "abc-123" in tokens
    assert any(token in tokens for token in ["接口", "排查"])
