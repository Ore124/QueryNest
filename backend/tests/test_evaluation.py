import json

from app.evaluation import (
    document_matches,
    first_chunk_rank,
    first_matching_chunk_rank,
    load_answer_records,
    parse_expected_chunk_ids,
    parse_question_set,
    percentile_nearest_rank,
    select_questions,
    serialize_ragas_result,
    summarize_retrieval_records,
)


def test_select_questions_uses_reproducible_sample():
    questions = [{"id": f"TQ-{index:03d}"} for index in range(1, 11)]

    first = select_questions(questions, limit=4, seed=42)
    second = select_questions(questions, limit=4, seed=42)

    assert first == second
    assert [item["id"] for item in first] == sorted(item["id"] for item in first)
    assert len(first) == 4


def test_document_matching_ignores_extension_and_keeps_chinese_names():
    assert document_matches("员工请假管理制度", "员工请假管理制度.pdf")
    assert document_matches("API接口说明文档", "API接口说明文档.md")
    assert not document_matches("员工请假管理制度", "差旅报销管理制度.md")


def test_first_chunk_rank_requires_exact_chunk_id():
    assert first_chunk_rank("chunk-b", ["chunk-a", "chunk-b"]) == 2
    assert first_chunk_rank("chunk-b", ["chunk-a", "chunk-c"]) is None


def test_multiple_expected_chunks_match_any_exact_chunk_id():
    expected = parse_expected_chunk_ids({"expected_chunk_ids": "chunk-b; chunk-c"})

    assert expected == {"chunk-b", "chunk-c"}
    assert first_matching_chunk_rank(expected, ["chunk-a", "chunk-c"]) == 2


def test_parse_question_csv_reads_expected_chunk_id(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "id,document,keyword,question,expected,expected_chunk_id\n"
        "CQ-001,制度.pdf,审批,谁审批？,由经理审批,chunk-123\n",
        encoding="utf-8",
    )

    questions = parse_question_set(path)

    assert questions[0]["expected_chunk_id"] == "chunk-123"


def test_retrieval_summary_reports_hit_mrr_and_p99():
    records = [
        {
            "first_relevant_rank": 1,
            "hit@5": True,
            "precision@5": 0.2,
            "recall@5": 1.0,
            "reciprocal_rank": 1.0,
            "latency_ms": 10.0,
        },
        {
            "first_relevant_rank": 5,
            "hit@5": True,
            "precision@5": 0.2,
            "recall@5": 1.0,
            "reciprocal_rank": 0.2,
            "latency_ms": 20.0,
        },
        {
            "first_relevant_rank": None,
            "hit@5": False,
            "precision@5": 0.0,
            "recall@5": 0.0,
            "reciprocal_rank": 0.0,
            "latency_ms": 30.0,
        },
    ]

    summary = summarize_retrieval_records(records, retrieval_top_k=20, hit_k=5)

    assert summary["hit@5"] == 0.6667
    assert summary["precision@5"] == 0.1333
    assert summary["recall@5"] == 0.6667
    assert summary["MRR"] == 0.4
    assert summary["miss_count"] == 1
    assert summary["P99_ms"] == 30.0
    assert percentile_nearest_rank([1, 5, 21], 99) == 21


def test_serialize_ragas_result_aggregates_scores_and_reports_valid_samples():
    class Result:
        scores = [
            {"faithfulness": 1.0, "answer_relevancy": 0.8},
            {"faithfulness": float("nan"), "answer_relevancy": 0.6},
        ]

    summary = serialize_ragas_result(Result())

    assert summary["sample_count"] == 2
    assert summary["metrics"]["faithfulness"] == 1.0
    assert summary["metrics"]["answer_relevancy"] == 0.7
    assert summary["valid_samples"]["faithfulness"] == 1
    assert summary["metric_coverage"]["faithfulness"] == 0.5
    assert summary["complete"] is False


def test_load_answer_records_supports_checkpoint_resume(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"id": "TQ-001", "response": "已生成"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    records = load_answer_records(path)

    assert records["TQ-001"]["response"] == "已生成"
