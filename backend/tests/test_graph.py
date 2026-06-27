from app.graph import RagService


def test_rewrite_keeps_first_turn_question():
    service = object.__new__(RagService)

    result = service._rewrite_question({"question": "接口 500 怎么排查？", "history": []})

    assert result["rewritten_question"] == "接口 500 怎么排查？"


def test_rewrite_combines_latest_user_question_without_model_call():
    service = object.__new__(RagService)

    result = service._rewrite_question(
        {
            "question": "第一步具体做什么？",
            "history": [
                {"role": "user", "content": "旧问题"},
                {"role": "assistant", "content": "旧回答"},
                {"role": "user", "content": "接口 500 怎么排查？"},
                {"role": "assistant", "content": "先看错误日志。"},
            ],
        }
    )

    assert result["rewritten_question"] == (
        "上一轮用户问题：接口 500 怎么排查？\n"
        "当前追问：第一步具体做什么？"
    )
