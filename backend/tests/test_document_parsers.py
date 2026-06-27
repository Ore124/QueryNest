import json

from app.document_parsers import paddle_result_to_text, parse_mineru_content_list


def test_parse_mineru_content_list_preserves_table_markdown_json_and_html():
    table_html = (
        "<table><tr><th>金额</th><th>审批人</th></tr>"
        "<tr><td>1000</td><td>经理</td></tr></table>"
    )
    blocks = parse_mineru_content_list(
        [
            {"type": "text", "text": "审批说明", "page_idx": 0},
            {
                "type": "table",
                "table_caption": ["审批矩阵"],
                "table_body": table_html,
                "page_idx": 1,
            },
        ],
        source_stem="采购制度",
    )

    assert blocks[0].content_type == "text"
    table = blocks[1]
    assert table.content_type == "table"
    assert table.page == 2
    assert table.section == "审批矩阵"
    assert "| 金额 | 审批人 |" in table.text
    assert table.table_html == table_html
    payload = json.loads(table.table_json or "{}")
    assert payload["headers"] == ["金额", "审批人"]
    assert payload["rows"] == [["1000", "经理"]]
    assert payload["cells"][0][0]["tag"] == "th"


def test_paddle_result_to_text_accepts_paddleocr_v3_json():
    result = {"rec_texts": ["服务器", "部署"], "rec_scores": [0.99, 0.95]}

    assert paddle_result_to_text(result) == "服务器\n部署"


def test_parse_mineru_table_uses_merged_first_row_as_title():
    blocks = parse_mineru_content_list(
        [
            {
                "type": "table",
                "table_body": (
                    "<table><tr><th colspan='2'>采购审批矩阵</th></tr>"
                    "<tr><td>金额范围</td><td>审批人</td></tr>"
                    "<tr><td>0-1000元</td><td>部门经理</td></tr></table>"
                ),
            }
        ],
        source_stem="approval_matrix",
    )

    table = blocks[0]
    assert table.text.startswith("### 采购审批矩阵")
    assert "| 金额范围 | 审批人 |" in table.text
    assert "| 0-1000元 | 部门经理 |" in table.text
