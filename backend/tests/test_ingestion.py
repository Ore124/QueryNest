from pathlib import Path

import fitz
from PIL import Image

from app.documents import RawDocument, discover_files, load_documents, split_documents
from app.providers import HashEmbeddings
from app.index import HybridIndex


def create_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def create_image(path: Path) -> None:
    image = Image.new("RGB", (120, 80), color=(240, 240, 240))
    image.save(path)


def test_ingestion_loads_supported_types_and_keeps_indexes_aligned(tmp_path, fake_milvus_client):
    (tmp_path / "01_制度与流程").mkdir()
    (tmp_path / "01_制度与流程" / "policy.md").write_text("# 请假制度\n\n审批路径 FAQ Q1", encoding="utf-8")
    (tmp_path / "01_制度与流程" / "notes.txt").write_text("# 客户问题\n\n接口 500 排查", encoding="utf-8")
    create_pdf(tmp_path / "01_制度与流程" / "manual.pdf", "PDF 审批规则")
    create_image(tmp_path / "01_制度与流程" / "flow.png")
    create_image(tmp_path / "01_制度与流程" / "flow.jpg")

    files = discover_files(tmp_path)
    assert {path.suffix.lower() for path in files} == {".md", ".txt", ".pdf", ".png", ".jpg"}

    class FakeParsers:
        def parse(self, path, scenario):
            return [
                RawDocument(
                    text=f"# 图片资产: {path.stem}\n\nOCR {path.name}",
                    source_path=str(path),
                    source_name=path.name,
                    file_type=path.suffix.lstrip("."),
                    scenario=scenario,
                    section=path.stem,
                    content_type="image",
                    parser="paddleocr",
                )
            ]

    raw_documents = load_documents(tmp_path, FakeParsers(), include_images=True)
    chunks = split_documents(raw_documents, chunk_size=200, chunk_overlap=20)
    index = HybridIndex(tmp_path / "index", HashEmbeddings(dimensions=64), milvus_client=fake_milvus_client)
    index.build(chunks)

    assert len(chunks) > 0
    assert len(index.chunks) == len(index.tokenized_corpus)
    assert index.ready
    assert index.scenarios() == ["制度与流程"]


def test_table_chunks_repeat_header_and_keep_structured_metadata():
    table = RawDocument(
        text=(
            "### 审批矩阵\n\n"
            "| 金额 | 审批人 |\n"
            "| --- | --- |\n"
            "| 1000 | 经理 |\n"
            "| 5000 | 总监 |\n"
            "| 10000 | 总经理 |"
        ),
        source_path="approval.xlsx",
        source_name="approval.xlsx",
        file_type="xlsx",
        scenario="制度与流程",
        section="审批矩阵",
        content_type="table",
        parser="mineru",
        table_id="table-1",
        table_markdown="完整表格",
        table_json='{"headers":["金额","审批人"]}',
        table_html="<table></table>",
    )

    chunks = split_documents([table], chunk_size=65, chunk_overlap=0)

    assert len(chunks) >= 2
    assert all("| 金额 | 审批人 |" in chunk.text for chunk in chunks)
    assert all(chunk.metadata["table_id"] == "table-1" for chunk in chunks)
    assert all(chunk.metadata["table_json"] == table.table_json for chunk in chunks)


def test_repeated_mineru_blocks_get_unique_chunk_ids():
    document = RawDocument(
        text="重复页眉",
        source_path="manual.pdf",
        source_name="manual.pdf",
        file_type="pdf",
        scenario="制度与流程",
        page=1,
        parser="mineru",
    )

    chunks = split_documents([document, document], chunk_size=200, chunk_overlap=0)

    assert len(chunks) == 2
    assert len({chunk.chunk_id for chunk in chunks}) == 2
