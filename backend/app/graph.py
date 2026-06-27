from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .history import HistoryStore
from .index import HybridIndex
from .providers import get_chat_model
from .schemas import ChatResponse, Source
from .settings import Settings


class RagState(TypedDict, total=False):
    session_id: str
    question: str
    rewritten_question: str
    history: list[dict[str, str]]
    scenario: str | None
    model: str | None
    top_k: int
    sources: list[Source]
    retrieval_debug: dict[str, Any]
    answer: str


class RagService:
    def __init__(self, settings: Settings, index: HybridIndex, history: HistoryStore) -> None:
        self.settings = settings
        self.index = index
        self.history = history
        self.graph = self._build_graph()

    def chat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        scenario: str | None = None,
        model: str | None = None,
        top_k: int | None = None,
    ) -> ChatResponse:
        resolved_session_id = session_id or str(uuid.uuid4())
        history = self.history.load(resolved_session_id)
        state = self.graph.invoke(
            {
                "session_id": resolved_session_id,
                "question": message,
                "history": history,
                "scenario": scenario,
                "model": model,
                "top_k": top_k or self.settings.final_top_k,
            },
            config={"configurable": {"thread_id": resolved_session_id}},
        )
        answer = state["answer"]
        self.history.append(resolved_session_id, "user", message)
        self.history.append(resolved_session_id, "assistant", answer)
        return ChatResponse(
            session_id=resolved_session_id,
            answer=answer,
            sources=state.get("sources", []),
            retrieval_debug=state.get("retrieval_debug", {}),
        )

    def _build_graph(self):
        graph = StateGraph(RagState)
        graph.add_node("rewrite", self._rewrite_question)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _rewrite_question(self, state: RagState) -> RagState:
        question = state["question"]
        history = state.get("history", [])
        if not history:
            return {"rewritten_question": question}
        previous_questions = [
            item["content"]
            for item in reversed(history)
            if item.get("role") == "user" and item.get("content")
        ]
        if not previous_questions:
            return {"rewritten_question": question}
        return {
            "rewritten_question": (
                f"上一轮用户问题：{previous_questions[0]}\n"
                f"当前追问：{question}"
            )
        }

    def _retrieve(self, state: RagState) -> RagState:
        sources, debug = self.index.search(
            state.get("rewritten_question") or state["question"],
            scenario=state.get("scenario"),
            dense_top_k=self.settings.dense_top_k,
            bm25_top_k=self.settings.bm25_top_k,
            final_top_k=state.get("top_k") or self.settings.final_top_k,
            rrf_k=self.settings.rrf_k,
        )
        return {"sources": sources, "retrieval_debug": debug}

    def _generate(self, state: RagState) -> RagState:
        sources = state.get("sources", [])
        context = format_context(sources)
        system = (
            "你是企业知识库 RAG 问答助手。只能基于给定引用回答；如果引用不足以支持结论，"
            "明确说明需要联系归口部门或补充资料。回答要简洁、可执行，并在关键句后标注引用编号。"
            "回答控制在 300 个汉字以内。"
        )
        user = (
            f"用户问题: {state['question']}\n\n"
            f"检索问题: {state.get('rewritten_question') or state['question']}\n\n"
            f"引用资料:\n{context}\n\n"
            "请用中文回答，并列出必要步骤、责任角色、时限或例外情况。"
        )
        llm = get_chat_model(
            self.settings,
            state.get("model"),
            temperature=0.2,
            thinking=False,
            max_tokens=700,
        )
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return {"answer": str(response.content).strip()}


def format_context(sources: list[Source]) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        page = f", 页码: {source.page}" if source.page is not None else ""
        section = f", 章节: {source.section}" if source.section else ""
        blocks.append(
            f"[{index}] 来源: {source.source_name}{page}{section}, 场景: {source.scenario}\n{source.text}"
        )
    return "\n\n".join(blocks) if blocks else "无可用引用。"
